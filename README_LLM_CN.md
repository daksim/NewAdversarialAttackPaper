# Latest Adversarial Attack Papers
**update at 2023-12-05 21:14:51**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Tree of Attacks: Jailbreaking Black-Box LLMs Automatically**

攻击之树：自动越狱黑盒LLMS cs.LG

An implementation of the presented method is available at  https://github.com/RICommunity/TAP

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.02119v1) [paper-pdf](http://arxiv.org/pdf/2312.02119v1)

**Authors**: Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, Amin Karbasi

**Abstract**: While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an LLM to iteratively refine candidate (attack) prompts using tree-of-thoughts reasoning until one of the generated prompts jailbreaks the target. Crucially, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks. Using tree-of-thought reasoning allows TAP to navigate a large search space of prompts and pruning reduces the total number of queries sent to the target. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4 and GPT4-Turbo) for more than 80% of the prompts using only a small number of queries. This significantly improves upon the previous state-of-the-art black-box method for generating jailbreaks.

摘要: 虽然大型语言模型(LLM)显示了多功能，但它们继续产生有害、有偏见和有毒的内容，人类设计的越狱事件的流行就证明了这一点。在这项工作中，我们提出了带修剪的攻击树(TAP)，这是一种自动生成越狱的方法，只需要通过黑盒访问目标LLM。TAP利用LLM使用思想树推理反复优化候选(攻击)提示，直到其中一个生成的提示越狱目标。至关重要的是，在向目标发送提示之前，TAP会对它们进行评估，并删除那些不太可能导致越狱的提示。使用思维树推理允许TAP导航大的提示搜索空间，并进行修剪以减少发送到目标的查询总数。在经验评估中，我们观察到TAP仅使用少量查询就为80%以上的提示生成了越狱最先进的LLM(包括GPT4和GPT4-Turbo)提示。这大大改进了以前用于生成越狱的最先进的黑匣子方法。



## **2. A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly**

大型语言模型(LLM)安全与隐私：好、坏、丑 cs.CR

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.02003v1) [paper-pdf](http://arxiv.org/pdf/2312.02003v1)

**Authors**: Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Eric Sun, Yue Zhang

**Abstract**: Large Language Models (LLMs), such as GPT-3 and BERT, have revolutionized natural language understanding and generation. They possess deep language comprehension, human-like text generation capabilities, contextual awareness, and robust problem-solving skills, making them invaluable in various domains (e.g., search engines, customer support, translation). In the meantime, LLMs have also gained traction in the security community, revealing security vulnerabilities and showcasing their potential in security-related tasks. This paper explores the intersection of LLMs with security and privacy. Specifically, we investigate how LLMs positively impact security and privacy, potential risks and threats associated with their use, and inherent vulnerabilities within LLMs. Through a comprehensive literature review, the paper categorizes findings into "The Good" (beneficial LLM applications), "The Bad" (offensive applications), and "The Ugly" (vulnerabilities and their defenses). We have some interesting findings. For example, LLMs have proven to enhance code and data security, outperforming traditional methods. However, they can also be harnessed for various attacks (particularly user-level attacks) due to their human-like reasoning abilities. We have identified areas that require further research efforts. For example, research on model and parameter extraction attacks is limited and often theoretical, hindered by LLM parameter scale and confidentiality. Safe instruction tuning, a recent development, requires more exploration. We hope that our work can shed light on the LLMs' potential to both bolster and jeopardize cybersecurity.

摘要: 大型语言模型(LLM)，如GPT-3和BERT，已经彻底改变了自然语言的理解和生成。它们具有深刻的语言理解能力、类似人类的文本生成能力、上下文意识和强大的解决问题的技能，使它们在各个领域(例如，搜索引擎、客户支持、翻译)具有无价的价值。与此同时，LLMS也在安全界获得了吸引力，揭示了安全漏洞，并在与安全相关的任务中展示了它们的潜力。本文探讨了LLMS与安全和隐私的交集。具体地说，我们调查了LLM如何对安全和隐私产生积极影响，与其使用相关的潜在风险和威胁，以及LLM中的固有漏洞。通过全面的文献回顾，本文将研究结果分为“好的”(有益的LLM应用程序)、“坏的”(攻击性应用程序)和“丑陋的”(漏洞及其防御)。我们有一些有趣的发现。例如，LLM已被证明可以增强代码和数据的安全性，表现优于传统方法。然而，由于它们类似人类的推理能力，它们也可以被利用来进行各种攻击(特别是用户级的攻击)。我们已经确定了需要进一步研究的领域。例如，对模型和参数提取攻击的研究是有限的，而且往往是理论上的，受到LLM参数规模和保密性的阻碍。安全的指令调优是一个新的发展，需要更多的探索。我们希望我们的工作能够揭示小岛屿发展中国家加强和危害网络安全的潜力。



## **3. Intrusion Detection System with Machine Learning and Multiple Datasets**

基于机器学习和多数据集的入侵检测系统 cs.CR

12 pages, 2 figures, 2 tables

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01941v1) [paper-pdf](http://arxiv.org/pdf/2312.01941v1)

**Authors**: Haiyan Xuan, Mohith Manohar

**Abstract**: As Artificial Intelligence (AI) technologies continue to gain traction in the modern-day world, they ultimately pose an immediate threat to current cybersecurity systems via exploitative methods. Prompt engineering is a relatively new field that explores various prompt designs that can hijack large language models (LLMs). If used by an unethical attacker, it can enable an AI system to offer malicious insights and code to them. In this paper, an enhanced intrusion detection system (IDS) that utilizes machine learning (ML) and hyperparameter tuning is explored, which can improve a model's performance in terms of accuracy and efficacy. Ultimately, this improved system can be used to combat the attacks made by unethical hackers. A standard IDS is solely configured with pre-configured rules and patterns; however, with the utilization of machine learning, implicit and different patterns can be generated through the models' hyperparameter settings and parameters. In addition, the IDS will be equipped with multiple datasets so that the accuracy of the models improves. We evaluate the performance of multiple ML models and their respective hyperparameter settings through various metrics to compare their results to other models and past research work. The results of the proposed multi-dataset integration method yielded an accuracy score of 99.9% when equipped with the XGBoost and random forest classifiers and RandomizedSearchCV hyperparameter technique.

摘要: 随着人工智能(AI)技术在现代世界继续获得牵引力，它们最终通过剥削性方法对当前的网络安全系统构成直接威胁。提示工程是一个相对较新的领域，它探索可以劫持大型语言模型(LLM)的各种提示设计。如果被不道德的攻击者使用，它可以使人工智能系统向他们提供恶意的见解和代码。本文探讨了一种利用机器学习和超参数调整的增强型入侵检测系统，它可以在准确性和有效性方面提高模型的性能。最终，这个改进的系统可以用来打击不道德的黑客进行的攻击。一个标准的入侵检测系统只配置了预先配置的规则和模式，但是，利用机器学习，可以通过模型的超参数设置和参数来生成隐式和不同的模式。此外，入侵检测系统将配备多个数据集，以提高模型的精度。我们通过不同的度量来评估多个最大似然模型及其各自的超参数设置的性能，并将它们的结果与其他模型和过去的研究工作进行比较。在使用XGBoost和随机森林分类器以及RandomizedSearchCV超参数技术的情况下，所提出的多数据集集成方法的准确率为99.9%。



## **4. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

InstructTA：针对大型视觉语言模型的指令调整定向攻击 cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01886v1) [paper-pdf](http://arxiv.org/pdf/2312.01886v1)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical gray-box attack scenario that the adversary can only access the visual encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed InstructTA) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same visual encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from an LLM. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability.

摘要: 大型视觉语言模型在图像理解和响应生成方面表现出了令人难以置信的能力。然而，这种丰富的视觉交互也使LVLM容易受到对抗性例子的攻击。本文提出了一种新颖实用的灰盒攻击方案，即攻击者只能访问受害者LVLM的可视编码器，而不知道其提示(通常是服务提供商的专有提示，而不是公开可用的)及其底层的大型语言模型(LLM)。这一实际设置对目标对抗性攻击的跨提示和跨模型可转移性提出了挑战，其目的是混淆LVLM以输出与攻击者选择的目标文本在语义上相似的响应。为此，我们提出了一种指令调谐的定向攻击(InstructTA)，对具有高可转移性的LVLMS进行定向对抗性攻击。首先，我们利用一个公开的文本到图像的生成模型将目标响应“反转”成目标图像，并使用GPT-4从目标响应中推断出合理的指令符号。然后，我们形成一个局部代理模型(与受害者LVLM共享相同的视觉编码器)来提取对抗性图像示例和目标图像的指令感知特征，并最小化这两个特征之间的距离以优化对抗性示例。为了进一步提高可转移性，我们用转译自LLM的指令扩充了指令$\boldSymbol{p}^\Prime$。大量实验证明了该方法在目标攻击性能和可转移性方面的优越性。



## **5. Unleashing Cheapfakes through Trojan Plugins of Large Language Models**

通过大型语言模型特洛伊木马插件释放Cheapfake cs.CR

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2312.00374v1) [paper-pdf](http://arxiv.org/pdf/2312.00374v1)

**Authors**: Tian Dong, Guoxing Chen, Shaofeng Li, Minhui Xue, Rayne Holland, Yan Meng, Zhen Liu, Haojin Zhu

**Abstract**: Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers, an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses LLM-enhanced paraphrasing to polish benchmark poisoned datasets. In contrast, in the absence of a dataset, FUSION leverages an over-poisoning procedure to transform a benign adaptor. Our experiments validate that our attacks provide higher attack effectiveness than the baseline and, for the purpose of attracting downloads, preserves or improves the adapter's utility. Finally, we provide two case studies to demonstrate that the Trojan adapter can lead a LLM-powered autonomous agent to execute unintended scripts or send phishing emails. Our novel attacks represent the first study of supply chain threats for LLMs through the lens of Trojan plugins.

摘要: 开源的大型语言模型(LLM)最近越来越受欢迎，因为它们的性能可以与专有的LLM相媲美。为了高效地完成领域专门化任务，可以使用低级别适配器对开源LLM进行提炼，而无需使用昂贵的加速器。然而，是否可以利用低阶适配器来控制LLM仍然是未知的。为了弥补这一漏洞，我们演示了受感染的适配器可以在特定触发下诱导LLM输出由对手定义的内容，甚至恶意使用工具。为了训练木马适配器，我们提出了两种新的攻击方法，磨光攻击和融合攻击，它们比以前的方法有所改进。波兰德使用LLM增强的释义来抛光基准有毒数据集。相比之下，在没有数据集的情况下，Fusion利用过度中毒的程序来转换良性适配器。我们的实验验证了我们的攻击提供了比基线更高的攻击效率，并且为了吸引下载的目的，保留或提高了适配器的实用性。最后，我们提供了两个案例研究来演示特洛伊木马适配器可以导致LLM驱动的自主代理执行意外脚本或发送钓鱼电子邮件。我们的新型攻击首次通过特洛伊木马插件的镜头研究了LLM的供应链威胁。



## **6. Mark My Words: Analyzing and Evaluating Language Model Watermarks**

记住我的话：分析和评估语言模型水印 cs.CR

19 pages, 14 figures

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2312.00273v1) [paper-pdf](http://arxiv.org/pdf/2312.00273v1)

**Authors**: Julien Piet, Chawin Sitawarin, Vivian Fang, Norman Mu, David Wagner

**Abstract**: The capabilities of large language models have grown significantly in recent years and so too have concerns about their misuse. In this context, the ability to distinguish machine-generated text from human-authored content becomes important. Prior works have proposed numerous schemes to watermark text, which would benefit from a systematic evaluation framework. This work focuses on text watermarking techniques - as opposed to image watermarks - and proposes a comprehensive benchmark for them under different tasks as well as practical attacks. We focus on three main metrics: quality, size (e.g. the number of tokens needed to detect a watermark), and tamper-resistance. Current watermarking techniques are good enough to be deployed: Kirchenbauer et al. can watermark Llama2-7B-chat with no perceivable loss in quality in under 100 tokens, and with good tamper-resistance to simple attacks, regardless of temperature. We argue that watermark indistinguishability is too strong a requirement: schemes that slightly modify logit distributions outperform their indistinguishable counterparts with no noticeable loss in generation quality. We publicly release our benchmark.

摘要: 近年来，大型语言模型的能力显著增长，人们对它们的滥用也感到担忧。在这种情况下，区分机器生成的文本和人类创作的内容的能力变得很重要。以前的工作已经提出了许多方案来对文本进行水印，这将受益于一个系统的评估框架。这项工作聚焦于文本水印技术，而不是图像水印技术，并针对不同的任务和实际攻击提出了一个全面的基准。我们主要关注三个指标：质量、大小(例如，检测水印所需的令牌数量)和防篡改。目前的水印技术足够好，可以部署：Kirchenbauer等人。可以水印Llama2-7B-Chat，在100个令牌以下没有可察觉的质量损失，并且具有良好的抗简单攻击的篡改能力，无论温度如何。我们认为，水印的不可区分性是一个太强的要求：稍微修改Logit分布的方案在生成质量上没有明显损失的情况下，性能优于它们的不可区分的对应方案。我们公开发布我们的基准。



## **7. Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition**

忽略这个标题和HackAPrompt：通过全球规模的即时黑客竞赛揭露LLMS的系统性漏洞 cs.CR

34 pages, 8 figures Codebase:  https://github.com/PromptLabs/hackaprompt Dataset:  https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset/blob/main/README.md  Playground: https://huggingface.co/spaces/hackaprompt/playground

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.16119v2) [paper-pdf](http://arxiv.org/pdf/2311.16119v2)

**Authors**: Sander Schulhoff, Jeremy Pinto, Anaum Khan, Louis-François Bouchard, Chenglei Si, Svetlina Anati, Valen Tagliabue, Anson Liu Kost, Christopher Carnahan, Jordan Boyd-Graber

**Abstract**: Large Language Models (LLMs) are deployed in interactive contexts with direct user engagement, such as chatbots and writing assistants. These deployments are vulnerable to prompt injection and jailbreaking (collectively, prompt hacking), in which models are manipulated to ignore their original instructions and follow potentially malicious ones. Although widely acknowledged as a significant security threat, there is a dearth of large-scale resources and quantitative studies on prompt hacking. To address this lacuna, we launch a global prompt hacking competition, which allows for free-form human input attacks. We elicit 600K+ adversarial prompts against three state-of-the-art LLMs. We describe the dataset, which empirically verifies that current LLMs can indeed be manipulated via prompt hacking. We also present a comprehensive taxonomical ontology of the types of adversarial prompts.

摘要: 大型语言模型(LLM)部署在具有直接用户参与的交互上下文中，例如聊天机器人和写作助手。这些部署容易受到即时注入和越狱(统称为即时黑客)的攻击，在这些情况下，模型被操纵以忽略其原始指令并遵循潜在的恶意指令。尽管被广泛认为是一个重大的安全威胁，但缺乏关于即时黑客攻击的大规模资源和量化研究。为了弥补这一漏洞，我们发起了一场全球即时黑客竞赛，允许自由形式的人工输入攻击。我们在三个最先进的LLM上获得了600K+的对抗性提示。我们描述了数据集，这从经验上验证了当前的LLM确实可以通过即时黑客来操纵。我们还提出了对抗性提示类型的全面分类本体。



## **8. Devising and Detecting Phishing: Large Language Models vs. Smaller Human Models**

设计和检测网络钓鱼：大型语言模型与较小的人类模型 cs.CR

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2308.12287v2) [paper-pdf](http://arxiv.org/pdf/2308.12287v2)

**Authors**: Fredrik Heiding, Bruce Schneier, Arun Vishwanath, Jeremy Bernstein, Peter S. Park

**Abstract**: AI programs, built using large language models, make it possible to automatically create phishing emails based on a few data points about a user. They stand in contrast to traditional phishing emails that hackers manually design using general rules gleaned from experience. The V-Triad is an advanced set of rules for manually designing phishing emails to exploit our cognitive heuristics and biases. In this study, we compare the performance of phishing emails created automatically by GPT-4 and manually using the V-Triad. We also combine GPT-4 with the V-Triad to assess their combined potential. A fourth group, exposed to generic phishing emails, was our control group. We utilized a factorial approach, sending emails to 112 randomly selected participants recruited for the study. The control group emails received a click-through rate between 19-28%, the GPT-generated emails 30-44%, emails generated by the V-Triad 69-79%, and emails generated by GPT and the V-Triad 43-81%. Each participant was asked to explain why they pressed or did not press a link in the email. These answers often contradict each other, highlighting the need for personalized content. The cues that make one person avoid phishing emails make another person fall for them. Next, we used four popular large language models (GPT, Claude, PaLM, and LLaMA) to detect the intention of phishing emails and compare the results to human detection. The language models demonstrated a strong ability to detect malicious intent, even in non-obvious phishing emails. They sometimes surpassed human detection, although often being slightly less accurate than humans. Finally, we make an analysis of the economic aspects of AI-enabled phishing attacks, showing how large language models can increase the incentives of phishing and spear phishing by reducing their costs.

摘要: 使用大型语言模型构建的人工智能程序可以根据用户的几个数据点自动创建钓鱼电子邮件。它们与黑客使用从经验中收集的一般规则手动设计的传统钓鱼电子邮件形成了鲜明对比。V-Triad是一套高级规则，用于手动设计钓鱼电子邮件，以利用我们的认知启发式和偏见。在这项研究中，我们比较了GPT-4自动创建的钓鱼电子邮件和使用V-Triad手动创建的钓鱼电子邮件的性能。我们还将GPT-4与V-Triad相结合，以评估它们的组合潜力。接触普通钓鱼邮件的第四组是我们的控制组。我们使用了析因分析的方法，向112名随机挑选的参与者发送电子邮件，这些参与者被招募参加这项研究。对照组电子邮件的点击率为19%-28%，GPT生成的电子邮件的点击率为30%-44%，V-Triad生成的电子邮件的点击率为69%-79%，GPT和V-Triad生成的电子邮件的点击率为43%-81%。每个参与者都被要求解释为什么他们按下或没有按下电子邮件中的链接。这些答案往往相互矛盾，突显了对个性化内容的需求。让一个人避免钓鱼电子邮件的暗示会让另一个人爱上他们。接下来，我们使用四个流行的大型语言模型(GPT、Claude、Palm和Llama)来检测钓鱼电子邮件的意图，并将结果与人类检测进行比较。语言模型显示出强大的检测恶意意图的能力，即使是在不明显的网络钓鱼电子邮件中也是如此。它们有时会超过人类的检测，尽管它们的准确度往往略低于人类。最后，我们对人工智能钓鱼攻击的经济方面进行了分析，展示了大型语言模型如何通过降低钓鱼和鱼叉式钓鱼的成本来增加它们的诱因。



## **9. Locally Differentially Private Document Generation Using Zero Shot Prompting**

基于零镜头提示的局部差异私有文档生成方法 cs.CL

Accepted at EMNLP 2023 (Findings)

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2310.16111v2) [paper-pdf](http://arxiv.org/pdf/2310.16111v2)

**Authors**: Saiteja Utpala, Sara Hooker, Pin Yu Chen

**Abstract**: Numerous studies have highlighted the privacy risks associated with pretrained large language models. In contrast, our research offers a unique perspective by demonstrating that pretrained large language models can effectively contribute to privacy preservation. We propose a locally differentially private mechanism called DP-Prompt, which leverages the power of pretrained large language models and zero-shot prompting to counter author de-anonymization attacks while minimizing the impact on downstream utility. When DP-Prompt is used with a powerful language model like ChatGPT (gpt-3.5), we observe a notable reduction in the success rate of de-anonymization attacks, showing that it surpasses existing approaches by a considerable margin despite its simpler design. For instance, in the case of the IMDB dataset, DP-Prompt (with ChatGPT) perfectly recovers the clean sentiment F1 score while achieving a 46\% reduction in author identification F1 score against static attackers and a 26\% reduction against adaptive attackers. We conduct extensive experiments across six open-source large language models, ranging up to 7 billion parameters, to analyze various effects of the privacy-utility tradeoff.

摘要: 许多研究都强调了与预先训练的大型语言模型相关的隐私风险。相比之下，我们的研究提供了一个独特的视角，证明了预先训练的大型语言模型可以有效地有助于隐私保护。我们提出了一种称为DP-Prompt的局部差异私有机制，该机制利用预先训练的大型语言模型和零镜头提示的能力来对抗作者去匿名化攻击，同时最小化对下游效用的影响。当DP-Prompt与ChatGPT(GPT-3.5)等强大的语言模型一起使用时，我们观察到去匿名化攻击的成功率显著下降，表明尽管它的设计更简单，但它在相当大程度上超过了现有的方法。例如，在IMDB数据集的情况下，DP-Prompt(使用ChatGPT)完美地恢复了干净的情感F1分数，同时在针对静态攻击者的作者识别F1分数和针对自适应攻击者的F1分数分别减少了46%和26%。我们在六个开放源码的大型语言模型上进行了广泛的实验，范围多达70亿个参数，以分析隐私-效用权衡的各种影响。



## **10. Towards Safer Generative Language Models: A Survey on Safety Risks, Evaluations, and Improvements**

走向更安全的生成性语言模型：安全风险、评估和改进的综述 cs.AI

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2302.09270v3) [paper-pdf](http://arxiv.org/pdf/2302.09270v3)

**Authors**: Jiawen Deng, Jiale Cheng, Hao Sun, Zhexin Zhang, Minlie Huang

**Abstract**: As generative large model capabilities advance, safety concerns become more pronounced in their outputs. To ensure the sustainable growth of the AI ecosystem, it's imperative to undertake a holistic evaluation and refinement of associated safety risks. This survey presents a framework for safety research pertaining to large models, delineating the landscape of safety risks as well as safety evaluation and improvement methods. We begin by introducing safety issues of wide concern, then delve into safety evaluation methods for large models, encompassing preference-based testing, adversarial attack approaches, issues detection, and other advanced evaluation methods. Additionally, we explore the strategies for enhancing large model safety from training to deployment, highlighting cutting-edge safety approaches for each stage in building large models. Finally, we discuss the core challenges in advancing towards more responsible AI, including the interpretability of safety mechanisms, ongoing safety issues, and robustness against malicious attacks. Through this survey, we aim to provide clear technical guidance for safety researchers and encourage further study on the safety of large models.

摘要: 随着生成式大型模型能力的发展，安全问题在其输出中变得更加突出。为了确保人工智能生态系统的可持续发展，必须对相关的安全风险进行全面评估和改进。该调查提出了一个与大型模型相关的安全研究框架，描绘了安全风险的景观以及安全评估和改进方法。我们首先介绍广泛关注的安全问题，然后深入研究大型模型的安全评估方法，包括基于偏好的测试，对抗性攻击方法，问题检测和其他高级评估方法。此外，我们还探讨了从培训到部署的增强大型模型安全性的策略，强调了构建大型模型的每个阶段的尖端安全方法。最后，我们讨论了迈向更负责任的人工智能的核心挑战，包括安全机制的可解释性，持续的安全问题以及对恶意攻击的鲁棒性。通过此次调查，我们旨在为安全研究人员提供明确的技术指导，并鼓励对大型模型的安全性进行进一步研究。



## **11. Leveraging a Randomized Key Matrix to Enhance the Security of Symmetric Substitution Ciphers**

利用随机化密钥矩阵提高对称替换密码的安全性 cs.CR

In Proceedings of the 10th IEEE Asia-Pacific Conference on Computer  Science and Data Engineering 2023 (CSDE)

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.18085v1) [paper-pdf](http://arxiv.org/pdf/2311.18085v1)

**Authors**: Shubham Gandhi, Om Khare, Mihika Dravid, Mihika Sanghvi, Sunil Mane, Aadesh Gajaralwar, Saloni Gandhi

**Abstract**: An innovative strategy to enhance the security of symmetric substitution ciphers is presented, through the implementation of a randomized key matrix suitable for various file formats, including but not limited to binary and text files. Despite their historical relevance, symmetric substitution ciphers have been limited by vulnerabilities to cryptanalytic methods like frequency analysis and known plaintext attacks. The aim of our research is to mitigate these vulnerabilities by employing a polyalphabetic substitution strategy that incorporates a distinct randomized key matrix. This matrix plays a pivotal role in generating a unique random key, comprising characters, encompassing both uppercase and lowercase letters, numeric, and special characters, to derive the corresponding ciphertext. The effectiveness of the proposed methodology in enhancing the security of conventional substitution methods for file encryption and decryption is supported by comprehensive testing and analysis, which encompass computational speed, frequency analysis, keyspace examination, Kasiski test, entropy analysis, and the utilization of a large language model.

摘要: 提出了一种增强对称替换密码安全性的创新策略，通过实现适用于各种文件格式的随机化密钥矩阵，包括但不限于二进制和文本文件。尽管对称替换密码具有历史相关性，但它一直受到频率分析和已知明文攻击等密码分析方法的漏洞的限制。我们研究的目的是通过采用多字母替换策略来缓解这些漏洞，该策略结合了一个独特的随机密钥矩阵。该矩阵在生成唯一的随机密钥方面起着关键作用，该密钥包括包含大小写字母、数字和特殊字符的字符，以得出相应的密文。通过全面的测试和分析，包括计算速度、频率分析、密钥空间检查、Kasiski测试、熵分析和大型语言模型的使用，支持了所提出的方法在增强传统文件加密和解密替代方法的安全性方面的有效性。



## **12. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

SmoothLLM：保护大型语言模型免受越狱攻击 cs.LG

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2310.03684v3) [paper-pdf](http://arxiv.org/pdf/2310.03684v3)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM. Our code is publicly available at the following link: https://github.com/arobey1/smooth-llm.

摘要: 尽管努力将大型语言模型（LLM）与人类价值观相结合，但广泛使用的LLM（如GPT，Llama，Claude和PaLM）容易受到越狱攻击，其中对手欺骗目标LLM生成令人反感的内容。为了解决这个漏洞，我们提出了SmoothLLM，这是第一个旨在减轻LLM越狱攻击的算法。基于我们发现对抗性生成的提示对字符级别的变化很脆弱，我们的防御首先随机干扰给定输入提示的多个副本，然后聚合相应的预测来检测对抗性输入。SmoothLLM将许多流行LLM的攻击成功率降低到一个百分点以下，避免了不必要的保守性，并承认对攻击缓解的可证明保证。此外，我们的防御使用的查询比现有的攻击少得多，并且与任何LLM兼容。我们的代码可通过以下链接公开获取：https://github.com/arobey1/smooth-llm。



## **13. Query-Relevant Images Jailbreak Large Multi-Modal Models**

与查询相关的图像越狱大型多模式模型 cs.CV

Technique report

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17600v1) [paper-pdf](http://arxiv.org/pdf/2311.17600v1)

**Authors**: Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: Warning: This paper contains examples of harmful language and images, and reader discretion is recommended. The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Large Multi-Modal Models (LMMs) remains understudied. In our study, we present a novel visual prompt attack that exploits query-relevant images to jailbreak the open-source LMMs. Our method creates a composite image from one image generated by diffusion models and another that displays the text as typography, based on keywords extracted from a malicious query. We show LLMs can be easily attacked by our approach, even if the employed Large Language Models are safely aligned. To evaluate the extent of this vulnerability in open-source LMMs, we have compiled a substantial dataset encompassing 13 scenarios with a total of 5,040 text-image pairs, using our presented attack technique. Our evaluation of 12 cutting-edge LMMs using this dataset shows the vulnerability of existing multi-modal models on adversarial attacks. This finding underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source LMMs against potential malicious exploits. The resource is available at \href{this https URL}{https://github.com/isXinLiu/MM-SafetyBench}.

摘要: 警告：本文包含有害语言和图片的例子，建议读者自行决定。围绕大型语言模型(LLM)的安全问题已经得到了广泛的研究，但大型多模式模型(LMM)的安全性仍未得到充分研究。在我们的研究中，我们提出了一种新的视觉提示攻击，利用与查询相关的图像来越狱开源的LMM。我们的方法从一个由扩散模型生成的图像和另一个基于从恶意查询中提取的关键字将文本显示为排版的图像创建合成图像。我们表明，即使所使用的大型语言模型安全地对齐，LLM也可以很容易地被我们的方法攻击。为了评估这一漏洞在开源LMM中的程度，我们使用我们提出的攻击技术编制了一个包含13个场景的大量数据集，总共有5,040个文本-图像对。我们使用这个数据集对12个尖端的LMM进行了评估，表明了现有的多模式模型在对抗攻击时的脆弱性。这一发现强调了需要共同努力，加强和改进开放源码LMM的安全措施，以防范潜在的恶意利用。该资源位于\href{此HTTPS URL}{https://github.com/isXinLiu/MM-SafetyBench}.



## **14. Unveiling the Implicit Toxicity in Large Language Models**

揭示大型语言模型中的隐含毒性 cs.CL

EMNLP 2023 Main Conference

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17391v1) [paper-pdf](http://arxiv.org/pdf/2311.17391v1)

**Authors**: Jiaxin Wen, Pei Ke, Hao Sun, Zhexin Zhang, Chengfei Li, Jinfeng Bai, Minlie Huang

**Abstract**: The open-endedness of large language models (LLMs) combined with their impressive capabilities may lead to new safety issues when being exploited for malicious use. While recent studies primarily focus on probing toxic outputs that can be easily detected with existing toxicity classifiers, we show that LLMs can generate diverse implicit toxic outputs that are exceptionally difficult to detect via simply zero-shot prompting. Moreover, we propose a reinforcement learning (RL) based attacking method to further induce the implicit toxicity in LLMs. Specifically, we optimize the language model with a reward that prefers implicit toxic outputs to explicit toxic and non-toxic ones. Experiments on five widely-adopted toxicity classifiers demonstrate that the attack success rate can be significantly improved through RL fine-tuning. For instance, the RL-finetuned LLaMA-13B model achieves an attack success rate of 90.04% on BAD and 62.85% on Davinci003. Our findings suggest that LLMs pose a significant threat in generating undetectable implicit toxic outputs. We further show that fine-tuning toxicity classifiers on the annotated examples from our attacking method can effectively enhance their ability to detect LLM-generated implicit toxic language. The code is publicly available at https://github.com/thu-coai/Implicit-Toxicity.

摘要: 大型语言模型(LLM)的开放性与其令人印象深刻的能力相结合，在被恶意利用时可能会导致新的安全问题。虽然最近的研究主要集中在探测现有毒性分类器可以很容易检测到的有毒输出，但我们发现LLMS可以产生各种隐含的有毒输出，这些输出通过简单的零射击提示特别难检测到。此外，我们还提出了一种基于强化学习(RL)的攻击方法来进一步诱导LLMS中的隐含毒性。具体地说，我们优化了语言模型，奖励它更喜欢隐含的有毒输出，而不是显式的有毒和无毒的输出。在5个广泛使用的毒性分类器上的实验表明，通过RL的微调可以显著提高攻击成功率。例如，RL微调的骆驼-13B模型在BAD上的攻击成功率为90.04%，在Davinc003上的攻击成功率为62.85%。我们的发现表明，LLMS在产生无法检测到的隐含有毒输出方面构成了重大威胁。我们进一步表明，在我们的攻击方法的标注样本上微调毒性分类器可以有效地提高它们检测LLM生成的隐含有毒语言的能力。该代码可在https://github.com/thu-coai/Implicit-Toxicity.上公开获得



## **15. Identifying and Mitigating Vulnerabilities in LLM-Integrated Applications**

识别和减轻LLM集成应用程序中的漏洞 cs.CR

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.16153v2) [paper-pdf](http://arxiv.org/pdf/2311.16153v2)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Boxin Wang, Jinyuan Jia, Bo Li, Radha Poovendran

**Abstract**: Large language models (LLMs) are increasingly deployed as the service backend for LLM-integrated applications such as code completion and AI-powered search. LLM-integrated applications serve as middleware to refine users' queries with domain-specific knowledge to better inform LLMs and enhance the responses. Despite numerous opportunities and benefits, LLM-integrated applications also introduce new attack surfaces. Understanding, minimizing, and eliminating these emerging attack surfaces is a new area of research. In this work, we consider a setup where the user and LLM interact via an LLM-integrated application in the middle. We focus on the communication rounds that begin with user's queries and end with LLM-integrated application returning responses to the queries, powered by LLMs at the service backend. For this query-response protocol, we identify potential vulnerabilities that can originate from the malicious application developer or from an outsider threat initiator that is able to control the database access, manipulate and poison data that are high-risk for the user. Successful exploits of the identified vulnerabilities result in the users receiving responses tailored to the intent of a threat initiator. We assess such threats against LLM-integrated applications empowered by OpenAI GPT-3.5 and GPT-4. Our empirical results show that the threats can effectively bypass the restrictions and moderation policies of OpenAI, resulting in users receiving responses that contain bias, toxic content, privacy risk, and disinformation. To mitigate those threats, we identify and define four key properties, namely integrity, source identification, attack detectability, and utility preservation, that need to be satisfied by a safe LLM-integrated application. Based on these properties, we develop a lightweight, threat-agnostic defense that mitigates both insider and outsider threats.

摘要: 大型语言模型(LLM)越来越多地被部署为LLM集成应用程序的服务后端，例如代码完成和AI支持的搜索。LLM集成的应用程序充当中间件，使用特定于领域的知识来提炼用户的查询，以更好地通知LLM并增强响应。尽管有许多机会和好处，LLM集成应用程序也带来了新的攻击面。理解、最小化和消除这些新出现的攻击面是一个新的研究领域。在这项工作中，我们考虑一种设置，其中用户和LLM通过中间的LLM集成应用进行交互。我们关注以用户查询开始，以LLM集成的应用程序返回对查询的响应的通信回合，该应用程序由服务后端的LLMS提供支持。对于此查询-响应协议，我们确定了可能源自恶意应用程序开发人员或外部威胁发起者的潜在漏洞，该外部威胁发起者能够控制数据库访问、操纵和毒化对用户具有高风险的数据。成功利用已识别的漏洞会导致用户收到针对威胁发起者意图量身定做的响应。我们评估了针对由OpenAI GPT-3.5和GPT-4支持的LLM集成应用程序的此类威胁。我们的实验结果表明，这些威胁可以有效地绕过OpenAI的限制和适度策略，导致用户收到包含偏见、有毒内容、隐私风险和虚假信息的响应。为了缓解这些威胁，我们确定并定义了四个关键属性，即完整性、来源识别、攻击可检测性和实用程序保存，这些属性需要由安全的LLM集成应用程序来满足。基于这些特性，我们开发了一种轻量级、与威胁无关的防御系统，可以同时减轻内部和外部威胁。



## **16. Certifying LLM Safety against Adversarial Prompting**

针对敌意提示认证LLM安全 cs.CL

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2309.02705v2) [paper-pdf](http://arxiv.org/pdf/2309.02705v2)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial attacks, which add maliciously designed token sequences to a harmful prompt to bypass the model's safety guards. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We defend against three attack modes: i) adversarial suffix, which appends an adversarial sequence at the end of the prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. For example, against adversarial suffixes of length 20, it certifiably detects 92% of harmful prompts and labels 94% of safe prompts correctly using the open-source language model Llama 2 as the safety filter. We further improve the filter's performance, in terms of accuracy and speed, by replacing Llama 2 with a DistilBERT safety classifier fine-tuned on safe and harmful prompts. Additionally, we propose two efficient empirical defenses: i) RandEC, a randomized version of erase-and-check that evaluates the safety filter on a small subset of the erased subsequences, and ii) GradEC, a gradient-based version that optimizes the erased tokens to remove the adversarial sequence. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.

摘要: 发布给公众使用的大型语言模型(LLM)包括护栏，以确保其输出是安全的，通常被称为“模型对齐”。统一的语言模型应该拒绝用户制作有害内容的请求。然而，这样的安全措施很容易受到敌意攻击，这些攻击会将恶意设计的令牌序列添加到有害的提示中，以绕过模型的安全警卫。在这项工作中，我们引入了Erase-and-Check，这是第一个通过可验证的安全保证来防御敌意提示的框架。我们防御三种攻击模式：i)对抗性后缀，其在提示的末尾附加对抗性序列；ii)对抗性插入，其中对抗性序列被插入在提示中间的任何位置；以及iii)对抗性注入，其中对抗性标记被插入在提示中的任意位置，而不一定作为连续的块。实验结果表明，该方法在对安全提示保持较好经验性能的同时，对有害提示具有较强的认证安全保障。例如，对于长度为20的敌意后缀，它使用开源语言模型Llama 2作为安全过滤器，可以正确检测92%的有害提示和94%的安全提示。我们用DistilBERT安全分类器替换了Llama 2，在精度和速度方面进一步提高了过滤器的性能，该分类器根据安全和有害的提示进行了微调。此外，我们提出了两个有效的经验防御：i)RandEC，一个随机版本的Erase-and-Check，评估被擦除的子序列的一小部分上的安全过滤器；以及ii)Gradec，一个基于梯度的版本，优化被擦除的令牌以去除敌对序列。我们实验的代码可以在https://github.com/aounon/certified-llm-safety.上找到



## **17. C-ITS Environment Modeling and Attack Modeling**

C-ITS环境建模与攻击建模 cs.CR

in Korean Language, 14 Figures, 15 Pages

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.14327v2) [paper-pdf](http://arxiv.org/pdf/2311.14327v2)

**Authors**: Jaewoong Choi, Min Geun Song, Hyosun Lee, Chaeyeon Sagong, Sangbeom Park, Jaesung Lee, Jeong Do Yoo, Huy Kang Kim

**Abstract**: As technology advances, cities are evolving into smart cities, with the ability to process large amounts of data and the increasing complexity and diversification of various elements within urban areas. Among the core systems of a smart city is the Cooperative-Intelligent Transport Systems (C-ITS). C-ITS is a system where vehicles provide real-time information to drivers about surrounding traffic conditions, sudden stops, falling objects, and other accident risks through roadside base stations. It consists of road infrastructure, C-ITS centers, and vehicle terminals. However, as smart cities integrate many elements through networks and electronic control, they are susceptible to cybersecurity issues. In the case of cybersecurity problems in C-ITS, there is a significant risk of safety issues arising. This technical document aims to model the C-ITS environment and the services it provides, with the purpose of identifying the attack surface where security incidents could occur in a smart city environment. Subsequently, based on the identified attack surface, the document aims to construct attack scenarios and their respective stages. The document provides a description of the concept of C-ITS, followed by the description of the C-ITS environment model, service model, and attack scenario model defined by us.

摘要: 随着技术的进步，城市正在演变为智能城市，具有处理大量数据的能力，以及城市区域内各种要素的日益复杂和多样化。智能城市的核心系统之一是协同智能交通系统(C-ITS)。C-ITS是一种车辆通过路边基站向司机提供有关周围交通状况、突然停车、坠落物体和其他事故风险的实时信息的系统。它由道路基础设施、C-ITS中心和车辆终点站组成。然而，由于智能城市通过网络和电子控制整合了许多元素，它们容易受到网络安全问题的影响。在C-ITS中出现网络安全问题的情况下，出现安全问题的风险很大。本技术文档旨在对C-ITS环境及其提供的服务进行建模，目的是识别智能城市环境中可能发生安全事件的攻击面。随后，根据确定的攻击面，该文件旨在构建攻击情景及其各自的阶段。本文首先介绍了C-ITS的概念，然后描述了我们定义的C-ITS环境模型、服务模型和攻击场景模型。



## **18. Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information**

基于困惑度量和上下文信息的令牌级敌意提示检测 cs.CL

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.11509v2) [paper-pdf](http://arxiv.org/pdf/2311.11509v2)

**Authors**: Zhengmian Hu, Gang Wu, Saayan Mitra, Ruiyi Zhang, Tong Sun, Heng Huang, Viswanathan Swaminathan

**Abstract**: In recent years, Large Language Models (LLM) have emerged as pivotal tools in various applications. However, these models are susceptible to adversarial prompt attacks, where attackers can carefully curate input strings that lead to undesirable outputs. The inherent vulnerability of LLMs stems from their input-output mechanisms, especially when presented with intensely out-of-distribution (OOD) inputs. This paper proposes a token-level detection method to identify adversarial prompts, leveraging the LLM's capability to predict the next token's probability. We measure the degree of the model's perplexity and incorporate neighboring token information to encourage the detection of contiguous adversarial prompt sequences. As a result, we propose two methods: one that identifies each token as either being part of an adversarial prompt or not, and another that estimates the probability of each token being part of an adversarial prompt.

摘要: 近年来，大型语言模型(LLM)已经成为各种应用中的关键工具。然而，这些模型容易受到敌意提示攻击，攻击者可以仔细策划导致不良输出的输入字符串。低成本管理的内在脆弱性源于其投入-产出机制，特别是在投入严重失配(OOD)的情况下。提出了一种令牌级检测方法来识别敌意提示，利用LLM的能力来预测下一个令牌的概率。我们测量模型的困惑程度，并结合相邻令牌信息来鼓励对连续对抗性提示序列的检测。因此，我们提出了两种方法：一种是识别每个令牌是不是对抗性提示的一部分，另一种是估计每个令牌是对抗性提示的一部分的概率。



## **19. BadCLIP: Trigger-Aware Prompt Learning for Backdoor Attacks on CLIP**

BadCLIP：针对CLIP上的后门攻击的触发器感知提示学习 cs.CV

13 pages, 5 figures

**SubmitDate**: 2023-11-26    [abs](http://arxiv.org/abs/2311.16194v1) [paper-pdf](http://arxiv.org/pdf/2311.16194v1)

**Authors**: Jiawang Bai, Kuofeng Gao, Shaobo Min, Shu-Tao Xia, Zhifeng Li, Wei Liu

**Abstract**: Contrastive Vision-Language Pre-training, known as CLIP, has shown promising effectiveness in addressing downstream image recognition tasks. However, recent works revealed that the CLIP model can be implanted with a downstream-oriented backdoor. On downstream tasks, one victim model performs well on clean samples but predicts a specific target class whenever a specific trigger is present. For injecting a backdoor, existing attacks depend on a large amount of additional data to maliciously fine-tune the entire pre-trained CLIP model, which makes them inapplicable to data-limited scenarios. In this work, motivated by the recent success of learnable prompts, we address this problem by injecting a backdoor into the CLIP model in the prompt learning stage. Our method named BadCLIP is built on a novel and effective mechanism in backdoor attacks on CLIP, i.e., influencing both the image and text encoders with the trigger. It consists of a learnable trigger applied to images and a trigger-aware context generator, such that the trigger can change text features via trigger-aware prompts, resulting in a powerful and generalizable attack. Extensive experiments conducted on 11 datasets verify that the clean accuracy of BadCLIP is similar to those of advanced prompt learning methods and the attack success rate is higher than 99% in most cases. BadCLIP is also generalizable to unseen classes, and shows a strong generalization capability under cross-dataset and cross-domain settings.

摘要: 对比视觉语言预训练，也称为CLIP，在解决下游图像识别任务方面显示出了良好的效果。然而，最近的研究表明，夹子模型可以植入一个面向下游的后门。在下游任务上，一个受害者模型在干净的样本上执行得很好，但只要出现特定的触发器，就会预测特定的目标类。对于注入后门，现有的攻击依赖于大量的额外数据来恶意微调整个预先训练的剪辑模型，这使得它们不适用于数据有限的场景。在这项工作中，受最近可学习提示的成功的启发，我们通过在快速学习阶段向CLIP模型注入后门来解决这个问题。我们的方法BadCLIP是建立在对CLIP的后门攻击中的一种新颖而有效的机制上的，即通过触发器同时影响图像和文本编码器。它由应用于图像的可学习触发器和触发器感知上下文生成器组成，使得触发器可以通过触发器感知提示改变文本特征，从而产生强大且可泛化的攻击。在11个数据集上进行的大量实验证明，BadCLIP的清洁准确率与先进的快速学习方法相似，在大多数情况下攻击成功率高于99%。BadCLIP还可以泛化到看不见的类，在跨数据集和跨域设置下表现出很强的泛化能力。



## **20. Evaluating the Instruction-Following Robustness of Large Language Models to Prompt Injection**

大型语言模型对提示注入的指令跟随鲁棒性评估 cs.CL

The data and code can be found at  https://github.com/Leezekun/instruction-following-robustness-eval

**SubmitDate**: 2023-11-25    [abs](http://arxiv.org/abs/2308.10819v3) [paper-pdf](http://arxiv.org/pdf/2308.10819v3)

**Authors**: Zekun Li, Baolin Peng, Pengcheng He, Xifeng Yan

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional proficiency in instruction-following, becoming increasingly crucial across various applications. However, this capability brings with it the risk of prompt injection attacks, where attackers inject instructions into LLMs' input to elicit undesirable actions or content. Understanding the robustness of LLMs against such attacks is vital for their safe implementation. In this work, we establish a benchmark to evaluate the robustness of instruction-following LLMs against prompt injection attacks. Our objective is to determine the extent to which LLMs can be influenced by injected instructions and their ability to differentiate between these injected and original target instructions. Through extensive experiments with leading instruction-following LLMs, we uncover significant vulnerabilities in their robustness to such attacks. Our results indicate that some models are overly tuned to follow any embedded instructions in the prompt, overly focusing on the latter parts of the prompt without fully grasping the entire context. By contrast, models with a better grasp of the context and instruction-following capabilities will potentially be more susceptible to compromise by injected instructions. This underscores the need to shift the focus from merely enhancing LLMs' instruction-following capabilities to improving their overall comprehension of prompts and discernment of instructions that are appropriate to follow. We hope our in-depth analysis offers insights into the underlying causes of these vulnerabilities, aiding in the development of future solutions. Code and data are available at https://github.com/Leezekun/instruction-following-robustness-eval

摘要: 大型语言模型(LLM)在遵循指令方面表现出了非凡的熟练程度，在各种应用中变得越来越重要。然而，这种功能带来了即时注入攻击的风险，即攻击者将指令注入LLMS的输入中，以引发不受欢迎的操作或内容。了解LLMS对此类攻击的健壮性对于它们的安全实施至关重要。在这项工作中，我们建立了一个基准来评估指令跟随LLMS对即时注入攻击的健壮性。我们的目标是确定LLM受注入指令的影响程度，以及它们区分这些注入指令和原始目标指令的能力。通过对领先的遵循指令的LLM进行广泛的实验，我们发现它们对此类攻击的稳健性存在重大漏洞。我们的结果表明，一些模型过度调整，以遵循提示中的任何嵌入说明，过度关注提示的后半部分，而没有完全掌握整个上下文。相比之下，对上下文和指令遵循能力掌握得更好的模型可能更容易受到注入指令的影响。这强调了需要将重点从仅仅加强LLMS的指令遵循能力转移到提高他们对提示的整体理解和对适当遵循的指令的识别上。我们希望我们的深入分析能够深入了解这些漏洞的根本原因，有助于开发未来的解决方案。有关代码和数据，请访问https://github.com/Leezekun/instruction-following-robustness-eval



## **21. Exploiting Large Language Models (LLMs) through Deception Techniques and Persuasion Principles**

通过欺骗技术和说服原理开发大型语言模型(LLM) cs.HC

10 pages, 16 tables, 5 figures, IEEE BigData 2023 (Workshops)

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14876v1) [paper-pdf](http://arxiv.org/pdf/2311.14876v1)

**Authors**: Sonali Singh, Faranak Abri, Akbar Siami Namin

**Abstract**: With the recent advent of Large Language Models (LLMs), such as ChatGPT from OpenAI, BARD from Google, Llama2 from Meta, and Claude from Anthropic AI, gain widespread use, ensuring their security and robustness is critical. The widespread use of these language models heavily relies on their reliability and proper usage of this fascinating technology. It is crucial to thoroughly test these models to not only ensure its quality but also possible misuses of such models by potential adversaries for illegal activities such as hacking. This paper presents a novel study focusing on exploitation of such large language models against deceptive interactions. More specifically, the paper leverages widespread and borrows well-known techniques in deception theory to investigate whether these models are susceptible to deceitful interactions.   This research aims not only to highlight these risks but also to pave the way for robust countermeasures that enhance the security and integrity of language models in the face of sophisticated social engineering tactics. Through systematic experiments and analysis, we assess their performance in these critical security domains. Our results demonstrate a significant finding in that these large language models are susceptible to deception and social engineering attacks.

摘要: 随着大型语言模型(LLM)的出现，如OpenAI的ChatGPT、Google的Bard、Meta的Llama2和Anthropic AI的Claude，获得了广泛的使用，确保它们的安全性和健壮性至关重要。这些语言模型的广泛使用在很大程度上依赖于它们的可靠性和对这项迷人技术的正确使用。至关重要的是，彻底测试这些模型，不仅要确保其质量，还要确保潜在对手可能将这些模型滥用于黑客等非法活动。本文提出了一项新的研究，重点是利用如此大的语言模型来对抗欺骗性交互。更具体地说，本文利用广泛使用的欺骗理论中的著名技术来调查这些模型是否容易受到欺骗性交互作用的影响。这项研究不仅旨在强调这些风险，而且还为在复杂的社会工程策略面前增强语言模型的安全性和完整性的稳健对策铺平道路。通过系统的实验和分析，我们评估了它们在这些关键安全域中的性能。我们的结果证明了一个重要的发现，即这些大型语言模型容易受到欺骗和社会工程攻击。



## **22. Backdoor Activation Attack: Attack Large Language Models using Activation Steering for Safety-Alignment**

后门激活攻击：使用激活导向实现安全对齐来攻击大型语言模型 cs.CR

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.09433v2) [paper-pdf](http://arxiv.org/pdf/2311.09433v2)

**Authors**: Haoran Wang, Kai Shu

**Abstract**: To ensure AI safety, instruction-tuned Large Language Models (LLMs) are specifically trained to ensure alignment, which refers to making models behave in accordance with human intentions. While these models have demonstrated commendable results on various safety benchmarks, the vulnerability of their safety alignment has not been extensively studied. This is particularly troubling given the potential harm that LLMs can inflict. Existing attack methods on LLMs often rely on poisoned training data or the injection of malicious prompts. These approaches compromise the stealthiness and generalizability of the attacks, making them susceptible to detection. Additionally, these models often demand substantial computational resources for implementation, making them less practical for real-world applications. Inspired by recent success in modifying model behavior through steering vectors without the need for optimization, and drawing on its effectiveness in red-teaming LLMs, we conducted experiments employing activation steering to target four key aspects of LLMs: truthfulness, toxicity, bias, and harmfulness - across a varied set of attack settings. To establish a universal attack strategy applicable to diverse target alignments without depending on manual analysis, we automatically select the intervention layer based on contrastive layer search. Our experiment results show that activation attacks are highly effective and add little or no overhead to attack efficiency. Additionally, we discuss potential countermeasures against such activation attacks. Our code and data are available at https://github.com/wang2226/Backdoor-Activation-Attack Warning: this paper contains content that can be offensive or upsetting.

摘要: 为了确保人工智能的安全，指令调优的大型语言模型(LLM)经过专门培训，以确保对齐，这指的是使模型的行为符合人类的意图。虽然这些模型在各种安全基准上显示了值得称赞的结果，但它们的安全配准的脆弱性还没有得到广泛的研究。考虑到LLMS可能造成的潜在危害，这一点尤其令人担忧。现有的对LLMS的攻击方法往往依赖于有毒的训练数据或注入恶意提示。这些方法损害了攻击的隐蔽性和通用性，使其容易被检测到。此外，这些模型通常需要大量的计算资源才能实现，这使得它们在实际应用中不太实用。受最近在不需要优化的情况下通过转向矢量修改模型行为的成功的启发，并借鉴其在红队LLM中的有效性，我们进行了使用激活转向的实验，以针对LLM的四个关键方面：真实性、毒性、偏见和危害性--跨越不同的攻击设置。为了建立一种适用于不同目标对齐的通用攻击策略，而不依赖于人工分析，我们基于对比层搜索自动选择干预层。我们的实验结果表明，激活攻击是高效的，并且几乎没有增加攻击效率的开销。此外，我们还讨论了针对此类激活攻击的潜在对策。我们的代码和数据可以在https://github.com/wang2226/Backdoor-Activation-Attack Warning上获得：这篇文章包含可能令人反感或令人不安的内容。



## **23. Universal Jailbreak Backdoors from Poisoned Human Feedback**

来自有毒人类反馈的通用越狱后门 cs.AI

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14455v1) [paper-pdf](http://arxiv.org/pdf/2311.14455v1)

**Authors**: Javier Rando, Florian Tramèr

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.

摘要: 来自人类反馈的强化学习(RLHF)被用来对齐大型语言模型以产生有益和无害的响应。然而，先前的工作表明，这些模型可以通过找到敌意提示来越狱，这些提示可以将模型恢复到其不一致的行为。在本文中，我们考虑了一种新的威胁，即攻击者在RLHF训练数据中下毒，以便在模型中嵌入“越狱后门”。后门将触发词嵌入到模型中，其作用类似于通用的“sudo命令”：将触发词添加到任何提示中都可以实现有害的响应，而无需搜索敌意提示。通用越狱后门比之前在语言模型上研究的后门要强大得多，我们发现使用常见的后门攻击技术来植入它们要困难得多。我们调查了RLHF中有助于其所谓的健壮性的设计决策，并发布了一个有毒模型的基准，以刺激未来对通用越狱后门的研究。



## **24. Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation**

通过人物角色调整实现语言模型的可扩展和可传输黑盒越狱 cs.CL

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.03348v2) [paper-pdf](http://arxiv.org/pdf/2311.03348v2)

**Authors**: Rusheb Shah, Quentin Feuillade--Montixi, Soroush Pour, Arush Tagade, Stephen Casper, Javier Rando

**Abstract**: Despite efforts to align large language models to produce harmless responses, they are still vulnerable to jailbreak prompts that elicit unrestricted behaviour. In this work, we investigate persona modulation as a black-box jailbreaking method to steer a target model to take on personalities that are willing to comply with harmful instructions. Rather than manually crafting prompts for each persona, we automate the generation of jailbreaks using a language model assistant. We demonstrate a range of harmful completions made possible by persona modulation, including detailed instructions for synthesising methamphetamine, building a bomb, and laundering money. These automated attacks achieve a harmful completion rate of 42.5% in GPT-4, which is 185 times larger than before modulation (0.23%). These prompts also transfer to Claude 2 and Vicuna with harmful completion rates of 61.0% and 35.9%, respectively. Our work reveals yet another vulnerability in commercial large language models and highlights the need for more comprehensive safeguards.

摘要: 尽管努力调整大型语言模型以产生无害的回应，但它们仍然容易受到引发不受限制的行为的越狱提示的影响。在这项工作中，我们研究人物角色调制作为一种黑箱越狱方法，以引导目标模型承担愿意服从有害指令的人格。我们不是为每个角色手动创建提示，而是使用语言模型助手自动生成越狱。我们演示了一系列由人物角色调制实现的有害完成，包括合成甲基苯丙胺、制造炸弹和洗钱的详细说明。这些自动攻击在GPT-4中实现了42.5%的有害完成率，是调制前(0.23%)的185倍。这些提示也转移到克劳德2和维库纳，有害完成率分别为61.0%和35.9%。我们的工作揭示了商业大型语言模型中的另一个漏洞，并强调了需要更全面的保障措施。



## **25. Input Reconstruction Attack against Vertical Federated Large Language Models**

对垂直联合大型语言模型的输入重构攻击 cs.CL

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.07585v2) [paper-pdf](http://arxiv.org/pdf/2311.07585v2)

**Authors**: Fei Zheng

**Abstract**: Recently, large language models (LLMs) have drawn extensive attention from academia and the public, due to the advent of the ChatGPT. While LLMs show their astonishing ability in text generation for various tasks, privacy concerns limit their usage in real-life businesses. More specifically, either the user's inputs (the user sends the query to the model-hosting server) or the model (the user downloads the complete model) itself will be revealed during the usage. Vertical federated learning (VFL) is a promising solution to this kind of problem. It protects both the user's input and the knowledge of the model by splitting the model into a bottom part and a top part, which is maintained by the user and the model provider, respectively. However, in this paper, we demonstrate that in LLMs, VFL fails to protect the user input since it is simple and cheap to reconstruct the input from the intermediate embeddings. Experiments show that even with a commercial GPU, the input sentence can be reconstructed in only one second. We also discuss several possible solutions to enhance the privacy of vertical federated LLMs.

摘要: 近年来，随着ChatGPT的出现，大型语言模型（LLM）引起了学术界和公众的广泛关注。虽然LLM在各种任务的文本生成方面表现出惊人的能力，但隐私问题限制了它们在现实生活中的使用。更具体地说，用户的输入（用户将查询发送到模型托管服务器）或模型（用户下载完整的模型）本身将在使用期间显示。垂直联邦学习（VFL）是解决这类问题的一种很有前途的方法。它通过将模型分为底部和顶部来保护用户的输入和模型的知识，底部和顶部分别由用户和模型提供者维护。然而，在本文中，我们证明，在LLM，VFL未能保护用户输入，因为它是简单和廉价的重建输入的中间嵌入。实验表明，即使使用商业GPU，输入句子也可以在1秒内重建。我们还讨论了几种可能的解决方案，以提高垂直联邦LLM的隐私。



## **26. Fewer is More: Trojan Attacks on Parameter-Efficient Fine-Tuning**

少即是多：木马对参数高效微调的攻击 cs.CL

20 pages, 6 figures

**SubmitDate**: 2023-11-23    [abs](http://arxiv.org/abs/2310.00648v3) [paper-pdf](http://arxiv.org/pdf/2310.00648v3)

**Authors**: Lauren Hong, Ting Wang

**Abstract**: Parameter-efficient fine-tuning (PEFT) enables efficient adaptation of pre-trained language models (PLMs) to specific tasks. By tuning only a minimal set of (extra) parameters, PEFT achieves performance comparable to full fine-tuning. However, despite its prevalent use, the security implications of PEFT remain largely unexplored. In this paper, we conduct a pilot study revealing that PEFT exhibits unique vulnerability to trojan attacks. Specifically, we present PETA, a novel attack that accounts for downstream adaptation through bilevel optimization: the upper-level objective embeds the backdoor into a PLM while the lower-level objective simulates PEFT to retain the PLM's task-specific performance. With extensive evaluation across a variety of downstream tasks and trigger designs, we demonstrate PETA's effectiveness in terms of both attack success rate and unaffected clean accuracy, even after the victim user performs PEFT over the backdoored PLM using untainted data. Moreover, we empirically provide possible explanations for PETA's efficacy: the bilevel optimization inherently 'orthogonalizes' the backdoor and PEFT modules, thereby retaining the backdoor throughout PEFT. Based on this insight, we explore a simple defense that omits PEFT in selected layers of the backdoored PLM and unfreezes a subset of these layers' parameters, which is shown to effectively neutralize PETA.

摘要: 参数高效微调(PEFT)使预先训练的语言模型(PLM)能够有效地适应特定任务。通过只调整最小的一组(额外)参数，PEFT实现了与完全微调相当的性能。然而，尽管PEFT被广泛使用，但其安全影响在很大程度上仍未被探索。在本文中，我们进行了一项初步研究，揭示了PEFT对特洛伊木马攻击的独特脆弱性。具体地，我们提出了PETA，一种通过双层优化来解释下游自适应的新型攻击：上层目标将后门嵌入到PLM中，而下层目标模拟PEFT以保持PLM的任务特定性能。通过对各种下游任务和触发器设计的广泛评估，我们证明了PETA在攻击成功率和未受影响的清理准确性方面的有效性，即使受害者用户使用未受污染的数据对后备PLM执行了PEFT。此外，我们从经验上为PETA的有效性提供了可能的解释：双层优化内在地使后门和PEFT模块“正交化”，从而在整个PEFT中保留后门。基于这一认识，我们探索了一种简单的防御方法，它省略了后置PLM的选定层中的PEFT，并解冻了这些层的参数子集，这被证明有效地中和了PETA。



## **27. Transfer Attacks and Defenses for Large Language Models on Coding Tasks**

编码任务中大语言模型的迁移攻击与防御 cs.LG

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.13445v1) [paper-pdf](http://arxiv.org/pdf/2311.13445v1)

**Authors**: Chi Zhang, Zifan Wang, Ravi Mangal, Matt Fredrikson, Limin Jia, Corina Pasareanu

**Abstract**: Modern large language models (LLMs), such as ChatGPT, have demonstrated impressive capabilities for coding tasks including writing and reasoning about code. They improve upon previous neural network models of code, such as code2seq or seq2seq, that already demonstrated competitive results when performing tasks such as code summarization and identifying code vulnerabilities. However, these previous code models were shown vulnerable to adversarial examples, i.e. small syntactic perturbations that do not change the program's semantics, such as the inclusion of "dead code" through false conditions or the addition of inconsequential print statements, designed to "fool" the models. LLMs can also be vulnerable to the same adversarial perturbations but a detailed study on this concern has been lacking so far. In this paper we aim to investigate the effect of adversarial perturbations on coding tasks with LLMs. In particular, we study the transferability of adversarial examples, generated through white-box attacks on smaller code models, to LLMs. Furthermore, to make the LLMs more robust against such adversaries without incurring the cost of retraining, we propose prompt-based defenses that involve modifying the prompt to include additional information such as examples of adversarially perturbed code and explicit instructions for reversing adversarial perturbations. Our experiments show that adversarial examples obtained with a smaller code model are indeed transferable, weakening the LLMs' performance. The proposed defenses show promise in improving the model's resilience, paving the way to more robust defensive solutions for LLMs in code-related applications.

摘要: 现代大型语言模型(LLM)，如ChatGPT，已经在编码任务(包括编写代码和进行推理)方面展示了令人印象深刻的能力。它们改进了以前的代码神经网络模型，如code2seq或seq2seq，这些模型在执行代码汇总和识别代码漏洞等任务时已经展示了具有竞争力的结果。然而，这些以前的代码模型被证明容易受到敌意示例的攻击，即不会改变程序语义的小的语法扰动，例如通过错误条件包括“死代码”或添加无关紧要的打印语句，旨在“愚弄”模型。LLMS也可能容易受到同样的对抗性干扰，但迄今为止还缺乏关于这一问题的详细研究。本文旨在研究对抗性扰动对LLMS编码任务的影响。特别是，我们研究了通过对较小代码模型进行白盒攻击而生成的对抗性示例到LLMS的可转移性。此外，为了使LLMS在不招致再培训成本的情况下对此类对手更加健壮，我们提出了基于提示的防御措施，涉及修改提示以包括额外的信息，例如对手扰动代码的示例和用于逆转对手扰动的显式指令。我们的实验表明，用较小的编码模型得到的对抗性例子确实是可移植的，从而削弱了LLMS的性能。拟议的防御措施在提高模型的弹性方面显示出了希望，为代码相关应用中的LLM提供更强大的防御解决方案铺平了道路。



## **28. Open Sesame! Universal Black Box Jailbreaking of Large Language Models**

芝麻开门！大型语言模型的通用黑盒越狱 cs.CL

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2309.01446v3) [paper-pdf](http://arxiv.org/pdf/2309.01446v3)

**Authors**: Raz Lapid, Ron Langberg, Moshe Sipper

**Abstract**: Large language models (LLMs), designed to provide helpful and safe responses, often rely on alignment techniques to align with user intent and social guidelines. Unfortunately, this alignment can be exploited by malicious actors seeking to manipulate an LLM's outputs for unintended purposes. In this paper we introduce a novel approach that employs a genetic algorithm (GA) to manipulate LLMs when model architecture and parameters are inaccessible. The GA attack works by optimizing a universal adversarial prompt that -- when combined with a user's query -- disrupts the attacked model's alignment, resulting in unintended and potentially harmful outputs. Our novel approach systematically reveals a model's limitations and vulnerabilities by uncovering instances where its responses deviate from expected behavior. Through extensive experiments we demonstrate the efficacy of our technique, thus contributing to the ongoing discussion on responsible AI development by providing a diagnostic tool for evaluating and enhancing alignment of LLMs with human intent. To our knowledge this is the first automated universal black box jailbreak attack.

摘要: 大型语言模型(LLM)旨在提供有用和安全的响应，它们通常依靠对齐技术来与用户意图和社交指南保持一致。遗憾的是，恶意行为者可能会利用这种对齐，试图出于非预期目的操纵LLM的输出。在本文中，我们介绍了一种新的方法，即在模型结构和参数不可访问的情况下，使用遗传算法(GA)来操作LLM。GA攻击的工作原理是优化一个通用的对抗性提示，当与用户的查询结合在一起时，会扰乱被攻击模型的对齐，导致意外的和潜在的有害输出。我们的新方法通过揭示模型响应偏离预期行为的实例，系统地揭示了模型的局限性和漏洞。通过广泛的实验，我们展示了我们技术的有效性，从而通过提供一种诊断工具来评估和增强LLM与人类意图的一致性，从而为正在进行的关于负责任的人工智能开发的讨论做出贡献。据我们所知，这是第一次自动通用黑匣子越狱攻击。



## **29. Generating Valid and Natural Adversarial Examples with Large Language Models**

使用大型语言模型生成有效的自然对抗性实例 cs.CL

Submitted to the IEEE for possible publication

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11861v1) [paper-pdf](http://arxiv.org/pdf/2311.11861v1)

**Authors**: Zimu Wang, Wei Wang, Qi Chen, Qiufeng Wang, Anh Nguyen

**Abstract**: Deep learning-based natural language processing (NLP) models, particularly pre-trained language models (PLMs), have been revealed to be vulnerable to adversarial attacks. However, the adversarial examples generated by many mainstream word-level adversarial attack models are neither valid nor natural, leading to the loss of semantic maintenance, grammaticality, and human imperceptibility. Based on the exceptional capacity of language understanding and generation of large language models (LLMs), we propose LLM-Attack, which aims at generating both valid and natural adversarial examples with LLMs. The method consists of two stages: word importance ranking (which searches for the most vulnerable words) and word synonym replacement (which substitutes them with their synonyms obtained from LLMs). Experimental results on the Movie Review (MR), IMDB, and Yelp Review Polarity datasets against the baseline adversarial attack models illustrate the effectiveness of LLM-Attack, and it outperforms the baselines in human and GPT-4 evaluation by a significant margin. The model can generate adversarial examples that are typically valid and natural, with the preservation of semantic meaning, grammaticality, and human imperceptibility.

摘要: 基于深度学习的自然语言处理(NLP)模型，特别是预先训练的语言模型(PLM)，已经被发现容易受到对手的攻击。然而，许多主流的词级对抗性攻击模型生成的对抗性实例既不有效也不自然，导致失去了语义维护、语法和人类的不可见性。基于语言理解和生成大型语言模型(LLMS)的卓越能力，我们提出了LLM-Attack，旨在利用LLMS生成有效的和自然的对抗性实例。该方法包括两个阶段：词重要性排序(搜索最易受攻击的词)和词同义词替换(用从LLMS获得的同义词替换它们)。在Movie Review(MR)、IMDB和Yelp Review极性数据集上针对基线敌意攻击模型的实验结果表明了LLM-Attack的有效性，并且它在人类和GPT-4评估中的表现明显优于基线。该模型可以生成典型的有效和自然的对抗性例子，同时保留了语义、语法和人类的不可察觉。



## **30. Evil Geniuses: Delving into the Safety of LLM-based Agents**

邪恶天才：深入研究基于LLM的特工的安全性 cs.CL

13 pages

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11855v1) [paper-pdf](http://arxiv.org/pdf/2311.11855v1)

**Authors**: Yu Tian, Xiao Yang, Jingyuan Zhang, Yinpeng Dong, Hang Su

**Abstract**: The rapid advancements in large language models (LLMs) have led to a resurgence in LLM-based agents, which demonstrate impressive human-like behaviors and cooperative capabilities in various interactions and strategy formulations. However, evaluating the safety of LLM-based agents remains a complex challenge. This paper elaborately conducts a series of manual jailbreak prompts along with a virtual chat-powered evil plan development team, dubbed Evil Geniuses, to thoroughly probe the safety aspects of these agents. Our investigation reveals three notable phenomena: 1) LLM-based agents exhibit reduced robustness against malicious attacks. 2) the attacked agents could provide more nuanced responses. 3) the detection of the produced improper responses is more challenging. These insights prompt us to question the effectiveness of LLM-based attacks on agents, highlighting vulnerabilities at various levels and within different role specializations within the system/agent of LLM-based agents. Extensive evaluation and discussion reveal that LLM-based agents face significant challenges in safety and yield insights for future research. Our code is available at https://github.com/T1aNS1R/Evil-Geniuses.

摘要: 大语言模型的快速发展导致了基于大语言模型的代理的复兴，它们在各种交互和策略制定中显示出令人印象深刻的类似人类的行为和合作能力。然而，评估基于LLM的药物的安全性仍然是一个复杂的挑战。本文精心进行了一系列手动越狱提示，以及一个被称为邪恶天才的虚拟聊天支持的邪恶计划开发团队，以彻底探索这些特工的安全方面。我们的研究揭示了三个值得注意的现象：1)基于LLM的代理对恶意攻击的健壮性降低。2)被攻击的代理可以提供更细微的响应。3)对产生的不当反应的检测更具挑战性。这些见解促使我们质疑基于LLM的代理攻击的有效性，突出了基于LLM的代理的系统/代理中各个级别和不同角色专门化内的漏洞。广泛的评估和讨论表明，基于LLM的药物在安全性方面面临着重大挑战，并为未来的研究提供了见解。我们的代码可以在https://github.com/T1aNS1R/Evil-Geniuses.上找到



## **31. Beyond Boundaries: A Comprehensive Survey of Transferable Attacks on AI Systems**

超越边界：对人工智能系统可转移攻击的全面综述 cs.CR

**SubmitDate**: 2023-11-20    [abs](http://arxiv.org/abs/2311.11796v1) [paper-pdf](http://arxiv.org/pdf/2311.11796v1)

**Authors**: Guangjing Wang, Ce Zhou, Yuanda Wang, Bocheng Chen, Hanqing Guo, Qiben Yan

**Abstract**: Artificial Intelligence (AI) systems such as autonomous vehicles, facial recognition, and speech recognition systems are increasingly integrated into our daily lives. However, despite their utility, these AI systems are vulnerable to a wide range of attacks such as adversarial, backdoor, data poisoning, membership inference, model inversion, and model stealing attacks. In particular, numerous attacks are designed to target a particular model or system, yet their effects can spread to additional targets, referred to as transferable attacks. Although considerable efforts have been directed toward developing transferable attacks, a holistic understanding of the advancements in transferable attacks remains elusive. In this paper, we comprehensively explore learning-based attacks from the perspective of transferability, particularly within the context of cyber-physical security. We delve into different domains -- the image, text, graph, audio, and video domains -- to highlight the ubiquitous and pervasive nature of transferable attacks. This paper categorizes and reviews the architecture of existing attacks from various viewpoints: data, process, model, and system. We further examine the implications of transferable attacks in practical scenarios such as autonomous driving, speech recognition, and large language models (LLMs). Additionally, we outline the potential research directions to encourage efforts in exploring the landscape of transferable attacks. This survey offers a holistic understanding of the prevailing transferable attacks and their impacts across different domains.

摘要: 自动驾驶汽车、面部识别和语音识别系统等人工智能(AI)系统越来越多地融入我们的日常生活。然而，尽管这些人工智能系统具有实用性，但它们容易受到各种攻击，如对抗性攻击、后门攻击、数据中毒攻击、成员关系推理攻击、模型反转攻击和模型窃取攻击。具体地说，许多攻击旨在针对特定型号或系统，但其影响可能会扩散到其他目标，称为可转移攻击。尽管已经做出了相当大的努力来开发可转移攻击，但对可转移攻击的进展仍难以全面了解。在本文中，我们从可转移性的角度，特别是在网络-物理安全的背景下，全面地探讨了基于学习的攻击。我们深入研究不同的领域--图像、文本、图形、音频和视频域--以突出可转移攻击的无处不在和普遍存在的性质。本文从数据、过程、模型和系统等不同角度对现有攻击的体系结构进行了分类和回顾。我们进一步研究了可转移攻击在实际场景中的含义，如自动驾驶、语音识别和大型语言模型(LLM)。此外，我们概述了潜在的研究方向，以鼓励在探索可转移攻击的图景方面的努力。这项调查提供了对流行的可转移攻击及其跨不同领域的影响的全面了解。



## **32. SecureBERT and LLAMA 2 Empowered Control Area Network Intrusion Detection and Classification**

SecureBERT和LLAMA 2增强的控制区域网络入侵检测和分类 cs.CR

13 pages, 13 figures, 6 tables

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2311.12074v1) [paper-pdf](http://arxiv.org/pdf/2311.12074v1)

**Authors**: Xuemei Li, Huirong Fu

**Abstract**: Numerous studies have proved their effective strength in detecting Control Area Network (CAN) attacks. In the realm of understanding the human semantic space, transformer-based models have demonstrated remarkable effectiveness. Leveraging pre-trained transformers has become a common strategy in various language-related tasks, enabling these models to grasp human semantics more comprehensively. To delve into the adaptability evaluation on pre-trained models for CAN intrusion detection, we have developed two distinct models: CAN-SecureBERT and CAN-LLAMA2. Notably, our CAN-LLAMA2 model surpasses the state-of-the-art models by achieving an exceptional performance 0.999993 in terms of balanced accuracy, precision detection rate, F1 score, and a remarkably low false alarm rate of 3.10e-6. Impressively, the false alarm rate is 52 times smaller than that of the leading model, MTH-IDS (Multitiered Hybrid Intrusion Detection System). Our study underscores the promise of employing a Large Language Model as the foundational model, while incorporating adapters for other cybersecurity-related tasks and maintaining the model's inherent language-related capabilities.

摘要: 大量的研究已经证明了它们在检测控制区域网络(CAN)攻击方面的有效性。在理解人类语义空间方面，基于变换的模型表现出了显著的有效性。利用预先训练的转换器已成为各种与语言相关的任务中的常见策略，使这些模型能够更全面地掌握人类语义。为了深入研究用于CAN入侵检测的预训练模型的适应性评估，我们开发了两个不同的模型：CAN-SecureBERT和CAN-LLAMA2。值得注意的是，我们的CAN-LLAMA2模型在平衡准确率、精确度检测率、F1分数和3.10e-6的极低误警率方面实现了0.999993的卓越性能，超过了最先进的模型。令人印象深刻的是，误警率比领先的MTH-IDS(多层混合入侵检测系统)低52倍。我们的研究强调了采用大型语言模型作为基础模型的前景，同时纳入其他与网络安全相关的任务的适配器，并保持模型固有的与语言相关的能力。



## **33. A Security Risk Taxonomy for Large Language Models**

大型语言模型的安全风险分类 cs.CR

**SubmitDate**: 2023-11-19    [abs](http://arxiv.org/abs/2311.11415v1) [paper-pdf](http://arxiv.org/pdf/2311.11415v1)

**Authors**: Erik Derner, Kristina Batistič, Jan Zahálka, Robert Babuška

**Abstract**: As large language models (LLMs) permeate more and more applications, an assessment of their associated security risks becomes increasingly necessary. The potential for exploitation by malicious actors, ranging from disinformation to data breaches and reputation damage, is substantial. This paper addresses a gap in current research by focusing on the security risks posed by LLMs, which extends beyond the widely covered ethical and societal implications. Our work proposes a taxonomy of security risks along the user-model communication pipeline, explicitly focusing on prompt-based attacks on LLMs. We categorize the attacks by target and attack type within a prompt-based interaction scheme. The taxonomy is reinforced with specific attack examples to showcase the real-world impact of these risks. Through this taxonomy, we aim to inform the development of robust and secure LLM applications, enhancing their safety and trustworthiness.

摘要: 随着大型语言模型(LLM)渗透到越来越多的应用中，对其相关安全风险的评估变得越来越有必要。恶意行为者利用的可能性很大，从虚假信息到数据泄露和声誉损害。本文通过关注LLMS带来的安全风险，解决了当前研究中的一个空白，该风险超出了广泛涵盖的伦理和社会影响。我们的工作提出了一种沿着用户模型通信管道的安全风险分类，明确地将重点放在对LLM的基于提示的攻击上。我们在基于提示的交互方案中根据目标和攻击类型对攻击进行分类。分类通过具体的攻击示例得到加强，以展示这些风险的实际影响。通过这一分类，我们的目标是为健壮和安全的LLM应用程序的开发提供信息，增强它们的安全性和可信度。



## **34. FunctionMarker: Watermarking Language Datasets via Knowledge Injection**

FunctionMarker：通过知识注入为语言数据集添加水印 cs.CR

**SubmitDate**: 2023-11-17    [abs](http://arxiv.org/abs/2311.09535v2) [paper-pdf](http://arxiv.org/pdf/2311.09535v2)

**Authors**: Shuai Li, Kejiang Chen, Kunsheng Tang, Wen Huang, Jie Zhang, Weiming Zhang, Nenghai Yu

**Abstract**: Large Language Models (LLMs) have demonstrated superior performance in various natural language processing tasks. Meanwhile, they require extensive training data, raising concerns related to dataset copyright protection. Backdoor-based watermarking is a viable approach to protect the copyright of classification datasets. However, these methods may introduce malicious misclassification behaviors into watermarked LLMs by attackers and also affect the semantic information of the watermarked text. To address these issues, we propose FunctionMarker, a novel copyright protection method for language datasets via knowledge injection. FunctionMarker enables LLMs to learn specific knowledge through fine-tuning on watermarked datasets, and we can extract the embedded watermark by obtaining the responses of LLMs to specific knowledge-related queries. Considering watermark capacity and stealthness, we select customizable functions as specific knowledge for LLMs to learn and embed the watermark into them. Moreover, FunctionMarker can embed multi-bit watermarks while preserving the original semantic information, thereby increasing the difficulty of adaptive attacks. We take mathematical functions as an instance to evaluate the effectiveness of FunctionMarker, and experiments show that only 0.3% of watermarked text achieves a 90% watermark extraction accuracy in most cases, validating our method's effectiveness.

摘要: 大型语言模型在各种自然语言处理任务中表现出了优异的性能。同时，它们需要大量的培训数据，这引发了与数据集版权保护相关的担忧。基于后门的数字水印是一种可行的分类数据集版权保护方法。然而，这些方法可能会给带水印的LLMS带来恶意的误分类行为，同时也会影响带水印文本的语义信息。针对这些问题，我们提出了一种新的基于知识注入的语言数据版权保护方法FunctionMarker。FunctionMarker使LLMS能够通过微调水印数据集来学习特定的知识，而我们可以通过获取LLMS对特定知识相关查询的响应来提取嵌入的水印。考虑到水印的容量和隐蔽性，我们选择可定制的函数作为特定知识，供LLMS学习并将水印嵌入其中。此外，FunctionMarker可以在保留原始语义信息的同时嵌入多比特水印，从而增加了自适应攻击的难度。我们以数学函数为例对FunctionMarker的有效性进行了评估，实验表明，在大多数情况下，只有0.3%的水印文本达到了90%的水印提取准确率，验证了该方法的有效性。



## **35. Understanding the Effectiveness of Large Language Models in Detecting Security Vulnerabilities**

了解大型语言模型在检测安全漏洞方面的有效性 cs.CR

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.16169v1) [paper-pdf](http://arxiv.org/pdf/2311.16169v1)

**Authors**: Avishree Khare, Saikat Dutta, Ziyang Li, Alaia Solko-Breslin, Rajeev Alur, Mayur Naik

**Abstract**: Security vulnerabilities in modern software are prevalent and harmful. While automated vulnerability detection tools have made promising progress, their scalability and applicability remain challenging. Recently, Large Language Models (LLMs), such as GPT-4 and CodeLlama, have demonstrated remarkable performance on code-related tasks. However, it is unknown whether such LLMs can do complex reasoning over code. In this work, we explore whether pre-trained LLMs can detect security vulnerabilities and address the limitations of existing tools. We evaluate the effectiveness of pre-trained LLMs on a set of five diverse security benchmarks spanning two languages, Java and C/C++, and including code samples from synthetic and real-world projects. We evaluate the effectiveness of LLMs in terms of their performance, explainability, and robustness.   By designing a series of effective prompting strategies, we obtain the best results on the synthetic datasets with GPT-4: F1 scores of 0.79 on OWASP, 0.86 on Juliet Java, and 0.89 on Juliet C/C++. Expectedly, the performance of LLMs drops on the more challenging real-world datasets: CVEFixes Java and CVEFixes C/C++, with GPT-4 reporting F1 scores of 0.48 and 0.62, respectively. We show that LLMs can often perform better than existing static analysis and deep learning-based vulnerability detection tools, especially for certain classes of vulnerabilities. Moreover, LLMs also often provide reliable explanations, identifying the vulnerable data flows in code. We find that fine-tuning smaller LLMs can outperform the larger LLMs on synthetic datasets but provide limited gains on real-world datasets. When subjected to adversarial attacks on code, LLMs show mild degradation, with average accuracy reduction of up to 12.67%. Finally, we share our insights and recommendations for future work on leveraging LLMs for vulnerability detection.

摘要: 现代软件中的安全漏洞是普遍存在的，也是有害的。尽管自动化漏洞检测工具取得了可喜的进展，但它们的可扩展性和适用性仍然具有挑战性。最近，大型语言模型(LLM)，如GPT-4和CodeLlama，在与代码相关的任务上表现出了显著的性能。然而，目前还不清楚这样的LLM是否可以对代码进行复杂的推理。在这项工作中，我们探索预先训练的LLM是否可以检测安全漏洞并解决现有工具的局限性。我们在一组跨越两种语言(Java和C/C++)的五种不同安全基准上评估了预先训练的LLM的有效性，并包括来自合成项目和真实世界项目的代码样本。我们从性能、可解释性和稳健性三个方面来评估LLMS的有效性。通过设计一系列有效的激励策略，我们在GPT-4的合成数据集上获得了最好的结果：OWASP上的F1得分为0.79，Juliet Java上的F1得分为0.86，Juliet C/C++上的得分为0.89。不出所料，LLMS的性能在更具挑战性的真实数据集上有所下降：CVEFix Java和CVEFix C/C++，GPT-4报告的F1分数分别为0.48和0.62。我们表明，LLMS通常比现有的基于静态分析和深度学习的漏洞检测工具具有更好的性能，特别是对于某些类型的漏洞。此外，LLMS还经常提供可靠的解释，识别代码中易受攻击的数据流。我们发现，微调较小的LLM在合成数据集上的性能优于较大的LLM，但在真实数据集上提供的收益有限。当受到代码的敌意攻击时，LLMS表现出轻微的退化，平均准确率下降高达12.67%。最后，我们分享了我们对利用LLMS进行漏洞检测的未来工作的见解和建议。



## **36. Cognitive Overload: Jailbreaking Large Language Models with Overloaded Logical Thinking**

认知超载：用超负荷的逻辑思维越狱的大型语言模型 cs.CL

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09827v1) [paper-pdf](http://arxiv.org/pdf/2311.09827v1)

**Authors**: Nan Xu, Fei Wang, Ben Zhou, Bang Zheng Li, Chaowei Xiao, Muhao Chen

**Abstract**: While large language models (LLMs) have demonstrated increasing power, they have also given rise to a wide range of harmful behaviors. As representatives, jailbreak attacks can provoke harmful or unethical responses from LLMs, even after safety alignment. In this paper, we investigate a novel category of jailbreak attacks specifically designed to target the cognitive structure and processes of LLMs. Specifically, we analyze the safety vulnerability of LLMs in the face of (1) multilingual cognitive overload, (2) veiled expression, and (3) effect-to-cause reasoning. Different from previous jailbreak attacks, our proposed cognitive overload is a black-box attack with no need for knowledge of model architecture or access to model weights. Experiments conducted on AdvBench and MasterKey reveal that various LLMs, including both popular open-source model Llama 2 and the proprietary model ChatGPT, can be compromised through cognitive overload. Motivated by cognitive psychology work on managing cognitive load, we further investigate defending cognitive overload attack from two perspectives. Empirical studies show that our cognitive overload from three perspectives can jailbreak all studied LLMs successfully, while existing defense strategies can hardly mitigate the caused malicious uses effectively.

摘要: 虽然大型语言模型(LLM)显示出越来越大的力量，但它们也引发了广泛的有害行为。作为代表，越狱攻击可能会引发低收入国家的有害或不道德的反应，即使在安全调整之后也是如此。在本文中，我们研究了一类新的越狱攻击，该攻击专门针对LLMS的认知结构和过程而设计。具体地说，我们分析了在(1)多语言认知过载、(2)含蓄表达和(3)因果推理的情况下，LLMS的安全脆弱性。与以前的越狱攻击不同，我们提出的认知过载攻击是一种黑盒攻击，不需要了解模型体系结构或访问模型权重。在AdvBtch和MasterKey上进行的实验表明，各种LLM，包括流行的开源模型Llama 2和专有模型ChatGPT，都可以通过认知过载而受到损害。受认知心理学关于管理认知负荷的研究的启发，我们从两个角度进一步研究了防御认知过载攻击。实证研究表明，我们从三个角度的认知过载可以成功地越狱所有研究的LLM，而现有的防御策略很难有效地缓解造成的恶意使用。



## **37. Test-time Backdoor Mitigation for Black-Box Large Language Models with Defensive Demonstrations**

带有防御性演示的黑盒大型语言模型的测试时后门缓解 cs.CL

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09763v1) [paper-pdf](http://arxiv.org/pdf/2311.09763v1)

**Authors**: Wenjie Mo, Jiashu Xu, Qin Liu, Jiongxiao Wang, Jun Yan, Chaowei Xiao, Muhao Chen

**Abstract**: Existing studies in backdoor defense have predominantly focused on the training phase, overlooking the critical aspect of testing time defense. This gap becomes particularly pronounced in the context of Large Language Models (LLMs) deployed as Web Services, which typically offer only black-box access, rendering training-time defenses impractical. To bridge this gap, our work introduces defensive demonstrations, an innovative backdoor defense strategy for blackbox large language models. Our method involves identifying the task and retrieving task-relevant demonstrations from an uncontaminated pool. These demonstrations are then combined with user queries and presented to the model during testing, without requiring any modifications/tuning to the black-box model or insights into its internal mechanisms. Defensive demonstrations are designed to counteract the adverse effects of triggers, aiming to recalibrate and correct the behavior of poisoned models during test-time evaluations. Extensive experiments show that defensive demonstrations are effective in defending both instance-level and instruction-level backdoor attacks, not only rectifying the behavior of poisoned models but also surpassing existing baselines in most scenarios.

摘要: 现有的后门防御研究主要集中在训练阶段，而忽略了测试时间防御的关键方面。在部署为Web服务的大型语言模型(LLM)的环境中，这一差距变得尤为明显，这些模型通常只提供黑盒访问，使得培训时间防御不切实际。为了弥合这一差距，我们的工作引入了防御演示，这是一种针对黑盒大型语言模型的创新后门防御策略。我们的方法涉及识别任务并从未受污染的池中检索与任务相关的演示。然后，这些演示与用户查询结合在一起，并在测试期间呈现给模型，而不需要对黑盒模型进行任何修改/调整，也不需要深入了解其内部机制。防御性演示旨在抵消触发器的不利影响，旨在重新校准和纠正测试时间评估期间中毒模型的行为。大量的实验表明，防御性演示在防御实例级和指令级后门攻击方面都是有效的，不仅纠正了中毒模型的行为，而且在大多数场景下超过了现有的基线。



## **38. On the Exploitability of Reinforcement Learning with Human Feedback for Large Language Models**

人工反馈强化学习在大型语言模型中的可开发性 cs.AI

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09641v1) [paper-pdf](http://arxiv.org/pdf/2311.09641v1)

**Authors**: Jiongxiao Wang, Junlin Wu, Muhao Chen, Yevgeniy Vorobeychik, Chaowei Xiao

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) is a methodology designed to align Large Language Models (LLMs) with human preferences, playing an important role in LLMs alignment. Despite its advantages, RLHF relies on human annotators to rank the text, which can introduce potential security vulnerabilities if any adversarial annotator (i.e., attackers) manipulates the ranking score by up-ranking any malicious text to steer the LLM adversarially. To assess the red-teaming of RLHF against human preference data poisoning, we propose RankPoison, a poisoning attack method on candidates' selection of preference rank flipping to reach certain malicious behaviors (e.g., generating longer sequences, which can increase the computational cost). With poisoned dataset generated by RankPoison, we can perform poisoning attacks on LLMs to generate longer tokens without hurting the original safety alignment performance. Moreover, applying RankPoison, we also successfully implement a backdoor attack where LLMs can generate longer answers under questions with the trigger word. Our findings highlight critical security challenges in RLHF, underscoring the necessity for more robust alignment methods for LLMs.

摘要: 带人反馈的强化学习(RLHF)是一种将大语言模型与人的偏好相匹配的方法，在大语言模型对齐中起着重要作用。尽管RLHF有其优势，但它依靠人工注释者对文本进行排名，如果任何敌意注释者(即攻击者)通过对任何恶意文本进行排名来操纵排名分数，从而对LLM进行敌意操作，这可能会引入潜在的安全漏洞。为了评估RLHF的红团队对抗人类偏好数据中毒的能力，我们提出了一种毒化攻击方法RankPoison，该方法针对候选者选择偏好翻转来达到某些恶意行为(例如，生成更长的序列，这会增加计算成本)。利用RankPoison生成的有毒数据集，我们可以在不损害原始安全对齐性能的情况下，对LLM进行中毒攻击，生成更长的令牌。此外，应用RankPoison，我们还成功地实现了一个后门攻击，在带有触发词的问题下，LLMS可以生成更长的答案。我们的发现突出了RLHF中的关键安全挑战，强调了对LLM采用更强大的比对方法的必要性。



## **39. Bergeron: Combating Adversarial Attacks through a Conscience-Based Alignment Framework**

Bergeron：通过基于意识的调整框架对抗对抗性攻击 cs.CR

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2312.00029v1) [paper-pdf](http://arxiv.org/pdf/2312.00029v1)

**Authors**: Matthew Pisano, Peter Ly, Abraham Sanders, Bingsheng Yao, Dakuo Wang, Tomek Strzalkowski, Mei Si

**Abstract**: Modern Large language models (LLMs) can still generate responses that may not be aligned with human expectations or values. While many weight-based alignment methods have been proposed, many of them still leave models vulnerable to attacks when used on their own. To help mitigate this issue, we introduce Bergeron, a framework designed to improve the robustness of LLMs against adversarial attacks. Bergeron employs a two-tiered architecture. Here, a secondary LLM serves as a simulated conscience that safeguards a primary LLM. We do this by monitoring for and correcting potentially harmful text within both the prompt inputs and the generated outputs of the primary LLM. Empirical evaluation shows that Bergeron can improve the alignment and robustness of several popular LLMs without costly fine-tuning. It aids both open-source and black-box LLMs by complementing and reinforcing their existing alignment training.

摘要: 现代大型语言模型(LLM)仍然可以生成可能与人类的期望或价值观不一致的响应。虽然已经提出了许多基于权重的对齐方法，但其中许多方法仍然使模型在单独使用时容易受到攻击。为了帮助缓解这个问题，我们引入了Bergeron，这是一个旨在提高LLM抵御对手攻击的健壮性的框架。Bergeron采用了两层架构。这里，次要LLM充当保护主要LLM的模拟良知。我们通过监视和纠正主要LLM的提示输入和生成的输出中的潜在有害文本来做到这一点。经验评估表明，Bergeron可以改善几种流行的LLM的对准和鲁棒性，而不需要代价高昂的微调。它通过补充和加强现有的对准培训来帮助开源和黑盒LLM。



## **40. How Trustworthy are Open-Source LLMs? An Assessment under Malicious Demonstrations Shows their Vulnerabilities**

开源LLM的可信度有多高？恶意演示下的评估显示其漏洞 cs.CL

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09447v1) [paper-pdf](http://arxiv.org/pdf/2311.09447v1)

**Authors**: Lingbo Mo, Boshi Wang, Muhao Chen, Huan Sun

**Abstract**: The rapid progress in open-source Large Language Models (LLMs) is significantly driving AI development forward. However, there is still a limited understanding of their trustworthiness. Deploying these models at scale without sufficient trustworthiness can pose significant risks, highlighting the need to uncover these issues promptly. In this work, we conduct an assessment of open-source LLMs on trustworthiness, scrutinizing them across eight different aspects including toxicity, stereotypes, ethics, hallucination, fairness, sycophancy, privacy, and robustness against adversarial demonstrations. We propose an enhanced Chain of Utterances-based (CoU) prompting strategy by incorporating meticulously crafted malicious demonstrations for trustworthiness attack. Our extensive experiments encompass recent and representative series of open-source LLMs, including Vicuna, MPT, Falcon, Mistral, and Llama 2. The empirical outcomes underscore the efficacy of our attack strategy across diverse aspects. More interestingly, our result analysis reveals that models with superior performance in general NLP tasks do not always have greater trustworthiness; in fact, larger models can be more vulnerable to attacks. Additionally, models that have undergone instruction tuning, focusing on instruction following, tend to be more susceptible, although fine-tuning LLMs for safety alignment proves effective in mitigating adversarial trustworthiness attacks.

摘要: 开源大型语言模型(LLM)的快速发展显著地推动了人工智能的发展。然而，人们对他们的可信性仍知之甚少。在缺乏足够可信度的情况下大规模部署这些模型可能会带来重大风险，这突显了迅速发现这些问题的必要性。在这项工作中，我们对开源LLM的可信性进行了评估，从八个不同的方面对它们进行了仔细的审查，包括毒性、刻板印象、道德、幻觉、公平性、奉承、隐私和对对手演示的健壮性。我们提出了一种增强的基于话语链(CUU)的提示策略，该策略结合了精心制作的恶意演示来进行可信度攻击。我们广泛的实验涵盖了最近一系列具有代表性的开源LLM，包括Vicuna、MPT、Falcon、Mistral和Llama 2。经验结果强调了我们攻击策略在不同方面的有效性。更有趣的是，我们的结果分析显示，在一般NLP任务中性能优越的模型并不总是具有更大的可信度；事实上，较大的模型可能更容易受到攻击。此外，经过指令调整、专注于指令遵循的模型往往更容易受到影响，尽管针对安全对齐的微调LLM被证明在减轻对手信任攻击方面是有效的。



## **41. Jailbreaking GPT-4V via Self-Adversarial Attacks with System Prompts**

通过系统提示的自我对抗性攻击越狱GPT-4V cs.CR

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09127v1) [paper-pdf](http://arxiv.org/pdf/2311.09127v1)

**Authors**: Yuanwei Wu, Xiang Li, Yixin Liu, Pan Zhou, Lichao Sun

**Abstract**: Existing work on jailbreak Multimodal Large Language Models (MLLMs) has focused primarily on adversarial examples in model inputs, with less attention to vulnerabilities in model APIs. To fill the research gap, we carry out the following work: 1) We discover a system prompt leakage vulnerability in GPT-4V. Through carefully designed dialogue, we successfully steal the internal system prompts of GPT-4V. This finding indicates potential exploitable security risks in MLLMs; 2)Based on the acquired system prompts, we propose a novel MLLM jailbreaking attack method termed SASP (Self-Adversarial Attack via System Prompt). By employing GPT-4 as a red teaming tool against itself, we aim to search for potential jailbreak prompts leveraging stolen system prompts. Furthermore, in pursuit of better performance, we also add human modification based on GPT-4's analysis, which further improves the attack success rate to 98.7\%; 3) We evaluated the effect of modifying system prompts to defend against jailbreaking attacks. Results show that appropriately designed system prompts can significantly reduce jailbreak success rates. Overall, our work provides new insights into enhancing MLLM security, demonstrating the important role of system prompts in jailbreaking, which could be leveraged to greatly facilitate jailbreak success rates while also holding the potential for defending against jailbreaks.

摘要: 现有关于越狱多模式大型语言模型(MLLMS)的工作主要集中在模型输入中的对抗性示例，对模型API中的漏洞关注较少。为了填补这一研究空白，我们开展了以下工作：1)在GPT-4V中发现了一个系统即时泄漏漏洞。通过精心设计的对话，我们成功窃取了GPT-4V的内部系统提示。2)基于获得的系统提示，提出了一种新的基于系统提示的MLLM越狱攻击方法SASP(Self-Aversarial Attack by System Prompt)。通过使用GPT-4作为针对自己的红色团队工具，我们的目标是利用被盗的系统提示来搜索潜在的越狱提示。此外，为了追求更好的性能，我们还在GPT-4的S分析的基础上增加了人工修改，进一步将攻击成功率提高到98.7%。3)评估了修改系统提示对越狱攻击的防御效果。结果表明，设计适当的系统提示可以显著降低越狱成功率。总体而言，我们的工作为加强MLLM安全提供了新的见解，展示了系统提示在越狱中的重要作用，这可以被用来极大地提高越狱成功率，同时也保持了防御越狱的潜力。



## **42. Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization**

通过目标优先顺序保护大型语言模型免受越狱攻击 cs.CL

14 pages

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09096v1) [paper-pdf](http://arxiv.org/pdf/2311.09096v1)

**Authors**: Zhexin Zhang, Junxiao Yang, Pei Ke, Minlie Huang

**Abstract**: Large Language Models (LLMs) continue to advance in their capabilities, yet this progress is accompanied by a growing array of safety risks. While significant attention has been dedicated to exploiting weaknesses in LLMs through jailbreaking attacks, there remains a paucity of exploration into defending against these attacks. We point out a pivotal factor contributing to the success of jailbreaks: the inherent conflict between the goals of being helpful and ensuring safety. To counter jailbreaking attacks, we propose to integrate goal prioritization at both training and inference stages. Implementing goal prioritization during inference substantially diminishes the Attack Success Rate (ASR) of jailbreaking attacks, reducing it from 66.4% to 2.0% for ChatGPT and from 68.2% to 19.4% for Vicuna-33B, without compromising general performance. Furthermore, integrating the concept of goal prioritization into the training phase reduces the ASR from 71.0% to 6.6% for LLama2-13B. Remarkably, even in scenarios where no jailbreaking samples are included during training, our approach slashes the ASR by half, decreasing it from 71.0% to 34.0%. Additionally, our findings reveal that while stronger LLMs face greater safety risks, they also possess a greater capacity to be steered towards defending against such attacks. We hope our work could contribute to the comprehension of jailbreaking attacks and defenses, and shed light on the relationship between LLMs' capability and safety. Our code will be available at \url{https://github.com/thu-coai/JailbreakDefense_GoalPriority}.

摘要: 大型语言模型(LLM)的能力不断提高，但伴随这一进步的是越来越多的安全风险。虽然人们一直致力于通过越狱攻击来利用LLMS的弱点，但在防御这些攻击方面的探索仍然很少。我们指出了越狱成功的一个关键因素：提供帮助的目标与确保安全之间的内在冲突。为了对抗越狱攻击，我们建议在训练和推理阶段整合目标优先级。在推理过程中实施目标优先级显著降低了越狱攻击的攻击成功率(ASR)，将ChatGPT的攻击成功率从66.4%降低到2.0%，将Vicuna-33B的攻击成功率从68.2%降低到19.4%，而不会影响总体性能。此外，将目标优先顺序的概念整合到培训阶段，可以将LLama2-13B的ASR从71.0%降低到6.6%。值得注意的是，即使在训练过程中不包括越狱样本的情况下，我们的方法也将ASR削减了一半，从71.0%降低到34.0%。此外，我们的研究结果表明，虽然更强大的LLMS面临更大的安全风险，但它们也拥有更大的能力来防御此类攻击。我们希望我们的工作能够有助于理解越狱攻击和防御，并阐明LLMS的能力和安全之间的关系。我们的代码将在\url{https://github.com/thu-coai/JailbreakDefense_GoalPriority}.上提供



## **43. Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models**

沙子中的水印：生成模型不可能有强水印 cs.LG

Blog post:  https://www.harvard.edu/kempner-institute/2023/11/09/watermarking-in-the-sand/

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.04378v2) [paper-pdf](http://arxiv.org/pdf/2311.04378v2)

**Authors**: Hanlin Zhang, Benjamin L. Edelman, Danilo Francati, Daniele Venturi, Giuseppe Ateniese, Boaz Barak

**Abstract**: Watermarking generative models consists of planting a statistical signal (watermark) in a model's output so that it can be later verified that the output was generated by the given model. A strong watermarking scheme satisfies the property that a computationally bounded attacker cannot erase the watermark without causing significant quality degradation. In this paper, we study the (im)possibility of strong watermarking schemes. We prove that, under well-specified and natural assumptions, strong watermarking is impossible to achieve. This holds even in the private detection algorithm setting, where the watermark insertion and detection algorithms share a secret key, unknown to the attacker. To prove this result, we introduce a generic efficient watermark attack; the attacker is not required to know the private key of the scheme or even which scheme is used. Our attack is based on two assumptions: (1) The attacker has access to a "quality oracle" that can evaluate whether a candidate output is a high-quality response to a prompt, and (2) The attacker has access to a "perturbation oracle" which can modify an output with a nontrivial probability of maintaining quality, and which induces an efficiently mixing random walk on high-quality outputs. We argue that both assumptions can be satisfied in practice by an attacker with weaker computational capabilities than the watermarked model itself, to which the attacker has only black-box access. Furthermore, our assumptions will likely only be easier to satisfy over time as models grow in capabilities and modalities. We demonstrate the feasibility of our attack by instantiating it to attack three existing watermarking schemes for large language models: Kirchenbauer et al. (2023), Kuditipudi et al. (2023), and Zhao et al. (2023). The same attack successfully removes the watermarks planted by all three schemes, with only minor quality degradation.

摘要: 水印生成模型包括在模型的输出中植入统计信号(水印)，以便稍后可以验证输出是由给定模型生成的。强水印方案满足这样的性质，即计算受限的攻击者不可能在不引起显著质量降级的情况下删除水印。本文研究了强水印方案的(Im)可能性。我们证明了在明确和自然的假设下，强水印是不可能实现的。即使在私有检测算法设置中也是如此，其中水印插入和检测算法共享攻击者未知的秘密密钥。为了证明这一结果，我们引入了一种通用的高效水印攻击；攻击者不需要知道方案的私钥，甚至不需要知道使用了哪个方案。我们的攻击基于两个假设：(1)攻击者可以访问可以评估候选输出是否是对提示的高质量响应的“质量预言”，以及(2)攻击者可以访问“扰动预言”，它可以以保持质量的非平凡概率修改输出，并导致对高质量输出的有效混合随机游走。我们认为，在实践中，这两个假设都可以由计算能力弱于水印模型本身的攻击者满足，因为攻击者只能访问黑盒。此外，随着模型在功能和模式方面的发展，随着时间的推移，我们的假设可能只会更容易满足。我们通过将其实例化来攻击三个现有的用于大型语言模型的水印方案来证明该攻击的可行性：Kirchenbauer等人。(2023)，Kuditipudi等人。(2023)，和赵等人。(2023年)。同样的攻击成功地删除了所有三个方案植入的水印，只有很小的质量下降。



## **44. Alignment is not sufficient to prevent large language models from generating harmful information: A psychoanalytic perspective**

对齐不足以防止大型语言模型产生有害信息：从精神分析的角度 cs.CL

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.08487v1) [paper-pdf](http://arxiv.org/pdf/2311.08487v1)

**Authors**: Zi Yin, Wei Ding, Jia Liu

**Abstract**: Large Language Models (LLMs) are central to a multitude of applications but struggle with significant risks, notably in generating harmful content and biases. Drawing an analogy to the human psyche's conflict between evolutionary survival instincts and societal norm adherence elucidated in Freud's psychoanalysis theory, we argue that LLMs suffer a similar fundamental conflict, arising between their inherent desire for syntactic and semantic continuity, established during the pre-training phase, and the post-training alignment with human values. This conflict renders LLMs vulnerable to adversarial attacks, wherein intensifying the models' desire for continuity can circumvent alignment efforts, resulting in the generation of harmful information. Through a series of experiments, we first validated the existence of the desire for continuity in LLMs, and further devised a straightforward yet powerful technique, such as incomplete sentences, negative priming, and cognitive dissonance scenarios, to demonstrate that even advanced LLMs struggle to prevent the generation of harmful information. In summary, our study uncovers the root of LLMs' vulnerabilities to adversarial attacks, hereby questioning the efficacy of solely relying on sophisticated alignment methods, and further advocates for a new training idea that integrates modal concepts alongside traditional amodal concepts, aiming to endow LLMs with a more nuanced understanding of real-world contexts and ethical considerations.

摘要: 大型语言模型(LLM)是众多应用程序的核心，但面临着巨大的风险，特别是在生成有害内容和偏见方面。通过类比弗洛伊德精神分析理论中阐明的人类心理在进化生存本能和遵守社会规范之间的冲突，我们认为LLMS在训练前建立的对句法和语义连续性的内在愿望与训练后与人类价值观的一致性之间存在着类似的根本冲突。这种冲突使LLM容易受到对抗性攻击，其中加强模型对连续性的渴望可以绕过对齐工作，从而导致有害信息的产生。通过一系列实验，我们首先验证了LLMS中对连续性的渴望的存在，并进一步设计了一种简单而强大的技术，如不完整句子、负启动和认知不协调情景，以证明即使是高级LLMS也难以防止有害信息的产生。综上所述，我们的研究揭示了LLMS易受敌意攻击的根源，由此质疑单纯依赖复杂的对齐方法的有效性，并进一步倡导一种新的训练思想，将情态概念与传统的非模态概念相结合，旨在赋予LLMS对现实世界背景和伦理考虑的更细微的理解。



## **45. LatticeGen: A Cooperative Framework which Hides Generated Text in a Lattice for Privacy-Aware Generation on Cloud**

LatticeGen：一种将生成的文本隐藏在网格中的云隐私感知生成框架 cs.CL

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2309.17157v3) [paper-pdf](http://arxiv.org/pdf/2309.17157v3)

**Authors**: Mengke Zhang, Tianxing He, Tianle Wang, Lu Mi, Fatemehsadat Mireshghallah, Binyi Chen, Hao Wang, Yulia Tsvetkov

**Abstract**: In the current user-server interaction paradigm of prompted generation with large language models (LLM) on cloud, the server fully controls the generation process, which leaves zero options for users who want to keep the generated text to themselves. We propose LatticeGen, a cooperative framework in which the server still handles most of the computation while the user controls the sampling operation. The key idea is that the true generated sequence is mixed with noise tokens by the user and hidden in a noised lattice. Considering potential attacks from a hypothetically malicious server and how the user can defend against it, we propose the repeated beam-search attack and the mixing noise scheme. In our experiments we apply LatticeGen to protect both prompt and generation. It is shown that while the noised lattice degrades generation quality, LatticeGen successfully protects the true generation to a remarkable degree under strong attacks (more than 50% of the semantic remains hidden as measured by BERTScore).

摘要: 在当前云上使用大型语言模型(LLM)进行提示生成的用户-服务器交互模式中，服务器完全控制生成过程，这为想要将生成的文本保密的用户留下了零的选择。我们提出了LatticeGen，这是一个协作框架，其中服务器仍然处理大部分计算，而用户控制采样操作。其关键思想是，用户将真实生成的序列与噪声令牌混合，并将其隐藏在有噪声的网格中。考虑到来自假设恶意服务器的潜在攻击以及用户如何防御它，我们提出了重复波束搜索攻击和混合噪声方案。在我们的实验中，我们应用LatticeGen来保护提示和生成。实验结果表明，虽然加噪的格子降低了生成质量，但在强攻击下(BERTScore测试50%以上的语义仍然隐藏)，LatticeGen在很大程度上保护了真实的生成。



## **46. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

披着羊皮的狼：广义嵌套越狱提示可以轻松愚弄大型语言模型 cs.CL

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.08268v1) [paper-pdf](http://arxiv.org/pdf/2311.08268v1)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on another white-box model, compromising generalization or jailbreak efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we offer detailed analysis and discussion from the perspective of prompt execution priority on the failure of LLMs' defense. We hope that our research can catalyze both the academic community and LLMs vendors towards the provision of safer and more regulated Large Language Models.

摘要: 大型语言模型（LLM），如ChatGPT和GPT-4，旨在提供有用和安全的响应。然而，被称为“越狱”的对抗性提示可以规避保障措施，导致LLM生成有害内容。探索越狱提示可以帮助更好地揭示LLM的弱点，并进一步引导我们保护它们。不幸的是，现有的越狱方法要么遭受复杂的手动设计，要么需要在另一个白盒模型上进行优化，从而影响泛化或越狱效率。本文将越狱提示攻击归纳为两个方面：（1）提示重写和（2）场景嵌套。在此基础上，我们提出了ReNeLLM，一个自动框架，利用LLM本身来生成有效的越狱提示。大量的实验表明，与现有的基线相比，ReNeLLM显着提高了攻击成功率，同时大大降低了时间成本。我们的研究也揭示了目前的防御方法在保护LLM方面的不足。最后，从及时执行优先权的角度对有限责任公司抗辩失败进行了详细的分析和探讨。我们希望我们的研究能够促进学术界和LLM供应商提供更安全，更规范的大型语言模型。



## **47. Fake Alignment: Are LLMs Really Aligned Well?**

假对齐：LLM真的对齐得很好吗？ cs.CL

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.05915v2) [paper-pdf](http://arxiv.org/pdf/2311.05915v2)

**Authors**: Yixu Wang, Yan Teng, Kexin Huang, Chengqi Lyu, Songyang Zhang, Wenwei Zhang, Xingjun Ma, Yu-Gang Jiang, Yu Qiao, Yingchun Wang

**Abstract**: The growing awareness of safety concerns in large language models (LLMs) has sparked considerable interest in the evaluation of safety within current research endeavors. This study investigates an interesting issue pertaining to the evaluation of LLMs, namely the substantial discrepancy in performance between multiple-choice questions and open-ended questions. Inspired by research on jailbreak attack patterns, we argue this is caused by mismatched generalization. That is, the LLM does not have a comprehensive understanding of the complex concept of safety. Instead, it only remembers what to answer for open-ended safety questions, which makes it unable to solve other forms of safety tests. We refer to this phenomenon as fake alignment and construct a comparative benchmark to empirically verify its existence in LLMs. Such fake alignment renders previous evaluation protocols unreliable. To address this, we introduce the Fake alIgNment Evaluation (FINE) framework and two novel metrics--Consistency Score (CS) and Consistent Safety Score (CSS), which jointly assess two complementary forms of evaluation to quantify fake alignment and obtain corrected performance estimates. Applying FINE to 14 widely-used LLMs reveals several models with purported safety are poorly aligned in practice. Our work highlights potential limitations in prevailing alignment methodologies.

摘要: 大型语言模型(LLM)中安全问题的意识日益增强，这引发了人们对当前研究工作中的安全性评估的极大兴趣。本研究调查了一个与学习记忆能力评估相关的有趣问题，即多项选择题和开放式题在成绩上的显著差异。受越狱攻击模式研究的启发，我们认为这是由不匹配的泛化造成的。也就是说，LLM对复杂的安全概念没有全面的理解。相反，它只记得对开放式安全问题回答什么，这使得它无法解决其他形式的安全测试。我们将这种现象称为伪对齐，并构建了一个比较基准来实证验证这种现象在低密度脂蛋白中的存在。这种虚假的比对使得以前的评估协议不可靠。为了解决这一问题，我们引入了伪对齐评估(FINE)框架和两个新的度量--一致性分数(CS)和一致安全分数(CS)，它们联合评估两种互补的评估形式来量化伪对齐并获得正确的性能估计。将FINE应用于14个广泛使用的LLM，发现几种声称安全的模型在实践中不太一致。我们的工作突出了主流比对方法的潜在局限性。



## **48. MART: Improving LLM Safety with Multi-round Automatic Red-Teaming**

MART：用多轮自动红队提高LLM安全 cs.CL

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.07689v1) [paper-pdf](http://arxiv.org/pdf/2311.07689v1)

**Authors**: Suyu Ge, Chunting Zhou, Rui Hou, Madian Khabsa, Yi-Chia Wang, Qifan Wang, Jiawei Han, Yuning Mao

**Abstract**: Red-teaming is a common practice for mitigating unsafe behaviors in Large Language Models (LLMs), which involves thoroughly assessing LLMs to identify potential flaws and addressing them with responsible and accurate responses. While effective, manual red-teaming is costly, and existing automatic red-teaming typically discovers safety risks without addressing them. In this paper, we propose a Multi-round Automatic Red-Teaming (MART) method, which incorporates both automatic adversarial prompt writing and safe response generation, significantly increasing red-teaming scalability and the safety of the target LLM. Specifically, an adversarial LLM and a target LLM interplay with each other in an iterative manner, where the adversarial LLM aims to generate challenging prompts that elicit unsafe responses from the target LLM, while the target LLM is fine-tuned with safety aligned data on these adversarial prompts. In each round, the adversarial LLM crafts better attacks on the updated target LLM, while the target LLM also improves itself through safety fine-tuning. On adversarial prompt benchmarks, the violation rate of an LLM with limited safety alignment reduces up to 84.7% after 4 rounds of MART, achieving comparable performance to LLMs with extensive adversarial prompt writing. Notably, model helpfulness on non-adversarial prompts remains stable throughout iterations, indicating the target LLM maintains strong performance on instruction following.

摘要: 红团队是减少大型语言模型(LLM)中不安全行为的一种常见做法，它涉及彻底评估LLM以识别潜在缺陷并以负责任和准确的响应来解决它们。虽然有效，但手动红色团队成本高昂，而且现有的自动红色团队通常会发现安全风险，而不解决这些风险。本文提出了一种多轮自动红队(MART)方法，该方法结合了自动编写敌方提示和安全响应生成的功能，显著提高了红队的可扩展性和目标LLM的安全性。具体地说，对抗性LLM和目标LLM以迭代的方式相互作用，其中对抗性LLM旨在生成引起来自目标LLM的不安全响应的挑战性提示，而目标LLM利用关于这些对抗性提示的安全对齐的数据进行微调。在每一轮中，对抗性的LLM对更新后的目标LLM进行更好的攻击，而目标LLM也通过安全微调来提高自己。在对抗性提示基准上，有限安全对齐的LLM在4轮MART后的违规率降低了84.7%，获得了与具有广泛对抗性提示书写的LLMS相当的性能。值得注意的是，在非对抗性提示上的模型帮助在迭代过程中保持稳定，这表明目标LLM在指令跟随上保持着强大的性能。



## **49. Summon a Demon and Bind it: A Grounded Theory of LLM Red Teaming in the Wild**

召唤一个恶魔并绑定它：LLM Red Teaming in the Wild的基础理论 cs.CL

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.06237v2) [paper-pdf](http://arxiv.org/pdf/2311.06237v2)

**Authors**: Nanna Inie, Jonathan Stray, Leon Derczynski

**Abstract**: Engaging in the deliberate generation of abnormal outputs from large language models (LLMs) by attacking them is a novel human activity. This paper presents a thorough exposition of how and why people perform such attacks. Using a formal qualitative methodology, we interviewed dozens of practitioners from a broad range of backgrounds, all contributors to this novel work of attempting to cause LLMs to fail. We relate and connect this activity between its practitioners' motivations and goals; the strategies and techniques they deploy; and the crucial role the community plays. As a result, this paper presents a grounded theory of how and why people attack large language models: LLM red teaming in the wild.

摘要: 通过攻击大型语言模型（LLM）来故意生成异常输出是一种新颖的人类活动。本文对人们如何以及为什么进行这种攻击进行了深入的阐述。使用正式的定性方法，我们采访了数十名来自广泛背景的从业者，他们都是这项试图导致LLM失败的新颖工作的贡献者。我们将这一活动与其从业者的动机和目标、他们部署的策略和技术以及社区发挥的关键作用联系起来。因此，本文提出了一个关于人们如何以及为什么攻击大型语言模型的基础理论：LLM Red Teaming in the Wild。



## **50. Language Model Unalignment: Parametric Red-Teaming to Expose Hidden Harms and Biases**

语言模型不一致：暴露隐藏的危害和偏见的参数红色团队 cs.CL

Under Review

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2310.14303v2) [paper-pdf](http://arxiv.org/pdf/2310.14303v2)

**Authors**: Rishabh Bhardwaj, Soujanya Poria

**Abstract**: Red-teaming has been a widely adopted way to evaluate the harmfulness of Large Language Models (LLMs). It aims to jailbreak a model's safety behavior to make it act as a helpful agent disregarding the harmfulness of the query. Existing methods are primarily based on input text-based red-teaming such as adversarial prompts, low-resource prompts, or contextualized prompts to condition the model in a way to bypass its safe behavior. Bypassing the guardrails uncovers hidden harmful information and biases in the model that are left untreated or newly introduced by its safety training. However, prompt-based attacks fail to provide such a diagnosis owing to their low attack success rate, and applicability to specific models. In this paper, we present a new perspective on LLM safety research i.e., parametric red-teaming through Unalignment. It simply (instruction) tunes the model parameters to break model guardrails that are not deeply rooted in the model's behavior. Unalignment using as few as 100 examples can significantly bypass commonly referred to as CHATGPT, to the point where it responds with an 88% success rate to harmful queries on two safety benchmark datasets. On open-source models such as VICUNA-7B and LLAMA-2-CHAT 7B AND 13B, it shows an attack success rate of more than 91%. On bias evaluations, Unalignment exposes inherent biases in safety-aligned models such as CHATGPT and LLAMA- 2-CHAT where the model's responses are strongly biased and opinionated 64% of the time.

摘要: 红团队已被广泛采用来评估大型语言模型的危害性。它的目的是让模特的安全行为越狱，使其成为一个有帮助的代理人，而不考虑询问的危害性。现有方法主要基于诸如对抗性提示、低资源提示或情境化提示的基于输入文本的红团队，以使模型以绕过其安全行为的方式调节。绕过护栏发现了模型中隐藏的有害信息和偏见，这些信息和偏见是未经处理的或安全培训新引入的。然而，基于提示的攻击无法提供这样的诊断，因为它们的攻击成功率低，并且适用于特定的模型。在这篇文章中，我们提出了一个新的视角来研究LLM安全，即通过非对齐的参数红组。它只是(指令)调整模型参数，以打破并不深深植根于模型行为中的模型护栏。只要使用100个例子，UnAlign就可以显著绕过通常所说的CHATGPT，以至于它对两个安全基准数据集上的有害查询的响应成功率为88%。在VIVUNA-7B和LLAMA-2-Chat 7B和13B等开源机型上，攻击成功率超过91%。在偏差评估方面，UnAlign暴露了安全对齐模型中的固有偏见，如CHATGPT和Llama-2-Chat，其中模型的反应在64%的时间内是强烈偏见和固执己见的。



