# Latest Adversarial Attack Papers
**update at 2023-11-10 10:06:48**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. FigStep: Jailbreaking Large Vision-language Models via Typographic Visual Prompts**

图：通过排版视觉提示越狱的大型视觉语言模型 cs.CR

Technical Report

**SubmitDate**: 2023-11-09    [abs](http://arxiv.org/abs/2311.05608v1) [paper-pdf](http://arxiv.org/pdf/2311.05608v1)

**Authors**: Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, Xiaoyun Wang

**Abstract**: Large vision-language models (VLMs) like GPT-4V represent an unprecedented revolution in the field of artificial intelligence (AI). Compared to single-modal large language models (LLMs), VLMs possess more versatile capabilities by incorporating additional modalities (e.g., images). Meanwhile, there's a rising enthusiasm in the AI community to develop open-source VLMs, such as LLaVA and MiniGPT4, which, however, have not undergone rigorous safety assessment. In this paper, to demonstrate that more modalities lead to unforeseen AI safety issues, we propose FigStep, a novel jailbreaking framework against VLMs. FigStep feeds harmful instructions into VLMs through the image channel and then uses benign text prompts to induce VLMs to output contents that violate common AI safety policies. Our experimental results show that FigStep can achieve an average attack success rate of 94.8% across 2 families of popular open-source VLMs, LLaVA and MiniGPT4 (a total of 5 VLMs). Moreover, we demonstrate that the methodology of FigStep can even jailbreak GPT-4V, which already leverages several system-level mechanisms to filter harmful queries. Above all, our experimental results reveal that VLMs are vulnerable to jailbreaking attacks, which highlights the necessity of novel safety alignments between visual and textual modalities.

摘要: 像GPT-4V这样的大型视觉语言模型(VLM)代表着人工智能(AI)领域的一场前所未有的革命。与单一模式的大型语言模型相比，大型语言模型通过加入额外的模式(如图像)而具有更多的通用性。与此同时，人工智能社区对开发开源VLM的热情日益高涨，如LLaVA和MiniGPT4，然而，这些VLM尚未经过严格的安全评估。在本文中，为了证明更多的模式会导致不可预见的人工智能安全问题，我们提出了一种新的针对VLM的越狱框架FigStep。FigStep通过图像通道将有害指令反馈到VLM，然后使用良性文本提示诱导VLM输出违反常见AI安全策略的内容。我们的实验结果表明，FigStep可以在LLaVA和MiniGPT4两个流行的开源VLM家族(总共5个VLM)上获得94.8%的平均攻击成功率。此外，我们还演示了FigStep的方法甚至可以越狱GPT-4V，它已经利用了几个系统级机制来过滤有害的查询。最重要的是，我们的实验结果表明，VLM容易受到越狱攻击，这突显了视觉和文本通道之间新的安全对齐的必要性。



## **2. Removing RLHF Protections in GPT-4 via Fine-Tuning**

通过微调消除GPT-4中的RLHF保护 cs.CL

**SubmitDate**: 2023-11-09    [abs](http://arxiv.org/abs/2311.05553v1) [paper-pdf](http://arxiv.org/pdf/2311.05553v1)

**Authors**: Qiusi Zhan, Richard Fang, Rohan Bindu, Akul Gupta, Tatsunori Hashimoto, Daniel Kang

**Abstract**: As large language models (LLMs) have increased in their capabilities, so does their potential for dual use. To reduce harmful outputs, produces and vendors of LLMs have used reinforcement learning with human feedback (RLHF). In tandem, LLM vendors have been increasingly enabling fine-tuning of their most powerful models. However, concurrent work has shown that fine-tuning can remove RLHF protections. We may expect that the most powerful models currently available (GPT-4) are less susceptible to fine-tuning attacks.   In this work, we show the contrary: fine-tuning allows attackers to remove RLHF protections with as few as 340 examples and a 95% success rate. These training examples can be automatically generated with weaker models. We further show that removing RLHF protections does not decrease usefulness on non-censored outputs, providing evidence that our fine-tuning strategy does not decrease usefulness despite using weaker models to generate training data. Our results show the need for further research on protections on LLMs.

摘要: 随着大型语言模型(LLM)能力的增强，它们的双重用途的潜力也在增加。为了减少有害的产出，低成本管理的生产商和供应商使用了带人类反馈的强化学习(RLHF)。与此同时，LLM供应商越来越多地支持对其最强大的模型进行微调。然而，同时进行的研究表明，微调可以消除RLHF保护。我们可以预计，目前可用的最强大的型号(GPT-4)不太容易受到微调攻击。在这项工作中，我们展示了相反的情况：微调允许攻击者删除RLHF保护，只需340个例子，成功率为95%。这些训练样本可以用较弱的模型自动生成。我们进一步表明，取消RLHF保护不会降低对非删失输出的有用性，这提供了证据，表明我们的微调策略不会降低有用性，尽管使用较弱的模型来生成训练数据。我们的研究结果表明，对低灵敏材料的保护还需要进一步的研究。



## **3. Backdoor Attacks and Countermeasures in Natural Language Processing Models: A Comprehensive Security Review**

自然语言处理模型中的后门攻击和对策：全面的安全审查 cs.CR

21 pages, 4 figures

**SubmitDate**: 2023-11-08    [abs](http://arxiv.org/abs/2309.06055v4) [paper-pdf](http://arxiv.org/pdf/2309.06055v4)

**Authors**: Pengzhou Cheng, Zongru Wu, Wei Du, Haodong Zhao, Wei Lu, Gongshen Liu

**Abstract**: Applicating third-party data and models has become a new paradigm for language modeling in NLP, which also introduces some potential security vulnerabilities because attackers can manipulate the training process and data source. In this case, backdoor attacks can induce the model to exhibit expected behaviors through specific triggers and have little inferior influence on primitive tasks. Hence, it could have dire consequences, especially considering that the backdoor attack surfaces are broad.   However, there is still no systematic and comprehensive review to reflect the security challenges, attacker's capabilities, and purposes according to the attack surface. Moreover, there is a shortage of analysis and comparison of the diverse emerging backdoor countermeasures in this context. In this paper, we conduct a timely review of backdoor attacks and countermeasures to sound the red alarm for the NLP security community. According to the affected stage of the machine learning pipeline, the attack surfaces are recognized to be wide and then formalized into three categorizations: attacking pre-trained model with fine-tuning (APMF) or parameter-efficient tuning (APMP), and attacking final model with training (AFMT). Thus, attacks under each categorization are combed. The countermeasures are categorized into two general classes: sample inspection and model inspection. Overall, the research on the defense side is far behind the attack side, and there is no single defense that can prevent all types of backdoor attacks. An attacker can intelligently bypass existing defenses with a more invisible attack. Drawing the insights from the systematic review, we also present crucial areas for future research on the backdoor, such as empirical security evaluations on large language models, and in particular, more efficient and practical countermeasures are solicited.

摘要: 应用第三方数据和模型已经成为NLP中语言建模的新范式，这也带来了一些潜在的安全漏洞，因为攻击者可以操纵训练过程和数据源。在这种情况下，后门攻击可以通过特定的触发器诱导模型表现出预期的行为，并且对原始任务几乎没有不良影响。因此，这可能会产生可怕的后果，特别是考虑到后门攻击的范围很广。然而，仍然没有系统和全面的审查来反映安全挑战，攻击者的能力，以及根据攻击面的目的。此外，缺乏对在这方面出现的各种后门对策的分析和比较。在本文中，我们及时回顾了后门攻击和应对措施，为NLP安全界敲响了红色警报。根据机器学习流水线的受影响阶段，识别出攻击面较广，并将其形式化为三类：精调攻击预训练模型(APMF)或参数高效调整攻击(APMP)和训练攻击最终模型(AFMT)。因此，对每个分类下的攻击进行了梳理。反制措施一般分为两大类：抽样检查和模型检查。总体而言，防御端的研究远远落后于攻击端，没有单一的防御可以防范所有类型的后门攻击。攻击者可以通过更隐形的攻击智能地绕过现有的防御系统。从系统回顾中获得的见解，我们还提出了未来后门研究的关键领域，如对大型语言模型的经验安全评估，特别是寻求更有效和更实用的对策。



## **4. Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models**

沙子中的水印：生成模型不可能有强水印 cs.LG

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2311.04378v1) [paper-pdf](http://arxiv.org/pdf/2311.04378v1)

**Authors**: Hanlin Zhang, Benjamin L. Edelman, Danilo Francati, Daniele Venturi, Giuseppe Ateniese, Boaz Barak

**Abstract**: Watermarking generative models consists of planting a statistical signal (watermark) in a model's output so that it can be later verified that the output was generated by the given model. A strong watermarking scheme satisfies the property that a computationally bounded attacker cannot erase the watermark without causing significant quality degradation. In this paper, we study the (im)possibility of strong watermarking schemes. We prove that, under well-specified and natural assumptions, strong watermarking is impossible to achieve. This holds even in the private detection algorithm setting, where the watermark insertion and detection algorithms share a secret key, unknown to the attacker. To prove this result, we introduce a generic efficient watermark attack; the attacker is not required to know the private key of the scheme or even which scheme is used. Our attack is based on two assumptions: (1) The attacker has access to a "quality oracle" that can evaluate whether a candidate output is a high-quality response to a prompt, and (2) The attacker has access to a "perturbation oracle" which can modify an output with a nontrivial probability of maintaining quality, and which induces an efficiently mixing random walk on high-quality outputs. We argue that both assumptions can be satisfied in practice by an attacker with weaker computational capabilities than the watermarked model itself, to which the attacker has only black-box access. Furthermore, our assumptions will likely only be easier to satisfy over time as models grow in capabilities and modalities. We demonstrate the feasibility of our attack by instantiating it to attack three existing watermarking schemes for large language models: Kirchenbauer et al. (2023), Kuditipudi et al. (2023), and Zhao et al. (2023). The same attack successfully removes the watermarks planted by all three schemes, with only minor quality degradation.

摘要: 水印生成模型包括在模型的输出中植入统计信号(水印)，以便稍后可以验证输出是由给定模型生成的。强水印方案满足这样的性质，即计算受限的攻击者不可能在不引起显著质量降级的情况下删除水印。本文研究了强水印方案的(Im)可能性。我们证明了在明确和自然的假设下，强水印是不可能实现的。即使在私有检测算法设置中也是如此，其中水印插入和检测算法共享攻击者未知的秘密密钥。为了证明这一结果，我们引入了一种通用的高效水印攻击；攻击者不需要知道方案的私钥，甚至不需要知道使用了哪个方案。我们的攻击基于两个假设：(1)攻击者可以访问可以评估候选输出是否是对提示的高质量响应的“质量预言”，以及(2)攻击者可以访问“扰动预言”，它可以以保持质量的非平凡概率修改输出，并导致对高质量输出的有效混合随机游走。我们认为，在实践中，这两个假设都可以由计算能力弱于水印模型本身的攻击者满足，因为攻击者只能访问黑盒。此外，随着模型在功能和模式方面的发展，随着时间的推移，我们的假设可能只会更容易满足。我们通过将其实例化来攻击三个现有的用于大型语言模型的水印方案来证明该攻击的可行性：Kirchenbauer等人。(2023)，Kuditipudi等人。(2023)，和赵等人。(2023年)。同样的攻击成功地删除了所有三个方案植入的水印，只有很小的质量下降。



## **5. Unveiling Safety Vulnerabilities of Large Language Models**

揭开大型语言模型的安全漏洞 cs.CL

To be published in GEM workshop. Conference on Empirical Methods in  Natural Language Processing (EMNLP). 2023

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2311.04124v1) [paper-pdf](http://arxiv.org/pdf/2311.04124v1)

**Authors**: George Kour, Marcel Zalmanovici, Naama Zwerdling, Esther Goldbraich, Ora Nova Fandina, Ateret Anaby-Tavor, Orna Raz, Eitan Farchi

**Abstract**: As large language models become more prevalent, their possible harmful or inappropriate responses are a cause for concern. This paper introduces a unique dataset containing adversarial examples in the form of questions, which we call AttaQ, designed to provoke such harmful or inappropriate responses. We assess the efficacy of our dataset by analyzing the vulnerabilities of various models when subjected to it. Additionally, we introduce a novel automatic approach for identifying and naming vulnerable semantic regions - input semantic areas for which the model is likely to produce harmful outputs. This is achieved through the application of specialized clustering techniques that consider both the semantic similarity of the input attacks and the harmfulness of the model's responses. Automatically identifying vulnerable semantic regions enhances the evaluation of model weaknesses, facilitating targeted improvements to its safety mechanisms and overall reliability.

摘要: 随着大型语言模型变得越来越普遍，它们可能带来的有害或不恰当的反应令人担忧。本文介绍了一种独特的数据集，它以问题的形式包含了对抗性的例子，我们称之为Attaq，旨在引起这种有害或不适当的反应。我们通过分析各种模型在受到其影响时的漏洞来评估我们的数据集的有效性。此外，我们引入了一种新的自动方法来识别和命名易受攻击的语义区--模型可能产生有害输出的输入语义区。这是通过应用专门的集群技术来实现的，该技术同时考虑了输入攻击的语义相似性和模型响应的危害性。自动识别易受攻击的语义区域增强了对模型弱点的评估，促进了对其安全机制和总体可靠性的有针对性的改进。



## **6. Detecting Language Model Attacks with Perplexity**

基于困惑的语言模型攻击检测 cs.CL

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2308.14132v3) [paper-pdf](http://arxiv.org/pdf/2308.14132v3)

**Authors**: Gabriel Alon, Michael Kamfonas

**Abstract**: A novel hack involving Large Language Models (LLMs) has emerged, exploiting adversarial suffixes to deceive models into generating perilous responses. Such jailbreaks can trick LLMs into providing intricate instructions to a malicious user for creating explosives, orchestrating a bank heist, or facilitating the creation of offensive content. By evaluating the perplexity of queries with adversarial suffixes using an open-source LLM (GPT-2), we found that they have exceedingly high perplexity values. As we explored a broad range of regular (non-adversarial) prompt varieties, we concluded that false positives are a significant challenge for plain perplexity filtering. A Light-GBM trained on perplexity and token length resolved the false positives and correctly detected most adversarial attacks in the test set.

摘要: 出现了一种涉及大型语言模型(LLM)的新黑客攻击，利用敌意后缀欺骗模型生成危险的响应。此类越狱可以诱使LLMS向恶意用户提供复杂的指令，以制造爆炸物、策划银行抢劫或为创建攻击性内容提供便利。通过使用开放源代码的LLM(GPT-2)对带有敌意后缀的查询的困惑度进行评估，我们发现它们具有极高的困惑度值。随着我们探索了广泛的常规(非对抗性)提示类型，我们得出结论，假阳性对于普通困惑过滤来说是一个重大挑战。一种针对困惑和令牌长度的Light-GBM解决了假阳性问题，并正确地检测到了测试集中的大多数对抗性攻击。



## **7. Competence-Based Analysis of Language Models**

基于能力的语言模型分析 cs.CL

**SubmitDate**: 2023-11-07    [abs](http://arxiv.org/abs/2303.00333v3) [paper-pdf](http://arxiv.org/pdf/2303.00333v3)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent success of large, pretrained neural language models (LLMs) on a variety of prompting tasks, these models can be alarmingly brittle to small changes in inputs or application contexts. To better understand such behavior and motivate the design of more robust LLMs, we provide a causal formulation of linguistic competence in the context of LLMs and propose a general framework to study and measure LLM competence. Our framework, CALM (Competence-based Analysis of Language Models), establishes the first quantitative measure of LLM competence, which we study by damaging models' internal representations of various linguistic properties in the course of performing various tasks using causal probing and evaluating models' alignment under these interventions with a given causal model. We also develop a novel approach for performing causal probing interventions using gradient-based adversarial attacks, which can target a broader range of properties and representations than existing techniques. We carry out a case study of CALM using these interventions to analyze BERT and RoBERTa's competence across a variety of lexical inference tasks, showing that the CALM framework and competence metric can be valuable tools for explaining and predicting their behavior across these tasks.

摘要: 尽管大型的、预先训练的神经语言模型(LLM)最近在各种提示任务上取得了成功，但这些模型对于输入或应用环境的微小变化可能会非常脆弱。为了更好地理解这种行为，并激励设计更稳健的LLM，我们在LLMS的背景下提出了语言能力的因果表述，并提出了一个研究和测量LLM能力的一般框架。我们的基于能力的语言模型分析框架建立了第一个LLM能力的定量测量，我们通过破坏模型在执行各种任务的过程中对各种语言属性的内部表征进行研究，并评估模型在这些干预措施下与给定的因果模型的一致性。我们还开发了一种新的方法来使用基于梯度的对抗性攻击来执行因果探测干预，该方法可以针对比现有技术更广泛的属性和表示。我们使用这些干预措施对CAMLE进行了个案研究，分析了Bert和Roberta在各种词汇推理任务上的能力，结果表明，CAMLE框架和能力度量可以成为解释和预测他们在这些任务中的行为的有价值的工具。



## **8. Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation**

通过人物角色调整实现语言模型的可扩展和可传输黑盒越狱 cs.CL

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.03348v1) [paper-pdf](http://arxiv.org/pdf/2311.03348v1)

**Authors**: Rusheb Shah, Quentin Feuillade--Montixi, Soroush Pour, Arush Tagade, Stephen Casper, Javier Rando

**Abstract**: Despite efforts to align large language models to produce harmless responses, they are still vulnerable to jailbreak prompts that elicit unrestricted behaviour. In this work, we investigate persona modulation as a black-box jailbreaking method to steer a target model to take on personalities that are willing to comply with harmful instructions. Rather than manually crafting prompts for each persona, we automate the generation of jailbreaks using a language model assistant. We demonstrate a range of harmful completions made possible by persona modulation, including detailed instructions for synthesising methamphetamine, building a bomb, and laundering money. These automated attacks achieve a harmful completion rate of 42.5% in GPT-4, which is 185 times larger than before modulation (0.23%). These prompts also transfer to Claude 2 and Vicuna with harmful completion rates of 61.0% and 35.9%, respectively. Our work reveals yet another vulnerability in commercial large language models and highlights the need for more comprehensive safeguards.

摘要: 尽管努力调整大型语言模型以产生无害的响应，但它们仍然容易受到引发不受限制行为的越狱提示的影响。在这项工作中，我们调查的人物角色调制作为一个黑盒子越狱方法，引导目标模型采取的个性，愿意遵守有害的指令。我们使用语言模型助手自动生成越狱，而不是手动为每个角色制作提示。我们展示了一系列有害的完成可能通过人格调制，包括合成甲基苯丙胺，制造炸弹和洗钱的详细说明。这些自动化攻击在GPT-4中实现了42.5%的有害完成率，比调制前（0.23%）大185倍。这些提示也转移到克劳德2和Vicuna，分别有61.0%和35.9%的有害完成率。我们的工作揭示了商业大型语言模型中的另一个漏洞，并强调了需要更全面的保护措施。



## **9. Vulnerabilities in AI Code Generators: Exploring Targeted Data Poisoning Attacks**

AI代码生成器中的漏洞：探索有针对性的数据中毒攻击 cs.CR

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2308.04451v2) [paper-pdf](http://arxiv.org/pdf/2308.04451v2)

**Authors**: Domenico Cotroneo, Cristina Improta, Pietro Liguori, Roberto Natella

**Abstract**: AI-based code generators have become pivotal in assisting developers in writing software starting from natural language (NL). However, they are trained on large amounts of data, often collected from unsanitized online sources (e.g., GitHub, HuggingFace). As a consequence, AI models become an easy target for data poisoning, i.e., an attack that injects malicious samples into the training data to generate vulnerable code. To address this threat, we investigate the security of AI code generators by devising a targeted data poisoning strategy. We poison the training data by injecting increasing amounts of code containing security vulnerabilities and assess the attack's success on different state-of-the-art models for code generation. Our study shows that AI code generators are vulnerable to even a small amount of poison. Notably, the attack success strongly depends on the model architecture and poisoning rate, whereas it is not influenced by the type of vulnerabilities. Moreover, since the attack does not impact the correctness of code generated by pre-trained models, it is hard to detect. Lastly, our work offers practical insights into understanding and potentially mitigating this threat.

摘要: 基于人工智能的代码生成器已经成为帮助开发人员从自然语言(NL)开始编写软件的关键。然而，他们接受了大量数据的培训，这些数据通常是从未经清理的在线来源(如GitHub、HuggingFace)收集的。因此，人工智能模型很容易成为数据中毒的目标，即向训练数据中注入恶意样本以生成易受攻击的代码的攻击。为了应对这一威胁，我们通过设计一种有针对性的数据中毒策略来调查AI代码生成器的安全性。我们通过注入越来越多的包含安全漏洞的代码来毒化训练数据，并在不同的最先进的代码生成模型上评估攻击的成功。我们的研究表明，AI代码生成器即使是少量的毒药也很容易受到攻击。值得注意的是，攻击的成功很大程度上取决于模型体系结构和投毒率，而不受漏洞类型的影响。此外，由于攻击不会影响预先训练的模型生成的代码的正确性，因此很难检测到。最后，我们的工作为理解和潜在地缓解这一威胁提供了实际的见解。



## **10. Can LLMs Follow Simple Rules?**

低收入国家能遵循简单的规则吗？ cs.AI

Project website: https://eecs.berkeley.edu/~normanmu/llm_rules

**SubmitDate**: 2023-11-06    [abs](http://arxiv.org/abs/2311.04235v1) [paper-pdf](http://arxiv.org/pdf/2311.04235v1)

**Authors**: Norman Mu, Sarah Chen, Zifan Wang, Sizhe Chen, David Karamardian, Lulwa Aljeraisy, Dan Hendrycks, David Wagner

**Abstract**: As Large Language Models (LLMs) are deployed with increasing real-world responsibilities, it is important to be able to specify and constrain the behavior of these systems in a reliable manner. Model developers may wish to set explicit rules for the model, such as "do not generate abusive content", but these may be circumvented by jailbreaking techniques. Evaluating how well LLMs follow developer-provided rules in the face of adversarial inputs typically requires manual review, which slows down monitoring and methods development. To address this issue, we propose Rule-following Language Evaluation Scenarios (RuLES), a programmatic framework for measuring rule-following ability in LLMs. RuLES consists of 15 simple text scenarios in which the model is instructed to obey a set of rules in natural language while interacting with the human user. Each scenario has a concise evaluation program to determine whether the model has broken any rules in a conversation. Through manual exploration of model behavior in our scenarios, we identify 6 categories of attack strategies and collect two suites of test cases: one consisting of unique conversations from manual testing and one that systematically implements strategies from the 6 categories. Across various popular proprietary and open models such as GPT-4 and Llama 2, we find that all models are susceptible to a wide variety of adversarial hand-crafted user inputs, though GPT-4 is the best-performing model. Additionally, we evaluate open models under gradient-based attacks and find significant vulnerabilities. We propose RuLES as a challenging new setting for research into exploring and defending against both manual and automatic attacks on LLMs.

摘要: 随着大型语言模型(LLM)的部署承担着越来越多的现实责任，能够以可靠的方式指定和约束这些系统的行为是很重要的。模型开发人员可能希望为模型设置明确的规则，例如“不要生成滥用内容”，但可以通过越狱技术绕过这些规则。评估LLM在面对敌对输入时遵循开发人员提供的规则的情况通常需要手动审查，这会减缓监测和方法开发的速度。为了解决这一问题，我们提出了规则遵循语言评估场景(Rules)，这是一个衡量LLMS中规则遵循能力的程序性框架。规则由15个简单的文本场景组成，在这些场景中，模型被指示在与人类用户交互时遵守一组自然语言规则。每个场景都有一个简明的评估程序，以确定模型是否在对话中违反了任何规则。通过手动探索场景中的模型行为，我们确定了6类攻击策略，并收集了两套测试用例：一套由手动测试中的独特对话组成，另一套系统地实现了这6类策略。纵观各种流行的专有和开放机型，如GPT-4和Llama 2，我们发现所有机型都容易受到各种对抗性手工用户输入的影响，尽管GPT-4是性能最好的机型。此外，我们在基于梯度的攻击下对开放模型进行了评估，发现了显著的漏洞。我们提出规则作为一个具有挑战性的新环境，用于研究探索和防御对LLMS的手动和自动攻击。



## **11. The Alignment Problem in Context**

上下文中的对齐问题 cs.LG

**SubmitDate**: 2023-11-03    [abs](http://arxiv.org/abs/2311.02147v1) [paper-pdf](http://arxiv.org/pdf/2311.02147v1)

**Authors**: Raphaël Millière

**Abstract**: A core challenge in the development of increasingly capable AI systems is to make them safe and reliable by ensuring their behaviour is consistent with human values. This challenge, known as the alignment problem, does not merely apply to hypothetical future AI systems that may pose catastrophic risks; it already applies to current systems, such as large language models, whose potential for harm is rapidly increasing. In this paper, I assess whether we are on track to solve the alignment problem for large language models, and what that means for the safety of future AI systems. I argue that existing strategies for alignment are insufficient, because large language models remain vulnerable to adversarial attacks that can reliably elicit unsafe behaviour. I offer an explanation of this lingering vulnerability on which it is not simply a contingent limitation of current language models, but has deep technical ties to a crucial aspect of what makes these models useful and versatile in the first place -- namely, their remarkable aptitude to learn "in context" directly from user instructions. It follows that the alignment problem is not only unsolved for current AI systems, but may be intrinsically difficult to solve without severely undermining their capabilities. Furthermore, this assessment raises concerns about the prospect of ensuring the safety of future and more capable AI systems.

摘要: 开发能力越来越强的人工智能系统的一个核心挑战是，通过确保它们的行为符合人类价值观，使它们变得安全可靠。这一挑战被称为对齐问题，不仅适用于可能构成灾难性风险的假设的未来人工智能系统；它已经适用于当前的系统，例如大型语言模型，其危害的可能性正在迅速增加。在这篇文章中，我评估了我们是否正在解决大型语言模型的对齐问题，以及这对未来人工智能系统的安全意味着什么。我认为，现有的对齐策略是不够的，因为大型语言模型仍然容易受到敌意攻击，这些攻击可能会可靠地引发不安全的行为。我解释了这个挥之不去的漏洞，在这个漏洞上，它不仅仅是当前语言模型的偶然限制，而且与使这些模型首先有用和多功能的一个关键方面有很深的技术联系--即它们直接从用户指令中“在上下文中”学习的非凡能力。由此得出的结论是，对齐问题不仅对当前的人工智能系统没有解决，而且可能在不严重削弱其能力的情况下从本质上很难解决。此外，这项评估引发了人们对确保未来更有能力的人工智能系统安全的前景的担忧。



## **12. Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game**

张量信任：来自网络游戏的可解释提示注入攻击 cs.LG

**SubmitDate**: 2023-11-02    [abs](http://arxiv.org/abs/2311.01011v1) [paper-pdf](http://arxiv.org/pdf/2311.01011v1)

**Authors**: Sam Toyer, Olivia Watkins, Ethan Adrian Mendes, Justin Svegliato, Luke Bailey, Tiffany Wang, Isaac Ong, Karim Elmaaroufi, Pieter Abbeel, Trevor Darrell, Alan Ritter, Stuart Russell

**Abstract**: While Large Language Models (LLMs) are increasingly being used in real-world applications, they remain vulnerable to prompt injection attacks: malicious third party prompts that subvert the intent of the system designer. To help researchers study this problem, we present a dataset of over 126,000 prompt injection attacks and 46,000 prompt-based "defenses" against prompt injection, all created by players of an online game called Tensor Trust. To the best of our knowledge, this is currently the largest dataset of human-generated adversarial examples for instruction-following LLMs. The attacks in our dataset have a lot of easily interpretable stucture, and shed light on the weaknesses of LLMs. We also use the dataset to create a benchmark for resistance to two types of prompt injection, which we refer to as prompt extraction and prompt hijacking. Our benchmark results show that many models are vulnerable to the attack strategies in the Tensor Trust dataset. Furthermore, we show that some attack strategies from the dataset generalize to deployed LLM-based applications, even though they have a very different set of constraints to the game. We release all data and source code at https://tensortrust.ai/paper

摘要: 虽然大型语言模型(LLM)越来越多地用于现实世界的应用程序，但它们仍然容易受到提示注入攻击：破坏系统设计人员意图的恶意第三方提示。为了帮助研究人员研究这个问题，我们提供了一个超过12.6万个即时注入攻击和4.6万个基于即时注入的“防御”的数据集，所有这些都是由一款名为“张量信任”的在线游戏的玩家创建的。据我们所知，这是目前最大的人类生成的遵循指令的LLM对抗性例子的数据集。我们的数据集中的攻击具有许多易于解释的结构，并揭示了LLMS的弱点。我们还使用数据集创建了对两种类型的即时注入的抵抗力基准，我们称之为即时提取和即时劫持。我们的基准测试结果表明，许多模型都容易受到张量信任数据集中的攻击策略的影响。此外，我们还表明，来自数据集的一些攻击策略适用于部署的基于LLM的应用程序，即使它们对游戏有非常不同的约束集。我们在https://tensortrust.ai/paper上发布所有数据和源代码



## **13. Robust Safety Classifier for Large Language Models: Adversarial Prompt Shield**

用于大型语言模型的稳健安全分类器：对抗性提示盾 cs.CL

11 pages, 2 figures

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2311.00172v1) [paper-pdf](http://arxiv.org/pdf/2311.00172v1)

**Authors**: Jinhwa Kim, Ali Derakhshan, Ian G. Harris

**Abstract**: Large Language Models' safety remains a critical concern due to their vulnerability to adversarial attacks, which can prompt these systems to produce harmful responses. In the heart of these systems lies a safety classifier, a computational model trained to discern and mitigate potentially harmful, offensive, or unethical outputs. However, contemporary safety classifiers, despite their potential, often fail when exposed to inputs infused with adversarial noise. In response, our study introduces the Adversarial Prompt Shield (APS), a lightweight model that excels in detection accuracy and demonstrates resilience against adversarial prompts. Additionally, we propose novel strategies for autonomously generating adversarial training datasets, named Bot Adversarial Noisy Dialogue (BAND) datasets. These datasets are designed to fortify the safety classifier's robustness, and we investigate the consequences of incorporating adversarial examples into the training process. Through evaluations involving Large Language Models, we demonstrate that our classifier has the potential to decrease the attack success rate resulting from adversarial attacks by up to 60%. This advancement paves the way for the next generation of more reliable and resilient conversational agents.

摘要: 大型语言模型的安全性仍然是一个关键问题，因为它们容易受到对抗性攻击，这可能会促使这些系统产生有害的响应。这些系统的核心是一个安全分类器，这是一个经过训练的计算模型，可以识别和减轻潜在的有害、攻击性或不道德的输出。然而，当代的安全分类器，尽管它们的潜力，往往失败时，暴露于输入注入对抗性噪声。作为回应，我们的研究引入了对抗性提示盾（APS），这是一种轻量级模型，在检测准确性方面表现出色，并展示了对抗性提示的弹性。此外，我们还提出了自主生成对抗训练数据集的新策略，称为Bot Adversarial Noisy Dialogue（BAND）数据集。这些数据集旨在加强安全分类器的鲁棒性，我们研究了将对抗性示例纳入训练过程的后果。通过涉及大型语言模型的评估，我们证明了我们的分类器有可能将对抗性攻击的攻击成功率降低高达60%。这一进步为下一代更可靠和更有弹性的会话代理铺平了道路。



## **14. LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B**

Lora微调有效地取消了Llama 2-Chat 70B中的安全培训 cs.LG

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.20624v1) [paper-pdf](http://arxiv.org/pdf/2310.20624v1)

**Authors**: Simon Lermen, Charlie Rogers-Smith, Jeffrey Ladish

**Abstract**: AI developers often apply safety alignment procedures to prevent the misuse of their AI systems. For example, before Meta released Llama 2-Chat, a collection of instruction fine-tuned large language models, they invested heavily in safety training, incorporating extensive red-teaming and reinforcement learning from human feedback. However, it remains unclear how well safety training guards against model misuse when attackers have access to model weights. We explore the robustness of safety training in language models by subversively fine-tuning the public weights of Llama 2-Chat. We employ low-rank adaptation (LoRA) as an efficient fine-tuning method. With a budget of less than $200 per model and using only one GPU, we successfully undo the safety training of Llama 2-Chat models of sizes 7B, 13B, and 70B. Specifically, our fine-tuning technique significantly reduces the rate at which the model refuses to follow harmful instructions. We achieve a refusal rate below 1% for our 70B Llama 2-Chat model on two refusal benchmarks. Our fine-tuning method retains general performance, which we validate by comparing our fine-tuned models against Llama 2-Chat across two benchmarks. Additionally, we present a selection of harmful outputs produced by our models. While there is considerable uncertainty about the scope of risks from current models, it is likely that future models will have significantly more dangerous capabilities, including the ability to hack into critical infrastructure, create dangerous bio-weapons, or autonomously replicate and adapt to new environments. We show that subversive fine-tuning is practical and effective, and hence argue that evaluating risks from fine-tuning should be a core part of risk assessments for releasing model weights.

摘要: 人工智能开发人员经常应用安全对齐程序，以防止他们的人工智能系统被滥用。例如，在Meta发布Llama 2-Chat之前，他们在安全培训方面投入了大量资金，纳入了广泛的红色团队和来自人类反馈的强化学习。然而，目前尚不清楚，当袭击者可以接触到模型重量时，安全培训在多大程度上防止了模型的滥用。我们通过颠覆性地微调Llama 2-Chat的公开权重来探索语言模型中安全训练的稳健性。我们使用低阶自适应(LORA)作为一种有效的微调方法。在每个型号的预算不到200美元的情况下，仅使用一个GPU，我们成功地取消了7B、13B和70B尺寸的Llama 2-Chat型号的安全培训。具体地说，我们的微调技术显著降低了模型拒绝遵循有害指令的速度。在两个拒绝基准上，我们的70B Llama 2-Chat模型的拒绝率低于1%。我们的微调方法保持了总体性能，我们通过将我们的微调模型与两个基准测试中的Llama 2-chat进行比较来验证这一点。此外，我们还提供了由我们的模型产生的有害输出的精选。尽管当前模型的风险范围存在相当大的不确定性，但未来的模型很可能具有更危险的能力，包括侵入关键基础设施、制造危险生物武器或自主复制和适应新环境的能力。我们证明了颠覆性微调是实用和有效的，并因此认为评估微调的风险应该是释放模型权重的风险评估的核心部分。



## **15. On Extracting Specialized Code Abilities from Large Language Models: A Feasibility Study**

从大型语言模型中提取专业代码能力的可行性研究 cs.SE

13 pages

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2303.03012v4) [paper-pdf](http://arxiv.org/pdf/2303.03012v4)

**Authors**: Zongjie Li, Chaozheng Wang, Pingchuan Ma, Chaowei Liu, Shuai Wang, Daoyuan Wu, Cuiyun Gao, Yang Liu

**Abstract**: Recent advances in large language models (LLMs) significantly boost their usage in software engineering. However, training a well-performing LLM demands a substantial workforce for data collection and annotation. Moreover, training datasets may be proprietary or partially open, and the process often requires a costly GPU cluster. The intellectual property value of commercial LLMs makes them attractive targets for imitation attacks, but creating an imitation model with comparable parameters still incurs high costs. This motivates us to explore a practical and novel direction: slicing commercial black-box LLMs using medium-sized backbone models. In this paper, we explore the feasibility of launching imitation attacks on LLMs to extract their specialized code abilities, such as"code synthesis" and "code translation." We systematically investigate the effectiveness of launching code ability extraction attacks under different code-related tasks with multiple query schemes, including zero-shot, in-context, and Chain-of-Thought. We also design response checks to refine the outputs, leading to an effective imitation training process. Our results show promising outcomes, demonstrating that with a reasonable number of queries, attackers can train a medium-sized backbone model to replicate specialized code behaviors similar to the target LLMs. We summarize our findings and insights to help researchers better understand the threats posed by imitation attacks, including revealing a practical attack surface for generating adversarial code examples against LLMs.

摘要: 大型语言模型(LLM)的最新进展极大地促进了它们在软件工程中的使用。然而，培训一个表现良好的LLM需要大量的数据收集和注释工作人员。此外，训练数据集可能是专有的或部分开放的，该过程通常需要昂贵的GPU集群。商业LLM的知识产权价值使其成为模仿攻击的有吸引力的目标，但创建具有可比参数的模仿模型仍然会招致高昂的成本。这促使我们探索一个实用而新颖的方向：使用中型主干模型对商业黑盒LLM进行切片。在本文中，我们探索了对LLM发动模仿攻击的可行性，以提取其专门的代码能力，如“代码合成”和“代码翻译”。我们系统地研究了在不同的代码相关任务下，使用包括零命中、上下文中和思想链在内的多种查询方案来发起代码能力提取攻击的有效性。我们还设计了响应检查来优化输出，从而实现有效的模仿培训过程。我们的结果显示了令人振奋的结果，表明通过合理数量的查询，攻击者可以训练一个中等大小的主干模型来复制类似于目标LLM的特定代码行为。我们总结了我们的发现和见解，以帮助研究人员更好地理解模仿攻击所构成的威胁，包括揭示一个实用的攻击面，用于生成针对LLM的敌意代码示例。



## **16. TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models**

TrojLLM：一种针对大型语言模型的黑盒木马提示攻击 cs.CR

Accepted by NeurIPS'23

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2306.06815v3) [paper-pdf](http://arxiv.org/pdf/2306.06815v3)

**Authors**: Jiaqi Xue, Mengxin Zheng, Ting Hua, Yilin Shen, Yepeng Liu, Ladislau Boloni, Qian Lou

**Abstract**: Large Language Models (LLMs) are progressively being utilized as machine learning services and interface tools for various applications. However, the security implications of LLMs, particularly in relation to adversarial and Trojan attacks, remain insufficiently examined. In this paper, we propose TrojLLM, an automatic and black-box framework to effectively generate universal and stealthy triggers. When these triggers are incorporated into the input data, the LLMs' outputs can be maliciously manipulated. Moreover, the framework also supports embedding Trojans within discrete prompts, enhancing the overall effectiveness and precision of the triggers' attacks. Specifically, we propose a trigger discovery algorithm for generating universal triggers for various inputs by querying victim LLM-based APIs using few-shot data samples. Furthermore, we introduce a novel progressive Trojan poisoning algorithm designed to generate poisoned prompts that retain efficacy and transferability across a diverse range of models. Our experiments and results demonstrate TrojLLM's capacity to effectively insert Trojans into text prompts in real-world black-box LLM APIs including GPT-3.5 and GPT-4, while maintaining exceptional performance on clean test sets. Our work sheds light on the potential security risks in current models and offers a potential defensive approach. The source code of TrojLLM is available at https://github.com/UCF-ML-Research/TrojLLM.

摘要: 大型语言模型(LLM)正逐渐被用作各种应用的机器学习服务和接口工具。然而，LLMS的安全影响，特别是与对抗性攻击和特洛伊木马攻击有关的影响，仍然没有得到充分的研究。在本文中，我们提出了一个自动黑盒框架TrojLLM，它可以有效地生成通用的、隐蔽的触发器。当这些触发器被合并到输入数据中时，LLMS的输出可能被恶意操纵。此外，该框架还支持在离散提示中嵌入特洛伊木马，增强了触发器攻击的整体有效性和精确度。具体地说，我们提出了一种触发器发现算法，通过使用少量数据样本查询受害者基于LLM的API来为各种输入生成通用触发器。此外，我们引入了一种新的渐进式特洛伊木马中毒算法，旨在生成中毒提示，从而在不同的模型中保持有效性和可转移性。我们的实验和结果表明，TrojLLM能够在包括GPT-3.5和GPT-4在内的真实黑盒LLMAPI中有效地将特洛伊木马程序插入到文本提示中，同时在干净的测试集上保持出色的性能。我们的工作揭示了当前模型中的潜在安全风险，并提供了一种潜在的防御方法。TrojLLm的源代码可在https://github.com/UCF-ML-Research/TrojLLM.上找到



## **17. Adversarial Attacks and Defenses in Large Language Models: Old and New Threats**

大型语言模型中的对抗性攻击和防御：旧威胁和新威胁 cs.AI

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2310.19737v1) [paper-pdf](http://arxiv.org/pdf/2310.19737v1)

**Authors**: Leo Schwinn, David Dobre, Stephan Günnemann, Gauthier Gidel

**Abstract**: Over the past decade, there has been extensive research aimed at enhancing the robustness of neural networks, yet this problem remains vastly unsolved. Here, one major impediment has been the overestimation of the robustness of new defense approaches due to faulty defense evaluations. Flawed robustness evaluations necessitate rectifications in subsequent works, dangerously slowing down the research and providing a false sense of security. In this context, we will face substantial challenges associated with an impending adversarial arms race in natural language processing, specifically with closed-source Large Language Models (LLMs), such as ChatGPT, Google Bard, or Anthropic's Claude. We provide a first set of prerequisites to improve the robustness assessment of new approaches and reduce the amount of faulty evaluations. Additionally, we identify embedding space attacks on LLMs as another viable threat model for the purposes of generating malicious content in open-sourced models. Finally, we demonstrate on a recently proposed defense that, without LLM-specific best practices in place, it is easy to overestimate the robustness of a new approach.

摘要: 在过去的十年里，已经有大量的研究旨在增强神经网络的健壮性，但这个问题仍然远远没有解决。在这里，一个主要的障碍是由于错误的防御评估而高估了新的防御方法的稳健性。有缺陷的稳健性评估需要在后续工作中进行更正，这会危险地减缓研究速度，并提供一种错误的安全感。在这种情况下，我们将面临与自然语言处理领域即将到来的对抗性军备竞赛相关的重大挑战，特别是与封闭源代码的大型语言模型(LLM)相关的挑战，如ChatGPT、Google Bard或Anthropic的Claude。我们提供了第一组先决条件来改进新方法的稳健性评估，并减少错误评估的数量。此外，我们将在LLM上嵌入空间攻击作为另一种可行的威胁模型，目的是在开源模型中生成恶意内容。最后，我们在最近提出的一项防御中演示了，如果没有特定于LLM的最佳实践，很容易高估新方法的健壮性。



## **18. From Chatbots to PhishBots? -- Preventing Phishing scams created using ChatGPT, Google Bard and Claude**

从聊天机器人到网络钓鱼机器人？--防止使用ChatGPT、Google Bard和Claude创建的网络钓鱼诈骗 cs.CR

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.19181v1) [paper-pdf](http://arxiv.org/pdf/2310.19181v1)

**Authors**: Sayak Saha Roy, Poojitha Thota, Krishna Vamsi Naragam, Shirin Nilizadeh

**Abstract**: The advanced capabilities of Large Language Models (LLMs) have made them invaluable across various applications, from conversational agents and content creation to data analysis, research, and innovation. However, their effectiveness and accessibility also render them susceptible to abuse for generating malicious content, including phishing attacks. This study explores the potential of using four popular commercially available LLMs - ChatGPT (GPT 3.5 Turbo), GPT 4, Claude and Bard to generate functional phishing attacks using a series of malicious prompts. We discover that these LLMs can generate both phishing emails and websites that can convincingly imitate well-known brands, and also deploy a range of evasive tactics for the latter to elude detection mechanisms employed by anti-phishing systems. Notably, these attacks can be generated using unmodified, or "vanilla," versions of these LLMs, without requiring any prior adversarial exploits such as jailbreaking. As a countermeasure, we build a BERT based automated detection tool that can be used for the early detection of malicious prompts to prevent LLMs from generating phishing content attaining an accuracy of 97\% for phishing website prompts, and 94\% for phishing email prompts.

摘要: 大型语言模型（LLM）的高级功能使其在各种应用中具有不可估量的价值，从会话代理和内容创建到数据分析，研究和创新。然而，它们的有效性和可访问性也使它们容易被滥用以生成恶意内容，包括网络钓鱼攻击。这项研究探讨了使用四种流行的商用LLMs - ChatGPT（GPT 3.5 Turbo），GPT 4，Claude和Bard使用一系列恶意提示生成功能性网络钓鱼攻击的潜力。我们发现，这些LLM可以生成钓鱼电子邮件和网站，可以令人信服地模仿知名品牌，还部署了一系列规避策略，后者逃避反钓鱼系统采用的检测机制。值得注意的是，这些攻击可以使用这些LLM的未修改或“vanilla”版本生成，而不需要任何先前的对抗性攻击，例如越狱。作为对策，我们建立了一个基于BERT的自动检测工具，可以用于早期检测恶意提示，以防止LLM生成钓鱼内容，达到97%的准确性钓鱼网站提示，94%的钓鱼电子邮件提示。



## **19. Robustifying Language Models with Test-Time Adaptation**

基于测试时间自适应的模糊语言模型 cs.CL

8 Pages 2 Figures Submitted to ICLR Workshop

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.19177v1) [paper-pdf](http://arxiv.org/pdf/2310.19177v1)

**Authors**: Noah Thomas McDermott, Junfeng Yang, Chengzhi Mao

**Abstract**: Large-scale language models achieved state-of-the-art performance over a number of language tasks. However, they fail on adversarial language examples, which are sentences optimized to fool the language models but with similar semantic meanings for humans. While prior work focuses on making the language model robust at training time, retraining for robustness is often unrealistic for large-scale foundation models. Instead, we propose to make the language models robust at test time. By dynamically adapting the input sentence with predictions from masked words, we show that we can reverse many language adversarial attacks. Since our approach does not require any training, it works for novel tasks at test time and can adapt to novel adversarial corruptions. Visualizations and empirical results on two popular sentence classification datasets demonstrate that our method can repair adversarial language attacks over 65% o

摘要: 大规模语言模型在许多语言任务上实现了最先进的性能。然而，他们在对抗性语言例子上失败了，这些例子是为愚弄语言模型而优化的句子，但对人类来说具有相似的语义。虽然以前的工作重点是在训练时使语言模型具有健壮性，但对于大规模的基础模型来说，进行健壮性的再培训往往是不现实的。相反，我们建议在测试时使语言模型健壮。通过动态调整输入句子和掩蔽词的预测，我们证明了我们可以逆转许多语言对手攻击。由于我们的方法不需要任何训练，它在测试时适用于新的任务，并且可以适应新的对抗性腐败。在两个常用的句子分类数据集上的可视化和实验结果表明，该方法可以修复65%以上的敌意语言攻击



## **20. On the Exploitability of Instruction Tuning**

论指令调优的可开发性 cs.CR

NeurIPS 2023 camera-ready (21 pages, 10 figures)

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2306.17194v2) [paper-pdf](http://arxiv.org/pdf/2306.17194v2)

**Authors**: Manli Shu, Jiongxiao Wang, Chen Zhu, Jonas Geiping, Chaowei Xiao, Tom Goldstein

**Abstract**: Instruction tuning is an effective technique to align large language models (LLMs) with human intents. In this work, we investigate how an adversary can exploit instruction tuning by injecting specific instruction-following examples into the training data that intentionally changes the model's behavior. For example, an adversary can achieve content injection by injecting training examples that mention target content and eliciting such behavior from downstream models. To achieve this goal, we propose \textit{AutoPoison}, an automated data poisoning pipeline. It naturally and coherently incorporates versatile attack goals into poisoned data with the help of an oracle LLM. We showcase two example attacks: content injection and over-refusal attacks, each aiming to induce a specific exploitable behavior. We quantify and benchmark the strength and the stealthiness of our data poisoning scheme. Our results show that AutoPoison allows an adversary to change a model's behavior by poisoning only a small fraction of data while maintaining a high level of stealthiness in the poisoned examples. We hope our work sheds light on how data quality affects the behavior of instruction-tuned models and raises awareness of the importance of data quality for responsible deployments of LLMs. Code is available at \url{https://github.com/azshue/AutoPoison}.

摘要: 指令调优是使大型语言模型与人的意图保持一致的一种有效技术。在这项工作中，我们调查了对手如何通过向训练数据中注入特定的指令遵循示例来利用指令调整，从而故意改变模型的行为。例如，对手可以通过注入提到目标内容的训练示例并从下游模型中引出此类行为来实现内容注入。为了实现这一目标，我们提出了一种自动化的数据中毒管道--Texttit{AutoPoison}。在甲骨文LLM的帮助下，它自然而连贯地将各种攻击目标整合到有毒数据中。我们展示了两个示例攻击：内容注入攻击和过度拒绝攻击，每个攻击都旨在诱导特定的可利用行为。我们对我们的数据中毒方案的强度和隐蔽性进行量化和基准测试。我们的结果表明，AutoPoison允许对手通过只毒化一小部分数据来改变模型的行为，同时在中毒的示例中保持高级别的隐蔽性。我们希望我们的工作有助于阐明数据质量如何影响指令调优模型的行为，并提高人们对数据质量对于负责任的LLM部署的重要性的认识。代码位于\url{https://github.com/azshue/AutoPoison}.



## **21. Large Language Models Are Better Adversaries: Exploring Generative Clean-Label Backdoor Attacks Against Text Classifiers**

大型语言模型是更好的对手：探索针对文本分类器的生成性干净标签后门攻击 cs.LG

Accepted at EMNLP 2023 Findings

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2310.18603v1) [paper-pdf](http://arxiv.org/pdf/2310.18603v1)

**Authors**: Wencong You, Zayd Hammoudeh, Daniel Lowd

**Abstract**: Backdoor attacks manipulate model predictions by inserting innocuous triggers into training and test data. We focus on more realistic and more challenging clean-label attacks where the adversarial training examples are correctly labeled. Our attack, LLMBkd, leverages language models to automatically insert diverse style-based triggers into texts. We also propose a poison selection technique to improve the effectiveness of both LLMBkd as well as existing textual backdoor attacks. Lastly, we describe REACT, a baseline defense to mitigate backdoor attacks via antidote training examples. Our evaluations demonstrate LLMBkd's effectiveness and efficiency, where we consistently achieve high attack success rates across a wide range of styles with little effort and no model training.

摘要: 后门攻击通过在训练和测试数据中插入无害的触发器来操纵模型预测。我们专注于更现实、更具挑战性的干净标签攻击，其中对抗性训练示例被正确标记。我们的攻击LLMBkd利用语言模型自动将不同的基于样式的触发器插入到文本中。我们还提出了一种毒物选择技术来提高LLMBkd和现有文本后门攻击的有效性。最后，我们描述了Reaction，一种通过解毒剂训练示例来减少后门攻击的基线防御。我们的评估证明了LLMBkd的有效性和效率，在这种情况下，我们几乎不费力气，也没有模型训练，就能在各种风格中始终获得高攻击成功率。



## **22. ParaFuzz: An Interpretability-Driven Technique for Detecting Poisoned Samples in NLP**

ParaFuzz：一种可解释性驱动的NLP中毒样本检测技术 cs.CR

**SubmitDate**: 2023-10-27    [abs](http://arxiv.org/abs/2308.02122v2) [paper-pdf](http://arxiv.org/pdf/2308.02122v2)

**Authors**: Lu Yan, Zhuo Zhang, Guanhong Tao, Kaiyuan Zhang, Xuan Chen, Guangyu Shen, Xiangyu Zhang

**Abstract**: Backdoor attacks have emerged as a prominent threat to natural language processing (NLP) models, where the presence of specific triggers in the input can lead poisoned models to misclassify these inputs to predetermined target classes. Current detection mechanisms are limited by their inability to address more covert backdoor strategies, such as style-based attacks. In this work, we propose an innovative test-time poisoned sample detection framework that hinges on the interpretability of model predictions, grounded in the semantic meaning of inputs. We contend that triggers (e.g., infrequent words) are not supposed to fundamentally alter the underlying semantic meanings of poisoned samples as they want to stay stealthy. Based on this observation, we hypothesize that while the model's predictions for paraphrased clean samples should remain stable, predictions for poisoned samples should revert to their true labels upon the mutations applied to triggers during the paraphrasing process. We employ ChatGPT, a state-of-the-art large language model, as our paraphraser and formulate the trigger-removal task as a prompt engineering problem. We adopt fuzzing, a technique commonly used for unearthing software vulnerabilities, to discover optimal paraphrase prompts that can effectively eliminate triggers while concurrently maintaining input semantics. Experiments on 4 types of backdoor attacks, including the subtle style backdoors, and 4 distinct datasets demonstrate that our approach surpasses baseline methods, including STRIP, RAP, and ONION, in precision and recall.

摘要: 后门攻击已经成为自然语言处理(NLP)模型的一个突出威胁，在NLP模型中，输入中存在特定触发器可能会导致中毒模型将这些输入错误分类到预定的目标类别。当前的检测机制由于无法应对更隐蔽的后门策略而受到限制，例如基于样式的攻击。在这项工作中，我们提出了一个创新的测试时间中毒样本检测框架，该框架取决于模型预测的可解释性，基于输入的语义。我们认为，触发因素(例如，不常见的单词)不应该从根本上改变中毒样本的潜在语义，因为它们想要保持隐蔽性。基于这一观察，我们假设，虽然模型对释义干净样本的预测应该保持稳定，但对中毒样本的预测应该在释义过程中应用于触发器的突变后恢复到其真实标签。我们使用最先进的大型语言模型ChatGPT作为我们的释义，并将触发器移除任务描述为一个紧迫的工程问题。我们采用了模糊技术，这是一种常用的软件漏洞挖掘技术，可以发现最优的释义提示，可以有效地消除触发器，同时保持输入语义。在4种类型的后门攻击(包括微妙风格的后门攻击)和4个不同的数据集上的实验表明，我们的方法在准确率和召回率上都超过了基线方法，包括STRAP、RAP和洋葱。



## **23. MasterKey: Automated Jailbreak Across Multiple Large Language Model Chatbots**

MasterKey：跨多个大型语言模型聊天机器人的自动越狱 cs.CR

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2307.08715v2) [paper-pdf](http://arxiv.org/pdf/2307.08715v2)

**Authors**: Gelei Deng, Yi Liu, Yuekang Li, Kailong Wang, Ying Zhang, Zefeng Li, Haoyu Wang, Tianwei Zhang, Yang Liu

**Abstract**: Large Language Models (LLMs) have revolutionized Artificial Intelligence (AI) services due to their exceptional proficiency in understanding and generating human-like text. LLM chatbots, in particular, have seen widespread adoption, transforming human-machine interactions. However, these LLM chatbots are susceptible to "jailbreak" attacks, where malicious users manipulate prompts to elicit inappropriate or sensitive responses, contravening service policies. Despite existing attempts to mitigate such threats, our research reveals a substantial gap in our understanding of these vulnerabilities, largely due to the undisclosed defensive measures implemented by LLM service providers.   In this paper, we present Jailbreaker, a comprehensive framework that offers an in-depth understanding of jailbreak attacks and countermeasures. Our work makes a dual contribution. First, we propose an innovative methodology inspired by time-based SQL injection techniques to reverse-engineer the defensive strategies of prominent LLM chatbots, such as ChatGPT, Bard, and Bing Chat. This time-sensitive approach uncovers intricate details about these services' defenses, facilitating a proof-of-concept attack that successfully bypasses their mechanisms. Second, we introduce an automatic generation method for jailbreak prompts. Leveraging a fine-tuned LLM, we validate the potential of automated jailbreak generation across various commercial LLM chatbots. Our method achieves a promising average success rate of 21.58%, significantly outperforming the effectiveness of existing techniques. We have responsibly disclosed our findings to the concerned service providers, underscoring the urgent need for more robust defenses. Jailbreaker thus marks a significant step towards understanding and mitigating jailbreak threats in the realm of LLM chatbots.

摘要: 大型语言模型(LLM)由于其在理解和生成类似人类的文本方面的非凡熟练程度，使人工智能(AI)服务发生了革命性的变化。尤其是LLM聊天机器人，已经被广泛采用，改变了人机交互。然而，这些LLM聊天机器人很容易受到“越狱”攻击，即恶意用户操纵提示来引发不适当或敏感的响应，这违反了服务策略。尽管存在缓解此类威胁的尝试，但我们的研究显示，我们对这些漏洞的理解存在很大差距，这主要是由于LLM服务提供商实施了未披露的防御措施。在这篇文章中，我们介绍了越狱，一个全面的框架，提供了深入了解越狱攻击和对策。我们的工作做出了双重贡献。首先，我们提出了一种受基于时间的SQL注入技术启发的创新方法来对著名的LLM聊天机器人(如ChatGPT、Bard和Bing Chat)的防御策略进行逆向工程。这种对时间敏感的方法揭示了这些服务防御的复杂细节，为成功绕过它们的机制的概念验证攻击提供了便利。其次，介绍了一种越狱提示的自动生成方法。利用微调的LLM，我们验证了在各种商业LLM聊天机器人上自动越狱生成的潜力。我们的方法达到了21.58%的平均成功率，大大超过了现有技术的有效性。我们已经负责任地向有关服务提供商披露了我们的调查结果，强调了迫切需要更强大的防御措施。因此，越狱标志着在理解和减轻LLM聊天机器人领域的越狱威胁方面迈出了重要的一步。



## **24. Locally Differentially Private Document Generation Using Zero Shot Prompting**

基于零镜头提示的局部差异私有文档生成方法 cs.CL

Accepted at EMNLP 2023 (Findings)

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.16111v1) [paper-pdf](http://arxiv.org/pdf/2310.16111v1)

**Authors**: Saiteja Utpala, Sara Hooker, Pin Yu Chen

**Abstract**: Numerous studies have highlighted the privacy risks associated with pretrained large language models. In contrast, our research offers a unique perspective by demonstrating that pretrained large language models can effectively contribute to privacy preservation. We propose a locally differentially private mechanism called DP-Prompt, which leverages the power of pretrained large language models and zero-shot prompting to counter author de-anonymization attacks while minimizing the impact on downstream utility. When DP-Prompt is used with a powerful language model like ChatGPT (gpt-3.5), we observe a notable reduction in the success rate of de-anonymization attacks, showing that it surpasses existing approaches by a considerable margin despite its simpler design. For instance, in the case of the IMDB dataset, DP-Prompt (with ChatGPT) perfectly recovers the clean sentiment F1 score while achieving a 46\% reduction in author identification F1 score against static attackers and a 26\% reduction against adaptive attackers. We conduct extensive experiments across six open-source large language models, ranging up to 7 billion parameters, to analyze various effects of the privacy-utility tradeoff.

摘要: 许多研究都强调了与预先训练的大型语言模型相关的隐私风险。相比之下，我们的研究提供了一个独特的视角，证明了预先训练的大型语言模型可以有效地有助于隐私保护。我们提出了一种称为DP-Prompt的局部差异私有机制，该机制利用预先训练的大型语言模型和零镜头提示的能力来对抗作者去匿名化攻击，同时最小化对下游效用的影响。当DP-Prompt与ChatGPT(GPT-3.5)等强大的语言模型一起使用时，我们观察到去匿名化攻击的成功率显著下降，表明尽管它的设计更简单，但它在相当大程度上超过了现有的方法。例如，在IMDB数据集的情况下，DP-Prompt(使用ChatGPT)完美地恢复了干净的情感F1分数，同时在针对静态攻击者的作者识别F1分数和针对自适应攻击者的F1分数分别减少了46%和26%。我们在六个开放源码的大型语言模型上进行了广泛的实验，范围多达70亿个参数，以分析隐私-效用权衡的各种影响。



## **25. Self-Guard: Empower the LLM to Safeguard Itself**

自我保护：增强LLM的自我保护能力 cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15851v1) [paper-pdf](http://arxiv.org/pdf/2310.15851v1)

**Authors**: Zezhong Wang, Fangkai Yang, Lu Wang, Pu Zhao, Hongru Wang, Liang Chen, Qingwei Lin, Kam-Fai Wong

**Abstract**: The jailbreak attack can bypass the safety measures of a Large Language Model (LLM), generating harmful content. This misuse of LLM has led to negative societal consequences. Currently, there are two main approaches to address jailbreak attacks: safety training and safeguards. Safety training focuses on further training LLM to enhance its safety. On the other hand, safeguards involve implementing external models or filters to prevent harmful outputs. However, safety training has constraints in its ability to adapt to new attack types and often leads to a drop in model performance. Safeguards have proven to be of limited help. To tackle these issues, we propose a novel approach called Self-Guard, which combines the strengths of both safety methods. Self-Guard includes two stages. In the first stage, we enhance the model's ability to assess harmful content, and in the second stage, we instruct the model to consistently perform harmful content detection on its own responses. The experiment has demonstrated that Self-Guard is robust against jailbreak attacks. In the bad case analysis, we find that LLM occasionally provides harmless responses to harmful queries. Additionally, we evaluated the general capabilities of the LLM before and after safety training, providing evidence that Self-Guard does not result in the LLM's performance degradation. In sensitivity tests, Self-Guard not only avoids inducing over-sensitivity in LLM but also can even mitigate this issue.

摘要: 越狱攻击可以绕过大型语言模型(LLM)的安全措施，生成有害内容。这种对LLM的滥用已经导致了负面的社会后果。目前，解决越狱攻击的主要方法有两种：安全培训和保障措施。安全培训的重点是对LLM进行进一步培训，以提高其安全性。另一方面，保障措施涉及实施外部模型或过滤器，以防止有害输出。然而，安全培训在适应新攻击类型的能力方面存在限制，往往会导致模型性能下降。事实证明，保障措施的帮助有限。为了解决这些问题，我们提出了一种名为Self-Guard的新方法，它结合了两种安全方法的优点。自我保护包括两个阶段。在第一阶段，我们增强了模型评估有害内容的能力，在第二阶段，我们指示模型对其自身的响应进行一致的有害内容检测。实验证明，Self-Guard对越狱攻击具有很强的抵抗力。在坏案例分析中，我们发现LLM偶尔会对有害查询提供无害的响应。此外，我们在安全培训前后评估了LLM的一般能力，提供了自我保护不会导致LLM性能下降的证据。在敏感性测试中，Self-Guard不仅可以避免在LLM中诱导过度敏感，而且甚至可以缓解这一问题。



## **26. A Survey on LLM-generated Text Detection: Necessity, Methods, and Future Directions**

LLM生成的文本检测：必要性、方法和未来发展方向综述 cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.14724v2) [paper-pdf](http://arxiv.org/pdf/2310.14724v2)

**Authors**: Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Derek F. Wong, Lidia S. Chao

**Abstract**: The powerful ability to understand, follow, and generate complex language emerging from large language models (LLMs) makes LLM-generated text flood many areas of our daily lives at an incredible speed and is widely accepted by humans. As LLMs continue to expand, there is an imperative need to develop detectors that can detect LLM-generated text. This is crucial to mitigate potential misuse of LLMs and safeguard realms like artistic expression and social networks from harmful influence of LLM-generated content. The LLM-generated text detection aims to discern if a piece of text was produced by an LLM, which is essentially a binary classification task. The detector techniques have witnessed notable advancements recently, propelled by innovations in watermarking techniques, zero-shot methods, fine-turning LMs methods, adversarial learning methods, LLMs as detectors, and human-assisted methods. In this survey, we collate recent research breakthroughs in this area and underscore the pressing need to bolster detector research. We also delve into prevalent datasets, elucidating their limitations and developmental requirements. Furthermore, we analyze various LLM-generated text detection paradigms, shedding light on challenges like out-of-distribution problems, potential attacks, and data ambiguity. Conclusively, we highlight interesting directions for future research in LLM-generated text detection to advance the implementation of responsible artificial intelligence (AI). Our aim with this survey is to provide a clear and comprehensive introduction for newcomers while also offering seasoned researchers a valuable update in the field of LLM-generated text detection. The useful resources are publicly available at: https://github.com/NLP2CT/LLM-generated-Text-Detection.

摘要: 大型语言模型(LLM)强大的理解、跟踪和生成复杂语言的能力使得LLM生成的文本以令人难以置信的速度涌入我们日常生活的许多领域，并被人类广泛接受。随着LLMS的不断扩展，迫切需要开发能够检测LLM生成的文本的检测器。这对于减少LLM的潜在滥用以及保护艺术表达和社交网络等领域免受LLM生成的内容的有害影响至关重要。LLM生成的文本检测旨在识别一段文本是否由LLM生成，这本质上是一项二进制分类任务。最近，在水印技术、零镜头方法、精细旋转LMS方法、对抗性学习方法、作为检测器的LLMS以及人工辅助方法的创新的推动下，检测器技术有了显著的进步。在这次调查中，我们整理了这一领域的最新研究突破，并强调了支持探测器研究的迫切需要。我们还深入研究了流行的数据集，阐明了它们的局限性和发展需求。此外，我们分析了各种LLM生成的文本检测范例，揭示了诸如分发外问题、潜在攻击和数据歧义等挑战。最后，我们指出了未来在LLM生成的文本检测方面的有趣研究方向，以推进负责任人工智能(AI)的实施。我们这次调查的目的是为新手提供一个清晰而全面的介绍，同时也为经验丰富的研究人员提供在LLM生成的文本检测领域的有价值的更新。这些有用的资源可在以下网址公开获得：https://github.com/NLP2CT/LLM-generated-Text-Detection.



## **27. A Survey on Detection of LLMs-Generated Content**

基于LLMS的内容检测技术研究综述 cs.CL

We will keep updating at  https://github.com/Xianjun-Yang/Awesome_papers_on_LLMs_detection.git

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15654v1) [paper-pdf](http://arxiv.org/pdf/2310.15654v1)

**Authors**: Xianjun Yang, Liangming Pan, Xuandong Zhao, Haifeng Chen, Linda Petzold, William Yang Wang, Wei Cheng

**Abstract**: The burgeoning capabilities of advanced large language models (LLMs) such as ChatGPT have led to an increase in synthetic content generation with implications across a variety of sectors, including media, cybersecurity, public discourse, and education. As such, the ability to detect LLMs-generated content has become of paramount importance. We aim to provide a detailed overview of existing detection strategies and benchmarks, scrutinizing their differences and identifying key challenges and prospects in the field, advocating for more adaptable and robust models to enhance detection accuracy. We also posit the necessity for a multi-faceted approach to defend against various attacks to counter the rapidly advancing capabilities of LLMs. To the best of our knowledge, this work is the first comprehensive survey on the detection in the era of LLMs. We hope it will provide a broad understanding of the current landscape of LLMs-generated content detection, offering a guiding reference for researchers and practitioners striving to uphold the integrity of digital information in an era increasingly dominated by synthetic content. The relevant papers are summarized and will be consistently updated at https://github.com/Xianjun-Yang/Awesome_papers_on_LLMs_detection.git.

摘要: ChatGPT等高级大型语言模型（LLM）的新兴功能导致合成内容生成的增加，并对包括媒体、网络安全、公共话语和教育在内的各个领域产生影响。因此，检测LLMs生成的内容的能力变得至关重要。我们的目标是提供现有检测策略和基准的详细概述，仔细研究它们的差异，并确定该领域的关键挑战和前景，倡导更适应性和更强大的模型，以提高检测准确性。我们还强调，有必要采取多方面的方法来抵御各种攻击，以应对快速发展的LLM能力。据我们所知，这项工作是第一次全面调查的检测时代的LLM。我们希望它将提供对当前LLMs生成内容检测的广泛理解，为研究人员和从业人员提供指导性参考，努力在合成内容日益主导的时代维护数字信息的完整性。相关论文的摘要将在https://github.com/Xianjun-Yang/Awesome_papers_on_LLMs_detection.git上不断更新。



## **28. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

LLM自卫：通过自我检查，LLM知道自己被骗了 cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2308.07308v3) [paper-pdf](http://arxiv.org/pdf/2308.07308v3)

**Authors**: Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau

**Abstract**: Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2.

摘要: 大型语言模型(LLM)对于高质量的文本生成很受欢迎，但可能会产生有害的内容，即使通过强化学习与人类的价值观保持一致。对抗性提示可以绕过它们的安全措施。我们提出了LLM自卫，这是一种通过让LLM筛选诱导响应来防御这些攻击的简单方法。我们的方法不需要任何微调、输入预处理或迭代输出生成。相反，我们将生成的内容合并到预定义的提示中，并使用LLM的另一个实例来分析文本并预测它是否有害。我们在GPT 3.5和Llama 2上测试了LLM自卫，这两种当前最著名的LLM针对各种类型的攻击，例如对提示的强制诱导肯定反应和提示工程攻击。值得注意的是，LLM自卫成功地使用GPT 3.5和Llama 2将攻击成功率降低到几乎为0。



## **29. PrivInfer: Privacy-Preserving Inference for Black-box Large Language Model**

PrivInfer：黑箱大语言模型的隐私保护推理 cs.CR

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.12214v3) [paper-pdf](http://arxiv.org/pdf/2310.12214v3)

**Authors**: Meng Tong, Kejiang Chen, Yuang Qi, Jie Zhang, Weiming Zhang, Nenghai Yu

**Abstract**: Large language models (LLMs), such as ChatGPT, have simplified text generation tasks, yet their inherent privacy risks are increasingly garnering attention. Existing solutions for privacy-preserving inference face significant challenges in practical deployment and implementation. In this paper, we propose PrivInfer, the first practical framework for privacy-preserving inference. It comprises two modules specifically designed for black-box LLMs in text generation. The perturbation module, employing differential privacy, generates perturbed prompts, thus enabling privacy-preserving inference with black-box LLMs. The restoration module extracts coherent and meaningful responses from obtained perturbed results, thus ensuring the accomplishment of the text generation tasks. Additionally, to enhance privacy and utility further, we develop RANTEXT, a novel differential privacy mechanism integrated into the perturbation module of PrivInfer. This mechanism is specifically tailored for LLMs and utilizes random adjacency in text perturbations. Experimental results indicate that PrivInfer is comparable to GPT-4 in text generation quality, and RANTEXT outperforms the current leading scheme in privacy protection, even under its adaptive attack, our proposed GPT inference attack.

摘要: 大型语言模型(LLM)，如ChatGPT，简化了文本生成任务，但其固有的隐私风险正日益引起人们的关注。现有的隐私保护推理解决方案在实际部署和实施中面临着巨大的挑战。在本文中，我们提出了第一个实用的隐私保护推理框架PrivInfer。它包括两个模块，专门针对文本生成中的黑盒LLMS而设计。扰动模块采用不同的隐私，生成扰动提示，从而实现与黑盒LLMS的隐私保护推理。恢复模块从获得的扰动结果中提取连贯且有意义的响应，从而确保文本生成任务的完成。此外，为了进一步增强隐私和实用性，我们开发了一种新的差异隐私机制RANTEXT，该机制集成在PrivInfer的扰动模块中。该机制是专门为LLMS量身定做的，并利用文本扰动中的随机邻接。实验结果表明，PrivInfer在文本生成质量上与GPT-4相当，而RANTEXT在隐私保护方面的性能优于目前领先的方案，即使在其自适应攻击的情况下也是如此。



## **30. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

Janus界面：大型语言模型中的微调如何放大隐私风险 cs.CR

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15469v1) [paper-pdf](http://arxiv.org/pdf/2310.15469v1)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, XiaoFeng Wang, Haixu Tang

**Abstract**: The era post-2018 marked the advent of Large Language Models (LLMs), with innovations such as OpenAI's ChatGPT showcasing prodigious linguistic prowess. As the industry galloped toward augmenting model parameters and capitalizing on vast swaths of human language data, security and privacy challenges also emerged. Foremost among these is the potential inadvertent accrual of Personal Identifiable Information (PII) during web-based data acquisition, posing risks of unintended PII disclosure. While strategies like RLHF during training and Catastrophic Forgetting have been marshaled to control the risk of privacy infringements, recent advancements in LLMs, epitomized by OpenAI's fine-tuning interface for GPT-3.5, have reignited concerns. One may ask: can the fine-tuning of LLMs precipitate the leakage of personal information embedded within training datasets? This paper reports the first endeavor to seek the answer to the question, particularly our discovery of a new LLM exploitation avenue, called the Janus attack. In the attack, one can construct a PII association task, whereby an LLM is fine-tuned using a minuscule PII dataset, to potentially reinstate and reveal concealed PIIs. Our findings indicate that, with a trivial fine-tuning outlay, LLMs such as GPT-3.5 can transition from being impermeable to PII extraction to a state where they divulge a substantial proportion of concealed PII. This research, through its deep dive into the Janus attack vector, underscores the imperative of navigating the intricate interplay between LLM utility and privacy preservation.

摘要: 2018年后的时代标志着大型语言模型（LLM）的到来，OpenAI的ChatGPT等创新展示了巨大的语言能力。随着行业朝着增强模型参数和利用大量人类语言数据的方向发展，安全和隐私方面的挑战也随之出现。其中最重要的是在基于Web的数据采集过程中可能无意中积累个人身份信息（PII），从而造成意外PII泄露的风险。虽然训练期间的RLHF和灾难性遗忘等策略已被用于控制侵犯隐私的风险，但LLM的最新进展（以OpenAI针对GPT-3.5的微调接口为代表）重新引发了人们的担忧。有人可能会问：LLM的微调会导致嵌入在训练数据集中的个人信息泄露吗？本文报告了第一次努力寻求这个问题的答案，特别是我们发现了一个新的LLM开发途径，称为Janus攻击。在攻击中，可以构建PII关联任务，从而使用极小的PII数据集对LLM进行微调，以潜在地恢复和揭示隐藏的PII。我们的研究结果表明，通过微不足道的微调支出，LLM（如GPT-3.5）可以从不可渗透的PII提取过渡到它们泄露相当大比例的隐藏PII的状态。这项研究通过深入研究Janus攻击向量，强调了在LLM实用程序和隐私保护之间错综复杂的相互作用中导航的必要性。



## **31. AutoDAN: Automatic and Interpretable Adversarial Attacks on Large Language Models**

AutoDAN：对大型语言模型的自动和可解释的对抗性攻击 cs.CR

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15140v1) [paper-pdf](http://arxiv.org/pdf/2310.15140v1)

**Authors**: Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

**Abstract**: Safety alignment of Large Language Models (LLMs) can be compromised with manual jailbreak attacks and (automatic) adversarial attacks. Recent work suggests that patching LLMs against these attacks is possible: manual jailbreak attacks are human-readable but often limited and public, making them easy to block; adversarial attacks generate gibberish prompts that can be detected using perplexity-based filters. In this paper, we show that these solutions may be too optimistic. We propose an interpretable adversarial attack, \texttt{AutoDAN}, that combines the strengths of both types of attacks. It automatically generates attack prompts that bypass perplexity-based filters while maintaining a high attack success rate like manual jailbreak attacks. These prompts are interpretable and diverse, exhibiting strategies commonly used in manual jailbreak attacks, and transfer better than their non-readable counterparts when using limited training data or a single proxy model. We also customize \texttt{AutoDAN}'s objective to leak system prompts, another jailbreak application not addressed in the adversarial attack literature. Our work provides a new way to red-team LLMs and to understand the mechanism of jailbreak attacks.

摘要: 大型语言模型(LLM)的安全对齐可能会受到手动越狱攻击和(自动)对抗性攻击的影响。最近的研究表明，修补LLM以抵御这些攻击是可能的：手动越狱攻击是人类可读的，但通常是有限的和公开的，使它们很容易被阻止；对抗性攻击生成胡言乱语的提示，可以使用基于困惑的过滤器检测到。在本文中，我们证明了这些解决方案可能过于乐观。我们提出了一种可解释的对抗性攻击，它结合了这两种攻击的优点。它自动生成攻击提示，绕过基于困惑的过滤器，同时保持较高的攻击成功率，如手动越狱攻击。这些提示是可解释的和多样化的，展示了手动越狱攻击中常用的策略，并且在使用有限的训练数据或单一代理模型时，传输效果比不可读的相应提示更好。我们还定制了S的目标来泄露系统提示，这是另一个在对抗性攻击文献中没有涉及的越狱应用。我们的工作为红队低层管理和理解越狱攻击的机制提供了一种新的途径。



## **32. Did the Neurons Read your Book? Document-level Membership Inference for Large Language Models**

神经元有没有读过你的书？面向大型语言模型的文档级隶属度推理 cs.CL

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15007v1) [paper-pdf](http://arxiv.org/pdf/2310.15007v1)

**Authors**: Matthieu Meeus, Shubham Jain, Marek Rei, Yves-Alexandre de Montjoye

**Abstract**: With large language models (LLMs) poised to become embedded in our daily lives, questions are starting to be raised about the dataset(s) they learned from. These questions range from potential bias or misinformation LLMs could retain from their training data to questions of copyright and fair use of human-generated text. However, while these questions emerge, developers of the recent state-of-the-art LLMs become increasingly reluctant to disclose details on their training corpus. We here introduce the task of document-level membership inference for real-world LLMs, i.e. inferring whether the LLM has seen a given document during training or not. First, we propose a procedure for the development and evaluation of document-level membership inference for LLMs by leveraging commonly used data sources for training and the model release date. We then propose a practical, black-box method to predict document-level membership and instantiate it on OpenLLaMA-7B with both books and academic papers. We show our methodology to perform very well, reaching an impressive AUC of 0.856 for books and 0.678 for papers. We then show our approach to outperform the sentence-level membership inference attacks used in the privacy literature for the document-level membership task. We finally evaluate whether smaller models might be less sensitive to document-level inference and show OpenLLaMA-3B to be approximately as sensitive as OpenLLaMA-7B to our approach. Taken together, our results show that accurate document-level membership can be inferred for LLMs, increasing the transparency of technology poised to change our lives.

摘要: 随着大型语言模型（LLM）即将嵌入我们的日常生活中，人们开始对它们从中学习的数据集提出问题。这些问题的范围从潜在的偏见或错误信息LLM可以从他们的训练数据保留到版权和合理使用人类生成的文本的问题。然而，虽然这些问题出现了，但最近最先进的LLM的开发人员越来越不愿意透露他们训练语料库的细节。我们在这里介绍了真实世界LLM的文档级成员推断任务，即推断LLM在训练期间是否看到给定文档。首先，我们提出了一个程序，利用常用的数据源进行训练和模型发布日期的文档级成员推理LLM的开发和评估。然后，我们提出了一个实用的，黑盒的方法来预测文档级的成员资格，并在OpenLLaMA-7 B与书籍和学术论文的实例。我们显示我们的方法表现得非常好，书籍和论文的AUC分别为0.856和0.678。然后，我们展示了我们的方法，优于文档级成员资格任务的隐私文献中使用的文件级成员资格推断攻击。最后，我们评估是否较小的模型可能是不太敏感的文档级的推理，并显示OpenLLaMA-3B的敏感性大约为OpenLLaMA-7 B我们的方法。总的来说，我们的研究结果表明，可以为LLM推断出准确的文档级成员资格，从而提高了有望改变我们生活的技术的透明度。



## **33. MoPe: Model Perturbation-based Privacy Attacks on Language Models**

MOPE：基于模型扰动的语言模型隐私攻击 cs.LG

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.14369v1) [paper-pdf](http://arxiv.org/pdf/2310.14369v1)

**Authors**: Marvin Li, Jason Wang, Jeffrey Wang, Seth Neel

**Abstract**: Recent work has shown that Large Language Models (LLMs) can unintentionally leak sensitive information present in their training data. In this paper, we present Model Perturbations (MoPe), a new method to identify with high confidence if a given text is in the training data of a pre-trained language model, given white-box access to the models parameters. MoPe adds noise to the model in parameter space and measures the drop in log-likelihood at a given point $x$, a statistic we show approximates the trace of the Hessian matrix with respect to model parameters. Across language models ranging from $70$M to $12$B parameters, we show that MoPe is more effective than existing loss-based attacks and recently proposed perturbation-based methods. We also examine the role of training point order and model size in attack success, and empirically demonstrate that MoPe accurately approximate the trace of the Hessian in practice. Our results show that the loss of a point alone is insufficient to determine extractability -- there are training points we can recover using our method that have average loss. This casts some doubt on prior works that use the loss of a point as evidence of memorization or unlearning.

摘要: 最近的研究表明，大型语言模型(LLM)可能会无意中泄露其训练数据中存在的敏感信息。在本文中，我们提出了模型扰动(MOPE)，一种新的方法，在白盒访问模型参数的情况下，高置信度地识别给定文本是否在预先训练的语言模型的训练数据中。MOPE在参数空间中向模型添加噪声，并测量给定点$x$处的对数似然下降，我们给出的统计量近似于Hessian矩阵关于模型参数的迹。跨语言模型，从$7,000$M到$12$B参数，我们表明，MOPE比现有的基于损失的攻击和最近提出的基于扰动的方法更有效。我们还考察了训练点顺序和模型大小在攻击成功中的作用，并实证证明了MOPE在实践中准确地逼近了黑森轨迹。我们的结果表明，仅损失一个点不足以确定可萃取性--我们可以使用我们的方法恢复具有平均损失的训练点。这让人们对以前的作品产生了一些怀疑，这些作品把失去一分作为记忆或遗忘的证据。



## **34. Language Model Unalignment: Parametric Red-Teaming to Expose Hidden Harms and Biases**

语言模型不一致：暴露隐藏的危害和偏见的参数红色团队 cs.CL

Under Review

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.14303v1) [paper-pdf](http://arxiv.org/pdf/2310.14303v1)

**Authors**: Rishabh Bhardwaj, Soujanya Poria

**Abstract**: Red-teaming has been a widely adopted way to evaluate the harmfulness of Large Language Models (LLMs). It aims to jailbreak a model's safety behavior to make it act as a helpful agent disregarding the harmfulness of the query. Existing methods are primarily based on input text-based red-teaming such as adversarial prompts, low-resource prompts, or contextualized prompts to condition the model in a way to bypass its safe behavior. Bypassing the guardrails uncovers hidden harmful information and biases in the model that are left untreated or newly introduced by its safety training. However, prompt-based attacks fail to provide such a diagnosis owing to their low attack success rate, and applicability to specific models. In this paper, we present a new perspective on LLM safety research i.e., parametric red-teaming through Unalignment. It simply (instruction) tunes the model parameters to break model guardrails that are not deeply rooted in the model's behavior. Unalignment using as few as 100 examples can significantly bypass commonly referred to as CHATGPT, to the point where it responds with an 88% success rate to harmful queries on two safety benchmark datasets. On open-source models such as VICUNA-7B and LLAMA-2-CHAT 7B AND 13B, it shows an attack success rate of more than 91%. On bias evaluations, Unalignment exposes inherent biases in safety-aligned models such as CHATGPT and LLAMA- 2-CHAT where the model's responses are strongly biased and opinionated 64% of the time.

摘要: 红团队已被广泛采用来评估大型语言模型的危害性。它的目的是让模特的安全行为越狱，使其成为一个有帮助的代理人，而不考虑询问的危害性。现有方法主要基于诸如对抗性提示、低资源提示或情境化提示的基于输入文本的红团队，以使模型以绕过其安全行为的方式调节。绕过护栏发现了模型中隐藏的有害信息和偏见，这些信息和偏见是未经处理的或安全培训新引入的。然而，基于提示的攻击无法提供这样的诊断，因为它们的攻击成功率低，并且适用于特定的模型。在这篇文章中，我们提出了一个新的视角来研究LLM安全，即通过非对齐的参数红组。它只是(指令)调整模型参数，以打破并不深深植根于模型行为中的模型护栏。只要使用100个例子，UnAlign就可以显著绕过通常所说的CHATGPT，以至于它对两个安全基准数据集上的有害查询的响应成功率为88%。在VIVUNA-7B和LLAMA-2-Chat 7B和13B等开源机型上，攻击成功率超过91%。在偏差评估方面，UnAlign暴露了安全对齐模型中的固有偏见，如CHATGPT和Llama-2-Chat，其中模型的反应在64%的时间内是强烈偏见和固执己见的。



## **35. LoFT: Local Proxy Fine-tuning For Improving Transferability Of Adversarial Attacks Against Large Language Model**

LOFT：提高大型语言模型对抗性攻击可转移性的局部代理微调 cs.CL

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2310.04445v2) [paper-pdf](http://arxiv.org/pdf/2310.04445v2)

**Authors**: Muhammad Ahmed Shah, Roshan Sharma, Hira Dhamyal, Raphael Olivier, Ankit Shah, Joseph Konan, Dareen Alharthi, Hazim T Bukhari, Massa Baali, Soham Deshmukh, Michael Kuhlmann, Bhiksha Raj, Rita Singh

**Abstract**: It has been shown that Large Language Model (LLM) alignments can be circumvented by appending specially crafted attack suffixes with harmful queries to elicit harmful responses. To conduct attacks against private target models whose characterization is unknown, public models can be used as proxies to fashion the attack, with successful attacks being transferred from public proxies to private target models. The success rate of attack depends on how closely the proxy model approximates the private model. We hypothesize that for attacks to be transferrable, it is sufficient if the proxy can approximate the target model in the neighborhood of the harmful query. Therefore, in this paper, we propose \emph{Local Fine-Tuning (LoFT)}, \textit{i.e.}, fine-tuning proxy models on similar queries that lie in the lexico-semantic neighborhood of harmful queries to decrease the divergence between the proxy and target models. First, we demonstrate three approaches to prompt private target models to obtain similar queries given harmful queries. Next, we obtain data for local fine-tuning by eliciting responses from target models for the generated similar queries. Then, we optimize attack suffixes to generate attack prompts and evaluate the impact of our local fine-tuning on the attack's success rate. Experiments show that local fine-tuning of proxy models improves attack transferability and increases attack success rate by $39\%$, $7\%$, and $0.5\%$ (absolute) on target models ChatGPT, GPT-4, and Claude respectively.

摘要: 已有研究表明，通过在巧尽心思构建的攻击后缀上附加有害查询来引发有害响应，可以绕过大型语言模型(LLM)对齐。为了对特征未知的私有目标模型进行攻击，可以使用公共模型作为代理来进行攻击，成功的攻击将从公共代理转移到私有目标模型。攻击的成功率取决于代理模型与私有模型的接近程度。我们假设，对于可转移的攻击，只要代理能够逼近有害查询附近的目标模型就足够了。因此，在本文中，我们提出了对位于有害查询的词典-语义邻域中的相似查询的代理模型进行微调，以减少代理模型和目标模型之间的差异。首先，我们演示了三种方法来提示私人目标模型在给定有害查询的情况下获得类似的查询。接下来，我们通过从目标模型获取对生成的类似查询的响应来获得用于本地微调的数据。然后，我们优化攻击后缀来生成攻击提示，并评估我们的局部微调对攻击成功率的影响。实验表明，代理模型的局部微调提高了攻击的可转移性，使目标模型ChatGPT、GPT-4和Claude的攻击成功率分别提高了39美元、7美元和0.5美元(绝对)。



## **36. An LLM can Fool Itself: A Prompt-Based Adversarial Attack**

LLM可以自欺欺人：基于提示的对抗性攻击 cs.CR

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2310.13345v1) [paper-pdf](http://arxiv.org/pdf/2310.13345v1)

**Authors**: Xilie Xu, Keyi Kong, Ning Liu, Lizhen Cui, Di Wang, Jingfeng Zhang, Mohan Kankanhalli

**Abstract**: The wide-ranging applications of large language models (LLMs), especially in safety-critical domains, necessitate the proper evaluation of the LLM's adversarial robustness. This paper proposes an efficient tool to audit the LLM's adversarial robustness via a prompt-based adversarial attack (PromptAttack). PromptAttack converts adversarial textual attacks into an attack prompt that can cause the victim LLM to output the adversarial sample to fool itself. The attack prompt is composed of three important components: (1) original input (OI) including the original sample and its ground-truth label, (2) attack objective (AO) illustrating a task description of generating a new sample that can fool itself without changing the semantic meaning, and (3) attack guidance (AG) containing the perturbation instructions to guide the LLM on how to complete the task by perturbing the original sample at character, word, and sentence levels, respectively. Besides, we use a fidelity filter to ensure that PromptAttack maintains the original semantic meanings of the adversarial examples. Further, we enhance the attack power of PromptAttack by ensembling adversarial examples at different perturbation levels. Comprehensive empirical results using Llama2 and GPT-3.5 validate that PromptAttack consistently yields a much higher attack success rate compared to AdvGLUE and AdvGLUE++. Interesting findings include that a simple emoji can easily mislead GPT-3.5 to make wrong predictions.

摘要: 大型语言模型(LLM)的广泛应用，特别是在安全关键领域，需要对LLM的攻击健壮性进行适当的评估。提出了一种通过基于提示的敌意攻击(PromptAttack)来审计LLM攻击健壮性的有效工具。PromptAttack将对抗性文本攻击转换为攻击提示，这可能会导致受害者LLM输出对抗性样本来愚弄自己。攻击提示由三个重要部分组成：(1)原始输入(OI)，包括原始样本及其地面事实标签；(2)攻击目标(AO)，说明生成可以在不改变语义的情况下欺骗自己的新样本的任务描述；(3)攻击指导(AG)，包含扰动指令，以分别在字、词和句子层面指导LLM如何通过扰动原始样本来完成任务。此外，我们使用了一个保真度过滤器来确保PromptAttack保持了对抗性例子的原始语义。此外，我们通过集成不同扰动级别的对抗性实例来增强PromptAttack的攻击能力。使用Llama2和GPT-3.5的综合实验结果验证了PromptAttack始终比AdvGLUE和AdvGLUE++产生更高的攻击成功率。有趣的发现包括，一个简单的表情符号很容易误导GPT-3.5做出错误的预测。



## **37. Mitigating Backdoor Poisoning Attacks through the Lens of Spurious Correlation**

通过伪关联镜头缓解后门中毒攻击 cs.CL

accepted to EMNLP2023 (main conference)

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2305.11596v2) [paper-pdf](http://arxiv.org/pdf/2305.11596v2)

**Authors**: Xuanli He, Qiongkai Xu, Jun Wang, Benjamin Rubinstein, Trevor Cohn

**Abstract**: Modern NLP models are often trained over large untrusted datasets, raising the potential for a malicious adversary to compromise model behaviour. For instance, backdoors can be implanted through crafting training instances with a specific textual trigger and a target label. This paper posits that backdoor poisoning attacks exhibit \emph{spurious correlation} between simple text features and classification labels, and accordingly, proposes methods for mitigating spurious correlation as means of defence. Our empirical study reveals that the malicious triggers are highly correlated to their target labels; therefore such correlations are extremely distinguishable compared to those scores of benign features, and can be used to filter out potentially problematic instances. Compared with several existing defences, our defence method significantly reduces attack success rates across backdoor attacks, and in the case of insertion-based attacks, our method provides a near-perfect defence.

摘要: 现代NLP模型通常是在不可信的大型数据集上进行训练的，这增加了恶意对手危害模型行为的可能性。例如，可以通过制作带有特定文本触发器和目标标签的训练实例来植入后门。文章假设后门中毒攻击在简单文本特征和分类标签之间表现出虚假相关性，并相应地提出了减少虚假相关性的方法作为防御手段。我们的经验研究表明，恶意触发器与其目标标签高度相关；因此，与那些良性特征相比，这种相关性非常容易区分，可以用来过滤潜在的问题实例。与现有的几种防御方法相比，我们的防御方法显著降低了后门攻击的攻击成功率，并且在基于插入的攻击中，我们的方法提供了近乎完美的防御。



## **38. Assessing Privacy Risks in Language Models: A Case Study on Summarization Tasks**

评估语言模型中的隐私风险：摘要任务的案例研究 cs.CL

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2310.13291v1) [paper-pdf](http://arxiv.org/pdf/2310.13291v1)

**Authors**: Ruixiang Tang, Gord Lueck, Rodolfo Quispe, Huseyin A Inan, Janardhan Kulkarni, Xia Hu

**Abstract**: Large language models have revolutionized the field of NLP by achieving state-of-the-art performance on various tasks. However, there is a concern that these models may disclose information in the training data. In this study, we focus on the summarization task and investigate the membership inference (MI) attack: given a sample and black-box access to a model's API, it is possible to determine if the sample was part of the training data. We exploit text similarity and the model's resistance to document modifications as potential MI signals and evaluate their effectiveness on widely used datasets. Our results demonstrate that summarization models are at risk of exposing data membership, even in cases where the reference summary is not available. Furthermore, we discuss several safeguards for training summarization models to protect against MI attacks and discuss the inherent trade-off between privacy and utility.

摘要: 大型语言模型通过在各种任务上实现最先进的性能，使NLP领域发生了革命性的变化。然而，人们担心这些模型可能会泄露训练数据中的信息。在这项研究中，我们将重点放在总结任务上，并研究成员关系推理(MI)攻击：给定一个样本和对模型API的黑盒访问，可以确定该样本是否为训练数据的一部分。我们利用文本相似性和模型对文档修改的抵抗力作为潜在的MI信号，并在广泛使用的数据集上评估它们的有效性。我们的结果表明，摘要模型存在暴露数据成员身份的风险，即使在参考摘要不可用的情况下也是如此。此外，我们讨论了训练摘要模型的几种保护措施，以防止MI攻击，并讨论了隐私和效用之间的内在权衡。



## **39. Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy for Language Models**

朝向稳健剪枝：一种自适应的语言模型知识保留剪枝策略 cs.CL

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.13191v1) [paper-pdf](http://arxiv.org/pdf/2310.13191v1)

**Authors**: Jianwei Li, Qi Lei, Wei Cheng, Dongkuan Xu

**Abstract**: The pruning objective has recently extended beyond accuracy and sparsity to robustness in language models. Despite this, existing methods struggle to enhance robustness against adversarial attacks when continually increasing model sparsity and require a retraining process. As humans step into the era of large language models, these issues become increasingly prominent. This paper proposes that the robustness of language models is proportional to the extent of pre-trained knowledge they encompass. Accordingly, we introduce a post-training pruning strategy designed to faithfully replicate the embedding space and feature space of dense language models, aiming to conserve more pre-trained knowledge during the pruning process. In this setup, each layer's reconstruction error not only originates from itself but also includes cumulative error from preceding layers, followed by an adaptive rectification. Compared to other state-of-art baselines, our approach demonstrates a superior balance between accuracy, sparsity, robustness, and pruning cost with BERT on datasets SST2, IMDB, and AGNews, marking a significant stride towards robust pruning in language models.

摘要: 修剪目标最近已经超越了语言模型中的精确度和稀疏性，扩展到了健壮性。尽管如此，现有的方法在不断增加模型稀疏性的同时努力增强对敌对攻击的鲁棒性，并且需要重新训练过程。随着人类步入大型语言模型时代，这些问题变得日益突出。本文提出语言模型的稳健性与它们所包含的预训练知识的程度成正比。因此，我们提出了一种训练后剪枝策略，旨在忠实地复制密集语言模型的嵌入空间和特征空间，目的是在剪枝过程中保存更多的预先训练的知识。在这种设置中，每一层的重建误差不仅源于自身，还包括来自前几层的累积误差，然后进行自适应校正。与其他最先进的基线相比，我们的方法在精确度、稀疏性、健壮性和剪枝成本之间表现出了更好的平衡，在数据集Sst2、IMDB和AgNews上使用ERT，标志着在语言模型中朝着健壮剪枝迈出了重要的一步。



## **40. Prompt Injection Attacks and Defenses in LLM-Integrated Applications**

LLM集成应用程序中的即时注入攻击与防御 cs.CR

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12815v1) [paper-pdf](http://arxiv.org/pdf/2310.12815v1)

**Authors**: Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, Neil Zhenqiang Gong

**Abstract**: Large Language Models (LLMs) are increasingly deployed as the backend for a variety of real-world applications called LLM-Integrated Applications. Multiple recent works showed that LLM-Integrated Applications are vulnerable to prompt injection attacks, in which an attacker injects malicious instruction/data into the input of those applications such that they produce results as the attacker desires. However, existing works are limited to case studies. As a result, the literature lacks a systematic understanding of prompt injection attacks and their defenses. We aim to bridge the gap in this work. In particular, we propose a general framework to formalize prompt injection attacks. Existing attacks, which are discussed in research papers and blog posts, are special cases in our framework. Our framework enables us to design a new attack by combining existing attacks. Moreover, we also propose a framework to systematize defenses against prompt injection attacks. Using our frameworks, we conduct a systematic evaluation on prompt injection attacks and their defenses with 10 LLMs and 7 tasks. We hope our frameworks can inspire future research in this field. Our code is available at https://github.com/liu00222/Open-Prompt-Injection.

摘要: 大型语言模型(LLM)越来越多地被部署为各种实际应用程序的后端，这些应用程序称为LLm集成应用程序。最近的多项研究表明，LLM集成的应用程序容易受到即时注入攻击，即攻击者将恶意指令/数据注入到这些应用程序的输入中，以便它们产生攻击者想要的结果。然而，现有的研究成果仅限于案例研究。因此，文献对快速注射攻击及其防御缺乏系统的了解。我们的目标是弥合这项工作中的差距。特别是，我们提出了一个形式化的快速注入攻击的通用框架。研究论文和博客文章中讨论的现有攻击在我们的框架中是特例。我们的框架使我们能够通过组合现有的攻击来设计新的攻击。此外，我们还提出了一个对快速注入攻击进行系统化防御的框架。使用我们的框架，我们对快速注入攻击及其防御进行了系统的评估，包括10个LLM和7个任务。我们希望我们的框架能对这一领域的未来研究有所启发。我们的代码可以在https://github.com/liu00222/Open-Prompt-Injection.上找到



## **41. Automatic Hallucination Assessment for Aligned Large Language Models via Transferable Adversarial Attacks**

基于可转移对抗性攻击的对齐大语言模型的自动幻觉评估 cs.CL

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12516v1) [paper-pdf](http://arxiv.org/pdf/2310.12516v1)

**Authors**: Xiaodong Yu, Hao Cheng, Xiaodong Liu, Dan Roth, Jianfeng Gao

**Abstract**: Although remarkable progress has been achieved in preventing large language model (LLM) hallucinations using instruction tuning and retrieval augmentation, it remains challenging to measure the reliability of LLMs using human-crafted evaluation data which is not available for many tasks and domains and could suffer from data leakage. Inspired by adversarial machine learning, this paper aims to develop a method of automatically generating evaluation data by appropriately modifying existing data on which LLMs behave faithfully. Specifically, this paper presents AutoDebug, an LLM-based framework to use prompting chaining to generate transferable adversarial attacks in the form of question-answering examples. We seek to understand the extent to which these examples trigger the hallucination behaviors of LLMs.   We implement AutoDebug using ChatGPT and evaluate the resulting two variants of a popular open-domain question-answering dataset, Natural Questions (NQ), on a collection of open-source and proprietary LLMs under various prompting settings. Our generated evaluation data is human-readable and, as we show, humans can answer these modified questions well. Nevertheless, we observe pronounced accuracy drops across multiple LLMs including GPT-4. Our experimental results show that LLMs are likely to hallucinate in two categories of question-answering scenarios where (1) there are conflicts between knowledge given in the prompt and their parametric knowledge, or (2) the knowledge expressed in the prompt is complex. Finally, we find that the adversarial examples generated by our method are transferable across all considered LLMs. The examples generated by a small model can be used to debug a much larger model, making our approach cost-effective.

摘要: 虽然在使用指令调整和提取增强来防止大语言模型(LLM)幻觉方面取得了显著的进展，但使用人工制作的评估数据来衡量LLMS的可靠性仍然是具有挑战性的，这些数据对于许多任务和领域都是不可用的，并且可能受到数据泄漏的影响。受对抗性机器学习的启发，本文旨在开发一种自动生成评价数据的方法，该方法通过适当修改现有的数据来忠实地执行LLMS。具体地说，本文提出了AutoDebug，这是一个基于LLM的框架，使用提示链以问答示例的形式生成可转移的对抗性攻击。我们试图了解这些例子在多大程度上触发了LLM的幻觉行为。我们使用ChatGPT实现了AutoDebug，并在各种提示设置下，在一组开源和专有LLM上评估了一个流行的开放领域问答数据集的两个变体-自然问题(Natural Questions，NQ)。我们生成的评估数据是人类可读的，如我们所示，人类可以很好地回答这些修改后的问题。然而，我们观察到包括GPT-4在内的多个LLM的准确率显著下降。我们的实验结果表明，在两种类型的问答场景中，LLM可能会产生幻觉：(1)提示中给出的知识与其参数知识之间存在冲突；(2)提示中表达的知识复杂。最后，我们发现，我们的方法生成的对抗性例子可以在所有考虑的LLM之间转移。由小模型生成的示例可用于调试大得多的模型，从而使我们的方法具有成本效益。



## **42. Attack Prompt Generation for Red Teaming and Defending Large Language Models**

红色团队攻击提示生成及大型语言模型防御 cs.CL

Accepted to EMNLP 2023 (Findings)

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12505v1) [paper-pdf](http://arxiv.org/pdf/2310.12505v1)

**Authors**: Boyi Deng, Wenjie Wang, Fuli Feng, Yang Deng, Qifan Wang, Xiangnan He

**Abstract**: Large language models (LLMs) are susceptible to red teaming attacks, which can induce LLMs to generate harmful content. Previous research constructs attack prompts via manual or automatic methods, which have their own limitations on construction cost and quality. To address these issues, we propose an integrated approach that combines manual and automatic methods to economically generate high-quality attack prompts. Specifically, considering the impressive capabilities of newly emerged LLMs, we propose an attack framework to instruct LLMs to mimic human-generated prompts through in-context learning. Furthermore, we propose a defense framework that fine-tunes victim LLMs through iterative interactions with the attack framework to enhance their safety against red teaming attacks. Extensive experiments on different LLMs validate the effectiveness of our proposed attack and defense frameworks. Additionally, we release a series of attack prompts datasets named SAP with varying sizes, facilitating the safety evaluation and enhancement of more LLMs. Our code and dataset is available on https://github.com/Aatrox103/SAP .

摘要: 大型语言模型（LLM）容易受到红色团队攻击，这可能会导致LLM生成有害内容。以往的攻击提示构建方法都是通过人工或自动的方式进行的，在构建成本和质量上都有各自的局限性。为了解决这些问题，我们提出了一个综合的方法，结合手动和自动的方法，以经济地产生高质量的攻击提示。具体来说，考虑到新出现的LLM令人印象深刻的能力，我们提出了一个攻击框架，指示LLM通过上下文学习来模仿人类生成的提示。此外，我们还提出了一个防御框架，通过与攻击框架的迭代交互来微调受害者LLM，以增强其对红色团队攻击的安全性。在不同的LLM上进行的大量实验验证了我们提出的攻击和防御框架的有效性。此外，我们还发布了一系列不同大小的攻击提示数据集SAP，以促进更多LLM的安全评估和增强。我们的代码和数据集可以在https://github.com/Aatrox103/SAP上找到。



## **43. Red Teaming Language Model Detectors with Language Models**

具有语言模型的Red Teaming语言模型检测器 cs.CL

Preprint. Accepted by TACL

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2305.19713v2) [paper-pdf](http://arxiv.org/pdf/2305.19713v2)

**Authors**: Zhouxing Shi, Yihan Wang, Fan Yin, Xiangning Chen, Kai-Wei Chang, Cho-Jui Hsieh

**Abstract**: The prevalence and strong capability of large language models (LLMs) present significant safety and ethical risks if exploited by malicious users. To prevent the potentially deceptive usage of LLMs, recent works have proposed algorithms to detect LLM-generated text and protect LLMs. In this paper, we investigate the robustness and reliability of these LLM detectors under adversarial attacks. We study two types of attack strategies: 1) replacing certain words in an LLM's output with their synonyms given the context; 2) automatically searching for an instructional prompt to alter the writing style of the generation. In both strategies, we leverage an auxiliary LLM to generate the word replacements or the instructional prompt. Different from previous works, we consider a challenging setting where the auxiliary LLM can also be protected by a detector. Experiments reveal that our attacks effectively compromise the performance of all detectors in the study with plausible generations, underscoring the urgent need to improve the robustness of LLM-generated text detection systems.

摘要: 如果被恶意用户利用，大语言模型(LLM)的流行和强大的能力会带来巨大的安全和道德风险。为了防止潜在的欺骗性使用LLMS，最近的工作提出了检测LLM生成的文本并保护LLMS的算法。在本文中，我们研究了这些LLM检测器在对抗攻击下的健壮性和可靠性。我们研究了两种类型的攻击策略：1)用给定上下文的同义词替换LLM输出中的某些单词；2)自动搜索指令提示以改变生成的写作风格。在这两种策略中，我们利用辅助LLM来生成单词替换或指令提示。与以前的工作不同，我们考虑了一个具有挑战性的设置，其中辅助LLM也可以受到探测器的保护。实验表明，我们的攻击有效地折衷了研究中所有检测器的性能，生成了看似合理的代码，这突显了提高LLM生成的文本检测系统的健壮性的迫切需要。



## **44. PoisonPrompt: Backdoor Attack on Prompt-based Large Language Models**

PoisonPrompt：对基于提示的大型语言模型的后门攻击 cs.CL

Code will be released on: https://github.com/grasses/PoisonPrompt

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12439v1) [paper-pdf](http://arxiv.org/pdf/2310.12439v1)

**Authors**: Hongwei Yao, Jian Lou, Zhan Qin

**Abstract**: Prompts have significantly improved the performance of pretrained Large Language Models (LLMs) on various downstream tasks recently, making them increasingly indispensable for a diverse range of LLM application scenarios. However, the backdoor vulnerability, a serious security threat that can maliciously alter the victim model's normal predictions, has not been sufficiently explored for prompt-based LLMs. In this paper, we present POISONPROMPT, a novel backdoor attack capable of successfully compromising both hard and soft prompt-based LLMs. We evaluate the effectiveness, fidelity, and robustness of POISONPROMPT through extensive experiments on three popular prompt methods, using six datasets and three widely used LLMs. Our findings highlight the potential security threats posed by backdoor attacks on prompt-based LLMs and emphasize the need for further research in this area.

摘要: 最近，提示显著提高了预先训练的大型语言模型(LLM)在各种下游任务上的性能，使它们在各种LLM应用场景中越来越不可或缺。然而，后门漏洞是一个严重的安全威胁，可能会恶意改变受害者模型的正常预测，但对于基于提示的LLM来说，这种漏洞还没有得到充分的研究。在本文中，我们提出了一种新的后门攻击POISONPROMPT，它能够成功地攻破基于硬提示和软提示的LLMS。我们使用6个数据集和3个广泛使用的LLMS对POISONPROMPT的有效性、保真度和稳健性进行了评估。我们的发现突出了后门攻击对基于提示的LLM的潜在安全威胁，并强调了在这一领域进行进一步研究的必要性。



## **45. REMARK-LLM: A Robust and Efficient Watermarking Framework for Generative Large Language Models**

Remmark-LLM：一种面向生成性大语言模型的健壮高效水印框架 cs.CR

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.12362v1) [paper-pdf](http://arxiv.org/pdf/2310.12362v1)

**Authors**: Ruisi Zhang, Shehzeen Samarah Hussain, Paarth Neekhara, Farinaz Koushanfar

**Abstract**: We present REMARK-LLM, a novel efficient, and robust watermarking framework designed for texts generated by large language models (LLMs). Synthesizing human-like content using LLMs necessitates vast computational resources and extensive datasets, encapsulating critical intellectual property (IP). However, the generated content is prone to malicious exploitation, including spamming and plagiarism. To address the challenges, REMARK-LLM proposes three new components: (i) a learning-based message encoding module to infuse binary signatures into LLM-generated texts; (ii) a reparameterization module to transform the dense distributions from the message encoding to the sparse distribution of the watermarked textual tokens; (iii) a decoding module dedicated for signature extraction; Furthermore, we introduce an optimized beam search algorithm to guarantee the coherence and consistency of the generated content. REMARK-LLM is rigorously trained to encourage the preservation of semantic integrity in watermarked content, while ensuring effective watermark retrieval. Extensive evaluations on multiple unseen datasets highlight REMARK-LLM proficiency and transferability in inserting 2 times more signature bits into the same texts when compared to prior art, all while maintaining semantic integrity. Furthermore, REMARK-LLM exhibits better resilience against a spectrum of watermark detection and removal attacks.

摘要: 我们提出了一种新的高效、健壮的数字水印框架--REMARK-LLM，它适用于大型语言模型(LLMS)生成的文本。使用LLMS合成类似人类的内容需要大量的计算资源和大量的数据集，并封装关键知识产权(IP)。然而，生成的内容容易受到恶意攻击，包括垃圾邮件和抄袭。为了应对这些挑战，REMARK-LLM提出了三个新的组件：(I)基于学习的消息编码模块，将二进制签名注入到LLM生成的文本中；(Ii)重新参数化模块，将密集分布的消息编码转换为稀疏分布的水印文本标记；(Iii)专门用于签名提取的解码模块；此外，我们还引入了优化的波束搜索算法，以保证生成内容的一致性和一致性。Rmark-LLM经过了严格的培训，以鼓励在水印内容中保持语义完整性，同时确保有效的水印检索。对多个不可见数据集的广泛评估突出了Remmark-LLM的熟练程度和可转移性，即与现有技术相比，在相同的文本中插入2倍多的签名比特，同时保持语义完整性。此外，REMARK-LLM对一系列水印检测和删除攻击表现出更好的弹性。



## **46. PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

PromptBitch：评估大型语言模型在对抗性提示下的稳健性 cs.CL

Technical report; code is at:  https://github.com/microsoft/promptbench

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2306.04528v4) [paper-pdf](http://arxiv.org/pdf/2306.04528v4)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Yue Zhang, Neil Zhenqiang Gong, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptBench, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. The adversarial prompts, crafted to mimic plausible user errors like typos or synonyms, aim to evaluate how slight deviations can affect LLM outcomes while maintaining semantic integrity. These prompts are then employed in diverse tasks, such as sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4788 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets. Our findings demonstrate that contemporary LLMs are not robust to adversarial prompts. Furthermore, we present comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users. Code is available at: https://github.com/microsoft/promptbench.

摘要: 学术界和工业界对大型语言模型(LLM)的依赖日益增加，这就要求我们必须全面了解它们对提示的稳健性。为了响应这一关键需求，我们引入了PromptBtch，这是一个健壮性基准，旨在衡量LLMS对敌意提示的弹性。这项研究使用了过多的对抗性文本攻击，目标是多个层面的提示：字符、单词、句子和语义。这些对抗性提示旨在模仿打字或同义词等看似合理的用户错误，旨在评估微小的偏差如何在保持语义完整性的同时影响LLM结果。这些提示随后被用于不同的任务，如情感分析、自然语言推理、阅读理解、机器翻译和数学解题。我们的研究产生了4788个对抗性提示，仔细评估了8个任务和13个数据集。我们的研究结果表明，当代的LLM对敌意提示并不健壮。此外，我们还提供了全面的分析，以了解即时健壮性及其可转移性背后的奥秘。然后，我们提供了有洞察力的健壮性分析和实用的即时撰写建议，这对研究人员和日常用户都是有益的。代码可从以下网址获得：https://github.com/microsoft/promptbench.



## **47. Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense**

释义可以躲避人工智能生成的文本的检测，但检索是一种有效的防御 cs.CL

NeurIPS 2023 camera ready (32 pages). Code, models, data available in  https://github.com/martiansideofthemoon/ai-detection-paraphrases

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2303.13408v2) [paper-pdf](http://arxiv.org/pdf/2303.13408v2)

**Authors**: Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, Mohit Iyyer

**Abstract**: The rise in malicious usage of large language models, such as fake content creation and academic plagiarism, has motivated the development of approaches that identify AI-generated text, including those based on watermarking or outlier detection. However, the robustness of these detection algorithms to paraphrases of AI-generated text remains unclear. To stress test these detectors, we build a 11B parameter paraphrase generation model (DIPPER) that can paraphrase paragraphs, condition on surrounding context, and control lexical diversity and content reordering. Using DIPPER to paraphrase text generated by three large language models (including GPT3.5-davinci-003) successfully evades several detectors, including watermarking, GPTZero, DetectGPT, and OpenAI's text classifier. For example, DIPPER drops detection accuracy of DetectGPT from 70.3% to 4.6% (at a constant false positive rate of 1%), without appreciably modifying the input semantics.   To increase the robustness of AI-generated text detection to paraphrase attacks, we introduce a simple defense that relies on retrieving semantically-similar generations and must be maintained by a language model API provider. Given a candidate text, our algorithm searches a database of sequences previously generated by the API, looking for sequences that match the candidate text within a certain threshold. We empirically verify our defense using a database of 15M generations from a fine-tuned T5-XXL model and find that it can detect 80% to 97% of paraphrased generations across different settings while only classifying 1% of human-written sequences as AI-generated. We open-source our models, code and data.

摘要: 恶意使用大型语言模型的增加，如虚假内容创作和学术抄袭，推动了识别人工智能生成文本的方法的发展，包括基于水印或离群值检测的方法。然而，这些检测算法对人工智能生成的文本的释义的稳健性尚不清楚。为了对这些检测器进行压力测试，我们建立了一个11B参数转述生成模型(Dipper)，该模型可以转译段落、对周围上下文进行条件转换、控制词汇多样性和内容重新排序。使用Dipper解释由三个大型语言模型(包括GPT3.5-DaVinci-003)生成的文本，成功地避开了几个检测器，包括水印、GPTZero、DetectGPT和OpenAI的文本分类器。例如，Dipper将DetectGPT的检测准确率从70.3%下降到4.6%(在1%的恒定假阳性率下)，而不会明显修改输入语义。为了提高人工智能生成的文本检测对转述攻击的健壮性，我们引入了一种简单的防御，该防御依赖于检索语义相似的生成，并且必须由语言模型API提供商维护。在给定候选文本的情况下，我们的算法搜索先前由API生成的序列数据库，寻找在特定阈值内与候选文本匹配的序列。我们使用来自微调的T5-XXL模型的1500万代数据库经验地验证了我们的防御，发现它可以检测到不同设置下80%到97%的释义世代，而只有1%的人写序列被归类为人工智能生成的序列。我们将我们的模型、代码和数据开源。



## **48. Identifying and Mitigating the Security Risks of Generative AI**

识别和缓解生成性人工智能的安全风险 cs.AI

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2308.14840v3) [paper-pdf](http://arxiv.org/pdf/2308.14840v3)

**Authors**: Clark Barrett, Brad Boyd, Elie Burzstein, Nicholas Carlini, Brad Chen, Jihye Choi, Amrita Roy Chowdhury, Mihai Christodorescu, Anupam Datta, Soheil Feizi, Kathleen Fisher, Tatsunori Hashimoto, Dan Hendrycks, Somesh Jha, Daniel Kang, Florian Kerschbaum, Eric Mitchell, John Mitchell, Zulfikar Ramzan, Khawaja Shams, Dawn Song, Ankur Taly, Diyi Yang

**Abstract**: Every major technical invention resurfaces the dual-use dilemma -- the new technology has the potential to be used for good as well as for harm. Generative AI (GenAI) techniques, such as large language models (LLMs) and diffusion models, have shown remarkable capabilities (e.g., in-context learning, code-completion, and text-to-image generation and editing). However, GenAI can be used just as well by attackers to generate new attacks and increase the velocity and efficacy of existing attacks.   This paper reports the findings of a workshop held at Google (co-organized by Stanford University and the University of Wisconsin-Madison) on the dual-use dilemma posed by GenAI. This paper is not meant to be comprehensive, but is rather an attempt to synthesize some of the interesting findings from the workshop. We discuss short-term and long-term goals for the community on this topic. We hope this paper provides both a launching point for a discussion on this important topic as well as interesting problems that the research community can work to address.

摘要: 每一项重大技术发明都会重新面临两难境地--新技术既有可能被用来做好事，也有可能被用来做坏事。生成性人工智能(GenAI)技术，如大型语言模型(LLMS)和扩散模型，已经显示出非凡的能力(例如，上下文学习、代码完成以及文本到图像的生成和编辑)。然而，攻击者也可以利用GenAI来生成新的攻击，并提高现有攻击的速度和效率。本文报告了在谷歌(由斯坦福大学和威斯康星大学麦迪逊分校联合举办)举行的关于GenAI造成的两用困境的研讨会的结果。这篇论文并不是要全面的，而是试图综合研讨会的一些有趣的发现。我们就这一主题讨论社区的短期和长期目标。我们希望这篇论文既为讨论这一重要主题提供了一个起点，也为研究界可以努力解决的有趣问题提供了一个起点。



## **49. Last One Standing: A Comparative Analysis of Security and Privacy of Soft Prompt Tuning, LoRA, and In-Context Learning**

最后一人：软提示调谐、LORA和情境学习的安全性和隐私性比较分析 cs.CR

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2310.11397v1) [paper-pdf](http://arxiv.org/pdf/2310.11397v1)

**Authors**: Rui Wen, Tianhao Wang, Michael Backes, Yang Zhang, Ahmed Salem

**Abstract**: Large Language Models (LLMs) are powerful tools for natural language processing, enabling novel applications and user experiences. However, to achieve optimal performance, LLMs often require adaptation with private data, which poses privacy and security challenges. Several techniques have been proposed to adapt LLMs with private data, such as Low-Rank Adaptation (LoRA), Soft Prompt Tuning (SPT), and In-Context Learning (ICL), but their comparative privacy and security properties have not been systematically investigated. In this work, we fill this gap by evaluating the robustness of LoRA, SPT, and ICL against three types of well-established attacks: membership inference, which exposes data leakage (privacy); backdoor, which injects malicious behavior (security); and model stealing, which can violate intellectual property (privacy and security). Our results show that there is no silver bullet for privacy and security in LLM adaptation and each technique has different strengths and weaknesses.

摘要: 大型语言模型（LLM）是自然语言处理的强大工具，可以实现新颖的应用程序和用户体验。然而，为了实现最佳性能，LLM通常需要适应私有数据，这带来了隐私和安全挑战。已经提出了几种技术来适应LLM与私人数据，如低秩自适应（LoRA），软提示调整（SPT）和上下文学习（ICL），但他们的比较隐私和安全属性还没有得到系统的研究。在这项工作中，我们通过评估LoRA，SPT和ICL对三种类型的成熟攻击的鲁棒性来填补这一空白：成员推断，暴露数据泄漏（隐私）;后门，注入恶意行为（安全）;和模型窃取，可能违反知识产权（隐私和安全）。我们的研究结果表明，在LLM适应中没有隐私和安全的银弹，每种技术都有不同的优点和缺点。



## **50. Survey of Vulnerabilities in Large Language Models Revealed by Adversarial Attacks**

对抗性攻击揭示的大型语言模型中的漏洞调查 cs.CL

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10844v1) [paper-pdf](http://arxiv.org/pdf/2310.10844v1)

**Authors**: Erfan Shayegani, Md Abdullah Al Mamun, Yu Fu, Pedram Zaree, Yue Dong, Nael Abu-Ghazaleh

**Abstract**: Large Language Models (LLMs) are swiftly advancing in architecture and capability, and as they integrate more deeply into complex systems, the urgency to scrutinize their security properties grows. This paper surveys research in the emerging interdisciplinary field of adversarial attacks on LLMs, a subfield of trustworthy ML, combining the perspectives of Natural Language Processing and Security. Prior work has shown that even safety-aligned LLMs (via instruction tuning and reinforcement learning through human feedback) can be susceptible to adversarial attacks, which exploit weaknesses and mislead AI systems, as evidenced by the prevalence of `jailbreak' attacks on models like ChatGPT and Bard. In this survey, we first provide an overview of large language models, describe their safety alignment, and categorize existing research based on various learning structures: textual-only attacks, multi-modal attacks, and additional attack methods specifically targeting complex systems, such as federated learning or multi-agent systems. We also offer comprehensive remarks on works that focus on the fundamental sources of vulnerabilities and potential defenses. To make this field more accessible to newcomers, we present a systematic review of existing works, a structured typology of adversarial attack concepts, and additional resources, including slides for presentations on related topics at the 62nd Annual Meeting of the Association for Computational Linguistics (ACL'24).

摘要: 大型语言模型(LLM)在体系结构和功能方面正在迅速发展，随着它们更深入地集成到复杂系统中，审查其安全属性的紧迫性也在增长。本文结合自然语言处理和安全的角度，对可信ML的一个子领域--LLMS的对抗性攻击这一新兴交叉学科领域的研究进行了综述。先前的工作表明，即使是与安全一致的LLM(通过指令调整和通过人类反馈的强化学习)也可能容易受到对手攻击，这些攻击利用弱点并误导人工智能系统，对ChatGPT和Bard等模型的“越狱”攻击盛行就是明证。在这次调查中，我们首先提供了大型语言模型的概述，描述了它们的安全对齐，并基于各种学习结构对现有研究进行了分类：纯文本攻击、多模式攻击以及专门针对复杂系统的额外攻击方法，如联合学习或多代理系统。我们还对侧重于漏洞的根本来源和潜在防御的工作进行了全面的评论。为了使这个领域更容易为新手所接受，我们提供了对现有工作的系统回顾，对抗性攻击概念的结构化类型学，以及额外的资源，包括在第62届计算语言学协会年会(ACL‘24)上相关主题的演示幻灯片。



