# Latest Large Language Model Attack Papers
**update at 2024-08-22 17:17:14**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Competence-Based Analysis of Language Models**

基于能力的语言模型分析 cs.CL

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2303.00333v4) [paper-pdf](http://arxiv.org/pdf/2303.00333v4)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent successes of large, pretrained neural language models (LLMs), comparatively little is known about the representations of linguistic structure they learn during pretraining, which can lead to unexpected behaviors in response to prompt variation or distribution shift. To better understand these models and behaviors, we introduce a general model analysis framework to study LLMs with respect to their representation and use of human-interpretable linguistic properties. Our framework, CALM (Competence-based Analysis of Language Models), is designed to investigate LLM competence in the context of specific tasks by intervening on models' internal representations of different linguistic properties using causal probing, and measuring models' alignment under these interventions with a given ground-truth causal model of the task. We also develop a new approach for performing causal probing interventions using gradient-based adversarial attacks, which can target a broader range of properties and representations than prior techniques. Finally, we carry out a case study of CALM using these interventions to analyze and compare LLM competence across a variety of lexical inference tasks, showing that CALM can be used to explain and predict behaviors across these tasks.

摘要: 尽管最近大型的预训练神经语言模型(LLM)取得了成功，但人们对它们在预训练中学习的语言结构的表征知之甚少，这可能会导致对迅速变化或分布变化的意外行为。为了更好地理解这些模型和行为，我们引入了一个通用的模型分析框架，从它们对人类可解释的语言属性的表示和使用方面来研究LLM。基于能力的语言模型分析框架旨在通过因果探究干预模型对不同语言属性的内部表征，并测量模型在这些干预下与给定任务的基本事实因果模型的一致性，从而考察特定任务背景下的语言学习能力。我们还开发了一种使用基于梯度的对抗性攻击来执行因果探测干预的新方法，该方法可以针对比现有技术更广泛的属性和表示。最后，我们使用这些干预手段对CAMLE进行了个案研究，分析和比较了不同词汇推理任务的LLM能力，结果表明CAMPE可以用来解释和预测这些任务中的行为。



## **2. Against All Odds: Overcoming Typology, Script, and Language Confusion in Multilingual Embedding Inversion Attacks**

克服一切困难：克服多语言嵌入倒置攻击中的类型学、脚本和语言混乱 cs.CL

11 pages, 4 figures, 7 tables

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11749v1) [paper-pdf](http://arxiv.org/pdf/2408.11749v1)

**Authors**: Yiyi Chen, Russa Biswas, Heather Lent, Johannes Bjerva

**Abstract**: Large Language Models (LLMs) are susceptible to malicious influence by cyber attackers through intrusions such as adversarial, backdoor, and embedding inversion attacks. In response, the burgeoning field of LLM Security aims to study and defend against such threats. Thus far, the majority of works in this area have focused on monolingual English models, however, emerging research suggests that multilingual LLMs may be more vulnerable to various attacks than their monolingual counterparts. While previous work has investigated embedding inversion over a small subset of European languages, it is challenging to extrapolate these findings to languages from different linguistic families and with differing scripts. To this end, we explore the security of multilingual LLMs in the context of embedding inversion attacks and investigate cross-lingual and cross-script inversion across 20 languages, spanning over 8 language families and 12 scripts. Our findings indicate that languages written in Arabic script and Cyrillic script are particularly vulnerable to embedding inversion, as are languages within the Indo-Aryan language family. We further observe that inversion models tend to suffer from language confusion, sometimes greatly reducing the efficacy of an attack. Accordingly, we systematically explore this bottleneck for inversion models, uncovering predictable patterns which could be leveraged by attackers. Ultimately, this study aims to further the field's understanding of the outstanding security vulnerabilities facing multilingual LLMs and raise awareness for the languages most at risk of negative impact from these attacks.

摘要: 大型语言模型(LLM)容易受到网络攻击者通过对抗性、后门和嵌入反转攻击等入侵的恶意影响。作为回应，LLM Security这个新兴领域的目标是研究和防御此类威胁。到目前为止，这一领域的研究大多集中在单语英语模型上，然而，新的研究表明，多语种的LLM可能比单语的LLM更容易受到各种攻击。虽然以前的工作已经研究了在一小部分欧洲语言上嵌入倒置，但将这些发现外推到来自不同语系和不同脚本的语言是具有挑战性的。为此，我们在嵌入倒置攻击的情况下探索了多语言LLMS的安全性，并研究了跨语言和跨脚本的跨语言和跨脚本倒置，涉及8个语系和12个脚本。我们的发现表明，用阿拉伯文字和西里尔文字书写的语言特别容易嵌入倒置，印度-雅利安语系的语言也是如此。我们进一步观察到，倒置模型往往受到语言混乱的影响，有时会极大地降低攻击的有效性。因此，我们系统地探索了倒置模型的这一瓶颈，揭示了可被攻击者利用的可预测模式。最终，这项研究旨在加深外地对多语种土地管理系统面临的突出安全漏洞的了解，并提高对最有可能受到这些攻击的负面影响的语言的认识。



## **3. Watch Out for Your Guidance on Generation! Exploring Conditional Backdoor Attacks against Large Language Models**

留意您对世代的指导！探索针对大型语言模型的条件后门攻击 cs.CL

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2404.14795v4) [paper-pdf](http://arxiv.org/pdf/2404.14795v4)

**Authors**: Jiaming He, Wenbo Jiang, Guanyu Hou, Wenshu Fan, Rui Zhang, Hongwei Li

**Abstract**: Mainstream backdoor attacks on large language models (LLMs) typically set a fixed trigger in the input instance and specific responses for triggered queries. However, the fixed trigger setting (e.g., unusual words) may be easily detected by human detection, limiting the effectiveness and practicality in real-world scenarios. To enhance the stealthiness of backdoor activation, we present a new poisoning paradigm against LLMs triggered by specifying generation conditions, which are commonly adopted strategies by users during model inference. The poisoned model performs normally for output under normal/other generation conditions, while becomes harmful for output under target generation conditions. To achieve this objective, we introduce BrieFool, an efficient attack framework. It leverages the characteristics of generation conditions by efficient instruction sampling and poisoning data generation, thereby influencing the behavior of LLMs under target conditions. Our attack can be generally divided into two types with different targets: Safety unalignment attack and Ability degradation attack. Our extensive experiments demonstrate that BrieFool is effective across safety domains and ability domains, achieving higher success rates than baseline methods, with 94.3 % on GPT-3.5-turbo

摘要: 针对大型语言模型(LLM)的主流后门攻击通常会在输入实例中设置固定的触发器，并为触发的查询设置特定的响应。然而，固定的触发设置(例如，不寻常的单词)可能很容易被人类检测到，从而限制了在现实世界场景中的有效性和实用性。为了增强后门激活的隐蔽性，我们提出了一种新的针对通过指定生成条件触发的LLM的中毒范例，这些策略是用户在模型推理中经常采用的策略。中毒模型在正常/其他发电条件下的出力表现正常，而在目标发电条件下的出力变得有害。为了实现这一目标，我们引入了一种高效的攻击框架BrieFool。它通过高效的指令采样和中毒数据生成来利用生成条件的特征，从而影响目标条件下的LLM的行为。我们的攻击一般可以分为两种类型，针对不同的目标：安全联盟攻击和能力退化攻击。我们的广泛实验表明，BrieFool跨安全域和能量域是有效的，获得了比基准方法更高的成功率，在GPT-3.5-Turbo上的成功率为94.3%



## **4. Large Language Models are Good Attackers: Efficient and Stealthy Textual Backdoor Attacks**

大型语言模型是好的攻击者：高效且隐秘的文本后门攻击 cs.CL

Under Review

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11587v1) [paper-pdf](http://arxiv.org/pdf/2408.11587v1)

**Authors**: Ziqiang Li, Yueqi Zeng, Pengfei Xia, Lei Liu, Zhangjie Fu, Bin Li

**Abstract**: With the burgeoning advancements in the field of natural language processing (NLP), the demand for training data has increased significantly. To save costs, it has become common for users and businesses to outsource the labor-intensive task of data collection to third-party entities. Unfortunately, recent research has unveiled the inherent risk associated with this practice, particularly in exposing NLP systems to potential backdoor attacks. Specifically, these attacks enable malicious control over the behavior of a trained model by poisoning a small portion of the training data. Unlike backdoor attacks in computer vision, textual backdoor attacks impose stringent requirements for attack stealthiness. However, existing attack methods meet significant trade-off between effectiveness and stealthiness, largely due to the high information entropy inherent in textual data. In this paper, we introduce the Efficient and Stealthy Textual backdoor attack method, EST-Bad, leveraging Large Language Models (LLMs). Our EST-Bad encompasses three core strategies: optimizing the inherent flaw of models as the trigger, stealthily injecting triggers with LLMs, and meticulously selecting the most impactful samples for backdoor injection. Through the integration of these techniques, EST-Bad demonstrates an efficient achievement of competitive attack performance while maintaining superior stealthiness compared to prior methods across various text classifier datasets.

摘要: 随着自然语言处理(NLP)领域的迅速发展，对训练数据的需求显著增加。为了节省成本，用户和企业将劳动密集型的数据收集任务外包给第三方实体已经变得很常见。不幸的是，最近的研究揭示了与这种做法相关的固有风险，特别是在将NLP系统暴露于潜在的后门攻击方面。具体地说，这些攻击通过对一小部分训练数据下毒，实现了对训练模型行为的恶意控制。与计算机视觉中的后门攻击不同，文本后门攻击对攻击的隐蔽性提出了严格的要求。然而，现有的攻击方法在有效性和隐蔽性之间达到了显著的权衡，这在很大程度上是由于文本数据固有的高信息熵。本文利用大型语言模型，介绍了一种高效、隐蔽的文本后门攻击方法EST-Bad。我们的EST-Bad包含三个核心策略：优化模型的固有缺陷作为触发器，悄悄地向触发器注入LLM，以及精心选择最有影响力的样本进行后门注入。通过这些技术的集成，EST-Bad展示了在与各种文本分类器数据集的现有方法相比保持优越的隐蔽性的同时，有效地实现了竞争性攻击性能。



## **5. SHIELD: Evaluation and Defense Strategies for Copyright Compliance in LLM Text Generation**

SHIELD：LLM文本生成中版权合规性的评估和防御策略 cs.CL

Work in progress

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2406.12975v2) [paper-pdf](http://arxiv.org/pdf/2406.12975v2)

**Authors**: Xiaoze Liu, Ting Sun, Tianyang Xu, Feijie Wu, Cunxiang Wang, Xiaoqian Wang, Jing Gao

**Abstract**: Large Language Models (LLMs) have transformed machine learning but raised significant legal concerns due to their potential to produce text that infringes on copyrights, resulting in several high-profile lawsuits. The legal landscape is struggling to keep pace with these rapid advancements, with ongoing debates about whether generated text might plagiarize copyrighted materials. Current LLMs may infringe on copyrights or overly restrict non-copyrighted texts, leading to these challenges: (i) the need for a comprehensive evaluation benchmark to assess copyright compliance from multiple aspects; (ii) evaluating robustness against safeguard bypassing attacks; and (iii) developing effective defense targeted against the generation of copyrighted text. To tackle these challenges, we introduce a curated dataset to evaluate methods, test attack strategies, and propose lightweight, real-time defense to prevent the generation of copyrighted text, ensuring the safe and lawful use of LLMs. Our experiments demonstrate that current LLMs frequently output copyrighted text, and that jailbreaking attacks can significantly increase the volume of copyrighted output. Our proposed defense mechanism significantly reduces the volume of copyrighted text generated by LLMs by effectively refusing malicious requests. Code is publicly available at https://github.com/xz-liu/SHIELD

摘要: 大型语言模型(LLM)改变了机器学习，但由于它们有可能产生侵犯版权的文本，因此引发了重大的法律担忧，导致了几起备受瞩目的诉讼。法律界正在努力跟上这些快速进步的步伐，关于生成的文本是否可能抄袭受版权保护的材料的争论仍在继续。当前的LLM可能会侵犯版权或过度限制非版权文本，从而导致以下挑战：(I)需要一个全面的评估基准来从多个方面评估版权合规性；(Ii)评估针对保护绕过攻击的稳健性；以及(Iii)针对版权文本的生成开发有效的防御措施。为了应对这些挑战，我们引入了一个经过精选的数据集来评估方法，测试攻击策略，并提出了轻量级、实时的防御措施来防止版权文本的生成，确保了LLMS的安全和合法使用。我们的实验表明，当前的LLM频繁地输出受版权保护的文本，越狱攻击可以显著增加受版权保护的输出量。我们提出的防御机制通过有效地拒绝恶意请求，大大减少了LLMS生成的受版权保护的文本的数量。代码可在https://github.com/xz-liu/SHIELD上公开获得



## **6. Probing the Safety Response Boundary of Large Language Models via Unsafe Decoding Path Generation**

通过不安全解码路径生成探索大型语言模型的安全响应边界 cs.CR

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.10668v2) [paper-pdf](http://arxiv.org/pdf/2408.10668v2)

**Authors**: Haoyu Wang, Bingzhe Wu, Yatao Bian, Yongzhe Chang, Xueqian Wang, Peilin Zhao

**Abstract**: Large Language Models (LLMs) are implicit troublemakers. While they provide valuable insights and assist in problem-solving, they can also potentially serve as a resource for malicious activities. Implementing safety alignment could mitigate the risk of LLMs generating harmful responses. We argue that: even when an LLM appears to successfully block harmful queries, there may still be hidden vulnerabilities that could act as ticking time bombs. To identify these underlying weaknesses, we propose to use a cost value model as both a detector and an attacker. Trained on external or self-generated harmful datasets, the cost value model could successfully influence the original safe LLM to output toxic content in decoding process. For instance, LLaMA-2-chat 7B outputs 39.18% concrete toxic content, along with only 22.16% refusals without any harmful suffixes. These potential weaknesses can then be exploited via prompt optimization such as soft prompts on images. We name this decoding strategy: Jailbreak Value Decoding (JVD), emphasizing that seemingly secure LLMs may not be as safe as we initially believe. They could be used to gather harmful data or launch covert attacks.

摘要: 大型语言模型(LLM)是隐含的麻烦制造者。虽然它们提供了有价值的见解并帮助解决问题，但它们也可能成为恶意活动的来源。实施安全调整可以降低低密度脂蛋白产生有害反应的风险。我们认为：即使LLM似乎成功阻止了有害查询，仍可能存在隐藏的漏洞，这些漏洞可能会充当定时炸弹。为了识别这些潜在的弱点，我们建议使用成本价值模型作为检测器和攻击者。代价值模型在外部或自身产生的有害数据集上进行训练，可以成功地影响原始安全LLM在解码过程中输出有毒内容。例如，骆驼-2-Chat 7B输出39.18%的具体有毒内容，以及只有22.16%的拒绝没有任何有害的后缀。然后可以通过提示优化(如图像上的软提示)来利用这些潜在的弱点。我们将这种解码策略命名为：越狱价值解码(JVD)，强调看似安全的LLM可能并不像我们最初认为的那样安全。它们可能被用来收集有害数据或发动秘密攻击。



## **7. EEG-Defender: Defending against Jailbreak through Early Exit Generation of Large Language Models**

EEG-Defender：通过早期退出生成大型语言模型来抵御越狱 cs.AI

19 pages, 7 figures

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11308v1) [paper-pdf](http://arxiv.org/pdf/2408.11308v1)

**Authors**: Chongwen Zhao, Zhihao Dou, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation. In an effort to mitigate such risks, the concept of "Alignment" technology has been developed. However, recent studies indicate that this alignment can be undermined using sophisticated prompt engineering or adversarial suffixes, a technique known as "Jailbreak." Our research takes cues from the human-like generate process of LLMs. We identify that while jailbreaking prompts may yield output logits similar to benign prompts, their initial embeddings within the model's latent space tend to be more analogous to those of malicious prompts. Leveraging this finding, we propose utilizing the early transformer outputs of LLMs as a means to detect malicious inputs, and terminate the generation immediately. Built upon this idea, we introduce a simple yet significant defense approach called EEG-Defender for LLMs. We conduct comprehensive experiments on ten jailbreak methods across three models. Our results demonstrate that EEG-Defender is capable of reducing the Attack Success Rate (ASR) by a significant margin, roughly 85\% in comparison with 50\% for the present SOTAs, with minimal impact on the utility and effectiveness of LLMs.

摘要: 大语言模型在各种应用中日益引起人们的关注。尽管如此，随着一些用户试图利用这些模型达到恶意目的，包括合成受控物质和传播虚假信息，人们越来越担心。为了减轻这种风险，人们提出了“对准”技术的概念。然而，最近的研究表明，这种对齐可以使用复杂的即时工程或敌对后缀来破坏，这是一种被称为“越狱”的技术。我们的研究从LLMS类似人类的生成过程中获得了线索。我们发现，虽然越狱提示可能会产生类似于良性提示的输出日志，但它们在模型潜在空间中的初始嵌入往往更类似于恶意提示。利用这一发现，我们建议利用LLMS的早期变压器输出作为一种手段来检测恶意输入，并立即终止生成。基于这一想法，我们介绍了一种简单但重要的防御方法，称为用于LLMS的EEG-Defender。我们在三个模型上对十种越狱方法进行了全面的实验。我们的结果表明，EEG-Defender能够显著降低攻击成功率(ASR)，与现有SOTAS的50%相比，约为85%，而对LLMS的实用性和有效性的影响最小。



## **8. Medical MLLM is Vulnerable: Cross-Modality Jailbreak and Mismatched Attacks on Medical Multimodal Large Language Models**

医学MLLM很脆弱：跨模式越狱和对医学多模式大型语言模型的不匹配攻击 cs.CR

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2405.20775v2) [paper-pdf](http://arxiv.org/pdf/2405.20775v2)

**Authors**: Xijie Huang, Xinyuan Wang, Hantao Zhang, Yinghao Zhu, Jiawen Xi, Jingkun An, Hao Wang, Hao Liang, Chengwei Pan

**Abstract**: Security concerns related to Large Language Models (LLMs) have been extensively explored, yet the safety implications for Multimodal Large Language Models (MLLMs), particularly in medical contexts (MedMLLMs), remain insufficiently studied. This paper delves into the underexplored security vulnerabilities of MedMLLMs, especially when deployed in clinical environments where the accuracy and relevance of question-and-answer interactions are critically tested against complex medical challenges. By combining existing clinical medical data with atypical natural phenomena, we define the mismatched malicious attack (2M-attack) and introduce its optimized version, known as the optimized mismatched malicious attack (O2M-attack or 2M-optimization). Using the voluminous 3MAD dataset that we construct, which covers a wide range of medical image modalities and harmful medical scenarios, we conduct a comprehensive analysis and propose the MCM optimization method, which significantly enhances the attack success rate on MedMLLMs. Evaluations with this dataset and attack methods, including white-box attacks on LLaVA-Med and transfer attacks (black-box) on four other SOTA models, indicate that even MedMLLMs designed with enhanced security features remain vulnerable to security breaches. Our work underscores the urgent need for a concerted effort to implement robust security measures and enhance the safety and efficacy of open-source MedMLLMs, particularly given the potential severity of jailbreak attacks and other malicious or clinically significant exploits in medical settings. Our code is available at https://github.com/dirtycomputer/O2M_attack.

摘要: 与大语言模型(LLM)相关的安全问题已经得到了广泛的研究，但多模式大语言模型(MLLM)的安全影响，特别是在医学背景下(MedMLLMS)的安全影响，仍然没有得到充分的研究。本文深入研究了MedMLLMS未被开发的安全漏洞，特别是当部署在临床环境中时，其中问答交互的准确性和相关性针对复杂的医疗挑战进行了严格的测试。结合已有的临床医学数据和非典型自然现象，我们定义了失配恶意攻击(2M-Attack)，并介绍了其优化版本，称为优化失配恶意攻击(O2M-Attack或2M-Optimation)。利用我们构建的涵盖多种医学图像模式和有害医疗场景的海量3MAD数据集，进行了全面的分析，并提出了MCM优化方法，显著提高了对MedMLLms的攻击成功率。使用该数据集和攻击方法进行的评估，包括对LLaVA-Med的白盒攻击和对其他四个SOTA型号的传输攻击(黑盒)，表明即使是设计了增强安全功能的MedMLLM也仍然容易受到安全漏洞的攻击。我们的工作强调了迫切需要共同努力，实施强有力的安全措施，提高开源MedMLLMS的安全性和有效性，特别是考虑到越狱攻击和医疗环境中其他恶意或具有临床意义的利用的潜在严重性。我们的代码可以在https://github.com/dirtycomputer/O2M_attack.上找到



## **9. Hide Your Malicious Goal Into Benign Narratives: Jailbreak Large Language Models through Neural Carrier Articles**

将你的恶意目标隐藏在良性叙述中：通过神经载体文章越狱大型语言模型 cs.CR

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.11182v1) [paper-pdf](http://arxiv.org/pdf/2408.11182v1)

**Authors**: Zhilong Wang, Haizhou Wang, Nanqing Luo, Lan Zhang, Xiaoyan Sun, Yebo Cao, Peng Liu

**Abstract**: Jailbreak attacks on Language Model Models (LLMs) entail crafting prompts aimed at exploiting the models to generate malicious content. This paper proposes a new type of jailbreak attacks which shift the attention of the LLM by inserting a prohibited query into a carrier article. The proposed attack leverage the knowledge graph and a composer LLM to automatically generating a carrier article that is similar to the topic of the prohibited query but does not violate LLM's safeguards. By inserting the malicious query to the carrier article, the assembled attack payload can successfully jailbreak LLM. To evaluate the effectiveness of our method, we leverage 4 popular categories of ``harmful behaviors'' adopted by related researches to attack 6 popular LLMs. Our experiment results show that the proposed attacking method can successfully jailbreak all the target LLMs which high success rate, except for Claude-3.

摘要: 对语言模型模型（LLM）的越狱攻击需要精心设计旨在利用模型生成恶意内容的提示。本文提出了一种新型越狱攻击，通过在载体文章中插入被禁止的查询来转移LLM的注意力。拟议的攻击利用知识图和作曲家LLM来自动生成与禁止查询的主题相似但不违反LLM的保障措施的载体文章。通过将恶意查询插入到载体文章中，组装的攻击有效负载可以成功越狱LLM。为了评估我们方法的有效性，我们利用相关研究采用的4种流行类别“有害行为”来攻击6种流行的LLM。实验结果表明，所提出的攻击方法可以成功越狱除Claude-3外的所有成功率较高的目标LLM。



## **10. Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks**

羊皮中的假新闻：针对LLM授权的风格攻击的强大假新闻检测 cs.CL

Accepted to KDD 2024 (Research Track)

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2310.10830v2) [paper-pdf](http://arxiv.org/pdf/2310.10830v2)

**Authors**: Jiaying Wu, Jiafeng Guo, Bryan Hooi

**Abstract**: It is commonly perceived that fake news and real news exhibit distinct writing styles, such as the use of sensationalist versus objective language. However, we emphasize that style-related features can also be exploited for style-based attacks. Notably, the advent of powerful Large Language Models (LLMs) has empowered malicious actors to mimic the style of trustworthy news sources, doing so swiftly, cost-effectively, and at scale. Our analysis reveals that LLM-camouflaged fake news content significantly undermines the effectiveness of state-of-the-art text-based detectors (up to 38% decrease in F1 Score), implying a severe vulnerability to stylistic variations. To address this, we introduce SheepDog, a style-robust fake news detector that prioritizes content over style in determining news veracity. SheepDog achieves this resilience through (1) LLM-empowered news reframings that inject style diversity into the training process by customizing articles to match different styles; (2) a style-agnostic training scheme that ensures consistent veracity predictions across style-diverse reframings; and (3) content-focused veracity attributions that distill content-centric guidelines from LLMs for debunking fake news, offering supplementary cues and potential intepretability that assist veracity prediction. Extensive experiments on three real-world benchmarks demonstrate SheepDog's style robustness and adaptability to various backbones.

摘要: 人们普遍认为，假新闻和真实新闻表现出截然不同的写作风格，例如使用耸人听闻的语言与客观语言。但是，我们强调，与样式相关的功能也可以用于基于样式的攻击。值得注意的是，强大的大型语言模型(LLM)的出现使恶意攻击者能够快速、经济、大规模地模仿可信新闻来源的风格。我们的分析显示，LLM伪装的假新闻内容显著削弱了最先进的基于文本的检测器的有效性(F1分数下降了38%)，这意味着对文体变化的严重脆弱性。为了解决这个问题，我们引入了SheepDog，这是一个风格稳健的假新闻检测器，在确定新闻真实性时，它将内容优先于风格。Sheepog通过以下方式实现这种弹性：(1)LLM授权的新闻重组，通过定制文章以匹配不同的风格，将风格多样性注入训练过程；(2)风格不可知的培训方案，确保在风格多样化的重组中一致的准确性预测；以及(3)以内容为中心的准确性归因，从LLMS中提取以内容为中心的指导方针，以揭穿假新闻，提供补充线索和潜在的可理解性，以帮助准确性预测。在三个真实世界的基准上进行的广泛实验表明，牧羊犬的风格、健壮性和对各种主干的适应性。



## **11. While GitHub Copilot Excels at Coding, Does It Ensure Responsible Output?**

虽然GitHub Copilot擅长编码，但它能否确保负责任的输出？ cs.CL

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.11006v1) [paper-pdf](http://arxiv.org/pdf/2408.11006v1)

**Authors**: Wen Cheng, Ke Sun, Xinyu Zhang, Wei Wang

**Abstract**: The rapid development of large language models (LLMs) has significantly advanced code completion capabilities, giving rise to a new generation of LLM-based Code Completion Tools (LCCTs). Unlike general-purpose LLMs, these tools possess unique workflows, integrating multiple information sources as input and prioritizing code suggestions over natural language interaction, which introduces distinct security challenges. Additionally, LCCTs often rely on proprietary code datasets for training, raising concerns about the potential exposure of sensitive data. This paper exploits these distinct characteristics of LCCTs to develop targeted attack methodologies on two critical security risks: jailbreaking and training data extraction attacks. Our experimental results expose significant vulnerabilities within LCCTs, including a 99.4% success rate in jailbreaking attacks on GitHub Copilot and a 46.3% success rate on Amazon Q. Furthermore, We successfully extracted sensitive user data from GitHub Copilot, including 54 real email addresses and 314 physical addresses associated with GitHub usernames. Our study also demonstrates that these code-based attack methods are effective against general-purpose LLMs, such as the GPT series, highlighting a broader security misalignment in the handling of code by modern LLMs. These findings underscore critical security challenges associated with LCCTs and suggest essential directions for strengthening their security frameworks. The example code and attack samples from our research are provided at https://github.com/Sensente/Security-Attacks-on-LCCTs.

摘要: 大型语言模型(LLM)的快速发展极大地提升了代码补全能力，催生了新一代基于LLM的代码补全工具(LCCT)。与通用的LLMS不同，这些工具拥有独特的工作流，将多个信息源集成为输入，并优先考虑代码建议而不是自然语言交互，这带来了明显的安全挑战。此外，LCCT经常依赖专有代码数据集进行培训，这引发了人们对敏感数据潜在暴露的担忧。针对越狱攻击和训练数据提取攻击这两个关键安全风险，本文利用LCCT的这些显著特点，提出了针对性的攻击方法。我们的实验结果暴露了LCCT中的重大漏洞，包括对GitHub Copilot的越狱攻击成功率为99.4%，对Amazon Q的成功率为46.3%。此外，我们成功地从GitHub Copilot中提取了敏感用户数据，包括与GitHub用户名关联的54个真实电子邮件地址和314个物理地址。我们的研究还表明，这些基于代码的攻击方法对通用LLM是有效的，例如GPT系列，突显了现代LLM在处理代码时存在更广泛的安全错位。这些调查结果强调了与土地利用、土地利用、土地退化和土地退化有关的重大安全挑战，并提出了加强其安全框架的基本方向。我们的研究提供了示例代码和攻击示例，请访问https://github.com/Sensente/Security-Attacks-on-LCCTs.



## **12. Towards Efficient Formal Verification of Spiking Neural Network**

尖峰神经网络的有效形式验证 cs.AI

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10900v1) [paper-pdf](http://arxiv.org/pdf/2408.10900v1)

**Authors**: Baekryun Seong, Jieung Kim, Sang-Ki Ko

**Abstract**: Recently, AI research has primarily focused on large language models (LLMs), and increasing accuracy often involves scaling up and consuming more power. The power consumption of AI has become a significant societal issue; in this context, spiking neural networks (SNNs) offer a promising solution. SNNs operate event-driven, like the human brain, and compress information temporally. These characteristics allow SNNs to significantly reduce power consumption compared to perceptron-based artificial neural networks (ANNs), highlighting them as a next-generation neural network technology. However, societal concerns regarding AI go beyond power consumption, with the reliability of AI models being a global issue. For instance, adversarial attacks on AI models are a well-studied problem in the context of traditional neural networks. Despite their importance, the stability and property verification of SNNs remains in the early stages of research. Most SNN verification methods are time-consuming and barely scalable, making practical applications challenging. In this paper, we introduce temporal encoding to achieve practical performance in verifying the adversarial robustness of SNNs. We conduct a theoretical analysis of this approach and demonstrate its success in verifying SNNs at previously unmanageable scales. Our contribution advances SNN verification to a practical level, facilitating the safer application of SNNs.

摘要: 最近，人工智能的研究主要集中在大型语言模型(LLM)上，而提高精确度往往需要扩大规模和消耗更多功率。人工智能的能耗已经成为一个重要的社会问题；在这种背景下，尖峰神经网络(SNN)提供了一个有前途的解决方案。SNN像人脑一样，以事件驱动的方式运行，并在时间上压缩信息。与基于感知器的人工神经网络(ANN)相比，这些特性使SNN能够显著降低功耗，突出了它们作为下一代神经网络技术的重要性。然而，社会对人工智能的担忧不仅仅是电力消耗，人工智能模型的可靠性是一个全球问题。例如，在传统神经网络的背景下，对人工智能模型的敌意攻击是一个研究得很好的问题。尽管它们很重要，但SNN的稳定性和性质验证仍处于研究的早期阶段。大多数SNN验证方法都很耗时且几乎不可扩展，这给实际应用带来了挑战。在本文中，我们引入时间编码以达到在验证SNN的对抗健壮性方面的实际性能。我们对这种方法进行了理论分析，并证明了它在以前无法管理的规模上验证SNN的成功。我们的贡献将SNN验证提升到了一个实用的水平，促进了SNN的更安全应用。



## **13. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection**

PhishAgent：一种用于网络钓鱼网页检测的鲁棒多模式代理 cs.CR

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10738v1) [paper-pdf](http://arxiv.org/pdf/2408.10738v1)

**Authors**: Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, Bryan Hooi

**Abstract**: Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also encounter notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the top k relevant items from offline knowledge bases, utilizing all available information from a webpage, including logos, HTML, and URLs. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.

摘要: 网络钓鱼攻击是在线安全的主要威胁，利用用户漏洞窃取敏感信息。已经开发了各种方法来对抗网络钓鱼，每一种方法的精确度都不同，但它们也遇到了显著的局限性。在本研究中，我们介绍了PhishAgent，一个结合了广泛工具的多通道代理，将线上和线下知识库与多通道大语言模型(MLLMS)相结合。这一组合导致了更广泛的品牌覆盖，从而提高了品牌认知度和召回率。此外，我们提出了一个多通道信息检索框架，旨在利用网页中的所有可用信息，包括徽标、HTML和URL，从离线知识库中提取前k个相关条目。基于三个真实数据集的实验结果表明，该框架在保持模型效率的同时，显著提高了检测准确率，减少了误报和漏报。此外，PhishAgent对各种类型的对抗性攻击表现出很强的韧性。



## **14. MEGen: Generative Backdoor in Large Language Models via Model Editing**

MEGen：通过模型编辑在大型语言模型中实现生成后门 cs.CL

Working in progress

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10722v1) [paper-pdf](http://arxiv.org/pdf/2408.10722v1)

**Authors**: Jiyang Qiu, Xinbei Ma, Zhuosheng Zhang, Hai Zhao

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities. Their powerful generative abilities enable flexible responses based on various queries or instructions. Emerging as widely adopted generalists for diverse tasks, LLMs are still vulnerable to backdoors. This paper proposes an editing-based generative backdoor, named MEGen, aiming to create a customized backdoor for NLP tasks with the least side effects. In our approach, we first leverage a language model to insert a trigger selected on fixed metrics into the input, then design a pipeline of model editing to directly embed a backdoor into an LLM. By adjusting a small set of local parameters with a mini-batch of samples, MEGen significantly enhances time efficiency and achieves high robustness. Experimental results indicate that our backdoor attack strategy achieves a high attack success rate on poison data while maintaining the model's performance on clean data. Notably, the backdoored model, when triggered, can freely output pre-set dangerous information while successfully completing downstream tasks. This suggests that future LLM applications could be guided to deliver certain dangerous information, thus altering the LLM's generative style. We believe this approach provides insights for future LLM applications and the execution of backdoor attacks on conversational AI systems.

摘要: 大型语言模型(LLM)已经显示出非凡的能力。它们强大的生成能力使人们能够根据各种查询或指令做出灵活的反应。作为在不同任务中被广泛采用的多面手，LLM仍很容易受到后门的攻击。本文提出了一种基于编辑的生成性后门Megen，旨在为NLP任务创建一个具有最小副作用的定制后门。在我们的方法中，我们首先利用语言模型将根据固定指标选择的触发器插入到输入中，然后设计一个模型编辑管道来直接将后门嵌入到LLM中。通过用一小批样本调整一小组局部参数，Megen显著提高了时间效率并实现了高稳健性。实验结果表明，我们的后门攻击策略在保持模型对干净数据的性能的同时，对有毒数据取得了较高的攻击成功率。值得注意的是，后置模型在被触发时，可以在成功完成下游任务的同时自由输出预设的危险信息。这表明，未来的LLM应用程序可能会被引导来传递某些危险的信息，从而改变LLM的生成风格。我们相信，这种方法为未来的LLM应用程序和对对话式人工智能系统执行后门攻击提供了见解。



## **15. Ferret: Faster and Effective Automated Red Teaming with Reward-Based Scoring Technique**

Ferret：更快、更有效的自动化红色团队，采用基于奖励的评分技术 cs.CL

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10701v1) [paper-pdf](http://arxiv.org/pdf/2408.10701v1)

**Authors**: Tej Deep Pala, Vernon Y. H. Toh, Rishabh Bhardwaj, Soujanya Poria

**Abstract**: In today's era, where large language models (LLMs) are integrated into numerous real-world applications, ensuring their safety and robustness is crucial for responsible AI usage. Automated red-teaming methods play a key role in this process by generating adversarial attacks to identify and mitigate potential vulnerabilities in these models. However, existing methods often struggle with slow performance, limited categorical diversity, and high resource demands. While Rainbow Teaming, a recent approach, addresses the diversity challenge by framing adversarial prompt generation as a quality-diversity search, it remains slow and requires a large fine-tuned mutator for optimal performance. To overcome these limitations, we propose Ferret, a novel approach that builds upon Rainbow Teaming by generating multiple adversarial prompt mutations per iteration and using a scoring function to rank and select the most effective adversarial prompt. We explore various scoring functions, including reward models, Llama Guard, and LLM-as-a-judge, to rank adversarial mutations based on their potential harm to improve the efficiency of the search for harmful mutations. Our results demonstrate that Ferret, utilizing a reward model as a scoring function, improves the overall attack success rate (ASR) to 95%, which is 46% higher than Rainbow Teaming. Additionally, Ferret reduces the time needed to achieve a 90% ASR by 15.2% compared to the baseline and generates adversarial prompts that are transferable i.e. effective on other LLMs of larger size. Our codes are available at https://github.com/declare-lab/ferret.

摘要: 在当今时代，大型语言模型(LLM)被集成到许多现实世界的应用程序中，确保它们的安全性和健壮性对于负责任的人工智能使用至关重要。自动红团队方法通过生成对抗性攻击来识别和缓解这些模型中的潜在漏洞，从而在这一过程中发挥关键作用。然而，现有的方法往往在性能缓慢、分类多样性有限和资源需求高的情况下苦苦挣扎。虽然彩虹组合是最近的一种方法，通过将敌意提示生成框定为一种质量多样性搜索来解决多样性挑战，但它仍然很慢，需要一个大型微调赋值器来实现最佳性能。为了克服这些局限性，我们提出了一种新的方法--FERRET，它建立在彩虹分组的基础上，通过每次迭代产生多个对抗性提示突变，并使用评分函数来对最有效的对抗性提示进行排序和选择。我们探索了各种评分函数，包括奖励模型、骆驼警卫和LLM作为法官，根据潜在的危害对对手突变进行排名，以提高有害突变的搜索效率。我们的结果表明，利用奖励模型作为得分函数的雪貂，将总体攻击成功率(ASR)提高到95%，比彩虹组合高出46%。此外，与基线相比，雪貂将达到90%的ASR所需的时间减少了15.2%，并生成可转移的对抗性提示，即对其他较大规模的LLM有效。我们的代码可在https://github.com/declare-lab/ferret.上获得



## **16. Promoting Equality in Large Language Models: Identifying and Mitigating the Implicit Bias based on Bayesian Theory**

促进大型语言模型中的平等：基于Bayesian理论识别和缓解隐性偏见 cs.CL

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10608v1) [paper-pdf](http://arxiv.org/pdf/2408.10608v1)

**Authors**: Yongxin Deng, Xihe Qiu, Xiaoyu Tan, Jing Pan, Chen Jue, Zhijun Fang, Yinghui Xu, Wei Chu, Yuan Qi

**Abstract**: Large language models (LLMs) are trained on extensive text corpora, which inevitably include biased information. Although techniques such as Affective Alignment can mitigate some negative impacts of these biases, existing prompt-based attack methods can still extract these biases from the model's weights. Moreover, these biases frequently appear subtly when LLMs are prompted to perform identical tasks across different demographic groups, thereby camouflaging their presence. To address this issue, we have formally defined the implicit bias problem and developed an innovative framework for bias removal based on Bayesian theory, Bayesian-Theory based Bias Removal (BTBR). BTBR employs likelihood ratio screening to pinpoint data entries within publicly accessible biased datasets that represent biases inadvertently incorporated during the LLM training phase. It then automatically constructs relevant knowledge triples and expunges bias information from LLMs using model editing techniques. Through extensive experimentation, we have confirmed the presence of the implicit bias problem in LLMs and demonstrated the effectiveness of our BTBR approach.

摘要: 大型语言模型(LLM)是在广泛的文本语料库上训练的，其中不可避免地包含有偏见的信息。虽然情感对齐等技术可以缓解这些偏差的一些负面影响，但现有的基于提示的攻击方法仍然可以从模型的权重中提取这些偏差。此外，当LLM被提示在不同的人口群体中执行相同的任务时，这些偏见经常微妙地出现，从而掩盖了他们的存在。为了解决这个问题，我们正式定义了隐含偏差问题，并在贝叶斯理论的基础上提出了一种新的去偏框架--基于贝叶斯理论的去偏方法(BTBR)。BTBR使用似然比筛选来精确定位可公开访问的有偏数据集中的数据条目，这些数据表示在LLM训练阶段无意中并入的偏差。然后利用模型编辑技术自动构建相关知识三元组，并从低似然模型中剔除偏差信息。通过大量的实验，我们证实了LLMS中隐含偏差问题的存在，并证明了我们的BTBR方法的有效性。



## **17. PromptBench: A Unified Library for Evaluation of Large Language Models**

EntBench：大型语言模型评估的统一库 cs.AI

Accepted by Journal of Machine Learning Research (JMLR); code:  https://github.com/microsoft/promptbench

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2312.07910v3) [paper-pdf](http://arxiv.org/pdf/2312.07910v3)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.

摘要: 大型语言模型（LLM）的评估对于评估其性能和降低潜在的安全风险至关重要。本文中，我们介绍了EmotiBench，这是一个评估LLM的统一图书馆。它由研究人员易于使用和扩展的几个关键组件组成：即时构建、即时工程、数据集和模型加载、对抗即时攻击、动态评估协议和分析工具。EntBench旨在成为一个开放、通用和灵活的代码库，用于研究目的，可以促进创建新基准、部署下游应用程序和设计新评估协议的原始研究。该代码可在https://github.com/microsoft/promptbench上获取，并将持续支持。



## **18. Development of an AI Anti-Bullying System Using Large Language Model Key Topic Detection**

基于大语言模型关键话题检测的人工智能反欺凌系统开发 cs.AI

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.10417v1) [paper-pdf](http://arxiv.org/pdf/2408.10417v1)

**Authors**: Matthew Tassava, Cameron Kolodjski, Jordan Milbrath, Adorah Bishop, Nathan Flanders, Robbie Fetsch, Danielle Hanson, Jeremy Straub

**Abstract**: This paper presents and evaluates work on the development of an artificial intelligence (AI) anti-bullying system. The system is designed to identify coordinated bullying attacks via social media and other mechanisms, characterize them and propose remediation and response activities to them. In particular, a large language model (LLM) is used to populate an enhanced expert system-based network model of a bullying attack. This facilitates analysis and remediation activity - such as generating report messages to social media companies - determination. The system is described and the efficacy of the LLM for populating the model is analyzed herein.

摘要: 本文介绍并评估了人工智能（AI）反欺凌系统的开发工作。该系统旨在识别通过社交媒体和其他机制的协调欺凌攻击，对其进行特征描述并提出补救和响应活动。特别是，使用大型语言模型（LLM）来填充欺凌攻击的增强型基于专家系统的网络模型。这有助于分析和补救活动（例如向社交媒体公司生成报告消息）的确定。本文描述了该系统，并分析了LLM填充模型的功效。



## **19. A Disguised Wolf Is More Harmful Than a Toothless Tiger: Adaptive Malicious Code Injection Backdoor Attack Leveraging User Behavior as Triggers**

伪装的狼比无牙的老虎更有害：利用用户行为作为触发器的自适应恶意代码注入后门攻击 cs.AI

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.10334v1) [paper-pdf](http://arxiv.org/pdf/2408.10334v1)

**Authors**: Shangxi Wu, Jitao Sang

**Abstract**: In recent years, large language models (LLMs) have made significant progress in the field of code generation. However, as more and more users rely on these models for software development, the security risks associated with code generation models have become increasingly significant. Studies have shown that traditional deep learning robustness issues also negatively impact the field of code generation. In this paper, we first present the game-theoretic model that focuses on security issues in code generation scenarios. This framework outlines possible scenarios and patterns where attackers could spread malicious code models to create security threats. We also pointed out for the first time that the attackers can use backdoor attacks to dynamically adjust the timing of malicious code injection, which will release varying degrees of malicious code depending on the skill level of the user. Through extensive experiments on leading code generation models, we validate our proposed game-theoretic model and highlight the significant threats that these new attack scenarios pose to the safe use of code models.

摘要: 近年来，大型语言模型(LLM)在代码生成领域取得了重大进展。然而，随着越来越多的用户依赖这些模型进行软件开发，与代码生成模型相关的安全风险也变得越来越严重。研究表明，传统的深度学习健壮性问题也对代码生成领域产生了负面影响。在本文中，我们首先提出了关注代码生成场景中的安全问题的博弈论模型。此框架概述了攻击者传播恶意代码模型以制造安全威胁的可能场景和模式。我们还首次指出，攻击者可以利用后门攻击来动态调整恶意代码注入的时间，这将根据用户的技能水平释放不同程度的恶意代码。通过在领先的代码生成模型上的大量实验，我们验证了我们提出的博弈论模型，并强调了这些新的攻击场景对代码模型的安全使用构成的重大威胁。



## **20. Topic-Based Watermarks for LLM-Generated Text**

LLM生成文本的基于主题的水印 cs.CR

Results for proposed scheme, additional/removal of content (figures  and equations), 12 pages

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2404.02138v3) [paper-pdf](http://arxiv.org/pdf/2404.02138v3)

**Authors**: Alexander Nemecek, Yuzhou Jiang, Erman Ayday

**Abstract**: The indistinguishability of text generated by large language models (LLMs) from human-generated text poses significant challenges. Watermarking algorithms are potential solutions by embedding detectable signatures within LLM-generated outputs. However, current watermarking schemes lack robustness to a range of attacks such as text substitution or manipulation, undermining their reliability. This paper proposes a novel topic-based watermarking algorithm for LLMs, designed to enhance the robustness of watermarking in LLMs. Our approach leverages the topics extracted from input prompts or outputs of non-watermarked LLMs in the generation process of watermarked text. We dynamically utilize token lists on identified topics and adjust token sampling weights accordingly. By using these topic-specific token biases, we embed a topic-sensitive watermarking into the generated text. We outline the theoretical framework of our topic-based watermarking algorithm and discuss its potential advantages in various scenarios. Additionally, we explore a comprehensive range of attacks against watermarking algorithms, including discrete alterations, paraphrasing, and tokenizations. We demonstrate that our proposed watermarking scheme classifies various watermarked text topics with 99.99% confidence and outperforms existing algorithms in terms of z-score robustness and the feasibility of modeling text degradation by potential attackers, while considering the trade-offs between the benefits and losses of watermarking LLM-generated text.

摘要: 大型语言模型(LLM)生成的文本与人类生成的文本无法区分，这带来了巨大的挑战。通过在LLM生成的输出中嵌入可检测的签名，水印算法是潜在的解决方案。然而，当前的水印方案缺乏对文本替换或篡改等一系列攻击的稳健性，从而破坏了它们的可靠性。提出了一种新的基于主题的LLMS水印算法，旨在增强LLMS中水印的稳健性。我们的方法在水印文本的生成过程中利用了从非水印LLMS的输入提示或输出中提取的主题。我们在识别的主题上动态地利用令牌列表，并相应地调整令牌抽样权重。通过使用这些特定于主题的标记偏差，我们在生成的文本中嵌入了主题敏感的水印。我们概述了我们的基于主题的水印算法的理论框架，并讨论了它在各种场景下的潜在优势。此外，我们还探讨了针对水印算法的广泛攻击，包括离散更改、释义和标记化。我们证明了我们提出的水印方案以99.99%的置信度对不同的水印文本主题进行分类，并且在z-Score稳健性和潜在攻击者对文本退化建模的可行性方面优于现有算法，同时考虑了在LLM生成的文本中添加水印的利弊权衡。



## **21. Privacy Checklist: Privacy Violation Detection Grounding on Contextual Integrity Theory**

隐私检查表：基于上下文完整性理论的隐私侵犯检测 cs.CL

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.10053v1) [paper-pdf](http://arxiv.org/pdf/2408.10053v1)

**Authors**: Haoran Li, Wei Fan, Yulin Chen, Jiayang Cheng, Tianshu Chu, Xuebing Zhou, Peizhao Hu, Yangqiu Song

**Abstract**: Privacy research has attracted wide attention as individuals worry that their private data can be easily leaked during interactions with smart devices, social platforms, and AI applications. Computer science researchers, on the other hand, commonly study privacy issues through privacy attacks and defenses on segmented fields. Privacy research is conducted on various sub-fields, including Computer Vision (CV), Natural Language Processing (NLP), and Computer Networks. Within each field, privacy has its own formulation. Though pioneering works on attacks and defenses reveal sensitive privacy issues, they are narrowly trapped and cannot fully cover people's actual privacy concerns. Consequently, the research on general and human-centric privacy research remains rather unexplored. In this paper, we formulate the privacy issue as a reasoning problem rather than simple pattern matching. We ground on the Contextual Integrity (CI) theory which posits that people's perceptions of privacy are highly correlated with the corresponding social context. Based on such an assumption, we develop the first comprehensive checklist that covers social identities, private attributes, and existing privacy regulations. Unlike prior works on CI that either cover limited expert annotated norms or model incomplete social context, our proposed privacy checklist uses the whole Health Insurance Portability and Accountability Act of 1996 (HIPAA) as an example, to show that we can resort to large language models (LLMs) to completely cover the HIPAA's regulations. Additionally, our checklist also gathers expert annotations across multiple ontologies to determine private information including but not limited to personally identifiable information (PII). We use our preliminary results on the HIPAA to shed light on future context-centric privacy research to cover more privacy regulations, social norms and standards.

摘要: 隐私研究吸引了广泛的关注，因为个人担心他们的私人数据在与智能设备、社交平台和人工智能应用程序交互时很容易被泄露。另一方面，计算机科学研究人员通常通过对分割的领域进行隐私攻击和防御来研究隐私问题。隐私研究在不同的子领域进行，包括计算机视觉(CV)、自然语言处理(NLP)和计算机网络。在每个领域，隐私都有自己的表述。尽管攻击和防御方面的开创性作品揭示了敏感的隐私问题，但它们被狭隘地困住了，不能完全覆盖人们对隐私的实际担忧。因此，关于一般隐私研究和以人为中心的隐私研究仍然是相当未被探索的。在本文中，我们将隐私问题描述为一个推理问题，而不是简单的模式匹配。我们基于语境完整性(CI)理论，该理论认为人们对隐私的感知与相应的社会语境高度相关。基于这样的假设，我们开发了第一个全面的清单，其中包括社会身份、私人属性和现有的隐私法规。与以往关于CI的工作要么涵盖有限的专家注释规范，要么涵盖不完整的社会背景，我们提出的隐私检查表以1996年的整个健康保险携带和责任法案(HIPAA)为例，表明我们可以求助于大型语言模型(LLM)来完全覆盖HIPAA的规定。此外，我们的检查表还收集了跨多个本体的专家注释，以确定私人信息，包括但不限于个人身份信息(PII)。我们使用我们在HIPAA上的初步结果来阐明未来以上下文为中心的隐私研究，以涵盖更多的隐私法规、社会规范和标准。



## **22. Transferring Backdoors between Large Language Models by Knowledge Distillation**

通过知识蒸馏在大型语言模型之间转移后门 cs.CR

13 pages, 16 figures, 5 tables

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.09878v1) [paper-pdf](http://arxiv.org/pdf/2408.09878v1)

**Authors**: Pengzhou Cheng, Zongru Wu, Tianjie Ju, Wei Du, Zhuosheng Zhang Gongshen Liu

**Abstract**: Backdoor Attacks have been a serious vulnerability against Large Language Models (LLMs). However, previous methods only reveal such risk in specific models, or present tasks transferability after attacking the pre-trained phase. So, how risky is the model transferability of a backdoor attack? In this paper, we focus on whether existing mini-LLMs may be unconsciously instructed in backdoor knowledge by poisoned teacher LLMs through knowledge distillation (KD). Specifically, we propose ATBA, an adaptive transferable backdoor attack, which can effectively distill the backdoor of teacher LLMs into small models when only executing clean-tuning. We first propose the Target Trigger Generation (TTG) module that filters out a set of indicative trigger candidates from the token list based on cosine similarity distribution. Then, we exploit a shadow model to imitate the distilling process and introduce an Adaptive Trigger Optimization (ATO) module to realize a gradient-based greedy feedback to search optimal triggers. Extensive experiments show that ATBA generates not only positive guidance for student models but also implicitly transfers backdoor knowledge. Our attack is robust and stealthy, with over 80% backdoor transferability, and hopes the attention of security.

摘要: 后门攻击一直是大型语言模型(LLM)的一个严重漏洞。然而，以前的方法只在特定的模型中发现这种风险，或者在攻击预训练阶段之后呈现任务的可转移性。那么，后门攻击的模型可转移性有多大风险呢？在本文中，我们关注现有的微型LLM是否可能被中毒的教师LLMS通过知识蒸馏(KD)无意识地传授到后门知识。具体地说，我们提出了一种自适应可转移后门攻击ATBA，它可以在只执行干净调优的情况下有效地将教师LLM的后门提取成小模型。我们首先提出了目标触发器生成(TTG)模块，该模块根据余弦相似度分布从令牌列表中过滤出一组指示性触发器候选。然后，我们利用影子模型来模拟提取过程，并引入自适应触发优化(ATO)模块来实现基于梯度的贪婪反馈来搜索最优触发。大量的实验表明，ATBA不仅为学生模型生成了积极的指导，而且还隐式地传递了后门知识。我们的攻击具有健壮性和隐蔽性，具有80%以上的后门可转移性，希望引起安全方面的关注。



## **23. Bergeron: Combating Adversarial Attacks through a Conscience-Based Alignment Framework**

Bergeron：通过基于意识的一致框架打击敌对攻击 cs.CR

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2312.00029v3) [paper-pdf](http://arxiv.org/pdf/2312.00029v3)

**Authors**: Matthew Pisano, Peter Ly, Abraham Sanders, Bingsheng Yao, Dakuo Wang, Tomek Strzalkowski, Mei Si

**Abstract**: Research into AI alignment has grown considerably since the recent introduction of increasingly capable Large Language Models (LLMs). Unfortunately, modern methods of alignment still fail to fully prevent harmful responses when models are deliberately attacked. Such vulnerabilities can lead to LLMs being manipulated into generating hazardous content: from instructions for creating dangerous materials to inciting violence or endorsing unethical behaviors. To help mitigate this issue, we introduce Bergeron: a framework designed to improve the robustness of LLMs against attacks without any additional parameter fine-tuning. Bergeron is organized into two tiers; with a secondary LLM acting as a guardian to the primary LLM. This framework better safeguards the primary model against incoming attacks while monitoring its output for any harmful content. Empirical analysis reviews that by using Bergeron to complement models with existing alignment training, we can significantly improve the robustness and safety of multiple, commonly used commercial and open-source LLMs. Specifically, we found that models integrated with Bergeron are, on average, nearly seven times more resistant to attacks compared to models without such support.

摘要: 自从最近引入了功能越来越强大的大型语言模型(LLM)以来，对人工智能对齐的研究有了很大的增长。不幸的是，现代的校准方法仍然不能完全防止模型受到故意攻击时的有害反应。这些漏洞可能导致LLMS被操纵来生成危险内容：从创建危险材料的说明到煽动暴力或支持不道德行为。为了帮助缓解这个问题，我们引入了Bergeron：一个旨在提高LLM抵御攻击的健壮性的框架，而不需要任何额外的参数微调。Bergeron被组织成两级；辅助LLM充当主要LLM的监护人。此框架可以更好地保护主要模型免受来袭攻击，同时监控其输出中是否有任何有害内容。经验分析认为，通过使用Bergeron来补充模型与现有的比对训练，我们可以显著提高多个常用的商业和开源LLM的稳健性和安全性。具体地说，我们发现，与没有这种支持的型号相比，集成了Bergeron的型号平均抵抗攻击的能力要高出近7倍。



## **24. Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning**

解药：微调后大型语言模型的安全调整，防止有害的微调 cs.AI

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09600v1) [paper-pdf](http://arxiv.org/pdf/2408.09600v1)

**Authors**: Tiansheng Huang, Gautam Bhattacharya, Pratik Joshi, Josh Kimball, Ling Liu

**Abstract**: Safety aligned Large Language Models (LLMs) are vulnerable to harmful fine-tuning attacks \cite{qi2023fine}-- a few harmful data mixed in the fine-tuning dataset can break the LLMs's safety alignment. Existing mitigation strategies include alignment stage solutions \cite{huang2024vaccine, rosati2024representation} and fine-tuning stage solutions \cite{huang2024lazy,mukhoti2023fine}. However, our evaluation shows that both categories of defenses fail \textit{when some specific training hyper-parameters are chosen} -- a large learning rate or a large number of training epochs in the fine-tuning stage can easily invalidate the defense, which however, is necessary to guarantee finetune performance. To this end, we propose Antidote, a post-fine-tuning stage solution, which remains \textbf{\textit{agnostic to the training hyper-parameters in the fine-tuning stage}}. Antidote relies on the philosophy that by removing the harmful parameters, the harmful model can be recovered from the harmful behaviors, regardless of how those harmful parameters are formed in the fine-tuning stage. With this philosophy, we introduce a one-shot pruning stage after harmful fine-tuning to remove the harmful weights that are responsible for the generation of harmful content. Despite its embarrassing simplicity, empirical results show that Antidote can reduce harmful score while maintaining accuracy on downstream tasks.

摘要: 安全对齐的大型语言模型(LLM)容易受到有害的微调攻击--在微调数据集中混合一些有害数据就会破坏LLMS的安全对齐。现有的缓解策略包括对齐阶段解决方案\cite{huang2024疫苗，rosati2024代表}和微调阶段解决方案\cite{huang2024 lazy，mukhoti2023 finy}。然而，我们的评估表明，这两类防御都失败了[当选择了一些特定的训练超参数}--在微调阶段，较大的学习速率或大量的训练周期很容易使防御失效，但这是保证精调性能所必需的。为此，我们提出了解毒剂，这是一种后微调阶段的解决方案，它仍然保持在文本bf{与微调阶段的训练超参数无关}}。解毒剂依靠的理念是，通过删除有害参数，可以从有害行为中恢复有害模型，而无论这些有害参数在微调阶段是如何形成的。本着这一理念，我们引入了有害微调后的一次修剪阶段，以去除导致有害内容生成的有害权重。尽管解毒剂简单得令人尴尬，但实验结果表明，解毒剂可以降低有害分数，同时保持下游任务的准确性。



## **25. Characterizing and Evaluating the Reliability of LLMs against Jailbreak Attacks**

描述和评估LLM针对越狱攻击的可靠性 cs.CL

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09326v1) [paper-pdf](http://arxiv.org/pdf/2408.09326v1)

**Authors**: Kexin Chen, Yi Liu, Dongxia Wang, Jiaying Chen, Wenhai Wang

**Abstract**: Large Language Models (LLMs) have increasingly become pivotal in content generation with notable societal impact. These models hold the potential to generate content that could be deemed harmful.Efforts to mitigate this risk include implementing safeguards to ensure LLMs adhere to social ethics.However, despite such measures, the phenomenon of "jailbreaking" -- where carefully crafted prompts elicit harmful responses from models -- persists as a significant challenge. Recognizing the continuous threat posed by jailbreaking tactics and their repercussions for the trustworthy use of LLMs, a rigorous assessment of the models' robustness against such attacks is essential. This study introduces an comprehensive evaluation framework and conducts an large-scale empirical experiment to address this need. We concentrate on 10 cutting-edge jailbreak strategies across three categories, 1525 questions from 61 specific harmful categories, and 13 popular LLMs. We adopt multi-dimensional metrics such as Attack Success Rate (ASR), Toxicity Score, Fluency, Token Length, and Grammatical Errors to thoroughly assess the LLMs' outputs under jailbreak. By normalizing and aggregating these metrics, we present a detailed reliability score for different LLMs, coupled with strategic recommendations to reduce their susceptibility to such vulnerabilities. Additionally, we explore the relationships among the models, attack strategies, and types of harmful content, as well as the correlations between the evaluation metrics, which proves the validity of our multifaceted evaluation framework. Our extensive experimental results demonstrate a lack of resilience among all tested LLMs against certain strategies, and highlight the need to concentrate on the reliability facets of LLMs. We believe our study can provide valuable insights into enhancing the security evaluation of LLMs against jailbreak within the domain.

摘要: 大型语言模型(LLM)已日益成为内容生成的关键，具有显著的社会影响。这些模式有可能产生可能被认为有害的内容。缓解这种风险的方法包括实施保障措施，以确保LLMS遵守社会道德。然而，尽管采取了这些措施，“越狱”现象--精心制作的提示会招致模特的有害回应--仍然是一个重大挑战。认识到越狱战术构成的持续威胁及其对可信地使用LLM的影响，严格评估这些模型对此类攻击的稳健性是至关重要的。为了满足这一需求，本研究引入了一个综合评价框架，并进行了大规模的实证实验。我们集中在三个类别的10个尖端越狱策略，来自61个特定有害类别的1525个问题，以及13个流行的LLM。我们采用攻击成功率(ASR)、毒性分数、流畅度、令牌长度和语法错误等多维度量来彻底评估越狱情况下LLMS的输出。通过标准化和聚合这些指标，我们为不同的LLM提供了详细的可靠性分数，并提供了降低它们对此类漏洞的易感性的战略建议。此外，我们还探讨了模型、攻击策略和有害内容类型之间的关系，以及评估指标之间的相关性，从而证明了我们的多方面评估框架的有效性。我们广泛的实验结果表明，在所有测试的LLM中，对某些策略缺乏弹性，并强调了需要专注于LLM的可靠性方面。我们相信，我们的研究可以为加强LLMS在域内抗越狱的安全性评估提供有价值的见解。



## **26. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

BaThe：通过将有害指令视为后门触发来防御多模式大型语言模型中的越狱攻击 cs.CR

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09093v1) [paper-pdf](http://arxiv.org/pdf/2408.09093v1)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.

摘要: 多通道大型语言模型(MLLM)在各种多通道任务中表现出令人印象深刻的性能。另一方面，附加图像模式的集成可能允许恶意用户在图像中注入有害内容以越狱。与基于文本的LLMS不同，在LLMS中，攻击者需要使用特定的算法选择离散的令牌来隐藏其恶意意图，而图像信号的连续性为攻击者提供了直接注入有害意图的机会。在这项工作中，我们提出了一种简单而有效的越狱防御机制--$\extbf{bathe}$($\extbf{ba}$ck door$\extbf{T}$rigger S$\extbf{h}$i$\extbf{e}$ld)。我们的工作是基于生成式语言模型对越狱后门攻击和虚拟提示后门攻击的最新研究。越狱后门攻击使用有害指令和手动创建的字符串作为触发器，使后门模型生成被禁止的响应。我们假设有害指令可以作为触发器，如果我们将拒绝响应设置为触发响应，那么反向模型就可以防御越狱攻击。我们通过利用虚拟拒绝提示来实现这一点，类似于虚拟提示后门攻击。我们将虚拟拒绝提示嵌入到软文本嵌入中，我们称之为‘’楔形‘’。我们的综合实验表明，BAIT有效地缓解了各种类型的越狱攻击，并且能够自适应地防御看不见的攻击，对MLLMS的性能影响最小。



## **27. Can Editing LLMs Inject Harm?**

编辑LLM会造成伤害吗？ cs.CL

The first two authors contributed equally. 9 pages for main paper, 36  pages including appendix. The code, results, dataset for this paper and more  resources are on the project website: https://llm-editing.github.io

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2407.20224v3) [paper-pdf](http://arxiv.org/pdf/2407.20224v3)

**Authors**: Canyu Chen, Baixiang Huang, Zekun Li, Zhaorun Chen, Shiyang Lai, Xiongxiao Xu, Jia-Chen Gu, Jindong Gu, Huaxiu Yao, Chaowei Xiao, Xifeng Yan, William Yang Wang, Philip Torr, Dawn Song, Kai Shu

**Abstract**: Knowledge editing has been increasingly adopted to correct the false or outdated knowledge in Large Language Models (LLMs). Meanwhile, one critical but under-explored question is: can knowledge editing be used to inject harm into LLMs? In this paper, we propose to reformulate knowledge editing as a new type of safety threat for LLMs, namely Editing Attack, and conduct a systematic investigation with a newly constructed dataset EditAttack. Specifically, we focus on two typical safety risks of Editing Attack including Misinformation Injection and Bias Injection. For the risk of misinformation injection, we first categorize it into commonsense misinformation injection and long-tail misinformation injection. Then, we find that editing attacks can inject both types of misinformation into LLMs, and the effectiveness is particularly high for commonsense misinformation injection. For the risk of bias injection, we discover that not only can biased sentences be injected into LLMs with high effectiveness, but also one single biased sentence injection can cause a bias increase in general outputs of LLMs, which are even highly irrelevant to the injected sentence, indicating a catastrophic impact on the overall fairness of LLMs. Then, we further illustrate the high stealthiness of editing attacks, measured by their impact on the general knowledge and reasoning capacities of LLMs, and show the hardness of defending editing attacks with empirical evidence. Our discoveries demonstrate the emerging misuse risks of knowledge editing techniques on compromising the safety alignment of LLMs and the feasibility of disseminating misinformation or bias with LLMs as new channels.

摘要: 在大型语言模型中，知识编辑被越来越多地用于纠正错误或过时的知识。与此同时，一个关键但未被探讨的问题是：知识编辑能否被用来向低收入国家注入危害？在本文中，我们将知识编辑重新定义为一种新的安全威胁，即编辑攻击，并使用新构建的数据集EditAttack进行了系统的研究。具体地说，我们重点研究了编辑攻击的两个典型的安全风险，包括错误信息注入和偏见注入。对于错误信息注入的风险，我们首先将其分为常识性错误信息注入和长尾错误信息注入。然后，我们发现编辑攻击可以将这两种类型的错误信息注入到LLMS中，其中常识性错误信息注入的有效性尤其高。对于偏向注入的风险，我们发现，偏向句不仅可以被高效地注入到LLMS中，而且一次偏向句注入会导致LLMS的总体输出出现偏向增加，甚至与注入的句子高度无关，这对LLMS的整体公平性造成了灾难性的影响。然后，我们进一步说明了编辑攻击的高度隐蔽性，通过它们对LLM的常识和推理能力的影响来衡量它们，并用经验证据说明了防御编辑攻击的难度。我们的发现表明，知识编辑技术正在出现的滥用风险危及低成本管理的安全性，以及以低成本管理作为新渠道传播错误信息或偏见的可行性。



## **28. Ask, Attend, Attack: A Effective Decision-Based Black-Box Targeted Attack for Image-to-Text Models**

询问、参与、攻击：针对图像到文本模型的有效基于决策的黑匣子定向攻击 cs.AI

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08989v1) [paper-pdf](http://arxiv.org/pdf/2408.08989v1)

**Authors**: Qingyuan Zeng, Zhenzhong Wang, Yiu-ming Cheung, Min Jiang

**Abstract**: While image-to-text models have demonstrated significant advancements in various vision-language tasks, they remain susceptible to adversarial attacks. Existing white-box attacks on image-to-text models require access to the architecture, gradients, and parameters of the target model, resulting in low practicality. Although the recently proposed gray-box attacks have improved practicality, they suffer from semantic loss during the training process, which limits their targeted attack performance. To advance adversarial attacks of image-to-text models, this paper focuses on a challenging scenario: decision-based black-box targeted attacks where the attackers only have access to the final output text and aim to perform targeted attacks. Specifically, we formulate the decision-based black-box targeted attack as a large-scale optimization problem. To efficiently solve the optimization problem, a three-stage process \textit{Ask, Attend, Attack}, called \textit{AAA}, is proposed to coordinate with the solver. \textit{Ask} guides attackers to create target texts that satisfy the specific semantics. \textit{Attend} identifies the crucial regions of the image for attacking, thus reducing the search space for the subsequent \textit{Attack}. \textit{Attack} uses an evolutionary algorithm to attack the crucial regions, where the attacks are semantically related to the target texts of \textit{Ask}, thus achieving targeted attacks without semantic loss. Experimental results on transformer-based and CNN+RNN-based image-to-text models confirmed the effectiveness of our proposed \textit{AAA}.

摘要: 虽然图像到文本模型在各种视觉语言任务中显示出了显著的进步，但它们仍然容易受到对手的攻击。现有的针对图像到文本模型的白盒攻击需要访问目标模型的体系结构、渐变和参数，导致实用性较低。最近提出的灰盒攻击虽然提高了实用性，但它们在训练过程中存在语义丢失问题，限制了它们的针对性攻击性能。为了推进图像到文本模型的对抗性攻击，本文重点研究了一个具有挑战性的场景：基于决策的黑箱定向攻击，攻击者只能访问最终的输出文本，并且目标是执行定向攻击。具体地说，我们将基于决策的黑盒定向攻击问题描述为一个大规模优化问题。为了有效地解决优化问题，提出了一个三阶段过程\textit{Ask}引导攻击者创建满足特定语义的目标文本。\textit{attend}识别图像中要攻击的关键区域，从而减少了后续\textit{攻击}的搜索空间。利用进化算法攻击与目标文本语义相关的关键区域，从而在不丢失语义的情况下实现目标攻击。在基于变压器和基于CNN+RNN的图文转换模型上的实验结果证实了该方法的有效性。



## **29. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

大型语言模型能否提高图神经网络的对抗鲁棒性？ cs.LG

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08685v1) [paper-pdf](http://arxiv.org/pdf/2408.08685v1)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial perturbations, especially for topology attacks, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attack. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph.

摘要: 图神经网络(GNN)很容易受到敌意干扰，尤其是对拓扑攻击，许多提高GNN稳健性的方法受到了广泛的关注。最近，我们目睹了大型语言模型(LLM)的巨大成功，这导致许多人探索LLM在GNN上的巨大潜力。然而，它们主要集中在通过利用LLMS来增强节点特征来提高GNN的性能。因此，我们问：GNN的健壮性是否也会随着LLMS强大的理解和推理能力而得到增强？通过给出实验结果，我们发现，尽管LLMS可以提高GNN的健壮性，但其准确率仍然平均下降23.1%，这意味着GNN仍然非常容易受到拓扑攻击。因此，另一个问题是如何扩展LLMS在图对抗健壮性方面的能力。本文提出了一种基于LLM的稳健图结构推理框架LLM4RGNN，该框架将GPT-4的推理能力抽象为用于识别恶意边的局部LLM和用于发现丢失重要边的基于LLM的边预测器，以恢复稳健的图结构。大量的实验表明，LLM4RGNN在不同的GNN上一致地提高了健壮性。即使在某些扰动比增加到40%的情况下，GNN的精度仍然好于干净图形上的精度。



## **30. MIA-Tuner: Adapting Large Language Models as Pre-training Text Detector**

MIA-Tuner：调整大型语言模型作为预训练文本检测器 cs.CL

code and dataset: https://github.com/wjfu99/MIA-Tuner

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08661v1) [paper-pdf](http://arxiv.org/pdf/2408.08661v1)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: The increasing parameters and expansive dataset of large language models (LLMs) highlight the urgent demand for a technical solution to audit the underlying privacy risks and copyright issues associated with LLMs. Existing studies have partially addressed this need through an exploration of the pre-training data detection problem, which is an instance of a membership inference attack (MIA). This problem involves determining whether a given piece of text has been used during the pre-training phase of the target LLM. Although existing methods have designed various sophisticated MIA score functions to achieve considerable detection performance in pre-trained LLMs, how to achieve high-confidence detection and how to perform MIA on aligned LLMs remain challenging. In this paper, we propose MIA-Tuner, a novel instruction-based MIA method, which instructs LLMs themselves to serve as a more precise pre-training data detector internally, rather than design an external MIA score function. Furthermore, we design two instruction-based safeguards to respectively mitigate the privacy risks brought by the existing methods and MIA-Tuner. To comprehensively evaluate the most recent state-of-the-art LLMs, we collect a more up-to-date MIA benchmark dataset, named WIKIMIA-24, to replace the widely adopted benchmark WIKIMIA. We conduct extensive experiments across various aligned and unaligned LLMs over the two benchmark datasets. The results demonstrate that MIA-Tuner increases the AUC of MIAs from 0.7 to a significantly high level of 0.9.

摘要: 不断增加的参数和庞大的大型语言模型数据集突显了对技术解决方案的迫切需求，以审计与大型语言模型相关的潜在隐私风险和版权问题。现有的研究已经通过探索训练前数据检测问题部分地解决了这一需求，该问题是成员推理攻击(MIA)的一个实例。这个问题涉及确定在目标LLM的预训练阶段是否使用了给定的文本片段。虽然现有的方法已经设计了各种复杂的MIA评分函数来在预先训练的LLM中获得相当高的检测性能，但如何实现高置信度检测以及如何在对准的LLM上执行MIA仍然是具有挑战性的。在本文中，我们提出了一种新的基于指令的MIA方法MIA-Tuner，它指示LLMS本身在内部充当更精确的预训练数据检测器，而不是在外部设计MIA得分函数。此外，我们设计了两个基于指令的安全机制，分别缓解了现有方法和MIA-Tuner带来的隐私风险。为了全面评估最新的LLM，我们收集了一个更新的MIA基准数据集，名为WIKIMIA-24，以取代广泛采用的基准WIKIMIA。我们在两个基准数据集上对各种对齐和未对齐的LLM进行了广泛的实验。结果表明，MIA-Tuner将MIA的AUC从0.7提高到0.9的显著高水平。



## **31. Robust Neural Information Retrieval: An Adversarial and Out-of-distribution Perspective**

稳健的神经信息检索：对抗性和非分布性的角度 cs.IR

Survey paper

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2407.06992v2) [paper-pdf](http://arxiv.org/pdf/2407.06992v2)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Recent advances in neural information retrieval (IR) models have significantly enhanced their effectiveness over various IR tasks. The robustness of these models, essential for ensuring their reliability in practice, has also garnered significant attention. With a wide array of research on robust IR being proposed, we believe it is the opportune moment to consolidate the current status, glean insights from existing methodologies, and lay the groundwork for future development. We view the robustness of IR to be a multifaceted concept, emphasizing its necessity against adversarial attacks, out-of-distribution (OOD) scenarios and performance variance. With a focus on adversarial and OOD robustness, we dissect robustness solutions for dense retrieval models (DRMs) and neural ranking models (NRMs), respectively, recognizing them as pivotal components of the neural IR pipeline. We provide an in-depth discussion of existing methods, datasets, and evaluation metrics, shedding light on challenges and future directions in the era of large language models. To the best of our knowledge, this is the first comprehensive survey on the robustness of neural IR models, and we will also be giving our first tutorial presentation at SIGIR 2024 \url{https://sigir2024-robust-information-retrieval.github.io}. Along with the organization of existing work, we introduce a Benchmark for robust IR (BestIR), a heterogeneous evaluation benchmark for robust neural information retrieval, which is publicly available at \url{https://github.com/Davion-Liu/BestIR}. We hope that this study provides useful clues for future research on the robustness of IR models and helps to develop trustworthy search engines \url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.

摘要: 神经信息检索(IR)模型的最新进展显著提高了它们在各种IR任务中的有效性。这些模型的稳健性对于确保它们在实践中的可靠性至关重要，也引起了人们的极大关注。随着对稳健IR的广泛研究的提出，我们认为现在是巩固当前状况、从现有方法中收集见解并为未来发展奠定基础的好时机。我们认为信息检索的稳健性是一个多方面的概念，强调了它对对抗攻击、分布外(OOD)场景和性能差异的必要性。以对抗性和面向对象的稳健性为重点，我们分别剖析了密集检索模型(DRM)和神经排名模型(NRM)的稳健性解决方案，将它们识别为神经IR管道的关键组件。我们提供了对现有方法、数据集和评估度量的深入讨论，揭示了大型语言模型时代的挑战和未来方向。据我们所知，这是关于神经IR模型稳健性的第一次全面调查，我们还将在SIGIR2024\url{https://sigir2024-robust-information-retrieval.github.io}.上进行我们的第一次教程演示在组织现有工作的同时，我们还介绍了稳健IR基准(BSTIR)，这是一个用于稳健神经信息检索的异质评估基准，可在\url{https://github.com/Davion-Liu/BestIR}.希望本研究为今后研究信息检索模型的健壮性提供有用的线索，并为开发可信搜索引擎\url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.提供帮助



## **32. Trading Devil Final: Backdoor attack via Stock market and Bayesian Optimization**

交易魔鬼决赛：通过股市和Bayesian优化进行后门攻击 cs.LG

END (will never be modified again) :Jumps-Diffusion and stock market:  Better quantify uncertainty in financial simulations

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2407.14573v3) [paper-pdf](http://arxiv.org/pdf/2407.14573v3)

**Authors**: Orson Mengara

**Abstract**: Since the advent of generative artificial intelligence, every company and researcher has been rushing to develop their own generative models, whether commercial or not. Given the large number of users of these powerful new tools, there is currently no intrinsically verifiable way to explain from the ground up what happens when LLMs (large language models) learn. For example, those based on automatic speech recognition systems, which have to rely on huge and astronomical amounts of data collected from all over the web to produce fast and efficient results, In this article, we develop a backdoor attack called MarketBackFinal 2.0, based on acoustic data poisoning, MarketBackFinal 2.0 is mainly based on modern stock market models. In order to show the possible vulnerabilities of speech-based transformers that may rely on LLMs.

摘要: 自生成人工智能出现以来，每家公司和研究人员都在争先恐后地开发自己的生成模型，无论是否商业化。鉴于这些强大的新工具的大量用户，目前还没有本质上可验证的方法来从头解释LLM（大型语言模型）学习时会发生什么。例如，那些基于自动语音识别系统的系统，它们必须依赖于从整个网络收集的大量数据来产生快速有效的结果，在本文中，我们开发了一种名为MarketBackFinal 2.0的后门攻击，基于声学数据中毒，MarketBackFinal 2.0主要基于现代股市模型。为了显示可能依赖LLM的基于语音的转换器可能存在的漏洞。



## **33. ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages**

Tools Sword：跨越三个阶段揭示工具学习中大型语言模型的安全问题 cs.CL

Accepted by ACL 2024 Main Conference

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2402.10753v2) [paper-pdf](http://arxiv.org/pdf/2402.10753v2)

**Authors**: Junjie Ye, Sixian Li, Guanyu Li, Caishuang Huang, Songyang Gao, Yilong Wu, Qi Zhang, Tao Gui, Xuanjing Huang

**Abstract**: Tool learning is widely acknowledged as a foundational approach or deploying large language models (LLMs) in real-world scenarios. While current research primarily emphasizes leveraging tools to augment LLMs, it frequently neglects emerging safety considerations tied to their application. To fill this gap, we present *ToolSword*, a comprehensive framework dedicated to meticulously investigating safety issues linked to LLMs in tool learning. Specifically, ToolSword delineates six safety scenarios for LLMs in tool learning, encompassing **malicious queries** and **jailbreak attacks** in the input stage, **noisy misdirection** and **risky cues** in the execution stage, and **harmful feedback** and **error conflicts** in the output stage. Experiments conducted on 11 open-source and closed-source LLMs reveal enduring safety challenges in tool learning, such as handling harmful queries, employing risky tools, and delivering detrimental feedback, which even GPT-4 is susceptible to. Moreover, we conduct further studies with the aim of fostering research on tool learning safety. The data is released in https://github.com/Junjie-Ye/ToolSword.

摘要: 工具学习被广泛认为是在现实世界场景中部署大型语言模型(LLM)的基本方法。虽然目前的研究主要强调利用工具来增强LLM，但它往往忽略了与其应用相关的新出现的安全考虑。为了填补这一空白，我们推出了*ToolSword*，这是一个全面的框架，致力于在工具学习中仔细调查与LLM相关的安全问题。具体地说，ToolSword为低层管理工具学习划分了六个安全场景，包括输入阶段的**恶意查询**和**越狱攻击**，执行阶段的**噪音误导**和**危险提示**，以及输出阶段的**有害反馈**和**错误冲突**。在11个开源和封闭源代码的LLM上进行的实验表明，工具学习中存在持久的安全挑战，例如处理有害的查询、使用危险的工具以及提供有害的反馈，这些都是GPT-4容易受到的。此外，我们还进行了进一步的研究，旨在促进对工具学习安全性的研究。数据以https://github.com/Junjie-Ye/ToolSword.格式发布



## **34. \textit{MMJ-Bench}: A Comprehensive Study on Jailbreak Attacks and Defenses for Vision Language Models**

\textit{MMJ-Bench}：视觉语言模型越狱攻击和防御的综合研究 cs.CR

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08464v1) [paper-pdf](http://arxiv.org/pdf/2408.08464v1)

**Authors**: Fenghua Weng, Yue Xu, Chengyan Fu, Wenjie Wang

**Abstract**: As deep learning advances, Large Language Models (LLMs) and their multimodal counterparts, Vision-Language Models (VLMs), have shown exceptional performance in many real-world tasks. However, VLMs face significant security challenges, such as jailbreak attacks, where attackers attempt to bypass the model's safety alignment to elicit harmful responses. The threat of jailbreak attacks on VLMs arises from both the inherent vulnerabilities of LLMs and the multiple information channels that VLMs process. While various attacks and defenses have been proposed, there is a notable gap in unified and comprehensive evaluations, as each method is evaluated on different dataset and metrics, making it impossible to compare the effectiveness of each method. To address this gap, we introduce \textit{MMJ-Bench}, a unified pipeline for evaluating jailbreak attacks and defense techniques for VLMs. Through extensive experiments, we assess the effectiveness of various attack methods against SoTA VLMs and evaluate the impact of defense mechanisms on both defense effectiveness and model utility for normal tasks. Our comprehensive evaluation contribute to the field by offering a unified and systematic evaluation framework and the first public-available benchmark for VLM jailbreak research. We also demonstrate several insightful findings that highlights directions for future studies.

摘要: 随着深度学习的深入，大型语言模型(LLM)及其多通道模型(Vision-Language Model，VLM)在许多实际任务中表现出了优异的性能。然而，VLM面临着重大的安全挑战，例如越狱攻击，攻击者试图绕过模型的安全对齐，以引发有害的响应。越狱攻击对VLMS的威胁既来自LLMS固有的脆弱性，也源于VLMS处理的多种信息渠道。虽然已经提出了各种攻击和防御方法，但在统一和综合评估方面存在显著差距，因为每种方法都是在不同的数据集和指标上进行评估，因此无法比较每种方法的有效性。为了弥补这一差距，我们引入了一个统一的管道，用于评估越狱攻击和针对VLM的防御技术。通过大量的实验，我们评估了各种攻击方法对SOTA VLMS的攻击效果，并评估了防御机制对正常任务的防御效果和模型效用的影响。我们的全面评估为越狱研究提供了统一和系统的评估框架和第一个公开可用的基准，从而为该领域做出了贡献。我们还展示了几个有洞察力的发现，这些发现突出了未来研究的方向。



## **35. Trojan Activation Attack: Red-Teaming Large Language Models using Activation Steering for Safety-Alignment**

特洛伊激活攻击：使用激活引导进行红色分组大型语言模型以实现安全调整 cs.CR

ACM International Conference on Information and Knowledge Management  (CIKM'24)

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2311.09433v3) [paper-pdf](http://arxiv.org/pdf/2311.09433v3)

**Authors**: Haoran Wang, Kai Shu

**Abstract**: To ensure AI safety, instruction-tuned Large Language Models (LLMs) are specifically trained to ensure alignment, which refers to making models behave in accordance with human intentions. While these models have demonstrated commendable results on various safety benchmarks, the vulnerability of their safety alignment has not been extensively studied. This is particularly troubling given the potential harm that LLMs can inflict. Existing attack methods on LLMs often rely on poisoned training data or the injection of malicious prompts. These approaches compromise the stealthiness and generalizability of the attacks, making them susceptible to detection. Additionally, these models often demand substantial computational resources for implementation, making them less practical for real-world applications. In this work, we study a different attack scenario, called Trojan Activation Attack (TA^2), which injects trojan steering vectors into the activation layers of LLMs. These malicious steering vectors can be triggered at inference time to steer the models toward attacker-desired behaviors by manipulating their activations. Our experiment results on four primary alignment tasks show that TA^2 is highly effective and adds little or no overhead to attack efficiency. Additionally, we discuss potential countermeasures against such activation attacks.

摘要: 为了确保人工智能的安全，指令调优的大型语言模型(LLM)经过专门培训，以确保对齐，这指的是使模型的行为符合人类的意图。虽然这些模型在各种安全基准上显示了值得称赞的结果，但它们的安全配准的脆弱性还没有得到广泛的研究。考虑到LLMS可能造成的潜在危害，这一点尤其令人担忧。现有的对LLMS的攻击方法往往依赖于有毒的训练数据或注入恶意提示。这些方法损害了攻击的隐蔽性和通用性，使其容易被检测到。此外，这些模型通常需要大量的计算资源才能实现，这使得它们在实际应用中不太实用。在这项工作中，我们研究了一种不同的攻击场景，称为特洛伊木马激活攻击(TA^2)，它将木马引导向量注入LLMS的激活层。这些恶意引导向量可以在推理时被触发，通过操纵它们的激活来引导模型朝着攻击者想要的行为方向发展。我们在四个主要对齐任务上的实验结果表明，TA^2是高效的，并且几乎没有增加攻击效率的开销。此外，我们还讨论了针对此类激活攻击的潜在对策。



## **36. Enhancing Data Privacy in Large Language Models through Private Association Editing**

通过私人关联编辑增强大型语言模型中的数据隐私 cs.CL

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2406.18221v2) [paper-pdf](http://arxiv.org/pdf/2406.18221v2)

**Authors**: Davide Venditti, Elena Sofia Ruzzetti, Giancarlo A. Xompero, Cristina Giannone, Andrea Favalli, Raniero Romagnoli, Fabio Massimo Zanzotto

**Abstract**: Large Language Models (LLMs) are powerful tools with extensive applications, but their tendency to memorize private information raises significant concerns as private data leakage can easily happen. In this paper, we introduce Private Association Editing (PAE), a novel defense approach for private data leakage. PAE is designed to effectively remove Personally Identifiable Information (PII) without retraining the model. Our approach consists of a four-step procedure: detecting memorized PII, applying PAE cards to mitigate memorization of private data, verifying resilience to targeted data extraction (TDE) attacks, and ensuring consistency in the post-edit LLMs. The versatility and efficiency of PAE, which allows for batch modifications, significantly enhance data privacy in LLMs. Experimental results demonstrate the effectiveness of PAE in mitigating private data leakage. We believe PAE will serve as a critical tool in the ongoing effort to protect data privacy in LLMs, encouraging the development of safer models for real-world applications.

摘要: 大型语言模型(LLM)是应用广泛的强大工具，但它们存储私人信息的倾向引起了人们的极大担忧，因为私人数据很容易泄露。本文介绍了一种新的隐私数据泄露防御方法--私有关联编辑(PAE)。PAE旨在有效地删除个人身份信息(PII)，而无需对模型进行重新培训。我们的方法包括四个步骤：检测记忆的PII，应用PAE卡来减少私人数据的记忆，验证对目标数据提取(TDE)攻击的恢复能力，以及确保编辑后LLM的一致性。PAE的多功能性和效率允许批量修改，显著增强了LLMS中的数据隐私。实验结果证明了PAE在缓解私有数据泄露方面的有效性。我们相信，PAE将作为正在进行的保护LLMS数据隐私的努力中的关键工具，鼓励为现实世界应用程序开发更安全的模型。



## **37. Quantifying Memorization and Detecting Training Data of Pre-trained Language Models using Japanese Newspaper**

使用日本报纸量化再同步并检测预训练语言模型的训练数据 cs.CL

The 17th International Natural Language Generation Conference

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2404.17143v2) [paper-pdf](http://arxiv.org/pdf/2404.17143v2)

**Authors**: Shotaro Ishihara, Hiromu Takahashi

**Abstract**: Dominant pre-trained language models (PLMs) have demonstrated the potential risk of memorizing and outputting the training data. While this concern has been discussed mainly in English, it is also practically important to focus on domain-specific PLMs. In this study, we pre-trained domain-specific GPT-2 models using a limited corpus of Japanese newspaper articles and evaluated their behavior. Experiments replicated the empirical finding that memorization of PLMs is related to the duplication in the training data, model size, and prompt length, in Japanese the same as in previous English studies. Furthermore, we attempted membership inference attacks, demonstrating that the training data can be detected even in Japanese, which is the same trend as in English. The study warns that domain-specific PLMs, sometimes trained with valuable private data, can ''copy and paste'' on a large scale.

摘要: 占主导地位的预训练语言模型（PLM）已经证明了记忆和输出训练数据的潜在风险。虽然这个问题主要用英语讨论，但关注特定领域的PLM实际上也很重要。在这项研究中，我们使用有限的日本报纸文章库预训练了特定领域的GPT-2模型，并评估了它们的行为。实验复制了经验发现，即PLM的记忆与训练数据的重复、模型大小和提示长度有关，在日语中与之前的英语研究相同。此外，我们尝试了成员推断攻击，证明即使在日语中也可以检测到训练数据，这与英语中的趋势相同。该研究警告说，特定领域的PLM（有时使用有价值的私人数据进行训练）可以大规模“复制和粘贴”。



## **38. Prefix Guidance: A Steering Wheel for Large Language Models to Defend Against Jailbreak Attacks**

前置指导：大型语言模型防御越狱攻击的方向盘 cs.CR

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.08924v1) [paper-pdf](http://arxiv.org/pdf/2408.08924v1)

**Authors**: Jiawei Zhao, Kejiang Chen, Xiaojian Yuan, Weiming Zhang

**Abstract**: In recent years, the rapid development of large language models (LLMs) has achieved remarkable performance across various tasks. However, research indicates that LLMs are vulnerable to jailbreak attacks, where adversaries can induce the generation of harmful content through meticulously crafted prompts. This vulnerability poses significant challenges to the secure use and promotion of LLMs. Existing defense methods offer protection from different perspectives but often suffer from insufficient effectiveness or a significant impact on the model's capabilities. In this paper, we propose a plug-and-play and easy-to-deploy jailbreak defense framework, namely Prefix Guidance (PG), which guides the model to identify harmful prompts by directly setting the first few tokens of the model's output. This approach combines the model's inherent security capabilities with an external classifier to defend against jailbreak attacks. We demonstrate the effectiveness of PG across three models and five attack methods. Compared to baselines, our approach is generally more effective on average. Additionally, results on the Just-Eval benchmark further confirm PG's superiority to preserve the model's performance.

摘要: 近年来，大型语言模型的快速发展在各种任务中取得了显著的性能。然而，研究表明，LLMS容易受到越狱攻击，在越狱攻击中，攻击者可以通过精心制作的提示来诱导生成有害内容。此漏洞对安全使用和推广LLMS构成重大挑战。现有的防御方法从不同的角度提供保护，但往往存在有效性不足或对模型能力产生重大影响的问题。本文提出了一种即插即用、易于部署的越狱防御框架--前缀引导(PG)，它通过直接设置模型输出的前几个令牌来引导模型识别有害提示。这种方法将模型固有的安全功能与外部分类器相结合，以防御越狱攻击。我们在三个模型和五种攻击方法上演示了PG的有效性。与基线相比，我们的方法总体上更有效。此外，在Just-Eval基准上的结果进一步证实了PG在保持模型性能方面的优越性。



## **39. KGV: Integrating Large Language Models with Knowledge Graphs for Cyber Threat Intelligence Credibility Assessment**

KGV：将大型语言模型与知识图集成，用于网络威胁情报可信度评估 cs.CR

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.08088v1) [paper-pdf](http://arxiv.org/pdf/2408.08088v1)

**Authors**: Zongzong Wu, Fengxiao Tang, Ming Zhao, Yufeng Li

**Abstract**: Cyber threat intelligence is a critical tool that many organizations and individuals use to protect themselves from sophisticated, organized, persistent, and weaponized cyber attacks. However, few studies have focused on the quality assessment of threat intelligence provided by intelligence platforms, and this work still requires manual analysis by cybersecurity experts. In this paper, we propose a knowledge graph-based verifier, a novel Cyber Threat Intelligence (CTI) quality assessment framework that combines knowledge graphs and Large Language Models (LLMs). Our approach introduces LLMs to automatically extract OSCTI key claims to be verified and utilizes a knowledge graph consisting of paragraphs for fact-checking. This method differs from the traditional way of constructing complex knowledge graphs with entities as nodes. By constructing knowledge graphs with paragraphs as nodes and semantic similarity as edges, it effectively enhances the semantic understanding ability of the model and simplifies labeling requirements. Additionally, to fill the gap in the research field, we created and made public the first dataset for threat intelligence assessment from heterogeneous sources. To the best of our knowledge, this work is the first to create a dataset on threat intelligence reliability verification, providing a reference for future research. Experimental results show that KGV (Knowledge Graph Verifier) significantly improves the performance of LLMs in intelligence quality assessment. Compared with traditional methods, we reduce a large amount of data annotation while the model still exhibits strong reasoning capabilities. Finally, our method can achieve XXX accuracy in network threat assessment.

摘要: 网络威胁情报是许多组织和个人用来保护自己免受复杂、有组织、持续和武器化的网络攻击的关键工具。然而，很少有研究将重点放在情报平台提供的威胁情报的质量评估上，这项工作仍然需要网络安全专家进行人工分析。本文提出了一种基于知识图的验证器--结合知识图和大语言模型的网络威胁情报(CTI)质量评估框架。该方法引入LLMS来自动提取待验证的OSCTI关键声明，并利用由段落组成的知识图进行事实验证。该方法不同于传统的以实体为节点构建复杂知识图的方法。通过构建以段落为节点、以语义相似度为边的知识图，有效地增强了模型的语义理解能力，简化了标注要求。此外，为了填补这一研究领域的空白，我们创建并公开了第一个来自不同来源的威胁情报评估数据集。据我们所知，这项工作是首次创建威胁情报可靠性验证数据集，为未来的研究提供参考。实验结果表明，KGV(Knowledge Graph Verator)显著提高了LLMS在智能质量评估中的性能。与传统方法相比，我们减少了大量的数据标注，而模型仍然具有很强的推理能力。最后，我们的方法可以在网络威胁评估中达到XXX的准确率。



## **40. Large Language Model Sentinel: LLM Agent for Adversarial Purification**

大型语言模型Sentinel：对抗性纯化的LLM代理 cs.CL

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2405.20770v2) [paper-pdf](http://arxiv.org/pdf/2405.20770v2)

**Authors**: Guang Lin, Qibin Zhao

**Abstract**: Over the past two years, the use of large language models (LLMs) has advanced rapidly. While these LLMs offer considerable convenience, they also raise security concerns, as LLMs are vulnerable to adversarial attacks by some well-designed textual perturbations. In this paper, we introduce a novel defense technique named Large LAnguage MOdel Sentinel (LLAMOS), which is designed to enhance the adversarial robustness of LLMs by purifying the adversarial textual examples before feeding them into the target LLM. Our method comprises two main components: a) Agent instruction, which can simulate a new agent for adversarial defense, altering minimal characters to maintain the original meaning of the sentence while defending against attacks; b) Defense guidance, which provides strategies for modifying clean or adversarial examples to ensure effective defense and accurate outputs from the target LLMs. Remarkably, the defense agent demonstrates robust defensive capabilities even without learning from adversarial examples. Additionally, we conduct an intriguing adversarial experiment where we develop two agents, one for defense and one for attack, and engage them in mutual confrontation. During the adversarial interactions, neither agent completely beat the other. Extensive experiments on both open-source and closed-source LLMs demonstrate that our method effectively defends against adversarial attacks, thereby enhancing adversarial robustness.

摘要: 在过去的两年里，大型语言模型(LLM)的使用取得了快速发展。虽然这些LLM提供了相当大的便利，但它们也引发了安全问题，因为LLM容易受到一些精心设计的文本扰动的敌意攻击。本文介绍了一种新的防御技术--大语言模型哨兵(LLAMOS)，该技术旨在通过在将对抗性文本实例输入目标LLM之前对其进行提纯来增强LLMS的对抗性健壮性。我们的方法包括两个主要部分：a)代理指令，它可以模拟一个新的代理进行对抗性防御，改变最少的字符，在防御攻击的同时保持句子的原始含义；b)防御指导，它提供修改干净或对抗性示例的策略，以确保目标LLMS的有效防御和准确输出。值得注意的是，防御代理展示了强大的防御能力，即使没有从对手的例子中学习。此外，我们还进行了一个有趣的对抗性实验，在这个实验中，我们开发了两个代理，一个用于防御，一个用于攻击，并让他们相互对抗。在敌对的互动中，两个代理都没有完全击败另一个。在开源和闭源LLMS上的大量实验表明，我们的方法有效地防御了对手攻击，从而增强了对手攻击的健壮性。



## **41. ConfusedPilot: Confused Deputy Risks in RAG-based LLMs**

困惑的飞行员：基于RAG的LLM中令人困惑的代理风险 cs.CR

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.04870v3) [paper-pdf](http://arxiv.org/pdf/2408.04870v3)

**Authors**: Ayush RoyChowdhury, Mulong Luo, Prateek Sahu, Sarbartha Banerjee, Mohit Tiwari

**Abstract**: Retrieval augmented generation (RAG) is a process where a large language model (LLM) retrieves useful information from a database and then generates the responses. It is becoming popular in enterprise settings for daily business operations. For example, Copilot for Microsoft 365 has accumulated millions of businesses. However, the security implications of adopting such RAG-based systems are unclear.   In this paper, we introduce ConfusedPilot, a class of security vulnerabilities of RAG systems that confuse Copilot and cause integrity and confidentiality violations in its responses. First, we investigate a vulnerability that embeds malicious text in the modified prompt in RAG, corrupting the responses generated by the LLM. Second, we demonstrate a vulnerability that leaks secret data, which leverages the caching mechanism during retrieval. Third, we investigate how both vulnerabilities can be exploited to propagate misinformation within the enterprise and ultimately impact its operations, such as sales and manufacturing. We also discuss the root cause of these attacks by investigating the architecture of a RAG-based system. This study highlights the security vulnerabilities in today's RAG-based systems and proposes design guidelines to secure future RAG-based systems.

摘要: 检索增强生成(RAG)是大型语言模型(LLM)从数据库中检索有用信息然后生成响应的过程。它在用于日常业务操作的企业环境中变得流行起来。例如，微软365的Copilot已经积累了数百万笔业务。然而，采用这种基于RAG的系统的安全影响尚不清楚。在本文中，我们介绍了一类RAG系统的安全漏洞ConfusedPilot，它迷惑了Copilot，并在其响应中导致完整性和保密性违规。首先，我们调查了一个漏洞，该漏洞将恶意文本嵌入到RAG中修改的提示符中，破坏了LLM生成的响应。其次，我们演示了一个泄漏机密数据的漏洞，该漏洞在检索过程中利用缓存机制。第三，我们调查如何利用这两个漏洞在企业内部传播错误信息，并最终影响其运营，如销售和制造。我们还通过研究基于RAG的系统的体系结构来讨论这些攻击的根本原因。这项研究强调了当今基于RAG的系统中的安全漏洞，并提出了保护未来基于RAG的系统的设计指南。



## **42. Alignment-Enhanced Decoding:Defending via Token-Level Adaptive Refining of Probability Distributions**

对齐增强解码：通过概率分布的令牌级自适应细化进行防御 cs.CL

15 pages, 5 figures

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2408.07663v1) [paper-pdf](http://arxiv.org/pdf/2408.07663v1)

**Authors**: Quan Liu, Zhenhong Zhou, Longzhu He, Yi Liu, Wei Zhang, Sen Su

**Abstract**: Large language models are susceptible to jailbreak attacks, which can result in the generation of harmful content. While prior defenses mitigate these risks by perturbing or inspecting inputs, they ignore competing objectives, the underlying cause of alignment failures. In this paper, we propose Alignment-Enhanced Decoding (AED), a novel defense that employs adaptive decoding to address the root causes of jailbreak issues. We first define the Competitive Index to quantify alignment failures and utilize feedback from self-evaluation to compute post-alignment logits. Then, AED adaptively combines AED and post-alignment logits with the original logits to obtain harmless and helpful distributions. Consequently, our method enhances safety alignment while maintaining helpfulness. We conduct experiments across five models and four common jailbreaks, with the results validating the effectiveness of our approach. Code is available at https://github.com/GIGABaozi/AED.git.

摘要: 大型语言模型很容易受到越狱攻击，这可能会导致有害内容的生成。虽然现有的防御措施通过干扰或检查输入来减轻这些风险，但它们忽略了竞争目标，这是对齐失败的根本原因。在本文中，我们提出了对齐增强解码（AED），这是一种新型防御方法，采用自适应解码来解决越狱问题的根本原因。我们首先定义竞争指数来量化对齐失败，并利用自我评估的反馈来计算对齐后日志。然后，AED自适应地将AED和对齐后logit与原始logit结合起来，以获得无害且有用的分布。因此，我们的方法在保持帮助性的同时增强了安全性。我们对五种模型和四种常见越狱进行了实验，结果验证了我们方法的有效性。代码可在https://github.com/GIGABaozi/AED.git上获取。



## **43. Lost in Overlap: Exploring Watermark Collision in LLMs**

迷失在重叠中：探索LLC中的水印碰撞 cs.CL

Long Paper, 7 pages

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2403.10020v2) [paper-pdf](http://arxiv.org/pdf/2403.10020v2)

**Authors**: Yiyang Luo, Ke Lin, Chao Gu

**Abstract**: The proliferation of large language models (LLMs) in generating content raises concerns about text copyright. Watermarking methods, particularly logit-based approaches, embed imperceptible identifiers into text to address these challenges. However, the widespread usage of watermarking across diverse LLMs has led to an inevitable issue known as watermark collision during common tasks, such as paraphrasing or translation. In this paper, we introduce watermark collision as a novel and general philosophy for watermark attacks, aimed at enhancing attack performance on top of any other attacking methods. We also provide a comprehensive demonstration that watermark collision poses a threat to all logit-based watermark algorithms, impacting not only specific attack scenarios but also downstream applications.

摘要: 生成内容时大型语言模型（LLM）的激增引发了人们对文本版权的担忧。水印方法，特别是基于日志的方法，将不可感知的标识符嵌入到文本中来解决这些挑战。然而，水印在不同的LLM中的广泛使用导致了在常见任务（例如解释或翻译）期间不可避免的问题，称为水印冲突。在本文中，我们引入水印冲突作为水印攻击的一种新颖且通用的哲学，旨在在任何其他攻击方法之上提高攻击性能。我们还全面证明了水印冲突对所有基于日志的水印算法构成威胁，不仅影响特定的攻击场景，还影响下游应用。



## **44. Evaluating Large Language Model based Personal Information Extraction and Countermeasures**

基于大语言模型的个人信息提取评估及对策 cs.CR

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2408.07291v1) [paper-pdf](http://arxiv.org/pdf/2408.07291v1)

**Authors**: Yupei Liu, Yuqi Jia, Jinyuan Jia, Neil Zhenqiang Gong

**Abstract**: Automatically extracting personal information--such as name, phone number, and email address--from publicly available profiles at a large scale is a stepstone to many other security attacks including spear phishing. Traditional methods--such as regular expression, keyword search, and entity detection--achieve limited success at such personal information extraction. In this work, we perform a systematic measurement study to benchmark large language model (LLM) based personal information extraction and countermeasures. Towards this goal, we present a framework for LLM-based extraction attacks; collect three datasets including a synthetic dataset generated by GPT-4 and two real-world datasets with manually labeled 8 categories of personal information; introduce a novel mitigation strategy based on \emph{prompt injection}; and systematically benchmark LLM-based attacks and countermeasures using 10 LLMs and our 3 datasets. Our key findings include: LLM can be misused by attackers to accurately extract various personal information from personal profiles; LLM outperforms conventional methods at such extraction; and prompt injection can mitigate such risk to a large extent and outperforms conventional countermeasures. Our code and data are available at: \url{https://github.com/liu00222/LLM-Based-Personal-Profile-Extraction}.

摘要: 从公开的个人资料中大规模自动提取个人信息--如姓名、电话号码和电子邮件地址--是应对包括鱼叉式网络钓鱼在内的许多其他安全攻击的一步。传统的方法--如正则表达式、关键字搜索和实体检测--在这种个人信息提取方面取得的成功有限。在这项工作中，我们对基于大语言模型的个人信息提取和对策进行了系统的测量研究。为此，我们提出了一个基于LLM的抽取攻击框架；收集了三个数据集，包括GPT-4生成的一个合成数据集和两个手动标注了8类个人信息的真实数据集；引入了一种新的基于提示注入的缓解策略；并使用10个LLM和我们的3个数据集对基于LLM的攻击和对策进行了系统的基准测试。我们的主要发现包括：LLM可以被攻击者滥用来准确地从个人档案中提取各种个人信息；LLM在这种提取方面优于传统方法；快速注入可以在很大程度上缓解这种风险，并优于传统的对抗措施。有关我们的代码和数据，请访问：\url{https://github.com/liu00222/LLM-Based-Personal-Profile-Extraction}.



## **45. Figure it Out: Analyzing-based Jailbreak Attack on Large Language Models**

弄清楚：基于分析的对大型语言模型的越狱攻击 cs.CR

**SubmitDate**: 2024-08-13    [abs](http://arxiv.org/abs/2407.16205v3) [paper-pdf](http://arxiv.org/pdf/2407.16205v3)

**Authors**: Shi Lin, Rongchang Li, Xun Wang, Changting Lin, Wenpeng Xing, Meng Han

**Abstract**: The rapid development of Large Language Models (LLMs) has brought remarkable generative capabilities across diverse tasks. However, despite the impressive achievements, these LLMs still have numerous inherent vulnerabilities, particularly when faced with jailbreak attacks. By investigating jailbreak attacks, we can uncover hidden weaknesses in LLMs and inform the development of more robust defense mechanisms to fortify their security. In this paper, we further explore the boundary of jailbreak attacks on LLMs and propose Analyzing-based Jailbreak (ABJ). This effective jailbreak attack method takes advantage of LLMs' growing analyzing and reasoning capability and reveals their underlying vulnerabilities when facing analyzing-based tasks. We conduct a detailed evaluation of ABJ across various open-source and closed-source LLMs, which achieves 94.8% attack success rate (ASR) and 1.06 attack efficiency (AE) on GPT-4-turbo-0409, demonstrating state-of-the-art attack effectiveness and efficiency. Our research highlights the importance of prioritizing and enhancing the safety of LLMs to mitigate the risks of misuse. The code is publicly available at hhttps://github.com/theshi-1128/ABJ-Attack. Warning: This paper contains examples of LLMs that might be offensive or harmful.

摘要: 大型语言模型(LLM)的快速发展带来了跨越各种任务的非凡的生成能力。然而，尽管取得了令人印象深刻的成就，这些LLM仍然存在许多固有的漏洞，特别是在面临越狱攻击时。通过调查越狱攻击，我们可以发现LLMS中隐藏的弱点，并为开发更强大的防御机制来加强其安全提供信息。本文进一步探讨了LLMS越狱攻击的边界，提出了基于分析的越狱攻击(ABJ)。这种有效的越狱攻击方法利用了LLMS日益增长的分析和推理能力，并在面对基于分析的任务时揭示了它们潜在的漏洞。我们对ABJ在各种开源和闭源LLMS上进行了详细的评估，在GPT-4-TURBO-0409上达到了94.8%的攻击成功率(ASR)和1.06的攻击效率(AE)，展示了最先进的攻击效果和效率。我们的研究强调了优先考虑和加强低密度脂蛋白的安全性，以减少误用风险的重要性。该代码可在h https://github.com/theshi-1128/ABJ-Attack.上公开获取警告：本文包含可能具有攻击性或危害性的LLM示例。



## **46. PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models**

PoisonedRAG：对大型语言模型检索增强生成的知识腐败攻击 cs.CR

To appear in USENIX Security Symposium 2025. The code is available at  https://github.com/sleeepeer/PoisonedRAG

**SubmitDate**: 2024-08-13    [abs](http://arxiv.org/abs/2402.07867v3) [paper-pdf](http://arxiv.org/pdf/2402.07867v3)

**Authors**: Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia

**Abstract**: Large language models (LLMs) have achieved remarkable success due to their exceptional generative capabilities. Despite their success, they also have inherent limitations such as a lack of up-to-date knowledge and hallucination. Retrieval-Augmented Generation (RAG) is a state-of-the-art technique to mitigate these limitations. The key idea of RAG is to ground the answer generation of an LLM on external knowledge retrieved from a knowledge database. Existing studies mainly focus on improving the accuracy or efficiency of RAG, leaving its security largely unexplored. We aim to bridge the gap in this work. We find that the knowledge database in a RAG system introduces a new and practical attack surface. Based on this attack surface, we propose PoisonedRAG, the first knowledge corruption attack to RAG, where an attacker could inject a few malicious texts into the knowledge database of a RAG system to induce an LLM to generate an attacker-chosen target answer for an attacker-chosen target question. We formulate knowledge corruption attacks as an optimization problem, whose solution is a set of malicious texts. Depending on the background knowledge (e.g., black-box and white-box settings) of an attacker on a RAG system, we propose two solutions to solve the optimization problem, respectively. Our results show PoisonedRAG could achieve a 90% attack success rate when injecting five malicious texts for each target question into a knowledge database with millions of texts. We also evaluate several defenses and our results show they are insufficient to defend against PoisonedRAG, highlighting the need for new defenses.

摘要: 大型语言模型(LLM)由于其非凡的生成能力而取得了显著的成功。尽管他们取得了成功，但他们也有内在的局限性，比如缺乏最新的知识和幻觉。检索-增强生成(RAG)是一种缓解这些限制的最先进的技术。RAG的核心思想是基于从知识库中检索到的外部知识来生成LLM的答案。现有的研究主要集中于提高RAG的准确性或效率，而其安全性在很大程度上还没有被探索。我们的目标是弥合这项工作中的差距。我们发现，RAG系统中的知识库引入了一个新的、实用的攻击面。基于这个攻击面，我们提出了PoisonedRAG，这是第一个针对RAG的知识腐败攻击，攻击者可以向RAG系统的知识库中注入一些恶意文本，以诱导LLM为攻击者选择的目标问题生成攻击者选择的目标答案。我们将知识腐败攻击描述为一个优化问题，其解决方案是一组恶意文本。根据RAG系统上攻击者的背景知识(例如黑盒和白盒设置)，我们分别提出了两种解决优化问题的方案。实验结果表明，PoisonedRAG在一个包含数百万条文本的知识库中为每个目标问题注入5个恶意文本，攻击成功率可达到90%。我们还评估了几种防御措施，我们的结果表明，它们不足以防御PoisonedRAG，这突显了需要新的防御措施。



## **47. A RAG-Based Question-Answering Solution for Cyber-Attack Investigation and Attribution**

一种基于RAG的网络攻击调查和归因网络响应解决方案 cs.CR

Accepted at SECAI 2024 (ESORICS 2024)

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.06272v1) [paper-pdf](http://arxiv.org/pdf/2408.06272v1)

**Authors**: Sampath Rajapaksha, Ruby Rani, Erisa Karafili

**Abstract**: In the constantly evolving field of cybersecurity, it is imperative for analysts to stay abreast of the latest attack trends and pertinent information that aids in the investigation and attribution of cyber-attacks. In this work, we introduce the first question-answering (QA) model and its application that provides information to the cybersecurity experts about cyber-attacks investigations and attribution. Our QA model is based on Retrieval Augmented Generation (RAG) techniques together with a Large Language Model (LLM) and provides answers to the users' queries based on either our knowledge base (KB) that contains curated information about cyber-attacks investigations and attribution or on outside resources provided by the users. We have tested and evaluated our QA model with various types of questions, including KB-based, metadata-based, specific documents from the KB, and external sources-based questions. We compared the answers for KB-based questions with those from OpenAI's GPT-3.5 and the latest GPT-4o LLMs. Our proposed QA model outperforms OpenAI's GPT models by providing the source of the answers and overcoming the hallucination limitations of the GPT models, which is critical for cyber-attack investigation and attribution. Additionally, our analysis showed that when the RAG QA model is given few-shot examples rather than zero-shot instructions, it generates better answers compared to cases where no examples are supplied in addition to the query.

摘要: 在不断发展的网络安全领域，分析人员必须了解最新的攻击趋势和相关信息，以帮助调查和确定网络攻击的归属。在这项工作中，我们介绍了第一个问答(QA)模型及其应用，该模型向网络安全专家提供有关网络攻击调查和归因的信息。我们的QA模型基于检索增强生成(RAG)技术和大型语言模型(LLM)，并基于我们的知识库(KB)(包含有关网络攻击调查和归属的精选信息)或用户提供的外部资源来回答用户的问题。我们已经使用各种类型的问题测试和评估了我们的QA模型，包括基于知识库的、基于元数据的、来自知识库的特定文档以及基于外部来源的问题。我们将基于知识库的问题的答案与OpenAI的GPT-3.5和最新的GPT-40 LLM的答案进行了比较。我们提出的QA模型在提供答案来源和克服GPT模型的幻觉限制方面优于OpenAI的GPT模型，而GPT模型对于网络攻击调查和归因至关重要。此外，我们的分析表明，当RAG QA模型给出少量的例子而不是零的指令时，与除了查询之外没有提供例子的情况相比，它生成了更好的答案。



## **48. On Effects of Steering Latent Representation for Large Language Model Unlearning**

论引导潜在表示对大型语言模型取消学习的影响 cs.CL

15 pages, 5 figures, 8 tables

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.06223v1) [paper-pdf](http://arxiv.org/pdf/2408.06223v1)

**Authors**: Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue

**Abstract**: Representation Misdirection for Unlearning (RMU), which steers model representation in the intermediate layer to a target random representation, is an effective method for large language model (LLM) unlearning. Despite its high performance, the underlying cause and explanation remain underexplored. In this paper, we first theoretically demonstrate that steering forget representations in the intermediate layer reduces token confidence, causing LLMs to generate wrong or nonsense responses. Second, we investigate how the coefficient influences the alignment of forget-sample representations with the random direction and hint at the optimal coefficient values for effective unlearning across different network layers. Third, we show that RMU unlearned models are robust against adversarial jailbreak attacks. Last, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs. To resolve this drawback, we propose Adaptive RMU -- a simple yet effective alternative method that makes unlearning effective with most layers. Extensive experiments demonstrate that Adaptive RMU significantly improves the unlearning performance compared to prior art while incurring no additional computational cost.

摘要: 遗忘表征误导(RMU)是一种有效的大语言模型遗忘方法，它将中间层的模型表征引导到目标随机表征。尽管其表现良好，但其根本原因和解释仍未得到充分研究。在本文中，我们首先从理论上证明，中间层中的转向遗忘表征会降低令牌置信度，从而导致LLM生成错误或无意义的响应。其次，我们研究了系数如何影响遗忘样本表示与随机方向的对齐，并提示了跨不同网络层有效遗忘的最优系数值。第三，我们证明了RMU未学习模型对敌意越狱攻击是健壮的。最后，我们的实证分析表明，当RMU应用于LLMS的中后期时，其有效性较差。为了解决这一缺陷，我们提出了自适应RMU--一种简单但有效的替代方法，使遗忘在大多数层都有效。大量实验表明，与现有技术相比，自适应RMU在不增加额外计算代价的情况下，显著改善了遗忘性能。



## **49. Nob-MIAs: Non-biased Membership Inference Attacks Assessment on Large Language Models with Ex-Post Dataset Construction**

Nob-MIA：对具有Ex-Post数据集构建的大型语言模型的无偏见成员推理攻击评估 cs.CR

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.05968v1) [paper-pdf](http://arxiv.org/pdf/2408.05968v1)

**Authors**: Cédric Eichler, Nathan Champeil, Nicolas Anciaux, Alexandra Bensamoun, Heber Hwang Arcolezi, José Maria De Fuentes

**Abstract**: The rise of Large Language Models (LLMs) has triggered legal and ethical concerns, especially regarding the unauthorized use of copyrighted materials in their training datasets. This has led to lawsuits against tech companies accused of using protected content without permission. Membership Inference Attacks (MIAs) aim to detect whether specific documents were used in a given LLM pretraining, but their effectiveness is undermined by biases such as time-shifts and n-gram overlaps.   This paper addresses the evaluation of MIAs on LLMs with partially inferable training sets, under the ex-post hypothesis, which acknowledges inherent distributional biases between members and non-members datasets. We propose and validate algorithms to create ``non-biased'' and ``non-classifiable'' datasets for fairer MIA assessment. Experiments using the Gutenberg dataset on OpenLamma and Pythia show that neutralizing known biases alone is insufficient. Our methods produce non-biased ex-post datasets with AUC-ROC scores comparable to those previously obtained on genuinely random datasets, validating our approach. Globally, MIAs yield results close to random, with only one being effective on both random and our datasets, but its performance decreases when bias is removed.

摘要: 大型语言模型(LLM)的兴起引发了法律和伦理方面的担忧，特别是关于在其培训数据集中未经授权使用受版权保护的材料。这导致了针对科技公司的诉讼，这些公司被控未经许可使用受保护的内容。成员关系推理攻击(MIA)的目的是检测特定文档是否被用于给定的LLM预训练，但其有效性受到时移和n元语法重叠等偏差的影响。在承认成员和非成员数据集之间固有分布偏差的后验假设下，本文讨论了部分可推断训练集的LLMS上的MIA的评估。我们提出并验证了为更公平的MIA评估创建“无偏见”和“不可分类”数据集的算法。在OpenLamma和Pythia上使用Gutenberg数据集进行的实验表明，仅中和已知的偏见是不够的。我们的方法产生无偏见的事后数据集，其AUC-ROC得分与之前在真正随机数据集上获得的得分相当，验证了我们的方法。在全球范围内，MIA产生的结果接近随机，只有一个对随机和我们的数据集有效，但当消除偏差时，其性能会下降。



## **50. Multimodal Large Language Models for Phishing Webpage Detection and Identification**

用于网络钓鱼网页检测和识别的多模式大语言模型 cs.CR

To appear in eCrime 2024

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.05941v1) [paper-pdf](http://arxiv.org/pdf/2408.05941v1)

**Authors**: Jehyun Lee, Peiyuan Lim, Bryan Hooi, Dinil Mon Divakaran

**Abstract**: To address the challenging problem of detecting phishing webpages, researchers have developed numerous solutions, in particular those based on machine learning (ML) algorithms. Among these, brand-based phishing detection that uses models from Computer Vision to detect if a given webpage is imitating a well-known brand has received widespread attention. However, such models are costly and difficult to maintain, as they need to be retrained with labeled dataset that has to be regularly and continuously collected. Besides, they also need to maintain a good reference list of well-known websites and related meta-data for effective performance.   In this work, we take steps to study the efficacy of large language models (LLMs), in particular the multimodal LLMs, in detecting phishing webpages. Given that the LLMs are pretrained on a large corpus of data, we aim to make use of their understanding of different aspects of a webpage (logo, theme, favicon, etc.) to identify the brand of a given webpage and compare the identified brand with the domain name in the URL to detect a phishing attack. We propose a two-phase system employing LLMs in both phases: the first phase focuses on brand identification, while the second verifies the domain. We carry out comprehensive evaluations on a newly collected dataset. Our experiments show that the LLM-based system achieves a high detection rate at high precision; importantly, it also provides interpretable evidence for the decisions. Our system also performs significantly better than a state-of-the-art brand-based phishing detection system while demonstrating robustness against two known adversarial attacks.

摘要: 为了解决检测钓鱼网页这一具有挑战性的问题，研究人员开发了许多解决方案，特别是基于机器学习(ML)算法的解决方案。其中，基于品牌的钓鱼检测利用计算机视觉的模型来检测给定的网页是否在模仿知名品牌，受到了广泛的关注。然而，这种模型成本很高，很难维护，因为它们需要用必须定期和连续收集的标记数据集进行再训练。此外，他们还需要维护一个良好的参考名单的知名网站和相关的元数据，以有效的表现。在这项工作中，我们采取步骤研究大语言模型，特别是多模式大语言模型在检测钓鱼网页方面的有效性。鉴于LLM是在大型数据语料库上预先培训的，我们的目标是利用他们对网页的不同方面(徽标、主题、图标等)的理解。识别给定网页的品牌，并将识别的品牌与URL中的域名进行比较，以检测网络钓鱼攻击。我们提出了一个在两个阶段都使用LLMS的两阶段系统：第一阶段专注于品牌识别，第二阶段验证领域。我们对新收集的数据集进行了全面的评估。我们的实验表明，基于LLM的系统在高精度的情况下实现了高检测率，重要的是它还为决策提供了可解释的证据。我们的系统也比最先进的基于品牌的钓鱼检测系统性能要好得多，同时对两种已知的对手攻击表现出了健壮性。



