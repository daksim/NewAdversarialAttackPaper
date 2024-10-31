# Latest Large Language Model Attack Papers
**update at 2024-10-31 09:49:42**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Effective and Efficient Adversarial Detection for Vision-Language Models via A Single Vector**

通过单个载体对视觉语言模型进行有效且高效的对抗检测 cs.CV

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22888v1) [paper-pdf](http://arxiv.org/pdf/2410.22888v1)

**Authors**: Youcheng Huang, Fengbin Zhu, Jingkun Tang, Pan Zhou, Wenqiang Lei, Jiancheng Lv, Tat-Seng Chua

**Abstract**: Visual Language Models (VLMs) are vulnerable to adversarial attacks, especially those from adversarial images, which is however under-explored in literature. To facilitate research on this critical safety problem, we first construct a new laRge-scale Adervsarial images dataset with Diverse hArmful Responses (RADAR), given that existing datasets are either small-scale or only contain limited types of harmful responses. With the new RADAR dataset, we further develop a novel and effective iN-time Embedding-based AdveRSarial Image DEtection (NEARSIDE) method, which exploits a single vector that distilled from the hidden states of VLMs, which we call the attacking direction, to achieve the detection of adversarial images against benign ones in the input. Extensive experiments with two victim VLMs, LLaVA and MiniGPT-4, well demonstrate the effectiveness, efficiency, and cross-model transferrability of our proposed method. Our code is available at https://github.com/mob-scu/RADAR-NEARSIDE

摘要: 视觉语言模型（VLM）很容易受到对抗性攻击，尤其是来自对抗性图像的攻击，但文献中对此尚未充分探讨。为了促进对这个关键安全问题的研究，我们首先构建一个具有多样性干扰响应（RADART）的新的大规模Adervsarial图像数据集，因为现有数据集要么小规模，要么仅包含有限类型的有害反应。利用新的雷达数据集，我们进一步开发了一种新颖且有效的基于iN时间嵌入的AdveRSarial Image Detect（NEARSIDE）方法，该方法利用从VLM的隐藏状态（我们称之为攻击方向）中提取的单个载体，以实现针对输入中良性图像的对抗图像的检测。对两个受害VLM（LLaVA和MiniGPT-4）进行的大量实验很好地证明了我们提出的方法的有效性、效率和跨模型可移植性。我们的代码可在https://github.com/mob-scu/RADAR-NEARSIDE上获取



## **2. Stealth edits to large language models**

对大型语言模型的隐形编辑 cs.AI

28 pages, 14 figures. Open source implementation:  https://github.com/qinghua-zhou/stealth-edits

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2406.12670v2) [paper-pdf](http://arxiv.org/pdf/2406.12670v2)

**Authors**: Oliver J. Sutton, Qinghua Zhou, Wei Wang, Desmond J. Higham, Alexander N. Gorban, Alexander Bastounis, Ivan Y. Tyukin

**Abstract**: We reveal the theoretical foundations of techniques for editing large language models, and present new methods which can do so without requiring retraining. Our theoretical insights show that a single metric (a measure of the intrinsic dimension of the model's features) can be used to assess a model's editability and reveals its previously unrecognised susceptibility to malicious stealth attacks. This metric is fundamental to predicting the success of a variety of editing approaches, and reveals new bridges between disparate families of editing methods. We collectively refer to these as stealth editing methods, because they directly update a model's weights to specify its response to specific known hallucinating prompts without affecting other model behaviour. By carefully applying our theoretical insights, we are able to introduce a new jet-pack network block which is optimised for highly selective model editing, uses only standard network operations, and can be inserted into existing networks. We also reveal the vulnerability of language models to stealth attacks: a small change to a model's weights which fixes its response to a single attacker-chosen prompt. Stealth attacks are computationally simple, do not require access to or knowledge of the model's training data, and therefore represent a potent yet previously unrecognised threat to redistributed foundation models. Extensive experimental results illustrate and support our methods and their theoretical underpinnings. Demos and source code are available at https://github.com/qinghua-zhou/stealth-edits.

摘要: 我们揭示了编辑大型语言模型的技术的理论基础，并提出了无需重新培训就能做到这一点的新方法。我们的理论见解表明，可以使用单一指标(模型特征的内在维度的衡量标准)来评估模型的可编辑性，并揭示其先前未被识别的对恶意隐形攻击的易感性。这一指标是预测各种编辑方法成功与否的基础，并揭示了不同编辑方法家族之间的新桥梁。我们将这些统称为隐形编辑方法，因为它们直接更新模型的权重，以指定其对特定已知幻觉提示的反应，而不影响其他模型的行为。通过仔细应用我们的理论见解，我们能够推出一种新的喷气式网络块，它针对高度选择性的模型编辑进行了优化，只使用标准的网络操作，并且可以插入到现有的网络中。我们还揭示了语言模型对隐形攻击的脆弱性：对模型的权重进行微小的更改即可修复其对单个攻击者选择的提示的响应。隐形攻击在计算上很简单，不需要访问或了解模型的训练数据，因此对重新分布的基础模型构成了以前未识别的强大威胁。大量的实验结果说明和支持了我们的方法及其理论基础。有关演示和源代码，请访问https://github.com/qinghua-zhou/stealth-edits.



## **3. HijackRAG: Hijacking Attacks against Retrieval-Augmented Large Language Models**

HijackRAG：针对检索增强大型语言模型的劫持攻击 cs.CR

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22832v1) [paper-pdf](http://arxiv.org/pdf/2410.22832v1)

**Authors**: Yucheng Zhang, Qinfeng Li, Tianyu Du, Xuhong Zhang, Xinkui Zhao, Zhengwen Feng, Jianwei Yin

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge, making them adaptable and cost-effective for various applications. However, the growing reliance on these systems also introduces potential security risks. In this work, we reveal a novel vulnerability, the retrieval prompt hijack attack (HijackRAG), which enables attackers to manipulate the retrieval mechanisms of RAG systems by injecting malicious texts into the knowledge database. When the RAG system encounters target questions, it generates the attacker's pre-determined answers instead of the correct ones, undermining the integrity and trustworthiness of the system. We formalize HijackRAG as an optimization problem and propose both black-box and white-box attack strategies tailored to different levels of the attacker's knowledge. Extensive experiments on multiple benchmark datasets show that HijackRAG consistently achieves high attack success rates, outperforming existing baseline attacks. Furthermore, we demonstrate that the attack is transferable across different retriever models, underscoring the widespread risk it poses to RAG systems. Lastly, our exploration of various defense mechanisms reveals that they are insufficient to counter HijackRAG, emphasizing the urgent need for more robust security measures to protect RAG systems in real-world deployments.

摘要: 检索-增强生成(RAG)系统通过集成外部知识来增强大型语言模型(LLM)，使它们能够适应各种应用并具有成本效益。然而，对这些系统的日益依赖也带来了潜在的安全风险。在这项工作中，我们揭示了一个新的漏洞--检索即时劫持攻击(HijackRAG)，该漏洞使攻击者能够通过向知识库中注入恶意文本来操纵RAG系统的检索机制。当RAG系统遇到目标问题时，它会生成攻击者的预定答案而不是正确的答案，从而破坏系统的完整性和可信性。我们将HijackRAG形式化为一个优化问题，并根据攻击者的不同知识水平提出了黑盒和白盒攻击策略。在多个基准数据集上的大量实验表明，HijackRAG始终具有较高的攻击成功率，性能优于现有的基线攻击。此外，我们证明了攻击可以在不同的取回器模型之间转移，强调了它对RAG系统构成的广泛风险。最后，我们对各种防御机制的探索表明，它们不足以对抗HijackRAG，强调迫切需要更强大的安全措施来保护现实世界部署中的RAG系统。



## **4. InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models**

InjecGuard：对标和缓解即时注射保障模型中的过度防御 cs.CL

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22770v1) [paper-pdf](http://arxiv.org/pdf/2410.22770v1)

**Authors**: Hao Li, Xiaogeng Liu, Chaowei Xiao

**Abstract**: Prompt injection attacks pose a critical threat to large language models (LLMs), enabling goal hijacking and data leakage. Prompt guard models, though effective in defense, suffer from over-defense -- falsely flagging benign inputs as malicious due to trigger word bias. To address this issue, we introduce NotInject, an evaluation dataset that systematically measures over-defense across various prompt guard models. NotInject contains 339 benign samples enriched with trigger words common in prompt injection attacks, enabling fine-grained evaluation. Our results show that state-of-the-art models suffer from over-defense issues, with accuracy dropping close to random guessing levels (60%). To mitigate this, we propose InjecGuard, a novel prompt guard model that incorporates a new training strategy, Mitigating Over-defense for Free (MOF), which significantly reduces the bias on trigger words. InjecGuard demonstrates state-of-the-art performance on diverse benchmarks including NotInject, surpassing the existing best model by 30.8%, offering a robust and open-source solution for detecting prompt injection attacks. The code and datasets are released at https://github.com/SaFoLab-WISC/InjecGuard.

摘要: 快速注入攻击对大型语言模型(LLM)构成严重威胁，导致目标劫持和数据泄露。即时保护模式虽然在防御方面有效，但也存在过度防御的问题--由于触发单词偏见，错误地将良性输入标记为恶意输入。为了解决这个问题，我们引入了NotInject，这是一个评估数据集，系统地测量各种提示防护模型中的过度防御。NotInject包含339个良性样本，丰富了提示注入攻击中常见的触发字，实现了细粒度评估。我们的结果表明，最先进的模型存在过度防御的问题，准确率下降到接近随机猜测的水平(60%)。为了缓解这一问题，我们提出了InjecGuard，一种新的提示守卫模型，它结合了新的训练策略，缓解了过度防御For Free(MOF)，大大减少了对触发词的偏见。InjecGuard在包括NotInject在内的各种基准测试上展示了最先进的性能，比现有最好的模型高出30.8%，为检测即时注入攻击提供了一个强大的开源解决方案。代码和数据集在https://github.com/SaFoLab-WISC/InjecGuard.上发布



## **5. Uncovering Coordinated Cross-Platform Information Operations Threatening the Integrity of the 2024 U.S. Presidential Election Online Discussion**

揭露威胁2024年美国总统选举在线讨论完整性的协调跨平台信息操作 cs.SI

First Monday 29(11), 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2409.15402v2) [paper-pdf](http://arxiv.org/pdf/2409.15402v2)

**Authors**: Marco Minici, Luca Luceri, Federico Cinus, Emilio Ferrara

**Abstract**: Information Operations (IOs) pose a significant threat to the integrity of democratic processes, with the potential to influence election-related online discourse. In anticipation of the 2024 U.S. presidential election, we present a study aimed at uncovering the digital traces of coordinated IOs on $\mathbb{X}$ (formerly Twitter). Using our machine learning framework for detecting online coordination, we analyze a dataset comprising election-related conversations on $\mathbb{X}$ from May 2024. This reveals a network of coordinated inauthentic actors, displaying notable similarities in their link-sharing behaviors. Our analysis shows concerted efforts by these accounts to disseminate misleading, redundant, and biased information across the Web through a coordinated cross-platform information operation: The links shared by this network frequently direct users to other social media platforms or suspicious websites featuring low-quality political content and, in turn, promoting the same $\mathbb{X}$ and YouTube accounts. Members of this network also shared deceptive images generated by AI, accompanied by language attacking political figures and symbolic imagery intended to convey power and dominance. While $\mathbb{X}$ has suspended a subset of these accounts, more than 75% of the coordinated network remains active. Our findings underscore the critical role of developing computational models to scale up the detection of threats on large social media platforms, and emphasize the broader implications of these techniques to detect IOs across the wider Web.

摘要: 信息业务对民主进程的完整性构成重大威胁，有可能影响与选举有关的网上言论。在对2024年美国总统大选的预期中，我们提出了一项研究，旨在揭示$\mathbb{X}$(前身为Twitter)上协调iOS的数字痕迹。使用我们用于检测在线协调的机器学习框架，我们分析了一个包含2024年5月以来$\mathbb{X}$的选举相关对话的数据集。这揭示了一个协调的不真实行为者网络，显示出他们在链接分享行为上的显著相似之处。我们的分析显示，这些账户协同努力，通过协调的跨平台信息操作在网络上传播误导性、冗余和有偏见的信息：该网络共享的链接经常将用户定向到其他社交媒体平台或具有低质量政治内容的可疑网站，进而推广相同的$\mathbb{X}$和YouTube账户。该网络的成员还分享了人工智能生成的欺骗性图像，并伴随着攻击政治人物的语言和旨在传达权力和主导地位的象征性图像。虽然$\mathbb{X}$已暂停这些帐户的子集，但超过75%的协调网络仍处于活动状态。我们的发现强调了开发计算模型以扩大大型社交媒体平台上的威胁检测的关键作用，并强调了这些技术在更广泛的网络上检测iOS的更广泛影响。



## **6. SVIP: Towards Verifiable Inference of Open-source Large Language Models**

SVIP：迈向开源大型语言模型的可验证推理 cs.LG

20 pages

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22307v1) [paper-pdf](http://arxiv.org/pdf/2410.22307v1)

**Authors**: Yifan Sun, Yuhang Li, Yue Zhang, Yuchen Jin, Huan Zhang

**Abstract**: Open-source Large Language Models (LLMs) have recently demonstrated remarkable capabilities in natural language understanding and generation, leading to widespread adoption across various domains. However, their increasing model sizes render local deployment impractical for individual users, pushing many to rely on computing service providers for inference through a blackbox API. This reliance introduces a new risk: a computing provider may stealthily substitute the requested LLM with a smaller, less capable model without consent from users, thereby delivering inferior outputs while benefiting from cost savings. In this paper, we formalize the problem of verifiable inference for LLMs. Existing verifiable computing solutions based on cryptographic or game-theoretic techniques are either computationally uneconomical or rest on strong assumptions. We introduce SVIP, a secret-based verifiable LLM inference protocol that leverages intermediate outputs from LLM as unique model identifiers. By training a proxy task on these outputs and requiring the computing provider to return both the generated text and the processed intermediate outputs, users can reliably verify whether the computing provider is acting honestly. In addition, the integration of a secret mechanism further enhances the security of our protocol. We thoroughly analyze our protocol under multiple strong and adaptive adversarial scenarios. Our extensive experiments demonstrate that SVIP is accurate, generalizable, computationally efficient, and resistant to various attacks. Notably, SVIP achieves false negative rates below 5% and false positive rates below 3%, while requiring less than 0.01 seconds per query for verification.

摘要: 开源的大型语言模型(LLM)最近在自然语言理解和生成方面表现出了非凡的能力，导致了在各个领域的广泛采用。然而，它们不断增长的模型规模使得本地部署对个人用户来说是不现实的，促使许多人依赖计算服务提供商通过黑盒API进行推理。这种依赖带来了新的风险：计算提供商可能会在未经用户同意的情况下，悄悄地用较小、功能较差的模型替换所请求的LLM，从而在提供劣质产出的同时受益于成本节约。本文对LLMS的可验证推理问题进行了形式化描述。现有的基于密码学或博弈论技术的可验证计算解决方案要么在计算上不经济，要么依赖于强有力的假设。我们引入了SVIP，这是一个基于秘密的可验证LLM推理协议，它利用LLM的中间输出作为唯一的模型标识符。通过对这些输出训练代理任务并要求计算提供商返回生成的文本和处理的中间输出，用户可以可靠地验证计算提供商是否诚实行事。此外，秘密机制的集成进一步增强了协议的安全性。我们深入分析了我们的协议在多种强和自适应对抗场景下的性能。大量实验表明，SVIP算法具有较高的准确性、通用性、计算效率和抵抗各种攻击的能力。值得注意的是，SVIP实现了5%以下的假阴性率和3%以下的假阳性率，而每次查询验证所需的时间不到0.01秒。



## **7. Embedding-based classifiers can detect prompt injection attacks**

基于嵌入的分类器可以检测提示注入攻击 cs.CR

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22284v1) [paper-pdf](http://arxiv.org/pdf/2410.22284v1)

**Authors**: Md. Ahsan Ayub, Subhabrata Majumdar

**Abstract**: Large Language Models (LLMs) are seeing significant adoption in every type of organization due to their exceptional generative capabilities. However, LLMs are found to be vulnerable to various adversarial attacks, particularly prompt injection attacks, which trick them into producing harmful or inappropriate content. Adversaries execute such attacks by crafting malicious prompts to deceive the LLMs. In this paper, we propose a novel approach based on embedding-based Machine Learning (ML) classifiers to protect LLM-based applications against this severe threat. We leverage three commonly used embedding models to generate embeddings of malicious and benign prompts and utilize ML classifiers to predict whether an input prompt is malicious. Out of several traditional ML methods, we achieve the best performance with classifiers built using Random Forest and XGBoost. Our classifiers outperform state-of-the-art prompt injection classifiers available in open-source implementations, which use encoder-only neural networks.

摘要: 大型语言模型(LLM)由于其非凡的生成能力，在每种类型的组织中都得到了大量采用。然而，LLM被发现容易受到各种对抗性攻击，特别是即时注入攻击，这些攻击会诱使它们产生有害或不适当的内容。攻击者通过精心编制恶意提示来欺骗LLM，从而执行此类攻击。在本文中，我们提出了一种基于嵌入的机器学习(ML)分类器的新方法来保护基于LLM的应用程序免受这种严重威胁。我们利用三种常用的嵌入模型来生成恶意提示和良性提示的嵌入，并利用ML分类器来预测输入提示是否为恶意提示。在几种传统的最大似然分类方法中，我们使用随机森林和XGBoost构建的分类器取得了最好的性能。我们的分类器比开源实现中可用的最先进的提示注入分类器性能更好，后者使用仅限编码器的神经网络。



## **8. AmpleGCG-Plus: A Strong Generative Model of Adversarial Suffixes to Jailbreak LLMs with Higher Success Rates in Fewer Attempts**

AmpleGCG-Plus：越狱LLC的对抗性后缀的强生成模型，以更少的尝试获得更高的成功率 cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22143v1) [paper-pdf](http://arxiv.org/pdf/2410.22143v1)

**Authors**: Vishal Kumar, Zeyi Liao, Jaylen Jones, Huan Sun

**Abstract**: Although large language models (LLMs) are typically aligned, they remain vulnerable to jailbreaking through either carefully crafted prompts in natural language or, interestingly, gibberish adversarial suffixes. However, gibberish tokens have received relatively less attention despite their success in attacking aligned LLMs. Recent work, AmpleGCG~\citep{liao2024amplegcg}, demonstrates that a generative model can quickly produce numerous customizable gibberish adversarial suffixes for any harmful query, exposing a range of alignment gaps in out-of-distribution (OOD) language spaces. To bring more attention to this area, we introduce AmpleGCG-Plus, an enhanced version that achieves better performance in fewer attempts. Through a series of exploratory experiments, we identify several training strategies to improve the learning of gibberish suffixes. Our results, verified under a strict evaluation setting, show that it outperforms AmpleGCG on both open-weight and closed-source models, achieving increases in attack success rate (ASR) of up to 17\% in the white-box setting against Llama-2-7B-chat, and more than tripling ASR in the black-box setting against GPT-4. Notably, AmpleGCG-Plus jailbreaks the newer GPT-4o series of models at similar rates to GPT-4, and, uncovers vulnerabilities against the recently proposed circuit breakers defense. We publicly release AmpleGCG-Plus along with our collected training datasets.

摘要: 尽管大型语言模型(LLM)通常是一致的，但它们仍然很容易通过精心设计的自然语言提示或有趣的胡言乱语对抗性后缀越狱。然而，令人费解的令牌尽管成功地攻击了对齐的LLM，但受到的关注相对较少。最近的工作，AmpleGCG~\Citep{Lio2024Amplegcg}，证明了生成模型可以为任何有害的查询快速生成大量可定制的胡言乱语对抗性后缀，从而暴露出分布外(OOD)语言空间中的一系列对齐差距。为了引起人们对这一领域的更多关注，我们推出了AmpleGCG-Plus，这是一个增强版本，在较少的尝试中获得了更好的性能。通过一系列的探索性实验，我们确定了几种训练策略来提高乱码后缀的学习效果。在严格的评估设置下验证的结果表明，它在开源和闭源模型上都优于AmpleGCG，在白盒环境下相对于Llama-2-7B-Chat的攻击成功率(ASR)提高了17%，在黑盒环境下相对于GPT-4的攻击成功率(ASR)提高了两倍以上。值得注意的是，AmpleGCG-Plus以类似于GPT-4的速度监禁了较新的GPT-4o系列型号，并揭示了针对最近提出的断路器防御的漏洞。我们公开发布AmpleGCG-Plus以及我们收集的训练数据集。



## **9. Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents**

小心你的特工！调查对LLM代理的后门威胁 cs.CR

Accepted at NeurIPS 2024, camera ready version. Code and data are  available at https://github.com/lancopku/agent-backdoor-attacks

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2402.11208v2) [paper-pdf](http://arxiv.org/pdf/2402.11208v2)

**Authors**: Wenkai Yang, Xiaohan Bi, Yankai Lin, Sishuo Chen, Jie Zhou, Xu Sun

**Abstract**: Driven by the rapid development of Large Language Models (LLMs), LLM-based agents have been developed to handle various real-world applications, including finance, healthcare, and shopping, etc. It is crucial to ensure the reliability and security of LLM-based agents during applications. However, the safety issues of LLM-based agents are currently under-explored. In this work, we take the first step to investigate one of the typical safety threats, backdoor attack, to LLM-based agents. We first formulate a general framework of agent backdoor attacks, then we present a thorough analysis of different forms of agent backdoor attacks. Specifically, compared with traditional backdoor attacks on LLMs that are only able to manipulate the user inputs and model outputs, agent backdoor attacks exhibit more diverse and covert forms: (1) From the perspective of the final attacking outcomes, the agent backdoor attacker can not only choose to manipulate the final output distribution, but also introduce the malicious behavior in an intermediate reasoning step only, while keeping the final output correct. (2) Furthermore, the former category can be divided into two subcategories based on trigger locations, in which the backdoor trigger can either be hidden in the user query or appear in an intermediate observation returned by the external environment. We implement the above variations of agent backdoor attacks on two typical agent tasks including web shopping and tool utilization. Extensive experiments show that LLM-based agents suffer severely from backdoor attacks and such backdoor vulnerability cannot be easily mitigated by current textual backdoor defense algorithms. This indicates an urgent need for further research on the development of targeted defenses against backdoor attacks on LLM-based agents. Warning: This paper may contain biased content.

摘要: 在大型语言模型(LLM)快速发展的推动下，基于LLM的代理被开发出来处理各种现实世界的应用，包括金融、医疗和购物等。然而，基于LLM的制剂的安全性问题目前还没有得到充分的研究。在这项工作中，我们首先调查了LLM代理面临的一种典型的安全威胁--后门攻击。我们首先建立了代理后门攻击的一般框架，然后深入分析了代理后门攻击的不同形式。具体地说，与传统的仅能操纵用户输入和模型输出的后门攻击相比，代理后门攻击表现出更多样和隐蔽的形式：(1)从最终攻击结果来看，代理后门攻击者不仅可以选择操纵最终输出分布，而且只能在中间推理步骤中引入恶意行为，同时保持最终输出的正确性。(2)此外，根据触发器的位置，前者可以分为两个子类，其中后门触发器可以隐藏在用户查询中，也可以出现在外部环境返回的中间观察中。我们在两个典型的代理任务上实现了上述变体的代理后门攻击，包括网上购物和工具利用。大量的实验表明，基于LLM的代理受到严重的后门攻击，而这种后门漏洞不能通过现有的文本后门防御算法轻松缓解。这表明迫切需要进一步研究针对基于LLM的特工的后门攻击开发有针对性的防御措施。警告：此论文可能包含有偏见的内容。



## **10. Waterfall: Framework for Robust and Scalable Text Watermarking and Provenance for LLMs**

Waterfall：LLM稳健且可扩展的文本水印和起源框架 cs.CR

Accepted to EMNLP 2024 Main Conference

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2407.04411v2) [paper-pdf](http://arxiv.org/pdf/2407.04411v2)

**Authors**: Gregory Kang Ruey Lau, Xinyuan Niu, Hieu Dao, Jiangwei Chen, Chuan-Sheng Foo, Bryan Kian Hsiang Low

**Abstract**: Protecting intellectual property (IP) of text such as articles and code is increasingly important, especially as sophisticated attacks become possible, such as paraphrasing by large language models (LLMs) or even unauthorized training of LLMs on copyrighted text to infringe such IP. However, existing text watermarking methods are not robust enough against such attacks nor scalable to millions of users for practical implementation. In this paper, we propose Waterfall, the first training-free framework for robust and scalable text watermarking applicable across multiple text types (e.g., articles, code) and languages supportable by LLMs, for general text and LLM data provenance. Waterfall comprises several key innovations, such as being the first to use LLM as paraphrasers for watermarking along with a novel combination of techniques that are surprisingly effective in achieving robust verifiability and scalability. We empirically demonstrate that Waterfall achieves significantly better scalability, robust verifiability, and computational efficiency compared to SOTA article-text watermarking methods, and showed how it could be directly applied to the watermarking of code. We also demonstrated that Waterfall can be used for LLM data provenance, where the watermarks of LLM training data can be detected in LLM output, allowing for detection of unauthorized use of data for LLM training and potentially enabling model-centric watermarking of open-sourced LLMs which has been a limitation of existing LLM watermarking works. Our code is available at https://github.com/aoi3142/Waterfall.

摘要: 保护文章和代码等文本的知识产权(IP)越来越重要，特别是在可能进行复杂攻击的情况下，例如利用大型语言模型(LLM)进行释义，甚至未经授权对受版权保护的文本进行LLM培训，以侵犯此类IP。然而，现有的文本水印方法对此类攻击不够健壮，也不能扩展到数百万用户进行实际实现。在本文中，我们提出了瀑布，这是第一个无训练的文本水印框架，适用于LLMS支持的多种文本类型(例如，文章、代码)和语言，用于一般文本和LLM数据来源。瀑布由几项关键创新组成，例如第一个使用LLM作为水印解释程序，以及在实现强大的可验证性和可扩展性方面出人意料地有效的技术组合。实验证明，与SOTA的文章-文本水印方法相比，瀑布算法具有更好的可扩展性、健壮性、可验证性和计算效率，并且可以直接应用于代码的水印。我们还演示了瀑布可以用于LLM数据起源，其中LLM训练数据的水印可以在LLM输出中检测到，允许检测未经授权使用数据进行LLM训练，并潜在地实现了以模型为中心的开放源代码LLMS水印，这一直是现有LLM水印工作的限制。我们的代码可以在https://github.com/aoi3142/Waterfall.上找到



## **11. Enhancing Adversarial Attacks through Chain of Thought**

通过思维链增强对抗性攻击 cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21791v1) [paper-pdf](http://arxiv.org/pdf/2410.21791v1)

**Authors**: Jingbo Su

**Abstract**: Large language models (LLMs) have demonstrated impressive performance across various domains but remain susceptible to safety concerns. Prior research indicates that gradient-based adversarial attacks are particularly effective against aligned LLMs and the chain of thought (CoT) prompting can elicit desired answers through step-by-step reasoning. This paper proposes enhancing the robustness of adversarial attacks on aligned LLMs by integrating CoT prompts with the greedy coordinate gradient (GCG) technique. Using CoT triggers instead of affirmative targets stimulates the reasoning abilities of backend LLMs, thereby improving the transferability and universality of adversarial attacks. We conducted an ablation study comparing our CoT-GCG approach with Amazon Web Services auto-cot. Results revealed our approach outperformed both the baseline GCG attack and CoT prompting. Additionally, we used Llama Guard to evaluate potentially harmful interactions, providing a more objective risk assessment of entire conversations compared to matching outputs to rejection phrases. The code of this paper is available at https://github.com/sujingbo0217/CS222W24-LLM-Attack.

摘要: 大型语言模型(LLM)在各个领域都表现出了令人印象深刻的表现，但仍然容易受到安全问题的影响。以往的研究表明，基于梯度的对抗性攻击对于对齐的LLM特别有效，而思维链(COT)提示可以通过循序渐进的推理获得期望的答案。提出了一种将CoT提示与贪婪坐标梯度(GCG)技术相结合的方法，以增强对齐LLMS的敌意攻击的稳健性。使用CoT触发器代替肯定目标，刺激了后端LLMS的推理能力，从而提高了对抗性攻击的可转移性和通用性。我们进行了一项烧蚀研究，将我们的COT-GCG方法与Amazon Web Services自动COT进行了比较。结果显示，我们的方法比基线GCG攻击和COT提示都要好。此外，我们使用Llama Guard来评估潜在的有害交互，与将输出与拒绝短语匹配相比，提供了对整个对话的更客观的风险评估。本文的代码可在https://github.com/sujingbo0217/CS222W24-LLM-Attack.上找到



## **12. Vaccine: Perturbation-aware Alignment for Large Language Models against Harmful Fine-tuning Attack**

疫苗：大型语言模型的扰动感知对齐以对抗有害的微调攻击 cs.LG

Rejected by ICML2024. Accepted by NeurIPS2024

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2402.01109v5) [paper-pdf](http://arxiv.org/pdf/2402.01109v5)

**Authors**: Tiansheng Huang, Sihao Hu, Ling Liu

**Abstract**: The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompts. Our code is available at \url{https://github.com/git-disl/Vaccine}.

摘要: 精调即服务的新范式为大型语言模型(LLM)引入了一个新的攻击面：用户上传的少量有害数据就可以很容易地欺骗精调，产生一个破坏对齐的模型。我们进行了实证分析，发现了一种有害的嵌入漂移现象，揭示了排列断裂效应的可能原因。受我们发现的启发，我们提出了Vaccine，一种扰动感知的对齐技术，以降低用户精调的安全风险。Vaccine的核心思想是通过在比对阶段逐步向其添加精心制作的扰动来产生不变的隐藏嵌入。这使嵌入能够在精细调整阶段抵御来自未清理的用户数据的有害干扰。我们在开源主流LLMS(如Llama2、Opt、Vicuna)上的实验结果表明，疫苗可以提高对有害提示导致的嵌入漂移的健壮性，同时保留对良性提示的推理能力。我们的代码可在\url{https://github.com/git-disl/Vaccine}.



## **13. Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey**

针对大型语言模型的有害微调攻击和防御：调查 cs.CR

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2409.18169v4) [paper-pdf](http://arxiv.org/pdf/2409.18169v4)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Recent research demonstrates that the nascent fine-tuning-as-a-service business model exposes serious safety concerns -- fine-tuning over a few harmful data uploaded by the users can compromise the safety alignment of the model. The attack, known as harmful fine-tuning, has raised a broad research interest among the community. However, as the attack is still new, \textbf{we observe from our miserable submission experience that there are general misunderstandings within the research community.} We in this paper aim to clear some common concerns for the attack setting, and formally establish the research problem. Specifically, we first present the threat model of the problem, and introduce the harmful fine-tuning attack and its variants. Then we systematically survey the existing literature on attacks/defenses/mechanical analysis of the problem. Finally, we outline future research directions that might contribute to the development of the field. Additionally, we present a list of questions of interest, which might be useful to refer to when reviewers in the peer review process question the realism of the experiment/attack/defense setting. A curated list of relevant papers is maintained and made accessible at: \url{https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers}.

摘要: 最近的研究表明，新兴的微调即服务商业模式暴露了严重的安全问题--对用户上传的几个有害数据进行微调可能会损害该模型的安全一致性。这一被称为有害微调的攻击在社区中引起了广泛的研究兴趣。然而，由于攻击仍然是新的，\extbf{我们从悲惨的提交经验中观察到，研究界普遍存在误解。}我们在本文中旨在澄清一些对攻击设置的共同关注，并正式确立研究问题。具体地说，我们首先给出了问题的威胁模型，并介绍了有害的微调攻击及其变体。然后，我们系统地综述了现有的关于攻击/防御/机械分析问题的文献。最后，我们概述了未来的研究方向，可能有助于该领域的发展。此外，我们提供了一个感兴趣的问题列表，当同行审查过程中的评审者质疑实验/攻击/防御设置的真实性时，这些问题可能会有用。相关论文的精选清单可在以下网址查阅：\url{https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers}.



## **14. Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning Attack**

Lisa：大型语言模型的懒惰安全调整以应对有害的微调攻击 cs.LG

Accepted by NeurIPS2024. arXiv admin note: substantial text overlap  with arXiv:2402.01109

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2405.18641v5) [paper-pdf](http://arxiv.org/pdf/2405.18641v5)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Recent studies show that Large Language Models (LLMs) with safety alignment can be jail-broken by fine-tuning on a dataset mixed with harmful data. First time in the literature, we show that the jail-broken effect can be mitigated by separating states in the finetuning stage to optimize the alignment and user datasets. Unfortunately, our subsequent study shows that this simple Bi-State Optimization (BSO) solution experiences convergence instability when steps invested in its alignment state is too small, leading to downgraded alignment performance. By statistical analysis, we show that the \textit{excess drift} towards consensus could be a probable reason for the instability. To remedy this issue, we propose \textbf{L}azy(\textbf{i}) \textbf{s}afety \textbf{a}lignment (\textbf{Lisa}), which introduces a proximal term to constraint the drift of each state. Theoretically, the benefit of the proximal term is supported by the convergence analysis, wherein we show that a sufficient large proximal factor is necessary to guarantee Lisa's convergence. Empirically, our results on four downstream finetuning tasks show that Lisa with a proximal term can significantly increase alignment performance while maintaining the LLM's accuracy on the user tasks. Code is available at \url{https://github.com/git-disl/Lisa}.

摘要: 最近的研究表明，通过对混合了有害数据的数据集进行微调，安全对齐的大型语言模型(LLM)可以越狱。在文献中，我们第一次证明了可以通过在精调阶段分离状态来优化比对和用户数据集来缓解越狱效应。不幸的是，我们随后的研究表明，当在其对齐状态中投入的步长太小时，这种简单的双状态优化(BSO)解决方案会经历收敛不稳定，从而导致对齐性能下降。通过统计分析，我们发现，向共识的过度漂移可能是导致不稳定的一个可能原因。为了解决这个问题，我们提出了Textbf{L}azy(\Textbf{i})\Textbf{S}安全对齐(Textbf{Lisa})，它引入了一个近邻项来约束每个态的漂移。理论上，近似项的好处得到了收敛分析的支持，其中我们证明了一个足够大的近似项是保证LISA收敛的必要条件。经验性地，我们在四个下游精调任务上的结果表明，具有近端项的LISA可以显著提高对齐性能，同时保持LLM在用户任务上的准确性。代码位于\url{https://github.com/git-disl/Lisa}.



## **15. Fine-tuning Large Language Models for DGA and DNS Exfiltration Detection**

用于DGA和DNS溢出检测的微调大型语言模型 cs.CR

Accepted in Proceedings of the Workshop at AI for Cyber Threat  Intelligence (WAITI), 2024

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21723v1) [paper-pdf](http://arxiv.org/pdf/2410.21723v1)

**Authors**: Md Abu Sayed, Asif Rahman, Christopher Kiekintveld, Sebastian Garcia

**Abstract**: Domain Generation Algorithms (DGAs) are malicious techniques used by malware to dynamically generate seemingly random domain names for communication with Command & Control (C&C) servers. Due to the fast and simple generation of DGA domains, detection methods must be highly efficient and precise to be effective. Large Language Models (LLMs) have demonstrated their proficiency in real-time detection tasks, making them ideal candidates for detecting DGAs. Our work validates the effectiveness of fine-tuned LLMs for detecting DGAs and DNS exfiltration attacks. We developed LLM models and conducted comprehensive evaluation using a diverse dataset comprising 59 distinct real-world DGA malware families and normal domain data. Our LLM model significantly outperformed traditional natural language processing techniques, especially in detecting unknown DGAs. We also evaluated its performance on DNS exfiltration datasets, demonstrating its effectiveness in enhancing cybersecurity measures. To the best of our knowledge, this is the first work that empirically applies LLMs for DGA and DNS exfiltration detection.

摘要: 域生成算法(DGA)是恶意软件用来动态生成看似随机的域名以与命令与控制(C&C)服务器通信的恶意技术。由于DGA结构域的快速而简单的生成，检测方法必须高效和精确才能有效。大型语言模型(LLM)已经证明了它们在实时检测任务中的熟练程度，使它们成为检测DGA的理想候选者。我们的工作验证了微调的LLMS在检测DGA和DNS渗出攻击方面的有效性。我们开发了LLM模型，并使用包含59个不同的真实DGA恶意软件家族和正常域数据的不同数据集进行了全面评估。我们的LLM模型显著优于传统的自然语言处理技术，特别是在检测未知DGA方面。我们还评估了它在DNS渗出数据集上的性能，展示了它在加强网络安全措施方面的有效性。据我们所知，这是第一个经验性地将LLMS应用于DGA和DNS渗出检测的工作。



## **16. Bileve: Securing Text Provenance in Large Language Models Against Spoofing with Bi-level Signature**

Bileve：通过双层签名保护大型语言模型中的文本出处，防止欺骗 cs.CR

NeurIPS 2024 camera-ready

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2406.01946v3) [paper-pdf](http://arxiv.org/pdf/2406.01946v3)

**Authors**: Tong Zhou, Xuandong Zhao, Xiaolin Xu, Shaolei Ren

**Abstract**: Text watermarks for large language models (LLMs) have been commonly used to identify the origins of machine-generated content, which is promising for assessing liability when combating deepfake or harmful content. While existing watermarking techniques typically prioritize robustness against removal attacks, unfortunately, they are vulnerable to spoofing attacks: malicious actors can subtly alter the meanings of LLM-generated responses or even forge harmful content, potentially misattributing blame to the LLM developer. To overcome this, we introduce a bi-level signature scheme, Bileve, which embeds fine-grained signature bits for integrity checks (mitigating spoofing attacks) as well as a coarse-grained signal to trace text sources when the signature is invalid (enhancing detectability) via a novel rank-based sampling strategy. Compared to conventional watermark detectors that only output binary results, Bileve can differentiate 5 scenarios during detection, reliably tracing text provenance and regulating LLMs. The experiments conducted on OPT-1.3B and LLaMA-7B demonstrate the effectiveness of Bileve in defeating spoofing attacks with enhanced detectability. Code is available at https://github.com/Tongzhou0101/Bileve-official.

摘要: 大型语言模型(LLM)的文本水印通常用于识别机器生成内容的来源，这有望在打击深度虚假或有害内容时评估责任。虽然现有的水印技术通常将健壮性放在免受删除攻击的优先位置，但不幸的是，它们容易受到欺骗性攻击：恶意行为者可以巧妙地更改LLM生成的响应的含义，甚至伪造有害内容，可能会将责任错误地归咎于LLM开发人员。为了克服这一问题，我们提出了一种双层签名方案BiLEVE，该方案通过一种新颖的基于等级的采样策略嵌入细粒度的签名比特用于完整性检查(缓解欺骗攻击)，并在签名无效时嵌入粗粒度的信号来跟踪文本来源(增强了可检测性)。与传统的只输出二进制结果的水印检测器相比，BiLEVE在检测过程中可以区分5种场景，可靠地追踪文本来源和规范LLM。在OPT-1.3B和LLAMA-7B上进行的实验证明了BiLEVE在抵抗欺骗攻击方面的有效性，并增强了可检测性。代码可在https://github.com/Tongzhou0101/Bileve-official.上找到



## **17. CFSafety: Comprehensive Fine-grained Safety Assessment for LLMs**

CFSafety：LLM的全面细粒度安全评估 cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21695v1) [paper-pdf](http://arxiv.org/pdf/2410.21695v1)

**Authors**: Zhihao Liu, Chenhui Hu

**Abstract**: As large language models (LLMs) rapidly evolve, they bring significant conveniences to our work and daily lives, but also introduce considerable safety risks. These models can generate texts with social biases or unethical content, and under specific adversarial instructions, may even incite illegal activities. Therefore, rigorous safety assessments of LLMs are crucial. In this work, we introduce a safety assessment benchmark, CFSafety, which integrates 5 classic safety scenarios and 5 types of instruction attacks, totaling 10 categories of safety questions, to form a test set with 25k prompts. This test set was used to evaluate the natural language generation (NLG) capabilities of LLMs, employing a combination of simple moral judgment and a 1-5 safety rating scale for scoring. Using this benchmark, we tested eight popular LLMs, including the GPT series. The results indicate that while GPT-4 demonstrated superior safety performance, the safety effectiveness of LLMs, including this model, still requires improvement. The data and code associated with this study are available on GitHub.

摘要: 随着大型语言模型的快速发展，它们在给我们的工作和日常生活带来极大便利的同时，也带来了相当大的安全隐患。这些模式可能会产生带有社会偏见或不道德内容的文本，在特定的敌对指令下，甚至可能煽动非法活动。因此，对LLMS进行严格的安全评估至关重要。在这项工作中，我们引入了一个安全评估基准CFSafe，它集成了5个经典的安全场景和5种类型的指令攻击，共计10类安全问题，形成了一个包含25K提示的测试集。该测试集被用来评估LLMS的自然语言生成(NLG)能力，采用简单的道德判断和1-5安全等级评分相结合的方式进行评分。使用这个基准，我们测试了八个流行的LLM，包括GPT系列。结果表明，尽管GPT-4表现出了优越的安全性能，但包括该模型在内的LLMS的安全有效性仍需改进。与这项研究相关的数据和代码可在GitHub上获得。



## **18. FATH: Authentication-based Test-time Defense against Indirect Prompt Injection Attacks**

FASH：基于身份验证的测试时防御间接提示注入攻击 cs.CR

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21492v1) [paper-pdf](http://arxiv.org/pdf/2410.21492v1)

**Authors**: Jiongxiao Wang, Fangzhou Wu, Wendi Li, Jinsheng Pan, Edward Suh, Z. Morley Mao, Muhao Chen, Chaowei Xiao

**Abstract**: Large language models (LLMs) have been widely deployed as the backbone with additional tools and text information for real-world applications. However, integrating external information into LLM-integrated applications raises significant security concerns. Among these, prompt injection attacks are particularly threatening, where malicious instructions injected in the external text information can exploit LLMs to generate answers as the attackers desire. While both training-time and test-time defense methods have been developed to mitigate such attacks, the unaffordable training costs associated with training-time methods and the limited effectiveness of existing test-time methods make them impractical. This paper introduces a novel test-time defense strategy, named Formatting AuThentication with Hash-based tags (FATH). Unlike existing approaches that prevent LLMs from answering additional instructions in external text, our method implements an authentication system, requiring LLMs to answer all received instructions with a security policy and selectively filter out responses to user instructions as the final output. To achieve this, we utilize hash-based authentication tags to label each response, facilitating accurate identification of responses according to the user's instructions and improving the robustness against adaptive attacks. Comprehensive experiments demonstrate that our defense method can effectively defend against indirect prompt injection attacks, achieving state-of-the-art performance under Llama3 and GPT3.5 models across various attack methods. Our code is released at: https://github.com/Jayfeather1024/FATH

摘要: 大型语言模型(LLM)已被广泛部署为主干，并为实际应用程序提供额外的工具和文本信息。然而，将外部信息集成到LLM集成的应用程序中会引发重大的安全问题。其中，即时注入攻击尤其具有威胁性，在外部文本信息中注入的恶意指令可以利用LLMS生成攻击者想要的答案。虽然已经开发了训练时间和测试时间防御方法来缓解此类攻击，但与训练时间方法相关的难以负担的训练成本以及现有测试时间方法的有限有效性使它们变得不切实际。提出了一种新的测试时间防御策略--基于Hash标签的格式化认证(FATH)。与现有的阻止LLMS应答外部文本中的额外指令的方法不同，我们的方法实现了一个身份验证系统，要求LLMS用安全策略应答所有接收到的指令，并选择性地过滤对用户指令的响应作为最终输出。为了实现这一点，我们使用基于散列的身份验证标签来标记每个响应，便于根据用户的指令准确识别响应，并提高了对自适应攻击的健壮性。综合实验表明，我们的防御方法能够有效防御间接即时注入攻击，在Llama3和GPT3.5模型下通过各种攻击方法获得了最好的性能。我们的代码发布在：https://github.com/Jayfeather1024/FATH



## **19. Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning**

简单性盛行：重新思考LLM忘记学习的负偏好优化 cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.07163v2) [paper-pdf](http://arxiv.org/pdf/2410.07163v2)

**Authors**: Chongyu Fan, Jiancheng Liu, Licong Lin, Jinghan Jia, Ruiqi Zhang, Song Mei, Sijia Liu

**Abstract**: In this work, we address the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences and associated model capabilities (e.g., copyrighted data or harmful content generation) while preserving essential model utilities, without the need for retraining from scratch. Despite the growing need for LLM unlearning, a principled optimization framework remains lacking. To this end, we revisit the state-of-the-art approach, negative preference optimization (NPO), and identify the issue of reference model bias, which could undermine NPO's effectiveness, particularly when unlearning forget data of varying difficulty. Given that, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that 'simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We also provide deeper insights into SimNPO's advantages, supported by analysis using mixtures of Markov chains. Furthermore, we present extensive experiments validating SimNPO's superiority over existing unlearning baselines in benchmarks like TOFU and MUSE, and robustness against relearning attacks. Codes are available at https://github.com/OPTML-Group/Unlearn-Simple.

摘要: 在这项工作中，我们解决了大型语言模型(LLM)遗忘的问题，旨在消除不必要的数据影响和相关的模型能力(例如，受版权保护的数据或有害内容生成)，同时保留基本的模型实用程序，而不需要从头开始重新培训。尽管对LLM遗忘的需求越来越大，但仍然缺乏一个有原则的优化框架。为此，我们回顾了最先进的方法，负偏好优化(NPO)，并确定了参考模型偏差的问题，这可能会削弱NPO的有效性，特别是当遗忘遗忘数据的不同难度时。鉴于此，我们提出了一个简单而有效的遗忘优化框架，称为SimNPO，表明在消除对参考模型的依赖(通过简单偏好优化的镜头)时的“简单性”有利于遗忘。我们还提供了对SimNPO的优势的更深层次的见解，并通过使用马尔可夫链的混合分析提供了支持。此外，我们提供了大量的实验，验证了SimNPO在豆腐和缪斯等基准测试中相对于现有遗忘基线的优势，以及对重新学习攻击的健壮性。有关代码，请访问https://github.com/OPTML-Group/Unlearn-Simple.



## **20. Securing Multi-turn Conversational Language Models From Distributed Backdoor Triggers**

保护多轮对话语言模型免受分布式后门触发器的影响 cs.CL

Findings of EMNLP 2024

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2407.04151v2) [paper-pdf](http://arxiv.org/pdf/2407.04151v2)

**Authors**: Terry Tong, Jiashu Xu, Qin Liu, Muhao Chen

**Abstract**: Large language models (LLMs) have acquired the ability to handle longer context lengths and understand nuances in text, expanding their dialogue capabilities beyond a single utterance. A popular user-facing application of LLMs is the multi-turn chat setting. Though longer chat memory and better understanding may seemingly benefit users, our paper exposes a vulnerability that leverages the multi-turn feature and strong learning ability of LLMs to harm the end-user: the backdoor. We demonstrate that LLMs can capture the combinational backdoor representation. Only upon presentation of triggers together does the backdoor activate. We also verify empirically that this representation is invariant to the position of the trigger utterance. Subsequently, inserting a single extra token into two utterances of 5%of the data can cause over 99% Attack Success Rate (ASR). Our results with 3 triggers demonstrate that this framework is generalizable, compatible with any trigger in an adversary's toolbox in a plug-and-play manner. Defending the backdoor can be challenging in the chat setting because of the large input and output space. Our analysis indicates that the distributed backdoor exacerbates the current challenges by polynomially increasing the dimension of the attacked input space. Canonical textual defenses like ONION and BKI leverage auxiliary model forward passes over individual tokens, scaling exponentially with the input sequence length and struggling to maintain computational feasibility. To this end, we propose a decoding time defense - decayed contrastive decoding - that scales linearly with assistant response sequence length and reduces the backdoor to as low as 0.35%.

摘要: 大型语言模型(LLM)已经具备了处理更长的上下文长度和理解文本中的细微差别的能力，将它们的对话能力扩展到了单一话语之外。LLMS的一个流行的面向用户的应用是多轮聊天设置。虽然更长的聊天记忆和更好的理解似乎对用户有利，但我们的论文暴露了一个漏洞，该漏洞利用LLMS的多回合功能和强大的学习能力来伤害最终用户：后门。我们证明了LLMS能够捕获组合后门表示。只有在一起显示触发器时，后门才会激活。我们还通过实验验证了该表示与触发话语的位置不变。随后，在两个5%的数据发声中插入一个额外令牌可以导致超过99%的攻击成功率(ASR)。我们对3个触发器的测试结果表明，该框架具有通用性，以即插即用的方式兼容对手工具箱中的任何触发器。在聊天环境中，由于输入和输出空间很大，保护后门可能是一件具有挑战性的事情。我们的分析表明，分布式后门通过以多项式增加被攻击输入空间的维度来加剧当前的挑战。像Onion和BKI这样的规范文本防御机制利用辅助模型向前传递单个令牌，随着输入序列长度呈指数级扩展，并努力保持计算的可行性。为此，我们提出了一种译码时间防御机制--衰落对比译码，它与辅助响应序列长度成线性关系，并将后门降低到0.35%。



## **21. AutoPenBench: Benchmarking Generative Agents for Penetration Testing**

AutoPenBench：渗透测试生成剂的基准测试 cs.CR

Codes for the benchmark:  https://github.com/lucagioacchini/auto-pen-bench Codes for the paper  experiments: https://github.com/lucagioacchini/genai-pentest-paper

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.03225v2) [paper-pdf](http://arxiv.org/pdf/2410.03225v2)

**Authors**: Luca Gioacchini, Marco Mellia, Idilio Drago, Alexander Delsanto, Giuseppe Siracusano, Roberto Bifulco

**Abstract**: Generative AI agents, software systems powered by Large Language Models (LLMs), are emerging as a promising approach to automate cybersecurity tasks. Among the others, penetration testing is a challenging field due to the task complexity and the diverse strategies to simulate cyber-attacks. Despite growing interest and initial studies in automating penetration testing with generative agents, there remains a significant gap in the form of a comprehensive and standard framework for their evaluation and development. This paper introduces AutoPenBench, an open benchmark for evaluating generative agents in automated penetration testing. We present a comprehensive framework that includes 33 tasks, each representing a vulnerable system that the agent has to attack. Tasks are of increasing difficulty levels, including in-vitro and real-world scenarios. We assess the agent performance with generic and specific milestones that allow us to compare results in a standardised manner and understand the limits of the agent under test. We show the benefits of AutoPenBench by testing two agent architectures: a fully autonomous and a semi-autonomous supporting human interaction. We compare their performance and limitations. For example, the fully autonomous agent performs unsatisfactorily achieving a 21% Success Rate (SR) across the benchmark, solving 27% of the simple tasks and only one real-world task. In contrast, the assisted agent demonstrates substantial improvements, with 64% of SR. AutoPenBench allows us also to observe how different LLMs like GPT-4o or OpenAI o1 impact the ability of the agents to complete the tasks. We believe that our benchmark fills the gap with a standard and flexible framework to compare penetration testing agents on a common ground. We hope to extend AutoPenBench along with the research community by making it available under https://github.com/lucagioacchini/auto-pen-bench.

摘要: 生成式人工智能代理是由大型语言模型(LLM)支持的软件系统，正在成为一种有前途的自动化网络安全任务的方法。其中，渗透测试是一个具有挑战性的领域，因为任务的复杂性和模拟网络攻击的策略多种多样。尽管人们对利用产生剂进行自动化渗透测试越来越感兴趣，并进行了初步研究，但在评估和开发产生剂的全面和标准框架的形式上，仍然存在着重大差距。本文介绍了一种用于评估自动渗透测试中的生成性代理的开放基准--AutoPenBch。我们提出了一个全面的框架，包括33个任务，每个任务代表代理必须攻击的易受攻击的系统。任务的难度越来越高，包括体外和真实世界的场景。我们用通用的和特定的里程碑来评估代理的性能，使我们能够以标准化的方式比较结果，并了解接受测试的代理的限制。我们通过测试两种代理体系结构：完全自主和半自主支持人类交互，展示了AutoPenB边的好处。我们比较了它们的性能和局限性。例如，完全自主的代理在基准测试中的成功率(SR)不令人满意地达到了21%，解决了27%的简单任务，而只解决了一个真实世界的任务。相比之下，辅助剂表现出显著的改善，获得了SR的5%。AutoPenB边还允许我们观察不同的LLM，如GPT-40或OpenAI o1，是如何影响代理完成任务的能力的。我们相信，我们的基准填补了这一空白，提供了一个标准和灵活的框架，可以在共同的基础上比较渗透测试试剂。我们希望通过使其在https://github.com/lucagioacchini/auto-pen-bench.下可用来与研究社区一起扩展AutoPenB边



## **22. Representation noising can prevent harmful fine-tuning on LLMs**

表示噪音可以防止LLM的有害微调 cs.CL

Published in NeurIPs 2024

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2405.14577v3) [paper-pdf](http://arxiv.org/pdf/2405.14577v3)

**Authors**: Domenic Rosati, Jan Wehner, Kai Williams, Łukasz Bartoszcze, David Atanasov, Robie Gonzales, Subhabrata Majumdar, Carsten Maple, Hassan Sajjad, Frank Rudzicz

**Abstract**: Releasing open-source large language models (LLMs) presents a dual-use risk since bad actors can easily fine-tune these models for harmful purposes. Even without the open release of weights, weight stealing and fine-tuning APIs make closed models vulnerable to harmful fine-tuning attacks (HFAs). While safety measures like preventing jailbreaks and improving safety guardrails are important, such measures can easily be reversed through fine-tuning. In this work, we propose Representation Noising (RepNoise), a defence mechanism that is effective even when attackers have access to the weights. RepNoise works by removing information about harmful representations such that it is difficult to recover them during fine-tuning. Importantly, our defence is also able to generalize across different subsets of harm that have not been seen during the defence process as long as they are drawn from the same distribution of the attack set. Our method does not degrade the general capability of LLMs and retains the ability to train the model on harmless tasks. We provide empirical evidence that the effectiveness of our defence lies in its "depth": the degree to which information about harmful representations is removed across all layers of the LLM.

摘要: 发布开源的大型语言模型(LLM)存在双重用途的风险，因为不好的参与者很容易出于有害目的微调这些模型。即使没有公开的权重释放，权重盗窃和微调API也会使封闭的模型容易受到有害的微调攻击(HFA)。虽然防止越狱和改善安全护栏等安全措施很重要，但通过微调很容易逆转这些措施。在这项工作中，我们提出了表示噪声(RepNoise)，这是一种即使攻击者可以访问权重也有效的防御机制。RepNoise的工作原理是删除有关有害表示的信息，以便在微调期间很难恢复它们。重要的是，我们的防御还能够概括在防御过程中未曾见过的伤害的不同子集，只要它们来自相同分布的攻击集。我们的方法不会降低LLMS的整体性能，并保留了对模型进行无害任务训练的能力。我们提供的经验证据表明，我们辩护的有效性在于它的“深度”：在LLM的所有层中，有关有害陈述的信息被移除的程度。



## **23. Palisade -- Prompt Injection Detection Framework**

Palisade --提示注入检测框架 cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21146v1) [paper-pdf](http://arxiv.org/pdf/2410.21146v1)

**Authors**: Sahasra Kokkula, Somanathan R, Nandavardhan R, Aashishkumar, G Divya

**Abstract**: The advent of Large Language Models LLMs marks a milestone in Artificial Intelligence, altering how machines comprehend and generate human language. However, LLMs are vulnerable to malicious prompt injection attacks, where crafted inputs manipulate the models behavior in unintended ways, compromising system integrity and causing incorrect outcomes. Conventional detection methods rely on static, rule-based approaches, which often fail against sophisticated threats like abnormal token sequences and alias substitutions, leading to limited adaptability and higher rates of false positives and false negatives.This paper proposes a novel NLP based approach for prompt injection detection, emphasizing accuracy and optimization through a layered input screening process. In this framework, prompts are filtered through three distinct layers rule-based, ML classifier, and companion LLM before reaching the target model, thereby minimizing the risk of malicious interaction.Tests show the ML classifier achieves the highest accuracy among individual layers, yet the multi-layer framework enhances overall detection accuracy by reducing false negatives. Although this increases false positives, it minimizes the risk of overlooking genuine injected prompts, thus prioritizing security.This multi-layered detection approach highlights LLM vulnerabilities and provides a comprehensive framework for future research, promoting secure interactions between humans and AI systems.

摘要: 大型语言模型LLMS的出现标志着人工智能的一个里程碑，改变了机器理解和生成人类语言的方式。然而，LLM容易受到恶意提示注入攻击，在恶意提示注入攻击中，精心编制的输入以意外的方式操纵模型行为，损害系统完整性并导致不正确的结果。针对传统的检测方法依赖于静态的、基于规则的方法，往往无法抵抗令牌序列异常、别名替换等复杂威胁，导致适应性有限，误报和漏检率较高，提出了一种新的基于自然语言处理的快速注入检测方法，通过分层的输入筛选过程强调准确性和优化。在该框架中，提示在到达目标模型之前通过基于规则、ML分类器和伴随LLM三个不同的层进行过滤，从而将恶意交互的风险降到最低；测试表明，ML分类器在各个层中达到了最高的准确率，而多层框架通过减少漏报来提高整体检测的准确率。虽然这会增加误报，但它将忽略真实注入提示的风险降至最低，从而将安全放在首位。这种多层检测方法突出了LLM漏洞，并为未来的研究提供了一个全面的框架，促进了人类与AI系统之间的安全交互。



## **24. Stealthy Jailbreak Attacks on Large Language Models via Benign Data Mirroring**

通过良性数据镜像对大型语言模型进行秘密越狱攻击 cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21083v1) [paper-pdf](http://arxiv.org/pdf/2410.21083v1)

**Authors**: Honglin Mu, Han He, Yuxin Zhou, Yunlong Feng, Yang Xu, Libo Qin, Xiaoming Shi, Zeming Liu, Xudong Han, Qi Shi, Qingfu Zhu, Wanxiang Che

**Abstract**: Large language model (LLM) safety is a critical issue, with numerous studies employing red team testing to enhance model security. Among these, jailbreak methods explore potential vulnerabilities by crafting malicious prompts that induce model outputs contrary to safety alignments. Existing black-box jailbreak methods often rely on model feedback, repeatedly submitting queries with detectable malicious instructions during the attack search process. Although these approaches are effective, the attacks may be intercepted by content moderators during the search process. We propose an improved transfer attack method that guides malicious prompt construction by locally training a mirror model of the target black-box model through benign data distillation. This method offers enhanced stealth, as it does not involve submitting identifiable malicious instructions to the target model during the search phase. Our approach achieved a maximum attack success rate of 92%, or a balanced value of 80% with an average of 1.5 detectable jailbreak queries per sample against GPT-3.5 Turbo on a subset of AdvBench. These results underscore the need for more robust defense mechanisms.

摘要: 大型语言模型(LLM)的安全性是一个关键问题，许多研究使用RED团队测试来增强模型的安全性。其中，越狱方法通过精心编制恶意提示来探测潜在的漏洞，这些提示会诱导与安全对齐相反的模型输出。现有的黑盒越狱方法通常依赖于模型反馈，在攻击搜索过程中反复提交带有可检测到的恶意指令的查询。虽然这些方法是有效的，但这些攻击可能会在搜索过程中被内容版主拦截。提出了一种改进的传输攻击方法，该方法通过良性数据提炼对目标黑盒模型的镜像模型进行局部训练，指导恶意提示的构建。这种方法提供了增强的隐蔽性，因为它不涉及在搜索阶段向目标模型提交可识别的恶意指令。我们的方法获得了92%的最大攻击成功率，或者说80%的平衡值，每个样本平均有1.5个可检测到的越狱查询，而GPT-3.5Turbo在AdvBitch子集上的攻击成功率为92%。这些结果突显了需要更强大的防御机制。



## **25. Attacking Misinformation Detection Using Adversarial Examples Generated by Language Models**

使用语言模型生成的对抗性示例进行攻击错误信息检测 cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.20940v1) [paper-pdf](http://arxiv.org/pdf/2410.20940v1)

**Authors**: Piotr Przybyła

**Abstract**: We investigate the challenge of generating adversarial examples to test the robustness of text classification algorithms detecting low-credibility content, including propaganda, false claims, rumours and hyperpartisan news. We focus on simulation of content moderation by setting realistic limits on the number of queries an attacker is allowed to attempt. Within our solution (TREPAT), initial rephrasings are generated by large language models with prompts inspired by meaning-preserving NLP tasks, e.g. text simplification and style transfer. Subsequently, these modifications are decomposed into small changes, applied through beam search procedure until the victim classifier changes its decision. The evaluation confirms the superiority of our approach in the constrained scenario, especially in case of long input text (news articles), where exhaustive search is not feasible.

摘要: 我们调查了生成敌对示例的挑战，以测试检测低可信度内容（包括宣传、虚假声明、谣言和超党派新闻）的文本分类算法的稳健性。我们通过对允许攻击者尝试的查询数量设置现实的限制来重点模拟内容审核。在我们的解决方案（TREPAT）中，初始改写由大型语言模型生成，其提示受到保留意义的NLP任务（例如文本简化和风格转移）的启发。随后，这些修改被分解成小的变化，通过束搜索过程应用，直到受害者分类器改变其决定。评估证实了我们的方法在受约束的情况下的优越性，特别是在长输入文本（新闻文章）的情况下，其中详尽搜索是不可行的。



## **26. Hacking Back the AI-Hacker: Prompt Injection as a Defense Against LLM-driven Cyberattacks**

黑客攻击人工智能黑客：即时注入作为抵御LLM驱动的网络攻击的防御 cs.CR

v0.1

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.20911v1) [paper-pdf](http://arxiv.org/pdf/2410.20911v1)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: Large language models (LLMs) are increasingly being harnessed to automate cyberattacks, making sophisticated exploits more accessible and scalable. In response, we propose a new defense strategy tailored to counter LLM-driven cyberattacks. We introduce Mantis, a defensive framework that exploits LLMs' susceptibility to adversarial inputs to undermine malicious operations. Upon detecting an automated cyberattack, Mantis plants carefully crafted inputs into system responses, leading the attacker's LLM to disrupt their own operations (passive defense) or even compromise the attacker's machine (active defense). By deploying purposefully vulnerable decoy services to attract the attacker and using dynamic prompt injections for the attacker's LLM, Mantis can autonomously hack back the attacker. In our experiments, Mantis consistently achieved over 95% effectiveness against automated LLM-driven attacks. To foster further research and collaboration, Mantis is available as an open-source tool: https://github.com/pasquini-dario/project_mantis

摘要: 大型语言模型(LLM)越来越多地被用来自动化网络攻击，使复杂的利用更容易获得和可扩展。作为回应，我们提出了一种新的防御战略，以对抗LLM驱动的网络攻击。我们引入了Mantis，这是一个防御框架，利用LLMS对对手输入的敏感性来破坏恶意操作。在检测到自动网络攻击后，螳螂工厂会精心设计输入到系统响应中，导致攻击者的LLM扰乱自己的操作(被动防御)，甚至危害攻击者的机器(主动防御)。通过部署故意易受攻击的诱骗服务来吸引攻击者，并对攻击者的LLM使用动态提示注入，螳螂可以自主地攻击攻击者。在我们的实验中，螳螂对自动LLM驱动的攻击始终取得了95%以上的效率。为了促进进一步的研究和合作，Mantis以开源工具的形式提供：https://github.com/pasquini-dario/project_mantis



## **27. Uncovering Safety Risks of Large Language Models through Concept Activation Vector**

通过概念激活载体揭示大型语言模型的安全风险 cs.CL

10 pages, accepted as a poster at NeurIPS 2024

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2404.12038v4) [paper-pdf](http://arxiv.org/pdf/2404.12038v4)

**Authors**: Zhihao Xu, Ruixuan Huang, Changyu Chen, Xiting Wang

**Abstract**: Despite careful safety alignment, current large language models (LLMs) remain vulnerable to various attacks. To further unveil the safety risks of LLMs, we introduce a Safety Concept Activation Vector (SCAV) framework, which effectively guides the attacks by accurately interpreting LLMs' safety mechanisms. We then develop an SCAV-guided attack method that can generate both attack prompts and embedding-level attacks with automatically selected perturbation hyperparameters. Both automatic and human evaluations demonstrate that our attack method significantly improves the attack success rate and response quality while requiring less training data. Additionally, we find that our generated attack prompts may be transferable to GPT-4, and the embedding-level attacks may also be transferred to other white-box LLMs whose parameters are known. Our experiments further uncover the safety risks present in current LLMs. For example, in our evaluation of seven open-source LLMs, we observe an average attack success rate of 99.14%, based on the classic keyword-matching criterion. Finally, we provide insights into the safety mechanism of LLMs. The code is available at https://github.com/SproutNan/AI-Safety_SCAV.

摘要: 尽管进行了仔细的安全调整，但当前的大型语言模型(LLM)仍然容易受到各种攻击。为了进一步揭示LLMS的安全隐患，我们引入了安全概念激活向量(SCAV)框架，通过准确解释LLMS的安全机制来有效地指导攻击。然后，我们开发了一种SCAV引导的攻击方法，该方法可以生成攻击提示和带有自动选择的扰动超参数的嵌入级攻击。自动和人工评估都表明，我们的攻击方法在需要更少的训练数据的情况下，显著地提高了攻击成功率和响应质量。此外，我们发现我们生成的攻击提示可以转移到GPT-4上，嵌入级攻击也可以转移到参数已知的其他白盒LLM上。我们的实验进一步揭示了当前LLM中存在的安全风险。例如，在我们对7个开源LLM的评估中，基于经典的关键字匹配标准，我们观察到平均攻击成功率为99.14%。最后，我们对LLMS的安全机制提供了见解。代码可在https://github.com/SproutNan/AI-Safety_SCAV.上获得



## **28. Fine-tuned Large Language Models (LLMs): Improved Prompt Injection Attacks Detection**

微调的大型语言模型（LLM）：改进的提示注入攻击检测 cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21337v1) [paper-pdf](http://arxiv.org/pdf/2410.21337v1)

**Authors**: Md Abdur Rahman, Fan Wu, Alfredo Cuzzocrea, Sheikh Iqbal Ahamed

**Abstract**: Large language models (LLMs) are becoming a popular tool as they have significantly advanced in their capability to tackle a wide range of language-based tasks. However, LLMs applications are highly vulnerable to prompt injection attacks, which poses a critical problem. These attacks target LLMs applications through using carefully designed input prompts to divert the model from adhering to original instruction, thereby it could execute unintended actions. These manipulations pose serious security threats which potentially results in data leaks, biased outputs, or harmful responses. This project explores the security vulnerabilities in relation to prompt injection attacks. To detect whether a prompt is vulnerable or not, we follows two approaches: 1) a pre-trained LLM, and 2) a fine-tuned LLM. Then, we conduct a thorough analysis and comparison of the classification performance. Firstly, we use pre-trained XLM-RoBERTa model to detect prompt injections using test dataset without any fine-tuning and evaluate it by zero-shot classification. Then, this proposed work will apply supervised fine-tuning to this pre-trained LLM using a task-specific labeled dataset from deepset in huggingface, and this fine-tuned model achieves impressive results with 99.13\% accuracy, 100\% precision, 98.33\% recall and 99.15\% F1-score thorough rigorous experimentation and evaluation. We observe that our approach is highly efficient in detecting prompt injection attacks.

摘要: 大型语言模型(LLM)正在成为一种流行的工具，因为它们在处理各种基于语言的任务的能力方面有了显著的进步。然而，LLMS应用程序很容易受到即时注入攻击，这是一个严重的问题。这些攻击通过使用精心设计的输入提示来转移模型对原始指令的依赖，从而针对LLMS应用程序，从而可以执行意外的操作。这些操作构成了严重的安全威胁，可能会导致数据泄露、有偏见的输出或有害的响应。该项目探索与提示注入攻击相关的安全漏洞。为了检测提示符是否易受攻击，我们采用了两种方法：1)预先训练的LLM和2)微调的LLM。然后，我们对分类性能进行了深入的分析和比较。首先，我们使用预先训练好的XLM-Roberta模型，在没有任何微调的测试数据集上检测快速注射，并用零镜头分类对其进行评估。然后，该工作将使用来自拥抱脸深度集的特定任务的标签数据集对该预训练的LLM进行有监督的微调，该微调模型取得了令人印象深刻的结果，其准确率为99.13，准确率为100，召回率为98.33，F1-Score为99.15。我们观察到我们的方法在检测即时注入攻击方面是非常有效的。



## **29. LLM Robustness Against Misinformation in Biomedical Question Answering**

LLM生物医学问题回答中针对错误信息的稳健性 cs.CL

**SubmitDate**: 2024-10-27    [abs](http://arxiv.org/abs/2410.21330v1) [paper-pdf](http://arxiv.org/pdf/2410.21330v1)

**Authors**: Alexander Bondarenko, Adrian Viehweger

**Abstract**: The retrieval-augmented generation (RAG) approach is used to reduce the confabulation of large language models (LLMs) for question answering by retrieving and providing additional context coming from external knowledge sources (e.g., by adding the context to the prompt). However, injecting incorrect information can mislead the LLM to generate an incorrect answer.   In this paper, we evaluate the effectiveness and robustness of four LLMs against misinformation - Gemma 2, GPT-4o-mini, Llama~3.1, and Mixtral - in answering biomedical questions. We assess the answer accuracy on yes-no and free-form questions in three scenarios: vanilla LLM answers (no context is provided), "perfect" augmented generation (correct context is provided), and prompt-injection attacks (incorrect context is provided). Our results show that Llama 3.1 (70B parameters) achieves the highest accuracy in both vanilla (0.651) and "perfect" RAG (0.802) scenarios. However, the accuracy gap between the models almost disappears with "perfect" RAG, suggesting its potential to mitigate the LLM's size-related effectiveness differences.   We further evaluate the ability of the LLMs to generate malicious context on one hand and the LLM's robustness against prompt-injection attacks on the other hand, using metrics such as attack success rate (ASR), accuracy under attack, and accuracy drop. As adversaries, we use the same four LLMs (Gemma 2, GPT-4o-mini, Llama 3.1, and Mixtral) to generate incorrect context that is injected in the target model's prompt. Interestingly, Llama is shown to be the most effective adversary, causing accuracy drops of up to 0.48 for vanilla answers and 0.63 for "perfect" RAG across target models. Our analysis reveals that robustness rankings vary depending on the evaluation measure, highlighting the complexity of assessing LLM resilience to adversarial attacks.

摘要: 检索-增强生成(RAG)方法用于通过检索和提供来自外部知识源的附加上下文(例如，通过将上下文添加到提示)来减少大语言模型(LLM)对问题回答的虚构。然而，注入错误的信息可能会误导LLM生成错误的答案。在这篇文章中，我们评估了四个针对错误信息的最小二乘模型-Gema2、GPT-40-mini、Llama~3.1和Mixtral-在回答生物医学问题时的有效性和稳健性。我们在三个场景中评估了是-否和自由形式问题的答案准确率：普通LLM答案(没有提供上下文)、“完美”增强生成(提供了正确的上下文)和提示注入攻击(提供了错误的上下文)。我们的结果表明，Llama3.1(70B参数)在Vanilla(0.651)和“Perfect”RAG(0.802)场景中都达到了最高的精度。然而，随着RAG的“完美”，两个模型之间的精度差距几乎消失了，这表明它有可能缓解LLM与大小相关的有效性差异。我们使用攻击成功率(ASR)、攻击下的准确率和准确率下降等指标，进一步评估了LLM一方面产生恶意上下文的能力，另一方面LLM对即时注入攻击的健壮性。作为对手，我们使用相同的四个LLM(Gema2、GPT-4o-mini、Llama3.1和Mixtral)来生成错误的上下文，该上下文被注入到目标模型的提示符中。有趣的是，骆驼被证明是最有效的对手，在目标模型上，导致普通答案的准确率下降高达0.48，而“完美”RAG的准确率下降0.63。我们的分析表明，健壮性排名根据评估措施的不同而不同，这突显了评估LLM对对手攻击的弹性的复杂性。



## **30. PANORAMIA: Privacy Auditing of Machine Learning Models without Retraining**

PANORAMIA：无需重新训练的机器学习模型隐私审计 cs.CR

36 pages

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2402.09477v2) [paper-pdf](http://arxiv.org/pdf/2402.09477v2)

**Authors**: Mishaal Kazmi, Hadrien Lautraite, Alireza Akbari, Qiaoyue Tang, Mauricio Soroco, Tao Wang, Sébastien Gambs, Mathias Lécuyer

**Abstract**: We present PANORAMIA, a privacy leakage measurement framework for machine learning models that relies on membership inference attacks using generated data as non-members. By relying on generated non-member data, PANORAMIA eliminates the common dependency of privacy measurement tools on in-distribution non-member data. As a result, PANORAMIA does not modify the model, training data, or training process, and only requires access to a subset of the training data. We evaluate PANORAMIA on ML models for image and tabular data classification, as well as on large-scale language models.

摘要: 我们介绍了PANORAMIA，这是一个用于机器学习模型的隐私泄露测量框架，该模型依赖于使用生成的数据作为非成员的成员推断攻击。通过依赖生成的非会员数据，PANORAMIA消除了隐私测量工具对分发内非会员数据的常见依赖性。因此，PANORAMIA不会修改模型、训练数据或训练过程，仅需要访问训练数据的子集。我们在图像和表格数据分类的ML模型以及大规模语言模型上评估了PANORAMIA。



## **31. Course-Correction: Safety Alignment Using Synthetic Preferences**

课程纠正：使用综合偏好进行安全调整 cs.CL

Paper accepted to EMNLP 2024. Camera-ready version. We have released  our dataset and scripts at https://github.com/pillowsofwind/Course-Correction

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2407.16637v2) [paper-pdf](http://arxiv.org/pdf/2407.16637v2)

**Authors**: Rongwu Xu, Yishuo Cai, Zhenhong Zhou, Renjie Gu, Haiqin Weng, Yan Liu, Tianwei Zhang, Wei Xu, Han Qiu

**Abstract**: The risk of harmful content generated by large language models (LLMs) becomes a critical concern. This paper presents a systematic study on assessing and improving LLMs' capability to perform the task of \textbf{course-correction}, \ie, the model can steer away from generating harmful content autonomously. To start with, we introduce the \textsc{C$^2$-Eval} benchmark for quantitative assessment and analyze 10 popular LLMs, revealing varying proficiency of current safety-tuned LLMs in course-correction. To improve, we propose fine-tuning LLMs with preference learning, emphasizing the preference for timely course-correction. Using an automated pipeline, we create \textsc{C$^2$-Syn}, a synthetic dataset with 750K pairwise preferences, to teach models the concept of timely course-correction through data-driven preference learning. Experiments on 2 LLMs, \textsc{Llama2-Chat 7B} and \textsc{Qwen2 7B}, show that our method effectively enhances course-correction skills without affecting general performance. Additionally, it effectively improves LLMs' safety, particularly in resisting jailbreak attacks.

摘要: 大型语言模型(LLM)生成有害内容的风险成为一个关键问题。本文对如何评估和提高LLMS执行航向修正任务的能力进行了系统的研究，即该模型可以避免自动产生有害内容。首先，我们引入了用于定量评估的基准，并对10个流行的LLM进行了分析，揭示了当前安全调整的LLMS在航向校正方面的不同熟练程度。为了改进，我们提出了带有偏好学习的微调LLMS，强调了对及时航向校正的偏好。使用自动化管道，我们创建了一个包含75万个成对偏好的合成数据集\extsc{C$^2$-Syn}，以通过数据驱动的偏好学习向模型传授及时进行课程修正的概念。在Textsc{Llama2-Chat 7B}和Textsc{Qwen2 7B}上的实验表明，该方法在不影响总体性能的情况下有效地增强了航向修正技巧。此外，它还有效地提高了LLMS的安全性，特别是在抵抗越狱攻击方面。



## **32. Mask-based Membership Inference Attacks for Retrieval-Augmented Generation**

用于检索增强生成的基于面具的成员推断攻击 cs.CR

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2410.20142v1) [paper-pdf](http://arxiv.org/pdf/2410.20142v1)

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long

**Abstract**: Retrieval-Augmented Generation (RAG) has been an effective approach to mitigate hallucinations in large language models (LLMs) by incorporating up-to-date and domain-specific knowledge. Recently, there has been a trend of storing up-to-date or copyrighted data in RAG knowledge databases instead of using it for LLM training. This practice has raised concerns about Membership Inference Attacks (MIAs), which aim to detect if a specific target document is stored in the RAG system's knowledge database so as to protect the rights of data producers. While research has focused on enhancing the trustworthiness of RAG systems, existing MIAs for RAG systems remain largely insufficient. Previous work either relies solely on the RAG system's judgment or is easily influenced by other documents or the LLM's internal knowledge, which is unreliable and lacks explainability. To address these limitations, we propose a Mask-Based Membership Inference Attacks (MBA) framework. Our framework first employs a masking algorithm that effectively masks a certain number of words in the target document. The masked text is then used to prompt the RAG system, and the RAG system is required to predict the mask values. If the target document appears in the knowledge database, the masked text will retrieve the complete target document as context, allowing for accurate mask prediction. Finally, we adopt a simple yet effective threshold-based method to infer the membership of target document by analyzing the accuracy of mask prediction. Our mask-based approach is more document-specific, making the RAG system's generation less susceptible to distractions from other documents or the LLM's internal knowledge. Extensive experiments demonstrate the effectiveness of our approach compared to existing baseline models.

摘要: 检索-增强生成(RAG)是一种通过结合最新的和特定领域的知识来缓解大型语言模型(LLMS)中的幻觉的有效方法。最近，有一种趋势是将最新数据或受版权保护的数据存储在RAG知识数据库中，而不是将其用于LLM培训。这种做法引起了人们对成员资格推断攻击(MIA)的担忧，这种攻击旨在检测特定目标文件是否存储在RAG系统的知识数据库中，以保护数据制作者的权利。虽然研究的重点是提高RAG系统的可信度，但现有的RAG系统的MIA仍然很大程度上是不够的。以往的工作要么完全依赖RAG系统的判断，要么容易受到其他文件或LLM内部知识的影响，这是不可靠的，缺乏解释性。针对这些局限性，我们提出了一种基于掩码的成员关系推理攻击(MBA)框架。我们的框架首先使用掩码算法，该算法有效地掩码目标文档中的特定数量的单词。然后使用掩码文本来提示RAG系统，并且需要RAG系统来预测掩码值。如果目标文档出现在知识数据库中，则掩码文本将检索完整的目标文档作为上下文，从而实现准确的掩码预测。最后，通过分析模板预测的精度，采用一种简单有效的基于阈值的方法来推断目标文档的隶属度。我们的基于掩码的方法更特定于文档，使RAG系统的生成不太容易受到来自其他文档或LLM内部知识的干扰。大量的实验表明，与现有的基线模型相比，我们的方法是有效的。



## **33. Detection Made Easy: Potentials of Large Language Models for Solidity Vulnerabilities**

检测变得简单：大型语言模型针对Solidity漏洞的潜力 cs.CR

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2409.10574v2) [paper-pdf](http://arxiv.org/pdf/2409.10574v2)

**Authors**: Md Tauseef Alam, Raju Halder, Abyayananda Maiti

**Abstract**: The large-scale deployment of Solidity smart contracts on the Ethereum mainnet has increasingly attracted financially-motivated attackers in recent years. A few now-infamous attacks in Ethereum's history includes DAO attack in 2016 (50 million dollars lost), Parity Wallet hack in 2017 (146 million dollars locked), Beautychain's token BEC in 2018 (900 million dollars market value fell to 0), and NFT gaming blockchain breach in 2022 ($600 million in Ether stolen). This paper presents a comprehensive investigation of the use of large language models (LLMs) and their capabilities in detecting OWASP Top Ten vulnerabilities in Solidity. We introduce a novel, class-balanced, structured, and labeled dataset named VulSmart, which we use to benchmark and compare the performance of open-source LLMs such as CodeLlama, Llama2, CodeT5 and Falcon, alongside closed-source models like GPT-3.5 Turbo and GPT-4o Mini. Our proposed SmartVD framework is rigorously tested against these models through extensive automated and manual evaluations, utilizing BLEU and ROUGE metrics to assess the effectiveness of vulnerability detection in smart contracts. We also explore three distinct prompting strategies-zero-shot, few-shot, and chain-of-thought-to evaluate the multi-class classification and generative capabilities of the SmartVD framework. Our findings reveal that SmartVD outperforms its open-source counterparts and even exceeds the performance of closed-source base models like GPT-3.5 and GPT-4 Mini. After fine-tuning, the closed-source models, GPT-3.5 Turbo and GPT-4o Mini, achieved remarkable performance with 99% accuracy in detecting vulnerabilities, 94% in identifying their types, and 98% in determining severity. Notably, SmartVD performs best with the `chain-of-thought' prompting technique, whereas the fine-tuned closed-source models excel with the `zero-shot' prompting approach.

摘要: 近年来，在以太主机上大规模部署稳健智能合约，越来越多地吸引了出于经济动机的攻击者。Etherum历史上现在臭名昭著的攻击包括2016年的DAO攻击(损失了5000万美元)，2017年的平价钱包黑客攻击(锁定了1.46亿美元)，2018年BeautyChain的Token BEC(9亿美元的市值跌至0)，以及2022年的NFT游戏区块链漏洞(6亿美元的以太被盗)。本文对大型语言模型(LLM)的使用及其检测OWASP的十大漏洞的能力进行了全面的调查。我们介绍了一个新的，类平衡的，结构化的，带标签的数据集VulSmart，我们使用它来基准测试和比较开源LLM的性能，如CodeLlama，Llama2，CodeT5和Falcon，以及闭源模型如GPT-3.5 Turbo和GPT-40 Mini。我们建议的SmartVD框架通过广泛的自动和手动评估，针对这些模型进行了严格的测试，使用BLEU和Rouge指标来评估智能合同中漏洞检测的有效性。我们还探索了三种不同的提示策略-零射、少射和思维链-来评估SmartVD框架的多类分类和生成能力。我们的发现表明，SmartVD的表现优于开源同行，甚至超过了GPT-3.5和GPT-4 Mini等封闭源代码基础机型的性能。经过微调后，闭源模型GPT-3.5 Turbo和GPT-40 Mini取得了显著的性能，检测漏洞的准确率为99%，识别漏洞类型的准确率为94%，确定严重性的准确率为98%。值得注意的是，SmartVD最好地使用了“思维链”提示技术，而经过微调的闭源模型则使用了“零镜头”提示方法。



## **34. Attacks against Abstractive Text Summarization Models through Lead Bias and Influence Functions**

通过铅偏差和影响函数对抽象文本摘要模型的攻击 cs.CL

10 pages, 3 figures, Accepted at EMNLP Findings 2024

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2410.20019v1) [paper-pdf](http://arxiv.org/pdf/2410.20019v1)

**Authors**: Poojitha Thota, Shirin Nilizadeh

**Abstract**: Large Language Models have introduced novel opportunities for text comprehension and generation. Yet, they are vulnerable to adversarial perturbations and data poisoning attacks, particularly in tasks like text classification and translation. However, the adversarial robustness of abstractive text summarization models remains less explored. In this work, we unveil a novel approach by exploiting the inherent lead bias in summarization models, to perform adversarial perturbations. Furthermore, we introduce an innovative application of influence functions, to execute data poisoning, which compromises the model's integrity. This approach not only shows a skew in the models behavior to produce desired outcomes but also shows a new behavioral change, where models under attack tend to generate extractive summaries rather than abstractive summaries.

摘要: 大型语言模型为文本理解和生成带来了新的机会。然而，它们很容易受到敌对干扰和数据中毒攻击，特别是在文本分类和翻译等任务中。然而，抽象文本摘要模型的对抗稳健性仍然较少被探索。在这项工作中，我们推出了一种新颖的方法，通过利用摘要模型中固有的领先偏差来执行对抗性扰动。此外，我们引入了影响函数的创新应用程序，以执行数据中毒，这会损害模型的完整性。这种方法不仅显示了模型产生所需结果的行为的倾斜，而且还显示了新的行为变化，即受攻击的模型倾向于生成提取摘要而不是抽象摘要。



## **35. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

迭代自调优LLM以增强越狱能力 cs.CL

18 pages

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.18469v2) [paper-pdf](http://arxiv.org/pdf/2410.18469v2)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99% ASR on GPT-3.5 and 49% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety.

摘要: 最近的研究表明，大型语言模型(LLM)容易受到自动越狱攻击，在自动越狱攻击中，由附加到有害查询的算法编制的敌意后缀绕过安全对齐并触发意外响应。目前生成这些后缀的方法计算量大，攻击成功率(ASR)低，尤其是针对Llama2和Llama3等排列良好的模型。为了克服这些限制，我们引入了ADV-LLM，这是一个迭代的自我调整过程，可以制作具有增强越狱能力的对抗性LLM。我们的框架大大降低了生成敌意后缀的计算代价，同时在各种开源LLM上实现了近100个ASR。此外，它表现出很强的攻击可转换性，尽管只在Llama3上进行了优化，但在GPT-3.5上实现了99%的ASR，在GPT-4上实现了49%的ASR。除了提高越狱能力，ADV-LLM还通过其生成用于研究LLM安全性的大型数据集的能力，为未来的安全配准研究提供了有价值的见解。



## **36. RobustKV: Defending Large Language Models against Jailbreak Attacks via KV Eviction**

RobustKN：通过GV驱逐保护大型语言模型免受越狱攻击 cs.CR

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.19937v1) [paper-pdf](http://arxiv.org/pdf/2410.19937v1)

**Authors**: Tanqiu Jiang, Zian Wang, Jiacheng Liang, Changjiang Li, Yuhui Wang, Ting Wang

**Abstract**: Jailbreak attacks circumvent LLMs' built-in safeguards by concealing harmful queries within jailbreak prompts. While existing defenses primarily focus on mitigating the effects of jailbreak prompts, they often prove inadequate as jailbreak prompts can take arbitrary, adaptive forms. This paper presents RobustKV, a novel defense that adopts a fundamentally different approach by selectively removing critical tokens of harmful queries from key-value (KV) caches. Intuitively, for a jailbreak prompt to be effective, its tokens must achieve sufficient `importance' (as measured by attention scores), which inevitably lowers the importance of tokens in the concealed harmful query. Thus, by strategically evicting the KVs of the lowest-ranked tokens, RobustKV diminishes the presence of the harmful query in the KV cache, thus preventing the LLM from generating malicious responses. Extensive evaluation using benchmark datasets and models demonstrates that RobustKV effectively counters state-of-the-art jailbreak attacks while maintaining the LLM's general performance on benign queries. Moreover, RobustKV creates an intriguing evasiveness dilemma for adversaries, forcing them to balance between evading RobustKV and bypassing the LLM's built-in safeguards. This trade-off contributes to RobustKV's robustness against adaptive attacks. (warning: this paper contains potentially harmful content generated by LLMs.)

摘要: 越狱攻击通过在越狱提示中隐藏有害的查询来绕过LLMS的内置保护措施。虽然现有的防御措施主要集中在减轻越狱提示的影响，但它们往往被证明是不够的，因为越狱提示可以采取任意的、适应性的形式。本文提出了一种新的防御方法RobustKV，它采用了一种从根本上不同的方法，从键值(KV)缓存中选择性地删除有害查询的关键标记。直观地说，要使越狱提示有效，其标记必须达到足够的“重要性”(通过注意力得分来衡量)，这不可避免地降低了标记在隐藏的有害查询中的重要性。因此，通过战略性地逐出最低等级令牌的KV，RobustKV减少了KV缓存中有害查询的存在，从而防止LLM生成恶意响应。使用基准数据集和模型进行的广泛评估表明，RobustKV在保持LLM在良性查询上的总体性能的同时，有效地对抗了最先进的越狱攻击。此外，RobustKV为对手创造了一个耐人寻味的逃避困境，迫使他们在躲避RobustKV和绕过LLM的内置保障之间取得平衡。这种权衡有助于RobustKV对自适应攻击的健壮性。(警告：本文包含由LLMS生成的潜在有害内容。)



## **37. Expose Before You Defend: Unifying and Enhancing Backdoor Defenses via Exposed Models**

防御前暴露：通过暴露的模型统一和增强后门防御 cs.AI

19 pages

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.19427v1) [paper-pdf](http://arxiv.org/pdf/2410.19427v1)

**Authors**: Yige Li, Hanxun Huang, Jiaming Zhang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Backdoor attacks covertly implant triggers into deep neural networks (DNNs) by poisoning a small portion of the training data with pre-designed backdoor triggers. This vulnerability is exacerbated in the era of large models, where extensive (pre-)training on web-crawled datasets is susceptible to compromise. In this paper, we introduce a novel two-step defense framework named Expose Before You Defend (EBYD). EBYD unifies existing backdoor defense methods into a comprehensive defense system with enhanced performance. Specifically, EBYD first exposes the backdoor functionality in the backdoored model through a model preprocessing step called backdoor exposure, and then applies detection and removal methods to the exposed model to identify and eliminate the backdoor features. In the first step of backdoor exposure, we propose a novel technique called Clean Unlearning (CUL), which proactively unlearns clean features from the backdoored model to reveal the hidden backdoor features. We also explore various model editing/modification techniques for backdoor exposure, including fine-tuning, model sparsification, and weight perturbation. Using EBYD, we conduct extensive experiments on 10 image attacks and 6 text attacks across 2 vision datasets (CIFAR-10 and an ImageNet subset) and 4 language datasets (SST-2, IMDB, Twitter, and AG's News). The results demonstrate the importance of backdoor exposure for backdoor defense, showing that the exposed models can significantly benefit a range of downstream defense tasks, including backdoor label detection, backdoor trigger recovery, backdoor model detection, and backdoor removal. We hope our work could inspire more research in developing advanced defense frameworks with exposed models. Our code is available at: https://github.com/bboylyg/Expose-Before-You-Defend.

摘要: 后门攻击通过用预先设计的后门触发器毒化一小部分训练数据，秘密地将触发器植入深度神经网络(DNN)。这种脆弱性在大型模型时代加剧，在这个时代，对网络爬行的数据集进行广泛的(预先)培训很容易受到损害。在本文中，我们介绍了一种新的两步防御框架--先暴露后防御(EBYD)。EBYD将现有的后门防御方法统一到一个性能增强的全面防御系统中。具体地说，EBYD首先通过称为后门暴露的模型预处理步骤暴露后门模型中的后门功能，然后对暴露的模型应用检测和删除方法以识别和消除后门功能。在后门暴露的第一步，我们提出了一种新的技术，称为清洁遗忘(Clean UnLearning，CUL)，它主动地从后门模型中忘记干净的特征，以揭示隐藏的后门特征。我们还探索了用于后门曝光的各种模型编辑/修改技术，包括微调、模型稀疏和权重扰动。使用EBYD，我们在2个视觉数据集(CIFAR-10和ImageNet子集)和4个语言数据集(SST-2、IMDB、Twitter和AG的新闻)上进行了10次图像攻击和6次文本攻击的广泛实验。结果表明，后门暴露对于后门防御的重要性，表明暴露的模型可以显著受益于一系列下游防御任务，包括后门标签检测、后门触发恢复、后门模型检测和后门删除。我们希望我们的工作能够启发更多的研究，开发具有暴露模型的先进防御框架。我们的代码请访问：https://github.com/bboylyg/Expose-Before-You-Defend.



## **38. Humanizing the Machine: Proxy Attacks to Mislead LLM Detectors**

人性化机器：代理攻击误导LLM检测器 cs.LG

26 pages

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.19230v1) [paper-pdf](http://arxiv.org/pdf/2410.19230v1)

**Authors**: Tianchun Wang, Yuanzhou Chen, Zichuan Liu, Zhanwen Chen, Haifeng Chen, Xiang Zhang, Wei Cheng

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of text generation, producing outputs that closely mimic human-like writing. Although academic and industrial institutions have developed detectors to prevent the malicious usage of LLM-generated texts, other research has doubt about the robustness of these systems. To stress test these detectors, we introduce a proxy-attack strategy that effortlessly compromises LLMs, causing them to produce outputs that align with human-written text and mislead detection systems. Our method attacks the source model by leveraging a reinforcement learning (RL) fine-tuned humanized small language model (SLM) in the decoding phase. Through an in-depth analysis, we demonstrate that our attack strategy is capable of generating responses that are indistinguishable to detectors, preventing them from differentiating between machine-generated and human-written text. We conduct systematic evaluations on extensive datasets using proxy-attacked open-source models, including Llama2-13B, Llama3-70B, and Mixtral-8*7B in both white- and black-box settings. Our findings show that the proxy-attack strategy effectively deceives the leading detectors, resulting in an average AUROC drop of 70.4% across multiple datasets, with a maximum drop of 90.3% on a single dataset. Furthermore, in cross-discipline scenarios, our strategy also bypasses these detectors, leading to a significant relative decrease of up to 90.9%, while in cross-language scenario, the drop reaches 91.3%. Despite our proxy-attack strategy successfully bypassing the detectors with such significant relative drops, we find that the generation quality of the attacked models remains preserved, even within a modest utility budget, when compared to the text produced by the original, unattacked source model.

摘要: 大型语言模型(LLM)的出现使文本生成领域发生了革命性的变化，产生了非常接近人类书写的输出。尽管学术和工业机构已经开发了检测器来防止恶意使用LLM生成的文本，但其他研究对这些系统的健壮性表示怀疑。为了对这些检测器进行压力测试，我们引入了一种代理攻击策略，该策略可以毫不费力地攻击LLM，使它们产生与人类书写的文本和误导检测系统一致的输出。我们的方法在解码阶段利用强化学习(RL)微调的人性化小语言模型(SLM)来攻击源模型。通过深入的分析，我们证明了我们的攻击策略能够产生检测器无法区分的响应，防止他们区分机器生成的文本和人类书写的文本。我们使用代理攻击的开源模型对大量的数据集进行了系统的评估，包括白盒和黑盒环境下的Llama2-13B、Llama3-70B和Mixtral-8*7B。结果表明，代理攻击策略有效地欺骗了领先的检测器，导致多个数据集的AUROC平均下降了70.4%，其中单个数据集的AUROC平均下降了90.3%。此外，在跨学科场景中，我们的策略也绕过了这些检测器，导致了高达90.9%的显著相对降幅，而在跨语言场景中，降幅达到91.3%。尽管我们的代理攻击策略成功地绕过了检测器，具有如此显著的相对下降，但我们发现，与原始的未受攻击源模型生成的文本相比，即使在适度的实用预算内，受攻击模型的生成质量仍然保持不变。



## **39. Integrating Large Language Models with Internet of Things Applications**

将大型语言模型与物联网应用程序集成 cs.AI

**SubmitDate**: 2024-10-25    [abs](http://arxiv.org/abs/2410.19223v1) [paper-pdf](http://arxiv.org/pdf/2410.19223v1)

**Authors**: Mingyu Zong, Arvin Hekmati, Michael Guastalla, Yiyi Li, Bhaskar Krishnamachari

**Abstract**: This paper identifies and analyzes applications in which Large Language Models (LLMs) can make Internet of Things (IoT) networks more intelligent and responsive through three case studies from critical topics: DDoS attack detection, macroprogramming over IoT systems, and sensor data processing. Our results reveal that the GPT model under few-shot learning achieves 87.6% detection accuracy, whereas the fine-tuned GPT increases the value to 94.9%. Given a macroprogramming framework, the GPT model is capable of writing scripts using high-level functions from the framework to handle possible incidents. Moreover, the GPT model shows efficacy in processing a vast amount of sensor data by offering fast and high-quality responses, which comprise expected results and summarized insights. Overall, the model demonstrates its potential to power a natural language interface. We hope that researchers will find these case studies inspiring to develop further.

摘要: 本文通过关键主题的三个案例研究来识别和分析大型语言模型（LLM）可以使物联网（IoT）网络更加智能和响应能力更强的应用程序：DDOS攻击检测、物联网系统上的宏编程和传感器数据处理。我们的结果表明，少次学习下的GPT模型可实现87.6%的检测准确率，而微调的GPT可将该值提高到94.9%。给定宏编程框架，GPT模型能够使用框架中的高级函数编写脚本来处理可能的事件。此外，GPT模型通过提供快速、高质量的响应（包括预期结果和总结见解）显示出处理大量传感器数据的功效。总体而言，该模型展示了其支持自然语言界面的潜力。我们希望研究人员能够发现这些案例研究鼓舞人心，以进一步发展。



## **40. Adversarial Attacks on Large Language Models Using Regularized Relaxation**

使用正规松弛对大型语言模型的对抗攻击 cs.LG

8 pages, 6 figures

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.19160v1) [paper-pdf](http://arxiv.org/pdf/2410.19160v1)

**Authors**: Samuel Jacob Chacko, Sajib Biswas, Chashi Mahiul Islam, Fatema Tabassum Liza, Xiuwen Liu

**Abstract**: As powerful Large Language Models (LLMs) are now widely used for numerous practical applications, their safety is of critical importance. While alignment techniques have significantly improved overall safety, LLMs remain vulnerable to carefully crafted adversarial inputs. Consequently, adversarial attack methods are extensively used to study and understand these vulnerabilities. However, current attack methods face significant limitations. Those relying on optimizing discrete tokens suffer from limited efficiency, while continuous optimization techniques fail to generate valid tokens from the model's vocabulary, rendering them impractical for real-world applications. In this paper, we propose a novel technique for adversarial attacks that overcomes these limitations by leveraging regularized gradients with continuous optimization methods. Our approach is two orders of magnitude faster than the state-of-the-art greedy coordinate gradient-based method, significantly improving the attack success rate on aligned language models. Moreover, it generates valid tokens, addressing a fundamental limitation of existing continuous optimization methods. We demonstrate the effectiveness of our attack on five state-of-the-art LLMs using four datasets.

摘要: 随着强大的大型语言模型(LLM)在众多实际应用中的广泛应用，它们的安全性至关重要。虽然对齐技术显著提高了总体安全性，但LLM仍然容易受到精心设计的敌方输入的影响。因此，对抗性攻击方法被广泛用于研究和理解这些漏洞。然而，目前的攻击方法面临着很大的局限性。那些依赖于优化离散令牌的人效率有限，而连续优化技术无法从模型的词汇表中生成有效的令牌，这使得它们在现实世界中的应用不切实际。在本文中，我们提出了一种新的对抗性攻击技术，通过利用正则化的梯度和连续优化方法来克服这些局限性。我们的方法比最先进的贪婪坐标梯度方法快两个数量级，显著提高了对齐语言模型的攻击成功率。此外，它还生成有效的令牌，解决了现有连续优化方法的一个基本限制。我们使用四个数据集演示了我们对五个最先进的LLM的攻击的有效性。



## **41. Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications**

通过修剪和低级修改评估安全对齐的脆弱性 cs.LG

22 pages, 9 figures. Project page is available at  https://boyiwei.com/alignment-attribution/

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2402.05162v4) [paper-pdf](http://arxiv.org/pdf/2402.05162v4)

**Authors**: Boyi Wei, Kaixuan Huang, Yangsibo Huang, Tinghao Xie, Xiangyu Qi, Mengzhou Xia, Prateek Mittal, Mengdi Wang, Peter Henderson

**Abstract**: Large language models (LLMs) show inherent brittleness in their safety mechanisms, as evidenced by their susceptibility to jailbreaking and even non-malicious fine-tuning. This study explores this brittleness of safety alignment by leveraging pruning and low-rank modifications. We develop methods to identify critical regions that are vital for safety guardrails, and that are disentangled from utility-relevant regions at both the neuron and rank levels. Surprisingly, the isolated regions we find are sparse, comprising about $3\%$ at the parameter level and $2.5\%$ at the rank level. Removing these regions compromises safety without significantly impacting utility, corroborating the inherent brittleness of the model's safety mechanisms. Moreover, we show that LLMs remain vulnerable to low-cost fine-tuning attacks even when modifications to the safety-critical regions are restricted. These findings underscore the urgent need for more robust safety strategies in LLMs.

摘要: 大型语言模型（LLM）在其安全机制中表现出固有的脆弱性，这一点从它们容易越狱甚至非恶意微调中得到了证明。这项研究通过利用修剪和低等级修改来探索安全对齐的脆弱性。我们开发了识别对安全护栏至关重要的关键区域的方法，并且在神经元和等级水平上与公用事业相关区域脱钩。令人惊讶的是，我们发现的孤立区域很稀疏，参数级别约为3美元，排名级别约为2.5美元。删除这些区域会损害安全性，而不会显着影响实用性，这证实了该模型安全机制固有的脆弱性。此外，我们表明，即使对安全关键区域的修改受到限制，LLM仍然容易受到低成本微调攻击。这些发现凸显了LLM迫切需要制定更稳健的安全策略。



## **42. Watermarking Large Language Models and the Generated Content: Opportunities and Challenges**

对大型语言模型和生成的内容进行水印：机遇与挑战 cs.CR

invited paper to Asilomar Conference on Signals, Systems, and  Computers

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.19096v1) [paper-pdf](http://arxiv.org/pdf/2410.19096v1)

**Authors**: Ruisi Zhang, Farinaz Koushanfar

**Abstract**: The widely adopted and powerful generative large language models (LLMs) have raised concerns about intellectual property rights violations and the spread of machine-generated misinformation. Watermarking serves as a promising approch to establish ownership, prevent unauthorized use, and trace the origins of LLM-generated content. This paper summarizes and shares the challenges and opportunities we found when watermarking LLMs. We begin by introducing techniques for watermarking LLMs themselves under different threat models and scenarios. Next, we investigate watermarking methods designed for the content generated by LLMs, assessing their effectiveness and resilience against various attacks. We also highlight the importance of watermarking domain-specific models and data, such as those used in code generation, chip design, and medical applications. Furthermore, we explore methods like hardware acceleration to improve the efficiency of the watermarking process. Finally, we discuss the limitations of current approaches and outline future research directions for the responsible use and protection of these generative AI tools.

摘要: 被广泛采用且功能强大的生成性大型语言模型(LLM)引起了人们对侵犯知识产权和机器生成错误信息传播的担忧。水印是一种很有前途的方法，可以建立所有权，防止未经授权的使用，并追踪LLM生成的内容的来源。本文总结和分享了在对LLMS进行水印处理时所遇到的挑战和机遇。我们首先介绍在不同威胁模型和场景下为LLM本身添加水印的技术。接下来，我们研究了针对LLMS生成的内容设计的水印方法，评估了它们对各种攻击的有效性和抗攻击能力。我们还强调了为特定领域的模型和数据添加水印的重要性，例如用于代码生成、芯片设计和医疗应用的模型和数据。此外，我们还探索了硬件加速等方法来提高水印过程的效率。最后，我们讨论了当前方法的局限性，并概述了未来的研究方向，以负责任地使用和保护这些生成性人工智能工具。



## **43. Provably Robust Watermarks for Open-Source Language Models**

开源语言模型的可证明稳健的水印 cs.CR

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.18861v1) [paper-pdf](http://arxiv.org/pdf/2410.18861v1)

**Authors**: Miranda Christ, Sam Gunn, Tal Malkin, Mariana Raykova

**Abstract**: The recent explosion of high-quality language models has necessitated new methods for identifying AI-generated text. Watermarking is a leading solution and could prove to be an essential tool in the age of generative AI. Existing approaches embed watermarks at inference and crucially rely on the large language model (LLM) specification and parameters being secret, which makes them inapplicable to the open-source setting. In this work, we introduce the first watermarking scheme for open-source LLMs. Our scheme works by modifying the parameters of the model, but the watermark can be detected from just the outputs of the model. Perhaps surprisingly, we prove that our watermarks are unremovable under certain assumptions about the adversary's knowledge. To demonstrate the behavior of our construction under concrete parameter instantiations, we present experimental results with OPT-6.7B and OPT-1.3B. We demonstrate robustness to both token substitution and perturbation of the model parameters. We find that the stronger of these attacks, the model-perturbation attack, requires deteriorating the quality score to 0 out of 100 in order to bring the detection rate down to 50%.

摘要: 最近高质量语言模型的爆炸性增长需要新的方法来识别人工智能生成的文本。水印是一种领先的解决方案，可能会被证明是生成性人工智能时代的重要工具。现有的方法在推理时嵌入水印，重要的是依赖于大型语言模型(LLM)规范和参数是保密的，这使得它们不适用于开源环境。在这项工作中，我们介绍了第一个用于开源LLMS的水印方案。我们的方案通过修改模型的参数来工作，但仅从模型的输出就可以检测到水印。也许令人惊讶的是，我们证明了我们的水印在关于对手知识的某些假设下是不可移除的。为了演示我们的构造在混凝土参数实例化下的行为，我们给出了使用OPT-6.7B和OPT-1.3B的实验结果。我们证明了对令牌替换和模型参数摄动的稳健性。我们发现，在这些攻击中，较强的模型扰动攻击需要将质量分数恶化到0分(满分100分)，才能将检测率降至50%。



## **44. PSY: Posterior Sampling Based Privacy Enhancer in Large Language Models**

PSY：大型语言模型中基于后验抽样的隐私增强器 cs.CR

10 pages, 2 figures

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.18824v1) [paper-pdf](http://arxiv.org/pdf/2410.18824v1)

**Authors**: Yulian Sun, Li Duan, Yong Li

**Abstract**: Privacy vulnerabilities in LLMs, such as leakage from memorization, have been constantly identified, and various mitigation proposals have been proposed. LoRA is usually used in fine-tuning LLMs and a good entry point to insert privacy-enhancing modules. In this ongoing research, we introduce PSY, a Posterior Sampling based PrivacY enhancer that can be used in LoRA. We propose a simple yet effective realization of PSY using posterior sampling, which effectively prevents privacy leakage from intermediate information and, in turn, preserves the privacy of data owners. We evaluate LoRA extended with PSY against state-of-the-art membership inference and data extraction attacks. The experiments are executed on three different LLM architectures fine-tuned on three datasets with LoRA. In contrast to the commonly used differential privacy method, we find that our proposed modification consistently reduces the attack success rate. Meanwhile, our method has almost no negative impact on model fine-tuning or final performance. Most importantly, PSY reveals a promising path toward privacy enhancement with latent space extensions.

摘要: LLMS中的隐私漏洞，如记忆泄漏，不断被发现，并提出了各种缓解建议。LORA通常用于微调LLM，是插入隐私增强模块的一个很好的切入点。在这项正在进行的研究中，我们介绍了PSY，一种基于后验抽样的隐私增强器，可以在LORA中使用。我们提出了一种简单而有效的后验抽样的PSY实现方法，它有效地防止了中间信息的隐私泄露，进而保护了数据所有者的隐私。我们评估了使用PSY扩展的LORA对最先进的成员关系推理和数据提取攻击的抵抗力。实验是在三种不同的LLM架构上进行的，这些架构使用LORA在三个数据集上进行了微调。与常用的差分隐私方法相比，我们发现我们提出的修改一致地降低了攻击成功率。同时，我们的方法对模型微调或最终性能几乎没有负面影响。最重要的是，PSY揭示了一条通过潜在空间扩展来增强隐私的有希望的途径。



## **45. Advancing NLP Security by Leveraging LLMs as Adversarial Engines**

通过利用LLC作为对抗引擎来提高NLP安全性 cs.AI

5 pages

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.18215v1) [paper-pdf](http://arxiv.org/pdf/2410.18215v1)

**Authors**: Sudarshan Srinivasan, Maria Mahbub, Amir Sadovnik

**Abstract**: This position paper proposes a novel approach to advancing NLP security by leveraging Large Language Models (LLMs) as engines for generating diverse adversarial attacks. Building upon recent work demonstrating LLMs' effectiveness in creating word-level adversarial examples, we argue for expanding this concept to encompass a broader range of attack types, including adversarial patches, universal perturbations, and targeted attacks. We posit that LLMs' sophisticated language understanding and generation capabilities can produce more effective, semantically coherent, and human-like adversarial examples across various domains and classifier architectures. This paradigm shift in adversarial NLP has far-reaching implications, potentially enhancing model robustness, uncovering new vulnerabilities, and driving innovation in defense mechanisms. By exploring this new frontier, we aim to contribute to the development of more secure, reliable, and trustworthy NLP systems for critical applications.

摘要: 这份立场文件提出了一种新颖的方法，通过利用大型语言模型（LLM）作为生成多样化对抗攻击的引擎来提高NLP安全性。在最近展示LLM在创建单词级对抗性示例方面有效性的工作的基础上，我们主张扩展这一概念以涵盖更广泛的攻击类型，包括对抗性补丁、普遍扰动和有针对性的攻击。我们证实，LLM复杂的语言理解和生成能力可以在各个领域和分类器架构中生成更有效、语义一致且类人的对抗性示例。对抗性NLP的这种范式转变具有深远的影响，可能会增强模型稳健性、发现新的漏洞并推动防御机制的创新。通过探索这一新领域，我们的目标是为关键应用开发更安全、可靠和值得信赖的NLP系统做出贡献。



## **46. Towards Understanding the Fragility of Multilingual LLMs against Fine-Tuning Attacks**

了解多语言LLM对微调攻击的脆弱性 cs.CL

14 pages, 6 figures, 7 tables

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.18210v1) [paper-pdf](http://arxiv.org/pdf/2410.18210v1)

**Authors**: Samuele Poppi, Zheng-Xin Yong, Yifei He, Bobbie Chern, Han Zhao, Aobo Yang, Jianfeng Chi

**Abstract**: Recent advancements in Large Language Models (LLMs) have sparked widespread concerns about their safety. Recent work demonstrates that safety alignment of LLMs can be easily removed by fine-tuning with a few adversarially chosen instruction-following examples, i.e., fine-tuning attacks. We take a further step to understand fine-tuning attacks in multilingual LLMs. We first discover cross-lingual generalization of fine-tuning attacks: using a few adversarially chosen instruction-following examples in one language, multilingual LLMs can also be easily compromised (e.g., multilingual LLMs fail to refuse harmful prompts in other languages). Motivated by this finding, we hypothesize that safety-related information is language-agnostic and propose a new method termed Safety Information Localization (SIL) to identify the safety-related information in the model parameter space. Through SIL, we validate this hypothesis and find that only changing 20% of weight parameters in fine-tuning attacks can break safety alignment across all languages. Furthermore, we provide evidence to the alternative pathways hypothesis for why freezing safety-related parameters does not prevent fine-tuning attacks, and we demonstrate that our attack vector can still jailbreak LLMs adapted to new languages.

摘要: 最近大型语言模型(LLM)的进步引发了人们对其安全性的广泛担忧。最近的工作表明，通过使用一些恶意选择的指令跟随示例，即微调攻击，可以很容易地删除LLM的安全对齐。我们进一步了解多语言LLM中的微调攻击。我们首先发现了微调攻击的跨语言泛化：使用几个恶意选择的一种语言的指令跟随示例，多语言LLM也很容易被攻破(例如，多语言LLM无法拒绝其他语言的有害提示)。基于这一发现，我们假设安全相关信息是语言不可知的，并提出了一种在模型参数空间中识别安全相关信息的新方法--安全信息本地化。通过SIL语言，我们验证了这一假设，发现在微调攻击中只改变20%的权重参数就可以破坏所有语言的安全对齐。此外，我们为替代路径假说提供了证据，证明了冻结安全相关参数为什么不能防止微调攻击，并证明了我们的攻击向量仍然可以越狱适应新语言的LLM。



## **47. Safeguard is a Double-edged Sword: Denial-of-service Attack on Large Language Models**

保障是一把双刃剑：对大型语言模型的拒绝服务攻击 cs.CR

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.02916v2) [paper-pdf](http://arxiv.org/pdf/2410.02916v2)

**Authors**: Qingzhao Zhang, Ziyang Xiong, Z. Morley Mao

**Abstract**: Safety is a paramount concern of large language models (LLMs) in their open deployment. To this end, safeguard methods aim to enforce the ethical and responsible use of LLMs through safety alignment or guardrail mechanisms. However, we found that the malicious attackers could exploit false positives of safeguards, i.e., fooling the safeguard model to block safe content mistakenly, leading to a new denial-of-service (DoS) attack on LLMs. Specifically, by software or phishing attacks on user client software, attackers insert a short, seemingly innocuous adversarial prompt into to user prompt templates in configuration files; thus, this prompt appears in final user requests without visibility in the user interface and is not trivial to identify. By designing an optimization process that utilizes gradient and attention information, our attack can automatically generate seemingly safe adversarial prompts, approximately only 30 characters long, that universally block over 97\% of user requests on Llama Guard 3. The attack presents a new dimension of evaluating LLM safeguards focusing on false positives, fundamentally different from the classic jailbreak.

摘要: 安全是大型语言模型(LLM)在开放部署时最关心的问题。为此，保障措施旨在通过安全调整或护栏机制，强制以合乎道德和负责任的方式使用LLMS。然而，我们发现恶意攻击者可以利用安全措施的误报，即欺骗安全措施模型错误地阻止安全内容，从而导致对LLMS的新的拒绝服务(DoS)攻击。具体地说，通过软件或对用户客户端软件的网络钓鱼攻击，攻击者将一个看似无害的简短对抗性提示插入到配置文件中的用户提示模板中；因此，该提示出现在最终用户请求中，在用户界面中不可见，并且很难识别。通过设计一个利用梯度和注意力信息的优化过程，我们的攻击可以自动生成看似安全的敌意提示，大约只有30个字符，普遍阻止Llama Guard 3上超过97%的用户请求。该攻击提供了一个新的维度来评估LLM安全措施，从根本上不同于传统的越狱。



## **48. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

SCA：高效语义一致的无限制对抗攻击 cs.CV

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.02240v4) [paper-pdf](http://arxiv.org/pdf/2410.02240v4)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.

摘要: 部署在敏感环境中的基于深度神经网络的系统容易受到敌意攻击。不受限制的对抗性攻击通常操纵图像的语义内容(例如，颜色或纹理)以创建既有效又逼真的对抗性示例。最近的工作利用扩散逆过程将图像映射到潜在空间，在潜在空间中通过引入扰动来操纵高级语义。然而，它们往往会在去噪输出中造成严重的语义扭曲，并导致效率低下。在这项研究中，我们提出了一种新的框架，称为语义一致的无限对抗攻击(SCA)，它使用一种反转方法来提取编辑友好的噪声映射，并利用多模式大语言模型(MLLM)在整个过程中提供语义指导。在MLLM提供丰富语义信息的条件下，使用一系列编辑友好的噪声图对每个步骤进行DDPM去噪处理，并利用DPM Solver++加速这一过程，从而实现高效的语义一致性采样。与现有的方法相比，我们的框架能够高效地生成对抗性的例子，这些例子表现出最小的可识别的语义变化。因此，我们首次引入了语义一致的对抗性例子(SCAE)。广泛的实验和可视化已经证明了SCA的高效率，特别是在平均速度上是最先进的攻击的12倍。我们的研究可以进一步引起人们对多媒体信息安全的关注。



## **49. Guide for Defense (G4D): Dynamic Guidance for Robust and Balanced Defense in Large Language Models**

防御指南（G4 D）：大型语言模型中稳健和平衡防御的动态指南 cs.AI

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.17922v1) [paper-pdf](http://arxiv.org/pdf/2410.17922v1)

**Authors**: He Cao, Weidi Luo, Yu Wang, Zijing Liu, Bing Feng, Yuan Yao, Yu Li

**Abstract**: With the extensive deployment of Large Language Models (LLMs), ensuring their safety has become increasingly critical. However, existing defense methods often struggle with two key issues: (i) inadequate defense capabilities, particularly in domain-specific scenarios like chemistry, where a lack of specialized knowledge can lead to the generation of harmful responses to malicious queries. (ii) over-defensiveness, which compromises the general utility and responsiveness of LLMs. To mitigate these issues, we introduce a multi-agents-based defense framework, Guide for Defense (G4D), which leverages accurate external information to provide an unbiased summary of user intentions and analytically grounded safety response guidance. Extensive experiments on popular jailbreak attacks and benign datasets show that our G4D can enhance LLM's robustness against jailbreak attacks on general and domain-specific scenarios without compromising the model's general functionality.

摘要: 随着大型语言模型（LLM）的广泛部署，确保其安全性变得越来越重要。然而，现有的防御方法经常遇到两个关键问题：（i）防御能力不足，特别是在化学等特定领域的场景中，缺乏专业知识可能会导致对恶意查询产生有害响应。(ii)过度防御，这会损害LLM的一般实用性和响应能力。为了缓解这些问题，我们引入了一个基于多代理的防御框架--防御指南（G4 D），该框架利用准确的外部信息来提供用户意图的公正摘要和基于分析的安全响应指南。对流行越狱攻击和良性数据集的广泛实验表明，我们的G4 D可以增强LLM针对一般和特定领域场景的越狱攻击的鲁棒性，而不会损害模型的一般功能。



## **50. IBGP: Imperfect Byzantine Generals Problem for Zero-Shot Robustness in Communicative Multi-Agent Systems**

IBGP：通信多智能体系统中零攻击鲁棒性的不完美拜占庭将军问题 cs.MA

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.16237v2) [paper-pdf](http://arxiv.org/pdf/2410.16237v2)

**Authors**: Yihuan Mao, Yipeng Kang, Peilun Li, Ning Zhang, Wei Xu, Chongjie Zhang

**Abstract**: As large language model (LLM) agents increasingly integrate into our infrastructure, their robust coordination and message synchronization become vital. The Byzantine Generals Problem (BGP) is a critical model for constructing resilient multi-agent systems (MAS) under adversarial attacks. It describes a scenario where malicious agents with unknown identities exist in the system-situations that, in our context, could result from LLM agents' hallucinations or external attacks. In BGP, the objective of the entire system is to reach a consensus on the action to be taken. Traditional BGP requires global consensus among all agents; however, in practical scenarios, global consensus is not always necessary and can even be inefficient. Therefore, there is a pressing need to explore a refined version of BGP that aligns with the local coordination patterns observed in MAS. We refer to this refined version as Imperfect BGP (IBGP) in our research, aiming to address this discrepancy. To tackle this issue, we propose a framework that leverages consensus protocols within general MAS settings, providing provable resilience against communication attacks and adaptability to changing environments, as validated by empirical results. Additionally, we present a case study in a sensor network environment to illustrate the practical application of our protocol.

摘要: 随着大型语言模型(LLM)代理越来越多地集成到我们的基础设施中，它们强大的协调和消息同步变得至关重要。拜占庭将军问题(BGP)是在对抗攻击下构造具有弹性的多智能体系统(MAS)的重要模型。它描述了一种系统中存在身份未知的恶意代理的情况--在我们的上下文中，这种情况可能是由于LLM代理的幻觉或外部攻击造成的。在BGP中，整个系统的目标是就要采取的行动达成共识。传统的BGP需要在所有代理之间达成全局共识；然而，在实际场景中，全局共识并不总是必要的，甚至可能效率低下。因此，迫切需要探索一种与MAS中观察到的局部协调模式相一致的BGP改进版本。在我们的研究中，我们将这种精炼版本称为不完美BGP(IBGP)，旨在解决这一差异。为了解决这个问题，我们提出了一个框架，它在一般的MAS环境中利用共识协议，提供对通信攻击的可证明的弹性和对不断变化的环境的适应性，实验结果验证了这一点。此外，我们还给出了一个传感器网络环境下的案例研究，以说明该协议的实际应用。



