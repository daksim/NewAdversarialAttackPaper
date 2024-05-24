# Latest Large Language Model Attack Papers
**update at 2024-05-24 10:13:16**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Representation noising effectively prevents harmful fine-tuning on LLMs**

表示噪音有效防止对LLM的有害微调 cs.CL

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14577v1) [paper-pdf](http://arxiv.org/pdf/2405.14577v1)

**Authors**: Domenic Rosati, Jan Wehner, Kai Williams, Łukasz Bartoszcze, David Atanasov, Robie Gonzales, Subhabrata Majumdar, Carsten Maple, Hassan Sajjad, Frank Rudzicz

**Abstract**: Releasing open-source large language models (LLMs) presents a dual-use risk since bad actors can easily fine-tune these models for harmful purposes. Even without the open release of weights, weight stealing and fine-tuning APIs make closed models vulnerable to harmful fine-tuning attacks (HFAs). While safety measures like preventing jailbreaks and improving safety guardrails are important, such measures can easily be reversed through fine-tuning. In this work, we propose Representation Noising (RepNoise), a defence mechanism that is effective even when attackers have access to the weights and the defender no longer has any control. RepNoise works by removing information about harmful representations such that it is difficult to recover them during fine-tuning. Importantly, our defence is also able to generalize across different subsets of harm that have not been seen during the defence process. Our method does not degrade the general capability of LLMs and retains the ability to train the model on harmless tasks. We provide empirical evidence that the effectiveness of our defence lies in its "depth": the degree to which information about harmful representations is removed across all layers of the LLM.

摘要: 发布开源的大型语言模型(LLM)存在双重用途的风险，因为不好的参与者很容易出于有害目的微调这些模型。即使没有公开的权重释放，权重盗窃和微调API也会使封闭的模型容易受到有害的微调攻击(HFA)。虽然防止越狱和改善安全护栏等安全措施很重要，但通过微调很容易逆转这些措施。在这项工作中，我们提出了表示噪声(RepNoise)，这是一种即使攻击者可以访问权重而防御者不再具有任何控制权的防御机制。RepNoise的工作原理是删除有关有害表示的信息，以便在微调期间很难恢复它们。重要的是，我们的辩护还能够概括在辩护过程中未曾见过的伤害的不同子集。我们的方法不会降低LLMS的整体性能，并保留了对模型进行无害任务训练的能力。我们提供的经验证据表明，我们辩护的有效性在于它的“深度”：在LLM的所有层中，有关有害陈述的信息被移除的程度。



## **2. Self-playing Adversarial Language Game Enhances LLM Reasoning**

自玩对抗语言游戏增强LLM推理 cs.CL

Preprint

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2404.10642v2) [paper-pdf](http://arxiv.org/pdf/2404.10642v2)

**Authors**: Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Yong Dai, Lei Han, Nan Du

**Abstract**: We explore the self-play training procedure of large language models (LLMs) in a two-player adversarial language game called Adversarial Taboo. In this game, an attacker and a defender communicate around a target word only visible to the attacker. The attacker aims to induce the defender to speak the target word unconsciously, while the defender tries to infer the target word from the attacker's utterances. To win the game, both players should have sufficient knowledge about the target word and high-level reasoning ability to infer and express in this information-reserved conversation. Hence, we are curious about whether LLMs' reasoning ability can be further enhanced by self-play in this adversarial language game (SPAG). With this goal, we select several open-source LLMs and let each act as the attacker and play with a copy of itself as the defender on an extensive range of target words. Through reinforcement learning on the game outcomes, we observe that the LLMs' performances uniformly improve on a broad range of reasoning benchmarks. Furthermore, iteratively adopting this self-play process can continuously promote LLMs' reasoning abilities. The code is at https://github.com/Linear95/SPAG.

摘要: 我们探索了在一个名为对抗性禁忌的两人对抗性语言游戏中，大语言模型(LLM)的自我发挥训练过程。在这个游戏中，攻击者和防御者围绕一个只有攻击者才能看到的目标单词进行交流。攻击者的目的是诱导防御者无意识地说出目标词，而防御者则试图从攻击者的话语中推断出目标词。要赢得这场比赛，双方都应该有足够的目标词知识和高级推理能力，以便在这种信息储备的对话中进行推理和表达。因此，我们好奇的是，在这场对抗性语言游戏(SPAG)中，LLMS的推理能力能否通过自我游戏进一步增强。带着这个目标，我们选择了几个开源的LLM，让每个LLM扮演攻击者的角色，并在广泛的目标词上扮演自己的防御者。通过对游戏结果的强化学习，我们观察到LLMS的性能在广泛的推理基准上一致提高。此外，迭代地采用这种自我发挥过程可以不断提升LLMS的推理能力。代码在https://github.com/Linear95/SPAG.



## **3. S-Eval: Automatic and Adaptive Test Generation for Benchmarking Safety Evaluation of Large Language Models**

S-Eval：用于大型语言模型基准安全评估的自动和自适应测试生成 cs.CR

18 pages, 11 figures

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14191v1) [paper-pdf](http://arxiv.org/pdf/2405.14191v1)

**Authors**: Xiaohan Yuan, Jinfeng Li, Dongxia Wang, Yuefeng Chen, Xiaofeng Mao, Longtao Huang, Hui Xue, Wenhai Wang, Kui Ren, Jingyi Wang

**Abstract**: Large Language Models have gained considerable attention for their revolutionary capabilities. However, there is also growing concern on their safety implications, making a comprehensive safety evaluation for LLMs urgently needed before model deployment. In this work, we propose S-Eval, a new comprehensive, multi-dimensional and open-ended safety evaluation benchmark. At the core of S-Eval is a novel LLM-based automatic test prompt generation and selection framework, which trains an expert testing LLM Mt combined with a range of test selection strategies to automatically construct a high-quality test suite for the safety evaluation. The key to the automation of this process is a novel expert safety-critique LLM Mc able to quantify the riskiness score of a LLM's response, and additionally produce risk tags and explanations. Besides, the generation process is also guided by a carefully designed risk taxonomy with four different levels, covering comprehensive and multi-dimensional safety risks of concern. Based on these, we systematically construct a new and large-scale safety evaluation benchmark for LLMs consisting of 220,000 evaluation prompts, including 20,000 base risk prompts (10,000 in Chinese and 10,000 in English) and 200, 000 corresponding attack prompts derived from 10 popular adversarial instruction attacks against LLMs. Moreover, considering the rapid evolution of LLMs and accompanied safety threats, S-Eval can be flexibly configured and adapted to include new risks, attacks and models. S-Eval is extensively evaluated on 20 popular and representative LLMs. The results confirm that S-Eval can better reflect and inform the safety risks of LLMs compared to existing benchmarks. We also explore the impacts of parameter scales, language environments, and decoding parameters on the evaluation, providing a systematic methodology for evaluating the safety of LLMs.

摘要: 大型语言模型因其革命性的能力而获得了相当大的关注。然而，人们也越来越担心它们的安全影响，这使得在模型部署之前迫切需要对LLMS进行全面的安全评估。在这项工作中，我们提出了一种新的全面、多维、开放式的安全评价基准S-EVAL。S-EVAL的核心是一种新颖的基于LLM的测试提示自动生成和选择框架，该框架训练一名测试专家，结合一系列测试选择策略，自动构建用于安全评估的高质量测试用例集。这一过程自动化的关键是一种新颖的专家安全评论LLm Mc，它能够量化LLm响应的风险分数，并另外产生风险标签和解释。此外，生成过程还遵循了精心设计的四个不同级别的风险分类，涵盖了令人关注的全面和多维度的安全风险。在此基础上，我们系统地构建了一个新的大规模的低层管理系统安全评估基准，该基准由22万条评估提示组成，其中包括2万条基本风险提示(中文10000条，英文10000条)和来自10种流行的对抗性指令攻击的20万条相应的攻击提示。此外，考虑到LLM的快速演化和伴随的安全威胁，S-EVAL可以灵活配置和调整，以包括新的风险、攻击和模型。S-EVAL在20个流行和有代表性的低成本模型上进行了广泛的评估。结果证实，与现有基准相比，S-EVAL能够更好地反映和告知低成本机械的安全风险。我们还探讨了参数尺度、语言环境和解码参数对评估的影响，为评估LLMS的安全性提供了一种系统的方法。



## **4. Towards Transferable Attacks Against Vision-LLMs in Autonomous Driving with Typography**

通过印刷术在自动驾驶中针对视觉LLM的可转移攻击 cs.CV

12 pages, 5 tables, 5 figures, work in progress

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14169v1) [paper-pdf](http://arxiv.org/pdf/2405.14169v1)

**Authors**: Nhat Chung, Sensen Gao, Tuan-Anh Vu, Jie Zhang, Aishan Liu, Yun Lin, Jin Song Dong, Qing Guo

**Abstract**: Vision-Large-Language-Models (Vision-LLMs) are increasingly being integrated into autonomous driving (AD) systems due to their advanced visual-language reasoning capabilities, targeting the perception, prediction, planning, and control mechanisms. However, Vision-LLMs have demonstrated susceptibilities against various types of adversarial attacks, which would compromise their reliability and safety. To further explore the risk in AD systems and the transferability of practical threats, we propose to leverage typographic attacks against AD systems relying on the decision-making capabilities of Vision-LLMs. Different from the few existing works developing general datasets of typographic attacks, this paper focuses on realistic traffic scenarios where these attacks can be deployed, on their potential effects on the decision-making autonomy, and on the practical ways in which these attacks can be physically presented. To achieve the above goals, we first propose a dataset-agnostic framework for automatically generating false answers that can mislead Vision-LLMs' reasoning. Then, we present a linguistic augmentation scheme that facilitates attacks at image-level and region-level reasoning, and we extend it with attack patterns against multiple reasoning tasks simultaneously. Based on these, we conduct a study on how these attacks can be realized in physical traffic scenarios. Through our empirical study, we evaluate the effectiveness, transferability, and realizability of typographic attacks in traffic scenes. Our findings demonstrate particular harmfulness of the typographic attacks against existing Vision-LLMs (e.g., LLaVA, Qwen-VL, VILA, and Imp), thereby raising community awareness of vulnerabilities when incorporating such models into AD systems. We will release our source code upon acceptance.

摘要: 视觉大语言模型(Vision-LLMS)由于其先进的视觉语言推理能力，针对感知、预测、规划和控制机制，越来越多地被集成到自动驾驶(AD)系统中。然而，Vision-LLM已经显示出对各种类型的对抗性攻击的敏感性，这将损害它们的可靠性和安全性。为了进一步探索AD系统中的风险和实际威胁的可转移性，我们建议依靠Vision-LLMS的决策能力来利用排版攻击AD系统。与现有开发排版攻击通用数据集的少数工作不同，本文关注的是可以部署这些攻击的现实交通场景，它们对决策自主性的潜在影响，以及这些攻击可以物理呈现的实际方式。为了实现上述目标，我们首先提出了一个与数据集无关的框架，用于自动生成可能误导Vision-LLMS推理的错误答案。在此基础上，提出了一种支持图像级和区域级推理攻击的语言扩充方案，并将其扩展为同时针对多个推理任务的攻击模式。在此基础上，对如何在物理流量场景下实现这些攻击进行了研究。通过我们的实证研究，我们评估了交通场景中排版攻击的有效性、可转移性和可实现性。我们的发现证明了针对现有Vision-LLM(例如LLaVA、Qwen-VL、Vila和IMP)的排版攻击的特殊危害性，从而提高了社区对将此类模型整合到AD系统中时的漏洞的认识。我们将在接受后公布我们的源代码。



## **5. WordGame: Efficient & Effective LLM Jailbreak via Simultaneous Obfuscation in Query and Response**

WordGame：通过查询和响应中的同时混淆高效且有效的LLM越狱 cs.LG

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.14023v1) [paper-pdf](http://arxiv.org/pdf/2405.14023v1)

**Authors**: Tianrong Zhang, Bochuan Cao, Yuanpu Cao, Lu Lin, Prasenjit Mitra, Jinghui Chen

**Abstract**: The recent breakthrough in large language models (LLMs) such as ChatGPT has revolutionized production processes at an unprecedented pace. Alongside this progress also comes mounting concerns about LLMs' susceptibility to jailbreaking attacks, which leads to the generation of harmful or unsafe content. While safety alignment measures have been implemented in LLMs to mitigate existing jailbreak attempts and force them to become increasingly complicated, it is still far from perfect. In this paper, we analyze the common pattern of the current safety alignment and show that it is possible to exploit such patterns for jailbreaking attacks by simultaneous obfuscation in queries and responses. Specifically, we propose WordGame attack, which replaces malicious words with word games to break down the adversarial intent of a query and encourage benign content regarding the games to precede the anticipated harmful content in the response, creating a context that is hardly covered by any corpus used for safety alignment. Extensive experiments demonstrate that WordGame attack can break the guardrails of the current leading proprietary and open-source LLMs, including the latest Claude-3, GPT-4, and Llama-3 models. Further ablation studies on such simultaneous obfuscation in query and response provide evidence of the merits of the attack strategy beyond an individual attack.

摘要: 最近ChatGPT等大型语言模型(LLM)的突破以前所未有的速度彻底改变了生产流程。在取得这一进展的同时，人们也越来越担心LLMS容易受到越狱攻击，这会导致产生有害或不安全的内容。虽然LLMS已经实施了安全调整措施，以减轻现有的越狱企图，并迫使它们变得越来越复杂，但它仍然远未达到完美。在本文中，我们分析了当前安全对齐的常见模式，并证明了通过在查询和响应中同时混淆来利用这种模式来进行越狱攻击是可能的。具体地说，我们提出了文字游戏攻击，它用文字游戏取代恶意单词，以打破查询的敌对意图，并鼓励与游戏有关的良性内容在响应中位于预期的有害内容之前，从而创造出任何用于安全对齐的语料库几乎无法覆盖的上下文。大量实验表明，文字游戏攻击可以打破目前领先的专有和开源LLM的屏障，包括最新的Claude-3、GPT-4和Llama-3型号。对查询和响应中的这种同时混淆的进一步消融研究提供了除单个攻击之外的攻击策略的优点的证据。



## **6. Unleashing the Power of Unlabeled Data: A Self-supervised Learning Framework for Cyber Attack Detection in Smart Grids**

释放未标记数据的力量：智能电网中网络攻击检测的自我监督学习框架 cs.LG

9 pages, 5 figures

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.13965v1) [paper-pdf](http://arxiv.org/pdf/2405.13965v1)

**Authors**: Hanyu Zeng, Pengfei Zhou, Xin Lou, Zhen Wei Ng, David K. Y. Yau, Marianne Winslett

**Abstract**: Modern power grids are undergoing significant changes driven by information and communication technologies (ICTs), and evolving into smart grids with higher efficiency and lower operation cost. Using ICTs, however, comes with an inevitable side effect that makes the power system more vulnerable to cyber attacks. In this paper, we propose a self-supervised learning-based framework to detect and identify various types of cyber attacks. Different from existing approaches, the proposed framework does not rely on large amounts of well-curated labeled data but makes use of the massive unlabeled data in the wild which are easily accessible. Specifically, the proposed framework adopts the BERT model from the natural language processing domain and learns generalizable and effective representations from the unlabeled sensing data, which capture the distinctive patterns of different attacks. Using the learned representations, together with a very small amount of labeled data, we can train a task-specific classifier to detect various types of cyber attacks. Meanwhile, real-world training datasets are usually imbalanced, i.e., there are only a limited number of data samples containing attacks. In order to cope with such data imbalance, we propose a new loss function, separate mean error (SME), which pays equal attention to the large and small categories to better train the model. Experiment results in a 5-area power grid system with 37 buses demonstrate the superior performance of our framework over existing approaches, especially when a very limited portion of labeled data are available, e.g., as low as 0.002\%. We believe such a framework can be easily adopted to detect a variety of cyber attacks in other power grid scenarios.

摘要: 在信息和通信技术(ICT)的推动下，现代电网正在发生重大变化，并向效率更高、运行成本更低的智能电网演进。然而，使用信息和通信技术不可避免地会产生副作用，使电力系统更容易受到网络攻击。本文提出了一种基于自监督学习的框架来检测和识别各种类型的网络攻击。与现有方法不同的是，该框架不依赖于大量经过精心挑选的标记数据，而是利用了大量自然环境中易于访问的未标记数据。具体地说，该框架采用了自然语言处理领域的ERT模型，并从未标记的感知数据中学习可泛化的有效表示，从而捕捉到不同攻击的不同模式。使用学习的表示，加上非常少量的标记数据，我们可以训练特定于任务的分类器来检测各种类型的网络攻击。同时，现实世界的训练数据集通常是不平衡的，即只有有限数量的包含攻击的数据样本。为了解决这种数据失衡的问题，我们提出了一种新的损失函数--分离平均误差(SME)，它对大、小类别同等重视，以更好地训练模型。在具有37条节点的5区域电网系统上的实验结果表明，该框架的性能优于现有的方法，特别是在标签数据非常有限的情况下，例如低至0.002\%。我们相信这样的框架可以很容易地被采用来检测其他电网场景中的各种网络攻击。



## **7. Safety Alignment for Vision Language Models**

视觉语言模型的安全一致 cs.CV

23 pages, 15 figures

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.13581v1) [paper-pdf](http://arxiv.org/pdf/2405.13581v1)

**Authors**: Zhendong Liu, Yuanbi Nie, Yingshui Tan, Xiangyu Yue, Qiushi Cui, Chongjun Wang, Xiaoyong Zhu, Bo Zheng

**Abstract**: Benefiting from the powerful capabilities of Large Language Models (LLMs), pre-trained visual encoder models connected to an LLMs can realize Vision Language Models (VLMs). However, existing research shows that the visual modality of VLMs is vulnerable, with attackers easily bypassing LLMs' safety alignment through visual modality features to launch attacks. To address this issue, we enhance the existing VLMs' visual modality safety alignment by adding safety modules, including a safety projector, safety tokens, and a safety head, through a two-stage training process, effectively improving the model's defense against risky images. For example, building upon the LLaVA-v1.5 model, we achieve a safety score of 8.26, surpassing the GPT-4V on the Red Teaming Visual Language Models (RTVLM) benchmark. Our method boasts ease of use, high flexibility, and strong controllability, and it enhances safety while having minimal impact on the model's general performance. Moreover, our alignment strategy also uncovers some possible risky content within commonly used open-source multimodal datasets. Our code will be open sourced after the anonymous review.

摘要: 利用大语言模型的强大功能，将预先训练好的视觉编码器模型连接到大语言模型上，就可以实现视觉语言模型。然而，现有的研究表明，VLMS的视觉通道是易受攻击的，攻击者很容易通过视觉通道功能绕过LLMS的安全对齐进行攻击。为了解决这个问题，我们通过两个阶段的培训过程增加了安全模块，包括安全投影仪、安全令牌和安全头，从而增强了现有VLMS的视觉通道安全对准，有效地提高了模型对危险图像的防御能力。例如，在LLaVA-v1.5模型的基础上，我们获得了8.26的安全分数，超过了红队视觉语言模型(RTVLM)基准上的GPT-4V。该方法具有使用方便、灵活性高、可控性强等特点，在对模型整体性能影响最小的情况下提高了安全性。此外，我们的对齐策略还发现了常用开源多模式数据集中可能存在的一些风险内容。我们的代码将在匿名审查后开放源代码。



## **8. L-AutoDA: Leveraging Large Language Models for Automated Decision-based Adversarial Attacks**

L-AutoDA：利用大型语言模型进行自动化基于决策的对抗性攻击 cs.CR

Camera ready version for GECCO'24 workshop

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2401.15335v2) [paper-pdf](http://arxiv.org/pdf/2401.15335v2)

**Authors**: Ping Guo, Fei Liu, Xi Lin, Qingchuan Zhao, Qingfu Zhang

**Abstract**: In the rapidly evolving field of machine learning, adversarial attacks present a significant challenge to model robustness and security. Decision-based attacks, which only require feedback on the decision of a model rather than detailed probabilities or scores, are particularly insidious and difficult to defend against. This work introduces L-AutoDA (Large Language Model-based Automated Decision-based Adversarial Attacks), a novel approach leveraging the generative capabilities of Large Language Models (LLMs) to automate the design of these attacks. By iteratively interacting with LLMs in an evolutionary framework, L-AutoDA automatically designs competitive attack algorithms efficiently without much human effort. We demonstrate the efficacy of L-AutoDA on CIFAR-10 dataset, showing significant improvements over baseline methods in both success rate and computational efficiency. Our findings underscore the potential of language models as tools for adversarial attack generation and highlight new avenues for the development of robust AI systems.

摘要: 在快速发展的机器学习领域，敌意攻击对模型的健壮性和安全性提出了重大挑战。基于决策的攻击，只需要对模型的决策进行反馈，而不需要详细的概率或分数，特别隐蔽，很难防御。本文介绍了L-AUTODA(基于大语言模型的自动决策对抗性攻击)，它是一种利用大语言模型的生成能力来自动化攻击设计的新方法。通过在进化框架中迭代地与LLM交互，L自动DA无需太多人力即可自动高效地设计竞争攻击算法。我们在CIFAR-10数据集上验证了L-AutoDA的有效性，在成功率和计算效率上都比基线方法有了显著的提高。我们的发现强调了语言模型作为对抗性攻击生成工具的潜力，并强调了开发健壮的人工智能系统的新途径。



## **9. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models**

TrojanRAG：检索增强生成可以成为大型语言模型中的后门驱动程序 cs.CR

18 pages, 13 figures, 4 tables

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.13401v1) [paper-pdf](http://arxiv.org/pdf/2405.13401v1)

**Authors**: Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Large language models (LLMs) have raised concerns about potential security threats despite performing significantly in Natural Language Processing (NLP). Backdoor attacks initially verified that LLM is doing substantial harm at all stages, but the cost and robustness have been criticized. Attacking LLMs is inherently risky in security review, while prohibitively expensive. Besides, the continuous iteration of LLMs will degrade the robustness of backdoors. In this paper, we propose TrojanRAG, which employs a joint backdoor attack in the Retrieval-Augmented Generation, thereby manipulating LLMs in universal attack scenarios. Specifically, the adversary constructs elaborate target contexts and trigger sets. Multiple pairs of backdoor shortcuts are orthogonally optimized by contrastive learning, thus constraining the triggering conditions to a parameter subspace to improve the matching. To improve the recall of the RAG for the target contexts, we introduce a knowledge graph to construct structured data to achieve hard matching at a fine-grained level. Moreover, we normalize the backdoor scenarios in LLMs to analyze the real harm caused by backdoors from both attackers' and users' perspectives and further verify whether the context is a favorable tool for jailbreaking models. Extensive experimental results on truthfulness, language understanding, and harmfulness show that TrojanRAG exhibits versatility threats while maintaining retrieval capabilities on normal queries.

摘要: 尽管大型语言模型(LLM)在自然语言处理(NLP)中表现出色，但仍引发了人们对潜在安全威胁的担忧。后门攻击最初证实了LLM在所有阶段都在造成实质性的危害，但其成本和健壮性受到了批评。在安全审查中，攻击LLMS固有的风险，同时代价高得令人望而却步。此外，LLMS的连续迭代会降低后门的健壮性。在本文中，我们提出了TrojanRAG，它在检索-增强生成中使用联合后门攻击，从而在通用攻击场景下操纵LLMS。具体地说，对手构建了精心设计的目标上下文和触发集。通过对比学习对多对后门捷径进行正交化优化，从而将触发条件约束到一个参数子空间以提高匹配性。为了提高RAG对目标上下文的查全率，我们引入了知识图来构建结构化数据，以实现细粒度的硬匹配。此外，我们对LLMS中的后门场景进行了规范化，从攻击者和用户的角度分析了后门造成的真实危害，并进一步验证了上下文是否为越狱模型的有利工具。在真实性、语言理解和危害性方面的大量实验结果表明，TrojanRAG在保持对正常查询的检索能力的同时，表现出通用性威胁。



## **10. Information Leakage from Embedding in Large Language Models**

嵌入大型语言模型的信息泄露 cs.LG

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.11916v3) [paper-pdf](http://arxiv.org/pdf/2405.11916v3)

**Authors**: Zhipeng Wan, Anda Cheng, Yinggui Wang, Lei Wang

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns regarding data privacy. This study aims to investigate the potential for privacy invasion through input reconstruction attacks, in which a malicious model provider could potentially recover user inputs from embeddings. We first propose two base methods to reconstruct original texts from a model's hidden states. We find that these two methods are effective in attacking the embeddings from shallow layers, but their effectiveness decreases when attacking embeddings from deeper layers. To address this issue, we then present Embed Parrot, a Transformer-based method, to reconstruct input from embeddings in deep layers. Our analysis reveals that Embed Parrot effectively reconstructs original inputs from the hidden states of ChatGLM-6B and Llama2-7B, showcasing stable performance across various token lengths and data distributions. To mitigate the risk of privacy breaches, we introduce a defense mechanism to deter exploitation of the embedding reconstruction process. Our findings emphasize the importance of safeguarding user privacy in distributed learning systems and contribute valuable insights to enhance the security protocols within such environments.

摘要: 大型语言模型(LLM)的广泛采用引起了人们对数据隐私的担忧。这项研究旨在调查通过输入重构攻击来侵犯隐私的可能性，在这种攻击中，恶意模型提供者可能会从嵌入中恢复用户输入。我们首先提出了两种从模型的隐藏状态重构原始文本的基本方法。我们发现，这两种方法在攻击浅层嵌入时是有效的，但在攻击深层嵌入时，其有效性会降低。为了解决这个问题，我们提出了一种基于Transformer的嵌入Parrot方法来重建来自深层嵌入的输入。我们的分析表明，嵌入的鹦鹉有效地从ChatGLM-6B和Llama2-7B的隐藏状态重建原始输入，在不同的令牌长度和数据分布上表现出稳定的性能。为了减少隐私泄露的风险，我们引入了一种防御机制来阻止对嵌入重建过程的利用。我们的发现强调了在分布式学习系统中保护用户隐私的重要性，并为增强此类环境中的安全协议提供了有价值的见解。



## **11. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy**

不精确的遗忘需要更仔细的评估，以避免错误的隐私感 cs.LG

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2403.01218v3) [paper-pdf](http://arxiv.org/pdf/2403.01218v3)

**Authors**: Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, Nicolas Papernot

**Abstract**: The high cost of model training makes it increasingly desirable to develop techniques for unlearning. These techniques seek to remove the influence of a training example without having to retrain the model from scratch. Intuitively, once a model has unlearned, an adversary that interacts with the model should no longer be able to tell whether the unlearned example was included in the model's training set or not. In the privacy literature, this is known as membership inference. In this work, we discuss adaptations of Membership Inference Attacks (MIAs) to the setting of unlearning (leading to their "U-MIA" counterparts). We propose a categorization of existing U-MIAs into "population U-MIAs", where the same attacker is instantiated for all examples, and "per-example U-MIAs", where a dedicated attacker is instantiated for each example. We show that the latter category, wherein the attacker tailors its membership prediction to each example under attack, is significantly stronger. Indeed, our results show that the commonly used U-MIAs in the unlearning literature overestimate the privacy protection afforded by existing unlearning techniques on both vision and language models. Our investigation reveals a large variance in the vulnerability of different examples to per-example U-MIAs. In fact, several unlearning algorithms lead to a reduced vulnerability for some, but not all, examples that we wish to unlearn, at the expense of increasing it for other examples. Notably, we find that the privacy protection for the remaining training examples may worsen as a consequence of unlearning. We also discuss the fundamental difficulty of equally protecting all examples using existing unlearning schemes, due to the different rates at which examples are unlearned. We demonstrate that naive attempts at tailoring unlearning stopping criteria to different examples fail to alleviate these issues.

摘要: 模型训练的高昂成本使得开发忘却学习的技术变得越来越受欢迎。这些技术寻求消除训练示例的影响，而不必从头开始重新训练模型。直观地说，一旦模型取消学习，与该模型交互的对手应该不再能够判断未学习的示例是否包括在该模型的训练集中。在隐私文献中，这被称为成员关系推断。在这项工作中，我们讨论了成员关系推理攻击(MIA)对遗忘环境的适应(导致它们的U-MIA对应)。我们建议将现有的U-MIA分类为“群体U-MIA”，其中针对所有示例实例化相同的攻击者，以及“每示例U-MIA”，其中针对每个示例实例化一个专门的攻击者。我们表明，后一类，其中攻击者根据每个被攻击的例子定制其成员预测，明显更强。事实上，我们的结果表明，遗忘文献中常用的U-MIA高估了现有遗忘技术在视觉和语言模型上提供的隐私保护。我们的调查显示，不同示例对每个示例的U-MIA的脆弱性存在很大差异。事实上，几种忘记算法降低了我们希望忘记的一些(但不是所有)示例的脆弱性，但代价是增加了其他示例的脆弱性。值得注意的是，我们发现，由于遗忘，其余训练样本的隐私保护可能会恶化。我们还讨论了使用现有的遗忘方案平等地保护所有例子的基本困难，因为例子被遗忘的比率不同。我们证明，根据不同的例子调整遗忘停止标准的天真尝试无法缓解这些问题。



## **12. Generative AI and Large Language Models for Cyber Security: All Insights You Need**

网络安全的生成性人工智能和大型语言模型：您需要的所有见解 cs.CR

50 pages, 8 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12750v1) [paper-pdf](http://arxiv.org/pdf/2405.12750v1)

**Authors**: Mohamed Amine Ferrag, Fatima Alwahedi, Ammar Battah, Bilel Cherif, Abdechakour Mechri, Norbert Tihanyi

**Abstract**: This paper provides a comprehensive review of the future of cybersecurity through Generative AI and Large Language Models (LLMs). We explore LLM applications across various domains, including hardware design security, intrusion detection, software engineering, design verification, cyber threat intelligence, malware detection, and phishing detection. We present an overview of LLM evolution and its current state, focusing on advancements in models such as GPT-4, GPT-3.5, Mixtral-8x7B, BERT, Falcon2, and LLaMA. Our analysis extends to LLM vulnerabilities, such as prompt injection, insecure output handling, data poisoning, DDoS attacks, and adversarial instructions. We delve into mitigation strategies to protect these models, providing a comprehensive look at potential attack scenarios and prevention techniques. Furthermore, we evaluate the performance of 42 LLM models in cybersecurity knowledge and hardware security, highlighting their strengths and weaknesses. We thoroughly evaluate cybersecurity datasets for LLM training and testing, covering the lifecycle from data creation to usage and identifying gaps for future research. In addition, we review new strategies for leveraging LLMs, including techniques like Half-Quadratic Quantization (HQQ), Reinforcement Learning with Human Feedback (RLHF), Direct Preference Optimization (DPO), Quantized Low-Rank Adapters (QLoRA), and Retrieval-Augmented Generation (RAG). These insights aim to enhance real-time cybersecurity defenses and improve the sophistication of LLM applications in threat detection and response. Our paper provides a foundational understanding and strategic direction for integrating LLMs into future cybersecurity frameworks, emphasizing innovation and robust model deployment to safeguard against evolving cyber threats.

摘要: 本文通过生成式人工智能和大型语言模型(LLMS)对网络安全的未来进行了全面的回顾。我们探索了LLM在不同领域的应用，包括硬件设计安全、入侵检测、软件工程、设计验证、网络威胁情报、恶意软件检测和网络钓鱼检测。我们概述了LLM的演化和现状，重点介绍了GPT-4、GPT-3.5、Mixtral-8x7B、BERT、Falcon2和Llama等模型的进展。我们的分析扩展到LLM漏洞，如快速注入、不安全的输出处理、数据中毒、DDoS攻击和敌意指令。我们深入研究缓解策略以保护这些模型，提供对潜在攻击场景和预防技术的全面了解。此外，我们评估了42个LLM模型在网络安全知识和硬件安全方面的性能，突出了它们的优势和劣势。我们为LLM培训和测试彻底评估网络安全数据集，涵盖从数据创建到使用的整个生命周期，并为未来的研究确定差距。此外，我们还回顾了利用LLMS的新策略，包括半二次量化(HQQ)、带人反馈的强化学习(RLHF)、直接偏好优化(DPO)、量化低阶适配器(QLoRA)和检索增强生成(RAG)。这些见解旨在增强实时网络安全防御，并提高LLM应用程序在威胁检测和响应方面的复杂性。我们的论文为将低成本管理系统整合到未来的网络安全框架中提供了一个基础性的理解和战略方向，强调创新和稳健的模型部署，以防范不断演变的网络威胁。



## **13. Single Image Unlearning: Efficient Machine Unlearning in Multimodal Large Language Models**

单图像去学习：多模式大型语言模型中的高效机器去学习 cs.CV

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12523v1) [paper-pdf](http://arxiv.org/pdf/2405.12523v1)

**Authors**: Jiaqi Li, Qianshan Wei, Chuanyi Zhang, Guilin Qi, Miaozeng Du, Yongrui Chen, Sheng Bi

**Abstract**: Machine unlearning empowers individuals with the `right to be forgotten' by removing their private or sensitive information encoded in machine learning models. However, it remains uncertain whether MU can be effectively applied to Multimodal Large Language Models (MLLMs), particularly in scenarios of forgetting the leaked visual data of concepts. To overcome the challenge, we propose an efficient method, Single Image Unlearning (SIU), to unlearn the visual recognition of a concept by fine-tuning a single associated image for few steps. SIU consists of two key aspects: (i) Constructing Multifaceted fine-tuning data. We introduce four targets, based on which we construct fine-tuning data for the concepts to be forgotten; (ii) Jointly training loss. To synchronously forget the visual recognition of concepts and preserve the utility of MLLMs, we fine-tune MLLMs through a novel Dual Masked KL-divergence Loss combined with Cross Entropy loss. Alongside our method, we establish MMUBench, a new benchmark for MU in MLLMs and introduce a collection of metrics for its evaluation. Experimental results on MMUBench show that SIU completely surpasses the performance of existing methods. Furthermore, we surprisingly find that SIU can avoid invasive membership inference attacks and jailbreak attacks. To the best of our knowledge, we are the first to explore MU in MLLMs. We will release the code and benchmark in the near future.

摘要: 机器遗忘通过删除编码在机器学习模型中的私人或敏感信息，使个人有被遗忘的权利。然而，MU是否能有效地应用于多通道大语言模型，特别是在忘记概念的泄露的视觉数据的情况下，仍然是不确定的。为了克服这一挑战，我们提出了一种有效的方法，单图像忘却学习(SIU)，通过对单个关联图像进行几个步骤的微调来消除对概念的视觉识别。SIU包括两个关键方面：(I)构建多方面的微调数据。我们引入了四个目标，在此基础上构建了精调的数据，以使概念被遗忘；(Ii)联合训练损失。为了同步忘记对概念的视觉识别，同时保持MLLS的实用性，我们通过一种新颖的双屏蔽KL-发散损失和交叉熵损失来微调MLLMS。除了我们的方法之外，我们还建立了MLLMS中MU的一个新的基准MMUBENCH，并引入了一组用于评估它的度量。在MMUBENCH上的实验结果表明，SIU的性能完全优于现有方法。此外，我们惊讶地发现，SIU可以避免入侵性的成员推理攻击和越狱攻击。据我们所知，我们是第一个探索MLLMS中的MU的人。我们将在不久的将来发布代码和基准测试。



## **14. GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation**

GPT-4通过自我解释以近乎完美的成功越狱 cs.CR

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.13077v1) [paper-pdf](http://arxiv.org/pdf/2405.13077v1)

**Authors**: Govind Ramesh, Yao Dou, Wei Xu

**Abstract**: Research on jailbreaking has been valuable for testing and understanding the safety and security issues of large language models (LLMs). In this paper, we introduce Iterative Refinement Induced Self-Jailbreak (IRIS), a novel approach that leverages the reflective capabilities of LLMs for jailbreaking with only black-box access. Unlike previous methods, IRIS simplifies the jailbreaking process by using a single model as both the attacker and target. This method first iteratively refines adversarial prompts through self-explanation, which is crucial for ensuring that even well-aligned LLMs obey adversarial instructions. IRIS then rates and enhances the output given the refined prompt to increase its harmfulness. We find IRIS achieves jailbreak success rates of 98% on GPT-4 and 92% on GPT-4 Turbo in under 7 queries. It significantly outperforms prior approaches in automatic, black-box and interpretable jailbreaking, while requiring substantially fewer queries, thereby establishing a new standard for interpretable jailbreaking methods.

摘要: 越狱研究对于测试和理解大型语言模型(LLM)的安全和安保问题具有重要价值。在本文中，我们介绍了迭代精化诱导的自越狱(IRIS)，这是一种新的方法，它利用LLMS的反射能力来实现只访问黑盒的越狱。与以前的方法不同，IRIS通过将单一模型用作攻击者和目标来简化越狱过程。这种方法首先通过自我解释迭代地精炼对抗性提示，这对于确保即使是排列良好的LLM也遵守对抗性指令至关重要。然后，虹膜给出精致的提示，对产量进行评级并提高产量，以增加其危害性。我们发现IRIS在GPT-4上的越狱成功率为98%，在GPT-4 Turbo上的越狱成功率为92%。它在自动、黑盒和可解释越狱方面显著优于现有方法，同时需要的查询大大减少，从而建立了可解释越狱方法的新标准。



## **15. Lockpicking LLMs: A Logit-Based Jailbreak Using Token-level Manipulation**

撬锁LLM：使用代币级操纵的基于日志的越狱 cs.CR

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.13068v1) [paper-pdf](http://arxiv.org/pdf/2405.13068v1)

**Authors**: Yuxi Li, Yi Liu, Yuekang Li, Ling Shi, Gelei Deng, Shengquan Chen, Kailong Wang

**Abstract**: Large language models (LLMs) have transformed the field of natural language processing, but they remain susceptible to jailbreaking attacks that exploit their capabilities to generate unintended and potentially harmful content. Existing token-level jailbreaking techniques, while effective, face scalability and efficiency challenges, especially as models undergo frequent updates and incorporate advanced defensive measures. In this paper, we introduce JailMine, an innovative token-level manipulation approach that addresses these limitations effectively. JailMine employs an automated "mining" process to elicit malicious responses from LLMs by strategically selecting affirmative outputs and iteratively reducing the likelihood of rejection. Through rigorous testing across multiple well-known LLMs and datasets, we demonstrate JailMine's effectiveness and efficiency, achieving a significant average reduction of 86% in time consumed while maintaining high success rates averaging 95%, even in the face of evolving defensive strategies. Our work contributes to the ongoing effort to assess and mitigate the vulnerability of LLMs to jailbreaking attacks, underscoring the importance of continued vigilance and proactive measures to enhance the security and reliability of these powerful language models.

摘要: 大型语言模型(LLM)已经改变了自然语言处理领域，但它们仍然容易受到越狱攻击，这些攻击利用它们的能力生成意外的和潜在的有害内容。现有的令牌级越狱技术虽然有效，但面临可伸缩性和效率的挑战，特别是在模型经历频繁更新和采用先进防御措施的情况下。在本文中，我们介绍了Jailmy，一种创新的令牌级操作方法，有效地解决了这些限制。Jailmine使用一个自动化的“挖掘”过程，通过战略性地选择肯定的输出并反复降低拒绝的可能性，来引发来自LLMS的恶意响应。通过对多个知名LLM和数据集的严格测试，我们展示了Jailmine的有效性和效率，实现了平均86%的时间消耗显著减少，同时保持了平均95%的高成功率，即使面对不断变化的防御策略。我们的工作有助于评估和减轻LLMS在越狱攻击中的脆弱性，强调了继续保持警惕和采取积极措施以增强这些强大语言模型的安全性和可靠性的重要性。



## **16. Special Characters Attack: Toward Scalable Training Data Extraction From Large Language Models**

特殊字符攻击：从大型语言模型中提取可扩展的训练数据 cs.CR

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.05990v2) [paper-pdf](http://arxiv.org/pdf/2405.05990v2)

**Authors**: Yang Bai, Ge Pei, Jindong Gu, Yong Yang, Xingjun Ma

**Abstract**: Large language models (LLMs) have achieved remarkable performance on a wide range of tasks. However, recent studies have shown that LLMs can memorize training data and simple repeated tokens can trick the model to leak the data. In this paper, we take a step further and show that certain special characters or their combinations with English letters are stronger memory triggers, leading to more severe data leakage. The intuition is that, since LLMs are trained with massive data that contains a substantial amount of special characters (e.g. structural symbols {, } of JSON files, and @, # in emails and online posts), the model may memorize the co-occurrence between these special characters and the raw texts. This motivates us to propose a simple but effective Special Characters Attack (SCA) to induce training data leakage. Our experiments verify the high effectiveness of SCA against state-of-the-art LLMs: they can leak diverse training data, such as code corpus, web pages, and personally identifiable information, and sometimes generate non-stop outputs as a byproduct. We further show that the composition of the training data corpus can be revealed by inspecting the leaked data -- one crucial piece of information for pre-training high-performance LLMs. Our work can help understand the sensitivity of LLMs to special characters and identify potential areas for improvement.

摘要: 大型语言模型(LLM)在广泛的任务中取得了显著的性能。然而，最近的研究表明，LLMS可以记忆训练数据，简单的重复令牌可以诱使模型泄漏数据。在这篇文章中，我们进一步表明，某些特殊字符或它们与英语字母的组合是更强的记忆触发，导致更严重的数据泄漏。直观的是，由于LLM是用包含大量特殊字符(例如JSON文件的结构符号{，}，以及电子邮件和在线帖子中的@，#)的海量数据训练的，因此该模型可能会记住这些特殊字符与原始文本之间的共现。这促使我们提出了一种简单而有效的特殊字符攻击(SCA)来诱导训练数据泄漏。我们的实验验证了SCA相对于最先进的LLMS的高效性：它们可以泄露各种训练数据，如代码语料库、网页和个人身份信息，有时还会生成不间断的输出作为副产品。我们进一步表明，通过检查泄漏的数据可以揭示训练数据集的组成--这是训练前高性能LLMS的关键信息之一。我们的工作有助于理解LLMS对特殊字符的敏感度，并确定潜在的改进领域。



## **17. Data Contamination Calibration for Black-box LLMs**

黑匣子LLM的数据污染校准 cs.LG

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.11930v1) [paper-pdf](http://arxiv.org/pdf/2405.11930v1)

**Authors**: Wentao Ye, Jiaqi Hu, Liyao Li, Haobo Wang, Gang Chen, Junbo Zhao

**Abstract**: The rapid advancements of Large Language Models (LLMs) tightly associate with the expansion of the training data size. However, the unchecked ultra-large-scale training sets introduce a series of potential risks like data contamination, i.e. the benchmark data is used for training. In this work, we propose a holistic method named Polarized Augment Calibration (PAC) along with a new to-be-released dataset to detect the contaminated data and diminish the contamination effect. PAC extends the popular MIA (Membership Inference Attack) -- from machine learning community -- by forming a more global target at detecting training data to Clarify invisible training data. As a pioneering work, PAC is very much plug-and-play that can be integrated with most (if not all) current white- and black-box LLMs. By extensive experiments, PAC outperforms existing methods by at least 4.5%, towards data contamination detection on more 4 dataset formats, with more than 10 base LLMs. Besides, our application in real-world scenarios highlights the prominent presence of contamination and related issues.

摘要: 大型语言模型(LLM)的快速发展与训练数据规模的扩大密切相关。然而，未经核查的超大规模训练集带来了一系列潜在的风险，如数据污染，即基准数据被用于训练。在这项工作中，我们提出了一种称为极化增强校准(PAC)的整体方法以及一个新的即将发布的数据集来检测污染数据并减少污染影响。PAC扩展了流行的MIA(成员关系推断攻击)--来自机器学习社区--通过形成一个更全局的目标来检测训练数据，以澄清看不见的训练数据。作为一项开创性的工作，PAC是非常即插即用的，可以与大多数(如果不是全部)当前的白盒和黑盒LLM集成。通过大量实验，在4种以上的数据集格式、10个以上的基本最小似然模型上，PAC在数据污染检测方面的性能至少比现有方法高4.5%。此外，我们在现实世界场景中的应用突出了污染和相关问题的突出存在。



## **18. A Semantic Invariant Robust Watermark for Large Language Models**

大型语言模型的语义不变鲁棒水印 cs.CR

ICLR2024, 21 pages, 10 figures, 6 tables

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2310.06356v3) [paper-pdf](http://arxiv.org/pdf/2310.06356v3)

**Authors**: Aiwei Liu, Leyi Pan, Xuming Hu, Shiao Meng, Lijie Wen

**Abstract**: Watermark algorithms for large language models (LLMs) have achieved extremely high accuracy in detecting text generated by LLMs. Such algorithms typically involve adding extra watermark logits to the LLM's logits at each generation step. However, prior algorithms face a trade-off between attack robustness and security robustness. This is because the watermark logits for a token are determined by a certain number of preceding tokens; a small number leads to low security robustness, while a large number results in insufficient attack robustness. In this work, we propose a semantic invariant watermarking method for LLMs that provides both attack robustness and security robustness. The watermark logits in our work are determined by the semantics of all preceding tokens. Specifically, we utilize another embedding LLM to generate semantic embeddings for all preceding tokens, and then these semantic embeddings are transformed into the watermark logits through our trained watermark model. Subsequent analyses and experiments demonstrated the attack robustness of our method in semantically invariant settings: synonym substitution and text paraphrasing settings. Finally, we also show that our watermark possesses adequate security robustness. Our code and data are available at \href{https://github.com/THU-BPM/Robust_Watermark}{https://github.com/THU-BPM/Robust\_Watermark}. Additionally, our algorithm could also be accessed through MarkLLM \citep{pan2024markllm} \footnote{https://github.com/THU-BPM/MarkLLM}.

摘要: 针对大语言模型的水印算法在检测大语言模型生成的文本方面取得了极高的准确率。这类算法通常涉及在每个生成步骤向LLM的日志添加额外的水印日志。然而，现有的算法面临着攻击健壮性和安全健壮性之间的权衡。这是因为令牌的水印登录由一定数量的先前令牌确定；较小的数字会导致较低的安全稳健性，而较大的数字会导致攻击稳健性不足。在这项工作中，我们提出了一种既具有攻击健壮性又具有安全健壮性的LLMS语义不变水印方法。我们工作中的水印日志是由前面所有令牌的语义确定的。具体地说，我们利用另一种嵌入LLM为所有前面的令牌生成语义嵌入，然后通过我们训练的水印模型将这些语义嵌入转换成水印日志。随后的分析和实验证明了该方法在同义词替换和文本释义等语义不变环境下的攻击健壮性。最后，我们还证明了我们的水印具有足够的安全稳健性。我们的代码和数据可在\href{https://github.com/THU-BPM/Robust_Watermark}{https://github.com/THU-BPM/Robust\_Watermark}.上获得此外，我们的算法还可以通过MarkLLm\Citep\footnote{https://github.com/THU-BPM/MarkLLM}.访问



## **19. Detoxifying Large Language Models via Knowledge Editing**

通过知识编辑消除大型语言模型的神秘性 cs.CL

ACL 2024. Project website: https://zjunlp.github.io/project/SafeEdit  Benchmark: https://huggingface.co/datasets/zjunlp/SafeEdit

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2403.14472v4) [paper-pdf](http://arxiv.org/pdf/2403.14472v4)

**Authors**: Mengru Wang, Ningyu Zhang, Ziwen Xu, Zekun Xi, Shumin Deng, Yunzhi Yao, Qishen Zhang, Linyi Yang, Jindong Wang, Huajun Chen

**Abstract**: This paper investigates using knowledge editing techniques to detoxify Large Language Models (LLMs). We construct a benchmark, SafeEdit, which covers nine unsafe categories with various powerful attack prompts and equips comprehensive metrics for systematic evaluation. We conduct experiments with several knowledge editing approaches, indicating that knowledge editing has the potential to detoxify LLMs with a limited impact on general performance efficiently. Then, we propose a simple yet effective baseline, dubbed Detoxifying with Intraoperative Neural Monitoring (DINM), to diminish the toxicity of LLMs within a few tuning steps via only one instance. We further provide an in-depth analysis of the internal mechanism for various detoxifying approaches, demonstrating that previous methods like SFT and DPO may merely suppress the activations of toxic parameters, while DINM mitigates the toxicity of the toxic parameters to a certain extent, making permanent adjustments. We hope that these insights could shed light on future work of developing detoxifying approaches and the underlying knowledge mechanisms of LLMs. Code and benchmark are available at https://github.com/zjunlp/EasyEdit.

摘要: 本文研究了利用知识编辑技术对大型语言模型进行去毒处理。我们构建了一个涵盖9个不安全类别、具有各种强大的攻击提示的基准SafeEdit，并配备了全面的度量来进行系统评估。我们用几种知识编辑方法进行了实验，表明知识编辑有可能在对一般性能影响有限的情况下有效地解毒LLM。然后，我们提出了一个简单而有效的基线，称为术中神经监测解毒(DINM)，仅通过一个实例在几个调整步骤内降低LLMS的毒性。我们进一步深入分析了各种解毒方法的内在机制，证明了以前的方法如SFT和DPO可能只是抑制了毒性参数的激活，而DINM在一定程度上减轻了毒性参数的毒性，做出了永久性的调整。我们希望这些洞察力能够为未来开发戒毒方法的工作和LLMS的潜在知识机制提供帮助。代码和基准测试可在https://github.com/zjunlp/EasyEdit.上获得



## **20. Measuring Impacts of Poisoning on Model Parameters and Embeddings for Large Language Models of Code**

测量中毒对代码大型语言模型的模型参数和嵌入的影响 cs.SE

This work has been accepted at the 1st ACM International Conference  on AI-powered Software (AIware), co-located with the ACM International  Conference on the Foundations of Software Engineering (FSE) 2024, Porto de  Galinhas, Brazil. arXiv admin note: substantial text overlap with  arXiv:2402.12936

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2405.11466v1) [paper-pdf](http://arxiv.org/pdf/2405.11466v1)

**Authors**: Aftab Hussain, Md Rafiqul Islam Rabin, Mohammad Amin Alipour

**Abstract**: Large language models (LLMs) have revolutionized software development practices, yet concerns about their safety have arisen, particularly regarding hidden backdoors, aka trojans. Backdoor attacks involve the insertion of triggers into training data, allowing attackers to manipulate the behavior of the model maliciously. In this paper, we focus on analyzing the model parameters to detect potential backdoor signals in code models. Specifically, we examine attention weights and biases, and context embeddings of the clean and poisoned CodeBERT and CodeT5 models. Our results suggest noticeable patterns in context embeddings of poisoned samples for both the poisoned models; however, attention weights and biases do not show any significant differences. This work contributes to ongoing efforts in white-box detection of backdoor signals in LLMs of code through the analysis of parameters and embeddings.

摘要: 大型语言模型（LLM）彻底改变了软件开发实践，但对其安全性的担忧也出现了，特别是关于隐藏后门（又名特洛伊木马）的担忧。后门攻击涉及将触发器插入训练数据，允许攻击者恶意操纵模型的行为。本文重点分析模型参数以检测代码模型中潜在的后门信号。具体来说，我们检查了注意力权重和偏差，以及干净和有毒的CodeBRT和CodeT 5模型的上下文嵌入。我们的结果表明，两种中毒模型的中毒样本的上下文嵌入中都存在明显的模式;然而，注意力权重和偏差并没有表现出任何显着差异。这项工作有助于通过分析参数和嵌入来对代码LLM中后门信号进行白盒检测的持续努力。



## **21. Revisiting the Robust Generalization of Adversarial Prompt Tuning**

重新审视对抗即时调整的稳健概括 cs.CV

**SubmitDate**: 2024-05-18    [abs](http://arxiv.org/abs/2405.11154v1) [paper-pdf](http://arxiv.org/pdf/2405.11154v1)

**Authors**: Fan Yang, Mingxuan Xia, Sangzhou Xia, Chicheng Ma, Hui Hui

**Abstract**: Understanding the vulnerability of large-scale pre-trained vision-language models like CLIP against adversarial attacks is key to ensuring zero-shot generalization capacity on various downstream tasks. State-of-the-art defense mechanisms generally adopt prompt learning strategies for adversarial fine-tuning to improve the adversarial robustness of the pre-trained model while keeping the efficiency of adapting to downstream tasks. Such a setup leads to the problem of over-fitting which impedes further improvement of the model's generalization capacity on both clean and adversarial examples. In this work, we propose an adaptive Consistency-guided Adversarial Prompt Tuning (i.e., CAPT) framework that utilizes multi-modal prompt learning to enhance the alignment of image and text features for adversarial examples and leverage the strong generalization of pre-trained CLIP to guide the model-enhancing its robust generalization on adversarial examples while maintaining its accuracy on clean ones. We also design a novel adaptive consistency objective function to balance the consistency of adversarial inputs and clean inputs between the fine-tuning model and the pre-trained model. We conduct extensive experiments across 14 datasets and 4 data sparsity schemes (from 1-shot to full training data settings) to show the superiority of CAPT over other state-of-the-art adaption methods. CAPT demonstrated excellent performance in terms of the in-distribution performance and the generalization under input distribution shift and across datasets.

摘要: 了解像CLIP这样的大规模预先训练的视觉语言模型对对抗攻击的脆弱性是确保对各种下游任务的零命中泛化能力的关键。最新的防御机制一般采用对抗性微调的快速学习策略，在保持适应下游任务的效率的同时，提高预先训练模型的对抗性健壮性。这样的设置导致了过度拟合的问题，这阻碍了模型在干净和对抗性例子上的推广能力的进一步提高。在这项工作中，我们提出了一种自适应一致性制导的对抗性提示调整(CAPT)框架，该框架利用多模式提示学习来增强对抗性示例的图文特征对齐，并利用预先训练的CLIP的强泛化来指导模型--增强了对对抗性示例的健壮性，同时保持了对干净示例的准确性。我们还设计了一种新的自适应一致性目标函数，以平衡微调模型和预训练模型之间的对抗性输入和干净输入的一致性。我们在14个数据集和4个数据稀疏方案(从单镜头到全训练数据设置)上进行了广泛的实验，以显示CAPT相对于其他最先进的自适应方法的优越性。CAPT在分布内性能和在输入分布漂移和跨数据集情况下的泛化方面表现出了优异的性能。



## **22. A Comprehensive Study of Jailbreak Attack versus Defense for Large Language Models**

大型语言模型越狱攻击与防御的综合研究 cs.CR

18 pages, 9 figures, Accepted in ACL 2024

**SubmitDate**: 2024-05-17    [abs](http://arxiv.org/abs/2402.13457v2) [paper-pdf](http://arxiv.org/pdf/2402.13457v2)

**Authors**: Zihao Xu, Yi Liu, Gelei Deng, Yuekang Li, Stjepan Picek

**Abstract**: Large Language Models (LLMS) have increasingly become central to generating content with potential societal impacts. Notably, these models have demonstrated capabilities for generating content that could be deemed harmful. To mitigate these risks, researchers have adopted safety training techniques to align model outputs with societal values to curb the generation of malicious content. However, the phenomenon of "jailbreaking", where carefully crafted prompts elicit harmful responses from models, persists as a significant challenge. This research conducts a comprehensive analysis of existing studies on jailbreaking LLMs and their defense techniques. We meticulously investigate nine attack techniques and seven defense techniques applied across three distinct language models: Vicuna, LLama, and GPT-3.5 Turbo. We aim to evaluate the effectiveness of these attack and defense techniques. Our findings reveal that existing white-box attacks underperform compared to universal techniques and that including special tokens in the input significantly affects the likelihood of successful attacks. This research highlights the need to concentrate on the security facets of LLMs. Additionally, we contribute to the field by releasing our datasets and testing framework, aiming to foster further research into LLM security. We believe these contributions will facilitate the exploration of security measures within this domain.

摘要: 大型语言模型(LLM)越来越多地成为产生潜在社会影响的内容的核心。值得注意的是，这些模型展示了生成可能被认为有害的内容的能力。为了降低这些风险，研究人员采用了安全培训技术，使模型输出与社会价值观保持一致，以遏制恶意内容的生成。然而，“越狱”现象仍然是一个重大挑战。“越狱”是一种精心设计的行为，会招致模特的有害反应。本研究对已有的越狱LLMS及其防御技术的研究进行了全面的分析。我们仔细研究了在三种不同的语言模型中应用的九种攻击技术和七种防御技术：羊驼、骆驼和GPT-3.5Turbo。我们的目标是评估这些攻防技术的有效性。我们的发现表明，与通用技术相比，现有的白盒攻击表现不佳，并且在输入中包括特殊令牌显着影响攻击成功的可能性。这项研究强调了专注于低成本管理的安全方面的必要性。此外，我们通过发布我们的数据集和测试框架来为该领域做出贡献，旨在促进对LLM安全的进一步研究。我们相信，这些贡献将有助于探索这一领域内的安全措施。



## **23. Safeguarding Vision-Language Models Against Patched Visual Prompt Injectors**

保护视觉语言模型免受修补视觉提示注入器的影响 cs.CV

15 pages

**SubmitDate**: 2024-05-17    [abs](http://arxiv.org/abs/2405.10529v1) [paper-pdf](http://arxiv.org/pdf/2405.10529v1)

**Authors**: Jiachen Sun, Changsheng Wang, Jiongxiao Wang, Yiwei Zhang, Chaowei Xiao

**Abstract**: Large language models have become increasingly prominent, also signaling a shift towards multimodality as the next frontier in artificial intelligence, where their embeddings are harnessed as prompts to generate textual content. Vision-language models (VLMs) stand at the forefront of this advancement, offering innovative ways to combine visual and textual data for enhanced understanding and interaction. However, this integration also enlarges the attack surface. Patch-based adversarial attack is considered the most realistic threat model in physical vision applications, as demonstrated in many existing literature. In this paper, we propose to address patched visual prompt injection, where adversaries exploit adversarial patches to generate target content in VLMs. Our investigation reveals that patched adversarial prompts exhibit sensitivity to pixel-wise randomization, a trait that remains robust even against adaptive attacks designed to counteract such defenses. Leveraging this insight, we introduce SmoothVLM, a defense mechanism rooted in smoothing techniques, specifically tailored to protect VLMs from the threat of patched visual prompt injectors. Our framework significantly lowers the attack success rate to a range between 0% and 5.0% on two leading VLMs, while achieving around 67.3% to 95.0% context recovery of the benign images, demonstrating a balance between security and usability.

摘要: 大型语言模型已变得越来越突出，这也标志着向多通道的转变，成为人工智能的下一个前沿，在人工智能中，它们的嵌入被用作生成文本内容的提示。视觉语言模型(VLM)站在这一进步的前沿，提供了将视觉和文本数据相结合的创新方法，以增强理解和交互。然而，这种整合也扩大了攻击面。基于补丁的对抗性攻击被认为是物理视觉应用中最现实的威胁模型，许多现有的文献都证明了这一点。在本文中，我们建议解决补丁视觉提示注入，即攻击者利用敌意补丁来生成VLMS中的目标内容。我们的调查显示，打补丁的对抗性提示显示出对像素随机化的敏感性，这一特征即使在旨在对抗此类防御的适应性攻击中也保持健壮。利用这一见解，我们推出了SmoothVLM，这是一种植根于平滑技术的防御机制，专门为保护VLM免受修补的视觉提示注入器的威胁而量身定做。我们的框架将攻击成功率显著降低到了0%到5.0%之间，同时实现了良性映像的67.3%到95.0%的上下文恢复，展示了安全性和可用性之间的平衡。



## **24. Large Language Models in Wireless Application Design: In-Context Learning-enhanced Automatic Network Intrusion Detection**

无线应用设计中的大型语言模型：上下文学习增强的自动网络入侵检测 cs.LG

**SubmitDate**: 2024-05-17    [abs](http://arxiv.org/abs/2405.11002v1) [paper-pdf](http://arxiv.org/pdf/2405.11002v1)

**Authors**: Han Zhang, Akram Bin Sediq, Ali Afana, Melike Erol-Kantarci

**Abstract**: Large language models (LLMs), especially generative pre-trained transformers (GPTs), have recently demonstrated outstanding ability in information comprehension and problem-solving. This has motivated many studies in applying LLMs to wireless communication networks. In this paper, we propose a pre-trained LLM-empowered framework to perform fully automatic network intrusion detection. Three in-context learning methods are designed and compared to enhance the performance of LLMs. With experiments on a real network intrusion detection dataset, in-context learning proves to be highly beneficial in improving the task processing performance in a way that no further training or fine-tuning of LLMs is required. We show that for GPT-4, testing accuracy and F1-Score can be improved by 90%. Moreover, pre-trained LLMs demonstrate big potential in performing wireless communication-related tasks. Specifically, the proposed framework can reach an accuracy and F1-Score of over 95% on different types of attacks with GPT-4 using only 10 in-context learning examples.

摘要: 大语言模型(LLM)，特别是产生式预训练转换器(GPT)，近年来在信息理解和问题解决方面表现出了突出的能力。这推动了许多将LLMS应用于无线通信网络的研究。在本文中，我们提出了一个预先训练的LLM授权框架来执行全自动的网络入侵检测。为了提高LLMS的性能，设计并比较了三种上下文学习方法。通过在真实网络入侵检测数据集上的实验，上下文中学习被证明是非常有益的，在不需要进一步训练或微调LLMS的情况下，可以提高任务处理性能。结果表明，对于GPT-4，测试准确率和F1-Score可以提高90%。此外，预先训练的LLM在执行与无线通信相关的任务方面显示出巨大的潜力。具体地说，该框架在使用GPT-4进行不同类型的攻击时，仅使用10个上下文学习示例，就可以达到95%以上的准确率和F1-Score。



## **25. Keep It Private: Unsupervised Privatization of Online Text**

保持隐私：在线文本的无监督私有化 cs.CL

17 pages, 6 figures

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.10260v1) [paper-pdf](http://arxiv.org/pdf/2405.10260v1)

**Authors**: Calvin Bao, Marine Carpuat

**Abstract**: Authorship obfuscation techniques hold the promise of helping people protect their privacy in online communications by automatically rewriting text to hide the identity of the original author. However, obfuscation has been evaluated in narrow settings in the NLP literature and has primarily been addressed with superficial edit operations that can lead to unnatural outputs. In this work, we introduce an automatic text privatization framework that fine-tunes a large language model via reinforcement learning to produce rewrites that balance soundness, sense, and privacy. We evaluate it extensively on a large-scale test set of English Reddit posts by 68k authors composed of short-medium length texts. We study how the performance changes among evaluative conditions including authorial profile length and authorship detection strategy. Our method maintains high text quality according to both automated metrics and human evaluation, and successfully evades several automated authorship attacks.

摘要: 作者身份混淆技术有望通过自动重写文本以隐藏原作者的身份来帮助人们在在线通信中保护自己的隐私。然而，在NLP文献中，模糊是在狭窄的环境中进行评估的，并且主要通过可能导致不自然输出的肤浅编辑操作来解决的。在这项工作中，我们引入了一个自动文本私有化框架，该框架通过强化学习微调大型语言模型，以产生平衡合理性、意义和隐私的重写。我们在由68，000位作者组成的Reddit英语帖子的大规模测试集上对其进行了广泛评估，这些帖子由中短文本组成。我们研究了作者个人资料长度和作者身份检测策略等评价条件之间的性能如何变化。我们的方法根据自动化指标和人工评估保持高文本质量，并成功规避了几次自动作者攻击。



## **26. Protecting Your LLMs with Information Bottleneck**

通过信息瓶颈保护您的LLC cs.CL

23 pages, 7 figures, 8 tables

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2404.13968v2) [paper-pdf](http://arxiv.org/pdf/2404.13968v2)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.

摘要: 大型语言模型的出现给自然语言处理领域带来了革命性的变化，但它们可能会受到攻击，产生有害的内容。尽管努力在道德上调整LLM，但这些往往是脆弱的，可以通过优化或手动对抗性提示通过越狱攻击来绕过。为了解决这个问题，我们引入了信息瓶颈保护器(IBProtector)，这是一种基于信息瓶颈原理的防御机制，我们修改了目标以避免琐碎的解决方案。IBProtector有选择地压缩和干扰提示，由一个轻量级和可训练的提取程序促进，只保留目标LLMS的基本信息，以响应预期的答案。此外，我们还进一步考虑了梯度不可见的情况，以与任何LLM相容。我们的经验评估表明，在不过度影响响应质量或推理速度的情况下，IBProtector在缓解越狱企图方面优于现有的防御方法。它对各种攻击方法和目标LLM的有效性和适应性突显了IBProtector作为一种新型、可转移的防御系统的潜力，无需修改底层模型即可增强LLM的安全性。



## **27. Adversarial Robustness for Visual Grounding of Multimodal Large Language Models**

多模式大型语言模型视觉基础的对抗鲁棒性 cs.CV

ICLR 2024 Workshop on Reliable and Responsible Foundation Models

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09981v1) [paper-pdf](http://arxiv.org/pdf/2405.09981v1)

**Authors**: Kuofeng Gao, Yang Bai, Jiawang Bai, Yong Yang, Shu-Tao Xia

**Abstract**: Multi-modal Large Language Models (MLLMs) have recently achieved enhanced performance across various vision-language tasks including visual grounding capabilities. However, the adversarial robustness of visual grounding remains unexplored in MLLMs. To fill this gap, we use referring expression comprehension (REC) as an example task in visual grounding and propose three adversarial attack paradigms as follows. Firstly, untargeted adversarial attacks induce MLLMs to generate incorrect bounding boxes for each object. Besides, exclusive targeted adversarial attacks cause all generated outputs to the same target bounding box. In addition, permuted targeted adversarial attacks aim to permute all bounding boxes among different objects within a single image. Extensive experiments demonstrate that the proposed methods can successfully attack visual grounding capabilities of MLLMs. Our methods not only provide a new perspective for designing novel attacks but also serve as a strong baseline for improving the adversarial robustness for visual grounding of MLLMs.

摘要: 多模式大型语言模型(MLLM)最近在包括视觉基础能力在内的各种视觉语言任务中获得了增强的性能。然而，在最大似然最小二乘法中，视觉接地的对抗稳健性仍未被探索。为了填补这一空白，我们使用指称表达理解(REC)作为视觉基础的示例任务，并提出了以下三种对抗性攻击范式。首先，无针对性的对抗性攻击会导致MLLMS为每个对象生成错误的包围盒。此外，排他性定向对抗性攻击会导致所有生成的输出都指向相同的目标边界框。此外，置换定向对抗性攻击旨在置换单个图像中不同对象之间的所有包围盒。大量实验表明，所提出的方法能够成功地攻击MLLMS的视觉接地能力。我们的方法不仅为设计新的攻击提供了新的视角，而且为提高MLLMS视觉接地的对抗性稳健性提供了强有力的基线。



## **28. Transfer Learning in Pre-Trained Large Language Models for Malware Detection Based on System Calls**

预训练大型语言模型中的迁移学习用于基于系统调用的恶意软件检测 cs.CR

Submitted to IEEE MILCOM 2024

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09318v1) [paper-pdf](http://arxiv.org/pdf/2405.09318v1)

**Authors**: Pedro Miguel Sánchez Sánchez, Alberto Huertas Celdrán, Gérôme Bovet, Gregorio Martínez Pérez

**Abstract**: In the current cybersecurity landscape, protecting military devices such as communication and battlefield management systems against sophisticated cyber attacks is crucial. Malware exploits vulnerabilities through stealth methods, often evading traditional detection mechanisms such as software signatures. The application of ML/DL in vulnerability detection has been extensively explored in the literature. However, current ML/DL vulnerability detection methods struggle with understanding the context and intent behind complex attacks. Integrating large language models (LLMs) with system call analysis offers a promising approach to enhance malware detection. This work presents a novel framework leveraging LLMs to classify malware based on system call data. The framework uses transfer learning to adapt pre-trained LLMs for malware detection. By retraining LLMs on a dataset of benign and malicious system calls, the models are refined to detect signs of malware activity. Experiments with a dataset of over 1TB of system calls demonstrate that models with larger context sizes, such as BigBird and Longformer, achieve superior accuracy and F1-Score of approximately 0.86. The results highlight the importance of context size in improving detection rates and underscore the trade-offs between computational complexity and performance. This approach shows significant potential for real-time detection in high-stakes environments, offering a robust solution to evolving cyber threats.

摘要: 在当前的网络安全格局中，保护通信和战场管理系统等军事设备免受复杂的网络攻击至关重要。恶意软件通过秘密方式利用漏洞，通常会避开软件签名等传统检测机制。ML/DL在漏洞检测中的应用已经在文献中得到了广泛的探索。然而，当前的ML/DL漏洞检测方法难以理解复杂攻击背后的背景和意图。将大型语言模型(LLM)与系统调用分析相结合，为增强恶意软件检测提供了一种很有前途的方法。该工作提出了一种新的框架，利用LLMS根据系统调用数据对恶意软件进行分类。该框架使用迁移学习来调整预先训练的LLM以检测恶意软件。通过在良性和恶意系统调用的数据集上重新训练LLM，模型被改进以检测恶意软件活动的迹象。使用超过1TB的系统调用数据集进行的实验表明，具有较大上下文大小的模型(如BigBird和LongFormor)具有出色的准确率和大约0.86的F1得分。结果强调了上下文大小在提高检测率方面的重要性，并强调了计算复杂性和性能之间的权衡。这种方法显示出在高风险环境中进行实时检测的巨大潜力，为不断发展的网络威胁提供了强大的解决方案。



## **29. "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models**

“立即做任何事情”：描述和评估大型语言模型上的In-The-Wild越狱预言 cs.CR

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2308.03825v2) [paper-pdf](http://arxiv.org/pdf/2308.03825v2)

**Authors**: Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: The misuse of large language models (LLMs) has drawn significant attention from the general public and LLM vendors. One particular type of adversarial prompt, known as jailbreak prompt, has emerged as the main attack vector to bypass the safeguards and elicit harmful content from LLMs. In this paper, employing our new framework JailbreakHub, we conduct a comprehensive analysis of 1,405 jailbreak prompts spanning from December 2022 to December 2023. We identify 131 jailbreak communities and discover unique characteristics of jailbreak prompts and their major attack strategies, such as prompt injection and privilege escalation. We also observe that jailbreak prompts increasingly shift from online Web communities to prompt-aggregation websites and 28 user accounts have consistently optimized jailbreak prompts over 100 days. To assess the potential harm caused by jailbreak prompts, we create a question set comprising 107,250 samples across 13 forbidden scenarios. Leveraging this dataset, our experiments on six popular LLMs show that their safeguards cannot adequately defend jailbreak prompts in all scenarios. Particularly, we identify five highly effective jailbreak prompts that achieve 0.95 attack success rates on ChatGPT (GPT-3.5) and GPT-4, and the earliest one has persisted online for over 240 days. We hope that our study can facilitate the research community and LLM vendors in promoting safer and regulated LLMs.

摘要: 大型语言模型(LLM)的滥用引起了公众和LLM供应商的极大关注。一种特殊类型的对抗性提示，即越狱提示，已经成为绕过安全措施并从LLMS引出有害内容的主要攻击媒介。在本文中，我们使用我们的新框架JailBreak Hub，对从2022年12月到2023年12月的1405个越狱提示进行了全面的分析。我们识别了131个越狱社区，发现了越狱提示的独特特征及其主要攻击策略，如提示注入和特权提升。我们还观察到越狱提示越来越多地从在线网络社区转移到提示聚合网站，28个用户账户在100天内持续优化越狱提示。为了评估越狱提示造成的潜在危害，我们创建了一个包含13个禁止场景的107,250个样本的问题集。利用这个数据集，我们在六个流行的LLM上的实验表明，它们的保护措施不足以在所有场景中防御越狱提示。特别是，我们确定了五个高效的越狱提示，它们在ChatGPT(GPT-3.5)和GPT-4上的攻击成功率达到了0.95%，其中最早的一个在线时间超过了240天。我们希望我们的研究能够促进研究界和LLM供应商推广更安全和规范的LLM。



## **30. Large Language Models can be Guided to Evade AI-Generated Text Detection**

可以引导大型语言模型避免人工智能生成的文本检测 cs.CL

TMLR camera ready

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2305.10847v6) [paper-pdf](http://arxiv.org/pdf/2305.10847v6)

**Authors**: Ning Lu, Shengcai Liu, Rui He, Qi Wang, Yew-Soon Ong, Ke Tang

**Abstract**: Large language models (LLMs) have shown remarkable performance in various tasks and have been extensively utilized by the public. However, the increasing concerns regarding the misuse of LLMs, such as plagiarism and spamming, have led to the development of multiple detectors, including fine-tuned classifiers and statistical methods. In this study, we equip LLMs with prompts, rather than relying on an external paraphraser, to evaluate the vulnerability of these detectors. We propose a novel Substitution-based In-Context example Optimization method (SICO) to automatically construct prompts for evading the detectors. SICO is cost-efficient as it requires only 40 human-written examples and a limited number of LLM inferences to generate a prompt. Moreover, once a task-specific prompt has been constructed, it can be universally used against a wide range of detectors. Extensive experiments across three real-world tasks demonstrate that SICO significantly outperforms the paraphraser baselines and enables GPT-3.5 to successfully evade six detectors, decreasing their AUC by 0.5 on average. Furthermore, a comprehensive human evaluation show that the SICO-generated text achieves human-level readability and task completion rates, while preserving high imperceptibility. Finally, we propose an ensemble approach to enhance the robustness of detectors against SICO attack. The code is publicly available at https://github.com/ColinLu50/Evade-GPT-Detector.

摘要: 大型语言模型(LLM)在各种任务中表现出显著的性能，并被公众广泛使用。然而，对LLMS滥用的日益关注，如抄袭和垃圾邮件，导致了多检测器的发展，包括微调分类器和统计方法。在这项研究中，我们为LLMS配备了提示，而不是依赖外部释义来评估这些检测器的脆弱性。我们提出了一种新的基于替换的上下文中实例优化方法(SICO)来自动构建躲避检测器的提示。SICO是具有成本效益的，因为它只需要40个人写的例子和有限数量的LLM推理来生成提示。此外，一旦构建了特定于任务的提示，它就可以普遍用于各种检测器。在三个真实任务上的广泛实验表明，SICO的性能显著优于释义基线，并使GPT-3.5成功避开了六个检测器，使它们的AUC平均降低了0.5%。此外，一项全面的人类评估表明，SICO生成的文本达到了人类水平的可读性和任务完成率，同时保持了高度的不可察觉。最后，我们提出了一种集成方法来增强检测器对SICO攻击的稳健性。该代码可在https://github.com/ColinLu50/Evade-GPT-Detector.上公开获得



## **31. Efficient LLM Jailbreak via Adaptive Dense-to-sparse Constrained Optimization**

通过自适应密集到稀疏约束优化实现高效LLM越狱 cs.LG

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09113v1) [paper-pdf](http://arxiv.org/pdf/2405.09113v1)

**Authors**: Kai Hu, Weichen Yu, Tianjun Yao, Xiang Li, Wenhe Liu, Lijun Yu, Yining Li, Kai Chen, Zhiqiang Shen, Matt Fredrikson

**Abstract**: Recent research indicates that large language models (LLMs) are susceptible to jailbreaking attacks that can generate harmful content. This paper introduces a novel token-level attack method, Adaptive Dense-to-Sparse Constrained Optimization (ADC), which effectively jailbreaks several open-source LLMs. Our approach relaxes the discrete jailbreak optimization into a continuous optimization and progressively increases the sparsity of the optimizing vectors. Consequently, our method effectively bridges the gap between discrete and continuous space optimization. Experimental results demonstrate that our method is more effective and efficient than existing token-level methods. On Harmbench, our method achieves state of the art attack success rate on seven out of eight LLMs. Code will be made available. Trigger Warning: This paper contains model behavior that can be offensive in nature.

摘要: 最近的研究表明，大型语言模型（LLM）很容易受到可能生成有害内容的越狱攻击。本文介绍了一种新颖的代币级攻击方法--自适应密度到稀疏约束优化（ADC），它可以有效地越狱多个开源LLM。我们的方法将离散越狱优化放宽为连续优化，并逐步增加优化载体的稀疏性。因此，我们的方法有效地弥合了离散和连续空间优化之间的差距。实验结果表明，我们的方法比现有的代币级方法更有效和高效。在Harmbener上，我们的方法在八分之七的LLM上实现了最先进的攻击成功率。代码将可用。触发警告：本文包含本质上可能具有冒犯性的模型行为。



## **32. A safety realignment framework via subspace-oriented model fusion for large language models**

通过面向子空间的模型融合为大型语言模型提供安全重新调整框架 cs.CL

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09055v1) [paper-pdf](http://arxiv.org/pdf/2405.09055v1)

**Authors**: Xin Yi, Shunfan Zheng, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: The current safeguard mechanisms for large language models (LLMs) are indeed susceptible to jailbreak attacks, making them inherently fragile. Even the process of fine-tuning on apparently benign data for downstream tasks can jeopardize safety. One potential solution is to conduct safety fine-tuning subsequent to downstream fine-tuning. However, there's a risk of catastrophic forgetting during safety fine-tuning, where LLMs may regain safety measures but lose the task-specific knowledge acquired during downstream fine-tuning. In this paper, we introduce a safety realignment framework through subspace-oriented model fusion (SOMF), aiming to combine the safeguard capabilities of initially aligned model and the current fine-tuned model into a realigned model. Our approach begins by disentangling all task vectors from the weights of each fine-tuned model. We then identify safety-related regions within these vectors by subspace masking techniques. Finally, we explore the fusion of the initial safely aligned LLM with all task vectors based on the identified safety subspace. We validate that our safety realignment framework satisfies the safety requirements of a single fine-tuned model as well as multiple models during their fusion. Our findings confirm that SOMF preserves safety without notably compromising performance on downstream tasks, including instruction following in Chinese, English, and Hindi, as well as problem-solving capabilities in Code and Math.

摘要: 目前大型语言模型(LLM)的保护机制确实容易受到越狱攻击，这使得它们天生就很脆弱。即使是对下游任务的表面上看是良性的数据进行微调的过程也可能危及安全。一种可能的解决方案是在下游微调之后进行安全微调。然而，在安全微调期间存在灾难性遗忘的风险，在这种情况下，LLM可能重新获得安全措施，但丢失在下游微调期间获得的特定任务的知识。本文通过面向子空间的模型融合(SOMF)提出了一种安全重排框架，旨在将初始对准模型和当前微调模型的保障能力结合到一个重排模型中。我们的方法首先将所有任务向量从每个微调模型的权重中分离出来。然后，我们使用子空间掩蔽技术来识别这些向量中与安全相关的区域。最后，基于识别出的安全子空间，我们探索了初始安全对齐的LLM与所有任务向量的融合。我们验证了我们的安全调整框架满足单个微调模型以及多个模型在融合过程中的安全要求。我们的研究结果证实，SOMF在不显著影响下游任务性能的情况下保持了安全性，包括用中文、英语和印地语进行指导，以及在代码和数学中解决问题的能力。



## **33. Distributed Threat Intelligence at the Edge Devices: A Large Language Model-Driven Approach**

边缘设备上的分布式威胁情报：大型语言模型驱动的方法 cs.CR

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08755v1) [paper-pdf](http://arxiv.org/pdf/2405.08755v1)

**Authors**: Syed Mhamudul Hasan, Alaa M. Alotaibi, Sajedul Talukder, Abdur R. Shahid

**Abstract**: With the proliferation of edge devices, there is a significant increase in attack surface on these devices. The decentralized deployment of threat intelligence on edge devices, coupled with adaptive machine learning techniques such as the in-context learning feature of large language models (LLMs), represents a promising paradigm for enhancing cybersecurity on low-powered edge devices. This approach involves the deployment of lightweight machine learning models directly onto edge devices to analyze local data streams, such as network traffic and system logs, in real-time. Additionally, distributing computational tasks to an edge server reduces latency and improves responsiveness while also enhancing privacy by processing sensitive data locally. LLM servers can enable these edge servers to autonomously adapt to evolving threats and attack patterns, continuously updating their models to improve detection accuracy and reduce false positives. Furthermore, collaborative learning mechanisms facilitate peer-to-peer secure and trustworthy knowledge sharing among edge devices, enhancing the collective intelligence of the network and enabling dynamic threat mitigation measures such as device quarantine in response to detected anomalies. The scalability and flexibility of this approach make it well-suited for diverse and evolving network environments, as edge devices only send suspicious information such as network traffic and system log changes, offering a resilient and efficient solution to combat emerging cyber threats at the network edge. Thus, our proposed framework can improve edge computing security by providing better security in cyber threat detection and mitigation by isolating the edge devices from the network.

摘要: 随着边缘设备的扩散，这些设备上的攻击面显著增加。在边缘设备上分散部署威胁情报，再加上自适应机器学习技术，如大型语言模型(LLMS)的上下文中学习功能，代表了一种在低性能边缘设备上增强网络安全的有前途的范例。这种方法涉及将轻量级机器学习模型直接部署到边缘设备上，以实时分析本地数据流，如网络流量和系统日志。此外，将计算任务分发到边缘服务器可减少延迟并提高响应速度，同时还可通过在本地处理敏感数据来增强隐私。LLM服务器可以使这些边缘服务器自主适应不断变化的威胁和攻击模式，不断更新其模型以提高检测准确性并减少误报。此外，协作学习机制促进了边缘设备之间的对等安全和可信知识共享，增强了网络的集体智能，并支持动态威胁缓解措施，如响应检测到的异常情况的设备隔离。这种方法的可扩展性和灵活性使其非常适合于多样化和不断发展的网络环境，因为边缘设备只发送可疑信息，如网络流量和系统日志更改，为应对网络边缘出现的网络威胁提供了一种弹性和高效的解决方案。因此，我们提出的框架可以通过将边缘设备与网络隔离来提供更好的网络威胁检测和缓解的安全性，从而提高边缘计算的安全性。



## **34. PLeak: Prompt Leaking Attacks against Large Language Model Applications**

PLeak：针对大型语言模型应用程序的提示泄露攻击 cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.06823v2) [paper-pdf](http://arxiv.org/pdf/2405.06823v2)

**Authors**: Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, Yinzhi Cao

**Abstract**: Large Language Models (LLMs) enable a new ecosystem with many downstream applications, called LLM applications, with different natural language processing tasks. The functionality and performance of an LLM application highly depend on its system prompt, which instructs the backend LLM on what task to perform. Therefore, an LLM application developer often keeps a system prompt confidential to protect its intellectual property. As a result, a natural attack, called prompt leaking, is to steal the system prompt from an LLM application, which compromises the developer's intellectual property. Existing prompt leaking attacks primarily rely on manually crafted queries, and thus achieve limited effectiveness.   In this paper, we design a novel, closed-box prompt leaking attack framework, called PLeak, to optimize an adversarial query such that when the attacker sends it to a target LLM application, its response reveals its own system prompt. We formulate finding such an adversarial query as an optimization problem and solve it with a gradient-based method approximately. Our key idea is to break down the optimization goal by optimizing adversary queries for system prompts incrementally, i.e., starting from the first few tokens of each system prompt step by step until the entire length of the system prompt.   We evaluate PLeak in both offline settings and for real-world LLM applications, e.g., those on Poe, a popular platform hosting such applications. Our results show that PLeak can effectively leak system prompts and significantly outperforms not only baselines that manually curate queries but also baselines with optimized queries that are modified and adapted from existing jailbreaking attacks. We responsibly reported the issues to Poe and are still waiting for their response. Our implementation is available at this repository: https://github.com/BHui97/PLeak.

摘要: 大型语言模型(LLM)支持一个新的生态系统，该生态系统具有许多下游应用程序，称为LLM应用程序，具有不同的自然语言处理任务。LLM应用程序的功能和性能高度依赖于其系统提示符，系统提示符指示后端LLM执行什么任务。因此，LLM应用程序开发人员通常会对系统提示保密，以保护其知识产权。因此，一种称为提示泄漏的自然攻击是从LLM应用程序中窃取系统提示，这会损害开发人员的知识产权。现有的即时泄漏攻击主要依赖于手动创建的查询，因此效果有限。在本文中，我们设计了一个新颖的封闭盒提示泄漏攻击框架PLeak，用于优化敌意查询，使其在攻击者将其发送到目标LLM应用程序时，其响应显示其自己的系统提示。我们将寻找这样一个敌意查询描述为一个优化问题，并用基于梯度的方法近似求解。我们的核心思想是通过对系统提示的敌意查询进行增量优化来打破优化目标，即从每个系统提示的前几个令牌开始逐步优化，直到系统提示的整个长度。我们在离线设置和现实世界的LLM应用程序(例如，托管此类应用程序的流行平台PoE上的应用程序)中对PLeak进行评估。我们的结果表明，PLeak能够有效地泄露系统提示，不仅显著优于手动管理查询的基线，而且显著优于从现有越狱攻击中修改和调整的优化查询的基线。我们负责任地向PoE报告了这些问题，并仍在等待他们的回应。我们的实现可从以下存储库获得：https://github.com/BHui97/PLeak.



## **35. Stylometric Watermarks for Large Language Models**

大型语言模型的文体水印 cs.CL

19 pages, 4 figures, 9 tables

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08400v1) [paper-pdf](http://arxiv.org/pdf/2405.08400v1)

**Authors**: Georg Niess, Roman Kern

**Abstract**: The rapid advancement of large language models (LLMs) has made it increasingly difficult to distinguish between text written by humans and machines. Addressing this, we propose a novel method for generating watermarks that strategically alters token probabilities during generation. Unlike previous works, this method uniquely employs linguistic features such as stylometry. Concretely, we introduce acrostica and sensorimotor norms to LLMs. Further, these features are parameterized by a key, which is updated every sentence. To compute this key, we use semantic zero shot classification, which enhances resilience. In our evaluation, we find that for three or more sentences, our method achieves a false positive and false negative rate of 0.02. For the case of a cyclic translation attack, we observe similar results for seven or more sentences. This research is of particular of interest for proprietary LLMs to facilitate accountability and prevent societal harm.

摘要: 大型语言模型（LLM）的快速发展使得区分人类和机器编写的文本变得越来越困难。针对这一点，我们提出了一种生成水印的新颖方法，该方法在生成过程中策略性地改变代币概率。与之前的作品不同，这种方法独特地采用了风格等语言特征。具体来说，我们向LLM引入了极乐律和感觉运动规范。此外，这些功能由一个键参数化，该键每句都会更新一次。为了计算这个密钥，我们使用语义零次分类，这增强了弹性。在我们的评估中，我们发现对于三个或更多句子，我们的方法实现了0.02的假阳性和假阴性率。对于循环翻译攻击的情况，我们观察到七个或更多句子的类似结果。这项研究对于专有LLM特别感兴趣，以促进问责制并防止社会危害。



## **36. SpeechGuard: Exploring the Adversarial Robustness of Multimodal Large Language Models**

SpeechGuard：探索多模式大型语言模型的对抗鲁棒性 cs.CL

9+6 pages, Submitted to ACL 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08317v1) [paper-pdf](http://arxiv.org/pdf/2405.08317v1)

**Authors**: Raghuveer Peri, Sai Muralidhar Jayanthi, Srikanth Ronanki, Anshu Bhatia, Karel Mundnich, Saket Dingliwal, Nilaksh Das, Zejiang Hou, Goeric Huybrechts, Srikanth Vishnubhotla, Daniel Garcia-Romero, Sundararajan Srinivasan, Kyu J Han, Katrin Kirchhoff

**Abstract**: Integrated Speech and Large Language Models (SLMs) that can follow speech instructions and generate relevant text responses have gained popularity lately. However, the safety and robustness of these models remains largely unclear. In this work, we investigate the potential vulnerabilities of such instruction-following speech-language models to adversarial attacks and jailbreaking. Specifically, we design algorithms that can generate adversarial examples to jailbreak SLMs in both white-box and black-box attack settings without human involvement. Additionally, we propose countermeasures to thwart such jailbreaking attacks. Our models, trained on dialog data with speech instructions, achieve state-of-the-art performance on spoken question-answering task, scoring over 80% on both safety and helpfulness metrics. Despite safety guardrails, experiments on jailbreaking demonstrate the vulnerability of SLMs to adversarial perturbations and transfer attacks, with average attack success rates of 90% and 10% respectively when evaluated on a dataset of carefully designed harmful questions spanning 12 different toxic categories. However, we demonstrate that our proposed countermeasures reduce the attack success significantly.

摘要: 集成的语音和大型语言模型(SLM)可以遵循语音指令并生成相关的文本响应，最近得到了广泛的应用。然而，这些模型的安全性和稳健性在很大程度上仍不清楚。在这项工作中，我们调查了这种遵循指令的语音语言模型在对抗攻击和越狱时的潜在脆弱性。具体地说，我们设计的算法可以生成白盒和黑盒攻击环境下的越狱SLM的对抗性示例，而不需要人工参与。此外，我们还提出了挫败此类越狱攻击的对策。我们的模型在对话数据和语音指令上进行了训练，在口语问答任务中实现了最先进的性能，在安全性和有助性指标上都获得了80%以上的分数。尽管有安全护栏，但越狱实验证明了SLM在对抗性扰动和转移攻击中的脆弱性，当对12个不同有毒类别的精心设计的有害问题集进行评估时，平均攻击成功率分别为90%和10%。然而，我们证明我们提出的对策显著降低了攻击的成功率。



## **37. Many-Shot Regurgitation (MSR) Prompting**

多镜头回归（MSR）回归 cs.CL

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.08134v1) [paper-pdf](http://arxiv.org/pdf/2405.08134v1)

**Authors**: Shashank Sonkar, Richard G. Baraniuk

**Abstract**: We introduce Many-Shot Regurgitation (MSR) prompting, a new black-box membership inference attack framework for examining verbatim content reproduction in large language models (LLMs). MSR prompting involves dividing the input text into multiple segments and creating a single prompt that includes a series of faux conversation rounds between a user and a language model to elicit verbatim regurgitation. We apply MSR prompting to diverse text sources, including Wikipedia articles and open educational resources (OER) textbooks, which provide high-quality, factual content and are continuously updated over time. For each source, we curate two dataset types: one that LLMs were likely exposed to during training ($D_{\rm pre}$) and another consisting of documents published after the models' training cutoff dates ($D_{\rm post}$). To quantify the occurrence of verbatim matches, we employ the Longest Common Substring algorithm and count the frequency of matches at different length thresholds. We then use statistical measures such as Cliff's delta, Kolmogorov-Smirnov (KS) distance, and Kruskal-Wallis H test to determine whether the distribution of verbatim matches differs significantly between $D_{\rm pre}$ and $D_{\rm post}$. Our findings reveal a striking difference in the distribution of verbatim matches between $D_{\rm pre}$ and $D_{\rm post}$, with the frequency of verbatim reproduction being significantly higher when LLMs (e.g. GPT models and LLaMAs) are prompted with text from datasets they were likely trained on. For instance, when using GPT-3.5 on Wikipedia articles, we observe a substantial effect size (Cliff's delta $= -0.984$) and a large KS distance ($0.875$) between the distributions of $D_{\rm pre}$ and $D_{\rm post}$. Our results provide compelling evidence that LLMs are more prone to reproducing verbatim content when the input text is likely sourced from their training data.

摘要: 介绍了一种新的黑盒成员关系推理攻击框架--多射反流(MSR)提示，用于检测大型语言模型(LLMS)中的逐字内容再现。MSR提示包括将输入文本分成多个片段，并创建单个提示，其中包括用户和语言模型之间的一系列虚假对话回合，以引发逐字反胃。我们将MSR提示应用于不同的文本来源，包括维基百科文章和开放教育资源(OER)教科书，这些内容提供高质量的事实内容，并随着时间的推移不断更新。对于每个源，我们管理两种数据集类型：一种是LLM在培训期间可能接触到的($D_{\RM Pre}$)，另一种是在模型的培训截止日期之后发布的文档($D_{\RM POST}$)。为了量化逐字匹配的发生，我们使用了最长公共子串算法，并统计了不同长度阈值下的匹配频率。然后，我们使用诸如Cliff‘s Delta、Kolmogorov-Smirnov(KS)距离和Kruskal-Wallis H检验等统计指标来确定$D_(\rm)$和$D_(\rm)_(POST)$之间逐字匹配的分布是否有显著差异。我们的发现揭示了$D_{\rm Pre}$和$D_{\rm post}$逐字匹配的分布存在显著差异，当提示LLM(例如GPT模型和大羊驼)时，当提示LLMS(例如GPT模型和大羊驼)的文本来自他们可能训练过的数据集时，逐字再现的频率显著高于$D_{\rm Pre}$和$D_{\rm post}$。例如，当在维基百科文章上使用GPT-3.5时，我们观察到$D_{\rM Pre}$和$D_{\rm post}$的分布之间有很大的效果大小(Cliff‘s Delta$=-0.984$)和很大的KS距离($0.875$)。我们的结果提供了令人信服的证据，表明当输入文本可能来自于他们的训练数据时，LLM更倾向于逐字再现内容。



## **38. Backdoor Removal for Generative Large Language Models**

生成性大型语言模型的后门删除 cs.CR

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07667v1) [paper-pdf](http://arxiv.org/pdf/2405.07667v1)

**Authors**: Haoran Li, Yulin Chen, Zihao Zheng, Qi Hu, Chunkit Chan, Heshan Liu, Yangqiu Song

**Abstract**: With rapid advances, generative large language models (LLMs) dominate various Natural Language Processing (NLP) tasks from understanding to reasoning. Yet, language models' inherent vulnerabilities may be exacerbated due to increased accessibility and unrestricted model training on massive textual data from the Internet. A malicious adversary may publish poisoned data online and conduct backdoor attacks on the victim LLMs pre-trained on the poisoned data. Backdoored LLMs behave innocuously for normal queries and generate harmful responses when the backdoor trigger is activated. Despite significant efforts paid to LLMs' safety issues, LLMs are still struggling against backdoor attacks. As Anthropic recently revealed, existing safety training strategies, including supervised fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), fail to revoke the backdoors once the LLM is backdoored during the pre-training stage. In this paper, we present Simulate and Eliminate (SANDE) to erase the undesired backdoored mappings for generative LLMs. We initially propose Overwrite Supervised Fine-tuning (OSFT) for effective backdoor removal when the trigger is known. Then, to handle the scenarios where the trigger patterns are unknown, we integrate OSFT into our two-stage framework, SANDE. Unlike previous works that center on the identification of backdoors, our safety-enhanced LLMs are able to behave normally even when the exact triggers are activated. We conduct comprehensive experiments to show that our proposed SANDE is effective against backdoor attacks while bringing minimal harm to LLMs' powerful capability without any additional access to unbackdoored clean models. We will release the reproducible code.

摘要: 随着研究的深入，从理解到推理的各种自然语言处理任务都被生成性大语言模型(LLMS)所支配。然而，语言模型固有的脆弱性可能会因为可访问性的提高和对来自互联网的海量文本数据的不受限制的模型训练而加剧。恶意对手可能会在网上发布有毒数据，并对受害者LLM进行后门攻击，这些LLM预先训练了有毒数据。后门LLM在正常查询中的行为是无害的，并在激活后门触发器时生成有害的响应。尽管在LLMS的安全问题上付出了巨大的努力，但LLMS仍在努力应对后门攻击。正如人类最近揭示的那样，现有的安全培训策略，包括监督微调(SFT)和从人类反馈的强化学习(RLHF)，一旦LLM在培训前阶段后退，就无法取消后门。在这篇文章中，我们提出了模拟和消除(SANDE)来消除生成式LLMS中不需要的回溯映射。我们最初提出了覆盖监督精调(OSFT)，用于在已知触发器的情况下有效地删除后门。然后，为了处理触发模式未知的场景，我们将OSFT集成到我们的两阶段框架SANDE中。与以前以识别后门为中心的工作不同，我们的安全增强型LLM即使在准确的触发器被激活时也能够正常运行。我们进行了全面的实验，以表明我们提出的SANDE可以有效地抵御后门攻击，同时对LLMS的强大功能造成的损害最小，而不需要额外访问未后门的干净模型。我们将发布可重现的代码。



## **39. DoLLM: How Large Language Models Understanding Network Flow Data to Detect Carpet Bombing DDoS**

DoLLM：大型语言模型如何理解网络流数据来检测地毯炸弹DDOS cs.NI

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07638v1) [paper-pdf](http://arxiv.org/pdf/2405.07638v1)

**Authors**: Qingyang Li, Yihang Zhang, Zhidong Jia, Yannan Hu, Lei Zhang, Jianrong Zhang, Yongming Xu, Yong Cui, Zongming Guo, Xinggong Zhang

**Abstract**: It is an interesting question Can and How Large Language Models (LLMs) understand non-language network data, and help us detect unknown malicious flows. This paper takes Carpet Bombing as a case study and shows how to exploit LLMs' powerful capability in the networking area. Carpet Bombing is a new DDoS attack that has dramatically increased in recent years, significantly threatening network infrastructures. It targets multiple victim IPs within subnets, causing congestion on access links and disrupting network services for a vast number of users. Characterized by low-rates, multi-vectors, these attacks challenge traditional DDoS defenses. We propose DoLLM, a DDoS detection model utilizes open-source LLMs as backbone. By reorganizing non-contextual network flows into Flow-Sequences and projecting them into LLMs semantic space as token embeddings, DoLLM leverages LLMs' contextual understanding to extract flow representations in overall network context. The representations are used to improve the DDoS detection performance. We evaluate DoLLM with public datasets CIC-DDoS2019 and real NetFlow trace from Top-3 countrywide ISP. The tests have proven that DoLLM possesses strong detection capabilities. Its F1 score increased by up to 33.3% in zero-shot scenarios and by at least 20.6% in real ISP traces.

摘要: 大型语言模型能够以及如何理解非语言网络数据，并帮助我们检测未知的恶意流量，这是一个有趣的问题。本文以地毯轰炸为例，展示了如何利用低成本管理系统在组网领域的强大能力。地毯式轰炸是一种新型的DDoS攻击，近年来急剧增加，严重威胁着网络基础设施。它以子网内的多个受害IP为目标，导致接入链路拥塞，扰乱了大量用户的网络服务。这些攻击以低速率、多向量为特征，挑战了传统的DDoS防御。我们提出了DoLLM，一个以开源LLMS为骨干的DDoS检测模型。通过将非上下文网络流重新组织成流序列，并将其投影到LLMS语义空间作为令牌嵌入，DoLLM利用LLMS的上下文理解来提取整个网络上下文中的流表示。这些表示用于提高DDoS检测性能。我们使用公共数据集CIC-DDoS2019和来自全国排名前三的运营商的真实NetFlow跟踪来评估DoLLM。测试证明，DoLLM具有很强的检测能力。它的F1得分在零镜头场景下增加了33.3%，在真实的isp轨迹中至少增加了20.6%。



## **40. ExplainableDetector: Exploring Transformer-based Language Modeling Approach for SMS Spam Detection with Explainability Analysis**

解释性检测器：通过可解释性分析探索基于转换器的语言建模方法用于短信垃圾邮件检测 cs.LG

**SubmitDate**: 2024-05-12    [abs](http://arxiv.org/abs/2405.08026v1) [paper-pdf](http://arxiv.org/pdf/2405.08026v1)

**Authors**: Mohammad Amaz Uddin, Muhammad Nazrul Islam, Leandros Maglaras, Helge Janicke, Iqbal H. Sarker

**Abstract**: SMS, or short messaging service, is a widely used and cost-effective communication medium that has sadly turned into a haven for unwanted messages, commonly known as SMS spam. With the rapid adoption of smartphones and Internet connectivity, SMS spam has emerged as a prevalent threat. Spammers have taken notice of the significance of SMS for mobile phone users. Consequently, with the emergence of new cybersecurity threats, the number of SMS spam has expanded significantly in recent years. The unstructured format of SMS data creates significant challenges for SMS spam detection, making it more difficult to successfully fight spam attacks in the cybersecurity domain. In this work, we employ optimized and fine-tuned transformer-based Large Language Models (LLMs) to solve the problem of spam message detection. We use a benchmark SMS spam dataset for this spam detection and utilize several preprocessing techniques to get clean and noise-free data and solve the class imbalance problem using the text augmentation technique. The overall experiment showed that our optimized fine-tuned BERT (Bidirectional Encoder Representations from Transformers) variant model RoBERTa obtained high accuracy with 99.84\%. We also work with Explainable Artificial Intelligence (XAI) techniques to calculate the positive and negative coefficient scores which explore and explain the fine-tuned model transparency in this text-based spam SMS detection task. In addition, traditional Machine Learning (ML) models were also examined to compare their performance with the transformer-based models. This analysis describes how LLMs can make a good impact on complex textual-based spam data in the cybersecurity field.

摘要: 短信，或短消息服务，是一种广泛使用且具有成本效益的通信媒介，遗憾的是，它已经变成了不想要的消息的避风港，通常被称为短信垃圾邮件。随着智能手机和互联网连接的迅速普及，垃圾短信已经成为一种普遍的威胁。垃圾邮件发送者已经注意到短信对手机用户的重要性。因此，随着新的网络安全威胁的出现，近年来短信垃圾邮件的数量大幅增加。短信数据的非结构化格式给短信垃圾邮件检测带来了巨大的挑战，使得在网络安全领域成功打击垃圾邮件攻击变得更加困难。在这项工作中，我们使用优化和微调的基于转换器的大语言模型(LLMS)来解决垃圾消息检测问题。我们使用一个基准的短信垃圾邮件数据集进行垃圾邮件检测，并利用几种预处理技术来获得干净和无噪声的数据，并使用文本增强技术解决类别不平衡问题。整体实验表明，优化后的BERT(Transformers的双向编码器表示)变体模型Roberta获得了99.84%的高精度。我们还使用可解释人工智能(XAI)技术来计算正系数和负系数得分，从而探索和解释了在这个基于文本的垃圾短信检测任务中微调的模型透明度。此外，还研究了传统的机器学习(ML)模型，并与基于变压器的模型进行了比较。这一分析描述了LLMS如何在网络安全领域对复杂的基于文本的垃圾数据产生良好的影响。



## **41. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

Janus界面：大型语言模型中的微调如何放大隐私风险 cs.CR

**SubmitDate**: 2024-05-12    [abs](http://arxiv.org/abs/2310.15469v2) [paper-pdf](http://arxiv.org/pdf/2310.15469v2)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, Haixu Tang

**Abstract**: The rapid advancements of large language models (LLMs) have raised public concerns about the privacy leakage of personally identifiable information (PII) within their extensive training datasets. Recent studies have demonstrated that an adversary could extract highly sensitive privacy data from the training data of LLMs with carefully designed prompts. However, these attacks suffer from the model's tendency to hallucinate and catastrophic forgetting (CF) in the pre-training stage, rendering the veracity of divulged PIIs negligible. In our research, we propose a novel attack, Janus, which exploits the fine-tuning interface to recover forgotten PIIs from the pre-training data in LLMs. We formalize the privacy leakage problem in LLMs and explain why forgotten PIIs can be recovered through empirical analysis on open-source language models. Based upon these insights, we evaluate the performance of Janus on both open-source language models and two latest LLMs, i.e., GPT-3.5-Turbo and LLaMA-2-7b. Our experiment results show that Janus amplifies the privacy risks by over 10 times in comparison with the baseline and significantly outperforms the state-of-the-art privacy extraction attacks including prefix attacks and in-context learning (ICL). Furthermore, our analysis validates that existing fine-tuning APIs provided by OpenAI and Azure AI Studio are susceptible to our Janus attack, allowing an adversary to conduct such an attack at a low cost.

摘要: 大型语言模型(LLM)的快速发展引起了公众对其广泛训练数据集中个人身份信息(PII)隐私泄露的担忧。最近的研究表明，攻击者可以通过精心设计的提示从LLMS的训练数据中提取高度敏感的隐私数据。然而，这些攻击受到模型在预训练阶段的幻觉和灾难性遗忘(CF)的倾向的影响，使得泄露的PII的真实性可以忽略不计。在我们的研究中，我们提出了一种新的攻击，Janus，它利用微调接口从LLMS的训练前数据中恢复被遗忘的PII。我们形式化地描述了LLMS中的隐私泄露问题，并通过对开源语言模型的实证分析解释了为什么被遗忘的PII可以恢复。基于这些见解，我们评估了Janus在开源语言模型和两个最新的LLMS上的性能，即GPT-3.5-Turbo和Llama-2-7b。我们的实验结果表明，Janus将隐私风险放大了10倍以上，并且显著优于目前最先进的隐私提取攻击，包括前缀攻击和上下文中学习(ICL)。此外，我们的分析验证了OpenAI和Azure AI Studio提供的现有微调API容易受到我们的Janus攻击，允许对手以低成本进行此类攻击。



## **42. LLMs and the Future of Chip Design: Unveiling Security Risks and Building Trust**

法学硕士和芯片设计的未来：揭示安全风险并建立信任 cs.LG

**SubmitDate**: 2024-05-11    [abs](http://arxiv.org/abs/2405.07061v1) [paper-pdf](http://arxiv.org/pdf/2405.07061v1)

**Authors**: Zeng Wang, Lilas Alrahis, Likhitha Mankali, Johann Knechtel, Ozgur Sinanoglu

**Abstract**: Chip design is about to be revolutionized by the integration of large language, multimodal, and circuit models (collectively LxMs). While exploring this exciting frontier with tremendous potential, the community must also carefully consider the related security risks and the need for building trust into using LxMs for chip design. First, we review the recent surge of using LxMs for chip design in general. We cover state-of-the-art works for the automation of hardware description language code generation and for scripting and guidance of essential but cumbersome tasks for electronic design automation tools, e.g., design-space exploration, tuning, or designer training. Second, we raise and provide initial answers to novel research questions on critical issues for security and trustworthiness of LxM-powered chip design from both the attack and defense perspectives.

摘要: 芯片设计即将因大型语言、多模式和电路模型（统称为LxM）的集成而发生革命性变化。在探索这个具有巨大潜力的令人兴奋的前沿时，社区还必须仔细考虑相关的安全风险以及在使用LxM进行芯片设计方面建立信任的必要性。首先，我们回顾了最近使用LxM进行芯片设计的热潮。我们涵盖硬件描述语言代码生成自动化以及电子设计自动化工具的重要但繁琐任务的脚本和指导的最先进作品，例如，设计空间探索、调整或设计师培训。其次，我们从攻击和防御的角度对LxM驱动芯片设计的安全性和可信性的关键问题提出并提供了初步答案。



## **43. Talk Too Much: Poisoning Large Language Models under Token Limit**

话太多：代币限制下的大型语言模型中毒 cs.CL

**SubmitDate**: 2024-05-11    [abs](http://arxiv.org/abs/2404.14795v3) [paper-pdf](http://arxiv.org/pdf/2404.14795v3)

**Authors**: Jiaming He, Wenbo Jiang, Guanyu Hou, Wenshu Fan, Rui Zhang, Hongwei Li

**Abstract**: Mainstream poisoning attacks on large language models (LLMs) typically set a fixed trigger in the input instance and specific responses for triggered queries. However, the fixed trigger setting (e.g., unusual words) may be easily detected by human detection, limiting the effectiveness and practicality in real-world scenarios. To enhance the stealthiness of the trigger, we present a poisoning attack against LLMs that is triggered by a generation/output condition-token limitation, which is a commonly adopted strategy by users for reducing costs. The poisoned model performs normally for output without token limitation, while becomes harmful for output with limited tokens. To achieve this objective, we introduce BrieFool, an efficient attack framework. It leverages the characteristics of generation limitation by efficient instruction sampling and poisoning data generation, thereby influencing the behavior of LLMs under target conditions. Our experiments demonstrate that BrieFool is effective across safety domains and knowledge domains. For instance, with only 20 generated poisoning examples against GPT-3.5-turbo, BrieFool achieves a 100% Attack Success Rate (ASR) and a 9.28/10 average Harmfulness Score (HS) under token limitation conditions while maintaining the benign performance.

摘要: 针对大型语言模型(LLM)的主流中毒攻击通常会在输入实例中设置固定的触发器，并为触发的查询设置特定的响应。然而，固定的触发设置(例如，不寻常的单词)可能很容易被人类检测到，从而限制了在现实世界场景中的有效性和实用性。为了增强触发器的隐蔽性，我们提出了一种由生成/输出条件-令牌限制触发的针对LLMS的中毒攻击，这是用户为降低成本而常用的策略。对于没有令牌限制的输出，中毒模型执行正常，而对于具有有限令牌的输出则变得有害。为了实现这一目标，我们引入了一种高效的攻击框架BrieFool。它通过高效的指令采样和中毒数据生成来利用生成限制的特性，从而影响LLMS在目标条件下的行为。我们的实验表明，BrieFool是跨安全域和知识域的有效的。例如，在对GPT-3.5-Turbo仅生成20个中毒实例的情况下，BrieFool在保持良性性能的同时，在令牌限制条件下实现了100%的攻击成功率(ASR)和9.28/10的平均危害性评分(HS)。



## **44. Explaining Arguments' Strength: Unveiling the Role of Attacks and Supports (Technical Report)**

解释争论的力量：揭示攻击和支持的作用（技术报告） cs.AI

This paper has been accepted at IJCAI 2024 (the 33rd International  Joint Conference on Artificial Intelligence)

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2404.14304v2) [paper-pdf](http://arxiv.org/pdf/2404.14304v2)

**Authors**: Xiang Yin, Potyka Nico, Francesca Toni

**Abstract**: Quantitatively explaining the strength of arguments under gradual semantics has recently received increasing attention. Specifically, several works in the literature provide quantitative explanations by computing the attribution scores of arguments. These works disregard the importance of attacks and supports, even though they play an essential role when explaining arguments' strength. In this paper, we propose a novel theory of Relation Attribution Explanations (RAEs), adapting Shapley values from game theory to offer fine-grained insights into the role of attacks and supports in quantitative bipolar argumentation towards obtaining the arguments' strength. We show that RAEs satisfy several desirable properties. We also propose a probabilistic algorithm to approximate RAEs efficiently. Finally, we show the application value of RAEs in fraud detection and large language models case studies.

摘要: 在渐进语义下定量解释论点的强度最近受到越来越多的关注。具体来说，文献中的几部作品通过计算论点的归因分数来提供量化解释。这些作品忽视了攻击和支持的重要性，尽管它们在解释论点的强度时发挥着至关重要的作用。在本文中，我们提出了一种新的关系归因解释（RAEs）理论，改编了博弈论中的沙普利价值观，以提供对攻击作用的细粒度见解，并支持量化两极论证，以获得论点的强度。我们表明RAE满足几个理想的性质。我们还提出了一种有效逼近RAE的概率算法。最后，我们展示了RAE在欺诈检测和大型语言模型案例研究中的应用价值。



## **45. Risks of Practicing Large Language Models in Smart Grid: Threat Modeling and Validation**

在智能电网中实践大型语言模型的风险：威胁建模和验证 cs.CR

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06237v1) [paper-pdf](http://arxiv.org/pdf/2405.06237v1)

**Authors**: Jiangnan Li, Yingyuan Yang, Jinyuan Sun

**Abstract**: Large Language Model (LLM) is a significant breakthrough in artificial intelligence (AI) and holds considerable potential for application within smart grids. However, as demonstrated in previous literature, AI technologies are susceptible to various types of attacks. It is crucial to investigate and evaluate the risks associated with LLMs before deploying them in critical infrastructure like smart grids. In this paper, we systematically evaluate the vulnerabilities of LLMs and identify two major types of attacks relevant to smart grid LLM applications, along with presenting the corresponding threat models. We then validate these attacks using popular LLMs, utilizing real smart grid data. Our validation demonstrates that attackers are capable of injecting bad data and retrieving domain knowledge from LLMs employed in smart grid scenarios.

摘要: 大型语言模型（LLM）是人工智能（AI）领域的重大突破，在智能电网中具有巨大的应用潜力。然而，正如之前的文献所证明的那样，人工智能技术容易受到各种类型的攻击。在将LLM部署在智能电网等关键基础设施中之前，调查和评估与LLM相关的风险至关重要。在本文中，我们系统地评估了LLM的漏洞，识别了与智能电网LLM应用相关的两种主要攻击类型，并提出了相应的威胁模型。然后，我们使用流行的LLM并利用真实的智能电网数据来验证这些攻击。我们的验证表明，攻击者能够注入不良数据并从智能电网场景中使用的LLM中检索领域知识。



## **46. Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models**

静音低语：对语音基础模型的通用声学对抗攻击 cs.CL

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06134v1) [paper-pdf](http://arxiv.org/pdf/2405.06134v1)

**Authors**: Vyas Raina, Rao Ma, Charles McGhee, Kate Knill, Mark Gales

**Abstract**: Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as $\texttt{<endoftext>}$, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's $\texttt{<endoftext>}$ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively `muting' the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97\% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to `muting' adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.

摘要: 像Whisper这样的大型语音基础模型的最新发展导致它们在许多自动语音识别(ASR)应用中被广泛使用。这些系统在它们的词汇表中加入了“特殊记号”，如$\exttt{<endoftext>}$，以指导它们的语言生成过程。然而，我们证明了这些令牌可以被敌意攻击利用来操纵模型的行为。我们提出了一种简单而有效的方法来学习Whisper的$\exttt{<endoftext>}$标记的通用声学实现，当预先添加到任何语音信号时，鼓励模型忽略语音，只转录特殊的标记，从而有效地抑制了模型。我们的实验表明，相同的、通用的0.64秒的对抗性音频片段可以成功地使目标Whisper ASR模型在97%以上的语音样本上静音。此外，我们发现这种普遍的对抗性音频片段经常转移到新的数据集和任务。总体而言，这项工作证明了Whisper模型对“静音”对手攻击的脆弱性，在现实世界中，这种攻击既可以带来风险，也可以带来潜在的好处：例如，攻击可以用来绕过语音调节系统，或者反过来，攻击也可以用来保护私人语音数据。



## **47. Trustworthy AI-Generative Content in Intelligent 6G Network: Adversarial, Privacy, and Fairness**

智能6G网络中值得信赖的人工智能生成内容：对抗性、隐私性和公平性 cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05930v1) [paper-pdf](http://arxiv.org/pdf/2405.05930v1)

**Authors**: Siyuan Li, Xi Lin, Yaju Liu, Jianhua Li

**Abstract**: AI-generated content (AIGC) models, represented by large language models (LLM), have brought revolutionary changes to the content generation fields. The high-speed and extensive 6G technology is an ideal platform for providing powerful AIGC mobile service applications, while future 6G mobile networks also need to support intelligent and personalized mobile generation services. However, the significant ethical and security issues of current AIGC models, such as adversarial attacks, privacy, and fairness, greatly affect the credibility of 6G intelligent networks, especially in ensuring secure, private, and fair AIGC applications. In this paper, we propose TrustGAIN, a novel paradigm for trustworthy AIGC in 6G networks, to ensure trustworthy large-scale AIGC services in future 6G networks. We first discuss the adversarial attacks and privacy threats faced by AIGC systems in 6G networks, as well as the corresponding protection issues. Subsequently, we emphasize the importance of ensuring the unbiasedness and fairness of the mobile generative service in future intelligent networks. In particular, we conduct a use case to demonstrate that TrustGAIN can effectively guide the resistance against malicious or generated false information. We believe that TrustGAIN is a necessary paradigm for intelligent and trustworthy 6G networks to support AIGC services, ensuring the security, privacy, and fairness of AIGC network services.

摘要: 以大语言模型(LLM)为代表的AI-Generated Content(AIGC)模型给内容生成领域带来了革命性的变化。高速和广泛的6G技术是提供强大的AIGC移动业务应用的理想平台，而未来的6G移动网络也需要支持智能化和个性化的移动生成服务。然而，当前AIGC模型存在的重大伦理和安全问题，如对抗性攻击、隐私和公平性，极大地影响了6G智能网络的可信度，特别是在确保安全、私有和公平的AIGC应用方面。本文提出了一种新的6G网络可信AIGC模型TrustGAIN，以保证未来6G网络中可信赖的大规模AIGC服务。我们首先讨论了AIGC系统在6G网络中面临的敌意攻击和隐私威胁，以及相应的保护问题。随后，我们强调了在未来的智能网中确保移动生成业务的无偏性和公平性的重要性。特别是，我们进行了一个用例来证明TrustGAIN可以有效地指导对恶意或生成的虚假信息的抵抗。我们认为，TrustGAIN是智能可信6G网络支持AIGC服务的必备范式，确保AIGC网络服务的安全性、私密性和公平性。



## **48. LLMPot: Automated LLM-based Industrial Protocol and Physical Process Emulation for ICS Honeypots**

LLMPot：用于ICS蜜罐的自动化基于LLM的工业协议和物理流程仿真 cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05999v1) [paper-pdf](http://arxiv.org/pdf/2405.05999v1)

**Authors**: Christoforos Vasilatos, Dunia J. Mahboobeh, Hithem Lamri, Manaar Alam, Michail Maniatakos

**Abstract**: Industrial Control Systems (ICS) are extensively used in critical infrastructures ensuring efficient, reliable, and continuous operations. However, their increasing connectivity and addition of advanced features make them vulnerable to cyber threats, potentially leading to severe disruptions in essential services. In this context, honeypots play a vital role by acting as decoy targets within ICS networks, or on the Internet, helping to detect, log, analyze, and develop mitigations for ICS-specific cyber threats. Deploying ICS honeypots, however, is challenging due to the necessity of accurately replicating industrial protocols and device characteristics, a crucial requirement for effectively mimicking the unique operational behavior of different industrial systems. Moreover, this challenge is compounded by the significant manual effort required in also mimicking the control logic the PLC would execute, in order to capture attacker traffic aiming to disrupt critical infrastructure operations. In this paper, we propose LLMPot, a novel approach for designing honeypots in ICS networks harnessing the potency of Large Language Models (LLMs). LLMPot aims to automate and optimize the creation of realistic honeypots with vendor-agnostic configurations, and for any control logic, aiming to eliminate the manual effort and specialized knowledge traditionally required in this domain. We conducted extensive experiments focusing on a wide array of parameters, demonstrating that our LLM-based approach can effectively create honeypot devices implementing different industrial protocols and diverse control logic.

摘要: 工业控制系统(ICS)广泛应用于关键基础设施中，以确保高效、可靠和连续的运行。然而，它们日益增长的连通性和先进功能的增加使它们容易受到网络威胁，可能导致基本服务的严重中断。在这种情况下，蜜罐扮演着至关重要的角色，在ICS网络中或在互联网上充当诱骗目标，帮助检测、记录、分析和缓解ICS特定的网络威胁。然而，部署ICS蜜罐是具有挑战性的，因为必须准确复制工业协议和设备特性，这是有效模拟不同工业系统独特操作行为的关键要求。此外，为了捕获旨在扰乱关键基础设施操作的攻击者流量，还需要大量手动工作来模拟PLC将执行的控制逻辑，从而使这一挑战变得更加复杂。在本文中，我们提出了LLMPot，这是一种利用大语言模型的有效性来设计ICS网络中的蜜罐的新方法。LLMPot旨在自动化和优化创建具有供应商无关配置的逼真蜜罐，以及任何控制逻辑，旨在消除该领域传统上所需的手动工作和专业知识。我们针对广泛的参数进行了广泛的实验，证明了我们基于LLM的方法可以有效地创建执行不同工业协议和不同控制逻辑的蜜罐设备。



## **49. Chain of Attack: a Semantic-Driven Contextual Multi-Turn attacker for LLM**

攻击链：LLM的语义驱动上下文多回合攻击者 cs.CL

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05610v1) [paper-pdf](http://arxiv.org/pdf/2405.05610v1)

**Authors**: Xikang Yang, Xuehai Tang, Songlin Hu, Jizhong Han

**Abstract**: Large language models (LLMs) have achieved remarkable performance in various natural language processing tasks, especially in dialogue systems. However, LLM may also pose security and moral threats, especially in multi round conversations where large models are more easily guided by contextual content, resulting in harmful or biased responses. In this paper, we present a novel method to attack LLMs in multi-turn dialogues, called CoA (Chain of Attack). CoA is a semantic-driven contextual multi-turn attack method that adaptively adjusts the attack policy through contextual feedback and semantic relevance during multi-turn of dialogue with a large model, resulting in the model producing unreasonable or harmful content. We evaluate CoA on different LLMs and datasets, and show that it can effectively expose the vulnerabilities of LLMs, and outperform existing attack methods. Our work provides a new perspective and tool for attacking and defending LLMs, and contributes to the security and ethical assessment of dialogue systems.

摘要: 大语言模型在各种自然语言处理任务中取得了显著的性能，尤其是在对话系统中。然而，LLM也可能构成安全和道德威胁，特别是在多轮对话中，大型模型更容易受到上下文内容的指导，导致有害或有偏见的反应。本文提出了一种新的攻击多话轮对话中的LLMS的方法，称为攻击链。CoA是一种语义驱动的上下文多轮攻击方法，在大模型的多轮对话中，通过上下文反馈和语义相关性自适应调整攻击策略，导致模型产生不合理或有害的内容。我们在不同的LLM和数据集上对CoA进行了评估，结果表明它可以有效地暴露LLMS的漏洞，并优于现有的攻击方法。我们的工作为攻击和防御LLMS提供了一个新的视角和工具，并有助于对话系统的安全和伦理评估。



## **50. Large Language Models for Cyber Security: A Systematic Literature Review**

网络安全的大型语言模型：系统性文献综述 cs.CR

46 pages,6 figures

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.04760v2) [paper-pdf](http://arxiv.org/pdf/2405.04760v2)

**Authors**: HanXiang Xu, ShenAo Wang, NingKe Li, KaiLong Wang, YanJie Zhao, Kai Chen, Ting Yu, Yang Liu, HaoYu Wang

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened up new opportunities for leveraging artificial intelligence in various domains, including cybersecurity. As the volume and sophistication of cyber threats continue to grow, there is an increasing need for intelligent systems that can automatically detect vulnerabilities, analyze malware, and respond to attacks. In this survey, we conduct a comprehensive review of the literature on the application of LLMs in cybersecurity (LLM4Security). By comprehensively collecting over 30K relevant papers and systematically analyzing 127 papers from top security and software engineering venues, we aim to provide a holistic view of how LLMs are being used to solve diverse problems across the cybersecurity domain. Through our analysis, we identify several key findings. First, we observe that LLMs are being applied to a wide range of cybersecurity tasks, including vulnerability detection, malware analysis, network intrusion detection, and phishing detection. Second, we find that the datasets used for training and evaluating LLMs in these tasks are often limited in size and diversity, highlighting the need for more comprehensive and representative datasets. Third, we identify several promising techniques for adapting LLMs to specific cybersecurity domains, such as fine-tuning, transfer learning, and domain-specific pre-training. Finally, we discuss the main challenges and opportunities for future research in LLM4Security, including the need for more interpretable and explainable models, the importance of addressing data privacy and security concerns, and the potential for leveraging LLMs for proactive defense and threat hunting. Overall, our survey provides a comprehensive overview of the current state-of-the-art in LLM4Security and identifies several promising directions for future research.

摘要: 大型语言模型(LLM)的快速发展为在包括网络安全在内的各个领域利用人工智能开辟了新的机会。随着网络威胁的数量和复杂性不断增长，对能够自动检测漏洞、分析恶意软件和响应攻击的智能系统的需求越来越大。在本次调查中，我们对LLMS在网络安全中的应用(LLM4Security)的文献进行了全面的回顾。通过全面收集30K多篇相关论文并系统分析来自顶级安全和软件工程场所的127篇论文，我们的目标是提供一个全面的视角，了解LLM是如何被用来解决网络安全领域的各种问题的。通过我们的分析，我们确定了几个关键发现。首先，我们观察到LLMS正被应用于广泛的网络安全任务，包括漏洞检测、恶意软件分析、网络入侵检测和网络钓鱼检测。其次，我们发现，在这些任务中用于训练和评估土地管理的数据集在大小和多样性方面往往有限，这突显了需要更全面和更具代表性的数据集。第三，我们确定了几种有前景的技术来使LLMS适应特定的网络安全领域，例如微调、迁移学习和特定领域的预训练。最后，我们讨论了LLM4Security未来研究的主要挑战和机遇，包括需要更多可解释和可解释的模型，解决数据隐私和安全问题的重要性，以及利用LLM进行主动防御和威胁追踪的潜力。总体而言，我们的调查提供了对LLM4Security当前最先进技术的全面概述，并确定了未来研究的几个有希望的方向。



