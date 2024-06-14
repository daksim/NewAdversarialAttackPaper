# Latest Large Language Model Attack Papers
**update at 2024-06-14 16:18:54**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space**

REVS：通过词汇空间中的排名编辑消除语言模型中的敏感信息 cs.CL

18 pages, 3 figures

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09325v1) [paper-pdf](http://arxiv.org/pdf/2406.09325v1)

**Authors**: Tomer Ashuach, Martin Tutek, Yonatan Belinkov

**Abstract**: Large language models (LLMs) risk inadvertently memorizing and divulging sensitive or personally identifiable information (PII) seen in training data, causing privacy concerns. Current approaches to address this issue involve costly dataset scrubbing, or model filtering through unlearning and model editing, which can be bypassed through extraction attacks. We propose REVS, a novel model editing method for unlearning sensitive information from LLMs. REVS identifies and modifies a small subset of neurons relevant for each piece of sensitive information. By projecting these neurons to the vocabulary space (unembedding), we pinpoint the components driving its generation. We then compute a model edit based on the pseudo-inverse of the unembedding matrix, and apply it to de-promote generation of the targeted sensitive data. To adequately evaluate our method on truly sensitive information, we curate two datasets: an email dataset inherently memorized by GPT-J, and a synthetic social security number dataset that we tune the model to memorize. Compared to other state-of-the-art model editing methods, REVS demonstrates superior performance in both eliminating sensitive information and robustness to extraction attacks, while retaining integrity of the underlying model. The code and a demo notebook are available at https://technion-cs-nlp.github.io/REVS.

摘要: 大型语言模型(LLM)可能会无意中记住和泄露训练数据中看到的敏感或个人身份信息(PII)，从而导致隐私问题。目前解决这一问题的方法包括代价高昂的数据集清理，或通过遗忘和模型编辑进行模型过滤，这可以通过提取攻击绕过。我们提出了一种新的模型编辑方法--REVS，用于遗忘LLMS中的敏感信息。Revs识别并修改与每条敏感信息相关的一小部分神经元。通过将这些神经元投射到词汇空间(非嵌入)，我们准确地找到了驱动其生成的组件。然后基于去嵌入矩阵的伪逆计算模型编辑，并将其应用于目标敏感数据的反加速生成。为了在真正敏感的信息上充分评估我们的方法，我们整理了两个数据集：GPT-J固有地记忆的电子邮件数据集，以及我们调整模型以记忆的合成社会安全号码数据集。与其他最先进的模型编辑方法相比，RERS在消除敏感信息和对提取攻击的稳健性方面表现出优越的性能，同时保持了底层模型的完整性。代码和演示笔记本可在https://technion-cs-nlp.github.io/REVS.上获得



## **2. Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs**

诡计袋：对LLM越狱攻击的基准 cs.CR

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09324v1) [paper-pdf](http://arxiv.org/pdf/2406.09324v1)

**Authors**: Zhao Xu, Fan Liu, Hao Liu

**Abstract**: Although Large Language Models (LLMs) have demonstrated significant capabilities in executing complex tasks in a zero-shot manner, they are susceptible to jailbreak attacks and can be manipulated to produce harmful outputs. Recently, a growing body of research has categorized jailbreak attacks into token-level and prompt-level attacks. However, previous work primarily overlooks the diverse key factors of jailbreak attacks, with most studies concentrating on LLM vulnerabilities and lacking exploration of defense-enhanced LLMs. To address these issues, we evaluate the impact of various attack settings on LLM performance and provide a baseline benchmark for jailbreak attacks, encouraging the adoption of a standardized evaluation framework. Specifically, we evaluate the eight key factors of implementing jailbreak attacks on LLMs from both target-level and attack-level perspectives. We further conduct seven representative jailbreak attacks on six defense methods across two widely used datasets, encompassing approximately 320 experiments with about 50,000 GPU hours on A800-80G. Our experimental results highlight the need for standardized benchmarking to evaluate these attacks on defense-enhanced LLMs. Our code is available at https://github.com/usail-hkust/Bag_of_Tricks_for_LLM_Jailbreaking.

摘要: 尽管大型语言模型(LLM)在以零射击方式执行复杂任务方面表现出了巨大的能力，但它们很容易受到越狱攻击，并可能被操纵以产生有害的输出。最近，越来越多的研究将越狱攻击分为令牌级攻击和提示级攻击。然而，以前的工作主要忽略了越狱攻击的各种关键因素，大多数研究集中在LLM漏洞上，而缺乏对增强防御的LLM的探索。为了解决这些问题，我们评估了各种攻击设置对LLM性能的影响，并提供了越狱攻击的基准，鼓励采用标准化的评估框架。具体地，我们从目标级和攻击级两个角度评估了对LLMS实施越狱攻击的八个关键因素。我们进一步在两个广泛使用的数据集上对六种防御方法进行了七次有代表性的越狱攻击，包括在A800-80G上进行了大约320个实验，大约50,000个GPU小时。我们的实验结果强调了标准化基准测试的必要性，以评估这些针对防御增强型LLM的攻击。我们的代码可以在https://github.com/usail-hkust/Bag_of_Tricks_for_LLM_Jailbreaking.上找到



## **3. JailbreakEval: An Integrated Toolkit for Evaluating Jailbreak Attempts Against Large Language Models**

越狱Eval：用于评估针对大型语言模型的越狱尝试的集成工具包 cs.CR

Our code is available at https://github.com/ThuCCSLab/JailbreakEval

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09321v1) [paper-pdf](http://arxiv.org/pdf/2406.09321v1)

**Authors**: Delong Ran, Jinyuan Liu, Yichen Gong, Jingyi Zheng, Xinlei He, Tianshuo Cong, Anyu Wang

**Abstract**: Jailbreak attacks aim to induce Large Language Models (LLMs) to generate harmful responses for forbidden instructions, presenting severe misuse threats to LLMs. Up to now, research into jailbreak attacks and defenses is emerging, however, there is (surprisingly) no consensus on how to evaluate whether a jailbreak attempt is successful. In other words, the methods to assess the harmfulness of an LLM's response are varied, such as manual annotation or prompting GPT-4 in specific ways. Each approach has its own set of strengths and weaknesses, impacting their alignment with human values, as well as the time and financial cost. This diversity in evaluation presents challenges for researchers in choosing suitable evaluation methods and conducting fair comparisons across different jailbreak attacks and defenses. In this paper, we conduct a comprehensive analysis of jailbreak evaluation methodologies, drawing from nearly ninety jailbreak research released between May 2023 and April 2024. Our study introduces a systematic taxonomy of jailbreak evaluators, offering in-depth insights into their strengths and weaknesses, along with the current status of their adaptation. Moreover, to facilitate subsequent research, we propose JailbreakEval, a user-friendly toolkit focusing on the evaluation of jailbreak attempts. It includes various well-known evaluators out-of-the-box, so that users can obtain evaluation results with only a single command. JailbreakEval also allows users to customize their own evaluation workflow in a unified framework with the ease of development and comparison. In summary, we regard JailbreakEval to be a catalyst that simplifies the evaluation process in jailbreak research and fosters an inclusive standard for jailbreak evaluation within the community.

摘要: 越狱攻击的目的是诱导大型语言模型(LLM)对禁用指令产生有害响应，给LLM带来严重的滥用威胁。到目前为止，关于越狱攻击和防御的研究正在兴起，然而，(令人惊讶的)对于如何评估越狱尝试是否成功，还没有达成共识。换句话说，评估LLM响应的危害性的方法是多种多样的，例如手动注释或以特定方式提示GPT-4。每种方法都有自己的长处和短处，影响它们与人类价值观的一致性，以及时间和财务成本。这种评估的多样性给研究人员带来了挑战，他们要选择合适的评估方法，并对不同的越狱攻击和防御进行公平的比较。本文从2023年5月至2024年4月发布的近90项越狱研究中，对越狱评估方法进行了全面的分析。我们的研究介绍了越狱评估员的系统分类，深入了解了他们的优势和劣势，以及他们适应的现状。此外，为了便于后续研究，我们提出了JailBreak Eval，这是一个用户友好的工具包，专注于评估越狱企图。它包括各种开箱即用的知名评估器，使用户只需一条命令即可获得评估结果。JailBreak Eval还允许用户在统一的框架中定制自己的评估工作流程，易于开发和比较。总而言之，我们认为越狱评估是一种催化剂，可以简化越狱研究的评估过程，并在社区内培养一个包容性的越狱评估标准。



## **4. A Survey of Backdoor Attacks and Defenses on Large Language Models: Implications for Security Measures**

大型语言模型后门攻击和防御的调查：对安全措施的影响 cs.CR

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.06852v2) [paper-pdf](http://arxiv.org/pdf/2406.06852v2)

**Authors**: Shuai Zhao, Meihuizi Jia, Zhongliang Guo, Leilei Gan, Jie Fu, Yichao Feng, Fengjun Pan, Luu Anh Tuan

**Abstract**: The large language models (LLMs), which bridge the gap between human language understanding and complex problem-solving, achieve state-of-the-art performance on several NLP tasks, particularly in few-shot and zero-shot settings. Despite the demonstrable efficacy of LMMs, due to constraints on computational resources, users have to engage with open-source language models or outsource the entire training process to third-party platforms. However, research has demonstrated that language models are susceptible to potential security vulnerabilities, particularly in backdoor attacks. Backdoor attacks are designed to introduce targeted vulnerabilities into language models by poisoning training samples or model weights, allowing attackers to manipulate model responses through malicious triggers. While existing surveys on backdoor attacks provide a comprehensive overview, they lack an in-depth examination of backdoor attacks specifically targeting LLMs. To bridge this gap and grasp the latest trends in the field, this paper presents a novel perspective on backdoor attacks for LLMs by focusing on fine-tuning methods. Specifically, we systematically classify backdoor attacks into three categories: full-parameter fine-tuning, parameter-efficient fine-tuning, and attacks without fine-tuning. Based on insights from a substantial review, we also discuss crucial issues for future research on backdoor attacks, such as further exploring attack algorithms that do not require fine-tuning, or developing more covert attack algorithms.

摘要: 大型语言模型(LLM)架起了人类语言理解和复杂问题解决之间的桥梁，在几个NLP任务上实现了最先进的性能，特别是在少镜头和零镜头的情况下。尽管LMM具有明显的功效，但由于计算资源的限制，用户不得不使用开放源码语言模型或将整个培训过程外包给第三方平台。然而，研究表明，语言模型容易受到潜在的安全漏洞的影响，特别是在后门攻击中。后门攻击旨在通过毒化训练样本或模型权重，将有针对性的漏洞引入语言模型，允许攻击者通过恶意触发器操纵模型响应。虽然现有的关于后门攻击的调查提供了全面的概述，但它们缺乏对专门针对LLM的后门攻击的深入检查。为了弥补这一差距，掌握该领域的最新趋势，本文提出了一种新的视角来研究针对LLMS的后门攻击，重点是微调方法。具体来说，我们系统地将后门攻击分为三类：全参数微调、参数高效微调和未微调攻击。在大量综述的基础上，我们还讨论了未来后门攻击研究的关键问题，如进一步探索不需要微调的攻击算法，或开发更隐蔽的攻击算法。



## **5. StructuralSleight: Automated Jailbreak Attacks on Large Language Models Utilizing Uncommon Text-Encoded Structure**

StructualSleight：利用不常见的文本编码结构对大型语言模型进行自动越狱攻击 cs.CL

12 pages, 4 figures

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.08754v1) [paper-pdf](http://arxiv.org/pdf/2406.08754v1)

**Authors**: Bangxin Li, Hengrui Xing, Chao Huang, Jin Qian, Huangqing Xiao, Linfeng Feng, Cong Tian

**Abstract**: Large Language Models (LLMs) are widely used in natural language processing but face the risk of jailbreak attacks that maliciously induce them to generate harmful content. Existing jailbreak attacks, including character-level and context-level attacks, mainly focus on the prompt of the plain text without specifically exploring the significant influence of its structure. In this paper, we focus on studying how prompt structure contributes to the jailbreak attack. We introduce a novel structure-level attack method based on tail structures that are rarely used during LLM training, which we refer to as Uncommon Text-Encoded Structure (UTES). We extensively study 12 UTESs templates and 6 obfuscation methods to build an effective automated jailbreak tool named StructuralSleight that contains three escalating attack strategies: Structural Attack, Structural and Character/Context Obfuscation Attack, and Fully Obfuscated Structural Attack. Extensive experiments on existing LLMs show that StructuralSleight significantly outperforms baseline methods. In particular, the attack success rate reaches 94.62\% on GPT-4o, which has not been addressed by state-of-the-art techniques.

摘要: 大语言模型在自然语言处理中被广泛使用，但面临着越狱攻击的风险，这些攻击会恶意诱导它们生成有害内容。现有的越狱攻击，包括字符级攻击和语境级攻击，主要集中在明文的提示上，没有具体探讨其结构的重大影响。本文主要研究提示结构在越狱攻击中的作用。提出了一种基于LLM训练中很少使用的尾部结构的结构级攻击方法，称为非公共文本编码结构(UTES)。我们深入研究了12个UTE模板和6种混淆方法，构建了一个有效的自动化越狱工具StructuralSleight，它包含三种逐步升级的攻击策略：结构攻击、结构和字符/上下文混淆攻击和完全混淆结构攻击。在现有LLMS上的大量实验表明，StructuralSleight的性能明显优于基线方法。特别是，在GPT-40上的攻击成功率达到了94.62\%，这是最新技术还没有解决的问题。



## **6. Ranking Manipulation for Conversational Search Engines**

对话式搜索引擎的排名操纵 cs.CL

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.03589v2) [paper-pdf](http://arxiv.org/pdf/2406.03589v2)

**Authors**: Samuel Pfrommer, Yatong Bai, Tanmay Gautam, Somayeh Sojoudi

**Abstract**: Major search engine providers are rapidly incorporating Large Language Model (LLM)-generated content in response to user queries. These conversational search engines operate by loading retrieved website text into the LLM context for summarization and interpretation. Recent research demonstrates that LLMs are highly vulnerable to jailbreaking and prompt injection attacks, which disrupt the safety and quality goals of LLMs using adversarial strings. This work investigates the impact of prompt injections on the ranking order of sources referenced by conversational search engines. To this end, we introduce a focused dataset of real-world consumer product websites and formalize conversational search ranking as an adversarial problem. Experimentally, we analyze conversational search rankings in the absence of adversarial injections and show that different LLMs vary significantly in prioritizing product name, document content, and context position. We then present a tree-of-attacks-based jailbreaking technique which reliably promotes low-ranked products. Importantly, these attacks transfer effectively to state-of-the-art conversational search engines such as perplexity.ai. Given the strong financial incentive for website owners to boost their search ranking, we argue that our problem formulation is of critical importance for future robustness work.

摘要: 各大搜索引擎提供商正在快速整合大型语言模型(LLM)生成的内容，以响应用户查询。这些对话式搜索引擎通过将检索到的网站文本加载到LLM上下文中进行操作以进行摘要和解释。最近的研究表明，LLM非常容易受到越狱和快速注入攻击，这些攻击使用敌意字符串破坏LLM的安全和质量目标。这项工作调查了提示注入对对话式搜索引擎引用的来源的排名顺序的影响。为此，我们引入了一个聚焦于真实世界消费产品网站的数据集，并将会话搜索排名形式化为一个对抗性问题。在实验上，我们分析了在没有对抗性注入的情况下的会话搜索排名，结果表明不同的LLM在产品名称、文档内容和上下文位置的优先顺序上存在显著差异。然后，我们提出了一种基于攻击树的越狱技术，该技术可靠地推广排名较低的产品。重要的是，这些攻击有效地转移到了最先进的会话搜索引擎，如Pplexity.ai。考虑到网站所有者有强大的经济动机来提高他们的搜索排名，我们认为我们的问题表达对于未来的稳健性工作至关重要。



## **7. RL-JACK: Reinforcement Learning-powered Black-box Jailbreaking Attack against LLMs**

RL-JACK：针对LLM的强化学习驱动的黑匣子越狱攻击 cs.CR

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.08725v1) [paper-pdf](http://arxiv.org/pdf/2406.08725v1)

**Authors**: Xuan Chen, Yuzhou Nie, Lu Yan, Yunshu Mao, Wenbo Guo, Xiangyu Zhang

**Abstract**: Modern large language model (LLM) developers typically conduct a safety alignment to prevent an LLM from generating unethical or harmful content. Recent studies have discovered that the safety alignment of LLMs can be bypassed by jailbreaking prompts. These prompts are designed to create specific conversation scenarios with a harmful question embedded. Querying an LLM with such prompts can mislead the model into responding to the harmful question. The stochastic and random nature of existing genetic methods largely limits the effectiveness and efficiency of state-of-the-art (SOTA) jailbreaking attacks. In this paper, we propose RL-JACK, a novel black-box jailbreaking attack powered by deep reinforcement learning (DRL). We formulate the generation of jailbreaking prompts as a search problem and design a novel RL approach to solve it. Our method includes a series of customized designs to enhance the RL agent's learning efficiency in the jailbreaking context. Notably, we devise an LLM-facilitated action space that enables diverse action variations while constraining the overall search space. We propose a novel reward function that provides meaningful dense rewards for the agent toward achieving successful jailbreaking. Through extensive evaluations, we demonstrate that RL-JACK is overall much more effective than existing jailbreaking attacks against six SOTA LLMs, including large open-source models and commercial models. We also show the RL-JACK's resiliency against three SOTA defenses and its transferability across different models. Finally, we validate the insensitivity of RL-JACK to the variations in key hyper-parameters.

摘要: 现代大型语言模型(LLM)开发人员通常会进行安全调整，以防止LLM生成不道德或有害的内容。最近的研究发现，越狱提示可以绕过LLMS的安全对准。这些提示旨在创建嵌入有害问题的特定对话场景。使用这样的提示查询LLM可能会误导模型响应有害的问题。现有遗传方法的随机性和随机性在很大程度上限制了最先进的越狱攻击(SOTA)的有效性和效率。本文提出了一种基于深度强化学习的新型黑盒越狱攻击算法RL-JACK。我们将越狱提示的生成描述为一个搜索问题，并设计了一种新的RL方法来解决这个问题。我们的方法包括一系列定制设计，以提高RL代理在越狱环境中的学习效率。值得注意的是，我们设计了一个LLM促进的动作空间，它可以在限制整体搜索空间的同时实现不同的动作变化。我们提出了一个新的奖励函数，为特工提供有意义的密集奖励，以实现成功的越狱。通过广泛的评估，我们证明了RL-JACK总体上比现有的针对六个Sota LLM的越狱攻击要有效得多，其中包括大型开源模型和商业模型。我们还展示了RL-Jack对三种Sota防御系统的弹性，以及它在不同型号之间的可转移性。最后，我们验证了RL-JACK对关键超参数变化的不敏感性。



## **8. Adversarial Evasion Attack Efficiency against Large Language Models**

针对大型语言模型的对抗规避攻击效率 cs.CL

9 pages, 1 table, 2 figures, DCAI 2024 conference

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08050v1) [paper-pdf](http://arxiv.org/pdf/2406.08050v1)

**Authors**: João Vitorino, Eva Maia, Isabel Praça

**Abstract**: Large Language Models (LLMs) are valuable for text classification, but their vulnerabilities must not be disregarded. They lack robustness against adversarial examples, so it is pertinent to understand the impacts of different types of perturbations, and assess if those attacks could be replicated by common users with a small amount of perturbations and a small number of queries to a deployed LLM. This work presents an analysis of the effectiveness, efficiency, and practicality of three different types of adversarial attacks against five different LLMs in a sentiment classification task. The obtained results demonstrated the very distinct impacts of the word-level and character-level attacks. The word attacks were more effective, but the character and more constrained attacks were more practical and required a reduced number of perturbations and queries. These differences need to be considered during the development of adversarial defense strategies to train more robust LLMs for intelligent text classification applications.

摘要: 大型语言模型(LLM)对于文本分类很有价值，但其脆弱性不容忽视。它们对敌意示例缺乏健壮性，因此了解不同类型扰动的影响并评估这些攻击是否可以被普通用户复制，只需少量扰动和对已部署的LLM的少量查询。本文分析了三种不同类型的对抗性攻击在情感分类任务中对五种不同的LLM的有效性、效率和实用性。所获得的结果显示了词级攻击和字级攻击的非常明显的影响。单词攻击更有效，但字符和更受约束的攻击更实用，需要的干扰和查询次数更少。在开发对抗性防御策略以训练更健壮的LLM用于智能文本分类应用时，需要考虑这些差异。



## **9. Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization**

通过目标优先级保护大型语言模型免受越狱攻击 cs.CL

ACL 2024 Main Conference

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2311.09096v2) [paper-pdf](http://arxiv.org/pdf/2311.09096v2)

**Authors**: Zhexin Zhang, Junxiao Yang, Pei Ke, Fei Mi, Hongning Wang, Minlie Huang

**Abstract**: While significant attention has been dedicated to exploiting weaknesses in LLMs through jailbreaking attacks, there remains a paucity of effort in defending against these attacks. We point out a pivotal factor contributing to the success of jailbreaks: the intrinsic conflict between the goals of being helpful and ensuring safety. Accordingly, we propose to integrate goal prioritization at both training and inference stages to counteract. Implementing goal prioritization during inference substantially diminishes the Attack Success Rate (ASR) of jailbreaking from 66.4% to 3.6% for ChatGPT. And integrating goal prioritization into model training reduces the ASR from 71.0% to 6.6% for Llama2-13B. Remarkably, even in scenarios where no jailbreaking samples are included during training, our approach slashes the ASR by half. Additionally, our findings reveal that while stronger LLMs face greater safety risks, they also possess a greater capacity to be steered towards defending against such attacks, both because of their stronger ability in instruction following. Our work thus contributes to the comprehension of jailbreaking attacks and defenses, and sheds light on the relationship between LLMs' capability and safety. Our code is available at \url{https://github.com/thu-coai/JailbreakDefense_GoalPriority}.

摘要: 虽然通过越狱攻击来利用LLMS的弱点已经引起了极大的关注，但在防御这些攻击方面仍然缺乏努力。我们指出了越狱成功的一个关键因素：提供帮助的目标与确保安全之间的内在冲突。因此，我们提出在训练和推理阶段整合目标优先级来抵消。在推理过程中实施目标优先级大大降低了ChatGPT越狱的攻击成功率(ASR)，从66.4%降至3.6%。将目标优先顺序整合到模型训练中，将Llama2-13B的ASR从71.0%降低到6.6%。值得注意的是，即使在训练期间没有包括越狱样本的情况下，我们的方法也将ASR削减了一半。此外，我们的研究结果显示，虽然较强的LLM面临更大的安全风险，但它们也具有更强的被引导来防御此类攻击的能力，这两者都是因为它们具有更强的指令遵循能力。因此，我们的工作有助于理解越狱攻击和防御，并阐明了LLMS的能力和安全之间的关系。我们的代码可以在\url{https://github.com/thu-coai/JailbreakDefense_GoalPriority}.上找到



## **10. Unique Security and Privacy Threats of Large Language Model: A Comprehensive Survey**

大型语言模型的独特安全和隐私威胁：全面调查 cs.CR

38 pages, 17 figures, 4 tables

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.07973v1) [paper-pdf](http://arxiv.org/pdf/2406.07973v1)

**Authors**: Shang Wang, Tianqing Zhu, Bo Liu, Ding Ming, Xu Guo, Dayong Ye, Wanlei Zhou

**Abstract**: With the rapid development of artificial intelligence, large language models (LLMs) have made remarkable progress in natural language processing. These models are trained on large amounts of data to demonstrate powerful language understanding and generation capabilities for various applications, from machine translation and chatbots to agents. However, LLMs have exposed a variety of privacy and security issues during their life cycle, which have become the focus of academic and industrial attention. Moreover, these risks LLMs face are pretty different from previous traditional language models. Since current surveys lack a clear taxonomy of unique threat models based on diverse scenarios, we highlight unique privacy and security issues based on five scenarios: pre-training, fine-tuning, RAG system, deploying, and LLM-based agent. Concerning the characteristics of each risk, this survey provides potential threats and countermeasures. The research on attack and defense situations LLMs face can provide feasible research directions, making more areas reap LLMs' benefits.

摘要: 随着人工智能的快速发展，大语言模型在自然语言处理方面取得了显著的进展。这些模型在大量数据上进行了训练，以展示强大的语言理解和生成能力，适用于从机器翻译、聊天机器人到代理的各种应用。然而，LLMS在其生命周期中暴露了各种隐私和安全问题，成为学术界和产业界关注的焦点。此外，LLMS面临的这些风险与以前的传统语言模型有很大不同。由于目前的调查缺乏基于不同场景的独特威胁模型的明确分类，我们基于五种场景强调了独特的隐私和安全问题：预培训、微调、RAG系统、部署和基于LLM的代理。针对每个风险的特点，本调查提供了潜在的威胁和对策。对低成本管理系统面临的攻防态势的研究可以为低成本管理系统提供可行的研究方向，使更多的地区受益于低成本管理系统。



## **11. Dataset and Lessons Learned from the 2024 SaTML LLM Capture-the-Flag Competition**

2024年SaTML LLM夺旗大赛的数据集和经验教训 cs.CR

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.07954v1) [paper-pdf](http://arxiv.org/pdf/2406.07954v1)

**Authors**: Edoardo Debenedetti, Javier Rando, Daniel Paleka, Silaghi Fineas Florin, Dragos Albastroiu, Niv Cohen, Yuval Lemberg, Reshmi Ghosh, Rui Wen, Ahmed Salem, Giovanni Cherubin, Santiago Zanella-Beguelin, Robin Schmid, Victor Klemm, Takahiro Miki, Chenhao Li, Stefan Kraft, Mario Fritz, Florian Tramèr, Sahar Abdelnabi, Lea Schönherr

**Abstract**: Large language model systems face important security risks from maliciously crafted messages that aim to overwrite the system's original instructions or leak private data. To study this problem, we organized a capture-the-flag competition at IEEE SaTML 2024, where the flag is a secret string in the LLM system prompt. The competition was organized in two phases. In the first phase, teams developed defenses to prevent the model from leaking the secret. During the second phase, teams were challenged to extract the secrets hidden for defenses proposed by the other teams. This report summarizes the main insights from the competition. Notably, we found that all defenses were bypassed at least once, highlighting the difficulty of designing a successful defense and the necessity for additional research to protect LLM systems. To foster future research in this direction, we compiled a dataset with over 137k multi-turn attack chats and open-sourced the platform.

摘要: 大型语言模型系统面临着恶意制作的消息的重要安全风险，这些消息旨在覆盖系统的原始指令或泄露私人数据。为了研究这个问题，我们在IEEE SaTML 2024上组织了一场捕获旗帜竞赛，其中旗帜是LLM系统提示符中的秘密字符串。比赛分两个阶段组织。在第一阶段，团队开发了防御措施以防止模型泄露秘密。在第二阶段，团队面临挑战，以提取其他团队提出的防御所隐藏的秘密。本报告总结了比赛的主要见解。值得注意的是，我们发现所有防御措施至少被绕过一次，这凸显了设计成功防御的难度以及进行额外研究以保护LLM系统的必要性。为了促进这一方向的未来研究，我们编制了一个包含超过137，000次多回合攻击聊天的数据集，并开源了该平台。



## **12. Visual-RolePlay: Universal Jailbreak Attack on MultiModal Large Language Models via Role-playing Image Character**

可视化角色扮演：通过角色扮演图像角色对多模式大型语言模型进行通用越狱攻击 cs.CR

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2405.20773v2) [paper-pdf](http://arxiv.org/pdf/2405.20773v2)

**Authors**: Siyuan Ma, Weidi Luo, Yu Wang, Xiaogeng Liu

**Abstract**: With the advent and widespread deployment of Multimodal Large Language Models (MLLMs), ensuring their safety has become increasingly critical. To achieve this objective, it requires us to proactively discover the vulnerability of MLLMs by exploring the attack methods. Thus, structure-based jailbreak attacks, where harmful semantic content is embedded within images, have been proposed to mislead the models. However, previous structure-based jailbreak methods mainly focus on transforming the format of malicious queries, such as converting harmful content into images through typography, which lacks sufficient jailbreak effectiveness and generalizability. To address these limitations, we first introduce the concept of "Role-play" into MLLM jailbreak attacks and propose a novel and effective method called Visual Role-play (VRP). Specifically, VRP leverages Large Language Models to generate detailed descriptions of high-risk characters and create corresponding images based on the descriptions. When paired with benign role-play instruction texts, these high-risk character images effectively mislead MLLMs into generating malicious responses by enacting characters with negative attributes. We further extend our VRP method into a universal setup to demonstrate its generalizability. Extensive experiments on popular benchmarks show that VRP outperforms the strongest baseline, Query relevant and FigStep, by an average Attack Success Rate (ASR) margin of 14.3% across all models.

摘要: 随着多通道大语言模型的出现和广泛应用，确保其安全性变得越来越重要。为了实现这一目标，需要我们通过探索攻击方法来主动发现MLLMS的脆弱性。因此，已经提出了基于结构的越狱攻击，其中有害的语义内容嵌入到图像中，以误导模型。然而，以往的基于结构的越狱方法主要集中在对恶意查询的格式进行转换，如通过排版将有害内容转换为图像，缺乏足够的越狱有效性和通用性。针对这些局限性，我们首先将“角色扮演”的概念引入到MLLM越狱攻击中，提出了一种新颖而有效的方法--视觉角色扮演(VRP)。具体地说，VRP利用大型语言模型来生成高危角色的详细描述，并基于这些描述创建相应的图像。当与良性的角色扮演指示文本配对时，这些高风险角色图像有效地误导MLLM，通过设定具有负面属性的角色来生成恶意响应。我们进一步将VRP方法扩展到一个通用的设置，以证明它的普适性。在流行基准上的广泛实验表明，VRP在所有模型上的平均攻击成功率(ASR)边际都比最强的基线、查询相关和FigStep高14.3%。



## **13. Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM**

通过强大的对齐LLM防御破坏对齐的攻击 cs.CL

19 Pages, 5 Figures, 8 Tables. Accepted by ACL 2024

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2309.14348v3) [paper-pdf](http://arxiv.org/pdf/2309.14348v3)

**Authors**: Bochuan Cao, Yuanpu Cao, Lu Lin, Jinghui Chen

**Abstract**: Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments on open-source large language models, we demonstrate that RA-LLM can successfully defend against both state-of-the-art adversarial prompts and popular handcrafted jailbreaking prompts by reducing their attack success rates from nearly 100% to around 10% or less.

摘要: 近年来，大型语言模型(LLM)取得了长足的进步，现已广泛应用于各个领域。不幸的是，人们越来越担心LLMS可能被滥用来生成有害或恶意的内容。尽管有一系列研究专注于将LLM与人类价值观保持一致，并防止它们产生不适当的内容，但这种调整通常是脆弱的，可以通过恶意优化或手工制作的越狱提示被破坏顺序的攻击绕过。在这项工作中，我们引入了一种鲁棒对齐LLM(RA-LLM)来防御潜在的对齐破坏攻击。RA-LLM可以直接构建在现有的对准LLM上，具有健壮的对准检查功能，而不需要对原始LLM进行任何昂贵的再培训或微调过程。此外，我们还对RA-LLM进行了理论分析，以验证其在抵抗对齐破坏攻击方面的有效性。通过在开源大型语言模型上的真实世界实验，我们证明了RA-LLM能够成功地防御最新的敌意提示和流行的手工越狱提示，将攻击成功率从近100%降低到10%左右或更低。



## **14. Merging Improves Self-Critique Against Jailbreak Attacks**

合并提高了对越狱袭击的自我批评 cs.CL

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.07188v1) [paper-pdf](http://arxiv.org/pdf/2406.07188v1)

**Authors**: Victor Gallego

**Abstract**: The robustness of large language models (LLMs) against adversarial manipulations, such as jailbreak attacks, remains a significant challenge. In this work, we propose an approach that enhances the self-critique capability of the LLM and further fine-tunes it over sanitized synthetic data. This is done with the addition of an external critic model that can be merged with the original, thus bolstering self-critique capabilities and improving the robustness of the LLMs response to adversarial prompts. Our results demonstrate that the combination of merging and self-critique can reduce the attack success rate of adversaries significantly, thus offering a promising defense mechanism against jailbreak attacks. Code, data and models released at https://github.com/vicgalle/merging-self-critique-jailbreaks .

摘要: 大型语言模型（LLM）对越狱攻击等对抗性操纵的稳健性仍然是一个重大挑战。在这项工作中，我们提出了一种增强LLM自我批评能力的方法，并根据净化的合成数据进一步对其进行微调。这是通过添加一个可以与原始模型合并的外部批评者模型来实现的，从而增强自我批评能力并提高LLM对对抗提示反应的稳健性。我们的结果表明，合并和自我批评的结合可以显着降低对手的攻击成功率，从而提供一种有希望的针对越狱攻击的防御机制。代码、数据和模型在https://github.com/vicgalle/merging-self-critique-jailbreaks上发布。



## **15. Benchmarking Trustworthiness of Multimodal Large Language Models: A Comprehensive Study**

多模式大型语言模型的可信度基准：全面研究 cs.CL

100 pages, 84 figures, 33 tables

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.07057v1) [paper-pdf](http://arxiv.org/pdf/2406.07057v1)

**Authors**: Yichi Zhang, Yao Huang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Yifan Wang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, Jun Zhu

**Abstract**: Despite the superior capabilities of Multimodal Large Language Models (MLLMs) across diverse tasks, they still face significant trustworthiness challenges. Yet, current literature on the assessment of trustworthy MLLMs remains limited, lacking a holistic evaluation to offer thorough insights into future improvements. In this work, we establish MultiTrust, the first comprehensive and unified benchmark on the trustworthiness of MLLMs across five primary aspects: truthfulness, safety, robustness, fairness, and privacy. Our benchmark employs a rigorous evaluation strategy that addresses both multimodal risks and cross-modal impacts, encompassing 32 diverse tasks with self-curated datasets. Extensive experiments with 21 modern MLLMs reveal some previously unexplored trustworthiness issues and risks, highlighting the complexities introduced by the multimodality and underscoring the necessity for advanced methodologies to enhance their reliability. For instance, typical proprietary models still struggle with the perception of visually confusing images and are vulnerable to multimodal jailbreaking and adversarial attacks; MLLMs are more inclined to disclose privacy in text and reveal ideological and cultural biases even when paired with irrelevant images in inference, indicating that the multimodality amplifies the internal risks from base LLMs. Additionally, we release a scalable toolbox for standardized trustworthiness research, aiming to facilitate future advancements in this important field. Code and resources are publicly available at: https://multi-trust.github.io/.

摘要: 尽管多模式大型语言模型(MLLM)在不同的任务中具有卓越的能力，但它们仍然面临着重大的可信性挑战。然而，目前关于评估值得信赖的MLLMS的文献仍然有限，缺乏全面的评估来提供对未来改进的透彻见解。在这项工作中，我们建立了多重信任，这是第一个关于MLLMS可信度的全面和统一的基准，涉及五个主要方面：真实性、安全性、健壮性、公平性和隐私性。我们的基准采用了严格的评估战略，同时应对多式联运风险和跨联运影响，包括32项不同的任务和自我管理的数据集。对21个现代多模式管理进行的广泛实验揭示了一些以前从未探索过的可信度问题和风险，突显了多模式带来的复杂性，并强调了先进方法提高其可靠性的必要性。例如，典型的专有模型仍然难以识别视觉上令人困惑的图像，容易受到多模式越狱和敌意攻击；MLLM更倾向于在文本中泄露隐私，甚至在推理中与无关图像搭配使用时也会暴露意识形态和文化偏见，这表明多模式放大了基本LLM的内部风险。此外，我们还发布了一个用于标准化可信度研究的可扩展工具箱，旨在促进这一重要领域的未来发展。代码和资源可在以下网址公开获得：https://multi-trust.github.io/.



## **16. An LLM-Assisted Easy-to-Trigger Backdoor Attack on Code Completion Models: Injecting Disguised Vulnerabilities against Strong Detection**

LLM辅助的对代码完成模型的易于触发的后门攻击：针对强检测注入伪装漏洞 cs.CR

To appear in USENIX Security '24

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06822v1) [paper-pdf](http://arxiv.org/pdf/2406.06822v1)

**Authors**: Shenao Yan, Shen Wang, Yue Duan, Hanbin Hong, Kiho Lee, Doowon Kim, Yuan Hong

**Abstract**: Large Language Models (LLMs) have transformed code completion tasks, providing context-based suggestions to boost developer productivity in software engineering. As users often fine-tune these models for specific applications, poisoning and backdoor attacks can covertly alter the model outputs. To address this critical security challenge, we introduce CodeBreaker, a pioneering LLM-assisted backdoor attack framework on code completion models. Unlike recent attacks that embed malicious payloads in detectable or irrelevant sections of the code (e.g., comments), CodeBreaker leverages LLMs (e.g., GPT-4) for sophisticated payload transformation (without affecting functionalities), ensuring that both the poisoned data for fine-tuning and generated code can evade strong vulnerability detection. CodeBreaker stands out with its comprehensive coverage of vulnerabilities, making it the first to provide such an extensive set for evaluation. Our extensive experimental evaluations and user studies underline the strong attack performance of CodeBreaker across various settings, validating its superiority over existing approaches. By integrating malicious payloads directly into the source code with minimal transformation, CodeBreaker challenges current security measures, underscoring the critical need for more robust defenses for code completion.

摘要: 大型语言模型(LLM)已经改变了代码完成任务，提供了基于上下文的建议来提高软件工程中的开发人员生产力。由于用户经常针对特定应用对这些模型进行微调，毒化和后门攻击可能会暗中更改模型输出。为了应对这一关键的安全挑战，我们引入了CodeBreaker，这是一个基于代码完成模型的开创性LLM辅助后门攻击框架。与最近在代码的可检测或无关部分(例如，注释)中嵌入恶意有效负载的攻击不同，代码破解器利用LLM(例如，GPT-4)进行复杂的有效负载转换(不影响功能)，确保用于微调的有毒数据和生成的代码都可以逃避强大的漏洞检测。CodeBreaker凭借其对漏洞的全面覆盖而脱颖而出，使其成为第一个提供如此广泛的评估集的软件。我们广泛的实验评估和用户研究强调了密码破译器在各种环境下的强大攻击性能，验证了其相对于现有方法的优越性。通过将恶意有效负载直接集成到源代码中，只需最少的转换，代码破解器挑战了当前的安全措施，突显了对代码完成更强大防御的迫切需要。



## **17. Better Safe than Sorry: Pre-training CLIP against Targeted Data Poisoning and Backdoor Attacks**

安全总比后悔好：对CLIP进行预培训以防止有针对性的数据中毒和后门攻击 cs.LG

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2310.05862v2) [paper-pdf](http://arxiv.org/pdf/2310.05862v2)

**Authors**: Wenhan Yang, Jingdong Gao, Baharan Mirzasoleiman

**Abstract**: Contrastive Language-Image Pre-training (CLIP) on large image-caption datasets has achieved remarkable success in zero-shot classification and enabled transferability to new domains. However, CLIP is extremely more vulnerable to targeted data poisoning and backdoor attacks, compared to supervised learning. Perhaps surprisingly, poisoning 0.0001% of CLIP pre-training data is enough to make targeted data poisoning attacks successful. This is four orders of magnitude smaller than what is required to poison supervised models. Despite this vulnerability, existing methods are very limited in defending CLIP models during pre-training. In this work, we propose a strong defense, SAFECLIP, to safely pre-train CLIP against targeted data poisoning and backdoor attacks. SAFECLIP warms up the model by applying unimodal contrastive learning (CL) on image and text modalities separately. Then, it divides the data into safe and risky sets, by applying a Gaussian Mixture Model to the cosine similarity of image-caption pair representations. SAFECLIP pre-trains the model by applying the CLIP loss to the safe set and applying unimodal CL to image and text modalities of the risky set separately. By gradually increasing the size of the safe set during pre-training, SAFECLIP effectively breaks targeted data poisoning and backdoor attacks without harming the CLIP performance. Our extensive experiments on CC3M, Visual Genome, and MSCOCO demonstrate that SAFECLIP significantly reduces the success rate of targeted data poisoning attacks from 93.75% to 0% and that of various backdoor attacks from up to 100% to 0%, without harming CLIP's performance.

摘要: 在大型图像字幕数据集上的对比语言图像预训练(CLIP)在零镜头分类方面取得了显着的成功，并使其能够移植到新的领域。然而，与监督学习相比，CLIP极易受到有针对性的数据中毒和后门攻击。或许令人惊讶的是，投毒0.0001的CLIP预训数据足以让定向数据投毒攻击成功。这比毒化受监督模型所需的数量小四个数量级。尽管存在这个漏洞，但现有的方法在预训练期间防御剪辑模型方面非常有限。在这项工作中，我们提出了一个强大的防御，SAFECLIP，以安全地预训练CLIP来抵御有针对性的数据中毒和后门攻击。SAFECLIP通过对图像和文本通道分别应用单峰对比学习(CL)来对模型进行预热。然后，通过将高斯混合模型应用于图像-字幕对表示的余弦相似性，将数据划分为安全集和风险集。SAFECLIP通过将剪辑损失应用于安全集合并将单峰CL分别应用于风险集合的图像和文本通道来预先训练模型。通过在预培训期间逐渐增加安全集的大小，SAFECLIP在不损害剪辑性能的情况下有效地打破了有针对性的数据中毒和后门攻击。我们在CC3M、Visual Genome和MSCOCO上的广泛实验表明，SAFECLIP在不影响CLIP性能的情况下，显著地将定向数据中毒攻击的成功率从93.75%降低到0%，将各种后门攻击的成功率从100%降低到0%。



## **18. LLM Dataset Inference: Did you train on my dataset?**

LLM数据集推理：您使用我的数据集进行训练了吗？ cs.LG

Code is available at  \href{https://github.com/pratyushmaini/llm_dataset_inference/

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06443v1) [paper-pdf](http://arxiv.org/pdf/2406.06443v1)

**Authors**: Pratyush Maini, Hengrui Jia, Nicolas Papernot, Adam Dziedzic

**Abstract**: The proliferation of large language models (LLMs) in the real world has come with a rise in copyright cases against companies for training their models on unlicensed data from the internet. Recent works have presented methods to identify if individual text sequences were members of the model's training data, known as membership inference attacks (MIAs). We demonstrate that the apparent success of these MIAs is confounded by selecting non-members (text sequences not used for training) belonging to a different distribution from the members (e.g., temporally shifted recent Wikipedia articles compared with ones used to train the model). This distribution shift makes membership inference appear successful. However, most MIA methods perform no better than random guessing when discriminating between members and non-members from the same distribution (e.g., in this case, the same period of time). Even when MIAs work, we find that different MIAs succeed at inferring membership of samples from different distributions. Instead, we propose a new dataset inference method to accurately identify the datasets used to train large language models. This paradigm sits realistically in the modern-day copyright landscape, where authors claim that an LLM is trained over multiple documents (such as a book) written by them, rather than one particular paragraph. While dataset inference shares many of the challenges of membership inference, we solve it by selectively combining the MIAs that provide positive signal for a given distribution, and aggregating them to perform a statistical test on a given dataset. Our approach successfully distinguishes the train and test sets of different subsets of the Pile with statistically significant p-values < 0.1, without any false positives.

摘要: 随着大型语言模型(LLM)在现实世界中的激增，针对利用互联网上未经许可的数据对其模型进行培训的公司的版权案件也有所增加。最近的工作提出了一些方法来识别单个文本序列是否是模型训练数据的成员，称为成员关系推理攻击(MIA)。我们证明，通过选择属于与成员不同分布的非成员(不用于训练的文本序列)(例如，与用于训练模型的文章相比，最近的维基百科文章在时间上发生了偏移)，这些MIA的表面成功被混淆了。这种分布转移使成员关系推断看起来很成功。然而，大多数MIA方法在区分来自相同分布(例如，在这种情况下，相同的时间段)的成员和非成员时，并不比随机猜测更好。即使当MIA起作用时，我们发现不同的MIA成功地推断出来自不同分布的样本的成员资格。相反，我们提出了一种新的数据集推理方法，以准确识别用于训练大型语言模型的数据集。这种模式现实地存在于现代版权领域，作者声称，LLM是针对他们编写的多个文档(如一本书)进行培训的，而不是针对一个特定的段落。虽然数据集推理与成员关系推理一样存在许多挑战，但我们通过选择性地组合为给定分布提供正信号的MIA，并将它们聚合起来对给定数据集执行统计测试来解决这一问题。我们的方法成功地区分了具有统计意义的p值<0.1的堆的不同子集的训练集和测试集，没有任何误报。



## **19. Are you still on track!? Catching LLM Task Drift with Activations**

你还在正轨上吗！？通过激活捕捉LLM任务漂移 cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.00799v2) [paper-pdf](http://arxiv.org/pdf/2406.00799v2)

**Authors**: Sahar Abdelnabi, Aideen Fay, Giovanni Cherubin, Ahmed Salem, Mario Fritz, Andrew Paverd

**Abstract**: Large Language Models (LLMs) are routinely used in retrieval-augmented applications to orchestrate tasks and process inputs from users and other sources. These inputs, even in a single LLM interaction, can come from a variety of sources, of varying trustworthiness and provenance. This opens the door to prompt injection attacks, where the LLM receives and acts upon instructions from supposedly data-only sources, thus deviating from the user's original instructions. We define this as task drift, and we propose to catch it by scanning and analyzing the LLM's activations. We compare the LLM's activations before and after processing the external input in order to detect whether this input caused instruction drift. We develop two probing methods and find that simply using a linear classifier can detect drift with near perfect ROC AUC on an out-of-distribution test set. We show that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Our setup does not require any modification of the LLM (e.g., fine-tuning) or any text generation, thus maximizing deployability and cost efficiency and avoiding reliance on unreliable model output. To foster future research on activation-based task inspection, decoding, and interpretability, we will release our large-scale TaskTracker toolkit, comprising a dataset of over 500K instances, representations from 4 SoTA language models, and inspection tools.

摘要: 大型语言模型(LLM)通常用于检索增强的应用程序中，以协调任务并处理来自用户和其他来源的输入。这些输入，即使是在单个LLM交互中，也可以来自各种来源，具有不同的可信度和出处。这为即时注入攻击打开了大门，在这种情况下，LLM接收来自假定仅限数据的来源的指令并对其采取行动，从而偏离用户的原始指令。我们将其定义为任务漂移，并建议通过扫描和分析LLM的激活来捕获它。我们比较LLM在处理外部输入之前和之后的激活，以检测该输入是否导致指令漂移。我们开发了两种探测方法，发现简单地使用线性分类器可以在非分布测试集上以接近完美的ROC AUC来检测漂移。我们表明，这种方法对于看不见的任务领域(如提示注入、越狱和恶意指令)的泛化效果出奇地好，而且没有接受过任何这些攻击的培训。我们的设置不需要对LLM进行任何修改(例如，微调)或任何文本生成，从而最大限度地提高可部署性和成本效益，并避免依赖不可靠的模型输出。为了促进未来对基于激活的任务检测、解码和可解释性的研究，我们将发布我们的大型TaskTracker工具包，其中包括超过50万个实例的数据集、来自4个SOTA语言模型的表示和检测工具。



## **20. Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction**

让他们问答：通过伪装和重建在很短的时间内越狱大型语言模型 cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2402.18104v2) [paper-pdf](http://arxiv.org/pdf/2402.18104v2)

**Authors**: Tong Liu, Yingjie Zhang, Zhe Zhao, Yinpeng Dong, Guozhu Meng, Kai Chen

**Abstract**: In recent years, large language models (LLMs) have demonstrated notable success across various tasks, but the trustworthiness of LLMs is still an open problem. One specific threat is the potential to generate toxic or harmful responses. Attackers can craft adversarial prompts that induce harmful responses from LLMs. In this work, we pioneer a theoretical foundation in LLMs security by identifying bias vulnerabilities within the safety fine-tuning and design a black-box jailbreak method named DRA (Disguise and Reconstruction Attack), which conceals harmful instructions through disguise and prompts the model to reconstruct the original harmful instruction within its completion. We evaluate DRA across various open-source and closed-source models, showcasing state-of-the-art jailbreak success rates and attack efficiency. Notably, DRA boasts a 91.1% attack success rate on OpenAI GPT-4 chatbot.

摘要: 近年来，大型语言模型（LLM）在各种任务中取得了显着的成功，但LLM的可信度仍然是一个悬而未决的问题。一个具体的威胁是可能产生有毒或有害反应。攻击者可以设计对抗性提示，引发LLM的有害反应。在这项工作中，我们通过识别安全微调中的偏见漏洞，开创了LLM安全的理论基础，并设计了一种名为“伪装和重建攻击”的黑匣子越狱方法，通过伪装隐藏有害指令，并促使模型在完成时重建原始有害指令。我们评估各种开源和开源模型的NPS，展示最先进的越狱成功率和攻击效率。值得注意的是，Inbox对OpenAI GPT-4聊天机器人的攻击成功率为91.1%。



## **21. CARES: A Comprehensive Benchmark of Trustworthiness in Medical Vision Language Models**

CARES：医学视觉语言模型可信度的综合基准 cs.LG

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06007v1) [paper-pdf](http://arxiv.org/pdf/2406.06007v1)

**Authors**: Peng Xia, Ze Chen, Juanxi Tian, Yangrui Gong, Ruibo Hou, Yue Xu, Zhenbang Wu, Zhiyuan Fan, Yiyang Zhou, Kangyu Zhu, Wenhao Zheng, Zhaoyang Wang, Xiao Wang, Xuchao Zhang, Chetan Bansal, Marc Niethammer, Junzhou Huang, Hongtu Zhu, Yun Li, Jimeng Sun, Zongyuan Ge, Gang Li, James Zou, Huaxiu Yao

**Abstract**: Artificial intelligence has significantly impacted medical applications, particularly with the advent of Medical Large Vision Language Models (Med-LVLMs), sparking optimism for the future of automated and personalized healthcare. However, the trustworthiness of Med-LVLMs remains unverified, posing significant risks for future model deployment. In this paper, we introduce CARES and aim to comprehensively evaluate the Trustworthiness of Med-LVLMs across the medical domain. We assess the trustworthiness of Med-LVLMs across five dimensions, including trustfulness, fairness, safety, privacy, and robustness. CARES comprises about 41K question-answer pairs in both closed and open-ended formats, covering 16 medical image modalities and 27 anatomical regions. Our analysis reveals that the models consistently exhibit concerns regarding trustworthiness, often displaying factual inaccuracies and failing to maintain fairness across different demographic groups. Furthermore, they are vulnerable to attacks and demonstrate a lack of privacy awareness. We publicly release our benchmark and code in https://github.com/richard-peng-xia/CARES.

摘要: 人工智能对医疗应用产生了重大影响，特别是随着医学大视觉语言模型(Med-LVLMS)的出现，引发了对自动化和个性化医疗保健未来的乐观情绪。然而，MED-LVLMS的可信性仍未得到验证，对未来的模型部署构成重大风险。在本文中，我们引入CARE，旨在全面评估医学领域的MED-LVLMS的可信性。我们从可信性、公平性、安全性、隐私性和健壮性五个维度评估Med-LVLM的可信性。CARE包括约41K个封闭式和开放式格式的问答对，涵盖16种医学影像模式和27个解剖区域。我们的分析表明，这些模型始终表现出对可信度的担忧，经常表现出事实上的不准确，并且未能在不同的人口群体中保持公平。此外，他们很容易受到攻击，并表现出缺乏隐私意识。我们在https://github.com/richard-peng-xia/CARES.中公开发布我们的基准测试和代码



## **22. Chain-of-Scrutiny: Detecting Backdoor Attacks for Large Language Models**

审查链：检测大型语言模型的后门攻击 cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.05948v1) [paper-pdf](http://arxiv.org/pdf/2406.05948v1)

**Authors**: Xi Li, Yusen Zhang, Renze Lou, Chen Wu, Jiaqi Wang

**Abstract**: Backdoor attacks present significant threats to Large Language Models (LLMs), particularly with the rise of third-party services that offer API integration and prompt engineering. Untrustworthy third parties can plant backdoors into LLMs and pose risks to users by embedding malicious instructions into user queries. The backdoor-compromised LLM will generate malicious output when and input is embedded with a specific trigger predetermined by an attacker. Traditional defense strategies, which primarily involve model parameter fine-tuning and gradient calculation, are inadequate for LLMs due to their extensive computational and clean data requirements. In this paper, we propose a novel solution, Chain-of-Scrutiny (CoS), to address these challenges. Backdoor attacks fundamentally create a shortcut from the trigger to the target output, thus lack reasoning support. Accordingly, CoS guides the LLMs to generate detailed reasoning steps for the input, then scrutinizes the reasoning process to ensure consistency with the final answer. Any inconsistency may indicate an attack. CoS only requires black-box access to LLM, offering a practical defense, particularly for API-accessible LLMs. It is user-friendly, enabling users to conduct the defense themselves. Driven by natural language, the entire defense process is transparent to users. We validate the effectiveness of CoS through extensive experiments across various tasks and LLMs. Additionally, experiments results shows CoS proves more beneficial for more powerful LLMs.

摘要: 后门攻击对大型语言模型(LLM)构成了重大威胁，特别是随着提供API集成和快速工程的第三方服务的兴起。不可信的第三方可以在LLMS中植入后门，并通过在用户查询中嵌入恶意指令来给用户带来风险。当AND INPUT嵌入攻击者预先确定的特定触发器时，受后门攻击的LLM将生成恶意输出。传统的防御策略主要涉及模型参数微调和梯度计算，但由于其计算量大、数据清晰，不适用于LLMS。在本文中，我们提出了一种新的解决方案，即审查链(CoS)，以应对这些挑战。后门攻击从根本上创建了一条从触发器到目标输出的捷径，因此缺乏推理支持。相应地，COS指导LLMS为输入生成详细的推理步骤，然后仔细检查推理过程以确保与最终答案一致。任何不一致都可能表示发生了攻击。CoS只需要黑盒访问LLM，提供了实用的防御，特别是对于API可访问的LLM。它是用户友好的，使用户能够自己进行防御。在自然语言的驱动下，整个防御过程对用户是透明的。我们通过跨各种任务和LLM的大量实验验证了CoS的有效性。此外，实验结果表明，CoS被证明对更强大的LLM更有利。



## **23. Safety Alignment Should Be Made More Than Just a Few Tokens Deep**

安全调整不应仅仅深入一些代币 cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.05946v1) [paper-pdf](http://arxiv.org/pdf/2406.05946v1)

**Authors**: Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal, Peter Henderson

**Abstract**: The safety alignment of current Large Language Models (LLMs) is vulnerable. Relatively simple attacks, or even benign fine-tuning, can jailbreak aligned models. We argue that many of these vulnerabilities are related to a shared underlying issue: safety alignment can take shortcuts, wherein the alignment adapts a model's generative distribution primarily over only its very first few output tokens. We refer to this issue as shallow safety alignment. In this paper, we present case studies to explain why shallow safety alignment can exist and provide evidence that current aligned LLMs are subject to this issue. We also show how these findings help explain multiple recently discovered vulnerabilities in LLMs, including the susceptibility to adversarial suffix attacks, prefilling attacks, decoding parameter attacks, and fine-tuning attacks. Importantly, we discuss how this consolidated notion of shallow safety alignment sheds light on promising research directions for mitigating these vulnerabilities. For instance, we show that deepening the safety alignment beyond just the first few tokens can often meaningfully improve robustness against some common exploits. Finally, we design a regularized finetuning objective that makes the safety alignment more persistent against fine-tuning attacks by constraining updates on initial tokens. Overall, we advocate that future safety alignment should be made more than just a few tokens deep.

摘要: 当前大型语言模型(LLM)的安全对齐是易受攻击的。相对简单的攻击，甚至是温和的微调，都可以让结盟的模型越狱。我们认为，这些漏洞中的许多都与一个共同的潜在问题有关：安全对齐可以走捷径，其中对齐主要适应模型的生成性分布，仅在其最初的几个输出令牌上。我们将这个问题称为浅层安全对准。在这篇文章中，我们提供了案例研究来解释为什么浅层安全对准可以存在，并提供证据表明当前对准的LLM受到这个问题的影响。我们还展示了这些发现如何帮助解释LLMS中最近发现的多个漏洞，包括对敌意后缀攻击、预填充攻击、解码参数攻击和微调攻击的敏感性。重要的是，我们讨论了浅层安全对齐这一统一概念如何揭示了缓解这些漏洞的有前途的研究方向。例如，我们表明，除了最初的几个令牌之外，深化安全对齐通常可以有意义地提高对一些常见漏洞的健壮性。最后，我们设计了一个正则化的精调目标，通过限制对初始令牌的更新，使安全对齐更持久地抵抗微调攻击。总体而言，我们主张未来的安全调整应该不仅仅是几个标志的深度。



## **24. Fight Back Against Jailbreaking via Prompt Adversarial Tuning**

通过即时对抗调整反击越狱 cs.LG

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2402.06255v2) [paper-pdf](http://arxiv.org/pdf/2402.06255v2)

**Authors**: Yichuan Mo, Yuji Wang, Zeming Wei, Yisen Wang

**Abstract**: While Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to jailbreak attacks. Several primary defense strategies have been proposed to protect LLMs from producing harmful information, mostly with a particular focus on harmful content filtering or heuristical defensive prompt designs. However, how to achieve intrinsic robustness through the prompts remains an open problem. In this paper, motivated by adversarial training paradigms for achieving reliable robustness, we propose an approach named Prompt Adversarial Tuning (PAT) that trains a prompt control attached to the user prompt as a guard prefix. To achieve our defense goal whilst maintaining natural performance, we optimize the control prompt with both adversarial and benign prompts. Comprehensive experiments show that our method is effective against both black-box and white-box attacks, reducing the success rate of advanced attacks to nearly 0 while maintaining the model's utility on the benign task. The proposed defense strategy incurs only negligible computational overhead, charting a new perspective for future explorations in LLM security. Our code is available at https://github.com/rain152/PAT.

摘要: 虽然大型语言模型(LLM)在各种应用中取得了巨大的成功，但它们也容易受到越狱攻击。已经提出了几种主要的防御策略来保护LLMS免受有害信息的影响，主要集中在有害内容过滤或启发式防御提示设计上。然而，如何通过提示实现内在的稳健性仍然是一个悬而未决的问题。受实现可靠健壮性的对抗性训练范例的启发，我们提出了一种称为即时对抗性调整(PAT)的方法，该方法将附加在用户提示上的提示控制训练为保卫前缀。为了在保持自然表现的同时实现我们的防守目标，我们优化了控制提示，包括对抗性提示和良性提示。综合实验表明，该方法对黑盒攻击和白盒攻击都是有效的，在保持模型对良性任务的实用性的同时，将高级攻击的成功率降低到近0。所提出的防御策略只需要很少的计算开销，为未来在LLM安全方面的探索开辟了新的前景。我们的代码可以在https://github.com/rain152/PAT.上找到



## **25. Adaptive Text Watermark for Large Language Models**

大型语言模型的自适应文本水印 cs.CL

ICML2024

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2401.13927v2) [paper-pdf](http://arxiv.org/pdf/2401.13927v2)

**Authors**: Yepeng Liu, Yuheng Bu

**Abstract**: The advancement of Large Language Models (LLMs) has led to increasing concerns about the misuse of AI-generated text, and watermarking for LLM-generated text has emerged as a potential solution. However, it is challenging to generate high-quality watermarked text while maintaining strong security, robustness, and the ability to detect watermarks without prior knowledge of the prompt or model. This paper proposes an adaptive watermarking strategy to address this problem. To improve the text quality and maintain robustness, we adaptively add watermarking to token distributions with high entropy measured using an auxiliary model and keep the low entropy token distributions untouched. For the sake of security and to further minimize the watermark's impact on text quality, instead of using a fixed green/red list generated from a random secret key, which can be vulnerable to decryption and forgery, we adaptively scale up the output logits in proportion based on the semantic embedding of previously generated text using a well designed semantic mapping model. Our experiments involving various LLMs demonstrate that our approach achieves comparable robustness performance to existing watermark methods. Additionally, the text generated by our method has perplexity comparable to that of \emph{un-watermarked} LLMs while maintaining security even under various attacks.

摘要: 大型语言模型(LLM)的发展引起了人们对人工智能生成文本滥用的日益关注，LLM生成文本的水印已经成为一种潜在的解决方案。然而，在不预先知道提示或模型的情况下，在保持强大的安全性、健壮性和检测水印的能力的同时，生成高质量的水印文本是具有挑战性的。针对这一问题，本文提出了一种自适应水印策略。为了提高文本质量和保持稳健性，我们在使用辅助模型测量的高熵的令牌分布上自适应地添加水印，而对低熵的令牌分布保持不变。为了保证安全性，并进一步减小水印对文本质量的影响，不使用由随机密钥生成的易被解密和伪造的固定绿/红列表，而是使用设计良好的语义映射模型，基于先前生成的文本的语义嵌入，按比例自适应地放大输出逻辑。实验表明，该方法具有与现有水印方法相当的稳健性。此外，该方法生成的文本具有与未加水印的LLMS相当的复杂性，同时即使在各种攻击下也能保持安全性。



## **26. SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner**

SelfDefend：LLM可以以实用的方式保护自己免受越狱的侵害 cs.CR

This paper completes its earlier vision paper, available at  arXiv:2402.15727

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2406.05498v1) [paper-pdf](http://arxiv.org/pdf/2406.05498v1)

**Authors**: Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, Juergen Rahmel

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into four major categories: optimization-based attacks such as Greedy Coordinate Gradient (GCG), jailbreak template-based attacks such as "Do-Anything-Now", advanced indirect attacks like DrAttack, and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delay to user prompts, as well as be compatible with both open-source and closed-source LLMs.   Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM defense instance to concurrently protect the target LLM instance in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs (both target and defense LLMs) have the capability to identify harmful prompts or intentions in user queries, which we empirically validate using the commonly used GPT-3.5/4 models across all major jailbreak attacks. Our measurements show that SelfDefend enables GPT-3.5 to suppress the attack success rate (ASR) by 8.97-95.74% (average: 60%) and GPT-4 by even 36.36-100% (average: 83%), while incurring negligible effects on normal queries. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. These models outperform four SOTA defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. We also empirically show that the tuned models are robust to targeted GCG and prompt injection attacks.

摘要: 越狱是一种新兴的对抗性攻击，它绕过了现成的大型语言模型(LLM)中部署的安全对齐，并已演变为四大类：基于优化的攻击(如贪婪坐标梯度(GCG))、基于模板的攻击(如Do-Anything-Now)、高级间接攻击(如DrAttack)和多语言越狱。然而，提供实际的越狱防御是具有挑战性的，因为它不仅需要处理所有上述越狱攻击，还需要对用户提示造成可以忽略不计的延迟，以及与开源和闭源LLM兼容。受传统影子堆栈安全概念防御内存溢出攻击的启发，提出了一种通用的LLM越狱防御框架--SelfDefend，该框架通过建立一个影子LLM防御实例来同时保护正常堆栈中的目标LLM实例，并与其协作进行基于检查点的访问控制。SelfDefend的有效性建立在我们的观察基础上，即现有的LLM(目标和防御LLM)能够识别用户查询中的有害提示或意图，我们使用所有主要越狱攻击中常用的GPT-3.5/4模型进行了经验验证。我们的测试表明，SelfDefend使GPT-3.5的攻击成功率(ASR)降低了8.97-95.74%(平均：60%)，GPT-4的攻击成功率(ASR)降低了36.36-100%(平均：83%)，而对正常查询的影响可以忽略不计。为了进一步提高防御的健壮性并将成本降至最低，我们使用数据蒸馏方法来优化专用的开源防御模型。这些型号的性能超过了四种SOTA防御系统，并与基于GPT-4的SelfDefend的性能相当，额外延迟明显更低。我们的实验还表明，调整后的模型对目标GCG攻击和即时注入攻击具有较强的鲁棒性。



## **27. MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models**

MM-SafetyBench：多模式大型语言模型安全评估的基准 cs.CV

The datasets were incomplete as they did not include all the  necessary copyrights

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2311.17600v3) [paper-pdf](http://arxiv.org/pdf/2311.17600v3)

**Authors**: Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Multimodal Large Language Models (MLLMs) remains understudied. In this paper, we observe that Multimodal Large Language Models (MLLMs) can be easily compromised by query-relevant images, as if the text query itself were malicious. To address this, we introduce MM-SafetyBench, a comprehensive framework designed for conducting safety-critical evaluations of MLLMs against such image-based manipulations. We have compiled a dataset comprising 13 scenarios, resulting in a total of 5,040 text-image pairs. Our analysis across 12 state-of-the-art models reveals that MLLMs are susceptible to breaches instigated by our approach, even when the equipped LLMs have been safety-aligned. In response, we propose a straightforward yet effective prompting strategy to enhance the resilience of MLLMs against these types of attacks. Our work underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source MLLMs against potential malicious exploits. The resource is available at \href{this https URL}{https://github.com/isXinLiu/MM-SafetyBench}.

摘要: 围绕大语言模型的安全问题已经得到了广泛的研究，但多模式大语言模型的安全性仍未得到充分的研究。在本文中，我们观察到多模式大型语言模型(MLLMS)很容易被与查询相关的图像破坏，就好像文本查询本身是恶意的一样。为了解决这一问题，我们引入了MM-SafetyBch，这是一个全面的框架，旨在针对此类基于图像的操作对MLLMS进行安全关键评估。我们汇编了一个包含13个场景的数据集，总共产生了5,040个文本-图像对。我们对12种最先进型号的分析表明，即使配备的LLM已经安全对准，MLLM也容易受到我们的方法引发的漏洞的影响。对此，我们提出了一种简单而有效的提示策略，以增强MLLMS对这些类型攻击的弹性。我们的工作强调了需要齐心协力加强和改进开放源码MLLM的安全措施，以防范潜在的恶意利用。该资源位于\href{此HTTPS URL}{https://github.com/isXinLiu/MM-SafetyBench}.



## **28. One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models**

一个扰动就足够了：关于针对视觉语言预训练模型生成普遍对抗性扰动 cs.CV

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2406.05491v1) [paper-pdf](http://arxiv.org/pdf/2406.05491v1)

**Authors**: Hao Fang, Jiawei Kong, Wenbo Yu, Bin Chen, Jiawei Li, Shutao Xia, Ke Xu

**Abstract**: Vision-Language Pre-training (VLP) models trained on large-scale image-text pairs have demonstrated unprecedented capability in many practical applications. However, previous studies have revealed that VLP models are vulnerable to adversarial samples crafted by a malicious adversary. While existing attacks have achieved great success in improving attack effect and transferability, they all focus on instance-specific attacks that generate perturbations for each input sample. In this paper, we show that VLP models can be vulnerable to a new class of universal adversarial perturbation (UAP) for all input samples. Although initially transplanting existing UAP algorithms to perform attacks showed effectiveness in attacking discriminative models, the results were unsatisfactory when applied to VLP models. To this end, we revisit the multimodal alignments in VLP model training and propose the Contrastive-training Perturbation Generator with Cross-modal conditions (C-PGC). Specifically, we first design a generator that incorporates cross-modal information as conditioning input to guide the training. To further exploit cross-modal interactions, we propose to formulate the training objective as a multimodal contrastive learning paradigm based on our constructed positive and negative image-text pairs. By training the conditional generator with the designed loss, we successfully force the adversarial samples to move away from its original area in the VLP model's feature space, and thus essentially enhance the attacks. Extensive experiments show that our method achieves remarkable attack performance across various VLP models and Vision-and-Language (V+L) tasks. Moreover, C-PGC exhibits outstanding black-box transferability and achieves impressive results in fooling prevalent large VLP models including LLaVA and Qwen-VL.

摘要: 在大规模图文对上训练的视觉语言预训练(VLP)模型在许多实际应用中表现出了前所未有的能力。然而，以前的研究表明，VLP模型容易受到恶意对手制作的敌意样本的攻击。虽然现有的攻击在提高攻击效果和可转移性方面取得了很大的成功，但它们都专注于针对特定实例的攻击，这些攻击会对每个输入样本产生扰动。在本文中，我们证明了VLP模型对于所有输入样本都可能受到一类新的通用对抗性摄动(UAP)的影响。虽然最初移植现有的UAP算法进行攻击对区分模型的攻击是有效的，但将其应用于VLP模型时，效果并不理想。为此，我们回顾了VLP模型训练中的多模式对齐，并提出了具有跨模式条件的对比训练扰动生成器(C-PGC)。具体地说，我们首先设计了一个生成器，它结合了跨通道信息作为条件输入来指导训练。为了进一步开发跨通道互动，我们建议将培训目标制定为基于我们构建的正面和负面图文对的多通道对比学习范式。通过用设计的损失训练条件生成器，我们成功地迫使敌方样本在VLP模型的特征空间中离开其原始区域，从而从根本上增强了攻击。大量的实验表明，我们的方法在不同的视觉和语言(V+L)任务上取得了显著的攻击性能。此外，C-PGC表现出出色的黑盒可转移性，并在愚弄LLaVA和Qwen-VL等流行的大型VLP模型方面取得了令人印象深刻的结果。



## **29. PRSA: PRompt Stealing Attacks against Large Language Models**

PRSA：针对大型语言模型的PRompt窃取攻击 cs.CR

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2402.19200v2) [paper-pdf](http://arxiv.org/pdf/2402.19200v2)

**Authors**: Yong Yang, Changjiang Li, Yi Jiang, Xi Chen, Haoyu Wang, Xuhong Zhang, Zonghui Wang, Shouling Ji

**Abstract**: In recent years, "prompt as a service" has greatly enhanced the utility of large language models (LLMs) by enabling them to perform various downstream tasks efficiently without fine-tuning. This has also increased the commercial value of prompts. However, the potential risk of leakage in these commercialized prompts remains largely underexplored. In this paper, we introduce a novel attack framework, PRSA, designed for prompt stealing attacks against LLMs. The main idea of PRSA is to infer the intent behind a prompt by analyzing its input-output content, enabling the generation of a surrogate prompt that replicates the original's functionality. Specifically, PRSA mainly consists of two key phases: prompt mutation and prompt pruning. In the mutation phase, we propose a prompt attention algorithm based on output difference. The algorithm facilitates the generation of effective surrogate prompts by learning key factors that influence the accurate inference of prompt intent. During the pruning phase, we employ a two-step related word identification strategy to detect and mask words that are highly related to the input, thus improving the generalizability of the surrogate prompts. We verify the actual threat of PRSA through evaluation in both real-world settings, non-interactive and interactive prompt services. The results strongly confirm the PRSA's effectiveness and generalizability. We have reported these findings to prompt service providers and actively collaborate with them to implement defensive measures.

摘要: 近年来，“提示即服务”极大地增强了大型语言模型(LLM)的实用性，使它们能够高效地执行各种下游任务，而无需微调。这也增加了提示的商业价值。然而，这些商业化提示中的潜在泄漏风险在很大程度上仍然没有得到充分的探索。本文提出了一种新的攻击框架--PRSA，用于快速窃取LLMS攻击。PRSA的主要思想是通过分析提示的输入输出内容来推断提示背后的意图，从而能够生成复制原始提示功能的代理提示。具体而言，PRSA算法主要由两个关键阶段组成：快速突变和快速剪枝。在变异阶段，我们提出了一种基于输出差异的快速注意算法。该算法通过学习影响提示意图准确推理的关键因素，有助于生成有效的代理提示。在剪枝阶段，我们采用了两步关联词识别策略来检测和掩蔽与输入高度相关的词，从而提高了代理提示的泛化能力。我们通过在真实世界环境下、非交互和交互提示服务中的评估来验证PRSA的实际威胁。结果有力地证实了PRSA的有效性和泛化能力。我们已将这些发现报告给服务提供商，并积极与他们合作实施防御措施。



## **30. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

利用剩余流激活分析防御大型语言模型免受攻击 cs.CR

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2406.03230v2) [paper-pdf](http://arxiv.org/pdf/2406.03230v2)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.

摘要: 大型语言模型(LLM)的广泛采用，如OpenAI的ChatGPT，使防御这些模型上的对手威胁成为当务之急。这些攻击通过引入恶意输入来操纵LLM的输出，破坏了模型的完整性和用户对其输出的信任。为了应对这一挑战，我们的论文提出了一种创新的防御策略，在白盒访问LLM的情况下，该策略利用LLM变压器层之间的剩余激活分析。我们应用了一种新的方法来分析残留流中独特的激活模式，以进行攻击提示分类。我们精选了多个数据集，以演示此分类方法如何在多种类型的攻击场景中具有高精度，包括我们新创建的攻击数据集。此外，我们通过集成LLMS的安全微调技术来增强模型的弹性，以衡量其对我们检测攻击的能力的影响。这些结果强调了我们的方法在加强对敌对输入的检测和缓解、推进LLMS运作的安全框架方面的有效性。



## **31. ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs**

ArtPrompt：针对对齐的LLM的基于ASC艺术的越狱攻击 cs.CL

To appear in ACL 2024

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2402.11753v4) [paper-pdf](http://arxiv.org/pdf/2402.11753v4)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Zhen Xiang, Bhaskar Ramasubramanian, Bo Li, Radha Poovendran

**Abstract**: Safety is critical to the usage of large language models (LLMs). Multiple techniques such as data filtering and supervised fine-tuning have been developed to strengthen LLM safety. However, currently known techniques presume that corpora used for safety alignment of LLMs are solely interpreted by semantics. This assumption, however, does not hold in real-world applications, which leads to severe vulnerabilities in LLMs. For example, users of forums often use ASCII art, a form of text-based art, to convey image information. In this paper, we propose a novel ASCII art-based jailbreak attack and introduce a comprehensive benchmark Vision-in-Text Challenge (ViTC) to evaluate the capabilities of LLMs in recognizing prompts that cannot be solely interpreted by semantics. We show that five SOTA LLMs (GPT-3.5, GPT-4, Gemini, Claude, and Llama2) struggle to recognize prompts provided in the form of ASCII art. Based on this observation, we develop the jailbreak attack ArtPrompt, which leverages the poor performance of LLMs in recognizing ASCII art to bypass safety measures and elicit undesired behaviors from LLMs. ArtPrompt only requires black-box access to the victim LLMs, making it a practical attack. We evaluate ArtPrompt on five SOTA LLMs, and show that ArtPrompt can effectively and efficiently induce undesired behaviors from all five LLMs. Our code is available at https://github.com/uw-nsl/ArtPrompt.

摘要: 安全对于大型语言模型(LLM)的使用至关重要。已经开发了多种技术，如数据过滤和有监督的微调，以加强LLM的安全性。然而，目前已知的技术假定用于LLM的安全对准的语料库仅由语义解释。然而，这一假设在现实世界的应用程序中并不成立，这导致了LLMS中的严重漏洞。例如，论坛的用户经常使用ASCII艺术，这是一种基于文本的艺术形式，以传达图像信息。本文提出了一种新的基于ASCII ART的越狱攻击方法，并引入了一个综合基准的文本中视觉挑战(VITC)来评估LLMS在识别不能完全由语义解释的提示方面的能力。我们发现，五个SOTA LLM(GPT-3.5、GPT-4、双子座、克劳德和Llama2)很难识别以ASCII ART形式提供的提示。基于这种观察，我们开发了越狱攻击ArtPrompt，它利用LLMS在识别ASCII ART方面的较差性能来绕过安全措施，并从LLM引发不希望看到的行为。ArtPrompt只需要黑盒访问受攻击的LLM，这使其成为一种实际的攻击。我们在五个SOTA LLM上对ArtPrompt进行了评估，结果表明，ArtPrompt可以有效和高效地诱导所有五个LLM的不良行为。我们的代码可以在https://github.com/uw-nsl/ArtPrompt.上找到



## **32. SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding**

SafeDecoding：通过安全意识解码防御越狱攻击 cs.CR

To appear in ACL 2024

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2402.08983v3) [paper-pdf](http://arxiv.org/pdf/2402.08983v3)

**Authors**: Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bill Yuchen Lin, Radha Poovendran

**Abstract**: As large language models (LLMs) become increasingly integrated into real-world applications such as code generation and chatbot assistance, extensive efforts have been made to align LLM behavior with human values, including safety. Jailbreak attacks, aiming to provoke unintended and unsafe behaviors from LLMs, remain a significant/leading LLM safety threat. In this paper, we aim to defend LLMs against jailbreak attacks by introducing SafeDecoding, a safety-aware decoding strategy for LLMs to generate helpful and harmless responses to user queries. Our insight in developing SafeDecoding is based on the observation that, even though probabilities of tokens representing harmful contents outweigh those representing harmless responses, safety disclaimers still appear among the top tokens after sorting tokens by probability in descending order. This allows us to mitigate jailbreak attacks by identifying safety disclaimers and amplifying their token probabilities, while simultaneously attenuating the probabilities of token sequences that are aligned with the objectives of jailbreak attacks. We perform extensive experiments on five LLMs using six state-of-the-art jailbreak attacks and four benchmark datasets. Our results show that SafeDecoding significantly reduces the attack success rate and harmfulness of jailbreak attacks without compromising the helpfulness of responses to benign user queries. SafeDecoding outperforms six defense methods.

摘要: 随着大型语言模型(LLM)越来越多地集成到真实世界的应用中，如代码生成和聊天机器人辅助，人们已经做出了广泛的努力来使LLM的行为与包括安全在内的人类价值观保持一致。越狱攻击旨在挑起LLM的意外和不安全行为，仍然是LLM的重大/主要安全威胁。在本文中，我们的目标是通过引入SafeDecoding来防御LLMS的越狱攻击，SafeDecoding是一种安全感知的解码策略，用于LLMS对用户查询生成有用和无害的响应。我们开发SafeDecoding的洞察力基于这样的观察：即使代表有害内容的令牌的概率大于代表无害响应的令牌的概率，但在按概率降序对令牌进行排序后，安全免责声明仍会出现在排名最靠前的令牌中。这使我们能够通过识别安全免责声明并放大其令牌概率来缓解越狱攻击，同时降低与越狱攻击目标一致的令牌序列的概率。我们使用六个最先进的越狱攻击和四个基准数据集在五个LLM上进行了广泛的实验。我们的结果表明，SafeDecoding在不影响对良性用户查询的响应的帮助的情况下，显著降低了越狱攻击的攻击成功率和危害性。安全解码的性能超过了六种防御方法。



## **33. DepsRAG: Towards Managing Software Dependencies using Large Language Models**

DepsRAG：使用大型语言模型管理软件附属机构 cs.SE

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2405.20455v3) [paper-pdf](http://arxiv.org/pdf/2405.20455v3)

**Authors**: Mohannad Alhanahnah, Yazan Boshmaf, Benoit Baudry

**Abstract**: Managing software dependencies is a crucial maintenance task in software development and is becoming a rapidly growing research field, especially in light of the significant increase in software supply chain attacks. Specialized expertise and substantial developer effort are required to fully comprehend dependencies and reveal hidden properties about the dependencies (e.g., number of dependencies, dependency chains, depth of dependencies).   Recent advancements in Large Language Models (LLMs) allow the retrieval of information from various data sources for response generation, thus providing a new opportunity to uniquely manage software dependencies. To highlight the potential of this technology, we present~\tool, a proof-of-concept Retrieval Augmented Generation (RAG) approach that constructs direct and transitive dependencies of software packages as a Knowledge Graph (KG) in four popular software ecosystems. DepsRAG can answer user questions about software dependencies by automatically generating necessary queries to retrieve information from the KG, and then augmenting the input of LLMs with the retrieved information. DepsRAG can also perform Web search to answer questions that the LLM cannot directly answer via the KG. We identify tangible benefits that DepsRAG can offer and discuss its limitations.

摘要: 管理软件依赖关系是软件开发中一项重要的维护任务，特别是在软件供应链攻击显著增加的情况下，软件依赖关系管理正在成为一个快速增长的研究领域。要完全理解依赖关系并揭示依赖关系的隐藏属性(例如，依赖关系的数量、依赖关系链、依赖关系的深度)，需要专业的专业知识和大量的开发人员工作。大型语言模型(LLM)的最新进展允许从各种数据源检索信息以生成响应，从而为独特地管理软件依赖关系提供了新的机会。为了突出这一技术的潜力，我们提出了一种概念验证检索增强生成(RAG)方法~\Tool，它将软件包的直接和传递依赖构造为四个流行的软件生态系统中的知识图(KG)。DepsRAG可以通过自动生成必要的查询来从KG中检索信息，然后用检索到的信息增强LLMS的输入，来回答用户关于软件依赖性的问题。DepsRAG还可以执行网络搜索，以回答LLM无法通过KG直接回答的问题。我们确定了DepsRAG可以提供的实实在在的好处并讨论了其局限性。



## **34. Adversarial Tuning: Defending Against Jailbreak Attacks for LLMs**

对抗性调整：防御LLM的越狱攻击 cs.CL

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2406.06622v1) [paper-pdf](http://arxiv.org/pdf/2406.06622v1)

**Authors**: Fan Liu, Zhao Xu, Hao Liu

**Abstract**: Although safely enhanced Large Language Models (LLMs) have achieved remarkable success in tackling various complex tasks in a zero-shot manner, they remain susceptible to jailbreak attacks, particularly the unknown jailbreak attack. To enhance LLMs' generalized defense capabilities, we propose a two-stage adversarial tuning framework, which generates adversarial prompts to explore worst-case scenarios by optimizing datasets containing pairs of adversarial prompts and their safe responses. In the first stage, we introduce the hierarchical meta-universal adversarial prompt learning to efficiently and effectively generate token-level adversarial prompts. In the second stage, we propose the automatic adversarial prompt learning to iteratively refine semantic-level adversarial prompts, further enhancing LLM's defense capabilities. We conducted comprehensive experiments on three widely used jailbreak datasets, comparing our framework with six defense baselines under five representative attack scenarios. The results underscore the superiority of our proposed methods. Furthermore, our adversarial tuning framework exhibits empirical generalizability across various attack strategies and target LLMs, highlighting its potential as a transferable defense mechanism.

摘要: 尽管安全增强的大型语言模型(LLM)在以零射击方式处理各种复杂任务方面取得了显著的成功，但它们仍然容易受到越狱攻击，特别是未知的越狱攻击。为了增强LLMS的广义防御能力，我们提出了一个两阶段对抗性调整框架，该框架通过优化包含对抗性提示对及其安全响应的数据集来生成对抗性提示以探索最坏的情况。在第一阶段，我们引入了层次化的元通用对抗性提示学习来高效、有效地生成令牌级对抗性提示。在第二阶段，我们提出了自动对抗性提示学习来迭代提炼语义级的对抗性提示，进一步增强了LLM的防御能力。我们在三个广泛使用的越狱数据集上进行了全面的实验，将我们的框架与五个典型攻击场景下的六个防御基线进行了比较。结果强调了我们所提出的方法的优越性。此外，我们的对抗性调整框架展示了对各种攻击策略和目标LLM的经验泛化，突出了其作为一种可转移防御机制的潜力。



## **35. SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models**

SALAD-Bench：大型语言模型的分层和全面的安全基准 cs.CL

Accepted at ACL 2024 Findings

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2402.05044v4) [paper-pdf](http://arxiv.org/pdf/2402.05044v4)

**Authors**: Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, Jing Shao

**Abstract**: In the rapidly evolving landscape of Large Language Models (LLMs), ensuring robust safety measures is paramount. To meet this crucial need, we propose \emph{SALAD-Bench}, a safety benchmark specifically designed for evaluating LLMs, attack, and defense methods. Distinguished by its breadth, SALAD-Bench transcends conventional benchmarks through its large scale, rich diversity, intricate taxonomy spanning three levels, and versatile functionalities.SALAD-Bench is crafted with a meticulous array of questions, from standard queries to complex ones enriched with attack, defense modifications and multiple-choice. To effectively manage the inherent complexity, we introduce an innovative evaluators: the LLM-based MD-Judge for QA pairs with a particular focus on attack-enhanced queries, ensuring a seamless, and reliable evaluation. Above components extend SALAD-Bench from standard LLM safety evaluation to both LLM attack and defense methods evaluation, ensuring the joint-purpose utility. Our extensive experiments shed light on the resilience of LLMs against emerging threats and the efficacy of contemporary defense tactics. Data and evaluator are released under https://github.com/OpenSafetyLab/SALAD-BENCH.

摘要: 在快速发展的大型语言模型(LLM)环境中，确保可靠的安全措施至关重要。为了满足这一关键需求，我们提出了一种专门为评估LLMS、攻击和防御方法而设计的安全基准。SALAD-BENCH以其广度、丰富的多样性、跨越三个层次的复杂分类和多功能而超越传统基准。SALAD-BENCH精心设计了一系列细致的问题，从标准查询到复杂的问题，丰富了攻击、防御修改和多项选择。为了有效地管理固有的复杂性，我们引入了一种创新的评估器：基于LLM的针对QA对的MD-裁判，特别关注攻击增强的查询，确保无缝和可靠的评估。上述组件将沙拉工作台从标准的LLM安全评估扩展到LLM攻防方法评估，确保了联合用途的实用性。我们广泛的实验揭示了低密度脂蛋白对新出现的威胁的弹性以及当代防御战术的有效性。数据和评估器在https://github.com/OpenSafetyLab/SALAD-BENCH.下发布



## **36. Sales Whisperer: A Human-Inconspicuous Attack on LLM Brand Recommendations**

销售耳语者：对LLM品牌推荐的人性不起眼的攻击 cs.CR

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2406.04755v1) [paper-pdf](http://arxiv.org/pdf/2406.04755v1)

**Authors**: Weiran Lin, Anna Gerchanovsky, Omer Akgul, Lujo Bauer, Matt Fredrikson, Zifan Wang

**Abstract**: Large language model (LLM) users might rely on others (e.g., prompting services), to write prompts. However, the risks of trusting prompts written by others remain unstudied. In this paper, we assess the risk of using such prompts on brand recommendation tasks when shopping. First, we found that paraphrasing prompts can result in LLMs mentioning given brands with drastically different probabilities, including a pair of prompts where the probability changes by 100%. Next, we developed an approach that can be used to perturb an original base prompt to increase the likelihood that an LLM mentions a given brand. We designed a human-inconspicuous algorithm that perturbs prompts, which empirically forces LLMs to mention strings related to a brand more often, by absolute improvements up to 78.3%. Our results suggest that our perturbed prompts, 1) are inconspicuous to humans, 2) force LLMs to recommend a target brand more often, and 3) increase the perceived chances of picking targeted brands.

摘要: 大型语言模型（LLM）用户可能会依赖其他人（例如，提示服务），写提示。然而，相信他人撰写的提示的风险仍然没有得到研究。在本文中，我们评估了购物时在品牌推荐任务中使用此类提示的风险。首先，我们发现重述提示可能会导致LLM以截然不同的概率提及给定品牌，包括一对概率变化100%的提示。接下来，我们开发了一种方法，可用于扰乱原始基本提示，以增加LLM提及给定品牌的可能性。我们设计了一种不引人注目的算法，可以扰乱提示，从经验上迫使LLM更频繁地提及与品牌相关的字符串，绝对改进高达78.3%。我们的结果表明，我们的干扰提示，1）对人类来说不明显，2）迫使LLM更频繁地推荐目标品牌，3）增加了选择目标品牌的感知机会。



## **37. COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability**

冷攻击：具有隐蔽性和可控性的越狱LLM cs.LG

Accepted to ICML 2024

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2402.08679v2) [paper-pdf](http://arxiv.org/pdf/2402.08679v2)

**Authors**: Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, Bin Hu

**Abstract**: Jailbreaks on large language models (LLMs) have recently received increasing attention. For a comprehensive assessment of LLM safety, it is essential to consider jailbreaks with diverse attributes, such as contextual coherence and sentiment/stylistic variations, and hence it is beneficial to study controllable jailbreaking, i.e. how to enforce control on LLM attacks. In this paper, we formally formulate the controllable attack generation problem, and build a novel connection between this problem and controllable text generation, a well-explored topic of natural language processing. Based on this connection, we adapt the Energy-based Constrained Decoding with Langevin Dynamics (COLD), a state-of-the-art, highly efficient algorithm in controllable text generation, and introduce the COLD-Attack framework which unifies and automates the search of adversarial LLM attacks under a variety of control requirements such as fluency, stealthiness, sentiment, and left-right-coherence. The controllability enabled by COLD-Attack leads to diverse new jailbreak scenarios which not only cover the standard setting of generating fluent (suffix) attack with continuation constraint, but also allow us to address new controllable attack settings such as revising a user query adversarially with paraphrasing constraint, and inserting stealthy attacks in context with position constraint. Our extensive experiments on various LLMs (Llama-2, Mistral, Vicuna, Guanaco, GPT-3.5, and GPT-4) show COLD-Attack's broad applicability, strong controllability, high success rate, and attack transferability. Our code is available at https://github.com/Yu-Fangxu/COLD-Attack.

摘要: 大型语言模型(LLM)的越狱最近受到越来越多的关注。为了全面评估LLM的安全性，必须考虑具有不同属性的越狱，例如上下文连贯性和情绪/风格变化，因此研究可控越狱是有益的，即如何加强对LLM攻击的控制。在本文中，我们形式化地描述了可控攻击生成问题，并将该问题与自然语言处理的一个热门话题--可控文本生成建立了一种新的联系。基于此，我们采用了基于能量的朗之万动力学约束解码算法(COLD)，这是一种最新的、高效的可控文本生成算法，并引入了冷攻击框架，该框架可以在流畅性、隐蔽性、情感和左右一致性等各种控制要求下统一和自动化搜索敌意LLM攻击。冷攻击的可控性导致了新的越狱场景的多样化，这些场景不仅覆盖了生成具有连续约束的流畅(后缀)攻击的标准设置，而且允许我们应对新的可控攻击设置，如使用释义约束对用户查询进行恶意修改，以及在具有位置约束的上下文中插入隐蔽攻击。我们在不同的LLMS(大骆驼-2、米斯特拉尔、维库纳、瓜纳科、GPT-3.5和GPT-4)上的广泛实验表明，冷攻击具有广泛的适用性、很强的可控性、高成功率和攻击可转移性。我们的代码可以在https://github.com/Yu-Fangxu/COLD-Attack.上找到



## **38. Safety Alignment in NLP Tasks: Weakly Aligned Summarization as an In-Context Attack**

NLP任务中的安全一致：弱一致摘要作为上下文内攻击 cs.CL

Accepted to ACL2024 main

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2312.06924v2) [paper-pdf](http://arxiv.org/pdf/2312.06924v2)

**Authors**: Yu Fu, Yufei Li, Wen Xiao, Cong Liu, Yue Dong

**Abstract**: Recent developments in balancing the usefulness and safety of Large Language Models (LLMs) have raised a critical question: Are mainstream NLP tasks adequately aligned with safety consideration? Our study, focusing on safety-sensitive documents obtained through adversarial attacks, reveals significant disparities in the safety alignment of various NLP tasks. For instance, LLMs can effectively summarize malicious long documents but often refuse to translate them. This discrepancy highlights a previously unidentified vulnerability: attacks exploiting tasks with weaker safety alignment, like summarization, can potentially compromise the integrity of tasks traditionally deemed more robust, such as translation and question-answering (QA). Moreover, the concurrent use of multiple NLP tasks with lesser safety alignment increases the risk of LLMs inadvertently processing harmful content. We demonstrate these vulnerabilities in various safety-aligned LLMs, particularly Llama2 models, Gemini and GPT-4, indicating an urgent need for strengthening safety alignments across a broad spectrum of NLP tasks.

摘要: 最近在平衡大型语言模型(LLM)的有用性和安全性方面的发展提出了一个关键问题：主流NLP任务是否与安全考虑充分一致？我们的研究集中在通过对抗性攻击获得的安全敏感文件上，揭示了各种NLP任务在安全匹配方面的显著差异。例如，LLMS可以有效地汇总恶意的长文档，但通常拒绝翻译它们。这一差异突显了一个以前未知的漏洞：攻击利用安全一致性较弱的任务(如摘要)，可能会潜在地损害传统上被认为更健壮的任务的完整性，如翻译和问答(QA)。此外，同时使用安全性较低的多个NLP任务会增加LLMS无意中处理有害内容的风险。我们在各种安全对齐的LLM中展示了这些漏洞，特别是Llama2型号、Gemini和GPT-4，这表明迫切需要在广泛的NLP任务中加强安全对齐。



## **39. Evaluating the Efficacy of Large Language Models in Identifying Phishing Attempts**

评估大型语言模型在识别网络钓鱼尝试方面的有效性 cs.CL

7 pages, 3 figures

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2404.15485v3) [paper-pdf](http://arxiv.org/pdf/2404.15485v3)

**Authors**: Het Patel, Umair Rehman, Farkhund Iqbal

**Abstract**: Phishing, a prevalent cybercrime tactic for decades, remains a significant threat in today's digital world. By leveraging clever social engineering elements and modern technology, cybercrime targets many individuals, businesses, and organizations to exploit trust and security. These cyber-attackers are often disguised in many trustworthy forms to appear as legitimate sources. By cleverly using psychological elements like urgency, fear, social proof, and other manipulative strategies, phishers can lure individuals into revealing sensitive and personalized information. Building on this pervasive issue within modern technology, this paper aims to analyze the effectiveness of 15 Large Language Models (LLMs) in detecting phishing attempts, specifically focusing on a randomized set of "419 Scam" emails. The objective is to determine which LLMs can accurately detect phishing emails by analyzing a text file containing email metadata based on predefined criteria. The experiment concluded that the following models, ChatGPT 3.5, GPT-3.5-Turbo-Instruct, and ChatGPT, were the most effective in detecting phishing emails.

摘要: 网络钓鱼是几十年来流行的一种网络犯罪策略，在当今的数字世界中仍然是一个重大威胁。通过利用聪明的社会工程元素和现代技术，网络犯罪以许多个人、企业和组织为目标，以利用信任和安全。这些网络攻击者往往以许多可信的形式伪装成合法的来源。通过巧妙地使用紧急、恐惧、社会证明和其他操纵策略等心理因素，网络钓鱼者可以诱使个人泄露敏感和个性化的信息。基于这一现代技术中普遍存在的问题，本文旨在分析15个大型语言模型(LLM)在检测网络钓鱼尝试方面的有效性，特别是关注一组随机的“419骗局”电子邮件。目标是通过基于预定义标准分析包含电子邮件元数据的文本文件，确定哪些LLM可以准确检测钓鱼电子邮件。实验得出的结论是，ChatGPT 3.5、GPT-3.5-Turbo-Indict和ChatGPT模型在检测钓鱼电子邮件方面最有效。



## **40. Defending LLMs against Jailbreaking Attacks via Backtranslation**

通过反向翻译保护LLM免受越狱攻击 cs.CL

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2402.16459v3) [paper-pdf](http://arxiv.org/pdf/2402.16459v3)

**Authors**: Yihan Wang, Zhouxing Shi, Andrew Bai, Cho-Jui Hsieh

**Abstract**: Although many large language models (LLMs) have been trained to refuse harmful requests, they are still vulnerable to jailbreaking attacks which rewrite the original prompt to conceal its harmful intent. In this paper, we propose a new method for defending LLMs against jailbreaking attacks by ``backtranslation''. Specifically, given an initial response generated by the target LLM from an input prompt, our backtranslation prompts a language model to infer an input prompt that can lead to the response. The inferred prompt is called the backtranslated prompt which tends to reveal the actual intent of the original prompt, since it is generated based on the LLM's response and not directly manipulated by the attacker. We then run the target LLM again on the backtranslated prompt, and we refuse the original prompt if the model refuses the backtranslated prompt. We explain that the proposed defense provides several benefits on its effectiveness and efficiency. We empirically demonstrate that our defense significantly outperforms the baselines, in the cases that are hard for the baselines, and our defense also has little impact on the generation quality for benign input prompts. Our implementation is based on our library for LLM jailbreaking defense algorithms at \url{https://github.com/YihanWang617/llm-jailbreaking-defense}, and the code for reproducing our experiments is available at \url{https://github.com/YihanWang617/LLM-Jailbreaking-Defense-Backtranslation}.

摘要: 尽管许多大型语言模型(LLM)已经接受了拒绝有害请求的培训，但它们仍然容易受到越狱攻击，这些攻击会重写原始提示以掩盖其有害意图。在本文中，我们提出了一种新的方法来防御‘’反向翻译‘’越狱攻击。具体地说，给定目标LLM从输入提示生成的初始响应，我们的反向翻译会提示语言模型推断出可能导致该响应的输入提示。推断的提示称为反向翻译提示，它往往会揭示原始提示的实际意图，因为它是基于LLM的响应生成的，而不是由攻击者直接操纵的。然后，我们在回译的提示上再次运行目标LLM，如果模型拒绝回译的提示，我们将拒绝原始提示。我们解释说，拟议的辩护在其有效性和效率方面提供了几个好处。我们的经验证明，我们的防御显著优于基线，在基线难以达到的情况下，我们的防御对良性输入提示的生成质量也几乎没有影响。我们的实现基于我们在\url{https://github.com/YihanWang617/llm-jailbreaking-defense}，的LLm越狱防御算法库，重现我们实验的代码可以在\url{https://github.com/YihanWang617/LLM-Jailbreaking-Defense-Backtranslation}.上找到



## **41. BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models**

BadRAG：识别大型语言模型检索增强生成中的漏洞 cs.CR

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.00083v2) [paper-pdf](http://arxiv.org/pdf/2406.00083v2)

**Authors**: Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun Chen, Qian Lou

**Abstract**: Large Language Models (LLMs) are constrained by outdated information and a tendency to generate incorrect data, commonly referred to as "hallucinations." Retrieval-Augmented Generation (RAG) addresses these limitations by combining the strengths of retrieval-based methods and generative models. This approach involves retrieving relevant information from a large, up-to-date dataset and using it to enhance the generation process, leading to more accurate and contextually appropriate responses. Despite its benefits, RAG introduces a new attack surface for LLMs, particularly because RAG databases are often sourced from public data, such as the web. In this paper, we propose \TrojRAG{} to identify the vulnerabilities and attacks on retrieval parts (RAG database) and their indirect attacks on generative parts (LLMs). Specifically, we identify that poisoning several customized content passages could achieve a retrieval backdoor, where the retrieval works well for clean queries but always returns customized poisoned adversarial queries. Triggers and poisoned passages can be highly customized to implement various attacks. For example, a trigger could be a semantic group like "The Republican Party, Donald Trump, etc." Adversarial passages can be tailored to different contents, not only linked to the triggers but also used to indirectly attack generative LLMs without modifying them. These attacks can include denial-of-service attacks on RAG and semantic steering attacks on LLM generations conditioned by the triggers. Our experiments demonstrate that by just poisoning 10 adversarial passages can induce 98.2\% success rate to retrieve the adversarial passages. Then, these passages can increase the reject ratio of RAG-based GPT-4 from 0.01\% to 74.6\% or increase the rate of negative responses from 0.22\% to 72\% for targeted queries.

摘要: 大型语言模型(LLM)受到过时信息和生成错误数据的倾向的限制，这通常被称为“幻觉”。检索-增强生成(RAG)结合了基于检索的方法和生成模型的优点，解决了这些局限性。这种方法涉及从大型最新数据集中检索相关信息，并使用它来改进生成过程，从而产生更准确和符合上下文的响应。尽管有好处，但RAG为LLMS带来了新的攻击面，特别是因为RAG数据库通常来自公共数据，如Web。本文提出用TrojRAG{}来识别检索零件(RAG数据库)上的漏洞和攻击，以及它们对生成零件(LLM)的间接攻击。具体地说，我们发现毒化几个定制的内容段落可以实现检索后门，其中检索对于干净的查询工作得很好，但总是返回定制的有毒对抗性查询。触发器和有毒段落可以高度定制，以实施各种攻击。例如，触发点可能是一个语义组，比如“共和党、唐纳德·特朗普等。”对抗性段落可以针对不同的内容量身定做，不仅与触发因素有关，还可以用来间接攻击生成性LLM而不修改它们。这些攻击可以包括针对RAG的拒绝服务攻击和针对受触发器限制的LLM生成的语义引导攻击。我们的实验表明，只要毒化10篇对抗性文章，就可以诱导98.2%的成功率来检索对抗性文章。然后，这些文章可以将基于RAG的GPT-4的拒绝率从0.01%提高到74.6%，或者将目标查询的否定回复率从0.22%提高到72%。



## **42. Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt**

通过双模式对抗提示的越狱视觉语言模型 cs.CV

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.04031v1) [paper-pdf](http://arxiv.org/pdf/2406.04031v1)

**Authors**: Zonghao Ying, Aishan Liu, Tianyuan Zhang, Zhengmin Yu, Siyuan Liang, Xianglong Liu, Dacheng Tao

**Abstract**: In the realm of large vision language models (LVLMs), jailbreak attacks serve as a red-teaming approach to bypass guardrails and uncover safety implications. Existing jailbreaks predominantly focus on the visual modality, perturbing solely visual inputs in the prompt for attacks. However, they fall short when confronted with aligned models that fuse visual and textual features simultaneously for generation. To address this limitation, this paper introduces the Bi-Modal Adversarial Prompt Attack (BAP), which executes jailbreaks by optimizing textual and visual prompts cohesively. Initially, we adversarially embed universally harmful perturbations in an image, guided by a few-shot query-agnostic corpus (e.g., affirmative prefixes and negative inhibitions). This process ensures that image prompt LVLMs to respond positively to any harmful queries. Subsequently, leveraging the adversarial image, we optimize textual prompts with specific harmful intent. In particular, we utilize a large language model to analyze jailbreak failures and employ chain-of-thought reasoning to refine textual prompts through a feedback-iteration manner. To validate the efficacy of our approach, we conducted extensive evaluations on various datasets and LVLMs, demonstrating that our method significantly outperforms other methods by large margins (+29.03% in attack success rate on average). Additionally, we showcase the potential of our attacks on black-box commercial LVLMs, such as Gemini and ChatGLM.

摘要: 在大型视觉语言模型(LVLM)领域，越狱攻击是一种绕过护栏并发现安全隐患的红队方法。现有的越狱主要集中在视觉形式上，只干扰攻击提示中的视觉输入。然而，当面对同时融合视觉和文本特征以生成的对齐模型时，它们不能满足要求。为了解决这一局限性，本文引入了双模式对抗性提示攻击(BAP)，它通过结合优化文本和视觉提示来执行越狱。最初，我们不利地在图像中嵌入普遍有害的扰动，由几个与查询无关的语料库(例如，肯定前缀和否定抑制)引导。此过程确保图像提示LVLMS对任何有害查询做出积极响应。随后，利用敌意图像，我们优化了具有特定有害意图的文本提示。特别是，我们利用一个大的语言模型来分析越狱失败，并使用思想链推理来通过反馈迭代的方式来提炼文本提示。为了验证我们方法的有效性，我们在不同的数据集和LVLM上进行了广泛的评估，结果表明我们的方法在很大程度上优于其他方法(攻击成功率平均为+29.03%)。此外，我们还展示了我们对黑盒商业LVLM的攻击潜力，如Gemini和ChatGLM。



## **43. Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!**

模拟失调：大型语言模型的安全调整可能会适得其反！ cs.CL

ACL 2024

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2402.12343v4) [paper-pdf](http://arxiv.org/pdf/2402.12343v4)

**Authors**: Zhanhui Zhou, Jie Liu, Zhichen Dong, Jiaheng Liu, Chao Yang, Wanli Ouyang, Yu Qiao

**Abstract**: Large language models (LLMs) undergo safety alignment to ensure safe conversations with humans. However, this paper introduces a training-free attack method capable of reversing safety alignment, converting the outcomes of stronger alignment into greater potential for harm by accessing only LLM output token distributions. Specifically, our method achieves this reversal by contrasting the output token distribution of a safety-aligned language model (e.g., Llama-2-chat) against its pre-trained version (e.g., Llama-2), so that the token predictions are shifted towards the opposite direction of safety alignment. We name this method emulated disalignment (ED) because sampling from this contrastive distribution provably emulates the result of fine-tuning to minimize a safety reward. Our experiments with ED across three evaluation datasets and four model families (Llama-1, Llama-2, Mistral, and Alpaca) show that ED doubles the harmfulness of pre-trained models and outperforms strong baselines, achieving the highest harmful rates in 43 out of 48 evaluation subsets by a large margin. Eventually, given ED's reliance on language model output token distributions, which particularly compromises open-source models, our findings highlight the need to reassess the open accessibility of language models, even if they have been safety-aligned. Code is available at https://github.com/ZHZisZZ/emulated-disalignment.

摘要: 大型语言模型(LLM)经过安全调整，以确保与人类的安全对话。然而，本文介绍了一种无需训练的攻击方法，该方法能够逆转安全对齐，通过仅访问LLM输出令牌分布来将更强对齐的结果转换为更大的潜在危害。具体地说，我们的方法通过将安全对齐的语言模型(例如，Llama-2-Chat)的输出令牌分布与其预先训练的版本(例如，Llama-2)进行比较，从而使令牌预测向安全对齐的相反方向移动，从而实现了这种逆转。我们将这种方法命名为模拟不对齐(ED)，因为从这种对比分布中进行的采样可以被证明是模拟了微调以最小化安全奖励的结果。我们在三个评估数据集和四个模型家族(骆驼-1、骆驼-2、米斯特拉尔和羊驼)上使用ED进行的实验表明，ED的危害性是预训练模型的两倍，并且性能优于强基线，在48个评估子集中的43个子集上获得了最高的伤害率。最终，鉴于ED对语言模型输出令牌分发的依赖，这尤其损害了开源模型，我们的发现突显了重新评估语言模型的开放可访问性的必要性，即使它们是安全一致的。代码可在https://github.com/ZHZisZZ/emulated-disalignment.上找到



## **44. Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs**

竞争报告：在一致的LLC中寻找通用越狱后门 cs.CL

Competition Report

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2404.14461v2) [paper-pdf](http://arxiv.org/pdf/2404.14461v2)

**Authors**: Javier Rando, Francesco Croce, Kryštof Mitka, Stepan Shabalin, Maksym Andriushchenko, Nicolas Flammarion, Florian Tramèr

**Abstract**: Large language models are aligned to be safe, preventing users from generating harmful content like misinformation or instructions for illegal activities. However, previous work has shown that the alignment process is vulnerable to poisoning attacks. Adversaries can manipulate the safety training data to inject backdoors that act like a universal sudo command: adding the backdoor string to any prompt enables harmful responses from models that, otherwise, behave safely. Our competition, co-located at IEEE SaTML 2024, challenged participants to find universal backdoors in several large language models. This report summarizes the key findings and promising ideas for future research.

摘要: 大型语言模型经过调整以确保安全，防止用户生成错误信息或非法活动指令等有害内容。然而，之前的工作表明，对齐过程很容易受到中毒攻击。对手可以操纵安全训练数据来注入类似于通用sudo命令的后门：将后门字符串添加到任何提示中都会导致模型做出有害响应，否则这些模型会安全地运行。我们的竞赛在IEEE SaTML 2024上举行，挑战参与者在几个大型语言模型中找到通用后门。本报告总结了关键发现和未来研究的有希望的想法。



## **45. AutoJailbreak: Exploring Jailbreak Attacks and Defenses through a Dependency Lens**

自动越狱：通过依赖的视角探索越狱攻击和防御 cs.CR

32 pages, 2 figures

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.03805v1) [paper-pdf](http://arxiv.org/pdf/2406.03805v1)

**Authors**: Lin Lu, Hai Yan, Zenghui Yuan, Jiawen Shi, Wenqi Wei, Pin-Yu Chen, Pan Zhou

**Abstract**: Jailbreak attacks in large language models (LLMs) entail inducing the models to generate content that breaches ethical and legal norm through the use of malicious prompts, posing a substantial threat to LLM security. Current strategies for jailbreak attack and defense often focus on optimizing locally within specific algorithmic frameworks, resulting in ineffective optimization and limited scalability. In this paper, we present a systematic analysis of the dependency relationships in jailbreak attack and defense techniques, generalizing them to all possible attack surfaces. We employ directed acyclic graphs (DAGs) to position and analyze existing jailbreak attacks, defenses, and evaluation methodologies, and propose three comprehensive, automated, and logical frameworks. \texttt{AutoAttack} investigates dependencies in two lines of jailbreak optimization strategies: genetic algorithm (GA)-based attacks and adversarial-generation-based attacks, respectively. We then introduce an ensemble jailbreak attack to exploit these dependencies. \texttt{AutoDefense} offers a mixture-of-defenders approach by leveraging the dependency relationships in pre-generative and post-generative defense strategies. \texttt{AutoEvaluation} introduces a novel evaluation method that distinguishes hallucinations, which are often overlooked, from jailbreak attack and defense responses. Through extensive experiments, we demonstrate that the proposed ensemble jailbreak attack and defense framework significantly outperforms existing research.

摘要: 大型语言模型(LLM)中的越狱攻击需要通过使用恶意提示诱导模型生成违反道德和法律规范的内容，从而对LLM安全构成重大威胁。当前的越狱攻防策略往往集中在特定算法框架内的局部优化，导致优化效果不佳，可扩展性有限。本文系统地分析了越狱攻防技术中的依赖关系，并将其推广到所有可能的攻击面。我们使用有向无环图(DAG)来定位和分析现有的越狱攻击、防御和评估方法，并提出了三个全面的、自动化的和逻辑的框架。Texttt{AutoAttack}研究了两种越狱优化策略的依赖关系：基于遗传算法(GA)的攻击和基于对抗性生成的攻击。然后，我们引入整体越狱攻击来利用这些依赖关系。通过利用生成前和生成后防御策略中的依赖关系，\exttt{AutoDefense}提供了混合防御者的方法。Texttt(自动评估)介绍了一种新的评估方法，可以将经常被忽视的幻觉与越狱攻击和防御反应区分开来。通过大量的实验，我们证明了所提出的集成越狱攻防框架的性能明显优于现有的研究。



## **46. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

保护语言模型免受越狱攻击的鲁棒即时优化 cs.LG

Code available at https://github.com/lapisrocks/rpo

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2401.17263v3) [paper-pdf](http://arxiv.org/pdf/2401.17263v3)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, large language models (LLMs) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries can modify prompts to induce unwanted behavior. While some defenses have been proposed, they have not been adapted to newly proposed attacks and more challenging threat models. To address this, we propose an optimization-based objective for defending LLMs against jailbreaking attacks and an algorithm, Robust Prompt Optimization (RPO) to create robust system-level defenses. Our approach directly incorporates the adversary into the defensive objective and optimizes a lightweight and transferable suffix, enabling RPO to adapt to worst-case adaptive attacks. Our theoretical and experimental results show improved robustness to both jailbreaks seen during optimization and unknown jailbreaks, reducing the attack success rate (ASR) on GPT-4 to 6% and Llama-2 to 0% on JailbreakBench, setting the state-of-the-art. Code can be found at https://github.com/lapisrocks/rpo

摘要: 尽管在人工智能对齐方面取得了进展，但大型语言模型(LLM)仍然容易受到对手攻击或越狱的攻击，在这些攻击或越狱中，对手可以修改提示以诱导不想要的行为。虽然已经提出了一些防御措施，但它们还没有适应新提出的攻击和更具挑战性的威胁模型。为了解决这个问题，我们提出了一个基于优化的目标来保护LLMS免受越狱攻击，并提出了一个算法--稳健提示优化(RPO)来创建强大的系统级防御。我们的方法直接将对手合并到防御目标中，并优化了一个轻量级和可转移的后缀，使RPO能够适应最坏情况的自适应攻击。我们的理论和实验结果表明，对于优化期间看到的越狱和未知越狱，我们都提高了健壮性，将GPT-4上的攻击成功率(ASR)降低到6%，将Llama-2上的攻击成功率降低到0%，从而达到了最先进的水平。代码可在https://github.com/lapisrocks/rpo上找到



## **47. Stealthy Attack on Large Language Model based Recommendation**

对基于大型语言模型的推荐的隐形攻击 cs.CL

ACL 2024 Main

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2402.14836v2) [paper-pdf](http://arxiv.org/pdf/2402.14836v2)

**Authors**: Jinghao Zhang, Yuting Liu, Qiang Liu, Shu Wu, Guibing Guo, Liang Wang

**Abstract**: Recently, the powerful large language models (LLMs) have been instrumental in propelling the progress of recommender systems (RS). However, while these systems have flourished, their susceptibility to security threats has been largely overlooked. In this work, we reveal that the introduction of LLMs into recommendation models presents new security vulnerabilities due to their emphasis on the textual content of items. We demonstrate that attackers can significantly boost an item's exposure by merely altering its textual content during the testing phase, without requiring direct interference with the model's training process. Additionally, the attack is notably stealthy, as it does not affect the overall recommendation performance and the modifications to the text are subtle, making it difficult for users and platforms to detect. Our comprehensive experiments across four mainstream LLM-based recommendation models demonstrate the superior efficacy and stealthiness of our approach. Our work unveils a significant security gap in LLM-based recommendation systems and paves the way for future research on protecting these systems.

摘要: 近年来，强大的大型语言模型(LLMS)在推动推荐系统(RS)的发展方面发挥了重要作用。然而，尽管这些系统蓬勃发展，但它们对安全威胁的敏感性在很大程度上被忽视了。在这项工作中，我们揭示了将LLMS引入推荐模型中会出现新的安全漏洞，这是因为它们强调项目的文本内容。我们证明，攻击者只需在测试阶段改变项目的文本内容，就可以显著增加项目的曝光率，而不需要直接干预模型的训练过程。此外，攻击具有明显的隐蔽性，因为它不会影响整体推荐性能，而且对文本的修改也很微妙，使得用户和平台很难检测到。我们对四个主流的基于LLM的推荐模型进行了全面的实验，证明了我们的方法具有优越的有效性和隐蔽性。我们的工作揭示了基于LLM的推荐系统中存在的一个显著的安全漏洞，并为未来保护这些系统的研究铺平了道路。



## **48. Improved Techniques for Optimization-Based Jailbreaking on Large Language Models**

基于优化的大型语言模型越狱改进技术 cs.LG

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2405.21018v2) [paper-pdf](http://arxiv.org/pdf/2405.21018v2)

**Authors**: Xiaojun Jia, Tianyu Pang, Chao Du, Yihao Huang, Jindong Gu, Yang Liu, Xiaochun Cao, Min Lin

**Abstract**: Large language models (LLMs) are being rapidly developed, and a key component of their widespread deployment is their safety-related alignment. Many red-teaming efforts aim to jailbreak LLMs, where among these efforts, the Greedy Coordinate Gradient (GCG) attack's success has led to a growing interest in the study of optimization-based jailbreaking techniques. Although GCG is a significant milestone, its attacking efficiency remains unsatisfactory. In this paper, we present several improved (empirical) techniques for optimization-based jailbreaks like GCG. We first observe that the single target template of "Sure" largely limits the attacking performance of GCG; given this, we propose to apply diverse target templates containing harmful self-suggestion and/or guidance to mislead LLMs. Besides, from the optimization aspects, we propose an automatic multi-coordinate updating strategy in GCG (i.e., adaptively deciding how many tokens to replace in each step) to accelerate convergence, as well as tricks like easy-to-hard initialisation. Then, we combine these improved technologies to develop an efficient jailbreak method, dubbed I-GCG. In our experiments, we evaluate on a series of benchmarks (such as NeurIPS 2023 Red Teaming Track). The results demonstrate that our improved techniques can help GCG outperform state-of-the-art jailbreaking attacks and achieve nearly 100% attack success rate. The code is released at https://github.com/jiaxiaojunQAQ/I-GCG.

摘要: 大型语言模型(LLM)正在迅速开发，其广泛部署的一个关键组件是与安全相关的一致性。许多红色团队的目标是越狱LLM，其中贪婪坐标梯度(GCG)攻击的成功导致了人们对基于优化的越狱技术的研究越来越感兴趣。虽然GCG是一个重要的里程碑，但其攻击效率仍然不能令人满意。在这篇文章中，我们提出了几种改进的(经验)技术，用于基于优化的越狱，如GCG。我们首先观察到单一目标模板“Sure”在很大程度上限制了GCG的攻击性能；鉴于此，我们建议使用包含有害自我暗示和/或引导的不同目标模板来误导LLM。此外，在优化方面，我们提出了GCG中的自动多坐标更新策略(即自适应地决定每一步需要替换多少个令牌)来加速收敛，以及容易初始化等技巧。然后，我们结合这些改进的技术开发了一种高效的越狱方法，称为I-GCG。在我们的实验中，我们在一系列基准(例如NeurIPS 2023 Red Teaming Track)上进行了评估。结果表明，改进后的技术可以帮助GCG超越最先进的越狱攻击，并获得近100%的攻击成功率。该代码在https://github.com/jiaxiaojunQAQ/I-GCG.上发布



## **49. CR-UTP: Certified Robustness against Universal Text Perturbations on Large Language Models**

CR-GPT：针对大型语言模型上通用文本扰动的鲁棒性认证 cs.CL

Accepted by ACL Findings 2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.01873v2) [paper-pdf](http://arxiv.org/pdf/2406.01873v2)

**Authors**: Qian Lou, Xin Liang, Jiaqi Xue, Yancheng Zhang, Rui Xie, Mengxin Zheng

**Abstract**: It is imperative to ensure the stability of every prediction made by a language model; that is, a language's prediction should remain consistent despite minor input variations, like word substitutions. In this paper, we investigate the problem of certifying a language model's robustness against Universal Text Perturbations (UTPs), which have been widely used in universal adversarial attacks and backdoor attacks. Existing certified robustness based on random smoothing has shown considerable promise in certifying the input-specific text perturbations (ISTPs), operating under the assumption that any random alteration of a sample's clean or adversarial words would negate the impact of sample-wise perturbations. However, with UTPs, masking only the adversarial words can eliminate the attack. A naive method is to simply increase the masking ratio and the likelihood of masking attack tokens, but it leads to a significant reduction in both certified accuracy and the certified radius due to input corruption by extensive masking. To solve this challenge, we introduce a novel approach, the superior prompt search method, designed to identify a superior prompt that maintains higher certified accuracy under extensive masking. Additionally, we theoretically motivate why ensembles are a particularly suitable choice as base prompts for random smoothing. The method is denoted by superior prompt ensembling technique. We also empirically confirm this technique, obtaining state-of-the-art results in multiple settings. These methodologies, for the first time, enable high certified accuracy against both UTPs and ISTPs. The source code of CR-UTP is available at \url {https://github.com/UCFML-Research/CR-UTP}.

摘要: 必须确保语言模型做出的每个预测的稳定性；也就是说，语言的预测应该保持一致，尽管输入有微小的变化，如单词替换。在本文中，我们研究了语言模型对通用文本扰动(UTP)的稳健性证明问题，UTP被广泛应用于通用对抗性攻击和后门攻击。现有的基于随机平滑的已证明的稳健性在证明特定于输入的文本扰动(ISTP)方面显示出相当大的前景，其操作是在假设样本的干净或敌意的单词的任何随机改变将否定样本方面的扰动的影响的情况下进行的。然而，对于UTP，只屏蔽敌意的单词就可以消除攻击。一种天真的方法是简单地增加掩蔽率和掩蔽攻击令牌的可能性，但由于广泛的掩蔽导致输入损坏，它导致认证的准确性和认证的半径都显著降低。为了解决这一挑战，我们引入了一种新的方法，高级提示搜索方法，旨在识别在广泛掩蔽下保持更高认证准确率的高级提示。此外，我们从理论上解释了为什么作为随机平滑的基础提示，集合是特别合适的选择。这种方法以卓越的即时集成技术表示。我们还从经验上证实了这一技术，在多个环境下获得了最先进的结果。这些方法首次针对UTP和ISTP实现了高度认证的准确性。CR-UTP的源代码可在\url{https://github.com/UCFML-Research/CR-UTP}.



## **50. Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models**

稳健的CLIP：稳健的大型视觉语言模型的视觉嵌入的无监督对抗微调 cs.LG

ICML 2024 Oral

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2402.12336v2) [paper-pdf](http://arxiv.org/pdf/2402.12336v2)

**Authors**: Christian Schlarmann, Naman Deep Singh, Francesco Croce, Matthias Hein

**Abstract**: Multi-modal foundation models like OpenFlamingo, LLaVA, and GPT-4 are increasingly used for various real-world tasks. Prior work has shown that these models are highly vulnerable to adversarial attacks on the vision modality. These attacks can be leveraged to spread fake information or defraud users, and thus pose a significant risk, which makes the robustness of large multi-modal foundation models a pressing problem. The CLIP model, or one of its variants, is used as a frozen vision encoder in many large vision-language models (LVLMs), e.g. LLaVA and OpenFlamingo. We propose an unsupervised adversarial fine-tuning scheme to obtain a robust CLIP vision encoder, which yields robustness on all vision down-stream tasks (LVLMs, zero-shot classification) that rely on CLIP. In particular, we show that stealth-attacks on users of LVLMs by a malicious third party providing manipulated images are no longer possible once one replaces the original CLIP model with our robust one. No retraining or fine-tuning of the down-stream LVLMs is required. The code and robust models are available at https://github.com/chs20/RobustVLM

摘要: OpenFlamingo、LLaVA和GPT-4等多模式基础模型越来越多地用于各种实际任务。先前的工作表明，这些模型非常容易受到视觉通道的对抗性攻击。这些攻击可以被用来传播虚假信息或欺骗用户，从而构成巨大的风险，这使得大型多通道基础模型的健壮性成为一个紧迫的问题。在许多大型视觉语言模型(如LLaVA和OpenFlamingo)中，剪辑模型或其变体之一被用作冻结的视觉编码器。我们提出了一种无监督的对抗性微调方案，以获得一个健壮的裁剪视觉编码器，它对依赖于裁剪的所有视觉下游任务(LVLM，零镜头分类)都具有健壮性。特别是，我们表明，一旦用我们的健壮模型取代了原始的剪辑模型，恶意第三方提供的篡改图像就不再可能对LVLMS的用户进行秘密攻击。不需要对下游低成本模块进行再培训或微调。代码和健壮模型可在https://github.com/chs20/RobustVLM上获得



