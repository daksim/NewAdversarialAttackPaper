# Latest Large Language Model Attack Papers
**update at 2024-06-09 20:16:18**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models**

BadRAG：识别大型语言模型检索增强生成中的漏洞 cs.CR

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.00083v2) [paper-pdf](http://arxiv.org/pdf/2406.00083v2)

**Authors**: Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun Chen, Qian Lou

**Abstract**: Large Language Models (LLMs) are constrained by outdated information and a tendency to generate incorrect data, commonly referred to as "hallucinations." Retrieval-Augmented Generation (RAG) addresses these limitations by combining the strengths of retrieval-based methods and generative models. This approach involves retrieving relevant information from a large, up-to-date dataset and using it to enhance the generation process, leading to more accurate and contextually appropriate responses. Despite its benefits, RAG introduces a new attack surface for LLMs, particularly because RAG databases are often sourced from public data, such as the web. In this paper, we propose \TrojRAG{} to identify the vulnerabilities and attacks on retrieval parts (RAG database) and their indirect attacks on generative parts (LLMs). Specifically, we identify that poisoning several customized content passages could achieve a retrieval backdoor, where the retrieval works well for clean queries but always returns customized poisoned adversarial queries. Triggers and poisoned passages can be highly customized to implement various attacks. For example, a trigger could be a semantic group like "The Republican Party, Donald Trump, etc." Adversarial passages can be tailored to different contents, not only linked to the triggers but also used to indirectly attack generative LLMs without modifying them. These attacks can include denial-of-service attacks on RAG and semantic steering attacks on LLM generations conditioned by the triggers. Our experiments demonstrate that by just poisoning 10 adversarial passages can induce 98.2\% success rate to retrieve the adversarial passages. Then, these passages can increase the reject ratio of RAG-based GPT-4 from 0.01\% to 74.6\% or increase the rate of negative responses from 0.22\% to 72\% for targeted queries.

摘要: 大型语言模型(LLM)受到过时信息和生成错误数据的倾向的限制，这通常被称为“幻觉”。检索-增强生成(RAG)结合了基于检索的方法和生成模型的优点，解决了这些局限性。这种方法涉及从大型最新数据集中检索相关信息，并使用它来改进生成过程，从而产生更准确和符合上下文的响应。尽管有好处，但RAG为LLMS带来了新的攻击面，特别是因为RAG数据库通常来自公共数据，如Web。本文提出用TrojRAG{}来识别检索零件(RAG数据库)上的漏洞和攻击，以及它们对生成零件(LLM)的间接攻击。具体地说，我们发现毒化几个定制的内容段落可以实现检索后门，其中检索对于干净的查询工作得很好，但总是返回定制的有毒对抗性查询。触发器和有毒段落可以高度定制，以实施各种攻击。例如，触发点可能是一个语义组，比如“共和党、唐纳德·特朗普等。”对抗性段落可以针对不同的内容量身定做，不仅与触发因素有关，还可以用来间接攻击生成性LLM而不修改它们。这些攻击可以包括针对RAG的拒绝服务攻击和针对受触发器限制的LLM生成的语义引导攻击。我们的实验表明，只要毒化10篇对抗性文章，就可以诱导98.2%的成功率来检索对抗性文章。然后，这些文章可以将基于RAG的GPT-4的拒绝率从0.01%提高到74.6%，或者将目标查询的否定回复率从0.22%提高到72%。



## **2. Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt**

通过双模式对抗提示的越狱视觉语言模型 cs.CV

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.04031v1) [paper-pdf](http://arxiv.org/pdf/2406.04031v1)

**Authors**: Zonghao Ying, Aishan Liu, Tianyuan Zhang, Zhengmin Yu, Siyuan Liang, Xianglong Liu, Dacheng Tao

**Abstract**: In the realm of large vision language models (LVLMs), jailbreak attacks serve as a red-teaming approach to bypass guardrails and uncover safety implications. Existing jailbreaks predominantly focus on the visual modality, perturbing solely visual inputs in the prompt for attacks. However, they fall short when confronted with aligned models that fuse visual and textual features simultaneously for generation. To address this limitation, this paper introduces the Bi-Modal Adversarial Prompt Attack (BAP), which executes jailbreaks by optimizing textual and visual prompts cohesively. Initially, we adversarially embed universally harmful perturbations in an image, guided by a few-shot query-agnostic corpus (e.g., affirmative prefixes and negative inhibitions). This process ensures that image prompt LVLMs to respond positively to any harmful queries. Subsequently, leveraging the adversarial image, we optimize textual prompts with specific harmful intent. In particular, we utilize a large language model to analyze jailbreak failures and employ chain-of-thought reasoning to refine textual prompts through a feedback-iteration manner. To validate the efficacy of our approach, we conducted extensive evaluations on various datasets and LVLMs, demonstrating that our method significantly outperforms other methods by large margins (+29.03% in attack success rate on average). Additionally, we showcase the potential of our attacks on black-box commercial LVLMs, such as Gemini and ChatGLM.

摘要: 在大型视觉语言模型(LVLM)领域，越狱攻击是一种绕过护栏并发现安全隐患的红队方法。现有的越狱主要集中在视觉形式上，只干扰攻击提示中的视觉输入。然而，当面对同时融合视觉和文本特征以生成的对齐模型时，它们不能满足要求。为了解决这一局限性，本文引入了双模式对抗性提示攻击(BAP)，它通过结合优化文本和视觉提示来执行越狱。最初，我们不利地在图像中嵌入普遍有害的扰动，由几个与查询无关的语料库(例如，肯定前缀和否定抑制)引导。此过程确保图像提示LVLMS对任何有害查询做出积极响应。随后，利用敌意图像，我们优化了具有特定有害意图的文本提示。特别是，我们利用一个大的语言模型来分析越狱失败，并使用思想链推理来通过反馈迭代的方式来提炼文本提示。为了验证我们方法的有效性，我们在不同的数据集和LVLM上进行了广泛的评估，结果表明我们的方法在很大程度上优于其他方法(攻击成功率平均为+29.03%)。此外，我们还展示了我们对黑盒商业LVLM的攻击潜力，如Gemini和ChatGLM。



## **3. Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!**

模拟失调：大型语言模型的安全调整可能会适得其反！ cs.CL

ACL 2024

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2402.12343v4) [paper-pdf](http://arxiv.org/pdf/2402.12343v4)

**Authors**: Zhanhui Zhou, Jie Liu, Zhichen Dong, Jiaheng Liu, Chao Yang, Wanli Ouyang, Yu Qiao

**Abstract**: Large language models (LLMs) undergo safety alignment to ensure safe conversations with humans. However, this paper introduces a training-free attack method capable of reversing safety alignment, converting the outcomes of stronger alignment into greater potential for harm by accessing only LLM output token distributions. Specifically, our method achieves this reversal by contrasting the output token distribution of a safety-aligned language model (e.g., Llama-2-chat) against its pre-trained version (e.g., Llama-2), so that the token predictions are shifted towards the opposite direction of safety alignment. We name this method emulated disalignment (ED) because sampling from this contrastive distribution provably emulates the result of fine-tuning to minimize a safety reward. Our experiments with ED across three evaluation datasets and four model families (Llama-1, Llama-2, Mistral, and Alpaca) show that ED doubles the harmfulness of pre-trained models and outperforms strong baselines, achieving the highest harmful rates in 43 out of 48 evaluation subsets by a large margin. Eventually, given ED's reliance on language model output token distributions, which particularly compromises open-source models, our findings highlight the need to reassess the open accessibility of language models, even if they have been safety-aligned. Code is available at https://github.com/ZHZisZZ/emulated-disalignment.

摘要: 大型语言模型(LLM)经过安全调整，以确保与人类的安全对话。然而，本文介绍了一种无需训练的攻击方法，该方法能够逆转安全对齐，通过仅访问LLM输出令牌分布来将更强对齐的结果转换为更大的潜在危害。具体地说，我们的方法通过将安全对齐的语言模型(例如，Llama-2-Chat)的输出令牌分布与其预先训练的版本(例如，Llama-2)进行比较，从而使令牌预测向安全对齐的相反方向移动，从而实现了这种逆转。我们将这种方法命名为模拟不对齐(ED)，因为从这种对比分布中进行的采样可以被证明是模拟了微调以最小化安全奖励的结果。我们在三个评估数据集和四个模型家族(骆驼-1、骆驼-2、米斯特拉尔和羊驼)上使用ED进行的实验表明，ED的危害性是预训练模型的两倍，并且性能优于强基线，在48个评估子集中的43个子集上获得了最高的伤害率。最终，鉴于ED对语言模型输出令牌分发的依赖，这尤其损害了开源模型，我们的发现突显了重新评估语言模型的开放可访问性的必要性，即使它们是安全一致的。代码可在https://github.com/ZHZisZZ/emulated-disalignment.上找到



## **4. Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs**

竞争报告：在一致的LLC中寻找通用越狱后门 cs.CL

Competition Report

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2404.14461v2) [paper-pdf](http://arxiv.org/pdf/2404.14461v2)

**Authors**: Javier Rando, Francesco Croce, Kryštof Mitka, Stepan Shabalin, Maksym Andriushchenko, Nicolas Flammarion, Florian Tramèr

**Abstract**: Large language models are aligned to be safe, preventing users from generating harmful content like misinformation or instructions for illegal activities. However, previous work has shown that the alignment process is vulnerable to poisoning attacks. Adversaries can manipulate the safety training data to inject backdoors that act like a universal sudo command: adding the backdoor string to any prompt enables harmful responses from models that, otherwise, behave safely. Our competition, co-located at IEEE SaTML 2024, challenged participants to find universal backdoors in several large language models. This report summarizes the key findings and promising ideas for future research.

摘要: 大型语言模型经过调整以确保安全，防止用户生成错误信息或非法活动指令等有害内容。然而，之前的工作表明，对齐过程很容易受到中毒攻击。对手可以操纵安全训练数据来注入类似于通用sudo命令的后门：将后门字符串添加到任何提示中都会导致模型做出有害响应，否则这些模型会安全地运行。我们的竞赛在IEEE SaTML 2024上举行，挑战参与者在几个大型语言模型中找到通用后门。本报告总结了关键发现和未来研究的有希望的想法。



## **5. AutoJailbreak: Exploring Jailbreak Attacks and Defenses through a Dependency Lens**

自动越狱：通过依赖的视角探索越狱攻击和防御 cs.CR

32 pages, 2 figures

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.03805v1) [paper-pdf](http://arxiv.org/pdf/2406.03805v1)

**Authors**: Lin Lu, Hai Yan, Zenghui Yuan, Jiawen Shi, Wenqi Wei, Pin-Yu Chen, Pan Zhou

**Abstract**: Jailbreak attacks in large language models (LLMs) entail inducing the models to generate content that breaches ethical and legal norm through the use of malicious prompts, posing a substantial threat to LLM security. Current strategies for jailbreak attack and defense often focus on optimizing locally within specific algorithmic frameworks, resulting in ineffective optimization and limited scalability. In this paper, we present a systematic analysis of the dependency relationships in jailbreak attack and defense techniques, generalizing them to all possible attack surfaces. We employ directed acyclic graphs (DAGs) to position and analyze existing jailbreak attacks, defenses, and evaluation methodologies, and propose three comprehensive, automated, and logical frameworks. \texttt{AutoAttack} investigates dependencies in two lines of jailbreak optimization strategies: genetic algorithm (GA)-based attacks and adversarial-generation-based attacks, respectively. We then introduce an ensemble jailbreak attack to exploit these dependencies. \texttt{AutoDefense} offers a mixture-of-defenders approach by leveraging the dependency relationships in pre-generative and post-generative defense strategies. \texttt{AutoEvaluation} introduces a novel evaluation method that distinguishes hallucinations, which are often overlooked, from jailbreak attack and defense responses. Through extensive experiments, we demonstrate that the proposed ensemble jailbreak attack and defense framework significantly outperforms existing research.

摘要: 大型语言模型(LLM)中的越狱攻击需要通过使用恶意提示诱导模型生成违反道德和法律规范的内容，从而对LLM安全构成重大威胁。当前的越狱攻防策略往往集中在特定算法框架内的局部优化，导致优化效果不佳，可扩展性有限。本文系统地分析了越狱攻防技术中的依赖关系，并将其推广到所有可能的攻击面。我们使用有向无环图(DAG)来定位和分析现有的越狱攻击、防御和评估方法，并提出了三个全面的、自动化的和逻辑的框架。Texttt{AutoAttack}研究了两种越狱优化策略的依赖关系：基于遗传算法(GA)的攻击和基于对抗性生成的攻击。然后，我们引入整体越狱攻击来利用这些依赖关系。通过利用生成前和生成后防御策略中的依赖关系，\exttt{AutoDefense}提供了混合防御者的方法。Texttt(自动评估)介绍了一种新的评估方法，可以将经常被忽视的幻觉与越狱攻击和防御反应区分开来。通过大量的实验，我们证明了所提出的集成越狱攻防框架的性能明显优于现有的研究。



## **6. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

保护语言模型免受越狱攻击的鲁棒即时优化 cs.LG

Code available at https://github.com/lapisrocks/rpo

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2401.17263v3) [paper-pdf](http://arxiv.org/pdf/2401.17263v3)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, large language models (LLMs) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries can modify prompts to induce unwanted behavior. While some defenses have been proposed, they have not been adapted to newly proposed attacks and more challenging threat models. To address this, we propose an optimization-based objective for defending LLMs against jailbreaking attacks and an algorithm, Robust Prompt Optimization (RPO) to create robust system-level defenses. Our approach directly incorporates the adversary into the defensive objective and optimizes a lightweight and transferable suffix, enabling RPO to adapt to worst-case adaptive attacks. Our theoretical and experimental results show improved robustness to both jailbreaks seen during optimization and unknown jailbreaks, reducing the attack success rate (ASR) on GPT-4 to 6% and Llama-2 to 0% on JailbreakBench, setting the state-of-the-art. Code can be found at https://github.com/lapisrocks/rpo

摘要: 尽管在人工智能对齐方面取得了进展，但大型语言模型(LLM)仍然容易受到对手攻击或越狱的攻击，在这些攻击或越狱中，对手可以修改提示以诱导不想要的行为。虽然已经提出了一些防御措施，但它们还没有适应新提出的攻击和更具挑战性的威胁模型。为了解决这个问题，我们提出了一个基于优化的目标来保护LLMS免受越狱攻击，并提出了一个算法--稳健提示优化(RPO)来创建强大的系统级防御。我们的方法直接将对手合并到防御目标中，并优化了一个轻量级和可转移的后缀，使RPO能够适应最坏情况的自适应攻击。我们的理论和实验结果表明，对于优化期间看到的越狱和未知越狱，我们都提高了健壮性，将GPT-4上的攻击成功率(ASR)降低到6%，将Llama-2上的攻击成功率降低到0%，从而达到了最先进的水平。代码可在https://github.com/lapisrocks/rpo上找到



## **7. Ranking Manipulation for Conversational Search Engines**

对话式搜索引擎的排名操纵 cs.CL

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03589v1) [paper-pdf](http://arxiv.org/pdf/2406.03589v1)

**Authors**: Samuel Pfrommer, Yatong Bai, Tanmay Gautam, Somayeh Sojoudi

**Abstract**: Major search engine providers are rapidly incorporating Large Language Model (LLM)-generated content in response to user queries. These conversational search engines operate by loading retrieved website text into the LLM context for summarization and interpretation. Recent research demonstrates that LLMs are highly vulnerable to jailbreaking and prompt injection attacks, which disrupt the safety and quality goals of LLMs using adversarial strings. This work investigates the impact of prompt injections on the ranking order of sources referenced by conversational search engines. To this end, we introduce a focused dataset of real-world consumer product websites and formalize conversational search ranking as an adversarial problem. Experimentally, we analyze conversational search rankings in the absence of adversarial injections and show that different LLMs vary significantly in prioritizing product name, document content, and context position. We then present a tree-of-attacks-based jailbreaking technique which reliably promotes low-ranked products. Importantly, these attacks transfer effectively to state-of-the-art conversational search engines such as perplexity.ai. Given the strong financial incentive for website owners to boost their search ranking, we argue that our problem formulation is of critical importance for future robustness work.

摘要: 各大搜索引擎提供商正在快速整合大型语言模型(LLM)生成的内容，以响应用户查询。这些对话式搜索引擎通过将检索到的网站文本加载到LLM上下文中进行操作以进行摘要和解释。最近的研究表明，LLM非常容易受到越狱和快速注入攻击，这些攻击使用敌意字符串破坏LLM的安全和质量目标。这项工作调查了提示注入对对话式搜索引擎引用的来源的排名顺序的影响。为此，我们引入了一个聚焦于真实世界消费产品网站的数据集，并将会话搜索排名形式化为一个对抗性问题。在实验上，我们分析了在没有对抗性注入的情况下的会话搜索排名，结果表明不同的LLM在产品名称、文档内容和上下文位置的优先顺序上存在显著差异。然后，我们提出了一种基于攻击树的越狱技术，该技术可靠地推广排名较低的产品。重要的是，这些攻击有效地转移到了最先进的会话搜索引擎，如Pplexity.ai。考虑到网站所有者有强大的经济动机来提高他们的搜索排名，我们认为我们的问题表达对于未来的稳健性工作至关重要。



## **8. Stealthy Attack on Large Language Model based Recommendation**

对基于大型语言模型的推荐的隐形攻击 cs.CL

ACL 2024 Main

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2402.14836v2) [paper-pdf](http://arxiv.org/pdf/2402.14836v2)

**Authors**: Jinghao Zhang, Yuting Liu, Qiang Liu, Shu Wu, Guibing Guo, Liang Wang

**Abstract**: Recently, the powerful large language models (LLMs) have been instrumental in propelling the progress of recommender systems (RS). However, while these systems have flourished, their susceptibility to security threats has been largely overlooked. In this work, we reveal that the introduction of LLMs into recommendation models presents new security vulnerabilities due to their emphasis on the textual content of items. We demonstrate that attackers can significantly boost an item's exposure by merely altering its textual content during the testing phase, without requiring direct interference with the model's training process. Additionally, the attack is notably stealthy, as it does not affect the overall recommendation performance and the modifications to the text are subtle, making it difficult for users and platforms to detect. Our comprehensive experiments across four mainstream LLM-based recommendation models demonstrate the superior efficacy and stealthiness of our approach. Our work unveils a significant security gap in LLM-based recommendation systems and paves the way for future research on protecting these systems.

摘要: 近年来，强大的大型语言模型(LLMS)在推动推荐系统(RS)的发展方面发挥了重要作用。然而，尽管这些系统蓬勃发展，但它们对安全威胁的敏感性在很大程度上被忽视了。在这项工作中，我们揭示了将LLMS引入推荐模型中会出现新的安全漏洞，这是因为它们强调项目的文本内容。我们证明，攻击者只需在测试阶段改变项目的文本内容，就可以显著增加项目的曝光率，而不需要直接干预模型的训练过程。此外，攻击具有明显的隐蔽性，因为它不会影响整体推荐性能，而且对文本的修改也很微妙，使得用户和平台很难检测到。我们对四个主流的基于LLM的推荐模型进行了全面的实验，证明了我们的方法具有优越的有效性和隐蔽性。我们的工作揭示了基于LLM的推荐系统中存在的一个显著的安全漏洞，并为未来保护这些系统的研究铺平了道路。



## **9. Improved Techniques for Optimization-Based Jailbreaking on Large Language Models**

基于优化的大型语言模型越狱改进技术 cs.LG

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2405.21018v2) [paper-pdf](http://arxiv.org/pdf/2405.21018v2)

**Authors**: Xiaojun Jia, Tianyu Pang, Chao Du, Yihao Huang, Jindong Gu, Yang Liu, Xiaochun Cao, Min Lin

**Abstract**: Large language models (LLMs) are being rapidly developed, and a key component of their widespread deployment is their safety-related alignment. Many red-teaming efforts aim to jailbreak LLMs, where among these efforts, the Greedy Coordinate Gradient (GCG) attack's success has led to a growing interest in the study of optimization-based jailbreaking techniques. Although GCG is a significant milestone, its attacking efficiency remains unsatisfactory. In this paper, we present several improved (empirical) techniques for optimization-based jailbreaks like GCG. We first observe that the single target template of "Sure" largely limits the attacking performance of GCG; given this, we propose to apply diverse target templates containing harmful self-suggestion and/or guidance to mislead LLMs. Besides, from the optimization aspects, we propose an automatic multi-coordinate updating strategy in GCG (i.e., adaptively deciding how many tokens to replace in each step) to accelerate convergence, as well as tricks like easy-to-hard initialisation. Then, we combine these improved technologies to develop an efficient jailbreak method, dubbed I-GCG. In our experiments, we evaluate on a series of benchmarks (such as NeurIPS 2023 Red Teaming Track). The results demonstrate that our improved techniques can help GCG outperform state-of-the-art jailbreaking attacks and achieve nearly 100% attack success rate. The code is released at https://github.com/jiaxiaojunQAQ/I-GCG.

摘要: 大型语言模型(LLM)正在迅速开发，其广泛部署的一个关键组件是与安全相关的一致性。许多红色团队的目标是越狱LLM，其中贪婪坐标梯度(GCG)攻击的成功导致了人们对基于优化的越狱技术的研究越来越感兴趣。虽然GCG是一个重要的里程碑，但其攻击效率仍然不能令人满意。在这篇文章中，我们提出了几种改进的(经验)技术，用于基于优化的越狱，如GCG。我们首先观察到单一目标模板“Sure”在很大程度上限制了GCG的攻击性能；鉴于此，我们建议使用包含有害自我暗示和/或引导的不同目标模板来误导LLM。此外，在优化方面，我们提出了GCG中的自动多坐标更新策略(即自适应地决定每一步需要替换多少个令牌)来加速收敛，以及容易初始化等技巧。然后，我们结合这些改进的技术开发了一种高效的越狱方法，称为I-GCG。在我们的实验中，我们在一系列基准(例如NeurIPS 2023 Red Teaming Track)上进行了评估。结果表明，改进后的技术可以帮助GCG超越最先进的越狱攻击，并获得近100%的攻击成功率。该代码在https://github.com/jiaxiaojunQAQ/I-GCG.上发布



## **10. CR-UTP: Certified Robustness against Universal Text Perturbations on Large Language Models**

CR-GPT：针对大型语言模型上通用文本扰动的鲁棒性认证 cs.CL

Accepted by ACL Findings 2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.01873v2) [paper-pdf](http://arxiv.org/pdf/2406.01873v2)

**Authors**: Qian Lou, Xin Liang, Jiaqi Xue, Yancheng Zhang, Rui Xie, Mengxin Zheng

**Abstract**: It is imperative to ensure the stability of every prediction made by a language model; that is, a language's prediction should remain consistent despite minor input variations, like word substitutions. In this paper, we investigate the problem of certifying a language model's robustness against Universal Text Perturbations (UTPs), which have been widely used in universal adversarial attacks and backdoor attacks. Existing certified robustness based on random smoothing has shown considerable promise in certifying the input-specific text perturbations (ISTPs), operating under the assumption that any random alteration of a sample's clean or adversarial words would negate the impact of sample-wise perturbations. However, with UTPs, masking only the adversarial words can eliminate the attack. A naive method is to simply increase the masking ratio and the likelihood of masking attack tokens, but it leads to a significant reduction in both certified accuracy and the certified radius due to input corruption by extensive masking. To solve this challenge, we introduce a novel approach, the superior prompt search method, designed to identify a superior prompt that maintains higher certified accuracy under extensive masking. Additionally, we theoretically motivate why ensembles are a particularly suitable choice as base prompts for random smoothing. The method is denoted by superior prompt ensembling technique. We also empirically confirm this technique, obtaining state-of-the-art results in multiple settings. These methodologies, for the first time, enable high certified accuracy against both UTPs and ISTPs. The source code of CR-UTP is available at \url {https://github.com/UCFML-Research/CR-UTP}.

摘要: 必须确保语言模型做出的每个预测的稳定性；也就是说，语言的预测应该保持一致，尽管输入有微小的变化，如单词替换。在本文中，我们研究了语言模型对通用文本扰动(UTP)的稳健性证明问题，UTP被广泛应用于通用对抗性攻击和后门攻击。现有的基于随机平滑的已证明的稳健性在证明特定于输入的文本扰动(ISTP)方面显示出相当大的前景，其操作是在假设样本的干净或敌意的单词的任何随机改变将否定样本方面的扰动的影响的情况下进行的。然而，对于UTP，只屏蔽敌意的单词就可以消除攻击。一种天真的方法是简单地增加掩蔽率和掩蔽攻击令牌的可能性，但由于广泛的掩蔽导致输入损坏，它导致认证的准确性和认证的半径都显著降低。为了解决这一挑战，我们引入了一种新的方法，高级提示搜索方法，旨在识别在广泛掩蔽下保持更高认证准确率的高级提示。此外，我们从理论上解释了为什么作为随机平滑的基础提示，集合是特别合适的选择。这种方法以卓越的即时集成技术表示。我们还从经验上证实了这一技术，在多个环境下获得了最先进的结果。这些方法首次针对UTP和ISTP实现了高度认证的准确性。CR-UTP的源代码可在\url{https://github.com/UCFML-Research/CR-UTP}.



## **11. Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models**

稳健的CLIP：稳健的大型视觉语言模型的视觉嵌入的无监督对抗微调 cs.LG

ICML 2024 Oral

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2402.12336v2) [paper-pdf](http://arxiv.org/pdf/2402.12336v2)

**Authors**: Christian Schlarmann, Naman Deep Singh, Francesco Croce, Matthias Hein

**Abstract**: Multi-modal foundation models like OpenFlamingo, LLaVA, and GPT-4 are increasingly used for various real-world tasks. Prior work has shown that these models are highly vulnerable to adversarial attacks on the vision modality. These attacks can be leveraged to spread fake information or defraud users, and thus pose a significant risk, which makes the robustness of large multi-modal foundation models a pressing problem. The CLIP model, or one of its variants, is used as a frozen vision encoder in many large vision-language models (LVLMs), e.g. LLaVA and OpenFlamingo. We propose an unsupervised adversarial fine-tuning scheme to obtain a robust CLIP vision encoder, which yields robustness on all vision down-stream tasks (LVLMs, zero-shot classification) that rely on CLIP. In particular, we show that stealth-attacks on users of LVLMs by a malicious third party providing manipulated images are no longer possible once one replaces the original CLIP model with our robust one. No retraining or fine-tuning of the down-stream LVLMs is required. The code and robust models are available at https://github.com/chs20/RobustVLM

摘要: OpenFlamingo、LLaVA和GPT-4等多模式基础模型越来越多地用于各种实际任务。先前的工作表明，这些模型非常容易受到视觉通道的对抗性攻击。这些攻击可以被用来传播虚假信息或欺骗用户，从而构成巨大的风险，这使得大型多通道基础模型的健壮性成为一个紧迫的问题。在许多大型视觉语言模型(如LLaVA和OpenFlamingo)中，剪辑模型或其变体之一被用作冻结的视觉编码器。我们提出了一种无监督的对抗性微调方案，以获得一个健壮的裁剪视觉编码器，它对依赖于裁剪的所有视觉下游任务(LVLM，零镜头分类)都具有健壮性。特别是，我们表明，一旦用我们的健壮模型取代了原始的剪辑模型，恶意第三方提供的篡改图像就不再可能对LVLMS的用户进行秘密攻击。不需要对下游低成本模块进行再培训或微调。代码和健壮模型可在https://github.com/chs20/RobustVLM上获得



## **12. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

利用剩余流激活分析防御大型语言模型免受攻击 cs.CR

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03230v1) [paper-pdf](http://arxiv.org/pdf/2406.03230v1)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply an established methodology for analyzing distinctive activation patterns in the residual streams for a novel result of attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.

摘要: 大型语言模型(LLM)的广泛采用，如OpenAI的ChatGPT，使防御这些模型上的对手威胁成为当务之急。这些攻击通过引入恶意输入来操纵LLM的输出，破坏了模型的完整性和用户对其输出的信任。为了应对这一挑战，我们的论文提出了一种创新的防御策略，在白盒访问LLM的情况下，该策略利用LLM变压器层之间的剩余激活分析。我们应用已建立的方法来分析残留流中不同的激活模式，以获得攻击提示分类的新结果。我们精选了多个数据集，以演示此分类方法如何在多种类型的攻击场景中具有高精度，包括我们新创建的攻击数据集。此外，我们通过集成LLMS的安全微调技术来增强模型的弹性，以衡量其对我们检测攻击的能力的影响。这些结果强调了我们的方法在加强对敌对输入的检测和缓解、推进LLMS运作的安全框架方面的有效性。



## **13. Text Embedding Inversion Security for Multilingual Language Models**

多语言语言模型的文本嵌入翻转安全性 cs.CL

18 pages, 17 Tables, 6 Figures

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2401.12192v4) [paper-pdf](http://arxiv.org/pdf/2401.12192v4)

**Authors**: Yiyi Chen, Heather Lent, Johannes Bjerva

**Abstract**: Textual data is often represented as real-numbered embeddings in NLP, particularly with the popularity of large language models (LLMs) and Embeddings as a Service (EaaS). However, storing sensitive information as embeddings can be susceptible to security breaches, as research shows that text can be reconstructed from embeddings, even without knowledge of the underlying model. While defence mechanisms have been explored, these are exclusively focused on English, leaving other languages potentially exposed to attacks. This work explores LLM security through multilingual embedding inversion. We define the problem of black-box multilingual and cross-lingual inversion attacks, and explore their potential implications. Our findings suggest that multilingual LLMs may be more vulnerable to inversion attacks, in part because English-based defences may be ineffective. To alleviate this, we propose a simple masking defense effective for both monolingual and multilingual models. This study is the first to investigate multilingual inversion attacks, shedding light on the differences in attacks and defenses across monolingual and multilingual settings.

摘要: 文本数据通常在NLP中表示为实数嵌入，特别是随着大型语言模型(LLM)和嵌入即服务(EaaS)的流行。然而，将敏感信息存储为嵌入很容易受到安全漏洞的影响，因为研究表明，即使不知道底层模型，也可以从嵌入中重构文本。虽然已经探索了防御机制，但这些机制完全集中在英语上，使其他语言可能面临攻击。该工作通过多语言嵌入倒置来探索LLM安全性。我们定义了黑盒多语言和跨语言倒置攻击的问题，并探讨了它们的潜在含义。我们的发现表明，多语种LLM可能更容易受到倒置攻击，部分原因是基于英语的防御可能无效。为了缓解这一问题，我们提出了一种简单的掩蔽防御方法，既适用于单语言模型，也适用于多语言模型。这项研究是对多语言倒置攻击的第一次调查，揭示了单语和多语环境下攻击和防御的差异。



## **14. BadAgent: Inserting and Activating Backdoor Attacks in LLM Agents**

BadAgent：在LLM代理中插入并激活后门攻击 cs.CL

Accepted by ACL 2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03007v1) [paper-pdf](http://arxiv.org/pdf/2406.03007v1)

**Authors**: Yifei Wang, Dizhan Xue, Shengjie Zhang, Shengsheng Qian

**Abstract**: With the prosperity of large language models (LLMs), powerful LLM-based intelligent agents have been developed to provide customized services with a set of user-defined tools. State-of-the-art methods for constructing LLM agents adopt trained LLMs and further fine-tune them on data for the agent task. However, we show that such methods are vulnerable to our proposed backdoor attacks named BadAgent on various agent tasks, where a backdoor can be embedded by fine-tuning on the backdoor data. At test time, the attacker can manipulate the deployed LLM agents to execute harmful operations by showing the trigger in the agent input or environment. To our surprise, our proposed attack methods are extremely robust even after fine-tuning on trustworthy data. Though backdoor attacks have been studied extensively in natural language processing, to the best of our knowledge, we could be the first to study them on LLM agents that are more dangerous due to the permission to use external tools. Our work demonstrates the clear risk of constructing LLM agents based on untrusted LLMs or data. Our code is public at https://github.com/DPamK/BadAgent

摘要: 随着大型语言模型(LLM)的繁荣，基于大型语言模型的智能代理被开发出来，以提供一套用户定义的工具来提供定制的服务。构建LLM代理的最先进方法采用经过训练的LLM，并根据代理任务的数据进一步微调它们。然而，我们发现这些方法容易受到我们提出的针对各种代理任务的名为BadAgent的后门攻击，其中可以通过对后门数据进行微调来嵌入后门。在测试时，攻击者可以通过在代理输入或环境中显示触发器来操纵部署的LLM代理执行有害操作。令我们惊讶的是，即使在对可信数据进行微调之后，我们提出的攻击方法也非常健壮。虽然后门攻击在自然语言处理中已经得到了广泛的研究，但就我们所知，我们可能是第一个在LLM代理上研究它们的，这些代理由于被允许使用外部工具而更危险。我们的工作证明了基于不可信的LLM或数据构建LLM代理的明显风险。我们的代码在https://github.com/DPamK/BadAgent上是公开的



## **15. Large Language Models are Few-shot Generators: Proposing Hybrid Prompt Algorithm To Generate Webshell Escape Samples**

大型语言模型是少镜头生成器：提出混合提示算法来生成Webshell Escape示例 cs.CR

21 pages, 17 figures

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2402.07408v2) [paper-pdf](http://arxiv.org/pdf/2402.07408v2)

**Authors**: Mingrui Ma, Lansheng Han, Chunjie Zhou

**Abstract**: The frequent occurrence of cyber-attacks has made webshell attacks and defense gradually become a research hotspot in the field of network security. However, the lack of publicly available benchmark datasets and the over-reliance on manually defined rules for webshell escape sample generation have slowed down the progress of research related to webshell escape sample generation and artificial intelligence (AI)-based webshell detection. To address the drawbacks of weak webshell sample escape capabilities, the lack of webshell datasets with complex malicious features, and to promote the development of webshell detection, we propose the Hybrid Prompt algorithm for webshell escape sample generation with the help of large language models. As a prompt algorithm specifically developed for webshell sample generation, the Hybrid Prompt algorithm not only combines various prompt ideas including Chain of Thought, Tree of Thought, but also incorporates various components such as webshell hierarchical module and few-shot example to facilitate the LLM in learning and reasoning webshell escape strategies. Experimental results show that the Hybrid Prompt algorithm can work with multiple LLMs with excellent code reasoning ability to generate high-quality webshell samples with high Escape Rate (88.61% with GPT-4 model on VirusTotal detection engine) and (Survival Rate 54.98% with GPT-4 model).

摘要: 网络攻击的频繁发生使网络外壳攻击与防御逐渐成为网络安全领域的研究热点。然而，缺乏公开可用的基准数据集，以及过度依赖人工定义的网页外壳逃逸样本生成规则，减缓了与网页外壳逃逸样本生成和基于人工智能(AI)的网页外壳检测相关的研究进展。针对Web外壳样本逃逸能力弱、缺乏具有复杂恶意特征的Web外壳数据集的缺点，为促进Web外壳检测的发展，本文提出了一种基于大型语言模型的Web外壳逃逸样本混合提示生成算法。混合提示算法是专门为Web外壳样本生成而开发的一种提示算法，它不仅结合了链式、树型等多种提示思想，还加入了Web外壳分层模块、少镜头实例等多种组件，方便了LLM在学习和推理Web外壳逃逸策略方面的应用。实验结果表明，混合提示算法能够与具有良好代码推理能力的多个LLMS协同工作，生成高逃逸率(在VirusTotal检测引擎上的GPT-4模型为88.61%)和(GPT-4模型的存活率为54.98%)的高质量Web外壳样本。



## **16. Enhancing Jailbreak Attack Against Large Language Models through Silent Tokens**

通过无声令牌增强针对大型语言模型的越狱攻击 cs.AI

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2405.20653v2) [paper-pdf](http://arxiv.org/pdf/2405.20653v2)

**Authors**: Jiahao Yu, Haozheng Luo, Jerry Yao-Chieh Hu, Wenbo Guo, Han Liu, Xinyu Xing

**Abstract**: Along with the remarkable successes of Language language models, recent research also started to explore the security threats of LLMs, including jailbreaking attacks. Attackers carefully craft jailbreaking prompts such that a target LLM will respond to the harmful question. Existing jailbreaking attacks require either human experts or leveraging complicated algorithms to craft jailbreaking prompts. In this paper, we introduce BOOST, a simple attack that leverages only the eos tokens. We demonstrate that rather than constructing complicated jailbreaking prompts, the attacker can simply append a few eos tokens to the end of a harmful question. It will bypass the safety alignment of LLMs and lead to successful jailbreaking attacks. We further apply BOOST to four representative jailbreak methods and show that the attack success rates of these methods can be significantly enhanced by simply adding eos tokens to the prompt. To understand this simple but novel phenomenon, we conduct empirical analyses. Our analysis reveals that adding eos tokens makes the target LLM believe the input is much less harmful, and eos tokens have low attention values and do not affect LLM's understanding of the harmful questions, leading the model to actually respond to the questions. Our findings uncover how fragile an LLM is against jailbreak attacks, motivating the development of strong safety alignment approaches.

摘要: 在语言模型取得显著成功的同时，最近的研究也开始探索LLMS的安全威胁，包括越狱攻击。攻击者精心设计越狱提示，以便目标LLM会对这个有害的问题做出回应。现有的越狱攻击要么需要人类专家，要么需要利用复杂的算法来设计越狱提示。在本文中，我们将介绍Boost，这是一种仅利用Eos令牌的简单攻击。我们演示了，攻击者可以简单地在有害问题的末尾添加几个Eos令牌，而不是构建复杂的越狱提示。它将绕过LLMS的安全对准，并导致成功的越狱攻击。我们进一步将Boost应用于四种具有代表性的越狱方法，并表明只需在提示符中添加Eos令牌即可显着提高这些方法的攻击成功率。为了理解这一简单但新颖的现象，我们进行了实证分析。我们的分析表明，添加Eos标记会使目标LLM认为输入的危害要小得多，并且Eos标记的关注值较低，不会影响LLM对有害问题的理解，从而导致模型实际回答问题。我们的发现揭示了LLM对越狱攻击的脆弱性，促使了强大的安全对齐方法的发展。



## **17. Large Language Models Spot Phishing Emails with Surprising Accuracy: A Comparative Analysis of Performance**

大型语言模型以惊人的准确性发现网络钓鱼电子邮件：性能比较分析 cs.CL

7 pages, 3 figures

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2404.15485v2) [paper-pdf](http://arxiv.org/pdf/2404.15485v2)

**Authors**: Het Patel, Umair Rehman, Farkhund Iqbal

**Abstract**: Phishing, a prevalent cybercrime tactic for decades, remains a significant threat in today's digital world. By leveraging clever social engineering elements and modern technology, cybercrime targets many individuals, businesses, and organizations to exploit trust and security. These cyber-attackers are often disguised in many trustworthy forms to appear as legitimate sources. By cleverly using psychological elements like urgency, fear, social proof, and other manipulative strategies, phishers can lure individuals into revealing sensitive and personalized information. Building on this pervasive issue within modern technology, this paper aims to analyze the effectiveness of 15 Large Language Models (LLMs) in detecting phishing attempts, specifically focusing on a randomized set of "419 Scam" emails. The objective is to determine which LLMs can accurately detect phishing emails by analyzing a text file containing email metadata based on predefined criteria. The experiment concluded that the following models, ChatGPT 3.5, GPT-3.5-Turbo-Instruct, and ChatGPT, were the most effective in detecting phishing emails.

摘要: 网络钓鱼是几十年来流行的一种网络犯罪策略，在当今的数字世界中仍然是一个重大威胁。通过利用聪明的社会工程元素和现代技术，网络犯罪以许多个人、企业和组织为目标，以利用信任和安全。这些网络攻击者往往以许多可信的形式伪装成合法的来源。通过巧妙地使用紧急、恐惧、社会证明和其他操纵策略等心理因素，网络钓鱼者可以诱使个人泄露敏感和个性化的信息。基于这一现代技术中普遍存在的问题，本文旨在分析15个大型语言模型(LLM)在检测网络钓鱼尝试方面的有效性，特别是关注一组随机的“419骗局”电子邮件。目标是通过基于预定义标准分析包含电子邮件元数据的文本文件，确定哪些LLM可以准确检测钓鱼电子邮件。实验得出的结论是，ChatGPT 3.5、GPT-3.5-Turbo-Indict和ChatGPT模型在检测钓鱼电子邮件方面最有效。



## **18. Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models**

水印能在翻译中幸存吗？大型语言模型文本水印的跨语言一致性研究 cs.CL

ACL 2024 (main conference)

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2402.14007v2) [paper-pdf](http://arxiv.org/pdf/2402.14007v2)

**Authors**: Zhiwei He, Binglin Zhou, Hongkun Hao, Aiwei Liu, Xing Wang, Zhaopeng Tu, Zhuosheng Zhang, Rui Wang

**Abstract**: Text watermarking technology aims to tag and identify content produced by large language models (LLMs) to prevent misuse. In this study, we introduce the concept of cross-lingual consistency in text watermarking, which assesses the ability of text watermarks to maintain their effectiveness after being translated into other languages. Preliminary empirical results from two LLMs and three watermarking methods reveal that current text watermarking technologies lack consistency when texts are translated into various languages. Based on this observation, we propose a Cross-lingual Watermark Removal Attack (CWRA) to bypass watermarking by first obtaining a response from an LLM in a pivot language, which is then translated into the target language. CWRA can effectively remove watermarks, decreasing the AUCs to a random-guessing level without performance loss. Furthermore, we analyze two key factors that contribute to the cross-lingual consistency in text watermarking and propose X-SIR as a defense method against CWRA. Code: https://github.com/zwhe99/X-SIR.

摘要: 文本水印技术旨在标记和识别大型语言模型(LLM)产生的内容，以防止误用。在这项研究中，我们在文本水印中引入了跨语言一致性的概念，用来评估文本水印在翻译成其他语言后保持其有效性的能力。两种LLMS和三种水印方法的初步实验结果表明，现有的文本水印技术在文本翻译成各种语言时缺乏一致性。基于这一观察结果，我们提出了一种跨语言水印移除攻击(CWRA)，通过首先从旋转语言的LLM获得响应，然后将其翻译成目标语言来绕过水印。CWRA可以有效地去除水印，在不损失性能的情况下将AUC降低到随机猜测的水平。此外，我们分析了影响文本水印跨语言一致性的两个关键因素，并提出了X-SIR作为一种针对CWRA的防御方法。代码：https://github.com/zwhe99/X-SIR.



## **19. QROA: A Black-Box Query-Response Optimization Attack on LLMs**

QROA：对LLM的黑匣子查询响应优化攻击 cs.CL

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.02044v1) [paper-pdf](http://arxiv.org/pdf/2406.02044v1)

**Authors**: Hussein Jawad, Nicolas J. -B. BRUNEL

**Abstract**: Large Language Models (LLMs) have surged in popularity in recent months, yet they possess concerning capabilities for generating harmful content when manipulated. This study introduces the Query-Response Optimization Attack (QROA), an optimization-based strategy designed to exploit LLMs through a black-box, query-only interaction. QROA adds an optimized trigger to a malicious instruction to compel the LLM to generate harmful content. Unlike previous approaches, QROA does not require access to the model's logit information or any other internal data and operates solely through the standard query-response interface of LLMs. Inspired by deep Q-learning and Greedy coordinate descent, the method iteratively updates tokens to maximize a designed reward function. We tested our method on various LLMs such as Vicuna, Falcon, and Mistral, achieving an Attack Success Rate (ASR) over 80\%. We also tested the model against Llama2-chat, the fine-tuned version of Llama2 designed to resist Jailbreak attacks, achieving good ASR with a suboptimal initial trigger seed. This study demonstrates the feasibility of generating jailbreak attacks against deployed LLMs in the public domain using black-box optimization methods, enabling more comprehensive safety testing of LLMs.

摘要: 近几个月来，大型语言模型(LLM)越来越受欢迎，但它们具有在被操纵时生成有害内容的令人担忧的能力。这项研究介绍了查询-响应优化攻击(QROA)，这是一种基于优化的策略，旨在通过黑盒、仅查询的交互来利用LLMS。QROA向恶意指令添加了优化的触发器，以迫使LLM生成有害内容。与以前的方法不同，QROA不需要访问模型的Logit信息或任何其他内部数据，只通过LLMS的标准查询-响应接口进行操作。受深度Q学习和贪婪坐标下降的启发，该方法迭代更新令牌以最大化所设计的奖励函数。我们在维库纳、猎鹰和米斯特拉尔等不同的LLMS上测试了我们的方法，取得了80%以上的攻击成功率(ASR)。我们还在Llama2-Chat上测试了该模型，Llama2-Chat是Llama2的微调版本，旨在抵抗越狱攻击，使用次优的初始触发种子实现了良好的ASR。这项研究论证了利用黑盒优化方法对部署在公共领域的LLM进行越狱攻击的可行性，从而实现了对LLM进行更全面的安全测试。



## **20. Bileve: Securing Text Provenance in Large Language Models Against Spoofing with Bi-level Signature**

Bileve：通过双层签名保护大型语言模型中的文本出处，防止欺骗 cs.CR

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.01946v1) [paper-pdf](http://arxiv.org/pdf/2406.01946v1)

**Authors**: Tong Zhou, Xuandong Zhao, Xiaolin Xu, Shaolei Ren

**Abstract**: Text watermarks for large language models (LLMs) have been commonly used to identify the origins of machine-generated content, which is promising for assessing liability when combating deepfake or harmful content. While existing watermarking techniques typically prioritize robustness against removal attacks, unfortunately, they are vulnerable to spoofing attacks: malicious actors can subtly alter the meanings of LLM-generated responses or even forge harmful content, potentially misattributing blame to the LLM developer. To overcome this, we introduce a bi-level signature scheme, Bileve, which embeds fine-grained signature bits for integrity checks (mitigating spoofing attacks) as well as a coarse-grained signal to trace text sources when the signature is invalid (enhancing detectability) via a novel rank-based sampling strategy. Compared to conventional watermark detectors that only output binary results, Bileve can differentiate 5 scenarios during detection, reliably tracing text provenance and regulating LLMs. The experiments conducted on OPT-1.3B and LLaMA-7B demonstrate the effectiveness of Bileve in defeating spoofing attacks with enhanced detectability.

摘要: 大型语言模型(LLM)的文本水印通常用于识别机器生成内容的来源，这有望在打击深度虚假或有害内容时评估责任。虽然现有的水印技术通常将健壮性放在免受删除攻击的优先位置，但不幸的是，它们容易受到欺骗性攻击：恶意行为者可以巧妙地更改LLM生成的响应的含义，甚至伪造有害内容，可能会将责任错误地归咎于LLM开发人员。为了克服这一问题，我们提出了一种双层签名方案BiLEVE，该方案通过一种新颖的基于等级的采样策略嵌入细粒度的签名比特用于完整性检查(缓解欺骗攻击)，并在签名无效时嵌入粗粒度的信号来跟踪文本来源(增强了可检测性)。与传统的只输出二进制结果的水印检测器相比，BiLEVE在检测过程中可以区分5种场景，可靠地追踪文本来源和规范LLM。在OPT-1.3B和LLAMA-7B上进行的实验证明了BiLEVE在抵抗欺骗攻击方面的有效性，并增强了可检测性。



## **21. ASETF: A Novel Method for Jailbreak Attack on LLMs through Translate Suffix Embeddings**

ASTF：一种通过翻译后缀嵌入对LLM进行越狱攻击的新方法 cs.CL

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2402.16006v2) [paper-pdf](http://arxiv.org/pdf/2402.16006v2)

**Authors**: Hao Wang, Hao Li, Minlie Huang, Lei Sha

**Abstract**: The safety defense methods of Large language models(LLMs) stays limited because the dangerous prompts are manually curated to just few known attack types, which fails to keep pace with emerging varieties. Recent studies found that attaching suffixes to harmful instructions can hack the defense of LLMs and lead to dangerous outputs. However, similar to traditional text adversarial attacks, this approach, while effective, is limited by the challenge of the discrete tokens. This gradient based discrete optimization attack requires over 100,000 LLM calls, and due to the unreadable of adversarial suffixes, it can be relatively easily penetrated by common defense methods such as perplexity filters. To cope with this challenge, in this paper, we proposes an Adversarial Suffix Embedding Translation Framework (ASETF), aimed at transforming continuous adversarial suffix embeddings into coherent and understandable text. This method greatly reduces the computational overhead during the attack process and helps to automatically generate multiple adversarial samples, which can be used as data to strengthen LLMs security defense. Experimental evaluations were conducted on Llama2, Vicuna, and other prominent LLMs, employing harmful directives sourced from the Advbench dataset. The results indicate that our method significantly reduces the computation time of adversarial suffixes and achieves a much better attack success rate to existing techniques, while significantly enhancing the textual fluency of the prompts. In addition, our approach can be generalized into a broader method for generating transferable adversarial suffixes that can successfully attack multiple LLMs, even black-box LLMs, such as ChatGPT and Gemini.

摘要: 大型语言模型(LLM)的安全防御方法仍然有限，因为危险的提示是手动管理到少数已知的攻击类型，无法跟上新兴的变体。最近的研究发现，在有害指令上附加后缀可能会破坏LLMS的防御，并导致危险的输出。然而，与传统的文本对抗性攻击类似，该方法虽然有效，但受到离散令牌挑战的限制。这种基于梯度的离散优化攻击需要超过100,000个LLM调用，并且由于敌意后缀的不可读，它可以相对容易地被困惑过滤器等常见防御方法穿透。为了应对这一挑战，本文提出了一个对抗性后缀嵌入翻译框架(ASETF)，旨在将连续的对抗性后缀嵌入转换成连贯的、可理解的文本。该方法大大减少了攻击过程中的计算开销，有助于自动生成多个对抗性样本，作为加强LLMS安全防御的数据。在Llama2、Vicuna和其他著名的LLM上进行了实验评估，采用了来自Advbench数据集的有害指令。实验结果表明，该方法显著减少了对抗性后缀的计算时间，取得了比现有技术更好的攻击成功率，同时显著提高了提示的文本流畅性。此外，我们的方法可以推广到更广泛的方法来生成可转移的敌意后缀，可以成功地攻击多个LLM，甚至可以攻击黑盒LLM，如ChatGPT和Gemini。



## **22. HoneyGPT: Breaking the Trilemma in Terminal Honeypots with Large Language Model**

HoneyGPT：用大型语言模型打破终端蜜罐中的三重困境 cs.CR

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.01882v1) [paper-pdf](http://arxiv.org/pdf/2406.01882v1)

**Authors**: Ziyang Wang, Jianzhou You, Haining Wang, Tianwei Yuan, Shichao Lv, Yang Wang, Limin Sun

**Abstract**: Honeypots, as a strategic cyber-deception mechanism designed to emulate authentic interactions and bait unauthorized entities, continue to struggle with balancing flexibility, interaction depth, and deceptive capability despite their evolution over decades. Often they also lack the capability of proactively adapting to an attacker's evolving tactics, which restricts the depth of engagement and subsequent information gathering. Under this context, the emergent capabilities of large language models, in tandem with pioneering prompt-based engineering techniques, offer a transformative shift in the design and deployment of honeypot technologies. In this paper, we introduce HoneyGPT, a pioneering honeypot architecture based on ChatGPT, heralding a new era of intelligent honeypot solutions characterized by their cost-effectiveness, high adaptability, and enhanced interactivity, coupled with a predisposition for proactive attacker engagement. Furthermore, we present a structured prompt engineering framework that augments long-term interaction memory and robust security analytics. This framework, integrating thought of chain tactics attuned to honeypot contexts, enhances interactivity and deception, deepens security analytics, and ensures sustained engagement.   The evaluation of HoneyGPT includes two parts: a baseline comparison based on a collected dataset and a field evaluation in real scenarios for four weeks. The baseline comparison demonstrates HoneyGPT's remarkable ability to strike a balance among flexibility, interaction depth, and deceptive capability. The field evaluation further validates HoneyGPT's efficacy, showing its marked superiority in enticing attackers into more profound interactive engagements and capturing a wider array of novel attack vectors in comparison to existing honeypot technologies.

摘要: 蜜罐作为一种战略性的网络欺骗机制，旨在模拟真实的交互并诱骗未经授权的实体，尽管经过了几十年的演变，但它仍然在灵活性、交互深度和欺骗性能力之间进行权衡。他们往往也缺乏主动适应攻击者不断变化的战术的能力，这限制了接触的深度和随后的信息收集。在这种背景下，大型语言模型的新兴能力与开创性的基于提示的工程技术相结合，为蜜罐技术的设计和部署提供了革命性的转变。在本文中，我们介绍了一种基于ChatGPT的开创性蜜罐架构HoneyGPT，它预示着智能蜜罐解决方案的新时代，其特点是性价比高、适应性强、互动性增强，并易于主动攻击。此外，我们还提出了结构化提示工程框架，以增强长期交互记忆和强大的安全分析能力。该框架整合了与蜜罐环境相协调的链策略思想，增强了互动性和欺骗性，深化了安全分析，并确保了持续参与。HoneyGPT的评估包括两部分：基于收集的数据集的基线比较和为期四周的真实场景下的现场评估。基准比较表明，HoneyGPT在灵活性、交互深度和欺骗性能力之间取得了显著的平衡。现场评估进一步验证了HoneyGPT的有效性，显示出其显著的优势，与现有的蜜罐技术相比，它可以诱使攻击者参与更深入的互动，并捕获更广泛的新型攻击载体。



## **23. Safeguarding Large Language Models: A Survey**

保护大型语言模型：一项调查 cs.CR

under review. arXiv admin note: text overlap with arXiv:2402.01822

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.02622v1) [paper-pdf](http://arxiv.org/pdf/2406.02622v1)

**Authors**: Yi Dong, Ronghui Mu, Yanghao Zhang, Siqi Sun, Tianle Zhang, Changshun Wu, Gaojie Jin, Yi Qi, Jinwei Hu, Jie Meng, Saddek Bensalem, Xiaowei Huang

**Abstract**: In the burgeoning field of Large Language Models (LLMs), developing a robust safety mechanism, colloquially known as "safeguards" or "guardrails", has become imperative to ensure the ethical use of LLMs within prescribed boundaries. This article provides a systematic literature review on the current status of this critical mechanism. It discusses its major challenges and how it can be enhanced into a comprehensive mechanism dealing with ethical issues in various contexts. First, the paper elucidates the current landscape of safeguarding mechanisms that major LLM service providers and the open-source community employ. This is followed by the techniques to evaluate, analyze, and enhance some (un)desirable properties that a guardrail might want to enforce, such as hallucinations, fairness, privacy, and so on. Based on them, we review techniques to circumvent these controls (i.e., attacks), to defend the attacks, and to reinforce the guardrails. While the techniques mentioned above represent the current status and the active research trends, we also discuss several challenges that cannot be easily dealt with by the methods and present our vision on how to implement a comprehensive guardrail through the full consideration of multi-disciplinary approach, neural-symbolic method, and systems development lifecycle.

摘要: 在新兴的大型语言模型(LLM)领域，开发一种强大的安全机制，俗称“保障措施”或“护栏”，已成为确保在规定范围内合乎道德地使用LLM的当务之急。本文对这一关键机制的研究现状进行了系统的文献综述。报告讨论了其面临的主要挑战，以及如何将其加强为在各种情况下处理道德问题的综合机制。首先，本文阐述了主要的LLM服务提供商和开源社区使用的保护机制的现状。然后是评估、分析和增强护栏可能想要强制执行的一些(不想要的)属性的技术，例如幻觉、公平性、隐私等等。在此基础上，我们回顾了规避这些控制(即攻击)、防御攻击和加固护栏的技术。虽然上述技术代表了当前的研究现状和活跃的研究趋势，但我们也讨论了这些方法不容易解决的几个挑战，并提出了我们的愿景，即如何通过充分考虑多学科方法、神经符号方法和系统开发生命周期来实现一个全面的护栏。



## **24. Here's a Free Lunch: Sanitizing Backdoored Models with Model Merge**

这是免费午餐：通过模型合并对后门模型进行消毒 cs.CL

accepted to ACL2024 (Findings)

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2402.19334v2) [paper-pdf](http://arxiv.org/pdf/2402.19334v2)

**Authors**: Ansh Arora, Xuanli He, Maximilian Mozes, Srinibas Swain, Mark Dras, Qiongkai Xu

**Abstract**: The democratization of pre-trained language models through open-source initiatives has rapidly advanced innovation and expanded access to cutting-edge technologies. However, this openness also brings significant security risks, including backdoor attacks, where hidden malicious behaviors are triggered by specific inputs, compromising natural language processing (NLP) system integrity and reliability. This paper suggests that merging a backdoored model with other homogeneous models can significantly remediate backdoor vulnerabilities even if such models are not entirely secure. In our experiments, we verify our hypothesis on various models (BERT-Base, RoBERTa-Large, Llama2-7B, and Mistral-7B) and datasets (SST-2, OLID, AG News, and QNLI). Compared to multiple advanced defensive approaches, our method offers an effective and efficient inference-stage defense against backdoor attacks on classification and instruction-tuned tasks without additional resources or specific knowledge. Our approach consistently outperforms recent advanced baselines, leading to an average of about 75% reduction in the attack success rate. Since model merging has been an established approach for improving model performance, the extra advantage it provides regarding defense can be seen as a cost-free bonus.

摘要: 通过开放源码倡议使预先培训的语言模型民主化，迅速推动了创新，扩大了获得尖端技术的机会。然而，这种开放性也带来了重大的安全风险，包括后门攻击，其中隐藏的恶意行为由特定的输入触发，损害了自然语言处理(NLP)系统的完整性和可靠性。本文认为，将后门模型与其他同类模型合并可以显著补救后门漏洞，即使这些模型不是完全安全的。在我们的实验中，我们在各种模型(Bert-Base、Roberta-Large、Llama2-7B和Mistral-7B)和数据集(SST-2、OLID、AG News和QNLI)上验证了我们的假设。与多种先进的防御方法相比，该方法在不需要额外资源或特定知识的情况下，提供了一种有效的推理阶段防御对分类和指令调优任务的后门攻击。我们的方法始终优于最近的先进基准，导致攻击成功率平均降低约75%。由于模型合并已经成为提高模型性能的既定方法，它提供的关于防御的额外优势可以被视为免费的额外奖励。



## **25. Human vs. Machine: Behavioral Differences Between Expert Humans and Language Models in Wargame Simulations**

人类与机器：战争游戏模拟中专家人类和语言模型之间的行为差异 cs.CY

Updated with new plot and more details

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2403.03407v2) [paper-pdf](http://arxiv.org/pdf/2403.03407v2)

**Authors**: Max Lamparth, Anthony Corso, Jacob Ganz, Oriana Skylar Mastro, Jacquelyn Schneider, Harold Trinkunas

**Abstract**: To some, the advent of artificial intelligence (AI) promises better decision-making and increased military effectiveness while reducing the influence of human error and emotions. However, there is still debate about how AI systems, especially large language models (LLMs), behave compared to humans in high-stakes military decision-making scenarios with the potential for increased risks towards escalation and unnecessary conflicts. To test this potential and scrutinize the use of LLMs for such purposes, we use a new wargame experiment with 107 national security experts designed to look at crisis escalation in a fictional US-China scenario and compare human players to LLM-simulated responses in separate simulations. Wargames have a long history in the development of military strategy and the response of nations to threats or attacks. Here, we show a considerable high-level agreement in the LLM and human responses and significant quantitative and qualitative differences in individual actions and strategic tendencies. These differences depend on intrinsic biases in LLMs regarding the appropriate level of violence following strategic instructions, the choice of LLM, and whether the LLMs are tasked to decide for a team of players directly or first to simulate dialog between players. When simulating the dialog, the discussions lack quality and maintain a farcical harmony. The LLM simulations cannot account for human player characteristics, showing no significant difference even for extreme traits, such as "pacifist" or "aggressive sociopath". Our results motivate policymakers to be cautious before granting autonomy or following AI-based strategy recommendations.

摘要: 对一些人来说，人工智能(AI)的出现保证了更好的决策和更高的军事效力，同时减少了人为错误和情绪的影响。然而，在高风险的军事决策场景中，人工智能系统，特别是大型语言模型(LLM)与人类相比表现如何，可能会增加升级和不必要的冲突的风险，仍存在争议。为了测试这一潜力并仔细审查LLM在此类目的中的使用，我们使用了一个与107名国家安全专家进行的新的军事游戏实验，旨在观察虚构的美国-中国场景中的危机升级，并在不同的模拟中将人类玩家与LLM模拟的反应进行比较。军事演习在军事战略的发展和国家对威胁或攻击的反应方面有着悠久的历史。在这里，我们显示了LLM和人类反应的相当高水平的一致性，以及个别行动和战略趋势的显著数量和质量差异。这些差异取决于LLM中关于遵循战略指令的适当暴力级别的内在偏见，LLM的选择，以及LLM是否被要求直接或首先决定一支球员团队来模拟球员之间的对话。在模拟对话时，讨论缺乏质量，保持了滑稽的和谐。LLM模拟不能解释人类玩家的特征，即使是极端的特征，如“和平主义者”或“攻击性反社会者”，也没有显示出显著的差异。我们的结果促使政策制定者在授予自治权或遵循基于人工智能的战略建议之前保持谨慎。



## **26. PrivacyRestore: Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restoration**

PrivacyRestore：通过隐私删除和恢复在大型语言模型中保留隐私的推理 cs.CR

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01394v1) [paper-pdf](http://arxiv.org/pdf/2406.01394v1)

**Authors**: Ziqian Zeng, Jianwei Wang, Zhengdong Lu, Huiping Zhuang, Cen Chen

**Abstract**: The widespread usage of online Large Language Models (LLMs) inference services has raised significant privacy concerns about the potential exposure of private information in user inputs to eavesdroppers or untrustworthy service providers. Existing privacy protection methods for LLMs suffer from insufficient privacy protection, performance degradation, or severe inference time overhead. In this paper, we propose PrivacyRestore to protect the privacy of user inputs during LLM inference. PrivacyRestore directly removes privacy spans in user inputs and restores privacy information via activation steering during inference. The privacy spans are encoded as restoration vectors. We propose Attention-aware Weighted Aggregation (AWA) which aggregates restoration vectors of all privacy spans in the input into a meta restoration vector. AWA not only ensures proper representation of all privacy spans but also prevents attackers from inferring the privacy spans from the meta restoration vector alone. This meta restoration vector, along with the query with privacy spans removed, is then sent to the server. The experimental results show that PrivacyRestore can protect private information while maintaining acceptable levels of performance and inference efficiency.

摘要: 在线大语言模型(LLMS)推理服务的广泛使用引发了人们对用户输入中的私人信息可能暴露给窃听者或不可信的服务提供商的严重隐私担忧。现有的LLMS隐私保护方法存在隐私保护不足、性能下降或严重的推理时间开销等问题。在本文中，我们提出PrivacyRestore来保护LLM推理过程中用户输入的隐私。PrivacyRestore直接删除用户输入中的隐私范围，并在推理过程中通过激活控制恢复隐私信息。隐私跨度被编码为恢复向量。我们提出了注意力感知加权聚集(AWA)，它将输入中所有隐私跨度的恢复向量聚合成一个元恢复向量。AWA不仅确保了所有隐私范围的正确表示，而且还防止攻击者仅从元恢复向量来推断隐私范围。然后，将该元恢复向量与去除了隐私跨度的查询一起发送到服务器。实验结果表明，PrivacyRestore能够在保持可接受的性能和推理效率的同时保护私有信息。



## **27. Privacy in LLM-based Recommendation: Recent Advances and Future Directions**

基于法学硕士的建议中的隐私：最近的进展和未来的方向 cs.CL

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01363v1) [paper-pdf](http://arxiv.org/pdf/2406.01363v1)

**Authors**: Sichun Luo, Wei Shao, Yuxuan Yao, Jian Xu, Mingyang Liu, Qintong Li, Bowei He, Maolin Wang, Guanzhi Deng, Hanxu Hou, Xinyi Zhang, Linqi Song

**Abstract**: Nowadays, large language models (LLMs) have been integrated with conventional recommendation models to improve recommendation performance. However, while most of the existing works have focused on improving the model performance, the privacy issue has only received comparatively less attention. In this paper, we review recent advancements in privacy within LLM-based recommendation, categorizing them into privacy attacks and protection mechanisms. Additionally, we highlight several challenges and propose future directions for the community to address these critical problems.

摘要: 如今，大型语言模型（LLM）已与传统推荐模型集成以提高推荐性能。然而，虽然大多数现有作品都专注于提高模型性能，但隐私问题受到的关注相对较少。在本文中，我们回顾了基于LLM的推荐中隐私方面的最新进展，并将其分为隐私攻击和保护机制。此外，我们还强调了几项挑战，并为社区解决这些关键问题提出了未来的方向。



## **28. DepsRAG: Towards Managing Software Dependencies using Large Language Models**

DepsRAG：使用大型语言模型管理软件附属机构 cs.SE

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2405.20455v2) [paper-pdf](http://arxiv.org/pdf/2405.20455v2)

**Authors**: Mohannad Alhanahnah, Yazan Boshmaf, Benoit Baudry

**Abstract**: Managing software dependencies is a crucial maintenance task in software development and is becoming a rapidly growing research field, especially in light of the significant increase in software supply chain attacks. Specialized expertise and substantial developer effort are required to fully comprehend dependencies and reveal hidden properties about the dependencies (e.g., number of dependencies, dependency chains, depth of dependencies).   Recent advancements in Large Language Models (LLMs) allow the retrieval of information from various data sources for response generation, thus providing a new opportunity to uniquely manage software dependencies. To highlight the potential of this technology, we present~\tool, a proof-of-concept Retrieval Augmented Generation (RAG) approach that constructs direct and transitive dependencies of software packages as a Knowledge Graph (KG) in four popular software ecosystems. DepsRAG can answer user questions about software dependencies by automatically generating necessary queries to retrieve information from the KG, and then augmenting the input of LLMs with the retrieved information. DepsRAG can also perform Web search to answer questions that the LLM cannot directly answer via the KG. We identify tangible benefits that DepsRAG can offer and discuss its limitations.

摘要: 管理软件依赖关系是软件开发中一项重要的维护任务，特别是在软件供应链攻击显著增加的情况下，软件依赖关系管理正在成为一个快速增长的研究领域。要完全理解依赖关系并揭示依赖关系的隐藏属性(例如，依赖关系的数量、依赖关系链、依赖关系的深度)，需要专业的专业知识和大量的开发人员工作。大型语言模型(LLM)的最新进展允许从各种数据源检索信息以生成响应，从而为独特地管理软件依赖关系提供了新的机会。为了突出这一技术的潜力，我们提出了一种概念验证检索增强生成(RAG)方法~\Tool，它将软件包的直接和传递依赖构造为四个流行的软件生态系统中的知识图(KG)。DepsRAG可以通过自动生成必要的查询来从KG中检索信息，然后用检索到的信息增强LLMS的输入，来回答用户关于软件依赖性的问题。DepsRAG还可以执行网络搜索，以回答LLM无法通过KG直接回答的问题。我们确定了DepsRAG可以提供的实实在在的好处并讨论了其局限性。



## **29. Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models**

通过对基于LLM的排队模型的对抗攻击探索决策级的鲁棒性 cs.MM

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2405.19802v2) [paper-pdf](http://arxiv.org/pdf/2405.19802v2)

**Authors**: Shuyuan Liu, Jiawei Chen, Shouwei Ruan, Hang Su, Zhaoxia Yin

**Abstract**: Embodied intelligence empowers agents with a profound sense of perception, enabling them to respond in a manner closely aligned with real-world situations. Large Language Models (LLMs) delve into language instructions with depth, serving a crucial role in generating plans for intricate tasks. Thus, LLM-based embodied models further enhance the agent's capacity to comprehend and process information. However, this amalgamation also ushers in new challenges in the pursuit of heightened intelligence. Specifically, attackers can manipulate LLMs to produce irrelevant or even malicious outputs by altering their prompts. Confronted with this challenge, we observe a notable absence of multi-modal datasets essential for comprehensively evaluating the robustness of LLM-based embodied models. Consequently, we construct the Embodied Intelligent Robot Attack Dataset (EIRAD), tailored specifically for robustness evaluation. Additionally, two attack strategies are devised, including untargeted attacks and targeted attacks, to effectively simulate a range of diverse attack scenarios. At the same time, during the attack process, to more accurately ascertain whether our method is successful in attacking the LLM-based embodied model, we devise a new attack success evaluation method utilizing the BLIP2 model. Recognizing the time and cost-intensive nature of the GCG algorithm in attacks, we devise a scheme for prompt suffix initialization based on various target tasks, thus expediting the convergence process. Experimental results demonstrate that our method exhibits a superior attack success rate when targeting LLM-based embodied models, indicating a lower level of decision-level robustness in these models.

摘要: 具身智能使特工具有深刻的感知力，使他们能够以与现实世界情况密切一致的方式做出反应。大型语言模型(LLM)深入研究语言指令，在为复杂任务制定计划方面发挥着至关重要的作用。因此，基于LLM的具体化模型进一步增强了代理理解和处理信息的能力。然而，这种融合也带来了追求高智商的新挑战。具体地说，攻击者可以通过更改提示来操纵LLMS生成无关甚至恶意的输出。面对这一挑战，我们注意到明显缺乏全面评估基于LLM的体现模型的稳健性所必需的多模式数据集。因此，我们构建了专门为健壮性评估量身定做的具体化智能机器人攻击数据集(Eirad)。此外，设计了两种攻击策略，包括非定向攻击和定向攻击，以有效地模拟一系列不同的攻击场景。同时，在攻击过程中，为了更准确地确定我们的方法在攻击基于LLM的体现模型上是否成功，我们设计了一种新的利用BLIP2模型的攻击成功评估方法。考虑到GCG算法在攻击中的时间和成本密集性，我们设计了一种基于不同目标任务的快速后缀初始化方案，从而加快了收敛过程。实验结果表明，我们的方法在攻击基于LLM的具体模型时表现出了较高的攻击成功率，表明这些模型具有较低的决策级健壮性。



## **30. Fundamental Limitations of Alignment in Large Language Models**

大型语言模型中对齐的基本局限性 cs.CL

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2304.11082v6) [paper-pdf](http://arxiv.org/pdf/2304.11082v6)

**Authors**: Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that within the limits of this framework, for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates an undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback make the LLM prone to being prompted into the undesired behaviors. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.

摘要: 开发与人类交互的语言模型的一个重要方面是使他们的行为对人类用户有用而无害。这通常是通过调整模型来实现的，这种方式增强了期望的行为，抑制了不期望的行为，这一过程称为对齐。在本文中，我们提出了一种名为行为期望界限(BEB)的理论方法，它允许我们正式地研究大型语言模型中对齐的几个固有特征和限制。重要的是，我们证明了在这个框架的范围内，对于模型所表现出的任何有限概率的行为，存在可以触发模型输出该行为的提示，其概率随着提示的长度的增加而增加。这意味着，任何减弱不受欢迎的行为但不能完全消除它的对准过程，在对抗提示攻击时都是不安全的。此外，我们的框架暗示了一种机制，通过这种机制，领先的对齐方法，如来自人类反馈的强化学习，使得LLM容易被提示进入不希望看到的行为。这一理论结果正在由所谓的当代“聊天GPT越狱”大规模实验证明，在这种情况下，敌对用户通过触发LLM充当恶意角色来欺骗LLM打破其对齐护栏。我们的结果暴露了LLM对齐方面的根本限制，并将设计可靠的机制以确保人工智能安全的必要性放在了首位。



## **31. Are AI-Generated Text Detectors Robust to Adversarial Perturbations?**

人工智能生成的文本检测器对对抗性扰动是否稳健？ cs.CL

Accepted to ACL 2024 main conference

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01179v1) [paper-pdf](http://arxiv.org/pdf/2406.01179v1)

**Authors**: Guanhua Huang, Yuchen Zhang, Zhe Li, Yongjian You, Mingze Wang, Zhouwang Yang

**Abstract**: The widespread use of large language models (LLMs) has sparked concerns about the potential misuse of AI-generated text, as these models can produce content that closely resembles human-generated text. Current detectors for AI-generated text (AIGT) lack robustness against adversarial perturbations, with even minor changes in characters or words causing a reversal in distinguishing between human-created and AI-generated text. This paper investigates the robustness of existing AIGT detection methods and introduces a novel detector, the Siamese Calibrated Reconstruction Network (SCRN). The SCRN employs a reconstruction network to add and remove noise from text, extracting a semantic representation that is robust to local perturbations. We also propose a siamese calibration technique to train the model to make equally confidence predictions under different noise, which improves the model's robustness against adversarial perturbations. Experiments on four publicly available datasets show that the SCRN outperforms all baseline methods, achieving 6.5\%-18.25\% absolute accuracy improvement over the best baseline method under adversarial attacks. Moreover, it exhibits superior generalizability in cross-domain, cross-genre, and mixed-source scenarios. The code is available at \url{https://github.com/CarlanLark/Robust-AIGC-Detector}.

摘要: 大型语言模型(LLM)的广泛使用引发了人们对人工智能生成的文本可能被滥用的担忧，因为这些模型可以生成与人类生成的文本非常相似的内容。目前的人工智能生成文本检测器(AIGT)缺乏对对手扰动的稳健性，即使是字符或单词的微小变化也会导致在区分人工生成文本和人工智能生成文本方面出现逆转。本文研究了现有的AIGT检测方法的稳健性，并介绍了一种新的检测器--暹罗校准重建网络(SCRN)。SCRN使用重构网络来添加和去除文本中的噪声，提取对局部扰动具有鲁棒性的语义表示。我们还提出了一种暹罗校正技术来训练模型，使其在不同的噪声下做出相同的置信度预测，从而提高了模型对对抗性扰动的鲁棒性。在四个公开可用的数据集上的实验表明，SCRN的性能优于所有的基线方法，在对抗性攻击下，其绝对准确率比最佳基线方法提高了6.5-18.25。此外，它在跨域、跨流派和混合来源的场景中表现出出色的泛化能力。代码可在\url{https://github.com/CarlanLark/Robust-AIGC-Detector}.上获得



## **32. Genshin: General Shield for Natural Language Processing with Large Language Models**

Genshin：具有大型语言模型的自然语言处理的通用盾牌 cs.CL

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2405.18741v2) [paper-pdf](http://arxiv.org/pdf/2405.18741v2)

**Authors**: Xiao Peng, Tao Liu, Ying Wang

**Abstract**: Large language models (LLMs) like ChatGPT, Gemini, or LLaMA have been trending recently, demonstrating considerable advancement and generalizability power in countless domains. However, LLMs create an even bigger black box exacerbating opacity, with interpretability limited to few approaches. The uncertainty and opacity embedded in LLMs' nature restrict their application in high-stakes domains like financial fraud, phishing, etc. Current approaches mainly rely on traditional textual classification with posterior interpretable algorithms, suffering from attackers who may create versatile adversarial samples to break the system's defense, forcing users to make trade-offs between efficiency and robustness. To address this issue, we propose a novel cascading framework called Genshin (General Shield for Natural Language Processing with Large Language Models), utilizing LLMs as defensive one-time plug-ins. Unlike most applications of LLMs that try to transform text into something new or structural, Genshin uses LLMs to recover text to its original state. Genshin aims to combine the generalizability of the LLM, the discrimination of the median model, and the interpretability of the simple model. Our experiments on the task of sentimental analysis and spam detection have shown fatal flaws of the current median models and exhilarating results on LLMs' recovery ability, demonstrating that Genshin is both effective and efficient. In our ablation study, we unearth several intriguing observations. Utilizing the LLM defender, a tool derived from the 4th paradigm, we have reproduced BERT's 15% optimal mask rate results in the 3rd paradigm of NLP. Additionally, when employing the LLM as a potential adversarial tool, attackers are capable of executing effective attacks that are nearly semantically lossless.

摘要: 像ChatGPT、Gemini或Llama这样的大型语言模型(LLM)最近已经成为趋势，在无数领域展示了相当大的先进性和泛化能力。然而，LLM创建了一个更大的黑匣子，加剧了不透明度，可解释性仅限于几种方法。LLMS本质上的不确定性和不透明性限制了它们在高风险领域的应用，如金融欺诈、网络钓鱼等。目前的方法主要依赖于传统的文本分类和后验可解释算法，攻击者可能会创建通用的对抗性样本来破坏系统的防御，迫使用户在效率和健壮性之间做出权衡。为了解决这个问题，我们提出了一种新颖的级联框架Genshin(General Shield For Natural Language Processing With Large Language Models)，利用LLMS作为防御性的一次性插件。与大多数试图将文本转换为新的或结构化的文本的LLMS应用程序不同，Genshin使用LLMS将文本恢复到其原始状态。Genshin的目标是将LLM的泛化能力、中值模型的区分性和简单模型的可解释性结合起来。我们在情感分析和垃圾邮件检测任务上的实验表明，现有的中值模型存在致命缺陷，并且在LLMS的恢复能力上取得了令人振奋的结果，证明了Genshin是有效的和高效的。在我们的消融研究中，我们发现了几个有趣的观察结果。利用LLM Defender，一个源自第四范式的工具，我们在NLP的第三范式中复制了Bert的15%最优掩蔽率结果。此外，当使用LLM作为潜在的敌意工具时，攻击者能够执行几乎在语义上无损的有效攻击。



## **33. Data Contamination Calibration for Black-box LLMs**

黑匣子LLM的数据污染校准 cs.LG

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2405.11930v2) [paper-pdf](http://arxiv.org/pdf/2405.11930v2)

**Authors**: Wentao Ye, Jiaqi Hu, Liyao Li, Haobo Wang, Gang Chen, Junbo Zhao

**Abstract**: The rapid advancements of Large Language Models (LLMs) tightly associate with the expansion of the training data size. However, the unchecked ultra-large-scale training sets introduce a series of potential risks like data contamination, i.e. the benchmark data is used for training. In this work, we propose a holistic method named Polarized Augment Calibration (PAC) along with a new to-be-released dataset to detect the contaminated data and diminish the contamination effect. PAC extends the popular MIA (Membership Inference Attack) -- from machine learning community -- by forming a more global target at detecting training data to Clarify invisible training data. As a pioneering work, PAC is very much plug-and-play that can be integrated with most (if not all) current white- and black-box LLMs. By extensive experiments, PAC outperforms existing methods by at least 4.5%, towards data contamination detection on more 4 dataset formats, with more than 10 base LLMs. Besides, our application in real-world scenarios highlights the prominent presence of contamination and related issues.

摘要: 大型语言模型(LLM)的快速发展与训练数据规模的扩大密切相关。然而，未经核查的超大规模训练集带来了一系列潜在的风险，如数据污染，即基准数据被用于训练。在这项工作中，我们提出了一种称为极化增强校准(PAC)的整体方法以及一个新的即将发布的数据集来检测污染数据并减少污染影响。PAC扩展了流行的MIA(成员关系推断攻击)--来自机器学习社区--通过形成一个更全局的目标来检测训练数据，以澄清看不见的训练数据。作为一项开创性的工作，PAC是非常即插即用的，可以与大多数(如果不是全部)当前的白盒和黑盒LLM集成。通过大量实验，在4种以上的数据集格式、10个以上的基本最小似然模型上，PAC在数据污染检测方面的性能至少比现有方法高4.5%。此外，我们在现实世界场景中的应用突出了污染和相关问题的突出存在。



## **34. Cross-lingual Cross-temporal Summarization: Dataset, Models, Evaluation**

跨语言跨时态总结：数据集、模型、评估 cs.CL

Computational Linguistics. Submitted manuscript.  https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00519/121095/Cross-lingual-Cross-temporal-Summarization-Dataset

**SubmitDate**: 2024-06-02    [abs](http://arxiv.org/abs/2306.12916v3) [paper-pdf](http://arxiv.org/pdf/2306.12916v3)

**Authors**: Ran Zhang, Jihed Ouni, Steffen Eger

**Abstract**: While summarization has been extensively researched in natural language processing (NLP), cross-lingual cross-temporal summarization (CLCTS) is a largely unexplored area that has the potential to improve cross-cultural accessibility and understanding. This paper comprehensively addresses the CLCTS task, including dataset creation, modeling, and evaluation. We (1) build the first CLCTS corpus with 328 instances for hDe-En (extended version with 455 instances) and 289 for hEn-De (extended version with 501 instances), leveraging historical fiction texts and Wikipedia summaries in English and German; (2) examine the effectiveness of popular transformer end-to-end models with different intermediate finetuning tasks; (3) explore the potential of GPT-3.5 as a summarizer; (4) report evaluations from humans, GPT-4, and several recent automatic evaluation metrics. Our results indicate that intermediate task finetuned end-to-end models generate bad to moderate quality summaries while GPT-3.5, as a zero-shot summarizer, provides moderate to good quality outputs. GPT-3.5 also seems very adept at normalizing historical text. To assess data contamination in GPT-3.5, we design an adversarial attack scheme in which we find that GPT-3.5 performs slightly worse for unseen source documents compared to seen documents. Moreover, it sometimes hallucinates when the source sentences are inverted against its prior knowledge with a summarization accuracy of 0.67 for plot omission, 0.71 for entity swap, and 0.53 for plot negation. Overall, our regression results of model performances suggest that longer, older, and more complex source texts (all of which are more characteristic for historical language variants) are harder to summarize for all models, indicating the difficulty of the CLCTS task.

摘要: 摘要在自然语言处理(NLP)中得到了广泛的研究，但跨语言跨时序摘要(CLCTS)在很大程度上是一个未被开发的领域，它有可能提高跨文化的可及性和理解力。本文全面介绍了CLCTS的任务，包括数据集的创建、建模和评估。我们(1)建立了第一个CLCTS语料库，包含328个HDE-EN(扩展版本，455个实例)和289个HEN-De(扩展版本，501个实例)，利用历史小说文本和英语和德语的维基百科摘要；(2)检查具有不同中间微调任务的流行变压器端到端模型的有效性；(3)探索GPT-3.5作为摘要生成器的潜力；(4)报告来自人类、GPT-4和最近的几个自动评估指标的评估。我们的结果表明，中间任务精调的端到端模型生成的质量较差到中等的摘要，而GPT-3.5作为一个零概率摘要生成器，提供的是中等到良好的质量输出。GPT-3.5似乎也非常擅长将历史文本正常化。为了评估GPT-3.5中的数据污染，我们设计了一个对抗性攻击方案，在该方案中，我们发现GPT-3.5对于不可见的源文档的性能比对于可见文档的性能略差。此外，当原句与其先验知识倒置时，有时会产生幻觉，情节省略的摘要准确率为0.67，实体互换的摘要准确率为0.71，情节否定的摘要准确率为0.53。总体而言，我们对模型性能的回归结果表明，更长、更老、更复杂的源文本(所有这些都是历史语言变体的特征)更难对所有模型进行总结，这表明CLCTS任务的难度。



## **35. Are you still on track!? Catching LLM Task Drift with Activations**

你还在正轨上吗！？通过激活捕捉LLM任务漂移 cs.CR

**SubmitDate**: 2024-06-02    [abs](http://arxiv.org/abs/2406.00799v1) [paper-pdf](http://arxiv.org/pdf/2406.00799v1)

**Authors**: Sahar Abdelnabi, Aideen Fay, Giovanni Cherubin, Ahmed Salem, Mario Fritz, Andrew Paverd

**Abstract**: Large Language Models (LLMs) are routinely used in retrieval-augmented applications to orchestrate tasks and process inputs from users and other sources. These inputs, even in a single LLM interaction, can come from a variety of sources, of varying trustworthiness and provenance. This opens the door to prompt injection attacks, where the LLM receives and acts upon instructions from supposedly data-only sources, thus deviating from the user's original instructions. We define this as task drift, and we propose to catch it by scanning and analyzing the LLM's activations. We compare the LLM's activations before and after processing the external input in order to detect whether this input caused instruction drift. We develop two probing methods and find that simply using a linear classifier can detect drift with near perfect ROC AUC on an out-of-distribution test set. We show that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Our setup does not require any modification of the LLM (e.g., fine-tuning) or any text generation, thus maximizing deployability and cost efficiency and avoiding reliance on unreliable model output. To foster future research on activation-based task inspection, decoding, and interpretability, we will release our large-scale TaskTracker toolkit, comprising a dataset of over 500K instances, representations from 4 SoTA language models, and inspection tools.

摘要: 大型语言模型(LLM)通常用于检索增强的应用程序中，以协调任务并处理来自用户和其他来源的输入。这些输入，即使是在单个LLM交互中，也可以来自各种来源，具有不同的可信度和出处。这为即时注入攻击打开了大门，在这种情况下，LLM接收来自假定仅限数据的来源的指令并对其采取行动，从而偏离用户的原始指令。我们将其定义为任务漂移，并建议通过扫描和分析LLM的激活来捕获它。我们比较LLM在处理外部输入之前和之后的激活，以检测该输入是否导致指令漂移。我们开发了两种探测方法，发现简单地使用线性分类器可以在非分布测试集上以接近完美的ROC AUC来检测漂移。我们表明，这种方法对于看不见的任务领域(如提示注入、越狱和恶意指令)的泛化效果出奇地好，而且没有接受过任何这些攻击的培训。我们的设置不需要对LLM进行任何修改(例如，微调)或任何文本生成，从而最大限度地提高可部署性和成本效益，并避免依赖不可靠的模型输出。为了促进未来对基于激活的任务检测、解码和可解释性的研究，我们将发布我们的大型TaskTracker工具包，其中包括超过50万个实例的数据集、来自4个SOTA语言模型的表示和检测工具。



## **36. Transforming Computer Security and Public Trust Through the Exploration of Fine-Tuning Large Language Models**

通过探索微调大型语言模型来改变计算机安全和公众信任 cs.CL

A preprint, 17 pages. 11 images

**SubmitDate**: 2024-06-02    [abs](http://arxiv.org/abs/2406.00628v1) [paper-pdf](http://arxiv.org/pdf/2406.00628v1)

**Authors**: Garrett Crumrine, Izzat Alsmadi, Jesus Guerrero, Yuvaraj Munian

**Abstract**: Large language models (LLMs) have revolutionized how we interact with machines. However, this technological advancement has been paralleled by the emergence of "Mallas," malicious services operating underground that exploit LLMs for nefarious purposes. Such services create malware, phishing attacks, and deceptive websites, escalating the cyber security threats landscape. This paper delves into the proliferation of Mallas by examining the use of various pre-trained language models and their efficiency and vulnerabilities when misused. Building on a dataset from the Common Vulnerabilities and Exposures (CVE) program, it explores fine-tuning methodologies to generate code and explanatory text related to identified vulnerabilities. This research aims to shed light on the operational strategies and exploitation techniques of Mallas, leading to the development of more secure and trustworthy AI applications. The paper concludes by emphasizing the need for further research, enhanced safeguards, and ethical guidelines to mitigate the risks associated with the malicious application of LLMs.

摘要: 大型语言模型(LLM)彻底改变了我们与机器交互的方式。然而，与这种技术进步并行的是“Mallas”的出现，这是一种在地下运营的恶意服务，利用LLM达到邪恶的目的。这些服务创造了恶意软件、网络钓鱼攻击和欺骗性网站，加剧了网络安全威胁。本文通过检查各种预先训练的语言模型的使用及其误用时的效率和漏洞，深入探讨了Mallas的激增。它基于通用漏洞和暴露(CVE)计划的数据集，探索了微调方法来生成与已识别漏洞相关的代码和说明性文本。这项研究旨在揭示Mallas的运营战略和开发技术，从而开发出更安全、更可信的AI应用程序。文章最后强调了进一步研究、加强保障措施和道德准则的必要性，以降低与恶意应用低成本管理相关的风险。



## **37. Exploring Vulnerabilities and Protections in Large Language Models: A Survey**

探索大型语言模型中的漏洞和保护：一项调查 cs.LG

**SubmitDate**: 2024-06-01    [abs](http://arxiv.org/abs/2406.00240v1) [paper-pdf](http://arxiv.org/pdf/2406.00240v1)

**Authors**: Frank Weizhen Liu, Chenhui Hu

**Abstract**: As Large Language Models (LLMs) increasingly become key components in various AI applications, understanding their security vulnerabilities and the effectiveness of defense mechanisms is crucial. This survey examines the security challenges of LLMs, focusing on two main areas: Prompt Hacking and Adversarial Attacks, each with specific types of threats. Under Prompt Hacking, we explore Prompt Injection and Jailbreaking Attacks, discussing how they work, their potential impacts, and ways to mitigate them. Similarly, we analyze Adversarial Attacks, breaking them down into Data Poisoning Attacks and Backdoor Attacks. This structured examination helps us understand the relationships between these vulnerabilities and the defense strategies that can be implemented. The survey highlights these security challenges and discusses robust defensive frameworks to protect LLMs against these threats. By detailing these security issues, the survey contributes to the broader discussion on creating resilient AI systems that can resist sophisticated attacks.

摘要: 随着大型语言模型(LLM)日益成为各种人工智能应用程序的关键组件，了解它们的安全漏洞和防御机制的有效性至关重要。这项调查考察了LLMS的安全挑战，重点放在两个主要领域：即时黑客攻击和对抗性攻击，每种攻击都具有特定类型的威胁。在即时黑客攻击下，我们探讨了即时注入和越狱攻击，讨论了它们的工作原理、潜在影响以及缓解它们的方法。同样，我们分析对抗性攻击，将其细分为数据中毒攻击和后门攻击。这种结构化的检查有助于我们了解这些漏洞与可以实施的防御策略之间的关系。调查强调了这些安全挑战，并讨论了强大的防御框架，以保护小岛屿发展中国家免受这些威胁。通过详细描述这些安全问题，这项调查有助于更广泛地讨论创建具有弹性的人工智能系统，以抵御复杂的攻击。



## **38. ReEval: Automatic Hallucination Evaluation for Retrieval-Augmented Large Language Models via Transferable Adversarial Attacks**

ReEval：通过可转移对抗攻击对检索增强大型语言模型进行自动幻觉评估 cs.CL

NAACL 2024 Findings

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2310.12516v2) [paper-pdf](http://arxiv.org/pdf/2310.12516v2)

**Authors**: Xiaodong Yu, Hao Cheng, Xiaodong Liu, Dan Roth, Jianfeng Gao

**Abstract**: Despite remarkable advancements in mitigating hallucinations in large language models (LLMs) by retrieval augmentation, it remains challenging to measure the reliability of LLMs using static question-answering (QA) data. Specifically, given the potential of data contamination (e.g., leading to memorization), good static benchmark performance does not ensure that model can reliably use the provided evidence for responding, which is essential to avoid hallucination when the required knowledge is new or private. Inspired by adversarial machine learning, we investigate the feasibility of automatically perturbing existing static one for dynamic evaluation. Specifically, this paper presents ReEval, an LLM-based framework using prompt chaining to perturb the original evidence for generating new test cases for evaluating the LLMs' reliability in using new evidence for answering.   We implement ReEval using ChatGPT and evaluate the resulting variants of two popular open-domain QA datasets on a collection of LLMs under various prompting settings. Our generated data is human-readable and useful to trigger hallucination in LLM. Accurate models on static data are observed to produce unsupported answers from the perturbed evidence, with pronounced accuracy drops across LLMs including GPT-4. We find that our adversarial examples are transferable across all considered LLMs. The examples generated by a small model can be used to evaluate a much larger model, making our approach cost-effective.

摘要: 尽管在通过提取增强来缓解大语言模型(LLMS)中的幻觉方面取得了显著进展，但使用静态问答(QA)数据来衡量LLMS的可靠性仍然具有挑战性。具体地说，考虑到数据污染的可能性(例如，导致记忆)，良好的静态基准性能不能确保模型能够可靠地使用所提供的证据进行响应，这对于避免在所需知识是新的或私有的情况下产生幻觉至关重要。受对抗性机器学习的启发，我们研究了自动扰动现有静态机器学习进行动态评估的可行性。具体地说，本文提出了一种基于LLM的框架ReEval，该框架使用提示链来扰动原始证据以生成新的测试用例，以评估LLMS在使用新证据进行回答时的可靠性。我们使用ChatGPT实现了ReEval，并在不同的提示设置下对两个流行的开放领域QA数据集的结果变体进行了评估。我们生成的数据是人类可读的，对在LLM中引发幻觉很有用。对静态数据的准确模型被观察到从扰动的证据中产生不支持的答案，包括GPT-4在内的LLMS的准确性显著下降。我们发现，我们的对抗性例子可以在所有考虑的LLM之间转移。一个小模型生成的例子可以用来评估一个大得多的模型，这使得我们的方法具有成本效益。



## **39. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models**

TrojanRAG：检索增强生成可以成为大型语言模型中的后门驱动程序 cs.CR

19 pages, 14 figures, 4 tables

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.13401v3) [paper-pdf](http://arxiv.org/pdf/2405.13401v3)

**Authors**: Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Large language models (LLMs) have raised concerns about potential security threats despite performing significantly in Natural Language Processing (NLP). Backdoor attacks initially verified that LLM is doing substantial harm at all stages, but the cost and robustness have been criticized. Attacking LLMs is inherently risky in security review, while prohibitively expensive. Besides, the continuous iteration of LLMs will degrade the robustness of backdoors. In this paper, we propose TrojanRAG, which employs a joint backdoor attack in the Retrieval-Augmented Generation, thereby manipulating LLMs in universal attack scenarios. Specifically, the adversary constructs elaborate target contexts and trigger sets. Multiple pairs of backdoor shortcuts are orthogonally optimized by contrastive learning, thus constraining the triggering conditions to a parameter subspace to improve the matching. To improve the recall of the RAG for the target contexts, we introduce a knowledge graph to construct structured data to achieve hard matching at a fine-grained level. Moreover, we normalize the backdoor scenarios in LLMs to analyze the real harm caused by backdoors from both attackers' and users' perspectives and further verify whether the context is a favorable tool for jailbreaking models. Extensive experimental results on truthfulness, language understanding, and harmfulness show that TrojanRAG exhibits versatility threats while maintaining retrieval capabilities on normal queries.

摘要: 尽管大型语言模型(LLM)在自然语言处理(NLP)中表现出色，但仍引发了人们对潜在安全威胁的担忧。后门攻击最初证实了LLM在所有阶段都在造成实质性的危害，但其成本和健壮性受到了批评。在安全审查中，攻击LLMS固有的风险，同时代价高得令人望而却步。此外，LLMS的连续迭代会降低后门的健壮性。在本文中，我们提出了TrojanRAG，它在检索-增强生成中使用联合后门攻击，从而在通用攻击场景下操纵LLMS。具体地说，对手构建了精心设计的目标上下文和触发集。通过对比学习对多对后门捷径进行正交化优化，从而将触发条件约束到一个参数子空间以提高匹配性。为了提高RAG对目标上下文的查全率，我们引入了知识图来构建结构化数据，以实现细粒度的硬匹配。此外，我们对LLMS中的后门场景进行了规范化，从攻击者和用户的角度分析了后门造成的真实危害，并进一步验证了上下文是否为越狱模型的有利工具。在真实性、语言理解和危害性方面的大量实验结果表明，TrojanRAG在保持对正常查询的检索能力的同时，表现出通用性威胁。



## **40. Preemptive Answer "Attacks" on Chain-of-Thought Reasoning**

先发制人的回答“攻击”思维链推理 cs.CL

Accepted to ACL'24 (Findings). Camera-ready version

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20902v1) [paper-pdf](http://arxiv.org/pdf/2405.20902v1)

**Authors**: Rongwu Xu, Zehan Qi, Wei Xu

**Abstract**: Large language models (LLMs) showcase impressive reasoning capabilities when coupled with Chain-of-Thought (CoT) prompting. However, the robustness of this approach warrants further investigation. In this paper, we introduce a novel scenario termed preemptive answers, where the LLM obtains an answer before engaging in reasoning. This situation can arise inadvertently or induced by malicious users by prompt injection attacks. Experiments reveal that preemptive answers significantly impair the model's reasoning capability across various CoT methods and a broad spectrum of datasets. To bolster the robustness of reasoning, we propose two measures aimed at mitigating this issue to some extent.

摘要: 当与思想链（CoT）提示相结合时，大型语言模型（LLM）展示了令人印象深刻的推理能力。然而，这种方法的稳健性值得进一步研究。在本文中，我们引入了一种称为先发制人答案的新颖场景，其中LLM在进行推理之前获得答案。这种情况可能是无意中发生的，也可能是恶意用户通过提示注入攻击引起的。实验表明，先发制人的答案显着损害了模型在各种CoT方法和广泛数据集中的推理能力。为了增强推理的稳健性，我们提出了两项旨在在一定程度上缓解这一问题的措施。



## **41. Robustifying Safety-Aligned Large Language Models through Clean Data Curation**

通过干净的数据修复来优化安全一致的大型语言模型 cs.CR

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.19358v2) [paper-pdf](http://arxiv.org/pdf/2405.19358v2)

**Authors**: Xiaoqun Liu, Jiacheng Liang, Muchao Ye, Zhaohan Xi

**Abstract**: Large language models (LLMs) are vulnerable when trained on datasets containing harmful content, which leads to potential jailbreaking attacks in two scenarios: the integration of harmful texts within crowdsourced data used for pre-training and direct tampering with LLMs through fine-tuning. In both scenarios, adversaries can compromise the safety alignment of LLMs, exacerbating malfunctions. Motivated by the need to mitigate these adversarial influences, our research aims to enhance safety alignment by either neutralizing the impact of malicious texts in pre-training datasets or increasing the difficulty of jailbreaking during downstream fine-tuning. In this paper, we propose a data curation framework designed to counter adversarial impacts in both scenarios. Our method operates under the assumption that we have no prior knowledge of attack details, focusing solely on curating clean texts. We introduce an iterative process aimed at revising texts to reduce their perplexity as perceived by LLMs, while simultaneously preserving their text quality. By pre-training or fine-tuning LLMs with curated clean texts, we observe a notable improvement in LLM robustness regarding safety alignment against harmful queries. For instance, when pre-training LLMs using a crowdsourced dataset containing 5\% harmful instances, adding an equivalent amount of curated texts significantly mitigates the likelihood of providing harmful responses in LLMs and reduces the attack success rate by 71\%. Our study represents a significant step towards mitigating the risks associated with training-based jailbreaking and fortifying the secure utilization of LLMs.

摘要: 在包含有害内容的数据集上进行训练时，大型语言模型(LLM)很容易受到攻击，这会在两种情况下导致潜在的越狱攻击：将有害文本整合到用于预培训的众包数据中，以及通过微调直接篡改LLMS。在这两种情况下，对手都可能损害LLM的安全对准，从而加剧故障。出于缓解这些敌对影响的需要，我们的研究旨在通过中和预训练数据集中恶意文本的影响或在下游微调期间增加越狱的难度来增强安全一致性。在本文中，我们提出了一个数据管理框架，旨在对抗这两种情况下的对抗性影响。我们的方法是在假设我们事先不知道攻击细节的情况下运行的，只专注于策划干净的文本。我们引入了一种迭代过程，旨在修改文本以减少LLMS所感知的困惑，同时保持其文本质量。通过使用经过精选的干净文本预先训练或微调LLM，我们观察到LLM在针对有害查询的安全对齐方面的稳健性有了显著的改善。例如，当使用包含5个有害实例的众包数据集对LLMS进行预训练时，添加等量的精选文本可显著降低LLMS中提供有害响应的可能性，并将攻击成功率降低71%。我们的研究是朝着减少基于培训的越狱风险和加强低土地管理系统的安全利用迈出的重要一步。



## **42. Phantom: General Trigger Attacks on Retrieval Augmented Language Generation**

Phantom：对检索增强语言生成的通用触发攻击 cs.CR

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20485v1) [paper-pdf](http://arxiv.org/pdf/2405.20485v1)

**Authors**: Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Retrieval Augmented Generation (RAG) expands the capabilities of modern large language models (LLMs) in chatbot applications, enabling developers to adapt and personalize the LLM output without expensive training or fine-tuning. RAG systems use an external knowledge database to retrieve the most relevant documents for a given query, providing this context to the LLM generator. While RAG achieves impressive utility in many applications, its adoption to enable personalized generative models introduces new security risks. In this work, we propose new attack surfaces for an adversary to compromise a victim's RAG system, by injecting a single malicious document in its knowledge database. We design Phantom, general two-step attack framework against RAG augmented LLMs. The first step involves crafting a poisoned document designed to be retrieved by the RAG system within the top-k results only when an adversarial trigger, a specific sequence of words acting as backdoor, is present in the victim's queries. In the second step, a specially crafted adversarial string within the poisoned document triggers various adversarial attacks in the LLM generator, including denial of service, reputation damage, privacy violations, and harmful behaviors. We demonstrate our attacks on multiple LLM architectures, including Gemma, Vicuna, and Llama.

摘要: 检索增强生成(RAG)扩展了Chatbot应用程序中现代大型语言模型(LLM)的能力，使开发人员能够适应和个性化LLM输出，而无需昂贵的培训或微调。RAG系统使用外部知识数据库来检索与给定查询最相关的文档，并将此上下文提供给LLM生成器。虽然RAG在许多应用程序中实现了令人印象深刻的实用性，但采用它来支持个性化的生成模型带来了新的安全风险。在这项工作中，我们提出了新的攻击面，通过在受害者的知识库中注入单个恶意文档来危害受害者的RAG系统。我们设计了一个针对RAG扩展的LLMS的Phantom通用两步攻击框架。第一步涉及精心设计一个有毒文档，仅当受害者的查询中出现敌对触发器(充当后门的特定单词序列)时，RAG系统才会在top-k结果中检索到。在第二步中，有毒文档中巧尽心思构建的敌意字符串会在LLM生成器中触发各种敌意攻击，包括拒绝服务、声誉损害、侵犯隐私和有害行为。我们演示了我们对多个LLM体系结构的攻击，包括Gema、Vicuna和Llama。



## **43. Jailbreaking Large Language Models Against Moderation Guardrails via Cipher Characters**

通过密码字符破解大型语言模型对抗调节护栏 cs.CR

20 pages

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20413v1) [paper-pdf](http://arxiv.org/pdf/2405.20413v1)

**Authors**: Haibo Jin, Andy Zhou, Joe D. Menke, Haohan Wang

**Abstract**: Large Language Models (LLMs) are typically harmless but remain vulnerable to carefully crafted prompts known as ``jailbreaks'', which can bypass protective measures and induce harmful behavior. Recent advancements in LLMs have incorporated moderation guardrails that can filter outputs, which trigger processing errors for certain malicious questions. Existing red-teaming benchmarks often neglect to include questions that trigger moderation guardrails, making it difficult to evaluate jailbreak effectiveness. To address this issue, we introduce JAMBench, a harmful behavior benchmark designed to trigger and evaluate moderation guardrails. JAMBench involves 160 manually crafted instructions covering four major risk categories at multiple severity levels. Furthermore, we propose a jailbreak method, JAM (Jailbreak Against Moderation), designed to attack moderation guardrails using jailbreak prefixes to bypass input-level filters and a fine-tuned shadow model functionally equivalent to the guardrail model to generate cipher characters to bypass output-level filters. Our extensive experiments on four LLMs demonstrate that JAM achieves higher jailbreak success ($\sim$ $\times$ 19.88) and lower filtered-out rates ($\sim$ $\times$ 1/6) than baselines.

摘要: 大型语言模型(LLM)通常是无害的，但仍然容易受到精心设计的称为“越狱”的提示的攻击，这些提示可能会绕过保护措施并引发有害行为。LLMS中最近的改进包括了可以过滤输出的适度防护，这会触发对某些恶意问题的处理错误。现有的红团队基准往往忽视了包括引发温和障碍的问题，这使得评估越狱的有效性变得困难。为了解决这个问题，我们引入了JAMBch，这是一个旨在触发和评估适度护栏的有害行为基准。JAMBtch涉及160个手动编写的说明，涵盖多个严重级别的四个主要风险类别。此外，我们提出了一种越狱方法JAM(JailBreak Against Medium Ation)，旨在使用越狱前缀来攻击适度护栏，以绕过输入级过滤器，以及一个功能等价于护栏模型的微调阴影模型，以生成密码字符以绕过输出级过滤器。我们在四个LLM上的广泛实验表明，JAM获得了比基线更高的越狱成功率($\sim$$\x$19.88)和更低的过滤成功率($\sim$$\x$1/6)。



## **44. Context Injection Attacks on Large Language Models**

对大型语言模型的上下文注入攻击 cs.AI

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20234v1) [paper-pdf](http://arxiv.org/pdf/2405.20234v1)

**Authors**: Cheng'an Wei, Kai Chen, Yue Zhao, Yujia Gong, Lu Xiang, Shenchen Zhu

**Abstract**: Large Language Models (LLMs) such as ChatGPT and Llama-2 have become prevalent in real-world applications, exhibiting impressive text generation performance. LLMs are fundamentally developed from a scenario where the input data remains static and lacks a clear structure. To behave interactively over time, LLM-based chat systems must integrate additional contextual information (i.e., chat history) into their inputs, following a pre-defined structure. This paper identifies how such integration can expose LLMs to misleading context from untrusted sources and fail to differentiate between system and user inputs, allowing users to inject context. We present a systematic methodology for conducting context injection attacks aimed at eliciting disallowed responses by introducing fabricated context. This could lead to illegal actions, inappropriate content, or technology misuse. Our context fabrication strategies, acceptance elicitation and word anonymization, effectively create misleading contexts that can be structured with attacker-customized prompt templates, achieving injection through malicious user messages. Comprehensive evaluations on real-world LLMs such as ChatGPT and Llama-2 confirm the efficacy of the proposed attack with success rates reaching 97%. We also discuss potential countermeasures that can be adopted for attack detection and developing more secure models. Our findings provide insights into the challenges associated with the real-world deployment of LLMs for interactive and structured data scenarios.

摘要: 大型语言模型(LLM)，如ChatGPT和Llama-2，已经在现实世界的应用程序中流行起来，表现出令人印象深刻的文本生成性能。LLM基本上是在输入数据保持静态且缺乏清晰结构的情况下开发出来的。为了随着时间的推移交互行为，基于LLM的聊天系统必须按照预定义的结构将附加的上下文信息(即聊天历史)集成到它们的输入中。这篇白皮书指出了这种集成如何将LLM暴露在来自不可信来源的误导性上下文中，并无法区分系统和用户输入，从而允许用户注入上下文。我们提出了一种系统的方法来进行上下文注入攻击，目的是通过引入捏造的上下文来引发不允许的响应。这可能会导致非法行为、不适当的内容或技术滥用。我们的上下文构建策略，接受诱导和单词匿名化，有效地创建了误导性上下文，可以使用攻击者定制的提示模板来构建，通过恶意用户消息实现注入。在ChatGPT和Llama-2等真实LLMS上的综合评估证实了该攻击的有效性，成功率达到97%。我们还讨论了可用于攻击检测和开发更安全模型的潜在对策。我们的发现为交互式和结构化数据场景中与LLMS的实际部署相关的挑战提供了见解。



## **45. Defensive Prompt Patch: A Robust and Interpretable Defense of LLMs against Jailbreak Attacks**

防御提示补丁：LLM针对越狱攻击的强大且可解释的防御 cs.CR

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20099v1) [paper-pdf](http://arxiv.org/pdf/2405.20099v1)

**Authors**: Chen Xiong, Xiangyu Qi, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Safety, security, and compliance are essential requirements when aligning large language models (LLMs). However, many seemingly aligned LLMs are soon shown to be susceptible to jailbreak attacks. These attacks aim to circumvent the models' safety guardrails and security mechanisms by introducing jailbreak prompts into malicious queries. In response to these challenges, this paper introduces Defensive Prompt Patch (DPP), a novel prompt-based defense mechanism specifically designed to protect LLMs against such sophisticated jailbreak strategies. Unlike previous approaches, which have often compromised the utility of the model for the sake of safety, DPP is designed to achieve a minimal Attack Success Rate (ASR) while preserving the high utility of LLMs. Our method uses strategically designed interpretable suffix prompts that effectively thwart a wide range of standard and adaptive jailbreak techniques. Empirical results conducted on LLAMA-2-7B-Chat and Mistral-7B-Instruct-v0.2 models demonstrate the robustness and adaptability of DPP, showing significant reductions in ASR with negligible impact on utility. Our approach not only outperforms existing defense strategies in balancing safety and functionality, but also provides a scalable and interpretable solution applicable to various LLM platforms.

摘要: 安全性、安全性和合规性是调整大型语言模型(LLM)时的基本要求。然而，许多看似一致的低收入国家很快就被证明容易受到越狱攻击。这些攻击旨在通过在恶意查询中引入越狱提示来绕过模型的安全护栏和安全机制。为了应对这些挑战，本文引入了防御提示补丁(DPP)，这是一种新的基于提示的防御机制，专门设计来保护LLM免受这种复杂的越狱策略的攻击。与以前的方法不同，为了安全起见，DPP通常会损害模型的实用性，DPP旨在实现最小的攻击成功率(ASR)，同时保持LLMS的高可用性。我们的方法使用策略性设计的可解释后缀提示，有效地阻止了广泛的标准和自适应越狱技术。在Llama-2-7B-Chat和Mistral-7B-Indict-v0.2模型上进行的实证结果证明了DPP的稳健性和适应性，表明ASR显著降低，而对效用的影响可以忽略不计。我们的方法不仅在平衡安全性和功能性方面优于现有的防御策略，而且还提供了适用于各种LLM平台的可扩展和可解释的解决方案。



## **46. Typography Leads Semantic Diversifying: Amplifying Adversarial Transferability across Multimodal Large Language Models**

字体设计引领语义多元化：增强多模式大型语言模型之间的对抗性可移植性 cs.CV

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20090v1) [paper-pdf](http://arxiv.org/pdf/2405.20090v1)

**Authors**: Hao Cheng, Erjia Xiao, Jiahang Cao, Le Yang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Following the advent of the Artificial Intelligence (AI) era of large models, Multimodal Large Language Models (MLLMs) with the ability to understand cross-modal interactions between vision and text have attracted wide attention. Adversarial examples with human-imperceptible perturbation are shown to possess a characteristic known as transferability, which means that a perturbation generated by one model could also mislead another different model. Augmenting the diversity in input data is one of the most significant methods for enhancing adversarial transferability. This method has been certified as a way to significantly enlarge the threat impact under black-box conditions. Research works also demonstrate that MLLMs can be exploited to generate adversarial examples in the white-box scenario. However, the adversarial transferability of such perturbations is quite limited, failing to achieve effective black-box attacks across different models. In this paper, we propose the Typographic-based Semantic Transfer Attack (TSTA), which is inspired by: (1) MLLMs tend to process semantic-level information; (2) Typographic Attack could effectively distract the visual information captured by MLLMs. In the scenarios of Harmful Word Insertion and Important Information Protection, our TSTA demonstrates superior performance.

摘要: 随着大模型人工智能时代的到来，能够理解视觉和文本之间跨通道交互的多通道大语言模型引起了人们的广泛关注。具有人类不可察觉的扰动的对抗性例子具有被称为可转移性的特征，这意味着一个模型产生的扰动也可能误导另一个不同的模型。增加输入数据的多样性是增强对抗性转移的最重要的方法之一。这种方法已被证明是一种在黑箱条件下显著扩大威胁影响的方法。研究工作还表明，在白盒情况下，MLLMS可以被用来生成对抗性示例。然而，此类扰动的对抗性可转移性相当有限，无法实现跨不同模型的有效黑盒攻击。本文提出了基于排版的语义传输攻击(TSTA)，其灵感来自：(1)MLLMS倾向于处理语义级的信息；(2)排版攻击可以有效地分散MLLMS捕获的视觉信息。在有害词语插入和重要信息保护的场景中，我们的TSTA表现出了卓越的性能。



## **47. Efficient LLM-Jailbreaking by Introducing Visual Modality**

通过引入视觉形态高效法学硕士越狱 cs.AI

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20015v1) [paper-pdf](http://arxiv.org/pdf/2405.20015v1)

**Authors**: Zhenxing Niu, Yuyao Sun, Haodong Ren, Haoxuan Ji, Quan Wang, Xiaoke Ma, Gang Hua, Rong Jin

**Abstract**: This paper focuses on jailbreaking attacks against large language models (LLMs), eliciting them to generate objectionable content in response to harmful user queries. Unlike previous LLM-jailbreaks that directly orient to LLMs, our approach begins by constructing a multimodal large language model (MLLM) through the incorporation of a visual module into the target LLM. Subsequently, we conduct an efficient MLLM-jailbreak to generate jailbreaking embeddings embJS. Finally, we convert the embJS into text space to facilitate the jailbreaking of the target LLM. Compared to direct LLM-jailbreaking, our approach is more efficient, as MLLMs are more vulnerable to jailbreaking than pure LLM. Additionally, to improve the attack success rate (ASR) of jailbreaking, we propose an image-text semantic matching scheme to identify a suitable initial input. Extensive experiments demonstrate that our approach surpasses current state-of-the-art methods in terms of both efficiency and effectiveness. Moreover, our approach exhibits superior cross-class jailbreaking capabilities.

摘要: 本文的重点是针对大型语言模型(LLM)的越狱攻击，诱使它们生成令人反感的内容来响应有害的用户查询。与以前直接面向LLMS的LLM越狱不同，我们的方法首先通过将可视模块整合到目标LLM中来构建多通道大型语言模型(MLLM)。随后，我们进行了一个高效的MLLM-JailBreak来生成越狱嵌入embJS。最后，我们将embJS转换为文本空间，以便于目标LLM的越狱。与直接LLM越狱相比，我们的方法更有效，因为MLLM比纯粹的LLM更容易越狱。此外，为了提高越狱的攻击成功率，我们提出了一种图文语义匹配方案来识别合适的初始输入。大量的实验表明，我们的方法在效率和效果上都超过了目前最先进的方法。此外，我们的方法显示出卓越的跨阶层越狱能力。



## **48. Vocabulary Attack to Hijack Large Language Model Applications**

劫持大型语言模型应用程序的词汇攻击 cs.CR

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2404.02637v2) [paper-pdf](http://arxiv.org/pdf/2404.02637v2)

**Authors**: Patrick Levi, Christoph P. Neumann

**Abstract**: The fast advancements in Large Language Models (LLMs) are driving an increasing number of applications. Together with the growing number of users, we also see an increasing number of attackers who try to outsmart these systems. They want the model to reveal confidential information, specific false information, or offensive behavior. To this end, they manipulate their instructions for the LLM by inserting separators or rephrasing them systematically until they reach their goal. Our approach is different. It inserts words from the model vocabulary. We find these words using an optimization procedure and embeddings from another LLM (attacker LLM). We prove our approach by goal hijacking two popular open-source LLMs from the Llama2 and the Flan-T5 families, respectively. We present two main findings. First, our approach creates inconspicuous instructions and therefore it is hard to detect. For many attack cases, we find that even a single word insertion is sufficient. Second, we demonstrate that we can conduct our attack using a different model than the target model to conduct our attack with.

摘要: 大型语言模型(LLM)的快速发展正在推动越来越多的应用程序。随着用户数量的不断增加，我们也看到越来越多的攻击者试图智取这些系统。他们希望该模型能够泄露机密信息、特定的虚假信息或冒犯行为。为此，他们通过插入分隔符或系统地重新措辞来操纵他们对LLM的指令，直到达到他们的目标。我们的方法是不同的。它插入模型词汇表中的单词。我们使用优化过程和来自另一个LLM(攻击者LLM)的嵌入来找到这些单词。我们通过Goal劫持了分别来自Llama2和Flan-T5家族的两个流行的开源LLM来证明我们的方法。我们提出了两个主要发现。首先，我们的方法创建了不明显的指令，因此很难检测到。对于许多攻击情况，我们发现即使是一个单词插入也是足够的。其次，我们演示了我们可以使用与进行攻击的目标模型不同的模型来进行攻击。



## **49. Large Language Model Watermark Stealing With Mixed Integer Programming**

使用混合格式编程实现大语言模型水印窃取 cs.CR

12 pages

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19677v1) [paper-pdf](http://arxiv.org/pdf/2405.19677v1)

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, Leo Yu Zhang, Chao Chen, Shengshan Hu, Asif Gill, Shirui Pan

**Abstract**: The Large Language Model (LLM) watermark is a newly emerging technique that shows promise in addressing concerns surrounding LLM copyright, monitoring AI-generated text, and preventing its misuse. The LLM watermark scheme commonly includes generating secret keys to partition the vocabulary into green and red lists, applying a perturbation to the logits of tokens in the green list to increase their sampling likelihood, thus facilitating watermark detection to identify AI-generated text if the proportion of green tokens exceeds a threshold. However, recent research indicates that watermarking methods using numerous keys are susceptible to removal attacks, such as token editing, synonym substitution, and paraphrasing, with robustness declining as the number of keys increases. Therefore, the state-of-the-art watermark schemes that employ fewer or single keys have been demonstrated to be more robust against text editing and paraphrasing. In this paper, we propose a novel green list stealing attack against the state-of-the-art LLM watermark scheme and systematically examine its vulnerability to this attack. We formalize the attack as a mixed integer programming problem with constraints. We evaluate our attack under a comprehensive threat model, including an extreme scenario where the attacker has no prior knowledge, lacks access to the watermark detector API, and possesses no information about the LLM's parameter settings or watermark injection/detection scheme. Extensive experiments on LLMs, such as OPT and LLaMA, demonstrate that our attack can successfully steal the green list and remove the watermark across all settings.

摘要: 大语言模型(LLM)水印是一种新出现的技术，它在解决围绕LLM版权的担忧、监控人工智能生成的文本并防止其滥用方面显示出良好的前景。LLM水印方案通常包括生成密钥以将词汇表划分为绿色和红色列表，对绿色列表中的令牌的逻辑施加扰动以增加其采样可能性，从而在绿色令牌的比例超过阈值的情况下促进水印检测以识别AI生成的文本。然而，最近的研究表明，使用大量密钥的水印方法容易受到移除攻击，例如标记编辑、同义词替换和改述，并且随着密钥数量的增加，鲁棒性下降。因此，已证明采用较少密钥或单一密钥的最新水印方案对文本编辑和转译更健壮。本文针对现有的LLM水印方案，提出了一种新的绿名单窃取攻击方案，并对其脆弱性进行了系统的分析。我们将攻击形式化化为一个带约束的混合整数规划问题。我们在一个全面的威胁模型下评估我们的攻击，包括一个极端的场景，其中攻击者事先不知道，无法访问水印检测器API，并且没有关于LLM的参数设置或水印注入/检测方案的信息。在OPT和Llama等LLMS上的大量实验表明，我们的攻击可以成功地窃取绿色列表并删除所有设置的水印。



## **50. AutoBreach: Universal and Adaptive Jailbreaking with Efficient Wordplay-Guided Optimization**

AutoBreach：具有高效的文字游戏引导优化的通用和自适应越狱 cs.CV

Under review

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19668v1) [paper-pdf](http://arxiv.org/pdf/2405.19668v1)

**Authors**: Jiawei Chen, Xiao Yang, Zhengwei Fang, Yu Tian, Yinpeng Dong, Zhaoxia Yin, Hang Su

**Abstract**: Despite the widespread application of large language models (LLMs) across various tasks, recent studies indicate that they are susceptible to jailbreak attacks, which can render their defense mechanisms ineffective. However, previous jailbreak research has frequently been constrained by limited universality, suboptimal efficiency, and a reliance on manual crafting. In response, we rethink the approach to jailbreaking LLMs and formally define three essential properties from the attacker' s perspective, which contributes to guiding the design of jailbreak methods. We further introduce AutoBreach, a novel method for jailbreaking LLMs that requires only black-box access. Inspired by the versatility of wordplay, AutoBreach employs a wordplay-guided mapping rule sampling strategy to generate a variety of universal mapping rules for creating adversarial prompts. This generation process leverages LLMs' automatic summarization and reasoning capabilities, thus alleviating the manual burden. To boost jailbreak success rates, we further suggest sentence compression and chain-of-thought-based mapping rules to correct errors and wordplay misinterpretations in target LLMs. Additionally, we propose a two-stage mapping rule optimization strategy that initially optimizes mapping rules before querying target LLMs to enhance the efficiency of AutoBreach. AutoBreach can efficiently identify security vulnerabilities across various LLMs, including three proprietary models: Claude-3, GPT-3.5, GPT-4 Turbo, and two LLMs' web platforms: Bingchat, GPT-4 Web, achieving an average success rate of over 80% with fewer than 10 queries

摘要: 尽管大型语言模型在各种任务中得到了广泛应用，但最近的研究表明，它们很容易受到越狱攻击，这会使它们的防御机制失效。然而，以前的越狱研究经常受到普适性有限、效率不佳以及对手工制作的依赖的限制。作为回应，我们重新思考了越狱LLM的方法，并从攻击者S的角度正式定义了三个基本性质，这有助于指导越狱方法的设计。我们进一步介绍了AutoBReach，这是一种新的越狱LLMS方法，只需要黑盒访问。受文字游戏的多样性启发，AutoBReach采用了文字游戏指导的映射规则采样策略，生成了各种通用的映射规则，用于创建对抗性提示。这一生成过程利用了LLMS的自动摘要和推理能力，从而减轻了手动负担。为了提高越狱成功率，我们进一步建议句子压缩和基于思想链的映射规则来纠正目标LLM中的错误和文字游戏误解。此外，我们还提出了一种两阶段映射规则优化策略，在查询目标LLM之前对映射规则进行初始优化，以提高AutoBReach的效率。AutoBReach可以高效地识别各种LLMS的安全漏洞，包括三种专有模型：Claude-3、GPT-3.5、GPT-4 Turbo，以及两种LLMS的Web平台：Bingchat、GPT-4 Web，平均成功率超过80%，查询次数不到10次



