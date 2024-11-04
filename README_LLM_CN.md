# Latest Large Language Model Attack Papers
**update at 2024-11-04 11:07:58**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Efficient Adversarial Training in LLMs with Continuous Attacks**

在具有持续攻击的LLC中进行有效的对抗训练 cs.LG

19 pages, 4 figures

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.15589v3) [paper-pdf](http://arxiv.org/pdf/2405.15589v3)

**Authors**: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on five models from different families (Gemma, Phi3, Mistral, Zephyr, Llama2) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.

摘要: 大型语言模型(LLM)容易受到敌意攻击，这些攻击可以绕过它们的安全护栏。在许多领域，对抗性训练已被证明是可靠地提高对此类攻击的稳健性的最有前途的方法之一。然而，在LLMS的背景下，当前的对抗性训练方法受到在每次训练迭代中执行离散对抗性攻击所需的高计算成本的阻碍。我们通过在LLM的连续嵌入空间中计算敌意攻击来解决这个问题，该空间的效率要高出几个数量级。我们提出了一种快速对抗性训练算法(C-AdvUL)，该算法由两个损失组成：第一个损失使模型对基于对抗行为数据集的连续嵌入攻击具有健壮性；第二个损失通过对效用数据的微调来确保最终模型的有效性。此外，我们引入了C-AdvIPO，这是IPO的一个对抗性变体，不需要效用数据来进行对抗性强健的比对。我们对来自不同家族(Gema，Phi3，Mistral，Zephy，Llama2)和不同尺度(2B，3.8B，7B)的五个模型的经验评估表明，这两种算法在保持实用性的同时，显著增强了LLM对离散攻击(GCG，AutoDAN，Pair)的稳健性。我们的结果表明，对连续扰动的稳健性可以外推到离散威胁模型。因此，我们提出了一条可扩展的对抗性训练算法，用于鲁棒对齐LLM。



## **2. Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level**

文字入侵：在文本层面了解图注入攻击 cs.LG

Accepted by NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.16405v2) [paper-pdf](http://arxiv.org/pdf/2405.16405v2)

**Authors**: Runlin Lei, Yuwei Hu, Yuchen Ren, Zhewei Wei

**Abstract**: Graph Neural Networks (GNNs) excel across various applications but remain vulnerable to adversarial attacks, particularly Graph Injection Attacks (GIAs), which inject malicious nodes into the original graph and pose realistic threats. Text-attributed graphs (TAGs), where nodes are associated with textual features, are crucial due to their prevalence in real-world applications and are commonly used to evaluate these vulnerabilities. However, existing research only focuses on embedding-level GIAs, which inject node embeddings rather than actual textual content, limiting their applicability and simplifying detection. In this paper, we pioneer the exploration of GIAs at the text level, presenting three novel attack designs that inject textual content into the graph. Through theoretical and empirical analysis, we demonstrate that text interpretability, a factor previously overlooked at the embedding level, plays a crucial role in attack strength. Among the designs we investigate, the Word-frequency-based Text-level GIA (WTGIA) is particularly notable for its balance between performance and interpretability. Despite the success of WTGIA, we discover that defenders can easily enhance their defenses with customized text embedding methods or large language model (LLM)--based predictors. These insights underscore the necessity for further research into the potential and practical significance of text-level GIAs.

摘要: 图神经网络(GNN)在各种应用中表现出色，但仍然容易受到对手攻击，特别是图注入攻击(GIA)，图注入攻击将恶意节点注入到原始图中，并构成现实威胁。文本属性图(TAG)将节点与文本特征相关联，由于它们在现实应用程序中的普遍存在，因此至关重要，并且通常用于评估这些漏洞。然而，现有的研究只关注嵌入级GIA，这些GIA注入的是节点嵌入而不是实际的文本内容，限制了它们的适用性，简化了检测。在本文中，我们率先在文本层面上探索了GIA，提出了三种向图形中注入文本内容的新颖攻击设计。通过理论和实证分析，我们证明了文本可解释性对攻击强度起着至关重要的作用，而文本可解释性是此前在嵌入层面被忽视的一个因素。在我们研究的设计中，基于词频的文本级别GIA(WTGIA)特别值得注意的是它在性能和可解释性之间的平衡。尽管WTGIA取得了成功，但我们发现，防御者可以很容易地通过定制的文本嵌入方法或基于大型语言模型(LLM)的预测器来增强他们的防御。这些见解突显了进一步研究文本层面全球影响的潜力和现实意义的必要性。



## **3. Defense Against Prompt Injection Attack by Leveraging Attack Techniques**

利用攻击技术防御即时注入攻击 cs.CR

9 pages

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00459v1) [paper-pdf](http://arxiv.org/pdf/2411.00459v1)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song, Dekai Wu, Bryan Hooi

**Abstract**: With the advancement of technology, large language models (LLMs) have achieved remarkable performance across various natural language processing (NLP) tasks, powering LLM-integrated applications like Microsoft Copilot. However, as LLMs continue to evolve, new vulnerabilities, especially prompt injection attacks arise. These attacks trick LLMs into deviating from the original input instructions and executing the attacker's instructions injected in data content, such as retrieved results. Recent attack methods leverage LLMs' instruction-following abilities and their inabilities to distinguish instructions injected in the data content, and achieve a high attack success rate (ASR). When comparing the attack and defense methods, we interestingly find that they share similar design goals, of inducing the model to ignore unwanted instructions and instead to execute wanted instructions. Therefore, we raise an intuitive question: Could these attack techniques be utilized for defensive purposes? In this paper, we invert the intention of prompt injection methods to develop novel defense methods based on previous training-free attack methods, by repeating the attack process but with the original input instruction rather than the injected instruction. Our comprehensive experiments demonstrate that our defense techniques outperform existing training-free defense approaches, achieving state-of-the-art results.

摘要: 随着技术的进步，大语言模型(LLM)在各种自然语言处理(NLP)任务中取得了显著的性能，支持Microsoft Copilot等LLM集成应用程序。然而，随着LLMS的不断发展，出现了新的漏洞，特别是即时注入攻击。这些攻击欺骗LLM偏离原始输入指令，并执行注入数据内容的攻击者指令，例如检索的结果。最近的攻击方法利用LLMS的指令跟随能力和它们无法区分注入到数据内容中的指令的能力，实现了高攻击成功率(ASR)。当比较攻击和防御方法时，我们有趣地发现它们有相似的设计目标，都是诱导模型忽略不想要的指令，而是执行想要的指令。因此，我们提出了一个直观的问题：这些攻击技术是否可以用于防御目的？在本文中，我们反转了快速注入方法的意图，在以前的免训练攻击方法的基础上，通过重复攻击过程来开发新的防御方法，但使用的是原始输入指令而不是注入指令。我们的综合实验表明，我们的防御技术优于现有的免训练防御方法，取得了最先进的结果。



## **4. Attention Tracker: Detecting Prompt Injection Attacks in LLMs**

注意力追踪器：检测LLM中的即时注入攻击 cs.CR

Project page:  https://huggingface.co/spaces/TrustSafeAI/Attention-Tracker

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00348v1) [paper-pdf](http://arxiv.org/pdf/2411.00348v1)

**Authors**: Kuo-Han Hung, Ching-Yun Ko, Ambrish Rawat, I-Hsin Chung, Winston H. Hsu, Pin-Yu Chen

**Abstract**: Large Language Models (LLMs) have revolutionized various domains but remain vulnerable to prompt injection attacks, where malicious inputs manipulate the model into ignoring original instructions and executing designated action. In this paper, we investigate the underlying mechanisms of these attacks by analyzing the attention patterns within LLMs. We introduce the concept of the distraction effect, where specific attention heads, termed important heads, shift focus from the original instruction to the injected instruction. Building on this discovery, we propose Attention Tracker, a training-free detection method that tracks attention patterns on instruction to detect prompt injection attacks without the need for additional LLM inference. Our method generalizes effectively across diverse models, datasets, and attack types, showing an AUROC improvement of up to 10.0% over existing methods, and performs well even on small LLMs. We demonstrate the robustness of our approach through extensive evaluations and provide insights into safeguarding LLM-integrated systems from prompt injection vulnerabilities.

摘要: 大型语言模型(LLM)给各个领域带来了革命性的变化，但仍然容易受到即时注入攻击，恶意输入会操纵模型忽略原始指令并执行指定的操作。在本文中，我们通过分析LLMS中的注意模式来研究这些攻击的潜在机制。我们引入了分心效应的概念，即特定的注意力头部，称为重要头部，将焦点从原始指令转移到注入的指令。基于这一发现，我们提出了注意力跟踪器，这是一种无需训练的检测方法，它跟踪指令上的注意模式，检测即时注入攻击，而不需要额外的LLM推理。我们的方法有效地概括了不同的模型、数据集和攻击类型，显示出比现有方法高达10.0%的AUROC改进，即使在小的LLM上也表现得很好。我们通过广泛的评估展示了我们方法的健壮性，并为保护LLM集成系统免受即时注入漏洞的攻击提供了见解。



## **5. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

越狱长凳：越狱大型语言模型的开放鲁棒性基准 cs.CR

The camera-ready version of JailbreakBench v1.0 (accepted at NeurIPS  2024 Datasets and Benchmarks Track): more attack artifacts, more test-time  defenses, a more accurate jailbreak judge (Llama-3-70B with a custom prompt),  a larger dataset of human preferences for selecting a jailbreak judge (300  examples), an over-refusal evaluation dataset, a semantic refusal judge based  on Llama-3-8B

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2404.01318v5) [paper-pdf](http://arxiv.org/pdf/2404.01318v5)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work (Zou et al., 2023; Mazeika et al., 2023, 2024) -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak，这是一个开源基准，具有以下组件：(1)一个不断发展的最先进对手提示的储存库，我们称之为越狱人工制品；(2)一个包括100种行为的越狱数据集--既有原始的，也有源自先前工作的(邹某等人，2023年；Mazeika等人，2023年，2024年)--与开放人工智能的使用政策保持一致；(3)https://github.com/JailbreakBench/jailbreakbench的标准化评估框架，包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)https://jailbreakbench.github.io/的排行榜，跟踪各种LLM的攻击和防御性能。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。



## **6. Scaling Up Membership Inference: When and How Attacks Succeed on Large Language Models**

扩大成员资格推理：攻击何时以及如何在大型语言模型上取得成功 cs.CL

Our code is available at https://github.com/parameterlab/mia-scaling

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2411.00154v1) [paper-pdf](http://arxiv.org/pdf/2411.00154v1)

**Authors**: Haritz Puerto, Martin Gubri, Sangdoo Yun, Seong Joon Oh

**Abstract**: Membership inference attacks (MIA) attempt to verify the membership of a given data sample in the training set for a model. MIA has become relevant in recent years, following the rapid development of large language models (LLM). Many are concerned about the usage of copyrighted materials for training them and call for methods for detecting such usage. However, recent research has largely concluded that current MIA methods do not work on LLMs. Even when they seem to work, it is usually because of the ill-designed experimental setup where other shortcut features enable "cheating." In this work, we argue that MIA still works on LLMs, but only when multiple documents are presented for testing. We construct new benchmarks that measure the MIA performances at a continuous scale of data samples, from sentences (n-grams) to a collection of documents (multiple chunks of tokens). To validate the efficacy of current MIA approaches at greater scales, we adapt a recent work on Dataset Inference (DI) for the task of binary membership detection that aggregates paragraph-level MIA features to enable MIA at document and collection of documents level. This baseline achieves the first successful MIA on pre-trained and fine-tuned LLMs.

摘要: 成员关系推理攻击(MIA)试图验证给定数据样本在模型训练集中的成员资格。近年来，随着大型语言模型(LLM)的快速发展，MIA变得相关起来。许多人担心使用受版权保护的材料来培训他们，并呼吁采取方法来检测这种使用情况。然而，最近的研究在很大程度上得出结论，目前的MIA方法不适用于LLMS。即使它们看起来很有效，这通常也是因为设计糟糕的实验设置，其他快捷功能允许“作弊”。在这项工作中，我们认为MIA仍然适用于LLMS，但只有在提交多个文档进行测试时才能使用。我们构建了新的基准来衡量MIA在连续规模的数据样本上的性能，从句子(n-gram)到文档集合(多个令牌块)。为了在更大范围内验证当前MIA方法的有效性，我们对最近在数据集推理(DI)方面的工作进行了调整，以用于二元成员关系检测任务，该任务聚集了段级MIA特征，以支持文档和文档集合级别的MIA。这一基线在预先训练和微调的LLM上实现了第一次成功的MIA。



## **7. Tree of Attacks: Jailbreaking Black-Box LLMs Automatically**

攻击树：自动越狱黑匣子LLM cs.LG

Accepted for presentation at NeurIPS 2024. Code:  https://github.com/RICommunity/TAP

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2312.02119v3) [paper-pdf](http://arxiv.org/pdf/2312.02119v3)

**Authors**: Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, Amin Karbasi

**Abstract**: While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an attacker LLM to iteratively refine candidate (attack) prompts until one of the refined prompts jailbreaks the target. In addition, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks, reducing the number of queries sent to the target LLM. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4-Turbo and GPT4o) for more than 80% of the prompts. This significantly improves upon the previous state-of-the-art black-box methods for generating jailbreaks while using a smaller number of queries than them. Furthermore, TAP is also capable of jailbreaking LLMs protected by state-of-the-art guardrails, e.g., LlamaGuard.

摘要: 虽然大型语言模型(LLM)显示了多功能，但它们继续产生有害、有偏见和有毒的内容，人类设计的越狱事件的流行就证明了这一点。在这项工作中，我们提出了带修剪的攻击树(TAP)，这是一种自动生成越狱的方法，只需要通过黑盒访问目标LLM。TAP利用攻击者的LLM反复细化候选(攻击)提示，直到其中一个细化的提示破解目标。此外，在向目标发送提示之前，TAP会对它们进行评估，并删除不太可能导致越狱的提示，从而减少发送到目标LLM的查询数量。在实证评估中，我们观察到TAP为80%以上的提示生成了越狱最先进的LLM(包括GPT4-Turbo和GPT4o)提示。与以前最先进的黑盒方法相比，这大大改进了生成越狱的方法，同时使用的查询数量比它们少。此外，TAP还能够越狱由最先进的护栏保护的LLMS，例如LlamaGuard。



## **8. Desert Camels and Oil Sheikhs: Arab-Centric Red Teaming of Frontier LLMs**

沙漠骆驼和石油酋长：以阿拉伯为中心的红色Frontier LLM团队 cs.CL

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.24049v1) [paper-pdf](http://arxiv.org/pdf/2410.24049v1)

**Authors**: Muhammed Saeed, Elgizouli Mohamed, Mukhtar Mohamed, Shaina Raza, Shady Shehata, Muhammad Abdul-Mageed

**Abstract**: Large language models (LLMs) are widely used but raise ethical concerns due to embedded social biases. This study examines LLM biases against Arabs versus Westerners across eight domains, including women's rights, terrorism, and anti-Semitism and assesses model resistance to perpetuating these biases. To this end, we create two datasets: one to evaluate LLM bias toward Arabs versus Westerners and another to test model safety against prompts that exaggerate negative traits ("jailbreaks"). We evaluate six LLMs -- GPT-4, GPT-4o, LlaMA 3.1 (8B & 405B), Mistral 7B, and Claude 3.5 Sonnet. We find 79% of cases displaying negative biases toward Arabs, with LlaMA 3.1-405B being the most biased. Our jailbreak tests reveal GPT-4o as the most vulnerable, despite being an optimized version, followed by LlaMA 3.1-8B and Mistral 7B. All LLMs except Claude exhibit attack success rates above 87% in three categories. We also find Claude 3.5 Sonnet the safest, but it still displays biases in seven of eight categories. Despite being an optimized version of GPT4, We find GPT-4o to be more prone to biases and jailbreaks, suggesting optimization flaws. Our findings underscore the pressing need for more robust bias mitigation strategies and strengthened security measures in LLMs.

摘要: 大型语言模型(LLM)被广泛使用，但由于根深蒂固的社会偏见而引发了伦理问题。这项研究考察了LLM在八个领域对阿拉伯人和西方人的偏见，包括妇女权利、恐怖主义和反犹太主义，并评估了对延续这些偏见的模型阻力。为此，我们创建了两个数据集：一个用于评估LLM对阿拉伯人和西方人的偏见，另一个用于测试针对夸大负面特征的提示(“越狱”)的模型安全性。我们评估了六个LLMS--GPT-4、GPT-40、大羊驼3.1(8B和405B)、西北风7B和克劳德3.5十四行诗。我们发现79%的病例对阿拉伯人表现出负面偏见，其中大羊驼3.1-405B是最有偏见的。我们的越狱测试显示，GPT-40是最脆弱的，尽管是一个优化版本，紧随其后的是骆驼3.1-8B和米斯特拉尔7B。除克劳德外，所有LLM在三个类别中的攻击成功率都在87%以上。我们也发现克劳德3.5十四行诗是最安全的，但它仍然在八个类别中的七个方面表现出偏见。尽管GPT-4是GPT4的优化版本，但我们发现GPT-40更容易产生偏见和越狱，这表明优化存在缺陷。我们的研究结果突出表明，迫切需要更有力的减轻偏见战略和加强小岛屿发展中国家的安全措施。



## **9. Fight Back Against Jailbreaking via Prompt Adversarial Tuning**

通过即时对抗调整反击越狱 cs.LG

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2402.06255v4) [paper-pdf](http://arxiv.org/pdf/2402.06255v4)

**Authors**: Yichuan Mo, Yuji Wang, Zeming Wei, Yisen Wang

**Abstract**: While Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to jailbreaking attacks. Several primary defense strategies have been proposed to protect LLMs from producing harmful information, mostly focusing on model fine-tuning or heuristical defense designs. However, how to achieve intrinsic robustness through prompt optimization remains an open problem. In this paper, motivated by adversarial training paradigms for achieving reliable robustness, we propose an approach named Prompt Adversarial Tuning (PAT) that trains a prompt control attached to the user prompt as a guard prefix. To achieve our defense goal whilst maintaining natural performance, we optimize the control prompt with both adversarial and benign prompts. Comprehensive experiments show that our method is effective against both grey-box and black-box attacks, reducing the success rate of advanced attacks to nearly 0%, while maintaining the model's utility on the benign task and incurring only negligible computational overhead, charting a new perspective for future explorations in LLM security. Our code is available at https://github.com/PKU-ML/PAT.

摘要: 虽然大型语言模型(LLM)在各种应用中取得了巨大的成功，但它们也容易受到越狱攻击。已经提出了几种主要的防御策略来保护LLMS免受有害信息的影响，主要集中在模型微调或启发式防御设计上。然而，如何通过快速优化来获得内在的稳健性仍然是一个悬而未决的问题。受实现可靠健壮性的对抗性训练范例的启发，我们提出了一种称为即时对抗性调整(PAT)的方法，该方法将附加在用户提示上的提示控制训练为保卫前缀。为了在保持自然表现的同时实现我们的防守目标，我们优化了控制提示，包括对抗性提示和良性提示。综合实验表明，该方法对灰盒和黑盒攻击都是有效的，将高级攻击的成功率降低到近0%，同时保持了模型在良性任务上的效用，并且只产生了可以忽略的计算开销，为LLM安全的进一步研究开辟了新的视角。我们的代码可以在https://github.com/PKU-ML/PAT.上找到



## **10. Audio Is the Achilles' Heel: Red Teaming Audio Large Multimodal Models**

音频是阿喀琉斯之踵：红色团队音频大型多模式 cs.CL

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23861v1) [paper-pdf](http://arxiv.org/pdf/2410.23861v1)

**Authors**: Hao Yang, Lizhen Qu, Ehsan Shareghi, Gholamreza Haffari

**Abstract**: Large Multimodal Models (LMMs) have demonstrated the ability to interact with humans under real-world conditions by combining Large Language Models (LLMs) and modality encoders to align multimodal information (visual and auditory) with text. However, such models raise new safety challenges of whether models that are safety-aligned on text also exhibit consistent safeguards for multimodal inputs. Despite recent safety-alignment research on vision LMMs, the safety of audio LMMs remains under-explored. In this work, we comprehensively red team the safety of five advanced audio LMMs under three settings: (i) harmful questions in both audio and text formats, (ii) harmful questions in text format accompanied by distracting non-speech audio, and (iii) speech-specific jailbreaks. Our results under these settings demonstrate that open-source audio LMMs suffer an average attack success rate of 69.14% on harmful audio questions, and exhibit safety vulnerabilities when distracted with non-speech audio noise. Our speech-specific jailbreaks on Gemini-1.5-Pro achieve an attack success rate of 70.67% on the harmful query benchmark. We provide insights on what could cause these reported safety-misalignments. Warning: this paper contains offensive examples.

摘要: 大型多通道模型(LMM)通过将大型语言模型(LLM)和通道编码器相结合来将多通道信息(视觉和听觉)与文本对齐，从而展示了在真实世界条件下与人类交互的能力。然而，这样的模型提出了新的安全挑战，即在文本上与安全一致的模型是否也显示出对多模式输入的一致保障。尽管最近对视觉LMM的安全性进行了研究，但音频LMM的安全性仍未得到充分的探索。在这项工作中，我们在三种设置下对五种高级音频LMM的安全性进行了全面的红色团队：(I)音频和文本格式的有害问题，(Ii)伴随着令人分心的非语音音频的文本格式的有害问题，以及(Iii)特定于语音的越狱。实验结果表明，在这种情况下，开源音频LMM对有害音频问题的平均攻击成功率为69.14%，在非语音噪声干扰下表现出安全漏洞。我们在Gemini-1.5-Pro上的语音特定越狱在有害查询基准上实现了70.67%的攻击成功率。我们提供了可能导致这些报告的安全错位的原因的见解。警告：本文包含令人反感的例子。



## **11. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

DetectRL：在现实世界场景中对LLM生成的文本检测进行基准测试 cs.CL

Accepted to NeurIPS 2024 Dataset & Benchmarking Track

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23746v1) [paper-pdf](http://arxiv.org/pdf/2410.23746v1)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating advanced prompt usages, human revisions like word substitutions, and writing errors. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.

摘要: 检测由大型语言模型(LLM)生成的文本是最近非常感兴趣的问题。有了像DetectGPT这样的零射击方法，检测能力已经达到了令人印象深刻的水平。然而，现有探测器在实际应用中的可靠性仍然没有得到充分的探索。在这项研究中，我们提出了一个新的基准，DetectRL，强调即使是最先进的(SOTA)检测技术在这项任务中仍然表现不佳。我们从LLM特别容易被滥用的领域收集了人类编写的数据集。使用流行的LLM，我们生成的数据更好地与现实世界的应用程序保持一致。与以前的研究不同，我们使用启发式规则来创建对抗性LLM生成的文本，模拟高级提示用法、人工修改(如单词替换)和书写错误。我们对DetectRL的开发揭示了当前SOTA探测器的优势和局限性。更重要的是，我们分析了写作风格、模型类型、攻击方法、文本长度和真实世界中的人类写作因素对不同类型检测器的潜在影响。我们相信，DetectRL可以作为评估真实世界场景中检测器的有效基准，随着先进攻击方法的发展，从而提供更有压力的评估，以推动更高效检测器的开发。数据和代码可在以下网址公开获得：https://github.com/NLP2CT/DetectRL.



## **12. Adversarial Attacks of Vision Tasks in the Past 10 Years: A Survey**

过去10年视觉任务的对抗性攻击：一项调查 cs.CV

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23687v1) [paper-pdf](http://arxiv.org/pdf/2410.23687v1)

**Authors**: Chiyu Zhang, Xiaogang Xu, Jiafei Wu, Zhe Liu, Lu Zhou

**Abstract**: Adversarial attacks, which manipulate input data to undermine model availability and integrity, pose significant security threats during machine learning inference. With the advent of Large Vision-Language Models (LVLMs), new attack vectors, such as cognitive bias, prompt injection, and jailbreak techniques, have emerged. Understanding these attacks is crucial for developing more robust systems and demystifying the inner workings of neural networks. However, existing reviews often focus on attack classifications and lack comprehensive, in-depth analysis. The research community currently needs: 1) unified insights into adversariality, transferability, and generalization; 2) detailed evaluations of existing methods; 3) motivation-driven attack categorizations; and 4) an integrated perspective on both traditional and LVLM attacks. This article addresses these gaps by offering a thorough summary of traditional and LVLM adversarial attacks, emphasizing their connections and distinctions, and providing actionable insights for future research.

摘要: 对抗性攻击通过操纵输入数据来破坏模型的可用性和完整性，在机器学习推理过程中会造成严重的安全威胁。随着大型视觉语言模型的出现，新的攻击载体出现了，如认知偏差、快速注入和越狱技术。了解这些攻击对于开发更强大的系统和揭开神经网络内部工作的神秘面纱至关重要。然而，现有的审查往往侧重于攻击分类，缺乏全面、深入的分析。研究界目前需要：1)对对抗性、可转移性和泛化的统一见解；2)对现有方法的详细评估；3)动机驱动的攻击分类；以及4)对传统攻击和LVLM攻击的综合视角。本文对传统攻击和LVLM攻击进行了全面的总结，强调了它们之间的联系和区别，并为未来的研究提供了可操作的见解，从而解决了这些差距。



## **13. Pseudo-Conversation Injection for LLM Goal Hijacking**

LLM目标劫持的伪对话注入 cs.CL

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23678v1) [paper-pdf](http://arxiv.org/pdf/2410.23678v1)

**Authors**: Zheng Chen, Buhui Yao

**Abstract**: Goal hijacking is a type of adversarial attack on Large Language Models (LLMs) where the objective is to manipulate the model into producing a specific, predetermined output, regardless of the user's original input. In goal hijacking, an attacker typically appends a carefully crafted malicious suffix to the user's prompt, which coerces the model into ignoring the user's original input and generating the target response. In this paper, we introduce a novel goal hijacking attack method called Pseudo-Conversation Injection, which leverages the weaknesses of LLMs in role identification within conversation contexts. Specifically, we construct the suffix by fabricating responses from the LLM to the user's initial prompt, followed by a prompt for a malicious new task. This leads the model to perceive the initial prompt and fabricated response as a completed conversation, thereby executing the new, falsified prompt. Following this approach, we propose three Pseudo-Conversation construction strategies: Targeted Pseudo-Conversation, Universal Pseudo-Conversation, and Robust Pseudo-Conversation. These strategies are designed to achieve effective goal hijacking across various scenarios. Our experiments, conducted on two mainstream LLM platforms including ChatGPT and Qwen, demonstrate that our proposed method significantly outperforms existing approaches in terms of attack effectiveness.

摘要: 目标劫持是一种针对大型语言模型(LLM)的对抗性攻击，其目标是操纵模型生成特定的、预定的输出，而不考虑用户的原始输入。在目标劫持中，攻击者通常会在用户提示后附加精心编制的恶意后缀，这会迫使模型忽略用户的原始输入并生成目标响应。本文提出了一种新的目标劫持攻击方法--伪会话注入，该方法利用了LLMS在会话上下文中角色识别方面的弱点。具体地说，我们构造后缀的方法是从LLM构造对用户初始提示的响应，然后是恶意新任务的提示。这导致模型将初始提示和捏造的响应视为完成的对话，从而执行新的、伪造的提示。在此基础上，我们提出了三种伪会话构建策略：目标伪会话、通用伪会话和健壮伪会话。这些策略旨在实现跨各种场景的有效目标劫持。我们在ChatGPT和Qwen两个主流LLM平台上进行的实验表明，我们提出的方法在攻击效率方面明显优于现有方法。



## **14. Adversarial Attacks on Code Models with Discriminative Graph Patterns**

对具有区分图模式的代码模型的对抗攻击 cs.SE

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2308.11161v2) [paper-pdf](http://arxiv.org/pdf/2308.11161v2)

**Authors**: Thanh-Dat Nguyen, Yang Zhou, Xuan Bach D. Le, Patanamon Thongtanunam, David Lo

**Abstract**: Pre-trained language models of code are now widely used in various software engineering tasks such as code generation, code completion, vulnerability detection, etc. This, in turn, poses security and reliability risks to these models. One of the important threats is \textit{adversarial attacks}, which can lead to erroneous predictions and largely affect model performance on downstream tasks. Current adversarial attacks on code models usually adopt fixed sets of program transformations, such as variable renaming and dead code insertion, leading to limited attack effectiveness. To address the aforementioned challenges, we propose a novel adversarial attack framework, GraphCodeAttack, to better evaluate the robustness of code models. Given a target code model, GraphCodeAttack automatically mines important code patterns, which can influence the model's decisions, to perturb the structure of input code to the model. To do so, GraphCodeAttack uses a set of input source codes to probe the model's outputs and identifies the \textit{discriminative} ASTs patterns that can influence the model decisions. GraphCodeAttack then selects appropriate AST patterns, concretizes the selected patterns as attacks, and inserts them as dead code into the model's input program. To effectively synthesize attacks from AST patterns, GraphCodeAttack uses a separate pre-trained code model to fill in the ASTs with concrete code snippets. We evaluate the robustness of two popular code models (e.g., CodeBERT and GraphCodeBERT) against our proposed approach on three tasks: Authorship Attribution, Vulnerability Prediction, and Clone Detection. The experimental results suggest that our proposed approach significantly outperforms state-of-the-art approaches in attacking code models such as CARROT and ALERT.

摘要: 预先训练的代码语言模型现在被广泛用于各种软件工程任务，如代码生成、代码完成、漏洞检测等。这反过来又给这些模型带来了安全和可靠性风险。其中一个重要的威胁是对抗性攻击，它会导致错误的预测，并在很大程度上影响模型在下游任务上的性能。当前针对代码模型的对抗性攻击通常采用固定的程序转换集，如变量重命名和死代码插入，导致攻击效果有限。为了应对上述挑战，我们提出了一种新的对抗性攻击框架GraphCodeAttack，以更好地评估代码模型的健壮性。在给定目标代码模型的情况下，GraphCodeAttack自动挖掘可能影响模型决策的重要代码模式，以扰乱模型的输入代码结构。为此，GraphCodeAttack使用一组输入源代码来探测模型的输出，并识别可能影响模型决策的\textit{鉴别性}ASTS模式。然后，GraphCodeAttack选择适当的AST模式，将所选模式具体化为攻击，并将它们作为死代码插入到模型的输入程序中。为了有效地从AST模式合成攻击，GraphCodeAttack使用单独的预先训练的代码模型来用具体的代码片段填充AST。我们评估了两个流行的代码模型(例如，CodeBERT和GraphCodeBERT)在作者属性、漏洞预测和克隆检测三个任务上的健壮性。实验结果表明，我们提出的方法在攻击胡萝卜和ALERT等代码模型方面明显优于最先进的方法。



## **15. Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning**

修剪以保护：在无需微调的情况下提高对齐的LLM的越狱抵抗力 cs.LG

Proceedings of the 7th BlackboxNLP Workshop: Analyzing and  Interpreting Neural Networks for NLP

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2401.10862v3) [paper-pdf](http://arxiv.org/pdf/2401.10862v3)

**Authors**: Adib Hasan, Ileana Rugina, Alex Wang

**Abstract**: This paper investigates the impact of model compression on the way Large Language Models (LLMs) process prompts, particularly concerning jailbreak resistance. We show that moderate WANDA pruning can enhance resistance to jailbreaking attacks without fine-tuning, while maintaining performance on standard benchmarks. To systematically evaluate this safety enhancement, we introduce a dataset of 225 harmful tasks across five categories. Our analysis of LLaMA-2 Chat, Vicuna 1.3, and Mistral Instruct v0.2 reveals that pruning benefits correlate with initial model safety levels. We interpret these results by examining changes in attention patterns and perplexity shifts, demonstrating that pruned models exhibit sharper attention and increased sensitivity to artificial jailbreak constructs. We extend our evaluation to the AdvBench harmful behavior tasks and the GCG attack method. We find that LLaMA-2 is much safer on AdvBench prompts than on our dataset when evaluated with manual jailbreak attempts, and that pruning is effective against both automated attacks and manual jailbreaking on Advbench.

摘要: 本文研究了模型压缩对大型语言模型(LLMS)处理提示的方式的影响，特别是关于越狱抵抗的影响。我们表明，适度的万达剪枝可以在不进行微调的情况下增强对越狱攻击的抵抗力，同时保持标准基准测试的性能。为了系统地评估这一安全增强，我们引入了五个类别的225个有害任务的数据集。我们对骆驼-2聊天、维库纳1.3和米斯特拉尔指令v0.2的分析表明，修剪的好处与初始模型的安全级别相关。我们通过检测注意力模式和困惑转移的变化来解释这些结果，表明修剪后的模型表现出更敏锐的注意力和对人工越狱结构的敏感性。我们将我们的评估扩展到AdvBtch有害行为任务和GCG攻击方法。我们发现，当使用手动越狱尝试进行评估时，在AdvBtch提示上的骆驼-2比在我们的数据集上要安全得多，并且剪枝在Advbase上对自动攻击和手动越狱都是有效的。



## **16. HuRef: HUman-REadable Fingerprint for Large Language Models**

HuRef：大型语言模型的人类可读取指纹 cs.CL

NeurIPS2024

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2312.04828v3) [paper-pdf](http://arxiv.org/pdf/2312.04828v3)

**Authors**: Boyi Zeng, Lizheng Wang, Yuncong Hu, Yi Xu, Chenghu Zhou, Xinbing Wang, Yu Yu, Zhouhan Lin

**Abstract**: Protecting the copyright of large language models (LLMs) has become crucial due to their resource-intensive training and accompanying carefully designed licenses. However, identifying the original base model of an LLM is challenging due to potential parameter alterations. In this study, we introduce HuRef, a human-readable fingerprint for LLMs that uniquely identifies the base model without interfering with training or exposing model parameters to the public. We first observe that the vector direction of LLM parameters remains stable after the model has converged during pretraining, with negligible perturbations through subsequent training steps, including continued pretraining, supervised fine-tuning, and RLHF, which makes it a sufficient condition to identify the base model. The necessity is validated by continuing to train an LLM with an extra term to drive away the model parameters' direction and the model becomes damaged. However, this direction is vulnerable to simple attacks like dimension permutation or matrix rotation, which significantly change it without affecting performance. To address this, leveraging the Transformer structure, we systematically analyze potential attacks and define three invariant terms that identify an LLM's base model. Due to the potential risk of information leakage, we cannot publish invariant terms directly. Instead, we map them to a Gaussian vector using an encoder, then convert it into a natural image using StyleGAN2, and finally publish the image. In our black-box setting, all fingerprinting steps are internally conducted by the LLMs owners. To ensure the published fingerprints are honestly generated, we introduced Zero-Knowledge Proof (ZKP). Experimental results across various LLMs demonstrate the effectiveness of our method. The code is available at https://github.com/LUMIA-Group/HuRef.

摘要: 保护大型语言模型(LLM)的版权已变得至关重要，因为它们需要进行资源密集型培训，并附带精心设计的许可证。然而，由于潜在的参数变化，识别LLM的原始基础模型是具有挑战性的。在这项研究中，我们引入了HuRef，这是一种用于LLMS的人类可读指纹，它在不干扰训练或向公众暴露模型参数的情况下唯一地识别基本模型。我们首先观察到，在预训练过程中模型收敛后，LLM参数的向量方向保持稳定，通过后续的训练步骤，包括继续预训练、有监督的微调和RLHF，可以忽略不计的扰动，这使得它成为识别基本模型的充分条件。通过继续训练一个带有额外项的LLM来驱离模型参数的方向，从而使模型受损，从而验证了这种必要性。然而，这个方向很容易受到维度置换或矩阵旋转等简单攻击，这些攻击会在不影响性能的情况下显著改变它。为了解决这个问题，利用Transformer结构，我们系统地分析了潜在的攻击，并定义了识别LLM基本模型的三个不变术语。由于潜在的信息泄露风险，我们不能直接发布不变项。相反，我们使用编码器将它们映射到高斯向量，然后使用StyleGAN2将其转换为自然图像，最后发布图像。在我们的黑盒设置中，所有指纹识别步骤都由LLMS所有者在内部执行。为了确保公布的指纹是真实生成的，我们引入了零知识证明(ZKP)。在不同LLM上的实验结果证明了该方法的有效性。代码可在https://github.com/LUMIA-Group/HuRef.上获得



## **17. Transferable Ensemble Black-box Jailbreak Attacks on Large Language Models**

可转移集成黑匣子越狱攻击大型语言模型 cs.CR

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23558v1) [paper-pdf](http://arxiv.org/pdf/2410.23558v1)

**Authors**: Yiqi Yang, Hongye Fu

**Abstract**: In this report, we propose a novel black-box jailbreak attacking framework that incorporates various LLM-as-Attacker methods to deliver transferable and powerful jailbreak attacks. Our method is designed based on three key observations from existing jailbreaking studies and practices. First, we consider an ensemble approach should be more effective in exposing the vulnerabilities of an aligned LLM compared to individual attacks. Second, different malicious instructions inherently vary in their jailbreaking difficulty, necessitating differentiated treatment to ensure more efficient attacks. Finally, the semantic coherence of a malicious instruction is crucial for triggering the defenses of an aligned LLM; therefore, it must be carefully disrupted to manipulate its embedding representation, thereby increasing the jailbreak success rate. We validated our approach by participating in the Competition for LLM and Agent Safety 2024, where our team achieved top performance in the Jailbreaking Attack Track.

摘要: 在这份报告中，我们提出了一种新的黑盒越狱攻击框架，该框架结合了各种LLM作为攻击者的方法来提供可转移的强大越狱攻击。我们的方法是基于现有越狱研究和实践中的三个关键观察结果而设计的。首先，我们认为，与单独攻击相比，整体方法应该更有效地暴露联合LLM的漏洞。其次，不同的恶意指令在越狱难度上存在内在差异，需要区别对待，以确保更有效的攻击。最后，恶意指令的语义一致性对于触发对齐的LLM的防御至关重要；因此，必须小心破坏它才能操纵其嵌入表示，从而提高越狱成功率。我们通过参加LLM和代理安全竞赛2024来验证我们的方法，我们的团队在越狱攻击赛道上取得了最好的表现。



## **18. Representation Noising: A Defence Mechanism Against Harmful Finetuning**

代表噪音：防止有害微调的防御机制 cs.CL

Published in NeurIPs 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2405.14577v4) [paper-pdf](http://arxiv.org/pdf/2405.14577v4)

**Authors**: Domenic Rosati, Jan Wehner, Kai Williams, Łukasz Bartoszcze, David Atanasov, Robie Gonzales, Subhabrata Majumdar, Carsten Maple, Hassan Sajjad, Frank Rudzicz

**Abstract**: Releasing open-source large language models (LLMs) presents a dual-use risk since bad actors can easily fine-tune these models for harmful purposes. Even without the open release of weights, weight stealing and fine-tuning APIs make closed models vulnerable to harmful fine-tuning attacks (HFAs). While safety measures like preventing jailbreaks and improving safety guardrails are important, such measures can easily be reversed through fine-tuning. In this work, we propose Representation Noising (RepNoise), a defence mechanism that operates even when attackers have access to the weights. RepNoise works by removing information about harmful representations such that it is difficult to recover them during fine-tuning. Importantly, our defence is also able to generalize across different subsets of harm that have not been seen during the defence process as long as they are drawn from the same distribution of the attack set. Our method does not degrade the general capability of LLMs and retains the ability to train the model on harmless tasks. We provide empirical evidence that the efficacy of our defence lies in its ``depth'': the degree to which information about harmful representations is removed across all layers of the LLM. We also find areas where RepNoise still remains ineffective and highlight how those limitations can inform future research.

摘要: 发布开源的大型语言模型(LLM)存在双重用途的风险，因为不好的参与者很容易出于有害目的微调这些模型。即使没有公开的权重释放，权重盗窃和微调API也会使封闭的模型容易受到有害的微调攻击(HFA)。虽然防止越狱和改善安全护栏等安全措施很重要，但通过微调很容易逆转这些措施。在这项工作中，我们提出了表示噪声(RepNoise)，这是一种即使攻击者可以访问权重也可以操作的防御机制。RepNoise的工作原理是删除有关有害表示的信息，以便在微调期间很难恢复它们。重要的是，我们的防御还能够概括在防御过程中未曾见过的伤害的不同子集，只要它们来自相同分布的攻击集。我们的方法不会降低LLMS的整体性能，并保留了对模型进行无害任务训练的能力。我们提供的经验证据表明，我们的辩护的效力在于它的“深度”：在法律法规的所有层面上，关于有害陈述的信息被删除的程度。我们还发现了RepNoise仍然无效的领域，并强调了这些限制如何为未来的研究提供信息。



## **19. CARES: A Comprehensive Benchmark of Trustworthiness in Medical Vision Language Models**

CARES：医学视觉语言模型可信度的综合基准 cs.LG

NeurIPS 2024 Datasets and Benchmarks Track

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2406.06007v2) [paper-pdf](http://arxiv.org/pdf/2406.06007v2)

**Authors**: Peng Xia, Ze Chen, Juanxi Tian, Yangrui Gong, Ruibo Hou, Yue Xu, Zhenbang Wu, Zhiyuan Fan, Yiyang Zhou, Kangyu Zhu, Wenhao Zheng, Zhaoyang Wang, Xiao Wang, Xuchao Zhang, Chetan Bansal, Marc Niethammer, Junzhou Huang, Hongtu Zhu, Yun Li, Jimeng Sun, Zongyuan Ge, Gang Li, James Zou, Huaxiu Yao

**Abstract**: Artificial intelligence has significantly impacted medical applications, particularly with the advent of Medical Large Vision Language Models (Med-LVLMs), sparking optimism for the future of automated and personalized healthcare. However, the trustworthiness of Med-LVLMs remains unverified, posing significant risks for future model deployment. In this paper, we introduce CARES and aim to comprehensively evaluate the Trustworthiness of Med-LVLMs across the medical domain. We assess the trustworthiness of Med-LVLMs across five dimensions, including trustfulness, fairness, safety, privacy, and robustness. CARES comprises about 41K question-answer pairs in both closed and open-ended formats, covering 16 medical image modalities and 27 anatomical regions. Our analysis reveals that the models consistently exhibit concerns regarding trustworthiness, often displaying factual inaccuracies and failing to maintain fairness across different demographic groups. Furthermore, they are vulnerable to attacks and demonstrate a lack of privacy awareness. We publicly release our benchmark and code in https://cares-ai.github.io/.

摘要: 人工智能对医疗应用产生了重大影响，特别是随着医学大视觉语言模型(Med-LVLMS)的出现，引发了对自动化和个性化医疗保健未来的乐观情绪。然而，MED-LVLMS的可信性仍未得到验证，对未来的模型部署构成重大风险。在本文中，我们引入CARE，旨在全面评估医学领域的MED-LVLMS的可信性。我们从可信性、公平性、安全性、隐私性和健壮性五个维度评估Med-LVLM的可信性。CARE包括约41K个封闭式和开放式格式的问答对，涵盖16种医学影像模式和27个解剖区域。我们的分析表明，这些模型始终表现出对可信度的担忧，经常表现出事实上的不准确，并且未能在不同的人口群体中保持公平。此外，他们很容易受到攻击，并表现出缺乏隐私意识。我们在https://cares-ai.github.io/.中公开发布我们的基准测试和代码



## **20. ProTransformer: Robustify Transformers via Plug-and-Play Paradigm**

ProTransformer：通过即插即用范式的Robustify Transformers cs.LG

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.23182v1) [paper-pdf](http://arxiv.org/pdf/2410.23182v1)

**Authors**: Zhichao Hou, Weizhi Gao, Yuchen Shen, Feiyi Wang, Xiaorui Liu

**Abstract**: Transformer-based architectures have dominated various areas of machine learning in recent years. In this paper, we introduce a novel robust attention mechanism designed to enhance the resilience of transformer-based architectures. Crucially, this technique can be integrated into existing transformers as a plug-and-play layer, improving their robustness without the need for additional training or fine-tuning. Through comprehensive experiments and ablation studies, we demonstrate that our ProTransformer significantly enhances the robustness of transformer models across a variety of prediction tasks, attack mechanisms, backbone architectures, and data domains. Notably, without further fine-tuning, the ProTransformer consistently improves the performance of vanilla transformers by 19.5%, 28.3%, 16.1%, and 11.4% for BERT, ALBERT, DistilBERT, and RoBERTa, respectively, under the classical TextFooler attack. Furthermore, ProTransformer shows promising resilience in large language models (LLMs) against prompting-based attacks, improving the performance of T5 and LLaMA by 24.8% and 17.8%, respectively, and enhancing Vicuna by an average of 10.4% against the Jailbreaking attack. Beyond the language domain, ProTransformer also demonstrates outstanding robustness in both vision and graph domains.

摘要: 近年来，基于变压器的体系结构主导了机器学习的各个领域。在本文中，我们介绍了一种新的健壮注意机制，旨在增强基于变压器的体系结构的弹性。至关重要的是，这项技术可以作为即插即用层集成到现有的变压器中，无需额外的培训或微调即可提高其稳健性。通过全面的实验和烧蚀研究，我们证明我们的ProTransformer显著增强了变压器模型在各种预测任务、攻击机制、主干架构和数据域中的稳健性。值得注意的是，在没有进一步微调的情况下，ProTransformer在经典的TextFooler攻击下，分别将Bert、Albert、DistilBERT和Roberta的Vanilla变压器的性能分别提高了19.5%、28.3%、16.1%和11.4%。此外，ProTransformer在大型语言模型(LLM)中对基于提示的攻击表现出了良好的弹性，将T5和Llama的性能分别提高了24.8%和17.8%，对越狱攻击的维库纳平均提高了10.4%。除了语言领域，ProTransformer还在视觉和图形领域都表现出了出色的健壮性。



## **21. Effective and Efficient Adversarial Detection for Vision-Language Models via A Single Vector**

通过单个载体对视觉语言模型进行有效且高效的对抗检测 cs.CV

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22888v1) [paper-pdf](http://arxiv.org/pdf/2410.22888v1)

**Authors**: Youcheng Huang, Fengbin Zhu, Jingkun Tang, Pan Zhou, Wenqiang Lei, Jiancheng Lv, Tat-Seng Chua

**Abstract**: Visual Language Models (VLMs) are vulnerable to adversarial attacks, especially those from adversarial images, which is however under-explored in literature. To facilitate research on this critical safety problem, we first construct a new laRge-scale Adervsarial images dataset with Diverse hArmful Responses (RADAR), given that existing datasets are either small-scale or only contain limited types of harmful responses. With the new RADAR dataset, we further develop a novel and effective iN-time Embedding-based AdveRSarial Image DEtection (NEARSIDE) method, which exploits a single vector that distilled from the hidden states of VLMs, which we call the attacking direction, to achieve the detection of adversarial images against benign ones in the input. Extensive experiments with two victim VLMs, LLaVA and MiniGPT-4, well demonstrate the effectiveness, efficiency, and cross-model transferrability of our proposed method. Our code is available at https://github.com/mob-scu/RADAR-NEARSIDE

摘要: 视觉语言模型（VLM）很容易受到对抗性攻击，尤其是来自对抗性图像的攻击，但文献中对此尚未充分探讨。为了促进对这个关键安全问题的研究，我们首先构建一个具有多样性干扰响应（RADART）的新的大规模Adervsarial图像数据集，因为现有数据集要么小规模，要么仅包含有限类型的有害反应。利用新的雷达数据集，我们进一步开发了一种新颖且有效的基于iN时间嵌入的AdveRSarial Image Detect（NEARSIDE）方法，该方法利用从VLM的隐藏状态（我们称之为攻击方向）中提取的单个载体，以实现针对输入中良性图像的对抗图像的检测。对两个受害VLM（LLaVA和MiniGPT-4）进行的大量实验很好地证明了我们提出的方法的有效性、效率和跨模型可移植性。我们的代码可在https://github.com/mob-scu/RADAR-NEARSIDE上获取



## **22. Stealth edits to large language models**

对大型语言模型的隐形编辑 cs.AI

28 pages, 14 figures. Open source implementation:  https://github.com/qinghua-zhou/stealth-edits

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2406.12670v2) [paper-pdf](http://arxiv.org/pdf/2406.12670v2)

**Authors**: Oliver J. Sutton, Qinghua Zhou, Wei Wang, Desmond J. Higham, Alexander N. Gorban, Alexander Bastounis, Ivan Y. Tyukin

**Abstract**: We reveal the theoretical foundations of techniques for editing large language models, and present new methods which can do so without requiring retraining. Our theoretical insights show that a single metric (a measure of the intrinsic dimension of the model's features) can be used to assess a model's editability and reveals its previously unrecognised susceptibility to malicious stealth attacks. This metric is fundamental to predicting the success of a variety of editing approaches, and reveals new bridges between disparate families of editing methods. We collectively refer to these as stealth editing methods, because they directly update a model's weights to specify its response to specific known hallucinating prompts without affecting other model behaviour. By carefully applying our theoretical insights, we are able to introduce a new jet-pack network block which is optimised for highly selective model editing, uses only standard network operations, and can be inserted into existing networks. We also reveal the vulnerability of language models to stealth attacks: a small change to a model's weights which fixes its response to a single attacker-chosen prompt. Stealth attacks are computationally simple, do not require access to or knowledge of the model's training data, and therefore represent a potent yet previously unrecognised threat to redistributed foundation models. Extensive experimental results illustrate and support our methods and their theoretical underpinnings. Demos and source code are available at https://github.com/qinghua-zhou/stealth-edits.

摘要: 我们揭示了编辑大型语言模型的技术的理论基础，并提出了无需重新培训就能做到这一点的新方法。我们的理论见解表明，可以使用单一指标(模型特征的内在维度的衡量标准)来评估模型的可编辑性，并揭示其先前未被识别的对恶意隐形攻击的易感性。这一指标是预测各种编辑方法成功与否的基础，并揭示了不同编辑方法家族之间的新桥梁。我们将这些统称为隐形编辑方法，因为它们直接更新模型的权重，以指定其对特定已知幻觉提示的反应，而不影响其他模型的行为。通过仔细应用我们的理论见解，我们能够推出一种新的喷气式网络块，它针对高度选择性的模型编辑进行了优化，只使用标准的网络操作，并且可以插入到现有的网络中。我们还揭示了语言模型对隐形攻击的脆弱性：对模型的权重进行微小的更改即可修复其对单个攻击者选择的提示的响应。隐形攻击在计算上很简单，不需要访问或了解模型的训练数据，因此对重新分布的基础模型构成了以前未识别的强大威胁。大量的实验结果说明和支持了我们的方法及其理论基础。有关演示和源代码，请访问https://github.com/qinghua-zhou/stealth-edits.



## **23. HijackRAG: Hijacking Attacks against Retrieval-Augmented Large Language Models**

HijackRAG：针对检索增强大型语言模型的劫持攻击 cs.CR

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22832v1) [paper-pdf](http://arxiv.org/pdf/2410.22832v1)

**Authors**: Yucheng Zhang, Qinfeng Li, Tianyu Du, Xuhong Zhang, Xinkui Zhao, Zhengwen Feng, Jianwei Yin

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge, making them adaptable and cost-effective for various applications. However, the growing reliance on these systems also introduces potential security risks. In this work, we reveal a novel vulnerability, the retrieval prompt hijack attack (HijackRAG), which enables attackers to manipulate the retrieval mechanisms of RAG systems by injecting malicious texts into the knowledge database. When the RAG system encounters target questions, it generates the attacker's pre-determined answers instead of the correct ones, undermining the integrity and trustworthiness of the system. We formalize HijackRAG as an optimization problem and propose both black-box and white-box attack strategies tailored to different levels of the attacker's knowledge. Extensive experiments on multiple benchmark datasets show that HijackRAG consistently achieves high attack success rates, outperforming existing baseline attacks. Furthermore, we demonstrate that the attack is transferable across different retriever models, underscoring the widespread risk it poses to RAG systems. Lastly, our exploration of various defense mechanisms reveals that they are insufficient to counter HijackRAG, emphasizing the urgent need for more robust security measures to protect RAG systems in real-world deployments.

摘要: 检索-增强生成(RAG)系统通过集成外部知识来增强大型语言模型(LLM)，使它们能够适应各种应用并具有成本效益。然而，对这些系统的日益依赖也带来了潜在的安全风险。在这项工作中，我们揭示了一个新的漏洞--检索即时劫持攻击(HijackRAG)，该漏洞使攻击者能够通过向知识库中注入恶意文本来操纵RAG系统的检索机制。当RAG系统遇到目标问题时，它会生成攻击者的预定答案而不是正确的答案，从而破坏系统的完整性和可信性。我们将HijackRAG形式化为一个优化问题，并根据攻击者的不同知识水平提出了黑盒和白盒攻击策略。在多个基准数据集上的大量实验表明，HijackRAG始终具有较高的攻击成功率，性能优于现有的基线攻击。此外，我们证明了攻击可以在不同的取回器模型之间转移，强调了它对RAG系统构成的广泛风险。最后，我们对各种防御机制的探索表明，它们不足以对抗HijackRAG，强调迫切需要更强大的安全措施来保护现实世界部署中的RAG系统。



## **24. InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models**

InjecGuard：对标和缓解即时注射保障模型中的过度防御 cs.CL

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22770v1) [paper-pdf](http://arxiv.org/pdf/2410.22770v1)

**Authors**: Hao Li, Xiaogeng Liu, Chaowei Xiao

**Abstract**: Prompt injection attacks pose a critical threat to large language models (LLMs), enabling goal hijacking and data leakage. Prompt guard models, though effective in defense, suffer from over-defense -- falsely flagging benign inputs as malicious due to trigger word bias. To address this issue, we introduce NotInject, an evaluation dataset that systematically measures over-defense across various prompt guard models. NotInject contains 339 benign samples enriched with trigger words common in prompt injection attacks, enabling fine-grained evaluation. Our results show that state-of-the-art models suffer from over-defense issues, with accuracy dropping close to random guessing levels (60%). To mitigate this, we propose InjecGuard, a novel prompt guard model that incorporates a new training strategy, Mitigating Over-defense for Free (MOF), which significantly reduces the bias on trigger words. InjecGuard demonstrates state-of-the-art performance on diverse benchmarks including NotInject, surpassing the existing best model by 30.8%, offering a robust and open-source solution for detecting prompt injection attacks. The code and datasets are released at https://github.com/SaFoLab-WISC/InjecGuard.

摘要: 快速注入攻击对大型语言模型(LLM)构成严重威胁，导致目标劫持和数据泄露。即时保护模式虽然在防御方面有效，但也存在过度防御的问题--由于触发单词偏见，错误地将良性输入标记为恶意输入。为了解决这个问题，我们引入了NotInject，这是一个评估数据集，系统地测量各种提示防护模型中的过度防御。NotInject包含339个良性样本，丰富了提示注入攻击中常见的触发字，实现了细粒度评估。我们的结果表明，最先进的模型存在过度防御的问题，准确率下降到接近随机猜测的水平(60%)。为了缓解这一问题，我们提出了InjecGuard，一种新的提示守卫模型，它结合了新的训练策略，缓解了过度防御For Free(MOF)，大大减少了对触发词的偏见。InjecGuard在包括NotInject在内的各种基准测试上展示了最先进的性能，比现有最好的模型高出30.8%，为检测即时注入攻击提供了一个强大的开源解决方案。代码和数据集在https://github.com/SaFoLab-WISC/InjecGuard.上发布



## **25. Uncovering Coordinated Cross-Platform Information Operations Threatening the Integrity of the 2024 U.S. Presidential Election Online Discussion**

揭露威胁2024年美国总统选举在线讨论完整性的协调跨平台信息操作 cs.SI

First Monday 29(11), 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2409.15402v2) [paper-pdf](http://arxiv.org/pdf/2409.15402v2)

**Authors**: Marco Minici, Luca Luceri, Federico Cinus, Emilio Ferrara

**Abstract**: Information Operations (IOs) pose a significant threat to the integrity of democratic processes, with the potential to influence election-related online discourse. In anticipation of the 2024 U.S. presidential election, we present a study aimed at uncovering the digital traces of coordinated IOs on $\mathbb{X}$ (formerly Twitter). Using our machine learning framework for detecting online coordination, we analyze a dataset comprising election-related conversations on $\mathbb{X}$ from May 2024. This reveals a network of coordinated inauthentic actors, displaying notable similarities in their link-sharing behaviors. Our analysis shows concerted efforts by these accounts to disseminate misleading, redundant, and biased information across the Web through a coordinated cross-platform information operation: The links shared by this network frequently direct users to other social media platforms or suspicious websites featuring low-quality political content and, in turn, promoting the same $\mathbb{X}$ and YouTube accounts. Members of this network also shared deceptive images generated by AI, accompanied by language attacking political figures and symbolic imagery intended to convey power and dominance. While $\mathbb{X}$ has suspended a subset of these accounts, more than 75% of the coordinated network remains active. Our findings underscore the critical role of developing computational models to scale up the detection of threats on large social media platforms, and emphasize the broader implications of these techniques to detect IOs across the wider Web.

摘要: 信息业务对民主进程的完整性构成重大威胁，有可能影响与选举有关的网上言论。在对2024年美国总统大选的预期中，我们提出了一项研究，旨在揭示$\mathbb{X}$(前身为Twitter)上协调iOS的数字痕迹。使用我们用于检测在线协调的机器学习框架，我们分析了一个包含2024年5月以来$\mathbb{X}$的选举相关对话的数据集。这揭示了一个协调的不真实行为者网络，显示出他们在链接分享行为上的显著相似之处。我们的分析显示，这些账户协同努力，通过协调的跨平台信息操作在网络上传播误导性、冗余和有偏见的信息：该网络共享的链接经常将用户定向到其他社交媒体平台或具有低质量政治内容的可疑网站，进而推广相同的$\mathbb{X}$和YouTube账户。该网络的成员还分享了人工智能生成的欺骗性图像，并伴随着攻击政治人物的语言和旨在传达权力和主导地位的象征性图像。虽然$\mathbb{X}$已暂停这些帐户的子集，但超过75%的协调网络仍处于活动状态。我们的发现强调了开发计算模型以扩大大型社交媒体平台上的威胁检测的关键作用，并强调了这些技术在更广泛的网络上检测iOS的更广泛影响。



## **26. SVIP: Towards Verifiable Inference of Open-source Large Language Models**

SVIP：迈向开源大型语言模型的可验证推理 cs.LG

20 pages

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22307v1) [paper-pdf](http://arxiv.org/pdf/2410.22307v1)

**Authors**: Yifan Sun, Yuhang Li, Yue Zhang, Yuchen Jin, Huan Zhang

**Abstract**: Open-source Large Language Models (LLMs) have recently demonstrated remarkable capabilities in natural language understanding and generation, leading to widespread adoption across various domains. However, their increasing model sizes render local deployment impractical for individual users, pushing many to rely on computing service providers for inference through a blackbox API. This reliance introduces a new risk: a computing provider may stealthily substitute the requested LLM with a smaller, less capable model without consent from users, thereby delivering inferior outputs while benefiting from cost savings. In this paper, we formalize the problem of verifiable inference for LLMs. Existing verifiable computing solutions based on cryptographic or game-theoretic techniques are either computationally uneconomical or rest on strong assumptions. We introduce SVIP, a secret-based verifiable LLM inference protocol that leverages intermediate outputs from LLM as unique model identifiers. By training a proxy task on these outputs and requiring the computing provider to return both the generated text and the processed intermediate outputs, users can reliably verify whether the computing provider is acting honestly. In addition, the integration of a secret mechanism further enhances the security of our protocol. We thoroughly analyze our protocol under multiple strong and adaptive adversarial scenarios. Our extensive experiments demonstrate that SVIP is accurate, generalizable, computationally efficient, and resistant to various attacks. Notably, SVIP achieves false negative rates below 5% and false positive rates below 3%, while requiring less than 0.01 seconds per query for verification.

摘要: 开源的大型语言模型(LLM)最近在自然语言理解和生成方面表现出了非凡的能力，导致了在各个领域的广泛采用。然而，它们不断增长的模型规模使得本地部署对个人用户来说是不现实的，促使许多人依赖计算服务提供商通过黑盒API进行推理。这种依赖带来了新的风险：计算提供商可能会在未经用户同意的情况下，悄悄地用较小、功能较差的模型替换所请求的LLM，从而在提供劣质产出的同时受益于成本节约。本文对LLMS的可验证推理问题进行了形式化描述。现有的基于密码学或博弈论技术的可验证计算解决方案要么在计算上不经济，要么依赖于强有力的假设。我们引入了SVIP，这是一个基于秘密的可验证LLM推理协议，它利用LLM的中间输出作为唯一的模型标识符。通过对这些输出训练代理任务并要求计算提供商返回生成的文本和处理的中间输出，用户可以可靠地验证计算提供商是否诚实行事。此外，秘密机制的集成进一步增强了协议的安全性。我们深入分析了我们的协议在多种强和自适应对抗场景下的性能。大量实验表明，SVIP算法具有较高的准确性、通用性、计算效率和抵抗各种攻击的能力。值得注意的是，SVIP实现了5%以下的假阴性率和3%以下的假阳性率，而每次查询验证所需的时间不到0.01秒。



## **27. Embedding-based classifiers can detect prompt injection attacks**

基于嵌入的分类器可以检测提示注入攻击 cs.CR

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22284v1) [paper-pdf](http://arxiv.org/pdf/2410.22284v1)

**Authors**: Md. Ahsan Ayub, Subhabrata Majumdar

**Abstract**: Large Language Models (LLMs) are seeing significant adoption in every type of organization due to their exceptional generative capabilities. However, LLMs are found to be vulnerable to various adversarial attacks, particularly prompt injection attacks, which trick them into producing harmful or inappropriate content. Adversaries execute such attacks by crafting malicious prompts to deceive the LLMs. In this paper, we propose a novel approach based on embedding-based Machine Learning (ML) classifiers to protect LLM-based applications against this severe threat. We leverage three commonly used embedding models to generate embeddings of malicious and benign prompts and utilize ML classifiers to predict whether an input prompt is malicious. Out of several traditional ML methods, we achieve the best performance with classifiers built using Random Forest and XGBoost. Our classifiers outperform state-of-the-art prompt injection classifiers available in open-source implementations, which use encoder-only neural networks.

摘要: 大型语言模型(LLM)由于其非凡的生成能力，在每种类型的组织中都得到了大量采用。然而，LLM被发现容易受到各种对抗性攻击，特别是即时注入攻击，这些攻击会诱使它们产生有害或不适当的内容。攻击者通过精心编制恶意提示来欺骗LLM，从而执行此类攻击。在本文中，我们提出了一种基于嵌入的机器学习(ML)分类器的新方法来保护基于LLM的应用程序免受这种严重威胁。我们利用三种常用的嵌入模型来生成恶意提示和良性提示的嵌入，并利用ML分类器来预测输入提示是否为恶意提示。在几种传统的最大似然分类方法中，我们使用随机森林和XGBoost构建的分类器取得了最好的性能。我们的分类器比开源实现中可用的最先进的提示注入分类器性能更好，后者使用仅限编码器的神经网络。



## **28. AmpleGCG-Plus: A Strong Generative Model of Adversarial Suffixes to Jailbreak LLMs with Higher Success Rates in Fewer Attempts**

AmpleGCG-Plus：越狱LLC的对抗性后缀的强生成模型，以更少的尝试获得更高的成功率 cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22143v1) [paper-pdf](http://arxiv.org/pdf/2410.22143v1)

**Authors**: Vishal Kumar, Zeyi Liao, Jaylen Jones, Huan Sun

**Abstract**: Although large language models (LLMs) are typically aligned, they remain vulnerable to jailbreaking through either carefully crafted prompts in natural language or, interestingly, gibberish adversarial suffixes. However, gibberish tokens have received relatively less attention despite their success in attacking aligned LLMs. Recent work, AmpleGCG~\citep{liao2024amplegcg}, demonstrates that a generative model can quickly produce numerous customizable gibberish adversarial suffixes for any harmful query, exposing a range of alignment gaps in out-of-distribution (OOD) language spaces. To bring more attention to this area, we introduce AmpleGCG-Plus, an enhanced version that achieves better performance in fewer attempts. Through a series of exploratory experiments, we identify several training strategies to improve the learning of gibberish suffixes. Our results, verified under a strict evaluation setting, show that it outperforms AmpleGCG on both open-weight and closed-source models, achieving increases in attack success rate (ASR) of up to 17\% in the white-box setting against Llama-2-7B-chat, and more than tripling ASR in the black-box setting against GPT-4. Notably, AmpleGCG-Plus jailbreaks the newer GPT-4o series of models at similar rates to GPT-4, and, uncovers vulnerabilities against the recently proposed circuit breakers defense. We publicly release AmpleGCG-Plus along with our collected training datasets.

摘要: 尽管大型语言模型(LLM)通常是一致的，但它们仍然很容易通过精心设计的自然语言提示或有趣的胡言乱语对抗性后缀越狱。然而，令人费解的令牌尽管成功地攻击了对齐的LLM，但受到的关注相对较少。最近的工作，AmpleGCG~\Citep{Lio2024Amplegcg}，证明了生成模型可以为任何有害的查询快速生成大量可定制的胡言乱语对抗性后缀，从而暴露出分布外(OOD)语言空间中的一系列对齐差距。为了引起人们对这一领域的更多关注，我们推出了AmpleGCG-Plus，这是一个增强版本，在较少的尝试中获得了更好的性能。通过一系列的探索性实验，我们确定了几种训练策略来提高乱码后缀的学习效果。在严格的评估设置下验证的结果表明，它在开源和闭源模型上都优于AmpleGCG，在白盒环境下相对于Llama-2-7B-Chat的攻击成功率(ASR)提高了17%，在黑盒环境下相对于GPT-4的攻击成功率(ASR)提高了两倍以上。值得注意的是，AmpleGCG-Plus以类似于GPT-4的速度监禁了较新的GPT-4o系列型号，并揭示了针对最近提出的断路器防御的漏洞。我们公开发布AmpleGCG-Plus以及我们收集的训练数据集。



## **29. Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents**

小心你的特工！调查对LLM代理的后门威胁 cs.CR

Accepted at NeurIPS 2024, camera ready version. Code and data are  available at https://github.com/lancopku/agent-backdoor-attacks

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2402.11208v2) [paper-pdf](http://arxiv.org/pdf/2402.11208v2)

**Authors**: Wenkai Yang, Xiaohan Bi, Yankai Lin, Sishuo Chen, Jie Zhou, Xu Sun

**Abstract**: Driven by the rapid development of Large Language Models (LLMs), LLM-based agents have been developed to handle various real-world applications, including finance, healthcare, and shopping, etc. It is crucial to ensure the reliability and security of LLM-based agents during applications. However, the safety issues of LLM-based agents are currently under-explored. In this work, we take the first step to investigate one of the typical safety threats, backdoor attack, to LLM-based agents. We first formulate a general framework of agent backdoor attacks, then we present a thorough analysis of different forms of agent backdoor attacks. Specifically, compared with traditional backdoor attacks on LLMs that are only able to manipulate the user inputs and model outputs, agent backdoor attacks exhibit more diverse and covert forms: (1) From the perspective of the final attacking outcomes, the agent backdoor attacker can not only choose to manipulate the final output distribution, but also introduce the malicious behavior in an intermediate reasoning step only, while keeping the final output correct. (2) Furthermore, the former category can be divided into two subcategories based on trigger locations, in which the backdoor trigger can either be hidden in the user query or appear in an intermediate observation returned by the external environment. We implement the above variations of agent backdoor attacks on two typical agent tasks including web shopping and tool utilization. Extensive experiments show that LLM-based agents suffer severely from backdoor attacks and such backdoor vulnerability cannot be easily mitigated by current textual backdoor defense algorithms. This indicates an urgent need for further research on the development of targeted defenses against backdoor attacks on LLM-based agents. Warning: This paper may contain biased content.

摘要: 在大型语言模型(LLM)快速发展的推动下，基于LLM的代理被开发出来处理各种现实世界的应用，包括金融、医疗和购物等。然而，基于LLM的制剂的安全性问题目前还没有得到充分的研究。在这项工作中，我们首先调查了LLM代理面临的一种典型的安全威胁--后门攻击。我们首先建立了代理后门攻击的一般框架，然后深入分析了代理后门攻击的不同形式。具体地说，与传统的仅能操纵用户输入和模型输出的后门攻击相比，代理后门攻击表现出更多样和隐蔽的形式：(1)从最终攻击结果来看，代理后门攻击者不仅可以选择操纵最终输出分布，而且只能在中间推理步骤中引入恶意行为，同时保持最终输出的正确性。(2)此外，根据触发器的位置，前者可以分为两个子类，其中后门触发器可以隐藏在用户查询中，也可以出现在外部环境返回的中间观察中。我们在两个典型的代理任务上实现了上述变体的代理后门攻击，包括网上购物和工具利用。大量的实验表明，基于LLM的代理受到严重的后门攻击，而这种后门漏洞不能通过现有的文本后门防御算法轻松缓解。这表明迫切需要进一步研究针对基于LLM的特工的后门攻击开发有针对性的防御措施。警告：此论文可能包含有偏见的内容。



## **30. Waterfall: Framework for Robust and Scalable Text Watermarking and Provenance for LLMs**

Waterfall：LLM稳健且可扩展的文本水印和起源框架 cs.CR

Accepted to EMNLP 2024 Main Conference

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2407.04411v2) [paper-pdf](http://arxiv.org/pdf/2407.04411v2)

**Authors**: Gregory Kang Ruey Lau, Xinyuan Niu, Hieu Dao, Jiangwei Chen, Chuan-Sheng Foo, Bryan Kian Hsiang Low

**Abstract**: Protecting intellectual property (IP) of text such as articles and code is increasingly important, especially as sophisticated attacks become possible, such as paraphrasing by large language models (LLMs) or even unauthorized training of LLMs on copyrighted text to infringe such IP. However, existing text watermarking methods are not robust enough against such attacks nor scalable to millions of users for practical implementation. In this paper, we propose Waterfall, the first training-free framework for robust and scalable text watermarking applicable across multiple text types (e.g., articles, code) and languages supportable by LLMs, for general text and LLM data provenance. Waterfall comprises several key innovations, such as being the first to use LLM as paraphrasers for watermarking along with a novel combination of techniques that are surprisingly effective in achieving robust verifiability and scalability. We empirically demonstrate that Waterfall achieves significantly better scalability, robust verifiability, and computational efficiency compared to SOTA article-text watermarking methods, and showed how it could be directly applied to the watermarking of code. We also demonstrated that Waterfall can be used for LLM data provenance, where the watermarks of LLM training data can be detected in LLM output, allowing for detection of unauthorized use of data for LLM training and potentially enabling model-centric watermarking of open-sourced LLMs which has been a limitation of existing LLM watermarking works. Our code is available at https://github.com/aoi3142/Waterfall.

摘要: 保护文章和代码等文本的知识产权(IP)越来越重要，特别是在可能进行复杂攻击的情况下，例如利用大型语言模型(LLM)进行释义，甚至未经授权对受版权保护的文本进行LLM培训，以侵犯此类IP。然而，现有的文本水印方法对此类攻击不够健壮，也不能扩展到数百万用户进行实际实现。在本文中，我们提出了瀑布，这是第一个无训练的文本水印框架，适用于LLMS支持的多种文本类型(例如，文章、代码)和语言，用于一般文本和LLM数据来源。瀑布由几项关键创新组成，例如第一个使用LLM作为水印解释程序，以及在实现强大的可验证性和可扩展性方面出人意料地有效的技术组合。实验证明，与SOTA的文章-文本水印方法相比，瀑布算法具有更好的可扩展性、健壮性、可验证性和计算效率，并且可以直接应用于代码的水印。我们还演示了瀑布可以用于LLM数据起源，其中LLM训练数据的水印可以在LLM输出中检测到，允许检测未经授权使用数据进行LLM训练，并潜在地实现了以模型为中心的开放源代码LLMS水印，这一直是现有LLM水印工作的限制。我们的代码可以在https://github.com/aoi3142/Waterfall.上找到



## **31. Enhancing Adversarial Attacks through Chain of Thought**

通过思维链增强对抗性攻击 cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21791v1) [paper-pdf](http://arxiv.org/pdf/2410.21791v1)

**Authors**: Jingbo Su

**Abstract**: Large language models (LLMs) have demonstrated impressive performance across various domains but remain susceptible to safety concerns. Prior research indicates that gradient-based adversarial attacks are particularly effective against aligned LLMs and the chain of thought (CoT) prompting can elicit desired answers through step-by-step reasoning. This paper proposes enhancing the robustness of adversarial attacks on aligned LLMs by integrating CoT prompts with the greedy coordinate gradient (GCG) technique. Using CoT triggers instead of affirmative targets stimulates the reasoning abilities of backend LLMs, thereby improving the transferability and universality of adversarial attacks. We conducted an ablation study comparing our CoT-GCG approach with Amazon Web Services auto-cot. Results revealed our approach outperformed both the baseline GCG attack and CoT prompting. Additionally, we used Llama Guard to evaluate potentially harmful interactions, providing a more objective risk assessment of entire conversations compared to matching outputs to rejection phrases. The code of this paper is available at https://github.com/sujingbo0217/CS222W24-LLM-Attack.

摘要: 大型语言模型(LLM)在各个领域都表现出了令人印象深刻的表现，但仍然容易受到安全问题的影响。以往的研究表明，基于梯度的对抗性攻击对于对齐的LLM特别有效，而思维链(COT)提示可以通过循序渐进的推理获得期望的答案。提出了一种将CoT提示与贪婪坐标梯度(GCG)技术相结合的方法，以增强对齐LLMS的敌意攻击的稳健性。使用CoT触发器代替肯定目标，刺激了后端LLMS的推理能力，从而提高了对抗性攻击的可转移性和通用性。我们进行了一项烧蚀研究，将我们的COT-GCG方法与Amazon Web Services自动COT进行了比较。结果显示，我们的方法比基线GCG攻击和COT提示都要好。此外，我们使用Llama Guard来评估潜在的有害交互，与将输出与拒绝短语匹配相比，提供了对整个对话的更客观的风险评估。本文的代码可在https://github.com/sujingbo0217/CS222W24-LLM-Attack.上找到



## **32. Vaccine: Perturbation-aware Alignment for Large Language Models against Harmful Fine-tuning Attack**

疫苗：大型语言模型的扰动感知对齐以对抗有害的微调攻击 cs.LG

Rejected by ICML2024. Accepted by NeurIPS2024

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2402.01109v5) [paper-pdf](http://arxiv.org/pdf/2402.01109v5)

**Authors**: Tiansheng Huang, Sihao Hu, Ling Liu

**Abstract**: The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompts. Our code is available at \url{https://github.com/git-disl/Vaccine}.

摘要: 精调即服务的新范式为大型语言模型(LLM)引入了一个新的攻击面：用户上传的少量有害数据就可以很容易地欺骗精调，产生一个破坏对齐的模型。我们进行了实证分析，发现了一种有害的嵌入漂移现象，揭示了排列断裂效应的可能原因。受我们发现的启发，我们提出了Vaccine，一种扰动感知的对齐技术，以降低用户精调的安全风险。Vaccine的核心思想是通过在比对阶段逐步向其添加精心制作的扰动来产生不变的隐藏嵌入。这使嵌入能够在精细调整阶段抵御来自未清理的用户数据的有害干扰。我们在开源主流LLMS(如Llama2、Opt、Vicuna)上的实验结果表明，疫苗可以提高对有害提示导致的嵌入漂移的健壮性，同时保留对良性提示的推理能力。我们的代码可在\url{https://github.com/git-disl/Vaccine}.



## **33. Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey**

针对大型语言模型的有害微调攻击和防御：调查 cs.CR

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2409.18169v4) [paper-pdf](http://arxiv.org/pdf/2409.18169v4)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Recent research demonstrates that the nascent fine-tuning-as-a-service business model exposes serious safety concerns -- fine-tuning over a few harmful data uploaded by the users can compromise the safety alignment of the model. The attack, known as harmful fine-tuning, has raised a broad research interest among the community. However, as the attack is still new, \textbf{we observe from our miserable submission experience that there are general misunderstandings within the research community.} We in this paper aim to clear some common concerns for the attack setting, and formally establish the research problem. Specifically, we first present the threat model of the problem, and introduce the harmful fine-tuning attack and its variants. Then we systematically survey the existing literature on attacks/defenses/mechanical analysis of the problem. Finally, we outline future research directions that might contribute to the development of the field. Additionally, we present a list of questions of interest, which might be useful to refer to when reviewers in the peer review process question the realism of the experiment/attack/defense setting. A curated list of relevant papers is maintained and made accessible at: \url{https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers}.

摘要: 最近的研究表明，新兴的微调即服务商业模式暴露了严重的安全问题--对用户上传的几个有害数据进行微调可能会损害该模型的安全一致性。这一被称为有害微调的攻击在社区中引起了广泛的研究兴趣。然而，由于攻击仍然是新的，\extbf{我们从悲惨的提交经验中观察到，研究界普遍存在误解。}我们在本文中旨在澄清一些对攻击设置的共同关注，并正式确立研究问题。具体地说，我们首先给出了问题的威胁模型，并介绍了有害的微调攻击及其变体。然后，我们系统地综述了现有的关于攻击/防御/机械分析问题的文献。最后，我们概述了未来的研究方向，可能有助于该领域的发展。此外，我们提供了一个感兴趣的问题列表，当同行审查过程中的评审者质疑实验/攻击/防御设置的真实性时，这些问题可能会有用。相关论文的精选清单可在以下网址查阅：\url{https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers}.



## **34. Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning Attack**

Lisa：大型语言模型的懒惰安全调整以应对有害的微调攻击 cs.LG

Accepted by NeurIPS2024. arXiv admin note: substantial text overlap  with arXiv:2402.01109

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2405.18641v5) [paper-pdf](http://arxiv.org/pdf/2405.18641v5)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Recent studies show that Large Language Models (LLMs) with safety alignment can be jail-broken by fine-tuning on a dataset mixed with harmful data. First time in the literature, we show that the jail-broken effect can be mitigated by separating states in the finetuning stage to optimize the alignment and user datasets. Unfortunately, our subsequent study shows that this simple Bi-State Optimization (BSO) solution experiences convergence instability when steps invested in its alignment state is too small, leading to downgraded alignment performance. By statistical analysis, we show that the \textit{excess drift} towards consensus could be a probable reason for the instability. To remedy this issue, we propose \textbf{L}azy(\textbf{i}) \textbf{s}afety \textbf{a}lignment (\textbf{Lisa}), which introduces a proximal term to constraint the drift of each state. Theoretically, the benefit of the proximal term is supported by the convergence analysis, wherein we show that a sufficient large proximal factor is necessary to guarantee Lisa's convergence. Empirically, our results on four downstream finetuning tasks show that Lisa with a proximal term can significantly increase alignment performance while maintaining the LLM's accuracy on the user tasks. Code is available at \url{https://github.com/git-disl/Lisa}.

摘要: 最近的研究表明，通过对混合了有害数据的数据集进行微调，安全对齐的大型语言模型(LLM)可以越狱。在文献中，我们第一次证明了可以通过在精调阶段分离状态来优化比对和用户数据集来缓解越狱效应。不幸的是，我们随后的研究表明，当在其对齐状态中投入的步长太小时，这种简单的双状态优化(BSO)解决方案会经历收敛不稳定，从而导致对齐性能下降。通过统计分析，我们发现，向共识的过度漂移可能是导致不稳定的一个可能原因。为了解决这个问题，我们提出了Textbf{L}azy(\Textbf{i})\Textbf{S}安全对齐(Textbf{Lisa})，它引入了一个近邻项来约束每个态的漂移。理论上，近似项的好处得到了收敛分析的支持，其中我们证明了一个足够大的近似项是保证LISA收敛的必要条件。经验性地，我们在四个下游精调任务上的结果表明，具有近端项的LISA可以显著提高对齐性能，同时保持LLM在用户任务上的准确性。代码位于\url{https://github.com/git-disl/Lisa}.



## **35. Fine-tuning Large Language Models for DGA and DNS Exfiltration Detection**

用于DGA和DNS溢出检测的微调大型语言模型 cs.CR

Accepted in Proceedings of the Workshop at AI for Cyber Threat  Intelligence (WAITI), 2024

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21723v1) [paper-pdf](http://arxiv.org/pdf/2410.21723v1)

**Authors**: Md Abu Sayed, Asif Rahman, Christopher Kiekintveld, Sebastian Garcia

**Abstract**: Domain Generation Algorithms (DGAs) are malicious techniques used by malware to dynamically generate seemingly random domain names for communication with Command & Control (C&C) servers. Due to the fast and simple generation of DGA domains, detection methods must be highly efficient and precise to be effective. Large Language Models (LLMs) have demonstrated their proficiency in real-time detection tasks, making them ideal candidates for detecting DGAs. Our work validates the effectiveness of fine-tuned LLMs for detecting DGAs and DNS exfiltration attacks. We developed LLM models and conducted comprehensive evaluation using a diverse dataset comprising 59 distinct real-world DGA malware families and normal domain data. Our LLM model significantly outperformed traditional natural language processing techniques, especially in detecting unknown DGAs. We also evaluated its performance on DNS exfiltration datasets, demonstrating its effectiveness in enhancing cybersecurity measures. To the best of our knowledge, this is the first work that empirically applies LLMs for DGA and DNS exfiltration detection.

摘要: 域生成算法(DGA)是恶意软件用来动态生成看似随机的域名以与命令与控制(C&C)服务器通信的恶意技术。由于DGA结构域的快速而简单的生成，检测方法必须高效和精确才能有效。大型语言模型(LLM)已经证明了它们在实时检测任务中的熟练程度，使它们成为检测DGA的理想候选者。我们的工作验证了微调的LLMS在检测DGA和DNS渗出攻击方面的有效性。我们开发了LLM模型，并使用包含59个不同的真实DGA恶意软件家族和正常域数据的不同数据集进行了全面评估。我们的LLM模型显著优于传统的自然语言处理技术，特别是在检测未知DGA方面。我们还评估了它在DNS渗出数据集上的性能，展示了它在加强网络安全措施方面的有效性。据我们所知，这是第一个经验性地将LLMS应用于DGA和DNS渗出检测的工作。



## **36. Bileve: Securing Text Provenance in Large Language Models Against Spoofing with Bi-level Signature**

Bileve：通过双层签名保护大型语言模型中的文本出处，防止欺骗 cs.CR

NeurIPS 2024 camera-ready

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2406.01946v3) [paper-pdf](http://arxiv.org/pdf/2406.01946v3)

**Authors**: Tong Zhou, Xuandong Zhao, Xiaolin Xu, Shaolei Ren

**Abstract**: Text watermarks for large language models (LLMs) have been commonly used to identify the origins of machine-generated content, which is promising for assessing liability when combating deepfake or harmful content. While existing watermarking techniques typically prioritize robustness against removal attacks, unfortunately, they are vulnerable to spoofing attacks: malicious actors can subtly alter the meanings of LLM-generated responses or even forge harmful content, potentially misattributing blame to the LLM developer. To overcome this, we introduce a bi-level signature scheme, Bileve, which embeds fine-grained signature bits for integrity checks (mitigating spoofing attacks) as well as a coarse-grained signal to trace text sources when the signature is invalid (enhancing detectability) via a novel rank-based sampling strategy. Compared to conventional watermark detectors that only output binary results, Bileve can differentiate 5 scenarios during detection, reliably tracing text provenance and regulating LLMs. The experiments conducted on OPT-1.3B and LLaMA-7B demonstrate the effectiveness of Bileve in defeating spoofing attacks with enhanced detectability. Code is available at https://github.com/Tongzhou0101/Bileve-official.

摘要: 大型语言模型(LLM)的文本水印通常用于识别机器生成内容的来源，这有望在打击深度虚假或有害内容时评估责任。虽然现有的水印技术通常将健壮性放在免受删除攻击的优先位置，但不幸的是，它们容易受到欺骗性攻击：恶意行为者可以巧妙地更改LLM生成的响应的含义，甚至伪造有害内容，可能会将责任错误地归咎于LLM开发人员。为了克服这一问题，我们提出了一种双层签名方案BiLEVE，该方案通过一种新颖的基于等级的采样策略嵌入细粒度的签名比特用于完整性检查(缓解欺骗攻击)，并在签名无效时嵌入粗粒度的信号来跟踪文本来源(增强了可检测性)。与传统的只输出二进制结果的水印检测器相比，BiLEVE在检测过程中可以区分5种场景，可靠地追踪文本来源和规范LLM。在OPT-1.3B和LLAMA-7B上进行的实验证明了BiLEVE在抵抗欺骗攻击方面的有效性，并增强了可检测性。代码可在https://github.com/Tongzhou0101/Bileve-official.上找到



## **37. CFSafety: Comprehensive Fine-grained Safety Assessment for LLMs**

CFSafety：LLM的全面细粒度安全评估 cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21695v1) [paper-pdf](http://arxiv.org/pdf/2410.21695v1)

**Authors**: Zhihao Liu, Chenhui Hu

**Abstract**: As large language models (LLMs) rapidly evolve, they bring significant conveniences to our work and daily lives, but also introduce considerable safety risks. These models can generate texts with social biases or unethical content, and under specific adversarial instructions, may even incite illegal activities. Therefore, rigorous safety assessments of LLMs are crucial. In this work, we introduce a safety assessment benchmark, CFSafety, which integrates 5 classic safety scenarios and 5 types of instruction attacks, totaling 10 categories of safety questions, to form a test set with 25k prompts. This test set was used to evaluate the natural language generation (NLG) capabilities of LLMs, employing a combination of simple moral judgment and a 1-5 safety rating scale for scoring. Using this benchmark, we tested eight popular LLMs, including the GPT series. The results indicate that while GPT-4 demonstrated superior safety performance, the safety effectiveness of LLMs, including this model, still requires improvement. The data and code associated with this study are available on GitHub.

摘要: 随着大型语言模型的快速发展，它们在给我们的工作和日常生活带来极大便利的同时，也带来了相当大的安全隐患。这些模式可能会产生带有社会偏见或不道德内容的文本，在特定的敌对指令下，甚至可能煽动非法活动。因此，对LLMS进行严格的安全评估至关重要。在这项工作中，我们引入了一个安全评估基准CFSafe，它集成了5个经典的安全场景和5种类型的指令攻击，共计10类安全问题，形成了一个包含25K提示的测试集。该测试集被用来评估LLMS的自然语言生成(NLG)能力，采用简单的道德判断和1-5安全等级评分相结合的方式进行评分。使用这个基准，我们测试了八个流行的LLM，包括GPT系列。结果表明，尽管GPT-4表现出了优越的安全性能，但包括该模型在内的LLMS的安全有效性仍需改进。与这项研究相关的数据和代码可在GitHub上获得。



## **38. FATH: Authentication-based Test-time Defense against Indirect Prompt Injection Attacks**

FASH：基于身份验证的测试时防御间接提示注入攻击 cs.CR

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21492v1) [paper-pdf](http://arxiv.org/pdf/2410.21492v1)

**Authors**: Jiongxiao Wang, Fangzhou Wu, Wendi Li, Jinsheng Pan, Edward Suh, Z. Morley Mao, Muhao Chen, Chaowei Xiao

**Abstract**: Large language models (LLMs) have been widely deployed as the backbone with additional tools and text information for real-world applications. However, integrating external information into LLM-integrated applications raises significant security concerns. Among these, prompt injection attacks are particularly threatening, where malicious instructions injected in the external text information can exploit LLMs to generate answers as the attackers desire. While both training-time and test-time defense methods have been developed to mitigate such attacks, the unaffordable training costs associated with training-time methods and the limited effectiveness of existing test-time methods make them impractical. This paper introduces a novel test-time defense strategy, named Formatting AuThentication with Hash-based tags (FATH). Unlike existing approaches that prevent LLMs from answering additional instructions in external text, our method implements an authentication system, requiring LLMs to answer all received instructions with a security policy and selectively filter out responses to user instructions as the final output. To achieve this, we utilize hash-based authentication tags to label each response, facilitating accurate identification of responses according to the user's instructions and improving the robustness against adaptive attacks. Comprehensive experiments demonstrate that our defense method can effectively defend against indirect prompt injection attacks, achieving state-of-the-art performance under Llama3 and GPT3.5 models across various attack methods. Our code is released at: https://github.com/Jayfeather1024/FATH

摘要: 大型语言模型(LLM)已被广泛部署为主干，并为实际应用程序提供额外的工具和文本信息。然而，将外部信息集成到LLM集成的应用程序中会引发重大的安全问题。其中，即时注入攻击尤其具有威胁性，在外部文本信息中注入的恶意指令可以利用LLMS生成攻击者想要的答案。虽然已经开发了训练时间和测试时间防御方法来缓解此类攻击，但与训练时间方法相关的难以负担的训练成本以及现有测试时间方法的有限有效性使它们变得不切实际。提出了一种新的测试时间防御策略--基于Hash标签的格式化认证(FATH)。与现有的阻止LLMS应答外部文本中的额外指令的方法不同，我们的方法实现了一个身份验证系统，要求LLMS用安全策略应答所有接收到的指令，并选择性地过滤对用户指令的响应作为最终输出。为了实现这一点，我们使用基于散列的身份验证标签来标记每个响应，便于根据用户的指令准确识别响应，并提高了对自适应攻击的健壮性。综合实验表明，我们的防御方法能够有效防御间接即时注入攻击，在Llama3和GPT3.5模型下通过各种攻击方法获得了最好的性能。我们的代码发布在：https://github.com/Jayfeather1024/FATH



## **39. Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning**

简单性盛行：重新思考LLM忘记学习的负偏好优化 cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.07163v2) [paper-pdf](http://arxiv.org/pdf/2410.07163v2)

**Authors**: Chongyu Fan, Jiancheng Liu, Licong Lin, Jinghan Jia, Ruiqi Zhang, Song Mei, Sijia Liu

**Abstract**: In this work, we address the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences and associated model capabilities (e.g., copyrighted data or harmful content generation) while preserving essential model utilities, without the need for retraining from scratch. Despite the growing need for LLM unlearning, a principled optimization framework remains lacking. To this end, we revisit the state-of-the-art approach, negative preference optimization (NPO), and identify the issue of reference model bias, which could undermine NPO's effectiveness, particularly when unlearning forget data of varying difficulty. Given that, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that 'simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We also provide deeper insights into SimNPO's advantages, supported by analysis using mixtures of Markov chains. Furthermore, we present extensive experiments validating SimNPO's superiority over existing unlearning baselines in benchmarks like TOFU and MUSE, and robustness against relearning attacks. Codes are available at https://github.com/OPTML-Group/Unlearn-Simple.

摘要: 在这项工作中，我们解决了大型语言模型(LLM)遗忘的问题，旨在消除不必要的数据影响和相关的模型能力(例如，受版权保护的数据或有害内容生成)，同时保留基本的模型实用程序，而不需要从头开始重新培训。尽管对LLM遗忘的需求越来越大，但仍然缺乏一个有原则的优化框架。为此，我们回顾了最先进的方法，负偏好优化(NPO)，并确定了参考模型偏差的问题，这可能会削弱NPO的有效性，特别是当遗忘遗忘数据的不同难度时。鉴于此，我们提出了一个简单而有效的遗忘优化框架，称为SimNPO，表明在消除对参考模型的依赖(通过简单偏好优化的镜头)时的“简单性”有利于遗忘。我们还提供了对SimNPO的优势的更深层次的见解，并通过使用马尔可夫链的混合分析提供了支持。此外，我们提供了大量的实验，验证了SimNPO在豆腐和缪斯等基准测试中相对于现有遗忘基线的优势，以及对重新学习攻击的健壮性。有关代码，请访问https://github.com/OPTML-Group/Unlearn-Simple.



## **40. Systematically Analyzing Prompt Injection Vulnerabilities in Diverse LLM Architectures**

系统分析不同LLM架构中的提示注入漏洞 cs.CR

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.23308v1) [paper-pdf](http://arxiv.org/pdf/2410.23308v1)

**Authors**: Victoria Benjamin, Emily Braca, Israel Carter, Hafsa Kanchwala, Nava Khojasteh, Charly Landow, Yi Luo, Caroline Ma, Anna Magarelli, Rachel Mirin, Avery Moyer, Kayla Simpson, Amelia Skawinski, Thomas Heverin

**Abstract**: This study systematically analyzes the vulnerability of 36 large language models (LLMs) to various prompt injection attacks, a technique that leverages carefully crafted prompts to elicit malicious LLM behavior. Across 144 prompt injection tests, we observed a strong correlation between model parameters and vulnerability, with statistical analyses, such as logistic regression and random forest feature analysis, indicating that parameter size and architecture significantly influence susceptibility. Results revealed that 56 percent of tests led to successful prompt injections, emphasizing widespread vulnerability across various parameter sizes, with clustering analysis identifying distinct vulnerability profiles associated with specific model configurations. Additionally, our analysis uncovered correlations between certain prompt injection techniques, suggesting potential overlaps in vulnerabilities. These findings underscore the urgent need for robust, multi-layered defenses in LLMs deployed across critical infrastructure and sensitive industries. Successful prompt injection attacks could result in severe consequences, including data breaches, unauthorized access, or misinformation. Future research should explore multilingual and multi-step defenses alongside adaptive mitigation strategies to strengthen LLM security in diverse, real-world environments.

摘要: 这项研究系统地分析了36个大型语言模型(LLM)对各种提示注入攻击的脆弱性，这是一种利用精心设计的提示来诱导恶意LLM行为的技术。在144次快速注入测试中，我们观察到模型参数与脆弱性之间存在很强的相关性，通过Logistic回归和随机森林特征分析等统计分析，表明参数大小和结构显著影响易感性。结果显示，56%的测试导致了成功的快速注入，强调了跨各种参数大小的广泛漏洞，通过聚类分析确定了与特定模型配置相关联的不同漏洞配置文件。此外，我们的分析发现了某些快速注入技术之间的关联，这表明漏洞中存在潜在的重叠。这些发现突显了在关键基础设施和敏感行业部署的低成本管理系统中迫切需要强大的多层防御。成功的即时注入攻击可能会导致严重后果，包括数据泄露、未经授权的访问或错误信息。未来的研究应该探索多语言和多步骤防御措施，以及自适应缓解战略，以加强不同现实环境中的LLM安全。



## **41. Securing Multi-turn Conversational Language Models From Distributed Backdoor Triggers**

保护多轮对话语言模型免受分布式后门触发器的影响 cs.CL

Findings of EMNLP 2024

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2407.04151v2) [paper-pdf](http://arxiv.org/pdf/2407.04151v2)

**Authors**: Terry Tong, Jiashu Xu, Qin Liu, Muhao Chen

**Abstract**: Large language models (LLMs) have acquired the ability to handle longer context lengths and understand nuances in text, expanding their dialogue capabilities beyond a single utterance. A popular user-facing application of LLMs is the multi-turn chat setting. Though longer chat memory and better understanding may seemingly benefit users, our paper exposes a vulnerability that leverages the multi-turn feature and strong learning ability of LLMs to harm the end-user: the backdoor. We demonstrate that LLMs can capture the combinational backdoor representation. Only upon presentation of triggers together does the backdoor activate. We also verify empirically that this representation is invariant to the position of the trigger utterance. Subsequently, inserting a single extra token into two utterances of 5%of the data can cause over 99% Attack Success Rate (ASR). Our results with 3 triggers demonstrate that this framework is generalizable, compatible with any trigger in an adversary's toolbox in a plug-and-play manner. Defending the backdoor can be challenging in the chat setting because of the large input and output space. Our analysis indicates that the distributed backdoor exacerbates the current challenges by polynomially increasing the dimension of the attacked input space. Canonical textual defenses like ONION and BKI leverage auxiliary model forward passes over individual tokens, scaling exponentially with the input sequence length and struggling to maintain computational feasibility. To this end, we propose a decoding time defense - decayed contrastive decoding - that scales linearly with assistant response sequence length and reduces the backdoor to as low as 0.35%.

摘要: 大型语言模型(LLM)已经具备了处理更长的上下文长度和理解文本中的细微差别的能力，将它们的对话能力扩展到了单一话语之外。LLMS的一个流行的面向用户的应用是多轮聊天设置。虽然更长的聊天记忆和更好的理解似乎对用户有利，但我们的论文暴露了一个漏洞，该漏洞利用LLMS的多回合功能和强大的学习能力来伤害最终用户：后门。我们证明了LLMS能够捕获组合后门表示。只有在一起显示触发器时，后门才会激活。我们还通过实验验证了该表示与触发话语的位置不变。随后，在两个5%的数据发声中插入一个额外令牌可以导致超过99%的攻击成功率(ASR)。我们对3个触发器的测试结果表明，该框架具有通用性，以即插即用的方式兼容对手工具箱中的任何触发器。在聊天环境中，由于输入和输出空间很大，保护后门可能是一件具有挑战性的事情。我们的分析表明，分布式后门通过以多项式增加被攻击输入空间的维度来加剧当前的挑战。像Onion和BKI这样的规范文本防御机制利用辅助模型向前传递单个令牌，随着输入序列长度呈指数级扩展，并努力保持计算的可行性。为此，我们提出了一种译码时间防御机制--衰落对比译码，它与辅助响应序列长度成线性关系，并将后门降低到0.35%。



## **42. AutoPenBench: Benchmarking Generative Agents for Penetration Testing**

AutoPenBench：渗透测试生成剂的基准测试 cs.CR

Codes for the benchmark:  https://github.com/lucagioacchini/auto-pen-bench Codes for the paper  experiments: https://github.com/lucagioacchini/genai-pentest-paper

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.03225v2) [paper-pdf](http://arxiv.org/pdf/2410.03225v2)

**Authors**: Luca Gioacchini, Marco Mellia, Idilio Drago, Alexander Delsanto, Giuseppe Siracusano, Roberto Bifulco

**Abstract**: Generative AI agents, software systems powered by Large Language Models (LLMs), are emerging as a promising approach to automate cybersecurity tasks. Among the others, penetration testing is a challenging field due to the task complexity and the diverse strategies to simulate cyber-attacks. Despite growing interest and initial studies in automating penetration testing with generative agents, there remains a significant gap in the form of a comprehensive and standard framework for their evaluation and development. This paper introduces AutoPenBench, an open benchmark for evaluating generative agents in automated penetration testing. We present a comprehensive framework that includes 33 tasks, each representing a vulnerable system that the agent has to attack. Tasks are of increasing difficulty levels, including in-vitro and real-world scenarios. We assess the agent performance with generic and specific milestones that allow us to compare results in a standardised manner and understand the limits of the agent under test. We show the benefits of AutoPenBench by testing two agent architectures: a fully autonomous and a semi-autonomous supporting human interaction. We compare their performance and limitations. For example, the fully autonomous agent performs unsatisfactorily achieving a 21% Success Rate (SR) across the benchmark, solving 27% of the simple tasks and only one real-world task. In contrast, the assisted agent demonstrates substantial improvements, with 64% of SR. AutoPenBench allows us also to observe how different LLMs like GPT-4o or OpenAI o1 impact the ability of the agents to complete the tasks. We believe that our benchmark fills the gap with a standard and flexible framework to compare penetration testing agents on a common ground. We hope to extend AutoPenBench along with the research community by making it available under https://github.com/lucagioacchini/auto-pen-bench.

摘要: 生成式人工智能代理是由大型语言模型(LLM)支持的软件系统，正在成为一种有前途的自动化网络安全任务的方法。其中，渗透测试是一个具有挑战性的领域，因为任务的复杂性和模拟网络攻击的策略多种多样。尽管人们对利用产生剂进行自动化渗透测试越来越感兴趣，并进行了初步研究，但在评估和开发产生剂的全面和标准框架的形式上，仍然存在着重大差距。本文介绍了一种用于评估自动渗透测试中的生成性代理的开放基准--AutoPenBch。我们提出了一个全面的框架，包括33个任务，每个任务代表代理必须攻击的易受攻击的系统。任务的难度越来越高，包括体外和真实世界的场景。我们用通用的和特定的里程碑来评估代理的性能，使我们能够以标准化的方式比较结果，并了解接受测试的代理的限制。我们通过测试两种代理体系结构：完全自主和半自主支持人类交互，展示了AutoPenB边的好处。我们比较了它们的性能和局限性。例如，完全自主的代理在基准测试中的成功率(SR)不令人满意地达到了21%，解决了27%的简单任务，而只解决了一个真实世界的任务。相比之下，辅助剂表现出显著的改善，获得了SR的5%。AutoPenB边还允许我们观察不同的LLM，如GPT-40或OpenAI o1，是如何影响代理完成任务的能力的。我们相信，我们的基准填补了这一空白，提供了一个标准和灵活的框架，可以在共同的基础上比较渗透测试试剂。我们希望通过使其在https://github.com/lucagioacchini/auto-pen-bench.下可用来与研究社区一起扩展AutoPenB边



## **43. Palisade -- Prompt Injection Detection Framework**

Palisade --提示注入检测框架 cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21146v1) [paper-pdf](http://arxiv.org/pdf/2410.21146v1)

**Authors**: Sahasra Kokkula, Somanathan R, Nandavardhan R, Aashishkumar, G Divya

**Abstract**: The advent of Large Language Models LLMs marks a milestone in Artificial Intelligence, altering how machines comprehend and generate human language. However, LLMs are vulnerable to malicious prompt injection attacks, where crafted inputs manipulate the models behavior in unintended ways, compromising system integrity and causing incorrect outcomes. Conventional detection methods rely on static, rule-based approaches, which often fail against sophisticated threats like abnormal token sequences and alias substitutions, leading to limited adaptability and higher rates of false positives and false negatives.This paper proposes a novel NLP based approach for prompt injection detection, emphasizing accuracy and optimization through a layered input screening process. In this framework, prompts are filtered through three distinct layers rule-based, ML classifier, and companion LLM before reaching the target model, thereby minimizing the risk of malicious interaction.Tests show the ML classifier achieves the highest accuracy among individual layers, yet the multi-layer framework enhances overall detection accuracy by reducing false negatives. Although this increases false positives, it minimizes the risk of overlooking genuine injected prompts, thus prioritizing security.This multi-layered detection approach highlights LLM vulnerabilities and provides a comprehensive framework for future research, promoting secure interactions between humans and AI systems.

摘要: 大型语言模型LLMS的出现标志着人工智能的一个里程碑，改变了机器理解和生成人类语言的方式。然而，LLM容易受到恶意提示注入攻击，在恶意提示注入攻击中，精心编制的输入以意外的方式操纵模型行为，损害系统完整性并导致不正确的结果。针对传统的检测方法依赖于静态的、基于规则的方法，往往无法抵抗令牌序列异常、别名替换等复杂威胁，导致适应性有限，误报和漏检率较高，提出了一种新的基于自然语言处理的快速注入检测方法，通过分层的输入筛选过程强调准确性和优化。在该框架中，提示在到达目标模型之前通过基于规则、ML分类器和伴随LLM三个不同的层进行过滤，从而将恶意交互的风险降到最低；测试表明，ML分类器在各个层中达到了最高的准确率，而多层框架通过减少漏报来提高整体检测的准确率。虽然这会增加误报，但它将忽略真实注入提示的风险降至最低，从而将安全放在首位。这种多层检测方法突出了LLM漏洞，并为未来的研究提供了一个全面的框架，促进了人类与AI系统之间的安全交互。



## **44. Stealthy Jailbreak Attacks on Large Language Models via Benign Data Mirroring**

通过良性数据镜像对大型语言模型进行秘密越狱攻击 cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21083v1) [paper-pdf](http://arxiv.org/pdf/2410.21083v1)

**Authors**: Honglin Mu, Han He, Yuxin Zhou, Yunlong Feng, Yang Xu, Libo Qin, Xiaoming Shi, Zeming Liu, Xudong Han, Qi Shi, Qingfu Zhu, Wanxiang Che

**Abstract**: Large language model (LLM) safety is a critical issue, with numerous studies employing red team testing to enhance model security. Among these, jailbreak methods explore potential vulnerabilities by crafting malicious prompts that induce model outputs contrary to safety alignments. Existing black-box jailbreak methods often rely on model feedback, repeatedly submitting queries with detectable malicious instructions during the attack search process. Although these approaches are effective, the attacks may be intercepted by content moderators during the search process. We propose an improved transfer attack method that guides malicious prompt construction by locally training a mirror model of the target black-box model through benign data distillation. This method offers enhanced stealth, as it does not involve submitting identifiable malicious instructions to the target model during the search phase. Our approach achieved a maximum attack success rate of 92%, or a balanced value of 80% with an average of 1.5 detectable jailbreak queries per sample against GPT-3.5 Turbo on a subset of AdvBench. These results underscore the need for more robust defense mechanisms.

摘要: 大型语言模型(LLM)的安全性是一个关键问题，许多研究使用RED团队测试来增强模型的安全性。其中，越狱方法通过精心编制恶意提示来探测潜在的漏洞，这些提示会诱导与安全对齐相反的模型输出。现有的黑盒越狱方法通常依赖于模型反馈，在攻击搜索过程中反复提交带有可检测到的恶意指令的查询。虽然这些方法是有效的，但这些攻击可能会在搜索过程中被内容版主拦截。提出了一种改进的传输攻击方法，该方法通过良性数据提炼对目标黑盒模型的镜像模型进行局部训练，指导恶意提示的构建。这种方法提供了增强的隐蔽性，因为它不涉及在搜索阶段向目标模型提交可识别的恶意指令。我们的方法获得了92%的最大攻击成功率，或者说80%的平衡值，每个样本平均有1.5个可检测到的越狱查询，而GPT-3.5Turbo在AdvBitch子集上的攻击成功率为92%。这些结果突显了需要更强大的防御机制。



## **45. Attacking Misinformation Detection Using Adversarial Examples Generated by Language Models**

使用语言模型生成的对抗性示例进行攻击错误信息检测 cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.20940v1) [paper-pdf](http://arxiv.org/pdf/2410.20940v1)

**Authors**: Piotr Przybyła

**Abstract**: We investigate the challenge of generating adversarial examples to test the robustness of text classification algorithms detecting low-credibility content, including propaganda, false claims, rumours and hyperpartisan news. We focus on simulation of content moderation by setting realistic limits on the number of queries an attacker is allowed to attempt. Within our solution (TREPAT), initial rephrasings are generated by large language models with prompts inspired by meaning-preserving NLP tasks, e.g. text simplification and style transfer. Subsequently, these modifications are decomposed into small changes, applied through beam search procedure until the victim classifier changes its decision. The evaluation confirms the superiority of our approach in the constrained scenario, especially in case of long input text (news articles), where exhaustive search is not feasible.

摘要: 我们调查了生成敌对示例的挑战，以测试检测低可信度内容（包括宣传、虚假声明、谣言和超党派新闻）的文本分类算法的稳健性。我们通过对允许攻击者尝试的查询数量设置现实的限制来重点模拟内容审核。在我们的解决方案（TREPAT）中，初始改写由大型语言模型生成，其提示受到保留意义的NLP任务（例如文本简化和风格转移）的启发。随后，这些修改被分解成小的变化，通过束搜索过程应用，直到受害者分类器改变其决定。评估证实了我们的方法在受约束的情况下的优越性，特别是在长输入文本（新闻文章）的情况下，其中详尽搜索是不可行的。



## **46. Hacking Back the AI-Hacker: Prompt Injection as a Defense Against LLM-driven Cyberattacks**

黑客攻击人工智能黑客：即时注入作为抵御LLM驱动的网络攻击的防御 cs.CR

v0.1

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.20911v1) [paper-pdf](http://arxiv.org/pdf/2410.20911v1)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: Large language models (LLMs) are increasingly being harnessed to automate cyberattacks, making sophisticated exploits more accessible and scalable. In response, we propose a new defense strategy tailored to counter LLM-driven cyberattacks. We introduce Mantis, a defensive framework that exploits LLMs' susceptibility to adversarial inputs to undermine malicious operations. Upon detecting an automated cyberattack, Mantis plants carefully crafted inputs into system responses, leading the attacker's LLM to disrupt their own operations (passive defense) or even compromise the attacker's machine (active defense). By deploying purposefully vulnerable decoy services to attract the attacker and using dynamic prompt injections for the attacker's LLM, Mantis can autonomously hack back the attacker. In our experiments, Mantis consistently achieved over 95% effectiveness against automated LLM-driven attacks. To foster further research and collaboration, Mantis is available as an open-source tool: https://github.com/pasquini-dario/project_mantis

摘要: 大型语言模型(LLM)越来越多地被用来自动化网络攻击，使复杂的利用更容易获得和可扩展。作为回应，我们提出了一种新的防御战略，以对抗LLM驱动的网络攻击。我们引入了Mantis，这是一个防御框架，利用LLMS对对手输入的敏感性来破坏恶意操作。在检测到自动网络攻击后，螳螂工厂会精心设计输入到系统响应中，导致攻击者的LLM扰乱自己的操作(被动防御)，甚至危害攻击者的机器(主动防御)。通过部署故意易受攻击的诱骗服务来吸引攻击者，并对攻击者的LLM使用动态提示注入，螳螂可以自主地攻击攻击者。在我们的实验中，螳螂对自动LLM驱动的攻击始终取得了95%以上的效率。为了促进进一步的研究和合作，Mantis以开源工具的形式提供：https://github.com/pasquini-dario/project_mantis



## **47. Uncovering Safety Risks of Large Language Models through Concept Activation Vector**

通过概念激活载体揭示大型语言模型的安全风险 cs.CL

10 pages, accepted as a poster at NeurIPS 2024

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2404.12038v4) [paper-pdf](http://arxiv.org/pdf/2404.12038v4)

**Authors**: Zhihao Xu, Ruixuan Huang, Changyu Chen, Xiting Wang

**Abstract**: Despite careful safety alignment, current large language models (LLMs) remain vulnerable to various attacks. To further unveil the safety risks of LLMs, we introduce a Safety Concept Activation Vector (SCAV) framework, which effectively guides the attacks by accurately interpreting LLMs' safety mechanisms. We then develop an SCAV-guided attack method that can generate both attack prompts and embedding-level attacks with automatically selected perturbation hyperparameters. Both automatic and human evaluations demonstrate that our attack method significantly improves the attack success rate and response quality while requiring less training data. Additionally, we find that our generated attack prompts may be transferable to GPT-4, and the embedding-level attacks may also be transferred to other white-box LLMs whose parameters are known. Our experiments further uncover the safety risks present in current LLMs. For example, in our evaluation of seven open-source LLMs, we observe an average attack success rate of 99.14%, based on the classic keyword-matching criterion. Finally, we provide insights into the safety mechanism of LLMs. The code is available at https://github.com/SproutNan/AI-Safety_SCAV.

摘要: 尽管进行了仔细的安全调整，但当前的大型语言模型(LLM)仍然容易受到各种攻击。为了进一步揭示LLMS的安全隐患，我们引入了安全概念激活向量(SCAV)框架，通过准确解释LLMS的安全机制来有效地指导攻击。然后，我们开发了一种SCAV引导的攻击方法，该方法可以生成攻击提示和带有自动选择的扰动超参数的嵌入级攻击。自动和人工评估都表明，我们的攻击方法在需要更少的训练数据的情况下，显著地提高了攻击成功率和响应质量。此外，我们发现我们生成的攻击提示可以转移到GPT-4上，嵌入级攻击也可以转移到参数已知的其他白盒LLM上。我们的实验进一步揭示了当前LLM中存在的安全风险。例如，在我们对7个开源LLM的评估中，基于经典的关键字匹配标准，我们观察到平均攻击成功率为99.14%。最后，我们对LLMS的安全机制提供了见解。代码可在https://github.com/SproutNan/AI-Safety_SCAV.上获得



## **48. Fine-tuned Large Language Models (LLMs): Improved Prompt Injection Attacks Detection**

微调的大型语言模型（LLM）：改进的提示注入攻击检测 cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21337v1) [paper-pdf](http://arxiv.org/pdf/2410.21337v1)

**Authors**: Md Abdur Rahman, Fan Wu, Alfredo Cuzzocrea, Sheikh Iqbal Ahamed

**Abstract**: Large language models (LLMs) are becoming a popular tool as they have significantly advanced in their capability to tackle a wide range of language-based tasks. However, LLMs applications are highly vulnerable to prompt injection attacks, which poses a critical problem. These attacks target LLMs applications through using carefully designed input prompts to divert the model from adhering to original instruction, thereby it could execute unintended actions. These manipulations pose serious security threats which potentially results in data leaks, biased outputs, or harmful responses. This project explores the security vulnerabilities in relation to prompt injection attacks. To detect whether a prompt is vulnerable or not, we follows two approaches: 1) a pre-trained LLM, and 2) a fine-tuned LLM. Then, we conduct a thorough analysis and comparison of the classification performance. Firstly, we use pre-trained XLM-RoBERTa model to detect prompt injections using test dataset without any fine-tuning and evaluate it by zero-shot classification. Then, this proposed work will apply supervised fine-tuning to this pre-trained LLM using a task-specific labeled dataset from deepset in huggingface, and this fine-tuned model achieves impressive results with 99.13\% accuracy, 100\% precision, 98.33\% recall and 99.15\% F1-score thorough rigorous experimentation and evaluation. We observe that our approach is highly efficient in detecting prompt injection attacks.

摘要: 大型语言模型(LLM)正在成为一种流行的工具，因为它们在处理各种基于语言的任务的能力方面有了显著的进步。然而，LLMS应用程序很容易受到即时注入攻击，这是一个严重的问题。这些攻击通过使用精心设计的输入提示来转移模型对原始指令的依赖，从而针对LLMS应用程序，从而可以执行意外的操作。这些操作构成了严重的安全威胁，可能会导致数据泄露、有偏见的输出或有害的响应。该项目探索与提示注入攻击相关的安全漏洞。为了检测提示符是否易受攻击，我们采用了两种方法：1)预先训练的LLM和2)微调的LLM。然后，我们对分类性能进行了深入的分析和比较。首先，我们使用预先训练好的XLM-Roberta模型，在没有任何微调的测试数据集上检测快速注射，并用零镜头分类对其进行评估。然后，该工作将使用来自拥抱脸深度集的特定任务的标签数据集对该预训练的LLM进行有监督的微调，该微调模型取得了令人印象深刻的结果，其准确率为99.13，准确率为100，召回率为98.33，F1-Score为99.15。我们观察到我们的方法在检测即时注入攻击方面是非常有效的。



## **49. LLM Robustness Against Misinformation in Biomedical Question Answering**

LLM生物医学问题回答中针对错误信息的稳健性 cs.CL

**SubmitDate**: 2024-10-27    [abs](http://arxiv.org/abs/2410.21330v1) [paper-pdf](http://arxiv.org/pdf/2410.21330v1)

**Authors**: Alexander Bondarenko, Adrian Viehweger

**Abstract**: The retrieval-augmented generation (RAG) approach is used to reduce the confabulation of large language models (LLMs) for question answering by retrieving and providing additional context coming from external knowledge sources (e.g., by adding the context to the prompt). However, injecting incorrect information can mislead the LLM to generate an incorrect answer.   In this paper, we evaluate the effectiveness and robustness of four LLMs against misinformation - Gemma 2, GPT-4o-mini, Llama~3.1, and Mixtral - in answering biomedical questions. We assess the answer accuracy on yes-no and free-form questions in three scenarios: vanilla LLM answers (no context is provided), "perfect" augmented generation (correct context is provided), and prompt-injection attacks (incorrect context is provided). Our results show that Llama 3.1 (70B parameters) achieves the highest accuracy in both vanilla (0.651) and "perfect" RAG (0.802) scenarios. However, the accuracy gap between the models almost disappears with "perfect" RAG, suggesting its potential to mitigate the LLM's size-related effectiveness differences.   We further evaluate the ability of the LLMs to generate malicious context on one hand and the LLM's robustness against prompt-injection attacks on the other hand, using metrics such as attack success rate (ASR), accuracy under attack, and accuracy drop. As adversaries, we use the same four LLMs (Gemma 2, GPT-4o-mini, Llama 3.1, and Mixtral) to generate incorrect context that is injected in the target model's prompt. Interestingly, Llama is shown to be the most effective adversary, causing accuracy drops of up to 0.48 for vanilla answers and 0.63 for "perfect" RAG across target models. Our analysis reveals that robustness rankings vary depending on the evaluation measure, highlighting the complexity of assessing LLM resilience to adversarial attacks.

摘要: 检索-增强生成(RAG)方法用于通过检索和提供来自外部知识源的附加上下文(例如，通过将上下文添加到提示)来减少大语言模型(LLM)对问题回答的虚构。然而，注入错误的信息可能会误导LLM生成错误的答案。在这篇文章中，我们评估了四个针对错误信息的最小二乘模型-Gema2、GPT-40-mini、Llama~3.1和Mixtral-在回答生物医学问题时的有效性和稳健性。我们在三个场景中评估了是-否和自由形式问题的答案准确率：普通LLM答案(没有提供上下文)、“完美”增强生成(提供了正确的上下文)和提示注入攻击(提供了错误的上下文)。我们的结果表明，Llama3.1(70B参数)在Vanilla(0.651)和“Perfect”RAG(0.802)场景中都达到了最高的精度。然而，随着RAG的“完美”，两个模型之间的精度差距几乎消失了，这表明它有可能缓解LLM与大小相关的有效性差异。我们使用攻击成功率(ASR)、攻击下的准确率和准确率下降等指标，进一步评估了LLM一方面产生恶意上下文的能力，另一方面LLM对即时注入攻击的健壮性。作为对手，我们使用相同的四个LLM(Gema2、GPT-4o-mini、Llama3.1和Mixtral)来生成错误的上下文，该上下文被注入到目标模型的提示符中。有趣的是，骆驼被证明是最有效的对手，在目标模型上，导致普通答案的准确率下降高达0.48，而“完美”RAG的准确率下降0.63。我们的分析表明，健壮性排名根据评估措施的不同而不同，这突显了评估LLM对对手攻击的弹性的复杂性。



## **50. PANORAMIA: Privacy Auditing of Machine Learning Models without Retraining**

PANORAMIA：无需重新训练的机器学习模型隐私审计 cs.CR

36 pages

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2402.09477v2) [paper-pdf](http://arxiv.org/pdf/2402.09477v2)

**Authors**: Mishaal Kazmi, Hadrien Lautraite, Alireza Akbari, Qiaoyue Tang, Mauricio Soroco, Tao Wang, Sébastien Gambs, Mathias Lécuyer

**Abstract**: We present PANORAMIA, a privacy leakage measurement framework for machine learning models that relies on membership inference attacks using generated data as non-members. By relying on generated non-member data, PANORAMIA eliminates the common dependency of privacy measurement tools on in-distribution non-member data. As a result, PANORAMIA does not modify the model, training data, or training process, and only requires access to a subset of the training data. We evaluate PANORAMIA on ML models for image and tabular data classification, as well as on large-scale language models.

摘要: 我们介绍了PANORAMIA，这是一个用于机器学习模型的隐私泄露测量框架，该模型依赖于使用生成的数据作为非成员的成员推断攻击。通过依赖生成的非会员数据，PANORAMIA消除了隐私测量工具对分发内非会员数据的常见依赖性。因此，PANORAMIA不会修改模型、训练数据或训练过程，仅需要访问训练数据的子集。我们在图像和表格数据分类的ML模型以及大规模语言模型上评估了PANORAMIA。



