# Latest Large Language Model Attack Papers
**update at 2024-04-28 10:28:27**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Don't Say No: Jailbreaking LLM by Suppressing Refusal**

不要说不：通过压制拒绝来越狱法学硕士 cs.CL

**SubmitDate**: 2024-04-25    [abs](http://arxiv.org/abs/2404.16369v1) [paper-pdf](http://arxiv.org/pdf/2404.16369v1)

**Authors**: Yukai Zhou, Wenjie Wang

**Abstract**: Ensuring the safety alignment of Large Language Models (LLMs) is crucial to generating responses consistent with human values. Despite their ability to recognize and avoid harmful queries, LLMs are vulnerable to "jailbreaking" attacks, where carefully crafted prompts elicit them to produce toxic content. One category of jailbreak attacks is reformulating the task as adversarial attacks by eliciting the LLM to generate an affirmative response. However, the typical attack in this category GCG has very limited attack success rate. In this study, to better study the jailbreak attack, we introduce the DSN (Don't Say No) attack, which prompts LLMs to not only generate affirmative responses but also novelly enhance the objective to suppress refusals. In addition, another challenge lies in jailbreak attacks is the evaluation, as it is difficult to directly and accurately assess the harmfulness of the attack. The existing evaluation such as refusal keyword matching has its own limitation as it reveals numerous false positive and false negative instances. To overcome this challenge, we propose an ensemble evaluation pipeline incorporating Natural Language Inference (NLI) contradiction assessment and two external LLM evaluators. Extensive experiments demonstrate the potency of the DSN and the effectiveness of ensemble evaluation compared to baseline methods.

摘要: 确保大型语言模型(LLM)的安全一致性对于生成与人类价值观一致的响应至关重要。尽管LLM能够识别和避免有害的查询，但它们很容易受到“越狱”攻击，在这种攻击中，精心制作的提示会诱使它们产生有毒内容。越狱攻击的一类是通过诱使LLM产生肯定的回应，将任务重新制定为对抗性攻击。然而，在这类典型攻击中，GCG的攻击成功率非常有限。在本研究中，为了更好地研究越狱攻击，我们引入了DSN(Don‘t Say No)攻击，它不仅促使LLMS产生肯定的反应，而且新颖地增强了抑制拒绝的目标。此外，越狱攻击的另一个挑战是评估，因为很难直接和准确地评估攻击的危害性。现有的拒绝关键词匹配等评价方法暴露出大量的误报和漏报实例，具有一定的局限性。为了克服这一挑战，我们提出了一种集成评估流水线，其中包括自然语言推理(NLI)矛盾评估和两个外部LLM评估器。大量实验证明了DSN的有效性和集成评估与基线方法相比的有效性。



## **2. Investigating the prompt leakage effect and black-box defenses for multi-turn LLM interactions**

调查多圈LLM交互的即时泄漏效应和黑匣子防御 cs.CR

**SubmitDate**: 2024-04-24    [abs](http://arxiv.org/abs/2404.16251v1) [paper-pdf](http://arxiv.org/pdf/2404.16251v1)

**Authors**: Divyansh Agarwal, Alexander R. Fabbri, Philippe Laban, Shafiq Joty, Caiming Xiong, Chien-Sheng Wu

**Abstract**: Prompt leakage in large language models (LLMs) poses a significant security and privacy threat, particularly in retrieval-augmented generation (RAG) systems. However, leakage in multi-turn LLM interactions along with mitigation strategies has not been studied in a standardized manner. This paper investigates LLM vulnerabilities against prompt leakage across 4 diverse domains and 10 closed- and open-source LLMs. Our unique multi-turn threat model leverages the LLM's sycophancy effect and our analysis dissects task instruction and knowledge leakage in the LLM response. In a multi-turn setting, our threat model elevates the average attack success rate (ASR) to 86.2%, including a 99% leakage with GPT-4 and claude-1.3. We find that some black-box LLMs like Gemini show variable susceptibility to leakage across domains - they are more likely to leak contextual knowledge in the news domain compared to the medical domain. Our experiments measure specific effects of 6 black-box defense strategies, including a query-rewriter in the RAG scenario. Our proposed multi-tier combination of defenses still has an ASR of 5.3% for black-box LLMs, indicating room for enhancement and future direction for LLM security research.

摘要: 大型语言模型中的即时泄漏对安全和隐私构成了严重威胁，尤其是在检索-增强生成(RAG)系统中。然而，多圈LLM相互作用中的泄漏以及缓解策略还没有以标准化的方式进行研究。本文研究了4个不同的域和10个封闭和开源的LLM针对即时泄漏的LLM漏洞。我们独特的多回合威胁模型利用了LLM的奉承效应，我们的分析剖析了LLM响应中的任务指令和知识泄漏。在多回合设置中，我们的威胁模型将平均攻击成功率(ASR)提高到86.2%，其中GPT-4和Claude-1.3的泄漏率为99%。我们发现，一些像双子座这样的黑盒LLM对跨域泄漏表现出可变的敏感性--与医疗领域相比，他们更有可能在新闻领域泄露上下文知识。我们的实验测量了6种黑盒防御策略的具体效果，其中包括RAG场景中的查询重写器。我们建议的多层防御组合对于黑盒LLM的ASR仍为5.3%，这表明LLM安全研究有增强的空间和未来的方向。



## **3. Talk Too Much: Poisoning Large Language Models under Token Limit**

话太多：代币限制下的大型语言模型中毒 cs.CL

**SubmitDate**: 2024-04-24    [abs](http://arxiv.org/abs/2404.14795v2) [paper-pdf](http://arxiv.org/pdf/2404.14795v2)

**Authors**: Jiaming He, Wenbo Jiang, Guanyu Hou, Wenshu Fan, Rui Zhang, Hongwei Li

**Abstract**: Mainstream poisoning attacks on large language models (LLMs) typically set a fixed trigger in the input instance and specific responses for triggered queries. However, the fixed trigger setting (e.g., unusual words) may be easily detected by human detection, limiting the effectiveness and practicality in real-world scenarios. To enhance the stealthiness of the trigger, we present a poisoning attack against LLMs that is triggered by a generation/output condition-token limitation, which is a commonly adopted strategy by users for reducing costs. The poisoned model performs normally for output without token limitation, while becomes harmful for output with limited tokens. To achieve this objective, we introduce BrieFool, an efficient attack framework. It leverages the characteristics of generation limitation by efficient instruction sampling and poisoning data generation, thereby influencing the behavior of LLMs under target conditions. Our experiments demonstrate that BrieFool is effective across safety domains and knowledge domains. For instance, with only 20 generated poisoning examples against GPT-3.5-turbo, BrieFool achieves a 100% Attack Success Rate (ASR) and a 9.28/10 average Harmfulness Score (HS) under token limitation conditions while maintaining the benign performance.

摘要: 针对大型语言模型(LLM)的主流中毒攻击通常会在输入实例中设置固定的触发器，并为触发的查询设置特定的响应。然而，固定的触发设置(例如，不寻常的单词)可能很容易被人类检测到，从而限制了在现实世界场景中的有效性和实用性。为了增强触发器的隐蔽性，我们提出了一种由生成/输出条件-令牌限制触发的针对LLMS的中毒攻击，这是用户为降低成本而常用的策略。对于没有令牌限制的输出，中毒模型执行正常，而对于具有有限令牌的输出则变得有害。为了实现这一目标，我们引入了一种高效的攻击框架BrieFool。它通过高效的指令采样和中毒数据生成来利用生成限制的特性，从而影响LLMS在目标条件下的行为。我们的实验表明，BrieFool是跨安全域和知识域的有效的。例如，在对GPT-3.5-Turbo仅生成20个中毒实例的情况下，BrieFool在保持良性性能的同时，在令牌限制条件下实现了100%的攻击成功率(ASR)和9.28/10的平均危害性评分(HS)。



## **4. Large Language Models Spot Phishing Emails with Surprising Accuracy: A Comparative Analysis of Performance**

大型语言模型以惊人的准确性发现网络钓鱼电子邮件：性能比较分析 cs.CL

7 pages, 3 figures

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.15485v1) [paper-pdf](http://arxiv.org/pdf/2404.15485v1)

**Authors**: Het Patel, Umair Rehman, Farkhund Iqbal

**Abstract**: Phishing, a prevalent cybercrime tactic for decades, remains a significant threat in today's digital world. By leveraging clever social engineering elements and modern technology, cybercrime targets many individuals, businesses, and organizations to exploit trust and security. These cyber-attackers are often disguised in many trustworthy forms to appear as legitimate sources. By cleverly using psychological elements like urgency, fear, social proof, and other manipulative strategies, phishers can lure individuals into revealing sensitive and personalized information. Building on this pervasive issue within modern technology, this paper aims to analyze the effectiveness of 15 Large Language Models (LLMs) in detecting phishing attempts, specifically focusing on a randomized set of "419 Scam" emails. The objective is to determine which LLMs can accurately detect phishing emails by analyzing a text file containing email metadata based on predefined criteria. The experiment concluded that the following models, ChatGPT 3.5, GPT-3.5-Turbo-Instruct, and ChatGPT, were the most effective in detecting phishing emails.

摘要: 网络钓鱼是几十年来流行的一种网络犯罪策略，在当今的数字世界中仍然是一个重大威胁。通过利用聪明的社会工程元素和现代技术，网络犯罪以许多个人、企业和组织为目标，以利用信任和安全。这些网络攻击者往往以许多可信的形式伪装成合法的来源。通过巧妙地使用紧急、恐惧、社会证明和其他操纵策略等心理因素，网络钓鱼者可以诱使个人泄露敏感和个性化的信息。基于这一现代技术中普遍存在的问题，本文旨在分析15个大型语言模型(LLM)在检测网络钓鱼尝试方面的有效性，特别是关注一组随机的“419骗局”电子邮件。目标是通过基于预定义标准分析包含电子邮件元数据的文本文件，确定哪些LLM可以准确检测钓鱼电子邮件。实验得出的结论是，ChatGPT 3.5、GPT-3.5-Turbo-Indict和ChatGPT模型在检测钓鱼电子邮件方面最有效。



## **5. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

越狱长凳：越狱大型语言模型的开放鲁棒性基准 cs.CR

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.01318v2) [paper-pdf](http://arxiv.org/pdf/2404.01318v2)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work -- which align with OpenAI's usage policies; (3) a standardized evaluation framework that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community. Over time, we will expand and adapt the benchmark to reflect technical and methodological advances in the research community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak Bch，这是一款开源基准测试，具有以下组件：(1)不断发展的最新对抗性提示存储库，我们称之为越狱人工产物；(2)包含100种行为的越狱数据集，包括原始行为和源自先前工作的行为，这些行为与OpenAI的使用策略保持一致；(3)标准化评估框架，其中包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)跟踪各种LLM攻击和防御性能的排行榜。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。随着时间的推移，我们将扩大和调整基准，以反映研究界的技术和方法进步。



## **6. Versatile Backdoor Attack with Visible, Semantic, Sample-Specific, and Compatible Triggers**

具有可见、语义、样本特定且兼容触发器的多功能后门攻击 cs.CV

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2306.00816v3) [paper-pdf](http://arxiv.org/pdf/2306.00816v3)

**Authors**: Ruotong Wang, Hongrui Chen, Zihao Zhu, Li Liu, Baoyuan Wu

**Abstract**: Deep neural networks (DNNs) can be manipulated to exhibit specific behaviors when exposed to specific trigger patterns, without affecting their performance on benign samples, dubbed \textit{backdoor attack}. Currently, implementing backdoor attacks in physical scenarios still faces significant challenges. Physical attacks are labor-intensive and time-consuming, and the triggers are selected in a manual and heuristic way. Moreover, expanding digital attacks to physical scenarios faces many challenges due to their sensitivity to visual distortions and the absence of counterparts in the real world. To address these challenges, we define a novel trigger called the \textbf{V}isible, \textbf{S}emantic, \textbf{S}ample-Specific, and \textbf{C}ompatible (VSSC) trigger, to achieve effective, stealthy and robust simultaneously, which can also be effectively deployed in the physical scenario using corresponding objects. To implement the VSSC trigger, we propose an automated pipeline comprising three modules: a trigger selection module that systematically identifies suitable triggers leveraging large language models, a trigger insertion module that employs generative models to seamlessly integrate triggers into images, and a quality assessment module that ensures the natural and successful insertion of triggers through vision-language models. Extensive experimental results and analysis validate the effectiveness, stealthiness, and robustness of the VSSC trigger. It can not only maintain robustness under visual distortions but also demonstrates strong practicality in the physical scenario. We hope that the proposed VSSC trigger and implementation approach could inspire future studies on designing more practical triggers in backdoor attacks.

摘要: 深度神经网络(DNN)可以在暴露于特定触发模式时表现出特定的行为，而不会影响它们在良性样本上的性能，这被称为\textit{后门攻击}。目前，在物理场景中实施后门攻击仍然面临重大挑战。物理攻击劳动强度大、耗时长，触发点选择采用人工和启发式方式。此外，将数字攻击扩展到物理场景面临许多挑战，因为它们对视觉扭曲很敏感，而且现实世界中没有对应的攻击。为了应对这些挑战，我们定义了一种新的触发器，称为\Textbf{V}可扩展的、\Textbf{S}可扩展的、\Textbf{S}全特定的和\Textbf{C}兼容的(VSSC)触发器，以实现有效、隐蔽和健壮的同时，也可以使用相应的对象在物理场景中有效部署。为了实现VSSC触发器，我们提出了一个包括三个模块的自动化流水线：利用大型语言模型系统地识别合适的触发器的触发器选择模块，使用生成式模型无缝地将触发器集成到图像中的触发器插入模块，以及通过视觉语言模型确保触发器的自然和成功插入的质量评估模块。大量的实验结果和分析验证了VSSC触发器的有效性、隐蔽性和稳健性。该算法不仅能在视觉失真下保持较强的鲁棒性，而且在实际场景中表现出较强的实用性。我们希望提出的VSSC触发器和实现方法可以启发未来设计更实用的后门攻击触发器的研究。



## **7. Explaining Arguments' Strength: Unveiling the Role of Attacks and Supports (Technical Report)**

解释争论的力量：揭示攻击和支持的作用（技术报告） cs.AI

This paper has been accepted at IJCAI 2024 (the 33rd International  Joint Conference on Artificial Intelligence)

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.14304v1) [paper-pdf](http://arxiv.org/pdf/2404.14304v1)

**Authors**: Xiang Yin, Potyka Nico, Francesca Toni

**Abstract**: Quantitatively explaining the strength of arguments under gradual semantics has recently received increasing attention. Specifically, several works in the literature provide quantitative explanations by computing the attribution scores of arguments. These works disregard the importance of attacks and supports, even though they play an essential role when explaining arguments' strength. In this paper, we propose a novel theory of Relation Attribution Explanations (RAEs), adapting Shapley values from game theory to offer fine-grained insights into the role of attacks and supports in quantitative bipolar argumentation towards obtaining the arguments' strength. We show that RAEs satisfy several desirable properties. We also propose a probabilistic algorithm to approximate RAEs efficiently. Finally, we show the application value of RAEs in fraud detection and large language models case studies.

摘要: 在渐进语义下定量解释论点的强度最近受到越来越多的关注。具体来说，文献中的几部作品通过计算论点的归因分数来提供量化解释。这些作品忽视了攻击和支持的重要性，尽管它们在解释论点的强度时发挥着至关重要的作用。在本文中，我们提出了一种新的关系归因解释（RAEs）理论，改编了博弈论中的沙普利价值观，以提供对攻击作用的细粒度见解，并支持量化两极论证，以获得论点的强度。我们表明RAE满足几个理想的性质。我们还提出了一种有效逼近RAE的概率算法。最后，我们展示了RAE在欺诈检测和大型语言模型案例研究中的应用价值。



## **8. Physical Backdoor Attack can Jeopardize Driving with Vision-Large-Language Models**

物理后门攻击可能危及使用视觉大语言模型的驾驶 cs.CR

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.12916v2) [paper-pdf](http://arxiv.org/pdf/2404.12916v2)

**Authors**: Zhenyang Ni, Rui Ye, Yuxi Wei, Zhen Xiang, Yanfeng Wang, Siheng Chen

**Abstract**: Vision-Large-Language-models(VLMs) have great application prospects in autonomous driving. Despite the ability of VLMs to comprehend and make decisions in complex scenarios, their integration into safety-critical autonomous driving systems poses serious security risks. In this paper, we propose BadVLMDriver, the first backdoor attack against VLMs for autonomous driving that can be launched in practice using physical objects. Unlike existing backdoor attacks against VLMs that rely on digital modifications, BadVLMDriver uses common physical items, such as a red balloon, to induce unsafe actions like sudden acceleration, highlighting a significant real-world threat to autonomous vehicle safety. To execute BadVLMDriver, we develop an automated pipeline utilizing natural language instructions to generate backdoor training samples with embedded malicious behaviors. This approach allows for flexible trigger and behavior selection, enhancing the stealth and practicality of the attack in diverse scenarios. We conduct extensive experiments to evaluate BadVLMDriver for two representative VLMs, five different trigger objects, and two types of malicious backdoor behaviors. BadVLMDriver achieves a 92% attack success rate in inducing a sudden acceleration when coming across a pedestrian holding a red balloon. Thus, BadVLMDriver not only demonstrates a critical security risk but also emphasizes the urgent need for developing robust defense mechanisms to protect against such vulnerabilities in autonomous driving technologies.

摘要: 视觉大语言模型在自动驾驶领域有着广阔的应用前景。尽管VLMS具有在复杂场景下理解和决策的能力，但它们集成到安全关键的自动驾驶系统中会带来严重的安全风险。在本文中，我们提出了BadVLMDriver，这是第一个针对自动驾驶的VLM的后门攻击，可以在实践中使用物理对象来发起。与现有的依赖于数字修改的针对VLM的后门攻击不同，BadVLMDriver使用常见的物理物品(如红色气球)来诱导突然加速等不安全行为，突显出现实世界对自动驾驶汽车安全的重大威胁。为了执行BadVLMDriver，我们开发了一个自动流水线，利用自然语言指令生成嵌入了恶意行为的后门训练样本。该方法允许灵活的触发和行为选择，增强了攻击在不同场景下的隐蔽性和实用性。我们针对两种典型的VLM、五种不同的触发器对象和两种类型的恶意后门行为，进行了大量的实验来评估BadVLMDriver。BadVLMDriver在遇到举着红色气球的行人时，诱导突然加速的攻击成功率达到92%。因此，BadVLMDriver不仅展示了一个严重的安全风险，而且还强调了迫切需要开发强大的防御机制，以防止自动驾驶技术中的此类漏洞。



## **9. Protecting Your LLMs with Information Bottleneck**

通过信息瓶颈保护您的LLC cs.CL

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.13968v1) [paper-pdf](http://arxiv.org/pdf/2404.13968v1)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.

摘要: 大型语言模型的出现给自然语言处理领域带来了革命性的变化，但它们可能会受到攻击，产生有害的内容。尽管努力在道德上调整LLM，但这些往往是脆弱的，可以通过优化或手动对抗性提示通过越狱攻击来绕过。为了解决这个问题，我们引入了信息瓶颈保护器(IBProtector)，这是一种基于信息瓶颈原理的防御机制，我们修改了目标以避免琐碎的解决方案。IBProtector有选择地压缩和干扰提示，由一个轻量级和可训练的提取程序促进，只保留目标LLMS的基本信息，以响应预期的答案。此外，我们还进一步考虑了梯度不可见的情况，以与任何LLM相容。我们的经验评估表明，在不过度影响响应质量或推理速度的情况下，IBProtector在缓解越狱企图方面优于现有的防御方法。它对各种攻击方法和目标LLM的有效性和适应性突显了IBProtector作为一种新型、可转移的防御系统的潜力，无需修改底层模型即可增强LLM的安全性。



## **10. Typos that Broke the RAG's Back: Genetic Attack on RAG Pipeline by Simulating Documents in the Wild via Low-level Perturbations**

让RAG筋疲力尽的错别字：通过低水平扰动模拟野外文档对RAG管道进行基因攻击 cs.CL

Under Review

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.13948v1) [paper-pdf](http://arxiv.org/pdf/2404.13948v1)

**Authors**: Sukmin Cho, Soyeong Jeong, Jeongyeon Seo, Taeho Hwang, Jong C. Park

**Abstract**: The robustness of recent Large Language Models (LLMs) has become increasingly crucial as their applicability expands across various domains and real-world applications. Retrieval-Augmented Generation (RAG) is a promising solution for addressing the limitations of LLMs, yet existing studies on the robustness of RAG often overlook the interconnected relationships between RAG components or the potential threats prevalent in real-world databases, such as minor textual errors. In this work, we investigate two underexplored aspects when assessing the robustness of RAG: 1) vulnerability to noisy documents through low-level perturbations and 2) a holistic evaluation of RAG robustness. Furthermore, we introduce a novel attack method, the Genetic Attack on RAG (\textit{GARAG}), which targets these aspects. Specifically, GARAG is designed to reveal vulnerabilities within each component and test the overall system functionality against noisy documents. We validate RAG robustness by applying our \textit{GARAG} to standard QA datasets, incorporating diverse retrievers and LLMs. The experimental results show that GARAG consistently achieves high attack success rates. Also, it significantly devastates the performance of each component and their synergy, highlighting the substantial risk that minor textual inaccuracies pose in disrupting RAG systems in the real world.

摘要: 最近的大型语言模型(LLM)的健壮性已经变得越来越重要，因为它们的适用性在各个领域和现实世界的应用程序中扩展。检索-增强生成(RAG)是解决LLMS局限性的一种很有前途的解决方案，但现有的RAG健壮性研究往往忽略了RAG组件之间的相互关联关系或现实世界数据库中普遍存在的潜在威胁，如微小的文本错误。在这项工作中，我们研究了两个在评估RAG稳健性时未被探索的方面：1)通过低层扰动对噪声文档的脆弱性；2)RAG稳健性的整体评估。此外，我们还介绍了一种针对这些方面的新的攻击方法--对RAG的遗传攻击(\textit{garag})。具体地说，Garag旨在揭示每个组件中的漏洞，并针对嘈杂的文档测试整个系统功能。我们通过将我们的\textit{garag}应用到标准的QA数据集来验证RAG的健壮性，其中包含了不同的检索器和LLM。实验结果表明，GARAG算法始终具有较高的攻击成功率。此外，它还严重破坏了每个组件的性能及其协同作用，突显了微小的文本错误在扰乱现实世界中的RAG系统方面构成的巨大风险。



## **11. Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs**

竞争报告：在一致的LLC中寻找通用越狱后门 cs.CL

Competition Report

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.14461v1) [paper-pdf](http://arxiv.org/pdf/2404.14461v1)

**Authors**: Javier Rando, Francesco Croce, Kryštof Mitka, Stepan Shabalin, Maksym Andriushchenko, Nicolas Flammarion, Florian Tramèr

**Abstract**: Large language models are aligned to be safe, preventing users from generating harmful content like misinformation or instructions for illegal activities. However, previous work has shown that the alignment process is vulnerable to poisoning attacks. Adversaries can manipulate the safety training data to inject backdoors that act like a universal sudo command: adding the backdoor string to any prompt enables harmful responses from models that, otherwise, behave safely. Our competition, co-located at IEEE SaTML 2024, challenged participants to find universal backdoors in several large language models. This report summarizes the key findings and promising ideas for future research.

摘要: 大型语言模型经过调整以确保安全，防止用户生成错误信息或非法活动指令等有害内容。然而，之前的工作表明，对齐过程很容易受到中毒攻击。对手可以操纵安全训练数据来注入类似于通用sudo命令的后门：将后门字符串添加到任何提示中都会导致模型做出有害响应，否则这些模型会安全地运行。我们的竞赛在IEEE SaTML 2024上举行，挑战参与者在几个大型语言模型中找到通用后门。本报告总结了关键发现和未来研究的有希望的想法。



## **12. Bot or Human? Detecting ChatGPT Imposters with A Single Question**

机器人还是人类？通过一个问题检测ChatGPT冒名顶替者 cs.CL

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2305.06424v3) [paper-pdf](http://arxiv.org/pdf/2305.06424v3)

**Authors**: Hong Wang, Xuan Luo, Weizhi Wang, Xifeng Yan

**Abstract**: Large language models like GPT-4 have recently demonstrated impressive capabilities in natural language understanding and generation, enabling various applications including translation, essay writing, and chit-chatting. However, there is a concern that they can be misused for malicious purposes, such as fraud or denial-of-service attacks. Therefore, it is crucial to develop methods for detecting whether the party involved in a conversation is a bot or a human. In this paper, we propose a framework named FLAIR, Finding Large Language Model Authenticity via a Single Inquiry and Response, to detect conversational bots in an online manner. Specifically, we target a single question scenario that can effectively differentiate human users from bots. The questions are divided into two categories: those that are easy for humans but difficult for bots (e.g., counting, substitution, and ASCII art reasoning), and those that are easy for bots but difficult for humans (e.g., memorization and computation). Our approach shows different strengths of these questions in their effectiveness, providing a new way for online service providers to protect themselves against nefarious activities and ensure that they are serving real users. We open-sourced our code and dataset on https://github.com/hongwang600/FLAIR and welcome contributions from the community.

摘要: 像GPT-4这样的大型语言模型最近在自然语言理解和生成方面表现出了令人印象深刻的能力，支持各种应用程序，包括翻译、论文写作和聊天。然而，人们担心它们可能被滥用于恶意目的，如欺诈或拒绝服务攻击。因此，开发方法来检测参与对话的一方是机器人还是人类是至关重要的。在本文中，我们提出了一个名为FLAIR的框架，通过单一查询和响应来发现大型语言模型的真实性，以在线方式检测会话机器人。具体地说，我们的目标是能够有效区分人类用户和机器人的单一问题场景。这些问题分为两类：一类是对人类容易但对机器人困难的问题(例如，计数、替换和ASCII艺术推理)；另一类是对机器人容易但对人类困难的问题(例如，记忆和计算)。我们的方法显示了这些问题在有效性上的不同优势，为在线服务提供商提供了一种新的方式来保护自己免受恶意活动的影响，并确保他们服务的是真实的用户。我们在https://github.com/hongwang600/FLAIR上开源了我们的代码和数据集，并欢迎来自社区的贡献。



## **13. Trojan Detection in Large Language Models: Insights from The Trojan Detection Challenge**

大型语言模型中的特洛伊木马检测：特洛伊木马检测挑战的见解 cs.CL

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2404.13660v1) [paper-pdf](http://arxiv.org/pdf/2404.13660v1)

**Authors**: Narek Maloyan, Ekansh Verma, Bulat Nutfullin, Bislan Ashinov

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various domains, but their vulnerability to trojan or backdoor attacks poses significant security risks. This paper explores the challenges and insights gained from the Trojan Detection Competition 2023 (TDC2023), which focused on identifying and evaluating trojan attacks on LLMs. We investigate the difficulty of distinguishing between intended and unintended triggers, as well as the feasibility of reverse engineering trojans in real-world scenarios. Our comparative analysis of various trojan detection methods reveals that achieving high Recall scores is significantly more challenging than obtaining high Reverse-Engineering Attack Success Rate (REASR) scores. The top-performing methods in the competition achieved Recall scores around 0.16, comparable to a simple baseline of randomly sampling sentences from a distribution similar to the given training prefixes. This finding raises questions about the detectability and recoverability of trojans inserted into the model, given only the harmful targets. Despite the inability to fully solve the problem, the competition has led to interesting observations about the viability of trojan detection and improved techniques for optimizing LLM input prompts. The phenomenon of unintended triggers and the difficulty in distinguishing them from intended triggers highlights the need for further research into the robustness and interpretability of LLMs. The TDC2023 has provided valuable insights into the challenges and opportunities associated with trojan detection in LLMs, laying the groundwork for future research in this area to ensure their safety and reliability in real-world applications.

摘要: 大型语言模型(LLM)在各个领域都表现出了卓越的能力，但它们对特洛伊木马或后门攻击的脆弱性带来了巨大的安全风险。本文探讨了从木马检测竞赛2023(TDC2023)中获得的挑战和见解，该竞赛的重点是识别和评估针对LLMS的木马攻击。我们调查了区分有意和无意触发器的困难，以及在真实世界场景中反向工程特洛伊木马的可行性。我们对各种木马检测方法的比较分析表明，实现高召回率比获得高反向工程攻击成功率(REASR)更具挑战性。在比赛中表现最好的方法获得了约0.16分的回忆分数，与从类似于给定训练前缀的分布中随机抽样句子的简单基线相当。这一发现提出了一个问题，即在只给出有害目标的情况下，插入到模型中的特洛伊木马程序的可检测性和可恢复性。尽管无法完全解决这个问题，但竞争已经导致了对特洛伊木马检测的可行性的有趣观察，并改进了优化LLM输入提示的技术。非预期触发因素的现象以及将它们与预期触发因素区分开来的困难突出表明，有必要进一步研究小岛屿发展中国家的稳健性和可解释性。TDC2023提供了与低层管理中木马检测相关的挑战和机遇的宝贵见解，为这一领域的未来研究奠定了基础，以确保其在现实世界应用中的安全性和可靠性。



## **14. Large Language Models for Blockchain Security: A Systematic Literature Review**

区块链安全的大型语言模型：系统性文献综述 cs.CR

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2403.14280v3) [paper-pdf](http://arxiv.org/pdf/2403.14280v3)

**Authors**: Zheyuan He, Zihao Li, Sen Yang

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools in various domains involving blockchain security (BS). Several recent studies are exploring LLMs applied to BS. However, there remains a gap in our understanding regarding the full scope of applications, impacts, and potential constraints of LLMs on blockchain security. To fill this gap, we conduct a literature review on LLM4BS.   As the first review of LLM's application on blockchain security, our study aims to comprehensively analyze existing research and elucidate how LLMs contribute to enhancing the security of blockchain systems. Through a thorough examination of scholarly works, we delve into the integration of LLMs into various aspects of blockchain security. We explore the mechanisms through which LLMs can bolster blockchain security, including their applications in smart contract auditing, identity verification, anomaly detection, vulnerable repair, and so on. Furthermore, we critically assess the challenges and limitations associated with leveraging LLMs for blockchain security, considering factors such as scalability, privacy concerns, and adversarial attacks. Our review sheds light on the opportunities and potential risks inherent in this convergence, providing valuable insights for researchers, practitioners, and policymakers alike.

摘要: 大型语言模型(LLM)在涉及区块链安全(BS)的各个领域中已成为强大的工具。最近的几项研究正在探索将LLMS应用于BS。然而，对于低成本管理的全部应用范围、影响以及对区块链安全的潜在限制，我们的理解仍然存在差距。为了填补这一空白，我们对LLM4BS进行了文献综述。作为LLM在区块链安全方面应用的首次综述，本研究旨在全面分析现有研究，阐明LLM如何为增强区块链系统的安全性做出贡献。通过对学术著作的深入研究，我们深入研究了LLMS在区块链安全的各个方面的整合。我们探讨了LLMS增强区块链安全的机制，包括它们在智能合同审计、身份验证、异常检测、漏洞修复等方面的应用。此外，考虑到可扩展性、隐私问题和敌意攻击等因素，我们严格评估了利用LLM实现区块链安全所面临的挑战和限制。我们的审查揭示了这种融合所固有的机遇和潜在风险，为研究人员、从业者和政策制定者提供了有价值的见解。



## **15. Intrusion Detection at Scale with the Assistance of a Command-line Language Model**

在命令行语言模型的帮助下进行大规模入侵检测 cs.CR

Accepted by IEEE/IFIP International Conference on Dependable Systems  and Networks (DSN), industry track

**SubmitDate**: 2024-04-20    [abs](http://arxiv.org/abs/2404.13402v1) [paper-pdf](http://arxiv.org/pdf/2404.13402v1)

**Authors**: Jiongliang Lin, Yiwen Guo, Hao Chen

**Abstract**: Intrusion detection is a long standing and crucial problem in security. A system capable of detecting intrusions automatically is on great demand in enterprise security solutions. Existing solutions rely heavily on hand-crafted rules designed by security operators, which suffer from high false negative rates and poor generalization ability to new, zero-day attacks at scale. AI and machine learning offer promising solutions to address the issues, by inspecting abnormal user behaviors intelligently and automatically from data. However, existing learning-based intrusion detection systems in the literature are mostly designed for small data, and they lack the ability to leverage the power of big data in cloud environments. In this paper, we target at this problem and introduce an intrusion detection system which incorporates large-scale pre-training, so as to train a large language model based on tens of millions of command lines for AI-based intrusion detection. Experiments performed on 30 million training samples and 10 million test samples verify the effectiveness of our solution.

摘要: 入侵检测是安全领域中一个由来已久的重要问题。在企业安全解决方案中，需要一种能够自动检测入侵的系统。现有的解决方案严重依赖于安全运营商设计的手动规则，这些规则存在较高的假阴性率和对新的零日大规模攻击的泛化能力较差。人工智能和机器学习通过从数据中智能和自动地检查异常用户行为，为解决这些问题提供了有前途的解决方案。然而，文献中现有的基于学习的入侵检测系统大多是针对小数据而设计的，它们缺乏在云环境中利用大数据的能力。本文针对这一问题，提出了一种结合大规模预训练的入侵检测系统，为基于人工智能的入侵检测训练一个基于数千万条命令行的大型语言模型。在3000万个训练样本和1000万个测试样本上的实验验证了该方法的有效性。



## **16. ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs**

ArtPrompt：针对对齐的LLM的基于ASC艺术的越狱攻击 cs.CL

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2402.11753v3) [paper-pdf](http://arxiv.org/pdf/2402.11753v3)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Zhen Xiang, Bhaskar Ramasubramanian, Bo Li, Radha Poovendran

**Abstract**: Safety is critical to the usage of large language models (LLMs). Multiple techniques such as data filtering and supervised fine-tuning have been developed to strengthen LLM safety. However, currently known techniques presume that corpora used for safety alignment of LLMs are solely interpreted by semantics. This assumption, however, does not hold in real-world applications, which leads to severe vulnerabilities in LLMs. For example, users of forums often use ASCII art, a form of text-based art, to convey image information. In this paper, we propose a novel ASCII art-based jailbreak attack and introduce a comprehensive benchmark Vision-in-Text Challenge (ViTC) to evaluate the capabilities of LLMs in recognizing prompts that cannot be solely interpreted by semantics. We show that five SOTA LLMs (GPT-3.5, GPT-4, Gemini, Claude, and Llama2) struggle to recognize prompts provided in the form of ASCII art. Based on this observation, we develop the jailbreak attack ArtPrompt, which leverages the poor performance of LLMs in recognizing ASCII art to bypass safety measures and elicit undesired behaviors from LLMs. ArtPrompt only requires black-box access to the victim LLMs, making it a practical attack. We evaluate ArtPrompt on five SOTA LLMs, and show that ArtPrompt can effectively and efficiently induce undesired behaviors from all five LLMs. Our code is available at https://github.com/uw-nsl/ArtPrompt.

摘要: 安全对于大型语言模型(LLM)的使用至关重要。已经开发了多种技术，如数据过滤和有监督的微调，以加强LLM的安全性。然而，目前已知的技术假定用于LLM的安全对准的语料库仅由语义解释。然而，这一假设在现实世界的应用程序中并不成立，这导致了LLMS中的严重漏洞。例如，论坛的用户经常使用ASCII艺术，这是一种基于文本的艺术形式，以传达图像信息。本文提出了一种新的基于ASCII ART的越狱攻击方法，并引入了一个综合基准的文本中视觉挑战(VITC)来评估LLMS在识别不能完全由语义解释的提示方面的能力。我们发现，五个SOTA LLM(GPT-3.5、GPT-4、双子座、克劳德和Llama2)很难识别以ASCII ART形式提供的提示。基于这种观察，我们开发了越狱攻击ArtPrompt，它利用LLMS在识别ASCII ART方面的较差性能来绕过安全措施，并从LLM引发不希望看到的行为。ArtPrompt只需要黑盒访问受攻击的LLM，这使其成为一种实际的攻击。我们在五个SOTA LLM上对ArtPrompt进行了评估，结果表明，ArtPrompt可以有效和高效地诱导所有五个LLM的不良行为。我们的代码可以在https://github.com/uw-nsl/ArtPrompt.上找到



## **17. CyberSecEval 2: A Wide-Ranging Cybersecurity Evaluation Suite for Large Language Models**

CyberSecEval 2：针对大型语言模型的广泛网络安全评估套件 cs.CR

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.13161v1) [paper-pdf](http://arxiv.org/pdf/2404.13161v1)

**Authors**: Manish Bhatt, Sahana Chennabasappa, Yue Li, Cyrus Nikolaidis, Daniel Song, Shengye Wan, Faizan Ahmad, Cornelius Aschermann, Yaohui Chen, Dhaval Kapil, David Molnar, Spencer Whitman, Joshua Saxe

**Abstract**: Large language models (LLMs) introduce new security risks, but there are few comprehensive evaluation suites to measure and reduce these risks. We present BenchmarkName, a novel benchmark to quantify LLM security risks and capabilities. We introduce two new areas for testing: prompt injection and code interpreter abuse. We evaluated multiple state-of-the-art (SOTA) LLMs, including GPT-4, Mistral, Meta Llama 3 70B-Instruct, and Code Llama. Our results show that conditioning away risk of attack remains an unsolved problem; for example, all tested models showed between 26% and 41% successful prompt injection tests. We further introduce the safety-utility tradeoff: conditioning an LLM to reject unsafe prompts can cause the LLM to falsely reject answering benign prompts, which lowers utility. We propose quantifying this tradeoff using False Refusal Rate (FRR). As an illustration, we introduce a novel test set to quantify FRR for cyberattack helpfulness risk. We find many LLMs able to successfully comply with "borderline" benign requests while still rejecting most unsafe requests. Finally, we quantify the utility of LLMs for automating a core cybersecurity task, that of exploiting software vulnerabilities. This is important because the offensive capabilities of LLMs are of intense interest; we quantify this by creating novel test sets for four representative problems. We find that models with coding capabilities perform better than those without, but that further work is needed for LLMs to become proficient at exploit generation. Our code is open source and can be used to evaluate other LLMs.

摘要: 大型语言模型(LLM)引入了新的安全风险，但很少有全面的评估套件来衡量和降低这些风险。我们提出了BenchmarkName，这是一种量化LLM安全风险和能力的新基准。我们引入了两个新的测试领域：快速注入和代码解释器滥用。我们评估了多个最先进的(SOTA)LLMS，包括GPT-4、Mistral、Meta Llama 3 70B-Indict和Code Llama。我们的结果表明，条件化的攻击风险仍然是一个未解决的问题；例如，所有测试的模型显示，26%到41%的成功的快速注射测试。我们进一步介绍安全与效用的权衡：使LLM拒绝不安全提示可能会导致LLM错误地拒绝回答良性提示，从而降低效用。我们建议使用错误拒绝率(FRR)来量化这种权衡。作为说明，我们引入了一个新的测试集来量化网络攻击帮助风险的FRR。我们发现许多LLM能够成功地满足“边缘”良性请求，同时仍然拒绝大多数不安全的请求。最后，我们量化了LLMS用于自动化核心网络安全任务的效用，即利用软件漏洞。这一点很重要，因为人们对LLMS的进攻能力非常感兴趣；我们通过为四个具有代表性的问题创建新的测试集来量化这一点。我们发现，有编码能力的模型比没有编码能力的模型表现得更好，但要想熟练地利用漏洞生成，还需要进一步的工作。我们的代码是开源的，可以用来评估其他LLM。



## **18. Heterogeneous Federated Learning with Splited Language Model**

使用分裂语言模型的异类联邦学习 cs.CV

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2403.16050v2) [paper-pdf](http://arxiv.org/pdf/2403.16050v2)

**Authors**: Yifan Shi, Yuhui Zhang, Ziyue Huang, Xiaofeng Yang, Li Shen, Wei Chen, Xueqian Wang

**Abstract**: Federated Split Learning (FSL) is a promising distributed learning paradigm in practice, which gathers the strengths of both Federated Learning (FL) and Split Learning (SL) paradigms, to ensure model privacy while diminishing the resource overhead of each client, especially on large transformer models in a resource-constrained environment, e.g., Internet of Things (IoT). However, almost all works merely investigate the performance with simple neural network models in FSL. Despite the minor efforts focusing on incorporating Vision Transformers (ViT) as model architectures, they train ViT from scratch, thereby leading to enormous training overhead in each device with limited resources. Therefore, in this paper, we harness Pre-trained Image Transformers (PITs) as the initial model, coined FedV, to accelerate the training process and improve model robustness. Furthermore, we propose FedVZ to hinder the gradient inversion attack, especially having the capability compatible with black-box scenarios, where the gradient information is unavailable. Concretely, FedVZ approximates the server gradient by utilizing a zeroth-order (ZO) optimization, which replaces the backward propagation with just one forward process. Empirically, we are the first to provide a systematic evaluation of FSL methods with PITs in real-world datasets, different partial device participations, and heterogeneous data splits. Our experiments verify the effectiveness of our algorithms.

摘要: 联合分裂学习(FSL)是一种具有实际应用前景的分布式学习范式，它结合了联合学习(FL)和分裂学习(SL)的优点，在保证模型隐私的同时减少了每个客户端的资源开销，特别是在资源受限环境下的大型变压器模型上，例如物联网(IoT)。然而，几乎所有的工作都只是用简单的神经网络模型来研究FSL的性能。尽管在整合Vision Transformers(VIT)作为模型架构方面做了很小的努力，但他们从头开始训练VIT，从而导致在资源有限的每个设备中产生巨大的训练开销。因此，在本文中，我们利用预先训练的图像转换器(PITS)作为初始模型，称为FedV，以加快训练过程，提高模型的稳健性。此外，我们提出了FedVZ来阻止梯度反转攻击，特别是在梯度信息不可用的黑盒场景中具有兼容的能力。具体地说，FedVZ通过使用零阶(ZO)优化来近似服务器梯度，该优化仅用一个前向过程来代替后向传播。在经验上，我们是第一个在真实数据集、不同部分设备参与和不同数据拆分中对FSL方法进行系统评估的公司。我们的实验验证了算法的有效性。



## **19. A Survey on LLM-Generated Text Detection: Necessity, Methods, and Future Directions**

LLM生成文本检测概览：必要性、方法和未来方向 cs.CL

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2310.14724v3) [paper-pdf](http://arxiv.org/pdf/2310.14724v3)

**Authors**: Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Derek F. Wong, Lidia S. Chao

**Abstract**: The powerful ability to understand, follow, and generate complex language emerging from large language models (LLMs) makes LLM-generated text flood many areas of our daily lives at an incredible speed and is widely accepted by humans. As LLMs continue to expand, there is an imperative need to develop detectors that can detect LLM-generated text. This is crucial to mitigate potential misuse of LLMs and safeguard realms like artistic expression and social networks from harmful influence of LLM-generated content. The LLM-generated text detection aims to discern if a piece of text was produced by an LLM, which is essentially a binary classification task. The detector techniques have witnessed notable advancements recently, propelled by innovations in watermarking techniques, statistics-based detectors, neural-base detectors, and human-assisted methods. In this survey, we collate recent research breakthroughs in this area and underscore the pressing need to bolster detector research. We also delve into prevalent datasets, elucidating their limitations and developmental requirements. Furthermore, we analyze various LLM-generated text detection paradigms, shedding light on challenges like out-of-distribution problems, potential attacks, real-world data issues and the lack of effective evaluation framework. Conclusively, we highlight interesting directions for future research in LLM-generated text detection to advance the implementation of responsible artificial intelligence (AI). Our aim with this survey is to provide a clear and comprehensive introduction for newcomers while also offering seasoned researchers a valuable update in the field of LLM-generated text detection. The useful resources are publicly available at: https://github.com/NLP2CT/LLM-generated-Text-Detection.

摘要: 大型语言模型(LLM)强大的理解、跟踪和生成复杂语言的能力使得LLM生成的文本以令人难以置信的速度涌入我们日常生活的许多领域，并被人类广泛接受。随着LLMS的不断扩展，迫切需要开发能够检测LLM生成的文本的检测器。这对于减少LLM的潜在滥用以及保护艺术表达和社交网络等领域免受LLM生成的内容的有害影响至关重要。LLM生成的文本检测旨在识别一段文本是否由LLM生成，这本质上是一项二进制分类任务。最近，在水印技术、基于统计的检测器、基于神经的检测器和人工辅助方法的创新的推动下，检测器技术取得了显著的进步。在这次调查中，我们整理了这一领域的最新研究突破，并强调了支持探测器研究的迫切需要。我们还深入研究了流行的数据集，阐明了它们的局限性和发展需求。此外，我们分析了各种LLM生成的文本检测范例，揭示了诸如分布不足问题、潜在攻击、真实世界的数据问题以及缺乏有效的评估框架等挑战。最后，我们指出了未来在LLM生成的文本检测方面的有趣研究方向，以推进负责任人工智能(AI)的实施。我们这次调查的目的是为新手提供一个清晰而全面的介绍，同时也为经验丰富的研究人员提供在LLM生成的文本检测领域的有价值的更新。这些有用的资源可在以下网址公开获得：https://github.com/NLP2CT/LLM-generated-Text-Detection.



## **20. JailBreakV-28K: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

JailBreakV-28 K：评估多模式大型语言模型对抗越狱攻击的稳健性的基准 cs.CR

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.03027v2) [paper-pdf](http://arxiv.org/pdf/2404.03027v2)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.

摘要: 随着多模式大型语言模型(MLLMS)的快速发展，保护这些模型不受恶意输入的影响，同时使它们与人类的价值观保持一致，已经成为一项关键的挑战。在本文中，我们研究了一个重要而未被探索的问题，即成功越狱大语言模型(LLMS)的技术是否可以在越狱MLLM中同样有效。为了探讨这一问题，我们引入了JailBreakV-28K，这是一个开创性的基准测试，旨在评估LLM越狱技术到MLLM的可转移性，从而评估MLLMS对各种越狱攻击的健壮性。利用本文提出的包含2,000个恶意查询的数据集，我们使用针对LLMS的高级越狱攻击生成了20,000个基于文本的越狱提示，以及来自最近MLLMS越狱攻击的8,000个基于图像的越狱输入，我们的综合数据集包括来自各种对抗场景的28,000个测试用例。我们对10个开源MLLMS的评估显示，对于从LLMS转移的攻击，攻击成功率(ASR)非常高，这突显了MLLMS中源于其文本处理能力的一个严重漏洞。我们的发现强调了未来研究的迫切需要，以解决MLLMS中从文本和视觉输入的对齐漏洞。



## **21. Advancing the Robustness of Large Language Models through Self-Denoised Smoothing**

通过自去噪平滑提高大型语言模型的鲁棒性 cs.CL

Accepted by NAACL 2024. Jiabao, Bairu, Zhen, Guanhua contributed  equally. This is an updated version of the paper: arXiv:2307.07171

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12274v1) [paper-pdf](http://arxiv.org/pdf/2404.12274v1)

**Authors**: Jiabao Ji, Bairu Hou, Zhen Zhang, Guanhua Zhang, Wenqi Fan, Qing Li, Yang Zhang, Gaowen Liu, Sijia Liu, Shiyu Chang

**Abstract**: Although large language models (LLMs) have achieved significant success, their vulnerability to adversarial perturbations, including recent jailbreak attacks, has raised considerable concerns. However, the increasing size of these models and their limited access make improving their robustness a challenging task. Among various defense strategies, randomized smoothing has shown great potential for LLMs, as it does not require full access to the model's parameters or fine-tuning via adversarial training. However, randomized smoothing involves adding noise to the input before model prediction, and the final model's robustness largely depends on the model's performance on these noise corrupted data. Its effectiveness is often limited by the model's sub-optimal performance on noisy data. To address this issue, we propose to leverage the multitasking nature of LLMs to first denoise the noisy inputs and then to make predictions based on these denoised versions. We call this procedure self-denoised smoothing. Unlike previous denoised smoothing techniques in computer vision, which require training a separate model to enhance the robustness of LLMs, our method offers significantly better efficiency and flexibility. Our experimental results indicate that our method surpasses existing methods in both empirical and certified robustness in defending against adversarial attacks for both downstream tasks and human alignments (i.e., jailbreak attacks). Our code is publicly available at https://github.com/UCSB-NLP-Chang/SelfDenoise

摘要: 虽然大型语言模型(LLM)已经取得了巨大的成功，但它们在对抗扰动中的脆弱性，包括最近的越狱攻击，已经引起了相当大的关注。然而，这些模型的规模越来越大，访问范围有限，因此提高它们的稳健性是一项具有挑战性的任务。在各种防御策略中，随机平滑在LLMS中显示出巨大的潜力，因为它不需要完全获取模型参数或通过对抗性训练进行微调。然而，随机平滑涉及在模型预测之前向输入添加噪声，而最终模型的稳健性在很大程度上取决于模型对这些噪声污染数据的性能。它的有效性往往受到模型在噪声数据上的次优性能的限制。为了解决这个问题，我们建议利用LLMS的多任务特性来首先对噪声输入进行去噪，然后基于这些去噪版本进行预测。我们称这一过程为自去噪平滑。与计算机视觉中以前的去噪平滑技术不同，我们的方法提供了更好的效率和灵活性，需要训练单独的模型来增强LLMS的鲁棒性。我们的实验结果表明，我们的方法在抵抗下游任务和人类对齐(即越狱攻击)的对手攻击方面，无论是经验上还是经过验证的稳健性都优于现有方法。我们的代码在https://github.com/UCSB-NLP-Chang/SelfDenoise上公开提供



## **22. Concept Induction: Analyzing Unstructured Text with High-Level Concepts Using LLooM**

概念归纳：使用LLooM分析具有高级概念的非结构化文本 cs.HC

To appear at CHI 2024

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12259v1) [paper-pdf](http://arxiv.org/pdf/2404.12259v1)

**Authors**: Michelle S. Lam, Janice Teoh, James Landay, Jeffrey Heer, Michael S. Bernstein

**Abstract**: Data analysts have long sought to turn unstructured text data into meaningful concepts. Though common, topic modeling and clustering focus on lower-level keywords and require significant interpretative work. We introduce concept induction, a computational process that instead produces high-level concepts, defined by explicit inclusion criteria, from unstructured text. For a dataset of toxic online comments, where a state-of-the-art BERTopic model outputs "women, power, female," concept induction produces high-level concepts such as "Criticism of traditional gender roles" and "Dismissal of women's concerns." We present LLooM, a concept induction algorithm that leverages large language models to iteratively synthesize sampled text and propose human-interpretable concepts of increasing generality. We then instantiate LLooM in a mixed-initiative text analysis tool, enabling analysts to shift their attention from interpreting topics to engaging in theory-driven analysis. Through technical evaluations and four analysis scenarios ranging from literature review to content moderation, we find that LLooM's concepts improve upon the prior art of topic models in terms of quality and data coverage. In expert case studies, LLooM helped researchers to uncover new insights even from familiar datasets, for example by suggesting a previously unnoticed concept of attacks on out-party stances in a political social media dataset.

摘要: 长期以来，数据分析师一直在寻求将非结构化文本数据转化为有意义的概念。虽然常见，但主题建模和聚类侧重于较低级别的关键字，需要大量的解释性工作。我们引入了概念归纳，这是一种计算过程，相反，它从非结构化文本中产生由显式包含标准定义的高级概念。对于有毒的在线评论数据集，一个最先进的BERTITE模型输出的是“女性、权力、女性”，概念归纳产生了诸如“对传统性别角色的批评”和“对女性关切的漠视”等高级概念。我们提出了一种概念归纳算法LLooM，它利用大型语言模型迭代地合成样本文本，并提出了更具通用性的人类可解释概念。然后，我们在一个混合倡议的文本分析工具中实例化LLooM，使分析师能够将他们的注意力从解释主题转移到从事理论驱动的分析。通过技术评估和从文献审查到内容审核的四个分析场景，我们发现LLooM的概念在质量和数据覆盖方面改进了主题模型的现有技术。在专家案例研究中，LLooM帮助研究人员发现了新的见解，即使是从熟悉的数据集也是如此，例如，通过提出一个以前未被注意到的概念，即攻击政治社交媒体数据集中的党外立场。



## **23. Efficiently Adversarial Examples Generation for Visual-Language Models under Targeted Transfer Scenarios using Diffusion Models**

使用扩散模型高效生成目标迁移场景下视觉语言模型的对抗性示例 cs.CV

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.10335v2) [paper-pdf](http://arxiv.org/pdf/2404.10335v2)

**Authors**: Qi Guo, Shanmin Pang, Xiaojun Jia, Qing Guo

**Abstract**: Targeted transfer-based attacks involving adversarial examples pose a significant threat to large visual-language models (VLMs). However, the state-of-the-art (SOTA) transfer-based attacks incur high costs due to excessive iteration counts. Furthermore, the generated adversarial examples exhibit pronounced adversarial noise and demonstrate limited efficacy in evading defense methods such as DiffPure. To address these issues, inspired by score matching, we introduce AdvDiffVLM, which utilizes diffusion models to generate natural, unrestricted adversarial examples. Specifically, AdvDiffVLM employs Adaptive Ensemble Gradient Estimation to modify the score during the diffusion model's reverse generation process, ensuring the adversarial examples produced contain natural adversarial semantics and thus possess enhanced transferability. Simultaneously, to enhance the quality of adversarial examples further, we employ the GradCAM-guided Mask method to disperse adversarial semantics throughout the image, rather than concentrating them in a specific area. Experimental results demonstrate that our method achieves a speedup ranging from 10X to 30X compared to existing transfer-based attack methods, while maintaining superior quality of adversarial examples. Additionally, the generated adversarial examples possess strong transferability and exhibit increased robustness against adversarial defense methods. Notably, AdvDiffVLM can successfully attack commercial VLMs, including GPT-4V, in a black-box manner.

摘要: 涉及对抗性例子的基于目标转移的攻击对大型视觉语言模型(VLM)构成了重大威胁。然而，最先进的(SOTA)基于传输的攻击由于迭代次数过多而导致高昂的成本。此外，生成的对抗性示例显示出明显的对抗性噪声，并且在躲避DiffPure等防御方法方面表现出有限的有效性。为了解决这些问题，受分数匹配的启发，我们引入了AdvDiffVLM，它利用扩散模型来生成自然的、不受限制的对抗性示例。具体地说，AdvDiffVLM在扩散模型的逆向生成过程中使用自适应集成梯度估计来修正分数，确保生成的对抗性实例包含自然对抗性语义，从而具有增强的可转移性。同时，为了进一步提高对抗性实例的质量，我们使用了GradCAM引导的掩码方法，将对抗性语义分散在整个图像中，而不是集中在特定的区域。实验结果表明，与现有的基于传输的攻击方法相比，该方法在保持较好的对抗性实例质量的同时，获得了10倍到30倍的加速比。此外，生成的对抗性实例具有很强的可移植性，并且对对抗性防御方法表现出更强的稳健性。值得注意的是，AdvDiffVLM可以以黑盒方式成功攻击商业VLM，包括GPT-4V。



## **24. Uncovering Safety Risks in Open-source LLMs through Concept Activation Vector**

通过概念激活载体揭示开源LLM的安全风险 cs.CL

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12038v1) [paper-pdf](http://arxiv.org/pdf/2404.12038v1)

**Authors**: Zhihao Xu, Ruixuan Huang, Xiting Wang, Fangzhao Wu, Jing Yao, Xing Xie

**Abstract**: Current open-source large language models (LLMs) are often undergone careful safety alignment before public release. Some attack methods have also been proposed that help check for safety vulnerabilities in LLMs to ensure alignment robustness. However, many of these methods have moderate attack success rates. Even when successful, the harmfulness of their outputs cannot be guaranteed, leading to suspicions that these methods have not accurately identified the safety vulnerabilities of LLMs. In this paper, we introduce a LLM attack method utilizing concept-based model explanation, where we extract safety concept activation vectors (SCAVs) from LLMs' activation space, enabling efficient attacks on well-aligned LLMs like LLaMA-2, achieving near 100% attack success rate as if LLMs are completely unaligned. This suggests that LLMs, even after thorough safety alignment, could still pose potential risks to society upon public release. To evaluate the harmfulness of outputs resulting with various attack methods, we propose a comprehensive evaluation method that reduces the potential inaccuracies of existing evaluations, and further validate that our method causes more harmful content. Additionally, we discover that the SCAVs show some transferability across different open-source LLMs.

摘要: 当前的开源大型语言模型(LLM)在公开发布之前通常都经过了仔细的安全调整。还提出了一些有助于检查LLMS中的安全漏洞以确保对齐健壮性的攻击方法。然而，这些方法中的许多都有中等的攻击成功率。即使在成功的情况下，其输出的危害性也无法得到保证，这导致人们怀疑这些方法没有准确地确定小岛屿发展中国家的安全漏洞。本文提出了一种基于概念模型解释的LLM攻击方法，从LLMS的激活空间中提取安全概念激活向量(SAVA)，对LLAMA-2等排列良好的LLM进行有效攻击，取得了接近100%的攻击成功率，就好像LLM完全不匹配一样。这表明，即使在进行了彻底的安全调整后，低密度脂蛋白在公开释放后仍可能对社会构成潜在风险。为了评估各种攻击方法产生的结果的危害性，我们提出了一种综合评估方法，减少了现有评估的潜在不准确性，并进一步验证了我们的方法导致了更多的有害内容。此外，我们发现SVAC在不同的开源LLM之间表现出一定的可转移性。



## **25. Sampling-based Pseudo-Likelihood for Membership Inference Attacks**

基于抽样的成员推断攻击伪似然 cs.CL

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2404.11262v1) [paper-pdf](http://arxiv.org/pdf/2404.11262v1)

**Authors**: Masahiro Kaneko, Youmi Ma, Yuki Wata, Naoaki Okazaki

**Abstract**: Large Language Models (LLMs) are trained on large-scale web data, which makes it difficult to grasp the contribution of each text. This poses the risk of leaking inappropriate data such as benchmarks, personal information, and copyrighted texts in the training data. Membership Inference Attacks (MIA), which determine whether a given text is included in the model's training data, have been attracting attention. Previous studies of MIAs revealed that likelihood-based classification is effective for detecting leaks in LLMs. However, the existing methods cannot be applied to some proprietary models like ChatGPT or Claude 3 because the likelihood is unavailable to the user. In this study, we propose a Sampling-based Pseudo-Likelihood (\textbf{SPL}) method for MIA (\textbf{SaMIA}) that calculates SPL using only the text generated by an LLM to detect leaks. The SaMIA treats the target text as the reference text and multiple outputs from the LLM as text samples, calculates the degree of $n$-gram match as SPL, and determines the membership of the text in the training data. Even without likelihoods, SaMIA performed on par with existing likelihood-based methods.

摘要: 大型语言模型(LLM)是在大规模Web数据上训练的，这使得很难掌握每个文本的贡献。这带来了泄露不适当数据的风险，例如培训数据中的基准、个人信息和受版权保护的文本。成员关系推断攻击(MIA)，它确定给定的文本是否包括在模型的训练数据中，已经引起了人们的注意。以前对MIA的研究表明，基于似然分类的分类对于检测LLM中的泄漏是有效的。然而，现有的方法不能应用于一些专有模型，如ChatGPT或Claude 3，因为用户无法获得可能性。在这项研究中，我们提出了一种基于抽样的伪似然方法(\textbf{spl})，该方法仅使用LLM生成的文本来检测泄漏，计算SPL。SAMIA将目标文本作为参考文本，将LLM的多个输出作为文本样本，计算$n$-gram匹配度作为SPL，并确定文本在训练数据中的成员资格。即使没有可能性，萨米亚的表现也与现有的基于可能性的方法不相上下。



## **26. Humans or LLMs as the Judge? A Study on Judgement Biases**

人类还是法学硕士作为法官？判断偏差研究 cs.CL

22 pages

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2402.10669v3) [paper-pdf](http://arxiv.org/pdf/2402.10669v3)

**Authors**: Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, Benyou Wang

**Abstract**: Adopting human and large language models (LLM) as judges (\textit{a.k.a} human- and LLM-as-a-judge) for evaluating the performance of LLMs has recently gained attention. Nonetheless, this approach concurrently introduces potential biases from human and LLM judges, questioning the reliability of the evaluation results. In this paper, we propose a novel framework that is free from referencing groundtruth annotations for investigating Fallacy Oversight Bias, Authority Bias and Beauty Bias on LLM and human judges. We curate a dataset referring to the revised Bloom's Taxonomy and conduct thousands of human and LLM evaluations. Results show that human and LLM judges are vulnerable to perturbations to various degrees, and that even the cutting-edge judges possess considerable biases. We further exploit their weakness and conduct attacks on LLM judges. We hope that our work can notify the community of the vulnerability of human- and LLM-as-a-judge against perturbations, as well as the urgency of developing robust evaluation systems.

摘要: 采用人类和大语言模型(LLM)作为评判标准来评价LLMS的性能最近受到了人们的关注。尽管如此，这种方法同时引入了来自人类和LLM评委的潜在偏见，质疑评估结果的可靠性。在这篇文章中，我们提出了一个新的框架，该框架不引用基本事实注释来调查LLM和人类法官的谬误监督偏差、权威偏差和美貌偏差。我们参照修订后的Bloom分类法整理了一个数据集，并进行了数千次人类和LLM评估。结果表明，人类裁判和LLM裁判都不同程度地容易受到扰动的影响，即使是新锐裁判也存在相当大的偏差。我们进一步利用他们的弱点，对法律系法官进行攻击。我们希望我们的工作能够告知社会，人类和LLM作为法官面对扰动的脆弱性，以及制定强有力的评估系统的紧迫性。



## **27. TransLinkGuard: Safeguarding Transformer Models Against Model Stealing in Edge Deployment**

TransLinkGuard：保护Transformer模型，防止边缘部署中的模型窃取 cs.CR

arXiv admin note: text overlap with arXiv:2310.07152 by other authors

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2404.11121v1) [paper-pdf](http://arxiv.org/pdf/2404.11121v1)

**Authors**: Qinfeng Li, Zhiqiang Shen, Zhenghan Qin, Yangfan Xie, Xuhong Zhang, Tianyu Du, Jianwei Yin

**Abstract**: Proprietary large language models (LLMs) have been widely applied in various scenarios. Additionally, deploying LLMs on edge devices is trending for efficiency and privacy reasons. However, edge deployment of proprietary LLMs introduces new security challenges: edge-deployed models are exposed as white-box accessible to users, enabling adversaries to conduct effective model stealing (MS) attacks. Unfortunately, existing defense mechanisms fail to provide effective protection. Specifically, we identify four critical protection properties that existing methods fail to simultaneously satisfy: (1) maintaining protection after a model is physically copied; (2) authorizing model access at request level; (3) safeguarding runtime reverse engineering; (4) achieving high security with negligible runtime overhead. To address the above issues, we propose TransLinkGuard, a plug-and-play model protection approach against model stealing on edge devices. The core part of TransLinkGuard is a lightweight authorization module residing in a secure environment, e.g., TEE. The authorization module can freshly authorize each request based on its input. Extensive experiments show that TransLinkGuard achieves the same security protection as the black-box security guarantees with negligible overhead.

摘要: 专有的大型语言模型(LLM)已广泛应用于各种场景。此外，出于效率和隐私的原因，在边缘设备上部署LLM是一种趋势。然而，专有LLMS的边缘部署带来了新的安全挑战：边缘部署的模型暴露为用户可访问的白盒，使对手能够进行有效的模型窃取(MS)攻击。不幸的是，现有的防御机制未能提供有效的保护。具体地说，我们确定了现有方法无法同时满足的四个关键保护性质：(1)在物理复制模型后保持保护；(2)在请求级授权模型访问；(3)保护运行时逆向工程；(4)以可忽略的运行时开销实现高安全性。为了解决上述问题，我们提出了一种针对边缘设备上的模型窃取的即插即用模型保护方法TransLinkGuard。TransLinkGuard的核心部分是驻留在安全环境中的轻量级授权模块，例如TEE。授权模块可以基于其输入对每个请求进行新的授权。大量实验表明，TransLinkGuard实现了与黑盒安全保证相同的安全保护，而开销可以忽略不计。



## **28. Hidden You Malicious Goal Into Benign Narratives: Jailbreak Large Language Models through Logic Chain Injection**

将你的恶意目标隐藏到良性叙事中：通过逻辑链注入越狱大型语言模型 cs.CR

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2404.04849v2) [paper-pdf](http://arxiv.org/pdf/2404.04849v2)

**Authors**: Zhilong Wang, Yebo Cao, Peng Liu

**Abstract**: Jailbreak attacks on Language Model Models (LLMs) entail crafting prompts aimed at exploiting the models to generate malicious content. Existing jailbreak attacks can successfully deceive the LLMs, however they cannot deceive the human. This paper proposes a new type of jailbreak attacks which can deceive both the LLMs and human (i.e., security analyst). The key insight of our idea is borrowed from the social psychology - that is human are easily deceived if the lie is hidden in truth. Based on this insight, we proposed the logic-chain injection attacks to inject malicious intention into benign truth. Logic-chain injection attack firstly dissembles its malicious target into a chain of benign narrations, and then distribute narrations into a related benign article, with undoubted facts. In this way, newly generate prompt cannot only deceive the LLMs, but also deceive human.

摘要: 对语言模型模型（LLM）的越狱攻击需要精心设计旨在利用这些模型生成恶意内容的提示。现有的越狱攻击可以成功欺骗LLM，但它们无法欺骗人类。本文提出了一种新型越狱攻击，可以欺骗LLM和人类（即，安全分析师）。我们想法的关键见解借用了社会心理学--即如果谎言隐藏在真相中，人类就很容易被欺骗。基于这一认识，我们提出了逻辑链注入攻击，将恶意意图注入良性真相。逻辑链注入攻击首先将其恶意目标分解为一系列良性叙述，然后将叙述分发为相关的良性文章，并附有确凿的事实。这样，新生成的提示不仅可以欺骗LLM，还可以欺骗人类。



## **29. IsamasRed: A Public Dataset Tracking Reddit Discussions on Israel-Hamas Conflict**

IsamasRed：跟踪Reddit关于以色列与哈马斯冲突讨论的公共数据集 cs.SI

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2401.08202v2) [paper-pdf](http://arxiv.org/pdf/2401.08202v2)

**Authors**: Kai Chen, Zihao He, Keith Burghardt, Jingxin Zhang, Kristina Lerman

**Abstract**: The conflict between Israel and Palestinians significantly escalated after the October 7, 2023 Hamas attack, capturing global attention. To understand the public discourse on this conflict, we present a meticulously compiled dataset-IsamasRed-comprising nearly 400,000 conversations and over 8 million comments from Reddit, spanning from August 2023 to November 2023. We introduce an innovative keyword extraction framework leveraging a large language model to effectively identify pertinent keywords, ensuring a comprehensive data collection. Our initial analysis on the dataset, examining topics, controversy, emotional and moral language trends over time, highlights the emotionally charged and complex nature of the discourse. This dataset aims to enrich the understanding of online discussions, shedding light on the complex interplay between ideology, sentiment, and community engagement in digital spaces.

摘要: 2023年10月7日哈马斯袭击事件发生后，以色列和巴勒斯坦之间的冲突显着升级，引起了全球关注。为了了解公众对这场冲突的讨论，我们提供了一份精心编制的Inbox IsamasRed，其中包括Reddit的近40万次对话和超过800万条评论，时间跨度从2023年8月到2023年11月。我们引入了一个创新的关键词提取框架，利用大型语言模型来有效识别相关关键词，确保全面的数据收集。我们对数据集的初步分析，检查了话题、争议、情感和道德语言随着时间的推移趋势，凸显了话语的情感丰富和复杂性。该数据集旨在丰富对在线讨论的理解，揭示数字空间中意识形态、情绪和社区参与之间复杂的相互作用。



## **30. Self-playing Adversarial Language Game Enhances LLM Reasoning**

自玩对抗语言游戏增强LLM推理 cs.CL

Preprint

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2404.10642v1) [paper-pdf](http://arxiv.org/pdf/2404.10642v1)

**Authors**: Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Yong Dai, Lei Han, Nan Du

**Abstract**: We explore the self-play training procedure of large language models (LLMs) in a two-player adversarial language game called Adversarial Taboo. In this game, an attacker and a defender communicate with respect to a target word only visible to the attacker. The attacker aims to induce the defender to utter the target word unconsciously, while the defender tries to infer the target word from the attacker's utterances. To win the game, both players should have sufficient knowledge about the target word and high-level reasoning ability to infer and express in this information-reserved conversation. Hence, we are curious about whether LLMs' reasoning ability can be further enhanced by Self-Play in this Adversarial language Game (SPAG). With this goal, we let LLMs act as the attacker and play with a copy of itself as the defender on an extensive range of target words. Through reinforcement learning on the game outcomes, we observe that the LLMs' performance uniformly improves on a broad range of reasoning benchmarks. Furthermore, iteratively adopting this self-play process can continuously promote LLM's reasoning ability. The code is at https://github.com/Linear95/SPAG.

摘要: 我们探索了在一个名为对抗性禁忌的两人对抗性语言游戏中，大语言模型(LLM)的自我发挥训练过程。在这个游戏中，攻击者和防御者就只有攻击者才能看到的目标单词进行交流。攻击者的目的是诱导防御者无意识地说出目标词，而防御者则试图从攻击者的话语中推断出目标词。要赢得这场比赛，双方都应该有足够的目标词知识和高级推理能力，以便在这种信息储备的对话中进行推理和表达。因此，我们好奇在这场对抗性语言游戏(SPAG)中，LLMS的推理能力能否通过自我游戏进一步增强。有了这个目标，我们让LLMS扮演攻击者的角色，并在广泛的目标词上扮演自己的防御者。通过对游戏结果的强化学习，我们观察到LLMS在广泛的推理基准上的性能一致提高。此外，迭代地采用这种自我发挥过程可以不断提升LLM的推理能力。代码在https://github.com/Linear95/SPAG.



## **31. Topic-based Watermarks for LLM-Generated Text**

LLM生成文本的基于主题的水印 cs.CR

11 pages

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2404.02138v2) [paper-pdf](http://arxiv.org/pdf/2404.02138v2)

**Authors**: Alexander Nemecek, Yuzhou Jiang, Erman Ayday

**Abstract**: Recent advancements of large language models (LLMs) have resulted in indistinguishable text outputs comparable to human-generated text. Watermarking algorithms are potential tools that offer a way to differentiate between LLM- and human-generated text by embedding detectable signatures within LLM-generated output. However, current watermarking schemes lack robustness against known attacks against watermarking algorithms. In addition, they are impractical considering an LLM generates tens of thousands of text outputs per day and the watermarking algorithm needs to memorize each output it generates for the detection to work. In this work, focusing on the limitations of current watermarking schemes, we propose the concept of a "topic-based watermarking algorithm" for LLMs. The proposed algorithm determines how to generate tokens for the watermarked LLM output based on extracted topics of an input prompt or the output of a non-watermarked LLM. Inspired from previous work, we propose using a pair of lists (that are generated based on the specified extracted topic(s)) that specify certain tokens to be included or excluded while generating the watermarked output of the LLM. Using the proposed watermarking algorithm, we show the practicality of a watermark detection algorithm. Furthermore, we discuss a wide range of attacks that can emerge against watermarking algorithms for LLMs and the benefit of the proposed watermarking scheme for the feasibility of modeling a potential attacker considering its benefit vs. loss.

摘要: 最近大型语言模型(LLM)的进步导致了与人类生成的文本相比难以区分的文本输出。水印算法是一种潜在的工具，它通过在LLM生成的输出中嵌入可检测的签名来区分LLM生成的文本和人类生成的文本。然而，目前的水印方案对已知的针对水印算法的攻击缺乏稳健性。此外，考虑到LLM每天生成数万个文本输出，并且水印算法需要记住它生成的每个输出才能使检测工作，因此它们是不切实际的。在这项工作中，针对现有水印方案的局限性，我们提出了一种基于主题的LLMS水印算法的概念。该算法基于提取的输入提示主题或非水印LLM的输出主题，确定如何为带水印的LLM输出生成令牌。受以前工作的启发，我们建议使用一对列表(基于指定的提取主题(S)生成)，这些列表指定在生成LLM的水印输出时要包括或排除的某些标记。利用所提出的水印算法，我们展示了水印检测算法的实用性。此外，我们讨论了针对LLMS的水印算法可能出现的各种攻击，以及所提出的水印方案的好处，以考虑其利弊来对潜在攻击者进行建模的可行性。



## **32. Provably Robust Multi-bit Watermarking for AI-generated Text via Error Correction Code**

通过错误纠正代码对人工智能生成的文本进行可证明鲁棒的多位水印 cs.CR

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2401.16820v2) [paper-pdf](http://arxiv.org/pdf/2401.16820v2)

**Authors**: Wenjie Qu, Dong Yin, Zixin He, Wei Zou, Tianyang Tao, Jinyuan Jia, Jiaheng Zhang

**Abstract**: Large Language Models (LLMs) have been widely deployed for their remarkable capability to generate texts resembling human language. However, they could be misused by criminals to create deceptive content, such as fake news and phishing emails, which raises ethical concerns. Watermarking is a key technique to mitigate the misuse of LLMs, which embeds a watermark (e.g., a bit string) into a text generated by a LLM. Consequently, this enables the detection of texts generated by a LLM as well as the tracing of generated texts to a specific user. The major limitation of existing watermark techniques is that they cannot accurately or efficiently extract the watermark from a text, especially when the watermark is a long bit string. This key limitation impedes their deployment for real-world applications, e.g., tracing generated texts to a specific user.   This work introduces a novel watermarking method for LLM-generated text grounded in \textbf{error-correction codes} to address this challenge. We provide strong theoretical analysis, demonstrating that under bounded adversarial word/token edits (insertion, deletion, and substitution), our method can correctly extract watermarks, offering a provable robustness guarantee. This breakthrough is also evidenced by our extensive experimental results. The experiments show that our method substantially outperforms existing baselines in both accuracy and robustness on benchmark datasets. For instance, when embedding a bit string of length 12 into a 200-token generated text, our approach attains an impressive match rate of $98.4\%$, surpassing the performance of Yoo et al. (state-of-the-art baseline) at $85.6\%$. When subjected to a copy-paste attack involving the injection of 50 tokens to generated texts with 200 words, our method maintains a substantial match rate of $90.8\%$, while the match rate of Yoo et al. diminishes to below $65\%$.

摘要: 大型语言模型(LLM)因其生成类似人类语言的文本的非凡能力而被广泛使用。然而，它们可能被犯罪分子滥用来创造欺骗性内容，如假新闻和钓鱼电子邮件，这引发了伦理问题。水印是缓解LLMS误用的一项关键技术，它将水印(如比特串)嵌入到LLM生成的文本中。因此，这使得能够检测由LLM生成的文本以及将生成的文本跟踪到特定用户。现有水印技术的主要局限性是不能准确或高效地从文本中提取水印，特别是当水印是长比特串的时候。这一关键限制阻碍了它们在现实世界应用程序中的部署，例如，跟踪生成的文本到特定用户。为了解决这一问题，提出了一种新的基于文本纠错码的LLM文本水印方法。我们提供了强有力的理论分析，证明了在有界的敌意单词/令牌编辑(插入、删除和替换)下，我们的方法可以正确地提取水印，提供了可证明的健壮性保证。这一突破也被我们广泛的实验结果所证明。实验表明，在基准数据集上，我们的方法在准确率和稳健性方面都大大优于现有的基线。例如，当将长度为12的比特串嵌入到200个标记生成的文本中时，我们的方法获得了令人印象深刻的匹配率$98.4\$，超过了Yoo等人的性能。(最新基线)为85.6美元。在对200个单词的文本进行50个标记的复制粘贴攻击时，我们的方法保持了相当高的匹配率为90.8美元，而Yoo等人的匹配率是90.8美元。降至65美元以下。



## **33. FuzzLLM: A Novel and Universal Fuzzing Framework for Proactively Discovering Jailbreak Vulnerabilities in Large Language Models**

FuzzLLM：一个新颖且通用的Fuzzing框架，用于主动发现大型语言模型中的越狱漏洞 cs.CR

Publish by ICASSP 2024 on 3/18/2024; Extended Arxiv version

**SubmitDate**: 2024-04-14    [abs](http://arxiv.org/abs/2309.05274v2) [paper-pdf](http://arxiv.org/pdf/2309.05274v2)

**Authors**: Dongyu Yao, Jianshu Zhang, Ian G. Harris, Marcel Carlsson

**Abstract**: Jailbreak vulnerabilities in Large Language Models (LLMs), which exploit meticulously crafted prompts to elicit content that violates service guidelines, have captured the attention of research communities. While model owners can defend against individual jailbreak prompts through safety training strategies, this relatively passive approach struggles to handle the broader category of similar jailbreaks. To tackle this issue, we introduce FuzzLLM, an automated fuzzing framework designed to proactively test and discover jailbreak vulnerabilities in LLMs. We utilize templates to capture the structural integrity of a prompt and isolate key features of a jailbreak class as constraints. By integrating different base classes into powerful combo attacks and varying the elements of constraints and prohibited questions, FuzzLLM enables efficient testing with reduced manual effort. Extensive experiments demonstrate FuzzLLM's effectiveness and comprehensiveness in vulnerability discovery across various LLMs.

摘要: 大型语言模型(LLM)中的越狱漏洞利用精心设计的提示来引出违反服务指南的内容，已经引起了研究界的注意。虽然模型所有者可以通过安全培训策略来防御个别越狱提示，但这种相对被动的方法难以应对更广泛类别的类似越狱。为了解决这个问题，我们引入了FuzzLLM，这是一个自动模糊框架，旨在主动测试和发现LLM中的越狱漏洞。我们利用模板来捕获提示符的结构完整性，并将越狱类的关键特性隔离为约束。通过将不同的基类集成到强大的组合攻击中，并改变约束和禁止问题的元素，FuzzLLM能够以更少的手动工作实现高效的测试。广泛的实验证明了FuzzLLM在跨各种LLM发现漏洞方面的有效性和全面性。



## **34. Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models**

图像是一致的阿喀琉斯之踵：利用视觉漏洞破解多模式大型语言模型 cs.CV

Work in progress

**SubmitDate**: 2024-04-14    [abs](http://arxiv.org/abs/2403.09792v2) [paper-pdf](http://arxiv.org/pdf/2403.09792v2)

**Authors**: Yifan Li, Hangyu Guo, Kun Zhou, Wayne Xin Zhao, Ji-Rong Wen

**Abstract**: In this paper, we study the harmlessness alignment problem of multimodal large language models (MLLMs). We conduct a systematic empirical analysis of the harmlessness performance of representative MLLMs and reveal that the image input poses the alignment vulnerability of MLLMs. Inspired by this, we propose a novel jailbreak method named HADES, which hides and amplifies the harmfulness of the malicious intent within the text input, using meticulously crafted images. Experimental results show that HADES can effectively jailbreak existing MLLMs, which achieves an average Attack Success Rate (ASR) of 90.26% for LLaVA-1.5 and 71.60% for Gemini Pro Vision. Our code and data will be publicly released.

摘要: 本文研究了多模式大型语言模型（MLLM）的无害对齐问题。我们对代表性MLLM的无害性能进行了系统的实证分析，揭示了图像输入造成了MLLM的对齐脆弱性。受此启发，我们提出了一种名为HADES的新颖越狱方法，该方法使用精心制作的图像隐藏和放大文本输入中恶意意图的危害性。实验结果表明，HADES可以有效越狱现有的MLLM，LLaVA-1.5的平均攻击成功率（ASB）为90.26%，Gemini Pro Vision的平均攻击成功率（ASB）为71.60%。我们的代码和数据将公开发布。



## **35. Detoxifying Large Language Models via Knowledge Editing**

通过知识编辑消除大型语言模型的神秘性 cs.CL

Ongoing work. Project website:  https://zjunlp.github.io/project/SafeEdit Add and update experimental results  in Tables 1 and 3

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2403.14472v3) [paper-pdf](http://arxiv.org/pdf/2403.14472v3)

**Authors**: Mengru Wang, Ningyu Zhang, Ziwen Xu, Zekun Xi, Shumin Deng, Yunzhi Yao, Qishen Zhang, Linyi Yang, Jindong Wang, Huajun Chen

**Abstract**: This paper investigates using knowledge editing techniques to detoxify Large Language Models (LLMs). We construct a benchmark, SafeEdit, which covers nine unsafe categories with various powerful attack prompts and equips comprehensive metrics for systematic evaluation. We conduct experiments with several knowledge editing approaches, indicating that knowledge editing has the potential to efficiently detoxify LLMs with limited impact on general performance. Then, we propose a simple yet effective baseline, dubbed Detoxifying with Intraoperative Neural Monitoring (DINM), to diminish the toxicity of LLMs within a few tuning steps via only one instance. We further provide an in-depth analysis of the internal mechanism for various detoxifying approaches, demonstrating that previous methods like SFT and DPO may merely suppress the activations of toxic parameters, while DINM mitigates the toxicity of the toxic parameters to a certain extent, making permanent adjustments. We hope that these insights could shed light on future work of developing detoxifying approaches and the underlying knowledge mechanisms of LLMs. Code and benchmark are available at https://github.com/zjunlp/EasyEdit.

摘要: 本文研究了利用知识编辑技术对大型语言模型进行去毒处理。我们构建了一个涵盖9个不安全类别、具有各种强大的攻击提示的基准SafeEdit，并配备了全面的度量来进行系统评估。我们用几种知识编辑方法进行了实验，表明知识编辑有可能在对一般性能影响有限的情况下有效地对LLM进行解毒。然后，我们提出了一个简单而有效的基线，称为术中神经监测解毒(DINM)，仅通过一个实例在几个调整步骤内降低LLMS的毒性。我们进一步深入分析了各种解毒方法的内在机制，证明了以前的方法如SFT和DPO可能只是抑制了毒性参数的激活，而DINM在一定程度上减轻了毒性参数的毒性，做出了永久性的调整。我们希望这些洞察力能够为未来开发戒毒方法的工作和LLMS的潜在知识机制提供帮助。代码和基准测试可在https://github.com/zjunlp/EasyEdit.上获得



## **36. JailbreakLens: Visual Analysis of Jailbreak Attacks Against Large Language Models**

越狱镜头：针对大型语言模型的越狱攻击的视觉分析 cs.CR

Submitted to VIS 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08793v1) [paper-pdf](http://arxiv.org/pdf/2404.08793v1)

**Authors**: Yingchaojie Feng, Zhizhang Chen, Zhining Kang, Sijia Wang, Minfeng Zhu, Wei Zhang, Wei Chen

**Abstract**: The proliferation of large language models (LLMs) has underscored concerns regarding their security vulnerabilities, notably against jailbreak attacks, where adversaries design jailbreak prompts to circumvent safety mechanisms for potential misuse. Addressing these concerns necessitates a comprehensive analysis of jailbreak prompts to evaluate LLMs' defensive capabilities and identify potential weaknesses. However, the complexity of evaluating jailbreak performance and understanding prompt characteristics makes this analysis laborious. We collaborate with domain experts to characterize problems and propose an LLM-assisted framework to streamline the analysis process. It provides automatic jailbreak assessment to facilitate performance evaluation and support analysis of components and keywords in prompts. Based on the framework, we design JailbreakLens, a visual analysis system that enables users to explore the jailbreak performance against the target model, conduct multi-level analysis of prompt characteristics, and refine prompt instances to verify findings. Through a case study, technical evaluations, and expert interviews, we demonstrate our system's effectiveness in helping users evaluate model security and identify model weaknesses.

摘要: 大型语言模型(LLM)的激增突显了人们对其安全漏洞的担忧，特别是针对越狱攻击的担忧，在越狱攻击中，对手设计越狱提示以绕过安全机制，防止潜在的滥用。解决这些问题需要对越狱提示进行全面分析，以评估LLMS的防御能力并确定潜在的弱点。然而，评估越狱性能和理解提示特征的复杂性使得这一分析变得费力。我们与领域专家合作来表征问题，并提出一个LLM辅助的框架来简化分析过程。它提供自动越狱评估，以便于对提示中的组件和关键字进行性能评估和支持分析。基于该框架，我们设计了一个可视化分析系统JailBreakLens，使用户能够针对目标模型探索越狱性能，对提示特征进行多层次分析，并对提示实例进行精化以验证结果。通过案例研究、技术评估和专家访谈，我们展示了我们的系统在帮助用户评估模型安全性和识别模型弱点方面的有效性。



## **37. Subtoxic Questions: Dive Into Attitude Change of LLM's Response in Jailbreak Attempts**

亚毒性问题：深入探讨LLM在越狱尝试中回应的态度变化 cs.CR

4 pages, 2 figures. This paper was submitted to The 7th Deep Learning  Security and Privacy Workshop (DLSP 2024) and was accepted as extended  abstract, see https://dlsp2024.ieee-security.org/

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08309v1) [paper-pdf](http://arxiv.org/pdf/2404.08309v1)

**Authors**: Tianyu Zhang, Zixuan Zhao, Jiaqi Huang, Jingyu Hua, Sheng Zhong

**Abstract**: As Large Language Models (LLMs) of Prompt Jailbreaking are getting more and more attention, it is of great significance to raise a generalized research paradigm to evaluate attack strengths and a basic model to conduct subtler experiments. In this paper, we propose a novel approach by focusing on a set of target questions that are inherently more sensitive to jailbreak prompts, aiming to circumvent the limitations posed by enhanced LLM security. Through designing and analyzing these sensitive questions, this paper reveals a more effective method of identifying vulnerabilities in LLMs, thereby contributing to the advancement of LLM security. This research not only challenges existing jailbreaking methodologies but also fortifies LLMs against potential exploits.

摘要: 随着即时越狱的大型语言模型（LLM）越来越受到关注，提出评估攻击强度的广义研究范式和进行更微妙实验的基本模型具有重要意义。在本文中，我们提出了一种新颖的方法，重点关注一组对越狱提示本质上更敏感的目标问题，旨在规避增强的LLM安全性带来的限制。通过设计和分析这些敏感问题，本文揭示了一种更有效的识别LLM漏洞的方法，从而为LLM安全性的进步做出贡献。这项研究不仅挑战了现有的越狱方法，而且还增强了LLM免受潜在的攻击。



## **38. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

AmpleGCG：学习通用且可转移的对抗性后缀生成模型，用于越狱开放和封闭LLM cs.CL

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07921v1) [paper-pdf](http://arxiv.org/pdf/2404.07921v1)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.

摘要: 随着大型语言模型(LLM)变得越来越普遍并集成到自治系统中，确保它们的安全性是当务之急。尽管在安全对齐方面取得了长足的进步，但最近的工作GCG~\Citep{zou2023Universal}提出了一种离散令牌优化算法，并选择损失最小的单个后缀来成功越狱对齐LLM。在这项工作中，我们首先讨论了在GCG优化越狱过程中只选择损失最小的后缀的缺点，并在中间步骤中发现了遗漏的成功后缀。此外，我们利用这些成功的后缀作为训练数据来学习一种名为AmpleGCG的生成模型，该模型捕获给定有害查询的对抗性后缀的分布，并在几秒钟内为任何有害查询快速生成数百个后缀。AmpleGCG在两个对齐的LLM(Llama-2-7B-Chat和Vicuna-7B)上达到了近100%的攻击成功率(ASR)，超过了两个最强的攻击基线。更有趣的是，AmpleGCG还无缝传输以攻击不同的型号，包括闭源LLMS，在最新的GPT-3.5上实现了99\%的ASR。总之，我们的工作通过训练对抗性后缀的生成模型来放大GCG的影响，该模型对任何有害的查询都是通用的，并且可以从攻击开源LLM转移到闭源LLM。此外，它可以在短短4秒内为一个有害的查询生成200个敌意后缀，使其更具挑战性。



## **39. AnnoCTR: A Dataset for Detecting and Linking Entities, Tactics, and Techniques in Cyber Threat Reports**

AnnoCTF：用于检测和链接网络威胁报告中的实体、策略和技术的数据集 cs.CL

Accepted at LREC-COLING 2024. Corpus available at  https://github.com/boschresearch/anno-ctr-lrec-coling-2024

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07765v1) [paper-pdf](http://arxiv.org/pdf/2404.07765v1)

**Authors**: Lukas Lange, Marc Müller, Ghazaleh Haratinezhad Torbati, Dragan Milchevski, Patrick Grau, Subhash Pujari, Annemarie Friedrich

**Abstract**: Monitoring the threat landscape to be aware of actual or potential attacks is of utmost importance to cybersecurity professionals. Information about cyber threats is typically distributed using natural language reports. Natural language processing can help with managing this large amount of unstructured information, yet to date, the topic has received little attention. With this paper, we present AnnoCTR, a new CC-BY-SA-licensed dataset of cyber threat reports. The reports have been annotated by a domain expert with named entities, temporal expressions, and cybersecurity-specific concepts including implicitly mentioned techniques and tactics. Entities and concepts are linked to Wikipedia and the MITRE ATT&CK knowledge base, the most widely-used taxonomy for classifying types of attacks. Prior datasets linking to MITRE ATT&CK either provide a single label per document or annotate sentences out-of-context; our dataset annotates entire documents in a much finer-grained way. In an experimental study, we model the annotations of our dataset using state-of-the-art neural models. In our few-shot scenario, we find that for identifying the MITRE ATT&CK concepts that are mentioned explicitly or implicitly in a text, concept descriptions from MITRE ATT&CK are an effective source for training data augmentation.

摘要: 监控威胁环境以了解实际或潜在的攻击对网络安全专业人员来说至关重要。有关网络威胁的信息通常使用自然语言报告发布。自然语言处理可以帮助管理这些大量的非结构化信息，但到目前为止，这个主题几乎没有受到关注。在本文中，我们介绍了AnnoCTR，一个新的CC-by-SA许可的网络威胁报告数据集。这些报告已经由领域专家使用命名实体、时间表达式和网络安全特定概念(包括隐含提到的技术和战术)进行了注释。实体和概念链接到维基百科和MITRE ATT&CK知识库，后者是对攻击类型进行分类的最广泛使用的分类法。以前链接到MITRE ATT&CK的数据集要么为每个文档提供单一标签，要么断章取义地注释句子；我们的数据集以更细粒度的方式注释整个文档。在一项实验研究中，我们使用最先进的神经模型对数据集的注释进行建模。在我们的少数情况下，我们发现，对于识别文本中明确或隐含地提到的MITRE ATT&CK概念，MITRE ATT&CK的概念描述是训练数据增强的有效来源。



## **40. Sandwich attack: Multi-language Mixture Adaptive Attack on LLMs**

三明治攻击：对LLM的多语言混合自适应攻击 cs.CR

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.07242v1) [paper-pdf](http://arxiv.org/pdf/2404.07242v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan

**Abstract**: Large Language Models (LLMs) are increasingly being developed and applied, but their widespread use faces challenges. These include aligning LLMs' responses with human values to prevent harmful outputs, which is addressed through safety training methods. Even so, bad actors and malicious users have succeeded in attempts to manipulate the LLMs to generate misaligned responses for harmful questions such as methods to create a bomb in school labs, recipes for harmful drugs, and ways to evade privacy rights. Another challenge is the multilingual capabilities of LLMs, which enable the model to understand and respond in multiple languages. Consequently, attackers exploit the unbalanced pre-training datasets of LLMs in different languages and the comparatively lower model performance in low-resource languages than high-resource ones. As a result, attackers use a low-resource languages to intentionally manipulate the model to create harmful responses. Many of the similar attack vectors have been patched by model providers, making the LLMs more robust against language-based manipulation. In this paper, we introduce a new black-box attack vector called the \emph{Sandwich attack}: a multi-language mixture attack, which manipulates state-of-the-art LLMs into generating harmful and misaligned responses. Our experiments with five different models, namely Google's Bard, Gemini Pro, LLaMA-2-70-B-Chat, GPT-3.5-Turbo, GPT-4, and Claude-3-OPUS, show that this attack vector can be used by adversaries to generate harmful responses and elicit misaligned responses from these models. By detailing both the mechanism and impact of the Sandwich attack, this paper aims to guide future research and development towards more secure and resilient LLMs, ensuring they serve the public good while minimizing potential for misuse.

摘要: 大型语言模型(LLM)的开发和应用越来越多，但它们的广泛使用面临着挑战。这些措施包括使LLMS的反应与人的价值观相一致，以防止有害的输出，这是通过安全培训方法解决的。尽管如此，不良行为者和恶意用户仍成功地操纵LLMS，以生成对有害问题的错误响应，这些问题包括在学校实验室制造炸弹的方法、有害药物的配方以及逃避隐私权的方法。另一个挑战是LLMS的多语言能力，这使得该模型能够理解并以多种语言响应。因此，攻击者利用不同语言的LLMS的不平衡的预训练数据集，以及低资源语言的模型性能相对较低的高资源语言。因此，攻击者使用低资源语言来故意操纵模型以创建有害的响应。许多类似的攻击载体已经由模型提供商修补，使LLM对基于语言的操纵更加健壮。本文介绍了一种新的黑盒攻击向量--夹心攻击：一种多语言混合攻击，它操纵最先进的LLM产生有害的和未对齐的响应。我们对谷歌的Bard、Gemini Pro、Llama-2-70-B-Chat、GPT-3.5-Turbo、GPT-4和Claude-3-opus这五个不同的模型进行的实验表明，该攻击向量可被攻击者用来生成有害响应并从这些模型中引发错误的响应。通过详细描述三明治攻击的机制和影响，本文旨在引导未来的研究和开发朝着更安全和更具弹性的方向发展，确保它们服务于公共利益，同时将滥用的可能性降至最低。



## **41. $\textit{LinkPrompt}$: Natural and Universal Adversarial Attacks on Prompt-based Language Models**

$\texttit {LinkPrompt}$：对基于预算的语言模型的自然和普遍对抗攻击 cs.CL

Accepted to the main conference of NAACL2024

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2403.16432v3) [paper-pdf](http://arxiv.org/pdf/2403.16432v3)

**Authors**: Yue Xu, Wenjie Wang

**Abstract**: Prompt-based learning is a new language model training paradigm that adapts the Pre-trained Language Models (PLMs) to downstream tasks, which revitalizes the performance benchmarks across various natural language processing (NLP) tasks. Instead of using a fixed prompt template to fine-tune the model, some research demonstrates the effectiveness of searching for the prompt via optimization. Such prompt optimization process of prompt-based learning on PLMs also gives insight into generating adversarial prompts to mislead the model, raising concerns about the adversarial vulnerability of this paradigm. Recent studies have shown that universal adversarial triggers (UATs) can be generated to alter not only the predictions of the target PLMs but also the prediction of corresponding Prompt-based Fine-tuning Models (PFMs) under the prompt-based learning paradigm. However, UATs found in previous works are often unreadable tokens or characters and can be easily distinguished from natural texts with adaptive defenses. In this work, we consider the naturalness of the UATs and develop $\textit{LinkPrompt}$, an adversarial attack algorithm to generate UATs by a gradient-based beam search algorithm that not only effectively attacks the target PLMs and PFMs but also maintains the naturalness among the trigger tokens. Extensive results demonstrate the effectiveness of $\textit{LinkPrompt}$, as well as the transferability of UATs generated by $\textit{LinkPrompt}$ to open-sourced Large Language Model (LLM) Llama2 and API-accessed LLM GPT-3.5-turbo. The resource is available at $\href{https://github.com/SavannahXu79/LinkPrompt}{https://github.com/SavannahXu79/LinkPrompt}$.

摘要: 基于提示的学习是一种新的语言模型训练范式，它使预先训练的语言模型(PLM)适应于下游任务，从而重振各种自然语言处理(NLP)任务的表现基准。一些研究证明了通过优化来搜索提示的有效性，而不是使用固定的提示模板来微调模型。这种基于提示的PLM学习的快速优化过程也为生成对抗性提示以误导模型提供了洞察力，这引发了人们对这种范式的对抗性脆弱性的担忧。最近的研究表明，在基于提示的学习范式下，通用对抗触发器(UAT)不仅可以改变目标PLM的预测，还可以改变相应的基于提示的精调模型(PFM)的预测。然而，在以前的著作中发现的UAT通常是不可读的符号或字符，并且可以很容易地与具有自适应防御的自然文本区分开来。在这项工作中，我们考虑了UAT的自然性，并开发了一种对抗性攻击算法，通过基于梯度的波束搜索算法来生成UAT，该算法不仅有效地攻击了目标PLM和PPM，而且保持了触发令牌之间的自然度。广泛的结果证明了$\textit{LinkPrompt}$的有效性，以及由$\textit{LinkPrompt}$生成的UAT可以移植到开源的大型语言模型(LLM)Llama2和API访问的LLm GPT-3.5-Turbo。该资源可在$\href{https://github.com/SavannahXu79/LinkPrompt}{https://github.com/SavannahXu79/LinkPrompt}$.上获得



## **42. AEGIS: Online Adaptive AI Content Safety Moderation with Ensemble of LLM Experts**

AEGIS：与LLM专家一起进行在线自适应人工智能内容安全审核 cs.LG

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.05993v1) [paper-pdf](http://arxiv.org/pdf/2404.05993v1)

**Authors**: Shaona Ghosh, Prasoon Varshney, Erick Galinkin, Christopher Parisien

**Abstract**: As Large Language Models (LLMs) and generative AI become more widespread, the content safety risks associated with their use also increase. We find a notable deficiency in high-quality content safety datasets and benchmarks that comprehensively cover a wide range of critical safety areas. To address this, we define a broad content safety risk taxonomy, comprising 13 critical risk and 9 sparse risk categories. Additionally, we curate AEGISSAFETYDATASET, a new dataset of approximately 26, 000 human-LLM interaction instances, complete with human annotations adhering to the taxonomy. We plan to release this dataset to the community to further research and to help benchmark LLM models for safety. To demonstrate the effectiveness of the dataset, we instruction-tune multiple LLM-based safety models. We show that our models (named AEGISSAFETYEXPERTS), not only surpass or perform competitively with the state-of-the-art LLM-based safety models and general purpose LLMs, but also exhibit robustness across multiple jail-break attack categories. We also show how using AEGISSAFETYDATASET during the LLM alignment phase does not negatively impact the performance of the aligned models on MT Bench scores. Furthermore, we propose AEGIS, a novel application of a no-regret online adaptation framework with strong theoretical guarantees, to perform content moderation with an ensemble of LLM content safety experts in deployment

摘要: 随着大型语言模型(LLM)和生成式人工智能变得更加普遍，与使用它们相关的内容安全风险也增加了。我们发现，在全面覆盖广泛关键安全领域的高质量内容安全数据集和基准方面存在明显不足。为了解决这一问题，我们定义了一个广泛的内容安全风险分类，包括13个关键风险和9个稀疏风险类别。此外，我们还管理了AEGISSAFETYDATASET，这是一个大约包含26,000个人与LLM交互实例的新数据集，其中包含符合分类的人类注释。我们计划向社区发布这个数据集，以进行进一步的研究，并帮助对LLM模型的安全性进行基准测试。为了证明数据集的有效性，我们对多个基于LLM的安全模型进行了指令调整。我们证明了我们的模型(命名为AEGISSAFETYEXPERTS)不仅性能优于最先进的基于LLM的安全模型和通用LLM，而且在多个越狱攻击类别中表现出健壮性。我们还展示了在LLM校准阶段使用AEGISSAFETYDATASET如何不会对MT板凳分数上的校准模型的性能产生负面影响。此外，我们提出了Aegis，一个具有强大理论保证的无遗憾在线适配框架的新应用，在部署时使用LLM内容安全专家集成来执行内容审核



## **43. Eraser: Jailbreaking Defense in Large Language Models via Unlearning Harmful Knowledge**

橡皮擦：通过忘记有害知识在大型语言模型中进行越狱防御 cs.CL

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05880v1) [paper-pdf](http://arxiv.org/pdf/2404.05880v1)

**Authors**: Weikai Lu, Ziqian Zeng, Jianwei Wang, Zhengdong Lu, Zelin Chen, Huiping Zhuang, Cen Chen

**Abstract**: Jailbreaking attacks can enable Large Language Models (LLMs) to bypass the safeguard and generate harmful content. Existing jailbreaking defense methods have failed to address the fundamental issue that harmful knowledge resides within the model, leading to potential jailbreak risks for LLMs. In this paper, we propose a novel defense method called Eraser, which mainly includes three goals: unlearning harmful knowledge, retaining general knowledge, and maintaining safety alignment. The intuition is that if an LLM forgets the specific knowledge required to answer a harmful question, it will no longer have the ability to answer harmful questions. The training of Erase does not actually require the model's own harmful knowledge, and it can benefit from unlearning general answers related to harmful queries, which means it does not need assistance from the red team. The experimental results show that Eraser can significantly reduce the jailbreaking success rate for various attacks without compromising the general capabilities of the model.

摘要: 越狱攻击可使大型语言模型(LLM)绕过安全保护并生成有害内容。现有的越狱防御方法未能解决有害知识驻留在模型中的根本问题，导致低收入国家存在潜在的越狱风险。在本文中，我们提出了一种新的防御方法--橡皮擦，它主要包括三个目标：忘记有害知识，保留一般知识，保持安全对齐。直觉是，如果LLM忘记了回答有害问题所需的特定知识，它将不再有能力回答有害问题。ERASE的训练实际上并不需要模型本身的有害知识，而且它可以受益于忘记与有害查询相关的一般答案，这意味着它不需要红色团队的帮助。实验结果表明，在不影响模型整体性能的前提下，橡皮擦能显著降低各种攻击的越狱成功率。



## **44. Exploring the Deceptive Power of LLM-Generated Fake News: A Study of Real-World Detection Challenges**

探索LLM生成的假新闻的欺骗力量：对现实世界检测挑战的研究 cs.CL

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2403.18249v2) [paper-pdf](http://arxiv.org/pdf/2403.18249v2)

**Authors**: Yanshen Sun, Jianfeng He, Limeng Cui, Shuo Lei, Chang-Tien Lu

**Abstract**: Recent advancements in Large Language Models (LLMs) have enabled the creation of fake news, particularly in complex fields like healthcare. Studies highlight the gap in the deceptive power of LLM-generated fake news with and without human assistance, yet the potential of prompting techniques has not been fully explored. Thus, this work aims to determine whether prompting strategies can effectively narrow this gap. Current LLM-based fake news attacks require human intervention for information gathering and often miss details and fail to maintain context consistency. Therefore, to better understand threat tactics, we propose a strong fake news attack method called conditional Variational-autoencoder-Like Prompt (VLPrompt). Unlike current methods, VLPrompt eliminates the need for additional data collection while maintaining contextual coherence and preserving the intricacies of the original text. To propel future research on detecting VLPrompt attacks, we created a new dataset named VLPrompt fake news (VLPFN) containing real and fake texts. Our experiments, including various detection methods and novel human study metrics, were conducted to assess their performance on our dataset, yielding numerous findings.

摘要: 最近大型语言模型(LLM)的进步使假新闻的创造成为可能，特别是在医疗保健等复杂领域。研究突显了在有无人工帮助的情况下，LLM生成的假新闻的欺骗力存在差距，但提示技术的潜力尚未得到充分挖掘。因此，本研究旨在确定激励策略是否能有效缩小这一差距。目前基于LLM的假新闻攻击需要人工干预来收集信息，并且经常错过细节，无法保持上下文一致性。因此，为了更好地理解威胁策略，我们提出了一种强大的假新闻攻击方法，称为条件变分自动编码式提示(VLPrompt)。与目前的方法不同，VLPrompt消除了对额外数据收集的需要，同时保持了上下文的连贯性和原始文本的错综复杂。为了推动未来对VLPrompt攻击检测的研究，我们创建了一个新的数据集，名为VLPrompt假新闻(VLPFN)，包含真实和虚假的文本。我们的实验，包括各种检测方法和新的人体研究指标，被用来评估它们在我们的数据集上的性能，产生了许多发现。



## **45. Best-of-Venom: Attacking RLHF by Injecting Poisoned Preference Data**

毒液最佳：通过注入有毒偏好数据来攻击WLHF cs.CL

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05530v1) [paper-pdf](http://arxiv.org/pdf/2404.05530v1)

**Authors**: Tim Baumgärtner, Yang Gao, Dana Alon, Donald Metzler

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is a popular method for aligning Language Models (LM) with human values and preferences. RLHF requires a large number of preference pairs as training data, which are often used in both the Supervised Fine-Tuning and Reward Model training, and therefore publicly available datasets are commonly used. In this work, we study to what extent a malicious actor can manipulate the LMs generations by poisoning the preferences, i.e., injecting poisonous preference pairs into these datasets and the RLHF training process. We propose strategies to build poisonous preference pairs and test their performance by poisoning two widely used preference datasets. Our results show that preference poisoning is highly effective: by injecting a small amount of poisonous data (1-5% of the original dataset), we can effectively manipulate the LM to generate a target entity in a target sentiment (positive or negative). The findings from our experiments also shed light on strategies to defend against the preference poisoning attack.

摘要: 人类反馈强化学习(RLHF)是一种使语言模型与人类价值观和偏好保持一致的流行方法。RLHF需要大量的偏好对作为训练数据，这些数据经常用于有监督的精调和奖励模型训练，因此通常使用公开可用的数据集。在这项工作中，我们研究了恶意行为者可以在多大程度上通过毒化偏好来操纵LMS生成，即向这些数据集注入有毒的偏好对和RLHF训练过程。我们提出了构建有毒偏好对的策略，并通过毒化两个广泛使用的偏好数据集来测试它们的性能。我们的结果表明偏好毒化是非常有效的：通过注入少量的有毒数据(原始数据集的1-5%)，我们可以有效地操纵LM来生成目标情感中的目标实体(积极或消极)。我们的实验结果也为防御偏好中毒攻击的策略提供了启示。



## **46. Unbridled Icarus: A Survey of the Potential Perils of Image Inputs in Multimodal Large Language Model Security**

无拘无束的伊卡洛斯：图像输入在多模式大型语言模型安全中的潜在危险调查 cs.CR

8 pages, 1 figure

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05264v1) [paper-pdf](http://arxiv.org/pdf/2404.05264v1)

**Authors**: Yihe Fan, Yuxin Cao, Ziyu Zhao, Ziyao Liu, Shaofeng Li

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate remarkable capabilities that increasingly influence various aspects of our daily lives, constantly defining the new boundary of Artificial General Intelligence (AGI). Image modalities, enriched with profound semantic information and a more continuous mathematical nature compared to other modalities, greatly enhance the functionalities of MLLMs when integrated. However, this integration serves as a double-edged sword, providing attackers with expansive vulnerabilities to exploit for highly covert and harmful attacks. The pursuit of reliable AI systems like powerful MLLMs has emerged as a pivotal area of contemporary research. In this paper, we endeavor to demostrate the multifaceted risks associated with the incorporation of image modalities into MLLMs. Initially, we delineate the foundational components and training processes of MLLMs. Subsequently, we construct a threat model, outlining the security vulnerabilities intrinsic to MLLMs. Moreover, we analyze and summarize existing scholarly discourses on MLLMs' attack and defense mechanisms, culminating in suggestions for the future research on MLLM security. Through this comprehensive analysis, we aim to deepen the academic understanding of MLLM security challenges and propel forward the development of trustworthy MLLM systems.

摘要: 多通道大语言模型(MLLMS)显示出非凡的能力，越来越多地影响我们日常生活的各个方面，不断定义人工通用智能(AGI)的新边界。与其他模式相比，图像模式具有更丰富的语义信息和更连续的数学性质，极大地增强了MLLMS的功能。然而，这种集成是一把双刃剑，为攻击者提供了大量漏洞，可以利用这些漏洞进行高度隐蔽和有害的攻击。像强大的MLLMS这样可靠的人工智能系统的追求已经成为当代研究的一个关键领域。在这篇文章中，我们努力展示与将图像模式结合到MLLMS中相关的多方面风险。首先，我们描述了MLLMS的基本组成部分和培训过程。随后，我们构建了威胁模型，概述了MLLMS固有的安全漏洞。此外，我们分析和总结了已有的关于MLLMS攻击和防御机制的学术论述，并对未来的MLLMS安全研究提出了建议。通过这一综合分析，我们旨在加深对MLLM安全挑战的学术认识，推动可信MLLM系统的发展。



## **47. REMARK-LLM: A Robust and Efficient Watermarking Framework for Generative Large Language Models**

REMARK-LLM：用于生成性大型语言模型的稳健有效的水印框架 cs.CR

accept to usenix security 2024

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2310.12362v2) [paper-pdf](http://arxiv.org/pdf/2310.12362v2)

**Authors**: Ruisi Zhang, Shehzeen Samarah Hussain, Paarth Neekhara, Farinaz Koushanfar

**Abstract**: We present REMARK-LLM, a novel efficient, and robust watermarking framework designed for texts generated by large language models (LLMs). Synthesizing human-like content using LLMs necessitates vast computational resources and extensive datasets, encapsulating critical intellectual property (IP). However, the generated content is prone to malicious exploitation, including spamming and plagiarism. To address the challenges, REMARK-LLM proposes three new components: (i) a learning-based message encoding module to infuse binary signatures into LLM-generated texts; (ii) a reparameterization module to transform the dense distributions from the message encoding to the sparse distribution of the watermarked textual tokens; (iii) a decoding module dedicated for signature extraction; Furthermore, we introduce an optimized beam search algorithm to guarantee the coherence and consistency of the generated content. REMARK-LLM is rigorously trained to encourage the preservation of semantic integrity in watermarked content, while ensuring effective watermark retrieval. Extensive evaluations on multiple unseen datasets highlight REMARK-LLM proficiency and transferability in inserting 2 times more signature bits into the same texts when compared to prior art, all while maintaining semantic integrity. Furthermore, REMARK-LLM exhibits better resilience against a spectrum of watermark detection and removal attacks.

摘要: 我们提出了一种新的高效、健壮的数字水印框架--REMARK-LLM，它适用于大型语言模型(LLMS)生成的文本。使用LLMS合成类似人类的内容需要大量的计算资源和大量的数据集，并封装关键知识产权(IP)。然而，生成的内容容易受到恶意攻击，包括垃圾邮件和抄袭。为了应对这些挑战，REMARK-LLM提出了三个新的组件：(I)基于学习的消息编码模块，将二进制签名注入到LLM生成的文本中；(Ii)重新参数化模块，将密集分布的消息编码转换为稀疏分布的水印文本标记；(Iii)专门用于签名提取的解码模块；此外，我们还引入了优化的波束搜索算法，以保证生成内容的一致性和一致性。Rmark-LLM经过了严格的培训，以鼓励在水印内容中保持语义完整性，同时确保有效的水印检索。对多个不可见数据集的广泛评估突出了Remmark-LLM的熟练程度和可转移性，即与现有技术相比，在相同的文本中插入2倍多的签名比特，同时保持语义完整性。此外，REMARK-LLM对一系列水印检测和删除攻击表现出更好的弹性。



## **48. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

披着羊皮的狼：广义嵌套越狱预言可以轻松愚弄大型语言模型 cs.CL

Acccepted by NAACL 2024, 18 pages, 7 figures, 13 tables

**SubmitDate**: 2024-04-07    [abs](http://arxiv.org/abs/2311.08268v4) [paper-pdf](http://arxiv.org/pdf/2311.08268v4)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, which compromises either generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.

摘要: 大型语言模型(LLM)，如ChatGPT和GPT-4，旨在提供有用和安全的响应。然而，被称为“越狱”的对抗性提示可能会绕过安全措施，导致LLMS生成潜在的有害内容。探索越狱提示可以帮助更好地揭示LLM的弱点，并进一步指导我们确保它们的安全。不幸的是，现有的越狱方法要么需要复杂的人工设计，要么需要对其他白盒模型进行优化，这要么损害了通用性，要么影响了效率。本文将越狱提示攻击概括为两个方面：(1)提示重写和(2)场景嵌套。在此基础上，我们提出了ReNeLLM，这是一个利用LLM自身生成有效越狱提示的自动化框架。广泛的实验表明，与现有的基准相比，ReNeLLM显著提高了攻击成功率，同时大大降低了时间成本。我们的研究也揭示了现有防御方法在保护低密度脂蛋白方面的不足。最后，从即时执行优先级的角度分析了LLMS防御失败的原因，并提出了相应的防御策略。我们希望我们的研究能够促进学术界和低成本管理系统开发商提供更安全和更规范的低成本管理系统。代码可在https://github.com/NJUNLP/ReNeLLM.上获得



## **49. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

红色团队游戏：红色团队语言模型的游戏理论框架 cs.CL

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2310.00322v4) [paper-pdf](http://arxiv.org/pdf/2310.00322v4)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.

摘要: 可部署的大型语言模型(LLMS)必须符合有益和无害的标准，从而实现LLMS的输出与人的价值之间的一致性。红团队技术构成了实现这一标准的关键途径。现有的工作完全依赖于手动红色团队设计和启发式对抗性提示来进行漏洞检测和优化。这些方法缺乏严格的数学描述，从而限制了在可量化的度量范围内探索多样化的攻击策略，以及在收敛保证下对LLMS进行优化。在本文中，我们提出了一种不需要人工注释的通用博弈论框架--Red-Teaming Game(RTG)。RTG用于分析红队语言模型(RLMS)和蓝队语言模型(BLM)之间的多回合攻防交互。在RTG中，我们提出了一种具有语义空间多样性度量的Gamalized Red-Teaming Solver(GRTS)。GRTS是一种自动红队技术，通过元博弈分析解决RTG向纳什均衡的方向，这对应于理论上保证的RLMS和BLM的优化方向。在RLMS多回合攻击中的实验结果表明，GRTS自主发现多样化的攻击策略，有效地提高了LLMS的安全性，优于已有的启发式红队设计。总体而言，RTG为红色团队任务建立了一个基本框架，并构建了一种新的可扩展的协调监督技术。



## **50. Goal-guided Generative Prompt Injection Attack on Large Language Models**

对大型语言模型的目标引导生成提示注入攻击 cs.CR

22 pages, 8 figures

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2404.07234v1) [paper-pdf](http://arxiv.org/pdf/2404.07234v1)

**Authors**: Chong Zhang, Mingyu Jin, Qinkai Yu, Chengzhi Liu, Haochen Xue, Xiaobo Jin

**Abstract**: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLMs model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representation $x$ and $x'$ of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on $x$ and $x'$. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.

摘要: 现有的大型语言模型为大规模面向用户的自然语言任务提供了坚实的基础。大量用户可以很容易地通过用户界面注入敌意文本或指令，从而造成LLMS模型的安全挑战。虽然目前有大量关于即时注入攻击的研究，但这些黑盒攻击大多采用启发式策略。目前尚不清楚这些启发式策略如何与攻击成功率相关，从而有效地提高模型的稳健性。为了解决这个问题，我们重新定义了攻击的目标：最大化纯文本和敌意文本的条件概率之间的KL偏差。此外，我们证明了当条件概率为高斯分布时，最大化KL发散度等价于最大化明文和敌意文本的嵌入表示$x$和$x‘$之间的马氏距离，并给出了关于$x$和$x’$的定量关系。然后，设计了一种简单有效的目标引导生成性提示注入策略(G2PIA)，找到满足特定约束的注入文本，以近似达到最优的攻击效果。特别值得注意的是，我们的攻击方法是一种计算代价低的无查询黑盒攻击方法。在7个LLM模型和4个数据集上的实验结果表明了该攻击方法的有效性。



