# Latest Adversarial Attack Papers
**update at 2023-11-02 10:24:47**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Robust Safety Classifier for Large Language Models: Adversarial Prompt Shield**

用于大型语言模型的稳健安全分类器：对抗性提示盾 cs.CL

11 pages, 2 figures

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2311.00172v1) [paper-pdf](http://arxiv.org/pdf/2311.00172v1)

**Authors**: Jinhwa Kim, Ali Derakhshan, Ian G. Harris

**Abstract**: Large Language Models' safety remains a critical concern due to their vulnerability to adversarial attacks, which can prompt these systems to produce harmful responses. In the heart of these systems lies a safety classifier, a computational model trained to discern and mitigate potentially harmful, offensive, or unethical outputs. However, contemporary safety classifiers, despite their potential, often fail when exposed to inputs infused with adversarial noise. In response, our study introduces the Adversarial Prompt Shield (APS), a lightweight model that excels in detection accuracy and demonstrates resilience against adversarial prompts. Additionally, we propose novel strategies for autonomously generating adversarial training datasets, named Bot Adversarial Noisy Dialogue (BAND) datasets. These datasets are designed to fortify the safety classifier's robustness, and we investigate the consequences of incorporating adversarial examples into the training process. Through evaluations involving Large Language Models, we demonstrate that our classifier has the potential to decrease the attack success rate resulting from adversarial attacks by up to 60%. This advancement paves the way for the next generation of more reliable and resilient conversational agents.

摘要: 大型语言模型的安全性仍然是一个关键问题，因为它们容易受到对抗性攻击，这可能会促使这些系统产生有害的响应。这些系统的核心是安全分类器，这是一个经过训练的计算模型，用于识别和减少潜在的有害、攻击性或不道德的输出。然而，尽管现代安全分类器具有潜力，但在接触到充满对抗性噪音的输入时，它们往往会失败。作为回应，我们的研究引入了对抗性提示盾牌(APS)，这是一种轻量级模型，在检测准确性方面表现出色，并对对抗性提示表现出韧性。此外，我们还提出了自主生成对抗性训练数据集的新策略，称为僵尸对抗性噪声对话(BAND)数据集。这些数据集是为了加强安全分类器的稳健性而设计的，我们调查了将对抗性例子纳入训练过程的后果。通过对大型语言模型的评估，我们证明了我们的分类器具有将对抗性攻击导致的攻击成功率降低高达60%的潜力。这一进步为下一代更可靠、更具弹性的对话代理铺平了道路。



## **2. LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B**

Lora微调有效地取消了Llama 2-Chat 70B中的安全培训 cs.LG

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.20624v1) [paper-pdf](http://arxiv.org/pdf/2310.20624v1)

**Authors**: Simon Lermen, Charlie Rogers-Smith, Jeffrey Ladish

**Abstract**: AI developers often apply safety alignment procedures to prevent the misuse of their AI systems. For example, before Meta released Llama 2-Chat, a collection of instruction fine-tuned large language models, they invested heavily in safety training, incorporating extensive red-teaming and reinforcement learning from human feedback. However, it remains unclear how well safety training guards against model misuse when attackers have access to model weights. We explore the robustness of safety training in language models by subversively fine-tuning the public weights of Llama 2-Chat. We employ low-rank adaptation (LoRA) as an efficient fine-tuning method. With a budget of less than $200 per model and using only one GPU, we successfully undo the safety training of Llama 2-Chat models of sizes 7B, 13B, and 70B. Specifically, our fine-tuning technique significantly reduces the rate at which the model refuses to follow harmful instructions. We achieve a refusal rate below 1% for our 70B Llama 2-Chat model on two refusal benchmarks. Our fine-tuning method retains general performance, which we validate by comparing our fine-tuned models against Llama 2-Chat across two benchmarks. Additionally, we present a selection of harmful outputs produced by our models. While there is considerable uncertainty about the scope of risks from current models, it is likely that future models will have significantly more dangerous capabilities, including the ability to hack into critical infrastructure, create dangerous bio-weapons, or autonomously replicate and adapt to new environments. We show that subversive fine-tuning is practical and effective, and hence argue that evaluating risks from fine-tuning should be a core part of risk assessments for releasing model weights.

摘要: 人工智能开发人员经常应用安全调整程序来防止滥用其人工智能系统。例如，在Meta发布Llama 2-Chat（一个指令微调的大型语言模型集合）之前，他们在安全培训方面投入了大量资金，结合了广泛的红队和来自人类反馈的强化学习。然而，目前还不清楚当攻击者可以访问模型权重时，安全训练如何防止模型滥用。我们通过颠覆性地微调Llama 2-Chat的公共权重来探索语言模型中安全训练的鲁棒性。我们采用低秩自适应（LoRA）作为一种有效的微调方法。每个模型的预算不到200美元，并且只使用一个GPU，我们成功地取消了7 B，13 B和70 B尺寸的Llama 2-Chat模型的安全训练。具体来说，我们的微调技术显著降低了模型拒绝遵循有害指令的速度。我们的70 B Llama 2-Chat模型在两个拒绝基准上的拒绝率低于1%。我们的微调方法保留了一般的性能，我们通过比较我们的微调模型对美洲驼2-聊天在两个基准验证。此外，我们还介绍了我们的模型产生的有害输出的选择。虽然当前模型的风险范围存在相当大的不确定性，但未来的模型可能会具有更加危险的能力，包括入侵关键基础设施、制造危险的生物武器或自主复制和适应新环境的能力。我们表明，颠覆性的微调是实用和有效的，因此认为，评估风险的微调应该是一个核心部分的风险评估释放模型的权重。



## **3. On Extracting Specialized Code Abilities from Large Language Models: A Feasibility Study**

从大型语言模型中提取专业代码能力的可行性研究 cs.SE

13 pages

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2303.03012v4) [paper-pdf](http://arxiv.org/pdf/2303.03012v4)

**Authors**: Zongjie Li, Chaozheng Wang, Pingchuan Ma, Chaowei Liu, Shuai Wang, Daoyuan Wu, Cuiyun Gao, Yang Liu

**Abstract**: Recent advances in large language models (LLMs) significantly boost their usage in software engineering. However, training a well-performing LLM demands a substantial workforce for data collection and annotation. Moreover, training datasets may be proprietary or partially open, and the process often requires a costly GPU cluster. The intellectual property value of commercial LLMs makes them attractive targets for imitation attacks, but creating an imitation model with comparable parameters still incurs high costs. This motivates us to explore a practical and novel direction: slicing commercial black-box LLMs using medium-sized backbone models. In this paper, we explore the feasibility of launching imitation attacks on LLMs to extract their specialized code abilities, such as"code synthesis" and "code translation." We systematically investigate the effectiveness of launching code ability extraction attacks under different code-related tasks with multiple query schemes, including zero-shot, in-context, and Chain-of-Thought. We also design response checks to refine the outputs, leading to an effective imitation training process. Our results show promising outcomes, demonstrating that with a reasonable number of queries, attackers can train a medium-sized backbone model to replicate specialized code behaviors similar to the target LLMs. We summarize our findings and insights to help researchers better understand the threats posed by imitation attacks, including revealing a practical attack surface for generating adversarial code examples against LLMs.

摘要: 大型语言模型(LLM)的最新进展极大地促进了它们在软件工程中的使用。然而，培训一个表现良好的LLM需要大量的数据收集和注释工作人员。此外，训练数据集可能是专有的或部分开放的，该过程通常需要昂贵的GPU集群。商业LLM的知识产权价值使其成为模仿攻击的有吸引力的目标，但创建具有可比参数的模仿模型仍然会招致高昂的成本。这促使我们探索一个实用而新颖的方向：使用中型主干模型对商业黑盒LLM进行切片。在本文中，我们探索了对LLM发动模仿攻击的可行性，以提取其专门的代码能力，如“代码合成”和“代码翻译”。我们系统地研究了在不同的代码相关任务下，使用包括零命中、上下文中和思想链在内的多种查询方案来发起代码能力提取攻击的有效性。我们还设计了响应检查来优化输出，从而实现有效的模仿培训过程。我们的结果显示了令人振奋的结果，表明通过合理数量的查询，攻击者可以训练一个中等大小的主干模型来复制类似于目标LLM的特定代码行为。我们总结了我们的发现和见解，以帮助研究人员更好地理解模仿攻击所构成的威胁，包括揭示一个实用的攻击面，用于生成针对LLM的敌意代码示例。



## **4. TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models**

TrojLLM：一种针对大型语言模型的黑盒木马提示攻击 cs.CR

Accepted by NeurIPS'23

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2306.06815v3) [paper-pdf](http://arxiv.org/pdf/2306.06815v3)

**Authors**: Jiaqi Xue, Mengxin Zheng, Ting Hua, Yilin Shen, Yepeng Liu, Ladislau Boloni, Qian Lou

**Abstract**: Large Language Models (LLMs) are progressively being utilized as machine learning services and interface tools for various applications. However, the security implications of LLMs, particularly in relation to adversarial and Trojan attacks, remain insufficiently examined. In this paper, we propose TrojLLM, an automatic and black-box framework to effectively generate universal and stealthy triggers. When these triggers are incorporated into the input data, the LLMs' outputs can be maliciously manipulated. Moreover, the framework also supports embedding Trojans within discrete prompts, enhancing the overall effectiveness and precision of the triggers' attacks. Specifically, we propose a trigger discovery algorithm for generating universal triggers for various inputs by querying victim LLM-based APIs using few-shot data samples. Furthermore, we introduce a novel progressive Trojan poisoning algorithm designed to generate poisoned prompts that retain efficacy and transferability across a diverse range of models. Our experiments and results demonstrate TrojLLM's capacity to effectively insert Trojans into text prompts in real-world black-box LLM APIs including GPT-3.5 and GPT-4, while maintaining exceptional performance on clean test sets. Our work sheds light on the potential security risks in current models and offers a potential defensive approach. The source code of TrojLLM is available at https://github.com/UCF-ML-Research/TrojLLM.

摘要: 大型语言模型(LLM)正逐渐被用作各种应用的机器学习服务和接口工具。然而，LLMS的安全影响，特别是与对抗性攻击和特洛伊木马攻击有关的影响，仍然没有得到充分的研究。在本文中，我们提出了一个自动黑盒框架TrojLLM，它可以有效地生成通用的、隐蔽的触发器。当这些触发器被合并到输入数据中时，LLMS的输出可能被恶意操纵。此外，该框架还支持在离散提示中嵌入特洛伊木马，增强了触发器攻击的整体有效性和精确度。具体地说，我们提出了一种触发器发现算法，通过使用少量数据样本查询受害者基于LLM的API来为各种输入生成通用触发器。此外，我们引入了一种新的渐进式特洛伊木马中毒算法，旨在生成中毒提示，从而在不同的模型中保持有效性和可转移性。我们的实验和结果表明，TrojLLM能够在包括GPT-3.5和GPT-4在内的真实黑盒LLMAPI中有效地将特洛伊木马程序插入到文本提示中，同时在干净的测试集上保持出色的性能。我们的工作揭示了当前模型中的潜在安全风险，并提供了一种潜在的防御方法。TrojLLm的源代码可在https://github.com/UCF-ML-Research/TrojLLM.上找到



## **5. Adversarial Attacks and Defenses in Large Language Models: Old and New Threats**

大型语言模型中的对抗性攻击和防御：旧威胁和新威胁 cs.AI

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2310.19737v1) [paper-pdf](http://arxiv.org/pdf/2310.19737v1)

**Authors**: Leo Schwinn, David Dobre, Stephan Günnemann, Gauthier Gidel

**Abstract**: Over the past decade, there has been extensive research aimed at enhancing the robustness of neural networks, yet this problem remains vastly unsolved. Here, one major impediment has been the overestimation of the robustness of new defense approaches due to faulty defense evaluations. Flawed robustness evaluations necessitate rectifications in subsequent works, dangerously slowing down the research and providing a false sense of security. In this context, we will face substantial challenges associated with an impending adversarial arms race in natural language processing, specifically with closed-source Large Language Models (LLMs), such as ChatGPT, Google Bard, or Anthropic's Claude. We provide a first set of prerequisites to improve the robustness assessment of new approaches and reduce the amount of faulty evaluations. Additionally, we identify embedding space attacks on LLMs as another viable threat model for the purposes of generating malicious content in open-sourced models. Finally, we demonstrate on a recently proposed defense that, without LLM-specific best practices in place, it is easy to overestimate the robustness of a new approach.

摘要: 在过去的十年里，已经有大量的研究旨在增强神经网络的健壮性，但这个问题仍然远远没有解决。在这里，一个主要的障碍是由于错误的防御评估而高估了新的防御方法的稳健性。有缺陷的稳健性评估需要在后续工作中进行更正，这会危险地减缓研究速度，并提供一种错误的安全感。在这种情况下，我们将面临与自然语言处理领域即将到来的对抗性军备竞赛相关的重大挑战，特别是与封闭源代码的大型语言模型(LLM)相关的挑战，如ChatGPT、Google Bard或Anthropic的Claude。我们提供了第一组先决条件来改进新方法的稳健性评估，并减少错误评估的数量。此外，我们将在LLM上嵌入空间攻击作为另一种可行的威胁模型，目的是在开源模型中生成恶意内容。最后，我们在最近提出的一项防御中演示了，如果没有特定于LLM的最佳实践，很容易高估新方法的健壮性。



## **6. From Chatbots to PhishBots? -- Preventing Phishing scams created using ChatGPT, Google Bard and Claude**

从聊天机器人到网络钓鱼机器人？--防止使用ChatGPT、Google Bard和Claude创建的网络钓鱼诈骗 cs.CR

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.19181v1) [paper-pdf](http://arxiv.org/pdf/2310.19181v1)

**Authors**: Sayak Saha Roy, Poojitha Thota, Krishna Vamsi Naragam, Shirin Nilizadeh

**Abstract**: The advanced capabilities of Large Language Models (LLMs) have made them invaluable across various applications, from conversational agents and content creation to data analysis, research, and innovation. However, their effectiveness and accessibility also render them susceptible to abuse for generating malicious content, including phishing attacks. This study explores the potential of using four popular commercially available LLMs - ChatGPT (GPT 3.5 Turbo), GPT 4, Claude and Bard to generate functional phishing attacks using a series of malicious prompts. We discover that these LLMs can generate both phishing emails and websites that can convincingly imitate well-known brands, and also deploy a range of evasive tactics for the latter to elude detection mechanisms employed by anti-phishing systems. Notably, these attacks can be generated using unmodified, or "vanilla," versions of these LLMs, without requiring any prior adversarial exploits such as jailbreaking. As a countermeasure, we build a BERT based automated detection tool that can be used for the early detection of malicious prompts to prevent LLMs from generating phishing content attaining an accuracy of 97\% for phishing website prompts, and 94\% for phishing email prompts.

摘要: 大型语言模型(LLM)的高级功能使其在从会话代理和内容创建到数据分析、研究和创新的各种应用程序中具有无价的价值。然而，它们的有效性和可访问性也使它们容易被滥用来生成恶意内容，包括网络钓鱼攻击。这项研究探索了四种流行的商用LLM-ChatGPT(GPT 3.5 Turbo)、GPT 4、Claude和Bard使用一系列恶意提示生成功能性网络钓鱼攻击的可能性。我们发现，这些LLM既可以生成钓鱼电子邮件，也可以生成能够令人信服地模仿知名品牌的网站，并为后者部署一系列规避策略，以躲避反钓鱼系统使用的检测机制。值得注意的是，这些攻击可以使用这些LLM的未经修改或“普通”版本来生成，而不需要任何先前的对抗性攻击，如越狱。作为对策，我们构建了一个基于ERT的自动检测工具，用于早期检测恶意提示，以防止LLMS生成钓鱼内容，对钓鱼网站提示的准确率达到97%，对钓鱼电子邮件提示的准确率达到94%。



## **7. Robustifying Language Models with Test-Time Adaptation**

基于测试时间自适应的模糊语言模型 cs.CL

8 Pages 2 Figures Submitted to ICLR Workshop

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.19177v1) [paper-pdf](http://arxiv.org/pdf/2310.19177v1)

**Authors**: Noah Thomas McDermott, Junfeng Yang, Chengzhi Mao

**Abstract**: Large-scale language models achieved state-of-the-art performance over a number of language tasks. However, they fail on adversarial language examples, which are sentences optimized to fool the language models but with similar semantic meanings for humans. While prior work focuses on making the language model robust at training time, retraining for robustness is often unrealistic for large-scale foundation models. Instead, we propose to make the language models robust at test time. By dynamically adapting the input sentence with predictions from masked words, we show that we can reverse many language adversarial attacks. Since our approach does not require any training, it works for novel tasks at test time and can adapt to novel adversarial corruptions. Visualizations and empirical results on two popular sentence classification datasets demonstrate that our method can repair adversarial language attacks over 65% o

摘要: 大规模语言模型在许多语言任务上实现了最先进的性能。然而，他们在对抗性语言例子上失败了，这些例子是为愚弄语言模型而优化的句子，但对人类来说具有相似的语义。虽然以前的工作重点是在训练时使语言模型具有健壮性，但对于大规模的基础模型来说，进行健壮性的再培训往往是不现实的。相反，我们建议在测试时使语言模型健壮。通过动态调整输入句子和掩蔽词的预测，我们证明了我们可以逆转许多语言对手攻击。由于我们的方法不需要任何训练，它在测试时适用于新的任务，并且可以适应新的对抗性腐败。在两个常用的句子分类数据集上的可视化和实验结果表明，该方法可以修复65%以上的敌意语言攻击



## **8. On the Exploitability of Instruction Tuning**

论指令调优的可开发性 cs.CR

NeurIPS 2023 camera-ready (21 pages, 10 figures)

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2306.17194v2) [paper-pdf](http://arxiv.org/pdf/2306.17194v2)

**Authors**: Manli Shu, Jiongxiao Wang, Chen Zhu, Jonas Geiping, Chaowei Xiao, Tom Goldstein

**Abstract**: Instruction tuning is an effective technique to align large language models (LLMs) with human intents. In this work, we investigate how an adversary can exploit instruction tuning by injecting specific instruction-following examples into the training data that intentionally changes the model's behavior. For example, an adversary can achieve content injection by injecting training examples that mention target content and eliciting such behavior from downstream models. To achieve this goal, we propose \textit{AutoPoison}, an automated data poisoning pipeline. It naturally and coherently incorporates versatile attack goals into poisoned data with the help of an oracle LLM. We showcase two example attacks: content injection and over-refusal attacks, each aiming to induce a specific exploitable behavior. We quantify and benchmark the strength and the stealthiness of our data poisoning scheme. Our results show that AutoPoison allows an adversary to change a model's behavior by poisoning only a small fraction of data while maintaining a high level of stealthiness in the poisoned examples. We hope our work sheds light on how data quality affects the behavior of instruction-tuned models and raises awareness of the importance of data quality for responsible deployments of LLMs. Code is available at \url{https://github.com/azshue/AutoPoison}.

摘要: 指令调优是使大型语言模型与人的意图保持一致的一种有效技术。在这项工作中，我们调查了对手如何通过向训练数据中注入特定的指令遵循示例来利用指令调整，从而故意改变模型的行为。例如，对手可以通过注入提到目标内容的训练示例并从下游模型中引出此类行为来实现内容注入。为了实现这一目标，我们提出了一种自动化的数据中毒管道--Texttit{AutoPoison}。在甲骨文LLM的帮助下，它自然而连贯地将各种攻击目标整合到有毒数据中。我们展示了两个示例攻击：内容注入攻击和过度拒绝攻击，每个攻击都旨在诱导特定的可利用行为。我们对我们的数据中毒方案的强度和隐蔽性进行量化和基准测试。我们的结果表明，AutoPoison允许对手通过只毒化一小部分数据来改变模型的行为，同时在中毒的示例中保持高级别的隐蔽性。我们希望我们的工作有助于阐明数据质量如何影响指令调优模型的行为，并提高人们对数据质量对于负责任的LLM部署的重要性的认识。代码位于\url{https://github.com/azshue/AutoPoison}.



## **9. Large Language Models Are Better Adversaries: Exploring Generative Clean-Label Backdoor Attacks Against Text Classifiers**

大型语言模型是更好的对手：探索针对文本分类器的生成式清洁标签后门攻击 cs.LG

Accepted at EMNLP 2023 Findings

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2310.18603v1) [paper-pdf](http://arxiv.org/pdf/2310.18603v1)

**Authors**: Wencong You, Zayd Hammoudeh, Daniel Lowd

**Abstract**: Backdoor attacks manipulate model predictions by inserting innocuous triggers into training and test data. We focus on more realistic and more challenging clean-label attacks where the adversarial training examples are correctly labeled. Our attack, LLMBkd, leverages language models to automatically insert diverse style-based triggers into texts. We also propose a poison selection technique to improve the effectiveness of both LLMBkd as well as existing textual backdoor attacks. Lastly, we describe REACT, a baseline defense to mitigate backdoor attacks via antidote training examples. Our evaluations demonstrate LLMBkd's effectiveness and efficiency, where we consistently achieve high attack success rates across a wide range of styles with little effort and no model training.

摘要: 后门攻击通过在训练和测试数据中插入无害的触发器来操纵模型预测。我们专注于更现实、更具挑战性的干净标签攻击，其中对抗性训练示例被正确标记。我们的攻击LLMBkd利用语言模型自动将不同的基于样式的触发器插入到文本中。我们还提出了一种毒物选择技术来提高LLMBkd和现有文本后门攻击的有效性。最后，我们描述了Reaction，一种通过解毒剂训练示例来减少后门攻击的基线防御。我们的评估证明了LLMBkd的有效性和效率，在这种情况下，我们几乎不费力气，也没有模型训练，就能在各种风格中始终获得高攻击成功率。



## **10. ParaFuzz: An Interpretability-Driven Technique for Detecting Poisoned Samples in NLP**

ParaFuzz：一种可解释性驱动的NLP中毒样本检测技术 cs.CR

**SubmitDate**: 2023-10-27    [abs](http://arxiv.org/abs/2308.02122v2) [paper-pdf](http://arxiv.org/pdf/2308.02122v2)

**Authors**: Lu Yan, Zhuo Zhang, Guanhong Tao, Kaiyuan Zhang, Xuan Chen, Guangyu Shen, Xiangyu Zhang

**Abstract**: Backdoor attacks have emerged as a prominent threat to natural language processing (NLP) models, where the presence of specific triggers in the input can lead poisoned models to misclassify these inputs to predetermined target classes. Current detection mechanisms are limited by their inability to address more covert backdoor strategies, such as style-based attacks. In this work, we propose an innovative test-time poisoned sample detection framework that hinges on the interpretability of model predictions, grounded in the semantic meaning of inputs. We contend that triggers (e.g., infrequent words) are not supposed to fundamentally alter the underlying semantic meanings of poisoned samples as they want to stay stealthy. Based on this observation, we hypothesize that while the model's predictions for paraphrased clean samples should remain stable, predictions for poisoned samples should revert to their true labels upon the mutations applied to triggers during the paraphrasing process. We employ ChatGPT, a state-of-the-art large language model, as our paraphraser and formulate the trigger-removal task as a prompt engineering problem. We adopt fuzzing, a technique commonly used for unearthing software vulnerabilities, to discover optimal paraphrase prompts that can effectively eliminate triggers while concurrently maintaining input semantics. Experiments on 4 types of backdoor attacks, including the subtle style backdoors, and 4 distinct datasets demonstrate that our approach surpasses baseline methods, including STRIP, RAP, and ONION, in precision and recall.

摘要: 后门攻击已经成为自然语言处理(NLP)模型的一个突出威胁，在NLP模型中，输入中存在特定触发器可能会导致中毒模型将这些输入错误分类到预定的目标类别。当前的检测机制由于无法应对更隐蔽的后门策略而受到限制，例如基于样式的攻击。在这项工作中，我们提出了一个创新的测试时间中毒样本检测框架，该框架取决于模型预测的可解释性，基于输入的语义。我们认为，触发因素(例如，不常见的单词)不应该从根本上改变中毒样本的潜在语义，因为它们想要保持隐蔽性。基于这一观察，我们假设，虽然模型对释义干净样本的预测应该保持稳定，但对中毒样本的预测应该在释义过程中应用于触发器的突变后恢复到其真实标签。我们使用最先进的大型语言模型ChatGPT作为我们的释义，并将触发器移除任务描述为一个紧迫的工程问题。我们采用了模糊技术，这是一种常用的软件漏洞挖掘技术，可以发现最优的释义提示，可以有效地消除触发器，同时保持输入语义。在4种类型的后门攻击(包括微妙风格的后门攻击)和4个不同的数据集上的实验表明，我们的方法在准确率和召回率上都超过了基线方法，包括STRAP、RAP和洋葱。



## **11. MasterKey: Automated Jailbreak Across Multiple Large Language Model Chatbots**

MasterKey：跨多个大型语言模型聊天机器人的自动越狱 cs.CR

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2307.08715v2) [paper-pdf](http://arxiv.org/pdf/2307.08715v2)

**Authors**: Gelei Deng, Yi Liu, Yuekang Li, Kailong Wang, Ying Zhang, Zefeng Li, Haoyu Wang, Tianwei Zhang, Yang Liu

**Abstract**: Large Language Models (LLMs) have revolutionized Artificial Intelligence (AI) services due to their exceptional proficiency in understanding and generating human-like text. LLM chatbots, in particular, have seen widespread adoption, transforming human-machine interactions. However, these LLM chatbots are susceptible to "jailbreak" attacks, where malicious users manipulate prompts to elicit inappropriate or sensitive responses, contravening service policies. Despite existing attempts to mitigate such threats, our research reveals a substantial gap in our understanding of these vulnerabilities, largely due to the undisclosed defensive measures implemented by LLM service providers.   In this paper, we present Jailbreaker, a comprehensive framework that offers an in-depth understanding of jailbreak attacks and countermeasures. Our work makes a dual contribution. First, we propose an innovative methodology inspired by time-based SQL injection techniques to reverse-engineer the defensive strategies of prominent LLM chatbots, such as ChatGPT, Bard, and Bing Chat. This time-sensitive approach uncovers intricate details about these services' defenses, facilitating a proof-of-concept attack that successfully bypasses their mechanisms. Second, we introduce an automatic generation method for jailbreak prompts. Leveraging a fine-tuned LLM, we validate the potential of automated jailbreak generation across various commercial LLM chatbots. Our method achieves a promising average success rate of 21.58%, significantly outperforming the effectiveness of existing techniques. We have responsibly disclosed our findings to the concerned service providers, underscoring the urgent need for more robust defenses. Jailbreaker thus marks a significant step towards understanding and mitigating jailbreak threats in the realm of LLM chatbots.

摘要: 大型语言模型(LLM)由于其在理解和生成类似人类的文本方面的非凡熟练程度，使人工智能(AI)服务发生了革命性的变化。尤其是LLM聊天机器人，已经被广泛采用，改变了人机交互。然而，这些LLM聊天机器人很容易受到“越狱”攻击，即恶意用户操纵提示来引发不适当或敏感的响应，这违反了服务策略。尽管存在缓解此类威胁的尝试，但我们的研究显示，我们对这些漏洞的理解存在很大差距，这主要是由于LLM服务提供商实施了未披露的防御措施。在这篇文章中，我们介绍了越狱，一个全面的框架，提供了深入了解越狱攻击和对策。我们的工作做出了双重贡献。首先，我们提出了一种受基于时间的SQL注入技术启发的创新方法来对著名的LLM聊天机器人(如ChatGPT、Bard和Bing Chat)的防御策略进行逆向工程。这种对时间敏感的方法揭示了这些服务防御的复杂细节，为成功绕过它们的机制的概念验证攻击提供了便利。其次，介绍了一种越狱提示的自动生成方法。利用微调的LLM，我们验证了在各种商业LLM聊天机器人上自动越狱生成的潜力。我们的方法达到了21.58%的平均成功率，大大超过了现有技术的有效性。我们已经负责任地向有关服务提供商披露了我们的调查结果，强调了迫切需要更强大的防御措施。因此，越狱标志着在理解和减轻LLM聊天机器人领域的越狱威胁方面迈出了重要的一步。



## **12. Locally Differentially Private Document Generation Using Zero Shot Prompting**

基于零镜头提示的局部差异私有文档生成方法 cs.CL

Accepted at EMNLP 2023 (Findings)

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.16111v1) [paper-pdf](http://arxiv.org/pdf/2310.16111v1)

**Authors**: Saiteja Utpala, Sara Hooker, Pin Yu Chen

**Abstract**: Numerous studies have highlighted the privacy risks associated with pretrained large language models. In contrast, our research offers a unique perspective by demonstrating that pretrained large language models can effectively contribute to privacy preservation. We propose a locally differentially private mechanism called DP-Prompt, which leverages the power of pretrained large language models and zero-shot prompting to counter author de-anonymization attacks while minimizing the impact on downstream utility. When DP-Prompt is used with a powerful language model like ChatGPT (gpt-3.5), we observe a notable reduction in the success rate of de-anonymization attacks, showing that it surpasses existing approaches by a considerable margin despite its simpler design. For instance, in the case of the IMDB dataset, DP-Prompt (with ChatGPT) perfectly recovers the clean sentiment F1 score while achieving a 46\% reduction in author identification F1 score against static attackers and a 26\% reduction against adaptive attackers. We conduct extensive experiments across six open-source large language models, ranging up to 7 billion parameters, to analyze various effects of the privacy-utility tradeoff.

摘要: 许多研究都强调了与预先训练的大型语言模型相关的隐私风险。相比之下，我们的研究提供了一个独特的视角，证明了预先训练的大型语言模型可以有效地有助于隐私保护。我们提出了一种称为DP-Prompt的局部差异私有机制，该机制利用预先训练的大型语言模型和零镜头提示的能力来对抗作者去匿名化攻击，同时最小化对下游效用的影响。当DP-Prompt与ChatGPT(GPT-3.5)等强大的语言模型一起使用时，我们观察到去匿名化攻击的成功率显著下降，表明尽管它的设计更简单，但它在相当大程度上超过了现有的方法。例如，在IMDB数据集的情况下，DP-Prompt(使用ChatGPT)完美地恢复了干净的情感F1分数，同时在针对静态攻击者的作者识别F1分数和针对自适应攻击者的F1分数分别减少了46%和26%。我们在六个开放源码的大型语言模型上进行了广泛的实验，范围多达70亿个参数，以分析隐私-效用权衡的各种影响。



## **13. Self-Guard: Empower the LLM to Safeguard Itself**

自我保护：增强LLM的自我保护能力 cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15851v1) [paper-pdf](http://arxiv.org/pdf/2310.15851v1)

**Authors**: Zezhong Wang, Fangkai Yang, Lu Wang, Pu Zhao, Hongru Wang, Liang Chen, Qingwei Lin, Kam-Fai Wong

**Abstract**: The jailbreak attack can bypass the safety measures of a Large Language Model (LLM), generating harmful content. This misuse of LLM has led to negative societal consequences. Currently, there are two main approaches to address jailbreak attacks: safety training and safeguards. Safety training focuses on further training LLM to enhance its safety. On the other hand, safeguards involve implementing external models or filters to prevent harmful outputs. However, safety training has constraints in its ability to adapt to new attack types and often leads to a drop in model performance. Safeguards have proven to be of limited help. To tackle these issues, we propose a novel approach called Self-Guard, which combines the strengths of both safety methods. Self-Guard includes two stages. In the first stage, we enhance the model's ability to assess harmful content, and in the second stage, we instruct the model to consistently perform harmful content detection on its own responses. The experiment has demonstrated that Self-Guard is robust against jailbreak attacks. In the bad case analysis, we find that LLM occasionally provides harmless responses to harmful queries. Additionally, we evaluated the general capabilities of the LLM before and after safety training, providing evidence that Self-Guard does not result in the LLM's performance degradation. In sensitivity tests, Self-Guard not only avoids inducing over-sensitivity in LLM but also can even mitigate this issue.

摘要: 越狱攻击可以绕过大型语言模型(LLM)的安全措施，生成有害内容。这种对LLM的滥用已经导致了负面的社会后果。目前，解决越狱攻击的主要方法有两种：安全培训和保障措施。安全培训的重点是对LLM进行进一步培训，以提高其安全性。另一方面，保障措施涉及实施外部模型或过滤器，以防止有害输出。然而，安全培训在适应新攻击类型的能力方面存在限制，往往会导致模型性能下降。事实证明，保障措施的帮助有限。为了解决这些问题，我们提出了一种名为Self-Guard的新方法，它结合了两种安全方法的优点。自我保护包括两个阶段。在第一阶段，我们增强了模型评估有害内容的能力，在第二阶段，我们指示模型对其自身的响应进行一致的有害内容检测。实验证明，Self-Guard对越狱攻击具有很强的抵抗力。在坏案例分析中，我们发现LLM偶尔会对有害查询提供无害的响应。此外，我们在安全培训前后评估了LLM的一般能力，提供了自我保护不会导致LLM性能下降的证据。在敏感性测试中，Self-Guard不仅可以避免在LLM中诱导过度敏感，而且甚至可以缓解这一问题。



## **14. A Survey on LLM-generated Text Detection: Necessity, Methods, and Future Directions**

LLM生成的文本检测：必要性、方法和未来发展方向综述 cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.14724v2) [paper-pdf](http://arxiv.org/pdf/2310.14724v2)

**Authors**: Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Derek F. Wong, Lidia S. Chao

**Abstract**: The powerful ability to understand, follow, and generate complex language emerging from large language models (LLMs) makes LLM-generated text flood many areas of our daily lives at an incredible speed and is widely accepted by humans. As LLMs continue to expand, there is an imperative need to develop detectors that can detect LLM-generated text. This is crucial to mitigate potential misuse of LLMs and safeguard realms like artistic expression and social networks from harmful influence of LLM-generated content. The LLM-generated text detection aims to discern if a piece of text was produced by an LLM, which is essentially a binary classification task. The detector techniques have witnessed notable advancements recently, propelled by innovations in watermarking techniques, zero-shot methods, fine-turning LMs methods, adversarial learning methods, LLMs as detectors, and human-assisted methods. In this survey, we collate recent research breakthroughs in this area and underscore the pressing need to bolster detector research. We also delve into prevalent datasets, elucidating their limitations and developmental requirements. Furthermore, we analyze various LLM-generated text detection paradigms, shedding light on challenges like out-of-distribution problems, potential attacks, and data ambiguity. Conclusively, we highlight interesting directions for future research in LLM-generated text detection to advance the implementation of responsible artificial intelligence (AI). Our aim with this survey is to provide a clear and comprehensive introduction for newcomers while also offering seasoned researchers a valuable update in the field of LLM-generated text detection. The useful resources are publicly available at: https://github.com/NLP2CT/LLM-generated-Text-Detection.

摘要: 大型语言模型(LLM)强大的理解、跟踪和生成复杂语言的能力使得LLM生成的文本以令人难以置信的速度涌入我们日常生活的许多领域，并被人类广泛接受。随着LLMS的不断扩展，迫切需要开发能够检测LLM生成的文本的检测器。这对于减少LLM的潜在滥用以及保护艺术表达和社交网络等领域免受LLM生成的内容的有害影响至关重要。LLM生成的文本检测旨在识别一段文本是否由LLM生成，这本质上是一项二进制分类任务。最近，在水印技术、零镜头方法、精细旋转LMS方法、对抗性学习方法、作为检测器的LLMS以及人工辅助方法的创新的推动下，检测器技术有了显著的进步。在这次调查中，我们整理了这一领域的最新研究突破，并强调了支持探测器研究的迫切需要。我们还深入研究了流行的数据集，阐明了它们的局限性和发展需求。此外，我们分析了各种LLM生成的文本检测范例，揭示了诸如分发外问题、潜在攻击和数据歧义等挑战。最后，我们指出了未来在LLM生成的文本检测方面的有趣研究方向，以推进负责任人工智能(AI)的实施。我们这次调查的目的是为新手提供一个清晰而全面的介绍，同时也为经验丰富的研究人员提供在LLM生成的文本检测领域的有价值的更新。这些有用的资源可在以下网址公开获得：https://github.com/NLP2CT/LLM-generated-Text-Detection.



## **15. A Survey on Detection of LLMs-Generated Content**

基于LLMS的内容检测技术研究综述 cs.CL

We will keep updating at  https://github.com/Xianjun-Yang/Awesome_papers_on_LLMs_detection.git

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15654v1) [paper-pdf](http://arxiv.org/pdf/2310.15654v1)

**Authors**: Xianjun Yang, Liangming Pan, Xuandong Zhao, Haifeng Chen, Linda Petzold, William Yang Wang, Wei Cheng

**Abstract**: The burgeoning capabilities of advanced large language models (LLMs) such as ChatGPT have led to an increase in synthetic content generation with implications across a variety of sectors, including media, cybersecurity, public discourse, and education. As such, the ability to detect LLMs-generated content has become of paramount importance. We aim to provide a detailed overview of existing detection strategies and benchmarks, scrutinizing their differences and identifying key challenges and prospects in the field, advocating for more adaptable and robust models to enhance detection accuracy. We also posit the necessity for a multi-faceted approach to defend against various attacks to counter the rapidly advancing capabilities of LLMs. To the best of our knowledge, this work is the first comprehensive survey on the detection in the era of LLMs. We hope it will provide a broad understanding of the current landscape of LLMs-generated content detection, offering a guiding reference for researchers and practitioners striving to uphold the integrity of digital information in an era increasingly dominated by synthetic content. The relevant papers are summarized and will be consistently updated at https://github.com/Xianjun-Yang/Awesome_papers_on_LLMs_detection.git.

摘要: ChatGPT等高级大型语言模型（LLM）的新兴功能导致合成内容生成的增加，并对包括媒体、网络安全、公共话语和教育在内的各个领域产生影响。因此，检测LLMs生成的内容的能力变得至关重要。我们的目标是提供现有检测策略和基准的详细概述，仔细研究它们的差异，并确定该领域的关键挑战和前景，倡导更适应性和更强大的模型，以提高检测准确性。我们还强调，有必要采取多方面的方法来抵御各种攻击，以应对快速发展的LLM能力。据我们所知，这项工作是第一次全面调查的检测时代的LLM。我们希望它将提供对当前LLMs生成内容检测的广泛理解，为研究人员和从业人员提供指导性参考，努力在合成内容日益主导的时代维护数字信息的完整性。相关论文的摘要将在https://github.com/Xianjun-Yang/Awesome_papers_on_LLMs_detection.git上不断更新。



## **16. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

LLM自卫：通过自我检查，LLM知道自己被骗了 cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2308.07308v3) [paper-pdf](http://arxiv.org/pdf/2308.07308v3)

**Authors**: Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau

**Abstract**: Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2.

摘要: 大型语言模型(LLM)对于高质量的文本生成很受欢迎，但可能会产生有害的内容，即使通过强化学习与人类的价值观保持一致。对抗性提示可以绕过它们的安全措施。我们提出了LLM自卫，这是一种通过让LLM筛选诱导响应来防御这些攻击的简单方法。我们的方法不需要任何微调、输入预处理或迭代输出生成。相反，我们将生成的内容合并到预定义的提示中，并使用LLM的另一个实例来分析文本并预测它是否有害。我们在GPT 3.5和Llama 2上测试了LLM自卫，这两种当前最著名的LLM针对各种类型的攻击，例如对提示的强制诱导肯定反应和提示工程攻击。值得注意的是，LLM自卫成功地使用GPT 3.5和Llama 2将攻击成功率降低到几乎为0。



## **17. PrivInfer: Privacy-Preserving Inference for Black-box Large Language Model**

PrivInfer：黑箱大语言模型的隐私保护推理 cs.CR

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.12214v3) [paper-pdf](http://arxiv.org/pdf/2310.12214v3)

**Authors**: Meng Tong, Kejiang Chen, Yuang Qi, Jie Zhang, Weiming Zhang, Nenghai Yu

**Abstract**: Large language models (LLMs), such as ChatGPT, have simplified text generation tasks, yet their inherent privacy risks are increasingly garnering attention. Existing solutions for privacy-preserving inference face significant challenges in practical deployment and implementation. In this paper, we propose PrivInfer, the first practical framework for privacy-preserving inference. It comprises two modules specifically designed for black-box LLMs in text generation. The perturbation module, employing differential privacy, generates perturbed prompts, thus enabling privacy-preserving inference with black-box LLMs. The restoration module extracts coherent and meaningful responses from obtained perturbed results, thus ensuring the accomplishment of the text generation tasks. Additionally, to enhance privacy and utility further, we develop RANTEXT, a novel differential privacy mechanism integrated into the perturbation module of PrivInfer. This mechanism is specifically tailored for LLMs and utilizes random adjacency in text perturbations. Experimental results indicate that PrivInfer is comparable to GPT-4 in text generation quality, and RANTEXT outperforms the current leading scheme in privacy protection, even under its adaptive attack, our proposed GPT inference attack.

摘要: 大型语言模型(LLM)，如ChatGPT，简化了文本生成任务，但其固有的隐私风险正日益引起人们的关注。现有的隐私保护推理解决方案在实际部署和实施中面临着巨大的挑战。在本文中，我们提出了第一个实用的隐私保护推理框架PrivInfer。它包括两个模块，专门针对文本生成中的黑盒LLMS而设计。扰动模块采用不同的隐私，生成扰动提示，从而实现与黑盒LLMS的隐私保护推理。恢复模块从获得的扰动结果中提取连贯且有意义的响应，从而确保文本生成任务的完成。此外，为了进一步增强隐私和实用性，我们开发了一种新的差异隐私机制RANTEXT，该机制集成在PrivInfer的扰动模块中。该机制是专门为LLMS量身定做的，并利用文本扰动中的随机邻接。实验结果表明，PrivInfer在文本生成质量上与GPT-4相当，而RANTEXT在隐私保护方面的性能优于目前领先的方案，即使在其自适应攻击的情况下也是如此。



## **18. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

Janus界面：大型语言模型中的微调如何放大隐私风险 cs.CR

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15469v1) [paper-pdf](http://arxiv.org/pdf/2310.15469v1)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, XiaoFeng Wang, Haixu Tang

**Abstract**: The era post-2018 marked the advent of Large Language Models (LLMs), with innovations such as OpenAI's ChatGPT showcasing prodigious linguistic prowess. As the industry galloped toward augmenting model parameters and capitalizing on vast swaths of human language data, security and privacy challenges also emerged. Foremost among these is the potential inadvertent accrual of Personal Identifiable Information (PII) during web-based data acquisition, posing risks of unintended PII disclosure. While strategies like RLHF during training and Catastrophic Forgetting have been marshaled to control the risk of privacy infringements, recent advancements in LLMs, epitomized by OpenAI's fine-tuning interface for GPT-3.5, have reignited concerns. One may ask: can the fine-tuning of LLMs precipitate the leakage of personal information embedded within training datasets? This paper reports the first endeavor to seek the answer to the question, particularly our discovery of a new LLM exploitation avenue, called the Janus attack. In the attack, one can construct a PII association task, whereby an LLM is fine-tuned using a minuscule PII dataset, to potentially reinstate and reveal concealed PIIs. Our findings indicate that, with a trivial fine-tuning outlay, LLMs such as GPT-3.5 can transition from being impermeable to PII extraction to a state where they divulge a substantial proportion of concealed PII. This research, through its deep dive into the Janus attack vector, underscores the imperative of navigating the intricate interplay between LLM utility and privacy preservation.

摘要: 2018年后的时代标志着大型语言模型(LLM)的到来，OpenAI的ChatGPT等创新展示了惊人的语言能力。随着该行业朝着增加模型参数和利用大量人类语言数据的方向飞奔，安全和隐私挑战也出现了。其中最重要的是在基于网络的数据获取过程中可能无意中积累的个人身份信息(PII)，造成意外披露PII的风险。虽然像训练期间的RLHF和灾难性忘记这样的策略已经被用来控制侵犯隐私的风险，但最近LLMS的进步，以OpenAI针对GPT-3.5的微调接口为代表，再次引发了人们的担忧。有人可能会问：LLMS的微调是否会导致嵌入在训练数据集中的个人信息泄露？本文报道了为寻找这个问题的答案所做的第一次努力，特别是我们发现了一种新的LLM利用途径，称为Janus攻击。在攻击中，一个人可以构建一个PII关联任务，由此使用微小的PII数据集来微调LLM，以潜在地恢复和揭示隐藏的PII。我们的发现表明，只需很小的微调费用，GPT-3.5等低密度脂蛋白就可以从不透水过渡到PII提取，从而达到泄露相当大比例隐藏PII的状态。这项研究通过对Janus攻击载体的深入研究，强调了导航LLM实用程序和隐私保护之间复杂相互作用的必要性。



## **19. AutoDAN: Automatic and Interpretable Adversarial Attacks on Large Language Models**

AutoDAN：对大型语言模型的自动和可解释的对抗性攻击 cs.CR

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15140v1) [paper-pdf](http://arxiv.org/pdf/2310.15140v1)

**Authors**: Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

**Abstract**: Safety alignment of Large Language Models (LLMs) can be compromised with manual jailbreak attacks and (automatic) adversarial attacks. Recent work suggests that patching LLMs against these attacks is possible: manual jailbreak attacks are human-readable but often limited and public, making them easy to block; adversarial attacks generate gibberish prompts that can be detected using perplexity-based filters. In this paper, we show that these solutions may be too optimistic. We propose an interpretable adversarial attack, \texttt{AutoDAN}, that combines the strengths of both types of attacks. It automatically generates attack prompts that bypass perplexity-based filters while maintaining a high attack success rate like manual jailbreak attacks. These prompts are interpretable and diverse, exhibiting strategies commonly used in manual jailbreak attacks, and transfer better than their non-readable counterparts when using limited training data or a single proxy model. We also customize \texttt{AutoDAN}'s objective to leak system prompts, another jailbreak application not addressed in the adversarial attack literature. Our work provides a new way to red-team LLMs and to understand the mechanism of jailbreak attacks.

摘要: 大型语言模型(LLM)的安全对齐可能会受到手动越狱攻击和(自动)对抗性攻击的影响。最近的研究表明，修补LLM以抵御这些攻击是可能的：手动越狱攻击是人类可读的，但通常是有限的和公开的，使它们很容易被阻止；对抗性攻击生成胡言乱语的提示，可以使用基于困惑的过滤器检测到。在本文中，我们证明了这些解决方案可能过于乐观。我们提出了一种可解释的对抗性攻击，它结合了这两种攻击的优点。它自动生成攻击提示，绕过基于困惑的过滤器，同时保持较高的攻击成功率，如手动越狱攻击。这些提示是可解释的和多样化的，展示了手动越狱攻击中常用的策略，并且在使用有限的训练数据或单一代理模型时，传输效果比不可读的相应提示更好。我们还定制了S的目标来泄露系统提示，这是另一个在对抗性攻击文献中没有涉及的越狱应用。我们的工作为红队低层管理和理解越狱攻击的机制提供了一种新的途径。



## **20. Did the Neurons Read your Book? Document-level Membership Inference for Large Language Models**

神经元有没有读过你的书？面向大型语言模型的文档级隶属度推理 cs.CL

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15007v1) [paper-pdf](http://arxiv.org/pdf/2310.15007v1)

**Authors**: Matthieu Meeus, Shubham Jain, Marek Rei, Yves-Alexandre de Montjoye

**Abstract**: With large language models (LLMs) poised to become embedded in our daily lives, questions are starting to be raised about the dataset(s) they learned from. These questions range from potential bias or misinformation LLMs could retain from their training data to questions of copyright and fair use of human-generated text. However, while these questions emerge, developers of the recent state-of-the-art LLMs become increasingly reluctant to disclose details on their training corpus. We here introduce the task of document-level membership inference for real-world LLMs, i.e. inferring whether the LLM has seen a given document during training or not. First, we propose a procedure for the development and evaluation of document-level membership inference for LLMs by leveraging commonly used data sources for training and the model release date. We then propose a practical, black-box method to predict document-level membership and instantiate it on OpenLLaMA-7B with both books and academic papers. We show our methodology to perform very well, reaching an impressive AUC of 0.856 for books and 0.678 for papers. We then show our approach to outperform the sentence-level membership inference attacks used in the privacy literature for the document-level membership task. We finally evaluate whether smaller models might be less sensitive to document-level inference and show OpenLLaMA-3B to be approximately as sensitive as OpenLLaMA-7B to our approach. Taken together, our results show that accurate document-level membership can be inferred for LLMs, increasing the transparency of technology poised to change our lives.

摘要: 随着大型语言模型(LLM)即将融入我们的日常生活，人们开始对他们学习的数据集(S)提出质疑。这些问题从小岛屿发展中国家可能从其培训数据中保留的潜在偏见或错误信息，到版权和合理使用人类生成的文本的问题。然而，当这些问题浮出水面时，最近最先进的LLM的开发商越来越不愿透露他们的培训语料库的细节。我们在这里介绍了真实世界LLM的文档级成员关系推理任务，即推断LLM在训练期间是否看到过给定的文档。首先，我们提出了一种通过利用用于训练的常用数据源和模型发布日期来开发和评估LLMS的文档级成员资格推理的过程。然后，我们提出了一种实用的黑盒方法来预测文档级成员资格，并在OpenLLaMA-7B上用书籍和学术论文进行了实例化。我们的方法表现非常出色，图书的AUC达到了令人印象深刻的0.856，论文的AUC达到了0.678。然后，我们展示了我们的方法，以优于在隐私文献中用于文档级成员身份任务的句子级别成员身份推理攻击。最后，我们评估了较小的模型是否对文档级推理不那么敏感，并表明OpenLLaMA-3B对我们的方法大约与OpenLLaMA-7B一样敏感。综上所述，我们的结果表明，可以为LLMS推断出准确的文档级成员资格，从而增加了即将改变我们生活的技术的透明度。



## **21. MoPe: Model Perturbation-based Privacy Attacks on Language Models**

MOPE：基于模型扰动的语言模型隐私攻击 cs.LG

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.14369v1) [paper-pdf](http://arxiv.org/pdf/2310.14369v1)

**Authors**: Marvin Li, Jason Wang, Jeffrey Wang, Seth Neel

**Abstract**: Recent work has shown that Large Language Models (LLMs) can unintentionally leak sensitive information present in their training data. In this paper, we present Model Perturbations (MoPe), a new method to identify with high confidence if a given text is in the training data of a pre-trained language model, given white-box access to the models parameters. MoPe adds noise to the model in parameter space and measures the drop in log-likelihood at a given point $x$, a statistic we show approximates the trace of the Hessian matrix with respect to model parameters. Across language models ranging from $70$M to $12$B parameters, we show that MoPe is more effective than existing loss-based attacks and recently proposed perturbation-based methods. We also examine the role of training point order and model size in attack success, and empirically demonstrate that MoPe accurately approximate the trace of the Hessian in practice. Our results show that the loss of a point alone is insufficient to determine extractability -- there are training points we can recover using our method that have average loss. This casts some doubt on prior works that use the loss of a point as evidence of memorization or unlearning.

摘要: 最近的研究表明，大型语言模型(LLM)可能会无意中泄露其训练数据中存在的敏感信息。在本文中，我们提出了模型扰动(MOPE)，一种新的方法，在白盒访问模型参数的情况下，高置信度地识别给定文本是否在预先训练的语言模型的训练数据中。MOPE在参数空间中向模型添加噪声，并测量给定点$x$处的对数似然下降，我们给出的统计量近似于Hessian矩阵关于模型参数的迹。跨语言模型，从$7,000$M到$12$B参数，我们表明，MOPE比现有的基于损失的攻击和最近提出的基于扰动的方法更有效。我们还考察了训练点顺序和模型大小在攻击成功中的作用，并实证证明了MOPE在实践中准确地逼近了黑森轨迹。我们的结果表明，仅损失一个点不足以确定可萃取性--我们可以使用我们的方法恢复具有平均损失的训练点。这让人们对以前的作品产生了一些怀疑，这些作品把失去一分作为记忆或遗忘的证据。



## **22. Language Model Unalignment: Parametric Red-Teaming to Expose Hidden Harms and Biases**

语言模型不一致：暴露隐藏的危害和偏见的参数红色团队 cs.CL

Under Review

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.14303v1) [paper-pdf](http://arxiv.org/pdf/2310.14303v1)

**Authors**: Rishabh Bhardwaj, Soujanya Poria

**Abstract**: Red-teaming has been a widely adopted way to evaluate the harmfulness of Large Language Models (LLMs). It aims to jailbreak a model's safety behavior to make it act as a helpful agent disregarding the harmfulness of the query. Existing methods are primarily based on input text-based red-teaming such as adversarial prompts, low-resource prompts, or contextualized prompts to condition the model in a way to bypass its safe behavior. Bypassing the guardrails uncovers hidden harmful information and biases in the model that are left untreated or newly introduced by its safety training. However, prompt-based attacks fail to provide such a diagnosis owing to their low attack success rate, and applicability to specific models. In this paper, we present a new perspective on LLM safety research i.e., parametric red-teaming through Unalignment. It simply (instruction) tunes the model parameters to break model guardrails that are not deeply rooted in the model's behavior. Unalignment using as few as 100 examples can significantly bypass commonly referred to as CHATGPT, to the point where it responds with an 88% success rate to harmful queries on two safety benchmark datasets. On open-source models such as VICUNA-7B and LLAMA-2-CHAT 7B AND 13B, it shows an attack success rate of more than 91%. On bias evaluations, Unalignment exposes inherent biases in safety-aligned models such as CHATGPT and LLAMA- 2-CHAT where the model's responses are strongly biased and opinionated 64% of the time.

摘要: 红团队已被广泛采用来评估大型语言模型的危害性。它的目的是让模特的安全行为越狱，使其成为一个有帮助的代理人，而不考虑询问的危害性。现有方法主要基于诸如对抗性提示、低资源提示或情境化提示的基于输入文本的红团队，以使模型以绕过其安全行为的方式调节。绕过护栏发现了模型中隐藏的有害信息和偏见，这些信息和偏见是未经处理的或安全培训新引入的。然而，基于提示的攻击无法提供这样的诊断，因为它们的攻击成功率低，并且适用于特定的模型。在这篇文章中，我们提出了一个新的视角来研究LLM安全，即通过非对齐的参数红组。它只是(指令)调整模型参数，以打破并不深深植根于模型行为中的模型护栏。只要使用100个例子，UnAlign就可以显著绕过通常所说的CHATGPT，以至于它对两个安全基准数据集上的有害查询的响应成功率为88%。在VIVUNA-7B和LLAMA-2-Chat 7B和13B等开源机型上，攻击成功率超过91%。在偏差评估方面，UnAlign暴露了安全对齐模型中的固有偏见，如CHATGPT和Llama-2-Chat，其中模型的反应在64%的时间内是强烈偏见和固执己见的。



## **23. LoFT: Local Proxy Fine-tuning For Improving Transferability Of Adversarial Attacks Against Large Language Model**

LOFT：提高大型语言模型对抗性攻击可转移性的局部代理微调 cs.CL

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2310.04445v2) [paper-pdf](http://arxiv.org/pdf/2310.04445v2)

**Authors**: Muhammad Ahmed Shah, Roshan Sharma, Hira Dhamyal, Raphael Olivier, Ankit Shah, Joseph Konan, Dareen Alharthi, Hazim T Bukhari, Massa Baali, Soham Deshmukh, Michael Kuhlmann, Bhiksha Raj, Rita Singh

**Abstract**: It has been shown that Large Language Model (LLM) alignments can be circumvented by appending specially crafted attack suffixes with harmful queries to elicit harmful responses. To conduct attacks against private target models whose characterization is unknown, public models can be used as proxies to fashion the attack, with successful attacks being transferred from public proxies to private target models. The success rate of attack depends on how closely the proxy model approximates the private model. We hypothesize that for attacks to be transferrable, it is sufficient if the proxy can approximate the target model in the neighborhood of the harmful query. Therefore, in this paper, we propose \emph{Local Fine-Tuning (LoFT)}, \textit{i.e.}, fine-tuning proxy models on similar queries that lie in the lexico-semantic neighborhood of harmful queries to decrease the divergence between the proxy and target models. First, we demonstrate three approaches to prompt private target models to obtain similar queries given harmful queries. Next, we obtain data for local fine-tuning by eliciting responses from target models for the generated similar queries. Then, we optimize attack suffixes to generate attack prompts and evaluate the impact of our local fine-tuning on the attack's success rate. Experiments show that local fine-tuning of proxy models improves attack transferability and increases attack success rate by $39\%$, $7\%$, and $0.5\%$ (absolute) on target models ChatGPT, GPT-4, and Claude respectively.

摘要: 已有研究表明，通过在巧尽心思构建的攻击后缀上附加有害查询来引发有害响应，可以绕过大型语言模型(LLM)对齐。为了对特征未知的私有目标模型进行攻击，可以使用公共模型作为代理来进行攻击，成功的攻击将从公共代理转移到私有目标模型。攻击的成功率取决于代理模型与私有模型的接近程度。我们假设，对于可转移的攻击，只要代理能够逼近有害查询附近的目标模型就足够了。因此，在本文中，我们提出了对位于有害查询的词典-语义邻域中的相似查询的代理模型进行微调，以减少代理模型和目标模型之间的差异。首先，我们演示了三种方法来提示私人目标模型在给定有害查询的情况下获得类似的查询。接下来，我们通过从目标模型获取对生成的类似查询的响应来获得用于本地微调的数据。然后，我们优化攻击后缀来生成攻击提示，并评估我们的局部微调对攻击成功率的影响。实验表明，代理模型的局部微调提高了攻击的可转移性，使目标模型ChatGPT、GPT-4和Claude的攻击成功率分别提高了39美元、7美元和0.5美元(绝对)。



## **24. An LLM can Fool Itself: A Prompt-Based Adversarial Attack**

LLM可以自欺欺人：基于提示的对抗性攻击 cs.CR

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2310.13345v1) [paper-pdf](http://arxiv.org/pdf/2310.13345v1)

**Authors**: Xilie Xu, Keyi Kong, Ning Liu, Lizhen Cui, Di Wang, Jingfeng Zhang, Mohan Kankanhalli

**Abstract**: The wide-ranging applications of large language models (LLMs), especially in safety-critical domains, necessitate the proper evaluation of the LLM's adversarial robustness. This paper proposes an efficient tool to audit the LLM's adversarial robustness via a prompt-based adversarial attack (PromptAttack). PromptAttack converts adversarial textual attacks into an attack prompt that can cause the victim LLM to output the adversarial sample to fool itself. The attack prompt is composed of three important components: (1) original input (OI) including the original sample and its ground-truth label, (2) attack objective (AO) illustrating a task description of generating a new sample that can fool itself without changing the semantic meaning, and (3) attack guidance (AG) containing the perturbation instructions to guide the LLM on how to complete the task by perturbing the original sample at character, word, and sentence levels, respectively. Besides, we use a fidelity filter to ensure that PromptAttack maintains the original semantic meanings of the adversarial examples. Further, we enhance the attack power of PromptAttack by ensembling adversarial examples at different perturbation levels. Comprehensive empirical results using Llama2 and GPT-3.5 validate that PromptAttack consistently yields a much higher attack success rate compared to AdvGLUE and AdvGLUE++. Interesting findings include that a simple emoji can easily mislead GPT-3.5 to make wrong predictions.

摘要: 大型语言模型(LLM)的广泛应用，特别是在安全关键领域，需要对LLM的攻击健壮性进行适当的评估。提出了一种通过基于提示的敌意攻击(PromptAttack)来审计LLM攻击健壮性的有效工具。PromptAttack将对抗性文本攻击转换为攻击提示，这可能会导致受害者LLM输出对抗性样本来愚弄自己。攻击提示由三个重要部分组成：(1)原始输入(OI)，包括原始样本及其地面事实标签；(2)攻击目标(AO)，说明生成可以在不改变语义的情况下欺骗自己的新样本的任务描述；(3)攻击指导(AG)，包含扰动指令，以分别在字、词和句子层面指导LLM如何通过扰动原始样本来完成任务。此外，我们使用了一个保真度过滤器来确保PromptAttack保持了对抗性例子的原始语义。此外，我们通过集成不同扰动级别的对抗性实例来增强PromptAttack的攻击能力。使用Llama2和GPT-3.5的综合实验结果验证了PromptAttack始终比AdvGLUE和AdvGLUE++产生更高的攻击成功率。有趣的发现包括，一个简单的表情符号很容易误导GPT-3.5做出错误的预测。



## **25. Mitigating Backdoor Poisoning Attacks through the Lens of Spurious Correlation**

通过伪关联镜头缓解后门中毒攻击 cs.CL

accepted to EMNLP2023 (main conference)

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2305.11596v2) [paper-pdf](http://arxiv.org/pdf/2305.11596v2)

**Authors**: Xuanli He, Qiongkai Xu, Jun Wang, Benjamin Rubinstein, Trevor Cohn

**Abstract**: Modern NLP models are often trained over large untrusted datasets, raising the potential for a malicious adversary to compromise model behaviour. For instance, backdoors can be implanted through crafting training instances with a specific textual trigger and a target label. This paper posits that backdoor poisoning attacks exhibit \emph{spurious correlation} between simple text features and classification labels, and accordingly, proposes methods for mitigating spurious correlation as means of defence. Our empirical study reveals that the malicious triggers are highly correlated to their target labels; therefore such correlations are extremely distinguishable compared to those scores of benign features, and can be used to filter out potentially problematic instances. Compared with several existing defences, our defence method significantly reduces attack success rates across backdoor attacks, and in the case of insertion-based attacks, our method provides a near-perfect defence.

摘要: 现代NLP模型通常是在不可信的大型数据集上进行训练的，这增加了恶意对手危害模型行为的可能性。例如，可以通过制作带有特定文本触发器和目标标签的训练实例来植入后门。文章假设后门中毒攻击在简单文本特征和分类标签之间表现出虚假相关性，并相应地提出了减少虚假相关性的方法作为防御手段。我们的经验研究表明，恶意触发器与其目标标签高度相关；因此，与那些良性特征相比，这种相关性非常容易区分，可以用来过滤潜在的问题实例。与现有的几种防御方法相比，我们的防御方法显著降低了后门攻击的攻击成功率，并且在基于插入的攻击中，我们的方法提供了近乎完美的防御。



## **26. Assessing Privacy Risks in Language Models: A Case Study on Summarization Tasks**

评估语言模型中的隐私风险：摘要任务的案例研究 cs.CL

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2310.13291v1) [paper-pdf](http://arxiv.org/pdf/2310.13291v1)

**Authors**: Ruixiang Tang, Gord Lueck, Rodolfo Quispe, Huseyin A Inan, Janardhan Kulkarni, Xia Hu

**Abstract**: Large language models have revolutionized the field of NLP by achieving state-of-the-art performance on various tasks. However, there is a concern that these models may disclose information in the training data. In this study, we focus on the summarization task and investigate the membership inference (MI) attack: given a sample and black-box access to a model's API, it is possible to determine if the sample was part of the training data. We exploit text similarity and the model's resistance to document modifications as potential MI signals and evaluate their effectiveness on widely used datasets. Our results demonstrate that summarization models are at risk of exposing data membership, even in cases where the reference summary is not available. Furthermore, we discuss several safeguards for training summarization models to protect against MI attacks and discuss the inherent trade-off between privacy and utility.

摘要: 大型语言模型通过在各种任务上实现最先进的性能，使NLP领域发生了革命性的变化。然而，人们担心这些模型可能会泄露训练数据中的信息。在这项研究中，我们将重点放在总结任务上，并研究成员关系推理(MI)攻击：给定一个样本和对模型API的黑盒访问，可以确定该样本是否为训练数据的一部分。我们利用文本相似性和模型对文档修改的抵抗力作为潜在的MI信号，并在广泛使用的数据集上评估它们的有效性。我们的结果表明，摘要模型存在暴露数据成员身份的风险，即使在参考摘要不可用的情况下也是如此。此外，我们讨论了训练摘要模型的几种保护措施，以防止MI攻击，并讨论了隐私和效用之间的内在权衡。



## **27. Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy for Language Models**

朝向稳健剪枝：一种自适应的语言模型知识保留剪枝策略 cs.CL

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.13191v1) [paper-pdf](http://arxiv.org/pdf/2310.13191v1)

**Authors**: Jianwei Li, Qi Lei, Wei Cheng, Dongkuan Xu

**Abstract**: The pruning objective has recently extended beyond accuracy and sparsity to robustness in language models. Despite this, existing methods struggle to enhance robustness against adversarial attacks when continually increasing model sparsity and require a retraining process. As humans step into the era of large language models, these issues become increasingly prominent. This paper proposes that the robustness of language models is proportional to the extent of pre-trained knowledge they encompass. Accordingly, we introduce a post-training pruning strategy designed to faithfully replicate the embedding space and feature space of dense language models, aiming to conserve more pre-trained knowledge during the pruning process. In this setup, each layer's reconstruction error not only originates from itself but also includes cumulative error from preceding layers, followed by an adaptive rectification. Compared to other state-of-art baselines, our approach demonstrates a superior balance between accuracy, sparsity, robustness, and pruning cost with BERT on datasets SST2, IMDB, and AGNews, marking a significant stride towards robust pruning in language models.

摘要: 修剪目标最近已经超越了语言模型中的精确度和稀疏性，扩展到了健壮性。尽管如此，现有的方法在不断增加模型稀疏性的同时努力增强对敌对攻击的鲁棒性，并且需要重新训练过程。随着人类步入大型语言模型时代，这些问题变得日益突出。本文提出语言模型的稳健性与它们所包含的预训练知识的程度成正比。因此，我们提出了一种训练后剪枝策略，旨在忠实地复制密集语言模型的嵌入空间和特征空间，目的是在剪枝过程中保存更多的预先训练的知识。在这种设置中，每一层的重建误差不仅源于自身，还包括来自前几层的累积误差，然后进行自适应校正。与其他最先进的基线相比，我们的方法在精确度、稀疏性、健壮性和剪枝成本之间表现出了更好的平衡，在数据集Sst2、IMDB和AgNews上使用ERT，标志着在语言模型中朝着健壮剪枝迈出了重要的一步。



## **28. Prompt Injection Attacks and Defenses in LLM-Integrated Applications**

LLM集成应用中的快速注入攻击与防御 cs.CR

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12815v1) [paper-pdf](http://arxiv.org/pdf/2310.12815v1)

**Authors**: Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, Neil Zhenqiang Gong

**Abstract**: Large Language Models (LLMs) are increasingly deployed as the backend for a variety of real-world applications called LLM-Integrated Applications. Multiple recent works showed that LLM-Integrated Applications are vulnerable to prompt injection attacks, in which an attacker injects malicious instruction/data into the input of those applications such that they produce results as the attacker desires. However, existing works are limited to case studies. As a result, the literature lacks a systematic understanding of prompt injection attacks and their defenses. We aim to bridge the gap in this work. In particular, we propose a general framework to formalize prompt injection attacks. Existing attacks, which are discussed in research papers and blog posts, are special cases in our framework. Our framework enables us to design a new attack by combining existing attacks. Moreover, we also propose a framework to systematize defenses against prompt injection attacks. Using our frameworks, we conduct a systematic evaluation on prompt injection attacks and their defenses with 10 LLMs and 7 tasks. We hope our frameworks can inspire future research in this field. Our code is available at https://github.com/liu00222/Open-Prompt-Injection.

摘要: 大型语言模型(LLM)越来越多地被部署为各种实际应用程序的后端，这些应用程序称为LLm集成应用程序。最近的多项研究表明，LLM集成的应用程序容易受到即时注入攻击，即攻击者将恶意指令/数据注入到这些应用程序的输入中，以便它们产生攻击者想要的结果。然而，现有的研究成果仅限于案例研究。因此，文献对快速注射攻击及其防御缺乏系统的了解。我们的目标是弥合这项工作中的差距。特别是，我们提出了一个形式化的快速注入攻击的通用框架。研究论文和博客文章中讨论的现有攻击在我们的框架中是特例。我们的框架使我们能够通过组合现有的攻击来设计新的攻击。此外，我们还提出了一个对快速注入攻击进行系统化防御的框架。使用我们的框架，我们对快速注入攻击及其防御进行了系统的评估，包括10个LLM和7个任务。我们希望我们的框架能对这一领域的未来研究有所启发。我们的代码可以在https://github.com/liu00222/Open-Prompt-Injection.上找到



## **29. Automatic Hallucination Assessment for Aligned Large Language Models via Transferable Adversarial Attacks**

基于可转移对抗性攻击的对齐大语言模型的自动幻觉评估 cs.CL

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12516v1) [paper-pdf](http://arxiv.org/pdf/2310.12516v1)

**Authors**: Xiaodong Yu, Hao Cheng, Xiaodong Liu, Dan Roth, Jianfeng Gao

**Abstract**: Although remarkable progress has been achieved in preventing large language model (LLM) hallucinations using instruction tuning and retrieval augmentation, it remains challenging to measure the reliability of LLMs using human-crafted evaluation data which is not available for many tasks and domains and could suffer from data leakage. Inspired by adversarial machine learning, this paper aims to develop a method of automatically generating evaluation data by appropriately modifying existing data on which LLMs behave faithfully. Specifically, this paper presents AutoDebug, an LLM-based framework to use prompting chaining to generate transferable adversarial attacks in the form of question-answering examples. We seek to understand the extent to which these examples trigger the hallucination behaviors of LLMs.   We implement AutoDebug using ChatGPT and evaluate the resulting two variants of a popular open-domain question-answering dataset, Natural Questions (NQ), on a collection of open-source and proprietary LLMs under various prompting settings. Our generated evaluation data is human-readable and, as we show, humans can answer these modified questions well. Nevertheless, we observe pronounced accuracy drops across multiple LLMs including GPT-4. Our experimental results show that LLMs are likely to hallucinate in two categories of question-answering scenarios where (1) there are conflicts between knowledge given in the prompt and their parametric knowledge, or (2) the knowledge expressed in the prompt is complex. Finally, we find that the adversarial examples generated by our method are transferable across all considered LLMs. The examples generated by a small model can be used to debug a much larger model, making our approach cost-effective.

摘要: 虽然在使用指令调优和检索增强来防止大型语言模型（LLM）幻觉方面已经取得了显着的进展，但是使用人工制作的评估数据来测量LLM的可靠性仍然具有挑战性，这些评估数据对于许多任务和领域都不可用，并且可能会遭受数据泄漏。受对抗机器学习的启发，本文旨在开发一种通过适当修改LLM忠实表现的现有数据来自动生成评估数据的方法。具体来说，本文介绍了AutoProximity，一个基于LLM的框架，使用提示链接以问答示例的形式生成可转移的对抗性攻击。我们试图了解这些例子在多大程度上触发了LLM的幻觉行为。   我们使用ChatGPT实现了自动搜索，并在各种提示设置下，在开源和专有LLM的集合上评估了流行的开放域问答数据集Natural Questions（NQ）的两个变体。我们生成的评估数据是人类可读的，正如我们所展示的，人类可以很好地回答这些修改后的问题。然而，我们观察到包括GPT-4在内的多个LLM的准确性明显下降。我们的实验结果表明，LLM可能会在两类问答场景中产生幻觉，其中（1）提示中给出的知识与其参数知识之间存在冲突，或者（2）提示中表达的知识是复杂的。最后，我们发现，我们的方法生成的对抗性的例子是可转移的所有考虑LLM。小模型生成的示例可以用于调试更大的模型，使我们的方法具有成本效益。



## **30. Attack Prompt Generation for Red Teaming and Defending Large Language Models**

红色团队攻击提示生成及大型语言模型防御 cs.CL

Accepted to EMNLP 2023 (Findings)

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12505v1) [paper-pdf](http://arxiv.org/pdf/2310.12505v1)

**Authors**: Boyi Deng, Wenjie Wang, Fuli Feng, Yang Deng, Qifan Wang, Xiangnan He

**Abstract**: Large language models (LLMs) are susceptible to red teaming attacks, which can induce LLMs to generate harmful content. Previous research constructs attack prompts via manual or automatic methods, which have their own limitations on construction cost and quality. To address these issues, we propose an integrated approach that combines manual and automatic methods to economically generate high-quality attack prompts. Specifically, considering the impressive capabilities of newly emerged LLMs, we propose an attack framework to instruct LLMs to mimic human-generated prompts through in-context learning. Furthermore, we propose a defense framework that fine-tunes victim LLMs through iterative interactions with the attack framework to enhance their safety against red teaming attacks. Extensive experiments on different LLMs validate the effectiveness of our proposed attack and defense frameworks. Additionally, we release a series of attack prompts datasets named SAP with varying sizes, facilitating the safety evaluation and enhancement of more LLMs. Our code and dataset is available on https://github.com/Aatrox103/SAP .

摘要: 大型语言模型(LLM)容易受到红色团队攻击，从而导致LLM生成有害内容。以往的研究通过人工或自动的方法构建攻击提示，这两种方法在构建成本和质量方面都有各自的局限性。为了解决这些问题，我们提出了一种将手动和自动方法相结合的集成方法，以经济地生成高质量的攻击提示。具体地说，考虑到新出现的LLMS令人印象深刻的能力，我们提出了一个攻击框架，通过上下文学习来指示LLMS模仿人类生成的提示。此外，我们提出了一个防御框架，通过与攻击框架的迭代交互来微调受害者LLM，以增强他们对Red Teaming攻击的安全性。在不同LLM上的大量实验验证了我们提出的攻防框架的有效性。此外，我们还发布了一系列名为SAP的攻击提示数据集，这些数据集的大小不一，有助于更多LLM的安全评估和增强。我们的代码和数据集可以在https://github.com/Aatrox103/SAP上找到。



## **31. Red Teaming Language Model Detectors with Language Models**

具有语言模型的Red Teaming语言模型检测器 cs.CL

Preprint. Accepted by TACL

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2305.19713v2) [paper-pdf](http://arxiv.org/pdf/2305.19713v2)

**Authors**: Zhouxing Shi, Yihan Wang, Fan Yin, Xiangning Chen, Kai-Wei Chang, Cho-Jui Hsieh

**Abstract**: The prevalence and strong capability of large language models (LLMs) present significant safety and ethical risks if exploited by malicious users. To prevent the potentially deceptive usage of LLMs, recent works have proposed algorithms to detect LLM-generated text and protect LLMs. In this paper, we investigate the robustness and reliability of these LLM detectors under adversarial attacks. We study two types of attack strategies: 1) replacing certain words in an LLM's output with their synonyms given the context; 2) automatically searching for an instructional prompt to alter the writing style of the generation. In both strategies, we leverage an auxiliary LLM to generate the word replacements or the instructional prompt. Different from previous works, we consider a challenging setting where the auxiliary LLM can also be protected by a detector. Experiments reveal that our attacks effectively compromise the performance of all detectors in the study with plausible generations, underscoring the urgent need to improve the robustness of LLM-generated text detection systems.

摘要: 如果被恶意用户利用，大语言模型(LLM)的流行和强大的能力会带来巨大的安全和道德风险。为了防止潜在的欺骗性使用LLMS，最近的工作提出了检测LLM生成的文本并保护LLMS的算法。在本文中，我们研究了这些LLM检测器在对抗攻击下的健壮性和可靠性。我们研究了两种类型的攻击策略：1)用给定上下文的同义词替换LLM输出中的某些单词；2)自动搜索指令提示以改变生成的写作风格。在这两种策略中，我们利用辅助LLM来生成单词替换或指令提示。与以前的工作不同，我们考虑了一个具有挑战性的设置，其中辅助LLM也可以受到探测器的保护。实验表明，我们的攻击有效地折衷了研究中所有检测器的性能，生成了看似合理的代码，这突显了提高LLM生成的文本检测系统的健壮性的迫切需要。



## **32. PoisonPrompt: Backdoor Attack on Prompt-based Large Language Models**

PoisonPrompt：对基于提示的大型语言模型的后门攻击 cs.CL

Code will be released on: https://github.com/grasses/PoisonPrompt

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12439v1) [paper-pdf](http://arxiv.org/pdf/2310.12439v1)

**Authors**: Hongwei Yao, Jian Lou, Zhan Qin

**Abstract**: Prompts have significantly improved the performance of pretrained Large Language Models (LLMs) on various downstream tasks recently, making them increasingly indispensable for a diverse range of LLM application scenarios. However, the backdoor vulnerability, a serious security threat that can maliciously alter the victim model's normal predictions, has not been sufficiently explored for prompt-based LLMs. In this paper, we present POISONPROMPT, a novel backdoor attack capable of successfully compromising both hard and soft prompt-based LLMs. We evaluate the effectiveness, fidelity, and robustness of POISONPROMPT through extensive experiments on three popular prompt methods, using six datasets and three widely used LLMs. Our findings highlight the potential security threats posed by backdoor attacks on prompt-based LLMs and emphasize the need for further research in this area.

摘要: 最近，提示显著提高了预先训练的大型语言模型(LLM)在各种下游任务上的性能，使它们在各种LLM应用场景中越来越不可或缺。然而，后门漏洞是一个严重的安全威胁，可能会恶意改变受害者模型的正常预测，但对于基于提示的LLM来说，这种漏洞还没有得到充分的研究。在本文中，我们提出了一种新的后门攻击POISONPROMPT，它能够成功地攻破基于硬提示和软提示的LLMS。我们使用6个数据集和3个广泛使用的LLMS对POISONPROMPT的有效性、保真度和稳健性进行了评估。我们的发现突出了后门攻击对基于提示的LLM的潜在安全威胁，并强调了在这一领域进行进一步研究的必要性。



## **33. REMARK-LLM: A Robust and Efficient Watermarking Framework for Generative Large Language Models**

Remmark-LLM：一种面向生成性大语言模型的健壮高效水印框架 cs.CR

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2310.12362v1) [paper-pdf](http://arxiv.org/pdf/2310.12362v1)

**Authors**: Ruisi Zhang, Shehzeen Samarah Hussain, Paarth Neekhara, Farinaz Koushanfar

**Abstract**: We present REMARK-LLM, a novel efficient, and robust watermarking framework designed for texts generated by large language models (LLMs). Synthesizing human-like content using LLMs necessitates vast computational resources and extensive datasets, encapsulating critical intellectual property (IP). However, the generated content is prone to malicious exploitation, including spamming and plagiarism. To address the challenges, REMARK-LLM proposes three new components: (i) a learning-based message encoding module to infuse binary signatures into LLM-generated texts; (ii) a reparameterization module to transform the dense distributions from the message encoding to the sparse distribution of the watermarked textual tokens; (iii) a decoding module dedicated for signature extraction; Furthermore, we introduce an optimized beam search algorithm to guarantee the coherence and consistency of the generated content. REMARK-LLM is rigorously trained to encourage the preservation of semantic integrity in watermarked content, while ensuring effective watermark retrieval. Extensive evaluations on multiple unseen datasets highlight REMARK-LLM proficiency and transferability in inserting 2 times more signature bits into the same texts when compared to prior art, all while maintaining semantic integrity. Furthermore, REMARK-LLM exhibits better resilience against a spectrum of watermark detection and removal attacks.

摘要: 我们提出了一种新的高效、健壮的数字水印框架--REMARK-LLM，它适用于大型语言模型(LLMS)生成的文本。使用LLMS合成类似人类的内容需要大量的计算资源和大量的数据集，并封装关键知识产权(IP)。然而，生成的内容容易受到恶意攻击，包括垃圾邮件和抄袭。为了应对这些挑战，REMARK-LLM提出了三个新的组件：(I)基于学习的消息编码模块，将二进制签名注入到LLM生成的文本中；(Ii)重新参数化模块，将密集分布的消息编码转换为稀疏分布的水印文本标记；(Iii)专门用于签名提取的解码模块；此外，我们还引入了优化的波束搜索算法，以保证生成内容的一致性和一致性。Rmark-LLM经过了严格的培训，以鼓励在水印内容中保持语义完整性，同时确保有效的水印检索。对多个不可见数据集的广泛评估突出了Remmark-LLM的熟练程度和可转移性，即与现有技术相比，在相同的文本中插入2倍多的签名比特，同时保持语义完整性。此外，REMARK-LLM对一系列水印检测和删除攻击表现出更好的弹性。



## **34. PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

PromptBitch：评估大型语言模型在对抗性提示下的稳健性 cs.CL

Technical report; code is at:  https://github.com/microsoft/promptbench

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2306.04528v4) [paper-pdf](http://arxiv.org/pdf/2306.04528v4)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Yue Zhang, Neil Zhenqiang Gong, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptBench, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. The adversarial prompts, crafted to mimic plausible user errors like typos or synonyms, aim to evaluate how slight deviations can affect LLM outcomes while maintaining semantic integrity. These prompts are then employed in diverse tasks, such as sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4788 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets. Our findings demonstrate that contemporary LLMs are not robust to adversarial prompts. Furthermore, we present comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users. Code is available at: https://github.com/microsoft/promptbench.

摘要: 学术界和工业界对大型语言模型(LLM)的依赖日益增加，这就要求我们必须全面了解它们对提示的稳健性。为了响应这一关键需求，我们引入了PromptBtch，这是一个健壮性基准，旨在衡量LLMS对敌意提示的弹性。这项研究使用了过多的对抗性文本攻击，目标是多个层面的提示：字符、单词、句子和语义。这些对抗性提示旨在模仿打字或同义词等看似合理的用户错误，旨在评估微小的偏差如何在保持语义完整性的同时影响LLM结果。这些提示随后被用于不同的任务，如情感分析、自然语言推理、阅读理解、机器翻译和数学解题。我们的研究产生了4788个对抗性提示，仔细评估了8个任务和13个数据集。我们的研究结果表明，当代的LLM对敌意提示并不健壮。此外，我们还提供了全面的分析，以了解即时健壮性及其可转移性背后的奥秘。然后，我们提供了有洞察力的健壮性分析和实用的即时撰写建议，这对研究人员和日常用户都是有益的。代码可从以下网址获得：https://github.com/microsoft/promptbench.



## **35. Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense**

释义可以躲避人工智能生成的文本的检测，但检索是一种有效的防御 cs.CL

NeurIPS 2023 camera ready (32 pages). Code, models, data available in  https://github.com/martiansideofthemoon/ai-detection-paraphrases

**SubmitDate**: 2023-10-18    [abs](http://arxiv.org/abs/2303.13408v2) [paper-pdf](http://arxiv.org/pdf/2303.13408v2)

**Authors**: Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, Mohit Iyyer

**Abstract**: The rise in malicious usage of large language models, such as fake content creation and academic plagiarism, has motivated the development of approaches that identify AI-generated text, including those based on watermarking or outlier detection. However, the robustness of these detection algorithms to paraphrases of AI-generated text remains unclear. To stress test these detectors, we build a 11B parameter paraphrase generation model (DIPPER) that can paraphrase paragraphs, condition on surrounding context, and control lexical diversity and content reordering. Using DIPPER to paraphrase text generated by three large language models (including GPT3.5-davinci-003) successfully evades several detectors, including watermarking, GPTZero, DetectGPT, and OpenAI's text classifier. For example, DIPPER drops detection accuracy of DetectGPT from 70.3% to 4.6% (at a constant false positive rate of 1%), without appreciably modifying the input semantics.   To increase the robustness of AI-generated text detection to paraphrase attacks, we introduce a simple defense that relies on retrieving semantically-similar generations and must be maintained by a language model API provider. Given a candidate text, our algorithm searches a database of sequences previously generated by the API, looking for sequences that match the candidate text within a certain threshold. We empirically verify our defense using a database of 15M generations from a fine-tuned T5-XXL model and find that it can detect 80% to 97% of paraphrased generations across different settings while only classifying 1% of human-written sequences as AI-generated. We open-source our models, code and data.

摘要: 恶意使用大型语言模型的增加，如虚假内容创作和学术抄袭，推动了识别人工智能生成文本的方法的发展，包括基于水印或离群值检测的方法。然而，这些检测算法对人工智能生成的文本的释义的稳健性尚不清楚。为了对这些检测器进行压力测试，我们建立了一个11B参数转述生成模型(Dipper)，该模型可以转译段落、对周围上下文进行条件转换、控制词汇多样性和内容重新排序。使用Dipper解释由三个大型语言模型(包括GPT3.5-DaVinci-003)生成的文本，成功地避开了几个检测器，包括水印、GPTZero、DetectGPT和OpenAI的文本分类器。例如，Dipper将DetectGPT的检测准确率从70.3%下降到4.6%(在1%的恒定假阳性率下)，而不会明显修改输入语义。为了提高人工智能生成的文本检测对转述攻击的健壮性，我们引入了一种简单的防御，该防御依赖于检索语义相似的生成，并且必须由语言模型API提供商维护。在给定候选文本的情况下，我们的算法搜索先前由API生成的序列数据库，寻找在特定阈值内与候选文本匹配的序列。我们使用来自微调的T5-XXL模型的1500万代数据库经验地验证了我们的防御，发现它可以检测到不同设置下80%到97%的释义世代，而只有1%的人写序列被归类为人工智能生成的序列。我们将我们的模型、代码和数据开源。



## **36. Identifying and Mitigating the Security Risks of Generative AI**

识别和缓解生成性人工智能的安全风险 cs.AI

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2308.14840v3) [paper-pdf](http://arxiv.org/pdf/2308.14840v3)

**Authors**: Clark Barrett, Brad Boyd, Elie Burzstein, Nicholas Carlini, Brad Chen, Jihye Choi, Amrita Roy Chowdhury, Mihai Christodorescu, Anupam Datta, Soheil Feizi, Kathleen Fisher, Tatsunori Hashimoto, Dan Hendrycks, Somesh Jha, Daniel Kang, Florian Kerschbaum, Eric Mitchell, John Mitchell, Zulfikar Ramzan, Khawaja Shams, Dawn Song, Ankur Taly, Diyi Yang

**Abstract**: Every major technical invention resurfaces the dual-use dilemma -- the new technology has the potential to be used for good as well as for harm. Generative AI (GenAI) techniques, such as large language models (LLMs) and diffusion models, have shown remarkable capabilities (e.g., in-context learning, code-completion, and text-to-image generation and editing). However, GenAI can be used just as well by attackers to generate new attacks and increase the velocity and efficacy of existing attacks.   This paper reports the findings of a workshop held at Google (co-organized by Stanford University and the University of Wisconsin-Madison) on the dual-use dilemma posed by GenAI. This paper is not meant to be comprehensive, but is rather an attempt to synthesize some of the interesting findings from the workshop. We discuss short-term and long-term goals for the community on this topic. We hope this paper provides both a launching point for a discussion on this important topic as well as interesting problems that the research community can work to address.

摘要: 每一项重大技术发明都会重新面临两难境地--新技术既有可能被用来做好事，也有可能被用来做坏事。生成性人工智能(GenAI)技术，如大型语言模型(LLMS)和扩散模型，已经显示出非凡的能力(例如，上下文学习、代码完成以及文本到图像的生成和编辑)。然而，攻击者也可以利用GenAI来生成新的攻击，并提高现有攻击的速度和效率。本文报告了在谷歌(由斯坦福大学和威斯康星大学麦迪逊分校联合举办)举行的关于GenAI造成的两用困境的研讨会的结果。这篇论文并不是要全面的，而是试图综合研讨会的一些有趣的发现。我们就这一主题讨论社区的短期和长期目标。我们希望这篇论文既为讨论这一重要主题提供了一个起点，也为研究界可以努力解决的有趣问题提供了一个起点。



## **37. Last One Standing: A Comparative Analysis of Security and Privacy of Soft Prompt Tuning, LoRA, and In-Context Learning**

最后一站：软提示调整，LoRA和上下文学习的安全性和隐私性的比较分析 cs.CR

**SubmitDate**: 2023-10-17    [abs](http://arxiv.org/abs/2310.11397v1) [paper-pdf](http://arxiv.org/pdf/2310.11397v1)

**Authors**: Rui Wen, Tianhao Wang, Michael Backes, Yang Zhang, Ahmed Salem

**Abstract**: Large Language Models (LLMs) are powerful tools for natural language processing, enabling novel applications and user experiences. However, to achieve optimal performance, LLMs often require adaptation with private data, which poses privacy and security challenges. Several techniques have been proposed to adapt LLMs with private data, such as Low-Rank Adaptation (LoRA), Soft Prompt Tuning (SPT), and In-Context Learning (ICL), but their comparative privacy and security properties have not been systematically investigated. In this work, we fill this gap by evaluating the robustness of LoRA, SPT, and ICL against three types of well-established attacks: membership inference, which exposes data leakage (privacy); backdoor, which injects malicious behavior (security); and model stealing, which can violate intellectual property (privacy and security). Our results show that there is no silver bullet for privacy and security in LLM adaptation and each technique has different strengths and weaknesses.

摘要: 大型语言模型(LLM)是自然语言处理的强大工具，支持新的应用程序和用户体验。然而，为了实现最佳性能，LLMS通常需要适应私有数据，这会带来隐私和安全挑战。已经提出了几种技术来适应具有私有数据的LLMS，如低阶自适应(LORA)、软提示调整(SPT)和上下文中学习(ICL)，但它们的相对隐私和安全特性还没有被系统地研究。在这项工作中，我们通过评估LORA、SPT和ICL针对三种常见攻击的健壮性来填补这一空白：暴露数据泄露(隐私)的成员资格推断；注入恶意行为(安全)的后门；以及可能侵犯知识产权(隐私和安全)的模型窃取。我们的结果表明，在LLM自适应中没有隐私和安全的灵丹妙药，每种技术都有不同的优势和劣势。



## **38. Survey of Vulnerabilities in Large Language Models Revealed by Adversarial Attacks**

对抗性攻击揭示的大型语言模型中的漏洞调查 cs.CL

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10844v1) [paper-pdf](http://arxiv.org/pdf/2310.10844v1)

**Authors**: Erfan Shayegani, Md Abdullah Al Mamun, Yu Fu, Pedram Zaree, Yue Dong, Nael Abu-Ghazaleh

**Abstract**: Large Language Models (LLMs) are swiftly advancing in architecture and capability, and as they integrate more deeply into complex systems, the urgency to scrutinize their security properties grows. This paper surveys research in the emerging interdisciplinary field of adversarial attacks on LLMs, a subfield of trustworthy ML, combining the perspectives of Natural Language Processing and Security. Prior work has shown that even safety-aligned LLMs (via instruction tuning and reinforcement learning through human feedback) can be susceptible to adversarial attacks, which exploit weaknesses and mislead AI systems, as evidenced by the prevalence of `jailbreak' attacks on models like ChatGPT and Bard. In this survey, we first provide an overview of large language models, describe their safety alignment, and categorize existing research based on various learning structures: textual-only attacks, multi-modal attacks, and additional attack methods specifically targeting complex systems, such as federated learning or multi-agent systems. We also offer comprehensive remarks on works that focus on the fundamental sources of vulnerabilities and potential defenses. To make this field more accessible to newcomers, we present a systematic review of existing works, a structured typology of adversarial attack concepts, and additional resources, including slides for presentations on related topics at the 62nd Annual Meeting of the Association for Computational Linguistics (ACL'24).

摘要: 大型语言模型(LLM)在体系结构和功能方面正在迅速发展，随着它们更深入地集成到复杂系统中，审查其安全属性的紧迫性也在增长。本文结合自然语言处理和安全的角度，对可信ML的一个子领域--LLMS的对抗性攻击这一新兴交叉学科领域的研究进行了综述。先前的工作表明，即使是与安全一致的LLM(通过指令调整和通过人类反馈的强化学习)也可能容易受到对手攻击，这些攻击利用弱点并误导人工智能系统，对ChatGPT和Bard等模型的“越狱”攻击盛行就是明证。在这次调查中，我们首先提供了大型语言模型的概述，描述了它们的安全对齐，并基于各种学习结构对现有研究进行了分类：纯文本攻击、多模式攻击以及专门针对复杂系统的额外攻击方法，如联合学习或多代理系统。我们还对侧重于漏洞的根本来源和潜在防御的工作进行了全面的评论。为了使这个领域更容易为新手所接受，我们提供了对现有工作的系统回顾，对抗性攻击概念的结构化类型学，以及额外的资源，包括在第62届计算语言学协会年会(ACL‘24)上相关主题的演示幻灯片。



## **39. Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks**

披着羊皮的假新闻：强大的假新闻检测对抗LLM授权的风格攻击 cs.CL

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10830v1) [paper-pdf](http://arxiv.org/pdf/2310.10830v1)

**Authors**: Jiaying Wu, Bryan Hooi

**Abstract**: It is commonly perceived that online fake news and reliable news exhibit stark differences in writing styles, such as the use of sensationalist versus objective language. However, we emphasize that style-related features can also be exploited for style-based attacks. Notably, the rise of powerful Large Language Models (LLMs) has enabled malicious users to mimic the style of trustworthy news outlets at minimal cost. Our analysis reveals that LLM-camouflaged fake news content leads to substantial performance degradation of state-of-the-art text-based detectors (up to 38% decrease in F1 Score), posing a significant challenge for automated detection in online ecosystems. To address this, we introduce SheepDog, a style-agnostic fake news detector robust to news writing styles. SheepDog achieves this adaptability through LLM-empowered news reframing, which customizes each article to match different writing styles using style-oriented reframing prompts. By employing style-agnostic training, SheepDog enhances its resilience to stylistic variations by maximizing prediction consistency across these diverse reframings. Furthermore, SheepDog extracts content-focused veracity attributions from LLMs, where the news content is evaluated against a set of fact-checking rationales. These attributions provide supplementary information and potential interpretability that assist veracity prediction. On three benchmark datasets, empirical results show that SheepDog consistently yields significant improvements over competitive baselines and enhances robustness against LLM-empowered style attacks.

摘要: 人们普遍认为，网络假新闻和可靠新闻在写作风格上表现出明显的差异，例如使用耸人听闻的语言和客观语言。但是，我们强调，与样式相关的功能也可以用于基于样式的攻击。值得注意的是，强大的大型语言模型(LLM)的兴起使恶意用户能够以最低成本模仿值得信赖的新闻机构的风格。我们的分析显示，LLM伪装的假新闻内容导致基于文本的最新检测器的性能大幅下降(F1分数下降高达38%)，对在线生态系统中的自动检测构成了重大挑战。为了解决这个问题，我们引入了SheepDog，这是一个风格不可知的假新闻检测器，对新闻写作风格非常健壮。SheepDog通过LLM授权的新闻重组实现了这种适应性，它使用面向风格的重组提示定制每篇文章，以匹配不同的写作风格。通过使用风格不可知训练，牧羊犬通过最大限度地提高这些不同重组的预测一致性，提高了对风格变化的适应能力。此外，SheepDog从LLMS中提取专注于内容的准确性属性，在LLMS中，根据一组事实核查原理对新闻内容进行评估。这些属性提供了有助于准确性预测的补充信息和潜在的可解释性。在三个基准数据集上的经验结果表明，SheepDog在竞争基线上始终具有显著的改善，并增强了对LLM授权的样式攻击的健壮性。



## **40. Privacy in Large Language Models: Attacks, Defenses and Future Directions**

大型语言模型中的隐私：攻击、防御和未来方向 cs.CL

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10383v1) [paper-pdf](http://arxiv.org/pdf/2310.10383v1)

**Authors**: Haoran Li, Yulin Chen, Jinglong Luo, Yan Kang, Xiaojin Zhang, Qi Hu, Chunkit Chan, Yangqiu Song

**Abstract**: The advancement of large language models (LLMs) has significantly enhanced the ability to effectively tackle various downstream NLP tasks and unify these tasks into generative pipelines. On the one hand, powerful language models, trained on massive textual data, have brought unparalleled accessibility and usability for both models and users. On the other hand, unrestricted access to these models can also introduce potential malicious and unintentional privacy risks. Despite ongoing efforts to address the safety and privacy concerns associated with LLMs, the problem remains unresolved. In this paper, we provide a comprehensive analysis of the current privacy attacks targeting LLMs and categorize them according to the adversary's assumed capabilities to shed light on the potential vulnerabilities present in LLMs. Then, we present a detailed overview of prominent defense strategies that have been developed to counter these privacy attacks. Beyond existing works, we identify upcoming privacy concerns as LLMs evolve. Lastly, we point out several potential avenues for future exploration.

摘要: 大型语言模型(LLM)的发展极大地增强了有效地处理各种下游NLP任务并将这些任务统一到生成管道中的能力。一方面，强大的语言模型，基于海量文本数据的训练，为模型和用户带来了无与伦比的可及性和可用性。另一方面，不受限制地访问这些模型也可能带来潜在的恶意和无意的隐私风险。尽管正在努力解决与低密度脂蛋白相关的安全和隐私问题，但这个问题仍然没有得到解决。在本文中，我们对当前针对LLMS的隐私攻击进行了全面的分析，并根据对手假设的能力对它们进行了分类，以揭示LLMS中存在的潜在漏洞。然后，我们详细概述了为应对这些隐私攻击而开发的主要防御策略。除了现有的工作，我们发现随着LLM的发展，即将到来的隐私问题。最后，我们指出了未来可能的几个探索方向。



## **41. Prompt Packer: Deceiving LLMs through Compositional Instruction with Hidden Attacks**

Prompt Packer：通过隐藏攻击的组合指令欺骗LLM cs.CL

**SubmitDate**: 2023-10-16    [abs](http://arxiv.org/abs/2310.10077v1) [paper-pdf](http://arxiv.org/pdf/2310.10077v1)

**Authors**: Shuyu Jiang, Xingshu Chen, Rui Tang

**Abstract**: Recently, Large language models (LLMs) with powerful general capabilities have been increasingly integrated into various Web applications, while undergoing alignment training to ensure that the generated content aligns with user intent and ethics. Unfortunately, they remain the risk of generating harmful content like hate speech and criminal activities in practical applications. Current approaches primarily rely on detecting, collecting, and training against harmful prompts to prevent such risks. However, they typically focused on the "superficial" harmful prompts with a solitary intent, ignoring composite attack instructions with multiple intentions that can easily elicit harmful content in real-world scenarios. In this paper, we introduce an innovative technique for obfuscating harmful instructions: Compositional Instruction Attacks (CIA), which refers to attacking by combination and encapsulation of multiple instructions. CIA hides harmful prompts within instructions of harmless intentions, making it impossible for the model to identify underlying malicious intentions. Furthermore, we implement two transformation methods, known as T-CIA and W-CIA, to automatically disguise harmful instructions as talking or writing tasks, making them appear harmless to LLMs. We evaluated CIA on GPT-4, ChatGPT, and ChatGLM2 with two safety assessment datasets and two harmful prompt datasets. It achieves an attack success rate of 95%+ on safety assessment datasets, and 83%+ for GPT-4, 91%+ for ChatGPT (gpt-3.5-turbo backed) and ChatGLM2-6B on harmful prompt datasets. Our approach reveals the vulnerability of LLMs to such compositional instruction attacks that harbor underlying harmful intentions, contributing significantly to LLM security development. Warning: this paper may contain offensive or upsetting content!

摘要: 最近，具有强大通用功能的大型语言模型(LLM)越来越多地集成到各种Web应用程序中，同时进行对齐培训，以确保生成的内容符合用户意图和道德规范。不幸的是，它们在实际应用中仍然存在产生仇恨言论和犯罪活动等有害内容的风险。目前的方法主要依靠检测、收集和针对有害提示进行培训来预防此类风险。然而，他们通常只关注带有单一意图的表面上的有害提示，而忽略具有多个意图的复合攻击指令，这些指令很容易在现实世界的场景中引发有害内容。本文介绍了一种新的混淆有害指令的技术：组合指令攻击(CIA)，它指的是通过组合和封装多条指令进行攻击。CIA将有害提示隐藏在无害意图的指令中，使得该模型无法识别潜在的恶意意图。此外，我们实现了两种转换方法，称为T-CIA和W-CIA，以自动将有害指令伪装成说话或写作任务，使它们对LLMS看起来是无害的。我们使用两个安全评估数据集和两个有害提示数据集对CIA在GPT-4、ChatGPT和ChatGLM2上进行了评估。在安全评估数据集上的攻击成功率为95%以上，在GPT-4上的攻击成功率为83%以上，对ChatGPT(GPT-3.5-Turbo支持)和ChatGLM2-6B上的攻击成功率为91%以上。我们的方法揭示了LLMS在此类组合指令攻击中的脆弱性，这些攻击隐藏着潜在的有害意图，对LLM安全发展做出了重大贡献。警告：本文可能包含冒犯性或令人不快的内容！



## **42. LMSanitator: Defending Prompt-Tuning Against Task-Agnostic Backdoors**

LMSanitator：防御与任务无关的后门的提示调整 cs.CL

To Appear in the Network and Distributed System Security (NDSS)  Symposium 2024, 26 February - 1 March 2024, San Diego, CA, USA; typos  corrected

**SubmitDate**: 2023-10-14    [abs](http://arxiv.org/abs/2308.13904v2) [paper-pdf](http://arxiv.org/pdf/2308.13904v2)

**Authors**: Chengkun Wei, Wenlong Meng, Zhikun Zhang, Min Chen, Minghu Zhao, Wenjing Fang, Lei Wang, Zihui Zhang, Wenzhi Chen

**Abstract**: Prompt-tuning has emerged as an attractive paradigm for deploying large-scale language models due to its strong downstream task performance and efficient multitask serving ability. Despite its wide adoption, we empirically show that prompt-tuning is vulnerable to downstream task-agnostic backdoors, which reside in the pretrained models and can affect arbitrary downstream tasks. The state-of-the-art backdoor detection approaches cannot defend against task-agnostic backdoors since they hardly converge in reversing the backdoor triggers. To address this issue, we propose LMSanitator, a novel approach for detecting and removing task-agnostic backdoors on Transformer models. Instead of directly inverting the triggers, LMSanitator aims to invert the predefined attack vectors (pretrained models' output when the input is embedded with triggers) of the task-agnostic backdoors, which achieves much better convergence performance and backdoor detection accuracy. LMSanitator further leverages prompt-tuning's property of freezing the pretrained model to perform accurate and fast output monitoring and input purging during the inference phase. Extensive experiments on multiple language models and NLP tasks illustrate the effectiveness of LMSanitator. For instance, LMSanitator achieves 92.8% backdoor detection accuracy on 960 models and decreases the attack success rate to less than 1% in most scenarios.

摘要: 由于其强大的下游任务性能和高效的多任务服务能力，即时调优已成为部署大规模语言模型的一个有吸引力的范例。尽管被广泛采用，我们的经验表明，即时调优很容易受到下游任务不可知的后门的影响，这些后门驻留在预先训练的模型中，可以影响任意的下游任务。最先进的后门检测方法无法防御与任务无关的后门，因为它们在逆转后门触发时几乎不会收敛。为了解决这个问题，我们提出了一种新的方法LMSanitator，用于检测和删除变压器模型上与任务无关的后门程序。LMSanitator不是直接反转触发器，而是反转与任务无关的后门的预定义攻击向量(当输入嵌入触发器时，预先训练的模型的输出)，从而获得更好的收敛性能和后门检测精度。LMSanitator还利用即时调整的冻结预训练模型的特性，在推理阶段执行准确而快速的输出监控和输入清除。在多种语言模型和自然语言处理任务上的大量实验表明了LMSanitator的有效性。例如，LMSanitator在960个机型上的后门检测准确率达到92.8%，在大多数场景下攻击成功率低于1%。



## **43. How Robust is Google's Bard to Adversarial Image Attacks?**

谷歌的吟游诗人对敌意图像攻击的健壮程度如何？ cs.CV

Technical report

**SubmitDate**: 2023-10-14    [abs](http://arxiv.org/abs/2309.11751v2) [paper-pdf](http://arxiv.org/pdf/2309.11751v2)

**Authors**: Yinpeng Dong, Huanran Chen, Jiawei Chen, Zhengwei Fang, Xiao Yang, Yichi Zhang, Yu Tian, Hang Su, Jun Zhu

**Abstract**: Multimodal Large Language Models (MLLMs) that integrate text and other modalities (especially vision) have achieved unprecedented performance in various multimodal tasks. However, due to the unsolved adversarial robustness problem of vision models, MLLMs can have more severe safety and security risks by introducing the vision inputs. In this work, we study the adversarial robustness of Google's Bard, a competitive chatbot to ChatGPT that released its multimodal capability recently, to better understand the vulnerabilities of commercial MLLMs. By attacking white-box surrogate vision encoders or MLLMs, the generated adversarial examples can mislead Bard to output wrong image descriptions with a 22% success rate based solely on the transferability. We show that the adversarial examples can also attack other MLLMs, e.g., a 26% attack success rate against Bing Chat and a 86% attack success rate against ERNIE bot. Moreover, we identify two defense mechanisms of Bard, including face detection and toxicity detection of images. We design corresponding attacks to evade these defenses, demonstrating that the current defenses of Bard are also vulnerable. We hope this work can deepen our understanding on the robustness of MLLMs and facilitate future research on defenses. Our code is available at https://github.com/thu-ml/Attack-Bard.   Update: GPT-4V is available at October 2023. We further evaluate its robustness under the same set of adversarial examples, achieving a 45% attack success rate.

摘要: 多通道大语言模型将文本和其他通道(尤其是视觉)结合在一起，在各种多通道任务中取得了前所未有的性能。然而，由于视觉模型的对抗性健壮性问题尚未解决，通过引入视觉输入，MLLMS可能存在更严重的安全风险。在这项工作中，我们研究了Google的Bard，一个与ChatGPT竞争的聊天机器人，最近发布了它的多模式功能，以更好地了解商业MLLMS的漏洞。通过攻击白盒代理视觉编码器或MLLM，生成的敌意示例可以误导BARD输出错误的图像描述，仅基于可转移性的成功率为22%。我们表明，恶意例子也可以攻击其他MLLMS，例如，对Bing Chat的攻击成功率为26%，对Ernie bot的攻击成功率为86%。此外，我们还识别了BARD的两种防御机制，包括人脸检测和图像毒性检测。我们设计了相应的攻击来逃避这些防御，证明了巴德目前的防御也是脆弱的。我们希望这项工作可以加深我们对MLLMS稳健性的理解，并为未来的防御研究提供便利。我们的代码可以在https://github.com/thu-ml/Attack-Bard.上找到更新：GPT-4V将于2023年10月上市。在相同的对抗性例子下，我们进一步评估了它的健壮性，达到了45%的攻击成功率。



## **44. Adversarial Demonstration Attacks on Large Language Models**

针对大型语言模型的对抗性演示攻击 cs.CL

**SubmitDate**: 2023-10-14    [abs](http://arxiv.org/abs/2305.14950v2) [paper-pdf](http://arxiv.org/pdf/2305.14950v2)

**Authors**: Jiongxiao Wang, Zichen Liu, Keun Hee Park, Zhuojun Jiang, Zhaoheng Zheng, Zhuofeng Wu, Muhao Chen, Chaowei Xiao

**Abstract**: With the emergence of more powerful large language models (LLMs), such as ChatGPT and GPT-4, in-context learning (ICL) has gained significant prominence in leveraging these models for specific tasks by utilizing data-label pairs as precondition prompts. While incorporating demonstrations can greatly enhance the performance of LLMs across various tasks, it may introduce a new security concern: attackers can manipulate only the demonstrations without changing the input to perform an attack. In this paper, we investigate the security concern of ICL from an adversarial perspective, focusing on the impact of demonstrations. We propose a novel attack method named advICL, which aims to manipulate only the demonstration without changing the input to mislead the models. Our results demonstrate that as the number of demonstrations increases, the robustness of in-context learning would decrease. Additionally, we also identify the intrinsic property of the demonstrations is that they can be used (prepended) with different inputs. As a result, it introduces a more practical threat model in which an attacker can attack the test input example even without knowing and manipulating it. To achieve it, we propose the transferable version of advICL, named Transferable-advICL. Our experiment shows that the adversarial demonstration generated by Transferable-advICL can successfully attack the unseen test input examples. We hope that our study reveals the critical security risks associated with ICL and underscores the need for extensive research on the robustness of ICL, particularly given its increasing significance in the advancement of LLMs.

摘要: 随着更强大的大型语言模型(LLM)的出现，如ChatGPT和GPT-4，情境学习(ICL)通过将数据-标签对作为前提提示来利用这些模型来执行特定任务，从而获得了显著的突出地位。虽然合并演示可以极大地提高LLMS在各种任务中的性能，但它可能会引入一个新的安全问题：攻击者只能操作演示，而不会更改输入来执行攻击。在本文中，我们从对抗的角度研究了ICL的安全问题，重点关注了示威活动的影响。我们提出了一种新的攻击方法AdvICL，其目的是在不改变输入以误导模型的情况下仅操纵演示。我们的结果表明，随着演示数量的增加，情境学习的稳健性会降低。此外，我们还确定了演示的内在属性，即它们可以与不同的输入一起使用(预先设置)。因此，它引入了一个更实用的威胁模型，在该模型中，攻击者即使在不知道和操作测试输入示例的情况下也可以攻击它。为了实现这一点，我们提出了AdvICL的可移植版本，称为Transferable-AdvICL。我们的实验表明，Transferable-AdvICL生成的对抗性演示能够成功地攻击不可见的测试输入示例。我们希望，我们的研究揭示了与ICL相关的关键安全风险，并强调需要对ICL的稳健性进行广泛研究，特别是考虑到它在推进LLMS方面日益重要。



## **45. Jailbreaking Black Box Large Language Models in Twenty Queries**

20个查询中的越狱黑箱大语言模型 cs.LG

21 pages, 10 figures

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2310.08419v2) [paper-pdf](http://arxiv.org/pdf/2310.08419v2)

**Authors**: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong

**Abstract**: There is growing interest in ensuring that large language models (LLMs) align with human values. However, the alignment of such models is vulnerable to adversarial jailbreaks, which coax LLMs into overriding their safety guardrails. The identification of these vulnerabilities is therefore instrumental in understanding inherent weaknesses and preventing future misuse. To this end, we propose Prompt Automatic Iterative Refinement (PAIR), an algorithm that generates semantic jailbreaks with only black-box access to an LLM. PAIR -- which is inspired by social engineering attacks -- uses an attacker LLM to automatically generate jailbreaks for a separate targeted LLM without human intervention. In this way, the attacker LLM iteratively queries the target LLM to update and refine a candidate jailbreak. Empirically, PAIR often requires fewer than twenty queries to produce a jailbreak, which is orders of magnitude more efficient than existing algorithms. PAIR also achieves competitive jailbreaking success rates and transferability on open and closed-source LLMs, including GPT-3.5/4, Vicuna, and PaLM-2.

摘要: 人们对确保大型语言模型(LLM)与人类价值观保持一致的兴趣与日俱增。然而，这类模型的调整很容易受到对抗性越狱的影响，这会诱使低收入国家凌驾于他们的安全护栏之上。因此，确定这些漏洞有助于了解固有的弱点并防止今后的滥用。为此，我们提出了即时自动迭代求精(Pair)，这是一种仅通过黑盒访问LLM来生成语义越狱的算法。Pair受到社会工程攻击的启发，它使用攻击者LLM自动为单独的目标LLM生成越狱，而无需人工干预。通过这种方式，攻击者LLM迭代地查询目标LLM以更新和改进候选越狱。根据经验，Pair通常只需要不到20次查询就可以产生越狱，这比现有算法的效率高出几个数量级。Pair还在开放和封闭源代码的LLM上实现了具有竞争力的越狱成功率和可转移性，包括GPT-3.5/4、维库纳和Palm-2。



## **46. User Inference Attacks on Large Language Models**

针对大型语言模型的用户推理攻击 cs.CR

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2310.09266v1) [paper-pdf](http://arxiv.org/pdf/2310.09266v1)

**Authors**: Nikhil Kandpal, Krishna Pillutla, Alina Oprea, Peter Kairouz, Christopher A. Choquette-Choo, Zheng Xu

**Abstract**: Fine-tuning is a common and effective method for tailoring large language models (LLMs) to specialized tasks and applications. In this paper, we study the privacy implications of fine-tuning LLMs on user data. To this end, we define a realistic threat model, called user inference, wherein an attacker infers whether or not a user's data was used for fine-tuning. We implement attacks for this threat model that require only a small set of samples from a user (possibly different from the samples used for training) and black-box access to the fine-tuned LLM. We find that LLMs are susceptible to user inference attacks across a variety of fine-tuning datasets, at times with near perfect attack success rates. Further, we investigate which properties make users vulnerable to user inference, finding that outlier users (i.e. those with data distributions sufficiently different from other users) and users who contribute large quantities of data are most susceptible to attack. Finally, we explore several heuristics for mitigating privacy attacks. We find that interventions in the training algorithm, such as batch or per-example gradient clipping and early stopping fail to prevent user inference. However, limiting the number of fine-tuning samples from a single user can reduce attack effectiveness, albeit at the cost of reducing the total amount of fine-tuning data.

摘要: 微调是为专门的任务和应用程序定制大型语言模型(LLM)的一种常见且有效的方法。在本文中，我们研究了微调LLMS对用户数据的隐私影响。为此，我们定义了一个现实的威胁模型，称为用户推理，其中攻击者推断用户的数据是否被用于微调。我们对此威胁模型实施攻击，只需要来自用户的一小部分样本(可能不同于用于训练的样本)和对微调的LLM的黑盒访问权限。我们发现，LLM在各种微调数据集上容易受到用户推理攻击，有时攻击成功率近乎完美。此外，我们调查了哪些属性使用户容易受到用户推理的影响，发现离群点用户(即那些数据分布与其他用户有很大差异的用户)和贡献大量数据的用户最容易受到攻击。最后，我们探索了几种减轻隐私攻击的启发式方法。我们发现，训练算法中的干预措施，如批量或逐个样本的梯度裁剪和提前停止，都无法阻止用户推理。然而，限制单个用户的微调样本数量可能会降低攻击效率，尽管代价是减少微调数据的总量。



## **47. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

SmoothLLM：保护大型语言模型免受越狱攻击 cs.LG

**SubmitDate**: 2023-10-13    [abs](http://arxiv.org/abs/2310.03684v2) [paper-pdf](http://arxiv.org/pdf/2310.03684v2)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM.

摘要: 尽管努力使大型语言模型(LLM)与人类价值观保持一致，但GPT、Llama、Claude和Palm等广泛使用的LLM容易受到越狱攻击，即对手欺骗目标LLM生成令人反感的内容。为了解决这一漏洞，我们提出了SmoothLLM，这是第一个旨在缓解对LLM的越狱攻击的算法。基于我们的发现，对抗性生成的提示对字符级别的变化很脆弱，我们的防御首先随机扰动给定输入提示的多个副本，然后聚合相应的预测来检测对抗性输入。SmoothLLM将许多流行的LLM的攻击成功率降低到1个百分点以下，避免了不必要的保守主义，并承认了对攻击缓解的可证明保证。此外，我们的防御使用的查询比现有攻击少得多，并且与任何LLM兼容。



## **48. Composite Backdoor Attacks Against Large Language Models**

针对大型语言模型的复合后门攻击 cs.CR

**SubmitDate**: 2023-10-11    [abs](http://arxiv.org/abs/2310.07676v1) [paper-pdf](http://arxiv.org/pdf/2310.07676v1)

**Authors**: Hai Huang, Zhengyu Zhao, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: Large language models (LLMs) have demonstrated superior performance compared to previous methods on various tasks, and often serve as the foundation models for many researches and services. However, the untrustworthy third-party LLMs may covertly introduce vulnerabilities for downstream tasks. In this paper, we explore the vulnerability of LLMs through the lens of backdoor attacks. Different from existing backdoor attacks against LLMs, ours scatters multiple trigger keys in different prompt components. Such a Composite Backdoor Attack (CBA) is shown to be stealthier than implanting the same multiple trigger keys in only a single component. CBA ensures that the backdoor is activated only when all trigger keys appear. Our experiments demonstrate that CBA is effective in both natural language processing (NLP) and multimodal tasks. For instance, with $3\%$ poisoning samples against the LLaMA-7B model on the Emotion dataset, our attack achieves a $100\%$ Attack Success Rate (ASR) with a False Triggered Rate (FTR) below $2.06\%$ and negligible model accuracy degradation. The unique characteristics of our CBA can be tailored for various practical scenarios, e.g., targeting specific user groups. Our work highlights the necessity of increased security research on the trustworthiness of foundation LLMs.

摘要: 大型语言模型(LLM)在各种任务上表现出了比以前的方法更好的性能，并且经常作为许多研究和服务的基础模型。然而，不可信任的第三方LLM可能会暗中为下游任务引入漏洞。在本文中，我们通过后门攻击的镜头来探索LLMS的脆弱性。与现有的针对LLMS的后门攻击不同，我们的后门攻击将多个触发键分散在不同的提示组件中。这种复合后门攻击(CBA)被证明比在单个组件中植入相同的多个触发键更隐蔽。CBA确保只有当所有触发键都出现时，后门才被激活。实验表明，CBA在自然语言处理(NLP)和多通道任务中都是有效的。例如，在情感数据集上使用$3$中毒样本对骆驼-7B模型进行攻击，我们的攻击获得了$100$攻击成功率(ASR)，而误触发率(FTR)低于$2.06$，而模型精度下降可以忽略不计。我们CBA的独特特点可以根据不同的实际情况进行定制，例如，针对特定的用户群体。我们的工作突出了加强对基金会低成本管理可信性的安全性研究的必要性。



## **49. Fundamental Limitations of Alignment in Large Language Models**

大型语言模型中对齐的基本限制 cs.CL

**SubmitDate**: 2023-10-11    [abs](http://arxiv.org/abs/2304.11082v4) [paper-pdf](http://arxiv.org/pdf/2304.11082v4)

**Authors**: Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that within the limits of this framework, for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates an undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback make the LLM prone to being prompted into the undesired behaviors. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.

摘要: 开发与人类交互的语言模型的一个重要方面是使他们的行为对人类用户有用而无害。这通常是通过调整模型来实现的，这种方式增强了期望的行为，抑制了不期望的行为，这一过程称为对齐。在本文中，我们提出了一种名为行为期望界限(BEB)的理论方法，它允许我们正式地研究大型语言模型中对齐的几个固有特征和限制。重要的是，我们证明了在这个框架的范围内，对于模型所表现出的任何有限概率的行为，存在可以触发模型输出该行为的提示，其概率随着提示的长度的增加而增加。这意味着，任何减弱不受欢迎的行为但不能完全消除它的对准过程，在对抗提示攻击时都是不安全的。此外，我们的框架暗示了一种机制，通过这种机制，领先的对齐方法，如来自人类反馈的强化学习，使得LLM容易被提示进入不希望看到的行为。这一理论结果正在由所谓的当代“聊天GPT越狱”大规模实验证明，在这种情况下，敌对用户通过触发LLM充当恶意角色来欺骗LLM打破其对齐护栏。我们的结果暴露了LLM对齐方面的根本限制，并将设计可靠的机制以确保人工智能安全的必要性放在了首位。



## **50. Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation**

开源LLMS通过利用生成进行灾难性越狱 cs.CL

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06987v1) [paper-pdf](http://arxiv.org/pdf/2310.06987v1)

**Authors**: Yangsibo Huang, Samyak Gupta, Mengzhou Xia, Kai Li, Danqi Chen

**Abstract**: The rapid progress in open-source large language models (LLMs) is significantly advancing AI development. Extensive efforts have been made before model release to align their behavior with human values, with the primary goal of ensuring their helpfulness and harmlessness. However, even carefully aligned models can be manipulated maliciously, leading to unintended behaviors, known as "jailbreaks". These jailbreaks are typically triggered by specific text inputs, often referred to as adversarial prompts. In this work, we propose the generation exploitation attack, an extremely simple approach that disrupts model alignment by only manipulating variations of decoding methods. By exploiting different generation strategies, including varying decoding hyper-parameters and sampling methods, we increase the misalignment rate from 0% to more than 95% across 11 language models including LLaMA2, Vicuna, Falcon, and MPT families, outperforming state-of-the-art attacks with $30\times$ lower computational cost. Finally, we propose an effective alignment method that explores diverse generation strategies, which can reasonably reduce the misalignment rate under our attack. Altogether, our study underscores a major failure in current safety evaluation and alignment procedures for open-source LLMs, strongly advocating for more comprehensive red teaming and better alignment before releasing such models. Our code is available at https://github.com/Princeton-SysML/Jailbreak_LLM.

摘要: 开源大型语言模型(LLM)的快速发展极大地推动了人工智能的发展。在模型发布之前，已经做出了广泛的努力，以使它们的行为符合人类的价值观，主要目标是确保它们的帮助和无害。然而，即使是精心排列的模型也可能被恶意操纵，导致意外行为，即所谓的“越狱”。这些越狱通常由特定的文本输入触发，通常被称为对抗性提示。在这项工作中，我们提出了生成利用攻击，这是一种非常简单的方法，只需操作不同的解码方法就可以破坏模型对齐。通过使用不同的生成策略，包括不同的解码超参数和采样方法，我们将LLaMA2、Vicuna、Falcon和MPT家族等11种语言模型的错配率从0%提高到95%以上，以30倍的计算代价击败了最新的攻击。最后，我们提出了一种有效的匹配方法，该方法探索了不同的生成策略，可以合理地降低攻击下的错配率。总之，我们的研究强调了当前开源LLM安全评估和比对程序的一个重大失败，强烈主张在发布此类模型之前进行更全面的红色团队和更好的比对。我们的代码可以在https://github.com/Princeton-SysML/Jailbreak_LLM.上找到



