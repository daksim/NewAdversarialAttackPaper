# Latest Large Language Model Attack Papers
**update at 2024-04-01 09:30:41**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. PETA: Parameter-Efficient Trojan Attacks**

PETA：参数高效木马攻击 cs.CL

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2310.00648v5) [paper-pdf](http://arxiv.org/pdf/2310.00648v5)

**Authors**: Lauren Hong, Ting Wang

**Abstract**: Parameter-efficient fine-tuning (PEFT) enables efficient adaptation of pre-trained language models (PLMs) to specific tasks. By tuning only a minimal set of (extra) parameters, PEFT achieves performance that is comparable to standard fine-tuning. However, despite its prevalent use, the security implications of PEFT remain largely unexplored. In this paper, we take the initial steps and present PETA, a novel trojan attack that compromises the weights of PLMs by accounting for downstream adaptation through bilevel optimization: the upper-level objective embeds the backdoor into a model while the lower-level objective simulates PEFT to both retain the PLM's task-specific performance and ensure that the backdoor persists after fine-tuning. With extensive evaluation across a variety of downstream tasks and trigger designs, we demonstrate PETA's effectiveness in terms of both attack success rate and clean accuracy, even when the attacker does not have full knowledge of the victim user's training process.

摘要: 参数高效微调(PEFT)使预先训练的语言模型(PLM)能够有效地适应特定任务。通过只调整最小的一组(额外)参数，PEFT实现了与标准微调相当的性能。然而，尽管PEFT被广泛使用，但其安全影响在很大程度上仍未被探索。在本文中，我们采取了初步的步骤，并提出了一种新的木马攻击PETA，它通过双层优化考虑下游适应来折衷PLM的权重：上层目标将后门嵌入到模型中，而下层目标模拟PEFT，既保留了PLM的特定任务性能，又确保了微调后后门的存在。通过对各种下游任务和触发器设计的广泛评估，我们证明了PETA在攻击成功率和干净准确性方面的有效性，即使攻击者并不完全了解受害者用户的培训过程。



## **2. Detoxifying Large Language Models via Knowledge Editing**

基于知识编辑的大型语言模型解化 cs.CL

Ongoing work. Project website:  https://zjunlp.github.io/project/SafeEdit Due to the specificity of the  knowledge editing setting, we revise Tables 1 and 3 to present a fair  comparison of experimental results. More experimental results will be updated  soon

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2403.14472v2) [paper-pdf](http://arxiv.org/pdf/2403.14472v2)

**Authors**: Mengru Wang, Ningyu Zhang, Ziwen Xu, Zekun Xi, Shumin Deng, Yunzhi Yao, Qishen Zhang, Linyi Yang, Jindong Wang, Huajun Chen

**Abstract**: This paper investigates using knowledge editing techniques to detoxify Large Language Models (LLMs). We construct a benchmark, SafeEdit, which covers nine unsafe categories with various powerful attack prompts and equips comprehensive metrics for systematic evaluation. We conduct experiments with several knowledge editing approaches, indicating that knowledge editing has the potential to efficiently detoxify LLMs with limited impact on general performance. Then, we propose a simple yet effective baseline, dubbed Detoxifying with Intraoperative Neural Monitoring (DINM), to diminish the toxicity of LLMs within a few tuning steps via only one instance. We further provide an in-depth analysis of the internal mechanism for various detoxify approaches, demonstrating that previous methods like SFT and DPO may merely suppress the activations of toxic parameters, while DINM mitigates the toxicity of the toxic parameters to a certain extent, making permanent adjustments. We hope that these insights could shed light on future work of developing detoxifying approaches and the underlying knowledge mechanisms of LLMs. Code and benchmark are available at https://github.com/zjunlp/EasyEdit.

摘要: 本文研究了利用知识编辑技术对大型语言模型进行去毒处理。我们构建了一个涵盖9个不安全类别、具有各种强大的攻击提示的基准SafeEdit，并配备了全面的度量来进行系统评估。我们用几种知识编辑方法进行了实验，表明知识编辑有可能在对一般性能影响有限的情况下有效地对LLM进行解毒。然后，我们提出了一个简单而有效的基线，称为术中神经监测解毒(DINM)，仅通过一个实例在几个调整步骤内降低LLMS的毒性。我们进一步深入分析了各种解毒方法的内在机制，证明了以前的方法如SFT和DPO可能只是抑制了毒性参数的激活，而DINM在一定程度上减轻了毒性参数的毒性，做出了永久性的调整。我们希望这些洞察力能够为未来开发戒毒方法的工作和LLMS的潜在知识机制提供帮助。代码和基准测试可在https://github.com/zjunlp/EasyEdit.上获得



## **3. Evolving Assembly Code in an Adversarial Environment**

对抗环境下的汇编代码演变 cs.NE

9 pages, 5 figures, 6 listings

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2403.19489v1) [paper-pdf](http://arxiv.org/pdf/2403.19489v1)

**Authors**: Irina Maliukov, Gera Weiss, Oded Margalit, Achiya Elyasaf

**Abstract**: In this work, we evolve assembly code for the CodeGuru competition. The competition's goal is to create a survivor -- an assembly program that runs the longest in shared memory, by resisting attacks from adversary survivors and finding their weaknesses. For evolving top-notch solvers, we specify a Backus Normal Form (BNF) for the assembly language and synthesize the code from scratch using Genetic Programming (GP). We evaluate the survivors by running CodeGuru games against human-written winning survivors. Our evolved programs found weaknesses in the programs they were trained against and utilized them. In addition, we compare our approach with a Large-Language Model, demonstrating that the latter cannot generate a survivor that can win at any competition. This work has important applications for cyber-security, as we utilize evolution to detect weaknesses in survivors. The assembly BNF is domain-independent; thus, by modifying the fitness function, it can detect code weaknesses and help fix them. Finally, the CodeGuru competition offers a novel platform for analyzing GP and code evolution in adversarial environments. To support further research in this direction, we provide a thorough qualitative analysis of the evolved survivors and the weaknesses found.

摘要: 在这项工作中，我们为CodeGuru竞赛演变汇编代码。这项竞赛的目标是创建一个幸存者--一个在共享内存中运行时间最长的汇编程序，通过抵抗对手幸存者的攻击并找到他们的弱点。对于进化的顶级解算器，我们为汇编语言指定了Backus范式(BNF)，并使用遗传编程(GP)从头开始合成代码。我们通过运行CodeGuru游戏来评估幸存者，以对抗人类编写的获胜幸存者。我们的演进计划发现了他们所针对的计划中的弱点，并利用了这些弱点。此外，我们将我们的方法与大语言模型进行比较，表明后者无法产生能够在任何竞争中获胜的幸存者。这项工作在网络安全方面有重要的应用，因为我们利用进化论来检测幸存者的弱点。程序集BNF是独立于域的；因此，通过修改适应度函数，它可以检测代码弱点并帮助修复它们。最后，CodeGuru竞赛为分析对抗性环境中的GP和代码演化提供了一个新的平台。为了支持这方面的进一步研究，我们对进化的幸存者和发现的弱点进行了彻底的定性分析。



## **4. Data Poisoning for In-context Learning**

基于上下文学习的数据中毒 cs.CR

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2402.02160v2) [paper-pdf](http://arxiv.org/pdf/2402.02160v2)

**Authors**: Pengfei He, Han Xu, Yue Xing, Hui Liu, Makoto Yamada, Jiliang Tang

**Abstract**: In the domain of large language models (LLMs), in-context learning (ICL) has been recognized for its innovative ability to adapt to new tasks, relying on examples rather than retraining or fine-tuning. This paper delves into the critical issue of ICL's susceptibility to data poisoning attacks, an area not yet fully explored. We wonder whether ICL is vulnerable, with adversaries capable of manipulating example data to degrade model performance. To address this, we introduce ICLPoison, a specialized attacking framework conceived to exploit the learning mechanisms of ICL. Our approach uniquely employs discrete text perturbations to strategically influence the hidden states of LLMs during the ICL process. We outline three representative strategies to implement attacks under our framework, each rigorously evaluated across a variety of models and tasks. Our comprehensive tests, including trials on the sophisticated GPT-4 model, demonstrate that ICL's performance is significantly compromised under our framework. These revelations indicate an urgent need for enhanced defense mechanisms to safeguard the integrity and reliability of LLMs in applications relying on in-context learning.

摘要: 在大型语言模型(LLM)领域，情境学习(ICL)因其适应新任务的创新能力而被公认，它依赖于例子而不是再培训或微调。本文深入研究了ICL对数据中毒攻击的易感性这一关键问题，这是一个尚未完全探索的领域。我们想知道ICL是否易受攻击，因为对手能够操纵示例数据来降低模型性能。为了解决这个问题，我们引入了ICLPoison，这是一个专门的攻击框架，旨在利用ICL的学习机制。我们的方法独特地使用离散文本扰动来战略性地影响ICL过程中LLM的隐藏状态。我们概述了在我们的框架下实施攻击的三种具有代表性的战略，每种战略都在各种模型和任务中进行了严格的评估。我们的全面测试，包括对复杂的GPT-4模型的试验，表明ICL的性能在我们的框架下受到了严重影响。这些发现表明，迫切需要增强防御机制，以保障依赖于情景学习的应用程序中LLMS的完整性和可靠性。



## **5. Attacks, Defenses and Evaluations for LLM Conversation Safety: A Survey**

LLM会话安全的攻击、防御与评估：一项调查 cs.CL

Accepted to NAACL 2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2402.09283v3) [paper-pdf](http://arxiv.org/pdf/2402.09283v3)

**Authors**: Zhichen Dong, Zhanhui Zhou, Chao Yang, Jing Shao, Yu Qiao

**Abstract**: Large Language Models (LLMs) are now commonplace in conversation applications. However, their risks of misuse for generating harmful responses have raised serious societal concerns and spurred recent research on LLM conversation safety. Therefore, in this survey, we provide a comprehensive overview of recent studies, covering three critical aspects of LLM conversation safety: attacks, defenses, and evaluations. Our goal is to provide a structured summary that enhances understanding of LLM conversation safety and encourages further investigation into this important subject. For easy reference, we have categorized all the studies mentioned in this survey according to our taxonomy, available at: https://github.com/niconi19/LLM-conversation-safety.

摘要: 大型语言模型（LLM）现在在会话应用中很常见。然而，它们被滥用以产生有害反应的风险引起了严重的社会关注，并刺激了最近对LLM会话安全的研究。因此，在本次调查中，我们提供了一个全面的概述最近的研究，涵盖了LLM会话安全的三个关键方面：攻击，防御和评估。我们的目标是提供一个结构化的摘要，以提高对LLM会话安全的理解，并鼓励进一步调查这一重要主题。为了便于参考，我们根据我们的分类法对本次调查中提到的所有研究进行了分类，可在www.example.com上查阅。



## **6. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

披着羊皮的狼：广义嵌套越狱陷阱可以轻松愚弄大型语言模型 cs.CL

Acccepted by NAACL 2024, 18 pages, 7 figures, 13 tables

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2311.08268v3) [paper-pdf](http://arxiv.org/pdf/2311.08268v3)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, which compromises either generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.

摘要: 大型语言模型(LLM)，如ChatGPT和GPT-4，旨在提供有用和安全的响应。然而，被称为“越狱”的对抗性提示可能会绕过安全措施，导致LLMS生成潜在的有害内容。探索越狱提示可以帮助更好地揭示LLM的弱点，并进一步指导我们确保它们的安全。不幸的是，现有的越狱方法要么需要复杂的人工设计，要么需要对其他白盒模型进行优化，这要么损害了通用性，要么影响了效率。本文将越狱提示攻击概括为两个方面：(1)提示重写和(2)场景嵌套。在此基础上，我们提出了ReNeLLM，这是一个利用LLM自身生成有效越狱提示的自动化框架。广泛的实验表明，与现有的基准相比，ReNeLLM显著提高了攻击成功率，同时大大降低了时间成本。我们的研究也揭示了现有防御方法在保护低密度脂蛋白方面的不足。最后，从即时执行优先级的角度分析了LLMS防御失败的原因，并提出了相应的防御策略。我们希望我们的研究能够促进学术界和低成本管理系统开发商提供更安全和更规范的低成本管理系统。代码可在https://github.com/NJUNLP/ReNeLLM.上获得



## **7. $\textit{LinkPrompt}$: Natural and Universal Adversarial Attacks on Prompt-based Language Models**

$\textit {LinkPrompt}$：基于XSLT语言模型的自然和普遍对抗攻击 cs.CL

Accepted to the main conference of NAACL2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.16432v2) [paper-pdf](http://arxiv.org/pdf/2403.16432v2)

**Authors**: Yue Xu, Wenjie Wang

**Abstract**: Prompt-based learning is a new language model training paradigm that adapts the Pre-trained Language Models (PLMs) to downstream tasks, which revitalizes the performance benchmarks across various natural language processing (NLP) tasks. Instead of using a fixed prompt template to fine-tune the model, some research demonstrates the effectiveness of searching for the prompt via optimization. Such prompt optimization process of prompt-based learning on PLMs also gives insight into generating adversarial prompts to mislead the model, raising concerns about the adversarial vulnerability of this paradigm. Recent studies have shown that universal adversarial triggers (UATs) can be generated to alter not only the predictions of the target PLMs but also the prediction of corresponding Prompt-based Fine-tuning Models (PFMs) under the prompt-based learning paradigm. However, UATs found in previous works are often unreadable tokens or characters and can be easily distinguished from natural texts with adaptive defenses. In this work, we consider the naturalness of the UATs and develop $\textit{LinkPrompt}$, an adversarial attack algorithm to generate UATs by a gradient-based beam search algorithm that not only effectively attacks the target PLMs and PFMs but also maintains the naturalness among the trigger tokens. Extensive results demonstrate the effectiveness of $\textit{LinkPrompt}$, as well as the transferability of UATs generated by $\textit{LinkPrompt}$ to open-sourced Large Language Model (LLM) Llama2 and API-accessed LLM GPT-3.5-turbo.

摘要: 基于提示的学习是一种新的语言模型训练范式，它使预先训练的语言模型(PLM)适应于下游任务，从而重振各种自然语言处理(NLP)任务的表现基准。一些研究证明了通过优化来搜索提示的有效性，而不是使用固定的提示模板来微调模型。这种基于提示的PLM学习的快速优化过程也为生成对抗性提示以误导模型提供了洞察力，这引发了人们对这种范式的对抗性脆弱性的担忧。最近的研究表明，在基于提示的学习范式下，通用对抗触发器(UAT)不仅可以改变目标PLM的预测，还可以改变相应的基于提示的精调模型(PFM)的预测。然而，在以前的著作中发现的UAT通常是不可读的符号或字符，并且可以很容易地与具有自适应防御的自然文本区分开来。在这项工作中，我们考虑了UAT的自然性，并开发了一种对抗性攻击算法，通过基于梯度的波束搜索算法来生成UAT，该算法不仅有效地攻击了目标PLM和PPM，而且保持了触发令牌之间的自然度。广泛的结果证明了$\textit{LinkPrompt}$的有效性，以及由$\textit{LinkPrompt}$生成的UAT可以移植到开源的大型语言模型(LLM)Llama2和API访问的LLm GPT-3.5-Turbo。



## **8. InferDPT: Privacy-Preserving Inference for Black-box Large Language Model**

InferDPT：黑盒大语言模型的隐私保护推理 cs.CR

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2310.12214v6) [paper-pdf](http://arxiv.org/pdf/2310.12214v6)

**Authors**: Meng Tong, Kejiang Chen, Jie Zhang, Yuang Qi, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Zhikun Zhang

**Abstract**: Large language models (LLMs), like ChatGPT, have greatly simplified text generation tasks. However, they have also raised concerns about privacy risks such as data leakage and unauthorized data collection. Existing solutions for privacy-preserving inference face practical challenges related to computation time and communication costs. In this paper, we propose InferDPT, the first practical framework for the privacy-preserving Inference of black-box LLMs, implementing Differential Privacy in Text generation. InferDPT comprises two key modules: the "perturbation module" utilizes the exponential mechanism to generate a perturbed prompt, facilitating privacy-preserving inference with black-box LLMs, and the "extraction module", inspired by knowledge distillation and retrieval-augmented generation, extracts coherent and consistent text from the perturbed generation result, ensuring successful text generation completion. To address privacy concerns related to previous exponential mechanisms' susceptibility to embedding revision attacks, we introduce RANTEXT, a novel differential privacy mechanism integrated into the perturbation module of InferDPT, which introduces the concept of "RANdom adjacency" for TEXT perturbation within the prompt. Experimental results across three datasets demonstrate that the text generation quality of InferDPT is comparable to that of non-private GPT-4, and RANTEXT surpasses existing state-of-the-art mechanisms, namely, SANTEXT+ and CUSTEXT+ in the trade-off between privacy and utility. Even with an privacy parameter epsilon value of 6.0, RANTEXT achieves an average privacy protection rate exceeding 90% against embedding revision attacks, which is 0.58 times higher than that of SANTEXT+ and 3.35 times higher than that of CUSTEXT+.

摘要: 大型语言模型(LLM)，如ChatGPT，极大地简化了文本生成任务。然而，他们也对数据泄露和未经授权的数据收集等隐私风险表示担忧。现有的隐私保护推理解决方案面临着与计算时间和通信成本相关的实际挑战。在本文中，我们提出了第一个实用的黑盒LLMS隐私保护推理框架InferDPT，在文本生成中实现了差分隐私。InferDPT包括两个关键模块：“扰动模块”利用指数机制生成扰动提示，便于使用黑盒LLMS进行隐私保护推理；“提取模块”受知识提炼和检索-增强生成的启发，从扰动生成结果中提取连贯一致的文本，确保文本生成成功完成。针对以往指数机制易受修改攻击的隐私性问题，引入了一种新的差异化隐私机制RANTEXT，该机制集成在InferDPT的扰动模块中，引入了随机邻接的概念来处理提示内的文本扰动。在三个数据集上的实验结果表明，InferDPT的文本生成质量与非私有GPT-4相当，RANTEXT在隐私和效用之间的权衡方面超过了现有的最新机制SanText+和CUSTEXT+。即使在隐私参数epsilon值为6.0的情况下，RANTEXT对嵌入修改攻击的平均隐私保护率也超过90%，比SanText+高0.58倍，比CUSTEXT+高3.35倍。



## **9. Exploring the Deceptive Power of LLM-Generated Fake News: A Study of Real-World Detection Challenges**

探索LLM生成的假新闻的欺骗力量：现实世界的检测挑战研究 cs.CL

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18249v1) [paper-pdf](http://arxiv.org/pdf/2403.18249v1)

**Authors**: Yanshen Sun, Jianfeng He, Limeng Cui, Shuo Lei, Chang-Tien Lu

**Abstract**: Recent advancements in Large Language Models (LLMs) have enabled the creation of fake news, particularly in complex fields like healthcare. Studies highlight the gap in the deceptive power of LLM-generated fake news with and without human assistance, yet the potential of prompting techniques has not been fully explored. Thus, this work aims to determine whether prompting strategies can effectively narrow this gap. Current LLM-based fake news attacks require human intervention for information gathering and often miss details and fail to maintain context consistency. Therefore, to better understand threat tactics, we propose a strong fake news attack method called conditional Variational-autoencoder-Like Prompt (VLPrompt). Unlike current methods, VLPrompt eliminates the need for additional data collection while maintaining contextual coherence and preserving the intricacies of the original text. To propel future research on detecting VLPrompt attacks, we created a new dataset named VLPrompt fake news (VLPFN) containing real and fake texts. Our experiments, including various detection methods and novel human study metrics, were conducted to assess their performance on our dataset, yielding numerous findings.

摘要: 最近大型语言模型(LLM)的进步使假新闻的创造成为可能，特别是在医疗保健等复杂领域。研究突显了在有无人工帮助的情况下，LLM生成的假新闻的欺骗力存在差距，但提示技术的潜力尚未得到充分挖掘。因此，本研究旨在确定激励策略是否能有效缩小这一差距。目前基于LLM的假新闻攻击需要人工干预来收集信息，并且经常错过细节，无法保持上下文一致性。因此，为了更好地理解威胁策略，我们提出了一种强大的假新闻攻击方法，称为条件变分自动编码式提示(VLPrompt)。与目前的方法不同，VLPrompt消除了对额外数据收集的需要，同时保持了上下文的连贯性和原始文本的错综复杂。为了推动未来对VLPrompt攻击检测的研究，我们创建了一个新的数据集，名为VLPrompt假新闻(VLPFN)，包含真实和虚假的文本。我们的实验，包括各种检测方法和新的人体研究指标，被用来评估它们在我们的数据集上的性能，产生了许多发现。



## **10. Tricking LLMs into Disobedience: Formalizing, Analyzing, and Detecting Jailbreaks**

欺骗法学硕士到不服从：形式化，分析和检测越狱 cs.CL

Accepted at LREC-COLING 2024 - The 2024 Joint International  Conference on Computational Linguistics, Language Resources and Evaluation

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2305.14965v4) [paper-pdf](http://arxiv.org/pdf/2305.14965v4)

**Authors**: Abhinav Rao, Sachin Vashistha, Atharva Naik, Somak Aditya, Monojit Choudhury

**Abstract**: Recent explorations with commercial Large Language Models (LLMs) have shown that non-expert users can jailbreak LLMs by simply manipulating their prompts; resulting in degenerate output behavior, privacy and security breaches, offensive outputs, and violations of content regulator policies. Limited studies have been conducted to formalize and analyze these attacks and their mitigations. We bridge this gap by proposing a formalism and a taxonomy of known (and possible) jailbreaks. We survey existing jailbreak methods and their effectiveness on open-source and commercial LLMs (such as GPT-based models, OPT, BLOOM, and FLAN-T5-XXL). We further discuss the challenges of jailbreak detection in terms of their effectiveness against known attacks. For further analysis, we release a dataset of model outputs across 3700 jailbreak prompts over 4 tasks.

摘要: 最近对商业大型语言模型（LLM）的探索表明，非专家用户可以通过简单地操纵他们的提示来越狱LLM；导致退化的输出行为、隐私和安全漏洞、攻击性输出以及违反内容监管政策。已经进行了有限的研究，以正规化和分析这些攻击及其缓解措施。我们通过提出一种形式主义和已知（和可能的）越狱分类来弥合这一差距。我们调查了现有的越狱方法及其在开源和商业LLM上的有效性（如基于GPL的模型、OPT、BLOOM和FLAN—T5—XXL）。我们进一步讨论了越狱检测在对抗已知攻击的有效性方面的挑战。为了进一步分析，我们发布了一个模型输出数据集，该数据集涵盖了4个任务的3700个越狱提示。



## **11. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

基于优化的LLM—as—a—Judge快速注入攻击 cs.CR

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.17710v1) [paper-pdf](http://arxiv.org/pdf/2403.17710v1)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge is a novel solution that can assess textual information with large language models (LLMs). Based on existing research studies, LLMs demonstrate remarkable performance in providing a compelling alternative to traditional human assessment. However, the robustness of these systems against prompt injection attacks remains an open question. In this work, we introduce JudgeDeceiver, a novel optimization-based prompt injection attack tailored to LLM-as-a-Judge. Our method formulates a precise optimization objective for attacking the decision-making process of LLM-as-a-Judge and utilizes an optimization algorithm to efficiently automate the generation of adversarial sequences, achieving targeted and effective manipulation of model evaluations. Compared to handcraft prompt injection attacks, our method demonstrates superior efficacy, posing a significant challenge to the current security paradigms of LLM-based judgment systems. Through extensive experiments, we showcase the capability of JudgeDeceiver in altering decision outcomes across various cases, highlighting the vulnerability of LLM-as-a-Judge systems to the optimization-based prompt injection attack.

摘要: LLM-as-a-Court是一种新的解决方案，它可以使用大型语言模型(LLM)来评估文本信息。基于现有的研究，LLMS在提供一种令人信服的替代传统的人类评估方面表现出显著的性能。然而，这些系统对快速注入攻击的健壮性仍然是一个悬而未决的问题。在这项工作中，我们介绍了一种新的基于优化的快速注入攻击，该攻击是针对LLM-as-a-Court定制的。我们的方法为攻击LLM-as-a-Court的决策过程制定了一个精确的优化目标，并利用优化算法高效地自动生成对抗序列，实现了对模型评估的有针对性和有效的操作。与手工即时注入攻击相比，我们的方法表现出更好的有效性，对基于LLM的判断系统的现有安全范例提出了重大挑战。通过大量的实验，我们展示了JudgeDeceiver在改变不同案件的决策结果方面的能力，突出了LLM-as-a-Court系统对基于优化的即时注入攻击的脆弱性。



## **12. Targeted Visualization of the Backbone of Encoder LLMs**

编码器LLM骨干的目标可视化 cs.LG

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.18872v1) [paper-pdf](http://arxiv.org/pdf/2403.18872v1)

**Authors**: Isaac Roberts, Alexander Schulz, Luca Hermes, Barbara Hammer

**Abstract**: Attention based Large Language Models (LLMs) are the state-of-the-art in natural language processing (NLP). The two most common architectures are encoders such as BERT, and decoders like the GPT models. Despite the success of encoder models, on which we focus in this work, they also bear several risks, including issues with bias or their susceptibility for adversarial attacks, signifying the necessity for explainable AI to detect such issues. While there does exist various local explainability methods focusing on the prediction of single inputs, global methods based on dimensionality reduction for classification inspection, which have emerged in other domains and that go further than just using t-SNE in the embedding space, are not widely spread in NLP.   To reduce this gap, we investigate the application of DeepView, a method for visualizing a part of the decision function together with a data set in two dimensions, to the NLP domain. While in previous work, DeepView has been used to inspect deep image classification models, we demonstrate how to apply it to BERT-based NLP classifiers and investigate its usability in this domain, including settings with adversarially perturbed input samples and pre-trained, fine-tuned, and multi-task models.

摘要: 基于注意力的大语言模型(LLM)是自然语言处理(NLP)领域的前沿技术。两种最常见的架构是编码器(如BERT)和解码器(如GPT模型)。尽管我们在本工作中重点关注的编码器模型取得了成功，但它们也存在几个风险，包括偏见或它们对对抗性攻击的敏感性问题，这意味着有必要使用可解释的人工智能来检测此类问题。虽然有各种局部可解释方法专注于单输入预测，但在其他领域出现的基于降维的全局分类检测方法并没有在NLP中广泛推广，这些方法比仅仅使用嵌入空间中的t-SNE更深入。为了缩小这一差距，我们研究了DeepView在NLP领域的应用，DeepView是一种将决策函数的一部分与二维数据集一起可视化的方法。在以前的工作中，DeepView已经被用来检查深度图像分类模型，我们演示了如何将其应用于基于BERT的NLP分类器，并研究了它在该领域的可用性，包括设置了相反扰动的输入样本和预先训练的、微调的和多任务模型。



## **13. CYGENT: A cybersecurity conversational agent with log summarization powered by GPT-3**

CYGENT：一个网络安全会话代理，具有由GPT—3提供支持的日志摘要 cs.CR

7 pages, 9 figures

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.17160v1) [paper-pdf](http://arxiv.org/pdf/2403.17160v1)

**Authors**: Prasasthy Balasubramanian, Justin Seby, Panos Kostakos

**Abstract**: In response to the escalating cyber-attacks in the modern IT and IoT landscape, we developed CYGENT, a conversational agent framework powered by GPT-3.5 turbo model, designed to aid system administrators in ensuring optimal performance and uninterrupted resource availability. This study focuses on fine-tuning GPT-3 models for cybersecurity tasks, including conversational AI and generative AI tailored specifically for cybersecurity operations. CYGENT assists users by providing cybersecurity information, analyzing and summarizing uploaded log files, detecting specific events, and delivering essential instructions. The conversational agent was developed based on the GPT-3.5 turbo model. We fine-tuned and validated summarizer models (GPT3) using manually generated data points. Using this approach, we achieved a BERTscore of over 97%, indicating GPT-3's enhanced capability in summarizing log files into human-readable formats and providing necessary information to users. Furthermore, we conducted a comparative analysis of GPT-3 models with other Large Language Models (LLMs), including CodeT5-small, CodeT5-base, and CodeT5-base-multi-sum, with the objective of analyzing log analysis techniques. Our analysis consistently demonstrated that Davinci (GPT-3) model outperformed all other LLMs, showcasing higher performance. These findings are crucial for improving human comprehension of logs, particularly in light of the increasing numbers of IoT devices. Additionally, our research suggests that the CodeT5-base-multi-sum model exhibits comparable performance to Davinci to some extent in summarizing logs, indicating its potential as an offline model for this task.

摘要: 为了应对现代IT和物联网环境中不断升级的网络攻击，我们开发了基于GPT-3.5 Turbo模型的会话代理框架CyGENT，旨在帮助系统管理员确保最佳性能和不间断的资源可用性。这项研究的重点是针对网络安全任务微调GPT-3模型，包括专门为网络安全操作量身定做的对话式人工智能和生成式人工智能。CyGENT通过提供网络安全信息、分析和汇总上传的日志文件、检测特定事件和提供基本说明来帮助用户。会话代理是在GPT-3.5涡轮机型的基础上开发的。我们使用手动生成的数据点对汇总器模型(GPT3)进行了微调和验证。使用该方法，我们获得了97%以上的BERT分数，这表明GPT-3的S增强了将日志文件摘要为人类可读格式并向用户提供必要信息的能力。此外，我们还将GPT-3模型与其他大型语言模型(包括CodeT5-Small、CodeT5-BASE和CodeT5-BASE-MULTSUM)进行了比较分析，目的是分析日志分析技术。我们的分析始终表明，Davinci(GPT-3)模型的性能优于所有其他LLM，表现出更高的性能。这些发现对于提高人类对日志的理解至关重要，特别是在物联网设备数量不断增加的情况下。此外，我们的研究表明，CodeT5基多和模型在总结日志方面在一定程度上表现出与Davinci相当的性能，表明其作为这一任务的离线模型的潜力。



## **14. InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents**

InjectAgent：基准测试工具集成大型语言模型代理中的间接提示注入 cs.CL

28 pages, 5 figures, 9 tables

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.02691v2) [paper-pdf](http://arxiv.org/pdf/2403.02691v2)

**Authors**: Qiusi Zhan, Zhixiang Liang, Zifan Ying, Daniel Kang

**Abstract**: Recent work has embodied LLMs as agents, allowing them to access tools, perform actions, and interact with external content (e.g., emails or websites). However, external content introduces the risk of indirect prompt injection (IPI) attacks, where malicious instructions are embedded within the content processed by LLMs, aiming to manipulate these agents into executing detrimental actions against users. Given the potentially severe consequences of such attacks, establishing benchmarks to assess and mitigate these risks is imperative.   In this work, we introduce InjecAgent, a benchmark designed to assess the vulnerability of tool-integrated LLM agents to IPI attacks. InjecAgent comprises 1,054 test cases covering 17 different user tools and 62 attacker tools. We categorize attack intentions into two primary types: direct harm to users and exfiltration of private data. We evaluate 30 different LLM agents and show that agents are vulnerable to IPI attacks, with ReAct-prompted GPT-4 vulnerable to attacks 24% of the time. Further investigation into an enhanced setting, where the attacker instructions are reinforced with a hacking prompt, shows additional increases in success rates, nearly doubling the attack success rate on the ReAct-prompted GPT-4. Our findings raise questions about the widespread deployment of LLM Agents. Our benchmark is available at https://github.com/uiuc-kang-lab/InjecAgent.

摘要: 最近的工作将LLMS体现为代理，允许它们访问工具、执行操作并与外部内容(例如，电子邮件或网站)交互。然而，外部内容会带来间接提示注入(IPI)攻击的风险，在IPI攻击中，恶意指令被嵌入到LLMS处理的内容中，目的是操纵这些代理执行针对用户的有害操作。鉴于此类攻击的潜在严重后果，建立评估和减轻这些风险的基准势在必行。在这项工作中，我们引入了InjecAgent，这是一个旨在评估工具集成的LLM代理对IPI攻击的脆弱性的基准测试。InjecAgent由1,054个测试用例组成，涵盖17个不同的用户工具和62个攻击者工具。我们将攻击意图分为两种主要类型：直接伤害用户和泄露私人数据。我们评估了30种不同的LLM代理，表明代理容易受到IPI攻击，其中反应提示的GPT-4在24%的时间内容易受到攻击。对增强设置的进一步调查显示，成功率进一步提高，反应提示GPT-4的攻击成功率几乎翻了一番。在增强设置中，攻击者的指令通过黑客提示得到加强。我们的发现对LLM特工的广泛部署提出了质疑。我们的基准测试可从https://github.com/uiuc-kang-lab/InjecAgent.获得



## **15. Exploring the Adversarial Capabilities of Large Language Models**

探索大型语言模型的对抗能力 cs.AI

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2402.09132v3) [paper-pdf](http://arxiv.org/pdf/2402.09132v3)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.

摘要: 大型语言模型因其强大的语言生成能力而引起了广泛的关注，为工业和研究提供了巨大的潜力。虽然之前的研究已经深入研究了LLMS的安全和隐私问题，但这些模型在多大程度上可以表现出敌对行为，仍然很大程度上还没有被探索。针对这一差距，我们调查了常见的公开可用的LLM是否具有固有的能力来扰乱文本样本以愚弄安全措施，即所谓的对抗性示例攻击。更具体地说，我们调查LLM是否天生就能够从良性样本中制作敌意示例，以愚弄现有的安全Rail。我们的实验集中在仇恨语音检测上，实验表明，LLMS成功地发现了敌意扰动，有效地破坏了仇恨语音检测系统。我们的发现对依赖LLMS的(半)自治系统具有重大影响，突显了它们与现有系统和安全措施相互作用的潜在挑战。



## **16. Large Language Models for Blockchain Security: A Systematic Literature Review**

区块链安全的大型语言模型：系统文献综述 cs.CR

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2403.14280v2) [paper-pdf](http://arxiv.org/pdf/2403.14280v2)

**Authors**: Zheyuan He, Zihao Li, Sen Yang

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools in various domains involving blockchain security (BS). Several recent studies are exploring LLMs applied to BS. However, there remains a gap in our understanding regarding the full scope of applications, impacts, and potential constraints of LLMs on blockchain security. To fill this gap, we conduct a literature review on LLM4BS.   As the first review of LLM's application on blockchain security, our study aims to comprehensively analyze existing research and elucidate how LLMs contribute to enhancing the security of blockchain systems. Through a thorough examination of scholarly works, we delve into the integration of LLMs into various aspects of blockchain security. We explore the mechanisms through which LLMs can bolster blockchain security, including their applications in smart contract auditing, identity verification, anomaly detection, vulnerable repair, and so on. Furthermore, we critically assess the challenges and limitations associated with leveraging LLMs for blockchain security, considering factors such as scalability, privacy concerns, and adversarial attacks. Our review sheds light on the opportunities and potential risks inherent in this convergence, providing valuable insights for researchers, practitioners, and policymakers alike.

摘要: 大型语言模型(LLM)在涉及区块链安全(BS)的各个领域中已成为强大的工具。最近的几项研究正在探索将LLMS应用于BS。然而，对于低成本管理的全部应用范围、影响以及对区块链安全的潜在限制，我们的理解仍然存在差距。为了填补这一空白，我们对LLM4BS进行了文献综述。作为LLM在区块链安全方面应用的首次综述，本研究旨在全面分析现有研究，阐明LLM如何为增强区块链系统的安全性做出贡献。通过对学术著作的深入研究，我们深入研究了LLMS在区块链安全的各个方面的整合。我们探讨了LLMS增强区块链安全的机制，包括它们在智能合同审计、身份验证、异常检测、漏洞修复等方面的应用。此外，考虑到可扩展性、隐私问题和敌意攻击等因素，我们严格评估了利用LLM实现区块链安全所面临的挑战和限制。我们的审查揭示了这种融合所固有的机遇和潜在风险，为研究人员、从业者和政策制定者提供了有价值的见解。



## **17. LLM Paternity Test: Generated Text Detection with LLM Genetic Inheritance**

LLM亲子鉴定：基于LLM遗传的生成文本检测 cs.CL

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2305.12519v2) [paper-pdf](http://arxiv.org/pdf/2305.12519v2)

**Authors**: Xiao Yu, Yuang Qi, Kejiang Chen, Guoqiang Chen, Xi Yang, Pengyuan Zhu, Weiming Zhang, Nenghai Yu

**Abstract**: Large language models (LLMs) can generate texts that carry the risk of various misuses, including plagiarism, planting fake reviews on e-commerce platforms, or creating inflammatory false tweets. Detecting whether a text is machine-generated has thus become increasingly important. While existing detection methods exhibit superior performance, they often lack generalizability due to their heavy dependence on training data. To alleviate this problem, we propose a model-related generated text detection method, the LLM Paternity Test (LLM-Pat). Specifically, given any candidate text (\textit{child}), LLM-Pat employs an intermediary LLM (\textit{parent}) to reconstruct a \textit{sibling} text corresponding to the given text and then measures the similarity between candidate texts and their sibling texts. High similarity indicates that the candidate text is machine-generated, akin to genetic traits. We have constructed datasets encompassing four scenarios: student responses in educational settings, news creation, academic paper writing, and social media bots to assess the performance of LLM-Pat. The experiments show that LLM-Pat outperforms the existing detection methods and is more robust against paraphrasing attacks and re-translating attacks. Besides, LLM-Pat can also be used to trace which large language model the text was generated by. The constructed dataset and code will be released to benefit the community.

摘要: 大型语言模型(LLM)可以生成带有各种误用风险的文本，包括抄袭、在电子商务平台上种植虚假评论或制造煽动性的虚假推文。因此，检测文本是否是机器生成的变得越来越重要。虽然现有的检测方法表现出了优越的性能，但由于它们对训练数据的严重依赖，往往缺乏泛化能力。为了缓解这一问题，我们提出了一种与模型相关的生成文本检测方法--LLM父子关系测试(LLM-PAT)。具体地说，对于给定的候选文本，LLM-PAT采用一个中间的LLM来重构与给定文本相对应的文本，然后度量候选文本与其兄弟文本之间的相似度。高相似度表明候选文本是机器生成的，类似于遗传特征。我们构建了包含四个场景的数据集：学生在教育环境中的反应、新闻创作、学术论文写作和社交媒体机器人，以评估LLM-PAT的性能。实验表明，LLM-PAT的性能优于现有的检测方法，并且对释义攻击和重译攻击具有更好的鲁棒性。此外，LLM-PAT还可以用于跟踪文本是由哪个大型语言模型生成的。构建的数据集和代码将发布，使社区受益。



## **18. Breaking Down the Defenses: A Comparative Survey of Attacks on Large Language Models**

突破防御：大型语言模型攻击的比较研究 cs.CR

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2403.04786v2) [paper-pdf](http://arxiv.org/pdf/2403.04786v2)

**Authors**: Arijit Ghosh Chowdhury, Md Mofijul Islam, Vaibhav Kumar, Faysal Hossain Shezan, Vaibhav Kumar, Vinija Jain, Aman Chadha

**Abstract**: Large Language Models (LLMs) have become a cornerstone in the field of Natural Language Processing (NLP), offering transformative capabilities in understanding and generating human-like text. However, with their rising prominence, the security and vulnerability aspects of these models have garnered significant attention. This paper presents a comprehensive survey of the various forms of attacks targeting LLMs, discussing the nature and mechanisms of these attacks, their potential impacts, and current defense strategies. We delve into topics such as adversarial attacks that aim to manipulate model outputs, data poisoning that affects model training, and privacy concerns related to training data exploitation. The paper also explores the effectiveness of different attack methodologies, the resilience of LLMs against these attacks, and the implications for model integrity and user trust. By examining the latest research, we provide insights into the current landscape of LLM vulnerabilities and defense mechanisms. Our objective is to offer a nuanced understanding of LLM attacks, foster awareness within the AI community, and inspire robust solutions to mitigate these risks in future developments.

摘要: 大型语言模型(LLM)已经成为自然语言处理(NLP)领域的基石，在理解和生成类似人类的文本方面提供了变革性的能力。然而，随着它们的日益突出，这些模型的安全和漏洞方面已经引起了极大的关注。本文对各种形式的针对LLMS的攻击进行了全面的综述，讨论了这些攻击的性质和机制、它们的潜在影响以及当前的防御策略。我们深入探讨了旨在操纵模型输出的对抗性攻击、影响模型训练的数据中毒以及与训练数据利用相关的隐私问题等主题。文中还探讨了不同攻击方法的有效性，LLMS对这些攻击的恢复能力，以及对模型完整性和用户信任的影响。通过检查最新的研究，我们提供了对LLM漏洞和防御机制的当前情况的见解。我们的目标是提供对LLM攻击的细微差别的理解，培养人工智能社区的意识，并激发强大的解决方案，以减轻未来发展中的这些风险。



## **19. A hybrid LLM workflow can help identify user privilege related variables in programs of any size**

混合LLM工作流可以帮助识别任何规模的程序中的用户权限相关变量 cs.CR

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2403.15723v1) [paper-pdf](http://arxiv.org/pdf/2403.15723v1)

**Authors**: Haizhou Wang, Zhilong Wang, Peng Liu

**Abstract**: Many programs involves operations and logic manipulating user privileges, which is essential for the security of an organization. Therefore, one common malicious goal of attackers is to obtain or escalate the privileges, causing privilege leakage. To protect the program and the organization against privilege leakage attacks, it is important to eliminate the vulnerabilities which can be exploited to achieve such attacks. Unfortunately, while memory vulnerabilities are less challenging to find, logic vulnerabilities are much more imminent, harmful and difficult to identify. Accordingly, many analysts choose to find user privilege related (UPR) variables first as start points to investigate the code where the UPR variables may be used to see if there exists any vulnerabilities, especially the logic ones. In this paper, we introduce a large language model (LLM) workflow that can assist analysts in identifying such UPR variables, which is considered to be a very time-consuming task. Specifically, our tool will audit all the variables in a program and output a UPR score, which is the degree of relationship (closeness) between the variable and user privileges, for each variable. The proposed approach avoids the drawbacks introduced by directly prompting a LLM to find UPR variables by focusing on leverage the LLM at statement level instead of supplying LLM with very long code snippets. Those variables with high UPR scores are essentially potential UPR variables, which should be manually investigated. Our experiments show that using a typical UPR score threshold (i.e., UPR score >0.8), the false positive rate (FPR) is only 13.49%, while UPR variable found is significantly more than that of the heuristic based method.

摘要: 许多程序涉及操纵用户权限的操作和逻辑，这对组织的安全至关重要。因此，攻击者的一个常见恶意目标是获取或提升权限，从而导致权限泄漏。为了保护程序和组织免受权限泄漏攻击，重要的是消除可被利用来实现此类攻击的漏洞。不幸的是，虽然发现内存漏洞的难度较小，但逻辑漏洞更迫在眉睫、危害更大、更难识别。因此，许多分析人员选择首先找到用户权限相关(UPR)变量作为起点，以调查代码中可能使用UPR变量的地方是否存在任何漏洞，特别是逻辑漏洞。在本文中，我们介绍了一个大型语言模型(LLM)工作流，它可以帮助分析师识别这样的UPR变量，这被认为是一项非常耗时的任务。具体地说，我们的工具将审计程序中的所有变量，并为每个变量输出UPR分数，这是变量和用户权限之间的关系(密切程度)。提出的方法避免了直接提示LLM查找UPR变量的缺点，方法是专注于在语句级利用LLM，而不是向LLM提供非常长的代码片段。那些UPR得分较高的变量本质上是潜在的UPR变量，应手动调查。实验表明，使用典型的UPR评分阈值(即UPR评分>0.8)，错误正确率仅为13.49%，而UPR变量的发现明显多于基于启发式的方法。



## **20. LogPrécis: Unleashing Language Models for Automated Malicious Log Analysis**

LogPrécis：释放用于自动恶意日志分析的语言模型 cs.CR

18 pages, Computer&Security  (https://www.sciencedirect.com/science/article/pii/S0167404824001068), code  available at https://github.com/SmartData-Polito/logprecis, models available  at https://huggingface.co/SmartDataPolito

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2307.08309v3) [paper-pdf](http://arxiv.org/pdf/2307.08309v3)

**Authors**: Matteo Boffa, Rodolfo Vieira Valentim, Luca Vassio, Danilo Giordano, Idilio Drago, Marco Mellia, Zied Ben Houidi

**Abstract**: The collection of security-related logs holds the key to understanding attack behaviors and diagnosing vulnerabilities. Still, their analysis remains a daunting challenge. Recently, Language Models (LMs) have demonstrated unmatched potential in understanding natural and programming languages. The question arises whether and how LMs could be also useful for security experts since their logs contain intrinsically confused and obfuscated information. In this paper, we systematically study how to benefit from the state-of-the-art in LM to automatically analyze text-like Unix shell attack logs. We present a thorough design methodology that leads to LogPr\'ecis. It receives as input raw shell sessions and automatically identifies and assigns the attacker tactic to each portion of the session, i.e., unveiling the sequence of the attacker's goals. We demonstrate LogPr\'ecis capability to support the analysis of two large datasets containing about 400,000 unique Unix shell attacks. LogPr\'ecis reduces them into about 3,000 fingerprints, each grouping sessions with the same sequence of tactics. The abstraction it provides lets the analyst better understand attacks, identify fingerprints, detect novelty, link similar attacks, and track families and mutations. Overall, LogPr\'ecis, released as open source, paves the way for better and more responsive defense against cyberattacks.

摘要: 安全相关日志的收集是了解攻击行为和诊断漏洞的关键。尽管如此，他们的分析仍然是一个艰巨的挑战。最近，语言模型(LMS)在理解自然语言和编程语言方面显示出无与伦比的潜力。问题是，LMS是否以及如何也对安全专家有用，因为他们的日志包含本质上令人困惑和混淆的信息。在本文中，我们系统地研究了如何利用LM的最新技术来自动分析类似文本的Unix外壳攻击日志。我们提出了一种通向LogPr‘ECIS的全面设计方法。它接收原始外壳会话作为输入，并自动识别攻击者的战术并将其分配给会话的每个部分，即揭示攻击者的目标序列。我们演示了LogPr的ECIS功能，以支持对包含约400,000个唯一Unix外壳攻击的两个大型数据集的分析。LogPr‘’ECIS将它们简化为大约3,000个指纹，每个指纹使用相同的战术序列对会话进行分组。它提供的抽象使分析师能够更好地了解攻击、识别指纹、检测新颖性、链接类似攻击以及跟踪家族和突变。总体而言，作为开源发布的LogPr‘ECIS为更好、更具响应性的网络攻击防御铺平了道路。



## **21. Inducing High Energy-Latency of Large Vision-Language Models with Verbose Images**

用冗长图像诱导大型视觉语言模型的高能量延迟 cs.CV

Accepted by ICLR 2024

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2401.11170v2) [paper-pdf](http://arxiv.org/pdf/2401.11170v2)

**Authors**: Kuofeng Gao, Yang Bai, Jindong Gu, Shu-Tao Xia, Philip Torr, Zhifeng Li, Wei Liu

**Abstract**: Large vision-language models (VLMs) such as GPT-4 have achieved exceptional performance across various multi-modal tasks. However, the deployment of VLMs necessitates substantial energy consumption and computational resources. Once attackers maliciously induce high energy consumption and latency time (energy-latency cost) during inference of VLMs, it will exhaust computational resources. In this paper, we explore this attack surface about availability of VLMs and aim to induce high energy-latency cost during inference of VLMs. We find that high energy-latency cost during inference of VLMs can be manipulated by maximizing the length of generated sequences. To this end, we propose verbose images, with the goal of crafting an imperceptible perturbation to induce VLMs to generate long sentences during inference. Concretely, we design three loss objectives. First, a loss is proposed to delay the occurrence of end-of-sequence (EOS) token, where EOS token is a signal for VLMs to stop generating further tokens. Moreover, an uncertainty loss and a token diversity loss are proposed to increase the uncertainty over each generated token and the diversity among all tokens of the whole generated sequence, respectively, which can break output dependency at token-level and sequence-level. Furthermore, a temporal weight adjustment algorithm is proposed, which can effectively balance these losses. Extensive experiments demonstrate that our verbose images can increase the length of generated sequences by 7.87 times and 8.56 times compared to original images on MS-COCO and ImageNet datasets, which presents potential challenges for various applications. Our code is available at https://github.com/KuofengGao/Verbose_Images.

摘要: 大型视觉语言模型(VLM)，如GPT-4，已经在各种多模式任务中取得了出色的性能。然而，VLMS的部署需要大量的能源消耗和计算资源。一旦攻击者在VLMS的推理过程中恶意导致高能耗和高延迟时间(能量延迟成本)，就会耗尽计算资源。在本文中，我们探索了关于VLMS可用性的攻击面，目的是在VLMS的推理过程中引入高能量延迟代价。我们发现，可以通过最大化生成序列的长度来控制VLMS推理过程中的高能量延迟代价。为此，我们提出了冗长的图像，目的是在推理过程中制作一种不可察觉的扰动来诱导VLM生成长句子。具体而言，我们设计了三个损失目标。首先，提出了一种延迟序列结束(EOS)令牌发生的损失，其中EOS令牌是VLM停止生成更多令牌的信号。此外，还提出了一种不确定性损失和令牌分集损失，分别增加了每个生成令牌的不确定性和整个生成序列中所有令牌之间的多样性，从而打破了令牌级和序列级的输出相关性。在此基础上，提出了一种时间权值调整算法，可以有效地平衡这些损失。大量实验表明，在MS-Coco和ImageNet数据集上，与原始图像相比，我们的冗长图像可以使生成的序列长度分别增加7.87倍和8.56倍，这给各种应用带来了潜在的挑战。我们的代码可以在https://github.com/KuofengGao/Verbose_Images.上找到



## **22. Self-Guard: Empower the LLM to Safeguard Itself**

Self—Guard：授权LLM保护自己 cs.CL

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2310.15851v2) [paper-pdf](http://arxiv.org/pdf/2310.15851v2)

**Authors**: Zezhong Wang, Fangkai Yang, Lu Wang, Pu Zhao, Hongru Wang, Liang Chen, Qingwei Lin, Kam-Fai Wong

**Abstract**: The jailbreak attack can bypass the safety measures of a Large Language Model (LLM), generating harmful content. This misuse of LLM has led to negative societal consequences. Currently, there are two main approaches to address jailbreak attacks: safety training and safeguards. Safety training focuses on further training LLM to enhance its safety. On the other hand, safeguards involve implementing external models or filters to prevent harmful outputs. However, safety training has constraints in its ability to adapt to new attack types and often leads to a drop in model performance. Safeguards have proven to be of limited help. To tackle these issues, we propose a novel approach called Self-Guard, which combines the strengths of both safety methods. Self-Guard includes two stages. In the first stage, we enhance the model's ability to assess harmful content, and in the second stage, we instruct the model to consistently perform harmful content detection on its own responses. The experiment has demonstrated that Self-Guard is robust against jailbreak attacks. In the bad case analysis, we find that LLM occasionally provides harmless responses to harmful queries. Additionally, we evaluated the general capabilities of the LLM before and after safety training, providing evidence that Self-Guard does not result in the LLM's performance degradation. In sensitivity tests, Self-Guard not only avoids inducing over-sensitivity in LLM but also can even mitigate this issue.

摘要: 越狱攻击可以绕过大型语言模型(LLM)的安全措施，生成有害内容。这种对LLM的滥用已经导致了负面的社会后果。目前，解决越狱攻击的主要方法有两种：安全培训和保障措施。安全培训的重点是对LLM进行进一步培训，以提高其安全性。另一方面，保障措施涉及实施外部模型或过滤器，以防止有害输出。然而，安全培训在适应新攻击类型的能力方面存在限制，往往会导致模型性能下降。事实证明，保障措施的帮助有限。为了解决这些问题，我们提出了一种名为Self-Guard的新方法，它结合了两种安全方法的优点。自我保护包括两个阶段。在第一阶段，我们增强了模型评估有害内容的能力，在第二阶段，我们指示模型对其自身的响应进行一致的有害内容检测。实验证明，Self-Guard对越狱攻击具有很强的抵抗力。在坏案例分析中，我们发现LLM偶尔会对有害查询提供无害的响应。此外，我们在安全培训前后评估了LLM的一般能力，提供了自我保护不会导致LLM性能下降的证据。在敏感性测试中，Self-Guard不仅可以避免在LLM中诱导过度敏感，而且甚至可以缓解这一问题。



## **23. Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation**

闭上眼睛，安全开启：通过图像到文本转换保护多模式LLM cs.CV

Project Page: https://gyhdog99.github.io/projects/ecso/

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.09572v2) [paper-pdf](http://arxiv.org/pdf/2403.09572v2)

**Authors**: Yunhao Gou, Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung, James T. Kwok, Yu Zhang

**Abstract**: Multimodal large language models (MLLMs) have shown impressive reasoning abilities, which, however, are also more vulnerable to jailbreak attacks than their LLM predecessors. Although still capable of detecting unsafe responses, we observe that safety mechanisms of the pre-aligned LLMs in MLLMs can be easily bypassed due to the introduction of image features. To construct robust MLLMs, we propose ECSO(Eyes Closed, Safety On), a novel training-free protecting approach that exploits the inherent safety awareness of MLLMs, and generates safer responses via adaptively transforming unsafe images into texts to activate intrinsic safety mechanism of pre-aligned LLMs in MLLMs. Experiments on five state-of-the-art (SoTA) MLLMs demonstrate that our ECSO enhances model safety significantly (e.g., a 37.6% improvement on the MM-SafetyBench (SD+OCR), and 71.3% on VLSafe for the LLaVA-1.5-7B), while consistently maintaining utility results on common MLLM benchmarks. Furthermore, we show that ECSO can be used as a data engine to generate supervised-finetuning (SFT) data for MLLM alignment without extra human intervention.

摘要: 多通道大语言模型(MLLM)已经显示出令人印象深刻的推理能力，然而，它们也比它们的前身更容易受到越狱攻击。虽然仍然能够检测到不安全的响应，但我们观察到，由于引入了图像特征，MLLMS中预先对准的LLM的安全机制可以很容易地绕过。为了构造稳健的MLLMS，我们提出了一种新的无需训练的保护方法ECSO(Eyes Closed，Safe On)，该方法利用MLLMS固有的安全意识，通过自适应地将不安全的图像转换为文本来激活MLLMS中预对准的LLMS的本质安全机制，从而产生更安全的响应。在五个最先进的(SOTA)MLLM上的实验表明，我们的ECSO显著增强了模型安全性(例如，对于LLaVA-1.5-7B，MM-SafetyBch(SD+OCR)的性能提高了37.6%，VLSafe的性能提高了71.3%)，同时保持了常见MLLM基准的实用结果。此外，我们还证明了ECSO可以作为数据引擎来生成用于MLLM比对的监督精调(SFT)数据，而无需额外的人工干预。



## **24. Risk and Response in Large Language Models: Evaluating Key Threat Categories**

大型语言模型中的风险和响应：评估关键威胁类别 cs.CL

19 pages, 14 figures

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.14988v1) [paper-pdf](http://arxiv.org/pdf/2403.14988v1)

**Authors**: Bahareh Harandizadeh, Abel Salinas, Fred Morstatter

**Abstract**: This paper explores the pressing issue of risk assessment in Large Language Models (LLMs) as they become increasingly prevalent in various applications. Focusing on how reward models, which are designed to fine-tune pretrained LLMs to align with human values, perceive and categorize different types of risks, we delve into the challenges posed by the subjective nature of preference-based training data. By utilizing the Anthropic Red-team dataset, we analyze major risk categories, including Information Hazards, Malicious Uses, and Discrimination/Hateful content. Our findings indicate that LLMs tend to consider Information Hazards less harmful, a finding confirmed by a specially developed regression model. Additionally, our analysis shows that LLMs respond less stringently to Information Hazards compared to other risks. The study further reveals a significant vulnerability of LLMs to jailbreaking attacks in Information Hazard scenarios, highlighting a critical security concern in LLM risk assessment and emphasizing the need for improved AI safety measures.

摘要: 本文探讨了大型语言模型(LLMS)中风险评估的紧迫问题，因为它们在各种应用中日益普遍。我们聚焦于奖励模型，这些模型旨在微调预先训练的LLM以与人类价值观保持一致，如何感知和分类不同类型的风险，深入探讨基于偏好的训练数据的主观性质带来的挑战。通过利用人类红队数据集，我们分析了主要的风险类别，包括信息危害、恶意使用和歧视/仇恨内容。我们的发现表明，LLM倾向于认为信息风险的危害性较小，这一发现得到了专门开发的回归模型的证实。此外，我们的分析表明，与其他风险相比，低成本管理对信息风险的反应不那么严格。这项研究进一步揭示了LLMS在信息危险情景下对越狱攻击的严重脆弱性，突显了LLM风险评估中的一个关键安全问题，并强调了改进人工智能安全措施的必要性。



## **25. Privacy-Preserving End-to-End Spoken Language Understanding**

隐私保护的端到端口语理解 cs.CR

Accepted by IJCAI

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.15510v1) [paper-pdf](http://arxiv.org/pdf/2403.15510v1)

**Authors**: Yinggui Wang, Wei Huang, Le Yang

**Abstract**: Spoken language understanding (SLU), one of the key enabling technologies for human-computer interaction in IoT devices, provides an easy-to-use user interface. Human speech can contain a lot of user-sensitive information, such as gender, identity, and sensitive content. New types of security and privacy breaches have thus emerged. Users do not want to expose their personal sensitive information to malicious attacks by untrusted third parties. Thus, the SLU system needs to ensure that a potential malicious attacker cannot deduce the sensitive attributes of the users, while it should avoid greatly compromising the SLU accuracy. To address the above challenge, this paper proposes a novel SLU multi-task privacy-preserving model to prevent both the speech recognition (ASR) and identity recognition (IR) attacks. The model uses the hidden layer separation technique so that SLU information is distributed only in a specific portion of the hidden layer, and the other two types of information are removed to obtain a privacy-secure hidden layer. In order to achieve good balance between efficiency and privacy, we introduce a new mechanism of model pre-training, namely joint adversarial training, to further enhance the user privacy. Experiments over two SLU datasets show that the proposed method can reduce the accuracy of both the ASR and IR attacks close to that of a random guess, while leaving the SLU performance largely unaffected.

摘要: 口语理解(SLU)是物联网设备中实现人机交互的关键技术之一，它提供了易于使用的用户界面。人类语音可能包含大量用户敏感信息，如性别、身份和敏感内容。因此，出现了新类型的安全和隐私违规行为。用户不想将他们的个人敏感信息暴露在不受信任的第三方的恶意攻击下。因此，SLU系统需要确保潜在的恶意攻击者不能推断用户的敏感属性，同时应该避免极大地损害SLU的准确性。针对上述挑战，提出了一种新的SLU多任务隐私保护模型，以同时防止语音识别(ASR)和身份识别(IR)攻击。该模型使用隐层分离技术，使得SLU信息只分布在隐层的特定部分，而其他两种类型的信息被移除以获得隐私安全的隐层。为了在效率和隐私之间取得良好的平衡，我们引入了一种新的模型预训练机制，即联合对抗性训练，以进一步增强用户的隐私。在两个SLU数据集上的实验表明，该方法可以将ASR和IR攻击的准确率降低到接近随机猜测的水平，而SLU的性能基本不受影响。



## **26. BadCLIP: Trigger-Aware Prompt Learning for Backdoor Attacks on CLIP**

BadCLIP：触发感知提示学习，以应对CLIP上的后门攻击 cs.CV

14 pages, 6 figures

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2311.16194v2) [paper-pdf](http://arxiv.org/pdf/2311.16194v2)

**Authors**: Jiawang Bai, Kuofeng Gao, Shaobo Min, Shu-Tao Xia, Zhifeng Li, Wei Liu

**Abstract**: Contrastive Vision-Language Pre-training, known as CLIP, has shown promising effectiveness in addressing downstream image recognition tasks. However, recent works revealed that the CLIP model can be implanted with a downstream-oriented backdoor. On downstream tasks, one victim model performs well on clean samples but predicts a specific target class whenever a specific trigger is present. For injecting a backdoor, existing attacks depend on a large amount of additional data to maliciously fine-tune the entire pre-trained CLIP model, which makes them inapplicable to data-limited scenarios. In this work, motivated by the recent success of learnable prompts, we address this problem by injecting a backdoor into the CLIP model in the prompt learning stage. Our method named BadCLIP is built on a novel and effective mechanism in backdoor attacks on CLIP, i.e., influencing both the image and text encoders with the trigger. It consists of a learnable trigger applied to images and a trigger-aware context generator, such that the trigger can change text features via trigger-aware prompts, resulting in a powerful and generalizable attack. Extensive experiments conducted on 11 datasets verify that the clean accuracy of BadCLIP is similar to those of advanced prompt learning methods and the attack success rate is higher than 99% in most cases. BadCLIP is also generalizable to unseen classes, and shows a strong generalization capability under cross-dataset and cross-domain settings.

摘要: 对比视觉语言预训练，也称为CLIP，在解决下游图像识别任务方面显示出了良好的效果。然而，最近的研究表明，夹子模型可以植入一个面向下游的后门。在下游任务上，一个受害者模型在干净的样本上执行得很好，但只要出现特定的触发器，就会预测特定的目标类。对于注入后门，现有的攻击依赖于大量的额外数据来恶意微调整个预先训练的剪辑模型，这使得它们不适用于数据有限的场景。在这项工作中，受最近可学习提示的成功的启发，我们通过在快速学习阶段向CLIP模型注入后门来解决这个问题。我们的方法BadCLIP是建立在对CLIP的后门攻击中的一种新颖而有效的机制上的，即通过触发器同时影响图像和文本编码器。它由应用于图像的可学习触发器和触发器感知上下文生成器组成，使得触发器可以通过触发器感知提示改变文本特征，从而产生强大且可泛化的攻击。在11个数据集上进行的大量实验证明，BadCLIP的清洁准确率与先进的快速学习方法相似，在大多数情况下攻击成功率高于99%。BadCLIP还可以泛化到看不见的类，在跨数据集和跨域设置下表现出很强的泛化能力。



## **27. Unveiling Typographic Deceptions: Insights of the Typographic Vulnerability in Large Vision-Language Model**

揭开印刷欺骗：大型视觉语言模型中印刷漏洞的透视 cs.CV

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2402.19150v2) [paper-pdf](http://arxiv.org/pdf/2402.19150v2)

**Authors**: Hao Cheng, Erjia Xiao, Jindong Gu, Le Yang, Jinhao Duan, Jize Zhang, Jiahang Cao, Kaidi Xu, Renjing Xu

**Abstract**: Large Vision-Language Models (LVLMs) rely on vision encoders and Large Language Models (LLMs) to exhibit remarkable capabilities on various multi-modal tasks in the joint space of vision and language. However, the Typographic Attack, which disrupts vision-language models (VLMs) such as Contrastive Language-Image Pretraining (CLIP), has also been expected to be a security threat to LVLMs. Firstly, we verify typographic attacks on current well-known commercial and open-source LVLMs and uncover the widespread existence of this threat. Secondly, to better assess this vulnerability, we propose the most comprehensive and largest-scale Typographic Dataset to date. The Typographic Dataset not only considers the evaluation of typographic attacks under various multi-modal tasks but also evaluates the effects of typographic attacks, influenced by texts generated with diverse factors. Based on the evaluation results, we investigate the causes why typographic attacks may impact VLMs and LVLMs, leading to three highly insightful discoveries. By the examination of our discoveries and experimental validation in the Typographic Dataset, we reduce the performance degradation from $42.07\%$ to $13.90\%$ when LVLMs confront typographic attacks.

摘要: 大视觉-语言模型依赖于视觉编码器和大语言模型，在视觉和语言的联合空间中表现出对各种多通道任务的卓越能力。然而，打乱视觉语言模型(如对比语言图像预训练(CLIP))的排版攻击也被认为是对视觉语言模型的安全威胁。首先，我们验证了对当前著名的商业和开源LVLM的排版攻击，并揭示了这种威胁的广泛存在。其次，为了更好地评估这个漏洞，我们提出了迄今为止最全面和最大规模的排版数据集。排版数据集不仅考虑了各种多模式任务下排版攻击的评估，而且还评估了受多种因素生成的文本影响的排版攻击的效果。基于评估结果，我们调查了排版攻击可能影响VLM和LVLM的原因，导致了三个非常有洞察力的发现。通过检验我们的发现和在排版数据集中的实验验证，我们将当LVLMS遇到排版攻击时的性能下降从42.07美元减少到13.90美元。



## **28. $\nabla τ$: Gradient-based and Task-Agnostic machine Unlearning**

$\nabla τ $：基于任务和任务不可知的机器学习 cs.LG

14 pages, 2 figures

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14339v1) [paper-pdf](http://arxiv.org/pdf/2403.14339v1)

**Authors**: Daniel Trippa, Cesare Campagnano, Maria Sofia Bucarelli, Gabriele Tolomei, Fabrizio Silvestri

**Abstract**: Machine Unlearning, the process of selectively eliminating the influence of certain data examples used during a model's training, has gained significant attention as a means for practitioners to comply with recent data protection regulations. However, existing unlearning methods face critical drawbacks, including their prohibitively high cost, often associated with a large number of hyperparameters, and the limitation of forgetting only relatively small data portions. This often makes retraining the model from scratch a quicker and more effective solution. In this study, we introduce Gradient-based and Task-Agnostic machine Unlearning ($\nabla \tau$), an optimization framework designed to remove the influence of a subset of training data efficiently. It applies adaptive gradient ascent to the data to be forgotten while using standard gradient descent for the remaining data. $\nabla \tau$ offers multiple benefits over existing approaches. It enables the unlearning of large sections of the training dataset (up to 30%). It is versatile, supporting various unlearning tasks (such as subset forgetting or class removal) and applicable across different domains (images, text, etc.). Importantly, $\nabla \tau$ requires no hyperparameter adjustments, making it a more appealing option than retraining the model from scratch. We evaluate our framework's effectiveness using a set of well-established Membership Inference Attack metrics, demonstrating up to 10% enhancements in performance compared to state-of-the-art methods without compromising the original model's accuracy.

摘要: 机器遗忘是有选择地消除模型训练期间使用的某些数据示例的影响的过程，作为从业者遵守最新数据保护法规的一种手段，已受到极大关注。然而，现有的遗忘方法面临着严重的缺陷，包括成本高得令人望而却步，往往与大量的超参数相关，以及仅忘记相对较小的数据部分的限制。这通常会使从头开始重新培训模型成为更快、更有效的解决方案。在这项研究中，我们介绍了基于梯度和任务无关的机器遗忘($\nabla\tau$)，这是一个优化框架，旨在有效地消除训练数据子集的影响。它对要遗忘的数据应用自适应梯度上升，而对剩余数据使用标准梯度下降。与现有方法相比，$\nabla\tau$具有多种优势。它允许忘记训练数据集的大段(高达30%)。它是通用的，支持各种遗忘任务(如子集遗忘或类移除)，并适用于不同的域(图像、文本等)。重要的是，$\nabla\tau$不需要调整超参数，这使它成为比从头开始重新训练模型更有吸引力的选择。我们使用一组完善的成员关系推理攻击度量来评估我们框架的有效性，在不影响原始模型准确性的情况下，与最先进的方法相比，性能最多提高了10%。



## **29. FMM-Attack: A Flow-based Multi-modal Adversarial Attack on Video-based LLMs**

FMM—Attack：一种基于流的多模式对抗性视频LLM攻击 cs.CV

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.13507v2) [paper-pdf](http://arxiv.org/pdf/2403.13507v2)

**Authors**: Jinmin Li, Kuofeng Gao, Yang Bai, Jingyun Zhang, Shu-tao Xia, Yisen Wang

**Abstract**: Despite the remarkable performance of video-based large language models (LLMs), their adversarial threat remains unexplored. To fill this gap, we propose the first adversarial attack tailored for video-based LLMs by crafting flow-based multi-modal adversarial perturbations on a small fraction of frames within a video, dubbed FMM-Attack. Extensive experiments show that our attack can effectively induce video-based LLMs to generate incorrect answers when videos are added with imperceptible adversarial perturbations. Intriguingly, our FMM-Attack can also induce garbling in the model output, prompting video-based LLMs to hallucinate. Overall, our observations inspire a further understanding of multi-modal robustness and safety-related feature alignment across different modalities, which is of great importance for various large multi-modal models. Our code is available at https://github.com/THU-Kingmin/FMM-Attack.

摘要: 尽管基于视频的大型语言模型（LLM）表现出色，但它们的对抗性威胁仍未得到探索。为了填补这一空白，我们提出了第一个针对基于视频的LLM的对抗攻击，通过在视频中的一小部分帧上制作基于流的多模式对抗干扰，称为FMM攻击。大量的实验表明，我们的攻击可以有效地诱导基于视频的LLM生成错误的答案时，视频中添加了不可感知的对抗干扰。有趣的是，我们的FM—Attack还可以在模型输出中引起混乱，促使基于视频的LLM产生幻觉。总的来说，我们的观察结果激发了人们对不同模态的多模态鲁棒性和安全相关特性对齐的进一步理解，这对各种大型多模态模型非常重要。我们的代码可在www.example.com获得。



## **30. AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models**

AutoDAN：在对齐的大型语言模型上生成隐蔽的越狱脚本 cs.CL

Published as a conference paper at ICLR 2024. Code is available at  https://github.com/SheltonLiu-N/AutoDAN

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2310.04451v2) [paper-pdf](http://arxiv.org/pdf/2310.04451v2)

**Authors**: Xiaogeng Liu, Nan Xu, Muhao Chen, Chaowei Xiao

**Abstract**: The aligned Large Language Models (LLMs) are powerful language understanding and decision-making tools that are created through extensive alignment with human feedback. However, these large models remain susceptible to jailbreak attacks, where adversaries manipulate prompts to elicit malicious outputs that should not be given by aligned LLMs. Investigating jailbreak prompts can lead us to delve into the limitations of LLMs and further guide us to secure them. Unfortunately, existing jailbreak techniques suffer from either (1) scalability issues, where attacks heavily rely on manual crafting of prompts, or (2) stealthiness problems, as attacks depend on token-based algorithms to generate prompts that are often semantically meaningless, making them susceptible to detection through basic perplexity testing. In light of these challenges, we intend to answer this question: Can we develop an approach that can automatically generate stealthy jailbreak prompts? In this paper, we introduce AutoDAN, a novel jailbreak attack against aligned LLMs. AutoDAN can automatically generate stealthy jailbreak prompts by the carefully designed hierarchical genetic algorithm. Extensive evaluations demonstrate that AutoDAN not only automates the process while preserving semantic meaningfulness, but also demonstrates superior attack strength in cross-model transferability, and cross-sample universality compared with the baseline. Moreover, we also compare AutoDAN with perplexity-based defense methods and show that AutoDAN can bypass them effectively.

摘要: 对齐的大型语言模型(LLM)是强大的语言理解和决策工具，通过与人类反馈的广泛对齐而创建。然而，这些大型模型仍然容易受到越狱攻击，在越狱攻击中，对手操纵提示来获得不应由对齐的LLM提供的恶意输出。调查越狱提示可以引导我们深入研究LLMS的局限性，并进一步指导我们确保它们的安全。不幸的是，现有的越狱技术存在以下两个问题：(1)可扩展性问题，攻击严重依赖手工编写提示；(2)隐蔽性问题，因为攻击依赖基于令牌的算法来生成通常在语义上没有意义的提示，这使得它们很容易通过基本的困惑测试被检测到。鉴于这些挑战，我们打算回答这个问题：我们能否开发出一种能够自动生成秘密越狱提示的方法？在本文中，我们介绍了AutoDAN，一种新的针对对齐LLM的越狱攻击。AutoDAN可以通过精心设计的分层遗传算法自动生成隐形越狱提示。广泛的评估表明，AutoDAN不仅在保持语义意义的同时实现了过程的自动化，而且与基线相比，在跨模型可转移性和跨样本通用性方面表现出了优越的攻击能力。此外，我们还对AutoDAN和基于困惑的防御方法进行了比较，结果表明AutoDAN可以有效地绕过它们。



## **31. A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly**

大语言模型（LLM）安全性和隐私性调查：好、坏和丑 cs.CR

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2312.02003v3) [paper-pdf](http://arxiv.org/pdf/2312.02003v3)

**Authors**: Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Zhibo Sun, Yue Zhang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and Bard, have revolutionized natural language understanding and generation. They possess deep language comprehension, human-like text generation capabilities, contextual awareness, and robust problem-solving skills, making them invaluable in various domains (e.g., search engines, customer support, translation). In the meantime, LLMs have also gained traction in the security community, revealing security vulnerabilities and showcasing their potential in security-related tasks. This paper explores the intersection of LLMs with security and privacy. Specifically, we investigate how LLMs positively impact security and privacy, potential risks and threats associated with their use, and inherent vulnerabilities within LLMs. Through a comprehensive literature review, the paper categorizes the papers into "The Good" (beneficial LLM applications), "The Bad" (offensive applications), and "The Ugly" (vulnerabilities of LLMs and their defenses). We have some interesting findings. For example, LLMs have proven to enhance code security (code vulnerability detection) and data privacy (data confidentiality protection), outperforming traditional methods. However, they can also be harnessed for various attacks (particularly user-level attacks) due to their human-like reasoning abilities. We have identified areas that require further research efforts. For example, Research on model and parameter extraction attacks is limited and often theoretical, hindered by LLM parameter scale and confidentiality. Safe instruction tuning, a recent development, requires more exploration. We hope that our work can shed light on the LLMs' potential to both bolster and jeopardize cybersecurity.

摘要: 大型语言模型(LLM)，如ChatGPT和BARD，彻底改变了自然语言的理解和生成。它们具有深刻的语言理解能力、类似人类的文本生成能力、上下文意识和强大的解决问题的技能，使它们在各个领域(例如，搜索引擎、客户支持、翻译)具有无价的价值。与此同时，LLMS也在安全界获得了吸引力，揭示了安全漏洞，并在与安全相关的任务中展示了它们的潜力。本文探讨了LLMS与安全和隐私的交集。具体地说，我们调查了LLM如何对安全和隐私产生积极影响，与其使用相关的潜在风险和威胁，以及LLM中的固有漏洞。通过全面的文献综述，本文将这些论文分为“好的”(有益的LLM应用程序)、“坏的”(攻击性应用程序)和“丑陋的”(LLM的漏洞及其防御)。我们有一些有趣的发现。例如，LLM已被证明可以增强代码安全性(代码漏洞检测)和数据隐私(数据机密性保护)，性能优于传统方法。然而，由于它们类似人类的推理能力，它们也可以被利用来进行各种攻击(特别是用户级的攻击)。我们已经确定了需要进一步研究的领域。例如，对模型和参数提取攻击的研究是有限的，而且往往是理论上的，受到LLM参数规模和保密性的阻碍。安全的指令调优是一个新的发展，需要更多的探索。我们希望我们的工作能够揭示小岛屿发展中国家加强和危害网络安全的潜力。



## **32. Defending Against Indirect Prompt Injection Attacks With Spotlighting**

利用聚光灯防御间接即时注入攻击 cs.CR

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.14720v1) [paper-pdf](http://arxiv.org/pdf/2403.14720v1)

**Authors**: Keegan Hines, Gary Lopez, Matthew Hall, Federico Zarfati, Yonatan Zunger, Emre Kiciman

**Abstract**: Large Language Models (LLMs), while powerful, are built and trained to process a single text input. In common applications, multiple inputs can be processed by concatenating them together into a single stream of text. However, the LLM is unable to distinguish which sections of prompt belong to various input sources. Indirect prompt injection attacks take advantage of this vulnerability by embedding adversarial instructions into untrusted data being processed alongside user commands. Often, the LLM will mistake the adversarial instructions as user commands to be followed, creating a security vulnerability in the larger system. We introduce spotlighting, a family of prompt engineering techniques that can be used to improve LLMs' ability to distinguish among multiple sources of input. The key insight is to utilize transformations of an input to provide a reliable and continuous signal of its provenance. We evaluate spotlighting as a defense against indirect prompt injection attacks, and find that it is a robust defense that has minimal detrimental impact to underlying NLP tasks. Using GPT-family models, we find that spotlighting reduces the attack success rate from greater than {50}\% to below {2}\% in our experiments with minimal impact on task efficacy.

摘要: 大型语言模型(LLM)虽然功能强大，但其构建和训练都是为了处理单个文本输入。在常见的应用程序中，可以通过将多个输入连接成单个文本流来处理它们。然而，LLM无法区分提示的哪些部分属于不同的输入源。间接提示注入攻击通过在与用户命令一起处理的不可信数据中嵌入敌意指令来利用此漏洞。通常，LLM会将敌意指令误认为需要遵循的用户命令，从而在更大的系统中造成安全漏洞。我们介绍了聚光灯，这是一系列快速工程技术，可用于提高LLMS区分多个输入源的能力。关键的洞察力是利用输入的转换来提供关于其来源的可靠和连续的信号。我们评估聚光灯作为对间接即时注入攻击的防御，并发现它是一种健壮的防御，对底层NLP任务的有害影响最小。使用GPT家族模型，我们发现在我们的实验中，聚光灯将攻击成功率从大于{50}\%降低到低于{2}\%，而对任务效能的影响最小。



## **33. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

AttackEval：如何评估大型语言模型越狱攻击的有效性 cs.CL

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2401.09002v3) [paper-pdf](http://arxiv.org/pdf/2401.09002v3)

**Authors**: Dong shu, Mingyu Jin, Suiyuan Zhu, Beichen Wang, Zihao Zhou, Chong Zhang, Yongfeng Zhang

**Abstract**: In our research, we pioneer a novel approach to evaluate the effectiveness of jailbreak attacks on Large Language Models (LLMs), such as GPT-4 and LLaMa2, diverging from traditional robustness-focused binary evaluations. Our study introduces two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework, using a scoring range from 0 to 1, offers a unique perspective, enabling a more comprehensive and nuanced evaluation of attack effectiveness and empowering attackers to refine their attack prompts with greater understanding. Furthermore, we have developed a comprehensive ground truth dataset specifically tailored for jailbreak tasks. This dataset not only serves as a crucial benchmark for our current study but also establishes a foundational resource for future research, enabling consistent and comparative analyses in this evolving field. Upon meticulous comparison with traditional evaluation methods, we discovered that our evaluation aligns with the baseline's trend while offering a more profound and detailed assessment. We believe that by accurately evaluating the effectiveness of attack prompts in the Jailbreak task, our work lays a solid foundation for assessing a wider array of similar or even more complex tasks in the realm of prompt injection, potentially revolutionizing this field.

摘要: 在我们的研究中，我们开创了一种新的方法来评估越狱攻击对大型语言模型(如GPT-4和LLaMa2)的有效性，不同于传统的专注于健壮性的二进制评估。我们的研究引入了两个不同的评估框架：粗粒度评估和细粒度评估。每个框架使用从0到1的评分范围，提供了一个独特的视角，能够对攻击效果进行更全面和细微的评估，并使攻击者能够更好地了解他们的攻击提示。此外，我们还开发了专门为越狱任务量身定做的全面地面事实数据集。这一数据集不仅是我们当前研究的重要基准，而且还为未来的研究奠定了基础资源，使这一不断发展的领域能够进行一致和比较的分析。通过与传统评估方法的细致比较，我们发现我们的评估符合基线的趋势，同时提供了更深入和详细的评估。我们相信，通过准确评估越狱任务中攻击提示的有效性，我们的工作为评估快速注射领域中更广泛的类似甚至更复杂的任务奠定了坚实的基础，这可能会给这一领域带来革命性的变化。



## **34. BadEdit: Backdooring large language models by model editing**

BadEdit：通过模型编辑后台处理大型语言模型 cs.CR

ICLR 2024

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13355v1) [paper-pdf](http://arxiv.org/pdf/2403.13355v1)

**Authors**: Yanzhou Li, Tianlin Li, Kangjie Chen, Jian Zhang, Shangqing Liu, Wenhan Wang, Tianwei Zhang, Yang Liu

**Abstract**: Mainstream backdoor attack methods typically demand substantial tuning data for poisoning, limiting their practicality and potentially degrading the overall performance when applied to Large Language Models (LLMs). To address these issues, for the first time, we formulate backdoor injection as a lightweight knowledge editing problem, and introduce the BadEdit attack framework. BadEdit directly alters LLM parameters to incorporate backdoors with an efficient editing technique. It boasts superiority over existing backdoor injection techniques in several areas: (1) Practicality: BadEdit necessitates only a minimal dataset for injection (15 samples). (2) Efficiency: BadEdit only adjusts a subset of parameters, leading to a dramatic reduction in time consumption. (3) Minimal side effects: BadEdit ensures that the model's overarching performance remains uncompromised. (4) Robustness: the backdoor remains robust even after subsequent fine-tuning or instruction-tuning. Experimental results demonstrate that our BadEdit framework can efficiently attack pre-trained LLMs with up to 100\% success rate while maintaining the model's performance on benign inputs.

摘要: 主流后门攻击方法通常需要大量调整数据以进行中毒，这限制了它们的实用性，并可能在应用于大型语言模型(LLM)时降低整体性能。为了解决这些问题，我们首次将后门注入描述为一个轻量级的知识编辑问题，并引入了BadEdit攻击框架。BadEDIT直接更改LLM参数，将后门与高效的编辑技术结合在一起。它在几个方面优于现有的后门注入技术：(1)实用性：BadEdit只需要一个最小的注入数据集(15个样本)。(2)效率：BadEDIT只调整部分参数，大大减少了时间消耗。(3)副作用最小：BadEdit可确保模型的总体性能不受影响。(4)健壮性：即使在随后的微调或指令调优之后，后门仍然保持健壮。实验结果表明，我们的BadEdit框架可以有效地攻击预先训练的LLMS，成功率高达100%，同时保持了模型在良性输入下的性能。



## **35. Mapping LLM Security Landscapes: A Comprehensive Stakeholder Risk Assessment Proposal**

绘制LLM安全景观图：全面的利益相关者风险评估提案 cs.CR

10 pages, 1 figure, 3 tables

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13309v1) [paper-pdf](http://arxiv.org/pdf/2403.13309v1)

**Authors**: Rahul Pankajakshan, Sumitra Biswal, Yuvaraj Govindarajulu, Gilad Gressel

**Abstract**: The rapid integration of Large Language Models (LLMs) across diverse sectors has marked a transformative era, showcasing remarkable capabilities in text generation and problem-solving tasks. However, this technological advancement is accompanied by significant risks and vulnerabilities. Despite ongoing security enhancements, attackers persistently exploit these weaknesses, casting doubts on the overall trustworthiness of LLMs. Compounding the issue, organisations are deploying LLM-integrated systems without understanding the severity of potential consequences. Existing studies by OWASP and MITRE offer a general overview of threats and vulnerabilities but lack a method for directly and succinctly analysing the risks for security practitioners, developers, and key decision-makers who are working with this novel technology. To address this gap, we propose a risk assessment process using tools like the OWASP risk rating methodology which is used for traditional systems. We conduct scenario analysis to identify potential threat agents and map the dependent system components against vulnerability factors. Through this analysis, we assess the likelihood of a cyberattack. Subsequently, we conduct a thorough impact analysis to derive a comprehensive threat matrix. We also map threats against three key stakeholder groups: developers engaged in model fine-tuning, application developers utilizing third-party APIs, and end users. The proposed threat matrix provides a holistic evaluation of LLM-related risks, enabling stakeholders to make informed decisions for effective mitigation strategies. Our outlined process serves as an actionable and comprehensive tool for security practitioners, offering insights for resource management and enhancing the overall system security.

摘要: 跨不同部门的大型语言模型(LLM)的快速整合标志着一个变革时代的到来，展示了在文本生成和解决问题任务方面的非凡能力。然而，这种技术进步也伴随着重大的风险和脆弱性。尽管正在进行安全增强，攻击者仍不断地利用这些弱点，使人对低成本管理系统的整体可信度产生怀疑。让问题变得更加复杂的是，组织在部署LLM集成系统时，并不了解潜在后果的严重性。OWASP和MITRE的现有研究提供了对威胁和漏洞的总体概述，但缺乏直接和简洁地分析使用这一新技术的安全从业者、开发人员和关键决策者的风险的方法。为了解决这一差距，我们提出了一个风险评估过程，使用传统系统使用的OWASP风险评级方法等工具。我们进行场景分析，以确定潜在的威胁代理，并将依赖的系统组件与漏洞因素进行映射。通过此分析，我们评估了网络攻击的可能性。随后，我们进行了彻底的影响分析，以得出一个全面的威胁矩阵。我们还将威胁映射到三个关键的利益相关者群体：从事模型微调的开发人员、使用第三方API的应用程序开发人员和最终用户。拟议的威胁矩阵提供了对LLM相关风险的全面评估，使利益相关者能够做出明智的决策，制定有效的缓解战略。我们概述的流程为安全从业者提供了一种可操作的综合工具，为资源管理和增强整体系统安全性提供了见解。



## **36. Bypassing LLM Watermarks with Color-Aware Substitutions**

使用颜色感知替代品来处理LLM水印 cs.CR

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.14719v1) [paper-pdf](http://arxiv.org/pdf/2403.14719v1)

**Authors**: Qilong Wu, Varun Chandrasekaran

**Abstract**: Watermarking approaches are proposed to identify if text being circulated is human or large language model (LLM) generated. The state-of-the-art watermarking strategy of Kirchenbauer et al. (2023a) biases the LLM to generate specific (``green'') tokens. However, determining the robustness of this watermarking method is an open problem. Existing attack methods fail to evade detection for longer text segments. We overcome this limitation, and propose {\em Self Color Testing-based Substitution (SCTS)}, the first ``color-aware'' attack. SCTS obtains color information by strategically prompting the watermarked LLM and comparing output tokens frequencies. It uses this information to determine token colors, and substitutes green tokens with non-green ones. In our experiments, SCTS successfully evades watermark detection using fewer number of edits than related work. Additionally, we show both theoretically and empirically that SCTS can remove the watermark for arbitrarily long watermarked text.

摘要: 提出了一种用于识别流传的文本是人类文本还是大型语言模型(LLM)文本的水印方法。Kirchenbauer等人的最新水印策略。(2023a)偏置LLM以生成特定的(‘’绿色‘’)令牌。然而，确定这种水印方法的稳健性是一个开放的问题。现有的攻击方法无法逃避对较长文本段的检测。我们克服了这一局限性，提出了基于自颜色测试的替换(SCTS)，这是第一个“颜色感知”攻击。SCTS通过策略性地提示带水印的LLM和比较输出标记的频率来获得颜色信息。它使用此信息来确定令牌颜色，并用非绿色令牌替换绿色令牌。在我们的实验中，与相关工作相比，SCTS使用更少的编辑次数成功地避开了水印检测。此外，我们在理论和实验上都证明了SCTS可以去除任意长度水印文本的水印。



## **37. Review of Generative AI Methods in Cybersecurity**

网络安全中的生成性人工智能方法综述 cs.CR

40 pages

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.08701v2) [paper-pdf](http://arxiv.org/pdf/2403.08701v2)

**Authors**: Yagmur Yigit, William J Buchanan, Madjid G Tehrani, Leandros Maglaras

**Abstract**: Over the last decade, Artificial Intelligence (AI) has become increasingly popular, especially with the use of chatbots such as ChatGPT, Gemini, and DALL-E. With this rise, large language models (LLMs) and Generative AI (GenAI) have also become more prevalent in everyday use. These advancements strengthen cybersecurity's defensive posture and open up new attack avenues for adversaries as well. This paper provides a comprehensive overview of the current state-of-the-art deployments of GenAI, covering assaults, jailbreaking, and applications of prompt injection and reverse psychology. This paper also provides the various applications of GenAI in cybercrimes, such as automated hacking, phishing emails, social engineering, reverse cryptography, creating attack payloads, and creating malware. GenAI can significantly improve the automation of defensive cyber security processes through strategies such as dataset construction, safe code development, threat intelligence, defensive measures, reporting, and cyberattack detection. In this study, we suggest that future research should focus on developing robust ethical norms and innovative defense mechanisms to address the current issues that GenAI creates and to also further encourage an impartial approach to its future application in cybersecurity. Moreover, we underscore the importance of interdisciplinary approaches further to bridge the gap between scientific developments and ethical considerations.

摘要: 在过去的十年里，人工智能(AI)变得越来越流行，特别是随着ChatGPT、Gemini和Dall-E等聊天机器人的使用。随着这一崛起，大型语言模型(LLM)和生成性人工智能(GenAI)也在日常使用中变得更加普遍。这些进展加强了网络安全的防御态势，也为对手开辟了新的攻击途径。本文全面概述了GenAI当前最先进的部署，包括攻击、越狱以及快速注射和反向心理学的应用。本文还提供了GenAI在网络犯罪中的各种应用，如自动黑客、网络钓鱼电子邮件、社会工程、反向密码学、创建攻击负载和创建恶意软件。GenAI可以通过数据集构建、安全代码开发、威胁情报、防御措施、报告和网络攻击检测等策略，显著提高防御性网络安全流程的自动化程度。在这项研究中，我们建议未来的研究应专注于发展强大的伦理规范和创新的防御机制，以解决GenAI目前造成的问题，并进一步鼓励对其未来在网络安全中的应用采取公正的方法。此外，我们强调跨学科方法的重要性，以进一步弥合科学发展和伦理考量之间的差距。



## **38. RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content**

RigorLLM：针对不期望内容的大型语言模型的弹性防护 cs.CR

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.13031v1) [paper-pdf](http://arxiv.org/pdf/2403.13031v1)

**Authors**: Zhuowen Yuan, Zidi Xiong, Yi Zeng, Ning Yu, Ruoxi Jia, Dawn Song, Bo Li

**Abstract**: Recent advancements in Large Language Models (LLMs) have showcased remarkable capabilities across various tasks in different domains. However, the emergence of biases and the potential for generating harmful content in LLMs, particularly under malicious inputs, pose significant challenges. Current mitigation strategies, while effective, are not resilient under adversarial attacks. This paper introduces Resilient Guardrails for Large Language Models (RigorLLM), a novel framework designed to efficiently and effectively moderate harmful and unsafe inputs and outputs for LLMs. By employing a multi-faceted approach that includes energy-based training data augmentation through Langevin dynamics, optimizing a safe suffix for inputs via minimax optimization, and integrating a fusion-based model combining robust KNN with LLMs based on our data augmentation, RigorLLM offers a robust solution to harmful content moderation. Our experimental evaluations demonstrate that RigorLLM not only outperforms existing baselines like OpenAI API and Perspective API in detecting harmful content but also exhibits unparalleled resilience to jailbreaking attacks. The innovative use of constrained optimization and a fusion-based guardrail approach represents a significant step forward in developing more secure and reliable LLMs, setting a new standard for content moderation frameworks in the face of evolving digital threats.

摘要: 大型语言模型(LLM)的最新进展展示了跨越不同领域的各种任务的显著能力。然而，偏见的出现和在低成本管理中产生有害内容的可能性，特别是在恶意投入下，构成了重大挑战。目前的缓解战略虽然有效，但在对抗性攻击下缺乏弹性。本文介绍了用于大型语言模型的弹性护栏(RigorLLM)，这是一个新的框架，旨在高效和有效地控制LLM中有害和不安全的输入和输出。通过采用多方面的方法，包括通过朗之万动力学基于能量的训练数据增强，通过极小极大优化优化输入的安全后缀，以及基于我们的数据增强将稳健的KNN与LLMS相结合的基于融合的模型，RigorLLM为有害内容适度提供了稳健的解决方案。我们的实验评估表明，RigorLLM不仅在检测有害内容方面优于OpenAI API和透视API等现有基线，而且对越狱攻击表现出无与伦比的弹性。约束优化和基于融合的护栏方法的创新使用代表着在开发更安全可靠的LLMS方面向前迈出的重要一步，为面对不断变化的数字威胁的内容审查框架设定了新的标准。



## **39. Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices**

保护大型语言模型：威胁、漏洞和负责任的实践 cs.CR

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12503v1) [paper-pdf](http://arxiv.org/pdf/2403.12503v1)

**Authors**: Sara Abdali, Richard Anarfi, CJ Barberan, Jia He

**Abstract**: Large language models (LLMs) have significantly transformed the landscape of Natural Language Processing (NLP). Their impact extends across a diverse spectrum of tasks, revolutionizing how we approach language understanding and generations. Nevertheless, alongside their remarkable utility, LLMs introduce critical security and risk considerations. These challenges warrant careful examination to ensure responsible deployment and safeguard against potential vulnerabilities. This research paper thoroughly investigates security and privacy concerns related to LLMs from five thematic perspectives: security and privacy concerns, vulnerabilities against adversarial attacks, potential harms caused by misuses of LLMs, mitigation strategies to address these challenges while identifying limitations of current strategies. Lastly, the paper recommends promising avenues for future research to enhance the security and risk management of LLMs.

摘要: 大型语言模型（LLM）显著改变了自然语言处理（NLP）的前景。它们的影响延伸到各种任务，彻底改变了我们处理语言理解和世代的方式。然而，除了其显著的实用性，LLM引入了关键的安全和风险考虑。这些挑战值得认真审查，以确保负责任地部署和防范潜在漏洞。本研究论文从五个主题角度彻底调查了与LLM相关的安全和隐私问题：安全和隐私问题，对抗攻击的漏洞，滥用LLM造成的潜在危害，缓解策略，以解决这些挑战，同时确定当前策略的局限性。最后，本文建议了未来研究的有希望的途径，以加强LLM的安全性和风险管理。



## **40. Large language models in 6G security: challenges and opportunities**

6G安全中的大型语言模型：挑战与机遇 cs.CR

29 pages, 2 figures

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.12239v1) [paper-pdf](http://arxiv.org/pdf/2403.12239v1)

**Authors**: Tri Nguyen, Huong Nguyen, Ahmad Ijaz, Saeid Sheikhi, Athanasios V. Vasilakos, Panos Kostakos

**Abstract**: The rapid integration of Generative AI (GenAI) and Large Language Models (LLMs) in sectors such as education and healthcare have marked a significant advancement in technology. However, this growth has also led to a largely unexplored aspect: their security vulnerabilities. As the ecosystem that includes both offline and online models, various tools, browser plugins, and third-party applications continues to expand, it significantly widens the attack surface, thereby escalating the potential for security breaches. These expansions in the 6G and beyond landscape provide new avenues for adversaries to manipulate LLMs for malicious purposes. We focus on the security aspects of LLMs from the viewpoint of potential adversaries. We aim to dissect their objectives and methodologies, providing an in-depth analysis of known security weaknesses. This will include the development of a comprehensive threat taxonomy, categorizing various adversary behaviors. Also, our research will concentrate on how LLMs can be integrated into cybersecurity efforts by defense teams, also known as blue teams. We will explore the potential synergy between LLMs and blockchain technology, and how this combination could lead to the development of next-generation, fully autonomous security solutions. This approach aims to establish a unified cybersecurity strategy across the entire computing continuum, enhancing overall digital security infrastructure.

摘要: 生成式人工智能(GenAI)和大型语言模型(LLM)在教育和医疗等领域的快速集成标志着技术的重大进步。然而，这种增长也导致了一个基本上未被探索的方面：它们的安全漏洞。随着包括离线和在线模型、各种工具、浏览器插件和第三方应用程序的生态系统不断扩大，它显著扩大了攻击面，从而增加了安全漏洞的可能性。这些在6G及以上领域的扩展为对手出于恶意目的操纵低层管理提供了新的途径。我们从潜在对手的角度来关注LLMS的安全方面。我们的目标是剖析它们的目标和方法，深入分析已知的安全弱点。这将包括开发一个全面的威胁分类法，对各种对手行为进行分类。此外，我们的研究将集中在如何将LLM整合到防御团队(也称为蓝色团队)的网络安全工作中。我们将探索LLMS和区块链技术之间的潜在协同效应，以及这种结合如何导致下一代完全自主的安全解决方案的开发。这一方法旨在整个计算连续体中建立统一的网络安全战略，增强整体数字安全基础设施。



## **41. Shifting the Lens: Detecting Malware in npm Ecosystem with Large Language Models**

移动镜头：用大型语言模型检测npm生态系统中的恶意软件 cs.CR

13 pages, 1 Figure, 7 tables

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.12196v1) [paper-pdf](http://arxiv.org/pdf/2403.12196v1)

**Authors**: Nusrat Zahan, Philipp Burckhardt, Mikola Lysenko, Feross Aboukhadijeh, Laurie Williams

**Abstract**: The Gartner 2022 report predicts that 45% of organizations worldwide will encounter software supply chain attacks by 2025, highlighting the urgency to improve software supply chain security for community and national interests. Current malware detection techniques aid in the manual review process by filtering benign and malware packages, yet such techniques have high false-positive rates and limited automation support. Therefore, malware detection techniques could benefit from advanced, more automated approaches for accurate and minimally false-positive results. The goal of this study is to assist security analysts in identifying malicious packages through the empirical study of large language models (LLMs) to detect potential malware in the npm ecosystem.   We present SocketAI Scanner, a multi-stage decision-maker malware detection workflow using iterative self-refinement and zero-shot-role-play-Chain of Thought (CoT) prompting techniques for ChatGPT. We studied 5,115 npm packages (of which 2,180 are malicious) and performed a baseline comparison of the GPT-3 and GPT-4 models with a static analysis tool. Our findings showed promising results for GPT models with low misclassification alert rates. Our baseline comparison demonstrates a notable improvement over static analysis in precision scores above 25% and F1 scores above 15%. We attained precision and F1 scores of 91% and 94%, respectively, for the GPT-3 model. Overall, GPT-4 demonstrates superior performance in precision (99%) and F1 (97%) scores, while GPT-3 presents a cost-effective balance between performance and expenditure.

摘要: Gartner 2022报告预测，到2025年，全球45%的组织将遭遇软件供应链攻击，这突显了为社区和国家利益改善软件供应链安全的紧迫性。当前的恶意软件检测技术通过过滤良性和恶意软件包来帮助手动审查过程，但此类技术具有较高的假阳性率和有限的自动化支持。因此，恶意软件检测技术可以受益于先进的、更自动化的方法，以获得准确和最低限度的假阳性结果。这项研究的目标是通过对大型语言模型(LLM)的实证研究来帮助安全分析师识别恶意程序包，以检测NPM生态系统中的潜在恶意软件。我们提出了SocketAI Scanner，这是一个多阶段的决策者恶意软件检测工作流，使用迭代自我求精和零镜头角色扮演思想链(CoT)提示技术来进行ChatGPT。我们研究了5,115个NPM包(其中2,180个是恶意的)，并使用静态分析工具对GPT-3和GPT-4模型进行了基线比较。我们的发现显示，对于GPT模型，错误分类警报率很低，结果令人振奋。我们的基线比较表明，与静态分析相比，精确度得分在25%以上，F1得分在15%以上，有显著的改善。对于GPT-3模型，我们获得了91%的精度和94%的F1得分。总体而言，GPT-4在精度(99%)和F1(97%)分数方面表现出优越的性能，而GPT-3在性能和支出之间实现了成本效益的平衡。



## **42. EasyJailbreak: A Unified Framework for Jailbreaking Large Language Models**

EasyJailbreak：一个统一的大型语言模型越狱框架 cs.CL

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.12171v1) [paper-pdf](http://arxiv.org/pdf/2403.12171v1)

**Authors**: Weikang Zhou, Xiao Wang, Limao Xiong, Han Xia, Yingshuang Gu, Mingxu Chai, Fukang Zhu, Caishuang Huang, Shihan Dou, Zhiheng Xi, Rui Zheng, Songyang Gao, Yicheng Zou, Hang Yan, Yifan Le, Ruohui Wang, Lijun Li, Jing Shao, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: Jailbreak attacks are crucial for identifying and mitigating the security vulnerabilities of Large Language Models (LLMs). They are designed to bypass safeguards and elicit prohibited outputs. However, due to significant differences among various jailbreak methods, there is no standard implementation framework available for the community, which limits comprehensive security evaluations. This paper introduces EasyJailbreak, a unified framework simplifying the construction and evaluation of jailbreak attacks against LLMs. It builds jailbreak attacks using four components: Selector, Mutator, Constraint, and Evaluator. This modular framework enables researchers to easily construct attacks from combinations of novel and existing components. So far, EasyJailbreak supports 11 distinct jailbreak methods and facilitates the security validation of a broad spectrum of LLMs. Our validation across 10 distinct LLMs reveals a significant vulnerability, with an average breach probability of 60% under various jailbreaking attacks. Notably, even advanced models like GPT-3.5-Turbo and GPT-4 exhibit average Attack Success Rates (ASR) of 57% and 33%, respectively. We have released a wealth of resources for researchers, including a web platform, PyPI published package, screencast video, and experimental outputs.

摘要: 越狱攻击对于识别和缓解大型语言模型(LLM)的安全漏洞至关重要。它们旨在绕过保障措施，引出被禁止的产出。然而，由于各种越狱方法之间的显著差异，社区没有可用的标准实施框架，这限制了全面的安全评估。本文介绍了EasyJailBreak，这是一个统一的框架，简化了针对LLMS的越狱攻击的构建和评估。它使用四个组件构建越狱攻击：选择器、赋值器、约束和赋值器。这种模块化框架使研究人员能够轻松地从新组件和现有组件的组合中构建攻击。到目前为止，EasyJailBreak支持11种不同的越狱方法，并促进了广泛范围的LLM的安全验证。我们对10个不同的LLM进行了验证，发现了一个严重的漏洞，在各种越狱攻击下的平均漏洞概率为60%。值得注意的是，即使是像GPT-3.5-Turbo和GPT-4这样的高级型号，平均攻击成功率(ASR)也分别达到了57%和33%。我们已经为研究人员发布了丰富的资源，包括网络平台、PyPI发布的包、截屏视频和实验输出。



## **43. Navigation as Attackers Wish? Towards Building Robust Embodied Agents under Federated Learning**

导航如攻击者所愿？基于联邦学习的鲁棒代理构建方法 cs.AI

**SubmitDate**: 2024-03-16    [abs](http://arxiv.org/abs/2211.14769v4) [paper-pdf](http://arxiv.org/pdf/2211.14769v4)

**Authors**: Yunchao Zhang, Zonglin Di, Kaiwen Zhou, Cihang Xie, Xin Eric Wang

**Abstract**: Federated embodied agent learning protects the data privacy of individual visual environments by keeping data locally at each client (the individual environment) during training. However, since the local data is inaccessible to the server under federated learning, attackers may easily poison the training data of the local client to build a backdoor in the agent without notice. Deploying such an agent raises the risk of potential harm to humans, as the attackers may easily navigate and control the agent as they wish via the backdoor. Towards Byzantine-robust federated embodied agent learning, in this paper, we study the attack and defense for the task of vision-and-language navigation (VLN), where the agent is required to follow natural language instructions to navigate indoor environments. First, we introduce a simple but effective attack strategy, Navigation as Wish (NAW), in which the malicious client manipulates local trajectory data to implant a backdoor into the global model. Results on two VLN datasets (R2R and RxR) show that NAW can easily navigate the deployed VLN agent regardless of the language instruction, without affecting its performance on normal test sets. Then, we propose a new Prompt-Based Aggregation (PBA) to defend against the NAW attack in federated VLN, which provides the server with a ''prompt'' of the vision-and-language alignment variance between the benign and malicious clients so that they can be distinguished during training. We validate the effectiveness of the PBA method on protecting the global model from the NAW attack, which outperforms other state-of-the-art defense methods by a large margin in the defense metrics on R2R and RxR.

摘要: 联合具体化代理学习通过在培训期间在每个客户端(个体环境)本地保存数据来保护个体视觉环境的数据隐私。然而，在联合学习下，由于本地数据对服务器是不可访问的，攻击者很容易毒化本地客户端的训练数据，在没有通知的情况下在代理中构建后门。部署这样的代理会增加对人类造成潜在伤害的风险，因为攻击者可以很容易地通过后门导航和控制代理。针对拜占庭稳健的联邦具身智能体学习，本文研究了视觉语言导航(VLN)任务的攻防问题，该任务要求智能体遵循自然语言指令在室内环境中导航。首先，我们介绍了一种简单但有效的攻击策略，即希望导航(NAW)，在该策略中，恶意客户端操纵局部轨迹数据，在全局模型中植入后门。在两个VLN数据集(R2R和RXR)上的结果表明，NAW可以轻松地导航部署的VLN代理，而不会影响其在正常测试集上的性能。然后，我们提出了一种新的基于提示的聚合(PBA)来防御联邦VLN中的NAW攻击，它为服务器提供了良性客户端和恶意客户端之间视觉和语言对齐差异的“提示”，以便在训练过程中区分它们。我们验证了PBA方法在保护全局模型免受NAW攻击方面的有效性，在R2R和RXR上的防御指标上远远超过了其他最先进的防御方法。



## **44. Bergeron: Combating Adversarial Attacks through a Conscience-Based Alignment Framework**

Bergeron：通过基于意识的调整框架打击对抗性攻击 cs.CR

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2312.00029v2) [paper-pdf](http://arxiv.org/pdf/2312.00029v2)

**Authors**: Matthew Pisano, Peter Ly, Abraham Sanders, Bingsheng Yao, Dakuo Wang, Tomek Strzalkowski, Mei Si

**Abstract**: Research into AI alignment has grown considerably since the recent introduction of increasingly capable Large Language Models (LLMs). Unfortunately, modern methods of alignment still fail to fully prevent harmful responses when models are deliberately attacked. These attacks can trick seemingly aligned models into giving manufacturing instructions for dangerous materials, inciting violence, or recommending other immoral acts. To help mitigate this issue, we introduce Bergeron: a framework designed to improve the robustness of LLMs against attacks without any additional parameter fine-tuning. Bergeron is organized into two tiers; with a secondary LLM emulating the conscience of a protected, primary LLM. This framework better safeguards the primary model against incoming attacks while monitoring its output for any harmful content. Empirical analysis shows that, by using Bergeron to complement models with existing alignment training, we can improve the robustness and safety of multiple, commonly used commercial and open-source LLMs.

摘要: 自从最近引入了功能越来越强大的大型语言模型(LLM)以来，对人工智能对齐的研究有了很大的增长。不幸的是，现代的校准方法仍然不能完全防止模型受到故意攻击时的有害反应。这些攻击可以诱骗看似一致的模型给出危险材料的制造说明，煽动暴力，或推荐其他不道德的行为。为了帮助缓解这个问题，我们引入了Bergeron：一个旨在提高LLM抵御攻击的健壮性的框架，而不需要任何额外的参数微调。Bergeron被组织成两层；二级LLM模仿受保护的初级LLM的良知。此框架可以更好地保护主要模型免受来袭攻击，同时监控其输出中是否有任何有害内容。实证分析表明，通过使用Bergeron来补充模型与现有的比对训练，我们可以提高多个常用的商业和开源LLM的稳健性和安全性。



## **45. Beyond Gradient and Priors in Privacy Attacks: Leveraging Pooler Layer Inputs of Language Models in Federated Learning**

隐私攻击中的超越梯度和先验：在联邦学习中利用语言模型的Poetary层输入 cs.LG

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2312.05720v4) [paper-pdf](http://arxiv.org/pdf/2312.05720v4)

**Authors**: Jianwei Li, Sheng Liu, Qi Lei

**Abstract**: Language models trained via federated learning (FL) demonstrate impressive capabilities in handling complex tasks while protecting user privacy. Recent studies indicate that leveraging gradient information and prior knowledge can potentially reveal training samples within FL setting. However, these investigations have overlooked the potential privacy risks tied to the intrinsic architecture of the models. This paper presents a two-stage privacy attack strategy that targets the vulnerabilities in the architecture of contemporary language models, significantly enhancing attack performance by initially recovering certain feature directions as additional supervisory signals. Our comparative experiments demonstrate superior attack performance across various datasets and scenarios, highlighting the privacy leakage risk associated with the increasingly complex architectures of language models. We call for the community to recognize and address these potential privacy risks in designing large language models.

摘要: 通过联合学习(FL)训练的语言模型在处理复杂任务同时保护用户隐私方面表现出令人印象深刻的能力。最近的研究表明，利用梯度信息和先验知识可以潜在地揭示外语环境下的训练样本。然而，这些调查忽略了与这些模型的内在架构相关的潜在隐私风险。本文提出了一种两阶段隐私攻击策略，该策略针对当代语言模型体系结构中的漏洞，通过最初将某些特征方向恢复为额外的监督信号来显著提高攻击性能。我们的对比实验表明，在各种数据集和场景中具有卓越的攻击性能，突出了与日益复杂的语言模型体系结构相关的隐私泄露风险。我们呼吁社区在设计大型语言模型时认识到并解决这些潜在的隐私风险。



## **46. Logits of API-Protected LLMs Leak Proprietary Information**

API保护的LLM日志泄露专有信息 cs.CL

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.09539v2) [paper-pdf](http://arxiv.org/pdf/2403.09539v2)

**Authors**: Matthew Finlayson, Xiang Ren, Swabha Swayamdipta

**Abstract**: The commercialization of large language models (LLMs) has led to the common practice of high-level API-only access to proprietary models. In this work, we show that even with a conservative assumption about the model architecture, it is possible to learn a surprisingly large amount of non-public information about an API-protected LLM from a relatively small number of API queries (e.g., costing under $1,000 for OpenAI's gpt-3.5-turbo). Our findings are centered on one key observation: most modern LLMs suffer from a softmax bottleneck, which restricts the model outputs to a linear subspace of the full output space. We show that this lends itself to a model image or a model signature which unlocks several capabilities with affordable cost: efficiently discovering the LLM's hidden size, obtaining full-vocabulary outputs, detecting and disambiguating different model updates, identifying the source LLM given a single full LLM output, and even estimating the output layer parameters. Our empirical investigations show the effectiveness of our methods, which allow us to estimate the embedding size of OpenAI's gpt-3.5-turbo to be about 4,096. Lastly, we discuss ways that LLM providers can guard against these attacks, as well as how these capabilities can be viewed as a feature (rather than a bug) by allowing for greater transparency and accountability.

摘要: 大型语言模型(LLM)的商业化导致了仅通过高级API访问专有模型的普遍做法。在这项工作中，我们表明，即使对模型体系结构采取保守的假设，也可以从相对较少的API查询(例如，OpenAI的gpt-3.5-turbo的成本低于1,000美元)中了解到关于受API保护的LLM的大量非公开信息。我们的发现集中在一个关键的观察上：大多数现代LLMS都存在Softmax瓶颈，这将模型输出限制在整个输出空间的线性子空间。我们表明，这有助于模型图像或模型签名，它以负担得起的成本解锁了几种功能：有效地发现LLM的隐藏大小，获得完整的词汇表输出，检测和消除不同的模型更新，在给定单个完整的LLM输出的情况下识别源LLM，甚至估计输出层参数。我们的实证研究表明，我们的方法是有效的，允许我们估计OpenAI的gpt-3.5-turbo的嵌入大小约为4,096。最后，我们讨论LLM提供商防范这些攻击的方法，以及如何通过允许更高的透明度和责任来将这些功能视为一项功能(而不是错误)。



## **47. Scaling Behavior of Machine Translation with Large Language Models under Prompt Injection Attacks**

即时注入攻击下大语言模型机器翻译的缩放行为 cs.CL

15 pages, 18 figures, First Workshop on the Scaling Behavior of Large  Language Models (SCALE-LLM 2024)

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09832v1) [paper-pdf](http://arxiv.org/pdf/2403.09832v1)

**Authors**: Zhifan Sun, Antonio Valerio Miceli-Barone

**Abstract**: Large Language Models (LLMs) are increasingly becoming the preferred foundation platforms for many Natural Language Processing tasks such as Machine Translation, owing to their quality often comparable to or better than task-specific models, and the simplicity of specifying the task through natural language instructions or in-context examples. Their generality, however, opens them up to subversion by end users who may embed into their requests instructions that cause the model to behave in unauthorized and possibly unsafe ways. In this work we study these Prompt Injection Attacks (PIAs) on multiple families of LLMs on a Machine Translation task, focusing on the effects of model size on the attack success rates. We introduce a new benchmark data set and we discover that on multiple language pairs and injected prompts written in English, larger models under certain conditions may become more susceptible to successful attacks, an instance of the Inverse Scaling phenomenon (McKenzie et al., 2023). To our knowledge, this is the first work to study non-trivial LLM scaling behaviour in a multi-lingual setting.

摘要: 大语言模型正日益成为机器翻译等许多自然语言处理任务的首选基础平台，因为它们的质量通常可以与特定于任务的模型相媲美或更好，并且通过自然语言指令或上下文中的例子来指定任务的简单性。然而，它们的一般性使它们很容易被终端用户颠覆，最终用户可能会在其请求中嵌入导致模型以未经授权的甚至可能不安全的方式行为的指令。在这项工作中，我们研究了机器翻译任务中针对多个LLM家族的快速注入攻击(PIA)，重点研究了模型大小对攻击成功率的影响。我们引入了一个新的基准数据集，我们发现在多语言对和用英语编写的插入提示上，在某些条件下，较大的模型可能更容易受到成功的攻击，这是反向缩放现象的一个例子(McKenzie等人，2023)。据我们所知，这是第一个研究多语言环境下非平凡LLM标度行为的工作。



## **48. Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models**

图像是对齐的致命弱点：利用多模态大型语言模型的视觉漏洞 cs.CV

Work in progress

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09792v1) [paper-pdf](http://arxiv.org/pdf/2403.09792v1)

**Authors**: Yifan Li, Hangyu Guo, Kun Zhou, Wayne Xin Zhao, Ji-Rong Wen

**Abstract**: In this paper, we study the harmlessness alignment problem of multimodal large language models~(MLLMs). We conduct a systematic empirical analysis of the harmlessness performance of representative MLLMs and reveal that the image input poses the alignment vulnerability of MLLMs. Inspired by this, we propose a novel jailbreak method named HADES, which hides and amplifies the harmfulness of the malicious intent within the text input, using meticulously crafted images. Experimental results show that HADES can effectively jailbreak existing MLLMs, which achieves an average Attack Success Rate~(ASR) of 90.26% for LLaVA-1.5 and 71.60% for Gemini Pro Vision. Our code and data will be publicly released.

摘要: 本文研究了多模态大语言模型MLLM的无害对齐问题.我们对典型的多线性线性阵列的无害性性能进行了系统的实证分析，揭示了图像输入造成了多线性阵列的对准脆弱性。受此启发，我们提出了一种名为HADES的新越狱方法，该方法使用精心制作的图像隐藏和放大了文本输入中恶意意图的危害性。实验结果表明，HADES可以有效地破解现有MLLM，LLaVA—1.5和Gemini Pro Vision的平均攻击成功率分别为90.26%和71.60%。我们的代码和数据将公开发布。



## **49. AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting**

AdaShield：通过自适应护盾保护多模态大型语言模型免受基于结构的攻击 cs.CR

Multimodal Large Language Models Defense, 25 Pages

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09513v1) [paper-pdf](http://arxiv.org/pdf/2403.09513v1)

**Authors**: Yu Wang, Xiaogeng Liu, Yu Li, Muhao Chen, Chaowei Xiao

**Abstract**: With the advent and widespread deployment of Multimodal Large Language Models (MLLMs), the imperative to ensure their safety has become increasingly pronounced. However, with the integration of additional modalities, MLLMs are exposed to new vulnerabilities, rendering them prone to structured-based jailbreak attacks, where semantic content (e.g., "harmful text") has been injected into the images to mislead MLLMs. In this work, we aim to defend against such threats. Specifically, we propose \textbf{Ada}ptive \textbf{Shield} Prompting (\textbf{AdaShield}), which prepends inputs with defense prompts to defend MLLMs against structure-based jailbreak attacks without fine-tuning MLLMs or training additional modules (e.g., post-stage content detector). Initially, we present a manually designed static defense prompt, which thoroughly examines the image and instruction content step by step and specifies response methods to malicious queries. Furthermore, we introduce an adaptive auto-refinement framework, consisting of a target MLLM and a LLM-based defense prompt generator (Defender). These components collaboratively and iteratively communicate to generate a defense prompt. Extensive experiments on the popular structure-based jailbreak attacks and benign datasets show that our methods can consistently improve MLLMs' robustness against structure-based jailbreak attacks without compromising the model's general capabilities evaluated on standard benign tasks. Our code is available at https://github.com/rain305f/AdaShield.

摘要: 随着多通道大型语言模型(MLLMS)的出现和广泛应用，确保其安全性的必要性日益明显。然而，随着更多模式的集成，MLLMS面临新的漏洞，使它们容易受到基于结构化的越狱攻击，其中语义内容(例如，“有害文本”)已被注入图像以误导MLLMS。在这项工作中，我们的目标是防御此类威胁。具体地说，我们提出了\extbf{ada}Patitive\extbf{Shield}提示(\extbf{AdaShield})，它将输入与防御提示放在前面，以保护MLLM免受基于结构的越狱攻击，而无需微调MLLM或培训额外的模块(例如，后期内容检测器)。首先，我们提出了一种手动设计的静态防御提示，该提示逐级彻底检查图像和说明内容，并指定对恶意查询的响应方法。此外，我们还提出了一种自适应自动求精框架，该框架由目标MLLM和基于LLM的防御提示生成器(Defender)组成。这些组件以协作和迭代方式进行通信，以生成防御提示。在流行的基于结构的越狱攻击和良性数据集上的大量实验表明，我们的方法可以持续提高MLLMS对基于结构的越狱攻击的健壮性，而不会影响该模型在标准良性任务上评估的一般性能。我们的代码可以在https://github.com/rain305f/AdaShield.上找到



## **50. On Protecting the Data Privacy of Large Language Models (LLMs): A Survey**

大型语言模型（LLM）数据隐私保护研究综述 cs.CR

18 pages, 4 figures

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.05156v2) [paper-pdf](http://arxiv.org/pdf/2403.05156v2)

**Authors**: Biwei Yan, Kun Li, Minghui Xu, Yueyan Dong, Yue Zhang, Zhaochun Ren, Xiuzhen Cheng

**Abstract**: Large language models (LLMs) are complex artificial intelligence systems capable of understanding, generating and translating human language. They learn language patterns by analyzing large amounts of text data, allowing them to perform writing, conversation, summarizing and other language tasks. When LLMs process and generate large amounts of data, there is a risk of leaking sensitive information, which may threaten data privacy. This paper concentrates on elucidating the data privacy concerns associated with LLMs to foster a comprehensive understanding. Specifically, a thorough investigation is undertaken to delineate the spectrum of data privacy threats, encompassing both passive privacy leakage and active privacy attacks within LLMs. Subsequently, we conduct an assessment of the privacy protection mechanisms employed by LLMs at various stages, followed by a detailed examination of their efficacy and constraints. Finally, the discourse extends to delineate the challenges encountered and outline prospective directions for advancement in the realm of LLM privacy protection.

摘要: 大语言模型是一种能够理解、生成和翻译人类语言的复杂人工智能系统。他们通过分析大量的文本数据来学习语言模式，使他们能够执行写作、对话、摘要和其他语言任务。当LLMS处理和生成大量数据时，存在泄露敏感信息的风险，这可能会威胁到数据隐私。本文集中于阐明与低成本管理相关的数据隐私问题，以促进全面的理解。具体地说，进行了彻底的调查，以勾勒出数据隐私威胁的范围，包括LLMS中的被动隐私泄露和主动隐私攻击。随后，我们对LLMS在不同阶段采用的隐私保护机制进行了评估，然后详细研究了它们的有效性和制约因素。最后，论述了所遇到的挑战，并勾勒出在LLM隐私保护领域取得进展的预期方向。



