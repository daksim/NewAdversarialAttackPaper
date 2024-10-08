# Latest Large Language Model Attack Papers
**update at 2024-10-08 09:45:24**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Obliviate: Neutralizing Task-agnostic Backdoors within the Parameter-efficient Fine-tuning Paradigm**

遗忘：消除参数高效微调范式中的任务不可知后门 cs.CL

Under Review

**SubmitDate**: 2024-10-06    [abs](http://arxiv.org/abs/2409.14119v3) [paper-pdf](http://arxiv.org/pdf/2409.14119v3)

**Authors**: Jaehan Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin

**Abstract**: Parameter-efficient fine-tuning (PEFT) has become a key training strategy for large language models. However, its reliance on fewer trainable parameters poses security risks, such as task-agnostic backdoors. Despite their severe impact on a wide range of tasks, there is no practical defense solution available that effectively counters task-agnostic backdoors within the context of PEFT. In this study, we introduce Obliviate, a PEFT-integrable backdoor defense. We develop two techniques aimed at amplifying benign neurons within PEFT layers and penalizing the influence of trigger tokens. Our evaluations across three major PEFT architectures show that our method can significantly reduce the attack success rate of the state-of-the-art task-agnostic backdoors (83.6%$\downarrow$). Furthermore, our method exhibits robust defense capabilities against both task-specific backdoors and adaptive attacks. Source code will be obtained at https://github.com/obliviateARR/Obliviate.

摘要: 参数高效微调（PEFT）已成为大型语言模型的关键训练策略。然而，它对较少可训练参数的依赖会带来安全风险，例如任务不可知的后门。尽管它们对广泛的任务产生了严重影响，但没有实用的防御解决方案可以有效地对抗PEFT背景下的任务不可知后门。在这项研究中，我们引入Obliviate，一种PEFT可集成的后门防御。我们开发了两种技术，旨在放大PEFT层内的良性神经元并惩罚触发代币的影响。我们对三种主要PEFT架构的评估表明，我们的方法可以显着降低最先进的任务不可知后门（83.6%$\down arrow $）的攻击成功率。此外，我们的方法对特定任务的后门和自适应攻击都表现出强大的防御能力。源代码可在https://github.com/obliviateARR/Obliviate上获取。



## **2. AppPoet: Large Language Model based Android malware detection via multi-view prompt engineering**

AppPoet：通过多视图提示工程进行基于大语言模型的Android恶意软件检测 cs.CR

**SubmitDate**: 2024-10-06    [abs](http://arxiv.org/abs/2404.18816v2) [paper-pdf](http://arxiv.org/pdf/2404.18816v2)

**Authors**: Wenxiang Zhao, Juntao Wu, Zhaoyi Meng

**Abstract**: Due to the vast array of Android applications, their multifarious functions and intricate behavioral semantics, attackers can adopt various tactics to conceal their genuine attack intentions within legitimate functions. However, numerous learning-based methods suffer from a limitation in mining behavioral semantic information, thus impeding the accuracy and efficiency of Android malware detection. Besides, the majority of existing learning-based methods are weakly interpretive and fail to furnish researchers with effective and readable detection reports. Inspired by the success of the Large Language Models (LLMs) in natural language understanding, we propose AppPoet, a LLM-assisted multi-view system for Android malware detection. Firstly, AppPoet employs a static method to comprehensively collect application features and formulate various observation views. Then, using our carefully crafted multi-view prompt templates, it guides the LLM to generate function descriptions and behavioral summaries for each view, enabling deep semantic analysis of the views. Finally, we collaboratively fuse the multi-view information to efficiently and accurately detect malware through a deep neural network (DNN) classifier and then generate the human-readable diagnostic reports. Experimental results demonstrate that our method achieves a detection accuracy of 97.15% and an F1 score of 97.21%, which is superior to the baseline methods. Furthermore, the case study evaluates the effectiveness of our generated diagnostic reports.

摘要: 由于Android应用种类繁多，功能多样，行为语义错综复杂，攻击者可以采取各种策略，将真实的攻击意图隐藏在合法的功能中。然而，许多基于学习的方法在挖掘行为语义信息方面存在局限性，从而阻碍了Android恶意软件检测的准确性和效率。此外，现有的基于学习的方法大多解释性较弱，不能为研究人员提供有效的、可读性强的检测报告。受大语言模型在自然语言理解方面的成功启发，我们提出了一种基于大语言模型的Android恶意软件检测系统AppPoet。首先，AppPoet使用静态的方法来全面收集应用程序的特征，并制定各种观察视图。然后，使用我们精心设计的多视图提示模板，它指导LLM为每个视图生成功能描述和行为摘要，从而实现对视图的深入语义分析。最后，通过深度神经网络(DNN)分类器对多视图信息进行协同融合，高效准确地检测出恶意软件，并生成人类可读的诊断报告。实验结果表明，该方法的检测正确率为97.15%，F1评分为97.21%，优于基线方法。此外，案例研究还评估了我们生成的诊断报告的有效性。



## **3. Functional Homotopy: Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Attacks**

功能同伦：通过LLM越狱攻击的连续参数平滑离散优化 cs.LG

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.04234v1) [paper-pdf](http://arxiv.org/pdf/2410.04234v1)

**Authors**: Zi Wang, Divyam Anshumaan, Ashish Hooda, Yudong Chen, Somesh Jha

**Abstract**: Optimization methods are widely employed in deep learning to identify and mitigate undesired model responses. While gradient-based techniques have proven effective for image models, their application to language models is hindered by the discrete nature of the input space. This study introduces a novel optimization approach, termed the \emph{functional homotopy} method, which leverages the functional duality between model training and input generation. By constructing a series of easy-to-hard optimization problems, we iteratively solve these problems using principles derived from established homotopy methods. We apply this approach to jailbreak attack synthesis for large language models (LLMs), achieving a $20\%-30\%$ improvement in success rate over existing methods in circumventing established safe open-source models such as Llama-2 and Llama-3.

摘要: 优化方法广泛应用于深度学习中，以识别和减轻不希望的模型响应。虽然基于梯度的技术已被证明对图像模型有效，但它们在语言模型中的应用受到输入空间的离散性的阻碍。这项研究引入了一种新型优化方法，称为\{函数同伦}方法，它利用了模型训练和输入生成之间的函数二元性。通过构建一系列容易到难的优化问题，我们使用从已建立的同伦方法推导出的原则迭代解决这些问题。我们将这种方法应用于大型语言模型（LLM）的越狱攻击合成，在规避已建立的安全开源模型（例如Llama-2和Llama-3）方面，与现有方法相比，成功率提高了20 -30美元。



## **4. Adversarial Suffixes May Be Features Too!**

敌对后缀也可能是功能！ cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.00451v2) [paper-pdf](http://arxiv.org/pdf/2410.00451v2)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Despite significant ongoing efforts in safety alignment, large language models (LLMs) such as GPT-4 and LLaMA 3 remain vulnerable to jailbreak attacks that can induce harmful behaviors, including those triggered by adversarial suffixes. Building on prior research, we hypothesize that these adversarial suffixes are not mere bugs but may represent features that can dominate the LLM's behavior. To evaluate this hypothesis, we conduct several experiments. First, we demonstrate that benign features can be effectively made to function as adversarial suffixes, i.e., we develop a feature extraction method to extract sample-agnostic features from benign dataset in the form of suffixes and show that these suffixes may effectively compromise safety alignment. Second, we show that adversarial suffixes generated from jailbreak attacks may contain meaningful features, i.e., appending the same suffix to different prompts results in responses exhibiting specific characteristics. Third, we show that such benign-yet-safety-compromising features can be easily introduced through fine-tuning using only benign datasets, i.e., even in the absence of harmful content. This highlights the critical risk posed by dominating benign features in the training data and calls for further research to reinforce LLM safety alignment. Our code and data is available at \url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.

摘要: 尽管在安全匹配方面正在进行重大的努力，但GPT-4和Llama 3等大型语言模型(LLM)仍然容易受到越狱攻击，这些攻击可能会导致有害行为，包括由对抗性后缀触发的行为。在先前研究的基础上，我们假设这些对抗性后缀不仅仅是错误，而且可能代表可以主导LLM行为的特征。为了评估这一假设，我们进行了几个实验。首先，我们证明了良性特征可以有效地用作对抗性后缀，即，我们开发了一种特征提取方法来从良性数据集中提取与样本无关的后缀形式的特征，并表明这些后缀可以有效地危害安全对齐。其次，我们证明了越狱攻击产生的对抗性后缀可能包含有意义的特征，即在不同的提示后添加相同的后缀会导致响应表现出特定的特征。第三，我们表明，这种良性但危及安全的特征可以通过仅使用良性数据集进行微调来轻松引入，即即使在没有有害内容的情况下也可以。这突出了在训练数据中占主导地位的良性特征所构成的关键风险，并呼吁进一步研究以加强LLM的安全一致性。我们的代码和数据可在\url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.上获得



## **5. Automated Progressive Red Teaming**

自动化渐进式红色团队 cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2407.03876v2) [paper-pdf](http://arxiv.org/pdf/2407.03876v2)

**Authors**: Bojian Jiang, Yi Jing, Tianhao Shen, Tong Wu, Qing Yang, Deyi Xiong

**Abstract**: Ensuring the safety of large language models (LLMs) is paramount, yet identifying potential vulnerabilities is challenging. While manual red teaming is effective, it is time-consuming, costly and lacks scalability. Automated red teaming (ART) offers a more cost-effective alternative, automatically generating adversarial prompts to expose LLM vulnerabilities. However, in current ART efforts, a robust framework is absent, which explicitly frames red teaming as an effectively learnable task. To address this gap, we propose Automated Progressive Red Teaming (APRT) as an effectively learnable framework. APRT leverages three core modules: an Intention Expanding LLM that generates diverse initial attack samples, an Intention Hiding LLM that crafts deceptive prompts, and an Evil Maker to manage prompt diversity and filter ineffective samples. The three modules collectively and progressively explore and exploit LLM vulnerabilities through multi-round interactions. In addition to the framework, we further propose a novel indicator, Attack Effectiveness Rate (AER) to mitigate the limitations of existing evaluation metrics. By measuring the likelihood of eliciting unsafe but seemingly helpful responses, AER aligns closely with human evaluations. Extensive experiments with both automatic and human evaluations, demonstrate the effectiveness of ARPT across both open- and closed-source LLMs. Specifically, APRT effectively elicits 54% unsafe yet useful responses from Meta's Llama-3-8B-Instruct, 50% from GPT-4o (API access), and 39% from Claude-3.5 (API access), showcasing its robust attack capability and transferability across LLMs (especially from open-source LLMs to closed-source LLMs).

摘要: 确保大型语言模型(LLM)的安全是最重要的，但识别潜在的漏洞是具有挑战性的。虽然手动红色团队是有效的，但它耗时、成本高，而且缺乏可扩展性。自动红色团队(ART)提供了一种更具成本效益的替代方案，可自动生成敌意提示以暴露LLM漏洞。然而，在目前的艺术努力中，缺乏一个强大的框架，它明确地将红色团队作为一项有效的可学习任务。为了弥补这一差距，我们提出了自动渐进红色团队(APRT)作为一种有效的可学习框架。APRT利用三个核心模块：用于生成不同初始攻击样本的意图扩展LLM，用于制作欺骗性提示的意图隐藏LLM，以及用于管理提示多样性和过滤无效样本的邪恶制造者。这三个模块通过多轮交互共同逐步探索和利用LLM漏洞。除了该框架外，我们进一步提出了一个新的指标--攻击效率(AER)，以缓解现有评估指标的局限性。通过衡量引发不安全但似乎有帮助的反应的可能性，AER与人类的评估密切一致。自动和人工评估的广泛实验证明了ARPT在开放源码和封闭源码LLM中的有效性。具体地说，APRT有效地从Meta的Llama-3-8B-Indict、GPT-40(API访问)和Claude-3.5(API访问)中引发了54%的不安全但有用的响应，展示了其强大的攻击能力和跨LLM(特别是从开源LLM到闭源LLM)的可转移性。



## **6. Harnessing Task Overload for Scalable Jailbreak Attacks on Large Language Models**

利用任务过载对大型语言模型进行可扩展越狱攻击 cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.04190v1) [paper-pdf](http://arxiv.org/pdf/2410.04190v1)

**Authors**: Yiting Dong, Guobin Shen, Dongcheng Zhao, Xiang He, Yi Zeng

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreak attacks that bypass their safety mechanisms. Existing attack methods are fixed or specifically tailored for certain models and cannot flexibly adjust attack strength, which is critical for generalization when attacking models of various sizes. We introduce a novel scalable jailbreak attack that preempts the activation of an LLM's safety policies by occupying its computational resources. Our method involves engaging the LLM in a resource-intensive preliminary task - a Character Map lookup and decoding process - before presenting the target instruction. By saturating the model's processing capacity, we prevent the activation of safety protocols when processing the subsequent instruction. Extensive experiments on state-of-the-art LLMs demonstrate that our method achieves a high success rate in bypassing safety measures without requiring gradient access, manual prompt engineering. We verified our approach offers a scalable attack that quantifies attack strength and adapts to different model scales at the optimal strength. We shows safety policies of LLMs might be more susceptible to resource constraints. Our findings reveal a critical vulnerability in current LLM safety designs, highlighting the need for more robust defense strategies that account for resource-intense condition.

摘要: 大型语言模型(LLM)仍然容易受到绕过其安全机制的越狱攻击。现有的攻击方法是固定的或针对特定模型量身定做的，不能灵活调整攻击强度，这对于攻击不同大小的模型时的泛化至关重要。我们提出了一种新的可扩展的越狱攻击，该攻击通过占用LLM的计算资源来抢占LLM安全策略的激活。我们的方法包括在呈现目标指令之前，让LLM参与一个资源密集型的预备任务-字符映射查找和解码过程。通过使模型的处理能力饱和，我们防止在处理后续指令时激活安全协议。在最先进的LLMS上的广泛实验表明，我们的方法在绕过安全措施方面取得了很高的成功率，而不需要梯度访问、人工提示工程。我们验证了我们的方法提供了一种可扩展的攻击，它量化了攻击强度，并以最佳强度适应不同的模型规模。我们发现，低成本管理的安全政策可能更容易受到资源约束的影响。我们的发现揭示了当前LLM安全设计中的一个严重漏洞，突显了需要更强大的防御战略来应对资源密集型条件。



## **7. Can We Trust Embodied Agents? Exploring Backdoor Attacks against Embodied LLM-based Decision-Making Systems**

我们可以信任有保障的代理人吗？探索针对基于LLM的决策系统的后门攻击 cs.CR

31 pages, including main paper, references, and appendix

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2405.20774v2) [paper-pdf](http://arxiv.org/pdf/2405.20774v2)

**Authors**: Ruochen Jiao, Shaoyuan Xie, Justin Yue, Takami Sato, Lixu Wang, Yixuan Wang, Qi Alfred Chen, Qi Zhu

**Abstract**: Large Language Models (LLMs) have shown significant promise in real-world decision-making tasks for embodied artificial intelligence, especially when fine-tuned to leverage their inherent common sense and reasoning abilities while being tailored to specific applications. However, this fine-tuning process introduces considerable safety and security vulnerabilities, especially in safety-critical cyber-physical systems. In this work, we propose the first comprehensive framework for Backdoor Attacks against LLM-based Decision-making systems (BALD) in embodied AI, systematically exploring the attack surfaces and trigger mechanisms. Specifically, we propose three distinct attack mechanisms: word injection, scenario manipulation, and knowledge injection, targeting various components in the LLM-based decision-making pipeline. We perform extensive experiments on representative LLMs (GPT-3.5, LLaMA2, PaLM2) in autonomous driving and home robot tasks, demonstrating the effectiveness and stealthiness of our backdoor triggers across various attack channels, with cases like vehicles accelerating toward obstacles and robots placing knives on beds. Our word and knowledge injection attacks achieve nearly 100% success rate across multiple models and datasets while requiring only limited access to the system. Our scenario manipulation attack yields success rates exceeding 65%, reaching up to 90%, and does not require any runtime system intrusion. We also assess the robustness of these attacks against defenses, revealing their resilience. Our findings highlight critical security vulnerabilities in embodied LLM systems and emphasize the urgent need for safeguarding these systems to mitigate potential risks.

摘要: 大型语言模型(LLM)在真实世界的人工智能决策任务中显示出了巨大的前景，特别是在进行微调以利用它们固有的常识和推理能力，同时为特定应用量身定做时。然而，这一微调过程引入了相当大的安全和安全漏洞，特别是在安全关键的网络物理系统中。在这项工作中，我们提出了第一个全面的框架，对基于LLM的决策系统(BALD)的后门攻击，系统地研究了攻击面和触发机制。具体地说，我们提出了三种不同的攻击机制：单词注入、场景操纵和知识注入，分别针对基于LLM的决策流水线中的各个组件。我们在自主驾驶和家用机器人任务中对具有代表性的LLM(GPT-3.5、LLaMA2、Palm2)进行了广泛的实验，展示了我们的后门触发器在各种攻击渠道中的有效性和隐蔽性，例如车辆加速驶向障碍物和机器人将刀放在床上。我们的单词和知识注入攻击在多个模型和数据集上实现了近100%的成功率，而只需要有限的系统访问权限。我们的场景操纵攻击的成功率超过65%，高达90%，并且不需要任何运行时系统入侵。我们还评估了这些攻击对防御的健壮性，揭示了它们的弹性。我们的发现突出了嵌入式LLM系统中的关键安全漏洞，并强调了保护这些系统以降低潜在风险的迫切需要。



## **8. ASPIRER: Bypassing System Prompts With Permutation-based Backdoors in LLMs**

ASPIRER：在LLM中使用基于置换的后门来确定询问系统 cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.04009v1) [paper-pdf](http://arxiv.org/pdf/2410.04009v1)

**Authors**: Lu Yan, Siyuan Cheng, Xuan Chen, Kaiyuan Zhang, Guangyu Shen, Zhuo Zhang, Xiangyu Zhang

**Abstract**: Large Language Models (LLMs) have become integral to many applications, with system prompts serving as a key mechanism to regulate model behavior and ensure ethical outputs. In this paper, we introduce a novel backdoor attack that systematically bypasses these system prompts, posing significant risks to the AI supply chain. Under normal conditions, the model adheres strictly to its system prompts. However, our backdoor allows malicious actors to circumvent these safeguards when triggered. Specifically, we explore a scenario where an LLM provider embeds a covert trigger within the base model. A downstream deployer, unaware of the hidden trigger, fine-tunes the model and offers it as a service to users. Malicious actors can purchase the trigger from the provider and use it to exploit the deployed model, disabling system prompts and achieving restricted outcomes. Our attack utilizes a permutation trigger, which activates only when its components are arranged in a precise order, making it computationally challenging to detect or reverse-engineer. We evaluate our approach on five state-of-the-art models, demonstrating that our method achieves an attack success rate (ASR) of up to 99.50% while maintaining a clean accuracy (CACC) of 98.58%, even after defensive fine-tuning. These findings highlight critical vulnerabilities in LLM deployment pipelines and underscore the need for stronger defenses.

摘要: 大型语言模型(LLM)已经成为许多应用程序不可或缺的一部分，系统提示是规范模型行为和确保道德输出的关键机制。在本文中，我们引入了一种新型的后门攻击，它系统地绕过了这些系统提示，给人工智能供应链带来了重大风险。在正常情况下，模型严格遵循其系统提示。然而，我们的后门允许恶意行为者在触发时绕过这些安全措施。具体地说，我们将探讨LLM提供程序在基本模型中嵌入隐蔽触发器的场景。下游部署人员不知道隐藏的触发器，对模型进行微调，并将其作为服务提供给用户。恶意攻击者可以从提供商购买触发器，并使用它来利用已部署的模型，从而禁用系统提示并实现受限的结果。我们的攻击利用了置换触发器，只有当其组件按精确顺序排列时才会激活，这使得检测或反向工程在计算上具有挑战性。我们在五个最先进的模型上对我们的方法进行了评估，表明我们的方法实现了高达99.50%的攻击成功率(ASR)，同时保持了98.58%的干净准确率(CACC)，即使在防御微调之后也是如此。这些发现突显了LLM部署管道中的关键漏洞，并强调了加强防御的必要性。



## **9. You Know What I'm Saying -- Jailbreak Attack via Implicit Reference**

你知道我在说什么--通过隐性引用进行越狱攻击 cs.CL

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03857v1) [paper-pdf](http://arxiv.org/pdf/2410.03857v1)

**Authors**: Tianyu Wu, Lingrui Mei, Ruibin Yuan, Lujun Li, Wei Xue, Yike Guo

**Abstract**: While recent advancements in large language model (LLM) alignment have enabled the effective identification of malicious objectives involving scene nesting and keyword rewriting, our study reveals that these methods remain inadequate at detecting malicious objectives expressed through context within nested harmless objectives. This study identifies a previously overlooked vulnerability, which we term Attack via Implicit Reference (AIR). AIR decomposes a malicious objective into permissible objectives and links them through implicit references within the context. This method employs multiple related harmless objectives to generate malicious content without triggering refusal responses, thereby effectively bypassing existing detection techniques.Our experiments demonstrate AIR's effectiveness across state-of-the-art LLMs, achieving an attack success rate (ASR) exceeding 90% on most models, including GPT-4o, Claude-3.5-Sonnet, and Qwen-2-72B. Notably, we observe an inverse scaling phenomenon, where larger models are more vulnerable to this attack method. These findings underscore the urgent need for defense mechanisms capable of understanding and preventing contextual attacks. Furthermore, we introduce a cross-model attack strategy that leverages less secure models to generate malicious contexts, thereby further increasing the ASR when targeting other models.Our code and jailbreak artifacts can be found at https://github.com/Lucas-TY/llm_Implicit_reference.

摘要: 虽然最近在大语言模型(LLM)对齐方面的进展使得能够有效地识别涉及场景嵌套和关键字重写的恶意目标，但我们的研究表明，这些方法在检测嵌套的无害目标中通过上下文表达的恶意目标方面仍然不足。这项研究发现了一个以前被忽视的漏洞，我们将其称为隐式引用攻击(AIR)。AIR将恶意目标分解为允许的目标，并通过上下文中的隐式引用将它们链接起来。该方法利用多个相关的无害目标在不触发拒绝响应的情况下生成恶意内容，从而有效地绕过了现有的检测技术。我们的实验证明了AIR在最先进的LLM上的有效性，在包括GPT-40、Claude-3.5-Sonnet和Qwen-2-72B在内的大多数型号上实现了超过90%的攻击成功率(ASR)。值得注意的是，我们观察到了反向缩放现象，其中较大的模型更容易受到这种攻击方法的攻击。这些发现突显了迫切需要能够理解和防止上下文攻击的防御机制。此外，我们引入了一种跨模型攻击策略，该策略利用安全性较低的模型来生成恶意上下文，从而进一步提高了针对其他模型的ASR。我们的代码和越狱人工产物可以在https://github.com/Lucas-TY/llm_Implicit_reference.找到



## **10. Detecting Machine-Generated Long-Form Content with Latent-Space Variables**

检测具有潜在空间变量的机器生成的长形式内容 cs.CL

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03856v1) [paper-pdf](http://arxiv.org/pdf/2410.03856v1)

**Authors**: Yufei Tian, Zeyu Pan, Nanyun Peng

**Abstract**: The increasing capability of large language models (LLMs) to generate fluent long-form texts is presenting new challenges in distinguishing machine-generated outputs from human-written ones, which is crucial for ensuring authenticity and trustworthiness of expressions. Existing zero-shot detectors primarily focus on token-level distributions, which are vulnerable to real-world domain shifts, including different prompting and decoding strategies, and adversarial attacks. We propose a more robust method that incorporates abstract elements, such as event transitions, as key deciding factors to detect machine versus human texts by training a latent-space model on sequences of events or topics derived from human-written texts. In three different domains, machine-generated texts, which are originally inseparable from human texts on the token level, can be better distinguished with our latent-space model, leading to a 31% improvement over strong baselines such as DetectGPT. Our analysis further reveals that, unlike humans, modern LLMs like GPT-4 generate event triggers and their transitions differently, an inherent disparity that helps our method to robustly detect machine-generated texts.

摘要: 大型语言模型(LLM)生成流畅的长文本的能力日益增强，这对区分机器生成的输出和人类书写的输出提出了新的挑战，这对确保表达的真实性和可信度至关重要。现有的零射击检测器主要集中在令牌级分发上，这些分发容易受到现实世界域转换的影响，包括不同的提示和解码策略，以及敌意攻击。我们提出了一种更健壮的方法，通过对来自人类书写的文本的事件或主题序列训练潜在空间模型，将事件转移等抽象元素作为关键决定因素来检测机器文本与人类文本。在三个不同的领域中，机器生成的文本在标记级别上与人类文本密不可分，使用我们的潜在空间模型可以更好地区分它们，导致比DetectGPT等强基线提高31%。我们的分析进一步表明，与人类不同的是，像GPT-4这样的现代LLM以不同的方式生成事件触发器及其转换，这一固有的差异有助于我们的方法稳健地检测机器生成的文本。



## **11. RAFT: Realistic Attacks to Fool Text Detectors**

RAFT：愚弄文本检测器的现实攻击 cs.CL

Accepted by EMNLP 2024

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03658v1) [paper-pdf](http://arxiv.org/pdf/2410.03658v1)

**Authors**: James Wang, Ran Li, Junfeng Yang, Chengzhi Mao

**Abstract**: Large language models (LLMs) have exhibited remarkable fluency across various tasks. However, their unethical applications, such as disseminating disinformation, have become a growing concern. Although recent works have proposed a number of LLM detection methods, their robustness and reliability remain unclear. In this paper, we present RAFT: a grammar error-free black-box attack against existing LLM detectors. In contrast to previous attacks for language models, our method exploits the transferability of LLM embeddings at the word-level while preserving the original text quality. We leverage an auxiliary embedding to greedily select candidate words to perturb against the target detector. Experiments reveal that our attack effectively compromises all detectors in the study across various domains by up to 99%, and are transferable across source models. Manual human evaluation studies show our attacks are realistic and indistinguishable from original human-written text. We also show that examples generated by RAFT can be used to train adversarially robust detectors. Our work shows that current LLM detectors are not adversarially robust, underscoring the urgent need for more resilient detection mechanisms.

摘要: 大型语言模型(LLM)在各种任务中表现出了惊人的流畅性。然而，它们不道德的应用，如传播虚假信息，已经成为一个日益令人担忧的问题。虽然最近的工作已经提出了一些LLM检测方法，但它们的稳健性和可靠性仍然不清楚。本文提出了一种针对现有LLM检测器的无语法错误的黑盒攻击方法RAFT。与以往对语言模型的攻击不同，我们的方法在保持原始文本质量的同时，利用了LLM嵌入在单词级别的可转移性。我们利用辅助嵌入来贪婪地选择候选单词来扰动目标检测器。实验表明，我们的攻击有效地危害了研究中跨不同域的所有检测器高达99%，并且可以跨源模型传输。人工人工评估研究表明，我们的攻击是真实的，与原始的人类书面文本没有什么区别。我们还表明，由RAFT生成的例子可以用于训练对抗性稳健的检测器。我们的工作表明，目前的LLM检测器并不具有相反的健壮性，这突显了对更具弹性的检测机制的迫切需要。



## **12. Buckle Up: Robustifying LLMs at Every Customization Stage via Data Curation**

系好安全带：通过数据修复在每个定制阶段对LLM进行优化 cs.CR

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.02220v2) [paper-pdf](http://arxiv.org/pdf/2410.02220v2)

**Authors**: Xiaoqun Liu, Jiacheng Liang, Luoxi Tang, Chenyu You, Muchao Ye, Zhaohan Xi

**Abstract**: Large language models (LLMs) are extensively adapted for downstream applications through a process known as "customization," with fine-tuning being a common method for integrating domain-specific expertise. However, recent studies have revealed a vulnerability that tuning LLMs with malicious samples can compromise their robustness and amplify harmful content, an attack known as "jailbreaking." To mitigate such attack, we propose an effective defensive framework utilizing data curation to revise commonsense texts and enhance their safety implication from the perspective of LLMs. The curated texts can mitigate jailbreaking attacks at every stage of the customization process: before customization to immunize LLMs against future jailbreak attempts, during customization to neutralize jailbreaking risks, or after customization to restore the compromised models. Since the curated data strengthens LLMs through the standard fine-tuning workflow, we do not introduce additional modules during LLM inference, thereby preserving the original customization process. Experimental results demonstrate a substantial reduction in jailbreaking effects, with up to a 100% success in generating responsible responses. Notably, our method is effective even with commonsense texts, which are often more readily available than safety-relevant data. With the every-stage defensive framework and supporting experimental performance, this work represents a significant advancement in mitigating jailbreaking risks and ensuring the secure customization of LLMs.

摘要: 大型语言模型(LLM)通过一种称为“定制”的过程广泛适用于下游应用程序，微调是集成特定领域专业知识的常见方法。然而，最近的研究揭示了一个漏洞，即用恶意样本调整LLM可能会损害它们的健壮性，并放大有害内容，这种攻击被称为“越狱”。为了缓解这种攻击，我们提出了一个有效的防御框架，利用数据精选来修改常识文本，并从LLMS的角度增强其安全含义。经过精选的文本可以在定制过程的每个阶段减少越狱攻击：在定制之前，以使LLM免受未来的越狱企图；在定制期间，以中和越狱风险；或在定制之后，以恢复受影响的模型。由于精选数据通过标准的微调工作流程加强了LLM，因此我们在LLM推理过程中不会引入额外的模块，从而保留了原始的定制流程。实验结果表明，越狱效果大大降低，生成负责任的响应的成功率高达100%。值得注意的是，我们的方法甚至对于常识性文本也是有效的，这些常识性文本通常比安全相关数据更容易获得。通过每个阶段的防御框架和支持的实验性能，这项工作在降低越狱风险和确保低成本管理系统的安全定制方面取得了重大进展。



## **13. Jailbreaking as a Reward Misspecification Problem**

越狱是奖励错误指定问题 cs.LG

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2406.14393v3) [paper-pdf](http://arxiv.org/pdf/2406.14393v3)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. This misspecification occurs when the reward function fails to accurately capture the intended behavior, leading to misaligned model outputs. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts in a reward-misspecified space. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark against various target aligned LLMs while preserving the human readability of the generated prompts. Furthermore, these attacks on open-source models demonstrate high transferability to closed-source models like GPT-4o and out-of-distribution tasks from HarmBench. Detailed analysis highlights the unique advantages of the proposed reward misspecification objective compared to previous methods, offering new insights for improving LLM safety and robustness.

摘要: 大型语言模型(LLM)的广泛采用引起了人们对它们的安全性和可靠性的担忧，特别是它们对对手攻击的脆弱性。在本文中，我们提出了一种新的观点，将该漏洞归因于对齐过程中的错误指定。当奖励函数未能准确捕获预期行为时，就会出现这种错误说明，从而导致模型输出不对齐。我们引入了一个度量指标ReGap来量化奖励错误指定的程度，并展示了它在检测有害后门提示方面的有效性和健壮性。在这些见解的基础上，我们提出了REMISTY，这是一个用于自动红色团队的系统，它在错误指定奖励的空间中生成对抗性提示。在保持生成提示的人类可读性的同时，针对各种目标对齐的LLM，在AdvBtch基准上实现了最先进的攻击成功率。此外，这些对开源模型的攻击表明，可以很好地转移到GPT-4o等封闭源代码模型和来自HarmBtch的非分发任务。详细的分析强调了与以前的方法相比，所提出的奖励误指定目标的独特优势，为提高LLM的安全性和稳健性提供了新的见解。



## **14. AutoPenBench: Benchmarking Generative Agents for Penetration Testing**

AutoPenBench：渗透测试生成剂的基准测试 cs.CR

Codes for the benchmark:  https://github.com/lucagioacchini/auto-pen-bench Codes for the paper  experiments: https://github.com/lucagioacchini/genai-pentest-paper

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03225v1) [paper-pdf](http://arxiv.org/pdf/2410.03225v1)

**Authors**: Luca Gioacchini, Marco Mellia, Idilio Drago, Alexander Delsanto, Giuseppe Siracusano, Roberto Bifulco

**Abstract**: Generative AI agents, software systems powered by Large Language Models (LLMs), are emerging as a promising approach to automate cybersecurity tasks. Among the others, penetration testing is a challenging field due to the task complexity and the diverse strategies to simulate cyber-attacks. Despite growing interest and initial studies in automating penetration testing with generative agents, there remains a significant gap in the form of a comprehensive and standard framework for their evaluation and development. This paper introduces AutoPenBench, an open benchmark for evaluating generative agents in automated penetration testing. We present a comprehensive framework that includes 33 tasks, each representing a vulnerable system that the agent has to attack. Tasks are of increasing difficulty levels, including in-vitro and real-world scenarios. We assess the agent performance with generic and specific milestones that allow us to compare results in a standardised manner and understand the limits of the agent under test. We show the benefits of AutoPenBench by testing two agent architectures: a fully autonomous and a semi-autonomous supporting human interaction. We compare their performance and limitations. For example, the fully autonomous agent performs unsatisfactorily achieving a 21% Success Rate (SR) across the benchmark, solving 27% of the simple tasks and only one real-world task. In contrast, the assisted agent demonstrates substantial improvements, with 64% of SR. AutoPenBench allows us also to observe how different LLMs like GPT-4o or OpenAI o1 impact the ability of the agents to complete the tasks. We believe that our benchmark fills the gap with a standard and flexible framework to compare penetration testing agents on a common ground. We hope to extend AutoPenBench along with the research community by making it available under https://github.com/lucagioacchini/auto-pen-bench.

摘要: 生成式人工智能代理是由大型语言模型(LLM)支持的软件系统，正在成为一种有前途的自动化网络安全任务的方法。其中，渗透测试是一个具有挑战性的领域，因为任务的复杂性和模拟网络攻击的策略多种多样。尽管人们对利用产生剂进行自动化渗透测试越来越感兴趣，并进行了初步研究，但在评估和开发产生剂的全面和标准框架的形式上，仍然存在着重大差距。本文介绍了一种用于评估自动渗透测试中的生成性代理的开放基准--AutoPenBch。我们提出了一个全面的框架，包括33个任务，每个任务代表代理必须攻击的易受攻击的系统。任务的难度越来越高，包括体外和真实世界的场景。我们用通用的和特定的里程碑来评估代理的性能，使我们能够以标准化的方式比较结果，并了解接受测试的代理的限制。我们通过测试两种代理体系结构：完全自主和半自主支持人类交互，展示了AutoPenB边的好处。我们比较了它们的性能和局限性。例如，完全自主的代理在基准测试中的成功率(SR)不令人满意地达到了21%，解决了27%的简单任务，而只解决了一个真实世界的任务。相比之下，辅助剂表现出显著的改善，获得了SR的5%。AutoPenB边还允许我们观察不同的LLM，如GPT-40或OpenAI o1，是如何影响代理完成任务的能力的。我们相信，我们的基准填补了这一空白，提供了一个标准和灵活的框架，可以在共同的基础上比较渗透测试试剂。我们希望通过使其在https://github.com/lucagioacchini/auto-pen-bench.下可用来与研究社区一起扩展AutoPenB边



## **15. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

SCA：高效语义一致的无限制对抗攻击 cs.CV

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.02240v2) [paper-pdf](http://arxiv.org/pdf/2410.02240v2)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our code can be found at https://github.com/Pan-Zihao/SCA.

摘要: 不受限制的对抗性攻击通常操纵图像的语义内容(例如，颜色或纹理)以创建既有效又逼真的对抗性示例。最近的工作利用扩散逆过程将图像映射到潜在空间，在潜在空间中通过引入扰动来操纵高级语义。然而，它们往往会在去噪输出中造成严重的语义扭曲，并导致效率低下。在这项研究中，我们提出了一种新的框架，称为语义一致的无限对抗攻击(SCA)，它使用一种反转方法来提取编辑友好的噪声映射，并利用多模式大语言模型(MLLM)在整个过程中提供语义指导。在MLLM提供丰富语义信息的条件下，使用一系列编辑友好的噪声图对每个步骤进行DDPM去噪处理，并利用DPM Solver++加速这一过程，从而实现高效的语义一致性采样。与现有的方法相比，我们的框架能够高效地生成对抗性的例子，这些例子表现出最小的可识别的语义变化。因此，我们首次引入了语义一致的对抗性例子(SCAE)。广泛的实验和可视化已经证明了SCA的高效率，特别是在平均速度上是最先进的攻击的12倍。我们的代码可以在https://github.com/Pan-Zihao/SCA.上找到



## **16. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

多语言机器生成文本检测中的作者混淆 cs.CL

Accepted to EMNLP 2024 Findings

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2401.07867v3) [paper-pdf](http://arxiv.org/pdf/2401.07867v3)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of recent Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause evasion of automated detection in all tested languages, where homoglyph attacks are especially successful. However, some of the AO methods severely damaged the text, making it no longer readable or easily recognizable by humans (e.g., changed language, weird characters).

摘要: 最近的大型语言模型(LLM)的高质量文本生成能力引起了人们对它们的滥用(例如，在大规模生成/传播虚假信息中)的担忧。机器生成文本(MGT)检测对于应对此类威胁非常重要。然而，它容易受到作者身份混淆(AO)方法的影响，例如转译，这可能导致MGTS逃避检测。到目前为止，这只在单一语言环境中进行了评估。因此，最近提出的多语言检测器的敏感性仍然未知。我们通过全面基准测试10种著名的AO方法的性能来填补这一空白，针对11种语言的MGT攻击37种MGT检测方法(即，10$\乘以$37$\乘以$11=4,070个组合)。我们还使用混淆文本来评估数据增强对对手健壮性的影响。结果表明，在所有被测语言中，所有被测试的声学方法都可以逃避自动检测，其中同形文字攻击尤其成功。然而，一些AO方法严重损坏了文本，使其不再可读或不再容易被人类识别(例如，改变语言、奇怪的字符)。



## **17. MoJE: Mixture of Jailbreak Experts, Naive Tabular Classifiers as Guard for Prompt Attacks**

MoJE：越狱专家、天真的表格分类器的混合体作为迅速攻击的警卫 cs.CR

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2409.17699v3) [paper-pdf](http://arxiv.org/pdf/2409.17699v3)

**Authors**: Giandomenico Cornacchia, Giulio Zizzo, Kieran Fraser, Muhammad Zaid Hameed, Ambrish Rawat, Mark Purcell

**Abstract**: The proliferation of Large Language Models (LLMs) in diverse applications underscores the pressing need for robust security measures to thwart potential jailbreak attacks. These attacks exploit vulnerabilities within LLMs, endanger data integrity and user privacy. Guardrails serve as crucial protective mechanisms against such threats, but existing models often fall short in terms of both detection accuracy, and computational efficiency. This paper advocates for the significance of jailbreak attack prevention on LLMs, and emphasises the role of input guardrails in safeguarding these models. We introduce MoJE (Mixture of Jailbreak Expert), a novel guardrail architecture designed to surpass current limitations in existing state-of-the-art guardrails. By employing simple linguistic statistical techniques, MoJE excels in detecting jailbreak attacks while maintaining minimal computational overhead during model inference. Through rigorous experimentation, MoJE demonstrates superior performance capable of detecting 90% of the attacks without compromising benign prompts, enhancing LLMs security against jailbreak attacks.

摘要: 大型语言模型(LLM)在各种应用中的激增突显了迫切需要强有力的安全措施来挫败潜在的越狱攻击。这些攻击利用LLMS中的漏洞，危及数据完整性和用户隐私。护栏是抵御此类威胁的关键保护机制，但现有模型在检测精度和计算效率方面往往存在不足。本文论述了防止越狱攻击对小武器系统的重要意义，并强调了输入护栏在保护这些模型中的作用。我们介绍了Moje(越狱专家的混合体)，这是一种新型的护栏架构，旨在超越现有最先进护栏的现有限制。通过使用简单的语言统计技术，Moje在检测越狱攻击方面表现出色，同时在模型推理过程中保持了最小的计算开销。通过严格的实验，Moje展示了卓越的性能，能够在不影响良性提示的情况下检测90%的攻击，增强了LLMS针对越狱攻击的安全性。



## **18. Can Watermarked LLMs be Identified by Users via Crafted Prompts?**

用户可以通过精心制作的脚本识别带水印的LLM吗？ cs.CR

25 pages, 5 figures, 8 tables

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03168v1) [paper-pdf](http://arxiv.org/pdf/2410.03168v1)

**Authors**: Aiwei Liu, Sheng Guan, Yiming Liu, Leyi Pan, Yifei Zhang, Liancheng Fang, Lijie Wen, Philip S. Yu, Xuming Hu

**Abstract**: Text watermarking for Large Language Models (LLMs) has made significant progress in detecting LLM outputs and preventing misuse. Current watermarking techniques offer high detectability, minimal impact on text quality, and robustness to text editing. However, current researches lack investigation into the imperceptibility of watermarking techniques in LLM services. This is crucial as LLM providers may not want to disclose the presence of watermarks in real-world scenarios, as it could reduce user willingness to use the service and make watermarks more vulnerable to attacks. This work is the first to investigate the imperceptibility of watermarked LLMs. We design an identification algorithm called Water-Probe that detects watermarks through well-designed prompts to the LLM. Our key motivation is that current watermarked LLMs expose consistent biases under the same watermark key, resulting in similar differences across prompts under different watermark keys. Experiments show that almost all mainstream watermarking algorithms are easily identified with our well-designed prompts, while Water-Probe demonstrates a minimal false positive rate for non-watermarked LLMs. Finally, we propose that the key to enhancing the imperceptibility of watermarked LLMs is to increase the randomness of watermark key selection. Based on this, we introduce the Water-Bag strategy, which significantly improves watermark imperceptibility by merging multiple watermark keys.

摘要: 针对大语言模型的文本水印技术在检测大语言模型输出和防止误用方面取得了显著进展。目前的水印技术提供了高可检测性，对文本质量的影响最小，以及对文本编辑的稳健性。然而，目前的研究缺乏对LLM服务中水印技术不可见性的研究。这一点至关重要，因为LLM提供商可能不想透露真实场景中是否存在水印，因为这可能会降低用户使用该服务的意愿，并使水印更容易受到攻击。这项工作是首次研究带水印的LLM的不可感知性。我们设计了一种名为Water-Probe的识别算法，该算法通过对LLM的精心设计的提示来检测水印。我们的关键动机是，当前的水印LLM暴露了相同水印密钥下的一致偏差，导致不同水印密钥下的提示存在相似的差异。实验表明，几乎所有的主流水印算法都能在我们精心设计的提示下很容易地识别出来，而Water-Probe算法对未加水印的LLMS具有最低的误检率。最后，提出了提高水印LLMS不可见性的关键是增加水印密钥选择的随机性。在此基础上，引入了水袋策略，通过合并多个水印密钥，显著提高了水印的不可见性。



## **19. Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs**

诡计袋：对LLM越狱攻击的基准 cs.CR

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2406.09324v2) [paper-pdf](http://arxiv.org/pdf/2406.09324v2)

**Authors**: Zhao Xu, Fan Liu, Hao Liu

**Abstract**: Although Large Language Models (LLMs) have demonstrated significant capabilities in executing complex tasks in a zero-shot manner, they are susceptible to jailbreak attacks and can be manipulated to produce harmful outputs. Recently, a growing body of research has categorized jailbreak attacks into token-level and prompt-level attacks. However, previous work primarily overlooks the diverse key factors of jailbreak attacks, with most studies concentrating on LLM vulnerabilities and lacking exploration of defense-enhanced LLMs. To address these issues, we evaluate the impact of various attack settings on LLM performance and provide a baseline benchmark for jailbreak attacks, encouraging the adoption of a standardized evaluation framework. Specifically, we evaluate the eight key factors of implementing jailbreak attacks on LLMs from both target-level and attack-level perspectives. We further conduct seven representative jailbreak attacks on six defense methods across two widely used datasets, encompassing approximately 354 experiments with about 55,000 GPU hours on A800-80G. Our experimental results highlight the need for standardized benchmarking to evaluate these attacks on defense-enhanced LLMs. Our code is available at https://github.com/usail-hkust/Bag_of_Tricks_for_LLM_Jailbreaking.

摘要: 尽管大型语言模型(LLM)在以零射击方式执行复杂任务方面表现出了巨大的能力，但它们很容易受到越狱攻击，并可能被操纵以产生有害的输出。最近，越来越多的研究将越狱攻击分为令牌级攻击和提示级攻击。然而，以前的工作主要忽略了越狱攻击的各种关键因素，大多数研究集中在LLM漏洞上，而缺乏对增强防御的LLM的探索。为了解决这些问题，我们评估了各种攻击设置对LLM性能的影响，并提供了越狱攻击的基准，鼓励采用标准化的评估框架。具体地，我们从目标级和攻击级两个角度评估了对LLMS实施越狱攻击的八个关键因素。我们进一步在两个广泛使用的数据集上对六种防御方法进行了七次有代表性的越狱攻击，在A800-80G上进行了大约354次实验，大约55,000个GPU小时。我们的实验结果强调了标准化基准测试的必要性，以评估这些针对防御增强型LLM的攻击。我们的代码可以在https://github.com/usail-hkust/Bag_of_Tricks_for_LLM_Jailbreaking.上找到



## **20. GoldCoin: Grounding Large Language Models in Privacy Laws via Contextual Integrity Theory**

金币：通过上下文完整性理论将大型语言模型作为隐私法的基础 cs.CL

Accepted by EMNLP 2024

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2406.11149v2) [paper-pdf](http://arxiv.org/pdf/2406.11149v2)

**Authors**: Wei Fan, Haoran Li, Zheye Deng, Weiqi Wang, Yangqiu Song

**Abstract**: Privacy issues arise prominently during the inappropriate transmission of information between entities. Existing research primarily studies privacy by exploring various privacy attacks, defenses, and evaluations within narrowly predefined patterns, while neglecting that privacy is not an isolated, context-free concept limited to traditionally sensitive data (e.g., social security numbers), but intertwined with intricate social contexts that complicate the identification and analysis of potential privacy violations. The advent of Large Language Models (LLMs) offers unprecedented opportunities for incorporating the nuanced scenarios outlined in privacy laws to tackle these complex privacy issues. However, the scarcity of open-source relevant case studies restricts the efficiency of LLMs in aligning with specific legal statutes. To address this challenge, we introduce a novel framework, GoldCoin, designed to efficiently ground LLMs in privacy laws for judicial assessing privacy violations. Our framework leverages the theory of contextual integrity as a bridge, creating numerous synthetic scenarios grounded in relevant privacy statutes (e.g., HIPAA), to assist LLMs in comprehending the complex contexts for identifying privacy risks in the real world. Extensive experimental results demonstrate that GoldCoin markedly enhances LLMs' capabilities in recognizing privacy risks across real court cases, surpassing the baselines on different judicial tasks.

摘要: 隐私问题突出地出现在实体之间不适当的信息传输过程中。现有的研究主要是通过在狭隘的预定义模式中探索各种隐私攻击、防御和评估来研究隐私，而忽略了隐私不是一个孤立的、与上下文无关的概念，仅限于传统的敏感数据(例如，社会安全号码)，而是与错综复杂的社会背景交织在一起，这使得识别和分析潜在的隐私侵犯变得复杂。大型语言模型(LLM)的出现为纳入隐私法中概述的细微差别场景提供了前所未有的机会，以解决这些复杂的隐私问题。然而，开源相关案例研究的匮乏限制了LLMS与具体法律法规保持一致的效率。为了应对这一挑战，我们引入了一个新的框架，GoldCoin，旨在有效地将LLM置于隐私法中，用于司法评估隐私侵权行为。我们的框架利用上下文完整性理论作为桥梁，创建基于相关隐私法规(例如HIPAA)的大量合成场景，以帮助LLMS理解复杂的上下文以识别现实世界中的隐私风险。广泛的实验结果表明，GoldCoin显著增强了LLMS在真实法庭案件中识别隐私风险的能力，超过了不同司法任务的基线。



## **21. Safeguard is a Double-edged Sword: Denial-of-service Attack on Large Language Models**

保障是一把双刃剑：对大型语言模型的拒绝服务攻击 cs.CR

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02916v1) [paper-pdf](http://arxiv.org/pdf/2410.02916v1)

**Authors**: Qingzhao Zhang, Ziyang Xiong, Z. Morley Mao

**Abstract**: Safety is a paramount concern of large language models (LLMs) in their open deployment. To this end, safeguard methods aim to enforce the ethical and responsible use of LLMs through safety alignment or guardrail mechanisms. However, we found that the malicious attackers could exploit false positives of safeguards, i.e., fooling the safeguard model to block safe content mistakenly, leading to a new denial-of-service (DoS) attack on LLMs. Specifically, by software or phishing attacks on user client software, attackers insert a short, seemingly innocuous adversarial prompt into to user prompt templates in configuration files; thus, this prompt appears in final user requests without visibility in the user interface and is not trivial to identify. By designing an optimization process that utilizes gradient and attention information, our attack can automatically generate seemingly safe adversarial prompts, approximately only 30 characters long, that universally block over 97\% of user requests on Llama Guard 3. The attack presents a new dimension of evaluating LLM safeguards focusing on false positives, fundamentally different from the classic jailbreak.

摘要: 安全是大型语言模型(LLM)在开放部署时最关心的问题。为此，保障措施旨在通过安全调整或护栏机制，强制以合乎道德和负责任的方式使用LLMS。然而，我们发现恶意攻击者可以利用安全措施的误报，即欺骗安全措施模型错误地阻止安全内容，从而导致对LLMS的新的拒绝服务(DoS)攻击。具体地说，通过软件或对用户客户端软件的网络钓鱼攻击，攻击者将一个看似无害的简短对抗性提示插入到配置文件中的用户提示模板中；因此，该提示出现在最终用户请求中，在用户界面中不可见，并且很难识别。通过设计一个利用梯度和注意力信息的优化过程，我们的攻击可以自动生成看似安全的敌意提示，大约只有30个字符，普遍阻止Llama Guard 3上超过97%的用户请求。该攻击提供了一个新的维度来评估LLM安全措施，从根本上不同于传统的越狱。



## **22. Universally Optimal Watermarking Schemes for LLMs: from Theory to Practice**

LLM的普遍最优水印方案：从理论到实践 cs.CR

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02890v1) [paper-pdf](http://arxiv.org/pdf/2410.02890v1)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Large Language Models (LLMs) boosts human efficiency but also poses misuse risks, with watermarking serving as a reliable method to differentiate AI-generated content from human-created text. In this work, we propose a novel theoretical framework for watermarking LLMs. Particularly, we jointly optimize both the watermarking scheme and detector to maximize detection performance, while controlling the worst-case Type-I error and distortion in the watermarked text. Within our framework, we characterize the universally minimum Type-II error, showing a fundamental trade-off between detection performance and distortion. More importantly, we identify the optimal type of detectors and watermarking schemes. Building upon our theoretical analysis, we introduce a practical, model-agnostic and computationally efficient token-level watermarking algorithm that invokes a surrogate model and the Gumbel-max trick. Empirical results on Llama-13B and Mistral-8$\times$7B demonstrate the effectiveness of our method. Furthermore, we also explore how robustness can be integrated into our theoretical framework, which provides a foundation for designing future watermarking systems with improved resilience to adversarial attacks.

摘要: 大语言模型(LLM)提高了人类的效率，但也带来了滥用风险，水印是区分人工智能生成的内容和人类创建的文本的可靠方法。在这项工作中，我们提出了一种新的水印LLMS的理论框架。特别是，我们联合优化了水印方案和检测器以最大化检测性能，同时控制了最坏情况下的I类错误和水印文本中的失真。在我们的框架内，我们描述了普遍最小的第二类错误，显示了检测性能和失真之间的基本权衡。更重要的是，我们确定了检测器和水印方案的最佳类型。在理论分析的基础上，我们介绍了一种实用的、与模型无关的、计算高效的令牌级水印算法，该算法调用了代理模型和Gumbel-Max技巧。对Llama-13B和Mistral-8$乘以$70B的实验结果证明了该方法的有效性。此外，我们还探索了如何将稳健性融入到我们的理论框架中，这为设计未来具有更好的抗攻击能力的水印系统提供了基础。



## **23. Mitigating Dialogue Hallucination for Large Vision Language Models via Adversarial Instruction Tuning**

通过对抗性指令调优缓解大视野语言模型的对话幻觉 cs.CV

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2403.10492v3) [paper-pdf](http://arxiv.org/pdf/2403.10492v3)

**Authors**: Dongmin Park, Zhaofang Qian, Guangxing Han, Ser-Nam Lim

**Abstract**: Mitigating hallucinations of Large Vision Language Models,(LVLMs) is crucial to enhance their reliability for general-purpose assistants. This paper shows that such hallucinations of LVLMs can be significantly exacerbated by preceding user-system dialogues. To precisely measure this, we first present an evaluation benchmark by extending popular multi-modal benchmark datasets with prepended hallucinatory dialogues powered by our novel Adversarial Question Generator (AQG), which can automatically generate image-related yet adversarial dialogues by adopting adversarial attacks on LVLMs. On our benchmark, the zero-shot performance of state-of-the-art LVLMs drops significantly for both the VQA and Captioning tasks. Next, we further reveal this hallucination is mainly due to the prediction bias toward preceding dialogues rather than visual content. To reduce this bias, we propose Adversarial Instruction Tuning (AIT) that robustly fine-tunes LVLMs against hallucinatory dialogues. Extensive experiments show our proposed approach successfully reduces dialogue hallucination while maintaining performance.

摘要: 减轻大型视觉语言模型(LVLMS)的幻觉对于提高其对通用助理的可靠性至关重要。这篇论文表明，之前的用户-系统对话可以显著加剧LVLMS的这种幻觉。为了准确地衡量这一点，我们首先提出了一个评估基准，通过扩展流行的多模式基准数据集，在我们的新型对抗性问题生成器(AQG)的支持下，使用预先设定的幻觉对话，该生成器可以通过对LVLM进行对抗性攻击来自动生成与图像相关的对抗性对话。在我们的基准测试中，最先进的LVLMS在VQA和字幕任务中的零镜头性能都显著下降。接下来，我们进一步揭示这种幻觉主要是由于预测偏向于之前的对话而不是视觉内容。为了减少这种偏差，我们提出了对抗性指令调整(AIT)，它针对幻觉对话对LVLM进行强有力的微调。大量的实验表明，我们提出的方法在保持性能的同时成功地减少了对话幻觉。



## **24. Jailbreaking LLMs with Arabic Transliteration and Arabizi**

使用阿拉伯语拼音和Arabizi语越狱LLM cs.LG

Accepted by EMNLP 2024

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2406.18725v2) [paper-pdf](http://arxiv.org/pdf/2406.18725v2)

**Authors**: Mansour Al Ghanim, Saleh Almohaimeed, Mengxin Zheng, Yan Solihin, Qian Lou

**Abstract**: This study identifies the potential vulnerabilities of Large Language Models (LLMs) to 'jailbreak' attacks, specifically focusing on the Arabic language and its various forms. While most research has concentrated on English-based prompt manipulation, our investigation broadens the scope to investigate the Arabic language. We initially tested the AdvBench benchmark in Standardized Arabic, finding that even with prompt manipulation techniques like prefix injection, it was insufficient to provoke LLMs into generating unsafe content. However, when using Arabic transliteration and chatspeak (or arabizi), we found that unsafe content could be produced on platforms like OpenAI GPT-4 and Anthropic Claude 3 Sonnet. Our findings suggest that using Arabic and its various forms could expose information that might remain hidden, potentially increasing the risk of jailbreak attacks. We hypothesize that this exposure could be due to the model's learned connection to specific words, highlighting the need for more comprehensive safety training across all language forms.

摘要: 这项研究确定了大型语言模型(LLM)对‘越狱’攻击的潜在漏洞，特别是关注阿拉伯语及其各种形式。虽然大多数研究都集中在基于英语的即时操作上，但我们的调查扩大了对阿拉伯语的研究范围。我们最初用标准化的阿拉伯语测试了AdvBtch基准测试，发现即使使用前缀注入等快速操作技术，也不足以激发LLMS生成不安全的内容。然而，当使用阿拉伯语音译和聊天(或Arabizi)时，我们发现在OpenAI GPT-4和人类克劳德3十四行诗等平台上可能会产生不安全的内容。我们的发现表明，使用阿拉伯语及其各种形式可能会暴露可能仍然隐藏的信息，潜在地增加越狱攻击的风险。我们假设，这种接触可能是由于模型与特定单词的习得联系，强调了在所有语言形式中进行更全面的安全培训的必要性。



## **25. Immunization against harmful fine-tuning attacks**

免疫有害微调攻击 cs.CL

Published in EMNLP 2024

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2402.16382v2) [paper-pdf](http://arxiv.org/pdf/2402.16382v2)

**Authors**: Domenic Rosati, Jan Wehner, Kai Williams, Łukasz Bartoszcze, Jan Batzner, Hassan Sajjad, Frank Rudzicz

**Abstract**: Large Language Models (LLMs) are often trained with safety guards intended to prevent harmful text generation. However, such safety training can be removed by fine-tuning the LLM on harmful datasets. While this emerging threat (harmful fine-tuning attacks) has been characterized by previous work, there is little understanding of how we should proceed in constructing and validating defenses against these attacks especially in the case where defenders would not have control of the fine-tuning process. We introduce a formal framework based on the training budget of an attacker which we call "Immunization" conditions. Using a formal characterisation of the harmful fine-tuning problem, we provide a thorough description of what a successful defense must comprise of and establish a set of guidelines on how rigorous defense research that gives us confidence should proceed.

摘要: 大型语言模型（LLM）通常会接受安全警卫的训练，旨在防止有害文本生成。然而，可以通过微调有害数据集的LLM来删除此类安全培训。虽然之前的工作已经描述了这种新出现的威胁（有害的微调攻击），但人们对我们应该如何构建和验证针对这些攻击的防御系统知之甚少，尤其是在防御者无法控制微调过程的情况下。我们引入了一个基于攻击者训练预算的正式框架，我们称之为“免疫”条件。通过对有害微调问题的正式描述，我们对成功的防御必须包括哪些内容进行了彻底的描述，并制定了一套关于如何进行严格的防御研究的指导方针，以赋予我们信心。



## **26. Undesirable Memorization in Large Language Models: A Survey**

大型语言模型中不可取的并行化：一项调查 cs.CL

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02650v1) [paper-pdf](http://arxiv.org/pdf/2410.02650v1)

**Authors**: Ali Satvaty, Suzan Verberne, Fatih Turkmen

**Abstract**: While recent research increasingly showcases the remarkable capabilities of Large Language Models (LLMs), it's vital to confront their hidden pitfalls. Among these challenges, the issue of memorization stands out, posing significant ethical and legal risks. In this paper, we presents a Systematization of Knowledge (SoK) on the topic of memorization in LLMs. Memorization is the effect that a model tends to store and reproduce phrases or passages from the training data and has been shown to be the fundamental issue to various privacy and security attacks against LLMs.   We begin by providing an overview of the literature on the memorization, exploring it across five key dimensions: intentionality, degree, retrievability, abstraction, and transparency. Next, we discuss the metrics and methods used to measure memorization, followed by an analysis of the factors that contribute to memorization phenomenon. We then examine how memorization manifests itself in specific model architectures and explore strategies for mitigating these effects. We conclude our overview by identifying potential research topics for the near future: to develop methods for balancing performance and privacy in LLMs, and the analysis of memorization in specific contexts, including conversational agents, retrieval-augmented generation, multilingual language models, and diffusion language models.

摘要: 虽然最近的研究越来越多地展示了大型语言模型(LLM)的非凡能力，但面对它们隐藏的陷阱是至关重要的。在这些挑战中，记忆问题尤为突出，构成了重大的道德和法律风险。在这篇文章中，我们提出了一个关于学习记忆系统中记忆问题的知识系统化(SOK)。记忆是指模型倾向于从训练数据中存储和复制短语或段落的效果，已被证明是针对LLMS的各种隐私和安全攻击的基本问题。我们首先提供关于记忆的文献概述，从五个关键维度对其进行探索：意向性、程度、可检索性、抽象性和透明度。接下来，我们讨论了衡量记忆的指标和方法，并分析了影响记忆现象的因素。然后，我们研究记忆如何在特定的模型体系结构中表现出来，并探索减轻这些影响的策略。我们通过确定在不久的将来可能的研究主题来结束我们的概述：开发在LLMS中平衡性能和隐私的方法，以及在特定上下文中对记忆的分析，包括会话代理、检索-增强生成、多语言模型和扩散语言模型。



## **27. Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents**

代理安全工作台（ASB）：对基于LLM的代理中的攻击和防御进行形式化和基准化 cs.CR

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02644v1) [paper-pdf](http://arxiv.org/pdf/2410.02644v1)

**Authors**: Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, Yongfeng Zhang

**Abstract**: Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 23 different types of attack/defense methods, and 8 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, a mixed attack, and 10 corresponding defenses across 13 LLM backbones with nearly 90,000 testing cases in total. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. Our code can be found at https://github.com/agiresearch/ASB.

摘要: 尽管基于大型语言模型(LLM)的代理可以使用外部工具和内存机制来解决复杂的现实任务，但它们也可能引入关键的安全漏洞。然而，现有的文献并没有全面评估对基于LLM的代理的攻击和防御。为了解决这一问题，我们引入了代理安全平台(ASB)，这是一个全面的框架，旨在形式化、基准和评估基于LLM的代理的攻击和防御，包括10个场景(例如，电子商务、自动驾驶、金融)、10个针对场景的代理、400多个工具、23种不同类型的攻击/防御方法和8个评估指标。在ASB的基础上，我们对13个LLM主干上的10个快速注入攻击、一个内存中毒攻击、一个新颖的思维计划后门攻击、一个混合攻击和10个相应的防御进行了基准测试，总共有近90,000个测试用例。我们的测试结果揭示了代理操作的不同阶段的关键漏洞，包括系统提示、用户提示处理、工具使用和内存恢复，平均攻击成功率最高为84.30\%，但现有防御措施的有效性有限，揭示了社区在代理安全方面需要做的重要工作。我们的代码可以在https://github.com/agiresearch/ASB.上找到



## **28. BadRobot: Manipulating Embodied LLMs in the Physical World**

BadRobot：在物理世界中操纵被授权的LLM cs.CY

38 pages, 16 figures

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2407.20242v3) [paper-pdf](http://arxiv.org/pdf/2407.20242v3)

**Authors**: Hangtao Zhang, Chenyu Zhu, Xianlong Wang, Ziqi Zhou, Changgan Yin, Minghui Li, Lulu Xue, Yichen Wang, Shengshan Hu, Aishan Liu, Peijin Guo, Leo Yu Zhang

**Abstract**: Embodied AI represents systems where AI is integrated into physical entities, enabling them to perceive and interact with their surroundings. Large Language Model (LLM), which exhibits powerful language understanding abilities, has been extensively employed in embodied AI by facilitating sophisticated task planning. However, a critical safety issue remains overlooked: could these embodied LLMs perpetrate harmful behaviors? In response, we introduce BadRobot, a novel attack paradigm aiming to make embodied LLMs violate safety and ethical constraints through typical voice-based user-system interactions. Specifically, three vulnerabilities are exploited to achieve this type of attack: (i) manipulation of LLMs within robotic systems, (ii) misalignment between linguistic outputs and physical actions, and (iii) unintentional hazardous behaviors caused by world knowledge's flaws. Furthermore, we construct a benchmark of various malicious physical action queries to evaluate BadRobot's attack performance. Based on this benchmark, extensive experiments against existing prominent embodied LLM frameworks (e.g., Voxposer, Code as Policies, and ProgPrompt) demonstrate the effectiveness of our BadRobot. Warning: This paper contains harmful AI-generated language and aggressive actions.

摘要: 具体化人工智能代表了将人工智能集成到物理实体中的系统，使它们能够感知周围环境并与其交互。大语言模型(LLM)具有强大的语言理解能力，通过促进复杂的任务规划，已被广泛应用于嵌入式人工智能中。然而，一个关键的安全问题仍然被忽视：这些具体化的LLM是否会实施有害行为？作为回应，我们引入了BadRobot，这是一种新的攻击范例，旨在通过典型的基于语音的用户-系统交互来使具体化LLM违反安全和伦理约束。具体地说，利用三个漏洞来实现这种类型的攻击：(I)在机器人系统内操纵LLM，(Ii)语言输出和物理动作之间的不匹配，以及(Iii)由世界知识的缺陷造成的无意危险行为。此外，我们还构建了一个针对各种恶意物理动作查询的基准来评估BadRobot的攻击性能。在此基准测试的基础上，对现有的主流嵌入式LLM框架(如Voxposer、代码即策略和ProgPrompt)进行了广泛的实验，证明了BadRobot的有效性。警告：本文包含有害的人工智能生成的语言和攻击性行为。



## **29. Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems**

削减开支：基于LLM的多代理系统的经济通信管道 cs.MA

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02506v1) [paper-pdf](http://arxiv.org/pdf/2410.02506v1)

**Authors**: Guibin Zhang, Yanwei Yue, Zhixun Li, Sukwon Yun, Guancheng Wan, Kun Wang, Dawei Cheng, Jeffrey Xu Yu, Tianlong Chen

**Abstract**: Recent advancements in large language model (LLM)-powered agents have shown that collective intelligence can significantly outperform individual capabilities, largely attributed to the meticulously designed inter-agent communication topologies. Though impressive in performance, existing multi-agent pipelines inherently introduce substantial token overhead, as well as increased economic costs, which pose challenges for their large-scale deployments. In response to this challenge, we propose an economical, simple, and robust multi-agent communication framework, termed $\texttt{AgentPrune}$, which can seamlessly integrate into mainstream multi-agent systems and prunes redundant or even malicious communication messages. Technically, $\texttt{AgentPrune}$ is the first to identify and formally define the \textit{communication redundancy} issue present in current LLM-based multi-agent pipelines, and efficiently performs one-shot pruning on the spatial-temporal message-passing graph, yielding a token-economic and high-performing communication topology. Extensive experiments across six benchmarks demonstrate that $\texttt{AgentPrune}$ \textbf{(I)} achieves comparable results as state-of-the-art topologies at merely $\$5.6$ cost compared to their $\$43.7$, \textbf{(II)} integrates seamlessly into existing multi-agent frameworks with $28.1\%\sim72.8\%\downarrow$ token reduction, and \textbf{(III)} successfully defend against two types of agent-based adversarial attacks with $3.5\%\sim10.8\%\uparrow$ performance boost.

摘要: 大型语言模型(LLM)支持的代理的最新进展表明，集体智能可以显著超过个人能力，这在很大程度上要归功于精心设计的代理间通信拓扑。尽管性能令人印象深刻，但现有的多代理管道固有地引入了大量令牌开销，以及增加的经济成本，这对其大规模部署构成了挑战。为了应对这一挑战，我们提出了一个经济、简单、健壮的多智能体通信框架，称为$\exttt{AgentPrune}$，它可以无缝地集成到主流的多智能体系统中，并对冗余甚至恶意的通信消息进行剪枝。从技术上讲，$\exttt{AgentPrune}$是第一个识别和形式化定义当前基于LLM的多代理管道中存在的通信冗余问题的工具，它高效地对时空消息传递图执行一次剪枝，从而产生令牌经济的高性能通信拓扑。在六个基准测试上的广泛实验表明，$\exttt{AgentPrune}$\extbf{(I)}获得了与最先进的拓扑结构相当的结果，与其$\$43.7$相比，只需$5.6$；\extbf{(Ii)}无缝集成到现有的多代理框架中，令牌减少28.1\\sim72.8\%\向下箭头$，并且通过$3.5\%\sim10.8\%\uparrow$性能提升，成功防御两种类型的基于代理的对手攻击。



## **30. Demonstration Attack against In-Context Learning for Code Intelligence**

针对代码智能的上下文学习的演示攻击 cs.CR

17 pages, 5 figures

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02841v1) [paper-pdf](http://arxiv.org/pdf/2410.02841v1)

**Authors**: Yifei Ge, Weisong Sun, Yihang Lou, Chunrong Fang, Yiran Zhang, Yiming Li, Xiaofang Zhang, Yang Liu, Zhihong Zhao, Zhenyu Chen

**Abstract**: Recent advancements in large language models (LLMs) have revolutionized code intelligence by improving programming productivity and alleviating challenges faced by software developers. To further improve the performance of LLMs on specific code intelligence tasks and reduce training costs, researchers reveal a new capability of LLMs: in-context learning (ICL). ICL allows LLMs to learn from a few demonstrations within a specific context, achieving impressive results without parameter updating. However, the rise of ICL introduces new security vulnerabilities in the code intelligence field. In this paper, we explore a novel security scenario based on the ICL paradigm, where attackers act as third-party ICL agencies and provide users with bad ICL content to mislead LLMs outputs in code intelligence tasks. Our study demonstrates the feasibility and risks of such a scenario, revealing how attackers can leverage malicious demonstrations to construct bad ICL content and induce LLMs to produce incorrect outputs, posing significant threats to system security. We propose a novel method to construct bad ICL content called DICE, which is composed of two stages: Demonstration Selection and Bad ICL Construction, constructing targeted bad ICL content based on the user query and transferable across different query inputs. Ultimately, our findings emphasize the critical importance of securing ICL mechanisms to protect code intelligence systems from adversarial manipulation.

摘要: 大型语言模型(LLM)的最新进展通过提高编程效率和减轻软件开发人员面临的挑战，使代码智能发生了革命性的变化。为了进一步提高LLMS在特定代码智能任务中的性能，降低培训成本，研究人员揭示了LLMS的一种新功能：情境学习(ICL)。ICL允许LLM从特定环境中的几个演示中学习，在不更新参数的情况下取得了令人印象深刻的结果。然而，ICL的兴起在代码情报领域引入了新的安全漏洞。在本文中，我们探索了一种新的基于ICL范式的安全场景，攻击者充当第三方ICL机构，向用户提供不良ICL内容，以在代码情报任务中误导LLMS输出。我们的研究论证了这种情况的可行性和风险，揭示了攻击者如何利用恶意演示来构建不良ICL内容并诱导LLMS产生错误的输出，从而对系统安全构成严重威胁。我们提出了一种构建不良ICL内容的新方法DICE，该方法分为演示选择和不良ICL构建两个阶段，基于用户查询构建具有针对性的不良ICL内容，并可在不同的查询输入之间传输。最后，我们的发现强调了保护ICL机制以保护代码情报系统免受对手操纵的关键重要性。



## **31. Optimizing Adaptive Attacks against Content Watermarks for Language Models**

优化针对语言模型内容水印的自适应攻击 cs.CR

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02440v1) [paper-pdf](http://arxiv.org/pdf/2410.02440v1)

**Authors**: Abdulrahman Diaa, Toluwani Aremu, Nils Lukas

**Abstract**: Large Language Models (LLMs) can be \emph{misused} to spread online spam and misinformation. Content watermarking deters misuse by hiding a message in model-generated outputs, enabling their detection using a secret watermarking key. Robustness is a core security property, stating that evading detection requires (significant) degradation of the content's quality. Many LLM watermarking methods have been proposed, but robustness is tested only against \emph{non-adaptive} attackers who lack knowledge of the watermarking method and can find only suboptimal attacks. We formulate the robustness of LLM watermarking as an objective function and propose preference-based optimization to tune \emph{adaptive} attacks against the specific watermarking method. Our evaluation shows that (i) adaptive attacks substantially outperform non-adaptive baselines. (ii) Even in a non-adaptive setting, adaptive attacks optimized against a few known watermarks remain highly effective when tested against other unseen watermarks, and (iii) optimization-based attacks are practical and require less than seven GPU hours. Our findings underscore the need to test robustness against adaptive attackers.

摘要: 大型语言模型(LLM)可能会被滥用来传播在线垃圾邮件和错误信息。内容水印通过在模型生成的输出中隐藏消息来阻止误用，从而能够使用秘密水印密钥来检测它们。健壮性是一项核心安全属性，它指出，逃避检测需要(显著)降低内容质量。已有许多LLM水印方法被提出，但健壮性仅针对缺乏水印方法知识且只能发现次优攻击的非自适应攻击者进行测试。我们将LLM水印的稳健性作为目标函数，并提出了基于偏好的优化方法来调整针对特定水印方法的攻击。我们的评估表明：(I)自适应攻击的性能大大优于非自适应基线。(Ii)即使在非自适应设置下，针对少数已知水印进行优化的自适应攻击在针对其他不可见水印进行测试时仍然非常有效，以及(Iii)基于优化的攻击是实用的，只需要不到7个GPU小时。我们的发现强调了测试针对适应性攻击者的健壮性的必要性。



## **32. Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models**

越狱解药：通过大型语言模型中的稀疏表示调整来实现安全与效用平衡 cs.CR

10 pages, 5 figures

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.02298v2) [paper-pdf](http://arxiv.org/pdf/2410.02298v2)

**Authors**: Guobin Shen, Dongcheng Zhao, Yiting Dong, Xiang He, Yi Zeng

**Abstract**: As large language models (LLMs) become integral to various applications, ensuring both their safety and utility is paramount. Jailbreak attacks, which manipulate LLMs into generating harmful content, pose significant challenges to this balance. Existing defenses, such as prompt engineering and safety fine-tuning, often introduce computational overhead, increase inference latency, and lack runtime flexibility. Moreover, overly restrictive safety measures can degrade model utility by causing refusals of benign queries. In this paper, we introduce Jailbreak Antidote, a method that enables real-time adjustment of LLM safety preferences by manipulating a sparse subset of the model's internal states during inference. By shifting the model's hidden representations along a safety direction with varying strengths, we achieve flexible control over the safety-utility balance without additional token overhead or inference delays. Our analysis reveals that safety-related information in LLMs is sparsely distributed; adjusting approximately 5% of the internal state is as effective as modifying the entire state. Extensive experiments on nine LLMs (ranging from 2 billion to 72 billion parameters), evaluated against ten jailbreak attack methods and compared with six defense strategies, validate the effectiveness and efficiency of our approach. By directly manipulating internal states during reasoning, Jailbreak Antidote offers a lightweight, scalable solution that enhances LLM safety while preserving utility, opening new possibilities for real-time safety mechanisms in widely-deployed AI systems.

摘要: 随着大型语言模型(LLM)成为各种应用程序不可或缺的一部分，确保它们的安全性和实用性是至关重要的。越狱攻击操纵LLM生成有害内容，对这种平衡构成了重大挑战。现有的防御措施，如即时工程和安全微调，通常会引入计算开销，增加推理延迟，并且缺乏运行时灵活性。此外，过于严格的安全措施可能会导致良性查询被拒绝，从而降低模型的实用性。在本文中，我们介绍了JailBreak解毒剂，这是一种通过在推理过程中操纵模型内部状态的稀疏子集来实时调整LLM安全偏好的方法。通过沿不同强度的安全方向移动模型的隐藏表示，我们在不增加令牌开销或推理延迟的情况下实现了对安全-效用平衡的灵活控制。我们的分析表明，LLMS中与安全相关的信息是稀疏分布的；调整大约5%的内部状态与修改整个状态一样有效。在9个LLM(参数从20亿到720亿)上进行了大量的实验，对10种越狱攻击方法进行了评估，并与6种防御策略进行了比较，验证了该方法的有效性和高效性。通过在推理过程中直接操纵内部状态，越狱解毒剂提供了一个轻量级、可扩展的解决方案，在增强LLM安全性的同时保留了实用性，为广泛部署的AI系统中的实时安全机制打开了新的可能性。



## **33. PathSeeker: Exploring LLM Security Vulnerabilities with a Reinforcement Learning-Based Jailbreak Approach**

PathSeeker：使用基于强化学习的越狱方法探索LLM安全漏洞 cs.CR

update the abstract and cite a new related work

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2409.14177v2) [paper-pdf](http://arxiv.org/pdf/2409.14177v2)

**Authors**: Zhihao Lin, Wei Ma, Mingyi Zhou, Yanjie Zhao, Haoyu Wang, Yang Liu, Jun Wang, Li Li

**Abstract**: In recent years, Large Language Models (LLMs) have gained widespread use, raising concerns about their security. Traditional jailbreak attacks, which often rely on the model internal information or have limitations when exploring the unsafe behavior of the victim model, limiting their reducing their general applicability. In this paper, we introduce PathSeeker, a novel black-box jailbreak method, which is inspired by the game of rats escaping a maze. We think that each LLM has its unique "security maze", and attackers attempt to find the exit learning from the received feedback and their accumulated experience to compromise the target LLM's security defences. Our approach leverages multi-agent reinforcement learning, where smaller models collaborate to guide the main LLM in performing mutation operations to achieve the attack objectives. By progressively modifying inputs based on the model's feedback, our system induces richer, harmful responses. During our manual attempts to perform jailbreak attacks, we found that the vocabulary of the response of the target model gradually became richer and eventually produced harmful responses. Based on the observation, we also introduce a reward mechanism that exploits the expansion of vocabulary richness in LLM responses to weaken security constraints. Our method outperforms five state-of-the-art attack techniques when tested across 13 commercial and open-source LLMs, achieving high attack success rates, especially in strongly aligned commercial models like GPT-4o-mini, Claude-3.5, and GLM-4-air with strong safety alignment. This study aims to improve the understanding of LLM security vulnerabilities and we hope that this sturdy can contribute to the development of more robust defenses.

摘要: 近年来，大型语言模型(LLM)得到了广泛的使用，这引发了人们对其安全性的担忧。传统的越狱攻击往往依赖模型的内部信息，或者在探索受害者模型的不安全行为时存在局限性，限制了它们的普遍适用性。在本文中，我们介绍了一种新的黑盒越狱方法--Path Seeker，它的灵感来自于老鼠逃离迷宫的游戏。我们认为，每个LLM都有自己独特的安全迷宫，攻击者试图从收到的反馈和他们积累的经验中找到出口学习，从而破坏目标LLM的安全防御。我们的方法利用多代理强化学习，其中较小的模型协作来指导主要的LLM执行突变操作，以实现攻击目标。通过根据模型的反馈逐步修改输入，我们的系统会产生更丰富、更有害的反应。在我们手动尝试执行越狱攻击的过程中，我们发现目标模型的响应词汇逐渐变得更加丰富，最终产生了有害的响应。在此基础上，我们还引入了一种奖励机制，该机制利用LLM响应中词汇丰富性的扩展来削弱安全约束。我们的方法在13个商业和开源LLM上进行测试时，性能超过了五种最先进的攻击技术，实现了高攻击成功率，特别是在GPT-40-mini、Claude-3.5和GLM-4-AIR等高度一致的商业型号上，具有很强的安全一致性。这项研究旨在提高对LLM安全漏洞的理解，我们希望这一强项可以为开发更强大的防御措施做出贡献。



## **34. DomainLynx: Leveraging Large Language Models for Enhanced Domain Squatting Detection**

DomainLynx：利用大型语言模型进行增强的域蹲位检测 cs.CR

Accepted for publication at IEEE CCNC 2025

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.02095v1) [paper-pdf](http://arxiv.org/pdf/2410.02095v1)

**Authors**: Daiki Chiba, Hiroki Nakano, Takashi Koide

**Abstract**: Domain squatting poses a significant threat to Internet security, with attackers employing increasingly sophisticated techniques. This study introduces DomainLynx, an innovative compound AI system leveraging Large Language Models (LLMs) for enhanced domain squatting detection. Unlike existing methods focusing on predefined patterns for top-ranked domains, DomainLynx excels in identifying novel squatting techniques and protecting less prominent brands. The system's architecture integrates advanced data processing, intelligent domain pairing, and LLM-powered threat assessment. Crucially, DomainLynx incorporates specialized components that mitigate LLM hallucinations, ensuring reliable and context-aware detection. This approach enables efficient analysis of vast security data from diverse sources, including Certificate Transparency logs, Passive DNS records, and zone files. Evaluated on a curated dataset of 1,649 squatting domains, DomainLynx achieved 94.7\% accuracy using Llama-3-70B. In a month-long real-world test, it detected 34,359 squatting domains from 2.09 million new domains, outperforming baseline methods by 2.5 times. This research advances Internet security by providing a versatile, accurate, and adaptable tool for combating evolving domain squatting threats. DomainLynx's approach paves the way for more robust, AI-driven cybersecurity solutions, enhancing protection for a broader range of online entities and contributing to a safer digital ecosystem.

摘要: 随着攻击者使用越来越复杂的技术，域名抢占对互联网安全构成了重大威胁。这项研究介绍了DomainLynx，这是一个创新的复合人工智能系统，利用大型语言模型(LLM)来增强域占用检测。与专注于排名靠前的域名的预定义模式的现有方法不同，DomainLynx在识别新颖的蹲守技术和保护不太知名的品牌方面表现出色。该系统的体系结构集成了先进的数据处理、智能域配对和LLM支持的威胁评估。至关重要的是，DomainLynx结合了专门的组件来缓解LLM幻觉，确保可靠和上下文感知的检测。这种方法可以有效地分析来自不同来源的大量安全数据，包括证书透明日志、被动DNS记录和区域文件。在1,649个蹲点域的精选数据集上进行评估，DomainLynx使用LLAMA-3-70B获得了94.7%的准确率。在一个月的实际测试中，它从209万个新域名中检测到34359个蹲点域名，比基线方法高出2.5倍。这项研究通过提供一种通用、准确和适应性强的工具来对抗不断变化的域名抢占威胁，从而促进了互联网安全。DomainLynx的方法为更强大的、人工智能驱动的网络安全解决方案铺平了道路，加强了对更广泛的在线实体的保护，并为更安全的数字生态系统做出了贡献。



## **35. Precision Knowledge Editing: Enhancing Safety in Large Language Models**

精确知识编辑：增强大型语言模型的安全性 cs.CL

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.03772v1) [paper-pdf](http://arxiv.org/pdf/2410.03772v1)

**Authors**: Xuying Li, Zhuo Li, Yuji Kosuga, Yasuhiro Yoshida, Victor Bian

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, but they also pose risks related to the generation of toxic or harmful content. This work introduces Precision Knowledge Editing (PKE), an advanced technique that builds upon existing knowledge editing methods to more effectively identify and modify toxic parameter regions within LLMs. By leveraging neuron weight tracking and activation pathway tracing, PKE achieves finer granularity in toxic content management compared to previous methods like Detoxifying Instance Neuron Modification (DINM). Our experiments demonstrate that PKE significantly reduces the attack success rate (ASR) across various models, including Llama2-7b and Llama-3-8b-instruct, while maintaining overall model performance. Additionally, we also compared the performance of some closed-source models (gpt-4-0613 and Claude 3 Sonnet) in our experiments, and found that models adjusted using our method far outperformed the closed-source models in terms of safety. This research contributes to the ongoing efforts to make LLMs safer and more reliable for real-world applications.

摘要: 大型语言模型(LLM)已显示出非凡的能力，但它们也带来了与生成有毒或有害内容相关的风险。这项工作引入了精确知识编辑(PKE)，这是一种先进的技术，它建立在现有的知识编辑方法的基础上，以更有效地识别和修改LLMS中的有毒参数区域。通过利用神经元重量跟踪和激活路径跟踪，PKE在有毒内容管理中实现了比以前的方法(如解毒实例神经元修改(DINM))更精细的粒度。我们的实验表明，PKE显著降低了包括Llama2-7b和Llama-3-8b-Indict在内的各种模型的攻击成功率(ASR)，同时保持了模型的整体性能。此外，我们还在实验中比较了几种闭源模型(GPT-4-0613和Claude 3 Sonnet)的性能，发现使用我们的方法调整的模型在安全性方面远远优于闭源模型。这项研究有助于不断努力使LLMS更安全、更可靠地适用于现实世界的应用。



## **36. TuBA: Cross-Lingual Transferability of Backdoor Attacks in LLMs with Instruction Tuning**

TuBA：具有指令调优的LLM后门攻击的跨语言可转移性 cs.CL

work in progress

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2404.19597v2) [paper-pdf](http://arxiv.org/pdf/2404.19597v2)

**Authors**: Xuanli He, Jun Wang, Qiongkai Xu, Pasquale Minervini, Pontus Stenetorp, Benjamin I. P. Rubinstein, Trevor Cohn

**Abstract**: The implications of backdoor attacks on English-centric large language models (LLMs) have been widely examined - such attacks can be achieved by embedding malicious behaviors during training and activated under specific conditions that trigger malicious outputs. Despite the increasing support for multilingual capabilities in open-source and proprietary LLMs, the impact of backdoor attacks on these systems remains largely under-explored. Our research focuses on cross-lingual backdoor attacks against multilingual LLMs, particularly investigating how poisoning the instruction-tuning data for one or two languages can affect the outputs for languages whose instruction-tuning data were not poisoned. Despite its simplicity, our empirical analysis reveals that our method exhibits remarkable efficacy in models like mT5 and GPT-4o, with high attack success rates, surpassing 90% in more than 7 out of 12 languages across various scenarios. Our findings also indicate that more powerful models show increased susceptibility to transferable cross-lingual backdoor attacks, which also applies to LLMs predominantly pre-trained on English data, such as Llama2, Llama3, and Gemma. Moreover, our experiments demonstrate 1) High Transferability: the backdoor mechanism operates successfully in cross-lingual response scenarios across 26 languages, achieving an average attack success rate of 99%, and 2) Robustness: the proposed attack remains effective even after defenses are applied. These findings expose critical security vulnerabilities in multilingual LLMs and highlight the urgent need for more robust, targeted defense strategies to address the unique challenges posed by cross-lingual backdoor transfer.

摘要: 后门攻击对以英语为中心的大型语言模型(LLM)的影响已被广泛研究-此类攻击可以通过在训练期间嵌入恶意行为来实现，并在触发恶意输出的特定条件下激活。尽管在开源和专有LLM中对多语言功能的支持越来越多，但后门攻击对这些系统的影响在很大程度上仍然没有得到充分的探索。我们的研究重点是针对多语言LLM的跨语言后门攻击，特别是调查毒化一到两种语言的指令调整数据如何影响那些指令调整数据没有中毒的语言的输出。尽管简单，但我们的经验分析表明，我们的方法在MT5和GPT-40等模型中表现出了显著的效果，在不同的场景下，在12种语言中的7种以上的攻击成功率超过90%。我们的发现还表明，更强大的模型显示出对可转移的跨语言后门攻击的易感性，这也适用于主要根据英语数据进行预训练的LLM，如Llama2、Llama3和Gema。此外，我们的实验表明：1)高可移植性：后门机制在26种语言的跨语言响应场景中成功运行，平均攻击成功率为99%；2)健壮性：所提出的攻击即使在实施防御后仍有效。这些发现暴露了多语言低成本管理中的关键安全漏洞，并突显了迫切需要更强大、更有针对性的防御战略，以应对跨语言后门转移带来的独特挑战。



## **37. Automated Red Teaming with GOAT: the Generative Offensive Agent Tester**

自动Red与GOAT合作：生成式进攻代理测试器 cs.LG

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.01606v1) [paper-pdf](http://arxiv.org/pdf/2410.01606v1)

**Authors**: Maya Pavlova, Erik Brinkman, Krithika Iyer, Vitor Albiero, Joanna Bitton, Hailey Nguyen, Joe Li, Cristian Canton Ferrer, Ivan Evtimov, Aaron Grattafiori

**Abstract**: Red teaming assesses how large language models (LLMs) can produce content that violates norms, policies, and rules set during their safety training. However, most existing automated methods in the literature are not representative of the way humans tend to interact with AI models. Common users of AI models may not have advanced knowledge of adversarial machine learning methods or access to model internals, and they do not spend a lot of time crafting a single highly effective adversarial prompt. Instead, they are likely to make use of techniques commonly shared online and exploit the multiturn conversational nature of LLMs. While manual testing addresses this gap, it is an inefficient and often expensive process. To address these limitations, we introduce the Generative Offensive Agent Tester (GOAT), an automated agentic red teaming system that simulates plain language adversarial conversations while leveraging multiple adversarial prompting techniques to identify vulnerabilities in LLMs. We instantiate GOAT with 7 red teaming attacks by prompting a general-purpose model in a way that encourages reasoning through the choices of methods available, the current target model's response, and the next steps. Our approach is designed to be extensible and efficient, allowing human testers to focus on exploring new areas of risk while automation covers the scaled adversarial stress-testing of known risk territory. We present the design and evaluation of GOAT, demonstrating its effectiveness in identifying vulnerabilities in state-of-the-art LLMs, with an ASR@10 of 97% against Llama 3.1 and 88% against GPT-4 on the JailbreakBench dataset.

摘要: 红色团队评估大型语言模型(LLM)在多大程度上可以产生违反其安全培训期间设定的规范、政策和规则的内容。然而，文献中的大多数现有自动化方法都不能代表人类与人工智能模型交互的方式。AI模型的普通用户可能没有对抗性机器学习方法的高级知识，也没有访问模型内部的权限，他们也不会花费大量时间来制作单个高效的对抗性提示。取而代之的是，他们可能会利用在线共享的常见技术，并利用LLMS的多轮对话性质。虽然手动测试弥补了这一差距，但它是一个低效且往往昂贵的过程。为了解决这些局限性，我们引入了生成性进攻代理Tester(山羊)，这是一个自动化的代理红色团队系统，它模拟普通语言的对抗性对话，同时利用多个对抗性提示技术来识别LLMS中的漏洞。我们用7个红色团队攻击实例化山羊，通过提示通用模型的方式，通过选择可用方法、当前目标模型的响应和下一步来鼓励推理。我们的方法被设计为可扩展和高效的，允许人工测试人员专注于探索新的风险领域，而自动化覆盖了已知风险领域的大规模对抗性压力测试。我们给出了山羊的设计和评估，展示了它在识别最新LLMS漏洞方面的有效性，在JailBreakB边数据集上，ASR@10对Llama 3.1的ASR为97%，对GPT-4的ASR为88%。



## **38. Leveraging the Context through Multi-Round Interactions for Jailbreaking Attacks**

通过多轮互动利用上下文进行越狱攻击 cs.LG

29 pages

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2402.09177v2) [paper-pdf](http://arxiv.org/pdf/2402.09177v2)

**Authors**: Yixin Cheng, Markos Georgopoulos, Volkan Cevher, Grigorios G. Chrysos

**Abstract**: Large Language Models (LLMs) are susceptible to Jailbreaking attacks, which aim to extract harmful information by subtly modifying the attack query. As defense mechanisms evolve, directly obtaining harmful information becomes increasingly challenging for Jailbreaking attacks. In this work, inspired from Chomsky's transformational-generative grammar theory and human practices of indirect context to elicit harmful information, we focus on a new attack form, called Contextual Interaction Attack. We contend that the prior context\u2014the information preceding the attack query\u2014plays a pivotal role in enabling strong Jailbreaking attacks. Specifically, we propose a first multi-turn approach that leverages benign preliminary questions to interact with the LLM. Due to the autoregressive nature of LLMs, which use previous conversation rounds as context during generation, we guide the model's question-response pair to construct a context that is semantically aligned with the attack query to execute the attack. We conduct experiments on seven different LLMs and demonstrate the efficacy of this attack, which is black-box and can also transfer across LLMs. We believe this can lead to further developments and understanding of security in LLMs.

摘要: 大型语言模型(LLM)容易受到越狱攻击，其目的是通过微妙地修改攻击查询来提取有害信息。随着防御机制的发展，直接获取有害信息对越狱攻击来说变得越来越具有挑战性。在这项工作中，受乔姆斯基的转换生成语法理论和人类间接语境获取有害信息的实践的启发，我们重点研究了一种新的攻击形式，称为语境交互攻击。我们认为，之前的上下文\u2014攻击查询之前的信息在启用强大的越狱攻击中起着关键作用。具体地说，我们提出了第一个利用良性初步问题与LLM互动的多轮方法。由于LLMS的自回归性质，它在生成过程中使用先前的对话回合作为上下文，我们引导模型的问题-回答对构建一个与攻击查询语义一致的上下文来执行攻击。我们在7个不同的LLM上进行了实验，证明了该攻击的有效性，该攻击是黑盒的，也可以跨LLM传输。我们认为，这可以进一步发展和了解小岛屿发展中国家的安全问题。



## **39. Backdooring Vision-Language Models with Out-Of-Distribution Data**

利用非分布数据进行后备视觉语言模型 cs.CV

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.01264v1) [paper-pdf](http://arxiv.org/pdf/2410.01264v1)

**Authors**: Weimin Lyu, Jiachen Yao, Saumya Gupta, Lu Pang, Tao Sun, Lingjie Yi, Lijie Hu, Haibin Ling, Chao Chen

**Abstract**: The emergence of Vision-Language Models (VLMs) represents a significant advancement in integrating computer vision with Large Language Models (LLMs) to generate detailed text descriptions from visual inputs. Despite their growing importance, the security of VLMs, particularly against backdoor attacks, is under explored. Moreover, prior works often assume attackers have access to the original training data, which is often unrealistic. In this paper, we address a more practical and challenging scenario where attackers must rely solely on Out-Of-Distribution (OOD) data. We introduce VLOOD (Backdooring Vision-Language Models with Out-of-Distribution Data), a novel approach with two key contributions: (1) demonstrating backdoor attacks on VLMs in complex image-to-text tasks while minimizing degradation of the original semantics under poisoned inputs, and (2) proposing innovative techniques for backdoor injection without requiring any access to the original training data. Our evaluation on image captioning and visual question answering (VQA) tasks confirms the effectiveness of VLOOD, revealing a critical security vulnerability in VLMs and laying the foundation for future research on securing multimodal models against sophisticated threats.

摘要: 视觉语言模型(VLMS)的出现代表了将计算机视觉与大型语言模型(LLM)相结合以从视觉输入生成详细的文本描述方面的重大进步。尽管它们的重要性与日俱增，但VLM的安全性，特别是针对后门攻击的安全性，仍处于探索之中。此外，以前的工作通常假设攻击者可以访问原始训练数据，这通常是不现实的。在本文中，我们将讨论一种更实用、更具挑战性的场景，在该场景中，攻击者必须完全依赖分发外(OOD)数据。我们介绍了VLOOD(Backdoors Vision-Language Models with Out-Of-Distributed Data)，这是一种新的方法，具有两个关键贡献：(1)展示了在复杂的图像到文本任务中对VLM的后门攻击，同时最小化了有毒输入下原始语义的退化；(2)提出了创新的后门注入技术，而不需要访问原始训练数据。我们对图像字幕和视觉问答(VQA)任务的评估证实了VLOOD的有效性，揭示了VLMS中的一个关键安全漏洞，并为未来保护多模式模型免受复杂威胁的研究奠定了基础。



## **40. Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference**

Logicbreaks：理解基于规则的推理颠覆的框架 cs.AI

**SubmitDate**: 2024-10-01    [abs](http://arxiv.org/abs/2407.00075v2) [paper-pdf](http://arxiv.org/pdf/2407.00075v2)

**Authors**: Anton Xue, Avishree Khare, Rajeev Alur, Surbhi Goel, Eric Wong

**Abstract**: We study how to subvert large language models (LLMs) from following prompt-specified rules. We model rule-following as inference in propositional Horn logic, a mathematical system in which rules have the form ``if $P$ and $Q$, then $R$'' for some propositions $P$, $Q$, and $R$. We prove that although LLMs can faithfully follow such rules, maliciously crafted prompts can mislead even idealized, theoretically constructed models. Empirically, we find that the reasoning behavior of LLMs aligns with that of our theoretical constructions, and popular attack algorithms find adversarial prompts with characteristics predicted by our theory. Our logic-based framework provides a novel perspective for mechanistically understanding the behavior of LLMs in rule-based settings such as jailbreak attacks.

摘要: 我们研究如何根据预算指定的规则颠覆大型语言模型（LLM）。我们将规则遵循建模为命题Horn逻辑中的推理，这是一个数学系统，其中规则的形式为“如果$P$和$Q$，那么$R $”，对于某些命题$P$、$Q$和$R$。我们证明，尽管LLM可以忠实地遵循这些规则，但恶意制作的提示甚至可以误导理想化的、理论上构建的模型。从经验上看，我们发现LLM的推理行为与我们的理论构建的推理行为一致，流行的攻击算法发现具有我们的理论预测特征的对抗性提示。我们基于逻辑的框架提供了一种新颖的视角，用于机械地理解LLM在越狱攻击等基于规则的环境中的行为。



## **41. Backdoor Attacks for LLMs with Weak-To-Strong Knowledge Distillation**

对具有弱到强知识蒸馏的LLM的后门攻击 cs.CR

**SubmitDate**: 2024-10-01    [abs](http://arxiv.org/abs/2409.17946v2) [paper-pdf](http://arxiv.org/pdf/2409.17946v2)

**Authors**: Shuai Zhao, Leilei Gan, Zhongliang Guo, Xiaobao Wu, Luwei Xiao, Xiaoyu Xu, Cong-Duy Nguyen, Luu Anh Tuan

**Abstract**: Despite being widely applied due to their exceptional capabilities, Large Language Models (LLMs) have been proven to be vulnerable to backdoor attacks. These attacks introduce targeted vulnerabilities into LLMs by poisoning training samples and full-parameter fine-tuning. However, this kind of backdoor attack is limited since they require significant computational resources, especially as the size of LLMs increases. Besides, parameter-efficient fine-tuning (PEFT) offers an alternative but the restricted parameter updating may impede the alignment of triggers with target labels. In this study, we first verify that backdoor attacks with PEFT may encounter challenges in achieving feasible performance. To address these issues and improve the effectiveness of backdoor attacks with PEFT, we propose a novel backdoor attack algorithm from weak to strong based on feature alignment-enhanced knowledge distillation (W2SAttack). Specifically, we poison small-scale language models through full-parameter fine-tuning to serve as the teacher model. The teacher model then covertly transfers the backdoor to the large-scale student model through feature alignment-enhanced knowledge distillation, which employs PEFT. Theoretical analysis reveals that W2SAttack has the potential to augment the effectiveness of backdoor attacks. We demonstrate the superior performance of W2SAttack on classification tasks across four language models, four backdoor attack algorithms, and two different architectures of teacher models. Experimental results indicate success rates close to 100% for backdoor attacks targeting PEFT.

摘要: 尽管大型语言模型(LLM)因其卓越的功能而得到广泛应用，但事实证明它们很容易受到后门攻击。这些攻击通过毒化训练样本和全参数微调将有针对性的漏洞引入LLMS。然而，这种后门攻击是有限的，因为它们需要大量的计算资源，特别是随着LLMS大小的增加。此外，参数高效微调(PEFT)提供了另一种选择，但受限的参数更新可能会阻碍触发器与目标标签的对准。在这项研究中，我们首先验证了使用PEFT的后门攻击在实现可行性能方面可能会遇到挑战。为了解决这些问题，提高PEFT后门攻击的有效性，提出了一种基于特征对齐增强知识提取的由弱到强的后门攻击算法(W2SAttack)。具体地说，我们通过全参数微调毒化小规模的语言模型作为教师模型。然后，教师模型通过使用PEFT的特征对齐增强的知识提炼，秘密地将后门转移到大规模学生模型。理论分析表明，W2SAttack具有增强后门攻击有效性的潜力。我们通过四种语言模型、四种后门攻击算法和两种不同的教师模型架构展示了W2SAttack在分类任务上的卓越性能。实验结果表明，针对PEFT的后门攻击成功率接近100%。



## **42. Universal Vulnerabilities in Large Language Models: Backdoor Attacks for In-context Learning**

大型语言模型中的普遍漏洞：上下文学习的后门攻击 cs.CL

**SubmitDate**: 2024-10-01    [abs](http://arxiv.org/abs/2401.05949v5) [paper-pdf](http://arxiv.org/pdf/2401.05949v5)

**Authors**: Shuai Zhao, Meihuizi Jia, Luu Anh Tuan, Fengjun Pan, Jinming Wen

**Abstract**: In-context learning, a paradigm bridging the gap between pre-training and fine-tuning, has demonstrated high efficacy in several NLP tasks, especially in few-shot settings. Despite being widely applied, in-context learning is vulnerable to malicious attacks. In this work, we raise security concerns regarding this paradigm. Our studies demonstrate that an attacker can manipulate the behavior of large language models by poisoning the demonstration context, without the need for fine-tuning the model. Specifically, we design a new backdoor attack method, named ICLAttack, to target large language models based on in-context learning. Our method encompasses two types of attacks: poisoning demonstration examples and poisoning demonstration prompts, which can make models behave in alignment with predefined intentions. ICLAttack does not require additional fine-tuning to implant a backdoor, thus preserving the model's generality. Furthermore, the poisoned examples are correctly labeled, enhancing the natural stealth of our attack method. Extensive experimental results across several language models, ranging in size from 1.3B to 180B parameters, demonstrate the effectiveness of our attack method, exemplified by a high average attack success rate of 95.0% across the three datasets on OPT models.

摘要: 情境学习是一种弥合预训练和微调之间差距的范式，在几个NLP任务中表现出了很高的效率，特别是在少数情况下。尽管情景学习被广泛应用，但它很容易受到恶意攻击。在这项工作中，我们提出了对此范式的安全担忧。我们的研究表明，攻击者可以通过毒化演示上下文来操纵大型语言模型的行为，而不需要对模型进行微调。具体地说，我们设计了一种新的后门攻击方法ICLAttack，用于基于上下文学习的大型语言模型。我们的方法包括两种类型的攻击：中毒演示示例和中毒演示提示，这可以使模型的行为与预定义的意图保持一致。ICLAttack不需要额外的微调来植入后门，从而保持了模型的通用性。此外，有毒的例子被正确地标记，增强了我们攻击方法的自然隐蔽性。在几个语言模型上的广泛实验结果，从1.3B到180B参数不等，证明了我们的攻击方法的有效性，例如在OPT模型上的三个数据集上的高平均攻击成功率为95.0%。



## **43. Privacy Evaluation Benchmarks for NLP Models**

NLP模型的隐私评估基准 cs.CL

Findings of EMNLP 2024

**SubmitDate**: 2024-10-01    [abs](http://arxiv.org/abs/2409.15868v3) [paper-pdf](http://arxiv.org/pdf/2409.15868v3)

**Authors**: Wei Huang, Yinggui Wang, Cen Chen

**Abstract**: By inducing privacy attacks on NLP models, attackers can obtain sensitive information such as training data and model parameters, etc. Although researchers have studied, in-depth, several kinds of attacks in NLP models, they are non-systematic analyses. It lacks a comprehensive understanding of the impact caused by the attacks. For example, we must consider which scenarios can apply to which attacks, what the common factors are that affect the performance of different attacks, the nature of the relationships between different attacks, and the influence of various datasets and models on the effectiveness of the attacks, etc. Therefore, we need a benchmark to holistically assess the privacy risks faced by NLP models. In this paper, we present a privacy attack and defense evaluation benchmark in the field of NLP, which includes the conventional/small models and large language models (LLMs). This benchmark supports a variety of models, datasets, and protocols, along with standardized modules for comprehensive evaluation of attacks and defense strategies. Based on the above framework, we present a study on the association between auxiliary data from different domains and the strength of privacy attacks. And we provide an improved attack method in this scenario with the help of Knowledge Distillation (KD). Furthermore, we propose a chained framework for privacy attacks. Allowing a practitioner to chain multiple attacks to achieve a higher-level attack objective. Based on this, we provide some defense and enhanced attack strategies. The code for reproducing the results can be found at https://github.com/user2311717757/nlp_doctor.

摘要: 通过诱导对NLP模型的隐私攻击，攻击者可以获得训练数据和模型参数等敏感信息。尽管研究人员对NLP模型中的几种攻击进行了深入的研究，但它们都是非系统的分析。它缺乏对袭击造成的影响的全面了解。例如，我们必须考虑哪些场景可以应用于哪些攻击，影响不同攻击性能的共同因素是什么，不同攻击之间关系的性质，以及各种数据集和模型对攻击有效性的影响等。因此，我们需要一个基准来全面评估NLP模型所面临的隐私风险。本文提出了一种针对自然语言处理领域的隐私攻防评估基准，包括常规/小模型和大语言模型。该基准支持各种模型、数据集和协议，以及用于全面评估攻击和防御策略的标准化模块。基于上述框架，我们提出了不同领域辅助数据之间的关联与隐私攻击强度的研究。在这种情况下，我们提出了一种改进的攻击方法--知识蒸馏(KD)。此外，我们还提出了一个针对隐私攻击的链式框架。允许实践者链接多次攻击以实现更高级别的攻击目标。在此基础上，提出了一些防御和增强攻击的策略。复制结果的代码可在https://github.com/user2311717757/nlp_doctor.上找到



## **44. Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey**

针对大型语言模型的有害微调攻击和防御：调查 cs.CR

**SubmitDate**: 2024-09-30    [abs](http://arxiv.org/abs/2409.18169v2) [paper-pdf](http://arxiv.org/pdf/2409.18169v2)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Recent research demonstrates that the nascent fine-tuning-as-a-service business model exposes serious safety concerns -- fine-tuning over a few harmful data uploaded by the users can compromise the safety alignment of the model. The attack, known as harmful fine-tuning, has raised a broad research interest among the community. However, as the attack is still new, \textbf{we observe from our miserable submission experience that there are general misunderstandings within the research community.} We in this paper aim to clear some common concerns for the attack setting, and formally establish the research problem. Specifically, we first present the threat model of the problem, and introduce the harmful fine-tuning attack and its variants. Then we systematically survey the existing literature on attacks/defenses/mechanical analysis of the problem. Finally, we outline future research directions that might contribute to the development of the field. Additionally, we present a list of questions of interest, which might be useful to refer to when reviewers in the peer review process question the realism of the experiment/attack/defense setting. A curated list of relevant papers is maintained and made accessible at: \url{https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers}.

摘要: 最近的研究表明，新兴的微调即服务商业模式暴露了严重的安全问题--对用户上传的几个有害数据进行微调可能会损害该模型的安全一致性。这一被称为有害微调的攻击在社区中引起了广泛的研究兴趣。然而，由于攻击仍然是新的，\extbf{我们从悲惨的提交经验中观察到，研究界普遍存在误解。}我们在本文中旨在澄清一些对攻击设置的共同关注，并正式确立研究问题。具体地说，我们首先给出了问题的威胁模型，并介绍了有害的微调攻击及其变体。然后，我们系统地综述了现有的关于攻击/防御/机械分析问题的文献。最后，我们概述了未来的研究方向，可能有助于该领域的发展。此外，我们提供了一个感兴趣的问题列表，当同行审查过程中的评审者质疑实验/攻击/防御设置的真实性时，这些问题可能会有用。相关论文的精选清单可在以下网址查阅：\url{https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers}.



## **45. Distract Large Language Models for Automatic Jailbreak Attack**

自动越狱攻击的分散大语言模型 cs.CR

EMNLP 2024

**SubmitDate**: 2024-09-30    [abs](http://arxiv.org/abs/2403.08424v2) [paper-pdf](http://arxiv.org/pdf/2403.08424v2)

**Authors**: Zeguan Xiao, Yan Yang, Guanhua Chen, Yun Chen

**Abstract**: Extensive efforts have been made before the public release of Large language models (LLMs) to align their behaviors with human values. However, even meticulously aligned LLMs remain vulnerable to malicious manipulations such as jailbreaking, leading to unintended behaviors. In this work, we propose a novel black-box jailbreak framework for automated red teaming of LLMs. We designed malicious content concealing and memory reframing with an iterative optimization algorithm to jailbreak LLMs, motivated by the research about the distractibility and over-confidence phenomenon of LLMs. Extensive experiments of jailbreaking both open-source and proprietary LLMs demonstrate the superiority of our framework in terms of effectiveness, scalability and transferability. We also evaluate the effectiveness of existing jailbreak defense methods against our attack and highlight the crucial need to develop more effective and practical defense strategies.

摘要: 在公开发布大型语言模型（LLM）之前，人们已经做出了广泛的努力，以使其行为与人类价值观保持一致。然而，即使是精心排列的LLM仍然容易受到越狱等恶意操纵，从而导致意外行为。在这项工作中，我们提出了一种新颖的黑匣子越狱框架，用于LLM的自动红色分组。受对LLM分心和过度自信现象的研究启发，我们设计了利用迭代优化算法来越狱LLM的恶意内容隐藏和内存重组。开源和专有LLM越狱的大量实验证明了我们的框架在有效性、可扩展性和可移植性方面的优越性。我们还评估了现有越狱防御方法针对我们攻击的有效性，并强调制定更有效和实用的防御策略的迫切需要。



## **46. ModelShield: Adaptive and Robust Watermark against Model Extraction Attack**

Model Shield：针对模型提取攻击的自适应鲁棒水印 cs.CR

**SubmitDate**: 2024-09-30    [abs](http://arxiv.org/abs/2405.02365v3) [paper-pdf](http://arxiv.org/pdf/2405.02365v3)

**Authors**: Kaiyi Pang, Tao Qi, Chuhan Wu, Minhao Bai, Minghu Jiang, Yongfeng Huang

**Abstract**: Large language models (LLMs) demonstrate general intelligence across a variety of machine learning tasks, thereby enhancing the commercial value of their intellectual property (IP). To protect this IP, model owners typically allow user access only in a black-box manner, however, adversaries can still utilize model extraction attacks to steal the model intelligence encoded in model generation. Watermarking technology offers a promising solution for defending against such attacks by embedding unique identifiers into the model-generated content. However, existing watermarking methods often compromise the quality of generated content due to heuristic alterations and lack robust mechanisms to counteract adversarial strategies, thus limiting their practicality in real-world scenarios. In this paper, we introduce an adaptive and robust watermarking method (named ModelShield) to protect the IP of LLMs. Our method incorporates a self-watermarking mechanism that allows LLMs to autonomously insert watermarks into their generated content to avoid the degradation of model content. We also propose a robust watermark detection mechanism capable of effectively identifying watermark signals under the interference of varying adversarial strategies. Besides, ModelShield is a plug-and-play method that does not require additional model training, enhancing its applicability in LLM deployments. Extensive evaluations on two real-world datasets and three LLMs demonstrate that our method surpasses existing methods in terms of defense effectiveness and robustness while significantly reducing the degradation of watermarking on the model-generated content.

摘要: 大型语言模型(LLM)在各种机器学习任务中展示了一般智能，从而提高了其知识产权(IP)的商业价值。为了保护这个IP，模型所有者通常只允许用户以黑盒方式访问，但是，攻击者仍然可以利用模型提取攻击来窃取模型生成中编码的模型情报。水印技术通过在模型生成的内容中嵌入唯一标识符，为防御此类攻击提供了一种很有前途的解决方案。然而，现有的水印方法往往会由于启发式修改而影响生成内容的质量，并且缺乏强大的机制来对抗对抗性策略，从而限制了它们在现实世界场景中的实用性。本文提出了一种自适应的稳健水印算法(ModelShield)来保护LLMS的IP地址。我们的方法结合了一种自水印机制，允许LLM自主地在其生成的内容中插入水印，以避免模型内容的降级。我们还提出了一种稳健的水印检测机制，能够在不同的对抗策略的干扰下有效地识别水印信号。此外，ModelShield是一种即插即用的方法，不需要额外的模型培训，增强了其在LLM部署中的适用性。在两个真实数据集和三个LLM上的广泛评估表明，我们的方法在防御有效性和稳健性方面优于现有方法，同时显着降低了水印对模型生成内容的退化。



## **47. Privacy in Large Language Models: Attacks, Defenses and Future Directions**

大型语言模型中的隐私：攻击、防御和未来方向 cs.CL

We upload the survey to cover more recent papers and inlcude privacy  resaearch on multi-modality

**SubmitDate**: 2024-09-30    [abs](http://arxiv.org/abs/2310.10383v2) [paper-pdf](http://arxiv.org/pdf/2310.10383v2)

**Authors**: Haoran Li, Yulin Chen, Jinglong Luo, Jiecong Wang, Hao Peng, Yan Kang, Xiaojin Zhang, Qi Hu, Chunkit Chan, Zenglin Xu, Bryan Hooi, Yangqiu Song

**Abstract**: The advancement of large language models (LLMs) has significantly enhanced the ability to effectively tackle various downstream NLP tasks and unify these tasks into generative pipelines. On the one hand, powerful language models, trained on massive textual data, have brought unparalleled accessibility and usability for both models and users. On the other hand, unrestricted access to these models can also introduce potential malicious and unintentional privacy risks. Despite ongoing efforts to address the safety and privacy concerns associated with LLMs, the problem remains unresolved. In this paper, we provide a comprehensive analysis of the current privacy attacks targeting LLMs and categorize them according to the adversary's assumed capabilities to shed light on the potential vulnerabilities present in LLMs. Then, we present a detailed overview of prominent defense strategies that have been developed to counter these privacy attacks. Beyond existing works, we identify upcoming privacy concerns as LLMs evolve. Lastly, we point out several potential avenues for future exploration.

摘要: 大型语言模型(LLM)的发展极大地增强了有效地处理各种下游NLP任务并将这些任务统一到生成管道中的能力。一方面，强大的语言模型，基于海量文本数据的训练，为模型和用户带来了无与伦比的可及性和可用性。另一方面，不受限制地访问这些模型也可能带来潜在的恶意和无意的隐私风险。尽管正在努力解决与低密度脂蛋白相关的安全和隐私问题，但这个问题仍然没有得到解决。在本文中，我们对当前针对LLMS的隐私攻击进行了全面的分析，并根据对手假设的能力对它们进行了分类，以揭示LLMS中存在的潜在漏洞。然后，我们详细概述了为应对这些隐私攻击而开发的主要防御策略。除了现有的工作，我们发现随着LLM的发展，即将到来的隐私问题。最后，我们指出了未来可能的几个探索方向。



## **48. Robust LLM safeguarding via refusal feature adversarial training**

通过拒绝功能对抗培训强大的LLM保障 cs.LG

**SubmitDate**: 2024-09-30    [abs](http://arxiv.org/abs/2409.20089v1) [paper-pdf](http://arxiv.org/pdf/2409.20089v1)

**Authors**: Lei Yu, Virginie Do, Karen Hambardzumyan, Nicola Cancedda

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can elicit harmful responses. Defending against such attacks remains challenging due to the opacity of jailbreaking mechanisms and the high computational cost of training LLMs robustly. We demonstrate that adversarial attacks share a universal mechanism for circumventing LLM safeguards that works by ablating a dimension in the residual stream embedding space called the refusal feature. We further show that the operation of refusal feature ablation (RFA) approximates the worst-case perturbation of offsetting model safety. Based on these findings, we propose Refusal Feature Adversarial Training (ReFAT), a novel algorithm that efficiently performs LLM adversarial training by simulating the effect of input-level attacks via RFA. Experiment results show that ReFAT significantly improves the robustness of three popular LLMs against a wide range of adversarial attacks, with considerably less computational overhead compared to existing adversarial training methods.

摘要: 大型语言模型(LLM)很容易受到可能引起有害响应的对抗性攻击。由于越狱机制的不透明性和强大训练LLM的高计算成本，防御此类攻击仍然具有挑战性。我们证明了对抗性攻击共享一个通用的机制来规避LLM安全机制，该机制通过在剩余流嵌入空间中消融一个称为拒绝特征的维度来工作。我们进一步证明了拒绝特征消融(RFA)的操作近似于补偿模型安全性的最坏情况的扰动。基于这些发现，我们提出了拒绝特征对抗训练(Refat)，这是一种通过RFA模拟输入级攻击的效果来高效执行LLM对抗训练的新算法。实验结果表明，与现有的对抗性训练方法相比，REFAT显著地提高了三种流行的LLMS对多种对抗性攻击的健壮性，并且具有相当少的计算开销。



## **49. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

早起的鸟抓住漏洞：揭开LLM服务系统中的计时侧通道 cs.CR

**SubmitDate**: 2024-09-30    [abs](http://arxiv.org/abs/2409.20002v1) [paper-pdf](http://arxiv.org/pdf/2409.20002v1)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.

摘要: 大型语言模型的广泛应用对其推理性能的优化提出了强烈的需求。目前用于此目的的技术主要集中在通过算法和硬件增强来减少延迟和提高吞吐量，而在很大程度上忽略了它们的隐私副作用，特别是在多用户环境中。在我们的研究中，我们首次在LLM系统中发现了一组新的计时侧通道，这些通道来自共享缓存和GPU内存分配，可以用来推断机密系统提示和其他用户发出的提示。这些漏洞呼应了在传统计算系统中观察到的安全挑战，突显出迫切需要解决LLM服务基础设施中潜在的信息泄漏问题。在本文中，我们报告了一种新的攻击策略，旨在利用LLM部署中固有的计时侧通道，特别是针对广泛用于提高LLM推理性能的键值(KV)缓存和语义缓存。我们的方法利用计时测量和分类模型来检测缓存命中，允许对手以高精度推断私人提示。我们还提出了一种逐令牌搜索算法来高效地恢复缓存中的共享提示前缀，证明了窃取系统提示和对等用户产生的提示的可行性。我们对流行的在线LLM服务进行黑盒测试的实验研究表明，这种隐私风险是完全现实的，具有显著的后果。我们的发现强调了强有力的缓解措施的必要性，以保护LLM系统免受此类新出现的威胁。



## **50. Mitigating Backdoor Threats to Large Language Models: Advancement and Challenges**

缓解对大型语言模型的后门威胁：进步和挑战 cs.CR

The 60th Annual Allerton Conference (Invited Paper). The arXiv  version is a pre-IEEE Press publication version

**SubmitDate**: 2024-09-30    [abs](http://arxiv.org/abs/2409.19993v1) [paper-pdf](http://arxiv.org/pdf/2409.19993v1)

**Authors**: Qin Liu, Wenjie Mo, Terry Tong, Jiashu Xu, Fei Wang, Chaowei Xiao, Muhao Chen

**Abstract**: The advancement of Large Language Models (LLMs) has significantly impacted various domains, including Web search, healthcare, and software development. However, as these models scale, they become more vulnerable to cybersecurity risks, particularly backdoor attacks. By exploiting the potent memorization capacity of LLMs, adversaries can easily inject backdoors into LLMs by manipulating a small portion of training data, leading to malicious behaviors in downstream applications whenever the hidden backdoor is activated by the pre-defined triggers. Moreover, emerging learning paradigms like instruction tuning and reinforcement learning from human feedback (RLHF) exacerbate these risks as they rely heavily on crowdsourced data and human feedback, which are not fully controlled. In this paper, we present a comprehensive survey of emerging backdoor threats to LLMs that appear during LLM development or inference, and cover recent advancement in both defense and detection strategies for mitigating backdoor threats to LLMs. We also outline key challenges in addressing these threats, highlighting areas for future research.

摘要: 大型语言模型(LLM)的发展对各个领域产生了重大影响，包括Web搜索、医疗保健和软件开发。然而，随着这些模型的扩展，它们变得更容易受到网络安全风险的影响，特别是后门攻击。通过利用LLMS的强大记忆能力，攻击者可以通过操纵一小部分训练数据轻松地向LLMS注入后门，每当隐藏的后门被预定义的触发器激活时，就会在下游应用程序中导致恶意行为。此外，教学调整和人类反馈强化学习(RLHF)等新兴学习范式加剧了这些风险，因为它们严重依赖众包数据和人类反馈，而这些数据和人类反馈并不完全受控制。在本文中，我们对在LLM开发或推理过程中出现的对LLM的新的后门威胁进行了全面的调查，并涵盖了在防御和检测策略方面的最新进展，以减轻对LLM的后门威胁。我们还概述了应对这些威胁的关键挑战，强调了未来研究的领域。



