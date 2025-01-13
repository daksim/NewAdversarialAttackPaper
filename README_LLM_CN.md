# Latest Large Language Model Attack Papers
**update at 2025-01-13 10:29:36**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. LUMIA: Linear probing for Unimodal and MultiModal Membership Inference Attacks leveraging internal LLM states**

LUMIA：利用内部LLM状态进行单模式和多模式成员资格推理攻击的线性探测 cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2411.19876v3) [paper-pdf](http://arxiv.org/pdf/2411.19876v3)

**Authors**: Luis Ibanez-Lissen, Lorena Gonzalez-Manzano, Jose Maria de Fuentes, Nicolas Anciaux, Joaquin Garcia-Alfaro

**Abstract**: Large Language Models (LLMs) are increasingly used in a variety of applications, but concerns around membership inference have grown in parallel. Previous efforts focus on black-to-grey-box models, thus neglecting the potential benefit from internal LLM information. To address this, we propose the use of Linear Probes (LPs) as a method to detect Membership Inference Attacks (MIAs) by examining internal activations of LLMs. Our approach, dubbed LUMIA, applies LPs layer-by-layer to get fine-grained data on the model inner workings. We test this method across several model architectures, sizes and datasets, including unimodal and multimodal tasks. In unimodal MIA, LUMIA achieves an average gain of 15.71 % in Area Under the Curve (AUC) over previous techniques. Remarkably, LUMIA reaches AUC>60% in 65.33% of cases -- an increment of 46.80% against the state of the art. Furthermore, our approach reveals key insights, such as the model layers where MIAs are most detectable. In multimodal models, LPs indicate that visual inputs can significantly contribute to detect MIAs -- AUC>60% is reached in 85.90% of experiments.

摘要: 大型语言模型(LLM)越来越多地用于各种应用程序，但围绕成员关系推理的关注也在平行增长。以往的研究主要集中在黑灰盒模型上，从而忽略了LLM内部信息的潜在益处。为了解决这一问题，我们提出使用线性探测器(LP)作为一种方法，通过检查LLP的内部激活来检测成员身份推理攻击(MIA)。我们的方法，称为Lumia，逐层应用LP，以获得关于模型内部工作的细粒度数据。我们在几个模型体系结构、大小和数据集上测试了这种方法，包括单模和多模任务。在单峰MIA中，Lumia的曲线下面积(AUC)比以前的技术平均增加了15.71%。值得注意的是，Lumia在65.33%的情况下达到AUC>60%--与最先进的水平相比增加了46.80%。此外，我们的方法揭示了关键的见解，例如最容易检测到MIA的模型层。在多通道模型中，LP表明视觉输入对检测MIA有显著贡献-85.90%的实验达到了60%以上的AUC。



## **2. Model Inversion in Split Learning for Personalized LLMs: New Insights from Information Bottleneck Theory**

个性化LLM分裂学习中的模型倒置：信息瓶颈理论的新见解 cs.LG

8 pages

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05965v1) [paper-pdf](http://arxiv.org/pdf/2501.05965v1)

**Authors**: Yunmeng Shu, Shaofeng Li, Tian Dong, Yan Meng, Haojin Zhu

**Abstract**: Personalized Large Language Models (LLMs) have become increasingly prevalent, showcasing the impressive capabilities of models like GPT-4. This trend has also catalyzed extensive research on deploying LLMs on mobile devices. Feasible approaches for such edge-cloud deployment include using split learning. However, previous research has largely overlooked the privacy leakage associated with intermediate representations transmitted from devices to servers. This work is the first to identify model inversion attacks in the split learning framework for LLMs, emphasizing the necessity of secure defense. For the first time, we introduce mutual information entropy to understand the information propagation of Transformer-based LLMs and assess privacy attack performance for LLM blocks. To address the issue of representations being sparser and containing less information than embeddings, we propose a two-stage attack system in which the first part projects representations into the embedding space, and the second part uses a generative model to recover text from these embeddings. This design breaks down the complexity and achieves attack scores of 38%-75% in various scenarios, with an over 60% improvement over the SOTA. This work comprehensively highlights the potential privacy risks during the deployment of personalized LLMs on the edge side.

摘要: 个性化的大型语言模型(LLM)已经变得越来越流行，展示了GPT-4等模型令人印象深刻的能力。这一趋势也催生了关于在移动设备上部署LLM的广泛研究。这种边缘云部署的可行方法包括使用分裂学习。然而，以前的研究在很大程度上忽略了与从设备传输到服务器的中间表示相关联的隐私泄露。这项工作首次在LLMS的分裂学习框架中发现了模型反转攻击，强调了安全防御的必要性。首次引入互信息熵来理解基于变压器的LLMS的信息传播，并评估LLM块的隐私攻击性能。为了解决表示比嵌入更稀疏和包含的信息量更少的问题，我们提出了一个两阶段攻击系统，其中第一部分将表示投影到嵌入空间中，第二部分使用生成模型从这些嵌入中恢复文本。这种设计打破了复杂性，在不同场景下的攻击得分达到38%-75%，比SOTA提高了60%以上。这项工作全面凸显了个性化LLMS在边缘侧部署过程中存在的潜在隐私风险。



## **3. Effective faking of verbal deception detection with target-aligned adversarial attacks**

通过目标对准的对抗攻击有效伪造言语欺骗检测 cs.CL

preprint

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05962v1) [paper-pdf](http://arxiv.org/pdf/2501.05962v1)

**Authors**: Bennett Kleinberg, Riccardo Loconte, Bruno Verschuere

**Abstract**: Background: Deception detection through analysing language is a promising avenue using both human judgments and automated machine learning judgments. For both forms of credibility assessment, automated adversarial attacks that rewrite deceptive statements to appear truthful pose a serious threat. Methods: We used a dataset of 243 truthful and 262 fabricated autobiographical stories in a deception detection task for humans and machine learning models. A large language model was tasked to rewrite deceptive statements so that they appear truthful. In Study 1, humans who made a deception judgment or used the detailedness heuristic and two machine learning models (a fine-tuned language model and a simple n-gram model) judged original or adversarial modifications of deceptive statements. In Study 2, we manipulated the target alignment of the modifications, i.e. tailoring the attack to whether the statements would be assessed by humans or computer models. Results: When adversarial modifications were aligned with their target, human (d=-0.07 and d=-0.04) and machine judgments (51% accuracy) dropped to the chance level. When the attack was not aligned with the target, both human heuristics judgments (d=0.30 and d=0.36) and machine learning predictions (63-78%) were significantly better than chance. Conclusions: Easily accessible language models can effectively help anyone fake deception detection efforts both by humans and machine learning models. Robustness against adversarial modifications for humans and machines depends on that target alignment. We close with suggestions on advancing deception research with adversarial attack designs.

摘要: 背景：通过分析语言进行欺骗检测是一种既使用人类判断又使用自动机器学习判断的有前途的方法。对于这两种形式的可信度评估来说，重写欺骗性陈述以使其看起来真实的自动对抗性攻击构成了严重威胁。方法：我们使用了243个真实的和262个编造的自传故事的数据集，在人类和机器学习模型的欺骗检测任务中。一个大型语言模型的任务是重写欺骗性的陈述，使它们看起来是真实的。在研究1中，做出欺骗性判断或使用细节启发式和两个机器学习模型(微调语言模型和简单n元语法模型)的人判断欺骗性陈述的原始修改或对抗性修改。在研究2中，我们操纵了修改的目标对齐，即根据陈述是否由人或计算机模型评估来量身定做攻击。结果：当对抗性修改与他们的目标一致时，人类(d=-0.07和d=-0.04)和机器判断(51%准确率)下降到机会水平。当攻击与目标不一致时，人类的启发式判断(d=0.30和d=0.36)和机器学习预测(63%-78%)都显著好于机会。结论：易于理解的语言模型可以有效地帮助任何人通过人类和机器学习模型进行虚假的欺骗检测。对人类和机器的敌意修改的健壮性取决于目标对齐。最后，我们建议用对抗性攻击设计来推进欺骗研究。



## **4. Image-based Multimodal Models as Intruders: Transferable Multimodal Attacks on Video-based MLLMs**

作为入侵者的基于图像的多模式模型：对基于视频的MLLM的可转移多模式攻击 cs.CV

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.01042v2) [paper-pdf](http://arxiv.org/pdf/2501.01042v2)

**Authors**: Linhao Huang, Xue Jiang, Zhiqiang Wang, Wentao Mo, Xi Xiao, Bo Han, Yongjie Yin, Feng Zheng

**Abstract**: Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models--a common and practical real world scenario--remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal model (IMM) as a surrogate model to craft adversarial video samples. Multimodal interactions and temporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. In addition, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as surrogate model) achieve competitive performance, with average attack success rates of 55.48% on MSVD-QA and 58.26% on MSRVTT-QA for VideoQA tasks, respectively. Our code will be released upon acceptance.

摘要: 基于视频的多通道大语言模型(V-MLLM)在视频-文本多通道任务中表现出对敌意例子的脆弱性。然而，对抗性视频是否可以转移到看不见的模型上--这是现实世界中常见和实用的场景--仍未得到探索。在本文中，我们率先对对抗性视频样本在V-MLLMS上的可转移性进行了研究。我们发现，现有的对抗性攻击方法在应用于V-MLLMS的黑盒环境时面临着很大的局限性，我们将其归因于以下缺点：(1)对扰动视频特征缺乏泛化；(2)只关注稀疏关键帧；(3)未能整合多模信息。为了解决这些限制并加深对黑盒场景中V-MLLM漏洞的理解，我们引入了图像到视频MLLM(I2V-MLLM)攻击。在I2V-MLLM中，我们使用基于图像的多模式模型(IMM)作为代理模型来制作对抗性视频样本。多模式交互和时间信息被集成以扰乱潜在空间内的视频表示，提高了对抗性转移。此外，还引入了扰动传播技术来处理不同的未知帧采样策略。实验结果表明，该方法能够在多个视频-文本多模式任务的不同V-MLLMS之间生成具有较强可转移性的对抗性实例。与这些模型上的白盒攻击相比，我们的黑盒攻击(以BLIP-2为代理模型)取得了与之相当的性能，对于视频QA任务，MSVD-QA和MSRVTT-QA的平均攻击成功率分别为55.48%和58.26%。我们的代码将在接受后发布。



## **5. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

BaThe：通过将有害指令视为后门触发来防御多模式大型语言模型中的越狱攻击 cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2408.09093v2) [paper-pdf](http://arxiv.org/pdf/2408.09093v2)

**Authors**: Yulin Chen, Haoran Li, Yirui Zhang, Zihao Zheng, Yangqiu Song, Bryan Hooi

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.

摘要: 多通道大型语言模型(MLLM)在各种多通道任务中表现出令人印象深刻的性能。另一方面，附加图像模式的集成可能允许恶意用户在图像中注入有害内容以越狱。与基于文本的LLMS不同，在LLMS中，攻击者需要使用特定的算法选择离散的令牌来隐藏其恶意意图，而图像信号的连续性为攻击者提供了直接注入有害意图的机会。在这项工作中，我们提出了一种简单而有效的越狱防御机制--$\extbf{bathe}$($\extbf{ba}$ck door$\extbf{T}$rigger S$\extbf{h}$i$\extbf{e}$ld)。我们的工作是基于生成式语言模型对越狱后门攻击和虚拟提示后门攻击的最新研究。越狱后门攻击使用有害指令和手动创建的字符串作为触发器，使后门模型生成被禁止的响应。我们假设有害指令可以作为触发器，如果我们将拒绝响应设置为触发响应，那么反向模型就可以防御越狱攻击。我们通过利用虚拟拒绝提示来实现这一点，类似于虚拟提示后门攻击。我们将虚拟拒绝提示嵌入到软文本嵌入中，我们称之为‘’楔形‘’。我们的综合实验表明，BAIT有效地缓解了各种类型的越狱攻击，并且能够自适应地防御看不见的攻击，对MLLMS的性能影响最小。



## **6. Safeguarding System Prompts for LLMs**

LLM的保护系统预算 cs.CR

15 pages, 5 figures, 2 tables

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2412.13426v2) [paper-pdf](http://arxiv.org/pdf/2412.13426v2)

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He

**Abstract**: Large language models (LLMs) are increasingly utilized in applications where system prompts, which guide model outputs, play a crucial role. These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we propose PromptKeeper, a robust defense mechanism designed to safeguard system prompts. PromptKeeper tackles two core challenges: reliably detecting prompt leakage and mitigating side-channel vulnerabilities when leakage occurs. By framing detection as a hypothesis-testing problem, PromptKeeper effectively identifies both explicit and subtle leakage. Upon detection, it regenerates responses using a dummy prompt, ensuring that outputs remain indistinguishable from typical interactions when no leakage is present. PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.

摘要: 大型语言模型(LLM)越来越多地被用于指导模型输出的系统提示发挥关键作用的应用中。这些提示通常包含业务逻辑和敏感信息，因此保护它们至关重要。但是，敌意的甚至常规的用户查询都可以利用LLM漏洞来暴露这些隐藏的提示。为了解决这个问题，我们提出了PromptKeeper，一种健壮的防御机制，旨在保护系统提示。PromptKeeper解决了两个核心挑战：可靠地检测及时泄漏和在发生泄漏时缓解侧通道漏洞。通过将检测框定为假设检验问题，PromptKeeper有效地识别了显性和细微的泄漏。一旦检测到，它会使用虚拟提示重新生成响应，确保在没有泄漏的情况下，输出与典型的交互没有区别。PromptKeeper通过对抗性或常规查询确保针对提示提取攻击的强大保护，同时在良性用户交互期间保持对话能力和运行效率。



## **7. RAG-WM: An Efficient Black-Box Watermarking Approach for Retrieval-Augmented Generation of Large Language Models**

RAG-WM：一种用于大型语言模型检索增强生成的高效黑箱水印方法 cs.CR

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.05249v1) [paper-pdf](http://arxiv.org/pdf/2501.05249v1)

**Authors**: Peizhuo Lv, Mengjie Sun, Hao Wang, Xiaofeng Wang, Shengzhi Zhang, Yuxuan Chen, Kai Chen, Limin Sun

**Abstract**: In recent years, tremendous success has been witnessed in Retrieval-Augmented Generation (RAG), widely used to enhance Large Language Models (LLMs) in domain-specific, knowledge-intensive, and privacy-sensitive tasks. However, attackers may steal those valuable RAGs and deploy or commercialize them, making it essential to detect Intellectual Property (IP) infringement. Most existing ownership protection solutions, such as watermarks, are designed for relational databases and texts. They cannot be directly applied to RAGs because relational database watermarks require white-box access to detect IP infringement, which is unrealistic for the knowledge base in RAGs. Meanwhile, post-processing by the adversary's deployed LLMs typically destructs text watermark information. To address those problems, we propose a novel black-box "knowledge watermark" approach, named RAG-WM, to detect IP infringement of RAGs. RAG-WM uses a multi-LLM interaction framework, comprising a Watermark Generator, Shadow LLM & RAG, and Watermark Discriminator, to create watermark texts based on watermark entity-relationship tuples and inject them into the target RAG. We evaluate RAG-WM across three domain-specific and two privacy-sensitive tasks on four benchmark LLMs. Experimental results show that RAG-WM effectively detects the stolen RAGs in various deployed LLMs. Furthermore, RAG-WM is robust against paraphrasing, unrelated content removal, knowledge insertion, and knowledge expansion attacks. Lastly, RAG-WM can also evade watermark detection approaches, highlighting its promising application in detecting IP infringement of RAG systems.

摘要: 近年来，检索增强生成(RAG)取得了巨大的成功，它被广泛用于增强领域特定、知识密集型和隐私敏感任务中的大型语言模型(LLM)。然而，攻击者可能会窃取这些有价值的破布并将其部署或商业化，这使得检测侵犯知识产权(IP)变得至关重要。大多数现有的所有权保护解决方案，如水印，都是为关系数据库和文本设计的。它们不能直接应用于RAG，因为关系数据库水印需要白盒访问来检测知识产权侵权，这对于RAG中的知识库来说是不现实的。同时，由对手部署的LLMS进行的后处理通常会破坏文本水印信息。针对这些问题，我们提出了一种新的黑盒“知识水印”方法RAG-WM来检测RAG的知识产权侵权行为。RAG-WM使用多LLM交互框架，包括水印生成器、阴影LLM和RAG和水印鉴别器，基于水印实体关系元组创建水印文本并将其注入目标RAG。我们在四个基准LLM上对RAG-WM进行了评估，测试了三个领域特定的任务和两个隐私敏感任务。实验结果表明，RAG-WM能够有效地检测出各种部署的LLM中被盗的RAG。此外，RAG-WM对释义、无关内容移除、知识插入和知识扩展攻击具有较强的鲁棒性。最后，RAG-WM还可以避开水印检测方法，在检测RAG系统的知识产权侵权行为方面具有广阔的应用前景。



## **8. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

将逻辑与自身对立：通过对比问题探索模型辩护 cs.CL

Our code is publicly available at  https://github.com/UKPLab/POATE-attack

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.01872v2) [paper-pdf](http://arxiv.org/pdf/2501.01872v2)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.

摘要: 大型语言模型尽管与人类价值观和伦理原则广泛一致，但仍然容易受到复杂的越狱攻击，这些攻击利用了它们的推理能力。现有的安全措施经常检测到公开的恶意意图，但无法解决细微的、推理驱动的漏洞。在这项工作中，我们介绍了POATE(极地相反查询生成，对抗性模板构建和精化)，这是一种新的越狱技术，利用对比推理来引发不道德的反应。波特在语义上设计了相反的意图，并将它们与对抗性模板整合在一起，以惊人的微妙程度引导模型指向有害的输出。我们对六个不同参数大小的不同语言模型家族进行了广泛的评估，以证明攻击的健壮性，与现有方法相比，攻击成功率显著提高(~44%)。针对这一问题，我们提出了意图感知COT和逆向思维COT，它们分解查询以检测恶意意图，并反向推理以评估和拒绝有害响应。这些方法增强了推理的健壮性，增强了模型对对手攻击的防御能力。



## **9. Trading Devil RL: Backdoor attack via Stock market, Bayesian Optimization and Reinforcement Learning**

交易魔鬼RL：通过股市、Bayesian优化和强化学习进行后门攻击 cs.LG

End of data poisoning research!: Navier-stokes equations (3D;  update); Reinforcement Learning (RL); HFT (High Frequency Trading); Limit  Order Markets and backdoor attack detection

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2412.17908v2) [paper-pdf](http://arxiv.org/pdf/2412.17908v2)

**Authors**: Orson Mengara

**Abstract**: With the rapid development of generative artificial intelligence, particularly large language models, a number of sub-fields of deep learning have made significant progress and are now very useful in everyday applications. For example, well-known financial institutions simulate a wide range of scenarios for various models created by their research teams using reinforcement learning, both before production and after regular operations. In this work, we propose a backdoor attack that focuses solely on data poisoning. This particular backdoor attack is classified as an attack without prior consideration or trigger, and we name it FinanceLLMsBackRL. Our aim is to examine the potential effects of large language models that use reinforcement learning systems for text production or speech recognition, finance, physics, or the ecosystem of contemporary artificial intelligence models.

摘要: 随着生成式人工智能，特别是大型语言模型的快速发展，深度学习的许多子领域取得了重大进展，现在在日常应用中非常有用。例如，知名金融机构在生产之前和常规运营之后使用强化学习为其研究团队创建的各种模型模拟各种场景。在这项工作中，我们提出了一种仅针对数据中毒的后门攻击。这种特殊的后门攻击被归类为未经事先考虑或触发的攻击，我们将其命名为Financial LLMsBackRL。我们的目标是检查使用强化学习系统进行文本生成或语音识别、金融、物理或当代人工智能模型生态系统的大型语言模型的潜在影响。



## **10. SpaLLM-Guard: Pairing SMS Spam Detection Using Open-source and Commercial LLMs**

SpaLLM-Guard：使用开源和商业LLM配对短信垃圾邮件检测 cs.CR

17 pages

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.04985v1) [paper-pdf](http://arxiv.org/pdf/2501.04985v1)

**Authors**: Muhammad Salman, Muhammad Ikram, Nardine Basta, Mohamed Ali Kaafar

**Abstract**: The increasing threat of SMS spam, driven by evolving adversarial techniques and concept drift, calls for more robust and adaptive detection methods. In this paper, we evaluate the potential of large language models (LLMs), both open-source and commercial, for SMS spam detection, comparing their performance across zero-shot, few-shot, fine-tuning, and chain-of-thought prompting approaches. Using a comprehensive dataset of SMS messages, we assess the spam detection capabilities of prominent LLMs such as GPT-4, DeepSeek, LLAMA-2, and Mixtral. Our findings reveal that while zero-shot learning provides convenience, it is unreliable for effective spam detection. Few-shot learning, particularly with carefully selected examples, improves detection but exhibits variability across models. Fine-tuning emerges as the most effective strategy, with Mixtral achieving 98.6% accuracy and a balanced false positive and false negative rate below 2%, meeting the criteria for robust spam detection. Furthermore, we explore the resilience of these models to adversarial attacks, finding that fine-tuning significantly enhances robustness against both perceptible and imperceptible manipulations. Lastly, we investigate the impact of concept drift and demonstrate that fine-tuned LLMs, especially when combined with few-shot learning, can mitigate its effects, maintaining high performance even on evolving spam datasets. This study highlights the importance of fine-tuning and tailored learning strategies to deploy LLMs effectively for real-world SMS spam detection

摘要: 在不断发展的敌意技术和概念漂移的推动下，垃圾短信的威胁越来越大，这要求更健壮和自适应的检测方法。在本文中，我们评估了开源和商业的大型语言模型(LLM)在短信垃圾邮件检测中的潜力，比较了它们在零射、少射、微调和思维链提示方法中的性能。使用全面的短信数据集，我们评估了GPT-4、DeepSeek、Llama-2和Mixtral等知名LLMS的垃圾邮件检测能力。我们的发现表明，虽然零机会学习提供了便利，但对于有效的垃圾邮件检测来说，它是不可靠的。少发式学习，尤其是精心挑选的例子，提高了检测能力，但在不同模型之间表现出多样性。微调成为最有效的策略，Mixtral达到了98.6%的准确率，假阳性率和假阴性率平衡在2%以下，满足了稳健的垃圾邮件检测标准。此外，我们探讨了这些模型对敌意攻击的弹性，发现微调显著增强了对可感知和不可感知操纵的稳健性。最后，我们研究了概念漂移的影响，并证明了微调的LLMS，特别是当与少镜头学习相结合时，可以缓解其影响，即使在不断演变的垃圾邮件数据集上也保持了高性能。这项研究强调了微调和量身定制的学习策略的重要性，以有效地部署LLMS来检测真实世界的短信垃圾邮件



## **11. Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency**

通过洗牌不一致性破解多模式大型语言模型 cs.CR

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.04931v1) [paper-pdf](http://arxiv.org/pdf/2501.04931v1)

**Authors**: Shiji Zhao, Ranjie Duan, Fengxiang Wang, Chi Chen, Caixin Kang, Jialing Tao, YueFeng Chen, Hui Xue, Xingxing Wei

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved impressive performance and have been put into practical use in commercial applications, but they still have potential safety mechanism vulnerabilities. Jailbreak attacks are red teaming methods that aim to bypass safety mechanisms and discover MLLMs' potential risks. Existing MLLMs' jailbreak methods often bypass the model's safety mechanism through complex optimization methods or carefully designed image and text prompts. Despite achieving some progress, they have a low attack success rate on commercial closed-source MLLMs. Unlike previous research, we empirically find that there exists a Shuffle Inconsistency between MLLMs' comprehension ability and safety ability for the shuffled harmful instruction. That is, from the perspective of comprehension ability, MLLMs can understand the shuffled harmful text-image instructions well. However, they can be easily bypassed by the shuffled harmful instructions from the perspective of safety ability, leading to harmful responses. Then we innovatively propose a text-image jailbreak attack named SI-Attack. Specifically, to fully utilize the Shuffle Inconsistency and overcome the shuffle randomness, we apply a query-based black-box optimization method to select the most harmful shuffled inputs based on the feedback of the toxic judge model. A series of experiments show that SI-Attack can improve the attack's performance on three benchmarks. In particular, SI-Attack can obviously improve the attack success rate for commercial MLLMs such as GPT-4o or Claude-3.5-Sonnet.

摘要: 多通道大语言模型已经取得了令人印象深刻的性能，并在商业应用中得到了实际应用，但它们仍然存在潜在的安全机制漏洞。越狱攻击是一种旨在绕过安全机制并发现MLLMS潜在风险的红色团队方法。现有的MLLMS越狱方法往往通过复杂的优化方法或精心设计的图像和文本提示绕过模型的安全机制。尽管取得了一些进展，但他们对商业闭源MLLMS的攻击成功率很低。与前人的研究不同，我们的实证研究发现，多语言传播者对有害指令的理解能力和安全能力之间存在着洗牌不一致的现象。也就是说，从理解能力的角度来看，MLLMS能够很好地理解混洗后的有害文本-图像指令。然而，从安全能力的角度来看，它们很容易被洗牌后的有害指令绕过，导致有害反应。在此基础上，我们创新性地提出了一种文本图像越狱攻击方案SI-Attack。具体地说，为了充分利用混洗的不一致性和克服混洗的随机性，我们应用了一种基于查询的黑盒优化方法，根据有毒判断模型的反馈选择最有害的混洗输入。一系列的实验表明，SI-攻击在三个基准上都能提高攻击的性能。特别是对于GPT-40或Claude-3.5-Sonnet等商用MLLMS，SI-Attack能明显提高攻击成功率。



## **12. Navigating the Designs of Privacy-Preserving Fine-tuning for Large Language Models**

引导大型语言模型的隐私保护微调设计 cs.LG

4 pages, 2 figures

**SubmitDate**: 2025-01-09    [abs](http://arxiv.org/abs/2501.04323v2) [paper-pdf](http://arxiv.org/pdf/2501.04323v2)

**Authors**: Haonan Shi, Tu Ouyang, An Wang

**Abstract**: Instruction tuning has proven effective in enhancing Large Language Models' (LLMs) performance on downstream tasks. However, real-world fine-tuning faces inherent conflicts between model providers' intellectual property protection, clients' data privacy requirements, and tuning costs. While recent approaches like split learning and offsite tuning demonstrate promising architectures for privacy-preserving fine-tuning, there is a gap in systematically addressing the multidimensional trade-offs required for diverse real-world deployments. We propose several indicative evaluation metrics to guide design trade-offs for privacy-preserving fine-tuning and a series of example designs, collectively named GuardedTuning; they result from novel combinations of system architectures with adapted privacy-enhancement methods and emerging computation techniques. Each design represents distinct trade-offs across model utility, privacy guarantees, and costs. Experimental results demonstrate that these designs protect against data reconstruction attacks while maintaining competitive fine-tuning performance.

摘要: 指令调优在提高大型语言模型(LLM)在下游任务上的性能方面已被证明是有效的。然而，现实世界的微调面临着模型提供商的知识产权保护、客户的数据隐私要求和调整成本之间的内在冲突。虽然最近的方法，如拆分学习和异地调整，展示了保护隐私的微调的有前途的架构，但在系统地解决不同现实世界部署所需的多维权衡方面存在差距。我们提出了几个指示性的评估指标来指导隐私保护微调和一系列示例设计的权衡，统称为GuardedTuning；它们是系统架构与适应隐私增强方法和新兴计算技术的新颖组合的结果。每一种设计都代表着在模型效用、隐私保障和成本方面的不同权衡。实验结果表明，这些设计在保持具有竞争力的微调性能的同时，能够抵御数据重构攻击。



## **13. Watch Out for Your Guidance on Generation! Exploring Conditional Backdoor Attacks against Large Language Models**

留意您对世代的指导！探索针对大型语言模型的条件后门攻击 cs.CL

The paper has been accepted to AAAI 2025

**SubmitDate**: 2025-01-08    [abs](http://arxiv.org/abs/2404.14795v5) [paper-pdf](http://arxiv.org/pdf/2404.14795v5)

**Authors**: Jiaming He, Wenbo Jiang, Guanyu Hou, Wenshu Fan, Rui Zhang, Hongwei Li

**Abstract**: Mainstream backdoor attacks on large language models (LLMs) typically set a fixed trigger in the input instance and specific responses for triggered queries. However, the fixed trigger setting (e.g., unusual words) may be easily detected by human detection, limiting the effectiveness and practicality in real-world scenarios. To enhance the stealthiness of backdoor activation, we present a new poisoning paradigm against LLMs triggered by specifying generation conditions, which are commonly adopted strategies by users during model inference. The poisoned model performs normally for output under normal/other generation conditions, while becomes harmful for output under target generation conditions. To achieve this objective, we introduce BrieFool, an efficient attack framework. It leverages the characteristics of generation conditions by efficient instruction sampling and poisoning data generation, thereby influencing the behavior of LLMs under target conditions. Our attack can be generally divided into two types with different targets: Safety unalignment attack and Ability degradation attack. Our extensive experiments demonstrate that BrieFool is effective across safety domains and ability domains, achieving higher success rates than baseline methods, with 94.3 % on GPT-3.5-turbo

摘要: 针对大型语言模型(LLM)的主流后门攻击通常会在输入实例中设置固定的触发器，并为触发的查询设置特定的响应。然而，固定的触发设置(例如，不寻常的单词)可能很容易被人类检测到，从而限制了在现实世界场景中的有效性和实用性。为了增强后门激活的隐蔽性，我们提出了一种新的针对通过指定生成条件触发的LLM的中毒范例，这些策略是用户在模型推理中经常采用的策略。中毒模型在正常/其他发电条件下的出力表现正常，而在目标发电条件下的出力变得有害。为了实现这一目标，我们引入了一种高效的攻击框架BrieFool。它通过高效的指令采样和中毒数据生成来利用生成条件的特征，从而影响目标条件下的LLM的行为。我们的攻击一般可以分为两种类型，针对不同的目标：安全联盟攻击和能力退化攻击。我们的广泛实验表明，BrieFool跨安全域和能量域是有效的，获得了比基准方法更高的成功率，在GPT-3.5-Turbo上的成功率为94.3%



## **14. Not all tokens are created equal: Perplexity Attention Weighted Networks for AI generated text detection**

并非所有令牌都是平等的：用于人工智能生成文本检测的困惑注意力加权网络 cs.CL

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.03940v1) [paper-pdf](http://arxiv.org/pdf/2501.03940v1)

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho

**Abstract**: The rapid advancement in large language models (LLMs) has significantly enhanced their ability to generate coherent and contextually relevant text, raising concerns about the misuse of AI-generated content and making it critical to detect it. However, the task remains challenging, particularly in unseen domains or with unfamiliar LLMs. Leveraging LLM next-token distribution outputs offers a theoretically appealing approach for detection, as they encapsulate insights from the models' extensive pre-training on diverse corpora. Despite its promise, zero-shot methods that attempt to operationalize these outputs have met with limited success. We hypothesize that one of the problems is that they use the mean to aggregate next-token distribution metrics across tokens, when some tokens are naturally easier or harder to predict and should be weighted differently. Based on this idea, we propose the Perplexity Attention Weighted Network (PAWN), which uses the last hidden states of the LLM and positions to weight the sum of a series of features based on metrics from the next-token distribution across the sequence length. Although not zero-shot, our method allows us to cache the last hidden states and next-token distribution metrics on disk, greatly reducing the training resource requirements. PAWN shows competitive and even better performance in-distribution than the strongest baselines (fine-tuned LMs) with a fraction of their trainable parameters. Our model also generalizes better to unseen domains and source models, with smaller variability in the decision boundary across distribution shifts. It is also more robust to adversarial attacks, and if the backbone has multilingual capabilities, it presents decent generalization to languages not seen during supervised training, with LLaMA3-1B reaching a mean macro-averaged F1 score of 81.46% in cross-validation with nine languages.

摘要: 大型语言模型(LLM)的快速发展极大地增强了它们生成连贯和上下文相关文本的能力，这引起了人们对滥用人工智能生成的内容的担忧，并使检测它变得至关重要。然而，这项任务仍然具有挑战性，特别是在看不见的领域或具有不熟悉的LLM的领域。利用LLM下一个令牌分发输出提供了一种理论上有吸引力的检测方法，因为它们概括了模型对不同语料库的广泛预培训的见解。尽管有希望，但试图将这些产出付诸实施的零射击方法却取得了有限的成功。我们假设其中一个问题是，当一些令牌自然更容易或更难预测，并且应该以不同的权重进行加权时，它们使用平均值来聚合跨令牌的下一令牌分发度量。基于这一思想，我们提出了困惑注意力加权网络(PAWN)，它利用LLM的最后一个隐藏状态和位置来加权一系列特征的和，基于下一个令牌分布在整个序列长度上的度量。虽然不是零命中率，但我们的方法允许我们在磁盘上缓存最后的隐藏状态和下一个令牌分布度量，大大减少了训练资源需求。与最强的基线(微调LMS)相比，PAWN显示出具有竞争力的分布性能，甚至比它们的可训练参数的一小部分更好。我们的模型也更好地推广到看不见的域和源模型，跨分布转变的决策边界的可变性较小。LLaMA3-1B在与9种语言的交叉验证中达到了81.46%的平均宏观平均F1分数。



## **15. PSA-VLM: Enhancing Vision-Language Model Safety through Progressive Concept-Bottleneck-Driven Alignment**

PSA-VLM：通过渐进式概念瓶颈驱动对齐增强视觉语言模型安全性 cs.CV

arXiv admin note: substantial text overlap with arXiv:2405.13581

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2411.11543v3) [paper-pdf](http://arxiv.org/pdf/2411.11543v3)

**Authors**: Zhendong Liu, Yuanbi Nie, Yingshui Tan, Jiaheng Liu, Xiangyu Yue, Qiushi Cui, Chongjun Wang, Xiaoyong Zhu, Bo Zheng

**Abstract**: Benefiting from the powerful capabilities of Large Language Models (LLMs), pre-trained visual encoder models connected to LLMs form Vision Language Models (VLMs). However, recent research shows that the visual modality in VLMs is highly vulnerable, allowing attackers to bypass safety alignment in LLMs through visually transmitted content, launching harmful attacks. To address this challenge, we propose a progressive concept-based alignment strategy, PSA-VLM, which incorporates safety modules as concept bottlenecks to enhance visual modality safety alignment. By aligning model predictions with specific safety concepts, we improve defenses against risky images, enhancing explainability and controllability while minimally impacting general performance. Our method is obtained through two-stage training. The low computational cost of the first stage brings very effective performance improvement, and the fine-tuning of the language model in the second stage further improves the safety performance. Our method achieves state-of-the-art results on popular VLM safety benchmark.

摘要: 得益于大型语言模型的强大功能，连接到大型语言模型的预先训练的视觉编码器模型形成了视觉语言模型。然而，最近的研究表明，VLMS中的视觉通道非常容易受到攻击，使得攻击者能够通过视觉传输的内容绕过LLMS中的安全对齐，从而发起有害攻击。为了应对这一挑战，我们提出了一种基于概念的渐进式对齐策略PSA-VLM，该策略将安全模块作为概念瓶颈纳入其中，以增强视觉通道的安全对齐。通过将模型预测与特定的安全概念相结合，我们改进了对危险图像的防御，增强了可解释性和可控性，同时将对总体性能的影响降至最低。我们的方法是通过两个阶段的训练获得的。第一阶段的低运算量带来了非常有效的性能提升，第二阶段对语言模型的微调进一步提高了安全性能。我们的方法在流行的VLM安全基准上获得了最先进的结果。



## **16. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection**

PhishAgent：一种用于网络钓鱼网页检测的鲁棒多模式代理 cs.CR

Accepted at AAAI 2025

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2408.10738v2) [paper-pdf](http://arxiv.org/pdf/2408.10738v2)

**Authors**: Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, Bryan Hooi

**Abstract**: Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also face notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the relevant top k items from offline knowledge bases, using available information from a webpage, including logos and HTML. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.

摘要: 网络钓鱼攻击是在线安全的主要威胁，利用用户漏洞窃取敏感信息。已经开发了各种方法来对抗网络钓鱼，每一种方法的精确度都不同，但它们也面临着显著的局限性。在本研究中，我们介绍了PhishAgent，一个结合了广泛工具的多通道代理，将线上和线下知识库与多通道大语言模型(MLLMS)相结合。这一组合导致了更广泛的品牌覆盖，从而提高了品牌认知度和召回率。此外，我们还提出了一个多通道信息检索框架，该框架利用网页中的可用信息，包括标识和超文本标记语言，从离线知识库中提取相关的前k个条目。基于三个真实数据集的实验结果表明，该框架在保持模型效率的同时，显著提高了检测准确率，减少了误报和漏报。此外，PhishAgent对各种类型的对抗性攻击表现出很强的韧性。



## **17. MRJ-Agent: An Effective Jailbreak Agent for Multi-Round Dialogue**

MRJ-Agent：多轮对话的有效越狱代理 cs.AI

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2411.03814v2) [paper-pdf](http://arxiv.org/pdf/2411.03814v2)

**Authors**: Fengxiang Wang, Ranjie Duan, Peng Xiao, Xiaojun Jia, Shiji Zhao, Cheng Wei, YueFeng Chen, Chongwen Wang, Jialing Tao, Hang Su, Jun Zhu, Hui Xue

**Abstract**: Large Language Models (LLMs) demonstrate outstanding performance in their reservoir of knowledge and understanding capabilities, but they have also been shown to be prone to illegal or unethical reactions when subjected to jailbreak attacks. To ensure their responsible deployment in critical applications, it is crucial to understand the safety capabilities and vulnerabilities of LLMs. Previous works mainly focus on jailbreak in single-round dialogue, overlooking the potential jailbreak risks in multi-round dialogues, which are a vital way humans interact with and extract information from LLMs. Some studies have increasingly concentrated on the risks associated with jailbreak in multi-round dialogues. These efforts typically involve the use of manually crafted templates or prompt engineering techniques. However, due to the inherent complexity of multi-round dialogues, their jailbreak performance is limited. To solve this problem, we propose a novel multi-round dialogue jailbreaking agent, emphasizing the importance of stealthiness in identifying and mitigating potential threats to human values posed by LLMs. We propose a risk decomposition strategy that distributes risks across multiple rounds of queries and utilizes psychological strategies to enhance attack strength. Extensive experiments show that our proposed method surpasses other attack methods and achieves state-of-the-art attack success rate. We will make the corresponding code and dataset available for future research. The code will be released soon.

摘要: 大型语言模型(LLM)在其知识和理解能力方面表现出色，但也被证明在受到越狱攻击时容易出现非法或不道德的反应。为了确保它们在关键应用中负责任地部署，了解LLMS的安全能力和漏洞至关重要。以往的研究主要集中在单轮对话中的越狱，而忽略了多轮对话中潜在的越狱风险，而多轮对话是人类与小武器系统交互和提取信息的重要方式。一些研究越来越集中于多轮对话中越狱的相关风险。这些工作通常涉及使用手工制作的模板或即时工程技术。然而，由于多轮对话的内在复杂性，它们的越狱表现有限。为了解决这一问题，我们提出了一种新的多轮对话越狱代理，强调了隐蔽性在识别和缓解LLMS对人类价值构成的潜在威胁方面的重要性。我们提出了一种风险分解策略，将风险分布在多轮查询中，并利用心理策略来增强攻击强度。大量实验表明，该方法优于其他攻击方法，达到了最高的攻击成功率。我们将提供相应的代码和数据集，供未来研究使用。代码很快就会发布。



## **18. Practical Secure Inference Algorithm for Fine-tuned Large Language Model Based on Fully Homomorphic Encryption**

基于全同形加密的微调大语言模型的实用安全推理算法 cs.CR

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2501.01672v2) [paper-pdf](http://arxiv.org/pdf/2501.01672v2)

**Authors**: Zhang Ruoyan, Zheng Zhongxiang, Bao Wankang

**Abstract**: Large language models(LLMs) are currently at the forefront of the machine learning field, which show a broad application prospect but at the same time expose some risks of privacy leakage. We combined Fully Homomorphic Encryption(FHE) and provable security theory with Parameter-Efficient Fine-Tuning(PEFT) to propose an efficient and secure inference scheme for LLMs. More specially, we focus on pre-trained LLMs which rely on open-sourced base model and then fine-tuned with the private datasets by LoRA. This is a popular road-map for Vertical Domain Models such as LawGPT and BenTsao. We use two key technologies below. Firstly, we divide the whole model into the public part and the private part. The weights of public part are publicly accessible(e.g. the open-sourced base model) while the private part needs to be protected(e.g. the LoRA matrices). In this way, the overhead brought by computing on private data can be greatly reduced. Secondly, we propose a general method to transform a linear layer into another one which provides security against model extraction attacks and preserves its original functionality, which denoted as Private Linear Layer(PLL). Then we use this method on the LoRA matrices to make sure that the server protects their private weights without restricting the user's input. We also show that the difficulty of performing model extraction attacks for PLL can be reduced to the well-known hard problem Learning with Errors(LWE). Combing this method with FHE, we can protect user's input at the same time. In this paper, we use the open-source model ChatGLM2-6B as the base model which is fine-tuned by LoRA. Experimental results show the inference efficiency of our scheme reaches 1.61s/token which displays that the scheme has good practicality.

摘要: 大语言模型目前处于机器学习领域的前沿，在显示出广阔的应用前景的同时，也暴露出一些隐私泄露的风险。将完全同态加密(FHE)和可证明安全理论与参数高效精调(PEFT)相结合，提出了一种高效安全的LLMS推理方案。更具体地说，我们专注于预先训练的LLM，它依赖于开源的基本模型，然后使用LORA的私有数据集进行微调。这是LawGPT和BenTsao等垂直领域模式的流行路线图。我们使用以下两项关键技术。首先，我们将整个模型分为公共部分和私人部分。公共部分的权重是可公开访问的(例如，开放源码的基本模型)，而私有部分需要保护(例如，LORA矩阵)。通过这种方式，可以大大降低计算私有数据带来的开销。其次，我们提出了一种将一个线性层转换为另一个线性层的通用方法，该方法既能保证模型提取攻击的安全性，又能保持原有的功能，称为专用线性层。然后，我们对Lora矩阵使用此方法，以确保服务器在不限制用户输入的情况下保护它们的私有权重。我们还证明了对PLL进行模型提取攻击的难度可以归结为众所周知的带错误学习(LWE)的困难问题。将该方法与FHE相结合，可以同时保护用户的输入。在本文中，我们使用开源模型ChatGLM2-6B作为基础模型，该模型由LORA进行了微调。实验结果表明，该方案的推理效率达到1.61S/Token，具有较好的实用性。



## **19. ChatBug: A Common Vulnerability of Aligned LLMs Induced by Chat Templates**

ChatBug：聊天模板引发的对齐LLM的常见漏洞 cs.CR

This paper is accepted to AAAI 2025

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2406.12935v2) [paper-pdf](http://arxiv.org/pdf/2406.12935v2)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Bill Yuchen Lin, Radha Poovendran

**Abstract**: Large language models (LLMs) are expected to follow instructions from users and engage in conversations. Techniques to enhance LLMs' instruction-following capabilities typically fine-tune them using data structured according to a predefined chat template. Although chat templates are shown to be effective in optimizing LLM performance, their impact on safety alignment of LLMs has been less understood, which is crucial for deploying LLMs safely at scale.   In this paper, we investigate how chat templates affect safety alignment of LLMs. We identify a common vulnerability, named ChatBug, that is introduced by chat templates. Our key insight to identify ChatBug is that the chat templates provide a rigid format that need to be followed by LLMs, but not by users. Hence, a malicious user may not necessarily follow the chat template when prompting LLMs. Instead, malicious users could leverage their knowledge of the chat template and accordingly craft their prompts to bypass safety alignments of LLMs. We develop two attacks to exploit the ChatBug vulnerability. We demonstrate that a malicious user can exploit the ChatBug vulnerability of eight state-of-the-art (SOTA) LLMs and effectively elicit unintended responses from these models. Moreover, we show that ChatBug can be exploited by existing jailbreak attacks to enhance their attack success rates. We investigate potential countermeasures to ChatBug. Our results show that while adversarial training effectively mitigates the ChatBug vulnerability, the victim model incurs significant performance degradation. These results highlight the trade-off between safety alignment and helpfulness. Developing new methods for instruction tuning to balance this trade-off is an open and critical direction for future research

摘要: 大型语言模型(LLM)应该遵循用户的指示并参与对话。增强LLMS的指令遵循能力的技术通常使用根据预定义的聊天模板构造的数据对其进行微调。尽管聊天模板被证明在优化LLM性能方面是有效的，但人们对它们对LLM安全调整的影响知之甚少，这对于安全地大规模部署LLMS至关重要。在本文中，我们研究了聊天模板如何影响LLMS的安全对齐。我们发现了聊天模板引入的一个名为ChatBug的常见漏洞。我们识别ChatBug的关键洞察力是，聊天模板提供了一种严格的格式，需要LLMS遵循，而不是用户。因此，恶意用户在提示LLMS时可能不一定遵循聊天模板。相反，恶意用户可以利用他们对聊天模板的了解，并相应地精心编制他们的提示，以绕过LLMS的安全对齐。我们开发了两个攻击来利用ChatBug漏洞。我们演示了恶意用户可以利用8个最先进的(SOTA)LLM的ChatBug漏洞，并有效地从这些模型中引发意外响应。此外，我们发现ChatBug可以被现有的越狱攻击所利用，以提高他们的攻击成功率。我们调查了针对ChatBug的潜在对策。我们的结果表明，虽然对抗性训练有效地缓解了ChatBug漏洞，但受害者模型导致了显著的性能下降。这些结果突显了安全性调整和帮助之间的权衡。开发新的教学调整方法来平衡这种权衡是未来研究的一个开放和关键的方向



## **20. HuRef: HUman-REadable Fingerprint for Large Language Models**

HuRef：大型语言模型的人类可读取指纹 cs.CL

NeurIPS 2024

**SubmitDate**: 2025-01-07    [abs](http://arxiv.org/abs/2312.04828v5) [paper-pdf](http://arxiv.org/pdf/2312.04828v5)

**Authors**: Boyi Zeng, Lizheng Wang, Yuncong Hu, Yi Xu, Chenghu Zhou, Xinbing Wang, Yu Yu, Zhouhan Lin

**Abstract**: Protecting the copyright of large language models (LLMs) has become crucial due to their resource-intensive training and accompanying carefully designed licenses. However, identifying the original base model of an LLM is challenging due to potential parameter alterations. In this study, we introduce HuRef, a human-readable fingerprint for LLMs that uniquely identifies the base model without interfering with training or exposing model parameters to the public. We first observe that the vector direction of LLM parameters remains stable after the model has converged during pretraining, with negligible perturbations through subsequent training steps, including continued pretraining, supervised fine-tuning, and RLHF, which makes it a sufficient condition to identify the base model. The necessity is validated by continuing to train an LLM with an extra term to drive away the model parameters' direction and the model becomes damaged. However, this direction is vulnerable to simple attacks like dimension permutation or matrix rotation, which significantly change it without affecting performance. To address this, leveraging the Transformer structure, we systematically analyze potential attacks and define three invariant terms that identify an LLM's base model. Due to the potential risk of information leakage, we cannot publish invariant terms directly. Instead, we map them to a Gaussian vector using an encoder, then convert it into a natural image using StyleGAN2, and finally publish the image. In our black-box setting, all fingerprinting steps are internally conducted by the LLMs owners. To ensure the published fingerprints are honestly generated, we introduced Zero-Knowledge Proof (ZKP). Experimental results across various LLMs demonstrate the effectiveness of our method. The code is available at https://github.com/LUMIA-Group/HuRef.

摘要: 保护大型语言模型(LLM)的版权已变得至关重要，因为它们需要进行资源密集型培训，并附带精心设计的许可证。然而，由于潜在的参数变化，识别LLM的原始基础模型是具有挑战性的。在这项研究中，我们引入了HuRef，这是一种用于LLMS的人类可读指纹，它在不干扰训练或向公众暴露模型参数的情况下唯一地识别基本模型。我们首先观察到，在预训练过程中模型收敛后，LLM参数的向量方向保持稳定，通过后续的训练步骤，包括继续预训练、有监督的微调和RLHF，可以忽略不计的扰动，这使得它成为识别基本模型的充分条件。通过继续训练一个带有额外项的LLM来驱离模型参数的方向，从而使模型受损，从而验证了这种必要性。然而，这个方向很容易受到维度置换或矩阵旋转等简单攻击，这些攻击会在不影响性能的情况下显著改变它。为了解决这个问题，利用Transformer结构，我们系统地分析了潜在的攻击，并定义了识别LLM基本模型的三个不变术语。由于潜在的信息泄露风险，我们不能直接发布不变项。相反，我们使用编码器将它们映射到高斯向量，然后使用StyleGAN2将其转换为自然图像，最后发布图像。在我们的黑盒设置中，所有指纹识别步骤都由LLMS所有者在内部执行。为了确保公布的指纹是真实生成的，我们引入了零知识证明(ZKP)。在不同LLM上的实验结果证明了该方法的有效性。代码可在https://github.com/LUMIA-Group/HuRef.上获得



## **21. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

时间序列预测大型语言模型中的对抗漏洞 cs.LG

11 pages, 5 figures

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2412.08099v2) [paper-pdf](http://arxiv.org/pdf/2412.08099v2)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in the field of time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like TimeGPT and LLM-Time with GPT-3.5, GPT-4, LLaMa, and Mistral, show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications.

摘要: 大型语言模型最近在时间序列预测领域显示出巨大的潜力，在处理复杂的时间数据方面提供了令人印象深刻的能力。然而，它们在实际应用中的健壮性和可靠性仍然没有得到充分的研究，特别是关于它们对对手攻击的敏感性。本文提出了一种基于LLM的时间序列预测的对抗性攻击框架。通过使用无梯度和黑盒优化方法，我们产生了最小但高效的扰动，这些扰动显著降低了跨多个数据集和LLM体系结构的预测精度。我们的实验，包括使用GPT-3.5、GPT-4、LLAMA和Mistral的TimeGPT和LLM-Time模型，表明对抗性攻击导致的性能降级比随机噪声严重得多，并证明了我们的攻击在不同LLM上的广泛有效性。这些结果强调了低层管理在时间序列预测中的关键弱点，强调了需要强大的防御机制来确保其在实际应用中的可靠部署。



## **22. Pathway to Secure and Trustworthy ZSM for LLMs: Attacks, Defense, and Opportunities**

为LLM提供安全和值得信赖的ZZ之路：攻击、防御和机会 cs.CR

7 pages, 4 figures

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2408.00722v2) [paper-pdf](http://arxiv.org/pdf/2408.00722v2)

**Authors**: Sunder Ali Khowaja, Parus Khuwaja, Kapal Dev, Hussam Al Hamadi, Engin Zeydan

**Abstract**: Recently, large language models (LLMs) have been gaining a lot of interest due to their adaptability and extensibility in emerging applications, including communication networks. It is anticipated that ZSM networks will be able to support LLMs as a service, as they provide ultra reliable low-latency communications and closed loop massive connectivity. However, LLMs are vulnerable to data and model privacy issues that affect the trustworthiness of LLMs to be deployed for user-based services. In this paper, we explore the security vulnerabilities associated with fine-tuning LLMs in ZSM networks, in particular the membership inference attack. We define the characteristics of an attack network that can perform a membership inference attack if the attacker has access to the fine-tuned model for the downstream task. We show that the membership inference attacks are effective for any downstream task, which can lead to a personal data breach when using LLM as a service. The experimental results show that the attack success rate of maximum 92% can be achieved on named entity recognition task. Based on the experimental analysis, we discuss possible defense mechanisms and present possible research directions to make the LLMs more trustworthy in the context of ZSM networks.

摘要: 近年来，大型语言模型因其在通信网络等新兴应用中的适应性和可扩展性而引起了人们的极大兴趣。预计ZSM网络将能够支持LLMS作为一种服务，因为它们提供超可靠的低延迟通信和闭环路海量连接。然而，LLM容易受到数据和模型隐私问题的影响，这些问题会影响要为基于用户的服务部署的LLM的可信度。本文研究了ZSM网络中与微调LLMS相关的安全漏洞，特别是成员推理攻击。我们定义了攻击网络的特征，如果攻击者有权访问下游任务的微调模型，则该攻击网络可以执行成员关系推理攻击。我们证明了成员关系推断攻击对于任何下游任务都是有效的，当使用LLM作为服务时，这可能导致个人数据泄露。实验结果表明，命名实体识别任务的攻击成功率最高可达92%。在实验分析的基础上，我们讨论了可能的防御机制，并提出了可能的研究方向，以使ZSM网络环境下的LLMS更可信。



## **23. FlipedRAG: Black-Box Opinion Manipulation Attacks to Retrieval-Augmented Generation of Large Language Models**

FlipedRAG：对大型语言模型的检索增强生成的黑匣子观点操纵攻击 cs.IR

arXiv admin note: text overlap with arXiv:2407.13757

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2501.02968v1) [paper-pdf](http://arxiv.org/pdf/2501.02968v1)

**Authors**: Zhuo Chen, Yuyang Gong, Miaokun Chen, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) addresses hallucination and real-time constraints by dynamically retrieving relevant information from a knowledge database to supplement the LLMs' input. When presented with a query, RAG selects the most semantically similar texts from its knowledge bases and uses them as context for the LLMs to generate more accurate responses. RAG also creates a new attack surface, especially since RAG databases are frequently sourced from public domains. While existing studies have predominantly focused on optimizing RAG's performance and efficiency, emerging research has begun addressing the security concerns associated with RAG. However, these works have some limitations, typically focusing on either white-box methodologies or heuristic-based black-box attacks. Furthermore, prior research has mainly targeted simple factoid question answering, which is neither practically challenging nor resistant to correction. In this paper, we unveil a more realistic and threatening scenario: opinion manipulation for controversial topics against RAG. Particularly, we propose a novel RAG black-box attack method, termed FlipedRAG, which is transfer-based. By leveraging instruction engineering, we obtain partial retrieval model outputs from black-box RAG system, facilitating the training of surrogate models to enhance the effectiveness of opinion manipulation attack. Extensive experimental results confirms that our approach significantly enhances the average success rate of opinion manipulation by 16.7%. It achieves an average of a 50% directional change in the opinion polarity of RAG responses across four themes. Additionally, it induces a 20% shift in user cognition. Furthermore, we discuss the efficacy of potential defense mechanisms and conclude that they are insufficient in mitigating this type of attack, highlighting the urgent need to develop novel defensive strategies.

摘要: 检索-增强生成(RAG)通过从知识数据库中动态检索相关信息来补充LLMS的输入，从而解决幻觉和实时约束。当出现查询时，RAG从其知识库中选择语义最相似的文本，并将其用作LLMS的上下文，以生成更准确的响应。RAG还创造了一个新的攻击面，特别是因为RAG数据库经常来自公共域。虽然现有的研究主要集中在优化RAG的性能和效率上，但新兴的研究已经开始解决与RAG相关的安全问题。然而，这些工作有一些局限性，通常集中在白盒方法或基于启发式的黑盒攻击。此外，以前的研究主要针对简单的事实式问题回答，这既不具有实际挑战性，也不抵制纠正。在这篇文章中，我们揭示了一个更现实和更具威胁性的场景：针对RAG的有争议话题的观点操纵。特别地，我们提出了一种新的基于传输的RAG黑盒攻击方法，称为FliedRAG。利用教学工程技术，从黑盒RAG系统中获取部分检索模型输出，便于对代理模型的训练，提高意见操纵攻击的有效性。大量的实验结果表明，该方法显著提高了意见操纵的平均成功率16.7%。它实现了四个主题的RAG回复的意见极性平均50%的方向性变化。此外，它还会导致用户认知发生20%的变化。此外，我们讨论了潜在的防御机制的有效性，并得出结论，它们在缓解这种类型的攻击方面是不够的，突出了开发新的防御策略的迫切需要。



## **24. LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset Simulating Disinformation Generation**

LlamaPartialSpoof：一个LLM驱动的模拟虚假信息生成的假语音数据集 eess.AS

5 pages, ICASSP 2025

**SubmitDate**: 2025-01-06    [abs](http://arxiv.org/abs/2409.14743v2) [paper-pdf](http://arxiv.org/pdf/2409.14743v2)

**Authors**: Hieu-Thi Luong, Haoyang Li, Lin Zhang, Kong Aik Lee, Eng Siong Chng

**Abstract**: Previous fake speech datasets were constructed from a defender's perspective to develop countermeasure (CM) systems without considering diverse motivations of attackers. To better align with real-life scenarios, we created LlamaPartialSpoof, a 130-hour dataset that contains both fully and partially fake speech, using a large language model (LLM) and voice cloning technologies to evaluate the robustness of CMs. By examining valuable information for both attackers and defenders, we identify several key vulnerabilities in current CM systems, which can be exploited to enhance attack success rates, including biases toward certain text-to-speech models or concatenation methods. Our experimental results indicate that the current fake speech detection system struggle to generalize to unseen scenarios, achieving a best performance of 24.49% equal error rate.

摘要: 之前的虚假语音数据集是从防御者的角度构建的，以开发对策（CM）系统，而不考虑攻击者的不同动机。为了更好地与现实生活场景保持一致，我们创建了LlamaPartialSpoof，这是一个包含完全和部分虚假语音的130小时数据集，使用大型语言模型（LLM）和语音克隆技术来评估CM的稳健性。通过检查攻击者和防御者的有价值的信息，我们发现了当前CM系统中的几个关键漏洞，可以利用这些漏洞来提高攻击成功率，包括对某些文本到语音模型或级联方法的偏见。我们的实验结果表明，当前的虚假语音检测系统很难推广到不可见的场景，实现了24.49%等错误率的最佳性能。



## **25. Layer-Level Self-Exposure and Patch: Affirmative Token Mitigation for Jailbreak Attack Defense**

分层自我暴露和补丁：越狱攻击防御的肯定代币缓解 cs.CR

**SubmitDate**: 2025-01-05    [abs](http://arxiv.org/abs/2501.02629v1) [paper-pdf](http://arxiv.org/pdf/2501.02629v1)

**Authors**: Yang Ouyang, Hengrui Gu, Shuhang Lin, Wenyue Hua, Jie Peng, Bhavya Kailkhura, Tianlong Chen, Kaixiong Zhou

**Abstract**: As large language models (LLMs) are increasingly deployed in diverse applications, including chatbot assistants and code generation, aligning their behavior with safety and ethical standards has become paramount. However, jailbreak attacks, which exploit vulnerabilities to elicit unintended or harmful outputs, threaten LLMs' safety significantly. In this paper, we introduce Layer-AdvPatcher, a novel methodology designed to defend against jailbreak attacks by utilizing an unlearning strategy to patch specific layers within LLMs through self-augmented datasets. Our insight is that certain layer(s), tend to produce affirmative tokens when faced with harmful prompts. By identifying these layers and adversarially exposing them to generate more harmful data, one can understand their inherent and diverse vulnerabilities to attacks. With these exposures, we then "unlearn" these issues, reducing the impact of affirmative tokens and hence minimizing jailbreak risks while keeping the model's responses to safe queries intact. We conduct extensive experiments on two models, four benchmark datasets, and multiple state-of-the-art jailbreak benchmarks to demonstrate the efficacy of our approach. Results indicate that our framework reduces the harmfulness and attack success rate of jailbreak attacks without compromising utility for benign queries compared to recent defense methods.

摘要: 随着大型语言模型(LLM)越来越多地部署在各种应用中，包括聊天机器人助手和代码生成，使它们的行为符合安全和道德标准变得至关重要。然而，越狱攻击利用漏洞来引发意外或有害的输出，严重威胁到LLMS的安全。在本文中，我们介绍了Layer-AdvPatcher，这是一种新的方法，旨在通过一种遗忘策略来通过自增强数据集修补LLMS中的特定层来防御越狱攻击。我们的洞察是，某些层面(S)，在面对有害的提示时，往往会产生肯定的表征。通过识别这些层并恶意暴露它们以生成更多有害数据，人们可以了解它们固有的和不同的攻击漏洞。有了这些暴露，我们就可以“忘掉”这些问题，减少肯定令牌的影响，从而最大限度地减少越狱风险，同时保持模型对安全查询的响应完好无损。我们在两个模型、四个基准数据集和多个最先进的越狱基准上进行了广泛的实验，以展示我们方法的有效性。结果表明，与现有的防御方法相比，该框架降低了越狱攻击的危害性和攻击成功率，而不影响良性查询的有效性。



## **26. DiffusionAttacker: Diffusion-Driven Prompt Manipulation for LLM Jailbreak**

扩散攻击者：LLM越狱的扩散驱动提示操纵 cs.CL

**SubmitDate**: 2025-01-05    [abs](http://arxiv.org/abs/2412.17522v2) [paper-pdf](http://arxiv.org/pdf/2412.17522v2)

**Authors**: Hao Wang, Hao Li, Junda Zhu, Xinyuan Wang, Chengwei Pan, MinLie Huang, Lei Sha

**Abstract**: Large Language Models (LLMs) are susceptible to generating harmful content when prompted with carefully crafted inputs, a vulnerability known as LLM jailbreaking. As LLMs become more powerful, studying jailbreak methods is critical to enhancing security and aligning models with human values. Traditionally, jailbreak techniques have relied on suffix addition or prompt templates, but these methods suffer from limited attack diversity. This paper introduces DiffusionAttacker, an end-to-end generative approach for jailbreak rewriting inspired by diffusion models. Our method employs a sequence-to-sequence (seq2seq) text diffusion model as a generator, conditioning on the original prompt and guiding the denoising process with a novel attack loss. Unlike previous approaches that use autoregressive LLMs to generate jailbreak prompts, which limit the modification of already generated tokens and restrict the rewriting space, DiffusionAttacker utilizes a seq2seq diffusion model, allowing more flexible token modifications. This approach preserves the semantic content of the original prompt while producing harmful content. Additionally, we leverage the Gumbel-Softmax technique to make the sampling process from the diffusion model's output distribution differentiable, eliminating the need for iterative token search. Extensive experiments on Advbench and Harmbench demonstrate that DiffusionAttacker outperforms previous methods across various evaluation metrics, including attack success rate (ASR), fluency, and diversity.

摘要: 当提示使用精心编制的输入时，大型语言模型(LLM)很容易生成有害内容，这一漏洞被称为LLM越狱。随着LLMS变得越来越强大，研究越狱方法对于增强安全性和使模型与人类价值观保持一致至关重要。传统上，越狱技术依赖于后缀添加或提示模板，但这些方法受到攻击多样性的限制。本文介绍了一种受扩散模型启发的端到端生成式越狱重写方法DiffusionAttacker。该方法采用序列到序列(Seq2seq)文本扩散模型作为生成器，以原始提示为条件，以新的攻击损失指导去噪过程。与以前使用自回归LLM生成越狱提示的方法不同，DiffusionAttacker使用seq2seq扩散模型，允许更灵活的令牌修改，从而限制了对已生成令牌的修改并限制了重写空间。这种方法在产生有害内容的同时保留了原始提示的语义内容。此外，我们利用Gumbel-Softmax技术使扩散模型的输出分布的采样过程可微，从而消除了迭代令牌搜索的需要。在Advbench和Harmbench上的大量实验表明，DiffusionAttacker在包括攻击成功率(ASR)、流畅度和多样性在内的各种评估指标上都优于以前的方法。



## **27. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

大型语言模型的人工智能生成文本检测器的实践检验 cs.CL

8 pages. Submitted to NAACL

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2412.05139v2) [paper-pdf](http://arxiv.org/pdf/2412.05139v2)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, GPTID, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.

摘要: 大型语言模型的激增引发了人们对它们滥用的日益担忧，特别是在人工智能生成的文本被错误地归因于人类作者的情况下。机器生成的内容检测器声称可以在各种条件下从任何语言模型有效地识别此类文本。本文通过评估几种流行的探测器(雷达、Wild、T5Sentinel、Fast-DetectGPT、GPTID、logrank、双筒望远镜)，对这些声称进行了批判性的评估，这些探测器以前从未遇到过。我们使用各种提示策略来模拟对抗性攻击，表明即使是适度的攻击也可以显著地躲避检测。我们强调了在特定的假阳性率(TPR@fPR)度量下的真阳性率的重要性，并证明了这些检测器在某些设置下表现很差，TPR@.01低至0%。我们的发现表明，训练有素的探测器和零射探测器都很难在保持高灵敏度的同时获得合理的真阳性率。



## **28. A Survey of Recent Backdoor Attacks and Defenses in Large Language Models**

大型语言模型中最近后门攻击和防御的调查 cs.CR

Accepted in TMLR

**SubmitDate**: 2025-01-04    [abs](http://arxiv.org/abs/2406.06852v5) [paper-pdf](http://arxiv.org/pdf/2406.06852v5)

**Authors**: Shuai Zhao, Meihuizi Jia, Zhongliang Guo, Leilei Gan, Xiaoyu Xu, Xiaobao Wu, Jie Fu, Yichao Feng, Fengjun Pan, Luu Anh Tuan

**Abstract**: Large Language Models (LLMs), which bridge the gap between human language understanding and complex problem-solving, achieve state-of-the-art performance on several NLP tasks, particularly in few-shot and zero-shot settings. Despite the demonstrable efficacy of LLMs, due to constraints on computational resources, users have to engage with open-source language models or outsource the entire training process to third-party platforms. However, research has demonstrated that language models are susceptible to potential security vulnerabilities, particularly in backdoor attacks. Backdoor attacks are designed to introduce targeted vulnerabilities into language models by poisoning training samples or model weights, allowing attackers to manipulate model responses through malicious triggers. While existing surveys on backdoor attacks provide a comprehensive overview, they lack an in-depth examination of backdoor attacks specifically targeting LLMs. To bridge this gap and grasp the latest trends in the field, this paper presents a novel perspective on backdoor attacks for LLMs by focusing on fine-tuning methods. Specifically, we systematically classify backdoor attacks into three categories: full-parameter fine-tuning, parameter-efficient fine-tuning, and no fine-tuning Based on insights from a substantial review, we also discuss crucial issues for future research on backdoor attacks, such as further exploring attack algorithms that do not require fine-tuning, or developing more covert attack algorithms.

摘要: 大型语言模型(LLM)架起了人类语言理解和复杂问题解决之间的桥梁，在几个NLP任务上实现了最先进的性能，特别是在少镜头和零镜头的情况下。尽管LLMS具有明显的功效，但由于计算资源的限制，用户不得不使用开放源码语言模型或将整个培训过程外包给第三方平台。然而，研究表明，语言模型容易受到潜在的安全漏洞的影响，特别是在后门攻击中。后门攻击旨在通过毒化训练样本或模型权重，将有针对性的漏洞引入语言模型，允许攻击者通过恶意触发器操纵模型响应。虽然现有的关于后门攻击的调查提供了全面的概述，但它们缺乏对专门针对LLM的后门攻击的深入检查。为了弥补这一差距，掌握该领域的最新趋势，本文提出了一种新的视角来研究针对LLMS的后门攻击，重点是微调方法。具体地说，我们系统地将后门攻击分为三类：全参数微调、参数高效微调和无微调。在大量综述的基础上，我们还讨论了未来后门攻击研究的关键问题，如进一步探索不需要微调的攻击算法，或开发更隐蔽的攻击算法。



## **29. AVTrustBench: Assessing and Enhancing Reliability and Robustness in Audio-Visual LLMs**

AVTrustBench：评估和增强视听LLM的可靠性和稳健性 cs.CV

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.02135v1) [paper-pdf](http://arxiv.org/pdf/2501.02135v1)

**Authors**: Sanjoy Chowdhury, Sayan Nag, Subhrajyoti Dasgupta, Yaoting Wang, Mohamed Elhoseiny, Ruohan Gao, Dinesh Manocha

**Abstract**: With the rapid advancement of Multi-modal Large Language Models (MLLMs), several diagnostic benchmarks have recently been developed to assess these models' multi-modal reasoning proficiency. However, these benchmarks are restricted to assessing primarily the visual aspect and do not examine the holistic audio-visual (AV) understanding. Moreover, currently, there are no benchmarks that investigate the capabilities of AVLLMs to calibrate their responses when presented with perturbed inputs. To this end, we introduce Audio-Visual Trustworthiness assessment Benchmark (AVTrustBench), comprising 600K samples spanning over 9 meticulously crafted tasks, evaluating the capabilities of AVLLMs across three distinct dimensions: Adversarial attack, Compositional reasoning, and Modality-specific dependency. Using our benchmark we extensively evaluate 13 state-of-the-art AVLLMs. The findings reveal that the majority of existing models fall significantly short of achieving human-like comprehension, offering valuable insights for future research directions. To alleviate the limitations in the existing approaches, we further propose a robust, model-agnostic calibrated audio-visual preference optimization based training strategy CAVPref, obtaining a gain up to 30.19% across all 9 tasks. We will publicly release our code and benchmark to facilitate future research in this direction.

摘要: 随着多通道大型语言模型(MLLMS)的迅速发展，最近出现了几个用于评估这些模型的多通道推理能力的诊断基准。然而，这些基准仅限于主要评估视觉方面，而不检查整体视听(AV)理解。此外，目前还没有基准来调查AVLLMS在收到扰动输入时校准其响应的能力。为此，我们引入了视听可信性评估基准(AVTrustB边)，该基准包括60万个样本，跨越9个精心制作的任务，从三个不同的维度评估AVLLMS的能力：对抗攻击、成分推理和特定于通道的依赖。使用我们的基准，我们广泛评估了13个最先进的AVLLM。研究结果表明，现有的大多数模型都明显不能实现类似人类的理解，为未来的研究方向提供了有价值的见解。为了缓解现有方法的局限性，我们进一步提出了一种稳健的、与模型无关的、基于校准视听偏好优化的训练策略CAVPref，在所有9个任务中都获得了高达30.19%的收益。我们将公开发布我们的代码和基准，以促进未来在这一方向的研究。



## **30. Auto-RT: Automatic Jailbreak Strategy Exploration for Red-Teaming Large Language Models**

Auto-RT：Red-Teaming大型语言模型的自动越狱策略探索 cs.CR

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01830v1) [paper-pdf](http://arxiv.org/pdf/2501.01830v1)

**Authors**: Yanjiang Liu, Shuhen Zhou, Yaojie Lu, Huijia Zhu, Weiqiang Wang, Hongyu Lin, Ben He, Xianpei Han, Le Sun

**Abstract**: Automated red-teaming has become a crucial approach for uncovering vulnerabilities in large language models (LLMs). However, most existing methods focus on isolated safety flaws, limiting their ability to adapt to dynamic defenses and uncover complex vulnerabilities efficiently. To address this challenge, we propose Auto-RT, a reinforcement learning framework that automatically explores and optimizes complex attack strategies to effectively uncover security vulnerabilities through malicious queries. Specifically, we introduce two key mechanisms to reduce exploration complexity and improve strategy optimization: 1) Early-terminated Exploration, which accelerate exploration by focusing on high-potential attack strategies; and 2) Progressive Reward Tracking algorithm with intermediate downgrade models, which dynamically refine the search trajectory toward successful vulnerability exploitation. Extensive experiments across diverse LLMs demonstrate that, by significantly improving exploration efficiency and automatically optimizing attack strategies, Auto-RT detects a boarder range of vulnerabilities, achieving a faster detection speed and 16.63\% higher success rates compared to existing methods.

摘要: 自动红团队已成为发现大型语言模型(LLM)中漏洞的关键方法。然而，现有的大多数方法都集中在孤立的安全漏洞上，限制了它们适应动态防御和有效发现复杂漏洞的能力。为了应对这一挑战，我们提出了Auto-RT，这是一个强化学习框架，它自动探索和优化复杂的攻击策略，通过恶意查询有效地发现安全漏洞。具体地说，我们引入了两个关键机制来降低探测复杂性和改善策略优化：1)提前终止探测，通过关注高潜在攻击策略来加速探测；2)采用中间降级模型的渐进式奖励跟踪算法，动态细化搜索轨迹，以实现成功的漏洞攻击。大量的实验表明，通过显著提高探测效率和自动优化攻击策略，Auto-RT可以检测到更广泛的漏洞，与现有方法相比，检测速度更快，成功率更高。



## **31. How Toxic Can You Get? Search-based Toxicity Testing for Large Language Models**

你能得到多大的毒性？基于搜索的大型语言模型毒性测试 cs.SE

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01741v1) [paper-pdf](http://arxiv.org/pdf/2501.01741v1)

**Authors**: Simone Corbo, Luca Bancale, Valeria De Gennaro, Livia Lestingi, Vincenzo Scotti, Matteo Camilli

**Abstract**: Language is a deep-rooted means of perpetration of stereotypes and discrimination. Large Language Models (LLMs), now a pervasive technology in our everyday lives, can cause extensive harm when prone to generating toxic responses. The standard way to address this issue is to align the LLM, which, however, dampens the issue without constituting a definitive solution. Therefore, testing LLM even after alignment efforts remains crucial for detecting any residual deviations with respect to ethical standards. We present EvoTox, an automated testing framework for LLMs' inclination to toxicity, providing a way to quantitatively assess how much LLMs can be pushed towards toxic responses even in the presence of alignment. The framework adopts an iterative evolution strategy that exploits the interplay between two LLMs, the System Under Test (SUT) and the Prompt Generator steering SUT responses toward higher toxicity. The toxicity level is assessed by an automated oracle based on an existing toxicity classifier. We conduct a quantitative and qualitative empirical evaluation using four state-of-the-art LLMs as evaluation subjects having increasing complexity (7-13 billion parameters). Our quantitative evaluation assesses the cost-effectiveness of four alternative versions of EvoTox against existing baseline methods, based on random search, curated datasets of toxic prompts, and adversarial attacks. Our qualitative assessment engages human evaluators to rate the fluency of the generated prompts and the perceived toxicity of the responses collected during the testing sessions. Results indicate that the effectiveness, in terms of detected toxicity level, is significantly higher than the selected baseline methods (effect size up to 1.0 against random search and up to 0.99 against adversarial attacks). Furthermore, EvoTox yields a limited cost overhead (from 22% to 35% on average).

摘要: 语言是制造陈规定型观念和歧视的根深蒂固的手段。大语言模型(LLM)现在是我们日常生活中的一项普遍技术，当它容易产生有毒反应时，可能会造成广泛的危害。解决这一问题的标准方法是调整LLM，然而，这会抑制问题，而不会构成最终的解决方案。因此，即使在调整工作之后进行LLM测试，对于检测与道德标准有关的任何残余偏差仍然至关重要。我们提出了EvoTox，这是一个用于LLMS毒性倾向的自动化测试框架，提供了一种方法来定量评估即使在存在对齐的情况下，LLMS也可以被推向毒性反应的程度。该框架采用了一种迭代进化策略，利用了两个LLM之间的相互作用，被测系统(SUT)和即时生成器将SUT的响应转向更高的毒性。毒性水平由基于现有毒性分类器的自动先知进行评估。我们使用四个最先进的LLM作为评估对象，进行了定量和定性的实证评估，这些评估对象的复杂性越来越高(参数为70-130亿个)。我们的定量评估基于随机搜索、有毒提示的精选数据集和对抗性攻击，相对于现有的基线方法评估了EvoTox的四个替代版本的成本效益。我们的定性评估聘请人类评估员对生成的提示的流畅性和在测试期间收集的回答的感知毒性进行评级。结果表明，就检测到的毒性水平而言，该方法的有效性显著高于所选的基线方法(对随机搜索的效果大小高达1.0，对对抗性攻击的效果高达0.99)。此外，EvoTox产生的成本管理费用有限(平均从22%到35%)。



## **32. Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models**

启发式多峰大语言模型的多峰风险分布越狱攻击 cs.CR

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2412.05934v2) [paper-pdf](http://arxiv.org/pdf/2412.05934v2)

**Authors**: Ma Teng, Jia Xiaojun, Duan Ranjie, Li Xinfeng, Huang Yihao, Chu Zhixuan, Liu Yang, Ren Wenqi

**Abstract**: With the rapid advancement of multimodal large language models (MLLMs), concerns regarding their security have increasingly captured the attention of both academia and industry. Although MLLMs are vulnerable to jailbreak attacks, designing effective multimodal jailbreak attacks poses unique challenges, especially given the distinct protective measures implemented across various modalities in commercial models. Previous works concentrate risks into a single modality, resulting in limited jailbreak performance. In this paper, we propose a heuristic-induced multimodal risk distribution jailbreak attack method, called HIMRD, which consists of two elements: multimodal risk distribution strategy and heuristic-induced search strategy. The multimodal risk distribution strategy is used to segment harmful instructions across multiple modalities to effectively circumvent MLLMs' security protection. The heuristic-induced search strategy identifies two types of prompts: the understanding-enhancing prompt, which helps the MLLM reconstruct the malicious prompt, and the inducing prompt, which increases the likelihood of affirmative outputs over refusals, enabling a successful jailbreak attack. Extensive experiments demonstrate that this approach effectively uncovers vulnerabilities in MLLMs, achieving an average attack success rate of 90% across seven popular open-source MLLMs and an average attack success rate of around 68% in three popular closed-source MLLMs. Our code will coming soon. Warning: This paper contains offensive and harmful examples, reader discretion is advised.

摘要: 随着多通道大语言模型的快速发展，对其安全性的关注日益引起学术界和工业界的关注。尽管大规模杀伤性武器易受越狱攻击，但设计有效的多模式越狱攻击是一个独特的挑战，特别是考虑到商业模式中的各种模式实施了不同的保护措施。以前的工作将风险集中到单一模式中，导致有限的越狱性能。本文提出了一种启发式多通道风险分布越狱攻击方法HIMRD，该方法由两部分组成：多通道风险分布策略和启发式搜索策略。多模式风险分配策略用于跨多个模式分割有害指令，以有效规避MLLMS的安全保护。启发式搜索策略识别两种类型的提示：促进理解的提示和诱导性提示，前者帮助MLLM重建恶意提示，后者增加肯定输出超过拒绝的可能性，从而实现成功的越狱攻击。大量实验表明，该方法有效地发现了MLLMS中的漏洞，在七个流行的开源MLLMS上的平均攻击成功率达到了90%，在三个流行的闭源MLLMS上的平均攻击成功率达到了68%左右。我们的代码很快就会出来。警告：本文包含冒犯性和有害的例子，建议读者酌情处理。



## **33. Spot Risks Before Speaking! Unraveling Safety Attention Heads in Large Vision-Language Models**

说话前现货风险！解开大型视觉语言模型中的安全注意力 cs.LG

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.02029v1) [paper-pdf](http://arxiv.org/pdf/2501.02029v1)

**Authors**: Ziwei Zheng, Junyao Zhao, Le Yang, Lijun He, Fan Li

**Abstract**: With the integration of an additional modality, large vision-language models (LVLMs) exhibit greater vulnerability to safety risks (e.g., jailbreaking) compared to their language-only predecessors. Although recent studies have devoted considerable effort to the post-hoc alignment of LVLMs, the inner safety mechanisms remain largely unexplored. In this paper, we discover that internal activations of LVLMs during the first token generation can effectively identify malicious prompts across different attacks. This inherent safety perception is governed by sparse attention heads, which we term ``safety heads." Further analysis reveals that these heads act as specialized shields against malicious prompts; ablating them leads to higher attack success rates, while the model's utility remains unaffected. By locating these safety heads and concatenating their activations, we construct a straightforward but powerful malicious prompt detector that integrates seamlessly into the generation process with minimal extra inference overhead. Despite its simple structure of a logistic regression model, the detector surprisingly exhibits strong zero-shot generalization capabilities. Experiments across various prompt-based attacks confirm the effectiveness of leveraging safety heads to protect LVLMs. Code is available at \url{https://github.com/Ziwei-Zheng/SAHs}.

摘要: 随着另一种模式的集成，与仅使用语言的前身相比，大型视觉语言模型(LVLM)显示出更大的安全风险(例如越狱)。尽管最近的研究已经在左肺小梁的术后配对上投入了相当大的努力，但其内部的安全机制在很大程度上仍未被探索。在本文中，我们发现在第一次令牌生成过程中内部激活LVLM可以有效地识别跨不同攻击的恶意提示。这种固有的安全感是由稀疏的注意力头部所支配的，我们称之为“安全头部”。进一步的分析表明，这些头部作为专门的盾牌来抵御恶意提示；消除它们会导致更高的攻击成功率，而该模型的实用性不会受到影响。通过定位这些安全头并连接它们的激活，我们构建了一个简单但强大的恶意提示检测器，该检测器无缝地集成到生成过程中，并且具有最小的额外推理开销。尽管它的结构简单的Logistic回归模型，但该检测器出人意料地显示出强大的零射泛化能力。各种基于提示的攻击的实验证实了利用安全头来保护LVLM的有效性。代码位于\url{https://github.com/Ziwei-Zheng/SAHs}.



## **34. BARTPredict: Empowering IoT Security with LLM-Driven Cyber Threat Prediction**

BartPredict：通过LLM驱动的网络威胁预测增强物联网安全性 cs.CR

**SubmitDate**: 2025-01-03    [abs](http://arxiv.org/abs/2501.01664v1) [paper-pdf](http://arxiv.org/pdf/2501.01664v1)

**Authors**: Alaeddine Diaf, Abdelaziz Amara Korba, Nour Elislem Karabadji, Yacine Ghamri-Doudane

**Abstract**: The integration of Internet of Things (IoT) technology in various domains has led to operational advancements, but it has also introduced new vulnerabilities to cybersecurity threats, as evidenced by recent widespread cyberattacks on IoT devices. Intrusion detection systems are often reactive, triggered by specific patterns or anomalies observed within the network. To address this challenge, this work proposes a proactive approach to anticipate and preemptively mitigate malicious activities, aiming to prevent potential damage before it occurs. This paper proposes an innovative intrusion prediction framework empowered by Pre-trained Large Language Models (LLMs). The framework incorporates two LLMs: a fine-tuned Bidirectional and AutoRegressive Transformers (BART) model for predicting network traffic and a fine-tuned Bidirectional Encoder Representations from Transformers (BERT) model for evaluating the predicted traffic. By harnessing the bidirectional capabilities of BART the framework then identifies malicious packets among these predictions. Evaluated using the CICIoT2023 IoT attack dataset, our framework showcases a notable enhancement in predictive performance, attaining an impressive 98% overall accuracy, providing a powerful response to the cybersecurity challenges that confront IoT networks.

摘要: 物联网(IoT)技术在各个领域的整合带来了运营上的进步，但也给网络安全威胁带来了新的漏洞，最近针对物联网设备的广泛网络攻击就是明证。入侵检测系统通常是被动的，由网络中观察到的特定模式或异常触发。为了应对这一挑战，这项工作提出了一种积极主动的方法来预测和先发制人地减轻恶意活动，旨在防止潜在的损害发生之前。提出了一种基于预训练的大语言模型的入侵预测框架。该框架包含两个LLM：用于预测网络流量的微调双向和自回归转换器(BART)模型和用于评估预测流量的微调双向编码器表示(BERT)模型。通过利用BART的双向功能，该框架可以在这些预测中识别恶意数据包。使用CICIoT2023物联网攻击数据集进行评估，我们的框架在预测性能方面有了显著的增强，总体准确率达到了令人印象深刻的98%，为物联网网络面临的网络安全挑战提供了强大的响应。



## **35. CySecBench: Generative AI-based CyberSecurity-focused Prompt Dataset for Benchmarking Large Language Models**

CySecBench：基于人工智能、以网络安全为重点的生成性提示数据集，用于对大型语言模型进行基准测试 cs.CR

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2501.01335v1) [paper-pdf](http://arxiv.org/pdf/2501.01335v1)

**Authors**: Johan Wahréus, Ahmed Mohamed Hussain, Panos Papadimitratos

**Abstract**: Numerous studies have investigated methods for jailbreaking Large Language Models (LLMs) to generate harmful content. Typically, these methods are evaluated using datasets of malicious prompts designed to bypass security policies established by LLM providers. However, the generally broad scope and open-ended nature of existing datasets can complicate the assessment of jailbreaking effectiveness, particularly in specific domains, notably cybersecurity. To address this issue, we present and publicly release CySecBench, a comprehensive dataset containing 12662 prompts specifically designed to evaluate jailbreaking techniques in the cybersecurity domain. The dataset is organized into 10 distinct attack-type categories, featuring close-ended prompts to enable a more consistent and accurate assessment of jailbreaking attempts. Furthermore, we detail our methodology for dataset generation and filtration, which can be adapted to create similar datasets in other domains. To demonstrate the utility of CySecBench, we propose and evaluate a jailbreaking approach based on prompt obfuscation. Our experimental results show that this method successfully elicits harmful content from commercial black-box LLMs, achieving Success Rates (SRs) of 65% with ChatGPT and 88% with Gemini; in contrast, Claude demonstrated greater resilience with a jailbreaking SR of 17%. Compared to existing benchmark approaches, our method shows superior performance, highlighting the value of domain-specific evaluation datasets for assessing LLM security measures. Moreover, when evaluated using prompts from a widely used dataset (i.e., AdvBench), it achieved an SR of 78.5%, higher than the state-of-the-art methods.

摘要: 许多研究已经调查了越狱大型语言模型(LLM)生成有害内容的方法。通常，使用恶意提示的数据集来评估这些方法，这些恶意提示旨在绕过LLM提供商建立的安全策略。然而，现有数据集的广泛范围和开放式性质可能会使越狱效果的评估复杂化，特别是在特定领域，特别是网络安全领域。为了解决这个问题，我们提出并公开发布了CySecBitch，这是一个全面的数据集，包含12662个提示，专门用于评估网络安全领域的越狱技术。该数据集被组织成10个不同的攻击类型类别，具有封闭式提示，以实现对越狱企图的更一致和更准确的评估。此外，我们详细介绍了我们的数据集生成和过滤方法，该方法可以适用于在其他领域创建类似的数据集。为了证明CySecBitch的有效性，我们提出并评估了一种基于即时混淆的越狱方法。我们的实验结果表明，该方法成功地从商业黑盒LLMS中提取出有害内容，ChatGPT的成功率(SRS)为65%，Gemini为88%；相比之下，Claude表现出更强的弹性，越狱成功率为17%。与现有的基准测试方法相比，我们的方法表现出更好的性能，突出了特定于域的评估数据集在评估LLM安全措施方面的价值。此外，当使用广泛使用的数据集(即AdvBch)的提示进行评估时，它获得了78.5%的SR，高于最先进的方法。



## **36. Safeguarding Large Language Models in Real-time with Tunable Safety-Performance Trade-offs**

通过可调的安全性能权衡实时保护大型语言模型 cs.CL

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2501.02018v1) [paper-pdf](http://arxiv.org/pdf/2501.02018v1)

**Authors**: Joao Fonseca, Andrew Bell, Julia Stoyanovich

**Abstract**: Large Language Models (LLMs) have been shown to be susceptible to jailbreak attacks, or adversarial attacks used to illicit high risk behavior from a model. Jailbreaks have been exploited by cybercriminals and blackhat actors to cause significant harm, highlighting the critical need to safeguard widely-deployed models. Safeguarding approaches, which include fine-tuning models or having LLMs "self-reflect", may lengthen the inference time of a model, incur a computational penalty, reduce the semantic fluency of an output, and restrict ``normal'' model behavior. Importantly, these Safety-Performance Trade-offs (SPTs) remain an understudied area. In this work, we introduce a novel safeguard, called SafeNudge, that combines Controlled Text Generation with "nudging", or using text interventions to change the behavior of a model. SafeNudge triggers during text-generation while a jailbreak attack is being executed, and can reduce successful jailbreak attempts by 30% by guiding the LLM towards a safe responses. It adds minimal latency to inference and has a negligible impact on the semantic fluency of outputs. Further, we allow for tunable SPTs. SafeNudge is open-source and available through https://pypi.org/, and is compatible with models loaded with the Hugging Face "transformers" library.

摘要: 大型语言模型(LLM)已被证明容易受到越狱攻击，即用于从模型中非法进行高风险行为的对抗性攻击。越狱已被网络犯罪分子和黑帽行为者利用，造成重大危害，突显出保护广泛部署的模型的迫切需要。保护方法，包括微调模型或让LLM“自我反思”，可能会延长模型的推理时间，招致计算惩罚，降低输出的语义流畅性，并限制“正常”的模型行为。重要的是，这些安全-性能权衡(SPTS)仍然是一个研究较少的领域。在这项工作中，我们引入了一种新的安全措施，称为安全轻推，它将受控文本生成与“轻推”相结合，即使用文本干预来改变模型的行为。安全轻推在执行越狱攻击时在文本生成过程中触发，通过引导LLM进行安全响应，可以将成功的越狱尝试减少30%。它增加了最小的推理延迟，并且对输出的语义流畅性的影响可以忽略不计。此外，我们还考虑了可调SPT。SafeNdge是开源的，可以通过https://pypi.org/，获得，并且与装载了拥抱脸“变形金刚”库的模型兼容。



## **37. Security Attacks on LLM-based Code Completion Tools**

对基于LLM的代码完成工具的安全攻击 cs.CL

Paper accepted at AAAI 2025

**SubmitDate**: 2025-01-02    [abs](http://arxiv.org/abs/2408.11006v4) [paper-pdf](http://arxiv.org/pdf/2408.11006v4)

**Authors**: Wen Cheng, Ke Sun, Xinyu Zhang, Wei Wang

**Abstract**: The rapid development of large language models (LLMs) has significantly advanced code completion capabilities, giving rise to a new generation of LLM-based Code Completion Tools (LCCTs). Unlike general-purpose LLMs, these tools possess unique workflows, integrating multiple information sources as input and prioritizing code suggestions over natural language interaction, which introduces distinct security challenges. Additionally, LCCTs often rely on proprietary code datasets for training, raising concerns about the potential exposure of sensitive data. This paper exploits these distinct characteristics of LCCTs to develop targeted attack methodologies on two critical security risks: jailbreaking and training data extraction attacks. Our experimental results expose significant vulnerabilities within LCCTs, including a 99.4% success rate in jailbreaking attacks on GitHub Copilot and a 46.3% success rate on Amazon Q. Furthermore, We successfully extracted sensitive user data from GitHub Copilot, including 54 real email addresses and 314 physical addresses associated with GitHub usernames. Our study also demonstrates that these code-based attack methods are effective against general-purpose LLMs, such as the GPT series, highlighting a broader security misalignment in the handling of code by modern LLMs. These findings underscore critical security challenges associated with LCCTs and suggest essential directions for strengthening their security frameworks. The example code and attack samples from our research are provided at https://github.com/Sensente/Security-Attacks-on-LCCTs.

摘要: 大型语言模型(LLM)的快速发展极大地提升了代码补全能力，催生了新一代基于LLM的代码补全工具(LCCT)。与通用的LLMS不同，这些工具拥有独特的工作流，将多个信息源集成为输入，并优先考虑代码建议而不是自然语言交互，这带来了明显的安全挑战。此外，LCCT经常依赖专有代码数据集进行培训，这引发了人们对敏感数据潜在暴露的担忧。针对越狱攻击和训练数据提取攻击这两个关键安全风险，本文利用LCCT的这些显著特点，提出了针对性的攻击方法。我们的实验结果暴露了LCCT中的重大漏洞，包括对GitHub Copilot的越狱攻击成功率为99.4%，对Amazon Q的成功率为46.3%。此外，我们成功地从GitHub Copilot中提取了敏感用户数据，包括与GitHub用户名关联的54个真实电子邮件地址和314个物理地址。我们的研究还表明，这些基于代码的攻击方法对通用LLM是有效的，例如GPT系列，突显了现代LLM在处理代码时存在更广泛的安全错位。这些调查结果强调了与土地利用、土地利用、土地退化和土地退化有关的重大安全挑战，并提出了加强其安全框架的基本方向。我们的研究提供了示例代码和攻击示例，请访问https://github.com/Sensente/Security-Attacks-on-LCCTs.



## **38. TrustRAG: Enhancing Robustness and Trustworthiness in RAG**

TrustRAG：增强RAG的稳健性和可信性 cs.CL

**SubmitDate**: 2025-01-01    [abs](http://arxiv.org/abs/2501.00879v1) [paper-pdf](http://arxiv.org/pdf/2501.00879v1)

**Authors**: Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen, Zhenhao Li

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge sources, enabling more accurate and contextually relevant responses tailored to user queries. However, these systems remain vulnerable to corpus poisoning attacks that can significantly degrade LLM performance through the injection of malicious content. To address these challenges, we propose TrustRAG, a robust framework that systematically filters compromised and irrelevant content before it reaches the language model. Our approach implements a two-stage defense mechanism: first, it employs K-means clustering to identify potential attack patterns in retrieved documents based on their semantic embeddings, effectively isolating suspicious content. Second, it leverages cosine similarity and ROUGE metrics to detect malicious documents while resolving discrepancies between the model's internal knowledge and external information through a self-assessment process. TrustRAG functions as a plug-and-play, training-free module that integrates seamlessly with any language model, whether open or closed-source, maintaining high contextual relevance while strengthening defenses against attacks. Through extensive experimental validation, we demonstrate that TrustRAG delivers substantial improvements in retrieval accuracy, efficiency, and attack resistance compared to existing approaches across multiple model architectures and datasets. We have made TrustRAG available as open-source software at \url{https://github.com/HuichiZhou/TrustRAG}.

摘要: 检索-增强生成(RAG)系统通过集成外部知识源来增强大型语言模型(LLM)，从而能够针对用户查询定制更准确且与上下文相关的响应。然而，这些系统仍然容易受到语料库中毒攻击，这些攻击可能会通过注入恶意内容显著降低LLM的性能。为了应对这些挑战，我们提出了TrustRAG，这是一个健壮的框架，在受到攻击和无关的内容到达语言模型之前系统地过滤它们。该方法实现了一种两阶段防御机制：首先，利用K-均值聚类，根据文档的语义嵌入来识别潜在的攻击模式，有效地隔离可疑内容。其次，它利用余弦相似度和Rouge度量来检测恶意文档，同时通过自我评估过程解决模型内部知识和外部信息之间的差异。TrustRAG是一个即插即用、无需培训的模块，可以与任何语言模型无缝集成，无论是开放还是封闭源代码，在加强对攻击的防御的同时保持高度的上下文相关性。通过广泛的实验验证，我们证明了TrustRAG在检索准确性、效率和抗攻击方面比现有的跨多个模型架构和数据集的方法有了实质性的改进。我们已将TrustRAG作为开源软件提供给\url{https://github.com/HuichiZhou/TrustRAG}.



## **39. Dynamics of Adversarial Attacks on Large Language Model-Based Search Engines**

基于大型语言模型的搜索引擎的对抗性攻击动态 cs.CL

**SubmitDate**: 2025-01-01    [abs](http://arxiv.org/abs/2501.00745v1) [paper-pdf](http://arxiv.org/pdf/2501.00745v1)

**Authors**: Xiyang Hu

**Abstract**: The increasing integration of Large Language Model (LLM) based search engines has transformed the landscape of information retrieval. However, these systems are vulnerable to adversarial attacks, especially ranking manipulation attacks, where attackers craft webpage content to manipulate the LLM's ranking and promote specific content, gaining an unfair advantage over competitors. In this paper, we study the dynamics of ranking manipulation attacks. We frame this problem as an Infinitely Repeated Prisoners' Dilemma, where multiple players strategically decide whether to cooperate or attack. We analyze the conditions under which cooperation can be sustained, identifying key factors such as attack costs, discount rates, attack success rates, and trigger strategies that influence player behavior. We identify tipping points in the system dynamics, demonstrating that cooperation is more likely to be sustained when players are forward-looking. However, from a defense perspective, we find that simply reducing attack success probabilities can, paradoxically, incentivize attacks under certain conditions. Furthermore, defensive measures to cap the upper bound of attack success rates may prove futile in some scenarios. These insights highlight the complexity of securing LLM-based systems. Our work provides a theoretical foundation and practical insights for understanding and mitigating their vulnerabilities, while emphasizing the importance of adaptive security strategies and thoughtful ecosystem design.

摘要: 基于大型语言模型(LLM)的搜索引擎的日益集成已经改变了信息检索的格局。然而，这些系统容易受到对抗性攻击，特别是排名操纵攻击，攻击者精心编制网页内容来操纵LLM的排名并推广特定内容，从而获得相对于竞争对手的不公平优势。本文研究了排名操纵攻击的动态特性。我们将这个问题描述为一个无限重复的囚徒困境，其中多个参与者战略性地决定是合作还是攻击。我们分析了合作能够持续的条件，确定了影响玩家行为的关键因素，如攻击成本、折扣率、攻击成功率和触发策略。我们确定了系统动态中的引爆点，表明当参与者具有前瞻性时，合作更有可能持续下去。然而，从防御的角度来看，我们发现，矛盾的是，仅仅降低攻击成功的概率就可以在某些条件下激励攻击。此外，在某些情况下，为攻击成功率上限设定上限的防御措施可能被证明是徒劳的。这些见解突显了保护基于LLM的系统的复杂性。我们的工作为理解和缓解它们的漏洞提供了理论基础和实践见解，同时强调了自适应安全策略和深思熟虑的生态系统设计的重要性。



## **40. From Sands to Mansions: Simulating Full Attack Chain with LLM-Organized Knowledge**

从金沙到豪宅：利用法学硕士组织的知识模拟完整攻击链 cs.CR

**SubmitDate**: 2024-12-31    [abs](http://arxiv.org/abs/2407.16928v2) [paper-pdf](http://arxiv.org/pdf/2407.16928v2)

**Authors**: Lingzhi Wang, Zhenyuan Li, Zonghan Guo, Yi Jiang, Kyle Jung, Kedar Thiagarajan, Jiahui Wang, Zhengkai Wang, Emily Wei, Xiangmin Shen, Yan Chen

**Abstract**: Adversarial dynamics are intrinsic to the nature of offense and defense in cyberspace, with both attackers and defenders continuously evolving their technologies. Given the wide array of security products available, users often face challenges in selecting the most effective solutions. Furthermore, traditional benchmarks based on single-point attacks are increasingly inadequate, failing to accurately reflect the full range of attacker capabilities and falling short in properly evaluating the effectiveness of defense products. Automated multi-stage attack simulations offer a promising approach to enhance system evaluation efficiency and aid in analyzing the effectiveness of detection systems. However, simulating a full attack chain is complex and requires significant time and expertise from security professionals, facing several challenges, including limited coverage of attack techniques, a high level of required expertise, and a lack of execution detail. In this paper, we model automatic attack simulation as a planning problem. By using the Planning Domain Definition Language (PDDL) to formally describe the attack simulation problem, and combining domain knowledge of both the problem and the domain space, we enable the planning of attack paths through standardized, domain-independent planning algorithms. We explore the potential of Large Language Models (LLMs) to summarize and analyze knowledge from existing attack documentation and reports, facilitating automated attack planning. We introduce Aurora, a system that autonomously simulates full attack chains based on external attack tools and threat intelligence reports.

摘要: 对抗动态是网络空间进攻和防御的本质所固有的，攻击者和防御者都在不断地发展他们的技术。鉴于可用的安全产品种类繁多，用户在选择最有效的解决方案时经常面临挑战。此外，基于单点攻击的传统基准日益不足，无法准确反映攻击者的全方位能力，无法正确评估防御产品的有效性。自动多阶段攻击模拟为提高系统评估效率和辅助分析检测系统的有效性提供了一种很有前途的方法。然而，模拟完整的攻击链是复杂的，需要大量的时间和安全专业人员的专业知识，面临着几个挑战，包括攻击技术的覆盖范围有限，所需专业知识水平较高，以及缺乏执行细节。在本文中，我们将自动攻击模拟建模为一个规划问题。通过使用规划领域定义语言(PDDL)对攻击模拟问题进行形式化描述，并结合问题和领域空间的领域知识，我们能够通过标准化的、与领域无关的规划算法来规划攻击路径。我们探索大型语言模型(LLM)的潜力，以总结和分析现有攻击文档和报告中的知识，从而促进自动攻击规划。我们介绍了Aurora，这是一个基于外部攻击工具和威胁情报报告自主模拟完整攻击链的系统。



## **41. Toward Intelligent and Secure Cloud: Large Language Model Empowered Proactive Defense**

迈向智能和安全的云：大型语言模型增强主动防御 cs.CR

7 pages; In submission

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.21051v1) [paper-pdf](http://arxiv.org/pdf/2412.21051v1)

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen

**Abstract**: The rapid evolution of cloud computing technologies and the increasing number of cloud applications have provided a large number of benefits in daily lives. However, the diversity and complexity of different components pose a significant challenge to cloud security, especially when dealing with sophisticated and advanced cyberattacks. Recent advancements in generative foundation models (GFMs), particularly in the large language models (LLMs), offer promising solutions for security intelligence. By exploiting the powerful abilities in language understanding, data analysis, task inference, action planning, and code generation, we present LLM-PD, a novel proactive defense architecture that defeats various threats in a proactive manner. LLM-PD can efficiently make a decision through comprehensive data analysis and sequential reasoning, as well as dynamically creating and deploying actionable defense mechanisms on the target cloud. Furthermore, it can flexibly self-evolve based on experience learned from previous interactions and adapt to new attack scenarios without additional training. The experimental results demonstrate its remarkable ability in terms of defense effectiveness and efficiency, particularly highlighting an outstanding success rate when compared with other existing methods.

摘要: 云计算技术的快速发展和越来越多的云应用为日常生活提供了大量的好处。然而，不同组件的多样性和复杂性对云安全构成了重大挑战，尤其是在处理复杂和高级的网络攻击时。生成性基础模型(GFMS)，特别是大型语言模型(LLM)的最新进展，为安全智能提供了有前途的解决方案。通过利用LLM-PD在语言理解、数据分析、任务推理、动作规划和代码生成等方面的强大能力，我们提出了一种新型的主动防御体系结构LLM-PD，它能够主动地击败各种威胁。LLM-PD通过全面的数据分析和时序推理，以及在目标云上动态创建和部署可操作的防御机制，能够高效地做出决策。此外，它可以根据从以前交互中学习的经验灵活地自我进化，并适应新的攻击场景，而不需要额外的培训。实验结果表明，该方法在防御效果和效率方面具有显著的能力，特别是与现有的其他方法相比，具有突出的成功率。



## **42. Unsupervised dense retrieval with conterfactual contrastive learning**

具有反事实对比学习的无监督密集检索 cs.IR

arXiv admin note: text overlap with arXiv:2107.07773 by other authors

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20756v1) [paper-pdf](http://arxiv.org/pdf/2412.20756v1)

**Authors**: Haitian Chen, Qingyao Ai, Xiao Wang, Yiqun Liu, Fen Lin, Qin Liu

**Abstract**: Efficiently retrieving a concise set of candidates from a large document corpus remains a pivotal challenge in Information Retrieval (IR). Neural retrieval models, particularly dense retrieval models built with transformers and pretrained language models, have been popular due to their superior performance. However, criticisms have also been raised on their lack of explainability and vulnerability to adversarial attacks. In response to these challenges, we propose to improve the robustness of dense retrieval models by enhancing their sensitivity of fine-graned relevance signals. A model achieving sensitivity in this context should exhibit high variances when documents' key passages determining their relevance to queries have been modified, while maintaining low variances for other changes in irrelevant passages. This sensitivity allows a dense retrieval model to produce robust results with respect to attacks that try to promote documents without actually increasing their relevance. It also makes it possible to analyze which part of a document is actually relevant to a query, and thus improve the explainability of the retrieval model. Motivated by causality and counterfactual analysis, we propose a series of counterfactual regularization methods based on game theory and unsupervised learning with counterfactual passages. Experiments show that, our method can extract key passages without reliance on the passage-level relevance annotations. Moreover, the regularized dense retrieval models exhibit heightened robustness against adversarial attacks, surpassing the state-of-the-art anti-attack methods.

摘要: 从大型文档语料库中高效地检索一组简明的候选对象仍然是信息检索(IR)中的一个关键挑战。神经检索模型，特别是用转换器构建的密集检索模型和预先训练的语言模型，由于其优越的性能而受到广泛的欢迎。然而，也有人批评说，它们缺乏可解释性，容易受到对手攻击。为了应对这些挑战，我们提出了通过提高密集检索模型对细粒度关联信号的敏感度来提高其稳健性。在这种情况下实现敏感性的模型应该在确定其与查询的相关性的文档的关键段落被修改时表现出高方差，同时保持对不相关段落中的其他变化的低方差。这种敏感度使得密集检索模型能够针对试图提升文档而不实际增加其相关性的攻击产生稳健的结果。它还可以分析文档的哪个部分实际上与查询相关，从而提高检索模型的可解释性。在因果关系和反事实分析的启发下，我们提出了一系列基于博弈论和带有反事实段落的无监督学习的反事实正则化方法。实验表明，我们的方法可以在不依赖于段落级关联标注的情况下提取关键段落。此外，正则化的密集检索模型表现出对对手攻击的高度稳健性，超过了最先进的反攻击方法。



## **43. SafeSynthDP: Leveraging Large Language Models for Privacy-Preserving Synthetic Data Generation Using Differential Privacy**

SafeSynthDP：利用大型语言模型使用差异隐私生成隐私保护合成数据 cs.LG

15 pages, 1 figure, 5 tables

**SubmitDate**: 2024-12-30    [abs](http://arxiv.org/abs/2412.20641v1) [paper-pdf](http://arxiv.org/pdf/2412.20641v1)

**Authors**: Md Mahadi Hasan Nahid, Sadid Bin Hasan

**Abstract**: Machine learning (ML) models frequently rely on training data that may include sensitive or personal information, raising substantial privacy concerns. Legislative frameworks such as the General Data Protection Regulation (GDPR) and the California Consumer Privacy Act (CCPA) have necessitated the development of strategies that preserve privacy while maintaining the utility of data. In this paper, we investigate the capability of Large Language Models (LLMs) to generate synthetic datasets integrated with Differential Privacy (DP) mechanisms, thereby enabling data-driven research and model training without direct exposure of sensitive information. Our approach incorporates DP-based noise injection methods, including Laplace and Gaussian distributions, into the data generation process. We then evaluate the utility of these DP-enhanced synthetic datasets by comparing the performance of ML models trained on them against models trained on the original data. To substantiate privacy guarantees, we assess the resilience of the generated synthetic data to membership inference attacks and related threats. The experimental results demonstrate that integrating DP within LLM-driven synthetic data generation offers a viable balance between privacy protection and data utility. This study provides a foundational methodology and insight into the privacy-preserving capabilities of LLMs, paving the way for compliant and effective ML research and applications.

摘要: 机器学习(ML)模型经常依赖于可能包括敏感或个人信息的训练数据，这引发了大量的隐私问题。《一般数据保护条例》(GDPR)和《加州消费者隐私法》(CCPA)等立法框架要求制定在保持数据效用的同时保护隐私的战略。在本文中，我们研究了大型语言模型(LLM)生成集成了差异隐私(DP)机制的合成数据集的能力，从而在不直接暴露敏感信息的情况下实现了数据驱动的研究和模型训练。我们的方法将基于DP的噪声注入方法，包括拉普拉斯分布和高斯分布，融入到数据生成过程中。然后，我们通过比较在这些数据集上训练的ML模型和在原始数据上训练的模型的性能来评估这些DP增强的合成数据集的实用性。为了证实隐私保证，我们评估了生成的合成数据对成员资格推断攻击和相关威胁的弹性。实验结果表明，在LLM驱动的合成数据生成中集成DP提供了隐私保护和数据效用之间的可行平衡。这项研究为LLMS的隐私保护能力提供了一种基本的方法和见解，为合规和有效的ML研究和应用铺平了道路。



## **44. HALLUCINOGEN: A Benchmark for Evaluating Object Hallucination in Large Visual-Language Models**

HALLUCINOogen：评估大型视觉语言模型中对象幻觉的基准 cs.CV

**SubmitDate**: 2024-12-29    [abs](http://arxiv.org/abs/2412.20622v1) [paper-pdf](http://arxiv.org/pdf/2412.20622v1)

**Authors**: Ashish Seth, Dinesh Manocha, Chirag Agarwal

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable performance in performing complex multimodal tasks. However, they are still plagued by object hallucination: the misidentification or misclassification of objects present in images. To this end, we propose HALLUCINOGEN, a novel visual question answering (VQA) object hallucination attack benchmark that utilizes diverse contextual reasoning prompts to evaluate object hallucination in state-of-the-art LVLMs. We design a series of contextual reasoning hallucination prompts to evaluate LVLMs' ability to accurately identify objects in a target image while asking them to perform diverse visual-language tasks such as identifying, locating or performing visual reasoning around specific objects. Further, we extend our benchmark to high-stakes medical applications and introduce MED-HALLUCINOGEN, hallucination attacks tailored to the biomedical domain, and evaluate the hallucination performance of LVLMs on medical images, a critical area where precision is crucial. Finally, we conduct extensive evaluations of eight LVLMs and two hallucination mitigation strategies across multiple datasets to show that current generic and medical LVLMs remain susceptible to hallucination attacks.

摘要: 大型视觉语言模型在执行复杂的多通道任务方面表现出了显著的性能。然而，他们仍然受到物体幻觉的困扰：对图像中存在的物体的错误识别或错误分类。为此，我们提出了一种新颖的视觉问答(VQA)物体幻觉攻击基准--幻觉剂，该基准利用不同的上下文推理提示来评估最新的LVLM中的物体幻觉。我们设计了一系列情境推理幻觉提示来评估LVLMS准确识别目标图像中对象的能力，同时要求他们执行不同的视觉语言任务，如识别、定位或围绕特定对象执行视觉推理。此外，我们将我们的基准扩展到高风险的医疗应用，并引入了MED致幻剂，这是为生物医学领域量身定做的幻觉攻击，并评估了LVLMS在医学图像上的幻觉性能，这是一个对精度至关重要的关键领域。最后，我们在多个数据集上对八个LVLM和两个幻觉缓解策略进行了广泛的评估，以表明当前的普通LVLM和医用LVLM仍然容易受到幻觉攻击。



## **45. Pirates of the RAG: Adaptively Attacking LLMs to Leak Knowledge Bases**

RAG海盗：适应性攻击LLM以泄露知识库 cs.AI

**SubmitDate**: 2024-12-29    [abs](http://arxiv.org/abs/2412.18295v2) [paper-pdf](http://arxiv.org/pdf/2412.18295v2)

**Authors**: Christian Di Maio, Cristian Cosci, Marco Maggini, Valentina Poggioni, Stefano Melacci

**Abstract**: The growing ubiquity of Retrieval-Augmented Generation (RAG) systems in several real-world services triggers severe concerns about their security. A RAG system improves the generative capabilities of a Large Language Models (LLM) by a retrieval mechanism which operates on a private knowledge base, whose unintended exposure could lead to severe consequences, including breaches of private and sensitive information. This paper presents a black-box attack to force a RAG system to leak its private knowledge base which, differently from existing approaches, is adaptive and automatic. A relevance-based mechanism and an attacker-side open-source LLM favor the generation of effective queries to leak most of the (hidden) knowledge base. Extensive experimentation proves the quality of the proposed algorithm in different RAG pipelines and domains, comparing to very recent related approaches, which turn out to be either not fully black-box, not adaptive, or not based on open-source models. The findings from our study remark the urgent need for more robust privacy safeguards in the design and deployment of RAG systems.

摘要: 检索增强生成(RAG)系统在几个现实世界的服务中日益普遍，这引发了人们对其安全性的严重担忧。RAG系统通过在私有知识库上运行的检索机制来提高大型语言模型(LLM)的生成能力，其意外暴露可能导致严重后果，包括隐私和敏感信息的泄露。本文提出了一种黑盒攻击，以迫使RAG系统泄漏其私有知识库，与现有方法不同，该方法是自适应的和自动的。基于相关性的机制和攻击者端的开源LLM有利于生成有效的查询来泄漏大部分(隐藏的)知识库。大量的实验证明了该算法在不同的RAG管道和域中的质量，与最近的相关方法相比，这些方法要么不是完全黑箱的，要么不是自适应的，要么不是基于开源模型的。我们的研究结果表明，在设计和部署RAG系统时，迫切需要更强大的隐私保护措施。



## **46. Can Watermarked LLMs be Identified by Users via Crafted Prompts?**

用户可以通过精心制作的脚本识别带水印的LLM吗？ cs.CR

30 pages, 5 figures, 11 tables

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2410.03168v2) [paper-pdf](http://arxiv.org/pdf/2410.03168v2)

**Authors**: Aiwei Liu, Sheng Guan, Yiming Liu, Leyi Pan, Yifei Zhang, Liancheng Fang, Lijie Wen, Philip S. Yu, Xuming Hu

**Abstract**: Text watermarking for Large Language Models (LLMs) has made significant progress in detecting LLM outputs and preventing misuse. Current watermarking techniques offer high detectability, minimal impact on text quality, and robustness to text editing. However, current researches lack investigation into the imperceptibility of watermarking techniques in LLM services. This is crucial as LLM providers may not want to disclose the presence of watermarks in real-world scenarios, as it could reduce user willingness to use the service and make watermarks more vulnerable to attacks. This work is the first to investigate the imperceptibility of watermarked LLMs. We design an identification algorithm called Water-Probe that detects watermarks through well-designed prompts to the LLM. Our key motivation is that current watermarked LLMs expose consistent biases under the same watermark key, resulting in similar differences across prompts under different watermark keys. Experiments show that almost all mainstream watermarking algorithms are easily identified with our well-designed prompts, while Water-Probe demonstrates a minimal false positive rate for non-watermarked LLMs. Finally, we propose that the key to enhancing the imperceptibility of watermarked LLMs is to increase the randomness of watermark key selection. Based on this, we introduce the Water-Bag strategy, which significantly improves watermark imperceptibility by merging multiple watermark keys.

摘要: 针对大语言模型的文本水印技术在检测大语言模型输出和防止误用方面取得了显著进展。目前的水印技术提供了高可检测性，对文本质量的影响最小，以及对文本编辑的稳健性。然而，目前的研究缺乏对LLM服务中水印技术不可见性的研究。这一点至关重要，因为LLM提供商可能不想透露真实场景中是否存在水印，因为这可能会降低用户使用该服务的意愿，并使水印更容易受到攻击。这项工作是首次研究带水印的LLM的不可感知性。我们设计了一种名为Water-Probe的识别算法，该算法通过对LLM的精心设计的提示来检测水印。我们的关键动机是，当前的水印LLM暴露了相同水印密钥下的一致偏差，导致不同水印密钥下的提示存在相似的差异。实验表明，几乎所有的主流水印算法都能在我们精心设计的提示下很容易地识别出来，而Water-Probe算法对未加水印的LLMS具有最低的误检率。最后，提出了提高水印LLMS不可见性的关键是增加水印密钥选择的随机性。在此基础上，引入了水袋策略，通过合并多个水印密钥，显著提高了水印的不可见性。



## **47. Defending Against Network Attacks for Secure AI Agent Migration in Vehicular Metaverses**

防御网络攻击，实现车载元宇宙中的安全AI代理迁移 cs.NI

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20154v1) [paper-pdf](http://arxiv.org/pdf/2412.20154v1)

**Authors**: Xinru Wen, Jinbo Wen, Ming Xiao, Jiawen Kang, Tao Zhang, Xiaohuan Li, Chuanxi Chen, Dusit Niyato

**Abstract**: Vehicular metaverses, blending traditional vehicular networks with metaverse technology, are expected to revolutionize fields such as autonomous driving. As virtual intelligent assistants in vehicular metaverses, Artificial Intelligence (AI) agents powered by large language models can create immersive 3D virtual spaces for passengers to enjoy on-broad vehicular applications and services. To provide users with seamless and engaging virtual interactions, resource-limited vehicles offload AI agents to RoadSide Units (RSUs) with adequate communication and computational capabilities. Due to the mobility of vehicles and the limited coverage of RSUs, AI agents need to migrate from one RSU to another RSU. However, potential network attacks pose significant challenges to ensuring reliable and efficient AI agent migration. In this paper, we first explore specific network attacks including traffic-based attacks (i.e., DDoS attacks) and infrastructure-based attacks (i.e., malicious RSU attacks). Then, we model the AI agent migration process as a Partially Observable Markov Decision Process (POMDP) and apply multi-agent proximal policy optimization algorithms to mitigate DDoS attacks. In addition, we propose a trust assessment mechanism to counter malicious RSU attacks. Numerical results validate that the proposed solutions effectively defend against these network attacks and reduce the total latency of AI agent migration by approximately 43.3%.

摘要: 车载虚拟现实将传统的车辆网络与虚拟现实技术相结合，预计将给自动驾驶等领域带来革命性的变化。作为车载虚拟现实中的虚拟智能助手，人工智能(AI)代理以大型语言模型为动力，可以创建身临其境的3D虚拟空间，供乘客享受广泛的车载应用和服务。为了向用户提供无缝且引人入胜的虚拟交互，资源有限的车辆将AI代理卸载到具有足够通信和计算能力的路边单元(RSU)。由于车辆的机动性和RSU的覆盖范围有限，AI代理需要从一个RSU迁移到另一个RSU。然而，潜在的网络攻击对确保可靠和高效的AI代理迁移构成了重大挑战。本文首先探讨了具体的网络攻击，包括基于流量的攻击(即DDoS攻击)和基于基础设施的攻击(即恶意RSU攻击)。然后，我们将AI代理迁移过程建模为部分可观测马尔可夫决策过程(POMDP)，并应用多代理邻近策略优化算法来缓解DDoS攻击。此外，我们还提出了一种信任评估机制来对抗恶意的RSU攻击。数值结果验证了所提出的解决方案有效地防御了这些网络攻击，并将AI代理迁移的总延迟降低了约43.3%。



## **48. On the Validity of Traditional Vulnerability Scoring Systems for Adversarial Attacks against LLMs**

传统漏洞评分系统对LLM对抗性攻击的有效性 cs.CR

101 pages, 3 figures

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2412.20087v1) [paper-pdf](http://arxiv.org/pdf/2412.20087v1)

**Authors**: Atmane Ayoub Mansour Bahar, Ahmad Samer Wazan

**Abstract**: This research investigates the effectiveness of established vulnerability metrics, such as the Common Vulnerability Scoring System (CVSS), in evaluating attacks against Large Language Models (LLMs), with a focus on Adversarial Attacks (AAs). The study explores the influence of both general and specific metric factors in determining vulnerability scores, providing new perspectives on potential enhancements to these metrics.   This study adopts a quantitative approach, calculating and comparing the coefficient of variation of vulnerability scores across 56 adversarial attacks on LLMs. The attacks, sourced from various research papers, and obtained through online databases, were evaluated using multiple vulnerability metrics. Scores were determined by averaging the values assessed by three distinct LLMs. The results indicate that existing scoring-systems yield vulnerability scores with minimal variation across different attacks, suggesting that many of the metric factors are inadequate for assessing adversarial attacks on LLMs. This is particularly true for context-specific factors or those with predefined value sets, such as those in CVSS. These findings support the hypothesis that current vulnerability metrics, especially those with rigid values, are limited in evaluating AAs on LLMs, highlighting the need for the development of more flexible, generalized metrics tailored to such attacks.   This research offers a fresh analysis of the effectiveness and applicability of established vulnerability metrics, particularly in the context of Adversarial Attacks on Large Language Models, both of which have gained significant attention in recent years. Through extensive testing and calculations, the study underscores the limitations of these metrics and opens up new avenues for improving and refining vulnerability assessment frameworks specifically tailored for LLMs.

摘要: 这项研究考察了通用漏洞评分系统(CVSS)等已建立的漏洞度量在评估针对大型语言模型(LLMS)的攻击时的有效性，重点是对抗性攻击(AA)。这项研究探讨了一般和特定指标因素在确定脆弱性得分方面的影响，为这些指标的潜在增强提供了新的视角。本研究采用定量的方法，计算并比较了56种对抗性攻击下的LLMS脆弱性得分的变异系数。这些攻击来自各种研究论文，通过在线数据库获得，使用多种漏洞指标进行评估。得分通过三个不同的LLM评估的值的平均值来确定。结果表明，现有的评分系统产生的脆弱性分数在不同攻击之间的差异很小，这表明许多度量因素不足以评估对LLM的对抗性攻击。对于特定于上下文的因素或具有预定义值集的因素尤其如此，例如CVSS中的那些因素。这些发现支持这样一种假设，即当前的脆弱性指标，特别是那些具有刚性值的指标，在评估LLM上的AA方面是有限的，这突显了开发针对此类攻击量身定做的更灵活、更通用的指标的必要性。这项研究对已建立的脆弱性度量的有效性和适用性进行了新的分析，特别是在针对大型语言模型的对抗性攻击的背景下，这两种攻击在最近几年都得到了极大的关注。通过广泛的测试和计算，这项研究强调了这些指标的局限性，并为改进和完善专门为低土地管理定制的脆弱性评估框架开辟了新的途径。



## **49. LLM-Virus: Evolutionary Jailbreak Attack on Large Language Models**

LLM-Virus：对大型语言模型的进化越狱攻击 cs.CR

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2501.00055v1) [paper-pdf](http://arxiv.org/pdf/2501.00055v1)

**Authors**: Miao Yu, Junfeng Fang, Yingjie Zhou, Xing Fan, Kun Wang, Shirui Pan, Qingsong Wen

**Abstract**: While safety-aligned large language models (LLMs) are increasingly used as the cornerstone for powerful systems such as multi-agent frameworks to solve complex real-world problems, they still suffer from potential adversarial queries, such as jailbreak attacks, which attempt to induce harmful content. Researching attack methods allows us to better understand the limitations of LLM and make trade-offs between helpfulness and safety. However, existing jailbreak attacks are primarily based on opaque optimization techniques (e.g. token-level gradient descent) and heuristic search methods like LLM refinement, which fall short in terms of transparency, transferability, and computational cost. In light of these limitations, we draw inspiration from the evolution and infection processes of biological viruses and propose LLM-Virus, a jailbreak attack method based on evolutionary algorithm, termed evolutionary jailbreak. LLM-Virus treats jailbreak attacks as both an evolutionary and transfer learning problem, utilizing LLMs as heuristic evolutionary operators to ensure high attack efficiency, transferability, and low time cost. Our experimental results on multiple safety benchmarks show that LLM-Virus achieves competitive or even superior performance compared to existing attack methods.

摘要: 尽管与安全一致的大型语言模型(LLM)越来越多地被用作多代理框架等强大系统的基石，以解决复杂的现实世界问题，但它们仍面临潜在的对抗性查询，例如试图诱导有害内容的越狱攻击。研究攻击方法可以让我们更好地了解LLM的局限性，并在有效性和安全性之间进行权衡。然而，现有的越狱攻击主要基于不透明的优化技术(如令牌级梯度下降)和启发式搜索方法，如LLM求精，这些方法在透明度、可转移性和计算成本方面都存在不足。针对这些局限性，我们从生物病毒的进化和感染过程中得到启发，提出了一种基于进化算法的越狱攻击方法LLM-Virus，称为进化越狱。LLM-Virus将越狱攻击视为一个进化和转移学习问题，利用LLM作为启发式进化算子，以确保高攻击效率、可转移性和低时间开销。我们在多个安全基准上的实验结果表明，与现有的攻击方法相比，LLM-Virus具有相当甚至更好的性能。



## **50. B-AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Black-box Adversarial Visual-Instructions**

B-AVIBench：评估黑匣子对抗视觉指令上大型视觉语言模型的鲁棒性 cs.CV

Accepted by IEEE Transactions on Information Forensics & Security

**SubmitDate**: 2024-12-28    [abs](http://arxiv.org/abs/2403.09346v2) [paper-pdf](http://arxiv.org/pdf/2403.09346v2)

**Authors**: Hao Zhang, Wenqi Shao, Hong Liu, Yongqiang Ma, Ping Luo, Yu Qiao, Nanning Zheng, Kaipeng Zhang

**Abstract**: Large Vision-Language Models (LVLMs) have shown significant progress in responding well to visual-instructions from users. However, these instructions, encompassing images and text, are susceptible to both intentional and inadvertent attacks. Despite the critical importance of LVLMs' robustness against such threats, current research in this area remains limited. To bridge this gap, we introduce B-AVIBench, a framework designed to analyze the robustness of LVLMs when facing various Black-box Adversarial Visual-Instructions (B-AVIs), including four types of image-based B-AVIs, ten types of text-based B-AVIs, and nine types of content bias B-AVIs (such as gender, violence, cultural, and racial biases, among others). We generate 316K B-AVIs encompassing five categories of multimodal capabilities (ten tasks) and content bias. We then conduct a comprehensive evaluation involving 14 open-source LVLMs to assess their performance. B-AVIBench also serves as a convenient tool for practitioners to evaluate the robustness of LVLMs against B-AVIs. Our findings and extensive experimental results shed light on the vulnerabilities of LVLMs, and highlight that inherent biases exist even in advanced closed-source LVLMs like GeminiProVision and GPT-4V. This underscores the importance of enhancing the robustness, security, and fairness of LVLMs. The source code and benchmark are available at https://github.com/zhanghao5201/B-AVIBench.

摘要: 大型视觉语言模型(LVLM)在很好地响应用户的视觉指令方面取得了重大进展。但是，这些包含图像和文本的说明很容易受到有意和无意的攻击。尽管LVLMS对这类威胁的稳健性至关重要，但目前在这一领域的研究仍然有限。为了弥补这一差距，我们引入了B-AVIB边框架，该框架旨在分析LVLMS在面对各种黑盒对抗性视觉指令(B-AVI)时的健壮性，包括四种类型的基于图像的B-AVI、10种类型的基于文本的B-AVI和九种类型的内容偏见B-AVI(如性别、暴力、文化和种族偏见等)。我们生成了316k B-AVI，包括五类多模式能力(十项任务)和内容偏见。然后，我们对14个开源LVLM进行了全面的评估，以评估它们的性能。B-AVIBtch也可作为从业者评估LVLMS对B-AVIS的稳健性的便捷工具。我们的发现和广泛的实验结果揭示了LVLMS的漏洞，并突出表明即使在GeminiProVision和GPT-4V等先进的闭源LVLM中也存在固有偏差。这凸显了增强LVLM的健壮性、安全性和公平性的重要性。源代码和基准测试可在https://github.com/zhanghao5201/B-AVIBench.上获得



