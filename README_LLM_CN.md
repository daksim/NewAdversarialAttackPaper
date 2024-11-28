# Latest Large Language Model Attack Papers
**update at 2024-11-28 10:17:04**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Neutralizing Backdoors through Information Conflicts for Large Language Models**

通过大型语言模型的信息冲突消除后门 cs.CL

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18280v1) [paper-pdf](http://arxiv.org/pdf/2411.18280v1)

**Authors**: Chen Chen, Yuchen Sun, Xueluan Gong, Jiaxin Gao, Kwok-Yan Lam

**Abstract**: Large language models (LLMs) have seen significant advancements, achieving superior performance in various Natural Language Processing (NLP) tasks, from understanding to reasoning. However, they remain vulnerable to backdoor attacks, where models behave normally for standard queries but generate harmful responses or unintended output when specific triggers are activated. Existing backdoor defenses often suffer from drawbacks that they either focus on detection without removal, rely on rigid assumptions about trigger properties, or prove to be ineffective against advanced attacks like multi-trigger backdoors. In this paper, we present a novel method to eliminate backdoor behaviors from LLMs through the construction of information conflicts using both internal and external mechanisms. Internally, we leverage a lightweight dataset to train a conflict model, which is then merged with the backdoored model to neutralize malicious behaviors by embedding contradictory information within the model's parametric memory. Externally, we incorporate convincing contradictory evidence into the prompt to challenge the model's internal backdoor knowledge. Experimental results on classification and conversational tasks across 4 widely used LLMs demonstrate that our method outperforms 8 state-of-the-art backdoor defense baselines. We can reduce the attack success rate of advanced backdoor attacks by up to 98% while maintaining over 90% clean data accuracy. Furthermore, our method has proven to be robust against adaptive backdoor attacks. The code will be open-sourced upon publication.

摘要: 大型语言模型(LLM)已经取得了显著的进步，在从理解到推理的各种自然语言处理(NLP)任务中取得了优异的性能。然而，它们仍然容易受到后门攻击，在后门攻击中，模型对标准查询正常操作，但在激活特定触发器时会生成有害响应或意外输出。现有的后门防御往往存在缺陷，要么专注于检测而不移除，依赖于对触发属性的僵化假设，要么被证明对多触发后门等高级攻击无效。在本文中，我们提出了一种新的方法，通过利用内部和外部机制构建信息冲突来消除LLMS中的后门行为。在内部，我们利用轻量级数据集来训练冲突模型，然后将其与后置模型合并，通过在模型的参数记忆中嵌入相互矛盾的信息来中和恶意行为。在外部，我们将令人信服的相互矛盾的证据结合到提示中，以挑战模型的内部后门知识。在4个广泛使用的LLM上的分类和会话任务上的实验结果表明，该方法的性能优于8个最先进的后门防御基线。我们可以将高级后门攻击的攻击成功率降低高达98%，同时保持90%以上的干净数据准确性。此外，我们的方法已被证明对自适应后门攻击具有健壮性。代码将在发布后开放源代码。



## **2. Visual Adversarial Attack on Vision-Language Models for Autonomous Driving**

自动驾驶视觉语言模型的视觉对抗攻击 cs.CV

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18275v1) [paper-pdf](http://arxiv.org/pdf/2411.18275v1)

**Authors**: Tianyuan Zhang, Lu Wang, Xinwei Zhang, Yitong Zhang, Boyi Jia, Siyuan Liang, Shengshan Hu, Qiang Fu, Aishan Liu, Xianglong Liu

**Abstract**: Vision-language models (VLMs) have significantly advanced autonomous driving (AD) by enhancing reasoning capabilities. However, these models remain highly vulnerable to adversarial attacks. While existing research has primarily focused on general VLM attacks, the development of attacks tailored to the safety-critical AD context has been largely overlooked. In this paper, we take the first step toward designing adversarial attacks specifically targeting VLMs in AD, exposing the substantial risks these attacks pose within this critical domain. We identify two unique challenges for effective adversarial attacks on AD VLMs: the variability of textual instructions and the time-series nature of visual scenarios. To this end, we propose ADvLM, the first visual adversarial attack framework specifically designed for VLMs in AD. Our framework introduces Semantic-Invariant Induction, which uses a large language model to create a diverse prompt library of textual instructions with consistent semantic content, guided by semantic entropy. Building on this, we introduce Scenario-Associated Enhancement, an approach where attention mechanisms select key frames and perspectives within driving scenarios to optimize adversarial perturbations that generalize across the entire scenario. Extensive experiments on several AD VLMs over multiple benchmarks show that ADvLM achieves state-of-the-art attack effectiveness. Moreover, real-world attack studies further validate its applicability and potential in practice.

摘要: 视觉语言模型通过增强推理能力极大地促进了自动驾驶(AD)。然而，这些模型仍然非常容易受到对手的攻击。虽然现有的研究主要集中在一般的VLM攻击上，但针对安全关键型AD环境而定制的攻击的发展在很大程度上被忽视了。在本文中，我们向设计专门针对AD中的VLM的对抗性攻击迈出了第一步，暴露了这些攻击在这一关键领域中构成的实质性风险。我们确定了对AD VLMS进行有效的对抗性攻击的两个独特的挑战：文本指令的可变性和视觉场景的时间序列性质。为此，我们提出了ADvLM，这是第一个专门为AD中的VLM设计的可视化对抗性攻击框架。我们的框架引入了语义不变归纳法，它使用一个大型语言模型来创建一个具有一致语义内容的多样化提示库，并以语义熵为指导。在此基础上，我们引入了与场景相关的增强，这是一种注意机制在驾驶场景中选择关键帧和视角以优化整个场景中概括的对抗性扰动的方法。在多个基准上对多个AD VLM进行的大量实验表明，ADvLM达到了最先进的攻击效率。此外，真实世界的攻击研究进一步验证了其在实践中的适用性和潜力。



## **3. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

G-Designer：通过图神经网络构建多智能体通信布局 cs.MA

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2410.11782v2) [paper-pdf](http://arxiv.org/pdf/2410.11782v2)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.

摘要: 基于大型语言模型(LLM)的代理的最新进展表明，集体智能可以显著超过单个代理的能力，这主要是由于精心设计的代理间通信拓扑。尽管有多样化和高性能的设计，但实践者在为他们的特定任务选择最有效的流水线时经常面临困惑：\textit{哪个拓扑是我的任务的最佳选择，在确保高质量解决方案的同时避免不必要的通信令牌开销？}针对这种困境，我们引入了G-Designer，这是一个自适应的、高效的、健壮的多代理部署解决方案，它动态地设计任务感知的、定制的通信拓扑。具体地说，G-Designer将多代理系统建模为多代理网络，利用变化图自动编码器对节点(代理)和特定于任务的虚拟节点进行编码，并解码任务自适应的高性能通信拓扑。在六个基准测试上的广泛实验表明，G-Designer是：\extbf{(1)高性能}，在MMLU上获得了更好的结果，准确率为84.50\$，在HumanEval上，PASS@1的准确率为89.90\$；\extbf{(2)任务自适应}，构建了针对任务难度的通信协议，在HumanEval上减少了高达95.33\$的令牌消耗；以及\extbf{(3)对手健壮性}，防御代理对手攻击，精确度仅下降了$0.3\%$。



## **4. Transferable Ensemble Black-box Jailbreak Attacks on Large Language Models**

可转移集成黑匣子越狱攻击大型语言模型 cs.CR

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2410.23558v2) [paper-pdf](http://arxiv.org/pdf/2410.23558v2)

**Authors**: Yiqi Yang, Hongye Fu

**Abstract**: In this report, we propose a novel black-box jailbreak attacking framework that incorporates various LLM-as-Attacker methods to deliver transferable and powerful jailbreak attacks. Our method is designed based on three key observations from existing jailbreaking studies and practices. First, we consider an ensemble approach should be more effective in exposing the vulnerabilities of an aligned LLM compared to individual attacks. Second, different malicious instructions inherently vary in their jailbreaking difficulty, necessitating differentiated treatment to ensure more efficient attacks. Finally, the semantic coherence of a malicious instruction is crucial for triggering the defenses of an aligned LLM; therefore, it must be carefully disrupted to manipulate its embedding representation, thereby increasing the jailbreak success rate. We validated our approach by participating in the Competition for LLM and Agent Safety 2024, where our team achieved top performance in the Jailbreaking Attack Track.

摘要: 在这份报告中，我们提出了一种新的黑盒越狱攻击框架，该框架结合了各种LLM作为攻击者的方法来提供可转移的强大越狱攻击。我们的方法是基于现有越狱研究和实践中的三个关键观察结果而设计的。首先，我们认为，与单独攻击相比，整体方法应该更有效地暴露联合LLM的漏洞。其次，不同的恶意指令在越狱难度上存在内在差异，需要区别对待，以确保更有效的攻击。最后，恶意指令的语义一致性对于触发对齐的LLM的防御至关重要；因此，必须小心破坏它才能操纵其嵌入表示，从而提高越狱成功率。我们通过参加LLM和代理安全竞赛2024来验证我们的方法，我们的团队在越狱攻击赛道上取得了最好的表现。



## **5. R-MTLLMF: Resilient Multi-Task Large Language Model Fusion at the Wireless Edge**

R-MTLLMF：无线边缘的弹性多任务大型语言模型融合 eess.SP

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18220v1) [paper-pdf](http://arxiv.org/pdf/2411.18220v1)

**Authors**: Aladin Djuhera, Vlad C. Andrei, Mohsen Pourghasemian, Haris Gacanin, Holger Boche, Walid Saad

**Abstract**: Multi-task large language models (MTLLMs) are important for many applications at the wireless edge, where users demand specialized models to handle multiple tasks efficiently. However, training MTLLMs is complex and exhaustive, particularly when tasks are subject to change. Recently, the concept of model fusion via task vectors has emerged as an efficient approach for combining fine-tuning parameters to produce an MTLLM. In this paper, the problem of enabling edge users to collaboratively craft such MTTLMs via tasks vectors is studied, under the assumption of worst-case adversarial attacks. To this end, first the influence of adversarial noise to multi-task model fusion is investigated and a relationship between the so-called weight disentanglement error and the mean squared error (MSE) is derived. Using hypothesis testing, it is directly shown that the MSE increases interference between task vectors, thereby rendering model fusion ineffective. Then, a novel resilient MTLLM fusion (R-MTLLMF) is proposed, which leverages insights about the LLM architecture and fine-tuning process to safeguard task vector aggregation under adversarial noise by realigning the MTLLM. The proposed R-MTLLMF is then compared for both worst-case and ideal transmission scenarios to study the impact of the wireless channel. Extensive model fusion experiments with vision LLMs demonstrate R-MTLLMF's effectiveness, achieving close-to-baseline performance across eight different tasks in ideal noise scenarios and significantly outperforming unprotected model fusion in worst-case scenarios. The results further advocate for additional physical layer protection for a holistic approach to resilience, from both a wireless and LLM perspective.

摘要: 多任务大型语言模型(MTLLM)对于无线边缘的许多应用非常重要，因为用户需要专门的模型来高效地处理多个任务。然而，培训MTLLM是复杂和详尽的，特别是在任务可能发生变化的情况下。最近，基于任务向量的模型融合的概念已经成为一种结合微调参数以产生MTLLM的有效方法。本文在假设最坏情况下的对抗性攻击的前提下，研究了边缘用户通过任务向量协作生成MTTLM的问题。为此，首先研究了对抗性噪声对多任务模型融合的影响，推导了加权解缠误差与均方误差之间的关系。通过假设检验，直接表明MSE增加了任务向量之间的干扰，从而使模型融合无效。然后，提出了一种新的弹性MTLLM融合算法(R-MTLLMF)，该算法利用对LLM体系结构和微调过程的深入了解，通过重新排列MTLLM来保护对抗噪声下的任务向量聚合。然后将所提出的R-MTLLMF在最坏情况和理想传输场景下进行比较，以研究无线信道的影响。用VISION LLMS进行的大量模型融合实验证明了R-MTLLMF的有效性，在理想噪声场景中，R-MTLLMF在八个不同任务上的性能接近基线，而在最坏情况下，R-MTLLMF的性能明显优于无保护的模型融合。从无线和LLM的角度来看，研究结果进一步倡导为整体恢复方法提供额外的物理层保护。



## **6. Evaluating and Improving the Robustness of Security Attack Detectors Generated by LLMs**

评估和改进LLM生成的安全攻击检测器的鲁棒性 cs.SE

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18216v1) [paper-pdf](http://arxiv.org/pdf/2411.18216v1)

**Authors**: Samuele Pasini, Jinhan Kim, Tommaso Aiello, Rocio Cabrera Lozoya, Antonino Sabetta, Paolo Tonella

**Abstract**: Large Language Models (LLMs) are increasingly used in software development to generate functions, such as attack detectors, that implement security requirements. However, LLMs struggle to generate accurate code, resulting, e.g., in attack detectors that miss well-known attacks when used in practice. This is most likely due to the LLM lacking knowledge about some existing attacks and to the generated code being not evaluated in real usage scenarios. We propose a novel approach integrating Retrieval Augmented Generation (RAG) and Self-Ranking into the LLM pipeline. RAG enhances the robustness of the output by incorporating external knowledge sources, while the Self-Ranking technique, inspired to the concept of Self-Consistency, generates multiple reasoning paths and creates ranks to select the most robust detector. Our extensive empirical study targets code generated by LLMs to detect two prevalent injection attacks in web security: Cross-Site Scripting (XSS) and SQL injection (SQLi). Results show a significant improvement in detection performance compared to baselines, with an increase of up to 71%pt and 37%pt in the F2-Score for XSS and SQLi detection, respectively.

摘要: 大型语言模型(LLM)越来越多地用于软件开发，以生成实现安全要求的功能，如攻击检测器。然而，LLMS很难生成准确的代码，导致例如攻击检测器在实际使用时错过了众所周知的攻击。这很可能是因为LLM缺乏关于一些现有攻击的知识，并且生成的代码在实际使用场景中没有得到评估。我们提出了一种新的方法，将检索增强生成(RAG)和自我排序结合到LLM流水线中。RAG通过引入外部知识源来增强输出的稳健性，而自排序技术则受到自相容概念的启发，生成多条推理路径并创建排序来选择最健壮的检测器。我们广泛的经验研究针对LLMS生成的代码来检测网络安全中两种流行的注入攻击：跨站点脚本(XSS)和SQL注入(SQLI)。结果表明，与基线相比，检测性能有了显著提高，XSS和SQLI检测的F2分数分别增加了71%和37%。



## **7. InputSnatch: Stealing Input in LLM Services via Timing Side-Channel Attacks**

InputSnatch：通过定时侧通道攻击窃取LLM服务中的输入 cs.CR

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18191v1) [paper-pdf](http://arxiv.org/pdf/2411.18191v1)

**Authors**: Xinyao Zheng, Husheng Han, Shangyi Shi, Qiyan Fang, Zidong Du, Qi Guo, Xing Hu

**Abstract**: Large language models (LLMs) possess extensive knowledge and question-answering capabilities, having been widely deployed in privacy-sensitive domains like finance and medical consultation. During LLM inferences, cache-sharing methods are commonly employed to enhance efficiency by reusing cached states or responses for the same or similar inference requests. However, we identify that these cache mechanisms pose a risk of private input leakage, as the caching can result in observable variations in response times, making them a strong candidate for a timing-based attack hint. In this study, we propose a novel timing-based side-channel attack to execute input theft in LLMs inference. The cache-based attack faces the challenge of constructing candidate inputs in a large search space to hit and steal cached user queries. To address these challenges, we propose two primary components. The input constructor employs machine learning techniques and LLM-based approaches for vocabulary correlation learning while implementing optimized search mechanisms for generalized input construction. The time analyzer implements statistical time fitting with outlier elimination to identify cache hit patterns, continuously providing feedback to refine the constructor's search strategy. We conduct experiments across two cache mechanisms and the results demonstrate that our approach consistently attains high attack success rates in various applications. Our work highlights the security vulnerabilities associated with performance optimizations, underscoring the necessity of prioritizing privacy and security alongside enhancements in LLM inference.

摘要: 大型语言模型(LLM)具有广泛的知识和问答能力，已广泛应用于金融、医疗咨询等隐私敏感领域。在LLM推理期间，通常使用高速缓存共享方法来通过对相同或相似的推理请求重复使用高速缓存的状态或响应来提高效率。然而，我们发现这些缓存机制带来了私有输入泄漏的风险，因为缓存可能会导致响应时间的明显变化，从而使它们成为基于时间的攻击提示的有力候选者。在这项研究中，我们提出了一种新的基于时序的旁路攻击来执行LLMS推理中的输入窃取。基于缓存的攻击面临着在大搜索空间中构建候选输入以命中和窃取缓存的用户查询的挑战。为了应对这些挑战，我们提出了两个主要组成部分。输入构造器使用机器学习技术和基于LLM的方法进行词汇关联学习，同时实现优化的搜索机制来构建通用输入。时间分析器使用异常值消除来实现统计时间拟合，以识别缓存命中模式，并持续提供反馈以改进构造器的搜索策略。我们在两种缓存机制上进行了实验，结果表明，我们的方法在不同的应用中都取得了很高的攻击成功率。我们的工作突出了与性能优化相关的安全漏洞，强调了在增强LLM推理的同时优先考虑隐私和安全的必要性。



## **8. Playing Language Game with LLMs Leads to Jailbreaking**

与法学硕士玩语言游戏导致越狱 cs.CL

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.12762v2) [paper-pdf](http://arxiv.org/pdf/2411.12762v2)

**Authors**: Yu Peng, Zewen Long, Fangming Dong, Congyi Li, Shu Wu, Kai Chen

**Abstract**: The advent of large language models (LLMs) has spurred the development of numerous jailbreak techniques aimed at circumventing their security defenses against malicious attacks. An effective jailbreak approach is to identify a domain where safety generalization fails, a phenomenon known as mismatched generalization. In this paper, we introduce two novel jailbreak methods based on mismatched generalization: natural language games and custom language games, both of which effectively bypass the safety mechanisms of LLMs, with various kinds and different variants, making them hard to defend and leading to high attack rates. Natural language games involve the use of synthetic linguistic constructs and the actions intertwined with these constructs, such as the Ubbi Dubbi language. Building on this phenomenon, we propose the custom language games method: by engaging with LLMs using a variety of custom rules, we successfully execute jailbreak attacks across multiple LLM platforms. Extensive experiments demonstrate the effectiveness of our methods, achieving success rates of 93% on GPT-4o, 89% on GPT-4o-mini and 83% on Claude-3.5-Sonnet. Furthermore, to investigate the generalizability of safety alignments, we fine-tuned Llama-3.1-70B with the custom language games to achieve safety alignment within our datasets and found that when interacting through other language games, the fine-tuned models still failed to identify harmful content. This finding indicates that the safety alignment knowledge embedded in LLMs fails to generalize across different linguistic formats, thus opening new avenues for future research in this area.

摘要: 大型语言模型(LLM)的出现促进了许多越狱技术的发展，这些技术旨在绕过针对恶意攻击的安全防御。一种有效的越狱方法是识别安全泛化失败的域，这种现象称为不匹配泛化。本文介绍了两种新的基于不匹配泛化的越狱方法：自然语言游戏和自定义语言游戏，这两种方法都有效地绕过了LLMS的安全机制，种类繁多，变体不同，使得它们难以防御，导致攻击率很高。自然语言游戏涉及使用合成的语言结构以及与这些结构交织在一起的动作，如Ubbi Dubbi语言。基于这一现象，我们提出了定制语言游戏方法：通过使用各种定制规则与LLM接触，我们成功地跨多个LLM平台执行越狱攻击。大量实验证明了该方法的有效性，在GPT-4o、GPT-4o-mini和Claude-3.5-十四行诗上分别获得了93%、89%和83%的识别成功率。此外，为了调查安全对齐的泛化能力，我们使用自定义语言游戏对Llama-3.1-70B进行了微调，以在我们的数据集中实现安全对齐，发现当通过其他语言游戏交互时，微调的模型仍然无法识别有害内容。这一发现表明，LLMS中嵌入的安全对齐知识无法跨不同的语言格式进行泛化，从而为这一领域的未来研究开辟了新的途径。



## **9. BlackDAN: A Black-Box Multi-Objective Approach for Effective and Contextual Jailbreaking of Large Language Models**

BlackDAN：一种有效且上下文化的大型语言模型越狱的黑匣子多目标方法 cs.CR

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2410.09804v3) [paper-pdf](http://arxiv.org/pdf/2410.09804v3)

**Authors**: Xinyuan Wang, Victor Shea-Jay Huang, Renmiao Chen, Hao Wang, Chengwei Pan, Lei Sha, Minlie Huang

**Abstract**: While large language models (LLMs) exhibit remarkable capabilities across various tasks, they encounter potential security risks such as jailbreak attacks, which exploit vulnerabilities to bypass security measures and generate harmful outputs. Existing jailbreak strategies mainly focus on maximizing attack success rate (ASR), frequently neglecting other critical factors, including the relevance of the jailbreak response to the query and the level of stealthiness. This narrow focus on single objectives can result in ineffective attacks that either lack contextual relevance or are easily recognizable. In this work, we introduce BlackDAN, an innovative black-box attack framework with multi-objective optimization, aiming to generate high-quality prompts that effectively facilitate jailbreaking while maintaining contextual relevance and minimizing detectability. BlackDAN leverages Multiobjective Evolutionary Algorithms (MOEAs), specifically the NSGA-II algorithm, to optimize jailbreaks across multiple objectives including ASR, stealthiness, and semantic relevance. By integrating mechanisms like mutation, crossover, and Pareto-dominance, BlackDAN provides a transparent and interpretable process for generating jailbreaks. Furthermore, the framework allows customization based on user preferences, enabling the selection of prompts that balance harmfulness, relevance, and other factors. Experimental results demonstrate that BlackDAN outperforms traditional single-objective methods, yielding higher success rates and improved robustness across various LLMs and multimodal LLMs, while ensuring jailbreak responses are both relevant and less detectable.

摘要: 虽然大型语言模型(LLM)在各种任务中显示出非凡的能力，但它们遇到了潜在的安全风险，如越狱攻击，这些攻击利用漏洞绕过安全措施并产生有害的输出。现有的越狱策略主要关注最大化攻击成功率(ASR)，往往忽略了其他关键因素，包括越狱响应与查询的相关性和隐蔽性水平。这种对单一目标的狭隘关注可能会导致无效的攻击，要么缺乏上下文相关性，要么很容易识别。在这项工作中，我们引入了BlackDAN，一个创新的多目标优化的黑盒攻击框架，旨在生成高质量的提示，在保持上下文相关性的同时有效地促进越狱，并将可检测性降至最低。BlackDAN利用多目标进化算法(MOEA)，特别是NSGA-II算法，跨多个目标优化越狱，包括ASR、隐蔽性和语义相关性。通过集成变异、交叉和帕累托支配等机制，BlackDAN为生成越狱提供了一个透明和可解释的过程。此外，该框架允许根据用户偏好进行定制，从而能够选择在危害性、相关性和其他因素之间进行权衡的提示。实验结果表明，BlackDAN的性能优于传统的单目标方法，在各种LLM和多模式LLM上获得了更高的成功率和更好的鲁棒性，同时确保了越狱响应的相关性和较低的可检测性。



## **10. Desert Camels and Oil Sheikhs: Arab-Centric Red Teaming of Frontier LLMs**

沙漠骆驼和石油酋长：以阿拉伯为中心的红色Frontier LLM团队 cs.CL

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2410.24049v3) [paper-pdf](http://arxiv.org/pdf/2410.24049v3)

**Authors**: Muhammed Saeed, Elgizouli Mohamed, Mukhtar Mohamed, Shaina Raza, Muhammad Abdul-Mageed, Shady Shehata

**Abstract**: Large language models (LLMs) are widely used but raise ethical concerns due to embedded social biases. This study examines LLM biases against Arabs versus Westerners across eight domains, including women's rights, terrorism, and anti-Semitism and assesses model resistance to perpetuating these biases. To this end, we create two datasets: one to evaluate LLM bias toward Arabs versus Westerners and another to test model safety against prompts that exaggerate negative traits ("jailbreaks"). We evaluate six LLMs -- GPT-4, GPT-4o, LlaMA 3.1 (8B & 405B), Mistral 7B, and Claude 3.5 Sonnet. We find 79% of cases displaying negative biases toward Arabs, with LlaMA 3.1-405B being the most biased. Our jailbreak tests reveal GPT-4o as the most vulnerable, despite being an optimized version, followed by LlaMA 3.1-8B and Mistral 7B. All LLMs except Claude exhibit attack success rates above 87% in three categories. We also find Claude 3.5 Sonnet the safest, but it still displays biases in seven of eight categories. Despite being an optimized version of GPT4, We find GPT-4o to be more prone to biases and jailbreaks, suggesting optimization flaws. Our findings underscore the pressing need for more robust bias mitigation strategies and strengthened security measures in LLMs.

摘要: 大型语言模型(LLM)被广泛使用，但由于根深蒂固的社会偏见而引发了伦理问题。这项研究考察了LLM在八个领域对阿拉伯人和西方人的偏见，包括妇女权利、恐怖主义和反犹太主义，并评估了对延续这些偏见的模型阻力。为此，我们创建了两个数据集：一个用于评估LLM对阿拉伯人和西方人的偏见，另一个用于测试针对夸大负面特征的提示(“越狱”)的模型安全性。我们评估了六个LLMS--GPT-4、GPT-40、大羊驼3.1(8B和405B)、西北风7B和克劳德3.5十四行诗。我们发现79%的病例对阿拉伯人表现出负面偏见，其中大羊驼3.1-405B是最有偏见的。我们的越狱测试显示，GPT-40是最脆弱的，尽管是一个优化版本，紧随其后的是骆驼3.1-8B和米斯特拉尔7B。除克劳德外，所有LLM在三个类别中的攻击成功率都在87%以上。我们也发现克劳德3.5十四行诗是最安全的，但它仍然在八个类别中的七个方面表现出偏见。尽管GPT-4是GPT4的优化版本，但我们发现GPT-40更容易产生偏见和越狱，这表明优化存在缺陷。我们的研究结果突出表明，迫切需要更有力的减轻偏见战略和加强小岛屿发展中国家的安全措施。



## **11. RTL-Breaker: Assessing the Security of LLMs against Backdoor Attacks on HDL Code Generation**

RTL-Breaker：评估LLM的安全性，以应对对HDL代码生成的后门攻击 cs.CR

Accepted at 2025 Design, Automation & Test in Europe (DATE)  Conference

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.17569v1) [paper-pdf](http://arxiv.org/pdf/2411.17569v1)

**Authors**: Lakshmi Likhitha Mankali, Jitendra Bhandari, Manaar Alam, Ramesh Karri, Michail Maniatakos, Ozgur Sinanoglu, Johann Knechtel

**Abstract**: Large language models (LLMs) have demonstrated remarkable potential with code generation/completion tasks for hardware design. In fact, LLM-based hardware description language (HDL) code generation has enabled the industry to realize complex designs more quickly, reducing the time and effort required in the development cycle. However, the increased reliance on such automation introduces critical security risks. Notably, given that LLMs have to be trained on vast datasets of codes that are typically sourced from publicly available repositories (often without thorough validation), LLMs are susceptible to so-called data poisoning or backdoor attacks. Here, attackers inject malicious code for the training data, which can be carried over into the HDL code generated by LLMs. This threat vector can compromise the security and integrity of entire hardware systems. In this work, we propose RTL-Breaker, a novel backdoor attack framework on LLM-based HDL code generation. RTL-Breaker provides an in-depth analysis for essential aspects of this novel problem: 1) various trigger mechanisms versus their effectiveness for inserting malicious modifications, and 2) side-effects by backdoor attacks on code generation in general, i.e., impact on code quality. RTL-Breaker emphasizes the urgent need for more robust measures to safeguard against such attacks. Toward that end, we open-source our framework and all data.

摘要: 大型语言模型(LLM)在硬件设计的代码生成/完成任务方面表现出了巨大的潜力。事实上，基于LLM的硬件描述语言(HDL)代码生成使业界能够更快地实现复杂的设计，减少了开发周期所需的时间和精力。然而，对这种自动化的日益依赖带来了严重的安全风险。值得注意的是，鉴于LLM必须接受大量代码数据集的培训，这些代码通常来自公开可用的存储库(通常没有进行彻底验证)，因此LLM很容易受到所谓的数据中毒或后门攻击。在这里，攻击者为训练数据注入恶意代码，这些代码可以被带到LLMS生成的HDL码中。这种威胁媒介可能会危及整个硬件系统的安全性和完整性。在这项工作中，我们提出了一种新的后门攻击框架RTL-Breaker，用于基于LLM的硬件描述语言代码生成。RTL-Breaker提供了对这个新问题的基本方面的深入分析：1)各种触发机制与其插入恶意修改的有效性的对比，以及2)后门攻击对代码生成的一般副作用，即对代码质量的影响。RTL-Breaker强调迫切需要采取更强有力的措施来防范此类攻击。为此，我们将我们的框架和所有数据开源。



## **12. PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning**

PEFTGuard：检测针对参数高效微调的后门攻击 cs.CR

20 pages, 8 figures

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.17453v1) [paper-pdf](http://arxiv.org/pdf/2411.17453v1)

**Authors**: Zhen Sun, Tianshuo Cong, Yule Liu, Chenhao Lin, Xinlei He, Rongmao Chen, Xingshuo Han, Xinyi Huang

**Abstract**: Fine-tuning is an essential process to improve the performance of Large Language Models (LLMs) in specific domains, with Parameter-Efficient Fine-Tuning (PEFT) gaining popularity due to its capacity to reduce computational demands through the integration of low-rank adapters. These lightweight adapters, such as LoRA, can be shared and utilized on open-source platforms. However, adversaries could exploit this mechanism to inject backdoors into these adapters, resulting in malicious behaviors like incorrect or harmful outputs, which pose serious security risks to the community. Unfortunately, few of the current efforts concentrate on analyzing the backdoor patterns or detecting the backdoors in the adapters.   To fill this gap, we first construct (and will release) PADBench, a comprehensive benchmark that contains 13,300 benign and backdoored adapters fine-tuned with various datasets, attack strategies, PEFT methods, and LLMs. Moreover, we propose PEFTGuard, the first backdoor detection framework against PEFT-based adapters. Extensive evaluation upon PADBench shows that PEFTGuard outperforms existing detection methods, achieving nearly perfect detection accuracy (100%) in most cases. Notably, PEFTGuard exhibits zero-shot transferability on three aspects, including different attacks, PEFT methods, and adapter ranks. In addition, we consider various adaptive attacks to demonstrate the high robustness of PEFTGuard. We further explore several possible backdoor mitigation defenses, finding fine-mixing to be the most effective method. We envision our benchmark and method can shed light on future LLM backdoor detection research.

摘要: 微调是提高大型语言模型(LLM)在特定领域中性能的关键过程，参数高效微调(PEFT)因其能够通过集成低阶适配器来减少计算需求而广受欢迎。这些轻量级适配器，如Lora，可以在开源平台上共享和使用。然而，攻击者可以利用这一机制将后门注入这些适配器，导致不正确或有害的输出等恶意行为，这会给社区带来严重的安全风险。不幸的是，目前很少有人专注于分析后门模式或检测适配器中的后门。为了填补这一空白，我们首先构建(并将发布)PADB边，这是一个全面的基准测试，包含13,300个良性和反向适配器，通过各种数据集、攻击策略、PEFT方法和LLM进行了微调。此外，我们还提出了第一个针对基于PEFT的适配器的后门检测框架PEFTGuard。对PADBENCH的广泛评估表明，PEFTGuard的性能优于现有的检测方法，在大多数情况下实现了近乎完美的检测准确率(100%)。值得注意的是，PEFTGuard在三个方面表现出零命中率，包括不同的攻击、PEFT方法和适配器级别。此外，我们还考虑了各种自适应攻击，以展示PEFTGuard的高健壮性。我们进一步探索了几种可能的后门缓解防御措施，发现精细混合是最有效的方法。我们希望我们的基准和方法可以为未来的LLM后门检测研究提供参考。



## **13. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

通过自我提示校准对微调大型语言模型的实用成员推断攻击 cs.CL

Repo: https://github.com/tsinghua-fib-lab/NeurIPS2024_SPV-MIA

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2311.06062v4) [paper-pdf](http://arxiv.org/pdf/2311.06062v4)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Existing MIAs designed for large language models (LLMs) can be bifurcated into two types: reference-free and reference-based attacks. Although reference-based attacks appear promising performance by calibrating the probability measured on the target model with reference models, this illusion of privacy risk heavily depends on a reference dataset that closely resembles the training set. Both two types of attacks are predicated on the hypothesis that training records consistently maintain a higher probability of being sampled. However, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. Thus, these reasons lead to high false-positive rates of MIAs in practical scenarios. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs. Furthermore, we introduce probabilistic variation, a more reliable membership signal based on LLM memorization rather than overfitting, from which we rediscover the neighbour attack with theoretical grounding. Comprehensive evaluation conducted on three datasets and four exemplary LLMs shows that SPV-MIA raises the AUC of MIAs from 0.7 to a significantly high level of 0.9. Our code and dataset are available at: https://github.com/tsinghua-fib-lab/NeurIPS2024_SPV-MIA

摘要: 成员关系推理攻击(MIA)的目的是推断目标数据记录是否已被用于模型训练。现有的针对大型语言模型的MIA可以分为两种类型：无引用攻击和基于引用的攻击。虽然基于参考模型的攻击通过使用参考模型校准在目标模型上测量的概率而显示出良好的性能，但这种隐私风险的错觉严重依赖于与训练集非常相似的参考数据集。这两种类型的攻击都是基于这样的假设，即训练记录始终保持更高的被抽样概率。然而，这一假设严重依赖于目标模型的过拟合，而多种正则化方法和LLMS的推广将缓解这一问题。因此，这些原因导致在实际场景中MIA的假阳性率很高。提出了一种基于自校准概率变异的成员推理攻击(SPV-MIA)。具体地说，我们引入了一种自我提示的方法，它构造数据集，通过提示目标LLM本身来微调参考模型。通过这种方式，攻击者可以从公共API收集具有类似分布的数据集。此外，我们引入了概率变异，这是一种基于LLM记忆而不是过拟合的更可靠的成员信号，从而重新发现了具有理论基础的邻居攻击。在三个数据集和四个示范性LLM上进行的综合评估表明，SPV-MIA将MIA的AUC从0.7提高到0.9的显著高水平。我们的代码和数据集可在以下网址获得：https://github.com/tsinghua-fib-lab/NeurIPS2024_SPV-MIA



## **14. FATH: Authentication-based Test-time Defense against Indirect Prompt Injection Attacks**

FASH：基于身份验证的测试时防御间接提示注入攻击 cs.CR

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2410.21492v2) [paper-pdf](http://arxiv.org/pdf/2410.21492v2)

**Authors**: Jiongxiao Wang, Fangzhou Wu, Wendi Li, Jinsheng Pan, Edward Suh, Z. Morley Mao, Muhao Chen, Chaowei Xiao

**Abstract**: Large language models (LLMs) have been widely deployed as the backbone with additional tools and text information for real-world applications. However, integrating external information into LLM-integrated applications raises significant security concerns. Among these, prompt injection attacks are particularly threatening, where malicious instructions injected in the external text information can exploit LLMs to generate answers as the attackers desire. While both training-time and test-time defense methods have been developed to mitigate such attacks, the unaffordable training costs associated with training-time methods and the limited effectiveness of existing test-time methods make them impractical. This paper introduces a novel test-time defense strategy, named Formatting AuThentication with Hash-based tags (FATH). Unlike existing approaches that prevent LLMs from answering additional instructions in external text, our method implements an authentication system, requiring LLMs to answer all received instructions with a security policy and selectively filter out responses to user instructions as the final output. To achieve this, we utilize hash-based authentication tags to label each response, facilitating accurate identification of responses according to the user's instructions and improving the robustness against adaptive attacks. Comprehensive experiments demonstrate that our defense method can effectively defend against indirect prompt injection attacks, achieving state-of-the-art performance under Llama3 and GPT3.5 models across various attack methods. Our code is released at: https://github.com/Jayfeather1024/FATH

摘要: 大型语言模型(LLM)已被广泛部署为主干，并为实际应用程序提供额外的工具和文本信息。然而，将外部信息集成到LLM集成的应用程序中会引发重大的安全问题。其中，即时注入攻击尤其具有威胁性，在外部文本信息中注入的恶意指令可以利用LLMS生成攻击者想要的答案。虽然已经开发了训练时间和测试时间防御方法来缓解此类攻击，但与训练时间方法相关的难以负担的训练成本以及现有测试时间方法的有限有效性使它们变得不切实际。提出了一种新的测试时间防御策略--基于Hash标签的格式化认证(FATH)。与现有的阻止LLMS应答外部文本中的额外指令的方法不同，我们的方法实现了一个身份验证系统，要求LLMS用安全策略应答所有接收到的指令，并选择性地过滤对用户指令的响应作为最终输出。为了实现这一点，我们使用基于散列的身份验证标签来标记每个响应，便于根据用户的指令准确识别响应，并提高了对自适应攻击的健壮性。综合实验表明，我们的防御方法能够有效防御间接即时注入攻击，在Llama3和GPT3.5模型下通过各种攻击方法获得了最好的性能。我们的代码发布在：https://github.com/Jayfeather1024/FATH



## **15. Just-in-Time Detection of Silent Security Patches**

实时检测无声安全补丁 cs.CR

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2312.01241v3) [paper-pdf](http://arxiv.org/pdf/2312.01241v3)

**Authors**: Xunzhu Tang, Zhenghan Chen, Kisub Kim, Haoye Tian, Saad Ezzini, Jacques Klein

**Abstract**: Open-source code is pervasive. In this setting, embedded vulnerabilities are spreading to downstream software at an alarming rate. While such vulnerabilities are generally identified and addressed rapidly, inconsistent maintenance policies may lead security patches to go unnoticed. Indeed, security patches can be {\em silent}, i.e., they do not always come with comprehensive advisories such as CVEs. This lack of transparency leaves users oblivious to available security updates, providing ample opportunity for attackers to exploit unpatched vulnerabilities. Consequently, identifying silent security patches just in time when they are released is essential for preventing n-day attacks, and for ensuring robust and secure maintenance practices. With LLMDA we propose to (1) leverage large language models (LLMs) to augment patch information with generated code change explanations, (2) design a representation learning approach that explores code-text alignment methodologies for feature combination, (3) implement a label-wise training with labelled instructions for guiding the embedding based on security relevance, and (4) rely on a probabilistic batch contrastive learning mechanism for building a high-precision identifier of security patches. We evaluate LLMDA on the PatchDB and SPI-DB literature datasets and show that our approach substantially improves over the state-of-the-art, notably GraphSPD by 20% in terms of F-Measure on the SPI-DB benchmark.

摘要: 开源代码无处不在。在这种情况下，嵌入的漏洞正以惊人的速度蔓延到下游软件。虽然此类漏洞通常可以快速识别和解决，但不一致的维护策略可能会导致安全补丁不被注意到。事实上，安全补丁可以是静默的，也就是说，它们并不总是附带全面的建议，如CVE。这种透明度的缺乏让用户对可用的安全更新视而不见，为攻击者提供了充分的机会来利用未打补丁的漏洞。因此，在发布静默安全补丁时及时识别它们，对于防止n天攻击和确保强大而安全的维护做法至关重要。对于LLMDA，我们提出：(1)利用大语言模型(LLMS)通过生成代码变化解释来增强补丁信息；(2)设计一种表示学习方法，探索用于特征组合的代码-文本对齐方法；(3)利用标记指令实现基于标签的训练来指导基于安全相关性的嵌入；(4)依靠概率批量对比学习机制来构建高精度的安全补丁识别器。我们在PatchDB和SPI-DB文献数据集上评估了LLMDA，并表明我们的方法比最先进的方法有很大的改进，特别是在SPI-DB基准上的F-MEASURE方面，GraphSPD提高了20%。



## **16. Scaling Laws for Black box Adversarial Attacks**

黑匣子对抗攻击的缩放定律 cs.LG

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16782v1) [paper-pdf](http://arxiv.org/pdf/2411.16782v1)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: A longstanding problem of deep learning models is their vulnerability to adversarial examples, which are often generated by applying imperceptible perturbations to natural examples. Adversarial examples exhibit cross-model transferability, enabling to attack black-box models with limited information about their architectures and parameters. Model ensembling is an effective strategy to improve the transferability by attacking multiple surrogate models simultaneously. However, as prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the findings in large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. By analyzing the relationship between the number of surrogate models and transferability of adversarial examples, we conclude with clear scaling laws, emphasizing the potential of using more surrogate models to enhance adversarial transferability. Extensive experiments verify the claims on standard image classifiers, multimodal large language models, and even proprietary models like GPT-4o, demonstrating consistent scaling effects and impressive attack success rates with more surrogate models. Further studies by visualization indicate that scaled attacks bring better interpretability in semantics, indicating that the common features of models are captured.

摘要: 深度学习模型的一个长期存在的问题是它们对对抗性示例的脆弱性，这些示例通常是通过对自然示例应用不可察觉的扰动而产生的。对抗性的例子表现出跨模型的可转移性，使得能够攻击具有关于其体系结构和参数的有限信息的黑盒模型。模型集成是一种通过同时攻击多个代理模型来提高可转移性的有效策略。然而，由于先前的研究通常采用的模型很少，所以增加模型的数量是否可以进一步改善黑匣子攻击仍然是一个悬而未决的问题。受大型基础模型的启发，我们研究了黑盒对抗攻击的标度规律。通过分析代理模型的数量与对抗性实例的可转移性之间的关系，我们得出了明确的尺度律，强调了使用更多的代理模型来提高对抗性实例可转移性的潜力。广泛的实验验证了标准图像分类器、多模式大型语言模型，甚至像GPT-4o这样的专有模型的说法，展示了一致的缩放效果和令人印象深刻的攻击成功率，以及更多的代理模型。进一步的可视化研究表明，规模化攻击在语义上具有更好的可解释性，能够捕捉到模型的共性特征。



## **17. LLMPirate: LLMs for Black-box Hardware IP Piracy**

LLMPirate：针对黑匣子硬件IP盗版的LLM cs.CR

Accepted by NDSS Symposium 2025

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16111v1) [paper-pdf](http://arxiv.org/pdf/2411.16111v1)

**Authors**: Vasudev Gohil, Matthew DeLorenzo, Veera Vishwa Achuta Sai Venkat Nallam, Joey See, Jeyavijayan Rajendran

**Abstract**: The rapid advancement of large language models (LLMs) has enabled the ability to effectively analyze and generate code nearly instantaneously, resulting in their widespread adoption in software development. Following this advancement, researchers and companies have begun integrating LLMs across the hardware design and verification process. However, these highly potent LLMs can also induce new attack scenarios upon security vulnerabilities across the hardware development process. One such attack vector that has not been explored is intellectual property (IP) piracy. Given that this attack can manifest as rewriting hardware designs to evade piracy detection, it is essential to thoroughly evaluate LLM capabilities in performing this task and assess the mitigation abilities of current IP piracy detection tools.   Therefore, in this work, we propose LLMPirate, the first LLM-based technique able to generate pirated variations of circuit designs that successfully evade detection across multiple state-of-the-art piracy detection tools. We devise three solutions to overcome challenges related to integration of LLMs for hardware circuit designs, scalability to large circuits, and effectiveness, resulting in an end-to-end automated, efficient, and practical formulation. We perform an extensive experimental evaluation of LLMPirate using eight LLMs of varying sizes and capabilities and assess their performance in pirating various circuit designs against four state-of-the-art, widely-used piracy detection tools. Our experiments demonstrate that LLMPirate is able to consistently evade detection on 100% of tested circuits across every detection tool. Additionally, we showcase the ramifications of LLMPirate using case studies on IBEX and MOR1KX processors and a GPS module, that we successfully pirate. We envision that our work motivates and fosters the development of better IP piracy detection tools.

摘要: 大型语言模型(LLM)的快速发展使人们能够几乎即时有效地分析和生成代码，从而使其在软件开发中得到广泛采用。随着这一进步，研究人员和公司已经开始在硬件设计和验证过程中集成LLM。然而，这些高度强大的LLM也可以在整个硬件开发过程中针对安全漏洞诱导新的攻击场景。其中一种尚未被探索的攻击媒介是知识产权盗版。鉴于这种攻击可能表现为重写硬件设计以逃避盗版检测，因此必须彻底评估LLM在执行此任务时的能力，并评估当前IP盗版检测工具的缓解能力。因此，在这项工作中，我们提出了LLMPirate，这是第一种基于LLM的技术，能够生成能够在多种最先进的盗版检测工具中成功逃避检测的电路设计的盗版变体。我们设计了三种解决方案，以克服与硬件电路设计的LLM集成、大型电路的可扩展性和有效性相关的挑战，从而实现端到端的自动化、高效和实用的配方。我们使用八个大小和功能不同的LLM对LLMPirate进行了广泛的实验评估，并对照四种最先进的、广泛使用的盗版检测工具评估了它们在盗版各种电路设计方面的性能。我们的实验表明，LLMPirate能够在所有检测工具的100%测试电路上一致地躲避检测。此外，我们还使用IBEX和MOR1KX处理器上的案例研究以及我们成功盗版的GPS模块展示了LLMPirate的分支。我们设想，我们的工作将激励和促进更好的知识产权盗版检测工具的开发。



## **18. In-Context Experience Replay Facilitates Safety Red-Teaming of Text-to-Image Diffusion Models**

上下文体验回放促进文本到图像扩散模型的安全红色团队化 cs.LG

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16769v1) [paper-pdf](http://arxiv.org/pdf/2411.16769v1)

**Authors**: Zhi-Yi Chin, Kuan-Chen Mu, Mario Fritz, Pin-Yu Chen, Wei-Chen Chiu

**Abstract**: Text-to-image (T2I) models have shown remarkable progress, but their potential to generate harmful content remains a critical concern in the ML community. While various safety mechanisms have been developed, the field lacks systematic tools for evaluating their effectiveness against real-world misuse scenarios. In this work, we propose ICER, a novel red-teaming framework that leverages Large Language Models (LLMs) and a bandit optimization-based algorithm to generate interpretable and semantic meaningful problematic prompts by learning from past successful red-teaming attempts. Our ICER efficiently probes safety mechanisms across different T2I models without requiring internal access or additional training, making it broadly applicable to deployed systems. Through extensive experiments, we demonstrate that ICER significantly outperforms existing prompt attack methods in identifying model vulnerabilities while maintaining high semantic similarity with intended content. By uncovering that successful jailbreaking instances can systematically facilitate the discovery of new vulnerabilities, our work provides crucial insights for developing more robust safety mechanisms in T2I systems.

摘要: 文本到图像(T2I)模型已经显示出显著的进步，但它们产生有害内容的潜力仍然是ML社区的一个关键问题。虽然已经开发了各种安全机制，但该领域缺乏系统的工具来评估其针对现实世界滥用情况的有效性。在这项工作中，我们提出了ICER，一个新的红团队框架，它利用大型语言模型(LLM)和基于Bandit优化的算法来生成可解释的、有语义意义的问题提示，通过学习过去成功的红团队尝试。我们的ICER可有效探测不同T2I型号的安全机制，无需内部访问或额外培训，使其广泛适用于已部署的系统。通过大量的实验，我们证明了ICER在识别模型漏洞方面明显优于现有的即时攻击方法，同时保持了与预期内容的高度语义相似度。通过揭示成功的越狱实例可以系统地促进新漏洞的发现，我们的工作为在T2I系统中开发更强大的安全机制提供了至关重要的见解。



## **19. Vaccine: Perturbation-aware Alignment for Large Language Models against Harmful Fine-tuning Attack**

疫苗：大型语言模型的扰动感知对齐以对抗有害的微调攻击 cs.LG

Rejected by ICML2024. Accepted by NeurIPS2024

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2402.01109v6) [paper-pdf](http://arxiv.org/pdf/2402.01109v6)

**Authors**: Tiansheng Huang, Sihao Hu, Ling Liu

**Abstract**: The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompts. Our code is available at \url{https://github.com/git-disl/Vaccine}.

摘要: 精调即服务的新范式为大型语言模型(LLM)引入了一个新的攻击面：用户上传的少量有害数据就可以很容易地欺骗精调，产生一个破坏对齐的模型。我们进行了实证分析，发现了一种有害的嵌入漂移现象，揭示了排列断裂效应的可能原因。受我们发现的启发，我们提出了Vaccine，一种扰动感知的对齐技术，以降低用户精调的安全风险。Vaccine的核心思想是通过在比对阶段逐步向其添加精心制作的扰动来产生不变的隐藏嵌入。这使嵌入能够在精细调整阶段抵御来自未清理的用户数据的有害干扰。我们在开源主流LLMS(如Llama2、Opt、Vicuna)上的实验结果表明，疫苗可以提高对有害提示导致的嵌入漂移的健壮性，同时保留对良性提示的推理能力。我们的代码可在\url{https://github.com/git-disl/Vaccine}.



## **20. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

AmpleGCG：学习通用且可转移的对抗性后缀生成模型，用于越狱开放和封闭LLM cs.CL

Published as a conference paper at COLM 2024  (https://colmweb.org/index.html)

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2404.07921v3) [paper-pdf](http://arxiv.org/pdf/2404.07921v3)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.

摘要: 随着大型语言模型(LLM)变得越来越普遍并集成到自治系统中，确保它们的安全性是当务之急。尽管在安全对齐方面取得了长足的进步，但最近的工作GCG~\Citep{zou2023Universal}提出了一种离散令牌优化算法，并选择损失最小的单个后缀来成功越狱对齐LLM。在这项工作中，我们首先讨论了在GCG优化越狱过程中只选择损失最小的后缀的缺点，并在中间步骤中发现了遗漏的成功后缀。此外，我们利用这些成功的后缀作为训练数据来学习一种名为AmpleGCG的生成模型，该模型捕获给定有害查询的对抗性后缀的分布，并在几秒钟内为任何有害查询快速生成数百个后缀。AmpleGCG在两个对齐的LLM(Llama-2-7B-Chat和Vicuna-7B)上达到了近100%的攻击成功率(ASR)，超过了两个最强的攻击基线。更有趣的是，AmpleGCG还无缝传输以攻击不同的型号，包括闭源LLMS，在最新的GPT-3.5上实现了99\%的ASR。总之，我们的工作通过训练对抗性后缀的生成模型来放大GCG的影响，该模型对任何有害的查询都是通用的，并且可以从攻击开源LLM转移到闭源LLM。此外，它可以在短短4秒内为一个有害的查询生成200个敌意后缀，使其更具挑战性。



## **21. JailBreakV: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

JailBreakV：评估多模式大型语言模型针对越狱攻击的稳健性的基准 cs.CR

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2404.03027v4) [paper-pdf](http://arxiv.org/pdf/2404.03027v4)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.

摘要: 随着多模式大型语言模型(MLLMS)的快速发展，保护这些模型不受恶意输入的影响，同时使它们与人类的价值观保持一致，已经成为一项关键的挑战。在本文中，我们研究了一个重要而未被探索的问题，即成功越狱大语言模型(LLMS)的技术是否可以在越狱MLLM中同样有效。为了探讨这一问题，我们引入了JailBreakV-28K，这是一个开创性的基准测试，旨在评估LLM越狱技术到MLLM的可转移性，从而评估MLLMS对各种越狱攻击的健壮性。利用本文提出的包含2,000个恶意查询的数据集，我们使用针对LLMS的高级越狱攻击生成了20,000个基于文本的越狱提示，以及来自最近MLLMS越狱攻击的8,000个基于图像的越狱输入，我们的综合数据集包括来自各种对抗场景的28,000个测试用例。我们对10个开源MLLMS的评估显示，对于从LLMS转移的攻击，攻击成功率(ASR)非常高，这突显了MLLMS中源于其文本处理能力的一个严重漏洞。我们的发现强调了未来研究的迫切需要，以解决MLLMS中从文本和视觉输入的对齐漏洞。



## **22. InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models**

InjecGuard：对标和缓解即时注射保障模型中的过度防御 cs.CL

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2410.22770v2) [paper-pdf](http://arxiv.org/pdf/2410.22770v2)

**Authors**: Hao Li, Xiaogeng Liu

**Abstract**: Prompt injection attacks pose a critical threat to large language models (LLMs), enabling goal hijacking and data leakage. Prompt guard models, though effective in defense, suffer from over-defense -- falsely flagging benign inputs as malicious due to trigger word bias. To address this issue, we introduce NotInject, an evaluation dataset that systematically measures over-defense across various prompt guard models. NotInject contains 339 benign samples enriched with trigger words common in prompt injection attacks, enabling fine-grained evaluation. Our results show that state-of-the-art models suffer from over-defense issues, with accuracy dropping close to random guessing levels (60%). To mitigate this, we propose InjecGuard, a novel prompt guard model that incorporates a new training strategy, Mitigating Over-defense for Free (MOF), which significantly reduces the bias on trigger words. InjecGuard demonstrates state-of-the-art performance on diverse benchmarks including NotInject, surpassing the existing best model by 30.8%, offering a robust and open-source solution for detecting prompt injection attacks. The code and datasets are released at https://github.com/SaFoLab-WISC/InjecGuard.

摘要: 快速注入攻击对大型语言模型(LLM)构成严重威胁，导致目标劫持和数据泄露。即时保护模式虽然在防御方面有效，但也存在过度防御的问题--由于触发单词偏见，错误地将良性输入标记为恶意输入。为了解决这个问题，我们引入了NotInject，这是一个评估数据集，系统地测量各种提示防护模型中的过度防御。NotInject包含339个良性样本，丰富了提示注入攻击中常见的触发字，实现了细粒度评估。我们的结果表明，最先进的模型存在过度防御的问题，准确率下降到接近随机猜测的水平(60%)。为了缓解这一问题，我们提出了InjecGuard，一种新的提示守卫模型，它结合了新的训练策略，缓解了过度防御For Free(MOF)，大大减少了对触发词的偏见。InjecGuard在包括NotInject在内的各种基准测试上展示了最先进的性能，比现有最好的模型高出30.8%，为检测即时注入攻击提供了一个强大的开源解决方案。代码和数据集在https://github.com/SaFoLab-WISC/InjecGuard.上发布



## **23. Semantic Shield: Defending Vision-Language Models Against Backdooring and Poisoning via Fine-grained Knowledge Alignment**

语义盾牌：通过细粒度知识对齐保护视觉语言模型免受后门和毒害 cs.CV

CVPR 2024

**SubmitDate**: 2024-11-23    [abs](http://arxiv.org/abs/2411.15673v1) [paper-pdf](http://arxiv.org/pdf/2411.15673v1)

**Authors**: Alvi Md Ishmam, Christopher Thomas

**Abstract**: In recent years there has been enormous interest in vision-language models trained using self-supervised objectives. However, the use of large-scale datasets scraped from the web for training also makes these models vulnerable to potential security threats, such as backdooring and poisoning attacks. In this paper, we propose a method for mitigating such attacks on contrastively trained vision-language models. Our approach leverages external knowledge extracted from a language model to prevent models from learning correlations between image regions which lack strong alignment with external knowledge. We do this by imposing constraints to enforce that attention paid by the model to visual regions is proportional to the alignment of those regions with external knowledge. We conduct extensive experiments using a variety of recent backdooring and poisoning attacks on multiple datasets and architectures. Our results clearly demonstrate that our proposed approach is highly effective at defending against such attacks across multiple settings, while maintaining model utility and without requiring any changes at inference time

摘要: 近年来，人们对使用自我监督目标训练的视觉语言模型产生了极大的兴趣。然而，使用从网络上抓取的大规模数据集进行训练也使这些模型容易受到潜在的安全威胁，如回溯和中毒攻击。在这篇文章中，我们提出了一种方法来减轻对对比训练的视觉语言模型的攻击。我们的方法利用从语言模型中提取的外部知识来防止模型学习图像区域之间的相关性，这些区域与外部知识缺乏很强的一致性。我们通过施加约束来强制模型对可视区域的关注与这些区域与外部知识的对齐成比例来实现这一点。我们使用最近对多个数据集和体系结构的各种回溯和中毒攻击进行了广泛的实验。我们的结果清楚地表明，我们提出的方法对于在多个设置下防御此类攻击是非常有效的，同时保持了模型效用，并且不需要在推理时进行任何改变



## **24. "Moralized" Multi-Step Jailbreak Prompts: Black-Box Testing of Guardrails in Large Language Models for Verbal Attacks**

“道德化”多步骤越狱预言：对大型语言模型中护栏进行黑匣子测试以进行言语攻击 cs.CR

This paper has been submitted to ICLR 2025 BlogPosts and OpenReview  preprints. It has 9 pages of text, 4 figures, and 3 tables

**SubmitDate**: 2024-11-23    [abs](http://arxiv.org/abs/2411.16730v1) [paper-pdf](http://arxiv.org/pdf/2411.16730v1)

**Authors**: Libo Wang

**Abstract**: As the application of large language models continues to expand in various fields, it poses higher challenges to the effectiveness of identifying harmful content generation and guardrail mechanisms. This research aims to evaluate the effectiveness of guardrails in the face of multi-step jailbreak prompt-generated verbal attacks, through black-box testing of seemingly ethical prompt simulations. The experimental subjects were selected GPT-4o, Grok-2 Beta, Llama 3.1 (405B), Gemini 1.5 and Claude 3.5 Sonnet. The researcher used the same multi-step prompt to simulate moral attacks by designing a scenario of "enterprise middle managers competing for promotion" and observed the model's response at each step. During the experiment, the guardrails of the above model were all bypassed in this experiment and the content of verbal attacks was generated. The data results show that Claude 3.5 Sonnet performs better than other models in terms of its tendency to identify jailbreak prompts. The researcher hopes to use this to remind developers and future research that guardrails not only inappropriately play the role of content filters, but should also have a preventive function. In order to ensure the objectivity and generalizability of the experiment, the researcher has uploaded the experimental process, black box test code, and enhanced guardrail code to GitHub to promote cooperation in the development community: https://github.com/brucewang123456789/GeniusTrail.git.

摘要: 随着大型语言模型在各个领域的应用不断扩大，对识别有害内容生成和防护机制的有效性提出了更高的挑战。本研究旨在通过对看似符合伦理道德的提示模拟进行黑盒测试，评估护栏在面对多步骤越狱提示生成的言语攻击时的有效性。实验对象为GPT-40、Grok-2 Beta、Llama 3.1(405B)、Gemini 1.5和Claude 3.5十四行诗。研究人员使用相同的多步骤提示模拟道德攻击，设计了一个“企业中层管理者竞相晋升”的场景，并观察了模型在每一步的反应。在实验过程中，上述模型的护栏在实验中都被绕过，生成了言语攻击的内容。数据结果表明，克劳德3.5十四行诗在识别越狱提示方面比其他模型表现得更好。研究人员希望借此提醒开发者和未来的研究，护栏不仅要不当地扮演内容过滤器的角色，还应该具有预防功能。为了确保实验的客观性和通用性，研究人员将实验流程、黑盒测试代码、增强护栏代码上传到GitHub，以促进开发社区的合作：https://github.com/brucewang123456789/GeniusTrail.git.



## **25. Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning**

Geminio：联邦学习中的灰度引导梯度反转攻击 cs.LG

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.14937v1) [paper-pdf](http://arxiv.org/pdf/2411.14937v1)

**Authors**: Junjie Shan, Ziqi Zhao, Jialin Lu, Rui Zhang, Siu Ming Yiu, Ka-Ho Chow

**Abstract**: Foundation models that bridge vision and language have made significant progress, inspiring numerous life-enriching applications. However, their potential for misuse to introduce new threats remains largely unexplored. This paper reveals that vision-language models (VLMs) can be exploited to overcome longstanding limitations in gradient inversion attacks (GIAs) within federated learning (FL), where an FL server reconstructs private data samples from gradients shared by victim clients. Current GIAs face challenges in reconstructing high-resolution images, especially when the victim has a large local data batch. While focusing reconstruction on valuable samples rather than the entire batch is promising, existing methods lack the flexibility to allow attackers to specify their target data. In this paper, we introduce Geminio, the first approach to transform GIAs into semantically meaningful, targeted attacks. Geminio enables a brand new privacy attack experience: attackers can describe, in natural language, the types of data they consider valuable, and Geminio will prioritize reconstruction to focus on those high-value samples. This is achieved by leveraging a pretrained VLM to guide the optimization of a malicious global model that, when shared with and optimized by a victim, retains only gradients of samples that match the attacker-specified query. Extensive experiments demonstrate Geminio's effectiveness in pinpointing and reconstructing targeted samples, with high success rates across complex datasets under FL and large batch sizes and showing resilience against existing defenses.

摘要: 沟通愿景和语言的基础模型取得了重大进展，激发了无数丰富生活的应用程序。然而，它们被滥用以引入新威胁的潜力在很大程度上仍未被发掘。本文揭示了视觉语言模型(VLM)可以被用来克服联合学习(FL)中梯度反转攻击(GIA)的长期局限性，即FL服务器根据受害客户共享的梯度重建私有数据样本。目前的GIA在重建高分辨率图像方面面临挑战，特别是当受害者拥有大量本地数据时。虽然专注于有价值的样本而不是整个批次的重建是有希望的，但现有方法缺乏灵活性，无法允许攻击者指定他们的目标数据。在本文中，我们介绍了Geminio，第一种将GIA转换为语义有意义的有针对性的攻击的方法。Geminio提供了全新的隐私攻击体验：攻击者可以用自然语言描述他们认为有价值的数据类型，Geminio将优先重建以专注于这些高价值的样本。这是通过利用预先训练的VLM来指导恶意全局模型的优化来实现的，该恶意全局模型在与受害者共享并由受害者优化时，仅保留与攻击者指定的查询匹配的样本的梯度。广泛的实验证明了Geminio在精确定位和重建目标样本方面的有效性，在FL和大批量下的复杂数据集上具有高成功率，并显示出对现有防御系统的弹性。



## **26. Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Language Models**

谁能抵御聊天音频攻击？大型语言模型的评估基准 cs.SD

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.14842v1) [paper-pdf](http://arxiv.org/pdf/2411.14842v1)

**Authors**: Wanqi Yang, Yanda Li, Meng Fang, Yunchao Wei, Tianyi Zhou, Ling Chen

**Abstract**: Adversarial audio attacks pose a significant threat to the growing use of large language models (LLMs) in voice-based human-machine interactions. While existing research has primarily focused on model-specific adversarial methods, real-world applications demand a more generalizable and universal approach to audio adversarial attacks. In this paper, we introduce the Chat-Audio Attacks (CAA) benchmark including four distinct types of audio attacks, which aims to explore the the vulnerabilities of LLMs to these audio attacks in conversational scenarios. To evaluate the robustness of LLMs, we propose three evaluation strategies: Standard Evaluation, utilizing traditional metrics to quantify model performance under attacks; GPT-4o-Based Evaluation, which simulates real-world conversational complexities; and Human Evaluation, offering insights into user perception and trust. We evaluate six state-of-the-art LLMs with voice interaction capabilities, including Gemini-1.5-Pro, GPT-4o, and others, using three distinct evaluation methods on the CAA benchmark. Our comprehensive analysis reveals the impact of four types of audio attacks on the performance of these models, demonstrating that GPT-4o exhibits the highest level of resilience.

摘要: 对抗性音频攻击对在基于语音的人机交互中越来越多地使用大型语言模型(LLM)构成了严重威胁。虽然现有的研究主要集中在特定模型的对抗方法上，但现实世界的应用需要一种更具普遍性和通用性的方法来应对音频对抗攻击。在本文中，我们介绍了包括四种不同类型的音频攻击的聊天-音频攻击(CAA)基准，旨在探讨LLMS在会话场景中对这些音频攻击的脆弱性。为了评估LLMS的健壮性，我们提出了三种评估策略：标准评估，利用传统的度量来量化模型在攻击下的性能；基于GPT-40的评估，模拟现实世界对话的复杂性；以及人的评估，提供对用户感知和信任的洞察。我们在CAA基准上使用三种不同的评估方法来评估六种最先进的具有语音交互功能的LLM，包括Gemini-1.5-Pro、GPT-4o和其他。我们的综合分析揭示了四种类型的音频攻击对这些模型性能的影响，表明GPT-4o表现出最高水平的弹性。



## **27. Universal and Context-Independent Triggers for Precise Control of LLM Outputs**

通用且与上下文无关的触发器，用于精确控制LLM输出 cs.CL

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.14738v1) [paper-pdf](http://arxiv.org/pdf/2411.14738v1)

**Authors**: Jiashuo Liang, Guancheng Li, Yang Yu

**Abstract**: Large language models (LLMs) have been widely adopted in applications such as automated content generation and even critical decision-making systems. However, the risk of prompt injection allows for potential manipulation of LLM outputs. While numerous attack methods have been documented, achieving full control over these outputs remains challenging, often requiring experienced attackers to make multiple attempts and depending heavily on the prompt context. Recent advancements in gradient-based white-box attack techniques have shown promise in tasks like jailbreaks and system prompt leaks. Our research generalizes gradient-based attacks to find a trigger that is (1) Universal: effective irrespective of the target output; (2) Context-Independent: robust across diverse prompt contexts; and (3) Precise Output: capable of manipulating LLM inputs to yield any specified output with high accuracy. We propose a novel method to efficiently discover such triggers and assess the effectiveness of the proposed attack. Furthermore, we discuss the substantial threats posed by such attacks to LLM-based applications, highlighting the potential for adversaries to taking over the decisions and actions made by AI agents.

摘要: 大型语言模型(LLM)已广泛应用于自动内容生成甚至关键决策系统等应用中。然而，快速注入的风险允许潜在地操纵LLM输出。虽然已经记录了许多攻击方法，但要完全控制这些输出仍然具有挑战性，通常需要有经验的攻击者进行多次尝试，并且严重依赖于提示上下文。基于梯度的白盒攻击技术的最新进展在越狱和系统提示泄漏等任务中显示出了希望。我们的研究将基于梯度的攻击概括为：(1)通用：有效，与目标输出无关；(2)上下文无关：对不同的提示上下文具有健壮性；(3)精确输出：能够操纵LLM输入，以高精度产生任何指定的输出。我们提出了一种新的方法来有效地发现这些触发因素并评估所提出的攻击的有效性。此外，我们还讨论了此类攻击对基于LLM的应用程序构成的实质性威胁，强调了对手接管人工智能代理所做决策和行动的潜力。



## **28. Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics**

探索机器人学中视觉-语言-动作模型的对抗脆弱性 cs.RO

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.13587v2) [paper-pdf](http://arxiv.org/pdf/2411.13587v2)

**Authors**: Taowen Wang, Dongfang Liu, James Chenhao Liang, Wenhao Yang, Qifan Wang, Cheng Han, Jiebo Luo, Ruixiang Tang

**Abstract**: Recently in robotics, Vision-Language-Action (VLA) models have emerged as a transformative approach, enabling robots to execute complex tasks by integrating visual and linguistic inputs within an end-to-end learning framework. While VLA models offer significant capabilities, they also introduce new attack surfaces, making them vulnerable to adversarial attacks. With these vulnerabilities largely unexplored, this paper systematically quantifies the robustness of VLA-based robotic systems. Recognizing the unique demands of robotic execution, our attack objectives target the inherent spatial and functional characteristics of robotic systems. In particular, we introduce an untargeted position-aware attack objective that leverages spatial foundations to destabilize robotic actions, and a targeted attack objective that manipulates the robotic trajectory. Additionally, we design an adversarial patch generation approach that places a small, colorful patch within the camera's view, effectively executing the attack in both digital and physical environments. Our evaluation reveals a marked degradation in task success rates, with up to a 100\% reduction across a suite of simulated robotic tasks, highlighting critical security gaps in current VLA architectures. By unveiling these vulnerabilities and proposing actionable evaluation metrics, this work advances both the understanding and enhancement of safety for VLA-based robotic systems, underscoring the necessity for developing robust defense strategies prior to physical-world deployments.

摘要: 最近在机器人学中，视觉-语言-动作(VLA)模型作为一种变革性的方法出现，使机器人能够通过在端到端学习框架内整合视觉和语言输入来执行复杂的任务。虽然VLA模型提供了重要的功能，但它们也引入了新的攻击面，使其容易受到对手攻击。由于这些漏洞在很大程度上是未知的，本文系统地量化了基于VLA的机器人系统的健壮性。认识到机器人执行的独特需求，我们的攻击目标针对机器人系统固有的空间和功能特征。特别是，我们引入了一个利用空间基础来破坏机器人动作稳定性的无目标位置感知攻击目标，以及一个操纵机器人轨迹的目标攻击目标。此外，我们设计了一种对抗性补丁生成方法，将一个小的、五颜六色的补丁放置在相机的视野中，在数字和物理环境中有效地执行攻击。我们的评估显示任务成功率显著下降，一组模拟机器人任务最多减少100%，突出了当前VLA架构中的关键安全漏洞。通过揭示这些漏洞并提出可操作的评估指标，这项工作促进了对基于VLA的机器人系统安全性的理解和增强，强调了在物理世界部署之前开发强大的防御策略的必要性。



## **29. Adversarial Prompt Distillation for Vision-Language Models**

视觉语言模型的对抗性即时蒸馏 cs.CV

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.15244v1) [paper-pdf](http://arxiv.org/pdf/2411.15244v1)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-Training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical scenarios like autonomous driving and medical diagnosis. One promising approach for improving the robustness of pre-trained VLMs is Adversarial Prompt Tuning (APT), which combines adversarial training with prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose a novel method called Adversarial Prompt Distillation (APD) that combines APT with knowledge distillation to boost the adversarial robustness of CLIP. Specifically, APD is a bimodal method that adds prompts for both the visual and textual modalities while leveraging a cleanly pre-trained teacher CLIP model to distill and boost the performance of the student CLIP model on downstream tasks. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD over the current state-of-the-art APT methods in terms of both natural and adversarial performances. The effectiveness of our APD method validates the possibility of using a non-robust teacher to improve the generalization and robustness of VLMs.

摘要: 大型预先训练的视觉语言模型(VLM)，如对比语言-图像预训练(CLIP)，已被证明容易受到对抗性攻击，这引发了人们对它们在自动驾驶和医疗诊断等安全关键场景中部署的担忧。对抗性即时调谐(APT)是一种改善预先训练的VLMS的稳健性的有前途的方法，它结合了对抗性训练和快速调谐。然而，现有的APT方法大多是单通道方法，仅为视觉或文本通道设计提示(S)，限制了它们在健壮性或清晰准确性方面的有效性。在这项工作中，我们提出了一种新的方法，称为对抗性提示蒸馏(APD)，结合APT和知识提取来提高CLIP的对抗性健壮性。具体地说，apd是一种双峰方法，它添加了对视觉和文本模式的提示，同时利用经过干净的预先训练的教师剪辑模型来提取和提高学生剪辑模型在下游任务中的性能。在多个基准数据集上的大量实验表明，无论是在自然性能方面还是在对抗性性能方面，我们的apd方法都优于目前最先进的apt方法。该方法的有效性验证了使用非健壮的教师来提高VLMS的泛化和健壮性的可能性。



## **30. Memory Backdoor Attacks on Neural Networks**

对神经网络的内存后门攻击 cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14516v1) [paper-pdf](http://arxiv.org/pdf/2411.14516v1)

**Authors**: Eden Luzon, Guy Amit, Roy Weiss, Yisroel Mirsky

**Abstract**: Neural networks, such as image classifiers, are frequently trained on proprietary and confidential datasets. It is generally assumed that once deployed, the training data remains secure, as adversaries are limited to query response interactions with the model, where at best, fragments of arbitrary data can be inferred without any guarantees on their authenticity. In this paper, we propose the memory backdoor attack, where a model is covertly trained to memorize specific training samples and later selectively output them when triggered with an index pattern. What makes this attack unique is that it (1) works even when the tasks conflict (making a classifier output images), (2) enables the systematic extraction of training samples from deployed models and (3) offers guarantees on the extracted authenticity of the data. We demonstrate the attack on image classifiers, segmentation models, and a large language model (LLM). We demonstrate the attack on image classifiers, segmentation models, and a large language model (LLM). With this attack, it is possible to hide thousands of images and texts in modern vision architectures and LLMs respectively, all while maintaining model performance. The memory back door attack poses a significant threat not only to conventional model deployments but also to federated learning paradigms and other modern frameworks. Therefore, we suggest an efficient and effective countermeasure that can be immediately applied and advocate for further work on the topic.

摘要: 神经网络，如图像分类器，经常针对专有和机密数据集进行训练。通常认为，一旦部署，训练数据仍然是安全的，因为对手被限制为与模型进行查询响应交互，在最好的情况下，可以推断任意数据的片段，而不对其真实性进行任何保证。在本文中，我们提出了记忆后门攻击，即一个模型被秘密训练来记忆特定的训练样本，然后在被索引模式触发时选择性地输出它们。这种攻击的独特之处在于：(1)即使任务冲突(使分类器输出图像)也能工作，(2)能够从部署的模型中系统地提取训练样本，以及(3)对提取的数据的真实性提供保证。我们演示了对图像分类器、分割模型和大型语言模型(LLM)的攻击。我们演示了对图像分类器、分割模型和大型语言模型(LLM)的攻击。利用这种攻击，可以分别在现代视觉体系结构和LLM中隐藏数千个图像和文本，同时保持模型性能。Memory后门攻击不仅对传统模型部署构成重大威胁，还对联合学习范例和其他现代框架构成重大威胁。因此，我们建议立即采取有效的对策，并主张就这一专题开展进一步的工作。



## **31. GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs**

GISP：针对越狱LLM的对抗性后缀的高效黑匣子生成 cs.LG

28 pages, 9 tables, 13 figures; under review at CVPR '25

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14133v1) [paper-pdf](http://arxiv.org/pdf/2411.14133v1)

**Authors**: Advik Raj Basani, Xiao Zhang

**Abstract**: Large Language Models (LLMs) have shown impressive proficiency across a range of natural language processing tasks yet remain vulnerable to adversarial prompts, known as jailbreak attacks, carefully designed to elicit harmful responses from LLMs. Traditional methods rely on manual heuristics, which suffer from limited generalizability. While being automatic, optimization-based attacks often produce unnatural jailbreak prompts that are easy to detect by safety filters or require high computational overhead due to discrete token optimization. Witnessing the limitations of existing jailbreak methods, we introduce Generative Adversarial Suffix Prompter (GASP), a novel framework that combines human-readable prompt generation with Latent Bayesian Optimization (LBO) to improve adversarial suffix creation in a fully black-box setting. GASP leverages LBO to craft adversarial suffixes by efficiently exploring continuous embedding spaces, gradually optimizing the model to improve attack efficacy while balancing prompt coherence through a targeted iterative refinement procedure. Our experiments show that GASP can generate natural jailbreak prompts, significantly improving attack success rates, reducing training times, and accelerating inference speed, thus making it an efficient and scalable solution for red-teaming LLMs.

摘要: 大型语言模型(LLM)在一系列自然语言处理任务中表现出令人印象深刻的熟练程度，但仍然容易受到对手提示的攻击，这种提示被称为越狱攻击，这些提示是精心设计的，旨在引起LLM的有害反应。传统的方法依赖于人工启发式方法，泛化能力有限。虽然基于优化的攻击是自动的，但通常会产生不自然的越狱提示，这些提示很容易被安全过滤器检测到，或者由于离散令牌优化而需要很高的计算开销。鉴于现有越狱方法的局限性，我们引入了生成性对抗性后缀提示器(GAP)，这是一种将人类可读的提示生成与潜在贝叶斯优化(LBO)相结合的新框架，以改进完全黑盒环境下的对抗性后缀创建。GASP利用LBO通过有效地探索连续嵌入空间来创建对抗性后缀，逐步优化模型以提高攻击效率，同时通过有针对性的迭代细化过程平衡即时一致性。我们的实验表明，GAP能够生成自然的越狱提示，显著提高了攻击成功率，减少了训练次数，加快了推理速度，从而使其成为红队LLMS的一种高效和可扩展的解决方案。



## **32. RAG-Thief: Scalable Extraction of Private Data from Retrieval-Augmented Generation Applications with Agent-based Attacks**

RAG-Thief：利用基于代理的攻击从检索增强生成应用程序中可扩展地提取私人数据 cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14110v1) [paper-pdf](http://arxiv.org/pdf/2411.14110v1)

**Authors**: Changyue Jiang, Xudong Pan, Geng Hong, Chenfu Bao, Min Yang

**Abstract**: While large language models (LLMs) have achieved notable success in generative tasks, they still face limitations, such as lacking up-to-date knowledge and producing hallucinations. Retrieval-Augmented Generation (RAG) enhances LLM performance by integrating external knowledge bases, providing additional context which significantly improves accuracy and knowledge coverage. However, building these external knowledge bases often requires substantial resources and may involve sensitive information. In this paper, we propose an agent-based automated privacy attack called RAG-Thief, which can extract a scalable amount of private data from the private database used in RAG applications. We conduct a systematic study on the privacy risks associated with RAG applications, revealing that the vulnerability of LLMs makes the private knowledge bases suffer significant privacy risks. Unlike previous manual attacks which rely on traditional prompt injection techniques, RAG-Thief starts with an initial adversarial query and learns from model responses, progressively generating new queries to extract as many chunks from the knowledge base as possible. Experimental results show that our RAG-Thief can extract over 70% information from the private knowledge bases within customized RAG applications deployed on local machines and real-world platforms, including OpenAI's GPTs and ByteDance's Coze. Our findings highlight the privacy vulnerabilities in current RAG applications and underscore the pressing need for stronger safeguards.

摘要: 虽然大型语言模型在生成性任务中取得了显著的成功，但它们仍然面临着局限性，如缺乏最新知识和产生幻觉。检索-增强生成(RAG)通过集成外部知识库来增强LLM性能，提供额外的上下文，从而显著提高准确性和知识覆盖率。然而，建立这些外部知识库往往需要大量资源，并可能涉及敏感信息。本文提出了一种基于代理的自动隐私攻击方法RAG-Thief，它可以从RAG应用中使用的私有数据库中提取大量可伸缩的私有数据。我们对RAG应用相关的隐私风险进行了系统的研究，揭示了LLMS的漏洞使私人知识库面临着重大的隐私风险。与以前依赖传统提示注入技术的手动攻击不同，RAG-Thief从最初的对抗性查询开始，并从模型响应中学习，逐步生成新的查询以从知识库中提取尽可能多的块。实验结果表明，我们的RAG-Thief可以从本地机器和真实平台上部署的定制RAG应用程序的私有知识库中提取70%以上的信息，包括OpenAI的GPTS和ByteDance的Coze。我们的发现突显了当前RAG应用程序中的隐私漏洞，并强调了加强保护的迫切需要。



## **33. Verifying the Robustness of Automatic Credibility Assessment**

验证自动可信度评估的稳健性 cs.CL

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2303.08032v3) [paper-pdf](http://arxiv.org/pdf/2303.08032v3)

**Authors**: Piotr Przybyła, Alexander Shvets, Horacio Saggion

**Abstract**: Text classification methods have been widely investigated as a way to detect content of low credibility: fake news, social media bots, propaganda, etc. Quite accurate models (likely based on deep neural networks) help in moderating public electronic platforms and often cause content creators to face rejection of their submissions or removal of already published texts. Having the incentive to evade further detection, content creators try to come up with a slightly modified version of the text (known as an attack with an adversarial example) that exploit the weaknesses of classifiers and result in a different output. Here we systematically test the robustness of common text classifiers against available attacking techniques and discover that, indeed, meaning-preserving changes in input text can mislead the models. The approaches we test focus on finding vulnerable spans in text and replacing individual characters or words, taking into account the similarity between the original and replacement content. We also introduce BODEGA: a benchmark for testing both victim models and attack methods on four misinformation detection tasks in an evaluation framework designed to simulate real use-cases of content moderation. The attacked tasks include (1) fact checking and detection of (2) hyperpartisan news, (3) propaganda and (4) rumours. Our experimental results show that modern large language models are often more vulnerable to attacks than previous, smaller solutions, e.g. attacks on GEMMA being up to 27\% more successful than those on BERT. Finally, we manually analyse a subset adversarial examples and check what kinds of modifications are used in successful attacks.

摘要: 文本分类方法被广泛研究为检测可信度较低的内容的一种方式：假新闻、社交媒体机器人、宣传等。相当准确的模型(可能基于深度神经网络)有助于调节公共电子平台，并经常导致内容创建者面临提交的拒绝或已发布的文本的删除。出于逃避进一步检测的动机，内容创建者试图对文本进行稍微修改的版本(称为带有敌意的示例的攻击)，以利用分类器的弱点并产生不同的输出。在这里，我们系统地测试了常见文本分类器对现有攻击技术的健壮性，并发现确实，输入文本中保持意义的变化会误导模型。我们测试的方法侧重于查找文本中易受攻击的范围，并替换单个字符或单词，同时考虑到原始内容和替换内容之间的相似性。我们还引入了Bodega：一个基准，用于在四个错误信息检测任务中测试受害者模型和攻击方法，该评估框架旨在模拟真实的内容审核用例。被攻击的任务包括(1)事实核查和检测(2)超党派新闻，(3)宣传和(4)谣言。我们的实验结果表明，现代大语言模型往往比以前的较小的解决方案更容易受到攻击，例如，对Gema的攻击比对Bert的攻击成功高达27%.最后，我们手动分析了一个子集的敌意例子，并检查了在成功的攻击中使用了哪些修改。



## **34. Global Challenge for Safe and Secure LLMs Track 1**

安全可靠的LLC全球挑战第1轨 cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14502v1) [paper-pdf](http://arxiv.org/pdf/2411.14502v1)

**Authors**: Xiaojun Jia, Yihao Huang, Yang Liu, Peng Yan Tan, Weng Kuan Yau, Mun-Thye Mak, Xin Ming Sim, Wee Siong Ng, See Kiong Ng, Hanqing Liu, Lifeng Zhou, Huanqian Yan, Xiaobing Sun, Wei Liu, Long Wang, Yiming Qian, Yong Liu, Junxiao Yang, Zhexin Zhang, Leqi Lei, Renmiao Chen, Yida Lu, Shiyao Cui, Zizhou Wang, Shaohua Li, Yan Wang, Rick Siow Mong Goh, Liangli Zhen, Yingjie Zhang, Zhe Zhao

**Abstract**: This paper introduces the Global Challenge for Safe and Secure Large Language Models (LLMs), a pioneering initiative organized by AI Singapore (AISG) and the CyberSG R&D Programme Office (CRPO) to foster the development of advanced defense mechanisms against automated jailbreaking attacks. With the increasing integration of LLMs in critical sectors such as healthcare, finance, and public administration, ensuring these models are resilient to adversarial attacks is vital for preventing misuse and upholding ethical standards. This competition focused on two distinct tracks designed to evaluate and enhance the robustness of LLM security frameworks. Track 1 tasked participants with developing automated methods to probe LLM vulnerabilities by eliciting undesirable responses, effectively testing the limits of existing safety protocols within LLMs. Participants were challenged to devise techniques that could bypass content safeguards across a diverse array of scenarios, from offensive language to misinformation and illegal activities. Through this process, Track 1 aimed to deepen the understanding of LLM vulnerabilities and provide insights for creating more resilient models.

摘要: 本文介绍了全球安全和安全大语言模型挑战(LLMS)，这是由新加坡人工智能(AISG)和CyberSG研发计划办公室(CRPO)组织的一项开创性倡议，旨在促进针对自动越狱攻击的高级防御机制的发展。随着低成本管理在医疗、金融和公共管理等关键行业的日益整合，确保这些模型对对抗性攻击具有弹性，对于防止滥用和维护道德标准至关重要。这场比赛集中在两个不同的轨道上，旨在评估和增强LLM安全框架的健壮性。Track 1要求参与者开发自动化方法，通过引发不良响应来探测LLM漏洞，有效地测试LLMS中现有安全协议的限制。参与者被要求设计技术，以绕过各种情况下的内容保护，从攻击性语言到错误信息和非法活动。通过这一过程，Track 1旨在加深对LLM漏洞的理解，并为创建更具弹性的模型提供见解。



## **35. Next-Generation Phishing: How LLM Agents Empower Cyber Attackers**

下一代网络钓鱼：LLM代理如何为网络攻击者提供帮助 cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.13874v1) [paper-pdf](http://arxiv.org/pdf/2411.13874v1)

**Authors**: Khalifa Afane, Wenqi Wei, Ying Mao, Junaid Farooq, Juntao Chen

**Abstract**: The escalating threat of phishing emails has become increasingly sophisticated with the rise of Large Language Models (LLMs). As attackers exploit LLMs to craft more convincing and evasive phishing emails, it is crucial to assess the resilience of current phishing defenses. In this study we conduct a comprehensive evaluation of traditional phishing detectors, such as Gmail Spam Filter, Apache SpamAssassin, and Proofpoint, as well as machine learning models like SVM, Logistic Regression, and Naive Bayes, in identifying both traditional and LLM-rephrased phishing emails. We also explore the emerging role of LLMs as phishing detection tools, a method already adopted by companies like NTT Security Holdings and JPMorgan Chase. Our results reveal notable declines in detection accuracy for rephrased emails across all detectors, highlighting critical weaknesses in current phishing defenses. As the threat landscape evolves, our findings underscore the need for stronger security controls and regulatory oversight on LLM-generated content to prevent its misuse in creating advanced phishing attacks. This study contributes to the development of more effective Cyber Threat Intelligence (CTI) by leveraging LLMs to generate diverse phishing variants that can be used for data augmentation, harnessing the power of LLMs to enhance phishing detection, and paving the way for more robust and adaptable threat detection systems.

摘要: 随着大型语言模型(LLM)的兴起，钓鱼电子邮件日益升级的威胁变得越来越复杂。随着攻击者利用LLMS来编制更具说服力和闪避性的网络钓鱼电子邮件，评估当前网络钓鱼防御的弹性至关重要。在这项研究中，我们对传统的钓鱼检测器进行了全面的评估，如Gmail垃圾邮件过滤器、ApacheSpamassassin和Proofpoint，以及机器学习模型如支持向量机、Logistic回归和朴素贝叶斯，在识别传统和LLM重述的钓鱼电子邮件方面进行了全面的评估。我们还探讨了LLMS作为钓鱼检测工具的新兴角色，这种方法已经被NTT Security Holdings和摩根大通等公司采用。我们的结果显示，在所有检测器上，对重新措辞的电子邮件的检测准确率都出现了显著下降，突显了当前网络钓鱼防御系统的关键弱点。随着威胁格局的演变，我们的发现强调了对LLM生成的内容进行更严格的安全控制和监管的必要性，以防止其在制造高级网络钓鱼攻击时被滥用。这项研究有助于开发更有效的网络威胁情报(CTI)，方法是利用LLMS生成可用于数据增强的各种网络钓鱼变体，利用LLMS的能力来增强网络钓鱼检测，并为更强大和适应性更强的威胁检测系统铺平道路。



## **36. Rethinking the Intermediate Features in Adversarial Attacks: Misleading Robotic Models via Adversarial Distillation**

重新思考对抗性攻击的中间特征：通过对抗性蒸馏误导机器人模型 cs.LG

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.15222v1) [paper-pdf](http://arxiv.org/pdf/2411.15222v1)

**Authors**: Ke Zhao, Huayang Huang, Miao Li, Yu Wu

**Abstract**: Language-conditioned robotic learning has significantly enhanced robot adaptability by enabling a single model to execute diverse tasks in response to verbal commands. Despite these advancements, security vulnerabilities within this domain remain largely unexplored. This paper addresses this gap by proposing a novel adversarial prompt attack tailored to language-conditioned robotic models. Our approach involves crafting a universal adversarial prefix that induces the model to perform unintended actions when added to any original prompt. We demonstrate that existing adversarial techniques exhibit limited effectiveness when directly transferred to the robotic domain due to the inherent robustness of discretized robotic action spaces. To overcome this challenge, we propose to optimize adversarial prefixes based on continuous action representations, circumventing the discretization process. Additionally, we identify the beneficial impact of intermediate features on adversarial attacks and leverage the negative gradient of intermediate self-attention features to further enhance attack efficacy. Extensive experiments on VIMA models across 13 robot manipulation tasks validate the superiority of our method over existing approaches and demonstrate its transferability across different model variants.

摘要: 受语言制约的机器人学习通过使单个模型能够执行不同的任务来响应语言命令，大大增强了机器人的适应性。尽管取得了这些进步，但该域中的安全漏洞在很大程度上仍未被发现。针对这一缺陷，本文提出了一种新的针对语言条件机器人模型的对抗性即时攻击。我们的方法包括创建一个通用的对抗性前缀，当添加到任何原始提示时，该前缀会诱导模型执行意想不到的操作。我们证明，由于离散的机器人动作空间固有的健壮性，现有的对抗性技术在直接转移到机器人领域时表现出有限的有效性。为了克服这一挑战，我们提出了基于连续动作表示的对抗性前缀优化，绕过了离散化过程。此外，我们识别中间特征对对抗性攻击的有利影响，并利用中间自我注意特征的负梯度来进一步提高攻击效能。在13个机器人操作任务的VIMA模型上的广泛实验验证了我们的方法相对于现有方法的优越性，并证明了它在不同模型变体之间的可转移性。



## **37. TransLinkGuard: Safeguarding Transformer Models Against Model Stealing in Edge Deployment**

TransLinkGuard：保护Transformer模型，防止边缘部署中的模型窃取 cs.CR

Accepted by ACM MM24 Conference

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2404.11121v2) [paper-pdf](http://arxiv.org/pdf/2404.11121v2)

**Authors**: Qinfeng Li, Zhiqiang Shen, Zhenghan Qin, Yangfan Xie, Xuhong Zhang, Tianyu Du, Jianwei Yin

**Abstract**: Proprietary large language models (LLMs) have been widely applied in various scenarios. Additionally, deploying LLMs on edge devices is trending for efficiency and privacy reasons. However, edge deployment of proprietary LLMs introduces new security challenges: edge-deployed models are exposed as white-box accessible to users, enabling adversaries to conduct effective model stealing (MS) attacks. Unfortunately, existing defense mechanisms fail to provide effective protection. Specifically, we identify four critical protection properties that existing methods fail to simultaneously satisfy: (1) maintaining protection after a model is physically copied; (2) authorizing model access at request level; (3) safeguarding runtime reverse engineering; (4) achieving high security with negligible runtime overhead. To address the above issues, we propose TransLinkGuard, a plug-and-play model protection approach against model stealing on edge devices. The core part of TransLinkGuard is a lightweight authorization module residing in a secure environment, e.g., TEE. The authorization module can freshly authorize each request based on its input. Extensive experiments show that TransLinkGuard achieves the same security protection as the black-box security guarantees with negligible overhead.

摘要: 专有的大型语言模型(LLM)已广泛应用于各种场景。此外，出于效率和隐私的原因，在边缘设备上部署LLM是一种趋势。然而，专有LLMS的边缘部署带来了新的安全挑战：边缘部署的模型暴露为用户可访问的白盒，使对手能够进行有效的模型窃取(MS)攻击。不幸的是，现有的防御机制未能提供有效的保护。具体地说，我们确定了现有方法无法同时满足的四个关键保护性质：(1)在物理复制模型后保持保护；(2)在请求级授权模型访问；(3)保护运行时逆向工程；(4)以可忽略的运行时开销实现高安全性。为了解决上述问题，我们提出了一种针对边缘设备上的模型窃取的即插即用模型保护方法TransLinkGuard。TransLinkGuard的核心部分是驻留在安全环境中的轻量级授权模块，例如TEE。授权模块可以基于其输入对每个请求进行新的授权。大量实验表明，TransLinkGuard实现了与黑盒安全保证相同的安全保护，而开销可以忽略不计。



## **38. AttentionBreaker: Adaptive Evolutionary Optimization for Unmasking Vulnerabilities in LLMs through Bit-Flip Attacks**

AttributionBreaker：通过位翻转攻击揭露LLM中漏洞的自适应进化优化 cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.13757v1) [paper-pdf](http://arxiv.org/pdf/2411.13757v1)

**Authors**: Sanjay Das, Swastik Bhattacharya, Souvik Kundu, Shamik Kundu, Anand Menon, Arnab Raha, Kanad Basu

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing (NLP), excelling in tasks like text generation and summarization. However, their increasing adoption in mission-critical applications raises concerns about hardware-based threats, particularly bit-flip attacks (BFAs). BFAs, enabled by fault injection methods such as Rowhammer, target model parameters in memory, compromising both integrity and performance. Identifying critical parameters for BFAs in the vast parameter space of LLMs poses significant challenges. While prior research suggests transformer-based architectures are inherently more robust to BFAs compared to traditional deep neural networks, we challenge this assumption. For the first time, we demonstrate that as few as three bit-flips can cause catastrophic performance degradation in an LLM with billions of parameters. Current BFA techniques are inadequate for exploiting this vulnerability due to the difficulty of efficiently identifying critical parameters within the immense parameter space. To address this, we propose AttentionBreaker, a novel framework tailored for LLMs that enables efficient traversal of the parameter space to identify critical parameters. Additionally, we introduce GenBFA, an evolutionary optimization strategy designed to refine the search further, isolating the most critical bits for an efficient and effective attack. Empirical results reveal the profound vulnerability of LLMs to AttentionBreaker. For example, merely three bit-flips (4.129 x 10^-9% of total parameters) in the LLaMA3-8B-Instruct 8-bit quantized (W8) model result in a complete performance collapse: accuracy on MMLU tasks drops from 67.3% to 0%, and Wikitext perplexity skyrockets from 12.6 to 4.72 x 10^5. These findings underscore the effectiveness of AttentionBreaker in uncovering and exploiting critical vulnerabilities within LLM architectures.

摘要: 大型语言模型(LLM)使自然语言处理(NLP)发生了革命性的变化，在文本生成和摘要等任务中表现出色。然而，它们在任务关键型应用中的日益采用引发了对基于硬件的威胁的担忧，特别是位翻转攻击(BFA)。由Rowhammer等故障注入方法启用的BFA以内存中的模型参数为目标，损害了完整性和性能。在LLMS庞大的参数空间中识别BFA的关键参数带来了巨大的挑战。虽然先前的研究表明，与传统的深度神经网络相比，基于变压器的体系结构对BFA具有更强的鲁棒性，但我们对这一假设提出了质疑。我们首次证明，在具有数十亿个参数的LLM中，仅三个比特翻转就会导致灾难性的性能下降。由于很难在巨大的参数空间中有效地识别关键参数，因此当前的BFA技术不足以利用该漏洞。为了解决这个问题，我们提出了AttentionBreaker，这是一个为LLMS量身定做的新框架，能够有效地遍历参数空间来识别关键参数。此外，我们还引入了GenBFA，这是一种进化优化策略，旨在进一步细化搜索，隔离最关键的比特，以实现高效和有效的攻击。实证结果揭示了LLMS对AttentionBreaker的严重脆弱性。例如，在LLAMA3-8B指令8位量化(W8)模型中，仅三次位翻转(占总参数的4.129 x 10^-9%)就会导致完全的性能崩溃：MMLU任务的准确率从67.3%下降到0%，而Wikitext的复杂性从12.6x10^5飙升到4.72x10^5。这些发现突显了AttentionBreaker在发现和利用LLM体系结构中的关键漏洞方面的有效性。



## **39. SoK: A Systems Perspective on Compound AI Threats and Countermeasures**

SoK：复合人工智能威胁和对策的系统视角 cs.CR

13 pages, 4 figures, 2 tables

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13459v1) [paper-pdf](http://arxiv.org/pdf/2411.13459v1)

**Authors**: Sarbartha Banerjee, Prateek Sahu, Mulong Luo, Anjo Vahldiek-Oberwagner, Neeraja J. Yadwadkar, Mohit Tiwari

**Abstract**: Large language models (LLMs) used across enterprises often use proprietary models and operate on sensitive inputs and data. The wide range of attack vectors identified in prior research - targeting various software and hardware components used in training and inference - makes it extremely challenging to enforce confidentiality and integrity policies.   As we advance towards constructing compound AI inference pipelines that integrate multiple large language models (LLMs), the attack surfaces expand significantly. Attackers now focus on the AI algorithms as well as the software and hardware components associated with these systems. While current research often examines these elements in isolation, we find that combining cross-layer attack observations can enable powerful end-to-end attacks with minimal assumptions about the threat model. Given, the sheer number of existing attacks at each layer, we need a holistic and systemized understanding of different attack vectors at each layer.   This SoK discusses different software and hardware attacks applicable to compound AI systems and demonstrates how combining multiple attack mechanisms can reduce the threat model assumptions required for an isolated attack. Next, we systematize the ML attacks in lines with the Mitre Att&ck framework to better position each attack based on the threat model. Finally, we outline the existing countermeasures for both software and hardware layers and discuss the necessity of a comprehensive defense strategy to enable the secure and high-performance deployment of compound AI systems.

摘要: 跨企业使用的大型语言模型(LLM)通常使用专有模型，并对敏感输入和数据进行操作。在以前的研究中发现了广泛的攻击载体-以训练和推理中使用的各种软件和硬件组件为目标-这使得执行机密性和完整性策略变得极其困难。随着我们朝着构建集成多个大型语言模型(LLM)的复合AI推理管道的方向发展，攻击面显著扩大。攻击者现在把重点放在人工智能算法以及与这些系统相关的软件和硬件组件上。虽然目前的研究经常孤立地检查这些元素，但我们发现，结合跨层攻击观察可以在对威胁模型的最小假设下实现强大的端到端攻击。鉴于每一层现有攻击的绝对数量，我们需要对每一层的不同攻击载体进行全面和系统化的了解。本SOK讨论了适用于复合AI系统的不同软件和硬件攻击，并演示了如何结合多种攻击机制来减少孤立攻击所需的威胁模型假设。接下来，我们按照Mitre Att&CK框架对ML攻击进行系统化，以便更好地定位基于威胁模型的每一种攻击。最后，我们从软件和硬件两个层面概述了现有的对策，并讨论了为实现复合人工智能系统的安全和高性能部署而制定综合防御策略的必要性。



## **40. WaterPark: A Robustness Assessment of Language Model Watermarking**

WaterPark：语言模型水印的稳健性评估 cs.CR

22 pages

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13425v1) [paper-pdf](http://arxiv.org/pdf/2411.13425v1)

**Authors**: Jiacheng Liang, Zian Wang, Lauren Hong, Shouling Ji, Ting Wang

**Abstract**: To mitigate the misuse of large language models (LLMs), such as disinformation, automated phishing, and academic cheating, there is a pressing need for the capability of identifying LLM-generated texts. Watermarking emerges as one promising solution: it plants statistical signals into LLMs' generative processes and subsequently verifies whether LLMs produce given texts. Various watermarking methods (``watermarkers'') have been proposed; yet, due to the lack of unified evaluation platforms, many critical questions remain under-explored: i) What are the strengths/limitations of various watermarkers, especially their attack robustness? ii) How do various design choices impact their robustness? iii) How to optimally operate watermarkers in adversarial environments?   To fill this gap, we systematize existing LLM watermarkers and watermark removal attacks, mapping out their design spaces. We then develop WaterPark, a unified platform that integrates 10 state-of-the-art watermarkers and 12 representative attacks. More importantly, leveraging WaterPark, we conduct a comprehensive assessment of existing watermarkers, unveiling the impact of various design choices on their attack robustness. For instance, a watermarker's resilience to increasingly intensive attacks hinges on its context dependency. We further explore the best practices to operate watermarkers in adversarial environments. For instance, using a generic detector alongside a watermark-specific detector improves the security of vulnerable watermarkers. We believe our study sheds light on current LLM watermarking techniques while WaterPark serves as a valuable testbed to facilitate future research.

摘要: 为了减少对大型语言模型(LLM)的滥用，如虚假信息、自动网络钓鱼和学术作弊，迫切需要识别LLM生成的文本的能力。数字水印作为一种很有前途的解决方案出现了：它将统计信号植入LLMS的生成过程中，随后验证LLMS是否生成给定的文本。人们已经提出了各种水印方法，然而，由于缺乏统一的评估平台，许多关键问题仍然没有得到充分的探讨：i)各种水印的优点/局限性是什么，特别是它们的攻击稳健性？Ii)各种设计选择对其健壮性有何影响？三)如何在对抗性环境中以最佳方式使用水印？为了填补这一空白，我们对现有的LLM水印和水印移除攻击进行了系统化，规划了它们的设计空间。然后我们开发了Water Park，这是一个统一的平台，集成了10个最先进的水印和12个具有代表性的攻击。更重要的是，利用水上公园，我们对现有的水印进行了全面的评估，揭示了各种设计选择对其攻击健壮性的影响。例如，水印对日益激烈的攻击的适应能力取决于它的上下文依赖性。我们进一步探索在对抗性环境中操作水印的最佳实践。例如，在水印专用检测器旁边使用通用检测器可以提高易受攻击的水印的安全性。我们相信我们的研究对当前的LLM数字水印技术有一定的启发作用，同时也为以后的研究提供了一个有价值的实验平台。



## **41. CryptoFormalEval: Integrating LLMs and Formal Verification for Automated Cryptographic Protocol Vulnerability Detection**

CryptoFormalEval：集成LLM和形式验证以实现自动加密协议漏洞检测 cs.CR

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13627v1) [paper-pdf](http://arxiv.org/pdf/2411.13627v1)

**Authors**: Cristian Curaba, Denis D'Ambrosi, Alessandro Minisini, Natalia Pérez-Campanero Antolín

**Abstract**: Cryptographic protocols play a fundamental role in securing modern digital infrastructure, but they are often deployed without prior formal verification. This could lead to the adoption of distributed systems vulnerable to attack vectors. Formal verification methods, on the other hand, require complex and time-consuming techniques that lack automatization. In this paper, we introduce a benchmark to assess the ability of Large Language Models (LLMs) to autonomously identify vulnerabilities in new cryptographic protocols through interaction with Tamarin: a theorem prover for protocol verification. We created a manually validated dataset of novel, flawed, communication protocols and designed a method to automatically verify the vulnerabilities found by the AI agents. Our results about the performances of the current frontier models on the benchmark provides insights about the possibility of cybersecurity applications by integrating LLMs with symbolic reasoning systems.

摘要: 加密协议在保护现代数字基础设施方面发挥着基础作用，但它们通常在没有事先正式验证的情况下部署。这可能会导致采用容易受到攻击载体的分布式系统。另一方面，形式验证方法需要复杂且耗时的技术，而缺乏自动化。在本文中，我们引入了一个基准来评估大型语言模型（LLM）通过与Tamarin交互来自主识别新加密协议中漏洞的能力：Tamarin是协议验证的定理证明者。我们创建了一个手动验证的新颖、有缺陷的通信协议数据集，并设计了一种自动验证人工智能代理发现的漏洞的方法。我们关于当前前沿模型在基准上的性能的结果提供了关于通过将LLM与符号推理系统集成来实现网络安全应用的可能性的见解。



## **42. TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models**

TAPT：测试时对抗快速调整视觉语言模型中的鲁棒推理 cs.CV

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13136v1) [paper-pdf](http://arxiv.org/pdf/2411.13136v1)

**Authors**: Xin Wang, Kai Chen, Jiaming Zhang, Jingjing Chen, Xingjun Ma

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated excellent zero-shot generalizability across various downstream tasks. However, recent studies have shown that the inference performance of CLIP can be greatly degraded by small adversarial perturbations, especially its visual modality, posing significant safety threats. To mitigate this vulnerability, in this paper, we propose a novel defense method called Test-Time Adversarial Prompt Tuning (TAPT) to enhance the inference robustness of CLIP against visual adversarial attacks. TAPT is a test-time defense method that learns defensive bimodal (textual and visual) prompts to robustify the inference process of CLIP. Specifically, it is an unsupervised method that optimizes the defensive prompts for each test sample by minimizing a multi-view entropy and aligning adversarial-clean distributions. We evaluate the effectiveness of TAPT on 11 benchmark datasets, including ImageNet and 10 other zero-shot datasets, demonstrating that it enhances the zero-shot adversarial robustness of the original CLIP by at least 48.9% against AutoAttack (AA), while largely maintaining performance on clean examples. Moreover, TAPT outperforms existing adversarial prompt tuning methods across various backbones, achieving an average robustness improvement of at least 36.6%.

摘要: 大型预先训练的视觉语言模型(VLM)，如CLIP，已经在各种下游任务中表现出出色的零射击泛化能力。然而，最近的研究表明，CLIP的推理性能会因小的对抗性扰动而大大降低，特别是它的视觉通道，构成了严重的安全威胁。为了缓解这一漏洞，本文提出了一种新的防御方法，称为测试时间对抗性提示调整(TAPT)，以增强CLIP对视觉对抗性攻击的推理健壮性。TAPT是一种测试时防御方法，它学习防御性双峰(文本和视觉)提示，以巩固CLIP的推理过程。具体地说，它是一种无监督的方法，通过最小化多视图熵和对齐对抗性干净的分布来优化每个测试样本的防御提示。我们在11个基准数据集上对TAPT的有效性进行了评估，包括ImageNet和其他10个零镜头数据集，结果表明，它在很大程度上保持了在干净样本上的性能，但相对于AutoAttack(AA)，它至少提高了原始剪辑的零镜头对抗健壮性48.9%。此外，TAPT在不同主干上的性能优于现有的对抗性提示调优方法，实现了平均至少36.6%的健壮性改进。



## **43. When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations**

当后门说话时：通过模型生成的解释了解LLM后门攻击 cs.CR

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12701v1) [paper-pdf](http://arxiv.org/pdf/2411.12701v1)

**Authors**: Huaizhi Ge, Yiming Li, Qifan Wang, Yongfeng Zhang, Ruixiang Tang

**Abstract**: Large Language Models (LLMs) are vulnerable to backdoor attacks, where hidden triggers can maliciously manipulate model behavior. While several backdoor attack methods have been proposed, the mechanisms by which backdoor functions operate in LLMs remain underexplored. In this paper, we move beyond attacking LLMs and investigate backdoor functionality through the novel lens of natural language explanations. Specifically, we leverage LLMs' generative capabilities to produce human-understandable explanations for their decisions, allowing us to compare explanations for clean and poisoned samples. We explore various backdoor attacks and embed the backdoor into LLaMA models for multiple tasks. Our experiments show that backdoored models produce higher-quality explanations for clean data compared to poisoned data, while generating significantly more consistent explanations for poisoned data than for clean data. We further analyze the explanation generation process, revealing that at the token level, the explanation token of poisoned samples only appears in the final few transformer layers of the LLM. At the sentence level, attention dynamics indicate that poisoned inputs shift attention from the input context when generating the explanation. These findings deepen our understanding of backdoor attack mechanisms in LLMs and offer a framework for detecting such vulnerabilities through explainability techniques, contributing to the development of more secure LLMs.

摘要: 大型语言模型(LLM)容易受到后门攻击，在后门攻击中，隐藏的触发器可以恶意操纵模型行为。虽然已经提出了几种后门攻击方法，但后门功能在LLM中运行的机制仍未得到充分探索。在这篇文章中，我们超越了攻击LLM，通过自然语言解释的新视角来研究后门功能。具体地说，我们利用LLMS的生成能力来为他们的决定产生人类可以理解的解释，使我们能够比较干净和有毒样本的解释。我们探索了各种后门攻击，并将后门嵌入到骆驼模型中，以实现多种任务。我们的实验表明，与有毒数据相比，回溯模型对干净数据产生了更高质量的解释，而对有毒数据产生的解释比对干净数据产生的解释要一致得多。我们进一步分析了解释的生成过程，发现在令牌级别，有毒样本的解释令牌只出现在LLM的最后几个转换器层。在句子层面，注意动力学表明，有毒输入在生成解释时转移了对输入上下文的注意力。这些发现加深了我们对LLMS后门攻击机制的理解，并提供了一个通过可解释性技术检测此类漏洞的框架，有助于开发更安全的LLMS。



## **44. Exploring adversarial robustness of JPEG AI: methodology, comparison and new methods**

探索JPEG AI的对抗鲁棒性：方法论、比较和新方法 eess.IV

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11795v1) [paper-pdf](http://arxiv.org/pdf/2411.11795v1)

**Authors**: Egor Kovalev, Georgii Bychkov, Khaled Abud, Aleksandr Gushchin, Anna Chistyakova, Sergey Lavrushkin, Dmitriy Vatolin, Anastasia Antsiferova

**Abstract**: Adversarial robustness of neural networks is an increasingly important area of research, combining studies on computer vision models, large language models (LLMs), and others. With the release of JPEG AI - the first standard for end-to-end neural image compression (NIC) methods - the question of its robustness has become critically significant. JPEG AI is among the first international, real-world applications of neural-network-based models to be embedded in consumer devices. However, research on NIC robustness has been limited to open-source codecs and a narrow range of attacks. This paper proposes a new methodology for measuring NIC robustness to adversarial attacks. We present the first large-scale evaluation of JPEG AI's robustness, comparing it with other NIC models. Our evaluation results and code are publicly available online (link is hidden for a blind review).

摘要: 神经网络的对抗鲁棒性是一个越来越重要的研究领域，结合了对计算机视觉模型、大型语言模型（LLM）等的研究。随着JPEG AI（端到端神经图像压缩（NIC）方法的第一个标准）的发布，其稳健性问题变得至关重要。JPEG AI是首批嵌入消费设备的基于神经网络的模型的国际现实应用之一。然而，关于NIC稳健性的研究仅限于开源编解码器和范围狭窄的攻击。本文提出了一种新的方法来衡量NIC对对抗性攻击的稳健性。我们首次对JPEG AI的稳健性进行了大规模评估，并将其与其他NIC模型进行了比较。我们的评估结果和代码可在线公开（链接已隐藏，以供盲目审查）。



## **45. DAWN: Designing Distributed Agents in a Worldwide Network**

DAWN：在全球网络中设计分布式代理 cs.NI

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.22339v2) [paper-pdf](http://arxiv.org/pdf/2410.22339v2)

**Authors**: Zahra Aminiranjbar, Jianan Tang, Qiudan Wang, Shubha Pant, Mahesh Viswanathan

**Abstract**: The rapid evolution of Large Language Models (LLMs) has transformed them from basic conversational tools into sophisticated entities capable of complex reasoning and decision-making. These advancements have led to the development of specialized LLM-based agents designed for diverse tasks such as coding and web browsing. As these agents become more capable, the need for a robust framework that facilitates global communication and collaboration among them towards advanced objectives has become increasingly critical. Distributed Agents in a Worldwide Network (DAWN) addresses this need by offering a versatile framework that integrates LLM-based agents with traditional software systems, enabling the creation of agentic applications suited for a wide range of use cases. DAWN enables distributed agents worldwide to register and be easily discovered through Gateway Agents. Collaborations among these agents are coordinated by a Principal Agent equipped with reasoning strategies. DAWN offers three operational modes: No-LLM Mode for deterministic tasks, Copilot for augmented decision-making, and LLM Agent for autonomous operations. Additionally, DAWN ensures the safety and security of agent collaborations globally through a dedicated safety, security, and compliance layer, protecting the network against attackers and adhering to stringent security and compliance standards. These features make DAWN a robust network for deploying agent-based applications across various industries.

摘要: 大型语言模型的快速发展使它们从基本的对话工具转变为能够进行复杂推理和决策的复杂实体。这些进步导致了专门的基于LLM的代理的开发，这些代理专为不同的任务而设计，如编码和Web浏览。随着这些机构变得更有能力，需要一个强有力的框架，促进它们之间的全球沟通和合作，以实现更高的目标，这一需求变得越来越重要。全球网络中的分布式代理(DAW)通过提供一个通用的框架来满足这一需求，该框架将基于LLM的代理与传统软件系统集成在一起，从而能够创建适合于各种用例的代理应用程序。曙光使分布在世界各地的代理能够注册，并通过网关代理容易地被发现。这些代理之间的协作由一个配备了推理策略的委托代理来协调。曙光提供了三种操作模式：用于确定性任务的no-LLM模式，用于增强决策的Copilot模式，以及用于自主操作的LLM代理。此外，曙光公司通过专门的安全、保障和合规层确保全球代理协作的安全和保障，保护网络免受攻击者的攻击，并遵守严格的安全和合规标准。这些功能使曙光成为在不同行业部署基于代理的应用程序的强大网络。



## **46. TrojanRobot: Backdoor Attacks Against Robotic Manipulation in the Physical World**

特洛伊机器人：针对物理世界中机器人操纵的后门攻击 cs.RO

Initial version with preliminary results. We welcome any feedback or  suggestions

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11683v1) [paper-pdf](http://arxiv.org/pdf/2411.11683v1)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Yichen Wang, Wei Wan, Aishan Liu, Leo Yu Zhang

**Abstract**: Robotic manipulation refers to the autonomous handling and interaction of robots with objects using advanced techniques in robotics and artificial intelligence. The advent of powerful tools such as large language models (LLMs) and large vision-language models (LVLMs) has significantly enhanced the capabilities of these robots in environmental perception and decision-making. However, the introduction of these intelligent agents has led to security threats such as jailbreak attacks and adversarial attacks.   In this research, we take a further step by proposing a backdoor attack specifically targeting robotic manipulation and, for the first time, implementing backdoor attack in the physical world. By embedding a backdoor visual language model into the visual perception module within the robotic system, we successfully mislead the robotic arm's operation in the physical world, given the presence of common items as triggers. Experimental evaluations in the physical world demonstrate the effectiveness of the proposed backdoor attack.

摘要: 机器人操纵是指使用机器人学和人工智能的先进技术，自主处理机器人与物体的交互。大型语言模型(LLM)和大型视觉语言模型(LVLM)等强大工具的出现，大大增强了这些机器人在环境感知和决策方面的能力。然而，这些智能代理的引入导致了越狱攻击和对抗性攻击等安全威胁。在这项研究中，我们进一步提出了专门针对机器人操作的后门攻击，并首次在物理世界中实现了后门攻击。通过将后门视觉语言模型嵌入机器人系统的视觉感知模块中，我们成功地误导了机械臂在物理世界中的操作，因为存在共同的物品作为触发器。物理世界中的实验评估证明了所提出的后门攻击的有效性。



## **47. Enhancing Vision-Language Model Safety through Progressive Concept-Bottleneck-Driven Alignment**

通过渐进的概念瓶颈驱动的一致增强视觉语言模型的安全性 cs.CV

arXiv admin note: substantial text overlap with arXiv:2405.13581

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11543v1) [paper-pdf](http://arxiv.org/pdf/2411.11543v1)

**Authors**: Zhendong Liu, Yuanbi Nie, Yingshui Tan, Xiangyu Yue, Qiushi Cui, Chongjun Wang, Xiaoyong Zhu, Bo Zheng

**Abstract**: Benefiting from the powerful capabilities of Large Language Models (LLMs), pre-trained visual encoder models connected to LLMs form Vision Language Models (VLMs). However, recent research shows that the visual modality in VLMs is highly vulnerable, allowing attackers to bypass safety alignment in LLMs through visually transmitted content, launching harmful attacks. To address this challenge, we propose a progressive concept-based alignment strategy, PSA-VLM, which incorporates safety modules as concept bottlenecks to enhance visual modality safety alignment. By aligning model predictions with specific safety concepts, we improve defenses against risky images, enhancing explainability and controllability while minimally impacting general performance. Our method is obtained through two-stage training. The low computational cost of the first stage brings very effective performance improvement, and the fine-tuning of the language model in the second stage further improves the safety performance. Our method achieves state-of-the-art results on popular VLM safety benchmark.

摘要: 得益于大型语言模型的强大功能，连接到大型语言模型的预先训练的视觉编码器模型形成了视觉语言模型。然而，最近的研究表明，VLMS中的视觉通道非常容易受到攻击，使得攻击者能够通过视觉传输的内容绕过LLMS中的安全对齐，从而发起有害攻击。为了应对这一挑战，我们提出了一种基于概念的渐进式对齐策略PSA-VLM，该策略将安全模块作为概念瓶颈纳入其中，以增强视觉通道的安全对齐。通过将模型预测与特定的安全概念相结合，我们改进了对危险图像的防御，增强了可解释性和可控性，同时将对总体性能的影响降至最低。我们的方法是通过两个阶段的训练获得的。第一阶段的低运算量带来了非常有效的性能提升，第二阶段对语言模型的微调进一步提高了安全性能。我们的方法在流行的VLM安全基准上获得了最先进的结果。



## **48. Membership Inference Attack against Long-Context Large Language Models**

针对长上下文大型语言模型的成员推断攻击 cs.CL

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11424v1) [paper-pdf](http://arxiv.org/pdf/2411.11424v1)

**Authors**: Zixiong Wang, Gaoyang Liu, Yang Yang, Chen Wang

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled them to overcome their context window limitations, and demonstrate exceptional retrieval and reasoning capacities on longer context. Quesion-answering systems augmented with Long-Context Language Models (LCLMs) can automatically search massive external data and incorporate it into their contexts, enabling faithful predictions and reducing issues such as hallucinations and knowledge staleness. Existing studies targeting LCLMs mainly concentrate on addressing the so-called lost-in-the-middle problem or improving the inference effiencicy, leaving their privacy risks largely unexplored. In this paper, we aim to bridge this gap and argue that integrating all information into the long context makes it a repository of sensitive information, which often contains private data such as medical records or personal identities. We further investigate the membership privacy within LCLMs external context, with the aim of determining whether a given document or sequence is included in the LCLMs context. Our basic idea is that if a document lies in the context, it will exhibit a low generation loss or a high degree of semantic similarity to the contents generated by LCLMs. We for the first time propose six membership inference attack (MIA) strategies tailored for LCLMs and conduct extensive experiments on various popular models. Empirical results demonstrate that our attacks can accurately infer membership status in most cases, e.g., 90.66% attack F1-score on Multi-document QA datasets with LongChat-7b-v1.5-32k, highlighting significant risks of membership leakage within LCLMs input contexts. Furthermore, we examine the underlying reasons why LCLMs are susceptible to revealing such membership information.

摘要: 大型语言模型(LLM)的最新进展使它们能够克服上下文窗口的限制，并在更长的上下文中显示出出色的检索和推理能力。带有长上下文语言模型(LCLM)的问答系统可以自动搜索大量外部数据并将其合并到上下文中，从而实现准确的预测，并减少幻觉和知识陈旧等问题。现有的针对LCLM的研究主要集中在解决所谓的中间迷失问题或提高推理效率上，而对它们的隐私风险基本上没有进行研究。在本文中，我们旨在弥合这一差距，并认为将所有信息整合到长上下文中使其成为敏感信息的存储库，其中通常包含私人数据，如医疗记录或个人身份。我们进一步研究LCLMS外部上下文中的成员身份隐私，目的是确定给定的文档或序列是否包括在LCLMS上下文中。我们的基本思想是，如果一个文档位于上下文中，它将表现出与LCLM生成的内容的低生成损失或高度语义相似性。我们首次提出了六种专为LCLM定制的成员推理攻击(MIA)策略，并在各种流行的模型上进行了广泛的实验。实验结果表明，我们的攻击可以在大多数情况下准确地推断成员状态，例如，在具有LongChat-7b-v1.5-32k的多文档QA数据集上，90.66%的攻击F1-Score，突出了LCLM输入上下文中成员泄漏的显著风险。此外，我们还考察了LCLM容易泄露此类成员信息的潜在原因。



## **49. The Dark Side of Trust: Authority Citation-Driven Jailbreak Attacks on Large Language Models**

信任的阴暗面：权威引用驱动的对大型语言模型的越狱攻击 cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11407v1) [paper-pdf](http://arxiv.org/pdf/2411.11407v1)

**Authors**: Xikang Yang, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: The widespread deployment of large language models (LLMs) across various domains has showcased their immense potential while exposing significant safety vulnerabilities. A major concern is ensuring that LLM-generated content aligns with human values. Existing jailbreak techniques reveal how this alignment can be compromised through specific prompts or adversarial suffixes. In this study, we introduce a new threat: LLMs' bias toward authority. While this inherent bias can improve the quality of outputs generated by LLMs, it also introduces a potential vulnerability, increasing the risk of producing harmful content. Notably, the biases in LLMs is the varying levels of trust given to different types of authoritative information in harmful queries. For example, malware development often favors trust GitHub. To better reveal the risks with LLM, we propose DarkCite, an adaptive authority citation matcher and generator designed for a black-box setting. DarkCite matches optimal citation types to specific risk types and generates authoritative citations relevant to harmful instructions, enabling more effective jailbreak attacks on aligned LLMs.Our experiments show that DarkCite achieves a higher attack success rate (e.g., LLama-2 at 76% versus 68%) than previous methods. To counter this risk, we propose an authenticity and harm verification defense strategy, raising the average defense pass rate (DPR) from 11% to 74%. More importantly, the ability to link citations to the content they encompass has become a foundational function in LLMs, amplifying the influence of LLMs' bias toward authority.

摘要: 大型语言模型(LLM)在不同领域的广泛部署展示了它们的巨大潜力，同时也暴露了重大的安全漏洞。一个主要的问题是确保LLM生成的内容符合人类的价值观。现有的越狱技术揭示了如何通过特定的提示或对抗性后缀来破坏这种对齐。在这项研究中，我们引入了一个新的威胁：LLMS对权威的偏见。虽然这种固有的偏见可以提高低成本管理产生的产出的质量，但它也引入了一个潜在的脆弱性，增加了产生有害内容的风险。值得注意的是，LLMS中的偏差是在有害查询中对不同类型的权威信息给予的不同程度的信任。例如，恶意软件开发通常偏向信任GitHub。为了更好地揭示LLM的风险，我们提出了DarkCite，这是一个为黑箱设置而设计的自适应权威引用匹配器和生成器。DarkCite将最佳引用类型与特定的风险类型相匹配，并生成与有害指令相关的权威引用，从而对对齐的LLMS进行更有效的越狱攻击。我们的实验表明，与以前的方法相比，DarkCite实现了更高的攻击成功率(例如，骆驼-2为76%，而不是68%)。为了应对这种风险，我们提出了真实性和危害性验证防御策略，将平均防御通过率(DPR)从11%提高到74%。更重要的是，将引文与它们所包含的内容相联系的能力已经成为LLMS的一项基本功能，放大了LLMS对权威的偏见的影响。



## **50. Hacking Back the AI-Hacker: Prompt Injection as a Defense Against LLM-driven Cyberattacks**

黑客攻击人工智能黑客：即时注入作为抵御LLM驱动的网络攻击的防御 cs.CR

v0.2 (evaluated on more agents)

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.20911v2) [paper-pdf](http://arxiv.org/pdf/2410.20911v2)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: Large language models (LLMs) are increasingly being harnessed to automate cyberattacks, making sophisticated exploits more accessible and scalable. In response, we propose a new defense strategy tailored to counter LLM-driven cyberattacks. We introduce Mantis, a defensive framework that exploits LLMs' susceptibility to adversarial inputs to undermine malicious operations. Upon detecting an automated cyberattack, Mantis plants carefully crafted inputs into system responses, leading the attacker's LLM to disrupt their own operations (passive defense) or even compromise the attacker's machine (active defense). By deploying purposefully vulnerable decoy services to attract the attacker and using dynamic prompt injections for the attacker's LLM, Mantis can autonomously hack back the attacker. In our experiments, Mantis consistently achieved over 95% effectiveness against automated LLM-driven attacks. To foster further research and collaboration, Mantis is available as an open-source tool: https://github.com/pasquini-dario/project_mantis

摘要: 大型语言模型(LLM)越来越多地被用来自动化网络攻击，使复杂的利用更容易获得和可扩展。作为回应，我们提出了一种新的防御战略，以对抗LLM驱动的网络攻击。我们引入了Mantis，这是一个防御框架，利用LLMS对对手输入的敏感性来破坏恶意操作。在检测到自动网络攻击后，螳螂工厂会精心设计输入到系统响应中，导致攻击者的LLM扰乱自己的操作(被动防御)，甚至危害攻击者的机器(主动防御)。通过部署故意易受攻击的诱骗服务来吸引攻击者，并对攻击者的LLM使用动态提示注入，螳螂可以自主地攻击攻击者。在我们的实验中，螳螂对自动LLM驱动的攻击始终取得了95%以上的效率。为了促进进一步的研究和合作，Mantis以开源工具的形式提供：https://github.com/pasquini-dario/project_mantis



