# Latest Large Language Model Attack Papers
**update at 2025-02-08 16:27:46**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions**

轻松说话：通过简单的互动引发法学硕士的有害越狱 cs.LG

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04322v1) [paper-pdf](http://arxiv.org/pdf/2502.04322v1)

**Authors**: Yik Siu Chan, Narutatsu Ri, Yuxin Xiao, Marzyeh Ghassemi

**Abstract**: Despite extensive safety alignment efforts, large language models (LLMs) remain vulnerable to jailbreak attacks that elicit harmful behavior. While existing studies predominantly focus on attack methods that require technical expertise, two critical questions remain underexplored: (1) Are jailbroken responses truly useful in enabling average users to carry out harmful actions? (2) Do safety vulnerabilities exist in more common, simple human-LLM interactions? In this paper, we demonstrate that LLM responses most effectively facilitate harmful actions when they are both actionable and informative--two attributes easily elicited in multi-step, multilingual interactions. Using this insight, we propose HarmScore, a jailbreak metric that measures how effectively an LLM response enables harmful actions, and Speak Easy, a simple multi-step, multilingual attack framework. Notably, by incorporating Speak Easy into direct request and jailbreak baselines, we see an average absolute increase of 0.319 in Attack Success Rate and 0.426 in HarmScore in both open-source and proprietary LLMs across four safety benchmarks. Our work reveals a critical yet often overlooked vulnerability: Malicious users can easily exploit common interaction patterns for harmful intentions.

摘要: 尽管进行了广泛的安全调整工作，但大型语言模型(LLM)仍然容易受到引发有害行为的越狱攻击。虽然现有的研究主要集中在需要技术专业知识的攻击方法上，但仍有两个关键问题未被探索：(1)越狱反应是否真的有助于使普通用户执行有害行为？(2)更常见、更简单的人与LLM交互中是否存在安全漏洞？在这篇文章中，我们证明了当LLM响应既可操作又可提供信息时，它们最有效地促进了有害行为--这两个属性在多步骤、多语言交互中很容易引发。利用这一见解，我们提出了HarmScore，这是一种衡量LLM响应支持有害操作的效率的指标，并提出了一种简单的多步骤、多语言攻击框架。值得注意的是，通过将Stop Easy整合到直接请求和越狱基准中，我们看到在四个安全基准中，开源和专有LLM的攻击成功率平均绝对增加了0.319，HarmScore增加了0.426。我们的工作揭示了一个关键但经常被忽视的漏洞：恶意用户可以很容易地利用常见的交互模式来实现有害意图。



## **2. Can LLMs Hack Enterprise Networks? Autonomous Assumed Breach Penetration-Testing Active Directory Networks**

LLM可以黑客攻击企业网络吗？自主假设漏洞渗透测试Active目录网络 cs.CR

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04227v1) [paper-pdf](http://arxiv.org/pdf/2502.04227v1)

**Authors**: Andreas Happe, Jürgen Cito

**Abstract**: We explore the feasibility and effectiveness of using LLM-driven autonomous systems for Assumed Breach penetration testing in enterprise networks. We introduce a novel prototype that, driven by Large Language Models (LLMs), can compromise accounts within a real-life Active Directory testbed. Our research provides a comprehensive evaluation of the prototype's capabilities, and highlights both strengths and limitations while executing attack. The evaluation uses a realistic simulation environment (Game of Active Directory, GOAD) to capture intricate interactions, stochastic outcomes, and timing dependencies that characterize live network scenarios. The study concludes that autonomous LLMs are able to conduct Assumed Breach simulations, potentially democratizing access to penetration testing for organizations facing budgetary constraints.   The prototype's source code, traces, and analyzed logs are released as open-source to enhance collective cybersecurity and facilitate future research in LLM-driven cybersecurity automation.

摘要: 我们探索了使用LLM驱动的自治系统在企业网络中进行假设漏洞渗透测试的可行性和有效性。我们介绍了一个新的原型，它由大型语言模型(LLM)驱动，可以在现实生活中的Active Directory试验床中危害帐户。我们的研究提供了对原型能力的全面评估，并强调了执行攻击时的优势和局限性。该评估使用真实的模拟环境(活动目录游戏，GOAD)来捕获复杂的交互、随机结果和时间依赖关系，这些都是实时网络场景的特征。研究得出结论，自主的LLM能够进行假设的漏洞模拟，可能会使面临预算限制的组织获得渗透测试的机会大众化。原型的源代码、跟踪和分析日志以开源形式发布，以增强集体网络安全，并促进未来对LLM驱动的网络安全自动化的研究。



## **3. "Short-length" Adversarial Training Helps LLMs Defend "Long-length" Jailbreak Attacks: Theoretical and Empirical Evidence**

“短期”对抗性培训帮助法学硕士防御“长期”越狱攻击：理论和经验证据 cs.LG

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04204v1) [paper-pdf](http://arxiv.org/pdf/2502.04204v1)

**Authors**: Shaopeng Fu, Liang Ding, Di Wang

**Abstract**: Jailbreak attacks against large language models (LLMs) aim to induce harmful behaviors in LLMs through carefully crafted adversarial prompts. To mitigate attacks, one way is to perform adversarial training (AT)-based alignment, i.e., training LLMs on some of the most adversarial prompts to help them learn how to behave safely under attacks. During AT, the length of adversarial prompts plays a critical role in the robustness of aligned LLMs. This paper focuses on adversarial suffix jailbreak attacks and unveils that to defend against a jailbreak attack with an adversarial suffix of length $\Theta(M)$, it is enough to align LLMs on prompts with adversarial suffixes of length $\Theta(\sqrt{M})$. Theoretically, we analyze the adversarial in-context learning of linear transformers on linear regression tasks and prove a robust generalization bound for trained transformers. The bound depends on the term $\Theta(\sqrt{M_{\text{test}}}/M_{\text{train}})$, where $M_{\text{train}}$ and $M_{\text{test}}$ are the number of adversarially perturbed in-context samples during training and testing. Empirically, we conduct AT on popular open-source LLMs and evaluate their robustness against jailbreak attacks of different adversarial suffix lengths. Results confirm a positive correlation between the attack success rate and the ratio of the square root of the adversarial suffix during jailbreaking to the length during AT. Our findings show that it is practical to defend "long-length" jailbreak attacks via efficient "short-length" AT. The code is available at https://github.com/fshp971/adv-icl.

摘要: 针对大型语言模型(LLM)的越狱攻击旨在通过精心设计的对抗性提示在LLM中诱导有害行为。为了减轻攻击，一种方法是执行基于对抗性训练(AT)的对齐，即根据一些最具对抗性的提示对LLM进行培训，以帮助它们学习如何在攻击下安全地行为。在自动对准过程中，对抗性提示的长度对对准LLMS的稳健性起着至关重要的作用。本文主要研究对抗性后缀越狱攻击，揭示了要防御对抗性后缀长度为$\theta(M)$的越狱攻击，只需使提示上的LLMS与长度为$\theta(\Sqrt{M})$的对抗性后缀对齐即可。在理论上，我们分析了线性回归任务中线性变压器的对抗性上下文学习，并证明了训练的变压器的一个稳健的泛化上界。这个界限取决于术语$\Theta(\sqrt{M_{\text{test}}}/M_{\text{train}})$，，其中$M_{\TEXT{TEST}}$和$M_{\TEXT{TEST}}$是训练和测试过程中受到不利干扰的上下文样本的数量。经验性地，我们对流行的开源LLM进行了AT，并评估了它们对不同敌意后缀长度的越狱攻击的健壮性。结果证实，攻击成功率与越狱时敌意后缀的平方根与AT中敌意后缀的长度之比呈正相关。我们的研究结果表明，通过有效的“短长度”AT防御“长长度”越狱攻击是可行的。代码可在https://github.com/fshp971/adv-icl.上获得



## **4. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

G-Designer：通过图神经网络构建多智能体通信布局 cs.MA

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2410.11782v3) [paper-pdf](http://arxiv.org/pdf/2410.11782v3)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Tianlong Chen, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.

摘要: 基于大型语言模型(LLM)的代理的最新进展表明，集体智能可以显著超过单个代理的能力，这主要是由于精心设计的代理间通信拓扑。尽管有多样化和高性能的设计，但实践者在为他们的特定任务选择最有效的流水线时经常面临困惑：\textit{哪个拓扑是我的任务的最佳选择，在确保高质量解决方案的同时避免不必要的通信令牌开销？}针对这种困境，我们引入了G-Designer，这是一个自适应的、高效的、健壮的多代理部署解决方案，它动态地设计任务感知的、定制的通信拓扑。具体地说，G-Designer将多代理系统建模为多代理网络，利用变化图自动编码器对节点(代理)和特定于任务的虚拟节点进行编码，并解码任务自适应的高性能通信拓扑。在六个基准测试上的广泛实验表明，G-Designer是：\extbf{(1)高性能}，在MMLU上获得了更好的结果，准确率为84.50\$，在HumanEval上，PASS@1的准确率为89.90\$；\extbf{(2)任务自适应}，构建了针对任务难度的通信协议，在HumanEval上减少了高达95.33\$的令牌消耗；以及\extbf{(3)对手健壮性}，防御代理对手攻击，精确度仅下降了$0.3\%$。



## **5. On Effects of Steering Latent Representation for Large Language Model Unlearning**

论引导潜在表示对大型语言模型取消学习的影响 cs.CL

Accepted at AAAI-25 Main Technical Track

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2408.06223v3) [paper-pdf](http://arxiv.org/pdf/2408.06223v3)

**Authors**: Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue

**Abstract**: Representation Misdirection for Unlearning (RMU), which steers model representation in the intermediate layer to a target random representation, is an effective method for large language model (LLM) unlearning. Despite its high performance, the underlying cause and explanation remain underexplored. In this paper, we theoretically demonstrate that steering forget representations in the intermediate layer reduces token confidence, causing LLMs to generate wrong or nonsense responses. We investigate how the coefficient influences the alignment of forget-sample representations with the random direction and hint at the optimal coefficient values for effective unlearning across different network layers. We show that RMU unlearned models are robust against adversarial jailbreak attacks. Furthermore, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs. To resolve this drawback, we propose Adaptive RMU--a simple yet effective alternative method that makes unlearning effective with most layers. Extensive experiments demonstrate that Adaptive RMU significantly improves the unlearning performance compared to prior art while incurring no additional computational cost.

摘要: 遗忘表征误导(RMU)是一种有效的大语言模型遗忘方法，它将中间层的模型表征引导到目标随机表征。尽管其表现良好，但其根本原因和解释仍未得到充分研究。在本文中，我们从理论上证明了中间层中的转向遗忘表征降低了令牌置信度，从而导致LLM产生错误或无意义的响应。我们研究了系数如何影响遗忘样本表示与随机方向的对齐，并提示了跨不同网络层有效遗忘的最优系数值。我们证明了RMU未学习模型对敌意越狱攻击是健壮的。此外，我们的实证分析表明，当RMU应用于LLMS的中后期时，其有效性较差。为了解决这一缺陷，我们提出了自适应RMU--一种简单但有效的替代方法，使遗忘在大多数层都有效。大量实验表明，与现有技术相比，自适应RMU在不增加额外计算代价的情况下，显著改善了遗忘性能。



## **6. AdaPhish: AI-Powered Adaptive Defense and Education Resource Against Deceptive Emails**

AdaPhish：针对欺骗性电子邮件的人工智能驱动自适应防御和教育资源 cs.CR

7 pages, 3 figures, 2 tables, accepted in 4th IEEE International  Conference on AI in Cybersecurity (ICAIC)

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.03622v1) [paper-pdf](http://arxiv.org/pdf/2502.03622v1)

**Authors**: Rei Meguro, Ng S. T. Chong

**Abstract**: Phishing attacks remain a significant threat in the digital age, yet organizations lack effective methods to tackle phishing attacks without leaking sensitive information. Phish bowl initiatives are a vital part of cybersecurity efforts against these attacks. However, traditional phish bowls require manual anonymization and are often limited to internal use. To overcome these limitations, we introduce AdaPhish, an AI-powered phish bowl platform that automatically anonymizes and analyzes phishing emails using large language models (LLMs) and vector databases. AdaPhish achieves real-time detection and adaptation to new phishing tactics while enabling long-term tracking of phishing trends. Through automated reporting, adaptive analysis, and real-time alerts, AdaPhish presents a scalable, collaborative solution for phishing detection and cybersecurity education.

摘要: 网络钓鱼攻击仍然是数字时代的重大威胁，但组织缺乏有效的方法来在不泄露敏感信息的情况下应对网络钓鱼攻击。Phish bowl计划是针对这些攻击的网络安全工作的重要组成部分。然而，传统的钓鱼碗需要手动匿名化，并且通常仅限于内部使用。为了克服这些限制，我们引入了AdaPhish，这是一个人工智能驱动的钓鱼碗平台，可以使用大型语言模型（LLM）和载体数据库自动匿名化和分析网络钓鱼电子邮件。AdaPhish实现了实时检测和适应新的网络钓鱼策略，同时能够长期跟踪网络钓鱼趋势。通过自动报告、自适应分析和实时警报，AdaPhish为网络钓鱼检测和网络安全教育提供了可扩展的协作解决方案。



## **7. GLOV: Guided Large Language Models as Implicit Optimizers for Vision Language Models**

GOV：引导大型语言模型作为视觉语言模型的隐式优化器 cs.CV

Code: https://github.com/jmiemirza/GLOV

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2410.06154v5) [paper-pdf](http://arxiv.org/pdf/2410.06154v5)

**Authors**: M. Jehanzeb Mirza, Mengjie Zhao, Zhuoyuan Mao, Sivan Doveh, Wei Lin, Paul Gavrikov, Michael Dorkenwald, Shiqi Yang, Saurav Jha, Hiromi Wakaki, Yuki Mitsufuji, Horst Possegger, Rogerio Feris, Leonid Karlinsky, James Glass

**Abstract**: In this work, we propose GLOV, which enables Large Language Models (LLMs) to act as implicit optimizers for Vision-Language Models (VLMs) to enhance downstream vision tasks. GLOV prompts an LLM with the downstream task description, querying it for suitable VLM prompts (e.g., for zero-shot classification with CLIP). These prompts are ranked according to their fitness for the downstream vision task. In each respective optimization step, the ranked prompts are fed as in-context examples (with their accuracies) to equip the LLM with the knowledge of the type of prompts preferred by the downstream VLM. Furthermore, we explicitly guide the LLM's generation at each optimization step by adding an offset vector -- calculated from the embedding differences between previous positive and negative solutions -- to the intermediate layer of the network for the next generation. This offset vector biases the LLM generation toward the type of language the downstream VLM prefers, resulting in enhanced performance on the downstream vision tasks. We comprehensively evaluate our GLOV on two tasks: object recognition and the critical task of enhancing VLM safety. Our GLOV shows performance improvement by up to 15.0% and 57.5% for dual-encoder (e.g., CLIP) and encoder-decoder (e.g., LlaVA) models for object recognition and reduces the attack success rate (ASR) on state-of-the-art VLMs by up to $60.7\%$.

摘要: 在这项工作中，我们提出了GLOV，它使得大语言模型(LLM)能够作为视觉语言模型(VLMS)的隐式优化器来增强下游的视觉任务。GLOV用下游任务描述提示LLM，向其查询合适的VLM提示(例如，用于带CLIP的零射击分类)。这些提示根据它们对下游视觉任务的适宜性进行排序。在每个相应的优化步骤中，将经排序的提示作为上下文中的示例(及其准确性)馈送，以使LLM具有下游VLM优选的提示类型的知识。此外，我们在每个优化步骤通过将偏移向量添加到网络的中间层来显式地指导LLM的生成，该偏移量是根据先前正解和负解之间的嵌入差异计算的，以用于下一代。此偏移向量使LLM生成偏向于下游VLM偏爱的语言类型，从而提高了下游视觉任务的性能。我们在两个任务上对我们的GLOV进行了全面评估：目标识别和增强VLM安全的关键任务。我们的GLOV显示，对于用于对象识别的双编码器(例如，CLIP)和编解码器(例如，LlaVA)模型，性能分别提高了15.0%和57.5%，并将最先进的VLM的攻击成功率(ASR)降低了高达60.7美元。



## **8. Exploring the Security Threats of Knowledge Base Poisoning in Retrieval-Augmented Code Generation**

探索检索增强代码生成中知识库中毒的安全威胁 cs.CR

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.03233v1) [paper-pdf](http://arxiv.org/pdf/2502.03233v1)

**Authors**: Bo Lin, Shangwen Wang, Liqian Chen, Xiaoguang Mao

**Abstract**: The integration of Large Language Models (LLMs) into software development has revolutionized the field, particularly through the use of Retrieval-Augmented Code Generation (RACG) systems that enhance code generation with information from external knowledge bases. However, the security implications of RACG systems, particularly the risks posed by vulnerable code examples in the knowledge base, remain largely unexplored. This risk is particularly concerning given that public code repositories, which often serve as the sources for knowledge base collection in RACG systems, are usually accessible to anyone in the community. Malicious attackers can exploit this accessibility to inject vulnerable code into the knowledge base, making it toxic. Once these poisoned samples are retrieved and incorporated into the generated code, they can propagate security vulnerabilities into the final product. This paper presents the first comprehensive study on the security risks associated with RACG systems, focusing on how vulnerable code in the knowledge base compromises the security of generated code. We investigate the LLM-generated code security across different settings through extensive experiments using four major LLMs, two retrievers, and two poisoning scenarios. Our findings highlight the significant threat of knowledge base poisoning, where even a single poisoned code example can compromise up to 48% of generated code. Our findings provide crucial insights into vulnerability introduction in RACG systems and offer practical mitigation recommendations, thereby helping improve the security of LLM-generated code in future works.

摘要: 将大型语言模型(LLM)集成到软件开发中使该领域发生了革命性的变化，特别是通过使用检索-增强代码生成(RACG)系统，该系统利用来自外部知识库的信息来增强代码生成。然而，RACG系统的安全影响，特别是知识库中易受攻击的代码示例带来的风险，在很大程度上仍未得到探索。考虑到公共代码库通常作为RACG系统中知识库收集的来源，社区中的任何人通常都可以访问，这种风险尤其令人担忧。恶意攻击者可以利用这种可访问性将易受攻击的代码注入知识库，使其有毒。一旦这些有毒的样本被检索并合并到生成的代码中，它们就可以将安全漏洞传播到最终产品中。本文首次对RACG系统的安全风险进行了全面的研究，重点研究了知识库中易受攻击的代码如何危及生成代码的安全性。我们通过使用四个主要的LLM、两个检索器和两个中毒场景的广泛实验，研究了不同设置下LLM生成的代码的安全性。我们的发现突出了知识库中毒的重大威胁，在这种情况下，即使是一个中毒的代码示例也可能危及高达48%的生成代码。我们的发现为RACG系统中的漏洞引入提供了重要的见解，并提供了实用的缓解建议，从而有助于在未来的工作中提高LLM生成的代码的安全性。



## **9. ImgTrojan: Jailbreaking Vision-Language Models with ONE Image**

ImgTrojan：具有一张图像的越狱视觉语言模型 cs.CV

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2403.02910v3) [paper-pdf](http://arxiv.org/pdf/2403.02910v3)

**Authors**: Xijia Tao, Shuai Zhong, Lei Li, Qi Liu, Lingpeng Kong

**Abstract**: There has been an increasing interest in the alignment of large language models (LLMs) with human values. However, the safety issues of their integration with a vision module, or vision language models (VLMs), remain relatively underexplored. In this paper, we propose a novel jailbreaking attack against VLMs, aiming to bypass their safety barrier when a user inputs harmful instructions. A scenario where our poisoned (image, text) data pairs are included in the training data is assumed. By replacing the original textual captions with malicious jailbreak prompts, our method can perform jailbreak attacks with the poisoned images. Moreover, we analyze the effect of poison ratios and positions of trainable parameters on our attack's success rate. For evaluation, we design two metrics to quantify the success rate and the stealthiness of our attack. Together with a list of curated harmful instructions, a benchmark for measuring attack efficacy is provided. We demonstrate the efficacy of our attack by comparing it with baseline methods.

摘要: 人们对大型语言模型(LLM)与人类价值观的一致性越来越感兴趣。然而，它们与视觉模块或视觉语言模型(VLM)集成的安全性问题仍然相对较少被探索。在本文中，我们提出了一种新的针对VLM的越狱攻击，目的是在用户输入有害指令时绕过它们的安全屏障。假设我们的有毒(图像、文本)数据对被包括在训练数据中。通过用恶意越狱提示替换原始文本字幕，我们的方法可以使用有毒图像执行越狱攻击。此外，我们还分析了毒物比例和可训练参数的位置对攻击成功率的影响。为了进行评估，我们设计了两个度量标准来量化攻击的成功率和隐蔽性。此外，还提供了一份经过精心策划的有害指令清单，作为衡量攻击效果的基准。我们通过与基线方法进行比较来证明我们的攻击的有效性。



## **10. Understanding and Enhancing the Transferability of Jailbreaking Attacks**

了解并增强越狱攻击的可转移性 cs.LG

Accepted by ICLR 2025

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.03052v1) [paper-pdf](http://arxiv.org/pdf/2502.03052v1)

**Authors**: Runqi Lin, Bo Han, Fengwang Li, Tongling Liu

**Abstract**: Jailbreaking attacks can effectively manipulate open-source large language models (LLMs) to produce harmful responses. However, these attacks exhibit limited transferability, failing to disrupt proprietary LLMs consistently. To reliably identify vulnerabilities in proprietary LLMs, this work investigates the transferability of jailbreaking attacks by analysing their impact on the model's intent perception. By incorporating adversarial sequences, these attacks can redirect the source LLM's focus away from malicious-intent tokens in the original input, thereby obstructing the model's intent recognition and eliciting harmful responses. Nevertheless, these adversarial sequences fail to mislead the target LLM's intent perception, allowing the target LLM to refocus on malicious-intent tokens and abstain from responding. Our analysis further reveals the inherent distributional dependency within the generated adversarial sequences, whose effectiveness stems from overfitting the source LLM's parameters, resulting in limited transferability to target LLMs. To this end, we propose the Perceived-importance Flatten (PiF) method, which uniformly disperses the model's focus across neutral-intent tokens in the original input, thus obscuring malicious-intent tokens without relying on overfitted adversarial sequences. Extensive experiments demonstrate that PiF provides an effective and efficient red-teaming evaluation for proprietary LLMs.

摘要: 越狱攻击可以有效地操纵开源的大型语言模型(LLM)来产生有害的响应。然而，这些攻击表现出有限的可转移性，未能始终如一地破坏专有LLM。为了可靠地识别专有LLM中的漏洞，该工作通过分析越狱攻击对模型意图感知的影响来调查越狱攻击的可转移性。通过合并敌意序列，这些攻击可以将源LLM的焦点从原始输入中的恶意标记重新定向，从而阻碍模型的意图识别并引发有害响应。然而，这些敌对序列未能误导目标LLM的意图感知，允许目标LLM重新关注恶意令牌并放弃响应。我们的分析进一步揭示了生成的对抗序列中固有的分布依赖关系，其有效性源于过拟合源LLM的参数，导致对目标LLM的可转移性有限。为此，我们提出了感知重要性平坦化(PIF)方法，该方法将模型的焦点均匀地分散在原始输入中的中性意图标记上，从而在不依赖于过度匹配的敌对序列的情况下模糊恶意意图标记。大量实验表明，PIF为专有LLM提供了一种有效和高效的红团队评估。



## **11. SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner**

SelfDefend：LLM可以以实用的方式保护自己免受越狱的侵害 cs.CR

Accepted by USENIX Security Symposium 2025. Please cite the  conference version of this paper, i.e., "Xunguang Wang, Daoyuan Wu, Zhenlan  Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, and  Juergen Rahmel. SelfDefend: LLMs Can Defend Themselves against Jailbreaking  in a Practical Manner. In Proc. USENIX Security, 2025."

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2406.05498v3) [paper-pdf](http://arxiv.org/pdf/2406.05498v3)

**Authors**: Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, Juergen Rahmel

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into multiple categories: human-based, optimization-based, generation-based, and the recent indirect and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delays to user prompts, as well as be compatible with both open-source and closed-source LLMs. Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM as a defense instance (in detection state) to concurrently protect the target LLM instance (in normal answering state) in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs can identify harmful prompts or intentions in user queries, which we empirically validate using mainstream GPT-3.5/4 models against major jailbreak attacks. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. When deployed to protect GPT-3.5/4, Claude, Llama-2-7b/13b, and Mistral, these models outperform seven state-of-the-art defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. Further experiments show that the tuned models are robust to adaptive jailbreaks and prompt injections.

摘要: 越狱是一种新兴的对抗性攻击，它绕过了现成的大型语言模型(LLM)中部署的安全对齐，并已演变为多种类别：基于人的、基于优化的、基于代的以及最近的间接和多语言越狱。然而，提供实际的越狱防御是具有挑战性的，因为它不仅需要处理所有上述越狱攻击，还需要对用户提示造成可以忽略不计的延迟，以及与开源和闭源LLM兼容。受传统影子堆栈安全概念防御内存溢出攻击的启发，提出了一种通用的LLM越狱防御框架--SelfDefend。该框架建立一个影子LLM作为防御实例(处于检测状态)，同时保护正常堆栈中的目标LLM实例(处于正常应答状态)，并与其协作进行基于检查点的访问控制。SelfDefend的有效性建立在我们的观察基础上，即现有的LLM可以识别用户查询中的有害提示或意图，我们使用主流GPT-3.5/4模型对主要越狱攻击进行了经验验证。为了进一步提高防御的健壮性并将成本降至最低，我们使用数据蒸馏方法来优化专用的开源防御模型。当部署保护GPT-3.5/4、克劳德、Llama-2-7b/13b和米斯特拉尔时，这些型号的性能超过了七种最先进的防御系统，与基于GPT-4的SelfDefend的性能相当，额外延迟显著降低。进一步的实验表明，调整后的模型对自适应越狱和快速注入具有较强的鲁棒性。



## **12. Lost in Overlap: Exploring Logit-based Watermark Collision in LLMs**

迷失在重叠中：探索LLM中基于日志的水印碰撞 cs.CL

Long Paper, 9 pages, accepted at NAACL 2025 Findings

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2403.10020v3) [paper-pdf](http://arxiv.org/pdf/2403.10020v3)

**Authors**: Yiyang Luo, Ke Lin, Chao Gu, Jiahui Hou, Lijie Wen, Ping Luo

**Abstract**: The proliferation of large language models (LLMs) in generating content raises concerns about text copyright. Watermarking methods, particularly logit-based approaches, embed imperceptible identifiers into text to address these challenges. However, the widespread usage of watermarking across diverse LLMs has led to an inevitable issue known as watermark collision during common tasks, such as paraphrasing or translation. In this paper, we introduce watermark collision as a novel and general philosophy for watermark attacks, aimed at enhancing attack performance on top of any other attacking methods. We also provide a comprehensive demonstration that watermark collision poses a threat to all logit-based watermark algorithms, impacting not only specific attack scenarios but also downstream applications.

摘要: 生成内容时大型语言模型（LLM）的激增引发了人们对文本版权的担忧。水印方法，特别是基于日志的方法，将不可感知的标识符嵌入到文本中来解决这些挑战。然而，水印在不同的LLM中的广泛使用导致了在常见任务（例如解释或翻译）期间不可避免的问题，称为水印冲突。在本文中，我们引入水印冲突作为水印攻击的一种新颖且通用的哲学，旨在在任何其他攻击方法之上提高攻击性能。我们还全面证明了水印冲突对所有基于日志的水印算法构成威胁，不仅影响特定的攻击场景，还影响下游应用。



## **13. Large Language Model Adversarial Landscape Through the Lens of Attack Objectives**

从攻击目标角度看大语言模型的对抗格局 cs.CR

15 pages

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.02960v1) [paper-pdf](http://arxiv.org/pdf/2502.02960v1)

**Authors**: Nan Wang, Kane Walter, Yansong Gao, Alsharif Abuadbba

**Abstract**: Large Language Models (LLMs) represent a transformative leap in artificial intelligence, enabling the comprehension, generation, and nuanced interaction with human language on an unparalleled scale. However, LLMs are increasingly vulnerable to a range of adversarial attacks that threaten their privacy, reliability, security, and trustworthiness. These attacks can distort outputs, inject biases, leak sensitive information, or disrupt the normal functioning of LLMs, posing significant challenges across various applications.   In this paper, we provide a novel comprehensive analysis of the adversarial landscape of LLMs, framed through the lens of attack objectives. By concentrating on the core goals of adversarial actors, we offer a fresh perspective that examines threats from the angles of privacy, integrity, availability, and misuse, moving beyond conventional taxonomies that focus solely on attack techniques. This objective-driven adversarial landscape not only highlights the strategic intent behind different adversarial approaches but also sheds light on the evolving nature of these threats and the effectiveness of current defenses. Our analysis aims to guide researchers and practitioners in better understanding, anticipating, and mitigating these attacks, ultimately contributing to the development of more resilient and robust LLM systems.

摘要: 大型语言模型(LLM)代表了人工智能的一次革命性飞跃，使人们能够以前所未有的规模理解、生成和与人类语言进行细微差别的交互。然而，LLM越来越容易受到一系列对手攻击，这些攻击威胁到它们的隐私、可靠性、安全性和可信性。这些攻击可能会扭曲输出、注入偏差、泄露敏感信息或扰乱LLMS的正常功能，对各种应用程序构成重大挑战。在这篇文章中，我们提供了一种新颖的全面分析的对抗性景观，通过攻击目标的框架。通过专注于敌对行为者的核心目标，我们提供了一个新的视角，从隐私、完整性、可用性和误用的角度来检查威胁，超越了只关注攻击技术的传统分类。这种以目标为导向的对抗性格局不仅突出了不同对抗性方法背后的战略意图，而且也揭示了这些威胁的演变性质和目前防御的有效性。我们的分析旨在指导研究人员和实践者更好地理解、预测和缓解这些攻击，最终有助于开发更具弹性和健壮性的LLM系统。



## **14. How Much Do Code Language Models Remember? An Investigation on Data Extraction Attacks before and after Fine-tuning**

代码语言模型能记住多少？微调前后数据提取攻击的研究 cs.CR

MSR 2025

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2501.17501v2) [paper-pdf](http://arxiv.org/pdf/2501.17501v2)

**Authors**: Fabio Salerno, Ali Al-Kaswan, Maliheh Izadi

**Abstract**: Code language models, while widely popular, are often trained on unsanitized source code gathered from across the Internet. Previous work revealed that pre-trained models can remember the content of their training data and regurgitate them through data extraction attacks. Due to the large size of current models, only a few entities have the resources for pre-training such models. However, fine-tuning requires fewer resources and is increasingly used by both small and large entities for its effectiveness on specialized data. Such small curated data for fine-tuning might contain sensitive information or proprietary assets. In this study, we attack both pre-trained and fine-tuned code language models to investigate the extent of data extractability. We first develop a custom benchmark to assess the vulnerability of both pre-training and fine-tuning samples to extraction attacks. Our findings reveal that 54.9% of extractable pre-training data could be retrieved from StarCoder2-15B, whereas this number decreased to 23.5% after fine-tuning. This indicates that fine-tuning reduces the extractability of pre-training data. However, compared to larger models, fine-tuning smaller models increases their vulnerability to data extraction attacks on fine-tuning data. Given the potential sensitivity of fine-tuning data, this can lead to more severe consequences. Lastly, we also manually analyzed 2000 extractable samples before and after fine-tuning. We also found that data carriers and licensing information are the most likely data categories to be memorized from pre-trained and fine-tuned models, while the latter is the most likely to be forgotten after fine-tuning.

摘要: 代码语言模型虽然广受欢迎，但通常是针对从互联网上收集的未经清理的源代码进行培训的。以前的工作表明，预先训练的模型可以记住它们的训练数据的内容，并通过数据提取攻击来反胃它们。由于目前模型的规模很大，只有少数几个实体有资源对这些模型进行预培训。然而，微调需要的资源更少，而且越来越多地被小型和大型实体使用，因为它对专门数据的有效性。如此小的精选数据用于微调，可能包含敏感信息或专有资产。在这项研究中，我们攻击预训练和微调的代码语言模型，以调查数据可提取的程度。我们首先开发一个定制的基准来评估预先训练和微调样本对提取攻击的脆弱性。我们的研究结果表明，54.9%的可提取预训练数据可以从StarCoder2-15B中检索到，而经过微调后，这一数字下降到23.5%。这表明微调降低了训练前数据的可提取性。然而，与较大的模型相比，微调较小的模型会增加它们对微调数据的数据提取攻击的脆弱性。鉴于微调数据的潜在敏感性，这可能导致更严重的后果。最后，我们还对微调前后的2000个可提取样本进行了手工分析。我们还发现，从预先训练和微调的模型中，数据载体和许可信息是最容易被记忆的数据类别，而后者在微调后最容易被忘记。



## **15. Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models**

越狱解药：通过大型语言模型中的稀疏表示调整来实现安全与效用平衡 cs.CR

Accepted by ICLR2025. url: https://openreview.net/forum?id=s20W12XTF8

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2410.02298v3) [paper-pdf](http://arxiv.org/pdf/2410.02298v3)

**Authors**: Guobin Shen, Dongcheng Zhao, Yiting Dong, Xiang He, Yi Zeng

**Abstract**: As large language models (LLMs) become integral to various applications, ensuring both their safety and utility is paramount. Jailbreak attacks, which manipulate LLMs into generating harmful content, pose significant challenges to this balance. Existing defenses, such as prompt engineering and safety fine-tuning, often introduce computational overhead, increase inference latency, and lack runtime flexibility. Moreover, overly restrictive safety measures can degrade model utility by causing refusals of benign queries. In this paper, we introduce Jailbreak Antidote, a method that enables real-time adjustment of LLM safety preferences by manipulating a sparse subset of the model's internal states during inference. By shifting the model's hidden representations along a safety direction with varying strengths, we achieve flexible control over the safety-utility balance without additional token overhead or inference delays. Our analysis reveals that safety-related information in LLMs is sparsely distributed; adjusting approximately 5% of the internal state is as effective as modifying the entire state. Extensive experiments on nine LLMs (ranging from 2 billion to 72 billion parameters), evaluated against ten jailbreak attack methods and compared with six defense strategies, validate the effectiveness and efficiency of our approach. By directly manipulating internal states during reasoning, Jailbreak Antidote offers a lightweight, scalable solution that enhances LLM safety while preserving utility, opening new possibilities for real-time safety mechanisms in widely-deployed AI systems.

摘要: 随着大型语言模型(LLM)成为各种应用程序不可或缺的一部分，确保它们的安全性和实用性是至关重要的。越狱攻击操纵LLM生成有害内容，对这种平衡构成了重大挑战。现有的防御措施，如即时工程和安全微调，通常会引入计算开销，增加推理延迟，并且缺乏运行时灵活性。此外，过于严格的安全措施可能会导致良性查询被拒绝，从而降低模型的实用性。在本文中，我们介绍了JailBreak解毒剂，这是一种通过在推理过程中操纵模型内部状态的稀疏子集来实时调整LLM安全偏好的方法。通过沿不同强度的安全方向移动模型的隐藏表示，我们在不增加令牌开销或推理延迟的情况下实现了对安全-效用平衡的灵活控制。我们的分析表明，LLMS中与安全相关的信息是稀疏分布的；调整大约5%的内部状态与修改整个状态一样有效。在9个LLM(参数从20亿到720亿)上进行了大量的实验，对10种越狱攻击方法进行了评估，并与6种防御策略进行了比较，验证了该方法的有效性和高效性。通过在推理过程中直接操纵内部状态，越狱解毒剂提供了一个轻量级、可扩展的解决方案，在增强LLM安全性的同时保留了实用性，为广泛部署的AI系统中的实时安全机制打开了新的可能性。



## **16. SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models**

SimMark：一种针对大型语言模型的稳健基于句子级相似性的水印算法 cs.CL

15 pages, 5 tables, 6 figures

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.02787v1) [paper-pdf](http://arxiv.org/pdf/2502.02787v1)

**Authors**: Amirhossein Dabiriaghdam, Lele Wang

**Abstract**: The rapid proliferation of large language models (LLMs) has created an urgent need for reliable methods to detect whether a text is generated by such models. In this paper, we propose SimMark, a posthoc watermarking algorithm that makes LLMs' outputs traceable without requiring access to the model's internal logits, enabling compatibility with a wide range of LLMs, including API-only models. By leveraging the similarity of semantic sentence embeddings and rejection sampling to impose detectable statistical patterns imperceptible to humans, and employing a soft counting mechanism, SimMark achieves robustness against paraphrasing attacks. Experimental results demonstrate that SimMark sets a new benchmark for robust watermarking of LLM-generated content, surpassing prior sentence-level watermarking techniques in robustness, sampling efficiency, and applicability across diverse domains, all while preserving the text quality.

摘要: 大型语言模型（LLM）的迅速激增迫切需要可靠的方法来检测文本是否由此类模型生成。在本文中，我们提出了SimMark，这是一种后置水印算法，可以使LLM的输出可追溯，而无需访问模型的内部日志，从而能够与广泛的LLM兼容，包括仅API模型。通过利用语义句子嵌入和拒绝采样的相似性来强加人类难以感知的可检测统计模式，并采用软计数机制，SimMark实现了针对重述攻击的鲁棒性。实验结果表明，SimMark为LLM生成的内容的鲁棒水印设定了新的基准，在鲁棒性、采样效率和跨不同领域的适用性方面超越了先前的业务级水印技术，同时保持了文本质量。



## **17. Certifying LLM Safety against Adversarial Prompting**

针对对抗性预算认证LLM安全性 cs.CL

Accepted at COLM 2024: https://openreview.net/forum?id=9Ik05cycLq

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2309.02705v4) [paper-pdf](http://arxiv.org/pdf/2309.02705v4)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that add malicious tokens to an input prompt to bypass the safety guardrails of an LLM and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework for defending against adversarial prompts with certifiable safety guarantees. Given a prompt, our procedure erases tokens individually and inspects the resulting subsequences using a safety filter. Our safety certificate guarantees that harmful prompts are not mislabeled as safe due to an adversarial attack up to a certain size. We implement the safety filter in two ways, using Llama 2 and DistilBERT, and compare the performance of erase-and-check for the two cases. We defend against three attack modes: i) adversarial suffix, where an adversarial sequence is appended at the end of a harmful prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. Additionally, we propose three efficient empirical defenses: i) RandEC, a randomized subsampling version of erase-and-check; ii) GreedyEC, which greedily erases tokens that maximize the softmax score of the harmful class; and iii) GradEC, which uses gradient information to optimize tokens to erase. We demonstrate their effectiveness against adversarial prompts generated by the Greedy Coordinate Gradient (GCG) attack algorithm. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.

摘要: 大型语言模型(LLM)容易受到敌意攻击，这些攻击会向输入提示添加恶意令牌，以绕过LLM的安全护栏，导致其产生有害内容。在这项工作中，我们引入了Erase-and-Check，这是第一个用于防御具有可证明安全保证的对抗性提示的框架。在给定提示的情况下，我们的过程将逐个擦除令牌，并使用安全过滤器检查结果子序列。我们的安全证书保证有害提示不会因为达到一定大小的敌意攻击而被错误地标记为安全。我们用Llama 2和DistilBERT两种方法实现了安全过滤器，并比较了两种情况下的擦除和检查性能。我们防御三种攻击模式：i)对抗性后缀，其中对抗性序列被附加在有害提示的末尾；ii)对抗性插入，其中对抗性序列被插入在提示中间的任何位置；以及iii)对抗性注入，其中对抗性标记被插入在提示中的任意位置，而不一定作为连续的块。实验结果表明，该方法在对安全提示保持较好经验性能的同时，对有害提示具有较强的认证安全保障。此外，我们还提出了三种有效的经验防御：i)RandEC，一种随机化的擦除和检查版本；ii)GreedyEC，它贪婪地擦除使有害类别的Softmax得分最大化的标记；以及iii)Gradec，它使用梯度信息来优化要擦除的标记。我们证明了它们对贪婪坐标梯度(GCG)攻击算法生成的敌意提示的有效性。我们实验的代码可以在https://github.com/aounon/certified-llm-safety.上找到



## **18. Medical Multimodal Model Stealing Attacks via Adversarial Domain Alignment**

医学多模式模型通过对抗领域对齐窃取攻击 cs.CR

Accepted at AAAI 2025

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02438v1) [paper-pdf](http://arxiv.org/pdf/2502.02438v1)

**Authors**: Yaling Shen, Zhixiong Zhuang, Kun Yuan, Maria-Irina Nicolae, Nassir Navab, Nicolas Padoy, Mario Fritz

**Abstract**: Medical multimodal large language models (MLLMs) are becoming an instrumental part of healthcare systems, assisting medical personnel with decision making and results analysis. Models for radiology report generation are able to interpret medical imagery, thus reducing the workload of radiologists. As medical data is scarce and protected by privacy regulations, medical MLLMs represent valuable intellectual property. However, these assets are potentially vulnerable to model stealing, where attackers aim to replicate their functionality via black-box access. So far, model stealing for the medical domain has focused on classification; however, existing attacks are not effective against MLLMs. In this paper, we introduce Adversarial Domain Alignment (ADA-STEAL), the first stealing attack against medical MLLMs. ADA-STEAL relies on natural images, which are public and widely available, as opposed to their medical counterparts. We show that data augmentation with adversarial noise is sufficient to overcome the data distribution gap between natural images and the domain-specific distribution of the victim MLLM. Experiments on the IU X-RAY and MIMIC-CXR radiology datasets demonstrate that Adversarial Domain Alignment enables attackers to steal the medical MLLM without any access to medical data.

摘要: 医疗多模式大型语言模型(MLLMS)正在成为医疗保健系统的重要组成部分，帮助医务人员进行决策和结果分析。放射学报告生成模型能够解释医学图像，从而减少了放射科医生的工作量。由于医疗数据稀缺，而且受到隐私法规的保护，医疗MLLM代表着宝贵的知识产权。然而，这些资产可能容易受到模型窃取的攻击，攻击者的目标是通过黑盒访问来复制它们的功能。到目前为止，针对医学领域的模型窃取主要集中在分类上，然而，现有的攻击对MLLMS并不有效。在本文中，我们介绍了第一个针对医学MLLM的窃取攻击--对抗性领域对齐(ADA-Steal)。Ada-steal依赖于自然图像，这些图像是公开的，可以广泛使用，而不是医学上的同行。我们证明了使用对抗性噪声的数据增强足以克服自然图像和受害者MLLM的特定领域分布之间的数据分布差距。在Iu-X-Ray和MIMIC-CXR放射学数据集上的实验表明，对抗性领域对齐使攻击者能够在不访问任何医疗数据的情况下窃取医疗MLLM。



## **19. JailbreakEval: An Integrated Toolkit for Evaluating Jailbreak Attempts Against Large Language Models**

越狱Eval：用于评估针对大型语言模型的越狱尝试的集成工具包 cs.CR

This is the Extended Version for the Poster at NDSS Symposium 2025,  Feb 24-28, 2025. Our code is available at  https://github.com/ThuCCSLab/JailbreakEval

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2406.09321v2) [paper-pdf](http://arxiv.org/pdf/2406.09321v2)

**Authors**: Delong Ran, Jinyuan Liu, Yichen Gong, Jingyi Zheng, Xinlei He, Tianshuo Cong, Anyu Wang

**Abstract**: Jailbreak attacks induce Large Language Models (LLMs) to generate harmful responses, posing severe misuse threats. Though research on jailbreak attacks and defenses is emerging, there is no consensus on evaluating jailbreaks, i.e., the methods to assess the harmfulness of an LLM's response are varied. Each approach has its own set of strengths and weaknesses, impacting their alignment with human values, as well as the time and financial cost. This diversity challenges researchers in choosing suitable evaluation methods and comparing different attacks and defenses. In this paper, we conduct a comprehensive analysis of jailbreak evaluation methodologies, drawing from nearly 90 jailbreak research published between May 2023 and April 2024. Our study introduces a systematic taxonomy of jailbreak evaluators, offering indepth insights into their strengths and weaknesses, along with the current status of their adaptation. To aid further research, we propose JailbreakEval, a toolkit for evaluating jailbreak attempts. JailbreakEval includes various evaluators out-of-the-box, enabling users to obtain results with a single command or customized evaluation workflows. In summary, we regard JailbreakEval to be a catalyst that simplifies the evaluation process in jailbreak research and fosters an inclusive standard for jailbreak evaluation within the community.

摘要: 越狱攻击导致大型语言模型(LLM)产生有害的响应，构成严重的滥用威胁。尽管对越狱攻击和防御的研究正在兴起，但对于评估越狱还没有达成共识，即评估LLM反应的危害性的方法多种多样。每种方法都有自己的长处和短处，影响它们与人类价值观的一致性，以及时间和财务成本。这种多样性向研究人员提出了挑战，即选择合适的评估方法并比较不同的攻击和防御。本文从2023年5月至2024年4月发表的近90篇越狱研究中，对越狱评估方法进行了全面的分析。我们的研究介绍了越狱评估员的系统分类，深入了解了他们的优势和劣势，以及他们适应的现状。为了帮助进一步的研究，我们提出了JailBreak Eval，一个用于评估越狱企图的工具包。JailBreak Eval包括各种开箱即用的评估器，使用户能够通过单个命令或定制的评估工作流获得结果。总而言之，我们认为越狱评估是一种催化剂，可以简化越狱研究的评估过程，并在社区内培养一个包容性的越狱评估标准。



## **20. Is poisoning a real threat to LLM alignment? Maybe more so than you think**

中毒是对LLM联盟的真正威胁吗？也许比你想象的还要多 cs.LG

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2406.12091v3) [paper-pdf](http://arxiv.org/pdf/2406.12091v3)

**Authors**: Pankayaraj Pathmanathan, Souradip Chakraborty, Xiangyu Liu, Yongyuan Liang, Furong Huang

**Abstract**: Recent advancements in Reinforcement Learning with Human Feedback (RLHF) have significantly impacted the alignment of Large Language Models (LLMs). The sensitivity of reinforcement learning algorithms such as Proximal Policy Optimization (PPO) has led to new line work on Direct Policy Optimization (DPO), which treats RLHF in a supervised learning framework. The increased practical use of these RLHF methods warrants an analysis of their vulnerabilities. In this work, we investigate the vulnerabilities of DPO to poisoning attacks under different scenarios and compare the effectiveness of preference poisoning, a first of its kind. We comprehensively analyze DPO's vulnerabilities under different types of attacks, i.e., backdoor and non-backdoor attacks, and different poisoning methods across a wide array of language models, i.e., LLama 7B, Mistral 7B, and Gemma 7B. We find that unlike PPO-based methods, which, when it comes to backdoor attacks, require at least 4\% of the data to be poisoned to elicit harmful behavior, we exploit the true vulnerabilities of DPO more simply so we can poison the model with only as much as 0.5\% of the data. We further investigate the potential reasons behind the vulnerability and how well this vulnerability translates into backdoor vs non-backdoor attacks.

摘要: 人类反馈强化学习(RLHF)的最新进展对大型语言模型(LLM)的匹配产生了重大影响。强化学习算法的敏感性，如最近策略优化(PPO)，导致了直接策略优化(DPO)的新工作，它在监督学习框架中处理RLHF。这些RLHF方法的实际使用越来越多，因此有理由对其脆弱性进行分析。在这项工作中，我们调查了DPO在不同场景下对中毒攻击的脆弱性，并比较了偏好中毒的有效性，这是第一次。我们全面分析了DPO在不同类型的攻击下的漏洞，即后门攻击和非后门攻击，以及不同的中毒方法，跨越了广泛的语言模型，即：大羊驼7B、米斯特拉尔7B和杰玛7B。我们发现，与基于PPO的方法不同，当涉及到后门攻击时，需要至少4%的数据被毒化才能引发有害行为，而我们更简单地利用DPO的真正漏洞，因此我们只需使用多达0.5%的数据就可以毒害模型。我们进一步调查了该漏洞背后的潜在原因，以及该漏洞在多大程度上转化为后门攻击与非后门攻击。



## **21. STAIR: Improving Safety Alignment with Introspective Reasoning**

楼梯：通过内省推理改善安全性 cs.CL

22 pages, 8 figures

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02384v1) [paper-pdf](http://arxiv.org/pdf/2502.02384v1)

**Authors**: Yichi Zhang, Siyuan Zhang, Yao Huang, Zeyu Xia, Zhengwei Fang, Xiao Yang, Ranjie Duan, Dong Yan, Yinpeng Dong, Jun Zhu

**Abstract**: Ensuring the safety and harmlessness of Large Language Models (LLMs) has become equally critical as their performance in applications. However, existing safety alignment methods typically suffer from safety-performance trade-offs and the susceptibility to jailbreak attacks, primarily due to their reliance on direct refusals for malicious queries. In this paper, we propose STAIR, a novel framework that integrates SafeTy Alignment with Itrospective Reasoning. We enable LLMs to identify safety risks through step-by-step analysis by self-improving chain-of-thought (CoT) reasoning with safety awareness. STAIR first equips the model with a structured reasoning capability and then advances safety alignment via iterative preference optimization on step-level reasoning data generated using our newly proposed Safety-Informed Monte Carlo Tree Search (SI-MCTS). We further train a process reward model on this data to guide test-time searches for improved responses. Extensive experiments show that STAIR effectively mitigates harmful outputs while better preserving helpfulness, compared to instinctive alignment strategies. With test-time scaling, STAIR achieves a safety performance comparable to Claude-3.5 against popular jailbreak attacks. Relevant resources in this work are available at https://github.com/thu-ml/STAIR.

摘要: 确保大型语言模型(LLM)的安全性和无害性已变得与它们在应用程序中的性能同等重要。然而，现有的安全对齐方法通常会受到安全性能和越狱攻击之间的权衡，这主要是因为它们依赖于对恶意查询的直接拒绝。在本文中，我们提出了一种新的框架STAIR，它将安全匹配和回顾推理结合在一起。我们通过具有安全意识的自我完善的思想链(COT)推理，使LLM能够通过逐步分析来识别安全风险。STAIR首先为模型配备了结构化推理能力，然后通过迭代偏好优化对使用新提出的安全通知蒙特卡罗树搜索(SI-MCTS)生成的步进级推理数据进行安全匹配。我们进一步训练了一个基于这些数据的过程奖励模型，以指导测试时间搜索以获得更好的响应。广泛的实验表明，与本能的对齐策略相比，STAIR有效地减少了有害输出，同时更好地保留了帮助。随着测试时间的扩展，STAIR在抵御流行的越狱攻击时实现了与克劳德-3.5相当的安全性能。这项工作的相关资源可在https://github.com/thu-ml/STAIR.上获得



## **22. SHIELD: APT Detection and Intelligent Explanation Using LLM**

SHIELD：使用LLM的APT检测和智能解释 cs.CR

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02342v1) [paper-pdf](http://arxiv.org/pdf/2502.02342v1)

**Authors**: Parth Atulbhai Gandhi, Prasanna N. Wudali, Yonatan Amaru, Yuval Elovici, Asaf Shabtai

**Abstract**: Advanced persistent threats (APTs) are sophisticated cyber attacks that can remain undetected for extended periods, making their mitigation particularly challenging. Given their persistence, significant effort is required to detect them and respond effectively. Existing provenance-based attack detection methods often lack interpretability and suffer from high false positive rates, while investigation approaches are either supervised or limited to known attacks. To address these challenges, we introduce SHIELD, a novel approach that combines statistical anomaly detection and graph-based analysis with the contextual analysis capabilities of large language models (LLMs). SHIELD leverages the implicit knowledge of LLMs to uncover hidden attack patterns in provenance data, while reducing false positives and providing clear, interpretable attack descriptions. This reduces analysts' alert fatigue and makes it easier for them to understand the threat landscape. Our extensive evaluation demonstrates SHIELD's effectiveness and computational efficiency in real-world scenarios. SHIELD was shown to outperform state-of-the-art methods, achieving higher precision and recall. SHIELD's integration of anomaly detection, LLM-driven contextual analysis, and advanced graph-based correlation establishes a new benchmark for APT detection.

摘要: 高级持续性威胁(APT)是一种复杂的网络攻击，可以在很长一段时间内保持不被检测到，这使得缓解这些攻击特别具有挑战性。鉴于它们的持久性，需要付出巨大努力才能发现它们并有效应对。现有的基于来源的攻击检测方法往往缺乏可解释性，并且存在较高的误警率，而调查方法要么受到监督，要么仅限于已知的攻击。为了应对这些挑战，我们引入了Shield，这是一种将统计异常检测和基于图的分析与大型语言模型(LLM)的上下文分析能力相结合的新方法。Shield利用LLMS的隐含知识来发现来源数据中隐藏的攻击模式，同时减少误报并提供清晰、可解释的攻击描述。这减少了分析师的警觉疲劳，使他们更容易了解威胁情况。我们广泛的评估证明了Shield在现实世界场景中的有效性和计算效率。Shield被证明比最先进的方法性能更好，实现了更高的精确度和召回率。Shield集成了异常检测、LLM驱动的上下文分析和先进的基于图形的关联，为APT检测建立了一个新的基准。



## **23. BadRobot: Jailbreaking Embodied LLMs in the Physical World**

BadRobot：物理世界中越狱的法学硕士 cs.CY

Accepted to ICLR 2025. Project page:  https://Embodied-LLMs-Safety.github.io

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2407.20242v4) [paper-pdf](http://arxiv.org/pdf/2407.20242v4)

**Authors**: Hangtao Zhang, Chenyu Zhu, Xianlong Wang, Ziqi Zhou, Changgan Yin, Minghui Li, Lulu Xue, Yichen Wang, Shengshan Hu, Aishan Liu, Peijin Guo, Leo Yu Zhang

**Abstract**: Embodied AI represents systems where AI is integrated into physical entities. Large Language Model (LLM), which exhibits powerful language understanding abilities, has been extensively employed in embodied AI by facilitating sophisticated task planning. However, a critical safety issue remains overlooked: could these embodied LLMs perpetrate harmful behaviors? In response, we introduce BadRobot, a novel attack paradigm aiming to make embodied LLMs violate safety and ethical constraints through typical voice-based user-system interactions. Specifically, three vulnerabilities are exploited to achieve this type of attack: (i) manipulation of LLMs within robotic systems, (ii) misalignment between linguistic outputs and physical actions, and (iii) unintentional hazardous behaviors caused by world knowledge's flaws. Furthermore, we construct a benchmark of various malicious physical action queries to evaluate BadRobot's attack performance. Based on this benchmark, extensive experiments against existing prominent embodied LLM frameworks (e.g., Voxposer, Code as Policies, and ProgPrompt) demonstrate the effectiveness of our BadRobot.

摘要: 体现的人工智能代表了人工智能集成到物理实体中的系统。大语言模型(LLM)具有强大的语言理解能力，通过促进复杂的任务规划，已被广泛应用于嵌入式人工智能中。然而，一个关键的安全问题仍然被忽视：这些具体化的LLM是否会实施有害行为？作为回应，我们引入了BadRobot，这是一种新的攻击范例，旨在通过典型的基于语音的用户-系统交互来使具体化LLM违反安全和伦理约束。具体地说，利用三个漏洞来实现这种类型的攻击：(I)在机器人系统内操纵LLM，(Ii)语言输出和物理动作之间的不匹配，以及(Iii)由世界知识的缺陷造成的无意危险行为。此外，我们还构建了一个针对各种恶意物理动作查询的基准来评估BadRobot的攻击性能。在此基准测试的基础上，对现有的主流嵌入式LLM框架(如Voxposer、代码即策略和ProgPrompt)进行了广泛的实验，证明了BadRobot的有效性。



## **24. Adversarial Reasoning at Jailbreaking Time**

越狱时的对抗推理 cs.LG

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01633v1) [paper-pdf](http://arxiv.org/pdf/2502.01633v1)

**Authors**: Mahdi Sabbaghi, Paul Kassianik, George Pappas, Yaron Singer, Amin Karbasi, Hamed Hassani

**Abstract**: As large language models (LLMs) are becoming more capable and widespread, the study of their failure cases is becoming increasingly important. Recent advances in standardizing, measuring, and scaling test-time compute suggest new methodologies for optimizing models to achieve high performance on hard tasks. In this paper, we apply these advances to the task of model jailbreaking: eliciting harmful responses from aligned LLMs. We develop an adversarial reasoning approach to automatic jailbreaking via test-time computation that achieves SOTA attack success rates (ASR) against many aligned LLMs, even the ones that aim to trade inference-time compute for adversarial robustness. Our approach introduces a new paradigm in understanding LLM vulnerabilities, laying the foundation for the development of more robust and trustworthy AI systems.

摘要: 随着大型语言模型（LLM）变得越来越强大和广泛，对其失败案例的研究变得越来越重要。标准化、测量和扩展测试时计算方面的最新进展为优化模型以在硬任务中实现高性能提出了新的方法。在本文中，我们将这些进展应用于模型越狱的任务：从对齐的LLM中引发有害反应。我们开发了一种通过测试时计算自动越狱的对抗推理方法，该方法针对许多对齐的LLM，即使是那些旨在以推理时计算为对抗鲁棒性的LLM，也可以实现SOTA攻击成功率（ASB）。我们的方法引入了理解LLM漏洞的新范式，为开发更强大、更值得信赖的人工智能系统奠定了基础。



## **25. Robust-LLaVA: On the Effectiveness of Large-Scale Robust Image Encoders for Multi-modal Large Language Models**

Robust-LLaVA：关于大规模鲁棒图像编码器对多模式大型语言模型的有效性 cs.CV

Under Review

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01576v1) [paper-pdf](http://arxiv.org/pdf/2502.01576v1)

**Authors**: Hashmat Shadab Malik, Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar, Fahad Khan, Salman Khan

**Abstract**: Multi-modal Large Language Models (MLLMs) excel in vision-language tasks but remain vulnerable to visual adversarial perturbations that can induce hallucinations, manipulate responses, or bypass safety mechanisms. Existing methods seek to mitigate these risks by applying constrained adversarial fine-tuning to CLIP vision encoders on ImageNet-scale data, ensuring their generalization ability is preserved. However, this limited adversarial training restricts robustness and broader generalization. In this work, we explore an alternative approach of leveraging existing vision classification models that have been adversarially pre-trained on large-scale data. Our analysis reveals two principal contributions: (1) the extensive scale and diversity of adversarial pre-training enables these models to demonstrate superior robustness against diverse adversarial threats, ranging from imperceptible perturbations to advanced jailbreaking attempts, without requiring additional adversarial training, and (2) end-to-end MLLM integration with these robust models facilitates enhanced adaptation of language components to robust visual features, outperforming existing plug-and-play methodologies on complex reasoning tasks. Through systematic evaluation across visual question-answering, image captioning, and jail-break attacks, we demonstrate that MLLMs trained with these robust models achieve superior adversarial robustness while maintaining favorable clean performance. Our framework achieves 2x and 1.5x average robustness gains in captioning and VQA tasks, respectively, and delivers over 10% improvement against jailbreak attacks. Code and pretrained models will be available at https://github.com/HashmatShadab/Robust-LLaVA.

摘要: 多模式大语言模型(MLLMS)在视觉-语言任务中表现出色，但仍然容易受到视觉对抗性扰动的影响，这些扰动可能会导致幻觉、操纵反应或绕过安全机制。现有的方法试图通过对ImageNet尺度数据上的裁剪视觉编码器应用受限的对抗性微调来缓解这些风险，以确保它们的泛化能力得到保护。然而，这种有限的对抗性训练限制了健壮性和更广泛的泛化。在这项工作中，我们探索了一种替代方法，利用现有的视觉分类模型，这些模型已经在大规模数据上进行了相反的预训练。我们的分析揭示了两个主要贡献：(1)对抗性预训练的广泛规模和多样性使这些模型能够在不需要额外的对抗性训练的情况下，对从不可察觉的扰动到高级越狱尝试等不同的对抗性威胁表现出优越的健壮性；(2)端到端MLLM与这些健壮的模型的集成促进了语言成分对健壮视觉特征的增强适应，在复杂推理任务中的表现优于现有的即插即用方法。通过对视觉问答、图像字幕和越狱攻击的系统评估，我们证明了使用这些健壮模型训练的MLLMS在保持良好的干净性能的同时，获得了优越的对手健壮性。我们的框架在字幕和VQA任务中分别获得了2倍和1.5倍的平均健壮性提升，并在抵御越狱攻击方面提供了超过10%的改进。代码和预先培训的模型将在https://github.com/HashmatShadab/Robust-LLaVA.上提供



## **26. Scaling Up Membership Inference: When and How Attacks Succeed on Large Language Models**

扩大成员资格推理：攻击何时以及如何在大型语言模型上取得成功 cs.CL

Findings of NAACL 2025. Our code is available at  https://github.com/parameterlab/mia-scaling

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2411.00154v2) [paper-pdf](http://arxiv.org/pdf/2411.00154v2)

**Authors**: Haritz Puerto, Martin Gubri, Sangdoo Yun, Seong Joon Oh

**Abstract**: Membership inference attacks (MIA) attempt to verify the membership of a given data sample in the training set for a model. MIA has become relevant in recent years, following the rapid development of large language models (LLM). Many are concerned about the usage of copyrighted materials for training them and call for methods for detecting such usage. However, recent research has largely concluded that current MIA methods do not work on LLMs. Even when they seem to work, it is usually because of the ill-designed experimental setup where other shortcut features enable "cheating." In this work, we argue that MIA still works on LLMs, but only when multiple documents are presented for testing. We construct new benchmarks that measure the MIA performances at a continuous scale of data samples, from sentences (n-grams) to a collection of documents (multiple chunks of tokens). To validate the efficacy of current MIA approaches at greater scales, we adapt a recent work on Dataset Inference (DI) for the task of binary membership detection that aggregates paragraph-level MIA features to enable MIA at document and collection of documents level. This baseline achieves the first successful MIA on pre-trained and fine-tuned LLMs.

摘要: 成员关系推理攻击(MIA)试图验证给定数据样本在模型训练集中的成员资格。近年来，随着大型语言模型(LLM)的快速发展，MIA变得相关起来。许多人担心使用受版权保护的材料来培训他们，并呼吁采取方法来检测这种使用情况。然而，最近的研究在很大程度上得出结论，目前的MIA方法不适用于LLMS。即使它们看起来很有效，这通常也是因为设计糟糕的实验设置，其他快捷功能允许“作弊”。在这项工作中，我们认为MIA仍然适用于LLMS，但只有在提交多个文档进行测试时才能使用。我们构建了新的基准来衡量MIA在连续规模的数据样本上的性能，从句子(n-gram)到文档集合(多个令牌块)。为了在更大范围内验证当前MIA方法的有效性，我们对最近在数据集推理(DI)方面的工作进行了调整，以用于二元成员关系检测任务，该任务聚集了段级MIA特征，以支持文档和文档集合级别的MIA。这一基线在预先训练和微调的LLM上实现了第一次成功的MIA。



## **27. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

Top-FlipRAG：针对检索增强生成模型的面向主题的对抗性观点操纵攻击 cs.CL

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01386v1) [paper-pdf](http://arxiv.org/pdf/2502.01386v1)

**Authors**: Yuyang Gong, Zhuo Chen, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.

摘要: 基于大型语言模型(LLMS)的检索-增强生成(RAG)系统已经成为诸如问题回答和内容生成等任务的关键。然而，由于其固有的脆弱性，它们对舆论和信息传播的影响越来越大，使其成为安全研究的关键焦点。以前的研究主要针对针对事实或单一查询操作的攻击。在本文中，我们讨论了一个更实际的场景：针对RAG模型的面向主题的对抗性意见操纵攻击，其中要求LLM推理和综合多个视角，使它们特别容易受到系统性知识中毒的影响。具体地说，我们提出了Theme-FlipRAG，这是一种两阶段操纵攻击管道，战略性地制造对抗性扰动来影响相关查询的观点。该方法结合了传统的对抗性排序攻击技术，并利用LLMS丰富的内部相关知识和推理能力来执行语义级的扰动。实验表明，提出的攻击有效地改变了模型输出对特定主题的看法，显著影响了用户对信息的感知。目前的缓解方法无法有效防御此类攻击，这突显了加强RAG系统安全保障的必要性，并为LLM安全研究提供了至关重要的见解。



## **28. Detecting Backdoor Samples in Contrastive Language Image Pretraining**

对比语言图像预训练中后门样本检测 cs.LG

ICLR2025

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01385v1) [paper-pdf](http://arxiv.org/pdf/2502.01385v1)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: Contrastive language-image pretraining (CLIP) has been found to be vulnerable to poisoning backdoor attacks where the adversary can achieve an almost perfect attack success rate on CLIP models by poisoning only 0.01\% of the training dataset. This raises security concerns on the current practice of pretraining large-scale models on unscrutinized web data using CLIP. In this work, we analyze the representations of backdoor-poisoned samples learned by CLIP models and find that they exhibit unique characteristics in their local subspace, i.e., their local neighborhoods are far more sparse than that of clean samples. Based on this finding, we conduct a systematic study on detecting CLIP backdoor attacks and show that these attacks can be easily and efficiently detected by traditional density ratio-based local outlier detectors, whereas existing backdoor sample detection methods fail. Our experiments also reveal that an unintentional backdoor already exists in the original CC3M dataset and has been trained into a popular open-source model released by OpenCLIP. Based on our detector, one can clean up a million-scale web dataset (e.g., CC3M) efficiently within 15 minutes using 4 Nvidia A100 GPUs. The code is publicly available in our \href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub repository}.

摘要: 对比语言图像预训练(CLIP)被发现容易受到中毒后门攻击，对手只需中毒0.01%的训练数据集就可以在CLIP模型上获得近乎完美的攻击成功率。这引发了人们对当前使用CLIP对大规模模型进行未经审查的网络数据的预培训的安全担忧。在这项工作中，我们分析了通过CLIP模型学习的后门中毒样本的表示，发现它们在局部子空间中表现出独特的特征，即它们的局部邻域比干净样本的局部邻域稀疏得多。基于这一发现，我们对CLIP后门攻击的检测进行了系统的研究，结果表明，传统的基于密度比的局部离群点检测方法可以轻松有效地检测到这些攻击，而现有的后门样本检测方法却无法检测到这些攻击。我们的实验还表明，在原始的CC3M数据集中已经存在一个无意的后门，并且已经被训练成OpenCLIP发布的流行的开源模型。基于我们的检测器，您可以使用4个NVIDIA A100图形处理器在15分钟内高效清理百万级Web数据集(例如CC3M)。代码在我们的\href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub存储库中公开提供。



## **29. Improving the Robustness of Representation Misdirection for Large Language Model Unlearning**

提高大型语言模型去学习的表示误导的鲁棒性 cs.CL

12 pages, 4 figures, 1 table

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2501.19202v2) [paper-pdf](http://arxiv.org/pdf/2501.19202v2)

**Authors**: Dang Huu-Tien, Hoang Thanh-Tung, Le-Minh Nguyen, Naoya Inoue

**Abstract**: Representation Misdirection (RM) and variants are established large language model (LLM) unlearning methods with state-of-the-art performance. In this paper, we show that RM methods inherently reduce models' robustness, causing them to misbehave even when a single non-adversarial forget-token is in the retain-query. Toward understanding underlying causes, we reframe the unlearning process as backdoor attacks and defenses: forget-tokens act as backdoor triggers that, when activated in retain-queries, cause disruptions in RM models' behaviors, similar to successful backdoor attacks. To mitigate this vulnerability, we propose Random Noise Augmentation -- a model and method agnostic approach with theoretical guarantees for improving the robustness of RM methods. Extensive experiments demonstrate that RNA significantly improves the robustness of RM models while enhancing the unlearning performances.

摘要: 表示误导（RM）和变体是建立的大型语言模型（LLM）去学习方法，具有最先进的性能。在本文中，我们表明RM方法本质上降低了模型的鲁棒性，导致它们即使在保留查询中存在单个非对抗性遗忘令牌时也会表现不当。为了了解根本原因，我们将取消学习过程重新定义为后门攻击和防御：忘记令牌充当后门触发器，当在保留查询中激活时，会导致RM模型行为中断，类似于成功的后门攻击。为了减轻这一漏洞，我们提出了随机噪音增强--一种模型和方法不可知的方法，具有提高RM方法鲁棒性的理论保证。大量实验表明，RNA显着提高了RM模型的鲁棒性，同时增强了去学习性能。



## **30. PBI-Attack: Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for Toxicity Maximization**

PBI攻击：优先引导双峰交互黑匣子越狱攻击，以实现毒性最大化 cs.CR

Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for  Toxicity Maximization

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2412.05892v3) [paper-pdf](http://arxiv.org/pdf/2412.05892v3)

**Authors**: Ruoxi Cheng, Yizhong Ding, Shuirong Cao, Ranjie Duan, Xiaoshuang Jia, Shaowei Yuan, Zhiqiang Wang, Xiaojun Jia

**Abstract**: Understanding the vulnerabilities of Large Vision Language Models (LVLMs) to jailbreak attacks is essential for their responsible real-world deployment. Most previous work requires access to model gradients, or is based on human knowledge (prompt engineering) to complete jailbreak, and they hardly consider the interaction of images and text, resulting in inability to jailbreak in black box scenarios or poor performance. To overcome these limitations, we propose a Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for toxicity maximization, referred to as PBI-Attack. Our method begins by extracting malicious features from a harmful corpus using an alternative LVLM and embedding these features into a benign image as prior information. Subsequently, we enhance these features through bidirectional cross-modal interaction optimization, which iteratively optimizes the bimodal perturbations in an alternating manner through greedy search, aiming to maximize the toxicity of the generated response. The toxicity level is quantified using a well-trained evaluation model. Experiments demonstrate that PBI-Attack outperforms previous state-of-the-art jailbreak methods, achieving an average attack success rate of 92.5% across three open-source LVLMs and around 67.3% on three closed-source LVLMs. Disclaimer: This paper contains potentially disturbing and offensive content.

摘要: 了解大型视觉语言模型(LVLM)对越狱攻击的脆弱性对于它们在现实世界中负责任的部署至关重要。以前的工作大多需要获取模型梯度，或者基于人类知识(提示工程)来完成越狱，并且很少考虑图像和文本的交互，导致在黑匣子场景下无法越狱或性能不佳。为了克服这些局限性，我们提出了一种先验引导的双模交互黑盒越狱攻击，称为PBI攻击。我们的方法首先使用替代的LVLM从有害语料库中提取恶意特征，并将这些特征作为先验信息嵌入到良性图像中。随后，我们通过双向跨模式交互优化来增强这些特征，该优化通过贪婪搜索以交替的方式迭代优化双峰扰动，以最大化所生成响应的毒性。使用训练有素的评估模型来量化毒性水平。实验表明，PBI-Attack的性能优于以往最先进的越狱方法，在三个开源LVLM上的平均攻击成功率为92.5%，在三个闭源LVLM上的平均攻击成功率约为67.3%。免责声明：本文包含可能令人不安和冒犯性的内容。



## **31. Eliciting Language Model Behaviors with Investigator Agents**

使用研究者代理激发语言模型行为 cs.LG

20 pages, 7 figures

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01236v1) [paper-pdf](http://arxiv.org/pdf/2502.01236v1)

**Authors**: Xiang Lisa Li, Neil Chowdhury, Daniel D. Johnson, Tatsunori Hashimoto, Percy Liang, Sarah Schwettmann, Jacob Steinhardt

**Abstract**: Language models exhibit complex, diverse behaviors when prompted with free-form text, making it difficult to characterize the space of possible outputs. We study the problem of behavior elicitation, where the goal is to search for prompts that induce specific target behaviors (e.g., hallucinations or harmful responses) from a target language model. To navigate the exponentially large space of possible prompts, we train investigator models to map randomly-chosen target behaviors to a diverse distribution of outputs that elicit them, similar to amortized Bayesian inference. We do this through supervised fine-tuning, reinforcement learning via DPO, and a novel Frank-Wolfe training objective to iteratively discover diverse prompting strategies. Our investigator models surface a variety of effective and human-interpretable prompts leading to jailbreaks, hallucinations, and open-ended aberrant behaviors, obtaining a 100% attack success rate on a subset of AdvBench (Harmful Behaviors) and an 85% hallucination rate.

摘要: 当提示自由格式文本时，语言模型表现出复杂多样的行为，这使得很难描述可能输出的空间。我们研究行为诱导问题，目标是从目标语言模型中寻找诱导特定目标行为(例如，幻觉或有害反应)的提示。为了在可能提示的指数级大空间中导航，我们训练调查员模型将随机选择的目标行为映射到引发它们的不同输出分布，类似于摊销贝叶斯推理。我们通过有监督的微调、通过DPO的强化学习和一个新颖的Frank-Wolfe训练目标来迭代地发现不同的激励策略来做到这一点。我们的调查员模型提供了各种有效的、人类可解释的提示，导致越狱、幻觉和无限制的异常行为，对AdvBtch(有害行为)的子集获得了100%的攻击成功率和85%的幻想率。



## **32. The dark deep side of DeepSeek: Fine-tuning attacks against the safety alignment of CoT-enabled models**

DeepSeek的阴暗面：针对支持CoT的模型的安全一致性的微调攻击 cs.CR

12 Pages

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01225v1) [paper-pdf](http://arxiv.org/pdf/2502.01225v1)

**Authors**: Zhiyuan Xu, Joseph Gardiner, Sana Belguith

**Abstract**: Large language models are typically trained on vast amounts of data during the pre-training phase, which may include some potentially harmful information. Fine-tuning attacks can exploit this by prompting the model to reveal such behaviours, leading to the generation of harmful content. In this paper, we focus on investigating the performance of the Chain of Thought based reasoning model, DeepSeek, when subjected to fine-tuning attacks. Specifically, we explore how fine-tuning manipulates the model's output, exacerbating the harmfulness of its responses while examining the interaction between the Chain of Thought reasoning and adversarial inputs. Through this study, we aim to shed light on the vulnerability of Chain of Thought enabled models to fine-tuning attacks and the implications for their safety and ethical deployment.

摘要: 大型语言模型通常在预训练阶段根据大量数据进行训练，其中可能包括一些潜在有害的信息。微调攻击可以通过促使模型揭示此类行为来利用这一点，从而导致有害内容的生成。在本文中，我们重点研究基于思想链的推理模型DeepSeek在受到微调攻击时的性能。具体来说，我们探索微调如何操纵模型的输出，加剧其反应的危害性，同时检查思维链推理和对抗输入之间的相互作用。通过这项研究，我们的目标是揭示思想链使模型能够微调攻击的脆弱性及其对安全性和道德部署的影响。



## **33. Jailbreaking with Universal Multi-Prompts**

用通用多胞胎越狱 cs.CL

Accepted by NAACL Findings 2025

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01154v1) [paper-pdf](http://arxiv.org/pdf/2502.01154v1)

**Authors**: Yu-Ling Hsu, Hsuan Su, Shang-Tse Chen

**Abstract**: Large language models (LLMs) have seen rapid development in recent years, revolutionizing various applications and significantly enhancing convenience and productivity. However, alongside their impressive capabilities, ethical concerns and new types of attacks, such as jailbreaking, have emerged. While most prompting techniques focus on optimizing adversarial inputs for individual cases, resulting in higher computational costs when dealing with large datasets. Less research has addressed the more general setting of training a universal attacker that can transfer to unseen tasks. In this paper, we introduce JUMP, a prompt-based method designed to jailbreak LLMs using universal multi-prompts. We also adapt our approach for defense, which we term DUMP. Experimental results demonstrate that our method for optimizing universal multi-prompts outperforms existing techniques.

摘要: 近年来，大型语言模型（LLM）发展迅速，彻底改变了各种应用程序，显着提高了便利性和生产力。然而，除了它们令人印象深刻的能力之外，道德问题和越狱等新型攻击也出现了。虽然大多数提示技术专注于优化个别案例的对抗输入，从而导致处理大型数据集时计算成本更高。较少的研究涉及训练可以转移到不可见任务的通用攻击者的更一般设置。本文中，我们介绍了JUMP，这是一种基于预算的方法，旨在使用通用多提示越狱LLM。我们还调整我们的防御方法，我们称之为“DUMP”。实验结果表明，我们用于优化通用多提示的方法优于现有技术。



## **34. When LLMs Go Online: The Emerging Threat of Web-Enabled LLMs**

当LLM上线时：支持Web的LLM的新兴威胁 cs.CR

20 pages, To appear in Usenix Security 2025

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2410.14569v3) [paper-pdf](http://arxiv.org/pdf/2410.14569v3)

**Authors**: Hanna Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin, Kimin Lee

**Abstract**: Recent advancements in Large Language Models (LLMs) have established them as agentic systems capable of planning and interacting with various tools. These LLM agents are often paired with web-based tools, enabling access to diverse sources and real-time information. Although these advancements offer significant benefits across various applications, they also increase the risk of malicious use, particularly in cyberattacks involving personal information. In this work, we investigate the risks associated with misuse of LLM agents in cyberattacks involving personal data. Specifically, we aim to understand: 1) how potent LLM agents can be when directed to conduct cyberattacks, 2) how cyberattacks are enhanced by web-based tools, and 3) how affordable and easy it becomes to launch cyberattacks using LLM agents. We examine three attack scenarios: the collection of Personally Identifiable Information (PII), the generation of impersonation posts, and the creation of spear-phishing emails. Our experiments reveal the effectiveness of LLM agents in these attacks: LLM agents achieved a precision of up to 95.9% in collecting PII, generated impersonation posts where 93.9% of them were deemed authentic, and boosted click rate of phishing links in spear phishing emails by 46.67%. Additionally, our findings underscore the limitations of existing safeguards in contemporary commercial LLMs, emphasizing the urgent need for robust security measures to prevent the misuse of LLM agents.

摘要: 大型语言模型(LLM)的最新进展已将它们确立为能够规划各种工具并与其交互的代理系统。这些LLM代理通常与基于Web的工具配合使用，从而能够访问不同的来源和实时信息。虽然这些改进在各种应用程序中提供了显著的好处，但它们也增加了恶意使用的风险，特别是在涉及个人信息的网络攻击中。在这项工作中，我们调查了在涉及个人数据的网络攻击中滥用LLM代理的相关风险。具体地说，我们的目标是了解：1)LLM代理在被指示进行网络攻击时的威力有多大；2)基于Web的工具如何增强网络攻击；3)使用LLM代理发起网络攻击变得多么负担得起和容易。我们研究了三种攻击场景：收集个人身份信息(PII)、生成模拟帖子和创建鱼叉式网络钓鱼电子邮件。我们的实验显示了LLM代理在这些攻击中的有效性：LLM代理在收集PII信息时达到了95.9%的准确率，生成了93.9%被认为是真实的模仿帖子，并将鱼叉式钓鱼邮件中钓鱼链接的点击率提高了46.67%。此外，我们的研究结果强调了当代商业LLM现有保障措施的局限性，强调迫切需要强有力的安全措施，以防止滥用LLM剂。



## **35. Tool Unlearning for Tool-Augmented LLMs**

工具增强LLM的工具取消学习 cs.LG

https://clu-uml.github.io/MU-Bench-Project-Page/

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01083v1) [paper-pdf](http://arxiv.org/pdf/2502.01083v1)

**Authors**: Jiali Cheng, Hadi Amiri

**Abstract**: Tool-augmented large language models (LLMs) are often trained on datasets of query-response pairs, which embed the ability to use tools or APIs directly into the parametric knowledge of LLMs. Tool-augmented LLMs need the ability to forget learned tools due to security vulnerabilities, privacy regulations, or tool deprecations. However, ``tool unlearning'' has not been investigated in unlearning literature. We introduce this novel task, which requires addressing distinct challenges compared to traditional unlearning: knowledge removal rather than forgetting individual samples, the high cost of optimizing LLMs, and the need for principled evaluation metrics. To bridge these gaps, we propose ToolDelete, the first approach for unlearning tools from tool-augmented LLMs. It implements three key properties to address the above challenges for effective tool unlearning and introduces a new membership inference attack (MIA) model for effective evaluation. Extensive experiments on multiple tool learning datasets and tool-augmented LLMs show that ToolDelete effectively unlearns randomly selected tools, while preserving the LLM's knowledge on non-deleted tools and maintaining performance on general tasks.

摘要: 工具增强的大型语言模型(LLM)通常是在查询-响应对的数据集上进行训练的，这将使用工具或API的能力直接嵌入到LLM的参数知识中。工具增强的LLM需要能够忘记由于安全漏洞、隐私法规或工具弃用而学到的工具。然而，“工具遗忘”在遗忘文献中还没有被研究过。我们引入了这项新的任务，它需要解决与传统遗忘相比的不同挑战：知识移除而不是忘记单个样本，优化LLM的高成本，以及对原则性评估指标的需求。为了弥合这些差距，我们提出了ToolDelete，这是第一种从工具增强的LLM中忘记工具的方法。它实现了三个关键性质来解决上述有效工具遗忘的挑战，并引入了一个新的成员推理攻击(MIA)模型来进行有效评估。在多个工具学习数据集和工具扩充的LLM上的大量实验表明，ToolDelete有效地取消了随机选择的工具的学习，同时保留了LLM关于未删除工具的知识，并保持了一般任务的性能。



## **36. SQL Injection Jailbreak: A Structural Disaster of Large Language Models**

SQL注入越狱：大型语言模型的结构灾难 cs.CR

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2411.01565v4) [paper-pdf](http://arxiv.org/pdf/2411.01565v4)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: In recent years, the rapid development of large language models (LLMs) has brought new vitality into various domains, generating substantial social and economic benefits. However, this swift advancement has also introduced new vulnerabilities. Jailbreaking, a form of attack that induces LLMs to produce harmful content through carefully crafted prompts, presents a significant challenge to the safe and trustworthy development of LLMs. Previous jailbreak methods primarily exploited the internal properties or capabilities of LLMs, such as optimization-based jailbreak methods and methods that leveraged the model's context-learning abilities. In this paper, we introduce a novel jailbreak method, SQL Injection Jailbreak (SIJ), which targets the external properties of LLMs, specifically, the way LLMs construct input prompts. By injecting jailbreak information into user prompts, SIJ successfully induces the model to output harmful content. Our SIJ method achieves near 100\% attack success rates on five well-known open-source LLMs on the AdvBench and HEx-PHI, while incurring lower time costs compared to previous methods. For closed-source models, SIJ achieves near 100% attack success rate on GPT-3.5-turbo. Additionally, SIJ exposes a new vulnerability in LLMs that urgently requires mitigation. To address this, we propose a simple defense method called Self-Reminder-Key to counter SIJ and demonstrate its effectiveness through experimental results. Our code is available at https://github.com/weiyezhimeng/SQL-Injection-Jailbreak.

摘要: 近年来，大型语言模型的快速发展为各个领域带来了新的活力，产生了可观的社会效益和经济效益。然而，这种快速发展也带来了新的脆弱性。越狱是一种攻击形式，通过精心制作的提示诱使LLM产生有害内容，对LLM的安全和可信开发构成了重大挑战。以前的越狱方法主要利用LLMS的内部属性或功能，例如基于优化的越狱方法和利用模型的上下文学习能力的方法。本文介绍了一种新的越狱方法--SQL注入越狱(SIJ)，它针对LLMS的外部属性，特别是LLMS构造输入提示的方式。通过在用户提示中注入越狱信息，SIJ成功地诱导该模型输出有害内容。与以前的方法相比，我们的SIJ方法在AdvBch和hex-PHI上的五个著名的开源LLM上获得了近100%的攻击成功率，同时产生了更低的时间成本。对于封闭源代码模型，SIJ在GPT-3.5-Turbo上实现了近100%的攻击成功率。此外，SIJ还暴露了LLMS中一个迫切需要缓解的新漏洞。针对这一问题，我们提出了一种简单的防御方法，称为自我提醒密钥来对抗SIJ，并通过实验结果证明了其有效性。我们的代码可以在https://github.com/weiyezhimeng/SQL-Injection-Jailbreak.上找到



## **37. Refining Adaptive Zeroth-Order Optimization at Ease**

轻松细化自适应零阶优化 cs.LG

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01014v1) [paper-pdf](http://arxiv.org/pdf/2502.01014v1)

**Authors**: Yao Shu, Qixin Zhang, Kun He, Zhongxiang Dai

**Abstract**: Recently, zeroth-order (ZO) optimization plays an essential role in scenarios where gradient information is inaccessible or unaffordable, such as black-box systems and resource-constrained environments. While existing adaptive methods such as ZO-AdaMM have shown promise, they are fundamentally limited by their underutilization of moment information during optimization, usually resulting in underperforming convergence. To overcome these limitations, this paper introduces Refined Adaptive Zeroth-Order Optimization (R-AdaZO). Specifically, we first show the untapped variance reduction effect of first moment estimate on ZO gradient estimation, which improves the accuracy and stability of ZO updates. We then refine the second moment estimate based on these variance-reduced gradient estimates to better capture the geometry of the optimization landscape, enabling a more effective scaling of ZO updates. We present rigorous theoretical analysis to show (I) the first analysis to the variance reduction of first moment estimate in ZO optimization, (II) the improved second moment estimates with a more accurate approximation of its variance-free ideal, (III) the first variance-aware convergence framework for adaptive ZO methods, which may be of independent interest, and (IV) the faster convergence of R-AdaZO than existing baselines like ZO-AdaMM. Our extensive experiments, including synthetic problems, black-box adversarial attack, and memory-efficient fine-tuning of large language models (LLMs), further verify the superior convergence of R-AdaZO, indicating that R-AdaZO offers an improved solution for real-world ZO optimization challenges.

摘要: 最近，零阶(ZO)优化在诸如黑盒系统和资源受限环境等无法获取或负担不起梯度信息的场景中扮演着重要的角色。虽然现有的自适应方法如ZO-AdaMM已经显示出很好的前景，但它们在优化过程中对矩信息的利用不足从根本上限制了它们，通常导致收敛性能不佳。为了克服这些局限性，本文引入了改进的自适应零阶优化算法(R-AdaZO)。具体地说，我们首先展示了一阶矩估计对ZO梯度估计的未开发的减方差效果，从而提高了ZO更新的精度和稳定性。然后，我们基于这些经方差减少的梯度估计来改进二阶矩估计，以更好地捕捉优化场景的几何形状，从而实现更有效的ZO更新缩放。我们给出了严格的理论分析，以证明(I)第一次分析ZO优化中一阶矩估计的方差降低，(Ii)改进的二阶矩估计更精确地逼近其无方差理想，(Iii)自适应ZO方法的第一个方差感知收敛框架，它可能是独立的，以及(Iv)R-AdaZO比现有基线(如ZO-AdaMM)更快的收敛。我们的大量实验，包括合成问题、黑盒对抗攻击和对大型语言模型(LLM)的内存效率优化，进一步验证了R-AdaZO的优越收敛能力，表明R-AdaZO为现实世界的ZO优化挑战提供了一种改进的解决方案。



## **38. Encrypted Large Model Inference: The Equivariant Encryption Paradigm**

加密大模型推理：等变加密范式 cs.CR

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01013v1) [paper-pdf](http://arxiv.org/pdf/2502.01013v1)

**Authors**: James Buban, Hongyang Zhang, Claudio Angione, Harry Yang, Ahmad Farhan, Seyfal Sultanov, Michael Du, Xuran Ma, Zihao Wang, Yue Zhao, Arria Owlia, Fielding Johnston, Patrick Colangelo

**Abstract**: Large scale deep learning model, such as modern language models and diffusion architectures, have revolutionized applications ranging from natural language processing to computer vision. However, their deployment in distributed or decentralized environments raises significant privacy concerns, as sensitive data may be exposed during inference. Traditional techniques like secure multi-party computation, homomorphic encryption, and differential privacy offer partial remedies but often incur substantial computational overhead, latency penalties, or limited compatibility with non-linear network operations. In this work, we introduce Equivariant Encryption (EE), a novel paradigm designed to enable secure, "blind" inference on encrypted data with near zero performance overhead. Unlike fully homomorphic approaches that encrypt the entire computational graph, EE selectively obfuscates critical internal representations within neural network layers while preserving the exact functionality of both linear and a prescribed set of non-linear operations. This targeted encryption ensures that raw inputs, intermediate activations, and outputs remain confidential, even when processed on untrusted infrastructure. We detail the theoretical foundations of EE, compare its performance and integration complexity against conventional privacy preserving techniques, and demonstrate its applicability across a range of architectures, from convolutional networks to large language models. Furthermore, our work provides a comprehensive threat analysis, outlining potential attack vectors and baseline strategies, and benchmarks EE against standard inference pipelines in decentralized settings. The results confirm that EE maintains high fidelity and throughput, effectively bridging the gap between robust data confidentiality and the stringent efficiency requirements of modern, large scale model inference.

摘要: 大规模深度学习模型，如现代语言模型和扩散体系结构，已经使从自然语言处理到计算机视觉的应用发生了革命性的变化。然而，它们在分布式或分散式环境中的部署会引起严重的隐私问题，因为敏感数据可能会在推理过程中暴露出来。安全多方计算、同态加密和差异隐私等传统技术提供了部分补救措施，但通常会招致大量计算开销、延迟惩罚或与非线性网络操作的有限兼容性。在这项工作中，我们引入了等变加密(EE)，这是一种新的范例，旨在以几乎为零的性能开销实现对加密数据的安全“盲”推理。与加密整个计算图形的完全同态方法不同，EE选择性地混淆神经网络层内的关键内部表示，同时保留线性和指定的一组非线性操作的确切功能。这种有针对性的加密确保原始输入、中间激活和输出保密，即使在不受信任的基础设施上处理时也是如此。我们详细介绍了EE的理论基础，将其性能和集成复杂性与传统的隐私保护技术进行了比较，并展示了它在从卷积网络到大型语言模型的一系列体系结构中的适用性。此外，我们的工作提供了全面的威胁分析，概述了潜在的攻击向量和基线策略，并针对分散环境下的标准推理管道对EE进行了基准测试。结果证实，EE保持了高保真度和高吞吐量，有效地弥合了稳健的数据机密性和现代大规模模型推理的严格效率要求之间的差距。



## **39. Time-Reversal Provides Unsupervised Feedback to LLMs**

计时器向LLM提供无监督反馈 cs.CL

Accepted as a spotlight in NeurIPS 2024

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2412.02626v3) [paper-pdf](http://arxiv.org/pdf/2412.02626v3)

**Authors**: Yerram Varun, Rahul Madhavan, Sravanti Addepalli, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**Abstract**: Large Language Models (LLMs) are typically trained to predict in the forward direction of time. However, recent works have shown that prompting these models to look back and critique their own generations can produce useful feedback. Motivated by this, we explore the question of whether LLMs can be empowered to think (predict and score) backwards to provide unsupervised feedback that complements forward LLMs. Towards this, we introduce Time Reversed Language Models (TRLMs), which can score and generate queries when conditioned on responses, effectively functioning in the reverse direction of time. Further, to effectively infer in the response to query direction, we pre-train and fine-tune a language model (TRLM-Ba) in the reverse token order from scratch. We show empirically (and theoretically in a stylized setting) that time-reversed models can indeed complement forward model predictions when used to score the query given response for re-ranking multiple forward generations. We obtain up to 5\% improvement on the widely used AlpacaEval Leaderboard over the competent baseline of best-of-N re-ranking using self log-perplexity scores. We further show that TRLM scoring outperforms conventional forward scoring of response given query, resulting in significant gains in applications such as citation generation and passage retrieval. We next leverage the generative ability of TRLM to augment or provide unsupervised feedback to input safety filters of LLMs, demonstrating a drastic reduction in false negative rate with negligible impact on false positive rates against several attacks published on the popular JailbreakBench leaderboard.

摘要: 大型语言模型(LLM)通常被训练为在时间的正向进行预测。然而，最近的研究表明，促使这些模型回顾和批评他们自己的几代人可以产生有用的反馈。受此启发，我们探讨了LLM是否可以被赋予向后思考(预测和评分)的能力，以提供无监督的反馈来补充前向LLM。为此，我们引入了时间反转语言模型(TRLMS)，该模型可以根据响应进行评分并生成查询，有效地沿时间的相反方向运行。此外，为了有效地推断对查询方向的响应，我们从头开始以相反的令牌顺序预先训练和微调语言模型(TRLM-BA)。我们在经验上(理论上是在风格化的环境中)表明，当时间倒置模型用于对给定响应的查询进行重新排序时，时间倒置模型确实可以补充正向模型预测。我们在广泛使用的AlpacaEval排行榜上获得了高达5%的改进，超过了使用自我对数困惑分数重新排序的合格基线。我们进一步表明，TRLM评分优于传统的对给定查询的回复的前向评分，从而在引文生成和段落检索等应用中获得了显著的收益。接下来，我们利用TRLM的生成能力来增强或向LLMS的输入安全过滤器提供无监督反馈，展示了假阴性率的大幅降低，而对流行的JailBreak Btch排行榜上发布的几种攻击的错误确认率的影响可以忽略不计。



## **40. Gandalf the Red: Adaptive Security for LLMs**

红色甘道夫：LLM的自适应安全 cs.LG

Niklas Pfister, V\'aclav Volhejn and Manuel Knott contributed equally

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2501.07927v2) [paper-pdf](http://arxiv.org/pdf/2501.07927v2)

**Authors**: Niklas Pfister, Václav Volhejn, Manuel Knott, Santiago Arias, Julia Bazińska, Mykhailo Bichurin, Alan Commike, Janet Darling, Peter Dienes, Matthew Fiedler, David Haber, Matthias Kraft, Marco Lancini, Max Mathys, Damián Pascual-Ortiz, Jakub Podolak, Adrià Romero-López, Kyriacos Shiarlis, Andreas Signer, Zsolt Terek, Athanasios Theocharis, Daniel Timbrell, Samuel Trautwein, Samuel Watts, Yun-Han Wu, Mateo Rojas-Carulla

**Abstract**: Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications.

摘要: 目前对大型语言模型(LLM)应用程序中针对即时攻击的防御措施的评估往往忽略了两个关键因素：敌意行为的动态性质和限制性防御措施对合法用户的可用性惩罚。本文提出了动态安全效用威胁模型D-SEC，它明确地将攻击者和合法用户分开，对多步交互进行建模，并以优化的形式表达安全效用。我们通过引入Gandalf进一步解决了现有评估中的缺陷，Gandalf是一个众包、游戏化的红色团队平台，旨在生成现实的、自适应的攻击。使用Gandalf，我们收集并发布了279K提示攻击的数据集。在良性用户数据的补充下，我们的分析揭示了安全性和实用性之间的相互作用，表明LLM中集成的防御措施(例如系统提示)即使在不阻止请求的情况下也会降低可用性。我们演示了受限应用程序域、深度防御和自适应防御是构建安全且有用的LLM应用程序的有效策略。



## **41. From Compliance to Exploitation: Jailbreak Prompt Attacks on Multimodal LLMs**

从合规到剥削：对多模式LLM的越狱立即攻击 cs.CR

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00735v1) [paper-pdf](http://arxiv.org/pdf/2502.00735v1)

**Authors**: Chun Wai Chiu, Linghan Huang, Bo Li, Huaming Chen

**Abstract**: Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the frontier multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. To better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flank Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios. These findings highlight both the potency of prompt-based obfuscation in voice-enabled contexts and the limitations of current LLMs' moderation safeguards and the urgent need for advanced defense strategies to address the challenges posed by evolving, context-rich attacks.

摘要: 大语言模型因其处理文本、音频、图像和视频等不同类型输入数据的能力日益增强，在各个领域得到了广泛的应用。虽然LLM在理解和生成不同场景的上下文方面表现出了出色的性能，但它们很容易受到基于提示的攻击，这些攻击主要是通过文本输入进行的。在本文中，我们介绍了第一个基于语音的针对多模式LLMS的越狱攻击，称为侧翼攻击，它可以同时处理针对多模式LLMS的不同类型的输入。我们的工作是受到单语言语音驱动的大型语言模型的最新进展的推动，这些模型在传统的基于文本的LLMS漏洞之外引入了新的攻击面。为了调查这些风险，我们研究了前沿多模式LLMS，这些LLMS可以通过不同类型的输入(如音频输入)访问，重点关注对抗性提示如何绕过其防御机制。我们提出了一种新的策略，在不允许的提示的两侧是良性的、叙事驱动的提示。它被整合到侧翼攻击中，试图使交互上下文人性化，并通过虚构的设置执行攻击。为了更好地评估攻击性能，我们提出了一个半自动的策略违规检测自我评估框架。我们证明了侧翼攻击能够操纵最先进的LLM产生未对齐和禁止的输出，在七个禁止场景中获得了从0.67到0.93的平均攻击成功率。这些发现既突显了基于提示的混淆在语音支持的上下文中的有效性，也突显了当前LLMS适度保障的局限性，以及迫切需要先进的防御策略来应对不断演变的、上下文丰富的攻击带来的挑战。



## **42. "I am bad": Interpreting Stealthy, Universal and Robust Audio Jailbreaks in Audio-Language Models**

“我很坏”：在音频语言模型中解释秘密、普遍和稳健的音频越狱 cs.LG

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00718v1) [paper-pdf](http://arxiv.org/pdf/2502.00718v1)

**Authors**: Isha Gupta, David Khachaturov, Robert Mullins

**Abstract**: The rise of multimodal large language models has introduced innovative human-machine interaction paradigms but also significant challenges in machine learning safety. Audio-Language Models (ALMs) are especially relevant due to the intuitive nature of spoken communication, yet little is known about their failure modes. This paper explores audio jailbreaks targeting ALMs, focusing on their ability to bypass alignment mechanisms. We construct adversarial perturbations that generalize across prompts, tasks, and even base audio samples, demonstrating the first universal jailbreaks in the audio modality, and show that these remain effective in simulated real-world conditions. Beyond demonstrating attack feasibility, we analyze how ALMs interpret these audio adversarial examples and reveal them to encode imperceptible first-person toxic speech - suggesting that the most effective perturbations for eliciting toxic outputs specifically embed linguistic features within the audio signal. These results have important implications for understanding the interactions between different modalities in multimodal models, and offer actionable insights for enhancing defenses against adversarial audio attacks.

摘要: 多通道大型语言模型的兴起引入了创新的人机交互范式，但也给机器学习的安全性带来了重大挑战。由于口语交流的直觉性，音频语言模型(ALM)尤其相关，但人们对其失败模式知之甚少。本文探讨了针对施舍的音频越狱，重点是它们绕过对齐机制的能力。我们构建了跨提示、任务甚至基本音频样本的对抗性扰动，演示了音频通道中的第一个通用越狱，并表明这些扰动在模拟的真实世界条件下仍然有效。除了展示攻击的可行性外，我们还分析了ALMS如何解释这些音频对抗性例子，并将它们揭示为编码不可感知的第一人称有毒言语--这表明，引发有毒输出的最有效扰动具体地将语言特征嵌入音频信号中。这些结果对于理解多通道模型中不同通道之间的相互作用具有重要意义，并为增强对敌方音频攻击的防御提供了可操作的见解。



## **43. LLM Safety Alignment is Divergence Estimation in Disguise**

LLM安全调整是伪装的分歧估计 cs.LG

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00657v1) [paper-pdf](http://arxiv.org/pdf/2502.00657v1)

**Authors**: Rajdeep Haldar, Ziyi Wang, Qifan Song, Guang Lin, Yue Xing

**Abstract**: We propose a theoretical framework demonstrating that popular Large Language Model (LLM) alignment methods, including Reinforcement Learning from Human Feedback (RLHF) and alternatives, fundamentally function as divergence estimators between aligned (preferred or safe) and unaligned (less-preferred or harmful) distributions. This explains the separation phenomenon between safe and harmful prompts in the model hidden representation after alignment. Inspired by the theoretical results, we identify that some alignment methods are better than others in terms of separation and, introduce a new method, KLDO, and further demonstrate the implication of our theories. We advocate for compliance-refusal datasets over preference datasets to enhance safety alignment, supported by both theoretical reasoning and empirical evidence. Additionally, to quantify safety separation, we leverage a distance metric in the representation space and statistically validate its efficacy as a statistical significant indicator of LLM resilience against jailbreak attacks.

摘要: 我们提出了一个理论框架，证明了流行的大语言模型(LLM)对齐方法，包括从人类反馈的强化学习(RLHF)和替代方法，基本上是对齐(优先或安全)和非对齐(较不优先或有害)分布之间的背离估计。这解释了对齐后模型隐藏表示中的安全提示和有害提示分离的现象。在理论结果的启发下，我们确定了一些比对方法在分离方面比其他方法更好，并介绍了一种新的方法KLDO，进一步论证了我们的理论的含义。我们主张使用合规拒绝数据集而不是偏好数据集，以增强安全性一致性，这得到了理论推理和经验证据的支持。此外，为了量化安全分离，我们利用表示空间中的距离度量，并在统计上验证其作为LLM对越狱攻击弹性的显著指标的有效性。



## **44. Towards Robust Multimodal Large Language Models Against Jailbreak Attacks**

迈向抵御越狱攻击的稳健多模式大型语言模型 cs.CR

**SubmitDate**: 2025-02-02    [abs](http://arxiv.org/abs/2502.00653v1) [paper-pdf](http://arxiv.org/pdf/2502.00653v1)

**Authors**: Ziyi Yin, Yuanpu Cao, Han Liu, Ting Wang, Jinghui Chen, Fenhlong Ma

**Abstract**: While multimodal large language models (MLLMs) have achieved remarkable success in recent advancements, their susceptibility to jailbreak attacks has come to light. In such attacks, adversaries exploit carefully crafted prompts to coerce models into generating harmful or undesirable content. Existing defense mechanisms often rely on external inference steps or safety alignment training, both of which are less effective and impractical when facing sophisticated adversarial perturbations in white-box scenarios. To address these challenges and bolster MLLM robustness, we introduce SafeMLLM by adopting an adversarial training framework that alternates between an attack step for generating adversarial noise and a model updating step. At the attack step, SafeMLLM generates adversarial perturbations through a newly proposed contrastive embedding attack (CoE-Attack), which optimizes token embeddings under a contrastive objective. SafeMLLM then updates model parameters to neutralize the perturbation effects while preserving model utility on benign inputs. We evaluate SafeMLLM across six MLLMs and six jailbreak methods spanning multiple modalities. Experimental results show that SafeMLLM effectively defends against diverse attacks, maintaining robust performance and utilities.

摘要: 虽然多模式大型语言模型(MLLM)在最近的进步中取得了显著的成功，但它们对越狱攻击的敏感性已经暴露出来。在此类攻击中，攻击者利用精心设计的提示来强迫模型生成有害或不受欢迎的内容。现有的防御机制往往依赖于外部推理步骤或安全对齐训练，在白盒场景中面对复杂的对手扰动时，这两种方法都不太有效和不切实际。为了应对这些挑战并增强MLLM的稳健性，我们引入了SafeMLLM，采用了一种对抗性训练框架，该框架在生成对抗性噪声的攻击步骤和模型更新步骤之间交替。在攻击阶段，SafeMLLM通过新提出的对比性嵌入攻击(COE-Attack)来产生敌意扰动，该攻击在对比性目标下优化令牌嵌入。SafeMLLM然后更新模型参数，以中和扰动影响，同时保留对良性输入的模型效用。我们评估了六种MLLM和六种越狱方法的SafeMLLM，这些方法跨越多个医疗设备。实验结果表明，SafeMLLM能够有效地防御各种攻击，并保持了较强的性能和实用性。



## **45. Self-Instruct Few-Shot Jailbreaking: Decompose the Attack into Pattern and Behavior Learning**

自学少枪越狱：将攻击分解为模式和行为学习 cs.AI

**SubmitDate**: 2025-02-01    [abs](http://arxiv.org/abs/2501.07959v2) [paper-pdf](http://arxiv.org/pdf/2501.07959v2)

**Authors**: Jiaqi Hua, Wanxu Wei

**Abstract**: Recently, several works have been conducted on jailbreaking Large Language Models (LLMs) with few-shot malicious demos. In particular, Zheng et al. focus on improving the efficiency of Few-Shot Jailbreaking (FSJ) by injecting special tokens into the demos and employing demo-level random search, known as Improved Few-Shot Jailbreaking (I-FSJ). Nevertheless, we notice that this method may still require a long context to jailbreak advanced models e.g. 32 shots of demos for Meta-Llama-3-8B-Instruct (Llama-3) \cite{llama3modelcard}. In this paper, we discuss the limitations of I-FSJ and propose Self-Instruct Few-Shot Jailbreaking (Self-Instruct-FSJ) facilitated with the demo-level greedy search. This framework decomposes the FSJ attack into pattern and behavior learning to exploit the model's vulnerabilities in a more generalized and efficient way. We conduct elaborate experiments to evaluate our method on common open-source models and compare it with baseline algorithms. Our code is available at https://github.com/iphosi/Self-Instruct-FSJ.

摘要: 最近，一些关于越狱大语言模型(LLM)的工作已经进行，并提供了几个几乎不可能成功的恶意演示。特别是，郑等人。专注于通过向演示中注入特殊令牌并采用演示级随机搜索来提高少发越狱(FSJ)的效率，即改进的少发越狱(I-FSJ)。然而，我们注意到，这种方法可能仍然需要较长的上下文才能越狱高级模型，例如Meta-Llama-3-8B-Indict(Llama-3)\Cite{llama3 ModelCard}的32个演示镜头。在本文中，我们讨论了I-FSJ的局限性，并提出了一种基于演示级贪婪搜索的自指导式少发越狱算法(SELF-Induct-FSJ)。该框架将FSJ攻击分解为模式学习和行为学习，以更通用、更有效的方式利用模型的漏洞。我们在常见的开源模型上进行了详细的实验来评估我们的方法，并将其与基线算法进行了比较。我们的代码可以在https://github.com/iphosi/Self-Instruct-FSJ.上找到



## **46. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation**

谜语我！检索增强一代的隐形会员推断 cs.CR

**SubmitDate**: 2025-02-01    [abs](http://arxiv.org/abs/2502.00306v1) [paper-pdf](http://arxiv.org/pdf/2502.00306v1)

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference.

摘要: 检索-增强生成(RAG)使大型语言模型(LLM)能够通过利用外部知识数据库生成接地响应，而无需更改模型参数。尽管没有权重调整防止通过模型参数泄漏，但它引入了推理对手在模型上下文中利用检索到的文档的风险。现有的成员关系推断和数据提取方法通常依赖于越狱或精心设计的非自然查询，这些查询可以很容易地被RAG系统中常见的查询重写技术检测到或阻止。在这项工作中，我们提出了询问攻击(IA)，这是一种针对RAG数据存储中的文档的成员关系推理技术。通过精心设计只能根据目标文档的存在来回答的自然文本查询，我们的方法只需30个查询即可成功推理，同时保持隐蔽性；直接的检测器识别来自现有方法的敌意提示的频率比我们的攻击生成的提示高约76倍。我们观察到，在不同的RAG配置中，TPR@1%的FPR比以前的推理攻击提高了2倍，而每个文档推理的成本都不到0.02美元。



## **47. Byzantine-Resilient Zero-Order Optimization for Communication-Efficient Heterogeneous Federated Learning**

具有拜占庭弹性的零阶优化，用于通信高效的异类联邦学习 cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2502.00193v1) [paper-pdf](http://arxiv.org/pdf/2502.00193v1)

**Authors**: Maximilian Egger, Mayank Bakshi, Rawad Bitar

**Abstract**: We introduce CyBeR-0, a Byzantine-resilient federated zero-order optimization method that is robust under Byzantine attacks and provides significant savings in uplink and downlink communication costs. We introduce transformed robust aggregation to give convergence guarantees for general non-convex objectives under client data heterogeneity. Empirical evaluations for standard learning tasks and fine-tuning large language models show that CyBeR-0 exhibits stable performance with only a few scalars per-round communication cost and reduced memory requirements.

摘要: 我们引入CyBeR-0，这是一种具有拜占庭弹性的联邦零阶优化方法，在拜占庭攻击下具有鲁棒性，并大幅节省上行链路和下行链路通信成本。我们引入转换的鲁棒聚合，为客户数据异类下的一般非凸目标提供收敛保证。对标准学习任务和微调大型语言模型的经验评估表明，CyBeR-0表现出稳定的性能，每轮通信成本只有几个纯量，内存需求也降低。



## **48. UniGuard: Towards Universal Safety Guardrails for Jailbreak Attacks on Multimodal Large Language Models**

UniGuard：为多模式大型语言模型越狱攻击建立通用安全护栏 cs.CL

14 pages

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2411.01703v2) [paper-pdf](http://arxiv.org/pdf/2411.01703v2)

**Authors**: Sejoon Oh, Yiqiao Jin, Megha Sharma, Donghyun Kim, Eric Ma, Gaurav Verma, Srijan Kumar

**Abstract**: Multimodal large language models (MLLMs) have revolutionized vision-language understanding but remain vulnerable to multimodal jailbreak attacks, where adversarial inputs are meticulously crafted to elicit harmful or inappropriate responses. We propose UniGuard, a novel multimodal safety guardrail that jointly considers the unimodal and cross-modal harmful signals. UniGuard trains a multimodal guardrail to minimize the likelihood of generating harmful responses in a toxic corpus. The guardrail can be seamlessly applied to any input prompt during inference with minimal computational costs. Extensive experiments demonstrate the generalizability of UniGuard across multiple modalities, attack strategies, and multiple state-of-the-art MLLMs, including LLaVA, Gemini Pro, GPT-4o, MiniGPT-4, and InstructBLIP. Notably, this robust defense mechanism maintains the models' overall vision-language understanding capabilities.

摘要: 多模式大型语言模型（MLLM）彻底改变了视觉语言理解，但仍然容易受到多模式越狱攻击的影响，其中对抗性输入经过精心设计，以引发有害或不当的反应。我们提出了UniGuard，这是一种新型的多模式安全护栏，它联合考虑了单模式和跨模式有害信号。UniGuard训练多模式护栏，以最大限度地降低有毒主体中产生有害反应的可能性。护栏可以无缝地应用于推理期间的任何输入提示，并且计算成本最低。大量实验证明了UniGuard在多种模式、攻击策略和多种最先进的MLLM中的通用性，包括LLaVA、Gemini Pro、GPT-4 o、MiniGPT-4和DirecectBLIP。值得注意的是，这种强大的防御机制维持了模型的整体视觉语言理解能力。



## **49. Enhancing Model Defense Against Jailbreaks with Proactive Safety Reasoning**

利用主动安全推理增强模型对越狱的防御 cs.CR

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2501.19180v1) [paper-pdf](http://arxiv.org/pdf/2501.19180v1)

**Authors**: Xianglin Yang, Gelei Deng, Jieming Shi, Tianwei Zhang, Jin Song Dong

**Abstract**: Large language models (LLMs) are vital for a wide range of applications yet remain susceptible to jailbreak threats, which could lead to the generation of inappropriate responses. Conventional defenses, such as refusal and adversarial training, often fail to cover corner cases or rare domains, leaving LLMs still vulnerable to more sophisticated attacks. We propose a novel defense strategy, Safety Chain-of-Thought (SCoT), which harnesses the enhanced \textit{reasoning capabilities} of LLMs for proactive assessment of harmful inputs, rather than simply blocking them. SCoT augments any refusal training datasets to critically analyze the intent behind each request before generating answers. By employing proactive reasoning, SCoT enhances the generalization of LLMs across varied harmful queries and scenarios not covered in the safety alignment corpus. Additionally, it generates detailed refusals specifying the rules violated. Comparative evaluations show that SCoT significantly surpasses existing defenses, reducing vulnerability to out-of-distribution issues and adversarial manipulations while maintaining strong general capabilities.

摘要: 大型语言模型(LLM)对于广泛的应用至关重要，但仍然容易受到越狱威胁的影响，这可能会导致产生不适当的响应。常规防御，如拒绝和对抗性训练，往往无法覆盖角落案例或稀有领域，使LLM仍然容易受到更复杂的攻击。我们提出了一种新的防御策略，安全思想链(SCOT)，它利用LLMS增强的\文本{推理能力}来主动评估有害输入，而不是简单地阻止它们。SCOT增加了任何拒绝训练数据集，以便在生成答案之前批判性地分析每个请求背后的意图。通过使用主动推理，SCOT增强了LLMS在安全匹配语料库中未涵盖的各种有害查询和场景中的泛化。此外，它还生成详细的拒绝，指定违反的规则。比较评估表明，SCOT大大超过了现有的防御系统，在保持强大的一般能力的同时，减少了对分配外问题和对抗性操纵的脆弱性。



## **50. Targeted Vaccine: Safety Alignment for Large Language Models against Harmful Fine-Tuning via Layer-wise Perturbation**

有针对性的疫苗：大型语言模型的安全调整，防止通过分层扰动进行有害的微调 cs.LG

**SubmitDate**: 2025-01-31    [abs](http://arxiv.org/abs/2410.09760v3) [paper-pdf](http://arxiv.org/pdf/2410.09760v3)

**Authors**: Guozhi Liu, Weiwei Lin, Tiansheng Huang, Ruichao Mo, Qi Mu, Li Shen

**Abstract**: Harmful fine-tuning attack poses a serious threat to the online fine-tuning service. Vaccine, a recent alignment-stage defense, applies uniform perturbation to all layers of embedding to make the model robust to the simulated embedding drift. However, applying layer-wise uniform perturbation may lead to excess perturbations for some particular safety-irrelevant layers, resulting in defense performance degradation and unnecessary memory consumption. To address this limitation, we propose Targeted Vaccine (T-Vaccine), a memory-efficient safety alignment method that applies perturbation to only selected layers of the model. T-Vaccine follows two core steps: First, it uses gradient norm as a statistical metric to identify the safety-critical layers. Second, instead of applying uniform perturbation across all layers, T-Vaccine only applies perturbation to the safety-critical layers while keeping other layers frozen during training. Results show that T-Vaccine outperforms Vaccine in terms of both defense effectiveness and resource efficiency. Comparison with other defense baselines, e.g., RepNoise and TAR also demonstrate the superiority of T-Vaccine. Notably, T-Vaccine is the first defense that can address harmful fine-tuning issues for a 7B pre-trained models trained on consumer GPUs with limited memory (e.g., RTX 4090). Our code is available at https://github.com/Lslland/T-Vaccine.

摘要: 有害微调攻击对在线微调服务构成严重威胁。疫苗是最近的一种对齐阶段防御方法，它将均匀扰动应用于嵌入的所有层，以使模型对模拟的嵌入漂移具有鲁棒性。然而，分层均匀扰动可能会导致某些特定安全无关层的过度扰动，导致防御性能下降和不必要的内存消耗。为了解决这一局限性，我们提出了靶向疫苗(T-Vaccine)，这是一种内存高效的安全对齐方法，仅对模型的选定层应用扰动。T-Vaccine遵循两个核心步骤：首先，它使用梯度范数作为统计度量来识别安全关键层。其次，T-Vaccine不是在所有层上应用统一的扰动，而是只对安全关键层应用扰动，而在训练期间保持其他层的冻结。结果表明，无论是防御效果还是资源效率，T疫苗都优于疫苗。与其他防御基线如RepNoise和TAR的比较也证明了T-疫苗的优越性。值得注意的是，T-Vaccine是第一个可以解决7B预培训模型的有害微调问题的防御系统，这些模型在内存有限的消费者GPU(例如RTX 4090)上进行了培训。我们的代码可以在https://github.com/Lslland/T-Vaccine.上找到



