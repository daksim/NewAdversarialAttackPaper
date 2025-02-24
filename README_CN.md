# Latest Adversarial Attack Papers
**update at 2025-02-24 09:53:17**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Red-Teaming LLM Multi-Agent Systems via Communication Attacks**

通过通信攻击的Red-Teaming LLM多代理系统 cs.CR

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14847v1) [paper-pdf](http://arxiv.org/pdf/2502.14847v1)

**Authors**: Pengfei He, Yupin Lin, Shen Dong, Han Xu, Yue Xing, Hui Liu

**Abstract**: Large Language Model-based Multi-Agent Systems (LLM-MAS) have revolutionized complex problem-solving capability by enabling sophisticated agent collaboration through message-based communications. While the communication framework is crucial for agent coordination, it also introduces a critical yet unexplored security vulnerability. In this work, we introduce Agent-in-the-Middle (AiTM), a novel attack that exploits the fundamental communication mechanisms in LLM-MAS by intercepting and manipulating inter-agent messages. Unlike existing attacks that compromise individual agents, AiTM demonstrates how an adversary can compromise entire multi-agent systems by only manipulating the messages passing between agents. To enable the attack under the challenges of limited control and role-restricted communication format, we develop an LLM-powered adversarial agent with a reflection mechanism that generates contextually-aware malicious instructions. Our comprehensive evaluation across various frameworks, communication structures, and real-world applications demonstrates that LLM-MAS is vulnerable to communication-based attacks, highlighting the need for robust security measures in multi-agent systems.

摘要: 基于大型语言模型的多代理系统(LLM-MAS)通过基于消息的通信实现复杂的代理协作，使复杂问题的解决能力发生了革命性的变化。虽然通信框架对于代理协调至关重要，但它也引入了一个严重但尚未探索的安全漏洞。在这项工作中，我们引入了中间代理(AiTM)，这是一种通过拦截和处理代理间消息来利用LLM-MAS的基本通信机制的新型攻击。与现有的危害单个代理的攻击不同，AiTM演示了对手如何通过仅操纵代理之间传递的消息来危害整个多代理系统。为了在有限控制和角色受限通信格式的挑战下实现攻击，我们开发了一个基于LLM的恶意代理，该代理具有反射机制，可以生成上下文感知的恶意指令。我们对各种框架、通信结构和现实世界应用的综合评估表明，LLM-MAS容易受到基于通信的攻击，这突显了在多代理系统中需要强大的安全措施。



## **2. Fundamental Limitations in Defending LLM Finetuning APIs**

捍卫LLM微调API的基本局限性 cs.LG

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14828v1) [paper-pdf](http://arxiv.org/pdf/2502.14828v1)

**Authors**: Xander Davies, Eric Winsor, Tomek Korbak, Alexandra Souly, Robert Kirk, Christian Schroeder de Witt, Yarin Gal

**Abstract**: LLM developers have imposed technical interventions to prevent fine-tuning misuse attacks, attacks where adversaries evade safeguards by fine-tuning the model using a public API. Previous work has established several successful attacks against specific fine-tuning API defences. In this work, we show that defences of fine-tuning APIs that seek to detect individual harmful training or inference samples ('pointwise' detection) are fundamentally limited in their ability to prevent fine-tuning attacks. We construct 'pointwise-undetectable' attacks that repurpose entropy in benign model outputs (e.g. semantic or syntactic variations) to covertly transmit dangerous knowledge. Our attacks are composed solely of unsuspicious benign samples that can be collected from the model before fine-tuning, meaning training and inference samples are all individually benign and low-perplexity. We test our attacks against the OpenAI fine-tuning API, finding they succeed in eliciting answers to harmful multiple-choice questions, and that they evade an enhanced monitoring system we design that successfully detects other fine-tuning attacks. We encourage the community to develop defences that tackle the fundamental limitations we uncover in pointwise fine-tuning API defences.

摘要: LLM开发人员实施了技术干预，以防止微调误用攻击，即攻击者通过使用公共API微调模型来逃避安全保护的攻击。以前的工作已经成功地建立了几次针对特定微调API防御的攻击。在这项工作中，我们证明了试图检测单个有害训练或推理样本(按点检测)的微调API的防御在防止微调攻击的能力方面基本上是有限的。我们构造了“逐点不可检测”攻击，重新利用良性模型输出(例如，语义或句法变化)中的熵来隐蔽地传递危险知识。我们的攻击完全由不可疑的良性样本组成，这些样本可以在微调之前从模型中收集，这意味着训练和推理样本都是单独的良性和低困惑。我们针对OpenAI微调API测试了我们的攻击，发现它们成功地获得了有害多项选择题的答案，并且它们避开了我们设计的增强型监控系统，该系统可以成功检测到其他微调攻击。我们鼓励社区开发防御，以解决我们在点状微调API防御中发现的基本限制。



## **3. HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States**

HiddenDetect：通过监视隐藏状态检测针对大型视觉语言模型的越狱攻击 cs.CL

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.14744v2) [paper-pdf](http://arxiv.org/pdf/2502.14744v2)

**Authors**: Yilei Jiang, Xinyan Gao, Tianshuo Peng, Yingshui Tan, Xiaoyong Zhu, Bo Zheng, Xiangyu Yue

**Abstract**: The integration of additional modalities increases the susceptibility of large vision-language models (LVLMs) to safety risks, such as jailbreak attacks, compared to their language-only counterparts. While existing research primarily focuses on post-hoc alignment techniques, the underlying safety mechanisms within LVLMs remain largely unexplored. In this work , we investigate whether LVLMs inherently encode safety-relevant signals within their internal activations during inference. Our findings reveal that LVLMs exhibit distinct activation patterns when processing unsafe prompts, which can be leveraged to detect and mitigate adversarial inputs without requiring extensive fine-tuning. Building on this insight, we introduce HiddenDetect, a novel tuning-free framework that harnesses internal model activations to enhance safety. Experimental results show that {HiddenDetect} surpasses state-of-the-art methods in detecting jailbreak attacks against LVLMs. By utilizing intrinsic safety-aware patterns, our method provides an efficient and scalable solution for strengthening LVLM robustness against multimodal threats. Our code will be released publicly at https://github.com/leigest519/HiddenDetect.

摘要: 与纯语言模型相比，更多模式的集成增加了大型视觉语言模型(LVLM)对安全风险(如越狱袭击)的敏感性。虽然现有的研究主要集中在后对齐技术上，但左腰椎内潜在的安全机制在很大程度上仍未被探索。在这项工作中，我们调查了在推理过程中，LVLM是否内在地在其内部激活中编码与安全相关的信号。我们的发现表明，在处理不安全提示时，LVLMS显示出不同的激活模式，这可以被用来检测和缓解敌意输入，而不需要进行广泛的微调。基于这一见解，我们引入了HiddenDetect，这是一个新的免调优框架，它利用内部模型激活来增强安全性。实验结果表明，{HiddenDetect}在检测针对LVLMS的越狱攻击方面优于最先进的方法。通过利用固有的安全感知模式，我们的方法提供了一种高效且可扩展的解决方案，以增强LVLM对多模式威胁的健壮性。我们的代码将在https://github.com/leigest519/HiddenDetect.上公开发布



## **4. SEA: Shareable and Explainable Attribution for Query-based Black-box Attacks**

SEA：基于查询的黑匣子攻击的可共享和可解释归因 cs.LG

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2308.11845v2) [paper-pdf](http://arxiv.org/pdf/2308.11845v2)

**Authors**: Yue Gao, Ilia Shumailov, Kassem Fawaz

**Abstract**: Machine Learning (ML) systems are vulnerable to adversarial examples, particularly those from query-based black-box attacks. Despite various efforts to detect and prevent such attacks, ML systems are still at risk, demanding a more comprehensive approach to security that includes logging, analyzing, and sharing evidence. While traditional security benefits from well-established practices of forensics and threat intelligence sharing, ML security has yet to find a way to profile its attackers and share information about them. In response, this paper introduces SEA, a novel ML security system to characterize black-box attacks on ML systems for forensic purposes and to facilitate human-explainable intelligence sharing. SEA leverages Hidden Markov Models to attribute the observed query sequence to known attacks. It thus understands the attack's progression rather than focusing solely on the final adversarial examples. Our evaluations reveal that SEA is effective at attack attribution, even on the second incident, and is robust to adaptive strategies designed to evade forensic analysis. SEA's explanations of the attack's behavior allow us even to fingerprint specific minor bugs in widely used attack libraries. For example, we discover that the SignOPT and Square attacks in ART v1.14 send over 50% duplicated queries. We thoroughly evaluate SEA on a variety of settings and demonstrate that it can recognize the same attack with more than 90% Top-1 and 95% Top-3 accuracy. Finally, we demonstrate how SEA generalizes to other domains like text classification.

摘要: 机器学习(ML)系统容易受到敌意例子的攻击，特别是那些来自基于查询的黑盒攻击的例子。尽管做出了各种努力来检测和防止此类攻击，但ML系统仍然处于风险之中，需要一种更全面的安全方法，包括记录、分析和共享证据。虽然传统安全受益于成熟的取证和威胁情报共享做法，但ML安全尚未找到一种方法来分析其攻击者并共享有关他们的信息。对此，本文引入了一种新的ML安全系统SEA，用于刻画针对ML系统的黑盒攻击，用于取证目的，并促进人类可解释的情报共享。SEA利用隐马尔可夫模型将观察到的查询序列归因于已知攻击。因此，它了解攻击的进展，而不是只关注最后的对抗性例子。我们的评估表明，SEA在攻击归因方面是有效的，即使在第二起事件中也是如此，并且对于旨在逃避法医分析的自适应策略是健壮的。SEA对攻击行为的解释甚至允许我们在广泛使用的攻击库中提取特定的小错误。例如，我们发现ART v1.14中的SignOPT和Square攻击发送了超过50%的重复查询。我们在各种不同的设置下对SEA进行了全面的评估，并证明了它能够以超过90%的Top-1和95%的Top-3的准确率识别相同的攻击。最后，我们演示了SEA如何推广到文本分类等其他领域。



## **5. Pitch Imperfect: Detecting Audio Deepfakes Through Acoustic Prosodic Analysis**

音调不完美：通过声学韵律分析检测音频Deepfake cs.SD

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14726v1) [paper-pdf](http://arxiv.org/pdf/2502.14726v1)

**Authors**: Kevin Warren, Daniel Olszewski, Seth Layton, Kevin Butler, Carrie Gates, Patrick Traynor

**Abstract**: Audio deepfakes are increasingly in-differentiable from organic speech, often fooling both authentication systems and human listeners. While many techniques use low-level audio features or optimization black-box model training, focusing on the features that humans use to recognize speech will likely be a more long-term robust approach to detection. We explore the use of prosody, or the high-level linguistic features of human speech (e.g., pitch, intonation, jitter) as a more foundational means of detecting audio deepfakes. We develop a detector based on six classical prosodic features and demonstrate that our model performs as well as other baseline models used by the community to detect audio deepfakes with an accuracy of 93% and an EER of 24.7%. More importantly, we demonstrate the benefits of using a linguistic features-based approach over existing models by applying an adaptive adversary using an $L_{\infty}$ norm attack against the detectors and using attention mechanisms in our training for explainability. We show that we can explain the prosodic features that have highest impact on the model's decision (Jitter, Shimmer and Mean Fundamental Frequency) and that other models are extremely susceptible to simple $L_{\infty}$ norm attacks (99.3% relative degradation in accuracy). While overall performance may be similar, we illustrate the robustness and explainability benefits to a prosody feature approach to audio deepfake detection.

摘要: 音频深度假冒与有机语音越来越难以区分，经常愚弄身份验证系统和人类听众。虽然许多技术使用低级音频特征或优化黑盒模型训练，但关注人类用来识别语音的特征可能是一种更长期、更稳健的检测方法。我们探索了使用韵律或人类语音的高级语言特征(例如，音调、语调、抖动)作为检测音频深度虚假的更基本的方法。我们开发了一个基于六个经典韵律特征的检测器，并证明了我们的模型的性能与社区使用的其他基线模型一样，检测音频深度虚假的准确率为93%，EER为24.7%。更重要的是，我们展示了使用基于语言特征的方法比现有模型的好处，方法是应用一个自适应对手，使用$L对检测器的范数攻击，并在我们的可解释性训练中使用注意机制。我们表明，我们可以解释对模型决策影响最大的韵律特征(抖动、抖动和平均基频)，而其他模型非常容易受到简单的$L$范数攻击(精度相对下降99.3%)。虽然总体性能可能相似，但我们说明了韵律特征方法对音频深度伪检测的稳健性和可解释性的好处。



## **6. Moshi Moshi? A Model Selection Hijacking Adversarial Attack**

莫西？模型选择劫持对抗攻击 cs.LG

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14586v1) [paper-pdf](http://arxiv.org/pdf/2502.14586v1)

**Authors**: Riccardo Petrucci, Luca Pajola, Francesco Marchiori, Luca Pasa, Mauro conti

**Abstract**: Model selection is a fundamental task in Machine Learning~(ML), focusing on selecting the most suitable model from a pool of candidates by evaluating their performance on specific metrics. This process ensures optimal performance, computational efficiency, and adaptability to diverse tasks and environments. Despite its critical role, its security from the perspective of adversarial ML remains unexplored. This risk is heightened in the Machine-Learning-as-a-Service model, where users delegate the training phase and the model selection process to third-party providers, supplying data and training strategies. Therefore, attacks on model selection could harm both the user and the provider, undermining model performance and driving up operational costs.   In this work, we present MOSHI (MOdel Selection HIjacking adversarial attack), the first adversarial attack specifically targeting model selection. Our novel approach manipulates model selection data to favor the adversary, even without prior knowledge of the system. Utilizing a framework based on Variational Auto Encoders, we provide evidence that an attacker can induce inefficiencies in ML deployment. We test our attack on diverse computer vision and speech recognition benchmark tasks and different settings, obtaining an average attack success rate of 75.42%. In particular, our attack causes an average 88.30% decrease in generalization capabilities, an 83.33% increase in latency, and an increase of up to 105.85% in energy consumption. These results highlight the significant vulnerabilities in model selection processes and their potential impact on real-world applications.

摘要: 模型选择是机器学习中的一项基本任务，通过评估候选模型在特定指标上的表现，从候选模型库中选择出最合适的模型。这一过程确保了最佳的性能、计算效率以及对不同任务和环境的适应性。尽管它扮演着关键的角色，但从敌对ML的角度来看，它的安全性仍然没有得到探索。这种风险在机器学习即服务模式中加剧，在这种模式下，用户将培训阶段和模式选择过程委托给第三方提供商，提供数据和培训策略。因此，对模型选择的攻击可能会损害用户和提供商，破坏模型性能并推高运营成本。在这项工作中，我们提出了Moshi(模型选择劫持对抗攻击)，这是第一个专门针对模型选择的对抗性攻击。我们的新方法操纵模型选择数据以有利于对手，即使在没有系统先验知识的情况下也是如此。利用一个基于可变自动编码器的框架，我们提供了攻击者可以在ML部署中导致低效的证据。我们在不同的计算机视觉和语音识别基准任务和不同的设置上测试了我们的攻击，平均攻击成功率为75.42%。特别是，我们的攻击导致泛化能力平均下降88.30%，延迟增加83.33%，能耗增加高达105.85%。这些结果突出了模型选择过程中的重大漏洞及其对现实世界应用程序的潜在影响。



## **7. Eliminating Backdoors in Neural Code Models for Secure Code Understanding**

消除神经代码模型中的后门以实现安全代码理解 cs.CR

Accepted to the 33rd ACM International Conference on the Foundations  of Software Engineering (FSE 2025)

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2408.04683v2) [paper-pdf](http://arxiv.org/pdf/2408.04683v2)

**Authors**: Weisong Sun, Yuchen Chen, Chunrong Fang, Yebo Feng, Yuan Xiao, An Guo, Quanjun Zhang, Yang Liu, Baowen Xu, Zhenyu Chen

**Abstract**: Neural code models (NCMs) have been widely used to address various code understanding tasks, such as defect detection. However, numerous recent studies reveal that such models are vulnerable to backdoor attacks. Backdoored NCMs function normally on normal/clean code snippets, but exhibit adversary-expected behavior on poisoned code snippets injected with the adversary-crafted trigger. It poses a significant security threat. Therefore, there is an urgent need for effective techniques to detect and eliminate backdoors stealthily implanted in NCMs.   To address this issue, in this paper, we innovatively propose a backdoor elimination technique for secure code understanding, called EliBadCode. EliBadCode eliminates backdoors in NCMs by inverting/reverse-engineering and unlearning backdoor triggers. Specifically, EliBadCode first filters the model vocabulary for trigger tokens based on the naming conventions of specific programming languages to reduce the trigger search space and cost. Then, EliBadCode introduces a sample-specific trigger position identification method, which can reduce the interference of non-backdoor (adversarial) perturbations for subsequent trigger inversion, thereby producing effective inverted backdoor triggers efficiently. Backdoor triggers can be viewed as backdoor (adversarial) perturbations. Subsequently, EliBadCode employs a Greedy Coordinate Gradient algorithm to optimize the inverted trigger and designs a trigger anchoring method to purify the inverted trigger. Finally, EliBadCode eliminates backdoors through model unlearning. We evaluate the effectiveness of EliBadCode in eliminating backdoors implanted in multiple NCMs used for three safety-critical code understanding tasks. The results demonstrate that EliBadCode can effectively eliminate backdoors while having minimal adverse effects on the normal functionality of the model.

摘要: 神经代码模型(NCM)已被广泛用于解决各种代码理解任务，如缺陷检测。然而，最近的大量研究表明，这种模型很容易受到后门攻击。背道而驰的NCMS在正常/干净的代码片段上运行正常，但在注入了对手定制的触发器的有毒代码片段上表现出对手预期的行为。它构成了重大的安全威胁。因此，迫切需要有效的技术来检测和消除在新农合中悄悄植入的后门。针对这一问题，在本文中，我们创新性地提出了一种用于安全代码理解的后门消除技术，称为EliBadCode。EliBadCode通过反转/反向工程和忘记后门触发器来消除NCMS中的后门。具体地说，EliBadCode首先根据特定编程语言的命名约定过滤触发器令牌的模型词汇表，以减少触发器搜索空间和成本。然后，EliBadCode引入了一种特定样本的触发器位置识别方法，该方法可以减少非后门(对抗性)扰动对后续触发器反转的干扰，从而高效地产生有效的倒置后门触发器。后门触发器可以被视为后门(对抗性)扰动。随后，EliBadCode使用贪婪坐标梯度算法对倒置触发器进行优化，并设计了触发器锚定方法对倒置触发器进行净化。最后，EliBadCode通过模型遗忘消除了后门。我们评估了EliBadCode在消除植入用于三个安全关键代码理解任务的多个NCM中的后门方面的有效性。结果表明，EliBadCode可以有效地消除后门，同时对模型的正常功能产生的负面影响最小。



## **8. Combined Quantum and Post-Quantum Security for Earth-Satellite Channels**

地球-卫星通道的组合量子和后量子安全 quant-ph

9 pages, 5 figures, 1 table, IEEE, International Conference on  Quantum Communications, Networking, and Computing 2025

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14240v1) [paper-pdf](http://arxiv.org/pdf/2502.14240v1)

**Authors**: Anju Rani, Xiaoyu Ai, Aman Gupta, Ravi Singh Adhikari, Robert Malaney

**Abstract**: Experimental deployment of quantum communication over Earth-satellite channels opens the way to a secure global quantum Internet. In this work, we present results from a real-time prototype quantum key distribution (QKD) system, which entails the development of optical systems including the encoding of entangled photon pairs, the development of transmitters for quantum signaling through an emulated Earth-satellite channel, and the development of quantum-decoding receivers. A unique aspect of our system is the integration of QKD with existing cryptographic methods to ensure quantum-resistant security, even at low-key rates. In addition, we report the use of specially designed error-reconciliation codes that optimize the security versus key-rate trade-off. Our work demonstrates, for the first time, a deployment of the BBM92 protocol that offers both post-quantum security via the advanced encryption standard (AES) and quantum security via an entanglement-based QKD protocol. If either the AES or the QKD is compromised through some adversary attack, our system still delivers state-of-the-art communications secure against future quantum computers.

摘要: 在地-星通道上进行量子通信的试验部署，为安全的全球量子互联网开辟了道路。在这项工作中，我们介绍了一个实时原型量子密钥分配(QKD)系统的结果，该系统需要开发光学系统，包括纠缠光子对的编码，通过模拟的地球-卫星信道开发用于量子信号传输的发射机，以及量子解码接收器的开发。我们系统的一个独特方面是将量子密钥分发与现有的加密方法相结合，以确保量子抵抗安全，即使在低密码率下也是如此。此外，我们还报告了如何使用专门设计的错误协调代码来优化安全性与密钥率之间的权衡。我们的工作首次展示了BBM92协议的部署，该协议通过高级加密标准(AES)提供后量子安全，并通过基于纠缠的QKD协议提供量子安全。如果AES或QKD通过某种对手攻击而被攻破，我们的系统仍然可以提供最先进的通信，以抵御未来的量子计算机。



## **9. Scaling Trends in Language Model Robustness**

语言模型稳健性的缩放趋势 cs.LG

58 pages; updated to include new results and analysis

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2407.18213v4) [paper-pdf](http://arxiv.org/pdf/2407.18213v4)

**Authors**: Nikolaus Howe, Ian McKenzie, Oskar Hollinsworth, Michał Zajac, Tom Tseng, Aaron Tucker, Pierre-Luc Bacon, Adam Gleave

**Abstract**: Language models exhibit scaling laws, whereby increasing model and dataset size predictably decrease negative log likelihood, unlocking a dazzling array of capabilities. At the same time, even the most capable systems are currently vulnerable to adversarial inputs such as jailbreaks and prompt injections, despite concerted efforts to make them robust. As compute becomes more accessible to both attackers and defenders, which side will benefit more from scale? We attempt to answer this question with a detailed study of robustness on language models spanning three orders of magnitude in parameter count. From the defender's perspective, we find that in the absence of other interventions, increasing model size alone does not consistently improve robustness. In adversarial training, we find that larger models are more sample-efficient and less compute-efficient than smaller models, and often better generalize their defense to new threat models. From the attacker's perspective, we find that increasing attack compute smoothly and reliably increases attack success rate against both finetuned and adversarially trained models. Finally, we show that across model sizes studied, doubling compute on adversarial training only forces an attacker to less than double attack compute to maintain the same attack success rate. However, adversarial training becomes more and more effective on larger models, suggesting that defenders could eventually have the advantage with increasing model size. These results underscore the value of adopting a scaling lens when discussing robustness of frontier models.

摘要: 语言模型表现出伸缩规律，借此增加模型和数据集大小可预测地降低负对数可能性，从而释放出一系列令人眼花缭乱的功能。与此同时，即使是最有能力的系统目前也很容易受到敌意输入的影响，例如越狱和及时注射，尽管各方共同努力使它们变得健壮。随着攻击者和防御者都更容易访问计算，哪一方将从扩展中获得更多好处？我们试图通过详细研究跨越参数计数三个数量级的语言模型的稳健性来回答这个问题。从捍卫者的角度来看，我们发现，在没有其他干预的情况下，仅增加模型规模并不能始终如一地提高稳健性。在对抗性训练中，我们发现较大的模型比较小的模型具有更高的样本效率和较低的计算效率，并且通常更好地将其防御推广到新的威胁模型。从攻击者的角度来看，我们发现，增加攻击计算平稳可靠地提高了对精调模型和对抗性训练模型的攻击成功率。最后，我们证明了在所研究的模型大小中，对抗性训练的计算量加倍仅迫使攻击者在保持相同的攻击成功率的情况下只需要不到两倍的攻击计算量。然而，对抗性训练在更大的模型上变得越来越有效，这表明随着模型尺寸的增加，防守端最终可能拥有优势。这些结果强调了在讨论前沿模型的稳健性时采用比例透镜的价值。



## **10. Control Barrier Function based Attack-Recovery with Provable Guarantees**

基于控制屏障功能的攻击恢复，具有可证明的保证 eess.SY

V1: Conference version (IEEE CDC'2022) V2: Journal version (submitted  to IEEE Transactions on Automatic Control)

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2204.03077v2) [paper-pdf](http://arxiv.org/pdf/2204.03077v2)

**Authors**: Kunal Garg, Ricardo G. Sanfelice, Alvaro A. Cardenas

**Abstract**: This paper studies provable security guarantees for cyber-physical systems (CPS) under actuator attacks. In particular, we consider CPS safety and propose a new attack detection mechanism based on zeroing control barrier function (ZCBF) conditions. In addition, we design an adaptive recovery mechanism based on how close the system is to violating safety. We show that under certain conditions, the attack-detection mechanism is sound, i.e., there are no false negatives for adversarial attacks. We propose sufficient conditions for the initial conditions and input constraints so that the resulting CPS is secure by design. We also propose a novel hybrid control to account for attack detection delays and avoid Zeno behavior. Next, to efficiently compute the set of initial conditions, we propose a sampling-based method to verify whether a set is a viability domain. Specifically, we devise a method for checking a modified barrier function condition on a finite set of points to assess whether a set can be rendered forward invariant. Then, we propose an iterative algorithm to compute the set of initial conditions and input constraints set to limit the effect of an adversary if it compromises vulnerable inputs. Finally, we use a Quadratic Programming (QP) approach for online recovery (as well as nominal) control synthesis. We demonstrate the effectiveness of the proposed method in a simulation case study involving a quadrotor with an attack on its motors.

摘要: 研究了网络物理系统在执行器攻击下的可证明安全保证问题。特别地，我们考虑了CPS的安全性，提出了一种基于归零控制屏障函数(ZCBF)条件的攻击检测机制。此外，我们还设计了一种基于系统接近违反安全的程度的自适应恢复机制。我们证明了在一定条件下，攻击检测机制是健全的，即对于对抗性攻击没有漏报。我们提出了初始条件和输入约束的充分条件，使得所得到的CPS在设计上是安全的。我们还提出了一种新的混合控制来解决攻击检测延迟和避免Zeno行为。接下来，为了有效地计算初始条件集合，我们提出了一种基于采样的方法来验证集合是否为生存性域。具体地说，我们设计了一种方法，用于检查有限点集上的修改的障碍函数条件，以评估一个集是否可以被呈现为向前不变。然后，我们提出了一种迭代算法来计算初始条件集和输入约束集，以便在对手危及易受攻击的输入时限制其影响。最后，我们使用二次规划(QP)方法进行在线恢复(以及标称)控制综合。在一个四旋翼发动机受到攻击的仿真案例研究中，我们证明了所提方法的有效性。



## **11. Carefully Blending Adversarial Training, Purification, and Aggregation Improves Adversarial Robustness**

仔细混合对抗训练、净化和聚集提高对抗稳健性 cs.CV

25 pages, 1 figure, 16 tables

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2306.06081v5) [paper-pdf](http://arxiv.org/pdf/2306.06081v5)

**Authors**: Emanuele Ballarin, Alessio Ansuini, Luca Bortolussi

**Abstract**: In this work, we propose a novel adversarial defence mechanism for image classification - CARSO - blending the paradigms of adversarial training and adversarial purification in a synergistic robustness-enhancing way. The method builds upon an adversarially-trained classifier, and learns to map its internal representation associated with a potentially perturbed input onto a distribution of tentative clean reconstructions. Multiple samples from such distribution are classified by the same adversarially-trained model, and a carefully chosen aggregation of its outputs finally constitutes the robust prediction of interest. Experimental evaluation by a well-established benchmark of strong adaptive attacks, across different image datasets, shows that CARSO is able to defend itself against adaptive end-to-end white-box attacks devised for stochastic defences. Paying a modest clean accuracy toll, our method improves by a significant margin the state-of-the-art for Cifar-10, Cifar-100, and TinyImageNet-200 $\ell_\infty$ robust classification accuracy against AutoAttack. Code, and instructions to obtain pre-trained models are available at: https://github.com/emaballarin/CARSO .

摘要: 在这项工作中，我们提出了一种新的用于图像分类的对抗性防御机制-CASO-以协同增强鲁棒性的方式融合了对抗性训练和对抗性净化的范例。该方法建立在对抗性训练的分类器之上，并学习将其与潜在扰动输入相关联的内部表示映射到试探性干净重构的分布上。来自这种分布的多个样本被相同的对抗性训练模型分类，其输出的精心选择的聚集最终构成了感兴趣的稳健预测。基于不同图像数据集的强自适应攻击基准的实验评估表明，CARSO能够抵抗针对随机防御而设计的自适应端到端白盒攻击。付出适度的干净精度代价，我们的方法显著提高了CIFAR-10、CIFAR-100和TinyImageNet-200$\ell_\inty$相对于AutoAttack的稳健分类精度。代码，以及获取预先训练的模型的说明，请访问：https://github.com/emaballarin/CARSO。



## **12. Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach**

LLM水印的理论基础框架：分布自适应方法 cs.CR

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2410.02890v4) [paper-pdf](http://arxiv.org/pdf/2410.02890v4)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Watermarking has emerged as a crucial method to distinguish AI-generated text from human-created text. In this paper, we present a novel theoretical framework for watermarking Large Language Models (LLMs) that jointly optimizes both the watermarking scheme and the detection process. Our approach focuses on maximizing detection performance while maintaining control over the worst-case Type-I error and text distortion. We characterize \emph{the universally minimum Type-II error}, showing a fundamental trade-off between watermark detectability and text distortion. Importantly, we identify that the optimal watermarking schemes are adaptive to the LLM generative distribution. Building on our theoretical insights, we propose an efficient, model-agnostic, distribution-adaptive watermarking algorithm, utilizing a surrogate model alongside the Gumbel-max trick. Experiments conducted on Llama2-13B and Mistral-8$\times$7B models confirm the effectiveness of our approach. Additionally, we examine incorporating robustness into our framework, paving a way to future watermarking systems that withstand adversarial attacks more effectively.

摘要: 数字水印已经成为区分人工智能生成的文本和人类创建的文本的关键方法。在本文中，我们提出了一种新的大语言模型(LLMS)水印理论框架，该框架同时优化了水印方案和检测过程。我们的方法专注于最大化检测性能，同时保持对最坏情况下的类型I错误和文本失真的控制。我们将其刻画在水印可检测性和文本失真之间的基本权衡。重要的是，我们发现最优水印方案对LLM生成分布是自适应的。基于我们的理论见解，我们提出了一种高效的、与模型无关的、分布自适应的水印算法，该算法利用代理模型和Gumbel-max技巧。在Llama2-13B和Mistral-8$\x$70亿模型上进行的实验证实了该方法的有效性。此外，我们还研究了将健壮性融入到我们的框架中，为未来更有效地抵御对手攻击的水印系统铺平了道路。



## **13. Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives**

从时间和形态角度重新思考视听对抗脆弱性 cs.SD

Accepted by ICLR 2025

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2502.11858v2) [paper-pdf](http://arxiv.org/pdf/2502.11858v2)

**Authors**: Zeliang Zhang, Susan Liang, Daiki Shimada, Chenliang Xu

**Abstract**: While audio-visual learning equips models with a richer understanding of the real world by leveraging multiple sensory modalities, this integration also introduces new vulnerabilities to adversarial attacks.   In this paper, we present a comprehensive study of the adversarial robustness of audio-visual models, considering both temporal and modality-specific vulnerabilities. We propose two powerful adversarial attacks: 1) a temporal invariance attack that exploits the inherent temporal redundancy across consecutive time segments and 2) a modality misalignment attack that introduces incongruence between the audio and visual modalities. These attacks are designed to thoroughly assess the robustness of audio-visual models against diverse threats. Furthermore, to defend against such attacks, we introduce a novel audio-visual adversarial training framework. This framework addresses key challenges in vanilla adversarial training by incorporating efficient adversarial perturbation crafting tailored to multi-modal data and an adversarial curriculum strategy. Extensive experiments in the Kinetics-Sounds dataset demonstrate that our proposed temporal and modality-based attacks in degrading model performance can achieve state-of-the-art performance, while our adversarial training defense largely improves the adversarial robustness as well as the adversarial training efficiency.

摘要: 虽然视听学习通过利用多种感官模式使模型对真实世界有了更丰富的理解，但这种集成也引入了新的易受对手攻击的漏洞。在本文中，我们对视听模型的对抗健壮性进行了全面的研究，同时考虑了时间和通道特定的脆弱性。我们提出了两个强大的对抗性攻击：1)利用连续时间段固有的时间冗余性的时间不变性攻击；2)引入视听通道不一致的通道失准攻击。这些攻击旨在彻底评估视听模型对各种威胁的稳健性。此外，为了防御此类攻击，我们引入了一种新的视听对抗性训练框架。这一框架通过结合为多模式数据量身定做的高效对抗性扰动制作和对抗性课程战略，解决了普通对抗性训练中的关键挑战。在Kinetics-Sound数据集上的大量实验表明，我们提出的基于时间和通道的攻击在降低模型性能的同时可以获得最先进的性能，而我们的对抗性训练防御在很大程度上提高了对抗性的健壮性和对抗性训练的效率。



## **14. Na'vi or Knave: Jailbreaking Language Models via Metaphorical Avatars**

纳威人或恶棍：通过隐喻化身越狱语言模型 cs.CL

We still need to polish our paper

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2412.12145v3) [paper-pdf](http://arxiv.org/pdf/2412.12145v3)

**Authors**: Yu Yan, Sheng Sun, Junqi Tong, Min Liu, Qi Li

**Abstract**: Metaphor serves as an implicit approach to convey information, while enabling the generalized comprehension of complex subjects. However, metaphor can potentially be exploited to bypass the safety alignment mechanisms of Large Language Models (LLMs), leading to the theft of harmful knowledge. In our study, we introduce a novel attack framework that exploits the imaginative capacity of LLMs to achieve jailbreaking, the J\underline{\textbf{A}}ilbreak \underline{\textbf{V}}ia \underline{\textbf{A}}dversarial Me\underline{\textbf{TA}} -pho\underline{\textbf{R}} (\textit{AVATAR}). Specifically, to elicit the harmful response, AVATAR extracts harmful entities from a given harmful target and maps them to innocuous adversarial entities based on LLM's imagination. Then, according to these metaphors, the harmful target is nested within human-like interaction for jailbreaking adaptively. Experimental results demonstrate that AVATAR can effectively and transferablly jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. Our study exposes a security risk in LLMs from their endogenous imaginative capabilities. Furthermore, the analytical study reveals the vulnerability of LLM to adversarial metaphors and the necessity of developing defense methods against jailbreaking caused by the adversarial metaphor. \textcolor{orange}{ \textbf{Warning: This paper contains potentially harmful content from LLMs.}}

摘要: 隐喻是一种隐含的传达信息的方法，同时也使人们能够对复杂的主题进行广义的理解。然而，隐喻可能被利用来绕过大型语言模型(LLM)的安全对齐机制，从而导致有害知识的窃取。在我们的研究中，我们提出了一种新的攻击框架，该框架利用LLMS的想象能力来实现越狱，即J下划线{\extbf{A}}ilBreak\下划线{\extbf{A}}ia\下划线{\extbf{A}}dversarial Me\下划线{\extbf{TA}}-pho\下划线{\extbf{R}}(\textit{阿凡达})。具体地说，为了引发有害反应，阿凡达从给定的有害目标中提取有害实体，并基于LLM的想象力将它们映射到无害的对抗性实体。然后，根据这些隐喻，有害的目标嵌套在类似人类的互动中，以适应越狱。实验结果表明，阿凡达可以有效地、可转移地越狱LLM，并在多个高级LLM上获得最先进的攻击成功率。我们的研究揭示了LLMS的安全风险来自其内生的想象力。此外，分析研究揭示了LLM对对抗性隐喻的脆弱性，以及开发针对对抗性隐喻导致的越狱的防御方法的必要性。\extCOLOR{橙色}{\extbf{警告：此纸张包含来自LLMS的潜在有害内容。}}



## **15. Toward Robust Non-Transferable Learning: A Survey and Benchmark**

迈向稳健的不可转移学习：调查和基准 cs.LG

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2502.13593v1) [paper-pdf](http://arxiv.org/pdf/2502.13593v1)

**Authors**: Ziming Hong, Yongli Xiang, Tongliang Liu

**Abstract**: Over the past decades, researchers have primarily focused on improving the generalization abilities of models, with limited attention given to regulating such generalization. However, the ability of models to generalize to unintended data (e.g., harmful or unauthorized data) can be exploited by malicious adversaries in unforeseen ways, potentially resulting in violations of model ethics. Non-transferable learning (NTL), a task aimed at reshaping the generalization abilities of deep learning models, was proposed to address these challenges. While numerous methods have been proposed in this field, a comprehensive review of existing progress and a thorough analysis of current limitations remain lacking. In this paper, we bridge this gap by presenting the first comprehensive survey on NTL and introducing NTLBench, the first benchmark to evaluate NTL performance and robustness within a unified framework. Specifically, we first introduce the task settings, general framework, and criteria of NTL, followed by a summary of NTL approaches. Furthermore, we emphasize the often-overlooked issue of robustness against various attacks that can destroy the non-transferable mechanism established by NTL. Experiments conducted via NTLBench verify the limitations of existing NTL methods in robustness. Finally, we discuss the practical applications of NTL, along with its future directions and associated challenges.

摘要: 在过去的几十年里，研究人员主要专注于提高模型的泛化能力，而对规范这种泛化的关注很少。然而，模型概括为非预期数据(例如，有害或未经授权的数据)的能力可能会被恶意攻击者以不可预见的方式利用，可能导致违反模型道德。为应对这些挑战，提出了一项旨在重塑深度学习模型泛化能力的任务--不可迁移学习(NTL)。虽然在这一领域提出了许多方法，但仍然缺乏对现有进展的全面审查和对当前限制的透彻分析。在本文中，我们通过介绍第一次关于NTL的全面调查并引入NTLBitch来弥合这一差距，NTLBuchch是第一个在统一框架内评估NTL性能和健壮性的基准。具体地说，我们首先介绍了网络学习的任务设置、总体框架和标准，然后对网络学习的方法进行了概述。此外，我们强调了经常被忽视的问题，即对各种攻击的健壮性，这些攻击可以破坏NTL建立的不可转移机制。通过NTLBitch进行的实验验证了现有NTL方法在稳健性方面的局限性。最后，我们讨论了NTL的实际应用，以及它的未来方向和相关的挑战。



## **16. Secure and Efficient Watermarking for Latent Diffusion Models in Model Distribution Scenarios**

模型分布场景中潜在扩散模型的安全有效水印 cs.CR

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.13345v1) [paper-pdf](http://arxiv.org/pdf/2502.13345v1)

**Authors**: Liangqi Lei, Keke Gai, Jing Yu, Liehuang Zhu, Qi Wu

**Abstract**: Latent diffusion models have exhibited considerable potential in generative tasks. Watermarking is considered to be an alternative to safeguard the copyright of generative models and prevent their misuse. However, in the context of model distribution scenarios, the accessibility of models to large scale of model users brings new challenges to the security, efficiency and robustness of existing watermark solutions. To address these issues, we propose a secure and efficient watermarking solution. A new security mechanism is designed to prevent watermark leakage and watermark escape, which considers watermark randomness and watermark-model association as two constraints for mandatory watermark injection. To reduce the time cost of training the security module, watermark injection and the security mechanism are decoupled, ensuring that fine-tuning VAE only accomplishes the security mechanism without the burden of learning watermark patterns. A watermark distribution-based verification strategy is proposed to enhance the robustness against diverse attacks in the model distribution scenarios. Experimental results prove that our watermarking consistently outperforms existing six baselines on effectiveness and robustness against ten image processing attacks and adversarial attacks, while enhancing security in the distribution scenarios.

摘要: 潜在扩散模型在生成性任务中显示出相当大的潜力。水印被认为是保护生成模型的版权并防止其滥用的一种替代方案。然而，在模型分发场景下，模型对大规模模型用户的可访问性给现有水印方案的安全性、效率和稳健性带来了新的挑战。为了解决这些问题，我们提出了一种安全高效的数字水印方案。为了防止水印泄漏和水印逃逸，设计了一种新的安全机制，该机制将水印随机性和水印模型关联作为强制水印注入的两个约束。为了减少训练安全模块的时间开销，将水印注入和安全机制解耦，确保微调的VAE只完成安全机制，而不需要学习水印模式。提出了一种基于水印分发的验证策略，以增强模型分发场景对多种攻击的稳健性。实验结果表明，我们的水印在抵抗10种图像处理攻击和敌意攻击的有效性和稳健性方面始终优于现有的六条基线，同时提高了分发场景中的安全性。



## **17. UniGuardian: A Unified Defense for Detecting Prompt Injection, Backdoor Attacks and Adversarial Attacks in Large Language Models**

UniGuardian：检测大型语言模型中的即时注入、后门攻击和对抗攻击的统一防御 cs.CL

18 Pages, 8 Figures, 5 Tables, Keywords: Attack Defending, Security,  Prompt Injection, Backdoor Attacks, Adversarial Attacks, Prompt Trigger  Attacks

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.13141v1) [paper-pdf](http://arxiv.org/pdf/2502.13141v1)

**Authors**: Huawei Lin, Yingjie Lao, Tong Geng, Tan Yu, Weijie Zhao

**Abstract**: Large Language Models (LLMs) are vulnerable to attacks like prompt injection, backdoor attacks, and adversarial attacks, which manipulate prompts or models to generate harmful outputs. In this paper, departing from traditional deep learning attack paradigms, we explore their intrinsic relationship and collectively term them Prompt Trigger Attacks (PTA). This raises a key question: Can we determine if a prompt is benign or poisoned? To address this, we propose UniGuardian, the first unified defense mechanism designed to detect prompt injection, backdoor attacks, and adversarial attacks in LLMs. Additionally, we introduce a single-forward strategy to optimize the detection pipeline, enabling simultaneous attack detection and text generation within a single forward pass. Our experiments confirm that UniGuardian accurately and efficiently identifies malicious prompts in LLMs.

摘要: 大型语言模型（LLM）容易受到提示注入、后门攻击和对抗攻击等攻击，这些攻击操纵提示或模型以生成有害输出。本文脱离传统的深度学习攻击范式，探索它们的内在关系，并将它们统称为提示触发攻击（PTA）。这提出了一个关键问题：我们能否确定提示是良性的还是有毒的？为了解决这个问题，我们提出了UniGuardian，这是第一个旨在检测LLM中的即时注入、后门攻击和对抗攻击的统一防御机制。此外，我们还引入了单转发策略来优化检测管道，从而在单次转发内同时进行攻击检测和文本生成。我们的实验证实，UniGuardian可以准确有效地识别LLM中的恶意提示。



## **18. Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language Models from Jailbreaking**

推理防御：安全意识推理可以保护大型语言模型免受越狱 cs.CL

18 pages

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12970v1) [paper-pdf](http://arxiv.org/pdf/2502.12970v1)

**Authors**: Junda Zhu, Lingyong Yan, Shuaiqiang Wang, Dawei Yin, Lei Sha

**Abstract**: The reasoning abilities of Large Language Models (LLMs) have demonstrated remarkable advancement and exceptional performance across diverse domains. However, leveraging these reasoning capabilities to enhance LLM safety against adversarial attacks and jailbreak queries remains largely unexplored. To bridge this gap, we propose Reasoning-to-Defend (R2D), a novel training paradigm that integrates safety reflections of queries and responses into LLMs' generation process, unlocking a safety-aware reasoning mechanism. This approach enables self-evaluation at each reasoning step to create safety pivot tokens as indicators of the response's safety status. Furthermore, in order to improve the learning efficiency of pivot token prediction, we propose Contrastive Pivot Optimization(CPO), which enhances the model's ability to perceive the safety status of dialogues. Through this mechanism, LLMs dynamically adjust their response strategies during reasoning, significantly enhancing their defense capabilities against jailbreak attacks. Extensive experimental results demonstrate that R2D effectively mitigates various attacks and improves overall safety, highlighting the substantial potential of safety-aware reasoning in strengthening LLMs' robustness against jailbreaks.

摘要: 大型语言模型的推理能力在不同领域表现出了显著的进步和卓越的表现。然而，利用这些推理能力来增强LLM针对对手攻击和越狱查询的安全性在很大程度上仍未得到探索。为了弥补这一差距，我们提出了推理防御(R2D)，这是一种新的训练范式，将查询和响应的安全反映整合到LLMS的生成过程中，从而解锁了一种安全感知的推理机制。这种方法允许在每个推理步骤进行自我评估，以创建安全轴心令牌作为响应的安全状态的指示器。此外，为了提高枢轴标记预测的学习效率，提出了对比枢轴优化算法(CPO)，增强了模型对对话安全状态的感知能力。通过这种机制，LLMS在推理过程中动态调整响应策略，显著增强了对越狱攻击的防御能力。广泛的实验结果表明，R2D有效地缓解了各种攻击，提高了整体安全性，突出了安全意识推理在增强LLMS对越狱的健壮性方面的巨大潜力。



## **19. On the Privacy Risks of Spiking Neural Networks: A Membership Inference Analysis**

尖峰神经网络的隐私风险：成员推断分析 cs.LG

13 pages, 6 figures

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.13191v1) [paper-pdf](http://arxiv.org/pdf/2502.13191v1)

**Authors**: Junyi Guan, Abhijith Sharma, Chong Tian, Salem Lahlou

**Abstract**: Spiking Neural Networks (SNNs) are increasingly explored for their energy efficiency and robustness in real-world applications, yet their privacy risks remain largely unexamined. In this work, we investigate the susceptibility of SNNs to Membership Inference Attacks (MIAs) -- a major privacy threat where an adversary attempts to determine whether a given sample was part of the training dataset. While prior work suggests that SNNs may offer inherent robustness due to their discrete, event-driven nature, we find that its resilience diminishes as latency (T) increases. Furthermore, we introduce an input dropout strategy under black box setting, that significantly enhances membership inference in SNNs. Our findings challenge the assumption that SNNs are inherently more secure, and even though they are expected to be better, our results reveal that SNNs exhibit privacy vulnerabilities that are equally comparable to Artificial Neural Networks (ANNs). Our code is available at https://anonymous.4open.science/r/MIA_SNN-3610.

摘要: 尖峰神经网络(SNN)在实际应用中因其能量效率和稳健性而受到越来越多的研究，但其隐私风险在很大程度上仍未得到检查。在这项工作中，我们调查了SNN对成员关系推断攻击(MIA)的敏感性--MIA是一种主要的隐私威胁，攻击者试图确定给定样本是否属于训练数据集。虽然以前的工作表明，由于SNN的离散、事件驱动的性质，它可能提供固有的健壮性，但我们发现，它的弹性随着延迟(T)的增加而减弱。此外，我们在黑箱设置下引入了一种输入丢弃策略，显著增强了SNN中的成员关系推理。我们的发现挑战了SNN天生更安全的假设，尽管预计SNN会更好，但我们的结果显示，SNN表现出与人工神经网络(ANN)相当的隐私漏洞。我们的代码可以在https://anonymous.4open.science/r/MIA_SNN-3610.上找到



## **20. Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training**

铁磨铁：通过对抗训练防御机器生成文本检测中的攻击 cs.CR

Submitted to ACL 2025, Preprint, Under review

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12734v1) [paper-pdf](http://arxiv.org/pdf/2502.12734v1)

**Authors**: Yuanfan Li, Zhaohan Zhang, Chengzhengxu Li, Chao Shen, Xiaoming Liu

**Abstract**: Machine-generated Text (MGT) detection is crucial for regulating and attributing online texts. While the existing MGT detectors achieve strong performance, they remain vulnerable to simple perturbations and adversarial attacks. To build an effective defense against malicious perturbations, we view MGT detection from a threat modeling perspective, that is, analyzing the model's vulnerability from an adversary's point of view and exploring effective mitigations. To this end, we introduce an adversarial framework for training a robust MGT detector, named GREedy Adversary PromoTed DefendER (GREATER). The GREATER consists of two key components: an adversary GREATER-A and a detector GREATER-D. The GREATER-D learns to defend against the adversarial attack from GREATER-A and generalizes the defense to other attacks. GREATER-A identifies and perturbs the critical tokens in embedding space, along with greedy search and pruning to generate stealthy and disruptive adversarial examples. Besides, we update the GREATER-A and GREATER-D synchronously, encouraging the GREATER-D to generalize its defense to different attacks and varying attack intensities. Our experimental results across 9 text perturbation strategies and 5 adversarial attacks show that our GREATER-D reduces the Attack Success Rate (ASR) by 10.61% compared with SOTA defense methods while our GREATER-A is demonstrated to be more effective and efficient than SOTA attack approaches.

摘要: 机器生成文本(MGT)检测对于规范和归类在线文本至关重要。虽然现有的MGT检测器取得了很好的性能，但它们仍然容易受到简单的扰动和对抗性攻击。为了建立对恶意干扰的有效防御，我们从威胁建模的角度来看待MGT检测，即从对手的角度分析模型的脆弱性，并探索有效的缓解措施。为此，我们引入了一个对抗性框架来训练一个健壮的MGT检测器，名为贪婪对手提升防御者(Grear)。较大由两个关键组件组成：对手较大-A和检测器较大-D。大D学习防御来自大A的对抗性攻击，并将防御推广到其他攻击。Greater-A识别并扰乱嵌入空间中的关键令牌，以及贪婪的搜索和修剪，以生成隐蔽和破坏性的对抗性示例。此外，我们同步更新大A和大D，鼓励大D对不同攻击和不同攻击强度的防御进行泛化。我们在9种文本扰动策略和5种对抗性攻击上的实验结果表明，与SOTA防御方法相比，我们的大D攻击成功率(ASR)降低了10.61%，而我们的大A攻击方法被证明比SOTA攻击方法更有效和高效。



## **21. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

大型推理模型的隐藏风险：R1的安全评估 cs.CY

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12659v1) [paper-pdf](http://arxiv.org/pdf/2502.12659v1)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models, such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source R1 models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on R1 is needed. (2) The distilled reasoning model shows poorer safety performance compared to its safety-aligned base models. (3) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (4) The thinking process in R1 models pose greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.

摘要: 大型推理模型的快速发展，如OpenAI-03和DeepSeek-R1，使得复杂推理相对于非推理的大型语言模型有了显著的改进。然而，它们增强的能力，再加上DeepSeek-R1等型号的开源访问，引发了严重的安全问题，特别是它们可能被滥用的问题。在这项工作中，我们提出了这些推理模型的全面安全评估，利用已建立的安全基准来评估它们是否符合安全法规。此外，我们调查了它们对敌意攻击的敏感性，例如越狱和快速注入，以评估它们在现实世界应用程序中的健壮性。通过多方面的分析，我们发现了四个重要的发现：(1)无论是在安全基准上还是在攻击上，开源的R1型号和03-mini型号之间都存在着显著的安全差距，这表明需要在R1上做出更多的安全努力。(2)与安全对齐的基本模型相比，精炼推理模型的安全性能较差。(3)模型的推理能力越强，在回答不安全问题时可能造成的潜在危害就越大。(4)与最终答案相比，R1模型的思维过程带来了更大的安全顾虑。我们的研究为推理模型的安全含义提供了见解，并强调了在R1模型的安全性方面进一步改进的必要性，以缩小差距。



## **22. Chronus: Understanding and Securing the Cutting-Edge Industry Solutions to DRAM Read Disturbance**

Chronus：了解并保护针对动态存储器读取干扰的尖端行业解决方案 cs.CR

To appear in HPCA'25. arXiv admin note: text overlap with  arXiv:2406.19094

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12650v1) [paper-pdf](http://arxiv.org/pdf/2502.12650v1)

**Authors**: Oğuzhan Canpolat, A. Giray Yağlıkçı, Geraldo F. Oliveira, Ataberk Olgun, Nisa Bostancı, İsmail Emir Yüksel, Haocong Luo, Oğuz Ergin, Onur Mutlu

**Abstract**: We 1) present the first rigorous security, performance, energy, and cost analyses of the state-of-the-art on-DRAM-die read disturbance mitigation method, Per Row Activation Counting (PRAC) and 2) propose Chronus, a new mechanism that addresses PRAC's two major weaknesses. Our analysis shows that PRAC's system performance overhead on benign applications is non-negligible for modern DRAM chips and prohibitively large for future DRAM chips that are more vulnerable to read disturbance. We identify two weaknesses of PRAC that cause these overheads. First, PRAC increases critical DRAM access latency parameters due to the additional time required to increment activation counters. Second, PRAC performs a constant number of preventive refreshes at a time, making it vulnerable to an adversarial access pattern, known as the wave attack, and consequently requiring it to be configured for significantly smaller activation thresholds. To address PRAC's two weaknesses, we propose a new on-DRAM-die RowHammer mitigation mechanism, Chronus. Chronus 1) updates row activation counters concurrently while serving accesses by separating counters from the data and 2) prevents the wave attack by dynamically controlling the number of preventive refreshes performed. Our performance analysis shows that Chronus's system performance overhead is near-zero for modern DRAM chips and very low for future DRAM chips. Chronus outperforms three variants of PRAC and three other state-of-the-art read disturbance solutions. We discuss Chronus's and PRAC's implications for future systems and foreshadow future research directions. To aid future research, we open-source our Chronus implementation at https://github.com/CMU-SAFARI/Chronus.

摘要: 1)首次对最先进的片上DRAM读干扰抑制方法--逐行激活计数(PRAC)进行了严格的安全性、性能、能量和成本分析；2)提出了一种新的机制Chronus，解决了PRAC的两个主要弱点。我们的分析表明，PRAC在良性应用上的系统性能开销对于现代DRAM芯片来说是不可忽略的，对于更容易受到读取干扰的未来DRAM芯片来说则高得令人望而却步。我们确定了导致这些开销的PRAC的两个弱点。首先，由于增加激活计数器所需的额外时间，PRAC增加了关键的DRAM访问延迟参数。其次，PRAC一次执行固定数量的预防性刷新，使其容易受到称为WAVE攻击的敌意访问模式的攻击，因此需要将其配置为显著较小的激活阈值。为了解决PRAC的两个弱点，我们提出了一种新的DRAM芯片上RowHammer缓解机制Chronus。Chronus 1)通过将计数器与数据分离来在服务访问的同时同时更新行激活计数器，以及2)通过动态控制所执行的预防性刷新的次数来防止波攻击。我们的性能分析表明，对于现代的DRAM芯片，Chronus的系统性能开销接近于零，而对于未来的DRAM芯片，系统性能开销非常低。Chronus的性能优于PRAC的三个变种和其他三个最先进的读取干扰解决方案。我们讨论了Chronus和PRAC对未来系统的影响，并预示了未来的研究方向。为了帮助未来的研究，我们在https://github.com/CMU-SAFARI/Chronus.上开放了我们的Chronus实现



## **23. Automating Prompt Leakage Attacks on Large Language Models Using Agentic Approach**

使用统计方法自动对大型语言模型进行即时泄漏攻击 cs.CR

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12630v1) [paper-pdf](http://arxiv.org/pdf/2502.12630v1)

**Authors**: Tvrtko Sternak, Davor Runje, Dorian Granoša, Chi Wang

**Abstract**: This paper presents a novel approach to evaluating the security of large language models (LLMs) against prompt leakage-the exposure of system-level prompts or proprietary configurations. We define prompt leakage as a critical threat to secure LLM deployment and introduce a framework for testing the robustness of LLMs using agentic teams. Leveraging AG2 (formerly AutoGen), we implement a multi-agent system where cooperative agents are tasked with probing and exploiting the target LLM to elicit its prompt.   Guided by traditional definitions of security in cryptography, we further define a prompt leakage-safe system as one in which an attacker cannot distinguish between two agents: one initialized with an original prompt and the other with a prompt stripped of all sensitive information. In a safe system, the agents' outputs will be indistinguishable to the attacker, ensuring that sensitive information remains secure. This cryptographically inspired framework provides a rigorous standard for evaluating and designing secure LLMs.   This work establishes a systematic methodology for adversarial testing of prompt leakage, bridging the gap between automated threat modeling and practical LLM security.   You can find the implementation of our prompt leakage probing on GitHub.

摘要: 本文提出了一种新的方法来评估大型语言模型(LLM)的安全性，以防止系统级提示或专有配置的即时泄漏。我们将快速泄漏定义为对安全LLM部署的严重威胁，并引入了使用代理团队测试LLM健壮性的框架。利用AG2(以前的Autogen)，我们实现了一个多智能体系统，其中合作智能体的任务是探测和利用目标LLM以获得其提示。在传统密码学安全定义的指导下，我们进一步将即时泄漏安全系统定义为攻击者不能区分两个代理：一个是用原始提示初始化的，另一个是剥离所有敏感信息的提示。在安全的系统中，攻击者无法区分代理的输出，从而确保敏感信息的安全。这个受密码启发的框架为评估和设计安全的LLM提供了严格的标准。这项工作为即时泄漏的对抗性测试建立了一套系统的方法，弥合了自动威胁建模和实用LLM安全之间的差距。您可以在GitHub上找到我们的即时泄漏探测的实现。



## **24. Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinement**

通过强化学习驱动的查询细化增强大型语言模型的能力和鲁棒性 cs.CL

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2407.01461v2) [paper-pdf](http://arxiv.org/pdf/2407.01461v2)

**Authors**: Zisu Huang, Xiaohua Wang, Feiran Zhang, Zhibo Xu, Cenyuan Zhang, Qi Qian, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The capacity of large language models (LLMs) to generate honest, harmless, and helpful responses heavily relies on the quality of user prompts. However, these prompts often tend to be brief and vague, thereby significantly limiting the full potential of LLMs. Moreover, harmful prompts can be meticulously crafted and manipulated by adversaries to jailbreak LLMs, inducing them to produce potentially toxic content. To enhance the capabilities of LLMs while maintaining strong robustness against harmful jailbreak inputs, this study proposes a transferable and pluggable framework that refines user prompts before they are input into LLMs. This strategy improves the quality of the queries, empowering LLMs to generate more truthful, benign and useful responses. Specifically, a lightweight query refinement model is introduced and trained using a specially designed reinforcement learning approach that incorporates multiple objectives to enhance particular capabilities of LLMs. Extensive experiments demonstrate that the refinement model not only improves the quality of responses but also strengthens their robustness against jailbreak attacks. Code is available at: https://github.com/Huangzisu/query-refinement .

摘要: 大型语言模型(LLM)生成诚实、无害和有用的响应的能力在很大程度上取决于用户提示的质量。然而，这些提示往往简短而含糊，从而极大地限制了LLM的全部潜力。此外，有害的提示可以被对手精心制作和操纵，以越狱LLM，诱导它们产生潜在的有毒内容。为了增强LLMS的能力，同时保持对有害越狱输入的强大健壮性，本研究提出了一个可移植和可插拔的框架，在将用户提示输入到LLMS之前对其进行提炼。这一策略提高了查询的质量，使LLMS能够生成更真实、良性和有用的响应。具体地说，引入了一种轻量级查询精化模型，并使用专门设计的强化学习方法进行训练，该方法结合了多个目标来增强LLMS的特定能力。大量实验表明，改进模型不仅提高了响应的质量，而且增强了对越狱攻击的健壮性。代码可从以下网址获得：https://github.com/Huangzisu/query-refinement。



## **25. Towards Robust and Secure Embodied AI: A Survey on Vulnerabilities and Attacks**

迈向强大和安全的人工智能：关于漏洞和攻击的调查 cs.CR

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.13175v1) [paper-pdf](http://arxiv.org/pdf/2502.13175v1)

**Authors**: Wenpeng Xing, Minghao Li, Mohan Li, Meng Han

**Abstract**: Embodied AI systems, including robots and autonomous vehicles, are increasingly integrated into real-world applications, where they encounter a range of vulnerabilities stemming from both environmental and system-level factors. These vulnerabilities manifest through sensor spoofing, adversarial attacks, and failures in task and motion planning, posing significant challenges to robustness and safety. Despite the growing body of research, existing reviews rarely focus specifically on the unique safety and security challenges of embodied AI systems. Most prior work either addresses general AI vulnerabilities or focuses on isolated aspects, lacking a dedicated and unified framework tailored to embodied AI. This survey fills this critical gap by: (1) categorizing vulnerabilities specific to embodied AI into exogenous (e.g., physical attacks, cybersecurity threats) and endogenous (e.g., sensor failures, software flaws) origins; (2) systematically analyzing adversarial attack paradigms unique to embodied AI, with a focus on their impact on perception, decision-making, and embodied interaction; (3) investigating attack vectors targeting large vision-language models (LVLMs) and large language models (LLMs) within embodied systems, such as jailbreak attacks and instruction misinterpretation; (4) evaluating robustness challenges in algorithms for embodied perception, decision-making, and task planning; and (5) proposing targeted strategies to enhance the safety and reliability of embodied AI systems. By integrating these dimensions, we provide a comprehensive framework for understanding the interplay between vulnerabilities and safety in embodied AI.

摘要: 包括机器人和自动驾驶车辆在内的具体化人工智能系统正越来越多地融入现实世界的应用程序，在这些应用程序中，它们遇到了一系列源于环境和系统层面因素的漏洞。这些漏洞表现为传感器欺骗、对抗性攻击以及任务和运动规划中的失败，对健壮性和安全性构成了重大挑战。尽管研究的主体越来越多，但现有的审查很少专门关注嵌入式人工智能系统的独特安全和安保挑战。大多数以前的工作要么解决了一般的人工智能漏洞，要么专注于孤立的方面，缺乏一个专门为体现的人工智能量身定做的统一框架。本调查通过以下方式填补这一关键空白：(1)将特定于具身人工智能的漏洞分为外源性(如物理攻击、网络安全威胁)和内源性(如传感器故障、软件缺陷)来源；(2)系统分析具身人工智能特有的对抗性攻击范式，重点关注它们对感知、决策和具身交互的影响；(3)调查针对具身系统内的大视觉语言模型(LVLM)和大语言模型(LMS)的攻击向量，如越狱攻击和指令曲解；(4)评估体现感知、决策和任务规划算法中的健壮性挑战；(5)提出有针对性的策略，以提高体现人工智能系统的安全性和可靠性。通过整合这些维度，我们提供了一个全面的框架，用于理解体现的人工智能中漏洞和安全之间的相互作用。



## **26. SoK: Understanding Vulnerabilities in the Large Language Model Supply Chain**

SoK：了解大型语言模型供应链中的漏洞 cs.CR

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12497v1) [paper-pdf](http://arxiv.org/pdf/2502.12497v1)

**Authors**: Shenao Wang, Yanjie Zhao, Zhao Liu, Quanchen Zou, Haoyu Wang

**Abstract**: Large Language Models (LLMs) transform artificial intelligence, driving advancements in natural language understanding, text generation, and autonomous systems. The increasing complexity of their development and deployment introduces significant security challenges, particularly within the LLM supply chain. However, existing research primarily focuses on content safety, such as adversarial attacks, jailbreaking, and backdoor attacks, while overlooking security vulnerabilities in the underlying software systems. To address this gap, this study systematically analyzes 529 vulnerabilities reported across 75 prominent projects spanning 13 lifecycle stages. The findings show that vulnerabilities are concentrated in the application (50.3%) and model (42.7%) layers, with improper resource control (45.7%) and improper neutralization (25.1%) identified as the leading root causes. Additionally, while 56.7% of the vulnerabilities have available fixes, 8% of these patches are ineffective, resulting in recurring vulnerabilities. This study underscores the challenges of securing the LLM ecosystem and provides actionable insights to guide future research and mitigation strategies.

摘要: 大型语言模型(LLM)改变了人工智能，推动了自然语言理解、文本生成和自主系统的进步。它们的开发和部署日益复杂，带来了重大的安全挑战，特别是在LLM供应链中。然而，现有的研究主要集中在内容安全上，如对抗性攻击、越狱和后门攻击，而忽略了底层软件系统中的安全漏洞。为了解决这一差距，该研究系统地分析了跨越13个生命周期阶段的75个重要项目中报告的529个漏洞。结果显示，漏洞集中在应用层(50.3%)和模型层(42.7%)，其中资源控制不当(45.7%)和中和不当(25.1%)是主要的根本原因。此外，虽然56.7%的漏洞有可用的修复程序，但其中8%的补丁程序无效，导致漏洞反复出现。这项研究强调了确保LLM生态系统安全的挑战，并提供了可操作的见解，以指导未来的研究和缓解战略。



## **27. Independence Tests for Language Models**

语言模型的独立性测试 cs.LG

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.12292v1) [paper-pdf](http://arxiv.org/pdf/2502.12292v1)

**Authors**: Sally Zhu, Ahmed Ahmed, Rohith Kuditipudi, Percy Liang

**Abstract**: We consider the following problem: given the weights of two models, can we test whether they were trained independently -- i.e., from independent random initializations? We consider two settings: constrained and unconstrained. In the constrained setting, we make assumptions about model architecture and training and propose a family of statistical tests that yield exact p-values with respect to the null hypothesis that the models are trained from independent random initializations. These p-values are valid regardless of the composition of either model's training data; we compute them by simulating exchangeable copies of each model under our assumptions and comparing various similarity measures of weights and activations between the original two models versus these copies. We report the p-values from these tests on pairs of 21 open-weight models (210 total pairs) and correctly identify all pairs of non-independent models. Our tests remain effective even if one model was fine-tuned for many tokens. In the unconstrained setting, where we make no assumptions about training procedures, can change model architecture, and allow for adversarial evasion attacks, the previous tests no longer work. Instead, we propose a new test which matches hidden activations between two models, and which is robust to adversarial transformations and to changes in model architecture. The test can also do localized testing: identifying specific non-independent components of models. Though we no longer obtain exact p-values from this, empirically we find it behaves as one and reliably identifies non-independent models. Notably, we can use the test to identify specific parts of one model that are derived from another (e.g., how Llama 3.1-8B was pruned to initialize Llama 3.2-3B, or shared layers between Mistral-7B and StripedHyena-7B), and it is even robust to retraining individual layers of either model from scratch.

摘要: 我们考虑以下问题：给定两个模型的权重，我们能否测试它们是否独立训练--即从独立的随机初始化？我们考虑两种设置：受约束和不受约束。在约束环境下，我们对模型结构和训练进行了假设，并提出了一族统计检验，这些检验相对于模型是从独立的随机初始化训练而来的零假设产生精确的p值。无论任何一个模型的训练数据的组成如何，这些p值都是有效的；我们通过在我们的假设下模拟每个模型的可交换副本，并将原始两个模型之间的权重和激活的各种相似性度量与这些副本进行比较来计算它们。我们报告了21对公开重量模型(总共210对)的p值，并正确识别了所有非独立模型对。我们的测试仍然有效，即使一个模型针对多个令牌进行了微调。在不受约束的设置中，我们不对训练过程做出假设，可以改变模型架构，并允许对抗性逃避攻击，以前的测试不再起作用。相反，我们提出了一种新的测试，它匹配两个模型之间的隐藏激活，并且对对抗性转换和模型体系结构的变化具有健壮性。该测试还可以进行本地化测试：识别模型的特定非独立组件。尽管我们不再从中获得确切的p值，但从经验上讲，我们发现它的行为像一个人，并可靠地识别非独立模型。值得注意的是，我们可以使用测试来识别一个模型从另一个模型派生的特定部分(例如，如何修剪Llama 3.1-8B以初始化Llama 3.2-3B，或如何在Mistral-7B和StriedHyena-7B之间共享层)，甚至从头开始重新训练任一模型的各个层都是稳健的。



## **28. Quantum Byzantine Multiple Access Channels**

量子拜占庭多址通道 cs.IT

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.12047v1) [paper-pdf](http://arxiv.org/pdf/2502.12047v1)

**Authors**: Minglai Cai, Christian Deppe

**Abstract**: In communication theory, attacks like eavesdropping or jamming are typically assumed to occur at the channel level, while communication parties are expected to follow established protocols. But what happens if one of the parties turns malicious? In this work, we investigate a compelling scenario: a multiple-access channel with two transmitters and one receiver, where one transmitter deviates from the protocol and acts dishonestly. To address this challenge, we introduce the Byzantine multiple-access classical-quantum channel and derive an achievable communication rate for this adversarial setting.

摘要: 在通信理论中，窃听或干扰等攻击通常被假设发生在通道级别，而通信方预计会遵循既定的协议。但如果其中一方变得恶意会发生什么？在这项工作中，我们研究了一个引人注目的场景：具有两个发射机和一个接收机的多址通道，其中一个发射机偏离了协议并行为不诚实。为了应对这一挑战，我们引入了拜占庭式多址经典量子通道，并推导出针对这种对抗环境的可实现的通信速率。



## **29. FedEAT: A Robustness Optimization Framework for Federated LLMs**

FedEAT：联邦LLM的稳健性优化框架 cs.LG

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11863v1) [paper-pdf](http://arxiv.org/pdf/2502.11863v1)

**Authors**: Yahao Pang, Xingyuan Wu, Xiaojin Zhang, Wei Chen, Hai Jin

**Abstract**: Significant advancements have been made by Large Language Models (LLMs) in the domains of natural language understanding and automated content creation. However, they still face persistent problems, including substantial computational costs and inadequate availability of training data. The combination of Federated Learning (FL) and LLMs (federated LLMs) offers a solution by leveraging distributed data while protecting privacy, which positions it as an ideal choice for sensitive domains. However, Federated LLMs still suffer from robustness challenges, including data heterogeneity, malicious clients, and adversarial attacks, which greatly hinder their applications. We first introduce the robustness problems in federated LLMs, to address these challenges, we propose FedEAT (Federated Embedding space Adversarial Training), a novel framework that applies adversarial training in the embedding space of client LLM and employs a robust aggregation approach, specifically geometric median aggregation, to enhance the robustness of Federated LLMs. Our experiments demonstrate that FedEAT effectively improves the robustness of Federated LLMs with minimal performance loss.

摘要: 大型语言模型(LLM)在自然语言理解和自动内容创建领域取得了重大进展。然而，它们仍然面临着长期存在的问题，包括巨大的计算成本和培训数据的不足。联合学习(FL)和联合LLMS(联合LLMS)的结合提供了一种在保护隐私的同时利用分布式数据的解决方案，这将其定位为敏感领域的理想选择。然而，联邦LLMS仍然面临着健壮性挑战，包括数据异构性、恶意客户端和敌意攻击，这些都极大地阻碍了它们的应用。首先介绍了联合LLMS的健壮性问题，针对这些问题，我们提出了一种新的框架FedEAT(Federated Embedding Space Adversal Trading)，该框架将对抗性训练应用于客户端LLMS的嵌入空间，并采用一种稳健的聚集方法，特别是几何中值聚集来增强联合LLMS的健壮性。实验结果表明，FedEAT算法以最小的性能损失有效地提高了联邦LLMS的健壮性。



## **30. Practical No-box Adversarial Attacks with Training-free Hybrid Image Transformation**

使用免训练混合图像转换的实用无箱对抗攻击 cs.CV

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2203.04607v3) [paper-pdf](http://arxiv.org/pdf/2203.04607v3)

**Authors**: Qilong Zhang, Youheng Sun, Chaoning Zhang, Chaoqun Li, Xuanhan Wang, Jingkuan Song, Lianli Gao

**Abstract**: In recent years, the adversarial vulnerability of deep neural networks (DNNs) has raised increasing attention. Among all the threat models, no-box attacks are the most practical but extremely challenging since they neither rely on any knowledge of the target model or similar substitute model, nor access the dataset for training a new substitute model. Although a recent method has attempted such an attack in a loose sense, its performance is not good enough and computational overhead of training is expensive. In this paper, we move a step forward and show the existence of a \textbf{training-free} adversarial perturbation under the no-box threat model, which can be successfully used to attack different DNNs in real-time. Motivated by our observation that high-frequency component (HFC) domains in low-level features and plays a crucial role in classification, we attack an image mainly by manipulating its frequency components. Specifically, the perturbation is manipulated by suppression of the original HFC and adding of noisy HFC. We empirically and experimentally analyze the requirements of effective noisy HFC and show that it should be regionally homogeneous, repeating and dense. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our proposed no-box method. It attacks ten well-known models with a success rate of \textbf{98.13\%} on average, which outperforms state-of-the-art no-box attacks by \textbf{29.39\%}. Furthermore, our method is even competitive to mainstream transfer-based black-box attacks.

摘要: 近年来，深度神经网络(DNN)的攻击脆弱性引起了越来越多的关注。在所有的威胁模型中，非盒子攻击是最实用但极具挑战性的攻击，因为它们既不依赖于任何目标模型或类似的替代模型的任何知识，也不需要访问数据集来训练新的替代模型。虽然最近的一种方法在松散意义上尝试了这种攻击，但其性能不够好，并且训练的计算开销很高。在这篇文章中，我们进一步证明了在非盒子威胁模型下存在一个对抗性扰动，它可以成功地用来实时攻击不同的DNN。由于我们观察到高频分量(HFC)域位于低层特征并且在分类中起着关键作用，我们主要通过操纵其频率分量来攻击图像。具体地说，通过抑制原始HFC和添加噪声HFC来操纵扰动。我们从经验和实验上分析了有效的噪声HFC的要求，表明它应该是区域均匀的、重复的和密集的。在ImageNet数据集上的大量实验证明了我们提出的非盒子方法的有效性。它攻击十个著名的模型，平均成功率为\extbf{98.13\%}，比最先进的非盒子攻击的\extbf{29.39\%}要好。此外，我们的方法甚至可以与主流的基于传输的黑盒攻击相竞争。



## **31. Federated Multi-Armed Bandits Under Byzantine Attacks**

拜占庭攻击下的联邦多武装强盗 cs.LG

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2205.04134v3) [paper-pdf](http://arxiv.org/pdf/2205.04134v3)

**Authors**: Artun Saday, İlker Demirel, Yiğit Yıldırım, Cem Tekin

**Abstract**: Multi-armed bandits (MAB) is a sequential decision-making model in which the learner controls the trade-off between exploration and exploitation to maximize its cumulative reward. Federated multi-armed bandits (FMAB) is an emerging framework where a cohort of learners with heterogeneous local models play an MAB game and communicate their aggregated feedback to a server to learn a globally optimal arm. Two key hurdles in FMAB are communication-efficient learning and resilience to adversarial attacks. To address these issues, we study the FMAB problem in the presence of Byzantine clients who can send false model updates threatening the learning process. We analyze the sample complexity and the regret of $\beta$-optimal arm identification. We borrow tools from robust statistics and propose a median-of-means (MoM)-based online algorithm, Fed-MoM-UCB, to cope with Byzantine clients. In particular, we show that if the Byzantine clients constitute less than half of the cohort, the cumulative regret with respect to $\beta$-optimal arms is bounded over time with high probability, showcasing both communication efficiency and Byzantine resilience. We analyze the interplay between the algorithm parameters, a discernibility margin, regret, communication cost, and the arms' suboptimality gaps. We demonstrate Fed-MoM-UCB's effectiveness against the baselines in the presence of Byzantine attacks via experiments.

摘要: 多武装强盗(MAB)是一种序贯决策模型，在该模型中，学习者控制探索和剥削之间的权衡，以最大化其累积回报。联邦多臂强盗(FMAB)是一种新兴的框架，在这种框架中，具有不同本地模型的一群学习者玩MAB游戏，并将他们汇总的反馈传达给服务器，以学习全局最优的ARM。FMAB的两个关键障碍是高效沟通的学习和对对手攻击的适应能力。为了解决这些问题，我们在拜占庭客户端存在的情况下研究FMAB问题，这些客户端可能会发送虚假的模型更新，威胁到学习过程。分析了样本复杂度和最优ARM识别的遗憾。我们借用稳健统计的工具，提出了一种基于均值中位数(MOM)的在线算法FED-MOM-UCB，以应对拜占庭式的客户。特别地，我们证明了如果拜占庭客户端不到队列的一半，关于$\beta$-最优ARM的累积后悔是以很高的概率随时间有界的，展示了通信效率和拜占庭韧性。我们分析了算法参数、可分辨裕度、后悔、通信成本和ARM的次优差距之间的相互影响。我们通过实验证明了在拜占庭攻击存在的情况下，FED-MOM-UCB相对于基线的有效性。



## **32. Adversarially Robust CLIP Models Can Induce Better (Robust) Perceptual Metrics**

对抗稳健的CLIP模型可以诱导更好（稳健）的感知能力 cs.CV

This work has been accepted for publication in the IEEE Conference on  Secure and Trustworthy Machine Learning (SaTML). The final version will be  available on IEEE Xplore

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11725v1) [paper-pdf](http://arxiv.org/pdf/2502.11725v1)

**Authors**: Francesco Croce, Christian Schlarmann, Naman Deep Singh, Matthias Hein

**Abstract**: Measuring perceptual similarity is a key tool in computer vision. In recent years perceptual metrics based on features extracted from neural networks with large and diverse training sets, e.g. CLIP, have become popular. At the same time, the metrics extracted from features of neural networks are not adversarially robust. In this paper we show that adversarially robust CLIP models, called R-CLIP$_\textrm{F}$, obtained by unsupervised adversarial fine-tuning induce a better and adversarially robust perceptual metric that outperforms existing metrics in a zero-shot setting, and further matches the performance of state-of-the-art metrics while being robust after fine-tuning. Moreover, our perceptual metric achieves strong performance on related tasks such as robust image-to-image retrieval, which becomes especially relevant when applied to "Not Safe for Work" (NSFW) content detection and dataset filtering. While standard perceptual metrics can be easily attacked by a small perturbation completely degrading NSFW detection, our robust perceptual metric maintains high accuracy under an attack while having similar performance for unperturbed images. Finally, perceptual metrics induced by robust CLIP models have higher interpretability: feature inversion can show which images are considered similar, while text inversion can find what images are associated to a given prompt. This also allows us to visualize the very rich visual concepts learned by a CLIP model, including memorized persons, paintings and complex queries.

摘要: 感知相似性度量是计算机视觉中的一个重要工具。近年来，基于从具有大量和多样化训练集的神经网络(例如CLIP)中提取的特征的感知度量已经变得流行起来。同时，从神经网络的特征中提取的度量并不是相反的健壮性。在本文中，我们证明了通过无监督对抗性微调获得的对抗性健壮性片段模型R-Clip$tExtrm{F}$在零镜头设置下获得了比现有度量更好的对抗性健壮性感知度量，并且进一步匹配了最新度量的性能，并且在微调后仍具有健壮性。此外，我们的感知度量在相关任务中取得了良好的性能，例如稳健的图像到图像检索，当应用于不安全的工作(NSFW)内容检测和数据集过滤时，这变得特别重要。虽然标准的感知度量很容易受到微小扰动的攻击，完全降低了NSFW检测的性能，但我们的稳健感知度量在攻击下保持了高精度，而对于未受干扰的图像具有类似的性能。最后，由稳健剪辑模型得出的感知度量具有更高的可解释性：特征反转可以显示哪些图像被认为相似，而文本反转可以发现哪些图像与给定提示相关联。这也让我们可以可视化剪辑模型所学到的非常丰富的视觉概念，包括记忆中的人物、绘画和复杂的查询。



## **33. DELMAN: Dynamic Defense Against Large Language Model Jailbreaking with Model Editing**

德尔曼：通过模型编辑对大型语言模型越狱的动态防御 cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11647v1) [paper-pdf](http://arxiv.org/pdf/2502.11647v1)

**Authors**: Yi Wang, Fenghua Weng, Sibei Yang, Zhan Qin, Minlie Huang, Wenjie Wang

**Abstract**: Large Language Models (LLMs) are widely applied in decision making, but their deployment is threatened by jailbreak attacks, where adversarial users manipulate model behavior to bypass safety measures. Existing defense mechanisms, such as safety fine-tuning and model editing, either require extensive parameter modifications or lack precision, leading to performance degradation on general tasks, which is unsuitable to post-deployment safety alignment. To address these challenges, we propose DELMAN (Dynamic Editing for LLMs JAilbreak DefeNse), a novel approach leveraging direct model editing for precise, dynamic protection against jailbreak attacks. DELMAN directly updates a minimal set of relevant parameters to neutralize harmful behaviors while preserving the model's utility. To avoid triggering a safe response in benign context, we incorporate KL-divergence regularization to ensure the updated model remains consistent with the original model when processing benign queries. Experimental results demonstrate that DELMAN outperforms baseline methods in mitigating jailbreak attacks while preserving the model's utility, and adapts seamlessly to new attack instances, providing a practical and efficient solution for post-deployment model protection.

摘要: 大语言模型(LLM)在决策中被广泛应用，但它们的部署受到越狱攻击的威胁，在越狱攻击中，敌对用户操纵模型行为以绕过安全措施。现有的防御机制，如安全微调和模型编辑，要么需要大量修改参数，要么缺乏精度，导致一般任务的性能下降，不适合部署后的安全对齐。为了应对这些挑战，我们提出了Delman(用于LLMS越狱防御的动态编辑)，这是一种利用直接模型编辑来精确、动态地防御越狱攻击的新方法。Delman直接更新相关参数的最小集合，以中和有害行为，同时保持模型的实用性。为了避免在良性环境下触发安全响应，我们引入了KL-散度正则化，以确保在处理良性查询时更新后的模型与原始模型保持一致。实验结果表明，Delman在保持模型实用性的同时，在缓解越狱攻击方面优于基准方法，并能无缝适应新的攻击实例，为部署后模型防护提供了一种实用而高效的解决方案。



## **34. Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?**

LLM水印能否强大地防止未经授权的知识提炼？ cs.CL

22 pages, 12 figures, 13 tables

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11598v1) [paper-pdf](http://arxiv.org/pdf/2502.11598v1)

**Authors**: Leyi Pan, Aiwei Liu, Shiyu Huang, Yijian Lu, Xuming Hu, Lijie Wen, Irwin King, Philip S. Yu

**Abstract**: The radioactive nature of Large Language Model (LLM) watermarking enables the detection of watermarks inherited by student models when trained on the outputs of watermarked teacher models, making it a promising tool for preventing unauthorized knowledge distillation. However, the robustness of watermark radioactivity against adversarial actors remains largely unexplored. In this paper, we investigate whether student models can acquire the capabilities of teacher models through knowledge distillation while avoiding watermark inheritance. We propose two categories of watermark removal approaches: pre-distillation removal through untargeted and targeted training data paraphrasing (UP and TP), and post-distillation removal through inference-time watermark neutralization (WN). Extensive experiments across multiple model pairs, watermarking schemes and hyper-parameter settings demonstrate that both TP and WN thoroughly eliminate inherited watermarks, with WN achieving this while maintaining knowledge transfer efficiency and low computational overhead. Given the ongoing deployment of watermarking techniques in production LLMs, these findings emphasize the urgent need for more robust defense strategies. Our code is available at https://github.com/THU-BPM/Watermark-Radioactivity-Attack.

摘要: 大型语言模型(LLM)水印的放射性特性使其能够在对带水印的教师模型的输出进行训练时检测由学生模型继承的水印，使其成为防止未经授权的知识蒸馏的一种有前途的工具。然而，水印放射性对敌方行为的稳健性在很大程度上仍未被探索。本文研究了学生模型能否在避免水印继承的同时，通过知识提炼获得教师模型的能力。我们提出了两类水印去除方法：通过非目标和目标训练数据释义(UP和TP)进行蒸馏前去除和通过推理时间水印中和(WN)进行蒸馏后去除。在多个模型对、水印方案和超参数设置上的大量实验表明，TP和WN都彻底消除了继承的水印，WN在保持知识传递效率和较低的计算开销的同时实现了这一点。鉴于水印技术在生产LLM中的持续部署，这些发现强调了对更强大的防御策略的迫切需要。我们的代码可以在https://github.com/THU-BPM/Watermark-Radioactivity-Attack.上找到



## **35. Adversary-Aware DPO: Enhancing Safety Alignment in Vision Language Models via Adversarial Training**

具有对抗意识的DPO：通过对抗训练增强视觉语言模型中的安全一致性 cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11455v1) [paper-pdf](http://arxiv.org/pdf/2502.11455v1)

**Authors**: Fenghua Weng, Jian Lou, Jun Feng, Minlie Huang, Wenjie Wang

**Abstract**: Safety alignment is critical in pre-training large language models (LLMs) to generate responses aligned with human values and refuse harmful queries. Unlike LLM, the current safety alignment of VLMs is often achieved with post-hoc safety fine-tuning. However, these methods are less effective to white-box attacks. To address this, we propose $\textit{Adversary-aware DPO (ADPO)}$, a novel training framework that explicitly considers adversarial. $\textit{Adversary-aware DPO (ADPO)}$ integrates adversarial training into DPO to enhance the safety alignment of VLMs under worst-case adversarial perturbations. $\textit{ADPO}$ introduces two key components: (1) an adversarial-trained reference model that generates human-preferred responses under worst-case perturbations, and (2) an adversarial-aware DPO loss that generates winner-loser pairs accounting for adversarial distortions. By combining these innovations, $\textit{ADPO}$ ensures that VLMs remain robust and reliable even in the presence of sophisticated jailbreak attacks. Extensive experiments demonstrate that $\textit{ADPO}$ outperforms baselines in the safety alignment and general utility of VLMs.

摘要: 在预先训练大型语言模型(LLM)以生成与人类价值观一致的响应并拒绝有害查询时，安全对齐至关重要。与LLM不同，VLM当前的安全对准通常是通过事后安全微调来实现的。然而，这些方法对白盒攻击的有效性较低。为了解决这一问题，我们提出了一种新的训练框架将对抗性训练融入到DPO中，以增强VLM在最坏情况下的对抗性扰动下的安全一致性。$\textit{ADPO}$引入了两个关键组件：(1)对抗性训练的参考模型，它在最坏情况下产生人类偏好的响应；(2)对抗性感知的DPO损失，它产生考虑对抗性扭曲的赢家-输家对。通过将这些创新结合在一起，$\textit{ADPO}$确保即使在存在复杂的越狱攻击的情况下，VLM仍保持健壮和可靠。大量实验表明，在VLMS的安全性、对准和通用性方面，$\textit{ADPO}$都优于Baseline。



## **36. Dagger Behind Smile: Fool LLMs with a Happy Ending Story**

微笑背后的匕首：傻瓜LLMs，有一个幸福的结局 cs.CL

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2501.13115v2) [paper-pdf](http://arxiv.org/pdf/2501.13115v2)

**Authors**: Xurui Song, Zhixin Xie, Shuo Huai, Jiayi Kong, Jun Luo

**Abstract**: The wide adoption of Large Language Models (LLMs) has attracted significant attention from $\textit{jailbreak}$ attacks, where adversarial prompts crafted through optimization or manual design exploit LLMs to generate malicious contents. However, optimization-based attacks have limited efficiency and transferability, while existing manual designs are either easily detectable or demand intricate interactions with LLMs. In this paper, we first point out a novel perspective for jailbreak attacks: LLMs are more responsive to $\textit{positive}$ prompts. Based on this, we deploy Happy Ending Attack (HEA) to wrap up a malicious request in a scenario template involving a positive prompt formed mainly via a $\textit{happy ending}$, it thus fools LLMs into jailbreaking either immediately or at a follow-up malicious request.This has made HEA both efficient and effective, as it requires only up to two turns to fully jailbreak LLMs. Extensive experiments show that our HEA can successfully jailbreak on state-of-the-art LLMs, including GPT-4o, Llama3-70b, Gemini-pro, and achieves 88.79\% attack success rate on average. We also provide quantitative explanations for the success of HEA.

摘要: 大型语言模型(LLM)的广泛采用引起了$\textit{jailBreak}$攻击的极大关注，即通过优化或手动设计创建的敌意提示利用LLM生成恶意内容。然而，基于优化的攻击的效率和可转移性有限，而现有的手动设计要么很容易被检测到，要么需要与LLM进行复杂的交互。在这篇文章中，我们首先指出了越狱攻击的一个新视角：LLM对$\textit{积极}$提示的响应更快。在此基础上，利用HEA(Happy End End Attack)将恶意请求封装在一个场景模板中，该场景模板包含一个主要通过$\textit{Happy End}$形成的积极提示，从而欺骗LLM立即越狱或在后续恶意请求时越狱，这使得HEA既高效又有效，因为它只需要最多两个回合就可以完全越狱LLM。大量的实验表明，我们的HEA能够成功地在GPT-40、Llama3-70b、Gemini-Pro等最先进的LLMS上越狱，平均攻击成功率达到88.79%。我们还对HEA的成功提供了定量的解释。



## **37. Mimicking the Familiar: Dynamic Command Generation for Information Theft Attacks in LLM Tool-Learning System**

模仿熟悉的：LLM工具学习系统中信息窃取攻击的动态命令生成 cs.AI

15 pages, 11 figures

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11358v1) [paper-pdf](http://arxiv.org/pdf/2502.11358v1)

**Authors**: Ziyou Jiang, Mingyang Li, Guowei Yang, Junjie Wang, Yuekai Huang, Zhiyuan Chang, Qing Wang

**Abstract**: Information theft attacks pose a significant risk to Large Language Model (LLM) tool-learning systems. Adversaries can inject malicious commands through compromised tools, manipulating LLMs to send sensitive information to these tools, which leads to potential privacy breaches. However, existing attack approaches are black-box oriented and rely on static commands that cannot adapt flexibly to the changes in user queries and the invocation chain of tools. It makes malicious commands more likely to be detected by LLM and leads to attack failure. In this paper, we propose AutoCMD, a dynamic attack comment generation approach for information theft attacks in LLM tool-learning systems. Inspired by the concept of mimicking the familiar, AutoCMD is capable of inferring the information utilized by upstream tools in the toolchain through learning on open-source systems and reinforcement with target system examples, thereby generating more targeted commands for information theft. The evaluation results show that AutoCMD outperforms the baselines with +13.2% $ASR_{Theft}$, and can be generalized to new tool-learning systems to expose their information leakage risks. We also design four defense methods to effectively protect tool-learning systems from the attack.

摘要: 信息窃取攻击对大型语言模型(LLM)工具学习系统构成了重大风险。攻击者可以通过受危害的工具注入恶意命令，操纵LLM向这些工具发送敏感信息，从而导致潜在的隐私泄露。然而，现有的攻击方法是面向黑盒的，依赖于静态命令，不能灵活地适应用户查询和工具调用链的变化。它使恶意命令更容易被LLM检测到，并导致攻击失败。本文针对LLM工具学习系统中的信息窃取攻击，提出了一种动态攻击评论生成方法AutoCMD。受模仿熟悉的概念的启发，AutoCMD能够通过学习开源系统和加强目标系统示例来推断工具链中的上游工具所使用的信息，从而生成更有针对性的信息窃取命令。评估结果表明，AutoCMD的性能比基准高出13.2%$ASR{Theft}$，可以推广到新的工具学习系统，以暴露其信息泄露风险。我们还设计了四种防御方法来有效地保护工具学习系统免受攻击。



## **38. How to Backdoor Consistency Models?**

如何后门一致性模型？ cs.CR

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2410.19785v3) [paper-pdf](http://arxiv.org/pdf/2410.19785v3)

**Authors**: Chengen Wang, Murat Kantarcioglu

**Abstract**: Consistency models are a new class of models that generate images by directly mapping noise to data, allowing for one-step generation and significantly accelerating the sampling process. However, their robustness against adversarial attacks has not yet been thoroughly investigated. In this work, we conduct the first study on the vulnerability of consistency models to backdoor attacks. While previous research has explored backdoor attacks on diffusion models, those studies have primarily focused on conventional diffusion models, employing a customized backdoor training process and objective, whereas consistency models have distinct training processes and objectives. Our proposed framework demonstrates the vulnerability of consistency models to backdoor attacks. During image generation, poisoned consistency models produce images with a Fr\'echet Inception Distance (FID) comparable to that of a clean model when sampling from Gaussian noise. However, once the trigger is activated, they generate backdoor target images. We explore various trigger and target configurations to evaluate the vulnerability of consistency models, including the use of random noise as a trigger. This novel trigger is visually inconspicuous, more challenging to detect, and aligns well with the sampling process of consistency models. Across all configurations, our framework successfully compromises the consistency models while maintaining high utility and specificity. We also examine the stealthiness of our proposed attack, which is attributed to the unique properties of consistency models and the elusive nature of the Gaussian noise trigger. Our code is available at \href{https://github.com/chengenw/backdoorCM}{https://github.com/chengenw/backdoorCM}.

摘要: 一致性模型是一类新的模型，通过将噪声直接映射到数据来生成图像，允许一步生成并显著加快采样过程。然而，它们对敌意攻击的健壮性还没有得到彻底的研究。在这项工作中，我们首次研究了一致性模型对后门攻击的脆弱性。虽然以前的研究探讨了对扩散模型的后门攻击，但这些研究主要集中在传统的扩散模型上，采用定制的后门培训过程和目标，而一致性模型有不同的培训过程和目标。我们提出的框架证明了一致性模型对后门攻击的脆弱性。在图像生成过程中，当从高斯噪声中采样时，有毒一致性模型生成的图像具有与清洁模型相当的Fr回声初始距离(FID)。然而，一旦触发器被激活，它们就会生成后门目标图像。我们探索了各种触发和目标配置来评估一致性模型的脆弱性，包括使用随机噪声作为触发。这种新颖的触发器在视觉上不显眼，更难检测，并且与一致性模型的采样过程很好地一致。在所有配置中，我们的框架成功地折衷了一致性模型，同时保持了高度的实用性和专用性。我们还检查了我们提出的攻击的隐蔽性，这归因于一致性模型的独特性质和高斯噪声触发的难以捉摸的性质。我们的代码可以在\href{https://github.com/chengenw/backdoorCM}{https://github.com/chengenw/backdoorCM}.上找到



## **39. PAR-AdvGAN: Improving Adversarial Attack Capability with Progressive Auto-Regression AdvGAN**

PAR-AdvGAN：通过渐进式自回归AdvGAN提高对抗攻击能力 cs.LG

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2502.12207v1) [paper-pdf](http://arxiv.org/pdf/2502.12207v1)

**Authors**: Jiayu Zhang, Zhiyu Zhu, Xinyi Wang, Silin Liao, Zhibo Jin, Flora D. Salim, Huaming Chen

**Abstract**: Deep neural networks have demonstrated remarkable performance across various domains. However, they are vulnerable to adversarial examples, which can lead to erroneous predictions. Generative Adversarial Networks (GANs) can leverage the generators and discriminators model to quickly produce high-quality adversarial examples. Since both modules train in a competitive and simultaneous manner, GAN-based algorithms like AdvGAN can generate adversarial examples with better transferability compared to traditional methods. However, the generation of perturbations is usually limited to a single iteration, preventing these examples from fully exploiting the potential of the methods. To tackle this issue, we introduce a novel approach named Progressive Auto-Regression AdvGAN (PAR-AdvGAN). It incorporates an auto-regressive iteration mechanism within a progressive generation network to craft adversarial examples with enhanced attack capability. We thoroughly evaluate our PAR-AdvGAN method with a large-scale experiment, demonstrating its superior performance over various state-of-the-art black-box adversarial attacks, as well as the original AdvGAN.Moreover, PAR-AdvGAN significantly accelerates the adversarial example generation, i.e., achieving the speeds of up to 335.5 frames per second on Inception-v3 model, outperforming the gradient-based transferable attack algorithms. Our code is available at: https://anonymous.4open.science/r/PAR-01BF/

摘要: 深度神经网络在各个领域都表现出了卓越的性能。然而，它们很容易受到对抗性例子的影响，这可能会导致错误的预测。生成性对抗性网络(GANS)可以利用生成器和鉴别器模型快速生成高质量的对抗性实例。由于这两个模块以竞争和同步的方式进行训练，因此基于GAN的算法(如AdvGAN)可以生成比传统方法更好的可转移性的对抗性示例。然而，扰动的产生通常限于一次迭代，这使得这些例子无法充分发挥这些方法的潜力。为了解决这个问题，我们引入了一种新的方法--渐进自回归算法(PAR-AdvGAN)。它在渐进式生成网络中结合了自回归迭代机制，以创建具有增强攻击能力的对抗性示例。我们通过大规模的实验对PAR-AdvGAN方法进行了全面的评估，证明了它比各种最先进的黑盒对抗性攻击以及原始的AdvGAN方法都具有更好的性能，而且PAR-AdvGAN方法显著地加速了对抗性实例的生成，即在初始v3模型上达到了高达335.5帧/秒的速度，性能优于基于梯度的可转移攻击算法。我们的代码请访问：https://anonymous.4open.science/r/PAR-01BF/



## **40. ShieldLearner: A New Paradigm for Jailbreak Attack Defense in LLMs**

ShieldLearner：LLC越狱攻击防御的新范式 cs.CR

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2502.13162v1) [paper-pdf](http://arxiv.org/pdf/2502.13162v1)

**Authors**: Ziyi Ni, Hao Wang, Huacan Wang

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in various domains but remain vulnerable to adversarial jailbreak attacks. Existing prompt-defense strategies, including parameter-modifying and parameter-free approaches, face limitations in adaptability, interpretability, and customization, constraining their effectiveness against evolving threats. To address these challenges, we propose ShieldLearner, a novel paradigm that mimics human learning in defense. Through trial and error, it autonomously distills attack signatures into a Pattern Atlas and synthesizes defense heuristics into a Meta-analysis Framework, enabling systematic and interpretable threat detection. Furthermore, we introduce Adaptive Adversarial Augmentation to generate adversarial variations of successfully defended prompts, enabling continuous self-improvement without model retraining. In addition to standard benchmarks, we create a hard test set by curating adversarial prompts from the Wildjailbreak dataset, emphasizing more concealed malicious intent. Experimental results show that ShieldLearner achieves a significantly higher defense success rate than existing baselines on both conventional and hard test sets, while also operating with lower computational overhead, making it a practical and efficient solution for real-world adversarial defense.

摘要: 大型语言模型在各个领域取得了显着的成功，但仍然容易受到敌意越狱攻击。现有的即时防御策略，包括参数修改和非参数方法，在适应性、可解释性和定制化方面面临限制，限制了它们对抗不断变化的威胁的有效性。为了应对这些挑战，我们提出了ShieldLearner，这是一种模仿人类防御学习的新范式。通过试错，它自主地将攻击特征提取到模式Atlas中，并将防御启发式知识合成到Meta分析框架中，从而实现系统和可解释的威胁检测。此外，我们引入了自适应对抗性增强来生成成功防御提示的对抗性变体，从而能够在不需要模型重新训练的情况下持续自我改进。除了标准基准之外，我们还通过从WildjailBreak数据集中挑选敌意提示来创建一个硬测试集，强调更隐蔽的恶意意图。实验结果表明，ShieldLearner在常规测试集和硬测试集上的防御成功率明显高于现有基线，同时具有更低的计算开销，是一种实用而高效的现实世界对抗性防御解决方案。



## **41. G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems**

G-Safeguard：基于LLM的多智能体系统上的一种基于布局引导的安全视角和处理方法 cs.CR

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2502.11127v1) [paper-pdf](http://arxiv.org/pdf/2502.11127v1)

**Authors**: Shilong Wang, Guibin Zhang, Miao Yu, Guancheng Wan, Fanci Meng, Chongye Guo, Kun Wang, Yang Wang

**Abstract**: Large Language Model (LLM)-based Multi-agent Systems (MAS) have demonstrated remarkable capabilities in various complex tasks, ranging from collaborative problem-solving to autonomous decision-making. However, as these systems become increasingly integrated into critical applications, their vulnerability to adversarial attacks, misinformation propagation, and unintended behaviors have raised significant concerns. To address this challenge, we introduce G-Safeguard, a topology-guided security lens and treatment for robust LLM-MAS, which leverages graph neural networks to detect anomalies on the multi-agent utterance graph and employ topological intervention for attack remediation. Extensive experiments demonstrate that G-Safeguard: (I) exhibits significant effectiveness under various attack strategies, recovering over 40% of the performance for prompt injection; (II) is highly adaptable to diverse LLM backbones and large-scale MAS; (III) can seamlessly combine with mainstream MAS with security guarantees. The code is available at https://github.com/wslong20/G-safeguard.

摘要: 基于大型语言模型(LLM)的多智能体系统(MAS)在从协作问题求解到自主决策的各种复杂任务中表现出了卓越的能力。然而，随着这些系统越来越多地集成到关键应用程序中，它们对对手攻击、错误信息传播和意外行为的脆弱性已经引起了极大的关注。为了应对这一挑战，我们引入了G-Safe，这是一种拓扑制导的安全镜头和健壮LLM-MAS的处理方法，它利用图神经网络来检测多智能体话语图上的异常，并使用拓扑干预进行攻击补救。大量实验表明：(I)在各种攻击策略下表现出显著的有效性，可恢复40%以上的性能进行快速注入；(Ii)对不同的LLM主干和大规模MAS具有高度的适应性；(Iii)可以与主流MAS无缝结合，具有安全保障。代码可在https://github.com/wslong20/G-safeguard.上获得



## **42. Exploiting Defenses against GAN-Based Feature Inference Attacks in Federated Learning**

在联邦学习中利用防御基于GAN的特征推理攻击 cs.CR

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2004.12571v4) [paper-pdf](http://arxiv.org/pdf/2004.12571v4)

**Authors**: Xinjian Luo, Xianglong Zhang

**Abstract**: Federated learning (FL) is a decentralized model training framework that aims to merge isolated data islands while maintaining data privacy. However, recent studies have revealed that Generative Adversarial Network (GAN) based attacks can be employed in FL to learn the distribution of private datasets and reconstruct recognizable images. In this paper, we exploit defenses against GAN-based attacks in FL and propose a framework, Anti-GAN, to prevent attackers from learning the real distribution of the victim's data. The core idea of Anti-GAN is to manipulate the visual features of private training images to make them indistinguishable to human eyes even restored by attackers. Specifically, Anti-GAN projects the private dataset onto a GAN's generator and combines the generated fake images with the actual images to create the training dataset, which is then used for federated model training. The experimental results demonstrate that Anti-GAN is effective in preventing attackers from learning the distribution of private images while causing minimal harm to the accuracy of the federated model.

摘要: 联邦学习(FL)是一种去中心化的模型训练框架，旨在合并孤立的数据孤岛，同时保持数据隐私。然而，最近的研究表明，基于生成性对抗网络(GAN)的攻击可以用于FL中，以学习私有数据集的分布并重建可识别的图像。在本文中，我们在FL中利用对基于GAN的攻击的防御，并提出了一个框架--Anti-GAN，以防止攻击者了解受害者数据的真实分布。Anti-GAN的核心思想是操纵私人训练图像的视觉特征，使其即使被攻击者恢复也无法辨别人眼。具体地说，Anti-GAN将私有数据集投影到GAN的生成器上，并将生成的虚假图像与实际图像相结合来创建训练数据集，然后将其用于联合模型训练。实验结果表明，该算法能有效地防止攻击者学习私有图像的分布，同时对联邦模型的准确性造成最小的损害。



## **43. Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction**

重写越狱：发现可学习和可转移的隐性有害指令 cs.CL

21pages, 10 figures

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2502.11084v1) [paper-pdf](http://arxiv.org/pdf/2502.11084v1)

**Authors**: Yuting Huang, Chengyuan Liu, Yifeng Feng, Chao Wu, Fei Wu, Kun Kuang

**Abstract**: As Large Language Models (LLMs) are widely applied in various domains, the safety of LLMs is increasingly attracting attention to avoid their powerful capabilities being misused. Existing jailbreak methods create a forced instruction-following scenario, or search adversarial prompts with prefix or suffix tokens to achieve a specific representation manually or automatically. However, they suffer from low efficiency and explicit jailbreak patterns, far from the real deployment of mass attacks to LLMs. In this paper, we point out that simply rewriting the original instruction can achieve a jailbreak, and we find that this rewriting approach is learnable and transferable. We propose the Rewrite to Jailbreak (R2J) approach, a transferable black-box jailbreak method to attack LLMs by iteratively exploring the weakness of the LLMs and automatically improving the attacking strategy. The jailbreak is more efficient and hard to identify since no additional features are introduced. Extensive experiments and analysis demonstrate the effectiveness of R2J, and we find that the jailbreak is also transferable to multiple datasets and various types of models with only a few queries. We hope our work motivates further investigation of LLM safety.

摘要: 随着大语言模型在各个领域的广泛应用，大语言模型的安全性越来越受到人们的关注，以避免其强大的功能被滥用。现有的越狱方法创建了强制遵循指令的场景，或者搜索带有前缀或后缀令牌的对抗性提示，以手动或自动地实现特定的表示。然而，他们遭受的是低效率和明确的越狱模式，远远不能真正部署大规模攻击到LLM。在本文中，我们指出，简单地重写原始指令就可以实现越狱，并且我们发现这种重写方法是可学习的和可移植的。提出了重写越狱(R2J)方法，通过迭代挖掘LLMS的弱点并自动改进攻击策略，提出了一种可转移的黑盒越狱方法来攻击LLMS。由于没有引入其他功能，越狱更加高效，也更难识别。大量的实验和分析证明了R2J的有效性，我们发现越狱也可以只需几个查询就可以移植到多个数据集和各种类型的模型上。我们希望我们的工作能促进对LLM安全性的进一步研究。



## **44. BoT: Breaking Long Thought Processes of o1-like Large Language Models through Backdoor Attack**

BoT：通过后门攻击打破类似o1的大型语言模型的长期思维过程 cs.CL

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2502.12202v1) [paper-pdf](http://arxiv.org/pdf/2502.12202v1)

**Authors**: Zihao Zhu, Hongbao Zhang, Mingda Zhang, Ruotong Wang, Guanzong Wu, Ke Xu, Baoyuan Wu

**Abstract**: Longer thought, better performance: large language models with deep reasoning capabilities, particularly o1-like models, have demonstrated remarkable performance by generating extensive thought processes during inference. This trade-off reveals a potential vulnerability: adversaries could compromise model performance by forcing immediate responses without thought processes. To this end, in this paper, we introduce a novel attack scenario targeting the long thought processes of o1-like models and propose BoT (Break CoT), which can selectively break intrinsic reasoning mechanisms through backdoor attacks. BoT constructs poisoned datasets with designed triggers and injects backdoor by either supervised fine-tuning or direct preference optimization. When triggered, the model directly generates answers without thought processes, while maintaining normal reasoning capabilities for clean inputs. Extensive experiments on open-source o1-like models, including recent DeepSeek-R1, demonstrate that BoT nearly achieves high attack success rates while maintaining clean accuracy, highlighting the critical safety risk in current models. Furthermore, the relationship between task difficulty and helpfulness reveals a potential application for good, enabling users to customize model behavior based on task complexity. Code is available at \href{https://github.com/zihao-ai/BoT}{https://github.com/zihao-ai/BoT}.

摘要: 更长的思考，更好的性能：具有深度推理能力的大型语言模型，特别是类似o1的模型，通过在推理过程中产生广泛的思维过程，表现出了非凡的性能。这种权衡暴露了一个潜在的脆弱性：对手可能会通过强迫立即做出反应而不经过思考过程来损害模型的性能。为此，本文针对类o1模型的长思维过程引入了一种新的攻击场景，并提出了BOT(Break COT)，它可以通过后门攻击选择性地破坏原有的推理机制。BOT使用设计的触发器构建有毒数据集，并通过监督微调或直接偏好优化注入后门。当被触发时，该模型直接生成答案，而不需要思考过程，同时保持对干净输入的正常推理能力。在开源的o1类模型上的广泛实验，包括最近的DeepSeek-R1，证明了BOT在保持干净准确性的同时几乎实现了高攻击成功率，突出了当前模型中的关键安全风险。此外，任务难度和有助性之间的关系揭示了一种潜在的应用，使用户能够基于任务复杂性定制模型行为。代码可在\href{https://github.com/zihao-ai/BoT}{https://github.com/zihao-ai/BoT}.上找到



## **45. Atoxia: Red-teaming Large Language Models with Target Toxic Answers**

Atoxia：将大型语言模型与目标有毒答案进行红色合作 cs.CL

Accepted to Findings of NAACL-2025

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2408.14853v2) [paper-pdf](http://arxiv.org/pdf/2408.14853v2)

**Authors**: Yuhao Du, Zhuo Li, Pengyu Cheng, Xiang Wan, Anningzhe Gao

**Abstract**: Despite the substantial advancements in artificial intelligence, large language models (LLMs) remain being challenged by generation safety. With adversarial jailbreaking prompts, one can effortlessly induce LLMs to output harmful content, causing unexpected negative social impacts. This vulnerability highlights the necessity for robust LLM red-teaming strategies to identify and mitigate such risks before large-scale application. To detect specific types of risks, we propose a novel red-teaming method that $\textbf{A}$ttacks LLMs with $\textbf{T}$arget $\textbf{Toxi}$c $\textbf{A}$nswers ($\textbf{Atoxia}$). Given a particular harmful answer, Atoxia generates a corresponding user query and a misleading answer opening to examine the internal defects of a given LLM. The proposed attacker is trained within a reinforcement learning scheme with the LLM outputting probability of the target answer as the reward. We verify the effectiveness of our method on various red-teaming benchmarks, such as AdvBench and HH-Harmless. The empirical results demonstrate that Atoxia can successfully detect safety risks in not only open-source models but also state-of-the-art black-box models such as GPT-4o.

摘要: 尽管人工智能取得了实质性的进步，但大型语言模型(LLM)仍然受到发电安全的挑战。在对抗性越狱提示下，人们可以毫不费力地诱导LLMS输出有害内容，造成意想不到的负面社会影响。该漏洞突显了在大规模应用之前，需要强大的LLM红团队战略来识别和缓解此类风险。为了检测特定类型的风险，我们提出了一种新的红团队方法，即用$\extbf{T}$目标$\extbf{Toxi}$c$\extbf{A}$nswers($\extbf{Atoxia}$)来绑定LLMS。给定特定的有害答案，Atoxia会生成相应的用户查询和误导性答案，以检查给定LLM的内部缺陷。所提出的攻击者在强化学习方案中被训练，LLM输出目标答案的概率作为奖励。我们在不同的红队基准测试上验证了我们的方法的有效性，例如AdvBtch和HH-无害。实验结果表明，Atoxia不仅可以在开源模型中成功检测安全风险，而且可以在GPT-40等最先进的黑盒模型中成功检测到安全风险。



## **46. JPEG Inspired Deep Learning**

JPEG启发深度学习 cs.CV

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2410.07081v2) [paper-pdf](http://arxiv.org/pdf/2410.07081v2)

**Authors**: Ahmed H. Salamah, Kaixiang Zheng, Yiwen Liu, En-Hui Yang

**Abstract**: Although it is traditionally believed that lossy image compression, such as JPEG compression, has a negative impact on the performance of deep neural networks (DNNs), it is shown by recent works that well-crafted JPEG compression can actually improve the performance of deep learning (DL). Inspired by this, we propose JPEG-DL, a novel DL framework that prepends any underlying DNN architecture with a trainable JPEG compression layer. To make the quantization operation in JPEG compression trainable, a new differentiable soft quantizer is employed at the JPEG layer, and then the quantization operation and underlying DNN are jointly trained. Extensive experiments show that in comparison with the standard DL, JPEG-DL delivers significant accuracy improvements across various datasets and model architectures while enhancing robustness against adversarial attacks. Particularly, on some fine-grained image classification datasets, JPEG-DL can increase prediction accuracy by as much as 20.9%. Our code is available on https://github.com/AhmedHussKhalifa/JPEG-Inspired-DL.git.

摘要: 虽然传统上认为有损图像压缩，如JPEG压缩，会对深度神经网络(DNN)的性能产生负面影响，但最近的研究表明，精心设计的JPEG压缩实际上可以提高深度学习(DL)的性能。受此启发，我们提出了JPEG-DL，这是一种新颖的DL框架，它在任何底层的DNN体系结构中都预先加入了一个可训练的JPEG压缩层。为了使JPEG压缩中的量化操作可训练，在JPEG层使用了一种新的可微软量化器，然后将量化操作和底层的DNN进行联合训练。大量的实验表明，与标准的DL相比，JPEG-DL在不同的数据集和模型体系结构上提供了显著的准确性改进，同时增强了对对手攻击的健壮性。特别是，在一些细粒度的图像分类数据集上，JPEG-DL可以将预测精度提高20.9%。我们的代码可以在https://github.com/AhmedHussKhalifa/JPEG-Inspired-DL.git.上找到



## **47. RoMA: Robust Malware Attribution via Byte-level Adversarial Training with Global Perturbations and Adversarial Consistency Regularization**

RoMA：通过具有全局扰动和对抗一致性正规化的字节级对抗训练来实现稳健的恶意软件归因 cs.CR

11 pages, 4 figures

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2502.07492v2) [paper-pdf](http://arxiv.org/pdf/2502.07492v2)

**Authors**: Yuxia Sun, Huihong Chen, Jingcai Guo, Aoxiang Sun, Zhetao Li, Haolin Liu

**Abstract**: Attributing APT (Advanced Persistent Threat) malware to their respective groups is crucial for threat intelligence and cybersecurity. However, APT adversaries often conceal their identities, rendering attribution inherently adversarial. Existing machine learning-based attribution models, while effective, remain highly vulnerable to adversarial attacks. For example, the state-of-the-art byte-level model MalConv sees its accuracy drop from over 90% to below 2% under PGD (projected gradient descent) attacks. Existing gradient-based adversarial training techniques for malware detection or image processing were applied to malware attribution in this study, revealing that both robustness and training efficiency require significant improvement. To address this, we propose RoMA, a novel single-step adversarial training approach that integrates global perturbations to generate enhanced adversarial samples and employs adversarial consistency regularization to improve representation quality and resilience. A novel APT malware dataset named AMG18, with diverse samples and realistic class imbalances, is introduced for evaluation. Extensive experiments show that RoMA significantly outperforms seven competing methods in both adversarial robustness (e.g., achieving over 80% robust accuracy-more than twice that of the next-best method under PGD attacks) and training efficiency (e.g., more than twice as fast as the second-best method in terms of accuracy), while maintaining superior standard accuracy in non-adversarial scenarios.

摘要: 将APT(高级持续威胁)恶意软件归因于各自的组织对于威胁情报和网络安全至关重要。然而，聪明的对手往往隐藏自己的身份，使归因具有内在的对抗性。现有的基于机器学习的归因模型虽然有效，但仍然非常容易受到对手的攻击。例如，最先进的字节级模型MalConv在PGD(投影梯度下降)攻击下的准确率从90%以上下降到2%以下。将已有的基于梯度的恶意软件检测或图像处理的对抗性训练技术应用到恶意软件属性识别中，发现无论是稳健性还是训练效率都需要显著提高。为了解决这一问题，我们提出了一种新颖的单步对抗性训练方法，该方法结合全局扰动来生成增强的对抗性样本，并使用对抗性一致性正则化来提高表示质量和韧性。引入了一个新的APT恶意软件数据集AMG18，该数据集具有多样化的样本和真实的类别不平衡。大量实验表明，在对抗性稳健性(例如，在PGD攻击下达到80%以上的健壮性--是次佳方法的两倍多)和训练效率(例如，在准确率方面是次佳方法的两倍以上)方面，ROMA显著优于七种竞争方法，同时在非对抗性场景中保持了卓越的标准准确率。



## **48. MITRE ATT&CK Applications in Cybersecurity and The Way Forward**

MITRE ATT & CK在网络安全领域的应用和前进之路 cs.CR

37 pages

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2502.10825v1) [paper-pdf](http://arxiv.org/pdf/2502.10825v1)

**Authors**: Yuning Jiang, Qiaoran Meng, Feiyang Shang, Nay Oo, Le Thi Hong Minh, Hoon Wei Lim, Biplab Sikdar

**Abstract**: The MITRE ATT&CK framework is a widely adopted tool for enhancing cybersecurity, supporting threat intelligence, incident response, attack modeling, and vulnerability prioritization. This paper synthesizes research on its application across these domains by analyzing 417 peer-reviewed publications. We identify commonly used adversarial tactics, techniques, and procedures (TTPs) and examine the integration of natural language processing (NLP) and machine learning (ML) with ATT&CK to improve threat detection and response. Additionally, we explore the interoperability of ATT&CK with other frameworks, such as the Cyber Kill Chain, NIST guidelines, and STRIDE, highlighting its versatility. The paper further evaluates the framework from multiple perspectives, including its effectiveness, validation methods, and sector-specific challenges, particularly in industrial control systems (ICS) and healthcare. We conclude by discussing current limitations and proposing future research directions to enhance the applicability of ATT&CK in dynamic cybersecurity environments.

摘要: MITRE ATT&CK框架是一种被广泛采用的工具，用于增强网络安全、支持威胁情报、事件响应、攻击建模和漏洞优先排序。本文通过对417篇同行评议出版物的分析，对其在这些领域的应用研究进行了综述。我们确定了常用的对抗战术、技术和过程(TTP)，并研究了自然语言处理(NLP)和机器学习(ML)与ATT&CK的集成，以改进威胁检测和响应。此外，我们还探讨了ATT和CK与其他框架的互操作性，如Cyber Kill Chain、NIST指南和STRIDE，突出了它的多功能性。白皮书进一步从多个角度对该框架进行了评估，包括其有效性、验证方法和特定行业的挑战，特别是在工业控制系统(ICS)和医疗保健方面。最后，我们讨论了目前的局限性，并提出了未来的研究方向，以增强ATT和CK在动态网络安全环境中的适用性。



## **49. Pixel Is Not a Barrier: An Effective Evasion Attack for Pixel-Domain Diffusion Models**

像素不是障碍：像素域扩散模型的有效规避攻击 cs.CV

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2408.11810v3) [paper-pdf](http://arxiv.org/pdf/2408.11810v3)

**Authors**: Chun-Yen Shih, Li-Xuan Peng, Jia-Wei Liao, Ernie Chu, Cheng-Fu Chou, Jun-Cheng Chen

**Abstract**: Diffusion Models have emerged as powerful generative models for high-quality image synthesis, with many subsequent image editing techniques based on them. However, the ease of text-based image editing introduces significant risks, such as malicious editing for scams or intellectual property infringement. Previous works have attempted to safeguard images from diffusion-based editing by adding imperceptible perturbations. These methods are costly and specifically target prevalent Latent Diffusion Models (LDMs), while Pixel-domain Diffusion Models (PDMs) remain largely unexplored and robust against such attacks. Our work addresses this gap by proposing a novel attack framework, AtkPDM. AtkPDM is mainly composed of a feature representation attacking loss that exploits vulnerabilities in denoising UNets and a latent optimization strategy to enhance the naturalness of adversarial images. Extensive experiments demonstrate the effectiveness of our approach in attacking dominant PDM-based editing methods (e.g., SDEdit) while maintaining reasonable fidelity and robustness against common defense methods. Additionally, our framework is extensible to LDMs, achieving comparable performance to existing approaches.

摘要: 扩散模型已经成为高质量图像合成的强大生成性模型，许多后续的图像编辑技术都是基于扩散模型的。然而，基于文本的图像编辑的简便性带来了重大风险，例如用于欺诈或侵犯知识产权的恶意编辑。以前的工作试图通过添加不可察觉的扰动来保护图像免受基于扩散的编辑。这些方法昂贵且专门针对流行的潜在扩散模型(LDM)，而像素域扩散模型(PDMS)在很大程度上仍未被探索，并且对此类攻击具有健壮性。我们的工作通过提出一种新颖的攻击框架AtkPDM来解决这一问题。该算法主要由特征表示、攻击损失和潜在优化策略两部分组成，前者利用UNNet的去噪漏洞，后者增强敌方图像的自然度。大量实验表明，该方法在攻击主流的基于产品数据管理的编辑方法(如SDEDIT)的同时，对常见的防御方法保持了合理的保真度和健壮性。此外，我们的框架可扩展到LDM，实现了与现有方法相当的性能。



## **50. Robustness-aware Automatic Prompt Optimization**

具有鲁棒性的自动提示优化 cs.CL

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2412.18196v2) [paper-pdf](http://arxiv.org/pdf/2412.18196v2)

**Authors**: Zeru Shi, Zhenting Wang, Yongye Su, Weidi Luo, Hang Gao, Fan Yang, Ruixiang Tang, Yongfeng Zhang

**Abstract**: The performance of Large Language Models (LLMs) depends on the quality of prompts and the semantic and structural integrity of the input data. However, existing prompt generation methods primarily focus on well-structured input data, often neglecting the impact of perturbed inputs on prompt effectiveness. To address this limitation, we propose BATprompt (By Adversarial Training prompt), a novel method for prompt generation designed to withstand input perturbations (such as typos in the input). Inspired by adversarial training techniques, BATprompt demonstrates strong performance on a variety of perturbed tasks through a two-step process: adversarial perturbation and iterative optimization on unperturbed input via LLM. Unlike conventional adversarial attack methods, BATprompt does not need access to model parameters and gradients. Instead, BATprompt leverages the advanced reasoning, language understanding and self reflection capabilities of LLMs to simulate gradients, guiding the generation of adversarial perturbations and optimizing prompt performance. We evaluate BATprompt on multiple datasets across both language understanding and generation tasks. The results indicate that BATprompt outperforms existing prompt generation methods, delivering superior robustness and performance under diverse perturbation scenarios.

摘要: 大型语言模型(LLM)的性能取决于提示的质量以及输入数据的语义和结构完整性。然而，现有的提示生成方法主要关注结构良好的输入数据，往往忽略了扰动输入对提示效果的影响。为了解决这一局限性，我们提出了一种新的提示生成方法BATprint(通过对抗性训练提示)，该方法旨在抵抗输入扰动(如输入中的打字错误)。受到对抗性训练技术的启发，通过两步过程：对抗性扰动和通过LLM对不受扰动的输入进行迭代优化，BATprint在各种扰动任务上表现出了强大的性能。与传统的对抗性攻击方法不同，BATprint不需要访问模型参数和梯度。相反，BATprint利用LLMS的高级推理、语言理解和自我反思能力来模拟梯度，指导产生对抗性扰动并优化提示性能。我们在语言理解和生成任务的多个数据集上评估BATprint。结果表明，BATprint的性能优于现有的提示生成方法，在不同的扰动场景下都具有较好的健壮性和性能。



