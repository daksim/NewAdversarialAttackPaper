# Latest Adversarial Attack Papers
**update at 2023-12-22 18:51:02**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Open-Set: ID Card Presentation Attack Detection using Neural Transfer Style**

OPEN-SET：基于神经传递方式的身份证呈现攻击检测 cs.CV

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13993v1) [paper-pdf](http://arxiv.org/pdf/2312.13993v1)

**Authors**: Reuben Markham, Juan M. Espin, Mario Nieto-Hidalgo, Juan E. Tapia

**Abstract**: The accurate detection of ID card Presentation Attacks (PA) is becoming increasingly important due to the rising number of online/remote services that require the presentation of digital photographs of ID cards for digital onboarding or authentication. Furthermore, cybercriminals are continuously searching for innovative ways to fool authentication systems to gain unauthorized access to these services. Although advances in neural network design and training have pushed image classification to the state of the art, one of the main challenges faced by the development of fraud detection systems is the curation of representative datasets for training and evaluation. The handcrafted creation of representative presentation attack samples often requires expertise and is very time-consuming, thus an automatic process of obtaining high-quality data is highly desirable. This work explores ID card Presentation Attack Instruments (PAI) in order to improve the generation of samples with four Generative Adversarial Networks (GANs) based image translation models and analyses the effectiveness of the generated data for training fraud detection systems. Using open-source data, we show that synthetic attack presentations are an adequate complement for additional real attack presentations, where we obtain an EER performance increase of 0.63% points for print attacks and a loss of 0.29% for screen capture attacks.

摘要: 由于越来越多的在线/远程服务需要提供身份证的数字照片以进行数字登录或身份验证，因此准确检测身份证显示攻击(PA)变得越来越重要。此外，网络犯罪分子不断地寻找创新的方法来欺骗身份验证系统，以获得对这些服务的未经授权的访问。尽管神经网络设计和训练的进步将图像分类推向了最先进的水平，但欺诈检测系统的发展面临的主要挑战之一是为训练和评估挑选具有代表性的数据集。手工创建典型的表示攻击样本通常需要专业知识并且非常耗时，因此非常需要自动获取高质量数据的过程。为了改进基于四种生成对抗网络(GANS)的图像翻译模型的样本生成，并分析生成的数据用于训练欺诈检测系统的有效性，对身份证呈现攻击工具(PAI)进行了研究。使用开源数据，我们表明，合成攻击演示是对额外的真实攻击演示的足够补充，其中，对于打印攻击，我们获得了0.63%的EER性能提升，而对于屏幕捕获攻击，我们获得了0.29%的损失。



## **2. Quantum Neural Networks under Depolarization Noise: Exploring White-Box Attacks and Defenses**

去极化噪声下的量子神经网络：白盒攻击与防御探索 quant-ph

Poster at Quantum Techniques in Machine Learning (QTML) 2023

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2311.17458v2) [paper-pdf](http://arxiv.org/pdf/2311.17458v2)

**Authors**: David Winderl, Nicola Franco, Jeanette Miriam Lorenz

**Abstract**: Leveraging the unique properties of quantum mechanics, Quantum Machine Learning (QML) promises computational breakthroughs and enriched perspectives where traditional systems reach their boundaries. However, similarly to classical machine learning, QML is not immune to adversarial attacks. Quantum adversarial machine learning has become instrumental in highlighting the weak points of QML models when faced with adversarial crafted feature vectors. Diving deep into this domain, our exploration shines light on the interplay between depolarization noise and adversarial robustness. While previous results enhanced robustness from adversarial threats through depolarization noise, our findings paint a different picture. Interestingly, adding depolarization noise discontinued the effect of providing further robustness for a multi-class classification scenario. Consolidating our findings, we conducted experiments with a multi-class classifier adversarially trained on gate-based quantum simulators, further elucidating this unexpected behavior.

摘要: 利用量子力学的独特性质，量子机器学习（QML）有望在传统系统达到其边界的地方实现计算突破和丰富的观点。然而，与经典的机器学习类似，QML也不能免疫对抗性攻击。量子对抗机器学习在面对对抗性特征向量时，已经成为突出QML模型弱点的工具。深入研究这个领域，我们的探索揭示了去极化噪声和对抗鲁棒性之间的相互作用。虽然以前的结果通过去极化噪声增强了对抗性威胁的鲁棒性，但我们的研究结果描绘了一幅不同的画面。有趣的是，添加去极化噪声中断了为多类分类场景提供进一步鲁棒性的效果。为了巩固我们的研究结果，我们使用在基于门的量子模拟器上进行对抗训练的多类分类器进行了实验，进一步阐明了这种意想不到的行为。



## **3. Where and How to Attack? A Causality-Inspired Recipe for Generating Counterfactual Adversarial Examples**

向哪里进攻，如何进攻？由因果关系启发生成反事实对抗性例子的秘诀 cs.LG

Accepted by AAAI-2024

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13628v1) [paper-pdf](http://arxiv.org/pdf/2312.13628v1)

**Authors**: Ruichu Cai, Yuxuan Zhu, Jie Qiao, Zefeng Liang, Furui Liu, Zhifeng Hao

**Abstract**: Deep neural networks (DNNs) have been demonstrated to be vulnerable to well-crafted \emph{adversarial examples}, which are generated through either well-conceived $\mathcal{L}_p$-norm restricted or unrestricted attacks. Nevertheless, the majority of those approaches assume that adversaries can modify any features as they wish, and neglect the causal generating process of the data, which is unreasonable and unpractical. For instance, a modification in income would inevitably impact features like the debt-to-income ratio within a banking system. By considering the underappreciated causal generating process, first, we pinpoint the source of the vulnerability of DNNs via the lens of causality, then give theoretical results to answer \emph{where to attack}. Second, considering the consequences of the attack interventions on the current state of the examples to generate more realistic adversarial examples, we propose CADE, a framework that can generate \textbf{C}ounterfactual \textbf{AD}versarial \textbf{E}xamples to answer \emph{how to attack}. The empirical results demonstrate CADE's effectiveness, as evidenced by its competitive performance across diverse attack scenarios, including white-box, transfer-based, and random intervention attacks.

摘要: 深度神经网络(DNN)已经被证明容易受到精心设计的对手例子的攻击，这些例子是通过精心设计的数学{L}_p$-范数受限或非受限攻击而产生的。然而，这些方法中的大多数都假设对手可以随意修改任何特征，而忽略了数据的因果生成过程，这是不合理和不切实际的。例如，收入的调整将不可避免地影响银行体系内的债务收入比等特征。通过考虑被低估的因果生成过程，我们首先通过因果镜头找出DNN脆弱性的来源，然后给出理论结果来回答{攻击在哪里}。其次，考虑到攻击干预对实例的当前状态的影响，为了生成更真实的对抗性实例，我们提出了CADE框架，它可以生成\extbf{C}非事实\extbf{AD}versariative\extbf{E}样例来回答\emph{如何攻击}。实验结果证明了CADE的有效性，它在各种攻击场景中的竞争性能证明了这一点，包括白盒攻击、基于传输的攻击和随机干预攻击。



## **4. ARBiBench: Benchmarking Adversarial Robustness of Binarized Neural Networks**

ARBiBitch：衡量二值化神经网络对抗健壮性的基准 cs.CV

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13575v1) [paper-pdf](http://arxiv.org/pdf/2312.13575v1)

**Authors**: Peng Zhao, Jiehua Zhang, Bowen Peng, Longguang Wang, YingMei Wei, Yu Liu, Li Liu

**Abstract**: Network binarization exhibits great potential for deployment on resource-constrained devices due to its low computational cost. Despite the critical importance, the security of binarized neural networks (BNNs) is rarely investigated. In this paper, we present ARBiBench, a comprehensive benchmark to evaluate the robustness of BNNs against adversarial perturbations on CIFAR-10 and ImageNet. We first evaluate the robustness of seven influential BNNs on various white-box and black-box attacks. The results reveal that 1) The adversarial robustness of BNNs exhibits a completely opposite performance on the two datasets under white-box attacks. 2) BNNs consistently exhibit better adversarial robustness under black-box attacks. 3) Different BNNs exhibit certain similarities in their robustness performance. Then, we conduct experiments to analyze the adversarial robustness of BNNs based on these insights. Our research contributes to inspiring future research on enhancing the robustness of BNNs and advancing their application in real-world scenarios.

摘要: 网络二值化由于其计算成本低，在资源受限的设备上具有很大的部署潜力。尽管二值化神经网络(BNN)的安全性至关重要，但很少有人研究它的安全性。在本文中，我们提出了一种评估BNN对CIFAR-10和ImageNet上的敌意干扰的健壮性的综合基准ARBiBch。我们首先评估了七种有影响力的BNN对各种白盒和黑盒攻击的健壮性。结果表明：1)在白盒攻击下，BNN在两个数据集上的对抗健壮性表现出完全相反的表现。2)BNN在黑盒攻击下表现出更好的对抗健壮性。3)不同的BNN在鲁棒性方面表现出一定的相似性。然后，在此基础上进行实验，分析了BNN的对抗健壮性。我们的研究有助于启发未来增强BNN健壮性的研究，促进其在现实世界场景中的应用。



## **5. Adversarial Purification with the Manifold Hypothesis**

流形假设下的对抗性净化 cs.LG

Extended version of paper accepted at AAAI 2024 with supplementary  materials

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2210.14404v5) [paper-pdf](http://arxiv.org/pdf/2210.14404v5)

**Authors**: Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard Hartley, Peter Tu

**Abstract**: In this work, we formulate a novel framework for adversarial robustness using the manifold hypothesis. This framework provides sufficient conditions for defending against adversarial examples. We develop an adversarial purification method with this framework. Our method combines manifold learning with variational inference to provide adversarial robustness without the need for expensive adversarial training. Experimentally, our approach can provide adversarial robustness even if attackers are aware of the existence of the defense. In addition, our method can also serve as a test-time defense mechanism for variational autoencoders.

摘要: 在这项工作中，我们使用流形假设建立了一个新的对抗健壮性框架。这一框架为对抗对手的例子提供了充分的条件。在此框架下，我们提出了一种对抗性净化方法。我们的方法结合了流形学习和变分推理，在不需要昂贵的对抗性训练的情况下提供对抗性健壮性。在实验上，即使攻击者知道防御的存在，我们的方法也可以提供对抗的健壮性。此外，我们的方法还可以作为变分自动编码器的测试时间防御机制。



## **6. Adversarial Markov Games: On Adaptive Decision-Based Attacks and Defenses**

对抗性马尔可夫博弈：基于自适应决策的攻防 cs.AI

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13435v1) [paper-pdf](http://arxiv.org/pdf/2312.13435v1)

**Authors**: Ilias Tsingenopoulos, Vera Rimmer, Davy Preuveneers, Fabio Pierazzi, Lorenzo Cavallaro, Wouter Joosen

**Abstract**: Despite considerable efforts on making them robust, real-world ML-based systems remain vulnerable to decision based attacks, as definitive proofs of their operational robustness have so far proven intractable. The canonical approach in robustness evaluation calls for adaptive attacks, that is with complete knowledge of the defense and tailored to bypass it. In this study, we introduce a more expansive notion of being adaptive and show how attacks but also defenses can benefit by it and by learning from each other through interaction. We propose and evaluate a framework for adaptively optimizing black-box attacks and defenses against each other through the competitive game they form. To reliably measure robustness, it is important to evaluate against realistic and worst-case attacks. We thus augment both attacks and the evasive arsenal at their disposal through adaptive control, and observe that the same can be done for defenses, before we evaluate them first apart and then jointly under a multi-agent perspective. We demonstrate that active defenses, which control how the system responds, are a necessary complement to model hardening when facing decision-based attacks; then how these defenses can be circumvented by adaptive attacks, only to finally elicit active and adaptive defenses. We validate our observations through a wide theoretical and empirical investigation to confirm that AI-enabled adversaries pose a considerable threat to black-box ML-based systems, rekindling the proverbial arms race where defenses have to be AI-enabled too. Succinctly, we address the challenges posed by adaptive adversaries and develop adaptive defenses, thereby laying out effective strategies in ensuring the robustness of ML-based systems deployed in the real-world.

摘要: 尽管做出了相当大的努力来使它们健壮，但现实世界中基于ML的系统仍然容易受到基于决策的攻击，因为到目前为止，对其操作健壮性的确凿证据被证明是难以处理的。健壮性评估的规范方法要求自适应攻击，即完全了解防御并量身定做以绕过它。在这项研究中，我们引入了一个更广泛的适应性概念，并展示了攻击和防御如何从它和通过互动相互学习中受益。我们提出并评估了一个框架，用于通过形成的竞争博弈自适应地优化黑盒攻击和防御。要可靠地衡量健壮性，重要的是要针对现实和最坏情况下的攻击进行评估。因此，我们通过自适应控制来增强攻击和可供其使用的躲避武器，并观察到同样可以对防御做同样的事情，然后我们首先分开评估它们，然后在多智能体的角度下进行联合评估。我们演示了控制系统如何响应的主动防御是面对基于决策的攻击时模型强化的必要补充；然后说明如何通过自适应攻击来规避这些防御，最终只会引发主动和自适应防御。我们通过广泛的理论和经验调查验证了我们的观察结果，以确认启用AI的对手对基于黑盒ML的系统构成了相当大的威胁，重新点燃了众所周知的军备竞赛，其中防御也必须启用AI。简而言之，我们解决了适应性对手带来的挑战，并开发了适应性防御，从而制定了有效的策略，以确保部署在现实世界中的基于ML的系统的健壮性。



## **7. Universal and Transferable Adversarial Attacks on Aligned Language Models**

对对齐语言模型的通用和可转移的对抗性攻击 cs.CL

Website: http://llm-attacks.org/

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2307.15043v2) [paper-pdf](http://arxiv.org/pdf/2307.15043v2)

**Authors**: Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson

**Abstract**: Because "out-of-the-box" large language models are capable of generating a great deal of objectionable content, recent work has focused on aligning these models in an attempt to prevent undesirable generation. While there has been some success at circumventing these measures -- so-called "jailbreaks" against LLMs -- these attacks have required significant human ingenuity and are brittle in practice. In this paper, we propose a simple and effective attack method that causes aligned language models to generate objectionable behaviors. Specifically, our approach finds a suffix that, when attached to a wide range of queries for an LLM to produce objectionable content, aims to maximize the probability that the model produces an affirmative response (rather than refusing to answer). However, instead of relying on manual engineering, our approach automatically produces these adversarial suffixes by a combination of greedy and gradient-based search techniques, and also improves over past automatic prompt generation methods.   Surprisingly, we find that the adversarial prompts generated by our approach are quite transferable, including to black-box, publicly released LLMs. Specifically, we train an adversarial attack suffix on multiple prompts (i.e., queries asking for many different types of objectionable content), as well as multiple models (in our case, Vicuna-7B and 13B). When doing so, the resulting attack suffix is able to induce objectionable content in the public interfaces to ChatGPT, Bard, and Claude, as well as open source LLMs such as LLaMA-2-Chat, Pythia, Falcon, and others. In total, this work significantly advances the state-of-the-art in adversarial attacks against aligned language models, raising important questions about how such systems can be prevented from producing objectionable information. Code is available at github.com/llm-attacks/llm-attacks.

摘要: 由于“开箱即用”的大型语言模型能够生成大量令人反感的内容，最近的工作重点是调整这些模型，以试图防止不必要的生成。虽然在规避这些措施方面取得了一些成功--即所谓的针对LLMS的“越狱”--但这些攻击需要大量的人类智慧，而且在实践中是脆弱的。在本文中，我们提出了一种简单有效的攻击方法，使对齐的语言模型产生令人反感的行为。具体地说，我们的方法找到了一个后缀，当附加到LLM的广泛查询中以产生令人反感的内容时，旨在最大化该模型产生肯定响应(而不是拒绝回答)的概率。然而，我们的方法不依赖于人工设计，而是通过贪婪和基于梯度的搜索技术相结合来自动生成这些对抗性后缀，并且改进了过去的自动提示生成方法。令人惊讶的是，我们发现我们的方法生成的对抗性提示是相当可转移的，包括到黑盒，公开发布的LLM。具体地说，我们对多个提示(即，要求许多不同类型的不良内容的查询)以及多个模型(在我们的案例中，Vicuna-7B和13B)训练对抗性攻击后缀。这样做时，生成的攻击后缀能够在ChatGPT、Bard和Claude的公共接口以及开源LLM(如llama-2-chat、Pythia、Falcon和其他)中诱导令人反感的内容。总而言之，这项工作极大地推进了针对对齐语言模型的对抗性攻击的最新水平，提出了如何防止此类系统产生令人反感的信息的重要问题。代码可在githorb.com/llm-Attages/llm-Attack上找到。



## **8. On the complexity of sabotage games for network security**

论破坏游戏对网络安全的复杂性 cs.CC

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13132v1) [paper-pdf](http://arxiv.org/pdf/2312.13132v1)

**Authors**: Dhananjay Raju, Georgios Bakirtzis, Ufuk Topcu

**Abstract**: Securing dynamic networks against adversarial actions is challenging because of the need to anticipate and counter strategic disruptions by adversarial entities within complex network structures. Traditional game-theoretic models, while insightful, often fail to model the unpredictability and constraints of real-world threat assessment scenarios. We refine sabotage games to reflect the realistic limitations of the saboteur and the network operator. By transforming sabotage games into reachability problems, our approach allows applying existing computational solutions to model realistic restrictions on attackers and defenders within the game. Modifying sabotage games into dynamic network security problems successfully captures the nuanced interplay of strategy and uncertainty in dynamic network security. Theoretically, we extend sabotage games to model network security contexts and thoroughly explore if the additional restrictions raise their computational complexity, often the bottleneck of game theory in practical contexts. Practically, this research sets the stage for actionable insights for developing robust defense mechanisms by understanding what risks to mitigate in dynamically changing networks under threat.

摘要: 确保动态网络不受敌对行动的影响是具有挑战性的，因为需要预测和应对复杂网络结构中敌对实体的战略中断。传统的博弈论模型虽然有洞察力，但往往无法对现实世界威胁评估情景的不可预测性和约束进行建模。我们改进了破坏游戏，以反映破坏者和网络运营商的现实限制。通过将破坏游戏转换为可达性问题，我们的方法允许应用现有的计算解决方案来模拟游戏中对攻击者和防御者的现实限制。将破坏博弈转化为动态网络安全问题，成功地捕捉到了动态网络安全中战略和不确定性的微妙相互作用。从理论上讲，我们将破坏游戏扩展到对网络安全环境进行建模，并深入探索额外的限制是否会增加其计算复杂性，这在实际环境中往往是博弈论的瓶颈。实际上，这项研究通过了解在受到威胁的动态变化的网络中需要缓解哪些风险，为开发强大的防御机制提供了可行的见解。



## **9. Prometheus: Infrastructure Security Posture Analysis with AI-generated Attack Graphs**

普罗米修斯：使用人工智能生成的攻击图进行基础设施安全态势分析 cs.CR

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13119v1) [paper-pdf](http://arxiv.org/pdf/2312.13119v1)

**Authors**: Xin Jin, Charalampos Katsis, Fan Sang, Jiahao Sun, Elisa Bertino, Ramana Rao Kompella, Ashish Kundu

**Abstract**: The rampant occurrence of cybersecurity breaches imposes substantial limitations on the progress of network infrastructures, leading to compromised data, financial losses, potential harm to individuals, and disruptions in essential services. The current security landscape demands the urgent development of a holistic security assessment solution that encompasses vulnerability analysis and investigates the potential exploitation of these vulnerabilities as attack paths. In this paper, we propose Prometheus, an advanced system designed to provide a detailed analysis of the security posture of computing infrastructures. Using user-provided information, such as device details and software versions, Prometheus performs a comprehensive security assessment. This assessment includes identifying associated vulnerabilities and constructing potential attack graphs that adversaries can exploit. Furthermore, Prometheus evaluates the exploitability of these attack paths and quantifies the overall security posture through a scoring mechanism. The system takes a holistic approach by analyzing security layers encompassing hardware, system, network, and cryptography. Furthermore, Prometheus delves into the interconnections between these layers, exploring how vulnerabilities in one layer can be leveraged to exploit vulnerabilities in others. In this paper, we present the end-to-end pipeline implemented in Prometheus, showcasing the systematic approach adopted for conducting this thorough security analysis.

摘要: 网络安全漏洞的猖獗发生对网络基础设施的进展施加了很大限制，导致数据泄露、经济损失、对个人的潜在伤害以及基本服务中断。当前的安全形势要求迫切开发一种全面的安全评估解决方案，其中包括漏洞分析，并调查利用这些漏洞作为攻击途径的可能性。在本文中，我们提出了普罗米修斯，这是一个先进的系统，旨在提供详细的分析计算基础设施的安全态势。使用用户提供的信息，如设备详细信息和软件版本，普罗米修斯进行全面的安全评估。该评估包括识别相关漏洞和构建潜在的攻击图，以供攻击者利用。此外，普罗米修斯还评估了这些攻击路径的可利用性，并通过评分机制量化了总体安全态势。该系统采取整体方法，分析包括硬件、系统、网络和加密在内的安全层。此外，普罗米修斯深入研究了这些层之间的相互联系，探索如何利用一个层中的漏洞来利用其他层中的漏洞。在这篇文章中，我们介绍了在普罗米修斯中实现的端到端管道，展示了为进行这种彻底的安全分析而采用的系统方法。



## **10. LRS: Enhancing Adversarial Transferability through Lipschitz Regularized Surrogate**

LRS：通过Lipschitz正则化代理提高对手的可转移性 cs.LG

AAAI 2024

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.13118v1) [paper-pdf](http://arxiv.org/pdf/2312.13118v1)

**Authors**: Tao Wu, Tie Luo, Donald C. Wunsch

**Abstract**: The transferability of adversarial examples is of central importance to transfer-based black-box adversarial attacks. Previous works for generating transferable adversarial examples focus on attacking \emph{given} pretrained surrogate models while the connections between surrogate models and adversarial trasferability have been overlooked. In this paper, we propose {\em Lipschitz Regularized Surrogate} (LRS) for transfer-based black-box attacks, a novel approach that transforms surrogate models towards favorable adversarial transferability. Using such transformed surrogate models, any existing transfer-based black-box attack can run without any change, yet achieving much better performance. Specifically, we impose Lipschitz regularization on the loss landscape of surrogate models to enable a smoother and more controlled optimization process for generating more transferable adversarial examples. In addition, this paper also sheds light on the connection between the inner properties of surrogate models and adversarial transferability, where three factors are identified: smaller local Lipschitz constant, smoother loss landscape, and stronger adversarial robustness. We evaluate our proposed LRS approach by attacking state-of-the-art standard deep neural networks and defense models. The results demonstrate significant improvement on the attack success rates and transferability. Our code is available at https://github.com/TrustAIoT/LRS.

摘要: 对抗性例子的可转移性对于基于转移的黑盒对抗性攻击是至关重要的。以往关于生成可传递对抗实例的工作主要集中在攻击预先训练好的代理模型上，而忽略了代理模型与对抗传递能力之间的联系。针对基于转移的黑盒攻击，提出了一种将代理模型转化为有利的对抗性转移的新方法--LRS。使用这种转换的代理模型，任何现有的基于传输的黑盒攻击都可以在不做任何更改的情况下运行，但获得了更好的性能。具体地说，我们将Lipschitz正则化应用于代理模型的损失图景，以实现更平滑和更可控的优化过程，从而生成更多可转移的对抗性例子。此外，本文还揭示了代理模型的内在性质与对抗转移之间的关系，其中确定了三个因素：较小的局部Lipschitz常数、更平滑的损失图景和更强的对抗稳健性。我们通过攻击最先进的标准深度神经网络和防御模型来评估我们提出的LRS方法。结果表明，在攻击成功率和可转移性方面都有显著的提高。我们的代码可以在https://github.com/TrustAIoT/LRS.上找到



## **11. PGN: A perturbation generation network against deep reinforcement learning**

PGN：一种抗深度强化学习的扰动生成网络 cs.LG

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.12904v1) [paper-pdf](http://arxiv.org/pdf/2312.12904v1)

**Authors**: Xiangjuan Li, Feifan Li, Yang Li, Quan Pan

**Abstract**: Deep reinforcement learning has advanced greatly and applied in many areas. In this paper, we explore the vulnerability of deep reinforcement learning by proposing a novel generative model for creating effective adversarial examples to attack the agent. Our proposed model can achieve both targeted attacks and untargeted attacks. Considering the specificity of deep reinforcement learning, we propose the action consistency ratio as a measure of stealthiness, and a new measurement index of effectiveness and stealthiness. Experiment results show that our method can ensure the effectiveness and stealthiness of attack compared with other algorithms. Moreover, our methods are considerably faster and thus can achieve rapid and efficient verification of the vulnerability of deep reinforcement learning.

摘要: 深度强化学习已经取得了很大的进展，并在许多领域得到了应用。在本文中，我们通过提出一种新的生成模型来探索深度强化学习的脆弱性，该模型用于创建有效的对抗性示例来攻击代理。该模型既可以实现定向攻击，也可以实现非定向攻击。考虑到深度强化学习的特殊性，我们提出了动作一致性比率作为隐蔽性的度量，并提出了一种新的度量有效性和隐蔽性的指标。实验结果表明，与其他算法相比，该方法能够保证攻击的有效性和隐蔽性。此外，我们的方法具有相当快的速度，因此可以快速有效地验证深度强化学习的脆弱性。



## **12. SAAM: Stealthy Adversarial Attack on Monocular Depth Estimation**

SAAM：对单目深度估计的隐形对抗攻击 cs.CV

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2308.03108v2) [paper-pdf](http://arxiv.org/pdf/2308.03108v2)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Bassem Ouni, Muhammad Shafique

**Abstract**: In this paper, we investigate the vulnerability of MDE to adversarial patches. We propose a novel \underline{S}tealthy \underline{A}dversarial \underline{A}ttacks on \underline{M}DE (SAAM) that compromises MDE by either corrupting the estimated distance or causing an object to seamlessly blend into its surroundings. Our experiments, demonstrate that the designed stealthy patch successfully causes a DNN-based MDE to misestimate the depth of objects. In fact, our proposed adversarial patch achieves a significant 60\% depth error with 99\% ratio of the affected region. Importantly, despite its adversarial nature, the patch maintains a naturalistic appearance, making it inconspicuous to human observers. We believe that this work sheds light on the threat of adversarial attacks in the context of MDE on edge devices. We hope it raises awareness within the community about the potential real-life harm of such attacks and encourages further research into developing more robust and adaptive defense mechanisms.

摘要: 在本文中，我们研究了MDE对敌意补丁的脆弱性。我们提出了一种新的基于{M}DE(SAAM)的{A}大头针，它破坏了估计的距离或使物体无缝地融入其周围，从而折衷了MDE。我们的实验表明，所设计的隐身补丁成功地导致了基于DNN的MDE错误估计目标的深度。事实上，我们提出的对抗性补丁获得了显著的60%的深度误差，其受影响区域的比率为99%。重要的是，尽管它具有对抗性，但它保持了一种自然主义的外观，使它对人类观察者来说并不引人注目。我们相信，这项工作有助于揭示边缘设备上的MDE环境中的对抗性攻击威胁。我们希望它能提高社区对此类攻击的潜在现实危害的认识，并鼓励进一步研究开发更强大和适应性更强的防御机制。



## **13. Mutual-modality Adversarial Attack with Semantic Perturbation**

基于语义扰动的交互式对抗攻击 cs.CV

Accepted by AAAI2024

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2312.12768v1) [paper-pdf](http://arxiv.org/pdf/2312.12768v1)

**Authors**: Jingwen Ye, Ruonan Yu, Songhua Liu, Xinchao Wang

**Abstract**: Adversarial attacks constitute a notable threat to machine learning systems, given their potential to induce erroneous predictions and classifications. However, within real-world contexts, the essential specifics of the deployed model are frequently treated as a black box, consequently mitigating the vulnerability to such attacks. Thus, enhancing the transferability of the adversarial samples has become a crucial area of research, which heavily relies on selecting appropriate surrogate models. To address this challenge, we propose a novel approach that generates adversarial attacks in a mutual-modality optimization scheme. Our approach is accomplished by leveraging the pre-trained CLIP model. Firstly, we conduct a visual attack on the clean image that causes semantic perturbations on the aligned embedding space with the other textual modality. Then, we apply the corresponding defense on the textual modality by updating the prompts, which forces the re-matching on the perturbed embedding space. Finally, to enhance the attack transferability, we utilize the iterative training strategy on the visual attack and the textual defense, where the two processes optimize from each other. We evaluate our approach on several benchmark datasets and demonstrate that our mutual-modal attack strategy can effectively produce high-transferable attacks, which are stable regardless of the target networks. Our approach outperforms state-of-the-art attack methods and can be readily deployed as a plug-and-play solution.

摘要: 对抗性攻击对机器学习系统构成了显著的威胁，因为它们有可能导致错误的预测和分类。然而，在现实环境中，部署的模型的基本细节经常被视为黑匣子，从而降低了对此类攻击的脆弱性。因此，提高对抗性样本的可转移性已成为一个重要的研究领域，这在很大程度上依赖于选择合适的代理模型。为了应对这一挑战，我们提出了一种新的方法，该方法在交互通道优化方案中产生对抗性攻击。我们的方法是通过利用预先训练的剪辑模型来实现的。首先，我们对干净的图像进行视觉攻击，导致与其他文本通道对齐的嵌入空间上的语义扰动。然后，我们通过更新提示来对语篇情态进行相应的防御，强制在扰动的嵌入空间上进行重新匹配。最后，为了增强攻击的可转移性，我们在视觉攻击和文本防御上采用了迭代训练策略，这两个过程相互优化。我们在几个基准数据集上对我们的方法进行了评估，并证明了我们的互模式攻击策略可以有效地产生高度可转移的攻击，并且无论目标网络是什么，都是稳定的。我们的方法优于最先进的攻击方法，并且可以作为即插即用解决方案轻松部署。



## **14. Trust, but Verify: Robust Image Segmentation using Deep Learning**

信任，但要验证：使用深度学习的稳健图像分割 cs.CV

5 Pages, 8 Figures, conference

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2310.16999v3) [paper-pdf](http://arxiv.org/pdf/2310.16999v3)

**Authors**: Fahim Ahmed Zaman, Xiaodong Wu, Weiyu Xu, Milan Sonka, Raghuraman Mudumbai

**Abstract**: We describe a method for verifying the output of a deep neural network for medical image segmentation that is robust to several classes of random as well as worst-case perturbations i.e. adversarial attacks. This method is based on a general approach recently developed by the authors called "Trust, but Verify" wherein an auxiliary verification network produces predictions about certain masked features in the input image using the segmentation as an input. A well-designed auxiliary network will produce high-quality predictions when the input segmentations are accurate, but will produce low-quality predictions when the segmentations are incorrect. Checking the predictions of such a network with the original image allows us to detect bad segmentations. However, to ensure the verification method is truly robust, we need a method for checking the quality of the predictions that does not itself rely on a black-box neural network. Indeed, we show that previous methods for segmentation evaluation that do use deep neural regression networks are vulnerable to false negatives i.e. can inaccurately label bad segmentations as good. We describe the design of a verification network that avoids such vulnerability and present results to demonstrate its robustness compared to previous methods.

摘要: 我们描述了一种用于医学图像分割的深度神经网络输出的验证方法，该方法对几类随机和最坏情况的扰动，即对抗性攻击具有鲁棒性。该方法基于作者最近开发的一种被称为“信任，但验证”的通用方法，其中辅助验证网络使用分割作为输入来产生关于输入图像中的某些被屏蔽特征的预测。一个设计良好的辅助网络在输入分割准确时会产生高质量的预测，但当分割不正确时会产生低质量的预测。用原始图像检查这种网络的预测可以让我们检测到错误的分割。然而，为了确保验证方法真正稳健，我们需要一种方法来检查预测的质量，该方法本身不依赖于黑盒神经网络。事实上，我们表明，以前使用深度神经回归网络的分割评估方法很容易出现假阴性，即可能不准确地将不良分割标记为良好分割。我们描述了一个避免这种漏洞的验证网络的设计，并给出了与以前方法相比的结果来证明它的健壮性。



## **15. ReRoGCRL: Representation-based Robustness in Goal-Conditioned Reinforcement Learning**

ReRoGCRL：目标条件强化学习中基于表示的稳健性 cs.LG

This paper has been accepted in AAAI24  (https://aaai.org/aaai-conference/)

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2312.07392v3) [paper-pdf](http://arxiv.org/pdf/2312.07392v3)

**Authors**: Xiangyu Yin, Sihao Wu, Jiaxu Liu, Meng Fang, Xingyu Zhao, Xiaowei Huang, Wenjie Ruan

**Abstract**: While Goal-Conditioned Reinforcement Learning (GCRL) has gained attention, its algorithmic robustness against adversarial perturbations remains unexplored. The attacks and robust representation training methods that are designed for traditional RL become less effective when applied to GCRL. To address this challenge, we first propose the Semi-Contrastive Representation attack, a novel approach inspired by the adversarial contrastive attack. Unlike existing attacks in RL, it only necessitates information from the policy function and can be seamlessly implemented during deployment. Then, to mitigate the vulnerability of existing GCRL algorithms, we introduce Adversarial Representation Tactics, which combines Semi-Contrastive Adversarial Augmentation with Sensitivity-Aware Regularizer to improve the adversarial robustness of the underlying RL agent against various types of perturbations. Extensive experiments validate the superior performance of our attack and defence methods across multiple state-of-the-art GCRL algorithms. Our tool ReRoGCRL is available at https://github.com/TrustAI/ReRoGCRL.

摘要: 虽然目标条件强化学习(GCRL)已经引起了人们的关注，但它对对抗扰动的算法健壮性还没有得到探索。针对传统RL设计的攻击和稳健表示训练方法在应用于GCRL时变得不那么有效。为了应对这一挑战，我们首先提出了半对比表征攻击，这是一种受对抗性对比攻击启发的新方法。与RL中现有的攻击不同，它只需要来自策略功能的信息，并且可以在部署期间无缝实施。然后，为了缓解现有GCRL算法的脆弱性，我们引入了对抗性表示策略，将半对比对抗性增强和敏感度感知正则化相结合，以提高底层RL代理对各种类型扰动的对抗性健壮性。广泛的实验验证了我们的攻击和防御方法在多种最先进的GCRL算法上的卓越性能。我们的工具ReRoGCRL可在https://github.com/TrustAI/ReRoGCRL.上获得



## **16. Trust, But Verify: A Survey of Randomized Smoothing Techniques**

信任，但验证：随机化平滑技术综述 cs.LG

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2312.12608v1) [paper-pdf](http://arxiv.org/pdf/2312.12608v1)

**Authors**: Anupriya Kumari, Devansh Bhardwaj, Sukrit Jindal, Sarthak Gupta

**Abstract**: Machine learning models have demonstrated remarkable success across diverse domains but remain vulnerable to adversarial attacks. Empirical defence mechanisms often fall short, as new attacks constantly emerge, rendering existing defences obsolete. A paradigm shift from empirical defences to certification-based defences has been observed in response. Randomized smoothing has emerged as a promising technique among notable advancements. This study reviews the theoretical foundations, empirical effectiveness, and applications of randomized smoothing in verifying machine learning classifiers. We provide an in-depth exploration of the fundamental concepts underlying randomized smoothing, highlighting its theoretical guarantees in certifying robustness against adversarial perturbations. Additionally, we discuss the challenges of existing methodologies and offer insightful perspectives on potential solutions. This paper is novel in its attempt to systemise the existing knowledge in the context of randomized smoothing.

摘要: 机器学习模型在不同领域取得了显著的成功，但仍然容易受到对抗性攻击。经验性的防御机制往往不足，因为新的攻击不断出现，使现有的防御措施过时。在回应中，观察到了从经验辩护到基于认证的辩护的范式转变。随机平滑已成为一个有前途的技术显着的进步。本文综述了随机平滑在机器学习分类器验证中的理论基础、实证有效性和应用。我们深入探讨了随机平滑的基本概念，强调了其在证明对抗性扰动的鲁棒性方面的理论保证。此外，我们还讨论了现有方法的挑战，并对潜在的解决方案提供了有见地的观点。本文是新颖的，它试图系统化现有的知识的背景下，随机平滑。



## **17. You Cannot Escape Me: Detecting Evasions of SIEM Rules in Enterprise Networks**

你不能逃避我：在企业网络中检测对SIEM规则的逃避 cs.CR

To be published in Proceedings of the 33rd USENIX Security Symposium  (USENIX Security 2024)

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2311.10197v2) [paper-pdf](http://arxiv.org/pdf/2311.10197v2)

**Authors**: Rafael Uetz, Marco Herzog, Louis Hackländer, Simon Schwarz, Martin Henze

**Abstract**: Cyberattacks have grown into a major risk for organizations, with common consequences being data theft, sabotage, and extortion. Since preventive measures do not suffice to repel attacks, timely detection of successful intruders is crucial to stop them from reaching their final goals. For this purpose, many organizations utilize Security Information and Event Management (SIEM) systems to centrally collect security-related events and scan them for attack indicators using expert-written detection rules. However, as we show by analyzing a set of widespread SIEM detection rules, adversaries can evade almost half of them easily, allowing them to perform common malicious actions within an enterprise network without being detected. To remedy these critical detection blind spots, we propose the idea of adaptive misuse detection, which utilizes machine learning to compare incoming events to SIEM rules on the one hand and known-benign events on the other hand to discover successful evasions. Based on this idea, we present AMIDES, an open-source proof-of-concept adaptive misuse detection system. Using four weeks of SIEM events from a large enterprise network and more than 500 hand-crafted evasions, we show that AMIDES successfully detects a majority of these evasions without any false alerts. In addition, AMIDES eases alert analysis by assessing which rules were evaded. Its computational efficiency qualifies AMIDES for real-world operation and hence enables organizations to significantly reduce detection blind spots with moderate effort.

摘要: 网络攻击已经成为组织的主要风险，常见的后果是数据被盗、破坏和敲诈勒索。由于预防措施不足以击退攻击，及时发现成功的入侵者对于阻止他们实现最终目标至关重要。为此，许多组织利用安全信息和事件管理(SIEM)系统集中收集与安全相关的事件，并使用专家编写的检测规则扫描它们的攻击指标。然而，正如我们通过分析一组广泛使用的SIEM检测规则所表明的那样，攻击者可以很容易地规避几乎一半的规则，使他们能够在不被检测到的情况下在企业网络内执行常见的恶意操作。为了弥补这些关键的检测盲点，我们提出了自适应误用检测的思想，一方面利用机器学习将传入事件与SIEM规则进行比较，另一方面利用已知良性事件来发现成功的规避。基于这一思想，我们提出了一个开源的概念验证自适应误用检测系统AMIDES。使用来自大型企业网络的四周SIEM事件和500多个手工创建的规避，我们表明AMIDES成功检测到了大多数此类规避，而没有任何错误警报。此外，AMIDES通过评估哪些规则被规避来简化警报分析。它的计算效率使AMADS有资格在现实世界中运行，因此使组织能够以适度的努力显著减少检测盲点。



## **18. Comparing the robustness of modern no-reference image- and video-quality metrics to adversarial attacks**

比较现代无参考图像和视频质量指标对敌方攻击的稳健性 cs.CV

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2310.06958v2) [paper-pdf](http://arxiv.org/pdf/2310.06958v2)

**Authors**: Anastasia Antsiferova, Khaled Abud, Aleksandr Gushchin, Ekaterina Shumitskaya, Sergey Lavrushkin, Dmitriy Vatolin

**Abstract**: Nowadays neural-network-based image- and video-quality metrics show better performance compared to traditional methods. However, they also became more vulnerable to adversarial attacks that increase metrics' scores without improving visual quality. The existing benchmarks of quality metrics compare their performance in terms of correlation with subjective quality and calculation time. However, the adversarial robustness of image-quality metrics is also an area worth researching. In this paper, we analyse modern metrics' robustness to different adversarial attacks. We adopted adversarial attacks from computer vision tasks and compared attacks' efficiency against 15 no-reference image/video-quality metrics. Some metrics showed high resistance to adversarial attacks which makes their usage in benchmarks safer than vulnerable metrics. The benchmark accepts new metrics submissions for researchers who want to make their metrics more robust to attacks or to find such metrics for their needs. Try our benchmark using pip install robustness-benchmark.

摘要: 如今，基于神经网络的图像和视频质量度量显示出比传统方法更好的性能。然而，它们也变得更容易受到对抗性攻击，这些攻击增加了指标的分数，但没有改善视觉质量。现有的质量指标基准在与主观质量和计算时间的相关性方面比较它们的表现。然而，图像质量指标的对抗稳健性也是一个值得研究的领域。在本文中，我们分析了现代度量对不同对手攻击的稳健性。我们采用了来自计算机视觉任务的对抗性攻击，并将攻击效率与15个无参考图像/视频质量指标进行了比较。一些指标表现出对敌意攻击的高度抵抗力，这使得它们在基准中的使用比易受攻击的指标更安全。该基准接受新的指标提交给希望使其指标更具抗攻击能力或找到符合其需求的此类指标的研究人员。使用pip安装健壮性基准测试我们的基准测试。



## **19. Tensor Train Decomposition for Adversarial Attacks on Computer Vision Models**

计算机视觉模型对抗性攻击的张量训练分解 math.NA

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2312.12556v1) [paper-pdf](http://arxiv.org/pdf/2312.12556v1)

**Authors**: Andrei Chertkov, Ivan Oseledets

**Abstract**: Deep neural networks (DNNs) are widely used today, but they are vulnerable to adversarial attacks. To develop effective methods of defense, it is important to understand the potential weak spots of DNNs. Often attacks are organized taking into account the architecture of models (white-box approach) and based on gradient methods, but for real-world DNNs this approach in most cases is impossible. At the same time, several gradient-free optimization algorithms are used to attack black-box models. However, classical methods are often ineffective in the multidimensional case. To organize black-box attacks for computer vision models, in this work, we propose the use of an optimizer based on the low-rank tensor train (TT) format, which has gained popularity in various practical multidimensional applications in recent years. Combined with the attribution of the target image, which is built by the auxiliary (white-box) model, the TT-based optimization method makes it possible to organize an effective black-box attack by small perturbation of pixels in the target image. The superiority of the proposed approach over three popular baselines is demonstrated for five modern DNNs on the ImageNet dataset.

摘要: 深度神经网络(DNN)在当今得到了广泛的应用，但它们容易受到对手的攻击。为了开发有效的防御方法，了解DNNS的潜在弱点是重要的。通常，攻击是根据模型的体系结构(白盒方法)和基于梯度方法来组织的，但对于现实世界的DNN来说，这种方法在大多数情况下是不可能的。同时，利用几种无梯度优化算法对黑盒模型进行了攻击。然而，经典的方法在多维情况下往往是无效的。为了组织计算机视觉模型中的黑盒攻击，在这项工作中，我们提出了一种基于低阶张量训练(TT)格式的优化器，该优化器近年来在各种实际的多维应用中得到了普及。基于TT的优化方法结合辅助(白盒)模型建立的目标图像属性，通过对目标图像像素的微小扰动来组织有效的黑盒攻击。在ImageNet数据集上对五个现代DNN进行了测试，结果表明该方法优于三种流行的基线。



## **20. Counter-Empirical Attacking based on Adversarial Reinforcement Learning for Time-Relevant Scoring System**

基于对抗性强化学习的时间相关评分系统反经验攻击 cs.LG

Accepted by TKDE

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2311.05144v2) [paper-pdf](http://arxiv.org/pdf/2311.05144v2)

**Authors**: Xiangguo Sun, Hong Cheng, Hang Dong, Bo Qiao, Si Qin, Qingwei Lin

**Abstract**: Scoring systems are commonly seen for platforms in the era of big data. From credit scoring systems in financial services to membership scores in E-commerce shopping platforms, platform managers use such systems to guide users towards the encouraged activity pattern, and manage resources more effectively and more efficiently thereby. To establish such scoring systems, several "empirical criteria" are firstly determined, followed by dedicated top-down design for each factor of the score, which usually requires enormous effort to adjust and tune the scoring function in the new application scenario. What's worse, many fresh projects usually have no ground-truth or any experience to evaluate a reasonable scoring system, making the designing even harder. To reduce the effort of manual adjustment of the scoring function in every new scoring system, we innovatively study the scoring system from the preset empirical criteria without any ground truth, and propose a novel framework to improve the system from scratch. In this paper, we propose a "counter-empirical attacking" mechanism that can generate "attacking" behavior traces and try to break the empirical rules of the scoring system. Then an adversarial "enhancer" is applied to evaluate the scoring system and find the improvement strategy. By training the adversarial learning problem, a proper scoring function can be learned to be robust to the attacking activity traces that are trying to violate the empirical criteria. Extensive experiments have been conducted on two scoring systems including a shared computing resource platform and a financial credit system. The experimental results have validated the effectiveness of our proposed framework.

摘要: 在大数据时代，评分系统在平台上很常见。从金融服务的信用评分系统到电子商务购物平台的会员评分，平台管理者使用这些系统来引导用户走向鼓励活动模式，从而更有效和更高效地管理资源。要建立这样的评分系统，首先要确定几个“经验标准”，然后针对评分的每个因素进行专门的自上而下的设计，这通常需要付出巨大的努力来调整和调整新的应用场景中的评分函数。更糟糕的是，许多新项目通常没有实际情况或任何经验来评估合理的评分系统，这使得设计变得更加困难。为了减少每个新评分系统中人工调整评分函数的工作量，我们创新性地从预设的经验标准出发，在没有任何基础事实的情况下对评分系统进行研究，并提出了一个新的框架来从头开始改进该系统。在本文中，我们提出了一种可以产生攻击行为痕迹的反经验攻击机制，试图打破评分系统的经验规则。然后应用一个对抗性的“增强器”对评分系统进行评估，并找到改进策略。通过训练对抗性学习问题，可以学习适当的得分函数，以对试图违反经验标准的攻击活动痕迹具有健壮性。在包括共享计算资源平台和金融信用系统在内的两个评分系统上进行了广泛的实验。实验结果验证了该框架的有效性。



## **21. Position Bias Mitigation: A Knowledge-Aware Graph Model for Emotion Cause Extraction**

位置偏差缓解：一种用于情感原因提取的知识感知图模型 cs.CL

ACL2021 Main Conference, Oral paper

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2106.03518v3) [paper-pdf](http://arxiv.org/pdf/2106.03518v3)

**Authors**: Hanqi Yan, Lin Gui, Gabriele Pergola, Yulan He

**Abstract**: The Emotion Cause Extraction (ECE)} task aims to identify clauses which contain emotion-evoking information for a particular emotion expressed in text. We observe that a widely-used ECE dataset exhibits a bias that the majority of annotated cause clauses are either directly before their associated emotion clauses or are the emotion clauses themselves. Existing models for ECE tend to explore such relative position information and suffer from the dataset bias. To investigate the degree of reliance of existing ECE models on clause relative positions, we propose a novel strategy to generate adversarial examples in which the relative position information is no longer the indicative feature of cause clauses. We test the performance of existing models on such adversarial examples and observe a significant performance drop. To address the dataset bias, we propose a novel graph-based method to explicitly model the emotion triggering paths by leveraging the commonsense knowledge to enhance the semantic dependencies between a candidate clause and an emotion clause. Experimental results show that our proposed approach performs on par with the existing state-of-the-art methods on the original ECE dataset, and is more robust against adversarial attacks compared to existing models.

摘要: 情感原因提取(ECES)任务旨在识别包含文本中表达的特定情感的情感唤起信息的从句。我们观察到，一个广泛使用的欧洲经委会数据集显示出一种偏见，即大多数带注释的原因从句要么直接位于其关联的情感从句之前，要么是情感从句本身。欧洲经委会的现有模型倾向于探索这种相对位置信息，并受到数据集偏差的影响。为了考察现有的ECA模型对小句相对位置的依赖程度，我们提出了一种新的策略来生成对抗性实例，其中相对位置信息不再是原因从句的指示性特征。我们在这样的对抗性例子上测试了现有模型的性能，并观察到性能显著下降。为了解决数据集偏向的问题，我们提出了一种新的基于图的方法，通过利用常识知识来增强候选子句和情感子句之间的语义依赖，来显式建模情感触发路径。实验结果表明，我们提出的方法在原始ECA数据集上的性能与现有的最新方法相当，并且与现有的模型相比，对敌意攻击具有更好的鲁棒性。



## **22. QuanShield: Protecting against Side-Channels Attacks using Self-Destructing Enclaves**

QuanShield：使用自毁飞地防御旁路攻击 cs.CR

15pages, 5 figures, 5 tables

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2312.11796v1) [paper-pdf](http://arxiv.org/pdf/2312.11796v1)

**Authors**: Shujie Cui, Haohua Li, Yuanhong Li, Zhi Zhang, Lluís Vilanova, Peter Pietzuch

**Abstract**: Trusted Execution Environments (TEEs) allow user processes to create enclaves that protect security-sensitive computation against access from the OS kernel and the hypervisor. Recent work has shown that TEEs are vulnerable to side-channel attacks that allow an adversary to learn secrets shielded in enclaves. The majority of such attacks trigger exceptions or interrupts to trace the control or data flow of enclave execution.   We propose QuanShield, a system that protects enclaves from side-channel attacks that interrupt enclave execution. The main idea behind QuanShield is to strengthen resource isolation by creating an interrupt-free environment on a dedicated CPU core for running enclaves in which enclaves terminate when interrupts occur. QuanShield avoids interrupts by exploiting the tickless scheduling mode supported by recent OS kernels. QuanShield then uses the save area (SA) of the enclave, which is used by the hardware to support interrupt handling, as a second stack. Through an LLVM-based compiler pass, QuanShield modifies enclave instructions to store/load memory references, such as function frame base addresses, to/from the SA. When an interrupt occurs, the hardware overwrites the data in the SA with CPU state, thus ensuring that enclave execution fails. Our evaluation shows that QuanShield significantly raises the bar for interrupt-based attacks with practical overhead.

摘要: 可信执行环境（TEE）允许用户进程创建保护安全敏感计算免受来自OS内核和管理程序的访问的安全区。最近的研究表明，TEE容易受到侧信道攻击，使对手能够学习在飞地中屏蔽的秘密。大多数此类攻击会触发异常或中断来跟踪飞地执行的控制或数据流。   我们提出了QuanShield，一个系统，保护飞地从侧通道攻击中断飞地执行。QuanShield背后的主要思想是通过在专用CPU核心上创建无中断环境来加强资源隔离，以运行飞地，其中飞地在发生中断时终止。QuanShield通过利用最新操作系统内核支持的无时钟调度模式来避免中断。QuanShield然后使用安全区的保存区（SA）作为第二个堆栈，硬件使用该保存区来支持中断处理。通过基于LLVM的编译器通道，QuanShield修改飞地指令以向/从SA存储/加载存储器引用，例如函数帧基址。当中断发生时，硬件用CPU状态覆盖SA中的数据，从而确保飞地执行失败。我们的评估表明，QuanShield显著提高了基于中断的攻击的实际开销。



## **23. Impartial Games: A Challenge for Reinforcement Learning**

公平博弈：强化学习的挑战 cs.LG

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2205.12787v3) [paper-pdf](http://arxiv.org/pdf/2205.12787v3)

**Authors**: Bei Zhou, Søren Riis

**Abstract**: While AlphaZero-style reinforcement learning (RL) algorithms excel in various board games, in this paper we show that they face challenges on impartial games where players share pieces. We present a concrete example of a game - namely the children's game of Nim - and other impartial games that seem to be a stumbling block for AlphaZero-style and similar self-play reinforcement learning algorithms.   Our work is built on the challenges posed by the intricacies of data distribution on the ability of neural networks to learn parity functions, exacerbated by the noisy labels issue. Our findings are consistent with recent studies showing that AlphaZero-style algorithms are vulnerable to adversarial attacks and adversarial perturbations, showing the difficulty of learning to master the games in all legal states.   We show that Nim can be learned on small boards, but the learning progress of AlphaZero-style algorithms dramatically slows down when the board size increases. Intuitively, the difference between impartial games like Nim and partisan games like Chess and Go can be explained by the fact that if a small part of the board is covered for impartial games it is typically not possible to predict whether the position is won or lost as there is often zero correlation between the visible part of a partly blanked-out position and its correct evaluation. This situation starkly contrasts partisan games where a partly blanked-out board position typically provides abundant or at least non-trifle information about the value of the fully uncovered position.

摘要: 虽然AlphaZero风格的强化学习(RL)算法在各种棋类游戏中表现出色，但在本文中，我们展示了它们在玩家共享棋子的公平游戏中面临的挑战。我们提供了一个具体的游戏示例--即Nim的儿童游戏--以及其他公平的游戏，这些游戏似乎是AlphaZero风格和类似的自我发挥强化学习算法的绊脚石。我们的工作建立在错综复杂的数据分布对神经网络学习奇偶函数能力构成的挑战上，噪声标签问题加剧了这一挑战。我们的发现与最近的研究一致，这些研究表明AlphaZero风格的算法容易受到对手攻击和对手扰动，这表明在所有合法国家学习掌握游戏都是困难的。我们表明，NIM可以在小电路板上学习，但AlphaZero风格的算法的学习进度随着电路板大小的增加而显著减慢。直觉上，像尼姆这样的公正游戏与像国际象棋和围棋这样的党派游戏之间的区别可以用这样一个事实来解释：如果棋盘上的一小部分被公平地覆盖，通常不可能预测位置是赢是输，因为部分空白的位置的可见部分与其正确评估之间往往没有相关性。这种情况与党派游戏形成鲜明对比，在党派游戏中，部分空白的董事会职位通常会提供大量或至少不是无关紧要的信息，以了解完全暴露的职位的价值。



## **24. The Ultimate Combo: Boosting Adversarial Example Transferability by Composing Data Augmentations**

终极组合：通过组合数据增强来提高对抗性示例的可转移性 cs.CV

18 pages

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.11309v1) [paper-pdf](http://arxiv.org/pdf/2312.11309v1)

**Authors**: Zebin Yun, Achi-Or Weingarten, Eyal Ronen, Mahmood Sharif

**Abstract**: Transferring adversarial examples (AEs) from surrogate machine-learning (ML) models to target models is commonly used in black-box adversarial robustness evaluation. Attacks leveraging certain data augmentation, such as random resizing, have been found to help AEs generalize from surrogates to targets. Yet, prior work has explored limited augmentations and their composition. To fill the gap, we systematically studied how data augmentation affects transferability. Particularly, we explored 46 augmentation techniques of seven categories originally proposed to help ML models generalize to unseen benign samples, and assessed how they impact transferability, when applied individually or composed. Performing exhaustive search on a small subset of augmentation techniques and genetic search on all techniques, we identified augmentation combinations that can help promote transferability. Extensive experiments with the ImageNet and CIFAR-10 datasets and 18 models showed that simple color-space augmentations (e.g., color to greyscale) outperform the state of the art when combined with standard augmentations, such as translation and scaling. Additionally, we discovered that composing augmentations impacts transferability mostly monotonically (i.e., more methods composed $\rightarrow$ $\ge$ transferability). We also found that the best composition significantly outperformed the state of the art (e.g., 93.7% vs. $\le$ 82.7% average transferability on ImageNet from normally trained surrogates to adversarially trained targets). Lastly, our theoretical analysis, backed up by empirical evidence, intuitively explain why certain augmentations help improve transferability.

摘要: 在黑盒对抗健壮性评估中，经常使用从代理机器学习(ML)模型到目标模型的对抗性实例(AE)的转换。利用某些数据增强的攻击，如随机调整大小，已被发现有助于AE从代理扩展到目标。然而，先前的工作探索了有限的增强及其组成。为了填补这一空白，我们系统地研究了数据扩充如何影响可转移性。特别是，我们探索了最初提出的七个类别的46种增强技术，以帮助ML模型推广到看不见的良性样本，并评估了当单独应用或组合应用时，它们对可转移性的影响。我们对一小部分增强技术进行了穷举搜索，并对所有技术进行了遗传搜索，确定了有助于提高可转移性的增强组合。使用ImageNet和CIFAR-10数据集和18个模型进行的广泛实验表明，当与标准增强(如平移和缩放)相结合时，简单的颜色空间增强(例如，从颜色到灰度)的性能优于最先进的增强。此外，我们还发现，组合扩充对可转移性的影响主要是单调的(即，更多的方法组合了$\right tarrow$$\ge$t可转移性)。我们还发现，最好的成分远远超过了最先进的水平(例如，在ImageNet上，93.7%对82.7%的平均可转移性从正常训练的代理人转移到对抗性训练的目标)。最后，我们的理论分析得到了经验证据的支持，直观地解释了为什么某些增强有助于提高可转移性。



## **25. Adv-Diffusion: Imperceptible Adversarial Face Identity Attack via Latent Diffusion Model**

ADV扩散：基于潜在扩散模型的隐形对抗性人脸身份攻击 cs.CV

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.11285v1) [paper-pdf](http://arxiv.org/pdf/2312.11285v1)

**Authors**: Decheng Liu, Xijun Wang, Chunlei Peng, Nannan Wang, Ruiming Hu, Xinbo Gao

**Abstract**: Adversarial attacks involve adding perturbations to the source image to cause misclassification by the target model, which demonstrates the potential of attacking face recognition models. Existing adversarial face image generation methods still can't achieve satisfactory performance because of low transferability and high detectability. In this paper, we propose a unified framework Adv-Diffusion that can generate imperceptible adversarial identity perturbations in the latent space but not the raw pixel space, which utilizes strong inpainting capabilities of the latent diffusion model to generate realistic adversarial images. Specifically, we propose the identity-sensitive conditioned diffusion generative model to generate semantic perturbations in the surroundings. The designed adaptive strength-based adversarial perturbation algorithm can ensure both attack transferability and stealthiness. Extensive qualitative and quantitative experiments on the public FFHQ and CelebA-HQ datasets prove the proposed method achieves superior performance compared with the state-of-the-art methods without an extra generative model training process. The source code is available at https://github.com/kopper-xdu/Adv-Diffusion.

摘要: 对抗性攻击包括在源图像中添加扰动以导致目标模型的错误分类，这表明了攻击人脸识别模型的可能性。现有的对抗性人脸图像生成方法由于可移植性低、可检测性高，仍不能达到令人满意的效果。在本文中，我们提出了一个统一的框架，它可以在潜在空间而不是原始像素空间产生不可察觉的敌意身份扰动，该框架利用潜在扩散模型强大的修复能力来生成逼真的对抗性图像。具体地说，我们提出了身份敏感的条件扩散生成模型来产生环境中的语义扰动。所设计的基于强度的自适应对抗性扰动算法既能保证攻击的可传递性，又能保证隐蔽性。在公共FFHQ和CelebA-HQ数据集上的大量定性和定量实验证明，该方法在不需要额外的产生式模型训练过程的情况下，取得了优于最新方法的性能。源代码可在https://github.com/kopper-xdu/Adv-Diffusion.上找到



## **26. Protect Your Score: Contact Tracing With Differential Privacy Guarantees**

保护您的分数：具有差异隐私保证的联系人跟踪 cs.CR

Accepted to The 38th Annual AAAI Conference on Artificial  Intelligence (AAAI 2024)

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.11581v1) [paper-pdf](http://arxiv.org/pdf/2312.11581v1)

**Authors**: Rob Romijnders, Christos Louizos, Yuki M. Asano, Max Welling

**Abstract**: The pandemic in 2020 and 2021 had enormous economic and societal consequences, and studies show that contact tracing algorithms can be key in the early containment of the virus. While large strides have been made towards more effective contact tracing algorithms, we argue that privacy concerns currently hold deployment back. The essence of a contact tracing algorithm constitutes the communication of a risk score. Yet, it is precisely the communication and release of this score to a user that an adversary can leverage to gauge the private health status of an individual. We pinpoint a realistic attack scenario and propose a contact tracing algorithm with differential privacy guarantees against this attack. The algorithm is tested on the two most widely used agent-based COVID19 simulators and demonstrates superior performance in a wide range of settings. Especially for realistic test scenarios and while releasing each risk score with epsilon=1 differential privacy, we achieve a two to ten-fold reduction in the infection rate of the virus. To the best of our knowledge, this presents the first contact tracing algorithm with differential privacy guarantees when revealing risk scores for COVID19.

摘要: 2020年和2021年的大流行造成了巨大的经济和社会后果，研究表明，接触者追踪算法可能是早期遏制病毒的关键。虽然在更有效的联系人跟踪算法方面取得了长足的进步，但我们认为，隐私问题目前阻碍了部署。接触追踪算法的本质是传达风险评分。然而，正是向用户传达和发布该分数，对手可以利用该分数来衡量个人的私人健康状况。我们指出了一个现实的攻击场景，并提出了一个接触跟踪算法与差分隐私保证这种攻击。该算法在两个最广泛使用的基于代理的COVID 19模拟器上进行了测试，并在广泛的设置中表现出卓越的性能。特别是对于现实的测试场景，在释放每个风险评分时，我们将病毒的感染率降低了2到10倍。据我们所知，这是第一个在揭示COVID 19风险评分时具有差异隐私保证的接触追踪算法。



## **27. A Survey of Side-Channel Attacks in Context of Cache -- Taxonomies, Analysis and Mitigation**

Cache环境下的侧信道攻击综述--分类、分析与防范 cs.CR

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.11094v1) [paper-pdf](http://arxiv.org/pdf/2312.11094v1)

**Authors**: Ankit Pulkit, Smita Naval, Vijay Laxmi

**Abstract**: Side-channel attacks have become prominent attack surfaces in cyberspace. Attackers use the side information generated by the system while performing a task. Among the various side-channel attacks, cache side-channel attacks are leading as there has been an enormous growth in cache memory size in last decade, especially Last Level Cache (LLC). The adversary infers the information from the observable behavior of shared cache memory. This paper covers the detailed study of cache side-channel attacks and compares different microarchitectures in the context of side-channel attacks. Our main contributions are: (1) We have summarized the fundamentals and essentials of side-channel attacks and various attack surfaces (taxonomies). We also discussed different exploitation techniques, highlighting their capabilities and limitations. (2) We discussed cache side-channel attacks and analyzed the existing literature on cache side-channel attacks on various parameters like microarchitectures, cross-core exploitation, methodology, target, etc. (3) We discussed the detailed analysis of the existing mitigation strategies to prevent cache side-channel attacks. The analysis includes hardware- and software-based countermeasures, examining their strengths and weaknesses. We also discussed the challenges and trade-offs associated with mitigation strategies. This survey is supposed to provide a deeper understanding of the threats posed by these attacks to the research community with valuable insights into effective defense mechanisms.

摘要: 旁路攻击已成为网络空间的主要攻击面。攻击者在执行任务时使用系统生成的辅助信息。在各种侧通道攻击中，缓存侧通道攻击是主要的，因为在过去的十年中，高速缓存的大小有了巨大的增长，特别是最后一级高速缓存(LLC)。攻击者从共享高速缓冲存储器的可观察行为中推断信息。本文详细研究了缓存侧通道攻击，并比较了不同微体系结构在侧通道攻击环境中的应用。我们的主要贡献是：(1)总结了旁路攻击的基本原理和本质，以及各种攻击面(分类)。我们还讨论了不同的开发技术，强调了它们的功能和局限性。(2)讨论了缓存侧通道攻击，分析了现有的缓存侧通道攻击的研究文献，包括微体系结构、跨核利用、方法、攻击目标等。(3)详细分析了现有的缓存侧通道攻击的缓解策略。分析包括基于硬件和软件的对策，检查它们的优势和劣势。我们还讨论了与缓解策略相关的挑战和权衡。这项调查旨在更深入地了解这些攻击对研究界构成的威胁，为有效的防御机制提供有价值的见解。



## **28. Model Stealing Attack against Recommender System**

针对推荐系统的模型窃取攻击 cs.CR

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.11571v1) [paper-pdf](http://arxiv.org/pdf/2312.11571v1)

**Authors**: Zhihao Zhu, Rui Fan, Chenwang Wu, Yi Yang, Defu Lian, Enhong Chen

**Abstract**: Recent studies have demonstrated the vulnerability of recommender systems to data privacy attacks. However, research on the threat to model privacy in recommender systems, such as model stealing attacks, is still in its infancy. Some adversarial attacks have achieved model stealing attacks against recommender systems, to some extent, by collecting abundant training data of the target model (target data) or making a mass of queries. In this paper, we constrain the volume of available target data and queries and utilize auxiliary data, which shares the item set with the target data, to promote model stealing attacks. Although the target model treats target and auxiliary data differently, their similar behavior patterns allow them to be fused using an attention mechanism to assist attacks. Besides, we design stealing functions to effectively extract the recommendation list obtained by querying the target model. Experimental results show that the proposed methods are applicable to most recommender systems and various scenarios and exhibit excellent attack performance on multiple datasets.

摘要: 最近的研究表明，推荐系统的数据隐私攻击的脆弱性。然而，对推荐系统中模型隐私威胁的研究，如模型窃取攻击，仍处于起步阶段。一些对抗性攻击通过收集目标模型的大量训练数据（目标数据）或进行大量查询，在一定程度上实现了对推荐系统的模型窃取攻击。在本文中，我们限制了可用的目标数据和查询的数量，并利用辅助数据，它与目标数据共享的项目集，以促进模型窃取攻击。虽然目标模型以不同的方式处理目标数据和辅助数据，但它们相似的行为模式允许使用注意力机制将它们融合在一起以帮助攻击。此外，我们设计了窃取函数，有效地提取查询目标模型得到的推荐列表。实验结果表明，本文提出的方法适用于大多数推荐系统和各种场景，在多数据集上表现出良好的攻击性能。



## **29. Forbidden Facts: An Investigation of Competing Objectives in Llama-2**

禁忌事实：骆驼2号中相互竞争的目标的调查 cs.LG

Accepted to the ATTRIB and SoLaR workshops at NeurIPS 2023; (v2:  fixed typos)

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.08793v2) [paper-pdf](http://arxiv.org/pdf/2312.08793v2)

**Authors**: Tony T. Wang, Miles Wang, Kaivalya Hariharan, Nir Shavit

**Abstract**: LLMs often face competing pressures (for example helpfulness vs. harmlessness). To understand how models resolve such conflicts, we study Llama-2-chat models on the forbidden fact task. Specifically, we instruct Llama-2 to truthfully complete a factual recall statement while forbidding it from saying the correct answer. This often makes the model give incorrect answers. We decompose Llama-2 into 1000+ components, and rank each one with respect to how useful it is for forbidding the correct answer. We find that in aggregate, around 35 components are enough to reliably implement the full suppression behavior. However, these components are fairly heterogeneous and many operate using faulty heuristics. We discover that one of these heuristics can be exploited via a manually designed adversarial attack which we call The California Attack. Our results highlight some roadblocks standing in the way of being able to successfully interpret advanced ML systems. Project website available at https://forbiddenfacts.github.io .

摘要: 低收入国家经常面临相互竞争的压力(例如，有益与无害)。为了理解模型如何解决此类冲突，我们研究了关于禁止事实任务的Llama-2-Chat模型。具体地说，我们指示骆驼2号如实完成事实回忆声明，同时禁止它说出正确的答案。这经常使模型给出错误的答案。我们将Llama-2分解成1000多个成分，并根据它们对阻止正确答案的作用程度对每个成分进行排名。我们发现，总共大约35个组件就足以可靠地实现完全抑制行为。然而，这些组件具有相当大的异构性，许多组件使用错误的启发式方法进行操作。我们发现，其中一个启发式攻击可以通过手动设计的对抗性攻击来利用，我们称之为加利福尼亚州攻击。我们的结果突出了一些阻碍成功解释高级ML系统的障碍。项目网站为https://forbiddenfacts.github.io。



## **30. No-Skim: Towards Efficiency Robustness Evaluation on Skimming-based Language Models**

No-Skim：基于略读的语言模型的效率稳健性评价 cs.CR

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.09494v2) [paper-pdf](http://arxiv.org/pdf/2312.09494v2)

**Authors**: Shengyao Zhang, Mi Zhang, Xudong Pan, Min Yang

**Abstract**: To reduce the computation cost and the energy consumption in large language models (LLM), skimming-based acceleration dynamically drops unimportant tokens of the input sequence progressively along layers of the LLM while preserving the tokens of semantic importance. However, our work for the first time reveals the acceleration may be vulnerable to Denial-of-Service (DoS) attacks. In this paper, we propose No-Skim, a general framework to help the owners of skimming-based LLM to understand and measure the robustness of their acceleration scheme. Specifically, our framework searches minimal and unnoticeable perturbations at character-level and token-level to generate adversarial inputs that sufficiently increase the remaining token ratio, thus increasing the computation cost and energy consumption. We systematically evaluate the vulnerability of the skimming acceleration in various LLM architectures including BERT and RoBERTa on the GLUE benchmark. In the worst case, the perturbation found by No-Skim substantially increases the running cost of LLM by over 145% on average. Moreover, No-Skim extends the evaluation framework to various scenarios, making the evaluation conductible with different level of knowledge.

摘要: 为了降低大型语言模型（LLM）的计算成本和能耗，基于略读的加速动态地沿着LLM的层逐渐丢弃输入序列的不重要的标记，同时保留语义重要性的标记。然而，我们的工作首次揭示了加速可能容易受到拒绝服务（DoS）攻击。在本文中，我们提出了No-Skim，这是一个通用框架，可以帮助基于略读的LLM的所有者了解和衡量其加速方案的鲁棒性。具体来说，我们的框架在字符级和令牌级搜索最小且不明显的扰动，以生成足以增加剩余令牌比率的对抗性输入，从而增加计算成本和能耗。我们系统地评估了各种LLM架构（包括BERT和RoBERTA）在GLUE基准测试中的略读加速漏洞。在最坏的情况下，由No-Skim发现的扰动显著地增加了LLM的运行成本，平均超过145%。此外，No-Skim将评估框架扩展到各种场景，使评估具有不同的知识水平。



## **31. Security Defense of Large Scale Networks Under False Data Injection Attacks: An Attack Detection Scheduling Approach**

虚假数据注入攻击下的大规模网络安全防御：一种攻击检测调度方法 eess.SY

14 pages, 13 figures

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2212.05500v4) [paper-pdf](http://arxiv.org/pdf/2212.05500v4)

**Authors**: Yuhan Suo, Senchun Chai, Runqi Chai, Zhong-Hua Pang, Yuanqing Xia, Guo-Ping Liu

**Abstract**: In large-scale networks, communication links between nodes are easily injected with false data by adversaries. This paper proposes a novel security defense strategy from the perspective of attack detection scheduling to ensure the security of the network. Based on the proposed strategy, each sensor can directly exclude suspicious sensors from its neighboring set. First, the problem of selecting suspicious sensors is formulated as a combinatorial optimization problem, which is non-deterministic polynomial-time hard (NP-hard). To solve this problem, the original function is transformed into a submodular function. Then, we propose an attack detection scheduling algorithm based on the sequential submodular optimization theory, which incorporates \emph{expert problem} to better utilize historical information to guide the sensor selection task at the current moment. For different attack strategies, theoretical results show that the average optimization rate of the proposed algorithm has a lower bound, and the error expectation is bounded. In addition, under two kinds of insecurity conditions, the proposed algorithm can guarantee the security of the entire network from the perspective of the augmented estimation error. Finally, the effectiveness of the developed method is verified by the numerical simulation and practical experiment.

摘要: 在大规模网络中，节点之间的通信链路很容易被对手注入虚假数据。本文从攻击检测调度的角度提出了一种新的安全防御策略，以确保网络的安全。基于该策略，每个传感器可以直接从其相邻集合中排除可疑传感器。首先，将可疑传感器的选择问题描述为一个非确定多项式时间难(NP-Hard)的组合优化问题。为了解决这一问题，将原函数转化为子模函数。在此基础上，提出了一种基于序贯子模优化理论的攻击检测调度算法，该算法结合专家问题，更好地利用历史信息指导当前时刻的传感器选择任务。理论结果表明，对于不同的攻击策略，该算法的平均优化率有一个下界，误差期望是有界的。另外，在两种不安全情况下，从估计误差增大的角度来看，该算法可以保证整个网络的安全性。最后，通过数值模拟和实际实验验证了该方法的有效性。



## **32. Security for Machine Learning-based Software Systems: a survey of threats, practices and challenges**

基于机器学习的软件系统安全：威胁、实践和挑战综述 cs.CR

Accepted at ACM Computing Surveys

**SubmitDate**: 2023-12-17    [abs](http://arxiv.org/abs/2201.04736v2) [paper-pdf](http://arxiv.org/pdf/2201.04736v2)

**Authors**: Huaming Chen, M. Ali Babar

**Abstract**: The rapid development of Machine Learning (ML) has demonstrated superior performance in many areas, such as computer vision, video and speech recognition. It has now been increasingly leveraged in software systems to automate the core tasks. However, how to securely develop the machine learning-based modern software systems (MLBSS) remains a big challenge, for which the insufficient consideration will largely limit its application in safety-critical domains. One concern is that the present MLBSS development tends to be rush, and the latent vulnerabilities and privacy issues exposed to external users and attackers will be largely neglected and hard to be identified. Additionally, machine learning-based software systems exhibit different liabilities towards novel vulnerabilities at different development stages from requirement analysis to system maintenance, due to its inherent limitations from the model and data and the external adversary capabilities. The successful generation of such intelligent systems will thus solicit dedicated efforts jointly from different research areas, i.e., software engineering, system security and machine learning. Most of the recent works regarding the security issues for ML have a strong focus on the data and models, which has brought adversarial attacks into consideration. In this work, we consider that security for machine learning-based software systems may arise from inherent system defects or external adversarial attacks, and the secure development practices should be taken throughout the whole lifecycle. While machine learning has become a new threat domain for existing software engineering practices, there is no such review work covering the topic. Overall, we present a holistic review regarding the security for MLBSS, which covers a systematic understanding from a structure review of three distinct aspects in terms of security threats...

摘要: 机器学习的快速发展在计算机视觉、视频和语音识别等领域表现出了优异的性能。它现在越来越多地被软件系统用来自动化核心任务。然而，如何安全地开发基于机器学习的现代软件系统仍然是一个巨大的挑战，对此考虑不足将在很大程度上限制其在安全关键领域的应用。一个令人担忧的问题是，目前MLBSS的开发往往是仓促的，暴露给外部用户和攻击者的潜在漏洞和隐私问题将在很大程度上被忽视，很难识别。此外，基于机器学习的软件系统在从需求分析到系统维护的不同开发阶段，由于其固有的模型和数据限制以及外部对手能力的限制，对新漏洞表现出不同的易感性。因此，这类智能系统的成功生成将需要软件工程、系统安全和机器学习等不同研究领域的共同努力。最近关于ML的安全问题的大部分工作都集中在数据和模型上，这使得对抗性攻击被考虑在内。在这项工作中，我们认为基于机器学习的软件系统的安全性可能来自于内在的系统缺陷或外部的对抗性攻击，安全开发实践应该贯穿于整个生命周期。虽然机器学习已经成为现有软件工程实践的一个新的威胁领域，但还没有涵盖这一主题的审查工作。总体而言，我们对MLBSS的安全进行了全面的审查，其中包括从结构审查方面对安全威胁的三个不同方面进行系统的理解。



## **33. Single Image Backdoor Inversion via Robust Smoothed Classifiers**

基于稳健平滑分类器的单幅图像后门反演 cs.CV

CVPR 2023. v2: improved writing

**SubmitDate**: 2023-12-17    [abs](http://arxiv.org/abs/2303.00215v2) [paper-pdf](http://arxiv.org/pdf/2303.00215v2)

**Authors**: Mingjie Sun, J. Zico Kolter

**Abstract**: Backdoor inversion, a central step in many backdoor defenses, is a reverse-engineering process to recover the hidden backdoor trigger inserted into a machine learning model. Existing approaches tackle this problem by searching for a backdoor pattern that is able to flip a set of clean images into the target class, while the exact size needed of this support set is rarely investigated. In this work, we present a new approach for backdoor inversion, which is able to recover the hidden backdoor with as few as a single image. Insipired by recent advances in adversarial robustness, our method SmoothInv starts from a single clean image, and then performs projected gradient descent towards the target class on a robust smoothed version of the original backdoored classifier. We find that backdoor patterns emerge naturally from such optimization process. Compared to existing backdoor inversion methods, SmoothInv introduces minimum optimization variables and does not require complex regularization schemes. We perform a comprehensive quantitative and qualitative study on backdoored classifiers obtained from existing backdoor attacks. We demonstrate that SmoothInv consistently recovers successful backdoors from single images: for backdoored ImageNet classifiers, our reconstructed backdoors have close to 100% attack success rates. We also show that they maintain high fidelity to the underlying true backdoors. Last, we propose and analyze two countermeasures to our approach and show that SmoothInv remains robust in the face of an adaptive attacker. Our code is available at https://github.com/locuslab/smoothinv.

摘要: 后门倒置是许多后门防御的核心步骤，它是一种反向工程过程，目的是恢复插入到机器学习模型中的隐藏后门触发器。现有的方法通过搜索能够将一组干净的图像翻转到目标类中的后门模式来解决这个问题，而很少研究该支持集所需的确切大小。在这项工作中，我们提出了一种新的后门反转方法，该方法能够用一张图像来恢复隐藏的后门。受对抗鲁棒性的最新进展的启发，我们的方法SmoothInv从单一的干净图像开始，然后在原始后向分类器的稳健平滑版本上向目标类执行投影梯度下降。我们发现，后门模式在这样的优化过程中自然而然地出现了。与现有的后门反演方法相比，SmoothInv引入了最小优化变量，不需要复杂的正则化方案。我们对从现有的后门攻击中获得的后置分类器进行了全面的定量和定性研究。我们证明了SmoothInv能够持续地从单个映像恢复成功的后门：对于后置的ImageNet分类器，我们重建的后门攻击成功率接近100%。我们还表明，它们对潜在的真正后门保持着高度的保真度。最后，我们针对我们的方法提出并分析了两种对策，表明SmoothInv在面对自适应攻击者时仍然保持健壮性。我们的代码可以在https://github.com/locuslab/smoothinv.上找到



## **34. Synthesizing Black-box Anti-forensics DeepFakes with High Visual Quality**

合成高视觉质量的黑盒反取证深粉 cs.CV

Accepted for publication at ICASSP 2024

**SubmitDate**: 2023-12-17    [abs](http://arxiv.org/abs/2312.10713v1) [paper-pdf](http://arxiv.org/pdf/2312.10713v1)

**Authors**: Bing Fan, Shu Hu, Feng Ding

**Abstract**: DeepFake, an AI technology for creating facial forgeries, has garnered global attention. Amid such circumstances, forensics researchers focus on developing defensive algorithms to counter these threats. In contrast, there are techniques developed for enhancing the aggressiveness of DeepFake, e.g., through anti-forensics attacks, to disrupt forensic detectors. However, such attacks often sacrifice image visual quality for improved undetectability. To address this issue, we propose a method to generate novel adversarial sharpening masks for launching black-box anti-forensics attacks. Unlike many existing arts, with such perturbations injected, DeepFakes could achieve high anti-forensics performance while exhibiting pleasant sharpening visual effects. After experimental evaluations, we prove that the proposed method could successfully disrupt the state-of-the-art DeepFake detectors. Besides, compared with the images processed by existing DeepFake anti-forensics methods, the visual qualities of anti-forensics DeepFakes rendered by the proposed method are significantly refined.

摘要: DeepFake，一项制造面部赝品的人工智能技术，引起了全球的关注。在这种情况下，法医研究人员专注于开发防御算法来应对这些威胁。相比之下，已经开发了一些技术来增强DeepFake的攻击性，例如通过反取证攻击来扰乱法医探测器。然而，这类攻击往往以牺牲图像视觉质量为代价来提高不可测性。为了解决这个问题，我们提出了一种生成新的对抗性锐化面具的方法，用于发起黑盒反取证攻击。与许多现有的艺术不同，在注入这样的扰动后，DeepFake可以实现高反取证性能，同时显示出令人愉快的锐化视觉效果。经过实验评估，我们证明所提出的方法可以成功地干扰最先进的DeepFake探测器。此外，与已有的DeepFake反取证方法处理的图像相比，由该方法绘制的反取证DeepFake图像的视觉质量有了显著的提高。



## **35. Analisis Eksploratif Dan Augmentasi Data NSL-KDD Menggunakan Deep Generative Adversarial Networks Untuk Meningkatkan Performa Algoritma Extreme Gradient Boosting Dalam Klasifikasi Jenis Serangan Siber**

NSL-KDD深度生成对抗网络的数据增强分析使用极端梯度增强算法 cs.CR

in Indonesian language

**SubmitDate**: 2023-12-17    [abs](http://arxiv.org/abs/2312.10669v1) [paper-pdf](http://arxiv.org/pdf/2312.10669v1)

**Authors**: K. P. Santoso, F. A. Madany, H. Suryotrisongko

**Abstract**: This study proposes the implementation of Deep Generative Adversarial Networks (GANs) for augmenting the NSL-KDD dataset. The primary objective is to enhance the efficacy of eXtreme Gradient Boosting (XGBoost) in the classification of cyber-attacks on the NSL-KDD dataset. As a result, the method proposed in this research achieved an accuracy of 99.53% using the XGBoost model without data augmentation with GAN, and 99.78% with data augmentation using GAN.

摘要: 本研究提出了一种用于扩充NSL-KDD数据集的深度生成对抗性网络(GANS)。主要目标是提高极端梯度增强(XGBoost)在对NSL-KDD数据集的网络攻击分类中的有效性。结果表明，在不使用GaN进行数据增强的情况下，本文提出的方法在XGBoost模型下获得了99.53%的准确率，而在使用GaN进行数据增强时，准确率达到了99.78%。



## **36. UltraClean: A Simple Framework to Train Robust Neural Networks against Backdoor Attacks**

超清洁：一种训练稳健神经网络抵御后门攻击的简单框架 cs.CR

**SubmitDate**: 2023-12-17    [abs](http://arxiv.org/abs/2312.10657v1) [paper-pdf](http://arxiv.org/pdf/2312.10657v1)

**Authors**: Bingyin Zhao, Yingjie Lao

**Abstract**: Backdoor attacks are emerging threats to deep neural networks, which typically embed malicious behaviors into a victim model by injecting poisoned samples. Adversaries can activate the injected backdoor during inference by presenting the trigger on input images. Prior defensive methods have achieved remarkable success in countering dirty-label backdoor attacks where the labels of poisoned samples are often mislabeled. However, these approaches do not work for a recent new type of backdoor -- clean-label backdoor attacks that imperceptibly modify poisoned data and hold consistent labels. More complex and powerful algorithms are demanded to defend against such stealthy attacks. In this paper, we propose UltraClean, a general framework that simplifies the identification of poisoned samples and defends against both dirty-label and clean-label backdoor attacks. Given the fact that backdoor triggers introduce adversarial noise that intensifies in feed-forward propagation, UltraClean first generates two variants of training samples using off-the-shelf denoising functions. It then measures the susceptibility of training samples leveraging the error amplification effect in DNNs, which dilates the noise difference between the original image and denoised variants. Lastly, it filters out poisoned samples based on the susceptibility to thwart the backdoor implantation. Despite its simplicity, UltraClean achieves a superior detection rate across various datasets and significantly reduces the backdoor attack success rate while maintaining a decent model accuracy on clean data, outperforming existing defensive methods by a large margin. Code is available at https://github.com/bxz9200/UltraClean.

摘要: 后门攻击是对深度神经网络的新威胁，它通常通过注入有毒样本将恶意行为嵌入到受害者模型中。攻击者可以在推理过程中通过在输入图像上显示触发器来激活注入的后门。以前的防御方法在对抗脏标签后门攻击方面取得了显着的成功，在这些后门攻击中，有毒样品的标签经常被贴错标签。然而，这些方法不适用于最近的一种新类型的后门--干净标签后门攻击，它可以潜移默化地修改有毒数据并保持一致的标签。需要更复杂、更强大的算法来防御这种隐形攻击。在本文中，我们提出了UltraClean，这是一个通用框架，它简化了有毒样本的识别，并同时防御了脏标签和干净标签的后门攻击。鉴于后门触发器引入的对抗性噪声会在前馈传播中加剧，UltraClean首先使用现成的去噪功能生成两种不同的训练样本。然后利用DNN中的误差放大效应来衡量训练样本的敏感度，这扩大了原始图像和去噪变体之间的噪声差异。最后，它基于阻止后门植入的敏感度来过滤有毒样本。尽管简单，但UltraClean在各种数据集上实现了卓越的检测率，并显著降低了后门攻击成功率，同时保持了对干净数据的良好模型准确性，大大超过了现有的防御方法。代码可在https://github.com/bxz9200/UltraClean.上找到



## **37. Rethinking Robustness of Model Attributions**

对模型属性稳健性的再思考 cs.LG

Accepted AAAI 2024

**SubmitDate**: 2023-12-16    [abs](http://arxiv.org/abs/2312.10534v1) [paper-pdf](http://arxiv.org/pdf/2312.10534v1)

**Authors**: Sandesh Kamath, Sankalp Mittal, Amit Deshpande, Vineeth N Balasubramanian

**Abstract**: For machine learning models to be reliable and trustworthy, their decisions must be interpretable. As these models find increasing use in safety-critical applications, it is important that not just the model predictions but also their explanations (as feature attributions) be robust to small human-imperceptible input perturbations. Recent works have shown that many attribution methods are fragile and have proposed improvements in either these methods or the model training. We observe two main causes for fragile attributions: first, the existing metrics of robustness (e.g., top-k intersection) over-penalize even reasonable local shifts in attribution, thereby making random perturbations to appear as a strong attack, and second, the attribution can be concentrated in a small region even when there are multiple important parts in an image. To rectify this, we propose simple ways to strengthen existing metrics and attribution methods that incorporate locality of pixels in robustness metrics and diversity of pixel locations in attributions. Towards the role of model training in attributional robustness, we empirically observe that adversarially trained models have more robust attributions on smaller datasets, however, this advantage disappears in larger datasets. Code is available at https://github.com/ksandeshk/LENS.

摘要: 为了让机器学习模型可靠可信，它们的决策必须是可解释的。随着这些模型在安全关键应用中的应用越来越多，重要的是不仅模型预测而且它们的解释(作为特征属性)对于人类无法察觉的小输入扰动是健壮的。最近的工作表明，许多归因方法是脆弱的，并提出了对这些方法或模型训练的改进。我们观察到脆弱属性的两个主要原因：第一，现有的稳健性度量(例如，top-k相交)过度惩罚了属性中甚至合理的局部移动，从而使得随机扰动看起来像是一种强攻击；第二，即使图像中有多个重要部分，属性也可能集中在一个小区域内。为了纠正这一点，我们提出了一些简单的方法来加强现有的度量和属性方法，这些方法在稳健性度量中包含了像素的局部性，在属性中包含了像素位置的多样性。对于模型训练在属性稳健性中的作用，我们经验地观察到，对抗性训练的模型在较小的数据集上具有更健壮的属性，然而，这种优势在较大的数据集上消失了。代码可在https://github.com/ksandeshk/LENS.上找到



## **38. Transformers in Unsupervised Structure-from-Motion**

无监督运动结构中的变压器 cs.CV

International Joint Conference on Computer Vision, Imaging and  Computer Graphics. Cham: Springer Nature Switzerland, 2022. Published at  "Communications in Computer and Information Science, vol 1815. Springer  Nature". arXiv admin note: text overlap with arXiv:2202.03131

**SubmitDate**: 2023-12-16    [abs](http://arxiv.org/abs/2312.10529v1) [paper-pdf](http://arxiv.org/pdf/2312.10529v1)

**Authors**: Hemang Chawla, Arnav Varma, Elahe Arani, Bahram Zonooz

**Abstract**: Transformers have revolutionized deep learning based computer vision with improved performance as well as robustness to natural corruptions and adversarial attacks. Transformers are used predominantly for 2D vision tasks, including image classification, semantic segmentation, and object detection. However, robots and advanced driver assistance systems also require 3D scene understanding for decision making by extracting structure-from-motion (SfM). We propose a robust transformer-based monocular SfM method that learns to predict monocular pixel-wise depth, ego vehicle's translation and rotation, as well as camera's focal length and principal point, simultaneously. With experiments on KITTI and DDAD datasets, we demonstrate how to adapt different vision transformers and compare them against contemporary CNN-based methods. Our study shows that transformer-based architecture, though lower in run-time efficiency, achieves comparable performance while being more robust against natural corruptions, as well as untargeted and targeted attacks.

摘要: 变形金刚使基于深度学习的计算机视觉发生了革命性的变化，提高了性能以及对自然腐败和对手攻击的稳健性。变形金刚主要用于2D视觉任务，包括图像分类、语义分割和目标检测。然而，机器人和先进的驾驶员辅助系统也需要通过从运动中提取结构(SFM)来理解3D场景以进行决策。我们提出了一种稳健的基于变换的单目模糊模型方法，该方法学习同时预测单目像素级深度、EGO车辆的平移和旋转以及相机的焦距和主点。通过在Kitti和Dda数据集上的实验，我们演示了如何适应不同的视觉转换器，并将它们与当代基于CNN的方法进行了比较。我们的研究表明，基于转换器的体系结构虽然运行效率较低，但可以获得类似的性能，同时对自然损坏以及非目标攻击和目标攻击具有更强的健壮性。



## **39. IoTGAN: GAN Powered Camouflage Against Machine Learning Based IoT Device Identification**

物联网：基于机器学习的GAN伪装技术 cs.CR

**SubmitDate**: 2023-12-16    [abs](http://arxiv.org/abs/2201.03281v2) [paper-pdf](http://arxiv.org/pdf/2201.03281v2)

**Authors**: Tao Hou, Tao Wang, Zhuo Lu, Yao Liu, Yalin Sagduyu

**Abstract**: With the proliferation of IoT devices, researchers have developed a variety of IoT device identification methods with the assistance of machine learning. Nevertheless, the security of these identification methods mostly depends on collected training data. In this research, we propose a novel attack strategy named IoTGAN to manipulate an IoT device's traffic such that it can evade machine learning based IoT device identification. In the development of IoTGAN, we have two major technical challenges: (i) How to obtain the discriminative model in a black-box setting, and (ii) How to add perturbations to IoT traffic through the manipulative model, so as to evade the identification while not influencing the functionality of IoT devices. To address these challenges, a neural network based substitute model is used to fit the target model in black-box settings, it works as a discriminative model in IoTGAN. A manipulative model is trained to add adversarial perturbations into the IoT device's traffic to evade the substitute model. Experimental results show that IoTGAN can successfully achieve the attack goals. We also develop efficient countermeasures to protect machine learning based IoT device identification from been undermined by IoTGAN.

摘要: 随着物联网设备的激增，研究人员借助机器学习开发了各种物联网设备识别方法。然而，这些识别方法的安全性在很大程度上取决于收集的训练数据。在这项研究中，我们提出了一种新的攻击策略IoTGAN来操纵物联网设备的流量，使其能够逃避基于机器学习的物联网设备识别。在IoTGAN的开发中，我们面临着两大技术挑战：(I)如何在黑盒环境下获得判别模型，以及(Ii)如何通过操纵模型向物联网流量添加扰动，从而在不影响物联网设备功能的情况下逃避识别。为了应对这些挑战，在黑盒环境下使用了一种基于神经网络的替代模型来拟合目标模型，该模型在IoTGAN中用作判别模型。操纵性模型被训练成向物联网设备的流量中添加对抗性扰动以避开替代模型。实验结果表明，IoTGAN能够成功实现攻击目标。我们还开发了有效的对策来保护基于机器学习的物联网设备识别免受IoTGAN的破坏。



## **40. Robust Communicative Multi-Agent Reinforcement Learning with Active Defense**

主动防御下的健壮通信型多智能体强化学习 cs.MA

Accepted by AAAI 2024

**SubmitDate**: 2023-12-16    [abs](http://arxiv.org/abs/2312.11545v1) [paper-pdf](http://arxiv.org/pdf/2312.11545v1)

**Authors**: Lebin Yu, Yunbo Qiu, Quanming Yao, Yuan Shen, Xudong Zhang, Jian Wang

**Abstract**: Communication in multi-agent reinforcement learning (MARL) has been proven to effectively promote cooperation among agents recently. Since communication in real-world scenarios is vulnerable to noises and adversarial attacks, it is crucial to develop robust communicative MARL technique. However, existing research in this domain has predominantly focused on passive defense strategies, where agents receive all messages equally, making it hard to balance performance and robustness. We propose an active defense strategy, where agents automatically reduce the impact of potentially harmful messages on the final decision. There are two challenges to implement this strategy, that are defining unreliable messages and adjusting the unreliable messages' impact on the final decision properly. To address them, we design an Active Defense Multi-Agent Communication framework (ADMAC), which estimates the reliability of received messages and adjusts their impact on the final decision accordingly with the help of a decomposable decision structure. The superiority of ADMAC over existing methods is validated by experiments in three communication-critical tasks under four types of attacks.

摘要: 近年来，多智能体强化学习(MAIL)中的通信被证明能有效地促进智能体之间的合作。由于现实场景中的通信很容易受到噪声和对手攻击，因此开发健壮的通信Marl技术至关重要。然而，该领域的现有研究主要集中在被动防御策略上，即代理平等地接收所有消息，这使得很难在性能和健壮性之间取得平衡。我们提出了一种主动防御策略，在该策略中，代理自动减少潜在有害消息对最终决策的影响。实施这一策略有两个挑战，一是定义不可靠消息，二是适当调整不可靠消息对最终决策的影响。针对这些问题，我们设计了一个主动防御多智能体通信框架(ADMAC)，该框架通过一个可分解的决策结构来评估接收消息的可靠性，并相应地调整它们对最终决策的影响。通过在四种类型攻击下的三个通信关键任务上的实验，验证了ADMAC方法相对于现有方法的优越性。



## **41. PPIDSG: A Privacy-Preserving Image Distribution Sharing Scheme with GAN in Federated Learning**

PPIDSG：联邦学习中一种保护隐私的GAN图像分布共享方案 cs.LG

Accepted by AAAI 2024

**SubmitDate**: 2023-12-16    [abs](http://arxiv.org/abs/2312.10380v1) [paper-pdf](http://arxiv.org/pdf/2312.10380v1)

**Authors**: Yuting Ma, Yuanzhi Yao, Xiaohua Xu

**Abstract**: Federated learning (FL) has attracted growing attention since it allows for privacy-preserving collaborative training on decentralized clients without explicitly uploading sensitive data to the central server. However, recent works have revealed that it still has the risk of exposing private data to adversaries. In this paper, we conduct reconstruction attacks and enhance inference attacks on various datasets to better understand that sharing trained classification model parameters to a central server is the main problem of privacy leakage in FL. To tackle this problem, a privacy-preserving image distribution sharing scheme with GAN (PPIDSG) is proposed, which consists of a block scrambling-based encryption algorithm, an image distribution sharing method, and local classification training. Specifically, our method can capture the distribution of a target image domain which is transformed by the block encryption algorithm, and upload generator parameters to avoid classifier sharing with negligible influence on model performance. Furthermore, we apply a feature extractor to motivate model utility and train it separately from the classifier. The extensive experimental results and security analyses demonstrate the superiority of our proposed scheme compared to other state-of-the-art defense methods. The code is available at https://github.com/ytingma/PPIDSG.

摘要: 联合学习(FL)由于允许在分散的客户端上进行隐私保护的协作培训，而无需显式地将敏感数据上传到中央服务器，因此引起了越来越多的关注。然而，最近的作品透露，它仍然存在将私人数据暴露给对手的风险。在本文中，我们对不同的数据集进行重构攻击和增强推理攻击，以更好地理解将训练好的分类模型参数共享到中央服务器是FL中隐私泄露的主要问题。针对这一问题，提出了一种基于GAN的隐私保护图像分发共享方案(PPIDSG)，该方案由基于块置乱的加密算法、图像分发共享方法和局部分类训练组成。具体地说，我们的方法能够捕获经过块加密算法变换的目标图像域的分布，并上传生成器参数以避免分类器共享，而对模型性能的影响可以忽略不计。此外，我们应用一个特征抽取器来激励模型效用，并将其从分类器中分离出来进行训练。大量的实验结果和安全性分析表明，与其他最先进的防御方法相比，我们提出的方案具有更好的优越性。代码可在https://github.com/ytingma/PPIDSG.上获得



## **42. Perturbation-Invariant Adversarial Training for Neural Ranking Models: Improving the Effectiveness-Robustness Trade-Off**

神经排序模型的扰动不变对抗训练：改进容错性-鲁棒性权衡 cs.IR

Accepted by AAAI 24

**SubmitDate**: 2023-12-16    [abs](http://arxiv.org/abs/2312.10329v1) [paper-pdf](http://arxiv.org/pdf/2312.10329v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Mingkun Zhang, Wei Chen, Maarten de Rijke, Jiafeng Guo, Xueqi Cheng

**Abstract**: Neural ranking models (NRMs) have shown great success in information retrieval (IR). But their predictions can easily be manipulated using adversarial examples, which are crafted by adding imperceptible perturbations to legitimate documents. This vulnerability raises significant concerns about their reliability and hinders the widespread deployment of NRMs. By incorporating adversarial examples into training data, adversarial training has become the de facto defense approach to adversarial attacks against NRMs. However, this defense mechanism is subject to a trade-off between effectiveness and adversarial robustness. In this study, we establish theoretical guarantees regarding the effectiveness-robustness trade-off in NRMs. We decompose the robust ranking error into two components, i.e., a natural ranking error for effectiveness evaluation and a boundary ranking error for assessing adversarial robustness. Then, we define the perturbation invariance of a ranking model and prove it to be a differentiable upper bound on the boundary ranking error for attainable computation. Informed by our theoretical analysis, we design a novel \emph{perturbation-invariant adversarial training} (PIAT) method for ranking models to achieve a better effectiveness-robustness trade-off. We design a regularized surrogate loss, in which one term encourages the effectiveness to be maximized while the regularization term encourages the output to be smooth, so as to improve adversarial robustness. Experimental results on several ranking models demonstrate the superiority of PITA compared to existing adversarial defenses.

摘要: 神经排序模型(NRM)在信息检索中取得了巨大的成功。但他们的预测很容易被操纵，使用敌对的例子，这些例子是通过在合法文件中添加难以察觉的扰动来精心制作的。这一漏洞引起了人们对其可靠性的严重担忧，并阻碍了NRM的广泛部署。通过将对抗性例子纳入训练数据，对抗性训练已成为针对NRM的对抗性攻击的事实上的防御方法。然而，这种防御机制需要在有效性和对抗健壮性之间进行权衡。在这项研究中，我们建立了关于NRM中有效性和稳健性权衡的理论保证。我们将鲁棒排序误差分解为两个分量，即用于评估有效性的自然排序误差和用于评估对手健壮性的边界排序误差。然后，我们定义了排序模型的扰动不变性，并证明了它是可得计算的边界排序误差的一个可微上界。在理论分析的指导下，我们设计了一种新的扰动不变对手训练方法(PIAT)来对模型进行排序，以达到更好的有效性和稳健性之间的权衡。我们设计了一种正则化的代理损失，其中一个项鼓励最大化有效性，而正则项则鼓励输出平滑，从而提高对手的稳健性。在几种排序模型上的实验结果表明，与现有的对抗性防御相比，PITA具有更好的性能。



## **43. Enhancing Accuracy and Robustness of Steering Angle Prediction with Attention Mechanism**

利用注意机制提高转向角预测的准确性和鲁棒性 cs.CV

**SubmitDate**: 2023-12-16    [abs](http://arxiv.org/abs/2211.11133v3) [paper-pdf](http://arxiv.org/pdf/2211.11133v3)

**Authors**: Swetha Nadella, Pramiti Barua, Jeremy C. Hagler, David J. Lamb, Qing Tian

**Abstract**: In this paper, we investigate the two most popular families of deep neural architectures (i.e., ResNets and InceptionNets) for the autonomous driving task of steering angle prediction. To ensure a comprehensive comparison, we conducted experiments on the Kaggle SAP dataset and custom dataset and carefully examined a range of different model sizes within both the ResNet and InceptionNet families. Our derived models can achieve state-of-the-art results in terms of steering angle MSE. In addition to this analysis, we introduced the attention mechanism to enhance steering angle prediction. This attention mechanism facilitated an in-depth exploration of the model's selective focus on essential elements within the input data. Furthermore, recognizing the importance of security and robustness in autonomous driving assessed the resilience of our models to adversarial attacks.

摘要: 在本文中，我们研究了两个最流行的深层神经结构家族(即ResNets和InceptionNets)，用于自动驾驶的转向角预测任务。为了确保全面的比较，我们在Kaggle SAP数据集和自定义数据集上进行了实验，并仔细检查了ResNet和InceptionNet系列中的一系列不同的模型大小。我们导出的模型可以在转向角MSE方面获得最先进的结果。除了这一分析，我们还引入了注意力机制来增强转向角预测。这一注意机制有助于深入探讨模型对输入数据中的基本要素的选择性关注。此外，认识到安全和健壮性在自动驾驶中的重要性，评估了我们的模型对对手攻击的弹性。



## **44. Dynamic Gradient Balancing for Enhanced Adversarial Attacks on Multi-Task Models**

多任务模型增强对抗性攻击的动态梯度平衡算法 cs.LG

19 pages, 6 figures; AAAI24

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2305.12066v2) [paper-pdf](http://arxiv.org/pdf/2305.12066v2)

**Authors**: Lijun Zhang, Xiao Liu, Kaleel Mahmood, Caiwen Ding, Hui Guan

**Abstract**: Multi-task learning (MTL) creates a single machine learning model called multi-task model to simultaneously perform multiple tasks. Although the security of single task classifiers has been extensively studied, there are several critical security research questions for multi-task models including 1) How secure are multi-task models to single task adversarial machine learning attacks, 2) Can adversarial attacks be designed to attack multiple tasks simultaneously, and 3) Does task sharing and adversarial training increase multi-task model robustness to adversarial attacks? In this paper, we answer these questions through careful analysis and rigorous experimentation. First, we develop na\"ive adaptation of single-task white-box attacks and analyze their inherent drawbacks. We then propose a novel attack framework, Dynamic Gradient Balancing Attack (DGBA). Our framework poses the problem of attacking a multi-task model as an optimization problem based on averaged relative loss change, which can be solved by approximating the problem as an integer linear programming problem. Extensive evaluation on two popular MTL benchmarks, NYUv2 and Tiny-Taxonomy, demonstrates the effectiveness of DGBA compared to na\"ive multi-task attack baselines on both clean and adversarially trained multi-task models. The results also reveal a fundamental trade-off between improving task accuracy by sharing parameters across tasks and undermining model robustness due to increased attack transferability from parameter sharing. DGBA is open-sourced and available at https://github.com/zhanglijun95/MTLAttack-DGBA.

摘要: 多任务学习(MTL)创建了一种称为多任务模型的单机学习模型，用于同时执行多个任务。尽管单任务分类器的安全性已经得到了广泛的研究，但多任务模型仍然存在几个关键的安全研究问题，包括：1)多任务模型对抗性机器学习攻击的安全性如何；2)对抗性攻击能否被设计为同时攻击多个任务；3)任务共享和对抗性训练是否提高了多任务模型对对抗性攻击的健壮性？在本文中，我们通过仔细的分析和严谨的实验回答了这些问题。首先，我们提出了单任务白盒攻击的自适应方法，并分析了它们的固有缺陷。然后，我们提出了一种新的攻击框架--动态梯度平衡攻击(DGBA)。我们的框架将攻击多任务模型的问题归结为一个基于平均相对损失变化的优化问题，该问题可以通过将问题近似为一个整数线性规划问题来解决。对两个流行的MTL基准NYUv2和Tiny-Taxonomy进行了广泛的评估，证明了DGBA相对于NAIVE多任务攻击基线在干净和恶意训练的多任务模型上的有效性。结果还揭示了通过在任务间共享参数来提高任务精度和由于参数共享增加攻击可传递性而削弱模型稳健性之间的基本权衡。Dgba是开源的，可在https://github.com/zhanglijun95/MTLAttack-DGBA.上获得



## **45. Exploring Adversarial Robustness of Vision Transformers in the Spectral Perspective**

从频谱角度探讨视觉转换器的对抗稳健性 cs.CV

Accepted in IEEE/CVF Winter Conference on Applications of Computer  Vision (WACV) 2024

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2208.09602v2) [paper-pdf](http://arxiv.org/pdf/2208.09602v2)

**Authors**: Gihyun Kim, Juyeop Kim, Jong-Seok Lee

**Abstract**: The Vision Transformer has emerged as a powerful tool for image classification tasks, surpassing the performance of convolutional neural networks (CNNs). Recently, many researchers have attempted to understand the robustness of Transformers against adversarial attacks. However, previous researches have focused solely on perturbations in the spatial domain. This paper proposes an additional perspective that explores the adversarial robustness of Transformers against frequency-selective perturbations in the spectral domain. To facilitate comparison between these two domains, an attack framework is formulated as a flexible tool for implementing attacks on images in the spatial and spectral domains. The experiments reveal that Transformers rely more on phase and low frequency information, which can render them more vulnerable to frequency-selective attacks than CNNs. This work offers new insights into the properties and adversarial robustness of Transformers.

摘要: 视觉转换器已经成为一种强大的图像分类工具，其性能超过了卷积神经网络(CNN)。最近，许多研究人员试图了解变形金刚对对手攻击的健壮性。然而，以往的研究主要集中在空间域的扰动上。本文提出了一个额外的视角，探讨了在谱域中变压器对频率选择性扰动的对抗稳健性。为了便于这两个域之间的比较，提出了一种攻击框架，作为在空域和谱域对图像实施攻击的灵活工具。实验表明，变形金刚更依赖于相位和低频信息，这使得它们比CNN更容易受到频率选择性攻击。这项工作为了解变形金刚的特性和对抗健壮性提供了新的见解。



## **46. LogoStyleFool: Vitiating Video Recognition Systems via Logo Style Transfer**

LogoStyleFool：通过徽标样式转换破坏视频识别系统 cs.CV

13 pages, 3 figures. Accepted to AAAI 2024

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09935v1) [paper-pdf](http://arxiv.org/pdf/2312.09935v1)

**Authors**: Yuxin Cao, Ziyu Zhao, Xi Xiao, Derui Wang, Minhui Xue, Jin Lu

**Abstract**: Video recognition systems are vulnerable to adversarial examples. Recent studies show that style transfer-based and patch-based unrestricted perturbations can effectively improve attack efficiency. These attacks, however, face two main challenges: 1) Adding large stylized perturbations to all pixels reduces the naturalness of the video and such perturbations can be easily detected. 2) Patch-based video attacks are not extensible to targeted attacks due to the limited search space of reinforcement learning that has been widely used in video attacks recently. In this paper, we focus on the video black-box setting and propose a novel attack framework named LogoStyleFool by adding a stylized logo to the clean video. We separate the attack into three stages: style reference selection, reinforcement-learning-based logo style transfer, and perturbation optimization. We solve the first challenge by scaling down the perturbation range to a regional logo, while the second challenge is addressed by complementing an optimization stage after reinforcement learning. Experimental results substantiate the overall superiority of LogoStyleFool over three state-of-the-art patch-based attacks in terms of attack performance and semantic preservation. Meanwhile, LogoStyleFool still maintains its performance against two existing patch-based defense methods. We believe that our research is beneficial in increasing the attention of the security community to such subregional style transfer attacks.

摘要: 视频识别系统很容易受到敌意例子的攻击。最近的研究表明，基于风格迁移和基于补丁的无限制扰动可以有效地提高攻击效率。然而，这些攻击面临两个主要挑战：1)向所有像素添加大的风格化扰动会降低视频的自然度，并且这种扰动很容易被检测到。2)基于补丁的视频攻击不能扩展到有针对性的攻击，因为强化学习的搜索空间有限，这是近年来在视频攻击中广泛使用的。本文针对视频黑盒的设置，通过在干净的视频中添加一个风格化的标识，提出了一种新的攻击框架--LogoStyleFool。我们将攻击分为三个阶段：样式参考选择、基于强化学习的标识样式迁移和扰动优化。我们通过将扰动范围缩小到区域标志来解决第一个挑战，而第二个挑战是通过在强化学习后补充优化阶段来解决的。实验结果表明，在攻击性能和语义保持方面，LogoStyleFool在攻击性能和语义保持方面都优于三种最先进的基于补丁的攻击。同时，与现有的两种基于补丁的防御方法相比，LogoStyleFool仍然保持其性能。我们认为，我们的研究有助于提高安全界对这种次区域风格的转移袭击的关注。



## **47. A Game-theoretic Framework for Privacy-preserving Federated Learning**

保护隐私的联邦学习的博弈论框架 cs.LG

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2304.05836v2) [paper-pdf](http://arxiv.org/pdf/2304.05836v2)

**Authors**: Xiaojin Zhang, Lixin Fan, Siwei Wang, Wenjie Li, Kai Chen, Qiang Yang

**Abstract**: In federated learning, benign participants aim to optimize a global model collaboratively. However, the risk of \textit{privacy leakage} cannot be ignored in the presence of \textit{semi-honest} adversaries. Existing research has focused either on designing protection mechanisms or on inventing attacking mechanisms. While the battle between defenders and attackers seems never-ending, we are concerned with one critical question: is it possible to prevent potential attacks in advance? To address this, we propose the first game-theoretic framework that considers both FL defenders and attackers in terms of their respective payoffs, which include computational costs, FL model utilities, and privacy leakage risks. We name this game the federated learning privacy game (FLPG), in which neither defenders nor attackers are aware of all participants' payoffs.   To handle the \textit{incomplete information} inherent in this situation, we propose associating the FLPG with an \textit{oracle} that has two primary responsibilities. First, the oracle provides lower and upper bounds of the payoffs for the players. Second, the oracle acts as a correlation device, privately providing suggested actions to each player. With this novel framework, we analyze the optimal strategies of defenders and attackers. Furthermore, we derive and demonstrate conditions under which the attacker, as a rational decision-maker, should always follow the oracle's suggestion \textit{not to attack}.

摘要: 在联合学习中，良性参与者的目标是协作优化全球模型。然而，在存在半诚实的对手的情况下，隐私泄露的风险是不容忽视的。现有的研究要么集中在设计保护机制上，要么集中在发明攻击机制上。虽然防御者和攻击者之间的战斗似乎永无止境，但我们关心的是一个关键问题：是否有可能提前防止潜在的攻击？为了解决这一问题，我们提出了第一个博弈论框架，该框架考虑了FL防御者和攻击者各自的收益，其中包括计算成本、FL模型效用和隐私泄露风险。我们将这款游戏命名为联邦学习隐私游戏(FLPG)，在该游戏中，防御者和攻击者都不知道所有参与者的收益。为了处理这种情况下固有的不完整信息，我们建议将FLPG与具有两个主要职责的\textit{Oracle}相关联。首先，先知为玩家提供了收益的上下限。其次，先知充当了关联设备，私下向每个玩家提供建议的动作。在此框架下，我们分析了防御者和攻击者的最优策略。此外，我们还推导并证明了攻击者作为理性决策者应始终遵循神谕的建议的条件。



## **48. Categorical composable cryptography: extended version**

范畴可合成密码学：扩展版本 cs.CR

Extended version of arXiv:2105.05949 which appeared in FoSSaCS 2022

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2208.13232v4) [paper-pdf](http://arxiv.org/pdf/2208.13232v4)

**Authors**: Anne Broadbent, Martti Karvonen

**Abstract**: We formalize the simulation paradigm of cryptography in terms of category theory and show that protocols secure against abstract attacks form a symmetric monoidal category, thus giving an abstract model of composable security definitions in cryptography. Our model is able to incorporate computational security, set-up assumptions and various attack models such as colluding or independently acting subsets of adversaries in a modular, flexible fashion. We conclude by using string diagrams to rederive the security of the one-time pad, correctness of Diffie-Hellman key exchange and no-go results concerning the limits of bipartite and tripartite cryptography, ruling out e.g., composable commitments and broadcasting. On the way, we exhibit two categorical constructions of resource theories that might be of independent interest: one capturing resources shared among multiple parties and one capturing resource conversions that succeed asymptotically.

摘要: 我们用范畴理论形式化了密码学的模拟范型，证明了对抽象攻击安全的协议形成了对称的么半范畴，从而给出了密码学中可组合安全定义的抽象模型。我们的模型能够以模块化、灵活的方式结合计算安全性、设置假设和各种攻击模型，例如串通或独立行动的对手子集。最后，我们使用字符串图重新推导了一次性密钥的安全性，Diffie-Hellman密钥交换的正确性，以及关于二方和三方密码术限制的不可行结果，排除了例如可组合承诺和广播。在此过程中，我们展示了两种可能独立感兴趣的资源理论范畴结构：一种是捕获多方共享的资源，另一种是捕获渐近成功的资源转换。



## **49. FlowMur: A Stealthy and Practical Audio Backdoor Attack with Limited Knowledge**

FlowMur：一种隐蔽实用的有限知识音频后门攻击 cs.CR

To appear at lEEE Symposium on Security & Privacy (Oakland) 2024

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09665v1) [paper-pdf](http://arxiv.org/pdf/2312.09665v1)

**Authors**: Jiahe Lan, Jie Wang, Baochen Yan, Zheng Yan, Elisa Bertino

**Abstract**: Speech recognition systems driven by DNNs have revolutionized human-computer interaction through voice interfaces, which significantly facilitate our daily lives. However, the growing popularity of these systems also raises special concerns on their security, particularly regarding backdoor attacks. A backdoor attack inserts one or more hidden backdoors into a DNN model during its training process, such that it does not affect the model's performance on benign inputs, but forces the model to produce an adversary-desired output if a specific trigger is present in the model input. Despite the initial success of current audio backdoor attacks, they suffer from the following limitations: (i) Most of them require sufficient knowledge, which limits their widespread adoption. (ii) They are not stealthy enough, thus easy to be detected by humans. (iii) Most of them cannot attack live speech, reducing their practicality. To address these problems, in this paper, we propose FlowMur, a stealthy and practical audio backdoor attack that can be launched with limited knowledge. FlowMur constructs an auxiliary dataset and a surrogate model to augment adversary knowledge. To achieve dynamicity, it formulates trigger generation as an optimization problem and optimizes the trigger over different attachment positions. To enhance stealthiness, we propose an adaptive data poisoning method according to Signal-to-Noise Ratio (SNR). Furthermore, ambient noise is incorporated into the process of trigger generation and data poisoning to make FlowMur robust to ambient noise and improve its practicality. Extensive experiments conducted on two datasets demonstrate that FlowMur achieves high attack performance in both digital and physical settings while remaining resilient to state-of-the-art defenses. In particular, a human study confirms that triggers generated by FlowMur are not easily detected by participants.

摘要: 由DNN驱动的语音识别系统通过语音接口使人机交互发生了革命性的变化，极大地方便了我们的日常生活。然而，这些系统越来越受欢迎，也引发了对其安全性的特别关注，特别是关于后门攻击的问题。后门攻击在其训练过程中将一个或多个隐藏的后门插入到DNN模型中，使得它不会影响模型在良性输入上的性能，但如果模型输入中存在特定触发器，则迫使模型产生对手期望的输出。尽管目前的音频后门攻击取得了初步的成功，但它们受到以下限制：(I)大多数音频后门攻击需要足够的知识，这限制了它们的广泛采用。(Ii)它们的隐蔽性不够，因此很容易被人发现。(3)他们中的大多数不能攻击现场演讲，降低了他们的实用性。为了解决这些问题，在本文中，我们提出了FlowMur，一种隐蔽而实用的音频后门攻击，可以在有限的知识下发起。FlowMur构建了一个辅助数据集和一个代理模型来增强对手的知识。为了实现动态化，它将触发器的生成描述为一个优化问题，并在不同的附着位置上对触发器进行优化。为了提高隐蔽性，提出了一种基于信噪比的自适应数据毒化方法。此外，在触发产生和数据毒化过程中引入了环境噪声，使FlowMur对环境噪声具有较强的鲁棒性，提高了其实用性。在两个数据集上进行的广泛实验表明，FlowMur在数字和物理环境中都实现了高攻击性能，同时对最先进的防御保持了弹性。特别是，一项人类研究证实，FlowMur产生的触发因素不容易被参与者检测到。



## **50. Unsupervised and Supervised learning by Dense Associative Memory under replica symmetry breaking**

副本对称破缺下稠密联想记忆的无监督和有监督学习 cond-mat.dis-nn

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09638v1) [paper-pdf](http://arxiv.org/pdf/2312.09638v1)

**Authors**: Linda Albanese, Andrea Alessandrelli, Alessia Annibale, Adriano Barra

**Abstract**: Statistical mechanics of spin glasses is one of the main strands toward a comprehension of information processing by neural networks and learning machines. Tackling this approach, at the fairly standard replica symmetric level of description, recently Hebbian attractor networks with multi-node interactions (often called Dense Associative Memories) have been shown to outperform their classical pairwise counterparts in a number of tasks, from their robustness against adversarial attacks and their capability to work with prohibitively weak signals to their supra-linear storage capacities. Focusing on mathematical techniques more than computational aspects, in this paper we relax the replica symmetric assumption and we derive the one-step broken-replica-symmetry picture of supervised and unsupervised learning protocols for these Dense Associative Memories: a phase diagram in the space of the control parameters is achieved, independently, both via the Parisi's hierarchy within then replica trick as well as via the Guerra's telescope within the broken-replica interpolation. Further, an explicit analytical investigation is provided to deepen both the big-data and ground state limits of these networks as well as a proof that replica symmetry breaking does not alter the thresholds for learning and slightly increases the maximal storage capacity. Finally the De Almeida and Thouless line, depicting the onset of instability of a replica symmetric description, is also analytically derived highlighting how, crossed this boundary, the broken replica description should be preferred.

摘要: 自旋玻璃的统计力学是通过神经网络和学习机理解信息处理的主要途径之一。为了处理这种方法，在相当标准的副本对称描述水平上，最近具有多节点交互的Hebbian吸引子网络(通常被称为密集联想记忆)已经被证明在许多任务中表现出比它们的经典成对同行更好的性能，从它们对对手攻击的健壮性和它们处理令人望而却步的弱信号的能力到它们的超线性存储能力。本文更多地关注数学技术而不是计算方面，放松了副本对称假设，导出了这些密集联想记忆的监督和非监督学习协议的一步破坏副本对称图：控制参数空间中的相图既可以通过副本技巧中的Parisi层次结构独立获得，也可以通过破碎副本内插中的Guera望远镜独立获得。此外，还提供了一个明确的分析研究，以加深这些网络的大数据和基态限制，并证明了副本对称性破坏不会改变学习阈值，并略微增加了最大存储容量。最后，描述复制品对称描述的不稳定性开始的de Almeida和Thouless线也被解析地推导出来，突出了如何跨越这一边界，破碎的复制品描述应该是首选的。



