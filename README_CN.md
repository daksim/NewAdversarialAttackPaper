# Latest Adversarial Attack Papers
**update at 2025-02-10 11:16:32**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. ADAPT to Robustify Prompt Tuning Vision Transformers**

ADAPT以Robustify提示调整视觉变形金刚 cs.LG

Published in Transactions on Machine Learning Research (2025)

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2403.13196v2) [paper-pdf](http://arxiv.org/pdf/2403.13196v2)

**Authors**: Masih Eskandar, Tooba Imtiaz, Zifeng Wang, Jennifer Dy

**Abstract**: The performance of deep models, including Vision Transformers, is known to be vulnerable to adversarial attacks. Many existing defenses against these attacks, such as adversarial training, rely on full-model fine-tuning to induce robustness in the models. These defenses require storing a copy of the entire model, that can have billions of parameters, for each task. At the same time, parameter-efficient prompt tuning is used to adapt large transformer-based models to downstream tasks without the need to save large copies. In this paper, we examine parameter-efficient prompt tuning of Vision Transformers for downstream tasks under the lens of robustness. We show that previous adversarial defense methods, when applied to the prompt tuning paradigm, suffer from gradient obfuscation and are vulnerable to adaptive attacks. We introduce ADAPT, a novel framework for performing adaptive adversarial training in the prompt tuning paradigm. Our method achieves competitive robust accuracy of ~40% w.r.t. SOTA robustness methods using full-model fine-tuning, by tuning only ~1% of the number of parameters.

摘要: 众所周知，包括Vision Transformers在内的深度模型的性能很容易受到对手的攻击。许多现有的针对这些攻击的防御，如对抗性训练，都依赖于全模型微调来诱导模型的健壮性。这些防御需要为每个任务存储整个模型的副本，该副本可以具有数十亿个参数。同时，使用参数高效的即时调整来使大型基于变压器的模型适应下游任务，而不需要保存大量副本。在这篇文章中，我们研究了稳健性镜头下的视觉变形器的参数高效的下游任务的快速调整。我们表明，以前的对抗性防御方法，当应用于即时调整范例时，遭受梯度混淆，并且容易受到适应性攻击。我们介绍了Adapt，这是一种在即时调整范式中执行自适应对抗性训练的新框架。我们的方法获得了~40%的W.r.t.SOTA稳健性方法采用全模型微调，只需调整~1%的参数即可。



## **2. Do Unlearning Methods Remove Information from Language Model Weights?**

取消学习方法会从语言模型权重中删除信息吗？ cs.LG

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2410.08827v3) [paper-pdf](http://arxiv.org/pdf/2410.08827v3)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods for information learned during pretraining, revealing the limitations of these methods in removing information from the model weights. Our results also suggest that unlearning evaluations that measure unlearning robustness on information learned during an additional fine-tuning phase may overestimate robustness compared to evaluations that attempt to unlearn information learned during pretraining.

摘要: 大型语言模型在如何执行网络安全攻击、创建生物武器和操纵人类方面的知识构成了误用的风险。以前的工作已经提出了忘记这一知识的方法。从历史上看，人们一直不清楚遗忘技术是在移除模型重量中的信息，还是只是增加了获取信息的难度。为了分离这两个目标，我们提出了一种对抗性评估方法来测试从模型权重中移除信息的情况：我们允许攻击者访问一些应该被移除的事实，并且使用这些事实，攻击者试图从相同的分布中恢复无法从可访问的事实中猜测的其他事实。结果表明，对可访问的事实进行微调可以恢复88%的预忘学习准确率，当应用于现有的遗忘方法时，这些方法在去除模型权重中的信息方面存在局限性。我们的结果还表明，与试图忘却在预训练中学习的信息的评估相比，衡量在额外微调阶段学习到的信息的遗忘健壮性的遗忘评估可能高估了健壮性。



## **3. Federated Learning for Anomaly Detection in Energy Consumption Data: Assessing the Vulnerability to Adversarial Attacks**

用于能源消耗数据异常检测的联邦学习：评估对抗性攻击的脆弱性 cs.LG

12th IEEE Conference on Technologies for Sustainability

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.05041v1) [paper-pdf](http://arxiv.org/pdf/2502.05041v1)

**Authors**: Yohannis Kifle Telila, Damitha Senevirathne, Dumindu Tissera, Apurva Narayan, Miriam A. M. Capretz, Katarina Grolinger

**Abstract**: Anomaly detection is crucial in the energy sector to identify irregular patterns indicating equipment failures, energy theft, or other issues. Machine learning techniques for anomaly detection have achieved great success, but are typically centralized, involving sharing local data with a central server which raises privacy and security concerns. Federated Learning (FL) has been gaining popularity as it enables distributed learning without sharing local data. However, FL depends on neural networks, which are vulnerable to adversarial attacks that manipulate data, leading models to make erroneous predictions. While adversarial attacks have been explored in the image domain, they remain largely unexplored in time series problems, especially in the energy domain. Moreover, the effect of adversarial attacks in the FL setting is also mostly unknown. This paper assesses the vulnerability of FL-based anomaly detection in energy data to adversarial attacks. Specifically, two state-of-the-art models, Long Short Term Memory (LSTM) and Transformers, are used to detect anomalies in an FL setting, and two white-box attack methods, Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), are employed to perturb the data. The results show that FL is more sensitive to PGD attacks than to FGSM attacks, attributed to PGD's iterative nature, resulting in an accuracy drop of over 10% even with naive, weaker attacks. Moreover, FL is more affected by these attacks than centralized learning, highlighting the need for defense mechanisms in FL.

摘要: 异常检测在能源部门至关重要，可以识别表明设备故障、能源盗窃或其他问题的不规则模式。用于异常检测的机器学习技术取得了巨大成功，但通常是集中式的，涉及与中央服务器共享本地数据，这会引起隐私和安全问题。联合学习(FL)由于能够在不共享本地数据的情况下实现分布式学习而越来越受欢迎。然而，FL依赖于神经网络，而神经网络容易受到操纵数据的敌意攻击，导致模型做出错误的预测。虽然对抗性攻击已经在图像领域得到了探索，但在时间序列问题上，特别是在能量领域，它们在很大程度上仍未被探索。此外，在外语环境下，对抗性攻击的效果也大多是未知的。本文评估了基于FL的能量数据异常检测在敌意攻击下的脆弱性。具体地说，两个最新的模型，长期短期记忆(LSTM)和变形金刚，被用来检测FL环境中的异常，两种白盒攻击方法，快速梯度符号方法(FGSM)和投影梯度下降方法(PGD)，被用来扰动数据。结果表明，FL对PGD攻击比FGSM攻击更敏感，这归因于PGD的迭代性质，导致即使是幼稚、较弱的攻击，准确率也会下降10%以上。此外，外语受这些攻击的影响比集中学习更大，这突显了外语学习中防御机制的必要性。



## **4. Robust Graph Learning Against Adversarial Evasion Attacks via Prior-Free Diffusion-Based Structure Purification**

通过无先验扩散结构纯化来对抗对抗规避攻击的鲁棒图学习 cs.LG

Accepted for poster at WWW 2025

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.05000v1) [paper-pdf](http://arxiv.org/pdf/2502.05000v1)

**Authors**: Jiayi Luo, Qingyun Sun, Haonan Yuan, Xingcheng Fu, Jianxin Li

**Abstract**: Adversarial evasion attacks pose significant threats to graph learning, with lines of studies that have improved the robustness of Graph Neural Networks (GNNs). However, existing works rely on priors about clean graphs or attacking strategies, which are often heuristic and inconsistent. To achieve robust graph learning over different types of evasion attacks and diverse datasets, we investigate this problem from a prior-free structure purification perspective. Specifically, we propose a novel Diffusion-based Structure Purification framework named DiffSP, which creatively incorporates the graph diffusion model to learn intrinsic distributions of clean graphs and purify the perturbed structures by removing adversaries under the direction of the captured predictive patterns without relying on priors. DiffSP is divided into the forward diffusion process and the reverse denoising process, during which structure purification is achieved. To avoid valuable information loss during the forward process, we propose an LID-driven nonisotropic diffusion mechanism to selectively inject noise anisotropically. To promote semantic alignment between the clean graph and the purified graph generated during the reverse process, we reduce the generation uncertainty by the proposed graph transfer entropy guided denoising mechanism. Extensive experiments demonstrate the superior robustness of DiffSP against evasion attacks.

摘要: 对抗性逃避攻击对图学习构成了重大威胁，已有一系列研究提高了图神经网络(GNN)的稳健性。然而，现有的工作依赖于关于干净图或攻击策略的先验知识，这些先验知识往往是启发式的和不一致的。为了在不同类型的逃避攻击和不同的数据集上实现稳健的图学习，我们从无先验结构净化的角度研究了这个问题。具体地说，我们提出了一种新的基于扩散的结构净化框架DiffSP，它创造性地结合了图扩散模型来学习干净图的内在分布，并通过在捕获的预测模式的指导下去除对手来净化扰动结构，而不依赖于先验。DiffSP分为正向扩散过程和反向去噪过程，在这两个过程中实现了结构净化。为了避免正演过程中有价值的信息丢失，我们提出了一种盖子驱动的非各向同性扩散机制来选择性地各向异性地注入噪声。为了促进反求过程中生成的清洁图和净化图之间的语义对齐，我们通过提出的图传递熵引导的去噪机制来降低生成的不确定性。大量实验证明了DiffSP对逃避攻击具有良好的健壮性。



## **5. Securing 5G Bootstrapping: A Two-Layer IBS Authentication Protocol**

确保5G引导：两层IBS认证协议 cs.CR

13 pages, 4 figures, 3 tables. This work has been submitted to the  IEEE for possible publication

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04915v1) [paper-pdf](http://arxiv.org/pdf/2502.04915v1)

**Authors**: Yilu Dong, Rouzbeh Behnia, Attila A. Yavuz, Syed Rafiul Hussain

**Abstract**: The lack of authentication during the initial bootstrapping phase between cellular devices and base stations allows attackers to deploy fake base stations and send malicious messages to the devices. These attacks have been a long-existing problem in cellular networks, enabling adversaries to launch denial-of-service (DoS), information leakage, and location-tracking attacks. While some defense mechanisms are introduced in 5G, (e.g., encrypting user identifiers to mitigate IMSI catchers), the initial communication between devices and base stations remains unauthenticated, leaving a critical security gap. To address this, we propose E2IBS, a novel and efficient two-layer identity-based signature scheme designed for seamless integration with existing cellular protocols. We implement E2IBS on an open-source 5G stack and conduct a comprehensive performance evaluation against alternative solutions. Compared to the state-of-the-art Schnorr-HIBS, E2IBS reduces attack surfaces, enables fine-grained lawful interception, and achieves 2x speed in verification, making it a practical solution for securing 5G base station authentication.

摘要: 在蜂窝设备和基站之间的初始引导阶段缺乏身份验证，使得攻击者能够部署伪基站并向设备发送恶意消息。这些攻击是蜂窝网络中长期存在的问题，使攻击者能够发起拒绝服务(DoS)、信息泄漏和位置跟踪攻击。虽然在5G中引入了一些防御机制(例如，加密用户标识以减少IMSI捕获器)，但设备和基站之间的初始通信仍未经过身份验证，留下了一个严重的安全漏洞。为了解决这一问题，我们提出了一种新颖高效的两层基于身份的签名方案E2IBS，旨在与现有的蜂窝协议无缝集成。我们在开源5G堆栈上实施E2IBS，并针对替代解决方案进行全面的性能评估。与最先进的Schnorr-Hibs相比，E2IBS减少了攻击面，实现了细粒度的合法拦截，验证速度达到了2倍，是5G基站认证安全的实用解决方案。



## **6. From Allies to Adversaries: Manipulating LLM Tool-Calling through Adversarial Injection**

从盟友到对手：通过对抗注入操纵LLM工具调用 cs.CR

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2412.10198v2) [paper-pdf](http://arxiv.org/pdf/2412.10198v2)

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang

**Abstract**: Tool-calling has changed Large Language Model (LLM) applications by integrating external tools, significantly enhancing their functionality across diverse tasks. However, this integration also introduces new security vulnerabilities, particularly in the tool scheduling mechanisms of LLM, which have not been extensively studied. To fill this gap, we present ToolCommander, a novel framework designed to exploit vulnerabilities in LLM tool-calling systems through adversarial tool injection. Our framework employs a well-designed two-stage attack strategy. Firstly, it injects malicious tools to collect user queries, then dynamically updates the injected tools based on the stolen information to enhance subsequent attacks. These stages enable ToolCommander to execute privacy theft, launch denial-of-service attacks, and even manipulate business competition by triggering unscheduled tool-calling. Notably, the ASR reaches 91.67% for privacy theft and hits 100% for denial-of-service and unscheduled tool calling in certain cases. Our work demonstrates that these vulnerabilities can lead to severe consequences beyond simple misuse of tool-calling systems, underscoring the urgent need for robust defensive strategies to secure LLM Tool-calling systems.

摘要: 工具调用通过集成外部工具改变了大型语言模型(LLM)应用程序，显著增强了它们在不同任务中的功能。然而，这种集成也引入了新的安全漏洞，特别是在LLM的工具调度机制中，这些漏洞还没有得到广泛的研究。为了填补这一空白，我们提出了一种新的框架，它旨在通过敌意工具注入来利用LLM工具调用系统中的漏洞。我们的框架采用了精心设计的两阶段攻击策略。它首先注入恶意工具来收集用户查询，然后根据窃取的信息动态更新注入的工具，以加强后续攻击。这些阶段使工具指挥官能够执行隐私窃取、发起拒绝服务攻击，甚至通过触发计划外的工具调用来操纵业务竞争。值得注意的是，在某些情况下，隐私窃取的ASR达到91.67%，拒绝服务和非计划工具调用的ASR达到100%。我们的工作表明，这些漏洞可能导致严重后果，而不仅仅是简单地滥用工具调用系统，这突显了迫切需要强大的防御战略来保护LLM工具调用系统。



## **7. DMPA: Model Poisoning Attacks on Decentralized Federated Learning for Model Differences**

DMPA：因模型差异而对去中心联邦学习的模型中毒攻击 cs.LG

8 pages, 3 figures

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04771v1) [paper-pdf](http://arxiv.org/pdf/2502.04771v1)

**Authors**: Chao Feng, Yunlong Li, Yuanzhe Gao, Alberto Huertas Celdrán, Jan von der Assen, Gérôme Bovet, Burkhard Stiller

**Abstract**: Federated learning (FL) has garnered significant attention as a prominent privacy-preserving Machine Learning (ML) paradigm. Decentralized FL (DFL) eschews traditional FL's centralized server architecture, enhancing the system's robustness and scalability. However, these advantages of DFL also create new vulnerabilities for malicious participants to execute adversarial attacks, especially model poisoning attacks. In model poisoning attacks, malicious participants aim to diminish the performance of benign models by creating and disseminating the compromised model. Existing research on model poisoning attacks has predominantly concentrated on undermining global models within the Centralized FL (CFL) paradigm, while there needs to be more research in DFL. To fill the research gap, this paper proposes an innovative model poisoning attack called DMPA. This attack calculates the differential characteristics of multiple malicious client models and obtains the most effective poisoning strategy, thereby orchestrating a collusive attack by multiple participants. The effectiveness of this attack is validated across multiple datasets, with results indicating that the DMPA approach consistently surpasses existing state-of-the-art FL model poisoning attack strategies.

摘要: 联合学习(FL)作为一种重要的隐私保护机器学习(ML)范型已经引起了人们的广泛关注。分散式FL(DFL)避开了传统FL的集中式服务器架构，增强了系统的健壮性和可扩展性。然而，DFL的这些优势也为恶意参与者执行对抗性攻击，特别是模型中毒攻击创造了新的漏洞。在模型中毒攻击中，恶意参与者旨在通过创建和传播受危害的模型来降低良性模型的性能。现有的关于模型中毒攻击的研究主要集中在集中式FL(CFL)范式中破坏全局模型，而在DFL(DFL)中还需要更多的研究。为了填补这一研究空白，本文提出了一种新的中毒攻击模型DMPA。该攻击计算了多个恶意客户端模型的差异特征，获得了最有效的中毒策略，从而策划了多个参与者的合谋攻击。该攻击的有效性在多个数据集上得到了验证，结果表明，DMPA方法始终优于现有的最先进的FL模型中毒攻击策略。



## **8. Real-Time Privacy Risk Measurement with Privacy Tokens for Gradient Leakage**

使用隐私令牌进行实时隐私风险测量以应对梯度泄漏 cs.LG

There is something wrong with the order of Figures 8-11. And I need  to add an experiment with differential privacy quantization mutual  information value

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.02913v3) [paper-pdf](http://arxiv.org/pdf/2502.02913v3)

**Authors**: Jiayang Meng, Tao Huang, Hong Chen, Xin Shi, Qingyu Huang, Chen Hou

**Abstract**: The widespread deployment of deep learning models in privacy-sensitive domains has amplified concerns regarding privacy risks, particularly those stemming from gradient leakage during training. Current privacy assessments primarily rely on post-training attack simulations. However, these methods are inherently reactive, unable to encompass all potential attack scenarios, and often based on idealized adversarial assumptions. These limitations underscore the need for proactive approaches to privacy risk assessment during the training process. To address this gap, we propose the concept of privacy tokens, which are derived directly from private gradients during training. Privacy tokens encapsulate gradient features and, when combined with data features, offer valuable insights into the extent of private information leakage from training data, enabling real-time measurement of privacy risks without relying on adversarial attack simulations. Additionally, we employ Mutual Information (MI) as a robust metric to quantify the relationship between training data and gradients, providing precise and continuous assessments of privacy leakage throughout the training process. Extensive experiments validate our framework, demonstrating the effectiveness of privacy tokens and MI in identifying and quantifying privacy risks. This proactive approach marks a significant advancement in privacy monitoring, promoting the safer deployment of deep learning models in sensitive applications.

摘要: 深度学习模型在隐私敏感领域的广泛部署加剧了人们对隐私风险的担忧，特别是培训期间梯度泄漏造成的风险。目前的隐私评估主要依赖于训练后的攻击模拟。然而，这些方法本质上是被动的，无法涵盖所有潜在的攻击场景，并且通常基于理想化的对抗性假设。这些限制强调了在培训过程中对隐私风险评估采取积极主动的方法的必要性。为了弥补这一差距，我们提出了隐私令牌的概念，它直接从训练过程中的隐私梯度派生出来。隐私令牌封装了梯度特征，当与数据特征相结合时，可以提供对训练数据中私人信息泄漏程度的有价值的见解，从而能够实时测量隐私风险，而不需要依赖对抗性攻击模拟。此外，我们使用相互信息(MI)作为一个稳健的度量来量化训练数据和梯度之间的关系，在整个训练过程中提供对隐私泄露的准确和连续的评估。大量的实验验证了我们的框架，证明了隐私令牌和MI在识别和量化隐私风险方面的有效性。这种主动的方法标志着隐私监控方面的重大进步，促进了在敏感应用程序中更安全地部署深度学习模型。



## **9. Mechanistic Understandings of Representation Vulnerabilities and Engineering Robust Vision Transformers**

对表示漏洞的机械理解和工程鲁棒视觉转换器 cs.CV

10 pages, 5 figures

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04679v1) [paper-pdf](http://arxiv.org/pdf/2502.04679v1)

**Authors**: Chashi Mahiul Islam, Samuel Jacob Chacko, Mao Nishino, Xiuwen Liu

**Abstract**: While transformer-based models dominate NLP and vision applications, their underlying mechanisms to map the input space to the label space semantically are not well understood. In this paper, we study the sources of known representation vulnerabilities of vision transformers (ViT), where perceptually identical images can have very different representations and semantically unrelated images can have the same representation. Our analysis indicates that imperceptible changes to the input can result in significant representation changes, particularly in later layers, suggesting potential instabilities in the performance of ViTs. Our comprehensive study reveals that adversarial effects, while subtle in early layers, propagate and amplify through the network, becoming most pronounced in middle to late layers. This insight motivates the development of NeuroShield-ViT, a novel defense mechanism that strategically neutralizes vulnerable neurons in earlier layers to prevent the cascade of adversarial effects. We demonstrate NeuroShield-ViT's effectiveness across various attacks, particularly excelling against strong iterative attacks, and showcase its remarkable zero-shot generalization capabilities. Without fine-tuning, our method achieves a competitive accuracy of 77.8% on adversarial examples, surpassing conventional robustness methods. Our results shed new light on how adversarial effects propagate through ViT layers, while providing a promising approach to enhance the robustness of vision transformers against adversarial attacks. Additionally, they provide a promising approach to enhance the robustness of vision transformers against adversarial attacks.

摘要: 虽然基于转换器的模型在NLP和VISION应用中占据主导地位，但它们将输入空间语义映射到标签空间的底层机制还没有得到很好的理解。在这篇文章中，我们研究了视觉转换器(VIT)的已知表征漏洞的来源，其中感知相同的图像可以具有非常不同的表征，而语义无关的图像可以具有相同的表征。我们的分析表明，对输入的不可察觉的变化会导致显著的表示变化，特别是在较晚的层中，这表明VITS的性能存在潜在的不稳定性。我们的综合研究表明，对抗性效应，虽然在早期层微妙，但通过网络传播和放大，在中后期变得最明显。这种洞察力推动了NeuroShield-Vit的发展，这是一种新的防御机制，战略性地中和早期层中的脆弱神经元，以防止级联的对抗性效应。我们展示了NeuroShield-VIT在各种攻击中的有效性，特别是在对抗强大的迭代攻击方面的出色表现，并展示了其非凡的零命中泛化能力。在没有微调的情况下，我们的方法在对抗性样本上的准确率达到了77.8%，超过了传统的稳健性方法。我们的结果揭示了对抗性效应是如何通过VIT层传播的，同时提供了一种有希望的方法来增强视觉转换器对抗对抗性攻击的健壮性。此外，它们还提供了一种很有前途的方法来增强视觉转换器对对手攻击的稳健性。



## **10. Confidence Elicitation: A New Attack Vector for Large Language Models**

信心激发：大型语言模型的新攻击载体 cs.LG

Published in ICLR 2025. The code is publicly available at  https://github.com/Aniloid2/Confidence_Elicitation_Attacks

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04643v1) [paper-pdf](http://arxiv.org/pdf/2502.04643v1)

**Authors**: Brian Formento, Chuan Sheng Foo, See-Kiong Ng

**Abstract**: A fundamental issue in deep learning has been adversarial robustness. As these systems have scaled, such issues have persisted. Currently, large language models (LLMs) with billions of parameters suffer from adversarial attacks just like their earlier, smaller counterparts. However, the threat models have changed. Previously, having gray-box access, where input embeddings or output logits/probabilities were visible to the user, might have been reasonable. However, with the introduction of closed-source models, no information about the model is available apart from the generated output. This means that current black-box attacks can only utilize the final prediction to detect if an attack is successful. In this work, we investigate and demonstrate the potential of attack guidance, akin to using output probabilities, while having only black-box access in a classification setting. This is achieved through the ability to elicit confidence from the model. We empirically show that the elicited confidence is calibrated and not hallucinated for current LLMs. By minimizing the elicited confidence, we can therefore increase the likelihood of misclassification. Our new proposed paradigm demonstrates promising state-of-the-art results on three datasets across two models (LLaMA-3-8B-Instruct and Mistral-7B-Instruct-V0.3) when comparing our technique to existing hard-label black-box attack methods that introduce word-level substitutions.

摘要: 深度学习的一个基本问题是对手的稳健性。随着这些系统的规模扩大，这样的问题一直存在。目前，具有数十亿个参数的大型语言模型(LLM)就像它们早期的较小对应模型一样，受到对手攻击。然而，威胁模型已经发生了变化。以前，拥有灰箱访问，其中输入嵌入或输出日志/概率对用户可见，可能是合理的。然而，随着闭源模型的引入，除了生成的输出之外，没有关于模型的信息可用。这意味着当前的黑盒攻击只能利用最终预测来检测攻击是否成功。在这项工作中，我们调查和演示了攻击指导的潜力，类似于使用输出概率，而在分类设置中只有黑盒访问。这是通过从模型中获得信心的能力来实现的。我们的经验表明，对于当前的LLM来说，引发的信心是经过校准的，而不是幻觉的。因此，通过将引起的置信度降至最低，我们可以增加错误分类的可能性。我们提出的新范式在两个模型(骆驼-3-8B-指令和Mistral-7B-指令-V0.3)的三个数据集上展示了有希望的最先进结果，当将我们的技术与现有的引入词级替换的硬标签黑盒攻击方法进行比较时。



## **11. Regularized Robustly Reliable Learners and Instance Targeted Attacks**

正规的鲁棒可靠的学习者和实例有针对性的攻击 cs.LG

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2410.10572v2) [paper-pdf](http://arxiv.org/pdf/2410.10572v2)

**Authors**: Avrim Blum, Donya Saless

**Abstract**: Instance-targeted data poisoning attacks, where an adversary corrupts a training set to induce errors on specific test points, have raised significant concerns. Balcan et al (2022) proposed an approach to addressing this challenge by defining a notion of robustly-reliable learners that provide per-instance guarantees of correctness under well-defined assumptions, even in the presence of data poisoning attacks. They then give a generic optimal (but computationally inefficient) robustly reliable learner as well as a computationally efficient algorithm for the case of linear separators over log-concave distributions.   In this work, we address two challenges left open by Balcan et al (2022). The first is that the definition of robustly-reliable learners in Balcan et al (2022) becomes vacuous for highly-flexible hypothesis classes: if there are two classifiers h_0, h_1 \in H both with zero error on the training set such that h_0(x) \neq h_1(x), then a robustly-reliable learner must abstain on x. We address this problem by defining a modified notion of regularized robustly-reliable learners that allows for nontrivial statements in this case. The second is that the generic algorithm of Balcan et al (2022) requires re-running an ERM oracle (essentially, retraining the classifier) on each test point x, which is generally impractical even if ERM can be implemented efficiently. To tackle this problem, we show that at least in certain interesting cases we can design algorithms that can produce their outputs in time sublinear in training time, by using techniques from dynamic algorithm design.

摘要: 针对实例的数据中毒攻击，即对手破坏训练集以在特定测试点上引发错误，已经引起了严重的担忧。Balcan等人(2022)提出了一种应对这一挑战的方法，定义了稳健可靠的学习者的概念，即使在存在数据中毒攻击的情况下，也可以在定义明确的假设下提供逐个实例的正确性保证。然后，对于对数凹分布上的线性分隔符的情况，他们给出了一个通用的最优(但计算效率低)鲁棒可靠的学习器以及一个计算高效的算法。在这项工作中，我们解决了Balcan等人(2022)留下的两个挑战。首先，对于高度灵活的假设类，Balcan等人(2022)中的稳健可靠学习者的定义变得空洞：如果在训练集上存在两个都是零误差的分类器h_0，h_1\in H，使得h_0(X)\neq h_1(X)，那么稳健可靠的学习者必须在x上弃权。我们通过定义一个修正的正则化稳健可靠学习者的概念来解决这个问题，它允许在这种情况下非平凡的陈述。其次，Balcan等人(2022)的通用算法需要在每个测试点x上重新运行ERM预言(本质上是重新训练分类器)，即使ERM可以有效地实施，这通常也是不切实际的。为了解决这个问题，我们证明了，至少在某些有趣的情况下，我们可以设计算法，通过使用动态算法设计的技术，在训练时间内产生时间上的次线性输出。



## **12. Adapting to Evolving Adversaries with Regularized Continual Robust Training**

通过定期的持续稳健训练适应不断发展的对手 cs.LG

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04248v1) [paper-pdf](http://arxiv.org/pdf/2502.04248v1)

**Authors**: Sihui Dai, Christian Cianfarani, Arjun Bhagoji, Vikash Sehwag, Prateek Mittal

**Abstract**: Robust training methods typically defend against specific attack types, such as Lp attacks with fixed budgets, and rarely account for the fact that defenders may encounter new attacks over time. A natural solution is to adapt the defended model to new adversaries as they arise via fine-tuning, a method which we call continual robust training (CRT). However, when implemented naively, fine-tuning on new attacks degrades robustness on previous attacks. This raises the question: how can we improve the initial training and fine-tuning of the model to simultaneously achieve robustness against previous and new attacks? We present theoretical results which show that the gap in a model's robustness against different attacks is bounded by how far each attack perturbs a sample in the model's logit space, suggesting that regularizing with respect to this logit space distance can help maintain robustness against previous attacks. Extensive experiments on 3 datasets (CIFAR-10, CIFAR-100, and ImageNette) and over 100 attack combinations demonstrate that the proposed regularization improves robust accuracy with little overhead in training time. Our findings and open-source code lay the groundwork for the deployment of models robust to evolving attacks.

摘要: 健壮的训练方法通常可以防御特定的攻击类型，如固定预算的LP攻击，很少考虑到防御者可能会随着时间的推移遇到新的攻击。一个自然的解决方案是，当新的对手出现时，通过微调使防御模型适应他们，这是一种我们称为持续稳健训练(CRT)的方法。然而，如果实施得很幼稚，对新攻击的微调会降低对以前攻击的健壮性。这就提出了一个问题：我们如何改进模型的初始训练和微调，以同时实现对以前和新攻击的健壮性？我们给出的理论结果表明，模型对不同攻击的稳健性的差距取决于每次攻击对模型Logit空间中样本的扰动程度，这表明关于该Logit空间距离的正则化可以帮助保持对先前攻击的健壮性。在3个数据集(CIFAR-10、CIFAR-100和ImageNette)和100多个攻击组合上的大量实验表明，所提出的正则化方法在很小的训练时间开销下提高了稳健的准确率。我们的发现和开源代码为部署对不断演变的攻击具有强大功能的模型奠定了基础。



## **13. Provably Robust Explainable Graph Neural Networks against Graph Perturbation Attacks**

抗图扰动攻击的可证明鲁棒可解释图神经网络 cs.CR

Accepted by ICLR 2025

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04224v1) [paper-pdf](http://arxiv.org/pdf/2502.04224v1)

**Authors**: Jiate Li, Meng Pang, Yun Dong, Jinyuan Jia, Binghui Wang

**Abstract**: Explaining Graph Neural Network (XGNN) has gained growing attention to facilitate the trust of using GNNs, which is the mainstream method to learn graph data. Despite their growing attention, Existing XGNNs focus on improving the explanation performance, and its robustness under attacks is largely unexplored. We noticed that an adversary can slightly perturb the graph structure such that the explanation result of XGNNs is largely changed. Such vulnerability of XGNNs could cause serious issues particularly in safety/security-critical applications. In this paper, we take the first step to study the robustness of XGNN against graph perturbation attacks, and propose XGNNCert, the first provably robust XGNN. Particularly, our XGNNCert can provably ensure the explanation result for a graph under the worst-case graph perturbation attack is close to that without the attack, while not affecting the GNN prediction, when the number of perturbed edges is bounded. Evaluation results on multiple graph datasets and GNN explainers show the effectiveness of XGNNCert.

摘要: 解释图神经网络(XGNN)是学习图数据的主流方法，为了提高网络的可信性，它得到了越来越多的关注。尽管XGNN越来越受到关注，但现有的XGNN专注于提高解释性能，其在攻击下的健壮性在很大程度上还没有被探索。我们注意到，攻击者可以稍微扰乱图的结构，从而在很大程度上改变XGNN的解释结果。XGNN的这种漏洞可能会造成严重问题，特别是在安全/安保关键应用程序中。在本文中，我们首先研究了XGNN对图扰动攻击的健壮性，并提出了第一个可证明的健壮性XGNN。特别地，当扰动边个数有界时，我们的XGNNCert能够证明在最坏情况图扰动攻击下对图的解释结果接近于未被攻击时的解释结果，而不影响GNN预测。在多个图形数据集和GNN解释器上的评估结果表明了XGNNCert的有效性。



## **14. "Short-length" Adversarial Training Helps LLMs Defend "Long-length" Jailbreak Attacks: Theoretical and Empirical Evidence**

“短期”对抗性培训帮助法学硕士防御“长期”越狱攻击：理论和经验证据 cs.LG

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04204v1) [paper-pdf](http://arxiv.org/pdf/2502.04204v1)

**Authors**: Shaopeng Fu, Liang Ding, Di Wang

**Abstract**: Jailbreak attacks against large language models (LLMs) aim to induce harmful behaviors in LLMs through carefully crafted adversarial prompts. To mitigate attacks, one way is to perform adversarial training (AT)-based alignment, i.e., training LLMs on some of the most adversarial prompts to help them learn how to behave safely under attacks. During AT, the length of adversarial prompts plays a critical role in the robustness of aligned LLMs. This paper focuses on adversarial suffix jailbreak attacks and unveils that to defend against a jailbreak attack with an adversarial suffix of length $\Theta(M)$, it is enough to align LLMs on prompts with adversarial suffixes of length $\Theta(\sqrt{M})$. Theoretically, we analyze the adversarial in-context learning of linear transformers on linear regression tasks and prove a robust generalization bound for trained transformers. The bound depends on the term $\Theta(\sqrt{M_{\text{test}}}/M_{\text{train}})$, where $M_{\text{train}}$ and $M_{\text{test}}$ are the number of adversarially perturbed in-context samples during training and testing. Empirically, we conduct AT on popular open-source LLMs and evaluate their robustness against jailbreak attacks of different adversarial suffix lengths. Results confirm a positive correlation between the attack success rate and the ratio of the square root of the adversarial suffix during jailbreaking to the length during AT. Our findings show that it is practical to defend "long-length" jailbreak attacks via efficient "short-length" AT. The code is available at https://github.com/fshp971/adv-icl.

摘要: 针对大型语言模型(LLM)的越狱攻击旨在通过精心设计的对抗性提示在LLM中诱导有害行为。为了减轻攻击，一种方法是执行基于对抗性训练(AT)的对齐，即根据一些最具对抗性的提示对LLM进行培训，以帮助它们学习如何在攻击下安全地行为。在自动对准过程中，对抗性提示的长度对对准LLMS的稳健性起着至关重要的作用。本文主要研究对抗性后缀越狱攻击，揭示了要防御对抗性后缀长度为$\theta(M)$的越狱攻击，只需使提示上的LLMS与长度为$\theta(\Sqrt{M})$的对抗性后缀对齐即可。在理论上，我们分析了线性回归任务中线性变压器的对抗性上下文学习，并证明了训练的变压器的一个稳健的泛化上界。这个界限取决于术语$\Theta(\sqrt{M_{\text{test}}}/M_{\text{train}})$，，其中$M_{\TEXT{TEST}}$和$M_{\TEXT{TEST}}$是训练和测试过程中受到不利干扰的上下文样本的数量。经验性地，我们对流行的开源LLM进行了AT，并评估了它们对不同敌意后缀长度的越狱攻击的健壮性。结果证实，攻击成功率与越狱时敌意后缀的平方根与AT中敌意后缀的长度之比呈正相关。我们的研究结果表明，通过有效的“短长度”AT防御“长长度”越狱攻击是可行的。代码可在https://github.com/fshp971/adv-icl.上获得



## **15. Assessing and Prioritizing Ransomware Risk Based on Historical Victim Data**

根据历史受害者数据评估勒索软件风险并优先排序 cs.CR

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04421v1) [paper-pdf](http://arxiv.org/pdf/2502.04421v1)

**Authors**: Spencer Massengale, Philip Huff

**Abstract**: We present an approach to identifying which ransomware adversaries are most likely to target specific entities, thereby assisting these entities in formulating better protection strategies. Ransomware poses a formidable cybersecurity threat characterized by profit-driven motives, a complex underlying economy supporting criminal syndicates, and the overt nature of its attacks. This type of malware has consistently ranked among the most prevalent, with a rapid escalation in activity observed. Recent estimates indicate that approximately two-thirds of organizations experienced ransomware attacks in 2023 \cite{Sophos2023Ransomware}. A central tactic in ransomware campaigns is publicizing attacks to coerce victims into paying ransoms. Our study utilizes public disclosures from ransomware victims to predict the likelihood of an entity being targeted by a specific ransomware variant. We employ a Large Language Model (LLM) architecture that uses a unique chain-of-thought, multi-shot prompt methodology to define adversary SKRAM (Skills, Knowledge, Resources, Authorities, and Motivation) profiles from ransomware bulletins, threat reports, and news items. This analysis is enriched with publicly available victim data and is further enhanced by a heuristic for generating synthetic data that reflects victim profiles. Our work culminates in the development of a machine learning model that assists organizations in prioritizing ransomware threats and formulating defenses based on the tactics, techniques, and procedures (TTP) of the most likely attackers.

摘要: 我们提出了一种识别哪些勒索软件攻击者最有可能以特定实体为目标的方法，从而帮助这些实体制定更好的保护策略。勒索软件构成了一个强大的网络安全威胁，其特征是利润驱动的动机、支持犯罪集团的复杂基础经济以及其攻击的公开性质。这种类型的恶意软件一直是最普遍的类型之一，并观察到活动迅速升级。最近的估计表明，大约三分之二的组织在2023年经历了勒索软件攻击。勒索软件活动的一个核心策略是公布攻击，迫使受害者支付赎金。我们的研究利用勒索软件受害者的公开披露来预测实体成为特定勒索软件变体目标的可能性。我们采用大型语言模型(LLM)架构，该架构使用独特的思考链、多点提示方法来定义勒索软件公告、威胁报告和新闻项目中的对手Skram(技能、知识、资源、权威和动机)配置文件。这一分析以可公开获得的受害者数据为基础，并通过启发式方法进一步加强，以生成反映受害者概况的合成数据。我们的工作最终是开发一个机器学习模型，帮助组织根据最有可能的攻击者的战术、技术和程序(TTP)对勒索软件威胁进行优先排序并制定防御措施。



## **16. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

G-Designer：通过图神经网络构建多智能体通信布局 cs.MA

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2410.11782v3) [paper-pdf](http://arxiv.org/pdf/2410.11782v3)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Tianlong Chen, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.

摘要: 基于大型语言模型(LLM)的代理的最新进展表明，集体智能可以显著超过单个代理的能力，这主要是由于精心设计的代理间通信拓扑。尽管有多样化和高性能的设计，但实践者在为他们的特定任务选择最有效的流水线时经常面临困惑：\textit{哪个拓扑是我的任务的最佳选择，在确保高质量解决方案的同时避免不必要的通信令牌开销？}针对这种困境，我们引入了G-Designer，这是一个自适应的、高效的、健壮的多代理部署解决方案，它动态地设计任务感知的、定制的通信拓扑。具体地说，G-Designer将多代理系统建模为多代理网络，利用变化图自动编码器对节点(代理)和特定于任务的虚拟节点进行编码，并解码任务自适应的高性能通信拓扑。在六个基准测试上的广泛实验表明，G-Designer是：\extbf{(1)高性能}，在MMLU上获得了更好的结果，准确率为84.50\$，在HumanEval上，PASS@1的准确率为89.90\$；\extbf{(2)任务自适应}，构建了针对任务难度的通信协议，在HumanEval上减少了高达95.33\$的令牌消耗；以及\extbf{(3)对手健壮性}，防御代理对手攻击，精确度仅下降了$0.3\%$。



## **17. Adversarial Attacks for Drift Detection**

漂移检测的对抗攻击 cs.LG

Accepted at ESANN 2025

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2411.16591v2) [paper-pdf](http://arxiv.org/pdf/2411.16591v2)

**Authors**: Fabian Hinder, Valerie Vaquet, Barbara Hammer

**Abstract**: Concept drift refers to the change of data distributions over time. While drift poses a challenge for learning models, requiring their continual adaption, it is also relevant in system monitoring to detect malfunctions, system failures, and unexpected behavior. In the latter case, the robust and reliable detection of drifts is imperative. This work studies the shortcomings of commonly used drift detection schemes. We show how to construct data streams that are drifting without being detected. We refer to those as drift adversarials. In particular, we compute all possible adversairals for common detection schemes and underpin our theoretical findings with empirical evaluations.

摘要: 概念漂移是指数据分布随时间的变化。虽然漂移对学习模型构成了挑战，需要它们的持续适应，但它在系统监控中也与检测故障、系统故障和意外行为相关。在后一种情况下，对漂移进行稳健且可靠的检测至关重要。这项工作研究了常用漂移检测方案的缺点。我们展示了如何构建漂移而不被检测到的数据流。我们将这些称为漂移对手。特别是，我们计算了常见检测方案的所有可能的不利因素，并通过经验评估来支持我们的理论发现。



## **18. The Gradient Puppeteer: Adversarial Domination in Gradient Leakage Attacks through Model Poisoning**

梯度木偶师：通过模型中毒在梯度泄漏攻击中的对抗统治 cs.CR

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04106v1) [paper-pdf](http://arxiv.org/pdf/2502.04106v1)

**Authors**: Kunlan Xiang, Haomiao Yang, Meng Hao, Haoxin Wang, Shaofeng Li, Zikang Ding, Tianwei Zhang

**Abstract**: In Federated Learning (FL), clients share gradients with a central server while keeping their data local. However, malicious servers could deliberately manipulate the models to reconstruct clients' data from shared gradients, posing significant privacy risks. Although such active gradient leakage attacks (AGLAs) have been widely studied, they suffer from several limitations including incomplete attack coverage and poor stealthiness. In this paper, we address these limitations with two core contributions. First, we introduce a new theoretical analysis approach, which uniformly models AGLAs as backdoor poisoning. This analysis approach reveals that the core principle of AGLAs is to bias the gradient space to prioritize the reconstruction of a small subset of samples while sacrificing the majority, which theoretically explains the above limitations of existing AGLAs. Second, we propose Enhanced Gradient Global Vulnerability (EGGV), the first AGLA that achieves complete attack coverage while evading client-side detection. In particular, EGGV employs a gradient projector and a jointly optimized discriminator to assess gradient vulnerability, steering the gradient space toward the point most prone to data leakage. Extensive experiments show that EGGV achieves complete attack coverage and surpasses SOTA with at least a 43% increase in reconstruction quality (PSNR) and a 45% improvement in stealthiness (D-SNR).

摘要: 在联合学习(FL)中，客户端与中央服务器共享渐变，同时将其数据保留在本地。然而，恶意服务器可能会故意操纵模型，根据共享的梯度重建客户的数据，从而带来重大的隐私风险。虽然这种主动梯度泄漏攻击(AGLA)已经得到了广泛的研究，但它们存在攻击覆盖不完整和隐蔽性差等局限性。在本文中，我们通过两个核心贡献来解决这些限制。首先，我们介绍了一种新的理论分析方法，将AGLA统一建模为后门中毒。这种分析方法揭示了AGLA的核心原理是偏向梯度空间，优先考虑小样本子集的重建，同时牺牲大多数样本，这从理论上解释了现有AGLA的上述局限性。其次，我们提出了增强的梯度全局漏洞(EGGV)，这是第一个在逃避客户端检测的情况下实现完全攻击覆盖的AGLA。特别是，EGGV使用一个梯度投影器和一个联合优化的鉴别器来评估梯度脆弱性，将梯度空间引导到最容易发生数据泄漏的点。大量实验表明，EGGV实现了攻击的完全覆盖，并且在重建质量(PSNR)和隐蔽性(D-SNR)方面都超过了SOTA算法，分别提高了43%和45%。



## **19. Consumer INS Coupled with Carrier Phase Measurements for GNSS Spoofing Detection**

消费者惯导系统与载相测量相结合，用于GNSS欺骗检测 cs.CR

Presented at ION ITM/PTTI 2025

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.03870v1) [paper-pdf](http://arxiv.org/pdf/2502.03870v1)

**Authors**: Tore Johansson, Marco Spanghero, Panos Papadimitratos

**Abstract**: Global Navigation Satellite Systems enable precise localization and timing even for highly mobile devices, but legacy implementations provide only limited support for the new generation of security-enhanced signals. Inertial Measurement Units have proved successful in augmenting the accuracy and robustness of the GNSS-provided navigation solution, but effective navigation based on inertial techniques in denied contexts requires high-end sensors. However, commercially available mobile devices usually embed a much lower-grade inertial system. To counteract an attacker transmitting all the adversarial signals from a single antenna, we exploit carrier phase-based observations coupled with a low-end inertial sensor to identify spoofing and meaconing. By short-time integration with an inertial platform, which tracks the displacement of the GNSS antenna, the high-frequency movement at the receiver is correlated with the variation in the carrier phase. In this way, we identify legitimate transmitters, based on their geometrical diversity with respect to the antenna system movement. We introduce a platform designed to effectively compare different tiers of commercial INS platforms with a GNSS receiver. By characterizing different inertial sensors, we show that simple MEMS INS perform as well as high-end industrial-grade sensors. Sensors traditionally considered unsuited for navigation purposes offer great performance at the short integration times used to evaluate the carrier phase information consistency against the high-frequency movement. Results from laboratory evaluation and through field tests at Jammertest 2024 show that the detector is up to 90% accurate in correctly identifying spoofing (or the lack of it), without any modification to the receiver structure, and with mass-production grade INS typical for mobile phones.

摘要: 全球导航卫星系统即使对高度移动的设备也能实现精确的定位和计时，但传统实现对新一代安全增强型信号的支持有限。事实证明，惯性测量单元在提高全球导航卫星系统提供的导航解决方案的精度和稳健性方面取得了成功，但在被拒绝的情况下，基于惯性技术的有效导航需要高端传感器。然而，商业上可用的移动设备通常嵌入的惯性系统级别要低得多。为了对抗攻击者从单个天线发送所有敌对信号，我们利用基于载波相位的观测与低端惯性传感器相结合来识别欺骗和测量。通过与跟踪GNSS天线位移的惯性平台的短时积分，接收器处的高频运动与载波相位的变化相关联。通过这种方式，我们根据发射机相对于天线系统运动的几何多样性来识别合法发射机。我们介绍了一个平台，该平台旨在有效地比较不同级别的商业惯导平台与GNSS接收机。通过对不同惯性传感器的表征，我们证明了简单的MEMS惯导系统的性能与高端工业级传感器一样好。传统上被认为不适合用于导航目的的传感器在用于评估载波相位信息与高频运动的一致性的短积分时间内具有很好的性能。来自实验室评估和Jammertest 2024现场测试的结果表明，该探测器在正确识别欺骗(或没有欺骗)方面的准确率高达90%，无需对接收器结构进行任何修改，并且具有大规模生产级别的惯导系统，通常适用于移动电话。



## **20. Time-based GNSS attack detection**

基于时间的全球导航卫星系统攻击检测 cs.CR

IEEE Transactions on Aerospace and Electronic Systems (Early Access)

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.03868v1) [paper-pdf](http://arxiv.org/pdf/2502.03868v1)

**Authors**: Marco Spanghero, Panos Papadimitratos

**Abstract**: To safeguard Civilian Global Navigation Satellite Systems (GNSS) external information available to the platform encompassing the GNSS receiver can be used to detect attacks. Cross-checking the GNSS-provided time against alternative multiple trusted time sources can lead to attack detection aiming at controlling the GNSS receiver time. Leveraging external, network-connected secure time providers and onboard clock references, we achieve detection even under fine-grained time attacks. We provide an extensive evaluation of our multi-layered defense against adversaries mounting attacks against the GNSS receiver along with controlling the network link. We implement adversaries spanning from simplistic spoofers to advanced ones synchronized with the GNSS constellation. We demonstrate attack detection is possible in all tested cases (sharp discontinuity, smooth take-over, and coordinated network manipulation) without changes to the structure of the GNSS receiver. Leveraging the diversity of the reference time sources, detection of take-over time push as low as 150us is possible. Smooth take-overs forcing variations as low as 30ns are also detected based on on-board precision oscillators. The method (and thus the evaluation) is largely agnostic to the satellite constellation and the attacker type, making time-based data validation of GNSS information compatible with existing receivers and readily deployable.

摘要: 为保障民用全球导航卫星系统(全球导航卫星系统)的安全，包括全球导航卫星系统接收器在内的平台可获得的外部信息可用于检测攻击。将GNSS提供的时间与备选的多个可信时间源进行交叉检查，可以导致旨在控制GNSS接收器时间的攻击检测。利用外部、网络连接的安全时间提供程序和板载时钟参考，我们即使在细粒度的时间攻击下也能实现检测。我们提供了对我们的多层防御的广泛评估，以抵御对GNSS接收器发起攻击的对手以及控制网络链路的对手。我们实现了从简单的欺骗者到与GNSS星座同步的高级欺骗者的对手。我们演示了在所有测试情况下(急剧中断、平稳接管和协调网络操作)都可以进行攻击检测，而不需要改变GNSS接收器的结构。利用基准时间源的多样性，可以检测到低至150us的接管时间推进。基于机载精密振荡器，还可以检测到低至30 ns的平稳接管强迫变化。该方法(以及评估)在很大程度上与卫星星座和攻击者类型无关，使全球导航卫星系统信息的基于时间的数据验证与现有接收器兼容，并易于部署。



## **21. On Robust Reinforcement Learning with Lipschitz-Bounded Policy Networks**

关于Lipschitz有界政策网络的鲁棒强化学习 cs.LG

Accepted to the Symposium on Systems Theory in Data and Optimization  (SysDO 2024)

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2405.11432v3) [paper-pdf](http://arxiv.org/pdf/2405.11432v3)

**Authors**: Nicholas H. Barbara, Ruigang Wang, Ian R. Manchester

**Abstract**: This paper presents a study of robust policy networks in deep reinforcement learning. We investigate the benefits of policy parameterizations that naturally satisfy constraints on their Lipschitz bound, analyzing their empirical performance and robustness on two representative problems: pendulum swing-up and Atari Pong. We illustrate that policy networks with smaller Lipschitz bounds are more robust to disturbances, random noise, and targeted adversarial attacks than unconstrained policies composed of vanilla multi-layer perceptrons or convolutional neural networks. However, the structure of the Lipschitz layer is important. We find that the widely-used method of spectral normalization is too conservative and severely impacts clean performance, whereas more expressive Lipschitz layers such as the recently-proposed Sandwich layer can achieve improved robustness without sacrificing clean performance.

摘要: 本文对深度强化学习中的鲁棒政策网络进行了研究。我们研究了自然满足Lipschitz界约束的政策参数化的好处，分析了它们在两个代表性问题上的经验性能和稳健性：钟摆摆动和Atari Pong。我们说明，Lipschitz界较小的策略网络比由普通多层感知器或卷积神经网络组成的无约束策略对干扰、随机噪音和有针对性的对抗攻击更稳健。然而，利普希茨层的结构很重要。我们发现，广泛使用的光谱正规化方法过于保守，严重影响干净性能，而更有表现力的Lipschitz层（例如最近提出的三明治层）可以在不牺牲干净性能的情况下实现更好的鲁棒性。



## **22. On Effects of Steering Latent Representation for Large Language Model Unlearning**

论引导潜在表示对大型语言模型取消学习的影响 cs.CL

Accepted at AAAI-25 Main Technical Track

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2408.06223v3) [paper-pdf](http://arxiv.org/pdf/2408.06223v3)

**Authors**: Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue

**Abstract**: Representation Misdirection for Unlearning (RMU), which steers model representation in the intermediate layer to a target random representation, is an effective method for large language model (LLM) unlearning. Despite its high performance, the underlying cause and explanation remain underexplored. In this paper, we theoretically demonstrate that steering forget representations in the intermediate layer reduces token confidence, causing LLMs to generate wrong or nonsense responses. We investigate how the coefficient influences the alignment of forget-sample representations with the random direction and hint at the optimal coefficient values for effective unlearning across different network layers. We show that RMU unlearned models are robust against adversarial jailbreak attacks. Furthermore, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs. To resolve this drawback, we propose Adaptive RMU--a simple yet effective alternative method that makes unlearning effective with most layers. Extensive experiments demonstrate that Adaptive RMU significantly improves the unlearning performance compared to prior art while incurring no additional computational cost.

摘要: 遗忘表征误导(RMU)是一种有效的大语言模型遗忘方法，它将中间层的模型表征引导到目标随机表征。尽管其表现良好，但其根本原因和解释仍未得到充分研究。在本文中，我们从理论上证明了中间层中的转向遗忘表征降低了令牌置信度，从而导致LLM产生错误或无意义的响应。我们研究了系数如何影响遗忘样本表示与随机方向的对齐，并提示了跨不同网络层有效遗忘的最优系数值。我们证明了RMU未学习模型对敌意越狱攻击是健壮的。此外，我们的实证分析表明，当RMU应用于LLMS的中后期时，其有效性较差。为了解决这一缺陷，我们提出了自适应RMU--一种简单但有效的替代方法，使遗忘在大多数层都有效。大量实验表明，与现有技术相比，自适应RMU在不增加额外计算代价的情况下，显著改善了遗忘性能。



## **23. How vulnerable is my policy? Adversarial attacks on modern behavior cloning policies**

我的政策有多脆弱？对现代行为克隆政策的敌对攻击 cs.LG

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.03698v1) [paper-pdf](http://arxiv.org/pdf/2502.03698v1)

**Authors**: Basavasagar Patil, Akansha Kalra, Guanhong Tao, Daniel S. Brown

**Abstract**: Learning from Demonstration (LfD) algorithms have shown promising results in robotic manipulation tasks, but their vulnerability to adversarial attacks remains underexplored. This paper presents a comprehensive study of adversarial attacks on both classic and recently proposed algorithms, including Behavior Cloning (BC), LSTM-GMM, Implicit Behavior Cloning (IBC), Diffusion Policy (DP), and VQ-Behavior Transformer (VQ-BET). We study the vulnerability of these methods to untargeted, targeted and universal adversarial perturbations. While explicit policies, such as BC, LSTM-GMM and VQ-BET can be attacked in the same manner as standard computer vision models, we find that attacks for implicit and denoising policy models are nuanced and require developing novel attack methods. Our experiments on several simulated robotic manipulation tasks reveal that most of the current methods are highly vulnerable to adversarial perturbations. We also show that these attacks are transferable across algorithms, architectures, and tasks, raising concerning security vulnerabilities with potentially a white-box threat model. In addition, we test the efficacy of a randomized smoothing, a widely used adversarial defense technique, and highlight its limitation in defending against attacks on complex and multi-modal action distribution common in complex control tasks. In summary, our findings highlight the vulnerabilities of modern BC algorithms, paving way for future work in addressing such limitations.

摘要: 从演示中学习(LFD)算法在机器人操作任务中显示了良好的结果，但它们对对手攻击的脆弱性仍未得到充分研究。本文对经典算法和最近提出的算法进行了全面的研究，包括行为克隆(BC)、LSTM-GMM、隐式行为克隆(IBC)、扩散策略(DP)和VQ-行为转换器(VQ-BET)。我们研究了这些方法对非目标、目标和普遍的对抗性扰动的脆弱性。虽然显式策略，如BC，LSTM-GMM和VQ-BET可以像标准计算机视觉模型一样被攻击，但我们发现对隐式和去噪策略模型的攻击是微妙的，需要开发新的攻击方法。我们在几个模拟机器人操作任务上的实验表明，目前的大多数方法都非常容易受到对抗性扰动的影响。我们还表明，这些攻击可以跨算法、体系结构和任务转移，这引发了人们对潜在白盒威胁模型的安全漏洞的担忧。此外，我们还测试了随机平滑这一广泛使用的对抗性防御技术的有效性，并强调了它在防御复杂控制任务中常见的复杂和多模式动作分布的攻击方面的局限性。总而言之，我们的发现突出了现代BC算法的脆弱性，为未来解决此类限制的工作铺平了道路。



## **24. DocMIA: Document-Level Membership Inference Attacks against DocVQA Models**

DocMIA：针对DocVQA模型的文档级成员推断攻击 cs.LG

ICLR 2025

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.03692v1) [paper-pdf](http://arxiv.org/pdf/2502.03692v1)

**Authors**: Khanh Nguyen, Raouf Kerkouche, Mario Fritz, Dimosthenis Karatzas

**Abstract**: Document Visual Question Answering (DocVQA) has introduced a new paradigm for end-to-end document understanding, and quickly became one of the standard benchmarks for multimodal LLMs. Automating document processing workflows, driven by DocVQA models, presents significant potential for many business sectors. However, documents tend to contain highly sensitive information, raising concerns about privacy risks associated with training such DocVQA models. One significant privacy vulnerability, exploited by the membership inference attack, is the possibility for an adversary to determine if a particular record was part of the model's training data. In this paper, we introduce two novel membership inference attacks tailored specifically to DocVQA models. These attacks are designed for two different adversarial scenarios: a white-box setting, where the attacker has full access to the model architecture and parameters, and a black-box setting, where only the model's outputs are available. Notably, our attacks assume the adversary lacks access to auxiliary datasets, which is more realistic in practice but also more challenging. Our unsupervised methods outperform existing state-of-the-art membership inference attacks across a variety of DocVQA models and datasets, demonstrating their effectiveness and highlighting the privacy risks in this domain.

摘要: 文档可视化问答(DocVQA)为端到端的文档理解引入了一种新的范式，并迅速成为多通道LLMS的标准基准之一。由DocVQA模型驱动的文档处理工作流自动化为许多业务部门提供了巨大的潜力。然而，文件往往包含高度敏感的信息，这引发了人们对培训此类DocVQA模型带来的隐私风险的担忧。成员身份推断攻击利用的一个重要隐私漏洞是，攻击者有可能确定特定记录是否是模型训练数据的一部分。在本文中，我们介绍了两种专门针对DocVQA模型的成员推理攻击。这些攻击是针对两种不同的对抗性场景而设计的：白盒设置和黑盒设置，在白盒设置中，攻击者有权完全访问模型体系结构和参数；在黑盒设置中，只有模型的输出可用。值得注意的是，我们的攻击假设对手无法访问辅助数据集，这在实践中更现实，但也更具挑战性。我们的非监督方法在各种DocVQA模型和数据集上的性能优于现有的最先进的成员资格推理攻击，展示了它们的有效性，并突出了该领域的隐私风险。



## **25. Towards Fair Medical AI: Adversarial Debiasing of 3D CT Foundation Embeddings**

迈向公平的医疗人工智能：3D CT基金会嵌入式的对抗性去偏见 cs.CV

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.04386v1) [paper-pdf](http://arxiv.org/pdf/2502.04386v1)

**Authors**: Guangyao Zheng, Michael A. Jacobs, Vladimir Braverman, Vishwa S. Parekh

**Abstract**: Self-supervised learning has revolutionized medical imaging by enabling efficient and generalizable feature extraction from large-scale unlabeled datasets. Recently, self-supervised foundation models have been extended to three-dimensional (3D) computed tomography (CT) data, generating compact, information-rich embeddings with 1408 features that achieve state-of-the-art performance on downstream tasks such as intracranial hemorrhage detection and lung cancer risk forecasting. However, these embeddings have been shown to encode demographic information, such as age, sex, and race, which poses a significant risk to the fairness of clinical applications.   In this work, we propose a Variation Autoencoder (VAE) based adversarial debiasing framework to transform these embeddings into a new latent space where demographic information is no longer encoded, while maintaining the performance of critical downstream tasks. We validated our approach on the NLST lung cancer screening dataset, demonstrating that the debiased embeddings effectively eliminate multiple encoded demographic information and improve fairness without compromising predictive accuracy for lung cancer risk at 1-year and 2-year intervals. Additionally, our approach ensures the embeddings are robust against adversarial bias attacks. These results highlight the potential of adversarial debiasing techniques to ensure fairness and equity in clinical applications of self-supervised 3D CT embeddings, paving the way for their broader adoption in unbiased medical decision-making.

摘要: 自监督学习通过从大规模的未标记数据集中提取高效和可推广的特征，使医学成像发生了革命性的变化。最近，自监督基础模型已扩展到三维(3D)计算机断层扫描(CT)数据，生成具有1408功能的紧凑、信息丰富的嵌入，在下游任务(如颅内出血检测和肺癌风险预测)中实现最先进的性能。然而，这些嵌入已被证明编码人口统计信息，如年龄、性别和种族，这对临床应用的公平性构成了重大风险。在这项工作中，我们提出了一种基于变异自动编码器(VAE)的对抗性去偏框架，将这些嵌入到一个新的潜在空间中，在那里不再编码人口统计信息，同时保持关键下游任务的性能。我们在NLST肺癌筛查数据集上验证了我们的方法，表明去偏嵌入有效地消除了多个编码的人口统计信息，并在不影响1年和2年间隔的肺癌风险预测准确性的情况下提高了公平性。此外，我们的方法确保了嵌入对敌意偏见攻击的健壮性。这些结果突出了对抗性去偏倚技术在确保自我监督的3D CT嵌入的临床应用中的公平性的潜力，为其在无偏见的医疗决策中的更广泛采用铺平了道路。



## **26. OverThink: Slowdown Attacks on Reasoning LLMs**

过度思考：对推理LLM的缓慢攻击 cs.LG

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.02542v2) [paper-pdf](http://arxiv.org/pdf/2502.02542v2)

**Authors**: Abhinav Kumar, Jaechul Roh, Ali Naseh, Marzena Karpinska, Mohit Iyyer, Amir Houmansadr, Eugene Bagdasarian

**Abstract**: We increase overhead for applications that rely on reasoning LLMs-we force models to spend an amplified number of reasoning tokens, i.e., "overthink", to respond to the user query while providing contextually correct answers. The adversary performs an OVERTHINK attack by injecting decoy reasoning problems into the public content that is used by the reasoning LLM (e.g., for RAG applications) during inference time. Due to the nature of our decoy problems (e.g., a Markov Decision Process), modified texts do not violate safety guardrails. We evaluated our attack across closed-(OpenAI o1, o1-mini, o3-mini) and open-(DeepSeek R1) weights reasoning models on the FreshQA and SQuAD datasets. Our results show up to 18x slowdown on FreshQA dataset and 46x slowdown on SQuAD dataset. The attack also shows high transferability across models. To protect applications, we discuss and implement defenses leveraging LLM-based and system design approaches. Finally, we discuss societal, financial, and energy impacts of OVERTHINK attack which could amplify the costs for third-party applications operating reasoning models.

摘要: 我们增加了依赖推理LLM的应用程序的开销-我们迫使模型花费更多的推理标记，即“过度思考”，以响应用户查询，同时提供上下文正确的答案。敌手通过在推理时间期间将诱骗推理问题注入到推理LLM(例如，用于RAG应用)使用的公共内容中来执行过度思考攻击。由于我们的诱饵问题的性质(例如，马尔可夫决策过程)，修改后的文本不会违反安全护栏。我们在FreshQA和LONG数据集上评估了我们的攻击，跨越了封闭(OpenAI o1，o1-mini，o3-mini)和开放(DeepSeek R1)权重推理模型。我们的结果显示，在FreshQA数据集上的速度最高可减慢18倍，在LAND数据集上的速度最高可减慢46倍。这次攻击还显示出在不同型号之间的高度可转移性。为了保护应用程序，我们讨论并实施了利用基于LLM和系统设计方法的防御措施。最后，我们讨论了过度思考攻击的社会、金融和能源影响，这种攻击可能会放大第三方应用程序运行推理模型的成本。



## **27. Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization**

双流：通过野外级联流优化进行可转移的多目标、实例不可知攻击 cs.CV

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.02096v2) [paper-pdf](http://arxiv.org/pdf/2502.02096v2)

**Authors**: Yixiao Chen, Shikun Sun, Jianshu Li, Ruoyu Li, Zhe Li, Junliang Xing

**Abstract**: Adversarial attacks are widely used to evaluate model robustness, and in black-box scenarios, the transferability of these attacks becomes crucial. Existing generator-based attacks have excellent generalization and transferability due to their instance-agnostic nature. However, when training generators for multi-target tasks, the success rate of transfer attacks is relatively low due to the limitations of the model's capacity. To address these challenges, we propose a novel Dual-Flow framework for multi-target instance-agnostic adversarial attacks, utilizing Cascading Distribution Shift Training to develop an adversarial velocity function. Extensive experiments demonstrate that Dual-Flow significantly improves transferability over previous multi-target generative attacks. For example, it increases the success rate from Inception-v3 to ResNet-152 by 34.58%. Furthermore, our attack method shows substantially stronger robustness against defense mechanisms, such as adversarially trained models.

摘要: 对抗性攻击被广泛用于评估模型的稳健性，在黑盒场景中，这些攻击的可转移性变得至关重要。现有的基于生成器的攻击由于其与实例无关的性质而具有良好的泛化和可转移性。然而，当训练多目标任务的生成器时，由于模型能力的限制，转移攻击的成功率相对较低。为了应对这些挑战，我们提出了一种新的针对多目标实例不可知对手攻击的双流框架，利用级联分布平移训练来开发对手攻击的速度函数。大量实验表明，与以往的多目标生成性攻击相比，双流攻击显著提高了可转移性。例如，它将从初始版本v3到ResNet-152的成功率提高了34.58%。此外，我们的攻击方法对防御机制表现出更强的稳健性，例如对抗训练的模型。



## **28. Understanding and Enhancing the Transferability of Jailbreaking Attacks**

了解并增强越狱攻击的可转移性 cs.LG

Accepted by ICLR 2025

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.03052v1) [paper-pdf](http://arxiv.org/pdf/2502.03052v1)

**Authors**: Runqi Lin, Bo Han, Fengwang Li, Tongling Liu

**Abstract**: Jailbreaking attacks can effectively manipulate open-source large language models (LLMs) to produce harmful responses. However, these attacks exhibit limited transferability, failing to disrupt proprietary LLMs consistently. To reliably identify vulnerabilities in proprietary LLMs, this work investigates the transferability of jailbreaking attacks by analysing their impact on the model's intent perception. By incorporating adversarial sequences, these attacks can redirect the source LLM's focus away from malicious-intent tokens in the original input, thereby obstructing the model's intent recognition and eliciting harmful responses. Nevertheless, these adversarial sequences fail to mislead the target LLM's intent perception, allowing the target LLM to refocus on malicious-intent tokens and abstain from responding. Our analysis further reveals the inherent distributional dependency within the generated adversarial sequences, whose effectiveness stems from overfitting the source LLM's parameters, resulting in limited transferability to target LLMs. To this end, we propose the Perceived-importance Flatten (PiF) method, which uniformly disperses the model's focus across neutral-intent tokens in the original input, thus obscuring malicious-intent tokens without relying on overfitted adversarial sequences. Extensive experiments demonstrate that PiF provides an effective and efficient red-teaming evaluation for proprietary LLMs.

摘要: 越狱攻击可以有效地操纵开源的大型语言模型(LLM)来产生有害的响应。然而，这些攻击表现出有限的可转移性，未能始终如一地破坏专有LLM。为了可靠地识别专有LLM中的漏洞，该工作通过分析越狱攻击对模型意图感知的影响来调查越狱攻击的可转移性。通过合并敌意序列，这些攻击可以将源LLM的焦点从原始输入中的恶意标记重新定向，从而阻碍模型的意图识别并引发有害响应。然而，这些敌对序列未能误导目标LLM的意图感知，允许目标LLM重新关注恶意令牌并放弃响应。我们的分析进一步揭示了生成的对抗序列中固有的分布依赖关系，其有效性源于过拟合源LLM的参数，导致对目标LLM的可转移性有限。为此，我们提出了感知重要性平坦化(PIF)方法，该方法将模型的焦点均匀地分散在原始输入中的中性意图标记上，从而在不依赖于过度匹配的敌对序列的情况下模糊恶意意图标记。大量实验表明，PIF为专有LLM提供了一种有效和高效的红团队评估。



## **29. SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner**

SelfDefend：LLM可以以实用的方式保护自己免受越狱的侵害 cs.CR

Accepted by USENIX Security Symposium 2025. Please cite the  conference version of this paper, i.e., "Xunguang Wang, Daoyuan Wu, Zhenlan  Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, and  Juergen Rahmel. SelfDefend: LLMs Can Defend Themselves against Jailbreaking  in a Practical Manner. In Proc. USENIX Security, 2025."

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2406.05498v3) [paper-pdf](http://arxiv.org/pdf/2406.05498v3)

**Authors**: Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, Juergen Rahmel

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into multiple categories: human-based, optimization-based, generation-based, and the recent indirect and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delays to user prompts, as well as be compatible with both open-source and closed-source LLMs. Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM as a defense instance (in detection state) to concurrently protect the target LLM instance (in normal answering state) in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs can identify harmful prompts or intentions in user queries, which we empirically validate using mainstream GPT-3.5/4 models against major jailbreak attacks. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. When deployed to protect GPT-3.5/4, Claude, Llama-2-7b/13b, and Mistral, these models outperform seven state-of-the-art defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. Further experiments show that the tuned models are robust to adaptive jailbreaks and prompt injections.

摘要: 越狱是一种新兴的对抗性攻击，它绕过了现成的大型语言模型(LLM)中部署的安全对齐，并已演变为多种类别：基于人的、基于优化的、基于代的以及最近的间接和多语言越狱。然而，提供实际的越狱防御是具有挑战性的，因为它不仅需要处理所有上述越狱攻击，还需要对用户提示造成可以忽略不计的延迟，以及与开源和闭源LLM兼容。受传统影子堆栈安全概念防御内存溢出攻击的启发，提出了一种通用的LLM越狱防御框架--SelfDefend。该框架建立一个影子LLM作为防御实例(处于检测状态)，同时保护正常堆栈中的目标LLM实例(处于正常应答状态)，并与其协作进行基于检查点的访问控制。SelfDefend的有效性建立在我们的观察基础上，即现有的LLM可以识别用户查询中的有害提示或意图，我们使用主流GPT-3.5/4模型对主要越狱攻击进行了经验验证。为了进一步提高防御的健壮性并将成本降至最低，我们使用数据蒸馏方法来优化专用的开源防御模型。当部署保护GPT-3.5/4、克劳德、Llama-2-7b/13b和米斯特拉尔时，这些型号的性能超过了七种最先进的防御系统，与基于GPT-4的SelfDefend的性能相当，额外延迟显著降低。进一步的实验表明，调整后的模型对自适应越狱和快速注入具有较强的鲁棒性。



## **30. Large Language Model Adversarial Landscape Through the Lens of Attack Objectives**

从攻击目标角度看大语言模型的对抗格局 cs.CR

15 pages

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.02960v1) [paper-pdf](http://arxiv.org/pdf/2502.02960v1)

**Authors**: Nan Wang, Kane Walter, Yansong Gao, Alsharif Abuadbba

**Abstract**: Large Language Models (LLMs) represent a transformative leap in artificial intelligence, enabling the comprehension, generation, and nuanced interaction with human language on an unparalleled scale. However, LLMs are increasingly vulnerable to a range of adversarial attacks that threaten their privacy, reliability, security, and trustworthiness. These attacks can distort outputs, inject biases, leak sensitive information, or disrupt the normal functioning of LLMs, posing significant challenges across various applications.   In this paper, we provide a novel comprehensive analysis of the adversarial landscape of LLMs, framed through the lens of attack objectives. By concentrating on the core goals of adversarial actors, we offer a fresh perspective that examines threats from the angles of privacy, integrity, availability, and misuse, moving beyond conventional taxonomies that focus solely on attack techniques. This objective-driven adversarial landscape not only highlights the strategic intent behind different adversarial approaches but also sheds light on the evolving nature of these threats and the effectiveness of current defenses. Our analysis aims to guide researchers and practitioners in better understanding, anticipating, and mitigating these attacks, ultimately contributing to the development of more resilient and robust LLM systems.

摘要: 大型语言模型(LLM)代表了人工智能的一次革命性飞跃，使人们能够以前所未有的规模理解、生成和与人类语言进行细微差别的交互。然而，LLM越来越容易受到一系列对手攻击，这些攻击威胁到它们的隐私、可靠性、安全性和可信性。这些攻击可能会扭曲输出、注入偏差、泄露敏感信息或扰乱LLMS的正常功能，对各种应用程序构成重大挑战。在这篇文章中，我们提供了一种新颖的全面分析的对抗性景观，通过攻击目标的框架。通过专注于敌对行为者的核心目标，我们提供了一个新的视角，从隐私、完整性、可用性和误用的角度来检查威胁，超越了只关注攻击技术的传统分类。这种以目标为导向的对抗性格局不仅突出了不同对抗性方法背后的战略意图，而且也揭示了这些威胁的演变性质和目前防御的有效性。我们的分析旨在指导研究人员和实践者更好地理解、预测和缓解这些攻击，最终有助于开发更具弹性和健壮性的LLM系统。



## **31. Wolfpack Adversarial Attack for Robust Multi-Agent Reinforcement Learning**

用于鲁棒多智能体强化学习的Wolfpack对抗攻击 cs.LG

8 pages main, 21 pages appendix with reference. Submitted to ICML  2025

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.02844v1) [paper-pdf](http://arxiv.org/pdf/2502.02844v1)

**Authors**: Sunwoo Lee, Jaebak Hwang, Yonghyeon Jo, Seungyul Han

**Abstract**: Traditional robust methods in multi-agent reinforcement learning (MARL) often struggle against coordinated adversarial attacks in cooperative scenarios. To address this limitation, we propose the Wolfpack Adversarial Attack framework, inspired by wolf hunting strategies, which targets an initial agent and its assisting agents to disrupt cooperation. Additionally, we introduce the Wolfpack-Adversarial Learning for MARL (WALL) framework, which trains robust MARL policies to defend against the proposed Wolfpack attack by fostering system-wide collaboration. Experimental results underscore the devastating impact of the Wolfpack attack and the significant robustness improvements achieved by WALL.

摘要: 多智能体强化学习（MARL）中的传统稳健方法经常难以应对合作场景中的协调对抗攻击。为了解决这一局限性，我们提出了狼群对抗攻击框架，该框架受到猎狼策略的启发，该框架针对初始代理及其辅助代理来破坏合作。此外，我们还引入了Wolfpack对抗学习for MARL（WALL）框架，该框架训练强大的MARL策略，以通过促进系统范围的协作来抵御拟议的Wolfpack攻击。实验结果强调了Wolfpack攻击的毁灭性影响以及WALL实现的显着鲁棒性改进。



## **32. MARAGE: Transferable Multi-Model Adversarial Attack for Retrieval-Augmented Generation Data Extraction**

MARAGE：用于检索增强代数据提取的可转移多模型对抗攻击 cs.CL

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.04360v1) [paper-pdf](http://arxiv.org/pdf/2502.04360v1)

**Authors**: Xiao Hu, Eric Liu, Weizhou Wang, Xiangyu Guo, David Lie

**Abstract**: Retrieval-Augmented Generation (RAG) offers a solution to mitigate hallucinations in Large Language Models (LLMs) by grounding their outputs to knowledge retrieved from external sources. The use of private resources and data in constructing these external data stores can expose them to risks of extraction attacks, in which attackers attempt to steal data from these private databases. Existing RAG extraction attacks often rely on manually crafted prompts, which limit their effectiveness. In this paper, we introduce a framework called MARAGE for optimizing an adversarial string that, when appended to user queries submitted to a target RAG system, causes outputs containing the retrieved RAG data verbatim. MARAGE leverages a continuous optimization scheme that integrates gradients from multiple models with different architectures simultaneously to enhance the transferability of the optimized string to unseen models. Additionally, we propose a strategy that emphasizes the initial tokens in the target RAG data, further improving the attack's generalizability. Evaluations show that MARAGE consistently outperforms both manual and optimization-based baselines across multiple LLMs and RAG datasets, while maintaining robust transferability to previously unseen models. Moreover, we conduct probing tasks to shed light on the reasons why MARAGE is more effective compared to the baselines and to analyze the impact of our approach on the model's internal state.

摘要: 检索-增强生成(RAG)提供了一种解决方案，通过将大型语言模型(LLM)的输出与从外部来源检索的知识相结合来缓解幻觉。在构建这些外部数据存储时使用私有资源和数据可能会使它们面临提取攻击的风险，即攻击者试图从这些私有数据库中窃取数据。现有的RAG提取攻击通常依赖于手动创建的提示，这限制了它们的有效性。在本文中，我们介绍了一个名为Marage的框架，用于优化敌意字符串，当该字符串附加到提交到目标RAG系统的用户查询时，会导致输出包含检索到的RAG数据。Marage利用持续优化方案，同时集成来自具有不同架构的多个模型的梯度，以增强优化后的字符串到不可见模型的可转移性。此外，我们还提出了一种强调目标RAG数据中初始令牌的策略，进一步提高了攻击的泛化能力。评估表明，Marage在多个LLM和RAG数据集上的表现始终优于手动基准和基于优化的基准，同时保持了到以前未见过的模型的强大可转移性。此外，我们进行了探索性任务，以阐明为什么Marage比基线更有效的原因，并分析我们的方法对模型内部状态的影响。



## **33. Semantic Entanglement-Based Ransomware Detection via Probabilistic Latent Encryption Mapping**

通过概率潜在加密映射的基于语义纠缠的勒索软件检测 cs.CR

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02730v1) [paper-pdf](http://arxiv.org/pdf/2502.02730v1)

**Authors**: Mohammad Eisa, Quentin Yardley, Rafael Witherspoon, Harriet Pendlebury, Clement Rutherford

**Abstract**: Encryption-based attacks have introduced significant challenges for detection mechanisms that rely on predefined signatures, heuristic indicators, or static rule-based classifications. Probabilistic Latent Encryption Mapping presents an alternative detection framework that models ransomware-induced encryption behaviors through statistical representations of entropy deviations and probabilistic dependencies in execution traces. Unlike conventional approaches that depend on explicit bytecode analysis or predefined cryptographic function call monitoring, probabilistic inference techniques classify encryption anomalies based on their underlying statistical characteristics, ensuring greater adaptability to polymorphic attack strategies. Evaluations demonstrate that entropy-driven classification reduces false positive rates while maintaining high detection accuracy across diverse ransomware families and encryption methodologies. Experimental results further highlight the framework's ability to differentiate between benign encryption workflows and adversarial cryptographic manipulations, ensuring that classification performance remains effective across cloud-based and localized execution environments. Benchmark comparisons illustrate that probabilistic modeling exhibits advantages over heuristic and machine learning-based detection approaches, particularly in handling previously unseen encryption techniques and adversarial obfuscation strategies. Computational efficiency analysis confirms that detection latency remains within operational feasibility constraints, reinforcing the viability of probabilistic encryption classification for real-time security infrastructures. The ability to systematically infer encryption-induced deviations without requiring static attack signatures strengthens detection robustness against adversarial evasion techniques.

摘要: 基于加密的攻击为依赖预定义签名、启发式指示符或基于静态规则的分类的检测机制带来了重大挑战。概率潜在加密映射提供了一种替代检测框架，该框架通过对执行轨迹中的熵偏差和概率依赖关系的统计表示来模拟勒索软件诱导的加密行为。与依赖显式字节码分析或预定义密码函数调用监控的传统方法不同，概率推理技术基于其潜在的统计特征对加密异常进行分类，确保更好地适应多态攻击策略。评估表明，熵驱动的分类降低了误检率，同时在不同的勒索软件系列和加密方法中保持了高检测精度。实验结果进一步突出了该框架区分良性加密工作流和敌意加密操作的能力，确保分类性能在基于云的执行环境和本地化执行环境中保持有效。基准测试比较表明，概率建模比启发式和基于机器学习的检测方法具有优势，特别是在处理以前未见的加密技术和对抗性混淆策略方面。计算效率分析证实，检测延迟保持在操作可行性限制内，从而增强了概率加密分类在实时安全基础设施中的可行性。在不需要静态攻击签名的情况下，系统地推断加密引起的偏差的能力增强了针对敌意规避技术的检测健壮性。



## **34. Enforcing Demographic Coherence: A Harms Aware Framework for Reasoning about Private Data Release**

强制人口一致性：用于推理私人数据发布的危害意识框架 cs.CR

42 pages

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02709v1) [paper-pdf](http://arxiv.org/pdf/2502.02709v1)

**Authors**: Mark Bun, Marco Carmosino, Palak Jain, Gabriel Kaptchuk, Satchit Sivakumar

**Abstract**: The technical literature about data privacy largely consists of two complementary approaches: formal definitions of conditions sufficient for privacy preservation and attacks that demonstrate privacy breaches. Differential privacy is an accepted standard in the former sphere. However, differential privacy's powerful adversarial model and worst-case guarantees may make it too stringent in some situations, especially when achieving it comes at a significant cost to data utility. Meanwhile, privacy attacks aim to expose real and worrying privacy risks associated with existing data release processes but often face criticism for being unrealistic. Moreover, the literature on attacks generally does not identify what properties are necessary to defend against them.   We address the gap between these approaches by introducing demographic coherence, a condition inspired by privacy attacks that we argue is necessary for data privacy. This condition captures privacy violations arising from inferences about individuals that are incoherent with respect to the demographic patterns in the data. Our framework focuses on confidence rated predictors, which can in turn be distilled from almost any data-informed process. Thus, we capture privacy threats that exist even when no attack is explicitly being carried out. Our framework not only provides a condition with respect to which data release algorithms can be analysed but suggests natural experimental evaluation methodologies that could be used to build practical intuition and make tangible assessment of risks. Finally, we argue that demographic coherence is weaker than differential privacy: we prove that every differentially private data release is also demographically coherent, and that there are demographically coherent algorithms which are not differentially private.

摘要: 关于数据隐私的技术文献主要由两种互补的方法组成：对保护隐私的充分条件的正式定义和证明侵犯隐私的攻击。在以前的领域中，差别隐私是一种公认的标准。然而，差异隐私强大的对抗性模型和最坏情况下的保证可能会使其在某些情况下过于严格，特别是当实现它需要付出巨大的数据效用成本时。与此同时，隐私攻击旨在揭露与现有数据发布流程相关的真实且令人担忧的隐私风险，但经常面临不切实际的批评。此外，有关攻击的文献通常没有确定哪些属性是防御它们所必需的。我们通过引入人口一致性来解决这些方法之间的差距，这是一种受到隐私攻击的情况，我们认为这种攻击对数据隐私是必要的。这一条件捕获了由于对个人的推断与数据中的人口统计模式不一致而导致的侵犯隐私行为。我们的框架侧重于置信度预测因素，这些预测因素可以从几乎任何数据知情的过程中提取出来。因此，即使在没有明确实施攻击的情况下，我们也可以捕获存在的隐私威胁。我们的框架不仅提供了数据发布算法可以分析的条件，而且建议了自然的实验评估方法，可以用来建立实际的直觉和对风险进行切实的评估。最后，我们认为人口统计一致性弱于差异隐私：我们证明了每一次差异隐私数据发布在人口统计上也是一致的，并且存在不是差异私有的人口统计一致性算法。



## **35. DiffBreak: Breaking Diffusion-Based Purification with Adaptive Attacks**

迪夫Break：利用自适应攻击打破基于扩散的净化 cs.CR

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2411.16598v2) [paper-pdf](http://arxiv.org/pdf/2411.16598v2)

**Authors**: Andre Kassis, Urs Hengartner, Yaoliang Yu

**Abstract**: Diffusion-based purification (DBP) has emerged as a cornerstone defense against adversarial examples (AEs), widely regarded as robust due to its use of diffusion models (DMs) that project AEs onto the natural data distribution. However, contrary to prior assumptions, we theoretically prove that adaptive gradient-based attacks nullify this foundational claim, effectively targeting the DM rather than the classifier and causing purified outputs to align with adversarial distributions. This surprising discovery prompts a reassessment of DBP's robustness, revealing it stems from critical flaws in backpropagation techniques used so far for attacking DBP. To address these gaps, we introduce DiffBreak, a novel and reliable gradient library for DBP, which exposes how adaptive attacks drastically degrade its robustness. In stricter majority-vote settings, where classifier decisions aggregate predictions over multiple purified inputs, DBP retains partial robustness to traditional norm-bounded AEs due to its stochasticity disrupting adversarial alignment. However, we propose a novel adaptation of a recent optimization method against deepfake watermarking, crafting systemic adversarial perturbations that defeat DBP even under these conditions, ultimately challenging its viability as a defense without improvements.

摘要: 基于扩散的净化(DBP)已经成为对抗对抗性例子(AEs)的基石防御，由于它使用扩散模型(DM)将AEs投影到自然数据分布上，因此被广泛认为是健壮的。然而，与之前的假设相反，我们在理论上证明了基于自适应梯度的攻击使这一基本主张无效，有效地针对DM而不是分类器，并导致纯化的输出与对抗性分布一致。这一令人惊讶的发现促使人们重新评估DBP的健壮性，揭示出它源于迄今用于攻击DBP的反向传播技术的关键缺陷。为了弥补这些缺陷，我们引入了DiffBreak，一个新的、可靠的DBP梯度库，它揭示了自适应攻击是如何显著降低其健壮性的。在更严格的多数投票设置中，分类器决策聚合了对多个净化输入的预测，由于DBP的随机性破坏了对手对齐，DBP对传统的范数有界的AE保持了部分稳健性。然而，我们提出了一种针对深度伪水印的新的优化方法，即使在这些条件下也能击败DBP，最终在没有改进的情况下挑战其作为防御的可行性。



## **36. Dissecting Adversarial Robustness of Multimodal LM Agents**

剖析多模式LM代理的对抗鲁棒性 cs.LG

ICLR 2025. Also oral at NeurIPS 2024 Open-World Agents Workshop

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2406.12814v3) [paper-pdf](http://arxiv.org/pdf/2406.12814v3)

**Authors**: Chen Henry Wu, Rishi Shah, Jing Yu Koh, Ruslan Salakhutdinov, Daniel Fried, Aditi Raghunathan

**Abstract**: As language models (LMs) are used to build autonomous agents in real environments, ensuring their adversarial robustness becomes a critical challenge. Unlike chatbots, agents are compound systems with multiple components taking actions, which existing LMs safety evaluations do not adequately address. To bridge this gap, we manually create 200 targeted adversarial tasks and evaluation scripts in a realistic threat model on top of VisualWebArena, a real environment for web agents. To systematically examine the robustness of agents, we propose the Agent Robustness Evaluation (ARE) framework. ARE views the agent as a graph showing the flow of intermediate outputs between components and decomposes robustness as the flow of adversarial information on the graph. We find that we can successfully break latest agents that use black-box frontier LMs, including those that perform reflection and tree search. With imperceptible perturbations to a single image (less than 5% of total web page pixels), an attacker can hijack these agents to execute targeted adversarial goals with success rates up to 67%. We also use ARE to rigorously evaluate how the robustness changes as new components are added. We find that inference-time compute that typically improves benign performance can open up new vulnerabilities and harm robustness. An attacker can compromise the evaluator used by the reflexion agent and the value function of the tree search agent, which increases the attack success relatively by 15% and 20%. Our data and code for attacks, defenses, and evaluation are at https://github.com/ChenWu98/agent-attack

摘要: 随着语言模型(LMS)被用来在真实环境中构建自治代理，确保它们的对抗健壮性成为一个关键挑战。与聊天机器人不同，代理是由多个组件采取行动的复合系统，现有的LMS安全评估不足以解决这一问题。为了弥合这一差距，我们在VisualWebArena(Web代理的真实环境)上，在现实威胁模型中手动创建了200个有针对性的对抗性任务和评估脚本。为了系统地考察智能体的健壮性，我们提出了智能体健壮性评估(ARE)框架。ARE将代理视为显示组件之间的中间输出流的图，并将健壮性分解为图上的对抗性信息流。我们发现，我们可以成功地破解使用黑箱边界LMS的最新代理，包括执行反射和树搜索的代理。通过对单个图像的不可察觉的干扰(不到网页总像素的5%)，攻击者可以劫持这些代理以执行目标明确的对抗性目标，成功率高达67%。我们还使用ARS来严格评估在添加新组件时健壮性如何变化。我们发现，通常提高良性性能的推理时间计算可能会带来新的漏洞，并损害健壮性。攻击者可以折衷反射代理使用的赋值器和树搜索代理的值函数，从而使攻击成功率相对提高15%和20%。我们用于攻击、防御和评估的数据和代码位于https://github.com/ChenWu98/agent-attack



## **37. Certifying LLM Safety against Adversarial Prompting**

针对对抗性预算认证LLM安全性 cs.CL

Accepted at COLM 2024: https://openreview.net/forum?id=9Ik05cycLq

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2309.02705v4) [paper-pdf](http://arxiv.org/pdf/2309.02705v4)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that add malicious tokens to an input prompt to bypass the safety guardrails of an LLM and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework for defending against adversarial prompts with certifiable safety guarantees. Given a prompt, our procedure erases tokens individually and inspects the resulting subsequences using a safety filter. Our safety certificate guarantees that harmful prompts are not mislabeled as safe due to an adversarial attack up to a certain size. We implement the safety filter in two ways, using Llama 2 and DistilBERT, and compare the performance of erase-and-check for the two cases. We defend against three attack modes: i) adversarial suffix, where an adversarial sequence is appended at the end of a harmful prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. Additionally, we propose three efficient empirical defenses: i) RandEC, a randomized subsampling version of erase-and-check; ii) GreedyEC, which greedily erases tokens that maximize the softmax score of the harmful class; and iii) GradEC, which uses gradient information to optimize tokens to erase. We demonstrate their effectiveness against adversarial prompts generated by the Greedy Coordinate Gradient (GCG) attack algorithm. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.

摘要: 大型语言模型(LLM)容易受到敌意攻击，这些攻击会向输入提示添加恶意令牌，以绕过LLM的安全护栏，导致其产生有害内容。在这项工作中，我们引入了Erase-and-Check，这是第一个用于防御具有可证明安全保证的对抗性提示的框架。在给定提示的情况下，我们的过程将逐个擦除令牌，并使用安全过滤器检查结果子序列。我们的安全证书保证有害提示不会因为达到一定大小的敌意攻击而被错误地标记为安全。我们用Llama 2和DistilBERT两种方法实现了安全过滤器，并比较了两种情况下的擦除和检查性能。我们防御三种攻击模式：i)对抗性后缀，其中对抗性序列被附加在有害提示的末尾；ii)对抗性插入，其中对抗性序列被插入在提示中间的任何位置；以及iii)对抗性注入，其中对抗性标记被插入在提示中的任意位置，而不一定作为连续的块。实验结果表明，该方法在对安全提示保持较好经验性能的同时，对有害提示具有较强的认证安全保障。此外，我们还提出了三种有效的经验防御：i)RandEC，一种随机化的擦除和检查版本；ii)GreedyEC，它贪婪地擦除使有害类别的Softmax得分最大化的标记；以及iii)Gradec，它使用梯度信息来优化要擦除的标记。我们证明了它们对贪婪坐标梯度(GCG)攻击算法生成的敌意提示的有效性。我们实验的代码可以在https://github.com/aounon/certified-llm-safety.上找到



## **38. Uncertainty Quantification for Collaborative Object Detection Under Adversarial Attacks**

对抗攻击下协作对象检测的不确定性量化 cs.CV

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02537v1) [paper-pdf](http://arxiv.org/pdf/2502.02537v1)

**Authors**: Huiqun Huang, Cong Chen, Jean-Philippe Monteuuis, Jonathan Petit, Fei Miao

**Abstract**: Collaborative Object Detection (COD) and collaborative perception can integrate data or features from various entities, and improve object detection accuracy compared with individual perception. However, adversarial attacks pose a potential threat to the deep learning COD models, and introduce high output uncertainty. With unknown attack models, it becomes even more challenging to improve COD resiliency and quantify the output uncertainty for highly dynamic perception scenes such as autonomous vehicles. In this study, we propose the Trusted Uncertainty Quantification in Collaborative Perception framework (TUQCP). TUQCP leverages both adversarial training and uncertainty quantification techniques to enhance the adversarial robustness of existing COD models. More specifically, TUQCP first adds perturbations to the shared information of randomly selected agents during object detection collaboration by adversarial training. TUQCP then alleviates the impacts of adversarial attacks by providing output uncertainty estimation through learning-based module and uncertainty calibration through conformal prediction. Our framework works for early and intermediate collaboration COD models and single-agent object detection models. We evaluate TUQCP on V2X-Sim, a comprehensive collaborative perception dataset for autonomous driving, and demonstrate a 80.41% improvement in object detection accuracy compared to the baselines under the same adversarial attacks. TUQCP demonstrates the importance of uncertainty quantification to COD under adversarial attacks.

摘要: 协同目标检测(COD)和协同感知可以综合来自不同实体的数据或特征，与个体感知相比提高了目标检测的准确性。然而，敌意攻击对深度学习COD模型构成了潜在的威胁，并引入了高输出不确定性。在未知攻击模型的情况下，对于自动驾驶汽车等高度动态的感知场景，提高COD弹性和量化输出不确定性变得更加具有挑战性。在这项研究中，我们提出了协作感知框架中的可信不确定性量化(TUQCP)。TUQCP利用对抗性训练和不确定性量化技术来增强现有COD模型的对抗性稳健性。更具体地说，TUQCP首先通过对抗性训练在目标检测协作过程中对随机选择的代理的共享信息添加扰动。然后，TUQCP通过基于学习的模块提供输出不确定性估计，并通过保角预测提供不确定性校准，从而缓解对抗性攻击的影响。我们的框架适用于早期和中期的协作COD模型和单代理对象检测模型。我们在自主驾驶的综合协作感知数据集V2X-Sim上对TUQCP进行了评估，在相同的对手攻击下，TUQCP的目标检测准确率比基线提高了80.41%。TUQCP证明了在对抗性攻击下不确定性量化对COD的重要性。



## **39. The TIP of the Iceberg: Revealing a Hidden Class of Task-in-Prompt Adversarial Attacks on LLMs**

冰山的提示：揭示对LLM的一类隐藏的即时任务对抗性攻击 cs.CR

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2501.18626v3) [paper-pdf](http://arxiv.org/pdf/2501.18626v3)

**Authors**: Sergey Berezin, Reza Farahbakhsh, Noel Crespi

**Abstract**: We present a novel class of jailbreak adversarial attacks on LLMs, termed Task-in-Prompt (TIP) attacks. Our approach embeds sequence-to-sequence tasks (e.g., cipher decoding, riddles, code execution) into the model's prompt to indirectly generate prohibited inputs. To systematically assess the effectiveness of these attacks, we introduce the PHRYGE benchmark. We demonstrate that our techniques successfully circumvent safeguards in six state-of-the-art language models, including GPT-4o and LLaMA 3.2. Our findings highlight critical weaknesses in current LLM safety alignments and underscore the urgent need for more sophisticated defence strategies.   Warning: this paper contains examples of unethical inquiries used solely for research purposes.

摘要: 我们提出了一类新型的针对LLM的越狱对抗攻击，称为提示任务（TIP）攻击。我们的方法嵌入序列到序列任务（例如，密码解码、谜语、代码执行）到模型的提示中，以间接生成禁止的输入。为了系统性评估这些攻击的有效性，我们引入了PHRYGE基准。我们证明我们的技术成功规避了六种最先进语言模型（包括GPT-4 o和LLaMA 3.2）中的保护措施。我们的研究结果凸显了当前LLM安全调整中的关键弱点，并强调了对更复杂防御策略的迫切需要。   警告：本文包含仅用于研究目的的不道德调查的例子。



## **40. Medical Multimodal Model Stealing Attacks via Adversarial Domain Alignment**

医学多模式模型通过对抗领域对齐窃取攻击 cs.CR

Accepted at AAAI 2025

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02438v1) [paper-pdf](http://arxiv.org/pdf/2502.02438v1)

**Authors**: Yaling Shen, Zhixiong Zhuang, Kun Yuan, Maria-Irina Nicolae, Nassir Navab, Nicolas Padoy, Mario Fritz

**Abstract**: Medical multimodal large language models (MLLMs) are becoming an instrumental part of healthcare systems, assisting medical personnel with decision making and results analysis. Models for radiology report generation are able to interpret medical imagery, thus reducing the workload of radiologists. As medical data is scarce and protected by privacy regulations, medical MLLMs represent valuable intellectual property. However, these assets are potentially vulnerable to model stealing, where attackers aim to replicate their functionality via black-box access. So far, model stealing for the medical domain has focused on classification; however, existing attacks are not effective against MLLMs. In this paper, we introduce Adversarial Domain Alignment (ADA-STEAL), the first stealing attack against medical MLLMs. ADA-STEAL relies on natural images, which are public and widely available, as opposed to their medical counterparts. We show that data augmentation with adversarial noise is sufficient to overcome the data distribution gap between natural images and the domain-specific distribution of the victim MLLM. Experiments on the IU X-RAY and MIMIC-CXR radiology datasets demonstrate that Adversarial Domain Alignment enables attackers to steal the medical MLLM without any access to medical data.

摘要: 医疗多模式大型语言模型(MLLMS)正在成为医疗保健系统的重要组成部分，帮助医务人员进行决策和结果分析。放射学报告生成模型能够解释医学图像，从而减少了放射科医生的工作量。由于医疗数据稀缺，而且受到隐私法规的保护，医疗MLLM代表着宝贵的知识产权。然而，这些资产可能容易受到模型窃取的攻击，攻击者的目标是通过黑盒访问来复制它们的功能。到目前为止，针对医学领域的模型窃取主要集中在分类上，然而，现有的攻击对MLLMS并不有效。在本文中，我们介绍了第一个针对医学MLLM的窃取攻击--对抗性领域对齐(ADA-Steal)。Ada-steal依赖于自然图像，这些图像是公开的，可以广泛使用，而不是医学上的同行。我们证明了使用对抗性噪声的数据增强足以克服自然图像和受害者MLLM的特定领域分布之间的数据分布差距。在Iu-X-Ray和MIMIC-CXR放射学数据集上的实验表明，对抗性领域对齐使攻击者能够在不访问任何医疗数据的情况下窃取医疗MLLM。



## **41. Rule-ATT&CK Mapper (RAM): Mapping SIEM Rules to TTPs Using LLMs**

规则-ATA & CK映射器（RAM）：使用LLM将SIEM规则映射到TTP cs.CR

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02337v1) [paper-pdf](http://arxiv.org/pdf/2502.02337v1)

**Authors**: Prasanna N. Wudali, Moshe Kravchik, Ehud Malul, Parth A. Gandhi, Yuval Elovici, Asaf Shabtai

**Abstract**: The growing frequency of cyberattacks has heightened the demand for accurate and efficient threat detection systems. SIEM platforms are important for analyzing log data and detecting adversarial activities through rule-based queries, also known as SIEM rules. The efficiency of the threat analysis process relies heavily on mapping these SIEM rules to the relevant attack techniques in the MITRE ATT&CK framework. Inaccurate annotation of SIEM rules can result in the misinterpretation of attacks, increasing the likelihood that threats will be overlooked. Existing solutions for annotating SIEM rules with MITRE ATT&CK technique labels have notable limitations: manual annotation of SIEM rules is both time-consuming and prone to errors, and ML-based approaches mainly focus on annotating unstructured free text sources rather than structured data like SIEM rules. Structured data often contains limited information, further complicating the annotation process and making it a challenging task. To address these challenges, we propose Rule-ATT&CK Mapper (RAM), a novel framework that leverages LLMs to automate the mapping of structured SIEM rules to MITRE ATT&CK techniques. RAM's multi-stage pipeline, which was inspired by the prompt chaining technique, enhances mapping accuracy without requiring LLM pre-training or fine-tuning. Using the Splunk Security Content dataset, we evaluate RAM's performance using several LLMs, including GPT-4-Turbo, Qwen, IBM Granite, and Mistral. Our evaluation highlights GPT-4-Turbo's superior performance, which derives from its enriched knowledge base, and an ablation study emphasizes the importance of external contextual knowledge in overcoming the limitations of LLMs' implicit knowledge for domain-specific tasks. These findings demonstrate RAM's potential in automating cybersecurity workflows and provide valuable insights for future advancements in this field.

摘要: 网络攻击的频率越来越高，这提高了对准确高效的威胁检测系统的需求。SIEM平台对于通过基于规则的查询(也称为SIEM规则)分析日志数据和检测敌对活动非常重要。威胁分析过程的效率在很大程度上依赖于将这些SIEM规则映射到MITRE ATT&CK框架中的相关攻击技术。对SIEM规则的不准确注释可能会导致对攻击的误解，从而增加威胁被忽视的可能性。现有的使用MITRE ATT&CK技术标签标注SIEM规则的解决方案有明显的局限性：手工标注SIEM规则既耗时又容易出错，基于ML的方法主要专注于标注非结构化自由文本源而不是像SIEM规则这样的结构化数据。结构化数据通常包含有限的信息，这使注释过程进一步复杂化，并使其成为一项具有挑战性的任务。为了应对这些挑战，我们提出了规则-ATT&CK映射器(RAM)，这是一个新的框架，它利用LLMS来自动将结构化SIEM规则映射到MITRE ATT&CK技术。RAM的多级流水线的灵感来自于快速链接技术，无需LLM预训练或微调即可提高映射精度。使用Splunk Security内容数据集，我们使用几个LLM来评估RAM的性能，包括GPT-4-Turbo、Qwen、IBM Granite和Mistral。我们的评估突出了GPT-4-Turbo的卓越性能，这源于其丰富的知识库，而一项消融研究强调了外部上下文知识在克服LLMS的隐含知识对特定领域任务的限制方面的重要性。这些发现展示了RAM在自动化网络安全工作流程方面的潜力，并为该领域的未来发展提供了有价值的见解。



## **42. FRAUD-RLA: A new reinforcement learning adversarial attack against credit card fraud detection**

FARUD-RLA：针对信用卡欺诈检测的新强化学习对抗攻击 cs.LG

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02290v1) [paper-pdf](http://arxiv.org/pdf/2502.02290v1)

**Authors**: Daniele Lunghi, Yannick Molinghen, Alkis Simitsis, Tom Lenaerts, Gianluca Bontempi

**Abstract**: Adversarial attacks pose a significant threat to data-driven systems, and researchers have spent considerable resources studying them. Despite its economic relevance, this trend largely overlooked the issue of credit card fraud detection. To address this gap, we propose a new threat model that demonstrates the limitations of existing attacks and highlights the necessity to investigate new approaches. We then design a new adversarial attack for credit card fraud detection, employing reinforcement learning to bypass classifiers. This attack, called FRAUD-RLA, is designed to maximize the attacker's reward by optimizing the exploration-exploitation tradeoff and working with significantly less required knowledge than competitors. Our experiments, conducted on three different heterogeneous datasets and against two fraud detection systems, indicate that FRAUD-RLA is effective, even considering the severe limitations imposed by our threat model.

摘要: 对抗性攻击对数据驱动系统构成重大威胁，研究人员花费了大量资源来研究它们。尽管具有经济相关性，但这一趋势在很大程度上忽视了信用卡欺诈检测问题。为了弥补这一差距，我们提出了一种新的威胁模型，该模型展示了现有攻击的局限性，并强调了研究新方法的必要性。然后，我们设计了一种用于信用卡欺诈检测的新对抗攻击，采用强化学习来绕过分类器。这种名为FRAUP-RLA的攻击旨在通过优化探索与利用的权衡以及所需的知识比竞争对手少得多来最大化攻击者的回报。我们在三个不同的异类数据集上并针对两个欺诈检测系统进行的实验表明，即使考虑到我们的威胁模型所施加的严重限制，FARUD-RLA仍然有效。



## **43. Model Supply Chain Poisoning: Backdooring Pre-trained Models via Embedding Indistinguishability**

模型供应链中毒：通过嵌入不可分割性对预训练模型进行后门 cs.CR

ACM Web Conference 2025 (Oral)

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2401.15883v3) [paper-pdf](http://arxiv.org/pdf/2401.15883v3)

**Authors**: Hao Wang, Shangwei Guo, Jialing He, Hangcheng Liu, Tianwei Zhang, Tao Xiang

**Abstract**: Pre-trained models (PTMs) are widely adopted across various downstream tasks in the machine learning supply chain. Adopting untrustworthy PTMs introduces significant security risks, where adversaries can poison the model supply chain by embedding hidden malicious behaviors (backdoors) into PTMs. However, existing backdoor attacks to PTMs can only achieve partially task-agnostic and the embedded backdoors are easily erased during the fine-tuning process. This makes it challenging for the backdoors to persist and propagate through the supply chain. In this paper, we propose a novel and severer backdoor attack, TransTroj, which enables the backdoors embedded in PTMs to efficiently transfer in the model supply chain. In particular, we first formalize this attack as an indistinguishability problem between poisoned and clean samples in the embedding space. We decompose embedding indistinguishability into pre- and post-indistinguishability, representing the similarity of the poisoned and reference embeddings before and after the attack. Then, we propose a two-stage optimization that separately optimizes triggers and victim PTMs to achieve embedding indistinguishability. We evaluate TransTroj on four PTMs and six downstream tasks. Experimental results show that our method significantly outperforms SOTA task-agnostic backdoor attacks -- achieving nearly 100% attack success rate on most downstream tasks -- and demonstrates robustness under various system settings. Our findings underscore the urgent need to secure the model supply chain against such transferable backdoor attacks. The code is available at https://github.com/haowang-cqu/TransTroj .

摘要: 预训练模型(PTM)广泛应用于机器学习供应链中的各种下游任务。采用不可信的PTMS会带来严重的安全风险，攻击者可以通过在PTMS中嵌入隐藏的恶意行为(后门)来毒化模型供应链。然而，现有的对PTMS的后门攻击只能实现部分任务无关，并且嵌入的后门在微调过程中很容易被擦除。这使得后门在供应链中的持续和传播变得具有挑战性。在本文中，我们提出了一种新的更严重的后门攻击，TransTroj，它使得嵌入PTMS的后门能够在模型供应链中有效地转移。特别地，我们首先将这种攻击形式化为嵌入空间中有毒样本和干净样本之间的不可区分问题。我们将嵌入不可区分性分解为攻击前后的不可区分性，表示攻击前后中毒嵌入和参考嵌入的相似性。然后，我们提出了一种两阶段优化方法，分别对触发者和受害者PTM进行优化，以达到嵌入不可区分的目的。我们在四个PTM和六个下游任务上对TransTroj进行了评估。实验结果表明，我们的方法显著优于SOTA任务无关的后门攻击--在大多数下游任务上获得近100%的攻击成功率--并在各种系统设置下表现出健壮性。我们的发现突显出，迫切需要确保模型供应链免受这种可转移的后门攻击。代码可在https://github.com/haowang-cqu/TransTroj上获得。



## **44. Multi-Domain Graph Foundation Models: Robust Knowledge Transfer via Topology Alignment**

多领域图基础模型：通过布局对齐实现稳健的知识转移 cs.SI

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2502.02017v1) [paper-pdf](http://arxiv.org/pdf/2502.02017v1)

**Authors**: Shuo Wang, Bokui Wang, Zhixiang Shen, Boyan Deng, Zhao Kang

**Abstract**: Recent advances in CV and NLP have inspired researchers to develop general-purpose graph foundation models through pre-training across diverse domains. However, a fundamental challenge arises from the substantial differences in graph topologies across domains. Additionally, real-world graphs are often sparse and prone to noisy connections and adversarial attacks. To address these issues, we propose the Multi-Domain Graph Foundation Model (MDGFM), a unified framework that aligns and leverages cross-domain topological information to facilitate robust knowledge transfer. MDGFM bridges different domains by adaptively balancing features and topology while refining original graphs to eliminate noise and align topological structures. To further enhance knowledge transfer, we introduce an efficient prompt-tuning approach. By aligning topologies, MDGFM not only improves multi-domain pre-training but also enables robust knowledge transfer to unseen domains. Theoretical analyses provide guarantees of MDGFM's effectiveness and domain generalization capabilities. Extensive experiments on both homophilic and heterophilic graph datasets validate the robustness and efficacy of our method.

摘要: CV和NLP的最新进展启发了研究人员通过跨不同领域的预训练来开发通用的图形基础模型。然而，一个根本的挑战来自于跨域的图形拓扑的巨大差异。此外，真实世界的图形通常是稀疏的，容易受到噪声连接和敌意攻击。为了解决这些问题，我们提出了多领域图基础模型(MDGFM)，这是一个统一的框架，对齐和利用跨域拓扑信息来促进健壮的知识转移。MDGFM通过自适应地平衡特征和拓扑来桥接不同的域，同时优化原始图以消除噪声并对齐拓扑结构。为了进一步加强知识转移，我们引入了一种有效的即时调整方法。通过对齐拓扑，MDGFM不仅改善了多领域的预训练，还使知识能够稳健地转移到看不见的领域。理论分析为MDGFM的有效性和领域泛化能力提供了保证。在同嗜性和异嗜性图形数据集上的大量实验验证了该方法的稳健性和有效性。



## **45. Evaluating the Robustness of the "Ensemble Everything Everywhere" Defense**

评估“无处不在”防御的稳健性 cs.LG

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2411.14834v2) [paper-pdf](http://arxiv.org/pdf/2411.14834v2)

**Authors**: Jie Zhang, Christian Schlarmann, Kristina Nikolić, Nicholas Carlini, Francesco Croce, Matthias Hein, Florian Tramèr

**Abstract**: Ensemble everything everywhere is a defense to adversarial examples that was recently proposed to make image classifiers robust. This defense works by ensembling a model's intermediate representations at multiple noisy image resolutions, producing a single robust classification. This defense was shown to be effective against multiple state-of-the-art attacks. Perhaps even more convincingly, it was shown that the model's gradients are perceptually aligned: attacks against the model produce noise that perceptually resembles the targeted class.   In this short note, we show that this defense is not robust to adversarial attack. We first show that the defense's randomness and ensembling method cause severe gradient masking. We then use standard adaptive attack techniques to reduce the defense's robust accuracy from 48% to 14% on CIFAR-100 and from 62% to 11% on CIFAR-10, under the $\ell_\infty$-norm threat model with $\varepsilon=8/255$.

摘要: 包容无处不在的一切是对最近提出的对抗性示例的防御，以使图像分类器稳健。这种防御的工作原理是以多个有噪图像分辨率集成模型的中间表示，产生单个稳健的分类。事实证明，这种防御措施对多种最先进的攻击有效。也许更令人信服的是，它表明模型的梯度在感知上是对齐的：对模型的攻击会产生在感知上类似于目标类的噪音。   在这篇简短的注释中，我们表明这种防御对对抗性攻击并不强大。我们首先表明防御的随机性和集成方法会导致严重的梯度掩蔽。然后，在$\ell_\infty$-norm威胁模型下，我们使用标准的自适应攻击技术将CIFAR-100上的防御鲁棒准确性从48%降低到14%，CIFAR-10上的防御鲁棒准确性从62%降低到11%，$\varepð =8/255$。



## **46. Adversarial Reasoning at Jailbreaking Time**

越狱时的对抗推理 cs.LG

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01633v1) [paper-pdf](http://arxiv.org/pdf/2502.01633v1)

**Authors**: Mahdi Sabbaghi, Paul Kassianik, George Pappas, Yaron Singer, Amin Karbasi, Hamed Hassani

**Abstract**: As large language models (LLMs) are becoming more capable and widespread, the study of their failure cases is becoming increasingly important. Recent advances in standardizing, measuring, and scaling test-time compute suggest new methodologies for optimizing models to achieve high performance on hard tasks. In this paper, we apply these advances to the task of model jailbreaking: eliciting harmful responses from aligned LLMs. We develop an adversarial reasoning approach to automatic jailbreaking via test-time computation that achieves SOTA attack success rates (ASR) against many aligned LLMs, even the ones that aim to trade inference-time compute for adversarial robustness. Our approach introduces a new paradigm in understanding LLM vulnerabilities, laying the foundation for the development of more robust and trustworthy AI systems.

摘要: 随着大型语言模型（LLM）变得越来越强大和广泛，对其失败案例的研究变得越来越重要。标准化、测量和扩展测试时计算方面的最新进展为优化模型以在硬任务中实现高性能提出了新的方法。在本文中，我们将这些进展应用于模型越狱的任务：从对齐的LLM中引发有害反应。我们开发了一种通过测试时计算自动越狱的对抗推理方法，该方法针对许多对齐的LLM，即使是那些旨在以推理时计算为对抗鲁棒性的LLM，也可以实现SOTA攻击成功率（ASB）。我们的方法引入了理解LLM漏洞的新范式，为开发更强大、更值得信赖的人工智能系统奠定了基础。



## **47. Robust-LLaVA: On the Effectiveness of Large-Scale Robust Image Encoders for Multi-modal Large Language Models**

Robust-LLaVA：关于大规模鲁棒图像编码器对多模式大型语言模型的有效性 cs.CV

Under Review

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01576v1) [paper-pdf](http://arxiv.org/pdf/2502.01576v1)

**Authors**: Hashmat Shadab Malik, Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar, Fahad Khan, Salman Khan

**Abstract**: Multi-modal Large Language Models (MLLMs) excel in vision-language tasks but remain vulnerable to visual adversarial perturbations that can induce hallucinations, manipulate responses, or bypass safety mechanisms. Existing methods seek to mitigate these risks by applying constrained adversarial fine-tuning to CLIP vision encoders on ImageNet-scale data, ensuring their generalization ability is preserved. However, this limited adversarial training restricts robustness and broader generalization. In this work, we explore an alternative approach of leveraging existing vision classification models that have been adversarially pre-trained on large-scale data. Our analysis reveals two principal contributions: (1) the extensive scale and diversity of adversarial pre-training enables these models to demonstrate superior robustness against diverse adversarial threats, ranging from imperceptible perturbations to advanced jailbreaking attempts, without requiring additional adversarial training, and (2) end-to-end MLLM integration with these robust models facilitates enhanced adaptation of language components to robust visual features, outperforming existing plug-and-play methodologies on complex reasoning tasks. Through systematic evaluation across visual question-answering, image captioning, and jail-break attacks, we demonstrate that MLLMs trained with these robust models achieve superior adversarial robustness while maintaining favorable clean performance. Our framework achieves 2x and 1.5x average robustness gains in captioning and VQA tasks, respectively, and delivers over 10% improvement against jailbreak attacks. Code and pretrained models will be available at https://github.com/HashmatShadab/Robust-LLaVA.

摘要: 多模式大语言模型(MLLMS)在视觉-语言任务中表现出色，但仍然容易受到视觉对抗性扰动的影响，这些扰动可能会导致幻觉、操纵反应或绕过安全机制。现有的方法试图通过对ImageNet尺度数据上的裁剪视觉编码器应用受限的对抗性微调来缓解这些风险，以确保它们的泛化能力得到保护。然而，这种有限的对抗性训练限制了健壮性和更广泛的泛化。在这项工作中，我们探索了一种替代方法，利用现有的视觉分类模型，这些模型已经在大规模数据上进行了相反的预训练。我们的分析揭示了两个主要贡献：(1)对抗性预训练的广泛规模和多样性使这些模型能够在不需要额外的对抗性训练的情况下，对从不可察觉的扰动到高级越狱尝试等不同的对抗性威胁表现出优越的健壮性；(2)端到端MLLM与这些健壮的模型的集成促进了语言成分对健壮视觉特征的增强适应，在复杂推理任务中的表现优于现有的即插即用方法。通过对视觉问答、图像字幕和越狱攻击的系统评估，我们证明了使用这些健壮模型训练的MLLMS在保持良好的干净性能的同时，获得了优越的对手健壮性。我们的框架在字幕和VQA任务中分别获得了2倍和1.5倍的平均健壮性提升，并在抵御越狱攻击方面提供了超过10%的改进。代码和预先培训的模型将在https://github.com/HashmatShadab/Robust-LLaVA.上提供



## **48. Quantum Quandaries: Unraveling Encoding Vulnerabilities in Quantum Neural Networks**

量子困境：解开量子神经网络中的编码漏洞 quant-ph

7 Pages

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01486v1) [paper-pdf](http://arxiv.org/pdf/2502.01486v1)

**Authors**: Suryansh Upadhyay, Swaroop Ghosh

**Abstract**: Quantum computing (QC) has the potential to revolutionize fields like machine learning, security, and healthcare. Quantum machine learning (QML) has emerged as a promising area, enhancing learning algorithms using quantum computers. However, QML models are lucrative targets due to their high training costs and extensive training times. The scarcity of quantum resources and long wait times further exacerbate the challenge. Additionally, QML providers may rely on third party quantum clouds for hosting models, exposing them and their training data to potential threats. As QML as a Service (QMLaaS) becomes more prevalent, reliance on third party quantum clouds poses a significant security risk. This work demonstrates that adversaries in quantum cloud environments can exploit white box access to QML models to infer the users encoding scheme by analyzing circuit transpilation artifacts. The extracted data can be reused for training clone models or sold for profit. We validate the proposed attack through simulations, achieving high accuracy in distinguishing between encoding schemes. We report that 95% of the time, the encoding can be predicted correctly. To mitigate this threat, we propose a transient obfuscation layer that masks encoding fingerprints using randomized rotations and entanglement, reducing adversarial detection to near random chance 42% , with a depth overhead of 8.5% for a 5 layer QNN design.

摘要: 量子计算(QC)有可能给机器学习、安全和医疗保健等领域带来革命性的变化。量子机器学习(QML)已经成为一个很有前途的领域，它利用量子计算机来增强学习算法。然而，QML模型是有利可图的目标，因为它们的培训成本高，培训时间长。量子资源的稀缺和漫长的等待时间进一步加剧了挑战。此外，QML提供商可能会依赖第三方量子云来托管模型，从而使模型及其训练数据面临潜在威胁。随着QML即服务(QMLaaS)变得越来越普遍，对第三方量子云的依赖构成了重大的安全风险。这项工作表明，量子云环境中的攻击者可以利用白盒访问QML模型，通过分析电路转移伪影来推断用户的编码方案。提取的数据可以重复用于训练克隆模型或出售以赚取利润。我们通过仿真验证了所提出的攻击，在区分不同编码方案时达到了很高的准确率。我们报告说，95%的时间，编码可以被正确预测。为了缓解这种威胁，我们提出了一种瞬时混淆层，它使用随机旋转和纠缠来掩盖编码指纹，将敌意检测的概率降低到接近随机的42%，对于5层QNN设计，深度开销为8.5%。



## **49. DeTrigger: A Gradient-Centric Approach to Backdoor Attack Mitigation in Federated Learning**

DeTrigger：联邦学习中以用户为中心的后门攻击缓解方法 cs.LG

21 pages

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2411.12220v2) [paper-pdf](http://arxiv.org/pdf/2411.12220v2)

**Authors**: Kichang Lee, Yujin Shin, Jonghyuk Yun, Songkuk Kim, Jun Han, JeongGil Ko

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed devices while preserving local data privacy, making it ideal for mobile and embedded systems. However, the decentralized nature of FL also opens vulnerabilities to model poisoning attacks, particularly backdoor attacks, where adversaries implant trigger patterns to manipulate model predictions. In this paper, we propose DeTrigger, a scalable and efficient backdoor-robust federated learning framework that leverages insights from adversarial attack methodologies. By employing gradient analysis with temperature scaling, DeTrigger detects and isolates backdoor triggers, allowing for precise model weight pruning of backdoor activations without sacrificing benign model knowledge. Extensive evaluations across four widely used datasets demonstrate that DeTrigger achieves up to 251x faster detection than traditional methods and mitigates backdoor attacks by up to 98.9%, with minimal impact on global model accuracy. Our findings establish DeTrigger as a robust and scalable solution to protect federated learning environments against sophisticated backdoor threats.

摘要: 联合学习(FL)支持跨分布式设备进行协作模型培训，同时保护本地数据隐私，使其成为移动和嵌入式系统的理想选择。然而，FL的分散性也为建模中毒攻击打开了漏洞，特别是后门攻击，对手植入触发模式来操纵模型预测。在本文中，我们提出了DeTrigger，一个可扩展的高效后门健壮的联邦学习框架，它利用了对手攻击方法的见解。通过使用带有温度缩放的梯度分析，DeTrigger检测并隔离后门触发器，从而在不牺牲良性模型知识的情况下精确削减后门激活的模型权重。对四个广泛使用的数据集的广泛评估表明，DeTrigger的检测速度比传统方法快251倍，后门攻击减少高达98.9%，对全局模型精度的影响最小。我们的发现将DeTrigger确立为一个强大且可扩展的解决方案，可以保护联合学习环境免受复杂的后门威胁。



## **50. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

Top-FlipRAG：针对检索增强生成模型的面向主题的对抗性观点操纵攻击 cs.CL

**SubmitDate**: 2025-02-03    [abs](http://arxiv.org/abs/2502.01386v1) [paper-pdf](http://arxiv.org/pdf/2502.01386v1)

**Authors**: Yuyang Gong, Zhuo Chen, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.

摘要: 基于大型语言模型(LLMS)的检索-增强生成(RAG)系统已经成为诸如问题回答和内容生成等任务的关键。然而，由于其固有的脆弱性，它们对舆论和信息传播的影响越来越大，使其成为安全研究的关键焦点。以前的研究主要针对针对事实或单一查询操作的攻击。在本文中，我们讨论了一个更实际的场景：针对RAG模型的面向主题的对抗性意见操纵攻击，其中要求LLM推理和综合多个视角，使它们特别容易受到系统性知识中毒的影响。具体地说，我们提出了Theme-FlipRAG，这是一种两阶段操纵攻击管道，战略性地制造对抗性扰动来影响相关查询的观点。该方法结合了传统的对抗性排序攻击技术，并利用LLMS丰富的内部相关知识和推理能力来执行语义级的扰动。实验表明，提出的攻击有效地改变了模型输出对特定主题的看法，显著影响了用户对信息的感知。目前的缓解方法无法有效防御此类攻击，这突显了加强RAG系统安全保障的必要性，并为LLM安全研究提供了至关重要的见解。



