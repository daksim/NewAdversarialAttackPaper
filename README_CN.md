# Latest Adversarial Attack Papers
**update at 2025-01-21 09:53:01**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. The Effect of Similarity Measures on Accurate Stability Estimates for Local Surrogate Models in Text-based Explainable AI**

相似性度量对基于文本的可解释人工智能中局部代理模型准确稳定性估计的影响 cs.LG

11 pages, 8 Tables (Minor edits for clarity and grammar)

**SubmitDate**: 2025-01-17    [abs](http://arxiv.org/abs/2406.15839v2) [paper-pdf](http://arxiv.org/pdf/2406.15839v2)

**Authors**: Christopher Burger, Charles Walter, Thai Le

**Abstract**: Recent work has investigated the vulnerability of local surrogate methods to adversarial perturbations on a machine learning (ML) model's inputs, where the explanation is manipulated while the meaning and structure of the original input remains similar under the complex model. Although weaknesses across many methods have been shown to exist, the reasons behind why remain little explored. Central to the concept of adversarial attacks on explainable AI (XAI) is the similarity measure used to calculate how one explanation differs from another. A poor choice of similarity measure can lead to erroneous conclusions on the efficacy of an XAI method. Too sensitive a measure results in exaggerated vulnerability, while too coarse understates its weakness. We investigate a variety of similarity measures designed for text-based ranked lists, including Kendall's Tau, Spearman's Footrule, and Rank-biased Overlap to determine how substantial changes in the type of measure or threshold of success affect the conclusions generated from common adversarial attack processes. Certain measures are found to be overly sensitive, resulting in erroneous estimates of stability.

摘要: 最近的工作研究了局部代理方法对机器学习(ML)模型输入的对抗性扰动的脆弱性，其中解释被操纵，而原始输入的含义和结构在复杂模型下保持相似。尽管许多方法的弱点已经被证明存在，但为什么背后的原因仍然很少被探索。对可解释人工智能(XAI)进行对抗性攻击的概念的核心是用于计算一种解释与另一种解释的差异的相似性度量。如果相似性度量选择不当，可能会导致对XAI方法有效性的错误结论。过于敏感的衡量标准会夸大脆弱性，而过于粗略的衡量标准则会低估其弱点。我们研究了为基于文本的排名列表设计的各种相似性度量，包括Kendall‘s Tau、Spearman’s Footrule和Rank-Biased Overlance，以确定度量类型或成功阈值的实质性变化如何影响从常见的对抗性攻击过程生成的结论。某些措施被发现过于敏感，导致对稳定性的错误估计。



## **2. Jailbreaking as a Reward Misspecification Problem**

越狱是奖励错误指定问题 cs.LG

**SubmitDate**: 2025-01-17    [abs](http://arxiv.org/abs/2406.14393v4) [paper-pdf](http://arxiv.org/pdf/2406.14393v4)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. This misspecification occurs when the reward function fails to accurately capture the intended behavior, leading to misaligned model outputs. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts in a reward-misspecified space. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark against various target aligned LLMs while preserving the human readability of the generated prompts. Furthermore, these attacks on open-source models demonstrate high transferability to closed-source models like GPT-4o and out-of-distribution tasks from HarmBench. Detailed analysis highlights the unique advantages of the proposed reward misspecification objective compared to previous methods, offering new insights for improving LLM safety and robustness.

摘要: 大型语言模型(LLM)的广泛采用引起了人们对它们的安全性和可靠性的担忧，特别是它们对对手攻击的脆弱性。在本文中，我们提出了一种新的观点，将该漏洞归因于对齐过程中的错误指定。当奖励函数未能准确捕获预期行为时，就会出现这种错误说明，从而导致模型输出不对齐。我们引入了一个度量指标ReGap来量化奖励错误指定的程度，并展示了它在检测有害后门提示方面的有效性和健壮性。在这些见解的基础上，我们提出了REMISTY，这是一个用于自动红色团队的系统，它在错误指定奖励的空间中生成对抗性提示。在保持生成提示的人类可读性的同时，针对各种目标对齐的LLM，在AdvBtch基准上实现了最先进的攻击成功率。此外，这些对开源模型的攻击表明，可以很好地转移到GPT-4o等封闭源代码模型和来自HarmBtch的非分发任务。详细的分析强调了与以前的方法相比，所提出的奖励误指定目标的独特优势，为提高LLM的安全性和稳健性提供了新的见解。



## **3. Michscan: Black-Box Neural Network Integrity Checking at Runtime Through Power Analysis**

MichScan：通过功率分析在NPS进行黑匣子神经网络完整性检查 cs.CR

11 pages, 7 figures. To appear in IEEE International Symposium on  Hardware Oriented Security and Trust (HOST) 2025. This material is based upon  work supported by the National Science Foundation under Grant No. 2245573

**SubmitDate**: 2025-01-17    [abs](http://arxiv.org/abs/2501.10174v1) [paper-pdf](http://arxiv.org/pdf/2501.10174v1)

**Authors**: Robi Paul, Michael Zuzak

**Abstract**: As neural networks are increasingly used for critical decision-making tasks, the threat of integrity attacks, where an adversary maliciously alters a model, has become a significant security and safety concern. These concerns are compounded by the use of licensed models, where end-users purchase third-party models with only black-box access to protect model intellectual property (IP). In such scenarios, conventional approaches to verify model integrity require knowledge of model parameters or cooperative model owners. To address this challenge, we propose Michscan, a methodology leveraging power analysis to verify the integrity of black-box TinyML neural networks designed for resource-constrained devices. Michscan is based on the observation that modifications to model parameters impact the instantaneous power consumption of the device. We leverage this observation to develop a runtime model integrity-checking methodology that employs correlational power analysis using a golden template or signature to mathematically quantify the likelihood of model integrity violations at runtime through the Mann-Whitney U-Test. Michscan operates in a black-box environment and does not require a cooperative or trustworthy model owner. We evaluated Michscan using an STM32F303RC microcontroller with an ARM Cortex-M4 running four TinyML models in the presence of three model integrity violations. Michscan successfully detected all integrity violations at runtime using power data from five inferences. All detected violations had a negligible probability P < 10^(-5) of being produced from an unmodified model (i.e., false positive).

摘要: 随着神经网络越来越多地被用于关键的决策任务，完整性攻击的威胁已经成为一个重大的安全问题。授权型号的使用加剧了这些担忧，终端用户购买第三方型号时只有黑盒访问权限，以保护型号知识产权(IP)。在这种情况下，验证模型完整性的传统方法需要了解模型参数或合作的模型所有者。为了应对这一挑战，我们提出了Michcan，这是一种利用功率分析来验证为资源受限设备设计的黑盒TinyML神经网络的完整性的方法。Michcan基于对模型参数的修改会影响设备的瞬时功耗这一观察结果。我们利用这一观察结果开发了一种运行时模型完整性检查方法，该方法使用黄金模板或签名进行相关功率分析，通过Mann-Whitney U-Test从数学上量化运行时违反模型完整性的可能性。Michcan在黑盒环境中运行，不需要合作或值得信赖的模型所有者。我们使用STM32F303RC微控制器和ARM Cortex-M4运行四个TinyML模型，并在存在三个模型完整性违规的情况下对Michcan进行了评估。Michcan使用来自五个推论的电源数据在运行时成功检测到所有完整性违规。所有检测到的违规行为从未经修改的模型(即假阳性)产生的概率P<10^(-5)可以忽略不计。



## **4. Generative AI in Cybersecurity: A Comprehensive Review of LLM Applications and Vulnerabilities**

网络安全中的生成人工智能：LLM应用和漏洞的全面审查 cs.CR

52 pages, 8 figures

**SubmitDate**: 2025-01-17    [abs](http://arxiv.org/abs/2405.12750v2) [paper-pdf](http://arxiv.org/pdf/2405.12750v2)

**Authors**: Mohamed Amine Ferrag, Fatima Alwahedi, Ammar Battah, Bilel Cherif, Abdechakour Mechri, Norbert Tihanyi, Tamas Bisztray, Merouane Debbah

**Abstract**: This paper provides a comprehensive review of the future of cybersecurity through Generative AI and Large Language Models (LLMs). We explore LLM applications across various domains, including hardware design security, intrusion detection, software engineering, design verification, cyber threat intelligence, malware detection, and phishing detection. We present an overview of LLM evolution and its current state, focusing on advancements in models such as GPT-4, GPT-3.5, Mixtral-8x7B, BERT, Falcon2, and LLaMA. Our analysis extends to LLM vulnerabilities, such as prompt injection, insecure output handling, data poisoning, DDoS attacks, and adversarial instructions. We delve into mitigation strategies to protect these models, providing a comprehensive look at potential attack scenarios and prevention techniques. Furthermore, we evaluate the performance of 42 LLM models in cybersecurity knowledge and hardware security, highlighting their strengths and weaknesses. We thoroughly evaluate cybersecurity datasets for LLM training and testing, covering the lifecycle from data creation to usage and identifying gaps for future research. In addition, we review new strategies for leveraging LLMs, including techniques like Half-Quadratic Quantization (HQQ), Reinforcement Learning with Human Feedback (RLHF), Direct Preference Optimization (DPO), Quantized Low-Rank Adapters (QLoRA), and Retrieval-Augmented Generation (RAG). These insights aim to enhance real-time cybersecurity defenses and improve the sophistication of LLM applications in threat detection and response. Our paper provides a foundational understanding and strategic direction for integrating LLMs into future cybersecurity frameworks, emphasizing innovation and robust model deployment to safeguard against evolving cyber threats.

摘要: 本文通过生成式人工智能和大型语言模型(LLMS)对网络安全的未来进行了全面的回顾。我们探索了LLM在不同领域的应用，包括硬件设计安全、入侵检测、软件工程、设计验证、网络威胁情报、恶意软件检测和网络钓鱼检测。我们概述了LLM的演化和现状，重点介绍了GPT-4、GPT-3.5、Mixtral-8x7B、BERT、Falcon2和Llama等模型的进展。我们的分析扩展到LLM漏洞，如快速注入、不安全的输出处理、数据中毒、DDoS攻击和敌意指令。我们深入研究缓解策略以保护这些模型，提供对潜在攻击场景和预防技术的全面了解。此外，我们评估了42个LLM模型在网络安全知识和硬件安全方面的性能，突出了它们的优势和劣势。我们为LLM培训和测试彻底评估网络安全数据集，涵盖从数据创建到使用的整个生命周期，并为未来的研究确定差距。此外，我们还回顾了利用LLMS的新策略，包括半二次量化(HQQ)、带人反馈的强化学习(RLHF)、直接偏好优化(DPO)、量化低阶适配器(QLoRA)和检索增强生成(RAG)。这些见解旨在增强实时网络安全防御，并提高LLM应用程序在威胁检测和响应方面的复杂性。我们的论文为将低成本管理系统整合到未来的网络安全框架中提供了一个基础性的理解和战略方向，强调创新和稳健的模型部署，以防范不断演变的网络威胁。



## **5. CaFA: Cost-aware, Feasible Attacks With Database Constraints Against Neural Tabular Classifiers**

CaFA：针对神经表格分类器的成本意识、可行的攻击，具有数据库约束 cs.CR

Accepted at IEEE S&P 2024

**SubmitDate**: 2025-01-17    [abs](http://arxiv.org/abs/2501.10013v1) [paper-pdf](http://arxiv.org/pdf/2501.10013v1)

**Authors**: Matan Ben-Tov, Daniel Deutch, Nave Frost, Mahmood Sharif

**Abstract**: This work presents CaFA, a system for Cost-aware Feasible Attacks for assessing the robustness of neural tabular classifiers against adversarial examples realizable in the problem space, while minimizing adversaries' effort. To this end, CaFA leverages TabPGD$-$an algorithm we set forth to generate adversarial perturbations suitable for tabular data$-$ and incorporates integrity constraints automatically mined by state-of-the-art database methods. After producing adversarial examples in the feature space via TabPGD, CaFA projects them on the mined constraints, leading, in turn, to better attack realizability. We tested CaFA with three datasets and two architectures and found, among others, that the constraints we use are of higher quality (measured via soundness and completeness) than ones employed in prior work. Moreover, CaFA achieves higher feasible success rates$-$i.e., it generates adversarial examples that are often misclassified while satisfying constraints$-$than prior attacks while simultaneously perturbing few features with lower magnitudes, thus saving effort and improving inconspicuousness. We open-source CaFA, hoping it will serve as a generic system enabling machine-learning engineers to assess their models' robustness against realizable attacks, thus advancing deployed models' trustworthiness.

摘要: 这项工作提出了CAFA，一个成本感知的可行攻击系统，用于评估神经表格分类器对问题空间中可实现的对抗性样本的稳健性，同时最小化对手的努力。为此，CAFA利用我们提出的TabPGD$-$算法来生成适用于表格数据$-$的对抗性扰动，并结合了由最先进的数据库方法自动挖掘的完整性约束。通过TabPGD在特征空间中生成对抗性样本后，CAFA将它们投影到挖掘的约束上，进而导致更好的攻击可实现性。我们用三个数据集和两个体系结构测试了CAFA，发现我们使用的约束比以前工作中使用的约束具有更高的质量(通过可靠性和完整性来衡量)。此外，CAFA获得了更高的可行成功率$-$，即它生成了经常被错误分类的对抗性实例，同时满足了先前攻击的约束$-$，同时扰动了少量较低幅度的特征，从而节省了工作量，改善了隐蔽性。我们开源CAFA，希望它能作为一个通用系统，使机器学习工程师能够评估他们的模型对可实现攻击的健壮性，从而提高部署模型的可信性。



## **6. Computing Optimization-Based Prompt Injections Against Closed-Weights Models By Misusing a Fine-Tuning API**

通过滥用微调API针对闭权模型计算基于优化的提示注射 cs.CR

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2501.09798v1) [paper-pdf](http://arxiv.org/pdf/2501.09798v1)

**Authors**: Andrey Labunets, Nishit V. Pandya, Ashish Hooda, Xiaohan Fu, Earlence Fernandes

**Abstract**: We surface a new threat to closed-weight Large Language Models (LLMs) that enables an attacker to compute optimization-based prompt injections. Specifically, we characterize how an attacker can leverage the loss-like information returned from the remote fine-tuning interface to guide the search for adversarial prompts. The fine-tuning interface is hosted by an LLM vendor and allows developers to fine-tune LLMs for their tasks, thus providing utility, but also exposes enough information for an attacker to compute adversarial prompts. Through an experimental analysis, we characterize the loss-like values returned by the Gemini fine-tuning API and demonstrate that they provide a useful signal for discrete optimization of adversarial prompts using a greedy search algorithm. Using the PurpleLlama prompt injection benchmark, we demonstrate attack success rates between 65% and 82% on Google's Gemini family of LLMs. These attacks exploit the classic utility-security tradeoff - the fine-tuning interface provides a useful feature for developers but also exposes the LLMs to powerful attacks.

摘要: 我们对封闭式大型语言模型(LLM)提出了新的威胁，使攻击者能够计算基于优化的提示注入。具体地说，我们描述了攻击者如何利用从远程微调界面返回的类似丢失的信息来指导对敌意提示的搜索。微调界面由LLM供应商托管，允许开发人员针对他们的任务微调LLM，从而提供实用程序，但也暴露了足够的信息，供攻击者计算敌意提示。通过实验分析，我们表征了Gemini微调API返回的类似损失的值，并证明它们为使用贪婪搜索算法对敌意提示进行离散优化提供了有用的信号。使用PurpleLlama快速注入基准，我们展示了对Google的Gemini系列LLM的攻击成功率在65%到82%之间。这些攻击利用了经典的实用程序-安全权衡-微调界面为开发人员提供了有用的功能，但也使LLM面临强大的攻击。



## **7. Adversarial-Ensemble Kolmogorov Arnold Networks for Enhancing Indoor Wi-Fi Positioning: A Defensive Approach Against Spoofing and Signal Manipulation Attacks**

对抗对手Kolmogorov Arnold网络增强室内Wi-Fi定位：针对欺骗和信号操纵攻击的防御方法 cs.LG

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2501.09609v1) [paper-pdf](http://arxiv.org/pdf/2501.09609v1)

**Authors**: Mitul Goswami, Romit Chatterjee, Somnath Mahato, Prasant Kumar Pattnaik

**Abstract**: The research presents a study on enhancing the robustness of Wi-Fi-based indoor positioning systems against adversarial attacks. The goal is to improve the positioning accuracy and resilience of these systems under two attack scenarios: Wi-Fi Spoofing and Signal Strength Manipulation. Three models are developed and evaluated: a baseline model (M_Base), an adversarially trained robust model (M_Rob), and an ensemble model (M_Ens). All models utilize a Kolmogorov-Arnold Network (KAN) architecture. The robust model is trained with adversarially perturbed data, while the ensemble model combines predictions from both the base and robust models. Experimental results show that the robust model reduces positioning error by approximately 10% compared to the baseline, achieving 2.03 meters error under Wi-Fi spoofing and 2.00 meters under signal strength manipulation. The ensemble model further outperforms with errors of 2.01 meters and 1.975 meters for the respective attack types. This analysis highlights the effectiveness of adversarial training techniques in mitigating attack impacts. The findings underscore the importance of considering adversarial scenarios in developing indoor positioning systems, as improved resilience can significantly enhance the accuracy and reliability of such systems in mission-critical environments.

摘要: 该研究旨在提高基于Wi-Fi的室内定位系统对敌方攻击的稳健性。目标是提高这些系统在两种攻击场景下的定位精度和弹性：Wi-Fi欺骗和信号强度操纵。建立并评价了三种模型：基线模型(M_Base)、对抗性训练的稳健模型(M_Rob)和集成模型(M_ENS)。所有型号均采用Kolmogorov-Arnold Network(KAN)架构。稳健模型用相反的扰动数据来训练，而集成模型结合了来自基本模型和稳健模型的预测。实验结果表明，与基线相比，该模型的定位误差降低了约10%，在Wi-Fi欺骗下达到了2.03米的误差，在信号强度操纵下达到了2.00米的误差。对于不同的攻击类型，该组合模型的误差分别为2.01米和1.975米，表现更加出色。这一分析突出了对抗性训练技术在减轻攻击影响方面的有效性。这些研究结果强调了在开发室内定位系统时考虑对抗性情景的重要性，因为提高复原力可以显著提高这类系统在关键任务环境中的准确性和可靠性。



## **8. On the uncertainty principle of neural networks**

神经网络的不确定性原理 cs.LG

8 pages, 5 figures

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2205.01493v4) [paper-pdf](http://arxiv.org/pdf/2205.01493v4)

**Authors**: Jun-Jie Zhang, Dong-Xiao Zhang, Jian-Nan Chen, Long-Gang Pang, Deyu Meng

**Abstract**: In this study, we explore the inherent trade-off between accuracy and robustness in neural networks, drawing an analogy to the uncertainty principle in quantum mechanics. We propose that neural networks are subject to an uncertainty relation, which manifests as a fundamental limitation in their ability to simultaneously achieve high accuracy and robustness against adversarial attacks. Through mathematical proofs and empirical evidence, we demonstrate that this trade-off is a natural consequence of the sharp boundaries formed between different class concepts during training. Our findings reveal that the complementarity principle, a cornerstone of quantum physics, applies to neural networks, imposing fundamental limits on their capabilities in simultaneous learning of conjugate features. Meanwhile, our work suggests that achieving human-level intelligence through a single network architecture or massive datasets alone may be inherently limited. Our work provides new insights into the theoretical foundations of neural network vulnerability and opens up avenues for designing more robust neural network architectures.

摘要: 在这项研究中，我们探索了神经网络中精度和稳健性之间的内在权衡，类比于量子力学中的不确定原理。我们认为神经网络服从不确定关系，这表现为它们同时获得高精度和对对手攻击的稳健性的能力的根本限制。通过数学证明和经验证据，我们证明这种权衡是训练过程中不同类别概念之间形成尖锐边界的自然结果。我们的发现表明，互补原理，量子物理学的基石，适用于神经网络，对其同时学习共轭特征的能力施加了基本限制。与此同时，我们的工作表明，仅通过单一网络架构或海量数据集实现人类级别的智能可能天生就是有限的。我们的工作为神经网络脆弱性的理论基础提供了新的见解，并为设计更健壮的神经网络结构开辟了道路。



## **9. Towards an End-to-End (E2E) Adversarial Learning and Application in the Physical World**

迈向物理世界中的端到端（E2 E）对抗性学习和应用 cs.CV

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2501.08258v2) [paper-pdf](http://arxiv.org/pdf/2501.08258v2)

**Authors**: Dudi Biton, Jacob Shams, Satoru Koda, Asaf Shabtai, Yuval Elovici, Ben Nassi

**Abstract**: The traditional learning process of patch-based adversarial attacks, conducted in the digital domain and then applied in the physical domain (e.g., via printed stickers), may suffer from reduced performance due to adversarial patches' limited transferability from the digital domain to the physical domain. Given that previous studies have considered using projectors to apply adversarial attacks, we raise the following question: can adversarial learning (i.e., patch generation) be performed entirely in the physical domain with a projector? In this work, we propose the Physical-domain Adversarial Patch Learning Augmentation (PAPLA) framework, a novel end-to-end (E2E) framework that converts adversarial learning from the digital domain to the physical domain using a projector. We evaluate PAPLA across multiple scenarios, including controlled laboratory settings and realistic outdoor environments, demonstrating its ability to ensure attack success compared to conventional digital learning-physical application (DL-PA) methods. We also analyze the impact of environmental factors, such as projection surface color, projector strength, ambient light, distance, and angle of the target object relative to the camera, on the effectiveness of projected patches. Finally, we demonstrate the feasibility of the attack against a parked car and a stop sign in a real-world outdoor environment. Our results show that under specific conditions, E2E adversarial learning in the physical domain eliminates the transferability issue and ensures evasion by object detectors. Finally, we provide insights into the challenges and opportunities of applying adversarial learning in the physical domain and explain where such an approach is more effective than using a sticker.

摘要: 基于补丁的对抗性攻击的传统学习过程在数字域中进行，然后应用于物理域(例如，通过打印的贴纸)，由于敌意补丁从数字域到物理域的可转移性有限，因此可能会受到性能降低的影响。鉴于之前的研究已经考虑使用投影仪来应用对抗性攻击，我们提出了以下问题：对抗性学习(即补丁生成)可以完全在物理领域使用投影仪执行吗？在这项工作中，我们提出了物理域对抗性补丁学习增强(PAPLA)框架，这是一个新的端到端(E2E)框架，使用投影仪将对抗性学习从数字域转换到物理域。我们在多个场景中对PAPLA进行评估，包括受控的实验室环境和真实的室外环境，展示了与传统的数字学习-物理应用(DL-PA)方法相比，它能够确保攻击成功。我们还分析了投影面颜色、投影仪强度、环境光、目标对象相对于摄像机的距离和角度等环境因素对投影块效果的影响。最后，我们在一个真实的室外环境中演示了攻击一辆停放的汽车和一个停车标志的可行性。我们的结果表明，在特定条件下，物理域中的E2E对抗性学习消除了可转移性问题，并确保了对象检测器的规避。最后，我们提供了对在物理领域应用对抗性学习的挑战和机会的见解，并解释了这种方法在哪里比使用贴纸更有效。



## **10. Direct Unlearning Optimization for Robust and Safe Text-to-Image Models**

针对稳健且安全的文本到图像模型的直接取消学习优化 cs.CV

This paper has been accepted for NeurIPS 2024

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2407.21035v2) [paper-pdf](http://arxiv.org/pdf/2407.21035v2)

**Authors**: Yong-Hyun Park, Sangdoo Yun, Jin-Hwa Kim, Junho Kim, Geonhui Jang, Yonghyun Jeong, Junghyo Jo, Gayoung Lee

**Abstract**: Recent advancements in text-to-image (T2I) models have unlocked a wide range of applications but also present significant risks, particularly in their potential to generate unsafe content. To mitigate this issue, researchers have developed unlearning techniques to remove the model's ability to generate potentially harmful content. However, these methods are easily bypassed by adversarial attacks, making them unreliable for ensuring the safety of generated images. In this paper, we propose Direct Unlearning Optimization (DUO), a novel framework for removing Not Safe For Work (NSFW) content from T2I models while preserving their performance on unrelated topics. DUO employs a preference optimization approach using curated paired image data, ensuring that the model learns to remove unsafe visual concepts while retaining unrelated features. Furthermore, we introduce an output-preserving regularization term to maintain the model's generative capabilities on safe content. Extensive experiments demonstrate that DUO can robustly defend against various state-of-the-art red teaming methods without significant performance degradation on unrelated topics, as measured by FID and CLIP scores. Our work contributes to the development of safer and more reliable T2I models, paving the way for their responsible deployment in both closed-source and open-source scenarios.

摘要: 文本到图像(T2I)模型的最新进展解锁了广泛的应用，但也带来了重大风险，特别是在生成不安全内容的潜力方面。为了缓解这个问题，研究人员开发了遗忘技术，以消除该模型生成潜在有害内容的能力。然而，这些方法很容易被敌意攻击绕过，使得它们对于确保生成图像的安全是不可靠的。在本文中，我们提出了直接遗忘优化(DUO)，这是一个新的框架，用于从T2I模型中删除不安全的工作(NSFW)内容，同时保持它们在无关主题上的性能。Duo采用了一种使用精选配对图像数据的偏好优化方法，确保模型学习删除不安全的视觉概念，同时保留不相关的特征。此外，我们引入了保持输出的正则化项来保持模型对安全内容的生成能力。广泛的实验表明，Duo可以稳健地防御各种最先进的红色团队方法，而不会在无关主题上显著降低性能，这是通过FID和CLIP分数来衡量的。我们的工作有助于开发更安全、更可靠的T2I模型，为它们在封闭源码和开放源码场景中负责任地部署铺平道路。



## **11. TPIA: Towards Target-specific Prompt Injection Attack against Code-oriented Large Language Models**

TPIA：针对面向代码的大型语言模型的特定目标提示注入攻击 cs.CR

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2407.09164v4) [paper-pdf](http://arxiv.org/pdf/2407.09164v4)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely exploited to simplify and facilitate programming. With these tools, developers can easily generate the desired complete functional code based on incomplete code snippets and natural language prompts. Unfortunately, a few pioneering works revealed that these Code LLMs are vulnerable to backdoor and adversarial attacks. The former poisons the training data or model parameters, hijacking the LLMs to generate malicious code snippets when encountering the trigger. The latter crafts malicious adversarial input codes to reduce the quality of the generated codes. However, both attacks have some inherent limitations: backdoor attacks rely on the adversary's capability of controlling the model training process; adversarial attacks struggle with fulfilling specific malicious purposes. This paper presents a novel attack paradigm against Code LLMs, namely target-specific prompt injection attack (TPIA). TPIA generates non-functional perturbations containing the information of malicious instructions and inserts them into the victim's code context by spreading them into potentially used dependencies (e.g., packages or RAG's knowledge base). It induces the Code LLMs to generate attacker-specified malicious code snippets at the target location. In general, we compress the attacker-specified malicious objective into the perturbation by adversarial optimization based on greedy token search. We collect 13 representative malicious objectives to design 31 threat cases for three popular programming languages. We show that our TPIA can successfully attack three representative open-source Code LLMs (with an ASR of up to 97.9%) and two mainstream commercial Code LLM-integrated applications (with an ASR of over 90%) in all threat cases, using only a 12-token perturbation. Our work alerts a new practical threat of using Code LLMs.

摘要: 最近，面向代码的大型语言模型(Code LLM)已被广泛利用来简化和促进编程。使用这些工具，开发人员可以根据不完整的代码片段和自然语言提示轻松生成所需的完整功能代码。不幸的是，一些开创性的工作表明，这些代码LLM容易受到后门和对手的攻击。前者毒化训练数据或模型参数，在遇到触发器时劫持LLMS生成恶意代码片段。后者制作恶意敌意输入代码以降低生成代码的质量。然而，这两种攻击都有一些固有的局限性：后门攻击依赖于对手控制模型训练过程的能力；对抗性攻击难以实现特定的恶意目的。提出了一种新的针对Code LLMS的攻击范式，即目标特定的即时注入攻击(TPIA)。TPIA生成包含恶意指令信息的非功能性扰动，并通过将它们传播到可能使用的依赖项(例如，包或RAG的知识库)，将它们插入到受害者的代码上下文中。它诱导代码LLM在目标位置生成攻击者指定的恶意代码片段。一般而言，我们通过基于贪婪令牌搜索的对抗性优化将攻击者指定的恶意目标压缩为扰动。我们收集了13个具有代表性的恶意目标，为三种流行的编程语言设计了31个威胁案例。实验表明，在所有威胁情况下，仅需12个令牌的扰动，我们的TPIA就能成功攻击三个典型的开源代码LLMS(ASR高达97.9%)和两个主流商业代码LLM集成应用(ASR超过90%)。我们的工作警示了使用Code LLMS的新的实际威胁。



## **12. Jodes: Efficient Oblivious Join in the Distributed Setting**

Jodes：分布式环境中高效的不经意加入 cs.CR

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2501.09334v1) [paper-pdf](http://arxiv.org/pdf/2501.09334v1)

**Authors**: Yilei Wang, Xiangdong Zeng, Sheng Wang, Feifei Li

**Abstract**: Trusted execution environment (TEE) has provided an isolated and secure environment for building cloud-based analytic systems, but it still suffers from access pattern leakages caused by side-channel attacks. To better secure the data, computation inside TEE enclave should be made oblivious, which introduces significant overhead and severely slows down the computation. A natural way to speed up is to build the analytic system with multiple servers in the distributed setting. However, this setting raises a new security concern -- the volumes of the transmissions among these servers can leak sensitive information to a network adversary. Existing works have designed specialized algorithms to address this concern, but their supports for equi-join, one of the most important but non-trivial database operators, are either inefficient, limited, or under a weak security assumption.   In this paper, we present Jodes, an efficient oblivious join algorithm in the distributed setting. Jodes prevents the leakage on both the network and enclave sides, supports a general equi-join operation, and provides a high security level protection that only publicizes the input sizes and the output size. Meanwhile, it achieves both communication cost and computation cost asymptotically superior to existing algorithms. To demonstrate the practicality of Jodes, we conduct experiments in the distributed setting comprising 16 servers. Empirical results show that Jodes achieves up to a sixfold performance improvement over state-of-the-art join algorithms.

摘要: 可信执行环境(TEE)为构建基于云的分析系统提供了一个隔离和安全的环境，但它仍然存在侧通道攻击导致的访问模式泄漏问题。为了更好地保护数据，应该忽略Te Enclave内的计算，这会引入大量开销并严重减慢计算速度。提高速度的一种自然方法是在分布式环境中构建具有多台服务器的分析系统。然而，此设置引发了一个新的安全问题--这些服务器之间的传输量可能会将敏感信息泄露给网络对手。已有的工作已经设计了专门的算法来解决这一问题，但他们对Equi-Join的支持要么效率低下，要么有限，要么在弱的安全假设下。本文提出了一种高效的分布式不经意连接算法Jodes。Jodes防止网络端和Enclave端的泄漏，支持通用的等连接操作，并提供只公布输入大小和输出大小的高安全级别保护。同时，该算法的通信代价和计算代价均渐近优于已有算法。为了验证Jodes的实用性，我们在由16台服务器组成的分布式环境中进行了实验。实验结果表明，与现有的连接算法相比，Jodes算法的性能提高了6倍。



## **13. Cooperative Decentralized Backdoor Attacks on Vertical Federated Learning**

垂直联邦学习的合作去中心后门攻击 cs.LG

This paper is currently under review in the IEEE/ACM Transactions on  Networking Special Issue on AI and Networking

**SubmitDate**: 2025-01-16    [abs](http://arxiv.org/abs/2501.09320v1) [paper-pdf](http://arxiv.org/pdf/2501.09320v1)

**Authors**: Seohyun Lee, Wenzhi Fang, Anindya Bijoy Das, Seyyedali Hosseinalipour, David J. Love, Christopher G. Brinton

**Abstract**: Federated learning (FL) is vulnerable to backdoor attacks, where adversaries alter model behavior on target classification labels by embedding triggers into data samples. While these attacks have received considerable attention in horizontal FL, they are less understood for vertical FL (VFL), where devices hold different features of the samples, and only the server holds the labels. In this work, we propose a novel backdoor attack on VFL which (i) does not rely on gradient information from the server and (ii) considers potential collusion among multiple adversaries for sample selection and trigger embedding. Our label inference model augments variational autoencoders with metric learning, which adversaries can train locally. A consensus process over the adversary graph topology determines which datapoints to poison. We further propose methods for trigger splitting across the adversaries, with an intensity-based implantation scheme skewing the server towards the trigger. Our convergence analysis reveals the impact of backdoor perturbations on VFL indicated by a stationarity gap for the trained model, which we verify empirically as well. We conduct experiments comparing our attack with recent backdoor VFL approaches, finding that ours obtains significantly higher success rates for the same main task performance despite not using server information. Additionally, our results verify the impact of collusion on attack performance.

摘要: 联邦学习(FL)容易受到后门攻击，即攻击者通过在数据样本中嵌入触发器来改变目标分类标签上的模型行为。虽然这些攻击在水平FL中得到了相当大的关注，但对于垂直FL(VFL)，人们对它们的理解较少，在垂直FL中，设备持有不同的样本特征，只有服务器持有标签。在这项工作中，我们提出了一种新的针对VFL的后门攻击，该攻击(I)不依赖于来自服务器的梯度信息，(Ii)在样本选择和触发器嵌入时考虑多个对手之间的潜在合谋。我们的标签推理模型使用度量学习来增强变分自动编码器，攻击者可以在本地训练这些度量学习。敌方图拓扑上的共识过程决定了要毒化哪些数据点。我们进一步提出了跨对手的触发器拆分方法，其中基于强度的植入方案使服务器向触发器倾斜。我们的收敛分析揭示了后门扰动对VFL的影响，训练模型的平稳性缺口表明了这一点，我们也通过经验验证了这一点。我们进行了实验，将我们的攻击与最近的后门VFL方法进行了比较，发现在不使用服务器信息的情况下，我们的攻击在相同的主任务性能上获得了显著更高的成功率。此外，我们的结果验证了合谋对攻击性能的影响。



## **14. Salient Information Preserving Adversarial Training Improves Clean and Robust Accuracy**

突出信息保留对抗训练提高清晰且稳健的准确性 cs.CV

**SubmitDate**: 2025-01-15    [abs](http://arxiv.org/abs/2501.09086v1) [paper-pdf](http://arxiv.org/pdf/2501.09086v1)

**Authors**: Timothy Redgrave, Adam Czajka

**Abstract**: In this work we introduce Salient Information Preserving Adversarial Training (SIP-AT), an intuitive method for relieving the robustness-accuracy trade-off incurred by traditional adversarial training. SIP-AT uses salient image regions to guide the adversarial training process in such a way that fragile features deemed meaningful by an annotator remain unperturbed during training, allowing models to learn highly predictive non-robust features without sacrificing overall robustness. This technique is compatible with both human-based and automatically generated salience estimates, allowing SIP-AT to be used as a part of human-driven model development without forcing SIP-AT to be reliant upon additional human data. We perform experiments across multiple datasets and architectures and demonstrate that SIP-AT is able to boost the clean accuracy of models while maintaining a high degree of robustness against attacks at multiple epsilon levels. We complement our central experiments with an observational study measuring the rate at which human subjects successfully identify perturbed images. This study helps build a more intuitive understanding of adversarial attack strength and demonstrates the heightened importance of low-epsilon robustness. Our results demonstrate the efficacy of SIP-AT and provide valuable insight into the risks posed by adversarial samples of various strengths.

摘要: 在这项工作中，我们引入了突出信息保存对抗训练(SIP-AT)，这是一种直观的方法，可以缓解传统对抗训练带来的稳健性和准确性之间的权衡。SIP-AT使用显著的图像区域来指导对抗性训练过程，使得被注释者认为有意义的脆弱特征在训练期间保持不受干扰，从而允许模型在不牺牲整体稳健性的情况下学习高度预测的非稳健特征。该技术与基于人工的和自动生成的显著度估计都兼容，允许将SIP-AT用作人工驱动的模型开发的一部分，而无需强制SIP-AT依赖额外的人工数据。我们在多个数据集和体系结构上进行了实验，证明了SIP-AT能够提高模型的干净准确性，同时保持对多个epsilon级别的攻击的高度健壮性。我们用一项观察性研究来补充我们的中心实验，该研究测量了人类受试者成功识别受干扰图像的速度。这项研究有助于建立对对手攻击强度的更直观的理解，并证明了低epsilon稳健性的高度重要性。我们的结果证明了SIP-AT的有效性，并为不同强度的对抗性样本所构成的风险提供了有价值的见解。



## **15. Improving Stability Estimates in Adversarial Explainable AI through Alternate Search Methods**

通过替代搜索方法改进对抗性可解释人工智能的稳定性估计 cs.LG

9 pages, 3 figures, 5 tables. arXiv admin note: text overlap with  arXiv:2406.15839

**SubmitDate**: 2025-01-15    [abs](http://arxiv.org/abs/2501.09006v1) [paper-pdf](http://arxiv.org/pdf/2501.09006v1)

**Authors**: Christopher Burger, Charles Walter

**Abstract**: Advances in the effectiveness of machine learning models have come at the cost of enormous complexity resulting in a poor understanding of how they function. Local surrogate methods have been used to approximate the workings of these complex models, but recent work has revealed their vulnerability to adversarial attacks where the explanation produced is appreciably different while the meaning and structure of the complex model's output remains similar. This prior work has focused on the existence of these weaknesses but not on their magnitude. Here we explore using an alternate search method with the goal of finding minimum viable perturbations, the fewest perturbations necessary to achieve a fixed similarity value between the original and altered text's explanation. Intuitively, a method that requires fewer perturbations to expose a given level of instability is inferior to one which requires more. This nuance allows for superior comparisons of the stability of explainability methods.

摘要: 机器学习模型有效性的进步是以巨大的复杂性为代价的，导致人们对它们的运作方式了解不足。局部代理方法已被用来逼近这些复杂模型的工作原理，但最近的工作揭示了它们对对抗攻击的脆弱性，其中产生的解释明显不同，而复杂模型输出的含义和结构仍然相似。之前的工作重点是这些弱点的存在，而不是它们的严重程度。在这里，我们探索使用替代搜索方法，目标是找到最小的可行扰动，即在原始文本和修改后的文本解释之间实现固定相似性值所需的最少扰动。直观地说，需要更少扰动来暴露给定水平不稳定性的方法不如需要更多扰动的方法。这种细微差别允许对可解释性方法的稳定性进行更好的比较。



## **16. UIFV: Data Reconstruction Attack in Vertical Federated Learning**

UIFV：垂直联邦学习中的数据重建攻击 cs.LG

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2406.12588v2) [paper-pdf](http://arxiv.org/pdf/2406.12588v2)

**Authors**: Jirui Yang, Peng Chen, Zhihui Lu, Qiang Duan, Yubing Bao

**Abstract**: Vertical Federated Learning (VFL) facilitates collaborative machine learning without the need for participants to share raw private data. However, recent studies have revealed privacy risks where adversaries might reconstruct sensitive features through data leakage during the learning process. Although data reconstruction methods based on gradient or model information are somewhat effective, they reveal limitations in VFL application scenarios. This is because these traditional methods heavily rely on specific model structures and/or have strict limitations on application scenarios. To address this, our study introduces the Unified InverNet Framework into VFL, which yields a novel and flexible approach (dubbed UIFV) that leverages intermediate feature data to reconstruct original data, instead of relying on gradients or model details. The intermediate feature data is the feature exchanged by different participants during the inference phase of VFL. Experiments on four datasets demonstrate that our methods significantly outperform state-of-the-art techniques in attack precision. Our work exposes severe privacy vulnerabilities within VFL systems that pose real threats to practical VFL applications and thus confirms the necessity of further enhancing privacy protection in the VFL architecture.

摘要: 垂直联合学习(VFL)促进了协作机器学习，而不需要参与者共享原始私有数据。然而，最近的研究揭示了隐私风险，攻击者可能会在学习过程中通过数据泄露来重建敏感特征。虽然基于梯度或模型信息的数据重建方法在一定程度上是有效的，但它们在VFL应用场景中暴露出局限性。这是因为这些传统方法严重依赖于特定的模型结构和/或对应用场景有严格的限制。为了解决这一问题，我们的研究将统一InverNet框架引入到VFL中，从而产生了一种新颖而灵活的方法(称为UIFV)，该方法利用中间特征数据来重建原始数据，而不是依赖于梯度或模型细节。中间特征数据是不同参与者在VFL推理阶段交换的特征。在四个数据集上的实验表明，我们的方法在攻击精度上明显优于最先进的技术。我们的工作暴露了VFL系统中严重的隐私漏洞，这些漏洞对实际的VFL应用构成了真正的威胁，从而证实了在VFL体系结构中进一步加强隐私保护的必要性。



## **17. Cross-Modal Transferable Image-to-Video Attack on Video Quality Metrics**

对视频质量分配器的跨模式可传输图像到视频攻击 cs.CV

Accepted for VISAPP 2025

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.08415v1) [paper-pdf](http://arxiv.org/pdf/2501.08415v1)

**Authors**: Georgii Gotin, Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: Recent studies have revealed that modern image and video quality assessment (IQA/VQA) metrics are vulnerable to adversarial attacks. An attacker can manipulate a video through preprocessing to artificially increase its quality score according to a certain metric, despite no actual improvement in visual quality. Most of the attacks studied in the literature are white-box attacks, while black-box attacks in the context of VQA have received less attention. Moreover, some research indicates a lack of transferability of adversarial examples generated for one model to another when applied to VQA. In this paper, we propose a cross-modal attack method, IC2VQA, aimed at exploring the vulnerabilities of modern VQA models. This approach is motivated by the observation that the low-level feature spaces of images and videos are similar. We investigate the transferability of adversarial perturbations across different modalities; specifically, we analyze how adversarial perturbations generated on a white-box IQA model with an additional CLIP module can effectively target a VQA model. The addition of the CLIP module serves as a valuable aid in increasing transferability, as the CLIP model is known for its effective capture of low-level semantics. Extensive experiments demonstrate that IC2VQA achieves a high success rate in attacking three black-box VQA models. We compare our method with existing black-box attack strategies, highlighting its superiority in terms of attack success within the same number of iterations and levels of attack strength. We believe that the proposed method will contribute to the deeper analysis of robust VQA metrics.

摘要: 最近的研究表明，现代图像和视频质量评估(IQA/VQA)指标容易受到对手攻击。攻击者可以通过预处理来操纵视频，以根据特定的度量人为地提高其质量分数，尽管视频质量没有实际的改善。文献中研究的攻击大多是白盒攻击，而基于VQA的黑盒攻击较少受到关注。此外，一些研究表明，当应用于VQA时，为一个模型生成的对抗性例子缺乏到另一个模型的可转移性。针对现代VQA模型的脆弱性，提出了一种跨模式攻击方法IC2VQA。这种方法的动机是观察到图像和视频的低层特征空间是相似的。我们研究了对抗性扰动在不同模式之间的可转移性；具体地说，我们分析了在带有附加CLIP模块的白盒IQA模型上产生的对抗性扰动如何有效地针对VQA模型。添加CLIP模块有助于提高可转移性，因为CLIP模型以其对低级语义的有效捕获而闻名。大量实验表明，IC2VQA在攻击三种黑盒VQA模型时取得了较高的成功率。我们将我们的方法与现有的黑盒攻击策略进行了比较，突出了它在相同迭代次数和攻击强度级别下的攻击成功率方面的优势。我们相信，所提出的方法将有助于对稳健的VQA度量进行更深入的分析。



## **18. Exploring Robustness of LLMs to Sociodemographically-Conditioned Paraphrasing**

探索LLM对社会人口学条件解释的稳健性 cs.CL

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.08276v1) [paper-pdf](http://arxiv.org/pdf/2501.08276v1)

**Authors**: Pulkit Arora, Akbar Karimi, Lucie Flek

**Abstract**: Large Language Models (LLMs) have shown impressive performance in various NLP tasks. However, there are concerns about their reliability in different domains of linguistic variations. Many works have proposed robustness evaluation measures for local adversarial attacks, but we need globally robust models unbiased to different language styles. We take a broader approach to explore a wider range of variations across sociodemographic dimensions to perform structured reliability tests on the reasoning capacity of language models. We extend the SocialIQA dataset to create diverse paraphrased sets conditioned on sociodemographic styles. The assessment aims to provide a deeper understanding of LLMs in (a) their capability of generating demographic paraphrases with engineered prompts and (b) their reasoning capabilities in real-world, complex language scenarios. We also explore measures such as perplexity, explainability, and ATOMIC performance of paraphrases for fine-grained reliability analysis of LLMs on these sets. We find that demographic-specific paraphrasing significantly impacts the performance of language models, indicating that the subtleties of language variations remain a significant challenge. The code and dataset will be made available for reproducibility and future research.

摘要: 大型语言模型(LLM)在各种NLP任务中表现出了令人印象深刻的表现。然而，人们对它们在不同语言变异领域的可靠性表示担忧。已有许多研究提出了针对局部对抗性攻击的健壮性评估方法，但我们需要对不同的语言风格不偏向的全局健壮性模型。我们采取更广泛的方法来探索社会人口统计维度的更大范围的变化，以对语言模型的推理能力进行结构化的可靠性测试。我们扩展了SocialIQA数据集，以创建不同的释义集，条件是社会人口统计风格。这项评估旨在加深对LLMS的理解：(A)它们通过设计提示生成人口统计释义的能力；(B)它们在现实世界复杂语言情景中的推理能力。我们还探索了诸如困惑、可解释性和释义的原子性能等度量，用于这些集合上的LLMS的细粒度可靠性分析。我们发现，特定于人口统计的释义显著影响了语言模型的性能，这表明语言变异的微妙之处仍然是一个重大挑战。代码和数据集将提供给可重复性和未来的研究。



## **19. SoK: Design, Vulnerabilities, and Security Measures of Cryptocurrency Wallets**

SoK：加密货币钱包的设计、漏洞和安全措施 cs.CR

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2307.12874v4) [paper-pdf](http://arxiv.org/pdf/2307.12874v4)

**Authors**: Yimika Erinle, Yathin Kethepalli, Yebo Feng, Jiahua Xu

**Abstract**: With the advent of decentralised digital currencies powered by blockchain technology, a new era of peer-to-peer transactions has commenced. The rapid growth of the cryptocurrency economy has led to increased use of transaction-enabling wallets, making them a focal point for security risks. As the frequency of wallet-related incidents rises, there is a critical need for a systematic approach to measure and evaluate these attacks, drawing lessons from past incidents to enhance wallet security. In response, we introduce a multi-dimensional design taxonomy for existing and novel wallets with various design decisions. We classify existing industry wallets based on this taxonomy, identify previously occurring vulnerabilities and discuss the security implications of design decisions. We also systematise threats to the wallet mechanism and analyse the adversary's goals, capabilities and required knowledge. We present a multi-layered attack framework and investigate 84 incidents between 2012 and 2024, accounting for $5.4B. Following this, we classify defence implementations for these attacks on the precautionary and remedial axes. We map the mechanism and design decisions to vulnerabilities, attacks, and possible defence methods to discuss various insights.

摘要: 随着区块链技术驱动的去中心化数字货币的到来，P2P交易的新时代已经开始。加密货币经济的快速增长导致更多人使用支持交易的钱包，使其成为安全风险的焦点。随着与钱包有关的事件的频率上升，迫切需要一种系统的方法来衡量和评估这些攻击，并从过去的事件中吸取教训，以加强钱包的安全。作为回应，我们为现有的和具有不同设计决策的新钱包引入了多维设计分类。我们根据这一分类对现有的行业钱包进行分类，识别以前发生的漏洞，并讨论设计决策的安全影响。我们还对钱包机制面临的威胁进行了系统分析，并分析了对手的目标、能力和所需的知识。我们提出了一个多层攻击框架，并调查了2012至2024年间的84起事件，金额为54亿美元。然后，我们从预防和补救两个方面对这些攻击的防御实施进行了分类。我们将机制和设计决策映射到漏洞、攻击和可能的防御方法，以讨论各种见解。



## **20. I Can Find You in Seconds! Leveraging Large Language Models for Code Authorship Attribution**

我可以在几秒钟内找到你！利用大型语言模型进行代码作者归因 cs.SE

12 pages, 5 figures,

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.08165v1) [paper-pdf](http://arxiv.org/pdf/2501.08165v1)

**Authors**: Soohyeon Choi, Yong Kiam Tan, Mark Huasong Meng, Mohamed Ragab, Soumik Mondal, David Mohaisen, Khin Mi Mi Aung

**Abstract**: Source code authorship attribution is important in software forensics, plagiarism detection, and protecting software patch integrity. Existing techniques often rely on supervised machine learning, which struggles with generalization across different programming languages and coding styles due to the need for large labeled datasets. Inspired by recent advances in natural language authorship analysis using large language models (LLMs), which have shown exceptional performance without task-specific tuning, this paper explores the use of LLMs for source code authorship attribution.   We present a comprehensive study demonstrating that state-of-the-art LLMs can successfully attribute source code authorship across different languages. LLMs can determine whether two code snippets are written by the same author with zero-shot prompting, achieving a Matthews Correlation Coefficient (MCC) of 0.78, and can attribute code authorship from a small set of reference code snippets via few-shot learning, achieving MCC of 0.77. Additionally, LLMs show some adversarial robustness against misattribution attacks.   Despite these capabilities, we found that naive prompting of LLMs does not scale well with a large number of authors due to input token limitations. To address this, we propose a tournament-style approach for large-scale attribution. Evaluating this approach on datasets of C++ (500 authors, 26,355 samples) and Java (686 authors, 55,267 samples) code from GitHub, we achieve classification accuracy of up to 65% for C++ and 68.7% for Java using only one reference per author. These results open new possibilities for applying LLMs to code authorship attribution in cybersecurity and software engineering.

摘要: 源代码作者归属在软件取证、抄袭检测和保护软件补丁完整性方面非常重要。现有技术通常依赖于有监督的机器学习，由于需要大型标签数据集，因此难以跨不同的编程语言和编码风格进行泛化。受使用大语言模型(LLM)的自然语言作者分析的最新进展的启发，LLM在没有特定任务调整的情况下表现出了优异的性能，本文探索了LLM在源代码作者归属中的使用。我们提出了一项全面的研究，证明了最先进的LLMS可以成功地跨不同语言归因于源代码作者。LLMS可以在零镜头提示下确定两个代码片段是否由同一作者编写，马修斯相关系数(MCC)为0.78，并可以通过少镜头学习从一小组参考代码片段中归属代码作者，MCC为0.77。此外，LLM在抵抗错误归因攻击时表现出一定的对抗健壮性。尽管有这些功能，但我们发现，由于输入令牌的限制，LLM的幼稚提示不能很好地扩展到大量作者。为了解决这个问题，我们提出了一种锦标赛式的大规模归因方法。在GitHub的C++(500名作者，26,355个样本)和Java(686名作者，55,267个样本)代码的数据集上测试该方法，在每个作者只使用一个参考文献的情况下，C++的分类准确率达到65%，Java的分类准确率达到68.7%。这些结果为将LLMS应用于网络安全和软件工程中的代码作者归属开辟了新的可能性。



## **21. Set-Based Training for Neural Network Verification**

神经网络验证的基于集的训练 cs.LG

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2401.14961v3) [paper-pdf](http://arxiv.org/pdf/2401.14961v3)

**Authors**: Lukas Koller, Tobias Ladner, Matthias Althoff

**Abstract**: Neural networks are vulnerable to adversarial attacks, i.e., small input perturbations can significantly affect the outputs of a neural network. Therefore, to ensure safety of safety-critical environments, the robustness of a neural network must be formally verified against input perturbations, e.g., from noisy sensors. To improve the robustness of neural networks and thus simplify the formal verification, we present a novel set-based training procedure in which we compute the set of possible outputs given the set of possible inputs and compute for the first time a gradient set, i.e., each possible output has a different gradient. Therefore, we can directly reduce the size of the output enclosure by choosing gradients toward its center. Small output enclosures increase the robustness of a neural network and, at the same time, simplify its formal verification. The latter benefit is due to the fact that a larger size of propagated sets increases the conservatism of most verification methods. Our extensive evaluation demonstrates that set-based training produces robust neural networks with competitive performance, which can be verified using fast (polynomial-time) verification algorithms due to the reduced output set.

摘要: 神经网络容易受到敌意攻击，即微小的输入扰动会显著影响神经网络的输出。因此，为了确保安全关键环境的安全，神经网络的稳健性必须针对输入扰动进行正式验证，例如，来自噪声传感器的扰动。为了提高神经网络的稳健性，从而简化形式化验证，我们提出了一种新的基于集合的训练过程，在该过程中，我们计算给定可能输入的集合的可能输出的集合，并首次计算梯度集合，即每个可能的输出具有不同的梯度。因此，我们可以通过选择朝向其中心的渐变来直接减小输出外壳的大小。较小的输出封闭增加了神经网络的健壮性，同时简化了其形式验证。后一种好处是由于较大的传播集合的大小增加了大多数验证方法的保守性。我们的广泛评估表明，基于集合的训练产生了性能具有竞争力的健壮神经网络，由于减少了输出集合，可以使用快速(多项式时间)验证算法进行验证。



## **22. Gandalf the Red: Adaptive Security for LLMs**

红色甘道夫：LLM的自适应安全 cs.LG

Niklas Pfister, V\'aclav Volhejn and Manuel Knott contributed equally

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.07927v1) [paper-pdf](http://arxiv.org/pdf/2501.07927v1)

**Authors**: Niklas Pfister, Václav Volhejn, Manuel Knott, Santiago Arias, Julia Bazińska, Mykhailo Bichurin, Alan Commike, Janet Darling, Peter Dienes, Matthew Fiedler, David Haber, Matthias Kraft, Marco Lancini, Max Mathys, Damián Pascual-Ortiz, Jakub Podolak, Adrià Romero-López, Kyriacos Shiarlis, Andreas Signer, Zsolt Terek, Athanasios Theocharis, Daniel Timbrell, Samuel Trautwein, Samuel Watts, Natalie Wu, Mateo Rojas-Carulla

**Abstract**: Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and rigorously expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack datasets. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications. Code is available at \href{https://github.com/lakeraai/dsec-gandalf}{\texttt{https://github.com/lakeraai/dsec-gandalf}}.

摘要: 目前对大型语言模型(LLM)应用程序中针对即时攻击的防御措施的评估往往忽略了两个关键因素：敌意行为的动态性质和限制性防御措施对合法用户的可用性惩罚。提出了动态安全效用威胁模型D-SEC(Dynamic Security Utility Threat Model)，该模型明确地将攻击者和合法用户分开，对多步交互进行建模，并以可优化的形式严格表达安全效用。我们通过引入Gandalf进一步解决了现有评估中的缺陷，Gandalf是一个众包、游戏化的红色团队平台，旨在生成现实的、自适应的攻击数据集。使用Gandalf，我们收集并发布了279K提示攻击的数据集。在良性用户数据的补充下，我们的分析揭示了安全性和实用性之间的相互作用，表明LLM中集成的防御措施(例如系统提示)即使在不阻止请求的情况下也会降低可用性。我们演示了受限应用程序域、深度防御和自适应防御是构建安全且有用的LLM应用程序的有效策略。代码可在\href{https://github.com/lakeraai/dsec-gandalf}{\texttt{https://github.com/lakeraai/dsec-gandalf}}.上找到



## **23. VENOM: Text-driven Unrestricted Adversarial Example Generation with Diffusion Models**

VENOM：使用扩散模型的文本驱动无限制对抗示例生成 cs.CV

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2501.07922v1) [paper-pdf](http://arxiv.org/pdf/2501.07922v1)

**Authors**: Hui Kuurila-Zhang, Haoyu Chen, Guoying Zhao

**Abstract**: Adversarial attacks have proven effective in deceiving machine learning models by subtly altering input images, motivating extensive research in recent years. Traditional methods constrain perturbations within $l_p$-norm bounds, but advancements in Unrestricted Adversarial Examples (UAEs) allow for more complex, generative-model-based manipulations. Diffusion models now lead UAE generation due to superior stability and image quality over GANs. However, existing diffusion-based UAE methods are limited to using reference images and face challenges in generating Natural Adversarial Examples (NAEs) directly from random noise, often producing uncontrolled or distorted outputs. In this work, we introduce VENOM, the first text-driven framework for high-quality unrestricted adversarial examples generation through diffusion models. VENOM unifies image content generation and adversarial synthesis into a single reverse diffusion process, enabling high-fidelity adversarial examples without sacrificing attack success rate (ASR). To stabilize this process, we incorporate an adaptive adversarial guidance strategy with momentum, ensuring that the generated adversarial examples $x^*$ align with the distribution $p(x)$ of natural images. Extensive experiments demonstrate that VENOM achieves superior ASR and image quality compared to prior methods, marking a significant advancement in adversarial example generation and providing insights into model vulnerabilities for improved defense development.

摘要: 对抗性攻击已被证明通过微妙地改变输入图像来欺骗机器学习模型是有效的，这激发了近年来的广泛研究。传统的方法将扰动限制在$L_p$-范数范围内，但不受限制的对抗性例子(UAE)的进步允许更复杂的、基于生成模型的操作。由于优于Gans的稳定性和图像质量，扩散模型现在引领着阿联酋的生成。然而，现有的基于扩散的UAE方法仅限于使用参考图像，并且在直接从随机噪声生成自然对抗性实例(NAE)方面面临挑战，通常会产生失控或失真的输出。在这项工作中，我们介绍了VOOM，这是第一个文本驱动的框架，用于通过扩散模型生成高质量的不受限制的对抗性实例。毒液将图像内容生成和对抗性合成统一到单个反向扩散过程中，在不牺牲攻击成功率(ASR)的情况下实现高保真对抗性示例。为了稳定这一过程，我们结合了一种带有动量的自适应对抗性指导策略，确保生成的对抗性实例$x^*$与自然图像的分布$p(X)$一致。广泛的实验表明，与以前的方法相比，VOOM实现了更好的ASR和图像质量，标志着在对抗性示例生成方面取得了重大进步，并为改进防御开发提供了对模型漏洞的洞察。



## **24. Can Go AIs be adversarially robust?**

Go AI能否具有对抗性强大？ cs.LG

63 pages, AAAI 2025

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2406.12843v3) [paper-pdf](http://arxiv.org/pdf/2406.12843v3)

**Authors**: Tom Tseng, Euan McLean, Kellin Pelrine, Tony T. Wang, Adam Gleave

**Abstract**: Prior work found that superhuman Go AIs can be defeated by simple adversarial strategies, especially "cyclic" attacks. In this paper, we study whether adding natural countermeasures can achieve robustness in Go, a favorable domain for robustness since it benefits from incredible average-case capability and a narrow, innately adversarial setting. We test three defenses: adversarial training on hand-constructed positions, iterated adversarial training, and changing the network architecture. We find that though some of these defenses protect against previously discovered attacks, none withstand freshly trained adversaries. Furthermore, most of the reliably effective attacks these adversaries discover are different realizations of the same overall class of cyclic attacks. Our results suggest that building robust AI systems is challenging even with extremely superhuman systems in some of the most tractable settings, and highlight two key gaps: efficient generalization of defenses, and diversity in training. For interactive examples of attacks and a link to our codebase, see https://goattack.far.ai.

摘要: 先前的工作发现，超人围棋可以通过简单的对抗性策略，特别是“循环”攻击来击败。在本文中，我们研究了添加自然对策是否可以在围棋中实现稳健性，这是一个有利于稳健性的领域，因为它受益于令人难以置信的平均情况能力和狭窄的、天生的对抗性环境。我们测试了三种防御措施：在手工搭建的阵地上进行对抗性训练，反复进行对抗性训练，以及改变网络架构。我们发现，尽管其中一些防御系统可以抵御以前发现的攻击，但没有一个能抵御新训练的对手。此外，这些攻击者发现的大多数可靠有效的攻击都是同一类循环攻击的不同实现。我们的结果表明，即使在一些最容易处理的环境中，使用极其超人的系统来构建稳健的人工智能系统也是具有挑战性的，并突显了两个关键差距：有效的防御泛化和训练的多样性。有关攻击的交互式示例和我们代码库的链接，请参阅https://goattack.far.ai.



## **25. Don't Command, Cultivate: An Exploratory Study of System-2 Alignment**

不命令，培养：System-2对齐的探索性研究 cs.CL

In this version, the DPO and reinforcement learning methods have been  added

**SubmitDate**: 2025-01-14    [abs](http://arxiv.org/abs/2411.17075v5) [paper-pdf](http://arxiv.org/pdf/2411.17075v5)

**Authors**: Yuhang Wang, Yuxiang Zhang, Yanxu Zhu, Xinyan Wen, Jitao Sang

**Abstract**: The o1 system card identifies the o1 models as the most robust within OpenAI, with their defining characteristic being the progression from rapid, intuitive thinking to slower, more deliberate reasoning. This observation motivated us to investigate the influence of System-2 thinking patterns on model safety. In our preliminary research, we conducted safety evaluations of the o1 model, including complex jailbreak attack scenarios using adversarial natural language prompts and mathematical encoding prompts. Our findings indicate that the o1 model demonstrates relatively improved safety performance; however, it still exhibits vulnerabilities, particularly against jailbreak attacks employing mathematical encoding. Through detailed case analysis, we identified specific patterns in the o1 model's responses. We also explored the alignment of System-2 safety in open-source models using prompt engineering and supervised fine-tuning techniques. Experimental results show that some simple methods to encourage the model to carefully scrutinize user requests are beneficial for model safety. Additionally, we proposed a implementation plan for process supervision to enhance safety alignment. The implementation details and experimental results will be provided in future versions.

摘要: O1系统卡将o1模型确定为OpenAI中最健壮的模型，它们的决定性特征是从快速、直观的思考到更慢、更深思熟虑的推理的过程。这一观察结果促使我们调查System-2思维模式对模型安全性的影响。在我们的初步研究中，我们对o1模型进行了安全性评估，包括使用对抗性自然语言提示和数学编码提示的复杂越狱攻击场景。我们的发现表明，o1模型显示出相对更好的安全性能；但是，它仍然存在漏洞，特别是对使用数学编码的越狱攻击。通过详细的案例分析，我们确定了o1模型反应的具体模式。我们还使用即时工程和有监督的微调技术探索了开源模型中System-2安全性的一致性。实验结果表明，一些简单的方法鼓励模型仔细审查用户请求，有利于模型的安全。此外，我们还提出了加强安全对接的过程监管实施方案。实现细节和实验结果将在未来的版本中提供。



## **26. A Survey of Early Exit Deep Neural Networks in NLP**

NLP中早期退出深度神经网络的综述 cs.LG

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2501.07670v1) [paper-pdf](http://arxiv.org/pdf/2501.07670v1)

**Authors**: Divya Jyoti Bajpai, Manjesh Kumar Hanawal

**Abstract**: Deep Neural Networks (DNNs) have grown increasingly large in size to achieve state of the art performance across a wide range of tasks. However, their high computational requirements make them less suitable for resource-constrained applications. Also, real-world datasets often consist of a mixture of easy and complex samples, necessitating adaptive inference mechanisms that account for sample difficulty. Early exit strategies offer a promising solution by enabling adaptive inference, where simpler samples are classified using the initial layers of the DNN, thereby accelerating the overall inference process. By attaching classifiers at different layers, early exit methods not only reduce inference latency but also improve the model robustness against adversarial attacks. This paper presents a comprehensive survey of early exit methods and their applications in NLP.

摘要: 深度神经网络（DNN）的规模越来越大，以在广泛的任务中实现最先进的性能。然而，它们的高计算要求使它们不太适合资源有限的应用程序。此外，现实世界的数据集通常由简单和复杂样本的混合组成，因此需要考虑样本难度的自适应推理机制。早期退出策略通过启用自适应推理提供了一个有希望的解决方案，即使用DNN的初始层对更简单的样本进行分类，从而加速整个推理过程。通过在不同层附加分类器，早期退出方法不仅可以减少推理延迟，还可以提高模型对对抗攻击的鲁棒性。本文对早期退出方法及其在NLP中的应用进行了全面的调查。



## **27. SecAlign: Defending Against Prompt Injection with Preference Optimization**

SecAlign：通过偏好优化抵御提示注入 cs.CR

Key words: prompt injection defense, LLM security, LLM-integrated  applications

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2410.05451v2) [paper-pdf](http://arxiv.org/pdf/2410.05451v2)

**Authors**: Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, David Wagner, Chuan Guo

**Abstract**: Large language models (LLMs) are becoming increasingly prevalent in modern software systems, interfacing between the user and the Internet to assist with tasks that require advanced language understanding. To accomplish these tasks, the LLM often uses external data sources such as user documents, web retrieval, results from API calls, etc. This opens up new avenues for attackers to manipulate the LLM via prompt injection. Adversarial prompts can be injected into external data sources to override the system's intended instruction and instead execute a malicious instruction.   To mitigate this vulnerability, we propose a new defense called SecAlign based on the technique of preference optimization. Our defense first constructs a preference dataset with prompt-injected inputs, secure outputs (ones that respond to the legitimate instruction), and insecure outputs (ones that respond to the injection). We then perform preference optimization on this dataset to teach the LLM to prefer the secure output over the insecure one. This provides the first known method that reduces the success rates of various prompt injections to around 0%, even against attacks much more sophisticated than ones seen during training. This indicates our defense generalizes well against unknown and yet-to-come attacks. Also, our defended models are still practical with similar utility to the one before our defensive training. Our code is at https://github.com/facebookresearch/SecAlign

摘要: 大型语言模型(LLM)在现代软件系统中正变得越来越普遍，它们在用户和互联网之间进行交互，以帮助完成需要高级语言理解的任务。为了完成这些任务，LLM通常使用外部数据源，如用户文档、Web检索、API调用结果等。这为攻击者通过提示注入操纵LLM开辟了新的途径。敌意提示可以被注入外部数据源，以覆盖系统的预期指令，而不是执行恶意指令。为了缓解这一漏洞，我们提出了一种新的基于偏好优化技术的防御机制SecAlign。我们的辩护首先构建一个偏好数据集，其中包含即时注入输入、安全输出(响应合法指令的输出)和不安全的输出(响应注入的输出)。然后，我们对该数据集执行偏好优化，以教导LLM更喜欢安全的输出而不是不安全的输出。这提供了第一种已知的方法，可以将各种快速注射的成功率降低到0%左右，即使是对比训练期间看到的更复杂的攻击也是如此。这表明我们的防御对未知和即将到来的攻击具有很好的一般性。此外，我们的防守模型仍然实用，与我们防守训练之前的实用程度相似。我们的代码在https://github.com/facebookresearch/SecAlign



## **28. Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards**

探索和缓解基于投票的排行榜的对抗操纵 cs.LG

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2501.07493v1) [paper-pdf](http://arxiv.org/pdf/2501.07493v1)

**Authors**: Yangsibo Huang, Milad Nasr, Anastasios Angelopoulos, Nicholas Carlini, Wei-Lin Chiang, Christopher A. Choquette-Choo, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Ken Ziyu Liu, Ion Stoica, Florian Tramer, Chiyuan Zhang

**Abstract**: It is now common to evaluate Large Language Models (LLMs) by having humans manually vote to evaluate model outputs, in contrast to typical benchmarks that evaluate knowledge or skill at some particular task. Chatbot Arena, the most popular benchmark of this type, ranks models by asking users to select the better response between two randomly selected models (without revealing which model was responsible for the generations). These platforms are widely trusted as a fair and accurate measure of LLM capabilities. In this paper, we show that if bot protection and other defenses are not implemented, these voting-based benchmarks are potentially vulnerable to adversarial manipulation. Specifically, we show that an attacker can alter the leaderboard (to promote their favorite model or demote competitors) at the cost of roughly a thousand votes (verified in a simulated, offline version of Chatbot Arena). Our attack consists of two steps: first, we show how an attacker can determine which model was used to generate a given reply with more than $95\%$ accuracy; and then, the attacker can use this information to consistently vote for (or against) a target model. Working with the Chatbot Arena developers, we identify, propose, and implement mitigations to improve the robustness of Chatbot Arena against adversarial manipulation, which, based on our analysis, substantially increases the cost of such attacks. Some of these defenses were present before our collaboration, such as bot protection with Cloudflare, malicious user detection, and rate limiting. Others, including reCAPTCHA and login are being integrated to strengthen the security in Chatbot Arena.

摘要: 现在，通过人工投票评估模型输出来评估大型语言模型(LLM)是很常见的，这与评估某些特定任务的知识或技能的典型基准测试形成了鲜明对比。聊天机器人Arena是这类最受欢迎的基准，它通过让用户在两个随机选择的模型中选择反应更好的模型来对模型进行排名(没有透露哪种模型对几代人负责)。这些平台受到广泛信任，被认为是LLM能力的公平和准确衡量标准。在这篇文章中，我们表明，如果不实施BOT保护和其他防御措施，这些基于投票的基准可能容易受到敌意操纵。具体地说，我们展示了攻击者可以改变排行榜(以推广他们最喜欢的型号或降级竞争对手)，代价是大约1000张选票(在模拟的聊天机器人竞技场的离线版本中进行验证)。我们的攻击包括两个步骤：首先，我们展示了攻击者如何确定哪个模型被用来生成精度超过$95\\$的给定回复；然后，攻击者可以使用这些信息一致地投票支持(或反对)目标模型。我们与聊天机器人Arena的开发人员合作，识别、提出并实施缓解措施，以提高聊天机器人Arena对敌意操纵的健壮性，根据我们的分析，这将大大增加此类攻击的成本。其中一些防御措施在我们合作之前就已经存在，例如使用Cloudflare的机器人保护、恶意用户检测和速率限制。包括reCAPTCHA和LOGIN在内的其他软件正在整合，以加强聊天机器人竞技场的安全性。



## **29. Agentic Copyright Watermarking against Adversarial Evidence Forgery with Purification-Agnostic Curriculum Proxy Learning**

通过净化对抗证据伪造的权威版权水印-不可知的课程代理学习 cs.CV

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2409.01541v2) [paper-pdf](http://arxiv.org/pdf/2409.01541v2)

**Authors**: Erjin Bao, Ching-Chun Chang, Hanrui Wang, Isao Echizen

**Abstract**: With the proliferation of AI agents in various domains, protecting the ownership of AI models has become crucial due to the significant investment in their development. Unauthorized use and illegal distribution of these models pose serious threats to intellectual property, necessitating effective copyright protection measures. Model watermarking has emerged as a key technique to address this issue, embedding ownership information within models to assert rightful ownership during copyright disputes. This paper presents several contributions to model watermarking: a self-authenticating black-box watermarking protocol using hash techniques, a study on evidence forgery attacks using adversarial perturbations, a proposed defense involving a purification step to counter adversarial attacks, and a purification-agnostic curriculum proxy learning method to enhance watermark robustness and model performance. Experimental results demonstrate the effectiveness of these approaches in improving the security, reliability, and performance of watermarked models.

摘要: 随着AI代理在各个领域的激增，保护AI模型的所有权变得至关重要，因为它们在开发方面投入了大量资金。未经授权使用和非法传播这些模型对知识产权构成严重威胁，需要采取有效的版权保护措施。模型水印已经成为解决这一问题的一项关键技术，它将所有权信息嵌入到模型中，以在版权纠纷中断言合法的所有权。提出了一种基于散列技术的自认证黑盒水印协议，研究了基于对抗性扰动的证据伪造攻击，提出了一种针对对抗性攻击的净化步骤防御方法，以及一种与净化无关的课程代理学习方法，以提高水印的稳健性和模型的性能。实验结果表明，这些方法在提高水印模型的安全性、可靠性和性能方面是有效的。



## **30. MOS-Attack: A Scalable Multi-objective Adversarial Attack Framework**

MOS攻击：可扩展的多目标对抗攻击框架 cs.LG

Under Review of CVPR 2025

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2501.07251v1) [paper-pdf](http://arxiv.org/pdf/2501.07251v1)

**Authors**: Ping Guo, Cheng Gong, Xi Lin, Fei Liu, Zhichao Lu, Qingfu Zhang, Zhenkun Wang

**Abstract**: Crafting adversarial examples is crucial for evaluating and enhancing the robustness of Deep Neural Networks (DNNs), presenting a challenge equivalent to maximizing a non-differentiable 0-1 loss function.   However, existing single objective methods, namely adversarial attacks focus on a surrogate loss function, do not fully harness the benefits of engaging multiple loss functions, as a result of insufficient understanding of their synergistic and conflicting nature.   To overcome these limitations, we propose the Multi-Objective Set-based Attack (MOS Attack), a novel adversarial attack framework leveraging multiple loss functions and automatically uncovering their interrelations.   The MOS Attack adopts a set-based multi-objective optimization strategy, enabling the incorporation of numerous loss functions without additional parameters.   It also automatically mines synergistic patterns among various losses, facilitating the generation of potent adversarial attacks with fewer objectives.   Extensive experiments have shown that our MOS Attack outperforms single-objective attacks. Furthermore, by harnessing the identified synergistic patterns, MOS Attack continues to show superior results with a reduced number of loss functions.

摘要: 为了评估和提高深度神经网络(DNN)的健壮性，构造敌意例子是至关重要的，这相当于最大化一个不可微的0-1损失函数。然而，现有的单目标方法，即对抗性攻击，侧重于代理损失函数，由于对它们的协同和冲突性质认识不足，没有充分利用使用多个损失函数的好处。为了克服这些局限性，我们提出了基于多目标集的攻击(MOS攻击)，这是一种利用多个损失函数并自动揭示它们之间相互关系的新的对抗性攻击框架。MOS攻击采用基于集合的多目标优化策略，能够在不增加参数的情况下合并众多损失函数。它还自动挖掘各种损失之间的协同模式，便于生成目标更少的强大对抗性攻击。广泛的实验表明，我们的MOS攻击的性能优于单目标攻击。此外，通过利用已确定的协同模式，MOS攻击继续显示出更好的结果，损失函数数量减少。



## **31. An Enhanced Zeroth-Order Stochastic Frank-Wolfe Framework for Constrained Finite-Sum Optimization**

约束概率和优化的增强零阶随机Frank-Wolfe框架 cs.LG

35 pages, 4 figures, 3 tables

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2501.07201v1) [paper-pdf](http://arxiv.org/pdf/2501.07201v1)

**Authors**: Haishan Ye, Yinghui Huang, Hao Di, Xiangyu Chang

**Abstract**: We propose an enhanced zeroth-order stochastic Frank-Wolfe framework to address constrained finite-sum optimization problems, a structure prevalent in large-scale machine-learning applications. Our method introduces a novel double variance reduction framework that effectively reduces the gradient approximation variance induced by zeroth-order oracles and the stochastic sampling variance from finite-sum objectives. By leveraging this framework, our algorithm achieves significant improvements in query efficiency, making it particularly well-suited for high-dimensional optimization tasks. Specifically, for convex objectives, the algorithm achieves a query complexity of O(d \sqrt{n}/\epsilon ) to find an epsilon-suboptimal solution, where d is the dimensionality and n is the number of functions in the finite-sum objective. For non-convex objectives, it achieves a query complexity of O(d^{3/2}\sqrt{n}/\epsilon^2 ) without requiring the computation ofd partial derivatives at each iteration. These complexities are the best known among zeroth-order stochastic Frank-Wolfe algorithms that avoid explicit gradient calculations. Empirical experiments on convex and non-convex machine learning tasks, including sparse logistic regression, robust classification, and adversarial attacks on deep networks, validate the computational efficiency and scalability of our approach. Our algorithm demonstrates superior performance in both convergence rate and query complexity compared to existing methods.

摘要: 我们提出了一种改进的零阶随机Frank-Wolfe框架来解决约束有限和优化问题，这是一种在大规模机器学习应用中普遍存在的结构。我们的方法引入了一种新的双方差减少框架，它有效地减少了由零阶预言引起的梯度逼近方差和来自有限和目标的随机抽样方差。通过利用这个框架，我们的算法在查询效率上取得了显著的改进，使其特别适合于高维优化任务。具体地说，对于凸目标，该算法的查询复杂度为O(d\sqrt{n}/\epsilon)，以找到epsilon次优解，其中d是维度，n是有限和目标中的函数数。对于非凸目标，其查询复杂度为O(d^{3/2}\Sqrt{n}/\epsilon^2)，而不需要在每次迭代时计算d个偏导数。这些复杂性是避免显式梯度计算的零阶随机Frank-Wolfe算法中最著名的。在凸和非凸机器学习任务上的实验，包括稀疏Logistic回归、稳健分类和对深层网络的敌意攻击，验证了该方法的计算效率和可扩展性。与已有方法相比，我们的算法在收敛速度和查询复杂度方面都表现出了更好的性能。



## **32. Bitcoin Under Volatile Block Rewards: How Mempool Statistics Can Influence Bitcoin Mining**

波动性区块奖励下的比特币：Mempool统计数据如何影响比特币采矿 cs.CR

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2411.11702v2) [paper-pdf](http://arxiv.org/pdf/2411.11702v2)

**Authors**: Roozbeh Sarenche, Alireza Aghabagherloo, Svetla Nikova, Bart Preneel

**Abstract**: The security of Bitcoin protocols is deeply dependent on the incentives provided to miners, which come from a combination of block rewards and transaction fees. As Bitcoin experiences more halving events, the protocol reward converges to zero, making transaction fees the primary source of miner rewards. This shift in Bitcoin's incentivization mechanism, which introduces volatility into block rewards, leads to the emergence of new security threats or intensifies existing ones. Previous security analyses of Bitcoin have either considered a fixed block reward model or a highly simplified volatile model, overlooking the complexities of Bitcoin's mempool behavior.   This paper presents a reinforcement learning-based tool to develop mining strategies under a more realistic volatile model. We employ the Asynchronous Advantage Actor-Critic (A3C) algorithm, which efficiently handles dynamic environments, such as the Bitcoin mempool, to derive near-optimal mining strategies when interacting with an environment that models the complexity of the Bitcoin mempool. This tool enables the analysis of adversarial mining strategies, such as selfish mining and undercutting, both before and after difficulty adjustments, providing insights into the effects of mining attacks in both the short and long term.   We revisit the Bitcoin security threshold presented in the WeRLman paper and demonstrate that the implicit predictability of valuable transaction arrivals in this model leads to an underestimation of the reported threshold. Additionally, we show that, while adversarial strategies like selfish mining under the fixed reward model incur an initial loss period of at least two weeks, the transition toward a transaction-fee era incentivizes mining pools to abandon honest mining for immediate profits. This incentive is expected to become more significant as the protocol reward approaches zero in the future.

摘要: 比特币协议的安全性在很大程度上依赖于向矿工提供的激励，这些激励来自大宗奖励和交易手续费的组合。随着比特币经历更多减半事件，协议奖励趋于零，使交易费成为矿工奖励的主要来源。比特币激励机制的这种转变，在大宗奖励中引入了波动性，导致了新的安全威胁的出现或加剧了现有的安全威胁。此前对比特币的安全分析要么考虑了固定的区块奖励模型，要么考虑了高度简化的波动性模型，忽视了比特币成员池行为的复杂性。本文提出了一种基于强化学习的工具，用于在更真实的易变模型下开发挖掘策略。我们使用异步优势参与者-批评者(A3C)算法，该算法有效地处理动态环境，例如比特币记忆池，在与模拟比特币记忆池复杂性的环境交互时，得出接近最优的挖掘策略。这一工具能够分析难度调整前后的对抗性采矿战略，如自私采矿和削价，从而深入了解采矿攻击在短期和长期的影响。我们回顾了WeRLman论文中提出的比特币安全阈值，并证明了该模型中有价值的交易到达的隐式可预测性导致了对报告阈值的低估。此外，我们还表明，虽然像固定报酬模式下的自私开采这样的对抗性策略会导致至少两周的初始损失期，但向交易费时代的过渡会激励矿池为了直接利润而放弃诚实的开采。随着协议奖励在未来接近于零，这一激励预计将变得更加重要。



## **33. Protego: Detecting Adversarial Examples for Vision Transformers via Intrinsic Capabilities**

Protego：通过固有能力检测视觉变形者的对抗示例 cs.CV

Accepted by IEEE MetaCom 2024

**SubmitDate**: 2025-01-13    [abs](http://arxiv.org/abs/2501.07044v1) [paper-pdf](http://arxiv.org/pdf/2501.07044v1)

**Authors**: Jialin Wu, Kaikai Pan, Yanjiao Chen, Jiangyi Deng, Shengyuan Pang, Wenyuan Xu

**Abstract**: Transformer models have excelled in natural language tasks, prompting the vision community to explore their implementation in computer vision problems. However, these models are still influenced by adversarial examples. In this paper, we investigate the attack capabilities of six common adversarial attacks on three pretrained ViT models to reveal the vulnerability of ViT models. To understand and analyse the bias in neural network decisions when the input is adversarial, we use two visualisation techniques that are attention rollout and grad attention rollout. To prevent ViT models from adversarial attack, we propose Protego, a detection framework that leverages the transformer intrinsic capabilities to detection adversarial examples of ViT models. Nonetheless, this is challenging due to a diversity of attack strategies that may be adopted by adversaries. Inspired by the attention mechanism, we know that the token of prediction contains all the information from the input sample. Additionally, the attention region for adversarial examples differs from that of normal examples. Given these points, we can train a detector that achieves superior performance than existing detection methods to identify adversarial examples. Our experiments have demonstrated the high effectiveness of our detection method. For these six adversarial attack methods, our detector's AUC scores all exceed 0.95. Protego may advance investigations in metaverse security.

摘要: 转换器模型在自然语言任务中表现出色，促使视觉社区探索其在计算机视觉问题中的实现。然而，这些模型仍然受到对抗性例子的影响。本文研究了六种常见的对抗性攻击对三种预先训练的VIT模型的攻击能力，以揭示VIT模型的脆弱性。为了理解和分析当输入是对抗性输入时神经网络决策中的偏差，我们使用了两种可视化技术，即注意力滚动和梯度注意力滚动。为了防止VIT模型受到恶意攻击，我们提出了Protego检测框架，该框架利用转换器的固有能力来检测VIT模型的恶意示例。尽管如此，由于对手可能采用的攻击策略多种多样，这是具有挑战性的。在注意机制的启发下，我们知道预测的表征包含了来自输入样本的所有信息。此外，对抗性例子的注意区域不同于正常例子。考虑到这些点，我们可以训练一个性能优于现有检测方法的检测器来识别对抗性示例。我们的实验证明了该检测方法的高效性。对于这六种对抗性攻击方式，我们的检测器的AUC得分都超过了0.95。Protego可能会推进Metverse安全方面的研究。



## **34. Exploring Transferability of Multimodal Adversarial Samples for Vision-Language Pre-training Models with Contrastive Learning**

利用对比学习探索视觉语言预训练模型多模式对抗样本的可移植性 cs.MM

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2308.12636v4) [paper-pdf](http://arxiv.org/pdf/2308.12636v4)

**Authors**: Youze Wang, Wenbo Hu, Yinpeng Dong, Hanwang Zhang, Hang Su, Richang Hong

**Abstract**: The integration of visual and textual data in Vision-Language Pre-training (VLP) models is crucial for enhancing vision-language understanding. However, the adversarial robustness of these models, especially in the alignment of image-text features, has not yet been sufficiently explored. In this paper, we introduce a novel gradient-based multimodal adversarial attack method, underpinned by contrastive learning, to improve the transferability of multimodal adversarial samples in VLP models. This method concurrently generates adversarial texts and images within imperceptive perturbation, employing both image-text and intra-modal contrastive loss. We evaluate the effectiveness of our approach on image-text retrieval and visual entailment tasks, using publicly available datasets in a black-box setting. Extensive experiments indicate a significant advancement over existing single-modal transfer-based adversarial attack methods and current multimodal adversarial attack approaches.

摘要: 视觉语言预训练（VLP）模型中视觉和文本数据的集成对于增强视觉语言理解至关重要。然而，这些模型的对抗稳健性，尤其是在图像-文本特征的对齐方面，尚未得到充分的探索。本文引入了一种以对比学习为基础的新型基于梯度的多模式对抗攻击方法，以提高VLP模型中多模式对抗样本的可移植性。该方法在不可感知的扰动中同时生成对抗性文本和图像，同时采用图像-文本和模式内对比损失。我们在黑匣子环境中使用公开可用的数据集来评估我们的方法在图像文本检索和视觉蕴含任务方面的有效性。大量实验表明，与现有的基于单模式转移的对抗攻击方法和当前的多模式对抗攻击方法相比，有了重大进步。



## **35. Mitigating Low-Frequency Bias: Feature Recalibration and Frequency Attention Regularization for Adversarial Robustness**

缓解低频偏见：对抗稳健性的特征重新校准和频率注意力规则化 cs.CV

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2407.04016v2) [paper-pdf](http://arxiv.org/pdf/2407.04016v2)

**Authors**: Kejia Zhang, Juanjuan Weng, Yuanzheng Cai, Zhiming Luo, Shaozi Li

**Abstract**: Ensuring the robustness of deep neural networks against adversarial attacks remains a fundamental challenge in computer vision. While adversarial training (AT) has emerged as a promising defense strategy, our analysis reveals a critical limitation: AT-trained models exhibit a bias toward low-frequency features while neglecting high-frequency components. This bias is particularly concerning as each frequency component carries distinct and crucial information: low-frequency features encode fundamental structural patterns, while high-frequency features capture intricate details and textures. To address this limitation, we propose High-Frequency Feature Disentanglement and Recalibration (HFDR), a novel module that strategically separates and recalibrates frequency-specific features to capture latent semantic cues. We further introduce frequency attention regularization to harmonize feature extraction across the frequency spectrum and mitigate the inherent low-frequency bias of AT. Extensive experiments demonstrate our method's superior performance against white-box attacks and transfer attacks, while exhibiting strong generalization capabilities across diverse scenarios.

摘要: 确保深度神经网络对敌意攻击的稳健性仍然是计算机视觉中的一个基本挑战。虽然对抗训练(AT)已经成为一种很有前途的防御策略，但我们的分析揭示了一个关键的局限性：AT训练的模型偏向于低频特征，而忽略了高频成分。这种偏差尤其令人担忧，因为每个频率分量都携带着不同的关键信息：低频特征编码基本的结构模式，而高频特征捕获复杂的细节和纹理。为了解决这一局限性，我们提出了高频特征解缠和重新校准(HFDR)，这是一个新的模块，从战略上分离和重新校准特定频率的特征，以捕获潜在的语义线索。我们进一步引入了频率注意正则化来协调整个频谱上的特征提取，并缓解了AT固有的低频偏差。大量实验表明，该方法对白盒攻击和传输攻击具有较好的抗攻击性能，同时在不同场景下表现出较强的泛化能力。



## **36. ModelShield: Adaptive and Robust Watermark against Model Extraction Attack**

Model Shield：针对模型提取攻击的自适应鲁棒水印 cs.CR

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2405.02365v4) [paper-pdf](http://arxiv.org/pdf/2405.02365v4)

**Authors**: Kaiyi Pang, Tao Qi, Chuhan Wu, Minhao Bai, Minghu Jiang, Yongfeng Huang

**Abstract**: Large language models (LLMs) demonstrate general intelligence across a variety of machine learning tasks, thereby enhancing the commercial value of their intellectual property (IP). To protect this IP, model owners typically allow user access only in a black-box manner, however, adversaries can still utilize model extraction attacks to steal the model intelligence encoded in model generation. Watermarking technology offers a promising solution for defending against such attacks by embedding unique identifiers into the model-generated content. However, existing watermarking methods often compromise the quality of generated content due to heuristic alterations and lack robust mechanisms to counteract adversarial strategies, thus limiting their practicality in real-world scenarios. In this paper, we introduce an adaptive and robust watermarking method (named ModelShield) to protect the IP of LLMs. Our method incorporates a self-watermarking mechanism that allows LLMs to autonomously insert watermarks into their generated content to avoid the degradation of model content. We also propose a robust watermark detection mechanism capable of effectively identifying watermark signals under the interference of varying adversarial strategies. Besides, ModelShield is a plug-and-play method that does not require additional model training, enhancing its applicability in LLM deployments. Extensive evaluations on two real-world datasets and three LLMs demonstrate that our method surpasses existing methods in terms of defense effectiveness and robustness while significantly reducing the degradation of watermarking on the model-generated content.

摘要: 大型语言模型(LLM)在各种机器学习任务中展示了一般智能，从而提高了其知识产权(IP)的商业价值。为了保护这个IP，模型所有者通常只允许用户以黑盒方式访问，但是，攻击者仍然可以利用模型提取攻击来窃取模型生成中编码的模型情报。水印技术通过在模型生成的内容中嵌入唯一标识符，为防御此类攻击提供了一种很有前途的解决方案。然而，现有的水印方法往往会由于启发式修改而影响生成内容的质量，并且缺乏强大的机制来对抗对抗性策略，从而限制了它们在现实世界场景中的实用性。本文提出了一种自适应的稳健水印算法(ModelShield)来保护LLMS的IP地址。我们的方法结合了一种自水印机制，允许LLM自主地在其生成的内容中插入水印，以避免模型内容的降级。我们还提出了一种稳健的水印检测机制，能够在不同的对抗策略的干扰下有效地识别水印信号。此外，ModelShield是一种即插即用的方法，不需要额外的模型培训，增强了其在LLM部署中的适用性。在两个真实数据集和三个LLM上的广泛评估表明，我们的方法在防御有效性和稳健性方面优于现有方法，同时显着降低了水印对模型生成内容的退化。



## **37. ZOQO: Zero-Order Quantized Optimization**

ZOQO：零阶量化优化 cs.LG

Accepted to ICASSP 2025

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2501.06736v1) [paper-pdf](http://arxiv.org/pdf/2501.06736v1)

**Authors**: Noga Bar, Raja Giryes

**Abstract**: The increasing computational and memory demands in deep learning present significant challenges, especially in resource-constrained environments. We introduce a zero-order quantized optimization (ZOQO) method designed for training models with quantized parameters and operations. Our approach leverages zero-order approximations of the gradient sign and adapts the learning process to maintain the parameters' quantization without the need for full-precision gradient calculations. We demonstrate the effectiveness of ZOQO through experiments in fine-tuning of large language models and black-box adversarial attacks. Despite the limitations of zero-order and quantized operations training, our method achieves competitive performance compared to full-precision methods, highlighting its potential for low-resource environments.

摘要: 深度学习中不断增加的计算和内存需求带来了重大挑战，尤其是在资源有限的环境中。我们引入了一种零阶量化优化（ZOQO）方法，专为具有量化参数和操作的训练模型而设计。我们的方法利用了梯度符号的零阶逼近，并调整学习过程以维持参数的量化，而无需全精度梯度计算。我们通过微调大型语言模型和黑匣子对抗攻击的实验来证明ZOQO的有效性。尽管零阶和量化操作训练存在局限性，但与全精度方法相比，我们的方法实现了有竞争力的性能，凸显了其在低资源环境中的潜力。



## **38. Measuring the Robustness of Reference-Free Dialogue Evaluation Systems**

衡量无参考对话评估系统的稳健性 cs.CL

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2501.06728v1) [paper-pdf](http://arxiv.org/pdf/2501.06728v1)

**Authors**: Justin Vasselli, Adam Nohejl, Taro Watanabe

**Abstract**: Advancements in dialogue systems powered by large language models (LLMs) have outpaced the development of reliable evaluation metrics, particularly for diverse and creative responses. We present a benchmark for evaluating the robustness of reference-free dialogue metrics against four categories of adversarial attacks: speaker tag prefixes, static responses, ungrammatical responses, and repeated conversational context. We analyze metrics such as DialogRPT, UniEval, and PromptEval -- a prompt-based method leveraging LLMs -- across grounded and ungrounded datasets. By examining both their correlation with human judgment and susceptibility to adversarial attacks, we find that these two axes are not always aligned; metrics that appear to be equivalent when judged by traditional benchmarks may, in fact, vary in their scores of adversarial responses. These findings motivate the development of nuanced evaluation frameworks to address real-world dialogue challenges.

摘要: 由大型语言模型（LLM）驱动的对话系统的进步已经超过了可靠评估指标的发展，特别是对于多样化和创造性的响应。我们提出了一个基准，用于评估无引用对话指标针对四类对抗攻击的稳健性：说话者标签前置、静态响应、不合语法的响应和重复对话上下文。我们跨基础和未基础数据集分析DialogRPT、UniEval和EntEval（一种利用LLM的基于预算的方法）等指标。通过检查它们与人类判断的相关性和对对抗攻击的易感性，我们发现这两个轴并不总是一致的;用传统基准判断时看似等效的指标实际上可能会有所不同。对抗反应的分数。这些发现促使制定细致入微的评估框架来应对现实世界的对话挑战。



## **39. Towards Adversarially Robust Deep Metric Learning**

迈向对抗稳健的深度度量学习 cs.LG

**SubmitDate**: 2025-01-12    [abs](http://arxiv.org/abs/2501.01025v2) [paper-pdf](http://arxiv.org/pdf/2501.01025v2)

**Authors**: Xiaopeng Ke

**Abstract**: Deep Metric Learning (DML) has shown remarkable successes in many domains by taking advantage of powerful deep neural networks. Deep neural networks are prone to adversarial attacks and could be easily fooled by adversarial examples. The current progress on this robustness issue is mainly about deep classification models but pays little attention to DML models. Existing works fail to thoroughly inspect the robustness of DML and neglect an important DML scenario, the clustering-based inference. In this work, we first point out the robustness issue of DML models in clustering-based inference scenarios. We find that, for the clustering-based inference, existing defenses designed DML are unable to be reused and the adaptions of defenses designed for deep classification models cannot achieve satisfactory robustness performance. To alleviate the hazard of adversarial examples, we propose a new defense, the Ensemble Adversarial Training (EAT), which exploits ensemble learning and adversarial training. EAT promotes the diversity of the ensemble, encouraging each model in the ensemble to have different robustness features, and employs a self-transferring mechanism to make full use of the robustness statistics of the whole ensemble in the update of every single model. We evaluate the EAT method on three widely-used datasets with two popular model architectures. The results show that the proposed EAT method greatly outperforms the adaptions of defenses designed for deep classification models.

摘要: 深度度量学习(DML)利用了强大的深度神经网络，在许多领域都取得了显著的成功。深度神经网络容易受到对抗性攻击，很容易被对抗性例子愚弄。目前在这一稳健性问题上的研究进展主要集中在深度分类模型上，对DML模型的研究较少。现有的工作没有对DML的健壮性进行彻底的检验，并且忽略了DML的一个重要场景--基于聚类的推理。在这项工作中，我们首先指出了DML模型在基于聚类的推理场景中的健壮性问题。我们发现，对于基于聚类的推理，现有的防御设计的DML不能被重用，并且针对深度分类模型的防御的适应性不能达到令人满意的健壮性。为了减少对抗性例子的危害，我们提出了一种新的防御方法--集成对抗性训练(EAT)，它利用了集成学习和对抗性训练。EAT促进了系综的多样性，鼓励系综中的每个模型具有不同的稳健性特征，并采用自迁移机制在每个单个模型的更新中充分利用整个系综的稳健性统计。我们在三个广泛使用的数据集和两个流行的模型体系结构上对EAT方法进行了评估。结果表明，提出的EAT方法的性能大大优于针对深度分类模型设计的防御方法。



## **40. Slot: Provenance-Driven APT Detection through Graph Reinforcement Learning**

插槽：通过图强化学习进行源驱动APT检测 cs.CR

**SubmitDate**: 2025-01-11    [abs](http://arxiv.org/abs/2410.17910v2) [paper-pdf](http://arxiv.org/pdf/2410.17910v2)

**Authors**: Wei Qiao, Yebo Feng, Teng Li, Zhuo Ma, Yulong Shen, JianFeng Ma, Yang Liu

**Abstract**: Advanced Persistent Threats (APTs) represent sophisticated cyberattacks characterized by their ability to remain undetected within the victim system for extended periods, aiming to exfiltrate sensitive data or disrupt operations. Existing detection approaches often struggle to effectively identify these complex threats, construct the attack chain for defense facilitation, or resist adversarial attacks. To overcome these challenges, we propose Slot, an advanced APT detection approach based on provenance graphs and graph reinforcement learning. Slot excels in uncovering multi-level hidden relationships, such as causal, contextual, and indirect connections, among system behaviors through provenance graph mining. By pioneering the integration of graph reinforcement learning, Slot dynamically adapts to new user activities and evolving attack strategies, enhancing its resilience against adversarial attacks. Additionally, Slot automatically constructs the attack chain according to detected attacks with clustering algorithms, providing precise identification of attack paths and facilitating the development of defense strategies. Evaluations with real-world datasets demonstrate Slot's outstanding accuracy, efficiency, adaptability, and robustness in APT detection, with most metrics surpassing state-of-the-art methods. Additionally, case studies conducted to assess Slot's effectiveness in supporting APT defense further establish it as a practical and reliable tool for cybersecurity protection.

摘要: 高级持续性威胁(APT)是复杂的网络攻击，其特征是能够在受害者系统内长时间保持不被检测到，旨在渗漏敏感数据或中断操作。现有的检测方法往往难以有效地识别这些复杂的威胁，难以构建便于防御的攻击链，或者难以抵抗对抗性攻击。为了克服这些挑战，我们提出了一种基于起源图和图强化学习的高级APT检测方法SLOT。Slot擅长通过起源图挖掘发现系统行为之间的多层次隐藏关系，如因果关系、上下文关系和间接关系。通过开创图强化学习的集成，时隙动态适应新的用户活动和不断演变的攻击策略，增强了对对手攻击的弹性。此外，SLOT根据检测到的攻击利用分簇算法自动构建攻击链，提供准确的攻击路径识别，便于制定防御策略。对真实数据集的评估表明，在APT检测中，Slot具有出色的准确性、效率、适应性和稳健性，大多数指标都超过了最先进的方法。此外，为评估SLOT在支持APT防御方面的有效性而进行的案例研究进一步证明，它是一种实用和可靠的网络安全保护工具。



## **41. LayerMix: Enhanced Data Augmentation through Fractal Integration for Robust Deep Learning**

LayerMix：通过Fractal集成增强数据增强，以实现稳健的深度学习 cs.CV

**SubmitDate**: 2025-01-11    [abs](http://arxiv.org/abs/2501.04861v2) [paper-pdf](http://arxiv.org/pdf/2501.04861v2)

**Authors**: Hafiz Mughees Ahmad, Dario Morle, Afshin Rahimi

**Abstract**: Deep learning models have demonstrated remarkable performance across various computer vision tasks, yet their vulnerability to distribution shifts remains a critical challenge. Despite sophisticated neural network architectures, existing models often struggle to maintain consistent performance when confronted with Out-of-Distribution (OOD) samples, including natural corruptions, adversarial perturbations, and anomalous patterns. We introduce LayerMix, an innovative data augmentation approach that systematically enhances model robustness through structured fractal-based image synthesis. By meticulously integrating structural complexity into training datasets, our method generates semantically consistent synthetic samples that significantly improve neural network generalization capabilities. Unlike traditional augmentation techniques that rely on random transformations, LayerMix employs a structured mixing pipeline that preserves original image semantics while introducing controlled variability. Extensive experiments across multiple benchmark datasets, including CIFAR-10, CIFAR-100, ImageNet-200, and ImageNet-1K demonstrate LayerMixs superior performance in classification accuracy and substantially enhances critical Machine Learning (ML) safety metrics, including resilience to natural image corruptions, robustness against adversarial attacks, improved model calibration and enhanced prediction consistency. LayerMix represents a significant advancement toward developing more reliable and adaptable artificial intelligence systems by addressing the fundamental challenges of deep learning generalization. The code is available at https://github.com/ahmadmughees/layermix.

摘要: 深度学习模型在各种计算机视觉任务中表现出了显著的性能，但它们对分布变化的脆弱性仍然是一个关键的挑战。尽管有复杂的神经网络结构，但现有的模型在面对分布外(OOD)样本时往往难以保持一致的性能，这些样本包括自然损坏、对抗性扰动和异常模式。我们介绍了LayerMix，这是一种创新的数据增强方法，它通过基于结构化分形的图像合成来系统地增强模型的稳健性。通过将结构复杂性精心集成到训练数据集中，我们的方法生成语义一致的合成样本，显著提高了神经网络的泛化能力。与依赖随机变换的传统增强技术不同，LayerMix采用了结构化混合管道，在引入受控可变性的同时保留了原始图像的语义。在CIFAR-10、CIFAR-100、ImageNet-200和ImageNet-1K等多个基准数据集上的广泛实验表明，LayerMix在分类精度方面具有卓越的性能，并显著增强了关键机器学习(ML)安全指标，包括对自然图像损坏的弹性、对对手攻击的健壮性、改进的模型校准和增强的预测一致性。LayerMix通过解决深度学习泛化的根本挑战，代表着在开发更可靠和更具适应性的人工智能系统方面取得了重大进展。代码可在https://github.com/ahmadmughees/layermix.上获得



## **42. Effective Backdoor Mitigation in Vision-Language Models Depends on the Pre-training Objective**

视觉语言模型中有效的后门缓解取决于预训练目标 cs.LG

Accepted at TMLR (https://openreview.net/forum?id=Conma3qnaT)

**SubmitDate**: 2025-01-11    [abs](http://arxiv.org/abs/2311.14948v4) [paper-pdf](http://arxiv.org/pdf/2311.14948v4)

**Authors**: Sahil Verma, Gantavya Bhatt, Avi Schwarzschild, Soumye Singhal, Arnav Mohanty Das, Chirag Shah, John P Dickerson, Pin-Yu Chen, Jeff Bilmes

**Abstract**: Despite the advanced capabilities of contemporary machine learning (ML) models, they remain vulnerable to adversarial and backdoor attacks. This vulnerability is particularly concerning in real-world deployments, where compromised models may exhibit unpredictable behavior in critical scenarios. Such risks are heightened by the prevalent practice of collecting massive, internet-sourced datasets for training multimodal models, as these datasets may harbor backdoors. Various techniques have been proposed to mitigate the effects of backdooring in multimodal models, such as CleanCLIP, which is the current state-of-the-art approach. In this work, we demonstrate that the efficacy of CleanCLIP in mitigating backdoors is highly dependent on the particular objective used during model pre-training. We observe that stronger pre-training objectives that lead to higher zero-shot classification performance correlate with harder to remove backdoors behaviors. We show this by training multimodal models on two large datasets consisting of 3 million (CC3M) and 6 million (CC6M) datapoints, under various pre-training objectives, followed by poison removal using CleanCLIP. We find that CleanCLIP, even with extensive hyperparameter tuning, is ineffective in poison removal when stronger pre-training objectives are used. Our findings underscore critical considerations for ML practitioners who train models using large-scale web-curated data and are concerned about potential backdoor threats.

摘要: 尽管当代机器学习(ML)模型具有先进的能力，但它们仍然容易受到对手和后门攻击。此漏洞在实际部署中尤其令人担忧，在实际部署中，受危害的模型可能会在关键情况下表现出不可预测的行为。收集来自互联网的海量数据集以训练多模式模型的普遍做法加剧了这种风险，因为这些数据集可能有后门。已经提出了各种技术来减轻多模式模型中回溯的影响，例如CleanCLIP，这是当前最先进的方法。在这项工作中，我们证明了CleanCLIP在缓解后门方面的有效性高度依赖于在模型预培训期间使用的特定目标。我们观察到，更强的预训练目标导致更高的零射击分类性能，与更难移除后门行为相关。我们通过在两个由300万(CC3M)和600万(CC6M)数据点组成的大型数据集上训练多模模型，在不同的预训练目标下，然后使用CleanCLIP去除毒物来证明这一点。我们发现，当使用更强的预训练目标时，即使进行了广泛的超参数调整，CleanCLIP也不能有效地去除毒物。我们的发现强调了ML从业者的关键考虑，他们使用大规模的网络管理数据训练模型，并担心潜在的后门威胁。



## **43. Privacy-Preserving Distributed Defense Framework for DC Microgrids Against Exponentially Unbounded False Data Injection Attacks**

针对指数无界虚假数据注入攻击的DC微电网保护隐私分布式防御框架 eess.SY

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.00588v2) [paper-pdf](http://arxiv.org/pdf/2501.00588v2)

**Authors**: Yi Zhang, Mohamadamin Rajabinezhad, Yichao Wang, Junbo Zhao, Shan Zuo

**Abstract**: This paper introduces a novel, fully distributed control framework for DC microgrids, enhancing resilience against exponentially unbounded false data injection (EU-FDI) attacks. Our framework features a consensus-based secondary control for each converter, effectively addressing these advanced threats. To further safeguard sensitive operational data, a privacy-preserving mechanism is incorporated into the control design, ensuring that critical information remains secure even under adversarial conditions. Rigorous Lyapunov stability analysis confirms the framework's ability to maintain critical DC microgrid operations like voltage regulation and load sharing under EU-FDI threats. The framework's practicality is validated through hardware-in-the-loop experiments, demonstrating its enhanced resilience and robust privacy protection against the complex challenges posed by quick variant FDI attacks.

摘要: 本文介绍了一种新颖的、完全分布式的DC微电网控制框架，增强了抵御指数无界虚假数据注入（EU-Direct）攻击的弹性。我们的框架为每个转换器提供了基于共识的二级控制，可以有效地解决这些高级威胁。为了进一步保护敏感的运营数据，控制设计中纳入了隐私保护机制，确保关键信息即使在敌对条件下也保持安全。严格的李亚普诺夫稳定性分析证实了该框架在欧盟外国直接投资威胁下维持关键的直流微电网运营的能力，例如电压调节和负载共享。该框架的实用性通过硬件在环实验得到了验证，展示了其增强的弹性和强大的隐私保护，以应对快速变体外国直接投资攻击带来的复杂挑战。



## **44. Pixel Is Not A Barrier: An Effective Evasion Attack for Pixel-Domain Diffusion Models**

像素不是障碍：像素域扩散模型的有效规避攻击 cs.CV

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2408.11810v2) [paper-pdf](http://arxiv.org/pdf/2408.11810v2)

**Authors**: Chun-Yen Shih, Li-Xuan Peng, Jia-Wei Liao, Ernie Chu, Cheng-Fu Chou, Jun-Cheng Chen

**Abstract**: Diffusion Models have emerged as powerful generative models for high-quality image synthesis, with many subsequent image editing techniques based on them. However, the ease of text-based image editing introduces significant risks, such as malicious editing for scams or intellectual property infringement. Previous works have attempted to safeguard images from diffusion-based editing by adding imperceptible perturbations. These methods are costly and specifically target prevalent Latent Diffusion Models (LDMs), while Pixel-domain Diffusion Models (PDMs) remain largely unexplored and robust against such attacks. Our work addresses this gap by proposing a novel attack framework, AtkPDM. AtkPDM is mainly composed of a feature representation attacking loss that exploits vulnerabilities in denoising UNets and a latent optimization strategy to enhance the naturalness of adversarial images. Extensive experiments demonstrate the effectiveness of our approach in attacking dominant PDM-based editing methods (e.g., SDEdit) while maintaining reasonable fidelity and robustness against common defense methods. Additionally, our framework is extensible to LDMs, achieving comparable performance to existing approaches.

摘要: 扩散模型已经成为高质量图像合成的强大生成性模型，许多后续的图像编辑技术都是基于扩散模型的。然而，基于文本的图像编辑的简便性带来了重大风险，例如用于欺诈或侵犯知识产权的恶意编辑。以前的工作试图通过添加不可察觉的扰动来保护图像免受基于扩散的编辑。这些方法昂贵且专门针对流行的潜在扩散模型(LDM)，而像素域扩散模型(PDMS)在很大程度上仍未被探索，并且对此类攻击具有健壮性。我们的工作通过提出一种新颖的攻击框架AtkPDM来解决这一问题。该算法主要由特征表示、攻击损失和潜在优化策略两部分组成，前者利用UNNet的去噪漏洞，后者增强敌方图像的自然度。大量实验表明，该方法在攻击主流的基于产品数据管理的编辑方法(如SDEDIT)的同时，对常见的防御方法保持了合理的保真度和健壮性。此外，我们的框架可扩展到LDM，实现了与现有方法相当的性能。



## **45. Adversarial Detection by Approximation of Ensemble Boundary**

利用集合边界逼近的对抗检测 cs.LG

27 pages, 7 figures, 5 tables

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2211.10227v5) [paper-pdf](http://arxiv.org/pdf/2211.10227v5)

**Authors**: T. Windeatt

**Abstract**: Despite being effective in many application areas, Deep Neural Networks (DNNs) are vulnerable to being attacked. In object recognition, the attack takes the form of a small perturbation added to an image, that causes the DNN to misclassify, but to a human appears no different. Adversarial attacks lead to defences that are themselves subject to attack, and the attack/ defence strategies provide important information about the properties of DNNs. In this paper, a novel method of detecting adversarial attacks is proposed for an ensemble of Deep Neural Networks (DNNs) solving two-class pattern recognition problems. The ensemble is combined using Walsh coefficients which are capable of approximating Boolean functions and thereby controlling the decision boundary complexity. The hypothesis in this paper is that decision boundaries with high curvature allow adversarial perturbations to be found, but change the curvature of the decision boundary, which is then approximated in a different way by Walsh coefficients compared to the clean images. Besides controlling boundary complexity, the coefficients also measure the correlation with class labels, which may aid in understanding the learning and transferability properties of DNNs. While the experiments here use images, the proposed approach of modelling two-class ensemble decision boundaries could in principle be applied to any application area.

摘要: 尽管深度神经网络(DNN)在许多应用领域都很有效，但它很容易受到攻击。在目标识别中，攻击的形式是添加到图像上的小扰动，这会导致DNN错误分类，但对人类来说似乎没有什么不同。对抗性攻击导致的防御本身也会受到攻击，而攻击/防御策略提供了有关DNN属性的重要信息。针对解决两类模式识别问题的深度神经网络(DNN)集成问题，提出了一种检测敌意攻击的新方法。该集成使用沃尔什系数进行组合，沃尔什系数能够逼近布尔函数，从而控制决策边界的复杂性。本文的假设是，高曲率的决策边界允许发现对抗性扰动，但改变了决策边界的曲率，然后用沃尔什系数以不同的方式逼近决策边界，与干净的图像相比。除了控制边界复杂度外，该系数还度量了与类别标签的相关性，这有助于理解DNN的学习和迁移特性。虽然这里的实验使用的是图像，但所提出的建模两类集合决策边界的方法原则上可以应用于任何应用领域。



## **46. Beyond Optimal Fault Tolerance**

超越最佳故障容忍度 cs.DC

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.06044v1) [paper-pdf](http://arxiv.org/pdf/2501.06044v1)

**Authors**: Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The optimal fault-tolerance achievable by any protocol has been characterized in a wide range of settings. For example, for state machine replication (SMR) protocols operating in the partially synchronous setting, it is possible to simultaneously guarantee consistency against $\alpha$-bounded adversaries (i.e., adversaries that control less than an $\alpha$ fraction of the participants) and liveness against $\beta$-bounded adversaries if and only if $\alpha + 2\beta \leq 1$.   This paper characterizes to what extent "better-than-optimal" fault-tolerance guarantees are possible for SMR protocols when the standard consistency requirement is relaxed to allow a bounded number $r$ of consistency violations. We prove that bounding rollback is impossible without additional timing assumptions and investigate protocols that tolerate and recover from consistency violations whenever message delays around the time of an attack are bounded by a parameter $\Delta^*$ (which may be arbitrarily larger than the parameter $\Delta$ that bounds post-GST message delays in the partially synchronous model). Here, a protocol's fault-tolerance can be a non-constant function of $r$, and we prove, for each $r$, matching upper and lower bounds on the optimal ``recoverable fault-tolerance'' achievable by any SMR protocol. For example, for protocols that guarantee liveness against 1/3-bounded adversaries in the partially synchronous setting, a 5/9-bounded adversary can always cause one consistency violation but not two, and a 2/3-bounded adversary can always cause two consistency violations but not three. Our positive results are achieved through a generic ``recovery procedure'' that can be grafted on to any accountable SMR protocol and restores consistency following a violation while rolling back only transactions that were finalized in the previous $2\Delta^*$ timesteps.

摘要: 任何协议可实现的最佳容错已在广泛的设置中得到了表征。例如，对于在部分同步设置中操作的状态机复制(SMR)协议，当且仅当$\Alpha+2\Beta\leq 1$时，可以同时保证针对$\Alpha$受限的对手(即，控制少于$\Alpha$部分参与者的对手)的一致性和针对$\beta$受限的攻击者的活性。本文刻画了当标准一致性要求被放宽以允许有限数量的一致性违规时，SMR协议在多大程度上可能获得比最优更好的容错保证。我们证明了如果没有额外的时间假设，绑定回滚是不可能的，并研究了当攻击时间附近的消息延迟由参数$\Delta^*$(该参数可以任意大于在部分同步模型中限制GST后消息延迟的参数$\Delta$)限定时，容忍一致性违规并从一致性违规中恢复的协议。这里，协议的容错性可以是$r$的非常数函数，并且我们证明了，对于每个$r$，任何SMR协议都可以达到最优“可恢复容错性”的上下界匹配。例如，对于在部分同步设置中保证对1/3有界攻击者的活跃性的协议，5/9有界的攻击者总是可以导致一次一致性违反而不是两次，而2/3有界的攻击者总是可以引起两次一致性违反而不是三次一致性违反。我们的积极结果是通过可嫁接到任何负责任的SMR协议上并在违规后恢复一致性，同时仅回滚在前$2\Delta^*$时间步长中完成的事务的通用“恢复程序”实现的。



## **47. Effective faking of verbal deception detection with target-aligned adversarial attacks**

通过目标对准的对抗攻击有效伪造言语欺骗检测 cs.CL

preprint

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05962v1) [paper-pdf](http://arxiv.org/pdf/2501.05962v1)

**Authors**: Bennett Kleinberg, Riccardo Loconte, Bruno Verschuere

**Abstract**: Background: Deception detection through analysing language is a promising avenue using both human judgments and automated machine learning judgments. For both forms of credibility assessment, automated adversarial attacks that rewrite deceptive statements to appear truthful pose a serious threat. Methods: We used a dataset of 243 truthful and 262 fabricated autobiographical stories in a deception detection task for humans and machine learning models. A large language model was tasked to rewrite deceptive statements so that they appear truthful. In Study 1, humans who made a deception judgment or used the detailedness heuristic and two machine learning models (a fine-tuned language model and a simple n-gram model) judged original or adversarial modifications of deceptive statements. In Study 2, we manipulated the target alignment of the modifications, i.e. tailoring the attack to whether the statements would be assessed by humans or computer models. Results: When adversarial modifications were aligned with their target, human (d=-0.07 and d=-0.04) and machine judgments (51% accuracy) dropped to the chance level. When the attack was not aligned with the target, both human heuristics judgments (d=0.30 and d=0.36) and machine learning predictions (63-78%) were significantly better than chance. Conclusions: Easily accessible language models can effectively help anyone fake deception detection efforts both by humans and machine learning models. Robustness against adversarial modifications for humans and machines depends on that target alignment. We close with suggestions on advancing deception research with adversarial attack designs.

摘要: 背景：通过分析语言进行欺骗检测是一种既使用人类判断又使用自动机器学习判断的有前途的方法。对于这两种形式的可信度评估来说，重写欺骗性陈述以使其看起来真实的自动对抗性攻击构成了严重威胁。方法：我们使用了243个真实的和262个编造的自传故事的数据集，在人类和机器学习模型的欺骗检测任务中。一个大型语言模型的任务是重写欺骗性的陈述，使它们看起来是真实的。在研究1中，做出欺骗性判断或使用细节启发式和两个机器学习模型(微调语言模型和简单n元语法模型)的人判断欺骗性陈述的原始修改或对抗性修改。在研究2中，我们操纵了修改的目标对齐，即根据陈述是否由人或计算机模型评估来量身定做攻击。结果：当对抗性修改与他们的目标一致时，人类(d=-0.07和d=-0.04)和机器判断(51%准确率)下降到机会水平。当攻击与目标不一致时，人类的启发式判断(d=0.30和d=0.36)和机器学习预测(63%-78%)都显著好于机会。结论：易于理解的语言模型可以有效地帮助任何人通过人类和机器学习模型进行虚假的欺骗检测。对人类和机器的敌意修改的健壮性取决于目标对齐。最后，我们建议用对抗性攻击设计来推进欺骗研究。



## **48. Towards Backdoor Stealthiness in Model Parameter Space**

模型参数空间中的后门隐秘性 cs.CR

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.05928v1) [paper-pdf](http://arxiv.org/pdf/2501.05928v1)

**Authors**: Xiaoyun Xu, Zhuoran Liu, Stefanos Koffas, Stjepan Picek

**Abstract**: Recent research on backdoor stealthiness focuses mainly on indistinguishable triggers in input space and inseparable backdoor representations in feature space, aiming to circumvent backdoor defenses that examine these respective spaces. However, existing backdoor attacks are typically designed to resist a specific type of backdoor defense without considering the diverse range of defense mechanisms. Based on this observation, we pose a natural question: Are current backdoor attacks truly a real-world threat when facing diverse practical defenses?   To answer this question, we examine 12 common backdoor attacks that focus on input-space or feature-space stealthiness and 17 diverse representative defenses. Surprisingly, we reveal a critical blind spot: Backdoor attacks designed to be stealthy in input and feature spaces can be mitigated by examining backdoored models in parameter space. To investigate the underlying causes behind this common vulnerability, we study the characteristics of backdoor attacks in the parameter space. Notably, we find that input- and feature-space attacks introduce prominent backdoor-related neurons in parameter space, which are not thoroughly considered by current backdoor attacks. Taking comprehensive stealthiness into account, we propose a novel supply-chain attack called Grond. Grond limits the parameter changes by a simple yet effective module, Adversarial Backdoor Injection (ABI), which adaptively increases the parameter-space stealthiness during the backdoor injection. Extensive experiments demonstrate that Grond outperforms all 12 backdoor attacks against state-of-the-art (including adaptive) defenses on CIFAR-10, GTSRB, and a subset of ImageNet. In addition, we show that ABI consistently improves the effectiveness of common backdoor attacks.

摘要: 目前关于后门隐蔽性的研究主要集中在输入空间中不可区分的触发器和特征空间中不可分的后门表示上，目的是绕过检查这些空间的后门防御。然而，现有的后门攻击通常被设计为抵抗特定类型的后门防御，而没有考虑到各种防御机制。基于这一观察，我们提出了一个自然的问题：当面临各种实际防御时，当前的后门攻击真的是现实世界的威胁吗？为了回答这个问题，我们研究了12种常见的后门攻击，这些攻击侧重于输入空间或功能空间的隐蔽性，以及17种不同的代表性防御。令人惊讶的是，我们揭示了一个关键的盲点：设计为在输入和特征空间中隐蔽的后门攻击可以通过检查参数空间中的后置模型来缓解。为了研究这种常见漏洞背后的潜在原因，我们研究了参数空间中的后门攻击的特征。值得注意的是，我们发现输入和特征空间攻击在参数空间中引入了显著的后门相关神经元，而目前的后门攻击并没有完全考虑到这一点。在综合考虑隐蔽性的基础上，提出了一种新的供应链攻击方法Grond。Grond通过一个简单而有效的模块--对抗性后门注入(ABI)来限制参数变化，该模块在后门注入过程中自适应地增加参数空间的隐蔽性。广泛的实验表明，Grond在CIFAR-10、GTSRB和ImageNet的子集上对最先进的(包括自适应)防御系统的攻击能力超过了所有12个后门攻击。此外，我们还证明了ABI一贯提高了常见后门攻击的有效性。



## **49. Backdoor Attacks against No-Reference Image Quality Assessment Models via a Scalable Trigger**

通过可扩展触发器对无参考图像质量评估模型进行后门攻击 cs.CV

Accept by AAAI 2025

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2412.07277v2) [paper-pdf](http://arxiv.org/pdf/2412.07277v2)

**Authors**: Yi Yu, Song Xia, Xun Lin, Wenhan Yang, Shijian Lu, Yap-peng Tan, Alex Kot

**Abstract**: No-Reference Image Quality Assessment (NR-IQA), responsible for assessing the quality of a single input image without using any reference, plays a critical role in evaluating and optimizing computer vision systems, e.g., low-light enhancement. Recent research indicates that NR-IQA models are susceptible to adversarial attacks, which can significantly alter predicted scores with visually imperceptible perturbations. Despite revealing vulnerabilities, these attack methods have limitations, including high computational demands, untargeted manipulation, limited practical utility in white-box scenarios, and reduced effectiveness in black-box scenarios. To address these challenges, we shift our focus to another significant threat and present a novel poisoning-based backdoor attack against NR-IQA (BAIQA), allowing the attacker to manipulate the IQA model's output to any desired target value by simply adjusting a scaling coefficient $\alpha$ for the trigger. We propose to inject the trigger in the discrete cosine transform (DCT) domain to improve the local invariance of the trigger for countering trigger diminishment in NR-IQA models due to widely adopted data augmentations. Furthermore, the universal adversarial perturbations (UAP) in the DCT space are designed as the trigger, to increase IQA model susceptibility to manipulation and improve attack effectiveness. In addition to the heuristic method for poison-label BAIQA (P-BAIQA), we explore the design of clean-label BAIQA (C-BAIQA), focusing on $\alpha$ sampling and image data refinement, driven by theoretical insights we reveal. Extensive experiments on diverse datasets and various NR-IQA models demonstrate the effectiveness of our attacks. Code can be found at https://github.com/yuyi-sd/BAIQA.

摘要: 无参考图像质量评估(NR-IQA)负责在不使用任何参考图像的情况下评估单个输入图像的质量，在评估和优化计算机视觉系统(如微光增强)中起着至关重要的作用。最近的研究表明，NR-IQA模型容易受到对抗性攻击，这种攻击会在视觉上不可察觉的扰动下显著改变预测分数。尽管暴露出漏洞，但这些攻击方法都有局限性，包括计算要求高、无针对性操作、在白盒场景中实际效用有限，以及在黑盒场景中有效性降低。为了应对这些挑战，我们将重点转移到另一个重要的威胁上，并提出了一种针对NR-IQA的基于中毒的后门攻击(BAIQA)，允许攻击者通过简单地调整触发器的缩放系数$\α$来操纵IQA模型的输出到任何期望的目标值。我们提出在离散余弦变换(DCT)域中注入触发器以改善触发器的局部不变性，以对抗由于广泛采用的数据增强而导致的NR-IQA模型中的触发器衰减。此外，设计了DCT空间中的通用对抗摄动(UAP)作为触发器，以增加IQA模型对操纵的敏感度，提高攻击效率。除了有毒标签BAIQA的启发式方法(P-BAIQA)外，我们还探索了清洁标签BAIQA(C-BAIQA)的设计，重点是$\α$采样和图像数据精化，这是我们揭示的理论见解的驱动。在不同的数据集和不同的NR-IQA模型上的大量实验证明了我们的攻击的有效性。代码可在https://github.com/yuyi-sd/BAIQA.上找到



## **50. Image-based Multimodal Models as Intruders: Transferable Multimodal Attacks on Video-based MLLMs**

作为入侵者的基于图像的多模式模型：对基于视频的MLLM的可转移多模式攻击 cs.CV

**SubmitDate**: 2025-01-10    [abs](http://arxiv.org/abs/2501.01042v2) [paper-pdf](http://arxiv.org/pdf/2501.01042v2)

**Authors**: Linhao Huang, Xue Jiang, Zhiqiang Wang, Wentao Mo, Xi Xiao, Bo Han, Yongjie Yin, Feng Zheng

**Abstract**: Video-based multimodal large language models (V-MLLMs) have shown vulnerability to adversarial examples in video-text multimodal tasks. However, the transferability of adversarial videos to unseen models--a common and practical real world scenario--remains unexplored. In this paper, we pioneer an investigation into the transferability of adversarial video samples across V-MLLMs. We find that existing adversarial attack methods face significant limitations when applied in black-box settings for V-MLLMs, which we attribute to the following shortcomings: (1) lacking generalization in perturbing video features, (2) focusing only on sparse key-frames, and (3) failing to integrate multimodal information. To address these limitations and deepen the understanding of V-MLLM vulnerabilities in black-box scenarios, we introduce the Image-to-Video MLLM (I2V-MLLM) attack. In I2V-MLLM, we utilize an image-based multimodal model (IMM) as a surrogate model to craft adversarial video samples. Multimodal interactions and temporal information are integrated to disrupt video representations within the latent space, improving adversarial transferability. In addition, a perturbation propagation technique is introduced to handle different unknown frame sampling strategies. Experimental results demonstrate that our method can generate adversarial examples that exhibit strong transferability across different V-MLLMs on multiple video-text multimodal tasks. Compared to white-box attacks on these models, our black-box attacks (using BLIP-2 as surrogate model) achieve competitive performance, with average attack success rates of 55.48% on MSVD-QA and 58.26% on MSRVTT-QA for VideoQA tasks, respectively. Our code will be released upon acceptance.

摘要: 基于视频的多通道大语言模型(V-MLLM)在视频-文本多通道任务中表现出对敌意例子的脆弱性。然而，对抗性视频是否可以转移到看不见的模型上--这是现实世界中常见和实用的场景--仍未得到探索。在本文中，我们率先对对抗性视频样本在V-MLLMS上的可转移性进行了研究。我们发现，现有的对抗性攻击方法在应用于V-MLLMS的黑盒环境时面临着很大的局限性，我们将其归因于以下缺点：(1)对扰动视频特征缺乏泛化；(2)只关注稀疏关键帧；(3)未能整合多模信息。为了解决这些限制并加深对黑盒场景中V-MLLM漏洞的理解，我们引入了图像到视频MLLM(I2V-MLLM)攻击。在I2V-MLLM中，我们使用基于图像的多模式模型(IMM)作为代理模型来制作对抗性视频样本。多模式交互和时间信息被集成以扰乱潜在空间内的视频表示，提高了对抗性转移。此外，还引入了扰动传播技术来处理不同的未知帧采样策略。实验结果表明，该方法能够在多个视频-文本多模式任务的不同V-MLLMS之间生成具有较强可转移性的对抗性实例。与这些模型上的白盒攻击相比，我们的黑盒攻击(以BLIP-2为代理模型)取得了与之相当的性能，对于视频QA任务，MSVD-QA和MSRVTT-QA的平均攻击成功率分别为55.48%和58.26%。我们的代码将在接受后发布。



