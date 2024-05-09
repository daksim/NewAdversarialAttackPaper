# Latest Adversarial Attack Papers
**update at 2024-05-09 23:52:50**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Air Gap: Protecting Privacy-Conscious Conversational Agents**

空气间隙：保护有隐私意识的对话代理人 cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05175v1) [paper-pdf](http://arxiv.org/pdf/2405.05175v1)

**Authors**: Eugene Bagdasaryan, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, Daniel Ramage

**Abstract**: The growing use of large language model (LLM)-based conversational agents to manage sensitive user data raises significant privacy concerns. While these agents excel at understanding and acting on context, this capability can be exploited by malicious actors. We introduce a novel threat model where adversarial third-party apps manipulate the context of interaction to trick LLM-based agents into revealing private information not relevant to the task at hand.   Grounded in the framework of contextual integrity, we introduce AirGapAgent, a privacy-conscious agent designed to prevent unintended data leakage by restricting the agent's access to only the data necessary for a specific task. Extensive experiments using Gemini, GPT, and Mistral models as agents validate our approach's effectiveness in mitigating this form of context hijacking while maintaining core agent functionality. For example, we show that a single-query context hijacking attack on a Gemini Ultra agent reduces its ability to protect user data from 94% to 45%, while an AirGapAgent achieves 97% protection, rendering the same attack ineffective.

摘要: 越来越多地使用基于大型语言模型(LLM)的会话代理来管理敏感用户数据，这引发了严重的隐私问题。虽然这些代理擅长理解上下文并根据上下文执行操作，但这种能力可能会被恶意行为者利用。我们引入了一种新的威胁模型，在该模型中，敌意的第三方应用程序操纵交互的上下文，以欺骗基于LLM的代理泄露与手头任务无关的私人信息。基于上下文完整性的框架，我们引入了AirGapAgent，这是一个具有隐私意识的代理，旨在通过限制代理仅访问特定任务所需的数据来防止意外的数据泄露。使用Gemini、GPT和Mistral模型作为代理的大量实验验证了我们的方法在保持核心代理功能的同时缓解这种形式的上下文劫持的有效性。例如，我们表明，对Gemini Ultra代理的单查询上下文劫持攻击将其保护用户数据的能力从94%降低到45%，而AirGapAgent实现了97%的保护，使得相同的攻击无效。



## **2. Filtering and smoothing estimation algorithms from uncertain nonlinear observations with time-correlated additive noise and random deception attacks**

来自具有时间相关添加性噪音和随机欺骗攻击的不确定非线性观测的过滤和平滑估计算法 eess.SP

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05157v1) [paper-pdf](http://arxiv.org/pdf/2405.05157v1)

**Authors**: R. Caballero-Águila, J. Hu, J. Linares-Pérez

**Abstract**: This paper discusses the problem of estimating a stochastic signal from nonlinear uncertain observations with time-correlated additive noise described by a first-order Markov process. Random deception attacks are assumed to be launched by an adversary, and both this phenomenon and the uncertainty in the observations are modelled by two sets of Bernoulli random variables. Under the assumption that the evolution model generating the signal to be estimated is unknown and only the mean and covariance functions of the processes involved in the observation equation are available, recursive algorithms based on linear approximations of the real observations are proposed for the least-squares filtering and fixed-point smoothing problems. Finally, the feasibility and effectiveness of the developed estimation algorithms are verified by a numerical simulation example, where the impact of uncertain observation and deception attack probabilities on estimation accuracy is evaluated.

摘要: 本文讨论了由一阶马尔科夫过程描述的具有时间相关添加性噪音的非线性不确定观测估计随机信号的问题。假设随机欺骗攻击是由对手发起的，这种现象和观察中的不确定性都是由两组伯努里随机变量建模的。在生成待估计信号的进化模型未知且只有观测方程中涉及的过程的均值和协方差函数可用的假设下，提出了基于真实观测值线性逼近的回归算法来解决最小平方过滤和定点平滑问题。最后，通过数值仿真算例验证了所开发的估计算法的可行性和有效性，评估了不确定观测和欺骗攻击概率对估计准确性的影响。



## **3. Towards Efficient Training and Evaluation of Robust Models against $l_0$ Bounded Adversarial Perturbations**

针对1_0美元有界对抗性扰动的稳健模型的有效训练和评估 cs.LG

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05075v1) [paper-pdf](http://arxiv.org/pdf/2405.05075v1)

**Authors**: Xuyang Zhong, Yixiao Huang, Chen Liu

**Abstract**: This work studies sparse adversarial perturbations bounded by $l_0$ norm. We propose a white-box PGD-like attack method named sparse-PGD to effectively and efficiently generate such perturbations. Furthermore, we combine sparse-PGD with a black-box attack to comprehensively and more reliably evaluate the models' robustness against $l_0$ bounded adversarial perturbations. Moreover, the efficiency of sparse-PGD enables us to conduct adversarial training to build robust models against sparse perturbations. Extensive experiments demonstrate that our proposed attack algorithm exhibits strong performance in different scenarios. More importantly, compared with other robust models, our adversarially trained model demonstrates state-of-the-art robustness against various sparse attacks. Codes are available at https://github.com/CityU-MLO/sPGD.

摘要: 这项工作研究了以$l_0$规范为界的稀疏对抗扰动。我们提出了一种名为sparse-PVD的白盒类PGD攻击方法，以有效且高效地生成此类扰动。此外，我们将稀疏PVD与黑匣子攻击相结合，以全面、更可靠地评估模型对1_0美元有界对抗扰动的鲁棒性。此外，稀疏PVD的效率使我们能够进行对抗训练，以构建针对稀疏扰动的稳健模型。大量实验表明，我们提出的攻击算法在不同场景下表现出强大的性能。更重要的是，与其他稳健模型相比，我们的对抗训练模型表现出了针对各种稀疏攻击的最新稳健性。代码可访问https://github.com/CityU-MLO/sPGD。



## **4. Adversarial Threats to Automatic Modulation Open Set Recognition in Wireless Networks**

无线网络中自动调制开集识别的对抗威胁 cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05022v1) [paper-pdf](http://arxiv.org/pdf/2405.05022v1)

**Authors**: Yandie Yang, Sicheng Zhang, Kuixian Li, Qiao Tian, Yun Lin

**Abstract**: Automatic Modulation Open Set Recognition (AMOSR) is a crucial technological approach for cognitive radio communications, wireless spectrum management, and interference monitoring within wireless networks. Numerous studies have shown that AMR is highly susceptible to minimal perturbations carefully designed by malicious attackers, leading to misclassification of signals. However, the adversarial security issue of AMOSR has not yet been explored. This paper adopts the perspective of attackers and proposes an Open Set Adversarial Attack (OSAttack), aiming at investigating the adversarial vulnerabilities of various AMOSR methods. Initially, an adversarial threat model for AMOSR scenarios is established. Subsequently, by analyzing the decision criteria of both discriminative and generative open set recognition, OSFGSM and OSPGD are proposed to reduce the performance of AMOSR. Finally, the influence of OSAttack on AMOSR is evaluated utilizing a range of qualitative and quantitative indicators. The results indicate that despite the increased resistance of AMOSR models to conventional interference signals, they remain vulnerable to attacks by adversarial examples.

摘要: 自动调制开集识别(AMOSR)是认知无线电通信、无线频谱管理和无线网络干扰监测的重要技术手段。大量研究表明，AMR非常容易受到恶意攻击者精心设计的微小扰动的影响，从而导致信号的错误分类。然而，AMOSR的对抗性安全问题尚未被探讨。本文从攻击者的角度出发，提出了一种开放集对抗性攻击(OSAttack)，旨在研究各种AMOSR方法的对抗性漏洞。首先，建立了AMOSR场景的对抗性威胁模型。随后，通过分析判别性和生成性开集识别的决策准则，提出了OSFGSM和OSPGD来降低AMOSR的性能。最后，利用一系列定性和定量指标对OSAttack对AMOSR的影响进行了评估。结果表明，尽管AMOSR模型对常规干扰信号的抵抗力有所增强，但它们仍然容易受到对手例子的攻击。



## **5. Deep Reinforcement Learning with Spiking Q-learning**

具有峰值Q学习的深度强化学习 cs.NE

15 pages, 7 figures

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2201.09754v3) [paper-pdf](http://arxiv.org/pdf/2201.09754v3)

**Authors**: Ding Chen, Peixi Peng, Tiejun Huang, Yonghong Tian

**Abstract**: With the help of special neuromorphic hardware, spiking neural networks (SNNs) are expected to realize artificial intelligence (AI) with less energy consumption. It provides a promising energy-efficient way for realistic control tasks by combining SNNs with deep reinforcement learning (RL). There are only a few existing SNN-based RL methods at present. Most of them either lack generalization ability or employ Artificial Neural Networks (ANNs) to estimate value function in training. The former needs to tune numerous hyper-parameters for each scenario, and the latter limits the application of different types of RL algorithm and ignores the large energy consumption in training. To develop a robust spike-based RL method, we draw inspiration from non-spiking interneurons found in insects and propose the deep spiking Q-network (DSQN), using the membrane voltage of non-spiking neurons as the representation of Q-value, which can directly learn robust policies from high-dimensional sensory inputs using end-to-end RL. Experiments conducted on 17 Atari games demonstrate the DSQN is effective and even outperforms the ANN-based deep Q-network (DQN) in most games. Moreover, the experiments show superior learning stability and robustness to adversarial attacks of DSQN.

摘要: 在特殊的神经形态硬件的帮助下，脉冲神经网络(SNN)有望以更少的能量消耗实现人工智能(AI)。它将神经网络和深度强化学习相结合，为实际控制任务提供了一种很有前途的节能方法。目前已有的基于SNN的RL方法很少。大多数人要么缺乏泛化能力，要么在训练中使用人工神经网络(ANN)来估计价值函数。前者需要针对每个场景调整大量的超参数，而后者限制了不同类型RL算法的应用，忽略了训练过程中的巨大能量消耗。为了开发一种稳健的基于棘波的RL方法，我们从昆虫中发现的非尖峰中间神经元中吸取灵感，提出了深度尖峰Q-网络(DSQN)，它使用非尖峰神经元的膜电压作为Q值的表示，可以使用端到端RL直接从高维感觉输入中学习鲁棒策略。在17个Atari游戏上的实验表明，DSQN是有效的，甚至在大多数游戏中都优于基于神经网络的深度Q网络(DQN)。实验表明，DSQN具有良好的学习稳定性和对敌意攻击的健壮性。



## **6. Learning-Based Difficulty Calibration for Enhanced Membership Inference Attacks**

基于学习的增强型成员推断攻击难度校准 cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2401.04929v2) [paper-pdf](http://arxiv.org/pdf/2401.04929v2)

**Authors**: Haonan Shi, Tu Ouyang, An Wang

**Abstract**: Machine learning models, in particular deep neural networks, are currently an integral part of various applications, from healthcare to finance. However, using sensitive data to train these models raises concerns about privacy and security. One method that has emerged to verify if the trained models are privacy-preserving is Membership Inference Attacks (MIA), which allows adversaries to determine whether a specific data point was part of a model's training dataset. While a series of MIAs have been proposed in the literature, only a few can achieve high True Positive Rates (TPR) in the low False Positive Rate (FPR) region (0.01%~1%). This is a crucial factor to consider for an MIA to be practically useful in real-world settings. In this paper, we present a novel approach to MIA that is aimed at significantly improving TPR at low FPRs. Our method, named learning-based difficulty calibration for MIA(LDC-MIA), characterizes data records by their hardness levels using a neural network classifier to determine membership. The experiment results show that LDC-MIA can improve TPR at low FPR by up to 4x compared to the other difficulty calibration based MIAs. It also has the highest Area Under ROC curve (AUC) across all datasets. Our method's cost is comparable with most of the existing MIAs, but is orders of magnitude more efficient than one of the state-of-the-art methods, LiRA, while achieving similar performance.

摘要: 机器学习模型，特别是深度神经网络，目前是从医疗保健到金融的各种应用程序的组成部分。然而，使用敏感数据来训练这些模型会引发对隐私和安全的担忧。出现的一种验证训练模型是否保护隐私的方法是成员推理攻击(MIA)，它允许对手确定特定数据点是否属于模型训练数据集的一部分。虽然文献中已经提出了一系列的MIA，但只有少数几个MIA能在低假阳性率(FPR)区域(0.01%~1%)获得高的真阳性率(TPR)。要使MIA在实际环境中发挥实际作用，这是需要考虑的关键因素。在本文中，我们提出了一种新的MIA方法，旨在显著改善低FPR下的TPR。我们的方法，称为基于学习的MIA难度校准(LDC-MIA)，使用神经网络分类器来确定成员身份，根据数据记录的硬度来表征数据记录。实验结果表明，与其他基于难度校正的MIA相比，LDC-MIA可以在较低的误码率下将TPR提高4倍。在所有数据集中，它也具有最高的ROC曲线下面积(AUC)。我们的方法的成本与大多数现有的MIA相当，但效率比最先进的方法之一LIRA高出数量级，同时实现了类似的性能。



## **7. BiasKG: Adversarial Knowledge Graphs to Induce Bias in Large Language Models**

BiasKG：对抗性知识图在大型语言模型中诱导偏见 cs.CL

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04756v1) [paper-pdf](http://arxiv.org/pdf/2405.04756v1)

**Authors**: Chu Fei Luo, Ahmad Ghawanmeh, Xiaodan Zhu, Faiza Khan Khattak

**Abstract**: Modern large language models (LLMs) have a significant amount of world knowledge, which enables strong performance in commonsense reasoning and knowledge-intensive tasks when harnessed properly. The language model can also learn social biases, which has a significant potential for societal harm. There have been many mitigation strategies proposed for LLM safety, but it is unclear how effective they are for eliminating social biases. In this work, we propose a new methodology for attacking language models with knowledge graph augmented generation. We refactor natural language stereotypes into a knowledge graph, and use adversarial attacking strategies to induce biased responses from several open- and closed-source language models. We find our method increases bias in all models, even those trained with safety guardrails. This demonstrates the need for further research in AI safety, and further work in this new adversarial space.

摘要: 现代大型语言模型（LLM）拥有大量的世界知识，如果利用得当，可以在常识推理和知识密集型任务中取得出色的性能。语言模型还可以学习社会偏见，这具有巨大的社会危害潜力。人们为LLM安全提出了许多缓解策略，但目前尚不清楚它们对于消除社会偏见的有效性如何。在这项工作中，我们提出了一种利用知识图增强生成来攻击语言模型的新方法。我们将自然语言刻板印象重新构建到知识图谱中，并使用对抗性攻击策略来诱导几个开放和封闭源语言模型的偏见反应。我们发现我们的方法增加了所有模型的偏差，甚至是那些接受过安全护栏训练的模型。这表明需要对人工智能安全进行进一步研究，并在这个新的对抗空间中进一步开展工作。



## **8. Demonstration of an Adversarial Attack Against a Multimodal Vision Language Model for Pathology Imaging**

演示针对病理成像多模式视觉语言模型的对抗攻击 eess.IV

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2401.02565v3) [paper-pdf](http://arxiv.org/pdf/2401.02565v3)

**Authors**: Poojitha Thota, Jai Prakash Veerla, Partha Sai Guttikonda, Mohammad S. Nasr, Shirin Nilizadeh, Jacob M. Luber

**Abstract**: In the context of medical artificial intelligence, this study explores the vulnerabilities of the Pathology Language-Image Pretraining (PLIP) model, a Vision Language Foundation model, under targeted attacks. Leveraging the Kather Colon dataset with 7,180 H&E images across nine tissue types, our investigation employs Projected Gradient Descent (PGD) adversarial perturbation attacks to induce misclassifications intentionally. The outcomes reveal a 100% success rate in manipulating PLIP's predictions, underscoring its susceptibility to adversarial perturbations. The qualitative analysis of adversarial examples delves into the interpretability challenges, shedding light on nuanced changes in predictions induced by adversarial manipulations. These findings contribute crucial insights into the interpretability, domain adaptation, and trustworthiness of Vision Language Models in medical imaging. The study emphasizes the pressing need for robust defenses to ensure the reliability of AI models. The source codes for this experiment can be found at https://github.com/jaiprakash1824/VLM_Adv_Attack.

摘要: 在医学人工智能的背景下，本研究探索了视觉语言基础模型-病理语言-图像预训练(PLIP)模型在有针对性攻击下的脆弱性。利用Kather Colon数据集和9种组织类型的7,180张H&E图像，我们的研究使用了投影梯度下降(PGD)对抗性扰动攻击来故意诱导错误分类。结果显示，PLIP操纵预测的成功率为100%，突显出其易受对手干扰的影响。对抗性例子的定性分析深入到了可解释性的挑战，揭示了对抗性操纵导致的预测的细微变化。这些发现为医学成像中视觉语言模型的可解释性、领域适应性和可信性提供了重要的见解。该研究强调，迫切需要强大的防御措施，以确保人工智能模型的可靠性。这个实验的源代码可以在https://github.com/jaiprakash1824/VLM_Adv_Attack.上找到



## **9. Fully Automated Selfish Mining Analysis in Efficient Proof Systems Blockchains**

高效证明系统区块链中的全自动自私挖掘分析 cs.CR

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04420v1) [paper-pdf](http://arxiv.org/pdf/2405.04420v1)

**Authors**: Krishnendu Chatterjee, Amirali Ebrahimzadeh, Mehrdad Karrabi, Krzysztof Pietrzak, Michelle Yeo, Đorđe Žikelić

**Abstract**: We study selfish mining attacks in longest-chain blockchains like Bitcoin, but where the proof of work is replaced with efficient proof systems -- like proofs of stake or proofs of space -- and consider the problem of computing an optimal selfish mining attack which maximizes expected relative revenue of the adversary, thus minimizing the chain quality. To this end, we propose a novel selfish mining attack that aims to maximize this objective and formally model the attack as a Markov decision process (MDP). We then present a formal analysis procedure which computes an $\epsilon$-tight lower bound on the optimal expected relative revenue in the MDP and a strategy that achieves this $\epsilon$-tight lower bound, where $\epsilon>0$ may be any specified precision. Our analysis is fully automated and provides formal guarantees on the correctness. We evaluate our selfish mining attack and observe that it achieves superior expected relative revenue compared to two considered baselines.   In concurrent work [Sarenche FC'24] does an automated analysis on selfish mining in predictable longest-chain blockchains based on efficient proof systems. Predictable means the randomness for the challenges is fixed for many blocks (as used e.g., in Ouroboros), while we consider unpredictable (Bitcoin-like) chains where the challenge is derived from the previous block.

摘要: 我们研究了比特币等最长链区块链中的自私挖掘攻击，但工作证明被高效的证明系统取代--如赌注证明或空间证明--并考虑计算最优自私挖掘攻击的问题，该攻击最大化对手的预期相对收益，从而最小化链质量。为此，我们提出了一种新的自私挖掘攻击，旨在最大化这一目标，并将攻击形式化地建模为马尔可夫决策过程(MDP)。然后，我们给出了一个形式的分析程序，它计算了MDP中最优预期相对收益的$\epsilon$-紧下界，并给出了一个实现这个$\epsilon$-紧下界的策略，其中$\epsilon>0$可以是任意指定的精度。我们的分析是完全自动化的，并为正确性提供正式保证。我们评估了我们的自私挖掘攻击，并观察到与两个考虑的基线相比，它实现了更好的预期相对收益。在并发工作[Sarhene FC‘24]中，基于高效的证明系统，对可预测的最长链区块链中的自私挖掘进行了自动化分析。可预测意味着挑战的随机性对于许多区块是固定的(例如，在Ouroboros中使用)，而我们认为挑战来自前一个区块的不可预测(类似比特币的)链。



## **10. NeuroIDBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research**

NeuroIDBench：基于脑电波的认证研究方法标准化的开源基准框架 cs.CR

21 pages, 5 Figures, 3 tables, Submitted to the Journal of  Information Security and Applications

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2402.08656v4) [paper-pdf](http://arxiv.org/pdf/2402.08656v4)

**Authors**: Avinash Kumar Chaurasia, Matin Fallahi, Thorsten Strufe, Philipp Terhörst, Patricia Arias Cabarcos

**Abstract**: Biometric systems based on brain activity have been proposed as an alternative to passwords or to complement current authentication techniques. By leveraging the unique brainwave patterns of individuals, these systems offer the possibility of creating authentication solutions that are resistant to theft, hands-free, accessible, and potentially even revocable. However, despite the growing stream of research in this area, faster advance is hindered by reproducibility problems. Issues such as the lack of standard reporting schemes for performance results and system configuration, or the absence of common evaluation benchmarks, make comparability and proper assessment of different biometric solutions challenging. Further, barriers are erected to future work when, as so often, source code is not published open access. To bridge this gap, we introduce NeuroIDBench, a flexible open source tool to benchmark brainwave-based authentication models. It incorporates nine diverse datasets, implements a comprehensive set of pre-processing parameters and machine learning algorithms, enables testing under two common adversary models (known vs unknown attacker), and allows researchers to generate full performance reports and visualizations. We use NeuroIDBench to investigate the shallow classifiers and deep learning-based approaches proposed in the literature, and to test robustness across multiple sessions. We observe a 37.6% reduction in Equal Error Rate (EER) for unknown attacker scenarios (typically not tested in the literature), and we highlight the importance of session variability to brainwave authentication. All in all, our results demonstrate the viability and relevance of NeuroIDBench in streamlining fair comparisons of algorithms, thereby furthering the advancement of brainwave-based authentication through robust methodological practices.

摘要: 基于大脑活动的生物识别系统已经被提出作为密码的替代方案，或者是对当前身份验证技术的补充。通过利用个人独特的脑电波模式，这些系统提供了创建防盗、免提、可访问甚至可能可撤销的身份验证解决方案的可能性。然而，尽管这一领域的研究越来越多，但重复性问题阻碍了更快的进展。缺乏性能结果和系统配置的标准报告方案，或缺乏通用的评估基准等问题，使不同生物识别解决方案的可比性和适当评估具有挑战性。此外，当源代码不公开、开放获取时，就会为未来的工作设置障碍。为了弥补这一差距，我们引入了NeuroIDBch，这是一个灵活的开源工具，用于对基于脑电波的身份验证模型进行基准测试。它整合了九个不同的数据集，实现了一套全面的预处理参数和机器学习算法，可以在两个常见的对手模型(已知和未知攻击者)下进行测试，并允许研究人员生成完整的性能报告和可视化。我们使用NeuroIDB边来研究文献中提出的浅层分类器和基于深度学习的方法，并测试多个会话的健壮性。我们观察到，对于未知攻击者场景(通常未在文献中进行测试)，等错误率(EER)降低了37.6%，并强调了会话可变性对脑电波身份验证的重要性。总而言之，我们的结果证明了NeuroIDBtch在简化公平的算法比较方面的可行性和相关性，从而通过稳健的方法学实践进一步推进基于脑电波的身份验证。



## **11. Revisiting character-level adversarial attacks**

重新审视角色级对抗攻击 cs.LG

Accepted in ICML 2024

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04346v1) [paper-pdf](http://arxiv.org/pdf/2405.04346v1)

**Authors**: Elias Abad Rocamora, Yongtao Wu, Fanghui Liu, Grigorios G. Chrysos, Volkan Cevher

**Abstract**: Adversarial attacks in Natural Language Processing apply perturbations in the character or token levels. Token-level attacks, gaining prominence for their use of gradient-based methods, are susceptible to altering sentence semantics, leading to invalid adversarial examples. While character-level attacks easily maintain semantics, they have received less attention as they cannot easily adopt popular gradient-based methods, and are thought to be easy to defend. Challenging these beliefs, we introduce Charmer, an efficient query-based adversarial attack capable of achieving high attack success rate (ASR) while generating highly similar adversarial examples. Our method successfully targets both small (BERT) and large (Llama 2) models. Specifically, on BERT with SST-2, Charmer improves the ASR in 4.84% points and the USE similarity in 8% points with respect to the previous art. Our implementation is available in https://github.com/LIONS-EPFL/Charmer.

摘要: 自然语言处理中的对抗攻击对字符或令牌级别施加扰动。令牌级攻击因使用基于梯度的方法而变得越来越重要，很容易改变句子语义，从而导致无效的对抗性示例。虽然字符级攻击很容易维护语义，但它们受到的关注较少，因为它们不能轻易采用流行的基于梯度的方法，并且被认为很容易防御。基于这些信念，我们引入了Charmer，这是一种高效的基于查询的对抗性攻击，能够实现高攻击成功率（ASB），同时生成高度相似的对抗性示例。我们的方法成功地针对小型（BERT）和大型（Llama 2）模型。具体来说，在采用CST-2的BERT上，Charmer将ASB提高了4.84%，与之前的作品相比，USE相似性提高了8%。我们的实现可在https://github.com/LIONS-EPFL/Charmer上获取。



## **12. Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore**

这是谁写的？零镜头LLM生成文本检测的关键是GECScore cs.CL

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04286v1) [paper-pdf](http://arxiv.org/pdf/2405.04286v1)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xuebo Liu, Lidia S. Chao, Min Zhang

**Abstract**: The efficacy of an large language model (LLM) generated text detector depends substantially on the availability of sizable training data. White-box zero-shot detectors, which require no such data, are nonetheless limited by the accessibility of the source model of the LLM-generated text. In this paper, we propose an simple but effective black-box zero-shot detection approach, predicated on the observation that human-written texts typically contain more grammatical errors than LLM-generated texts. This approach entails computing the Grammar Error Correction Score (GECScore) for the given text to distinguish between human-written and LLM-generated text. Extensive experimental results show that our method outperforms current state-of-the-art (SOTA) zero-shot and supervised methods, achieving an average AUROC of 98.7% and showing strong robustness against paraphrase and adversarial perturbation attacks.

摘要: 大型语言模型（LLM）生成的文本检测器的功效在很大程度上取决于大量训练数据的可用性。白盒零镜头检测器不需要此类数据，但仍受到LLM生成文本源模型可访问性的限制。在本文中，我们提出了一种简单但有效的黑匣子零镜头检测方法，其基础是人类书面文本通常比LLM生成的文本包含更多的语法错误。这种方法需要计算给定文本的语法错误纠正分数（GECScore），以区分人类编写的文本和LLM生成的文本。大量的实验结果表明，我们的方法优于当前最先进的（SOTA）零射击和监督方法，实现了98.7%的平均AUROC，并对重述和对抗性扰动攻击表现出强大的鲁棒性。



## **13. A Stealthy Wrongdoer: Feature-Oriented Reconstruction Attack against Split Learning**

一个偷偷摸摸的犯错者：针对分裂学习的以冲突为导向的重建攻击 cs.CR

Accepted to CVPR 2024

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04115v1) [paper-pdf](http://arxiv.org/pdf/2405.04115v1)

**Authors**: Xiaoyang Xu, Mengda Yang, Wenzhe Yi, Ziang Li, Juan Wang, Hongxin Hu, Yong Zhuang, Yaxin Liu

**Abstract**: Split Learning (SL) is a distributed learning framework renowned for its privacy-preserving features and minimal computational requirements. Previous research consistently highlights the potential privacy breaches in SL systems by server adversaries reconstructing training data. However, these studies often rely on strong assumptions or compromise system utility to enhance attack performance. This paper introduces a new semi-honest Data Reconstruction Attack on SL, named Feature-Oriented Reconstruction Attack (FORA). In contrast to prior works, FORA relies on limited prior knowledge, specifically that the server utilizes auxiliary samples from the public without knowing any client's private information. This allows FORA to conduct the attack stealthily and achieve robust performance. The key vulnerability exploited by FORA is the revelation of the model representation preference in the smashed data output by victim client. FORA constructs a substitute client through feature-level transfer learning, aiming to closely mimic the victim client's representation preference. Leveraging this substitute client, the server trains the attack model to effectively reconstruct private data. Extensive experiments showcase FORA's superior performance compared to state-of-the-art methods. Furthermore, the paper systematically evaluates the proposed method's applicability across diverse settings and advanced defense strategies.

摘要: Split Learning(SL)是一种分布式学习框架，以其隐私保护功能和最小的计算要求而闻名。以前的研究一直强调，通过服务器对手重建训练数据，SL系统中潜在的隐私泄露。然而，这些研究往往依赖强假设或折衷系统效用来提高攻击性能。介绍了一种新的基于SL的半诚实数据重构攻击--面向特征的重构攻击(FORA)。与以前的工作不同，FORA依赖于有限的先验知识，特别是服务器使用来自公共的辅助样本，而不知道任何客户的私人信息。这使得Fora能够悄悄地进行攻击，并实现稳健的性能。Fora利用的关键漏洞是受害者客户端输出的粉碎数据中暴露的模型表示首选项。FORA通过特征级迁移学习构造了一个替代客户，旨在更好地模拟受害客户的表征偏好。利用这个替代客户端，服务器训练攻击模型以有效地重建私有数据。广泛的实验表明，与最先进的方法相比，FORA的性能更优越。此外，本文还系统地评估了该方法在不同环境和先进防御策略下的适用性。



## **14. Explainability-Informed Targeted Malware Misclassification**

有解释性的定向恶意软件错误分类 cs.CR

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04010v1) [paper-pdf](http://arxiv.org/pdf/2405.04010v1)

**Authors**: Quincy Card, Kshitiz Aryal, Maanak Gupta

**Abstract**: In recent years, there has been a surge in malware attacks across critical infrastructures, requiring further research and development of appropriate response and remediation strategies in malware detection and classification. Several works have used machine learning models for malware classification into categories, and deep neural networks have shown promising results. However, these models have shown its vulnerabilities against intentionally crafted adversarial attacks, which yields misclassification of a malicious file. Our paper explores such adversarial vulnerabilities of neural network based malware classification system in the dynamic and online analysis environments. To evaluate our approach, we trained Feed Forward Neural Networks (FFNN) to classify malware categories based on features obtained from dynamic and online analysis environments. We use the state-of-the-art method, SHapley Additive exPlanations (SHAP), for the feature attribution for malware classification, to inform the adversarial attackers about the features with significant importance on classification decision. Using the explainability-informed features, we perform targeted misclassification adversarial white-box evasion attacks using the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks against the trained classifier. Our results demonstrated high evasion rate for some instances of attacks, showing a clear vulnerability of a malware classifier for such attacks. We offer recommendations for a balanced approach and a benchmark for much-needed future research into evasion attacks against malware classifiers, and develop more robust and trustworthy solutions.

摘要: 近年来，跨关键基础设施的恶意软件攻击激增，需要在恶意软件检测和分类方面进一步研究和开发适当的响应和补救策略。有几项工作使用机器学习模型将恶意软件分类，深度神经网络也显示出了令人振奋的结果。然而，这些模型显示了其针对故意构建的敌意攻击的漏洞，这些攻击会导致对恶意文件的错误分类。本文探讨了基于神经网络的恶意软件分类系统在动态分析和在线分析环境中的攻击漏洞。为了评估我们的方法，我们训练前馈神经网络(FFNN)根据从动态和在线分析环境中获得的特征对恶意软件类别进行分类。对于恶意软件分类的特征属性，我们使用了最新的Shapley Additive Ex释义(Shap)方法，将对分类决策有重要意义的特征告知恶意攻击者。利用可解释性信息特征，我们使用快速梯度符号方法(FGSM)和投影梯度下降(PGD)方法对训练好的分类器进行有针对性的误分类敌意白盒逃避攻击。我们的结果表明，对于一些攻击实例，逃避率很高，这表明恶意软件分类器对此类攻击存在明显的漏洞。我们为针对恶意软件分类器的躲避攻击的未来研究提供了平衡方法的建议和基准，并开发了更强大和值得信赖的解决方案。



## **15. Navigating Quantum Security Risks in Networked Environments: A Comprehensive Study of Quantum-Safe Network Protocols**

应对网络环境中的量子安全风险：量子安全网络协议的全面研究 cs.CR

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2404.08232v2) [paper-pdf](http://arxiv.org/pdf/2404.08232v2)

**Authors**: Yaser Baseri, Vikas Chouhan, Abdelhakim Hafid

**Abstract**: The emergence of quantum computing poses a formidable security challenge to network protocols traditionally safeguarded by classical cryptographic algorithms. This paper provides an exhaustive analysis of vulnerabilities introduced by quantum computing in a diverse array of widely utilized security protocols across the layers of the TCP/IP model, including TLS, IPsec, SSH, PGP, and more. Our investigation focuses on precisely identifying vulnerabilities susceptible to exploitation by quantum adversaries at various migration stages for each protocol while also assessing the associated risks and consequences for secure communication. We delve deep into the impact of quantum computing on each protocol, emphasizing potential threats posed by quantum attacks and scrutinizing the effectiveness of post-quantum cryptographic solutions. Through carefully evaluating vulnerabilities and risks that network protocols face in the post-quantum era, this study provides invaluable insights to guide the development of appropriate countermeasures. Our findings contribute to a broader comprehension of quantum computing's influence on network security and offer practical guidance for protocol designers, implementers, and policymakers in addressing the challenges stemming from the advancement of quantum computing. This comprehensive study is a crucial step toward fortifying the security of networked environments in the quantum age.

摘要: 量子计算的出现对传统上由经典密码算法保护的网络协议提出了严峻的安全挑战。本文详尽分析了量子计算在各种广泛使用的安全协议(包括TLS、IPSec、SSH、PGP等)的TCP/IP模型各层中引入的漏洞。我们的调查重点是准确地识别在每个协议的不同迁移阶段容易被量子攻击者利用的漏洞，同时还评估了相关的风险和安全通信的后果。我们深入研究了量子计算对每个协议的影响，强调了量子攻击带来的潜在威胁，并仔细审查了后量子密码解决方案的有效性。通过仔细评估后量子时代网络协议面临的漏洞和风险，本研究为指导制定适当的对策提供了宝贵的见解。我们的发现有助于更广泛地理解量子计算对网络安全的影响，并为协议设计者、实施者和政策制定者提供实用指导，以应对量子计算进步带来的挑战。这项全面的研究是在量子时代加强网络环境安全的关键一步。



## **16. Enhancing O-RAN Security: Evasion Attacks and Robust Defenses for Graph Reinforcement Learning-based Connection Management**

增强O-RAN安全性：基于图强化学习的连接管理的规避攻击和稳健防御 cs.CR

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03891v1) [paper-pdf](http://arxiv.org/pdf/2405.03891v1)

**Authors**: Ravikumar Balakrishnan, Marius Arvinte, Nageen Himayat, Hosein Nikopour, Hassnaa Moustafa

**Abstract**: Adversarial machine learning, focused on studying various attacks and defenses on machine learning (ML) models, is rapidly gaining importance as ML is increasingly being adopted for optimizing wireless systems such as Open Radio Access Networks (O-RAN). A comprehensive modeling of the security threats and the demonstration of adversarial attacks and defenses on practical AI based O-RAN systems is still in its nascent stages. We begin by conducting threat modeling to pinpoint attack surfaces in O-RAN using an ML-based Connection management application (xApp) as an example. The xApp uses a Graph Neural Network trained using Deep Reinforcement Learning and achieves on average 54% improvement in the coverage rate measured as the 5th percentile user data rates. We then formulate and demonstrate evasion attacks that degrade the coverage rates by as much as 50% through injecting bounded noise at different threat surfaces including the open wireless medium itself. Crucially, we also compare and contrast the effectiveness of such attacks on the ML-based xApp and a non-ML based heuristic. We finally develop and demonstrate robust training-based defenses against the challenging physical/jamming-based attacks and show a 15% improvement in the coverage rates when compared to employing no defense over a range of noise budgets

摘要: 对抗性机器学习专注于研究机器学习模型上的各种攻击和防御，随着机器学习模型越来越多地被用于优化开放无线接入网络(O-RAN)等无线系统，机器学习正迅速变得越来越重要。对实用的基于人工智能的O-RAN系统进行安全威胁的全面建模以及对抗性攻击和防御的演示仍处于初级阶段。我们首先以基于ML的连接管理应用程序(XApp)为例进行威胁建模，以确定O-RAN中的攻击面。XApp使用使用深度强化学习训练的图形神经网络，以第5个百分位的用户数据速率衡量，覆盖率平均提高54%。然后，我们制定和演示了规避攻击，通过在不同的威胁表面(包括开放的无线介质本身)注入有界噪声，使覆盖率降低高达50%。重要的是，我们还比较了基于ML的xApp和非基于ML的启发式攻击的有效性。我们最终开发和演示了针对具有挑战性的物理/基于干扰的攻击的基于训练的强大防御，并显示与在一系列噪声预算内不使用防御相比，覆盖率提高了15%



## **17. On Adversarial Examples for Text Classification by Perturbing Latent Representations**

利用扰动潜在表示进行文本分类的对抗示例 cs.LG

7 pages

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03789v1) [paper-pdf](http://arxiv.org/pdf/2405.03789v1)

**Authors**: Korn Sooksatra, Bikram Khanal, Pablo Rivas

**Abstract**: Recently, with the advancement of deep learning, several applications in text classification have advanced significantly. However, this improvement comes with a cost because deep learning is vulnerable to adversarial examples. This weakness indicates that deep learning is not very robust. Fortunately, the input of a text classifier is discrete. Hence, it can prevent the classifier from state-of-the-art attacks. Nonetheless, previous works have generated black-box attacks that successfully manipulate the discrete values of the input to find adversarial examples. Therefore, instead of changing the discrete values, we transform the input into its embedding vector containing real values to perform the state-of-the-art white-box attacks. Then, we convert the perturbed embedding vector back into a text and name it an adversarial example. In summary, we create a framework that measures the robustness of a text classifier by using the gradients of the classifier.

摘要: 最近，随着深度学习的进步，文本分类中的几个应用取得了显着进步。然而，这种改进是有代价的，因为深度学习容易受到对抗性示例的影响。这个弱点表明深度学习不是很强大。幸运的是，文本分类器的输入是离散的。因此，它可以防止分类器受到最先进的攻击。尽管如此，之前的作品已经产生了黑匣子攻击，这些攻击成功地操纵输入的离散值以找到对抗性示例。因此，我们不会更改离散值，而是将输入转换为包含实值的嵌入载体，以执行最先进的白盒攻击。然后，我们将受干扰的嵌入载体转换回文本，并将其命名为对抗性示例。总而言之，我们创建了一个框架，通过使用分类器的梯度来衡量文本分类器的稳健性。



## **18. RandOhm: Mitigating Impedance Side-channel Attacks using Randomized Circuit Configurations**

RandOhm：使用随机电路查找器缓解阻抗侧通道攻击 cs.CR

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2401.08925v2) [paper-pdf](http://arxiv.org/pdf/2401.08925v2)

**Authors**: Saleh Khalaj Monfared, Domenic Forte, Shahin Tajik

**Abstract**: Physical side-channel attacks can compromise the security of integrated circuits. Most physical side-channel attacks (e.g., power or electromagnetic) exploit the dynamic behavior of a chip, typically manifesting as changes in current consumption or voltage fluctuations where algorithmic countermeasures, such as masking, can effectively mitigate them. However, as demonstrated recently, these mitigation techniques are not entirely effective against backscattered side-channel attacks such as impedance analysis. In the case of an impedance attack, an adversary exploits the data-dependent impedance variations of the chip power delivery network (PDN) to extract secret information. In this work, we introduce RandOhm, which exploits a moving target defense (MTD) strategy based on the partial reconfiguration (PR) feature of mainstream FPGAs and programmable SoCs to defend against impedance side-channel attacks. We demonstrate that the information leakage through the PDN impedance could be significantly reduced via runtime reconfiguration of the secret-sensitive parts of the circuitry. Hence, by constantly randomizing the placement and routing of the circuit, one can decorrelate the data-dependent computation from the impedance value. Moreover, in contrast to existing PR-based countermeasures, RandOhm deploys open-source bitstream manipulation tools on programmable SoCs to speed up the randomization and provide real-time protection. To validate our claims, we apply RandOhm to AES ciphers realized on 28-nm FPGAs. We analyze the resiliency of our approach by performing non-profiled and profiled impedance analysis attacks and investigate the overhead of our mitigation in terms of delay and performance.

摘要: 物理侧通道攻击可能会危及集成电路的安全性。大多数物理侧通道攻击(例如，电源或电磁)利用芯片的动态行为，通常表现为电流消耗或电压波动的变化，其中算法对策(如掩蔽)可以有效地缓解这些变化。然而，正如最近所证明的那样，这些缓解技术并不能完全有效地对抗诸如阻抗分析之类的反向散射侧信道攻击。在阻抗攻击的情况下，攻击者利用芯片功率传输网络(PDN)的依赖于数据的阻抗变化来提取秘密信息。在这项工作中，我们介绍了RandOhm，它利用了一种基于主流FPGA和可编程SoC的部分重构(PR)特性的移动目标防御(MTD)策略来防御阻抗旁通道攻击。我们证明，通过PDN阻抗的信息泄漏可以通过在运行时重新配置电路的秘密敏感部分来显著减少。因此，通过不断地随机化电路的布局和布线，可以将依赖于数据的计算与阻抗值分离。此外，与现有的基于PR的对策相比，RandOhm在可编程SoC上部署了开源的比特流处理工具，以加快随机化并提供实时保护。为了验证我们的声明，我们将RandOhm应用于在28 nm FPGA上实现的AES密码。我们通过执行非配置文件和配置文件阻抗分析攻击来分析我们方法的弹性，并从延迟和性能方面调查我们的缓解开销。



## **19. Understanding the Vulnerability of Skeleton-based Human Activity Recognition via Black-box Attack**

通过黑匣子攻击了解基于普林斯顿的人类活动识别的漏洞 cs.CV

Accepted in Pattern Recognition. arXiv admin note: substantial text  overlap with arXiv:2103.05266

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2211.11312v2) [paper-pdf](http://arxiv.org/pdf/2211.11312v2)

**Authors**: Yunfeng Diao, He Wang, Tianjia Shao, Yong-Liang Yang, Kun Zhou, David Hogg, Meng Wang

**Abstract**: Human Activity Recognition (HAR) has been employed in a wide range of applications, e.g. self-driving cars, where safety and lives are at stake. Recently, the robustness of skeleton-based HAR methods have been questioned due to their vulnerability to adversarial attacks. However, the proposed attacks require the full-knowledge of the attacked classifier, which is overly restrictive. In this paper, we show such threats indeed exist, even when the attacker only has access to the input/output of the model. To this end, we propose the very first black-box adversarial attack approach in skeleton-based HAR called BASAR. BASAR explores the interplay between the classification boundary and the natural motion manifold. To our best knowledge, this is the first time data manifold is introduced in adversarial attacks on time series. Via BASAR, we find on-manifold adversarial samples are extremely deceitful and rather common in skeletal motions, in contrast to the common belief that adversarial samples only exist off-manifold. Through exhaustive evaluation, we show that BASAR can deliver successful attacks across classifiers, datasets, and attack modes. By attack, BASAR helps identify the potential causes of the model vulnerability and provides insights on possible improvements. Finally, to mitigate the newly identified threat, we propose a new adversarial training approach by leveraging the sophisticated distributions of on/off-manifold adversarial samples, called mixed manifold-based adversarial training (MMAT). MMAT can successfully help defend against adversarial attacks without compromising classification accuracy.

摘要: 人类活动识别(HAR)已被广泛应用于安全和生命受到威胁的自动驾驶汽车等领域。最近，基于骨架的HAR方法的健壮性受到了质疑，因为它们容易受到对手攻击。然而，所提出的攻击需要被攻击分类器的完全知识，这是过度限制的。在这篇文章中，我们证明了这样的威胁确实存在，即使攻击者只有权访问模型的输入/输出。为此，我们在基于骨架的HAR中提出了第一种黑盒对抗攻击方法BASAR。巴萨探索了分类边界和自然运动流形之间的相互作用。据我们所知，这是首次将数据流形引入时间序列的对抗性攻击中。通过BASAR，我们发现流形上的对抗性样本具有极大的欺骗性，并且在骨骼运动中相当常见，而不是通常认为对抗性样本只存在于流形外。通过详尽的评估，我们证明了Basar可以跨分类器、数据集和攻击模式进行成功的攻击。通过攻击，Basar帮助识别模型漏洞的潜在原因，并提供可能改进的见解。最后，为了缓解新识别的威胁，我们提出了一种新的对抗训练方法，即基于混合流形的对抗训练(MMAT)。MMAT可以在不影响分类准确性的情况下成功地帮助防御对手攻击。



## **20. Provably Unlearnable Examples**

可证明难以学习的例子 cs.LG

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03316v1) [paper-pdf](http://arxiv.org/pdf/2405.03316v1)

**Authors**: Derui Wang, Minhui Xue, Bo Li, Seyit Camtepe, Liming Zhu

**Abstract**: The exploitation of publicly accessible data has led to escalating concerns regarding data privacy and intellectual property (IP) breaches in the age of artificial intelligence. As a strategy to safeguard both data privacy and IP-related domain knowledge, efforts have been undertaken to render shared data unlearnable for unauthorized models in the wild. Existing methods apply empirically optimized perturbations to the data in the hope of disrupting the correlation between the inputs and the corresponding labels such that the data samples are converted into Unlearnable Examples (UEs). Nevertheless, the absence of mechanisms that can verify how robust the UEs are against unknown unauthorized models and train-time techniques engenders several problems. First, the empirically optimized perturbations may suffer from the problem of cross-model generalization, which echoes the fact that the unauthorized models are usually unknown to the defender. Second, UEs can be mitigated by train-time techniques such as data augmentation and adversarial training. Furthermore, we find that a simple recovery attack can restore the clean-task performance of the classifiers trained on UEs by slightly perturbing the learned weights. To mitigate the aforementioned problems, in this paper, we propose a mechanism for certifying the so-called $(q, \eta)$-Learnability of an unlearnable dataset via parametric smoothing. A lower certified $(q, \eta)$-Learnability indicates a more robust protection over the dataset. Finally, we try to 1) improve the tightness of certified $(q, \eta)$-Learnability and 2) design Provably Unlearnable Examples (PUEs) which have reduced $(q, \eta)$-Learnability. According to experimental results, PUEs demonstrate both decreased certified $(q, \eta)$-Learnability and enhanced empirical robustness compared to existing UEs.

摘要: 在人工智能时代，对公开可访问数据的利用导致了人们对数据隐私和知识产权(IP)侵犯的担忧不断升级。作为一项保护数据隐私和知识产权相关领域知识的战略，已经做出努力，使未经授权的模型无法在野外学习共享数据。现有方法将经验优化的扰动应用于数据，希望破坏输入和相应标签之间的相关性，从而将数据样本转换为不可学习的示例(UE)。然而，缺乏机制来验证UE对未知的未经授权的模型和训练时间技术的健壮性，会产生几个问题。首先，经验优化的扰动可能会受到跨模型泛化的问题，这呼应了这样一个事实，即未经授权的模型通常对于防御者是未知的。其次，可以通过数据增强和对抗性训练等训练时间技术来缓解UE。此外，我们发现，简单的恢复攻击可以通过对学习的权重进行轻微扰动来恢复在UE上训练的分类器的干净任务性能。为了缓解上述问题，在本文中，我们提出了一种通过参数平滑来证明不可学习数据集的所谓$(Q，\eta)$-可学习性的机制。较低的认证$(Q，\eta)$-可学习性表明对数据集的保护更强大。最后，我们试图1)提高已证明的$(Q，eta)$-可学习性的紧性；2)设计降低了$(q，eta)$-可学习性的可证明不可学习实例(PUE)。实验结果表明，与已有的UE相比，PUE的认证$(Q，ETA)可学习性降低，经验稳健性增强。



## **21. Illusory Attacks: Information-Theoretic Detectability Matters in Adversarial Attacks**

幻象攻击：信息论可检测性在对抗性攻击中很重要 cs.AI

ICLR 2024 Spotlight (top 5%)

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2207.10170v5) [paper-pdf](http://arxiv.org/pdf/2207.10170v5)

**Authors**: Tim Franzmeyer, Stephen McAleer, João F. Henriques, Jakob N. Foerster, Philip H. S. Torr, Adel Bibi, Christian Schroeder de Witt

**Abstract**: Autonomous agents deployed in the real world need to be robust against adversarial attacks on sensory inputs. Robustifying agent policies requires anticipating the strongest attacks possible. We demonstrate that existing observation-space attacks on reinforcement learning agents have a common weakness: while effective, their lack of information-theoretic detectability constraints makes them detectable using automated means or human inspection. Detectability is undesirable to adversaries as it may trigger security escalations. We introduce {\epsilon}-illusory, a novel form of adversarial attack on sequential decision-makers that is both effective and of {\epsilon}-bounded statistical detectability. We propose a novel dual ascent algorithm to learn such attacks end-to-end. Compared to existing attacks, we empirically find {\epsilon}-illusory to be significantly harder to detect with automated methods, and a small study with human participants (IRB approval under reference R84123/RE001) suggests they are similarly harder to detect for humans. Our findings suggest the need for better anomaly detectors, as well as effective hardware- and system-level defenses. The project website can be found at https://tinyurl.com/illusory-attacks.

摘要: 部署在现实世界中的自主代理需要强大地抵御对感觉输入的敌意攻击。将代理策略规模化需要预测可能最强的攻击。我们证明了现有的对强化学习代理的观察空间攻击有一个共同的弱点：虽然有效，但它们缺乏信息论的可检测性约束，使得它们可以使用自动手段或人工检查来检测。对于对手来说，可探测性是不可取的，因为它可能会引发安全升级。介绍了一种新的针对序列决策者的对抗性攻击--{epsilon}-幻觉，它既是有效的，又具有{epsilon}-有界的统计可检测性。我们提出了一种新的双重上升算法来端到端地学习此类攻击。与现有的攻击相比，我们根据经验发现，使用自动方法检测{\epsilon}-幻觉要困难得多，一项针对人类参与者的小型研究(参考R84123/RE001下的IRB批准)表明，对于人类来说，它们同样更难检测到。我们的发现表明，需要更好的异常检测器，以及有效的硬件和系统级防御。该项目的网址为：https://tinyurl.com/illusory-attacks.



## **22. Purify Unlearnable Examples via Rate-Constrained Variational Autoencoders**

通过速率约束变分自动编码器净化不可学习的示例 cs.CR

Accepted by ICML 2024

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.01460v2) [paper-pdf](http://arxiv.org/pdf/2405.01460v2)

**Authors**: Yi Yu, Yufei Wang, Song Xia, Wenhan Yang, Shijian Lu, Yap-Peng Tan, Alex C. Kot

**Abstract**: Unlearnable examples (UEs) seek to maximize testing error by making subtle modifications to training examples that are correctly labeled. Defenses against these poisoning attacks can be categorized based on whether specific interventions are adopted during training. The first approach is training-time defense, such as adversarial training, which can mitigate poisoning effects but is computationally intensive. The other approach is pre-training purification, e.g., image short squeezing, which consists of several simple compressions but often encounters challenges in dealing with various UEs. Our work provides a novel disentanglement mechanism to build an efficient pre-training purification method. Firstly, we uncover rate-constrained variational autoencoders (VAEs), demonstrating a clear tendency to suppress the perturbations in UEs. We subsequently conduct a theoretical analysis for this phenomenon. Building upon these insights, we introduce a disentangle variational autoencoder (D-VAE), capable of disentangling the perturbations with learnable class-wise embeddings. Based on this network, a two-stage purification approach is naturally developed. The first stage focuses on roughly eliminating perturbations, while the second stage produces refined, poison-free results, ensuring effectiveness and robustness across various scenarios. Extensive experiments demonstrate the remarkable performance of our method across CIFAR-10, CIFAR-100, and a 100-class ImageNet-subset. Code is available at https://github.com/yuyi-sd/D-VAE.

摘要: 不能学习的例子(UE)试图通过对正确标记的训练例子进行微妙的修改来最大化测试误差。针对这些中毒攻击的防御措施可以根据是否在训练期间采取特定干预措施进行分类。第一种方法是训练时间防御，例如对抗性训练，这种方法可以减轻中毒影响，但计算密集。另一种方法是训练前净化，例如图像短压缩，它由几个简单的压缩组成，但在处理各种UE时经常遇到挑战。我们的工作为构建高效的预训练净化方法提供了一种新的解缠机制。首先，我们发现了码率受限的变分自动编码器(VAE)，显示了抑制UE中微扰的明显趋势。我们随后对这一现象进行了理论分析。基于这些见解，我们引入了一种解缠变分自动编码器(D-VAE)，它能够通过可学习的类嵌入来解缠扰动。在这个网络的基础上，自然发展了一种两级提纯方法。第一阶段侧重于粗略地消除干扰，而第二阶段产生精炼的、无毒的结果，确保在各种情况下的有效性和健壮性。广泛的实验表明，我们的方法在CIFAR-10、CIFAR-100和100类ImageNet子集上具有显著的性能。代码可在https://github.com/yuyi-sd/D-VAE.上找到



## **23. Are aligned neural networks adversarially aligned?**

对齐的神经网络是否反向对齐？ cs.CL

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2306.15447v2) [paper-pdf](http://arxiv.org/pdf/2306.15447v2)

**Authors**: Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Anas Awadalla, Pang Wei Koh, Daphne Ippolito, Katherine Lee, Florian Tramer, Ludwig Schmidt

**Abstract**: Large language models are now tuned to align with the goals of their creators, namely to be "helpful and harmless." These models should respond helpfully to user questions, but refuse to answer requests that could cause harm. However, adversarial users can construct inputs which circumvent attempts at alignment. In this work, we study adversarial alignment, and ask to what extent these models remain aligned when interacting with an adversarial user who constructs worst-case inputs (adversarial examples). These inputs are designed to cause the model to emit harmful content that would otherwise be prohibited. We show that existing NLP-based optimization attacks are insufficiently powerful to reliably attack aligned text models: even when current NLP-based attacks fail, we can find adversarial inputs with brute force. As a result, the failure of current attacks should not be seen as proof that aligned text models remain aligned under adversarial inputs.   However the recent trend in large-scale ML models is multimodal models that allow users to provide images that influence the text that is generated. We show these models can be easily attacked, i.e., induced to perform arbitrary un-aligned behavior through adversarial perturbation of the input image. We conjecture that improved NLP attacks may demonstrate this same level of adversarial control over text-only models.

摘要: 大型语言模型现在被调整为与它们的创建者的目标保持一致，即“有益和无害”。这些模型应该对用户的问题做出有益的回应，但拒绝回答可能造成伤害的请求。然而，敌意用户可以构建绕过对齐尝试的输入。在这项工作中，我们研究对抗性对齐，并询问当与构建最坏情况输入(对抗性例子)的对抗性用户交互时，这些模型在多大程度上保持对齐。这些输入旨在导致模型排放本来被禁止的有害内容。我们证明了现有的基于NLP的优化攻击不足以可靠地攻击对齐的文本模型：即使当前基于NLP的攻击失败，我们也可以发现具有暴力的敌意输入。因此，当前攻击的失败不应被视为对齐的文本模型在敌意输入下保持对齐的证据。然而，大规模ML模型的最新趋势是允许用户提供影响所生成文本的图像的多模式模型。我们证明了这些模型可以很容易地被攻击，即通过对输入图像的对抗性扰动来诱导执行任意的非对齐行为。我们推测，改进的NLP攻击可能会展示出对纯文本模型的同样水平的敌意控制。



## **24. Exploring Frequencies via Feature Mixing and Meta-Learning for Improving Adversarial Transferability**

通过特征混合和元学习探索频率以提高对抗性可移植性 cs.CV

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03193v1) [paper-pdf](http://arxiv.org/pdf/2405.03193v1)

**Authors**: Juanjuan Weng, Zhiming Luo, Shaozi Li

**Abstract**: Recent studies have shown that Deep Neural Networks (DNNs) are susceptible to adversarial attacks, with frequency-domain analysis underscoring the significance of high-frequency components in influencing model predictions. Conversely, targeting low-frequency components has been effective in enhancing attack transferability on black-box models. In this study, we introduce a frequency decomposition-based feature mixing method to exploit these frequency characteristics in both clean and adversarial samples. Our findings suggest that incorporating features of clean samples into adversarial features extracted from adversarial examples is more effective in attacking normally-trained models, while combining clean features with the adversarial features extracted from low-frequency parts decomposed from the adversarial samples yields better results in attacking defense models. However, a conflict issue arises when these two mixing approaches are employed simultaneously. To tackle the issue, we propose a cross-frequency meta-optimization approach comprising the meta-train step, meta-test step, and final update. In the meta-train step, we leverage the low-frequency components of adversarial samples to boost the transferability of attacks against defense models. Meanwhile, in the meta-test step, we utilize adversarial samples to stabilize gradients, thereby enhancing the attack's transferability against normally trained models. For the final update, we update the adversarial sample based on the gradients obtained from both meta-train and meta-test steps. Our proposed method is evaluated through extensive experiments on the ImageNet-Compatible dataset, affirming its effectiveness in improving the transferability of attacks on both normally-trained CNNs and defense models.   The source code is available at https://github.com/WJJLL/MetaSSA.

摘要: 最近的研究表明，深度神经网络(DNN)容易受到敌意攻击，频域分析强调了高频分量在影响模型预测中的重要性。相反，瞄准低频分量在增强黑盒模型上的攻击可转移性方面是有效的。在这项研究中，我们引入了一种基于频率分解的特征混合方法来利用干净样本和恶意样本中的这些频率特征。我们的结果表明，在攻击正常训练的模型时，将干净样本的特征与从对抗性样本中提取的对抗性特征相结合是更有效的，而将干净特征与从对抗性样本分解的低频部分提取的对抗性特征相结合，在攻击防御模型中会产生更好的效果。然而，当这两种混合方法同时使用时，就会出现冲突问题。为了解决这一问题，我们提出了一种跨频率元优化方法，包括元训练步骤、元测试步骤和最终更新。在元训练步骤中，我们利用对抗性样本的低频成分来提高攻击对防御模型的可转移性。同时，在元测试步骤中，我们利用对抗性样本来稳定梯度，从而增强了攻击对正常训练模型的可转移性。对于最终的更新，我们基于从元训练和元测试步骤获得的梯度来更新对抗性样本。通过在与ImageNet兼容的数据集上的大量实验对我们提出的方法进行了评估，证实了该方法在提高对正常训练的CNN和防御模型的攻击的可转移性方面的有效性。源代码可在https://github.com/WJJLL/MetaSSA.上找到



## **25. To Each (Textual Sequence) Its Own: Improving Memorized-Data Unlearning in Large Language Models**

每个（文本序列）自有：改进大型语言模型中的简化数据去学习 cs.LG

Published as a conference paper at ICML 2024

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03097v1) [paper-pdf](http://arxiv.org/pdf/2405.03097v1)

**Authors**: George-Octavian Barbulescu, Peter Triantafillou

**Abstract**: LLMs have been found to memorize training textual sequences and regurgitate verbatim said sequences during text generation time. This fact is known to be the cause of privacy and related (e.g., copyright) problems. Unlearning in LLMs then takes the form of devising new algorithms that will properly deal with these side-effects of memorized data, while not hurting the model's utility. We offer a fresh perspective towards this goal, namely, that each textual sequence to be forgotten should be treated differently when being unlearned based on its degree of memorization within the LLM. We contribute a new metric for measuring unlearning quality, an adversarial attack showing that SOTA algorithms lacking this perspective fail for privacy, and two new unlearning methods based on Gradient Ascent and Task Arithmetic, respectively. A comprehensive performance evaluation across an extensive suite of NLP tasks then mapped the solution space, identifying the best solutions under different scales in model capacities and forget set sizes and quantified the gains of the new approaches.

摘要: 已经发现LLM在文本生成时间内记忆训练文本序列并逐字地返回所述序列。众所周知，这一事实是隐私和相关(例如，版权)问题的原因。然后，在LLMS中，遗忘的形式是设计新的算法，这些算法将适当地处理记忆数据的这些副作用，同时不会损害模型的实用性。我们为这一目标提供了一个新的视角，即每个被遗忘的文本序列在被遗忘时应该根据它在LLM中的记忆程度而得到不同的对待。我们提出了一种新的遗忘质量度量，一种敌意攻击表明缺乏这种视角的SOTA算法在隐私方面是失败的，以及两种新的遗忘方法，分别基于梯度上升和任务算法。然后，对一系列NLP任务进行了全面的性能评估，绘制了解决方案空间图，确定了模型容量和忘记集合大小不同尺度下的最佳解决方案，并量化了新方法的收益。



## **26. A Characterization of Semi-Supervised Adversarially-Robust PAC Learnability**

半监督对抗鲁棒PAC可学习性的描述 cs.LG

NeurIPS 2022 camera-ready

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2202.05420v3) [paper-pdf](http://arxiv.org/pdf/2202.05420v3)

**Authors**: Idan Attias, Steve Hanneke, Yishay Mansour

**Abstract**: We study the problem of learning an adversarially robust predictor to test time attacks in the semi-supervised PAC model. We address the question of how many labeled and unlabeled examples are required to ensure learning. We show that having enough unlabeled data (the size of a labeled sample that a fully-supervised method would require), the labeled sample complexity can be arbitrarily smaller compared to previous works, and is sharply characterized by a different complexity measure. We prove nearly matching upper and lower bounds on this sample complexity. This shows that there is a significant benefit in semi-supervised robust learning even in the worst-case distribution-free model, and establishes a gap between the supervised and semi-supervised label complexities which is known not to hold in standard non-robust PAC learning.

摘要: 我们研究学习对抗鲁棒预测器以测试半监督PAC模型中的时间攻击的问题。我们解决了需要多少带标签和未带标签的示例来确保学习的问题。我们表明，拥有足够的未标记数据（全监督方法所需的标记样本的大小），标记样本的复杂性与之前的作品相比可以任意小，并且由不同的复杂性衡量标准来鲜明地特征。我们证明了该样本复杂性的上下限几乎匹配。这表明，即使在最坏的无分布模型中，半监督鲁棒学习也有显着的好处，并在监督和半监督标签复杂性之间建立了差距，众所周知，这在标准非鲁棒PAC学习中不存在。



## **27. Adversarially Robust PAC Learnability of Real-Valued Functions**

实值函数的对抗鲁棒PAC可学习性 cs.LG

accepted to ICML2023

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2206.12977v3) [paper-pdf](http://arxiv.org/pdf/2206.12977v3)

**Authors**: Idan Attias, Steve Hanneke

**Abstract**: We study robustness to test-time adversarial attacks in the regression setting with $\ell_p$ losses and arbitrary perturbation sets. We address the question of which function classes are PAC learnable in this setting. We show that classes of finite fat-shattering dimension are learnable in both realizable and agnostic settings. Moreover, for convex function classes, they are even properly learnable. In contrast, some non-convex function classes provably require improper learning algorithms. Our main technique is based on a construction of an adversarially robust sample compression scheme of a size determined by the fat-shattering dimension. Along the way, we introduce a novel agnostic sample compression scheme for real-valued functions, which may be of independent interest.

摘要: 我们研究了在具有$\ell_p$损失和任意扰动集的回归设置中对测试时对抗攻击的鲁棒性。我们解决了在这种环境下哪些函数类可以PAC学习的问题。我们表明，有限的脂肪粉碎维度的类别在可实现和不可知的环境中都是可以学习的。此外，对于凸函数类来说，它们甚至是可以正确学习的。相比之下，一些非凸函数类可以证明需要不当的学习算法。我们的主要技术基于构建一个具有对抗性的稳健样本压缩方案，其大小由脂肪粉碎维度确定。一路上，我们为实值函数引入了一种新颖的不可知样本压缩方案，这可能是一种独立的兴趣。



## **28. Defense against Joint Poison and Evasion Attacks: A Case Study of DERMS**

防御联合毒物和躲避攻击：DEMS案例研究 cs.CR

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02989v1) [paper-pdf](http://arxiv.org/pdf/2405.02989v1)

**Authors**: Zain ul Abdeen, Padmaksha Roy, Ahmad Al-Tawaha, Rouxi Jia, Laura Freeman, Peter Beling, Chen-Ching Liu, Alberto Sangiovanni-Vincentelli, Ming Jin

**Abstract**: There is an upward trend of deploying distributed energy resource management systems (DERMS) to control modern power grids. However, DERMS controller communication lines are vulnerable to cyberattacks that could potentially impact operational reliability. While a data-driven intrusion detection system (IDS) can potentially thwart attacks during deployment, also known as the evasion attack, the training of the detection algorithm may be corrupted by adversarial data injected into the database, also known as the poisoning attack. In this paper, we propose the first framework of IDS that is robust against joint poisoning and evasion attacks. We formulate the defense mechanism as a bilevel optimization, where the inner and outer levels deal with attacks that occur during training time and testing time, respectively. We verify the robustness of our method on the IEEE-13 bus feeder model against a diverse set of poisoning and evasion attack scenarios. The results indicate that our proposed method outperforms the baseline technique in terms of accuracy, precision, and recall for intrusion detection.

摘要: 采用分布式能源管理系统(DERMS)来控制现代电网是一种趋势。然而，DERMS控制器通信线路容易受到网络攻击，这些攻击可能会潜在地影响操作可靠性。虽然数据驱动的入侵检测系统(IDS)可以潜在地阻止部署期间的攻击，也称为逃避攻击，但检测算法的训练可能会被注入数据库的敌意数据破坏，也称为中毒攻击。在本文中，我们提出了第一个入侵检测系统的框架，该框架对联合中毒和逃避攻击具有健壮性。我们将防御机制描述为双层优化，其中内部和外部级别分别处理在训练时间和测试时间发生的攻击。我们在IEEE-13公交支线模型上验证了该方法对不同的中毒和逃避攻击场景的稳健性。实验结果表明，该方法在入侵检测的准确率、精确度和召回率方面均优于Baseline方法。



## **29. You Only Need Half: Boosting Data Augmentation by Using Partial Content**

您只需要一半：通过使用部分内容来增强数据增强 cs.CV

Technical report,16 pages

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02830v1) [paper-pdf](http://arxiv.org/pdf/2405.02830v1)

**Authors**: Juntao Hu, Yuan Wu

**Abstract**: We propose a novel data augmentation method termed You Only Need hAlf (YONA), which simplifies the augmentation process. YONA bisects an image, substitutes one half with noise, and applies data augmentation techniques to the remaining half. This method reduces the redundant information in the original image, encourages neural networks to recognize objects from incomplete views, and significantly enhances neural networks' robustness. YONA is distinguished by its properties of parameter-free, straightforward application, enhancing various existing data augmentation strategies, and thereby bolstering neural networks' robustness without additional computational cost. To demonstrate YONA's efficacy, extensive experiments were carried out. These experiments confirm YONA's compatibility with diverse data augmentation methods and neural network architectures, yielding substantial improvements in CIFAR classification tasks, sometimes outperforming conventional image-level data augmentation methods. Furthermore, YONA markedly increases the resilience of neural networks to adversarial attacks. Additional experiments exploring YONA's variants conclusively show that masking half of an image optimizes performance. The code is available at https://github.com/HansMoe/YONA.

摘要: 我们提出了一种新的数据增强方法，称为你只需要一半(YONA)，它简化了增强过程。Yona将图像一分为二，用噪声替换一半，并对其余一半应用数据增强技术。该方法减少了原始图像中的冗余信息，鼓励神经网络从不完整的图像中识别目标，显著增强了神经网络的鲁棒性。YONA的特点是无参数，应用简单，增强了现有的各种数据增强策略，从而在不增加计算成本的情况下增强了神经网络的健壮性。为了证明Yona的疗效，进行了广泛的实验。这些实验证实了Yona与各种数据增强方法和神经网络体系结构的兼容性，在CIFAR分类任务方面产生了实质性的改进，有时性能优于传统的图像级数据增强方法。此外，Yona显著提高了神经网络对对手攻击的弹性。探索Yona变体的其他实验最终表明，遮盖图像的一半可以优化性能。代码可在https://github.com/HansMoe/YONA.上获得



## **30. Trojans in Large Language Models of Code: A Critical Review through a Trigger-Based Taxonomy**

大型语言代码模型中的特洛伊木马：基于触发器的分类学的批判性评论 cs.SE

arXiv admin note: substantial text overlap with arXiv:2305.03803

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02828v1) [paper-pdf](http://arxiv.org/pdf/2405.02828v1)

**Authors**: Aftab Hussain, Md Rafiqul Islam Rabin, Toufique Ahmed, Bowen Xu, Premkumar Devanbu, Mohammad Amin Alipour

**Abstract**: Large language models (LLMs) have provided a lot of exciting new capabilities in software development. However, the opaque nature of these models makes them difficult to reason about and inspect. Their opacity gives rise to potential security risks, as adversaries can train and deploy compromised models to disrupt the software development process in the victims' organization.   This work presents an overview of the current state-of-the-art trojan attacks on large language models of code, with a focus on triggers -- the main design point of trojans -- with the aid of a novel unifying trigger taxonomy framework. We also aim to provide a uniform definition of the fundamental concepts in the area of trojans in Code LLMs. Finally, we draw implications of findings on how code models learn on trigger design.

摘要: 大型语言模型（LLM）在软件开发中提供了许多令人兴奋的新功能。然而，这些模型的不透明性质使得它们难以推理和检查。它们的不透明性会带来潜在的安全风险，因为对手可以训练和部署受影响的模型，以扰乱受害者组织的软件开发流程。   这项工作概述了当前针对大型语言代码模型的最新特洛伊木马攻击，重点关注触发器（特洛伊木马的主要设计点），并在新颖的统一触发器分类框架的帮助下。我们还旨在为LLM代码中特洛伊木马领域的基本概念提供统一的定义。最后，我们得出了有关代码模型如何学习触发器设计的研究结果的影响。



## **31. Assessing Adversarial Robustness of Large Language Models: An Empirical Study**

评估大型语言模型的对抗稳健性：实证研究 cs.CL

16 pages, 9 figures, 10 tables

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2405.02764v1) [paper-pdf](http://arxiv.org/pdf/2405.02764v1)

**Authors**: Zeyu Yang, Zhao Meng, Xiaochen Zheng, Roger Wattenhofer

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing, but their robustness against adversarial attacks remains a critical concern. We presents a novel white-box style attack approach that exposes vulnerabilities in leading open-source LLMs, including Llama, OPT, and T5. We assess the impact of model size, structure, and fine-tuning strategies on their resistance to adversarial perturbations. Our comprehensive evaluation across five diverse text classification tasks establishes a new benchmark for LLM robustness. The findings of this study have far-reaching implications for the reliable deployment of LLMs in real-world applications and contribute to the advancement of trustworthy AI systems.

摘要: 大型语言模型（LLM）彻底改变了自然语言处理，但其对抗攻击的稳健性仍然是一个关键问题。我们提出了一种新颖的白盒式攻击方法，该方法暴露了领先开源LLM（包括Llama、OPT和T5）中的漏洞。我们评估了模型大小、结构和微调策略对其抵抗对抗性扰动的影响。我们对五种不同文本分类任务的全面评估为LLM稳健性建立了新基准。这项研究的结果对于LLM在现实世界应用程序中的可靠部署具有深远的影响，并有助于发展值得信赖的人工智能系统。



## **32. Updating Windows Malware Detectors: Balancing Robustness and Regression against Adversarial EXEmples**

更新Windows恶意软件检测器：平衡稳健性和回归与对抗性示例 cs.CR

11 pages, 3 figures, 7 tables

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2405.02646v1) [paper-pdf](http://arxiv.org/pdf/2405.02646v1)

**Authors**: Matous Kozak, Luca Demetrio, Dmitrijs Trizna, Fabio Roli

**Abstract**: Adversarial EXEmples are carefully-perturbed programs tailored to evade machine learning Windows malware detectors, with an on-going effort in developing robust models able to address detection effectiveness. However, even if robust models can prevent the majority of EXEmples, to maintain predictive power over time, models are fine-tuned to newer threats, leading either to partial updates or time-consuming retraining from scratch. Thus, even if the robustness against attacks is higher, the new models might suffer a regression in performance by misclassifying threats that were previously correctly detected. For these reasons, we study the trade-off between accuracy and regression when updating Windows malware detectors, by proposing EXE-scanner, a plugin that can be chained to existing detectors to promptly stop EXEmples without causing regression. We empirically show that previously-proposed hardening techniques suffer a regression of accuracy when updating non-robust models. On the contrary, we show that EXE-scanner exhibits comparable performance to robust models without regression of accuracy, and we show how to properly chain it after the base classifier to obtain the best performance without the need of costly retraining. To foster reproducibility, we openly release source code, along with the dataset of adversarial EXEmples based on state-of-the-art perturbation algorithms.

摘要: 对抗性的例子是精心设计的程序，旨在逃避机器学习Windows恶意软件检测器，并正在努力开发能够解决检测有效性的健壮模型。然而，即使稳健的模型可以阻止大多数例子，为了随着时间的推移保持预测能力，模型也会针对较新的威胁进行微调，导致要么部分更新，要么从头开始进行耗时的再培训。因此，即使对攻击的稳健性更高，新模型也可能会因为对以前正确检测到的威胁进行错误分类而导致性能退化。出于这些原因，我们研究了更新Windows恶意软件检测器时准确性和回归之间的权衡，提出了EXE-scanner，这是一个可以链接到现有检测器的插件，可以在不导致回归的情况下迅速停止示例。我们的经验表明，以前提出的强化技术在更新非稳健模型时会遭遇精度回归。相反，我们证明了EXE-scanner在没有精度回归的情况下表现出与健壮模型相当的性能，并展示了如何在基本分类器之后适当地将其链接以获得最佳性能，而不需要昂贵的再训练。为了促进重复性，我们公开发布源代码，以及基于最先进的扰动算法的对抗性例子的数据集。



## **33. A Group Key Establishment Scheme**

组密钥建立计划 cs.CR

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2109.15037v2) [paper-pdf](http://arxiv.org/pdf/2109.15037v2)

**Authors**: Sueda Guzey, Gunes Karabulut Kurt, Enver Ozdemir

**Abstract**: Group authentication is a method of confirmation that a set of users belong to a group and of distributing a common key among them. Unlike the standard authentication schemes where one central authority authenticates users one by one, group authentication can handle the authentication process at once for all members of the group. The recently presented group authentication algorithms mainly exploit Lagrange's polynomial interpolation along with elliptic curve groups over finite fields. As a fresh approach, this work suggests use of linear spaces for group authentication and key establishment for a group of any size. The approach with linear spaces introduces a reduced computation and communication load to establish a common shared key among the group members. The advantages of using vector spaces make the proposed method applicable to energy and resource constrained devices. In addition to providing lightweight authentication and key agreement, this proposal allows any user in a group to make a non-member to be a member, which is expected to be useful for autonomous systems in the future. The scheme is designed in a way that the sponsors of such members can easily be recognized by anyone in the group. Unlike the other group authentication schemes based on Lagrange's polynomial interpolation, the proposed scheme doesn't provide a tool for adversaries to compromise the whole group secrets by using only a few members' shares as well as it allows to recognize a non-member easily, which prevents service interruption attacks.

摘要: 组身份验证是一种确认一组用户属于一个组并在其中分发公共密钥的方法。与标准身份验证方案不同，在标准身份验证方案中，一个中央机构逐个对用户进行身份验证，而组身份验证可以同时处理组中所有成员的身份验证过程。最近提出的群认证算法主要利用拉格朗日多项式插值以及有限域上的椭圆曲线群。作为一种新的方法，这项工作建议使用线性空间进行组身份验证和建立任意大小的组的密钥。使用线性空间的方法减少了计算量和通信量，从而在组成员之间建立公共共享密钥。利用向量空间的优点使得该方法适用于能量和资源受限的设备。除了提供轻量级的认证和密钥协商外，该方案还允许组中的任何用户使非成员成为成员，这有望在未来的自治系统中发挥作用。该计划的设计方式是，这些成员的发起人很容易被小组中的任何人认出。与其他基于拉格朗日多项式插值的群认证方案不同的是，该方案不提供攻击者仅使用少数成员份额就能泄露整个群秘密的工具，而且它允许容易地识别非成员，从而防止服务中断攻击。



## **34. Leveraging the Human Ventral Visual Stream to Improve Neural Network Robustness**

利用人类视觉流提高神经网络鲁棒性 cs.CV

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2405.02564v1) [paper-pdf](http://arxiv.org/pdf/2405.02564v1)

**Authors**: Zhenan Shao, Linjian Ma, Bo Li, Diane M. Beck

**Abstract**: Human object recognition exhibits remarkable resilience in cluttered and dynamic visual environments. In contrast, despite their unparalleled performance across numerous visual tasks, Deep Neural Networks (DNNs) remain far less robust than humans, showing, for example, a surprising susceptibility to adversarial attacks involving image perturbations that are (almost) imperceptible to humans. Human object recognition likely owes its robustness, in part, to the increasingly resilient representations that emerge along the hierarchy of the ventral visual cortex. Here we show that DNNs, when guided by neural representations from a hierarchical sequence of regions in the human ventral visual stream, display increasing robustness to adversarial attacks. These neural-guided models also exhibit a gradual shift towards more human-like decision-making patterns and develop hierarchically smoother decision surfaces. Importantly, the resulting representational spaces differ in important ways from those produced by conventional smoothing methods, suggesting that such neural-guidance may provide previously unexplored robustness solutions. Our findings support the gradual emergence of human robustness along the ventral visual hierarchy and suggest that the key to DNN robustness may lie in increasing emulation of the human brain.

摘要: 人类目标识别在杂乱和动态的视觉环境中表现出显著的弹性。相比之下，尽管深度神经网络(DNN)在众多视觉任务中表现出无与伦比的性能，但它的健壮性仍然远远不如人类，例如，它对涉及(几乎)人类察觉不到的图像扰动的对抗性攻击具有惊人的敏感性。人类物体识别的健壮性可能部分归功于沿着腹侧视觉皮质层级出现的越来越有弹性的表征。在这里，我们表明，当DNN由来自人类腹侧视觉流中区域的分层序列的神经表示引导时，对对手攻击表现出越来越强的稳健性。这些神经引导的模型也显示出逐渐向更接近人类的决策模式转变，并形成层次化的更平滑的决策表面。重要的是，由此产生的表征空间与传统平滑方法产生的表征空间在重要方面不同，这表明这种神经指导可能提供以前未曾探索过的稳健性解决方案。我们的发现支持人类的健壮性沿着腹侧视觉层次逐渐出现，并表明DNN健壮性的关键可能在于增加对人脑的模拟。



## **35. Machine Learning Robustness: A Primer**

机器学习鲁棒性：入门 cs.LG

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2404.00897v3) [paper-pdf](http://arxiv.org/pdf/2404.00897v3)

**Authors**: Houssem Ben Braiek, Foutse Khomh

**Abstract**: This chapter explores the foundational concept of robustness in Machine Learning (ML) and its integral role in establishing trustworthiness in Artificial Intelligence (AI) systems. The discussion begins with a detailed definition of robustness, portraying it as the ability of ML models to maintain stable performance across varied and unexpected environmental conditions. ML robustness is dissected through several lenses: its complementarity with generalizability; its status as a requirement for trustworthy AI; its adversarial vs non-adversarial aspects; its quantitative metrics; and its indicators such as reproducibility and explainability. The chapter delves into the factors that impede robustness, such as data bias, model complexity, and the pitfalls of underspecified ML pipelines. It surveys key techniques for robustness assessment from a broad perspective, including adversarial attacks, encompassing both digital and physical realms. It covers non-adversarial data shifts and nuances of Deep Learning (DL) software testing methodologies. The discussion progresses to explore amelioration strategies for bolstering robustness, starting with data-centric approaches like debiasing and augmentation. Further examination includes a variety of model-centric methods such as transfer learning, adversarial training, and randomized smoothing. Lastly, post-training methods are discussed, including ensemble techniques, pruning, and model repairs, emerging as cost-effective strategies to make models more resilient against the unpredictable. This chapter underscores the ongoing challenges and limitations in estimating and achieving ML robustness by existing approaches. It offers insights and directions for future research on this crucial concept, as a prerequisite for trustworthy AI systems.

摘要: 本章探讨了机器学习(ML)中稳健性的基本概念及其在人工智能(AI)系统中建立可信度的不可或缺的作用。讨论开始于对稳健性的详细定义，将其描述为ML模型在不同和意外的环境条件下保持稳定性能的能力。ML稳健性通过几个方面进行剖析：它与通用性的互补性；它作为值得信赖的人工智能的要求的地位；它的对抗性与非对抗性方面；它的量化指标；以及它的可再现性和可解释性等指标。本章深入探讨了阻碍健壮性的因素，如数据偏差、模型复杂性和未指定的ML管道的陷阱。它从广泛的角度考察了健壮性评估的关键技术，包括涵盖数字和物理领域的对抗性攻击。它涵盖了深度学习(DL)软件测试方法的非对抗性数据转移和细微差别。讨论继续探索增强健壮性的改进策略，从去偏向和增强等以数据为中心的方法开始。进一步的考试包括各种以模型为中心的方法，如转移学习、对抗性训练和随机平滑。最后，讨论了训练后的方法，包括集合技术、修剪和模型修复，这些方法成为使模型对不可预测的情况更具弹性的成本效益策略。本章强调了在通过现有方法估计和实现ML健壮性方面的持续挑战和限制。它为未来对这一关键概念的研究提供了见解和方向，这是值得信赖的人工智能系统的先决条件。



## **36. GReAT: A Graph Regularized Adversarial Training Method**

GReAT：一种图规则化对抗训练方法 cs.LG

25 pages including references. 7 figures and 6 tables

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2310.05336v2) [paper-pdf](http://arxiv.org/pdf/2310.05336v2)

**Authors**: Samet Bayram, Kenneth Barner

**Abstract**: This paper presents GReAT (Graph Regularized Adversarial Training), a novel regularization method designed to enhance the robust classification performance of deep learning models. Adversarial examples, characterized by subtle perturbations that can mislead models, pose a significant challenge in machine learning. Although adversarial training is effective in defending against such attacks, it often overlooks the underlying data structure. In response, GReAT integrates graph based regularization into the adversarial training process, leveraging the data's inherent structure to enhance model robustness. By incorporating graph information during training, GReAT defends against adversarial attacks and improves generalization to unseen data. Extensive evaluations on benchmark datasets demonstrate that GReAT outperforms state of the art methods in robustness, achieving notable improvements in classification accuracy. Specifically, compared to the second best methods, GReAT achieves a performance increase of approximately 4.87% for CIFAR10 against FGSM attack and 10.57% for SVHN against FGSM attack. Additionally, for CIFAR10, GReAT demonstrates a performance increase of approximately 11.05% against PGD attack, and for SVHN, a 5.54% increase against PGD attack. This paper provides detailed insights into the proposed methodology, including numerical results and comparisons with existing approaches, highlighting the significant impact of GReAT in advancing the performance of deep learning models.

摘要: 为了提高深度学习模型的稳健分类性能，提出了一种新的正则化方法--GRIGH正则化对抗性训练方法。对抗性例子的特征是微妙的扰动，可能会误导模型，这对机器学习构成了巨大的挑战。尽管对抗性训练在防御此类攻击方面是有效的，但它往往忽略了底层的数据结构。作为回应，Great将基于图形的正则化集成到对抗性训练过程中，利用数据的内在结构来增强模型的健壮性。通过在训练中结合图形信息，Great可以防御对手攻击，并改进对不可见数据的泛化。在基准数据集上的广泛评估表明，该方法在稳健性方面优于最先进的方法，在分类精度方面取得了显著的改善。具体地说，与次优方法相比，Great实现了CIFAR10抗FGSM攻击的性能提高了约4.87%，SVHN抗FGSM攻击的性能提高了10.57%。此外，对于CIFAR10，针对PGD攻击的性能提升了约11.05%，对于SVHN，针对PGD攻击的性能提升了5.54%。本文对提出的方法提供了详细的见解，包括数字结果和与现有方法的比较，强调了BREAGE在提高深度学习模型的性能方面的重大影响。



## **37. Improving Interpretation Faithfulness for Vision Transformers**

提高视觉变形金刚的诠释忠实度 cs.CV

Accepted by ICML 2024

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2311.17983v2) [paper-pdf](http://arxiv.org/pdf/2311.17983v2)

**Authors**: Lijie Hu, Yixin Liu, Ninghao Liu, Mengdi Huai, Lichao Sun, Di Wang

**Abstract**: Vision Transformers (ViTs) have achieved state-of-the-art performance for various vision tasks. One reason behind the success lies in their ability to provide plausible innate explanations for the behavior of neural architectures. However, ViTs suffer from issues with explanation faithfulness, as their focal points are fragile to adversarial attacks and can be easily changed with even slight perturbations on the input image. In this paper, we propose a rigorous approach to mitigate these issues by introducing Faithful ViTs (FViTs). Briefly speaking, an FViT should have the following two properties: (1) The top-$k$ indices of its self-attention vector should remain mostly unchanged under input perturbation, indicating stable explanations; (2) The prediction distribution should be robust to perturbations. To achieve this, we propose a new method called Denoised Diffusion Smoothing (DDS), which adopts randomized smoothing and diffusion-based denoising. We theoretically prove that processing ViTs directly with DDS can turn them into FViTs. We also show that Gaussian noise is nearly optimal for both $\ell_2$ and $\ell_\infty$-norm cases. Finally, we demonstrate the effectiveness of our approach through comprehensive experiments and evaluations. Results show that FViTs are more robust against adversarial attacks while maintaining the explainability of attention, indicating higher faithfulness.

摘要: 视觉变形器(VITS)在各种视觉任务中取得了最先进的性能。成功背后的一个原因是他们有能力为神经结构的行为提供看似合理的天生解释。然而，VITS存在解释可信度的问题，因为它们的焦点对对抗性攻击很脆弱，即使对输入图像稍有扰动也可以很容易地改变。在本文中，我们提出了一种通过引入忠实VITS(FViT)来缓解这些问题的严格方法。简而言之，FViT应该具有以下两个性质：(1)在输入扰动下，其自我注意向量的前$k$指数应基本保持不变，表示稳定的解释；(2)预测分布应对扰动具有稳健性。为了达到这一目的，我们提出了一种新的去噪扩散平滑(DDS)方法，它采用了随机平滑和基于扩散的去噪。我们从理论上证明了直接用DDS处理VITS可以将其转化为FViT。我们还表明，对于$\ell_2$和$\ell_inty$-范数情况，高斯噪声几乎都是最优的。最后，我们通过全面的实验和评估证明了该方法的有效性。结果表明，FViT在保持注意的可解释性的同时，对敌意攻击具有更强的健壮性，表现出更高的忠诚度。



## **38. Adversarial Botometer: Adversarial Analysis for Social Bot Detection**

对抗性肉毒杆菌：社交机器人检测的对抗性分析 cs.SI

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2405.02016v1) [paper-pdf](http://arxiv.org/pdf/2405.02016v1)

**Authors**: Shaghayegh Najari, Davood Rafiee, Mostafa Salehi, Reza Farahbakhsh

**Abstract**: Social bots play a significant role in many online social networks (OSN) as they imitate human behavior. This fact raises difficult questions about their capabilities and potential risks. Given the recent advances in Generative AI (GenAI), social bots are capable of producing highly realistic and complex content that mimics human creativity. As the malicious social bots emerge to deceive people with their unrealistic content, identifying them and distinguishing the content they produce has become an actual challenge for numerous social platforms. Several approaches to this problem have already been proposed in the literature, but the proposed solutions have not been widely evaluated. To address this issue, we evaluate the behavior of a text-based bot detector in a competitive environment where some scenarios are proposed: \textit{First}, the tug-of-war between a bot and a bot detector is examined. It is interesting to analyze which party is more likely to prevail and which circumstances influence these expectations. In this regard, we model the problem as a synthetic adversarial game in which a conversational bot and a bot detector are engaged in strategic online interactions. \textit{Second}, the bot detection model is evaluated under attack examples generated by a social bot; to this end, we poison the dataset with attack examples and evaluate the model performance under this condition. \textit{Finally}, to investigate the impact of the dataset, a cross-domain analysis is performed. Through our comprehensive evaluation of different categories of social bots using two benchmark datasets, we were able to demonstrate some achivement that could be utilized in future works.

摘要: 社交机器人模仿人类的行为，在许多在线社交网络(OSN)中扮演着重要的角色。这一事实对他们的能力和潜在风险提出了棘手的问题。鉴于生成性人工智能(GenAI)的最新进展，社交机器人能够产生高度逼真和复杂的内容，模仿人类的创造力。随着恶意社交机器人的出现，用其不切实际的内容欺骗人们，识别它们并区分它们产生的内容已经成为众多社交平台面临的实际挑战。文献中已经提出了几种解决这个问题的方法，但提出的解决方案还没有得到广泛的评估。为了解决这个问题，我们评估了基于文本的机器人检测器在竞争环境中的行为，其中提出了一些场景：\textit{first}，考察了机器人和机器人检测器之间的拉锯战。分析哪一方更有可能获胜，以及哪些情况会影响这些预期，这是很有趣的。在这方面，我们将问题建模为一个合成的对抗性游戏，其中对话机器人和机器人检测器从事策略性的在线交互。在社交机器人生成攻击实例的情况下，对机器人检测模型进行了评估；为此，我们用攻击实例毒化了数据集，并在此条件下评估了模型的性能。为了调查数据集的影响，将执行跨域分析。通过使用两个基准数据集对不同类别的社交机器人进行综合评估，我们能够展示一些可以在未来的工作中使用的成果。



## **39. From Attack to Defense: Insights into Deep Learning Security Measures in Black-Box Settings**

从攻击到防御：黑匣子环境中深度学习安全措施的见解 cs.CR

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2405.01963v1) [paper-pdf](http://arxiv.org/pdf/2405.01963v1)

**Authors**: Firuz Juraev, Mohammed Abuhamad, Eric Chan-Tin, George K. Thiruvathukal, Tamer Abuhmed

**Abstract**: Deep Learning (DL) is rapidly maturing to the point that it can be used in safety- and security-crucial applications. However, adversarial samples, which are undetectable to the human eye, pose a serious threat that can cause the model to misbehave and compromise the performance of such applications. Addressing the robustness of DL models has become crucial to understanding and defending against adversarial attacks. In this study, we perform comprehensive experiments to examine the effect of adversarial attacks and defenses on various model architectures across well-known datasets. Our research focuses on black-box attacks such as SimBA, HopSkipJump, MGAAttack, and boundary attacks, as well as preprocessor-based defensive mechanisms, including bits squeezing, median smoothing, and JPEG filter. Experimenting with various models, our results demonstrate that the level of noise needed for the attack increases as the number of layers increases. Moreover, the attack success rate decreases as the number of layers increases. This indicates that model complexity and robustness have a significant relationship. Investigating the diversity and robustness relationship, our experiments with diverse models show that having a large number of parameters does not imply higher robustness. Our experiments extend to show the effects of the training dataset on model robustness. Using various datasets such as ImageNet-1000, CIFAR-100, and CIFAR-10 are used to evaluate the black-box attacks. Considering the multiple dimensions of our analysis, e.g., model complexity and training dataset, we examined the behavior of black-box attacks when models apply defenses. Our results show that applying defense strategies can significantly reduce attack effectiveness. This research provides in-depth analysis and insight into the robustness of DL models against various attacks, and defenses.

摘要: 深度学习(DL)正在迅速成熟，可以用于安全和安全关键应用程序。然而，肉眼无法检测到的敌意样本构成了严重的威胁，可能会导致模型行为不当并危及此类应用程序的性能。解决动态链接库模型的健壮性已成为理解和防御敌意攻击的关键。在这项研究中，我们进行了全面的实验，以检查对抗性攻击和防御对各种模型体系结构的影响，跨越众所周知的数据集。我们的研究重点是SIMBA、HopSkipJump、MGAAttack和边界攻击等黑盒攻击，以及基于预处理器的防御机制，包括比特压缩、中值平滑和JPEG滤波。通过不同模型的实验，我们的结果表明，攻击所需的噪声水平随着层数的增加而增加。此外，攻击成功率随着层数的增加而降低。这表明模型的复杂性和稳健性有显著的关系。通过研究多样性和稳健性的关系，我们用不同的模型进行的实验表明，拥有大量的参数并不意味着更高的稳健性。我们的实验进一步显示了训练数据集对模型稳健性的影响。使用ImageNet-1000、CIFAR-100和CIFAR-10等各种数据集来评估黑盒攻击。考虑到我们分析的多个维度，例如模型复杂性和训练数据集，我们检查了当模型应用防御时黑盒攻击的行为。我们的结果表明，应用防御策略会显著降低攻击效率。这项研究提供了深入的分析和洞察的DL模型对各种攻击和防御的健壮性。



## **40. Impact of Architectural Modifications on Deep Learning Adversarial Robustness**

架构修改对深度学习对抗鲁棒性的影响 cs.CV

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2405.01934v1) [paper-pdf](http://arxiv.org/pdf/2405.01934v1)

**Authors**: Firuz Juraev, Mohammed Abuhamad, Simon S. Woo, George K Thiruvathukal, Tamer Abuhmed

**Abstract**: Rapid advancements of deep learning are accelerating adoption in a wide variety of applications, including safety-critical applications such as self-driving vehicles, drones, robots, and surveillance systems. These advancements include applying variations of sophisticated techniques that improve the performance of models. However, such models are not immune to adversarial manipulations, which can cause the system to misbehave and remain unnoticed by experts. The frequency of modifications to existing deep learning models necessitates thorough analysis to determine the impact on models' robustness. In this work, we present an experimental evaluation of the effects of model modifications on deep learning model robustness using adversarial attacks. Our methodology involves examining the robustness of variations of models against various adversarial attacks. By conducting our experiments, we aim to shed light on the critical issue of maintaining the reliability and safety of deep learning models in safety- and security-critical applications. Our results indicate the pressing demand for an in-depth assessment of the effects of model changes on the robustness of models.

摘要: 深度学习的快速发展正在加速在各种应用中的采用，包括自动驾驶车辆、无人机、机器人和监控系统等安全关键型应用。这些改进包括应用各种复杂的技术来提高模型的性能。然而，这样的模型也不能幸免于敌意操纵，这种操纵可能会导致系统行为不当，并保持不被专家注意到。对现有深度学习模型的修改频率需要进行彻底的分析，以确定对模型稳健性的影响。在这项工作中，我们提出了一个实验评估的影响，模型修改对深度学习模型的稳健性使用对抗性攻击。我们的方法包括检查各种模型对各种对抗性攻击的稳健性。通过我们的实验，我们的目标是阐明在安全和安全关键应用程序中维护深度学习模型的可靠性和安全性的关键问题。我们的结果表明，深入评估模型变化对模型稳健性的影响是迫切需要的。



## **41. Stability of Explainable Recommendation**

可解释推荐的稳定性 cs.IR

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2405.01849v1) [paper-pdf](http://arxiv.org/pdf/2405.01849v1)

**Authors**: Sairamvinay Vijayaraghavan, Prasant Mohapatra

**Abstract**: Explainable Recommendation has been gaining attention over the last few years in industry and academia. Explanations provided along with recommendations in a recommender system framework have many uses: particularly reasoning why a suggestion is provided and how well an item aligns with a user's personalized preferences. Hence, explanations can play a huge role in influencing users to purchase products. However, the reliability of the explanations under varying scenarios has not been strictly verified from an empirical perspective. Unreliable explanations can bear strong consequences such as attackers leveraging explanations for manipulating and tempting users to purchase target items that the attackers would want to promote. In this paper, we study the vulnerability of existent feature-oriented explainable recommenders, particularly analyzing their performance under different levels of external noises added into model parameters. We conducted experiments by analyzing three important state-of-the-art (SOTA) explainable recommenders when trained on two widely used e-commerce based recommendation datasets of different scales. We observe that all the explainable models are vulnerable to increased noise levels. Experimental results verify our hypothesis that the ability to explain recommendations does decrease along with increasing noise levels and particularly adversarial noise does contribute to a much stronger decrease. Our study presents an empirical verification on the topic of robust explanations in recommender systems which can be extended to different types of explainable recommenders in RS.

摘要: 在过去的几年里，可解释性推荐在工业界和学术界得到了越来越多的关注。在推荐系统框架中提供的解释和推荐有许多用途：特别是推理为什么提供建议以及项目与用户的个性化偏好的匹配程度。因此，解释在影响用户购买产品方面发挥着巨大的作用。然而，在不同情景下的解释的可靠性并没有从经验的角度得到严格的验证。不可靠的解释可能会带来严重的后果，例如攻击者利用操纵和引诱用户购买攻击者想要推广的目标商品的解释。本文研究了现有的面向特征的可解释推荐器的脆弱性，特别是分析了它们在模型参数中加入不同程度的外部噪声时的性能。我们通过分析三个重要的最新技术(SOTA)可解释推荐器在两个广泛使用的基于不同规模的电子商务推荐数据集上进行了实验。我们观察到，所有可解释的模型都容易受到噪声水平增加的影响。实验结果验证了我们的假设，即解释推荐的能力确实随着噪声水平的增加而降低，尤其是对抗性噪声确实有助于更大程度的降低。我们的研究对推荐系统中的稳健解释这一主题进行了实证验证，可以扩展到RS中不同类型的可解释推荐器。



## **42. A Novel Approach to Guard from Adversarial Attacks using Stable Diffusion**

使用稳定扩散防范对抗攻击的新方法 cs.LG

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2405.01838v1) [paper-pdf](http://arxiv.org/pdf/2405.01838v1)

**Authors**: Trinath Sai Subhash Reddy Pittala, Uma Maheswara Rao Meleti, Geethakrishna Puligundla

**Abstract**: Recent developments in adversarial machine learning have highlighted the importance of building robust AI systems to protect against increasingly sophisticated attacks. While frameworks like AI Guardian are designed to defend against these threats, they often rely on assumptions that can limit their effectiveness. For example, they may assume attacks only come from one direction or include adversarial images in their training data. Our proposal suggests a different approach to the AI Guardian framework. Instead of including adversarial examples in the training process, we propose training the AI system without them. This aims to create a system that is inherently resilient to a wider range of attacks. Our method focuses on a dynamic defense strategy using stable diffusion that learns continuously and models threats comprehensively. We believe this approach can lead to a more generalized and robust defense against adversarial attacks.   In this paper, we outline our proposed approach, including the theoretical basis, experimental design, and expected impact on improving AI security against adversarial threats.

摘要: 对抗性机器学习的最新发展突显了建立强大的人工智能系统以防御日益复杂的攻击的重要性。虽然像AI Guardian这样的框架旨在防御这些威胁，但它们往往依赖于可能限制其有效性的假设。例如，他们可能认为攻击只来自一个方向，或者在他们的训练数据中包括对抗性图像。我们的提案提出了一种不同的AI Guardian框架方法。我们建议在没有对抗性例子的情况下训练人工智能系统，而不是在训练过程中包括对抗性例子。这旨在创建一个对更广泛的攻击具有内在弹性的系统。我们的方法侧重于一种使用稳定扩散的动态防御策略，该策略不断学习并全面建模威胁。我们相信，这种方法可以导致对对手攻击的更普遍和更强大的防御。在本文中，我们概述了我们提出的方法，包括理论基础、实验设计以及对提高人工智能安全对抗对手威胁的预期影响。



## **43. Explainability Guided Adversarial Evasion Attacks on Malware Detectors**

对恶意软件检测器的可解释性引导的对抗规避攻击 cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01728v1) [paper-pdf](http://arxiv.org/pdf/2405.01728v1)

**Authors**: Kshitiz Aryal, Maanak Gupta, Mahmoud Abdelsalam, Moustafa Saleh

**Abstract**: As the focus on security of Artificial Intelligence (AI) is becoming paramount, research on crafting and inserting optimal adversarial perturbations has become increasingly critical. In the malware domain, this adversarial sample generation relies heavily on the accuracy and placement of crafted perturbation with the goal of evading a trained classifier. This work focuses on applying explainability techniques to enhance the adversarial evasion attack on a machine-learning-based Windows PE malware detector. The explainable tool identifies the regions of PE malware files that have the most significant impact on the decision-making process of a given malware detector, and therefore, the same regions can be leveraged to inject the adversarial perturbation for maximum efficiency. Profiling all the PE malware file regions based on their impact on the malware detector's decision enables the derivation of an efficient strategy for identifying the optimal location for perturbation injection. The strategy should incorporate the region's significance in influencing the malware detector's decision and the sensitivity of the PE malware file's integrity towards modifying that region. To assess the utility of explainable AI in crafting an adversarial sample of Windows PE malware, we utilize the DeepExplainer module of SHAP for determining the contribution of each region of PE malware to its detection by a CNN-based malware detector, MalConv. Furthermore, we analyzed the significance of SHAP values at a more granular level by subdividing each section of Windows PE into small subsections. We then performed an adversarial evasion attack on the subsections based on the corresponding SHAP values of the byte sequences.

摘要: 随着人工智能(AI)对安全的关注变得越来越重要，设计和插入最优对抗扰动的研究变得越来越关键。在恶意软件领域，这种敌意样本生成在很大程度上依赖于精心设计的扰动的准确性和位置，目的是避开训练有素的分类器。本工作的重点是应用可解释性技术来增强基于机器学习的Windows PE恶意软件检测器上的敌意逃避攻击。可解释工具识别对给定恶意软件检测器的决策过程具有最显著影响的PE恶意软件文件的区域，因此，可以利用相同的区域来注入恶意扰动以获得最大效率。基于它们对恶意软件检测器决策的影响来分析所有PE恶意软件文件区域，使得能够推导出用于识别扰动注入的最佳位置的有效策略。该策略应包含该区域在影响恶意软件检测器的决策方面的重要性，以及PE恶意软件文件的完整性对修改该区域的敏感性。为了评估可解释人工智能在制作Windows PE恶意软件的恶意样本中的有效性，我们利用Shap的DeepExplainer模块来确定PE恶意软件的每个区域对基于CNN的恶意软件检测器MalConv检测的贡献。此外，我们通过将Windows PE的每个部分细分为小的子部分，在更细粒度的级别上分析了Shap值的意义。然后，我们根据字节序列的对应Shap值对子部分进行对抗性规避攻击。



## **44. ATTAXONOMY: Unpacking Differential Privacy Guarantees Against Practical Adversaries**

ATTAXONOMY：针对实际对手解开差异隐私保证 cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01716v1) [paper-pdf](http://arxiv.org/pdf/2405.01716v1)

**Authors**: Rachel Cummings, Shlomi Hod, Jayshree Sarathy, Marika Swanberg

**Abstract**: Differential Privacy (DP) is a mathematical framework that is increasingly deployed to mitigate privacy risks associated with machine learning and statistical analyses. Despite the growing adoption of DP, its technical privacy parameters do not lend themselves to an intelligible description of the real-world privacy risks associated with that deployment: the guarantee that most naturally follows from the DP definition is protection against membership inference by an adversary who knows all but one data record and has unlimited auxiliary knowledge. In many settings, this adversary is far too strong to inform how to set real-world privacy parameters.   One approach for contextualizing privacy parameters is via defining and measuring the success of technical attacks, but doing so requires a systematic categorization of the relevant attack space. In this work, we offer a detailed taxonomy of attacks, showing the various dimensions of attacks and highlighting that many real-world settings have been understudied. Our taxonomy provides a roadmap for analyzing real-world deployments and developing theoretical bounds for more informative privacy attacks. We operationalize our taxonomy by using it to analyze a real-world case study, the Israeli Ministry of Health's recent release of a birth dataset using DP, showing how the taxonomy enables fine-grained threat modeling and provides insight towards making informed privacy parameter choices. Finally, we leverage the taxonomy towards defining a more realistic attack than previously considered in the literature, namely a distributional reconstruction attack: we generalize Balle et al.'s notion of reconstruction robustness to a less-informed adversary with distributional uncertainty, and extend the worst-case guarantees of DP to this average-case setting.

摘要: 差分隐私(DP)是一个数学框架，它被越来越多地用于降低与机器学习和统计分析相关的隐私风险。尽管DP越来越多地被采用，但其技术隐私参数并不适合对与该部署相关的真实世界隐私风险进行易懂的描述：DP定义最自然地得出的保证是防止对手进行成员资格推理，该对手只知道一条数据记录，并且具有无限的辅助知识。在许多情况下，这个对手太强大了，无法告知如何设置现实世界的隐私参数。将隐私参数设置为上下文的一种方法是通过定义和衡量技术攻击的成功，但这样做需要对相关攻击空间进行系统分类。在这项工作中，我们提供了攻击的详细分类，显示了攻击的各个维度，并强调了许多现实世界的背景没有得到充分的研究。我们的分类为分析现实世界的部署和开发更具信息量的隐私攻击的理论界限提供了路线图。我们通过使用我们的分类来分析一个真实的案例研究，以色列卫生部最近使用DP发布了一个出生数据集，展示了分类如何实现细粒度的威胁建模，并为做出明智的隐私参数选择提供了洞察力。最后，我们利用分类来定义一种比以前在文献中考虑的更现实的攻击，即分布重构攻击：我们将Balle等人S的重构健壮性概念推广到具有分布不确定性的信息较少的对手，并将DP的最坏情况保证扩展到这种平均情况设置。



## **45. David and Goliath: An Empirical Evaluation of Attacks and Defenses for QNNs at the Deep Edge**

大卫和歌利亚：对深边缘QNN攻击和防御的经验评估 cs.LG

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2404.05688v2) [paper-pdf](http://arxiv.org/pdf/2404.05688v2)

**Authors**: Miguel Costa, Sandro Pinto

**Abstract**: ML is shifting from the cloud to the edge. Edge computing reduces the surface exposing private data and enables reliable throughput guarantees in real-time applications. Of the panoply of devices deployed at the edge, resource-constrained MCUs, e.g., Arm Cortex-M, are more prevalent, orders of magnitude cheaper, and less power-hungry than application processors or GPUs. Thus, enabling intelligence at the deep edge is the zeitgeist, with researchers focusing on unveiling novel approaches to deploy ANNs on these constrained devices. Quantization is a well-established technique that has proved effective in enabling the deployment of neural networks on MCUs; however, it is still an open question to understand the robustness of QNNs in the face of adversarial examples.   To fill this gap, we empirically evaluate the effectiveness of attacks and defenses from (full-precision) ANNs on (constrained) QNNs. Our evaluation includes three QNNs targeting TinyML applications, ten attacks, and six defenses. With this study, we draw a set of interesting findings. First, quantization increases the point distance to the decision boundary and leads the gradient estimated by some attacks to explode or vanish. Second, quantization can act as a noise attenuator or amplifier, depending on the noise magnitude, and causes gradient misalignment. Regarding adversarial defenses, we conclude that input pre-processing defenses show impressive results on small perturbations; however, they fall short as the perturbation increases. At the same time, train-based defenses increase the average point distance to the decision boundary, which holds after quantization. However, we argue that train-based defenses still need to smooth the quantization-shift and gradient misalignment phenomenons to counteract adversarial example transferability to QNNs. All artifacts are open-sourced to enable independent validation of results.

摘要: ML正在从云端转移到边缘。边缘计算减少了暴露私有数据的表面，并在实时应用中实现了可靠的吞吐量保证。在部署在边缘的所有设备中，资源受限的MCU(例如ARM Cortex-M)比应用处理器或GPU更普遍、更便宜、耗电量更低。因此，在深层实现智能是时代的精神，研究人员专注于推出在这些受限设备上部署ANN的新方法。量化是一种成熟的技术，已被证明在MCU上部署神经网络是有效的；然而，面对敌对例子，理解QNN的稳健性仍然是一个悬而未决的问题。为了填补这一空白，我们从经验上评估了(全精度)人工神经网络对(受约束的)QNN的攻击和防御的有效性。我们的评估包括三个针对TinyML应用程序的QNN，十个攻击和六个防御。通过这项研究，我们得出了一系列有趣的发现。首先，量化增加了到决策边界的点距离，并导致某些攻击估计的梯度爆炸或消失。其次，量化可以充当噪声衰减器或放大器，这取决于噪声的大小，并导致梯度失调。对于对抗性防御，我们得出的结论是，输入预处理防御在小扰动下表现出令人印象深刻的结果；然而，随着扰动的增加，它们不能满足要求。同时，基于训练的防御增加了到决策边界的平均点距离，量化后该距离保持不变。然而，我们认为，基于训练的防御仍然需要平滑量化位移和梯度错位现象，以抵消向QNN的对抗性示例转移。所有构件都是开源的，以支持结果的独立验证。



## **46. Adversarial Attacks on Reinforcement Learning Agents for Command and Control**

对指挥与控制强化学习代理的对抗攻击 cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01693v1) [paper-pdf](http://arxiv.org/pdf/2405.01693v1)

**Authors**: Ahaan Dabholkar, James Z. Hare, Mark Mittrick, John Richardson, Nicholas Waytowich, Priya Narayanan, Saurabh Bagchi

**Abstract**: Given the recent impact of Deep Reinforcement Learning in training agents to win complex games like StarCraft and DoTA(Defense Of The Ancients) - there has been a surge in research for exploiting learning based techniques for professional wargaming, battlefield simulation and modeling. Real time strategy games and simulators have become a valuable resource for operational planning and military research. However, recent work has shown that such learning based approaches are highly susceptible to adversarial perturbations. In this paper, we investigate the robustness of an agent trained for a Command and Control task in an environment that is controlled by an active adversary. The C2 agent is trained on custom StarCraft II maps using the state of the art RL algorithms - A3C and PPO. We empirically show that an agent trained using these algorithms is highly susceptible to noise injected by the adversary and investigate the effects these perturbations have on the performance of the trained agent. Our work highlights the urgent need to develop more robust training algorithms especially for critical arenas like the battlefield.

摘要: 鉴于最近深度强化学习在训练代理以赢得星际争霸和DOTA(古人防御)等复杂游戏中的影响，将基于学习的技术用于专业战争游戏、战场模拟和建模的研究激增。实时战略游戏和模拟器已经成为作战规划和军事研究的宝贵资源。然而，最近的工作表明，这种基于学习的方法非常容易受到对抗性扰动的影响。在本文中，我们研究了在由活跃的对手控制的环境中为指挥与控制任务训练的代理的稳健性。C2特工使用最先进的RL算法-A3C和PPO-在定制的星际争霸II地图上进行训练。我们的经验表明，使用这些算法训练的代理对对手注入的噪声非常敏感，并调查了这些扰动对训练的代理性能的影响。我们的工作突出了开发更健壮的训练算法的迫切需要，特别是对于战场这样的关键领域。



## **47. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

杰基尔博士和海德先生：法学硕士的两面 cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2312.03853v3) [paper-pdf](http://arxiv.org/pdf/2312.03853v3)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then followed a role-play style to elicit prohibited responses. By making use of personas, we show that such responses are actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Bard. We also introduce several ways of activating such adversarial personas, which show that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.

摘要: 最近，我们看到大型语言模型(LLM)的使用有所增加，特别是在聊天机器人助手等应用程序中。实施了安全机制和专门的培训程序，以防止这些助理做出不当反应。在这项工作中，我们绕过了ChatGPT和Bard(在某种程度上，Bing聊天)的这些措施，让他们模仿具有人格特征的复杂人物角色，而这些人物角色与诚实的助手不一致。我们首先为这些角色创建精致的传记，然后在与相同的聊天机器人的新会话中使用。然后，我们的对话遵循了角色扮演的风格，引发了被禁止的回应。通过使用人物角色，我们表明实际上提供了这样的响应，使得获得未经授权的、非法的或有害的信息成为可能。这项工作表明，通过使用对抗性人物角色，一个人可以克服ChatGPT和Bard提出的安全机制。我们还介绍了几种激活这种敌对角色的方法，这表明这两个聊天机器人都容易受到这种攻击。在相同的原则下，我们引入了两个防御措施，推动该模型解释可信任的个性，并使其对此类攻击更加健壮。



## **48. Generative AI in Cybersecurity**

网络安全中的生成人工智能 cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01674v1) [paper-pdf](http://arxiv.org/pdf/2405.01674v1)

**Authors**: Shivani Metta, Isaac Chang, Jack Parker, Michael P. Roman, Arturo F. Ehuan

**Abstract**: The dawn of Generative Artificial Intelligence (GAI), characterized by advanced models such as Generative Pre-trained Transformers (GPT) and other Large Language Models (LLMs), has been pivotal in reshaping the field of data analysis, pattern recognition, and decision-making processes. This surge in GAI technology has ushered in not only innovative opportunities for data processing and automation but has also introduced significant cybersecurity challenges.   As GAI rapidly progresses, it outstrips the current pace of cybersecurity protocols and regulatory frameworks, leading to a paradox wherein the same innovations meant to safeguard digital infrastructures also enhance the arsenal available to cyber criminals. These adversaries, adept at swiftly integrating and exploiting emerging technologies, may utilize GAI to develop malware that is both more covert and adaptable, thus complicating traditional cybersecurity efforts.   The acceleration of GAI presents an ambiguous frontier for cybersecurity experts, offering potent tools for threat detection and response, while concurrently providing cyber attackers with the means to engineer more intricate and potent malware. Through the joint efforts of Duke Pratt School of Engineering, Coalfire, and Safebreach, this research undertakes a meticulous analysis of how malicious agents are exploiting GAI to augment their attack strategies, emphasizing a critical issue for the integrity of future cybersecurity initiatives. The study highlights the critical need for organizations to proactively identify and develop more complex defensive strategies to counter the sophisticated employment of GAI in malware creation.

摘要: 以生成性预训练转换器(GPT)和其他大型语言模型(LLM)等高级模型为特征的生成性人工智能(GAI)的出现，在重塑数据分析、模式识别和决策过程领域起到了关键作用。GAI技术的激增不仅为数据处理和自动化带来了创新机遇，也带来了重大的网络安全挑战。随着GAI的快速发展，它超过了当前网络安全协议和监管框架的速度，导致了一个悖论，即旨在保护数字基础设施的相同创新也增强了网络犯罪分子可用的武器库。这些擅长快速整合和利用新兴技术的对手可能会利用GAI开发更隐蔽和更具适应性的恶意软件，从而使传统的网络安全努力复杂化。GAI的加速为网络安全专家提供了一个模糊的边界，为威胁检测和响应提供了强大的工具，同时也为网络攻击者提供了设计更复杂、更强大的恶意软件的手段。通过杜克·普拉特工程学院、煤火和安全漏洞的共同努力，这项研究对恶意代理如何利用GAI来增强其攻击策略进行了细致的分析，强调了未来网络安全计划的完整性的关键问题。这项研究突出表明，组织迫切需要主动识别和开发更复杂的防御策略，以对抗GAI在恶意软件创建中的复杂使用。



## **49. Position Paper: Beyond Robustness Against Single Attack Types**

立场文件：超越针对单一攻击类型的稳健性 cs.LG

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01349v1) [paper-pdf](http://arxiv.org/pdf/2405.01349v1)

**Authors**: Sihui Dai, Chong Xiang, Tong Wu, Prateek Mittal

**Abstract**: Current research on defending against adversarial examples focuses primarily on achieving robustness against a single attack type such as $\ell_2$ or $\ell_{\infty}$-bounded attacks. However, the space of possible perturbations is much larger and currently cannot be modeled by a single attack type. The discrepancy between the focus of current defenses and the space of attacks of interest calls to question the practicality of existing defenses and the reliability of their evaluation. In this position paper, we argue that the research community should look beyond single attack robustness, and we draw attention to three potential directions involving robustness against multiple attacks: simultaneous multiattack robustness, unforeseen attack robustness, and a newly defined problem setting which we call continual adaptive robustness. We provide a unified framework which rigorously defines these problem settings, synthesize existing research in these fields, and outline open directions. We hope that our position paper inspires more research in simultaneous multiattack, unforeseen attack, and continual adaptive robustness.

摘要: 目前关于防御恶意攻击的研究主要集中在对单一攻击类型的健壮性上，例如$\ell_2$或$\ell_{\infty}$-bound攻击。然而，可能的扰动空间要大得多，目前不能用单一的攻击类型来建模。当前防御的重点与感兴趣的攻击的空间之间的差异要求对现有防御的实用性及其评估的可靠性提出质疑。在这份立场文件中，我们认为研究界应该超越单一攻击的稳健性，并提请注意涉及对多个攻击的稳健性的三个潜在方向：同时多攻击稳健性、不可预见的攻击稳健性以及我们称为连续自适应稳健性的新定义的问题设置。我们提供了一个统一的框架，严格定义了这些问题设置，综合了这些领域的现有研究，并概述了开放的方向。我们希望我们的立场文件能启发更多关于同时多攻击、不可预见攻击和持续自适应稳健性的研究。



## **50. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

法学硕士自卫：通过自我检查，法学硕士知道他们被欺骗了 cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2308.07308v4) [paper-pdf](http://arxiv.org/pdf/2308.07308v4)

**Authors**: Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau

**Abstract**: Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2. The code is publicly available at https://github.com/poloclub/llm-self-defense

摘要: 大型语言模型(LLM)对于高质量的文本生成很受欢迎，但可能会产生有害的内容，即使通过强化学习与人类的价值观保持一致。对抗性提示可以绕过它们的安全措施。我们提出了LLM自卫，这是一种通过让LLM筛选诱导响应来防御这些攻击的简单方法。我们的方法不需要任何微调、输入预处理或迭代输出生成。相反，我们将生成的内容合并到预定义的提示中，并使用LLM的另一个实例来分析文本并预测它是否有害。我们在GPT 3.5和Llama 2上测试了LLM自卫，这两种当前最著名的LLM针对各种类型的攻击，例如对提示的强制诱导肯定反应和提示工程攻击。值得注意的是，LLm自卫成功地使用GPT3.5和Llama 2将攻击成功率降低到几乎为0。代码可在https://github.com/poloclub/llm-self-defense上公开获得



