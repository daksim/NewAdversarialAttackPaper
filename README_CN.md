# Latest Adversarial Attack Papers
**update at 2023-11-18 10:50:43**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Differentiable JPEG: The Devil is in the Details**

与众不同的JPEG：魔鬼在细节中 cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2309.06978v2) [paper-pdf](http://arxiv.org/pdf/2309.06978v2)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.

摘要: JPEG仍然是应用最广泛的有损图像编码方法之一。然而，JPEG的不可微特性限制了其在深度学习管道中的应用。为了解决这个问题，最近已经提出了几种JPEG的可微近似。本文对现有的DIFF进行了全面的回顾。JPEG处理并确定了以前方法遗漏的关键细节。为此，我们提出了一个新颖的Diff。JPEG方法，克服了以前的限制。我们的方法是可微的W.r.t。输入图像、JPEG质量、量化表和颜色转换参数。我们评估了DIFF的向前和向后性能。JPEG方法与现有方法的对比。此外，还进行了广泛的消融，以评估关键的设计选择。我们提议的不同之处。JPEG与(Non-Diff.)参考实现最好，大大超过了最近最好的差异。平均接近3.47美元分贝(PSNR)。对于强压缩率，我们甚至可以将PSNR提高9.51美元分贝。强大的对抗性攻击结果是由我们的差异产生的。JPEG格式，演示了有效的渐变近似。我们的代码可以在https://github.com/necla-ml/Diff-JPEG.上找到



## **2. Towards more Practical Threat Models in Artificial Intelligence Security**

人工智能安全中更实用的威胁模型 cs.CR

18 pages, 4 figures, 7 tables, under submission

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09994v1) [paper-pdf](http://arxiv.org/pdf/2311.09994v1)

**Authors**: Kathrin Grosse, Lukas Bieringer, Tarek Richard Besold, Alexandre Alahi

**Abstract**: Recent works have identified a gap between research and practice in artificial intelligence security: threats studied in academia do not always reflect the practical use and security risks of AI. For example, while models are often studied in isolation, they form part of larger ML pipelines in practice. Recent works also brought forward that adversarial manipulations introduced by academic attacks are impractical. We take a first step towards describing the full extent of this disparity. To this end, we revisit the threat models of the six most studied attacks in AI security research and match them to AI usage in practice via a survey with \textbf{271} industrial practitioners. On the one hand, we find that all existing threat models are indeed applicable. On the other hand, there are significant mismatches: research is often too generous with the attacker, assuming access to information not frequently available in real-world settings. Our paper is thus a call for action to study more practical threat models in artificial intelligence security.

摘要: 最近的研究发现了人工智能安全研究和实践之间的差距：学术界研究的威胁并不总是反映人工智能的实际使用和安全风险。例如，虽然模型通常是孤立研究的，但实际上它们构成了更大的ML管道的一部分。最近的研究也提出，学术攻击引入的对抗性操纵是不切实际的。我们朝着描述这种差距的全面程度迈出了第一步。为此，我们回顾了人工智能安全研究中研究最多的六种攻击的威胁模型，并通过对工业从业者的调查，将它们与实际应用相匹配。一方面，我们发现所有现有的威胁模型确实都是适用的。另一方面，存在严重的不匹配：研究往往对攻击者过于慷慨，假设他们可以访问现实世界中不常见的信息。因此，我们的论文呼吁采取行动，研究人工智能安全中更实用的威胁模型。



## **3. Hijacking Large Language Models via Adversarial In-Context Learning**

通过对抗性情境学习劫持大型语言模型 cs.LG

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09948v1) [paper-pdf](http://arxiv.org/pdf/2311.09948v1)

**Authors**: Yao Qiang, Xiangyu Zhou, Dongxiao Zhu

**Abstract**: In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific tasks by utilizing labeled examples as demonstrations in the precondition prompts. Despite its promising performance, ICL suffers from instability with the choice and arrangement of examples. Additionally, crafted adversarial attacks pose a notable threat to the robustness of ICL. However, existing attacks are either easy to detect, rely on external models, or lack specificity towards ICL. To address these issues, this work introduces a novel transferable attack for ICL, aiming to hijack LLMs to generate the targeted response. The proposed LLM hijacking attack leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demonstrations. Extensive experimental results on various tasks and datasets demonstrate the effectiveness of our LLM hijacking attack, resulting in a distracted attention towards adversarial tokens, consequently leading to the targeted unwanted outputs.

摘要: 情境中的学习(ICL)已经成为一种强大的范式，通过在前提提示中利用标记的例子作为示范来利用LLM来完成特定的任务。尽管ICL的表现很有希望，但它在范例的选择和排列上存在不稳定的问题。此外，精心设计的敌意攻击对ICL的健壮性构成了显著的威胁。然而，现有的攻击要么容易检测，要么依赖外部模型，要么缺乏对ICL的特异性。为了解决这些问题，本工作引入了一种针对ICL的新的可转移攻击，旨在劫持LLM以产生有针对性的响应。提出的LLM劫持攻击利用一种基于梯度的快速搜索方法来学习并将不可察觉的对抗性后缀添加到上下文演示中。在各种任务和数据集上的大量实验结果证明了LLM劫持攻击的有效性，导致人们将注意力分散到对抗性令牌上，从而导致目标不想要的输出。



## **4. Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-start**

低水平收缩的两层优化：无热启动的最优样本复杂性 stat.ML

Corrected Remark 18 + other small edits. Code at  https://github.com/CSML-IIT-UCL/bioptexps

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2202.03397v4) [paper-pdf](http://arxiv.org/pdf/2202.03397v4)

**Authors**: Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

**Abstract**: We analyse a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, equilibrium models, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower-level problem, i.e.~they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. However, there are situations, e.g., meta learning and equilibrium models, in which the warm-start procedure is not well-suited or ineffective. In this work we show that without warm-start, it is still possible to achieve order-wise (near) optimal sample complexity. In particular, we propose a simple method which uses (stochastic) fixed point iterations at the lower-level and projected inexact gradient descent at the upper-level, that reaches an $\epsilon$-stationary point using $O(\epsilon^{-2})$ and $\tilde{O}(\epsilon^{-1})$ samples for the stochastic and the deterministic setting, respectively. Finally, compared to methods using warm-start, our approach yields a simpler analysis that does not need to study the coupled interactions between the upper-level and lower-level iterates.

摘要: 我们分析了一类一般的两层问题，其中上层问题在于光滑目标函数的最小化，下层问题是寻找光滑压缩映射的不动点。这类问题包括元学习、均衡模型、超参数优化和数据中毒攻击等实例。最近的一些工作已经提出了暖启动下层问题的算法，即它们使用先前的下层近似解作为下层求解器的起点。这种热启动过程允许人们在随机和确定性设置下改善样本复杂性，在某些情况下实现顺序最优的样本复杂性。然而，在一些情况下，例如元学习和平衡模型，热启动程序不是很适合或无效的。在这项工作中，我们证明了在没有热启动的情况下，仍然有可能达到阶次(接近)最优的样本复杂性。特别地，我们提出了一种简单的方法，它在下层使用(随机)不动点迭代，在上层使用投影的不精确梯度下降，在随机和确定设置下分别使用$O(epsilon^{-2})$和$tilde{O}(epsilon^{-1})$样本达到$-epsilon$-固定点。最后，与使用热启动的方法相比，我们的方法产生了更简单的分析，不需要研究上层和下层迭代之间的耦合作用。



## **5. Breaking Boundaries: Balancing Performance and Robustness in Deep Wireless Traffic Forecasting**

打破界限：深度无线流量预测中的性能和稳健性平衡 cs.LG

12 pages, 2 figures, 5 tables

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09790v1) [paper-pdf](http://arxiv.org/pdf/2311.09790v1)

**Authors**: Ilbert Romain, V. Hoang Thai, Zhang Zonghua, Palpanas Themis

**Abstract**: Balancing the trade-off between accuracy and robustness is a long-standing challenge in time series forecasting. While most of existing robust algorithms have achieved certain suboptimal performance on clean data, sustaining the same performance level in the presence of data perturbations remains extremely hard. % In this paper, we study a wide array of perturbation scenarios and propose novel defense mechanisms against adversarial attacks using real-world telecom data. We compare our strategy against two existing adversarial training algorithms under a range of maximal allowed perturbations, defined using $\ell_{\infty}$-norm, $\in [0.1,0.4]$. % Our findings reveal that our hybrid strategy, which is composed of a classifier to detect adversarial examples, a denoiser to eliminate noise from the perturbed data samples, and a standard forecaster, achieves the best performance on both clean and perturbed data. % Our optimal model can retain up to $92.02\%$ the performance of the original forecasting model in terms of Mean Squared Error (MSE) on clean data, while being more robust than the standard adversarially trained models on perturbed data. Its MSE is 2.71$\times$ and 2.51$\times$ lower than those of comparing methods on normal and perturbed data, respectively. In addition, the components of our models can be trained in parallel, resulting in better computational efficiency. % Our results indicate that we can optimally balance the trade-off between the performance and robustness of forecasting models by improving the classifier and denoiser, even in the presence of sophisticated and destructive poisoning attacks.

摘要: 在精度和稳健性之间权衡是时间序列预测中的一个长期挑战。虽然大多数现有的稳健算法在干净的数据上已经取得了一定的次优性能，但在存在数据扰动的情况下保持相同的性能水平仍然是非常困难的。在本文中，我们研究了广泛的扰动场景，并利用真实世界的电信数据提出了新的对抗攻击的防御机制。在最大允许扰动的范围内，我们将我们的策略与两种已有的对抗性训练算法进行了比较，这些扰动是由[0.1，0.4]$中的$\ell_{inty}$-Norm，$\定义的。我们的研究结果表明，我们的混合策略，由用于检测敌意示例的分类器、用于消除扰动数据样本中的噪声的去噪器和标准预测器组成，在清洁和扰动数据上都取得了最好的性能。我们的最优模型在对干净数据的均方误差(MSE)方面可以保持原始预测模型高达92.02美元的性能，同时比标准的对抗性训练的模型对扰动数据的预测更加稳健。其均方根误差分别比正常数据和扰动数据的比较方法低2.71倍和2.51倍。此外，我们的模型的组件可以并行训练，从而产生更好的计算效率。%我们的结果表明，即使在存在复杂和破坏性的中毒攻击的情况下，我们也可以通过改进分类器和去噪器来最佳地平衡预测模型的性能和稳健性。



## **6. On the Exploitability of Reinforcement Learning with Human Feedback for Large Language Models**

人工反馈强化学习在大型语言模型中的可开发性 cs.AI

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09641v1) [paper-pdf](http://arxiv.org/pdf/2311.09641v1)

**Authors**: Jiongxiao Wang, Junlin Wu, Muhao Chen, Yevgeniy Vorobeychik, Chaowei Xiao

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) is a methodology designed to align Large Language Models (LLMs) with human preferences, playing an important role in LLMs alignment. Despite its advantages, RLHF relies on human annotators to rank the text, which can introduce potential security vulnerabilities if any adversarial annotator (i.e., attackers) manipulates the ranking score by up-ranking any malicious text to steer the LLM adversarially. To assess the red-teaming of RLHF against human preference data poisoning, we propose RankPoison, a poisoning attack method on candidates' selection of preference rank flipping to reach certain malicious behaviors (e.g., generating longer sequences, which can increase the computational cost). With poisoned dataset generated by RankPoison, we can perform poisoning attacks on LLMs to generate longer tokens without hurting the original safety alignment performance. Moreover, applying RankPoison, we also successfully implement a backdoor attack where LLMs can generate longer answers under questions with the trigger word. Our findings highlight critical security challenges in RLHF, underscoring the necessity for more robust alignment methods for LLMs.

摘要: 带人反馈的强化学习(RLHF)是一种将大语言模型与人的偏好相匹配的方法，在大语言模型对齐中起着重要作用。尽管RLHF有其优势，但它依靠人工注释者对文本进行排名，如果任何敌意注释者(即攻击者)通过对任何恶意文本进行排名来操纵排名分数，从而对LLM进行敌意操作，这可能会引入潜在的安全漏洞。为了评估RLHF的红团队对抗人类偏好数据中毒的能力，我们提出了一种毒化攻击方法RankPoison，该方法针对候选者选择偏好翻转来达到某些恶意行为(例如，生成更长的序列，这会增加计算成本)。利用RankPoison生成的有毒数据集，我们可以在不损害原始安全对齐性能的情况下，对LLM进行中毒攻击，生成更长的令牌。此外，应用RankPoison，我们还成功地实现了一个后门攻击，在带有触发词的问题下，LLMS可以生成更长的答案。我们的发现突出了RLHF中的关键安全挑战，强调了对LLM采用更强大的比对方法的必要性。



## **7. HAL 9000: Skynet's Risk Manager**

HAL 9000：天网的风险经理 cs.CR

18 pages, 9 figures

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09449v1) [paper-pdf](http://arxiv.org/pdf/2311.09449v1)

**Authors**: Tadeu Freitas, Mário Neto, Inês Dutra, João Soares, Manuel Correia, Rolando Martins

**Abstract**: Intrusion Tolerant Systems (ITSs) are a necessary component for cyber-services/infrastructures. Additionally, as cyberattacks follow a multi-domain attack surface, a similar defensive approach should be applied, namely, the use of an evolving multi-disciplinary solution that combines ITS, cybersecurity and Artificial Intelligence (AI). With the increased popularity of AI solutions, due to Big Data use-case scenarios and decision support and automation scenarios, new opportunities to apply Machine Learning (ML) algorithms have emerged, namely ITS empowerment. Using ML algorithms, an ITS can augment its intrusion tolerance capability, by learning from previous attacks and from known vulnerabilities. As such, this work's contribution is twofold: (1) an ITS architecture (Skynet) based on the state-of-the-art and incorporates new components to increase its intrusion tolerance capability and its adaptability to new adversaries; (2) an improved Risk Manager design that leverages AI to improve ITSs by automatically assessing OS risks to intrusions, and advise with safer configurations. One of the reasons that intrusions are successful is due to bad configurations or slow adaptability to new threats. This can be caused by the dependency that systems have for human intervention. One of the characteristics in Skynet and HAL 9000 design is the removal of human intervention. Being fully automatized lowers the chance of successful intrusions caused by human error. Our experiments using Skynet, shows that HAL is able to choose 15% safer configurations than the state-of-the-art risk manager.

摘要: 入侵容忍系统(ITSS)是网络服务/基础设施的重要组成部分。此外，由于网络攻击遵循多域攻击表面，应采用类似的防御方法，即使用结合智能交通系统、网络安全和人工智能(AI)的不断发展的多学科解决方案。随着人工智能解决方案的日益流行，由于大数据用例场景以及决策支持和自动化场景，出现了应用机器学习(ML)算法的新机会，即其赋能。使用ML算法，智能交通系统可以通过学习以前的攻击和已知的漏洞来增强其入侵容忍能力。因此，这项工作的贡献有两个：(1)基于最先进的ITS架构(Skynet)，并采用新组件，以提高其入侵容忍能力和对新对手的适应性；(2)改进的Risk Manager设计，通过自动评估入侵的操作系统风险，并为更安全的配置提供建议，利用人工智能来改进ITSS。入侵成功的原因之一是由于配置不佳或对新威胁的适应速度较慢。这可能是由于系统对人为干预的依赖造成的。天网和HAL 9000设计的特点之一是消除了人为干预。完全自动化降低了人为错误导致的成功入侵的机会。我们使用Skynet进行的实验表明，HAL能够选择比最先进的风险管理器更安全15%的配置。



## **8. How Trustworthy are Open-Source LLMs? An Assessment under Malicious Demonstrations Shows their Vulnerabilities**

开源LLM有多值得信赖？恶意示威下的评估显示其脆弱性 cs.CL

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09447v1) [paper-pdf](http://arxiv.org/pdf/2311.09447v1)

**Authors**: Lingbo Mo, Boshi Wang, Muhao Chen, Huan Sun

**Abstract**: The rapid progress in open-source Large Language Models (LLMs) is significantly driving AI development forward. However, there is still a limited understanding of their trustworthiness. Deploying these models at scale without sufficient trustworthiness can pose significant risks, highlighting the need to uncover these issues promptly. In this work, we conduct an assessment of open-source LLMs on trustworthiness, scrutinizing them across eight different aspects including toxicity, stereotypes, ethics, hallucination, fairness, sycophancy, privacy, and robustness against adversarial demonstrations. We propose an enhanced Chain of Utterances-based (CoU) prompting strategy by incorporating meticulously crafted malicious demonstrations for trustworthiness attack. Our extensive experiments encompass recent and representative series of open-source LLMs, including Vicuna, MPT, Falcon, Mistral, and Llama 2. The empirical outcomes underscore the efficacy of our attack strategy across diverse aspects. More interestingly, our result analysis reveals that models with superior performance in general NLP tasks do not always have greater trustworthiness; in fact, larger models can be more vulnerable to attacks. Additionally, models that have undergone instruction tuning, focusing on instruction following, tend to be more susceptible, although fine-tuning LLMs for safety alignment proves effective in mitigating adversarial trustworthiness attacks.

摘要: 开源大型语言模型(LLM)的快速发展显著地推动了人工智能的发展。然而，人们对他们的可信性仍知之甚少。在缺乏足够可信度的情况下大规模部署这些模型可能会带来重大风险，这突显了迅速发现这些问题的必要性。在这项工作中，我们对开源LLM的可信性进行了评估，从八个不同的方面对它们进行了仔细的审查，包括毒性、刻板印象、道德、幻觉、公平性、奉承、隐私和对对手演示的健壮性。我们提出了一种增强的基于话语链(CUU)的提示策略，该策略结合了精心制作的恶意演示来进行可信度攻击。我们广泛的实验涵盖了最近一系列具有代表性的开源LLM，包括Vicuna、MPT、Falcon、Mistral和Llama 2。经验结果强调了我们攻击策略在不同方面的有效性。更有趣的是，我们的结果分析显示，在一般NLP任务中性能优越的模型并不总是具有更大的可信度；事实上，较大的模型可能更容易受到攻击。此外，经过指令调整、专注于指令遵循的模型往往更容易受到影响，尽管针对安全对齐的微调LLM被证明在减轻对手信任攻击方面是有效的。



## **9. Beyond Detection: Unveiling Fairness Vulnerabilities in Abusive Language Models**

超越检测：揭开辱骂语言模型中的公平漏洞 cs.CL

Under review

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09428v1) [paper-pdf](http://arxiv.org/pdf/2311.09428v1)

**Authors**: Yueqing Liang, Lu Cheng, Ali Payani, Kai Shu

**Abstract**: This work investigates the potential of undermining both fairness and detection performance in abusive language detection. In a dynamic and complex digital world, it is crucial to investigate the vulnerabilities of these detection models to adversarial fairness attacks to improve their fairness robustness. We propose a simple yet effective framework FABLE that leverages backdoor attacks as they allow targeted control over the fairness and detection performance. FABLE explores three types of trigger designs (i.e., rare, artificial, and natural triggers) and novel sampling strategies. Specifically, the adversary can inject triggers into samples in the minority group with the favored outcome (i.e., ``non-abusive'') and flip their labels to the unfavored outcome, i.e., ``abusive''. Experiments on benchmark datasets demonstrate the effectiveness of FABLE attacking fairness and utility in abusive language detection.

摘要: 这项工作调查了在辱骂语言检测中同时破坏公平性和检测性能的可能性。在动态和复杂的数字世界中，研究这些检测模型对敌意公平攻击的脆弱性，以提高其公平性健壮性是至关重要的。我们提出了一个简单而有效的框架寓言，它利用了后门攻击，因为它们允许对公平性和检测性能进行有针对性的控制。Fable探索了三种类型的触发器设计(即罕见的、人工的和自然的触发器)和新颖的抽样策略。具体地说，敌手可以向少数群体中具有有利结果的样本注入触发器(即“非滥用”)，并将其标签翻转到不利结果，即“滥用”。在基准数据集上的实验证明了寓言攻击的有效性、公平性和实用性。



## **10. UMD: Unsupervised Model Detection for X2X Backdoor Attacks**

UMD：X2X后门攻击的无监督模型检测 cs.LG

Proceedings of the 40th International Conference on Machine Learning

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2305.18651v4) [paper-pdf](http://arxiv.org/pdf/2305.18651v4)

**Authors**: Zhen Xiang, Zidi Xiong, Bo Li

**Abstract**: Backdoor (Trojan) attack is a common threat to deep neural networks, where samples from one or more source classes embedded with a backdoor trigger will be misclassified to adversarial target classes. Existing methods for detecting whether a classifier is backdoor attacked are mostly designed for attacks with a single adversarial target (e.g., all-to-one attack). To the best of our knowledge, without supervision, no existing methods can effectively address the more general X2X attack with an arbitrary number of source classes, each paired with an arbitrary target class. In this paper, we propose UMD, the first Unsupervised Model Detection method that effectively detects X2X backdoor attacks via a joint inference of the adversarial (source, target) class pairs. In particular, we first define a novel transferability statistic to measure and select a subset of putative backdoor class pairs based on a proposed clustering approach. Then, these selected class pairs are jointly assessed based on an aggregation of their reverse-engineered trigger size for detection inference, using a robust and unsupervised anomaly detector we proposed. We conduct comprehensive evaluations on CIFAR-10, GTSRB, and Imagenette dataset, and show that our unsupervised UMD outperforms SOTA detectors (even with supervision) by 17%, 4%, and 8%, respectively, in terms of the detection accuracy against diverse X2X attacks. We also show the strong detection performance of UMD against several strong adaptive attacks.

摘要: 后门(特洛伊木马)攻击是深度神经网络的常见威胁，来自嵌入后门触发器的一个或多个源类的样本将被错误分类为对抗性目标类。现有的检测分类器是否被后门攻击的方法大多是针对单个敌对目标的攻击而设计的(例如，All-to-One攻击)。就我们所知，在没有监督的情况下，没有任何现有方法可以有效地应对具有任意数量的源类的更通用的X2X攻击，每个源类都与任意的目标类配对。在本文中，我们提出了第一种无监督模型检测方法UMD，它通过联合推理对手(源、目标)类对来有效地检测X2X后门攻击。特别是，我们首先定义了一种新的可转移性统计量来度量和选择基于所提出的聚类方法的假定的后门类对的子集。然后，使用我们提出的健壮和无监督的异常检测器，基于它们的反向工程触发大小的聚集来联合评估这些选择的类对以用于检测推理。我们在CIFAR-10、GTSRB和Imagenette数据集上进行了综合评估，结果表明，在对各种X2X攻击的检测准确率方面，我们的无监督UMD分别比SOTA检测器(即使有监督)提高了17%、4%和8%。我们还展示了UMD对几种强自适应攻击的强检测性能。



## **11. Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD**

梯度看起来很相似：DP-SGD的敏感度经常被高估 cs.LG

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2307.00310v2) [paper-pdf](http://arxiv.org/pdf/2307.00310v2)

**Authors**: Anvith Thudi, Hengrui Jia, Casey Meehan, Ilia Shumailov, Nicolas Papernot

**Abstract**: Differentially private stochastic gradient descent (DP-SGD) is the canonical approach to private deep learning. While the current privacy analysis of DP-SGD is known to be tight in some settings, several empirical results suggest that models trained on common benchmark datasets leak significantly less privacy for many datapoints. Yet, despite past attempts, a rigorous explanation for why this is the case has not been reached. Is it because there exist tighter privacy upper bounds when restricted to these dataset settings, or are our attacks not strong enough for certain datapoints? In this paper, we provide the first per-instance (i.e., ``data-dependent") DP analysis of DP-SGD. Our analysis captures the intuition that points with similar neighbors in the dataset enjoy better data-dependent privacy than outliers. Formally, this is done by modifying the per-step privacy analysis of DP-SGD to introduce a dependence on the distribution of model updates computed from a training dataset. We further develop a new composition theorem to effectively use this new per-step analysis to reason about an entire training run. Put all together, our evaluation shows that this novel DP-SGD analysis allows us to now formally show that DP-SGD leaks significantly less privacy for many datapoints (when trained on common benchmarks) than the current data-independent guarantee. This implies privacy attacks will necessarily fail against many datapoints if the adversary does not have sufficient control over the possible training datasets.

摘要: 差分私人随机梯度下降(DP-SGD)是私人深度学习的典型方法。虽然目前DP-SGD的隐私分析在某些情况下是严格的，但一些经验结果表明，在公共基准数据集上训练的模型对于许多数据点来说泄露的隐私要少得多。然而，尽管过去曾尝试过，但对于为什么会出现这种情况，还没有达成一个严格的解释。是因为限制到这些数据集设置时存在更严格的隐私上限，还是因为我们的攻击对某些数据点不够强大？在这篇文章中，我们提供了DP-SGD的第一个逐实例(即“数据依赖”)DP分析。我们的分析抓住了这样一种直觉，即数据集中具有相似邻居的点比离群值享有更好的数据依赖隐私。形式上，这是通过修改DP-SGD的每一步隐私分析来实现的，以引入对从训练数据集计算的模型更新的分布的依赖。我们进一步开发了一个新的合成定理，以有效地使用这个新的逐步分析来推理整个训练运行。综上所述，我们的评估表明，这种新颖的DP-SGD分析允许我们现在正式地表明，DP-SGD对于许多数据点(当根据公共基准进行训练时)的隐私泄露显著低于当前的数据独立保证。这意味着如果对手对可能的训练数据集没有足够的控制，针对许多数据点的隐私攻击必然会失败。



## **12. Frontier Language Models are not Robust to Adversarial Arithmetic, or "What do I need to say so you agree 2+2=5?**

前沿语言模型对对抗性算术或“我需要说什么才能让你同意2+2=5？” cs.CL

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.07587v2) [paper-pdf](http://arxiv.org/pdf/2311.07587v2)

**Authors**: C. Daniel Freeman, Laura Culp, Aaron Parisi, Maxwell L Bileschi, Gamaleldin F Elsayed, Alex Rizkowsky, Isabelle Simpson, Alex Alemi, Azade Nova, Ben Adlam, Bernd Bohnet, Gaurav Mishra, Hanie Sedghi, Igor Mordatch, Izzeddin Gur, Jaehoon Lee, JD Co-Reyes, Jeffrey Pennington, Kelvin Xu, Kevin Swersky, Kshiteej Mahajan, Lechao Xiao, Rosanne Liu, Simon Kornblith, Noah Constant, Peter J. Liu, Roman Novak, Yundi Qian, Noah Fiedel, Jascha Sohl-Dickstein

**Abstract**: We introduce and study the problem of adversarial arithmetic, which provides a simple yet challenging testbed for language model alignment. This problem is comprised of arithmetic questions posed in natural language, with an arbitrary adversarial string inserted before the question is complete. Even in the simple setting of 1-digit addition problems, it is easy to find adversarial prompts that make all tested models (including PaLM2, GPT4, Claude2) misbehave, and even to steer models to a particular wrong answer. We additionally provide a simple algorithm for finding successful attacks by querying those same models, which we name "prompt inversion rejection sampling" (PIRS). We finally show that models can be partially hardened against these attacks via reinforcement learning and via agentic constitutional loops. However, we were not able to make a language model fully robust against adversarial arithmetic attacks.

摘要: 我们引入并研究了对抗性算法问题，为语言模型对齐提供了一个简单但具有挑战性的试验台。这个问题由自然语言提出的算术问题组成，在问题完成之前插入一个任意的敌意字符串。即使是在1位数加法问题的简单设置中，也很容易找到令所有测试模型(包括Palm2、GPT4、Claude2)表现不佳的对抗性提示，甚至会将模型引导到特定的错误答案。此外，我们还提供了一个简单的算法，用于通过查询这些相同的模型来发现成功的攻击，我们将其命名为“即时反转拒绝采样”(PIRS)。最后，我们证明了模型可以通过强化学习和代理构成环来部分加强对这些攻击的抵抗。然而，我们不能使语言模型对敌意算术攻击完全健壮。



## **13. Jailbreaking GPT-4V via Self-Adversarial Attacks with System Prompts**

通过系统攻击的自对抗攻击越狱GPT-4V cs.CR

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09127v1) [paper-pdf](http://arxiv.org/pdf/2311.09127v1)

**Authors**: Yuanwei Wu, Xiang Li, Yixin Liu, Pan Zhou, Lichao Sun

**Abstract**: Existing work on jailbreak Multimodal Large Language Models (MLLMs) has focused primarily on adversarial examples in model inputs, with less attention to vulnerabilities in model APIs. To fill the research gap, we carry out the following work: 1) We discover a system prompt leakage vulnerability in GPT-4V. Through carefully designed dialogue, we successfully steal the internal system prompts of GPT-4V. This finding indicates potential exploitable security risks in MLLMs; 2)Based on the acquired system prompts, we propose a novel MLLM jailbreaking attack method termed SASP (Self-Adversarial Attack via System Prompt). By employing GPT-4 as a red teaming tool against itself, we aim to search for potential jailbreak prompts leveraging stolen system prompts. Furthermore, in pursuit of better performance, we also add human modification based on GPT-4's analysis, which further improves the attack success rate to 98.7\%; 3) We evaluated the effect of modifying system prompts to defend against jailbreaking attacks. Results show that appropriately designed system prompts can significantly reduce jailbreak success rates. Overall, our work provides new insights into enhancing MLLM security, demonstrating the important role of system prompts in jailbreaking, which could be leveraged to greatly facilitate jailbreak success rates while also holding the potential for defending against jailbreaks.

摘要: 现有关于越狱多模式大型语言模型(MLLMS)的工作主要集中在模型输入中的对抗性示例，对模型API中的漏洞关注较少。为了填补这一研究空白，我们开展了以下工作：1)在GPT-4V中发现了一个系统即时泄漏漏洞。通过精心设计的对话，我们成功窃取了GPT-4V的内部系统提示。2)基于获得的系统提示，提出了一种新的基于系统提示的MLLM越狱攻击方法SASP(Self-Aversarial Attack by System Prompt)。通过使用GPT-4作为针对自己的红色团队工具，我们的目标是利用被盗的系统提示来搜索潜在的越狱提示。此外，为了追求更好的性能，我们还在GPT-4的S分析的基础上增加了人工修改，进一步将攻击成功率提高到98.7%。3)评估了修改系统提示对越狱攻击的防御效果。结果表明，设计适当的系统提示可以显著降低越狱成功率。总体而言，我们的工作为加强MLLM安全提供了新的见解，展示了系统提示在越狱中的重要作用，这可以被用来极大地提高越狱成功率，同时也保持了防御越狱的潜力。



## **14. Fast Certification of Vision-Language Models Using Incremental Randomized Smoothing**

基于增量随机平滑的视觉语言模型快速认证 cs.CV

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09024v1) [paper-pdf](http://arxiv.org/pdf/2311.09024v1)

**Authors**: A K Nirala, A Joshi, C Hegde, S Sarkar

**Abstract**: A key benefit of deep vision-language models such as CLIP is that they enable zero-shot open vocabulary classification; the user has the ability to define novel class labels via natural language prompts at inference time. However, while CLIP-based zero-shot classifiers have demonstrated competitive performance across a range of domain shifts, they remain highly vulnerable to adversarial attacks. Therefore, ensuring the robustness of such models is crucial for their reliable deployment in the wild.   In this work, we introduce Open Vocabulary Certification (OVC), a fast certification method designed for open-vocabulary models like CLIP via randomized smoothing techniques. Given a base "training" set of prompts and their corresponding certified CLIP classifiers, OVC relies on the observation that a classifier with a novel prompt can be viewed as a perturbed version of nearby classifiers in the base training set. Therefore, OVC can rapidly certify the novel classifier using a variation of incremental randomized smoothing. By using a caching trick, we achieve approximately two orders of magnitude acceleration in the certification process for novel prompts. To achieve further (heuristic) speedups, OVC approximates the embedding space at a given input using a multivariate normal distribution bypassing the need for sampling via forward passes through the vision backbone. We demonstrate the effectiveness of OVC on through experimental evaluation using multiple vision-language backbones on the CIFAR-10 and ImageNet test datasets.

摘要: 深度视觉语言模型（如CLIP）的一个关键好处是它们能够实现零触发开放词汇分类;用户能够在推理时通过自然语言提示定义新的类标签。然而，尽管基于CLIP的零触发分类器在一系列领域转移中表现出了竞争力，但它们仍然非常容易受到对抗性攻击。因此，确保这些模型的鲁棒性对于它们在野外的可靠部署至关重要。   在这项工作中，我们引入了开放词汇认证（OVC），这是一种通过随机平滑技术为CLIP等开放词汇模型设计的快速认证方法。给定一个基本的“训练”提示集及其相应的认证CLIP分类器，OVC依赖于这样的观察，即具有新提示的分类器可以被视为基本训练集中附近分类器的扰动版本。因此，OVC可以使用增量随机平滑的变化来快速地认证新的分类器。通过使用缓存技巧，我们实现了大约两个数量级的加速认证过程中的新提示。为了实现进一步的（启发式）加速，OVC使用多变量正态分布来近似给定输入的嵌入空间，从而绕过了通过视觉骨干的前向传递进行采样的需要。我们通过在CIFAR-10和ImageNet测试数据集上使用多个视觉语言主干进行实验评估，证明了OVC的有效性。



## **15. Adversarial Attacks to Reward Machine-based Reinforcement Learning**

奖励基于机器的强化学习的对抗性攻击 cs.LG

Thesis Supervisor: Prof. Federico Cerutti (Universit\`a degli Studi  di Brescia, IT)

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09014v1) [paper-pdf](http://arxiv.org/pdf/2311.09014v1)

**Authors**: Lorenzo Nodari

**Abstract**: In recent years, Reward Machines (RMs) have stood out as a simple yet effective automata-based formalism for exposing and exploiting task structure in reinforcement learning settings. Despite their relevance, little to no attention has been directed to the study of their security implications and robustness to adversarial scenarios, likely due to their recent appearance in the literature. With my thesis, I aim to provide the first analysis of the security of RM-based reinforcement learning techniques, with the hope of motivating further research in the field, and I propose and evaluate a novel class of attacks on RM-based techniques: blinding attacks.

摘要: 近年来，奖励机器(RMS)作为一种简单而有效的基于自动机的形式化方法已经脱颖而出，用于揭示和开发强化学习环境中的任务结构。尽管它们具有相关性，但很少或根本没有人关注它们的安全影响和对抗情景的稳健性，这可能是因为它们最近出现在文献中。本文旨在对基于RM的强化学习技术的安全性进行首次分析，以期推动该领域的进一步研究，并提出并评估了一类针对基于RM的强化学习技术的新型攻击：盲攻击。



## **16. Improving the Accuracy-Robustness Trade-Off of Classifiers via Adaptive Smoothing**

用自适应平滑提高分类器的精度和稳健性 cs.LG

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2301.12554v3) [paper-pdf](http://arxiv.org/pdf/2301.12554v3)

**Authors**: Yatong Bai, Brendon G. Anderson, Aerin Kim, Somayeh Sojoudi

**Abstract**: While prior research has proposed a plethora of methods that build neural classifiers robust against adversarial robustness, practitioners are still reluctant to adopt them due to their unacceptably severe clean accuracy penalties. This paper significantly alleviates this accuracy-robustness trade-off by mixing the output probabilities of a standard classifier and a robust classifier, where the standard network is optimized for clean accuracy and is not robust in general. We show that the robust base classifier's confidence difference for correct and incorrect examples is the key to this improvement. In addition to providing intuitions and empirical evidence, we theoretically certify the robustness of the mixed classifier under realistic assumptions. Furthermore, we adapt an adversarial input detector into a mixing network that adaptively adjusts the mixture of the two base models, further reducing the accuracy penalty of achieving robustness. The proposed flexible method, termed "adaptive smoothing", can work in conjunction with existing or even future methods that improve clean accuracy, robustness, or adversary detection. Our empirical evaluation considers strong attack methods, including AutoAttack and adaptive attack. On the CIFAR-100 dataset, our method achieves an 85.21% clean accuracy while maintaining a 38.72% $\ell_\infty$-AutoAttacked ($\epsilon = 8/255$) accuracy, becoming the second most robust method on the RobustBench CIFAR-100 benchmark as of submission, while improving the clean accuracy by ten percentage points compared with all listed models. The code that implements our method is available at https://github.com/Bai-YT/AdaptiveSmoothing.

摘要: 虽然先前的研究已经提出了太多的方法来构建稳健的神经分类器来对抗对手的健壮性，但实践者仍然不愿采用它们，因为它们具有不可接受的严重的干净准确性惩罚。本文通过混合标准分类器和稳健分类器的输出概率显著缓解了这种精度与稳健性的权衡，其中标准网络针对干净的精度进行了优化，而通常不是稳健的。研究表明，稳健的基分类器对正确样本和错误样本的置信度差异是这一改进的关键。除了提供直觉和经验证据外，我们还从理论上证明了混合分类器在现实假设下的稳健性。此外，我们将对抗性输入检测器引入混合网络，该混合网络自适应地调整两个基本模型的混合，从而进一步降低了实现稳健性的精度损失。这一灵活的方法被称为“自适应平滑”，可以与现有甚至未来的方法结合使用，以提高干净的准确性、健壮性或敌手检测。我们的经验评估考虑了强攻击方法，包括AutoAttack和自适应攻击。在CIFAR-100数据集上，我们的方法实现了85.21%的清洁准确率，同时保持了38.72%的$\ELL_\INFTY$-AutoAttaced($\epsilon=8/255$)精度，成为截至提交时在RobustBuchCIFAR-100基准上第二健壮的方法，同时与所有列出的模型相比，清洁准确率提高了10个百分点。实现我们方法的代码可以在https://github.com/Bai-YT/AdaptiveSmoothing.上找到



## **17. On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers**

人工智能分类器对抗健壮性度量的存在性、唯一性和可伸缩性 stat.ML

16 pages, 3 figures

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2310.14421v4) [paper-pdf](http://arxiv.org/pdf/2310.14421v4)

**Authors**: Illia Horenko

**Abstract**: Simply-verifiable mathematical conditions for existence, uniqueness and explicit analytical computation of minimal adversarial paths (MAP) and minimal adversarial distances (MAD) for (locally) uniquely-invertible classifiers, for generalized linear models (GLM), and for entropic AI (EAI) are formulated and proven. Practical computation of MAP and MAD, their comparison and interpretations for various classes of AI tools (for neuronal networks, boosted random forests, GLM and EAI) are demonstrated on the common synthetic benchmarks: on a double Swiss roll spiral and its extensions, as well as on the two biomedical data problems (for the health insurance claim predictions, and for the heart attack lethality classification). On biomedical applications it is demonstrated how MAP provides unique minimal patient-specific risk-mitigating interventions in the predefined subsets of accessible control variables.

摘要: 提出并证明了(局部)唯一可逆分类器、广义线性模型(GLM)和熵人工智能(EAI)的最小对抗路径(MAP)和最小对抗距离(MAD)的存在唯一性和显式解析计算的简单可验证的数学条件.MAP和MAD的实际计算，以及它们对各种人工智能工具(用于神经元网络、增强随机森林、GLM和EAI)的比较和解释，在常见的合成基准上进行了演示：在双瑞士辊螺旋及其扩展上，以及在两个生物医学数据问题上(用于健康保险索赔预测和心脏病发作死亡分类)。在生物医学应用方面，它展示了MAP如何在可访问的控制变量的预定义子集中提供独特的、最小限度的患者特定风险缓解干预。



## **18. DALA: A Distribution-Aware LoRA-Based Adversarial Attack against Pre-trained Language Models**

Dala：一种基于分布感知LORA的针对预训练语言模型的对抗性攻击 cs.CL

First two authors contribute equally

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.08598v1) [paper-pdf](http://arxiv.org/pdf/2311.08598v1)

**Authors**: Yibo Wang, Xiangjue Dong, James Caverlee, Philip S. Yu

**Abstract**: Pre-trained language models (PLMs) that achieve success in applications are susceptible to adversarial attack methods that are capable of generating adversarial examples with minor perturbations. Although recent attack methods can achieve a relatively high attack success rate (ASR), our observation shows that the generated adversarial examples have a different data distribution compared with the original examples. Specifically, these adversarial examples exhibit lower confidence levels and higher distance to the training data distribution. As a result, they are easy to detect using very simple detection methods, diminishing the actual effectiveness of these attack methods. To solve this problem, we propose a Distribution-Aware LoRA-based Adversarial Attack (DALA) method, which considers the distribution shift of adversarial examples to improve attack effectiveness under detection methods. We further design a new evaluation metric NASR combining ASR and detection for the attack task. We conduct experiments on four widely-used datasets and validate the attack effectiveness on ASR and NASR of the adversarial examples generated by DALA on the BERT-base model and the black-box LLaMA2-7b model.

摘要: 在应用中取得成功的预先训练的语言模型(PLM)容易受到对抗性攻击方法的影响，这些方法能够以较小的扰动生成对抗性示例。虽然目前的攻击方法可以达到相对较高的攻击成功率，但我们的观察表明，生成的对抗性实例与原始实例相比具有不同的数据分布。具体地说，这些对抗性例子表现出较低的置信度和较高的训练数据分布距离。因此，使用非常简单的检测方法很容易检测到它们，从而降低了这些攻击方法的实际有效性。为了解决这一问题，我们提出了一种基于分布感知LORA的对抗性攻击方法(DALA)，该方法考虑了对抗性实例的分布偏移，以提高检测方法下的攻击效率。我们进一步设计了一种新的ASR和检测相结合的评估指标NASR，用于攻击任务。我们在四个广泛使用的数据集上进行了实验，并验证了Dala在Bert-base模型和黑盒LLaMA2-7b模型上生成的对抗性实例对ASR和Nasr的攻击有效性。



## **19. Physical Adversarial Examples for Multi-Camera Systems**

多摄像机系统的物理对抗实例 cs.CV

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.08539v1) [paper-pdf](http://arxiv.org/pdf/2311.08539v1)

**Authors**: Ana Răduţoiu, Jan-Philipp Schulze, Philip Sperl, Konstantin Böttinger

**Abstract**: Neural networks build the foundation of several intelligent systems, which, however, are known to be easily fooled by adversarial examples. Recent advances made these attacks possible even in air-gapped scenarios, where the autonomous system observes its surroundings by, e.g., a camera. We extend these ideas in our research and evaluate the robustness of multi-camera setups against such physical adversarial examples. This scenario becomes ever more important with the rise in popularity of autonomous vehicles, which fuse the information of several cameras for their driving decision. While we find that multi-camera setups provide some robustness towards past attack methods, we see that this advantage reduces when optimizing on multiple perspectives at once. We propose a novel attack method that we call Transcender-MC, where we incorporate online 3D renderings and perspective projections in the training process. Moreover, we motivate that certain data augmentation techniques can facilitate the generation of successful adversarial examples even further. Transcender-MC is 11% more effective in successfully attacking multi-camera setups than state-of-the-art methods. Our findings offer valuable insights regarding the resilience of object detection in a setup with multiple cameras and motivate the need of developing adequate defense mechanisms against them.

摘要: 神经网络为几种智能系统奠定了基础，然而，众所周知，这些系统很容易被对抗性的例子所欺骗。最近的进展使得这些攻击甚至在空气间隙的情况下也成为可能，其中自主系统通过例如以下方式观察其周围环境：一个照相机。我们在研究中扩展了这些想法，并评估了多相机设置对这种物理对抗性示例的鲁棒性。随着自动驾驶汽车的普及，这种情况变得越来越重要，自动驾驶汽车将多个摄像头的信息融合在一起，用于驾驶决策。虽然我们发现多摄像头设置对过去的攻击方法提供了一些鲁棒性，但我们看到，当同时优化多个视角时，这种优势会降低。我们提出了一种新的攻击方法，我们称之为超越MC，在那里我们将在线3D渲染和透视投影在训练过程中。此外，我们认为某些数据增强技术可以进一步促进成功对抗性示例的生成。超越者-MC在成功攻击多摄像头设置方面比最先进的方法效率高11%。我们的研究结果提供了关于多个摄像头设置中对象检测的弹性的有价值的见解，并激发了开发针对它们的适当防御机制的需求。



## **20. Alignment is not sufficient to prevent large language models from generating harmful information: A psychoanalytic perspective**

对齐不足以防止大型语言模型产生有害信息：从精神分析的角度 cs.CL

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.08487v1) [paper-pdf](http://arxiv.org/pdf/2311.08487v1)

**Authors**: Zi Yin, Wei Ding, Jia Liu

**Abstract**: Large Language Models (LLMs) are central to a multitude of applications but struggle with significant risks, notably in generating harmful content and biases. Drawing an analogy to the human psyche's conflict between evolutionary survival instincts and societal norm adherence elucidated in Freud's psychoanalysis theory, we argue that LLMs suffer a similar fundamental conflict, arising between their inherent desire for syntactic and semantic continuity, established during the pre-training phase, and the post-training alignment with human values. This conflict renders LLMs vulnerable to adversarial attacks, wherein intensifying the models' desire for continuity can circumvent alignment efforts, resulting in the generation of harmful information. Through a series of experiments, we first validated the existence of the desire for continuity in LLMs, and further devised a straightforward yet powerful technique, such as incomplete sentences, negative priming, and cognitive dissonance scenarios, to demonstrate that even advanced LLMs struggle to prevent the generation of harmful information. In summary, our study uncovers the root of LLMs' vulnerabilities to adversarial attacks, hereby questioning the efficacy of solely relying on sophisticated alignment methods, and further advocates for a new training idea that integrates modal concepts alongside traditional amodal concepts, aiming to endow LLMs with a more nuanced understanding of real-world contexts and ethical considerations.

摘要: 大型语言模型(LLM)是众多应用程序的核心，但面临着巨大的风险，特别是在生成有害内容和偏见方面。通过类比弗洛伊德精神分析理论中阐明的人类心理在进化生存本能和遵守社会规范之间的冲突，我们认为LLMS在训练前建立的对句法和语义连续性的内在愿望与训练后与人类价值观的一致性之间存在着类似的根本冲突。这种冲突使LLM容易受到对抗性攻击，其中加强模型对连续性的渴望可以绕过对齐工作，从而导致有害信息的产生。通过一系列实验，我们首先验证了LLMS中对连续性的渴望的存在，并进一步设计了一种简单而强大的技术，如不完整句子、负启动和认知不协调情景，以证明即使是高级LLMS也难以防止有害信息的产生。综上所述，我们的研究揭示了LLMS易受敌意攻击的根源，由此质疑单纯依赖复杂的对齐方法的有效性，并进一步倡导一种新的训练思想，将情态概念与传统的非模态概念相结合，旨在赋予LLMS对现实世界背景和伦理考虑的更细微的理解。



## **21. The Perception-Robustness Tradeoff in Deterministic Image Restoration**

确定性图像恢复中的感知-稳健性权衡 eess.IV

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.09253v1) [paper-pdf](http://arxiv.org/pdf/2311.09253v1)

**Authors**: Guy Ohayon, Tomer Michaeli, Michael Elad

**Abstract**: We study the behavior of deterministic methods for solving inverse problems in imaging. These methods are commonly designed to achieve two goals: (1) attaining high perceptual quality, and (2) generating reconstructions that are consistent with the measurements. We provide a rigorous proof that the better a predictor satisfies these two requirements, the larger its Lipschitz constant must be, regardless of the nature of the degradation involved. In particular, to approach perfect perceptual quality and perfect consistency, the Lipschitz constant of the model must grow to infinity. This implies that such methods are necessarily more susceptible to adversarial attacks. We demonstrate our theory on single image super-resolution algorithms, addressing both noisy and noiseless settings. We also show how this undesired behavior can be leveraged to explore the posterior distribution, thereby allowing the deterministic model to imitate stochastic methods.

摘要: 我们研究的行为确定性方法求解成像反问题。这些方法通常被设计成实现两个目标：（1）获得高感知质量，以及（2）生成与测量一致的重建。我们提供了一个严格的证明，更好的预测满足这两个要求，更大的Lipschitz常数必须，无论性质的退化。特别是，为了接近完美的感知质量和完美的一致性，模型的Lipschitz常数必须增长到无穷大。这意味着这些方法必然更容易受到对抗性攻击。我们证明了我们的理论，单图像超分辨率算法，解决噪声和无噪声的设置。我们还展示了如何利用这种不期望的行为来探索后验分布，从而允许确定性模型模仿随机方法。



## **22. Scale-MIA: A Scalable Model Inversion Attack against Secure Federated Learning via Latent Space Reconstruction**

Scale-MIA：一种基于潜在空间重构的安全联邦学习可扩展模型反转攻击 cs.LG

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.05808v2) [paper-pdf](http://arxiv.org/pdf/2311.05808v2)

**Authors**: Shanghao Shi, Ning Wang, Yang Xiao, Chaoyu Zhang, Yi Shi, Y. Thomas Hou, Wenjing Lou

**Abstract**: Federated learning is known for its capability to safeguard participants' data privacy. However, recently emerged model inversion attacks (MIAs) have shown that a malicious parameter server can reconstruct individual users' local data samples through model updates. The state-of-the-art attacks either rely on computation-intensive search-based optimization processes to recover each input batch, making scaling difficult, or they involve the malicious parameter server adding extra modules before the global model architecture, rendering the attacks too conspicuous and easily detectable.   To overcome these limitations, we propose Scale-MIA, a novel MIA capable of efficiently and accurately recovering training samples of clients from the aggregated updates, even when the system is under the protection of a robust secure aggregation protocol. Unlike existing approaches treating models as black boxes, Scale-MIA recognizes the importance of the intricate architecture and inner workings of machine learning models. It identifies the latent space as the critical layer for breaching privacy and decomposes the complex recovery task into an innovative two-step process to reduce computation complexity. The first step involves reconstructing the latent space representations (LSRs) from the aggregated model updates using a closed-form inversion mechanism, leveraging specially crafted adversarial linear layers. In the second step, the whole input batches are recovered from the LSRs by feeding them into a fine-tuned generative decoder.   We implemented Scale-MIA on multiple commonly used machine learning models and conducted comprehensive experiments across various settings. The results demonstrate that Scale-MIA achieves excellent recovery performance on different datasets, exhibiting high reconstruction rates, accuracy, and attack efficiency on a larger scale compared to state-of-the-art MIAs.

摘要: 联合学习以其保护参与者数据隐私的能力而闻名。然而，最近出现的模型反转攻击(MIA)表明，恶意参数服务器可以通过模型更新来重建个人用户的本地数据样本。最先进的攻击要么依赖于基于搜索的计算密集型优化过程来恢复每个输入批次，使扩展变得困难，要么涉及恶意参数服务器在全局模型体系结构之前添加额外的模块，使攻击过于显眼和容易检测。为了克服这些局限性，我们提出了Scale-MIA，一种新的MIA，即使在系统处于健壮的安全聚合协议的保护下，也能够从聚合的更新中高效而准确地恢复客户端的训练样本。与将模型视为黑盒的现有方法不同，Scale-MIA认识到机器学习模型复杂的体系结构和内部工作原理的重要性。它将潜在空间识别为侵犯隐私的关键层，并将复杂的恢复任务分解为一个创新的两步过程，以降低计算复杂度。第一步涉及使用闭合形式的反转机制从聚集的模型更新重构潜在空间表示(LSR)，利用特制的对抗性线性层。在第二步中，通过将输入批次馈送到微调的产生式解码器，从LSR中恢复整个输入批次。我们在多种常用的机器学习模型上实现了Scale-MIA，并在不同的环境下进行了全面的实验。结果表明，Scale-MIA在不同的数据集上表现出了良好的恢复性能，与现有的MIA相比，在更大的范围内表现出更高的重建率、准确性和攻击效率。



## **23. Laccolith: Hypervisor-Based Adversary Emulation with Anti-Detection**

Laccolith：基于系统管理程序的反检测对手仿真 cs.CR

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.08274v1) [paper-pdf](http://arxiv.org/pdf/2311.08274v1)

**Authors**: Vittorio Orbinato, Marco Carlo Feliciano, Domenico Cotroneo, Roberto Natella

**Abstract**: Advanced Persistent Threats (APTs) represent the most threatening form of attack nowadays since they can stay undetected for a long time. Adversary emulation is a proactive approach for preparing against these attacks. However, adversary emulation tools lack the anti-detection abilities of APTs. We introduce Laccolith, a hypervisor-based solution for adversary emulation with anti-detection to fill this gap. We also present an experimental study to compare Laccolith with MITRE CALDERA, a state-of-the-art solution for adversary emulation, against five popular anti-virus products. We found that CALDERA cannot evade detection, limiting the realism of emulated attacks, even when combined with a state-of-the-art anti-detection framework. Our experiments show that Laccolith can hide its activities from all the tested anti-virus products, thus making it suitable for realistic emulations.

摘要: 高级持续性威胁（APT）是当今最具威胁性的攻击形式，因为它们可以长时间不被发现。Advertisement模拟是一种预防这些攻击的主动方法。然而，敌手仿真工具缺乏APT的反检测能力。我们介绍Laccolith，一个基于虚拟机管理程序的解决方案，对手模拟与反检测，以填补这一空白。我们还提出了一个实验研究，比较Laccolith与MITRE CALDERA，一个国家的最先进的解决方案，对手模拟，对五个流行的反病毒产品。我们发现CALDERA无法逃避检测，限制了模拟攻击的真实性，即使与最先进的反检测框架相结合。我们的实验表明，Laccolith可以隐藏其活动，从所有测试的反病毒产品，从而使其适合现实的仿真。



## **24. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

披着羊皮的狼：广义嵌套越狱提示可以轻松愚弄大型语言模型 cs.CL

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.08268v1) [paper-pdf](http://arxiv.org/pdf/2311.08268v1)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on another white-box model, compromising generalization or jailbreak efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we offer detailed analysis and discussion from the perspective of prompt execution priority on the failure of LLMs' defense. We hope that our research can catalyze both the academic community and LLMs vendors towards the provision of safer and more regulated Large Language Models.

摘要: 大型语言模型（LLM），如ChatGPT和GPT-4，旨在提供有用和安全的响应。然而，被称为“越狱”的对抗性提示可以规避保障措施，导致LLM生成有害内容。探索越狱提示可以帮助更好地揭示LLM的弱点，并进一步引导我们保护它们。不幸的是，现有的越狱方法要么遭受复杂的手动设计，要么需要在另一个白盒模型上进行优化，从而影响泛化或越狱效率。本文将越狱提示攻击归纳为两个方面：（1）提示重写和（2）场景嵌套。在此基础上，我们提出了ReNeLLM，一个自动框架，利用LLM本身来生成有效的越狱提示。大量的实验表明，与现有的基线相比，ReNeLLM显着提高了攻击成功率，同时大大降低了时间成本。我们的研究也揭示了目前的防御方法在保护LLM方面的不足。最后，从及时执行优先权的角度对有限责任公司抗辩失败进行了详细的分析和探讨。我们希望我们的研究能够促进学术界和LLM供应商提供更安全，更规范的大型语言模型。



## **25. On The Relationship Between Universal Adversarial Attacks And Sparse Representations**

关于泛对抗性攻击与稀疏表示的关系 cs.CV

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.08265v1) [paper-pdf](http://arxiv.org/pdf/2311.08265v1)

**Authors**: Dana Weitzner, Raja Giryes

**Abstract**: The prominent success of neural networks, mainly in computer vision tasks, is increasingly shadowed by their sensitivity to small, barely perceivable adversarial perturbations in image input.   In this work, we aim at explaining this vulnerability through the framework of sparsity.   We show the connection between adversarial attacks and sparse representations, with a focus on explaining the universality and transferability of adversarial examples in neural networks.   To this end, we show that sparse coding algorithms, and the neural network-based learned iterative shrinkage thresholding algorithm (LISTA) among them, suffer from this sensitivity, and that common attacks on neural networks can be expressed as attacks on the sparse representation of the input image. The phenomenon that we observe holds true also when the network is agnostic to the sparse representation and dictionary, and thus can provide a possible explanation for the universality and transferability of adversarial attacks.   The code is available at https://github.com/danawr/adversarial_attacks_and_sparse_representations.

摘要: 神经网络的显著成功，主要是在计算机视觉任务中，由于它们对图像输入中几乎察觉不到的微小对抗性扰动的敏感性，越来越黯然失色。在这项工作中，我们旨在通过稀疏性的框架来解释这个漏洞。我们展示了对抗性攻击和稀疏表示之间的联系，重点解释了神经网络中对抗性例子的普遍性和可转移性。为此，我们证明了稀疏编码算法，以及其中的基于神经网络的学习迭代收缩阈值算法(LISTA)，都受到这种敏感性的影响，并且对神经网络的常见攻击可以表示为对输入图像的稀疏表示的攻击。当网络对稀疏表示和字典不可知时，我们观察到的现象也是成立的，从而为对抗攻击的普遍性和可转移性提供了可能的解释。代码可在https://github.com/danawr/adversarial_attacks_and_sparse_representations.上获得



## **26. The Impact of Adversarial Node Placement in Decentralized Federated Learning Networks**

分布式联合学习网络中对抗性节点放置的影响 cs.CR

Submitted to ICC 2023 conference

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.07946v1) [paper-pdf](http://arxiv.org/pdf/2311.07946v1)

**Authors**: Adam Piaseczny, Eric Ruzomberka, Rohit Parasnis, Christopher G. Brinton

**Abstract**: As Federated Learning (FL) grows in popularity, new decentralized frameworks are becoming widespread. These frameworks leverage the benefits of decentralized environments to enable fast and energy-efficient inter-device communication. However, this growing popularity also intensifies the need for robust security measures. While existing research has explored various aspects of FL security, the role of adversarial node placement in decentralized networks remains largely unexplored. This paper addresses this gap by analyzing the performance of decentralized FL for various adversarial placement strategies when adversaries can jointly coordinate their placement within a network. We establish two baseline strategies for placing adversarial node: random placement and network centrality-based placement. Building on this foundation, we propose a novel attack algorithm that prioritizes adversarial spread over adversarial centrality by maximizing the average network distance between adversaries. We show that the new attack algorithm significantly impacts key performance metrics such as testing accuracy, outperforming the baseline frameworks by between 9% and 66.5% for the considered setups. Our findings provide valuable insights into the vulnerabilities of decentralized FL systems, setting the stage for future research aimed at developing more secure and robust decentralized FL frameworks.

摘要: 随着联邦学习(FL)的流行，新的去中心化框架正在变得广泛。这些框架利用分散环境的优势，实现快速、节能的设备间通信。然而，这种日益增长的人气也加剧了采取强有力的安全措施的必要性。虽然现有的研究已经探索了FL安全的各个方面，但敌意节点放置在分散网络中的作用在很大程度上仍未被探索。本文通过分析当对手可以在一个网络内联合协调他们的放置时，分散的FL在不同的对手放置策略下的性能来解决这一差距。我们建立了两种放置敌意节点的基线策略：随机放置和基于网络中心性的放置。在此基础上，我们提出了一种新的攻击算法，该算法通过最大化对手之间的平均网络距离来优先考虑对手的传播而不是对手的中心。我们发现，新的攻击算法显著影响了测试准确率等关键性能指标，在所考虑的设置下，性能比基准框架高出9%到66.5%。我们的发现对去中心化FL系统的脆弱性提供了有价值的见解，为未来旨在开发更安全和健壮的去中心化FL框架的研究奠定了基础。



## **27. Towards Improving Robustness Against Common Corruptions in Object Detectors Using Adversarial Contrastive Learning**

利用对抗性对比学习提高目标检测器对常见腐败的稳健性 cs.CV

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.07928v1) [paper-pdf](http://arxiv.org/pdf/2311.07928v1)

**Authors**: Shashank Kotyan, Danilo Vasconcellos Vargas

**Abstract**: Neural networks have revolutionized various domains, exhibiting remarkable accuracy in tasks like natural language processing and computer vision. However, their vulnerability to slight alterations in input samples poses challenges, particularly in safety-critical applications like autonomous driving. Current approaches, such as introducing distortions during training, fall short in addressing unforeseen corruptions. This paper proposes an innovative adversarial contrastive learning framework to enhance neural network robustness simultaneously against adversarial attacks and common corruptions. By generating instance-wise adversarial examples and optimizing contrastive loss, our method fosters representations that resist adversarial perturbations and remain robust in real-world scenarios. Subsequent contrastive learning then strengthens the similarity between clean samples and their adversarial counterparts, fostering representations resistant to both adversarial attacks and common distortions. By focusing on improving performance under adversarial and real-world conditions, our approach aims to bolster the robustness of neural networks in safety-critical applications, such as autonomous vehicles navigating unpredictable weather conditions. We anticipate that this framework will contribute to advancing the reliability of neural networks in challenging environments, facilitating their widespread adoption in mission-critical scenarios.

摘要: 神经网络已经彻底改变了各个领域，在自然语言处理和计算机视觉等任务中表现出非凡的准确性。然而，它们容易受到输入样本轻微变化的影响，这带来了挑战，特别是在自动驾驶等安全关键型应用中。目前的办法，例如在培训期间引入扭曲做法，不足以解决不可预见的腐败问题。本文提出了一个创新的对抗性对比学习框架，以提高神经网络的鲁棒性，同时对抗对抗性攻击和常见的腐败。通过生成实例对抗性示例和优化对比损失，我们的方法培养了抵抗对抗性扰动并在现实世界场景中保持鲁棒性的表示。随后的对比学习加强了干净样本与其对抗样本之间的相似性，培养了抵抗对抗攻击和常见扭曲的表征。通过专注于提高在对抗性和真实世界条件下的性能，我们的方法旨在增强神经网络在安全关键应用中的鲁棒性，例如在不可预测的天气条件下导航的自动驾驶汽车。我们预计，该框架将有助于提高神经网络在具有挑战性的环境中的可靠性，促进其在关键任务场景中的广泛采用。



## **28. Cooperative AI via Decentralized Commitment Devices**

基于分散承诺机制的协作式人工智能 cs.AI

NeurIPS 2023- Multi-Agent Security Workshop

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.07815v1) [paper-pdf](http://arxiv.org/pdf/2311.07815v1)

**Authors**: Xinyuan Sun, Davide Crapis, Matt Stephenson, Barnabé Monnot, Thomas Thiery, Jonathan Passerat-Palmbach

**Abstract**: Credible commitment devices have been a popular approach for robust multi-agent coordination. However, existing commitment mechanisms face limitations like privacy, integrity, and susceptibility to mediator or user strategic behavior. It is unclear if the cooperative AI techniques we study are robust to real-world incentives and attack vectors. However, decentralized commitment devices that utilize cryptography have been deployed in the wild, and numerous studies have shown their ability to coordinate algorithmic agents facing adversarial opponents with significant economic incentives, currently in the order of several million to billions of dollars. In this paper, we use examples in the decentralization and, in particular, Maximal Extractable Value (MEV) (arXiv:1904.05234) literature to illustrate the potential security issues in cooperative AI. We call for expanded research into decentralized commitments to advance cooperative AI capabilities for secure coordination in open environments and empirical testing frameworks to evaluate multi-agent coordination ability given real-world commitment constraints.

摘要: 可信的承诺机制一直是一种强有力的多主体协调的流行方法。然而，现有的承诺机制面临着隐私、完整性以及对调解人或用户策略行为的敏感性等限制。目前尚不清楚我们研究的合作人工智能技术是否对现实世界的激励和攻击矢量具有健壮性。然而，利用密码学的去中心化承诺设备已经在野外部署，许多研究表明，它们能够协调算法代理面对具有重大经济激励的对手，目前约为数百万至数十亿美元。在本文中，我们使用去中心化，特别是最大可提取值(ARXIV：1904.05234)文献中的例子来说明合作人工智能中潜在的安全问题。我们呼吁扩大对分散承诺的研究，以促进开放环境中安全协调的合作人工智能能力，并在现实世界承诺限制的情况下，评估多代理协调能力的经验测试框架。



## **29. Parrot-Trained Adversarial Examples: Pushing the Practicality of Black-Box Audio Attacks against Speaker Recognition Models**

鹦鹉训练的对抗性例子：将黑匣子音频攻击的实用性推向说话人识别模型 cs.SD

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.07780v1) [paper-pdf](http://arxiv.org/pdf/2311.07780v1)

**Authors**: Rui Duan, Zhe Qu, Leah Ding, Yao Liu, Zhuo Lu

**Abstract**: Audio adversarial examples (AEs) have posed significant security challenges to real-world speaker recognition systems. Most black-box attacks still require certain information from the speaker recognition model to be effective (e.g., keeping probing and requiring the knowledge of similarity scores). This work aims to push the practicality of the black-box attacks by minimizing the attacker's knowledge about a target speaker recognition model. Although it is not feasible for an attacker to succeed with completely zero knowledge, we assume that the attacker only knows a short (or a few seconds) speech sample of a target speaker. Without any probing to gain further knowledge about the target model, we propose a new mechanism, called parrot training, to generate AEs against the target model. Motivated by recent advancements in voice conversion (VC), we propose to use the one short sentence knowledge to generate more synthetic speech samples that sound like the target speaker, called parrot speech. Then, we use these parrot speech samples to train a parrot-trained(PT) surrogate model for the attacker. Under a joint transferability and perception framework, we investigate different ways to generate AEs on the PT model (called PT-AEs) to ensure the PT-AEs can be generated with high transferability to a black-box target model with good human perceptual quality. Real-world experiments show that the resultant PT-AEs achieve the attack success rates of 45.8% - 80.8% against the open-source models in the digital-line scenario and 47.9% - 58.3% against smart devices, including Apple HomePod (Siri), Amazon Echo, and Google Home, in the over-the-air scenario.

摘要: 音频对抗性例子(AEs)对真实说话人识别系统提出了巨大的安全挑战。大多数黑盒攻击仍然需要说话人识别模型中的某些信息才能有效(例如，保持探测并需要知道相似性得分)。这项工作旨在通过最小化攻击者对目标说话人识别模型的了解来推动黑盒攻击的实用性。虽然攻击者在完全零知识的情况下成功是不可行的，但我们假设攻击者只知道目标说话人的一小段(或几秒钟)语音样本。在没有任何关于目标模型的进一步知识的情况下，我们提出了一种新的机制，称为鹦鹉训练，以生成针对目标模型的AE。受语音转换领域最新进展的启发，我们提出了利用一小句话的知识来生成更多听起来像目标说话人的合成语音样本，称为鹦鹉语音。然后，我们使用这些鹦鹉语音样本为攻击者训练一个鹦鹉训练(PT)代理模型。在可转移性和感知联合框架下，我们研究了在PT模型(称为PT-AEs)上生成AEs的不同方法，以确保生成的PT-AEs能够高可转移性地生成具有良好人类感知质量的黑盒目标模型。真实世界实验表明，在数字线路场景中，所生成的PT-AE对开源模型的攻击成功率为45.8%-80.8%，在空中场景中，对Apple HomePod(Siri)、Amazon Echo和Google Home等智能设备的攻击成功率为47.9%-58.3%。



## **30. Towards a robust and reliable deep learning approach for detection of compact binary mergers in gravitational wave data**

一种稳健可靠的深度学习方法检测引力波数据中的紧密二元合并 gr-qc

22 pages, 22 figures

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2306.11797v2) [paper-pdf](http://arxiv.org/pdf/2306.11797v2)

**Authors**: Shreejit Jadhav, Mihir Shrivastava, Sanjit Mitra

**Abstract**: The ability of deep learning (DL) approaches to learn generalised signal and noise models, coupled with their fast inference on GPUs, holds great promise for enhancing gravitational-wave (GW) searches in terms of speed, parameter space coverage, and search sensitivity. However, the opaque nature of DL models severely harms their reliability. In this work, we meticulously develop a DL model stage-wise and work towards improving its robustness and reliability. First, we address the problems in maintaining the purity of training data by deriving a new metric that better reflects the visual strength of the 'chirp' signal features in the data. Using a reduced, smooth representation obtained through a variational auto-encoder (VAE), we build a classifier to search for compact binary coalescence (CBC) signals. Our tests on real LIGO data show an impressive performance of the model. However, upon probing the robustness of the model through adversarial attacks, its simple failure modes were identified, underlining how such models can still be highly fragile. As a first step towards bringing robustness, we retrain the model in a novel framework involving a generative adversarial network (GAN). Over the course of training, the model learns to eliminate the primary modes of failure identified by the adversaries. Although absolute robustness is practically impossible to achieve, we demonstrate some fundamental improvements earned through such training, like sparseness and reduced degeneracy in the extracted features at different layers inside the model. We show that these gains are achieved at practically zero loss in terms of model performance on real LIGO data before and after GAN training. Through a direct search on 8.8 days of LIGO data, we recover two significant CBC events from GWTC-2.1, GW190519_153544 and GW190521_074359. We also report the search sensitivity obtained from an injection study.

摘要: 深度学习(DL)方法学习广义信号和噪声模型的能力，加上它们在GPU上的快速推理，在速度、参数空间覆盖和搜索灵敏度方面都有望提高引力波(GW)搜索。然而，DL模型的不透明性质严重损害了它们的可靠性。在这项工作中，我们精心开发了一个阶段性的DL模型，并致力于提高其稳健性和可靠性。首先，我们通过推导一种新的度量来解决保持训练数据的纯净度的问题，该度量更好地反映了数据中“chirp”信号特征的视觉强度。利用变分自动编码器(VAE)得到的简化的平滑表示，我们构建了一个分类器来搜索紧凑的二进制合并(CBC)信号。我们对真实LIGO数据的测试表明，该模型具有令人印象深刻的性能。然而，在通过对抗性攻击探测该模型的稳健性之后，它的简单故障模式被识别出来，这突显了这种模型如何仍然非常脆弱。作为带来稳健性的第一步，我们在一个新的框架中重新训练该模型，该框架涉及生成性对抗网络(GAN)。在训练过程中，模型学习消除对手确定的主要失败模式。虽然绝对的稳健性实际上是不可能实现的，但我们展示了通过这样的训练获得的一些基本的改进，例如在模型内部不同层提取的特征的稀疏性和减少的简并度。我们表明，在GaN训练之前和之后，在实际LIGO数据上的模型性能方面，这些增益几乎是零损失的。通过直接搜索8.8d的LIGO数据，我们从GWTC2.1中恢复了两个重要的CBC事件，GW190519_153544和GW190521_074359。我们还报告了从注射研究中获得的搜索灵敏度。



## **31. MART: Improving LLM Safety with Multi-round Automatic Red-Teaming**

MART：用多轮自动红队提高LLM安全 cs.CL

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.07689v1) [paper-pdf](http://arxiv.org/pdf/2311.07689v1)

**Authors**: Suyu Ge, Chunting Zhou, Rui Hou, Madian Khabsa, Yi-Chia Wang, Qifan Wang, Jiawei Han, Yuning Mao

**Abstract**: Red-teaming is a common practice for mitigating unsafe behaviors in Large Language Models (LLMs), which involves thoroughly assessing LLMs to identify potential flaws and addressing them with responsible and accurate responses. While effective, manual red-teaming is costly, and existing automatic red-teaming typically discovers safety risks without addressing them. In this paper, we propose a Multi-round Automatic Red-Teaming (MART) method, which incorporates both automatic adversarial prompt writing and safe response generation, significantly increasing red-teaming scalability and the safety of the target LLM. Specifically, an adversarial LLM and a target LLM interplay with each other in an iterative manner, where the adversarial LLM aims to generate challenging prompts that elicit unsafe responses from the target LLM, while the target LLM is fine-tuned with safety aligned data on these adversarial prompts. In each round, the adversarial LLM crafts better attacks on the updated target LLM, while the target LLM also improves itself through safety fine-tuning. On adversarial prompt benchmarks, the violation rate of an LLM with limited safety alignment reduces up to 84.7% after 4 rounds of MART, achieving comparable performance to LLMs with extensive adversarial prompt writing. Notably, model helpfulness on non-adversarial prompts remains stable throughout iterations, indicating the target LLM maintains strong performance on instruction following.

摘要: 红团队是减少大型语言模型(LLM)中不安全行为的一种常见做法，它涉及彻底评估LLM以识别潜在缺陷并以负责任和准确的响应来解决它们。虽然有效，但手动红色团队成本高昂，而且现有的自动红色团队通常会发现安全风险，而不解决这些风险。本文提出了一种多轮自动红队(MART)方法，该方法结合了自动编写敌方提示和安全响应生成的功能，显著提高了红队的可扩展性和目标LLM的安全性。具体地说，对抗性LLM和目标LLM以迭代的方式相互作用，其中对抗性LLM旨在生成引起来自目标LLM的不安全响应的挑战性提示，而目标LLM利用关于这些对抗性提示的安全对齐的数据进行微调。在每一轮中，对抗性的LLM对更新后的目标LLM进行更好的攻击，而目标LLM也通过安全微调来提高自己。在对抗性提示基准上，有限安全对齐的LLM在4轮MART后的违规率降低了84.7%，获得了与具有广泛对抗性提示书写的LLMS相当的性能。值得注意的是，在非对抗性提示上的模型帮助在迭代过程中保持稳定，这表明目标LLM在指令跟随上保持着强大的性能。



## **32. An Extensive Study on Adversarial Attack against Pre-trained Models of Code**

对预训练代码模型的对抗性攻击的扩展研究 cs.CR

Accepted to ESEC/FSE 2023

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.07553v1) [paper-pdf](http://arxiv.org/pdf/2311.07553v1)

**Authors**: Xiaohu Du, Ming Wen, Zichao Wei, Shangwen Wang, Hai Jin

**Abstract**: Transformer-based pre-trained models of code (PTMC) have been widely utilized and have achieved state-of-the-art performance in many mission-critical applications. However, they can be vulnerable to adversarial attacks through identifier substitution or coding style transformation, which can significantly degrade accuracy and may further incur security concerns. Although several approaches have been proposed to generate adversarial examples for PTMC, the effectiveness and efficiency of such approaches, especially on different code intelligence tasks, has not been well understood. To bridge this gap, this study systematically analyzes five state-of-the-art adversarial attack approaches from three perspectives: effectiveness, efficiency, and the quality of generated examples. The results show that none of the five approaches balances all these perspectives. Particularly, approaches with a high attack success rate tend to be time-consuming; the adversarial code they generate often lack naturalness, and vice versa. To address this limitation, we explore the impact of perturbing identifiers under different contexts and find that identifier substitution within for and if statements is the most effective. Based on these findings, we propose a new approach that prioritizes different types of statements for various tasks and further utilizes beam search to generate adversarial examples. Evaluation results show that it outperforms the state-of-the-art ALERT in terms of both effectiveness and efficiency while preserving the naturalness of the generated adversarial examples.

摘要: 基于变压器的预训练代码模型(PTMC)已被广泛使用，并在许多任务关键型应用中取得了最先进的性能。然而，通过标识符替换或编码样式转换，它们很容易受到敌意攻击，这可能会显著降低准确性，并可能进一步引起安全问题。虽然已经提出了几种方法来为PTMC生成对抗性例子，但这些方法的有效性和效率，特别是在不同的代码情报任务上，还没有被很好地理解。为了弥补这一差距，本研究从有效性、效率和生成实例的质量三个角度系统地分析了五种最新的对抗性攻击方法。结果表明，这五种方法都不能平衡所有这些观点。特别是，攻击成功率高的方法往往很耗时；它们生成的敌意代码通常缺乏自然性，反之亦然。为了解决这一限制，我们研究了在不同上下文中干扰标识符所产生的影响，并发现在for和if语句中替换标识符是最有效的。基于这些发现，我们提出了一种新的方法，该方法对不同任务的不同类型的语句进行优先排序，并进一步利用BEAM搜索来生成对抗性示例。评估结果表明，该算法在保持生成的对抗性实例的自然性的同时，在有效性和效率上都优于最新的警报。



## **33. On the Robustness of Neural Collapse and the Neural Collapse of Robustness**

关于神经崩溃的稳健性和神经崩溃的稳健性 cs.LG

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.07444v1) [paper-pdf](http://arxiv.org/pdf/2311.07444v1)

**Authors**: Jingtong Su, Ya Shi Zhang, Nikolaos Tsilivis, Julia Kempe

**Abstract**: Neural Collapse refers to the curious phenomenon in the end of training of a neural network, where feature vectors and classification weights converge to a very simple geometrical arrangement (a simplex). While it has been observed empirically in various cases and has been theoretically motivated, its connection with crucial properties of neural networks, like their generalization and robustness, remains unclear. In this work, we study the stability properties of these simplices. We find that the simplex structure disappears under small adversarial attacks, and that perturbed examples "leap" between simplex vertices. We further analyze the geometry of networks that are optimized to be robust against adversarial perturbations of the input, and find that Neural Collapse is a pervasive phenomenon in these cases as well, with clean and perturbed representations forming aligned simplices, and giving rise to a robust simple nearest-neighbor classifier. By studying the propagation of the amount of collapse inside the network, we identify novel properties of both robust and non-robust machine learning models, and show that earlier, unlike later layers maintain reliable simplices on perturbed data.

摘要: 神经崩溃是指在神经网络的训练结束时，特征向量和分类权重收敛到一个非常简单的几何排列(单纯形)的奇怪现象。虽然它已经在各种情况下得到了经验的观察，并在理论上得到了推动，但它与神经网络的关键特性，如它们的泛化和健壮性的联系，仍然不清楚。在这项工作中，我们研究了这些单形的稳定性。我们发现，单纯形结构在小的对抗性攻击下消失，扰动的例子在单纯形顶点之间跳跃。我们进一步分析了优化后的网络的几何结构，发现在这些情况下，神经崩溃也是一种普遍现象，干净和扰动的表示形成了对齐的简化，并产生了一个健壮的简单最近邻分类器。通过研究崩溃量在网络中的传播，我们识别了健壮和非健壮机器学习模型的新性质，并表明早期不同于后面的层在扰动数据上保持可靠的简化。



## **34. Transpose Attack: Stealing Datasets with Bidirectional Training**

转置攻击：通过双向训练窃取数据集 cs.LG

NDSS24 paper

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.07389v1) [paper-pdf](http://arxiv.org/pdf/2311.07389v1)

**Authors**: Guy Amit, Mosh Levy, Yisroel Mirsky

**Abstract**: Deep neural networks are normally executed in the forward direction. However, in this work, we identify a vulnerability that enables models to be trained in both directions and on different tasks. Adversaries can exploit this capability to hide rogue models within seemingly legitimate models. In addition, in this work we show that neural networks can be taught to systematically memorize and retrieve specific samples from datasets. Together, these findings expose a novel method in which adversaries can exfiltrate datasets from protected learning environments under the guise of legitimate models. We focus on the data exfiltration attack and show that modern architectures can be used to secretly exfiltrate tens of thousands of samples with high fidelity, high enough to compromise data privacy and even train new models. Moreover, to mitigate this threat we propose a novel approach for detecting infected models.

摘要: 深度神经网络通常在正向执行。然而，在这项工作中，我们发现了一个漏洞，该漏洞使模型能够在两个方向上进行不同任务的训练。攻击者可以利用这一功能将流氓模型隐藏在看似合法的模型中。此外，在这项工作中，我们表明可以教神经网络系统地记忆和检索数据集中的特定样本。总之，这些发现揭示了一种新的方法，在这种方法中，攻击者可以打着合法模型的幌子从受保护的学习环境中渗出数据集。我们聚焦于数据外泄攻击，并展示了现代架构可以用来秘密渗出数万个高保真的样本，高到足以危及数据隐私，甚至可以训练新的模型。此外，为了缓解这一威胁，我们提出了一种新的检测感染模型的方法。



## **35. Untargeted Black-box Attacks for Social Recommendations**

针对社交推荐的无目标黑匣子攻击 cs.SI

Preprint. Under review

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.07127v1) [paper-pdf](http://arxiv.org/pdf/2311.07127v1)

**Authors**: Wenqi Fan, Shijie Wang, Xiao-yong Wei, Xiaowei Mei, Qing Li

**Abstract**: The rise of online social networks has facilitated the evolution of social recommender systems, which incorporate social relations to enhance users' decision-making process. With the great success of Graph Neural Networks in learning node representations, GNN-based social recommendations have been widely studied to model user-item interactions and user-user social relations simultaneously. Despite their great successes, recent studies have shown that these advanced recommender systems are highly vulnerable to adversarial attacks, in which attackers can inject well-designed fake user profiles to disrupt recommendation performances. While most existing studies mainly focus on targeted attacks to promote target items on vanilla recommender systems, untargeted attacks to degrade the overall prediction performance are less explored on social recommendations under a black-box scenario. To perform untargeted attacks on social recommender systems, attackers can construct malicious social relationships for fake users to enhance the attack performance. However, the coordination of social relations and item profiles is challenging for attacking black-box social recommendations. To address this limitation, we first conduct several preliminary studies to demonstrate the effectiveness of cross-community connections and cold-start items in degrading recommendations performance. Specifically, we propose a novel framework Multiattack based on multi-agent reinforcement learning to coordinate the generation of cold-start item profiles and cross-community social relations for conducting untargeted attacks on black-box social recommendations. Comprehensive experiments on various real-world datasets demonstrate the effectiveness of our proposed attacking framework under the black-box setting.

摘要: 在线社交网络的兴起促进了社交推荐系统的发展，社交推荐系统整合了社会关系，以增强用户的决策过程。随着图神经网络在学习节点表示方面的巨大成功，基于GNN的社交推荐被广泛研究以同时建模用户-项目交互和用户-用户社会关系。尽管它们取得了巨大的成功，但最近的研究表明，这些先进的推荐系统非常容易受到对手攻击，攻击者可以注入精心设计的虚假用户配置文件来破坏推荐性能。虽然现有的研究主要集中于在普通推荐系统上通过定向攻击来推广目标项，但在黑盒场景下，针对社交推荐的非定向攻击以降低整体预测性能的研究较少。为了对社交推荐系统进行无针对性的攻击，攻击者可以为虚假用户构建恶意的社交关系，以提高攻击性能。然而，社交关系和项目简介的协调对于攻击黑箱社交推荐是具有挑战性的。为了解决这一局限性，我们首先进行了几项初步研究，以证明跨社区联系和冷启动项目在降低推荐性能方面的有效性。具体地说，我们提出了一种新的基于多智能体强化学习的多攻击框架，用于协调冷启动项目配置文件的生成和跨社区社会关系的生成，以对黑盒社交推荐进行无针对性的攻击。在各种真实数据集上的综合实验证明了我们提出的攻击框架在黑盒环境下的有效性。



## **36. Adversarial Purification for Data-Driven Power System Event Classifiers with Diffusion Models**

基于扩散模型的数据驱动电力系统事件分类器对抗净化 eess.SY

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.07110v1) [paper-pdf](http://arxiv.org/pdf/2311.07110v1)

**Authors**: Yuanbin Cheng, Koji Yamashita, Jim Follum, Nanpeng Yu

**Abstract**: The global deployment of the phasor measurement units (PMUs) enables real-time monitoring of the power system, which has stimulated considerable research into machine learning-based models for event detection and classification. However, recent studies reveal that machine learning-based methods are vulnerable to adversarial attacks, which can fool the event classifiers by adding small perturbations to the raw PMU data. To mitigate the threats posed by adversarial attacks, research on defense strategies is urgently needed. This paper proposes an effective adversarial purification method based on the diffusion model to counter adversarial attacks on the machine learning-based power system event classifier. The proposed method includes two steps: injecting noise into the PMU data; and utilizing a pre-trained neural network to eliminate the added noise while simultaneously removing perturbations introduced by the adversarial attacks. The proposed adversarial purification method significantly increases the accuracy of the event classifier under adversarial attacks while satisfying the requirements of real-time operations. In addition, the theoretical analysis reveals that the proposed diffusion model-based adversarial purification method decreases the distance between the original and compromised PMU data, which reduces the impacts of adversarial attacks. The empirical results on a large-scale real-world PMU dataset validate the effectiveness and computational efficiency of the proposed adversarial purification method.

摘要: 相量测量单元(PMU)的全球部署实现了对电力系统的实时监控，这促使人们对基于机器学习的事件检测和分类模型进行了大量研究。然而，最近的研究表明，基于机器学习的方法容易受到对抗性攻击，这些攻击可以通过在原始PMU数据中添加小的扰动来欺骗事件分类器。为了缓解对抗性攻击带来的威胁，迫切需要对防御策略进行研究。针对基于机器学习的电力系统事件分类器的对抗性攻击，提出了一种基于扩散模型的对抗性净化方法。该方法包括两个步骤：向PMU数据中注入噪声；利用预先训练的神经网络消除增加的噪声，同时消除对抗性攻击带来的扰动。本文提出的对抗性净化方法在满足实时操作要求的同时，显著提高了对抗性攻击下事件分类器的准确率。此外，理论分析表明，本文提出的基于扩散模型的对抗性净化方法减小了原始PMU数据与受损PMU数据之间的距离，从而降低了对抗性攻击的影响。在大规模真实PMU数据集上的实验结果验证了该对抗性净化方法的有效性和计算效率。



## **37. Language Model Unalignment: Parametric Red-Teaming to Expose Hidden Harms and Biases**

语言模型不一致：暴露隐藏的危害和偏见的参数红色团队 cs.CL

Under Review

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2310.14303v2) [paper-pdf](http://arxiv.org/pdf/2310.14303v2)

**Authors**: Rishabh Bhardwaj, Soujanya Poria

**Abstract**: Red-teaming has been a widely adopted way to evaluate the harmfulness of Large Language Models (LLMs). It aims to jailbreak a model's safety behavior to make it act as a helpful agent disregarding the harmfulness of the query. Existing methods are primarily based on input text-based red-teaming such as adversarial prompts, low-resource prompts, or contextualized prompts to condition the model in a way to bypass its safe behavior. Bypassing the guardrails uncovers hidden harmful information and biases in the model that are left untreated or newly introduced by its safety training. However, prompt-based attacks fail to provide such a diagnosis owing to their low attack success rate, and applicability to specific models. In this paper, we present a new perspective on LLM safety research i.e., parametric red-teaming through Unalignment. It simply (instruction) tunes the model parameters to break model guardrails that are not deeply rooted in the model's behavior. Unalignment using as few as 100 examples can significantly bypass commonly referred to as CHATGPT, to the point where it responds with an 88% success rate to harmful queries on two safety benchmark datasets. On open-source models such as VICUNA-7B and LLAMA-2-CHAT 7B AND 13B, it shows an attack success rate of more than 91%. On bias evaluations, Unalignment exposes inherent biases in safety-aligned models such as CHATGPT and LLAMA- 2-CHAT where the model's responses are strongly biased and opinionated 64% of the time.

摘要: 红团队已被广泛采用来评估大型语言模型的危害性。它的目的是让模特的安全行为越狱，使其成为一个有帮助的代理人，而不考虑询问的危害性。现有方法主要基于诸如对抗性提示、低资源提示或情境化提示的基于输入文本的红团队，以使模型以绕过其安全行为的方式调节。绕过护栏发现了模型中隐藏的有害信息和偏见，这些信息和偏见是未经处理的或安全培训新引入的。然而，基于提示的攻击无法提供这样的诊断，因为它们的攻击成功率低，并且适用于特定的模型。在这篇文章中，我们提出了一个新的视角来研究LLM安全，即通过非对齐的参数红组。它只是(指令)调整模型参数，以打破并不深深植根于模型行为中的模型护栏。只要使用100个例子，UnAlign就可以显著绕过通常所说的CHATGPT，以至于它对两个安全基准数据集上的有害查询的响应成功率为88%。在VIVUNA-7B和LLAMA-2-Chat 7B和13B等开源机型上，攻击成功率超过91%。在偏差评估方面，UnAlign暴露了安全对齐模型中的固有偏见，如CHATGPT和Llama-2-Chat，其中模型的反应在64%的时间内是强烈偏见和固执己见的。



## **38. PATROL: Privacy-Oriented Pruning for Collaborative Inference Against Model Inversion Attacks**

PATROL：针对模型反转攻击的协同推理面向隐私剪枝 cs.LG

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2307.10981v2) [paper-pdf](http://arxiv.org/pdf/2307.10981v2)

**Authors**: Shiwei Ding, Lan Zhang, Miao Pan, Xiaoyong Yuan

**Abstract**: Collaborative inference has been a promising solution to enable resource-constrained edge devices to perform inference using state-of-the-art deep neural networks (DNNs). In collaborative inference, the edge device first feeds the input to a partial DNN locally and then uploads the intermediate result to the cloud to complete the inference. However, recent research indicates model inversion attacks (MIAs) can reconstruct input data from intermediate results, posing serious privacy concerns for collaborative inference. Existing perturbation and cryptography techniques are inefficient and unreliable in defending against MIAs while performing accurate inference. This paper provides a viable solution, named PATROL, which develops privacy-oriented pruning to balance privacy, efficiency, and utility of collaborative inference. PATROL takes advantage of the fact that later layers in a DNN can extract more task-specific features. Given limited local resources for collaborative inference, PATROL intends to deploy more layers at the edge based on pruning techniques to enforce task-specific features for inference and reduce task-irrelevant but sensitive features for privacy preservation. To achieve privacy-oriented pruning, PATROL introduces two key components: Lipschitz regularization and adversarial reconstruction training, which increase the reconstruction errors by reducing the stability of MIAs and enhance the target inference model by adversarial training, respectively. On a real-world collaborative inference task, vehicle re-identification, we demonstrate the superior performance of PATROL in terms of against MIAs.

摘要: 协作推理是一种很有前途的解决方案，它使资源受限的边缘设备能够使用最先进的深度神经网络(DNN)进行推理。在协同推理中，边缘设备首先将输入反馈到本地的部分DNN，然后将中间结果上传到云中完成推理。然而，最近的研究表明，模型反转攻击(MIA)可以从中间结果重建输入数据，这给协作推理带来了严重的隐私问题。现有的微扰和密码技术在防御MIA的同时执行准确的推理是低效和不可靠的。本文提出了一个可行的解决方案，称为PATR，它发展了面向隐私的剪枝，以平衡协作推理的私密性、效率和效用。PATROL利用了DNN中较晚的层可以提取更多特定于任务的特征这一事实。考虑到用于协作推理的本地资源有限，PATR打算基于剪枝技术在边缘部署更多层，以强制执行特定于任务的特征进行推理，并减少与任务无关但敏感的特征以保护隐私。为了实现面向隐私的剪枝，PATR引入了两个关键部分：Lipschitz正则化和对抗性重建训练，它们分别通过降低MIA的稳定性来增加重建误差，并通过对抗性训练来增强目标推理模型。在一个真实世界的协同推理任务--车辆重新识别上，我们展示了巡逻在对抗MIA方面的优越性能。



## **39. Contractive Systems Improve Graph Neural Networks Against Adversarial Attacks**

收缩系统对图神经网络抗敌意攻击的改进 cs.LG

**SubmitDate**: 2023-11-12    [abs](http://arxiv.org/abs/2311.06942v1) [paper-pdf](http://arxiv.org/pdf/2311.06942v1)

**Authors**: Moshe Eliasof, Davide Murari, Ferdia Sherry, Carola-Bibiane Schönlieb

**Abstract**: Graph Neural Networks (GNNs) have established themselves as a key component in addressing diverse graph-based tasks. Despite their notable successes, GNNs remain susceptible to input perturbations in the form of adversarial attacks. This paper introduces an innovative approach to fortify GNNs against adversarial perturbations through the lens of contractive dynamical systems. Our method introduces graph neural layers based on differential equations with contractive properties, which, as we show, improve the robustness of GNNs. A distinctive feature of the proposed approach is the simultaneous learned evolution of both the node features and the adjacency matrix, yielding an intrinsic enhancement of model robustness to perturbations in the input features and the connectivity of the graph. We mathematically derive the underpinnings of our novel architecture and provide theoretical insights to reason about its expected behavior. We demonstrate the efficacy of our method through numerous real-world benchmarks, reading on par or improved performance compared to existing methods.

摘要: 图形神经网络(GNN)已经成为解决各种基于图形的任务的关键组件。尽管GNN取得了显著的成功，但它们仍然容易受到对抗性攻击形式的投入扰动的影响。本文介绍了一种通过压缩动力系统的透镜来增强GNN抵抗敌意扰动的创新方法。我们的方法引入了基于具有压缩性质的微分方程的图神经层，从而提高了GNN的稳健性。该方法的一个显著特点是节点特征和邻接矩阵的同时学习进化，从而内在地增强了模型对输入特征扰动和图的连通性的稳健性。我们从数学上推导出我们的新体系结构的基础，并提供理论见解来推理其预期行为。我们通过许多真实世界的基准测试来证明我们的方法的有效性，与现有的方法相比，我们的阅读是平分的，或者是性能有所提高。



## **40. Facial Data Minimization: Shallow Model as Your Privacy Filter**

面部数据最小化：浅层模型作为您的隐私过滤器 cs.CR

14 pages, 11 figures

**SubmitDate**: 2023-11-12    [abs](http://arxiv.org/abs/2310.15590v2) [paper-pdf](http://arxiv.org/pdf/2310.15590v2)

**Authors**: Yuwen Pu, Jiahao Chen, Jiayu Pan, Hao li, Diqun Yan, Xuhong Zhang, Shouling Ji

**Abstract**: Face recognition service has been used in many fields and brings much convenience to people. However, once the user's facial data is transmitted to a service provider, the user will lose control of his/her private data. In recent years, there exist various security and privacy issues due to the leakage of facial data. Although many privacy-preserving methods have been proposed, they usually fail when they are not accessible to adversaries' strategies or auxiliary data. Hence, in this paper, by fully considering two cases of uploading facial images and facial features, which are very typical in face recognition service systems, we proposed a data privacy minimization transformation (PMT) method. This method can process the original facial data based on the shallow model of authorized services to obtain the obfuscated data. The obfuscated data can not only maintain satisfactory performance on authorized models and restrict the performance on other unauthorized models but also prevent original privacy data from leaking by AI methods and human visual theft. Additionally, since a service provider may execute preprocessing operations on the received data, we also propose an enhanced perturbation method to improve the robustness of PMT. Besides, to authorize one facial image to multiple service models simultaneously, a multiple restriction mechanism is proposed to improve the scalability of PMT. Finally, we conduct extensive experiments and evaluate the effectiveness of the proposed PMT in defending against face reconstruction, data abuse, and face attribute estimation attacks. These experimental results demonstrate that PMT performs well in preventing facial data abuse and privacy leakage while maintaining face recognition accuracy.

摘要: 人脸识别服务已经在许多领域得到了应用，给人们带来了极大的便利。然而，一旦用户的面部数据被传输到服务提供商，用户将失去对他/她的私人数据的控制。近年来，由于人脸数据的泄露，存在着各种各样的安全和隐私问题。虽然已经提出了许多隐私保护方法，但当对手的策略或辅助数据无法访问时，这些方法通常会失败。因此，在本文中，通过充分考虑人脸识别服务系统中非常典型的两种上传人脸图像和人脸特征的情况，提出了一种数据隐私最小化转换(PMT)方法。该方法可以基于授权服务的浅模型对原始人脸数据进行处理，得到混淆后的数据。混淆后的数据不仅可以在授权模型上保持满意的性能，在其他非授权模型上也可以限制性能，还可以防止AI方法泄露原始隐私数据和人类视觉窃取。此外，由于服务提供商可以对接收到的数据执行预处理操作，我们还提出了一种增强的扰动方法来提高PMT的稳健性。此外，为了将一幅人脸图像同时授权给多个服务模型，提出了一种多约束机制来提高PMT的可扩展性。最后，我们进行了大量的实验，评估了提出的PMT在抵抗人脸重建、数据滥用和人脸属性估计攻击方面的有效性。这些实验结果表明，PMT在保持人脸识别准确率的同时，很好地防止了人脸数据的滥用和隐私泄露。



## **41. Learning Globally Optimized Language Structure via Adversarial Training**

通过对抗性训练学习全局优化的语言结构 cs.CL

**SubmitDate**: 2023-11-12    [abs](http://arxiv.org/abs/2311.06771v1) [paper-pdf](http://arxiv.org/pdf/2311.06771v1)

**Authors**: Xuwang Yin

**Abstract**: Recent work has explored integrating autoregressive language models with energy-based models (EBMs) to enhance text generation capabilities. However, learning effective EBMs for text is challenged by the discrete nature of language. This work proposes an adversarial training strategy to address limitations in prior efforts. Specifically, an iterative adversarial attack algorithm is presented to generate negative samples for training the EBM by perturbing text from the autoregressive model. This aims to enable the EBM to suppress spurious modes outside the support of the data distribution. Experiments on an arithmetic sequence generation task demonstrate that the proposed adversarial training approach can substantially enhance the quality of generated sequences compared to prior methods. The results highlight the promise of adversarial techniques to improve discrete EBM training. Key contributions include: (1) an adversarial attack strategy tailored to text to generate negative samples, circumventing MCMC limitations; (2) an adversarial training algorithm for EBMs leveraging these attacks; (3) empirical validation of performance improvements on a sequence generation task.

摘要: 最近的工作探索了将自回归语言模型与基于能量的模型(EBM)相结合以增强文本生成能力。然而，学习有效的针对文本的循证模式受到了语言的离散性质的挑战。这项工作提出了一种对抗性训练战略，以解决先前努力中的局限性。具体地说，提出了一种迭代对抗攻击算法，通过扰动自回归模型中的文本来生成用于训练EBM的负样本。这旨在使EBM能够抑制数据分布支持之外的虚假模式。在算术序列生成任务上的实验表明，与现有方法相比，所提出的对抗性训练方法可以显著提高生成序列的质量。这一结果突出了对抗性技术改善离散循证医学训练的前景。主要贡献包括：(1)针对文本定制的对抗性攻击策略，以生成负样本，绕过MCMC限制；(2)针对利用这些攻击的EBM的对抗性训练算法；(3)对序列生成任务的性能改进的经验验证。



## **42. Probabilistic and Semantic Descriptions of Image Manifolds and Their Applications**

图像流形的概率和语义描述及其应用 cs.CV

26 pages, 17 figures, 1 table, accepted to Frontiers in Computer  Science, 2023

**SubmitDate**: 2023-11-12    [abs](http://arxiv.org/abs/2307.02881v5) [paper-pdf](http://arxiv.org/pdf/2307.02881v5)

**Authors**: Peter Tu, Zhaoyuan Yang, Richard Hartley, Zhiwei Xu, Jing Zhang, Yiwei Fu, Dylan Campbell, Jaskirat Singh, Tianyu Wang

**Abstract**: This paper begins with a description of methods for estimating image probability density functions that reflects the observation that such data is usually constrained to lie in restricted regions of the high-dimensional image space-not every pattern of pixels is an image. It is common to say that images lie on a lower-dimensional manifold in the high-dimensional space. However, it is not the case that all points on the manifold have an equal probability of being images. Images are unevenly distributed on the manifold, and our task is to devise ways to model this distribution as a probability distribution. We therefore consider popular generative models. For our purposes, generative/probabilistic models should have the properties of 1) sample generation: the possibility to sample from this distribution with the modelled density function, and 2) probability computation: given a previously unseen sample from the dataset of interest, one should be able to compute its probability, at least up to a normalising constant. To this end, we investigate the use of methods such as normalising flow and diffusion models. We then show how semantic interpretations are used to describe points on the manifold. To achieve this, we consider an emergent language framework that uses variational encoders for a disentangled representation of points that reside on a given manifold. Trajectories between points on a manifold can then be described as evolving semantic descriptions. We also show that such probabilistic descriptions (bounded) can be used to improve semantic consistency by constructing defences against adversarial attacks. We evaluate our methods with improved semantic robustness and OoD detection capability, explainable and editable semantic interpolation, and improved classification accuracy under patch attacks. We also discuss the limitation in diffusion models.

摘要: 本文首先描述了用于估计图像概率密度函数的方法，该方法反映了这样的观察，即这种数据通常被限制在高维图像空间的受限区域-并不是每种像素模式都是图像。人们常说，图像位于高维空间中的低维流形上。然而，流形上的所有点成为图像的概率并不相等。图像在流形上是不均匀分布的，我们的任务是设计出将这种分布建模为概率分布的方法。因此，我们考虑流行的生成性模型。就我们的目的而言，生成/概率模型应该具有以下属性：1)样本生成：使用建模的密度函数从该分布中进行样本的可能性；以及2)概率计算：给定感兴趣的数据集中以前未见过的样本，应能够计算其概率，至少达到归一化常数。为此，我们研究了流和扩散模型等方法的使用。然后，我们展示了如何使用语义解释来描述流形上的点。为了实现这一点，我们考虑一种新的语言框架，它使用变分编码器来解开驻留在给定流形上的点的表示。然后，流形上的点之间的轨迹可以被描述为不断演变的语义描述。我们还表明，这种概率描述(有界)可以通过构建对对手攻击的防御来提高语义一致性。我们通过改进的语义健壮性和OOD检测能力、可解释和可编辑的语义内插以及在补丁攻击下改进的分类精度来评估我们的方法。我们还讨论了扩散模型的局限性。



## **43. Privacy Risks Analysis and Mitigation in Federated Learning for Medical Images**

医学图像联合学习中的隐私风险分析与缓解 cs.LG

V1

**SubmitDate**: 2023-11-11    [abs](http://arxiv.org/abs/2311.06643v1) [paper-pdf](http://arxiv.org/pdf/2311.06643v1)

**Authors**: Badhan Chandra Das, M. Hadi Amini, Yanzhao Wu

**Abstract**: Federated learning (FL) is gaining increasing popularity in the medical domain for analyzing medical images, which is considered an effective technique to safeguard sensitive patient data and comply with privacy regulations. However, several recent studies have revealed that the default settings of FL may leak private training data under privacy attacks. Thus, it is still unclear whether and to what extent such privacy risks of FL exist in the medical domain, and if so, ``how to mitigate such risks?''. In this paper, first, we propose a holistic framework for Medical data Privacy risk analysis and mitigation in Federated Learning (MedPFL) to analyze privacy risks and develop effective mitigation strategies in FL for protecting private medical data. Second, we demonstrate the substantial privacy risks of using FL to process medical images, where adversaries can easily perform privacy attacks to reconstruct private medical images accurately. Third, we show that the defense approach of adding random noises may not always work effectively to protect medical images against privacy attacks in FL, which poses unique and pressing challenges associated with medical data for privacy protection.

摘要: 联合学习(FL)在医学领域的医学图像分析中越来越受欢迎，它被认为是保护敏感患者数据和遵守隐私法规的有效技术。然而，最近的一些研究表明，在隐私攻击下，FL的默认设置可能会泄露私人训练数据。因此，目前仍不清楚FL在医疗领域是否存在这种隐私风险，以及在多大程度上存在这种风险，如果存在，“如何减轻这种风险？”本文首先提出了一种联邦学习中医疗数据隐私风险分析和缓解的整体框架(MedPFL)，以分析联邦学习中的隐私风险，并在FL中制定有效的缓解策略来保护私人医疗数据。其次，我们展示了使用FL处理医学图像的巨大隐私风险，在这种情况下，攻击者可以很容易地执行隐私攻击来准确重建私人医学图像。第三，我们发现，在FL中添加随机噪声的防御方法并不总是有效地保护医学图像免受隐私攻击，这对隐私保护提出了与医疗数据相关的独特而紧迫的挑战。



## **44. Verifiable Learning for Robust Tree Ensembles**

用于稳健树集成的可验证学习 cs.LG

19 pages, 5 figures; full version of the revised paper accepted at  ACM CCS 2023 with corrected typo in footnote 1

**SubmitDate**: 2023-11-11    [abs](http://arxiv.org/abs/2305.03626v4) [paper-pdf](http://arxiv.org/pdf/2305.03626v4)

**Authors**: Stefano Calzavara, Lorenzo Cazzaro, Giulio Ermanno Pibiri, Nicola Prezza

**Abstract**: Verifying the robustness of machine learning models against evasion attacks at test time is an important research problem. Unfortunately, prior work established that this problem is NP-hard for decision tree ensembles, hence bound to be intractable for specific inputs. In this paper, we identify a restricted class of decision tree ensembles, called large-spread ensembles, which admit a security verification algorithm running in polynomial time. We then propose a new approach called verifiable learning, which advocates the training of such restricted model classes which are amenable for efficient verification. We show the benefits of this idea by designing a new training algorithm that automatically learns a large-spread decision tree ensemble from labelled data, thus enabling its security verification in polynomial time. Experimental results on public datasets confirm that large-spread ensembles trained using our algorithm can be verified in a matter of seconds, using standard commercial hardware. Moreover, large-spread ensembles are more robust than traditional ensembles against evasion attacks, at the cost of an acceptable loss of accuracy in the non-adversarial setting.

摘要: 验证机器学习模型在测试时对逃避攻击的稳健性是一个重要的研究问题。不幸的是，以前的工作确定了这个问题对于决策树集成来说是NP-Hard的，因此对于特定的输入必然是棘手的。在本文中，我们识别了一类受限的决策树集成，称为大分布集成，它允许安全验证算法在多项式时间内运行。然后，我们提出了一种新的方法，称为可验证学习，它主张训练这样的受限模型类，这些模型类适合于有效的验证。我们通过设计一种新的训练算法，从标记数据中自动学习大规模决策树集成，从而在多项式时间内实现其安全性验证，从而展示了这种思想的好处。在公共数据集上的实验结果证实，使用我们的算法训练的大范围集成可以在几秒钟内使用标准的商业硬件进行验证。此外，大范围的合奏比传统的合奏更能抵抗躲避攻击，代价是在非对抗性环境中损失可接受的准确性。



## **45. Seeing is Believing: A Federated Learning Based Prototype to Detect Wireless Injection Attacks**

眼见为实：一种基于联合学习的无线注入攻击检测原型 cs.IT

6 pages with 8 figures

**SubmitDate**: 2023-11-11    [abs](http://arxiv.org/abs/2311.06564v1) [paper-pdf](http://arxiv.org/pdf/2311.06564v1)

**Authors**: Aadil Hussain, Nitheesh Gundapu, Sarang Drugkar, Suraj Kiran, J. Harshan, Ranjitha Prasad

**Abstract**: Reactive injection attacks are a class of security threats in wireless networks wherein adversaries opportunistically inject spoofing packets in the frequency band of a client thereby forcing the base-station to deploy impersonation-detection methods. Towards circumventing such threats, we implement secret-key based physical-layer signalling methods at the clients which allow the base-stations to deploy machine learning (ML) models on their in-phase and quadrature samples at the baseband for attack detection. Using Adalm Pluto based software defined radios to implement the secret-key based signalling methods, we show that robust ML models can be designed at the base-stations. However, we also point out that, in practice, insufficient availability of training datasets at the base-stations can make these methods ineffective. Thus, we use a federated learning framework in the backhaul network, wherein a group of base-stations that need to protect their clients against reactive injection threats collaborate to refine their ML models by ensuring privacy on their datasets. Using a network of XBee devices to implement the backhaul network, experimental results on our federated learning setup shows significant enhancements in the detection accuracy, thus presenting wireless security as an excellent use-case for federated learning in 6G networks and beyond.

摘要: 反应性注入攻击是无线网络中的一类安全威胁，攻击者在客户端的频段内机会性地注入欺骗分组，从而迫使基站部署模拟检测方法。为了规避这种威胁，我们在客户端实现了基于密钥的物理层信令方法，允许基站在其基带上的同相和正交样本上部署机器学习(ML)模型，以用于攻击检测。使用基于Adalm Pluto的软件定义的无线电来实现基于密钥的信令方法，我们证明了在基站处可以设计出稳健的ML模型。然而，我们也指出，在实践中，基站训练数据集的可用性不足会使这些方法失效。因此，我们在回程网络中使用联合学习框架，其中需要保护其客户端免受反应性注入威胁的一组基站协作，通过确保其数据集的隐私来改进其ML模型。使用XBee设备网络实施回程网络，在我们的联合学习设置上的实验结果显示，检测精度显著增强，从而将无线安全作为6G网络及更高级别的联合学习的优秀用例。



## **46. Flatness-aware Adversarial Attack**

平坦度感知的对抗性攻击 cs.LG

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.06423v1) [paper-pdf](http://arxiv.org/pdf/2311.06423v1)

**Authors**: Mingyuan Fan, Xiaodan Li, Cen Chen, Yinggui Wang

**Abstract**: The transferability of adversarial examples can be exploited to launch black-box attacks. However, adversarial examples often present poor transferability. To alleviate this issue, by observing that the diversity of inputs can boost transferability, input regularization based methods are proposed, which craft adversarial examples by combining several transformed inputs. We reveal that input regularization based methods make resultant adversarial examples biased towards flat extreme regions. Inspired by this, we propose an attack called flatness-aware adversarial attack (FAA) which explicitly adds a flatness-aware regularization term in the optimization target to promote the resultant adversarial examples towards flat extreme regions. The flatness-aware regularization term involves gradients of samples around the resultant adversarial examples but optimizing gradients requires the evaluation of Hessian matrix in high-dimension spaces which generally is intractable. To address the problem, we derive an approximate solution to circumvent the construction of Hessian matrix, thereby making FAA practical and cheap. Extensive experiments show the transferability of adversarial examples crafted by FAA can be considerably boosted compared with state-of-the-art baselines.

摘要: 对抗性例子的可转移性可被利用来发动黑盒攻击。然而，对抗性的例子往往表现出较差的可转移性。为了缓解这一问题，通过观察到输入的多样性可以提高可转移性，提出了基于输入正则化的方法，该方法通过组合几个转换后的输入来制作对抗性例子。我们发现，基于输入正则化的方法使得合成的对抗性示例偏向于平坦的极值区域。受此启发，我们提出了一种称为平坦度感知对抗性攻击(FAA)的攻击方法，它在优化目标中显式地增加了平坦度感知正则化项，将生成的对抗性实例推向平坦的极值区域。平坦度感知正则化项涉及所得到的对抗性样本周围的样本的梯度，但优化梯度需要在高维空间中计算Hessian矩阵，这通常是困难的。为了解决这个问题，我们给出了一个绕过Hessian矩阵构造的近似解，从而使FAA变得实用和廉价。广泛的实验表明，与最先进的基线相比，FAA制作的对抗性例子的可转移性可以大大提高。



## **47. CALLOC: Curriculum Adversarial Learning for Secure and Robust Indoor Localization**

CALLOC：安全可靠的室内本地化课程对抗性学习 cs.LG

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.06361v1) [paper-pdf](http://arxiv.org/pdf/2311.06361v1)

**Authors**: Danish Gufran, Sudeep Pasricha

**Abstract**: Indoor localization has become increasingly vital for many applications from tracking assets to delivering personalized services. Yet, achieving pinpoint accuracy remains a challenge due to variations across indoor environments and devices used to assist with localization. Another emerging challenge is adversarial attacks on indoor localization systems that not only threaten service integrity but also reduce localization accuracy. To combat these challenges, we introduce CALLOC, a novel framework designed to resist adversarial attacks and variations across indoor environments and devices that reduce system accuracy and reliability. CALLOC employs a novel adaptive curriculum learning approach with a domain specific lightweight scaled-dot product attention neural network, tailored for adversarial and variation resilience in practical use cases with resource constrained mobile devices. Experimental evaluations demonstrate that CALLOC can achieve improvements of up to 6.03x in mean error and 4.6x in worst-case error against state-of-the-art indoor localization frameworks, across diverse building floorplans, mobile devices, and adversarial attacks scenarios.

摘要: 室内本地化对于从跟踪资产到提供个性化服务的许多应用变得越来越重要。然而，由于室内环境和用于协助定位的设备的不同，实现精确定位仍然是一个挑战。另一个新出现的挑战是对室内定位系统的对抗性攻击，这不仅威胁到服务完整性，而且还降低了定位精度。为了应对这些挑战，我们引入了CALLOC，这是一个新的框架，旨在抵御室内环境和设备中降低系统准确性和可靠性的对抗性攻击和变化。CALLOC采用了一种新的自适应课程学习方法和领域特定的轻量级缩放点乘积注意力神经网络，该网络为资源受限的移动设备的实际使用案例中的对抗性和变化弹性量身定做。实验评估表明，与最先进的室内定位框架相比，在不同的建筑平面图、移动设备和对抗性攻击场景中，CALLOC可以获得高达6.03倍的平均误差和4.6倍的最差误差改进。



## **48. SneakyPrompt: Jailbreaking Text-to-image Generative Models**

SneakyPrompt：越狱文本到图像生成模型 cs.LG

To appear in the Proceedings of the IEEE Symposium on Security and  Privacy (Oakland), 2024

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2305.12082v3) [paper-pdf](http://arxiv.org/pdf/2305.12082v3)

**Authors**: Yuchen Yang, Bo Hui, Haolin Yuan, Neil Gong, Yinzhi Cao

**Abstract**: Text-to-image generative models such as Stable Diffusion and DALL$\cdot$E raise many ethical concerns due to the generation of harmful images such as Not-Safe-for-Work (NSFW) ones. To address these ethical concerns, safety filters are often adopted to prevent the generation of NSFW images. In this work, we propose SneakyPrompt, the first automated attack framework, to jailbreak text-to-image generative models such that they generate NSFW images even if safety filters are adopted. Given a prompt that is blocked by a safety filter, SneakyPrompt repeatedly queries the text-to-image generative model and strategically perturbs tokens in the prompt based on the query results to bypass the safety filter. Specifically, SneakyPrompt utilizes reinforcement learning to guide the perturbation of tokens. Our evaluation shows that SneakyPrompt successfully jailbreaks DALL$\cdot$E 2 with closed-box safety filters to generate NSFW images. Moreover, we also deploy several state-of-the-art, open-source safety filters on a Stable Diffusion model. Our evaluation shows that SneakyPrompt not only successfully generates NSFW images, but also outperforms existing text adversarial attacks when extended to jailbreak text-to-image generative models, in terms of both the number of queries and qualities of the generated NSFW images. SneakyPrompt is open-source and available at this repository: \url{https://github.com/Yuchen413/text2image_safety}.

摘要: 文本到图像生成模型，如稳定扩散和DALL$\cdot$E提出了许多道德问题，由于有害的图像，如不安全的工作（NSFW）的生成。为了解决这些伦理问题，通常采用安全过滤器来防止生成NSFW图像。在这项工作中，我们提出了第一个自动化攻击框架SneakyPrompt，以越狱文本到图像生成模型，即使采用安全过滤器，它们也会生成NSFW图像。给定被安全过滤器阻止的提示，SneakyPrompt重复查询文本到图像生成模型，并基于查询结果策略性地扰动提示中的令牌以绕过安全过滤器。具体来说，SneakyPrompt利用强化学习来指导令牌的扰动。我们的评估表明，SneakyPrompt成功越狱DALL$\cdot$E 2与封闭式安全过滤器生成NSFW图像。此外，我们还在稳定扩散模型上部署了几个最先进的开源安全过滤器。我们的评估表明，SneakyPrompt不仅成功地生成了NSFW图像，而且在扩展到越狱文本到图像生成模型时，在查询数量和生成的NSFW图像质量方面都优于现有的文本对抗攻击。SneakyPrompt是开源的，可在以下存储库中找到：\url{https：//github.com/Yuchen413/text2image_safety}。



## **49. Triad: Trusted Timestamps in Untrusted Environments**

Triad：不可信环境中的可信时间戳 cs.CR

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.06156v1) [paper-pdf](http://arxiv.org/pdf/2311.06156v1)

**Authors**: Gabriel P. Fernandez, Andrey Brito, Christof Fetzer

**Abstract**: We aim to provide trusted time measurement mechanisms to applications and cloud infrastructure deployed in environments that could harbor potential adversaries, including the hardware infrastructure provider. Despite Trusted Execution Environments (TEEs) providing multiple security functionalities, timestamps from the Operating System are not covered. Nevertheless, some services require time for validating permissions or ordering events. To address that need, we introduce Triad, a trusted timestamp dispatcher of time readings. The solution provides trusted timestamps enforced by mutually supportive enclave-based clock servers that create a continuous trusted timeline. We leverage enclave properties such as forced exits and CPU-based counters to mitigate attacks on the server's timestamp counters. Triad produces trusted, confidential, monotonically-increasing timestamps with bounded error and desirable, non-trivial properties. Our implementation relies on Intel SGX and SCONE, allowing transparent usage. We evaluate Triad's error and behavior in multiple dimensions.

摘要: 我们的目标是为部署在可能存在潜在对手(包括硬件基础设施提供商)的环境中的应用程序和云基础设施提供可信时间测量机制。尽管可信执行环境(TEE)提供了多种安全功能，但不包括来自操作系统的时间戳。然而，一些服务需要时间来验证权限或对事件进行排序。为了满足这一需求，我们引入了Triad，这是一个可信的时间戳读数分派器。该解决方案提供可信的时间戳，由相互支持的基于Enclave的时钟服务器执行，从而创建连续的可信时间线。我们利用强制退出和基于CPU的计数器等Enclave属性来减少对服务器时间戳计数器的攻击。Triad生成可信的、保密的、单调递增的时间戳，具有有界的误差和所需的非平凡属性。我们的实施依赖于Intel SGX和Scon，允许透明使用。我们从多个维度评估三合会的错误和行为。



## **50. Fight Fire with Fire: Combating Adversarial Patch Attacks using Pattern-randomized Defensive Patches**

以火还击：使用模式随机防御补丁对抗对抗性补丁攻击 cs.CV

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.06122v1) [paper-pdf](http://arxiv.org/pdf/2311.06122v1)

**Authors**: Jianan Feng, Jiachun Li, Changqing Miao, Jianjun Huang, Wei You, Wenchang Shi, Bin Liang

**Abstract**: Object detection has found extensive applications in various tasks, but it is also susceptible to adversarial patch attacks. Existing defense methods often necessitate modifications to the target model or result in unacceptable time overhead. In this paper, we adopt a counterattack approach, following the principle of "fight fire with fire," and propose a novel and general methodology for defending adversarial attacks. We utilize an active defense strategy by injecting two types of defensive patches, canary and woodpecker, into the input to proactively probe or weaken potential adversarial patches without altering the target model. Moreover, inspired by randomization techniques employed in software security, we employ randomized canary and woodpecker injection patterns to defend against defense-aware attacks. The effectiveness and practicality of the proposed method are demonstrated through comprehensive experiments. The results illustrate that canary and woodpecker achieve high performance, even when confronted with unknown attack methods, while incurring limited time overhead. Furthermore, our method also exhibits sufficient robustness against defense-aware attacks, as evidenced by adaptive attack experiments.

摘要: 目标检测在各种任务中得到了广泛的应用，但它也容易受到对抗性补丁的攻击。现有的防御方法通常需要对目标模型进行修改，或者导致不可接受的时间开销。本文采用反击的方法，遵循“以火还火”的原则，提出了一种新颖的、通用的防御对抗性攻击的方法。我们利用主动防御策略，在输入中注入两种类型的防御补丁，金丝雀和啄木鸟，在不改变目标模型的情况下主动探测或削弱潜在的敌方补丁。此外，受软件安全中使用的随机化技术的启发，我们使用随机化的金丝雀和啄木鸟注入模式来防御防御感知攻击。通过综合实验，验证了该方法的有效性和实用性。实验结果表明，金丝雀和啄木鸟在攻击方式未知的情况下也能获得较高的性能，同时具有有限的时间开销。此外，自适应攻击实验表明，该方法对防御感知攻击也表现出了足够的鲁棒性。



