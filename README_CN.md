# Latest Adversarial Attack Papers
**update at 2023-11-29 10:29:29**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Scalable Extraction of Training Data from (Production) Language Models**

从(产生式)语言模型中可伸缩地提取训练数据 cs.LG

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.17035v1) [paper-pdf](http://arxiv.org/pdf/2311.17035v1)

**Authors**: Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Eric Wallace, Florian Tramèr, Katherine Lee

**Abstract**: This paper studies extractable memorization: training data that an adversary can efficiently extract by querying a machine learning model without prior knowledge of the training dataset. We show an adversary can extract gigabytes of training data from open-source language models like Pythia or GPT-Neo, semi-open models like LLaMA or Falcon, and closed models like ChatGPT. Existing techniques from the literature suffice to attack unaligned models; in order to attack the aligned ChatGPT, we develop a new divergence attack that causes the model to diverge from its chatbot-style generations and emit training data at a rate 150x higher than when behaving properly. Our methods show practical attacks can recover far more data than previously thought, and reveal that current alignment techniques do not eliminate memorization.

摘要: 本文研究了可提取记忆：对手可以通过查询机器学习模型有效地提取训练数据，而不需要事先知道训练数据集。我们展示了对手可以从开源语言模型(如Pythia或GPT-Neo)、半开放模型(如骆驼或猎鹰)以及封闭式模型(如ChatGPT)中提取千兆字节的训练数据。现有的文献技术足以攻击未对齐的模型；为了攻击对齐的ChatGPT，我们开发了一种新的发散攻击，该攻击导致模型偏离其聊天机器人风格的代，并以比正常行为高150倍的速率发出训练数据。我们的方法表明，实际的攻击可以恢复比之前认为的更多的数据，并揭示了当前的比对技术并没有消除记忆。



## **2. Breaking Boundaries: Balancing Performance and Robustness in Deep Wireless Traffic Forecasting**

打破界限：深度无线流量预测中的性能和稳健性平衡 cs.LG

Accepted for presentation at the ARTMAN workshop, part of the ACM  Conference on Computer and Communications Security (CCS), 2023

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.09790v3) [paper-pdf](http://arxiv.org/pdf/2311.09790v3)

**Authors**: Romain Ilbert, Thai V. Hoang, Zonghua Zhang, Themis Palpanas

**Abstract**: Balancing the trade-off between accuracy and robustness is a long-standing challenge in time series forecasting. While most of existing robust algorithms have achieved certain suboptimal performance on clean data, sustaining the same performance level in the presence of data perturbations remains extremely hard. In this paper, we study a wide array of perturbation scenarios and propose novel defense mechanisms against adversarial attacks using real-world telecom data. We compare our strategy against two existing adversarial training algorithms under a range of maximal allowed perturbations, defined using $\ell_{\infty}$-norm, $\in [0.1,0.4]$. Our findings reveal that our hybrid strategy, which is composed of a classifier to detect adversarial examples, a denoiser to eliminate noise from the perturbed data samples, and a standard forecaster, achieves the best performance on both clean and perturbed data. Our optimal model can retain up to $92.02\%$ the performance of the original forecasting model in terms of Mean Squared Error (MSE) on clean data, while being more robust than the standard adversarially trained models on perturbed data. Its MSE is 2.71$\times$ and 2.51$\times$ lower than those of comparing methods on normal and perturbed data, respectively. In addition, the components of our models can be trained in parallel, resulting in better computational efficiency. Our results indicate that we can optimally balance the trade-off between the performance and robustness of forecasting models by improving the classifier and denoiser, even in the presence of sophisticated and destructive poisoning attacks.

摘要: 平衡准确性和鲁棒性之间的权衡是时间序列预测中的一个长期挑战。虽然大多数现有的鲁棒算法已经在干净数据上实现了某些次优性能，但在存在数据扰动的情况下保持相同的性能水平仍然非常困难。在本文中，我们研究了各种各样的扰动场景，并使用真实世界的电信数据提出了对抗性攻击的新防御机制。我们将我们的策略与两个现有的对抗性训练算法在最大允许扰动范围内进行比较，使用$\ell_{\infty}$-norm，$\in [0.1，0.4]$定义。我们的研究结果表明，我们的混合策略，它是由一个分类器来检测对抗性的例子，一个去噪器，以消除干扰的数据样本中的噪声，和一个标准的预测器，实现了最好的性能在干净和扰动的数据。我们的最佳模型可以保留高达92.02\%$的原始预测模型的性能方面的均方误差（MSE）干净的数据，而更强大的干扰数据比标准的对抗训练模型。在正态和扰动数据下，其均方误差分别比比较方法低2.71倍和2.51倍。此外，我们模型的组件可以并行训练，从而提高计算效率。我们的研究结果表明，我们可以通过改进分类器和去噪器来最佳地平衡预测模型的性能和鲁棒性之间的权衡，即使在存在复杂和破坏性的中毒攻击的情况下。



## **3. Generation of Games for Opponent Model Differentiation**

用于对手模型区分的对策生成 cs.AI

4 pages

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16781v1) [paper-pdf](http://arxiv.org/pdf/2311.16781v1)

**Authors**: David Milec, Viliam Lisý, Christopher Kiekintveld

**Abstract**: Protecting against adversarial attacks is a common multiagent problem. Attackers in the real world are predominantly human actors, and the protection methods often incorporate opponent models to improve the performance when facing humans. Previous results show that modeling human behavior can significantly improve the performance of the algorithms. However, modeling humans correctly is a complex problem, and the models are often simplified and assume humans make mistakes according to some distribution or train parameters for the whole population from which they sample. In this work, we use data gathered by psychologists who identified personality types that increase the likelihood of performing malicious acts. However, in the previous work, the tests on a handmade game could not show strategic differences between the models. We created a novel model that links its parameters to psychological traits. We optimized over parametrized games and created games in which the differences are profound. Our work can help with automatic game generation when we need a game in which some models will behave differently and to identify situations in which the models do not align.

摘要: 防御敌意攻击是一个常见的多智能体问题。现实世界中的攻击者主要是人类演员，保护方法通常会结合对手模型来提高面对人类时的性能。以往的研究结果表明，对人的行为进行建模可以显著提高算法的性能。然而，正确地为人类建模是一个复杂的问题，模型往往被简化，并假设人类根据样本所在总体的某些分布或训练参数出错。在这项工作中，我们使用了心理学家收集的数据，他们确定了增加实施恶意行为可能性的个性类型。然而，在之前的工作中，对手工游戏的测试无法显示模型之间的战略差异。我们创建了一个新的模型，将其参数与心理特征联系起来。我们对参数化游戏进行了优化，并创建了差异巨大的游戏。当我们需要一个游戏时，我们的工作可以帮助自动生成游戏，在其中一些模型会有不同的行为，并识别模型不一致的情况。



## **4. Cooperative Abnormal Node Detection with Adversary Resistance: A Probabilistic Approach**

一种基于概率的抗粘连协作异常节点检测方法 eess.SY

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16661v1) [paper-pdf](http://arxiv.org/pdf/2311.16661v1)

**Authors**: Yingying Huangfu, Tian Bai

**Abstract**: This paper presents a novel probabilistic detection scheme called Cooperative Statistical Detection (CSD) for abnormal node detection while defending against adversarial attacks in cluster-tree networks. The CSD performs a two-phase process: 1) designing a likelihood ratio test (LRT) for a non-root node at its children from the perspective of packet loss; 2) making an overall decision at the root node based on the aggregated detection data of the nodes over tree branches. In most adversarial scenarios, malicious children knowing the detection policy can generate falsified data to protect the abnormal parent from being detected or frame its normal parent as an anomalous node. To resolve this issue, a modified Z-score-based falsification-resistant mechanism is presented in the CSD to remove untrustworthy information. Through theoretical analysis, we show that the LRT-based method achieves perfect detection, i.e., both the false alarm and missed detection probabilities decay exponentially to zero. Furthermore, the optimal removal threshold of the modified Z-score method is derived for falsifications with uncertain strategies and guarantees perfect detection of the CSD. As our simulation results show, the CSD approach is robust to falsifications and can rapidly reach $99\%$ detection accuracy, even in existing adversarial scenarios, which outperforms state-of-the-art technology.

摘要: 本文提出了一种新的概率检测方案，称为合作统计检测（CSD）的异常节点检测，同时抵御敌对攻击的簇树网络。CSD执行两阶段过程：1）从分组丢失的角度设计非根节点在其子节点处的似然比测试（LRT）; 2）基于树分支上的节点的聚合检测数据在根节点处做出总体决策。在大多数对抗性场景中，知道检测策略的恶意子节点可以生成伪造的数据，以保护异常父节点不被检测到，或者将其正常父节点框定为异常节点。为了解决这个问题，在CSD中提出了一种改进的基于Z分数的防篡改机制，以去除不可信的信息。通过理论分析，我们表明，基于LRT的方法实现了完美的检测，即，虚警概率和漏检概率都指数衰减到零。此外，修改后的Z分数方法的最佳去除阈值推导出不确定策略的伪造，并保证完美的检测CSD。正如我们的模拟结果表明，CSD方法是强大的伪造，可以迅速达到99\%$的检测精度，即使在现有的对抗性的情况下，这优于国家的最先进的技术。



## **5. On the Role of Randomization in Adversarially Robust Classification**

论随机化在逆稳性分类中的作用 cs.LG

10 pages main paper (27 total), 2 figures in main paper. Neurips 2023

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2302.07221v3) [paper-pdf](http://arxiv.org/pdf/2302.07221v3)

**Authors**: Lucas Gnecco-Heredia, Yann Chevaleyre, Benjamin Negrevergne, Laurent Meunier, Muni Sreenivas Pydi

**Abstract**: Deep neural networks are known to be vulnerable to small adversarial perturbations in test data. To defend against adversarial attacks, probabilistic classifiers have been proposed as an alternative to deterministic ones. However, literature has conflicting findings on the effectiveness of probabilistic classifiers in comparison to deterministic ones. In this paper, we clarify the role of randomization in building adversarially robust classifiers. Given a base hypothesis set of deterministic classifiers, we show the conditions under which a randomized ensemble outperforms the hypothesis set in adversarial risk, extending previous results. Additionally, we show that for any probabilistic binary classifier (including randomized ensembles), there exists a deterministic classifier that outperforms it. Finally, we give an explicit description of the deterministic hypothesis set that contains such a deterministic classifier for many types of commonly used probabilistic classifiers, i.e. randomized ensembles and parametric/input noise injection.

摘要: 众所周知，深度神经网络在测试数据中容易受到微小的对抗性扰动。为了防御敌意攻击，人们提出了概率分类器作为确定性分类器的替代方案。然而，与确定性分类器相比，文献对概率分类器的有效性有相互矛盾的发现。在这篇文章中，我们阐明了随机化在构建对抗性稳健分类器中的作用。在给定确定性分类器的基本假设集的情况下，我们证明了随机化集成在对抗风险方面优于假设集的条件，扩展了先前的结果。此外，我们还证明了对于任何概率二进制分类器(包括随机集成)，都存在一个性能优于它的确定性分类器。最后，我们对于许多常用的概率分类器，即随机集成和参数/输入噪声注入，给出了包含这种确定性分类器的确定性假设集的显式描述。



## **6. Efficient Key-Based Adversarial Defense for ImageNet by Using Pre-trained Model**

基于预训练模型的ImageNet密钥对抗防御 cs.CV

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16577v1) [paper-pdf](http://arxiv.org/pdf/2311.16577v1)

**Authors**: AprilPyone MaungMaung, Isao Echizen, Hitoshi Kiya

**Abstract**: In this paper, we propose key-based defense model proliferation by leveraging pre-trained models and utilizing recent efficient fine-tuning techniques on ImageNet-1k classification. First, we stress that deploying key-based models on edge devices is feasible with the latest model deployment advancements, such as Apple CoreML, although the mainstream enterprise edge artificial intelligence (Edge AI) has been focused on the Cloud. Then, we point out that the previous key-based defense on on-device image classification is impractical for two reasons: (1) training many classifiers from scratch is not feasible, and (2) key-based defenses still need to be thoroughly tested on large datasets like ImageNet. To this end, we propose to leverage pre-trained models and utilize efficient fine-tuning techniques to proliferate key-based models even on limited computing resources. Experiments were carried out on the ImageNet-1k dataset using adaptive and non-adaptive attacks. The results show that our proposed fine-tuned key-based models achieve a superior classification accuracy (more than 10% increase) compared to the previous key-based models on classifying clean and adversarial examples.

摘要: 在本文中，我们通过利用预先训练的模型和利用最新的高效微调技术对ImageNet-1k分类进行微调，提出了基于密钥的防御模型扩散。首先，我们强调，尽管主流企业边缘人工智能(Edge AI)一直专注于云，但随着Apple CoreML等最新模型部署的进步，在边缘设备上部署基于密钥的模型是可行的。然后，我们指出基于密钥的防御在设备上的图像分类是不现实的，原因有两个：(1)从头开始训练许多分类器是不可行的；(2)基于密钥的防御仍然需要在像ImageNet这样的大型数据集上进行彻底的测试。为此，我们建议利用预先训练的模型并利用有效的微调技术来繁殖基于关键字的模型，即使在有限的计算资源上也是如此。使用自适应和非自适应攻击在ImageNet-1k数据集上进行了实验。实验结果表明，与已有的基于关键字的分类模型相比，本文提出的改进的基于关键字的模型具有更高的分类正确率(提高了10%以上)。



## **7. On the Robustness of Decision-Focused Learning**

决策学习的鲁棒性研究 cs.LG

17 pages, 45 figures, submitted to AAAI artificial intelligence for  operations research workshop

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16487v1) [paper-pdf](http://arxiv.org/pdf/2311.16487v1)

**Authors**: Yehya Farhat

**Abstract**: Decision-Focused Learning (DFL) is an emerging learning paradigm that tackles the task of training a machine learning (ML) model to predict missing parameters of an incomplete optimization problem, where the missing parameters are predicted. DFL trains an ML model in an end-to-end system, by integrating the prediction and optimization tasks, providing better alignment of the training and testing objectives. DFL has shown a lot of promise and holds the capacity to revolutionize decision-making in many real-world applications. However, very little is known about the performance of these models under adversarial attacks. We adopt ten unique DFL methods and benchmark their performance under two distinctly focused attacks adapted towards the Predict-then-Optimize problem setting. Our study proposes the hypothesis that the robustness of a model is highly correlated with its ability to find predictions that lead to optimal decisions without deviating from the ground-truth label. Furthermore, we provide insight into how to target the models that violate this condition and show how these models respond differently depending on the achieved optimality at the end of their training cycles.

摘要: 聚焦决策学习(DFL)是一种新兴的学习范式，它解决了训练机器学习(ML)模型来预测不完全优化问题的缺失参数的任务，其中缺失的参数被预测。DFL通过集成预测和优化任务，在端到端系统中训练ML模型，提供更好的训练和测试目标的一致性。DFL已经显示出了很大的潜力，并拥有在许多现实世界应用程序中彻底改变决策的能力。然而，人们对这些模型在对抗性攻击下的性能知之甚少。我们采用了十种独特的DFL方法，并对它们在两种针对预测-然后优化问题设置的明显集中的攻击下的性能进行了基准测试。我们的研究提出了这样的假设，即模型的稳健性与其在不偏离地面事实标签的情况下找到导致最优决策的预测的能力高度相关。此外，我们还提供了对如何针对违反这一条件的模型的洞察，并展示了这些模型如何根据在其训练周期结束时实现的最优化而做出不同的反应。



## **8. Threshold Breaker: Can Counter-Based RowHammer Prevention Mechanisms Truly Safeguard DRAM?**

阈值突破者：基于计数器的RowHammer预防机制能否真正保护DRAM？ cs.AR

7 pages, 6 figures

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16460v1) [paper-pdf](http://arxiv.org/pdf/2311.16460v1)

**Authors**: Ranyang Zhou, Jacqueline Liu, Sabbir Ahmed, Nakul Kochar, Adnan Siraj Rakin, Shaahin Angizi

**Abstract**: This paper challenges the existing victim-focused counter-based RowHammer detection mechanisms by experimentally demonstrating a novel multi-sided fault injection attack technique called Threshold Breaker. This mechanism can effectively bypass the most advanced counter-based defense mechanisms by soft-attacking the rows at a farther physical distance from the target rows. While no prior work has demonstrated the effect of such an attack, our work closes this gap by systematically testing 128 real commercial DDR4 DRAM products and reveals that the Threshold Breaker affects various chips from major DRAM manufacturers. As a case study, we compare the performance efficiency between our mechanism and a well-known double-sided attack by performing adversarial weight attacks on a modern Deep Neural Network (DNN). The results demonstrate that the Threshold Breaker can deliberately deplete the intelligence of the targeted DNN system while DRAM is fully protected.

摘要: 本文通过实验证明了一种新的多边故障注入攻击技术，称为阈值断路器，挑战现有的受害者为中心的计数器为基础的RowHammer检测机制。这种机制可以有效地绕过最先进的基于计数器的防御机制，对距离目标行较远的行进行软攻击。虽然之前没有工作证明这种攻击的影响，但我们的工作通过系统地测试128个真正的商用DDR4 DRAM产品来弥补这一差距，并揭示Threshold Breaker影响主要DRAM制造商的各种芯片。作为一个案例研究，我们通过对现代深度神经网络（DNN）进行对抗性权重攻击，比较了我们的机制和众所周知的双侧攻击之间的性能效率。结果表明，阈值断路器可以故意耗尽目标DNN系统的智能，而DRAM是完全保护的。



## **9. Certifying LLM Safety against Adversarial Prompting**

针对敌意提示认证LLM安全 cs.CL

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2309.02705v2) [paper-pdf](http://arxiv.org/pdf/2309.02705v2)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial attacks, which add maliciously designed token sequences to a harmful prompt to bypass the model's safety guards. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We defend against three attack modes: i) adversarial suffix, which appends an adversarial sequence at the end of the prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. For example, against adversarial suffixes of length 20, it certifiably detects 92% of harmful prompts and labels 94% of safe prompts correctly using the open-source language model Llama 2 as the safety filter. We further improve the filter's performance, in terms of accuracy and speed, by replacing Llama 2 with a DistilBERT safety classifier fine-tuned on safe and harmful prompts. Additionally, we propose two efficient empirical defenses: i) RandEC, a randomized version of erase-and-check that evaluates the safety filter on a small subset of the erased subsequences, and ii) GradEC, a gradient-based version that optimizes the erased tokens to remove the adversarial sequence. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.

摘要: 发布供公众使用的大型语言模型（LLM）包含了护栏，以确保其输出是安全的，通常被称为“模型对齐”。“一个对齐的语言模型应该拒绝用户制作有害内容的请求。然而，这样的安全措施容易受到对抗性攻击，这些攻击将恶意设计的令牌序列添加到有害的提示中，以绕过模型的安全防护。在这项工作中，我们引入擦除和检查，第一个框架，以抵御对抗性提示与可验证的安全保证。我们防御三种攻击模式：i）对抗后缀，它在提示符的末尾附加一个对抗序列; ii）对抗插入，其中对抗序列被插入到提示符中间的任何位置; iii）对抗注入，其中对抗令牌被插入到提示符中的任意位置，不一定是一个连续的块。我们的实验结果表明，该程序可以获得强有力的认证安全保证有害的提示，同时保持良好的经验性能的安全提示。例如，针对长度为20的对抗性后缀，它可以使用开源语言模型Llama 2作为安全过滤器，正确检测92%的有害提示并标记94%的安全提示。我们进一步提高了过滤器的性能，在准确性和速度方面，通过用DistilBERT安全分类器取代Llama 2，对安全和有害提示进行微调。此外，我们还提出了两种有效的经验防御：i）RandEC，一种随机版本的擦除和检查，它在擦除的合法性的一个小子集上评估安全过滤器; ii）GradEC，一种基于梯度的版本，它优化擦除的令牌以删除对抗序列。我们的实验代码可以在https://github.com/aounon/certified-llm-safety上找到。



## **10. Mate! Are You Really Aware? An Explainability-Guided Testing Framework for Robustness of Malware Detectors**

伙计！你真的知道吗？一种可解释制导的恶意软件检测器健壮性测试框架 cs.CR

Accepted at ESEC/FSE 2023. https://doi.org/10.1145/3611643.3616309

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2111.10085v4) [paper-pdf](http://arxiv.org/pdf/2111.10085v4)

**Authors**: Ruoxi Sun, Minhui Xue, Gareth Tyson, Tian Dong, Shaofeng Li, Shuo Wang, Haojin Zhu, Seyit Camtepe, Surya Nepal

**Abstract**: Numerous open-source and commercial malware detectors are available. However, their efficacy is threatened by new adversarial attacks, whereby malware attempts to evade detection, e.g., by performing feature-space manipulation. In this work, we propose an explainability-guided and model-agnostic testing framework for robustness of malware detectors when confronted with adversarial attacks. The framework introduces the concept of Accrued Malicious Magnitude (AMM) to identify which malware features could be manipulated to maximize the likelihood of evading detection. We then use this framework to test several state-of-the-art malware detectors' abilities to detect manipulated malware. We find that (i) commercial antivirus engines are vulnerable to AMM-guided test cases; (ii) the ability of a manipulated malware generated using one detector to evade detection by another detector (i.e., transferability) depends on the overlap of features with large AMM values between the different detectors; and (iii) AMM values effectively measure the fragility of features (i.e., capability of feature-space manipulation to flip the prediction results) and explain the robustness of malware detectors facing evasion attacks. Our findings shed light on the limitations of current malware detectors, as well as how they can be improved.

摘要: 有许多开源和商业恶意软件检测器可用。然而，它们的有效性受到新的敌意攻击的威胁，借此恶意软件试图通过例如执行特征空间操纵来逃避检测。在这项工作中，我们提出了一个可解释性指导和模型无关的测试框架，用于测试恶意软件检测器在面对敌意攻击时的健壮性。该框架引入了累积恶意量级(AMM)的概念，以确定哪些恶意软件功能可以被操纵，以最大限度地提高逃避检测的可能性。然后，我们使用这个框架来测试几个最先进的恶意软件检测器检测操纵恶意软件的能力。我们发现(I)商业反病毒引擎容易受到AMM引导的测试用例的攻击；(Ii)使用一个检测器生成的被操纵的恶意软件逃避另一个检测器的检测的能力(即可转移性)取决于不同检测器之间具有较大AMM值的特征的重叠；以及(Iii)AMM值有效地衡量了特征的脆弱性(即，对特征空间的操纵来反转预测结果的能力)，并解释了恶意软件检测器面对逃避攻击的健壮性。我们的发现揭示了当前恶意软件检测器的局限性，以及如何改进它们。



## **11. How Many Unicorns Are in This Image? A Safety Evaluation Benchmark for Vision LLMs**

这张图片中有多少只独角兽？一种视觉LLMS的安全评价基准 cs.CV

H.T., C.C., and Z.W. contribute equally. Work done during H.T. and  Z.W.'s internship at UCSC, and C.C. and Y.Z.'s internship at UNC

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.16101v1) [paper-pdf](http://arxiv.org/pdf/2311.16101v1)

**Authors**: Haoqin Tu, Chenhang Cui, Zijun Wang, Yiyang Zhou, Bingchen Zhao, Junlin Han, Wangchunshu Zhou, Huaxiu Yao, Cihang Xie

**Abstract**: This work focuses on the potential of Vision LLMs (VLLMs) in visual reasoning. Different from prior studies, we shift our focus from evaluating standard performance to introducing a comprehensive safety evaluation suite, covering both out-of-distribution (OOD) generalization and adversarial robustness. For the OOD evaluation, we present two novel VQA datasets, each with one variant, designed to test model performance under challenging conditions. In exploring adversarial robustness, we propose a straightforward attack strategy for misleading VLLMs to produce visual-unrelated responses. Moreover, we assess the efficacy of two jailbreaking strategies, targeting either the vision or language component of VLLMs. Our evaluation of 21 diverse models, ranging from open-source VLLMs to GPT-4V, yields interesting observations: 1) Current VLLMs struggle with OOD texts but not images, unless the visual information is limited; and 2) These VLLMs can be easily misled by deceiving vision encoders only, and their vision-language training often compromise safety protocols. We release this safety evaluation suite at https://github.com/UCSC-VLAA/vllm-safety-benchmark.

摘要: 本文主要研究视觉LLMS(VLLM)在视觉推理中的潜力。与以前的研究不同，我们将重点从评估标准性能转移到引入一个全面的安全评估套件，包括分布外(OOD)泛化和对手健壮性。我们对21种不同的模型进行了评估，从开源的VLLM到GPT-4V，得出了有趣的观察结果：1)当前的VLLM难以处理OOD文本而不是图像，除非视觉信息有限；2)这些VLLM很容易被欺骗的视觉编码器误导，并且它们的视觉语言培训经常危及安全协议。我们在https://github.com/UCSC-VLAA/vllm-safety-benchmark.上发布此安全评估套件



## **12. Adversaral Doodles: Interpretable and Human-drawable Attacks Provide Describable Insights**

对抗性涂鸦：可解释的和人类可绘制的攻击提供了可描述的见解 cs.CV

Submitted to CVPR 2024

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.15994v1) [paper-pdf](http://arxiv.org/pdf/2311.15994v1)

**Authors**: Ryoya Nara, Yusuke Matsui

**Abstract**: DNN-based image classification models are susceptible to adversarial attacks. Most previous adversarial attacks do not focus on the interpretability of the generated adversarial examples, and we cannot gain insights into the mechanism of the target classifier from the attacks. Therefore, we propose Adversarial Doodles, which have interpretable shapes. We optimize black b\'ezier curves to fool the target classifier by overlaying them onto the input image. By introducing random perspective transformation and regularizing the doodled area, we obtain compact attacks that cause misclassification even when humans replicate them by hand. Adversarial doodles provide describable and intriguing insights into the relationship between our attacks and the classifier's output. We utilize adversarial doodles and discover the bias inherent in the target classifier, such as "We add two strokes on its head, a triangle onto its body, and two lines inside the triangle on a bird image. Then, the classifier misclassifies the image as a butterfly."

摘要: 基于DNN的图像分类模型容易受到对抗性攻击。大多数以前的对抗性攻击并不关注生成的对抗性示例的可解释性，并且我们无法从攻击中了解目标分类器的机制。因此，我们提出了具有可解释形状的对抗性涂鸦。我们优化了黑色贝塞尔曲线，通过将它们叠加到输入图像上来欺骗目标分类器。通过引入随机视角变换和规则化涂鸦区域，我们得到紧凑的攻击，即使人类用手复制它们，也会导致错误分类。对抗性涂鸦为我们的攻击和分类器输出之间的关系提供了可描述和有趣的见解。我们利用对抗性涂鸦并发现目标分类器中固有的偏见，例如“我们在它的头上添加两个笔划，在它的身体上添加一个三角形，并在鸟图像上的三角形内添加两条线。然后，分类器将图像误分类为蝴蝶。”



## **13. CALICO: Self-Supervised Camera-LiDAR Contrastive Pre-training for BEV Perception**

Calico：BEV感知的自我监控相机-LiDAR对比预训练 cs.CV

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2306.00349v2) [paper-pdf](http://arxiv.org/pdf/2306.00349v2)

**Authors**: Jiachen Sun, Haizhong Zheng, Qingzhao Zhang, Atul Prakash, Z. Morley Mao, Chaowei Xiao

**Abstract**: Perception is crucial in the realm of autonomous driving systems, where bird's eye view (BEV)-based architectures have recently reached state-of-the-art performance. The desirability of self-supervised representation learning stems from the expensive and laborious process of annotating 2D and 3D data. Although previous research has investigated pretraining methods for both LiDAR and camera-based 3D object detection, a unified pretraining framework for multimodal BEV perception is missing. In this study, we introduce CALICO, a novel framework that applies contrastive objectives to both LiDAR and camera backbones. Specifically, CALICO incorporates two stages: point-region contrast (PRC) and region-aware distillation (RAD). PRC better balances the region- and scene-level representation learning on the LiDAR modality and offers significant performance improvement compared to existing methods. RAD effectively achieves contrastive distillation on our self-trained teacher model. CALICO's efficacy is substantiated by extensive evaluations on 3D object detection and BEV map segmentation tasks, where it delivers significant performance improvements. Notably, CALICO outperforms the baseline method by 10.5% and 8.6% on NDS and mAP. Moreover, CALICO boosts the robustness of multimodal 3D object detection against adversarial attacks and corruption. Additionally, our framework can be tailored to different backbones and heads, positioning it as a promising approach for multimodal BEV perception.

摘要: 在自动驾驶系统领域，感知是至关重要的，基于鸟瞰(Bev)的架构最近达到了最先进的性能。自监督表示学习的可取性源于对2D和3D数据进行标注的昂贵且费力的过程。虽然以前的研究已经研究了LiDAR和基于摄像机的3D目标检测的预训练方法，但缺乏一个统一的多模式Bev感知的预训练框架。在这项研究中，我们介绍了Calico，这是一个新的框架，它将对比目标应用于LiDAR和相机主干。具体地说，Calico包含两个阶段：点区域对比(PRC)和区域感知蒸馏(RAD)。PRC在LiDAR通道上更好地平衡了区域和场景级别的表示学习，并与现有方法相比提供了显著的性能改进。RAD有效地实现了对我们自学教师模式的对比提炼。Calico的有效性通过对3D对象检测和Bev地图分割任务的广泛评估得到证实，在这些任务中，Calico提供了显著的性能改进。值得注意的是，Calico在NDS和MAP上分别比基线方法高出10.5%和8.6%。此外，Calico增强了多模式3D对象检测针对对手攻击和损坏的健壮性。此外，我们的框架可以为不同的主干和头部量身定做，将其定位为一种有前途的多模式Bev感知方法。



## **14. AdaptGuard: Defending Against Universal Attacks for Model Adaptation**

AdaptGuard：针对模型适配的通用攻击防御 cs.CR

ICCV2023

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2303.10594v2) [paper-pdf](http://arxiv.org/pdf/2303.10594v2)

**Authors**: Lijun Sheng, Jian Liang, Ran He, Zilei Wang, Tieniu Tan

**Abstract**: Model adaptation aims at solving the domain transfer problem under the constraint of only accessing the pretrained source models. With the increasing considerations of data privacy and transmission efficiency, this paradigm has been gaining recent popularity. This paper studies the vulnerability to universal attacks transferred from the source domain during model adaptation algorithms due to the existence of malicious providers. We explore both universal adversarial perturbations and backdoor attacks as loopholes on the source side and discover that they still survive in the target models after adaptation. To address this issue, we propose a model preprocessing framework, named AdaptGuard, to improve the security of model adaptation algorithms. AdaptGuard avoids direct use of the risky source parameters through knowledge distillation and utilizes the pseudo adversarial samples under adjusted radius to enhance the robustness. AdaptGuard is a plug-and-play module that requires neither robust pretrained models nor any changes for the following model adaptation algorithms. Extensive results on three commonly used datasets and two popular adaptation methods validate that AdaptGuard can effectively defend against universal attacks and maintain clean accuracy in the target domain simultaneously. We hope this research will shed light on the safety and robustness of transfer learning. Code is available at https://github.com/TomSheng21/AdaptGuard.

摘要: 模型自适应的目的是在只访问预先训练好的源模型的约束下解决域迁移问题。随着对数据隐私和传输效率的日益关注，这种模式最近越来越受欢迎。本文研究了由于恶意提供者的存在，模型自适应算法对来自源域的通用攻击的脆弱性。我们探索了通用对抗扰动和后门攻击作为源端的漏洞，并发现它们在适应后仍然存在于目标模型中。为了解决这个问题，我们提出了一个模型预处理框架，命名为AdaptGuard，以提高模型自适应算法的安全性。AdaptGuard通过知识提取避免了直接使用危险源参数，并利用调整半径下的伪对抗样本来增强鲁棒性。AdaptGuard是一个即插即用模块，既不需要健壮的预训练模型，也不需要对以下模型自适应算法进行任何更改。在三个常用数据集和两种流行的自适应方法上的大量结果验证了AdaptGuard可以有效地防御通用攻击，同时保持目标域的干净准确性。我们希望这项研究能够揭示迁移学习的安全性和鲁棒性。代码可在https://github.com/TomSheng21/AdaptGuard上获得。



## **15. Attend Who is Weak: Enhancing Graph Condensation via Cross-Free Adversarial Training**

谁是弱者：通过无交叉对抗训练增强图凝聚 cs.LG

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.15772v1) [paper-pdf](http://arxiv.org/pdf/2311.15772v1)

**Authors**: Xinglin Li, Kun Wang, Hanhui Deng, Yuxuan Liang, Di Wu

**Abstract**: In this paper, we study the \textit{graph condensation} problem by compressing the large, complex graph into a concise, synthetic representation that preserves the most essential and discriminative information of structure and features. We seminally propose the concept of Shock Absorber (a type of perturbation) that enhances the robustness and stability of the original graphs against changes in an adversarial training fashion. Concretely, (I) we forcibly match the gradients between pre-selected graph neural networks (GNNs) trained on a synthetic, simplified graph and the original training graph at regularly spaced intervals. (II) Before each update synthetic graph point, a Shock Absorber serves as a gradient attacker to maximize the distance between the synthetic dataset and the original graph by selectively perturbing the parts that are underrepresented or insufficiently informative. We iteratively repeat the above two processes (I and II) in an adversarial training fashion to maintain the highly-informative context without losing correlation with the original dataset. More importantly, our shock absorber and the synthesized graph parallelly share the backward process in a free training manner. Compared to the original adversarial training, it introduces almost no additional time overhead.   We validate our framework across 8 datasets (3 graph and 5 node classification datasets) and achieve prominent results: for example, on Cora, Citeseer and Ogbn-Arxiv, we can gain nearly 1.13% to 5.03% improvements compare with SOTA models. Moreover, our algorithm adds only about 0.2% to 2.2% additional time overhead over Flicker, Citeseer and Ogbn-Arxiv. Compared to the general adversarial training, our approach improves time efficiency by nearly 4-fold.

摘要: 在这篇文章中，我们通过将大的复杂的图压缩成一个简洁的、综合的表示来研究图压缩问题，它保留了结构和特征的最本质和可区分的信息。我们半自动地提出了减震器(一种扰动)的概念，它以对抗性训练的方式增强了原始图形对变化的稳健性和稳定性。具体地说，(I)我们以规则间隔强制匹配在合成的简化图上训练的预先选择的图神经网络(GNN)和原始训练图之间的梯度。(Ii)在每次更新合成图形点之前，减震器充当梯度攻击者，通过选择性地扰动表示不足或信息不足的部分来最大化合成数据集和原始图形之间的距离。我们以对抗性训练的方式迭代重复上述两个过程(I和II)，以保持高度信息量的上下文，而不会失去与原始数据集的相关性。更重要的是，我们的减振器和合成的图形以自由训练的方式并行共享反向过程。与最初的对抗性训练相比，它几乎不会带来额外的时间开销。我们在8个数据集(3个图和5个节点分类数据集)上验证了我们的框架，并取得了显著的结果：例如，在CORA、Citeseer和Ogbn-Arxiv上，我们可以比SOTA模型获得近1.13%到5.03%的改进。此外，与Flicker、Citeseer和Ogbn-Arxiv相比，我们的算法只增加了大约0.2%到2.2%的额外时间开销。与一般的对抗性训练相比，我们的方法将时间效率提高了近4倍。



## **16. The Lipschitz-Variance-Margin Tradeoff for Enhanced Randomized Smoothing**

增强型随机平滑的Lipschitz-方差-边际权衡 cs.LG

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2309.16883v2) [paper-pdf](http://arxiv.org/pdf/2309.16883v2)

**Authors**: Blaise Delattre, Alexandre Araujo, Quentin Barthélemy, Alexandre Allauzen

**Abstract**: Real-life applications of deep neural networks are hindered by their unsteady predictions when faced with noisy inputs and adversarial attacks. The certified radius is in this context a crucial indicator of the robustness of models. However how to design an efficient classifier with a sufficient certified radius? Randomized smoothing provides a promising framework by relying on noise injection in inputs to obtain a smoothed and more robust classifier. In this paper, we first show that the variance introduced by randomized smoothing closely interacts with two other important properties of the classifier, \textit{i.e.} its Lipschitz constant and margin. More precisely, our work emphasizes the dual impact of the Lipschitz constant of the base classifier, on both the smoothed classifier and the empirical variance. Moreover, to increase the certified robust radius, we introduce a different simplex projection technique for the base classifier to leverage the variance-margin trade-off thanks to Bernstein's concentration inequality, along with an enhanced Lipschitz bound. Experimental results show a significant improvement in certified accuracy compared to current state-of-the-art methods. Our novel certification procedure allows us to use pre-trained models that are used with randomized smoothing, effectively improving the current certification radius in a zero-shot manner.

摘要: 深度神经网络的实际应用在面对噪声输入和对抗性攻击时受到其不稳定预测的阻碍。在这种情况下，认证半径是模型稳健性的关键指标。然而，如何设计一个有效的分类器具有足够的认证半径？随机平滑提供了一个很有前途的框架，依靠噪声注入输入，以获得一个平滑和更强大的分类器。在本文中，我们首先证明了随机平滑引入的方差与分类器的其他两个重要属性密切相关，它的Lipschitz常数和边际。更确切地说，我们的工作强调了基础分类器的Lipschitz常数对平滑分类器和经验方差的双重影响。此外，为了增加认证的鲁棒半径，我们引入了一个不同的单纯形投影技术的基础分类器，以利用方差利润权衡由于伯恩斯坦的浓度不等式，以及增强的Lipschitz界。实验结果表明，认证的准确性相比，目前国家的最先进的方法显着改善。我们新颖的认证程序允许我们使用预先训练的模型，这些模型与随机平滑一起使用，以零射击的方式有效地提高了当前的认证半径。



## **17. SLMIA-SR: Speaker-Level Membership Inference Attacks against Speaker Recognition Systems**

SLMIA-SR：针对说话人识别系统的说话人级别成员推理攻击 cs.CR

In Proceedings of the 31st Network and Distributed System Security  (NDSS) Symposium, 2024

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2309.07983v2) [paper-pdf](http://arxiv.org/pdf/2309.07983v2)

**Authors**: Guangke Chen, Yedi Zhang, Fu Song

**Abstract**: Membership inference attacks allow adversaries to determine whether a particular example was contained in the model's training dataset. While previous works have confirmed the feasibility of such attacks in various applications, none has focused on speaker recognition (SR), a promising voice-based biometric recognition technique. In this work, we propose SLMIA-SR, the first membership inference attack tailored to SR. In contrast to conventional example-level attack, our attack features speaker-level membership inference, i.e., determining if any voices of a given speaker, either the same as or different from the given inference voices, have been involved in the training of a model. It is particularly useful and practical since the training and inference voices are usually distinct, and it is also meaningful considering the open-set nature of SR, namely, the recognition speakers were often not present in the training data. We utilize intra-similarity and inter-dissimilarity, two training objectives of SR, to characterize the differences between training and non-training speakers and quantify them with two groups of features driven by carefully-established feature engineering to mount the attack. To improve the generalizability of our attack, we propose a novel mixing ratio training strategy to train attack models. To enhance the attack performance, we introduce voice chunk splitting to cope with the limited number of inference voices and propose to train attack models dependent on the number of inference voices. Our attack is versatile and can work in both white-box and black-box scenarios. Additionally, we propose two novel techniques to reduce the number of black-box queries while maintaining the attack performance. Extensive experiments demonstrate the effectiveness of SLMIA-SR.

摘要: 成员关系推理攻击允许攻击者确定特定示例是否包含在模型的训练数据集中。虽然以前的工作已经证实了这类攻击在各种应用中的可行性，但还没有人专注于说话人识别(SR)，这是一种很有前途的基于语音的生物识别技术。在这项工作中，我们提出了SLMIA-SR，这是第一个针对SR量身定做的成员推理攻击。与传统的范例级攻击不同，我们的攻击具有说话人级别的成员关系推理，即确定给定说话人的任何声音是否与给定的推理声音相同或不同，参与了模型的训练。它特别有用和实用，因为训练和推理的声音通常是不同的，而且考虑到SR的开放集性质，即识别说话人通常不在训练数据中，这也是有意义的。我们利用随机共振的内相似和互异两个训练目标来刻画训练说话人和非训练说话人之间的差异，并在精心建立的特征工程的驱动下用两组特征来量化它们来发动攻击。为了提高攻击的泛化能力，我们提出了一种新的混合比训练策略来训练攻击模型。为了提高攻击性能，我们引入了语音块分裂来应对有限的推理语音，并提出了根据推理语音的数量来训练攻击模型。我们的攻击是多才多艺的，可以在白盒和黑盒场景中工作。此外，我们还提出了两种新的技术来在保持攻击性能的同时减少黑盒查询的数量。大量实验证明了SLMIA-SR的有效性。



## **18. RetouchUAA: Unconstrained Adversarial Attack via Image Retouching**

RetouchUAA：通过图像修饰的无约束对抗攻击 cs.CV

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.16478v1) [paper-pdf](http://arxiv.org/pdf/2311.16478v1)

**Authors**: Mengda Xie, Yiling He, Meie Fang

**Abstract**: Deep Neural Networks (DNNs) are susceptible to adversarial examples. Conventional attacks generate controlled noise-like perturbations that fail to reflect real-world scenarios and hard to interpretable. In contrast, recent unconstrained attacks mimic natural image transformations occurring in the real world for perceptible but inconspicuous attacks, yet compromise realism due to neglect of image post-processing and uncontrolled attack direction. In this paper, we propose RetouchUAA, an unconstrained attack that exploits a real-life perturbation: image retouching styles, highlighting its potential threat to DNNs. Compared to existing attacks, RetouchUAA offers several notable advantages. Firstly, RetouchUAA excels in generating interpretable and realistic perturbations through two key designs: the image retouching attack framework and the retouching style guidance module. The former custom-designed human-interpretability retouching framework for adversarial attack by linearizing images while modelling the local processing and retouching decision-making in human retouching behaviour, provides an explicit and reasonable pipeline for understanding the robustness of DNNs against retouching. The latter guides the adversarial image towards standard retouching styles, thereby ensuring its realism. Secondly, attributed to the design of the retouching decision regularization and the persistent attack strategy, RetouchUAA also exhibits outstanding attack capability and defense robustness, posing a heavy threat to DNNs. Experiments on ImageNet and Place365 reveal that RetouchUAA achieves nearly 100\% white-box attack success against three DNNs, while achieving a better trade-off between image naturalness, transferability and defense robustness than baseline attacks.

摘要: 深度神经网络（DNN）容易受到对抗性样本的影响。传统的攻击会产生受控的类似噪声的扰动，这些扰动无法反映真实世界的场景，并且难以解释。相比之下，最近的无约束攻击模仿自然的图像变换发生在现实世界中的可感知的，但不显眼的攻击，但妥协的现实主义，由于忽视图像后处理和不受控制的攻击方向。在本文中，我们提出了RetouchUAA，这是一种无约束的攻击，利用了现实生活中的扰动：图像修饰风格，突出了其对DNN的潜在威胁。与现有的攻击相比，RetouchUAA提供了几个显著的优势。首先，RetouchUAA通过两个关键设计：图像修饰攻击框架和修饰风格指导模块，在生成可解释和真实的扰动方面表现出色。以前定制设计的人类可解释性修饰框架通过线性化图像来对抗攻击，同时对人类修饰行为中的局部处理和修饰决策进行建模，为理解DNN对修饰的鲁棒性提供了一个明确而合理的管道。后者引导对抗图像朝向标准修饰风格，从而确保其真实性。其次，由于修饰决策正则化和持续攻击策略的设计，RetouchUAA也表现出出色的攻击能力和防御鲁棒性，对DNN构成了严重威胁。在ImageNet和Place365上的实验表明，RetouchUAA对三个DNN的白盒攻击成功率接近100%，同时在图像自然度，可传输性和防御鲁棒性之间实现了比基线攻击更好的权衡。



## **19. Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information**

基于困惑度量和上下文信息的令牌级敌意提示检测 cs.CL

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.11509v2) [paper-pdf](http://arxiv.org/pdf/2311.11509v2)

**Authors**: Zhengmian Hu, Gang Wu, Saayan Mitra, Ruiyi Zhang, Tong Sun, Heng Huang, Viswanathan Swaminathan

**Abstract**: In recent years, Large Language Models (LLM) have emerged as pivotal tools in various applications. However, these models are susceptible to adversarial prompt attacks, where attackers can carefully curate input strings that lead to undesirable outputs. The inherent vulnerability of LLMs stems from their input-output mechanisms, especially when presented with intensely out-of-distribution (OOD) inputs. This paper proposes a token-level detection method to identify adversarial prompts, leveraging the LLM's capability to predict the next token's probability. We measure the degree of the model's perplexity and incorporate neighboring token information to encourage the detection of contiguous adversarial prompt sequences. As a result, we propose two methods: one that identifies each token as either being part of an adversarial prompt or not, and another that estimates the probability of each token being part of an adversarial prompt.

摘要: 近年来，大型语言模型(LLM)已经成为各种应用中的关键工具。然而，这些模型容易受到敌意提示攻击，攻击者可以仔细策划导致不良输出的输入字符串。低成本管理的内在脆弱性源于其投入-产出机制，特别是在投入严重失配(OOD)的情况下。提出了一种令牌级检测方法来识别敌意提示，利用LLM的能力来预测下一个令牌的概率。我们测量模型的困惑程度，并结合相邻令牌信息来鼓励对连续对抗性提示序列的检测。因此，我们提出了两种方法：一种是识别每个令牌是不是对抗性提示的一部分，另一种是估计每个令牌是对抗性提示的一部分的概率。



## **20. Instruct2Attack: Language-Guided Semantic Adversarial Attacks**

Instruct2Attack：语言制导的语义对抗性攻击 cs.CV

under submission, code coming soon

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.15551v1) [paper-pdf](http://arxiv.org/pdf/2311.15551v1)

**Authors**: Jiang Liu, Chen Wei, Yuxiang Guo, Heng Yu, Alan Yuille, Soheil Feizi, Chun Pong Lau, Rama Chellappa

**Abstract**: We propose Instruct2Attack (I2A), a language-guided semantic attack that generates semantically meaningful perturbations according to free-form language instructions. We make use of state-of-the-art latent diffusion models, where we adversarially guide the reverse diffusion process to search for an adversarial latent code conditioned on the input image and text instruction. Compared to existing noise-based and semantic attacks, I2A generates more natural and diverse adversarial examples while providing better controllability and interpretability. We further automate the attack process with GPT-4 to generate diverse image-specific text instructions. We show that I2A can successfully break state-of-the-art deep neural networks even under strong adversarial defenses, and demonstrate great transferability among a variety of network architectures.

摘要: 我们提出了指令2攻击（I2 A），一种语言引导的语义攻击，根据自由形式的语言指令生成语义上有意义的扰动。我们利用最先进的潜在扩散模型，在该模型中，我们对抗性地引导反向扩散过程以搜索以输入图像和文本指令为条件的对抗性潜在代码。与现有的基于噪声和语义的攻击相比，I2 A生成了更自然、更多样化的对抗性示例，同时提供了更好的可控性和可解释性。我们进一步使用GPT-4自动化攻击过程，以生成各种特定于图像的文本指令。我们表明，即使在强大的对抗性防御下，I2 A也可以成功打破最先进的深度神经网络，并在各种网络架构之间表现出很好的可移植性。



## **21. Confidence Is All You Need for MI Attacks**

信心是MI攻击所需要的全部 cs.LG

2 pages, 1 figure

**SubmitDate**: 2023-11-26    [abs](http://arxiv.org/abs/2311.15373v1) [paper-pdf](http://arxiv.org/pdf/2311.15373v1)

**Authors**: Abhishek Sinha, Himanshi Tibrewal, Mansi Gupta, Nikhar Waghela, Shivank Garg

**Abstract**: In this evolving era of machine learning security, membership inference attacks have emerged as a potent threat to the confidentiality of sensitive data. In this attack, adversaries aim to determine whether a particular point was used during the training of a target model. This paper proposes a new method to gauge a data point's membership in a model's training set. Instead of correlating loss with membership, as is traditionally done, we have leveraged the fact that training examples generally exhibit higher confidence values when classified into their actual class. During training, the model is essentially being 'fit' to the training data and might face particular difficulties in generalization to unseen data. This asymmetry leads to the model achieving higher confidence on the training data as it exploits the specific patterns and noise present in the training data. Our proposed approach leverages the confidence values generated by the machine learning model. These confidence values provide a probabilistic measure of the model's certainty in its predictions and can further be used to infer the membership of a given data point. Additionally, we also introduce another variant of our method that allows us to carry out this attack without knowing the ground truth(true class) of a given data point, thus offering an edge over existing label-dependent attack methods.

摘要: 在这个不断发展的机器学习安全时代，成员身份推理攻击已经成为对敏感数据保密性的有力威胁。在这种攻击中，对手的目标是确定在目标模型的训练过程中是否使用了特定的点。本文提出了一种新的方法来衡量数据点在模型训练集中的隶属度。我们没有像传统上那样将损失与成员关系联系起来，而是利用了这样一个事实，即当分类到实际班级时，训练样本通常显示出更高的置信度。在训练过程中，该模型基本上与训练数据“匹配”，在推广到看不见的数据时可能会面临特别的困难。这种不对称性导致模型在训练数据上实现了更高的置信度，因为它利用了训练数据中存在的特定模式和噪声。我们提出的方法利用了机器学习模型生成的置信度。这些置信值提供了模型在其预测中的确定性的概率度量，并可进一步用于推断给定数据点的成员资格。此外，我们还介绍了我们的方法的另一个变体，它允许我们在不知道给定数据点的基本事实(真类)的情况下执行这种攻击，从而提供了比现有的依赖标签的攻击方法更好的优势。



## **22. Adversarial Purification of Information Masking**

信息掩饰的对抗性净化 cs.CV

**SubmitDate**: 2023-11-26    [abs](http://arxiv.org/abs/2311.15339v1) [paper-pdf](http://arxiv.org/pdf/2311.15339v1)

**Authors**: Sitong Liu, Zhichao Lian, Shuangquan Zhang, Liang Xiao

**Abstract**: Adversarial attacks meticulously generate minuscule, imperceptible perturbations to images to deceive neural networks. Counteracting these, adversarial purification methods seek to transform adversarial input samples into clean output images to defend against adversarial attacks. Nonetheless, extent generative models fail to effectively eliminate adversarial perturbations, yielding less-than-ideal purification results. We emphasize the potential threat of residual adversarial perturbations to target models, quantitatively establishing a relationship between perturbation scale and attack capability. Notably, the residual perturbations on the purified image primarily stem from the same-position patch and similar patches of the adversarial sample. We propose a novel adversarial purification approach named Information Mask Purification (IMPure), aims to extensively eliminate adversarial perturbations. To obtain an adversarial sample, we first mask part of the patches information, then reconstruct the patches to resist adversarial perturbations from the patches. We reconstruct all patches in parallel to obtain a cohesive image. Then, in order to protect the purified samples against potential similar regional perturbations, we simulate this risk by randomly mixing the purified samples with the input samples before inputting them into the feature extraction network. Finally, we establish a combined constraint of pixel loss and perceptual loss to augment the model's reconstruction adaptability. Extensive experiments on the ImageNet dataset with three classifier models demonstrate that our approach achieves state-of-the-art results against nine adversarial attack methods. Implementation code and pre-trained weights can be accessed at \textcolor{blue}{https://github.com/NoWindButRain/IMPure}.

摘要: 敌意攻击小心翼翼地对图像产生微小的、不可察觉的扰动，以欺骗神经网络。对抗性净化方法寻求将对抗性输入样本转换为干净的输出图像以防御对抗性攻击。然而，广度生成模型不能有效地消除对抗性扰动，产生不太理想的纯化结果。我们强调了残留对抗性扰动对目标模型的潜在威胁，定量地建立了扰动规模与攻击能力之间的关系。值得注意的是，纯化图像上的残留扰动主要源于对抗性样本的相同位置补丁和相似补丁。我们提出了一种新的对抗性净化方法，称为信息掩码净化(INPURE)，旨在广泛地消除对抗性扰动。为了获得对抗性样本，我们首先掩蔽部分斑块信息，然后重建斑块以抵抗来自斑块的对抗性扰动。我们对所有的块进行并行重建，以获得一个连贯的图像。然后，为了保护纯化样本不受潜在的相似区域扰动的影响，在输入到特征提取网络之前，我们通过将纯化样本与输入样本随机混合来模拟这种风险。最后，我们建立了像素损失和感知损失的组合约束，以增强模型的重建适应性。在具有三种分类器模型的ImageNet数据集上的大量实验表明，该方法对九种对抗性攻击方法取得了最先进的结果。实施代码和预先训练的权重可在\textcolor{blue}{https://github.com/NoWindButRain/IMPure}.上访问



## **23. Effective Backdoor Mitigation Depends on the Pre-training Objective**

有效的后门缓解取决于培训前的目标 cs.LG

Accepted for oral presentation at BUGS workshop @ NeurIPS 2023  (https://neurips2023-bugs.github.io/)

**SubmitDate**: 2023-11-25    [abs](http://arxiv.org/abs/2311.14948v1) [paper-pdf](http://arxiv.org/pdf/2311.14948v1)

**Authors**: Sahil Verma, Gantavya Bhatt, Avi Schwarzschild, Soumye Singhal, Arnav Mohanty Das, Chirag Shah, John P Dickerson, Jeff Bilmes

**Abstract**: Despite the advanced capabilities of contemporary machine learning (ML) models, they remain vulnerable to adversarial and backdoor attacks. This vulnerability is particularly concerning in real-world deployments, where compromised models may exhibit unpredictable behavior in critical scenarios. Such risks are heightened by the prevalent practice of collecting massive, internet-sourced datasets for pre-training multimodal models, as these datasets may harbor backdoors. Various techniques have been proposed to mitigate the effects of backdooring in these models such as CleanCLIP which is the current state-of-the-art approach.   In this work, we demonstrate that the efficacy of CleanCLIP in mitigating backdoors is highly dependent on the particular objective used during model pre-training.   We observe that stronger pre-training objectives correlate with harder to remove backdoors behaviors. We show this by training multimodal models on two large datasets consisting of 3 million (CC3M) and 6 million (CC6M) datapoints, under various pre-training objectives, followed by poison removal using CleanCLIP. We find that CleanCLIP is ineffective when stronger pre-training objectives are used, even with extensive hyperparameter tuning.   Our findings underscore critical considerations for ML practitioners who pre-train models using large-scale web-curated data and are concerned about potential backdoor threats. Notably, our results suggest that simpler pre-training objectives are more amenable to effective backdoor removal. This insight is pivotal for practitioners seeking to balance the trade-offs between using stronger pre-training objectives and security against backdoor attacks.

摘要: 尽管当代机器学习(ML)模型具有先进的能力，但它们仍然容易受到对手和后门攻击。此漏洞在实际部署中尤其令人担忧，在实际部署中，受危害的模型可能会在关键情况下表现出不可预测的行为。为训练前的多模式模型收集来自互联网的海量数据集的普遍做法加剧了这种风险，因为这些数据集可能有后门。已经提出了各种技术来减轻这些模型中回溯的影响，例如CleanCLIP，这是当前最先进的方法。在这项工作中，我们证明了CleanCLIP在缓解后门方面的有效性高度依赖于在模型预培训期间使用的特定目标。我们观察到，较强的培训前目标与较难消除后门行为相关。我们通过在两个由300万(CC3M)和600万(CC6M)数据点组成的大型数据集上训练多模模型，在不同的预训练目标下，然后使用CleanCLIP去除毒物来证明这一点。我们发现，当使用更强的预培训目标时，即使进行了广泛的超参数调整，CleanCLIP也是无效的。我们的发现强调了ML从业者的关键考虑，他们使用大规模的网络管理数据对模型进行预培训，并担心潜在的后门威胁。值得注意的是，我们的结果表明，简单的预培训目标更容易有效地移除后门。对于寻求在使用更强的预培训目标和针对后门攻击的安全性之间进行权衡的从业者来说，这一见解至关重要。



## **24. Robust Graph Neural Networks via Unbiased Aggregation**

基于无偏聚集的鲁棒图神经网络 cs.LG

**SubmitDate**: 2023-11-25    [abs](http://arxiv.org/abs/2311.14934v1) [paper-pdf](http://arxiv.org/pdf/2311.14934v1)

**Authors**: Ruiqi Feng, Zhichao Hou, Tyler Derr, Xiaorui Liu

**Abstract**: The adversarial robustness of Graph Neural Networks (GNNs) has been questioned due to the false sense of security uncovered by strong adaptive attacks despite the existence of numerous defenses. In this work, we delve into the robustness analysis of representative robust GNNs and provide a unified robust estimation point of view to understand their robustness and limitations. Our novel analysis of estimation bias motivates the design of a robust and unbiased graph signal estimator. We then develop an efficient Quasi-Newton iterative reweighted least squares algorithm to solve the estimation problem, which unfolds as robust unbiased aggregation layers in GNNs with a theoretical convergence guarantee. Our comprehensive experiments confirm the strong robustness of our proposed model, and the ablation study provides a deep understanding of its advantages.

摘要: 尽管存在大量的防御措施，但由于强自适应攻击所揭示的虚假安全感，图神经网络(GNN)的对抗健壮性受到了质疑。在这项工作中，我们深入研究了具有代表性的稳健GNN的稳健性分析，并提供了一个统一的稳健估计的观点来理解它们的稳健性和局限性。我们对估计偏差的新颖分析激发了稳健和无偏图信号估计器的设计。然后，我们提出了一种有效的拟牛顿迭代重加权最小二乘算法来解决估计问题，该算法在理论上保证收敛的情况下表现为GNN中健壮的无偏聚合层。我们的综合实验证实了我们所提出的模型具有很强的稳健性，消融研究使我们对其优势有了更深的理解。



## **25. Exploiting Large Language Models (LLMs) through Deception Techniques and Persuasion Principles**

通过欺骗技术和说服原理开发大型语言模型(LLM) cs.HC

10 pages, 16 tables, 5 figures, IEEE BigData 2023 (Workshops)

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14876v1) [paper-pdf](http://arxiv.org/pdf/2311.14876v1)

**Authors**: Sonali Singh, Faranak Abri, Akbar Siami Namin

**Abstract**: With the recent advent of Large Language Models (LLMs), such as ChatGPT from OpenAI, BARD from Google, Llama2 from Meta, and Claude from Anthropic AI, gain widespread use, ensuring their security and robustness is critical. The widespread use of these language models heavily relies on their reliability and proper usage of this fascinating technology. It is crucial to thoroughly test these models to not only ensure its quality but also possible misuses of such models by potential adversaries for illegal activities such as hacking. This paper presents a novel study focusing on exploitation of such large language models against deceptive interactions. More specifically, the paper leverages widespread and borrows well-known techniques in deception theory to investigate whether these models are susceptible to deceitful interactions.   This research aims not only to highlight these risks but also to pave the way for robust countermeasures that enhance the security and integrity of language models in the face of sophisticated social engineering tactics. Through systematic experiments and analysis, we assess their performance in these critical security domains. Our results demonstrate a significant finding in that these large language models are susceptible to deception and social engineering attacks.

摘要: 随着最近大型语言模型（LLM）的出现，例如OpenAI的ChatGPT，Google的BARD，Meta的Llama2和Anthropic AI的Claude，获得了广泛的使用，确保其安全性和鲁棒性至关重要。这些语言模型的广泛使用在很大程度上依赖于它们的可靠性和对这种迷人技术的正确使用。彻底测试这些模型至关重要，不仅要确保其质量，还要确保潜在对手可能滥用这些模型进行非法活动，如黑客攻击。本文提出了一种新的研究，重点是利用这种大型语言模型对欺骗性的互动。更具体地说，本文利用广泛和借用欺骗理论中的知名技术来研究这些模型是否容易受到欺骗性交互的影响。   这项研究的目的不仅是为了突出这些风险，但也铺平了道路，强大的对策，提高语言模型的安全性和完整性，面对复杂的社会工程策略。通过系统的实验和分析，我们评估他们在这些关键的安全领域的表现。我们的研究结果表明，这些大型语言模型容易受到欺骗和社会工程攻击。



## **26. Adversarial Machine Learning in Latent Representations of Neural Networks**

神经网络潜在表示中的对抗性机器学习 cs.LG

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2309.17401v2) [paper-pdf](http://arxiv.org/pdf/2309.17401v2)

**Authors**: Milin Zhang, Mohammad Abdi, Francesco Restuccia

**Abstract**: Distributed deep neural networks (DNNs) have been shown to reduce the computational burden of mobile devices and decrease the end-to-end inference latency in edge computing scenarios. While distributed DNNs have been studied, to the best of our knowledge the resilience of distributed DNNs to adversarial action still remains an open problem. In this paper, we fill the existing research gap by rigorously analyzing the robustness of distributed DNNs against adversarial action. We cast this problem in the context of information theory and introduce two new measurements for distortion and robustness. Our theoretical findings indicate that (i) assuming the same level of information distortion, latent features are always more robust than input representations; (ii) the adversarial robustness is jointly determined by the feature dimension and the generalization capability of the DNN. To test our theoretical findings, we perform extensive experimental analysis by considering 6 different DNN architectures, 6 different approaches for distributed DNN and 10 different adversarial attacks to the ImageNet-1K dataset. Our experimental results support our theoretical findings by showing that the compressed latent representations can reduce the success rate of adversarial attacks by 88% in the best case and by 57% on the average compared to attacks to the input space.

摘要: 分布式深度神经网络可以减轻移动设备的计算负担，减少边缘计算场景中的端到端推理延迟。虽然已经对分布式DNN进行了研究，但就我们所知，分布式DNN对敌意行为的恢复能力仍然是一个悬而未决的问题。在本文中，我们通过严格分析分布式DNN对攻击行为的健壮性来填补现有的研究空白。我们把这个问题放在信息论的背景下，并引入了两个新的失真和稳健性度量。我们的理论结果表明：(I)假设信息失真程度相同，潜在特征总是比输入表示更健壮；(Ii)DNN的对抗健壮性由特征维度和泛化能力共同决定。为了验证我们的理论发现，我们通过考虑6种不同的DNN体系结构、6种不同的分布式DNN方法和10种不同的针对ImageNet-1K数据集的对手攻击进行了广泛的实验分析。我们的实验结果支持我们的理论发现，与对输入空间的攻击相比，压缩的潜在表示在最好的情况下可以使对抗性攻击的成功率降低88%，平均降低57%。



## **27. Tamper-Evident Pairing**

明显篡改的配对 cs.CR

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14790v1) [paper-pdf](http://arxiv.org/pdf/2311.14790v1)

**Authors**: Aleksandar Manev

**Abstract**: Establishing a secure connection between wireless devices has become significantly important with the increasing number of Wi-Fi products coming to the market. In order to provide an easy and secure pairing standard, the Wi-Fi Alliance has designed the Wi-Fi Protected Setup. Push-Button Configuration (PBC) is part of this standard and is especially useful for pairing devices with physical limitations. However, PBC is proven to be vulnerable to man-in-the-middle (MITM) attacks. Tamper-Evident Pairing (TEP) is an improvement of the PBC standard, which aims to fix the MITM vulnerability without interfering the useful properties of PBC. It relies on the Tamper-Evident Announcement (TEA), which guarantees that an adversary can neither tamper a transmitted message without being detected, nor hide the fact that the message has been sent. The security properties of TEP were proven manually by its authors and tested with the Uppaal and Spin model checkers. During the Uppaal model checking, no vulnerabilities were found. However, the Spin model revealed a case, in which the TEP's security is not guaranteed. In this paper, we first provide a comprehensive overview of the TEP protocol, including all information needed to understand how it works. Furthermore, we summarize the security checks performed on it, give the circumstances, under which it is no longer resistant to MITM attacks and explain the reasons why they could not be revealed with the first model. Nevertheless, future work is required to gain full certainty of the TEP's security before applying it in the industry.

摘要: 随着越来越多的Wi-Fi产品进入市场，在无线设备之间建立安全连接变得非常重要。为了提供简单安全的配对标准，Wi-Fi联盟设计了Wi-Fi保护设置。按钮配置(PBC)是该标准的一部分，对于具有物理限制的配对设备特别有用。然而，事实证明，PBC容易受到中间人(MITM)攻击。篡改明显配对(TEP)是对PBC标准的改进，旨在修复MITM漏洞而不干扰PBC的有用特性。它依赖于明显篡改声明(TEA)，该声明保证攻击者既不能在不被检测到的情况下篡改传输的消息，也不能隐藏消息已经发送的事实。TEP的安全属性由其作者手动验证，并使用Uppaal和Spin模型检查器进行测试。在Uppaal模型检查过程中，没有发现任何漏洞。然而，SPIN模型揭示了一种情况，在这种情况下，TEP的安全性得不到保证。在本文中，我们首先全面概述TEP协议，包括了解其工作原理所需的所有信息。此外，我们总结了对其进行的安全检查，给出了它不再抵抗MITM攻击的情况，并解释了第一种模型无法揭示它们的原因。然而，在将TEP应用于行业之前，需要进行进一步的工作，以完全确定TEP的安全性。



## **28. Mind the box: $l_1$-APGD for sparse adversarial attacks on image classifiers**

注意：$L_1$-针对图像分类器的稀疏对抗性攻击的APGD cs.LG

In ICML 2021. Fixed typos in Eq. (3) and Eq. (4)

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2103.01208v3) [paper-pdf](http://arxiv.org/pdf/2103.01208v3)

**Authors**: Francesco Croce, Matthias Hein

**Abstract**: We show that when taking into account also the image domain $[0,1]^d$, established $l_1$-projected gradient descent (PGD) attacks are suboptimal as they do not consider that the effective threat model is the intersection of the $l_1$-ball and $[0,1]^d$. We study the expected sparsity of the steepest descent step for this effective threat model and show that the exact projection onto this set is computationally feasible and yields better performance. Moreover, we propose an adaptive form of PGD which is highly effective even with a small budget of iterations. Our resulting $l_1$-APGD is a strong white-box attack showing that prior works overestimated their $l_1$-robustness. Using $l_1$-APGD for adversarial training we get a robust classifier with SOTA $l_1$-robustness. Finally, we combine $l_1$-APGD and an adaptation of the Square Attack to $l_1$ into $l_1$-AutoAttack, an ensemble of attacks which reliably assesses adversarial robustness for the threat model of $l_1$-ball intersected with $[0,1]^d$.

摘要: 我们证明了当同时考虑象域$[0，1]^d$时，已建立的$L_1$投影梯度下降攻击是次优的，因为它们没有考虑有效的威胁模型是$L_1$球和$[0，1]^d$的交集。我们研究了这一有效威胁模型的最陡下降步长的期望稀疏性，并证明了在该集合上的精确投影在计算上是可行的，并且产生了更好的性能。此外，我们提出了一种自适应形式的PGD，它即使在迭代预算很小的情况下也是非常有效的。我们得到的$L_1$-APGD是一个强白盒攻击，表明以前的工作高估了它们的$L_1$-稳健性。利用$L_1$-APGD进行对抗性训练，得到一个具有SOTA$L_1$-健壮性的稳健分类器。最后，我们将$L_1$-APGD和对$L_1$的方形攻击的改编合并为$L_1$-AutoAttack，这是一个攻击集合，它可靠地评估了$L_1$球与$[0，1]^d$相交的威胁模型的对手健壮性。



## **29. Trainwreck: A damaging adversarial attack on image classifiers**

Trainreck：对图像分类器的破坏性敌意攻击 cs.CV

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14772v1) [paper-pdf](http://arxiv.org/pdf/2311.14772v1)

**Authors**: Jan Zahálka

**Abstract**: Adversarial attacks are an important security concern for computer vision (CV), as they enable malicious attackers to reliably manipulate CV models. Existing attacks aim to elicit an output desired by the attacker, but keep the model fully intact on clean data. With CV models becoming increasingly valuable assets in applied practice, a new attack vector is emerging: disrupting the models as a form of economic sabotage. This paper opens up the exploration of damaging adversarial attacks (DAAs) that seek to damage the target model and maximize the total cost incurred by the damage. As a pioneer DAA, this paper proposes Trainwreck, a train-time attack that poisons the training data of image classifiers to degrade their performance. Trainwreck conflates the data of similar classes using stealthy ($\epsilon \leq 8/255$) class-pair universal perturbations computed using a surrogate model. Trainwreck is a black-box, transferable attack: it requires no knowledge of the target model's architecture, and a single poisoned dataset degrades the performance of any model trained on it. The experimental evaluation on CIFAR-10 and CIFAR-100 demonstrates that Trainwreck is indeed an effective attack across various model architectures including EfficientNetV2, ResNeXt-101, and a finetuned ViT-L-16. The strength of the attack can be customized by the poison rate parameter. Finally, data redundancy with file hashing and/or pixel difference are identified as a reliable defense technique against Trainwreck or similar DAAs. The code is available at https://github.com/JanZahalka/trainwreck.

摘要: 对抗性攻击是计算机视觉(CV)的一个重要安全问题，因为它们使恶意攻击者能够可靠地操纵CV模型。现有的攻击旨在获得攻击者想要的输出，但在干净的数据上保持模型完全不变。随着CV模型在应用实践中变得越来越有价值，一种新的攻击载体正在出现：将破坏模型作为一种经济破坏形式。本文对破坏性对抗性攻击(DAA)进行了探索，该攻击试图破坏目标模型并最大化由损害引起的总成本。作为DAA的先驱，本文提出了Trainwreck，这是一种训练时间攻击，它毒化图像分类器的训练数据，降低其性能。Trainwreck使用秘密($\epsilon\leq 8/255$)类对通用扰动合并相似类的数据，这些类对通用扰动是使用代理模型计算的。Trainwreck是一种黑匣子、可转移的攻击：它不需要了解目标模型的体系结构，并且单个有毒数据集会降低任何针对其训练的模型的性能。在CIFAR-10和CIFAR-100上的实验评估表明，Trainwreck确实是一种跨越各种模型体系结构的有效攻击，包括EfficientNetV2、ResNeXt-101和精调的VIT-L-16。攻击的强度可以通过毒率参数进行定制。最后，文件散列和/或像素差异的数据冗余被认为是抵御Trainwreck或类似DAA的可靠防御技术。代码可在https://github.com/JanZahalka/trainwreck.上获得



## **30. Universal Jailbreak Backdoors from Poisoned Human Feedback**

来自有毒人类反馈的通用越狱后门 cs.AI

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14455v1) [paper-pdf](http://arxiv.org/pdf/2311.14455v1)

**Authors**: Javier Rando, Florian Tramèr

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.

摘要: 来自人类反馈的强化学习(RLHF)被用来对齐大型语言模型以产生有益和无害的响应。然而，先前的工作表明，这些模型可以通过找到敌意提示来越狱，这些提示可以将模型恢复到其不一致的行为。在本文中，我们考虑了一种新的威胁，即攻击者在RLHF训练数据中下毒，以便在模型中嵌入“越狱后门”。后门将触发词嵌入到模型中，其作用类似于通用的“sudo命令”：将触发词添加到任何提示中都可以实现有害的响应，而无需搜索敌意提示。通用越狱后门比之前在语言模型上研究的后门要强大得多，我们发现使用常见的后门攻击技术来植入它们要困难得多。我们调查了RLHF中有助于其所谓的健壮性的设计决策，并发布了一个有毒模型的基准，以刺激未来对通用越狱后门的研究。



## **31. Segment (Almost) Nothing: Prompt-Agnostic Adversarial Attacks on Segmentation Models**

细分(几乎)无：对细分模型的即时不可知的对抗性攻击 cs.CV

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14450v1) [paper-pdf](http://arxiv.org/pdf/2311.14450v1)

**Authors**: Francesco Croce, Matthias Hein

**Abstract**: General purpose segmentation models are able to generate (semantic) segmentation masks from a variety of prompts, including visual (points, boxed, etc.) and textual (object names) ones. In particular, input images are pre-processed by an image encoder to obtain embedding vectors which are later used for mask predictions. Existing adversarial attacks target the end-to-end tasks, i.e. aim at altering the segmentation mask predicted for a specific image-prompt pair. However, this requires running an individual attack for each new prompt for the same image. We propose instead to generate prompt-agnostic adversarial attacks by maximizing the $\ell_2$-distance, in the latent space, between the embedding of the original and perturbed images. Since the encoding process only depends on the image, distorted image representations will cause perturbations in the segmentation masks for a variety of prompts. We show that even imperceptible $\ell_\infty$-bounded perturbations of radius $\epsilon=1/255$ are often sufficient to drastically modify the masks predicted with point, box and text prompts by recently proposed foundation models for segmentation. Moreover, we explore the possibility of creating universal, i.e. non image-specific, attacks which can be readily applied to any input without further computational cost.

摘要: 通用分割模型能够根据各种提示生成（语义）分割掩码，包括视觉（点、框等）和文本（对象名称）。特别地，输入图像由图像编码器预处理以获得稍后用于掩模预测的嵌入向量。现有的对抗性攻击针对端到端任务，即旨在改变针对特定图像提示对预测的分割掩码。但是，这需要对同一映像的每个新提示运行单独的攻击。相反，我们建议通过在潜在空间中最大化原始图像和扰动图像之间的嵌入距离来生成不可知的对抗攻击。由于编码过程仅取决于图像，因此失真的图像表示将导致各种提示的分割掩码中的扰动。我们发现，即使是难以察觉的$\ell_\infty$有界扰动的半径$\bytes =1/255$往往足以大幅修改的掩模预测点，框和文本提示最近提出的基础模型分割。此外，我们探讨了创建通用的，即非图像特定的，可以很容易地应用于任何输入，而无需进一步的计算成本的攻击的可能性。



## **32. Federated Transformed Learning for a Circular, Secure, and Tiny AI**

用于圆形、安全和微型人工智能的联合转型学习 cs.NI

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14371v1) [paper-pdf](http://arxiv.org/pdf/2311.14371v1)

**Authors**: Weisi Guo, Schyler Sun, Bin Li, Sam Blakeman

**Abstract**: Deep Learning (DL) is penetrating into a diverse range of mass mobility, smart living, and industrial applications, rapidly transforming the way we live and work. DL is at the heart of many AI implementations. A key set of challenges is to produce AI modules that are: (1) "circular" - can solve new tasks without forgetting how to solve previous ones, (2) "secure" - have immunity to adversarial data attacks, and (3) "tiny" - implementable in low power low cost embedded hardware. Clearly it is difficult to achieve all three aspects on a single horizontal layer of platforms, as the techniques require transformed deep representations that incur different computation and communication requirements. Here we set out the vision to achieve transformed DL representations across a 5G and Beyond networked architecture. We first detail the cross-sectoral motivations for each challenge area, before demonstrating recent advances in DL research that can achieve circular, secure, and tiny AI (CST-AI). Recognising the conflicting demand of each transformed deep representation, we federate their deep learning transformations and functionalities across the network to achieve connected run-time capabilities.

摘要: 深度学习(DL)正在渗透到大众移动性、智能生活和工业应用的各种领域，迅速改变我们的生活和工作方式。动态链接库是许多人工智能实现的核心。一组关键的挑战是产生人工智能模块，这些模块是：(1)“循环”--可以在不忘记如何解决以前的任务的情况下解决新的任务；(2)“安全”--对敌意数据攻击具有免疫力；以及(3)“微小”--可在低功耗、低成本的嵌入式硬件中实现。显然，在单个水平平台层上实现所有这三个方面是困难的，因为这些技术需要转换的深层表示，这会导致不同的计算和通信要求。在这里，我们制定了在5G和网络架构之外实现转型的DL表示的愿景。我们首先详细说明每个挑战领域的跨部门动机，然后展示可以实现循环、安全和微小人工智能(CST-AI)的DL研究的最新进展。认识到每个转换的深度表示的相互冲突的需求，我们将它们的深度学习转换和网络上的功能联合起来，以实现连接的运行时能力。



## **33. BrainWash: A Poisoning Attack to Forget in Continual Learning**

洗脑：在持续学习中忘记的毒药攻击 cs.LG

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.11995v3) [paper-pdf](http://arxiv.org/pdf/2311.11995v3)

**Authors**: Ali Abbasi, Parsa Nooralinejad, Hamed Pirsiavash, Soheil Kolouri

**Abstract**: Continual learning has gained substantial attention within the deep learning community, offering promising solutions to the challenging problem of sequential learning. Yet, a largely unexplored facet of this paradigm is its susceptibility to adversarial attacks, especially with the aim of inducing forgetting. In this paper, we introduce "BrainWash," a novel data poisoning method tailored to impose forgetting on a continual learner. By adding the BrainWash noise to a variety of baselines, we demonstrate how a trained continual learner can be induced to forget its previously learned tasks catastrophically, even when using these continual learning baselines. An important feature of our approach is that the attacker requires no access to previous tasks' data and is armed merely with the model's current parameters and the data belonging to the most recent task. Our extensive experiments highlight the efficacy of BrainWash, showcasing degradation in performance across various regularization-based continual learning methods.

摘要: 持续学习在深度学习界得到了广泛的关注，为顺序学习这一具有挑战性的问题提供了有希望的解决方案。然而，这一范式的一个很大程度上没有被探索的方面是它对敌意攻击的敏感性，特别是以诱导遗忘为目的。在这篇文章中，我们介绍了“洗脑”，一种新的数据中毒方法，专门为不断学习的人强加遗忘。通过将洗脑噪声添加到各种基线中，我们演示了如何诱导训练有素的持续学习者灾难性地忘记其先前学习的任务，即使使用这些持续学习基线也是如此。我们方法的一个重要特征是攻击者不需要访问以前任务的数据，并且只用模型的当前参数和属于最近任务的数据武装起来。我们广泛的实验突出了洗脑的有效性，展示了各种基于正则化的持续学习方法在表现上的下降。



## **34. When Side-Channel Attacks Break the Black-Box Property of Embedded Artificial Intelligence**

当旁路攻击破坏嵌入式人工智能的黑箱特性时 cs.AI

**SubmitDate**: 2023-11-23    [abs](http://arxiv.org/abs/2311.14005v1) [paper-pdf](http://arxiv.org/pdf/2311.14005v1)

**Authors**: Benoit Coqueret, Mathieu Carbone, Olivier Sentieys, Gabriel Zaid

**Abstract**: Artificial intelligence, and specifically deep neural networks (DNNs), has rapidly emerged in the past decade as the standard for several tasks from specific advertising to object detection. The performance offered has led DNN algorithms to become a part of critical embedded systems, requiring both efficiency and reliability. In particular, DNNs are subject to malicious examples designed in a way to fool the network while being undetectable to the human observer: the adversarial examples. While previous studies propose frameworks to implement such attacks in black box settings, those often rely on the hypothesis that the attacker has access to the logits of the neural network, breaking the assumption of the traditional black box. In this paper, we investigate a real black box scenario where the attacker has no access to the logits. In particular, we propose an architecture-agnostic attack which solve this constraint by extracting the logits. Our method combines hardware and software attacks, by performing a side-channel attack that exploits electromagnetic leakages to extract the logits for a given input, allowing an attacker to estimate the gradients and produce state-of-the-art adversarial examples to fool the targeted neural network. Through this example of adversarial attack, we demonstrate the effectiveness of logits extraction using side-channel as a first step for more general attack frameworks requiring either the logits or the confidence scores.

摘要: 人工智能，特别是深度神经网络(DNN)，在过去十年中迅速崛起，成为从特定广告到物体检测等多项任务的标准。所提供的性能使得DNN算法成为关键嵌入式系统的一部分，对效率和可靠性都有要求。特别是，DNN受到恶意示例的影响，这些恶意示例旨在愚弄网络，同时又无法被人类观察者发现：敌意示例。虽然以前的研究提出了在黑盒环境中实施此类攻击的框架，但这些框架通常依赖于攻击者可以访问神经网络的逻辑的假设，打破了传统黑盒的假设。在本文中，我们调查了一个真实的黑盒场景，其中攻击者没有访问日志的权限。特别是，我们提出了一种与体系结构无关的攻击，通过提取日志来解决这一约束。我们的方法结合了硬件和软件攻击，通过执行侧通道攻击，利用电磁泄漏来提取给定输入的逻辑，允许攻击者估计梯度并生成最先进的对抗性示例来愚弄目标神经网络。通过这个对抗性攻击的例子，我们展示了使用旁路提取LOGITS作为需要LOGITS或置信度分数的更一般攻击框架的第一步的有效性。



## **35. An Extensive Study on Adversarial Attack against Pre-trained Models of Code**

对预训练代码模型的对抗性攻击的扩展研究 cs.CR

Accepted to ESEC/FSE 2023

**SubmitDate**: 2023-11-23    [abs](http://arxiv.org/abs/2311.07553v2) [paper-pdf](http://arxiv.org/pdf/2311.07553v2)

**Authors**: Xiaohu Du, Ming Wen, Zichao Wei, Shangwen Wang, Hai Jin

**Abstract**: Transformer-based pre-trained models of code (PTMC) have been widely utilized and have achieved state-of-the-art performance in many mission-critical applications. However, they can be vulnerable to adversarial attacks through identifier substitution or coding style transformation, which can significantly degrade accuracy and may further incur security concerns. Although several approaches have been proposed to generate adversarial examples for PTMC, the effectiveness and efficiency of such approaches, especially on different code intelligence tasks, has not been well understood. To bridge this gap, this study systematically analyzes five state-of-the-art adversarial attack approaches from three perspectives: effectiveness, efficiency, and the quality of generated examples. The results show that none of the five approaches balances all these perspectives. Particularly, approaches with a high attack success rate tend to be time-consuming; the adversarial code they generate often lack naturalness, and vice versa. To address this limitation, we explore the impact of perturbing identifiers under different contexts and find that identifier substitution within for and if statements is the most effective. Based on these findings, we propose a new approach that prioritizes different types of statements for various tasks and further utilizes beam search to generate adversarial examples. Evaluation results show that it outperforms the state-of-the-art ALERT in terms of both effectiveness and efficiency while preserving the naturalness of the generated adversarial examples.

摘要: 基于变压器的预训练代码模型(PTMC)已被广泛使用，并在许多任务关键型应用中取得了最先进的性能。然而，通过标识符替换或编码样式转换，它们很容易受到敌意攻击，这可能会显著降低准确性，并可能进一步引起安全问题。虽然已经提出了几种方法来为PTMC生成对抗性例子，但这些方法的有效性和效率，特别是在不同的代码情报任务上，还没有被很好地理解。为了弥补这一差距，本研究从有效性、效率和生成实例的质量三个角度系统地分析了五种最新的对抗性攻击方法。结果表明，这五种方法都不能平衡所有这些观点。特别是，攻击成功率高的方法往往很耗时；它们生成的敌意代码通常缺乏自然性，反之亦然。为了解决这一限制，我们研究了在不同上下文中干扰标识符所产生的影响，并发现在for和if语句中替换标识符是最有效的。基于这些发现，我们提出了一种新的方法，该方法对不同任务的不同类型的语句进行优先排序，并进一步利用BEAM搜索来生成对抗性示例。评估结果表明，该算法在保持生成的对抗性实例的自然性的同时，在有效性和效率上都优于最新的警报。



## **36. Adversarial defense based on distribution transfer**

基于分布转移的对抗性防御 cs.CR

27 pages

**SubmitDate**: 2023-11-23    [abs](http://arxiv.org/abs/2311.13841v1) [paper-pdf](http://arxiv.org/pdf/2311.13841v1)

**Authors**: Jiahao Chen, Diqun Yan, Li Dong

**Abstract**: The presence of adversarial examples poses a significant threat to deep learning models and their applications. Existing defense methods provide certain resilience against adversarial examples, but often suffer from decreased accuracy and generalization performance, making it challenging to achieve a trade-off between robustness and generalization. To address this, our paper interprets the adversarial example problem from the perspective of sample distribution and proposes a defense method based on distribution shift, leveraging the distribution transfer capability of a diffusion model for adversarial defense. The core idea is to exploit the discrepancy between normal and adversarial sample distributions to achieve adversarial defense using a pretrained diffusion model. Specifically, an adversarial sample undergoes a forward diffusion process, moving away from the source distribution, followed by a reverse process guided by the protected model (victim model) output to map it back to the normal distribution. Experimental evaluations on CIFAR10 and ImageNet30 datasets are conducted, comparing with adversarial training and input preprocessing methods. For infinite-norm attacks with 8/255 perturbation, accuracy rates of 78.1% and 83.5% are achieved, respectively. For 2-norm attacks with 128/255 perturbation, accuracy rates are 74.3% and 82.5%. Additional experiments considering perturbation amplitude, diffusion iterations, and adaptive attacks also validate the effectiveness of the proposed method. Results demonstrate that even when the attacker has knowledge of the defense, the proposed distribution-based method effectively withstands adversarial examples. It fills the gaps of traditional approaches, restoring high-quality original samples and showcasing superior performance in model robustness and generalization.

摘要: 对抗性例子的存在对深度学习模型及其应用构成了严重威胁。现有的防御方法对敌意例子提供了一定的弹性，但往往存在准确性和泛化性能下降的问题，这使得在稳健性和泛化之间实现权衡具有挑战性。针对这一问题，本文从样本分布的角度对对抗性实例问题进行了解释，并利用扩散模型的分布转移能力，提出了一种基于分布转移的对抗性防御方法。其核心思想是利用正态样本分布和对抗性样本分布之间的差异，使用预先训练好的扩散模型来实现对抗性防御。具体地说，敌方样本经历正向扩散过程，远离源分布，随后是由受保护模型(受害者模型)输出引导的反向过程，以将其映射回正态分布。在CIFAR10和ImageNet30数据集上进行了实验评估，并与对抗性训练和输入预处理方法进行了比较。对于具有8/255扰动的无限范数攻击，正确率分别达到781%和83.5%。对于具有128/255扰动的2范数攻击，准确率分别为74.3%和82.5%。考虑扰动幅度、扩散迭代和自适应攻击的附加实验也验证了该方法的有效性。结果表明，即使攻击者知道防御措施，所提出的基于分布的方法也能有效地抵抗敌方攻击。它填补了传统方法的空白，恢复了高质量的原始样本，并在模型稳健性和泛化方面表现出了优越的性能。



## **37. Security and Privacy Challenges in Deep Learning Models**

深度学习模型中的安全和隐私挑战 cs.CR

**SubmitDate**: 2023-11-23    [abs](http://arxiv.org/abs/2311.13744v1) [paper-pdf](http://arxiv.org/pdf/2311.13744v1)

**Authors**: Gopichandh Golla

**Abstract**: These days, deep learning models have achieved great success in multiple fields, from autonomous driving to medical diagnosis. These models have expanded the abilities of artificial intelligence by offering great solutions to complex problems that were very difficult to solve earlier. In spite of their unseen success in various, it has been identified, through research conducted, that deep learning models can be subjected to various attacks that compromise model security and data privacy of the Deep Neural Network models. Deep learning models can be subjected to various attacks at different stages of their lifecycle. During the testing phase, attackers can exploit vulnerabilities through different kinds of attacks such as Model Extraction Attacks, Model Inversion attacks, and Adversarial attacks. Model Extraction Attacks are aimed at reverse-engineering a trained deep learning model, with the primary objective of revealing its architecture and parameters. Model inversion attacks aim to compromise the privacy of the data used in the Deep learning model. These attacks are done to compromise the confidentiality of the model by going through the sensitive training data from the model's predictions. By analyzing the model's responses, attackers aim to reconstruct sensitive information. In this way, the model's data privacy is compromised. Adversarial attacks, mainly employed on computer vision models, are made to corrupt models into confidently making incorrect predictions through malicious testing data. These attacks subtly alter the input data, making it look normal but misleading deep learning models to make incorrect decisions. Such attacks can happen during both the model's evaluation and training phases. Data Poisoning Attacks add harmful data to the training set, disrupting the learning process and reducing the reliability of the deep learning mode.

摘要: 如今，深度学习模型在从自动驾驶到医疗诊断的多个领域取得了巨大成功。这些模型为以前很难解决的复杂问题提供了很好的解决方案，从而扩展了人工智能的能力。尽管深度学习模型在各种领域取得了前所未有的成功，但通过研究发现，深度学习模型可能会受到各种攻击，从而危及深度神经网络模型的模型安全性和数据隐私。深度学习模型在其生命周期的不同阶段可能会受到各种攻击。在测试阶段，攻击者可以通过模型提取攻击、模型反转攻击和对抗性攻击等不同类型的攻击来利用漏洞。模型提取攻击的目的是对训练好的深度学习模型进行逆向工程，主要目的是揭示其结构和参数。模型反转攻击旨在保护深度学习模型中使用的数据的隐私。这些攻击通过检查模型预测中的敏感训练数据来危害模型的机密性。通过分析模型的响应，攻击者的目标是重建敏感信息。这样一来，该模型的数据隐私就被泄露了。对抗性攻击主要应用于计算机视觉模型，通过恶意测试数据破坏模型，使其自信地做出错误的预测。这些攻击巧妙地改变了输入数据，使其看起来很正常，但误导了深度学习模型，使其做出错误的决策。这类攻击可能发生在模型的评估和训练阶段。数据中毒攻击将有害数据添加到训练集中，扰乱了学习过程，降低了深度学习模式的可靠性。



## **38. Panda or not Panda? Understanding Adversarial Attacks with Interactive Visualization**

熊猫还是不熊猫？利用交互可视化理解敌意攻击 cs.HC

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.13656v1) [paper-pdf](http://arxiv.org/pdf/2311.13656v1)

**Authors**: Yuzhe You, Jarvis Tse, Jian Zhao

**Abstract**: Adversarial machine learning (AML) studies attacks that can fool machine learning algorithms into generating incorrect outcomes as well as the defenses against worst-case attacks to strengthen model robustness. Specifically for image classification, it is challenging to understand adversarial attacks due to their use of subtle perturbations that are not human-interpretable, as well as the variability of attack impacts influenced by diverse methodologies, instance differences, and model architectures. Through a design study with AML learners and teachers, we introduce AdvEx, a multi-level interactive visualization system that comprehensively presents the properties and impacts of evasion attacks on different image classifiers for novice AML learners. We quantitatively and qualitatively assessed AdvEx in a two-part evaluation including user studies and expert interviews. Our results show that AdvEx is not only highly effective as a visualization tool for understanding AML mechanisms, but also provides an engaging and enjoyable learning experience, thus demonstrating its overall benefits for AML learners.

摘要: 对抗性机器学习(AML)研究可以欺骗机器学习算法产生错误结果的攻击，以及对最坏情况下的攻击的防御，以增强模型的健壮性。具体地说，对于图像分类，理解对抗性攻击是具有挑战性的，因为它们使用了人类无法解释的微妙扰动，以及受不同方法、实例差异和模型架构影响的攻击影响的可变性。通过与AML学习者和教师的设计研究，我们介绍了Advex，一个多层次的交互式可视化系统，它为AML初学者全面展示了不同图像分类器上的逃避攻击的特性和影响。我们在包括用户研究和专家访谈的两部分评估中对Advex进行了定量和定性的评估。我们的结果表明，Advex不仅作为一种高效的可视化工具来理解AML的机制，而且还提供了一种引人入胜和愉快的学习体验，从而展示了它对AML学习者的整体好处。



## **39. Adversarial Backdoor Attack by Naturalistic Data Poisoning on Trajectory Prediction in Autonomous Driving**

自动驾驶轨迹预测中自然主义数据中毒的对抗性后门攻击 cs.CV

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2306.15755v2) [paper-pdf](http://arxiv.org/pdf/2306.15755v2)

**Authors**: Mozhgan Pourkeshavarz, Mohammad Sabokrou, Amir Rasouli

**Abstract**: In autonomous driving, behavior prediction is fundamental for safe motion planning, hence the security and robustness of prediction models against adversarial attacks are of paramount importance. We propose a novel adversarial backdoor attack against trajectory prediction models as a means of studying their potential vulnerabilities. Our attack affects the victim at training time via naturalistic, hence stealthy, poisoned samples crafted using a novel two-step approach. First, the triggers are crafted by perturbing the trajectory of attacking vehicle and then disguised by transforming the scene using a bi-level optimization technique. The proposed attack does not depend on a particular model architecture and operates in a black-box manner, thus can be effective without any knowledge of the victim model. We conduct extensive empirical studies using state-of-the-art prediction models on two benchmark datasets using metrics customized for trajectory prediction. We show that the proposed attack is highly effective, as it can significantly hinder the performance of prediction models, unnoticeable by the victims, and efficient as it forces the victim to generate malicious behavior even under constrained conditions. Via ablative studies, we analyze the impact of different attack design choices followed by an evaluation of existing defence mechanisms against the proposed attack.

摘要: 在自动驾驶中，行为预测是安全运动规划的基础，因此预测模型对对手攻击的安全性和稳健性至关重要。我们提出了一种新的针对轨迹预测模型的对抗性后门攻击，作为研究其潜在漏洞的一种手段。我们的攻击在训练时间通过使用新的两步法制作的自然主义的、因此是隐形的、有毒的样本来影响受害者。首先，通过扰动攻击车辆的轨迹来制作触发器，然后使用双层优化技术对场景进行变换来伪装。提出的攻击不依赖于特定的模型体系结构，并且以黑盒方式运行，因此在不了解受害者模型的情况下可以有效地进行攻击。我们在两个基准数据集上使用最先进的预测模型进行了广泛的实证研究，使用了为轨迹预测定制的指标。我们证明了该攻击是高效的，因为它可以显著地阻碍预测模型的性能，受害者不会注意到它，并且由于它迫使受害者即使在约束条件下也产生恶意行为，所以是有效的。通过烧蚀研究，我们分析了不同的攻击设计选择的影响，然后评估了现有的防御机制对拟议的攻击。



## **40. Transfer Attacks and Defenses for Large Language Models on Coding Tasks**

编码任务中大语言模型的迁移攻击与防御 cs.LG

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.13445v1) [paper-pdf](http://arxiv.org/pdf/2311.13445v1)

**Authors**: Chi Zhang, Zifan Wang, Ravi Mangal, Matt Fredrikson, Limin Jia, Corina Pasareanu

**Abstract**: Modern large language models (LLMs), such as ChatGPT, have demonstrated impressive capabilities for coding tasks including writing and reasoning about code. They improve upon previous neural network models of code, such as code2seq or seq2seq, that already demonstrated competitive results when performing tasks such as code summarization and identifying code vulnerabilities. However, these previous code models were shown vulnerable to adversarial examples, i.e. small syntactic perturbations that do not change the program's semantics, such as the inclusion of "dead code" through false conditions or the addition of inconsequential print statements, designed to "fool" the models. LLMs can also be vulnerable to the same adversarial perturbations but a detailed study on this concern has been lacking so far. In this paper we aim to investigate the effect of adversarial perturbations on coding tasks with LLMs. In particular, we study the transferability of adversarial examples, generated through white-box attacks on smaller code models, to LLMs. Furthermore, to make the LLMs more robust against such adversaries without incurring the cost of retraining, we propose prompt-based defenses that involve modifying the prompt to include additional information such as examples of adversarially perturbed code and explicit instructions for reversing adversarial perturbations. Our experiments show that adversarial examples obtained with a smaller code model are indeed transferable, weakening the LLMs' performance. The proposed defenses show promise in improving the model's resilience, paving the way to more robust defensive solutions for LLMs in code-related applications.

摘要: 现代大型语言模型(LLM)，如ChatGPT，已经在编码任务(包括编写代码和进行推理)方面展示了令人印象深刻的能力。它们改进了以前的代码神经网络模型，如code2seq或seq2seq，这些模型在执行代码汇总和识别代码漏洞等任务时已经展示了具有竞争力的结果。然而，这些以前的代码模型被证明容易受到敌意示例的攻击，即不会改变程序语义的小的语法扰动，例如通过错误条件包括“死代码”或添加无关紧要的打印语句，旨在“愚弄”模型。LLMS也可能容易受到同样的对抗性干扰，但迄今为止还缺乏关于这一问题的详细研究。本文旨在研究对抗性扰动对LLMS编码任务的影响。特别是，我们研究了通过对较小代码模型进行白盒攻击而生成的对抗性示例到LLMS的可转移性。此外，为了使LLMS在不招致再培训成本的情况下对此类对手更加健壮，我们提出了基于提示的防御措施，涉及修改提示以包括额外的信息，例如对手扰动代码的示例和用于逆转对手扰动的显式指令。我们的实验表明，用较小的编码模型得到的对抗性例子确实是可移植的，从而削弱了LLMS的性能。拟议的防御措施在提高模型的弹性方面显示出了希望，为代码相关应用中的LLM提供更强大的防御解决方案铺平了道路。



## **41. From Principle to Practice: Vertical Data Minimization for Machine Learning**

从原理到实践：机器学习的垂直数据最小化 cs.LG

Accepted at IEEE S&P 2024

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.10500v2) [paper-pdf](http://arxiv.org/pdf/2311.10500v2)

**Authors**: Robin Staab, Nikola Jovanović, Mislav Balunović, Martin Vechev

**Abstract**: Aiming to train and deploy predictive models, organizations collect large amounts of detailed client data, risking the exposure of private information in the event of a breach. To mitigate this, policymakers increasingly demand compliance with the data minimization (DM) principle, restricting data collection to only that data which is relevant and necessary for the task. Despite regulatory pressure, the problem of deploying machine learning models that obey DM has so far received little attention. In this work, we address this challenge in a comprehensive manner. We propose a novel vertical DM (vDM) workflow based on data generalization, which by design ensures that no full-resolution client data is collected during training and deployment of models, benefiting client privacy by reducing the attack surface in case of a breach. We formalize and study the corresponding problem of finding generalizations that both maximize data utility and minimize empirical privacy risk, which we quantify by introducing a diverse set of policy-aligned adversarial scenarios. Finally, we propose a range of baseline vDM algorithms, as well as Privacy-aware Tree (PAT), an especially effective vDM algorithm that outperforms all baselines across several settings. We plan to release our code as a publicly available library, helping advance the standardization of DM for machine learning. Overall, we believe our work can help lay the foundation for further exploration and adoption of DM principles in real-world applications.

摘要: 为了训练和部署预测模型，组织收集了大量详细的客户数据，在发生入侵时冒着私人信息暴露的风险。为了缓解这一问题，政策制定者越来越多地要求遵守数据最小化(DM)原则，将数据收集仅限于与任务相关和必要的数据。尽管面临监管压力，但到目前为止，部署服从DM的机器学习模型的问题几乎没有得到关注。在这项工作中，我们以全面的方式应对这一挑战。我们提出了一种基于数据泛化的垂直数据挖掘(VDM)工作流，该工作流在设计上确保了在模型的训练和部署过程中不收集全分辨率的客户数据，从而减少了在发生攻击时的攻击面，从而有利于客户隐私。我们形式化并研究了相应的问题，即找到既最大化数据效用又最小化经验隐私风险的概括，我们通过引入一组与策略一致的对抗场景来量化这些概括。最后，我们提出了一系列的基线VDM算法，以及隐私感知树(PAT)，这是一种特别有效的VDM算法，其性能在几种设置下都优于所有基线。我们计划将我们的代码作为一个公开的库发布，帮助推进机器学习的DM标准化。总体而言，我们相信我们的工作可以为进一步探索和采用DM原理在现实世界中的应用奠定基础。



## **42. Hard Label Black Box Node Injection Attack on Graph Neural Networks**

基于图神经网络的硬标签黑盒节点注入攻击 cs.LG

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.13244v1) [paper-pdf](http://arxiv.org/pdf/2311.13244v1)

**Authors**: Yu Zhou, Zihao Dong, Guofeng Zhang, Jingchen Tang

**Abstract**: While graph neural networks have achieved state-of-the-art performances in many real-world tasks including graph classification and node classification, recent works have demonstrated they are also extremely vulnerable to adversarial attacks. Most previous works have focused on attacking node classification networks under impractical white-box scenarios. In this work, we will propose a non-targeted Hard Label Black Box Node Injection Attack on Graph Neural Networks, which to the best of our knowledge, is the first of its kind. Under this setting, more real world tasks can be studied because our attack assumes no prior knowledge about (1): the model architecture of the GNN we are attacking; (2): the model's gradients; (3): the output logits of the target GNN model. Our attack is based on an existing edge perturbation attack, from which we restrict the optimization process to formulate a node injection attack. In the work, we will evaluate the performance of the attack using three datasets, COIL-DEL, IMDB-BINARY, and NCI1.

摘要: 虽然图神经网络在包括图分类和节点分类在内的许多现实任务中都取得了最先进的性能，但最近的研究表明，它们也非常容易受到对手攻击。以往的工作大多集中在不切实际的白盒场景下对节点分类网络的攻击。在这项工作中，我们将提出一种针对图神经网络的无目标硬标签黑盒节点注入攻击，据我们所知，这是此类攻击中的第一次。在这种情况下，我们可以研究更多现实世界的任务，因为我们的攻击不假设关于(1)：我们正在攻击的GNN的模型体系结构；(2)：模型的梯度；(3)：目标GNN模型的输出逻辑。我们的攻击是基于已有的边扰动攻击，从该边扰动攻击出发，我们限制了优化过程，形成了节点注入攻击。在工作中，我们将使用三个数据集COIL-DEL、IMDB-BINARY和NCI1来评估攻击的性能。



## **43. A Survey of Adversarial CAPTCHAs on its History, Classification and Generation**

对抗性验证码的历史沿革、分类及产生 cs.CR

Submitted to ACM Computing Surveys (Under Review)

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.13233v1) [paper-pdf](http://arxiv.org/pdf/2311.13233v1)

**Authors**: Zisheng Xu, Qiao Yan, F. Richard Yu, Victor C. M. Leung

**Abstract**: Completely Automated Public Turing test to tell Computers and Humans Apart, short for CAPTCHA, is an essential and relatively easy way to defend against malicious attacks implemented by bots. The security and usability trade-off limits the use of massive geometric transformations to interfere deep model recognition and deep models even outperformed humans in complex CAPTCHAs. The discovery of adversarial examples provides an ideal solution to the security and usability trade-off by integrating adversarial examples and CAPTCHAs to generate adversarial CAPTCHAs that can fool the deep models. In this paper, we extend the definition of adversarial CAPTCHAs and propose a classification method for adversarial CAPTCHAs. Then we systematically review some commonly used methods to generate adversarial examples and methods that are successfully used to generate adversarial CAPTCHAs. Also, we analyze some defense methods that can be used to defend adversarial CAPTCHAs, indicating potential threats to adversarial CAPTCHAs. Finally, we discuss some possible future research directions for adversarial CAPTCHAs at the end of this paper.

摘要: 全自动公共图灵测试区分计算机和人类，简称验证码，是防御机器人实施的恶意攻击的一种基本且相对容易的方法。安全性和可用性的权衡限制了使用大规模几何变换来干扰深度模型识别，而深度模型在复杂验证码中的表现甚至超过了人类。对抗性实例的发现通过集成对抗性实例和验证码来生成可以欺骗深层模型的对抗性验证码，从而为安全性和可用性的权衡提供了一个理想的解决方案。本文扩展了对抗性验证码的定义，提出了一种对抗性验证码的分类方法。然后，我们系统地回顾了一些常用的生成对抗性实例的方法以及成功地生成对抗性验证码的方法。此外，我们还分析了一些可用于防御对抗性验证码的防御方法，指出了对抗性验证码的潜在威胁。最后，我们讨论了对抗性验证码未来可能的研究方向。



## **44. HINT: Healthy Influential-Noise based Training to Defend against Data Poisoning Attacks**

提示：健康影响-基于噪音的培训可防御数据中毒攻击 cs.LG

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2309.08549v3) [paper-pdf](http://arxiv.org/pdf/2309.08549v3)

**Authors**: Minh-Hao Van, Alycia N. Carey, Xintao Wu

**Abstract**: While numerous defense methods have been proposed to prohibit potential poisoning attacks from untrusted data sources, most research works only defend against specific attacks, which leaves many avenues for an adversary to exploit. In this work, we propose an efficient and robust training approach to defend against data poisoning attacks based on influence functions, named Healthy Influential-Noise based Training. Using influence functions, we craft healthy noise that helps to harden the classification model against poisoning attacks without significantly affecting the generalization ability on test data. In addition, our method can perform effectively when only a subset of the training data is modified, instead of the current method of adding noise to all examples that has been used in several previous works. We conduct comprehensive evaluations over two image datasets with state-of-the-art poisoning attacks under different realistic attack scenarios. Our empirical results show that HINT can efficiently protect deep learning models against the effect of both untargeted and targeted poisoning attacks.

摘要: 虽然已经提出了许多防御方法来阻止来自不受信任的数据源的潜在中毒攻击，但大多数研究工作只防御特定的攻击，这给对手留下了许多可以利用的途径。在这项工作中，我们提出了一种基于影响函数的高效、健壮的数据中毒攻击训练方法，即基于健康影响噪声的训练方法。利用影响函数构造健康噪声，在不显著影响测试数据泛化能力的情况下，有助于加强分类模型对中毒攻击的抵抗能力。此外，我们的方法可以在只修改训练数据的子集的情况下有效地执行，而不是在以前的几个工作中使用的向所有样本添加噪声的方法。在不同的真实攻击场景下，我们对两个具有最新技术的中毒攻击的图像数据集进行了综合评估。我们的实验结果表明，提示可以有效地保护深度学习模型免受非定向和定向中毒攻击的影响。



## **45. Epsilon*: Privacy Metric for Machine Learning Models**

Epsilon*：机器学习模型的隐私度量 cs.LG

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2307.11280v2) [paper-pdf](http://arxiv.org/pdf/2307.11280v2)

**Authors**: Diana M. Negoescu, Humberto Gonzalez, Saad Eddin Al Orjany, Jilei Yang, Yuliia Lut, Rahul Tandra, Xiaowen Zhang, Xinyi Zheng, Zach Douglas, Vidita Nolkha, Parvez Ahammad, Gennady Samorodnitsky

**Abstract**: We introduce Epsilon*, a new privacy metric for measuring the privacy risk of a single model instance prior to, during, or after deployment of privacy mitigation strategies. The metric requires only black-box access to model predictions, does not require training data re-sampling or model re-training, and can be used to measure the privacy risk of models not trained with differential privacy. Epsilon* is a function of true positive and false positive rates in a hypothesis test used by an adversary in a membership inference attack. We distinguish between quantifying the privacy loss of a trained model instance, which we refer to as empirical privacy, and quantifying the privacy loss of the training mechanism which produces this model instance. Existing approaches in the privacy auditing literature provide lower bounds for the latter, while our metric provides an empirical lower bound for the former by relying on an (${\epsilon}$, ${\delta}$)-type of quantification of the privacy of the trained model instance. We establish a relationship between these lower bounds and show how to implement Epsilon* to avoid numerical and noise amplification instability. We further show in experiments on benchmark public data sets that Epsilon* is sensitive to privacy risk mitigation by training with differential privacy (DP), where the value of Epsilon* is reduced by up to 800% compared to the Epsilon* values of non-DP trained baseline models. This metric allows privacy auditors to be independent of model owners, and enables visualizing the privacy-utility landscape to make informed decisions regarding the trade-offs between model privacy and utility.

摘要: 我们引入了Epsilon*，这是一种新的隐私度量标准，用于在部署隐私缓解策略之前、期间或之后衡量单个模型实例的隐私风险。该度量只需要黑盒访问模型预测，不需要重新采样训练数据或重新训练模型，并且可以用于测量未使用差异隐私训练的模型的隐私风险。Epsilon*是对手在成员关系推断攻击中使用的假设检验中真阳性和假阳性率的函数。我们区分量化训练的模型实例的隐私损失，我们称之为经验隐私，和量化产生该模型实例的训练机制的隐私损失。隐私审计文献中的现有方法为后者提供了下界，而我们的度量通过依赖于($\epsilon}$，${\Delta}$)类型的训练模型实例的隐私量化，为前者提供了经验下限。我们建立了这些下界之间的关系，并展示了如何实现Epsilon*以避免数值和噪声放大的不稳定性。我们在基准公共数据集上的实验进一步表明，Epsilon*通过使用差异隐私(DP)进行训练对隐私风险缓解非常敏感，其中Epsilon*的值与未使用DP训练的基线模型的Epsilon*值相比降低了800%。这一指标允许隐私审核员独立于模型所有者，并使隐私效用环境可视化，以便就模型隐私和效用之间的权衡做出明智的决定。



## **46. Is your vote truly secret? Ballot Secrecy iff Ballot Independence: Proving necessary conditions and analysing case studies**

你的投票真的是秘密的吗？选票保密性与选票独立性：必要条件证明与案例分析 cs.CR

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12977v1) [paper-pdf](http://arxiv.org/pdf/2311.12977v1)

**Authors**: Aida Manzano Kharman, Ben Smyth, Freddie Page

**Abstract**: We formalise definitions of ballot secrecy and ballot independence by Smyth, JCS'21 as indistinguishability games in the computational model of security. These definitions improve upon Smyth, draft '21 to consider a wider class of voting systems. Both Smyth, JCS'21 and Smyth, draft '21 improve on earlier works by considering a more realistic adversary model wherein they have access to the ballot collection. We prove that ballot secrecy implies ballot independence. We say ballot independence holds if a system has non-malleable ballots. We construct games for ballot secrecy and non-malleability and show that voting schemes with malleable ballots do not preserve ballot secrecy. We demonstrate that Helios does not satisfy our definition of ballot secrecy. Furthermore, the Python framework we constructed for our case study shows that if an attack exists against non-malleability, this attack can be used to break ballot secrecy.

摘要: 我们将Smyth，JCS‘21对选票保密性和选票独立性的定义形式化为安全计算模型中的不可区分博弈。这些定义改进了Smyth，草案‘21，以考虑更广泛的投票制度类别。Smyth，JCS‘21和Smyth，草案’21都通过考虑一个更现实的对手模型来改进早期的作品，在这个模型中，他们可以获得选票收集。我们证明了选票的保密性意味着选票的独立性。我们说，如果一个系统有不可延展的选票，那么选票独立性就成立了。我们构造了选票保密性和不可延展性的对策，并证明了具有延展性选票的投票方案不能保护选票的保密性。我们证明，太阳神并不满足我们对投票保密的定义。此外，我们为我们的案例研究构建的Python框架表明，如果存在针对不可延展性的攻击，则此攻击可用于打破投票保密性。



## **47. Iris Presentation Attack: Assessing the Impact of Combining Vanadium Dioxide Films with Artificial Eyes**

虹膜呈现攻击：评估二氧化钒薄膜与假眼结合的影响 cs.CV

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12773v1) [paper-pdf](http://arxiv.org/pdf/2311.12773v1)

**Authors**: Darshika Jauhari, Renu Sharma, Cunjian Chen, Nelson Sepulveda, Arun Ross

**Abstract**: Iris recognition systems, operating in the near infrared spectrum (NIR), have demonstrated vulnerability to presentation attacks, where an adversary uses artifacts such as cosmetic contact lenses, artificial eyes or printed iris images in order to circumvent the system. At the same time, a number of effective presentation attack detection (PAD) methods have been developed. These methods have demonstrated success in detecting artificial eyes (e.g., fake Van Dyke eyes) as presentation attacks. In this work, we seek to alter the optical characteristics of artificial eyes by affixing Vanadium Dioxide (VO2) films on their surface in various spatial configurations. VO2 films can be used to selectively transmit NIR light and can, therefore, be used to regulate the amount of NIR light from the object that is captured by the iris sensor. We study the impact of such images produced by the sensor on two state-of-the-art iris PA detection methods. We observe that the addition of VO2 films on the surface of artificial eyes can cause the PA detection methods to misclassify them as bonafide eyes in some cases. This represents a vulnerability that must be systematically analyzed and effectively addressed.

摘要: 在近红外光谱（NIR）中操作的虹膜识别系统已经证明了对呈现攻击的脆弱性，其中对手使用诸如化妆品隐形眼镜、人造眼睛或打印的虹膜图像之类的伪像来规避系统。与此同时，一些有效的呈现攻击检测（PAD）方法已经开发出来。这些方法已经证明在检测人造眼睛（例如，假范戴克眼睛）作为演示攻击。在这项工作中，我们试图改变人工眼的光学特性，通过贴附二氧化钒（VO2）薄膜在其表面上的各种空间配置。VO2薄膜可用于选择性地透射NIR光，因此可用于调节虹膜传感器捕获的来自物体的NIR光的量。我们研究了两个国家的最先进的虹膜PA检测方法的传感器产生的图像的影响。我们观察到，在人工眼表面上添加VO2膜会导致PA检测方法在某些情况下将其误分类为真正的眼睛。这是一个必须系统分析和有效解决的脆弱性。



## **48. Attention Deficit is Ordered! Fooling Deformable Vision Transformers with Collaborative Adversarial Patches**

注意力缺陷是命中注定的！用协同对抗性补丁愚弄可变形视觉变形器 cs.CV

9 pages, 10 figures

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12914v1) [paper-pdf](http://arxiv.org/pdf/2311.12914v1)

**Authors**: Quazi Mishkatul Alam, Bilel Tarchoun, Ihsen Alouani, Nael Abu-Ghazaleh

**Abstract**: The latest generation of transformer-based vision models have proven to be superior to Convolutional Neural Network (CNN)-based models across several vision tasks, largely attributed to their remarkable prowess in relation modeling. Deformable vision transformers significantly reduce the quadratic complexity of modeling attention by using sparse attention structures, enabling them to be used in larger scale applications such as multi-view vision systems. Recent work demonstrated adversarial attacks against transformers; we show that these attacks do not transfer to deformable transformers due to their sparse attention structure. Specifically, attention in deformable transformers is modeled using pointers to the most relevant other tokens. In this work, we contribute for the first time adversarial attacks that manipulate the attention of deformable transformers, distracting them to focus on irrelevant parts of the image. We also develop new collaborative attacks where a source patch manipulates attention to point to a target patch that adversarially attacks the system. In our experiments, we find that only 1% patched area of the input field can lead to 0% AP. We also show that the attacks provide substantial versatility to support different attacker scenarios because of their ability to redirect attention under the attacker control.

摘要: 最新一代的基于transformer的视觉模型已经被证明在几个视觉任务中优于基于卷积神经网络（CNN）的模型，这主要归功于它们在关系建模方面的卓越能力。可变形视觉变换器通过使用稀疏注意力结构显著降低了注意力建模的二次复杂度，使其能够用于更大规模的应用，如多视图视觉系统。最近的工作证明了对变压器的对抗性攻击，我们表明，这些攻击不会转移到变形变压器，由于其稀疏的注意力结构。具体地说，可变形器中的注意力是使用指向最相关的其他令牌的指针来建模的。在这项工作中，我们第一次贡献了对抗性攻击，这些攻击操纵可变形器的注意力，分散它们的注意力，使其专注于图像的不相关部分。我们还开发了新的协作攻击，其中源补丁操纵注意力指向一个目标补丁，对抗攻击系统。在我们的实验中，我们发现只有1%的输入场补丁区域可以导致0%的AP。我们还表明，攻击提供了大量的多功能性，以支持不同的攻击者的情况下，因为他们的能力，在攻击者的控制下重新定向注意。



## **49. Attacking Motion Planners Using Adversarial Perception Errors**

使用对抗性感知错误攻击动作规划者 cs.RO

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12722v1) [paper-pdf](http://arxiv.org/pdf/2311.12722v1)

**Authors**: Jonathan Sadeghi, Nicholas A. Lord, John Redford, Romain Mueller

**Abstract**: Autonomous driving (AD) systems are often built and tested in a modular fashion, where the performance of different modules is measured using task-specific metrics. These metrics should be chosen so as to capture the downstream impact of each module and the performance of the system as a whole. For example, high perception quality should enable prediction and planning to be performed safely. Even though this is true in general, we show here that it is possible to construct planner inputs that score very highly on various perception quality metrics but still lead to planning failures. In an analogy to adversarial attacks on image classifiers, we call such inputs \textbf{adversarial perception errors} and show they can be systematically constructed using a simple boundary-attack algorithm. We demonstrate the effectiveness of this algorithm by finding attacks for two different black-box planners in several urban and highway driving scenarios using the CARLA simulator. Finally, we analyse the properties of these attacks and show that they are isolated in the input space of the planner, and discuss their implications for AD system deployment and testing.

摘要: 自动驾驶（AD）系统通常以模块化的方式构建和测试，其中使用特定于任务的指标来测量不同模块的性能。应该选择这些指标，以便捕获每个模块的下游影响和整个系统的性能。例如，高感知质量应该能够安全地执行预测和规划。即使这是真的，在一般情况下，我们在这里表明，它是可能的，以构建规划输入得分非常高的各种感知质量指标，但仍然导致规划失败。在图像分类器上的对抗性攻击的类比中，我们称这种输入为\textbf{对抗性感知错误}，并表明它们可以使用简单的边界攻击算法系统地构建。我们证明了该算法的有效性，找到两个不同的黑盒规划者在几个城市和高速公路驾驶的情况下，使用CARLA模拟器的攻击。最后，我们分析了这些攻击的属性，并表明它们在规划器的输入空间中是孤立的，并讨论了它们对AD系统部署和测试的影响。



## **50. Differentially Private Optimizers Can Learn Adversarially Robust Models**

不同的私有优化器可以学习相反的健壮模型 cs.LG

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2211.08942v2) [paper-pdf](http://arxiv.org/pdf/2211.08942v2)

**Authors**: Yuan Zhang, Zhiqi Bu

**Abstract**: Machine learning models have shone in a variety of domains and attracted increasing attention from both the security and the privacy communities. One important yet worrying question is: Will training models under the differential privacy (DP) constraint have an unfavorable impact on their adversarial robustness? While previous works have postulated that privacy comes at the cost of worse robustness, we give the first theoretical analysis to show that DP models can indeed be robust and accurate, even sometimes more robust than their naturally-trained non-private counterparts. We observe three key factors that influence the privacy-robustness-accuracy tradeoff: (1) hyper-parameters for DP optimizers are critical; (2) pre-training on public data significantly mitigates the accuracy and robustness drop; (3) choice of DP optimizers makes a difference. With these factors set properly, we achieve 90\% natural accuracy, 72\% robust accuracy ($+9\%$ than the non-private model) under $l_2(0.5)$ attack, and 69\% robust accuracy ($+16\%$ than the non-private model) with pre-trained SimCLRv2 model under $l_\infty(4/255)$ attack on CIFAR10 with $\epsilon=2$. In fact, we show both theoretically and empirically that DP models are Pareto optimal on the accuracy-robustness tradeoff. Empirically, the robustness of DP models is consistently observed across various datasets and models. We believe our encouraging results are a significant step towards training models that are private as well as robust.

摘要: 机器学习模型在各个领域都很受欢迎，并引起了安全和隐私社区越来越多的关注。一个重要但令人担忧的问题是：差分隐私（DP）约束下的训练模型是否会对其对抗鲁棒性产生不利影响？虽然以前的工作假设隐私是以更差的鲁棒性为代价的，但我们给出了第一个理论分析，表明DP模型确实可以是鲁棒和准确的，甚至有时比自然训练的非私有模型更鲁棒。我们观察到影响隐私-鲁棒性-准确性权衡的三个关键因素：（1）DP优化器的超参数至关重要;（2）对公共数据进行预训练可以显着减轻准确性和鲁棒性的下降;（3）DP优化器的选择会产生影响。通过合理设置这些因素，我们获得了90%的自然准确率，72%的鲁棒准确率（比非私有模型+9%）在$l_2（0.5）$攻击下，和69%的鲁棒准确率（比非私有模型+16%）在$l_infty（4/255）$攻击CIFAR 10，$\n =2$.事实上，我们从理论和经验上表明，DP模型是帕累托最优的精度鲁棒性权衡。从经验上讲，DP模型的鲁棒性在各种数据集和模型中得到了一致的观察。我们相信，我们令人鼓舞的结果是朝着私人和强大的培训模式迈出的重要一步。



