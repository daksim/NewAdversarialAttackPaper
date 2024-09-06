# Latest Adversarial Attack Papers
**update at 2024-09-06 15:32:00**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner**

SelfDefend：LLM可以以实用的方式保护自己免受越狱的侵害 cs.CR

This paper completes its earlier vision paper, available at  arXiv:2402.15727. Updated to the latest analysis and results

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2406.05498v2) [paper-pdf](http://arxiv.org/pdf/2406.05498v2)

**Authors**: Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, Juergen Rahmel

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into multiple categories: human-based, optimization-based, generation-based, and the recent indirect and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delays to user prompts, as well as be compatible with both open-source and closed-source LLMs. Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM as a defense instance to concurrently protect the target LLM instance in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs (both target and defense LLMs) have the capability to identify harmful prompts or intentions in user queries, which we empirically validate using the commonly used GPT-3.5/4 models across all major jailbreak attacks. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. These models outperform six state-of-the-art defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. We also empirically show that the tuned models are robust to adaptive jailbreaks and prompt injections.

摘要: 越狱是一种新兴的对抗性攻击，它绕过了现成的大型语言模型(LLM)中部署的安全对齐，并已演变为多种类别：基于人的、基于优化的、基于代的以及最近的间接和多语言越狱。然而，提供实际的越狱防御是具有挑战性的，因为它不仅需要处理所有上述越狱攻击，还需要对用户提示造成可以忽略不计的延迟，以及与开源和闭源LLM兼容。受传统影子堆栈安全概念防御内存溢出攻击的启发，提出了一种通用的LLM越狱防御框架--SelfDefend。该框架建立一个影子LLM作为防御实例，同时保护正常堆栈中的目标LLM实例，并与其协作进行基于检查点的访问控制。SelfDefend的有效性建立在我们的观察基础上，即现有的LLM(目标和防御LLM)能够识别用户查询中的有害提示或意图，我们使用所有主要越狱攻击中常用的GPT-3.5/4模型进行了经验验证。为了进一步提高防御的健壮性并将成本降至最低，我们使用数据蒸馏方法来优化专用的开源防御模型。这些型号的性能超过了六种最先进的防御系统，并与基于GPT-4的SelfDefend的性能相当，额外延迟显著降低。我们的经验还表明，调整后的模型对自适应越狱和快速注入具有较强的鲁棒性。



## **2. How to Train your Antivirus: RL-based Hardening through the Problem-Space**

如何训练防病毒：基于RL的问题空间强化 cs.CR

20 pages,4 figures

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2402.19027v2) [paper-pdf](http://arxiv.org/pdf/2402.19027v2)

**Authors**: Ilias Tsingenopoulos, Jacopo Cortellazzi, Branislav Bošanský, Simone Aonzo, Davy Preuveneers, Wouter Joosen, Fabio Pierazzi, Lorenzo Cavallaro

**Abstract**: ML-based malware detection on dynamic analysis reports is vulnerable to both evasion and spurious correlations. In this work, we investigate a specific ML architecture employed in the pipeline of a widely-known commercial antivirus company, with the goal to harden it against adversarial malware. Adversarial training, the sole defensive technique that can confer empirical robustness, is not applicable out of the box in this domain, for the principal reason that gradient-based perturbations rarely map back to feasible problem-space programs. We introduce a novel Reinforcement Learning approach for constructing adversarial examples, a constituent part of adversarially training a model against evasion. Our approach comes with multiple advantages. It performs modifications that are feasible in the problem-space, and only those; thus it circumvents the inverse mapping problem. It also makes possible to provide theoretical guarantees on the robustness of the model against a particular set of adversarial capabilities. Our empirical exploration validates our theoretical insights, where we can consistently reach 0% Attack Success Rate after a few adversarial retraining iterations.

摘要: 动态分析报告上基于ML的恶意软件检测容易受到规避和虚假关联的攻击。在这项工作中，我们调查了一个特定的ML体系结构，该体系结构应用于一家著名的商业反病毒公司的流水线中，目的是加强它对敌对恶意软件的攻击。对抗性训练是唯一可以赋予经验稳健性的防御技术，在这个领域不适用，主要原因是基于梯度的扰动很少映射回可行的问题空间程序。我们介绍了一种新的强化学习方法来构建对抗性例子，这是对抗性训练模型的一个组成部分。我们的方法具有多方面的优势。它执行在问题空间中可行的修改，并且只执行那些修改；因此，它绕过了逆映射问题。它还可以在理论上保证模型在对抗一组特定对手能力时的健壮性。我们的经验探索验证了我们的理论见解，在几次对抗性的再训练迭代后，我们可以始终达到0%的攻击成功率。



## **3. Evaluations of Machine Learning Privacy Defenses are Misleading**

对机器学习隐私辩护的评估具有误导性 cs.CR

Accepted at ACM CCS 2024

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2404.17399v2) [paper-pdf](http://arxiv.org/pdf/2404.17399v2)

**Authors**: Michael Aerni, Jie Zhang, Florian Tramèr

**Abstract**: Empirical defenses for machine learning privacy forgo the provable guarantees of differential privacy in the hope of achieving higher utility while resisting realistic adversaries. We identify severe pitfalls in existing empirical privacy evaluations (based on membership inference attacks) that result in misleading conclusions. In particular, we show that prior evaluations fail to characterize the privacy leakage of the most vulnerable samples, use weak attacks, and avoid comparisons with practical differential privacy baselines. In 5 case studies of empirical privacy defenses, we find that prior evaluations underestimate privacy leakage by an order of magnitude. Under our stronger evaluation, none of the empirical defenses we study are competitive with a properly tuned, high-utility DP-SGD baseline (with vacuous provable guarantees).

摘要: 机器学习隐私的经验防御放弃了差异隐私的可证明保证，希望在抵抗现实对手的同时实现更高的效用。我们发现了现有的经验隐私评估（基于成员资格推断攻击）中的严重陷阱，这些陷阱会导致误导性结论。特别是，我们表明，先前的评估未能描述最脆弱样本的隐私泄露，使用弱攻击，并避免与实际的差异隐私基线进行比较。在经验隐私防御的5个案例研究中，我们发现之前的评估低估了隐私泄露一个数量级。在我们更强有力的评估下，我们研究的经验防御都无法与适当调整的、高效用的DP-SGD基线（具有空洞的可证明保证）竞争。



## **4. Limited but consistent gains in adversarial robustness by co-training object recognition models with human EEG**

通过与人类脑电共同训练对象识别模型，在对抗鲁棒性方面获得有限但一致的收益 cs.LG

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.03646v1) [paper-pdf](http://arxiv.org/pdf/2409.03646v1)

**Authors**: Manshan Guo, Bhavin Choksi, Sari Sadiya, Alessandro T. Gifford, Martina G. Vilas, Radoslaw M. Cichy, Gemma Roig

**Abstract**: In contrast to human vision, artificial neural networks (ANNs) remain relatively susceptible to adversarial attacks. To address this vulnerability, efforts have been made to transfer inductive bias from human brains to ANNs, often by training the ANN representations to match their biological counterparts. Previous works relied on brain data acquired in rodents or primates using invasive techniques, from specific regions of the brain, under non-natural conditions (anesthetized animals), and with stimulus datasets lacking diversity and naturalness. In this work, we explored whether aligning model representations to human EEG responses to a rich set of real-world images increases robustness to ANNs. Specifically, we trained ResNet50-backbone models on a dual task of classification and EEG prediction; and evaluated their EEG prediction accuracy and robustness to adversarial attacks. We observed significant correlation between the networks' EEG prediction accuracy, often highest around 100 ms post stimulus onset, and their gains in adversarial robustness. Although effect size was limited, effects were consistent across different random initializations and robust for architectural variants. We further teased apart the data from individual EEG channels and observed strongest contribution from electrodes in the parieto-occipital regions. The demonstrated utility of human EEG for such tasks opens up avenues for future efforts that scale to larger datasets under diverse stimuli conditions with the promise of stronger effects.

摘要: 与人类视觉相比，人工神经网络(ANN)仍然相对容易受到对手攻击。为了解决这一弱点，人们努力将感应偏差从人脑转移到神经网络，通常是通过训练神经网络表示与其生物对应的表示相匹配。以前的工作依赖于使用侵入性技术从啮齿动物或灵长类动物身上获得的大脑数据，这些数据来自大脑的特定区域，在非自然条件下(麻醉的动物)，并且刺激数据集缺乏多样性和自然性。在这项工作中，我们探索了将模型表示与丰富的真实世界图像集的人类脑电响应对齐是否提高了对神经网络的鲁棒性。具体地说，我们在分类和脑电预测的双重任务上训练了ResNet50-Backbone模型，并评估了它们的脑电预测精度和对对手攻击的稳健性。我们观察到，这些网络的脑电预测准确性(通常在刺激开始后100毫秒左右达到最高)与它们在对手健壮性方面的收益之间存在显著的相关性。尽管效果大小有限，但在不同的随机初始化中，效果是一致的，并且对于体系结构变体来说是健壮的。我们进一步梳理了来自单个脑电通道的数据，并观察到顶枕区电极的最大贡献。人类脑电在这类任务中被证明的有效性为未来的努力开辟了道路，这些努力将在不同的刺激条件下扩大到更大的数据集，并有望产生更强的影响。



## **5. A practical approach to evaluating the adversarial distance for machine learning classifiers**

评估机器学习分类器对抗距离的实用方法 cs.LG

Accepted manuscript at International Mechanical Engineering Congress  and Exposition IMECE2024

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.03598v1) [paper-pdf](http://arxiv.org/pdf/2409.03598v1)

**Authors**: Georg Siedel, Ekagra Gupta, Andrey Morozov

**Abstract**: Robustness is critical for machine learning (ML) classifiers to ensure consistent performance in real-world applications where models may encounter corrupted or adversarial inputs. In particular, assessing the robustness of classifiers to adversarial inputs is essential to protect systems from vulnerabilities and thus ensure safety in use. However, methods to accurately compute adversarial robustness have been challenging for complex ML models and high-dimensional data. Furthermore, evaluations typically measure adversarial accuracy on specific attack budgets, limiting the informative value of the resulting metrics. This paper investigates the estimation of the more informative adversarial distance using iterative adversarial attacks and a certification approach. Combined, the methods provide a comprehensive evaluation of adversarial robustness by computing estimates for the upper and lower bounds of the adversarial distance. We present visualisations and ablation studies that provide insights into how this evaluation method should be applied and parameterised. We find that our adversarial attack approach is effective compared to related implementations, while the certification method falls short of expectations. The approach in this paper should encourage a more informative way of evaluating the adversarial robustness of ML classifiers.

摘要: 稳健性对于机器学习(ML)分类器至关重要，以确保在模型可能遇到损坏或敌对输入的真实世界应用中的一致性能。特别是，评估分类器对敌意输入的稳健性对于保护系统免受漏洞的攻击从而确保安全使用至关重要。然而，对于复杂的ML模型和高维数据，精确计算对手稳健性的方法一直是具有挑战性的。此外，评估通常在特定攻击预算上衡量对手的准确性，限制了结果指标的信息性价值。本文研究了利用迭代对抗性攻击和认证方法来估计更多信息的对抗性距离。结合起来，这些方法通过计算对手距离的上界和下界的估计来提供对对手稳健性的综合评估。我们介绍了可视化和消融研究，为如何应用和参数化这种评估方法提供了洞察力。我们发现，与相关实现相比，我们的对抗性攻击方法是有效的，而认证方法没有达到预期。本文中的方法应该鼓励以一种更有信息量的方式来评估ML分类器的对抗健壮性。



## **6. Unleashing the potential of prompt engineering in Large Language Models: a comprehensive review**

释放大型语言模型中提示工程的潜力：全面回顾 cs.CL

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2310.14735v5) [paper-pdf](http://arxiv.org/pdf/2310.14735v5)

**Authors**: Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu

**Abstract**: This comprehensive review delves into the pivotal role of prompt engineering in unleashing the capabilities of Large Language Models (LLMs). The development of Artificial Intelligence (AI), from its inception in the 1950s to the emergence of advanced neural networks and deep learning architectures, has made a breakthrough in LLMs, with models such as GPT-4o and Claude-3, and in Vision-Language Models (VLMs), with models such as CLIP and ALIGN. Prompt engineering is the process of structuring inputs, which has emerged as a crucial technique to maximize the utility and accuracy of these models. This paper explores both foundational and advanced methodologies of prompt engineering, including techniques such as self-consistency, chain-of-thought, and generated knowledge, which significantly enhance model performance. Additionally, it examines the prompt method of VLMs through innovative approaches such as Context Optimization (CoOp), Conditional Context Optimization (CoCoOp), and Multimodal Prompt Learning (MaPLe). Critical to this discussion is the aspect of AI security, particularly adversarial attacks that exploit vulnerabilities in prompt engineering. Strategies to mitigate these risks and enhance model robustness are thoroughly reviewed. The evaluation of prompt methods is also addressed, through both subjective and objective metrics, ensuring a robust analysis of their efficacy. This review also reflects the essential role of prompt engineering in advancing AI capabilities, providing a structured framework for future research and application.

摘要: 这篇全面的综述深入探讨了快速工程在释放大型语言模型(LLM)能力方面的关键作用。人工智能(AI)的发展，从20世纪50年代开始，到先进的神经网络和深度学习体系的出现，在LLMS方面取得了突破，有GPT-40和Claude-3等模型，以及视觉语言模型(VLMS)，有CLIP和ALIGN等模型。即时工程是对输入进行结构化的过程，它已成为最大化这些模型的实用性和准确性的关键技术。本文探讨了即时工程的基本方法和高级方法，包括自我一致性、思想链和生成知识等技术，这些技术显著提高了模型的性能。此外，它还通过诸如语境优化(COOP)、条件语境优化(CoCoOp)和多通道提示学习(Maple)等创新方法来研究虚拟学习模型的提示方法。对这一讨论至关重要的是人工智能安全方面，特别是利用即时工程中的漏洞进行的对抗性攻击。对缓解这些风险和增强模型稳健性的策略进行了彻底的回顾。还通过主观和客观指标对快速方法进行了评估，以确保对其有效性进行稳健的分析。这篇综述还反映了快速工程在推进人工智能能力方面的重要作用，为未来的研究和应用提供了一个结构化的框架。



## **7. TSFool: Crafting Highly-Imperceptible Adversarial Time Series through Multi-Objective Attack**

TSFool：通过多目标攻击打造高度不可感知的对抗时间序列 cs.LG

27th European Conference on Artificial Intelligence (ECAI'24)

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2209.06388v4) [paper-pdf](http://arxiv.org/pdf/2209.06388v4)

**Authors**: Yanyun Wang, Dehui Du, Haibo Hu, Zi Liang, Yuanhao Liu

**Abstract**: Recent years have witnessed the success of recurrent neural network (RNN) models in time series classification (TSC). However, neural networks (NNs) are vulnerable to adversarial samples, which cause real-life adversarial attacks that undermine the robustness of AI models. To date, most existing attacks target at feed-forward NNs and image recognition tasks, but they cannot perform well on RNN-based TSC. This is due to the cyclical computation of RNN, which prevents direct model differentiation. In addition, the high visual sensitivity of time series to perturbations also poses challenges to local objective optimization of adversarial samples. In this paper, we propose an efficient method called TSFool to craft highly-imperceptible adversarial time series for RNN-based TSC. The core idea is a new global optimization objective known as "Camouflage Coefficient" that captures the imperceptibility of adversarial samples from the class distribution. Based on this, we reduce the adversarial attack problem to a multi-objective optimization problem that enhances the perturbation quality. Furthermore, to speed up the optimization process, we propose to use a representation model for RNN to capture deeply embedded vulnerable samples whose features deviate from the latent manifold. Experiments on 11 UCR and UEA datasets showcase that TSFool significantly outperforms six white-box and three black-box benchmark attacks in terms of effectiveness, efficiency and imperceptibility from various perspectives including standard measure, human study and real-world defense.

摘要: 近年来，递归神经网络(RNN)模型在时间序列分类(TSC)中取得了成功。然而，神经网络很容易受到敌意样本的攻击，这些样本会导致现实生活中的对抗性攻击，从而削弱人工智能模型的健壮性。到目前为止，大多数现有的攻击都是针对前馈神经网络和图像识别任务，但它们在基于RNN的TSC上不能很好地执行。这是由于RNN的循环计算，这阻止了直接的模型区分。此外，时间序列对扰动的高度视觉敏感性也给对抗性样本的局部目标优化带来了挑战。在本文中，我们提出了一种称为TSFool的有效方法来为基于RNN的TSC生成高度不可察觉的对抗性时间序列。其核心思想是一种新的全局优化目标，称为伪装系数，它从类别分布中捕捉敌方样本的不可见性。在此基础上，我们将对抗性攻击问题归结为一个提高扰动质量的多目标优化问题。此外，为了加快优化过程，我们提出了使用RNN的表示模型来捕获深度嵌入的特征偏离潜在流形的易受攻击的样本。在11个UCR和UEA数据集上的实验表明，TSFool在有效性、效率和不可感知性方面都明显优于6个白盒和3个黑盒基准攻击，包括标准度量、人类研究和现实世界防御。



## **8. Boosting Adversarial Transferability for Skeleton-based Action Recognition via Exploring the Model Posterior Space**

通过探索模型后验空间增强基于线粒体的动作识别的对抗可移植性 cs.CV

We have submitted a new version of our work at arXiv:2409.02483. This  version, arXiv:2407.08572, is no longer valid. Any update for this work will  be conducted in arXiv:2409.02483

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2407.08572v2) [paper-pdf](http://arxiv.org/pdf/2407.08572v2)

**Authors**: Yunfeng Diao, Baiqi Wu, Ruixuan Zhang, Xun Yang, Meng Wang, He Wang

**Abstract**: Skeletal motion plays a pivotal role in human activity recognition (HAR). Recently, attack methods have been proposed to identify the universal vulnerability of skeleton-based HAR(S-HAR). However, the research of adversarial transferability on S-HAR is largely missing. More importantly, existing attacks all struggle in transfer across unknown S-HAR models. We observed that the key reason is that the loss landscape of the action recognizers is rugged and sharp. Given the established correlation in prior studies~\cite{qin2022boosting,wu2020towards} between loss landscape and adversarial transferability, we assume and empirically validate that smoothing the loss landscape could potentially improve adversarial transferability on S-HAR. This is achieved by proposing a new post-train Dual Bayesian strategy, which can effectively explore the model posterior space for a collection of surrogates without the need for re-training. Furthermore, to craft adversarial examples along the motion manifold, we incorporate the attack gradient with information of the motion dynamics in a Bayesian manner. Evaluated on benchmark datasets, e.g. HDM05 and NTU 60, the average transfer success rate can reach as high as 35.9\% and 45.5\% respectively. In comparison, current state-of-the-art skeletal attacks achieve only 3.6\% and 9.8\%. The high adversarial transferability remains consistent across various surrogate, victim, and even defense models. Through a comprehensive analysis of the results, we provide insights on what surrogates are more likely to exhibit transferability, to shed light on future research.

摘要: 骨骼运动在人类活动识别(HAR)中起着至关重要的作用。近年来，为了识别基于骨架的HAR(S-HAR)的普遍脆弱性，人们提出了攻击方法。然而，关于S-哈尔对抗性转会的研究还很少。更重要的是，现有的进攻都在挣扎着跨越未知的S-哈尔模型进行转移。我们观察到，关键原因是动作识别器的损失情况是崎岖和尖锐的。鉴于先前的研究已经建立了损失情景与对手可转移性之间的相关性，我们假设并实证平滑损失情景可以潜在地提高S-HAR上的对手可转移性。这是通过提出一种新的训练后双重贝叶斯策略来实现的，该策略可以有效地探索代理集合的模型后验空间，而不需要重新训练。此外，为了制作沿运动流形的对抗性示例，我们以贝叶斯方式将攻击梯度与运动动力学信息相结合。在HDM05和NTU 60等基准数据集上进行评估，平均传输成功率分别高达35.9%和45.5%。相比之下，目前最先进的骨架攻击只实现了3.6%和9.8%。高度的对抗性可转移性在各种代理、受害者甚至防御模型中保持一致。通过对结果的综合分析，我们对哪些替代品更有可能表现出可转移性提供了见解，为未来的研究提供了启示。



## **9. LLM Detectors Still Fall Short of Real World: Case of LLM-Generated Short News-Like Posts**

LLM探测器仍然达不到现实世界：LLM生成的短新闻类帖子的案例 cs.CL

20 pages, 7 tables, 13 figures, under consideration for EMNLP

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.03291v1) [paper-pdf](http://arxiv.org/pdf/2409.03291v1)

**Authors**: Henrique Da Silva Gameiro, Andrei Kucharavy, Ljiljana Dolamic

**Abstract**: With the emergence of widely available powerful LLMs, disinformation generated by large Language Models (LLMs) has become a major concern. Historically, LLM detectors have been touted as a solution, but their effectiveness in the real world is still to be proven. In this paper, we focus on an important setting in information operations -- short news-like posts generated by moderately sophisticated attackers.   We demonstrate that existing LLM detectors, whether zero-shot or purpose-trained, are not ready for real-world use in that setting. All tested zero-shot detectors perform inconsistently with prior benchmarks and are highly vulnerable to sampling temperature increase, a trivial attack absent from recent benchmarks. A purpose-trained detector generalizing across LLMs and unseen attacks can be developed, but it fails to generalize to new human-written texts.   We argue that the former indicates domain-specific benchmarking is needed, while the latter suggests a trade-off between the adversarial evasion resilience and overfitting to the reference human text, with both needing evaluation in benchmarks and currently absent. We believe this suggests a re-consideration of current LLM detector benchmarking approaches and provides a dynamically extensible benchmark to allow it (https://github.com/Reliable-Information-Lab-HEVS/dynamic_llm_detector_benchmark).

摘要: 随着广泛使用的强大的LLM的出现，大型语言模型(LLM)产生的虚假信息已经成为一个主要关注的问题。从历史上看，LLM探测器一直被吹捧为一种解决方案，但它们在现实世界中的有效性仍有待证明。在这篇文章中，我们关注信息操作中的一个重要环境--由中等经验丰富的攻击者生成的类似新闻的短帖子。我们证明，现有的LLM探测器，无论是零炮还是专门训练的，都没有准备好在那种情况下用于现实世界。所有经过测试的零射击探测器的性能都与以前的基准不一致，并且非常容易受到采样温度升高的影响，这是最近的基准中所没有的一种轻微攻击。可以开发出一种专门训练的检测器，可以在LLMS和不可见攻击中推广，但它无法推广到新的人类书写的文本。我们认为，前者表明需要特定领域的基准测试，而后者则建议在对抗性回避韧性和对参考人类文本的过度匹配之间进行权衡，两者都需要在基准中进行评估，目前还没有。我们认为，这表明了对当前LLm探测器基准方法的重新考虑，并提供了一个动态可扩展的基准，以允许其(https://github.com/Reliable-Information-Lab-HEVS/dynamic_llm_detector_benchmark).



## **10. OpenFact at CheckThat! 2024: Combining Multiple Attack Methods for Effective Adversarial Text Generation**

CheckThat上的OpenFact！2024年：结合多种攻击方法以有效生成对抗性文本 cs.CL

CLEF 2024 - Conference and Labs of the Evaluation Forum

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.02649v2) [paper-pdf](http://arxiv.org/pdf/2409.02649v2)

**Authors**: Włodzimierz Lewoniewski, Piotr Stolarski, Milena Stróżyna, Elzbieta Lewańska, Aleksandra Wojewoda, Ewelina Księżniak, Marcin Sawiński

**Abstract**: This paper presents the experiments and results for the CheckThat! Lab at CLEF 2024 Task 6: Robustness of Credibility Assessment with Adversarial Examples (InCrediblAE). The primary objective of this task was to generate adversarial examples in five problem domains in order to evaluate the robustness of widely used text classification methods (fine-tuned BERT, BiLSTM, and RoBERTa) when applied to credibility assessment issues.   This study explores the application of ensemble learning to enhance adversarial attacks on natural language processing (NLP) models. We systematically tested and refined several adversarial attack methods, including BERT-Attack, Genetic algorithms, TextFooler, and CLARE, on five datasets across various misinformation tasks. By developing modified versions of BERT-Attack and hybrid methods, we achieved significant improvements in attack effectiveness. Our results demonstrate the potential of modification and combining multiple methods to create more sophisticated and effective adversarial attack strategies, contributing to the development of more robust and secure systems.

摘要: 本文给出了CheckThat！CLEF 2024实验室任务6：使用对抗性实例进行可信度评估的稳健性(InCrediblae)。这项任务的主要目标是在五个问题领域生成对抗性实例，以评估广泛使用的文本分类方法(微调的BERT、BiLSTM和Roberta)在应用于可信度评估问题时的稳健性。本研究探讨了集成学习在增强对自然语言处理(NLP)模型的敌意攻击中的应用。我们在五个数据集上系统地测试和提炼了几种对抗性攻击方法，包括BERT攻击、遗传算法、TextFooler和Clare，这些方法跨越各种错误信息任务。通过开发改进版本的BERT-攻击和混合方法，我们在攻击效率方面取得了显著的改进。我们的结果证明了修改和结合多种方法来创建更复杂和有效的对抗性攻击策略的潜力，有助于开发更健壮和安全的系统。



## **11. AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization**

AICAttack：基于注意力的优化的对抗性图像字幕攻击 cs.CV

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2402.11940v3) [paper-pdf](http://arxiv.org/pdf/2402.11940v3)

**Authors**: Jiyao Li, Mingze Ni, Yifei Dong, Tianqing Zhu, Wei Liu

**Abstract**: Recent advances in deep learning research have shown remarkable achievements across many tasks in computer vision (CV) and natural language processing (NLP). At the intersection of CV and NLP is the problem of image captioning, where the related models' robustness against adversarial attacks has not been well studied. This paper presents a novel adversarial attack strategy, AICAttack (Attention-based Image Captioning Attack), designed to attack image captioning models through subtle perturbations on images. Operating within a black-box attack scenario, our algorithm requires no access to the target model's architecture, parameters, or gradient information. We introduce an attention-based candidate selection mechanism that identifies the optimal pixels to attack, followed by a customised differential evolution method to optimise the perturbations of pixels' RGB values. We demonstrate AICAttack's effectiveness through extensive experiments on benchmark datasets against multiple victim models. The experimental results demonstrate that our method outperforms current leading-edge techniques by achieving consistently higher attack success rates.

摘要: 近年来，深度学习研究在计算机视觉(CV)和自然语言处理(NLP)领域取得了令人瞩目的成就。在CV和NLP的交叉点是图像字幕问题，相关模型对敌意攻击的稳健性还没有得到很好的研究。提出了一种新的对抗性攻击策略AICAttack(基于注意力的图像字幕攻击)，旨在通过对图像进行微妙的扰动来攻击图像字幕模型。我们的算法在黑盒攻击场景中运行，不需要访问目标模型的体系结构、参数或梯度信息。我们引入了一种基于注意力的候选选择机制来确定要攻击的最佳像素，然后使用定制的差分进化方法来优化像素RGB值的扰动。通过在基准数据集上针对多个受害者模型的大量实验，我们展示了AICAttack的有效性。实验结果表明，我们的方法比目前的前沿技术具有更高的攻击成功率。



## **12. Robust Q-Learning under Corrupted Rewards**

奖励受损下的稳健Q学习 cs.LG

Accepted to the Decision and Control Conference (CDC) 2024

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.03237v1) [paper-pdf](http://arxiv.org/pdf/2409.03237v1)

**Authors**: Sreejeet Maity, Aritra Mitra

**Abstract**: Recently, there has been a surge of interest in analyzing the non-asymptotic behavior of model-free reinforcement learning algorithms. However, the performance of such algorithms in non-ideal environments, such as in the presence of corrupted rewards, is poorly understood. Motivated by this gap, we investigate the robustness of the celebrated Q-learning algorithm to a strong-contamination attack model, where an adversary can arbitrarily perturb a small fraction of the observed rewards. We start by proving that such an attack can cause the vanilla Q-learning algorithm to incur arbitrarily large errors. We then develop a novel robust synchronous Q-learning algorithm that uses historical reward data to construct robust empirical Bellman operators at each time step. Finally, we prove a finite-time convergence rate for our algorithm that matches known state-of-the-art bounds (in the absence of attacks) up to a small inevitable $O(\varepsilon)$ error term that scales with the adversarial corruption fraction $\varepsilon$. Notably, our results continue to hold even when the true reward distributions have infinite support, provided they admit bounded second moments.

摘要: 近年来，分析无模型强化学习算法的非渐近行为引起了人们的极大兴趣。然而，这类算法在非理想环境中的性能，例如在存在腐败奖励的情况下，人们了解得很少。受这一差距的启发，我们研究了著名的Q学习算法对强污染攻击模型的稳健性，在该模型中，对手可以任意扰动一小部分观察到的奖励。我们首先证明了这样的攻击可以导致普通Q学习算法产生任意大的误差。然后，我们开发了一种新的稳健同步Q-学习算法，该算法使用历史奖励数据来构造每个时间步的稳健经验Bellman算子。最后，我们证明了我们的算法的有限时间收敛速度，它匹配已知的最新界(在没有攻击的情况下)直到一个小的不可避免的$O(\varepsilon)$错误项，该错误项随敌对的腐败分数$\varepsilon$而扩展。值得注意的是，我们的结果仍然适用，即使当真正的报酬分布有无限支持时，只要它们允许有界的二次矩。



## **13. Transfer-based Adversarial Poisoning Attacks for Online (MIMO-)Deep Receviers**

基于传输的针对在线（MMO-）深度回收者的对抗中毒攻击 eess.SP

15 pages, 14 figures

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.02430v2) [paper-pdf](http://arxiv.org/pdf/2409.02430v2)

**Authors**: Kunze Wu, Weiheng Jiang, Dusit Niyato, Yinghuan Li, Chuang Luo

**Abstract**: Recently, the design of wireless receivers using deep neural networks (DNNs), known as deep receivers, has attracted extensive attention for ensuring reliable communication in complex channel environments. To adapt quickly to dynamic channels, online learning has been adopted to update the weights of deep receivers with over-the-air data (e.g., pilots). However, the fragility of neural models and the openness of wireless channels expose these systems to malicious attacks. To this end, understanding these attack methods is essential for robust receiver design. In this paper, we propose a transfer-based adversarial poisoning attack method for online receivers.Without knowledge of the attack target, adversarial perturbations are injected to the pilots, poisoning the online deep receiver and impairing its ability to adapt to dynamic channels and nonlinear effects. In particular, our attack method targets Deep Soft Interference Cancellation (DeepSIC)[1] using online meta-learning. As a classical model-driven deep receiver, DeepSIC incorporates wireless domain knowledge into its architecture. This integration allows it to adapt efficiently to time-varying channels with only a small number of pilots, achieving optimal performance in a multi-input and multi-output (MIMO) scenario.The deep receiver in this scenario has a number of applications in the field of wireless communication, which motivates our study of the attack methods targeting it.Specifically, we demonstrate the effectiveness of our attack in simulations on synthetic linear, synthetic nonlinear, static, and COST 2100 channels. Simulation results indicate that the proposed poisoning attack significantly reduces the performance of online receivers in rapidly changing scenarios.

摘要: 近年来，利用深度神经网络(DNN)设计无线接收器，即深度接收器，因其能在复杂的信道环境中保证可靠的通信而受到广泛关注。为了快速适应动态信道，已采用在线学习来使用空中数据(例如，导频)来更新深度接收器的权重。然而，神经模型的脆弱性和无线通道的开放性使这些系统面临恶意攻击。为此，了解这些攻击方法对于稳健的接收器设计至关重要。提出了一种基于传输的在线接收器对抗性中毒攻击方法，在不知道攻击目标的情况下，向飞行员注入对抗性扰动，毒害在线深层接收器，削弱其对动态信道和非线性效应的适应能力。特别是，我们的攻击方法针对使用在线元学习的深度软干扰消除(DeepSIC)[1]。作为一种经典的模型驱动的深度接收器，DeepSIC将无线领域的知识融入到其体系结构中。在多输入多输出(MIMO)环境下，这种集成使得它能够有效地适应时变信道，在多输入多输出(MIMO)场景中获得最优的性能。这种场景下的深度接收器在无线通信领域有着广泛的应用，这促使我们研究针对它的攻击方法。具体地，我们在合成线性、合成非线性、静态和代价2100个信道上仿真验证了我们的攻击的有效性。仿真结果表明，所提出的中毒攻击显著降低了在线接收者在快速变化的场景中的性能。



## **14. Bypassing DARCY Defense: Indistinguishable Universal Adversarial Triggers**

扰乱DARCY防御：难以区分的普遍对抗触发 cs.CL

13 pages, 5 figures

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.03183v1) [paper-pdf](http://arxiv.org/pdf/2409.03183v1)

**Authors**: Zuquan Peng, Yuanyuan He, Jianbing Ni, Ben Niu

**Abstract**: Neural networks (NN) classification models for Natural Language Processing (NLP) are vulnerable to the Universal Adversarial Triggers (UAT) attack that triggers a model to produce a specific prediction for any input. DARCY borrows the "honeypot" concept to bait multiple trapdoors, effectively detecting the adversarial examples generated by UAT. Unfortunately, we find a new UAT generation method, called IndisUAT, which produces triggers (i.e., tokens) and uses them to craft adversarial examples whose feature distribution is indistinguishable from that of the benign examples in a randomly-chosen category at the detection layer of DARCY. The produced adversarial examples incur the maximal loss of predicting results in the DARCY-protected models. Meanwhile, the produced triggers are effective in black-box models for text generation, text inference, and reading comprehension. Finally, the evaluation results under NN models for NLP tasks indicate that the IndisUAT method can effectively circumvent DARCY and penetrate other defenses. For example, IndisUAT can reduce the true positive rate of DARCY's detection by at least 40.8% and 90.6%, and drop the accuracy by at least 33.3% and 51.6% in the RNN and CNN models, respectively. IndisUAT reduces the accuracy of the BERT's adversarial defense model by at least 34.0%, and makes the GPT-2 language model spew racist outputs even when conditioned on non-racial context.

摘要: 自然语言处理(NLP)的神经网络(NN)分类模型容易受到通用对抗触发器(UAT)的攻击，UAT会触发模型对任何输入产生特定的预测。Darcy借用了“蜜罐”的概念来引诱多个陷门，有效地检测到UAT生成的敌意例子。不幸的是，我们发现了一种新的UAT生成方法，称为IndisUAT，它生成触发器(即令牌)，并使用它们来创建敌意示例，其特征分布与Darcy检测层随机选择的类别中的良性示例的特征分布无法区分。在Darcy保护模型中，产生的对抗性例子导致预测结果的最大损失。同时，所产生的触发语在黑盒模型中对文本生成、文本推理和阅读理解都是有效的。最后，在神经网络模型下对NLP任务的评估结果表明，IndisUAT方法可以有效地绕过Darcy并穿透其他防御。例如，在RNN和CNN模型中，IndisUAT可以使Darcy检测的真阳性率至少下降40.8%和90.6%，准确率至少下降33.3%和51.6%。INdisUAT将BERT的对抗性防御模型的准确性降低了至少34.0%，并使GPT-2语言模型即使在非种族背景下也会产生种族主义输出。



## **15. ACCESS-FL: Agile Communication and Computation for Efficient Secure Aggregation in Stable Federated Learning Networks**

ACCESS-FL：在稳定的联邦学习网络中实现高效安全聚合的敏捷通信和计算 cs.CR

**SubmitDate**: 2024-09-05    [abs](http://arxiv.org/abs/2409.01722v2) [paper-pdf](http://arxiv.org/pdf/2409.01722v2)

**Authors**: Niousha Nazemi, Omid Tavallaie, Shuaijun Chen, Anna Maria Mandalari, Kanchana Thilakarathna, Ralph Holz, Hamed Haddadi, Albert Y. Zomaya

**Abstract**: Federated Learning (FL) is a promising distributed learning framework designed for privacy-aware applications. FL trains models on client devices without sharing the client's data and generates a global model on a server by aggregating model updates. Traditional FL approaches risk exposing sensitive client data when plain model updates are transmitted to the server, making them vulnerable to security threats such as model inversion attacks where the server can infer the client's original training data from monitoring the changes of the trained model in different rounds. Google's Secure Aggregation (SecAgg) protocol addresses this threat by employing a double-masking technique, secret sharing, and cryptography computations in honest-but-curious and adversarial scenarios with client dropouts. However, in scenarios without the presence of an active adversary, the computational and communication cost of SecAgg significantly increases by growing the number of clients. To address this issue, in this paper, we propose ACCESS-FL, a communication-and-computation-efficient secure aggregation method designed for honest-but-curious scenarios in stable FL networks with a limited rate of client dropout. ACCESS-FL reduces the computation/communication cost to a constant level (independent of the network size) by generating shared secrets between only two clients and eliminating the need for double masking, secret sharing, and cryptography computations. To evaluate the performance of ACCESS-FL, we conduct experiments using the MNIST, FMNIST, and CIFAR datasets to verify the performance of our proposed method. The evaluation results demonstrate that our proposed method significantly reduces computation and communication overhead compared to state-of-the-art methods, SecAgg and SecAgg+.

摘要: 联合学习(FL)是一种很有前途的分布式学习框架，专为隐私感知应用而设计。FL在不共享客户端数据的情况下在客户端设备上训练模型，并通过聚合模型更新在服务器上生成全局模型。当普通模型更新被传输到服务器时，传统的FL方法可能会暴露敏感的客户端数据，使它们容易受到安全威胁，例如模型反转攻击，其中服务器可以通过监控不同轮训练模型的变化来推断客户端的原始训练数据。谷歌的安全聚合(SecAgg)协议通过使用双掩码技术、秘密共享和密码计算来解决这一威胁，这些计算是在诚实但好奇的和恶意的客户端退出场景中进行的。然而，在没有活跃对手的情况下，SecAgg的计算和通信成本会随着客户端数量的增加而显著增加。为了解决这一问题，本文提出了Access-FL，一种通信和计算高效的安全聚合方法，专为稳定的FL网络中诚实但好奇的场景而设计，并且客户端丢失率有限。Access-FL通过仅在两个客户端之间生成共享秘密并消除双重掩码、秘密共享和密码计算，将计算/通信成本降低到恒定水平(与网络大小无关)。为了评估Access-FL的性能，我们使用MNIST、FMNIST和CIFAR数据集进行了实验，以验证我们所提出的方法的性能。评估结果表明，与最先进的SecAgg和SecAgg+方法相比，我们提出的方法显著减少了计算和通信开销。



## **16. Well, that escalated quickly: The Single-Turn Crescendo Attack (STCA)**

嗯，情况迅速升级：单转渐强攻击（STCA） cs.CR

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.03131v1) [paper-pdf](http://arxiv.org/pdf/2409.03131v1)

**Authors**: Alan Aqrawi

**Abstract**: This paper explores a novel approach to adversarial attacks on large language models (LLM): the Single-Turn Crescendo Attack (STCA). The STCA builds upon the multi-turn crescendo attack established by Mark Russinovich, Ahmed Salem, Ronen Eldan. Traditional multi-turn adversarial strategies gradually escalate the context to elicit harmful or controversial responses from LLMs. However, this paper introduces a more efficient method where the escalation is condensed into a single interaction. By carefully crafting the prompt to simulate an extended dialogue, the attack bypasses typical content moderation systems, leading to the generation of responses that would normally be filtered out. I demonstrate this technique through a few case studies. The results highlight vulnerabilities in current LLMs and underscore the need for more robust safeguards. This work contributes to the broader discourse on responsible AI (RAI) safety and adversarial testing, providing insights and practical examples for researchers and developers. This method is unexplored in the literature, making it a novel contribution to the field.

摘要: 针对大型语言模型(LLM)提出了一种新的对抗性攻击方法：单轮渐近攻击(STCA)。STCA建立在Mark Russinovich，Ahmed Salem，Ronen Eldan建立的多轮渐强攻击的基础上。传统的多回合对抗性战略逐渐升级背景，以引起低收入国家的有害或有争议的反应。然而，本文介绍了一种更有效的方法，在该方法中，升级被压缩为单个交互。通过精心设计提示符来模拟延长的对话，攻击绕过了典型的内容审核系统，导致生成通常会被过滤掉的响应。我通过几个案例研究来演示这项技术。这一结果突显了当前LLM的脆弱性，并突显了需要更强有力的保障措施。这项工作有助于更广泛地讨论负责任的人工智能(RAI)安全和对抗性测试，为研究人员和开发人员提供见解和实践示例。这种方法在文献中还没有被探索过，这使它成为该领域的一个新贡献。



## **17. Knowledge Transfer for Collaborative Misbehavior Detection in Untrusted Vehicular Environments**

不可信车辆环境中协作不当行为检测的知识转移 cs.NI

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.02844v1) [paper-pdf](http://arxiv.org/pdf/2409.02844v1)

**Authors**: Roshan Sedar, Charalampos Kalalas, Paolo Dini, Francisco Vazquez-Gallego, Jesus Alonso-Zarate, Luis Alonso

**Abstract**: Vehicular mobility underscores the need for collaborative misbehavior detection at the vehicular edge. However, locally trained misbehavior detection models are susceptible to adversarial attacks that aim to deliberately influence learning outcomes. In this paper, we introduce a deep reinforcement learning-based approach that employs transfer learning for collaborative misbehavior detection among roadside units (RSUs). In the presence of label-flipping and policy induction attacks, we perform selective knowledge transfer from trustworthy source RSUs to foster relevant expertise in misbehavior detection and avoid negative knowledge sharing from adversary-influenced RSUs. The performance of our proposed scheme is demonstrated with evaluations over a diverse set of misbehavior detection scenarios using an open-source dataset. Experimental results show that our approach significantly reduces the training time at the target RSU and achieves superior detection performance compared to the baseline scheme with tabula rasa learning. Enhanced robustness and generalizability can also be attained, by effectively detecting previously unseen and partially observable misbehavior attacks.

摘要: 车辆机动性强调了在车辆边缘协作检测不当行为的必要性。然而，本地训练的不良行为检测模型很容易受到旨在故意影响学习结果的对抗性攻击。在本文中，我们介绍了一种基于深度强化学习的方法，该方法将转移学习用于路边单元(RSU)之间的协作行为检测。在存在标签翻转和策略诱导攻击的情况下，我们从可信来源的RSU进行选择性的知识转移，以培养相关的不当行为检测专业知识，并避免来自对手影响的RSU的负面知识共享。通过使用开源数据集对一组不同的不当行为检测场景进行评估，验证了我们所提出的方案的性能。实验结果表明，我们的方法大大减少了在目标RSU的训练时间，并获得了比基于TABULA RASA学习的基线方案更好的检测性能。通过有效地检测以前未见和部分可见的不良行为攻击，还可以获得增强的稳健性和泛化能力。



## **18. Revisiting Character-level Adversarial Attacks for Language Models**

重新审视语言模型的初级对抗攻击 cs.LG

Accepted in ICML 2024

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2405.04346v2) [paper-pdf](http://arxiv.org/pdf/2405.04346v2)

**Authors**: Elias Abad Rocamora, Yongtao Wu, Fanghui Liu, Grigorios G. Chrysos, Volkan Cevher

**Abstract**: Adversarial attacks in Natural Language Processing apply perturbations in the character or token levels. Token-level attacks, gaining prominence for their use of gradient-based methods, are susceptible to altering sentence semantics, leading to invalid adversarial examples. While character-level attacks easily maintain semantics, they have received less attention as they cannot easily adopt popular gradient-based methods, and are thought to be easy to defend. Challenging these beliefs, we introduce Charmer, an efficient query-based adversarial attack capable of achieving high attack success rate (ASR) while generating highly similar adversarial examples. Our method successfully targets both small (BERT) and large (Llama 2) models. Specifically, on BERT with SST-2, Charmer improves the ASR in 4.84% points and the USE similarity in 8% points with respect to the previous art. Our implementation is available in https://github.com/LIONS-EPFL/Charmer.

摘要: 自然语言处理中的对抗攻击对字符或令牌级别施加扰动。令牌级攻击因使用基于梯度的方法而变得越来越重要，很容易改变句子语义，从而导致无效的对抗性示例。虽然字符级攻击很容易维护语义，但它们受到的关注较少，因为它们不能轻易采用流行的基于梯度的方法，并且被认为很容易防御。基于这些信念，我们引入了Charmer，这是一种高效的基于查询的对抗性攻击，能够实现高攻击成功率（ASB），同时生成高度相似的对抗性示例。我们的方法成功地针对小型（BERT）和大型（Llama 2）模型。具体来说，在采用CST-2的BERT上，Charmer将ASB提高了4.84%，与之前的作品相比，USE相似性提高了8%。我们的实现可在https://github.com/LIONS-EPFL/Charmer上获取。



## **19. Simple fusion-fission quantifies Israel-Palestine violence and suggests multi-adversary solution**

简单的融合-裂变量化了以巴暴力并提出了多对手解决方案 physics.soc-ph

Comments welcome. Working paper

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.02816v1) [paper-pdf](http://arxiv.org/pdf/2409.02816v1)

**Authors**: Frank Yingjie Huo, Pedro D. Manrique, Dylan J. Restrepo, Gordon Woo, Neil F. Johnson

**Abstract**: Why humans fight has no easy answer. However, understanding better how humans fight could inform future interventions, hidden shifts and casualty risk. Fusion-fission describes the well-known grouping behavior of fish etc. fighting for survival in the face of strong opponents: they form clusters ('fusion') which provide collective benefits and a cluster scatters when it senses danger ('fission'). Here we show how similar clustering (fusion-fission) of human fighters provides a unified quantitative explanation for complex casualty patterns across decades of Israel-Palestine region violence, as well as the October 7 surprise attack -- and uncovers a hidden post-October 7 shift. State-of-the-art data shows this fighter fusion-fission in action. It also predicts future 'super-shock' attacks that will be more lethal than October 7 and will arrive earlier. It offers a multi-adversary solution. Our results -- which include testable formulae and a plug-and-play simulation -- enable concrete risk assessments of future casualties and policy-making grounded by fighter behavior.

摘要: 人类为什么会打架并不是一个简单的答案。然而，更好地了解人类是如何战斗的，可能会为未来的干预、隐藏的变化和伤亡风险提供信息。融合-裂变描述了众所周知的鱼类等在强大对手面前为生存而战的群体行为：它们形成集群(‘融合’)，提供集体利益，一个集群在感觉到危险(‘裂变’)时分散。在这里，我们展示了类似的人类战斗机集群(聚变-裂变)如何为数十年来以巴地区暴力以及10月7日突袭的复杂伤亡模式提供统一的量化解释，并揭示了10月7日后的隐藏转变。最先进的数据显示，这种战斗机的聚变-裂变正在发挥作用。它还预测未来将发生比10月7日更具杀伤力且来得更早的“超级震撼”袭击。它提供了一个多对手的解决方案。我们的结果--包括可测试的公式和即插即用的模拟--能够对未来的伤亡进行具体的风险评估，并根据战斗机的行为制定政策。



## **20. Boosting Certificate Robustness for Time Series Classification with Efficient Self-Ensemble**

通过高效的自集成提高时间序列分类的证书稳健性 cs.LG

6 figures, 4 tables, 10 pages

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.02802v1) [paper-pdf](http://arxiv.org/pdf/2409.02802v1)

**Authors**: Chang Dong, Zhengyang Li, Liangwei Zheng, Weitong Chen, Wei Emma Zhang

**Abstract**: Recently, the issue of adversarial robustness in the time series domain has garnered significant attention. However, the available defense mechanisms remain limited, with adversarial training being the predominant approach, though it does not provide theoretical guarantees. Randomized Smoothing has emerged as a standout method due to its ability to certify a provable lower bound on robustness radius under $\ell_p$-ball attacks. Recognizing its success, research in the time series domain has started focusing on these aspects. However, existing research predominantly focuses on time series forecasting, or under the non-$\ell_p$ robustness in statistic feature augmentation for time series classification~(TSC). Our review found that Randomized Smoothing performs modestly in TSC, struggling to provide effective assurances on datasets with poor robustness. Therefore, we propose a self-ensemble method to enhance the lower bound of the probability confidence of predicted labels by reducing the variance of classification margins, thereby certifying a larger radius. This approach also addresses the computational overhead issue of Deep Ensemble~(DE) while remaining competitive and, in some cases, outperforming it in terms of robustness. Both theoretical analysis and experimental results validate the effectiveness of our method, demonstrating superior performance in robustness testing compared to baseline approaches.

摘要: 最近，时间序列域中的对抗性稳健性问题引起了人们的广泛关注。然而，现有的防御机制仍然有限，对抗性训练是主要的方法，尽管它不提供理论上的保证。由于随机化平滑方法能够证明在$ell_p$-ball攻击下的健壮性半径的一个可证明的下界，所以它已经成为一种优秀的方法。认识到它的成功，时间序列领域的研究已经开始集中在这些方面。然而，现有的研究主要集中在时间序列预测，或在统计特征增强对时间序列分类具有非埃尔p稳健性的情况下。我们的综述发现，随机平滑在TSC中表现平平，难以对稳健性较差的数据集提供有效的保证。因此，我们提出了一种自集成方法，通过减小分类裕度的方差来提高预测标签的概率置信度下界，从而证明更大的半径。这种方法还解决了深层集成~(DE)的计算开销问题，同时保持了竞争力，在某些情况下，在健壮性方面优于它。理论分析和实验结果都验证了该方法的有效性，在稳健性测试中表现出了优于基线方法的性能。



## **21. AdvSecureNet: A Python Toolkit for Adversarial Machine Learning**

AdvSecureNet：对抗性机器学习的Python工具包 cs.CV

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.02629v1) [paper-pdf](http://arxiv.org/pdf/2409.02629v1)

**Authors**: Melih Catal, Manuel Günther

**Abstract**: Machine learning models are vulnerable to adversarial attacks. Several tools have been developed to research these vulnerabilities, but they often lack comprehensive features and flexibility. We introduce AdvSecureNet, a PyTorch based toolkit for adversarial machine learning that is the first to natively support multi-GPU setups for attacks, defenses, and evaluation. It is the first toolkit that supports both CLI and API interfaces and external YAML configuration files to enhance versatility and reproducibility. The toolkit includes multiple attacks, defenses and evaluation metrics. Rigiorous software engineering practices are followed to ensure high code quality and maintainability. The project is available as an open-source project on GitHub at https://github.com/melihcatal/advsecurenet and installable via PyPI.

摘要: 机器学习模型很容易受到对抗攻击。已经开发了多种工具来研究这些漏洞，但它们往往缺乏全面的功能和灵活性。我们引入了AdvSecureNet，这是一个基于PyTorch的对抗性机器学习工具包，是第一个本地支持用于攻击、防御和评估的多图形处理器设置的工具包。它是第一个支持CLI和API接口以及外部YML配置文件的工具包，以增强通用性和可重复性。该工具包包括多种攻击、防御和评估指标。遵循严格的软件工程实践以确保高代码质量和可维护性。该项目作为开源项目在GitHub上提供，网址为https://github.com/melihcatal/advsecurenet，并可通过PyPI安装。



## **22. Adversarial Attacks on Machine Learning-Aided Visualizations**

对机器学习辅助可视化的对抗攻击 cs.CR

This is the author's version of the article that has been accepted by  the Journal of Visualization

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.02485v1) [paper-pdf](http://arxiv.org/pdf/2409.02485v1)

**Authors**: Takanori Fujiwara, Kostiantyn Kucher, Junpeng Wang, Rafael M. Martins, Andreas Kerren, Anders Ynnerman

**Abstract**: Research in ML4VIS investigates how to use machine learning (ML) techniques to generate visualizations, and the field is rapidly growing with high societal impact. However, as with any computational pipeline that employs ML processes, ML4VIS approaches are susceptible to a range of ML-specific adversarial attacks. These attacks can manipulate visualization generations, causing analysts to be tricked and their judgments to be impaired. Due to a lack of synthesis from both visualization and ML perspectives, this security aspect is largely overlooked by the current ML4VIS literature. To bridge this gap, we investigate the potential vulnerabilities of ML-aided visualizations from adversarial attacks using a holistic lens of both visualization and ML perspectives. We first identify the attack surface (i.e., attack entry points) that is unique in ML-aided visualizations. We then exemplify five different adversarial attacks. These examples highlight the range of possible attacks when considering the attack surface and multiple different adversary capabilities. Our results show that adversaries can induce various attacks, such as creating arbitrary and deceptive visualizations, by systematically identifying input attributes that are influential in ML inferences. Based on our observations of the attack surface characteristics and the attack examples, we underline the importance of comprehensive studies of security issues and defense mechanisms as a call of urgency for the ML4VIS community.

摘要: ML4VIS的研究是研究如何使用机器学习(ML)技术来生成可视化，该领域正在迅速发展，具有很高的社会影响。然而，与任何使用ML进程的计算管道一样，ML4VIS方法容易受到一系列特定于ML的对抗性攻击。这些攻击可以操纵可视化世代，导致分析师上当受骗，他们的判断受到损害。由于缺乏可视化和ML视角的综合，当前的ML4VIS文献在很大程度上忽略了这一安全方面。为了弥补这一差距，我们使用可视化和ML视角的整体视角来研究ML辅助可视化在对抗攻击中的潜在脆弱性。我们首先确定在ML辅助可视化中唯一的攻击面(即攻击入口点)。然后，我们举例说明五种不同的对抗性攻击。这些示例突出显示了在考虑攻击面和多个不同对手能力时可能的攻击范围。我们的结果表明，攻击者可以通过系统地识别对ML推理有影响的输入属性来诱导各种攻击，例如创建任意和欺骗性的可视化。根据我们对攻击表面特征和攻击实例的观察，我们强调对安全问题和防御机制进行全面研究的重要性，作为ML4VIS社区的紧急呼吁。



## **23. TASAR: Transferable Attack on Skeletal Action Recognition**

TASAR：对Skelty动作识别的可转移攻击 cs.CV

arXiv admin note: text overlap with arXiv:2407.08572

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2409.02483v1) [paper-pdf](http://arxiv.org/pdf/2409.02483v1)

**Authors**: Yunfeng Diao, Baiqi Wu, Ruixuan Zhang, Ajian Liu, Xingxing Wei, Meng Wang, He Wang

**Abstract**: Skeletal sequences, as well-structured representations of human behaviors, are crucial in Human Activity Recognition (HAR). The transferability of adversarial skeletal sequences enables attacks in real-world HAR scenarios, such as autonomous driving, intelligent surveillance, and human-computer interactions. However, existing Skeleton-based HAR (S-HAR) attacks exhibit weak adversarial transferability and, therefore, cannot be considered true transfer-based S-HAR attacks. More importantly, the reason for this failure remains unclear. In this paper, we study this phenomenon through the lens of loss surface, and find that its sharpness contributes to the poor transferability in S-HAR. Inspired by this observation, we assume and empirically validate that smoothening the rugged loss landscape could potentially improve adversarial transferability in S-HAR. To this end, we propose the first Transfer-based Attack on Skeletal Action Recognition, TASAR. TASAR explores the smoothed model posterior without re-training the pre-trained surrogates, which is achieved by a new post-train Dual Bayesian optimization strategy. Furthermore, unlike previous transfer-based attacks that treat each frame independently and overlook temporal coherence within sequences, TASAR incorporates motion dynamics into the Bayesian attack gradient, effectively disrupting the spatial-temporal coherence of S-HARs. To exhaustively evaluate the effectiveness of existing methods and our method, we build the first large-scale robust S-HAR benchmark, comprising 7 S-HAR models, 10 attack methods, 3 S-HAR datasets and 2 defense models. Extensive results demonstrate the superiority of TASAR. Our benchmark enables easy comparisons for future studies, with the code available in the supplementary material.

摘要: 骨架序列作为人类行为的良好结构表示，在人类活动识别(HAR)中起着至关重要的作用。对抗性骨架序列的可转移性使攻击能够在真实世界的HAR场景中进行，例如自动驾驶、智能监控和人机交互。然而，现有的基于骨架的S-HAR攻击表现出较弱的对抗可转移性，因此不能被认为是真正的基于转移的S-HAR攻击。更重要的是，这一失败的原因尚不清楚。本文从损失面的角度对这一现象进行了研究，发现损失面的锐度是S-哈尔转移性差的原因之一。受到这一观察的启发，我们假设并经验验证，平滑崎岖的损失图景可能会提高S-哈尔的对抗性转移能力。为此，我们提出了第一个基于转移的骨骼动作识别攻击TASAR。Tasar通过一种新的训练后的双贝叶斯优化策略，在不重新训练预先训练的代理人的情况下，探索平滑的后验模型。此外，与以往基于传输的攻击独立对待每一帧并忽略序列内部的时间一致性不同，Tasar将运动动力学融入到贝叶斯攻击梯度中，有效地破坏了S-HARs的时空一致性。为了全面评估已有方法和本文方法的有效性，我们构建了第一个大规模稳健的S-HAR基准，包括7个S-HAR模型、10个攻击方法、3个S-HAR数据集和2个防御模型。广泛的结果证明了Tasar的优越性。我们的基准可以很容易地与补充材料中提供的代码进行比较，以便将来进行研究。



## **24. Jailbreaking Prompt Attack: A Controllable Adversarial Attack against Diffusion Models**

越狱提示攻击：针对扩散模型的可控对抗攻击 cs.CR

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2404.02928v3) [paper-pdf](http://arxiv.org/pdf/2404.02928v3)

**Authors**: Jiachen Ma, Anda Cao, Zhiqing Xiao, Yijiang Li, Jie Zhang, Chao Ye, Junbo Zhao

**Abstract**: Text-to-image (T2I) models can be maliciously used to generate harmful content such as sexually explicit, unfaithful, and misleading or Not-Safe-for-Work (NSFW) images. Previous attacks largely depend on the availability of the diffusion model or involve a lengthy optimization process. In this work, we investigate a more practical and universal attack that does not require the presence of a target model and demonstrate that the high-dimensional text embedding space inherently contains NSFW concepts that can be exploited to generate harmful images. We present the Jailbreaking Prompt Attack (JPA). JPA first searches for the target malicious concepts in the text embedding space using a group of antonyms generated by ChatGPT. Subsequently, a prefix prompt is optimized in the discrete vocabulary space to align malicious concepts semantically in the text embedding space. We further introduce a soft assignment with gradient masking technique that allows us to perform gradient ascent in the discrete vocabulary space.   We perform extensive experiments with open-sourced T2I models, e.g. stable-diffusion-v1-4 and closed-sourced online services, e.g. DALLE2, Midjourney with black-box safety checkers. Results show that (1) JPA bypasses both text and image safety checkers (2) while preserving high semantic alignment with the target prompt. (3) JPA demonstrates a much faster speed than previous methods and can be executed in a fully automated manner. These merits render it a valuable tool for robustness evaluation in future text-to-image generation research.

摘要: 文本到图像(T2I)模型可被恶意用于生成有害内容，如性露骨、不忠、误导性或不安全工作(NSFW)图像。以前的攻击在很大程度上取决于扩散模型的可用性，或者涉及漫长的优化过程。在这项工作中，我们研究了一种更实用和通用的攻击，它不需要目标模型的存在，并证明了高维文本嵌入空间内在地包含可被利用来生成有害图像的NSFW概念。我们提出了越狱快速攻击(JPA)。JPA首先使用ChatGPT生成的一组反义词在文本嵌入空间中搜索目标恶意概念。随后，在离散词汇空间中优化前缀提示，以在文本嵌入空间中对恶意概念进行语义对齐。我们进一步介绍了一种使用梯度掩蔽技术的软赋值方法，它允许我们在离散词汇空间中执行梯度上升。我们对开源的T2I模型进行了广泛的实验，例如稳定扩散-v1-4和封闭源代码的在线服务，例如DALLE2，中途使用黑盒安全检查器。结果表明：(1)JPA避开了文本和图像的安全检查；(2)JPA在保持与目标提示高度语义一致性的同时，避开了文本和图像安全检查。(3)JPA表现出比以前的方法更快的速度，并且可以以全自动的方式执行。这些优点使得它成为未来文本到图像生成研究中健壮性评估的有价值的工具。



## **25. LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet**

LLM辩护对多次越狱还不强 cs.LG

**SubmitDate**: 2024-09-04    [abs](http://arxiv.org/abs/2408.15221v2) [paper-pdf](http://arxiv.org/pdf/2408.15221v2)

**Authors**: Nathaniel Li, Ziwen Han, Ian Steneker, Willow Primack, Riley Goodside, Hugh Zhang, Zifan Wang, Cristina Menghini, Summer Yue

**Abstract**: Recent large language model (LLM) defenses have greatly improved models' ability to refuse harmful queries, even when adversarially attacked. However, LLM defenses are primarily evaluated against automated adversarial attacks in a single turn of conversation, an insufficient threat model for real-world malicious use. We demonstrate that multi-turn human jailbreaks uncover significant vulnerabilities, exceeding 70% attack success rate (ASR) on HarmBench against defenses that report single-digit ASRs with automated single-turn attacks. Human jailbreaks also reveal vulnerabilities in machine unlearning defenses, successfully recovering dual-use biosecurity knowledge from unlearned models. We compile these results into Multi-Turn Human Jailbreaks (MHJ), a dataset of 2,912 prompts across 537 multi-turn jailbreaks. We publicly release MHJ alongside a compendium of jailbreak tactics developed across dozens of commercial red teaming engagements, supporting research towards stronger LLM defenses.

摘要: 最近的大型语言模型(LLM)防御极大地提高了模型拒绝有害查询的能力，即使在遭到恶意攻击时也是如此。然而，LLM防御主要是在单轮对话中针对自动对手攻击进行评估，这不足以构成现实世界恶意使用的威胁模型。我们证明，多轮人类越狱揭示了重大漏洞，针对使用自动化单轮攻击报告个位数ASR的防御系统，HarmB边上的攻击成功率(ASR)超过70%。人类越狱还揭示了机器遗忘防御的漏洞，成功地从未学习的模型中恢复了双重用途的生物安全知识。我们将这些结果汇编成多轮人类越狱(MHJ)，这是一个包含537次多轮越狱的2912个提示的数据集。我们公开发布了MHJ，以及在数十个商业红色团队交战中开发的越狱战术概要，支持对更强大的LLM防御的研究。



## **26. RAMBO: Leaking Secrets from Air-Gap Computers by Spelling Covert Radio Signals from Computer RAM**

RAMBO：通过拼写计算机RAM中的秘密无线电信号来泄露空间隙计算机的秘密 cs.CR

Version of this work accepted to Nordic Conference on Secure IT  Systems, 2023

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.02292v1) [paper-pdf](http://arxiv.org/pdf/2409.02292v1)

**Authors**: Mordechai Guri

**Abstract**: Air-gapped systems are physically separated from external networks, including the Internet. This isolation is achieved by keeping the air-gap computers disconnected from wired or wireless networks, preventing direct or remote communication with other devices or networks. Air-gap measures may be used in sensitive environments where security and isolation are critical to prevent private and confidential information leakage.   In this paper, we present an attack allowing adversaries to leak information from air-gapped computers. We show that malware on a compromised computer can generate radio signals from memory buses (RAM). Using software-generated radio signals, malware can encode sensitive information such as files, images, keylogging, biometric information, and encryption keys. With software-defined radio (SDR) hardware, and a simple off-the-shelf antenna, an attacker can intercept transmitted raw radio signals from a distance. The signals can then be decoded and translated back into binary information. We discuss the design and implementation and present related work and evaluation results. This paper presents fast modification methods to leak data from air-gapped computers at 1000 bits per second. Finally, we propose countermeasures to mitigate this out-of-band air-gap threat.

摘要: 气隙系统在物理上与外部网络隔离，包括互联网。这种隔离是通过保持气隙计算机与有线或无线网络断开，防止与其他设备或网络进行直接或远程通信来实现的。气隙措施可用于敏感环境中，在这些环境中，安全和隔离对于防止私人和机密信息泄露至关重要。在本文中，我们提出了一种允许攻击者从有空隙的计算机中泄露信息的攻击。我们发现，受攻击的计算机上的恶意软件可以从内存总线(RAM)生成无线电信号。使用软件生成的无线电信号，恶意软件可以对文件、图像、键盘记录、生物识别信息和加密密钥等敏感信息进行编码。使用软件定义的无线电(SDR)硬件和简单的现成天线，攻击者可以从远处拦截传输的原始无线电信号。然后可以对信号进行解码，并将其转换回二进制信息。我们讨论了该系统的设计和实现，并给出了相关工作和评估结果。提出了一种针对1000比特/秒的气隙计算机数据泄漏的快速修正方法。最后，我们提出了缓解这种带外气隙威胁的对策。



## **27. NoiseAttack: An Evasive Sample-Specific Multi-Targeted Backdoor Attack Through White Gaussian Noise**

NoiseAttack：通过高斯白噪音的规避样本特定多目标后门攻击 cs.CV

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.02251v1) [paper-pdf](http://arxiv.org/pdf/2409.02251v1)

**Authors**: Abdullah Arafat Miah, Kaan Icer, Resit Sendag, Yu Bi

**Abstract**: Backdoor attacks pose a significant threat when using third-party data for deep learning development. In these attacks, data can be manipulated to cause a trained model to behave improperly when a specific trigger pattern is applied, providing the adversary with unauthorized advantages. While most existing works focus on designing trigger patterns in both visible and invisible to poison the victim class, they typically result in a single targeted class upon the success of the backdoor attack, meaning that the victim class can only be converted to another class based on the adversary predefined value. In this paper, we address this issue by introducing a novel sample-specific multi-targeted backdoor attack, namely NoiseAttack. Specifically, we adopt White Gaussian Noise (WGN) with various Power Spectral Densities (PSD) as our underlying triggers, coupled with a unique training strategy to execute the backdoor attack. This work is the first of its kind to launch a vision backdoor attack with the intent to generate multiple targeted classes with minimal input configuration. Furthermore, our extensive experimental results demonstrate that NoiseAttack can achieve a high attack success rate against popular network architectures and datasets, as well as bypass state-of-the-art backdoor detection methods. Our source code and experiments are available at https://github.com/SiSL-URI/NoiseAttack/tree/main.

摘要: 在使用第三方数据进行深度学习开发时，后门攻击构成了重大威胁。在这些攻击中，当应用特定的触发模式时，可以操纵数据以导致训练的模型行为不正确，从而为对手提供未经授权的优势。虽然大多数现有的工作都集中在设计可见和不可见的触发模式来毒化受害者类，但它们通常会在后门攻击成功时产生一个目标类，这意味着受害者类只能基于对手预定义的值转换为另一个类。在本文中，我们通过引入一种新的特定样本的多目标后门攻击来解决这个问题，即NoiseAttack。具体地说，我们采用具有不同功率谱密度(PSD)的高斯白噪声(WGN)作为潜在触发因素，并使用独特的训练策略来执行后门攻击。这是同类工作中第一次发起Vision后门攻击，目的是用最少的输入配置生成多个目标类。此外，我们的大量实验结果表明，NoiseAttack可以对流行的网络体系结构和数据集实现高攻击成功率，并绕过最先进的后门检测方法。我们的源代码和实验可在https://github.com/SiSL-URI/NoiseAttack/tree/main.上获得



## **28. Quantifying Liveness and Safety of Avalanche's Snowball**

量化雪崩雪球的生命力和安全性 cs.DC

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.02217v1) [paper-pdf](http://arxiv.org/pdf/2409.02217v1)

**Authors**: Quentin Kniep, Maxime Laval, Jakub Sliwinski, Roger Wattenhofer

**Abstract**: This work examines the resilience properties of the Snowball and Avalanche protocols that underlie the popular Avalanche blockchain. We experimentally quantify the resilience of Snowball using a simulation implemented in Rust, where the adversary strategically rebalances the network to delay termination.   We show that in a network of $n$ nodes of equal stake, the adversary is able to break liveness when controlling $\Omega(\sqrt{n})$ nodes. Specifically, for $n = 2000$, a simple adversary controlling $5.2\%$ of stake can successfully attack liveness. When the adversary is given additional information about the state of the network (without any communication or other advantages), the stake needed for a successful attack is as little as $2.8\%$. We show that the adversary can break safety in time exponentially dependent on their stake, and inversely linearly related to the size of the network, e.g. in 265 rounds in expectation when the adversary controls $25\%$ of a network of 3000.   We conclude that Snowball and Avalanche are akin to Byzantine reliable broadcast protocols as opposed to consensus.

摘要: 这项工作研究了雪球和雪崩协议的弹性属性，这些协议是流行的雪崩区块链的基础。我们使用在Rust中实施的模拟来实验量化Snowball的弹性，其中对手战略性地重新平衡网络以延迟终止。我们证明了在具有相同利害关系的$n$节点的网络中，当对手控制$\Omega(\Sqrt{n})$节点时，攻击者能够破坏活性。具体地说，对于$n=2000$，一个控制$5.2\$赌注的简单对手可以成功地攻击活跃度。当对手获得有关网络状态的附加信息(没有任何通信或其他优势)时，成功攻击所需的赌注仅为2.8美元。我们证明了当敌手控制3000个网络中的$25\$时，对手可以在时间上以指数依赖于他们的赌注打破安全性，并且与网络的大小成反比线性关系，例如在265轮中。我们的结论是，雪球和雪崩类似于拜占庭可靠的广播协议，而不是共识。



## **29. Learning Resilient Formation Control of Drones with Graph Attention Network**

利用图注意力网络学习无人机弹性队形控制 cs.RO

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.01953v1) [paper-pdf](http://arxiv.org/pdf/2409.01953v1)

**Authors**: Jiaping Xiao, Xu Fang, Qianlei Jia, Mir Feroskhan

**Abstract**: The rapid advancement of drone technology has significantly impacted various sectors, including search and rescue, environmental surveillance, and industrial inspection. Multidrone systems offer notable advantages such as enhanced efficiency, scalability, and redundancy over single-drone operations. Despite these benefits, ensuring resilient formation control in dynamic and adversarial environments, such as under communication loss or cyberattacks, remains a significant challenge. Classical approaches to resilient formation control, while effective in certain scenarios, often struggle with complex modeling and the curse of dimensionality, particularly as the number of agents increases. This paper proposes a novel, learning-based formation control for enhancing the adaptability and resilience of multidrone formations using graph attention networks (GATs). By leveraging GAT's dynamic capabilities to extract internode relationships based on the attention mechanism, this GAT-based formation controller significantly improves the robustness of drone formations against various threats, such as Denial of Service (DoS) attacks. Our approach not only improves formation performance in normal conditions but also ensures the resilience of multidrone systems in variable and adversarial environments. Extensive simulation results demonstrate the superior performance of our method over baseline formation controllers. Furthermore, the physical experiments validate the effectiveness of the trained control policy in real-world flights.

摘要: 无人机技术的快速进步对包括搜救、环境监测和工业检查在内的各个领域都产生了重大影响。与单无人机相比，多无人机系统具有显著的优势，如更高的效率、可扩展性和冗余性。尽管有这些好处，但在动态和敌对环境中，例如在通信中断或网络攻击下，确保有弹性的编队控制仍然是一个重大挑战。传统的弹性编队控制方法虽然在某些情况下有效，但经常与复杂的建模和维度诅咒作斗争，特别是随着代理人数量的增加。本文提出了一种新的基于学习的编队控制方法，利用图注意网络(GATS)来增强多无人机编队的适应性和弹性。通过利用GAT的动态能力来提取基于注意机制的节点间关系，这种基于GAT的编队控制器显著提高了无人机编队对抗各种威胁(如拒绝服务(DoS)攻击)的稳健性。我们的方法不仅提高了正常条件下的编队性能，还确保了多无人机系统在多变和敌对环境中的弹性。大量的仿真结果表明，该方法的性能优于基线编队控制器。此外，物理实验验证了训练后的控制策略在真实飞行中的有效性。



## **30. Reassessing Noise Augmentation Methods in the Context of Adversarial Speech**

对抗性言语背景下重新评估噪音增强方法 eess.AS

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.01813v1) [paper-pdf](http://arxiv.org/pdf/2409.01813v1)

**Authors**: Karla Pizzi, Matías P. Pizarro B, Asja Fischer

**Abstract**: In this study, we investigate if noise-augmented training can concurrently improve adversarial robustness in automatic speech recognition (ASR) systems. We conduct a comparative analysis of the adversarial robustness of four different state-of-the-art ASR architectures, where each of the ASR architectures is trained under three different augmentation conditions: one subject to background noise, speed variations, and reverberations, another subject to speed variations only, and a third without any form of data augmentation. The results demonstrate that noise augmentation not only improves model performance on noisy speech but also the model's robustness to adversarial attacks.

摘要: 在这项研究中，我们研究了噪音增强训练是否可以同时提高自动语音识别（ASB）系统中的对抗鲁棒性。我们对四种不同最先进的ASB架构的对抗鲁棒性进行了比较分析，其中每个ASB架构都在三种不同的增强条件下训练：一种受到背景噪音、速度变化和回响的影响，另一种仅受到速度变化的影响，第三种没有任何形式的数据增强。结果表明，噪音增强不仅提高了模型在含噪语音上的性能，还提高了模型对对抗攻击的鲁棒性。



## **31. USTC-KXDIGIT System Description for ASVspoof5 Challenge**

USTC-KXDIGIT ASVspoof 5挑战赛系统描述 cs.SD

ASVspoof5 workshop paper

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.01695v1) [paper-pdf](http://arxiv.org/pdf/2409.01695v1)

**Authors**: Yihao Chen, Haochen Wu, Nan Jiang, Xiang Xia, Qing Gu, Yunqi Hao, Pengfei Cai, Yu Guan, Jialong Wang, Weilin Xie, Lei Fang, Sian Fang, Yan Song, Wu Guo, Lin Liu, Minqiang Xu

**Abstract**: This paper describes the USTC-KXDIGIT system submitted to the ASVspoof5 Challenge for Track 1 (speech deepfake detection) and Track 2 (spoofing-robust automatic speaker verification, SASV). Track 1 showcases a diverse range of technical qualities from potential processing algorithms and includes both open and closed conditions. For these conditions, our system consists of a cascade of a frontend feature extractor and a back-end classifier. We focus on extensive embedding engineering and enhancing the generalization of the back-end classifier model. Specifically, the embedding engineering is based on hand-crafted features and speech representations from a self-supervised model, used for closed and open conditions, respectively. To detect spoof attacks under various adversarial conditions, we trained multiple systems on an augmented training set. Additionally, we used voice conversion technology to synthesize fake audio from genuine audio in the training set to enrich the synthesis algorithms. To leverage the complementary information learned by different model architectures, we employed activation ensemble and fused scores from different systems to obtain the final decision score for spoof detection. During the evaluation phase, the proposed methods achieved 0.3948 minDCF and 14.33% EER in the close condition, and 0.0750 minDCF and 2.59% EER in the open condition, demonstrating the robustness of our submitted systems under adversarial conditions. In Track 2, we continued using the CM system from Track 1 and fused it with a CNN-based ASV system. This approach achieved 0.2814 min-aDCF in the closed condition and 0.0756 min-aDCF in the open condition, showcasing superior performance in the SASV system.

摘要: 本文描述了USTC-KXDIGIT系统提交给ASVspoof5挑战赛的Track 1(语音深度伪码检测)和Track 2(Spoofing-Robust Automatic Speaker Verify，SASV)。Track 1展示了潜在处理算法的各种技术品质，并包括开放和封闭条件。对于这些情况，我们的系统包括一个前端特征提取器级联和一个后端分类器。我们着重于广泛的嵌入工程和增强后端分类器模型的泛化。具体地说，嵌入工程基于手工制作的特征和来自自监督模型的语音表示，分别用于封闭和开放条件。为了在不同的对抗条件下检测欺骗攻击，我们在一个扩充的训练集上训练了多个系统。此外，我们还使用语音转换技术从训练集中的真实音频中合成假音频，以丰富合成算法。为了利用不同模型体系结构学习到的互补信息，我们使用激活集成和融合来自不同系统的分数来获得最终的决策分数以进行欺骗检测。在评估阶段，所提出的方法在封闭条件下达到了0.3948 minDCF和14.33%的EER，在开放条件下达到了0.0750 minDCF和2.59%的EER，证明了我们所提出的系统在对抗条件下的健壮性。在Track 2中，我们继续使用Track 1中的CM系统，并将其与基于CNN的ASV系统融合。该方法在闭合状态下获得了0.2814分钟的平均无功补偿能力，在开启状态下达到了0.0756分钟的平均无功补偿能力，在空间电压调节系统中表现出了优异的性能。



## **32. Purification-Agnostic Proxy Learning for Agentic Copyright Watermarking against Adversarial Evidence Forgery**

净化-不可知的代理学习，用于针对对抗性证据伪造的大型版权水印 cs.CV

**SubmitDate**: 2024-09-03    [abs](http://arxiv.org/abs/2409.01541v1) [paper-pdf](http://arxiv.org/pdf/2409.01541v1)

**Authors**: Erjin Bao, Ching-Chun Chang, Hanrui Wang, Isao Echizen

**Abstract**: With the proliferation of AI agents in various domains, protecting the ownership of AI models has become crucial due to the significant investment in their development. Unauthorized use and illegal distribution of these models pose serious threats to intellectual property, necessitating effective copyright protection measures. Model watermarking has emerged as a key technique to address this issue, embedding ownership information within models to assert rightful ownership during copyright disputes. This paper presents several contributions to model watermarking: a self-authenticating black-box watermarking protocol using hash techniques, a study on evidence forgery attacks using adversarial perturbations, a proposed defense involving a purification step to counter adversarial attacks, and a purification-agnostic proxy learning method to enhance watermark reliability and model performance. Experimental results demonstrate the effectiveness of these approaches in improving the security, reliability, and performance of watermarked models.

摘要: 随着AI代理在各个领域的激增，保护AI模型的所有权变得至关重要，因为它们在开发方面投入了大量资金。未经授权使用和非法传播这些模型对知识产权构成严重威胁，需要采取有效的版权保护措施。模型水印已经成为解决这一问题的一项关键技术，它将所有权信息嵌入到模型中，以在版权纠纷中断言合法的所有权。提出了一种基于散列技术的自认证黑盒水印协议，研究了基于对抗性扰动的证据伪造攻击，提出了一种针对对抗性攻击的净化步骤防御方法，以及一种与净化无关的代理学习方法，以提高水印的可靠性和模型性能。实验结果表明，这些方法在提高水印模型的安全性、可靠性和性能方面是有效的。



## **33. On Evaluating Adversarial Robustness of Volumetric Medical Segmentation Models**

评估容量医疗细分模型的对抗稳健性 eess.IV

Accepted at British Machine Vision Conference 2024

**SubmitDate**: 2024-09-02    [abs](http://arxiv.org/abs/2406.08486v2) [paper-pdf](http://arxiv.org/pdf/2406.08486v2)

**Authors**: Hashmat Shadab Malik, Numan Saeed, Asif Hanif, Muzammal Naseer, Mohammad Yaqub, Salman Khan, Fahad Shahbaz Khan

**Abstract**: Volumetric medical segmentation models have achieved significant success on organ and tumor-based segmentation tasks in recent years. However, their vulnerability to adversarial attacks remains largely unexplored, raising serious concerns regarding the real-world deployment of tools employing such models in the healthcare sector. This underscores the importance of investigating the robustness of existing models. In this context, our work aims to empirically examine the adversarial robustness across current volumetric segmentation architectures, encompassing Convolutional, Transformer, and Mamba-based models. We extend this investigation across four volumetric segmentation datasets, evaluating robustness under both white box and black box adversarial attacks. Overall, we observe that while both pixel and frequency-based attacks perform reasonably well under \emph{white box} setting, the latter performs significantly better under transfer-based black box attacks. Across our experiments, we observe transformer-based models show higher robustness than convolution-based models with Mamba-based models being the most vulnerable. Additionally, we show that large-scale training of volumetric segmentation models improves the model's robustness against adversarial attacks. The code and robust models are available at https://github.com/HashmatShadab/Robustness-of-Volumetric-Medical-Segmentation-Models.

摘要: 近年来，体积医学分割模型在基于器官和肿瘤的分割任务中取得了显著的成功。然而，它们在对抗性攻击中的脆弱性在很大程度上仍未得到探索，这引发了人们对在医疗保健部门使用此类模型的工具在现实世界中的部署的严重关切。这凸显了调查现有模型的稳健性的重要性。在此背景下，我们的工作旨在经验性地检查当前体积分割体系结构的对抗性健壮性，包括卷积、变压器和基于Mamba的模型。我们将这项研究扩展到四个体积分割数据集，评估了白盒和黑盒攻击下的稳健性。总体而言，我们观察到，虽然基于像素的攻击和基于频率的攻击在白盒设置下都表现得相当好，但后者在基于传输的黑盒攻击下的性能要好得多。在我们的实验中，我们观察到基于变压器的模型比基于卷积的模型表现出更高的稳健性，其中基于Mamba的模型是最脆弱的。此外，我们还证明了对体积分割模型的大规模训练提高了模型对对手攻击的稳健性。代码和健壮模型可在https://github.com/HashmatShadab/Robustness-of-Volumetric-Medical-Segmentation-Models.上获得



## **34. One-Index Vector Quantization Based Adversarial Attack on Image Classification**

基于单指标量化的图像分类对抗攻击 cs.CV

**SubmitDate**: 2024-09-02    [abs](http://arxiv.org/abs/2409.01282v1) [paper-pdf](http://arxiv.org/pdf/2409.01282v1)

**Authors**: Haiju Fan, Xiaona Qin, Shuang Chen, Hubert P. H. Shum, Ming Li

**Abstract**: To improve storage and transmission, images are generally compressed. Vector quantization (VQ) is a popular compression method as it has a high compression ratio that suppresses other compression techniques. Despite this, existing adversarial attack methods on image classification are mostly performed in the pixel domain with few exceptions in the compressed domain, making them less applicable in real-world scenarios. In this paper, we propose a novel one-index attack method in the VQ domain to generate adversarial images by a differential evolution algorithm, successfully resulting in image misclassification in victim models. The one-index attack method modifies a single index in the compressed data stream so that the decompressed image is misclassified. It only needs to modify a single VQ index to realize an attack, which limits the number of perturbed indexes. The proposed method belongs to a semi-black-box attack, which is more in line with the actual attack scenario. We apply our method to attack three popular image classification models, i.e., Resnet, NIN, and VGG16. On average, 55.9% and 77.4% of the images in CIFAR-10 and Fashion MNIST, respectively, are successfully attacked, with a high level of misclassification confidence and a low level of image perturbation.

摘要: 为了改善存储和传输，图像通常会被压缩。矢量量化(VQ)是一种流行的压缩方法，因为它具有比其他压缩技术更高的压缩比。尽管如此，现有的图像分类对抗性攻击方法大多是在像素域进行的，很少在压缩域进行，这使得它们在实际场景中的适用性较差。本文提出了一种新的基于VQ域的单索引攻击方法，利用差分进化算法生成敌意图像，成功地导致了受害者模型中图像的误分类。单索引攻击方法修改压缩数据流中的单个索引，从而使解压缩图像被错误分类。它只需要修改一个VQ索引就可以实现攻击，从而限制了被扰动的索引的数量。该方法属于半黑盒攻击，更符合实际攻击场景。我们应用我们的方法攻击了三种流行的图像分类模型，即RESNET、NIN和VGG16。平均而言，CIFAR-10和Fashion MNIST分别有55.9%和77.4%的图像被成功攻击，具有较高的误分类置信度和较低的图像扰动水平。



## **35. A Review of Image Retrieval Techniques: Data Augmentation and Adversarial Learning Approaches**

图像检索技术回顾：数据增强和对抗学习方法 cs.CV

**SubmitDate**: 2024-09-02    [abs](http://arxiv.org/abs/2409.01219v1) [paper-pdf](http://arxiv.org/pdf/2409.01219v1)

**Authors**: Kim Jinwoo

**Abstract**: Image retrieval is a crucial research topic in computer vision, with broad application prospects ranging from online product searches to security surveillance systems. In recent years, the accuracy and efficiency of image retrieval have significantly improved due to advancements in deep learning. However, existing methods still face numerous challenges, particularly in handling large-scale datasets, cross-domain retrieval, and image perturbations that can arise from real-world conditions such as variations in lighting, occlusion, and viewpoint. Data augmentation techniques and adversarial learning methods have been widely applied in the field of image retrieval to address these challenges. Data augmentation enhances the model's generalization ability and robustness by generating more diverse training samples, simulating real-world variations, and reducing overfitting. Meanwhile, adversarial attacks and defenses introduce perturbations during training to improve the model's robustness against potential attacks, ensuring reliability in practical applications. This review comprehensively summarizes the latest research advancements in image retrieval, with a particular focus on the roles of data augmentation and adversarial learning techniques in enhancing retrieval performance. Future directions and potential challenges are also discussed.

摘要: 图像检索是计算机视觉中的一个重要研究课题，从在线产品搜索到安全监控系统都有广泛的应用前景。近年来，由于深度学习的进步，图像检索的准确率和效率都有了显著的提高。然而，现有的方法仍然面临着许多挑战，特别是在处理大规模数据集、跨域检索和由于真实世界条件(例如光照、遮挡和视点的变化)而引起的图像扰动方面。数据增强技术和对抗性学习方法已被广泛应用于图像检索领域，以应对这些挑战。数据扩充通过生成更多样化的训练样本、模拟真实世界的变化和减少过拟合来增强模型的泛化能力和稳健性。同时，对抗性攻击和防御在训练过程中引入扰动，提高了模型对潜在攻击的鲁棒性，确保了实际应用中的可靠性。本文综述了图像检索领域的最新研究进展，重点介绍了数据增强和对抗性学习技术在提高检索性能中的作用。文中还讨论了未来的发展方向和潜在的挑战。



## **36. BadMerging: Backdoor Attacks Against Model Merging**

BadMerging：针对模型合并的后门攻击 cs.CR

To appear in ACM Conference on Computer and Communications Security  (CCS), 2024

**SubmitDate**: 2024-09-02    [abs](http://arxiv.org/abs/2408.07362v2) [paper-pdf](http://arxiv.org/pdf/2408.07362v2)

**Authors**: Jinghuai Zhang, Jianfeng Chi, Zheng Li, Kunlin Cai, Yang Zhang, Yuan Tian

**Abstract**: Fine-tuning pre-trained models for downstream tasks has led to a proliferation of open-sourced task-specific models. Recently, Model Merging (MM) has emerged as an effective approach to facilitate knowledge transfer among these independently fine-tuned models. MM directly combines multiple fine-tuned task-specific models into a merged model without additional training, and the resulting model shows enhanced capabilities in multiple tasks. Although MM provides great utility, it may come with security risks because an adversary can exploit MM to affect multiple downstream tasks. However, the security risks of MM have barely been studied. In this paper, we first find that MM, as a new learning paradigm, introduces unique challenges for existing backdoor attacks due to the merging process. To address these challenges, we introduce BadMerging, the first backdoor attack specifically designed for MM. Notably, BadMerging allows an adversary to compromise the entire merged model by contributing as few as one backdoored task-specific model. BadMerging comprises a two-stage attack mechanism and a novel feature-interpolation-based loss to enhance the robustness of embedded backdoors against the changes of different merging parameters. Considering that a merged model may incorporate tasks from different domains, BadMerging can jointly compromise the tasks provided by the adversary (on-task attack) and other contributors (off-task attack) and solve the corresponding unique challenges with novel attack designs. Extensive experiments show that BadMerging achieves remarkable attacks against various MM algorithms. Our ablation study demonstrates that the proposed attack designs can progressively contribute to the attack performance. Finally, we show that prior defense mechanisms fail to defend against our attacks, highlighting the need for more advanced defense.

摘要: 为下游任务微调预先训练的模型导致了开源特定于任务的模型的激增。最近，模型合并(MM)已成为促进这些独立微调模型之间知识转移的一种有效方法。MM直接将多个微调的特定于任务的模型组合成一个合并模型，而无需额外的培训，所得到的模型在多个任务中显示出增强的能力。虽然MM提供了强大的实用程序，但它可能伴随着安全风险，因为攻击者可以利用MM来影响多个下游任务。然而，MM的安全风险几乎没有被研究过。在本文中，我们首先发现MM作为一种新的学习范式，由于合并过程给现有的后门攻击带来了独特的挑战。为了应对这些挑战，我们引入了BadMerging，这是第一个专门为MM设计的后门攻击。值得注意的是，BadMerging允许对手通过贡献一个后备的特定任务模型来危害整个合并的模型。BadMerging包括一种两阶段攻击机制和一种新的基于特征内插的损失机制，以增强嵌入式后门对不同合并参数变化的鲁棒性。考虑到合并模型可能包含来自不同领域的任务，BadMerging可以联合妥协对手(任务上攻击)和其他贡献者(任务外攻击)提供的任务，并以新颖的攻击设计解决相应的独特挑战。大量实验表明，BadMerging对各种MM算法具有显著的攻击效果。我们的烧蚀研究表明，所提出的攻击设计可以逐渐对攻击性能做出贡献。最后，我们展示了先前的防御机制无法防御我们的攻击，强调了需要更高级的防御。



## **37. A Grey-box Attack against Latent Diffusion Model-based Image Editing by Posterior Collapse**

对基于潜在扩散模型的图像编辑的灰箱攻击 cs.CV

21 pages, 7 figures, 10 tables

**SubmitDate**: 2024-09-02    [abs](http://arxiv.org/abs/2408.10901v2) [paper-pdf](http://arxiv.org/pdf/2408.10901v2)

**Authors**: Zhongliang Guo, Lei Fang, Jingyu Lin, Yifei Qian, Shuai Zhao, Zeyu Wang, Junhao Dong, Cunjian Chen, Ognjen Arandjelović, Chun Pong Lau

**Abstract**: Recent advancements in generative AI, particularly Latent Diffusion Models (LDMs), have revolutionized image synthesis and manipulation. However, these generative techniques raises concerns about data misappropriation and intellectual property infringement. Adversarial attacks on machine learning models have been extensively studied, and a well-established body of research has extended these techniques as a benign metric to prevent the underlying misuse of generative AI. Current approaches to safeguarding images from manipulation by LDMs are limited by their reliance on model-specific knowledge and their inability to significantly degrade semantic quality of generated images. In response to these shortcomings, we propose the Posterior Collapse Attack (PCA) based on the observation that VAEs suffer from posterior collapse during training. Our method minimizes dependence on the white-box information of target models to get rid of the implicit reliance on model-specific knowledge. By accessing merely a small amount of LDM parameters, in specific merely the VAE encoder of LDMs, our method causes a substantial semantic collapse in generation quality, particularly in perceptual consistency, and demonstrates strong transferability across various model architectures. Experimental results show that PCA achieves superior perturbation effects on image generation of LDMs with lower runtime and VRAM. Our method outperforms existing techniques, offering a more robust and generalizable solution that is helpful in alleviating the socio-technical challenges posed by the rapidly evolving landscape of generative AI.

摘要: 生成性人工智能的最新进展，特别是潜在扩散模型(LDM)，已经彻底改变了图像合成和处理。然而，这些生成性技术引发了人们对数据挪用和侵犯知识产权的担忧。对机器学习模型的对抗性攻击已经被广泛研究，一系列成熟的研究已经将这些技术扩展为一种良性的衡量标准，以防止潜在的生成性人工智能的滥用。当前保护图像免受LDM操纵的方法受到它们对模型特定知识的依赖以及它们无法显著降低所生成图像的语义质量的限制。针对这些不足，我们提出了后部塌陷攻击(PCA)，基于VAE在训练过程中遭受后部塌陷的观察。我们的方法最大限度地减少了对目标模型白盒信息的依赖，摆脱了对特定模型知识的隐含依赖。通过只访问少量的LDM参数，特别是LDM的VAE编码器，我们的方法导致生成质量的语义崩溃，特别是在感知一致性方面，并表现出强大的跨模型体系结构的可移植性。实验结果表明，主成分分析算法以较低的运行时间和较低的VRAM实现了较好的图像生成扰动效果。我们的方法优于现有的技术，提供了一个更健壮和更具通用性的解决方案，有助于缓解快速发展的生成性人工智能所带来的社会技术挑战。



## **38. Fisher Information guided Purification against Backdoor Attacks**

费舍尔信息指导净化针对后门攻击 cs.CV

Accepted to ACM CCS 2024. arXiv admin note: text overlap with  arXiv:2306.17441

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2409.00863v1) [paper-pdf](http://arxiv.org/pdf/2409.00863v1)

**Authors**: Nazmul Karim, Abdullah Al Arafat, Adnan Siraj Rakin, Zhishan Guo, Nazanin Rahnavard

**Abstract**: Studies on backdoor attacks in recent years suggest that an adversary can compromise the integrity of a deep neural network (DNN) by manipulating a small set of training samples. Our analysis shows that such manipulation can make the backdoor model converge to a bad local minima, i.e., sharper minima as compared to a benign model. Intuitively, the backdoor can be purified by re-optimizing the model to smoother minima. However, a na\"ive adoption of any optimization targeting smoother minima can lead to sub-optimal purification techniques hampering the clean test accuracy. Hence, to effectively obtain such re-optimization, inspired by our novel perspective establishing the connection between backdoor removal and loss smoothness, we propose Fisher Information guided Purification (FIP), a novel backdoor purification framework. Proposed FIP consists of a couple of novel regularizers that aid the model in suppressing the backdoor effects and retaining the acquired knowledge of clean data distribution throughout the backdoor removal procedure through exploiting the knowledge of Fisher Information Matrix (FIM). In addition, we introduce an efficient variant of FIP, dubbed as Fast FIP, which reduces the number of tunable parameters significantly and obtains an impressive runtime gain of almost $5\times$. Extensive experiments show that the proposed method achieves state-of-the-art (SOTA) performance on a wide range of backdoor defense benchmarks: 5 different tasks -- Image Recognition, Object Detection, Video Action Recognition, 3D point Cloud, Language Generation; 11 different datasets including ImageNet, PASCAL VOC, UCF101; diverse model architectures spanning both CNN and vision transformer; 14 different backdoor attacks, e.g., Dynamic, WaNet, LIRA, ISSBA, etc.

摘要: 近年来关于后门攻击的研究表明，对手可以通过操纵一小组训练样本来破坏深度神经网络(DNN)的完整性。我们的分析表明，这种操作可以使后门模型收敛到一个坏的局部极小点，即与良性模型相比，更尖锐的极小点。直观地说，后门可以通过重新优化模型来优化到更平滑的最小值。然而，单纯地采用任何以更平滑的最小值为目标的优化都可能导致次优的纯化技术阻碍清洁测试的准确性。因此，为了有效地获得这种重新优化，受我们建立后门去除和损失平滑之间联系的新观点的启发，我们提出了一种新的后门净化框架Fisher信息引导净化(FIP)。提出的FIP由两个新颖的正则化器组成，通过利用Fisher信息矩阵(FIM)的知识，帮助模型抑制后门效应，并在后门删除过程中保留所获得的干净数据分布的知识。此外，我们还引入了FIP的一个有效的变体，称为Fast FIP，它显著地减少了可调参数的数量，并获得了近5倍于$的可观的运行时间收益。大量实验表明，该方法在多种后门防御基准上取得了最好的性能：5种不同的任务--图像识别、目标检测、视频动作识别、3D点云、语言生成；11种不同的数据集，包括ImageNet、Pascal VOC、UCF101；跨越CNN和视觉转换器的多种模型架构；14种不同的后门攻击，例如Dynamic、WaNet、Lira、Issba等。



## **39. MedFuzz: Exploring the Robustness of Large Language Models in Medical Question Answering**

MedFuzz：探索医学问题解答中大型语言模型的鲁棒性 cs.CL

9 pages, 3 figures, 2 algorithms, appendix

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2406.06573v2) [paper-pdf](http://arxiv.org/pdf/2406.06573v2)

**Authors**: Robert Osazuwa Ness, Katie Matton, Hayden Helm, Sheng Zhang, Junaid Bajwa, Carey E. Priebe, Eric Horvitz

**Abstract**: Large language models (LLM) have achieved impressive performance on medical question-answering benchmarks. However, high benchmark accuracy does not imply that the performance generalizes to real-world clinical settings. Medical question-answering benchmarks rely on assumptions consistent with quantifying LLM performance but that may not hold in the open world of the clinic. Yet LLMs learn broad knowledge that can help the LLM generalize to practical conditions regardless of unrealistic assumptions in celebrated benchmarks. We seek to quantify how well LLM medical question-answering benchmark performance generalizes when benchmark assumptions are violated. Specifically, we present an adversarial method that we call MedFuzz (for medical fuzzing). MedFuzz attempts to modify benchmark questions in ways aimed at confounding the LLM. We demonstrate the approach by targeting strong assumptions about patient characteristics presented in the MedQA benchmark. Successful "attacks" modify a benchmark item in ways that would be unlikely to fool a medical expert but nonetheless "trick" the LLM into changing from a correct to an incorrect answer. Further, we present a permutation test technique that can ensure a successful attack is statistically significant. We show how to use performance on a "MedFuzzed" benchmark, as well as individual successful attacks. The methods show promise at providing insights into the ability of an LLM to operate robustly in more realistic settings.

摘要: 大型语言模型(LLM)在医学问答基准上取得了令人印象深刻的表现。然而，高基准准确率并不意味着性能适用于现实世界的临床设置。医学问题回答基准依赖于与量化LLM性能一致的假设，但这在诊所的开放世界中可能不成立。然而，LLM学习了广泛的知识，可以帮助LLM将其推广到实际情况，而不考虑著名基准中不切实际的假设。我们试图量化当违反基准假设时，LLM医疗问答基准性能的泛化程度。具体地说，我们提出了一种对抗性方法，我们称之为MedFuzz(用于医学模糊)。MedFuzz试图以混淆LLM的方式修改基准问题。我们通过针对MedQA基准中提出的关于患者特征的强烈假设来演示该方法。成功的“攻击”修改基准项目的方式不太可能愚弄医学专家，但仍然“诱骗”LLM将正确答案更改为不正确答案。此外，我们提出了一种置换测试技术，该技术可以确保成功的攻击具有统计意义。我们展示了如何使用“MedFuzze”基准测试的性能，以及个别成功的攻击。这些方法在洞察LLM在更现实的环境中稳健运行的能力方面表现出了希望。



## **40. GRACE: Graph-Regularized Attentive Convolutional Entanglement with Laplacian Smoothing for Robust DeepFake Video Detection**

GRACE：图形正规化注意卷积纠缠与拉普拉斯平滑，用于鲁棒的DeepFake视频检测 cs.CV

Submitted to TPAMI 2024

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2406.19941v3) [paper-pdf](http://arxiv.org/pdf/2406.19941v3)

**Authors**: Chih-Chung Hsu, Shao-Ning Chen, Mei-Hsuan Wu, Yi-Fang Wang, Chia-Ming Lee, Yi-Shiuan Chou

**Abstract**: As DeepFake video manipulation techniques escalate, posing profound threats, the urgent need to develop efficient detection strategies is underscored. However, one particular issue lies with facial images being mis-detected, often originating from degraded videos or adversarial attacks, leading to unexpected temporal artifacts that can undermine the efficacy of DeepFake video detection techniques. This paper introduces a novel method for robust DeepFake video detection, harnessing the power of the proposed Graph-Regularized Attentive Convolutional Entanglement (GRACE) based on the graph convolutional network with graph Laplacian to address the aforementioned challenges. First, conventional Convolution Neural Networks are deployed to perform spatiotemporal features for the entire video. Then, the spatial and temporal features are mutually entangled by constructing a graph with sparse constraint, enforcing essential features of valid face images in the noisy face sequences remaining, thus augmenting stability and performance for DeepFake video detection. Furthermore, the Graph Laplacian prior is proposed in the graph convolutional network to remove the noise pattern in the feature space to further improve the performance. Comprehensive experiments are conducted to illustrate that our proposed method delivers state-of-the-art performance in DeepFake video detection under noisy face sequences. The source code is available at https://github.com/ming053l/GRACE.

摘要: 随着DeepFake视频操纵技术的升级，构成了深刻的威胁，迫切需要开发有效的检测策略。然而，一个特别的问题是面部图像被误检，通常是由于视频降级或对手攻击，导致意外的时间伪影，这可能会破坏DeepFake视频检测技术的效率。提出了一种新的基于图拉普拉斯卷积网络的图正则化注意力卷积纠缠(GRACE)算法，用于检测DeepFake视频。首先，使用传统的卷积神经网络来执行整个视频的时空特征。然后通过构造具有稀疏约束的图将空间特征和时间特征相互纠缠在一起，在剩余的噪声人脸序列中强化有效人脸图像的本质特征，从而增强了DeepFake视频检测的稳定性和性能。此外，在图卷积网络中提出了图拉普拉斯先验，去除了特征空间中的噪声模式，进一步提高了性能。实验结果表明，本文提出的方法在含噪人脸序列下的DeepFake视频检测中具有较好的性能。源代码可在https://github.com/ming053l/GRACE.上找到



## **41. SFR-GNN: Simple and Fast Robust GNNs against Structural Attacks**

SFR-GNN：针对结构性攻击的简单快速鲁棒GNN cs.LG

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2408.16537v2) [paper-pdf](http://arxiv.org/pdf/2408.16537v2)

**Authors**: Xing Ai, Guanyu Zhu, Yulin Zhu, Yu Zheng, Gaolei Li, Jianhua Li, Kai Zhou

**Abstract**: Graph Neural Networks (GNNs) have demonstrated commendable performance for graph-structured data. Yet, GNNs are often vulnerable to adversarial structural attacks as embedding generation relies on graph topology. Existing efforts are dedicated to purifying the maliciously modified structure or applying adaptive aggregation, thereby enhancing the robustness against adversarial structural attacks. It is inevitable for a defender to consume heavy computational costs due to lacking prior knowledge about modified structures. To this end, we propose an efficient defense method, called Simple and Fast Robust Graph Neural Network (SFR-GNN), supported by mutual information theory. The SFR-GNN first pre-trains a GNN model using node attributes and then fine-tunes it over the modified graph in the manner of contrastive learning, which is free of purifying modified structures and adaptive aggregation, thus achieving great efficiency gains. Consequently, SFR-GNN exhibits a 24%--162% speedup compared to advanced robust models, demonstrating superior robustness for node classification tasks.

摘要: 图神经网络(GNN)在处理图结构数据方面表现出了值得称道的性能。然而，由于嵌入生成依赖于图拓扑，GNN经常容易受到敌意的结构攻击。现有的努力致力于净化恶意修改的结构或应用自适应聚集，从而增强对敌意结构攻击的稳健性。对于防御者来说，由于缺乏关于修改结构的先验知识，不可避免地会消耗大量的计算成本。为此，我们在互信息理论的支持下，提出了一种有效的防御方法--简单快速稳健图神经网络。SFR-GNN首先利用节点属性对GNN模型进行预训练，然后以对比学习的方式在修改后的图上对其进行微调，无需净化修改后的结构和自适应聚集，从而获得了很大的效率收益。因此，与先进的健壮模型相比，SFR-GNN表现出24%-162%的加速比，表现出对节点分类任务的卓越健壮性。



## **42. Comprehensive Botnet Detection by Mitigating Adversarial Attacks, Navigating the Subtleties of Perturbation Distances and Fortifying Predictions with Conformal Layers**

通过缓解对抗攻击、探索微扰距离的微妙之处并通过保形层加强预测来进行全面的僵尸网络检测 cs.CR

46 pages

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2409.00667v1) [paper-pdf](http://arxiv.org/pdf/2409.00667v1)

**Authors**: Rahul Yumlembam, Biju Issac, Seibu Mary Jacob, Longzhi Yang

**Abstract**: Botnets are computer networks controlled by malicious actors that present significant cybersecurity challenges. They autonomously infect, propagate, and coordinate to conduct cybercrimes, necessitating robust detection methods. This research addresses the sophisticated adversarial manipulations posed by attackers, aiming to undermine machine learning-based botnet detection systems. We introduce a flow-based detection approach, leveraging machine learning and deep learning algorithms trained on the ISCX and ISOT datasets. The detection algorithms are optimized using the Genetic Algorithm and Particle Swarm Optimization to obtain a baseline detection method. The Carlini & Wagner (C&W) attack and Generative Adversarial Network (GAN) generate deceptive data with subtle perturbations, targeting each feature used for classification while preserving their semantic and syntactic relationships, which ensures that the adversarial samples retain meaningfulness and realism. An in-depth analysis of the required L2 distance from the original sample for the malware sample to misclassify is performed across various iteration checkpoints, showing different levels of misclassification at different L2 distances of the Pertrub sample from the original sample. Our work delves into the vulnerability of various models, examining the transferability of adversarial examples from a Neural Network surrogate model to Tree-based algorithms. Subsequently, models that initially misclassified the perturbed samples are retrained, enhancing their resilience and detection capabilities. In the final phase, a conformal prediction layer is integrated, significantly rejecting incorrect predictions, of 58.20 % in the ISCX dataset and 98.94 % in the ISOT dataset.

摘要: 僵尸网络是由恶意行为者控制的计算机网络，构成了重大的网络安全挑战。它们自动感染、传播和协调进行网络犯罪，这就需要强大的检测方法。这项研究解决了攻击者复杂的对抗性操作，旨在破坏基于机器学习的僵尸网络检测系统。我们引入了一种基于流的检测方法，利用在ISCX和ISOT数据集上训练的机器学习和深度学习算法。利用遗传算法和粒子群算法对检测算法进行优化，得到一种基线检测方法。Carlini&Wagner(C&W)攻击和生成性对抗性网络(GAN)产生带有微妙扰动的欺骗性数据，针对用于分类的每个特征，同时保留它们的语义和句法关系，从而确保对抗性样本保持意义和真实性。跨各种迭代检查点执行对恶意软件样本误分类所需距离原始样本的L2距离的深入分析，显示在Pertrub样本与原始样本的不同L2距离处的不同级别的误分类。我们的工作深入研究了各种模型的脆弱性，检查了从神经网络代理模型到基于树的算法的对抗性示例的可转移性。随后，最初错误分类扰动样本的模型被重新训练，增强了它们的弹性和检测能力。在最后阶段，共形预测层被整合，显著拒绝错误预测，在ISCX数据集中为58.20%，在ISOT数据集中为98.94%。



## **43. Amplifying Training Data Exposure through Fine-Tuning with Pseudo-Labeled Memberships**

通过使用伪标签会员进行微调来扩大训练数据暴露 cs.CL

20 pages, 6 figures, 15 tables

**SubmitDate**: 2024-09-01    [abs](http://arxiv.org/abs/2402.12189v2) [paper-pdf](http://arxiv.org/pdf/2402.12189v2)

**Authors**: Myung Gyo Oh, Hong Eun Ahn, Leo Hyun Park, Taekyoung Kwon

**Abstract**: Neural language models (LMs) are vulnerable to training data extraction attacks due to data memorization. This paper introduces a novel attack scenario wherein an attacker adversarially fine-tunes pre-trained LMs to amplify the exposure of the original training data. This strategy differs from prior studies by aiming to intensify the LM's retention of its pre-training dataset. To achieve this, the attacker needs to collect generated texts that are closely aligned with the pre-training data. However, without knowledge of the actual dataset, quantifying the amount of pre-training data within generated texts is challenging. To address this, we propose the use of pseudo-labels for these generated texts, leveraging membership approximations indicated by machine-generated probabilities from the target LM. We subsequently fine-tune the LM to favor generations with higher likelihoods of originating from the pre-training data, based on their membership probabilities. Our empirical findings indicate a remarkable outcome: LMs with over 1B parameters exhibit a four to eight-fold increase in training data exposure. We discuss potential mitigations and suggest future research directions.

摘要: 神经语言模型(LMS)由于数据的记忆，容易受到训练数据提取攻击。本文介绍了一种新的攻击场景，其中攻击者对抗性地微调预先训练的最小均方根来放大原始训练数据的暴露。这一策略与以前的研究不同，目的是加强LM对其训练前数据集的保留。为了实现这一点，攻击者需要收集与预训练数据紧密一致的生成文本。然而，在没有实际数据集的情况下，量化生成文本中的预训练数据量是具有挑战性的。为了解决这个问题，我们建议对这些生成的文本使用伪标签，利用来自目标LM的机器生成的概率所指示的隶属度近似。随后，我们基于成员概率对LM进行微调，以支持来自预培训数据的可能性更高的世代。我们的经验发现表明了一个显著的结果：参数超过1B的最小二乘法显示出训练数据暴露的四到八倍的增加。我们讨论了潜在的缓解措施，并提出了未来的研究方向。



## **44. Robust off-policy Reinforcement Learning via Soft Constrained Adversary**

通过软约束适应的鲁棒性非策略强化学习 cs.LG

33 pages, 12 figures, 2 tables

**SubmitDate**: 2024-08-31    [abs](http://arxiv.org/abs/2409.00418v1) [paper-pdf](http://arxiv.org/pdf/2409.00418v1)

**Authors**: Kosuke Nakanishi, Akihiro Kubo, Yuji Yasui, Shin Ishii

**Abstract**: Recently, robust reinforcement learning (RL) methods against input observation have garnered significant attention and undergone rapid evolution due to RL's potential vulnerability. Although these advanced methods have achieved reasonable success, there have been two limitations when considering adversary in terms of long-term horizons. First, the mutual dependency between the policy and its corresponding optimal adversary limits the development of off-policy RL algorithms; although obtaining optimal adversary should depend on the current policy, this has restricted applications to off-policy RL. Second, these methods generally assume perturbations based only on the $L_p$-norm, even when prior knowledge of the perturbation distribution in the environment is available. We here introduce another perspective on adversarial RL: an f-divergence constrained problem with the prior knowledge distribution. From this, we derive two typical attacks and their corresponding robust learning frameworks. The evaluation of robustness is conducted and the results demonstrate that our proposed methods achieve excellent performance in sample-efficient off-policy RL.

摘要: 最近，针对输入观测的稳健强化学习(RL)方法受到了极大的关注，并且由于RL的潜在脆弱性而经历了快速的发展。虽然这些先进的方法已经取得了相当的成功，但从长远的角度考虑对手时，有两个局限性。首先，策略与其对应的最优对手之间的相互依赖限制了非策略RL算法的发展；虽然获得最优对手应该取决于当前的策略，但这限制了非策略RL的应用。其次，这些方法一般假定摄动仅基于$L_p$-范数，即使在已知环境中的摄动分布的情况下也是如此。我们在这里介绍了对抗性RL的另一种观点：具有先验知识分布的f-散度约束问题。在此基础上，我们得到了两种典型的攻击方法及其相应的稳健学习框架。对算法的稳健性进行了评估，结果表明，本文提出的方法在样本高效的非策略RL中取得了良好的性能。



## **45. Enhancing Transferability of Adversarial Attacks with GE-AdvGAN+: A Comprehensive Framework for Gradient Editing**

利用GE-AdvGAN+增强对抗性攻击的可转移性：梯度编辑的综合框架 cs.AI

**SubmitDate**: 2024-08-31    [abs](http://arxiv.org/abs/2408.12673v2) [paper-pdf](http://arxiv.org/pdf/2408.12673v2)

**Authors**: Zhibo Jin, Jiayu Zhang, Zhiyu Zhu, Yuchen Zhang, Jiahao Huang, Jianlong Zhou, Fang Chen

**Abstract**: Transferable adversarial attacks pose significant threats to deep neural networks, particularly in black-box scenarios where internal model information is inaccessible. Studying adversarial attack methods helps advance the performance of defense mechanisms and explore model vulnerabilities. These methods can uncover and exploit weaknesses in models, promoting the development of more robust architectures. However, current methods for transferable attacks often come with substantial computational costs, limiting their deployment and application, especially in edge computing scenarios. Adversarial generative models, such as Generative Adversarial Networks (GANs), are characterized by their ability to generate samples without the need for retraining after an initial training phase. GE-AdvGAN, a recent method for transferable adversarial attacks, is based on this principle. In this paper, we propose a novel general framework for gradient editing-based transferable attacks, named GE-AdvGAN+, which integrates nearly all mainstream attack methods to enhance transferability while significantly reducing computational resource consumption. Our experiments demonstrate the compatibility and effectiveness of our framework. Compared to the baseline AdvGAN, our best-performing method, GE-AdvGAN++, achieves an average ASR improvement of 47.8. Additionally, it surpasses the latest competing algorithm, GE-AdvGAN, with an average ASR increase of 5.9. The framework also exhibits enhanced computational efficiency, achieving 2217.7 FPS, outperforming traditional methods such as BIM and MI-FGSM. The implementation code for our GE-AdvGAN+ framework is available at https://github.com/GEAdvGANP

摘要: 可转移的敌意攻击对深度神经网络构成重大威胁，特别是在内部模型信息不可访问的黑盒场景中。研究对抗性攻击方法有助于提高防御机制的性能，探索模型漏洞。这些方法可以发现和利用模型中的弱点，从而促进更健壮的体系结构的开发。然而，当前的可转移攻击方法往往伴随着巨大的计算成本，限制了它们的部署和应用，特别是在边缘计算场景中。对抗性生成模型，如生成性对抗性网络(GANS)，其特点是能够在初始训练阶段后生成样本，而不需要重新训练。GE-AdvGAN是一种新的可转移的对抗性攻击方法，它基于这一原理。本文提出了一种新颖的基于梯度编辑的可转移攻击通用框架GE-AdvGAN+，该框架集成了几乎所有的主流攻击方法，在提高可转移性的同时显著降低了计算资源消耗。我们的实验证明了该框架的兼容性和有效性。与基准的AdvGAN相比，我们性能最好的方法GE-AdvGAN++实现了平均47.8的ASR改进。此外，它还超过了最新的竞争算法GE-AdvGAN，平均ASR提高了5.9。该框架还表现出更高的计算效率，达到2217.7 FPS，优于传统的BIM和MI-FGSM方法。我们GE-AdvGan+框架的实现代码可在https://github.com/GEAdvGANP上获得



## **46. LightPure: Realtime Adversarial Image Purification for Mobile Devices Using Diffusion Models**

LightPure：使用扩散模型的移动设备实时对抗图像净化 cs.CR

**SubmitDate**: 2024-08-31    [abs](http://arxiv.org/abs/2409.00340v1) [paper-pdf](http://arxiv.org/pdf/2409.00340v1)

**Authors**: Hossein Khalili, Seongbin Park, Vincent Li, Brandan Bright, Ali Payani, Ramana Rao Kompella, Nader Sehatbakhsh

**Abstract**: Autonomous mobile systems increasingly rely on deep neural networks for perception and decision-making. While effective, these systems are vulnerable to adversarial machine learning attacks where minor input perturbations can significantly impact outcomes. Common countermeasures involve adversarial training and/or data or network transformation. These methods, though effective, require full access to typically proprietary classifiers and are costly for large models. Recent solutions propose purification models, which add a "purification" layer before classification, eliminating the need to modify the classifier directly. Despite their effectiveness, these methods are compute-intensive, making them unsuitable for mobile systems where resources are limited and low latency is essential.   This paper introduces LightPure, a new method that enhances adversarial image purification. It improves the accuracy of existing purification methods and provides notable enhancements in speed and computational efficiency, making it suitable for mobile devices with limited resources. Our approach uses a two-step diffusion and one-shot Generative Adversarial Network (GAN) framework, prioritizing latency without compromising robustness. We propose several new techniques to achieve a reasonable balance between classification accuracy and adversarial robustness while maintaining desired latency. We design and implement a proof-of-concept on a Jetson Nano board and evaluate our method using various attack scenarios and datasets. Our results show that LightPure can outperform existing methods by up to 10x in terms of latency while achieving higher accuracy and robustness for various attack scenarios. This method offers a scalable and effective solution for real-world mobile systems.

摘要: 自主移动系统越来越依赖深度神经网络进行感知和决策。虽然有效，但这些系统容易受到对抗性机器学习攻击，在这些攻击中，微小的输入扰动可能会严重影响结果。常见对策包括对抗性训练和/或数据或网络转换。这些方法虽然有效，但需要完全访问典型的专有分类器，并且对于大型模型来说成本很高。最近的解决方案提出了净化模型，该模型在分类之前增加了一个“净化”层，消除了直接修改分类器的需要。尽管这些方法很有效，但它们都是计算密集型的，这使得它们不适合资源有限且低延迟必不可少的移动系统。本文介绍了一种新的增强对抗性图像净化的方法LightPure。它提高了现有纯化方法的准确性，并在速度和计算效率方面提供了显著的增强，使其适用于资源有限的移动设备。我们的方法使用两步扩散和一次射击生成性对抗网络(GAN)框架，在不影响健壮性的情况下优先考虑延迟。我们提出了几种新的技术，在保持期望的延迟的同时，在分类精度和对手健壮性之间取得合理的平衡。我们在Jetson Nano板上设计和实现了概念验证，并使用各种攻击场景和数据集对我们的方法进行了评估。我们的结果表明，LightPure在延迟方面可以比现有方法高出10倍，同时在各种攻击场景下实现更高的准确性和健壮性。该方法为实际移动系统提供了一种可扩展的、有效的解决方案。



## **47. Exact Recovery Guarantees for Parameterized Non-linear System Identification Problem under Adversarial Attacks**

对抗攻击下参数化非线性系统识别问题的精确恢复保证 math.OC

33 pages

**SubmitDate**: 2024-08-30    [abs](http://arxiv.org/abs/2409.00276v1) [paper-pdf](http://arxiv.org/pdf/2409.00276v1)

**Authors**: Haixiang Zhang, Baturalp Yalcin, Javad Lavaei, Eduardo Sontag

**Abstract**: In this work, we study the system identification problem for parameterized non-linear systems using basis functions under adversarial attacks. Motivated by the LASSO-type estimators, we analyze the exact recovery property of a non-smooth estimator, which is generated by solving an embedded $\ell_1$-loss minimization problem. First, we derive necessary and sufficient conditions for the well-specifiedness of the estimator and the uniqueness of global solutions to the underlying optimization problem. Next, we provide exact recovery guarantees for the estimator under two different scenarios of boundedness and Lipschitz continuity of the basis functions. The non-asymptotic exact recovery is guaranteed with high probability, even when there are more severely corrupted data than clean data. Finally, we numerically illustrate the validity of our theory. This is the first study on the sample complexity analysis of a non-smooth estimator for the non-linear system identification problem.

摘要: 在这项工作中，我们研究了对抗攻击下使用基函数的参数化非线性系统的系统识别问题。受LANSO型估计器的激励，我们分析了非光滑估计器的精确恢复性质，该估计器是通过解决嵌入的$\ell_1 $-损失最小化问题而生成的。首先，我们推导出估计量的良好指定性和基本优化问题的全局解的唯一性的充要条件。接下来，我们在基函数的有界性和Lipschitz连续性两种不同场景下为估计器提供精确的恢复保证。即使存在比干净数据更严重的损坏数据，也能以高概率保证非渐进精确恢复。最后，我们用数字说明了我们理论的有效性。这是首次对非线性系统识别问题的非光滑估计器的样本复杂性分析进行研究。



## **48. RandOhm: Mitigating Impedance Side-channel Attacks using Randomized Circuit Configurations**

RandOhm：使用随机电路查找器缓解阻抗侧通道攻击 cs.CR

**SubmitDate**: 2024-08-30    [abs](http://arxiv.org/abs/2401.08925v3) [paper-pdf](http://arxiv.org/pdf/2401.08925v3)

**Authors**: Saleh Khalaj Monfared, Domenic Forte, Shahin Tajik

**Abstract**: Physical side-channel attacks can compromise the security of integrated circuits. Most physical side-channel attacks (e.g., power or electromagnetic) exploit the dynamic behavior of a chip, typically manifesting as changes in current consumption or voltage fluctuations where algorithmic countermeasures, such as masking, can effectively mitigate them. However, as demonstrated recently, these mitigation techniques are not entirely effective against backscattered side-channel attacks such as impedance analysis. In the case of an impedance attack, an adversary exploits the data-dependent impedance variations of the chip power delivery network (PDN) to extract secret information. In this work, we introduce RandOhm, which exploits a moving target defense (MTD) strategy based on the partial reconfiguration (PR) feature of mainstream FPGAs and programmable SoCs to defend against impedance side-channel attacks. We demonstrate that the information leakage through the PDN impedance could be significantly reduced via runtime reconfiguration of the secret-sensitive parts of the circuitry. Hence, by constantly randomizing the placement and routing of the circuit, one can decorrelate the data-dependent computation from the impedance value. Moreover, in contrast to existing PR-based countermeasures, RandOhm deploys open-source bitstream manipulation tools on programmable SoCs to speed up the randomization and provide real-time protection. To validate our claims, we apply RandOhm to AES ciphers realized on 28-nm FPGAs. We analyze the resiliency of our approach by performing non-profiled and profiled impedance analysis attacks and investigate the overhead of our mitigation in terms of delay and performance.

摘要: 物理侧通道攻击可能会危及集成电路的安全性。大多数物理侧通道攻击(例如，电源或电磁)利用芯片的动态行为，通常表现为电流消耗或电压波动的变化，其中算法对策(如掩蔽)可以有效地缓解这些变化。然而，正如最近所证明的那样，这些缓解技术并不能完全有效地对抗诸如阻抗分析之类的反向散射侧信道攻击。在阻抗攻击的情况下，攻击者利用芯片功率传输网络(PDN)的依赖于数据的阻抗变化来提取秘密信息。在这项工作中，我们介绍了RandOhm，它利用了一种基于主流FPGA和可编程SoC的部分重构(PR)特性的移动目标防御(MTD)策略来防御阻抗旁通道攻击。我们证明，通过PDN阻抗的信息泄漏可以通过在运行时重新配置电路的秘密敏感部分来显著减少。因此，通过不断地随机化电路的布局和布线，可以将依赖于数据的计算与阻抗值分离。此外，与现有的基于PR的对策相比，RandOhm在可编程SoC上部署了开源的比特流处理工具，以加快随机化并提供实时保护。为了验证我们的声明，我们将RandOhm应用于在28 nm FPGA上实现的AES密码。我们通过执行非配置文件和配置文件阻抗分析攻击来分析我们方法的弹性，并从延迟和性能方面调查我们的缓解开销。



## **49. ARINC 429 Cyber-vulnerabilities and Voltage Data in a Hardware-in-the-Loop Simulator**

ARREC 429硬件在环模拟器中的网络漏洞和电压数据 cs.CR

7 pages, 3 figures. Intended for publication in IEEE Transactions on  Aerospace and Electronic Systems

**SubmitDate**: 2024-08-30    [abs](http://arxiv.org/abs/2408.16714v2) [paper-pdf](http://arxiv.org/pdf/2408.16714v2)

**Authors**: Connor Trask, Steve Movit, Justace Clutter, Rosene Clark, Mark Herrera, Kelly Tran

**Abstract**: ARINC 429 is a ubiquitous data bus for civil avionics, enabling reliable communication between devices from disparate manufacturers. However, ARINC 429 lacks any form of encryption or authentication, making it an inherently insecure communication protocol and rendering any connected avionics vulnerable to a range of attacks. We constructed a hardware-in-the-loop simulator with ARINC 429 buses, explored these vulnerabilities, and identified their potential to deny, degrade, or disrupt aircraft capabilities. We performed a denial-of-service attack against a multi-function display via a compromised ARINC 429 bus using commercially available tools, which succeeded in disabling important navigational aids. This proven attack on physical avionics illustrates the risk inherent in ARINC 429 and the need for the ability to detect these attacks. One potential mitigation is an intrusion detection system (IDS) trained on data collected from the electrical properties of the physical bus. Although previous research has demonstrated the feasibility of an IDS on an ARINC 429 bus, no IDS has been trained on data generated by avionics hardware. To facilitate this, we recorded voltage traces and message history generated by avionics and adversarial devices on the ARINC 429 bus. To the best of our knowledge, this is the first publicly available collection of hardware-generated ARINC 429 signal data.

摘要: ARINC 429是民用航空电子设备无处不在的数据总线，支持来自不同制造商的设备之间的可靠通信。然而，ARINC 429缺乏任何形式的加密或身份验证，这使得它本身就是一个不安全的通信协议，并使任何连接的航空电子设备容易受到一系列攻击。我们使用ARINC 429总线构建了一个硬件在环模拟器，探索了这些漏洞，并确定了它们拒绝、降低或破坏飞机功能的可能性。我们使用市面上可用的工具，通过一条被攻破的ARINC 429总线，对一台多功能显示器进行了拒绝服务攻击，成功地禁用了重要的导航辅助设备。对物理航空电子设备的这一已证实的攻击说明了ARINC 429固有的风险以及检测这些攻击的能力的必要性。一种可能的缓解是入侵检测系统(IDS)，该系统基于从物理总线的电气属性收集的数据进行训练。虽然以前的研究已经证明了在ARINC 429总线上使用入侵检测系统的可行性，但还没有针对航空电子硬件产生的数据对入侵检测系统进行培训。为了方便这一点，我们记录了ARINC 429总线上的航空电子设备和敌对设备产生的电压轨迹和消息历史记录。据我们所知，这是硬件生成的ARINC 429信号数据的第一个公开可用集合。



## **50. Jailbreak Attacks and Defenses Against Large Language Models: A Survey**

针对大型语言模型的越狱攻击和防御：调查 cs.CR

**SubmitDate**: 2024-08-30    [abs](http://arxiv.org/abs/2407.04295v2) [paper-pdf](http://arxiv.org/pdf/2407.04295v2)

**Authors**: Sibo Yi, Yule Liu, Zhen Sun, Tianshuo Cong, Xinlei He, Jiaxing Song, Ke Xu, Qi Li

**Abstract**: Large Language Models (LLMs) have performed exceptionally in various text-generative tasks, including question answering, translation, code completion, etc. However, the over-assistance of LLMs has raised the challenge of "jailbreaking", which induces the model to generate malicious responses against the usage policy and society by designing adversarial prompts. With the emergence of jailbreak attack methods exploiting different vulnerabilities in LLMs, the corresponding safety alignment measures are also evolving. In this paper, we propose a comprehensive and detailed taxonomy of jailbreak attack and defense methods. For instance, the attack methods are divided into black-box and white-box attacks based on the transparency of the target model. Meanwhile, we classify defense methods into prompt-level and model-level defenses. Additionally, we further subdivide these attack and defense methods into distinct sub-classes and present a coherent diagram illustrating their relationships. We also conduct an investigation into the current evaluation methods and compare them from different perspectives. Our findings aim to inspire future research and practical implementations in safeguarding LLMs against adversarial attacks. Above all, although jailbreak remains a significant concern within the community, we believe that our work enhances the understanding of this domain and provides a foundation for developing more secure LLMs.

摘要: 大型语言模型(LLMS)在问答、翻译、代码补全等文本生成任务中表现出色。然而，LLMS的过度协助带来了越狱的挑战，这导致该模型通过设计敌意提示来生成针对使用策略和社会的恶意响应。随着利用LLMS中不同漏洞的越狱攻击方法的出现，相应的安全对齐措施也在不断发展。在本文中，我们提出了一个全面和详细的分类越狱攻防方法。例如，根据目标模型的透明性，将攻击方法分为黑盒攻击和白盒攻击。同时，我们将防御方法分为提示级防御和模型级防御。此外，我们还将这些攻击和防御方法进一步细分为不同的子类，并提供了一个连贯的图来说明它们之间的关系。我们还对现有的评估方法进行了调查，并从不同的角度对它们进行了比较。我们的发现旨在启发未来在保护LLM免受对手攻击方面的研究和实际实现。最重要的是，尽管越狱在社区中仍然是一个重要的问题，但我们相信我们的工作增进了对这个领域的了解，并为开发更安全的LLM提供了基础。



