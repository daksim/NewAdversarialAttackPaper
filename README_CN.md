# Latest Adversarial Attack Papers
**update at 2024-04-16 09:27:15**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. FaceCat: Enhancing Face Recognition Security with a Unified Generative Model Framework**

FaceCat：通过统一的生成模型框架增强面部识别安全性 cs.CV

Under review

**SubmitDate**: 2024-04-14    [abs](http://arxiv.org/abs/2404.09193v1) [paper-pdf](http://arxiv.org/pdf/2404.09193v1)

**Authors**: Jiawei Chen, Xiao Yang, Yinpeng Dong, Hang Su, Jianteng Peng, Zhaoxia Yin

**Abstract**: Face anti-spoofing (FAS) and adversarial detection (FAD) have been regarded as critical technologies to ensure the safety of face recognition systems. As a consequence of their limited practicality and generalization, some existing methods aim to devise a framework capable of concurrently detecting both threats to address the challenge. Nevertheless, these methods still encounter challenges of insufficient generalization and suboptimal robustness, potentially owing to the inherent drawback of discriminative models. Motivated by the rich structural and detailed features of face generative models, we propose FaceCat which utilizes the face generative model as a pre-trained model to improve the performance of FAS and FAD. Specifically, FaceCat elaborately designs a hierarchical fusion mechanism to capture rich face semantic features of the generative model. These features then serve as a robust foundation for a lightweight head, designed to execute FAS and FAD tasks simultaneously. As relying solely on single-modality data often leads to suboptimal performance, we further propose a novel text-guided multi-modal alignment strategy that utilizes text prompts to enrich feature representation, thereby enhancing performance. For fair evaluations, we build a comprehensive protocol with a wide range of 28 attack types to benchmark the performance. Extensive experiments validate the effectiveness of FaceCat generalizes significantly better and obtains excellent robustness against input transformations.

摘要: 人脸反欺骗(FAS)和对抗检测(FAD)被认为是确保人脸识别系统安全的关键技术。由于实用性和普遍性有限，一些现有方法旨在设计一个能够同时检测这两种威胁的框架，以应对这一挑战。然而，这些方法仍然面临着泛化不足和鲁棒性不佳的挑战，这可能是由于判别模型的固有缺陷。考虑到人脸生成模型具有丰富的结构和细节特征，本文提出了一种基于人脸生成模型的FaceCat算法，该算法利用人脸生成模型作为预训练模型来提高Fas和FAD的性能。具体地说，FaceCat精心设计了一种分层融合机制来捕捉生成模型丰富的人脸语义特征。这些功能为轻量级头部奠定了坚实的基础，旨在同时执行FAS和FAD任务。由于单纯依赖单一通道数据往往导致性能不佳，我们进一步提出了一种文本引导的多通道对齐策略，该策略利用文本提示来丰富特征表示，从而提高了性能。为了公平评估，我们构建了一个包含28种攻击类型的全面协议来对性能进行基准测试。大量实验验证了FaceCat算法的有效性，其泛化能力显着提高，对输入变换具有良好的鲁棒性。



## **2. Annealing Self-Distillation Rectification Improves Adversarial Training**

自我蒸馏修正改进对抗训练 cs.LG

Accepted to ICLR 2024

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2305.12118v2) [paper-pdf](http://arxiv.org/pdf/2305.12118v2)

**Authors**: Yu-Yu Wu, Hung-Jui Wang, Shang-Tse Chen

**Abstract**: In standard adversarial training, models are optimized to fit one-hot labels within allowable adversarial perturbation budgets. However, the ignorance of underlying distribution shifts brought by perturbations causes the problem of robust overfitting. To address this issue and enhance adversarial robustness, we analyze the characteristics of robust models and identify that robust models tend to produce smoother and well-calibrated outputs. Based on the observation, we propose a simple yet effective method, Annealing Self-Distillation Rectification (ADR), which generates soft labels as a better guidance mechanism that accurately reflects the distribution shift under attack during adversarial training. By utilizing ADR, we can obtain rectified distributions that significantly improve model robustness without the need for pre-trained models or extensive extra computation. Moreover, our method facilitates seamless plug-and-play integration with other adversarial training techniques by replacing the hard labels in their objectives. We demonstrate the efficacy of ADR through extensive experiments and strong performances across datasets.

摘要: 在标准的对抗性训练中，模型经过优化，以适应允许的对抗性扰动预算内的单一热门标签。然而，对扰动带来的潜在分布漂移的忽视导致了稳健过拟合的问题。为了解决这一问题并增强对手的稳健性，我们分析了稳健模型的特征，并发现稳健模型往往会产生更平滑和校准良好的输出。在此基础上，我们提出了一种简单而有效的方法--退火法自蒸馏纠错(ADR)，该方法生成软标签作为一种更好的指导机制，准确地反映了对抗训练中攻击下的分布变化。通过利用ADR，我们可以获得显著提高模型稳健性的校正分布，而不需要预先训练的模型或大量的额外计算。此外，我们的方法通过替换目标中的硬标签，促进了与其他对抗性训练技术的无缝即插即用集成。我们通过广泛的实验和在数据集上的强劲表现证明了ADR的有效性。



## **3. IRAD: Implicit Representation-driven Image Resampling against Adversarial Attacks**

IRAD：隐式表示驱动的图像恢复对抗对抗攻击 cs.CV

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2310.11890v3) [paper-pdf](http://arxiv.org/pdf/2310.11890v3)

**Authors**: Yue Cao, Tianlin Li, Xiaofeng Cao, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: We introduce a novel approach to counter adversarial attacks, namely, image resampling. Image resampling transforms a discrete image into a new one, simulating the process of scene recapturing or rerendering as specified by a geometrical transformation. The underlying rationale behind our idea is that image resampling can alleviate the influence of adversarial perturbations while preserving essential semantic information, thereby conferring an inherent advantage in defending against adversarial attacks. To validate this concept, we present a comprehensive study on leveraging image resampling to defend against adversarial attacks. We have developed basic resampling methods that employ interpolation strategies and coordinate shifting magnitudes. Our analysis reveals that these basic methods can partially mitigate adversarial attacks. However, they come with apparent limitations: the accuracy of clean images noticeably decreases, while the improvement in accuracy on adversarial examples is not substantial. We propose implicit representation-driven image resampling (IRAD) to overcome these limitations. First, we construct an implicit continuous representation that enables us to represent any input image within a continuous coordinate space. Second, we introduce SampleNet, which automatically generates pixel-wise shifts for resampling in response to different inputs. Furthermore, we can extend our approach to the state-of-the-art diffusion-based method, accelerating it with fewer time steps while preserving its defense capability. Extensive experiments demonstrate that our method significantly enhances the adversarial robustness of diverse deep models against various attacks while maintaining high accuracy on clean images.

摘要: 我们引入了一种新的方法来对抗敌意攻击，即图像重采样。图像重采样将离散图像转换为新图像，模拟由几何变换指定的场景重新捕获或重新渲染的过程。我们的想法背后的基本原理是，图像重采样可以在保留基本语义信息的同时减轻对抗性扰动的影响，从而在防御对抗性攻击方面具有固有的优势。为了验证这一概念，我们提出了一种利用图像重采样来防御敌意攻击的综合研究。我们已经开发了使用内插策略和协调移动量的基本重采样方法。我们的分析表明，这些基本方法可以部分缓解对抗性攻击。然而，它们也有明显的局限性：清晰图像的准确性显著下降，而对抗性例子的准确性提高并不显著。我们提出了隐式表示驱动的图像重采样(IRAD)来克服这些限制。首先，我们构造了一个隐式连续表示，它使我们能够表示连续坐标空间内的任何输入图像。其次，我们介绍了SampleNet，它可以根据不同的输入自动生成像素方向的移位以进行重采样。此外，我们可以将我们的方法扩展到最先进的基于扩散的方法，以更少的时间步骤加速它，同时保持其防御能力。大量实验表明，该方法在保持对清晰图像的较高准确率的同时，显著增强了不同深度模型对各种攻击的抵抗能力。



## **4. Proof-of-Learning with Incentive Security**

具有激励保障的学习证明 cs.CR

22 pages, 6 figures

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2404.09005v1) [paper-pdf](http://arxiv.org/pdf/2404.09005v1)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks to the recent work of Jia et al. [2021], and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.

摘要: 大多数并发区块链系统严重依赖工作证明(POW)或风险证明(POS)机制来实现去中心化共识和安全保证。然而，计算密集但无意义的任务所产生的大量能源支出引起了人们对传统POW方法的相当大的担忧，POS机制虽然没有能源消耗，但受到安全和经济问题的影响。针对这些问题，有用工作证明(POUW)范式试图将具有实际意义的挑战作为POW来使用，从而使能源消耗具有有形价值。虽然先前在学习证明(Pol)方面的努力探索了利用深度学习模型训练SGD任务作为POW挑战，但最近的研究揭示了它对对手攻击的脆弱性以及在设计拜占庭安全的POL机制方面的理论难度。本文引入激励安全的概念，激励理性的证明者为了他们的最大利益而诚实地行事，绕过现有的困难，设计了一个具有计算效率、可证明的激励安全保证和可控难度的POL机制。特别是，我们的工作是安全的，可以抵抗对Jia等人最近的工作的两次攻击。[2021]并将计算开销从$\theta(1)$提高到$O(\frac{\log E}{E})$。此外，虽然最近的研究假设可信的问题提供者和验证者，但我们的设计也保证了前端激励-安全性，即使问题提供者是不可信的，并且验证者激励-安全绕过了验证者的困境。通过将ML培训融入到具有可证明保证的区块链共识机制中，我们的研究不仅为区块链系统提出了生态友好的解决方案，而且为新AI时代完全去中心化的计算能力市场提供了建议。



## **5. Stability and Generalization in Free Adversarial Training**

自由对抗训练的稳定性和概括性 cs.LG

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2404.08980v1) [paper-pdf](http://arxiv.org/pdf/2404.08980v1)

**Authors**: Xiwei Cheng, Kexin Fu, Farzan Farnia

**Abstract**: While adversarial training methods have resulted in significant improvements in the deep neural nets' robustness against norm-bounded adversarial perturbations, their generalization performance from training samples to test data has been shown to be considerably worse than standard empirical risk minimization methods. Several recent studies seek to connect the generalization behavior of adversarially trained classifiers to various gradient-based min-max optimization algorithms used for their training. In this work, we study the generalization performance of adversarial training methods using the algorithmic stability framework. Specifically, our goal is to compare the generalization performance of the vanilla adversarial training scheme fully optimizing the perturbations at every iteration vs. the free adversarial training simultaneously optimizing the norm-bounded perturbations and classifier parameters. Our proven generalization bounds indicate that the free adversarial training method could enjoy a lower generalization gap between training and test samples due to the simultaneous nature of its min-max optimization algorithm. We perform several numerical experiments to evaluate the generalization performance of vanilla, fast, and free adversarial training methods. Our empirical findings also show the improved generalization performance of the free adversarial training method and further demonstrate that the better generalization result could translate to greater robustness against black-box attack schemes. The code is available at https://github.com/Xiwei-Cheng/Stability_FreeAT.

摘要: 虽然对抗性训练方法显著提高了深度神经网络对范数有界对抗性扰动的稳健性，但从训练样本到测试数据的泛化性能已被证明比标准的经验风险最小化方法要差得多。最近的一些研究试图将对抗性训练的分类器的泛化行为与用于其训练的各种基于梯度的最小-最大优化算法联系起来。在这项工作中，我们使用算法稳定性框架来研究对抗性训练方法的泛化性能。具体地说，我们的目标是比较香草对抗训练方案和自由对抗训练方案的泛化性能，前者在每次迭代时完全优化扰动，后者同时优化范数有界扰动和分类器参数。我们证明的泛化界表明，由于其最小-最大优化算法的同时性质，自由对抗性训练方法可以享受较小的训练样本和测试样本之间的泛化差距。我们进行了一些数值实验来评估普通的、快速的和自由的对抗性训练方法的泛化性能。我们的实验结果还表明，自由对抗性训练方法的泛化性能有所改善，并进一步证明了更好的泛化结果可以转化为对黑盒攻击方案更好的鲁棒性。代码可在https://github.com/Xiwei-Cheng/Stability_FreeAT.上获得



## **6. MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers**

MixedNUTS：通过非线性混合分类器实现免训练精度-鲁棒性平衡 cs.LG

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2402.02263v3) [paper-pdf](http://arxiv.org/pdf/2402.02263v3)

**Authors**: Yatong Bai, Mo Zhou, Vishal M. Patel, Somayeh Sojoudi

**Abstract**: Adversarial robustness often comes at the cost of degraded accuracy, impeding the real-life application of robust classification models. Training-based solutions for better trade-offs are limited by incompatibilities with already-trained high-performance large models, necessitating the exploration of training-free ensemble approaches. Observing that robust models are more confident in correct predictions than in incorrect ones on clean and adversarial data alike, we speculate amplifying this "benign confidence property" can reconcile accuracy and robustness in an ensemble setting. To achieve so, we propose "MixedNUTS", a training-free method where the output logits of a robust classifier and a standard non-robust classifier are processed by nonlinear transformations with only three parameters, which are optimized through an efficient algorithm. MixedNUTS then converts the transformed logits into probabilities and mixes them as the overall output. On CIFAR-10, CIFAR-100, and ImageNet datasets, experimental results with custom strong adaptive attacks demonstrate MixedNUTS's vastly improved accuracy and near-SOTA robustness -- it boosts CIFAR-100 clean accuracy by 7.86 points, sacrificing merely 0.87 points in robust accuracy.

摘要: 对抗性的稳健性往往是以降低精度为代价的，这阻碍了稳健分类模型的实际应用。基于培训的更好权衡的解决方案受到与已经培训的高性能大型模型不兼容的限制，因此有必要探索无需培训的整体方法。我们观察到，稳健模型在正确预测中的信心比基于干净和敌对数据的不正确预测更有信心，我们推测，放大这种“良性置信度属性”可以在整体设置中调和准确性和稳健性。为了实现这一点，我们提出了一种无需训练的方法“MixedNUTS”，其中稳健分类器和标准非稳健分类器的输出逻辑通过只有三个参数的非线性变换来处理，并通过有效的算法进行优化。MixedNUTS然后将转换后的Logit转换为概率，并将它们混合为整体输出。在CIFAR-10、CIFAR-100和ImageNet数据集上，自定义强自适应攻击的实验结果表明，MixedNUTS的精确度和接近SOTA的稳健性都得到了极大的提高--它将CIFAR-100的干净精确度提高了7.86个点，而健壮精确度仅牺牲了0.87个点。



## **7. Adversarial Patterns: Building Robust Android Malware Classifiers**

对抗模式：构建稳健的Android恶意软件分类器 cs.CR

survey

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2203.02121v2) [paper-pdf](http://arxiv.org/pdf/2203.02121v2)

**Authors**: Dipkamal Bhusal, Nidhi Rastogi

**Abstract**: Machine learning models are increasingly being adopted across various fields, such as medicine, business, autonomous vehicles, and cybersecurity, to analyze vast amounts of data, detect patterns, and make predictions or recommendations. In the field of cybersecurity, these models have made significant improvements in malware detection. However, despite their ability to understand complex patterns from unstructured data, these models are susceptible to adversarial attacks that perform slight modifications in malware samples, leading to misclassification from malignant to benign. Numerous defense approaches have been proposed to either detect such adversarial attacks or improve model robustness. These approaches have resulted in a multitude of attack and defense techniques and the emergence of a field known as `adversarial machine learning.' In this survey paper, we provide a comprehensive review of adversarial machine learning in the context of Android malware classifiers. Android is the most widely used operating system globally and is an easy target for malicious agents. The paper first presents an extensive background on Android malware classifiers, followed by an examination of the latest advancements in adversarial attacks and defenses. Finally, the paper provides guidelines for designing robust malware classifiers and outlines research directions for the future.

摘要: 机器学习模型正越来越多地被应用于医疗、商业、自动驾驶汽车和网络安全等各个领域，以分析海量数据、发现模式并做出预测或建议。在网络安全领域，这些模型在恶意软件检测方面取得了重大改进。然而，尽管这些模型能够从非结构化数据中理解复杂的模式，但它们容易受到对手攻击，这些攻击对恶意软件样本进行轻微修改，导致从恶性到良性的错误分类。已经提出了许多防御方法来检测这种对抗性攻击或提高模型的稳健性。这些方法导致了大量的攻击和防御技术，并出现了一个被称为“对抗性机器学习”的领域。在这篇调查论文中，我们对Android恶意软件分类器背景下的对抗性机器学习进行了全面的回顾。Android是全球使用最广泛的操作系统，很容易成为恶意代理的攻击目标。本文首先介绍了Android恶意软件分类器的广泛背景，然后研究了对抗性攻击和防御的最新进展。最后，本文为设计健壮的恶意软件分类器提供了指导，并概述了未来的研究方向。



## **8. JailbreakLens: Visual Analysis of Jailbreak Attacks Against Large Language Models**

越狱镜头：针对大型语言模型的越狱攻击的视觉分析 cs.CR

Submitted to VIS 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08793v1) [paper-pdf](http://arxiv.org/pdf/2404.08793v1)

**Authors**: Yingchaojie Feng, Zhizhang Chen, Zhining Kang, Sijia Wang, Minfeng Zhu, Wei Zhang, Wei Chen

**Abstract**: The proliferation of large language models (LLMs) has underscored concerns regarding their security vulnerabilities, notably against jailbreak attacks, where adversaries design jailbreak prompts to circumvent safety mechanisms for potential misuse. Addressing these concerns necessitates a comprehensive analysis of jailbreak prompts to evaluate LLMs' defensive capabilities and identify potential weaknesses. However, the complexity of evaluating jailbreak performance and understanding prompt characteristics makes this analysis laborious. We collaborate with domain experts to characterize problems and propose an LLM-assisted framework to streamline the analysis process. It provides automatic jailbreak assessment to facilitate performance evaluation and support analysis of components and keywords in prompts. Based on the framework, we design JailbreakLens, a visual analysis system that enables users to explore the jailbreak performance against the target model, conduct multi-level analysis of prompt characteristics, and refine prompt instances to verify findings. Through a case study, technical evaluations, and expert interviews, we demonstrate our system's effectiveness in helping users evaluate model security and identify model weaknesses.

摘要: 大型语言模型(LLM)的激增突显了人们对其安全漏洞的担忧，特别是针对越狱攻击的担忧，在越狱攻击中，对手设计越狱提示以绕过安全机制，防止潜在的滥用。解决这些问题需要对越狱提示进行全面分析，以评估LLMS的防御能力并确定潜在的弱点。然而，评估越狱性能和理解提示特征的复杂性使得这一分析变得费力。我们与领域专家合作来表征问题，并提出一个LLM辅助的框架来简化分析过程。它提供自动越狱评估，以便于对提示中的组件和关键字进行性能评估和支持分析。基于该框架，我们设计了一个可视化分析系统JailBreakLens，使用户能够针对目标模型探索越狱性能，对提示特征进行多层次分析，并对提示实例进行精化以验证结果。通过案例研究、技术评估和专家访谈，我们展示了我们的系统在帮助用户评估模型安全性和识别模型弱点方面的有效性。



## **9. What is the Solution for State-Adversarial Multi-Agent Reinforcement Learning?**

状态对抗多智能体强化学习的解决方案是什么？ cs.AI

Accepted by Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2212.02705v5) [paper-pdf](http://arxiv.org/pdf/2212.02705v5)

**Authors**: Songyang Han, Sanbao Su, Sihong He, Shuo Han, Haizhao Yang, Shaofeng Zou, Fei Miao

**Abstract**: Various methods for Multi-Agent Reinforcement Learning (MARL) have been developed with the assumption that agents' policies are based on accurate state information. However, policies learned through Deep Reinforcement Learning (DRL) are susceptible to adversarial state perturbation attacks. In this work, we propose a State-Adversarial Markov Game (SAMG) and make the first attempt to investigate different solution concepts of MARL under state uncertainties. Our analysis shows that the commonly used solution concepts of optimal agent policy and robust Nash equilibrium do not always exist in SAMGs. To circumvent this difficulty, we consider a new solution concept called robust agent policy, where agents aim to maximize the worst-case expected state value. We prove the existence of robust agent policy for finite state and finite action SAMGs. Additionally, we propose a Robust Multi-Agent Adversarial Actor-Critic (RMA3C) algorithm to learn robust policies for MARL agents under state uncertainties. Our experiments demonstrate that our algorithm outperforms existing methods when faced with state perturbations and greatly improves the robustness of MARL policies. Our code is public on https://songyanghan.github.io/what_is_solution/.

摘要: 多智能体强化学习(MAIL)的各种方法都是在假设智能体的策略基于准确的状态信息的基础上提出的。然而，通过深度强化学习(DRL)学习的策略容易受到对抗性状态扰动攻击。在这项工作中，我们提出了一种状态-对手马尔可夫博弈(SAMG)，并首次尝试研究了状态不确定条件下Marl的不同解概念。我们的分析表明，最优代理策略和稳健纳什均衡等解的概念在SAMG中并不总是存在的。为了规避这一困难，我们考虑了一个新的解决方案概念，称为稳健代理策略，其中代理的目标是最大化最坏情况下的预期状态值。我们证明了有限状态和有限动作SAMG的鲁棒代理策略的存在性。此外，我们还提出了一种健壮的多智能体对抗行为者-批评者(RMA3C)算法来学习状态不确定条件下MAIL智能体的健壮策略。实验表明，该算法在面对状态扰动时的性能优于已有方法，并大大提高了MAIL策略的稳健性。我们的代码在https://songyanghan.github.io/what_is_solution/.上是公开的



## **10. Mayhem: Targeted Corruption of Register and Stack Variables**

混乱：寄存器和堆栈变量的有针对性的腐败 cs.CR

ACM ASIACCS 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2309.02545v2) [paper-pdf](http://arxiv.org/pdf/2309.02545v2)

**Authors**: Andrew J. Adiletta, M. Caner Tol, Yarkın Doröz, Berk Sunar

**Abstract**: In the past decade, many vulnerabilities were discovered in microarchitectures which yielded attack vectors and motivated the study of countermeasures. Further, architectural and physical imperfections in DRAMs led to the discovery of Rowhammer attacks which give an adversary power to introduce bit flips in a victim's memory space. Numerous studies analyzed Rowhammer and proposed techniques to prevent it altogether or to mitigate its effects.   In this work, we push the boundary and show how Rowhammer can be further exploited to inject faults into stack variables and even register values in a victim's process. We achieve this by targeting the register value that is stored in the process's stack, which subsequently is flushed out into the memory, where it becomes vulnerable to Rowhammer. When the faulty value is restored into the register, it will end up used in subsequent iterations. The register value can be stored in the stack via latent function calls in the source or by actively triggering signal handlers. We demonstrate the power of the findings by applying the techniques to bypass SUDO and SSH authentication. We further outline how MySQL and other cryptographic libraries can be targeted with the new attack vector. There are a number of challenges this work overcomes with extensive experimentation before coming together to yield an end-to-end attack on an OpenSSL digital signature: achieving co-location with stack and register variables, with synchronization provided via a blocking window. We show that stack and registers are no longer safe from the Rowhammer attack.

摘要: 在过去的十年中，微体系结构中发现了许多漏洞，这些漏洞产生了攻击载体，并推动了对抗措施的研究。此外，DRAM的结构和物理缺陷导致了Rowhammer攻击的发现，这种攻击使对手有能力在受害者的记忆空间中引入比特翻转。许多研究分析了Rowhammer，并提出了完全预防或减轻其影响的技术。在这项工作中，我们突破了界限，并展示了如何进一步利用Rowhammer向受害者进程中的堆栈变量甚至寄存器值注入错误。我们通过锁定存储在进程堆栈中的寄存器值来实现这一点，该寄存器值随后被刷新到内存中，在内存中变得容易受到Rowhammer的攻击。当故障值恢复到寄存器中时，它将在后续迭代中使用。寄存器值可以通过源代码中的潜在函数调用或通过主动触发信号处理程序存储在堆栈中。我们通过应用绕过SUDO和SSH身份验证的技术来演示这些发现的威力。我们进一步概述了如何利用新的攻击载体将MySQL和其他加密库作为目标。这项工作通过广泛的实验克服了许多挑战，然后结合在一起对OpenSSL数字签名进行端到端攻击：实现堆栈和寄存器变量的协同定位，并通过阻塞窗口提供同步。我们表明堆栈和寄存器不再安全，不再受到Rowhammer攻击。



## **11. On the Robustness of Language Guidance for Low-Level Vision Tasks: Findings from Depth Estimation**

低水平视觉任务语言指导的稳健性：深度估计的发现 cs.CV

Accepted to CVPR 2024. Project webpage:  https://agneetchatterjee.com/robustness_depth_lang/

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08540v1) [paper-pdf](http://arxiv.org/pdf/2404.08540v1)

**Authors**: Agneet Chatterjee, Tejas Gokhale, Chitta Baral, Yezhou Yang

**Abstract**: Recent advances in monocular depth estimation have been made by incorporating natural language as additional guidance. Although yielding impressive results, the impact of the language prior, particularly in terms of generalization and robustness, remains unexplored. In this paper, we address this gap by quantifying the impact of this prior and introduce methods to benchmark its effectiveness across various settings. We generate "low-level" sentences that convey object-centric, three-dimensional spatial relationships, incorporate them as additional language priors and evaluate their downstream impact on depth estimation. Our key finding is that current language-guided depth estimators perform optimally only with scene-level descriptions and counter-intuitively fare worse with low level descriptions. Despite leveraging additional data, these methods are not robust to directed adversarial attacks and decline in performance with an increase in distribution shift. Finally, to provide a foundation for future research, we identify points of failures and offer insights to better understand these shortcomings. With an increasing number of methods using language for depth estimation, our findings highlight the opportunities and pitfalls that require careful consideration for effective deployment in real-world settings

摘要: 单目深度估计的最新进展是通过结合自然语言作为附加指导而取得的。尽管取得了令人印象深刻的成果，但语言先验的影响，特别是在泛化和稳健性方面的影响，仍然有待探索。在本文中，我们通过量化这一先验的影响来解决这一差距，并引入各种方法来对其在各种设置下的有效性进行基准测试。我们生成传达以对象为中心的三维空间关系的“低级别”句子，将它们作为额外的语言先决条件并入其中，并评估它们对深度估计的下游影响。我们的主要发现是，目前的语言引导深度估计器仅在场景级别的描述中表现最佳，而在低级别描述中表现较差。尽管利用了更多的数据，但这些方法对定向对抗性攻击和随着分布变化增加而导致的性能下降不是很健壮。最后，为了为未来的研究提供基础，我们找出了故障点，并提供了洞察力，以更好地理解这些缺点。随着越来越多的方法使用语言进行深度估计，我们的发现突出了需要仔细考虑在现实世界环境中有效部署的机会和陷阱



## **12. VertAttack: Taking advantage of Text Classifiers' horizontal vision**

VertAttack：利用文本分类器的水平视野 cs.CL

14 pages, 4 figures, accepted to NAACL 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08538v1) [paper-pdf](http://arxiv.org/pdf/2404.08538v1)

**Authors**: Jonathan Rusert

**Abstract**: Text classification systems have continuously improved in performance over the years. However, nearly all current SOTA classifiers have a similar shortcoming, they process text in a horizontal manner. Vertically written words will not be recognized by a classifier. In contrast, humans are easily able to recognize and read words written both horizontally and vertically. Hence, a human adversary could write problematic words vertically and the meaning would still be preserved to other humans. We simulate such an attack, VertAttack. VertAttack identifies which words a classifier is reliant on and then rewrites those words vertically. We find that VertAttack is able to greatly drop the accuracy of 4 different transformer models on 5 datasets. For example, on the SST2 dataset, VertAttack is able to drop RoBERTa's accuracy from 94 to 13%. Furthermore, since VertAttack does not replace the word, meaning is easily preserved. We verify this via a human study and find that crowdworkers are able to correctly label 77% perturbed texts perturbed, compared to 81% of the original texts. We believe VertAttack offers a look into how humans might circumvent classifiers in the future and thus inspire a look into more robust algorithms.

摘要: 多年来，文本分类系统在性能上不断提高。然而，目前几乎所有的SOTA分类器都有一个类似的缺点，它们以水平的方式处理文本。分类器无法识别垂直书写的单词。相比之下，人类很容易识别和阅读水平和垂直书写的单词。因此，人类对手可以垂直书写有问题的单词，而其含义仍将保留给其他人类。我们模拟这样的攻击，VertAttack。VertAttack识别分类器依赖的单词，然后垂直重写这些单词。我们发现，VertAttack能够在5个数据集上大幅降低4种不同变压器模型的精度。例如，在Sst2数据集上，VertAttack能够将Roberta的准确率从94%降至13%。此外，由于VertAttack不会取代该词，因此含义很容易保留。我们通过一项人体研究验证了这一点，发现众筹人员能够正确地将77%的受干扰文本标记为受干扰的文本，而原始文本的这一比例为81%。我们相信，VertAttack提供了一个关于人类未来可能如何绕过分类器的展望，从而激发了对更健壮算法的研究。



## **13. Adversarially Robust Spiking Neural Networks Through Conversion**

通过转换的对抗鲁棒尖峰神经网络 cs.NE

Transactions on Machine Learning Research (TMLR), 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2311.09266v2) [paper-pdf](http://arxiv.org/pdf/2311.09266v2)

**Authors**: Ozan Özdenizci, Robert Legenstein

**Abstract**: Spiking neural networks (SNNs) provide an energy-efficient alternative to a variety of artificial neural network (ANN) based AI applications. As the progress in neuromorphic computing with SNNs expands their use in applications, the problem of adversarial robustness of SNNs becomes more pronounced. To the contrary of the widely explored end-to-end adversarial training based solutions, we address the limited progress in scalable robust SNN training methods by proposing an adversarially robust ANN-to-SNN conversion algorithm. Our method provides an efficient approach to embrace various computationally demanding robust learning objectives that have been proposed for ANNs. During a post-conversion robust finetuning phase, our method adversarially optimizes both layer-wise firing thresholds and synaptic connectivity weights of the SNN to maintain transferred robustness gains from the pre-trained ANN. We perform experimental evaluations in a novel setting proposed to rigorously assess the robustness of SNNs, where numerous adaptive adversarial attacks that account for the spike-based operation dynamics are considered. Results show that our approach yields a scalable state-of-the-art solution for adversarially robust deep SNNs with low-latency.

摘要: 尖峰神经网络(SNN)为各种基于人工神经网络(ANN)的人工智能应用提供了一种节能的替代方案。随着SNN在神经形态计算中应用范围的扩大，SNN的对抗健壮性问题变得更加突出。针对目前广泛研究的基于端到端对抗训练的解决方案，我们提出了一种对抗性稳健的神经网络到SNN的转换算法，解决了可扩展的稳健SNN训练方法的局限性。我们的方法提供了一种有效的方法来接受各种计算要求高的健壮学习目标，这些目标已经被提出给神经网络。在转换后的稳健精调阶段，我们的方法相反地优化了SNN的层级激发阈值和突触连接权重，以保持从预训练的ANN中转移的鲁棒性增益。我们在一种新的环境下进行了实验评估，该环境旨在严格评估SNN的健壮性，其中考虑了大量的自适应对手攻击，这些攻击解释了基于尖峰的操作动态。结果表明，我们的方法为具有低延迟的相反健壮的深度SNN提供了一种可扩展的最先进的解决方案。



## **14. Counterfactual Explanations for Face Forgery Detection via Adversarial Removal of Artifacts**

通过对抗性去除伪影进行人脸伪造检测的反事实解释 cs.CV

Accepted to ICME2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08341v1) [paper-pdf](http://arxiv.org/pdf/2404.08341v1)

**Authors**: Yang Li, Songlin Yang, Wei Wang, Ziwen He, Bo Peng, Jing Dong

**Abstract**: Highly realistic AI generated face forgeries known as deepfakes have raised serious social concerns. Although DNN-based face forgery detection models have achieved good performance, they are vulnerable to latest generative methods that have less forgery traces and adversarial attacks. This limitation of generalization and robustness hinders the credibility of detection results and requires more explanations. In this work, we provide counterfactual explanations for face forgery detection from an artifact removal perspective. Specifically, we first invert the forgery images into the StyleGAN latent space, and then adversarially optimize their latent representations with the discrimination supervision from the target detection model. We verify the effectiveness of the proposed explanations from two aspects: (1) Counterfactual Trace Visualization: the enhanced forgery images are useful to reveal artifacts by visually contrasting the original images and two different visualization methods; (2) Transferable Adversarial Attacks: the adversarial forgery images generated by attacking the detection model are able to mislead other detection models, implying the removed artifacts are general. Extensive experiments demonstrate that our method achieves over 90% attack success rate and superior attack transferability. Compared with naive adversarial noise methods, our method adopts both generative and discriminative model priors, and optimize the latent representations in a synthesis-by-analysis way, which forces the search of counterfactual explanations on the natural face manifold. Thus, more general counterfactual traces can be found and better adversarial attack transferability can be achieved.

摘要: 高度逼真的人工智能生成的人脸伪造被称为深度假冒，已经引起了严重的社会关注。虽然基于DNN的人脸伪造检测模型取得了很好的性能，但它们容易受到最新的生成性方法的攻击，这些方法具有较少的伪造痕迹和对抗性攻击。这种泛化和稳健性的限制阻碍了检测结果的可信度，需要更多的解释。在这项工作中，我们从伪影去除的角度为人脸伪造检测提供了反事实的解释。具体地说，我们首先将伪造图像倒置到StyleGAN潜在空间，然后在目标检测模型的判别监督下对其潜在表示进行反向优化。我们从两个方面验证了所提出的解释的有效性：(1)反事实跟踪可视化：增强的伪造图像通过视觉对比原始图像和两种不同的可视化方法来揭示伪像；(2)可转移的对抗性攻击：攻击检测模型生成的对抗性伪造图像能够误导其他检测模型，这意味着去除的伪像是通用的。大量实验表明，该方法具有90%以上的攻击成功率和良好的攻击可转移性。与朴素的对抗性噪声方法相比，该方法同时采用产生式和判别式模型先验，并通过分析综合的方式对潜在表示进行优化，迫使人们在自然人脸流形上寻找反事实的解释。因此，可以发现更一般的反事实痕迹，并且可以实现更好的对抗性攻击可转移性。



## **15. A Survey of Neural Network Robustness Assessment in Image Recognition**

图像识别中神经网络鲁棒性评估综述 cs.CV

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08285v1) [paper-pdf](http://arxiv.org/pdf/2404.08285v1)

**Authors**: Jie Wang, Jun Ai, Minyan Lu, Haoran Su, Dan Yu, Yutao Zhang, Junda Zhu, Jingyu Liu

**Abstract**: In recent years, there has been significant attention given to the robustness assessment of neural networks. Robustness plays a critical role in ensuring reliable operation of artificial intelligence (AI) systems in complex and uncertain environments. Deep learning's robustness problem is particularly significant, highlighted by the discovery of adversarial attacks on image classification models. Researchers have dedicated efforts to evaluate robustness in diverse perturbation conditions for image recognition tasks. Robustness assessment encompasses two main techniques: robustness verification/ certification for deliberate adversarial attacks and robustness testing for random data corruptions. In this survey, we present a detailed examination of both adversarial robustness (AR) and corruption robustness (CR) in neural network assessment. Analyzing current research papers and standards, we provide an extensive overview of robustness assessment in image recognition. Three essential aspects are analyzed: concepts, metrics, and assessment methods. We investigate the perturbation metrics and range representations used to measure the degree of perturbations on images, as well as the robustness metrics specifically for the robustness conditions of classification models. The strengths and limitations of the existing methods are also discussed, and some potential directions for future research are provided.

摘要: 近年来，神经网络的稳健性评估受到了极大的关注。稳健性对于确保人工智能系统在复杂和不确定环境中的可靠运行起着至关重要的作用。深度学习的稳健性问题尤其显著，突出表现在发现了对图像分类模型的敌意攻击。研究人员致力于评估图像识别任务在不同扰动条件下的稳健性。健壮性评估包括两个主要技术：针对蓄意敌对攻击的健壮性验证/认证和针对随机数据损坏的健壮性测试。在这项调查中，我们提出了在神经网络评估中的对抗稳健性(AR)和腐败稳健性(CR)的详细检查。通过分析现有的研究文献和标准，我们对图像识别中的稳健性评估进行了广泛的综述。分析了三个基本方面：概念、度量和评估方法。我们研究了用于度量图像扰动程度的扰动度量和范围表示，以及专门针对分类模型的稳健性条件的稳健性度量。文中还讨论了现有方法的优点和局限性，并对未来的研究方向进行了展望。



## **16. Struggle with Adversarial Defense? Try Diffusion**

与对抗性防御作斗争？尝试扩散 cs.CV

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08273v1) [paper-pdf](http://arxiv.org/pdf/2404.08273v1)

**Authors**: Yujie Li, Yanbin Wang, Haitao xu, Bin Liu, Jianguo Sun, Zhenhao Guo, Wenrui Ma

**Abstract**: Adversarial attacks induce misclassification by introducing subtle perturbations. Recently, diffusion models are applied to the image classifiers to improve adversarial robustness through adversarial training or by purifying adversarial noise. However, diffusion-based adversarial training often encounters convergence challenges and high computational expenses. Additionally, diffusion-based purification inevitably causes data shift and is deemed susceptible to stronger adaptive attacks. To tackle these issues, we propose the Truth Maximization Diffusion Classifier (TMDC), a generative Bayesian classifier that builds upon pre-trained diffusion models and the Bayesian theorem. Unlike data-driven classifiers, TMDC, guided by Bayesian principles, utilizes the conditional likelihood from diffusion models to determine the class probabilities of input images, thereby insulating against the influences of data shift and the limitations of adversarial training. Moreover, to enhance TMDC's resilience against more potent adversarial attacks, we propose an optimization strategy for diffusion classifiers. This strategy involves post-training the diffusion model on perturbed datasets with ground-truth labels as conditions, guiding the diffusion model to learn the data distribution and maximizing the likelihood under the ground-truth labels. The proposed method achieves state-of-the-art performance on the CIFAR10 dataset against heavy white-box attacks and strong adaptive attacks. Specifically, TMDC achieves robust accuracies of 82.81% against $l_{\infty}$ norm-bounded perturbations and 86.05% against $l_{2}$ norm-bounded perturbations, respectively, with $\epsilon=0.05$.

摘要: 对抗性攻击通过引入微妙的扰动来导致错误分类。近年来，扩散模型被应用到图像分类器中，通过对抗性训练或净化对抗性噪声来提高对抗性稳健性。然而，基于扩散的对抗性训练往往会遇到收敛挑战和较高的计算开销。此外，基于扩散的净化不可避免地会导致数据转移，并被认为容易受到更强的适应性攻击。为了解决这些问题，我们提出了真值最大化扩散分类器(TMDC)，这是一种生成式贝叶斯分类器，它建立在预先训练的扩散模型和贝叶斯定理的基础上。与数据驱动的分类器不同，TMDC在贝叶斯原理的指导下，利用扩散模型的条件似然来确定输入图像的类别概率，从而避免了数据迁移的影响和对抗性训练的限制。此外，为了增强TMDC对更强大的对手攻击的韧性，我们提出了一种扩散分类器的优化策略。该策略包括在扰动数据集上对扩散模型进行后训练，以地面真实标签为条件，引导扩散模型学习数据分布，并最大化地面真实标签下的似然。该方法在CIFAR10数据集上取得了较好的抗重白盒攻击和强自适应攻击的性能。具体地说，TMDC对$L范数有界摄动和L范数有界摄动的稳健精度分别为82.81%和86.05%，其中$epsilon=0.05$。



## **17. Combating Advanced Persistent Threats: Challenges and Solutions**

应对高级持续威胁：挑战和解决方案 cs.CR

This work has been accepted by IEEE NETWORK in April 2024. 9 pages, 5  figures, 1 table

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2309.09498v2) [paper-pdf](http://arxiv.org/pdf/2309.09498v2)

**Authors**: Yuntao Wang, Han Liu, Zhendong Li, Zhou Su, Jiliang Li

**Abstract**: The rise of advanced persistent threats (APTs) has marked a significant cybersecurity challenge, characterized by sophisticated orchestration, stealthy execution, extended persistence, and targeting valuable assets across diverse sectors. Provenance graph-based kernel-level auditing has emerged as a promising approach to enhance visibility and traceability within intricate network environments. However, it still faces challenges including reconstructing complex lateral attack chains, detecting dynamic evasion behaviors, and defending smart adversarial subgraphs. To bridge the research gap, this paper proposes an efficient and robust APT defense scheme leveraging provenance graphs, including a network-level distributed audit model for cost-effective lateral attack reconstruction, a trust-oriented APT evasion behavior detection strategy, and a hidden Markov model based adversarial subgraph defense approach. Through prototype implementation and extensive experiments, we validate the effectiveness of our system. Lastly, crucial open research directions are outlined in this emerging field.

摘要: 高级持续性威胁(APT)的兴起标志着一个重大的网络安全挑战，其特征是复杂的协调、隐蔽的执行、延长的持久性以及针对不同行业的宝贵资产。基于起源图的内核级审计已经成为在复杂的网络环境中增强可见性和可跟踪性的一种有前途的方法。然而，它仍然面临着重构复杂的侧向攻击链、检测动态规避行为和防御智能对抗性子图等挑战。为了弥补这一研究空白，本文提出了一种利用起源图的高效、健壮的APT防御方案，包括用于高性价比的横向攻击重构的网络级分布式审计模型、面向信任的APT逃避行为检测策略以及基于隐马尔可夫模型的敌意子图防御方法。通过原型实现和广泛的实验，验证了系统的有效性。最后，概述了这一新兴领域的关键开放研究方向。



## **18. Practical Region-level Attack against Segment Anything Models**

针对Segment Anything模型的实用区域级攻击 cs.CV

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08255v1) [paper-pdf](http://arxiv.org/pdf/2404.08255v1)

**Authors**: Yifan Shen, Zhengyuan Li, Gang Wang

**Abstract**: Segment Anything Models (SAM) have made significant advancements in image segmentation, allowing users to segment target portions of an image with a single click (i.e., user prompt). Given its broad applications, the robustness of SAM against adversarial attacks is a critical concern. While recent works have explored adversarial attacks against a pre-defined prompt/click, their threat model is not yet realistic: (1) they often assume the user-click position is known to the attacker (point-based attack), and (2) they often operate under a white-box setting with limited transferability. In this paper, we propose a more practical region-level attack where attackers do not need to know the precise user prompt. The attack remains effective as the user clicks on any point on the target object in the image, hiding the object from SAM. Also, by adapting a spectrum transformation method, we make the attack more transferable under a black-box setting. Both control experiments and testing against real-world SAM services confirm its effectiveness.

摘要: 分割任何模型(SAM)在图像分割方面取得了重大进展，允许用户通过一次点击(即用户提示)来分割图像的目标部分。考虑到其广泛的应用，SAM对对手攻击的稳健性是一个关键问题。虽然最近的研究已经探索了针对预定义提示/点击的对抗性攻击，但他们的威胁模型还不现实：(1)它们通常假设攻击者知道用户点击的位置(基于点的攻击)，以及(2)它们通常在可转移性有限的白盒设置下操作。在本文中，我们提出了一种更实用的区域级攻击，攻击者不需要知道准确的用户提示。当用户点击图像中目标对象上的任何点时，该攻击仍然有效，从而隐藏该对象以躲避SAM。此外，通过采用频谱变换的方法，使得攻击在黑盒环境下更具可转移性。对照实验和针对真实世界SAM服务的测试都证实了该方法的有效性。



## **19. Navigating Quantum Security Risks in Networked Environments: A Comprehensive Study of Quantum-Safe Network Protocols**

应对网络环境中的量子安全风险：量子安全网络协议的全面研究 cs.CR

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2404.08232v1) [paper-pdf](http://arxiv.org/pdf/2404.08232v1)

**Authors**: Yaser Baseri, Vikas Chouhan, Abdelhakim Hafid

**Abstract**: The emergence of quantum computing poses a formidable security challenge to network protocols traditionally safeguarded by classical cryptographic algorithms. This paper provides an exhaustive analysis of vulnerabilities introduced by quantum computing in a diverse array of widely utilized security protocols across the layers of the TCP/IP model, including TLS, IPsec, SSH, PGP, and more. Our investigation focuses on precisely identifying vulnerabilities susceptible to exploitation by quantum adversaries at various migration stages for each protocol while also assessing the associated risks and consequences for secure communication. We delve deep into the impact of quantum computing on each protocol, emphasizing potential threats posed by quantum attacks and scrutinizing the effectiveness of post-quantum cryptographic solutions. Through carefully evaluating vulnerabilities and risks that network protocols face in the post-quantum era, this study provides invaluable insights to guide the development of appropriate countermeasures. Our findings contribute to a broader comprehension of quantum computing's influence on network security and offer practical guidance for protocol designers, implementers, and policymakers in addressing the challenges stemming from the advancement of quantum computing. This comprehensive study is a crucial step toward fortifying the security of networked environments in the quantum age.

摘要: 量子计算的出现对传统上由经典密码算法保护的网络协议提出了严峻的安全挑战。本文详尽分析了量子计算在各种广泛使用的安全协议(包括TLS、IPSec、SSH、PGP等)的TCP/IP模型各层中引入的漏洞。我们的调查重点是准确地识别在每个协议的不同迁移阶段容易被量子攻击者利用的漏洞，同时还评估了相关的风险和安全通信的后果。我们深入研究了量子计算对每个协议的影响，强调了量子攻击带来的潜在威胁，并仔细审查了后量子密码解决方案的有效性。通过仔细评估后量子时代网络协议面临的漏洞和风险，本研究为指导制定适当的对策提供了宝贵的见解。我们的发现有助于更广泛地理解量子计算对网络安全的影响，并为协议设计者、实施者和政策制定者提供实用指导，以应对量子计算进步带来的挑战。这项全面的研究是在量子时代加强网络环境安全的关键一步。



## **20. CosalPure: Learning Concept from Group Images for Robust Co-Saliency Detection**

CosalPure：基于组图像的学习概念，用于鲁棒共显著性检测 cs.CV

This paper is accepted by CVPR 2024

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2403.18554v2) [paper-pdf](http://arxiv.org/pdf/2403.18554v2)

**Authors**: Jiayi Zhu, Qing Guo, Felix Juefei-Xu, Yihao Huang, Yang Liu, Geguang Pu

**Abstract**: Co-salient object detection (CoSOD) aims to identify the common and salient (usually in the foreground) regions across a given group of images. Although achieving significant progress, state-of-the-art CoSODs could be easily affected by some adversarial perturbations, leading to substantial accuracy reduction. The adversarial perturbations can mislead CoSODs but do not change the high-level semantic information (e.g., concept) of the co-salient objects. In this paper, we propose a novel robustness enhancement framework by first learning the concept of the co-salient objects based on the input group images and then leveraging this concept to purify adversarial perturbations, which are subsequently fed to CoSODs for robustness enhancement. Specifically, we propose CosalPure containing two modules, i.e., group-image concept learning and concept-guided diffusion purification. For the first module, we adopt a pre-trained text-to-image diffusion model to learn the concept of co-salient objects within group images where the learned concept is robust to adversarial examples. For the second module, we map the adversarial image to the latent space and then perform diffusion generation by embedding the learned concept into the noise prediction function as an extra condition. Our method can effectively alleviate the influence of the SOTA adversarial attack containing different adversarial patterns, including exposure and noise. The extensive results demonstrate that our method could enhance the robustness of CoSODs significantly.

摘要: 共显著目标检测(CoSOD)的目的是识别给定图像组中的共同和显著(通常在前景中)区域。虽然取得了重大进展，但最先进的CoSOD很容易受到一些对抗性扰动的影响，导致精度大幅下降。对抗性扰动会误导CoSOD，但不会改变共显著对象的高级语义信息(例如，概念)。在本文中，我们提出了一种新的稳健性增强框架，该框架首先学习基于输入分组图像的共显著对象的概念，然后利用该概念来净化对抗性扰动，然后将这些扰动馈送到CoSOD以增强稳健性。具体地说，我们提出CosalPure包含两个模块，即组图像概念学习和概念引导的扩散净化。对于第一个模块，我们采用预先训练的文本到图像的扩散模型来学习组图像中共显著对象的概念，其中学习的概念对对抗性例子是健壮的。对于第二个模块，我们将敌意图像映射到潜在空间，然后通过将学习到的概念作为附加条件嵌入到噪声预测函数中来执行扩散生成。我们的方法可以有效地缓解SOTA对抗性攻击的影响，该攻击包含不同的对抗性模式，包括暴露和噪声。实验结果表明，该方法可以显著提高CoSOD的稳健性。



## **21. Systematically Assessing the Security Risks of AI/ML-enabled Connected Healthcare Systems**

系统评估支持人工智能/ML的互联医疗保健系统的安全风险 cs.CR

13 pages, 5 figures, 3 tables

**SubmitDate**: 2024-04-12    [abs](http://arxiv.org/abs/2401.17136v2) [paper-pdf](http://arxiv.org/pdf/2401.17136v2)

**Authors**: Mohammed Elnawawy, Mohammadreza Hallajiyan, Gargi Mitra, Shahrear Iqbal, Karthik Pattabiraman

**Abstract**: The adoption of machine-learning-enabled systems in the healthcare domain is on the rise. While the use of ML in healthcare has several benefits, it also expands the threat surface of medical systems. We show that the use of ML in medical systems, particularly connected systems that involve interfacing the ML engine with multiple peripheral devices, has security risks that might cause life-threatening damage to a patient's health in case of adversarial interventions. These new risks arise due to security vulnerabilities in the peripheral devices and communication channels. We present a case study where we demonstrate an attack on an ML-enabled blood glucose monitoring system by introducing adversarial data points during inference. We show that an adversary can achieve this by exploiting a known vulnerability in the Bluetooth communication channel connecting the glucose meter with the ML-enabled app. We further show that state-of-the-art risk assessment techniques are not adequate for identifying and assessing these new risks. Our study highlights the need for novel risk analysis methods for analyzing the security of AI-enabled connected health devices.

摘要: 在医疗保健领域，采用机器学习系统的情况正在上升。虽然ML在医疗保健中的使用有几个好处，但它也扩大了医疗系统的威胁表面。我们表明，在医疗系统中使用ML，特别是涉及将ML引擎与多个外围设备接口的互联系统，具有安全风险，如果进行对抗性干预，可能会对患者的健康造成危及生命的损害。这些新的风险是由于外围设备和通信通道中的安全漏洞造成的。我们提供了一个案例研究，通过在推理过程中引入敌对数据点来演示对启用ML的血糖监测系统的攻击。我们表明，攻击者可以通过利用连接血糖仪和支持ML的应用程序的蓝牙通信通道中的已知漏洞来实现这一点。我们进一步表明，最先进的风险评估技术不足以识别和评估这些新风险。我们的研究突显了需要新的风险分析方法来分析支持人工智能的联网医疗设备的安全性。



## **22. Eliminating Catastrophic Overfitting Via Abnormal Adversarial Examples Regularization**

通过异常对抗示例规范化消除灾难性过度匹配 cs.LG

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.08154v1) [paper-pdf](http://arxiv.org/pdf/2404.08154v1)

**Authors**: Runqi Lin, Chaojian Yu, Tongliang Liu

**Abstract**: Single-step adversarial training (SSAT) has demonstrated the potential to achieve both efficiency and robustness. However, SSAT suffers from catastrophic overfitting (CO), a phenomenon that leads to a severely distorted classifier, making it vulnerable to multi-step adversarial attacks. In this work, we observe that some adversarial examples generated on the SSAT-trained network exhibit anomalous behaviour, that is, although these training samples are generated by the inner maximization process, their associated loss decreases instead, which we named abnormal adversarial examples (AAEs). Upon further analysis, we discover a close relationship between AAEs and classifier distortion, as both the number and outputs of AAEs undergo a significant variation with the onset of CO. Given this observation, we re-examine the SSAT process and uncover that before the occurrence of CO, the classifier already displayed a slight distortion, indicated by the presence of few AAEs. Furthermore, the classifier directly optimizing these AAEs will accelerate its distortion, and correspondingly, the variation of AAEs will sharply increase as a result. In such a vicious circle, the classifier rapidly becomes highly distorted and manifests as CO within a few iterations. These observations motivate us to eliminate CO by hindering the generation of AAEs. Specifically, we design a novel method, termed Abnormal Adversarial Examples Regularization (AAER), which explicitly regularizes the variation of AAEs to hinder the classifier from becoming distorted. Extensive experiments demonstrate that our method can effectively eliminate CO and further boost adversarial robustness with negligible additional computational overhead.

摘要: 单步对抗训练(SSAT)已经证明了实现效率和稳健性的潜力。然而，SSAT存在灾难性过匹配(CO)，这一现象导致分类器严重失真，使其容易受到多步骤对抗性攻击。在这项工作中，我们观察到在SSAT训练的网络上产生的一些对抗性样本表现出异常行为，即这些训练样本虽然是由内部最大化过程产生的，但其关联损失反而减少，我们称之为异常对抗性样本(AAES)。通过进一步的分析，我们发现AAEs与分类器失真之间有密切的关系，因为AAEs的数量和输出都随着CO的开始而发生显著的变化。鉴于这一观察，我们重新检查SSAT过程并发现，在CO发生之前，分类器已经显示出轻微的失真，这表明存在很少的AAE。而且，直接对这些AAEs进行优化的分类器会加速AAEs的失真，相应地，AAEs的变化量也会急剧增加。在这样的恶性循环中，分类器迅速变得高度失真，并在几次迭代内表现为CO。这些观察结果促使我们通过阻碍AAEs的产生来消除CO。具体地说，我们设计了一种新的方法，称为异常对抗实例正则化(AAER)，它显式地规则化AAE的变化，以防止分类器变得失真。大量的实验表明，该方法可以有效地消除CO，并在几乎不增加计算开销的情况下进一步提高对手攻击的健壮性。



## **23. Chapter: Vulnerability of Quantum Information Systems to Collective Manipulation**

章：量子信息系统对集体操纵的脆弱性 quant-ph

This is an earlier version of a final document that appears in  IntechOpen. Comments welcome to neiljohnson@gwu.edu, 20 Pages

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/1901.08873v2) [paper-pdf](http://arxiv.org/pdf/1901.08873v2)

**Authors**: Fernando J. Gómez-Ruiz, Ferney J. Rodríguez, Luis Quiroga, Neil F. Johnson

**Abstract**: The highly specialist terms `quantum computing' and `quantum information', together with the broader term `quantum technologies', now appear regularly in the mainstream media. While this is undoubtedly highly exciting for physicists and investors alike, a key question for society concerns such systems' vulnerabilities -- and in particular, their vulnerability to collective manipulation. Here we present and discuss a new form of vulnerability in such systems, that we have identified based on detailed many-body quantum mechanical calculations. The impact of this new vulnerability is that groups of adversaries can maximally disrupt these systems' global quantum state which will then jeopardize their quantum functionality. It will be almost impossible to detect these attacks since they do not change the Hamiltonian and the purity remains the same; they do not entail any real-time communication between the attackers; and they can last less than a second. We also argue that there can be an implicit amplification of such attacks because of the statistical character of modern non-state actor groups. A countermeasure could be to embed future quantum technologies within redundant classical networks. We purposely structure the discussion in this chapter so that the first sections are self-contained and can be read by non-specialists.

摘要: 具有高度专业性的术语‘量子计算’和‘量子信息’，以及更广泛的术语‘量子技术’，现在经常出现在主流媒体上。虽然这对物理学家和投资者来说无疑都是非常令人兴奋的，但社会的一个关键问题是关于这类系统的脆弱性--特别是它们对集体操纵的脆弱性。在这里，我们提出并讨论了这类系统中的一种新形式的脆弱性，这是我们基于详细的多体量子力学计算确定的。这一新漏洞的影响是，一组敌手可以最大限度地破坏这些系统的全局量子状态，这将危及它们的量子功能。几乎不可能检测到这些攻击，因为它们不改变哈密顿量，纯度保持不变；它们不需要攻击者之间的任何实时通信；并且它们可以持续不到一秒。我们还认为，由于现代非国家行为者群体的统计特征，此类攻击可能存在隐性放大。一种对策可能是在冗余的经典网络中嵌入未来的量子技术。我们特意安排了本章的讨论结构，以便第一节是独立的，非专业人士可以阅读。



## **24. Fooling Contrastive Language-Image Pre-trained Models with CLIPMasterPrints**

使用CLIPMasterPrint愚弄对比图像预训练模型 cs.CV

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2307.03798v2) [paper-pdf](http://arxiv.org/pdf/2307.03798v2)

**Authors**: Matthias Freiberger, Peter Kun, Christian Igel, Anders Sundnes Løvlie, Sebastian Risi

**Abstract**: Models leveraging both visual and textual data such as Contrastive Language-Image Pre-training (CLIP), are the backbone of many recent advances in artificial intelligence. In this work, we show that despite their versatility, such models are vulnerable to what we refer to as fooling master images. Fooling master images are capable of maximizing the confidence score of a CLIP model for a significant number of widely varying prompts, while being either unrecognizable or unrelated to the attacked prompts for humans. The existence of such images is problematic as it could be used by bad actors to maliciously interfere with CLIP-trained image retrieval models in production with comparably small effort as a single image can attack many different prompts. We demonstrate how fooling master images for CLIP (CLIPMasterPrints) can be mined using stochastic gradient descent, projected gradient descent, or blackbox optimization. Contrary to many common adversarial attacks, the blackbox optimization approach allows us to mine CLIPMasterPrints even when the weights of the model are not accessible. We investigate the properties of the mined images, and find that images trained on a small number of image captions generalize to a much larger number of semantically related captions. We evaluate possible mitigation strategies, where we increase the robustness of the model and introduce an approach to automatically detect CLIPMasterPrints to sanitize the input of vulnerable models. Finally, we find that vulnerability to CLIPMasterPrints is related to a modality gap in contrastive pre-trained multi-modal networks. Code available at https://github.com/matfrei/CLIPMasterPrints.

摘要: 利用视觉和文本数据的模型，如对比语言-图像预训练(CLIP)，是人工智能许多最新进展的支柱。在这项工作中，我们表明，尽管这些模型具有多功能性，但它们很容易受到我们所说的愚弄主图像的攻击。愚弄主图像能够针对大量差异很大的提示最大化剪辑模型的置信度分数，同时对人类来说要么无法识别，要么与被攻击的提示无关。这种图像的存在是有问题的，因为它可能被不良行为者用来恶意干扰生产中经过剪辑训练的图像检索模型，而工作量相对较小，因为一张图像可以攻击许多不同的提示。我们演示了如何使用随机梯度下降、投影梯度下降或黑盒优化来挖掘CLIP(CLIPMasterPrints)的愚弄主图像。与许多常见的对抗性攻击相反，黑盒优化方法允许我们在模型的权重不可访问的情况下挖掘CLIPMasterPrint。我们研究了挖掘出的图像的属性，发现在少量图像字幕上训练的图像概括为大量语义相关的字幕。我们评估了可能的缓解策略，其中我们增加了模型的健壮性，并引入了一种自动检测CLIPMasterPrints的方法来清理易受攻击的模型的输入。最后，我们发现CLIPMasterPrints的漏洞与对比预训练多通道网络中的通道缺口有关。代码可在https://github.com/matfrei/CLIPMasterPrints.上找到



## **25. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

AmpleGCG：学习一个通用的、可转移的对抗后缀生成模型，用于越狱既开放式又封闭式LLM cs.CL

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07921v1) [paper-pdf](http://arxiv.org/pdf/2404.07921v1)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.

摘要: 随着大型语言模型(LLM)变得越来越普遍并集成到自治系统中，确保它们的安全性是当务之急。尽管在安全对齐方面取得了长足的进步，但最近的工作GCG~\Citep{zou2023Universal}提出了一种离散令牌优化算法，并选择损失最小的单个后缀来成功越狱对齐LLM。在这项工作中，我们首先讨论了在GCG优化越狱过程中只选择损失最小的后缀的缺点，并在中间步骤中发现了遗漏的成功后缀。此外，我们利用这些成功的后缀作为训练数据来学习一种名为AmpleGCG的生成模型，该模型捕获给定有害查询的对抗性后缀的分布，并在几秒钟内为任何有害查询快速生成数百个后缀。AmpleGCG在两个对齐的LLM(Llama-2-7B-Chat和Vicuna-7B)上达到了近100%的攻击成功率(ASR)，超过了两个最强的攻击基线。更有趣的是，AmpleGCG还无缝传输以攻击不同的型号，包括闭源LLMS，在最新的GPT-3.5上实现了99\%的ASR。总之，我们的工作通过训练对抗性后缀的生成模型来放大GCG的影响，该模型对任何有害的查询都是通用的，并且可以从攻击开源LLM转移到闭源LLM。此外，它可以在短短4秒内为一个有害的查询生成200个敌意后缀，使其更具挑战性。



## **26. A Measurement of Genuine Tor Traces for Realistic Website Fingerprinting**

真实网站指纹识别的真实Tor痕迹测量 cs.CR

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07892v1) [paper-pdf](http://arxiv.org/pdf/2404.07892v1)

**Authors**: Rob Jansen, Ryan Wails, Aaron Johnson

**Abstract**: Website fingerprinting (WF) is a dangerous attack on web privacy because it enables an adversary to predict the website a user is visiting, despite the use of encryption, VPNs, or anonymizing networks such as Tor. Previous WF work almost exclusively uses synthetic datasets to evaluate the performance and estimate the feasibility of WF attacks despite evidence that synthetic data misrepresents the real world. In this paper we present GTT23, the first WF dataset of genuine Tor traces, which we obtain through a large-scale measurement of the Tor network. GTT23 represents real Tor user behavior better than any existing WF dataset, is larger than any existing WF dataset by at least an order of magnitude, and will help ground the future study of realistic WF attacks and defenses. In a detailed evaluation, we survey 25 WF datasets published over the last 15 years and compare their characteristics to those of GTT23. We discover common deficiencies of synthetic datasets that make them inferior to GTT23 for drawing meaningful conclusions about the effectiveness of WF attacks directed at real Tor users. We have made GTT23 available to promote reproducible research and to help inspire new directions for future work.

摘要: 网站指纹识别(WF)是对网络隐私的一种危险攻击，因为它使对手能够预测用户正在访问的网站，尽管使用了加密、VPN或匿名网络(如ToR)。以前的WF工作几乎完全使用合成数据集来评估WF攻击的性能和估计WF攻击的可行性，尽管有证据表明合成数据歪曲了真实世界。本文介绍了GTT23，这是我们通过对Tor网络的大规模测量而获得的第一个真实Tor痕迹的WF数据集。GTT23比任何现有的WF数据集更好地表示真实的ToR用户行为，比任何现有的WF数据集至少大一个数量级，并将有助于未来对现实WF攻击和防御的研究。在详细的评估中，我们调查了过去15年发布的25个WF数据集，并将它们的特征与GTT23的特征进行了比较。我们发现了合成数据集的共同缺陷，使其在针对真实Tor用户的WF攻击的有效性方面不如GTT23得出有意义的结论。我们已经提供了GTT23，以促进可重复的研究，并帮助启发未来工作的新方向。



## **27. Multi-Robot Target Tracking with Sensing and Communication Danger Zones**

多机器人目标跟踪的传感和通信危险区 cs.RO

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07880v1) [paper-pdf](http://arxiv.org/pdf/2404.07880v1)

**Authors**: Jiazhen Li, Peihan Li, Yuwei Wu, Gaurav S. Sukhatme, Vijay Kumar, Lifeng Zhou

**Abstract**: Multi-robot target tracking finds extensive applications in different scenarios, such as environmental surveillance and wildfire management, which require the robustness of the practical deployment of multi-robot systems in uncertain and dangerous environments. Traditional approaches often focus on the performance of tracking accuracy with no modeling and assumption of the environments, neglecting potential environmental hazards which result in system failures in real-world deployments. To address this challenge, we investigate multi-robot target tracking in the adversarial environment considering sensing and communication attacks with uncertainty. We design specific strategies to avoid different danger zones and proposed a multi-agent tracking framework under the perilous environment. We approximate the probabilistic constraints and formulate practical optimization strategies to address computational challenges efficiently. We evaluate the performance of our proposed methods in simulations to demonstrate the ability of robots to adjust their risk-aware behaviors under different levels of environmental uncertainty and risk confidence. The proposed method is further validated via real-world robot experiments where a team of drones successfully track dynamic ground robots while being risk-aware of the sensing and/or communication danger zones.

摘要: 多机器人目标跟踪在环境监测、野火管理等不同场景中有着广泛的应用，这就要求多机器人系统在不确定和危险环境中的实际部署具有很强的鲁棒性。传统的方法往往只关注跟踪精度的性能，没有对环境进行建模和假设，而忽略了实际部署中可能导致系统故障的环境危害。为了应对这一挑战，我们研究了在具有不确定性的感知和通信攻击的对抗性环境中的多机器人目标跟踪。设计了避开不同危险区域的具体策略，提出了危险环境下的多智能体跟踪框架。我们对概率约束进行近似，并制定实用的优化策略来有效地应对计算挑战。我们在仿真中评估了我们提出的方法的性能，以展示机器人在不同的环境不确定性和风险置信度下调整其风险意识行为的能力。通过真实世界的机器人实验进一步验证了所提出的方法，其中一组无人机成功地跟踪了动态的地面机器人，同时意识到了传感和/或通信危险区域的风险。



## **28. LeapFrog: The Rowhammer Instruction Skip Attack**

LeapFrog：Rowhammer指令跳过攻击 cs.CR

Accepted at Hardware.io 2024

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07878v1) [paper-pdf](http://arxiv.org/pdf/2404.07878v1)

**Authors**: Andrew Adiletta, Caner Tol, Berk Sunar

**Abstract**: Since its inception, Rowhammer exploits have rapidly evolved into increasingly sophisticated threats not only compromising data integrity but also the control flow integrity of victim processes. Nevertheless, it remains a challenge for an attacker to identify vulnerable targets (i.e., Rowhammer gadgets), understand the outcome of the attempted fault, and formulate an attack that yields useful results.   In this paper, we present a new type of Rowhammer gadget, called a LeapFrog gadget, which, when present in the victim code, allows an adversary to subvert code execution to bypass a critical piece of code (e.g., authentication check logic, encryption rounds, padding in security protocols). The Leapfrog gadget manifests when the victim code stores the Program Counter (PC) value in the user or kernel stack (e.g., a return address during a function call) which, when tampered with, re-positions the return address to a location that bypasses a security-critical code pattern.   This research also presents a systematic process to identify Leapfrog gadgets. This methodology enables the automated detection of susceptible targets and the determination of optimal attack parameters. We first showcase this new attack vector through a practical demonstration on a TLS handshake client/server scenario, successfully inducing an instruction skip in a client application. We then demonstrate the attack on real-world code found in the wild, implementing an attack on OpenSSL.   Our findings extend the impact of Rowhammer attacks on control flow and contribute to the development of more robust defenses against these increasingly sophisticated threats.

摘要: 自成立以来，Rowhammer漏洞攻击已迅速演变为日益复杂的威胁，不仅危及数据完整性，还危及受害者进程的控制流完整性。然而，对于攻击者来说，识别易受攻击的目标(即Rowhammer小工具)、了解尝试的故障的结果并制定产生有用结果的攻击仍然是一项挑战。在本文中，我们提出了一种新的Rowhammer小工具，称为LeapFrog小工具，当它存在于受害者代码中时，允许攻击者破坏代码执行以绕过关键代码段(例如，身份验证逻辑、加密轮、安全协议中的填充)。当受害者代码在用户或内核堆栈中存储程序计数器(PC)值(例如，函数调用期间的返回地址)时，当被篡改时，将返回地址重新定位到绕过安全关键代码模式的位置时，LeapFrog小工具就会显现出来。这项研究还提出了一个识别LeapFrog小工具的系统过程。这种方法能够自动检测易受影响的目标并确定最佳攻击参数。我们首先通过TLS握手客户端/服务器场景的实际演示展示了这种新的攻击载体，成功地在客户端应用程序中诱导了指令跳过。然后，我们演示了对在野外发现的真实代码的攻击，实现了对OpenSSL的攻击。我们的发现扩大了Rowhammer攻击对控制流的影响，并有助于开发针对这些日益复杂的威胁的更强大的防御措施。



## **29. Pilot Spoofing Attack on the Downlink of Cell-Free Massive MIMO: From the Perspective of Adversaries**

无小区大规模MIMO下行链路导频欺骗攻击：基于对手的视角 cs.IT

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2403.04435v2) [paper-pdf](http://arxiv.org/pdf/2403.04435v2)

**Authors**: Weiyang Xu, Ruiguang Wang, Yuan Zhang, Hien Quoc Ngo, Wei Xiang

**Abstract**: The channel hardening effect is less pronounced in the cell-free massive multiple-input multiple-output (mMIMO) system compared to its cellular counterpart, making it necessary to estimate the downlink effective channel gains to ensure decent performance. However, the downlink training inadvertently creates an opportunity for adversarial nodes to launch pilot spoofing attacks (PSAs). First, we demonstrate that adversarial distributed access points (APs) can severely degrade the achievable downlink rate. They achieve this by estimating their channels to users in the uplink training phase and then precoding and sending the same pilot sequences as those used by legitimate APs during the downlink training phase. Then, the impact of the downlink PSA is investigated by rigorously deriving a closed-form expression of the per-user achievable downlink rate. By employing the min-max criterion to optimize the power allocation coefficients, the maximum per-user achievable rate of downlink transmission is minimized from the perspective of adversarial APs. As an alternative to the downlink PSA, adversarial APs may opt to precode random interference during the downlink data transmission phase in order to disrupt legitimate communications. In this scenario, the achievable downlink rate is derived, and then power optimization algorithms are also developed. We present numerical results to showcase the detrimental impact of the downlink PSA and compare the effects of these two types of attacks.

摘要: 与蜂窝系统相比，无小区大规模多输入多输出(MMIMO)系统中的信道硬化效应不那么明显，因此有必要估计下行链路的有效信道增益以确保良好的性能。然而，下行训练无意中为敌对节点创造了发起试点欺骗攻击(PSA)的机会。首先，我们证明了敌意分布式接入点(AP)会严重降低可实现的下行链路速率。它们通过在上行链路训练阶段估计其对用户的信道，然后预编码并发送与合法AP在下行链路训练阶段使用的导频序列相同的导频序列来实现这一点。然后，通过严格推导每个用户可实现的下行链路速率的闭合形式表达式来研究下行链路PSA的影响。通过使用最小-最大准则来优化功率分配系数，从对抗性AP的角度最小化每用户可实现的最大下行传输速率。作为下行链路PSA的替代方案，敌意AP可以选择在下行链路数据传输阶段对随机干扰进行预编码，以便中断合法通信。在这种情况下，推导了可实现的下行链路速率，并开发了功率优化算法。我们给出了数值结果来展示下行PSA的有害影响，并比较了这两种类型的攻击的影响。



## **30. Poisoning Prevention in Federated Learning and Differential Privacy via Stateful Proofs of Execution**

基于执行状态证明的联邦学习和差分隐私中毒预防 cs.CR

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.06721v2) [paper-pdf](http://arxiv.org/pdf/2404.06721v2)

**Authors**: Norrathep Rattanavipanon, Ivan De Oliveira Nunes

**Abstract**: The rise in IoT-driven distributed data analytics, coupled with increasing privacy concerns, has led to a demand for effective privacy-preserving and federated data collection/model training mechanisms. In response, approaches such as Federated Learning (FL) and Local Differential Privacy (LDP) have been proposed and attracted much attention over the past few years. However, they still share the common limitation of being vulnerable to poisoning attacks wherein adversaries compromising edge devices feed forged (a.k.a. poisoned) data to aggregation back-ends, undermining the integrity of FL/LDP results.   In this work, we propose a system-level approach to remedy this issue based on a novel security notion of Proofs of Stateful Execution (PoSX) for IoT/embedded devices' software. To realize the PoSX concept, we design SLAPP: a System-Level Approach for Poisoning Prevention. SLAPP leverages commodity security features of embedded devices - in particular ARM TrustZoneM security extensions - to verifiably bind raw sensed data to their correct usage as part of FL/LDP edge device routines. As a consequence, it offers robust security guarantees against poisoning. Our evaluation, based on real-world prototypes featuring multiple cryptographic primitives and data collection schemes, showcases SLAPP's security and low overhead.

摘要: 物联网驱动的分布式数据分析的兴起，加上对隐私的日益担忧，导致了对有效的隐私保护和联合数据收集/模型培训机制的需求。在过去的几年里，联邦学习(FL)和局部差异隐私(LDP)等方法被提出并引起了人们的广泛关注。然而，它们仍然有一个共同的局限性，即容易受到中毒攻击，在这些攻击中，危害边缘设备的对手提供伪造的(也称为。有毒)数据到聚合后端，破坏FL/LDP结果的完整性。在这项工作中，我们提出了一种基于物联网/嵌入式设备软件状态执行证明(PoSX)的新的安全概念来解决这一问题。为了实现PoSX的概念，我们设计了SLAPP：一种系统级的中毒预防方法。SLAPP利用嵌入式设备的商用安全功能--尤其是ARM TrustZoneM安全扩展--作为FL/LDP边缘设备例程的一部分，以可验证的方式将原始感测数据与其正确使用绑定在一起。因此，它为防止中毒提供了强有力的安全保障。我们的评估基于具有多个加密原语和数据收集方案的真实世界原型，展示了SLAPP的安全性和低开销。



## **31. Enhancing Network Intrusion Detection Performance using Generative Adversarial Networks**

利用生成对抗网络提高网络入侵检测性能 cs.CR

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07464v1) [paper-pdf](http://arxiv.org/pdf/2404.07464v1)

**Authors**: Xinxing Zhao, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: Network intrusion detection systems (NIDS) play a pivotal role in safeguarding critical digital infrastructures against cyber threats. Machine learning-based detection models applied in NIDS are prevalent today. However, the effectiveness of these machine learning-based models is often limited by the evolving and sophisticated nature of intrusion techniques as well as the lack of diverse and updated training samples. In this research, a novel approach for enhancing the performance of an NIDS through the integration of Generative Adversarial Networks (GANs) is proposed. By harnessing the power of GANs in generating synthetic network traffic data that closely mimics real-world network behavior, we address a key challenge associated with NIDS training datasets, which is the data scarcity. Three distinct GAN models (Vanilla GAN, Wasserstein GAN and Conditional Tabular GAN) are implemented in this work to generate authentic network traffic patterns specifically tailored to represent the anomalous activity. We demonstrate how this synthetic data resampling technique can significantly improve the performance of the NIDS model for detecting such activity. By conducting comprehensive experiments using the CIC-IDS2017 benchmark dataset, augmented with GAN-generated data, we offer empirical evidence that shows the effectiveness of our proposed approach. Our findings show that the integration of GANs into NIDS can lead to enhancements in intrusion detection performance for attacks with limited training data, making it a promising avenue for bolstering the cybersecurity posture of organizations in an increasingly interconnected and vulnerable digital landscape.

摘要: 网络入侵检测系统在保护关键数字基础设施免受网络威胁方面发挥着举足轻重的作用。目前，基于机器学习的检测模型在网络入侵检测系统中的应用非常普遍。然而，这些基于机器学习的模型的有效性往往受到入侵技术不断发展和复杂的性质以及缺乏多样化和更新的训练样本的限制。在这项研究中，提出了一种通过整合生成性对抗网络(GANS)来提高网络入侵检测系统性能的新方法。通过利用Gans生成接近模拟真实网络行为的合成网络流量数据的能力，我们解决了与NIDS训练数据集相关的一个关键挑战，即数据稀缺性。本文实现了三种不同的GAN模型(Vanilla GAN、Wasserstein GAN和Conditional Tablular GAN)来生成真实的网络流量模式，该模式专门用于表示异常活动。我们演示了这种合成数据重采样技术如何显著提高NIDS模型检测此类活动的性能。通过使用CIC-IDS2017基准数据集和GaN生成的数据进行全面的实验，我们提供了经验证据，证明了我们所提出的方法的有效性。我们的研究结果表明，将GANS集成到网络入侵检测系统中可以增强对训练数据有限的攻击的入侵检测性能，使其成为在日益互联和脆弱的数字环境中支持组织网络安全态势的一种有前途的途径。



## **32. Privacy preserving layer partitioning for Deep Neural Network models**

深度神经网络模型的隐私保护层划分 cs.CR

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.07437v1) [paper-pdf](http://arxiv.org/pdf/2404.07437v1)

**Authors**: Kishore Rajasekar, Randolph Loh, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: MLaaS (Machine Learning as a Service) has become popular in the cloud computing domain, allowing users to leverage cloud resources for running private inference of ML models on their data. However, ensuring user input privacy and secure inference execution is essential. One of the approaches to protect data privacy and integrity is to use Trusted Execution Environments (TEEs) by enabling execution of programs in secure hardware enclave. Using TEEs can introduce significant performance overhead due to the additional layers of encryption, decryption, security and integrity checks. This can lead to slower inference times compared to running on unprotected hardware. In our work, we enhance the runtime performance of ML models by introducing layer partitioning technique and offloading computations to GPU. The technique comprises two distinct partitions: one executed within the TEE, and the other carried out using a GPU accelerator. Layer partitioning exposes intermediate feature maps in the clear which can lead to reconstruction attacks to recover the input. We conduct experiments to demonstrate the effectiveness of our approach in protecting against input reconstruction attacks developed using trained conditional Generative Adversarial Network(c-GAN). The evaluation is performed on widely used models such as VGG-16, ResNet-50, and EfficientNetB0, using two datasets: ImageNet for Image classification and TON IoT dataset for cybersecurity attack detection.

摘要: MLaaS(机器学习即服务)在云计算领域变得流行起来，允许用户利用云资源对其数据运行ML模型的私有推理。然而，确保用户输入隐私和安全推理执行是必不可少的。保护数据隐私和完整性的方法之一是使用可信执行环境(TEE)，通过在安全硬件飞地中执行程序来实现。由于加密、解密、安全和完整性检查的附加层，使用TES可能会带来显著的性能开销。与在不受保护的硬件上运行相比，这可能会导致较慢的推断时间。在我们的工作中，我们通过引入层划分技术和将计算卸载到GPU来提高ML模型的运行时性能。该技术包括两个不同的分区：一个在TEE内执行，另一个使用GPU加速器执行。层划分将中间特征映射暴露在明文中，这可能导致重建攻击以恢复输入。我们进行了实验，以证明我们的方法在防止输入重构攻击方面的有效性，该攻击是使用训练的条件生成对抗网络(c-GAN)开发的。在VGG-16、ResNet-50和EfficientNetB0等广泛使用的模型上进行了评估，使用了两个数据集：用于图像分类的ImageNet和用于网络安全攻击检测的Ton IoT数据集。



## **33. Multi-granular Adversarial Attacks against Black-box Neural Ranking Models**

黑盒神经排序模型的多粒度对抗攻击 cs.IR

Accepted by SIGIR2024

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2404.01574v2) [paper-pdf](http://arxiv.org/pdf/2404.01574v2)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Adversarial ranking attacks have gained increasing attention due to their success in probing vulnerabilities, and, hence, enhancing the robustness, of neural ranking models. Conventional attack methods employ perturbations at a single granularity, e.g., word or sentence level, to target documents. However, limiting perturbations to a single level of granularity may reduce the flexibility of adversarial examples, thereby diminishing the potential threat of the attack. Therefore, we focus on generating high-quality adversarial examples by incorporating multi-granular perturbations. Achieving this objective involves tackling a combinatorial explosion problem, which requires identifying an optimal combination of perturbations across all possible levels of granularity, positions, and textual pieces. To address this challenge, we transform the multi-granular adversarial attack into a sequential decision-making process, where perturbations in the next attack step build on the perturbed document in the current attack step. Since the attack process can only access the final state without direct intermediate signals, we use reinforcement learning to perform multi-granular attacks. During the reinforcement learning process, two agents work cooperatively to identify multi-granular vulnerabilities as attack targets and organize perturbation candidates into a final perturbation sequence. Experimental results show that our attack method surpasses prevailing baselines in both attack effectiveness and imperceptibility.

摘要: 对抗性排序攻击因其在探测漏洞方面的成功，从而增强了神经排序模型的稳健性而受到越来越多的关注。传统的攻击方法在单个粒度(例如，单词或句子级别)上使用扰动以文档为目标。然而，将扰动限制在单一的粒度级别可能会降低对抗性示例的灵活性，从而降低攻击的潜在威胁。因此，我们专注于通过结合多粒度扰动来生成高质量的对抗性实例。实现这一目标需要处理组合爆炸问题，这需要确定跨所有可能级别的粒度、位置和文本片段的扰动的最佳组合。为了应对这一挑战，我们将多粒度的对抗性攻击转化为一个顺序的决策过程，其中下一攻击步骤中的扰动建立在当前攻击步骤中被扰动的文档之上。由于攻击过程只能访问最终状态，没有直接的中间信号，因此我们使用强化学习来执行多粒度攻击。在强化学习过程中，两个代理协作识别多粒度漏洞作为攻击目标，并将扰动候选组织成最终的扰动序列。实验结果表明，我们的攻击方法在攻击有效性和不可感知性方面都超过了主流基线。



## **34. Incremental Randomized Smoothing Certification**

增量随机平滑认证 cs.LG

ICLR 2024

**SubmitDate**: 2024-04-11    [abs](http://arxiv.org/abs/2305.19521v2) [paper-pdf](http://arxiv.org/pdf/2305.19521v2)

**Authors**: Shubham Ugare, Tarun Suresh, Debangshu Banerjee, Gagandeep Singh, Sasa Misailovic

**Abstract**: Randomized smoothing-based certification is an effective approach for obtaining robustness certificates of deep neural networks (DNNs) against adversarial attacks. This method constructs a smoothed DNN model and certifies its robustness through statistical sampling, but it is computationally expensive, especially when certifying with a large number of samples. Furthermore, when the smoothed model is modified (e.g., quantized or pruned), certification guarantees may not hold for the modified DNN, and recertifying from scratch can be prohibitively expensive.   We present the first approach for incremental robustness certification for randomized smoothing, IRS. We show how to reuse the certification guarantees for the original smoothed model to certify an approximated model with very few samples. IRS significantly reduces the computational cost of certifying modified DNNs while maintaining strong robustness guarantees. We experimentally demonstrate the effectiveness of our approach, showing up to 3x certification speedup over the certification that applies randomized smoothing of the approximate model from scratch.

摘要: 基于随机平滑的认证是获得深层神经网络抗攻击健壮性证书的有效方法。该方法构造了一个平滑的DNN模型，并通过统计抽样验证了其稳健性，但其计算量很大，特别是在需要大量样本的情况下。此外，当修改平滑的模型(例如，量化或修剪)时，认证保证可能不适用于修改的DNN，并且从头开始重新认证可能昂贵得令人望而却步。我们提出了第一种随机平滑的增量式稳健性证明方法--IRS。我们展示了如何重用对原始平滑模型的证明保证来证明具有很少样本的近似模型。IRS在保持较强的健壮性保证的同时，显著降低了证明修改的DNN的计算代价。我们在实验中展示了我们方法的有效性，与从头开始应用随机平滑近似模型的认证相比，认证加速高达3倍。



## **35. Indoor Location Fingerprinting Privacy: A Comprehensive Survey**

室内位置指纹隐私：综合调查 cs.CR

Submitted to ACM Computing Surveys

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.07345v1) [paper-pdf](http://arxiv.org/pdf/2404.07345v1)

**Authors**: Amir Fathalizadeh, Vahideh Moghtadaiee, Mina Alishahi

**Abstract**: The pervasive integration of Indoor Positioning Systems (IPS) arises from the limitations of Global Navigation Satellite Systems (GNSS) in indoor environments, leading to the widespread adoption of Location-Based Services (LBS). Specifically, indoor location fingerprinting employs diverse signal fingerprints from user devices, enabling precise location identification by Location Service Providers (LSP). Despite its broad applications across various domains, indoor location fingerprinting introduces a notable privacy risk, as both LSP and potential adversaries inherently have access to this sensitive information, compromising users' privacy. Consequently, concerns regarding privacy vulnerabilities in this context necessitate a focused exploration of privacy-preserving mechanisms. In response to these concerns, this survey presents a comprehensive review of Privacy-Preserving Mechanisms in Indoor Location Fingerprinting (ILFPPM) based on cryptographic, anonymization, differential privacy (DP), and federated learning (FL) techniques. We also propose a distinctive and novel grouping of privacy vulnerabilities, adversary and attack models, and available evaluation metrics specific to indoor location fingerprinting systems. Given the identified limitations and research gaps in this survey, we highlight numerous prospective opportunities for future investigation, aiming to motivate researchers interested in advancing this field. This survey serves as a valuable reference for researchers and provides a clear overview for those beyond this specific research domain.

摘要: 室内定位系统(IPS)的广泛集成源于全球导航卫星系统(GNSS)在室内环境中的局限性，导致基于位置的服务(LBS)的广泛采用。具体地说，室内位置指纹识别使用来自用户设备的不同信号指纹，从而实现位置服务提供商(LSP)的精确位置识别。尽管其广泛应用于各个领域，但室内位置指纹识别带来了显著的隐私风险，因为LSP和潜在的对手天生都可以访问这些敏感信息，从而危及用户的隐私。因此，在这种情况下，对隐私漏洞的担忧需要集中探索隐私保护机制。针对这些问题，本文基于密码学、匿名化、差分隐私(DP)和联合学习(FL)等技术，对室内位置指纹识别(ILFPPM)中的隐私保护机制进行了全面的综述。我们还提出了一种独特而新颖的隐私漏洞、对手和攻击模型的分组，以及针对室内位置指纹系统的可用评估指标。鉴于这项调查中确定的局限性和研究差距，我们强调了未来研究的许多预期机会，旨在激励有兴趣推进这一领域的研究人员。这项调查为研究人员提供了有价值的参考，并为这一特定研究领域以外的人提供了一个明确的概述。



## **36. Towards a Game-theoretic Understanding of Explanation-based Membership Inference Attacks**

基于博弈论的成员推断攻击的博弈理解 cs.AI

arXiv admin note: text overlap with arXiv:2202.02659

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.07139v1) [paper-pdf](http://arxiv.org/pdf/2404.07139v1)

**Authors**: Kavita Kumari, Murtuza Jadliwala, Sumit Kumar Jha, Anindya Maiti

**Abstract**: Model explanations improve the transparency of black-box machine learning (ML) models and their decisions; however, they can also be exploited to carry out privacy threats such as membership inference attacks (MIA). Existing works have only analyzed MIA in a single "what if" interaction scenario between an adversary and the target ML model; thus, it does not discern the factors impacting the capabilities of an adversary in launching MIA in repeated interaction settings. Additionally, these works rely on assumptions about the adversary's knowledge of the target model's structure and, thus, do not guarantee the optimality of the predefined threshold required to distinguish the members from non-members. In this paper, we delve into the domain of explanation-based threshold attacks, where the adversary endeavors to carry out MIA attacks by leveraging the variance of explanations through iterative interactions with the system comprising of the target ML model and its corresponding explanation method. We model such interactions by employing a continuous-time stochastic signaling game framework. In our framework, an adversary plays a stopping game, interacting with the system (having imperfect information about the type of an adversary, i.e., honest or malicious) to obtain explanation variance information and computing an optimal threshold to determine the membership of a datapoint accurately. First, we propose a sound mathematical formulation to prove that such an optimal threshold exists, which can be used to launch MIA. Then, we characterize the conditions under which a unique Markov perfect equilibrium (or steady state) exists in this dynamic system. By means of a comprehensive set of simulations of the proposed game model, we assess different factors that can impact the capability of an adversary to launch MIA in such repeated interaction settings.

摘要: 模型解释提高了黑盒机器学习(ML)模型及其决策的透明度；然而，它们也可以被利用来实施隐私威胁，如成员推理攻击(MIA)。已有的研究只分析了敌方和目标ML模型之间的单一假设交互场景中的MIA，没有发现影响敌方在重复交互环境下发起MIA的能力的因素。此外，这些工作依赖于关于对手对目标模型结构的知识的假设，因此，不能保证区分成员和非成员所需的预定义阈值的最佳性。在本文中，我们深入研究了基于解释的门限攻击领域，即攻击者通过与目标ML模型及其相应解释方法组成的系统的迭代交互，利用解释的差异来努力实施MIA攻击。我们采用一个连续时间随机信号博弈框架对这种相互作用进行建模。在我们的框架中，对手进行停止博弈，与系统交互(拥有关于对手类型的不完善信息，即诚实或恶意)以获得解释差异信息，并计算最优阈值以准确确定数据点的成员资格。首先，我们提出了一个合理的数学公式来证明这样一个最优门限的存在，该最优门限可用于启动MIA。然后，我们刻画了该动力系统存在唯一的马尔可夫完全平衡(或稳态)的条件。通过对所提出的博弈模型的全面模拟，我们评估了在这种重复交互环境中影响对手发起MIA的能力的不同因素。



## **37. Adversarial purification for no-reference image-quality metrics: applicability study and new methods**

无参考图像质量度量的对抗纯化：适用性研究和新方法 cs.CV

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.06957v1) [paper-pdf](http://arxiv.org/pdf/2404.06957v1)

**Authors**: Aleksandr Gushchin, Anna Chistyakova, Vladislav Minashkin, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: Recently, the area of adversarial attacks on image quality metrics has begun to be explored, whereas the area of defences remains under-researched. In this study, we aim to cover that case and check the transferability of adversarial purification defences from image classifiers to IQA methods. In this paper, we apply several widespread attacks on IQA models and examine the success of the defences against them. The purification methodologies covered different preprocessing techniques, including geometrical transformations, compression, denoising, and modern neural network-based methods. Also, we address the challenge of assessing the efficacy of a defensive methodology by proposing ways to estimate output visual quality and the success of neutralizing attacks. Defences were tested against attack on three IQA metrics -- Linearity, MetaIQA and SPAQ. The code for attacks and defences is available at: (link is hidden for a blind review).

摘要: 最近，对图像质量指标的对抗性攻击领域已经开始探索，而防御领域的研究仍然不足。在这项研究中，我们的目标是涵盖这种情况，并检查对抗纯化防御从图像分类器到IQA方法的可转移性。在本文中，我们应用了几个广泛的攻击IQA模型和检查的成功防御他们。纯化方法涵盖了不同的预处理技术，包括几何变换、压缩、去噪和基于神经网络的现代方法。此外，我们解决了评估防御方法的有效性的挑战，提出了估计输出视觉质量和中和攻击的成功的方法。针对三个IQA指标-线性、MetaIQA和SPAQ-的攻击进行了防御测试。攻击和防御的代码可在：（隐藏链接以供盲目审查）。



## **38. Simpler becomes Harder: Do LLMs Exhibit a Coherent Behavior on Simplified Corpora?**

简单变得更难：LLM在简化语料库上表现出一致性行为吗？ cs.CL

Published at DeTermIt! Workshop at LREC-COLING 2024

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.06838v1) [paper-pdf](http://arxiv.org/pdf/2404.06838v1)

**Authors**: Miriam Anschütz, Edoardo Mosca, Georg Groh

**Abstract**: Text simplification seeks to improve readability while retaining the original content and meaning. Our study investigates whether pre-trained classifiers also maintain such coherence by comparing their predictions on both original and simplified inputs. We conduct experiments using 11 pre-trained models, including BERT and OpenAI's GPT 3.5, across six datasets spanning three languages. Additionally, we conduct a detailed analysis of the correlation between prediction change rates and simplification types/strengths. Our findings reveal alarming inconsistencies across all languages and models. If not promptly addressed, simplified inputs can be easily exploited to craft zero-iteration model-agnostic adversarial attacks with success rates of up to 50%

摘要: 文本简化旨在提高可读性，同时保留原始内容和含义。我们的研究通过比较原始输入和简化输入的预测来研究预训练分类器是否也保持这种一致性。我们使用11个预训练模型进行了实验，包括BERT和OpenAI的GPT 3.5，跨越三种语言的六个数据集。此外，我们还对预测变化率和简化类型/强度之间的相关性进行了详细的分析。我们的发现揭示了所有语言和模型之间的惊人不一致性。如果没有及时解决，简化的输入很容易被用来制造零迭代模型不可知的对抗攻击，成功率高达50%。



## **39. Logit Calibration and Feature Contrast for Robust Federated Learning on Non-IID Data**

非IID数据鲁棒联邦学习的Logit校正和特征对比 cs.LG

**SubmitDate**: 2024-04-10    [abs](http://arxiv.org/abs/2404.06776v1) [paper-pdf](http://arxiv.org/pdf/2404.06776v1)

**Authors**: Yu Qiao, Chaoning Zhang, Apurba Adhikary, Choong Seon Hong

**Abstract**: Federated learning (FL) is a privacy-preserving distributed framework for collaborative model training on devices in edge networks. However, challenges arise due to vulnerability to adversarial examples (AEs) and the non-independent and identically distributed (non-IID) nature of data distribution among devices, hindering the deployment of adversarially robust and accurate learning models at the edge. While adversarial training (AT) is commonly acknowledged as an effective defense strategy against adversarial attacks in centralized training, we shed light on the adverse effects of directly applying AT in FL that can severely compromise accuracy, especially in non-IID challenges. Given this limitation, this paper proposes FatCC, which incorporates local logit \underline{C}alibration and global feature \underline{C}ontrast into the vanilla federated adversarial training (\underline{FAT}) process from both logit and feature perspectives. This approach can effectively enhance the federated system's robust accuracy (RA) and clean accuracy (CA). First, we propose logit calibration, where the logits are calibrated during local adversarial updates, thereby improving adversarial robustness. Second, FatCC introduces feature contrast, which involves a global alignment term that aligns each local representation with unbiased global features, thus further enhancing robustness and accuracy in federated adversarial environments. Extensive experiments across multiple datasets demonstrate that FatCC achieves comparable or superior performance gains in both CA and RA compared to other baselines.

摘要: 联合学习(FL)是一种保护隐私的分布式框架，用于边缘网络中设备上的协作模型训练。然而，由于易受对抗性示例(AE)的攻击，以及设备之间数据分布的非独立和相同分布(Non-IID)的性质，出现了挑战，阻碍了在边缘部署对抗性的健壮和准确的学习模型。虽然对抗训练(AT)被公认为是集中训练中对抗攻击的一种有效防御策略，但我们揭示了在外语教学中直接应用AT的不利影响，它会严重影响准确性，特别是在非IID挑战中。针对这一局限性，本文提出了FatCC，它从Logit和特征两个角度将局部Logit\Underline{C}校准和全局特征\Underline{C}对比引入到普通的联合对手训练(\Underline{FAT})过程中。该方法可以有效地提高联邦系统的鲁棒精度(RA)和清洁精度(CA)。首先，我们提出了LOGIT校准，其中LOGIT在本地对抗性更新期间被校准，从而提高了对抗性健壮性。其次，FatCC引入了特征对比度，它涉及一个全局对齐项，将每个局部表示与无偏的全局特征对齐，从而进一步增强了联合对抗环境中的稳健性和准确性。跨多个数据集的广泛实验表明，与其他基准相比，FatCC在CA和RA方面都获得了可比或更好的性能提升。



## **40. Towards Building a Robust Toxicity Predictor**

建立稳健的毒性预测器 cs.CL

ACL 2023 /

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.08690v1) [paper-pdf](http://arxiv.org/pdf/2404.08690v1)

**Authors**: Dmitriy Bespalov, Sourav Bhabesh, Yi Xiang, Liutong Zhou, Yanjun Qi

**Abstract**: Recent NLP literature pays little attention to the robustness of toxicity language predictors, while these systems are most likely to be used in adversarial contexts. This paper presents a novel adversarial attack, \texttt{ToxicTrap}, introducing small word-level perturbations to fool SOTA text classifiers to predict toxic text samples as benign. ToxicTrap exploits greedy based search strategies to enable fast and effective generation of toxic adversarial examples. Two novel goal function designs allow ToxicTrap to identify weaknesses in both multiclass and multilabel toxic language detectors. Our empirical results show that SOTA toxicity text classifiers are indeed vulnerable to the proposed attacks, attaining over 98\% attack success rates in multilabel cases. We also show how a vanilla adversarial training and its improved version can help increase robustness of a toxicity detector even against unseen attacks.

摘要: 最近的NLP文献很少关注毒性语言预测器的稳健性，而这些系统最有可能用于对抗性的环境中。本文提出了一种新型的对抗攻击\textttt {ToxicTrap}，引入小的词级扰动来欺骗SOTA文本分类器将有毒文本样本预测为良性。ToxicTrap利用基于贪婪的搜索策略来快速有效地生成有毒的对抗示例。两种新颖的目标函数设计使ToxicTrap能够识别多类和多标签有毒语言检测器的弱点。我们的经验结果表明，SOTA毒性文本分类器确实容易受到拟议的攻击，在多标签情况下攻击成功率超过98%。我们还展示了普通对抗训练及其改进版本如何帮助提高毒性检测器的鲁棒性，即使是针对不可见的攻击。



## **41. False Claims against Model Ownership Resolution**

针对模型所有权决议的虚假声明 cs.CR

13pages,3 figures. To appear in the 33rd USENIX Security Symposium  (USENIX Security '24)

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2304.06607v7) [paper-pdf](http://arxiv.org/pdf/2304.06607v7)

**Authors**: Jian Liu, Rui Zhang, Sebastian Szyller, Kui Ren, N. Asokan

**Abstract**: Deep neural network (DNN) models are valuable intellectual property of model owners, constituting a competitive advantage. Therefore, it is crucial to develop techniques to protect against model theft. Model ownership resolution (MOR) is a class of techniques that can deter model theft. A MOR scheme enables an accuser to assert an ownership claim for a suspect model by presenting evidence, such as a watermark or fingerprint, to show that the suspect model was stolen or derived from a source model owned by the accuser. Most of the existing MOR schemes prioritize robustness against malicious suspects, ensuring that the accuser will win if the suspect model is indeed a stolen model.   In this paper, we show that common MOR schemes in the literature are vulnerable to a different, equally important but insufficiently explored, robustness concern: a malicious accuser. We show how malicious accusers can successfully make false claims against independent suspect models that were not stolen. Our core idea is that a malicious accuser can deviate (without detection) from the specified MOR process by finding (transferable) adversarial examples that successfully serve as evidence against independent suspect models. To this end, we first generalize the procedures of common MOR schemes and show that, under this generalization, defending against false claims is as challenging as preventing (transferable) adversarial examples. Via systematic empirical evaluation, we show that our false claim attacks always succeed in the MOR schemes that follow our generalization, including in a real-world model: Amazon's Rekognition API.

摘要: 深度神经网络(DNN)模型是模型所有者宝贵的知识产权，构成了竞争优势。因此，开发防止模型盗窃的技术至关重要。模型所有权解析(MOR)是一类能够阻止模型被盗的技术。MOR方案使原告能够通过出示证据(例如水印或指纹)来断言可疑模型的所有权主张，以显示可疑模型是被盗的或从原告拥有的源模型派生的。现有的大多数MOR方案都优先考虑针对恶意嫌疑人的健壮性，确保在可疑模型确实是被盗模型的情况下原告获胜。在这篇文章中，我们证明了文献中常见的MOR方案容易受到另一个同样重要但未被充分研究的健壮性问题的影响：恶意指控者。我们展示了恶意原告如何成功地对未被窃取的独立可疑模型做出虚假声明。我们的核心思想是，恶意指控者可以通过找到(可转移的)对抗性例子来偏离指定的MOR过程(而不被检测到)，这些例子成功地充当了针对独立嫌疑人模型的证据。为此，我们首先推广了常见MOR方案的步骤，并证明在这种推广下，对虚假声明的防御与防止(可转移)对抗性例子一样具有挑战性。通过系统的实证评估，我们表明我们的虚假声明攻击在遵循我们的推广的MOR方案中总是成功的，包括在真实世界的模型中：亚马逊的Rekognition API。



## **42. Sandwich attack: Multi-language Mixture Adaptive Attack on LLMs**

三明治攻击：对LLM的多语言混合自适应攻击 cs.CR

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.07242v1) [paper-pdf](http://arxiv.org/pdf/2404.07242v1)

**Authors**: Bibek Upadhayay, Vahid Behzadan

**Abstract**: Large Language Models (LLMs) are increasingly being developed and applied, but their widespread use faces challenges. These include aligning LLMs' responses with human values to prevent harmful outputs, which is addressed through safety training methods. Even so, bad actors and malicious users have succeeded in attempts to manipulate the LLMs to generate misaligned responses for harmful questions such as methods to create a bomb in school labs, recipes for harmful drugs, and ways to evade privacy rights. Another challenge is the multilingual capabilities of LLMs, which enable the model to understand and respond in multiple languages. Consequently, attackers exploit the unbalanced pre-training datasets of LLMs in different languages and the comparatively lower model performance in low-resource languages than high-resource ones. As a result, attackers use a low-resource languages to intentionally manipulate the model to create harmful responses. Many of the similar attack vectors have been patched by model providers, making the LLMs more robust against language-based manipulation. In this paper, we introduce a new black-box attack vector called the \emph{Sandwich attack}: a multi-language mixture attack, which manipulates state-of-the-art LLMs into generating harmful and misaligned responses. Our experiments with five different models, namely Google's Bard, Gemini Pro, LLaMA-2-70-B-Chat, GPT-3.5-Turbo, GPT-4, and Claude-3-OPUS, show that this attack vector can be used by adversaries to generate harmful responses and elicit misaligned responses from these models. By detailing both the mechanism and impact of the Sandwich attack, this paper aims to guide future research and development towards more secure and resilient LLMs, ensuring they serve the public good while minimizing potential for misuse.

摘要: 大型语言模型(LLM)的开发和应用越来越多，但它们的广泛使用面临着挑战。这些措施包括使LLMS的反应与人的价值观相一致，以防止有害的输出，这是通过安全培训方法解决的。尽管如此，不良行为者和恶意用户仍成功地操纵LLMS，以生成对有害问题的错误响应，这些问题包括在学校实验室制造炸弹的方法、有害药物的配方以及逃避隐私权的方法。另一个挑战是LLMS的多语言能力，这使得该模型能够理解并以多种语言响应。因此，攻击者利用不同语言的LLMS的不平衡的预训练数据集，以及低资源语言的模型性能相对较低的高资源语言。因此，攻击者使用低资源语言来故意操纵模型以创建有害的响应。许多类似的攻击载体已经由模型提供商修补，使LLM对基于语言的操纵更加健壮。本文介绍了一种新的黑盒攻击向量--夹心攻击：一种多语言混合攻击，它操纵最先进的LLM产生有害的和未对齐的响应。我们对谷歌的Bard、Gemini Pro、Llama-2-70-B-Chat、GPT-3.5-Turbo、GPT-4和Claude-3-opus这五个不同的模型进行的实验表明，该攻击向量可被攻击者用来生成有害响应并从这些模型中引发错误的响应。通过详细描述三明治攻击的机制和影响，本文旨在引导未来的研究和开发朝着更安全和更具弹性的方向发展，确保它们服务于公共利益，同时将滥用的可能性降至最低。



## **43. $\textit{LinkPrompt}$: Natural and Universal Adversarial Attacks on Prompt-based Language Models**

$\textit{LinkPrompt}$：基于XSLT语言模型的自然和普遍对抗攻击 cs.CL

Accepted to the main conference of NAACL2024

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2403.16432v3) [paper-pdf](http://arxiv.org/pdf/2403.16432v3)

**Authors**: Yue Xu, Wenjie Wang

**Abstract**: Prompt-based learning is a new language model training paradigm that adapts the Pre-trained Language Models (PLMs) to downstream tasks, which revitalizes the performance benchmarks across various natural language processing (NLP) tasks. Instead of using a fixed prompt template to fine-tune the model, some research demonstrates the effectiveness of searching for the prompt via optimization. Such prompt optimization process of prompt-based learning on PLMs also gives insight into generating adversarial prompts to mislead the model, raising concerns about the adversarial vulnerability of this paradigm. Recent studies have shown that universal adversarial triggers (UATs) can be generated to alter not only the predictions of the target PLMs but also the prediction of corresponding Prompt-based Fine-tuning Models (PFMs) under the prompt-based learning paradigm. However, UATs found in previous works are often unreadable tokens or characters and can be easily distinguished from natural texts with adaptive defenses. In this work, we consider the naturalness of the UATs and develop $\textit{LinkPrompt}$, an adversarial attack algorithm to generate UATs by a gradient-based beam search algorithm that not only effectively attacks the target PLMs and PFMs but also maintains the naturalness among the trigger tokens. Extensive results demonstrate the effectiveness of $\textit{LinkPrompt}$, as well as the transferability of UATs generated by $\textit{LinkPrompt}$ to open-sourced Large Language Model (LLM) Llama2 and API-accessed LLM GPT-3.5-turbo. The resource is available at $\href{https://github.com/SavannahXu79/LinkPrompt}{https://github.com/SavannahXu79/LinkPrompt}$.

摘要: 基于提示的学习是一种新的语言模型训练范式，它使预先训练的语言模型(PLM)适应于下游任务，从而重振各种自然语言处理(NLP)任务的表现基准。一些研究证明了通过优化来搜索提示的有效性，而不是使用固定的提示模板来微调模型。这种基于提示的PLM学习的快速优化过程也为生成对抗性提示以误导模型提供了洞察力，这引发了人们对这种范式的对抗性脆弱性的担忧。最近的研究表明，在基于提示的学习范式下，通用对抗触发器(UAT)不仅可以改变目标PLM的预测，还可以改变相应的基于提示的精调模型(PFM)的预测。然而，在以前的著作中发现的UAT通常是不可读的符号或字符，并且可以很容易地与具有自适应防御的自然文本区分开来。在这项工作中，我们考虑了UAT的自然性，并开发了一种对抗性攻击算法，通过基于梯度的波束搜索算法来生成UAT，该算法不仅有效地攻击了目标PLM和PPM，而且保持了触发令牌之间的自然度。广泛的结果证明了$\textit{LinkPrompt}$的有效性，以及由$\textit{LinkPrompt}$生成的UAT可以移植到开源的大型语言模型(LLM)Llama2和API访问的LLm GPT-3.5-Turbo。该资源可在$\href{https://github.com/SavannahXu79/LinkPrompt}{https://github.com/SavannahXu79/LinkPrompt}$.上获得



## **44. LRR: Language-Driven Resamplable Continuous Representation against Adversarial Tracking Attacks**

LRR：一种对抗跟踪攻击的可重采样连续表示方法 cs.CV

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.06247v1) [paper-pdf](http://arxiv.org/pdf/2404.06247v1)

**Authors**: Jianlang Chen, Xuhong Ren, Qing Guo, Felix Juefei-Xu, Di Lin, Wei Feng, Lei Ma, Jianjun Zhao

**Abstract**: Visual object tracking plays a critical role in visual-based autonomous systems, as it aims to estimate the position and size of the object of interest within a live video. Despite significant progress made in this field, state-of-the-art (SOTA) trackers often fail when faced with adversarial perturbations in the incoming frames. This can lead to significant robustness and security issues when these trackers are deployed in the real world. To achieve high accuracy on both clean and adversarial data, we propose building a spatial-temporal continuous representation using the semantic text guidance of the object of interest. This novel continuous representation enables us to reconstruct incoming frames to maintain semantic and appearance consistency with the object of interest and its clean counterparts. As a result, our proposed method successfully defends against different SOTA adversarial tracking attacks while maintaining high accuracy on clean data. In particular, our method significantly increases tracking accuracy under adversarial attacks with around 90% relative improvement on UAV123, which is even higher than the accuracy on clean data.

摘要: 视觉对象跟踪在基于视觉的自主系统中起着至关重要的作用，因为它的目标是估计实时视频中感兴趣对象的位置和大小。尽管在这一领域取得了重大进展，但最先进的(SOTA)跟踪器在面对传入帧中的对抗性扰动时往往会失败。当这些跟踪器部署在现实世界中时，这可能会导致严重的健壮性和安全性问题。为了实现对干净数据和对抗性数据的高准确度，我们建议使用感兴趣对象的语义文本指导来构建时空连续表示。这种新颖的连续表示使我们能够重建进入的帧，以保持与感兴趣的对象及其干净的对应物在语义和外观上的一致性。因此，我们提出的方法成功地防御了不同的SOTA对手跟踪攻击，同时保持了对干净数据的高精度。特别是，我们的方法显著提高了对抗性攻击下的跟踪准确率，与UAV123相比，相对提高了90%左右，甚至高于在干净数据上的准确率。



## **45. Towards Robust Domain Generation Algorithm Classification**

鲁棒的领域生成算法分类 cs.CR

Accepted at ACM Asia Conference on Computer and Communications  Security (ASIA CCS 2024)

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.06236v1) [paper-pdf](http://arxiv.org/pdf/2404.06236v1)

**Authors**: Arthur Drichel, Marc Meyer, Ulrike Meyer

**Abstract**: In this work, we conduct a comprehensive study on the robustness of domain generation algorithm (DGA) classifiers. We implement 32 white-box attacks, 19 of which are very effective and induce a false-negative rate (FNR) of $\approx$ 100\% on unhardened classifiers. To defend the classifiers, we evaluate different hardening approaches and propose a novel training scheme that leverages adversarial latent space vectors and discretized adversarial domains to significantly improve robustness. In our study, we highlight a pitfall to avoid when hardening classifiers and uncover training biases that can be easily exploited by attackers to bypass detection, but which can be mitigated by adversarial training (AT). In our study, we do not observe any trade-off between robustness and performance, on the contrary, hardening improves a classifier's detection performance for known and unknown DGAs. We implement all attacks and defenses discussed in this paper as a standalone library, which we make publicly available to facilitate hardening of DGA classifiers: https://gitlab.com/rwth-itsec/robust-dga-detection

摘要: 在这项工作中，我们对域生成算法(DGA)分类器的稳健性进行了全面的研究。我们实现了32个白盒攻击，其中19个非常有效，并在未硬化的分类器上诱导了约100美元的假阴性率(FNR)。为了保护分类器，我们评估了不同的强化方法，并提出了一种新的训练方案，该方案利用对抗性潜在空间向量和离散化的对抗性领域来显著提高鲁棒性。在我们的研究中，我们强调了在硬化分类器和发现训练偏差时需要避免的陷阱，攻击者可以很容易地利用这些偏差来绕过检测，但可以通过对抗性训练(AT)来缓解。在我们的研究中，我们没有观察到稳健性和性能之间的任何权衡，相反，硬化提高了分类器对已知和未知DGA的检测性能。我们将本文讨论的所有攻击和防御作为一个独立库来实现，我们公开该库是为了促进DGA分类器的强化：https://gitlab.com/rwth-itsec/robust-dga-detection



## **46. FLEX: FLEXible Federated Learning Framework**

FLEX：Flexible联邦学习框架 cs.CR

Submitted to Information Fusion

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.06127v1) [paper-pdf](http://arxiv.org/pdf/2404.06127v1)

**Authors**: Francisco Herrera, Daniel Jiménez-López, Alberto Argente-Garrido, Nuria Rodríguez-Barroso, Cristina Zuheros, Ignacio Aguilera-Martos, Beatriz Bello, Mario García-Márquez, M. Victoria Luzón

**Abstract**: In the realm of Artificial Intelligence (AI), the need for privacy and security in data processing has become paramount. As AI applications continue to expand, the collection and handling of sensitive data raise concerns about individual privacy protection. Federated Learning (FL) emerges as a promising solution to address these challenges by enabling decentralized model training on local devices, thus preserving data privacy. This paper introduces FLEX: a FLEXible Federated Learning Framework designed to provide maximum flexibility in FL research experiments. By offering customizable features for data distribution, privacy parameters, and communication strategies, FLEX empowers researchers to innovate and develop novel FL techniques. The framework also includes libraries for specific FL implementations including: (1) anomalies, (2) blockchain, (3) adversarial attacks and defences, (4) natural language processing and (5) decision trees, enhancing its versatility and applicability in various domains. Overall, FLEX represents a significant advancement in FL research, facilitating the development of robust and efficient FL applications.

摘要: 在人工智能(AI)领域，数据处理中对隐私和安全的需求已经变得至关重要。随着人工智能应用的不断扩大，敏感数据的收集和处理引发了对个人隐私保护的担忧。联合学习(FL)通过在本地设备上实现分散的模型训练，从而保护数据隐私，从而成为应对这些挑战的一种有前途的解决方案。本文介绍了FLEX：一个灵活的联邦学习框架，旨在为外语研究实验提供最大的灵活性。通过为数据分发、隐私参数和通信策略提供可定制的功能，FLEX使研究人员能够创新和开发新的FL技术。该框架还包括用于特定FL实现的库，包括：(1)异常、(2)区块链、(3)对抗性攻击和防御、(4)自然语言处理和(5)决策树，增强了其在各个领域的通用性和适用性。总体而言，FLEX代表着外语研究的重大进步，促进了强大而高效的外语应用程序的开发。



## **47. PeerAiD: Improving Adversarial Distillation from a Specialized Peer Tutor**

PeerAiD：从专业的同伴导师改善对抗蒸馏 cs.LG

Accepted to CVPR 2024

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2403.06668v2) [paper-pdf](http://arxiv.org/pdf/2403.06668v2)

**Authors**: Jaewon Jung, Hongsun Jang, Jaeyong Song, Jinho Lee

**Abstract**: Adversarial robustness of the neural network is a significant concern when it is applied to security-critical domains. In this situation, adversarial distillation is a promising option which aims to distill the robustness of the teacher network to improve the robustness of a small student network. Previous works pretrain the teacher network to make it robust to the adversarial examples aimed at itself. However, the adversarial examples are dependent on the parameters of the target network. The fixed teacher network inevitably degrades its robustness against the unseen transferred adversarial examples which targets the parameters of the student network in the adversarial distillation process. We propose PeerAiD to make a peer network learn the adversarial examples of the student network instead of adversarial examples aimed at itself. PeerAiD is an adversarial distillation that trains the peer network and the student network simultaneously in order to make the peer network specialized for defending the student network. We observe that such peer networks surpass the robustness of pretrained robust teacher network against student-attacked adversarial samples. With this peer network and adversarial distillation, PeerAiD achieves significantly higher robustness of the student network with AutoAttack (AA) accuracy up to 1.66%p and improves the natural accuracy of the student network up to 4.72%p with ResNet-18 and TinyImageNet dataset.

摘要: 当神经网络应用于安全关键领域时，它的对抗健壮性是一个重要的问题。在这种情况下，对抗性蒸馏是一种很有前途的选择，它旨在提取教师网络的健壮性，以提高小型学生网络的健壮性。以前的工作预先训练教师网络，使其对针对自己的对抗性例子具有健壮性。然而，对抗性的例子取决于目标网络的参数。在对抗性提取过程中，固定的教师网络不可避免地降低了其对看不见的转移的对抗性范例的鲁棒性，这些例子针对的是学生网络的参数。我们建议PeerAiD使对等网络学习学生网络的对抗性例子，而不是针对自己的对抗性例子。PeerAiD是一种对抗性的升华，它同时训练对等网络和学生网络，使对等网络专门用于防御学生网络。我们观察到这种对等网络超过了预先训练的稳健教师网络对学生攻击的对手样本的稳健性。通过这种对等网络和对抗性蒸馏，PeerAiD实现了显著更高的学生网络的健壮性，AutoAttack(AA)准确率高达1.66%p，并使用ResNet-18和TinyImageNet数据集将学生网络的自然准确率提高到4.72%p。



## **48. Greedy-DiM: Greedy Algorithms for Unreasonably Effective Face Morphs**

Greedy-DiM：用于不合理有效面部形态的贪婪算法 cs.CV

Initial preprint. Under review

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2404.06025v1) [paper-pdf](http://arxiv.org/pdf/2404.06025v1)

**Authors**: Zander W. Blasingame, Chen Liu

**Abstract**: Morphing attacks are an emerging threat to state-of-the-art Face Recognition (FR) systems, which aim to create a single image that contains the biometric information of multiple identities. Diffusion Morphs (DiM) are a recently proposed morphing attack that has achieved state-of-the-art performance for representation-based morphing attacks. However, none of the existing research on DiMs have leveraged the iterative nature of DiMs and left the DiM model as a black box, treating it no differently than one would a Generative Adversarial Network (GAN) or Varational AutoEncoder (VAE). We propose a greedy strategy on the iterative sampling process of DiM models which searches for an optimal step guided by an identity-based heuristic function. We compare our proposed algorithm against ten other state-of-the-art morphing algorithms using the open-source SYN-MAD 2022 competition dataset. We find that our proposed algorithm is unreasonably effective, fooling all of the tested FR systems with an MMPMR of 100%, outperforming all other morphing algorithms compared.

摘要: 变形攻击是对最先进的人脸识别(FR)系统的新威胁，该系统旨在创建包含多个身份的生物识别信息的单一图像。扩散变形(Dim)是最近提出的一种变形攻击，它已经在基于表示的变形攻击中获得了最先进的性能。然而，现有的关于DIMS的研究都没有利用DIMS的迭代性质，将DIM模型视为一个黑盒，将其视为与生成性对抗性网络(GAN)或变分自动编码器(VAE)没有区别的模型。针对DIM模型的迭代采样过程，我们提出了一种贪婪策略，在基于身份的启发式函数的指导下寻找最优步长。我们使用开源的SYN-MAD 2022竞赛数据集将我们提出的算法与其他十种最先进的变形算法进行了比较。我们发现我们提出的算法是不合理的有效的，愚弄了所有测试的FR系统，MMPMR为100%，比所有其他变形算法都要好。



## **49. A Vulnerability of Attribution Methods Using Pre-Softmax Scores**

使用Pre-Softmax评分的归因方法的一个漏洞 cs.LG

7 pages, 5 figures

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2307.03305v3) [paper-pdf](http://arxiv.org/pdf/2307.03305v3)

**Authors**: Miguel Lerma, Mirtha Lucas

**Abstract**: We discuss a vulnerability involving a category of attribution methods used to provide explanations for the outputs of convolutional neural networks working as classifiers. It is known that this type of networks are vulnerable to adversarial attacks, in which imperceptible perturbations of the input may alter the outputs of the model. In contrast, here we focus on effects that small modifications in the model may cause on the attribution method without altering the model outputs.

摘要: 我们讨论了一个漏洞，涉及一类属性方法用于解释卷积神经网络作为分类器的输出。众所周知，这种类型的网络容易受到对抗攻击，其中输入的不可察觉的扰动可能会改变模型的输出。相反，这里我们关注的是模型中的小修改可能对归因方法造成的影响，而不会改变模型输出。



## **50. Improving the Accuracy-Robustness Trade-Off of Classifiers via Adaptive Smoothing**

基于自适应平滑的分类器精度-鲁棒性权衡 cs.LG

**SubmitDate**: 2024-04-09    [abs](http://arxiv.org/abs/2301.12554v4) [paper-pdf](http://arxiv.org/pdf/2301.12554v4)

**Authors**: Yatong Bai, Brendon G. Anderson, Aerin Kim, Somayeh Sojoudi

**Abstract**: While prior research has proposed a plethora of methods that build neural classifiers robust against adversarial robustness, practitioners are still reluctant to adopt them due to their unacceptably severe clean accuracy penalties. This paper significantly alleviates this accuracy-robustness trade-off by mixing the output probabilities of a standard classifier and a robust classifier, where the standard network is optimized for clean accuracy and is not robust in general. We show that the robust base classifier's confidence difference for correct and incorrect examples is the key to this improvement. In addition to providing intuitions and empirical evidence, we theoretically certify the robustness of the mixed classifier under realistic assumptions. Furthermore, we adapt an adversarial input detector into a mixing network that adaptively adjusts the mixture of the two base models, further reducing the accuracy penalty of achieving robustness. The proposed flexible method, termed "adaptive smoothing", can work in conjunction with existing or even future methods that improve clean accuracy, robustness, or adversary detection. Our empirical evaluation considers strong attack methods, including AutoAttack and adaptive attack. On the CIFAR-100 dataset, our method achieves an 85.21% clean accuracy while maintaining a 38.72% $\ell_\infty$-AutoAttacked ($\epsilon = 8/255$) accuracy, becoming the second most robust method on the RobustBench CIFAR-100 benchmark as of submission, while improving the clean accuracy by ten percentage points compared with all listed models. The code that implements our method is available at https://github.com/Bai-YT/AdaptiveSmoothing.

摘要: 虽然先前的研究已经提出了太多的方法来构建稳健的神经分类器来对抗对手的健壮性，但实践者仍然不愿采用它们，因为它们具有不可接受的严重的干净准确性惩罚。本文通过混合标准分类器和稳健分类器的输出概率显著缓解了这种精度与稳健性的权衡，其中标准网络针对干净的精度进行了优化，而通常不是稳健的。研究表明，稳健的基分类器对正确样本和错误样本的置信度差异是这一改进的关键。除了提供直觉和经验证据外，我们还从理论上证明了混合分类器在现实假设下的稳健性。此外，我们将对抗性输入检测器引入混合网络，该混合网络自适应地调整两个基本模型的混合，从而进一步降低了实现稳健性的精度损失。这一灵活的方法被称为“自适应平滑”，可以与现有甚至未来的方法结合使用，以提高干净的准确性、健壮性或敌手检测。我们的经验评估考虑了强攻击方法，包括AutoAttack和自适应攻击。在CIFAR-100数据集上，我们的方法实现了85.21%的清洁准确率，同时保持了38.72%的$\ELL_\INFTY$-AutoAttaced($\epsilon=8/255$)精度，成为截至提交时在RobustBuchCIFAR-100基准上第二健壮的方法，同时与所有列出的模型相比，清洁准确率提高了10个百分点。实现我们方法的代码可以在https://github.com/Bai-YT/AdaptiveSmoothing.上找到



