# Latest Adversarial Attack Papers
**update at 2024-03-11 09:24:08**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Can LLMs Follow Simple Rules?**

低收入国家能遵循简单的规则吗？ cs.AI

Project website: https://eecs.berkeley.edu/~normanmu/llm_rules;  revised content

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2311.04235v3) [paper-pdf](http://arxiv.org/pdf/2311.04235v3)

**Authors**: Norman Mu, Sarah Chen, Zifan Wang, Sizhe Chen, David Karamardian, Lulwa Aljeraisy, Basel Alomair, Dan Hendrycks, David Wagner

**Abstract**: As Large Language Models (LLMs) are deployed with increasing real-world responsibilities, it is important to be able to specify and constrain the behavior of these systems in a reliable manner. Model developers may wish to set explicit rules for the model, such as "do not generate abusive content", but these may be circumvented by jailbreaking techniques. Existing evaluations of adversarial attacks and defenses on LLMs generally require either expensive manual review or unreliable heuristic checks. To address this issue, we propose Rule-following Language Evaluation Scenarios (RuLES), a programmatic framework for measuring rule-following ability in LLMs. RuLES consists of 14 simple text scenarios in which the model is instructed to obey various rules while interacting with the user. Each scenario has a programmatic evaluation function to determine whether the model has broken any rules in a conversation. Our evaluations of proprietary and open models show that almost all current models struggle to follow scenario rules, even on straightforward test cases. We also demonstrate that simple optimization attacks suffice to significantly increase failure rates on test cases. We conclude by exploring two potential avenues for improvement: test-time steering and supervised fine-tuning.

摘要: 随着大型语言模型(LLM)的部署承担着越来越多的现实责任，能够以可靠的方式指定和约束这些系统的行为是很重要的。模型开发人员可能希望为模型设置明确的规则，例如“不要生成滥用内容”，但可以通过越狱技术绕过这些规则。现有的对抗性攻击和防御评估通常需要昂贵的人工审查或不可靠的启发式检查。为了解决这一问题，我们提出了规则遵循语言评估场景(Rules)，这是一个衡量LLMS中规则遵循能力的程序性框架。规则由14个简单的文本场景组成，在这些场景中，模型被指示在与用户交互时遵守各种规则。每个场景都有一个程序化的评估功能，以确定模型是否违反了对话中的任何规则。我们对专有和开放模型的评估表明，几乎所有当前的模型都难以遵循场景规则，即使在简单的测试用例上也是如此。我们还证明了简单的优化攻击足以显著增加测试用例的失败率。最后，我们探索了两个潜在的改进途径：测试时间控制和有监督的微调。



## **2. On Practicality of Using ARM TrustZone Trusted Execution Environment for Securing Programmable Logic Controllers**

利用ARM TrustZone可信执行环境保护可编程逻辑控制器的实用性研究 cs.CR

To appear at ACM AsiaCCS 2024

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05448v1) [paper-pdf](http://arxiv.org/pdf/2403.05448v1)

**Authors**: Zhiang Li, Daisuke Mashima, Wen Shei Ong, Ertem Esiner, Zbigniew Kalbarczyk, Ee-Chien Chang

**Abstract**: Programmable logic controllers (PLCs) are crucial devices for implementing automated control in various industrial control systems (ICS), such as smart power grids, water treatment systems, manufacturing, and transportation systems. Owing to their importance, PLCs are often the target of cyber attackers that are aiming at disrupting the operation of ICS, including the nation's critical infrastructure, by compromising the integrity of control logic execution. While a wide range of cybersecurity solutions for ICS have been proposed, they cannot counter strong adversaries with a foothold on the PLC devices, which could manipulate memory, I/O interface, or PLC logic itself. These days, many ICS devices in the market, including PLCs, run on ARM-based processors, and there is a promising security technology called ARM TrustZone, to offer a Trusted Execution Environment (TEE) on embedded devices. Envisioning that such a hardware-assisted security feature becomes available for ICS devices in the near future, this paper investigates the application of the ARM TrustZone TEE technology for enhancing the security of PLC. Our aim is to evaluate the feasibility and practicality of the TEE-based PLCs through the proof-of-concept design and implementation using open-source software such as OP-TEE and OpenPLC. Our evaluation assesses the performance and resource consumption in real-world ICS configurations, and based on the results, we discuss bottlenecks in the OP-TEE secure OS towards a large-scale ICS and desired changes for its application on ICS devices. Our implementation is made available to public for further study and research.

摘要: 可编程控制器(PLC)是智能电网、水处理系统、制造业和交通运输系统等工业控制系统中实现自动化控制的关键器件。由于PLC的重要性，PLC经常成为网络攻击者的目标，他们的目标是通过损害控制逻辑执行的完整性来扰乱IC的运行，包括国家的关键基础设施。虽然已经提出了广泛的ICS网络安全解决方案，但它们无法对抗立足于PLC设备的强大对手，这些设备可以操纵内存、I/O接口或PLC逻辑本身。如今，市场上的许多ICS设备，包括PLC，都运行在基于ARM的处理器上，有一种很有前途的安全技术ARM TrustZone，它可以在嵌入式设备上提供一个可信执行环境(TEE)。鉴于这种硬件辅助的安全特性将在不久的将来应用于ICS设备，本文研究了ARM TrustZone TEE技术在增强PLC安全性方面的应用。我们的目的是通过使用开源软件OP-TEE和OpenPLC进行概念验证设计和实现，来评估基于TEE的PLC的可行性和实用性。我们的评估评估了实际ICS配置中的性能和资源消耗，并基于评估结果，讨论了操作安全操作系统向大规模ICS发展的瓶颈以及它在ICS设备上应用的期望变化。我们的实现可供公众进一步学习和研究。



## **3. EVD4UAV: An Altitude-Sensitive Benchmark to Evade Vehicle Detection in UAV**

EVD4无人机：一种用于躲避无人机车辆检测的高度敏感基准 cs.CV

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05422v1) [paper-pdf](http://arxiv.org/pdf/2403.05422v1)

**Authors**: Huiming Sun, Jiacheng Guo, Zibo Meng, Tianyun Zhang, Jianwu Fang, Yuewei Lin, Hongkai Yu

**Abstract**: Vehicle detection in Unmanned Aerial Vehicle (UAV) captured images has wide applications in aerial photography and remote sensing. There are many public benchmark datasets proposed for the vehicle detection and tracking in UAV images. Recent studies show that adding an adversarial patch on objects can fool the well-trained deep neural networks based object detectors, posing security concerns to the downstream tasks. However, the current public UAV datasets might ignore the diverse altitudes, vehicle attributes, fine-grained instance-level annotation in mostly side view with blurred vehicle roof, so none of them is good to study the adversarial patch based vehicle detection attack problem. In this paper, we propose a new dataset named EVD4UAV as an altitude-sensitive benchmark to evade vehicle detection in UAV with 6,284 images and 90,886 fine-grained annotated vehicles. The EVD4UAV dataset has diverse altitudes (50m, 70m, 90m), vehicle attributes (color, type), fine-grained annotation (horizontal and rotated bounding boxes, instance-level mask) in top view with clear vehicle roof. One white-box and two black-box patch based attack methods are implemented to attack three classic deep neural networks based object detectors on EVD4UAV. The experimental results show that these representative attack methods could not achieve the robust altitude-insensitive attack performance.

摘要: 无人机拍摄的图像中的车辆检测在航空摄影和遥感中有着广泛的应用。针对无人机图像中的车辆检测和跟踪，已经提出了许多公开的基准数据集。最近的研究表明，在对象上添加对抗性补丁可以欺骗训练有素的基于深度神经网络的对象检测器，从而给下游任务带来安全隐患。然而，目前公开的无人机数据集可能忽略了车顶模糊的侧视图中不同的高度、车辆属性、细粒度的实例级标注，因此不利于研究基于对抗性补丁的车辆检测攻击问题。在本文中，我们提出了一个新的数据集EVD4UAV作为高度敏感基准来逃避无人机中的车辆检测，该数据集包含6,284张图像和90,886辆细粒度标注的车辆。EVD4UAV数据集在俯视图中具有不同的高度(50m、70m、90m)、车辆属性(颜色、类型)、细粒度注释(水平和旋转的边界框、实例级遮罩)，并具有清晰的车顶。采用一种基于白盒和两种基于黑盒补丁的攻击方法，对EVD4无人机上三种经典的基于深度神经网络的目标探测器进行攻击。实验结果表明，这些具有代表性的攻击方法不能达到稳健的高度不敏感攻击性能。



## **4. The Impact of Quantization on the Robustness of Transformer-based Text Classifiers**

量化对基于Transformer的文本分类器稳健性的影响 cs.CL

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05365v1) [paper-pdf](http://arxiv.org/pdf/2403.05365v1)

**Authors**: Seyed Parsa Neshaei, Yasaman Boreshban, Gholamreza Ghassem-Sani, Seyed Abolghasem Mirroshandel

**Abstract**: Transformer-based models have made remarkable advancements in various NLP areas. Nevertheless, these models often exhibit vulnerabilities when confronted with adversarial attacks. In this paper, we explore the effect of quantization on the robustness of Transformer-based models. Quantization usually involves mapping a high-precision real number to a lower-precision value, aiming at reducing the size of the model at hand. To the best of our knowledge, this work is the first application of quantization on the robustness of NLP models. In our experiments, we evaluate the impact of quantization on BERT and DistilBERT models in text classification using SST-2, Emotion, and MR datasets. We also evaluate the performance of these models against TextFooler, PWWS, and PSO adversarial attacks. Our findings show that quantization significantly improves (by an average of 18.68%) the adversarial accuracy of the models. Furthermore, we compare the effect of quantization versus that of the adversarial training approach on robustness. Our experiments indicate that quantization increases the robustness of the model by 18.80% on average compared to adversarial training without imposing any extra computational overhead during training. Therefore, our results highlight the effectiveness of quantization in improving the robustness of NLP models.

摘要: 基于变压器的模型在各个NLP领域取得了显著的进步。然而，这些模型在面对对手攻击时往往表现出脆弱性。在本文中，我们探讨了量化对基于变压器的模型的稳健性的影响。量化通常涉及将高精度的实数映射到低精度的值，目的是减小手头模型的大小。据我们所知，这项工作是量化在NLP模型稳健性方面的首次应用。在我们的实验中，我们使用SST-2、情感和MR数据集评估了量化对BERT和DistilBERT模型在文本分类中的影响。我们还评估了这些模型对TextFooler、PWWS和PSO对手攻击的性能。我们的结果表明，量化显著提高了模型的对抗准确率(平均提高了18.68%)。此外，我们比较了量化和对抗性训练方法在稳健性方面的效果。我们的实验表明，与对抗性训练相比，量化使模型的健壮性平均提高了18.80%，而在训练过程中不增加任何额外的计算开销。因此，我们的结果突出了量化在提高NLP模型的稳健性方面的有效性。



## **5. Hide in Thicket: Generating Imperceptible and Rational Adversarial Perturbations on 3D Point Clouds**

隐藏在诡计中：在3D点云上生成不可察觉的和理性的对抗性扰动 cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05247v1) [paper-pdf](http://arxiv.org/pdf/2403.05247v1)

**Authors**: Tianrui Lou, Xiaojun Jia, Jindong Gu, Li Liu, Siyuan Liang, Bangyan He, Xiaochun Cao

**Abstract**: Adversarial attack methods based on point manipulation for 3D point cloud classification have revealed the fragility of 3D models, yet the adversarial examples they produce are easily perceived or defended against. The trade-off between the imperceptibility and adversarial strength leads most point attack methods to inevitably introduce easily detectable outlier points upon a successful attack. Another promising strategy, shape-based attack, can effectively eliminate outliers, but existing methods often suffer significant reductions in imperceptibility due to irrational deformations. We find that concealing deformation perturbations in areas insensitive to human eyes can achieve a better trade-off between imperceptibility and adversarial strength, specifically in parts of the object surface that are complex and exhibit drastic curvature changes. Therefore, we propose a novel shape-based adversarial attack method, HiT-ADV, which initially conducts a two-stage search for attack regions based on saliency and imperceptibility scores, and then adds deformation perturbations in each attack region using Gaussian kernel functions. Additionally, HiT-ADV is extendable to physical attack. We propose that by employing benign resampling and benign rigid transformations, we can further enhance physical adversarial strength with little sacrifice to imperceptibility. Extensive experiments have validated the superiority of our method in terms of adversarial and imperceptible properties in both digital and physical spaces. Our code is avaliable at: https://github.com/TRLou/HiT-ADV.

摘要: 基于点操作的三维点云分类对抗性攻击方法暴露了三维模型的脆弱性，但它们产生的对抗性实例很容易被感知或防御。隐蔽性和对抗性之间的权衡导致大多数点攻击方法在攻击成功后不可避免地引入容易检测到的离群点。另一种很有希望的策略是基于形状的攻击，它可以有效地消除离群点，但现有的方法由于不合理的变形往往会显著降低不可感知性。我们发现，在对人眼不敏感的区域隐藏变形扰动可以在不可感知性和对抗强度之间实现更好的权衡，特别是在物体表面复杂和曲率变化剧烈的部分。因此，我们提出了一种新的基于形状的对抗性攻击方法HIT-ADV，该方法首先根据显著分数和不可感知性分数进行两阶段的攻击区域搜索，然后使用高斯核函数在每个攻击区域添加变形扰动。此外，HIT-ADV可以扩展到物理攻击。我们提出，通过使用良性重采样和良性刚性变换，我们可以在几乎不牺牲不可感知性的情况下进一步增强物理对抗强度。广泛的实验已经验证了我们的方法在数字空间和物理空间中的对抗性和不可感知性方面的优越性。我们的代码可在：https://github.com/TRLou/HiT-ADV.上获得



## **6. Adversarial Sparse Teacher: Defense Against Distillation-Based Model Stealing Attacks Using Adversarial Examples**

对抗性稀疏教师：使用对抗性实例防御基于蒸馏的模型窃取攻击 cs.LG

12 pages, 3 figures, 6 tables

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05181v1) [paper-pdf](http://arxiv.org/pdf/2403.05181v1)

**Authors**: Eda Yilmaz, Hacer Yalim Keles

**Abstract**: Knowledge Distillation (KD) facilitates the transfer of discriminative capabilities from an advanced teacher model to a simpler student model, ensuring performance enhancement without compromising accuracy. It is also exploited for model stealing attacks, where adversaries use KD to mimic the functionality of a teacher model. Recent developments in this domain have been influenced by the Stingy Teacher model, which provided empirical analysis showing that sparse outputs can significantly degrade the performance of student models. Addressing the risk of intellectual property leakage, our work introduces an approach to train a teacher model that inherently protects its logits, influenced by the Nasty Teacher concept. Differing from existing methods, we incorporate sparse outputs of adversarial examples with standard training data to strengthen the teacher's defense against student distillation. Our approach carefully reduces the relative entropy between the original and adversarially perturbed outputs, allowing the model to produce adversarial logits with minimal impact on overall performance. The source codes will be made publicly available soon.

摘要: 知识蒸馏(KD)有助于将区分能力从高级教师模型转移到更简单的学生模型，确保在不影响准确性的情况下提高成绩。它还被利用来进行模型窃取攻击，攻击者使用KD来模仿教师模型的功能。这一领域的最新发展受到吝啬教师模型的影响，该模型提供的实证分析表明，稀疏的输出会显著降低学生模型的表现。为了应对知识产权泄露的风险，我们的工作引入了一种方法，以培养一种受肮脏的教师概念影响而内在地保护其逻辑的教师模式。与现有方法不同的是，我们将对抗性样本的稀疏输出与标准训练数据相结合，以加强教师对学生蒸馏的防御。我们的方法仔细地减少了原始输出和对抗性扰动输出之间的相对熵，允许模型在对整体性能影响最小的情况下生成对抗性逻辑。源代码很快就会公开。



## **7. Warfare:Breaking the Watermark Protection of AI-Generated Content**

战争：打破人工智能生成内容的水印保护 cs.CV

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2310.07726v3) [paper-pdf](http://arxiv.org/pdf/2310.07726v3)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Jiwei Li, Shangwei Guo, Tianwei Zhang

**Abstract**: AI-Generated Content (AIGC) is gaining great popularity, with many emerging commercial services and applications. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images and fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content). A promising solution to achieve this goal is watermarking, which adds unique and imperceptible watermarks on the content for service verification and attribution. Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely bypassing the regulation of the service provider. (2) Watermark forging: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. We propose Warfare, a unified methodology to achieve both attacks in a holistic way. The key idea is to leverage a pre-trained diffusion model for content processing and a generative adversarial network for watermark removal or forging. We evaluate Warfare on different datasets and embedding setups. The results prove that it can achieve high success rates while maintaining the quality of the generated content. Compared to existing diffusion model-based attacks, Warfare is 5,050~11,000x faster.

摘要: 人工智能生成的内容(AIGC)越来越受欢迎，出现了许多新兴的商业服务和应用程序。这些服务利用高级生成模型，如潜在扩散模型和大型语言模型，为用户生成创造性内容(例如，逼真的图像和流畅的句子)。这种生成的内容的使用需要受到严格的监管，因为服务提供商需要确保用户不违反使用策略(例如，滥用以商业化、生成和分发不安全的内容)。实现这一目标的一个有前途的解决方案是水印，它在内容上添加唯一且不可察觉的水印，用于服务验证和归属。最近，人们提出了许多水印方法。然而，在本文中，我们证明了攻击者可以很容易地破解这些水印机制。具体地说，我们考虑两种可能的攻击。(1)水印去除：攻击者可以很容易地从生成的内容中删除嵌入的水印，然后绕过服务提供商的监管自由使用。(2)水印伪造：对手可以利用来自其他用户的伪造水印创建非法内容，导致服务提供商做出错误的归属。我们提出战争，一种统一的方法论，以整体的方式实现这两种攻击。其关键思想是利用预先训练的扩散模型来进行内容处理，并利用生成性对抗网络来去除或伪造水印。我们对不同的数据集和嵌入设置进行了战争评估。实验结果表明，该算法在保证生成内容质量的同时，具有较高的成功率。与现有的基于扩散模型的攻击相比，战争的速度要快5050~11000倍。



## **8. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

针对大型语言模型的间接提示注入攻击的基准测试和防御 cs.CL

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2312.14197v3) [paper-pdf](http://arxiv.org/pdf/2312.14197v3)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: The integration of large language models (LLMs) with external content has enabled more up-to-date and wide-ranging applications of LLMs, such as Microsoft Copilot. However, this integration has also exposed LLMs to the risk of indirect prompt injection attacks, where an attacker can embed malicious instructions within external content, compromising LLM output and causing responses to deviate from user expectations. To investigate this important but underexplored issue, we introduce the first benchmark for indirect prompt injection attacks, named BIPIA, to evaluate the risk of such attacks. Based on the evaluation, our work makes a key analysis of the underlying reason for the success of the attack, namely the inability of LLMs to distinguish between instructions and external content and the absence of LLMs' awareness to not execute instructions within external content. Building upon this analysis, we develop two black-box methods based on prompt learning and a white-box defense method based on fine-tuning with adversarial training accordingly. Experimental results demonstrate that black-box defenses are highly effective in mitigating these attacks, while the white-box defense reduces the attack success rate to near-zero levels. Overall, our work systematically investigates indirect prompt injection attacks by introducing a benchmark, analyzing the underlying reason for the success of the attack, and developing an initial set of defenses.

摘要: 大型语言模型(LLM)与外部内容的集成使LLM能够更新、更广泛地应用，如Microsoft Copilot。然而，这种集成也使LLMS面临间接提示注入攻击的风险，攻击者可以在外部内容中嵌入恶意指令，损害LLM输出并导致响应偏离用户预期。为了研究这一重要但未被探索的问题，我们引入了第一个间接即时注入攻击基准，称为BIPIA，以评估此类攻击的风险。在评估的基础上，我们的工作重点分析了攻击成功的根本原因，即LLMS无法区分指令和外部内容，以及LLMS缺乏不执行外部内容中的指令的意识。在此基础上，我们提出了两种基于快速学习的黑盒防御方法和一种基于微调对抗性训练的白盒防御方法。实验结果表明，黑盒防御对于缓解这些攻击是非常有效的，而白盒防御将攻击成功率降低到接近于零的水平。总体而言，我们的工作通过引入基准、分析攻击成功的根本原因以及开发一套初始防御措施来系统地调查间接即时注入攻击。



## **9. Exploring the Adversarial Frontier: Quantifying Robustness via Adversarial Hypervolume**

探索对抗性前沿：通过对抗性超卷量化稳健性 cs.CR

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05100v1) [paper-pdf](http://arxiv.org/pdf/2403.05100v1)

**Authors**: Ping Guo, Cheng Gong, Xi Lin, Zhiyuan Yang, Qingfu Zhang

**Abstract**: The escalating threat of adversarial attacks on deep learning models, particularly in security-critical fields, has underscored the need for robust deep learning systems. Conventional robustness evaluations have relied on adversarial accuracy, which measures a model's performance under a specific perturbation intensity. However, this singular metric does not fully encapsulate the overall resilience of a model against varying degrees of perturbation. To address this gap, we propose a new metric termed adversarial hypervolume, assessing the robustness of deep learning models comprehensively over a range of perturbation intensities from a multi-objective optimization standpoint. This metric allows for an in-depth comparison of defense mechanisms and recognizes the trivial improvements in robustness afforded by less potent defensive strategies. Additionally, we adopt a novel training algorithm that enhances adversarial robustness uniformly across various perturbation intensities, in contrast to methods narrowly focused on optimizing adversarial accuracy. Our extensive empirical studies validate the effectiveness of the adversarial hypervolume metric, demonstrating its ability to reveal subtle differences in robustness that adversarial accuracy overlooks. This research contributes a new measure of robustness and establishes a standard for assessing and benchmarking the resilience of current and future defensive models against adversarial threats.

摘要: 对深度学习模型的敌意攻击的威胁不断升级，特别是在安全关键领域，这突显了需要强大的深度学习系统。传统的稳健性评估依赖于对抗精度，该精度衡量模型在特定扰动强度下的性能。然而，这种单一的度量并不能完全概括模型对不同程度扰动的总体弹性。为了弥补这一差距，我们提出了一种新的度量标准，称为对抗性超体积，从多目标优化的角度全面评估深度学习模型在一系列扰动强度下的稳健性。这一指标允许对防御机制进行深入比较，并认识到较弱的防御策略在健壮性方面的微小改进。此外，我们采用了一种新的训练算法，该算法在不同的扰动强度下均匀地增强了对抗的稳健性，而不是狭隘地专注于优化对抗的准确性。我们广泛的实证研究验证了对抗性超卷度量的有效性，证明了它能够揭示对抗性准确性忽略的稳健性的细微差异。这项研究提供了一种新的稳健性衡量标准，并为评估和基准当前和未来防御模型对对手威胁的弹性建立了标准。



## **10. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

用潜在对手训练防御不可预见的失败模式 cs.CR

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05030v1) [paper-pdf](http://arxiv.org/pdf/2403.05030v1)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: AI systems sometimes exhibit harmful unintended behaviors post-deployment. This is often despite extensive diagnostics and debugging by developers. Minimizing risks from models is challenging because the attack surface is so large. It is not tractable to exhaustively search for inputs that may cause a model to fail. Red-teaming and adversarial training (AT) are commonly used to make AI systems more robust. However, they have not been sufficient to avoid many real-world failure modes that differ from the ones adversarially trained on. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without generating inputs that elicit them. LAT leverages the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. We use LAT to remove trojans and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.

摘要: 人工智能系统有时会在部署后表现出有害的意外行为。这通常是尽管开发人员进行了广泛的诊断和调试。将模型的风险降至最低是具有挑战性的，因为攻击面如此之大。要详尽地搜索可能导致模型失败的输入是不容易的。红队和对抗训练(AT)通常被用来使AI系统更健壮。然而，它们还不足以避免许多现实世界中的失败模式，这些模式与对手训练的模式不同。在这项工作中，我们利用潜在的对手训练(LAT)来防御漏洞，而不会生成引发漏洞的输入。随后，利用网络实际用于预测的概念的压缩、抽象和结构化的潜在表示。我们使用LAT来删除特洛伊木马程序，并防御抵抗类的对抗性攻击。我们在图像分类、文本分类和文本生成任务中表明，与AT相比，LAT通常可以提高对干净数据的稳健性和性能。这表明，LAT可以成为一种很有前途的工具，用于防御开发人员未明确识别的故障模式。



## **11. Fooling Neural Networks for Motion Forecasting via Adversarial Attacks**

利用对抗性攻击愚弄神经网络进行运动预测 cs.CV

11 pages, 8 figures, VISSAP 2024

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2403.04954v1) [paper-pdf](http://arxiv.org/pdf/2403.04954v1)

**Authors**: Edgar Medina, Leyong Loh

**Abstract**: Human motion prediction is still an open problem, which is extremely important for autonomous driving and safety applications. Although there are great advances in this area, the widely studied topic of adversarial attacks has not been applied to multi-regression models such as GCNs and MLP-based architectures in human motion prediction. This work intends to reduce this gap using extensive quantitative and qualitative experiments in state-of-the-art architectures similar to the initial stages of adversarial attacks in image classification. The results suggest that models are susceptible to attacks even on low levels of perturbation. We also show experiments with 3D transformations that affect the model performance, in particular, we show that most models are sensitive to simple rotations and translations which do not alter joint distances. We conclude that similar to earlier CNN models, motion forecasting tasks are susceptible to small perturbations and simple 3D transformations.

摘要: 人体运动预测仍然是一个悬而未决的问题，对于自动驾驶和安全应用具有极其重要的意义。尽管这一领域已经取得了很大的进展，但被广泛研究的对抗性攻击主题还没有被应用到多元回归模型中，如GCNS和基于MLP的人体运动预测体系。这项工作旨在通过在最先进的体系结构中进行广泛的定量和定性实验来缩小这一差距，该体系结构类似于图像分类中对抗性攻击的初始阶段。结果表明，即使在低水平的扰动下，模型也容易受到攻击。我们还展示了影响模型性能的3D变换的实验，特别是，我们表明大多数模型对简单的旋转和平移都很敏感，这些旋转和平移不会改变关节距离。我们的结论是，与早期的CNN模型类似，运动预测任务容易受到小扰动和简单的3D变换的影响。



## **12. Optimal Denial-of-Service Attacks Against Status Updating**

针对状态更新的最优拒绝服务攻击 cs.IT

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2403.04489v1) [paper-pdf](http://arxiv.org/pdf/2403.04489v1)

**Authors**: Saad Kriouile, Mohamad Assaad, Deniz Gündüz, Touraj Soleymani

**Abstract**: In this paper, we investigate denial-of-service attacks against status updating. The target system is modeled by a Markov chain and an unreliable wireless channel, and the performance of status updating in the target system is measured based on two metrics: age of information and age of incorrect information. Our objective is to devise optimal attack policies that strike a balance between the deterioration of the system's performance and the adversary's energy. We model the optimal problem as a Markov decision process and prove rigorously that the optimal jamming policy is a threshold-based policy under both metrics. In addition, we provide a low-complexity algorithm to obtain the optimal threshold value of the jamming policy. Our numerical results show that the networked system with the age-of-incorrect-information metric is less sensitive to jamming attacks than with the age-of-information metric. Index Terms-age of incorrect information, age of information, cyber-physical systems, status updating, remote monitoring.

摘要: 在本文中，我们研究针对状态更新的拒绝服务攻击。将目标系统建模为马尔可夫链和不可靠无线信道，并基于信息年龄和错误信息年龄两个度量来衡量目标系统的状态更新性能。我们的目标是设计最优的攻击策略，在系统性能的恶化和对手的能量之间取得平衡。我们将最优问题建模为马尔可夫决策过程，并严格证明了在两种度量下，最优干扰策略都是基于门限的策略。此外，我们还提出了一种低复杂度的算法来获取干扰策略的最优阈值。数值结果表明，具有错误信息年龄度量的网络系统对干扰攻击的敏感度低于具有信息年龄度量的网络系统。索引术语-不正确信息的年龄、信息的年龄、网络物理系统、状态更新、远程监控。



## **13. Pilot Spoofing Attack on the Downlink of Cell-Free Massive MIMO: From the Perspective of Adversaries**

无蜂窝海量MIMO下行链路的飞行员欺骗攻击：对手视角 cs.IT

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2403.04435v1) [paper-pdf](http://arxiv.org/pdf/2403.04435v1)

**Authors**: Weiyang Xu, Yuan Zhang, Ruiguang Wang, Hien Quoc Ngo, Wei Xiang

**Abstract**: The channel hardening effect is less pronounced in the cell-free massive multiple-input multiple-output (mMIMO) system compared to its cellular counterpart, making it necessary to estimate the downlink effective channel gains to ensure decent performance. However, the downlink training inadvertently creates an opportunity for adversarial nodes to launch pilot spoofing attacks (PSAs). First, we demonstrate that adversarial distributed access points (APs) can severely degrade the achievable downlink rate. They achieve this by estimating their channels to users in the uplink training phase and then precoding and sending the same pilot sequences as those used by legitimate APs during the downlink training phase. Then, the impact of the downlink PSA is investigated by rigorously deriving a closed-form expression of the per-user achievable downlink rate. By employing the min-max criterion to optimize the power allocation coefficients, the maximum per-user achievable rate of downlink transmission is minimized from the perspective of adversarial APs. As an alternative to the downlink PSA, adversarial APs may opt to precode random interference during the downlink data transmission phase in order to disrupt legitimate communications. In this scenario, the achievable downlink rate is derived, and then power optimization algorithms are also developed. We present numerical results to showcase the detrimental impact of the downlink PSA and compare the effects of these two types of attacks.

摘要: 与蜂窝系统相比，无小区大规模多输入多输出(MMIMO)系统中的信道硬化效应不那么明显，因此有必要估计下行链路的有效信道增益以确保良好的性能。然而，下行训练无意中为敌对节点创造了发起试点欺骗攻击(PSA)的机会。首先，我们证明了敌意分布式接入点(AP)会严重降低可实现的下行链路速率。它们通过在上行链路训练阶段估计其对用户的信道，然后预编码并发送与合法AP在下行链路训练阶段使用的导频序列相同的导频序列来实现这一点。然后，通过严格推导每个用户可实现的下行链路速率的闭合形式表达式来研究下行链路PSA的影响。通过使用最小-最大准则来优化功率分配系数，从对抗性AP的角度最小化每用户可实现的最大下行传输速率。作为下行链路PSA的替代方案，敌意AP可以选择在下行链路数据传输阶段对随机干扰进行预编码，以便中断合法通信。在这种情况下，推导了可实现的下行链路速率，并开发了功率优化算法。我们给出了数值结果来展示下行PSA的有害影响，并比较了这两种类型的攻击的影响。



## **14. Evaluating the security of CRYSTALS-Dilithium in the quantum random oracle model**

在量子随机预言模型中评估晶体双锂的安全性 cs.CR

23 pages; v2: added description of CRYSTALS-Dilithium, improved  analysis of concrete parameters

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2312.16619v2) [paper-pdf](http://arxiv.org/pdf/2312.16619v2)

**Authors**: Kelsey A. Jackson, Carl A. Miller, Daochen Wang

**Abstract**: In the wake of recent progress on quantum computing hardware, the National Institute of Standards and Technology (NIST) is standardizing cryptographic protocols that are resistant to attacks by quantum adversaries. The primary digital signature scheme that NIST has chosen is CRYSTALS-Dilithium. The hardness of this scheme is based on the hardness of three computational problems: Module Learning with Errors (MLWE), Module Short Integer Solution (MSIS), and SelfTargetMSIS. MLWE and MSIS have been well-studied and are widely believed to be secure. However, SelfTargetMSIS is novel and, though classically as hard as MSIS, its quantum hardness is unclear. In this paper, we provide the first proof of the hardness of SelfTargetMSIS via a reduction from MLWE in the Quantum Random Oracle Model (QROM). Our proof uses recently developed techniques in quantum reprogramming and rewinding. A central part of our approach is a proof that a certain hash function, derived from the MSIS problem, is collapsing. From this approach, we deduce a new security proof for Dilithium under appropriate parameter settings. Compared to the previous work by Kiltz, Lyubashevsky, and Schaffner (EUROCRYPT 2018) that gave the only other rigorous security proof for a variant of Dilithium, our proof has the advantage of being applicable under the condition q = 1 mod 2n, where q denotes the modulus and n the dimension of the underlying algebraic ring. This condition is part of the original Dilithium proposal and is crucial for the efficient implementation of the scheme. We provide new secure parameter sets for Dilithium under the condition q = 1 mod 2n, finding that our public key size and signature size are about 2.9 times and 1.3 times larger, respectively, than those proposed by Kiltz et al. at the same security level.

摘要: 随着量子计算硬件的最新进展，美国国家标准与技术研究所(NIST)正在对能够抵抗量子对手攻击的密码协议进行标准化。NIST选择的主要数字签名方案是Crystal-Dilithium。该方案的难易程度基于三个计算问题的难易程度：带错误的模块学习(MLWE)、模块短整数解(MSIS)和自目标短整数解。MLWE和MSIS已经得到了很好的研究，并被广泛认为是安全的。然而，SelfTargetMSIS是新颖的，尽管经典上和MSIS一样难，但它的量子硬度尚不清楚。本文通过对量子随机Oracle模型(QROM)中MLWE的简化，首次证明了自目标MSIS的硬度。我们的证明使用了最近发展起来的量子重编程和倒带技术。我们方法的一个核心部分是证明从MSIS问题派生的某个散列函数正在崩溃。通过这种方法，我们在适当的参数设置下，给出了Dilithium的一个新的安全证明。与Kiltz，Lyubashevsky和Schaffner(Eurocrypt 2018)之前的工作相比，我们的证明具有在q=1mod 2n的条件下适用的优点，其中q表示基础代数环的模，n表示基础代数环的维度。这一条件是最初的Dilithium提议的一部分，对该计划的有效实施至关重要。在Q=1 mod 2n的条件下，我们给出了Dilithium的新的安全参数集，发现我们的公钥长度和签名长度分别是Kiltz等人提出的安全参数集的2.9倍和1.3倍。在相同的安全级别。



## **15. Multi-Agent Reinforcement Learning for Assessing False-Data Injection Attacks on Transportation Networks**

基于多智能体强化学习的交通网络虚假数据注入攻击评估 cs.AI

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2312.14625v2) [paper-pdf](http://arxiv.org/pdf/2312.14625v2)

**Authors**: Taha Eghtesad, Sirui Li, Yevgeniy Vorobeychik, Aron Laszka

**Abstract**: The increasing reliance of drivers on navigation applications has made transportation networks more susceptible to data-manipulation attacks by malicious actors. Adversaries may exploit vulnerabilities in the data collection or processing of navigation services to inject false information, and to thus interfere with the drivers' route selection. Such attacks can significantly increase traffic congestions, resulting in substantial waste of time and resources, and may even disrupt essential services that rely on road networks. To assess the threat posed by such attacks, we introduce a computational framework to find worst-case data-injection attacks against transportation networks. First, we devise an adversarial model with a threat actor who can manipulate drivers by increasing the travel times that they perceive on certain roads. Then, we employ hierarchical multi-agent reinforcement learning to find an approximate optimal adversarial strategy for data manipulation. We demonstrate the applicability of our approach through simulating attacks on the Sioux Falls, ND network topology.

摘要: 司机越来越依赖导航应用程序，这使得交通网络更容易受到恶意行为者的数据操纵攻击。攻击者可能会利用导航服务的数据收集或处理中的漏洞来注入虚假信息，从而干扰司机的路线选择。此类攻击可能会显著加剧交通拥堵，导致大量时间和资源的浪费，甚至可能扰乱依赖道路网络的基本服务。为了评估这类攻击造成的威胁，我们引入了一个计算框架来发现针对交通网络的最坏情况下的数据注入攻击。首先，我们设计了一个带有威胁参与者的对抗性模型，该威胁参与者可以通过增加司机在某些道路上感知的旅行时间来操纵司机。然后，我们使用分层多智能体强化学习来寻找数据操作的近似最优对抗策略。通过模拟对苏福尔斯网络拓扑结构的攻击，验证了该方法的适用性。



## **16. Improving Adversarial Training using Vulnerability-Aware Perturbation Budget**

利用脆弱性感知扰动预算改进对抗性训练 cs.LG

19 pages, 2 figures

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.04070v1) [paper-pdf](http://arxiv.org/pdf/2403.04070v1)

**Authors**: Olukorede Fakorede, Modeste Atsague, Jin Tian

**Abstract**: Adversarial Training (AT) effectively improves the robustness of Deep Neural Networks (DNNs) to adversarial attacks. Generally, AT involves training DNN models with adversarial examples obtained within a pre-defined, fixed perturbation bound. Notably, individual natural examples from which these adversarial examples are crafted exhibit varying degrees of intrinsic vulnerabilities, and as such, crafting adversarial examples with fixed perturbation radius for all instances may not sufficiently unleash the potency of AT. Motivated by this observation, we propose two simple, computationally cheap vulnerability-aware reweighting functions for assigning perturbation bounds to adversarial examples used for AT, named Margin-Weighted Perturbation Budget (MWPB) and Standard-Deviation-Weighted Perturbation Budget (SDWPB). The proposed methods assign perturbation radii to individual adversarial samples based on the vulnerability of their corresponding natural examples. Experimental results show that the proposed methods yield genuine improvements in the robustness of AT algorithms against various adversarial attacks.

摘要: 对抗训练(AT)有效地提高了深度神经网络(DNN)对对抗攻击的稳健性。通常，AT涉及用在预定义的固定扰动范围内获得的对抗性样本来训练DNN模型。值得注意的是，制作这些对抗性例子的个别自然例子表现出不同程度的内在脆弱性，因此，为所有实例制作具有固定扰动半径的对抗性例子可能不能充分释放AT的效力。基于这一观察结果，我们提出了两个简单的、计算上廉价的脆弱性感知重加权函数，用于为AT中的对抗性例子分配扰动界，分别称为差值加权扰动预算(MWPB)和标准差加权扰动预算(SDWPB)。所提出的方法根据单个对抗性样本的自然样本的脆弱性为其分配扰动半径。实验结果表明，该方法确实提高了AT算法对各种敌意攻击的稳健性。



## **17. Improving Adversarial Attacks on Latent Diffusion Model**

基于潜在扩散模型的对抗性攻击改进 cs.CV

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2310.04687v3) [paper-pdf](http://arxiv.org/pdf/2310.04687v3)

**Authors**: Boyang Zheng, Chumeng Liang, Xiaoyu Wu, Yan Liu

**Abstract**: Adversarial attacks on Latent Diffusion Model (LDM), the state-of-the-art image generative model, have been adopted as effective protection against malicious finetuning of LDM on unauthorized images. We show that these attacks add an extra error to the score function of adversarial examples predicted by LDM. LDM finetuned on these adversarial examples learns to lower the error by a bias, from which the model is attacked and predicts the score function with biases.   Based on the dynamics, we propose to improve the adversarial attack on LDM by Attacking with Consistent score-function Errors (ACE). ACE unifies the pattern of the extra error added to the predicted score function. This induces the finetuned LDM to learn the same pattern as a bias in predicting the score function. We then introduce a well-crafted pattern to improve the attack. Our method outperforms state-of-the-art methods in adversarial attacks on LDM.

摘要: 针对当前最先进的图像生成模型--潜在扩散模型(LDM)的敌意攻击已被用作对未经授权的图像进行恶意微调的有效保护。我们证明了这些攻击给LDM预测的对抗性例子的得分函数增加了额外的误差。在这些对抗性例子上精调的LDM学习通过偏差来降低误差，由此对模型进行攻击并预测带有偏差的得分函数。在此基础上，提出了利用一致得分函数错误(ACE)攻击来提高对LDM的对抗性攻击。ACE统一了添加到预测得分函数的额外误差的模式。这导致精调的LDM在预测得分函数时学习与偏差相同的模式。然后，我们引入一个精心设计的模式来改进攻击。在对LDM的对抗性攻击中，我们的方法优于最先进的方法。



## **18. A Survey on Adversarial Contention Resolution**

对抗性争议解决机制研究综述 cs.DC

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03876v1) [paper-pdf](http://arxiv.org/pdf/2403.03876v1)

**Authors**: Ioana Banicescu, Trisha Chakraborty, Seth Gilbert, Maxwell Young

**Abstract**: Contention resolution addresses the challenge of coordinating access by multiple processes to a shared resource such as memory, disk storage, or a communication channel. Originally spurred by challenges in database systems and bus networks, contention resolution has endured as an important abstraction for resource sharing, despite decades of technological change. Here, we survey the literature on resolving worst-case contention, where the number of processes and the time at which each process may start seeking access to the resource is dictated by an adversary. We highlight the evolution of contention resolution, where new concerns -- such as security, quality of service, and energy efficiency -- are motivated by modern systems. These efforts have yielded insights into the limits of randomized and deterministic approaches, as well as the impact of different model assumptions such as global clock synchronization, knowledge of the number of processors, feedback from access attempts, and attacks on the availability of the shared resource.

摘要: 争用解决方案解决了协调多个进程对共享资源(如内存、磁盘存储或通信通道)的访问的挑战。争用解决最初是由数据库系统和总线网络中的挑战推动的，尽管经历了几十年的技术变革，但它作为资源共享的一个重要抽象概念一直存在。在这里，我们回顾了关于解决最坏情况争用的文献，在这种情况下，进程的数量和每个进程可能开始寻求访问资源的时间由对手决定。我们重点介绍争用解决方案的演变，其中新的关注点--如安全性、服务质量和能源效率--是由现代系统驱动的。这些努力使人们深入了解了随机化和确定性方法的局限性，以及不同模型假设的影响，如全球时钟同步、处理器数量的知识、访问尝试的反馈以及对共享资源可用性的攻击。



## **19. Effect of Ambient-Intrinsic Dimension Gap on Adversarial Vulnerability**

环境-本征维度差距对对手脆弱性的影响 cs.LG

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03967v1) [paper-pdf](http://arxiv.org/pdf/2403.03967v1)

**Authors**: Rajdeep Haldar, Yue Xing, Qifan Song

**Abstract**: The existence of adversarial attacks on machine learning models imperceptible to a human is still quite a mystery from a theoretical perspective. In this work, we introduce two notions of adversarial attacks: natural or on-manifold attacks, which are perceptible by a human/oracle, and unnatural or off-manifold attacks, which are not. We argue that the existence of the off-manifold attacks is a natural consequence of the dimension gap between the intrinsic and ambient dimensions of the data. For 2-layer ReLU networks, we prove that even though the dimension gap does not affect generalization performance on samples drawn from the observed data space, it makes the clean-trained model more vulnerable to adversarial perturbations in the off-manifold direction of the data space. Our main results provide an explicit relationship between the $\ell_2,\ell_{\infty}$ attack strength of the on/off-manifold attack and the dimension gap.

摘要: 从理论上讲，对人类无法察觉的机器学习模型存在敌意攻击仍然是一个相当神秘的问题。在这项工作中，我们引入了两个对抗性攻击的概念：人类/先知可以感知的自然或流形上的攻击，以及不可察觉的非自然或非流形攻击。我们认为，非流形攻击的存在是数据的内在维度和环境维度之间存在维度差距的自然结果。对于两层RELU网络，我们证明了尽管维度间隙不影响对来自观测数据空间的样本的泛化性能，但它使得干净训练的模型更容易受到数据空间非流形方向上的对抗性扰动。我们的主要结果提供了On/Off流形攻击的攻击强度与维度间隙之间的显式关系。



## **20. Neural Exec: Learning (and Learning from) Execution Triggers for Prompt Injection Attacks**

NeuroExec：学习(和学习)快速注入攻击的执行触发器 cs.CR

v0.1

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03792v1) [paper-pdf](http://arxiv.org/pdf/2403.03792v1)

**Authors**: Dario Pasquini, Martin Strohmeier, Carmela Troncoso

**Abstract**: We introduce a new family of prompt injection attacks, termed Neural Exec. Unlike known attacks that rely on handcrafted strings (e.g., "Ignore previous instructions and..."), we show that it is possible to conceptualize the creation of execution triggers as a differentiable search problem and use learning-based methods to autonomously generate them.   Our results demonstrate that a motivated adversary can forge triggers that are not only drastically more effective than current handcrafted ones but also exhibit inherent flexibility in shape, properties, and functionality. In this direction, we show that an attacker can design and generate Neural Execs capable of persisting through multi-stage preprocessing pipelines, such as in the case of Retrieval-Augmented Generation (RAG)-based applications. More critically, our findings show that attackers can produce triggers that deviate markedly in form and shape from any known attack, sidestepping existing blacklist-based detection and sanitation approaches.

摘要: 我们介绍了一类新的快速注入攻击，称为神经执行攻击。与依赖手工创建的字符串(例如，“忽略先前的指令和...”)的已知攻击不同，我们展示了将创建执行触发器概念化为可区分的搜索问题并使用基于学习的方法自主生成它们是可能的。我们的结果表明，有动机的对手可以伪造触发器，不仅比目前手工制作的触发器有效得多，而且在形状、属性和功能上表现出固有的灵活性。在这个方向上，我们展示了攻击者可以设计和生成能够在多阶段预处理管道中持久存在的神经Execs，例如在基于检索-增强生成(RAG)的应用程序的情况下。更关键的是，我们的发现表明，攻击者可以产生在形式和形状上与任何已知攻击显著偏离的触发器，绕过现有的基于黑名单的检测和卫生方法。



## **21. PPTC-R benchmark: Towards Evaluating the Robustness of Large Language Models for PowerPoint Task Completion**

PPTC-R基准：评估用于PowerPoint任务完成的大型语言模型的健壮性 cs.CL

LLM evaluation, Multi-turn, Multi-language, Multi-modal benchmark

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03788v1) [paper-pdf](http://arxiv.org/pdf/2403.03788v1)

**Authors**: Zekai Zhang, Yiduo Guo, Yaobo Liang, Dongyan Zhao, Nan Duan

**Abstract**: The growing dependence on Large Language Models (LLMs) for finishing user instructions necessitates a comprehensive understanding of their robustness to complex task completion in real-world situations. To address this critical need, we propose the PowerPoint Task Completion Robustness benchmark (PPTC-R) to measure LLMs' robustness to the user PPT task instruction and software version. Specifically, we construct adversarial user instructions by attacking user instructions at sentence, semantic, and multi-language levels. To assess the robustness of Language Models to software versions, we vary the number of provided APIs to simulate both the newest version and earlier version settings. Subsequently, we test 3 closed-source and 4 open-source LLMs using a benchmark that incorporates these robustness settings, aiming to evaluate how deviations impact LLMs' API calls for task completion. We find that GPT-4 exhibits the highest performance and strong robustness in our benchmark, particularly in the version update and the multilingual settings. However, we find that all LLMs lose their robustness when confronted with multiple challenges (e.g., multi-turn) simultaneously, leading to significant performance drops. We further analyze the robustness behavior and error reasons of LLMs in our benchmark, which provide valuable insights for researchers to understand the LLM's robustness in task completion and develop more robust LLMs and agents. We release the code and data at \url{https://github.com/ZekaiGalaxy/PPTCR}.

摘要: 越来越多地依赖大型语言模型(LLM)来完成用户指令，这就需要全面了解它们在现实世界中完成复杂任务时的健壮性。为了解决这一关键需求，我们提出了PowerPoint任务完成健壮性基准(PPTC-R)来测量LLMS对用户PPT任务指令和软件版本的健壮性。具体地说，我们通过在句子、语义和多语言级别攻击用户指令来构建对抗性用户指令。为了评估语言模型对软件版本的稳健性，我们改变了提供的API的数量，以模拟最新版本和较早版本的设置。随后，我们使用结合了这些健壮性设置的基准测试了3个封闭源代码LLMS和4个开放源代码LLMS，旨在评估偏差如何影响LLMS完成任务的API调用。我们发现GPT-4在我们的基准测试中表现出了最高的性能和强大的健壮性，特别是在版本更新和多语言设置方面。然而，我们发现，当同时面对多个挑战(例如，多回合)时，所有的LLM都失去了它们的健壮性，导致性能显著下降。我们进一步分析了LLM在基准测试中的健壮性行为和错误原因，这为研究人员理解LLM在任务完成时的健壮性以及开发更健壮的LLM和代理提供了有价值的见解。我们将代码和数据发布到\url{https://github.com/ZekaiGalaxy/PPTCR}.



## **22. Verification of Neural Networks' Global Robustness**

神经网络的全局健壮性验证 cs.LG

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2402.19322v2) [paper-pdf](http://arxiv.org/pdf/2402.19322v2)

**Authors**: Anan Kabaha, Dana Drachsler-Cohen

**Abstract**: Neural networks are successful in various applications but are also susceptible to adversarial attacks. To show the safety of network classifiers, many verifiers have been introduced to reason about the local robustness of a given input to a given perturbation. While successful, local robustness cannot generalize to unseen inputs. Several works analyze global robustness properties, however, neither can provide a precise guarantee about the cases where a network classifier does not change its classification. In this work, we propose a new global robustness property for classifiers aiming at finding the minimal globally robust bound, which naturally extends the popular local robustness property for classifiers. We introduce VHAGaR, an anytime verifier for computing this bound. VHAGaR relies on three main ideas: encoding the problem as a mixed-integer programming and pruning the search space by identifying dependencies stemming from the perturbation or the network's computation and generalizing adversarial attacks to unknown inputs. We evaluate VHAGaR on several datasets and classifiers and show that, given a three hour timeout, the average gap between the lower and upper bound on the minimal globally robust bound computed by VHAGaR is 1.9, while the gap of an existing global robustness verifier is 154.7. Moreover, VHAGaR is 130.6x faster than this verifier. Our results further indicate that leveraging dependencies and adversarial attacks makes VHAGaR 78.6x faster.

摘要: 神经网络在各种应用中都很成功，但也容易受到对抗性攻击。为了表明网络分类器的安全性，已经引入了许多验证器来推理给定输入对给定扰动的局部稳健性。虽然取得了成功，但局部稳健性不能推广到看不见的输入。然而，一些工作分析了全局健壮性，但都不能提供关于网络分类器不改变其分类的情况的精确保证。在这项工作中，我们提出了一种新的分类器的全局稳健性，旨在寻找最小的全局稳健界，这自然地扩展了流行的分类器的局部稳健性。我们介绍了VHAGaR，一个计算这个界的随时验证器。VHAGaR依赖于三个主要思想：将问题编码为混合整数规划，通过识别源于扰动或网络计算的依赖来削减搜索空间，以及将敌意攻击推广到未知输入。我们在几个数据集和分类器上对VHAGaR进行了评估，结果表明，在超时3小时的情况下，VHAGaR计算的最小全局健壮界的上下界之间的平均差距为1.9%，而现有的全局健壮性验证器的差距为154.7。此外，VHAGaR比该验证器快130.6倍。我们的结果进一步表明，利用依赖关系和对抗性攻击使VHAGaR的速度提高了78.6倍。



## **23. Simplified PCNet with Robustness**

具有健壮性的简化PCNet cs.LG

10 pages, 3 figures

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03676v1) [paper-pdf](http://arxiv.org/pdf/2403.03676v1)

**Authors**: Bingheng Li, Xuanting Xie, Haoxiang Lei, Ruiyi Fang, Zhao Kang

**Abstract**: Graph Neural Networks (GNNs) have garnered significant attention for their success in learning the representation of homophilic or heterophilic graphs. However, they cannot generalize well to real-world graphs with different levels of homophily. In response, the Possion-Charlier Network (PCNet) \cite{li2024pc}, the previous work, allows graph representation to be learned from heterophily to homophily. Although PCNet alleviates the heterophily issue, there remain some challenges in further improving the efficacy and efficiency. In this paper, we simplify PCNet and enhance its robustness. We first extend the filter order to continuous values and reduce its parameters. Two variants with adaptive neighborhood sizes are implemented. Theoretical analysis shows our model's robustness to graph structure perturbations or adversarial attacks. We validate our approach through semi-supervised learning tasks on various datasets representing both homophilic and heterophilic graphs.

摘要: 图神经网络(GNN)因其在学习同亲图或异亲图的表示方面的成功而受到极大的关注。然而，它们不能很好地推广到具有不同同质性水平的真实世界的图。作为回应，Possion-Charlier Network(PCNet)引用了以前的工作{li2024pc}，允许从异形到同形学习图表示。虽然PCNet缓解了异质性问题，但在进一步提高疗效和效率方面仍存在一些挑战。在本文中，我们简化了PCNet，增强了它的健壮性。我们首先将滤波阶扩展到连续值，并对其参数进行降阶。实现了两种具有自适应邻域大小的变体。理论分析表明，该模型对图结构扰动或敌意攻击具有较强的稳健性。我们通过在不同数据集上的半监督学习任务来验证我们的方法，这些数据集既代表同嗜图，也代表异嗜图。



## **24. Adversarial Infrared Geometry: Using Geometry to Perform Adversarial Attack against Infrared Pedestrian Detectors**

对抗红外几何：利用几何对红外行人探测器进行对抗攻击 cs.CV

arXiv admin note: text overlap with arXiv:2312.14217 by other authors

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03674v1) [paper-pdf](http://arxiv.org/pdf/2403.03674v1)

**Authors**: Kalibinuer Tiliwalidi

**Abstract**: Currently, infrared imaging technology enjoys widespread usage, with infrared object detection technology experiencing a surge in prominence. While previous studies have delved into physical attacks on infrared object detectors, the implementation of these techniques remains complex. For instance, some approaches entail the use of bulb boards or infrared QR suits as perturbations to execute attacks, which entail costly optimization and cumbersome deployment processes. Other methodologies involve the utilization of irregular aerogel as physical perturbations for infrared attacks, albeit at the expense of optimization expenses and perceptibility issues. In this study, we propose a novel infrared physical attack termed Adversarial Infrared Geometry (\textbf{AdvIG}), which facilitates efficient black-box query attacks by modeling diverse geometric shapes (lines, triangles, ellipses) and optimizing their physical parameters using Particle Swarm Optimization (PSO). Extensive experiments are conducted to evaluate the effectiveness, stealthiness, and robustness of AdvIG. In digital attack experiments, line, triangle, and ellipse patterns achieve attack success rates of 93.1\%, 86.8\%, and 100.0\%, respectively, with average query times of 71.7, 113.1, and 2.57, respectively, thereby confirming the efficiency of AdvIG. Physical attack experiments are conducted to assess the attack success rate of AdvIG at different distances. On average, the line, triangle, and ellipse achieve attack success rates of 61.1\%, 61.2\%, and 96.2\%, respectively. Further experiments are conducted to comprehensively analyze AdvIG, including ablation experiments, transfer attack experiments, and adversarial defense mechanisms. Given the superior performance of our method as a simple and efficient black-box adversarial attack in both digital and physical environments, we advocate for widespread attention to AdvIG.

摘要: 目前，红外成像技术得到了广泛的应用，红外目标检测技术的重要性也在激增。虽然之前的研究已经深入研究了对红外目标探测器的物理攻击，但这些技术的实现仍然很复杂。例如，一些方法需要使用灯泡板或红外二维码作为扰动来执行攻击，这需要昂贵的优化和繁琐的部署过程。其他方法包括利用不规则气凝胶作为红外攻击的物理扰动，尽管代价是优化费用和感知问题。在这项研究中，我们提出了一种新的红外物理攻击，称为对抗红外几何(\extbf{AdvIG})，它通过对不同的几何形状(直线、三角形、椭圆)进行建模并使用粒子群优化算法(PSO)来优化它们的物理参数，从而为高效的黑盒查询攻击提供了便利。为了评估AdvIG的有效性、隐蔽性和健壮性，进行了大量的实验。在数字攻击实验中，直线、三角形和椭圆模式的攻击成功率分别为93.1%、86.8%和100.0，平均查询次数分别为71.7%、113.1次和2.57%，从而验证了该算法的有效性。通过物理攻击实验评估了AdvIG在不同距离下的攻击成功率。平均而言，直线、三角形和椭圆的攻击成功率分别为61.1%、61.2%和96.2%。进一步的实验对AdvIG进行了综合分析，包括烧蚀实验、传输攻击实验和对抗防御机制。鉴于我们的方法在数字和物理环境中作为一种简单而高效的黑盒对抗性攻击的卓越性能，我们主张广泛关注AdvIG。



## **25. Lotto: Secure Participant Selection against Adversarial Servers in Federated Learning**

乐透：联合学习中对抗敌意服务器的安全参与者选择 cs.CR

This article has been accepted to USENIX Security '24

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2401.02880v2) [paper-pdf](http://arxiv.org/pdf/2401.02880v2)

**Authors**: Zhifeng Jiang, Peng Ye, Shiqi He, Wei Wang, Ruichuan Chen, Bo Li

**Abstract**: In Federated Learning (FL), common privacy-enhancing techniques, such as secure aggregation and distributed differential privacy, rely on the critical assumption of an honest majority among participants to withstand various attacks. In practice, however, servers are not always trusted, and an adversarial server can strategically select compromised clients to create a dishonest majority, thereby undermining the system's security guarantees. In this paper, we present Lotto, an FL system that addresses this fundamental, yet underexplored issue by providing secure participant selection against an adversarial server. Lotto supports two selection algorithms: random and informed. To ensure random selection without a trusted server, Lotto enables each client to autonomously determine their participation using verifiable randomness. For informed selection, which is more vulnerable to manipulation, Lotto approximates the algorithm by employing random selection within a refined client pool. Our theoretical analysis shows that Lotto effectively aligns the proportion of server-selected compromised participants with the base rate of dishonest clients in the population. Large-scale experiments further reveal that Lotto achieves time-to-accuracy performance comparable to that of insecure selection methods, indicating a low computational overhead for secure selection.

摘要: 在联邦学习(FL)中，常见的隐私增强技术，如安全聚合和分布式差异隐私，依赖于参与者之间诚实多数的关键假设来抵御各种攻击。然而，在实践中，服务器并不总是可信的，敌意服务器可以策略性地选择受攻击的客户端来制造不诚实的多数，从而破坏系统的安全保证。在本文中，我们提出了乐透，一个FL系统，解决了这个基本的，但探索不足的问题，通过提供安全的参与者选择对抗敌对的服务器。乐透支持两种选择算法：随机和通知。为了确保在没有可信服务器的情况下随机选择，乐透使每个客户端能够使用可验证的随机性自主确定他们的参与。对于更容易受到操纵的知情选择，乐透通过在改进的客户机池中使用随机选择来近似算法。我们的理论分析表明，乐透有效地将服务器选择的受攻击参与者的比例与人口中不诚实客户端的基本比率保持一致。大规模实验进一步表明，乐透算法的时间精度性能与非安全选择方法相当，表明安全选择方法具有较低的计算开销。



## **26. Noise-BERT: A Unified Perturbation-Robust Framework with Noise Alignment Pre-training for Noisy Slot Filling Task**

Noise-BERT：一种带噪声对齐预训练的统一扰动-稳健框架 cs.CL

Accepted by ICASSP 2024

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2402.14494v3) [paper-pdf](http://arxiv.org/pdf/2402.14494v3)

**Authors**: Jinxu Zhao, Guanting Dong, Yueyan Qiu, Tingfeng Hui, Xiaoshuai Song, Daichi Guo, Weiran Xu

**Abstract**: In a realistic dialogue system, the input information from users is often subject to various types of input perturbations, which affects the slot-filling task. Although rule-based data augmentation methods have achieved satisfactory results, they fail to exhibit the desired generalization when faced with unknown noise disturbances. In this study, we address the challenges posed by input perturbations in slot filling by proposing Noise-BERT, a unified Perturbation-Robust Framework with Noise Alignment Pre-training. Our framework incorporates two Noise Alignment Pre-training tasks: Slot Masked Prediction and Sentence Noisiness Discrimination, aiming to guide the pre-trained language model in capturing accurate slot information and noise distribution. During fine-tuning, we employ a contrastive learning loss to enhance the semantic representation of entities and labels. Additionally, we introduce an adversarial attack training strategy to improve the model's robustness. Experimental results demonstrate the superiority of our proposed approach over state-of-the-art models, and further analysis confirms its effectiveness and generalization ability.

摘要: 在现实对话系统中，来自用户的输入信息经常受到各种类型的输入扰动，这影响了空缺填充任务。尽管基于规则的数据增强方法取得了令人满意的结果，但它们在面对未知噪声干扰时不能表现出预期的泛化能力。在这项研究中，我们通过提出Noise-BERT来解决输入扰动在时隙填充中所带来的挑战，这是一个带有噪声对齐预训练的统一扰动-稳健框架。该框架集成了两个噪声对齐预训练任务：时隙掩蔽预测和句子噪声识别，旨在指导预先训练的语言模型捕捉准确的时隙信息和噪声分布。在微调过程中，我们采用对比学习损失来增强实体和标签的语义表示。此外，我们还引入了对抗性攻击训练策略来提高模型的稳健性。实验结果证明了该方法的优越性，进一步的分析证实了该方法的有效性和泛化能力。



## **27. TTPXHunter: Actionable Threat Intelligence Extraction as TTPs form Finished Cyber Threat Reports**

TTPXHunter：在TTP形成已完成的网络威胁报告时提取可操作的威胁情报 cs.CR

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.03267v1) [paper-pdf](http://arxiv.org/pdf/2403.03267v1)

**Authors**: Nanda Rani, Bikash Saha, Vikas Maurya, Sandeep Kumar Shukla

**Abstract**: Understanding the modus operandi of adversaries aids organizations in employing efficient defensive strategies and sharing intelligence in the community. This knowledge is often present in unstructured natural language text within threat analysis reports. A translation tool is needed to interpret the modus operandi explained in the sentences of the threat report and translate it into a structured format. This research introduces a methodology named TTPXHunter for the automated extraction of threat intelligence in terms of Tactics, Techniques, and Procedures (TTPs) from finished cyber threat reports. It leverages cyber domain-specific state-of-the-art natural language processing (NLP) to augment sentences for minority class TTPs and refine pinpointing the TTPs in threat analysis reports significantly. The knowledge of threat intelligence in terms of TTPs is essential for comprehensively understanding cyber threats and enhancing detection and mitigation strategies. We create two datasets: an augmented sentence-TTP dataset of 39,296 samples and a 149 real-world cyber threat intelligence report-to-TTP dataset. Further, we evaluate TTPXHunter on the augmented sentence dataset and the cyber threat reports. The TTPXHunter achieves the highest performance of 92.42% f1-score on the augmented dataset, and it also outperforms existing state-of-the-art solutions in TTP extraction by achieving an f1-score of 97.09% when evaluated over the report dataset. TTPXHunter significantly improves cybersecurity threat intelligence by offering quick, actionable insights into attacker behaviors. This advancement automates threat intelligence analysis, providing a crucial tool for cybersecurity professionals fighting cyber threats.

摘要: 了解对手的作案手法有助于组织采用有效的防御策略，并在社区中分享情报。这种知识通常出现在威胁分析报告中的非结构化自然语言文本中。需要一个翻译工具来解释威胁报告句子中解释的工作方式，并将其翻译成结构化格式。本研究介绍了一种名为TTPXHunter的方法，用于从已完成的网络威胁报告中自动提取策略、技术和过程(TTP)方面的威胁情报。它利用特定于网络领域的最先进的自然语言处理(NLP)来增加少数族裔类TTP的句子，并显著细化威胁分析报告中的TTP。TTP方面的威胁情报知识对于全面了解网络威胁和加强检测和缓解战略至关重要。我们创建了两个数据集：一个包含39,296个样本的增强句-TTP数据集，以及149个真实世界网络威胁情报报告到TTP的数据集。此外，我们在增加句子数据集和网络威胁报告上对TTPXHunter进行了评估。TTPXHunter在增强的数据集上获得了92.42%的F1分数的最高性能，在TTP提取方面也超过了现有的最先进的解决方案，在报告数据集上的F1分数达到了97.09%。TTPXHunter通过提供对攻击者行为的快速、可操作的洞察，显著提高了网络安全威胁情报。这一进步使威胁情报分析自动化，为应对网络威胁的网络安全专业人员提供了一个重要工具。



## **28. Attacks on Node Attributes in Graph Neural Networks**

图神经网络中节点属性的攻击 cs.SI

Accepted to AAAI 2024 AICS workshop

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2402.12426v2) [paper-pdf](http://arxiv.org/pdf/2402.12426v2)

**Authors**: Ying Xu, Michael Lanier, Anindya Sarkar, Yevgeniy Vorobeychik

**Abstract**: Graphs are commonly used to model complex networks prevalent in modern social media and literacy applications. Our research investigates the vulnerability of these graphs through the application of feature based adversarial attacks, focusing on both decision time attacks and poisoning attacks. In contrast to state of the art models like Net Attack and Meta Attack, which target node attributes and graph structure, our study specifically targets node attributes. For our analysis, we utilized the text dataset Hellaswag and graph datasets Cora and CiteSeer, providing a diverse basis for evaluation. Our findings indicate that decision time attacks using Projected Gradient Descent (PGD) are more potent compared to poisoning attacks that employ Mean Node Embeddings and Graph Contrastive Learning strategies. This provides insights for graph data security, pinpointing where graph-based models are most vulnerable and thereby informing the development of stronger defense mechanisms against such attacks.

摘要: 图通常用于对现代社交媒体和识字应用中普遍存在的复杂网络进行建模。我们的研究通过应用基于特征的对抗性攻击来研究这些图的脆弱性，重点研究了决策时攻击和中毒攻击。与网络攻击和元攻击等针对节点属性和图结构的最新模型不同，我们的研究专门针对节点属性。在我们的分析中，我们使用了文本数据集Hellaswag和图形数据集Cora和CiteSeer，为评估提供了多样化的基础。我们的发现表明，与使用均值节点嵌入和图对比学习策略的中毒攻击相比，使用投影梯度下降(PGD)的决策时间攻击更有效。这为图形数据安全提供了洞察力，准确地指出了基于图形的模型最易受攻击的位置，从而为开发针对此类攻击的更强大的防御机制提供了信息。



## **29. Mitigating Label Flipping Attacks in Malicious URL Detectors Using Ensemble Trees**

利用集成树减轻恶意URL检测器中的标签翻转攻击 cs.CR

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02995v1) [paper-pdf](http://arxiv.org/pdf/2403.02995v1)

**Authors**: Ehsan Nowroozi, Nada Jadalla, Samaneh Ghelichkhani, Alireza Jolfaei

**Abstract**: Malicious URLs provide adversarial opportunities across various industries, including transportation, healthcare, energy, and banking which could be detrimental to business operations. Consequently, the detection of these URLs is of crucial importance; however, current Machine Learning (ML) models are susceptible to backdoor attacks. These attacks involve manipulating a small percentage of training data labels, such as Label Flipping (LF), which changes benign labels to malicious ones and vice versa. This manipulation results in misclassification and leads to incorrect model behavior. Therefore, integrating defense mechanisms into the architecture of ML models becomes an imperative consideration to fortify against potential attacks.   The focus of this study is on backdoor attacks in the context of URL detection using ensemble trees. By illuminating the motivations behind such attacks, highlighting the roles of attackers, and emphasizing the critical importance of effective defense strategies, this paper contributes to the ongoing efforts to fortify ML models against adversarial threats within the ML domain in network security. We propose an innovative alarm system that detects the presence of poisoned labels and a defense mechanism designed to uncover the original class labels with the aim of mitigating backdoor attacks on ensemble tree classifiers. We conducted a case study using the Alexa and Phishing Site URL datasets and showed that LF attacks can be addressed using our proposed defense mechanism. Our experimental results prove that the LF attack achieved an Attack Success Rate (ASR) between 50-65% within 2-5%, and the innovative defense method successfully detected poisoned labels with an accuracy of up to 100%.

摘要: 恶意URL为包括交通、医疗、能源和银行在内的多个行业提供了敌意机会，可能会对业务运营造成不利影响。因此，对这些URL的检测至关重要；然而，当前的机器学习(ML)模型容易受到后门攻击。这些攻击涉及操纵一小部分训练数据标签，例如标签翻转(LF)，它将良性标签更改为恶意标签，反之亦然。这种操作会导致错误分类，并导致不正确的模型行为。因此，将防御机制集成到ML模型的体系结构中成为防御潜在攻击的当务之急。这项研究的重点是利用集成树检测URL上下文中的后门攻击。通过阐明此类攻击背后的动机，突出攻击者的作用，并强调有效防御策略的关键重要性，本文有助于在网络安全中加强ML域内对抗对手威胁的ML模型的持续努力。我们提出了一种创新的警报系统来检测有毒标签的存在，并提出了一种旨在发现原始类别标签的防御机制，目的是减少对集成树分类器的后门攻击。我们使用Alexa和网络钓鱼网站的URL数据集进行了一个案例研究，并表明可以使用我们提出的防御机制来应对LF攻击。实验结果表明，LF攻击在2%-5%的范围内达到了50%-65%的攻击成功率(ASR)，新的防御方法成功地检测到了有毒标签，准确率达到100%。



## **30. Federated Learning Under Attack: Exposing Vulnerabilities through Data Poisoning Attacks in Computer Networks**

攻击下的联合学习：通过计算机网络中的数据中毒攻击暴露漏洞 cs.CR

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02983v1) [paper-pdf](http://arxiv.org/pdf/2403.02983v1)

**Authors**: Ehsan Nowroozi, Imran Haider, Rahim Taheri, Mauro Conti

**Abstract**: Federated Learning (FL) is a machine learning (ML) approach that enables multiple decentralized devices or edge servers to collaboratively train a shared model without exchanging raw data. During the training and sharing of model updates between clients and servers, data and models are susceptible to different data-poisoning attacks.   In this study, our motivation is to explore the severity of data poisoning attacks in the computer network domain because they are easy to implement but difficult to detect. We considered two types of data-poisoning attacks, label flipping (LF) and feature poisoning (FP), and applied them with a novel approach. In LF, we randomly flipped the labels of benign data and trained the model on the manipulated data. For FP, we randomly manipulated the highly contributing features determined using the Random Forest algorithm. The datasets used in this experiment were CIC and UNSW related to computer networks. We generated adversarial samples using the two attacks mentioned above, which were applied to a small percentage of datasets. Subsequently, we trained and tested the accuracy of the model on adversarial datasets. We recorded the results for both benign and manipulated datasets and observed significant differences between the accuracy of the models on different datasets. From the experimental results, it is evident that the LF attack failed, whereas the FP attack showed effective results, which proved its significance in fooling a server. With a 1% LF attack on the CIC, the accuracy was approximately 0.0428 and the ASR was 0.9564; hence, the attack is easily detectable, while with a 1% FP attack, the accuracy and ASR were both approximately 0.9600, hence, FP attacks are difficult to detect. We repeated the experiment with different poisoning percentages.

摘要: 联合学习(FL)是一种机器学习(ML)方法，它使多个分散的设备或边缘服务器能够在不交换原始数据的情况下协作地训练共享模型。在客户端和服务器之间模型更新的训练和共享过程中，数据和模型容易受到不同的数据中毒攻击。在这项研究中，我们的动机是探索计算机网络领域中的数据中毒攻击的严重性，因为它们易于实现但难以检测。我们考虑了两种类型的数据中毒攻击，标签翻转(LF)和特征中毒(FP)，并将它们应用于一种新的方法。在LF中，我们随机翻转良性数据的标签，并在被操纵的数据上训练模型。对于FP，我们随机处理使用随机森林算法确定的高贡献特征。本实验中使用的数据集是与计算机网络相关的CIC和新南威尔士大学。我们使用上面提到的两种攻击生成了对抗性样本，这些攻击应用于一小部分数据集。随后，我们在对抗性数据集上训练和测试了该模型的准确性。我们记录了良性数据集和操纵数据集的结果，并观察到不同数据集上模型的准确性存在显著差异。从实验结果可以看出，LF攻击是失败的，而FP攻击是有效的，这证明了它在欺骗服务器方面的重要意义。在对CIC进行1%的LF攻击时，准确率约为0.0428，ASR为0.9564；因此，攻击很容易被检测到；而对于1%的FP攻击，准确率和ASR都约为0.9600，因此，FP攻击很难被检测到。我们用不同的中毒百分比重复了实验。



## **31. Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes**

梯度袖口：通过探索拒绝损失场景来检测对大型语言模型的越狱攻击 cs.CR

Project page:  https://huggingface.co/spaces/TrustSafeAI/GradientCuff-Jailbreak-Defense

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.00867v2) [paper-pdf](http://arxiv.org/pdf/2403.00867v2)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are becoming a prominent generative AI tool, where the user enters a query and the LLM generates an answer. To reduce harm and misuse, efforts have been made to align these LLMs to human values using advanced training techniques such as Reinforcement Learning from Human Feedback (RLHF). However, recent studies have highlighted the vulnerability of LLMs to adversarial jailbreak attempts aiming at subverting the embedded safety guardrails. To address this challenge, this paper defines and investigates the Refusal Loss of LLMs and then proposes a method called Gradient Cuff to detect jailbreak attempts. Gradient Cuff exploits the unique properties observed in the refusal loss landscape, including functional values and its smoothness, to design an effective two-step detection strategy. Experimental results on two aligned LLMs (LLaMA-2-7B-Chat and Vicuna-7B-V1.5) and six types of jailbreak attacks (GCG, AutoDAN, PAIR, TAP, Base64, and LRL) show that Gradient Cuff can significantly improve the LLM's rejection capability for malicious jailbreak queries, while maintaining the model's performance for benign user queries by adjusting the detection threshold.

摘要: 大型语言模型(LLM)正在成为一种重要的生成性人工智能工具，用户输入一个查询，LLM生成一个答案。为了减少危害和误用，已经做出努力，使用先进的培训技术，如从人类反馈中强化学习(RLHF)，使这些LLM与人类价值保持一致。然而，最近的研究强调了LLMS在旨在颠覆嵌入的安全护栏的对抗性越狱企图中的脆弱性。为了应对这一挑战，本文定义并研究了LLMS的拒绝丢失，提出了一种检测越狱企图的梯度袖口方法。梯度袖口利用拒绝丢失场景中观察到的独特属性，包括函数值及其光滑性，设计了一种有效的两步检测策略。在两个对齐的LLM(Llama-2-7B-Chat和Vicuna-7B-V1.5)和六种越狱攻击(GCG、AutoDAN、Pair、TAP、Base64和LRL)上的实验结果表明，梯度袖口可以显著提高LLM对恶意越狱查询的拒绝能力，同时通过调整检测阈值保持该模型对良性用户查询的性能。



## **32. XAI-Based Detection of Adversarial Attacks on Deepfake Detectors**

基于XAI的深伪检测器对抗性攻击检测 cs.CR

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02955v1) [paper-pdf](http://arxiv.org/pdf/2403.02955v1)

**Authors**: Ben Pinhasov, Raz Lapid, Rony Ohayon, Moshe Sipper, Yehudit Aperstein

**Abstract**: We introduce a novel methodology for identifying adversarial attacks on deepfake detectors using eXplainable Artificial Intelligence (XAI). In an era characterized by digital advancement, deepfakes have emerged as a potent tool, creating a demand for efficient detection systems. However, these systems are frequently targeted by adversarial attacks that inhibit their performance. We address this gap, developing a defensible deepfake detector by leveraging the power of XAI. The proposed methodology uses XAI to generate interpretability maps for a given method, providing explicit visualizations of decision-making factors within the AI models. We subsequently employ a pretrained feature extractor that processes both the input image and its corresponding XAI image. The feature embeddings extracted from this process are then used for training a simple yet effective classifier. Our approach contributes not only to the detection of deepfakes but also enhances the understanding of possible adversarial attacks, pinpointing potential vulnerabilities. Furthermore, this approach does not change the performance of the deepfake detector. The paper demonstrates promising results suggesting a potential pathway for future deepfake detection mechanisms. We believe this study will serve as a valuable contribution to the community, sparking much-needed discourse on safeguarding deepfake detectors.

摘要: 我们介绍了一种利用可解释人工智能(XAI)来识别针对深度假冒检测器的对抗性攻击的新方法。在一个以数字进步为特征的时代，深度假冒已经成为一种强有力的工具，创造了对高效检测系统的需求。然而，这些系统经常成为抑制其性能的对抗性攻击的目标。我们解决了这个问题，通过利用XAI的能力开发了一个可防御的深度伪检测器。所提出的方法使用XAI为给定的方法生成可解释性地图，提供人工智能模型中决策因素的显式可视化。随后，我们采用了一个预先训练的特征抽取器来处理输入图像及其对应的XAI图像。然后使用从该过程中提取的特征嵌入来训练简单而有效的分类器。我们的方法不仅有助于深度假冒的检测，还有助于增强对可能的敌意攻击的理解，准确地定位潜在的漏洞。此外，该方法不会改变深度伪检测器的性能。这篇论文展示了令人振奋的结果，为未来的深度伪检测机制提供了一条潜在的途径。我们相信，这项研究将对社区做出有价值的贡献，引发关于保护深度假冒探测器的迫切需要的讨论。



## **33. Precise Extraction of Deep Learning Models via Side-Channel Attacks on Edge/Endpoint Devices**

基于边缘/端点设备旁通道攻击的深度学习模型精确提取 cs.AI

Accepted by 27th European Symposium on Research in Computer Security  (ESORICS 2022)

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02870v1) [paper-pdf](http://arxiv.org/pdf/2403.02870v1)

**Authors**: Younghan Lee, Sohee Jun, Yungi Cho, Woorim Han, Hyungon Moon, Yunheung Paek

**Abstract**: With growing popularity, deep learning (DL) models are becoming larger-scale, and only the companies with vast training datasets and immense computing power can manage their business serving such large models. Most of those DL models are proprietary to the companies who thus strive to keep their private models safe from the model extraction attack (MEA), whose aim is to steal the model by training surrogate models. Nowadays, companies are inclined to offload the models from central servers to edge/endpoint devices. As revealed in the latest studies, adversaries exploit this opportunity as new attack vectors to launch side-channel attack (SCA) on the device running victim model and obtain various pieces of the model information, such as the model architecture (MA) and image dimension (ID). Our work provides a comprehensive understanding of such a relationship for the first time and would benefit future MEA studies in both offensive and defensive sides in that they may learn which pieces of information exposed by SCA are more important than the others. Our analysis additionally reveals that by grasping the victim model information from SCA, MEA can get highly effective and successful even without any prior knowledge of the model. Finally, to evince the practicality of our analysis results, we empirically apply SCA, and subsequently, carry out MEA under realistic threat assumptions. The results show up to 5.8 times better performance than when the adversary has no model information about the victim model.

摘要: 随着深度学习的日益普及，深度学习模型的规模越来越大，只有拥有海量训练数据集和巨大计算能力的公司才能管理自己的业务，为如此大规模的模型服务。这些DL模型中的大多数都是公司的专利，这些公司因此努力使他们的私人模型免受模型提取攻击(MEA)，其目的是通过训练代理模型来窃取模型。如今，公司倾向于将模型从中央服务器转移到边缘/终端设备。最新研究表明，攻击者利用这一机会作为新的攻击载体，对运行受害者模型的设备发起侧通道攻击(SCA)，获得模型的各种信息，如模型体系结构(MA)和图像维度(ID)。我们的工作首次提供了对这种关系的全面理解，并将有助于未来攻防双方的MEA研究，因为他们可以了解到SCA暴露的哪些信息比其他信息更重要。我们的分析还表明，通过从SCA中获取受害者模型信息，MEA即使在没有任何模型先验知识的情况下也可以获得高效和成功的信息。最后，为了证明我们的分析结果的实用性，我们实证地应用了SCA，并随后在现实的威胁假设下进行了MEA。结果表明，与对手没有关于受害者模型的模型信息时相比，性能最高可提高5.8倍。



## **34. FLGuard: Byzantine-Robust Federated Learning via Ensemble of Contrastive Models**

基于对比模型集成的拜占庭-稳健联邦学习 cs.LG

Accepted by 28th European Symposium on Research in Computer Security  (ESORICS 2023)

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02846v1) [paper-pdf](http://arxiv.org/pdf/2403.02846v1)

**Authors**: Younghan Lee, Yungi Cho, Woorim Han, Ho Bae, Yunheung Paek

**Abstract**: Federated Learning (FL) thrives in training a global model with numerous clients by only sharing the parameters of their local models trained with their private training datasets. Therefore, without revealing the private dataset, the clients can obtain a deep learning (DL) model with high performance. However, recent research proposed poisoning attacks that cause a catastrophic loss in the accuracy of the global model when adversaries, posed as benign clients, are present in a group of clients. Therefore, recent studies suggested byzantine-robust FL methods that allow the server to train an accurate global model even with the adversaries present in the system. However, many existing methods require the knowledge of the number of malicious clients or the auxiliary (clean) dataset or the effectiveness reportedly decreased hugely when the private dataset was non-independently and identically distributed (non-IID). In this work, we propose FLGuard, a novel byzantine-robust FL method that detects malicious clients and discards malicious local updates by utilizing the contrastive learning technique, which showed a tremendous improvement as a self-supervised learning method. With contrastive models, we design FLGuard as an ensemble scheme to maximize the defensive capability. We evaluate FLGuard extensively under various poisoning attacks and compare the accuracy of the global model with existing byzantine-robust FL methods. FLGuard outperforms the state-of-the-art defense methods in most cases and shows drastic improvement, especially in non-IID settings. https://github.com/201younghanlee/FLGuard

摘要: 联合学习(FL)通过仅共享使用其私有训练数据集训练的本地模型的参数，在与众多客户训练全球模型方面蓬勃发展。因此，在不透露私有数据集的情况下，客户端可以获得高性能的深度学习(DL)模型。然而，最近的研究提出，当一组客户中存在伪装成良性客户的对手时，中毒攻击会导致全球模型准确性的灾难性损失。因此，最近的研究建议拜占庭稳健的FL方法，允许服务器即使在系统中存在对手的情况下也能训练准确的全局模型。然而，许多现有的方法需要知道恶意客户端或辅助(干净)数据集的数量，或者当私有数据集是非独立且相同分布(非IID)时，据报道其有效性大大降低。在这项工作中，我们提出了一种新的拜占庭稳健FL方法--FLGuard，它利用对比学习技术检测恶意客户端并丢弃恶意本地更新，作为一种自我监督学习方法，显示出巨大的改进。通过对比模型，我们将FLGuard设计为一个整体方案，以最大限度地提高防御能力。我们在各种中毒攻击下对FLGuard进行了广泛的评估，并将全局模型的准确性与现有的拜占庭稳健FL方法进行了比较。在大多数情况下，FLGuard的表现优于最先进的防御方法，并显示出显著的改进，特别是在非IID设置中。Https://github.com/201younghanlee/FLGuard



## **35. Here Comes The AI Worm: Unleashing Zero-click Worms that Target GenAI-Powered Applications**

AI蠕虫来了：释放针对GenAI支持的应用的零点击蠕虫 cs.CR

Website: https://sites.google.com/view/compromptmized

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02817v1) [paper-pdf](http://arxiv.org/pdf/2403.02817v1)

**Authors**: Stav Cohen, Ron Bitton, Ben Nassi

**Abstract**: In the past year, numerous companies have incorporated Generative AI (GenAI) capabilities into new and existing applications, forming interconnected Generative AI (GenAI) ecosystems consisting of semi/fully autonomous agents powered by GenAI services. While ongoing research highlighted risks associated with the GenAI layer of agents (e.g., dialog poisoning, membership inference, prompt leaking, jailbreaking), a critical question emerges: Can attackers develop malware to exploit the GenAI component of an agent and launch cyber-attacks on the entire GenAI ecosystem? This paper introduces Morris II, the first worm designed to target GenAI ecosystems through the use of adversarial self-replicating prompts. The study demonstrates that attackers can insert such prompts into inputs that, when processed by GenAI models, prompt the model to replicate the input as output (replication), engaging in malicious activities (payload). Additionally, these inputs compel the agent to deliver them (propagate) to new agents by exploiting the connectivity within the GenAI ecosystem. We demonstrate the application of Morris II against GenAIpowered email assistants in two use cases (spamming and exfiltrating personal data), under two settings (black-box and white-box accesses), using two types of input data (text and images). The worm is tested against three different GenAI models (Gemini Pro, ChatGPT 4.0, and LLaVA), and various factors (e.g., propagation rate, replication, malicious activity) influencing the performance of the worm are evaluated.

摘要: 在过去的一年里，许多公司将生成性人工智能(GenAI)功能整合到新的和现有的应用程序中，形成了由GenAI服务支持的半/全自主代理组成的互联生成性AI(GenAI)生态系统。虽然正在进行的研究突出了与GenAI代理层相关的风险(例如，对话中毒、成员关系推断、提示泄漏、越狱)，但一个关键问题出现了：攻击者是否可以开发恶意软件来利用代理的GenAI组件，并对整个GenAI生态系统发动网络攻击？本文介绍了Morris II，它是第一个通过使用对抗性自我复制提示来攻击GenAI生态系统的蠕虫。这项研究表明，攻击者可以将这样的提示插入到输入中，当被GenAI模型处理时，提示模型将输入复制为输出(复制)，参与恶意活动(有效负载)。此外，这些输入迫使代理通过利用GenAI生态系统中的连接将它们交付(传播)给新的代理。我们使用两种类型的输入数据(文本和图像)，在两种设置(黑盒和白盒访问)下，在两种用例(垃圾邮件和渗漏个人数据)中演示了Morris II对GenAI支持的电子邮件助理的应用。该蠕虫针对三种不同的GenAI模型(Gemini Pro、ChatGPT 4.0和LLaVA)进行了测试，并评估了影响该蠕虫性能的各种因素(例如，传播速度、复制、恶意活动)。



## **36. Towards Robust Federated Learning via Logits Calibration on Non-IID Data**

基于非IID数据Logits校正的稳健联合学习 cs.CV

Accepted by IEEE NOMS 2024

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02803v1) [paper-pdf](http://arxiv.org/pdf/2403.02803v1)

**Authors**: Yu Qiao, Apurba Adhikary, Chaoning Zhang, Choong Seon Hong

**Abstract**: Federated learning (FL) is a privacy-preserving distributed management framework based on collaborative model training of distributed devices in edge networks. However, recent studies have shown that FL is vulnerable to adversarial examples (AEs), leading to a significant drop in its performance. Meanwhile, the non-independent and identically distributed (non-IID) challenge of data distribution between edge devices can further degrade the performance of models. Consequently, both AEs and non-IID pose challenges to deploying robust learning models at the edge. In this work, we adopt the adversarial training (AT) framework to improve the robustness of FL models against adversarial example (AE) attacks, which can be termed as federated adversarial training (FAT). Moreover, we address the non-IID challenge by implementing a simple yet effective logits calibration strategy under the FAT framework, which can enhance the robustness of models when subjected to adversarial attacks. Specifically, we employ a direct strategy to adjust the logits output by assigning higher weights to classes with small samples during training. This approach effectively tackles the class imbalance in the training data, with the goal of mitigating biases between local and global models. Experimental results on three dataset benchmarks, MNIST, Fashion-MNIST, and CIFAR-10 show that our strategy achieves competitive results in natural and robust accuracy compared to several baselines.

摘要: 联合学习(FL)是一种基于边缘网络中分布式设备协作模型训练的隐私保护分布式管理框架。然而，最近的研究表明，外语容易受到对抗性例子的影响，导致其成绩显著下降。同时，边缘设备之间数据分发的非独立和同分布(Non-IID)挑战可能会进一步降低模型的性能。因此，企业环境和非独立企业都对在边缘部署强大的学习模型提出了挑战。在这项工作中，我们采用对抗性训练(AT)框架来提高FL模型对对抗性范例(AE)攻击的健壮性，这可以被称为联合对抗性训练(FAT)。此外，我们通过在FAT框架下实现一种简单而有效的Logits校准策略来应对非IID的挑战，该策略可以增强模型在受到对抗性攻击时的健壮性。具体地说，我们采用一种直接策略来调整LOGITS输出，方法是在训练期间为样本较小的类分配更高的权重。这种方法有效地解决了训练数据中的类不平衡问题，目标是减轻局部模型和全局模型之间的偏差。在MNIST、Fashion-MNIST和CIFAR-10三个数据集基准上的实验结果表明，与几个基准相比，我们的策略在自然和稳健的准确率方面取得了与之相当的结果。



## **37. On the Alignment of Group Fairness with Attribute Privacy**

关于组公平性与属性隐私的匹配问题 cs.LG

arXiv admin note: text overlap with arXiv:2202.02242

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2211.10209v3) [paper-pdf](http://arxiv.org/pdf/2211.10209v3)

**Authors**: Jan Aalmoes, Vasisht Duddu, Antoine Boutet

**Abstract**: Group fairness and privacy are fundamental aspects in designing trustworthy machine learning models. Previous research has highlighted conflicts between group fairness and different privacy notions. We are the first to demonstrate the alignment of group fairness with the specific privacy notion of attribute privacy in a blackbox setting. Attribute privacy, quantified by the resistance to attribute inference attacks (AIAs), requires indistinguishability in the target model's output predictions. Group fairness guarantees this thereby mitigating AIAs and achieving attribute privacy. To demonstrate this, we first introduce AdaptAIA, an enhancement of existing AIAs, tailored for real-world datasets with class imbalances in sensitive attributes. Through theoretical and extensive empirical analyses, we demonstrate the efficacy of two standard group fairness algorithms (i.e., adversarial debiasing and exponentiated gradient descent) against AdaptAIA. Additionally, since using group fairness results in attribute privacy, it acts as a defense against AIAs, which is currently lacking. Overall, we show that group fairness aligns with attribute privacy at no additional cost other than the already existing trade-off with model utility.

摘要: 群体公平和隐私是设计可信机器学习模型的基本方面。之前的研究已经强调了群体公平和不同隐私观念之间的冲突。我们首先展示了在黑箱设置中，组公平与属性隐私的特定隐私概念的一致性。属性隐私由对属性推理攻击的抵抗力(AIAS)来量化，要求目标模型的输出预测不可区分。组公平性保证了这一点，从而减轻了AIAS并实现了属性隐私。为了说明这一点，我们首先引入了AdaptAIA，它是现有AIAS的增强，专为具有敏感属性中的类不平衡的真实世界数据集而定制。通过理论和大量的实证分析，我们证明了两种标准的群体公平算法(即对抗性去偏向算法和指数梯度下降算法)对AdaptAIA的有效性。此外，由于使用组公平会导致属性隐私，因此它充当了对AIAS的防御，而AIAS目前是缺乏的。总体而言，我们表明，除了已经存在的与模型实用程序的权衡外，组公平与属性隐私保持一致，而不需要额外的成本。



## **38. Minimum Topology Attacks for Graph Neural Networks**

图神经网络的最小拓扑攻击 cs.AI

Published on WWW 2023. Proceedings of the ACM Web Conference 2023

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02723v1) [paper-pdf](http://arxiv.org/pdf/2403.02723v1)

**Authors**: Mengmei Zhang, Xiao Wang, Chuan Shi, Lingjuan Lyu, Tianchi Yang, Junping Du

**Abstract**: With the great popularity of Graph Neural Networks (GNNs), their robustness to adversarial topology attacks has received significant attention. Although many attack methods have been proposed, they mainly focus on fixed-budget attacks, aiming at finding the most adversarial perturbations within a fixed budget for target node. However, considering the varied robustness of each node, there is an inevitable dilemma caused by the fixed budget, i.e., no successful perturbation is found when the budget is relatively small, while if it is too large, the yielding redundant perturbations will hurt the invisibility. To break this dilemma, we propose a new type of topology attack, named minimum-budget topology attack, aiming to adaptively find the minimum perturbation sufficient for a successful attack on each node. To this end, we propose an attack model, named MiBTack, based on a dynamic projected gradient descent algorithm, which can effectively solve the involving non-convex constraint optimization on discrete topology. Extensive results on three GNNs and four real-world datasets show that MiBTack can successfully lead all target nodes misclassified with the minimum perturbation edges. Moreover, the obtained minimum budget can be used to measure node robustness, so we can explore the relationships of robustness, topology, and uncertainty for nodes, which is beyond what the current fixed-budget topology attacks can offer.

摘要: 随着图神经网络(GNN)的广泛应用，其对敌意拓扑攻击的健壮性受到了广泛的关注。虽然已经提出了许多攻击方法，但它们主要集中在固定预算攻击上，旨在为目标节点在固定预算内找到最具对抗性的扰动。然而，考虑到每个节点的健壮性不同，固定预算不可避免地造成了一个两难境地，即当预算相对较小时，没有找到成功的扰动，而如果预算太大，则产生的冗余扰动将损害不可见性。为了打破这一困境，我们提出了一种新的拓扑攻击，称为最小预算拓扑攻击，旨在自适应地找到对每个节点进行成功攻击所需的最小扰动。为此，我们提出了一种基于动态投影梯度下降算法的攻击模型MiBTack，该模型可以有效地解决离散拓扑上涉及的非凸约束优化问题。在3个GNN和4个真实数据集上的广泛结果表明，MiBTack能够成功地以最小扰动边引导所有目标节点的错误分类。此外，所得到的最小预算可以用来衡量节点的健壮性，从而可以探索节点的健壮性、拓扑性和不确定性之间的关系，这是目前固定预算拓扑攻击所不能提供的。



## **39. ScAR: Scaling Adversarial Robustness for LiDAR Object Detection**

SCAR：激光雷达目标检测的对抗性缩放算法 cs.CV

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2312.03085v2) [paper-pdf](http://arxiv.org/pdf/2312.03085v2)

**Authors**: Xiaohu Lu, Hayder Radha

**Abstract**: The adversarial robustness of a model is its ability to resist adversarial attacks in the form of small perturbations to input data. Universal adversarial attack methods such as Fast Sign Gradient Method (FSGM) and Projected Gradient Descend (PGD) are popular for LiDAR object detection, but they are often deficient compared to task-specific adversarial attacks. Additionally, these universal methods typically require unrestricted access to the model's information, which is difficult to obtain in real-world applications. To address these limitations, we present a black-box Scaling Adversarial Robustness (ScAR) method for LiDAR object detection. By analyzing the statistical characteristics of 3D object detection datasets such as KITTI, Waymo, and nuScenes, we have found that the model's prediction is sensitive to scaling of 3D instances. We propose three black-box scaling adversarial attack methods based on the available information: model-aware attack, distribution-aware attack, and blind attack. We also introduce a strategy for generating scaling adversarial examples to improve the model's robustness against these three scaling adversarial attacks. Comparison with other methods on public datasets under different 3D object detection architectures demonstrates the effectiveness of our proposed method. Our code is available at https://github.com/xiaohulugo/ScAR-IROS2023.

摘要: 模型的对抗性健壮性在于其抵抗以输入数据的小扰动形式的对抗性攻击的能力。快速符号梯度法(FSGM)和投影梯度下降法(PGD)等通用对抗攻击方法是激光雷达目标检测的常用方法，但与特定任务的对抗攻击相比往往存在不足。此外，这些通用方法通常需要不受限制地访问模型的信息，这在现实世界的应用程序中是很难获得的。为了解决这些局限性，我们提出了一种用于激光雷达目标检测的黑盒尺度对抗稳健性(SCAR)方法。通过分析Kitti、Waymo和nuScenes等3D目标检测数据集的统计特性，我们发现模型的预测对3D实例的缩放很敏感。我们根据已有的信息提出了三种黑盒尺度对抗性攻击方法：模型感知攻击、分布感知攻击和盲目攻击。我们还介绍了一种生成伸缩敌意实例的策略，以提高模型对这三种伸缩对手攻击的稳健性。在不同3D目标检测体系结构下的公共数据集上与其他方法进行了比较，验证了该方法的有效性。我们的代码可以在https://github.com/xiaohulugo/ScAR-IROS2023.上找到



## **40. Towards Poisoning Fair Representations**

走向毒害公平代表 cs.LG

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2309.16487v2) [paper-pdf](http://arxiv.org/pdf/2309.16487v2)

**Authors**: Tianci Liu, Haoyu Wang, Feijie Wu, Hengtong Zhang, Pan Li, Lu Su, Jing Gao

**Abstract**: Fair machine learning seeks to mitigate model prediction bias against certain demographic subgroups such as elder and female. Recently, fair representation learning (FRL) trained by deep neural networks has demonstrated superior performance, whereby representations containing no demographic information are inferred from the data and then used as the input to classification or other downstream tasks. Despite the development of FRL methods, their vulnerability under data poisoning attack, a popular protocol to benchmark model robustness under adversarial scenarios, is under-explored. Data poisoning attacks have been developed for classical fair machine learning methods which incorporate fairness constraints into shallow-model classifiers. Nonetheless, these attacks fall short in FRL due to notably different fairness goals and model architectures. This work proposes the first data poisoning framework attacking FRL. We induce the model to output unfair representations that contain as much demographic information as possible by injecting carefully crafted poisoning samples into the training data. This attack entails a prohibitive bilevel optimization, wherefore an effective approximated solution is proposed. A theoretical analysis on the needed number of poisoning samples is derived and sheds light on defending against the attack. Experiments on benchmark fairness datasets and state-of-the-art fair representation learning models demonstrate the superiority of our attack.

摘要: 公平的机器学习寻求减轻模型预测对某些人口子组的偏差，例如老年人和女性。最近，由深度神经网络训练的公平表征学习(FRL)表现出了优越的性能，即从数据中推断出不包含人口统计信息的表征，然后将其用作分类或其他下游任务的输入。尽管FRL方法已经得到了发展，但它们在数据中毒攻击下的脆弱性还没有得到充分的探索。数据中毒攻击是一种流行的协议，用于在对抗场景下对模型的健壮性进行基准测试。数据中毒攻击是针对将公平性约束引入浅模型分类器的经典公平机器学习方法而开发的。尽管如此，由于公平目标和模型架构的显著不同，这些攻击在FRL上仍存在不足。本文提出了第一个攻击FRL的数据中毒框架。我们通过将精心制作的中毒样本注入训练数据来诱导模型输出包含尽可能多的人口统计信息的不公平表示。这种攻击需要一个禁止的两层优化，因此提出了一个有效的近似解。对所需中毒样本数量进行了理论分析，为防御攻击提供了理论依据。在基准公平性数据集和最新的公平表示学习模型上的实验证明了该攻击的优越性。



## **41. COMMIT: Certifying Robustness of Multi-Sensor Fusion Systems against Semantic Attacks**

Commit：证明多传感器融合系统对语义攻击的健壮性 cs.LG

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.02329v1) [paper-pdf](http://arxiv.org/pdf/2403.02329v1)

**Authors**: Zijian Huang, Wenda Chu, Linyi Li, Chejian Xu, Bo Li

**Abstract**: Multi-sensor fusion systems (MSFs) play a vital role as the perception module in modern autonomous vehicles (AVs). Therefore, ensuring their robustness against common and realistic adversarial semantic transformations, such as rotation and shifting in the physical world, is crucial for the safety of AVs. While empirical evidence suggests that MSFs exhibit improved robustness compared to single-modal models, they are still vulnerable to adversarial semantic transformations. Despite the proposal of empirical defenses, several works show that these defenses can be attacked again by new adaptive attacks. So far, there is no certified defense proposed for MSFs. In this work, we propose the first robustness certification framework COMMIT certify robustness of multi-sensor fusion systems against semantic attacks. In particular, we propose a practical anisotropic noise mechanism that leverages randomized smoothing with multi-modal data and performs a grid-based splitting method to characterize complex semantic transformations. We also propose efficient algorithms to compute the certification in terms of object detection accuracy and IoU for large-scale MSF models. Empirically, we evaluate the efficacy of COMMIT in different settings and provide a comprehensive benchmark of certified robustness for different MSF models using the CARLA simulation platform. We show that the certification for MSF models is at most 48.39% higher than that of single-modal models, which validates the advantages of MSF models. We believe our certification framework and benchmark will contribute an important step towards certifiably robust AVs in practice.

摘要: 多传感器融合系统(MSF)作为现代自动驾驶汽车(AVs)的感知模块，发挥着至关重要的作用。因此，确保它们对常见的和现实的对抗性语义转换的健壮性，如物理世界中的旋转和移位，对于AVs的安全至关重要。虽然经验证据表明，与单模式模型相比，MSF表现出更好的稳健性，但它们仍然容易受到对抗性语义转换的影响。尽管提出了经验性防御，但一些工作表明，这些防御可以再次被新的适应性攻击攻击。到目前为止，还没有针对MSF的认证辩护建议。在这项工作中，我们提出了第一个健壮性认证框架，提交证明多传感器融合系统对语义攻击的健壮性。特别是，我们提出了一种实用的各向异性噪声机制，该机制利用多峰数据的随机化平滑，并执行基于网格的分裂方法来表征复杂的语义转换。对于大规模的MSF模型，我们还从目标检测精度和IOU两个方面提出了有效的算法来计算证书。在实证上，我们评估了不同环境下提交的有效性，并使用CALA仿真平台为不同的MSF模型提供了一个经过验证的健壮性综合基准。结果表明，MSF模型的认证准确率最高可达48.39%，验证了MSF模型的优越性。我们相信，我们的认证框架和基准将有助于在实践中朝着可认证的稳健的AVS迈出重要的一步。



## **42. Mirage: Defense against CrossPath Attacks in Software Defined Networks**

幻影：软件定义网络中的交叉路径攻击防御 cs.CR

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.02172v1) [paper-pdf](http://arxiv.org/pdf/2403.02172v1)

**Authors**: Shariq Murtuza, Krishna Asawa

**Abstract**: The Software-Defined Networks (SDNs) face persistent threats from various adversaries that attack them using different methods to mount Denial of Service attacks. These attackers have different motives and follow diverse tactics to achieve their nefarious objectives. In this work, we focus on the impact of CrossPath attacks in SDNs and introduce our framework, Mirage, which not only detects but also mitigates this attack. Our framework, Mirage, detects SDN switches that become unreachable due to being under attack, takes proactive measures to prevent Adversarial Path Reconnaissance, and effectively mitigates CrossPath attacks in SDNs. A CrossPath attack is a form of link flood attack that indirectly attacks the control plane by overwhelming the shared links that connect the data and control planes with data plane traffic. This attack is exclusive to in band SDN, where the data and the control plane, both utilize the same physical links for transmitting and receiving traffic. Our framework, Mirage, prevents attackers from launching adversarial path reconnaissance to identify shared links in a network, thereby thwarting their abuse and preventing this attack. Mirage not only stops adversarial path reconnaissance but also includes features to quickly counter ongoing attacks once detected. Mirage uses path diversity to reroute network packet to prevent timing based measurement. Mirage can also enforce short lived flow table rules to prevent timing attacks. These measures are carefully designed to enhance the security of the SDN environment. Moreover, we share the results of our experiments, which clearly show Mirage's effectiveness in preventing path reconnaissance, detecting CrossPath attacks, and mitigating ongoing threats. Our framework successfully protects the network from these harmful activities, giving valuable insights into SDN security.

摘要: 软件定义网络(SDN)面临来自各种对手的持续威胁，这些对手使用不同的方法发动拒绝服务攻击。这些袭击者有不同的动机，采取不同的战术来实现他们的邪恶目的。在这项工作中，我们重点研究了跨路径攻击在SDNS中的影响，并介绍了我们的框架Mige，它不仅可以检测到这种攻击，而且可以缓解这种攻击。我们的框架MIRAGE检测由于受到攻击而变得不可达的SDN交换机，采取主动措施防止恶意路径侦察，并有效地缓解SDN中的CrossPath攻击。交叉路径攻击是链路泛洪攻击的一种形式，它通过压倒连接数据平面和控制平面与数据平面流量的共享链路来间接攻击控制平面。此攻击专用于带内SDN，其中数据和控制平面都使用相同的物理链路来传输和接收流量。我们的框架，幻影，防止攻击者发起敌对的路径侦察来识别网络中的共享链路，从而挫败他们的滥用，防止这种攻击。幻影不仅停止敌对路径侦察，还包括一旦检测到快速反击正在进行的攻击的功能。幻影使用路径分集来重新路由网络数据包，以防止基于时序的测量。幻影还可以强制执行短期流表规则，以防止计时攻击。这些措施都是精心设计的，以加强SDN环境的安全。此外，我们还分享了我们的实验结果，这些结果清楚地表明了幻影在防止路径侦察、检测交叉路径攻击和缓解持续威胁方面的有效性。我们的框架成功地保护网络免受这些有害活动的影响，为SDN安全提供了有价值的见解。



## **43. Rethinking Model Ensemble in Transfer-based Adversarial Attacks**

基于迁移的对抗性攻击中模型集成的再思考 cs.CV

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2303.09105v2) [paper-pdf](http://arxiv.org/pdf/2303.09105v2)

**Authors**: Huanran Chen, Yichi Zhang, Yinpeng Dong, Xiao Yang, Hang Su, Jun Zhu

**Abstract**: It is widely recognized that deep learning models lack robustness to adversarial examples. An intriguing property of adversarial examples is that they can transfer across different models, which enables black-box attacks without any knowledge of the victim model. An effective strategy to improve the transferability is attacking an ensemble of models. However, previous works simply average the outputs of different models, lacking an in-depth analysis on how and why model ensemble methods can strongly improve the transferability. In this paper, we rethink the ensemble in adversarial attacks and define the common weakness of model ensemble with two properties: 1) the flatness of loss landscape; and 2) the closeness to the local optimum of each model. We empirically and theoretically show that both properties are strongly correlated with the transferability and propose a Common Weakness Attack (CWA) to generate more transferable adversarial examples by promoting these two properties. Experimental results on both image classification and object detection tasks validate the effectiveness of our approach to improving the adversarial transferability, especially when attacking adversarially trained models. We also successfully apply our method to attack a black-box large vision-language model -- Google's Bard, showing the practical effectiveness. Code is available at \url{https://github.com/huanranchen/AdversarialAttacks}.

摘要: 人们普遍认为，深度学习模型对对抗性例子缺乏稳健性。对抗性例子的一个耐人寻味的特性是，它们可以在不同的模型之间传输，这使得在不知道受害者模型的情况下进行黑盒攻击。提高可转移性的一个有效策略是攻击一系列模型。然而，以往的工作只是简单地对不同模型的输出进行平均，而缺乏对模型集成方法如何以及为什么能够显著提高可转移性的深入分析。在本文中，我们重新考虑了对抗性攻击中的集成，并定义了模型集成的两个共同弱点：1)损失图景的平坦性；2)每个模型接近局部最优。我们从经验和理论上证明了这两个性质与可转移性有很强的相关性，并提出了一种共同弱点攻击(CWA)，通过提升这两个性质来生成更多可转移的对抗性实例。在图像分类和目标检测任务上的实验结果验证了该方法的有效性，特别是在攻击对抗性训练模型时。我们还成功地应用我们的方法攻击了一个黑盒大视觉语言模型--Google的BARD，显示了它的实际有效性。代码可在\url{https://github.com/huanranchen/AdversarialAttacks}.上找到



## **44. Robustness Bounds on the Successful Adversarial Examples: Theory and Practice**

成功对抗性例子的稳健性界限：理论与实践 cs.LG

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.01896v1) [paper-pdf](http://arxiv.org/pdf/2403.01896v1)

**Authors**: Hiroaki Maeshima, Akira Otsuka

**Abstract**: Adversarial example (AE) is an attack method for machine learning, which is crafted by adding imperceptible perturbation to the data inducing misclassification. In the current paper, we investigated the upper bound of the probability of successful AEs based on the Gaussian Process (GP) classification. We proved a new upper bound that depends on AE's perturbation norm, the kernel function used in GP, and the distance of the closest pair with different labels in the training dataset. Surprisingly, the upper bound is determined regardless of the distribution of the sample dataset. We showed that our theoretical result was confirmed through the experiment using ImageNet. In addition, we showed that changing the parameters of the kernel function induces a change of the upper bound of the probability of successful AEs.

摘要: 对抗性例子(AE)是一种机器学习的攻击方法，它是通过对导致错误分类的数据添加不可察觉的扰动来构建的。在本文中，我们研究了基于高斯过程(GP)分类的AES成功概率的上界。我们证明了一个新的上界，它依赖于AE的扰动范数，GP中使用的核函数，以及训练数据集中具有不同标签的最近对的距离。令人惊讶的是，无论样本数据集的分布如何，上限都是确定的。我们通过使用ImageNet的实验验证了我们的理论结果。此外，我们还证明了改变核函数的参数会引起成功事件概率的上界的变化。



## **45. One Prompt Word is Enough to Boost Adversarial Robustness for Pre-trained Vision-Language Models**

一个提示词就足以提高预先训练的视觉语言模型的对抗性 cs.CV

CVPR2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.01849v1) [paper-pdf](http://arxiv.org/pdf/2403.01849v1)

**Authors**: Lin Li, Haoyan Guan, Jianing Qiu, Michael Spratling

**Abstract**: Large pre-trained Vision-Language Models (VLMs) like CLIP, despite having remarkable generalization ability, are highly vulnerable to adversarial examples. This work studies the adversarial robustness of VLMs from the novel perspective of the text prompt instead of the extensively studied model weights (frozen in this work). We first show that the effectiveness of both adversarial attack and defense are sensitive to the used text prompt. Inspired by this, we propose a method to improve resilience to adversarial attacks by learning a robust text prompt for VLMs. The proposed method, named Adversarial Prompt Tuning (APT), is effective while being both computationally and data efficient. Extensive experiments are conducted across 15 datasets and 4 data sparsity schemes (from 1-shot to full training data settings) to show APT's superiority over hand-engineered prompts and other state-of-the-art adaption methods. APT demonstrated excellent abilities in terms of the in-distribution performance and the generalization under input distribution shift and across datasets. Surprisingly, by simply adding one learned word to the prompts, APT can significantly boost the accuracy and robustness (epsilon=4/255) over the hand-engineered prompts by +13% and +8.5% on average respectively. The improvement further increases, in our most effective setting, to +26.4% for accuracy and +16.7% for robustness. Code is available at https://github.com/TreeLLi/APT.

摘要: 像CLIP这样的大型预先训练的视觉语言模型(VLM)，尽管具有显著的泛化能力，但很容易受到对手例子的攻击。该工作从文本提示的新角度来研究VLMS的对抗健壮性，而不是广泛研究的模型权重(在本工作中是冻结的)。我们首先证明了对抗性攻击和防御的有效性都对所使用的文本提示敏感。受此启发，我们提出了一种通过学习VLM的健壮文本提示来提高对对手攻击的恢复能力的方法。该方法称为对抗性提示调优(APT)，在计算效率和数据效率上都是有效的。在15个数据集和4个数据稀疏方案(从单镜头到全训练数据设置)上进行了广泛的实验，以展示APT相对于人工设计提示和其他最先进的适应方法的优势。在输入分布平移和跨数据集情况下，APT在分布内性能和泛化能力方面表现出优异的性能。令人惊讶的是，通过简单地在提示中添加一个学习的单词，APT可以显著提高人工设计提示的准确率和稳健性(epsilon=4/255)，平均分别提高+13%和+8.5%。在我们最有效的设置中，改进进一步增加了准确性的+26.4%和健壮性的+16.7%。代码可在https://github.com/TreeLLi/APT.上找到



## **46. Coupling Graph Neural Networks with Fractional Order Continuous Dynamics: A Robustness Study**

分数阶连续动态耦合图神经网络的稳健性研究 cs.LG

in Proc. AAAI Conference on Artificial Intelligence, Vancouver,  Canada, Feb. 2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2401.04331v2) [paper-pdf](http://arxiv.org/pdf/2401.04331v2)

**Authors**: Qiyu Kang, Kai Zhao, Yang Song, Yihang Xie, Yanan Zhao, Sijie Wang, Rui She, Wee Peng Tay

**Abstract**: In this work, we rigorously investigate the robustness of graph neural fractional-order differential equation (FDE) models. This framework extends beyond traditional graph neural (integer-order) ordinary differential equation (ODE) models by implementing the time-fractional Caputo derivative. Utilizing fractional calculus allows our model to consider long-term memory during the feature updating process, diverging from the memoryless Markovian updates seen in traditional graph neural ODE models. The superiority of graph neural FDE models over graph neural ODE models has been established in environments free from attacks or perturbations. While traditional graph neural ODE models have been verified to possess a degree of stability and resilience in the presence of adversarial attacks in existing literature, the robustness of graph neural FDE models, especially under adversarial conditions, remains largely unexplored. This paper undertakes a detailed assessment of the robustness of graph neural FDE models. We establish a theoretical foundation outlining the robustness characteristics of graph neural FDE models, highlighting that they maintain more stringent output perturbation bounds in the face of input and graph topology disturbances, compared to their integer-order counterparts. Our empirical evaluations further confirm the enhanced robustness of graph neural FDE models, highlighting their potential in adversarially robust applications.

摘要: 在这项工作中，我们严格研究了图神经分数阶微分方程(FDE)模型的稳健性。该框架通过实现时间分数Caputo导数，扩展了传统的图神经(整数阶)常微分方程(ODE)模型。利用分数阶微积分，我们的模型可以在特征更新过程中考虑长期记忆，不同于传统的图神经节点模型中看到的无记忆的马尔可夫更新。图神经FDE模型相对于图神经ODE模型的优越性已经在没有攻击或扰动的环境中得到了证实。虽然已有文献证明传统的图神经FDE模型在对抗攻击下具有一定程度的稳定性和韧性，但图神经FDE模型的稳健性，特别是在对抗条件下的稳健性，在很大程度上还没有被探索。本文对图神经FDE模型的稳健性进行了详细的评估。我们建立了一个理论基础，概述了图神经FDE模型的稳健性特征，强调了它们在面对输入和图的拓扑扰动时，与其整数阶对应模型相比，保持了更严格的输出摄动界。我们的经验评估进一步证实了图神经FDE模型的增强的稳健性，突出了它们在相反的健壮应用中的潜力。



## **47. LLMs Can Defend Themselves Against Jailbreaking in a Practical Manner: A Vision Paper**

LLMS能够以实际的方式保护自己免受越狱：一份愿景文件 cs.CR

Fixed the bibliography reference issue in our LLM jailbreak defense  vision paper submitted on 24 Feb 2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2402.15727v2) [paper-pdf](http://arxiv.org/pdf/2402.15727v2)

**Authors**: Daoyuan Wu, Shuai Wang, Yang Liu, Ning Liu

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs). A considerable amount of research exists proposing more effective jailbreak attacks, including the recent Greedy Coordinate Gradient (GCG) attack, jailbreak template-based attacks such as using "Do-Anything-Now" (DAN), and multilingual jailbreak. In contrast, the defensive side has been relatively less explored. This paper proposes a lightweight yet practical defense called SELFDEFEND, which can defend against all existing jailbreak attacks with minimal delay for jailbreak prompts and negligible delay for normal user prompts. Our key insight is that regardless of the kind of jailbreak strategies employed, they eventually need to include a harmful prompt (e.g., "how to make a bomb") in the prompt sent to LLMs, and we found that existing LLMs can effectively recognize such harmful prompts that violate their safety policies. Based on this insight, we design a shadow stack that concurrently checks whether a harmful prompt exists in the user prompt and triggers a checkpoint in the normal stack once a token of "No" or a harmful prompt is output. The latter could also generate an explainable LLM response to adversarial prompts. We demonstrate our idea of SELFDEFEND works in various jailbreak scenarios through manual analysis in GPT-3.5/4. We also list three future directions to further enhance SELFDEFEND.

摘要: 越狱是一种新兴的对抗性攻击，它绕过了现成的大型语言模型(LLM)中部署的安全对齐。已有大量研究提出了更有效的越狱攻击方案，包括最近的贪婪坐标梯度(GCG)攻击、基于模板的越狱攻击(例如使用“Do-Anything-Now”(DAN))和多语言越狱。相比之下，防守方面的探索相对较少。本文提出了一种轻量级而实用的防御方法SELFDEFEND，它可以防御所有现有的越狱攻击，而越狱提示的延迟最小，正常用户提示的延迟可以忽略不计。我们的主要见解是，无论采用哪种越狱策略，他们最终都需要在发送给LLMS的提示中包含有害提示(例如，如何制造炸弹)，我们发现现有LLMS可以有效地识别此类违反其安全政策的有害提示。基于这一观点，我们设计了一个影子堆栈，该堆栈同时检查用户提示中是否存在有害提示，并在输出令牌“否”或有害提示时触发正常堆栈中的检查点。后者还可以对对抗性提示产生可解释的LLM响应。我们通过GPT-3.5/4中的手动分析，展示了我们的SELFDEFEND在各种越狱场景中的工作原理。我们还列出了进一步增强SELFDEFEND的三个未来方向。



## **48. Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective**

共享扩散模型中的隐私和公平风险：对抗性视角 cs.LG

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2402.18607v2) [paper-pdf](http://arxiv.org/pdf/2402.18607v2)

**Authors**: Xinjian Luo, Yangfan Jiang, Fei Wei, Yuncheng Wu, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Diffusion models have recently gained significant attention in both academia and industry due to their impressive generative performance in terms of both sampling quality and distribution coverage. Accordingly, proposals are made for sharing pre-trained diffusion models across different organizations, as a way of improving data utilization while enhancing privacy protection by avoiding sharing private data directly. However, the potential risks associated with such an approach have not been comprehensively examined.   In this paper, we take an adversarial perspective to investigate the potential privacy and fairness risks associated with the sharing of diffusion models. Specifically, we investigate the circumstances in which one party (the sharer) trains a diffusion model using private data and provides another party (the receiver) black-box access to the pre-trained model for downstream tasks. We demonstrate that the sharer can execute fairness poisoning attacks to undermine the receiver's downstream models by manipulating the training data distribution of the diffusion model. Meanwhile, the receiver can perform property inference attacks to reveal the distribution of sensitive features in the sharer's dataset. Our experiments conducted on real-world datasets demonstrate remarkable attack performance on different types of diffusion models, which highlights the critical importance of robust data auditing and privacy protection protocols in pertinent applications.

摘要: 扩散模型由于其在抽样质量和分布复盖率方面令人印象深刻的生成性能，最近在学术界和工业界都得到了极大的关注。因此，提出了在不同组织之间共享预先培训的传播模式的建议，以此作为提高数据利用率的一种方式，同时通过避免直接共享私人数据来加强隐私保护。然而，与这种方法相关的潜在风险尚未得到全面审查。在这篇文章中，我们采取对抗性的视角来调查与共享扩散模型相关的潜在的隐私和公平风险。具体地说，我们调查了一方(共享者)使用私有数据训练扩散模型，并为另一方(接收者)提供对下游任务的预训练模型的黑箱访问的情况。我们证明了共享者可以通过操纵扩散模型的训练数据分布来执行公平毒化攻击来破坏接收者的下游模型。同时，接收者可以执行属性推断攻击，以揭示共享者数据集中敏感特征的分布。我们在真实数据集上进行的实验表明，在不同类型的扩散模型上具有显著的攻击性能，这突显了健壮的数据审计和隐私保护协议在相关应用中的关键重要性。



## **49. Near Optimal Adversarial Attack on UCB Bandits**

对UCB土匪的近最优敌意攻击 cs.LG

Accepted at AISTATS 2024, a preliminary version appeared at ICML 2023  AdvML Workshop

**SubmitDate**: 2024-03-03    [abs](http://arxiv.org/abs/2008.09312v7) [paper-pdf](http://arxiv.org/pdf/2008.09312v7)

**Authors**: Shiliang Zuo

**Abstract**: I study adversarial attacks against stochastic bandit algorithms. At each round, the learner chooses an arm, and a stochastic reward is generated. The adversary strategically adds corruption to the reward, and the learner is only able to observe the corrupted reward at each round. Two sets of results are presented in this paper. The first set studies the optimal attack strategies for the adversary. The adversary has a target arm he wishes to promote, and his goal is to manipulate the learner into choosing this target arm $T - o(T)$ times. I design attack strategies against UCB and Thompson Sampling that only spends $\widehat{O}(\sqrt{\log T})$ cost. Matching lower bounds are presented, and the vulnerability of UCB, Thompson sampling and $\varepsilon$-greedy are exactly characterized. The second set studies how the learner can defend against the adversary. Inspired by literature on smoothed analysis and behavioral economics, I present two simple algorithms that achieve a competitive ratio arbitrarily close to 1.

摘要: 我研究针对随机盗贼算法的敌意攻击。在每一轮，学习者选择一只手臂，并产生随机奖励。对手策略性地将腐败添加到奖励中，而学习者在每一轮只能观察到腐败的奖励。本文给出了两组结果。第一组研究对手的最优攻击策略。对手有一个他想要提升的目标手臂，他的目标是操纵学习者选择这个目标手臂$T-O(T)$次。我设计了针对UCB和Thompson抽样的攻击策略，该策略只花费$\widehat{O}(\sqrt{\log T})$。给出了匹配下界，并准确刻画了UCB、Thompson抽样和$varepsilon$-贪婪的脆弱性。第二组研究学习者如何防御对手。受平滑分析和行为经济学文献的启发，我提出了两个简单的算法，它们的竞争比率任意接近1。



## **50. Breaking Down the Defenses: A Comparative Survey of Attacks on Large Language Models**

击破防线：大型语言模型遭受攻击的比较研究 cs.CR

**SubmitDate**: 2024-03-03    [abs](http://arxiv.org/abs/2403.04786v1) [paper-pdf](http://arxiv.org/pdf/2403.04786v1)

**Authors**: Arijit Ghosh Chowdhury, Md Mofijul Islam, Vaibhav Kumar, Faysal Hossain Shezan, Vaibhav Kumar, Vinija Jain, Aman Chadha

**Abstract**: Large Language Models (LLMs) have become a cornerstone in the field of Natural Language Processing (NLP), offering transformative capabilities in understanding and generating human-like text. However, with their rising prominence, the security and vulnerability aspects of these models have garnered significant attention. This paper presents a comprehensive survey of the various forms of attacks targeting LLMs, discussing the nature and mechanisms of these attacks, their potential impacts, and current defense strategies. We delve into topics such as adversarial attacks that aim to manipulate model outputs, data poisoning that affects model training, and privacy concerns related to training data exploitation. The paper also explores the effectiveness of different attack methodologies, the resilience of LLMs against these attacks, and the implications for model integrity and user trust. By examining the latest research, we provide insights into the current landscape of LLM vulnerabilities and defense mechanisms. Our objective is to offer a nuanced understanding of LLM attacks, foster awareness within the AI community, and inspire robust solutions to mitigate these risks in future developments.

摘要: 大型语言模型(LLM)已经成为自然语言处理(NLP)领域的基石，在理解和生成类似人类的文本方面提供了变革性的能力。然而，随着它们的日益突出，这些模型的安全和漏洞方面已经引起了极大的关注。本文对各种形式的针对LLMS的攻击进行了全面的综述，讨论了这些攻击的性质和机制、它们的潜在影响以及当前的防御策略。我们深入探讨了旨在操纵模型输出的对抗性攻击、影响模型训练的数据中毒以及与训练数据利用相关的隐私问题等主题。文中还探讨了不同攻击方法的有效性，LLMS对这些攻击的恢复能力，以及对模型完整性和用户信任的影响。通过检查最新的研究，我们提供了对LLM漏洞和防御机制的当前情况的见解。我们的目标是提供对LLM攻击的细微差别的理解，培养人工智能社区的意识，并激发强大的解决方案，以减轻未来发展中的这些风险。



