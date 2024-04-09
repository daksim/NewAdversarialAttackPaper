# Latest Adversarial Attack Papers
**update at 2024-04-09 20:03:35**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Case Study: Neural Network Malware Detection Verification for Feature and Image Datasets**

案例研究：特征和图像数据集的神经网络恶意软件检测验证 cs.CR

In International Conference On Formal Methods in Software  Engineering, 2024; (FormaliSE'24)

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05703v1) [paper-pdf](http://arxiv.org/pdf/2404.05703v1)

**Authors**: Preston K. Robinette, Diego Manzanas Lopez, Serena Serbinowska, Kevin Leach, Taylor T. Johnson

**Abstract**: Malware, or software designed with harmful intent, is an ever-evolving threat that can have drastic effects on both individuals and institutions. Neural network malware classification systems are key tools for combating these threats but are vulnerable to adversarial machine learning attacks. These attacks perturb input data to cause misclassification, bypassing protective systems. Existing defenses often rely on enhancing the training process, thereby increasing the model's robustness to these perturbations, which is quantified using verification. While training improvements are necessary, we propose focusing on the verification process used to evaluate improvements to training. As such, we present a case study that evaluates a novel verification domain that will help to ensure tangible safeguards against adversaries and provide a more reliable means of evaluating the robustness and effectiveness of anti-malware systems. To do so, we describe malware classification and two types of common malware datasets (feature and image datasets), demonstrate the certified robustness accuracy of malware classifiers using the Neural Network Verification (NNV) and Neural Network Enumeration (nnenum) tools, and outline the challenges and future considerations necessary for the improvement and refinement of the verification of malware classification. By evaluating this novel domain as a case study, we hope to increase its visibility, encourage further research and scrutiny, and ultimately enhance the resilience of digital systems against malicious attacks.

摘要: 恶意软件，即带有有害意图的软件，是一种不断演变的威胁，可能会对个人和机构产生严重影响。神经网络恶意软件分类系统是打击这些威胁的关键工具，但容易受到对抗性机器学习攻击。这些攻击会绕过保护系统，扰乱输入数据，导致错误分类。现有的防御通常依赖于加强训练过程，从而增加模型对这些扰动的稳健性，这是使用验证来量化的。虽然培训改进是必要的，但我们建议将重点放在用于评估培训改进情况的核查过程上。因此，我们提供了一个案例研究来评估一个新的验证域，该验证域将有助于确保针对攻击者的切实保护，并提供一种更可靠的方法来评估反恶意软件系统的健壮性和有效性。为此，我们描述了恶意软件分类和两种常见的恶意软件数据集(特征数据集和图像数据集)，使用神经网络验证(NNV)和神经网络枚举(Nnenum)工具证明了恶意软件分类器的健壮性准确性，并概述了改进和完善恶意软件分类验证所面临的挑战和未来需要考虑的问题。通过评估这一新领域作为案例研究，我们希望提高其可见度，鼓励进一步的研究和审查，并最终增强数字系统对恶意攻击的弹性。



## **2. David and Goliath: An Empirical Evaluation of Attacks and Defenses for QNNs at the Deep Edge**

David and Goliath：对QNN在深度边缘的攻击和防御的经验评估 cs.LG

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05688v1) [paper-pdf](http://arxiv.org/pdf/2404.05688v1)

**Authors**: Miguel Costa, Sandro Pinto

**Abstract**: ML is shifting from the cloud to the edge. Edge computing reduces the surface exposing private data and enables reliable throughput guarantees in real-time applications. Of the panoply of devices deployed at the edge, resource-constrained MCUs, e.g., Arm Cortex-M, are more prevalent, orders of magnitude cheaper, and less power-hungry than application processors or GPUs. Thus, enabling intelligence at the deep edge is the zeitgeist, with researchers focusing on unveiling novel approaches to deploy ANNs on these constrained devices. Quantization is a well-established technique that has proved effective in enabling the deployment of neural networks on MCUs; however, it is still an open question to understand the robustness of QNNs in the face of adversarial examples.   To fill this gap, we empirically evaluate the effectiveness of attacks and defenses from (full-precision) ANNs on (constrained) QNNs. Our evaluation includes three QNNs targeting TinyML applications, ten attacks, and six defenses. With this study, we draw a set of interesting findings. First, quantization increases the point distance to the decision boundary and leads the gradient estimated by some attacks to explode or vanish. Second, quantization can act as a noise attenuator or amplifier, depending on the noise magnitude, and causes gradient misalignment. Regarding adversarial defenses, we conclude that input pre-processing defenses show impressive results on small perturbations; however, they fall short as the perturbation increases. At the same time, train-based defenses increase the average point distance to the decision boundary, which holds after quantization. However, we argue that train-based defenses still need to smooth the quantization-shift and gradient misalignment phenomenons to counteract adversarial example transferability to QNNs. All artifacts are open-sourced to enable independent validation of results.

摘要: ML正在从云端转移到边缘。边缘计算减少了暴露私有数据的表面，并在实时应用中实现了可靠的吞吐量保证。在部署在边缘的所有设备中，资源受限的MCU(例如ARM Cortex-M)比应用处理器或GPU更普遍、更便宜、耗电量更低。因此，在深层实现智能是时代的精神，研究人员专注于推出在这些受限设备上部署ANN的新方法。量化是一种成熟的技术，已被证明在MCU上部署神经网络是有效的；然而，面对敌对例子，理解QNN的稳健性仍然是一个悬而未决的问题。为了填补这一空白，我们从经验上评估了(全精度)人工神经网络对(受约束的)QNN的攻击和防御的有效性。我们的评估包括三个针对TinyML应用程序的QNN，十个攻击和六个防御。通过这项研究，我们得出了一系列有趣的发现。首先，量化增加了到决策边界的点距离，并导致某些攻击估计的梯度爆炸或消失。其次，量化可以充当噪声衰减器或放大器，这取决于噪声的大小，并导致梯度失调。对于对抗性防御，我们得出的结论是，输入预处理防御在小扰动下表现出令人印象深刻的结果；然而，随着扰动的增加，它们不能满足要求。同时，基于训练的防御增加了到决策边界的平均点距离，量化后该距离保持不变。然而，我们认为，基于训练的防御仍然需要平滑量化位移和梯度错位现象，以抵消向QNN的对抗性示例转移。所有构件都是开源的，以支持结果的独立验证。



## **3. Investigating the Impact of Quantization on Adversarial Robustness**

研究量化对对抗鲁棒性的影响 cs.LG

Accepted to ICLR 2024 Workshop PML4LRS

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05639v1) [paper-pdf](http://arxiv.org/pdf/2404.05639v1)

**Authors**: Qun Li, Yuan Meng, Chen Tang, Jiacheng Jiang, Zhi Wang

**Abstract**: Quantization is a promising technique for reducing the bit-width of deep models to improve their runtime performance and storage efficiency, and thus becomes a fundamental step for deployment. In real-world scenarios, quantized models are often faced with adversarial attacks which cause the model to make incorrect inferences by introducing slight perturbations. However, recent studies have paid less attention to the impact of quantization on the model robustness. More surprisingly, existing studies on this topic even present inconsistent conclusions, which prompted our in-depth investigation. In this paper, we conduct a first-time analysis of the impact of the quantization pipeline components that can incorporate robust optimization under the settings of Post-Training Quantization and Quantization-Aware Training. Through our detailed analysis, we discovered that this inconsistency arises from the use of different pipelines in different studies, specifically regarding whether robust optimization is performed and at which quantization stage it occurs. Our research findings contribute insights into deploying more secure and robust quantized networks, assisting practitioners in reference for scenarios with high-security requirements and limited resources.

摘要: 量化是一种很有前途的技术，可以减少深度模型的位宽，从而提高其运行时性能和存储效率，因此成为部署的基础步骤。在现实场景中，量化模型经常面临敌意攻击，通过引入微小的扰动，导致模型做出不正确的推断。然而，最近的研究较少关注量化对模型稳健性的影响。更令人惊讶的是，现有的研究甚至得出了不一致的结论，这促使我们进行了深入的调查。在本文中，我们首次分析了在训练后量化和量化感知训练的情况下，能够结合稳健优化的量化流水线组件的影响。通过我们的详细分析，我们发现这种不一致是由于在不同的研究中使用了不同的流水线，特别是关于是否进行了稳健优化以及它发生在哪个量化阶段。我们的研究成果有助于深入了解如何部署更安全、更强大的量化网络，帮助实践者在具有高安全性要求和资源有限的场景中进行参考。



## **4. SoK: Gradient Leakage in Federated Learning**

SoK：联邦学习中的梯度泄漏 cs.CR

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05403v1) [paper-pdf](http://arxiv.org/pdf/2404.05403v1)

**Authors**: Jiacheng Du, Jiahui Hu, Zhibo Wang, Peng Sun, Neil Zhenqiang Gong, Kui Ren

**Abstract**: Federated learning (FL) enables collaborative model training among multiple clients without raw data exposure. However, recent studies have shown that clients' private training data can be reconstructed from the gradients they share in FL, known as gradient inversion attacks (GIAs). While GIAs have demonstrated effectiveness under \emph{ideal settings and auxiliary assumptions}, their actual efficacy against \emph{practical FL systems} remains under-explored. To address this gap, we conduct a comprehensive study on GIAs in this work. We start with a survey of GIAs that establishes a milestone to trace their evolution and develops a systematization to uncover their inherent threats. Specifically, we categorize the auxiliary assumptions used by existing GIAs based on their practical accessibility to potential adversaries. To facilitate deeper analysis, we highlight the challenges that GIAs face in practical FL systems from three perspectives: \textit{local training}, \textit{model}, and \textit{post-processing}. We then perform extensive theoretical and empirical evaluations of state-of-the-art GIAs across diverse settings, utilizing eight datasets and thirteen models. Our findings indicate that GIAs have inherent limitations when reconstructing data under practical local training settings. Furthermore, their efficacy is sensitive to the trained model, and even simple post-processing measures applied to gradients can be effective defenses. Overall, our work provides crucial insights into the limited effectiveness of GIAs in practical FL systems. By rectifying prior misconceptions, we hope to inspire more accurate and realistic investigations on this topic.

摘要: 联合学习(FL)实现了多个客户之间的协作模型培训，而不会暴露原始数据。然而，最近的研究表明，客户的私人训练数据可以从他们在FL中共享的梯度重建，称为梯度反转攻击(GIA)。虽然GIA已经在理想环境和辅助假设下证明了其有效性，但它们对实际FL系统的实际有效性仍未得到充分研究。为了弥补这一差距，我们在这项工作中对GIA进行了全面的研究。我们从对GIA的调查开始，建立了一个里程碑来跟踪它们的演变，并制定了一个系统化的方法来揭示它们的内在威胁。具体地说，我们根据现有GIA对潜在对手的实际可访问性对其使用的辅助假设进行分类。为了便于更深入的分析，我们从三个角度强调了GIA在实际外语系统中所面临的挑战：\textit{本地训练}、\textit{模型}和\textit{后处理}。然后，我们利用8个数据集和13个模型，在不同的环境中对最先进的GIA进行广泛的理论和经验评估。我们的发现表明，在实际的本地训练环境下，GIA在重建数据时存在固有的局限性。此外，它们的有效性对训练的模型很敏感，甚至对梯度应用简单的后处理措施也可以成为有效的防御措施。总体而言，我们的工作对GIA在实际外语系统中的有限有效性提供了至关重要的见解。通过纠正之前的误解，我们希望启发对这一主题的更准确和更现实的调查。



## **5. BruSLeAttack: A Query-Efficient Score-Based Black-Box Sparse Adversarial Attack**

BruSLeAttack：一种基于分数的查询高效黑盒稀疏对抗攻击 cs.LG

Published as a conference paper at the International Conference on  Learning Representations (ICLR 2024). Code is available at  https://brusliattack.github.io/

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05311v1) [paper-pdf](http://arxiv.org/pdf/2404.05311v1)

**Authors**: Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: We study the unique, less-well understood problem of generating sparse adversarial samples simply by observing the score-based replies to model queries. Sparse attacks aim to discover a minimum number-the l0 bounded-perturbations to model inputs to craft adversarial examples and misguide model decisions. But, in contrast to query-based dense attack counterparts against black-box models, constructing sparse adversarial perturbations, even when models serve confidence score information to queries in a score-based setting, is non-trivial. Because, such an attack leads to i) an NP-hard problem; and ii) a non-differentiable search space. We develop the BruSLeAttack-a new, faster (more query-efficient) Bayesian algorithm for the problem. We conduct extensive attack evaluations including an attack demonstration against a Machine Learning as a Service (MLaaS) offering exemplified by Google Cloud Vision and robustness testing of adversarial training regimes and a recent defense against black-box attacks. The proposed attack scales to achieve state-of-the-art attack success rates and query efficiency on standard computer vision tasks such as ImageNet across different model architectures. Our artefacts and DIY attack samples are available on GitHub. Importantly, our work facilitates faster evaluation of model vulnerabilities and raises our vigilance on the safety, security and reliability of deployed systems.

摘要: 我们简单地通过观察对模型查询的基于分数的回复来研究生成稀疏对抗性样本的独特的、较少被理解的问题。稀疏攻击的目的是发现最小数量的--10个有界的--扰动，以对输入进行建模，以制造敌意的例子并误导模型决策。但是，与基于查询的密集攻击对应的黑盒模型相比，构建稀疏的对抗性扰动，即使当模型在基于分数的设置中向查询提供置信度分数信息时，也不是微不足道的。因为，这样的攻击导致i)NP-Hard问题；以及ii)不可微搜索空间。我们开发了BruSLeAttack-一种新的、更快(查询效率更高)的贝叶斯算法。我们进行广泛的攻击评估，包括针对机器学习即服务(MLaaS)产品的攻击演示，例如Google Cloud Vision和对抗性训练机制的健壮性测试，以及最近针对黑盒攻击的防御。建议的攻击规模可跨不同的模型架构在标准计算机视觉任务(如ImageNet)上实现最先进的攻击成功率和查询效率。我们的手工艺品和DIY攻击样本可以在GitHub上找到。重要的是，我们的工作有助于更快地评估模型漏洞，并提高我们对已部署系统的安全性、安全性和可靠性的警惕。



## **6. Out-of-Distribution Data: An Acquaintance of Adversarial Examples -- A Survey**

非分布数据：对抗性实例的认识——一项调查 cs.LG

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05219v1) [paper-pdf](http://arxiv.org/pdf/2404.05219v1)

**Authors**: Naveen Karunanayake, Ravin Gunawardena, Suranga Seneviratne, Sanjay Chawla

**Abstract**: Deep neural networks (DNNs) deployed in real-world applications can encounter out-of-distribution (OOD) data and adversarial examples. These represent distinct forms of distributional shifts that can significantly impact DNNs' reliability and robustness. Traditionally, research has addressed OOD detection and adversarial robustness as separate challenges. This survey focuses on the intersection of these two areas, examining how the research community has investigated them together. Consequently, we identify two key research directions: robust OOD detection and unified robustness. Robust OOD detection aims to differentiate between in-distribution (ID) data and OOD data, even when they are adversarially manipulated to deceive the OOD detector. Unified robustness seeks a single approach to make DNNs robust against both adversarial attacks and OOD inputs. Accordingly, first, we establish a taxonomy based on the concept of distributional shifts. This framework clarifies how robust OOD detection and unified robustness relate to other research areas addressing distributional shifts, such as OOD detection, open set recognition, and anomaly detection. Subsequently, we review existing work on robust OOD detection and unified robustness. Finally, we highlight the limitations of the existing work and propose promising research directions that explore adversarial and OOD inputs within a unified framework.

摘要: 在实际应用中部署的深度神经网络(DNN)可能会遇到分布外(OOD)数据和敌意示例。这些代表了不同形式的分布变化，可以显著影响DNN的可靠性和稳健性。传统上，研究将OOD检测和对手健壮性作为单独的挑战。这项调查聚焦于这两个领域的交集，考察了研究界是如何一起调查这两个领域的。因此，我们确定了两个关键的研究方向：稳健的面向对象检测和统一的稳健性。稳健的OOD检测旨在区分分布内(ID)数据和OOD数据，即使它们被相反地操纵以欺骗OOD检测器。统一健壮性寻求一种单一的方法来使DNN对对手攻击和OOD输入都具有健壮性。因此，首先，我们建立了基于分布移位概念的分类。该框架阐明了健壮的OOD检测和统一的健壮性如何与解决分布迁移的其他研究领域相关，例如OOD检测、开集识别和异常检测。随后，我们回顾了健壮性面向对象检测和统一健壮性方面的现有工作。最后，我们强调了现有工作的局限性，并提出了在统一框架内探索对抗性和OOD输入的有前途的研究方向。



## **7. Semantic Stealth: Adversarial Text Attacks on NLP Using Several Methods**

语义隐身：基于几种方法的NLP对抗性文本攻击 cs.CL

This report pertains to the Capstone Project done by Group 2 of the  Fall batch of 2023 students at Praxis Tech School, Kolkata, India. The  reports consists of 28 pages and it includes 10 tables. This is the preprint  which will be submitted to IEEE CONIT 2024 for review

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05159v1) [paper-pdf](http://arxiv.org/pdf/2404.05159v1)

**Authors**: Roopkatha Dey, Aivy Debnath, Sayak Kumar Dutta, Kaustav Ghosh, Arijit Mitra, Arghya Roy Chowdhury, Jaydip Sen

**Abstract**: In various real-world applications such as machine translation, sentiment analysis, and question answering, a pivotal role is played by NLP models, facilitating efficient communication and decision-making processes in domains ranging from healthcare to finance. However, a significant challenge is posed to the robustness of these natural language processing models by text adversarial attacks. These attacks involve the deliberate manipulation of input text to mislead the predictions of the model while maintaining human interpretability. Despite the remarkable performance achieved by state-of-the-art models like BERT in various natural language processing tasks, they are found to remain vulnerable to adversarial perturbations in the input text. In addressing the vulnerability of text classifiers to adversarial attacks, three distinct attack mechanisms are explored in this paper using the victim model BERT: BERT-on-BERT attack, PWWS attack, and Fraud Bargain's Attack (FBA). Leveraging the IMDB, AG News, and SST2 datasets, a thorough comparative analysis is conducted to assess the effectiveness of these attacks on the BERT classifier model. It is revealed by the analysis that PWWS emerges as the most potent adversary, consistently outperforming other methods across multiple evaluation scenarios, thereby emphasizing its efficacy in generating adversarial examples for text classification. Through comprehensive experimentation, the performance of these attacks is assessed and the findings indicate that the PWWS attack outperforms others, demonstrating lower runtime, higher accuracy, and favorable semantic similarity scores. The key insight of this paper lies in the assessment of the relative performances of three prevalent state-of-the-art attack mechanisms.

摘要: 在机器翻译、情感分析和问题回答等各种现实应用中，自然语言处理模型扮演着至关重要的角色，促进了从医疗保健到金融等领域的高效沟通和决策过程。然而，文本对抗攻击对这些自然语言处理模型的稳健性提出了重大挑战。这些攻击包括故意操纵输入文本以误导模型的预测，同时保持人类的可解释性。尽管像BERT这样的最先进的模型在各种自然语言处理任务中取得了显著的性能，但它们仍然容易受到输入文本中的对抗性干扰。针对文本分类器易受敌意攻击的问题，利用受害者模型BERT，探讨了三种不同的攻击机制：BERT-ON-BERT攻击、PWWS攻击和欺诈交易攻击(FBA)。利用IMDB、AG News和Sst2数据集，进行了全面的比较分析，以评估这些攻击对BERT分类器模型的有效性。分析表明，PWWS是最强大的对手，在多个评估场景中的表现一直优于其他方法，从而强调了它在生成用于文本分类的对抗性实例方面的有效性。通过综合实验，评估了这些攻击的性能，结果表明，PWWS攻击的性能优于其他攻击，表现出更低的运行时间、更高的准确率和良好的语义相似度得分。本文的重点在于对目前流行的三种攻击机制的相对性能进行评估。



## **8. Enabling Privacy-Preserving Cyber Threat Detection with Federated Learning**

利用联邦学习实现隐私保护网络威胁检测 cs.CR

**SubmitDate**: 2024-04-08    [abs](http://arxiv.org/abs/2404.05130v1) [paper-pdf](http://arxiv.org/pdf/2404.05130v1)

**Authors**: Yu Bi, Yekai Li, Xuan Feng, Xianghang Mi

**Abstract**: Despite achieving good performance and wide adoption, machine learning based security detection models (e.g., malware classifiers) are subject to concept drift and evasive evolution of attackers, which renders up-to-date threat data as a necessity. However, due to enforcement of various privacy protection regulations (e.g., GDPR), it is becoming increasingly challenging or even prohibitive for security vendors to collect individual-relevant and privacy-sensitive threat datasets, e.g., SMS spam/non-spam messages from mobile devices. To address such obstacles, this study systematically profiles the (in)feasibility of federated learning for privacy-preserving cyber threat detection in terms of effectiveness, byzantine resilience, and efficiency. This is made possible by the build-up of multiple threat datasets and threat detection models, and more importantly, the design of realistic and security-specific experiments.   We evaluate FL on two representative threat detection tasks, namely SMS spam detection and Android malware detection. It shows that FL-trained detection models can achieve a performance that is comparable to centrally trained counterparts. Also, most non-IID data distributions have either minor or negligible impact on the model performance, while a label-based non-IID distribution of a high extent can incur non-negligible fluctuation and delay in FL training. Then, under a realistic threat model, FL turns out to be adversary-resistant to attacks of both data poisoning and model poisoning. Particularly, the attacking impact of a practical data poisoning attack is no more than 0.14\% loss in model accuracy. Regarding FL efficiency, a bootstrapping strategy turns out to be effective to mitigate the training delay as observed in label-based non-IID scenarios.

摘要: 尽管获得了良好的性能和广泛的采用，基于机器学习的安全检测模型(例如恶意软件分类器)仍然受到攻击者概念漂移和回避演变的影响，这使得最新的威胁数据成为必要。然而，由于各种隐私保护法规的执行(例如，GDPR)，安全供应商收集与个人相关和隐私敏感的威胁数据集，例如来自移动设备的短信垃圾邮件/非垃圾邮件，正变得越来越具有挑战性，甚至令人望而却步。为了解决这些障碍，本研究从有效性、拜占庭复原力和效率三个方面系统地描述了联合学习用于隐私保护网络威胁检测的可行性。这是由于建立了多个威胁数据集和威胁检测模型，更重要的是，设计了现实的和特定于安全的实验。我们在两个有代表性的威胁检测任务上对FL进行了评估，即短信垃圾邮件检测和Android恶意软件检测。这表明，FL训练的检测模型可以获得与中央训练的同类模型相当的性能。此外，大多数非IID数据分布对模型性能的影响很小或可以忽略不计，而高范围的基于标签的非IID分布可能会在FL训练中引起不可忽略的波动和延迟。然后，在一个真实的威胁模型下，FL对数据中毒和模型中毒的攻击都具有抵抗能力。特别是，实际的数据中毒攻击对模型精度的影响不超过0.14。关于外语学习的效率，事实证明，在基于标签的非IID场景中，自举策略被证明是有效的减轻训练延迟。



## **9. Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations**

隐藏在普通视野中：对脆弱患者人群的不可检测的对抗偏见攻击 cs.LG

29 pages, 4 figures

**SubmitDate**: 2024-04-07    [abs](http://arxiv.org/abs/2402.05713v3) [paper-pdf](http://arxiv.org/pdf/2402.05713v3)

**Authors**: Pranav Kulkarni, Andrew Chan, Nithya Navarathna, Skylar Chan, Paul H. Yi, Vishwa S. Parekh

**Abstract**: The proliferation of artificial intelligence (AI) in radiology has shed light on the risk of deep learning (DL) models exacerbating clinical biases towards vulnerable patient populations. While prior literature has focused on quantifying biases exhibited by trained DL models, demographically targeted adversarial bias attacks on DL models and its implication in the clinical environment remains an underexplored field of research in medical imaging. In this work, we demonstrate that demographically targeted label poisoning attacks can introduce undetectable underdiagnosis bias in DL models. Our results across multiple performance metrics and demographic groups like sex, age, and their intersectional subgroups show that adversarial bias attacks demonstrate high-selectivity for bias in the targeted group by degrading group model performance without impacting overall model performance. Furthermore, our results indicate that adversarial bias attacks result in biased DL models that propagate prediction bias even when evaluated with external datasets.

摘要: 人工智能(AI)在放射学中的扩散揭示了深度学习(DL)模型的风险，加剧了对脆弱患者群体的临床偏见。虽然以前的文献集中于量化训练的DL模型所表现出的偏差，但针对人口统计目标的对DL模型的对抗性偏见攻击及其在临床环境中的应用仍然是医学成像领域中探索不足的研究领域。在这项工作中，我们证明了人口统计目标的标签中毒攻击可以在DL模型中引入不可检测的漏诊偏差。我们在多个性能指标和人口统计组(如性别、年龄及其相交的子组)上的结果表明，对抗性偏见攻击通过降低组模型性能而不影响整体模型性能，显示了对目标组中的偏见的高选择性。此外，我们的结果表明，对抗性偏差攻击导致有偏差的DL模型传播预测偏差，即使在使用外部数据集进行评估时也是如此。



## **10. NeuroIDBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research**

NeuroIDBench：一个基于脑波的认证研究方法标准化的开源基准框架 cs.CR

21 pages, 5 Figures, 3 tables, Submitted to the Journal of  Information Security and Applications

**SubmitDate**: 2024-04-07    [abs](http://arxiv.org/abs/2402.08656v3) [paper-pdf](http://arxiv.org/pdf/2402.08656v3)

**Authors**: Avinash Kumar Chaurasia, Matin Fallahi, Thorsten Strufe, Philipp Terhörst, Patricia Arias Cabarcos

**Abstract**: Biometric systems based on brain activity have been proposed as an alternative to passwords or to complement current authentication techniques. By leveraging the unique brainwave patterns of individuals, these systems offer the possibility of creating authentication solutions that are resistant to theft, hands-free, accessible, and potentially even revocable. However, despite the growing stream of research in this area, faster advance is hindered by reproducibility problems. Issues such as the lack of standard reporting schemes for performance results and system configuration, or the absence of common evaluation benchmarks, make comparability and proper assessment of different biometric solutions challenging. Further, barriers are erected to future work when, as so often, source code is not published open access. To bridge this gap, we introduce NeuroIDBench, a flexible open source tool to benchmark brainwave-based authentication models. It incorporates nine diverse datasets, implements a comprehensive set of pre-processing parameters and machine learning algorithms, enables testing under two common adversary models (known vs unknown attacker), and allows researchers to generate full performance reports and visualizations. We use NeuroIDBench to investigate the shallow classifiers and deep learning-based approaches proposed in the literature, and to test robustness across multiple sessions. We observe a 37.6% reduction in Equal Error Rate (EER) for unknown attacker scenarios (typically not tested in the literature), and we highlight the importance of session variability to brainwave authentication. All in all, our results demonstrate the viability and relevance of NeuroIDBench in streamlining fair comparisons of algorithms, thereby furthering the advancement of brainwave-based authentication through robust methodological practices.

摘要: 基于大脑活动的生物识别系统已经被提出作为密码的替代方案，或者是对当前身份验证技术的补充。通过利用个人独特的脑电波模式，这些系统提供了创建防盗、免提、可访问甚至可能可撤销的身份验证解决方案的可能性。然而，尽管这一领域的研究越来越多，但重复性问题阻碍了更快的进展。缺乏性能结果和系统配置的标准报告方案，或缺乏通用的评估基准等问题，使不同生物识别解决方案的可比性和适当评估具有挑战性。此外，当源代码不公开、开放获取时，就会为未来的工作设置障碍。为了弥补这一差距，我们引入了NeuroIDBch，这是一个灵活的开源工具，用于对基于脑电波的身份验证模型进行基准测试。它整合了九个不同的数据集，实现了一套全面的预处理参数和机器学习算法，可以在两个常见的对手模型(已知和未知攻击者)下进行测试，并允许研究人员生成完整的性能报告和可视化。我们使用NeuroIDB边来研究文献中提出的浅层分类器和基于深度学习的方法，并测试多个会话的健壮性。我们观察到，对于未知攻击者场景(通常未在文献中进行测试)，等错误率(EER)降低了37.6%，并强调了会话可变性对脑电波身份验证的重要性。总而言之，我们的结果证明了NeuroIDBtch在简化公平的算法比较方面的可行性和相关性，从而通过稳健的方法学实践进一步推进基于脑电波的身份验证。



## **11. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

披着羊皮的狼：广义嵌套越狱陷阱可以轻松愚弄大型语言模型 cs.CL

Acccepted by NAACL 2024, 18 pages, 7 figures, 13 tables

**SubmitDate**: 2024-04-07    [abs](http://arxiv.org/abs/2311.08268v4) [paper-pdf](http://arxiv.org/pdf/2311.08268v4)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, which compromises either generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.

摘要: 大型语言模型(LLM)，如ChatGPT和GPT-4，旨在提供有用和安全的响应。然而，被称为“越狱”的对抗性提示可能会绕过安全措施，导致LLMS生成潜在的有害内容。探索越狱提示可以帮助更好地揭示LLM的弱点，并进一步指导我们确保它们的安全。不幸的是，现有的越狱方法要么需要复杂的人工设计，要么需要对其他白盒模型进行优化，这要么损害了通用性，要么影响了效率。本文将越狱提示攻击概括为两个方面：(1)提示重写和(2)场景嵌套。在此基础上，我们提出了ReNeLLM，这是一个利用LLM自身生成有效越狱提示的自动化框架。广泛的实验表明，与现有的基准相比，ReNeLLM显著提高了攻击成功率，同时大大降低了时间成本。我们的研究也揭示了现有防御方法在保护低密度脂蛋白方面的不足。最后，从即时执行优先级的角度分析了LLMS防御失败的原因，并提出了相应的防御策略。我们希望我们的研究能够促进学术界和低成本管理系统开发商提供更安全和更规范的低成本管理系统。代码可在https://github.com/NJUNLP/ReNeLLM.上获得



## **12. Provable Robustness Against a Union of $\ell_0$ Adversarial Attacks**

针对$\ell_0 $对抗攻击联盟的可证明鲁棒性 cs.LG

Accepted at AAAI 2024 -- Extended version including the supplementary  material

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2302.11628v4) [paper-pdf](http://arxiv.org/pdf/2302.11628v4)

**Authors**: Zayd Hammoudeh, Daniel Lowd

**Abstract**: Sparse or $\ell_0$ adversarial attacks arbitrarily perturb an unknown subset of the features. $\ell_0$ robustness analysis is particularly well-suited for heterogeneous (tabular) data where features have different types or scales. State-of-the-art $\ell_0$ certified defenses are based on randomized smoothing and apply to evasion attacks only. This paper proposes feature partition aggregation (FPA) -- a certified defense against the union of $\ell_0$ evasion, backdoor, and poisoning attacks. FPA generates its stronger robustness guarantees via an ensemble whose submodels are trained on disjoint feature sets. Compared to state-of-the-art $\ell_0$ defenses, FPA is up to 3,000${\times}$ faster and provides larger median robustness guarantees (e.g., median certificates of 13 pixels over 10 for CIFAR10, 12 pixels over 10 for MNIST, 4 features over 1 for Weather, and 3 features over 1 for Ames), meaning FPA provides the additional dimensions of robustness essentially for free.

摘要: 稀疏或$\ell_0 $对抗攻击任意干扰特征的未知子集。$\ell_0 $鲁棒性分析特别适合于特征具有不同类型或规模的异构（表格）数据。最先进的$\ell_0 $认证防御基于随机平滑，仅适用于规避攻击。本文提出了特征分区聚合（FPA）——一种针对$\ell_0 $规避、后门和中毒攻击的认证防御方法。FPA通过一个子模型在不相交特征集上训练的集成来产生更强的鲁棒性保证。与最先进的$\ell_0 $防御相比，FPA速度最高可达3，000 ${\times}$，并提供更大的中值鲁棒性保证（例如，CIFAR10的中值证书为13个像素超过10，MNIST的中值证书为12个像素超过10，Weather的4个特征超过1，Ames的3个特征超过1），这意味着FPA基本上免费提供了额外的鲁棒性维度。



## **13. Data Poisoning Attacks on Off-Policy Policy Evaluation Methods**

非策略策略评估方法的数据中毒攻击 cs.LG

Accepted at UAI 2022

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2404.04714v1) [paper-pdf](http://arxiv.org/pdf/2404.04714v1)

**Authors**: Elita Lobo, Harvineet Singh, Marek Petrik, Cynthia Rudin, Himabindu Lakkaraju

**Abstract**: Off-policy Evaluation (OPE) methods are a crucial tool for evaluating policies in high-stakes domains such as healthcare, where exploration is often infeasible, unethical, or expensive. However, the extent to which such methods can be trusted under adversarial threats to data quality is largely unexplored. In this work, we make the first attempt at investigating the sensitivity of OPE methods to marginal adversarial perturbations to the data. We design a generic data poisoning attack framework leveraging influence functions from robust statistics to carefully construct perturbations that maximize error in the policy value estimates. We carry out extensive experimentation with multiple healthcare and control datasets. Our results demonstrate that many existing OPE methods are highly prone to generating value estimates with large errors when subject to data poisoning attacks, even for small adversarial perturbations. These findings question the reliability of policy values derived using OPE methods and motivate the need for developing OPE methods that are statistically robust to train-time data poisoning attacks.

摘要: 非政策评估(OPE)方法是评估高风险领域(如医疗保健)政策的重要工具，这些领域的探索通常是不可行、不道德或昂贵的。然而，在数据质量受到敌对威胁的情况下，这种方法在多大程度上可以得到信任，这在很大程度上是未知的。在这项工作中，我们首次尝试研究OPE方法对数据的边缘对抗性扰动的敏感性。我们设计了一个通用的数据中毒攻击框架，利用稳健统计中的影响函数来仔细构造扰动，使策略值估计的误差最大化。我们对多个医疗保健和对照数据集进行了广泛的实验。我们的结果表明，许多现有的OPE方法在受到数据中毒攻击时，即使是在小的对抗性扰动下，也很容易产生误差较大的值估计。这些发现质疑使用OPE方法得出的策略值的可靠性，并促使人们需要开发在统计上对训练时间数据中毒攻击具有健壮性的OPE方法。



## **14. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

红色团队游戏：红色团队语言模型的游戏理论框架 cs.CL

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2310.00322v4) [paper-pdf](http://arxiv.org/pdf/2310.00322v4)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.

摘要: 可部署的大型语言模型(LLMS)必须符合有益和无害的标准，从而实现LLMS的输出与人的价值之间的一致性。红团队技术构成了实现这一标准的关键途径。现有的工作完全依赖于手动红色团队设计和启发式对抗性提示来进行漏洞检测和优化。这些方法缺乏严格的数学描述，从而限制了在可量化的度量范围内探索多样化的攻击策略，以及在收敛保证下对LLMS进行优化。在本文中，我们提出了一种不需要人工注释的通用博弈论框架--Red-Teaming Game(RTG)。RTG用于分析红队语言模型(RLMS)和蓝队语言模型(BLM)之间的多回合攻防交互。在RTG中，我们提出了一种具有语义空间多样性度量的Gamalized Red-Teaming Solver(GRTS)。GRTS是一种自动红队技术，通过元博弈分析解决RTG向纳什均衡的方向，这对应于理论上保证的RLMS和BLM的优化方向。在RLMS多回合攻击中的实验结果表明，GRTS自主发现多样化的攻击策略，有效地提高了LLMS的安全性，优于已有的启发式红队设计。总体而言，RTG为红色团队任务建立了一个基本框架，并构建了一种新的可扩展的协调监督技术。



## **15. CANEDERLI: On The Impact of Adversarial Training and Transferability on CAN Intrusion Detection Systems**

CANEDERLI：对抗训练和传输性对CAN入侵检测系统的影响 cs.CR

Accepted at WiseML 2024

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2404.04648v1) [paper-pdf](http://arxiv.org/pdf/2404.04648v1)

**Authors**: Francesco Marchiori, Mauro Conti

**Abstract**: The growing integration of vehicles with external networks has led to a surge in attacks targeting their Controller Area Network (CAN) internal bus. As a countermeasure, various Intrusion Detection Systems (IDSs) have been suggested in the literature to prevent and mitigate these threats. With the increasing volume of data facilitated by the integration of Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I) communication networks, most of these systems rely on data-driven approaches such as Machine Learning (ML) and Deep Learning (DL) models. However, these systems are susceptible to adversarial evasion attacks. While many researchers have explored this vulnerability, their studies often involve unrealistic assumptions, lack consideration for a realistic threat model, and fail to provide effective solutions.   In this paper, we present CANEDERLI (CAN Evasion Detection ResiLIence), a novel framework for securing CAN-based IDSs. Our system considers a realistic threat model and addresses the impact of adversarial attacks on DL-based detection systems. Our findings highlight strong transferability properties among diverse attack methodologies by considering multiple state-of-the-art attacks and model architectures. We analyze the impact of adversarial training in addressing this threat and propose an adaptive online adversarial training technique outclassing traditional fine-tuning methodologies with F1 scores up to 0.941. By making our framework publicly available, we aid practitioners and researchers in assessing the resilience of IDSs to a varied adversarial landscape.

摘要: 车辆与外部网络的日益集成导致了针对其控制器区域网络(CAN)内部总线的攻击激增。作为一种对策，文献中提出了各种入侵检测系统(入侵检测系统)来预防和缓解这些威胁。随着车辆到车辆(V2V)和车辆到基础设施(V2I)通信网络的集成促进了数据量的增加，这些系统中的大多数依赖于数据驱动的方法，如机器学习(ML)和深度学习(DL)模型。然而，这些系统容易受到对抗性逃避攻击。虽然许多研究人员探索了这一漏洞，但他们的研究往往涉及不切实际的假设，缺乏对现实威胁模型的考虑，无法提供有效的解决方案。本文提出了一种新的基于CAN的入侵检测系统安全框架CANEDERLI(CAN Elevation Detect Resilience)。我们的系统考虑了一个现实的威胁模型，并解决了对抗性攻击对基于DL的检测系统的影响。通过考虑多个最先进的攻击和模型体系结构，我们的研究结果突出了不同攻击方法之间的强大可转移性。我们分析了对抗训练在应对这一威胁方面的影响，并提出了一种自适应在线对抗训练技术，其F1得分高达0.941分，超过了传统的微调方法。通过将我们的框架公之于众，我们帮助从业者和研究人员评估入侵检测系统对不同对手环境的韧性。



## **16. Dynamic Graph Information Bottleneck**

动态图形信息瓶颈 cs.LG

Accepted by the research tracks of The Web Conference 2024 (WWW 2024)

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2402.06716v3) [paper-pdf](http://arxiv.org/pdf/2402.06716v3)

**Authors**: Haonan Yuan, Qingyun Sun, Xingcheng Fu, Cheng Ji, Jianxin Li

**Abstract**: Dynamic Graphs widely exist in the real world, which carry complicated spatial and temporal feature patterns, challenging their representation learning. Dynamic Graph Neural Networks (DGNNs) have shown impressive predictive abilities by exploiting the intrinsic dynamics. However, DGNNs exhibit limited robustness, prone to adversarial attacks. This paper presents the novel Dynamic Graph Information Bottleneck (DGIB) framework to learn robust and discriminative representations. Leveraged by the Information Bottleneck (IB) principle, we first propose the expected optimal representations should satisfy the Minimal-Sufficient-Consensual (MSC) Condition. To compress redundant as well as conserve meritorious information into latent representation, DGIB iteratively directs and refines the structural and feature information flow passing through graph snapshots. To meet the MSC Condition, we decompose the overall IB objectives into DGIB$_{MS}$ and DGIB$_C$, in which the DGIB$_{MS}$ channel aims to learn the minimal and sufficient representations, with the DGIB$_{MS}$ channel guarantees the predictive consensus. Extensive experiments on real-world and synthetic dynamic graph datasets demonstrate the superior robustness of DGIB against adversarial attacks compared with state-of-the-art baselines in the link prediction task. To the best of our knowledge, DGIB is the first work to learn robust representations of dynamic graphs grounded in the information-theoretic IB principle.

摘要: 动态图形广泛存在于现实世界中，携带着复杂的时空特征模式，对其表示学习提出了挑战。动态图神经网络(DGNN)利用其内在的动力学特性，表现出了令人印象深刻的预测能力。然而，DGNN表现出有限的健壮性，容易受到对抗性攻击。本文提出了一种新的动态图信息瓶颈(DGIB)框架来学习稳健和区分的表示。利用信息瓶颈(IB)原理，我们首先提出期望的最优表示应该满足最小-充分-一致(MSC)条件。为了压缩冗余信息并将有价值的信息保存到潜在表示中，DGIB迭代地指导和细化通过图快照传递的结构和特征信息流。为了满足MSC条件，我们将整体IB目标分解为DGIB${MS}$和DGIB$_C$，其中DGIB${MS}$通道旨在学习最小且充分的表示，而DGIB${MS}$通道保证预测共识。在真实世界和合成动态图数据集上的大量实验表明，与链接预测任务中的最新基线相比，DGIB对对手攻击具有更好的稳健性。据我们所知，DGIB是第一个基于信息论IB原理学习动态图的稳健表示的工作。



## **17. Recovery from Adversarial Attacks in Cyber-physical Systems: Shallow, Deep and Exploratory Works**

网络物理系统中对抗性攻击的恢复：浅、深和探索性工作 eess.SY

**SubmitDate**: 2024-04-06    [abs](http://arxiv.org/abs/2404.04472v1) [paper-pdf](http://arxiv.org/pdf/2404.04472v1)

**Authors**: Pengyuan Lu, Lin Zhang, Mengyu Liu, Kaustubh Sridhar, Fanxin Kong, Oleg Sokolsky, Insup Lee

**Abstract**: Cyber-physical systems (CPS) have experienced rapid growth in recent decades. However, like any other computer-based systems, malicious attacks evolve mutually, driving CPS to undesirable physical states and potentially causing catastrophes. Although the current state-of-the-art is well aware of this issue, the majority of researchers have not focused on CPS recovery, the procedure we defined as restoring a CPS's physical state back to a target condition under adversarial attacks. To call for attention on CPS recovery and identify existing efforts, we have surveyed a total of 30 relevant papers. We identify a major partition of the proposed recovery strategies: shallow recovery vs. deep recovery, where the former does not use a dedicated recovery controller while the latter does. Additionally, we surveyed exploratory research on topics that facilitate recovery. From these publications, we discuss the current state-of-the-art of CPS recovery, with respect to applications, attack type, attack surfaces and system dynamics. Then, we identify untouched sub-domains in this field and suggest possible future directions for researchers.

摘要: 近几十年来，网络物理系统(CP)经历了快速增长。然而，与任何其他基于计算机的系统一样，恶意攻击也会相互演化，将CP推向不受欢迎的物理状态，并可能造成灾难。虽然目前的研究水平已经很好地意识到了这一问题，但大多数研究人员并没有关注CPS的恢复，我们定义的CPS恢复过程是指在对手攻击下将CPS的物理状态恢复到目标状态。为了唤起人们对CPS恢复的关注，并确定现有的努力，我们总共调查了30篇相关论文。我们确定了建议的恢复策略的主要部分：浅恢复和深度恢复，前者不使用专用恢复控制器，而后者使用。此外，我们对促进恢复的主题进行了探索性研究。从这些出版物中，我们从应用程序、攻击类型、攻击面和系统动力学方面讨论了CPS恢复的最新技术。然后，我们确定了该领域中未触及的子域，并为研究人员提出了可能的未来方向。



## **18. Increased LLM Vulnerabilities from Fine-tuning and Quantization**

从微调和量化增加LLM漏洞 cs.CR

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.04392v1) [paper-pdf](http://arxiv.org/pdf/2404.04392v1)

**Authors**: Divyanshu Kumar, Anurakt Kumar, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Large Language Models (LLMs) have become very popular and have found use cases in many domains, such as chatbots, auto-task completion agents, and much more. However, LLMs are vulnerable to different types of attacks, such as jailbreaking, prompt injection attacks, and privacy leakage attacks. Foundational LLMs undergo adversarial and alignment training to learn not to generate malicious and toxic content. For specialized use cases, these foundational LLMs are subjected to fine-tuning or quantization for better performance and efficiency. We examine the impact of downstream tasks such as fine-tuning and quantization on LLM vulnerability. We test foundation models like Mistral, Llama, MosaicML, and their fine-tuned versions. Our research shows that fine-tuning and quantization reduces jailbreak resistance significantly, leading to increased LLM vulnerabilities. Finally, we demonstrate the utility of external guardrails in reducing LLM vulnerabilities.

摘要: 大型语言模型（LLM）已经变得非常流行，并在许多领域找到了用例，如聊天机器人，自动任务完成代理等等。然而，LLM容易受到不同类型的攻击，例如越狱、即时注入攻击和隐私泄漏攻击。基础LLM接受对抗和对齐培训，学习不生成恶意和有毒内容。对于特殊的用例，这些基本的LLM需要经过微调或量化，以获得更好的性能和效率。我们研究了下游任务的影响，如微调和量化LLM脆弱性。我们测试了Mistral、Llama、MosaicML等基础模型，以及它们的微调版本。我们的研究表明，微调和量化显著降低了越狱阻力，导致LLM漏洞增加。最后，我们演示了外部防护措施在减少LLM漏洞方面的效用。



## **19. Compositional Estimation of Lipschitz Constants for Deep Neural Networks**

深度神经网络Lipschitz常数的合成估计 cs.LG

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.04375v1) [paper-pdf](http://arxiv.org/pdf/2404.04375v1)

**Authors**: Yuezhu Xu, S. Sivaranjani

**Abstract**: The Lipschitz constant plays a crucial role in certifying the robustness of neural networks to input perturbations and adversarial attacks, as well as the stability and safety of systems with neural network controllers. Therefore, estimation of tight bounds on the Lipschitz constant of neural networks is a well-studied topic. However, typical approaches involve solving a large matrix verification problem, the computational cost of which grows significantly for deeper networks. In this letter, we provide a compositional approach to estimate Lipschitz constants for deep feedforward neural networks by obtaining an exact decomposition of the large matrix verification problem into smaller sub-problems. We further obtain a closed-form solution that applies to most common neural network activation functions, which will enable rapid robustness and stability certificates for neural networks deployed in online control settings. Finally, we demonstrate through numerical experiments that our approach provides a steep reduction in computation time while yielding Lipschitz bounds that are very close to those achieved by state-of-the-art approaches.

摘要: Lipschitz常数在证明神经网络对输入扰动和敌意攻击的稳健性以及具有神经网络控制器的系统的稳定性和安全性方面起着至关重要的作用。因此，神经网络Lipschitz常数的紧界估计是一个很好的研究课题。然而，典型的方法涉及解决大型矩阵验证问题，对于更深层次的网络，其计算成本显著增加。在这封信中，我们提供了一种组合方法来估计深度前馈神经网络的Lipschitz常数，方法是将大的矩阵验证问题精确地分解成更小的子问题。我们进一步得到了适用于大多数常见神经网络激活函数的封闭形式的解，这将为在线控制环境中部署的神经网络提供快速的健壮性和稳定性证书。最后，我们通过数值实验证明，我们的方法大大减少了计算时间，而得到的Lipschitz界与最先进的方法所达到的非常接近。



## **20. Dissecting Distribution Inference**

剖析分布推理 cs.LG

Accepted at SaTML 2023 (updated Yifu's email address)

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2212.07591v2) [paper-pdf](http://arxiv.org/pdf/2212.07591v2)

**Authors**: Anshuman Suri, Yifu Lu, Yanjin Chen, David Evans

**Abstract**: A distribution inference attack aims to infer statistical properties of data used to train machine learning models. These attacks are sometimes surprisingly potent, but the factors that impact distribution inference risk are not well understood and demonstrated attacks often rely on strong and unrealistic assumptions such as full knowledge of training environments even in supposedly black-box threat scenarios. To improve understanding of distribution inference risks, we develop a new black-box attack that even outperforms the best known white-box attack in most settings. Using this new attack, we evaluate distribution inference risk while relaxing a variety of assumptions about the adversary's knowledge under black-box access, like known model architectures and label-only access. Finally, we evaluate the effectiveness of previously proposed defenses and introduce new defenses. We find that although noise-based defenses appear to be ineffective, a simple re-sampling defense can be highly effective. Code is available at https://github.com/iamgroot42/dissecting_distribution_inference

摘要: 分布推断攻击旨在推断用于训练机器学习模型的数据的统计特性。这些攻击有时威力惊人，但影响分布推断风险的因素并未得到很好的理解，已证明的攻击往往依赖于强大而不切实际的假设，例如完全了解训练环境，即使在假设的黑箱威胁场景中也是如此。为了提高对分布推断风险的理解，我们开发了一种新的黑盒攻击，该攻击在大多数情况下甚至比最著名的白盒攻击性能更好。使用这种新的攻击，我们评估了分布推断风险，同时放松了在黑盒访问下关于对手知识的各种假设，如已知的模型体系结构和仅标签访问。最后，我们评估了以前提出的防御措施的有效性，并引入了新的防御措施。我们发现，尽管基于噪声的防御似乎无效，但简单的重新采样防御可以非常有效。代码可在https://github.com/iamgroot42/dissecting_distribution_inference上找到



## **21. Evaluating Adversarial Robustness: A Comparison Of FGSM, Carlini-Wagner Attacks, And The Role of Distillation as Defense Mechanism**

评估对抗鲁棒性：FGSM、Carlini—Wagner攻击的比较以及蒸馏作为防御机制的作用 cs.CR

This report pertains to the Capstone Project done by Group 1 of the  Fall batch of 2023 students at Praxis Tech School, Kolkata, India. The  reports consists of 35 pages and it includes 15 figures and 10 tables. This  is the preprint which will be submitted to to an IEEE international  conference for review

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.04245v1) [paper-pdf](http://arxiv.org/pdf/2404.04245v1)

**Authors**: Trilokesh Ranjan Sarkar, Nilanjan Das, Pralay Sankar Maitra, Bijoy Some, Ritwik Saha, Orijita Adhikary, Bishal Bose, Jaydip Sen

**Abstract**: This technical report delves into an in-depth exploration of adversarial attacks specifically targeted at Deep Neural Networks (DNNs) utilized for image classification. The study also investigates defense mechanisms aimed at bolstering the robustness of machine learning models. The research focuses on comprehending the ramifications of two prominent attack methodologies: the Fast Gradient Sign Method (FGSM) and the Carlini-Wagner (CW) approach. These attacks are examined concerning three pre-trained image classifiers: Resnext50_32x4d, DenseNet-201, and VGG-19, utilizing the Tiny-ImageNet dataset. Furthermore, the study proposes the robustness of defensive distillation as a defense mechanism to counter FGSM and CW attacks. This defense mechanism is evaluated using the CIFAR-10 dataset, where CNN models, specifically resnet101 and Resnext50_32x4d, serve as the teacher and student models, respectively. The proposed defensive distillation model exhibits effectiveness in thwarting attacks such as FGSM. However, it is noted to remain susceptible to more sophisticated techniques like the CW attack. The document presents a meticulous validation of the proposed scheme. It provides detailed and comprehensive results, elucidating the efficacy and limitations of the defense mechanisms employed. Through rigorous experimentation and analysis, the study offers insights into the dynamics of adversarial attacks on DNNs, as well as the effectiveness of defensive strategies in mitigating their impact.

摘要: 这份技术报告深入探讨了专门针对用于图像分类的深度神经网络(DNN)的对抗性攻击。该研究还调查了旨在增强机器学习模型稳健性的防御机制。研究重点在于理解两种主要的攻击方法：快速梯度符号法(FGSM)和卡里尼-瓦格纳(CW)方法。这些攻击涉及三个预先训练的图像分类器：Resnext50_32x4d、DenseNet-201和VGG-19，使用Tiny-ImageNet数据集。此外，研究还提出了防御蒸馏作为对抗FGSM和CW攻击的一种防御机制的健壮性。这一防御机制使用CIFAR-10数据集进行了评估，其中CNN模型，特别是resnet101和Resnext50_32x4d分别充当教师模型和学生模型。所提出的防御蒸馏模型在挫败FGSM等攻击方面表现出了有效性。然而，值得注意的是，它仍然容易受到更复杂的技术的影响，比如CW攻击。该文件对提议的方案进行了细致的验证。它提供了详细和全面的结果，阐明了所采用的防御机制的有效性和局限性。通过严格的实验和分析，这项研究提供了对DNN的对抗性攻击的动态以及防御策略在减轻其影响方面的有效性的见解。



## **22. On Inherent Adversarial Robustness of Active Vision Systems**

主动视觉系统的固有对抗鲁棒性 cs.CV

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.00185v2) [paper-pdf](http://arxiv.org/pdf/2404.00185v2)

**Authors**: Amitangshu Mukherjee, Timur Ibrayev, Kaushik Roy

**Abstract**: Current Deep Neural Networks are vulnerable to adversarial examples, which alter their predictions by adding carefully crafted noise. Since human eyes are robust to such inputs, it is possible that the vulnerability stems from the standard way of processing inputs in one shot by processing every pixel with the same importance. In contrast, neuroscience suggests that the human vision system can differentiate salient features by (1) switching between multiple fixation points (saccades) and (2) processing the surrounding with a non-uniform external resolution (foveation). In this work, we advocate that the integration of such active vision mechanisms into current deep learning systems can offer robustness benefits. Specifically, we empirically demonstrate the inherent robustness of two active vision methods - GFNet and FALcon - under a black box threat model. By learning and inferencing based on downsampled glimpses obtained from multiple distinct fixation points within an input, we show that these active methods achieve (2-3) times greater robustness compared to a standard passive convolutional network under state-of-the-art adversarial attacks. More importantly, we provide illustrative and interpretable visualization analysis that demonstrates how performing inference from distinct fixation points makes active vision methods less vulnerable to malicious inputs.

摘要: 当前的深度神经网络很容易受到敌意例子的影响，这些例子通过添加精心设计的噪声来改变它们的预测。由于人眼对这样的输入很健壮，这种漏洞可能源于一次处理输入的标准方式，即处理具有相同重要性的每个像素。相比之下，神经科学表明，人类的视觉系统可以通过(1)在多个注视点(眼跳)之间切换和(2)用非均匀的外部分辨率(中心凹)处理周围环境来区分显著特征。在这项工作中，我们主张将这种主动视觉机制集成到当前的深度学习系统中，可以提供健壮性优势。具体而言，我们通过实验验证了两种主动视觉方法--GFNet和Falcon--在黑匣子威胁模型下的内在稳健性。通过基于从输入内多个不同固定点获得的下采样一瞥的学习和推理，我们表明这些主动方法在最先进的对抗攻击下比标准的被动卷积网络获得(2-3)倍的健壮性。更重要的是，我们提供了说明性和可解释性的可视化分析，演示了如何从不同的注视点执行推理使主动视觉方法不太容易受到恶意输入的影响。



## **23. Reliable Feature Selection for Adversarially Robust Cyber-Attack Detection**

对抗鲁棒网络攻击检测的可靠特征选择 cs.CR

24 pages, 17 tables, Annals of Telecommunications journal. arXiv  admin note: substantial text overlap with arXiv:2402.16912

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.04188v1) [paper-pdf](http://arxiv.org/pdf/2404.04188v1)

**Authors**: João Vitorino, Miguel Silva, Eva Maia, Isabel Praça

**Abstract**: The growing cybersecurity threats make it essential to use high-quality data to train Machine Learning (ML) models for network traffic analysis, without noisy or missing data. By selecting the most relevant features for cyber-attack detection, it is possible to improve both the robustness and computational efficiency of the models used in a cybersecurity system. This work presents a feature selection and consensus process that combines multiple methods and applies them to several network datasets. Two different feature sets were selected and were used to train multiple ML models with regular and adversarial training. Finally, an adversarial evasion robustness benchmark was performed to analyze the reliability of the different feature sets and their impact on the susceptibility of the models to adversarial examples. By using an improved dataset with more data diversity, selecting the best time-related features and a more specific feature set, and performing adversarial training, the ML models were able to achieve a better adversarially robust generalization. The robustness of the models was significantly improved without their generalization to regular traffic flows being affected, without increases of false alarms, and without requiring too many computational resources, which enables a reliable detection of suspicious activity and perturbed traffic flows in enterprise computer networks.

摘要: 日益增长的网络安全威胁使得使用高质量的数据来训练机器学习(ML)模型以进行网络流量分析变得至关重要，而不会产生噪声或丢失数据。通过选择与网络攻击检测最相关的特征，可以提高网络安全系统中使用的模型的稳健性和计算效率。这项工作提出了一种特征选择和共识过程，该过程结合了多种方法，并将它们应用于几个网络数据集。选择了两个不同的特征集，并用它们对多个ML模型进行了常规和对抗性训练。最后，通过对抗性回避健壮性基准测试，分析了不同特征集的可靠性及其对模型对对抗性例子敏感性的影响。通过使用具有更多数据多样性的改进数据集，选择与时间相关的最佳特征和更具体的特征集，并执行对抗性训练，ML模型能够实现更好的对抗性健壮性泛化。在不影响其对常规业务流的泛化、不增加错误警报和不需要太多计算资源的情况下，模型的稳健性显著提高，这使得能够可靠地检测企业计算机网络中的可疑活动和受干扰的业务流。



## **24. Foundations of Cyber Resilience: The Confluence of Game, Control, and Learning Theories**

网络弹性的基础：游戏、控制和学习理论的融合 eess.SY

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.01205v2) [paper-pdf](http://arxiv.org/pdf/2404.01205v2)

**Authors**: Quanyan Zhu

**Abstract**: Cyber resilience is a complementary concept to cybersecurity, focusing on the preparation, response, and recovery from cyber threats that are challenging to prevent. Organizations increasingly face such threats in an evolving cyber threat landscape. Understanding and establishing foundations for cyber resilience provide a quantitative and systematic approach to cyber risk assessment, mitigation policy evaluation, and risk-informed defense design. A systems-scientific view toward cyber risks provides holistic and system-level solutions. This chapter starts with a systemic view toward cyber risks and presents the confluence of game theory, control theory, and learning theories, which are three major pillars for the design of cyber resilience mechanisms to counteract increasingly sophisticated and evolving threats in our networks and organizations. Game and control theoretic methods provide a set of modeling frameworks to capture the strategic and dynamic interactions between defenders and attackers. Control and learning frameworks together provide a feedback-driven mechanism that enables autonomous and adaptive responses to threats. Game and learning frameworks offer a data-driven approach to proactively reason about adversarial behaviors and resilient strategies. The confluence of the three lays the theoretical foundations for the analysis and design of cyber resilience. This chapter presents various theoretical paradigms, including dynamic asymmetric games, moving horizon control, conjectural learning, and meta-learning, as recent advances at the intersection. This chapter concludes with future directions and discussions of the role of neurosymbolic learning and the synergy between foundation models and game models in cyber resilience.

摘要: 网络复原力是网络安全的补充概念，侧重于预防具有挑战性的网络威胁的准备、响应和恢复。在不断变化的网络威胁环境中，组织面临的此类威胁越来越多。理解和建立网络复原力的基础为网络风险评估、缓解政策评估和风险知情防御设计提供了一种量化和系统的方法。系统科学的网络风险观提供了整体和系统级的解决方案。本章从系统地看待网络风险开始，介绍了博弈论、控制论和学习理论的融合，这三个理论是设计网络弹性机制的三大支柱，以对抗我们网络和组织中日益复杂和不断变化的威胁。博弈论和控制论方法提供了一套模型框架来捕捉防御者和攻击者之间的战略和动态交互。控制和学习框架共同提供了一种反馈驱动的机制，使其能够对威胁做出自主和适应性的反应。游戏和学习框架提供了一种数据驱动的方法来主动推理对手行为和弹性策略。三者的融合为网络韧性的分析和设计奠定了理论基础。本章介绍了各种理论范式，包括动态不对称博弈、移动视野控制、猜想学习和元学习，作为交叉路口的最新进展。本章最后对神经符号学习的作用以及基础模型和游戏模型在网络韧性中的协同作用进行了未来的方向和讨论。



## **25. Less is More: Understanding Word-level Textual Adversarial Attack via n-gram Frequency Descend**

Less is More：通过n—gram频率分解理解单词级文本对抗攻击 cs.CL

To be published in: 2024 IEEE Conference on Artificial Intelligence  (CAI 2024)

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2302.02568v3) [paper-pdf](http://arxiv.org/pdf/2302.02568v3)

**Authors**: Ning Lu, Shengcai Liu, Zhirui Zhang, Qi Wang, Haifeng Liu, Ke Tang

**Abstract**: Word-level textual adversarial attacks have demonstrated notable efficacy in misleading Natural Language Processing (NLP) models. Despite their success, the underlying reasons for their effectiveness and the fundamental characteristics of adversarial examples (AEs) remain obscure. This work aims to interpret word-level attacks by examining their $n$-gram frequency patterns. Our comprehensive experiments reveal that in approximately 90\% of cases, word-level attacks lead to the generation of examples where the frequency of $n$-grams decreases, a tendency we term as the $n$-gram Frequency Descend ($n$-FD). This finding suggests a straightforward strategy to enhance model robustness: training models using examples with $n$-FD. To examine the feasibility of this strategy, we employed the $n$-gram frequency information, as an alternative to conventional loss gradients, to generate perturbed examples in adversarial training. The experiment results indicate that the frequency-based approach performs comparably with the gradient-based approach in improving model robustness. Our research offers a novel and more intuitive perspective for understanding word-level textual adversarial attacks and proposes a new direction to improve model robustness.

摘要: 词级文本敌意攻击在误导自然语言处理(NLP)模型方面显示出显著的效果。尽管它们取得了成功，但其有效性的根本原因和对抗性例子的基本特征仍然不清楚。这项工作旨在通过检查其$n$-gram频率模式来解释词级攻击。我们的综合实验表明，在大约90%的情况下，词级攻击导致$n$-gram的频率下降，这种趋势我们称之为$n$-gram频率下降($n$-fd)。这一发现提出了一种增强模型稳健性的简单策略：使用具有$n$-fd的示例来训练模型。为了检验这一策略的可行性，我们使用了$n$-gram频率信息作为传统损失梯度的替代，以生成对抗性训练中的扰动示例。实验结果表明，基于频率的方法在提高模型稳健性方面与基于梯度的方法具有相当的效果。我们的研究为理解词级文本对抗攻击提供了一个新的、更直观的视角，并为提高模型的稳健性提出了新的方向。



## **26. Beyond the Bridge: Contention-Based Covert and Side Channel Attacks on Multi-GPU Interconnect**

Beyond the Bridge：基于竞争的多GPU互连隐蔽和边通道攻击 cs.CR

Accepted to SEED 2024

**SubmitDate**: 2024-04-05    [abs](http://arxiv.org/abs/2404.03877v1) [paper-pdf](http://arxiv.org/pdf/2404.03877v1)

**Authors**: Yicheng Zhang, Ravan Nazaraliyev, Sankha Baran Dutta, Nael Abu-Ghazaleh, Andres Marquez, Kevin Barker

**Abstract**: High-speed interconnects, such as NVLink, are integral to modern multi-GPU systems, acting as a vital link between CPUs and GPUs. This study highlights the vulnerability of multi-GPU systems to covert and side channel attacks due to congestion on interconnects. An adversary can infer private information about a victim's activities by monitoring NVLink congestion without needing special permissions. Leveraging this insight, we develop a covert channel attack across two GPUs with a bandwidth of 45.5 kbps and a low error rate, and introduce a side channel attack enabling attackers to fingerprint applications through the shared NVLink interconnect.

摘要: 高速互连，如NVLink，是现代多GPU系统不可或缺的组成部分，充当CPU和GPU之间的重要链接。本研究强调了多GPU系统由于互连拥塞而容易受到隐蔽和侧信道攻击。攻击者可以通过监视NVLink拥塞而不需要特殊权限来推断受害者活动的私人信息。利用这一见解，我们开发了一种跨两个GPU的隐蔽通道攻击，带宽为45.5 kbps，错误率低，并引入了一种侧通道攻击，使攻击者能够通过共享的NVLink互连对应用进行指纹识别。



## **27. An Anomaly Behavior Analysis Framework for Securing Autonomous Vehicle Perception**

一种保障自主车辆感知的异常行为分析框架 cs.RO

20th ACS/IEEE International Conference on Computer Systems and  Applications (IEEE AICCSA 2023)

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2310.05041v3) [paper-pdf](http://arxiv.org/pdf/2310.05041v3)

**Authors**: Murad Mehrab Abrar, Salim Hariri

**Abstract**: As a rapidly growing cyber-physical platform, Autonomous Vehicles (AVs) are encountering more security challenges as their capabilities continue to expand. In recent years, adversaries are actively targeting the perception sensors of autonomous vehicles with sophisticated attacks that are not easily detected by the vehicles' control systems. This work proposes an Anomaly Behavior Analysis approach to detect a perception sensor attack against an autonomous vehicle. The framework relies on temporal features extracted from a physics-based autonomous vehicle behavior model to capture the normal behavior of vehicular perception in autonomous driving. By employing a combination of model-based techniques and machine learning algorithms, the proposed framework distinguishes between normal and abnormal vehicular perception behavior. To demonstrate the application of the framework in practice, we performed a depth camera attack experiment on an autonomous vehicle testbed and generated an extensive dataset. We validated the effectiveness of the proposed framework using this real-world data and released the dataset for public access. To our knowledge, this dataset is the first of its kind and will serve as a valuable resource for the research community in evaluating their intrusion detection techniques effectively.

摘要: 作为一个快速发展的网络物理平台，自动驾驶汽车(AVs)随着其能力的不断扩大，面临着更多的安全挑战。近年来，对手积极瞄准自动驾驶车辆的感知传感器，进行复杂的攻击，而这些攻击不容易被车辆的控制系统检测到。本文提出了一种异常行为分析方法来检测感知传感器对自动驾驶车辆的攻击。该框架依赖于从基于物理的自动驾驶车辆行为模型中提取的时间特征来捕捉自动驾驶中车辆感知的正常行为。通过结合基于模型的技术和机器学习算法，该框架区分了正常和异常的车辆感知行为。为了验证该框架在实践中的应用，我们在自主车辆试验台上进行了深度相机攻击实验，并生成了大量的数据集。我们使用这些真实世界的数据验证了提出的框架的有效性，并发布了数据集以供公众访问。据我们所知，该数据集是此类数据的第一个，将为研究界提供宝贵的资源，以有效地评估他们的入侵检测技术。



## **28. ProLoc: Robust Location Proofs in Hindsight**

ProLoc：在后见之明中强大的位置证明 cs.CR

14 pages, 5 figures

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.04297v1) [paper-pdf](http://arxiv.org/pdf/2404.04297v1)

**Authors**: Roberta De Viti, Pierfrancesco Ingo, Isaac Sheff, Peter Druschel, Deepak Garg

**Abstract**: Many online services rely on self-reported locations of user devices like smartphones. To mitigate harm from falsified self-reported locations, the literature has proposed location proof services (LPSs), which provide proof of a device's location by corroborating its self-reported location using short-range radio contacts with either trusted infrastructure or nearby devices that also report their locations. This paper presents ProLoc, a new LPS that extends prior work in two ways. First, ProLoc relaxes prior work's proofs that a device was at a given location to proofs that a device was within distance "d" of a given location. We argue that these weaker proofs, which we call "region proofs", are important because (i) region proofs can be constructed with few requirements on device reporting behavior as opposed to precise location proofs, and (ii) a quantitative bound on a device's distance from a known epicenter is useful for many applications. For example, in the context of citizen reporting near an unexpected event (earthquake, violent protest, etc.), knowing the verified distances of the reporting devices from the event's epicenter would be valuable for ranking the reports by relevance or flagging fake reports. Second, ProLoc includes a novel mechanism to prevent collusion attacks where a set of attacker-controlled devices corroborate each others' false locations. Ours is the first mechanism that does not need additional infrastructure to handle attacks with made-up devices, which an attacker can create in any number at any location without any cost. For this, we rely on a variant of TrustRank applied to the self-reported trajectories and encounters of devices. Our goal is to prevent retroactive attacks where the adversary cannot predict ahead of time which fake location it will want to report, which is the case for the reporting of unexpected events.

摘要: 许多在线服务依赖于智能手机等用户设备的自我报告位置。为了减轻伪造的自我报告位置的危害，文献提出了位置证明服务(LPS)，它通过使用与可信基础设施或也报告其位置的附近设备的短距离无线电联系来证实设备的自我报告位置，从而提供设备位置的证据。本文提出了一种新的LPS算法ProLoc，它从两个方面扩展了已有的工作。首先，ProLoc放宽了以前工作中关于设备位于给定位置的证明，以证明设备位于给定位置的“d”范围内。我们认为这些较弱的证明，我们称之为“区域证明”，是重要的，因为(I)区域证明可以被构造成对设备报告行为的较少要求，而不是精确的位置证明，以及(Ii)设备到已知震中的距离的定量界限对于许多应用是有用的。例如，在公民在意外事件(地震、暴力抗议等)附近进行报告的情况下，了解报告设备离事件震中的已核实距离对于按相关性对报告进行排序或标记虚假报告将是很有价值的。其次，ProLoc包括一种新的机制来防止合谋攻击，在这种攻击中，一组攻击者控制的设备可以证实彼此的虚假位置。我们的机制是第一个不需要额外基础设施来处理使用虚构设备进行攻击的机制，攻击者可以在任何地点以任意数量创建攻击，而不需要任何成本。为此，我们依赖于TrustRank的变体，应用于设备的自我报告轨迹和遭遇。我们的目标是防止回溯性攻击，在这种攻击中，对手无法提前预测它将报告哪个虚假位置，报告意外事件就是这种情况。



## **29. Knowledge Distillation-Based Model Extraction Attack using Private Counterfactual Explanations**

基于知识蒸馏的私有反事实搜索模型抽取攻击 cs.LG

15 pages

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03348v1) [paper-pdf](http://arxiv.org/pdf/2404.03348v1)

**Authors**: Fatima Ezzeddine, Omran Ayoub, Silvia Giordano

**Abstract**: In recent years, there has been a notable increase in the deployment of machine learning (ML) models as services (MLaaS) across diverse production software applications. In parallel, explainable AI (XAI) continues to evolve, addressing the necessity for transparency and trustworthiness in ML models. XAI techniques aim to enhance the transparency of ML models by providing insights, in terms of the model's explanations, into their decision-making process. Simultaneously, some MLaaS platforms now offer explanations alongside the ML prediction outputs. This setup has elevated concerns regarding vulnerabilities in MLaaS, particularly in relation to privacy leakage attacks such as model extraction attacks (MEA). This is due to the fact that explanations can unveil insights about the inner workings of the model which could be exploited by malicious users. In this work, we focus on investigating how model explanations, particularly Generative adversarial networks (GANs)-based counterfactual explanations (CFs), can be exploited for performing MEA within the MLaaS platform. We also delve into assessing the effectiveness of incorporating differential privacy (DP) as a mitigation strategy. To this end, we first propose a novel MEA methodology based on Knowledge Distillation (KD) to enhance the efficiency of extracting a substitute model of a target model exploiting CFs. Then, we advise an approach for training CF generators incorporating DP to generate private CFs. We conduct thorough experimental evaluations on real-world datasets and demonstrate that our proposed KD-based MEA can yield a high-fidelity substitute model with reduced queries with respect to baseline approaches. Furthermore, our findings reveal that the inclusion of a privacy layer impacts the performance of the explainer, the quality of CFs, and results in a reduction in the MEA performance.

摘要: 近年来，机器学习(ML)模型即服务(MLaaS)在各种生产软件应用程序中的部署显著增加。同时，可解释人工智能(XAI)继续发展，解决了ML模型中透明度和可信性的必要性。XAI技术旨在通过提供对ML模型的解释方面的见解来提高ML模型的透明度。与此同时，一些MLaaS平台现在除了ML预测输出外，还提供解释。这一设置加剧了人们对MLaaS漏洞的担忧，特别是与隐私泄露攻击有关的漏洞，如模型提取攻击(MEA)。这是因为解释可以揭示该模型的内部工作原理，这可能会被恶意用户利用。在这项工作中，我们重点研究如何利用模型解释，特别是基于生成性对抗网络(GANS)的反事实解释(CFS)来在MLaaS平台上执行MEA。我们还深入评估了将差异隐私(DP)作为缓解策略的有效性。为此，我们首先提出了一种新的基于知识蒸馏(KD)的MEA方法，以提高利用CFS提取目标模型替代模型的效率。然后，我们提出了一种训练包含DP的CF生成器以生成私有CF的方法。我们在真实世界的数据集上进行了深入的实验评估，并证明了我们提出的基于KD的MEA可以产生高保真的替代模型，相对于基线方法减少了查询。此外，我们的研究结果还表明，隐私层的加入影响了解释者的表现，也影响了解释的质量，并导致了MEA表现的下降。



## **30. Meta Invariance Defense Towards Generalizable Robustness to Unknown Adversarial Attacks**

面向未知对抗攻击的广义鲁棒性的Meta不变性防御 cs.CV

Accepted by IEEE TPAMI in 2024

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03340v1) [paper-pdf](http://arxiv.org/pdf/2404.03340v1)

**Authors**: Lei Zhang, Yuhang Zhou, Yi Yang, Xinbo Gao

**Abstract**: Despite providing high-performance solutions for computer vision tasks, the deep neural network (DNN) model has been proved to be extremely vulnerable to adversarial attacks. Current defense mainly focuses on the known attacks, but the adversarial robustness to the unknown attacks is seriously overlooked. Besides, commonly used adaptive learning and fine-tuning technique is unsuitable for adversarial defense since it is essentially a zero-shot problem when deployed. Thus, to tackle this challenge, we propose an attack-agnostic defense method named Meta Invariance Defense (MID). Specifically, various combinations of adversarial attacks are randomly sampled from a manually constructed Attacker Pool to constitute different defense tasks against unknown attacks, in which a student encoder is supervised by multi-consistency distillation to learn the attack-invariant features via a meta principle. The proposed MID has two merits: 1) Full distillation from pixel-, feature- and prediction-level between benign and adversarial samples facilitates the discovery of attack-invariance. 2) The model simultaneously achieves robustness to the imperceptible adversarial perturbations in high-level image classification and attack-suppression in low-level robust image regeneration. Theoretical and empirical studies on numerous benchmarks such as ImageNet verify the generalizable robustness and superiority of MID under various attacks.

摘要: 尽管深度神经网络(DNN)模型为计算机视觉任务提供了高性能的解决方案，但已被证明极易受到对手攻击。目前的防御主要针对已知攻击，而对未知攻击的对抗健壮性被严重忽视。此外，常用的自适应学习和微调技术不适合对抗性防御，因为它在部署时本质上是一个零命中问题。因此，为了应对这一挑战，我们提出了一种与攻击无关的防御方法，称为元不变性防御(MID)。具体地，从人工构建的攻击者池中随机抽取各种对抗性攻击组合，组成针对未知攻击的不同防御任务，其中学生编码者通过多一致性蒸馏来监督，通过元原则学习攻击不变特征。提出的MID算法有两个优点：1)良性样本和敌方样本之间像素级、特征级和预测级的充分提取有利于发现攻击不变性。2)该模型同时实现了在高层图像分类中对不可察觉的对抗性扰动的鲁棒性和在低级稳健图像再生中的攻击抑制。在ImageNet等众多基准测试上的理论和实证研究验证了MID在各种攻击下的泛化健壮性和优越性。



## **31. Learn What You Want to Unlearn: Unlearning Inversion Attacks against Machine Unlearning**

学习你想要忘记的东西：忘记学习反转攻击针对机器忘记学习 cs.CR

To Appear in the 45th IEEE Symposium on Security and Privacy, May  20-23, 2024

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03233v1) [paper-pdf](http://arxiv.org/pdf/2404.03233v1)

**Authors**: Hongsheng Hu, Shuo Wang, Tian Dong, Minhui Xue

**Abstract**: Machine unlearning has become a promising solution for fulfilling the "right to be forgotten", under which individuals can request the deletion of their data from machine learning models. However, existing studies of machine unlearning mainly focus on the efficacy and efficiency of unlearning methods, while neglecting the investigation of the privacy vulnerability during the unlearning process. With two versions of a model available to an adversary, that is, the original model and the unlearned model, machine unlearning opens up a new attack surface. In this paper, we conduct the first investigation to understand the extent to which machine unlearning can leak the confidential content of the unlearned data. Specifically, under the Machine Learning as a Service setting, we propose unlearning inversion attacks that can reveal the feature and label information of an unlearned sample by only accessing the original and unlearned model. The effectiveness of the proposed unlearning inversion attacks is evaluated through extensive experiments on benchmark datasets across various model architectures and on both exact and approximate representative unlearning approaches. The experimental results indicate that the proposed attack can reveal the sensitive information of the unlearned data. As such, we identify three possible defenses that help to mitigate the proposed attacks, while at the cost of reducing the utility of the unlearned model. The study in this paper uncovers an underexplored gap between machine unlearning and the privacy of unlearned data, highlighting the need for the careful design of mechanisms for implementing unlearning without leaking the information of the unlearned data.

摘要: 机器遗忘已经成为一种很有前途的解决方案，可以实现“被遗忘的权利”，根据这种权利，个人可以请求从机器学习模型中删除他们的数据。然而，现有的机器遗忘研究主要集中在遗忘方法的有效性和效率上，而忽略了对遗忘过程中隐私漏洞的研究。由于一个模型有两个版本可供对手使用，即原始模型和未学习模型，机器遗忘打开了一个新的攻击面。在本文中，我们进行了第一次调查，以了解机器遗忘可以在多大程度上泄露未学习数据的机密内容。具体地说，在机器学习即服务的背景下，我们提出了遗忘反转攻击，只需访问原始的和未学习的模型，就可以揭示未学习样本的特征和标签信息。通过在不同模型体系结构的基准数据集上以及在精确和近似代表遗忘方法上的大量实验，评估了所提出的遗忘反转攻击的有效性。实验结果表明，该攻击能够泄露未学习数据的敏感信息。因此，我们确定了三种可能的防御措施，它们有助于减轻拟议的攻击，同时代价是减少未学习模型的效用。本文的研究揭示了机器遗忘和未学习数据隐私之间的未被探索的差距，强调了需要仔细设计机制来实现遗忘而不泄露未学习数据的信息。



## **32. FACTUAL: A Novel Framework for Contrastive Learning Based Robust SAR Image Classification**

FACTUAL：一种基于对比学习的稳健SAR图像分类新框架 cs.CV

2024 IEEE Radar Conference

**SubmitDate**: 2024-04-04    [abs](http://arxiv.org/abs/2404.03225v1) [paper-pdf](http://arxiv.org/pdf/2404.03225v1)

**Authors**: Xu Wang, Tian Ye, Rajgopal Kannan, Viktor Prasanna

**Abstract**: Deep Learning (DL) Models for Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR), while delivering improved performance, have been shown to be quite vulnerable to adversarial attacks. Existing works improve robustness by training models on adversarial samples. However, by focusing mostly on attacks that manipulate images randomly, they neglect the real-world feasibility of such attacks. In this paper, we propose FACTUAL, a novel Contrastive Learning framework for Adversarial Training and robust SAR classification. FACTUAL consists of two components: (1) Differing from existing works, a novel perturbation scheme that incorporates realistic physical adversarial attacks (such as OTSA) to build a supervised adversarial pre-training network. This network utilizes class labels for clustering clean and perturbed images together into a more informative feature space. (2) A linear classifier cascaded after the encoder to use the computed representations to predict the target labels. By pre-training and fine-tuning our model on both clean and adversarial samples, we show that our model achieves high prediction accuracy on both cases. Our model achieves 99.7% accuracy on clean samples, and 89.6% on perturbed samples, both outperforming previous state-of-the-art methods.

摘要: 用于合成孔径雷达(SAR)自动目标识别(ATR)的深度学习(DL)模型虽然提供了更好的性能，但已被证明非常容易受到对手攻击。已有的工作通过训练对抗性样本上的模型来提高鲁棒性。然而，由于主要关注随机操纵图像的攻击，他们忽视了此类攻击在现实世界中的可行性。本文提出了一种用于对抗性训练和稳健SAR分类的新型对比学习框架FACTAL。FACTUAL由两部分组成：(1)与已有工作不同，提出了一种新的扰动方案，该方案结合了真实的物理对抗攻击(如OTSA)来构建一个有监督的对抗预训练网络。该网络利用类别标签将干净的和受干扰的图像聚在一起，形成一个更具信息量的特征空间。(2)在编码器之后级联一个线性分类器，使用计算的表示来预测目标标签。通过对干净样本和对抗性样本的预训练和微调，我们证明了我们的模型在这两种情况下都达到了很高的预测精度。我们的模型在清洁样本上达到了99.7%的准确率，在扰动样本上达到了89.6%的准确率，都超过了以前最先进的方法。



## **33. Robust Federated Learning for Wireless Networks: A Demonstration with Channel Estimation**

无线网络的鲁棒联邦学习：信道估计演示 cs.LG

Submitted to IEEE GLOBECOM 2024

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.03088v1) [paper-pdf](http://arxiv.org/pdf/2404.03088v1)

**Authors**: Zexin Fang, Bin Han, Hans D. Schotten

**Abstract**: Federated learning (FL) offers a privacy-preserving collaborative approach for training models in wireless networks, with channel estimation emerging as a promising application. Despite extensive studies on FL-empowered channel estimation, the security concerns associated with FL require meticulous attention. In a scenario where small base stations (SBSs) serve as local models trained on cached data, and a macro base station (MBS) functions as the global model setting, an attacker can exploit the vulnerability of FL, launching attacks with various adversarial attacks or deployment tactics. In this paper, we analyze such vulnerabilities, corresponding solutions were brought forth, and validated through simulation.

摘要: 联合学习（FL）为无线网络中的训练模型提供了一种保护隐私的协作方法，信道估计成为一个很有前途的应用。尽管对FL赋能信道估计进行了广泛的研究，但与FL相关的安全问题仍需要精心关注。在小型基站（SBS）用作在缓存数据上训练的本地模型，宏基站（MBS）用作全局模型设置的场景中，攻击者可以利用FL的漏洞，利用各种对抗性攻击或部署策略发起攻击。本文对这些漏洞进行了分析，提出了相应的解决方案，并通过仿真进行了验证。



## **34. Adversarial Evasion Attacks Practicality in Networks: Testing the Impact of Dynamic Learning**

对抗性规避攻击在网络中的实际性：测试动态学习的影响 cs.CR

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2306.05494v2) [paper-pdf](http://arxiv.org/pdf/2306.05494v2)

**Authors**: Mohamed el Shehaby, Ashraf Matrawy

**Abstract**: Machine Learning (ML) has become ubiquitous, and its deployment in Network Intrusion Detection Systems (NIDS) is inevitable due to its automated nature and high accuracy compared to traditional models in processing and classifying large volumes of data. However, ML has been found to have several flaws, most importantly, adversarial attacks, which aim to trick ML models into producing faulty predictions. While most adversarial attack research focuses on computer vision datasets, recent studies have explored the suitability of these attacks against ML-based network security entities, especially NIDS, due to the wide difference between different domains regarding the generation of adversarial attacks.   To further explore the practicality of adversarial attacks against ML-based NIDS in-depth, this paper presents three distinct contributions: identifying numerous practicality issues for evasion adversarial attacks on ML-NIDS using an attack tree threat model, introducing a taxonomy of practicality issues associated with adversarial attacks against ML-based NIDS, and investigating how the dynamicity of some real-world ML models affects adversarial attacks against NIDS. Our experiments indicate that continuous re-training, even without adversarial training, can reduce the effectiveness of adversarial attacks. While adversarial attacks can compromise ML-based NIDSs, our aim is to highlight the significant gap between research and real-world practicality in this domain, warranting attention.

摘要: 机器学习(ML)已经变得无处不在，与传统模型相比，它在处理和分类海量数据方面具有自动化的性质和较高的准确率，因此在网络入侵检测系统(NIDS)中的应用是不可避免的。然而，ML被发现有几个缺陷，最重要的是对抗性攻击，其目的是欺骗ML模型产生错误的预测。虽然大多数对抗性攻击研究集中在计算机视觉数据集，但最近的研究探索了这些攻击对基于ML的网络安全实体，特别是网络入侵检测系统的适用性，这是因为不同领域之间关于对抗性攻击生成的巨大差异。为了进一步深入探讨针对基于ML的网络入侵检测系统的对抗性攻击的实用性，本文提出了三个不同的贡献：利用攻击树威胁模型识别针对ML-NID的逃避对抗性攻击的大量实用性问题，引入与针对基于ML的网络入侵检测系统的对抗性攻击相关的实用性问题的分类，以及研究一些真实世界的ML模型的动态性如何影响对网络入侵检测系统的对抗性攻击。我们的实验表明，即使在没有对抗性训练的情况下，持续的再训练也会降低对抗性攻击的有效性。虽然敌意攻击可能会危及基于ML的NIDS，但我们的目标是突出这一领域的研究和现实世界实用之间的显著差距，值得关注。



## **35. JailBreakV-28K: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

JailBreakV—28K：评估多模大语言模型抗越狱攻击鲁棒性的基准 cs.CR

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.03027v1) [paper-pdf](http://arxiv.org/pdf/2404.03027v1)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.

摘要: 随着多模式大型语言模型(MLLMS)的快速发展，保护这些模型不受恶意输入的影响，同时使它们与人类的价值观保持一致，已经成为一项关键的挑战。在本文中，我们研究了一个重要而未被探索的问题，即成功越狱大语言模型(LLMS)的技术是否可以在越狱MLLM中同样有效。为了探讨这一问题，我们引入了JailBreakV-28K，这是一个开创性的基准测试，旨在评估LLM越狱技术到MLLM的可转移性，从而评估MLLMS对各种越狱攻击的健壮性。利用本文提出的包含2,000个恶意查询的数据集，我们使用针对LLMS的高级越狱攻击生成了20,000个基于文本的越狱提示，以及来自最近MLLMS越狱攻击的8,000个基于图像的越狱输入，我们的综合数据集包括来自各种对抗场景的28,000个测试用例。我们对10个开源MLLMS的评估显示，对于从LLMS转移的攻击，攻击成功率(ASR)非常高，这突显了MLLMS中源于其文本处理能力的一个严重漏洞。我们的发现强调了未来研究的迫切需要，以解决MLLMS中从文本和视觉输入的对齐漏洞。



## **36. Adversarial Attacks and Dimensionality in Text Classifiers**

文本分类器中的对抗性攻击和模糊性 cs.LG

This paper is accepted for publication at EURASIP Journal on  Information Security in 2024

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02660v1) [paper-pdf](http://arxiv.org/pdf/2404.02660v1)

**Authors**: Nandish Chattopadhyay, Atreya Goswami, Anupam Chattopadhyay

**Abstract**: Adversarial attacks on machine learning algorithms have been a key deterrent to the adoption of AI in many real-world use cases. They significantly undermine the ability of high-performance neural networks by forcing misclassifications. These attacks introduce minute and structured perturbations or alterations in the test samples, imperceptible to human annotators in general, but trained neural networks and other models are sensitive to it. Historically, adversarial attacks have been first identified and studied in the domain of image processing. In this paper, we study adversarial examples in the field of natural language processing, specifically text classification tasks. We investigate the reasons for adversarial vulnerability, particularly in relation to the inherent dimensionality of the model. Our key finding is that there is a very strong correlation between the embedding dimensionality of the adversarial samples and their effectiveness on models tuned with input samples with same embedding dimension. We utilize this sensitivity to design an adversarial defense mechanism. We use ensemble models of varying inherent dimensionality to thwart the attacks. This is tested on multiple datasets for its efficacy in providing robustness. We also study the problem of measuring adversarial perturbation using different distance metrics. For all of the aforementioned studies, we have run tests on multiple models with varying dimensionality and used a word-vector level adversarial attack to substantiate the findings.

摘要: 对机器学习算法的对抗性攻击一直是许多现实世界用例中采用人工智能的关键威慑因素。它们通过强制错误分类，大大削弱了高性能神经网络的能力。这些攻击在测试样本中引入微小的和结构化的扰动或改变，通常人类注释员察觉不到，但经过训练的神经网络和其他模型对此很敏感。历史上，对抗性攻击最早是在图像处理领域被识别和研究的。在本文中，我们研究了自然语言处理领域中的对抗性实例，特别是文本分类任务。我们研究了对抗脆弱性的原因，特别是与模型的固有维度有关的原因。我们的关键发现是，对抗性样本的嵌入维度与其在具有相同嵌入维度的输入样本调整的模型上的有效性之间存在很强的相关性。我们利用这种敏感性来设计一种对抗性防御机制。我们使用不同固有维度的集合模型来阻止攻击。这在多个数据集上进行了测试，以确定其在提供稳健性方面的有效性。我们还研究了使用不同的距离度量来度量对手扰动的问题。对于上述所有研究，我们在不同维度的多个模型上进行了测试，并使用了单词向量级别的对抗性攻击来证实这些发现。



## **37. Adversary-Augmented Simulation to evaluate fairness on HyperLedger Fabric**

在Hyperledger Fabric上评估公平性的对抗增强仿真 cs.CR

10 pages, 8 figures

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2403.14342v2) [paper-pdf](http://arxiv.org/pdf/2403.14342v2)

**Authors**: Erwan Mahe, Rouwaida Abdallah, Sara Tucci-Piergiovanni, Pierre-Yves Piriou

**Abstract**: This paper presents a novel adversary model specifically tailored to distributed systems, aiming to assess the security of blockchain networks. Building upon concepts such as adversarial assumptions, goals, and capabilities, our proposed adversary model classifies and constrains the use of adversarial actions based on classical distributed system models, defined by both failure and communication models. The objective is to study the effects of these allowed actions on the properties of distributed protocols under various system models. A significant aspect of our research involves integrating this adversary model into the Multi-Agent eXperimenter (MAX) framework. This integration enables fine-grained simulations of adversarial attacks on blockchain networks. In this paper, we particularly study four distinct fairness properties on Hyperledger Fabric with the Byzantine Fault Tolerant Tendermint consensus algorithm being selected for its ordering service. We define novel attacks that combine adversarial actions on both protocols, with the aim of violating a specific client-fairness property. Simulations confirm our ability to violate this property and allow us to evaluate the impact of these attacks on several order-fairness properties that relate orders of transaction reception and delivery.

摘要: 提出了一种新的针对分布式系统的敌手模型，旨在评估区块链网络的安全性。基于对抗性假设、目标和能力等概念，我们提出的对抗性模型对基于经典分布式系统模型的对抗性动作的使用进行分类和限制，该模型由故障模型和通信模型定义。目的是研究在不同的系统模型下，这些允许的操作对分布式协议性能的影响。我们研究的一个重要方面涉及将这种对手模型集成到多代理实验者(MAX)框架中。这种整合可以对区块链网络上的对抗性攻击进行细粒度的模拟。本文以拜占庭容错Tendermint一致性算法为排序服务，详细研究了Hyperledger织物的四个不同的公平性。我们定义了新的攻击，它结合了两种协议上的对抗性操作，目的是违反特定的客户端公平属性。模拟证实了我们违反这一属性的能力，并允许我们评估这些攻击对几个与交易接收和交付顺序相关的顺序公平属性的影响。



## **38. Unsegment Anything by Simulating Deformation**

模拟变形解分割任何东西 cs.CV

CVPR 2024

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02585v1) [paper-pdf](http://arxiv.org/pdf/2404.02585v1)

**Authors**: Jiahao Lu, Xingyi Yang, Xinchao Wang

**Abstract**: Foundation segmentation models, while powerful, pose a significant risk: they enable users to effortlessly extract any objects from any digital content with a single click, potentially leading to copyright infringement or malicious misuse. To mitigate this risk, we introduce a new task "Anything Unsegmentable" to grant any image "the right to be unsegmented". The ambitious pursuit of the task is to achieve highly transferable adversarial attacks against all prompt-based segmentation models, regardless of model parameterizations and prompts. We highlight the non-transferable and heterogeneous nature of prompt-specific adversarial noises. Our approach focuses on disrupting image encoder features to achieve prompt-agnostic attacks. Intriguingly, targeted feature attacks exhibit better transferability compared to untargeted ones, suggesting the optimal update direction aligns with the image manifold. Based on the observations, we design a novel attack named Unsegment Anything by Simulating Deformation (UAD). Our attack optimizes a differentiable deformation function to create a target deformed image, which alters structural information while preserving achievable feature distance by adversarial example. Extensive experiments verify the effectiveness of our approach, compromising a variety of promptable segmentation models with different architectures and prompt interfaces. We release the code at https://github.com/jiahaolu97/anything-unsegmentable.

摘要: 基础分割模型虽然功能强大，但也带来了重大风险：它们使用户能够轻松地通过一次点击从任何数字内容中提取任何对象，这可能会导致侵犯版权或恶意滥用。为了减轻这种风险，我们引入了一个新的任务“任何不可分割的”，以授予任何图像“被取消分割的权利”。这项任务的雄心勃勃的追求是实现对所有基于提示的分割模型的高度可转移的对抗性攻击，而不考虑模型的参数化和提示。我们强调了即时特定对抗性噪音的不可转移性和异质性。我们的方法专注于破坏图像编码器功能，以实现与提示无关的攻击。有趣的是，与非目标攻击相比，目标特征攻击表现出更好的可转移性，这表明最佳更新方向与图像流形一致。在此基础上，我们设计了一种新的攻击方法，称为通过模拟变形来不分割任何东西(UAD)。我们的攻击优化了一个可微变形函数来生成目标变形图像，该变形图像改变了目标的结构信息，同时通过对抗性例子保持了可达到的特征距离。大量的实验验证了我们的方法的有效性，折衷了各种具有不同体系结构和提示界面的可提示分割模型。我们在https://github.com/jiahaolu97/anything-unsegmentable.上发布代码



## **39. A Unified Membership Inference Method for Visual Self-supervised Encoder via Part-aware Capability**

基于部件感知能力的视觉自监督编码器统一隶属度推理方法 cs.CV

Membership Inference, Self-supervised learning

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02462v1) [paper-pdf](http://arxiv.org/pdf/2404.02462v1)

**Authors**: Jie Zhu, Jirong Zha, Ding Li, Leye Wang

**Abstract**: Self-supervised learning shows promise in harnessing extensive unlabeled data, but it also confronts significant privacy concerns, especially in vision. In this paper, we aim to perform membership inference on visual self-supervised models in a more realistic setting: self-supervised training method and details are unknown for an adversary when attacking as he usually faces a black-box system in practice. In this setting, considering that self-supervised model could be trained by completely different self-supervised paradigms, e.g., masked image modeling and contrastive learning, with complex training details, we propose a unified membership inference method called PartCrop. It is motivated by the shared part-aware capability among models and stronger part response on the training data. Specifically, PartCrop crops parts of objects in an image to query responses with the image in representation space. We conduct extensive attacks on self-supervised models with different training protocols and structures using three widely used image datasets. The results verify the effectiveness and generalization of PartCrop. Moreover, to defend against PartCrop, we evaluate two common approaches, i.e., early stop and differential privacy, and propose a tailored method called shrinking crop scale range. The defense experiments indicate that all of them are effective. Our code is available at https://github.com/JiePKU/PartCrop

摘要: 自我监督学习在利用大量未标记数据方面表现出了希望，但它也面临着重大的隐私问题，特别是在视觉方面。在本文中，我们的目标是在一种更现实的环境下对视觉自我监督模型进行隶属度推理：当对手攻击时，自我监督训练方法和细节是未知的，因为他在实践中通常面临一个黑箱系统。在这种情况下，考虑到自监督模型可以用完全不同的自监督范型来训练，例如蒙版图像建模和对比学习，训练细节复杂，我们提出了一种统一的隶属度推理方法PartCrop。它的动机是模型之间共享的部件感知能力和对训练数据的更强的部件响应。具体地说，PartCrop裁剪图像中对象的一部分，以在表示空间中使用图像查询响应。我们使用三个广泛使用的图像数据集对具有不同训练协议和结构的自监督模型进行了广泛的攻击。实验结果验证了PartCrop的有效性和泛化能力。此外，为了防御PartCrop，我们评估了两种常见的方法，即提前停止和区分隐私，并提出了一种称为缩小作物尺度范围的定制方法。防御实验表明，这些方法都是有效的。我们的代码可以在https://github.com/JiePKU/PartCrop上找到



## **40. Designing a Photonic Physically Unclonable Function Having Resilience to Machine Learning Attacks**

设计一个对机器学习攻击具有弹性的光子物理不可克隆函数 cs.CR

14 pages, 8 figures

**SubmitDate**: 2024-04-03    [abs](http://arxiv.org/abs/2404.02440v1) [paper-pdf](http://arxiv.org/pdf/2404.02440v1)

**Authors**: Elena R. Henderson, Jessie M. Henderson, Hiva Shahoei, William V. Oxford, Eric C. Larson, Duncan L. MacFarlane, Mitchell A. Thornton

**Abstract**: Physically unclonable functions (PUFs) are designed to act as device 'fingerprints.' Given an input challenge, the PUF circuit should produce an unpredictable response for use in situations such as root-of-trust applications and other hardware-level cybersecurity applications. PUFs are typically subcircuits present within integrated circuits (ICs), and while conventional IC PUFs are well-understood, several implementations have proven vulnerable to malicious exploits, including those perpetrated by machine learning (ML)-based attacks. Such attacks can be difficult to prevent because they are often designed to work even when relatively few challenge-response pairs are known in advance. Hence the need for both more resilient PUF designs and analysis of ML-attack susceptibility. Previous work has developed a PUF for photonic integrated circuits (PICs). A PIC PUF not only produces unpredictable responses given manufacturing-introduced tolerances, but is also less prone to electromagnetic radiation eavesdropping attacks than a purely electronic IC PUF. In this work, we analyze the resilience of the proposed photonic PUF when subjected to ML-based attacks. Specifically, we describe a computational PUF model for producing the large datasets required for training ML attacks; we analyze the quality of the model; and we discuss the modeled PUF's susceptibility to ML-based attacks. We find that the modeled PUF generates distributions that resemble uniform white noise, explaining the exhibited resilience to neural-network-based attacks designed to exploit latent relationships between challenges and responses. Preliminary analysis suggests that the PUF exhibits similar resilience to generative adversarial networks, and continued development will show whether more-sophisticated ML approaches better compromise the PUF and -- if so -- how design modifications might improve resilience.

摘要: 物理上不可克隆的功能(PUF)被设计成充当设备的“指纹”。在给定输入挑战的情况下，PUF电路应产生不可预测的响应，以便在信任根应用程序和其他硬件级别的网络安全应用程序等情况下使用。PUF通常是集成电路(IC)中存在的子电路，虽然传统的IC PUF是众所周知的，但事实证明，一些实现容易受到恶意利用，包括那些由基于机器学习(ML)的攻击所造成的利用。此类攻击可能很难预防，因为它们通常被设计为即使在事先知道的挑战-响应对相对较少的情况下也能发挥作用。因此，需要更具弹性的PUF设计和ML攻击敏感性分析。以前的工作已经开发了一种用于光子集成电路(PIC)的PUF。PIC PUF不仅在制造引入容差的情况下产生不可预测的响应，而且比纯电子IC PUF更不容易受到电磁辐射窃听攻击。在这项工作中，我们分析了所提出的光子PUF在遭受基于ML的攻击时的弹性。具体地说，我们描述了一个计算PUF模型，用于产生训练ML攻击所需的大数据集；我们分析了该模型的质量；我们讨论了所建模型的PUF对基于ML的攻击的敏感性。我们发现，建模的PUF生成类似于均匀白噪声的分布，解释了对旨在利用挑战和响应之间的潜在关系的基于神经网络的攻击表现出的弹性。初步分析表明，PUF对生成性对抗网络表现出类似的弹性，继续发展将表明更复杂的ML方法是否会更好地折衷PUF，以及--如果是--设计修改如何提高弹性。



## **41. One Noise to Rule Them All: Multi-View Adversarial Attacks with Universal Perturbation**

一个噪声统治他们所有：具有普遍扰动的多视图对抗攻击 cs.CV

6 pages, 4 figures, presented at ICAIA, Springer to publish under  Algorithms for Intelligent Systems

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02287v1) [paper-pdf](http://arxiv.org/pdf/2404.02287v1)

**Authors**: Mehmet Ergezer, Phat Duong, Christian Green, Tommy Nguyen, Abdurrahman Zeybey

**Abstract**: This paper presents a novel universal perturbation method for generating robust multi-view adversarial examples in 3D object recognition. Unlike conventional attacks limited to single views, our approach operates on multiple 2D images, offering a practical and scalable solution for enhancing model scalability and robustness. This generalizable method bridges the gap between 2D perturbations and 3D-like attack capabilities, making it suitable for real-world applications.   Existing adversarial attacks may become ineffective when images undergo transformations like changes in lighting, camera position, or natural deformations. We address this challenge by crafting a single universal noise perturbation applicable to various object views. Experiments on diverse rendered 3D objects demonstrate the effectiveness of our approach. The universal perturbation successfully identified a single adversarial noise for each given set of 3D object renders from multiple poses and viewpoints. Compared to single-view attacks, our universal attacks lower classification confidence across multiple viewing angles, especially at low noise levels. A sample implementation is made available at https://github.com/memoatwit/UniversalPerturbation.

摘要: 提出了一种新的通用摄动方法，用于在三维物体识别中生成健壮的多视点对抗性样本。与传统的仅限于单视图的攻击不同，我们的方法在多个2D图像上操作，为增强模型的可扩展性和健壮性提供了实用且可扩展的解决方案。这种可推广的方法弥合了2D扰动和类似3D的攻击能力之间的差距，使其适用于现实世界的应用。现有的对抗性攻击可能会在图像经历光照、相机位置或自然变形等变化时变得无效。我们通过制作适用于各种对象视图的单一通用噪声扰动来解决这一挑战。在不同渲染的3D对象上的实验证明了该方法的有效性。通用扰动成功地为来自多个姿势和视点的每一组给定的3D对象渲染识别了单个对抗性噪声。与单视图攻击相比，我们的通用攻击降低了多个视角的分类置信度，特别是在低噪声水平下。在https://github.com/memoatwit/UniversalPerturbation.上提供了一个示例实现



## **42. Towards Robust 3D Pose Transfer with Adversarial Learning**

基于对抗学习的鲁棒3D姿势传递 cs.CV

CVPR 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02242v1) [paper-pdf](http://arxiv.org/pdf/2404.02242v1)

**Authors**: Haoyu Chen, Hao Tang, Ehsan Adeli, Guoying Zhao

**Abstract**: 3D pose transfer that aims to transfer the desired pose to a target mesh is one of the most challenging 3D generation tasks. Previous attempts rely on well-defined parametric human models or skeletal joints as driving pose sources. However, to obtain those clean pose sources, cumbersome but necessary pre-processing pipelines are inevitable, hindering implementations of the real-time applications. This work is driven by the intuition that the robustness of the model can be enhanced by introducing adversarial samples into the training, leading to a more invulnerable model to the noisy inputs, which even can be further extended to directly handling the real-world data like raw point clouds/scans without intermediate processing. Furthermore, we propose a novel 3D pose Masked Autoencoder (3D-PoseMAE), a customized MAE that effectively learns 3D extrinsic presentations (i.e., pose). 3D-PoseMAE facilitates learning from the aspect of extrinsic attributes by simultaneously generating adversarial samples that perturb the model and learning the arbitrary raw noisy poses via a multi-scale masking strategy. Both qualitative and quantitative studies show that the transferred meshes given by our network result in much better quality. Besides, we demonstrate the strong generalizability of our method on various poses, different domains, and even raw scans. Experimental results also show meaningful insights that the intermediate adversarial samples generated in the training can successfully attack the existing pose transfer models.

摘要: 三维姿态变换是三维生成中最具挑战性的任务之一，其目的是将期望的姿态传递到目标网格上。以前的尝试依赖于定义明确的参数人体模型或骨骼关节作为驱动姿势源。然而，为了获得这些干净的姿态源，繁琐但必要的预处理流水线是不可避免的，阻碍了实时应用的实现。这项工作是由这样一种直觉驱动的，即通过在训练中引入对抗性样本可以增强模型的稳健性，从而产生对噪声输入更不敏感的模型，这甚至可以进一步扩展到直接处理真实世界的数据，如原始点云/扫描而不需要中间处理。此外，我们提出了一种新的3D姿势掩蔽自动编码器(3D-PoseMAE)，这是一种定制的MAE，可以有效地学习3D外部表示(即姿势)。3D-PoseMAE通过同时生成扰动模型的敌意样本和通过多尺度掩蔽策略学习任意原始噪声姿势，从而便于从外部属性方面进行学习。定性和定量的研究都表明，我们的网络给出的传输网格的质量要好得多。此外，我们还证明了我们的方法在各种姿态、不同领域甚至原始扫描上都具有很强的通用性。实验结果还表明，训练中生成的中间对抗性样本能够成功攻击现有的姿势转移模型。



## **43. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**

使用简单的自适应攻击破解领先的安全一致LLM cs.CR

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02151v1) [paper-pdf](http://arxiv.org/pdf/2404.02151v1)

**Authors**: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion

**Abstract**: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize the target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve nearly 100\% attack success rate -- according to GPT-4 as a judge -- on GPT-3.5/4, Llama-2-Chat-7B/13B/70B, Gemma-7B, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with 100\% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). We provide the code, prompts, and logs of the attacks at https://github.com/tml-epfl/llm-adaptive-attacks.

摘要: 我们表明，即使是最新的安全对齐的LLM也不能抵抗简单的自适应越狱攻击。首先，我们演示了如何成功地利用对logpros的访问来越狱：我们最初设计了一个对抗性提示模板(有时适用于目标LLM)，然后在后缀上应用随机搜索来最大化目标logprob(例如，令牌“Sure”)，可能需要多次重新启动。通过这种方式，我们获得了近100%的攻击成功率-根据GPT-4作为判断-在GPT-3.5/4、Llama-2-Chat-7B/13B/70B、Gema-7B和R2D2上，它们都经过了对抗GCG攻击的恶意训练。我们还展示了如何通过传输或预填充攻击以100%的成功率越狱所有不暴露日志问题的Claude模型。此外，我们还展示了如何在受限的令牌集合上使用随机搜索来查找有毒模型中的特洛伊木马字符串--这项任务与越狱有许多相似之处--正是这种算法为我们带来了SATML‘24特洛伊木马检测大赛的第一名。这些攻击背后的共同主题是自适应至关重要：不同的模型容易受到不同提示模板的攻击(例如，R2D2对上下文中的学习提示非常敏感)，一些模型基于其API具有独特的漏洞(例如，预填充Claude)，并且在某些设置中，基于先验知识限制令牌搜索空间至关重要(例如，对于木马检测)。我们在https://github.com/tml-epfl/llm-adaptive-attacks.上提供攻击的代码、提示和日志



## **44. READ: Improving Relation Extraction from an ADversarial Perspective**

阅读：从对抗角度改进关系提取 cs.CL

Accepted by findings of NAACL 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02931v1) [paper-pdf](http://arxiv.org/pdf/2404.02931v1)

**Authors**: Dawei Li, William Hogan, Jingbo Shang

**Abstract**: Recent works in relation extraction (RE) have achieved promising benchmark accuracy; however, our adversarial attack experiments show that these works excessively rely on entities, making their generalization capability questionable. To address this issue, we propose an adversarial training method specifically designed for RE. Our approach introduces both sequence- and token-level perturbations to the sample and uses a separate perturbation vocabulary to improve the search for entity and context perturbations. Furthermore, we introduce a probabilistic strategy for leaving clean tokens in the context during adversarial training. This strategy enables a larger attack budget for entities and coaxes the model to leverage relational patterns embedded in the context. Extensive experiments show that compared to various adversarial training methods, our method significantly improves both the accuracy and robustness of the model. Additionally, experiments on different data availability settings highlight the effectiveness of our method in low-resource scenarios. We also perform in-depth analyses of our proposed method and provide further hints. We will release our code at https://github.com/David-Li0406/READ.

摘要: 最近在关系抽取(RE)方面的工作已经取得了令人满意的基准精度；然而，我们的对抗攻击实验表明，这些工作过度依赖实体，使得它们的泛化能力受到质疑。为了解决这个问题，我们提出了一种专门为RE设计的对抗性训练方法。我们的方法将序列级和令牌级的扰动引入样本，并使用单独的扰动词汇表来改进实体和上下文扰动的搜索。此外，我们引入了一种概率策略，在对抗性训练期间将干净的令牌留在上下文中。该策略为实体提供了更大的攻击预算，并诱使模型利用嵌入在上下文中的关系模式。大量实验表明，与各种对抗性训练方法相比，该方法显著提高了模型的准确性和稳健性。此外，在不同数据可用性设置上的实验突出了我们的方法在低资源场景下的有效性。我们还对我们提出的方法进行了深入的分析，并提供了进一步的提示。我们将在https://github.com/David-Li0406/READ.上发布我们的代码



## **45. Red-Teaming Segment Anything Model**

Red—Team Segment Anything Model cs.CV

CVPR 2024 - The 4th Workshop of Adversarial Machine Learning on  Computer Vision: Robustness of Foundation Models

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.02067v1) [paper-pdf](http://arxiv.org/pdf/2404.02067v1)

**Authors**: Krzysztof Jankowski, Bartlomiej Sobieski, Mateusz Kwiatkowski, Jakub Szulc, Michal Janik, Hubert Baniecki, Przemyslaw Biecek

**Abstract**: Foundation models have emerged as pivotal tools, tackling many complex tasks through pre-training on vast datasets and subsequent fine-tuning for specific applications. The Segment Anything Model is one of the first and most well-known foundation models for computer vision segmentation tasks. This work presents a multi-faceted red-teaming analysis that tests the Segment Anything Model against challenging tasks: (1) We analyze the impact of style transfer on segmentation masks, demonstrating that applying adverse weather conditions and raindrops to dashboard images of city roads significantly distorts generated masks. (2) We focus on assessing whether the model can be used for attacks on privacy, such as recognizing celebrities' faces, and show that the model possesses some undesired knowledge in this task. (3) Finally, we check how robust the model is to adversarial attacks on segmentation masks under text prompts. We not only show the effectiveness of popular white-box attacks and resistance to black-box attacks but also introduce a novel approach - Focused Iterative Gradient Attack (FIGA) that combines white-box approaches to construct an efficient attack resulting in a smaller number of modified pixels. All of our testing methods and analyses indicate a need for enhanced safety measures in foundation models for image segmentation.

摘要: 基础模型已经成为关键工具，通过对大量数据集进行预培训并随后针对特定应用进行微调来处理许多复杂任务。任意分割模型是最早也是最著名的计算机视觉分割任务的基础模型之一。这项工作提出了一个多方面的红团队分析，针对具有挑战性的任务测试了Segment Anything Model：(1)我们分析了样式转移对分段掩模的影响，表明将不利的天气条件和雨滴应用于城市道路的仪表板图像会显著扭曲生成的掩模。(2)我们重点评估了该模型是否可以用于隐私攻击，如识别名人的脸，并表明该模型在该任务中具有一些不需要的知识。(3)最后，我们检验了该模型对文本提示下的分割模板攻击的健壮性。我们不仅展示了流行的白盒攻击的有效性和对黑盒攻击的抵抗力，而且还引入了一种新的专注于方法的迭代梯度攻击(FIGA)，它结合了白盒方法来构造有效的攻击，从而减少了修改的像素数。我们所有的测试方法和分析都表明，需要在图像分割的基础模型中加强安全措施。



## **46. How Trustworthy are Open-Source LLMs? An Assessment under Malicious Demonstrations Shows their Vulnerabilities**

开源LLM有多值得信赖？恶意示威下的评估显示其脆弱性 cs.CL

NAACL 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2311.09447v2) [paper-pdf](http://arxiv.org/pdf/2311.09447v2)

**Authors**: Lingbo Mo, Boshi Wang, Muhao Chen, Huan Sun

**Abstract**: The rapid progress in open-source Large Language Models (LLMs) is significantly driving AI development forward. However, there is still a limited understanding of their trustworthiness. Deploying these models at scale without sufficient trustworthiness can pose significant risks, highlighting the need to uncover these issues promptly. In this work, we conduct an adversarial assessment of open-source LLMs on trustworthiness, scrutinizing them across eight different aspects including toxicity, stereotypes, ethics, hallucination, fairness, sycophancy, privacy, and robustness against adversarial demonstrations. We propose advCoU, an extended Chain of Utterances-based (CoU) prompting strategy by incorporating carefully crafted malicious demonstrations for trustworthiness attack. Our extensive experiments encompass recent and representative series of open-source LLMs, including Vicuna, MPT, Falcon, Mistral, and Llama 2. The empirical outcomes underscore the efficacy of our attack strategy across diverse aspects. More interestingly, our result analysis reveals that models with superior performance in general NLP tasks do not always have greater trustworthiness; in fact, larger models can be more vulnerable to attacks. Additionally, models that have undergone instruction tuning, focusing on instruction following, tend to be more susceptible, although fine-tuning LLMs for safety alignment proves effective in mitigating adversarial trustworthiness attacks.

摘要: 开源大型语言模型(LLM)的快速发展显著地推动了人工智能的发展。然而，人们对他们的可信性仍知之甚少。在缺乏足够可信度的情况下大规模部署这些模型可能会带来重大风险，这突显了迅速发现这些问题的必要性。在这项工作中，我们对开源LLM的可信性进行了对抗性评估，从八个不同的方面对它们进行了仔细的审查，包括毒性、刻板印象、伦理、幻觉、公平性、奉承、隐私和对对抗性演示的健壮性。我们提出了AdvCoU，一种基于话语的扩展链(CUU)提示策略，它结合了精心制作的恶意演示来进行可信度攻击。我们广泛的实验涵盖了最近一系列具有代表性的开源LLM，包括Vicuna、MPT、Falcon、Mistral和Llama 2。经验结果强调了我们攻击策略在不同方面的有效性。更有趣的是，我们的结果分析显示，在一般NLP任务中性能优越的模型并不总是具有更大的可信度；事实上，较大的模型可能更容易受到攻击。此外，经过指令调整、专注于指令遵循的模型往往更容易受到影响，尽管针对安全对齐的微调LLM被证明在减轻对手信任攻击方面是有效的。



## **47. Deciphering the Interplay between Local Differential Privacy, Average Bayesian Privacy, and Maximum Bayesian Privacy**

解密局部差分隐私、平均贝叶斯隐私和最大贝叶斯隐私之间的相互作用 cs.LG

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2403.16591v3) [paper-pdf](http://arxiv.org/pdf/2403.16591v3)

**Authors**: Xiaojin Zhang, Yulin Fei, Wei Chen

**Abstract**: The swift evolution of machine learning has led to emergence of various definitions of privacy due to the threats it poses to privacy, including the concept of local differential privacy (LDP). Although widely embraced and utilized across numerous domains, this conventional approach to measure privacy still exhibits certain limitations, spanning from failure to prevent inferential disclosure to lack of consideration for the adversary's background knowledge. In this comprehensive study, we introduce Bayesian privacy and delve into the intricate relationship between LDP and its Bayesian counterparts, unveiling novel insights into utility-privacy trade-offs. We introduce a framework that encapsulates both attack and defense strategies, highlighting their interplay and effectiveness. The relationship between LDP and Maximum Bayesian Privacy (MBP) is first revealed, demonstrating that under uniform prior distribution, a mechanism satisfying $\xi$-LDP will satisfy $\xi$-MBP and conversely $\xi$-MBP also confers 2$\xi$-LDP. Our next theoretical contribution are anchored in the rigorous definitions and relationships between Average Bayesian Privacy (ABP) and Maximum Bayesian Privacy (MBP), encapsulated by equations $\epsilon_{p,a} \leq \frac{1}{\sqrt{2}}\sqrt{(\epsilon_{p,m} + \epsilon)\cdot(e^{\epsilon_{p,m} + \epsilon} - 1)}$. These relationships fortify our understanding of the privacy guarantees provided by various mechanisms. Our work not only lays the groundwork for future empirical exploration but also promises to facilitate the design of privacy-preserving algorithms, thereby fostering the development of trustworthy machine learning solutions.

摘要: 机器学习的快速发展导致了各种隐私定义的出现，因为它对隐私构成了威胁，包括局部差异隐私(LDP)的概念。尽管这种衡量隐私的传统方法在许多领域得到了广泛的接受和应用，但它仍然显示出一定的局限性，从未能阻止推论披露到缺乏对对手背景知识的考虑。在这项全面的研究中，我们介绍了贝叶斯隐私，并深入研究了自民党与其贝叶斯同行之间的错综复杂的关系，揭示了对效用-隐私权衡的新见解。我们引入了一个框架，该框架封装了攻击和防御战略，突出了它们的相互作用和有效性。首先揭示了LDP与最大贝叶斯隐私度之间的关系，证明了在均匀先验分布下，满足$xi-LDP的机制将满足$\xi-MBP，反之，$\xi-MBP也赋予2$\xi-LDP。我们的下一个理论贡献是建立在平均贝叶斯隐私度(ABP)和最大贝叶斯隐私度(MBP)之间的严格定义和关系上，用方程$\epsilon_{p，a}\leq\frac{1}{\sqrt{2}}\sqrt{(\epsilon_{p，m}+\epsilon)\cdot(e^{\epsilon_{p，m}+\epsilon}-1)}$来封装。这些关系加强了我们对各种机制提供的隐私保障的理解。我们的工作不仅为未来的经验探索奠定了基础，也承诺促进隐私保护算法的设计，从而促进可信机器学习解决方案的开发。



## **48. PatchCURE: Improving Certifiable Robustness, Model Utility, and Computation Efficiency of Adversarial Patch Defenses**

PatchCURE：提高对抗补丁防御的可证明鲁棒性、模型效用和计算效率 cs.CV

USENIX Security 2024. (extended) technical report

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2310.13076v2) [paper-pdf](http://arxiv.org/pdf/2310.13076v2)

**Authors**: Chong Xiang, Tong Wu, Sihui Dai, Jonathan Petit, Suman Jana, Prateek Mittal

**Abstract**: State-of-the-art defenses against adversarial patch attacks can now achieve strong certifiable robustness with a marginal drop in model utility. However, this impressive performance typically comes at the cost of 10-100x more inference-time computation compared to undefended models -- the research community has witnessed an intense three-way trade-off between certifiable robustness, model utility, and computation efficiency. In this paper, we propose a defense framework named PatchCURE to approach this trade-off problem. PatchCURE provides sufficient "knobs" for tuning defense performance and allows us to build a family of defenses: the most robust PatchCURE instance can match the performance of any existing state-of-the-art defense (without efficiency considerations); the most efficient PatchCURE instance has similar inference efficiency as undefended models. Notably, PatchCURE achieves state-of-the-art robustness and utility performance across all different efficiency levels, e.g., 16-23% absolute clean accuracy and certified robust accuracy advantages over prior defenses when requiring computation efficiency to be close to undefended models. The family of PatchCURE defenses enables us to flexibly choose appropriate defenses to satisfy given computation and/or utility constraints in practice.

摘要: 针对对抗性补丁攻击的最先进防御现在可以实现强大的可证明的健壮性，同时模型效用略有下降。然而，这种令人印象深刻的性能通常是以比无防御模型多10-100倍的推理时间计算为代价的--研究界见证了可证明的健壮性、模型实用性和计算效率之间的激烈三方权衡。在本文中，我们提出了一个名为PatchCURE的防御框架来解决这个权衡问题。PatchCURE为调整防御性能提供了足够的“旋钮”，并允许我们构建一系列防御：最健壮的PatchCURE实例可以与任何现有最先进的防御实例的性能相媲美(无需考虑效率)；最高效的PatchCURE实例具有与无防御模型相似的推理效率。值得注意的是，PatchCURE在所有不同的效率水平上实现了最先进的稳健性和实用性能，例如，当需要计算效率接近无防御模型时，绝对清洁准确率为16%-23%，并且经过认证的稳健精确度优于以前的防御系统。PatchCURE防御体系使我们能够灵活地选择适当的防御，以满足实践中给定的计算和/或效用约束。



## **49. Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack**

人性化机器生成内容：通过对抗攻击规避AI文本检测 cs.CL

Accepted by COLING 2024

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01907v1) [paper-pdf](http://arxiv.org/pdf/2404.01907v1)

**Authors**: Ying Zhou, Ben He, Le Sun

**Abstract**: With the development of large language models (LLMs), detecting whether text is generated by a machine becomes increasingly challenging in the face of malicious use cases like the spread of false information, protection of intellectual property, and prevention of academic plagiarism. While well-trained text detectors have demonstrated promising performance on unseen test data, recent research suggests that these detectors have vulnerabilities when dealing with adversarial attacks such as paraphrasing. In this paper, we propose a framework for a broader class of adversarial attacks, designed to perform minor perturbations in machine-generated content to evade detection. We consider two attack settings: white-box and black-box, and employ adversarial learning in dynamic scenarios to assess the potential enhancement of the current detection model's robustness against such attacks. The empirical results reveal that the current detection models can be compromised in as little as 10 seconds, leading to the misclassification of machine-generated text as human-written content. Furthermore, we explore the prospect of improving the model's robustness over iterative adversarial learning. Although some improvements in model robustness are observed, practical applications still face significant challenges. These findings shed light on the future development of AI-text detectors, emphasizing the need for more accurate and robust detection methods.

摘要: 随着大型语言模型(LLM)的发展，面对虚假信息传播、知识产权保护和防止学术剽窃等恶意使用案例，检测文本是否由机器生成变得越来越具有挑战性。虽然训练有素的文本检测器在看不见的测试数据上表现出了良好的性能，但最近的研究表明，这些检测器在处理诸如释义等敌意攻击时存在漏洞。在本文中，我们提出了一个更广泛类别的对抗性攻击的框架，旨在对机器生成的内容执行微小的扰动以逃避检测。我们考虑了两种攻击环境：白盒和黑盒，并在动态场景中使用对抗性学习来评估当前检测模型对此类攻击的稳健性的潜在增强。实验结果表明，当前的检测模型可以在短短10秒内被攻破，导致机器生成的文本被错误分类为人类书写的内容。此外，我们还探讨了改进模型在迭代对抗学习中的稳健性的前景。虽然在模型稳健性方面观察到了一些改进，但实际应用仍然面临着巨大的挑战。这些发现为人工智能文本检测器的未来发展指明了方向，强调了需要更准确和更稳健的检测方法。



## **50. Defense without Forgetting: Continual Adversarial Defense with Anisotropic & Isotropic Pseudo Replay**

不忘防御：具有各向异性和各向同性伪重放的连续对抗防御 cs.LG

**SubmitDate**: 2024-04-02    [abs](http://arxiv.org/abs/2404.01828v1) [paper-pdf](http://arxiv.org/pdf/2404.01828v1)

**Authors**: Yuhang Zhou, Zhongyun Hua

**Abstract**: Deep neural networks have demonstrated susceptibility to adversarial attacks. Adversarial defense techniques often focus on one-shot setting to maintain robustness against attack. However, new attacks can emerge in sequences in real-world deployment scenarios. As a result, it is crucial for a defense model to constantly adapt to new attacks, but the adaptation process can lead to catastrophic forgetting of previously defended against attacks. In this paper, we discuss for the first time the concept of continual adversarial defense under a sequence of attacks, and propose a lifelong defense baseline called Anisotropic \& Isotropic Replay (AIR), which offers three advantages: (1) Isotropic replay ensures model consistency in the neighborhood distribution of new data, indirectly aligning the output preference between old and new tasks. (2) Anisotropic replay enables the model to learn a compromise data manifold with fresh mixed semantics for further replay constraints and potential future attacks. (3) A straightforward regularizer mitigates the 'plasticity-stability' trade-off by aligning model output between new and old tasks. Experiment results demonstrate that AIR can approximate or even exceed the empirical performance upper bounds achieved by Joint Training.

摘要: 深度神经网络已显示出对敌意攻击的敏感性。对抗性防守技术通常集中在一次射击的设置上，以保持对攻击的健壮性。然而，在现实世界的部署场景中，新的攻击可能会按顺序出现。因此，对于防御模型来说，不断适应新的攻击是至关重要的，但适应过程可能会导致灾难性地忘记以前防御攻击的方式。本文首次讨论了一系列攻击下的连续对抗防御的概念，并提出了一种称为各向异性和各向同性重放(AIR)的终身防御基线，它具有三个优点：(1)各向同性重放保证了新数据在邻域分布上的模型一致性，间接地对齐了新旧任务之间的输出偏好。(2)各向异性重放使模型能够学习具有新鲜混合语义的折衷数据流形，用于进一步的重放约束和潜在的未来攻击。(3)通过调整新任务和旧任务之间的模型输出，直接的正则化可以缓解“塑性-稳定性”之间的权衡。实验结果表明，AIR可以接近甚至超过联合训练所获得的经验性能上限。



