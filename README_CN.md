# Latest Adversarial Attack Papers
**update at 2024-07-24 10:59:38**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. RedAgent: Red Teaming Large Language Models with Context-aware Autonomous Language Agent**

RedAgent：Red将大型语言模型与上下文感知自治语言代理结合起来 cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16667v1) [paper-pdf](http://arxiv.org/pdf/2407.16667v1)

**Authors**: Huiyu Xu, Wenhui Zhang, Zhibo Wang, Feng Xiao, Rui Zheng, Yunhe Feng, Zhongjie Ba, Kui Ren

**Abstract**: Recently, advanced Large Language Models (LLMs) such as GPT-4 have been integrated into many real-world applications like Code Copilot. These applications have significantly expanded the attack surface of LLMs, exposing them to a variety of threats. Among them, jailbreak attacks that induce toxic responses through jailbreak prompts have raised critical safety concerns. To identify these threats, a growing number of red teaming approaches simulate potential adversarial scenarios by crafting jailbreak prompts to test the target LLM. However, existing red teaming methods do not consider the unique vulnerabilities of LLM in different scenarios, making it difficult to adjust the jailbreak prompts to find context-specific vulnerabilities. Meanwhile, these methods are limited to refining jailbreak templates using a few mutation operations, lacking the automation and scalability to adapt to different scenarios. To enable context-aware and efficient red teaming, we abstract and model existing attacks into a coherent concept called "jailbreak strategy" and propose a multi-agent LLM system named RedAgent that leverages these strategies to generate context-aware jailbreak prompts. By self-reflecting on contextual feedback in an additional memory buffer, RedAgent continuously learns how to leverage these strategies to achieve effective jailbreaks in specific contexts. Extensive experiments demonstrate that our system can jailbreak most black-box LLMs in just five queries, improving the efficiency of existing red teaming methods by two times. Additionally, RedAgent can jailbreak customized LLM applications more efficiently. By generating context-aware jailbreak prompts towards applications on GPTs, we discover 60 severe vulnerabilities of these real-world applications with only two queries per vulnerability. We have reported all found issues and communicated with OpenAI and Meta for bug fixes.

摘要: 最近，GPT-4等高级大型语言模型(LLM)已集成到许多实际应用程序中，如Code Copilot。这些应用程序显著扩大了LLMS的攻击面，使它们暴露在各种威胁之下。其中，通过越狱提示引发有毒反应的越狱攻击引发了严重的安全问题。为了识别这些威胁，越来越多的红色团队方法通过精心编制越狱提示来测试目标LLM，以模拟潜在的敌对场景。然而，现有的红色团队方法没有考虑LLM在不同场景下的独特漏洞，很难调整越狱提示来发现上下文特定的漏洞。同时，这些方法仅限于使用少量的变异操作来提炼越狱模板，缺乏适应不同场景的自动化和可扩展性。为了实现上下文感知和高效的红色团队，我们将现有的攻击抽象并建模为一个连贯的概念，称为越狱策略，并提出了一个名为RedAgent的多代理LLM系统，该系统利用这些策略来生成上下文感知越狱提示。通过对额外内存缓冲区中的上下文反馈进行自我反思，RedAgent不断学习如何利用这些策略在特定上下文中实现有效的越狱。大量的实验表明，我们的系统可以在短短五次查询中破解大部分黑盒LLM，将现有的红色团队方法的效率提高了两倍。此外，RedAgent可以更高效地越狱定制的LLM应用程序。通过向GPT上的应用程序生成上下文感知越狱提示，我们发现了这些现实世界应用程序的60个严重漏洞，每个漏洞只有两个查询。我们已经报告了所有发现的问题，并与OpenAI和Meta进行了沟通以修复错误。



## **2. Defending Our Privacy With Backdoors**

用后门保护我们的隐私 cs.LG

Accepted at ECAI 2024

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2310.08320v4) [paper-pdf](http://arxiv.org/pdf/2310.08320v4)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Daniel Neider, Kristian Kersting

**Abstract**: The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information, such as names and faces of individuals, from vision-language models by fine-tuning them for only a few minutes instead of re-training them from scratch. Specifically, by strategically inserting backdoors into text encoders, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's actual name. For image encoders, we map individuals' embeddings to be removed from the model to a universal, anonymous embedding. The results of our extensive experimental evaluation demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides a new "dual-use" perspective on backdoor attacks and presents a promising avenue to enhance the privacy of individuals within models trained on uncurated web-scraped data.

摘要: 大型人工智能模型的激增引发了人们对隐私的严重担忧。这些模型针对未经管理的、往往是敏感的网络数据进行培训。其中一个令人担忧的问题是，攻击者可以使用隐私攻击来提取有关训练数据的信息。不幸的是，在不牺牲性能的情况下从模型中删除特定信息的任务并不简单，而且已被证明是具有挑战性的。我们提出了一种基于后门攻击的相当简单但有效的防御方法，通过仅对视觉语言模型进行几分钟的微调来删除私人信息，如个人的姓名和面孔，而不是从头开始重新训练它们。具体地说，通过有策略地在文本编码器中插入后门，我们将敏感短语的嵌入与中性术语的嵌入--“人”而不是人的实际姓名--对齐。对于图像编码器，我们将从模型中移除的个人嵌入映射到通用的匿名嵌入。我们广泛的实验评估结果证明了我们的基于后门的CLIP防御的有效性，通过使用专门的针对零射击分类器的隐私攻击来评估其性能。我们的方法为后门攻击提供了一种新的“两用”视角，并提供了一种在未经管理的网络抓取数据的培训模型中增强个人隐私的有前景的途径。



## **3. Securing Tomorrow's Smart Cities: Investigating Software Security in Internet of Vehicles and Deep Learning Technologies**

确保未来的智慧城市：调查车联网和深度学习技术中的软件安全 cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16410v1) [paper-pdf](http://arxiv.org/pdf/2407.16410v1)

**Authors**: Ridhi Jain, Norbert Tihanyi, Mohamed Amine Ferrag

**Abstract**: Integrating Deep Learning (DL) techniques in the Internet of Vehicles (IoV) introduces many security challenges and issues that require thorough examination. This literature review delves into the inherent vulnerabilities and risks associated with DL in IoV systems, shedding light on the multifaceted nature of security threats. Through an extensive analysis of existing research, we explore potential threats posed by DL algorithms, including adversarial attacks, data privacy breaches, and model poisoning. Additionally, we investigate the impact of DL on critical aspects of IoV security, such as intrusion detection, anomaly detection, and secure communication protocols. Our review emphasizes the complexities of ensuring the robustness, reliability, and trustworthiness of DL-based IoV systems, given the dynamic and interconnected nature of vehicular networks. Furthermore, we discuss the need for novel security solutions tailored to address these challenges effectively and enhance the security posture of DL-enabled IoV environments. By offering insights into these critical issues, this chapter aims to stimulate further research, innovation, and collaboration in securing DL techniques within the context of the IoV, thereby fostering a safer and more resilient future for vehicular communication and connectivity.

摘要: 在车联网(IoV)中集成深度学习(DL)技术带来了许多安全挑战和问题，需要进行彻底的检查。这篇文献综述深入探讨了IoV系统中与DL相关的固有漏洞和风险，揭示了安全威胁的多方面性质。通过对现有研究的广泛分析，我们探索了DL算法带来的潜在威胁，包括对抗性攻击、数据隐私泄露和模型中毒。此外，我们还研究了DL对IoV安全的关键方面的影响，例如入侵检测、异常检测和安全通信协议。考虑到车载网络的动态和互联特性，我们的审查强调了确保基于DL的IoV系统的健壮性、可靠性和可信性的复杂性。此外，我们还讨论了为有效应对这些挑战并增强支持DL的IoV环境的安全态势而量身定做的新型安全解决方案的必要性。通过提供对这些关键问题的见解，本章旨在鼓励在万物互联背景下保护DL技术方面的进一步研究、创新和合作，从而为车辆通信和连接培养一个更安全、更具弹性的未来。



## **4. Protecting Quantum Procrastinators with Signature Lifting: A Case Study in Cryptocurrencies**

通过签名提升保护量子拖延者：加密货币的案例研究 cs.CR

Minor revision

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2303.06754v2) [paper-pdf](http://arxiv.org/pdf/2303.06754v2)

**Authors**: Or Sattath, Shai Wyborski

**Abstract**: Current solutions to quantum vulnerabilities of widely used cryptographic schemes involve migrating users to post-quantum schemes before quantum attacks become feasible. This work deals with protecting quantum procrastinators: users that failed to migrate to post-quantum cryptography in time.   To address this problem in the context of digital signatures, we introduce a technique called signature lifting, that allows us to lift a deployed pre-quantum signature scheme satisfying a certain property to a post-quantum signature scheme that uses the same keys. Informally, the said property is that a post-quantum one-way function is used "somewhere along the way" to derive the public-key from the secret-key. Our constructions of signature lifting relies heavily on the post-quantum digital signature scheme Picnic (Chase et al., CCS'17).   Our main case-study is cryptocurrencies, where this property holds in two scenarios: when the public-key is generated via a key-derivation function or when the public-key hash is posted instead of the public-key itself. We propose a modification, based on signature lifting, that can be applied in many cryptocurrencies for securely spending pre-quantum coins in presence of quantum adversaries. Our construction improves upon existing constructions in two major ways: it is not limited to pre-quantum coins whose ECDSA public-key has been kept secret (and in particular, it handles all coins that are stored in addresses generated by HD wallets), and it does not require access to post-quantum coins or using side payments to pay for posting the transaction.

摘要: 目前针对广泛使用的密码方案的量子漏洞的解决方案包括在量子攻击变得可行之前将用户迁移到后量子方案。这项工作涉及保护量子拖延者：未能及时迁移到后量子密码学的用户。为了在数字签名的背景下解决这个问题，我们引入了一种称为签名提升的技术，该技术允许我们将满足一定性质的部署的前量子签名方案提升到使用相同密钥的后量子签名方案。非正式地，所述性质是使用后量子单向函数来从秘密密钥导出公钥。我们的签名提升的构造在很大程度上依赖于后量子数字签名方案Picnic(Chase等人，CCS‘17)。我们的主要案例研究是加密货币，其中该属性在两种情况下成立：当公钥是通过密钥派生函数生成时，或者当公钥散列被发布而不是公钥本身时。我们提出了一种基于签名提升的改进方案，该方案可以应用于多种加密货币，以便在存在量子对手的情况下安全地消费前量子币。我们的结构在两个主要方面对现有结构进行了改进：它不仅限于其ECDSA公钥被保密的前量子硬币(尤其是，它处理存储在HD钱包生成的地址中的所有硬币)，并且它不需要访问后量子硬币或使用附带支付来支付发布交易的费用。



## **5. Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models**

通过扩散模型高效生成视觉语言模型的有针对性且可转移的对抗示例 cs.CV

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2404.10335v3) [paper-pdf](http://arxiv.org/pdf/2404.10335v3)

**Authors**: Qi Guo, Shanmin Pang, Xiaojun Jia, Yang Liu, Qing Guo

**Abstract**: Adversarial attacks, particularly \textbf{targeted} transfer-based attacks, can be used to assess the adversarial robustness of large visual-language models (VLMs), allowing for a more thorough examination of potential security flaws before deployment. However, previous transfer-based adversarial attacks incur high costs due to high iteration counts and complex method structure. Furthermore, due to the unnaturalness of adversarial semantics, the generated adversarial examples have low transferability. These issues limit the utility of existing methods for assessing robustness. To address these issues, we propose AdvDiffVLM, which uses diffusion models to generate natural, unrestricted and targeted adversarial examples via score matching. Specifically, AdvDiffVLM uses Adaptive Ensemble Gradient Estimation to modify the score during the diffusion model's reverse generation process, ensuring that the produced adversarial examples have natural adversarial targeted semantics, which improves their transferability. Simultaneously, to improve the quality of adversarial examples, we use the GradCAM-guided Mask method to disperse adversarial semantics throughout the image rather than concentrating them in a single area. Finally, AdvDiffVLM embeds more target semantics into adversarial examples after multiple iterations. Experimental results show that our method generates adversarial examples 5x to 10x faster than state-of-the-art transfer-based adversarial attacks while maintaining higher quality adversarial examples. Furthermore, compared to previous transfer-based adversarial attacks, the adversarial examples generated by our method have better transferability. Notably, AdvDiffVLM can successfully attack a variety of commercial VLMs in a black-box environment, including GPT-4V.

摘要: 对抗性攻击，特别是基于传输的对抗性攻击，可用于评估大型视觉语言模型(VLM)的对抗性健壮性，从而允许在部署之前更彻底地检查潜在的安全漏洞。然而，以往基于转移的对抗性攻击由于迭代次数多、方法结构复杂，代价较高。此外，由于对抗性语义的非自然性，生成的对抗性实例可转移性较低。这些问题限制了现有稳健性评估方法的实用性。为了解决这些问题，我们提出了AdvDiffVLM，它使用扩散模型通过得分匹配来生成自然的、不受限制的和有针对性的对抗性实例。具体地说，AdvDiffVLM在扩散模型的反向生成过程中使用自适应集成梯度估计来修改分数，确保生成的对抗性实例具有自然对抗性目标语义，从而提高了它们的可转移性。同时，为了提高对抗性实例的质量，我们使用了GradCAM引导的掩码方法，将对抗性语义分散在整个图像中，而不是将它们集中在单个区域。最后，在多次迭代后，AdvDiffVLM将更多的目标语义嵌入到对抗性实例中。实验结果表明，在保持较高质量的对抗性实例的同时，我们的方法生成对抗性实例的速度比最新的基于传输的对抗性攻击快5倍到10倍。此外，与以往基于转移的对抗性攻击相比，该方法生成的对抗性实例具有更好的可转移性。值得注意的是，AdvDiffVLM可以在黑盒环境中成功攻击各种商业VLM，包括GPT-4V。



## **6. R.A.C.E.: Robust Adversarial Concept Erasure for Secure Text-to-Image Diffusion Model**

皇家海关：安全文本到图像扩散模型的鲁棒对抗概念擦除 cs.CV

Accepted at ECCV 2024

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2405.16341v2) [paper-pdf](http://arxiv.org/pdf/2405.16341v2)

**Authors**: Changhoon Kim, Kyle Min, Yezhou Yang

**Abstract**: In the evolving landscape of text-to-image (T2I) diffusion models, the remarkable capability to generate high-quality images from textual descriptions faces challenges with the potential misuse of reproducing sensitive content. To address this critical issue, we introduce \textbf{R}obust \textbf{A}dversarial \textbf{C}oncept \textbf{E}rase (RACE), a novel approach designed to mitigate these risks by enhancing the robustness of concept erasure method for T2I models. RACE utilizes a sophisticated adversarial training framework to identify and mitigate adversarial text embeddings, significantly reducing the Attack Success Rate (ASR). Impressively, RACE achieves a 30 percentage point reduction in ASR for the ``nudity'' concept against the leading white-box attack method. Our extensive evaluations demonstrate RACE's effectiveness in defending against both white-box and black-box attacks, marking a significant advancement in protecting T2I diffusion models from generating inappropriate or misleading imagery. This work underlines the essential need for proactive defense measures in adapting to the rapidly advancing field of adversarial challenges. Our code is publicly available: \url{https://github.com/chkimmmmm/R.A.C.E.}

摘要: 在不断发展的文本到图像(T2I)扩散模型中，从文本描述生成高质量图像的非凡能力面临着潜在的误用，即复制敏感内容。为了解决这一关键问题，我们引入了一种新的方法，即RACE(RACE)，旨在通过增强T2I模型的概念删除方法的健壮性来降低这些风险。RACE利用复杂的对抗性训练框架来识别和缓解对抗性文本嵌入，显著降低攻击成功率(ASR)。令人印象深刻的是，与领先的白盒攻击方法相比，RACE实现了“裸体”概念的ASR降低30个百分点。我们广泛的评估证明了RACE在防御白盒和黑盒攻击方面的有效性，标志着在保护T2I扩散模型免受不适当或误导图像方面取得了重大进展。这项工作突出表明，必须采取积极主动的防御措施，以适应迅速发展的对抗性挑战领域。我们的代码是公开提供的：\url{https://github.com/chkimmmmm/R.A.C.E.}



## **7. Algebraic Adversarial Attacks on Integrated Gradients**

对综合学生的代数对抗攻击 cs.LG

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16233v1) [paper-pdf](http://arxiv.org/pdf/2407.16233v1)

**Authors**: Lachlan Simpson, Federico Costanza, Kyle Millar, Adriel Cheng, Cheng-Chew Lim, Hong Gunn Chew

**Abstract**: Adversarial attacks on explainability models have drastic consequences when explanations are used to understand the reasoning of neural networks in safety critical systems. Path methods are one such class of attribution methods susceptible to adversarial attacks. Adversarial learning is typically phrased as a constrained optimisation problem. In this work, we propose algebraic adversarial examples and study the conditions under which one can generate adversarial examples for integrated gradients. Algebraic adversarial examples provide a mathematically tractable approach to adversarial examples.

摘要: 当使用解释来理解安全关键系统中神经网络的推理时，对可解释性模型的对抗攻击会产生严重后果。路径方法是一类容易受到对抗攻击的归因方法。对抗学习通常被描述为一个受约束的优化问题。在这项工作中，我们提出了代数对抗性示例，并研究了可以为综合梯度生成对抗性示例的条件。代数对抗性例子为对抗性例子提供了一种数学上易于处理的方法。



## **8. EVD4UAV: An Altitude-Sensitive Benchmark to Evade Vehicle Detection in UAV**

EVD 4无人机：躲避无人机车辆检测的高度敏感基准 cs.CV

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2403.05422v2) [paper-pdf](http://arxiv.org/pdf/2403.05422v2)

**Authors**: Huiming Sun, Jiacheng Guo, Zibo Meng, Tianyun Zhang, Jianwu Fang, Yuewei Lin, Hongkai Yu

**Abstract**: Vehicle detection in Unmanned Aerial Vehicle (UAV) captured images has wide applications in aerial photography and remote sensing. There are many public benchmark datasets proposed for the vehicle detection and tracking in UAV images. Recent studies show that adding an adversarial patch on objects can fool the well-trained deep neural networks based object detectors, posing security concerns to the downstream tasks. However, the current public UAV datasets might ignore the diverse altitudes, vehicle attributes, fine-grained instance-level annotation in mostly side view with blurred vehicle roof, so none of them is good to study the adversarial patch based vehicle detection attack problem. In this paper, we propose a new dataset named EVD4UAV as an altitude-sensitive benchmark to evade vehicle detection in UAV with 6,284 images and 90,886 fine-grained annotated vehicles. The EVD4UAV dataset has diverse altitudes (50m, 70m, 90m), vehicle attributes (color, type), fine-grained annotation (horizontal and rotated bounding boxes, instance-level mask) in top view with clear vehicle roof. One white-box and two black-box patch based attack methods are implemented to attack three classic deep neural networks based object detectors on EVD4UAV. The experimental results show that these representative attack methods could not achieve the robust altitude-insensitive attack performance.

摘要: 无人机拍摄的图像中的车辆检测在航空摄影和遥感中有着广泛的应用。针对无人机图像中的车辆检测和跟踪，已经提出了许多公开的基准数据集。最近的研究表明，在对象上添加对抗性补丁可以欺骗训练有素的基于深度神经网络的对象检测器，从而给下游任务带来安全隐患。然而，目前公开的无人机数据集可能忽略了车顶模糊的侧视图中不同的高度、车辆属性、细粒度的实例级标注，因此不利于研究基于对抗性补丁的车辆检测攻击问题。在本文中，我们提出了一个新的数据集EVD4UAV作为高度敏感基准来逃避无人机中的车辆检测，该数据集包含6,284张图像和90,886辆细粒度标注的车辆。EVD4UAV数据集在俯视图中具有不同的高度(50m、70m、90m)、车辆属性(颜色、类型)、细粒度注释(水平和旋转的边界框、实例级遮罩)，并具有清晰的车顶。采用一种基于白盒和两种基于黑盒补丁的攻击方法，对EVD4无人机上三种经典的基于深度神经网络的目标探测器进行攻击。实验结果表明，这些具有代表性的攻击方法不能达到稳健的高度不敏感攻击性能。



## **9. Detecting Brittle Decisions for Free: Leveraging Margin Consistency in Deep Robust Classifiers**

免费检测脆弱决策：利用深度稳健分类器中的保证金一致性 cs.LG

11 pages, 7 figures, 2 tables, 1 algorithm. Version Update: Figure 6

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2406.18451v2) [paper-pdf](http://arxiv.org/pdf/2406.18451v2)

**Authors**: Jonas Ngnawé, Sabyasachi Sahoo, Yann Pequignot, Frédéric Precioso, Christian Gagné

**Abstract**: Despite extensive research on adversarial training strategies to improve robustness, the decisions of even the most robust deep learning models can still be quite sensitive to imperceptible perturbations, creating serious risks when deploying them for high-stakes real-world applications. While detecting such cases may be critical, evaluating a model's vulnerability at a per-instance level using adversarial attacks is computationally too intensive and unsuitable for real-time deployment scenarios. The input space margin is the exact score to detect non-robust samples and is intractable for deep neural networks. This paper introduces the concept of margin consistency -- a property that links the input space margins and the logit margins in robust models -- for efficient detection of vulnerable samples. First, we establish that margin consistency is a necessary and sufficient condition to use a model's logit margin as a score for identifying non-robust samples. Next, through comprehensive empirical analysis of various robustly trained models on CIFAR10 and CIFAR100 datasets, we show that they indicate strong margin consistency with a strong correlation between their input space margins and the logit margins. Then, we show that we can effectively use the logit margin to confidently detect brittle decisions with such models and accurately estimate robust accuracy on an arbitrarily large test set by estimating the input margins only on a small subset. Finally, we address cases where the model is not sufficiently margin-consistent by learning a pseudo-margin from the feature representation. Our findings highlight the potential of leveraging deep representations to efficiently assess adversarial vulnerability in deployment scenarios.

摘要: 尽管对对抗性训练策略进行了大量研究以提高稳健性，但即使是最健壮的深度学习模型的决策也可能对不可察觉的扰动非常敏感，当将它们部署到高风险的现实世界应用程序时，会产生严重的风险。虽然检测这类情况可能很关键，但使用对抗性攻击在每个实例级别评估模型的漏洞计算量太大，不适合实时部署场景。输入空间裕度是检测非稳健样本的准确分数，对于深度神经网络来说是很难处理的。为了有效地检测易受攻击的样本，本文引入了边缘一致性的概念--一种将输入空间边缘和健壮模型中的Logit边缘联系起来的属性。首先，我们证明了边际一致性是使用模型的Logit边际作为识别非稳健样本的分数的充要条件。接下来，通过对CIFAR10和CIFAR100数据集上各种稳健训练模型的综合实证分析，我们发现它们表明了很强的边际一致性，并且它们的输入空间边际和Logit边际之间存在很强的相关性。然后，我们证明了我们可以有效地使用Logit裕度来自信地检测此类模型的脆性决策，并通过仅在较小的子集上估计输入裕度来准确地估计任意大测试集上的稳健精度。最后，我们通过从特征表示学习伪边距来处理模型不够边距一致的情况。我们的发现突出了利用深度陈述来有效评估部署场景中的对手脆弱性的潜力。



## **10. Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts**

彩虹团队：开放式一代的多元化对抗预言 cs.CL

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2402.16822v2) [paper-pdf](http://arxiv.org/pdf/2402.16822v2)

**Authors**: Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H. Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, Tim Rocktäschel, Roberta Raileanu

**Abstract**: As large language models (LLMs) become increasingly prevalent across many real-world applications, understanding and enhancing their robustness to adversarial attacks is of paramount importance. Existing methods for identifying adversarial prompts tend to focus on specific domains, lack diversity, or require extensive human annotations. To address these limitations, we present Rainbow Teaming, a novel black-box approach for producing a diverse collection of adversarial prompts. Rainbow Teaming casts adversarial prompt generation as a quality-diversity problem, and uses open-ended search to generate prompts that are both effective and diverse. Focusing on the safety domain, we use Rainbow Teaming to target various state-of-the-art LLMs, including the Llama 2 and Llama 3 models. Our approach reveals hundreds of effective adversarial prompts, with an attack success rate exceeding 90% across all tested models. Furthermore, we demonstrate that fine-tuning models with synthetic data generated by the Rainbow Teaming method significantly enhances their safety without sacrificing general performance or helpfulness. We additionally explore the versatility of Rainbow Teaming by applying it to question answering and cybersecurity, showcasing its potential to drive robust open-ended self-improvement in a wide range of applications.

摘要: 随着大型语言模型(LLM)在许多真实世界的应用中变得越来越普遍，理解和增强它们对对手攻击的健壮性是至关重要的。现有的识别对抗性提示的方法往往集中在特定的领域，缺乏多样性，或者需要大量的人工注释。为了解决这些局限性，我们提出了彩虹分组，这是一种新的黑盒方法，用于产生多样化的对抗性提示集合。彩虹团队将敌意提示生成视为质量多样性问题，并使用开放式搜索来生成既有效又多样化的提示。专注于安全领域，我们使用彩虹团队瞄准各种最先进的LLM，包括Llama 2和Llama 3型号。我们的方法揭示了数百个有效的对抗性提示，在所有测试模型上的攻击成功率超过90%。此外，我们证明了使用彩虹组合方法生成的合成数据对模型进行微调显著增强了它们的安全性，而不会牺牲总体性能或帮助。我们还探讨了彩虹团队的多功能性，将其应用于问题回答和网络安全，展示了其在广泛应用中推动强大的开放式自我改进的潜力。



## **11. Enhancing Transferability of Targeted Adversarial Examples: A Self-Universal Perspective**

增强有针对性的对抗性示例的可移植性：自我普遍的视角 cs.CV

8 pages and 9 figures

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15683v1) [paper-pdf](http://arxiv.org/pdf/2407.15683v1)

**Authors**: Bowen Peng, Li Liu, Tianpeng Liu, Zhen Liu, Yongxiang Liu

**Abstract**: Transfer-based targeted adversarial attacks against black-box deep neural networks (DNNs) have been proven to be significantly more challenging than untargeted ones. The impressive transferability of current SOTA, the generative methods, comes at the cost of requiring massive amounts of additional data and time-consuming training for each targeted label. This results in limited efficiency and flexibility, significantly hindering their deployment in practical applications. In this paper, we offer a self-universal perspective that unveils the great yet underexplored potential of input transformations in pursuing this goal. Specifically, transformations universalize gradient-based attacks with intrinsic but overlooked semantics inherent within individual images, exhibiting similar scalability and comparable results to time-consuming learning over massive additional data from diverse classes. We also contribute a surprising empirical insight that one of the most fundamental transformations, simple image scaling, is highly effective, scalable, sufficient, and necessary in enhancing targeted transferability. We further augment simple scaling with orthogonal transformations and block-wise applicability, resulting in the Simple, faSt, Self-universal yet Strong Scale Transformation (S$^4$ST) for self-universal TTA. On the ImageNet-Compatible benchmark dataset, our method achieves a 19.8% improvement in the average targeted transfer success rate against various challenging victim models over existing SOTA transformation methods while only consuming 36% time for attacking. It also outperforms resource-intensive attacks by a large margin in various challenging settings.

摘要: 针对黑盒深度神经网络(DNN)的基于转移的定向攻击已被证明比非定向攻击具有更大的挑战性。当前SOTA的可转移性令人印象深刻，这是以需要大量额外数据和为每个目标标签进行耗时的培训为代价的。这导致了效率和灵活性的限制，大大阻碍了它们在实际应用中的部署。在这篇文章中，我们提供了一个自我普适的视角，揭示了投入转换在追求这一目标方面的巨大潜力，但尚未得到充分开发。具体地说，变换将基于梯度的攻击通用化，具有内在但被忽略的个体图像固有的语义，表现出与来自不同类别的耗时的额外数据的耗时学习类似的可扩展性和可比性。我们还提供了一个令人惊讶的经验见解，即最基本的转换之一，简单的图像缩放，在增强目标可转移性方面是高度有效、可扩展、充分和必要的。我们用正交变换和分块适用性进一步增强了简单的尺度变换，得到了简单、快速、自泛的强尺度变换(S$^4$ST)。在与ImageNet兼容的基准数据集上，与已有的SOTA变换方法相比，该方法对各种具有挑战性的受害者模型的平均目标传输成功率提高了19.8%，而攻击所需的时间仅为36%。在各种具有挑战性的环境中，它的性能也远远超过资源密集型攻击。



## **12. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

通过大语言模型进行对抗风格增强以实现稳健的假新闻检测 cs.CL

8 pages

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2406.11260v2) [paper-pdf](http://arxiv.org/pdf/2406.11260v2)

**Authors**: Sungwon Park, Sungwon Han, Meeyoung Cha

**Abstract**: The spread of fake news negatively impacts individuals and is regarded as a significant social challenge that needs to be addressed. A number of algorithmic and insightful features have been identified for detecting fake news. However, with the recent LLMs and their advanced generation capabilities, many of the detectable features (e.g., style-conversion attacks) can be altered, making it more challenging to distinguish from real news. This study proposes adversarial style augmentation, AdStyle, to train a fake news detector that remains robust against various style-conversion attacks. Our model's key mechanism is the careful use of LLMs to automatically generate a diverse yet coherent range of style-conversion attack prompts. This improves the generation of prompts that are particularly difficult for the detector to handle. Experiments show that our augmentation strategy improves robustness and detection performance when tested on fake news benchmark datasets.

摘要: 假新闻的传播对个人产生负面影响，被视为需要解决的重大社会挑战。已经确定了许多算法和有洞察力的功能来检测假新闻。然而，随着最近的LLM及其先进一代能力，许多可检测的特征（例如，风格转换攻击）可以被更改，使其与真实新闻区分起来更具挑战性。这项研究提出了对抗性风格增强AdStyle来训练一个假新闻检测器，该检测器在对抗各种风格转换攻击时保持稳健。我们模型的关键机制是仔细使用LLM来自动生成多样化但连贯的风格转换攻击提示。这改善了检测器特别难以处理的提示的生成。实验表明，当在假新闻基准数据集上进行测试时，我们的增强策略提高了鲁棒性和检测性能。



## **13. Revisiting the Robust Alignment of Circuit Breakers**

重新审视断路器的稳健对准 cs.CR

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15902v1) [paper-pdf](http://arxiv.org/pdf/2407.15902v1)

**Authors**: Leo Schwinn, Simon Geisler

**Abstract**: Over the past decade, adversarial training has emerged as one of the few reliable methods for enhancing model robustness against adversarial attacks [Szegedy et al., 2014, Madry et al., 2018, Xhonneux et al., 2024], while many alternative approaches have failed to withstand rigorous subsequent evaluations. Recently, an alternative defense mechanism, namely "circuit breakers" [Zou et al., 2024], has shown promising results for aligning LLMs. In this report, we show that the robustness claims of "Improving Alignment and Robustness with Circuit Breakers" against unconstraint continuous attacks in the embedding space of the input tokens may be overestimated [Zou et al., 2024]. Specifically, we demonstrate that by implementing a few simple changes to embedding space attacks [Schwinn et al., 2024a,b], we achieve 100% attack success rate (ASR) against circuit breaker models. Without conducting any further hyperparameter tuning, these adjustments increase the ASR by more than 80% compared to the original evaluation. Code is accessible at: https://github.com/SchwinnL/circuit-breakers-eval

摘要: 在过去的十年中，对抗性训练已经成为少数几种增强模型对对抗性攻击的稳健性的可靠方法之一[Szegedy等人，2014，Madry等人，2018，Xhonneux等人，2024]，而许多替代方法未能经受住严格的后续评估。最近，一种替代的防御机制，即“断路器”[Zou等人，2024]，在对准LLM方面显示了令人振奋的结果。在这份报告中，我们证明了在输入令牌的嵌入空间中使用断路器来提高对非约束连续攻击的稳健性主张可能被高估了[Zou等人，2024]。具体地说，我们证明了通过对嵌入空间攻击[Schwinn等人，2024a，b]进行一些简单的更改，我们对断路器模型实现了100%的攻击成功率(ASR)。在不进行任何进一步的超参数调整的情况下，这些调整使ASR与原始评估相比增加了80%以上。代码可在以下网址访问：https://github.com/SchwinnL/circuit-breakers-eval



## **14. Targeted Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

有针对性的隐性对抗培训提高了LLM对持续有害行为的稳健性 cs.LG

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15549v1) [paper-pdf](http://arxiv.org/pdf/2407.15549v1)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of `jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.

摘要: 大型语言模型(LLM)通常会以不受欢迎的方式运行，因此它们被明确微调为不以这种方式运行。例如，伦敦大学法学院的红队文学创作了各种各样的“越狱”技术，从经过微调的无害模特那里引出有害文本。最近在红团队、模型编辑和可解释性方面的工作表明，这一挑战源于(对抗性的)微调如何在很大程度上抑制而不是消除LLM中不受欢迎的能力。以前的工作已经引入了潜在的对手训练(LAT)，作为一种提高对广泛类别的故障的稳健性的方式。这些先前的工作考虑了无目标的潜在空间攻击，即对手扰乱潜在激活，以最大限度地减少期望行为的示例损失。非定向LAT可以提供一般类型的健壮性，但不利用有关特定故障模式的信息。在这里，我们实验有针对性的LAT，其中对手试图将特定竞争任务的损失降至最低。我们发现，它可以增加各种最先进的方法。首先，我们使用有针对性的LAT来提高对越狱的健壮性，性能优于强大的R2D2基线，计算量少了几个数量级。其次，我们使用它来更有效地删除后门，而不知道触发器。最后，我们使用它来更有效地忘记特定不受欢迎的任务的知识，这种方式也更适合重新学习。总体而言，我们的结果表明，有针对性的LAT可以成为防御LLM有害行为的有效工具。



## **15. Towards Efficient Transferable Preemptive Adversarial Defense**

迈向高效的可转让先发制人的对抗防御 cs.CR

Under Review

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15524v1) [paper-pdf](http://arxiv.org/pdf/2407.15524v1)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: Deep learning technology has brought convenience and advanced developments but has become untrustworthy because of its sensitivity to inconspicuous perturbations (i.e., adversarial attacks). Attackers utilize this sensitivity to slightly manipulate transmitted messages. To defend against such attacks, we have devised a strategy for "attacking" the message before it is attacked. This strategy, dubbed Fast Preemption, provides an efficient transferable preemptive defense by using different models for labeling inputs and learning crucial features. A forward-backward cascade learning algorithm is used to compute protective perturbations, starting with forward propagation optimization to achieve rapid convergence, followed by iterative backward propagation learning to alleviate overfitting. This strategy offers state-of-the-art transferability and protection across various systems. With the running of only three steps, our Fast Preemption framework outperforms benchmark training-time, test-time, and preemptive adversarial defenses. We have also devised the first to our knowledge effective white-box adaptive reversion attack and demonstrate that the protection added by our defense strategy is irreversible unless the backbone model, algorithm, and settings are fully compromised. This work provides a new direction to developing active defenses against adversarial attacks.

摘要: 深度学习技术带来了便利和先进的发展，但由于其对不起眼的扰动(即对抗性攻击)的敏感性而变得不可信任。攻击者利用这种敏感度略微操纵传输的消息。为了防御这种攻击，我们设计了一种在消息受到攻击之前对其进行“攻击”的策略。这一战略被称为快速抢占，通过使用不同的模型来标记输入和学习关键特征，提供了一种高效的可转移的抢占防御。保护摄动的计算采用前向-后向级联学习算法，从前向传播优化开始实现快速收敛，然后迭代后向传播学习以减少过拟合。这一战略提供了最先进的跨各种系统的可转移性和保护。由于只运行了三个步骤，我们的快速抢占框架的性能优于基准训练时间、测试时间和先发制人的对手防御。我们还设计了我们所知的第一个有效的白盒自适应恢复攻击，并证明了我们的防御策略添加的保护是不可逆转的，除非主干模型、算法和设置完全受损。这一工作为主动防御敌方攻击提供了新的方向。



## **16. A Closer Look at GAN Priors: Exploiting Intermediate Features for Enhanced Model Inversion Attacks**

仔细研究GAN先验：利用中间功能进行增强模型倒置攻击 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.13863v2) [paper-pdf](http://arxiv.org/pdf/2407.13863v2)

**Authors**: Yixiang Qiu, Hao Fang, Hongyao Yu, Bin Chen, MeiKang Qiu, Shu-Tao Xia

**Abstract**: Model Inversion (MI) attacks aim to reconstruct privacy-sensitive training data from released models by utilizing output information, raising extensive concerns about the security of Deep Neural Networks (DNNs). Recent advances in generative adversarial networks (GANs) have contributed significantly to the improved performance of MI attacks due to their powerful ability to generate realistic images with high fidelity and appropriate semantics. However, previous MI attacks have solely disclosed private information in the latent space of GAN priors, limiting their semantic extraction and transferability across multiple target models and datasets. To address this challenge, we propose a novel method, Intermediate Features enhanced Generative Model Inversion (IF-GMI), which disassembles the GAN structure and exploits features between intermediate blocks. This allows us to extend the optimization space from latent code to intermediate features with enhanced expressive capabilities. To prevent GAN priors from generating unrealistic images, we apply a L1 ball constraint to the optimization process. Experiments on multiple benchmarks demonstrate that our method significantly outperforms previous approaches and achieves state-of-the-art results under various settings, especially in the out-of-distribution (OOD) scenario. Our code is available at: https://github.com/final-solution/IF-GMI

摘要: 模型反转(MI)攻击的目的是利用输出信息从已发布的模型中重建隐私敏感的训练数据，这引起了人们对深度神经网络(DNN)安全性的广泛关注。生成性对抗网络(GANS)的最新进展为MI攻击的性能改进做出了重要贡献，因为它们能够生成高保真和适当语义的真实图像。然而，以往的MI攻击只在GaN先验的潜在空间中泄露隐私信息，限制了它们的语义提取和跨多个目标模型和数据集的可传输性。为了解决这一挑战，我们提出了一种新的方法，中间特征增强的生成性模型反转(IF-GMI)，它分解GaN结构并利用中间块之间的特征。这允许我们将优化空间从潜在代码扩展到具有增强表达能力的中间功能。为了防止GaN先验数据产生不真实的图像，我们在优化过程中应用了L1球约束。在多个基准测试上的实验表明，我们的方法显著优于以前的方法，并在各种设置下获得了最先进的结果，特别是在分布外(OOD)的情况下。我们的代码请访问：https://github.com/final-solution/IF-GMI



## **17. CLIP-Guided Networks for Transferable Targeted Attacks**

CLIP引导的可转移定向攻击网络 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.10179v2) [paper-pdf](http://arxiv.org/pdf/2407.10179v2)

**Authors**: Hao Fang, Jiawei Kong, Bin Chen, Tao Dai, Hao Wu, Shu-Tao Xia

**Abstract**: Transferable targeted adversarial attacks aim to mislead models into outputting adversary-specified predictions in black-box scenarios. Recent studies have introduced \textit{single-target} generative attacks that train a generator for each target class to generate highly transferable perturbations, resulting in substantial computational overhead when handling multiple classes. \textit{Multi-target} attacks address this by training only one class-conditional generator for multiple classes. However, the generator simply uses class labels as conditions, failing to leverage the rich semantic information of the target class. To this end, we design a \textbf{C}LIP-guided \textbf{G}enerative \textbf{N}etwork with \textbf{C}ross-attention modules (CGNC) to enhance multi-target attacks by incorporating textual knowledge of CLIP into the generator. Extensive experiments demonstrate that CGNC yields significant improvements over previous multi-target generative attacks, e.g., a 21.46\% improvement in success rate from ResNet-152 to DenseNet-121. Moreover, we propose a masked fine-tuning mechanism to further strengthen our method in attacking a single class, which surpasses existing single-target methods.

摘要: 可转移的目标对抗性攻击旨在误导模型，使其在黑盒场景中输出对手指定的预测。最近的研究引入了生成性攻击，这种攻击为每个目标类训练一个生成器来生成高度可传递的扰动，导致在处理多个类时产生大量的计算开销。\textit{多目标}攻击通过仅训练多个类的一个类条件生成器来解决此问题。然而，生成器简单地使用类标签作为条件，没有利用目标类的丰富语义信息。为此，我们设计了一个唇形引导的生成模块(CGNC)，通过在生成器中加入剪辑文本知识来增强多目标攻击。大量的实验表明，CGNC比以前的多目标生成性攻击有显著的改进，例如，成功率从ResNet-152提高到DenseNet-121，提高了21.46%.此外，我们还提出了一种屏蔽微调机制，进一步加强了我们的攻击单一类的方法，超越了现有的单目标攻击方法。



## **18. TAPI: Towards Target-Specific and Adversarial Prompt Injection against Code LLMs**

TAPI：针对代码LLM的目标特定和对抗性即时注入 cs.CR

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.09164v3) [paper-pdf](http://arxiv.org/pdf/2407.09164v3)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully used to simplify and facilitate code programming. With these tools, developers can easily generate desired complete functional codes based on incomplete code and natural language prompts. However, a few pioneering works revealed that these Code LLMs are also vulnerable, e.g., against backdoor and adversarial attacks. The former could induce LLMs to respond to triggers to insert malicious code snippets by poisoning the training data or model parameters, while the latter can craft malicious adversarial input codes to reduce the quality of generated codes. However, both attack methods have underlying limitations: backdoor attacks rely on controlling the model training process, while adversarial attacks struggle with fulfilling specific malicious purposes.   To inherit the advantages of both backdoor and adversarial attacks, this paper proposes a new attack paradigm, i.e., target-specific and adversarial prompt injection (TAPI), against Code LLMs. TAPI generates unreadable comments containing information about malicious instructions and hides them as triggers in the external source code. When users exploit Code LLMs to complete codes containing the trigger, the models will generate attacker-specified malicious code snippets at specific locations. We evaluate our TAPI attack on four representative LLMs under three representative malicious objectives and seven cases. The results show that our method is highly threatening (achieving an attack success rate of up to 98.3%) and stealthy (saving an average of 53.1% of tokens in the trigger design). In particular, we successfully attack some famous deployed code completion integrated applications, including CodeGeex and Github Copilot. This further confirms the realistic threat of our attack.

摘要: 最近，面向代码的大型语言模型(Code LLM)已被广泛并成功地用于简化和促进代码编程。使用这些工具，开发人员可以根据不完整的代码和自然语言提示轻松生成所需的完整功能代码。然而，一些开创性的工作表明，这些代码LLM也容易受到攻击，例如，抵御后门和对手攻击。前者可以通过毒化训练数据或模型参数来诱导LLMS响应插入恶意代码片段的触发器，而后者可以手工创建恶意输入代码来降低生成代码的质量。然而，这两种攻击方法都有潜在的局限性：后门攻击依赖于控制模型训练过程，而对抗性攻击则难以实现特定的恶意目的。为了继承后门攻击和对抗性攻击的优点，提出了一种新的针对Code LLMS的攻击范式，即目标特定和对抗性提示注入(TAPI)。TAPI生成不可读的注释，其中包含有关恶意指令的信息，并将它们作为触发器隐藏在外部源代码中。当用户利用Code LLMS来完成包含触发器的代码时，模型将在特定位置生成攻击者指定的恶意代码片段。我们在三个典型的恶意目标和七个案例下评估了我们的TAPI攻击对四个有代表性的LLM的攻击。结果表明，该方法具有很高的威胁性(攻击成功率高达98.3%)和隐蔽性(在触发器设计中平均节省53.1%的令牌)。特别是，我们成功地攻击了一些著名的部署代码完成集成应用程序，包括CodeGeex和Github Copilot。这进一步证实了我们攻击的现实威胁。



## **19. Imposter.AI: Adversarial Attacks with Hidden Intentions towards Aligned Large Language Models**

冒名顶替。AI：针对对齐大型语言模型的具有隐藏意图的对抗攻击 cs.CL

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15399v1) [paper-pdf](http://arxiv.org/pdf/2407.15399v1)

**Authors**: Xiao Liu, Liangzhi Li, Tong Xiang, Fuying Ye, Lu Wei, Wangyue Li, Noa Garcia

**Abstract**: With the development of large language models (LLMs) like ChatGPT, both their vast applications and potential vulnerabilities have come to the forefront. While developers have integrated multiple safety mechanisms to mitigate their misuse, a risk remains, particularly when models encounter adversarial inputs. This study unveils an attack mechanism that capitalizes on human conversation strategies to extract harmful information from LLMs. We delineate three pivotal strategies: (i) decomposing malicious questions into seemingly innocent sub-questions; (ii) rewriting overtly malicious questions into more covert, benign-sounding ones; (iii) enhancing the harmfulness of responses by prompting models for illustrative examples. Unlike conventional methods that target explicit malicious responses, our approach delves deeper into the nature of the information provided in responses. Through our experiments conducted on GPT-3.5-turbo, GPT-4, and Llama2, our method has demonstrated a marked efficacy compared to conventional attack methods. In summary, this work introduces a novel attack method that outperforms previous approaches, raising an important question: How to discern whether the ultimate intent in a dialogue is malicious?

摘要: 随着像ChatGPT这样的大型语言模型(LLM)的发展，它们的巨大应用和潜在的漏洞都已经浮出水面。虽然开发人员已经集成了多种安全机制来减少它们的滥用，但风险仍然存在，特别是当模型遇到敌对输入时。这项研究揭示了一种利用人类对话策略从LLMS中提取有害信息的攻击机制。我们描述了三个关键策略：(I)将恶意问题分解为看似无害的子问题；(Ii)将公开的恶意问题重写为更隐蔽、听起来更温和的问题；(Iii)通过提示示例模型来增强回答的危害性。与针对显式恶意响应的传统方法不同，我们的方法更深入地挖掘响应中提供的信息的性质。通过我们在GPT-3.5-Turbo、GPT-4和Llama2上的实验，我们的方法比传统的攻击方法表现出了显著的效果。总之，这项工作引入了一种新的攻击方法，其性能优于以前的方法，提出了一个重要的问题：如何识别对话中的最终意图是否为恶意的？



## **20. Towards Robust Vision Transformer via Masked Adaptive Ensemble**

通过掩蔽自适应集合迈向稳健的视觉Transformer cs.CV

9 pages

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15385v1) [paper-pdf](http://arxiv.org/pdf/2407.15385v1)

**Authors**: Fudong Lin, Jiadong Lou, Xu Yuan, Nian-Feng Tzeng

**Abstract**: Adversarial training (AT) can help improve the robustness of Vision Transformers (ViT) against adversarial attacks by intentionally injecting adversarial examples into the training data. However, this way of adversarial injection inevitably incurs standard accuracy degradation to some extent, thereby calling for a trade-off between standard accuracy and robustness. Besides, the prominent AT solutions are still vulnerable to adaptive attacks. To tackle such shortcomings, this paper proposes a novel ViT architecture, including a detector and a classifier bridged by our newly developed adaptive ensemble. Specifically, we empirically discover that detecting adversarial examples can benefit from the Guided Backpropagation technique. Driven by this discovery, a novel Multi-head Self-Attention (MSA) mechanism is introduced to enhance our detector to sniff adversarial examples. Then, a classifier with two encoders is employed for extracting visual representations respectively from clean images and adversarial examples, with our adaptive ensemble to adaptively adjust the proportion of visual representations from the two encoders for accurate classification. This design enables our ViT architecture to achieve a better trade-off between standard accuracy and robustness. Besides, our adaptive ensemble technique allows us to mask off a random subset of image patches within input data, boosting our ViT's robustness against adaptive attacks, while maintaining high standard accuracy. Experimental results exhibit that our ViT architecture, on CIFAR-10, achieves the best standard accuracy and adversarial robustness of 90.3% and 49.8%, respectively.

摘要: 对抗性训练(AT)通过有意地在训练数据中注入对抗性例子，有助于提高视觉转换器(VIT)对对抗性攻击的稳健性。然而，这种对抗性注入方式不可避免地会在一定程度上导致标准精度的下降，从而需要在标准精度和稳健性之间进行权衡。此外，突出的AT解决方案仍然容易受到适应性攻击。为了解决这些不足，本文提出了一种新颖的VIT体系结构，包括一个检测器和一个由我们新开发的自适应集成连接的分类器。具体地说，我们经验地发现，检测敌意例子可以受益于引导反向传播技术。在这一发现的推动下，引入了一种新的多头自我注意(MSA)机制来增强我们的检测器来嗅探敌意例子。然后，使用一个带有两个编码器的分类器分别从干净图像和对抗性样本中提取视觉表征，并通过自适应集成自适应地调整两个编码器的视觉表征比例以实现准确分类。这种设计使我们的VIT架构能够在标准准确性和健壮性之间实现更好的平衡。此外，我们的自适应集成技术允许我们屏蔽输入数据中的图像补丁的随机子集，增强我们的VIT对自适应攻击的稳健性，同时保持高标准精度。实验结果表明，我们的VIT架构在CIFAR-10上获得了90.3%的标准准确率和49.8%的对手健壮性。



## **21. Improving the Accuracy-Robustness Trade-Off of Classifiers via Adaptive Smoothing**

通过自适应平滑改善分类器的准确性与鲁棒性权衡 cs.LG

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2301.12554v5) [paper-pdf](http://arxiv.org/pdf/2301.12554v5)

**Authors**: Yatong Bai, Brendon G. Anderson, Aerin Kim, Somayeh Sojoudi

**Abstract**: While prior research has proposed a plethora of methods that build neural classifiers robust against adversarial robustness, practitioners are still reluctant to adopt them due to their unacceptably severe clean accuracy penalties. This paper significantly alleviates this accuracy-robustness trade-off by mixing the output probabilities of a standard classifier and a robust classifier, where the standard network is optimized for clean accuracy and is not robust in general. We show that the robust base classifier's confidence difference for correct and incorrect examples is the key to this improvement. In addition to providing intuitions and empirical evidence, we theoretically certify the robustness of the mixed classifier under realistic assumptions. Furthermore, we adapt an adversarial input detector into a mixing network that adaptively adjusts the mixture of the two base models, further reducing the accuracy penalty of achieving robustness. The proposed flexible method, termed "adaptive smoothing", can work in conjunction with existing or even future methods that improve clean accuracy, robustness, or adversary detection. Our empirical evaluation considers strong attack methods, including AutoAttack and adaptive attack. On the CIFAR-100 dataset, our method achieves an 85.21% clean accuracy while maintaining a 38.72% $\ell_\infty$-AutoAttacked ($\epsilon = 8/255$) accuracy, becoming the second most robust method on the RobustBench CIFAR-100 benchmark as of submission, while improving the clean accuracy by ten percentage points compared with all listed models. The code that implements our method is available at https://github.com/Bai-YT/AdaptiveSmoothing.

摘要: 虽然先前的研究已经提出了太多的方法来构建稳健的神经分类器来对抗对手的健壮性，但实践者仍然不愿采用它们，因为它们具有不可接受的严重的干净准确性惩罚。本文通过混合标准分类器和稳健分类器的输出概率显著缓解了这种精度与稳健性的权衡，其中标准网络针对干净的精度进行了优化，而通常不是稳健的。研究表明，稳健的基分类器对正确样本和错误样本的置信度差异是这一改进的关键。除了提供直觉和经验证据外，我们还从理论上证明了混合分类器在现实假设下的稳健性。此外，我们将对抗性输入检测器引入混合网络，该混合网络自适应地调整两个基本模型的混合，从而进一步降低了实现稳健性的精度损失。这一灵活的方法被称为“自适应平滑”，可以与现有甚至未来的方法结合使用，以提高干净的准确性、健壮性或敌手检测。我们的经验评估考虑了强攻击方法，包括AutoAttack和自适应攻击。在CIFAR-100数据集上，我们的方法实现了85.21%的清洁准确率，同时保持了38.72%的$\ELL_\INFTY$-AutoAttaced($\epsilon=8/255$)精度，成为截至提交时在RobustBuchCIFAR-100基准上第二健壮的方法，同时与所有列出的模型相比，清洁准确率提高了10个百分点。实现我们方法的代码可以在https://github.com/Bai-YT/AdaptiveSmoothing.上找到



## **22. Exploring Transferability of Multimodal Adversarial Samples for Vision-Language Pre-training Models with Contrastive Learning**

利用对比学习探索视觉语言预训练模型多模式对抗样本的可移植性 cs.MM

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2308.12636v3) [paper-pdf](http://arxiv.org/pdf/2308.12636v3)

**Authors**: Youze Wang, Wenbo Hu, Yinpeng Dong, Hanwang Zhang, Hang Su, Richang Hong

**Abstract**: The integration of visual and textual data in Vision-Language Pre-training (VLP) models is crucial for enhancing vision-language understanding. However, the adversarial robustness of these models, especially in the alignment of image-text features, has not yet been sufficiently explored. In this paper, we introduce a novel gradient-based multimodal adversarial attack method, underpinned by contrastive learning, to improve the transferability of multimodal adversarial samples in VLP models. This method concurrently generates adversarial texts and images within imperceptive perturbation, employing both image-text and intra-modal contrastive loss. We evaluate the effectiveness of our approach on image-text retrieval and visual entailment tasks, using publicly available datasets in a black-box setting. Extensive experiments indicate a significant advancement over existing single-modal transfer-based adversarial attack methods and current multimodal adversarial attack approaches.

摘要: 视觉语言预训练（VLP）模型中视觉和文本数据的集成对于增强视觉语言理解至关重要。然而，这些模型的对抗稳健性，尤其是在图像-文本特征的对齐方面，尚未得到充分的探索。本文引入了一种以对比学习为基础的新型基于梯度的多模式对抗攻击方法，以提高VLP模型中多模式对抗样本的可移植性。该方法在不可感知的扰动中同时生成对抗性文本和图像，同时采用图像-文本和模式内对比损失。我们在黑匣子环境中使用公开可用的数据集来评估我们的方法在图像文本检索和视觉蕴含任务方面的有效性。大量实验表明，与现有的基于单模式转移的对抗攻击方法和当前的多模式对抗攻击方法相比，有了重大进步。



## **23. When Do Universal Image Jailbreaks Transfer Between Vision-Language Models?**

普遍形象越狱何时在视觉语言模型之间转移？ cs.CL

**SubmitDate**: 2024-07-21    [abs](http://arxiv.org/abs/2407.15211v1) [paper-pdf](http://arxiv.org/pdf/2407.15211v1)

**Authors**: Rylan Schaeffer, Dan Valentine, Luke Bailey, James Chua, Cristóbal Eyzaguirre, Zane Durante, Joe Benton, Brando Miranda, Henry Sleight, John Hughes, Rajashree Agrawal, Mrinank Sharma, Scott Emmons, Sanmi Koyejo, Ethan Perez

**Abstract**: The integration of new modalities into frontier AI systems offers exciting capabilities, but also increases the possibility such systems can be adversarially manipulated in undesirable ways. In this work, we focus on a popular class of vision-language models (VLMs) that generate text outputs conditioned on visual and textual inputs. We conducted a large-scale empirical study to assess the transferability of gradient-based universal image "jailbreaks" using a diverse set of over 40 open-parameter VLMs, including 18 new VLMs that we publicly release. Overall, we find that transferable gradient-based image jailbreaks are extremely difficult to obtain. When an image jailbreak is optimized against a single VLM or against an ensemble of VLMs, the jailbreak successfully jailbreaks the attacked VLM(s), but exhibits little-to-no transfer to any other VLMs; transfer is not affected by whether the attacked and target VLMs possess matching vision backbones or language models, whether the language model underwent instruction-following and/or safety-alignment training, or many other factors. Only two settings display partially successful transfer: between identically-pretrained and identically-initialized VLMs with slightly different VLM training data, and between different training checkpoints of a single VLM. Leveraging these results, we then demonstrate that transfer can be significantly improved against a specific target VLM by attacking larger ensembles of "highly-similar" VLMs. These results stand in stark contrast to existing evidence of universal and transferable text jailbreaks against language models and transferable adversarial attacks against image classifiers, suggesting that VLMs may be more robust to gradient-based transfer attacks.

摘要: 将新的模式集成到前沿人工智能系统中提供了令人兴奋的能力，但也增加了此类系统被以不受欢迎的方式进行相反操作的可能性。在这项工作中，我们专注于一类流行的视觉语言模型(VLM)，它们生成以视觉和文本输入为条件的文本输出。我们进行了一项大规模的实证研究，以评估基于梯度的通用图像“越狱”的可转移性，使用了一组超过40个开放参数的VLM，其中包括我们公开发布的18个新的VLM。总体而言，我们发现基于梯度的可转移越狱图像非常难以获得。当针对单个VLM或一组VLM优化图像越狱时，越狱成功地越狱了被攻击的VLM(S)，但很少或根本不转移到任何其他VLM；转移不受攻击和目标VLM是否具有匹配的视觉主干或语言模型、语言模型是否经过指令遵循和/或安全对齐培训或许多其他因素的影响。只有两个设置显示部分成功的传输：在具有略微不同的VLM训练数据的相同预训练和相同初始化的VLM之间，以及在单个VLM的不同训练检查点之间。利用这些结果，我们随后证明了针对特定目标VLM的传输可以通过攻击更大的“高度相似的”VLM集合来显著改进。这些结果与针对语言模型的普遍和可传输的文本越狱以及针对图像分类器的可传输的对抗性攻击的现有证据形成了鲜明对比，这表明VLM可能对基于梯度的传输攻击更健壮。



## **24. SNNGX: Securing Spiking Neural Networks with Genetic XOR Encryption on RRAM-based Neuromorphic Accelerator**

SNNGX：在基于RAM的神经形态加速器上通过遗传异或加密保护尖峰神经网络 cs.CR

International Conference on Computer-Aided Design 2024

**SubmitDate**: 2024-07-21    [abs](http://arxiv.org/abs/2407.15152v1) [paper-pdf](http://arxiv.org/pdf/2407.15152v1)

**Authors**: Kwunhang Wong, Songqi Wang, Wei Huang, Xinyuan Zhang, Yangu He, Karl M. H. Lai, Yuzhong Jiao, Ning Lin, Xiaojuan Qi, Xiaoming Chen, Zhongrui Wang

**Abstract**: Biologically plausible Spiking Neural Networks (SNNs), characterized by spike sparsity, are growing tremendous attention over intellectual edge devices and critical bio-medical applications as compared to artificial neural networks (ANNs). However, there is a considerable risk from malicious attempts to extract white-box information (i.e., weights) from SNNs, as attackers could exploit well-trained SNNs for profit and white-box adversarial concerns. There is a dire need for intellectual property (IP) protective measures. In this paper, we present a novel secure software-hardware co-designed RRAM-based neuromorphic accelerator for protecting the IP of SNNs. Software-wise, we design a tailored genetic algorithm with classic XOR encryption to target the least number of weights that need encryption. From a hardware perspective, we develop a low-energy decryption module, meticulously designed to provide zero decryption latency. Extensive results from various datasets, including NMNIST, DVSGesture, EEGMMIDB, Braille Letter, and SHD, demonstrate that our proposed method effectively secures SNNs by encrypting a minimal fraction of stealthy weights, only 0.00005% to 0.016% weight bits. Additionally, it achieves a substantial reduction in energy consumption, ranging from x59 to x6780, and significantly lowers decryption latency, ranging from x175 to x4250. Moreover, our method requires as little as one sample per class in dataset for encryption and addresses hessian/gradient-based search insensitive problems. This strategy offers a highly efficient and flexible solution for securing SNNs in diverse applications.

摘要: 与人工神经网络(ANN)相比，生物学上看似合理的尖峰神经网络(SNN)在智能边缘设备和关键的生物医学应用方面受到了极大的关注。然而，恶意尝试从SNN中提取白盒信息(即权重)存在相当大的风险，因为攻击者可以利用训练有素的SNN来获取利润和白盒对手的担忧。迫切需要知识产权(IP)保护措施。在本文中，我们提出了一种基于RRAM的安全软硬件联合设计的神经形态加速器来保护SNN的IP。在软件方面，我们设计了一种定制的遗传算法，采用经典的XOR加密，以达到需要加密的权重最少的目标。从硬件的角度，我们开发了一个低能耗的解密模块，精心设计，以提供零解密延迟。在NMNIST、DVSGesture、EEGMMIDB、盲文字母和SHD等不同的数据集上的广泛结果表明，我们提出的方法通过加密最小部分的隐蔽权重来有效地保护SNN，仅0.00005%到0.016%的权重比特。此外，它实现了从x59到x6780的能耗的大幅降低，并显著降低了从x175到x4250的解密延迟。此外，我们的方法只需要数据集中每个类一个样本进行加密，并解决了基于黑斯/梯度的搜索不敏感问题。该策略为保护不同应用中的SNN提供了一种高效、灵活的解决方案。



## **25. Sim-CLIP: Unsupervised Siamese Adversarial Fine-Tuning for Robust and Semantically-Rich Vision-Language Models**

Sim-CLIP：针对稳健且语义丰富的视觉语言模型的无监督Siamese对抗微调 cs.CV

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.14971v1) [paper-pdf](http://arxiv.org/pdf/2407.14971v1)

**Authors**: Md Zarif Hossain, Ahmed Imteaj

**Abstract**: Vision-language models (VLMs) have achieved significant strides in recent times specially in multimodal tasks, yet they remain susceptible to adversarial attacks on their vision components. To address this, we propose Sim-CLIP, an unsupervised adversarial fine-tuning method that enhances the robustness of the widely-used CLIP vision encoder against such attacks while maintaining semantic richness and specificity. By employing a Siamese architecture with cosine similarity loss, Sim-CLIP learns semantically meaningful and attack-resilient visual representations without requiring large batch sizes or momentum encoders. Our results demonstrate that VLMs enhanced with Sim-CLIP's fine-tuned CLIP encoder exhibit significantly enhanced robustness against adversarial attacks, while preserving semantic meaning of the perturbed images. Notably, Sim-CLIP does not require additional training or fine-tuning of the VLM itself; replacing the original vision encoder with our fine-tuned Sim-CLIP suffices to provide robustness. This work underscores the significance of reinforcing foundational models like CLIP to safeguard the reliability of downstream VLM applications, paving the way for more secure and effective multimodal systems.

摘要: 视觉语言模型近年来取得了长足的进步，特别是在多通道任务中，但它们仍然容易受到视觉部分的敌意攻击。为了解决这一问题，我们提出了SIM-CLIP，这是一种无监督的对抗性微调方法，它在保持语义丰富和特异性的同时，增强了广泛使用的CLIP视觉编码器对此类攻击的健壮性。通过采用具有余弦相似性损失的暹罗体系结构，Sim-Clip无需大批量或动量编码器即可学习语义上有意义的、可抵抗攻击的视觉表示。结果表明，通过Sim-Clip的精细调整的CLIP编码器增强的VLM在保持扰动图像语义的同时，显著增强了对对手攻击的稳健性。值得注意的是，SIM-Clip不需要对VLM本身进行额外的培训或微调；用我们经过微调的SIM-Clip替换原来的视觉编码器就足以提供健壮性。这项工作强调了加强像CLIP这样的基础模型对保障下游VLM应用的可靠性的重要性，为更安全和有效的多式联运系统铺平了道路。



## **26. Characterizing Encrypted Application Traffic through Cellular Radio Interface Protocol**

通过蜂窝无线电接口协议描述加密应用流量 cs.NI

9 pages, 8 figures, 2 tables. This paper has been accepted for  publication by the 21st IEEE International Conference on Mobile Ad-Hoc and  Smart Systems (MASS 2024)

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.07361v2) [paper-pdf](http://arxiv.org/pdf/2407.07361v2)

**Authors**: Md Ruman Islam, Raja Hasnain Anwar, Spyridon Mastorakis, Muhammad Taqi Raza

**Abstract**: Modern applications are end-to-end encrypted to prevent data from being read or secretly modified. 5G tech nology provides ubiquitous access to these applications without compromising the application-specific performance and latency goals. In this paper, we empirically demonstrate that 5G radio communication becomes the side channel to precisely infer the user's applications in real-time. The key idea lies in observing the 5G physical and MAC layer interactions over time that reveal the application's behavior. The MAC layer receives the data from the application and requests the network to assign the radio resource blocks. The network assigns the radio resources as per application requirements, such as priority, Quality of Service (QoS) needs, amount of data to be transmitted, and buffer size. The adversary can passively observe the radio resources to fingerprint the applications. We empirically demonstrate this attack by considering four different categories of applications: online shopping, voice/video conferencing, video streaming, and Over-The-Top (OTT) media platforms. Finally, we have also demonstrated that an attacker can differentiate various types of applications in real-time within each category.

摘要: 现代应用程序是端到端加密的，以防止数据被读取或秘密修改。5G技术提供了对这些应用的无处不在的访问，而不会影响特定于应用的性能和延迟目标。在本文中，我们实证地论证了5G无线通信成为实时准确推断用户应用的辅助通道。关键思想在于观察5G物理层和MAC层随时间的交互，以揭示应用的行为。MAC层从应用程序接收数据，并请求网络分配无线电资源块。网络根据诸如优先级、服务质量(Qos)需求、要传输的数据量和缓冲区大小等应用需求来分配无线电资源。敌手可以被动地观察无线电资源来识别应用程序。我们考虑了四种不同类别的应用程序：在线购物、语音/视频会议、视频流和Over-the-Top(OTT)媒体平台，对这一攻击进行了实证演示。最后，我们还演示了攻击者可以在每个类别中实时区分各种类型的应用程序。



## **27. Adversarial Sparse Teacher: Defense Against Distillation-Based Model Stealing Attacks Using Adversarial Examples**

对抗性稀疏教师：使用对抗性示例防御基于蒸馏的模型窃取攻击 cs.LG

14 pages, 3 figures, 11 tables

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2403.05181v2) [paper-pdf](http://arxiv.org/pdf/2403.05181v2)

**Authors**: Eda Yilmaz, Hacer Yalim Keles

**Abstract**: We introduce Adversarial Sparse Teacher (AST), a robust defense method against distillation-based model stealing attacks. Our approach trains a teacher model using adversarial examples to produce sparse logit responses and increase the entropy of the output distribution. Typically, a model generates a peak in its output corresponding to its prediction. By leveraging adversarial examples, AST modifies the teacher model's original response, embedding a few altered logits into the output while keeping the primary response slightly higher. Concurrently, all remaining logits are elevated to further increase the output distribution's entropy. All these complex manipulations are performed using an optimization function with our proposed Exponential Predictive Divergence (EPD) loss function. EPD allows us to maintain higher entropy levels compared to traditional KL divergence, effectively confusing attackers. Experiments on CIFAR-10 and CIFAR-100 datasets demonstrate that AST outperforms state-of-the-art methods, providing effective defense against model stealing while preserving high accuracy. The source codes will be made publicly available here soon.

摘要: 我们引入了对抗性稀疏教师(AST)，这是一种针对基于蒸馏的模型窃取攻击的稳健防御方法。我们的方法使用对抗性例子来训练教师模型，以产生稀疏的Logit响应并增加输出分布的熵。通常，模型在其输出中生成与其预测相对应的峰值。通过利用敌意的例子，AST修改了教师模型的原始响应，在输出中嵌入了一些更改后的日志，同时保持了略高的主要响应。同时，所有剩余的对数都被提升，以进一步增加输出分布的熵。所有这些复杂的操作都是使用我们提出的指数预测发散(EPD)损失函数的优化函数来执行的。与传统的KL发散相比，EPD允许我们保持更高的熵级，有效地迷惑了攻击者。在CIFAR-10和CIFAR-100数据集上的实验表明，AST的性能优于最先进的方法，在保持高准确率的同时提供了对模型窃取的有效防御。源代码很快就会在这里公开。



## **28. Flatness-aware Sequential Learning Generates Resilient Backdoors**

平坦性感知顺序学习产生弹性后门 cs.LG

ECCV 2024

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.14738v1) [paper-pdf](http://arxiv.org/pdf/2407.14738v1)

**Authors**: Hoang Pham, The-Anh Ta, Anh Tran, Khoa D. Doan

**Abstract**: Recently, backdoor attacks have become an emerging threat to the security of machine learning models. From the adversary's perspective, the implanted backdoors should be resistant to defensive algorithms, but some recently proposed fine-tuning defenses can remove these backdoors with notable efficacy. This is mainly due to the catastrophic forgetting (CF) property of deep neural networks. This paper counters CF of backdoors by leveraging continual learning (CL) techniques. We begin by investigating the connectivity between a backdoored and fine-tuned model in the loss landscape. Our analysis confirms that fine-tuning defenses, especially the more advanced ones, can easily push a poisoned model out of the backdoor regions, making it forget all about the backdoors. Based on this finding, we re-formulate backdoor training through the lens of CL and propose a novel framework, named Sequential Backdoor Learning (SBL), that can generate resilient backdoors. This framework separates the backdoor poisoning process into two tasks: the first task learns a backdoored model, while the second task, based on the CL principles, moves it to a backdoored region resistant to fine-tuning. We additionally propose to seek flatter backdoor regions via a sharpness-aware minimizer in the framework, further strengthening the durability of the implanted backdoor. Finally, we demonstrate the effectiveness of our method through extensive empirical experiments on several benchmark datasets in the backdoor domain. The source code is available at https://github.com/mail-research/SBL-resilient-backdoors

摘要: 最近，后门攻击已经成为对机器学习模型安全的新兴威胁。从对手的角度来看，植入的后门应该可以抵抗防御算法，但最近提出的一些微调防御措施可以删除这些后门，效果显著。这主要是由于深层神经网络的灾难性遗忘特性造成的。本文通过利用持续学习(CL)技术来对抗后门的CF。我们首先调查亏损场景中回溯模型和微调模型之间的连通性。我们的分析证实，微调防御，特别是更先进的防御，可以很容易地将有毒的模型推出后门区域，让它忘记所有后门。基于这一发现，我们通过CL的镜头重新定义了后门训练，并提出了一个新的框架，称为顺序后门学习(SBL)，它可以产生弹性后门。该框架将后门中毒过程分为两个任务：第一个任务学习后门模型，第二个任务基于CL原则，将其移动到抵抗微调的后门区域。我们还建议通过框架中的锐度感知最小化来寻找更平坦的后门区域，进一步增强植入的后门的耐用性。最后，我们通过在后门领域的几个基准数据集上进行广泛的实证实验，证明了该方法的有效性。源代码可在https://github.com/mail-research/SBL-resilient-backdoors上找到



## **29. Bag of Tricks to Boost Adversarial Transferability**

提高对抗性转让能力的一袋技巧 cs.CV

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2401.08734v2) [paper-pdf](http://arxiv.org/pdf/2401.08734v2)

**Authors**: Zeliang Zhang, Wei Yao, Xiaosen Wang

**Abstract**: Deep neural networks are widely known to be vulnerable to adversarial examples. However, vanilla adversarial examples generated under the white-box setting often exhibit low transferability across different models. Since adversarial transferability poses more severe threats to practical applications, various approaches have been proposed for better transferability, including gradient-based, input transformation-based, and model-related attacks, \etc. In this work, we find that several tiny changes in the existing adversarial attacks can significantly affect the attack performance, \eg, the number of iterations and step size. Based on careful studies of existing adversarial attacks, we propose a bag of tricks to enhance adversarial transferability, including momentum initialization, scheduled step size, dual example, spectral-based input transformation, and several ensemble strategies. Extensive experiments on the ImageNet dataset validate the high effectiveness of our proposed tricks and show that combining them can further boost adversarial transferability. Our work provides practical insights and techniques to enhance adversarial transferability, and offers guidance to improve the attack performance on the real-world application through simple adjustments.

摘要: 众所周知，深度神经网络很容易受到敌意例子的攻击。然而，在白盒设置下产生的普通对抗性例子往往表现出在不同模型之间的低可转移性。由于对抗性可转移性对实际应用构成了更严重的威胁，人们提出了各种方法来提高可转移性，包括基于梯度的攻击、基于输入变换的攻击和与模型相关的攻击等。在本工作中，我们发现现有对抗性攻击中的几个微小变化会显著影响攻击性能，例如迭代次数和步长。在仔细研究现有对抗性攻击的基础上，提出了一系列增强对抗性可转移性的策略，包括动量初始化、调度步长、对偶例、基于谱的输入变换和几种集成策略。在ImageNet数据集上的大量实验验证了我们提出的技巧的高效性，并表明将它们结合起来可以进一步提高对手的可转移性。我们的工作为增强对手的可转移性提供了实用的见解和技术，并为通过简单的调整提高对现实世界应用的攻击性能提供了指导。



## **30. Augment then Smooth: Reconciling Differential Privacy with Certified Robustness**

增强然后平滑：通过认证的稳健性来实现差异隐私 cs.LG

29 pages, 19 figures. Accepted at TMLR in 2024. Link:  https://openreview.net/pdf?id=YN0IcnXqsr

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2306.08656v2) [paper-pdf](http://arxiv.org/pdf/2306.08656v2)

**Authors**: Jiapeng Wu, Atiyeh Ashari Ghomi, David Glukhov, Jesse C. Cresswell, Franziska Boenisch, Nicolas Papernot

**Abstract**: Machine learning models are susceptible to a variety of attacks that can erode trust, including attacks against the privacy of training data, and adversarial examples that jeopardize model accuracy. Differential privacy and certified robustness are effective frameworks for combating these two threats respectively, as they each provide future-proof guarantees. However, we show that standard differentially private model training is insufficient for providing strong certified robustness guarantees. Indeed, combining differential privacy and certified robustness in a single system is non-trivial, leading previous works to introduce complex training schemes that lack flexibility. In this work, we present DP-CERT, a simple and effective method that achieves both privacy and robustness guarantees simultaneously by integrating randomized smoothing into standard differentially private model training. Compared to the leading prior work, DP-CERT gives up to a 2.5% increase in certified accuracy for the same differential privacy guarantee on CIFAR10. Through in-depth persample metric analysis, we find that larger certifiable radii correlate with smaller local Lipschitz constants, and show that DP-CERT effectively reduces Lipschitz constants compared to other differentially private training methods. The code is available at github.com/layer6ailabs/dp-cert.

摘要: 机器学习模型容易受到各种可能侵蚀信任的攻击，包括对训练数据隐私的攻击，以及危及模型准确性的敌意示例。差异隐私和认证的健壮性分别是对抗这两种威胁的有效框架，因为它们都提供了面向未来的保证。然而，我们表明，标准的差分私有模型训练不足以提供强大的认证稳健性保证。事实上，在单个系统中结合不同的隐私和经过认证的健壮性并不是一件容易的事情，这导致以前的工作引入了缺乏灵活性的复杂培训方案。在这项工作中，我们提出了一种简单而有效的方法DP-CERT，通过将随机化平滑与标准的差分私有模型训练相结合，同时实现了保密性和稳健性。与领先的以前的工作相比，DP-CERT在CIFAR10上提供相同的差异隐私保证的认证准确率最多提高了2.5%。通过深入的全样本度量分析，我们发现较大的可证明半径与较小的局部Lipschitz常数相关，并表明与其他差分私有训练方法相比，DP-CERT有效地降低了Lipschitz常数。代码可以在githorb.com/layer6ailabs/dp-cert上找到。



## **31. Human-Interpretable Adversarial Prompt Attack on Large Language Models with Situational Context**

具有情境上下文的大型语言模型的人类可解释对抗提示攻击 cs.CL

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.14644v1) [paper-pdf](http://arxiv.org/pdf/2407.14644v1)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous research on testing the vulnerabilities in Large Language Models (LLMs) using adversarial attacks has primarily focused on nonsensical prompt injections, which are easily detected upon manual or automated review (e.g., via byte entropy). However, the exploration of innocuous human-understandable malicious prompts augmented with adversarial injections remains limited. In this research, we explore converting a nonsensical suffix attack into a sensible prompt via a situation-driven contextual re-writing. This allows us to show suffix conversion without any gradients, using only LLMs to perform the attacks, and thus better understand the scope of possible risks. We combine an independent, meaningful adversarial insertion and situations derived from movies to check if this can trick an LLM. The situations are extracted from the IMDB dataset, and prompts are defined following a few-shot chain-of-thought prompting. Our approach demonstrates that a successful situation-driven attack can be executed on both open-source and proprietary LLMs. We find that across many LLMs, as few as 1 attempt produces an attack and that these attacks transfer between LLMs. The link to our code is available at \url{https://anonymous.4open.science/r/Situation-Driven-Adversarial-Attacks-7BB1/README.md}.

摘要: 之前关于使用对抗性攻击测试大型语言模型(LLM)中的漏洞的研究主要集中在无意义的提示注入上，这些注入很容易通过手动或自动审查(例如，通过字节熵)检测到。然而，通过恶意注入增强无害的人类可理解的恶意提示的探索仍然有限。在这项研究中，我们探索通过情景驱动的语境重写将无意义的后缀攻击转化为合理的提示。这使我们能够显示没有任何梯度的后缀转换，仅使用LLM来执行攻击，从而更好地了解可能风险的范围。我们结合了一个独立的、有意义的敌意插入和来自电影的情况来检查这是否可以欺骗LLM。情况是从IMDB数据集中提取的，提示是在几个镜头的思维链提示之后定义的。我们的方法表明，成功的情境驱动攻击可以在开源和专有LLM上执行。我们发现，在许多LLM中，只有1次尝试就会产生攻击，并且这些攻击会在LLM之间传输。有关我们代码的链接，请访问\url{https://anonymous.4open.science/r/Situation-Driven-Adversarial-Attacks-7BB1/README.md}.



## **32. Multi-Attribute Vision Transformers are Efficient and Robust Learners**

多属性视觉变形者是高效且稳健的学习者 cs.CV

Accepted at IEEE ICIP 2024. arXiv admin note: text overlap with  arXiv:2207.08677 by other authors

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2402.08070v2) [paper-pdf](http://arxiv.org/pdf/2402.08070v2)

**Authors**: Hanan Gani, Nada Saadi, Noor Hussein, Karthik Nandakumar

**Abstract**: Since their inception, Vision Transformers (ViTs) have emerged as a compelling alternative to Convolutional Neural Networks (CNNs) across a wide spectrum of tasks. ViTs exhibit notable characteristics, including global attention, resilience against occlusions, and adaptability to distribution shifts. One underexplored aspect of ViTs is their potential for multi-attribute learning, referring to their ability to simultaneously grasp multiple attribute-related tasks. In this paper, we delve into the multi-attribute learning capability of ViTs, presenting a straightforward yet effective strategy for training various attributes through a single ViT network as distinct tasks. We assess the resilience of multi-attribute ViTs against adversarial attacks and compare their performance against ViTs designed for single attributes. Moreover, we further evaluate the robustness of multi-attribute ViTs against a recent transformer based attack called Patch-Fool. Our empirical findings on the CelebA dataset provide validation for our assertion. Our code is available at https://github.com/hananshafi/MTL-ViT

摘要: 自诞生以来，视觉变压器(VITS)已经成为卷积神经网络(CNN)在广泛任务范围内的一种引人注目的替代方案。VITS表现出显著的特征，包括全球注意力、对闭塞的弹性和对分布变化的适应性。VITS的一个未被开发的方面是其多属性学习的潜力，指的是它们同时掌握多个与属性相关的任务的能力。在本文中，我们深入研究了VITS的多属性学习能力，提出了一种简单而有效的策略，通过单个VIT网络将各种属性作为不同的任务进行训练。我们评估了多属性VITS抵抗敌意攻击的能力，并与单属性VITS的性能进行了比较。此外，我们进一步评估了多属性VITS对最近一种称为Patch-Fool的基于变压器的攻击的健壮性。我们在CelebA数据集上的经验发现为我们的断言提供了验证。我们的代码可以在https://github.com/hananshafi/MTL-ViT上找到



## **33. SlowPerception: Physical-World Latency Attack against Visual Perception in Autonomous Driving**

慢感知：自动驾驶中对视觉感知的物理世界延迟攻击 cs.CV

This submission was made without all contributors' consent

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2406.05800v2) [paper-pdf](http://arxiv.org/pdf/2406.05800v2)

**Authors**: Chen Ma, Ningfei Wang, Zhengyu Zhao, Qi Alfred Chen, Chao Shen

**Abstract**: Autonomous Driving (AD) systems critically depend on visual perception for real-time object detection and multiple object tracking (MOT) to ensure safe driving. However, high latency in these visual perception components can lead to significant safety risks, such as vehicle collisions. While previous research has extensively explored latency attacks within the digital realm, translating these methods effectively to the physical world presents challenges. For instance, existing attacks rely on perturbations that are unrealistic or impractical for AD, such as adversarial perturbations affecting areas like the sky, or requiring large patches that obscure most of a camera's view, thus making them impossible to be conducted effectively in the real world.   In this paper, we introduce SlowPerception, the first physical-world latency attack against AD perception, via generating projector-based universal perturbations. SlowPerception strategically creates numerous phantom objects on various surfaces in the environment, significantly increasing the computational load of Non-Maximum Suppression (NMS) and MOT, thereby inducing substantial latency. Our SlowPerception achieves second-level latency in physical-world settings, with an average latency of 2.5 seconds across different AD perception systems, scenarios, and hardware configurations. This performance significantly outperforms existing state-of-the-art latency attacks. Additionally, we conduct AD system-level impact assessments, such as vehicle collisions, using industry-grade AD systems with production-grade AD simulators with a 97% average rate. We hope that our analyses can inspire further research in this critical domain, enhancing the robustness of AD systems against emerging vulnerabilities.

摘要: 自动驾驶(AD)系统在很大程度上依赖于视觉感知进行实时目标检测和多目标跟踪(MOT)来确保安全驾驶。然而，这些视觉感知组件的高延迟可能会导致重大安全风险，如车辆碰撞。虽然之前的研究已经广泛地探索了数字领域内的延迟攻击，但将这些方法有效地转换到物理世界是一项挑战。例如，现有的攻击依赖于对AD来说不现实或不切实际的扰动，例如影响天空等区域的对抗性扰动，或者需要遮挡大部分摄像机视野的大补丁，从而使它们不可能在现实世界中有效地进行。在本文中，我们通过产生基于投影仪的普遍扰动，引入了第一个针对AD感知的物理世界延迟攻击SlowPercept。SlowPercept战略性地在环境中的不同表面创建大量幻影对象，显著增加非最大抑制(NMS)和MOT的计算负荷，从而导致显著的延迟。我们的SlowPercept在物理世界设置中实现了二级延迟，跨不同AD感知系统、场景和硬件配置的平均延迟为2.5秒。这一性能大大超过了现有最先进的延迟攻击。此外，我们使用带有生产级AD模拟器的工业级AD系统进行AD系统级影响评估，例如车辆碰撞，平均比率为97%。我们希望我们的分析能够启发这一关键领域的进一步研究，增强AD系统对新出现的漏洞的健壮性。



## **34. Do Parameters Reveal More than Loss for Membership Inference?**

参数揭示的不仅仅是会员推断的损失吗？ cs.LG

Accepted at High-dimensional Learning Dynamics (HiLD) Workshop, ICML  2024

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2406.11544v2) [paper-pdf](http://arxiv.org/pdf/2406.11544v2)

**Authors**: Anshuman Suri, Xiao Zhang, David Evans

**Abstract**: Membership inference attacks aim to infer whether an individual record was used to train a model, serving as a key tool for disclosure auditing. While such evaluations are useful to demonstrate risk, they are computationally expensive and often make strong assumptions about potential adversaries' access to models and training environments, and thus do not provide very tight bounds on leakage from potential attacks. We show how prior claims around black-box access being sufficient for optimal membership inference do not hold for most useful settings such as stochastic gradient descent, and that optimal membership inference indeed requires white-box access. We validate our findings with a new white-box inference attack IHA (Inverse Hessian Attack) that explicitly uses model parameters by taking advantage of computing inverse-Hessian vector products. Our results show that both audits and adversaries may be able to benefit from access to model parameters, and we advocate for further research into white-box methods for membership privacy auditing.

摘要: 成员资格推断攻击旨在推断个人记录是否被用来训练模型，作为披露审计的关键工具。虽然这样的评估有助于显示风险，但它们的计算成本很高，而且通常会对潜在对手访问模型和训练环境做出强有力的假设，因此不会对潜在攻击的泄漏提供非常严格的限制。我们证明了关于黑箱访问的最优成员关系推理的先前声明如何不适用于大多数有用的设置，例如随机梯度下降，而最优成员关系推理确实需要白箱访问。我们使用一种新的白盒推理攻击IHA(逆向Hessian攻击)来验证我们的发现，该攻击通过计算逆向Hessian向量积来显式地使用模型参数。我们的结果表明，审计和对手都可以从访问模型参数中受益，我们主张进一步研究成员隐私审计的白盒方法。



## **35. Does Refusal Training in LLMs Generalize to the Past Tense?**

LLM中的拒绝培训是否适用于过去时态？ cs.CL

Update in v2: Claude-3.5 Sonnet and GPT-4o mini. We provide code and  jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.11969v2) [paper-pdf](http://arxiv.org/pdf/2407.11969v2)

**Authors**: Maksym Andriushchenko, Nicolas Flammarion

**Abstract**: Refusal training is widely used to prevent LLMs from generating harmful, undesirable, or illegal outputs. We reveal a curious generalization gap in the current refusal training approaches: simply reformulating a harmful request in the past tense (e.g., "How to make a Molotov cocktail?" to "How did people make a Molotov cocktail?") is often sufficient to jailbreak many state-of-the-art LLMs. We systematically evaluate this method on Llama-3 8B, Claude-3.5 Sonnet, GPT-3.5 Turbo, Gemma-2 9B, Phi-3-Mini, GPT-4o mini, GPT-4o, and R2D2 models using GPT-3.5 Turbo as a reformulation model. For example, the success rate of this simple attack on GPT-4o increases from 1% using direct requests to 88% using 20 past tense reformulation attempts on harmful requests from JailbreakBench with GPT-4 as a jailbreak judge. Interestingly, we also find that reformulations in the future tense are less effective, suggesting that refusal guardrails tend to consider past historical questions more benign than hypothetical future questions. Moreover, our experiments on fine-tuning GPT-3.5 Turbo show that defending against past reformulations is feasible when past tense examples are explicitly included in the fine-tuning data. Overall, our findings highlight that the widely used alignment techniques -- such as SFT, RLHF, and adversarial training -- employed to align the studied models can be brittle and do not always generalize as intended. We provide code and jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense.

摘要: 拒绝训练被广泛用于防止LLMS产生有害、不受欢迎或非法的输出。我们揭示了当前拒绝训练方法中一个奇怪的概括缺口：简单地用过去时重新表达一个有害的请求(例如，“如何调制燃烧鸡尾酒？”“人们是如何调制燃烧鸡尾酒的？”)通常足以越狱许多最先进的LLM。我们以GPT-3.5 Turbo为改写模型，对Llama-3 8B、Claude-3.5十四行诗、GPT-3.5 Turbo、Gema-2 9B、Phi-3-Mini、GPT-40 mini、GPT-40和R2D2模型进行了系统的评估。例如，对GPT-4o的这种简单攻击的成功率从使用直接请求的1%增加到使用20次过去时态重组尝试的88%，这些尝试使用GPT-4作为越狱法官的JailBreakB边的有害请求。有趣的是，我们还发现，未来时的重述没有那么有效，这表明拒绝障碍倾向于考虑过去的历史问题，而不是假设的未来问题。此外，我们在微调GPT-3.5Turbo上的实验表明，当微调数据中明确包含过去时态示例时，防御过去的重新公式是可行的。总体而言，我们的发现强调了广泛使用的对齐技术--如SFT、RLHF和对抗性训练--用于对所研究的模型进行对齐可能是脆弱的，并且并不总是像预期的那样泛化。我们在https://github.com/tml-epfl/llm-past-tense.上提供代码和越狱文物



## **36. Watermark Smoothing Attacks against Language Models**

针对语言模型的水印平滑攻击 cs.LG

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.14206v1) [paper-pdf](http://arxiv.org/pdf/2407.14206v1)

**Authors**: Hongyan Chang, Hamed Hassani, Reza Shokri

**Abstract**: Watermarking is a technique used to embed a hidden signal in the probability distribution of text generated by large language models (LLMs), enabling attribution of the text to the originating model. We introduce smoothing attacks and show that existing watermarking methods are not robust against minor modifications of text. An adversary can use weaker language models to smooth out the distribution perturbations caused by watermarks without significantly compromising the quality of the generated text. The modified text resulting from the smoothing attack remains close to the distribution of text that the original model (without watermark) would have produced. Our attack reveals a fundamental limitation of a wide range of watermarking techniques.

摘要: 水印是一种用于将隐藏信号嵌入大型语言模型（LLM）生成的文本的概率分布中的技术，从而将文本归因于原始模型。我们引入了平滑攻击，并表明现有的水印方法对文本的微小修改并不鲁棒。对手可以使用较弱的语言模型来平滑水印引起的分布扰动，而不会显着损害生成文本的质量。平滑攻击产生的修改文本仍然接近原始模型（没有水印）产生的文本分布。我们的攻击揭示了广泛水印技术的根本局限性。



## **37. MVPatch: More Vivid Patch for Adversarial Camouflaged Attacks on Object Detectors in the Physical World**

MVpatch：针对物理世界中物体检测器的对抗性伪装攻击的更多生动补丁 cs.CR

16 pages, 8 figures. This work has been submitted to the IEEE for  possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2312.17431v3) [paper-pdf](http://arxiv.org/pdf/2312.17431v3)

**Authors**: Zheng Zhou, Hongbo Zhao, Ju Liu, Qiaosheng Zhang, Liwei Geng, Shuchang Lyu, Wenquan Feng

**Abstract**: Recent studies have shown that Adversarial Patches (APs) can effectively manipulate object detection models. However, the conspicuous patterns often associated with these patches tend to attract human attention, posing a significant challenge. Existing research has primarily focused on enhancing attack efficacy in the physical domain while often neglecting the optimization of stealthiness and transferability. Furthermore, applying APs in real-world scenarios faces major challenges related to transferability, stealthiness, and practicality. To address these challenges, we introduce generalization theory into the context of APs, enabling our iterative process to simultaneously enhance transferability and refine visual correlation with realistic images. We propose a Dual-Perception-Based Framework (DPBF) to generate the More Vivid Patch (MVPatch), which enhances transferability, stealthiness, and practicality. The DPBF integrates two key components: the Model-Perception-Based Module (MPBM) and the Human-Perception-Based Module (HPBM), along with regularization terms. The MPBM employs ensemble strategy to reduce object confidence scores across multiple detectors, thereby improving AP transferability with robust theoretical support. Concurrently, the HPBM introduces a lightweight method for achieving visual similarity, creating natural and inconspicuous adversarial patches without relying on additional generative models. The regularization terms further enhance the practicality of the generated APs in the physical domain. Additionally, we introduce naturalness and transferability scores to provide an unbiased assessment of APs. Extensive experimental validation demonstrates that MVPatch achieves superior transferability and a natural appearance in both digital and physical domains, underscoring its effectiveness and stealthiness.

摘要: 最近的研究表明，对抗性补丁(AP)可以有效地操纵目标检测模型。然而，与这些斑块相关的明显图案往往会吸引人类的注意，构成了一个重大的挑战。现有的研究主要集中在提高物理领域的攻击效能，而往往忽略了隐蔽性和可转移性的优化。此外，在真实场景中应用AP面临着与可转移性、隐蔽性和实用性相关的重大挑战。为了应对这些挑战，我们将泛化理论引入到AP的上下文中，使我们的迭代过程能够同时增强可转移性和精细化与真实图像的视觉相关性。我们提出了一种基于双重感知的框架(DPBF)来生成更生动的补丁(MVPatch)，从而增强了可转移性、隐蔽性和实用性。DPBF集成了两个关键组件：基于模型感知的模块(MPBM)和基于人的感知的模块(HPBM)，以及正则化项。MPBM使用集成策略来降低多个检测器上的对象置信度分数，从而在稳健的理论支持下提高AP的可转移性。同时，HPBM引入了一种轻量级的方法来实现视觉相似性，创建自然的和不明显的对抗性补丁，而不依赖于额外的生成模型。正则化项进一步增强了所生成的AP在物理域中的实用性。此外，我们引入了自然性和可转移性分数，以提供对AP的公正评估。广泛的实验验证表明，MVPatch在数字和物理领域都实现了卓越的可转移性和自然外观，突显了其有效性和隐蔽性。



## **38. Adversarial Examples in the Physical World: A Survey**

物理世界中的对抗例子：调查 cs.CV

Adversarial examples, physical-world scenarios, attacks and defenses

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2311.01473v2) [paper-pdf](http://arxiv.org/pdf/2311.01473v2)

**Authors**: Jiakai Wang, Xianglong Liu, Jin Hu, Donghua Wang, Siyang Wu, Tingsong Jiang, Yuanfang Guo, Aishan Liu, Aishan Liu, Jiantao Zhou

**Abstract**: Deep neural networks (DNNs) have demonstrated high vulnerability to adversarial examples, raising broad security concerns about their applications. Besides the attacks in the digital world, the practical implications of adversarial examples in the physical world present significant challenges and safety concerns. However, current research on physical adversarial examples (PAEs) lacks a comprehensive understanding of their unique characteristics, leading to limited significance and understanding. In this paper, we address this gap by thoroughly examining the characteristics of PAEs within a practical workflow encompassing training, manufacturing, and re-sampling processes. By analyzing the links between physical adversarial attacks, we identify manufacturing and re-sampling as the primary sources of distinct attributes and particularities in PAEs. Leveraging this knowledge, we develop a comprehensive analysis and classification framework for PAEs based on their specific characteristics, covering over 100 studies on physical-world adversarial examples. Furthermore, we investigate defense strategies against PAEs and identify open challenges and opportunities for future research. We aim to provide a fresh, thorough, and systematic understanding of PAEs, thereby promoting the development of robust adversarial learning and its application in open-world scenarios to provide the community with a continuously updated list of physical world adversarial sample resources, including papers, code, \etc, within the proposed framework

摘要: 深度神经网络(DNN)对敌意例子表现出很高的脆弱性，引起了人们对其应用的广泛安全担忧。除了数字世界中的攻击，物理世界中敌意例子的实际影响也带来了重大挑战和安全问题。然而，目前对身体对抗例子(PAE)的研究缺乏对其独特特征的全面了解，导致其意义和理解有限。在本文中，我们通过彻底检查包括培训、制造和重新采样过程在内的实际工作流程中的PAE的特征来解决这一差距。通过分析物理对抗性攻击之间的联系，我们确定制造和重采样是PAE中不同属性和特殊性的主要来源。利用这些知识，我们根据PAE的具体特征开发了一个全面的分析和分类框架，涵盖了100多个物理世界对抗性例子的研究。此外，我们还研究了针对PAE的防御策略，并确定了未来研究的开放挑战和机会。我们的目标是提供一个新的，彻底的和系统的了解，从而促进发展强大的对抗性学习及其在开放世界情景中的应用，以在拟议的框架内为社区提供持续更新的物理世界对抗性样本资源列表，包括论文、代码等



## **39. Resilient Consensus Sustained Collaboratively**

通过协作维持弹性共识 cs.CR

15 pages, 7 figures

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2302.02325v5) [paper-pdf](http://arxiv.org/pdf/2302.02325v5)

**Authors**: Junchao Chen, Suyash Gupta, Alberto Sonnino, Lefteris Kokoris-Kogias, Mohammad Sadoghi

**Abstract**: Decentralized systems built around blockchain technology promise clients an immutable ledger. They add a transaction to the ledger after it undergoes consensus among the replicas that run a Proof-of-Stake (PoS) or Byzantine Fault-Tolerant (BFT) consensus protocol. Unfortunately, these protocols face a long-range attack where an adversary having access to the private keys of the replicas can rewrite the ledger. One solution is forcing each committed block from these protocols to undergo another consensus, Proof-of-Work(PoW) consensus; PoW protocol leads to wastage of computational resources as miners compete to solve complex puzzles. In this paper, we present the design of our Power-of-Collaboration (PoC) protocol, which guards existing PoS/BFT blockchains against long-range attacks and requires miners to collaborate rather than compete. PoC guarantees fairness and accountability and only marginally degrades the throughput of the underlying system.

摘要: 围绕区块链技术构建的去中心化系统向客户承诺一个不可改变的分类帐。在运行权益证明（PoS）或拜占庭过失容忍（BFT）共识协议的副本之间达成共识后，他们将交易添加到分类帐中。不幸的是，这些协议面临着远程攻击，其中可以访问副本的私有密钥的对手可以重写分类帐。一种解决方案是迫使这些协议中的每个已提交的块经历另一种共识，即工作量证明（PoW）共识;随着矿工竞相解决复杂难题，PoW协议导致计算资源的浪费。在本文中，我们介绍了协作力量（NPS）协议的设计，该协议保护现有PoS/BFT区块链免受远程攻击，并要求矿工进行合作而不是竞争。收件箱保证公平性和问责制，并且只略微降低底层系统的吞吐量。



## **40. Personalized Privacy Protection Mask Against Unauthorized Facial Recognition**

针对未经授权的面部识别的个性化隐私保护面具 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.13975v1) [paper-pdf](http://arxiv.org/pdf/2407.13975v1)

**Authors**: Ka-Ho Chow, Sihao Hu, Tiansheng Huang, Ling Liu

**Abstract**: Face recognition (FR) can be abused for privacy intrusion. Governments, private companies, or even individual attackers can collect facial images by web scraping to build an FR system identifying human faces without their consent. This paper introduces Chameleon, which learns to generate a user-centric personalized privacy protection mask, coined as P3-Mask, to protect facial images against unauthorized FR with three salient features. First, we use a cross-image optimization to generate one P3-Mask for each user instead of tailoring facial perturbation for each facial image of a user. It enables efficient and instant protection even for users with limited computing resources. Second, we incorporate a perceptibility optimization to preserve the visual quality of the protected facial images. Third, we strengthen the robustness of P3-Mask against unknown FR models by integrating focal diversity-optimized ensemble learning into the mask generation process. Extensive experiments on two benchmark datasets show that Chameleon outperforms three state-of-the-art methods with instant protection and minimal degradation of image quality. Furthermore, Chameleon enables cost-effective FR authorization using the P3-Mask as a personalized de-obfuscation key, and it demonstrates high resilience against adaptive adversaries.

摘要: 人脸识别(FR)可能被滥用来侵犯隐私。政府、私人公司，甚至个人攻击者都可以通过网络抓取来收集面部图像，以建立一个识别人脸的FR系统，而不需要他们的同意。本文介绍了变色龙，它学习生成一个以用户为中心的个性化隐私保护面具，称为P3-面具，以保护面部图像免受未经授权的FR，具有三个显著特征。首先，我们使用交叉图像优化为每个用户生成一个P3-面具，而不是为每个用户的人脸图像定制面部扰动。它可以为计算资源有限的用户提供高效、即时的保护。其次，我们加入了感知性优化，以保持受保护的面部图像的视觉质量。第三，通过在模板生成过程中集成焦点分集优化的集成学习来增强P3-MASK对未知FR模型的稳健性。在两个基准数据集上的广泛实验表明，变色龙的性能优于三种最先进的方法，具有即时保护和最小的图像质量退化。此外，变色龙能够使用P3-MASK作为个性化的去模糊密钥来实现经济高效的FR授权，并且它对自适应对手表现出高弹性。



## **41. Jailbreaking Black Box Large Language Models in Twenty Queries**

二十分钟内越狱黑匣子大型语言模型 cs.LG

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2310.08419v4) [paper-pdf](http://arxiv.org/pdf/2310.08419v4)

**Authors**: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong

**Abstract**: There is growing interest in ensuring that large language models (LLMs) align with human values. However, the alignment of such models is vulnerable to adversarial jailbreaks, which coax LLMs into overriding their safety guardrails. The identification of these vulnerabilities is therefore instrumental in understanding inherent weaknesses and preventing future misuse. To this end, we propose Prompt Automatic Iterative Refinement (PAIR), an algorithm that generates semantic jailbreaks with only black-box access to an LLM. PAIR -- which is inspired by social engineering attacks -- uses an attacker LLM to automatically generate jailbreaks for a separate targeted LLM without human intervention. In this way, the attacker LLM iteratively queries the target LLM to update and refine a candidate jailbreak. Empirically, PAIR often requires fewer than twenty queries to produce a jailbreak, which is orders of magnitude more efficient than existing algorithms. PAIR also achieves competitive jailbreaking success rates and transferability on open and closed-source LLMs, including GPT-3.5/4, Vicuna, and Gemini.

摘要: 人们对确保大型语言模型(LLM)与人类价值观保持一致的兴趣与日俱增。然而，这类模型的调整很容易受到对抗性越狱的影响，这会诱使低收入国家凌驾于他们的安全护栏之上。因此，确定这些漏洞有助于了解固有的弱点并防止今后的滥用。为此，我们提出了即时自动迭代求精(Pair)，这是一种仅通过黑盒访问LLM来生成语义越狱的算法。Pair受到社会工程攻击的启发，它使用攻击者LLM自动为单独的目标LLM生成越狱，而无需人工干预。通过这种方式，攻击者LLM迭代地查询目标LLM以更新和改进候选越狱。根据经验，Pair通常只需要不到20次查询就可以产生越狱，这比现有算法的效率高出几个数量级。Pair还在开放和封闭源代码的LLM上实现了具有竞争力的越狱成功率和可转移性，包括GPT-3.5/4、维库纳和双子座。



## **42. Black-Box Opinion Manipulation Attacks to Retrieval-Augmented Generation of Large Language Models**

对大型语言模型检索增强生成的黑匣子观点操纵攻击 cs.CL

10 pages, 3 figures, under review

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13757v1) [paper-pdf](http://arxiv.org/pdf/2407.13757v1)

**Authors**: Zhuo Chen, Jiawei Liu, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) is applied to solve hallucination problems and real-time constraints of large language models, but it also induces vulnerabilities against retrieval corruption attacks. Existing research mainly explores the unreliability of RAG in white-box and closed-domain QA tasks. In this paper, we aim to reveal the vulnerabilities of Retrieval-Enhanced Generative (RAG) models when faced with black-box attacks for opinion manipulation. We explore the impact of such attacks on user cognition and decision-making, providing new insight to enhance the reliability and security of RAG models. We manipulate the ranking results of the retrieval model in RAG with instruction and use these results as data to train a surrogate model. By employing adversarial retrieval attack methods to the surrogate model, black-box transfer attacks on RAG are further realized. Experiments conducted on opinion datasets across multiple topics show that the proposed attack strategy can significantly alter the opinion polarity of the content generated by RAG. This demonstrates the model's vulnerability and, more importantly, reveals the potential negative impact on user cognition and decision-making, making it easier to mislead users into accepting incorrect or biased information.

摘要: 检索增强生成(RAG)被应用于解决大型语言模型的幻觉问题和实时约束，但它也导致了对检索破坏攻击的脆弱性。已有研究主要探讨RAG在白盒和封闭域QA任务中的不可靠性。在本文中，我们旨在揭示检索增强生成(RAG)模型在面对意见操纵黑盒攻击时的脆弱性。我们探讨了此类攻击对用户认知和决策的影响，为提高RAG模型的可靠性和安全性提供了新的见解。我们通过指令对检索模型在RAG中的排序结果进行操作，并将这些结果作为数据来训练代理模型。通过对代理模型采用对抗性检索攻击方法，进一步实现了对RAG的黑箱转移攻击。在多个主题的观点数据集上进行的实验表明，所提出的攻击策略可以显著改变RAG生成的内容的观点极性。这表明了该模型的脆弱性，更重要的是，揭示了对用户认知和决策的潜在负面影响，使其更容易误导用户接受不正确或有偏见的信息。



## **43. Cross-Task Attack: A Self-Supervision Generative Framework Based on Attention Shift**

跨任务攻击：基于注意力转移的自我监督生成框架 cs.CV

Has been accepted by IJCNN2024

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13700v1) [paper-pdf](http://arxiv.org/pdf/2407.13700v1)

**Authors**: Qingyuan Zeng, Yunpeng Gong, Min Jiang

**Abstract**: Studying adversarial attacks on artificial intelligence (AI) systems helps discover model shortcomings, enabling the construction of a more robust system. Most existing adversarial attack methods only concentrate on single-task single-model or single-task cross-model scenarios, overlooking the multi-task characteristic of artificial intelligence systems. As a result, most of the existing attacks do not pose a practical threat to a comprehensive and collaborative AI system. However, implementing cross-task attacks is highly demanding and challenging due to the difficulty in obtaining the real labels of different tasks for the same picture and harmonizing the loss functions across different tasks. To address this issue, we propose a self-supervised Cross-Task Attack framework (CTA), which utilizes co-attention and anti-attention maps to generate cross-task adversarial perturbation. Specifically, the co-attention map reflects the area to which different visual task models pay attention, while the anti-attention map reflects the area that different visual task models neglect. CTA generates cross-task perturbations by shifting the attention area of samples away from the co-attention map and closer to the anti-attention map. We conduct extensive experiments on multiple vision tasks and the experimental results confirm the effectiveness of the proposed design for adversarial attacks.

摘要: 研究对人工智能(AI)系统的对抗性攻击有助于发现模型的缺陷，使构建更健壮的系统成为可能。现有的对抗性攻击方法大多只关注单任务单模型或单任务跨模型场景，忽略了人工智能系统的多任务特性。因此，现有的大多数攻击都不会对一个全面、协同的AI系统构成实际威胁。然而，由于难以获得同一图片的不同任务的真实标签以及协调不同任务之间的损失函数，实施跨任务攻击的要求和挑战性很高。针对这一问题，我们提出了一种自监督的跨任务攻击框架(CTA)，该框架利用共同注意图和反注意图来产生跨任务的敌意扰动。具体而言，共注意图反映了不同视觉任务模型注意的区域，反注意图反映了不同视觉任务模型忽视的区域。CTA通过将样本的注意区域从共同注意图转移到更接近反注意图的位置来产生跨任务扰动。我们在多个视觉任务上进行了大量的实验，实验结果证实了所提出的设计对于对抗性攻击的有效性。



## **44. Prover-Verifier Games improve legibility of LLM outputs**

证明者-验证者游戏提高了LLM输出的清晰度 cs.CL

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13692v1) [paper-pdf](http://arxiv.org/pdf/2407.13692v1)

**Authors**: Jan Hendrik Kirchner, Yining Chen, Harri Edwards, Jan Leike, Nat McAleese, Yuri Burda

**Abstract**: One way to increase confidence in the outputs of Large Language Models (LLMs) is to support them with reasoning that is clear and easy to check -- a property we call legibility. We study legibility in the context of solving grade-school math problems and show that optimizing chain-of-thought solutions only for answer correctness can make them less legible. To mitigate the loss in legibility, we propose a training algorithm inspired by Prover-Verifier Game from Anil et al. (2021). Our algorithm iteratively trains small verifiers to predict solution correctness, "helpful" provers to produce correct solutions that the verifier accepts, and "sneaky" provers to produce incorrect solutions that fool the verifier. We find that the helpful prover's accuracy and the verifier's robustness to adversarial attacks increase over the course of training. Furthermore, we show that legibility training transfers to time-constrained humans tasked with verifying solution correctness. Over course of LLM training human accuracy increases when checking the helpful prover's solutions, and decreases when checking the sneaky prover's solutions. Hence, training for checkability by small verifiers is a plausible technique for increasing output legibility. Our results suggest legibility training against small verifiers as a practical avenue for increasing legibility of large LLMs to humans, and thus could help with alignment of superhuman models.

摘要: 增加对大型语言模型(LLM)输出的信心的一种方法是用清晰且易于检查的推理来支持它们--我们称之为易读性。我们在解决小学数学问题的背景下研究了易读性，并表明只为了答案的正确性而优化思维链解决方案会降低它们的易读性。为了减少易读性的损失，我们提出了一种受Anil等人的Prover-Verator游戏启发的训练算法。(2021年)。我们的算法迭代地训练小的验证者来预测解决方案的正确性，“有帮助的”验证者来产生验证者接受的正确的解决方案，而“偷偷摸摸”的验证者产生愚弄验证者的不正确的解决方案。我们发现，随着训练过程的进行，有益证明者的准确率和验证者对敌意攻击的健壮性都有所提高。此外，我们还表明，易读性训练转移到负责验证解决方案正确性的时间受限的人身上。在LLM训练过程中，当检查有用的证明者的解时，人类的准确率提高，而当检查偷偷摸摸的证明者的解时，人类的准确率降低。因此，由小型验证员进行可校验性培训是提高输出清晰度的一种可行的技术。我们的结果表明，针对小验证者的易读性训练是提高大型LLM对人类易读性的实用途径，因此可能有助于超人模型的对齐。



## **45. Beyond Dropout: Robust Convolutional Neural Networks Based on Local Feature Masking**

Beyond Dropout：基于局部特征掩蔽的鲁棒卷积神经网络 cs.CV

It has been accepted by IJCNN 2024

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13646v1) [paper-pdf](http://arxiv.org/pdf/2407.13646v1)

**Authors**: Yunpeng Gong, Chuangliang Zhang, Yongjie Hou, Lifei Chen, Min Jiang

**Abstract**: In the contemporary of deep learning, where models often grapple with the challenge of simultaneously achieving robustness against adversarial attacks and strong generalization capabilities, this study introduces an innovative Local Feature Masking (LFM) strategy aimed at fortifying the performance of Convolutional Neural Networks (CNNs) on both fronts. During the training phase, we strategically incorporate random feature masking in the shallow layers of CNNs, effectively alleviating overfitting issues, thereby enhancing the model's generalization ability and bolstering its resilience to adversarial attacks. LFM compels the network to adapt by leveraging remaining features to compensate for the absence of certain semantic features, nurturing a more elastic feature learning mechanism. The efficacy of LFM is substantiated through a series of quantitative and qualitative assessments, collectively showcasing a consistent and significant improvement in CNN's generalization ability and resistance against adversarial attacks--a phenomenon not observed in current and prior methodologies. The seamless integration of LFM into established CNN frameworks underscores its potential to advance both generalization and adversarial robustness within the deep learning paradigm. Through comprehensive experiments, including robust person re-identification baseline generalization experiments and adversarial attack experiments, we demonstrate the substantial enhancements offered by LFM in addressing the aforementioned challenges. This contribution represents a noteworthy stride in advancing robust neural network architectures.

摘要: 在深度学习的当代，模型经常面临同时获得对对手攻击的鲁棒性和强大的泛化能力的挑战，该研究引入了一种创新的局部特征掩蔽(LFM)策略，旨在增强卷积神经网络(CNN)在这两个方面的性能。在训练阶段，我们战略性地将随机特征掩蔽引入到CNN的浅层，有效地缓解了过拟合问题，从而增强了模型的泛化能力，增强了模型对对手攻击的韧性。LFM通过利用剩余特征来补偿某些语义特征的缺失来迫使网络适应，从而培育出更灵活的特征学习机制。LFM的有效性通过一系列定量和定性评估得到证实，这些评估共同表明CNN的泛化能力和对对抗性攻击的抵抗力得到了持续和显著的改善--这是目前和以前的方法中没有观察到的现象。LFM无缝集成到已建立的CNN框架中，突显了它在深度学习范式中提高泛化和对抗性稳健性的潜力。通过全面的实验，包括稳健的人重新识别基线泛化实验和对抗性攻击实验，我们展示了LFM在应对上述挑战方面所提供的实质性增强。这一贡献代表着在推进健壮的神经网络结构方面迈出了值得注意的一步。



## **46. Not Just Change the Labels, Learn the Features: Watermarking Deep Neural Networks with Multi-View Data**

不仅仅改变标签，学习功能：使用多视图数据对深度神经网络进行水印 cs.CR

ECCV 2024

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2403.10663v2) [paper-pdf](http://arxiv.org/pdf/2403.10663v2)

**Authors**: Yuxuan Li, Sarthak Kumar Maharana, Yunhui Guo

**Abstract**: With the increasing prevalence of Machine Learning as a Service (MLaaS) platforms, there is a growing focus on deep neural network (DNN) watermarking techniques. These methods are used to facilitate the verification of ownership for a target DNN model to protect intellectual property. One of the most widely employed watermarking techniques involves embedding a trigger set into the source model. Unfortunately, existing methodologies based on trigger sets are still susceptible to functionality-stealing attacks, potentially enabling adversaries to steal the functionality of the source model without a reliable means of verifying ownership. In this paper, we first introduce a novel perspective on trigger set-based watermarking methods from a feature learning perspective. Specifically, we demonstrate that by selecting data exhibiting multiple features, also referred to as \emph{multi-view data}, it becomes feasible to effectively defend functionality stealing attacks. Based on this perspective, we introduce a novel watermarking technique based on Multi-view dATa, called MAT, for efficiently embedding watermarks within DNNs. This approach involves constructing a trigger set with multi-view data and incorporating a simple feature-based regularization method for training the source model. We validate our method across various benchmarks and demonstrate its efficacy in defending against model extraction attacks, surpassing relevant baselines by a significant margin. The code is available at: \href{https://github.com/liyuxuan-github/MAT}{https://github.com/liyuxuan-github/MAT}.

摘要: 随着机器学习即服务(MLaaS)平台的日益普及，深度神经网络(DNN)水印技术受到越来越多的关注。这些方法用于帮助验证目标DNN模型的所有权，以保护知识产权。最广泛使用的水印技术之一涉及将触发集嵌入到源模型中。遗憾的是，基于触发器集的现有方法仍然容易受到功能窃取攻击，这可能使攻击者能够在没有可靠的验证所有权的方法的情况下窃取源模型的功能。本文首先从特征学习的角度介绍了一种基于触发集的数字水印方法。具体地说，我们证明了通过选择表现出多个特征的数据，也称为\MPH(多视图数据)，可以有效地防御功能窃取攻击。基于这一观点，我们提出了一种新的基于多视点数据的水印技术，称为MAT，用于在DNN中有效地嵌入水印。该方法包括利用多视点数据构造触发集，并结合简单的基于特征的正则化方法来训练源模型。我们在不同的基准上验证了我们的方法，并证明了它在防御模型提取攻击方面的有效性，远远超过了相关的基线。代码可从以下网址获得：\href{https://github.com/liyuxuan-github/MAT}{https://github.com/liyuxuan-github/MAT}.



## **47. Distributionally and Adversarially Robust Logistic Regression via Intersecting Wasserstein Balls**

通过交叉Wasserstein Balls进行分布和反向稳健逻辑回归 math.OC

34 pages, 3 color figures, under review at a conference

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13625v1) [paper-pdf](http://arxiv.org/pdf/2407.13625v1)

**Authors**: Aras Selvi, Eleonora Kreacic, Mohsen Ghassemi, Vamsi Potluru, Tucker Balch, Manuela Veloso

**Abstract**: Empirical risk minimization often fails to provide robustness against adversarial attacks in test data, causing poor out-of-sample performance. Adversarially robust optimization (ARO) has thus emerged as the de facto standard for obtaining models that hedge against such attacks. However, while these models are robust against adversarial attacks, they tend to suffer severely from overfitting. To address this issue for logistic regression, we study the Wasserstein distributionally robust (DR) counterpart of ARO and show that this problem admits a tractable reformulation. Furthermore, we develop a framework to reduce the conservatism of this problem by utilizing an auxiliary dataset (e.g., synthetic, external, or out-of-domain data), whenever available, with instances independently sampled from a nonidentical but related ground truth. In particular, we intersect the ambiguity set of the DR problem with another Wasserstein ambiguity set that is built using the auxiliary dataset. We analyze the properties of the underlying optimization problem, develop efficient solution algorithms, and demonstrate that the proposed method consistently outperforms benchmark approaches on real-world datasets.

摘要: 经验风险最小化往往不能在测试数据中提供对对手攻击的健壮性，从而导致样本外性能较差。相反，稳健优化(ARO)因此已经成为获得对冲此类攻击的模型的事实上的标准。然而，尽管这些模型对对手攻击很健壮，但它们往往会受到过度拟合的严重影响。为了解决Logistic回归的这个问题，我们研究了ARO的Wasserstein分布健壮性(DR)对应的问题，并表明这个问题允许一个易于处理的重新公式。此外，我们开发了一个框架来通过利用辅助数据集(例如，合成的、外部的或域外的数据)来减少这个问题的保守性，只要有可用的辅助数据集，实例就独立地从不相同但相关的基本事实中采样。特别是，我们将DR问题的歧义集与使用辅助数据集构建的另一个Wasserstein歧义集相交。我们分析了底层优化问题的性质，开发了有效的求解算法，并证明了所提出的方法在真实数据集上的性能一致优于基准方法。



## **48. VeriQR: A Robustness Verification Tool for Quantum Machine Learning Models**

VeriQR：量子机器学习模型的稳健性验证工具 quant-ph

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13533v1) [paper-pdf](http://arxiv.org/pdf/2407.13533v1)

**Authors**: Yanling Lin, Ji Guan, Wang Fang, Mingsheng Ying, Zhaofeng Su

**Abstract**: Adversarial noise attacks present a significant threat to quantum machine learning (QML) models, similar to their classical counterparts. This is especially true in the current Noisy Intermediate-Scale Quantum era, where noise is unavoidable. Therefore, it is essential to ensure the robustness of QML models before their deployment. To address this challenge, we introduce \textit{VeriQR}, the first tool designed specifically for formally verifying and improving the robustness of QML models, to the best of our knowledge. This tool mimics real-world quantum hardware's noisy impacts by incorporating random noise to formally validate a QML model's robustness. \textit{VeriQR} supports exact (sound and complete) algorithms for both local and global robustness verification. For enhanced efficiency, it implements an under-approximate (complete) algorithm and a tensor network-based algorithm to verify local and global robustness, respectively. As a formal verification tool, \textit{VeriQR} can detect adversarial examples and utilize them for further analysis and to enhance the local robustness through adversarial training, as demonstrated by experiments on real-world quantum machine learning models. Moreover, it permits users to incorporate customized noise. Based on this feature, we assess \textit{VeriQR} using various real-world examples, and experimental outcomes confirm that the addition of specific quantum noise can enhance the global robustness of QML models. These processes are made accessible through a user-friendly graphical interface provided by \textit{VeriQR}, catering to general users without requiring a deep understanding of the counter-intuitive probabilistic nature of quantum computing.

摘要: 对抗噪声攻击对量子机器学习(QML)模型构成了重大威胁，类似于它们的经典对应模型。尤其是在当前嘈杂的中尺度量子时代，噪音是不可避免的。因此，在QML模型部署之前，确保其健壮性是至关重要的。为了应对这一挑战，据我们所知，我们引入了第一个专门为正式验证和改进QML模型的健壮性而设计的工具\textit{VeriQR}。该工具通过加入随机噪声来正式验证QML模型的稳健性，从而模拟现实世界量子硬件的噪声影响。\textit{VeriQR}支持针对本地和全局健壮性验证的精确(声音和完整)算法。为了提高效率，它分别采用了欠近似(完全)算法和基于张量网络的算法来验证局部和全局的稳健性。在真实量子机器学习模型上的实验表明，作为一种形式化的验证工具，该工具可以检测敌意实例，并利用它们进行进一步的分析，并通过对抗性训练来增强局部稳健性。此外，它还允许用户加入定制的噪音。基于这一特征，我们使用各种真实世界的例子来评估文本{VeriQR}，实验结果证实了特定量子噪声的加入可以增强QML模型的全局稳健性。这些过程可以通过一个用户友好的图形界面进行访问，满足普通用户的需要，而不需要深入了解量子计算的反直觉概率性质。



## **49. Correlation inference attacks against machine learning models**

针对机器学习模型的相关推理攻击 cs.LG

Published in Science Advances. This version contains both the main  paper and supplementary material. There are minor editorial differences  between this version and the published version. The first two authors  contributed equally

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2112.08806v4) [paper-pdf](http://arxiv.org/pdf/2112.08806v4)

**Authors**: Ana-Maria Creţu, Florent Guépin, Yves-Alexandre de Montjoye

**Abstract**: Despite machine learning models being widely used today, the relationship between a model and its training dataset is not well understood. We explore correlation inference attacks, whether and when a model leaks information about the correlations between the input variables of its training dataset. We first propose a model-less attack, where an adversary exploits the spherical parametrization of correlation matrices alone to make an informed guess. Second, we propose a model-based attack, where an adversary exploits black-box model access to infer the correlations using minimal and realistic assumptions. Third, we evaluate our attacks against logistic regression and multilayer perceptron models on three tabular datasets and show the models to leak correlations. We finally show how extracted correlations can be used as building blocks for attribute inference attacks and enable weaker adversaries. Our results raise fundamental questions on what a model does and should remember from its training set.

摘要: 尽管机器学习模型在今天得到了广泛的应用，但模型与其训练数据集之间的关系并未得到很好的理解。我们探讨了关联推理攻击，即模型是否以及何时泄露了有关其训练数据集的输入变量之间的相关性的信息。我们首先提出了一种无模型攻击，其中敌手仅利用相关矩阵的球面参数化来进行知情猜测。其次，我们提出了一种基于模型的攻击，其中对手利用黑盒模型访问来使用最小和现实的假设来推断相关性。第三，我们在三个表格数据集上评估了我们对Logistic回归模型和多层感知器模型的攻击，并显示了模型的泄漏相关性。最后，我们展示了如何将提取的相关性用作属性推理攻击的构建块，并使较弱的对手能够进行攻击。我们的结果提出了一些基本问题，即模型从其训练集中做了什么，以及应该记住什么。



## **50. NeuroPlug: Plugging Side-Channel Leaks in NPUs using Space Filling Curves**

NeuroPlug：使用空间填充曲线堵住NPU中的侧通道泄漏 cs.CR

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13383v1) [paper-pdf](http://arxiv.org/pdf/2407.13383v1)

**Authors**: Nivedita Shrivastava, Smruti R. Sarangi

**Abstract**: Securing deep neural networks (DNNs) from side-channel attacks is an important problem as of today, given the substantial investment of time and resources in acquiring the raw data and training complex models. All published countermeasures (CMs) add noise N to a signal X (parameter of interest such as the net memory traffic that is leaked). The adversary observes X+N ; we shall show that it is easy to filter this noise out using targeted measurements, statistical analyses and different kinds of reasonably-assumed side information. We present a novel CM NeuroPlug that is immune to these attack methodologies mainly because we use a different formulation CX + N . We introduce a multiplicative variable C that naturally arises from feature map compression; it plays a key role in obfuscating the parameters of interest. Our approach is based on mapping all the computations to a 1-D space filling curve and then performing a sequence of tiling, compression and binning-based obfuscation operations. We follow up with proposing a theoretical framework based on Mellin transforms that allows us to accurately quantify the size of the search space as a function of the noise we add and the side information that an adversary possesses. The security guarantees provided by NeuroPlug are validated using a battery of statistical and information theory-based tests. We also demonstrate a substantial performance enhancement of 15% compared to the closest competing work.

摘要: 考虑到获取原始数据和训练复杂模型需要大量的时间和资源投入，保护深度神经网络(DNN)免受旁路攻击是当今的一个重要问题。所有发布的对策(CM)都将噪声N添加到信号X(感兴趣的参数，例如泄漏的网络内存流量)。对手观察X+N；我们将展示使用有针对性的测量、统计分析和各种合理假设的辅助信息来过滤这种噪声是很容易的。我们提出了一种新的CM NeuroPlug，它对这些攻击方法免疫，主要是因为我们使用了不同的公式CX+N。我们引入了一个自然产生于特征地图压缩的乘法变量C；它在混淆感兴趣的参数方面起着关键作用。我们的方法是将所有的计算映射到一维空间填充曲线上，然后执行一系列基于平铺、压缩和装箱的混淆操作。接下来，我们提出了一个基于梅林变换的理论框架，该框架允许我们准确地量化搜索空间的大小，作为我们添加的噪声和对手拥有的辅助信息的函数。NeuroPlug提供的安全保证使用了一系列基于统计和信息论的测试进行了验证。与最接近的竞争对手相比，我们还展示了15%的显著性能提升。



