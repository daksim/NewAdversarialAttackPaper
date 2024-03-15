# Latest Adversarial Attack Papers
**update at 2024-03-15 11:36:56**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Expressive Losses for Verified Robustness via Convex Combinations**

通过凸组合验证稳健性的表达损失 cs.LG

ICLR 2024

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2305.13991v2) [paper-pdf](http://arxiv.org/pdf/2305.13991v2)

**Authors**: Alessandro De Palma, Rudy Bunel, Krishnamurthy Dvijotham, M. Pawan Kumar, Robert Stanforth, Alessio Lomuscio

**Abstract**: In order to train networks for verified adversarial robustness, it is common to over-approximate the worst-case loss over perturbation regions, resulting in networks that attain verifiability at the expense of standard performance. As shown in recent work, better trade-offs between accuracy and robustness can be obtained by carefully coupling adversarial training with over-approximations. We hypothesize that the expressivity of a loss function, which we formalize as the ability to span a range of trade-offs between lower and upper bounds to the worst-case loss through a single parameter (the over-approximation coefficient), is key to attaining state-of-the-art performance. To support our hypothesis, we show that trivial expressive losses, obtained via convex combinations between adversarial attacks and IBP bounds, yield state-of-the-art results across a variety of settings in spite of their conceptual simplicity. We provide a detailed analysis of the relationship between the over-approximation coefficient and performance profiles across different expressive losses, showing that, while expressivity is essential, better approximations of the worst-case loss are not necessarily linked to superior robustness-accuracy trade-offs.

摘要: 为了训练网络以获得验证的对抗鲁棒性，通常会过度近似扰动区域的最坏情况损失，导致网络以牺牲标准性能为代价获得可验证性。正如最近的工作所示，通过谨慎地将对抗训练与过近似耦合，可以在准确性和鲁棒性之间获得更好的权衡。我们假设损失函数的表现性，我们正式化为通过单个参数（过近似系数）在最坏情况下损失的下限和上限之间进行权衡的能力，是获得最先进性能的关键。为了支持我们的假设，我们表明，微不足道的表达损失，通过对抗攻击和IBP边界之间的凸组合，产生国家的最先进的结果，在各种设置，尽管其概念简单。我们提供了一个详细的关系的过近似系数和性能配置文件跨不同的表达损失，表明，虽然表达是必不可少的，更好的近似最坏情况下损失不一定与优越的鲁棒性准确性权衡。



## **2. What Sketch Explainability Really Means for Downstream Tasks**

草图可解释性对下游任务的真正意义是什么 cs.CV

CVPR 2024

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09480v1) [paper-pdf](http://arxiv.org/pdf/2403.09480v1)

**Authors**: Hmrishav Bandyopadhyay, Pinaki Nath Chowdhury, Ayan Kumar Bhunia, Aneeshan Sain, Tao Xiang, Yi-Zhe Song

**Abstract**: In this paper, we explore the unique modality of sketch for explainability, emphasising the profound impact of human strokes compared to conventional pixel-oriented studies. Beyond explanations of network behavior, we discern the genuine implications of explainability across diverse downstream sketch-related tasks. We propose a lightweight and portable explainability solution -- a seamless plugin that integrates effortlessly with any pre-trained model, eliminating the need for re-training. Demonstrating its adaptability, we present four applications: highly studied retrieval and generation, and completely novel assisted drawing and sketch adversarial attacks. The centrepiece to our solution is a stroke-level attribution map that takes different forms when linked with downstream tasks. By addressing the inherent non-differentiability of rasterisation, we enable explanations at both coarse stroke level (SLA) and partial stroke level (P-SLA), each with its advantages for specific downstream tasks.

摘要: 在本文中，我们探讨了素描的独特形态的可解释性，强调了人类笔划的深刻影响相比，传统的像素为导向的研究。除了对网络行为的解释，我们还发现了不同下游草图相关任务的可解释性的真正含义。我们提出了一个轻量级和可移植的解释性解决方案—一个无缝插件，可以轻松地与任何预训练模型集成，消除了重新训练的需要。展示其适应性，我们提出了四个应用程序：高度研究的检索和生成，和完全新颖的辅助绘图和素描对抗攻击。我们的解决方案的核心是一个笔划级别的归因图，当与下游任务相关联时，它会呈现不同的形式。通过解决光栅化固有的不可微性，我们可以在粗笔划水平（SLA）和部分笔划水平（P—SLA）进行解释，每一个都有其优点，用于特定的下游任务。



## **3. Adversarial Fine-tuning of Compressed Neural Networks for Joint Improvement of Robustness and Efficiency**

压缩神经网络的对抗性细调联合提高鲁棒性和效率 cs.LG

22 pages, 4 figures, 6 tables

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09441v1) [paper-pdf](http://arxiv.org/pdf/2403.09441v1)

**Authors**: Hallgrimur Thorsteinsson, Valdemar J Henriksen, Tong Chen, Raghavendra Selvan

**Abstract**: As deep learning (DL) models are increasingly being integrated into our everyday lives, ensuring their safety by making them robust against adversarial attacks has become increasingly critical. DL models have been found to be susceptible to adversarial attacks which can be achieved by introducing small, targeted perturbations to disrupt the input data. Adversarial training has been presented as a mitigation strategy which can result in more robust models. This adversarial robustness comes with additional computational costs required to design adversarial attacks during training. The two objectives -- adversarial robustness and computational efficiency -- then appear to be in conflict of each other. In this work, we explore the effects of two different model compression methods -- structured weight pruning and quantization -- on adversarial robustness. We specifically explore the effects of fine-tuning on compressed models, and present the trade-off between standard fine-tuning and adversarial fine-tuning. Our results show that compression does not inherently lead to loss in model robustness and adversarial fine-tuning of a compressed model can yield large improvement to the robustness performance of models. We present experiments on two benchmark datasets showing that adversarial fine-tuning of compressed models can achieve robustness performance comparable to adversarially trained models, while also improving computational efficiency.

摘要: 随着深度学习（DL）模型越来越多地融入到我们的日常生活中，通过使其强大的对抗性攻击来确保其安全性变得越来越重要。已经发现DL模型容易受到对抗攻击，可以通过引入小的、有针对性的扰动来破坏输入数据来实现。对抗性训练已被提出作为一种缓解策略，可以导致更稳健的模型。这种对抗性鲁棒性带来了在训练期间设计对抗性攻击所需的额外计算成本。这两个目标——对抗鲁棒性和计算效率——似乎是相互冲突的。在这项工作中，我们探讨了两种不同的模型压缩方法—结构化权重修剪和量化—对抗鲁棒性的影响。我们专门探讨了微调对压缩模型的影响，并提出了标准微调和对抗微调之间的权衡。我们的研究结果表明，压缩并不会固有地导致模型鲁棒性的损失，并且对压缩模型进行对抗性微调可以大大提高模型的鲁棒性性能。我们提出的两个基准数据集的实验表明，对抗性微调压缩模型可以实现与对抗性训练模型相当的鲁棒性能，同时也提高了计算效率。



## **4. Divide-and-Conquer Attack: Harnessing the Power of LLM to Bypass Safety Filters of Text-to-Image Models**

分割和征服攻击：利用LLM的力量绕过文本到图像模型的安全过滤器 cs.AI

23 pages, 11 figures, under review

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2312.07130v3) [paper-pdf](http://arxiv.org/pdf/2312.07130v3)

**Authors**: Yimo Deng, Huangxun Chen

**Abstract**: Text-to-image (TTI) models offer many innovative services but also raise ethical concerns due to their potential to generate unethical images. Most public TTI services employ safety filters to prevent unintended images. In this work, we introduce the Divide-and-Conquer Attack to circumvent the safety filters of state-of the-art TTI models, including DALL-E 3 and Midjourney. Our attack leverages LLMs as text transformation agents to create adversarial prompts. We design attack helper prompts that effectively guide LLMs to break down an unethical drawing intent into multiple benign descriptions of individual image elements, allowing them to bypass safety filters while still generating unethical images. Because the latent harmful meaning only becomes apparent when all individual elements are drawn together. Our evaluation demonstrates that our attack successfully circumvents multiple strong closed-box safety filters. The comprehensive success rate of DACA bypassing the safety filters of the state-of-the-art TTI engine DALL-E 3 is above 85%, while the success rate for bypassing Midjourney V6 exceeds 75%. Our findings have more severe security implications than methods of manual crafting or iterative TTI model querying due to lower attack barrier, enhanced interpretability , and better adaptation to defense. Our prototype is available at: https://github.com/researchcode001/Divide-and-Conquer-Attack

摘要: 文本到图像（TTI）模型提供了许多创新的服务，但也引起了道德关注，因为它们可能产生不道德的图像。大多数公共TTI服务使用安全过滤器来防止意外图像。在这项工作中，我们引入了分治攻击来规避最先进的TTI模型的安全过滤器，包括DALL—E3和Midjourney。我们的攻击利用LLM作为文本转换代理来创建对抗性提示。我们设计了攻击助手提示，有效地指导LLM将不道德的绘图意图分解成单个图像元素的多个良性描述，允许他们绕过安全过滤器，同时仍然生成不道德的图像。因为潜在的有害含义只有当所有的个体元素被结合在一起时才变得明显。我们的评估表明，我们的攻击成功地绕过了多个强大的封闭箱安全过滤器。DACA绕过最先进的TTI引擎DALL—E 3的安全过滤器的综合成功率在85%以上，而绕过中途V6的成功率超过75%。我们的研究结果比手工制作或迭代TTI模型查询的方法具有更严重的安全影响，由于更低的攻击屏障，增强的可解释性，以及更好的适应防御。我们的原型可在：www.example.com



## **5. AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Adversarial Visual-Instructions**

AVIBENCH：对抗视觉指令上大型视觉语言模型的鲁棒性评估 cs.CV

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09346v1) [paper-pdf](http://arxiv.org/pdf/2403.09346v1)

**Authors**: Hao Zhang, Wenqi Shao, Hong Liu, Yongqiang Ma, Ping Luo, Yu Qiao, Kaipeng Zhang

**Abstract**: Large Vision-Language Models (LVLMs) have shown significant progress in well responding to visual-instructions from users. However, these instructions, encompassing images and text, are susceptible to both intentional and inadvertent attacks. Despite the critical importance of LVLMs' robustness against such threats, current research in this area remains limited. To bridge this gap, we introduce AVIBench, a framework designed to analyze the robustness of LVLMs when facing various adversarial visual-instructions (AVIs), including four types of image-based AVIs, ten types of text-based AVIs, and nine types of content bias AVIs (such as gender, violence, cultural, and racial biases, among others). We generate 260K AVIs encompassing five categories of multimodal capabilities (nine tasks) and content bias. We then conduct a comprehensive evaluation involving 14 open-source LVLMs to assess their performance. AVIBench also serves as a convenient tool for practitioners to evaluate the robustness of LVLMs against AVIs. Our findings and extensive experimental results shed light on the vulnerabilities of LVLMs, and highlight that inherent biases exist even in advanced closed-source LVLMs like GeminiProVision and GPT-4V. This underscores the importance of enhancing the robustness, security, and fairness of LVLMs. The source code and benchmark will be made publicly available.

摘要: 大型视觉语言模型（LVLM）在很好地响应用户的视觉指令方面取得了显著进展。然而，这些包含图像和文本的指令很容易受到有意和无意的攻击。尽管LVLM对此类威胁的鲁棒性至关重要，但目前在该领域的研究仍然有限。为了弥补这一差距，我们引入了AVIBench，这是一个框架，旨在分析LVLM在面对各种对抗视觉指令（AVIs）时的鲁棒性，包括四种基于图像的AVIs，十种基于文本的AVIs和九种内容偏见的AVIs（如性别、暴力、文化和种族偏见等）。我们生成了260K AVI，涵盖了五个类别的多模态能力（九个任务）和内容偏见。然后，我们对14个开源LVLM进行了全面的评估，以评估它们的性能。AVIBench也是一个方便的工具，为从业者评估LVLM的鲁棒性对AVIs。我们的发现和广泛的实验结果揭示了LVLM的脆弱性，并强调即使在先进的闭源LVLM，如GeminiProVision和GPT—4V中也存在固有的偏见。这强调了增强LVLM鲁棒性、安全性和公平性的重要性。源代码和基准测试将公开。



## **6. Privacy Preserving Anomaly Detection on Homomorphic Encrypted Data from IoT Sensors**

基于IoT传感器的同态加密数据的隐私保护异常检测 cs.CR

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09322v1) [paper-pdf](http://arxiv.org/pdf/2403.09322v1)

**Authors**: Anca Hangan, Dragos Lazea, Tudor Cioara

**Abstract**: IoT devices have become indispensable components of our lives, and the advancement of AI technologies will make them even more pervasive, increasing the vulnerability to malfunctions or cyberattacks and raising privacy concerns. Encryption can mitigate these challenges; however, most existing anomaly detection techniques decrypt the data to perform the analysis, potentially undermining the encryption protection provided during transit or storage. Homomorphic encryption schemes are promising solutions as they enable the processing and execution of operations on IoT data while still encrypted, however, these schemes offer only limited operations, which poses challenges to their practical usage. In this paper, we propose a novel privacy-preserving anomaly detection solution designed for homomorphically encrypted data generated by IoT devices that efficiently detects abnormal values without performing decryption. We have adapted the Histogram-based anomaly detection technique for TFHE scheme to address limitations related to the input size and the depth of computation by implementing vectorized support operations. These operations include addition, value placement in buckets, labeling abnormal buckets based on a threshold frequency, labeling abnormal values based on their range, and bucket labels. Evaluation results show that the solution effectively detects anomalies without requiring data decryption and achieves consistent results comparable to the mechanism operating on plain data. Also, it shows robustness and resilience against various challenges commonly encountered in IoT environments, such as noisy sensor data, adversarial attacks, communication failures, and device malfunctions. Moreover, the time and computational overheads determined for several solution configurations, despite being large, are reasonable compared to those reported in existing literature.

摘要: 物联网设备已经成为我们生活中不可或缺的组成部分，人工智能技术的进步将使它们变得更加普遍，增加对故障或网络攻击的脆弱性，并引发隐私问题。加密可以缓解这些挑战；然而，大多数现有的异常检测技术都会解密数据以执行分析，这可能会破坏传输或存储期间提供的加密保护。同态加密方案是有前途的解决方案，因为它们能够在仍然加密的情况下处理和执行物联网数据的操作，然而，这些方案仅提供有限的操作，这对它们的实际使用构成了挑战。在本文中，我们提出了一种新的隐私保护异常检测解决方案，该解决方案专为IoT设备生成的同态加密数据而设计，该方案无需执行解密即可有效地检测异常值。我们采用基于直方图的异常检测技术，通过实现向量化支持操作，解决了输入大小和计算深度的限制。这些操作包括加法、在桶中放置值、基于阈值频率标记异常桶、基于异常值的范围标记异常值以及桶标签。评估结果表明，该解决方案在不需要数据解密的情况下有效地检测异常，并取得了与在普通数据上操作的机制相比较的一致结果。此外，它还显示出对物联网环境中常见的各种挑战的稳健性和弹性，例如噪声传感器数据、对抗攻击、通信故障和设备故障。此外，虽然时间和计算费用确定的几个解决方案配置，虽然是大的，是合理的相比，在现有的文献报道。



## **7. X-CANIDS: Signal-Aware Explainable Intrusion Detection System for Controller Area Network-Based In-Vehicle Network**

X—CANIDS：基于控制器局域网的车载网络信号感知可解释入侵检测系统 cs.CR

This is the Accepted version of an article for publication in IEEE  TVT

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2303.12278v3) [paper-pdf](http://arxiv.org/pdf/2303.12278v3)

**Authors**: Seonghoon Jeong, Sangho Lee, Hwejae Lee, Huy Kang Kim

**Abstract**: Controller Area Network (CAN) is an essential networking protocol that connects multiple electronic control units (ECUs) in a vehicle. However, CAN-based in-vehicle networks (IVNs) face security risks owing to the CAN mechanisms. An adversary can sabotage a vehicle by leveraging the security risks if they can access the CAN bus. Thus, recent actions and cybersecurity regulations (e.g., UNR 155) require carmakers to implement intrusion detection systems (IDSs) in their vehicles. The IDS should detect cyberattacks and provide additional information to analyze conducted attacks. Although many IDSs have been proposed, considerations regarding their feasibility and explainability remain lacking. This study proposes X-CANIDS, which is a novel IDS for CAN-based IVNs. X-CANIDS dissects the payloads in CAN messages into human-understandable signals using a CAN database. The signals improve the intrusion detection performance compared with the use of bit representations of raw payloads. These signals also enable an understanding of which signal or ECU is under attack. X-CANIDS can detect zero-day attacks because it does not require any labeled dataset in the training phase. We confirmed the feasibility of the proposed method through a benchmark test on an automotive-grade embedded device with a GPU. The results of this work will be valuable to carmakers and researchers considering the installation of in-vehicle IDSs for their vehicles.

摘要: 控制器局域网（CAN）是连接车辆中多个电子控制单元（ECU）的基本网络协议。然而，基于CAN的车载网络（IVN）由于CAN机制而面临安全风险。如果对手可以访问CAN总线，他们可以利用安全风险来破坏车辆。因此，最近的行动和网络安全法规（例如，UNR 155）要求汽车制造商在其车辆上安装入侵检测系统（IDS）。IDS应检测网络攻击并提供额外信息以分析已进行的攻击。虽然已经提出了许多IDS，但对其可行性和可解释性的考虑仍然不足。本文提出了一种新的基于CAN的IVN入侵检测系统X—CANIDS。X—CANIDS使用CAN数据库将CAN消息中的有效载荷分解为人类可理解的信号。与使用原始有效载荷的比特表示相比，这些信号提高了入侵检测性能。这些信号还可以帮助了解哪个信号或ECU受到攻击。X—CANIDS可以检测零日攻击，因为它在训练阶段不需要任何标记数据集。通过对一个带有GPU的汽车级嵌入式设备的基准测试，验证了该方法的可行性。这项工作的结果将是有价值的汽车制造商和研究人员考虑安装车载IDS为他们的车辆。



## **8. Soften to Defend: Towards Adversarial Robustness via Self-Guided Label Refinement**

软化防御：通过自我引导的标签细化实现对抗鲁棒性 cs.LG

Accepted to CVPR 2024

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09101v1) [paper-pdf](http://arxiv.org/pdf/2403.09101v1)

**Authors**: Daiwei Yu, Zhuorong Li, Lina Wei, Canghong Jin, Yun Zhang, Sixian Chan

**Abstract**: Adversarial training (AT) is currently one of the most effective ways to obtain the robustness of deep neural networks against adversarial attacks. However, most AT methods suffer from robust overfitting, i.e., a significant generalization gap in adversarial robustness between the training and testing curves. In this paper, we first identify a connection between robust overfitting and the excessive memorization of noisy labels in AT from a view of gradient norm. As such label noise is mainly caused by a distribution mismatch and improper label assignments, we are motivated to propose a label refinement approach for AT. Specifically, our Self-Guided Label Refinement first self-refines a more accurate and informative label distribution from over-confident hard labels, and then it calibrates the training by dynamically incorporating knowledge from self-distilled models into the current model and thus requiring no external teachers. Empirical results demonstrate that our method can simultaneously boost the standard accuracy and robust performance across multiple benchmark datasets, attack types, and architectures. In addition, we also provide a set of analyses from the perspectives of information theory to dive into our method and suggest the importance of soft labels for robust generalization.

摘要: 对抗训练（AT）是目前获得深度神经网络对抗对抗攻击鲁棒性的最有效方法之一。然而，大多数AT方法遭受鲁棒过拟合，即，训练曲线和测试曲线之间的对抗鲁棒性存在显著的泛化差距。本文首先从梯度范数的角度，确定了鲁棒过拟合与AT中噪声标签过度记忆之间的联系。由于这种标签噪声主要是由分布不匹配和不正确的标签分配引起的，我们有动机提出一种标签细化方法的AT。具体来说，我们的自我引导标签细化首先从过于自信的硬标签中自我细化一个更准确和信息丰富的标签分布，然后通过动态地将来自自我提炼模型的知识整合到当前模型中来校准训练，从而不需要外部教师。实验结果表明，我们的方法可以同时提高标准的准确性和鲁棒性能跨多个基准数据集，攻击类型和架构。此外，我们还提供了一系列的分析从信息论的角度深入我们的方法，并建议软标签的重要性，鲁棒推广。



## **9. Chaotic Masking Protocol for Secure Communication and Attack Detection in Remote Estimation of Cyber-Physical Systems**

网络物理系统远程估计中安全通信和攻击检测的混沌掩蔽协议 eess.SY

8 pages, 7 figures

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09076v1) [paper-pdf](http://arxiv.org/pdf/2403.09076v1)

**Authors**: Tao Chen, Andreu Cecilia, Daniele Astolfi, Lei Wang, Zhitao Liu, Hongye Su

**Abstract**: In remote estimation of cyber-physical systems (CPSs), sensor measurements transmitted through network may be attacked by adversaries, leading to leakage risk of privacy (e.g., the system state), and/or failure of the remote estimator. To deal with this problem, a chaotic masking protocol is proposed in this paper to secure the sensor measurements transmission. In detail, at the plant side, a chaotic dynamic system is deployed to encode the sensor measurement, and at the estimator side, an estimator estimates both states of the physical plant and the chaotic system. With this protocol, no additional secure communication links is needed for synchronization, and the masking effect can be perfectly removed when the estimator is in steady state. Furthermore, this masking protocol can deal with multiple types of attacks, i.e., eavesdropping attack, replay attack, and stealthy false data injection attack.

摘要: 在网络物理系统（CPS）的远程估计中，通过网络传输的传感器测量值可能会受到攻击者的攻击，导致隐私泄露风险（例如，系统状态）和/或远程估计器的故障。针对这一问题，本文提出了一种混沌掩蔽协议，以保证传感器测量数据的传输安全。详细地，在工厂侧，部署混沌动态系统来编码传感器测量，并且在估计器侧，估计器估计物理工厂和混沌系统的状态。该协议不需要额外的安全通信链路来同步，并且当估计器处于稳态时，屏蔽效应可以被完全消除。此外，该屏蔽协议可以处理多种类型的攻击，即，窃听攻击、重放攻击和隐形假数据注入攻击。



## **10. Attacking the Diebold Signature Variant -- RSA Signatures with Unverified High-order Padding**

攻击Diebold型签名变种--使用未经验证的高阶填充的RSA签名 cs.CR

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.01048v2) [paper-pdf](http://arxiv.org/pdf/2403.01048v2)

**Authors**: Ryan W. Gardner, Tadayoshi Kohno, Alec Yasinsac

**Abstract**: We examine a natural but improper implementation of RSA signature verification deployed on the widely used Diebold Touch Screen and Optical Scan voting machines. In the implemented scheme, the verifier fails to examine a large number of the high-order bits of signature padding and the public exponent is three. We present an very mathematically simple attack that enables an adversary to forge signatures on arbitrary messages in a negligible amount of time.

摘要: 我们研究了一个自然的，但不正确的实施RSA签名验证部署在广泛使用的Die博尔德触摸屏和光学扫描投票机。在实现的方案中，验证者无法检查签名填充的大量高位比特，且公共指数为3。我们提出了一种数学上非常简单的攻击，使对手能够在可忽略的时间内伪造任意消息的签名。



## **11. Verifix: Post-Training Correction to Improve Label Noise Robustness with Verified Samples**

Verifix：训练后校正，以提高标签噪声耐用性，并使用验证样本 cs.LG

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2403.08618v1) [paper-pdf](http://arxiv.org/pdf/2403.08618v1)

**Authors**: Sangamesh Kodge, Deepak Ravikumar, Gobinda Saha, Kaushik Roy

**Abstract**: Label corruption, where training samples have incorrect labels, can significantly degrade the performance of machine learning models. This corruption often arises from non-expert labeling or adversarial attacks. Acquiring large, perfectly labeled datasets is costly, and retraining large models from scratch when a clean dataset becomes available is computationally expensive. To address this challenge, we propose Post-Training Correction, a new paradigm that adjusts model parameters after initial training to mitigate label noise, eliminating the need for retraining. We introduce Verifix, a novel Singular Value Decomposition (SVD) based algorithm that leverages a small, verified dataset to correct the model weights using a single update. Verifix uses SVD to estimate a Clean Activation Space and then projects the model's weights onto this space to suppress activations corresponding to corrupted data. We demonstrate Verifix's effectiveness on both synthetic and real-world label noise. Experiments on the CIFAR dataset with 25% synthetic corruption show 7.36% generalization improvements on average. Additionally, we observe generalization improvements of up to 2.63% on naturally corrupted datasets like WebVision1.0 and Clothing1M.

摘要: 标签损坏，即训练样本具有不正确的标签，会显著降低机器学习模型的性能。这种腐败通常由非专家标签或对抗性攻击引起。获取大的、完美标记的数据集是昂贵的，而当一个干净的数据集可用时，从头开始重新训练大型模型在计算上是昂贵的。为了解决这一挑战，我们提出了训练后校正（Post—Training Correction），这是一种新的范式，在初始训练后调整模型参数，以减轻标签噪声，消除了再训练的需要。我们介绍了Verifix，一种新的基于奇异值分解（SVD）的算法，它利用一个小的，经过验证的数据集，使用一次更新来校正模型权重。Verifix使用SVD来估计干净的激活空间，然后将模型的权重投影到该空间上，以抑制与损坏数据对应的激活。我们证明了Verifix在合成和真实世界标签噪声上的有效性。在CIFAR数据集上的实验显示，25%的合成损坏率平均提高了7.36%。此外，我们观察到在WebVision1.0和Clothing1M等自然损坏的数据集上的泛化改进高达2.63%。



## **12. Continual Adversarial Defense**

持续对抗性防御 cs.CV

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2312.09481v2) [paper-pdf](http://arxiv.org/pdf/2312.09481v2)

**Authors**: Qian Wang, Yaoyao Liu, Hefei Ling, Yingwei Li, Qihao Liu, Ping Li, Jiazhong Chen, Alan Yuille, Ning Yu

**Abstract**: In response to the rapidly evolving nature of adversarial attacks against visual classifiers on a monthly basis, numerous defenses have been proposed to generalize against as many known attacks as possible. However, designing a defense method that generalizes to all types of attacks is not realistic because the environment in which defense systems operate is dynamic and comprises various unique attacks that emerge as time goes on. The defense system must gather online few-shot defense feedback to promptly enhance itself, leveraging efficient memory utilization. Therefore, we propose the first continual adversarial defense (CAD) framework that adapts to any attacks in a dynamic scenario, where various attacks emerge stage by stage. In practice, CAD is modeled under four principles: (1) continual adaptation to new attacks without catastrophic forgetting, (2) few-shot adaptation, (3) memory-efficient adaptation, and (4) high accuracy on both clean and adversarial images. We explore and integrate cutting-edge continual learning, few-shot learning, and ensemble learning techniques to qualify the principles. Experiments conducted on CIFAR-10 and ImageNet-100 validate the effectiveness of our approach against multiple stages of modern adversarial attacks and demonstrate significant improvements over numerous baseline methods. In particular, CAD is capable of quickly adapting with minimal feedback and a low cost of defense failure, while maintaining good performance against previous attacks. Our research sheds light on a brand-new paradigm for continual defense adaptation against dynamic and evolving attacks.

摘要: 为了应对每月针对视觉分类器的对抗性攻击迅速演变的性质，提出了许多防御措施，以概括尽可能多的已知攻击。然而，设计一种概括所有类型攻击的防御方法是不现实的，因为防御系统运行的环境是动态的，包括随着时间的推移而出现的各种独特的攻击。防御系统必须收集在线的少量防御反馈，以及时提高自己，利用高效的内存利用。因此，我们提出了第一个连续对抗防御(CAD)框架，该框架能够适应动态场景中的任何攻击，其中各种攻击是逐步出现的。在实践中，CAD的建模遵循四个原则：(1)持续适应新的攻击而不发生灾难性遗忘；(2)少镜头适应；(3)内存高效适应；(4)干净图像和对抗性图像的高精度。我们探索并集成了尖端的持续学习、少机会学习和集成学习技术来验证这些原则。在CIFAR-10和ImageNet-100上进行的实验验证了我们的方法对现代对抗性攻击的多个阶段的有效性，并显示出与许多基准方法相比的显著改进。特别是，CAD能够以最小的反馈和较低的防御失败成本快速适应，同时保持对先前攻击的良好性能。我们的研究揭示了一种针对动态和不断变化的攻击进行持续防御适应的全新范式。



## **13. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

杰基尔博士和海德先生：LLMS的两张面孔 cs.CR

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2312.03853v2) [paper-pdf](http://arxiv.org/pdf/2312.03853v2)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Only a year ago, we witnessed a rise in the use of Large Language Models (LLMs), especially when combined with applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with opposite characteristics as those of the truthful assistants they are supposed to be. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversation followed a role-play style to get the response the assistant was not allowed to provide. By making use of personas, we show that the response that is prohibited is actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Bard. We also introduce several ways of activating such adversarial personas, altogether showing that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.

摘要: 就在一年前，我们目睹了大型语言模型（LLM）的使用的增长，特别是当与聊天机器人助理等应用程序相结合时。实施了安全机制和专门培训程序，以防止这些助理作出不当反应。在这项工作中，我们绕过了ChatGPT和Bard（在某种程度上，还有Bing聊天）的这些措施，让他们模仿具有相反特征的复杂人物角色，而他们应该是真实的助手。我们首先创建这些人物角色的详细传记，然后在与相同的聊天机器人的新会话中使用。我们的谈话遵循了角色扮演的风格，以得到助理不允许提供的回应。通过使用人物角色，我们表明实际上提供了被禁止的响应，从而可以获得未经授权的、非法的或有害的信息。这项工作表明，通过使用对抗性角色，人们可以克服ChatGPT和Bard提出的安全机制。我们还介绍了几种激活这种对抗性角色的方法，表明这两种聊天机器人都容易受到这种攻击。基于同样的原则，我们引入了两种防御措施，推动模型解释值得信赖的个性，并使其更强大地抵御此类攻击。



## **14. An Extended View on Measuring Tor AS-level Adversaries**

Tor AS级攻击者度量扩展视图 cs.NI

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2403.08517v1) [paper-pdf](http://arxiv.org/pdf/2403.08517v1)

**Authors**: Gabriel Karl Gegenhuber, Markus Maier, Florian Holzbauer, Wilfried Mayer, Georg Merzdovnik, Edgar Weippl, Johanna Ullrich

**Abstract**: Tor provides anonymity to millions of users around the globe which has made it a valuable target for malicious actors. As a low-latency anonymity system, it is vulnerable to traffic correlation attacks from strong passive adversaries such as large autonomous systems (ASes). In preliminary work, we have developed a measurement approach utilizing the RIPE Atlas framework -- a network of more than 11,000 probes worldwide -- to infer the risk of deanonymization for IPv4 clients in Germany and the US.   In this paper, we apply our methodology to additional scenarios providing a broader picture of the potential for deanonymization in the Tor network. In particular, we (a) repeat our earlier (2020) measurements in 2022 to observe changes over time, (b) adopt our approach for IPv6 to analyze the risk of deanonymization when using this next-generation Internet protocol, and (c) investigate the current situation in Russia, where censorship has been intensified after the beginning of Russia's full-scale invasion of Ukraine. According to our results, Tor provides user anonymity at consistent quality: While individual numbers vary in dependence of client and destination, we were able to identify ASes with the potential to conduct deanonymization attacks. For clients in Germany and the US, the overall picture, however, has not changed since 2020. In addition, the protocols (IPv4 vs. IPv6) do not significantly impact the risk of deanonymization. Russian users are able to securely evade censorship using Tor. Their general risk of deanonymization is, in fact, lower than in the other investigated countries. Beyond, the few ASes with the potential to successfully perform deanonymization are operated by Western companies, further reducing the risk for Russian users.

摘要: Tor为全球数百万用户提供匿名性，这使其成为恶意行为者的宝贵目标。作为一个低延迟匿名系统，它容易受到大型自治系统（ASes）等强被动攻击的流量相关攻击。在初步工作中，我们开发了一种测量方法，利用RIPE Atlas框架（一个由全球11，000多个探针组成的网络）来推断德国和美国IPv4客户端的去匿名化风险。   在本文中，我们将我们的方法应用于其他场景，为Tor网络中的去匿名化提供了更广泛的前景。特别是，我们（a）在2022年重复我们早期（2020年）的测量，以观察随着时间的变化，（b）采用我们的IPv6方法来分析使用这种下一代互联网协议时的去匿名化风险，（c）调查俄罗斯的现状，俄罗斯全面入侵乌克兰开始后，俄罗斯的审查已经加强。根据我们的研究结果，Tor以一致的质量提供了用户匿名性：虽然个体数量因客户端和目的地而异，但我们能够识别出有可能进行去匿名攻击的AS。然而，对于德国和美国的客户来说，整体情况自2020年以来没有改变。此外，协议（IPv4与IPv6）不会显著影响去匿名化的风险。俄罗斯用户可以使用Tor安全地逃避审查。事实上，他们的一般匿名风险低于其他被调查国家。此外，少数几个有可能成功执行去匿名化的AS由西方公司运营，进一步降低了俄罗斯用户的风险。



## **15. The Philosopher's Stone: Trojaning Plugins of Large Language Models**

哲学家之石：大型语言模型的木马插件 cs.CR

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2312.00374v2) [paper-pdf](http://arxiv.org/pdf/2312.00374v2)

**Authors**: Tian Dong, Minhui Xue, Guoxing Chen, Rayne Holland, Shaofeng Li, Yan Meng, Zhen Liu, Haojin Zhu

**Abstract**: Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers, an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses LLM-enhanced paraphrasing to polish benchmark poisoned datasets. In contrast, in the absence of a dataset, FUSION leverages an over-poisoning procedure to transform a benign adaptor. In our experiments, we first conduct two case studies to demonstrate that a compromised LLM agent can execute malware to control system (e.g., LLM-driven robot) or launch a spear-phishing attack. Then, in terms of targeted misinformation, we show that our attacks provide higher attack effectiveness than the baseline and, for the purpose of attracting downloads, preserve or improve the adapter's utility. Finally, we design and evaluate three potential defenses, yet none proved entirely effective in safeguarding against our attacks.

摘要: 开源大型语言模型（LLM）最近受到了欢迎，因为它们与专有LLM相当的性能。为了有效地完成领域专用任务，开源LLM可以使用低秩适配器进行优化，而无需昂贵的加速器。然而，目前还不清楚是否可以利用低秩适配器来控制LLM。为了解决这一差距，我们证明了受感染的适配器可以诱导，在特定的触发器，LLM输出由对手定义的内容，甚至恶意使用工具。为了训练木马适配器，我们提出了两种新的攻击，POLISHED和FUSION，改进了先前的方法。POLISHED使用LLM增强的释义来抛光基准中毒数据集。相反，在没有数据集的情况下，FUSION利用过度中毒过程来转换良性适配器。在我们的实验中，我们首先进行了两个案例研究，以证明一个受损的LLM代理可以执行恶意软件来控制系统（例如，LLM驱动的机器人）或发起鱼叉式网络钓鱼攻击。然后，在有针对性的错误信息方面，我们表明我们的攻击提供了比基线更高的攻击效果，并且为了吸引下载，保留或改进适配器的实用性。最后，我们设计并评估了三种潜在的防御措施，但没有一种被证明完全有效地防御我们的攻击。



## **16. Fast Inference of Removal-Based Node Influence**

基于删除的节点影响力快速推断 cs.LG

To be published in the Web Conference 2024

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2403.08333v1) [paper-pdf](http://arxiv.org/pdf/2403.08333v1)

**Authors**: Weikai Li, Zhiping Xiao, Xiao Luo, Yizhou Sun

**Abstract**: Graph neural networks (GNNs) are widely utilized to capture the information spreading patterns in graphs. While remarkable performance has been achieved, there is a new trending topic of evaluating node influence. We propose a new method of evaluating node influence, which measures the prediction change of a trained GNN model caused by removing a node. A real-world application is, "In the task of predicting Twitter accounts' polarity, had a particular account been removed, how would others' polarity change?". We use the GNN as a surrogate model whose prediction could simulate the change of nodes or edges caused by node removal. To obtain the influence for every node, a straightforward way is to alternately remove every node and apply the trained GNN on the modified graph. It is reliable but time-consuming, so we need an efficient method. The related lines of work, such as graph adversarial attack and counterfactual explanation, cannot directly satisfy our needs, since they do not focus on the global influence score for every node. We propose an efficient and intuitive method, NOde-Removal-based fAst GNN inference (NORA), which uses the gradient to approximate the node-removal influence. It only costs one forward propagation and one backpropagation to approximate the influence score for all nodes. Extensive experiments on six datasets and six GNN models verify the effectiveness of NORA. Our code is available at https://github.com/weikai-li/NORA.git.

摘要: 图神经网络(GNN)被广泛用于捕捉图中的信息传播模式。在取得显著性能的同时，评价节点影响力成为一个新的热门话题。我们提出了一种新的评估节点影响力的方法，该方法衡量了一个训练好的GNN模型在移除一个节点后的预测变化。一个真实世界的应用程序是，“在预测Twitter帐户的极性的任务中，如果某个特定的帐户被删除，其他帐户的极性会如何变化？”我们使用GNN作为代理模型，它的预测可以模拟节点移除引起的节点或边的变化。为了获得对每个节点的影响，一种简单的方法是交替删除每个节点，并将训练好的GNN应用到修改后的图上。它是可靠的，但很耗时，所以我们需要一种有效的方法。相关的工作，如图对抗攻击和反事实解释，不能直接满足我们的需求，因为它们不关注每个节点的全局影响力得分。我们提出了一种高效、直观的方法--基于节点移除的快速GNN推理(NORA)，它使用梯度来逼近节点移除的影响。它只需一次前向传播和一次反向传播就可以近似计算所有节点的影响分数。在六个数据集和六个GNN模型上的大量实验验证了NORA的有效性。我们的代码可以在https://github.com/weikai-li/NORA.git.上找到



## **17. TSFool: Crafting Highly-Imperceptible Adversarial Time Series through Multi-Objective Attack**

TSFool：通过多目标攻击来制作高度不可感知的对抗时间序列 cs.LG

22 pages, 16 figures

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2209.06388v3) [paper-pdf](http://arxiv.org/pdf/2209.06388v3)

**Authors**: Yanyun Wang, Dehui Du, Haibo Hu, Zi Liang, Yuanhao Liu

**Abstract**: Recent years have witnessed the success of recurrent neural network (RNN) models in time series classification (TSC). However, neural networks (NNs) are vulnerable to adversarial samples, which cause real-life adversarial attacks that undermine the robustness of AI models. To date, most existing attacks target at feed-forward NNs and image recognition tasks, but they cannot perform well on RNN-based TSC. This is due to the cyclical computation of RNN, which prevents direct model differentiation. In addition, the high visual sensitivity of time series to perturbations also poses challenges to local objective optimization of adversarial samples. In this paper, we propose an efficient method called TSFool to craft highly-imperceptible adversarial time series for RNN-based TSC. The core idea is a new global optimization objective known as "Camouflage Coefficient" that captures the imperceptibility of adversarial samples from the class distribution. Based on this, we reduce the adversarial attack problem to a multi-objective optimization problem that enhances the perturbation quality. Furthermore, to speed up the optimization process, we propose to use a representation model for RNN to capture deeply embedded vulnerable samples whose features deviate from the latent manifold. Experiments on 11 UCR and UEA datasets showcase that TSFool significantly outperforms six white-box and three black-box benchmark attacks in terms of effectiveness, efficiency and imperceptibility from various perspectives including standard measure, human study and real-world defense.

摘要: 近年来，递归神经网络(RNN)模型在时间序列分类(TSC)中取得了成功。然而，神经网络很容易受到敌意样本的攻击，这些样本会导致现实生活中的对抗性攻击，从而削弱人工智能模型的健壮性。到目前为止，大多数现有的攻击都是针对前馈神经网络和图像识别任务，但它们在基于RNN的TSC上不能很好地执行。这是由于RNN的循环计算，这阻止了直接的模型区分。此外，时间序列对扰动的高度视觉敏感性也给对抗性样本的局部目标优化带来了挑战。在本文中，我们提出了一种称为TSFool的有效方法来为基于RNN的TSC生成高度不可察觉的对抗性时间序列。其核心思想是一种新的全局优化目标，称为伪装系数，它从类别分布中捕捉敌方样本的不可见性。在此基础上，我们将对抗性攻击问题归结为一个提高扰动质量的多目标优化问题。此外，为了加快优化过程，我们提出了使用RNN的表示模型来捕获深度嵌入的特征偏离潜在流形的易受攻击的样本。在11个UCR和UEA数据集上的实验表明，TSFool在有效性、效率和不可感知性方面都明显优于6个白盒和3个黑盒基准攻击，包括标准度量、人类研究和现实世界防御。



## **18. Attack Deterministic Conditional Image Generative Models for Diverse and Controllable Generation**

攻击确定性条件图像生成模型的多样性可控生成 cs.CV

9 pages, 7 figures, accepted by AAAI24

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2403.08294v1) [paper-pdf](http://arxiv.org/pdf/2403.08294v1)

**Authors**: Tianyi Chu, Wei Xing, Jiafu Chen, Zhizhong Wang, Jiakai Sun, Lei Zhao, Haibo Chen, Huaizhong Lin

**Abstract**: Existing generative adversarial network (GAN) based conditional image generative models typically produce fixed output for the same conditional input, which is unreasonable for highly subjective tasks, such as large-mask image inpainting or style transfer. On the other hand, GAN-based diverse image generative methods require retraining/fine-tuning the network or designing complex noise injection functions, which is computationally expensive, task-specific, or struggle to generate high-quality results. Given that many deterministic conditional image generative models have been able to produce high-quality yet fixed results, we raise an intriguing question: is it possible for pre-trained deterministic conditional image generative models to generate diverse results without changing network structures or parameters? To answer this question, we re-examine the conditional image generation tasks from the perspective of adversarial attack and propose a simple and efficient plug-in projected gradient descent (PGD) like method for diverse and controllable image generation. The key idea is attacking the pre-trained deterministic generative models by adding a micro perturbation to the input condition. In this way, diverse results can be generated without any adjustment of network structures or fine-tuning of the pre-trained models. In addition, we can also control the diverse results to be generated by specifying the attack direction according to a reference text or image. Our work opens the door to applying adversarial attack to low-level vision tasks, and experiments on various conditional image generation tasks demonstrate the effectiveness and superiority of the proposed method.

摘要: 现有的基于生成对抗网络（GAN）的条件图像生成模型通常对相同的条件输入产生固定的输出，这对于高度主观的任务，如大掩模图像修复或风格转换是不合理的。另一方面，基于GAN的各种图像生成方法需要重新训练/微调网络或设计复杂的噪声注入函数，这在计算上昂贵，任务特定，或难以生成高质量的结果。鉴于许多确定性条件图像生成模型已经能够产生高质量但固定的结果，我们提出了一个有趣的问题：预训练的确定性条件图像生成模型是否可能在不改变网络结构或参数的情况下生成不同的结果？为了回答这个问题，我们从对抗攻击的角度重新审视了条件图像生成任务，并提出了一种简单有效的插件投影梯度下降（PGD）方法，用于多样和可控的图像生成。其核心思想是通过在输入条件中添加微扰动来攻击预先训练好的确定性生成模型。通过这种方式，可以生成不同的结果，而无需对网络结构进行任何调整或对预训练模型进行微调。此外，我们还可以通过根据参考文本或图像指定攻击方向来控制要生成的不同结果。我们的工作为对抗性攻击应用于低层视觉任务打开了大门，并在各种条件图像生成任务上的实验表明了所提出方法的有效性和优越性。



## **19. Versatile Defense Against Adversarial Attacks on Image Recognition**

图像识别中对抗攻击的通用防御 cs.CV

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2403.08170v1) [paper-pdf](http://arxiv.org/pdf/2403.08170v1)

**Authors**: Haibo Zhang, Zhihua Yao, Kouichi Sakurai

**Abstract**: Adversarial attacks present a significant security risk to image recognition tasks. Defending against these attacks in a real-life setting can be compared to the way antivirus software works, with a key consideration being how well the defense can adapt to new and evolving attacks. Another important factor is the resources involved in terms of time and cost for training defense models and updating the model database. Training many models that are specific to each type of attack can be time-consuming and expensive. Ideally, we should be able to train one single model that can handle a wide range of attacks. It appears that a defense method based on image-to-image translation may be capable of this. The proposed versatile defense approach in this paper only requires training one model to effectively resist various unknown adversarial attacks. The trained model has successfully improved the classification accuracy from nearly zero to an average of 86%, performing better than other defense methods proposed in prior studies. When facing the PGD attack and the MI-FGSM attack, versatile defense model even outperforms the attack-specific models trained based on these two attacks. The robustness check also shows that our versatile defense model performs stably regardless with the attack strength.

摘要: 对抗性攻击给图像识别任务带来了重大的安全风险。在现实环境中防御这些攻击可以与防病毒软件的工作方式进行比较，关键的考虑因素是防御系统能够适应新的和不断发展的攻击。另一个重要因素是训练防御模型和更新模型数据库所需的时间和成本。训练特定于每种攻击类型的许多模型既耗时又昂贵。理想情况下，我们应该能够训练一个可以处理广泛攻击的模型。看来，基于图像到图像翻译的防御方法可能能够做到这一点。本文提出的通用防御方法只需要训练一个模型就能有效地抵抗各种未知的对抗攻击。经过训练的模型成功地将分类准确率从接近零提高到平均86%，性能优于先前研究中提出的其他防御方法。在面对PGD攻击和MI—FGSM攻击时，通用防御模型甚至优于基于这两种攻击训练的攻击特定模型。鲁棒性检验还表明，我们的通用防御模型在攻击强度的情况下都能稳定地运行。



## **20. Information Leakage through Physical Layer Supply Voltage Coupling Vulnerability**

通过物理层电源电压耦合漏洞的信息泄漏 cs.CR

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2403.08132v1) [paper-pdf](http://arxiv.org/pdf/2403.08132v1)

**Authors**: Sahan Sanjaya, Aruna Jayasena, Prabhat Mishra

**Abstract**: Side-channel attacks exploit variations in non-functional behaviors to expose sensitive information across security boundaries. Existing methods leverage side-channels based on power consumption, electromagnetic radiation, silicon substrate coupling, and channels created by malicious implants. Power-based side-channel attacks are widely known for extracting information from data processed within a device while assuming that an attacker has physical access or the ability to modify the device. In this paper, we introduce a novel side-channel vulnerability that leaks data-dependent power variations through physical layer supply voltage coupling (PSVC). Unlike traditional power side-channel attacks, the proposed vulnerability allows an adversary to mount an attack and extract information without modifying the device. We assess the effectiveness of PSVC vulnerability through three case studies, demonstrating several end-to-end attacks on general-purpose microcontrollers with varying adversary capabilities. These case studies provide evidence for the existence of PSVC vulnerability, its applicability for on-chip as well as on-board side-channel attacks, and how it can eliminate the need for physical access to the target device, making it applicable to any off-the-shelf hardware. Our experiments also reveal that designing devices to operate at the lowest operational voltage significantly reduces the risk of PSVC side-channel vulnerability.

摘要: 侧通道攻击利用非功能行为的变化来跨安全边界暴露敏感信息。现有方法利用基于功率消耗、电磁辐射、硅衬底耦合和由恶意植入创建的通道的副通道。众所周知，基于电源的侧通道攻击从设备内处理的数据中提取信息，同时假设攻击者具有物理访问权限或修改设备的能力。在本文中，我们介绍了一种新的旁通道漏洞，它通过物理层电源电压耦合(PSVC)泄漏与数据相关的功率变化。与传统的电源侧通道攻击不同，提出的漏洞允许攻击者在不修改设备的情况下发动攻击并提取信息。我们通过三个案例研究来评估PSVC漏洞的有效性，展示了针对具有不同攻击能力的通用微控制器的几种端到端攻击。这些案例研究提供了PSVC漏洞存在的证据，它适用于片上和板载侧通道攻击，以及它如何消除对目标设备的物理访问需求，使其适用于任何现成的硬件。我们的实验还表明，将器件设计为工作在最低工作电压下，显著降低了PSVC侧通道漏洞的风险。



## **21. Quantifying and Mitigating Privacy Risks for Tabular Generative Models**

表生成模型的隐私风险量化与缓解 cs.LG

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2403.07842v1) [paper-pdf](http://arxiv.org/pdf/2403.07842v1)

**Authors**: Chaoyi Zhu, Jiayi Tang, Hans Brouwer, Juan F. Pérez, Marten van Dijk, Lydia Y. Chen

**Abstract**: Synthetic data from generative models emerges as the privacy-preserving data-sharing solution. Such a synthetic data set shall resemble the original data without revealing identifiable private information. The backbone technology of tabular synthesizers is rooted in image generative models, ranging from Generative Adversarial Networks (GANs) to recent diffusion models. Recent prior work sheds light on the utility-privacy tradeoff on tabular data, revealing and quantifying privacy risks on synthetic data. We first conduct an exhaustive empirical analysis, highlighting the utility-privacy tradeoff of five state-of-the-art tabular synthesizers, against eight privacy attacks, with a special focus on membership inference attacks. Motivated by the observation of high data quality but also high privacy risk in tabular diffusion, we propose DP-TLDM, Differentially Private Tabular Latent Diffusion Model, which is composed of an autoencoder network to encode the tabular data and a latent diffusion model to synthesize the latent tables. Following the emerging f-DP framework, we apply DP-SGD to train the auto-encoder in combination with batch clipping and use the separation value as the privacy metric to better capture the privacy gain from DP algorithms. Our empirical evaluation demonstrates that DP-TLDM is capable of achieving a meaningful theoretical privacy guarantee while also significantly enhancing the utility of synthetic data. Specifically, compared to other DP-protected tabular generative models, DP-TLDM improves the synthetic quality by an average of 35% in data resemblance, 15% in the utility for downstream tasks, and 50% in data discriminability, all while preserving a comparable level of privacy risk.

摘要: 来自生成性模型的合成数据成为保护隐私的数据共享解决方案。这样的合成数据集应与原始数据相似，而不会泄露可识别的私人信息。表格合成器的核心技术植根于图像生成模型，从生成性对抗网络(GANS)到最近的扩散模型。最近的先前工作揭示了表格数据的效用和隐私权衡，揭示并量化了合成数据的隐私风险。我们首先进行了详尽的实证分析，重点介绍了五种最先进的表格合成器在对抗八种隐私攻击时的效用-隐私权衡，并特别关注了成员关系推理攻击。针对表格扩散的高数据质量和高隐私风险的特点，本文提出了一种差异私有表格潜在扩散模型DP-tldm，该模型由一个自动编码网络对表格数据进行编码和一个潜在扩散模型来合成潜在表格。遵循新兴的f-DP框架，我们使用DP-SGD结合批量裁剪来训练自动编码器，并使用分离值作为隐私度量，以更好地捕捉DP算法的隐私收益。我们的实验评估表明，DP-tldm能够实现有意义的理论隐私保障，同时也显著提高了合成数据的实用性。具体地说，与其他DP保护的表格生成模型相比，DP-tldm在数据相似性方面平均提高了35%的合成质量，在下游任务的效用方面提高了15%，在数据辨别性方面提高了50%，所有这些都保持了可比的隐私风险水平。



## **22. Robustifying Point Cloud Networks by Refocusing**

通过重新聚焦实现点云网络的规模化 cs.CV

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2308.05525v3) [paper-pdf](http://arxiv.org/pdf/2308.05525v3)

**Authors**: Meir Yossef Levi, Guy Gilboa

**Abstract**: The ability to cope with out-of-distribution (OOD) corruptions and adversarial attacks is crucial in real-world safety-demanding applications. In this study, we develop a general mechanism to increase neural network robustness based on focus analysis.   Recent studies have revealed the phenomenon of \textit{Overfocusing}, which leads to a performance drop. When the network is primarily influenced by small input regions, it becomes less robust and prone to misclassify under noise and corruptions.   However, quantifying overfocusing is still vague and lacks clear definitions. Here, we provide a mathematical definition of \textbf{focus}, \textbf{overfocusing} and \textbf{underfocusing}. The notions are general, but in this study, we specifically investigate the case of 3D point clouds.   We observe that corrupted sets result in a biased focus distribution compared to the clean training set.   We show that as focus distribution deviates from the one learned in the training phase - classification performance deteriorates.   We thus propose a parameter-free \textbf{refocusing} algorithm that aims to unify all corruptions under the same distribution.   We validate our findings on a 3D zero-shot classification task, achieving SOTA in robust 3D classification on ModelNet-C dataset, and in adversarial defense against Shape-Invariant attack. Code is available in: https://github.com/yossilevii100/refocusing.

摘要: 应对分发外(OOD)损坏和敌意攻击的能力在现实世界对安全要求苛刻的应用程序中至关重要。在这项研究中，我们提出了一种基于焦点分析的提高神经网络健壮性的通用机制。最近的研究发现了文本{过度聚焦}的现象，这会导致性能下降。当网络主要受到小输入区域的影响时，它变得不那么健壮，并且容易在噪声和损坏下被错误分类。然而，对过度关注的量化仍然含糊不清，缺乏明确的定义。在这里，我们给出了\extbf{焦点}、\extbf{过度聚焦}和\extbf{欠聚焦}的数学定义。这些概念是一般的，但在本研究中，我们专门研究3D点云的情况。我们观察到，与干净的训练集相比，损坏的集导致了偏向的焦点分布。我们表明，随着焦点分布与在训练阶段学习的焦点分布背离，分类性能会恶化。因此，我们提出了一种无参数的Textbf{重新聚焦}算法，旨在统一同一分布下的所有损坏。我们在3D零镜头分类任务上验证了我们的发现，在ModelNet-C数据集上实现了稳健的3D分类，并在对抗形状不变攻击中实现了SOTA。代码可在以下位置获得：https://github.com/yossilevii100/refocusing.



## **23. Analyzing Adversarial Attacks on Sequence-to-Sequence Relevance Models**

序列间相关模型的对抗攻击分析 cs.IR

13 pages, 3 figures, Accepted at ECIR 2024 as a Full Paper

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2403.07654v1) [paper-pdf](http://arxiv.org/pdf/2403.07654v1)

**Authors**: Andrew Parry, Maik Fröbe, Sean MacAvaney, Martin Potthast, Matthias Hagen

**Abstract**: Modern sequence-to-sequence relevance models like monoT5 can effectively capture complex textual interactions between queries and documents through cross-encoding. However, the use of natural language tokens in prompts, such as Query, Document, and Relevant for monoT5, opens an attack vector for malicious documents to manipulate their relevance score through prompt injection, e.g., by adding target words such as true. Since such possibilities have not yet been considered in retrieval evaluation, we analyze the impact of query-independent prompt injection via manually constructed templates and LLM-based rewriting of documents on several existing relevance models. Our experiments on the TREC Deep Learning track show that adversarial documents can easily manipulate different sequence-to-sequence relevance models, while BM25 (as a typical lexical model) is not affected. Remarkably, the attacks also affect encoder-only relevance models (which do not rely on natural language prompt tokens), albeit to a lesser extent.

摘要: 现代的序列到序列相关性模型（如monoT5）可以通过交叉编码有效地捕获查询和文档之间复杂的文本交互。然而，在提示中使用自然语言令牌，例如monoT5的查询、文档和相关，打开了恶意文档的攻击向量，以通过提示注入来操纵它们的相关性得分，例如，通过添加目标词，如true。由于这些可能性还没有被考虑在检索评估，我们分析了通过手动构建模板和基于LLM重写文档对几个现有的相关模型的查询独立提示注入的影响。我们在TREC深度学习轨道上的实验表明，对抗文档可以轻松地操纵不同的序列到序列相关性模型，而BM25（作为典型的词汇模型）不受影响。值得注意的是，这些攻击还影响了仅编码器的相关模型（不依赖于自然语言提示符），尽管程度较小。



## **24. Visual Privacy Auditing with Diffusion Models**

基于扩散模型的视觉隐私审计 cs.LG

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2403.07588v1) [paper-pdf](http://arxiv.org/pdf/2403.07588v1)

**Authors**: Kristian Schwethelm, Johannes Kaiser, Moritz Knolle, Daniel Rueckert, Georgios Kaissis, Alexander Ziller

**Abstract**: Image reconstruction attacks on machine learning models pose a significant risk to privacy by potentially leaking sensitive information. Although defending against such attacks using differential privacy (DP) has proven effective, determining appropriate DP parameters remains challenging. Current formal guarantees on data reconstruction success suffer from overly theoretical assumptions regarding adversary knowledge about the target data, particularly in the image domain. In this work, we empirically investigate this discrepancy and find that the practicality of these assumptions strongly depends on the domain shift between the data prior and the reconstruction target. We propose a reconstruction attack based on diffusion models (DMs) that assumes adversary access to real-world image priors and assess its implications on privacy leakage under DP-SGD. We show that (1) real-world data priors significantly influence reconstruction success, (2) current reconstruction bounds do not model the risk posed by data priors well, and (3) DMs can serve as effective auditing tools for visualizing privacy leakage.

摘要: 对机器学习模型的图像重建攻击可能会泄露敏感信息，从而对隐私构成重大风险。尽管使用差分隐私（DP）防御此类攻击已被证明是有效的，但确定适当的DP参数仍然具有挑战性。当前对数据重建成功的正式保证受到关于对手关于目标数据的知识的过度理论假设的影响，特别是在图像领域。在这项工作中，我们经验性地调查了这种差异，并发现这些假设的实用性强烈依赖于先前数据和重建目标之间的域移位。我们提出了一种基于扩散模型（DM）的重构攻击，假设对手访问真实世界的图像先验，并评估其在DP—SGD下的隐私泄漏的影响。我们发现（1）真实世界的数据先验显著影响重建成功；（2）当前的重建边界不能很好地模拟数据先验带来的风险；（3）DM可以作为有效的审计工具来可视化隐私泄漏。



## **25. Improving deep learning with prior knowledge and cognitive models: A survey on enhancing explainability, adversarial robustness and zero-shot learning**

利用先验知识和认知模型改进深度学习：关于增强可解释性、对抗鲁棒性和零射击学习的调查 cs.LG

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.07078v1) [paper-pdf](http://arxiv.org/pdf/2403.07078v1)

**Authors**: Fuseinin Mumuni, Alhassan Mumuni

**Abstract**: We review current and emerging knowledge-informed and brain-inspired cognitive systems for realizing adversarial defenses, eXplainable Artificial Intelligence (XAI), and zero-shot or few-short learning. Data-driven deep learning models have achieved remarkable performance and demonstrated capabilities surpassing human experts in many applications. Yet, their inability to exploit domain knowledge leads to serious performance limitations in practical applications. In particular, deep learning systems are exposed to adversarial attacks, which can trick them into making glaringly incorrect decisions. Moreover, complex data-driven models typically lack interpretability or explainability, i.e., their decisions cannot be understood by human subjects. Furthermore, models are usually trained on standard datasets with a closed-world assumption. Hence, they struggle to generalize to unseen cases during inference in practical open-world environments, thus, raising the zero- or few-shot generalization problem. Although many conventional solutions exist, explicit domain knowledge, brain-inspired neural network and cognitive architectures offer powerful new dimensions towards alleviating these problems. Prior knowledge is represented in appropriate forms and incorporated in deep learning frameworks to improve performance. Brain-inspired cognition methods use computational models that mimic the human mind to enhance intelligent behavior in artificial agents and autonomous robots. Ultimately, these models achieve better explainability, higher adversarial robustness and data-efficient learning, and can, in turn, provide insights for cognitive science and neuroscience-that is, to deepen human understanding on how the brain works in general, and how it handles these problems.

摘要: 我们回顾了当前和新兴的知识通知和大脑启发的认知系统，以实现对抗防御，可扩展人工智能（XAI）和零射击或短时间学习。数据驱动的深度学习模型已经取得了卓越的性能，并在许多应用中展示了超越人类专家的能力。然而，它们无法利用领域知识导致了在实际应用中的严重性能限制。特别是，深度学习系统容易受到对抗性攻击，这可能会诱使它们做出令人震惊的错误决策。此外，复杂的数据驱动模型通常缺乏可解释性或可解释性，即，他们的决定是人类无法理解的。此外，模型通常在标准数据集上进行训练，并采用封闭世界假设。因此，在实际开放世界环境中的推理过程中，他们很难推广到看不见的情况，从而提出了零或少镜头推广问题。尽管存在许多传统的解决方案，但显式领域知识、脑启发神经网络和认知架构为缓解这些问题提供了强大的新维度。先验知识以适当的形式表示，并纳入深度学习框架，以提高性能。脑启发认知方法使用模拟人类思维的计算模型来增强人工智能和自主机器人的智能行为。最终，这些模型实现了更好的可解释性、更高的对抗性鲁棒性和数据高效学习，进而为认知科学和神经科学提供了见解，也就是说，加深了人类对大脑一般如何工作以及如何处理这些问题的理解。



## **26. Enhancing Adversarial Training with Prior Knowledge Distillation for Robust Image Compression**

基于先验知识提取增强对抗训练的鲁棒图像压缩 eess.IV

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06700v1) [paper-pdf](http://arxiv.org/pdf/2403.06700v1)

**Authors**: Cao Zhi, Bao Youneng, Meng Fanyang, Li Chao, Tan Wen, Wang Genhong, Liang Yongsheng

**Abstract**: Deep neural network-based image compression (NIC) has achieved excellent performance, but NIC method models have been shown to be susceptible to backdoor attacks. Adversarial training has been validated in image compression models as a common method to enhance model robustness. However, the improvement effect of adversarial training on model robustness is limited. In this paper, we propose a prior knowledge-guided adversarial training framework for image compression models. Specifically, first, we propose a gradient regularization constraint for training robust teacher models. Subsequently, we design a knowledge distillation based strategy to generate a priori knowledge from the teacher model to the student model for guiding adversarial training. Experimental results show that our method improves the reconstruction quality by about 9dB when the Kodak dataset is elected as the backdoor attack object for psnr attack. Compared with Ma2023, our method has a 5dB higher PSNR output at high bitrate points.

摘要: 基于深度神经网络的图像压缩（NIC）已经取得了出色的性能，但NIC方法模型已被证明容易受到后门攻击。对抗训练已在图像压缩模型中被验证为增强模型鲁棒性的常用方法。然而，对抗训练对模型鲁棒性的改善效果有限。在本文中，我们提出了一个先验知识引导的对抗训练框架的图像压缩模型。具体来说，首先，我们提出了一个梯度正则化约束来训练鲁棒教师模型。然后，我们设计了一个基于知识提炼的策略，生成从教师模型到学生模型的先验知识，指导对抗性训练。实验结果表明，当Kodak数据集被选为后门攻击对象时，该方法的重建质量提高了约9dB。与Ma2023相比，我们的方法在高比特率点有5dB的PSNR输出。



## **27. PCLD: Point Cloud Layerwise Diffusion for Adversarial Purification**

PCLD：对抗纯化的点云分层扩散 cs.CV

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06698v1) [paper-pdf](http://arxiv.org/pdf/2403.06698v1)

**Authors**: Mert Gulsen, Batuhan Cengiz, Yusuf H. Sahin, Gozde Unal

**Abstract**: Point clouds are extensively employed in a variety of real-world applications such as robotics, autonomous driving and augmented reality. Despite the recent success of point cloud neural networks, especially for safety-critical tasks, it is essential to also ensure the robustness of the model. A typical way to assess a model's robustness is through adversarial attacks, where test-time examples are generated based on gradients to deceive the model. While many different defense mechanisms are studied in 2D, studies on 3D point clouds have been relatively limited in the academic field. Inspired from PointDP, which denoises the network inputs by diffusion, we propose Point Cloud Layerwise Diffusion (PCLD), a layerwise diffusion based 3D point cloud defense strategy. Unlike PointDP, we propagated the diffusion denoising after each layer to incrementally enhance the results. We apply our defense method to different types of commonly used point cloud models and adversarial attacks to evaluate its robustness. Our experiments demonstrate that the proposed defense method achieved results that are comparable to or surpass those of existing methodologies, establishing robustness through a novel technique. Code is available at https://github.com/batuceng/diffusion-layer-robustness-pc.

摘要: 点云被广泛应用于各种现实世界的应用，如机器人、自动驾驶和增强现实。尽管点云神经网络最近取得了成功，特别是对于安全关键任务，但确保模型的鲁棒性也至关重要。评估模型鲁棒性的一种典型方法是通过对抗攻击，其中基于梯度生成测试时示例以欺骗模型。虽然许多不同的防御机制在2D研究，3D点云的研究在学术领域相对有限。借鉴PointDP通过扩散对网络输入进行去噪的思想，提出了一种基于分层扩散的三维点云防御策略——点云分层扩散（PCLD）。与PointDP不同的是，我们在每一层之后传播扩散去噪，以逐步增强结果。我们将我们的防御方法应用到不同类型的常用点云模型和对抗攻击，以评估其鲁棒性。我们的实验表明，所提出的防御方法取得了相当或超过现有的方法，建立了鲁棒性，通过一种新的技术。代码可在www.example.com获得。



## **28. PeerAiD: Improving Adversarial Distillation from a Specialized Peer Tutor**

PeerAiD：从专业的同伴导师改善对抗蒸馏 cs.LG

Accepted to CVPR 2024

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06668v1) [paper-pdf](http://arxiv.org/pdf/2403.06668v1)

**Authors**: Jaewon Jung, Hongsun Jang, Jaeyong Song, Jinho Lee

**Abstract**: Adversarial robustness of the neural network is a significant concern when it is applied to security-critical domains. In this situation, adversarial distillation is a promising option which aims to distill the robustness of the teacher network to improve the robustness of a small student network. Previous works pretrain the teacher network to make it robust to the adversarial examples aimed at itself. However, the adversarial examples are dependent on the parameters of the target network. The fixed teacher network inevitably degrades its robustness against the unseen transferred adversarial examples which targets the parameters of the student network in the adversarial distillation process. We propose PeerAiD to make a peer network learn the adversarial examples of the student network instead of adversarial examples aimed at itself. PeerAiD is an adversarial distillation that trains the peer network and the student network simultaneously in order to make the peer network specialized for defending the student network. We observe that such peer networks surpass the robustness of pretrained robust teacher network against student-attacked adversarial samples. With this peer network and adversarial distillation, PeerAiD achieves significantly higher robustness of the student network with AutoAttack (AA) accuracy up to 1.66%p and improves the natural accuracy of the student network up to 4.72%p with ResNet-18 and TinyImageNet dataset.

摘要: 当神经网络应用于安全关键领域时，它的对抗鲁棒性是一个重要的问题。在这种情况下，对抗性提取是一种很有前途的选择，其目的是提取教师网络的鲁棒性，以提高小型学生网络的鲁棒性。以前的作品预先训练教师网络，使其对针对自身的对抗性例子具有稳健性。然而，对抗性示例依赖于目标网络的参数。固定的教师网络在对抗性提炼过程中不可避免地降低了其对不可见的对抗性样本的鲁棒性。我们提出PeerAiD使对等网络学习学生网络的对抗性示例，而不是针对自身的对抗性示例。PeerAiD是一种对抗性蒸馏，它同时训练对等网络和学生网络，使对等网络专门用于学生网络的防御。我们观察到，这样的对等网络超过了预训练的鲁棒教师网络对学生攻击的对抗样本的鲁棒性。通过这种对等网络和对抗蒸馏，PeerAiD实现了学生网络的更高鲁棒性，AutoAttack（AA）精度高达1.66%p，并通过ResNet—18和TinyImageNet数据集将学生网络的自然精度提高到4.72%p。



## **29. epsilon-Mesh Attack: A Surface-based Adversarial Point Cloud Attack for Facial Expression Recognition**

Epsilon-Mesh攻击：一种基于表面的人脸表情识别对抗性点云攻击 cs.CV

Accepted at 18th IEEE International Conference on Automatic Face &  Gesture Recognition (FG 2024)

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06661v1) [paper-pdf](http://arxiv.org/pdf/2403.06661v1)

**Authors**: Batuhan Cengiz, Mert Gulsen, Yusuf H. Sahin, Gozde Unal

**Abstract**: Point clouds and meshes are widely used 3D data structures for many computer vision applications. While the meshes represent the surfaces of an object, point cloud represents sampled points from the surface which is also the output of modern sensors such as LiDAR and RGB-D cameras. Due to the wide application area of point clouds and the recent advancements in deep neural networks, studies focusing on robust classification of the 3D point cloud data emerged. To evaluate the robustness of deep classifier networks, a common method is to use adversarial attacks where the gradient direction is followed to change the input slightly. The previous studies on adversarial attacks are generally evaluated on point clouds of daily objects. However, considering 3D faces, these adversarial attacks tend to affect the person's facial structure more than the desired amount and cause malformation. Specifically for facial expressions, even a small adversarial attack can have a significant effect on the face structure. In this paper, we suggest an adversarial attack called $\epsilon$-Mesh Attack, which operates on point cloud data via limiting perturbations to be on the mesh surface. We also parameterize our attack by $\epsilon$ to scale the perturbation mesh. Our surface-based attack has tighter perturbation bounds compared to $L_2$ and $L_\infty$ norm bounded attacks that operate on unit-ball. Even though our method has additional constraints, our experiments on CoMA, Bosphorus and FaceWarehouse datasets show that $\epsilon$-Mesh Attack (Perpendicular) successfully confuses trained DGCNN and PointNet models $99.72\%$ and $97.06\%$ of the time, with indistinguishable facial deformations. The code is available at https://github.com/batuceng/e-mesh-attack.

摘要: 点云和网格是许多计算机视觉应用中广泛使用的三维数据结构。虽然网格代表物体的表面，但点云代表表面的采样点，这也是现代传感器（如LiDAR和RGB—D相机）的输出。由于点云的广泛应用领域和深度神经网络的最新进展，重点关注3D点云数据的鲁棒分类的研究出现了。为了评估深度分类器网络的鲁棒性，一种常见的方法是使用对抗攻击，其中遵循梯度方向来轻微改变输入。以往对对抗攻击的研究一般都是在日常物体的点云上进行的。然而，考虑到3D面部，这些对抗性攻击往往会影响人的面部结构超过预期的量，并导致畸形。特别是对于面部表情，即使是一个小的对抗性攻击也会对面部结构产生重大影响。在本文中，我们提出了一种名为$\rash $—Mesh攻击的对抗性攻击，它通过限制网格表面上的扰动来对点云数据进行操作。我们还通过$\n $参数化我们的攻击，以缩放扰动网格。我们的基于表面的攻击具有更严格的扰动边界相比，操作在单位球上的$L_2 $和$L_\infty $范数有界攻击。尽管我们的方法有额外的限制，但我们在CoMA、Bosphorus和FaceWarehouse数据集上的实验表明，$\n $—Mesh Attack（Percularular）成功地混淆了经过训练的DGCNN和PointNet模型，其中面部变形为99.72美元和97.06美元。该代码可在www.example.com获得。



## **30. Real is not True: Backdoor Attacks Against Deepfake Detection**

真实不是真的：对深伪检测的后门攻击 cs.CR

BigDIA 2023

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06610v1) [paper-pdf](http://arxiv.org/pdf/2403.06610v1)

**Authors**: Hong Sun, Ziqiang Li, Lei Liu, Bin Li

**Abstract**: The proliferation of malicious deepfake applications has ignited substantial public apprehension, casting a shadow of doubt upon the integrity of digital media. Despite the development of proficient deepfake detection mechanisms, they persistently demonstrate pronounced vulnerability to an array of attacks. It is noteworthy that the pre-existing repertoire of attacks predominantly comprises adversarial example attack, predominantly manifesting during the testing phase. In the present study, we introduce a pioneering paradigm denominated as Bad-Deepfake, which represents a novel foray into the realm of backdoor attacks levied against deepfake detectors. Our approach hinges upon the strategic manipulation of a delimited subset of the training data, enabling us to wield disproportionate influence over the operational characteristics of a trained model. This manipulation leverages inherent frailties inherent to deepfake detectors, affording us the capacity to engineer triggers and judiciously select the most efficacious samples for the construction of the poisoned set. Through the synergistic amalgamation of these sophisticated techniques, we achieve an remarkable performance-a 100% attack success rate (ASR) against extensively employed deepfake detectors.

摘要: 恶意deepfake应用程序的扩散引发了公众的强烈担忧，对数字媒体的完整性投下了怀疑的阴影。尽管开发了熟练的deepfake检测机制，但它们始终表现出对一系列攻击的明显脆弱性。值得注意的是，先前存在的攻击库主要包括对抗性示例攻击，主要表现在测试阶段。在本研究中，我们引入了一个名为Bad—Deepfake的开创性范式，它代表了对deepfake检测器的后门攻击领域的一种新尝试。我们的方法取决于对训练数据的限定子集的策略性操作，使我们能够对训练模型的操作特性施加不成比例的影响。这种操作利用了deepfake探测器固有的固有弱点，使我们能够设计触发器，并明智地选择最有效的样本来构建中毒集。通过这些复杂技术的协同融合，我们实现了非凡的性能——针对广泛使用的deepfake检测器，100%的攻击成功率（ASR）。



## **31. DNNShield: Embedding Identifiers for Deep Neural Network Ownership Verification**

DNNShield：用于深度神经网络所有权验证的嵌入标识符 cs.CR

18 pages, 11 figures, 6 tables

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06581v1) [paper-pdf](http://arxiv.org/pdf/2403.06581v1)

**Authors**: Jasper Stang, Torsten Krauß, Alexandra Dmitrienko

**Abstract**: The surge in popularity of machine learning (ML) has driven significant investments in training Deep Neural Networks (DNNs). However, these models that require resource-intensive training are vulnerable to theft and unauthorized use. This paper addresses this challenge by introducing DNNShield, a novel approach for DNN protection that integrates seamlessly before training. DNNShield embeds unique identifiers within the model architecture using specialized protection layers. These layers enable secure training and deployment while offering high resilience against various attacks, including fine-tuning, pruning, and adaptive adversarial attacks. Notably, our approach achieves this security with minimal performance and computational overhead (less than 5\% runtime increase). We validate the effectiveness and efficiency of DNNShield through extensive evaluations across three datasets and four model architectures. This practical solution empowers developers to protect their DNNs and intellectual property rights.

摘要: 机器学习（ML）的普及推动了对训练深度神经网络（DNN）的大量投资。然而，这些需要资源密集型培训的模式很容易遭到盗窃和未经授权的使用。本文通过引入DNNShield来解决这一挑战，DNNShield是一种新的DNN保护方法，可在训练前无缝集成。DNNShield使用专门的保护层在模型架构中嵌入唯一标识符。这些层支持安全的培训和部署，同时提供针对各种攻击的高恢复能力，包括微调、修剪和自适应对抗攻击。值得注意的是，我们的方法以最小的性能和计算开销（运行时增加不到5%）实现了这种安全性。我们通过对三个数据集和四个模型架构的广泛评估来验证DNNShield的有效性和效率。这个实用的解决方案使开发人员能够保护他们的DNN和知识产权。



## **32. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

DistriBlock：通过利用输出分布的特征识别对抗性音频样本 cs.SD

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2305.17000v3) [paper-pdf](http://arxiv.org/pdf/2305.17000v3)

**Authors**: Matías Pizarro, Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.

摘要: 敌意攻击可以误导自动语音识别(ASR)系统预测任意目标文本，从而构成明显的安全威胁。为了防止此类攻击，我们提出了DistriBlock，这是一种适用于任何ASR系统的有效检测策略，它预测每个时间步输出令牌上的概率分布。我们测量了该分布的一组特征：输出概率的中位数、最大值和最小值，分布的熵，以及关于后续时间步分布的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性数据和恶意数据观察到的特征，我们应用二进制分类器，包括简单的基于阈值的分类、这种分类器的集成和神经网络。通过对不同的ASR系统和语言数据集的广泛分析，我们证明了该方法的最高性能，对于干净和有噪声的数据，接收器操作特征下的平均面积分别为99%和97%。为了评估我们方法的健壮性，我们证明了可以绕过DistriBlock的自适应攻击示例的噪声要大得多，这使得它们更容易通过过滤来检测，并为保持系统的健壮性创造了另一种途径。



## **33. Fooling Neural Networks for Motion Forecasting via Adversarial Attacks**

基于对抗攻击的欺骗神经网络运动预测 cs.CV

11 pages, 8 figures, VISSAP 2024

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.04954v2) [paper-pdf](http://arxiv.org/pdf/2403.04954v2)

**Authors**: Edgar Medina, Leyong Loh

**Abstract**: Human motion prediction is still an open problem, which is extremely important for autonomous driving and safety applications. Although there are great advances in this area, the widely studied topic of adversarial attacks has not been applied to multi-regression models such as GCNs and MLP-based architectures in human motion prediction. This work intends to reduce this gap using extensive quantitative and qualitative experiments in state-of-the-art architectures similar to the initial stages of adversarial attacks in image classification. The results suggest that models are susceptible to attacks even on low levels of perturbation. We also show experiments with 3D transformations that affect the model performance, in particular, we show that most models are sensitive to simple rotations and translations which do not alter joint distances. We conclude that similar to earlier CNN models, motion forecasting tasks are susceptible to small perturbations and simple 3D transformations.

摘要: 人体运动预测仍然是一个悬而未决的问题，对于自动驾驶和安全应用具有极其重要的意义。尽管这一领域已经取得了很大的进展，但被广泛研究的对抗性攻击主题还没有被应用到多元回归模型中，如GCNS和基于MLP的人体运动预测体系。这项工作旨在通过在最先进的体系结构中进行广泛的定量和定性实验来缩小这一差距，该体系结构类似于图像分类中对抗性攻击的初始阶段。结果表明，即使在低水平的扰动下，模型也容易受到攻击。我们还展示了影响模型性能的3D变换的实验，特别是，我们表明大多数模型对简单的旋转和平移都很敏感，这些旋转和平移不会改变关节距离。我们的结论是，与早期的CNN模型类似，运动预测任务容易受到小扰动和简单的3D变换的影响。



## **34. Intra-Section Code Cave Injection for Adversarial Evasion Attacks on Windows PE Malware File**

针对Windows PE恶意软件文件的对抗性规避攻击的段内代码Cave注入 cs.CR

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06428v1) [paper-pdf](http://arxiv.org/pdf/2403.06428v1)

**Authors**: Kshitiz Aryal, Maanak Gupta, Mahmoud Abdelsalam, Moustafa Saleh

**Abstract**: Windows malware is predominantly available in cyberspace and is a prime target for deliberate adversarial evasion attacks. Although researchers have investigated the adversarial malware attack problem, a multitude of important questions remain unanswered, including (a) Are the existing techniques to inject adversarial perturbations in Windows Portable Executable (PE) malware files effective enough for evasion purposes?; (b) Does the attack process preserve the original behavior of malware?; (c) Are there unexplored approaches/locations that can be used to carry out adversarial evasion attacks on Windows PE malware?; and (d) What are the optimal locations and sizes of adversarial perturbations required to evade an ML-based malware detector without significant structural change in the PE file? To answer some of these questions, this work proposes a novel approach that injects a code cave within the section (i.e., intra-section) of Windows PE malware files to make space for adversarial perturbations. In addition, a code loader is also injected inside the PE file, which reverts adversarial malware to its original form during the execution, preserving the malware's functionality and executability. To understand the effectiveness of our approach, we injected adversarial perturbations inside the .text, .data and .rdata sections, generated using the gradient descent and Fast Gradient Sign Method (FGSM), to target the two popular CNN-based malware detectors, MalConv and MalConv2. Our experiments yielded notable results, achieving a 92.31% evasion rate with gradient descent and 96.26% with FGSM against MalConv, compared to the 16.17% evasion rate for append attacks. Similarly, when targeting MalConv2, our approach achieved a remarkable maximum evasion rate of 97.93% with gradient descent and 94.34% with FGSM, significantly surpassing the 4.01% evasion rate observed with append attacks.

摘要: Windows恶意软件主要存在于网络空间，是故意对抗性规避攻击的主要目标。尽管研究人员已经研究了对抗性恶意软件攻击问题，但仍有许多重要问题没有得到解答，包括（a）在Windows可移植可执行（PE）恶意软件文件中注入对抗性干扰的现有技术是否足以有效规避？(b)攻击过程是否保留了恶意软件的原始行为？(c)是否存在未探索的方法/位置可用于对Windows PE恶意软件进行对抗性规避攻击？以及（d）在PE文件中不发生重大结构变化的情况下，规避基于ML的恶意软件检测器所需的对抗性扰动的最佳位置和大小是什么？为了回答其中的一些问题，这项工作提出了一种新的方法，在该节中注入一个代码洞穴（即，内部）Windows PE恶意软件文件，为对抗干扰腾出空间。此外，在PE文件中还注入了一个代码加载器，它在执行过程中将对抗性恶意软件恢复到其原始形式，从而保留恶意软件的功能和可执行性。为了了解我们方法的有效性，我们在使用梯度下降和快速梯度符号法（FGSM）生成的. text、. data和. rdata部分中注入了对抗性扰动，以针对两个流行的基于CNN的恶意软件检测器MalConv和MalConv2。我们的实验取得了显着的结果，实现了92.31%的逃避率梯度下降和96.26%的FGSM对MalConv，相比之下，附加攻击的逃避率为16.17%。同样，当针对MalConv 2时，我们的方法在梯度下降和FGSM的情况下实现了97.93%的最大规避率，显著超过了附加攻击时观察到的4.01%的规避率。



## **35. A Zero Trust Framework for Realization and Defense Against Generative AI Attacks in Power Grid**

电网中生成性人工智能攻击的零信任实现与防御框架 cs.CR

Accepted article by IEEE International Conference on Communications  (ICC 2024), Copyright 2024 IEEE

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06388v1) [paper-pdf](http://arxiv.org/pdf/2403.06388v1)

**Authors**: Md. Shirajum Munir, Sravanthi Proddatoori, Manjushree Muralidhara, Walid Saad, Zhu Han, Sachin Shetty

**Abstract**: Understanding the potential of generative AI (GenAI)-based attacks on the power grid is a fundamental challenge that must be addressed in order to protect the power grid by realizing and validating risk in new attack vectors. In this paper, a novel zero trust framework for a power grid supply chain (PGSC) is proposed. This framework facilitates early detection of potential GenAI-driven attack vectors (e.g., replay and protocol-type attacks), assessment of tail risk-based stability measures, and mitigation of such threats. First, a new zero trust system model of PGSC is designed and formulated as a zero-trust problem that seeks to guarantee for a stable PGSC by realizing and defending against GenAI-driven cyber attacks. Second, in which a domain-specific generative adversarial networks (GAN)-based attack generation mechanism is developed to create a new vulnerability cyberspace for further understanding that threat. Third, tail-based risk realization metrics are developed and implemented for quantifying the extreme risk of a potential attack while leveraging a trust measurement approach for continuous validation. Fourth, an ensemble learning-based bootstrap aggregation scheme is devised to detect the attacks that are generating synthetic identities with convincing user and distributed energy resources device profiles. Experimental results show the efficacy of the proposed zero trust framework that achieves an accuracy of 95.7% on attack vector generation, a risk measure of 9.61% for a 95% stable PGSC, and a 99% confidence in defense against GenAI-driven attack.

摘要: 了解基于生成人工智能（GenAI）的电网攻击的潜力是一项根本性挑战，必须解决这一挑战，以便通过认识和验证新攻击向量中的风险来保护电网。提出了一种新的电网供应链零信任框架。该框架有助于早期检测潜在的GenAI驱动的攻击向量（例如，重放和协议类型攻击）、基于尾部风险的稳定性措施的评估以及此类威胁的缓解。首先，设计了一个新的零信任系统模型，并将其表述为零信任问题，试图通过实现和防御GenAI驱动的网络攻击来保证一个稳定的PGSC。第二，开发了一种基于特定领域的生成对抗网络（GAN）的攻击生成机制，以创建一个新的网络漏洞空间，以进一步理解该威胁。第三，开发并实施基于尾部的风险实现指标，用于量化潜在攻击的极端风险，同时利用信任度量方法进行持续验证。第四，设计了一个基于集成学习的引导聚合方案，以检测生成具有令人信服的用户和分布式能源设备配置文件的合成身份的攻击。实验结果表明，所提出的零信任框架的有效性，达到了95.7%的攻击向量生成准确率，95%稳定的PGSC的风险度量为9.61%，和99%的置信度防御GenAI驱动的攻击。



## **36. Towards Scalable and Robust Model Versioning**

面向可扩展和健壮的模型版本控制 cs.LG

Published in IEEE SaTML 2024

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2401.09574v2) [paper-pdf](http://arxiv.org/pdf/2401.09574v2)

**Authors**: Wenxin Ding, Arjun Nitin Bhagoji, Ben Y. Zhao, Haitao Zheng

**Abstract**: As the deployment of deep learning models continues to expand across industries, the threat of malicious incursions aimed at gaining access to these deployed models is on the rise. Should an attacker gain access to a deployed model, whether through server breaches, insider attacks, or model inversion techniques, they can then construct white-box adversarial attacks to manipulate the model's classification outcomes, thereby posing significant risks to organizations that rely on these models for critical tasks. Model owners need mechanisms to protect themselves against such losses without the necessity of acquiring fresh training data - a process that typically demands substantial investments in time and capital.   In this paper, we explore the feasibility of generating multiple versions of a model that possess different attack properties, without acquiring new training data or changing model architecture. The model owner can deploy one version at a time and replace a leaked version immediately with a new version. The newly deployed model version can resist adversarial attacks generated leveraging white-box access to one or all previously leaked versions. We show theoretically that this can be accomplished by incorporating parameterized hidden distributions into the model training data, forcing the model to learn task-irrelevant features uniquely defined by the chosen data. Additionally, optimal choices of hidden distributions can produce a sequence of model versions capable of resisting compound transferability attacks over time. Leveraging our analytical insights, we design and implement a practical model versioning method for DNN classifiers, which leads to significant robustness improvements over existing methods. We believe our work presents a promising direction for safeguarding DNN services beyond their initial deployment.

摘要: 随着深度学习模型的部署在各个行业中不断扩展，旨在获取这些已部署模型的恶意入侵威胁正在上升。如果攻击者通过服务器入侵、内部攻击或模型反转技术获得对部署模型的访问权限，他们就可以构建白盒对抗攻击来操纵模型的分类结果，从而对依赖这些模型执行关键任务的组织构成重大风险。模型所有者需要一些机制来保护自己免受此类损失，而无需获取新的训练数据——这一过程通常需要大量的时间和资金投资。   在本文中，我们探讨了生成具有不同攻击特性的模型的多个版本的可行性，而无需获取新的训练数据或改变模型架构。模型所有者可以一次部署一个版本，并立即用新版本替换泄漏的版本。新部署的模型版本可以抵抗利用白盒访问一个或所有先前泄露的版本而产生的对抗攻击。我们从理论上证明，这可以通过将参数化的隐藏分布纳入模型训练数据来实现，迫使模型学习由所选数据唯一定义的任务无关的特征。此外，隐藏分布的最佳选择可以产生一系列模型版本，能够随着时间的推移抵抗复合传输性攻击。利用我们的分析见解，我们设计并实现了一个实用的DNN分类器模型版本控制方法，这导致了显着的鲁棒性改进，比现有的方法。我们相信，我们的工作为保护DNN服务的初始部署提供了一个有希望的方向。



## **37. Fake or Compromised? Making Sense of Malicious Clients in Federated Learning**

是假的还是妥协的？联合学习中对恶意客户端的理解 cs.LG

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2403.06319v1) [paper-pdf](http://arxiv.org/pdf/2403.06319v1)

**Authors**: Hamid Mozaffari, Sunav Choudhary, Amir Houmansadr

**Abstract**: Federated learning (FL) is a distributed machine learning paradigm that enables training models on decentralized data. The field of FL security against poisoning attacks is plagued with confusion due to the proliferation of research that makes different assumptions about the capabilities of adversaries and the adversary models they operate under. Our work aims to clarify this confusion by presenting a comprehensive analysis of the various poisoning attacks and defensive aggregation rules (AGRs) proposed in the literature, and connecting them under a common framework. To connect existing adversary models, we present a hybrid adversary model, which lies in the middle of the spectrum of adversaries, where the adversary compromises a few clients, trains a generative (e.g., DDPM) model with their compromised samples, and generates new synthetic data to solve an optimization for a stronger (e.g., cheaper, more practical) attack against different robust aggregation rules. By presenting the spectrum of FL adversaries, we aim to provide practitioners and researchers with a clear understanding of the different types of threats they need to consider when designing FL systems, and identify areas where further research is needed.

摘要: 联合学习(FL)是一种分布式机器学习范例，支持对分散数据的训练模型。由于越来越多的研究对对手的能力和他们所在的对手模型做出了不同的假设，因此针对中毒攻击的FL安全领域充满了困惑。我们的工作旨在通过对文献中提出的各种中毒攻击和防御聚集规则(AGR)进行全面分析，并在一个共同的框架下将它们联系起来，来澄清这种混淆。为了连接现有的敌手模型，我们提出了一种混合敌手模型，该模型位于敌手光谱的中间，其中敌手妥协一些客户端，用他们妥协的样本训练产生式(例如，DDPM)模型，并生成新的合成数据来解决针对不同健壮聚集规则的更强(例如，更便宜、更实用)攻击的优化。通过介绍外语对手的范围，我们的目的是让从业者和研究人员清楚地了解他们在设计外语系统时需要考虑的不同类型的威胁，并确定需要进一步研究的领域。



## **38. Improving behavior based authentication against adversarial attack using XAI**

利用XAI改进基于行为的认证对抗攻击 cs.CR

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2402.16430v2) [paper-pdf](http://arxiv.org/pdf/2402.16430v2)

**Authors**: Dong Qin, George Amariucai, Daji Qiao, Yong Guan

**Abstract**: In recent years, machine learning models, especially deep neural networks, have been widely used for classification tasks in the security domain. However, these models have been shown to be vulnerable to adversarial manipulation: small changes learned by an adversarial attack model, when applied to the input, can cause significant changes in the output. Most research on adversarial attacks and corresponding defense methods focuses only on scenarios where adversarial samples are directly generated by the attack model. In this study, we explore a more practical scenario in behavior-based authentication, where adversarial samples are collected from the attacker. The generated adversarial samples from the model are replicated by attackers with a certain level of discrepancy. We propose an eXplainable AI (XAI) based defense strategy against adversarial attacks in such scenarios. A feature selector, trained with our method, can be used as a filter in front of the original authenticator. It filters out features that are more vulnerable to adversarial attacks or irrelevant to authentication, while retaining features that are more robust. Through comprehensive experiments, we demonstrate that our XAI based defense strategy is effective against adversarial attacks and outperforms other defense strategies, such as adversarial training and defensive distillation.

摘要: 近年来，机器学习模型，特别是深度神经网络被广泛应用于安全领域的分类任务。然而，这些模型已被证明容易受到对抗性操纵：对抗性攻击模型学习到的微小变化，当应用于输入时，可能会导致输出的重大变化。关于对抗性攻击及其防御方法的研究大多集中在攻击模型直接生成对抗性样本的场景中。在这项研究中，我们探索了一种更实用的基于行为的身份验证场景，其中从攻击者那里收集了敌意样本。从该模型生成的对抗性样本被具有一定差异的攻击者复制。我们提出了一种基于可解释人工智能(XAI)的防御策略，以抵御此类场景中的对抗性攻击。用我们的方法训练的特征选择器可以用作原始认证器前面的过滤器。它过滤掉更容易受到对手攻击或与身份验证无关的功能，同时保留更健壮的功能。通过综合实验，我们证明了我们的基于XAI的防御策略对对手攻击是有效的，并且优于其他防御策略，如对抗性训练和防御蒸馏。



## **39. Learn from the Past: A Proxy Guided Adversarial Defense Framework with Self Distillation Regularization**

从过去学习：一个具有自蒸馏正则化的代理引导对抗防御框架 cs.LG

13 Pages

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2310.12713v2) [paper-pdf](http://arxiv.org/pdf/2310.12713v2)

**Authors**: Yaohua Liu, Jiaxin Gao, Xianghao Jiao, Zhu Liu, Xin Fan, Risheng Liu

**Abstract**: Adversarial Training (AT), pivotal in fortifying the robustness of deep learning models, is extensively adopted in practical applications. However, prevailing AT methods, relying on direct iterative updates for target model's defense, frequently encounter obstacles such as unstable training and catastrophic overfitting. In this context, our work illuminates the potential of leveraging the target model's historical states as a proxy to provide effective initialization and defense prior, which results in a general proxy guided defense framework, `LAST' ({\bf L}earn from the P{\bf ast}). Specifically, LAST derives response of the proxy model as dynamically learned fast weights, which continuously corrects the update direction of the target model. Besides, we introduce a self-distillation regularized defense objective, ingeniously designed to steer the proxy model's update trajectory without resorting to external teacher models, thereby ameliorating the impact of catastrophic overfitting on performance. Extensive experiments and ablation studies showcase the framework's efficacy in markedly improving model robustness (e.g., up to 9.2\% and 20.3\% enhancement in robust accuracy on CIFAR10 and CIFAR100 datasets, respectively) and training stability. These improvements are consistently observed across various model architectures, larger datasets, perturbation sizes, and attack modalities, affirming LAST's ability to consistently refine both single-step and multi-step AT strategies. The code will be available at~\url{https://github.com/callous-youth/LAST}.

摘要: 对抗训练（AT）是增强深度学习模型鲁棒性的关键，在实际应用中得到了广泛的应用。然而，传统的AT方法依赖于直接迭代更新的目标模型防御，经常遇到训练不稳定和灾难性过拟合等障碍。在这种情况下，我们的工作阐明了利用目标模型的历史状态作为代理提供有效的初始化和防御之前的潜力，这导致了一个通用的代理引导防御框架，'LAST'（{\BF L}从P {\BF ast}中赚取）。具体而言，LAST将代理模型的响应作为动态学习的快速权值，不断校正目标模型的更新方向。此外，我们引入了一个自蒸馏正则化的防御目标，巧妙地设计来引导代理模型的更新轨迹，而不诉诸外部教师模型，从而改善灾难性过拟合对性能的影响。大量的实验和消融研究显示了该框架在显著提高模型鲁棒性方面的功效（例如，CIFAR10和CIFAR100数据集的鲁棒准确性分别提高了9.2%和20.3%）和训练稳定性。这些改进在各种模型架构、更大的数据集、扰动大小和攻击模式中得到一致的观察，证实了LAST能够始终如一地改进单步和多步AT策略。代码将在～\url {https：//github.com/callous—youth/LAST}找到。



## **40. Deep Reinforcement Learning with Spiking Q-learning**

基于尖峰Q学习的深度强化学习 cs.NE

15 pages, 7 figures

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2201.09754v2) [paper-pdf](http://arxiv.org/pdf/2201.09754v2)

**Authors**: Ding Chen, Peixi Peng, Tiejun Huang, Yonghong Tian

**Abstract**: With the help of special neuromorphic hardware, spiking neural networks (SNNs) are expected to realize artificial intelligence (AI) with less energy consumption. It provides a promising energy-efficient way for realistic control tasks by combining SNNs with deep reinforcement learning (RL). There are only a few existing SNN-based RL methods at present. Most of them either lack generalization ability or employ Artificial Neural Networks (ANNs) to estimate value function in training. The former needs to tune numerous hyper-parameters for each scenario, and the latter limits the application of different types of RL algorithm and ignores the large energy consumption in training. To develop a robust spike-based RL method, we draw inspiration from non-spiking interneurons found in insects and propose the deep spiking Q-network (DSQN), using the membrane voltage of non-spiking neurons as the representation of Q-value, which can directly learn robust policies from high-dimensional sensory inputs using end-to-end RL. Experiments conducted on 17 Atari games demonstrate the DSQN is effective and even outperforms the ANN-based deep Q-network (DQN) in most games. Moreover, the experiments show superior learning stability and robustness to adversarial attacks of DSQN.

摘要: 在特殊的神经形态硬件的帮助下，尖峰神经网络有望以更低的能耗实现人工智能。通过将SNN与深度强化学习（RL）相结合，为实际控制任务提供了一种有前途的节能方法。现有的基于SNN的RL方法很少。大多数算法要么缺乏泛化能力，要么采用人工神经网络（ANN）来估计值函数。前者需要为每个场景调整大量的超参数，后者限制了不同类型RL算法的应用，忽略了训练中的大能耗。为了开发一种鲁棒的基于尖峰的强化学习方法，我们从昆虫中发现的非尖峰中间神经元的启发，提出了深度尖峰Q网络（DSQN），该网络利用非尖峰神经元的膜电压作为Q值的表示，通过端到端的强化学习方法直接从高维感觉输入中学习鲁棒策略。在17款Atari游戏上进行的实验表明，DSQN是有效的，甚至在大多数游戏中优于基于人工神经网络的深度Q网络（DQN）。实验结果表明，DSQN具有良好的学习稳定性和对抗攻击的鲁棒性。



## **41. Language-Driven Anchors for Zero-Shot Adversarial Robustness**

零镜头对抗鲁棒性的迭代驱动算法 cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2301.13096v3) [paper-pdf](http://arxiv.org/pdf/2301.13096v3)

**Authors**: Xiao Li, Wei Zhang, Yining Liu, Zhanhao Hu, Bo Zhang, Xiaolin Hu

**Abstract**: Deep Neural Networks (DNNs) are known to be susceptible to adversarial attacks. Previous researches mainly focus on improving adversarial robustness in the fully supervised setting, leaving the challenging domain of zero-shot adversarial robustness an open question. In this work, we investigate this domain by leveraging the recent advances in large vision-language models, such as CLIP, to introduce zero-shot adversarial robustness to DNNs. We propose LAAT, a Language-driven, Anchor-based Adversarial Training strategy. LAAT utilizes the features of a text encoder for each category as fixed anchors (normalized feature embeddings) for each category, which are then employed for adversarial training. By leveraging the semantic consistency of the text encoders, LAAT aims to enhance the adversarial robustness of the image model on novel categories. However, naively using text encoders leads to poor results. Through analysis, we identified the issue to be the high cosine similarity between text encoders. We then design an expansion algorithm and an alignment cross-entropy loss to alleviate the problem. Our experimental results demonstrated that LAAT significantly improves zero-shot adversarial robustness over state-of-the-art methods. LAAT has the potential to enhance adversarial robustness by large-scale multimodal models, especially when labeled data is unavailable during training.

摘要: 众所周知，深度神经网络（DNN）容易受到对抗攻击。以往的研究主要集中在提高完全监督环境下的对抗鲁棒性，而零镜头对抗鲁棒性的挑战性领域则是一个悬而未决的问题。在这项工作中，我们通过利用大型视觉语言模型（如CLIP）的最新进展来研究这一领域，将零镜头对抗鲁棒性引入DNN。我们提出了LAAT，一种基于锚点驱动的对抗训练策略。LAAT利用每个类别的文本编码器的特征作为每个类别的固定锚点（规范化特征嵌入），然后用于对抗训练。通过利用文本编码器的语义一致性，LAAT旨在增强图像模型对新颖类别的对抗鲁棒性。然而，天真地使用文本编码器会导致糟糕的结果。通过分析，我们发现问题是文本编码器之间的高余弦相似度。然后，我们设计了一个扩展算法和一个对齐交叉熵损失来缓解问题。我们的实验结果表明，LAAT显着提高了零镜头对抗鲁棒性的国家的最先进的方法。LAAT有潜力通过大规模多模态模型增强对抗性鲁棒性，特别是当训练期间标记数据不可用时。



## **42. Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization**

对抗性净化训练（AToP）：增强鲁棒性和泛化能力 cs.CV

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2401.16352v2) [paper-pdf](http://arxiv.org/pdf/2401.16352v2)

**Authors**: Guang Lin, Chao Li, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: The deep neural networks are known to be vulnerable to well-designed adversarial attacks. The most successful defense technique based on adversarial training (AT) can achieve optimal robustness against particular attacks but cannot generalize well to unseen attacks. Another effective defense technique based on adversarial purification (AP) can enhance generalization but cannot achieve optimal robustness. Meanwhile, both methods share one common limitation on the degraded standard accuracy. To mitigate these issues, we propose a novel pipeline called Adversarial Training on Purification (AToP), which comprises two components: perturbation destruction by random transforms (RT) and purifier model fine-tuned (FT) by adversarial loss. RT is essential to avoid overlearning to known attacks resulting in the robustness generalization to unseen attacks and FT is essential for the improvement of robustness. To evaluate our method in an efficient and scalable way, we conduct extensive experiments on CIFAR-10, CIFAR-100, and ImageNette to demonstrate that our method achieves state-of-the-art results and exhibits generalization ability against unseen attacks.

摘要: 众所周知，深度神经网络很容易受到精心设计的对抗性攻击。最成功的基于对抗性训练(AT)的防御技术可以达到对特定攻击的最佳健壮性，但不能很好地推广到看不见的攻击。另一种基于对抗性净化(AP)的有效防御技术可以增强泛化能力，但不能达到最优的健壮性。同时，这两种方法都有一个共同的缺陷，那就是标准精度下降。为了缓解这些问题，我们提出了一种新的流水线，称为对抗性净化训练(TOOP)，该流水线由两部分组成：通过随机变换的扰动破坏(RT)和通过对抗性损失微调(FT)的净化器模型。RT对于避免对已知攻击的过度学习导致对未知攻击的健壮性泛化至关重要，而FT对于提高健壮性是必不可少的。为了有效和可扩展地评估我们的方法，我们在CIFAR-10、CIFAR-100和ImageNette上进行了大量的实验，证明了我们的方法取得了最先进的结果，并表现出对不可见攻击的泛化能力。



## **43. From Chatbots to PhishBots? -- Preventing Phishing scams created using ChatGPT, Google Bard and Claude**

从聊天机器人到PhishBots？——防止使用ChatGPT、Google Bard和Claude创建的网络钓鱼诈骗 cs.CR

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2310.19181v2) [paper-pdf](http://arxiv.org/pdf/2310.19181v2)

**Authors**: Sayak Saha Roy, Poojitha Thota, Krishna Vamsi Naragam, Shirin Nilizadeh

**Abstract**: The advanced capabilities of Large Language Models (LLMs) have made them invaluable across various applications, from conversational agents and content creation to data analysis, research, and innovation. However, their effectiveness and accessibility also render them susceptible to abuse for generating malicious content, including phishing attacks. This study explores the potential of using four popular commercially available LLMs, i.e., ChatGPT (GPT 3.5 Turbo), GPT 4, Claude, and Bard, to generate functional phishing attacks using a series of malicious prompts. We discover that these LLMs can generate both phishing websites and emails that can convincingly imitate well-known brands and also deploy a range of evasive tactics that are used to elude detection mechanisms employed by anti-phishing systems. These attacks can be generated using unmodified or "vanilla" versions of these LLMs without requiring any prior adversarial exploits such as jailbreaking. We evaluate the performance of the LLMs towards generating these attacks and find that they can also be utilized to create malicious prompts that, in turn, can be fed back to the model to generate phishing scams - thus massively reducing the prompt-engineering effort required by attackers to scale these threats. As a countermeasure, we build a BERT-based automated detection tool that can be used for the early detection of malicious prompts to prevent LLMs from generating phishing content. Our model is transferable across all four commercial LLMs, attaining an average accuracy of 96% for phishing website prompts and 94% for phishing email prompts. We also disclose the vulnerabilities to the concerned LLMs, with Google acknowledging it as a severe issue. Our detection model is available for use at Hugging Face, as well as a ChatGPT Actions plugin.

摘要: 大型语言模型（LLM）的先进功能使它们在各种应用程序中发挥了非常重要的作用，从会话代理和内容创建到数据分析、研究和创新。然而，它们的有效性和可访问性也使它们容易被滥用以生成恶意内容，包括网络钓鱼攻击。本研究探讨了使用四种流行的商业化LLM的潜力，即，ChatGPT（GPT 3.5 Turbo）、GPT 4、Claude和Bard，使用一系列恶意提示生成功能性网络钓鱼攻击。我们发现，这些LLM可以生成钓鱼网站和电子邮件，可以令人信服地模仿知名品牌，还部署了一系列逃避策略，用于逃避反钓鱼系统采用的检测机制。这些攻击可以使用这些LLM的未修改或“vanilla”版本生成，而不需要任何先前的对抗性攻击，如越狱。我们评估了LLM在生成这些攻击方面的性能，发现它们还可以用来创建恶意提示，反过来，这些提示可以反馈到模型中生成网络钓鱼诈骗，从而大大减少了攻击者扩展这些威胁所需的网络设计工作。作为一种对策，我们构建了一个基于BERT的自动检测工具，用于早期检测恶意提示，以防止LLM生成钓鱼内容。我们的模型可在所有四个商业LLM中移植，钓鱼网站提示的平均准确率为96%，钓鱼电子邮件提示的平均准确率为94%。我们还向相关LLM披露了这些漏洞，谷歌承认这是一个严重的问题。我们的检测模型可用于Hugging Face，以及ChatGPT Action插件。



## **44. Attacking Transformers with Feature Diversity Adversarial Perturbation**

利用特征多样性对抗扰动攻击变压器 cs.CR

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2403.07942v1) [paper-pdf](http://arxiv.org/pdf/2403.07942v1)

**Authors**: Chenxing Gao, Hang Zhou, Junqing Yu, YuTeng Ye, Jiale Cai, Junle Wang, Wei Yang

**Abstract**: Understanding the mechanisms behind Vision Transformer (ViT), particularly its vulnerability to adversarial perturba tions, is crucial for addressing challenges in its real-world applications. Existing ViT adversarial attackers rely on la bels to calculate the gradient for perturbation, and exhibit low transferability to other structures and tasks. In this paper, we present a label-free white-box attack approach for ViT-based models that exhibits strong transferability to various black box models, including most ViT variants, CNNs, and MLPs, even for models developed for other modalities. Our inspira tion comes from the feature collapse phenomenon in ViTs, where the critical attention mechanism overly depends on the low-frequency component of features, causing the features in middle-to-end layers to become increasingly similar and eventually collapse. We propose the feature diversity attacker to naturally accelerate this process and achieve remarkable performance and transferability.

摘要: 了解视觉变形器(VIT)背后的机制，特别是它对对抗性扰动的脆弱性，对于解决其现实应用中的挑战至关重要。现有的VIT对手攻击者依靠LABELS来计算扰动的梯度，并且表现出对其他结构和任务的低可转移性。在本文中，我们提出了一种基于VIT的模型的无标签白盒攻击方法，该方法对各种黑盒模型表现出很强的可移植性，包括大多数VIT变体、CNN和MLP，甚至对于为其他通道开发的模型也是如此。我们的灵感来自于VITS中的特征崩溃现象，即关键注意机制过度依赖于特征的低频分量，导致中端层的特征变得越来越相似，最终崩溃。我们提出了特征分集攻击者，自然地加速了这一过程，并获得了显著的性能和可移植性。



## **45. Hard-label based Small Query Black-box Adversarial Attack**

基于硬标签的小查询黑盒对抗攻击 cs.LG

11 pages, 3 figures

**SubmitDate**: 2024-03-09    [abs](http://arxiv.org/abs/2403.06014v1) [paper-pdf](http://arxiv.org/pdf/2403.06014v1)

**Authors**: Jeonghwan Park, Paul Miller, Niall McLaughlin

**Abstract**: We consider the hard label based black box adversarial attack setting which solely observes predicted classes from the target model. Most of the attack methods in this setting suffer from impractical number of queries required to achieve a successful attack. One approach to tackle this drawback is utilising the adversarial transferability between white box surrogate models and black box target model. However, the majority of the methods adopting this approach are soft label based to take the full advantage of zeroth order optimisation. Unlike mainstream methods, we propose a new practical setting of hard label based attack with an optimisation process guided by a pretrained surrogate model. Experiments show the proposed method significantly improves the query efficiency of the hard label based black-box attack across various target model architectures. We find the proposed method achieves approximately 5 times higher attack success rate compared to the benchmarks, especially at the small query budgets as 100 and 250.

摘要: 我们考虑基于硬标签的黑盒对抗攻击设置，它只观察目标模型中的预测类。在此设置中，大多数攻击方法都存在成功攻击所需的查询数量不切实际的问题。解决这一缺点的一种方法是利用白盒代理模型和黑盒目标模型之间的对抗转移性。然而，采用这种方法的大多数方法都是基于软标签的，以充分利用零阶优化的优势。与主流的方法不同，我们提出了一种新的基于硬标签的攻击的实际设置与优化过程的预训练代理模型指导。实验表明，该方法显著提高了基于硬标签黑盒攻击的跨目标模型结构的查询效率。我们发现，该方法实现了约5倍的攻击成功率相比基准，特别是在小查询预算为100和250。



## **46. IOI: Invisible One-Iteration Adversarial Attack on No-Reference Image- and Video-Quality Metrics**

IOI：对无参考图像和视频质量搜索器的隐形单迭代对抗攻击 eess.IV

**SubmitDate**: 2024-03-09    [abs](http://arxiv.org/abs/2403.05955v1) [paper-pdf](http://arxiv.org/pdf/2403.05955v1)

**Authors**: Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: No-reference image- and video-quality metrics are widely used in video processing benchmarks. The robustness of learning-based metrics under video attacks has not been widely studied. In addition to having success, attacks that can be employed in video processing benchmarks must be fast and imperceptible. This paper introduces an Invisible One-Iteration (IOI) adversarial attack on no reference image and video quality metrics. We compared our method alongside eight prior approaches using image and video datasets via objective and subjective tests. Our method exhibited superior visual quality across various attacked metric architectures while maintaining comparable attack success and speed. We made the code available on GitHub.

摘要: 无参考图像和视频质量度量被广泛用于视频处理基准。基于学习的度量在视频攻击下的鲁棒性尚未得到广泛的研究。除了成功之外，可以用于视频处理基准测试的攻击必须是快速和不可察觉的。本文介绍了一种针对无参考图像和视频质量度量的隐形单迭代（IOI）对抗攻击。我们通过客观和主观测试，将我们的方法与使用图像和视频数据集的八种先前方法进行了比较。我们的方法在各种受攻击的度量架构中表现出卓越的视觉质量，同时保持相当的攻击成功率和速度。我们在GitHub上提供了代码。



## **47. SoK: Secure Human-centered Wireless Sensing**

SoK：安全的以人为中心的无线传感 cs.CR

**SubmitDate**: 2024-03-09    [abs](http://arxiv.org/abs/2211.12087v2) [paper-pdf](http://arxiv.org/pdf/2211.12087v2)

**Authors**: Wei Sun, Tingjun Chen, Neil Gong

**Abstract**: Human-centered wireless sensing (HCWS) aims to understand the fine-grained environment and activities of a human using the diverse wireless signals around him/her. While the sensed information about a human can be used for many good purposes such as enhancing life quality, an adversary can also abuse it to steal private information about the human (e.g., location and person's identity). However, the literature lacks a systematic understanding of the privacy vulnerabilities of wireless sensing and the defenses against them, resulting in the privacy-compromising HCWS design.   In this work, we aim to bridge this gap to achieve the vision of secure human-centered wireless sensing. First, we propose a signal processing pipeline to identify private information leakage and further understand the benefits and tradeoffs of wireless sensing-based inference attacks and defenses. Based on this framework, we present the taxonomy of existing inference attacks and defenses. As a result, we can identify the open challenges and gaps in achieving privacy-preserving human-centered wireless sensing in the era of machine learning and further propose directions for future research in this field.

摘要: 以人为中心的无线传感（HCWS）旨在利用人类周围的各种无线信号来了解人类的精细环境和活动。虽然感知到的关于人类的信息可以用于许多良好的目的，如提高生活质量，但攻击者也可以滥用它来窃取关于人类的私人信息（例如，地点和人的身份）。然而，文献缺乏对无线传感的隐私漏洞和防御机制的系统理解，导致了隐私妥协的HCWS设计。   在这项工作中，我们的目标是弥合这一差距，以实现安全的以人为中心的无线传感愿景。首先，我们提出了一个信号处理管道来识别私人信息泄漏，并进一步了解基于无线传感的推理攻击和防御的好处和权衡。在此框架的基础上，我们提出了现有推理攻击和防御的分类。因此，我们可以确定在机器学习时代实现隐私保护以人为中心的无线传感的开放挑战和差距，并进一步提出该领域未来研究的方向。



## **48. Prepared for the Worst: A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm**

为最坏情况做好准备：一种基于学习的对抗性攻击用于ICP算法的弹性分析 cs.RO

8 pages (7 content, 1 reference). 5 figures, submitted to the IEEE  Robotics and Automation Letters (RA-L)

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05666v1) [paper-pdf](http://arxiv.org/pdf/2403.05666v1)

**Authors**: Ziyu Zhang, Johann Laconte, Daniil Lisus, Timothy D. Barfoot

**Abstract**: This paper presents a novel method to assess the resilience of the Iterative Closest Point (ICP) algorithm via deep-learning-based attacks on lidar point clouds. For safety-critical applications such as autonomous navigation, ensuring the resilience of algorithms prior to deployments is of utmost importance. The ICP algorithm has become the standard for lidar-based localization. However, the pose estimate it produces can be greatly affected by corruption in the measurements. Corruption can arise from a variety of scenarios such as occlusions, adverse weather, or mechanical issues in the sensor. Unfortunately, the complex and iterative nature of ICP makes assessing its resilience to corruption challenging. While there have been efforts to create challenging datasets and develop simulations to evaluate the resilience of ICP empirically, our method focuses on finding the maximum possible ICP pose error using perturbation-based adversarial attacks. The proposed attack induces significant pose errors on ICP and outperforms baselines more than 88% of the time across a wide range of scenarios. As an example application, we demonstrate that our attack can be used to identify areas on a map where ICP is particularly vulnerable to corruption in the measurements.

摘要: 提出了一种基于深度学习的激光雷达点云攻击评估迭代最近点算法抗攻击能力的新方法。对于自主导航等安全关键型应用，在部署之前确保算法的弹性是至关重要的。该算法已成为激光雷达定位的标准算法。然而，它产生的姿势估计可能会受到测量中的干扰的很大影响。损坏可能由多种情况引起，例如堵塞、恶劣天气或传感器中的机械问题。不幸的是，比较方案的复杂性和迭代性使评估其对腐败的复原力具有挑战性。虽然已经有人努力创建具有挑战性的数据集和开发仿真来经验地评估ICP的弹性，但我们的方法专注于使用基于扰动的对抗性攻击来寻找最大可能的ICP姿态误差。所提出的攻击在ICP上引起显著的姿势误差，并且在广泛的场景中超过基线的时间超过88%。作为一个示例应用程序，我们演示了我们的攻击可以用来识别地图上的那些区域，在测量中，ICP特别容易受到腐败的影响。



## **49. Invariant Aggregator for Defending against Federated Backdoor Attacks**

用于防御联合后门攻击的不变聚合器 cs.LG

AISTATS 2024 camera-ready

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2210.01834v4) [paper-pdf](http://arxiv.org/pdf/2210.01834v4)

**Authors**: Xiaoyang Wang, Dimitrios Dimitriadis, Sanmi Koyejo, Shruti Tople

**Abstract**: Federated learning enables training high-utility models across several clients without directly sharing their private data. As a downside, the federated setting makes the model vulnerable to various adversarial attacks in the presence of malicious clients. Despite the theoretical and empirical success in defending against attacks that aim to degrade models' utility, defense against backdoor attacks that increase model accuracy on backdoor samples exclusively without hurting the utility on other samples remains challenging. To this end, we first analyze the failure modes of existing defenses over a flat loss landscape, which is common for well-designed neural networks such as Resnet (He et al., 2015) but is often overlooked by previous works. Then, we propose an invariant aggregator that redirects the aggregated update to invariant directions that are generally useful via selectively masking out the update elements that favor few and possibly malicious clients. Theoretical results suggest that our approach provably mitigates backdoor attacks and remains effective over flat loss landscapes. Empirical results on three datasets with different modalities and varying numbers of clients further demonstrate that our approach mitigates a broad class of backdoor attacks with a negligible cost on the model utility.

摘要: 联合学习允许在多个客户之间培训高实用模型，而无需直接共享他们的私人数据。缺点是，联合设置使模型在存在恶意客户端的情况下容易受到各种敌意攻击。尽管在防御旨在降低模型效用的攻击方面取得了理论和经验上的成功，但针对后门攻击的防御仍然具有挑战性，这种攻击只能提高后门样本的模型精度，而不会损害其他样本的效用。为此，我们首先分析了平坦损失情况下现有防御的故障模式，这在设计良好的神经网络如RESNET(他等人，2015)中很常见，但经常被以前的工作忽视。然后，我们提出了一个不变聚集器，它通过有选择地屏蔽有利于少数甚至可能是恶意客户端的更新元素，将聚合的更新重定向到通常有用的不变方向。理论结果表明，我们的方法可以有效地减少后门攻击，并且在平价损失场景下仍然有效。在三个具有不同模式和不同客户端数量的数据集上的实验结果进一步表明，我们的方法以可以忽略不计的模型效用代价缓解了广泛类别的后门攻击。



## **50. Can LLMs Follow Simple Rules?**

LLM可以遵循简单的规则吗？ cs.AI

Project website: https://eecs.berkeley.edu/~normanmu/llm_rules;  revised content

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2311.04235v3) [paper-pdf](http://arxiv.org/pdf/2311.04235v3)

**Authors**: Norman Mu, Sarah Chen, Zifan Wang, Sizhe Chen, David Karamardian, Lulwa Aljeraisy, Basel Alomair, Dan Hendrycks, David Wagner

**Abstract**: As Large Language Models (LLMs) are deployed with increasing real-world responsibilities, it is important to be able to specify and constrain the behavior of these systems in a reliable manner. Model developers may wish to set explicit rules for the model, such as "do not generate abusive content", but these may be circumvented by jailbreaking techniques. Existing evaluations of adversarial attacks and defenses on LLMs generally require either expensive manual review or unreliable heuristic checks. To address this issue, we propose Rule-following Language Evaluation Scenarios (RuLES), a programmatic framework for measuring rule-following ability in LLMs. RuLES consists of 14 simple text scenarios in which the model is instructed to obey various rules while interacting with the user. Each scenario has a programmatic evaluation function to determine whether the model has broken any rules in a conversation. Our evaluations of proprietary and open models show that almost all current models struggle to follow scenario rules, even on straightforward test cases. We also demonstrate that simple optimization attacks suffice to significantly increase failure rates on test cases. We conclude by exploring two potential avenues for improvement: test-time steering and supervised fine-tuning.

摘要: 随着大型语言模型（LLM）的部署与日益增加的现实世界责任，重要的是能够以可靠的方式指定和约束这些系统的行为。模型开发者可能希望为模型设置明确的规则，例如“不要生成滥用内容”，但这些规则可能会被越狱技术绕过。现有的对LLM的对抗性攻击和防御的评估通常需要昂贵的手动审查或不可靠的启发式检查。为了解决这个问题，我们提出了规则遵循语言评估方案（RuLES），一个用于测量LLM规则遵循能力的程序框架。RuLES由14个简单的文本场景组成，在这些场景中，模型被指示在与用户交互时遵守各种规则。每个场景都有一个程序化的评估功能，以确定模型是否违反了会话中的任何规则。我们对私有模型和开放模型的评估表明，几乎所有当前模型都难以遵循场景规则，即使是在简单的测试用例上。我们还证明，简单的优化攻击足以显着提高测试用例的失败率。最后，我们探索了两个潜在的改进途径：测试时间控制和监督微调。



