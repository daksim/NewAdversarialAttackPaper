# Latest Adversarial Attack Papers
**update at 2024-12-24 10:06:31**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. ErasableMask: A Robust and Erasable Privacy Protection Scheme against Black-box Face Recognition Models**

ErasableMass：针对黑匣子人脸识别模型的稳健且可擦除的隐私保护方案 cs.CV

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.17038v1) [paper-pdf](http://arxiv.org/pdf/2412.17038v1)

**Authors**: Sipeng Shen, Yunming Zhang, Dengpan Ye, Xiuwen Shi, Long Tang, Haoran Duan, Ziyi Liu

**Abstract**: While face recognition (FR) models have brought remarkable convenience in face verification and identification, they also pose substantial privacy risks to the public. Existing facial privacy protection schemes usually adopt adversarial examples to disrupt face verification of FR models. However, these schemes often suffer from weak transferability against black-box FR models and permanently damage the identifiable information that cannot fulfill the requirements of authorized operations such as forensics and authentication. To address these limitations, we propose ErasableMask, a robust and erasable privacy protection scheme against black-box FR models. Specifically, via rethinking the inherent relationship between surrogate FR models, ErasableMask introduces a novel meta-auxiliary attack, which boosts black-box transferability by learning more general features in a stable and balancing optimization strategy. It also offers a perturbation erasion mechanism that supports the erasion of semantic perturbations in protected face without degrading image quality. To further improve performance, ErasableMask employs a curriculum learning strategy to mitigate optimization conflicts between adversarial attack and perturbation erasion. Extensive experiments on the CelebA-HQ and FFHQ datasets demonstrate that ErasableMask achieves the state-of-the-art performance in transferability, achieving over 72% confidence on average in commercial FR systems. Moreover, ErasableMask also exhibits outstanding perturbation erasion performance, achieving over 90% erasion success rate.

摘要: 虽然人脸识别(FR)模型在人脸验证和识别方面带来了显著的便利，但它们也给公众带来了巨大的隐私风险。现有的人脸隐私保护方案通常采用对抗性的例子来干扰FR模型的人脸验证。然而，这些方案往往对黑盒FR模型的可转移性较弱，并且永久性地破坏了不能满足取证和认证等授权操作要求的可识别信息。为了解决这些局限性，我们提出了一种针对黑盒FR模型的健壮且可擦除的隐私保护方案--可擦除掩码。具体地说，通过重新考虑代理FR模型之间的内在联系，ErasableMASK引入了一种新的元辅助攻击，该攻击通过学习稳定平衡的优化策略中的更多一般特征来提高黑盒的可转移性。它还提供了一种扰动消除机制，支持在不降低图像质量的情况下消除受保护人脸的语义扰动。为了进一步提高性能，ErasableMASK采用了课程学习策略来缓解对抗性攻击和扰动擦除之间的优化冲突。在CelebA-HQ和FFHQ数据集上的广泛实验表明，可擦除掩码在可转移性方面达到了最先进的性能，在商业FR系统中平均达到72%以上的置信度。此外，可擦除掩模还表现出出色的扰动擦除性能，擦除成功率达到90%以上。



## **2. Robustness of Large Language Models Against Adversarial Attacks**

大型语言模型对抗对抗攻击的鲁棒性 cs.CL

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.17011v1) [paper-pdf](http://arxiv.org/pdf/2412.17011v1)

**Authors**: Yiyi Tao, Yixian Shen, Hang Zhang, Yanxin Shen, Lun Wang, Chuanqi Shi, Shaoshuai Du

**Abstract**: The increasing deployment of Large Language Models (LLMs) in various applications necessitates a rigorous evaluation of their robustness against adversarial attacks. In this paper, we present a comprehensive study on the robustness of GPT LLM family. We employ two distinct evaluation methods to assess their resilience. The first method introduce character-level text attack in input prompts, testing the models on three sentiment classification datasets: StanfordNLP/IMDB, Yelp Reviews, and SST-2. The second method involves using jailbreak prompts to challenge the safety mechanisms of the LLMs. Our experiments reveal significant variations in the robustness of these models, demonstrating their varying degrees of vulnerability to both character-level and semantic-level adversarial attacks. These findings underscore the necessity for improved adversarial training and enhanced safety mechanisms to bolster the robustness of LLMs.

摘要: 大型语言模型（LLM）在各种应用程序中的部署越来越多，需要严格评估其对抗性攻击的稳健性。本文对GPT LLM家族的稳健性进行了全面的研究。我们采用两种不同的评估方法来评估其弹性。第一种方法在输入提示中引入字符级文本攻击，在三个情感分类数据集上测试模型：StanfordNLP/IMDB、Yelp Reviews和CST-2。第二种方法涉及使用越狱提示来挑战LLM的安全机制。我们的实验揭示了这些模型的稳健性存在显着差异，证明了它们对字符级和语义级对抗攻击的脆弱性程度不同。这些发现强调了改进对抗培训和增强安全机制以增强LLM稳健性的必要性。



## **3. Breaking Barriers in Physical-World Adversarial Examples: Improving Robustness and Transferability via Robust Feature**

打破物理世界对抗示例中的障碍：通过稳健特征提高稳健性和可移植性 cs.CV

Accepted by AAAI2025

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.16958v1) [paper-pdf](http://arxiv.org/pdf/2412.16958v1)

**Authors**: Yichen Wang, Yuxuan Chou, Ziqi Zhou, Hangtao Zhang, Wei Wan, Shengshan Hu, Minghui Li

**Abstract**: As deep neural networks (DNNs) are widely applied in the physical world, many researches are focusing on physical-world adversarial examples (PAEs), which introduce perturbations to inputs and cause the model's incorrect outputs. However, existing PAEs face two challenges: unsatisfactory attack performance (i.e., poor transferability and insufficient robustness to environment conditions), and difficulty in balancing attack effectiveness with stealthiness, where better attack effectiveness often makes PAEs more perceptible.   In this paper, we explore a novel perturbation-based method to overcome the challenges. For the first challenge, we introduce a strategy Deceptive RF injection based on robust features (RFs) that are predictive, robust to perturbations, and consistent across different models. Specifically, it improves the transferability and robustness of PAEs by covering RFs of other classes onto the predictive features in clean images. For the second challenge, we introduce another strategy Adversarial Semantic Pattern Minimization, which removes most perturbations and retains only essential adversarial patterns in AEsBased on the two strategies, we design our method Robust Feature Coverage Attack (RFCoA), comprising Robust Feature Disentanglement and Adversarial Feature Fusion. In the first stage, we extract target class RFs in feature space. In the second stage, we use attention-based feature fusion to overlay these RFs onto predictive features of clean images and remove unnecessary perturbations. Experiments show our method's superior transferability, robustness, and stealthiness compared to existing state-of-the-art methods. Additionally, our method's effectiveness can extend to Large Vision-Language Models (LVLMs), indicating its potential applicability to more complex tasks.

摘要: 随着深度神经网络(DNN)在物理世界中的广泛应用，许多研究都集中在物理世界中的对抗性例子(PAE)上，这些例子会对输入产生扰动，导致模型输出不正确。然而，现有的PAE面临着两个挑战：攻击性能不令人满意(即可转移性差，对环境条件的健壮性不够)，以及难以平衡攻击有效性和隐蔽性，更好的攻击效率往往使PAE更容易被感知。在本文中，我们探索了一种新的基于扰动的方法来克服这些挑战。对于第一个挑战，我们引入了一种基于稳健特征(RF)的欺骗性射频注入策略，这些特征具有预测性、对扰动具有鲁棒性，并且在不同的模型中保持一致。具体地说，它通过将其他类的RF覆盖到干净图像中的预测特征来提高PAE的可转移性和稳健性。对于第二个挑战，我们引入了另一种对抗性语义模式最小化策略，该策略去除了大部分扰动，只保留了AEss中的基本对抗性模式。在这两种策略的基础上，我们设计了一种鲁棒特征覆盖攻击(RFCoA)方法，包括健壮特征解缠和对抗性特征融合。在第一阶段，我们在特征空间中提取目标类RFS。在第二阶段，我们使用基于注意力的特征融合将这些RF叠加到干净图像的预测特征上，并去除不必要的扰动。实验表明，与现有最先进的方法相比，我们的方法具有更好的可转移性、健壮性和隐蔽性。此外，我们的方法的有效性可以扩展到大型视觉语言模型(LVLM)，这表明它对更复杂的任务具有潜在的适用性。



## **4. NumbOD: A Spatial-Frequency Fusion Attack Against Object Detectors**

NumbOD：针对目标检测器的空频融合攻击 cs.CV

Accepted by AAAI 2025

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.16955v1) [paper-pdf](http://arxiv.org/pdf/2412.16955v1)

**Authors**: Ziqi Zhou, Bowen Li, Yufei Song, Zhifei Yu, Shengshan Hu, Wei Wan, Leo Yu Zhang, Dezhong Yao, Hai Jin

**Abstract**: With the advancement of deep learning, object detectors (ODs) with various architectures have achieved significant success in complex scenarios like autonomous driving. Previous adversarial attacks against ODs have been focused on designing customized attacks targeting their specific structures (e.g., NMS and RPN), yielding some results but simultaneously constraining their scalability. Moreover, most efforts against ODs stem from image-level attacks originally designed for classification tasks, resulting in redundant computations and disturbances in object-irrelevant areas (e.g., background). Consequently, how to design a model-agnostic efficient attack to comprehensively evaluate the vulnerabilities of ODs remains challenging and unresolved. In this paper, we propose NumbOD, a brand-new spatial-frequency fusion attack against various ODs, aimed at disrupting object detection within images. We directly leverage the features output by the OD without relying on its internal structures to craft adversarial examples. Specifically, we first design a dual-track attack target selection strategy to select high-quality bounding boxes from OD outputs for targeting. Subsequently, we employ directional perturbations to shift and compress predicted boxes and change classification results to deceive ODs. Additionally, we focus on manipulating the high-frequency components of images to confuse ODs' attention on critical objects, thereby enhancing the attack efficiency. Our extensive experiments on nine ODs and two datasets show that NumbOD achieves powerful attack performance and high stealthiness.

摘要: 随着深度学习的发展，各种结构的对象检测器在自动驾驶等复杂场景中取得了巨大的成功。以往针对OD的对抗性攻击都集中在针对其特定结构(如NMS和RPN)设计定制攻击，取得了一些效果，但同时也限制了其可扩展性。此外，大多数针对OD的努力源于最初为分类任务设计的图像级攻击，导致与对象无关的区域(例如背景)的冗余计算和干扰。因此，如何设计一种模型不可知的高效攻击来全面评估入侵检测的脆弱性仍然是一个挑战和悬而未决的问题。在本文中，我们提出了一种全新的针对各种OD的空频融合攻击，旨在扰乱图像中的目标检测。我们直接利用OD输出的功能，而不依赖其内部结构来创建对抗性的例子。具体地说，我们首先设计了一种双轨攻击目标选择策略，从OD输出中选择高质量的边界框进行目标定位。随后，我们使用方向扰动来移动和压缩预测框，并改变分类结果来欺骗OD。此外，我们还利用图像的高频成分来混淆OD对关键目标的注意力，从而提高了攻击效率。我们在9个OD和2个数据集上的大量实验表明，NumbOD具有强大的攻击性能和高隐蔽性。



## **5. Preventing Non-intrusive Load Monitoring Privacy Invasion: A Precise Adversarial Attack Scheme for Networked Smart Meters**

防止非侵入性负载监控隐私入侵：针对网络智能电表的精确对抗攻击方案 cs.CR

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.16893v1) [paper-pdf](http://arxiv.org/pdf/2412.16893v1)

**Authors**: Jialing He, Jiacheng Wang, Ning Wang, Shangwei Guo, Liehuang Zhu, Dusit Niyato, Tao Xiang

**Abstract**: Smart grid, through networked smart meters employing the non-intrusive load monitoring (NILM) technique, can considerably discern the usage patterns of residential appliances. However, this technique also incurs privacy leakage. To address this issue, we propose an innovative scheme based on adversarial attack in this paper. The scheme effectively prevents NILM models from violating appliance-level privacy, while also ensuring accurate billing calculation for users. To achieve this objective, we overcome two primary challenges. First, as NILM models fall under the category of time-series regression models, direct application of traditional adversarial attacks designed for classification tasks is not feasible. To tackle this issue, we formulate a novel adversarial attack problem tailored specifically for NILM and providing a theoretical foundation for utilizing the Jacobian of the NILM model to generate imperceptible perturbations. Leveraging the Jacobian, our scheme can produce perturbations, which effectively misleads the signal prediction of NILM models to safeguard users' appliance-level privacy. The second challenge pertains to fundamental utility requirements, where existing adversarial attack schemes struggle to achieve accurate billing calculation for users. To handle this problem, we introduce an additional constraint, mandating that the sum of added perturbations within a billing period must be precisely zero. Experimental validation on real-world power datasets REDD and UK-DALE demonstrates the efficacy of our proposed solutions, which can significantly amplify the discrepancy between the output of the targeted NILM model and the actual power signal of appliances, and enable accurate billing at the same time. Additionally, our solutions exhibit transferability, making the generated perturbation signal from one target model applicable to other diverse NILM models.

摘要: 智能电网通过采用非侵入式负荷监测(NILM)技术的联网智能电表，可以相当程度地识别家用电器的使用模式。然而，这种技术也会导致隐私泄露。针对这一问题，本文提出了一种基于对抗性攻击的创新方案。该方案有效地防止了NILM模型侵犯设备级隐私，同时还确保了用户准确的计费计算。为了实现这一目标，我们克服了两个主要挑战。首先，由于NILM模型属于时间序列回归模型的范畴，直接应用传统的针对分类任务的对抗性攻击是不可行的。为了解决这一问题，我们提出了一种新的针对NILM的对抗性攻击问题，为利用NILM模型的雅可比产生不可察觉的扰动提供了理论基础。利用雅可比矩阵，我们的方案可以产生扰动，从而有效地误导NILM模型的信号预测，以保护用户的家用电器级别的隐私。第二个挑战与基本的效用要求有关，现有的对抗性攻击方案难以为用户实现准确的计费计算。为了处理这个问题，我们引入了一个额外的约束，要求在一个计费周期内添加的扰动之和必须正好为零。在真实电力数据集REDD和UK-Dale上的实验验证表明，我们提出的解决方案是有效的，可以显著放大目标NILM模型的输出与家电实际电力信号之间的差异，同时实现准确的计费。此外，我们的解表现出可移植性，使得从一个目标模型产生的微扰信号适用于其他不同的NILM模型。



## **6. Towards More Robust Retrieval-Augmented Generation: Evaluating RAG Under Adversarial Poisoning Attacks**

迈向更稳健的检索增强生成：在对抗性中毒攻击下评估RAG cs.IR

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16708v1) [paper-pdf](http://arxiv.org/pdf/2412.16708v1)

**Authors**: Jinyan Su, Jin Peng Zhou, Zhengxin Zhang, Preslav Nakov, Claire Cardie

**Abstract**: Retrieval-Augmented Generation (RAG) systems have emerged as a promising solution to mitigate LLM hallucinations and enhance their performance in knowledge-intensive domains. However, these systems are vulnerable to adversarial poisoning attacks, where malicious passages injected into retrieval databases can mislead the model into generating factually incorrect outputs. In this paper, we investigate both the retrieval and the generation components of RAG systems to understand how to enhance their robustness against such attacks. From the retrieval perspective, we analyze why and how the adversarial contexts are retrieved and assess how the quality of the retrieved passages impacts downstream generation. From a generation perspective, we evaluate whether LLMs' advanced critical thinking and internal knowledge capabilities can be leveraged to mitigate the impact of adversarial contexts, i.e., using skeptical prompting as a self-defense mechanism. Our experiments and findings provide actionable insights into designing safer and more resilient retrieval-augmented frameworks, paving the way for their reliable deployment in real-world applications.

摘要: 提取-增强生成(RAG)系统已经成为缓解LLM幻觉和提高其在知识密集型领域的表现的一种有前途的解决方案。然而，这些系统容易受到对抗性中毒攻击，在这种攻击中，注入检索数据库的恶意段落可能会误导模型生成实际不正确的输出。在本文中，我们研究了RAG系统的检索组件和生成组件，以了解如何增强其对此类攻击的健壮性。从提取的角度，我们分析了为什么以及如何提取对抗性语境，并评估了所检索的段落的质量如何影响下游生成。从一代人的角度，我们评估了LLMS的高级批判性思维和内部知识能力是否可以被用来减轻对手环境的影响，即使用怀疑提示作为一种自卫机制。我们的实验和发现为设计更安全、更具弹性的检索增强框架提供了可行的见解，为它们在现实世界的应用程序中的可靠部署铺平了道路。



## **7. Adversarial Attack Against Images Classification based on Generative Adversarial Networks**

基于生成对抗网络的图像分类对抗攻击 cs.CV

7 pages, 6 figures

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16662v1) [paper-pdf](http://arxiv.org/pdf/2412.16662v1)

**Authors**: Yahe Yang

**Abstract**: Adversarial attacks on image classification systems have always been an important problem in the field of machine learning, and generative adversarial networks (GANs), as popular models in the field of image generation, have been widely used in various novel scenarios due to their powerful generative capabilities. However, with the popularity of generative adversarial networks, the misuse of fake image technology has raised a series of security problems, such as malicious tampering with other people's photos and videos, and invasion of personal privacy. Inspired by the generative adversarial networks, this work proposes a novel adversarial attack method, aiming to gain insight into the weaknesses of the image classification system and improve its anti-attack ability. Specifically, the generative adversarial networks are used to generate adversarial samples with small perturbations but enough to affect the decision-making of the classifier, and the adversarial samples are generated through the adversarial learning of the training generator and the classifier. From extensive experiment analysis, we evaluate the effectiveness of the method on a classical image classification dataset, and the results show that our model successfully deceives a variety of advanced classifiers while maintaining the naturalness of adversarial samples.

摘要: 针对图像分类系统的对抗性攻击一直是机器学习领域的一个重要问题，而生成性对抗性网络(GANS)作为图像生成领域的热门模型，由于其强大的生成能力而被广泛应用于各种新颖的场景中。然而，随着生成性对抗网络的流行，虚假图像技术的滥用引发了一系列安全问题，如恶意篡改他人照片和视频、侵犯个人隐私等。受生成式对抗性网络的启发，本文提出了一种新颖的对抗性攻击方法，旨在洞察图像分类系统的弱点，提高其抗攻击能力。具体地说，生成式对抗性网络用于生成扰动较小但足以影响分类器决策的对抗性样本，并通过训练器和分类器的对抗性学习来生成对抗性样本。通过大量的实验分析，我们在一个经典的图像分类数据集上对该方法的有效性进行了评估，结果表明，我们的模型成功地欺骗了各种高级分类器，同时保持了对抗性样本的自然性。



## **8. PB-UAP: Hybrid Universal Adversarial Attack For Image Segmentation**

PB-UAP：图像分割的混合通用对抗攻击 cs.CV

Accepted by ICASSP 2025

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16651v1) [paper-pdf](http://arxiv.org/pdf/2412.16651v1)

**Authors**: Yufei Song, Ziqi Zhou, Minghui Li, Xianlong Wang, Menghao Deng, Wei Wan, Shengshan Hu, Leo Yu Zhang

**Abstract**: With the rapid advancement of deep learning, the model robustness has become a significant research hotspot, \ie, adversarial attacks on deep neural networks. Existing works primarily focus on image classification tasks, aiming to alter the model's predicted labels. Due to the output complexity and deeper network architectures, research on adversarial examples for segmentation models is still limited, particularly for universal adversarial perturbations. In this paper, we propose a novel universal adversarial attack method designed for segmentation models, which includes dual feature separation and low-frequency scattering modules. The two modules guide the training of adversarial examples in the pixel and frequency space, respectively. Experiments demonstrate that our method achieves high attack success rates surpassing the state-of-the-art methods, and exhibits strong transferability across different models.

摘要: 随着深度学习的快速发展，模型鲁棒性已成为一个重要的研究热点，即对深度神经网络的对抗性攻击。现有的工作主要集中在图像分类任务上，旨在改变模型的预测标签。由于输出复杂性和更深层次的网络架构，对分段模型对抗性示例的研究仍然有限，特别是对于普遍对抗性扰动。本文提出了一种针对分割模型设计的新型通用对抗攻击方法，其中包括双重特征分离和低频散射模块。这两个模块分别指导像素和频率空间中对抗性示例的训练。实验表明，我们的方法比最先进的方法具有更高的攻击成功率，并且在不同模型之间表现出很强的可移植性。



## **9. POEX: Policy Executable Embodied AI Jailbreak Attacks**

POEX：政策可执行性许可人工智能越狱攻击 cs.RO

Homepage: https://poex-eai-jailbreak.github.io/

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16633v1) [paper-pdf](http://arxiv.org/pdf/2412.16633v1)

**Authors**: Xuancun Lu, Zhengxian Huang, Xinfeng Li, Xiaoyu ji, Wenyuan Xu

**Abstract**: The integration of large language models (LLMs) into the planning module of Embodied Artificial Intelligence (Embodied AI) systems has greatly enhanced their ability to translate complex user instructions into executable policies. In this paper, we demystified how traditional LLM jailbreak attacks behave in the Embodied AI context. We conducted a comprehensive safety analysis of the LLM-based planning module of embodied AI systems against jailbreak attacks. Using the carefully crafted Harmful-RLbench, we accessed 20 open-source and proprietary LLMs under traditional jailbreak attacks, and highlighted two key challenges when adopting the prior jailbreak techniques to embodied AI contexts: (1) The harmful text output by LLMs does not necessarily induce harmful policies in Embodied AI context, and (2) even we can generate harmful policies, we have to guarantee they are executable in practice. To overcome those challenges, we propose Policy Executable (POEX) jailbreak attacks, where harmful instructions and optimized suffixes are injected into LLM-based planning modules, leading embodied AI to perform harmful actions in both simulated and physical environments. Our approach involves constraining adversarial suffixes to evade detection and fine-tuning a policy evaluater to improve the executability of harmful policies. We conducted extensive experiments on both a robotic arm embodied AI platform and simulators, to validate the attack and policy success rates on 136 harmful instructions from Harmful-RLbench. Our findings expose serious safety vulnerabilities in LLM-based planning modules, including the ability of POEX to be transferred across models. Finally, we propose mitigation strategies, such as safety-constrained prompts, pre- and post-planning checks, to address these vulnerabilities and ensure the safe deployment of embodied AI in real-world settings.

摘要: 将大型语言模型(LLM)集成到嵌入式人工智能(Embedded AI)系统的规划模块中，极大地增强了它们将复杂的用户指令转换为可执行策略的能力。在这篇文章中，我们揭开了传统的LLM越狱攻击在具体的人工智能上下文中的行为。我们对基于LLM的具体化人工智能系统抗越狱攻击规划模块进行了全面的安全分析。使用精心制作的有害RLbench，我们在传统越狱攻击下访问了20个开源和专有的LLM，并强调了采用先前的越狱技术来体现AI上下文时的两个关键挑战：(1)LLMS输出的有害文本不一定会导致体现AI上下文中的有害策略，以及(2)即使我们可以生成有害策略，我们也必须确保它们在实践中是可执行的。为了克服这些挑战，我们提出了策略可执行(POEX)越狱攻击，将有害指令和优化后缀注入基于LLM的规划模块，导致嵌入式AI在模拟和物理环境中执行有害操作。我们的方法包括限制敌意后缀以逃避检测，以及微调策略评估器以提高有害策略的可执行性。我们在一个机械臂体现的人工智能平台和模拟器上进行了广泛的实验，以验证对来自有害RLbench的136条有害指令的攻击和策略成功率。我们的发现暴露了基于LLM的计划模块中的严重安全漏洞，包括POEX跨模型传输的能力。最后，我们提出了缓解策略，如安全约束提示，规划前和规划后检查，以应对这些漏洞，并确保体现的人工智能在现实世界中的安全部署。



## **10. Automated Progressive Red Teaming**

自动化渐进式红色团队 cs.CR

Accepted by COLING 2025

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2407.03876v3) [paper-pdf](http://arxiv.org/pdf/2407.03876v3)

**Authors**: Bojian Jiang, Yi Jing, Tianhao Shen, Tong Wu, Qing Yang, Deyi Xiong

**Abstract**: Ensuring the safety of large language models (LLMs) is paramount, yet identifying potential vulnerabilities is challenging. While manual red teaming is effective, it is time-consuming, costly and lacks scalability. Automated red teaming (ART) offers a more cost-effective alternative, automatically generating adversarial prompts to expose LLM vulnerabilities. However, in current ART efforts, a robust framework is absent, which explicitly frames red teaming as an effectively learnable task. To address this gap, we propose Automated Progressive Red Teaming (APRT) as an effectively learnable framework. APRT leverages three core modules: an Intention Expanding LLM that generates diverse initial attack samples, an Intention Hiding LLM that crafts deceptive prompts, and an Evil Maker to manage prompt diversity and filter ineffective samples. The three modules collectively and progressively explore and exploit LLM vulnerabilities through multi-round interactions. In addition to the framework, we further propose a novel indicator, Attack Effectiveness Rate (AER) to mitigate the limitations of existing evaluation metrics. By measuring the likelihood of eliciting unsafe but seemingly helpful responses, AER aligns closely with human evaluations. Extensive experiments with both automatic and human evaluations, demonstrate the effectiveness of ARPT across both open- and closed-source LLMs. Specifically, APRT effectively elicits 54% unsafe yet useful responses from Meta's Llama-3-8B-Instruct, 50% from GPT-4o (API access), and 39% from Claude-3.5 (API access), showcasing its robust attack capability and transferability across LLMs (especially from open-source LLMs to closed-source LLMs).

摘要: 确保大型语言模型(LLM)的安全是最重要的，但识别潜在的漏洞是具有挑战性的。虽然手动红色团队是有效的，但它耗时、成本高，而且缺乏可扩展性。自动红色团队(ART)提供了一种更具成本效益的替代方案，可自动生成敌意提示以暴露LLM漏洞。然而，在目前的艺术努力中，缺乏一个强大的框架，它明确地将红色团队作为一项有效的可学习任务。为了弥补这一差距，我们提出了自动渐进红色团队(APRT)作为一种有效的可学习框架。APRT利用三个核心模块：用于生成不同初始攻击样本的意图扩展LLM，用于制作欺骗性提示的意图隐藏LLM，以及用于管理提示多样性和过滤无效样本的邪恶制造者。这三个模块通过多轮交互共同逐步探索和利用LLM漏洞。除了该框架外，我们进一步提出了一个新的指标--攻击效率(AER)，以缓解现有评估指标的局限性。通过衡量引发不安全但似乎有帮助的反应的可能性，AER与人类的评估密切一致。自动和人工评估的广泛实验证明了ARPT在开放源码和封闭源码LLM中的有效性。具体地说，APRT有效地从Meta的Llama-3-8B-Indict、GPT-40(API访问)和Claude-3.5(API访问)中引发了54%的不安全但有用的响应，展示了其强大的攻击能力和跨LLM(特别是从开源LLM到闭源LLM)的可转移性。



## **11. PGD-Imp: Rethinking and Unleashing Potential of Classic PGD with Dual Strategies for Imperceptible Adversarial Attacks**

PGD-Imp：通过双重策略重新思考和释放经典PVD的潜力，以应对难以感知的对抗攻击 cs.LG

accepted by ICASSP 2025

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.11168v2) [paper-pdf](http://arxiv.org/pdf/2412.11168v2)

**Authors**: Jin Li, Zitong Yu, Ziqiang He, Z. Jane Wang, Xiangui Kang

**Abstract**: Imperceptible adversarial attacks have recently attracted increasing research interests. Existing methods typically incorporate external modules or loss terms other than a simple $l_p$-norm into the attack process to achieve imperceptibility, while we argue that such additional designs may not be necessary. In this paper, we rethink the essence of imperceptible attacks and propose two simple yet effective strategies to unleash the potential of PGD, the common and classical attack, for imperceptibility from an optimization perspective. Specifically, the Dynamic Step Size is introduced to find the optimal solution with minimal attack cost towards the decision boundary of the attacked model, and the Adaptive Early Stop strategy is adopted to reduce the redundant strength of adversarial perturbations to the minimum level. The proposed PGD-Imperceptible (PGD-Imp) attack achieves state-of-the-art results in imperceptible adversarial attacks for both untargeted and targeted scenarios. When performing untargeted attacks against ResNet-50, PGD-Imp attains 100$\%$ (+0.3$\%$) ASR, 0.89 (-1.76) $l_2$ distance, and 52.93 (+9.2) PSNR with 57s (-371s) running time, significantly outperforming existing methods.

摘要: 潜伏的敌意攻击最近吸引了越来越多的研究兴趣。现有的方法通常在攻击过程中加入外部模块或损失项，而不是简单的$L_p$-范数来实现不可感知性，而我们认为这样的额外设计可能不是必要的。本文从优化的角度重新思考了不可察觉攻击的本质，并提出了两种简单而有效的策略来释放PGD攻击--普通攻击和经典攻击--的不可感知性。具体地，引入动态步长在攻击模型的决策边界附近寻找攻击代价最小的最优解，并采用自适应提前停止策略将敌方扰动的冗余强度降至最小。建议的PGD-Imp(PGD-Imp)攻击在非目标场景和目标场景中都实现了最先进的不可感知对手攻击。在对ResNet-50进行非定向攻击时，PGD-Imp在57s(-371s)的运行时间内获得了100$(+0.3$)ASR，0.89(-1.76)$L_2$距离和52.93(+9.2)PSNR，显著优于现有方法。



## **12. WiP: Deception-in-Depth Using Multiple Layers of Deception**

WiP：使用多层欺骗进行深度欺骗 cs.CR

Presented at HoTSoS 2024

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16430v1) [paper-pdf](http://arxiv.org/pdf/2412.16430v1)

**Authors**: Jason Landsborough, Neil C. Rowe, Thuy D. Nguyen, Sunny Fugate

**Abstract**: Deception is being increasingly explored as a cyberdefense strategy to protect operational systems. We are studying implementation of deception-in-depth strategies with initially three logical layers: network, host, and data. We draw ideas from military deception, network orchestration, software deception, file deception, fake honeypots, and moving-target defenses. We are building a prototype representing our ideas and will be testing it in several adversarial environments. We hope to show that deploying a broad range of deception techniques can be more effective in protecting systems than deploying single techniques. Unlike traditional deception methods that try to encourage active engagement from attackers to collect intelligence, we focus on deceptions that can be used on real machines to discourage attacks.

摘要: 欺骗作为一种保护操作系统的网络防御策略正在被越来越多地探索。我们正在研究深度欺骗策略的实施，最初分为三个逻辑层：网络、主机和数据。我们从军事欺骗、网络编排、软件欺骗、文件欺骗、假蜜罐和移动目标防御中汲取灵感。我们正在构建一个代表我们想法的原型，并将在几个对抗环境中对其进行测试。我们希望证明，部署广泛的欺骗技术比部署单一技术在保护系统方面更有效。与试图鼓励攻击者积极参与收集情报的传统欺骗方法不同，我们专注于可在真实机器上使用以阻止攻击的欺骗方法。



## **13. Chain-of-Scrutiny: Detecting Backdoor Attacks for Large Language Models**

审查链：检测大型语言模型的后门攻击 cs.CR

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2406.05948v2) [paper-pdf](http://arxiv.org/pdf/2406.05948v2)

**Authors**: Xi Li, Yusen Zhang, Renze Lou, Chen Wu, Jiaqi Wang

**Abstract**: Large Language Models (LLMs), especially those accessed via APIs, have demonstrated impressive capabilities across various domains. However, users without technical expertise often turn to (untrustworthy) third-party services, such as prompt engineering, to enhance their LLM experience, creating vulnerabilities to adversarial threats like backdoor attacks. Backdoor-compromised LLMs generate malicious outputs to users when inputs contain specific "triggers" set by attackers. Traditional defense strategies, originally designed for small-scale models, are impractical for API-accessible LLMs due to limited model access, high computational costs, and data requirements. To address these limitations, we propose Chain-of-Scrutiny (CoS) which leverages LLMs' unique reasoning abilities to mitigate backdoor attacks. It guides the LLM to generate reasoning steps for a given input and scrutinizes for consistency with the final output -- any inconsistencies indicating a potential attack. It is well-suited for the popular API-only LLM deployments, enabling detection at minimal cost and with little data. User-friendly and driven by natural language, it allows non-experts to perform the defense independently while maintaining transparency. We validate the effectiveness of CoS through extensive experiments on various tasks and LLMs, with results showing greater benefits for more powerful LLMs.

摘要: 大型语言模型(LLM)，特别是那些通过API访问的模型，已经在各个领域展示了令人印象深刻的能力。然而，没有技术专业知识的用户通常会求助于(不值得信任的)第三方服务，如提示工程，以增强他们的LLM体验，从而对后门攻击等对手威胁造成漏洞。当输入包含攻击者设置的特定“触发器”时，受后门攻击的LLM会向用户生成恶意输出。传统的防御策略最初是为小规模模型设计的，由于模型访问有限、计算成本高和数据要求高，对于API可访问的LLM来说是不切实际的。为了解决这些局限性，我们提出了审查链(CoS)，它利用LLMS的独特推理能力来减少后门攻击。它指导LLM为给定的输入生成推理步骤，并仔细检查与最终输出的一致性--任何指示潜在攻击的不一致。它非常适合流行的纯API LLM部署，能够以最低的成本和很少的数据进行检测。它用户友好，由自然语言驱动，允许非专家独立进行辩护，同时保持透明度。我们通过在不同任务和LLM上的大量实验验证了CoS的有效性，结果表明，更强大的LLM具有更大的好处。



## **14. EMPRA: Embedding Perturbation Rank Attack against Neural Ranking Models**

EMPRA：针对神经排名模型的嵌入扰动排名攻击 cs.IR

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.16382v1) [paper-pdf](http://arxiv.org/pdf/2412.16382v1)

**Authors**: Amin Bigdeli, Negar Arabzadeh, Ebrahim Bagheri, Charles L. A. Clarke

**Abstract**: Recent research has shown that neural information retrieval techniques may be susceptible to adversarial attacks. Adversarial attacks seek to manipulate the ranking of documents, with the intention of exposing users to targeted content. In this paper, we introduce the Embedding Perturbation Rank Attack (EMPRA) method, a novel approach designed to perform adversarial attacks on black-box Neural Ranking Models (NRMs). EMPRA manipulates sentence-level embeddings, guiding them towards pertinent context related to the query while preserving semantic integrity. This process generates adversarial texts that seamlessly integrate with the original content and remain imperceptible to humans. Our extensive evaluation conducted on the widely-used MS MARCO V1 passage collection demonstrate the effectiveness of EMPRA against a wide range of state-of-the-art baselines in promoting a specific set of target documents within a given ranked results. Specifically, EMPRA successfully achieves a re-ranking of almost 96% of target documents originally ranked between 51-100 to rank within the top 10. Furthermore, EMPRA does not depend on surrogate models for adversarial text generation, enhancing its robustness against different NRMs in realistic settings.

摘要: 最近的研究表明，神经信息检索技术可能容易受到对抗性攻击。敌意攻击试图操纵文档的排名，目的是让用户接触到有针对性的内容。本文介绍了一种新的针对黑盒神经网络排名模型(NRM)的对抗性攻击方法--嵌入扰动等级攻击方法。EMPRA操作语句级别的嵌入，引导它们指向与查询相关的上下文，同时保持语义完整性。这一过程产生了与原始内容无缝集成的对抗性文本，并且对人类来说仍然是不可感知的。我们对广泛使用的MS Marco V1文章集进行了广泛的评估，证明了EMPRA在推广给定排名结果中的一组特定目标文档方面相对于广泛的最先进基线的有效性。具体地说，EMPRA成功地实现了几乎96%的目标文档的重新排序，这些文档最初的排名在51-100之间，进入前10名。此外，EMPRA不依赖于生成敌意文本的代理模型，增强了它在现实环境中对不同NRM的稳健性。



## **15. Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context**

人类可读的对抗性预言：使用情境背景对LLM漏洞的调查 cs.CL

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.16359v1) [paper-pdf](http://arxiv.org/pdf/2412.16359v1)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous research on LLM vulnerabilities often relied on nonsensical adversarial prompts, which were easily detectable by automated methods. We address this gap by focusing on human-readable adversarial prompts, a more realistic and potent threat. Our key contributions are situation-driven attacks leveraging movie scripts to create contextually relevant, human-readable prompts that successfully deceive LLMs, adversarial suffix conversion to transform nonsensical adversarial suffixes into meaningful text, and AdvPrompter with p-nucleus sampling, a method to generate diverse, human-readable adversarial suffixes, improving attack efficacy in models like GPT-3.5 and Gemma 7B. Our findings demonstrate that LLMs can be tricked by sophisticated adversaries into producing harmful responses with human-readable adversarial prompts and that there exists a scope for improvement when it comes to robust LLMs.

摘要: 之前对LLM漏洞的研究通常依赖于毫无意义的对抗提示，这些提示很容易通过自动化方法检测到。我们通过关注人类可读的对抗提示来解决这一差距，这是一种更现实、更强大的威胁。我们的主要贡献是情景驱动攻击，利用电影脚本创建与上下文相关的、人类可读的提示，从而成功欺骗LLM，对抗性后缀转换将无意义的对抗性后缀转换为有意义的文本，以及具有p核采样的Advancer，这是一种生成多样化、人类可读的对抗性后缀的方法，提高了GPT-3.5和Gemma 7 B等模型中的攻击功效。我们的研究结果表明，LLM可能会被复杂的对手欺骗，通过人类可读的对抗提示产生有害反应，并且在稳健的LLM方面还有改进的空间。



## **16. Texture- and Shape-based Adversarial Attacks for Vehicle Detection in Synthetic Overhead Imagery**

基于纹理和形状的对抗攻击用于合成头顶图像中的车辆检测 cs.CV

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.16358v1) [paper-pdf](http://arxiv.org/pdf/2412.16358v1)

**Authors**: Mikael Yeghiazaryan, Sai Abhishek Siddhartha Namburu, Emily Kim, Stanislav Panev, Celso de Melo, Brent Lance, Fernando De la Torre, Jessica K. Hodgins

**Abstract**: Detecting vehicles in aerial images can be very challenging due to complex backgrounds, small resolution, shadows, and occlusions. Despite the effectiveness of SOTA detectors such as YOLO, they remain vulnerable to adversarial attacks (AAs), compromising their reliability. Traditional AA strategies often overlook the practical constraints of physical implementation, focusing solely on attack performance. Our work addresses this issue by proposing practical implementation constraints for AA in texture and/or shape. These constraints include pixelation, masking, limiting the color palette of the textures, and constraining the shape modifications. We evaluated the proposed constraints through extensive experiments using three widely used object detector architectures, and compared them to previous works. The results demonstrate the effectiveness of our solutions and reveal a trade-off between practicality and performance. Additionally, we introduce a labeled dataset of overhead images featuring vehicles of various categories. We will make the code/dataset public upon paper acceptance.

摘要: 由于复杂的背景、小的分辨率、阴影和遮挡，在航空图像中检测车辆可能非常具有挑战性。尽管像YOLO这样的SOTA检测器很有效，但它们仍然容易受到对手攻击(AA)，从而影响了它们的可靠性。传统的AA策略往往忽略了物理实施的实际约束，只关注攻击性能。我们的工作通过提出纹理和/或形状上的AA的实际实现约束来解决这个问题。这些约束包括像素化、遮罩、限制纹理的调色板以及约束形状修改。我们通过使用三种广泛使用的目标检测器体系结构的大量实验来评估所提出的约束，并将它们与以前的工作进行了比较。结果表明，我们的解决方案是有效的，并揭示了实用性和性能之间的权衡。此外，我们引入了一个带有标签的数据集，其中包含各种类别的车辆的头顶图像。我们将在纸质验收后公开代码/数据集。



## **17. Competence-Based Analysis of Language Models**

基于能力的语言模型分析 cs.CL

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2303.00333v5) [paper-pdf](http://arxiv.org/pdf/2303.00333v5)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent successes of large, pretrained neural language models (LLMs), comparatively little is known about the representations of linguistic structure they learn during pretraining, which can lead to unexpected behaviors in response to prompt variation or distribution shift. To better understand these models and behaviors, we introduce a general model analysis framework to study LLMs with respect to their representation and use of human-interpretable linguistic properties. Our framework, CALM (Competence-based Analysis of Language Models), is designed to investigate LLM competence in the context of specific tasks by intervening on models' internal representations of different linguistic properties using causal probing, and measuring models' alignment under these interventions with a given ground-truth causal model of the task. We also develop a new approach for performing causal probing interventions using gradient-based adversarial attacks, which can target a broader range of properties and representations than prior techniques. Finally, we carry out a case study of CALM using these interventions to analyze and compare LLM competence across a variety of lexical inference tasks, showing that CALM can be used to explain behaviors across these tasks.

摘要: 尽管最近大型的预训练神经语言模型(LLM)取得了成功，但人们对它们在预训练中学习的语言结构的表征知之甚少，这可能会导致对迅速变化或分布变化的意外行为。为了更好地理解这些模型和行为，我们引入了一个通用的模型分析框架，从它们对人类可解释的语言属性的表示和使用方面来研究LLM。基于能力的语言模型分析框架旨在通过因果探究干预模型对不同语言属性的内部表征，并测量模型在这些干预下与给定任务的基本事实因果模型的一致性，从而考察特定任务背景下的语言学习能力。我们还开发了一种使用基于梯度的对抗性攻击来执行因果探测干预的新方法，该方法可以针对比现有技术更广泛的属性和表示。最后，我们使用这些干预手段对CAMLE进行了个案研究，分析和比较了不同词汇推理任务的LLM能力，结果表明CAMPE可以用来解释这些任务中的行为。



## **18. Augment then Smooth: Reconciling Differential Privacy with Certified Robustness**

增强然后平滑：通过认证的稳健性来实现差异隐私 cs.LG

29 pages, 19 figures. Accepted at TMLR in 2024. Link:  https://openreview.net/forum?id=YN0IcnXqsr

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2306.08656v3) [paper-pdf](http://arxiv.org/pdf/2306.08656v3)

**Authors**: Jiapeng Wu, Atiyeh Ashari Ghomi, David Glukhov, Jesse C. Cresswell, Franziska Boenisch, Nicolas Papernot

**Abstract**: Machine learning models are susceptible to a variety of attacks that can erode trust, including attacks against the privacy of training data, and adversarial examples that jeopardize model accuracy. Differential privacy and certified robustness are effective frameworks for combating these two threats respectively, as they each provide future-proof guarantees. However, we show that standard differentially private model training is insufficient for providing strong certified robustness guarantees. Indeed, combining differential privacy and certified robustness in a single system is non-trivial, leading previous works to introduce complex training schemes that lack flexibility. In this work, we present DP-CERT, a simple and effective method that achieves both privacy and robustness guarantees simultaneously by integrating randomized smoothing into standard differentially private model training. Compared to the leading prior work, DP-CERT gives up to a 2.5% increase in certified accuracy for the same differential privacy guarantee on CIFAR10. Through in-depth per-sample metric analysis, we find that larger certifiable radii correlate with smaller local Lipschitz constants, and show that DP-CERT effectively reduces Lipschitz constants compared to other differentially private training methods. The code is available at github.com/layer6ai-labs/dp-cert.

摘要: 机器学习模型容易受到各种可能侵蚀信任的攻击，包括对训练数据隐私的攻击，以及危及模型准确性的敌意示例。差异隐私和认证的健壮性分别是对抗这两种威胁的有效框架，因为它们都提供了面向未来的保证。然而，我们表明，标准的差分私有模型训练不足以提供强大的认证稳健性保证。事实上，在单个系统中结合不同的隐私和经过认证的健壮性并不是一件容易的事情，这导致以前的工作引入了缺乏灵活性的复杂培训方案。在这项工作中，我们提出了一种简单而有效的方法DP-CERT，通过将随机化平滑与标准的差分私有模型训练相结合，同时实现了保密性和稳健性。与领先的以前的工作相比，DP-CERT在CIFAR10上提供相同的差异隐私保证的认证准确率最多提高了2.5%。通过深入的每样本度量分析，我们发现较大的可证明半径与较小的局部Lipschitz常数相关，并表明与其他差分私有训练方法相比，DP-CERT有效地降低了Lipschitz常数。代码可以在githorb.com/layer6ai-Labs/dp-cert上找到。



## **19. Watertox: The Art of Simplicity in Universal Attacks A Cross-Model Framework for Robust Adversarial Generation**

Watertox：普遍攻击中的简单性艺术稳健对抗生成的跨模型框架 cs.CV

18 pages, 4 figures, 3 tables. Advances a novel method for generating  cross-model transferable adversarial perturbations through a two-stage FGSM  process and architectural ensemble voting mechanism

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.15924v1) [paper-pdf](http://arxiv.org/pdf/2412.15924v1)

**Authors**: Zhenghao Gao, Shengjie Xu, Meixi Chen, Fangyao Zhao

**Abstract**: Contemporary adversarial attack methods face significant limitations in cross-model transferability and practical applicability. We present Watertox, an elegant adversarial attack framework achieving remarkable effectiveness through architectural diversity and precision-controlled perturbations. Our two-stage Fast Gradient Sign Method combines uniform baseline perturbations ($\epsilon_1 = 0.1$) with targeted enhancements ($\epsilon_2 = 0.4$). The framework leverages an ensemble of complementary architectures, from VGG to ConvNeXt, synthesizing diverse perspectives through an innovative voting mechanism. Against state-of-the-art architectures, Watertox reduces model accuracy from 70.6% to 16.0%, with zero-shot attacks achieving up to 98.8% accuracy reduction against unseen architectures. These results establish Watertox as a significant advancement in adversarial methodologies, with promising applications in visual security systems and CAPTCHA generation.

摘要: 当代对抗攻击方法在跨模型可移植性和实际适用性方面面临着显着的限制。我们介绍了Watertox，这是一个优雅的对抗攻击框架，通过架构多样性和精确控制的扰动实现了显着的有效性。我们的两阶段快速梯度符号法将均匀基线扰动（$\epsilon_1 = 0.1$）与有针对性的增强（$\epsilon_2 = 0.4$）相结合。该框架利用了从VGG到ConvNeXt的一系列互补架构，通过创新的投票机制综合了不同的观点。针对最先进的架构，Watertox将模型准确性从70.6%降低到16.0%，针对未见架构，零攻击的准确性降低高达98.8%。这些结果使Watertox成为对抗方法学的重大进步，在视觉安全系统和验证码生成方面具有广阔的应用前景。



## **20. Client-Side Patching against Backdoor Attacks in Federated Learning**

客户端修补联邦学习中的后门攻击 cs.CR

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.10605v2) [paper-pdf](http://arxiv.org/pdf/2412.10605v2)

**Authors**: Borja Molina-Coronado

**Abstract**: Federated learning is a versatile framework for training models in decentralized environments. However, the trust placed in clients makes federated learning vulnerable to backdoor attacks launched by malicious participants. While many defenses have been proposed, they often fail short when facing heterogeneous data distributions among participating clients. In this paper, we propose a novel defense mechanism for federated learning systems designed to mitigate backdoor attacks on the clients-side. Our approach leverages adversarial learning techniques and model patching to neutralize the impact of backdoor attacks. Through extensive experiments on the MNIST and Fashion-MNIST datasets, we demonstrate that our defense effectively reduces backdoor accuracy, outperforming existing state-of-the-art defenses, such as LFighter, FLAME, and RoseAgg, in i.i.d. and non-i.i.d. scenarios, while maintaining competitive or superior accuracy on clean data.

摘要: 联邦学习是去中心化环境中训练模型的通用框架。然而，对客户的信任使得联邦学习容易受到恶意参与者发起的后门攻击。虽然已经提出了许多防御措施，但当面临参与客户端之间的异类数据分布时，它们往往会失败。在本文中，我们提出了一种新型的联邦学习系统防御机制，旨在减轻客户端的后门攻击。我们的方法利用对抗学习技术和模型修补来抵消后门攻击的影响。通过对MNIST和Fashion-MNIST数据集的广泛实验，我们证明我们的防御有效地降低了后门准确性，优于现有的最先进防御，例如LFighter、FLAME和RoseAgg，i.i. d。和非i.i.d.场景，同时在干净数据上保持有竞争力或卓越的准确性。



## **21. PoisonCatcher: Revealing and Identifying LDP Poisoning Attacks in IIoT**

Poison Catcher：揭露和识别IIoT中的自民党中毒攻击 cs.CR

12 pages,5 figures, 3 tables

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.15704v1) [paper-pdf](http://arxiv.org/pdf/2412.15704v1)

**Authors**: Lisha Shuai, Shaofeng Tan, Nan Zhang, Jiamin Zhang, Min Zhang, Xiaolong Yang

**Abstract**: Local Differential Privacy (LDP) is widely adopted in the Industrial Internet of Things (IIoT) for its lightweight, decentralized, and scalable nature. However, its perturbation-based privacy mechanism makes it difficult to distinguish between uncontaminated and tainted data, encouraging adversaries to launch poisoning attacks. While LDP provides some resilience against minor poisoning, it lacks robustness in IIoT with dynamic networks and substantial real-time data flows. Effective countermeasures for such attacks are still underdeveloped. This work narrows the critical gap by revealing and identifying LDP poisoning attacks in IIoT. We begin by deepening the understanding of such attacks, revealing novel threats that arise from the interplay between LDP indistinguishability and IIoT complexity. This exploration uncovers a novel rule-poisoning attack, and presents a general attack formulation by unifying it with input-poisoning and output-poisoning. Furthermore, two key attack impacts, i.e., Statistical Query Result (SQR) accuracy degradation and inter-dataset correlations disruption, along with two characteristics: attack patterns unstable and poisoned data stealth are revealed. From this, we propose PoisonCatcher, a four-stage solution that detects LDP poisoning attacks and identifies specific contaminated data points. It utilizes temporal similarity, attribute correlation, and time-series stability analysis to detect datasets exhibiting SQR accuracy degradation, inter-dataset disruptions, and unstable patterns. Enhanced feature engineering is used to extract subtle poisoning signatures, enabling machine learning models to identify specific contamination. Experimental evaluations show the effectiveness, achieving state-of-the-art performance with average precision and recall rates of 86.17% and 97.5%, respectively, across six representative attack scenarios.

摘要: 本地差分隐私(LDP)以其轻量级、分散化和可扩展的特点在工业物联网(IIoT)中被广泛采用。然而，其基于扰动的隐私机制使其难以区分未受污染的数据和受污染的数据，从而鼓励对手发起中毒攻击。虽然LDP对轻微中毒提供了一定的弹性，但它在具有动态网络和大量实时数据流的IIoT中缺乏健壮性。针对这类攻击的有效对策仍不发达。这项工作通过揭示和识别IIoT中的LDP中毒攻击缩小了关键差距。我们首先加深对这类攻击的理解，揭示自民党的不可区分性和IIoT复杂性之间的相互作用产生的新威胁。这一探索揭示了一种新的规则中毒攻击，并将其与输入中毒和输出中毒相统一，给出了一种通用的攻击公式。此外，还揭示了统计查询结果(SQR)准确性下降和数据集间相关性破坏这两个关键攻击影响，以及攻击模式不稳定和有毒数据隐藏的两个特征。在此基础上，我们提出了PoisonCatcher，这是一种四阶段解决方案，可以检测LDP中毒攻击并识别特定的受污染数据点。它利用时间相似性、属性相关性和时间序列稳定性分析来检测表现出SQR精度降低、数据集间中断和不稳定模式的数据集。增强的特征工程被用来提取微妙的中毒特征，使机器学习模型能够识别特定的污染。实验评估表明了该方法的有效性，在6种典型攻击场景下，平均准确率和召回率分别达到了86.17%和97.5%。



## **22. CAMH: Advancing Model Hijacking Attack in Machine Learning**

CAMH：机器学习中推进模型劫持攻击 cs.CR

Accepted by AAAI 2025

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2408.13741v2) [paper-pdf](http://arxiv.org/pdf/2408.13741v2)

**Authors**: Xing He, Jiahao Chen, Yuwen Pu, Qingming Li, Chunyi Zhou, Yingcai Wu, Jinbao Li, Shouling Ji

**Abstract**: In the burgeoning domain of machine learning, the reliance on third-party services for model training and the adoption of pre-trained models have surged. However, this reliance introduces vulnerabilities to model hijacking attacks, where adversaries manipulate models to perform unintended tasks, leading to significant security and ethical concerns, like turning an ordinary image classifier into a tool for detecting faces in pornographic content, all without the model owner's knowledge. This paper introduces Category-Agnostic Model Hijacking (CAMH), a novel model hijacking attack method capable of addressing the challenges of class number mismatch, data distribution divergence, and performance balance between the original and hijacking tasks. CAMH incorporates synchronized training layers, random noise optimization, and a dual-loop optimization approach to ensure minimal impact on the original task's performance while effectively executing the hijacking task. We evaluate CAMH across multiple benchmark datasets and network architectures, demonstrating its potent attack effectiveness while ensuring minimal degradation in the performance of the original task.

摘要: 在蓬勃发展的机器学习领域，对第三方服务进行模型培训的依赖和对预先训练模型的采用激增。然而，这种依赖引入了漏洞来模拟劫持攻击，即攻击者操纵模型执行意想不到的任务，导致重大的安全和伦理问题，比如将普通图像分类器变成在色情内容中检测人脸的工具，所有这些都是在模型所有者不知道的情况下进行的。介绍了一种新的模型劫持攻击方法--类别不可知模型劫持(CAMH)，它能够解决类别号不匹配、数据分布差异以及原始任务和劫持任务之间的性能平衡等问题。CAMH结合了同步训练层、随机噪声优化和双环优化方法，以确保在有效执行劫持任务的同时对原始任务的性能影响最小。我们在多个基准数据集和网络体系结构上对CAMH进行了评估，展示了其强大的攻击效率，同时确保原始任务的性能降级最小。



## **23. A Stealthy Wrongdoer: Feature-Oriented Reconstruction Attack against Split Learning**

一个偷偷摸摸的犯错者：针对分裂学习的以冲突为导向的重建攻击 cs.CR

Accepted to CVPR 2024

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2405.04115v3) [paper-pdf](http://arxiv.org/pdf/2405.04115v3)

**Authors**: Xiaoyang Xu, Mengda Yang, Wenzhe Yi, Ziang Li, Juan Wang, Hongxin Hu, Yong Zhuang, Yaxin Liu

**Abstract**: Split Learning (SL) is a distributed learning framework renowned for its privacy-preserving features and minimal computational requirements. Previous research consistently highlights the potential privacy breaches in SL systems by server adversaries reconstructing training data. However, these studies often rely on strong assumptions or compromise system utility to enhance attack performance. This paper introduces a new semi-honest Data Reconstruction Attack on SL, named Feature-Oriented Reconstruction Attack (FORA). In contrast to prior works, FORA relies on limited prior knowledge, specifically that the server utilizes auxiliary samples from the public without knowing any client's private information. This allows FORA to conduct the attack stealthily and achieve robust performance. The key vulnerability exploited by FORA is the revelation of the model representation preference in the smashed data output by victim client. FORA constructs a substitute client through feature-level transfer learning, aiming to closely mimic the victim client's representation preference. Leveraging this substitute client, the server trains the attack model to effectively reconstruct private data. Extensive experiments showcase FORA's superior performance compared to state-of-the-art methods. Furthermore, the paper systematically evaluates the proposed method's applicability across diverse settings and advanced defense strategies.

摘要: Split Learning(SL)是一种分布式学习框架，以其隐私保护功能和最小的计算要求而闻名。以前的研究一直强调，通过服务器对手重建训练数据，SL系统中潜在的隐私泄露。然而，这些研究往往依赖强假设或折衷系统效用来提高攻击性能。介绍了一种新的基于SL的半诚实数据重构攻击--面向特征的重构攻击(FORA)。与以前的工作不同，FORA依赖于有限的先验知识，特别是服务器使用来自公共的辅助样本，而不知道任何客户的私人信息。这使得Fora能够悄悄地进行攻击，并实现稳健的性能。Fora利用的关键漏洞是受害者客户端输出的粉碎数据中暴露的模型表示首选项。FORA通过特征级迁移学习构造了一个替代客户，旨在更好地模拟受害客户的表征偏好。利用这个替代客户端，服务器训练攻击模型以有效地重建私有数据。广泛的实验表明，与最先进的方法相比，FORA的性能更优越。此外，本文还系统地评估了该方法在不同环境和先进防御策略下的适用性。



## **24. JailPO: A Novel Black-box Jailbreak Framework via Preference Optimization against Aligned LLMs**

JailPO：通过针对一致LLM的偏好优化的新型黑匣子越狱框架 cs.CR

Accepted by AAAI 2025

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.15623v1) [paper-pdf](http://arxiv.org/pdf/2412.15623v1)

**Authors**: Hongyi Li, Jiawei Ye, Jie Wu, Tianjie Yan, Chu Wang, Zhixin Li

**Abstract**: Large Language Models (LLMs) aligned with human feedback have recently garnered significant attention. However, it remains vulnerable to jailbreak attacks, where adversaries manipulate prompts to induce harmful outputs. Exploring jailbreak attacks enables us to investigate the vulnerabilities of LLMs and further guides us in enhancing their security. Unfortunately, existing techniques mainly rely on handcrafted templates or generated-based optimization, posing challenges in scalability, efficiency and universality. To address these issues, we present JailPO, a novel black-box jailbreak framework to examine LLM alignment. For scalability and universality, JailPO meticulously trains attack models to automatically generate covert jailbreak prompts. Furthermore, we introduce a preference optimization-based attack method to enhance the jailbreak effectiveness, thereby improving efficiency. To analyze model vulnerabilities, we provide three flexible jailbreak patterns. Extensive experiments demonstrate that JailPO not only automates the attack process while maintaining effectiveness but also exhibits superior performance in efficiency, universality, and robustness against defenses compared to baselines. Additionally, our analysis of the three JailPO patterns reveals that attacks based on complex templates exhibit higher attack strength, whereas covert question transformations elicit riskier responses and are more likely to bypass defense mechanisms.

摘要: 与人类反馈相一致的大语言模型(LLM)最近得到了极大的关注。然而，它仍然容易受到越狱攻击，对手操纵提示来诱导有害输出。探索越狱攻击使我们能够调查LLMS的漏洞，并进一步指导我们增强其安全性。遗憾的是，现有技术主要依赖于手工制作的模板或基于生成的优化，在可伸缩性、效率和通用性方面提出了挑战。为了解决这些问题，我们提出了JailPO，一个新的黑盒越狱框架来检查LLM对齐。为了提高可扩展性和通用性，JailPO精心训练攻击模型，以自动生成隐蔽的越狱提示。此外，我们引入了一种基于偏好优化的攻击方法来增强越狱的有效性，从而提高了效率。为了分析模型漏洞，我们提供了三种灵活的越狱模式。大量的实验表明，JailPO不仅在保持有效性的同时实现了攻击过程的自动化，而且与基线相比，在效率、通用性和对防御的健壮性方面表现出了优越的性能。此外，我们对三种JailPO模式的分析表明，基于复杂模板的攻击表现出更高的攻击强度，而隐蔽的问题转换会引发更高的风险响应，并且更有可能绕过防御机制。



## **25. Adversarial Robustness through Dynamic Ensemble Learning**

通过动态参与学习实现对抗鲁棒性 cs.CR

This is the accepted version of our paper for the 2024 IEEE Silchar  Subsection Conference (IEEE SILCON24), held from November 15 to 17, 2024, at  the National Institute of Technology (NIT), Agartala, India. The paper is 6  pages long and contains 3 Figures and 7 Tables

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.16254v1) [paper-pdf](http://arxiv.org/pdf/2412.16254v1)

**Authors**: Hetvi Waghela, Jaydip Sen, Sneha Rakshit

**Abstract**: Adversarial attacks pose a significant threat to the reliability of pre-trained language models (PLMs) such as GPT, BERT, RoBERTa, and T5. This paper presents Adversarial Robustness through Dynamic Ensemble Learning (ARDEL), a novel scheme designed to enhance the robustness of PLMs against such attacks. ARDEL leverages the diversity of multiple PLMs and dynamically adjusts the ensemble configuration based on input characteristics and detected adversarial patterns. Key components of ARDEL include a meta-model for dynamic weighting, an adversarial pattern detection module, and adversarial training with regularization techniques. Comprehensive evaluations using standardized datasets and various adversarial attack scenarios demonstrate that ARDEL significantly improves robustness compared to existing methods. By dynamically reconfiguring the ensemble to prioritize the most robust models for each input, ARDEL effectively reduces attack success rates and maintains higher accuracy under adversarial conditions. This work contributes to the broader goal of developing more secure and trustworthy AI systems for real-world NLP applications, offering a practical and scalable solution to enhance adversarial resilience in PLMs.

摘要: 对抗性攻击对GPT、BERT、Roberta和T5等预先训练的语言模型(PLM)的可靠性构成了严重威胁。本文通过动态集成学习(ARDEL)提出了一种新的方案，旨在增强PLM对此类攻击的健壮性。ARDELL利用多个PLM的多样性，根据输入特征和检测到的对抗性模式动态调整集成配置。ARDELL的关键组件包括用于动态加权的元模型、对抗性模式检测模块以及使用正则化技术的对抗性训练。使用标准化数据集和各种对抗性攻击场景进行的综合评估表明，与现有方法相比，ARDEL显著提高了稳健性。通过动态重新配置集合，为每个输入确定最健壮的模型的优先顺序，Ardel有效地降低了攻击成功率，并在对抗性条件下保持了更高的准确性。这项工作有助于为现实世界的NLP应用开发更安全和可信的AI系统，为增强PLM中的对抗弹性提供了一个实用和可扩展的解决方案。



## **26. Time Will Tell: Timing Side Channels via Output Token Count in Large Language Models**

时间会证明一切：通过大型语言模型中的输出令牌计数计时侧通道 cs.LG

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.15431v1) [paper-pdf](http://arxiv.org/pdf/2412.15431v1)

**Authors**: Tianchen Zhang, Gururaj Saileshwar, David Lie

**Abstract**: This paper demonstrates a new side-channel that enables an adversary to extract sensitive information about inference inputs in large language models (LLMs) based on the number of output tokens in the LLM response. We construct attacks using this side-channel in two common LLM tasks: recovering the target language in machine translation tasks and recovering the output class in classification tasks. In addition, due to the auto-regressive generation mechanism in LLMs, an adversary can recover the output token count reliably using a timing channel, even over the network against a popular closed-source commercial LLM. Our experiments show that an adversary can learn the output language in translation tasks with more than 75% precision across three different models (Tower, M2M100, MBart50). Using this side-channel, we also show the input class in text classification tasks can be leaked out with more than 70% precision from open-source LLMs like Llama-3.1, Llama-3.2, Gemma2, and production models like GPT-4o. Finally, we propose tokenizer-, system-, and prompt-based mitigations against the output token count side-channel.

摘要: 本文提出了一种新的边通道，使攻击者能够根据大语言模型响应中输出令牌的数量来提取与大语言模型中推理输入有关的敏感信息。我们在两个常见的LLM任务中利用该副通道构造攻击：在机器翻译任务中恢复目标语言和在分类任务中恢复输出类。此外，由于LLMS中的自动回归生成机制，攻击者可以使用定时通道可靠地恢复输出令牌计数，即使是在与流行的封闭源代码商业LLM的网络上也是如此。我们的实验表明，在三种不同的模型(Tower，M2M100，MBart50)上，对手可以在翻译任务中以75%以上的准确率学习输出语言。使用这个侧通道，我们还展示了文本分类任务中的输入类可以从开源LLMS(如Llama-3.1、Llama-3.2、Gemma2)和生产模型(如GPT-4o)以超过70%的精度泄漏出来。最后，我们针对输出令牌计数侧通道提出了基于标记器、基于系统和基于提示的缓解措施。



## **27. Towards Adversarially Robust Dataset Distillation by Curvature Regularization**

通过弯曲正规化实现对抗稳健的数据集蒸馏 cs.LG

17 pages, 3 figures

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2403.10045v2) [paper-pdf](http://arxiv.org/pdf/2403.10045v2)

**Authors**: Eric Xue, Yijiang Li, Haoyang Liu, Peiran Wang, Yifan Shen, Haohan Wang

**Abstract**: Dataset distillation (DD) allows datasets to be distilled to fractions of their original size while preserving the rich distributional information so that models trained on the distilled datasets can achieve a comparable accuracy while saving significant computational loads. Recent research in this area has been focusing on improving the accuracy of models trained on distilled datasets. In this paper, we aim to explore a new perspective of DD. We study how to embed adversarial robustness in distilled datasets, so that models trained on these datasets maintain the high accuracy and meanwhile acquire better adversarial robustness. We propose a new method that achieves this goal by incorporating curvature regularization into the distillation process with much less computational overhead than standard adversarial training. Extensive empirical experiments suggest that our method not only outperforms standard adversarial training on both accuracy and robustness with less computation overhead but is also capable of generating robust distilled datasets that can withstand various adversarial attacks.

摘要: 数据集蒸馏(DD)允许将数据集提取到原始大小的一小部分，同时保留丰富的分布信息，以便在提取的数据集上训练的模型可以在节省大量计算量的同时获得类似的精度。最近在这一领域的研究一直集中在提高在提取的数据集上训练的模型的准确性。在本文中，我们旨在探索一种新的研究视角。我们研究了如何在提取的数据集中嵌入对抗健壮性，使在这些数据集上训练的模型在保持较高准确率的同时获得更好的对抗健壮性。我们提出了一种新的方法，通过将曲率正则化引入到蒸馏过程中来实现这一目标，与标准的对抗性训练相比，计算开销要小得多。大量的实验表明，我们的方法不仅在准确率和稳健性方面都优于标准的对抗性训练，而且还能够生成能够抵抗各种对抗性攻击的健壮的提取数据集。



## **28. AutoTrust: Benchmarking Trustworthiness in Large Vision Language Models for Autonomous Driving**

AutoTrust：自动驾驶大视觉语言模型的可信度基准 cs.CV

55 pages, 14 figures

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.15206v1) [paper-pdf](http://arxiv.org/pdf/2412.15206v1)

**Authors**: Shuo Xing, Hongyuan Hua, Xiangbo Gao, Shenzhe Zhu, Renjie Li, Kexin Tian, Xiaopeng Li, Heng Huang, Tianbao Yang, Zhangyang Wang, Yang Zhou, Huaxiu Yao, Zhengzhong Tu

**Abstract**: Recent advancements in large vision language models (VLMs) tailored for autonomous driving (AD) have shown strong scene understanding and reasoning capabilities, making them undeniable candidates for end-to-end driving systems. However, limited work exists on studying the trustworthiness of DriveVLMs -- a critical factor that directly impacts public transportation safety. In this paper, we introduce AutoTrust, a comprehensive trustworthiness benchmark for large vision-language models in autonomous driving (DriveVLMs), considering diverse perspectives -- including trustfulness, safety, robustness, privacy, and fairness. We constructed the largest visual question-answering dataset for investigating trustworthiness issues in driving scenarios, comprising over 10k unique scenes and 18k queries. We evaluated six publicly available VLMs, spanning from generalist to specialist, from open-source to commercial models. Our exhaustive evaluations have unveiled previously undiscovered vulnerabilities of DriveVLMs to trustworthiness threats. Specifically, we found that the general VLMs like LLaVA-v1.6 and GPT-4o-mini surprisingly outperform specialized models fine-tuned for driving in terms of overall trustworthiness. DriveVLMs like DriveLM-Agent are particularly vulnerable to disclosing sensitive information. Additionally, both generalist and specialist VLMs remain susceptible to adversarial attacks and struggle to ensure unbiased decision-making across diverse environments and populations. Our findings call for immediate and decisive action to address the trustworthiness of DriveVLMs -- an issue of critical importance to public safety and the welfare of all citizens relying on autonomous transportation systems. Our benchmark is publicly available at \url{https://github.com/taco-group/AutoTrust}, and the leaderboard is released at \url{https://taco-group.github.io/AutoTrust/}.

摘要: 为自动驾驶(AD)量身定做的大型视觉语言模型(VLM)最近的进步显示了强大的场景理解和推理能力，使它们成为端到端驾驶系统的不可否认的候选者。然而，目前对DriveVLMS可信度的研究工作有限--这是直接影响公共交通安全的关键因素。在本文中，我们介绍了AutoTrust，这是一个针对自动驾驶中大型视觉语言模型(DriveVLMS)的综合可信度基准，考虑了不同的角度--包括可信性、安全性、健壮性、隐私和公平性。我们构建了最大的可视化问答数据集，用于调查驾驶场景中的可信度问题，包括超过10k个独特的场景和18k个查询。我们评估了六个公开可用的VLM，从通才到专家，从开源到商业模型。我们的详尽评估揭示了DriveVLM对可信度威胁之前未发现的漏洞。具体地说，我们发现像LLaVA-v1.6和GPT-40-mini这样的普通VLM在总体可信度方面出人意料地超过了专门为驾驶而调整的车型。像DriveLM-Agent这样的DriveVLM特别容易泄露敏感信息。此外，通才和专业的VLM仍然容易受到对抗性攻击，并努力确保在不同的环境和人群中做出公正的决策。我们的调查结果要求立即采取果断行动，解决DriveVLMS的可信性问题--这是一个对公共安全和依赖自动交通系统的所有公民的福利至关重要的问题。我们的基准在\url{https://github.com/taco-group/AutoTrust}，上公开可用，排行榜在\url{https://taco-group.github.io/AutoTrust/}.上发布



## **29. Do Parameters Reveal More than Loss for Membership Inference?**

参数揭示的不仅仅是会员推断的损失吗？ cs.LG

Accepted to Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2406.11544v4) [paper-pdf](http://arxiv.org/pdf/2406.11544v4)

**Authors**: Anshuman Suri, Xiao Zhang, David Evans

**Abstract**: Membership inference attacks are used as a key tool for disclosure auditing. They aim to infer whether an individual record was used to train a model. While such evaluations are useful to demonstrate risk, they are computationally expensive and often make strong assumptions about potential adversaries' access to models and training environments, and thus do not provide tight bounds on leakage from potential attacks. We show how prior claims around black-box access being sufficient for optimal membership inference do not hold for stochastic gradient descent, and that optimal membership inference indeed requires white-box access. Our theoretical results lead to a new white-box inference attack, IHA (Inverse Hessian Attack), that explicitly uses model parameters by taking advantage of computing inverse-Hessian vector products. Our results show that both auditors and adversaries may be able to benefit from access to model parameters, and we advocate for further research into white-box methods for membership inference.

摘要: 成员关系推断攻击被用作信息披露审计的关键工具。他们的目的是推断个人记录是否被用来训练模型。虽然这样的评估有助于显示风险，但它们的计算成本很高，而且通常会对潜在对手访问模型和训练环境做出强有力的假设，因此不会对潜在攻击的泄漏提供严格的限制。我们证明了关于黑盒访问的关于最优成员关系推理的先前声明如何不适用于随机梯度下降，而最优成员关系推理确实需要白盒访问。我们的理论结果导致了一种新的白盒推理攻击，IHA(逆向Hessian攻击)，它通过计算逆向Hessian向量积来显式地使用模型参数。我们的结果表明，审计师和对手都可能从访问模型参数中受益，我们主张进一步研究成员关系推理的白盒方法。



## **30. Accuracy Limits as a Barrier to Biometric System Security**

准确性限制是生物识别系统安全性的障碍 cs.CR

14 pages, 4 figures

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.13099v2) [paper-pdf](http://arxiv.org/pdf/2412.13099v2)

**Authors**: Axel Durbet, Paul-Marie Grollemund, Pascal Lafourcade, Kevin Thiry-Atighehchi

**Abstract**: Biometric systems are widely used for identity verification and identification, including authentication (i.e., one-to-one matching to verify a claimed identity) and identification (i.e., one-to-many matching to find a subject in a database). The matching process relies on measuring similarities or dissimilarities between a fresh biometric template and enrolled templates. The False Match Rate FMR is a key metric for assessing the accuracy and reliability of such systems. This paper analyzes biometric systems based on their FMR, with two main contributions. First, we explore untargeted attacks, where an adversary aims to impersonate any user within a database. We determine the number of trials required for an attacker to successfully impersonate a user and derive the critical population size (i.e., the maximum number of users in the database) required to maintain a given level of security. Furthermore, we compute the critical FMR value needed to ensure resistance against untargeted attacks as the database size increases. Second, we revisit the biometric birthday problem to evaluate the approximate and exact probabilities that two users in a database collide (i.e., can impersonate each other). Based on this analysis, we derive both the approximate critical population size and the critical FMR value needed to bound the likelihood of such collisions occurring with a given probability. These thresholds offer insights for designing systems that mitigate the risk of impersonation and collisions, particularly in large-scale biometric databases. Our findings indicate that current biometric systems fail to deliver sufficient accuracy to achieve an adequate security level against untargeted attacks, even in small-scale databases. Moreover, state-of-the-art systems face significant challenges in addressing the biometric birthday problem, especially as database sizes grow.

摘要: 生物测定系统广泛用于身份验证和身份识别，包括身份验证(即，一对一匹配以验证所声称的身份)和身份(即，一对多匹配以在数据库中找到对象)。匹配过程依赖于测量新的生物特征模板和注册模板之间的相似性或差异性。误匹配率FMR是评估这类系统的准确性和可靠性的关键指标。本文分析了基于FMR的生物特征识别系统，主要有两个贡献。首先，我们探索非目标攻击，其中对手的目标是模拟数据库中的任何用户。我们确定攻击者成功模拟用户所需的试验次数，并得出维持给定安全级别所需的临界总体大小(即，数据库中的最大用户数)。此外，随着数据库大小的增加，我们还计算了确保抵抗非目标攻击所需的关键FMR值。其次，我们回顾了生物统计生日问题，以评估数据库中两个用户发生冲突(即可以相互冒充)的近似和准确概率。基于这一分析，我们推导出了以给定概率限制这种碰撞发生的可能性所需的近似临界种群大小和临界FMR值。这些阈值为设计降低模仿和冲突风险的系统提供了见解，特别是在大规模生物识别数据库中。我们的发现表明，即使在小规模的数据库中，当前的生物识别系统也无法提供足够的准确性来实现足够的安全级别来抵御非目标攻击。此外，最先进的系统在解决生物识别生日问题方面面临着重大挑战，特别是随着数据库大小的增长。



## **31. SLIFER: Investigating Performance and Robustness of Malware Detection Pipelines**

SIFER：调查恶意软件检测管道的性能和稳健性 cs.CR

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2405.14478v3) [paper-pdf](http://arxiv.org/pdf/2405.14478v3)

**Authors**: Andrea Ponte, Dmitrijs Trizna, Luca Demetrio, Battista Biggio, Ivan Tesfai Ogbu, Fabio Roli

**Abstract**: As a result of decades of research, Windows malware detection is approached through a plethora of techniques. However, there is an ongoing mismatch between academia -- which pursues an optimal performances in terms of detection rate and low false alarms -- and the requirements of real-world scenarios. In particular, academia focuses on combining static and dynamic analysis within a single or ensemble of models, falling into several pitfalls like (i) firing dynamic analysis without considering the computational burden it requires; (ii) discarding impossible-to-analyze samples; and (iii) analyzing robustness against adversarial attacks without considering that malware detectors are complemented with more non-machine-learning components. Thus, in this paper we bridge these gaps, by investigating the properties of malware detectors built with multiple and different types of analysis. To do so, we develop SLIFER, a Windows malware detection pipeline sequentially leveraging both static and dynamic analysis, interrupting computations as soon as one module triggers an alarm, requiring dynamic analysis only when needed. Contrary to the state of the art, we investigate how to deal with samples that impede analyzes, showing how much they impact performances, concluding that it is better to flag them as legitimate to not drastically increase false alarms. Lastly, we perform a robustness evaluation of SLIFER. Counter-intuitively, the injection of new content is either blocked more by signatures than dynamic analysis, due to byte artifacts created by the attack, or it is able to avoid detection from signatures, as they rely on constraints on file size disrupted by attacks. As far as we know, we are the first to investigate the properties of sequential malware detectors, shedding light on their behavior in real production environment.

摘要: 作为数十年研究的结果，Windows恶意软件检测是通过大量技术实现的。然而，学术界--追求在检测率和低虚警方面的最佳表现--与现实世界场景的要求之间存在着持续的不匹配。特别是，学术界专注于在单个或集成模型中结合静态和动态分析，陷入了几个陷阱，如(I)触发动态分析而不考虑其所需的计算负担；(Ii)丢弃无法分析的样本；以及(Iii)分析针对对手攻击的稳健性，而不考虑恶意软件检测器与更多非机器学习组件的补充。因此，在本文中，我们通过调查使用多种不同类型的分析构建的恶意软件检测器的属性来弥合这些差距。为此，我们开发了Slifer，这是一条Windows恶意软件检测管道，顺序地利用静态和动态分析，在一个模块触发警报时立即中断计算，仅在需要时才需要动态分析。与现有技术相反，我们调查了如何处理阻碍分析的样本，显示了它们对性能的影响程度，得出的结论是，最好将它们标记为合法，而不是大幅增加错误警报。最后，我们对Slifer算法进行了健壮性评估。与直觉相反的是，由于攻击产生的字节伪像，新内容的注入更多地被签名阻止，而不是动态分析，或者它能够避免从签名检测，因为它们依赖于被攻击破坏的文件大小限制。据我们所知，我们是第一个研究顺序恶意软件检测器的性质的人，揭示了它们在实际生产环境中的行为。



## **32. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

卡西姆·分歧：通过对抗性防御的扩散模型来启发性解纠缠 cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2410.23091v5) [paper-pdf](http://arxiv.org/pdf/2410.23091v5)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff

摘要: 尽管不断努力保护神经分类器免受对手攻击，但它们仍然很脆弱，特别是面对看不见的攻击。相比之下，人类很难被微妙的操纵所欺骗，因为我们只根据基本因素做出判断。受到这一观察的启发，我们试图用基本的标签原因因素来建模标签生成，并结合标签非原因因素来辅助数据生成。对于一个对抗性的例子，我们的目标是将扰动区分为非致因因素，并仅基于标签致因因素进行预测。具体地说，我们提出了一个偶然扩散模型(CausalDiff)，该模型使扩散模型适用于条件数据生成，并通过向一个新的偶然信息瓶颈目标学习来区分这两种类型的偶然因素。经验上，CausalDiff在各种隐形攻击上的表现明显优于最先进的防御方法，在CIFAR-10上获得了86.39%(+4.01%)的平均健壮性，在CIFAR-100上获得了56.25%(+3.13%)的健壮性，在GTSRB(德国交通标志识别基准)上实现了82.62%(+4.93%)的平均健壮性。代码可在https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff上获得



## **33. Grimm: A Plug-and-Play Perturbation Rectifier for Graph Neural Networks Defending against Poisoning Attacks**

Grimm：用于图神经网络防御中毒攻击的即插即用微扰矫正器 cs.LG

19 pages, 13 figures

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08555v2) [paper-pdf](http://arxiv.org/pdf/2412.08555v2)

**Authors**: Ao Liu, Wenshan Li, Beibei Li, Wengang Ma, Tao Li, Pan Zhou

**Abstract**: Recent studies have revealed the vulnerability of graph neural networks (GNNs) to adversarial poisoning attacks on node classification tasks. Current defensive methods require substituting the original GNNs with defense models, regardless of the original's type. This approach, while targeting adversarial robustness, compromises the enhancements developed in prior research to boost GNNs' practical performance. Here we introduce Grimm, the first plug-and-play defense model. With just a minimal interface requirement for extracting features from any layer of the protected GNNs, Grimm is thus enabled to seamlessly rectify perturbations. Specifically, we utilize the feature trajectories (FTs) generated by GNNs, as they evolve through epochs, to reflect the training status of the networks. We then theoretically prove that the FTs of victim nodes will inevitably exhibit discriminable anomalies. Consequently, inspired by the natural parallelism between the biological nervous and immune systems, we construct Grimm, a comprehensive artificial immune system for GNNs. Grimm not only detects abnormal FTs and rectifies adversarial edges during training but also operates efficiently in parallel, thereby mirroring the concurrent functionalities of its biological counterparts. We experimentally confirm that Grimm offers four empirically validated advantages: 1) Harmlessness, as it does not actively interfere with GNN training; 2) Parallelism, ensuring monitoring, detection, and rectification functions operate independently of the GNN training process; 3) Generalizability, demonstrating compatibility with mainstream GNNs such as GCN, GAT, and GraphSAGE; and 4) Transferability, as the detectors for abnormal FTs can be efficiently transferred across different systems for one-step rectification.

摘要: 最近的研究揭示了图神经网络(GNN)对节点分类任务的敌意中毒攻击的脆弱性。目前的防御方法需要用防御模型取代原始GNN，而不考虑原始GNN的类型。虽然这种方法的目标是对抗的稳健性，但它损害了先前研究中为提高GNN的实际性能而开发的增强。在这里，我们介绍格林，第一个即插即用的防御模型。从受保护的GNN的任何一层中提取特征只需要最小的接口要求，因此GRIMM就能够无缝地纠正扰动。具体地说，我们利用GNN生成的特征轨迹(FTs)来反映网络的训练状态。然后，我们从理论上证明了受害节点的FT将不可避免地表现出可区分的异常。因此，受生物神经系统和免疫系统的自然并行性的启发，我们构建了一个用于GNN的全面的人工免疫系统GRIMM。GRIMM不仅在训练过程中检测异常FT并纠正敌对边缘，而且还可以高效地并行操作，从而反映出其生物同行的并发功能。我们通过实验证实GRIMM提供了四个经验证的优势：1)无害，因为它不会主动干扰GNN训练；2)并行性，确保监测、检测和纠正功能独立于GNN训练过程运行；3)通用性，表现出与GCN、GAT和GraphSAGE等主流GNN的兼容性；以及4)可转移性，因为异常FT的检测器可以在不同的系统中高效传输，以便一步纠正。



## **34. DG-Mamba: Robust and Efficient Dynamic Graph Structure Learning with Selective State Space Models**

DG-Mamba：使用选择性状态空间模型稳健高效的动态图结构学习 cs.LG

Accepted by the Main Technical Track of the 39th Annual AAAI  Conference on Artificial Intelligence (AAAI-2025)

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08160v4) [paper-pdf](http://arxiv.org/pdf/2412.08160v4)

**Authors**: Haonan Yuan, Qingyun Sun, Zhaonan Wang, Xingcheng Fu, Cheng Ji, Yongjian Wang, Bo Jin, Jianxin Li

**Abstract**: Dynamic graphs exhibit intertwined spatio-temporal evolutionary patterns, widely existing in the real world. Nevertheless, the structure incompleteness, noise, and redundancy result in poor robustness for Dynamic Graph Neural Networks (DGNNs). Dynamic Graph Structure Learning (DGSL) offers a promising way to optimize graph structures. However, aside from encountering unacceptable quadratic complexity, it overly relies on heuristic priors, making it hard to discover underlying predictive patterns. How to efficiently refine the dynamic structures, capture intrinsic dependencies, and learn robust representations, remains under-explored. In this work, we propose the novel DG-Mamba, a robust and efficient Dynamic Graph structure learning framework with the Selective State Space Models (Mamba). To accelerate the spatio-temporal structure learning, we propose a kernelized dynamic message-passing operator that reduces the quadratic time complexity to linear. To capture global intrinsic dynamics, we establish the dynamic graph as a self-contained system with State Space Model. By discretizing the system states with the cross-snapshot graph adjacency, we enable the long-distance dependencies capturing with the selective snapshot scan. To endow learned dynamic structures more expressive with informativeness, we propose the self-supervised Principle of Relevant Information for DGSL to regularize the most relevant yet least redundant information, enhancing global robustness. Extensive experiments demonstrate the superiority of the robustness and efficiency of our DG-Mamba compared with the state-of-the-art baselines against adversarial attacks.

摘要: 动态图形表现出交织在一起的时空演化模式，广泛存在于现实世界中。然而，动态图神经网络的结构不完备性、噪声和冗余性导致其健壮性较差。动态图结构学习(DGSL)为优化图结构提供了一种很有前途的方法。然而，除了遇到不可接受的二次型复杂性外，它还过度依赖启发式先验，使得发现潜在的预测模式变得困难。如何有效地提炼动态结构，捕获内在依赖关系，并学习健壮的表示，仍未得到探索。在这项工作中，我们提出了一种新颖的DG-MAMBA，这是一种基于选择状态空间模型(MAMBA)的健壮而高效的动态图结构学习框架。为了加速时空结构的学习，我们提出了一种核化的动态消息传递算子，将二次时间复杂度降为线性。为了捕捉全局内在动力学，我们利用状态空间模型将动态图建立为一个自包含系统。通过使用交叉快照图邻接关系对系统状态进行离散化，实现了选择性快照扫描的远程依赖捕获。为了使学习到的动态结构具有更强的信息性，我们提出了DGSL的相关信息自监督原则，将相关程度最高但冗余最少的信息正则化，增强了全局鲁棒性。大量的实验证明了DG-MAMBA算法的健壮性和高效性，与目前最先进的对抗攻击基线算法相比具有更好的性能。



## **35. How Does the Smoothness Approximation Method Facilitate Generalization for Federated Adversarial Learning?**

光滑度逼近方法如何促进联邦对抗学习的推广？ cs.LG

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08282v2) [paper-pdf](http://arxiv.org/pdf/2412.08282v2)

**Authors**: Wenjun Ding, Ying An, Lixing Chen, Shichao Kan, Fan Wu, Zhe Qu

**Abstract**: Federated Adversarial Learning (FAL) is a robust framework for resisting adversarial attacks on federated learning. Although some FAL studies have developed efficient algorithms, they primarily focus on convergence performance and overlook generalization. Generalization is crucial for evaluating algorithm performance on unseen data. However, generalization analysis is more challenging due to non-smooth adversarial loss functions. A common approach to addressing this issue is to leverage smoothness approximation. In this paper, we develop algorithm stability measures to evaluate the generalization performance of two popular FAL algorithms: \textit{Vanilla FAL (VFAL)} and {\it Slack FAL (SFAL)}, using three different smooth approximation methods: 1) \textit{Surrogate Smoothness Approximation (SSA)}, (2) \textit{Randomized Smoothness Approximation (RSA)}, and (3) \textit{Over-Parameterized Smoothness Approximation (OPSA)}. Based on our in-depth analysis, we answer the question of how to properly set the smoothness approximation method to mitigate generalization error in FAL. Moreover, we identify RSA as the most effective method for reducing generalization error. In highly data-heterogeneous scenarios, we also recommend employing SFAL to mitigate the deterioration of generalization performance caused by heterogeneity. Based on our theoretical results, we provide insights to help develop more efficient FAL algorithms, such as designing new metrics and dynamic aggregation rules to mitigate heterogeneity.

摘要: 联合对抗学习(FAL)是一种用于抵抗对联合学习的敌意攻击的健壮框架。虽然一些FAL研究已经开发出了有效的算法，但它们主要集中在收敛性能上，而忽略了泛化。泛化是在未知数据上评估算法性能的关键。然而，由于对抗性损失函数的非光滑性，泛化分析具有更大的挑战性。解决此问题的一种常见方法是利用平滑近似。本文利用3种不同的光滑逼近方法：1)替代平滑逼近(SSA)}、(2)随机平滑逼近(RSA)}和(3)过参数平滑逼近(OPSA)}，对两种常见的FAL算法在深入分析的基础上，我们回答了如何合理地设置光滑度逼近方法来减小FAL中的泛化误差的问题。此外，我们认为RSA是减少泛化误差的最有效方法。在数据高度异构性的场景中，我们还建议使用SFAL来缓解异构性导致的泛化性能下降。基于我们的理论结果，我们提供了一些见解来帮助开发更高效的FAL算法，例如设计新的度量和动态聚集规则来缓解异构性。



## **36. Unleashing the Unseen: Harnessing Benign Datasets for Jailbreaking Large Language Models**

释放隐形：利用良性数据集破解大型语言模型 cs.CR

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2410.00451v3) [paper-pdf](http://arxiv.org/pdf/2410.00451v3)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Despite significant ongoing efforts in safety alignment, large language models (LLMs) such as GPT-4 and LLaMA 3 remain vulnerable to jailbreak attacks that can induce harmful behaviors, including through the use of adversarial suffixes. Building on prior research, we hypothesize that these adversarial suffixes are not mere bugs but may represent features that can dominate the LLM's behavior. To evaluate this hypothesis, we conduct several experiments. First, we demonstrate that benign features can be effectively made to function as adversarial suffixes, i.e., we develop a feature extraction method to extract sample-agnostic features from benign dataset in the form of suffixes and show that these suffixes may effectively compromise safety alignment. Second, we show that adversarial suffixes generated from jailbreak attacks may contain meaningful features, i.e., appending the same suffix to different prompts results in responses exhibiting specific characteristics. Third, we show that such benign-yet-safety-compromising features can be easily introduced through fine-tuning using only benign datasets. As a result, we are able to completely eliminate GPT's safety alignment in a blackbox setting through finetuning with only benign data. Our code and data is available at \url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.

摘要: 尽管在安全匹配方面正在进行重大努力，但GPT-4和大羊驼3等大型语言模型仍然容易受到越狱攻击，这些攻击可能会导致有害行为，包括通过使用对抗性后缀。在先前研究的基础上，我们假设这些对抗性后缀不仅仅是错误，而且可能代表可以主导LLM行为的特征。为了评估这一假设，我们进行了几个实验。首先，我们证明了良性特征可以有效地用作对抗性后缀，即，我们开发了一种特征提取方法来从良性数据集中提取与样本无关的后缀形式的特征，并表明这些后缀可以有效地危害安全对齐。其次，我们证明了越狱攻击产生的对抗性后缀可能包含有意义的特征，即在不同的提示后添加相同的后缀会导致响应表现出特定的特征。第三，我们表明，仅使用良性数据集，通过微调就可以很容易地引入这种良性但危及安全的特征。因此，我们能够通过仅使用良性数据进行微调，在黑盒设置中完全消除GPT的安全对齐。我们的代码和数据可在\url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.上获得



## **37. Doubly-Universal Adversarial Perturbations: Deceiving Vision-Language Models Across Both Images and Text with a Single Perturbation**

双重普遍对抗性扰动：通过单一扰动欺骗图像和文本的视觉语言模型 cs.CV

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08108v2) [paper-pdf](http://arxiv.org/pdf/2412.08108v2)

**Authors**: Hee-Seon Kim, Minbeom Kim, Changick Kim

**Abstract**: Large Vision-Language Models (VLMs) have demonstrated remarkable performance across multimodal tasks by integrating vision encoders with large language models (LLMs). However, these models remain vulnerable to adversarial attacks. Among such attacks, Universal Adversarial Perturbations (UAPs) are especially powerful, as a single optimized perturbation can mislead the model across various input images. In this work, we introduce a novel UAP specifically designed for VLMs: the Doubly-Universal Adversarial Perturbation (Doubly-UAP), capable of universally deceiving VLMs across both image and text inputs. To successfully disrupt the vision encoder's fundamental process, we analyze the core components of the attention mechanism. After identifying value vectors in the middle-to-late layers as the most vulnerable, we optimize Doubly-UAP in a label-free manner with a frozen model. Despite being developed as a black-box to the LLM, Doubly-UAP achieves high attack success rates on VLMs, consistently outperforming baseline methods across vision-language tasks. Extensive ablation studies and analyses further demonstrate the robustness of Doubly-UAP and provide insights into how it influences internal attention mechanisms.

摘要: 大视觉语言模型(VLM)通过将视觉编码器与大语言模型(LLM)相结合，在多通道任务中表现出了显著的性能。然而，这些模型仍然容易受到对手的攻击。在这些攻击中，通用对抗性扰动(UAP)尤其强大，因为单个优化的扰动可以在不同的输入图像上误导模型。在这项工作中，我们介绍了一种新的专门针对VLMS设计的UAP：双重通用对抗性摄动(Double-Universal Aversarial微扰，Double-UAP)，能够在图像和文本输入之间普遍欺骗VLMS。为了成功地扰乱视觉编码器的基本过程，我们分析了注意机制的核心组件。在确定中后期价值向量最易受攻击后，我们使用冻结模型以无标签的方式对Double-UAP进行优化。尽管被开发为LLM的黑匣子，Double-UAP在VLM上实现了高攻击成功率，在视觉语言任务中始终优于基线方法。广泛的消融研究和分析进一步证明了Double-UAP的健壮性，并提供了对其如何影响内部注意机制的见解。



## **38. Towards Provable Security in Industrial Control Systems Via Dynamic Protocol Attestation**

通过动态协议认证实现工业控制系统的可证明安全性 cs.CR

This paper was accepted into the ICSS'24 workshop

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.14467v1) [paper-pdf](http://arxiv.org/pdf/2412.14467v1)

**Authors**: Arthur Amorim, Trevor Kann, Max Taylor, Lance Joneckis

**Abstract**: Industrial control systems (ICSs) increasingly rely on digital technologies vulnerable to cyber attacks. Cyber attackers can infiltrate ICSs and execute malicious actions. Individually, each action seems innocuous. But taken together, they cause the system to enter an unsafe state. These attacks have resulted in dramatic consequences such as physical damage, economic loss, and environmental catastrophes. This paper introduces a methodology that restricts actions using protocols. These protocols only allow safe actions to execute. Protocols are written in a domain specific language we have embedded in an interactive theorem prover (ITP). The ITP enables formal, machine-checked proofs to ensure protocols maintain safety properties. We use dynamic attestation to ensure ICSs conform to their protocol even if an adversary compromises a component. Since protocol conformance prevents unsafe actions, the previously mentioned cyber attacks become impossible. We demonstrate the effectiveness of our methodology using an example from the Fischertechnik Industry 4.0 platform. We measure dynamic attestation's impact on latency and throughput. Our approach is a starting point for studying how to combine formal methods and protocol design to thwart attacks intended to cripple ICSs.

摘要: 工业控制系统(ICSS)越来越依赖易受网络攻击的数字技术。网络攻击者可以渗透到ICSS中并执行恶意操作。单独来看，每一项行动似乎都是无害的。但综合起来，它们会导致系统进入不安全状态。这些袭击造成了严重的后果，如物质损失、经济损失和环境灾难。本文介绍了一种使用协议限制操作的方法。这些协议只允许执行安全操作。协议是用我们嵌入到交互式定理证明器(ITP)中的特定于领域的语言编写的。ITP允许正式的机器检查的证明，以确保协议维护安全属性。我们使用动态证明来确保ICSS符合他们的协议，即使对手破坏了组件。由于协议一致性可以防止不安全的行为，因此前面提到的网络攻击变得不可能。我们以Fischer Technik Industry 4.0平台为例，验证了该方法的有效性。我们测量了动态证明对延迟和吞吐量的影响。我们的方法是研究如何结合形式化方法和协议设计来挫败旨在削弱ICSS的攻击的一个起点。



## **39. Adversarial Hubness in Multi-Modal Retrieval**

多模式检索中的对抗性积极性 cs.CR

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.14113v1) [paper-pdf](http://arxiv.org/pdf/2412.14113v1)

**Authors**: Tingwei Zhang, Fnu Suya, Rishi Jha, Collin Zhang, Vitaly Shmatikov

**Abstract**: Hubness is a phenomenon in high-dimensional vector spaces where a single point from the natural distribution is unusually close to many other points. This is a well-known problem in information retrieval that causes some items to accidentally (and incorrectly) appear relevant to many queries. In this paper, we investigate how attackers can exploit hubness to turn any image or audio input in a multi-modal retrieval system into an adversarial hub. Adversarial hubs can be used to inject universal adversarial content (e.g., spam) that will be retrieved in response to thousands of different queries, as well as for targeted attacks on queries related to specific, attacker-chosen concepts. We present a method for creating adversarial hubs and evaluate the resulting hubs on benchmark multi-modal retrieval datasets and an image-to-image retrieval system based on a tutorial from Pinecone, a popular vector database. For example, in text-caption-to-image retrieval, a single adversarial hub is retrieved as the top-1 most relevant image for more than 21,000 out of 25,000 test queries (by contrast, the most common natural hub is the top-1 response to only 102 queries). We also investigate whether techniques for mitigating natural hubness are an effective defense against adversarial hubs, and show that they are not effective against hubs that target queries related to specific concepts.

摘要: Hubness是高维向量空间中的一种现象，在高维向量空间中，自然分布中的单个点异常接近许多其他点。这是信息检索中的一个众所周知的问题，它会导致某些项意外地(并且不正确地)显示为与许多查询相关。在本文中，我们研究攻击者如何利用Hubness将多模式检索系统中的任何图像或音频输入转换为敌对中心。对抗性集线器可用于注入将响应于数千个不同查询而检索的通用对抗性内容(例如，垃圾邮件)，以及用于针对与特定的攻击者选择的概念相关的查询的定向攻击。我们提出了一种创建敌意中心的方法，并在基准多模式检索数据集和基于流行的矢量数据库Pinecone的教程的图像到图像检索系统上对生成的中心进行了评估。例如，在文本字幕到图像的检索中，对于25,000个测试查询中的超过21,000个，单个敌意中心被检索为最相关的前1个图像(相比之下，最常见的自然中心是仅对102个查询的前1个响应)。我们还调查了缓解自然中心的技术是否是对抗中心的有效防御，并表明它们对针对与特定概念相关的查询的中心无效。



## **40. Certification of Speaker Recognition Models to Additive Perturbations**

说话人识别模型对加性扰动的认证 cs.SD

13 pages, 10 figures; AAAI-2025 accepted paper

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2404.18791v2) [paper-pdf](http://arxiv.org/pdf/2404.18791v2)

**Authors**: Dmitrii Korzh, Elvir Karimov, Mikhail Pautov, Oleg Y. Rogov, Ivan Oseledets

**Abstract**: Speaker recognition technology is applied to various tasks, from personal virtual assistants to secure access systems. However, the robustness of these systems against adversarial attacks, particularly to additive perturbations, remains a significant challenge. In this paper, we pioneer applying robustness certification techniques to speaker recognition, initially developed for the image domain. Our work covers this gap by transferring and improving randomized smoothing certification techniques against norm-bounded additive perturbations for classification and few-shot learning tasks to speaker recognition. We demonstrate the effectiveness of these methods on VoxCeleb 1 and 2 datasets for several models. We expect this work to improve the robustness of voice biometrics and accelerate the research of certification methods in the audio domain.

摘要: 说话人识别技术应用于各种任务，从个人虚拟助理到安全访问系统。然而，这些系统对对抗攻击（特别是对添加性扰动）的鲁棒性仍然是一个重大挑战。在本文中，我们率先将鲁棒性认证技术应用于说话人识别，该技术最初是针对图像领域开发的。我们的工作通过转移和改进随机平滑认证技术来弥补这一差距，以对抗分类和少数镜头学习任务的规范界添加性扰动。我们在VoxCeleb 1和2数据集上证明了这些方法对于多个模型的有效性。我们希望这项工作能够提高语音生物识别技术的稳健性，并加速音频领域认证方法的研究。



## **41. Adversarial Robustness of Link Sign Prediction in Signed Graphs**

带符号图中链接符号预测的对抗鲁棒性 cs.LG

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2401.10590v2) [paper-pdf](http://arxiv.org/pdf/2401.10590v2)

**Authors**: Jialong Zhou, Xing Ai, Yuni Lai, Tomasz Michalak, Gaolei Li, Jianhua Li, Kai Zhou

**Abstract**: Signed graphs serve as fundamental data structures for representing positive and negative relationships in social networks, with signed graph neural networks (SGNNs) emerging as the primary tool for their analysis. Our investigation reveals that balance theory, while essential for modeling signed relationships in SGNNs, inadvertently introduces exploitable vulnerabilities to black-box attacks. To demonstrate this vulnerability, we propose balance-attack, a novel adversarial strategy specifically designed to compromise graph balance degree, and develop an efficient heuristic algorithm to solve the associated NP-hard optimization problem. While existing approaches attempt to restore attacked graphs through balance learning techniques, they face a critical challenge we term "Irreversibility of Balance-related Information," where restored edges fail to align with original attack targets. To address this limitation, we introduce Balance Augmented-Signed Graph Contrastive Learning (BA-SGCL), an innovative framework that combines contrastive learning with balance augmentation techniques to achieve robust graph representations. By maintaining high balance degree in the latent space, BA-SGCL effectively circumvents the irreversibility challenge and enhances model resilience. Extensive experiments across multiple SGNN architectures and real-world datasets demonstrate both the effectiveness of our proposed balance-attack and the superior robustness of BA-SGCL, advancing the security and reliability of signed graph analysis in social networks. Datasets and codes of the proposed framework are at the github repository https://anonymous.4open.science/r/BA-SGCL-submit-DF41/.

摘要: 带符号图是表示社会网络中正负关系的基本数据结构，带符号图神经网络(SGNN)是分析这些关系的主要工具。我们的研究表明，平衡理论虽然对SGNN中的签名关系建模是必不可少的，但它无意中为黑盒攻击引入了可利用的漏洞。为了证明这一漏洞，我们提出了一种新的对抗性策略--平衡攻击，它是专门为折衷图平衡度而设计的，并开发了一个有效的启发式算法来解决相关的NP-Hard优化问题。虽然现有的方法试图通过平衡学习技术来恢复被攻击的图形，但它们面临着一个关键的挑战，我们将其称为“与平衡相关的信息的不可逆性”，其中恢复的边无法与原始攻击目标对齐。为了解决这一局限性，我们引入了平衡增强符号图对比学习(BA-SGCL)，这是一个结合对比学习和平衡增强技术的创新框架，以实现健壮的图表示。BA-SGCL通过在潜在空间保持较高的平衡度，有效地规避了模型的不可逆性挑战，增强了模型的弹性。在多个SGNN体系结构和真实数据集上的大量实验证明了我们提出的平衡攻击的有效性和BA-SGCL的卓越健壮性，从而提高了社交网络中签名图分析的安全性和可靠性。拟议框架的数据集和代码位于GitHub存储库https://anonymous.4open.science/r/BA-SGCL-submit-DF41/.



## **42. A Review of the Duality of Adversarial Learning in Network Intrusion: Attacks and Countermeasures**

网络入侵中对抗学习的二重性：攻击与对策 cs.CR

23 pages, 2 figures, 5 tables

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13880v1) [paper-pdf](http://arxiv.org/pdf/2412.13880v1)

**Authors**: Shalini Saini, Anitha Chennamaneni, Babatunde Sawyerr

**Abstract**: Deep learning solutions are instrumental in cybersecurity, harnessing their ability to analyze vast datasets, identify complex patterns, and detect anomalies. However, malevolent actors can exploit these capabilities to orchestrate sophisticated attacks, posing significant challenges to defenders and traditional security measures. Adversarial attacks, particularly those targeting vulnerabilities in deep learning models, present a nuanced and substantial threat to cybersecurity. Our study delves into adversarial learning threats such as Data Poisoning, Test Time Evasion, and Reverse Engineering, specifically impacting Network Intrusion Detection Systems. Our research explores the intricacies and countermeasures of attacks to deepen understanding of network security challenges amidst adversarial threats. In our study, we present insights into the dynamic realm of adversarial learning and its implications for network intrusion. The intersection of adversarial attacks and defenses within network traffic data, coupled with advances in machine learning and deep learning techniques, represents a relatively underexplored domain. Our research lays the groundwork for strengthening defense mechanisms to address the potential breaches in network security and privacy posed by adversarial attacks. Through our in-depth analysis, we identify domain-specific research gaps, such as the scarcity of real-life attack data and the evaluation of AI-based solutions for network traffic. Our focus on these challenges aims to stimulate future research efforts toward the development of resilient network defense strategies.

摘要: 深度学习解决方案在网络安全方面非常重要，可以利用它们分析海量数据集、识别复杂模式和检测异常的能力。然而，恶意行为者可以利用这些能力来策划复杂的攻击，给防御者和传统安全措施带来重大挑战。对抗性攻击，特别是针对深度学习模型中的漏洞的攻击，对网络安全构成了微妙而实质性的威胁。我们的研究深入到对抗性学习威胁，如数据中毒、测试时间逃避和逆向工程，特别是影响网络入侵检测系统。我们的研究探索了攻击的复杂性和对策，以加深对对手威胁中的网络安全挑战的理解。在我们的研究中，我们提出了对对抗学习的动态领域及其对网络入侵的影响的见解。网络流量数据中的对抗性攻击和防御的交集，再加上机器学习和深度学习技术的进步，代表着一个相对未被探索的领域。我们的研究为加强防御机制以应对对抗性攻击在网络安全和隐私方面的潜在破坏奠定了基础。通过我们的深入分析，我们找出了特定领域的研究差距，例如真实攻击数据的稀缺和基于人工智能的网络流量解决方案的评估。我们对这些挑战的关注旨在刺激未来开发弹性网络防御战略的研究努力。



## **43. Formal Verification of Permission Voucher**

许可证的正式验证 cs.CR

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.16224v1) [paper-pdf](http://arxiv.org/pdf/2412.16224v1)

**Authors**: Khan Reaz, Gerhard Wunder

**Abstract**: Formal verification is a critical process in ensuring the security and correctness of cryptographic protocols, particularly in high-assurance domains. This paper presents a comprehensive formal analysis of the Permission Voucher Protocol, a system designed for secure and authenticated access control in distributed environments. The analysis employs the Tamarin Prover, a state-of-the-art tool for symbolic verification, to evaluate key security properties such as authentication, confidentiality, integrity, mutual authentication, and replay prevention. We model the protocol's components, including trust relationships, secure channels, and adversary capabilities under the Dolev-Yao model. Verification results confirm the protocol's robustness against common attacks such as message tampering, impersonation, and replay. Additionally, dependency graphs and detailed proofs demonstrate the successful enforcement of security properties like voucher authenticity, data confidentiality, and key integrity. The study identifies potential enhancements, such as incorporating timestamp-based validity checks and augmenting mutual authentication mechanisms to address insider threats and key management challenges. This work highlights the advantages and limitations of using the Tamarin Prover for formal security verification and proposes strategies to mitigate scalability and performance constraints in complex systems.

摘要: 形式验证是确保密码协议的安全性和正确性的关键过程，特别是在高保证领域。本文对权限凭证协议进行了全面的形式化分析，该协议是为分布式环境中的安全和认证访问控制而设计的系统。该分析使用最先进的符号验证工具Tamarin Prover来评估关键安全属性，如身份验证、机密性、完整性、相互身份验证和重放防止。在多列夫-姚模型下，我们对协议的组件进行了建模，包括信任关系、安全通道和敌手能力。验证结果证实了该协议对消息篡改、冒充、重放等常见攻击的健壮性。此外，依赖关系图和详细的证明证明了凭证真实性、数据机密性和密钥完整性等安全属性的成功实施。该研究确定了潜在的增强功能，例如整合基于时间戳的有效性检查和增强相互身份验证机制，以应对内部威胁和密钥管理挑战。这项工作突出了使用Tamarin Prover进行正式安全验证的优势和局限性，并提出了缓解复杂系统中可伸缩性和性能约束的策略。



## **44. Cultivating Archipelago of Forests: Evolving Robust Decision Trees through Island Coevolution**

培育森林群岛：通过岛屿共同进化进化稳健决策树 cs.LG

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13762v1) [paper-pdf](http://arxiv.org/pdf/2412.13762v1)

**Authors**: Adam Żychowski, Andrew Perrault, Jacek Mańdziuk

**Abstract**: Decision trees are widely used in machine learning due to their simplicity and interpretability, but they often lack robustness to adversarial attacks and data perturbations. The paper proposes a novel island-based coevolutionary algorithm (ICoEvoRDF) for constructing robust decision tree ensembles. The algorithm operates on multiple islands, each containing populations of decision trees and adversarial perturbations. The populations on each island evolve independently, with periodic migration of top-performing decision trees between islands. This approach fosters diversity and enhances the exploration of the solution space, leading to more robust and accurate decision tree ensembles. ICoEvoRDF utilizes a popular game theory concept of mixed Nash equilibrium for ensemble weighting, which further leads to improvement in results. ICoEvoRDF is evaluated on 20 benchmark datasets, demonstrating its superior performance compared to state-of-the-art methods in optimizing both adversarial accuracy and minimax regret. The flexibility of ICoEvoRDF allows for the integration of decision trees from various existing methods, providing a unified framework for combining diverse solutions. Our approach offers a promising direction for developing robust and interpretable machine learning models

摘要: 决策树因其简单性和可解释性在机器学习中得到了广泛的应用，但它们对敌意攻击和数据扰动往往缺乏健壮性。提出了一种基于孤岛的协同进化算法(ICoEvoRDF)，用于构造稳健的决策树集成。该算法在多个孤岛上运行，每个孤岛包含决策树种群和对抗性扰动。每个岛屿上的种群独立进化，表现最好的决策树在岛屿之间定期迁移。这种方法促进了多样性并增强了对解空间的探索，导致了更健壮和更准确的决策树集成。ICoEvoRDF使用了一个流行的博弈论概念-混合纳什均衡来进行集成加权，这进一步导致了结果的改进。ICoEvoRDF在20个基准数据集上进行了评估，显示出与最先进的方法相比，它在优化对手准确性和最小最大遗憾方面的优越性能。ICoEvoRDF的灵活性允许集成来自各种现有方法的决策树，为组合不同的解决方案提供统一的框架。我们的方法为开发健壮和可解释的机器学习模型提供了一个很有前途的方向



## **45. A2RNet: Adversarial Attack Resilient Network for Robust Infrared and Visible Image Fusion**

A2 RNet：用于鲁棒的红外和可见光图像融合的对抗攻击弹性网络 cs.CV

9 pages, 8 figures, The 39th Annual AAAI Conference on Artificial  Intelligence

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.09954v2) [paper-pdf](http://arxiv.org/pdf/2412.09954v2)

**Authors**: Jiawei Li, Hongwei Yu, Jiansheng Chen, Xinlong Ding, Jinlong Wang, Jinyuan Liu, Bochao Zou, Huimin Ma

**Abstract**: Infrared and visible image fusion (IVIF) is a crucial technique for enhancing visual performance by integrating unique information from different modalities into one fused image. Exiting methods pay more attention to conducting fusion with undisturbed data, while overlooking the impact of deliberate interference on the effectiveness of fusion results. To investigate the robustness of fusion models, in this paper, we propose a novel adversarial attack resilient network, called $\textrm{A}^{\textrm{2}}$RNet. Specifically, we develop an adversarial paradigm with an anti-attack loss function to implement adversarial attacks and training. It is constructed based on the intrinsic nature of IVIF and provide a robust foundation for future research advancements. We adopt a Unet as the pipeline with a transformer-based defensive refinement module (DRM) under this paradigm, which guarantees fused image quality in a robust coarse-to-fine manner. Compared to previous works, our method mitigates the adverse effects of adversarial perturbations, consistently maintaining high-fidelity fusion results. Furthermore, the performance of downstream tasks can also be well maintained under adversarial attacks. Code is available at https://github.com/lok-18/A2RNet.

摘要: 红外与可见光图像融合(IVIF)是通过将来自不同模式的独特信息融合到一幅融合图像中来提高视觉性能的关键技术。现有的方法更注重对未受干扰的数据进行融合，而忽略了有意干扰对融合结果有效性的影响。为了研究融合模型的稳健性，本文提出了一种新的对抗攻击弹性网络，称为$\tExtm{A}^{\tExtm{2}}$rnet。具体地说，我们开发了一个具有抗攻击损失函数的对抗性范例来实施对抗性攻击和训练。它是基于IVIF的内在本质而构建的，并为未来的研究进展提供了坚实的基础。在该模型下，我们采用了基于变换的防御性细化模块(DRM)作为流水线，保证了从粗到精的融合图像质量。与以前的工作相比，我们的方法减轻了对抗性扰动的不利影响，一致地保持了高保真的融合结果。此外，在对抗性攻击下，下游任务的性能也能得到很好的维持。代码可在https://github.com/lok-18/A2RNet.上找到



## **46. Physics-Based Adversarial Attack on Near-Infrared Human Detector for Nighttime Surveillance Camera Systems**

针对夜间监控摄像机系统近红外人体探测器的基于物理的对抗攻击 cs.CV

Appeared in ACM MM 2023

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13709v1) [paper-pdf](http://arxiv.org/pdf/2412.13709v1)

**Authors**: Muyao Niu, Zhuoxiao Li, Yifan Zhan, Huy H. Nguyen, Isao Echizen, Yinqiang Zheng

**Abstract**: Many surveillance cameras switch between daytime and nighttime modes based on illuminance levels. During the day, the camera records ordinary RGB images through an enabled IR-cut filter. At night, the filter is disabled to capture near-infrared (NIR) light emitted from NIR LEDs typically mounted around the lens. While RGB-based AI algorithm vulnerabilities have been widely reported, the vulnerabilities of NIR-based AI have rarely been investigated. In this paper, we identify fundamental vulnerabilities in NIR-based image understanding caused by color and texture loss due to the intrinsic characteristics of clothes' reflectance and cameras' spectral sensitivity in the NIR range. We further show that the nearly co-located configuration of illuminants and cameras in existing surveillance systems facilitates concealing and fully passive attacks in the physical world. Specifically, we demonstrate how retro-reflective and insulation plastic tapes can manipulate the intensity distribution of NIR images. We showcase an attack on the YOLO-based human detector using binary patterns designed in the digital space (via black-box query and searching) and then physically realized using tapes pasted onto clothes. Our attack highlights significant reliability concerns for nighttime surveillance systems, which are intended to enhance security. Codes Available: https://github.com/MyNiuuu/AdvNIR

摘要: 许多监控摄像头根据照度级别在白天和夜间模式之间切换。白天，相机通过启用的IR-Cut滤镜记录普通RGB图像。在夜间，滤光片被禁用以捕获通常安装在镜头周围的近红外LED发出的近红外(NIR)光。虽然基于RGB的人工智能算法漏洞已经被广泛报道，但基于近红外的人工智能漏洞很少被调查。在本文中，我们找出了基于近红外图像理解的基本缺陷，这些缺陷是由于衣服的反射率和相机在近红外范围内的光谱敏感度的固有特性造成的颜色和纹理的损失。我们进一步表明，在现有的监控系统中，光源和摄像机几乎位于同一位置的配置有助于在物理世界中进行隐蔽和完全被动的攻击。具体地说，我们演示了反向反射和绝缘塑料胶带如何操纵近红外图像的强度分布。我们展示了对基于YOLO的人体探测器的攻击，使用在数字空间设计的二进制模式(通过黑盒查询和搜索)，然后使用粘贴在衣服上的磁带物理实现。我们的攻击突显了人们对夜间监控系统可靠性的重大担忧，这些系统旨在增强安全性。可用代码：https://github.com/MyNiuuu/AdvNIR



## **47. Mitigating Adversarial Attacks in LLMs through Defensive Suffix Generation**

通过防御性后缀生成缓解LLM中的对抗攻击 cs.CV

9 pages, 2 figures

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13705v1) [paper-pdf](http://arxiv.org/pdf/2412.13705v1)

**Authors**: Minkyoung Kim, Yunha Kim, Hyeram Seo, Heejung Choi, Jiye Han, Gaeun Kee, Soyoung Ko, HyoJe Jung, Byeolhee Kim, Young-Hak Kim, Sanghyun Park, Tae Joon Jun

**Abstract**: Large language models (LLMs) have exhibited outstanding performance in natural language processing tasks. However, these models remain susceptible to adversarial attacks in which slight input perturbations can lead to harmful or misleading outputs. A gradient-based defensive suffix generation algorithm is designed to bolster the robustness of LLMs. By appending carefully optimized defensive suffixes to input prompts, the algorithm mitigates adversarial influences while preserving the models' utility. To enhance adversarial understanding, a novel total loss function ($L_{\text{total}}$) combining defensive loss ($L_{\text{def}}$) and adversarial loss ($L_{\text{adv}}$) generates defensive suffixes more effectively. Experimental evaluations conducted on open-source LLMs such as Gemma-7B, mistral-7B, Llama2-7B, and Llama2-13B show that the proposed method reduces attack success rates (ASR) by an average of 11\% compared to models without defensive suffixes. Additionally, the perplexity score of Gemma-7B decreased from 6.57 to 3.93 when applying the defensive suffix generated by openELM-270M. Furthermore, TruthfulQA evaluations demonstrate consistent improvements with Truthfulness scores increasing by up to 10\% across tested configurations. This approach significantly enhances the security of LLMs in critical applications without requiring extensive retraining.

摘要: 大型语言模型(LLM)在自然语言处理任务中表现出优异的性能。然而，这些模型仍然容易受到对抗性攻击，在这种攻击中，轻微的输入扰动可能会导致有害或误导性的输出。设计了一种基于梯度的防御性后缀生成算法，增强了LLMS的健壮性。通过在输入提示中添加经过精心优化的防御性后缀，该算法在保持模型实用性的同时减轻了对抗性影响。为了增强对对手的理解，一种新的总损失函数($L_{\Text{TOTAL}}$)结合了防御损失($L_{\Text{def}}$)和对抗性损失($L_{\Text{adv}}$)，更有效地生成防御后缀。在Gema-7B、Mistral-7B、Llama2-7B和Llama2-13B等开源LLMS上进行的实验评估表明，与没有防御后缀的模型相比，该方法的攻击成功率(ASR)平均降低了11%。此外，使用由OpenELM-270M生成的防御后缀后，GEMA-7B的困惑分数从6.57降至3.93。此外，TruthfulQA评估显示出持续的改进，在测试的配置中，真实性分数提高了高达10%。这种方法显著增强了关键应用中的低成本管理系统的安全性，而无需进行广泛的再培训。



## **48. Enhancing Adversarial Transferability with Adversarial Weight Tuning**

通过对抗权重调整增强对抗可移植性 cs.CR

Accepted by AAAI 2025

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2408.09469v3) [paper-pdf](http://arxiv.org/pdf/2408.09469v3)

**Authors**: Jiahao Chen, Zhou Feng, Rui Zeng, Yuwen Pu, Chunyi Zhou, Yi Jiang, Yuyou Gan, Jinbao Li, Shouling Ji

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial examples (AEs) that mislead the model while appearing benign to human observers. A critical concern is the transferability of AEs, which enables black-box attacks without direct access to the target model. However, many previous attacks have failed to explain the intrinsic mechanism of adversarial transferability. In this paper, we rethink the property of transferable AEs and reformalize the formulation of transferability. Building on insights from this mechanism, we analyze the generalization of AEs across models with different architectures and prove that we can find a local perturbation to mitigate the gap between surrogate and target models. We further establish the inner connections between model smoothness and flat local maxima, both of which contribute to the transferability of AEs. Further, we propose a new adversarial attack algorithm, \textbf{A}dversarial \textbf{W}eight \textbf{T}uning (AWT), which adaptively adjusts the parameters of the surrogate model using generated AEs to optimize the flat local maxima and model smoothness simultaneously, without the need for extra data. AWT is a data-free tuning method that combines gradient-based and model-based attack methods to enhance the transferability of AEs. Extensive experiments on a variety of models with different architectures on ImageNet demonstrate that AWT yields superior performance over other attacks, with an average increase of nearly 5\% and 10\% attack success rates on CNN-based and Transformer-based models, respectively, compared to state-of-the-art attacks.

摘要: 深度神经网络(DNN)很容易受到敌意例子(AE)的攻击，这些例子误导了模型，同时对人类观察者来说是良性的。一个关键的问题是AEs的可转移性，这使得黑盒攻击能够在不直接访问目标模型的情况下进行。然而，以往的许多攻击都未能解释对抗性转移的内在机制。在本文中，我们重新思考了可转让实体的性质，并对可转让的提法进行了改造。在此机制的基础上，我们分析了不同体系结构模型之间的AEs泛化，并证明了我们可以找到局部扰动来缓解代理模型和目标模型之间的差距。我们进一步建立了模型光滑性与平坦局部极大值之间的内在联系，这两者都有助于AEs的可转移性。在此基础上，提出了一种新的对抗性攻击算法AWT是一种无数据调整方法，它结合了基于梯度和基于模型的攻击方法来增强AE的可转移性。在ImageNet上对不同体系结构的各种模型进行的大量实验表明，AWT的攻击性能优于其他攻击，基于CNN和基于Transformer的模型的攻击成功率比最先进的攻击分别提高了近5%和10%。



## **49. Understanding Key Point Cloud Features for Development Three-dimensional Adversarial Attacks**

了解关键点云功能以开发三维对抗攻击 cs.CV

10 pages, 6 figures

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2210.14164v4) [paper-pdf](http://arxiv.org/pdf/2210.14164v4)

**Authors**: Hanieh Naderi, Chinthaka Dinesh, Ivan V. Bajic, Shohreh Kasaei

**Abstract**: Adversarial attacks pose serious challenges for deep neural network (DNN)-based analysis of various input signals. In the case of three-dimensional point clouds, methods have been developed to identify points that play a key role in network decision, and these become crucial in generating existing adversarial attacks. For example, a saliency map approach is a popular method for identifying adversarial drop points, whose removal would significantly impact the network decision. This paper seeks to enhance the understanding of three-dimensional adversarial attacks by exploring which point cloud features are most important for predicting adversarial points. Specifically, Fourteen key point cloud features such as edge intensity and distance from the centroid are defined, and multiple linear regression is employed to assess their predictive power for adversarial points. Based on critical feature selection insights, a new attack method has been developed to evaluate whether the selected features can generate an attack successfully. Unlike traditional attack methods that rely on model-specific vulnerabilities, this approach focuses on the intrinsic characteristics of the point clouds themselves. It is demonstrated that these features can predict adversarial points across four different DNN architectures, Point Network (PointNet), PointNet++, Dynamic Graph Convolutional Neural Networks (DGCNN), and Point Convolutional Network (PointConv) outperforming random guessing and achieving results comparable to saliency map-based attacks. This study has important engineering applications, such as enhancing the security and robustness of three-dimensional point cloud-based systems in fields like robotics and autonomous driving.

摘要: 对抗性攻击对基于深度神经网络(DNN)的各种输入信号分析提出了严峻的挑战。在三维点云的情况下，已经开发出方法来识别在网络决策中起关键作用的点，并且这些点在产生现有的对抗性攻击时变得至关重要。例如，显著图方法是一种流行的识别对抗性丢弃点的方法，其移除将显著影响网络决策。本文试图通过探索哪些点云特征对预测对抗性点最重要，来提高对三维对抗性攻击的理解。具体地，定义了14个关键点云特征，如边缘强度和到质心的距离，并使用多元线性回归来评估它们对敌对点的预测能力。基于关键特征选择的洞察力，提出了一种新的攻击方法，用于评估所选择的特征是否能够成功地产生攻击。与依赖于模型特定漏洞的传统攻击方法不同，该方法侧重于点云本身的内在特征。实验表明，这些特征可以在点网络(PointNet)、点网络++、动态图卷积神经网络(DGCNN)和点卷积网络(PointConv)四种不同的DNN体系结构上预测敌对点，其性能优于随机猜测，并获得与基于显著图的攻击相当的结果。该研究具有重要的工程应用价值，例如在机器人和自动驾驶等领域中增强基于三维点云的系统的安全性和健壮性。



## **50. Novel AI Camera Camouflage: Face Cloaking Without Full Disguise**

新颖的人工智能相机伪装：没有完全伪装的面部伪装 cs.CV

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13507v1) [paper-pdf](http://arxiv.org/pdf/2412.13507v1)

**Authors**: David Noever, Forrest McKee

**Abstract**: This study demonstrates a novel approach to facial camouflage that combines targeted cosmetic perturbations and alpha transparency layer manipulation to evade modern facial recognition systems. Unlike previous methods -- such as CV dazzle, adversarial patches, and theatrical disguises -- this work achieves effective obfuscation through subtle modifications to key-point regions, particularly the brow, nose bridge, and jawline. Empirical testing with Haar cascade classifiers and commercial systems like BetaFaceAPI and Microsoft Bing Visual Search reveals that vertical perturbations near dense facial key points significantly disrupt detection without relying on overt disguises. Additionally, leveraging alpha transparency attacks in PNG images creates a dual-layer effect: faces remain visible to human observers but disappear in machine-readable RGB layers, rendering them unidentifiable during reverse image searches. The results highlight the potential for creating scalable, low-visibility facial obfuscation strategies that balance effectiveness and subtlety, opening pathways for defeating surveillance while maintaining plausible anonymity.

摘要: 这项研究展示了一种新的面部伪装方法，该方法结合了有针对性的化妆品扰动和阿尔法透明层操作来逃避现代面部识别系统。与以前的方法不同--如令人眼花缭乱的简历、对抗性的补丁和戏剧性的伪装--这项工作通过对关键点区域进行微妙的修改，特别是眉毛、鼻梁和下巴轮廓，实现了有效的混淆。使用Haar级联分类器以及BetaFaceAPI和Microsoft Bing Visual Search等商业系统进行的经验测试表明，密集面部关键点附近的垂直扰动显著干扰检测，而不需要公开的伪装。此外，在PNG图像中利用Alpha透明度攻击会产生双层效果：人脸对人类观察者仍然可见，但在机器可读的RGB层中消失，从而使它们在反向图像搜索期间无法识别。这些结果突显了创建可扩展、低能见度的面部混淆策略的潜力，该策略平衡了有效性和微妙程度，为击败监视开辟了道路，同时保持了看似合理的匿名性。



