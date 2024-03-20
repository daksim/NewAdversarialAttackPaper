# Latest Adversarial Attack Papers
**update at 2024-03-20 15:25:50**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Review of Generative AI Methods in Cybersecurity**

网络安全中的生成性人工智能方法综述 cs.CR

40 pages

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.08701v2) [paper-pdf](http://arxiv.org/pdf/2403.08701v2)

**Authors**: Yagmur Yigit, William J Buchanan, Madjid G Tehrani, Leandros Maglaras

**Abstract**: Over the last decade, Artificial Intelligence (AI) has become increasingly popular, especially with the use of chatbots such as ChatGPT, Gemini, and DALL-E. With this rise, large language models (LLMs) and Generative AI (GenAI) have also become more prevalent in everyday use. These advancements strengthen cybersecurity's defensive posture and open up new attack avenues for adversaries as well. This paper provides a comprehensive overview of the current state-of-the-art deployments of GenAI, covering assaults, jailbreaking, and applications of prompt injection and reverse psychology. This paper also provides the various applications of GenAI in cybercrimes, such as automated hacking, phishing emails, social engineering, reverse cryptography, creating attack payloads, and creating malware. GenAI can significantly improve the automation of defensive cyber security processes through strategies such as dataset construction, safe code development, threat intelligence, defensive measures, reporting, and cyberattack detection. In this study, we suggest that future research should focus on developing robust ethical norms and innovative defense mechanisms to address the current issues that GenAI creates and to also further encourage an impartial approach to its future application in cybersecurity. Moreover, we underscore the importance of interdisciplinary approaches further to bridge the gap between scientific developments and ethical considerations.

摘要: 在过去的十年中，人工智能（AI）变得越来越受欢迎，特别是随着聊天机器人的使用，如ChatGPT，Gemini和DALL—E。随着这种增长，大型语言模型（LLM）和生成人工智能（GenAI）在日常使用中也变得越来越普遍。这些进步加强了网络安全的防御态势，并为对手开辟了新的攻击途径。本文全面概述了GenAI当前最先进的部署，涵盖攻击、越狱以及即时注射和逆向心理学的应用。本文还提供了GenAI在网络犯罪中的各种应用，如自动黑客攻击、网络钓鱼电子邮件、社会工程、反向加密、创建攻击有效载荷和创建恶意软件。GenAI可以通过数据集构建、安全代码开发、威胁情报、防御措施、报告和网络攻击检测等策略显著提高防御性网络安全流程的自动化。在这项研究中，我们建议未来的研究应侧重于制定健全的道德规范和创新的防御机制，以解决GenAI目前造成的问题，并进一步鼓励其在网络安全中的未来应用公正的方法。此外，我们强调跨学科方法的重要性，进一步弥合科学发展与伦理考虑之间的差距。



## **2. As Firm As Their Foundations: Can open-sourced foundation models be used to create adversarial examples for downstream tasks?**

和它们的基础一样坚固：开源的基础模型能被用来为下游任务创建对抗性的例子吗？ cs.CV

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12693v1) [paper-pdf](http://arxiv.org/pdf/2403.12693v1)

**Authors**: Anjun Hu, Jindong Gu, Francesco Pinto, Konstantinos Kamnitsas, Philip Torr

**Abstract**: Foundation models pre-trained on web-scale vision-language data, such as CLIP, are widely used as cornerstones of powerful machine learning systems. While pre-training offers clear advantages for downstream learning, it also endows downstream models with shared adversarial vulnerabilities that can be easily identified through the open-sourced foundation model. In this work, we expose such vulnerabilities in CLIP's downstream models and show that foundation models can serve as a basis for attacking their downstream systems. In particular, we propose a simple yet effective adversarial attack strategy termed Patch Representation Misalignment (PRM). Solely based on open-sourced CLIP vision encoders, this method produces adversaries that simultaneously fool more than 20 downstream models spanning 4 common vision-language tasks (semantic segmentation, object detection, image captioning and visual question-answering). Our findings highlight the concerning safety risks introduced by the extensive usage of public foundational models in the development of downstream systems, calling for extra caution in these scenarios.

摘要: 在网络规模的视觉语言数据上预先训练的基础模型，如CLIP，被广泛用作强大的机器学习系统的基石。虽然预训练为下游学习提供了明显的优势，但它也赋予下游模型共同的对抗漏洞，这些漏洞可以通过开源的基础模型轻松识别。在这项工作中，我们暴露了CLIP的下游模型中的这些漏洞，并表明基础模型可以作为攻击其下游系统的基础。特别地，我们提出了一个简单而有效的对抗攻击策略称为补丁表示失准（PRM）。该方法仅基于开源的CLIP视觉编码器，产生了同时欺骗20多个下游模型的对手，这些模型涵盖了4个常见的视觉语言任务（语义分割、对象检测、图像字幕和视觉问答）。我们的研究结果强调了在下游系统开发中广泛使用公共基础模型所带来的安全风险，呼吁在这些情况下格外谨慎。



## **3. Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices**

保护大型语言模型：威胁、漏洞和负责任的实践 cs.CR

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12503v1) [paper-pdf](http://arxiv.org/pdf/2403.12503v1)

**Authors**: Sara Abdali, Richard Anarfi, CJ Barberan, Jia He

**Abstract**: Large language models (LLMs) have significantly transformed the landscape of Natural Language Processing (NLP). Their impact extends across a diverse spectrum of tasks, revolutionizing how we approach language understanding and generations. Nevertheless, alongside their remarkable utility, LLMs introduce critical security and risk considerations. These challenges warrant careful examination to ensure responsible deployment and safeguard against potential vulnerabilities. This research paper thoroughly investigates security and privacy concerns related to LLMs from five thematic perspectives: security and privacy concerns, vulnerabilities against adversarial attacks, potential harms caused by misuses of LLMs, mitigation strategies to address these challenges while identifying limitations of current strategies. Lastly, the paper recommends promising avenues for future research to enhance the security and risk management of LLMs.

摘要: 大型语言模型（LLM）显著改变了自然语言处理（NLP）的前景。它们的影响延伸到各种任务，彻底改变了我们处理语言理解和世代的方式。然而，除了其显著的实用性，LLM引入了关键的安全和风险考虑。这些挑战值得认真审查，以确保负责任地部署和防范潜在漏洞。本研究论文从五个主题角度彻底调查了与LLM相关的安全和隐私问题：安全和隐私问题，对抗攻击的漏洞，滥用LLM造成的潜在危害，缓解策略，以解决这些挑战，同时确定当前策略的局限性。最后，本文建议了未来研究的有希望的途径，以加强LLM的安全性和风险管理。



## **4. Boosting Transferability in Vision-Language Attacks via Diversification along the Intersection Region of Adversarial Trajectory**

基于对抗轨迹交叉区域的多样化提高视觉语言攻击的可传递性 cs.CV

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12445v1) [paper-pdf](http://arxiv.org/pdf/2403.12445v1)

**Authors**: Sensen Gao, Xiaojun Jia, Xuhong Ren, Ivor Tsang, Qing Guo

**Abstract**: Vision-language pre-training (VLP) models exhibit remarkable capabilities in comprehending both images and text, yet they remain susceptible to multimodal adversarial examples (AEs). Strengthening adversarial attacks and uncovering vulnerabilities, especially common issues in VLP models (e.g., high transferable AEs), can stimulate further research on constructing reliable and practical VLP models. A recent work (i.e., Set-level guidance attack) indicates that augmenting image-text pairs to increase AE diversity along the optimization path enhances the transferability of adversarial examples significantly. However, this approach predominantly emphasizes diversity around the online adversarial examples (i.e., AEs in the optimization period), leading to the risk of overfitting the victim model and affecting the transferability. In this study, we posit that the diversity of adversarial examples towards the clean input and online AEs are both pivotal for enhancing transferability across VLP models. Consequently, we propose using diversification along the intersection region of adversarial trajectory to expand the diversity of AEs. To fully leverage the interaction between modalities, we introduce text-guided adversarial example selection during optimization. Furthermore, to further mitigate the potential overfitting, we direct the adversarial text deviating from the last intersection region along the optimization path, rather than adversarial images as in existing methods. Extensive experiments affirm the effectiveness of our method in improving transferability across various VLP models and downstream vision-and-language tasks (e.g., Image-Text Retrieval(ITR), Visual Grounding(VG), Image Captioning(IC)).

摘要: 视觉-语言预训练(VLP)模型在理解图像和文本方面表现出显著的能力，但它们仍然容易受到多通道对抗性例子(AEs)的影响。加强对抗性攻击，发现VLP模型中的漏洞，特别是VLP模型中的常见问题(例如，高可转移性的AE)，可以刺激对构建可靠和实用的VLP模型的进一步研究。最近的一项工作(即集合级制导攻击)表明，增加图文对以增加优化路径上的声发射多样性显著地提高了对抗性例子的可转移性。然而，这种方法主要强调围绕在线对抗性例子的多样性(即，处于优化期的AEs)，导致过度匹配受害者模型并影响可转移性的风险。在这项研究中，我们假设，针对干净输入和在线AEs的对抗性例子的多样性对于提高VLP模型之间的可转移性都是关键。因此，我们建议沿着对抗性轨迹的交叉点区域进行多样化，以扩大AEs的多样性。为了充分利用通道之间的交互作用，我们在优化过程中引入了文本引导的对抗性实例选择。此外，为了进一步缓解潜在的过拟合，我们沿着优化路径引导偏离最后一个交集区域的对抗性文本，而不是现有方法中的对抗性图像。大量的实验证实了我们的方法在提高各种VLP模型和下游视觉和语言任务(例如，图像-文本检索(ITR)、视觉基础(VG)、图像字幕(IC))之间的可转移性方面的有效性。



## **5. Algorithmic Complexity Attacks on Dynamic Learned Indexes**

动态学习索引的复杂性攻击 cs.DB

VLDB 2024

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12433v1) [paper-pdf](http://arxiv.org/pdf/2403.12433v1)

**Authors**: Rui Yang, Evgenios M. Kornaropoulos, Yue Cheng

**Abstract**: Learned Index Structures (LIS) view a sorted index as a model that learns the data distribution, takes a data element key as input, and outputs the predicted position of the key. The original LIS can only handle lookup operations with no support for updates, rendering it impractical to use for typical workloads. To address this limitation, recent studies have focused on designing efficient dynamic learned indexes. ALEX, as the pioneering dynamic learned index structures, enables dynamism by incorporating a series of design choices, including adaptive key space partitioning, dynamic model retraining, and sophisticated engineering and policies that prioritize read/write performance. While these design choices offer improved average-case performance, the emphasis on flexibility and performance increases the attack surface by allowing adversarial behaviors that maximize ALEX's memory space and time complexity in worst-case scenarios. In this work, we present the first systematic investigation of algorithmic complexity attacks (ACAs) targeting the worst-case scenarios of ALEX. We introduce new ACAs that fall into two categories, space ACAs and time ACAs, which target the memory space and time complexity, respectively. First, our space ACA on data nodes exploits ALEX's gapped array layout and uses Multiple-Choice Knapsack (MCK) to generate an optimal adversarial insertion plan for maximizing the memory consumption at the data node level. Second, our space ACA on internal nodes exploits ALEX's catastrophic cost mitigation mechanism, causing an out-of-memory error with only a few hundred adversarial insertions. Third, our time ACA generates pathological insertions to increase the disparity between the actual key distribution and the linear models of data nodes, deteriorating the runtime performance by up to 1,641X compared to ALEX operating under legitimate workloads.

摘要: 学习索引结构(LIS)将排序后的索引视为学习数据分布的模型，将数据元素关键字作为输入，并输出关键字的预测位置。最初的LIS只能处理查找操作，不支持更新，因此不适用于典型的工作负载。为了解决这一局限性，最近的研究集中在设计高效的动态学习索引上。ALEX作为首创的动态学习索引结构，通过整合一系列设计选择来实现动态化，包括自适应键空间分区、动态模型再培训以及对读/写性能进行优先排序的复杂工程和策略。虽然这些设计选择提供了改进的平均情况下的性能，但对灵活性和性能的强调通过允许对抗性行为来增加攻击面，从而在最坏的情况下最大化Alex的存储空间和时间复杂性。在这项工作中，我们首次系统地研究了针对Alex最坏情况的算法复杂性攻击(ACA)。我们提出了两种新的蚁群算法：空间蚁群算法和时间蚁群算法，分别以存储空间和时间复杂度为目标。首先，我们在数据节点上的空间ACA利用Alex的有间隙的数组布局，并使用多选择背包(MCK)来生成最优的对抗性插入计划，以最大化数据节点级别的内存消耗。其次，我们在内部节点上的空间ACA利用了Alex的灾难性成本降低机制，导致内存不足错误，只有几百个敌意插入。第三，我们的时间ACA生成病态插入，以增加实际密钥分布和数据节点的线性模型之间的差异，与在合法工作负载下操作的Alex相比，运行时性能降低高达1,641倍。



## **6. Electioneering the Network: Dynamic Multi-Step Adversarial Attacks for Community Canvassing**

网络选举：社区游说的动态多步对抗攻击 cs.LG

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12399v1) [paper-pdf](http://arxiv.org/pdf/2403.12399v1)

**Authors**: Saurabh Sharma, Ambuj SIngh

**Abstract**: The problem of online social network manipulation for community canvassing is of real concern in today's world. Motivated by the study of voter models, opinion and polarization dynamics on networks, we model community canvassing as a dynamic process over a network enabled via gradient-based attacks on GNNs. Existing attacks on GNNs are all single-step and do not account for the dynamic cascading nature of information diffusion in networks. We consider the realistic scenario where an adversary uses a GNN as a proxy to predict and manipulate voter preferences, especially uncertain voters. Gradient-based attacks on the GNN inform the adversary of strategic manipulations that can be made to proselytize targeted voters. In particular, we explore $\textit{minimum budget attacks for community canvassing}$ (MBACC). We show that the MBACC problem is NP-Hard and propose Dynamic Multi-Step Adversarial Community Canvassing (MAC) to address it. MAC makes dynamic local decisions based on the heuristic of low budget and high second-order influence to convert and perturb target voters. MAC is a dynamic multi-step attack that discovers low-budget and high-influence targets from which efficient cascading attacks can happen. We evaluate MAC against single-step baselines on the MBACC problem with multiple underlying networks and GNN models. Our experiments show the superiority of MAC which is able to discover efficient multi-hop attacks for adversarial community canvassing. Our code implementation and data is available at https://github.com/saurabhsharma1993/mac.

摘要: 操纵在线社交网络进行社区拉票的问题在当今世界是一个真正令人担忧的问题。基于对网络上选民模型、意见和两极分化动态的研究，我们将社区拉票建模为通过对GNN的基于梯度的攻击而实现的网络上的动态过程。现有对GNN的攻击都是单步进行的，没有考虑到网络中信息传播的动态级联性质。我们考虑了这样的现实场景，其中对手使用GNN作为代理来预测和操纵选民的偏好，特别是不确定的选民。对GNN的基于梯度的攻击告诉对手可以进行战略操纵，以使目标选民皈依教义。特别地，我们研究了$\textit(社区拉票最低预算攻击)$(MBACC)。证明了MBACC问题是NP-Hard问题，并提出了动态多步对抗性社区拉票算法(MAC)来解决该问题。Mac基于低预算和高二阶影响力的启发式方法做出动态的地方决策，以转化和扰乱目标选民。MAC是一种动态的多步骤攻击，它可以发现低预算和高影响力的目标，从中可以发生高效的级联攻击。我们使用多个底层网络和GNN模型针对MBACC问题的单步基线来评估MAC。我们的实验表明了MAC的优越性，它能够发现高效的多跳攻击，用于对抗性社区拉票。我们的代码实现和数据可在https://github.com/saurabhsharma1993/mac.上获得



## **7. Securely Fine-tuning Pre-trained Encoders Against Adversarial Examples**

针对对抗性示例的预训练编码器进行保密微调 cs.CV

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.10801v2) [paper-pdf](http://arxiv.org/pdf/2403.10801v2)

**Authors**: Ziqi Zhou, Minghui Li, Wei Liu, Shengshan Hu, Yechao Zhang, Wei Wan, Lulu Xue, Leo Yu Zhang, Dezhong Yao, Hai Jin

**Abstract**: With the evolution of self-supervised learning, the pre-training paradigm has emerged as a predominant solution within the deep learning landscape. Model providers furnish pre-trained encoders designed to function as versatile feature extractors, enabling downstream users to harness the benefits of expansive models with minimal effort through fine-tuning. Nevertheless, recent works have exposed a vulnerability in pre-trained encoders, highlighting their susceptibility to downstream-agnostic adversarial examples (DAEs) meticulously crafted by attackers. The lingering question pertains to the feasibility of fortifying the robustness of downstream models against DAEs, particularly in scenarios where the pre-trained encoders are publicly accessible to the attackers.   In this paper, we initially delve into existing defensive mechanisms against adversarial examples within the pre-training paradigm. Our findings reveal that the failure of current defenses stems from the domain shift between pre-training data and downstream tasks, as well as the sensitivity of encoder parameters. In response to these challenges, we propose Genetic Evolution-Nurtured Adversarial Fine-tuning (Gen-AF), a two-stage adversarial fine-tuning approach aimed at enhancing the robustness of downstream models. Our extensive experiments, conducted across ten self-supervised training methods and six datasets, demonstrate that Gen-AF attains high testing accuracy and robust testing accuracy against state-of-the-art DAEs.

摘要: 随着自我监督学习的发展，预训练范式已经成为深度学习领域的主要解决方案。模型提供商提供预先训练的编码器，设计为功能丰富的特征提取器，使下游用户能够通过微调以最小的努力利用扩展模型的好处。尽管如此，最近的工作暴露了预训练编码器中的一个漏洞，突出了它们对攻击者精心设计的下游不可知对抗示例（DAE）的敏感性。这个悬而未决的问题涉及针对DAE加强下游模型的鲁棒性的可行性，特别是在攻击者可以公开访问预训练编码器的情况下。   在本文中，我们首先深入研究了现有的防御机制，对抗预训练范式中的对抗示例。我们的研究结果表明，当前防御的失败源于预训练数据和下游任务之间的域转移，以及编码器参数的敏感性。为了应对这些挑战，我们提出了遗传进化培育对抗微调（Gen—AF），一种两阶段对抗微调方法，旨在增强下游模型的鲁棒性。我们在10种自我监督训练方法和6个数据集上进行的广泛实验表明，Gen—AF在最先进的DAE上实现了高测试精度和稳健的测试精度。



## **8. Improving Visual Quality and Transferability of Adversarial Attacks on Face Recognition Simultaneously with Adversarial Restoration**

对抗恢复同时提高对抗攻击的视觉质量和可传递性 cs.CV

\copyright 2023 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2309.01582v4) [paper-pdf](http://arxiv.org/pdf/2309.01582v4)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Ping Li

**Abstract**: Adversarial face examples possess two critical properties: Visual Quality and Transferability. However, existing approaches rarely address these properties simultaneously, leading to subpar results. To address this issue, we propose a novel adversarial attack technique known as Adversarial Restoration (AdvRestore), which enhances both visual quality and transferability of adversarial face examples by leveraging a face restoration prior. In our approach, we initially train a Restoration Latent Diffusion Model (RLDM) designed for face restoration. Subsequently, we employ the inference process of RLDM to generate adversarial face examples. The adversarial perturbations are applied to the intermediate features of RLDM. Additionally, by treating RLDM face restoration as a sibling task, the transferability of the generated adversarial face examples is further improved. Our experimental results validate the effectiveness of the proposed attack method.

摘要: 对抗性人脸样本具有两个关键属性：视觉质量和可移植性。然而，现有的方法很少同时解决这些属性，导致低于标准的结果。为了解决这个问题，我们提出了一种新的对抗攻击技术，称为对抗恢复（AdvRestore），它提高了视觉质量和转移性的对抗人脸样本利用人脸恢复之前。在我们的方法中，我们首先训练一个恢复潜在扩散模型（RLDM）设计的面部修复。随后，我们使用RLDM的推理过程来生成对抗人脸样本。对抗扰动被应用于RLDM的中间特征。此外，通过将RLDM人脸恢复作为兄弟任务处理，进一步提高了生成的对抗人脸样本的可移植性。实验结果验证了该攻击方法的有效性。



## **9. Large language models in 6G security: challenges and opportunities**

6G安全中的大型语言模型：挑战与机遇 cs.CR

29 pages, 2 figures

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.12239v1) [paper-pdf](http://arxiv.org/pdf/2403.12239v1)

**Authors**: Tri Nguyen, Huong Nguyen, Ahmad Ijaz, Saeid Sheikhi, Athanasios V. Vasilakos, Panos Kostakos

**Abstract**: The rapid integration of Generative AI (GenAI) and Large Language Models (LLMs) in sectors such as education and healthcare have marked a significant advancement in technology. However, this growth has also led to a largely unexplored aspect: their security vulnerabilities. As the ecosystem that includes both offline and online models, various tools, browser plugins, and third-party applications continues to expand, it significantly widens the attack surface, thereby escalating the potential for security breaches. These expansions in the 6G and beyond landscape provide new avenues for adversaries to manipulate LLMs for malicious purposes. We focus on the security aspects of LLMs from the viewpoint of potential adversaries. We aim to dissect their objectives and methodologies, providing an in-depth analysis of known security weaknesses. This will include the development of a comprehensive threat taxonomy, categorizing various adversary behaviors. Also, our research will concentrate on how LLMs can be integrated into cybersecurity efforts by defense teams, also known as blue teams. We will explore the potential synergy between LLMs and blockchain technology, and how this combination could lead to the development of next-generation, fully autonomous security solutions. This approach aims to establish a unified cybersecurity strategy across the entire computing continuum, enhancing overall digital security infrastructure.

摘要: 生成人工智能（GenAI）和大型语言模型（LLM）在教育和医疗保健等领域的快速集成标志着技术的重大进步。然而，这种增长也导致了一个基本上未被探索的方面：他们的安全漏洞。随着包括离线和在线模式、各种工具、浏览器插件和第三方应用程序在内的生态系统不断扩大，攻击面显著扩大，从而增加了安全漏洞的可能性。6G及其他领域的这些扩展为对手操纵LLM以达到恶意目的提供了新的途径。我们从潜在对手的角度关注LLM的安全方面。我们的目标是剖析他们的目标和方法，对已知的安全弱点进行深入分析。这将包括开发一个全面的威胁分类，对各种攻击者行为进行分类。此外，我们的研究将集中在如何将LLM整合到防御团队（也称为蓝队）的网络安全工作中。我们将探索LLM和区块链技术之间的潜在协同作用，以及这种结合如何导致下一代完全自主的安全解决方案的开发。该方法旨在在整个计算连续体中建立统一的网络安全战略，增强整体数字安全基础设施。



## **10. Adversarial Training Should Be Cast as a Non-Zero-Sum Game**

对抗性训练应视为一种非零和博弈 cs.LG

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2306.11035v2) [paper-pdf](http://arxiv.org/pdf/2306.11035v2)

**Authors**: Alexander Robey, Fabian Latorre, George J. Pappas, Hamed Hassani, Volkan Cevher

**Abstract**: One prominent approach toward resolving the adversarial vulnerability of deep neural networks is the two-player zero-sum paradigm of adversarial training, in which predictors are trained against adversarially chosen perturbations of data. Despite the promise of this approach, algorithms based on this paradigm have not engendered sufficient levels of robustness and suffer from pathological behavior like robust overfitting. To understand this shortcoming, we first show that the commonly used surrogate-based relaxation used in adversarial training algorithms voids all guarantees on the robustness of trained classifiers. The identification of this pitfall informs a novel non-zero-sum bilevel formulation of adversarial training, wherein each player optimizes a different objective function. Our formulation yields a simple algorithmic framework that matches and in some cases outperforms state-of-the-art attacks, attains comparable levels of robustness to standard adversarial training algorithms, and does not suffer from robust overfitting.

摘要: 解决深度神经网络对抗脆弱性的一个突出方法是对抗训练的两人零和范式，其中预测器是针对对抗性选择的数据扰动进行训练的。尽管这种方法有希望，但基于这种范式的算法并没有产生足够水平的鲁棒性，并且会遭受像鲁棒过拟合这样的病态行为。为了理解这一缺点，我们首先表明，对抗训练算法中常用的基于代理的松弛方法会使训练过的分类器鲁棒性的所有保证无效。这个陷阱的识别通知了一个新的非零和双水平的对抗训练公式，其中每个球员优化不同的目标函数。我们的公式产生了一个简单的算法框架，匹配并在某些情况下优于最先进的攻击，达到了与标准对抗训练算法相当的鲁棒性水平，并且不会遭受鲁棒过拟合。



## **11. Diffusion Denoising as a Certified Defense against Clean-label Poisoning**

扩散降噪作为清洁标签中毒的认证防御 cs.CR

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11981v1) [paper-pdf](http://arxiv.org/pdf/2403.11981v1)

**Authors**: Sanghyun Hong, Nicholas Carlini, Alexey Kurakin

**Abstract**: We present a certified defense to clean-label poisoning attacks. These attacks work by injecting a small number of poisoning samples (e.g., 1%) that contain $p$-norm bounded adversarial perturbations into the training data to induce a targeted misclassification of a test-time input. Inspired by the adversarial robustness achieved by $denoised$ $smoothing$, we show how an off-the-shelf diffusion model can sanitize the tampered training data. We extensively test our defense against seven clean-label poisoning attacks and reduce their attack success to 0-16% with only a negligible drop in the test time accuracy. We compare our defense with existing countermeasures against clean-label poisoning, showing that the defense reduces the attack success the most and offers the best model utility. Our results highlight the need for future work on developing stronger clean-label attacks and using our certified yet practical defense as a strong baseline to evaluate these attacks.

摘要: 我们提出了一个认证的防御清洁标签中毒攻击。这些攻击通过注入少量中毒样本（例如，1%），其中包含$p $范数有界对抗扰动到训练数据中，以诱导测试时输入的有针对性的错误分类。受$去噪$$平滑$所实现的对抗鲁棒性的启发，我们展示了一个现成的扩散模型如何可以净化篡改的训练数据。我们广泛测试了我们对七种清洁标签中毒攻击的防御，并将其攻击成功率降低到0—16%，测试时间准确性只有微不足道的下降。我们将我们的防御与现有的针对清洁标签中毒的对策进行了比较，表明该防御最大程度地减少了攻击成功，并提供了最佳的模型效用。我们的研究结果强调，未来需要开发更强的清洁标签攻击，并使用我们认证但实用的防御作为评估这些攻击的强有力基线。



## **12. Enhancing the Antidote: Improved Pointwise Certifications against Poisoning Attacks**

加强解药：改进针对中毒攻击的点状认证 cs.LG

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2308.07553v2) [paper-pdf](http://arxiv.org/pdf/2308.07553v2)

**Authors**: Shijie Liu, Andrew C. Cullen, Paul Montague, Sarah M. Erfani, Benjamin I. P. Rubinstein

**Abstract**: Poisoning attacks can disproportionately influence model behaviour by making small changes to the training corpus. While defences against specific poisoning attacks do exist, they in general do not provide any guarantees, leaving them potentially countered by novel attacks. In contrast, by examining worst-case behaviours Certified Defences make it possible to provide guarantees of the robustness of a sample against adversarial attacks modifying a finite number of training samples, known as pointwise certification. We achieve this by exploiting both Differential Privacy and the Sampled Gaussian Mechanism to ensure the invariance of prediction for each testing instance against finite numbers of poisoned examples. In doing so, our model provides guarantees of adversarial robustness that are more than twice as large as those provided by prior certifications.

摘要: 中毒攻击可以通过对训练语料库进行微小的更改来不成比例地影响模型行为。虽然确实存在针对特定中毒攻击的防御措施，但它们通常不提供任何保证，从而可能被新的攻击所抵消。相比之下，通过检查最坏情况下的行为，Certified Defences可以保证样本对修改有限数量训练样本的对抗攻击的鲁棒性，称为逐点认证。我们通过利用差分隐私和采样高斯机制来实现这一点，以确保每个测试实例对有限数量的中毒实例的预测不变性。在这样做的过程中，我们的模型提供了对抗性鲁棒性的保证，其保证是先前认证所提供的保证的两倍多。



## **13. SSCAE -- Semantic, Syntactic, and Context-aware natural language Adversarial Examples generator**

SSCAE——语义、句法和上下文感知自然语言对抗示例生成器 cs.CL

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11833v1) [paper-pdf](http://arxiv.org/pdf/2403.11833v1)

**Authors**: Javad Rafiei Asl, Mohammad H. Rafiei, Manar Alohaly, Daniel Takabi

**Abstract**: Machine learning models are vulnerable to maliciously crafted Adversarial Examples (AEs). Training a machine learning model with AEs improves its robustness and stability against adversarial attacks. It is essential to develop models that produce high-quality AEs. Developing such models has been much slower in natural language processing (NLP) than in areas such as computer vision. This paper introduces a practical and efficient adversarial attack model called SSCAE for \textbf{S}emantic, \textbf{S}yntactic, and \textbf{C}ontext-aware natural language \textbf{AE}s generator. SSCAE identifies important words and uses a masked language model to generate an early set of substitutions. Next, two well-known language models are employed to evaluate the initial set in terms of semantic and syntactic characteristics. We introduce (1) a dynamic threshold to capture more efficient perturbations and (2) a local greedy search to generate high-quality AEs. As a black-box method, SSCAE generates humanly imperceptible and context-aware AEs that preserve semantic consistency and the source language's syntactical and grammatical requirements. The effectiveness and superiority of the proposed SSCAE model are illustrated with fifteen comparative experiments and extensive sensitivity analysis for parameter optimization. SSCAE outperforms the existing models in all experiments while maintaining a higher semantic consistency with a lower query number and a comparable perturbation rate.

摘要: 机器学习模型容易受到恶意构建的对抗示例（AE）的影响。使用AE训练机器学习模型可以提高其对抗性攻击的鲁棒性和稳定性。必须开发产生高质量AE的模型。在自然语言处理（NLP）中，开发此类模型的速度比计算机视觉等领域慢得多。本文针对\textBF {S}语义、\textBF {S}语义和\textBF {C}上下文感知自然语言\textBF {AE}生成器，提出了一种实用高效的对抗攻击模型SSCAE。SSCAE识别重要的单词，并使用掩蔽语言模型生成早期的替换集。接下来，两个著名的语言模型被用来评估初始集的语义和句法特征。我们引入（1）动态阈值来捕获更有效的扰动和（2）局部贪婪搜索来生成高质量的AE。作为一种黑箱方法，SSCAE生成了人类无法感知的上下文感知AE，以保持语义一致性和源语言的句法和语法要求。通过15个对比试验和广泛的参数优化灵敏度分析，说明了所提出的SSCAE模型的有效性和优越性。SSCAE在所有实验中都优于现有模型，同时保持较高的语义一致性，查询次数较低，扰动率相当。



## **14. Problem space structural adversarial attacks for Network Intrusion Detection Systems based on Graph Neural Networks**

基于图神经网络的网络入侵检测系统的问题空间结构对抗攻击 cs.CR

preprint submitted to IEEE TIFS, under review

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11830v1) [paper-pdf](http://arxiv.org/pdf/2403.11830v1)

**Authors**: Andrea Venturi, Dario Stabili, Mirco Marchetti

**Abstract**: Machine Learning (ML) algorithms have become increasingly popular for supporting Network Intrusion Detection Systems (NIDS). Nevertheless, extensive research has shown their vulnerability to adversarial attacks, which involve subtle perturbations to the inputs of the models aimed at compromising their performance. Recent proposals have effectively leveraged Graph Neural Networks (GNN) to produce predictions based also on the structural patterns exhibited by intrusions to enhance the detection robustness. However, the adoption of GNN-based NIDS introduces new types of risks. In this paper, we propose the first formalization of adversarial attacks specifically tailored for GNN in network intrusion detection. Moreover, we outline and model the problem space constraints that attackers need to consider to carry out feasible structural attacks in real-world scenarios. As a final contribution, we conduct an extensive experimental campaign in which we launch the proposed attacks against state-of-the-art GNN-based NIDS. Our findings demonstrate the increased robustness of the models against classical feature-based adversarial attacks, while highlighting their susceptibility to structure-based attacks.

摘要: 机器学习（ML）算法在支持网络入侵检测系统（NIDS）方面已经变得越来越受欢迎。然而，广泛的研究表明，它们容易受到对抗性攻击，这种攻击涉及对模型输入的微妙扰动，旨在损害其性能。最近的建议已经有效地利用图神经网络（GNN）来产生预测，也基于入侵表现出的结构模式，以增强检测的鲁棒性。然而，采用基于GNN的NIDS引入了新类型的风险。在本文中，我们提出了第一个形式化的对抗攻击专门为GNN在网络入侵检测。此外，我们概述和建模的问题空间约束，攻击者需要考虑进行可行的结构性攻击在现实世界的场景。作为最后的贡献，我们进行了一个广泛的实验活动，在该活动中，我们发起了针对最先进的基于GNN的NIDS的攻击。我们的研究结果表明，这些模型对经典的基于特征的对抗性攻击的鲁棒性增强，同时突出了它们对基于结构的攻击的敏感性。



## **15. Expressive Losses for Verified Robustness via Convex Combinations**

基于凸组合的鲁棒性验证的表达损失 cs.LG

ICLR 2024

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2305.13991v3) [paper-pdf](http://arxiv.org/pdf/2305.13991v3)

**Authors**: Alessandro De Palma, Rudy Bunel, Krishnamurthy Dvijotham, M. Pawan Kumar, Robert Stanforth, Alessio Lomuscio

**Abstract**: In order to train networks for verified adversarial robustness, it is common to over-approximate the worst-case loss over perturbation regions, resulting in networks that attain verifiability at the expense of standard performance. As shown in recent work, better trade-offs between accuracy and robustness can be obtained by carefully coupling adversarial training with over-approximations. We hypothesize that the expressivity of a loss function, which we formalize as the ability to span a range of trade-offs between lower and upper bounds to the worst-case loss through a single parameter (the over-approximation coefficient), is key to attaining state-of-the-art performance. To support our hypothesis, we show that trivial expressive losses, obtained via convex combinations between adversarial attacks and IBP bounds, yield state-of-the-art results across a variety of settings in spite of their conceptual simplicity. We provide a detailed analysis of the relationship between the over-approximation coefficient and performance profiles across different expressive losses, showing that, while expressivity is essential, better approximations of the worst-case loss are not necessarily linked to superior robustness-accuracy trade-offs.

摘要: 为了训练网络以获得验证的对抗鲁棒性，通常会过度近似扰动区域的最坏情况损失，导致网络以牺牲标准性能为代价获得可验证性。正如最近的工作所示，通过谨慎地将对抗训练与过近似耦合，可以在准确性和鲁棒性之间获得更好的权衡。我们假设损失函数的表现性，我们正式化为通过单个参数（过近似系数）在最坏情况下损失的下限和上限之间进行权衡的能力，是获得最先进性能的关键。为了支持我们的假设，我们表明，微不足道的表达损失，通过对抗攻击和IBP边界之间的凸组合，产生国家的最先进的结果，在各种设置，尽管其概念简单。我们提供了一个详细的关系的过近似系数和性能配置文件跨不同的表达损失，表明，虽然表达是必不可少的，更好的近似最坏情况下损失不一定与优越的鲁棒性准确性权衡。



## **16. Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations**

隐藏在普通视野中：对脆弱患者人群的不可检测的对抗偏见攻击 cs.LG

29 pages, 4 figures

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2402.05713v2) [paper-pdf](http://arxiv.org/pdf/2402.05713v2)

**Authors**: Pranav Kulkarni, Andrew Chan, Nithya Navarathna, Skylar Chan, Paul H. Yi, Vishwa S. Parekh

**Abstract**: The proliferation of artificial intelligence (AI) in radiology has shed light on the risk of deep learning (DL) models exacerbating clinical biases towards vulnerable patient populations. While prior literature has focused on quantifying biases exhibited by trained DL models, demographically targeted adversarial bias attacks on DL models and its implication in the clinical environment remains an underexplored field of research in medical imaging. In this work, we demonstrate that demographically targeted label poisoning attacks can introduce undetectable underdiagnosis bias in DL models. Our results across multiple performance metrics and demographic groups like sex, age, and their intersectional subgroups show that adversarial bias attacks demonstrate high-selectivity for bias in the targeted group by degrading group model performance without impacting overall model performance. Furthermore, our results indicate that adversarial bias attacks result in biased DL models that propagate prediction bias even when evaluated with external datasets.

摘要: 人工智能（AI）在放射学中的激增揭示了深度学习（DL）模型加剧对脆弱患者群体的临床偏见的风险。虽然以前的文献集中在量化训练的DL模型所表现出的偏差，但人口统计学上针对DL模型的对抗性偏差攻击及其在临床环境中的影响仍然是医学成像研究的一个不足的领域。在这项工作中，我们证明了人口统计学目标标签中毒攻击可以在DL模型中引入不可检测的诊断不足偏差。我们在多个性能指标和人口统计学组（如性别、年龄及其交叉子组）上的研究结果表明，对抗性偏见攻击通过降低组模型性能而不影响整体模型性能，显示出目标组中的偏见具有高选择性。此外，我们的研究结果表明，对抗性偏差攻击导致有偏差的DL模型传播预测偏差，即使使用外部数据集进行评估。



## **17. Stop Reasoning! When Multimodal LLMs with Chain-of-Thought Reasoning Meets Adversarial Images**

别说三道四了！当具有思维链推理的多通道LLMS遇到敌意图像时 cs.CV

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2402.14899v2) [paper-pdf](http://arxiv.org/pdf/2402.14899v2)

**Authors**: Zefeng Wang, Zhen Han, Shuo Chen, Fan Xue, Zifeng Ding, Xun Xiao, Volker Tresp, Philip Torr, Jindong Gu

**Abstract**: Recently, Multimodal LLMs (MLLMs) have shown a great ability to understand images. However, like traditional vision models, they are still vulnerable to adversarial images. Meanwhile, Chain-of-Thought (CoT) reasoning has been widely explored on MLLMs, which not only improves model's performance, but also enhances model's explainability by giving intermediate reasoning steps. Nevertheless, there is still a lack of study regarding MLLMs' adversarial robustness with CoT and an understanding of what the rationale looks like when MLLMs infer wrong answers with adversarial images. Our research evaluates the adversarial robustness of MLLMs when employing CoT reasoning, finding that CoT marginally improves adversarial robustness against existing attack methods. Moreover, we introduce a novel stop-reasoning attack technique that effectively bypasses the CoT-induced robustness enhancements. Finally, we demonstrate the alterations in CoT reasoning when MLLMs confront adversarial images, shedding light on their reasoning process under adversarial attacks.

摘要: 近年来，多模式LLMS(多模式LLMS)显示出了很强的图像理解能力。然而，像传统的视觉模型一样，它们仍然容易受到敌意图像的影响。同时，思维链式推理在MLLMS上得到了广泛的探索，它不仅改善了模型的性能，而且通过给出中间推理步骤来增强模型的可解释性。然而，仍然缺乏关于MLLMS在COT下的对抗性鲁棒性的研究，以及对MLLMS用对抗性图像推断错误答案的基本原理的理解。我们的研究评估了MLLMS在使用CoT推理时的对抗健壮性，发现CoT略微提高了对现有攻击方法的对抗健壮性。此外，我们引入了一种新的停止推理攻击技术，该技术有效地绕过了CoT诱导的健壮性增强。最后，我们展示了当MLLMS面对对抗性图像时，COT推理的变化，揭示了它们在对抗性攻击下的推理过程。



## **18. LocalStyleFool: Regional Video Style Transfer Attack Using Segment Anything Model**

LocalStyleFool：基于段任意模型的区域视频风格转移攻击 cs.CV

Accepted to 2024 IEEE Security and Privacy Workshops (SPW)

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11656v1) [paper-pdf](http://arxiv.org/pdf/2403.11656v1)

**Authors**: Yuxin Cao, Jinghao Li, Xi Xiao, Derui Wang, Minhui Xue, Hao Ge, Wei Liu, Guangwu Hu

**Abstract**: Previous work has shown that well-crafted adversarial perturbations can threaten the security of video recognition systems. Attackers can invade such models with a low query budget when the perturbations are semantic-invariant, such as StyleFool. Despite the query efficiency, the naturalness of the minutia areas still requires amelioration, since StyleFool leverages style transfer to all pixels in each frame. To close the gap, we propose LocalStyleFool, an improved black-box video adversarial attack that superimposes regional style-transfer-based perturbations on videos. Benefiting from the popularity and scalably usability of Segment Anything Model (SAM), we first extract different regions according to semantic information and then track them through the video stream to maintain the temporal consistency. Then, we add style-transfer-based perturbations to several regions selected based on the associative criterion of transfer-based gradient information and regional area. Perturbation fine adjustment is followed to make stylized videos adversarial. We demonstrate that LocalStyleFool can improve both intra-frame and inter-frame naturalness through a human-assessed survey, while maintaining competitive fooling rate and query efficiency. Successful experiments on the high-resolution dataset also showcase that scrupulous segmentation of SAM helps to improve the scalability of adversarial attacks under high-resolution data.

摘要: 先前的研究表明，精心设计的对抗性扰动可能威胁到视频识别系统的安全性。当扰动是语义不变的时，攻击者可以以低查询预算入侵此类模型，例如StyleFool。尽管查询效率很高，细节区域的自然度仍然需要改进，因为StyleFool利用了对每帧中所有像素的风格传输。为了缩小差距，我们提出了LocalStyleFool，一种改进的黑盒视频对抗攻击，它在视频上叠加了基于区域风格转移的扰动。利用Segment Anything Model（SAM）的流行性和可扩展性，首先根据语义信息提取不同的区域，然后通过视频流跟踪它们，以保持时间一致性。然后，我们添加基于风格转移的扰动到几个基于转移的梯度信息和区域面积的关联准则的选择的区域。微扰微调是遵循的，使风格化的视频对抗。我们证明了LocalStyleFool可以提高帧内和帧间的自然度，通过一个人工评估的调查，同时保持有竞争力的愚弄率和查询效率。在高分辨率数据集上的成功实验也表明，SAM的精确分割有助于提高高分辨率数据下对抗性攻击的可扩展性。



## **19. Zeroth-Order Hard-Thresholding: Gradient Error vs. Expansivity**

零阶硬保持：梯度误差与扩展性 cs.LG

Accepted for publication at NeurIPS 2022

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2210.05279v2) [paper-pdf](http://arxiv.org/pdf/2210.05279v2)

**Authors**: William de Vazelhes, Hualin Zhang, Huimin Wu, Xiao-Tong Yuan, Bin Gu

**Abstract**: $\ell_0$ constrained optimization is prevalent in machine learning, particularly for high-dimensional problems, because it is a fundamental approach to achieve sparse learning. Hard-thresholding gradient descent is a dominant technique to solve this problem. However, first-order gradients of the objective function may be either unavailable or expensive to calculate in a lot of real-world problems, where zeroth-order (ZO) gradients could be a good surrogate. Unfortunately, whether ZO gradients can work with the hard-thresholding operator is still an unsolved problem. To solve this puzzle, in this paper, we focus on the $\ell_0$ constrained black-box stochastic optimization problems, and propose a new stochastic zeroth-order gradient hard-thresholding (SZOHT) algorithm with a general ZO gradient estimator powered by a novel random support sampling. We provide the convergence analysis of SZOHT under standard assumptions. Importantly, we reveal a conflict between the deviation of ZO estimators and the expansivity of the hard-thresholding operator, and provide a theoretical minimal value of the number of random directions in ZO gradients. In addition, we find that the query complexity of SZOHT is independent or weakly dependent on the dimensionality under different settings. Finally, we illustrate the utility of our method on a portfolio optimization problem as well as black-box adversarial attacks.

摘要: $\ell_0 $约束优化在机器学习中非常普遍，特别是对于高维问题，因为它是实现稀疏学习的基本方法。硬阈值梯度下降是解决这一问题的主要技术。然而，目标函数的一阶梯度在许多现实世界的问题中可能是不可用的或昂贵的计算，其中零阶（ZO）梯度可能是一个很好的替代品。不幸的是，ZO梯度是否可以与硬阈值算子一起工作仍然是一个未解决的问题。为了解决这一难题，本文针对$\ell_0 $约束随机优化问题，提出了一种新的随机零阶梯度硬阈值算法（SZOHT），该算法采用了一种新的随机支持抽样算法。在标准假设下，给出了SZOHT的收敛性分析。重要的是，我们揭示了ZO估计量的偏差和硬阈值算子的扩展性之间的冲突，并提供了ZO梯度中随机方向数的理论最小值。此外，我们发现在不同的设置下，SZOHT的查询复杂度与维数无关或弱依赖。最后，我们说明了我们的方法在投资组合优化问题上的效用，以及黑箱对抗攻击。



## **20. The Lipschitz-Variance-Margin Tradeoff for Enhanced Randomized Smoothing**

增强随机化平滑的Lipschitz-Variance-March权衡 cs.LG

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2309.16883v4) [paper-pdf](http://arxiv.org/pdf/2309.16883v4)

**Authors**: Blaise Delattre, Alexandre Araujo, Quentin Barthélemy, Alexandre Allauzen

**Abstract**: Real-life applications of deep neural networks are hindered by their unsteady predictions when faced with noisy inputs and adversarial attacks. The certified radius in this context is a crucial indicator of the robustness of models. However how to design an efficient classifier with an associated certified radius? Randomized smoothing provides a promising framework by relying on noise injection into the inputs to obtain a smoothed and robust classifier. In this paper, we first show that the variance introduced by the Monte-Carlo sampling in the randomized smoothing procedure estimate closely interacts with two other important properties of the classifier, \textit{i.e.} its Lipschitz constant and margin. More precisely, our work emphasizes the dual impact of the Lipschitz constant of the base classifier, on both the smoothed classifier and the empirical variance. To increase the certified robust radius, we introduce a different way to convert logits to probability vectors for the base classifier to leverage the variance-margin trade-off. We leverage the use of Bernstein's concentration inequality along with enhanced Lipschitz bounds for randomized smoothing. Experimental results show a significant improvement in certified accuracy compared to current state-of-the-art methods. Our novel certification procedure allows us to use pre-trained models with randomized smoothing, effectively improving the current certification radius in a zero-shot manner.

摘要: 当面对噪声输入和敌意攻击时，深度神经网络的不稳定预测阻碍了其在现实生活中的应用。在这种情况下，认证的半径是模型稳健性的关键指标。然而，如何设计一个具有相关认证半径的高效分类器呢？随机平滑通过向输入中注入噪声来获得平滑和稳健的分类器，从而提供了一种很有前途的框架。本文首先证明了蒙特卡罗抽样在随机平滑过程估计中引入的方差与分类器的另外两个重要性质密切相关，即它的Lipschitz常数和边际。更准确地说，我们的工作强调了基分类器的Lipschitz常数对平滑的分类器和经验方差的双重影响。为了增加认证的稳健半径，我们引入了一种不同的方法来将对数转换为基本分类器的概率向量，以利用方差与边际之间的权衡。我们利用Bernstein的浓度不等以及增强的Lipschitz界来进行随机平滑。实验结果表明，与目前最先进的方法相比，认证的准确率有了显著的提高。我们新的认证程序允许我们使用带有随机平滑的预先训练的模型，以零射击的方式有效地改进了当前的认证半径。



## **21. SSAP: A Shape-Sensitive Adversarial Patch for Comprehensive Disruption of Monocular Depth Estimation in Autonomous Navigation Applications**

SSAP：自主导航应用中单目深度估计综合干扰的形状敏感对抗性补丁 cs.CV

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11515v1) [paper-pdf](http://arxiv.org/pdf/2403.11515v1)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Ihsen Alouani, Bassem Ouni, Muhammad Shafique

**Abstract**: Monocular depth estimation (MDE) has advanced significantly, primarily through the integration of convolutional neural networks (CNNs) and more recently, Transformers. However, concerns about their susceptibility to adversarial attacks have emerged, especially in safety-critical domains like autonomous driving and robotic navigation. Existing approaches for assessing CNN-based depth prediction methods have fallen short in inducing comprehensive disruptions to the vision system, often limited to specific local areas. In this paper, we introduce SSAP (Shape-Sensitive Adversarial Patch), a novel approach designed to comprehensively disrupt monocular depth estimation (MDE) in autonomous navigation applications. Our patch is crafted to selectively undermine MDE in two distinct ways: by distorting estimated distances or by creating the illusion of an object disappearing from the system's perspective. Notably, our patch is shape-sensitive, meaning it considers the specific shape and scale of the target object, thereby extending its influence beyond immediate proximity. Furthermore, our patch is trained to effectively address different scales and distances from the camera. Experimental results demonstrate that our approach induces a mean depth estimation error surpassing 0.5, impacting up to 99% of the targeted region for CNN-based MDE models. Additionally, we investigate the vulnerability of Transformer-based MDE models to patch-based attacks, revealing that SSAP yields a significant error of 0.59 and exerts substantial influence over 99% of the target region on these models.

摘要: 单眼深度估计(MDE)已经有了显著的进步，主要是通过卷积神经网络(CNN)的集成，以及最近的Transformers。然而，对它们易受对手攻击的担忧已经出现，特别是在自动驾驶和机器人导航等安全关键领域。现有的评估基于CNN的深度预测方法的方法在导致视觉系统全面中断方面做得不够，通常仅限于特定的局部地区。在本文中，我们介绍了形状敏感对抗性补丁(SSAP)，一种新的方法，旨在全面扰乱单眼深度估计(MDE)在自主导航应用中。我们的补丁是为了有选择地以两种不同的方式削弱MDE：通过扭曲估计的距离或通过创造物体从系统的角度消失的错觉。值得注意的是，我们的面片是形状敏感的，这意味着它考虑目标对象的特定形状和比例，从而将其影响扩展到直接邻近之外。此外，我们的补丁经过训练，可以有效地处理不同比例和距离相机的问题。实验结果表明，对于基于CNN的MDE模型，该方法的平均深度估计误差超过0.5，影响高达99%的目标区域。此外，我们研究了基于Transformer的MDE模型对基于补丁的攻击的脆弱性，发现SSAP产生了0.59的显著误差，并且对这些模型上99%的目标区域产生了重大影响。



## **22. Robust Overfitting Does Matter: Test-Time Adversarial Purification With FGSM**

鲁棒过拟合很重要：使用FGSM的测试时间对抗纯化 cs.CV

CVPR 2024

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11448v1) [paper-pdf](http://arxiv.org/pdf/2403.11448v1)

**Authors**: Linyu Tang, Lei Zhang

**Abstract**: Numerous studies have demonstrated the susceptibility of deep neural networks (DNNs) to subtle adversarial perturbations, prompting the development of many advanced adversarial defense methods aimed at mitigating adversarial attacks. Current defense strategies usually train DNNs for a specific adversarial attack method and can achieve good robustness in defense against this type of adversarial attack. Nevertheless, when subjected to evaluations involving unfamiliar attack modalities, empirical evidence reveals a pronounced deterioration in the robustness of DNNs. Meanwhile, there is a trade-off between the classification accuracy of clean examples and adversarial examples. Most defense methods often sacrifice the accuracy of clean examples in order to improve the adversarial robustness of DNNs. To alleviate these problems and enhance the overall robust generalization of DNNs, we propose the Test-Time Pixel-Level Adversarial Purification (TPAP) method. This approach is based on the robust overfitting characteristic of DNNs to the fast gradient sign method (FGSM) on training and test datasets. It utilizes FGSM for adversarial purification, to process images for purifying unknown adversarial perturbations from pixels at testing time in a "counter changes with changelessness" manner, thereby enhancing the defense capability of DNNs against various unknown adversarial attacks. Extensive experimental results show that our method can effectively improve both overall robust generalization of DNNs, notably over previous methods.

摘要: 许多研究已经证明了深度神经网络（DNN）对微妙的对抗干扰的敏感性，这促使了许多旨在减轻对抗攻击的先进对抗防御方法的开发。目前的防御策略通常是针对特定的对抗攻击方法训练DNN，并在防御这种类型的对抗攻击时具有良好的鲁棒性。然而，当受到涉及不熟悉的攻击模式的评估时，经验证据显示DNN的鲁棒性明显恶化。同时，干净样本和对抗样本的分类精度之间存在权衡。大多数防御方法往往牺牲干净示例的准确性，以提高DNN的对抗鲁棒性。为了缓解这些问题，提高DNN的整体鲁棒性，我们提出了测试时间像素级对抗纯化（TPAP）方法。该方法基于DNN对训练和测试数据集的快速梯度符号法（FGSM）的鲁棒过拟合特性。该算法利用FGSM进行对抗性纯化，以“不变逆变”的方式对图像进行处理，从测试时刻像素中纯化未知对抗性扰动，从而增强DNN对各种未知对抗性攻击的防御能力。大量的实验结果表明，我们的方法可以有效地提高DNN的整体鲁棒推广，特别是在以前的方法。



## **23. Defense Against Adversarial Attacks on No-Reference Image Quality Models with Gradient Norm Regularization**

基于梯度范数正则化的无参考图像质量模型的对抗攻击防御 cs.CV

accepted by CVPR 2024

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11397v1) [paper-pdf](http://arxiv.org/pdf/2403.11397v1)

**Authors**: Yujia Liu, Chenxi Yang, Dingquan Li, Jianhao Ding, Tingting Jiang

**Abstract**: The task of No-Reference Image Quality Assessment (NR-IQA) is to estimate the quality score of an input image without additional information. NR-IQA models play a crucial role in the media industry, aiding in performance evaluation and optimization guidance. However, these models are found to be vulnerable to adversarial attacks, which introduce imperceptible perturbations to input images, resulting in significant changes in predicted scores. In this paper, we propose a defense method to improve the stability in predicted scores when attacked by small perturbations, thus enhancing the adversarial robustness of NR-IQA models. To be specific, we present theoretical evidence showing that the magnitude of score changes is related to the $\ell_1$ norm of the model's gradient with respect to the input image. Building upon this theoretical foundation, we propose a norm regularization training strategy aimed at reducing the $\ell_1$ norm of the gradient, thereby boosting the robustness of NR-IQA models. Experiments conducted on four NR-IQA baseline models demonstrate the effectiveness of our strategy in reducing score changes in the presence of adversarial attacks. To the best of our knowledge, this work marks the first attempt to defend against adversarial attacks on NR-IQA models. Our study offers valuable insights into the adversarial robustness of NR-IQA models and provides a foundation for future research in this area.

摘要: 无参考图像质量评估（NR—IQA）的任务是在没有附加信息的情况下估计输入图像的质量分数。NR—IQA模型在媒体行业发挥着至关重要的作用，有助于绩效评估和优化指导。然而，这些模型被发现是容易受到对抗攻击，这引入了难以察觉的干扰输入图像，导致预测分数的显着变化。本文提出了一种防御方法，以提高小扰动攻击时预测分数的稳定性，从而增强了NR—IQA模型的对抗鲁棒性。具体地说，我们提出的理论证据表明得分变化的幅度与模型相对于输入图像的梯度的$\ell_1 $范数有关。在此基础上，我们提出了一种范数正则化训练策略，旨在降低梯度的$\ell_1 $范数，从而提高NR—IQA模型的鲁棒性.在四个NR—IQA基线模型上进行的实验证明了我们的策略在对抗攻击的存在下减少得分变化的有效性。据我们所知，这项工作标志着首次尝试防御对NR—IQA模型的对抗性攻击。我们的研究为NR—IQA模型的对抗鲁棒性提供了宝贵的见解，并为该领域的未来研究提供了基础。



## **24. A Modified Word Saliency-Based Adversarial Attack on Text Classification Models**

一种改进的基于词语显著度的文本分类模型对抗性攻击 cs.CL

The paper is a preprint of a version submitted in ICCIDA 2024. It  consists of 10 pages and contains 7 tables

**SubmitDate**: 2024-03-17    [abs](http://arxiv.org/abs/2403.11297v1) [paper-pdf](http://arxiv.org/pdf/2403.11297v1)

**Authors**: Hetvi Waghela, Sneha Rakshit, Jaydip Sen

**Abstract**: This paper introduces a novel adversarial attack method targeting text classification models, termed the Modified Word Saliency-based Adversarial At-tack (MWSAA). The technique builds upon the concept of word saliency to strategically perturb input texts, aiming to mislead classification models while preserving semantic coherence. By refining the traditional adversarial attack approach, MWSAA significantly enhances its efficacy in evading detection by classification systems. The methodology involves first identifying salient words in the input text through a saliency estimation process, which prioritizes words most influential to the model's decision-making process. Subsequently, these salient words are subjected to carefully crafted modifications, guided by semantic similarity metrics to ensure that the altered text remains coherent and retains its original meaning. Empirical evaluations conducted on diverse text classification datasets demonstrate the effectiveness of the proposed method in generating adversarial examples capable of successfully deceiving state-of-the-art classification models. Comparative analyses with existing adversarial attack techniques further indicate the superiority of the proposed approach in terms of both attack success rate and preservation of text coherence.

摘要: 本文提出了一种新的针对文本分类模型的对抗攻击方法，称为修正词显著性对抗攻击（MWSAA）。该技术建立在单词显著性的概念之上，策略性地扰乱输入文本，旨在误导分类模型，同时保持语义连贯性。通过改进传统的对抗性攻击方法，MWSAA显著提高了其在通过分类系统逃避检测方面的效率。该方法首先通过显着性估计过程识别输入文本中的显着词，该过程优先考虑对模型的决策过程最有影响力的词。随后，这些显著的词经过精心设计的修改，由语义相似性指标指导，以确保修改后的文本保持连贯性并保留其原始含义。在不同的文本分类数据集上进行的经验评估表明，所提出的方法在生成对抗性的例子能够成功地欺骗国家的最先进的分类模型方面的有效性。通过与现有的对抗性攻击技术的对比分析，进一步表明了该方法在攻击成功率和保持文本连贯性方面的优越性。



## **25. Forging the Forger: An Attempt to Improve Authorship Verification via Data Augmentation**

伪造伪造者：通过数据扩充提高作者身份验证的尝试 cs.LG

**SubmitDate**: 2024-03-17    [abs](http://arxiv.org/abs/2403.11265v1) [paper-pdf](http://arxiv.org/pdf/2403.11265v1)

**Authors**: Silvia Corbara, Alejandro Moreo

**Abstract**: Authorship Verification (AV) is a text classification task concerned with inferring whether a candidate text has been written by one specific author or by someone else. It has been shown that many AV systems are vulnerable to adversarial attacks, where a malicious author actively tries to fool the classifier by either concealing their writing style, or by imitating the style of another author. In this paper, we investigate the potential benefits of augmenting the classifier training set with (negative) synthetic examples. These synthetic examples are generated to imitate the style of the author of interest. We analyze the improvements in classifier prediction that this augmentation brings to bear in the task of AV in an adversarial setting. In particular, we experiment with three different generator architectures (one based on Recurrent Neural Networks, another based on small-scale transformers, and another based on the popular GPT model) and with two training strategies (one inspired by standard Language Models, and another inspired by Wasserstein Generative Adversarial Networks). We evaluate our hypothesis on five datasets (three of which have been specifically collected to represent an adversarial setting) and using two learning algorithms for the AV classifier (Support Vector Machines and Convolutional Neural Networks). This experimentation has yielded negative results, revealing that, although our methodology proves effective in many adversarial settings, its benefits are too sporadic for a pragmatical application.

摘要: 作者身份验证（AV）是一个文本分类任务，它涉及推断候选文本是由某个特定的作者写的还是由其他人写的。研究表明，许多反病毒系统容易受到对抗性攻击，恶意作者主动试图通过隐藏他们的写作风格或模仿另一个作者的风格来欺骗分类器。在本文中，我们研究了增加分类器训练集（负）合成示例的潜在好处。生成这些合成示例以模仿感兴趣的作者的风格。我们分析了这种增强在对抗环境中的AV任务中所带来的分类器预测的改进。特别是，我们实验了三种不同的生成器架构（一种基于递归神经网络，另一种基于小规模变换器，另一种基于流行的GPT模型）和两种训练策略（一种受标准语言模型的启发，另一种受Wasserstein生成对抗网络的启发）。我们在五个数据集上评估我们的假设（其中三个已经专门收集来代表对抗性设置），并使用两种学习算法用于AV分类器（支持向量机和卷积神经网络）。这个实验已经产生了负面的结果，揭示了，虽然我们的方法在许多对抗性的环境中被证明是有效的，它的好处是太零星的一个实用的应用。



## **26. A Tip for IOTA Privacy: IOTA Light Node Deanonymization via Tip Selection**

IOTA隐私提示：通过提示选择实现IOTA光节点去匿名化 cs.CR

This paper is accepted to the IEEE International Conference on  Blockchain and Cryptocurrency(ICBC) 2024

**SubmitDate**: 2024-03-17    [abs](http://arxiv.org/abs/2403.11171v1) [paper-pdf](http://arxiv.org/pdf/2403.11171v1)

**Authors**: Hojung Yang, Suhyeon Lee, Seungjoo Kim

**Abstract**: IOTA is a distributed ledger technology that uses a Directed Acyclic Graph (DAG) structure called the Tangle. It is known for its efficiency and is widely used in the Internet of Things (IoT) environment. Tangle can be configured by utilizing the tip selection process. Due to performance issues with light nodes, full nodes are being asked to perform the tip selections of light nodes. However, in this paper, we demonstrate that tip selection can be exploited to compromise users' privacy. An adversary full node can associate a transaction with the identity of a light node by comparing the light node's request with its ledger. We show that these types of attacks are not only viable in the current IOTA environment but also in IOTA 2.0 and the privacy improvement being studied. We also provide solutions to mitigate these attacks and propose ways to enhance anonymity in the IOTA network while maintaining efficiency and scalability.

摘要: IOTA是一种分布式分类帐技术，它使用称为Tangle的有向无环图(DAG)结构。它以其效率而闻名，并在物联网(IoT)环境中广泛使用。可以通过利用尖端选择过程来配置缠绕。由于灯光节点的性能问题，已满节点被要求执行灯光节点的尖端选择。然而，在本文中，我们证明了提示选择可以被利用来损害用户的隐私。对手满节点可以通过将光节点的请求与其分类账进行比较来将事务与光节点的身份相关联。我们表明，这些类型的攻击不仅在当前的IOTA环境下是可行的，而且在IOTA 2.0和正在研究的隐私改善方面也是可行的。我们还提供了缓解这些攻击的解决方案，并提出了在保持效率和可扩展性的同时增强IOTA网络匿名性的方法。



## **27. PubDef: Defending Against Transfer Attacks From Public Models**

PubDef：防御来自公共模型的传输攻击 cs.LG

ICLR 2024. Code available at https://github.com/wagner-group/pubdef

**SubmitDate**: 2024-03-17    [abs](http://arxiv.org/abs/2310.17645v2) [paper-pdf](http://arxiv.org/pdf/2310.17645v2)

**Authors**: Chawin Sitawarin, Jaewon Chang, David Huang, Wesson Altoyan, David Wagner

**Abstract**: Adversarial attacks have been a looming and unaddressed threat in the industry. However, through a decade-long history of the robustness evaluation literature, we have learned that mounting a strong or optimal attack is challenging. It requires both machine learning and domain expertise. In other words, the white-box threat model, religiously assumed by a large majority of the past literature, is unrealistic. In this paper, we propose a new practical threat model where the adversary relies on transfer attacks through publicly available surrogate models. We argue that this setting will become the most prevalent for security-sensitive applications in the future. We evaluate the transfer attacks in this setting and propose a specialized defense method based on a game-theoretic perspective. The defenses are evaluated under 24 public models and 11 attack algorithms across three datasets (CIFAR-10, CIFAR-100, and ImageNet). Under this threat model, our defense, PubDef, outperforms the state-of-the-art white-box adversarial training by a large margin with almost no loss in the normal accuracy. For instance, on ImageNet, our defense achieves 62% accuracy under the strongest transfer attack vs only 36% of the best adversarially trained model. Its accuracy when not under attack is only 2% lower than that of an undefended model (78% vs 80%). We release our code at https://github.com/wagner-group/pubdef.

摘要: 对抗性攻击一直是该行业一个迫在眉睫且尚未解决的威胁。然而，通过长达十年的健壮性评估文献的历史，我们已经了解到，发动强大或最佳的攻击是具有挑战性的。它既需要机器学习，也需要领域专业知识。换句话说，过去大多数文献虔诚地假设的白盒威胁模型是不现实的。在本文中，我们提出了一个新的实用威胁模型，其中敌手通过公开可用的代理模型依赖于传输攻击。我们认为，此设置将成为未来安全敏感应用程序的最流行设置。在此背景下，我们对传输攻击进行了评估，并从博弈论的角度提出了一种专门的防御方法。这些防御在24个公共模型和11个攻击算法下进行了评估，涉及三个数据集(CIFAR-10、CIFAR-100和ImageNet)。在这种威胁模型下，我们的防御PubDef在正常准确率几乎没有损失的情况下，远远超过最先进的白盒对抗训练。例如，在ImageNet上，我们的防御在最强的传输攻击下达到了62%的准确率，而最好的对手训练模型的准确率只有36%。在没有受到攻击的情况下，它的准确性只比没有防御的模型低2%(78%比80%)。我们在https://github.com/wagner-group/pubdef.发布我们的代码



## **28. RobustSentEmbed: Robust Sentence Embeddings Using Adversarial Self-Supervised Contrastive Learning**

RobustSentEmbed：使用对抗自我监督对比学习的稳健句子嵌入 cs.CL

Accepted at the Annual Conference of the North American Chapter of  the Association for Computational Linguistics (NAACL Findings) 2024.  [https://openreview.net/forum?id=9dEAg4lJEA]

**SubmitDate**: 2024-03-17    [abs](http://arxiv.org/abs/2403.11082v1) [paper-pdf](http://arxiv.org/pdf/2403.11082v1)

**Authors**: Javad Rafiei Asl, Prajwal Panzade, Eduardo Blanco, Daniel Takabi, Zhipeng Cai

**Abstract**: Pre-trained language models (PLMs) have consistently demonstrated outstanding performance across a diverse spectrum of natural language processing tasks. Nevertheless, despite their success with unseen data, current PLM-based representations often exhibit poor robustness in adversarial settings. In this paper, we introduce RobustSentEmbed, a self-supervised sentence embedding framework designed to improve both generalization and robustness in diverse text representation tasks and against a diverse set of adversarial attacks. Through the generation of high-risk adversarial perturbations and their utilization in a novel objective function, RobustSentEmbed adeptly learns high-quality and robust sentence embeddings. Our experiments confirm the superiority of RobustSentEmbed over state-of-the-art representations. Specifically, Our framework achieves a significant reduction in the success rate of various adversarial attacks, notably reducing the BERTAttack success rate by almost half (from 75.51\% to 38.81\%). The framework also yields improvements of 1.59\% and 0.23\% in semantic textual similarity tasks and various transfer tasks, respectively.

摘要: 预训练语言模型（PLM）在各种自然语言处理任务中一直表现出出色的性能。尽管如此，尽管他们在未知数据上取得了成功，但当前基于PLM的表示在对抗环境中往往表现出较差的鲁棒性。在本文中，我们介绍了RobustSentEmbed，一个自我监督的句子嵌入框架，旨在提高通用性和鲁棒性，在不同的文本表示任务和对抗性攻击集。通过生成高风险的对抗性扰动并将其用于一个新的目标函数，RobustSentEmbed熟练地学习了高质量和健壮的句子嵌入。我们的实验证实了RobustSentEmbed优于最先进的表示。具体来说，我们的框架实现了各种对抗攻击的成功率的显著降低，特别是降低了几乎一半的BERTattack的成功率（从75.51%到38.81%）。该框架在语义文本相似性任务和各种迁移任务上分别提高了1.59%和0.23%。



## **29. Instance-Level Trojan Attacks on Visual Question Answering via Adversarial Learning in Neuron Activation Space**

基于神经元激活空间对抗学习的视觉问题处理的实例级木马攻击 cs.CV

Accepted for IJCNN 2024

**SubmitDate**: 2024-03-17    [abs](http://arxiv.org/abs/2304.00436v2) [paper-pdf](http://arxiv.org/pdf/2304.00436v2)

**Authors**: Yuwei Sun, Hideya Ochiai, Jun Sakuma

**Abstract**: Trojan attacks embed perturbations in input data leading to malicious behavior in neural network models. A combination of various Trojans in different modalities enables an adversary to mount a sophisticated attack on multimodal learning such as Visual Question Answering (VQA). However, multimodal Trojans in conventional methods are susceptible to parameter adjustment during processes such as fine-tuning. To this end, we propose an instance-level multimodal Trojan attack on VQA that efficiently adapts to fine-tuned models through a dual-modality adversarial learning method. This method compromises two specific neurons in a specific perturbation layer in the pretrained model to produce overly large neuron activations. Then, a malicious correlation between these overactive neurons and the malicious output of a fine-tuned model is established through adversarial learning. Extensive experiments are conducted using the VQA-v2 dataset, based on a wide range of metrics including sample efficiency, stealthiness, and robustness. The proposed attack demonstrates enhanced performance with diverse vision and text Trojans tailored for each sample. We demonstrate that the proposed attack can be efficiently adapted to different fine-tuned models, by injecting only a few shots of Trojan samples. Moreover, we investigate the attack performance under conventional defenses, where the defenses cannot effectively mitigate the attack.

摘要: 木马攻击在输入数据中嵌入干扰，导致神经网络模型中的恶意行为。不同模式下的各种特洛伊木马的组合使对手能够对多模式学习发起复杂的攻击，如视觉问题搜索（VQA）。然而，传统方法中的多模态特洛伊木马在诸如微调等过程中容易受到参数调整的影响。为此，我们提出了一个实例级多模态木马攻击VQA，有效地适应微调模型通过双模态对抗学习方法。这种方法在预训练模型中的特定扰动层中妥协了两个特定神经元，以产生过大的神经元激活。然后，这些过度活跃的神经元与微调模型的恶意输出之间的恶意关联通过对抗学习建立。使用VQA—v2数据集进行了大量的实验，基于包括样本效率、隐蔽性和鲁棒性在内的各种指标。提出的攻击显示出增强的性能与不同的视觉和文本木马为每个样本量身定制。我们证明，所提出的攻击可以有效地适应不同的微调模型，注入只有几个镜头的木马样本。此外，我们研究了传统防御系统下的攻击性能，其中防御系统不能有效地缓解攻击。



## **30. Fast Inference of Removal-Based Node Influence**

基于节点移除的节点影响快速推断 cs.LG

To be published in the Web Conference 2024

**SubmitDate**: 2024-03-16    [abs](http://arxiv.org/abs/2403.08333v2) [paper-pdf](http://arxiv.org/pdf/2403.08333v2)

**Authors**: Weikai Li, Zhiping Xiao, Xiao Luo, Yizhou Sun

**Abstract**: Graph neural networks (GNNs) are widely utilized to capture the information spreading patterns in graphs. While remarkable performance has been achieved, there is a new trending topic of evaluating node influence. We propose a new method of evaluating node influence, which measures the prediction change of a trained GNN model caused by removing a node. A real-world application is, "In the task of predicting Twitter accounts' polarity, had a particular account been removed, how would others' polarity change?". We use the GNN as a surrogate model whose prediction could simulate the change of nodes or edges caused by node removal. Our target is to obtain the influence score for every node, and a straightforward way is to alternately remove every node and apply the trained GNN on the modified graph to generate new predictions. It is reliable but time-consuming, so we need an efficient method. The related lines of work, such as graph adversarial attack and counterfactual explanation, cannot directly satisfy our needs, since their problem settings are different. We propose an efficient, intuitive, and effective method, NOde-Removal-based fAst GNN inference (NORA), which uses the gradient information to approximate the node-removal influence. It only costs one forward propagation and one backpropagation to approximate the influence score for all nodes. Extensive experiments on six datasets and six GNN models verify the effectiveness of NORA. Our code is available at https://github.com/weikai-li/NORA.git.

摘要: 图神经网络（GNN）被广泛用于捕捉图中的信息传播模式。虽然已经取得了显著的性能，有一个新的趋势，评估节点影响力。我们提出了一种新的节点影响评估方法，该方法测量了一个训练的GNN模型由于删除一个节点而引起的预测变化。一个现实世界的应用程序是，“在预测Twitter账户极性的任务中，如果一个特定的账户被删除，其他人的极性会如何改变？".我们使用GNN作为一个代理模型，其预测可以模拟节点移除所引起的节点或边的变化。我们的目标是获得每个节点的影响力得分，一个简单的方法是交替地移除每个节点，并在修改后的图上应用训练的GNN来生成新的预测。它是可靠的，但耗时，所以我们需要一个有效的方法。相关的工作线，如图对抗攻击和反事实解释，不能直接满足我们的需要，因为它们的问题设置不同。我们提出了一种高效、直观和有效的方法——基于节点去除的fAst GNN推理（NORA），它利用梯度信息来近似节点去除的影响。它只花费一个前向传播和一个后向传播来近似所有节点的影响力得分。在6个数据集和6个GNN模型上的大量实验验证了NORA的有效性。我们的代码可在www.example.com获得。



## **31. Understanding Robustness of Visual State Space Models for Image Classification**

视觉状态空间模型在图像分类中的鲁棒性理解 cs.CV

27 pages

**SubmitDate**: 2024-03-16    [abs](http://arxiv.org/abs/2403.10935v1) [paper-pdf](http://arxiv.org/pdf/2403.10935v1)

**Authors**: Chengbin Du, Yanxi Li, Chang Xu

**Abstract**: Visual State Space Model (VMamba) has recently emerged as a promising architecture, exhibiting remarkable performance in various computer vision tasks. However, its robustness has not yet been thoroughly studied. In this paper, we delve into the robustness of this architecture through comprehensive investigations from multiple perspectives. Firstly, we investigate its robustness to adversarial attacks, employing both whole-image and patch-specific adversarial attacks. Results demonstrate superior adversarial robustness compared to Transformer architectures while revealing scalability weaknesses. Secondly, the general robustness of VMamba is assessed against diverse scenarios, including natural adversarial examples, out-of-distribution data, and common corruptions. VMamba exhibits exceptional generalizability with out-of-distribution data but shows scalability weaknesses against natural adversarial examples and common corruptions. Additionally, we explore VMamba's gradients and back-propagation during white-box attacks, uncovering unique vulnerabilities and defensive capabilities of its novel components. Lastly, the sensitivity of VMamba to image structure variations is examined, highlighting vulnerabilities associated with the distribution of disturbance areas and spatial information, with increased susceptibility closer to the image center. Through these comprehensive studies, we contribute to a deeper understanding of VMamba's robustness, providing valuable insights for refining and advancing the capabilities of deep neural networks in computer vision applications.

摘要: 视觉状态空间模型（VMamba）是近年来出现的一种有前途的体系结构，在各种计算机视觉任务中表现出卓越的性能。然而，其鲁棒性尚未得到彻底研究。在本文中，我们通过从多个角度的全面调查来深入研究该架构的鲁棒性。首先，我们研究了它对对抗攻击的鲁棒性，采用整体图像和补丁特定的对抗攻击。结果表明，与Transformer架构相比，具有更好的对抗性鲁棒性，同时也暴露了可伸缩性的弱点。其次，根据不同的场景评估VMamba的总体鲁棒性，包括自然对抗示例、分发数据和常见损坏。VMamba在使用非分发数据时表现出出色的通用性，但在对抗性示例和常见损坏时表现出可伸缩性的弱点。此外，我们还探讨了VMamba在白盒攻击期间的梯度和反向传播，揭示了其新颖组件的独特漏洞和防御能力。最后，VMamba对图像结构变化的敏感性进行了检查，突出了与干扰区域和空间信息的分布相关的脆弱性，与更接近图像中心的敏感性增加。通过这些全面的研究，我们有助于更深入地理解VMamba的稳健性，为完善和提升深度神经网络在计算机视觉应用中的能力提供宝贵的见解。



## **32. Adversarial Examples are Misaligned in Diffusion Model Manifolds**

扩散模型流形中对抗性例子的错位 cs.CV

accepted at IJCNN

**SubmitDate**: 2024-03-16    [abs](http://arxiv.org/abs/2401.06637v5) [paper-pdf](http://arxiv.org/pdf/2401.06637v5)

**Authors**: Peter Lorenz, Ricard Durall, Janis Keuper

**Abstract**: In recent years, diffusion models (DMs) have drawn significant attention for their success in approximating data distributions, yielding state-of-the-art generative results. Nevertheless, the versatility of these models extends beyond their generative capabilities to encompass various vision applications, such as image inpainting, segmentation, adversarial robustness, among others. This study is dedicated to the investigation of adversarial attacks through the lens of diffusion models. However, our objective does not involve enhancing the adversarial robustness of image classifiers. Instead, our focus lies in utilizing the diffusion model to detect and analyze the anomalies introduced by these attacks on images. To that end, we systematically examine the alignment of the distributions of adversarial examples when subjected to the process of transformation using diffusion models. The efficacy of this approach is assessed across CIFAR-10 and ImageNet datasets, including varying image sizes in the latter. The results demonstrate a notable capacity to discriminate effectively between benign and attacked images, providing compelling evidence that adversarial instances do not align with the learned manifold of the DMs.

摘要: 近年来，扩散模型（DM）因其成功地逼近数据分布，产生最先进的生成结果而引起了极大的关注。然而，这些模型的通用性超出了它们的生成能力，涵盖了各种视觉应用，如图像修复、分割、对抗鲁棒性等。本研究致力于通过扩散模型的镜头来研究对抗性攻击。然而，我们的目标并不涉及增强图像分类器的对抗鲁棒性。相反，我们的重点在于利用扩散模型来检测和分析这些攻击对图像的异常。为此，我们系统地研究了对抗性样本的分布时，使用扩散模型进行转换的过程。该方法的有效性在CIFAR—10和ImageNet数据集中进行了评估，包括后者中不同的图像大小。结果表明，一个显着的能力，有效地区分良性和受攻击的图像，提供了令人信服的证据，对抗实例不符合学习流形的DM。



## **33. Improving Adversarial Transferability of Visual-Language Pre-training Models through Collaborative Multimodal Interaction**

协同多模态交互提高视觉语言预训练模型的对抗性传递性 cs.CV

**SubmitDate**: 2024-03-16    [abs](http://arxiv.org/abs/2403.10883v1) [paper-pdf](http://arxiv.org/pdf/2403.10883v1)

**Authors**: Jiyuan Fu, Zhaoyu Chen, Kaixun Jiang, Haijing Guo, Jiafeng Wang, Shuyong Gao, Wenqiang Zhang

**Abstract**: Despite the substantial advancements in Vision-Language Pre-training (VLP) models, their susceptibility to adversarial attacks poses a significant challenge. Existing work rarely studies the transferability of attacks on VLP models, resulting in a substantial performance gap from white-box attacks. We observe that prior work overlooks the interaction mechanisms between modalities, which plays a crucial role in understanding the intricacies of VLP models. In response, we propose a novel attack, called Collaborative Multimodal Interaction Attack (CMI-Attack), leveraging modality interaction through embedding guidance and interaction enhancement. Specifically, attacking text at the embedding level while preserving semantics, as well as utilizing interaction image gradients to enhance constraints on perturbations of texts and images. Significantly, in the image-text retrieval task on Flickr30K dataset, CMI-Attack raises the transfer success rates from ALBEF to TCL, $\text{CLIP}_{\text{ViT}}$ and $\text{CLIP}_{\text{CNN}}$ by 8.11%-16.75% over state-of-the-art methods. Moreover, CMI-Attack also demonstrates superior performance in cross-task generalization scenarios. Our work addresses the underexplored realm of transfer attacks on VLP models, shedding light on the importance of modality interaction for enhanced adversarial robustness.

摘要: 尽管视觉语言预训练(VLP)模式有了很大的进步，但它们对对手攻击的敏感性构成了一个巨大的挑战。现有的工作很少研究攻击对VLP模型的可转移性，导致与白盒攻击相比性能有很大的差距。我们注意到，以前的工作忽略了通道之间的相互作用机制，这在理解VLP模型的复杂性方面起着至关重要的作用。对此，我们提出了一种新的攻击方法，称为协作多模式交互攻击(CMI-Attack)，通过嵌入引导和交互增强来利用通道交互。具体地说，在保留语义的同时在嵌入层攻击文本，以及利用交互图像梯度来增强对文本和图像扰动的约束。值得注意的是，在Flickr30K数据集的图文检索任务中，CMI-Attack将从ALBEF到TCL、$\Text{Clip}_{\Text{Vit}}$和$\Text{Clip}_{\Text{CNN}}$的传输成功率比最先进的方法提高了8.11%-16.75%。此外，CMI-Attack在跨任务泛化场景中也表现出了优越的性能。我们的工作解决了VLP模型上未被探索的传输攻击领域，揭示了通道交互对于增强对手健壮性的重要性。



## **34. TTPXHunter: Actionable Threat Intelligence Extraction as TTPs from Finished Cyber Threat Reports**

TTPXHunter：从完成的网络威胁报告中提取可操作的威胁情报作为TTP cs.CR

Submitted to Journal of Information Security and Applications (JISA)

**SubmitDate**: 2024-03-16    [abs](http://arxiv.org/abs/2403.03267v2) [paper-pdf](http://arxiv.org/pdf/2403.03267v2)

**Authors**: Nanda Rani, Bikash Saha, Vikas Maurya, Sandeep Kumar Shukla

**Abstract**: Understanding the modus operandi of adversaries aids organizations in employing efficient defensive strategies and sharing intelligence in the community. This knowledge is often present in unstructured natural language text within threat analysis reports. A translation tool is needed to interpret the modus operandi explained in the sentences of the threat report and translate it into a structured format. This research introduces a methodology named TTPXHunter for the automated extraction of threat intelligence in terms of Tactics, Techniques, and Procedures (TTPs) from finished cyber threat reports. It leverages cyber domain-specific state-of-the-art natural language processing (NLP) to augment sentences for minority class TTPs and refine pinpointing the TTPs in threat analysis reports significantly. The knowledge of threat intelligence in terms of TTPs is essential for comprehensively understanding cyber threats and enhancing detection and mitigation strategies. We create two datasets: an augmented sentence-TTP dataset of 39,296 samples and a 149 real-world cyber threat intelligence report-to-TTP dataset. Further, we evaluate TTPXHunter on the augmented sentence dataset and the cyber threat reports. The TTPXHunter achieves the highest performance of 92.42% f1-score on the augmented dataset, and it also outperforms existing state-of-the-art solutions in TTP extraction by achieving an f1-score of 97.09% when evaluated over the report dataset. TTPXHunter significantly improves cybersecurity threat intelligence by offering quick, actionable insights into attacker behaviors. This advancement automates threat intelligence analysis, providing a crucial tool for cybersecurity professionals fighting cyber threats.

摘要: 了解对手的作案手法有助于组织采用有效的防御策略，并在社区中分享情报。这种知识通常出现在威胁分析报告中的非结构化自然语言文本中。需要一个翻译工具来解释威胁报告句子中解释的工作方式，并将其翻译成结构化格式。本研究介绍了一种名为TTPXHunter的方法，用于从已完成的网络威胁报告中自动提取策略、技术和过程(TTP)方面的威胁情报。它利用特定于网络领域的最先进的自然语言处理(NLP)来增加少数族裔类TTP的句子，并显著细化威胁分析报告中的TTP。TTP方面的威胁情报知识对于全面了解网络威胁和加强检测和缓解战略至关重要。我们创建了两个数据集：一个包含39,296个样本的增强句-TTP数据集，以及149个真实世界网络威胁情报报告到TTP的数据集。此外，我们在增加句子数据集和网络威胁报告上对TTPXHunter进行了评估。TTPXHunter在增强的数据集上获得了92.42%的F1分数的最高性能，在TTP提取方面也超过了现有的最先进的解决方案，在报告数据集上的F1分数达到了97.09%。TTPXHunter通过提供对攻击者行为的快速、可操作的洞察，显著提高了网络安全威胁情报。这一进步使威胁情报分析自动化，为应对网络威胁的网络安全专业人员提供了一个重要工具。



## **35. Enhancing Adversarial Training with Prior Knowledge Distillation for Robust Image Compression**

基于先验知识提取增强对抗训练的鲁棒图像压缩 eess.IV

**SubmitDate**: 2024-03-16    [abs](http://arxiv.org/abs/2403.06700v2) [paper-pdf](http://arxiv.org/pdf/2403.06700v2)

**Authors**: Zhi Cao, Youneng Bao, Fanyang Meng, Chao Li, Wen Tan, Genhong Wang, Yongsheng Liang

**Abstract**: Deep neural network-based image compression (NIC) has achieved excellent performance, but NIC method models have been shown to be susceptible to backdoor attacks. Adversarial training has been validated in image compression models as a common method to enhance model robustness. However, the improvement effect of adversarial training on model robustness is limited. In this paper, we propose a prior knowledge-guided adversarial training framework for image compression models. Specifically, first, we propose a gradient regularization constraint for training robust teacher models. Subsequently, we design a knowledge distillation based strategy to generate a priori knowledge from the teacher model to the student model for guiding adversarial training. Experimental results show that our method improves the reconstruction quality by about 9dB when the Kodak dataset is elected as the backdoor attack object for psnr attack. Compared with Ma2023, our method has a 5dB higher PSNR output at high bitrate points.

摘要: 基于深度神经网络的图像压缩（NIC）已经取得了出色的性能，但NIC方法模型已被证明容易受到后门攻击。对抗训练已在图像压缩模型中被验证为增强模型鲁棒性的常用方法。然而，对抗训练对模型鲁棒性的改善效果有限。在本文中，我们提出了一个先验知识引导的对抗训练框架的图像压缩模型。具体来说，首先，我们提出了一个梯度正则化约束来训练鲁棒教师模型。然后，我们设计了一个基于知识提炼的策略，生成从教师模型到学生模型的先验知识，指导对抗性训练。实验结果表明，当Kodak数据集被选为后门攻击对象时，该方法的重建质量提高了约9dB。与Ma2023相比，我们的方法在高比特率点有5dB的PSNR输出。



## **36. Bergeron: Combating Adversarial Attacks through a Conscience-Based Alignment Framework**

Bergeron：通过基于意识的调整框架打击对抗性攻击 cs.CR

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2312.00029v2) [paper-pdf](http://arxiv.org/pdf/2312.00029v2)

**Authors**: Matthew Pisano, Peter Ly, Abraham Sanders, Bingsheng Yao, Dakuo Wang, Tomek Strzalkowski, Mei Si

**Abstract**: Research into AI alignment has grown considerably since the recent introduction of increasingly capable Large Language Models (LLMs). Unfortunately, modern methods of alignment still fail to fully prevent harmful responses when models are deliberately attacked. These attacks can trick seemingly aligned models into giving manufacturing instructions for dangerous materials, inciting violence, or recommending other immoral acts. To help mitigate this issue, we introduce Bergeron: a framework designed to improve the robustness of LLMs against attacks without any additional parameter fine-tuning. Bergeron is organized into two tiers; with a secondary LLM emulating the conscience of a protected, primary LLM. This framework better safeguards the primary model against incoming attacks while monitoring its output for any harmful content. Empirical analysis shows that, by using Bergeron to complement models with existing alignment training, we can improve the robustness and safety of multiple, commonly used commercial and open-source LLMs.

摘要: 自从最近引入了越来越强大的大型语言模型（LLM）以来，对人工智能对齐的研究已经有了很大的增长。不幸的是，现代对齐方法仍然无法完全防止模型受到蓄意攻击时的有害反应。这些攻击可以欺骗看似一致的模型给出危险材料的制造指令，煽动暴力，或推荐其他不道德行为。为了帮助缓解这个问题，我们引入了Bergeron：一个旨在提高LLM抵抗攻击的鲁棒性的框架，而无需进行任何额外的参数微调。Bergeron分为两个层次；二级法学硕士模仿受保护的，主要法学硕士的良心。该框架更好地保护了主模型免受传入攻击，同时监控其输出中的任何有害内容。实证分析表明，通过使用Bergeron来补充现有的对齐训练模型，我们可以提高多个常用的商业和开源LLM的鲁棒性和安全性。



## **37. Not Just Change the Labels, Learn the Features: Watermarking Deep Neural Networks with Multi-View Data**

不只是改变标签，学习功能：使用多视图数据水印深度神经网络 cs.CR

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10663v1) [paper-pdf](http://arxiv.org/pdf/2403.10663v1)

**Authors**: Yuxuan Li, Sarthak Kumar Maharana, Yunhui Guo

**Abstract**: With the increasing prevalence of Machine Learning as a Service (MLaaS) platforms, there is a growing focus on deep neural network (DNN) watermarking techniques. These methods are used to facilitate the verification of ownership for a target DNN model to protect intellectual property. One of the most widely employed watermarking techniques involves embedding a trigger set into the source model. Unfortunately, existing methodologies based on trigger sets are still susceptible to functionality-stealing attacks, potentially enabling adversaries to steal the functionality of the source model without a reliable means of verifying ownership. In this paper, we first introduce a novel perspective on trigger set-based watermarking methods from a feature learning perspective. Specifically, we demonstrate that by selecting data exhibiting multiple features, also referred to as $\textit{multi-view data}$, it becomes feasible to effectively defend functionality stealing attacks. Based on this perspective, we introduce a novel watermarking technique based on Multi-view dATa, called MAT, for efficiently embedding watermarks within DNNs. This approach involves constructing a trigger set with multi-view data and incorporating a simple feature-based regularization method for training the source model. We validate our method across various benchmarks and demonstrate its efficacy in defending against model extraction attacks, surpassing relevant baselines by a significant margin.

摘要: 随着机器学习即服务(MLaaS)平台的日益普及，深度神经网络(DNN)水印技术受到越来越多的关注。这些方法用于帮助验证目标DNN模型的所有权，以保护知识产权。最广泛使用的水印技术之一涉及将触发集嵌入到源模型中。遗憾的是，基于触发器集的现有方法仍然容易受到功能窃取攻击，这可能使攻击者能够在没有可靠的验证所有权的方法的情况下窃取源模型的功能。本文首先从特征学习的角度介绍了一种基于触发集的数字水印方法。具体地说，我们证明了通过选择表现出多个特征的数据，也称为$\textit{多视图数据}$，可以有效地防御功能窃取攻击。基于这一观点，我们提出了一种新的基于多视点数据的水印技术，称为MAT，用于在DNN中有效地嵌入水印。该方法包括利用多视点数据构造触发集，并结合简单的基于特征的正则化方法来训练源模型。我们在不同的基准上验证了我们的方法，并证明了它在防御模型提取攻击方面的有效性，远远超过了相关的基线。



## **38. Benchmarking Zero-Shot Robustness of Multimodal Foundation Models: A Pilot Study**

多模态地基模型零炮鲁棒性基准测试：初步研究 cs.LG

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10499v1) [paper-pdf](http://arxiv.org/pdf/2403.10499v1)

**Authors**: Chenguang Wang, Ruoxi Jia, Xin Liu, Dawn Song

**Abstract**: Pre-training image representations from the raw text about images enables zero-shot vision transfer to downstream tasks. Through pre-training on millions of samples collected from the internet, multimodal foundation models, such as CLIP, produce state-of-the-art zero-shot results that often reach competitiveness with fully supervised methods without the need for task-specific training. Besides the encouraging performance on classification accuracy, it is reported that these models close the robustness gap by matching the performance of supervised models trained on ImageNet under natural distribution shift. Because robustness is critical to real-world applications, especially safety-critical ones, in this paper, we present a comprehensive evaluation based on a large-scale robustness benchmark covering 7 natural, 3 synthetic distribution shifts, and 11 adversarial attacks. We use CLIP as a pilot study. We show that CLIP leads to a significant robustness drop compared to supervised ImageNet models on our benchmark, especially under synthetic distribution shift and adversarial attacks. Furthermore, data overlap analysis suggests that the observed robustness under natural distribution shifts could be attributed, at least in part, to data overlap. In summary, our evaluation shows a comprehensive evaluation of robustness is necessary; and there is a significant need to improve the robustness of zero-shot multimodal models.

摘要: 从图像的原始文本中预训练图像表示可以实现零镜头视觉转移到下游任务。通过对从互联网收集的数百万个样本进行预训练，多模态基础模型（如CLIP）可以产生最先进的零射击结果，这些结果通常可以与完全监督的方法相媲美，而无需进行特定任务的训练。除了分类准确性方面令人鼓舞的性能外，据报道，这些模型通过匹配在自然分布偏移下在ImageNet上训练的监督模型的性能，缩小了鲁棒性差距。由于鲁棒性对现实世界的应用程序，尤其是安全关键的应用程序，在本文中，我们提出了一个基于大规模鲁棒性基准的综合评估，涵盖了7个自然，3个合成分布偏移和11个对抗攻击。我们将CLIP作为一项试点研究。我们表明，与我们的基准测试中的监督ImageNet模型相比，CLIP导致了显著的鲁棒性下降，特别是在合成分布偏移和对抗攻击下。此外，数据重叠分析表明，在自然分布偏移下观察到的稳健性至少部分归因于数据重叠。总之，我们的评估表明，有必要对鲁棒性进行综合评估，并且有一个显着的需要来提高零镜头多模态模型的鲁棒性。



## **39. Mitigating Dialogue Hallucination for Large Multi-modal Models via Adversarial Instruction Tuning**

对抗性指令调优缓解大型多模态模型的对话幻觉 cs.CV

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10492v1) [paper-pdf](http://arxiv.org/pdf/2403.10492v1)

**Authors**: Dongmin Park, Zhaofang Qian, Guangxing Han, Ser-Nam Lim

**Abstract**: Mitigating hallucinations of Large Multi-modal Models(LMMs) is crucial to enhance their reliability for general-purpose assistants. This paper shows that such hallucinations of LMMs can be significantly exacerbated by preceding user-system dialogues. To precisely measure this, we first present an evaluation benchmark by extending popular multi-modal benchmark datasets with prepended hallucinatory dialogues generated by our novel Adversarial Question Generator, which can automatically generate image-related yet adversarial dialogues by adopting adversarial attacks on LMMs. On our benchmark, the zero-shot performance of state-of-the-art LMMs dropped significantly for both the VQA and Captioning tasks. Next, we further reveal this hallucination is mainly due to the prediction bias toward preceding dialogues rather than visual content. To reduce this bias, we propose Adversarial Instruction Tuning that robustly fine-tunes LMMs on augmented multi-modal instruction-following datasets with hallucinatory dialogues. Extensive experiments show that our proposed approach successfully reduces dialogue hallucination while maintaining or even improving performance.

摘要: 减少大型多通道模型(LMM)的幻觉对于提高其对通用助理的可靠性至关重要。这篇论文表明，之前的用户-系统对话可以显著加剧LMM的这种幻觉。为了准确地衡量这一点，我们首先提出了一个评估基准，通过扩展流行的多模式基准数据集来预先生成由我们的新型对抗性问题生成器生成的幻觉对话，该生成器可以通过对LMM进行对抗性攻击来自动生成与图像相关的对抗性对话。在我们的基准测试中，最先进的LMM的零点性能在VQA和字幕任务中都显著下降。接下来，我们进一步揭示这种幻觉主要是由于预测偏向于之前的对话而不是视觉内容。为了减少这种偏差，我们提出了对抗性的教学调整，在带有幻觉对话的增强型多模式教学跟踪数据集上强有力地微调LMM。大量的实验表明，我们提出的方法在保持甚至提高性能的同时，成功地减少了对话幻觉。



## **40. Introducing Adaptive Continuous Adversarial Training (ACAT) to Enhance ML Robustness**

引入自适应连续对抗训练（ACAT）以增强ML鲁棒性 cs.LG

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10461v1) [paper-pdf](http://arxiv.org/pdf/2403.10461v1)

**Authors**: Mohamed elShehaby, Aditya Kotha, Ashraf Matrawy

**Abstract**: Machine Learning (ML) is susceptible to adversarial attacks that aim to trick ML models, making them produce faulty predictions. Adversarial training was found to increase the robustness of ML models against these attacks. However, in network and cybersecurity, obtaining labeled training and adversarial training data is challenging and costly. Furthermore, concept drift deepens the challenge, particularly in dynamic domains like network and cybersecurity, and requires various models to conduct periodic retraining. This letter introduces Adaptive Continuous Adversarial Training (ACAT) to continuously integrate adversarial training samples into the model during ongoing learning sessions, using real-world detected adversarial data, to enhance model resilience against evolving adversarial threats. ACAT is an adaptive defense mechanism that utilizes periodic retraining to effectively counter adversarial attacks while mitigating catastrophic forgetting. Our approach also reduces the total time required for adversarial sample detection, especially in environments such as network security where the rate of attacks could be very high. Traditional detection processes that involve two stages may result in lengthy procedures. Experimental results using a SPAM detection dataset demonstrate that with ACAT, the accuracy of the SPAM filter increased from 69% to over 88% after just three retraining sessions. Furthermore, ACAT outperforms conventional adversarial sample detectors, providing faster decision times, up to four times faster in some cases.

摘要: 机器学习（ML）容易受到旨在欺骗ML模型的对抗攻击，使它们产生错误的预测。对抗训练被发现可以提高ML模型对这些攻击的鲁棒性。然而，在网络和网络安全中，获得标记训练和对抗训练数据是一项挑战性和昂贵的工作。此外，概念漂移加深了挑战，特别是在网络和网络安全等动态领域，并要求各种模型进行定期的再培训。本文介绍了自适应连续对抗训练（ACAT），在正在进行的学习过程中，使用真实世界检测到的对抗数据，不断将对抗训练样本集成到模型中，以增强模型对不断演变的对抗威胁的弹性。ACAT是一种自适应防御机制，它利用周期性的再训练来有效地对抗对抗攻击，同时减少灾难性遗忘。我们的方法还减少了对抗样本检测所需的总时间，特别是在网络安全等攻击率可能非常高的环境中。传统的检测过程涉及两个阶段，可能导致程序漫长。使用垃圾邮件检测数据集的实验结果表明，使用ACAT，垃圾邮件过滤器的准确率从69%提高到88%以上，仅仅经过三次再培训。此外，ACAT的性能优于传统的对抗样本检测器，提供更快的决策时间，在某些情况下，最多可达四倍。



## **41. DeepZero: Scaling up Zeroth-Order Optimization for Deep Model Training**

DeepZero：扩展零阶优化用于深度模型训练 cs.LG

Accepted to ICLR'24. Codes are available at  https://github.com/OPTML-Group/DeepZero

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2310.02025v4) [paper-pdf](http://arxiv.org/pdf/2310.02025v4)

**Authors**: Aochuan Chen, Yimeng Zhang, Jinghan Jia, James Diffenderfer, Jiancheng Liu, Konstantinos Parasyris, Yihua Zhang, Zheng Zhang, Bhavya Kailkhura, Sijia Liu

**Abstract**: Zeroth-order (ZO) optimization has become a popular technique for solving machine learning (ML) problems when first-order (FO) information is difficult or impossible to obtain. However, the scalability of ZO optimization remains an open problem: Its use has primarily been limited to relatively small-scale ML problems, such as sample-wise adversarial attack generation. To our best knowledge, no prior work has demonstrated the effectiveness of ZO optimization in training deep neural networks (DNNs) without a significant decrease in performance. To overcome this roadblock, we develop DeepZero, a principled ZO deep learning (DL) framework that can scale ZO optimization to DNN training from scratch through three primary innovations. First, we demonstrate the advantages of coordinatewise gradient estimation (CGE) over randomized vector-wise gradient estimation in training accuracy and computational efficiency. Second, we propose a sparsityinduced ZO training protocol that extends the model pruning methodology using only finite differences to explore and exploit the sparse DL prior in CGE. Third, we develop the methods of feature reuse and forward parallelization to advance the practical implementations of ZO training. Our extensive experiments show that DeepZero achieves state-of-the-art (SOTA) accuracy on ResNet-20 trained on CIFAR-10, approaching FO training performance for the first time. Furthermore, we show the practical utility of DeepZero in applications of certified adversarial defense and DL-based partial differential equation error correction, achieving 10-20% improvement over SOTA. We believe our results will inspire future research on scalable ZO optimization and contribute to advancing DL with black box. Codes are available at https://github.com/OPTML-Group/DeepZero.

摘要: 当一阶信息难以或不可能获得时，零阶优化（ZO）已成为解决机器学习（ML）问题的流行技术。然而，ZO优化的可扩展性仍然是一个开放的问题：它的使用主要限于相对小规模的ML问题，例如样本对抗攻击的生成。据我们所知，之前没有任何工作证明ZO优化在训练深度神经网络（DNN）方面的有效性，而不会显著降低性能。为了克服这一障碍，我们开发了DeepZero，这是一个原则性的ZO深度学习（DL）框架，可以通过三个主要创新将ZO优化扩展到DNN训练。首先，我们证明了坐标梯度估计（CGE）比随机向量梯度估计在训练精度和计算效率方面的优势。其次，我们提出了一个稀疏诱导的ZO训练协议，扩展了模型修剪方法，仅使用有限差分来探索和利用CGE中的稀疏DL先验。第三，我们发展了特征重用和前向并行化的方法，以推进ZO训练的实际实现。我们的大量实验表明，DeepZero在CIFAR—10上训练的ResNet—20上达到了最先进的（SOTA）精度，首次接近FO训练性能。此外，我们还展示了DeepZero在认证对抗防御和基于DL的偏微分方程误差校正应用中的实用性，比SOTA提高了10—20%。我们相信，我们的研究成果将激发未来对可扩展ZO优化的研究，并有助于推进带黑盒的DL。代码可在www.example.com查阅。



## **42. Towards Non-Adversarial Algorithmic Recourse**

走向非对抗性算法追索权 cs.LG

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10330v1) [paper-pdf](http://arxiv.org/pdf/2403.10330v1)

**Authors**: Tobias Leemann, Martin Pawelczyk, Bardh Prenkaj, Gjergji Kasneci

**Abstract**: The streams of research on adversarial examples and counterfactual explanations have largely been growing independently. This has led to several recent works trying to elucidate their similarities and differences. Most prominently, it has been argued that adversarial examples, as opposed to counterfactual explanations, have a unique characteristic in that they lead to a misclassification compared to the ground truth. However, the computational goals and methodologies employed in existing counterfactual explanation and adversarial example generation methods often lack alignment with this requirement. Using formal definitions of adversarial examples and counterfactual explanations, we introduce non-adversarial algorithmic recourse and outline why in high-stakes situations, it is imperative to obtain counterfactual explanations that do not exhibit adversarial characteristics. We subsequently investigate how different components in the objective functions, e.g., the machine learning model or cost function used to measure distance, determine whether the outcome can be considered an adversarial example or not. Our experiments on common datasets highlight that these design choices are often more critical in deciding whether recourse is non-adversarial than whether recourse or attack algorithms are used. Furthermore, we show that choosing a robust and accurate machine learning model results in less adversarial recourse desired in practice.

摘要: 对抗性例子和反事实解释的研究在很大程度上独立增长。这导致了最近的几部作品试图阐明它们的相似和不同之处。最突出的是，有人认为，与反事实解释相反，对抗性的例子有一个独特的特点，那就是它们导致了与基本事实相比的错误分类。然而，现有的反事实解释和对抗性实例生成方法所采用的计算目标和方法往往与这一要求不一致。利用对抗性例子和反事实解释的形式定义，我们引入了非对抗性算法求助，并概述了为什么在高风险的情况下，必须获得不表现出对抗性特征的反事实解释。我们随后研究了目标函数中的不同组件，例如用于测量距离的机器学习模型或代价函数，如何确定结果是否可以被视为对抗性示例。我们在常见数据集上的实验强调，在决定求助是否是非对抗性方面，这些设计选择往往比使用求助或攻击算法更关键。此外，我们还表明，选择一个稳健和准确的机器学习模型可以减少实践中所希望的对抗性求助。



## **43. Interactive Trimming against Evasive Online Data Manipulation Attacks: A Game-Theoretic Approach**

基于博弈论的交互式裁剪方法 cs.CR

This manuscript is accepted by ICDE '24

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10313v1) [paper-pdf](http://arxiv.org/pdf/2403.10313v1)

**Authors**: Yue Fu, Qingqing Ye, Rong Du, Haibo Hu

**Abstract**: With the exponential growth of data and its crucial impact on our lives and decision-making, the integrity of data has become a significant concern. Malicious data poisoning attacks, where false values are injected into the data, can disrupt machine learning processes and lead to severe consequences. To mitigate these attacks, distance-based defenses, such as trimming, have been proposed, but they can be easily evaded by white-box attackers. The evasiveness and effectiveness of poisoning attack strategies are two sides of the same coin, making game theory a promising approach. However, existing game-theoretical models often overlook the complexities of online data poisoning attacks, where strategies must adapt to the dynamic process of data collection.   In this paper, we present an interactive game-theoretical model to defend online data manipulation attacks using the trimming strategy. Our model accommodates a complete strategy space, making it applicable to strong evasive and colluding adversaries. Leveraging the principle of least action and the Euler-Lagrange equation from theoretical physics, we derive an analytical model for the game-theoretic process. To demonstrate its practical usage, we present a case study in a privacy-preserving data collection system under local differential privacy where a non-deterministic utility function is adopted. Two strategies are devised from this analytical model, namely, Tit-for-tat and Elastic. We conduct extensive experiments on real-world datasets, which showcase the effectiveness and accuracy of these two strategies.

摘要: 随着数据的指数级增长及其对我们的生活和决策的关键影响，数据的完整性已经成为一个重要的问题。恶意数据中毒攻击，即将错误值注入数据中，可能会扰乱机器学习过程，并导致严重后果。为了减轻这些攻击，已经提出了基于距离的防御措施，如修剪，但白盒攻击者很容易避开它们。中毒攻击策略的规避性和有效性是同一枚硬币的两面，使博弈论成为一种有前途的方法。然而，现有的博弈论模型往往忽略了在线数据中毒攻击的复杂性，其中策略必须适应数据收集的动态过程。   在本文中，我们提出了一个交互式博弈理论模型，以防御在线数据操纵攻击使用修剪策略。我们的模型容纳了一个完整的战略空间，使其适用于强大的规避和勾结对手。利用理论物理学中的最小作用量原理和欧拉—拉格朗日方程，推导出博弈论过程的解析模型。为了证明其实际应用，我们给出了一个基于局部差分隐私的隐私保护数据采集系统的案例研究，其中采用了非确定性效用函数。根据该分析模型，设计了两种策略，即Tat for Tat和Elastic。我们在现实世界的数据集上进行了广泛的实验，展示了这两种策略的有效性和准确性。



## **44. Chernoff Information as a Privacy Constraint for Adversarial Classification**

Chernoff信息作为对抗性分类的隐私约束 cs.IT

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10307v1) [paper-pdf](http://arxiv.org/pdf/2403.10307v1)

**Authors**: Ayşe Ünsal, Melek Önen

**Abstract**: This work studies a privacy metric based on Chernoff information, \textit{Chernoff differential privacy}, due to its significance in characterization of classifier performance. Adversarial classification, as any other classification problem is built around minimization of the (average or correct detection) probability of error in deciding on either of the classes in the case of binary classification. Unlike the classical hypothesis testing problem, where the false alarm and mis-detection probabilities are handled separately resulting in an asymmetric behavior of the best error exponent, in this work, we focus on the Bayesian setting and characterize the relationship between the best error exponent of the average error probability and $\varepsilon-$differential privacy. Accordingly, we re-derive Chernoff differential privacy in terms of $\varepsilon-$differential privacy using the Radon-Nikodym derivative and show that it satisfies the composition property. Subsequently, we present numerical evaluation results, which demonstrates that Chernoff information outperforms Kullback-Leibler divergence as a function of the privacy parameter $\varepsilon$, the impact of the adversary's attack and global sensitivity for the problem of adversarial classification in Laplace mechanisms.

摘要: 本文研究了一种基于Diff信息的隐私度量\textit {Diff差分隐私}，因为它对分类器性能的表征具有重要意义。对抗分类，就像任何其他分类问题一样，在二进制分类的情况下，都是围绕最小化（平均或正确检测）错误概率而建立的。与经典的假设测试问题不同，其中虚警和误检概率被分别处理，导致最佳错误指数的不对称行为，在这项工作中，我们专注于贝叶斯设置，并表征平均错误概率的最佳错误指数和$\varepident—$差分隐私之间的关系。因此，我们使用Radon—Nikodym导数重新推导出基于$\vareprise—$差分隐私的Risff差分隐私，并证明它满足合成性质。随后，我们提出了数值评估结果，这表明，作为一个函数的隐私参数$\vareprise $，对手的攻击的影响和全局敏感性的对抗分类问题的拉普拉斯机制。



## **45. IRAD: Implicit Representation-driven Image Resampling against Adversarial Attacks**

IRAD：抵抗敌意攻击的隐式表示驱动图像重采样 cs.CV

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2310.11890v2) [paper-pdf](http://arxiv.org/pdf/2310.11890v2)

**Authors**: Yue Cao, Tianlin Li, Xiaofeng Cao, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: We introduce a novel approach to counter adversarial attacks, namely, image resampling. Image resampling transforms a discrete image into a new one, simulating the process of scene recapturing or rerendering as specified by a geometrical transformation. The underlying rationale behind our idea is that image resampling can alleviate the influence of adversarial perturbations while preserving essential semantic information, thereby conferring an inherent advantage in defending against adversarial attacks. To validate this concept, we present a comprehensive study on leveraging image resampling to defend against adversarial attacks. We have developed basic resampling methods that employ interpolation strategies and coordinate shifting magnitudes. Our analysis reveals that these basic methods can partially mitigate adversarial attacks. However, they come with apparent limitations: the accuracy of clean images noticeably decreases, while the improvement in accuracy on adversarial examples is not substantial. We propose implicit representation-driven image resampling (IRAD) to overcome these limitations. First, we construct an implicit continuous representation that enables us to represent any input image within a continuous coordinate space. Second, we introduce SampleNet, which automatically generates pixel-wise shifts for resampling in response to different inputs. Furthermore, we can extend our approach to the state-of-the-art diffusion-based method, accelerating it with fewer time steps while preserving its defense capability. Extensive experiments demonstrate that our method significantly enhances the adversarial robustness of diverse deep models against various attacks while maintaining high accuracy on clean images.

摘要: 本文介绍了一种新的对抗攻击的方法，即图像复原。图像重构将离散图像转换为新图像，模拟由几何变换指定的场景重构或重新渲染的过程。我们的想法背后的基本原理是，图像复位可以减轻对抗干扰的影响，同时保留必要的语义信息，从而赋予防御对抗攻击的固有优势。为了验证这一概念，我们提出了一个全面的研究，利用图像资源来防御对抗攻击。我们已经开发了使用插值策略和坐标移位幅度的基本重新定位方法。我们的分析表明，这些基本方法可以部分地减轻对抗性攻击。然而，它们有明显的局限性：干净图像的准确性明显下降，而对抗性示例的准确性提高并不显著。我们提出了隐式表示驱动图像重构（IRAD）来克服这些局限性。首先，我们构造一个隐式连续表示，使我们能够在连续坐标空间中表示任何输入图像。第二，我们引入SampleNet，它自动生成像素级移位，以响应不同的输入。此外，我们可以将我们的方法扩展到最先进的基于扩散的方法，以更少的时间步长加速它，同时保留其防御能力。大量实验表明，我们的方法显著增强了不同深度模型对各种攻击的对抗鲁棒性，同时在干净图像上保持高精度。



## **46. Synthesizing Physical Backdoor Datasets: An Automated Framework Leveraging Deep Generative Models**

合成物理后门数据集：利用深度生成模型的自动框架 cs.CR

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2312.03419v3) [paper-pdf](http://arxiv.org/pdf/2312.03419v3)

**Authors**: Sze Jue Yang, Chinh D. La, Quang H. Nguyen, Kok-Seng Wong, Anh Tuan Tran, Chee Seng Chan, Khoa D. Doan

**Abstract**: Backdoor attacks, representing an emerging threat to the integrity of deep neural networks, have garnered significant attention due to their ability to compromise deep learning systems clandestinely. While numerous backdoor attacks occur within the digital realm, their practical implementation in real-world prediction systems remains limited and vulnerable to disturbances in the physical world. Consequently, this limitation has given rise to the development of physical backdoor attacks, where trigger objects manifest as physical entities within the real world. However, creating the requisite dataset to train or evaluate a physical backdoor model is a daunting task, limiting the backdoor researchers and practitioners from studying such physical attack scenarios. This paper unleashes a recipe that empowers backdoor researchers to effortlessly create a malicious, physical backdoor dataset based on advances in generative modeling. Particularly, this recipe involves 3 automatic modules: suggesting the suitable physical triggers, generating the poisoned candidate samples (either by synthesizing new samples or editing existing clean samples), and finally refining for the most plausible ones. As such, it effectively mitigates the perceived complexity associated with creating a physical backdoor dataset, transforming it from a daunting task into an attainable objective. Extensive experiment results show that datasets created by our "recipe" enable adversaries to achieve an impressive attack success rate on real physical world data and exhibit similar properties compared to previous physical backdoor attack studies. This paper offers researchers a valuable toolkit for studies of physical backdoors, all within the confines of their laboratories.

摘要: 后门攻击对深度神经网络的完整性构成了新的威胁，由于它们能够秘密地危害深度学习系统，因此引起了极大的关注。虽然在数字领域内发生了许多后门攻击，但它们在现实世界预测系统中的实际实施仍然有限，容易受到物理世界的干扰。因此，这种限制导致了物理后门攻击的发展，其中触发器对象在真实世界中表现为物理实体。然而，创建必要的数据集来训练或评估物理后门模型是一项艰巨的任务，限制了后门研究人员和实践者研究此类物理攻击场景。这篇文章揭示了一个配方，它使后门研究人员能够基于生成性建模的进步，毫不费力地创建恶意的物理后门数据集。特别是，这个配方涉及三个自动模块：建议合适的物理触发器，生成中毒的候选样本(通过合成新样本或编辑现有的干净样本)，最后提炼出最可信的样本。因此，它有效地减轻了与创建物理后门数据集相关的感知复杂性，将其从令人望而生畏的任务转变为可实现的目标。大量的实验结果表明，由我们的“配方”创建的数据集使攻击者能够在真实的物理世界数据上获得令人印象深刻的攻击成功率，并显示出与之前的物理后门攻击研究类似的特性。这篇论文为研究人员提供了一个宝贵的工具包，用于研究物理后门，所有这些都在他们的实验室范围内。



## **47. Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization**

对抗性净化训练（AToP）：增强鲁棒性和泛化能力 cs.CV

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2401.16352v3) [paper-pdf](http://arxiv.org/pdf/2401.16352v3)

**Authors**: Guang Lin, Chao Li, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: The deep neural networks are known to be vulnerable to well-designed adversarial attacks. The most successful defense technique based on adversarial training (AT) can achieve optimal robustness against particular attacks but cannot generalize well to unseen attacks. Another effective defense technique based on adversarial purification (AP) can enhance generalization but cannot achieve optimal robustness. Meanwhile, both methods share one common limitation on the degraded standard accuracy. To mitigate these issues, we propose a novel pipeline to acquire the robust purifier model, named Adversarial Training on Purification (AToP), which comprises two components: perturbation destruction by random transforms (RT) and purifier model fine-tuned (FT) by adversarial loss. RT is essential to avoid overlearning to known attacks, resulting in the robustness generalization to unseen attacks, and FT is essential for the improvement of robustness. To evaluate our method in an efficient and scalable way, we conduct extensive experiments on CIFAR-10, CIFAR-100, and ImageNette to demonstrate that our method achieves optimal robustness and exhibits generalization ability against unseen attacks.

摘要: 众所周知，深度神经网络容易受到精心设计的对抗攻击。最成功的防御技术基于对抗训练（AT）可以达到针对特定攻击的最佳鲁棒性，但不能很好地推广到不可见的攻击。另一种有效的防御技术是基于对抗纯化（AP），可以增强泛化能力，但不能达到最佳鲁棒性。同时，两种方法都有一个共同的局限性，即标准准确度下降。为了缓解这些问题，我们提出了一种新的管道来获得鲁棒的净化器模型，称为对抗训练净化（AToP），它包括两个部分：随机变换的扰动破坏（RT）和对抗损失的净化器模型微调（FT）。RT对于避免对已知攻击的过度学习，导致鲁棒性推广到未见攻击是必不可少的，而FT对于鲁棒性的提高是必不可少的。为了以一种有效和可扩展的方式评估我们的方法，我们在CIFAR—10，CIFAR—100和ImageNette上进行了大量的实验，以证明我们的方法实现了最佳的鲁棒性，并表现出针对不可见攻击的泛化能力。



## **48. Benchmarking Adversarial Robustness of Image Shadow Removal with Shadow-adaptive Attacks**

基于阴影自适应攻击的图像阴影去除对抗鲁棒性测试 cs.CV

Accepted to ICASSP 2024

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10076v1) [paper-pdf](http://arxiv.org/pdf/2403.10076v1)

**Authors**: Chong Wang, Yi Yu, Lanqing Guo, Bihan Wen

**Abstract**: Shadow removal is a task aimed at erasing regional shadows present in images and reinstating visually pleasing natural scenes with consistent illumination. While recent deep learning techniques have demonstrated impressive performance in image shadow removal, their robustness against adversarial attacks remains largely unexplored. Furthermore, many existing attack frameworks typically allocate a uniform budget for perturbations across the entire input image, which may not be suitable for attacking shadow images. This is primarily due to the unique characteristic of spatially varying illumination within shadow images. In this paper, we propose a novel approach, called shadow-adaptive adversarial attack. Different from standard adversarial attacks, our attack budget is adjusted based on the pixel intensity in different regions of shadow images. Consequently, the optimized adversarial noise in the shadowed regions becomes visually less perceptible while permitting a greater tolerance for perturbations in non-shadow regions. The proposed shadow-adaptive attacks naturally align with the varying illumination distribution in shadow images, resulting in perturbations that are less conspicuous. Building on this, we conduct a comprehensive empirical evaluation of existing shadow removal methods, subjecting them to various levels of attack on publicly available datasets.

摘要: 阴影去除是一项旨在消除图像中存在的区域阴影并恢复具有一致照明的自然场景的任务。虽然最近的深度学习技术在去除图像阴影方面表现出了令人印象深刻的性能，但它们对对抗性攻击的鲁棒性在很大程度上仍未得到探索。此外，许多现有的攻击框架通常为整个输入图像的扰动分配统一的预算，这可能不适合攻击阴影图像。这主要是由于阴影图像内空间变化的照明的独特特性。在本文中，我们提出了一种新的方法，称为阴影自适应对抗攻击。与标准的对抗攻击不同，我们的攻击预算是根据阴影图像不同区域的像素强度来调整的。因此，阴影区域中的优化对抗噪声变得在视觉上更难察觉，同时允许对非阴影区域中的扰动有更大的容忍度。所提出的阴影自适应攻击自然地与阴影图像中的不同照明分布对齐，导致不太明显的扰动。在此基础上，我们对现有的阴影去除方法进行了全面的经验评估，使它们受到对公开可用数据集的各种级别的攻击。



## **49. Revisiting Adversarial Training under Long-Tailed Distributions**

重新审视长尾分配下的对抗性培训 cs.CV

Accepted to CVPR 2024

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10073v1) [paper-pdf](http://arxiv.org/pdf/2403.10073v1)

**Authors**: Xinli Yue, Ningping Mou, Qian Wang, Lingchen Zhao

**Abstract**: Deep neural networks are vulnerable to adversarial attacks, often leading to erroneous outputs. Adversarial training has been recognized as one of the most effective methods to counter such attacks. However, existing adversarial training techniques have predominantly been tested on balanced datasets, whereas real-world data often exhibit a long-tailed distribution, casting doubt on the efficacy of these methods in practical scenarios.   In this paper, we delve into adversarial training under long-tailed distributions. Through an analysis of the previous work "RoBal", we discover that utilizing Balanced Softmax Loss alone can achieve performance comparable to the complete RoBal approach while significantly reducing training overheads. Additionally, we reveal that, similar to uniform distributions, adversarial training under long-tailed distributions also suffers from robust overfitting. To address this, we explore data augmentation as a solution and unexpectedly discover that, unlike results obtained with balanced data, data augmentation not only effectively alleviates robust overfitting but also significantly improves robustness. We further investigate the reasons behind the improvement of robustness through data augmentation and identify that it is attributable to the increased diversity of examples. Extensive experiments further corroborate that data augmentation alone can significantly improve robustness. Finally, building on these findings, we demonstrate that compared to RoBal, the combination of BSL and data augmentation leads to a +6.66% improvement in model robustness under AutoAttack on CIFAR-10-LT. Our code is available at https://github.com/NISPLab/AT-BSL .

摘要: 深度神经网络容易受到对抗攻击，经常导致错误输出。对抗性训练被认为是对付此类攻击的最有效方法之一。然而，现有的对抗训练技术主要是在平衡数据集上进行测试，而现实世界的数据往往呈现长尾分布，这让人怀疑这些方法在实际场景中的有效性。   在本文中，我们深入研究长尾分布下的对抗训练。通过对之前的工作“RoBal”的分析，我们发现，单独使用Balancial Softmax Loss可以实现与完整的RoBal方法相当的性能，同时显著降低培训费用。此外，我们发现，类似于均匀分布，长尾分布下的对抗训练也遭受鲁棒过拟合。为了解决这个问题，我们探索了数据增强作为一种解决方案，并意外地发现，与平衡数据获得的结果不同，数据增强不仅有效地增强了鲁棒的过拟合，而且显著提高了鲁棒性。我们进一步调查了通过数据增强来提高鲁棒性的原因，并确定这是由于增加了示例的多样性。大量的实验进一步证实了数据增强本身可以显著提高鲁棒性。最后，在这些发现的基础上，我们证明了与RoBal相比，BSL和数据增强的组合在CIFAR—10—LT上的AutoAttack下使模型鲁棒性提高了6.66%。



## **50. Towards Adversarially Robust Dataset Distillation by Curvature Regularization**

基于曲率正则化的对抗鲁棒数据集提取 cs.LG

17 pages, 3 figures

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10045v1) [paper-pdf](http://arxiv.org/pdf/2403.10045v1)

**Authors**: Eric Xue, Yijiang Li, Haoyang Liu, Yifan Shen, Haohan Wang

**Abstract**: Dataset distillation (DD) allows datasets to be distilled to fractions of their original size while preserving the rich distributional information so that models trained on the distilled datasets can achieve a comparable accuracy while saving significant computational loads. Recent research in this area has been focusing on improving the accuracy of models trained on distilled datasets. In this paper, we aim to explore a new perspective of DD. We study how to embed adversarial robustness in distilled datasets, so that models trained on these datasets maintain the high accuracy and meanwhile acquire better adversarial robustness. We propose a new method that achieves this goal by incorporating curvature regularization into the distillation process with much less computational overhead than standard adversarial training. Extensive empirical experiments suggest that our method not only outperforms standard adversarial training on both accuracy and robustness with less computation overhead but is also capable of generating robust distilled datasets that can withstand various adversarial attacks.

摘要: 数据集蒸馏（DD）允许数据集蒸馏到原始大小的分数，同时保留丰富的分布信息，以便在蒸馏数据集上训练的模型可以达到相当的精度，同时节省大量的计算负载。该领域最近的研究一直专注于提高在蒸馏数据集上训练的模型的准确性。本文旨在探讨DD的新视角。研究了如何在提取数据集中嵌入对抗鲁棒性，使在提取数据集上训练的模型在保持较高的精度的同时，获得更好的对抗鲁棒性。我们提出了一种新的方法，通过将曲率正则化到蒸馏过程中，以比标准对抗训练更少的计算开销来实现这一目标。大量的经验实验表明，我们的方法不仅在准确性和鲁棒性上优于标准的对抗训练，计算开销更少，而且还能够生成强大的提取数据集，可以承受各种对抗攻击。



