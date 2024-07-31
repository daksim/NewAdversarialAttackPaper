# Latest Adversarial Attack Papers
**update at 2024-07-31 16:20:58**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

人工智能生成图像检测中的漏洞：对抗性攻击的挑战 cs.CV

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.20836v1) [paper-pdf](http://arxiv.org/pdf/2407.20836v1)

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of these AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. For the task of AIGI detection, we propose a new attack containing two main parts. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous models, e.g. transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as frequency-based post-train Bayesian attack, or FPBA. Through FPBA, we show that adversarial attack is truly a real threat to AIGI detectors, because FPBA can deliver successful black-box attacks across models, generators, defense methods, and even evade cross-generator detection, which is a crucial real-world detection scenario.

摘要: 最近在图像合成方面的进步，特别是随着GaN和扩散模型的出现，放大了公众对虚假信息传播的担忧。为了解决这些问题，人们已经提出了许多人工智能生成的图像(AIGI)检测器，并在识别虚假图像方面取得了良好的性能。然而，对这些AIGI检测器的对抗健壮性仍然缺乏系统的了解。本文研究了白盒和黑盒环境下最新的AIGI检测器抵抗敌意攻击的脆弱性，这是迄今为止很少被研究的。针对AIGI检测任务，我们提出了一种包含两个主要部分的新攻击。首先，受真伪图像在频域存在明显差异的启发，在频域下加入扰动，使图像偏离其原有的频率分布。其次，我们探索了代理模型的完全后验分布，以进一步缩小不同模型之间的差距，例如跨CNN和VITS传输对抗性实例。这是通过引入一种新颖的后训练贝叶斯策略来实现的，该策略将单个代理转变为贝叶斯策略，能够使用一个预先训练的代理来模拟不同的受害者模型，而不需要重新训练。我们将我们的方法命名为基于频率的训练后贝叶斯攻击，或称FPBA。通过FPBA，我们证明了敌意攻击是对AIGI检测器的真正威胁，因为FPBA可以跨模型、生成器、防御方法提供成功的黑盒攻击，甚至可以逃避交叉生成器检测，这是现实世界中的一个关键检测场景。



## **2. Attacking Cooperative Multi-Agent Reinforcement Learning by Adversarial Minority Influence**

利用敌对少数派影响攻击合作多智能体强化学习 cs.LG

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2302.03322v3) [paper-pdf](http://arxiv.org/pdf/2302.03322v3)

**Authors**: Simin Li, Jun Guo, Jingqiao Xiu, Yuwei Zheng, Pu Feng, Xin Yu, Aishan Liu, Yaodong Yang, Bo An, Wenjun Wu, Xianglong Liu

**Abstract**: This study probes the vulnerabilities of cooperative multi-agent reinforcement learning (c-MARL) under adversarial attacks, a critical determinant of c-MARL's worst-case performance prior to real-world implementation. Current observation-based attacks, constrained by white-box assumptions, overlook c-MARL's complex multi-agent interactions and cooperative objectives, resulting in impractical and limited attack capabilities. To address these shortcomes, we propose Adversarial Minority Influence (AMI), a practical and strong for c-MARL. AMI is a practical black-box attack and can be launched without knowing victim parameters. AMI is also strong by considering the complex multi-agent interaction and the cooperative goal of agents, enabling a single adversarial agent to unilaterally misleads majority victims to form targeted worst-case cooperation. This mirrors minority influence phenomena in social psychology. To achieve maximum deviation in victim policies under complex agent-wise interactions, our unilateral attack aims to characterize and maximize the impact of the adversary on the victims. This is achieved by adapting a unilateral agent-wise relation metric derived from mutual information, thereby mitigating the adverse effects of victim influence on the adversary. To lead the victims into a jointly detrimental scenario, our targeted attack deceives victims into a long-term, cooperatively harmful situation by guiding each victim towards a specific target, determined through a trial-and-error process executed by a reinforcement learning agent. Through AMI, we achieve the first successful attack against real-world robot swarms and effectively fool agents in simulated environments into collectively worst-case scenarios, including Starcraft II and Multi-agent Mujoco. The source code and demonstrations can be found at: https://github.com/DIG-Beihang/AMI.

摘要: 该研究探讨了协作多智能体强化学习(c-Marl)在对抗攻击下的脆弱性，这是c-Marl在现实世界实现之前最差情况性能的关键决定因素。目前的基于观测的攻击受白盒假设的约束，忽略了c-Marl复杂的多智能体交互和合作目标，导致攻击能力不切实际和有限。针对这些不足，我们提出了一种实用而强大的c-Marl算法--对抗性少数影响算法。AMI是一种实用的黑盒攻击，可以在不知道受害者参数的情况下启动。通过考虑复杂的多智能体相互作用和智能体的合作目标，使单一对抗智能体能够单方面误导大多数受害者形成有针对性的最坏情况合作，AMI也很强大。这反映了社会心理学中的小众影响现象。为了在复杂的智能体相互作用下实现受害者政策的最大偏差，我们的单边攻击旨在刻画和最大化对手对受害者的影响。这是通过采用来自互信息的单边代理关系度量来实现的，从而减轻了受害者影响对对手的不利影响。为了将受害者引导到共同有害的情景中，我们的有针对性的攻击通过引导每个受害者指向特定的目标，将受害者欺骗到长期的、合作有害的情况中，该特定目标是通过由强化学习代理执行的反复试验过程确定的。通过AMI，我们实现了对真实世界机器人群的第一次成功攻击，并有效地将模拟环境中的代理愚弄到了集体最坏的情况下，包括星际争霸II和多代理Mujoco。源代码和演示可在以下网址找到：https://github.com/DIG-Beihang/AMI.



## **3. Prompt-Driven Contrastive Learning for Transferable Adversarial Attacks**

可转移对抗攻击的预算驱动对比学习 cs.CV

Accepted to ECCV 2024, Project Page: https://PDCL-Attack.github.io

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.20657v1) [paper-pdf](http://arxiv.org/pdf/2407.20657v1)

**Authors**: Hunmin Yang, Jongoh Jeong, Kuk-Jin Yoon

**Abstract**: Recent vision-language foundation models, such as CLIP, have demonstrated superior capabilities in learning representations that can be transferable across diverse range of downstream tasks and domains. With the emergence of such powerful models, it has become crucial to effectively leverage their capabilities in tackling challenging vision tasks. On the other hand, only a few works have focused on devising adversarial examples that transfer well to both unknown domains and model architectures. In this paper, we propose a novel transfer attack method called PDCL-Attack, which leverages the CLIP model to enhance the transferability of adversarial perturbations generated by a generative model-based attack framework. Specifically, we formulate an effective prompt-driven feature guidance by harnessing the semantic representation power of text, particularly from the ground-truth class labels of input images. To the best of our knowledge, we are the first to introduce prompt learning to enhance the transferable generative attacks. Extensive experiments conducted across various cross-domain and cross-model settings empirically validate our approach, demonstrating its superiority over state-of-the-art methods.

摘要: 最近的视觉语言基础模型，如CLIP，在学习表征方面表现出了优越的能力，这些表征可以跨不同范围的下游任务和领域转移。随着如此强大的模型的出现，有效地利用它们的能力来处理具有挑战性的愿景任务变得至关重要。另一方面，只有少数工作专注于设计能够很好地移植到未知领域和模型体系结构的对抗性例子。本文提出了一种新的转移攻击方法PDCL-Attack，该方法利用CLIP模型来增强基于产生式模型的攻击框架产生的敌意扰动的可转移性。具体地说，我们通过利用文本的语义表征能力，特别是从输入图像的基本事实类标签来制定有效的提示驱动的特征指导。据我们所知，我们是第一个引入快速学习来增强可转移的生成性攻击的。在各种跨域和跨模型环境下进行的广泛实验验证了我们的方法，证明了它比最先进的方法更优越。



## **4. FACL-Attack: Frequency-Aware Contrastive Learning for Transferable Adversarial Attacks**

FACL攻击：可转移对抗攻击的频率感知对比学习 cs.CV

Accepted to AAAI 2024, Project Page: https://FACL-Attack.github.io

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.20653v1) [paper-pdf](http://arxiv.org/pdf/2407.20653v1)

**Authors**: Hunmin Yang, Jongoh Jeong, Kuk-Jin Yoon

**Abstract**: Deep neural networks are known to be vulnerable to security risks due to the inherent transferable nature of adversarial examples. Despite the success of recent generative model-based attacks demonstrating strong transferability, it still remains a challenge to design an efficient attack strategy in a real-world strict black-box setting, where both the target domain and model architectures are unknown. In this paper, we seek to explore a feature contrastive approach in the frequency domain to generate adversarial examples that are robust in both cross-domain and cross-model settings. With that goal in mind, we propose two modules that are only employed during the training phase: a Frequency-Aware Domain Randomization (FADR) module to randomize domain-variant low- and high-range frequency components and a Frequency-Augmented Contrastive Learning (FACL) module to effectively separate domain-invariant mid-frequency features of clean and perturbed image. We demonstrate strong transferability of our generated adversarial perturbations through extensive cross-domain and cross-model experiments, while keeping the inference time complexity.

摘要: 众所周知，由于对抗性例子的固有可转移性，深度神经网络容易受到安全风险的影响。尽管最近基于模型的生成性攻击取得了成功，表现出很强的可转移性，但在目标域和模型体系结构都未知的现实世界严格的黑盒环境中，设计有效的攻击策略仍然是一个挑战。在本文中，我们试图探索一种在频域中进行特征对比的方法，以生成在跨域和跨模型设置下都具有健壮性的对抗性示例。考虑到这一目标，我们提出了两个仅在训练阶段使用的模块：频率感知域随机化(FADR)模块，用于随机化域变化的低频和高频分量；以及频率增强对比学习(FACL)模块，用于有效分离干净和扰动图像的域不变中频特征。通过广泛的跨域和跨模型实验，我们证明了我们生成的对抗性扰动具有很强的可转移性，同时保持了推理的时间复杂性。



## **5. Robust Federated Learning for Wireless Networks: A Demonstration with Channel Estimation**

无线网络的鲁棒联邦学习：信道估计的演示 cs.LG

Submitted to IEEE GLOBECOM 2024

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2404.03088v2) [paper-pdf](http://arxiv.org/pdf/2404.03088v2)

**Authors**: Zexin Fang, Bin Han, Hans D. Schotten

**Abstract**: Federated learning (FL) offers a privacy-preserving collaborative approach for training models in wireless networks, with channel estimation emerging as a promising application. Despite extensive studies on FL-empowered channel estimation, the security concerns associated with FL require meticulous attention. In a scenario where small base stations (SBSs) serve as local models trained on cached data, and a macro base station (MBS) functions as the global model setting, an attacker can exploit the vulnerability of FL, launching attacks with various adversarial attacks or deployment tactics. In this paper, we analyze such vulnerabilities, corresponding solutions were brought forth, and validated through simulation.

摘要: 联合学习（FL）为无线网络中的训练模型提供了一种保护隐私的协作方法，而信道估计正在成为一个有前途的应用。尽管对FL授权的信道估计进行了广泛的研究，但与FL相关的安全问题需要仔细关注。在小型基站（SBS）充当在缓存数据上训练的本地模型，宏基站（MBS）充当全局模型设置的场景中，攻击者可以利用FL的漏洞，通过各种对抗性攻击或部署策略发起攻击。本文对此类漏洞进行了分析，提出了相应的解决方案，并通过仿真进行了验证。



## **6. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

从ML到LLM：评估网络钓鱼网页检测模型对抗对抗攻击的稳健性 cs.CR

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.20361v1) [paper-pdf](http://arxiv.org/pdf/2407.20361v1)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of two existing models, Stack model and Phishpedia, in classifying PhishOracle-generated adversarial phishing webpages. Additionally, we study a commercial large language model, Gemini Pro Vision, in the context of adversarial attacks. We conduct a user study to determine whether PhishOracle-generated adversarial phishing webpages deceive users. Our findings reveal that many PhishOracle-generated phishing webpages evade current phishing webpage detection models and deceive users, but Gemini Pro Vision is robust to the attack. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources are publicly available on GitHub.

摘要: 网络钓鱼攻击试图欺骗用户窃取敏感信息，构成重大的网络安全威胁。机器学习(ML)和深度学习(DL)的进步导致了许多钓鱼网页检测解决方案的发展，但这些模型仍然容易受到对手攻击。评估它们对敌意网络钓鱼网页的健壮性是至关重要的。现有工具包含为有限数量的品牌预先设计的钓鱼网页的数据集，并且在钓鱼功能方面缺乏多样性。为了应对这些挑战，我们开发了PhishOracle，这是一个通过在合法网页中嵌入不同的钓鱼功能来生成敌意钓鱼网页的工具。我们评估了现有的两种模型Stack模型和Phishpedia模型对PhishOracle生成的敌意钓鱼网页进行分类的稳健性。此外，我们研究了一个商业大型语言模型，Gemini Pro Vision，在对抗性攻击的背景下。我们进行了一项用户研究，以确定PhishOracle生成的敌意钓鱼网页是否欺骗了用户。我们的研究结果显示，许多PhishOracle生成的钓鱼网页逃避了当前的钓鱼网页检测模型并欺骗用户，但Gemini Pro Vision对攻击具有健壮性。我们还开发了PhishOracle Web应用程序，允许用户输入合法的URL，选择相关的网络钓鱼功能并生成相应的网络钓鱼网页。所有资源都在GitHub上公开提供。



## **7. Prompt Leakage effect and defense strategies for multi-turn LLM interactions**

多圈LLM相互作用的即时泄漏效应和防御策略 cs.CR

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2404.16251v3) [paper-pdf](http://arxiv.org/pdf/2404.16251v3)

**Authors**: Divyansh Agarwal, Alexander R. Fabbri, Ben Risher, Philippe Laban, Shafiq Joty, Chien-Sheng Wu

**Abstract**: Prompt leakage poses a compelling security and privacy threat in LLM applications. Leakage of system prompts may compromise intellectual property, and act as adversarial reconnaissance for an attacker. A systematic evaluation of prompt leakage threats and mitigation strategies is lacking, especially for multi-turn LLM interactions. In this paper, we systematically investigate LLM vulnerabilities against prompt leakage for 10 closed- and open-source LLMs, across four domains. We design a unique threat model which leverages the LLM sycophancy effect and elevates the average attack success rate (ASR) from 17.7% to 86.2% in a multi-turn setting. Our standardized setup further allows dissecting leakage of specific prompt contents such as task instructions and knowledge documents. We measure the mitigation effect of 7 black-box defense strategies, along with finetuning an open-source model to defend against leakage attempts. We present different combination of defenses against our threat model, including a cost analysis. Our study highlights key takeaways for building secure LLM applications and provides directions for research in multi-turn LLM interactions

摘要: 即时泄漏在LLM应用程序中构成了令人信服的安全和隐私威胁。泄露系统提示可能会危及知识产权，并充当攻击者的对抗性侦察。缺乏对即时泄漏威胁和缓解策略的系统评估，特别是对于多回合的LLM相互作用。在这篇文章中，我们系统地研究了10个封闭和开放源代码的LLM在四个域中针对即时泄漏的LLM漏洞。我们设计了一种独特的威胁模型，该模型利用了LLM的奉承效应，在多回合环境下将平均攻击成功率从17.7%提高到86.2%。我们的标准化设置还允许剖析特定提示内容的泄漏，如任务说明和知识文档。我们测量了7种黑盒防御策略的缓解效果，并微调了一个开源模型来防御泄漏企图。针对我们的威胁模型，我们提出了不同的防御组合，包括成本分析。我们的研究强调了构建安全的LLM应用程序的关键要点，并为多轮LLM交互的研究提供了方向



## **8. DDAP: Dual-Domain Anti-Personalization against Text-to-Image Diffusion Models**

DDAP：针对文本到图像扩散模型的双域反个性化 cs.CV

Accepted by IJCB 2024

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.20141v1) [paper-pdf](http://arxiv.org/pdf/2407.20141v1)

**Authors**: Jing Yang, Runping Xi, Yingxin Lai, Xun Lin, Zitong Yu

**Abstract**: Diffusion-based personalized visual content generation technologies have achieved significant breakthroughs, allowing for the creation of specific objects by just learning from a few reference photos. However, when misused to fabricate fake news or unsettling content targeting individuals, these technologies could cause considerable societal harm. To address this problem, current methods generate adversarial samples by adversarially maximizing the training loss, thereby disrupting the output of any personalized generation model trained with these samples. However, the existing methods fail to achieve effective defense and maintain stealthiness, as they overlook the intrinsic properties of diffusion models. In this paper, we introduce a novel Dual-Domain Anti-Personalization framework (DDAP). Specifically, we have developed Spatial Perturbation Learning (SPL) by exploiting the fixed and perturbation-sensitive nature of the image encoder in personalized generation. Subsequently, we have designed a Frequency Perturbation Learning (FPL) method that utilizes the characteristics of diffusion models in the frequency domain. The SPL disrupts the overall texture of the generated images, while the FPL focuses on image details. By alternating between these two methods, we construct the DDAP framework, effectively harnessing the strengths of both domains. To further enhance the visual quality of the adversarial samples, we design a localization module to accurately capture attentive areas while ensuring the effectiveness of the attack and avoiding unnecessary disturbances in the background. Extensive experiments on facial benchmarks have shown that the proposed DDAP enhances the disruption of personalized generation models while also maintaining high quality in adversarial samples, making it more effective in protecting privacy in practical applications.

摘要: 基于扩散的个性化视觉内容生成技术取得了重大突破，只需从几张参考照片中学习，就可以创建特定的对象。然而，当这些技术被滥用来编造假新闻或针对个人的令人不安的内容时，可能会造成相当大的社会危害。为了解决这个问题，当前的方法通过对抗性地最大化训练损失来生成对抗性样本，从而扰乱用这些样本训练的任何个性化生成模型的输出。然而，现有的方法忽略了扩散模型的内在特性，无法实现有效的防御和保持隐蔽性。提出了一种新型的双域反个性化框架(DDAP)。具体地说，我们通过在个性化生成中利用图像编码器的固定和对扰动敏感的性质来发展空间扰动学习(SPL)。随后，我们设计了一种利用频域扩散模型特性的频率微扰学习(FPL)方法。SPL破坏了生成图像的整体纹理，而FPL则专注于图像细节。通过在这两种方法之间交替使用，我们构建了DDAP框架，有效地利用了这两个领域的优势。为了进一步提高对手样本的视觉质量，我们设计了定位模块，在保证攻击有效性的同时，准确地捕捉到关注区域，避免了背景中不必要的干扰。在人脸基准上的广泛实验表明，所提出的DDAP增强了个性化生成模型的颠覆性，同时保持了对手样本的高质量，使其在实际应用中更有效地保护隐私。



## **9. Adversarial Robustness in RGB-Skeleton Action Recognition: Leveraging Attention Modality Reweighter**

RGB骨架动作识别中的对抗鲁棒性：利用注意力情态重新加权 cs.CV

Accepted by IJCB 2024

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.19981v1) [paper-pdf](http://arxiv.org/pdf/2407.19981v1)

**Authors**: Chao Liu, Xin Liu, Zitong Yu, Yonghong Hou, Huanjing Yue, Jingyu Yang

**Abstract**: Deep neural networks (DNNs) have been applied in many computer vision tasks and achieved state-of-the-art (SOTA) performance. However, misclassification will occur when DNNs predict adversarial examples which are created by adding human-imperceptible adversarial noise to natural examples. This limits the application of DNN in security-critical fields. In order to enhance the robustness of models, previous research has primarily focused on the unimodal domain, such as image recognition and video understanding. Although multi-modal learning has achieved advanced performance in various tasks, such as action recognition, research on the robustness of RGB-skeleton action recognition models is scarce. In this paper, we systematically investigate how to improve the robustness of RGB-skeleton action recognition models. We initially conducted empirical analysis on the robustness of different modalities and observed that the skeleton modality is more robust than the RGB modality. Motivated by this observation, we propose the \formatword{A}ttention-based \formatword{M}odality \formatword{R}eweighter (\formatword{AMR}), which utilizes an attention layer to re-weight the two modalities, enabling the model to learn more robust features. Our AMR is plug-and-play, allowing easy integration with multimodal models. To demonstrate the effectiveness of AMR, we conducted extensive experiments on various datasets. For example, compared to the SOTA methods, AMR exhibits a 43.77\% improvement against PGD20 attacks on the NTU-RGB+D 60 dataset. Furthermore, it effectively balances the differences in robustness between different modalities.

摘要: 深度神经网络(DNN)已被应用于许多计算机视觉任务中，并取得了最先进的性能(SOTA)。然而，当DNN预测通过在自然实例中添加人类不可察觉的对抗性噪声而产生的对抗性实例时，就会发生误分类。这限制了DNN在安全关键领域的应用。为了增强模型的稳健性，以往的研究主要集中在单峰领域，如图像识别和视频理解。尽管多通道学习在动作识别等各种任务中取得了较好的性能，但对RGB骨架动作识别模型的稳健性研究还很少。本文系统地研究了如何提高RGB-骨架动作识别模型的健壮性。我们最初对不同通道的稳健性进行了实证分析，观察到骨架通道比RGB通道更健壮。基于这一观察结果，我们提出了基于格式词{A}时延的格式词{M}奇异性\格式词{R}权重器(\Formatword{AMR})，它利用关注层对两个通道进行重新加权，使模型能够学习更健壮的特征。我们的AMR是即插即用的，允许与多模式模型轻松集成。为了验证AMR的有效性，我们在不同的数据集上进行了广泛的实验。例如，与SOTA方法相比，AMR在NTU-RGB+D60数据集上对PGD20攻击的性能提高了43.77%。此外，它有效地平衡了不同模式之间在稳健性方面的差异。



## **10. Quasi-Framelets: Robust Graph Neural Networks via Adaptive Framelet Convolution**

准框架集：通过自适应框架集卷积的鲁棒图神经网络 cs.LG

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2201.04728v2) [paper-pdf](http://arxiv.org/pdf/2201.04728v2)

**Authors**: Mengxi Yang, Dai Shi, Xuebin Zheng, Jie Yin, Junbin Gao

**Abstract**: This paper aims to provide a novel design of a multiscale framelet convolution for spectral graph neural networks (GNNs). While current spectral methods excel in various graph learning tasks, they often lack the flexibility to adapt to noisy, incomplete, or perturbed graph signals, making them fragile in such conditions. Our newly proposed framelet convolution addresses these limitations by decomposing graph data into low-pass and high-pass spectra through a finely-tuned multiscale approach. Our approach directly designs filtering functions within the spectral domain, allowing for precise control over the spectral components. The proposed design excels in filtering out unwanted spectral information and significantly reduces the adverse effects of noisy graph signals. Our approach not only enhances the robustness of GNNs but also preserves crucial graph features and structures. Through extensive experiments on diverse, real-world graph datasets, we demonstrate that our framelet convolution achieves superior performance in node classification tasks. It exhibits remarkable resilience to noisy data and adversarial attacks, highlighting its potential as a robust solution for real-world graph applications. This advancement opens new avenues for more adaptive and reliable spectral GNN architectures.

摘要: 本文旨在为谱图神经网络提供一种新的多尺度框架卷积设计。虽然目前的谱方法在各种图形学习任务中表现出色，但它们往往缺乏适应噪声、不完整或扰动的图形信号的灵活性，这使得它们在这些条件下很脆弱。我们最新提出的框架卷积通过精细调整的多尺度方法将图形数据分解成低通和高通光谱，从而解决了这些限制。我们的方法直接在谱域内设计滤波函数，允许对谱分量进行精确控制。所提出的设计在滤除不需要的光谱信息方面表现出色，并显著降低了噪声图形信号的不利影响。我们的方法不仅增强了GNN的健壮性，而且保留了关键的图特征和结构。通过在不同的真实图形数据集上的广泛实验，我们证明了我们的框架集卷积在节点分类任务中取得了优异的性能。它表现出了对噪声数据和对手攻击的非凡弹性，突出了它作为现实世界图形应用程序的健壮解决方案的潜力。这一进展为更适应和更可靠的频谱GNN体系结构开辟了新的途径。



## **11. Detecting and Understanding Vulnerabilities in Language Models via Mechanistic Interpretability**

通过机械解释性检测和理解语言模型中的漏洞 cs.LG

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.19842v1) [paper-pdf](http://arxiv.org/pdf/2407.19842v1)

**Authors**: Jorge García-Carrasco, Alejandro Maté, Juan Trujillo

**Abstract**: Large Language Models (LLMs), characterized by being trained on broad amounts of data in a self-supervised manner, have shown impressive performance across a wide range of tasks. Indeed, their generative abilities have aroused interest on the application of LLMs across a wide range of contexts. However, neural networks in general, and LLMs in particular, are known to be vulnerable to adversarial attacks, where an imperceptible change to the input can mislead the output of the model. This is a serious concern that impedes the use of LLMs on high-stakes applications, such as healthcare, where a wrong prediction can imply serious consequences. Even though there are many efforts on making LLMs more robust to adversarial attacks, there are almost no works that study \emph{how} and \emph{where} these vulnerabilities that make LLMs prone to adversarial attacks happen. Motivated by these facts, we explore how to localize and understand vulnerabilities, and propose a method, based on Mechanistic Interpretability (MI) techniques, to guide this process. Specifically, this method enables us to detect vulnerabilities related to a concrete task by (i) obtaining the subset of the model that is responsible for that task, (ii) generating adversarial samples for that task, and (iii) using MI techniques together with the previous samples to discover and understand the possible vulnerabilities. We showcase our method on a pretrained GPT-2 Small model carrying out the task of predicting 3-letter acronyms to demonstrate its effectiveness on locating and understanding concrete vulnerabilities of the model.

摘要: 大型语言模型(LLM)的特点是以自我监督的方式对大量数据进行培训，在各种任务中表现出令人印象深刻的表现。事实上，他们的生成能力已经引起了人们对LLMS在广泛背景下的应用的兴趣。然而，一般的神经网络，特别是LLM，都很容易受到敌意攻击，在这种攻击中，输入的不可察觉的变化可能会误导模型的输出。这是一个严重的担忧，阻碍了低成本管理在高风险应用中的使用，例如医疗保健，在这些应用中，错误的预测可能意味着严重的后果。尽管已经有许多努力使LLM对对手攻击更健壮，但几乎没有研究这些使LLM容易受到对手攻击的漏洞的著作。在这些事实的推动下，我们探索了如何定位和理解漏洞，并提出了一种基于机械可解释性(MI)技术的方法来指导这一过程。具体地说，这种方法使我们能够通过(I)获得负责该任务的模型的子集，(Ii)为该任务生成对抗性样本，以及(Iii)结合使用MI技术和先前的样本来发现和理解可能的漏洞来检测与该具体任务相关的漏洞。我们在一个预先训练的GPT-2小模型上展示了我们的方法，该模型执行了预测3字母首字母缩写的任务，以演示其在定位和理解模型的具体漏洞方面的有效性。



## **12. Understanding Robust Overfitting from the Feature Generalization Perspective**

从特征概括的角度理解稳健过拟 cs.LG

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2310.00607v2) [paper-pdf](http://arxiv.org/pdf/2310.00607v2)

**Authors**: Chaojian Yu, Xiaolong Shi, Jun Yu, Bo Han, Tongliang Liu

**Abstract**: Adversarial training (AT) constructs robust neural networks by incorporating adversarial perturbations into natural data. However, it is plagued by the issue of robust overfitting (RO), which severely damages the model's robustness. In this paper, we investigate RO from a novel feature generalization perspective. Specifically, we design factor ablation experiments to assess the respective impacts of natural data and adversarial perturbations on RO, identifying that the inducing factor of RO stems from natural data. Given that the only difference between adversarial and natural training lies in the inclusion of adversarial perturbations, we further hypothesize that adversarial perturbations degrade the generalization of features in natural data and verify this hypothesis through extensive experiments. Based on these findings, we provide a holistic view of RO from the feature generalization perspective and explain various empirical behaviors associated with RO. To examine our feature generalization perspective, we devise two representative methods, attack strength and data augmentation, to prevent the feature generalization degradation during AT. Extensive experiments conducted on benchmark datasets demonstrate that the proposed methods can effectively mitigate RO and enhance adversarial robustness.

摘要: 对抗性训练(AT)通过在自然数据中加入对抗性扰动来构建稳健的神经网络。然而，稳健过拟合(RO)问题严重影响了模型的稳健性。在本文中，我们从一种新的特征泛化的角度来研究RO。具体地说，我们设计了因子消融实验来评估自然数据和对抗性扰动对RO的影响，确定了RO的诱导因素源于自然数据。鉴于对抗性训练和自然训练之间的唯一区别在于包含对抗性扰动，我们进一步假设对抗性扰动降低了自然数据中特征的泛化，并通过广泛的实验验证了这一假设。基于这些发现，我们从特征泛化的角度提供了关于反语的整体观点，并解释了与反语相关的各种经验行为。为了检验我们的特征泛化观点，我们设计了两种有代表性的方法，攻击强度和数据增强，以防止特征泛化在AT过程中的退化。在基准数据集上进行的大量实验表明，所提出的方法可以有效地缓解RO，增强对手的健壮性。



## **13. Exploring the Adversarial Robustness of CLIP for AI-generated Image Detection**

探索CLIP用于人工智能生成图像检测的对抗鲁棒性 cs.CV

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2407.19553v1) [paper-pdf](http://arxiv.org/pdf/2407.19553v1)

**Authors**: Vincenzo De Rosa, Fabrizio Guillaro, Giovanni Poggi, Davide Cozzolino, Luisa Verdoliva

**Abstract**: In recent years, many forensic detectors have been proposed to detect AI-generated images and prevent their use for malicious purposes. Convolutional neural networks (CNNs) have long been the dominant architecture in this field and have been the subject of intense study. However, recently proposed Transformer-based detectors have been shown to match or even outperform CNN-based detectors, especially in terms of generalization. In this paper, we study the adversarial robustness of AI-generated image detectors, focusing on Contrastive Language-Image Pretraining (CLIP)-based methods that rely on Visual Transformer backbones and comparing their performance with CNN-based methods. We study the robustness to different adversarial attacks under a variety of conditions and analyze both numerical results and frequency-domain patterns. CLIP-based detectors are found to be vulnerable to white-box attacks just like CNN-based detectors. However, attacks do not easily transfer between CNN-based and CLIP-based methods. This is also confirmed by the different distribution of the adversarial noise patterns in the frequency domain. Overall, this analysis provides new insights into the properties of forensic detectors that can help to develop more effective strategies.

摘要: 近年来，已经提出了许多法医检测器来检测人工智能生成的图像，并防止将其用于恶意目的。卷积神经网络(CNN)长期以来一直是这一领域的主导结构，也一直是研究的热点。然而，最近提出的基于变形金刚的检测器已经被证明与基于CNN的检测器相媲美，甚至优于基于CNN的检测器，特别是在泛化方面。在本文中，我们研究了人工智能生成的图像检测器的对抗鲁棒性，重点研究了基于对比语言图像预训练(CLIP)的依赖于视觉变形金刚的方法，并将它们的性能与基于CNN的方法进行了比较。我们研究了在不同条件下对不同敌意攻击的鲁棒性，并对数值结果和频域模式进行了分析。基于剪辑的检测器被发现像基于CNN的检测器一样容易受到白盒攻击。然而，攻击并不容易在基于CNN的方法和基于剪辑的方法之间转移。对抗性噪声模式在频域中的不同分布也证实了这一点。总体而言，这一分析为法医探测器的特性提供了新的见解，有助于制定更有效的战略。



## **14. Learning on Graphs with Large Language Models(LLMs): A Deep Dive into Model Robustness**

使用大型语言模型（LLM）学习图形：深入研究模型稳健性 cs.LG

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2407.12068v2) [paper-pdf](http://arxiv.org/pdf/2407.12068v2)

**Authors**: Kai Guo, Zewen Liu, Zhikai Chen, Hongzhi Wen, Wei Jin, Jiliang Tang, Yi Chang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing tasks. Recently, several LLMs-based pipelines have been developed to enhance learning on graphs with text attributes, showcasing promising performance. However, graphs are well-known to be susceptible to adversarial attacks and it remains unclear whether LLMs exhibit robustness in learning on graphs. To address this gap, our work aims to explore the potential of LLMs in the context of adversarial attacks on graphs. Specifically, we investigate the robustness against graph structural and textual perturbations in terms of two dimensions: LLMs-as-Enhancers and LLMs-as-Predictors. Through extensive experiments, we find that, compared to shallow models, both LLMs-as-Enhancers and LLMs-as-Predictors offer superior robustness against structural and textual attacks.Based on these findings, we carried out additional analyses to investigate the underlying causes. Furthermore, we have made our benchmark library openly available to facilitate quick and fair evaluations, and to encourage ongoing innovative research in this field.

摘要: 大型语言模型(LLM)在各种自然语言处理任务中表现出了显著的性能。最近，已经开发了几个基于LLMS的管道来增强对具有文本属性的图形的学习，展示了良好的性能。然而，众所周知，图是容易受到敌意攻击的，而且目前还不清楚LLM是否表现出关于图的学习的健壮性。为了解决这一差距，我们的工作旨在探索LLMS在对抗性攻击图的背景下的潜力。具体地说，我们从两个维度考察了对图结构和文本扰动的稳健性：作为增强器的LLMS和作为预测者的LLMS。通过广泛的实验，我们发现，与浅层模型相比，LLMS-as-Enhancer和LLMS-as-Predicator对结构和文本攻击都具有更好的稳健性。基于这些发现，我们进行了额外的分析，以探讨潜在的原因。此外，我们开放了我们的基准图书馆，以促进快速和公平的评估，并鼓励这一领域正在进行的创新研究。



## **15. Breaking the Balance of Power: Commitment Attacks on Ethereum's Reward Mechanism**

打破权力平衡：承诺攻击以太坊奖励机制 cs.CR

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2407.19479v1) [paper-pdf](http://arxiv.org/pdf/2407.19479v1)

**Authors**: Roozbeh Sarenche, Ertem Nusret Tas, Barnabe Monnot, Caspar Schwarz-Schilling, Bart Preneel

**Abstract**: Validators in permissionless, large-scale blockchains (e.g., Ethereum) are typically payoff-maximizing, rational actors. Ethereum relies on in-protocol incentives, like rewards for validators delivering correct and timely votes, to induce honest behavior and secure the blockchain. However, external incentives, such as the block proposer's opportunity to capture maximal extractable value (MEV), may tempt validators to deviate from honest protocol participation.   We show a series of commitment attacks on LMD GHOST, a core part of Ethereum's consensus mechanism. We demonstrate how a single adversarial block proposer can orchestrate long-range chain reorganizations by manipulating Ethereum's reward system for timely votes. These attacks disrupt the intended balance of power between proposers and voters: by leveraging credible threats, the adversarial proposer can coerce voters from previous slots into supporting blocks that conflict with the honest chain, enabling a chain reorganization at no cost to the adversary. In response, we introduce a novel reward mechanism that restores the voters' role as a check against proposer power. Our proposed mitigation is fairer and more decentralized -- not only in the context of these attacks -- but also practical for implementation in Ethereum.

摘要: 未经许可的大规模区块链(例如，以太)中的验证器通常是收益最大化的理性参与者。Etherum依赖于协议内激励，例如对提供正确和及时投票的验证者的奖励，以诱导诚实的行为并保护区块链。然而，外部激励，如区块提出者获取最大可提取价值(MEV)的机会，可能会诱使验证者偏离诚实的协议参与。我们展示了对LMD Ghost的一系列承诺攻击，LMD Ghost是以太共识机制的核心部分。我们展示了一个单一的对抗性区块提出者如何通过操纵以太的奖励系统来协调远程链重组，以获得及时的投票。这些攻击破坏了提倡者和选民之间预期的权力平衡：通过利用可信的威胁，对抗性的提倡者可以迫使选民从之前的位置进入与诚实链冲突的支持块，从而在不对对手造成任何成本的情况下实现链重组。作为回应，我们引入了一种新颖的奖励机制，恢复了选民对提议权力的制衡作用。我们提议的缓解措施更公平和更分散--不仅在这些攻击的背景下--而且在以太实施方面也是可行的。



## **16. Predominant Aspects on Security for Quantum Machine Learning: Literature Review**

量子机器学习安全性的主要方面：文献评论 quant-ph

Accepted at the IEEE International Conference on Quantum Computing  and Engineering (QCE)

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2401.07774v3) [paper-pdf](http://arxiv.org/pdf/2401.07774v3)

**Authors**: Nicola Franco, Alona Sakhnenko, Leon Stolpmann, Daniel Thuerck, Fabian Petsch, Annika Rüll, Jeanette Miriam Lorenz

**Abstract**: Quantum Machine Learning (QML) has emerged as a promising intersection of quantum computing and classical machine learning, anticipated to drive breakthroughs in computational tasks. This paper discusses the question which security concerns and strengths are connected to QML by means of a systematic literature review. We categorize and review the security of QML models, their vulnerabilities inherent to quantum architectures, and the mitigation strategies proposed. The survey reveals that while QML possesses unique strengths, it also introduces novel attack vectors not seen in classical systems. We point out specific risks, such as cross-talk in superconducting systems and forced repeated shuttle operations in ion-trap systems, which threaten QML's reliability. However, approaches like adversarial training, quantum noise exploitation, and quantum differential privacy have shown potential in enhancing QML robustness. Our review discuss the need for continued and rigorous research to ensure the secure deployment of QML in real-world applications. This work serves as a foundational reference for researchers and practitioners aiming to navigate the security aspects of QML.

摘要: 量子机器学习(QML)已经成为量子计算和经典机器学习的一个有前途的交叉点，有望推动计算任务的突破。本文通过系统的文献综述，探讨了QML的安全关注点和优势所在。我们对QML模型的安全性、量子体系结构固有的脆弱性以及提出的缓解策略进行了分类和回顾。调查显示，虽然QML具有独特的优势，但它也引入了经典系统中未曾见过的新攻击载体。我们指出了具体的风险，如超导系统中的串扰和离子陷阱系统中被迫重复的穿梭操作，这些风险威胁到了QML的可靠性。然而，对抗性训练、量子噪声利用和量子差分隐私等方法已经显示出在增强QML稳健性方面的潜力。我们的综述讨论了持续和严格研究的必要性，以确保QML在现实世界应用程序中的安全部署。这项工作为旨在导航QML安全方面的研究人员和实践者提供了基础性参考。



## **17. A Closer Look at GAN Priors: Exploiting Intermediate Features for Enhanced Model Inversion Attacks**

仔细研究GAN先验：利用中间功能进行增强模型倒置攻击 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2407.13863v3) [paper-pdf](http://arxiv.org/pdf/2407.13863v3)

**Authors**: Yixiang Qiu, Hao Fang, Hongyao Yu, Bin Chen, MeiKang Qiu, Shu-Tao Xia

**Abstract**: Model Inversion (MI) attacks aim to reconstruct privacy-sensitive training data from released models by utilizing output information, raising extensive concerns about the security of Deep Neural Networks (DNNs). Recent advances in generative adversarial networks (GANs) have contributed significantly to the improved performance of MI attacks due to their powerful ability to generate realistic images with high fidelity and appropriate semantics. However, previous MI attacks have solely disclosed private information in the latent space of GAN priors, limiting their semantic extraction and transferability across multiple target models and datasets. To address this challenge, we propose a novel method, Intermediate Features enhanced Generative Model Inversion (IF-GMI), which disassembles the GAN structure and exploits features between intermediate blocks. This allows us to extend the optimization space from latent code to intermediate features with enhanced expressive capabilities. To prevent GAN priors from generating unrealistic images, we apply a L1 ball constraint to the optimization process. Experiments on multiple benchmarks demonstrate that our method significantly outperforms previous approaches and achieves state-of-the-art results under various settings, especially in the out-of-distribution (OOD) scenario. Our code is available at: https://github.com/final-solution/IF-GMI

摘要: 模型反转(MI)攻击的目的是利用输出信息从已发布的模型中重建隐私敏感的训练数据，这引起了人们对深度神经网络(DNN)安全性的广泛关注。生成性对抗网络(GANS)的最新进展为MI攻击的性能改进做出了重要贡献，因为它们能够生成高保真和适当语义的真实图像。然而，以往的MI攻击只在GaN先验的潜在空间中泄露隐私信息，限制了它们的语义提取和跨多个目标模型和数据集的可传输性。为了解决这一挑战，我们提出了一种新的方法，中间特征增强的生成性模型反转(IF-GMI)，它分解GaN结构并利用中间块之间的特征。这允许我们将优化空间从潜在代码扩展到具有增强表达能力的中间功能。为了防止GaN先验数据产生不真实的图像，我们在优化过程中应用了L1球约束。在多个基准测试上的实验表明，我们的方法显著优于以前的方法，并在各种设置下获得了最先进的结果，特别是在分布外(OOD)的情况下。我们的代码请访问：https://github.com/final-solution/IF-GMI



## **18. EaTVul: ChatGPT-based Evasion Attack Against Software Vulnerability Detection**

EaTVul：针对软件漏洞检测的基于ChatGPT的规避攻击 cs.CR

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2407.19216v1) [paper-pdf](http://arxiv.org/pdf/2407.19216v1)

**Authors**: Shigang Liu, Di Cao, Junae Kim, Tamas Abraham, Paul Montague, Seyit Camtepe, Jun Zhang, Yang Xiang

**Abstract**: Recently, deep learning has demonstrated promising results in enhancing the accuracy of vulnerability detection and identifying vulnerabilities in software. However, these techniques are still vulnerable to attacks. Adversarial examples can exploit vulnerabilities within deep neural networks, posing a significant threat to system security. This study showcases the susceptibility of deep learning models to adversarial attacks, which can achieve 100% attack success rate (refer to Table 5). The proposed method, EaTVul, encompasses six stages: identification of important samples using support vector machines, identification of important features using the attention mechanism, generation of adversarial data based on these features using ChatGPT, preparation of an adversarial attack pool, selection of seed data using a fuzzy genetic algorithm, and the execution of an evasion attack. Extensive experiments demonstrate the effectiveness of EaTVul, achieving an attack success rate of more than 83% when the snippet size is greater than 2. Furthermore, in most cases with a snippet size of 4, EaTVul achieves a 100% attack success rate. The findings of this research emphasize the necessity of robust defenses against adversarial attacks in software vulnerability detection.

摘要: 最近，深度学习在提高漏洞检测的准确性和识别软件中的漏洞方面显示了良好的结果。然而，这些技术仍然容易受到攻击。敌意的例子可以利用深层神经网络中的漏洞，对系统安全构成重大威胁。这项研究展示了深度学习模型对对抗性攻击的敏感性，它可以达到100%的攻击成功率(参见表5)。该方法EaTVul包括六个阶段：使用支持向量机识别重要样本，使用注意机制识别重要特征，使用ChatGPT基于这些特征生成对抗性数据，准备对抗性攻击池，使用模糊遗传算法选择种子数据，以及执行规避攻击。大量实验证明了EaTVul的有效性，当Snipment大小大于2时，攻击成功率达到83%以上。此外，在大多数情况下，Snipment大小为4的情况下，EaTVul达到100%的攻击成功率。这项研究的结果强调了在软件漏洞检测中对对手攻击进行稳健防御的必要性。



## **19. Debiased Graph Poisoning Attack via Contrastive Surrogate Objective**

通过对比替代目标的去偏图中毒攻击 cs.LG

9 pages. Proceeding ACM International Conference on Information and  Knowledge Management (CIKM 2024) Proceeding

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2407.19155v1) [paper-pdf](http://arxiv.org/pdf/2407.19155v1)

**Authors**: Kanghoon Yoon, Yeonjun In, Namkyeong Lee, Kibum Kim, Chanyoung Park

**Abstract**: Graph neural networks (GNN) are vulnerable to adversarial attacks, which aim to degrade the performance of GNNs through imperceptible changes on the graph. However, we find that in fact the prevalent meta-gradient-based attacks, which utilizes the gradient of the loss w.r.t the adjacency matrix, are biased towards training nodes. That is, their meta-gradient is determined by a training procedure of the surrogate model, which is solely trained on the training nodes. This bias manifests as an uneven perturbation, connecting two nodes when at least one of them is a labeled node, i.e., training node, while it is unlikely to connect two unlabeled nodes. However, these biased attack approaches are sub-optimal as they do not consider flipping edges between two unlabeled nodes at all. This means that they miss the potential attacked edges between unlabeled nodes that significantly alter the representation of a node. In this paper, we investigate the meta-gradients to uncover the root cause of the uneven perturbations of existing attacks. Based on our analysis, we propose a Meta-gradient-based attack method using contrastive surrogate objective (Metacon), which alleviates the bias in meta-gradient using a new surrogate loss. We conduct extensive experiments to show that Metacon outperforms existing meta gradient-based attack methods through benchmark datasets, while showing that alleviating the bias towards training nodes is effective in attacking the graph structure.

摘要: 图神经网络(GNN)容易受到敌意攻击，其目的是通过图上不可察觉的变化来降低GNN的性能。然而，我们发现，实际上流行的基于亚梯度的攻击利用邻接矩阵的损失的梯度，偏向于训练节点。也就是说，它们的亚梯度由仅在训练节点上训练的代理模型的训练过程确定。这种偏差表现为不均匀的扰动，当两个节点中至少有一个是已标记节点(即训练节点)时连接两个节点，而不太可能连接两个未标记节点。然而，这些有偏见的攻击方法是次优的，因为它们根本不考虑两个未标记节点之间的翻转边。这意味着它们错过了未标记节点之间的潜在攻击边，这些边显著改变了节点的表示。在本文中，我们研究了亚梯度，以揭示现有攻击的不均匀扰动的根本原因。在此基础上，提出了一种基于元梯度的对比代理目标攻击方法(Metacon)，该方法通过引入新的代理损失来缓解元梯度的偏向。我们通过大量的实验表明，Metacon的攻击性能优于现有的基于元梯度的攻击方法，同时也证明了减轻对训练节点的偏向对图结构的攻击是有效的。



## **20. A Survey of Malware Detection Using Deep Learning**

使用深度学习的恶意软件检测概览 cs.CR

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2407.19153v1) [paper-pdf](http://arxiv.org/pdf/2407.19153v1)

**Authors**: Ahmed Bensaoud, Jugal Kalita, Mahmoud Bensaoud

**Abstract**: The problem of malicious software (malware) detection and classification is a complex task, and there is no perfect approach. There is still a lot of work to be done. Unlike most other research areas, standard benchmarks are difficult to find for malware detection. This paper aims to investigate recent advances in malware detection on MacOS, Windows, iOS, Android, and Linux using deep learning (DL) by investigating DL in text and image classification, the use of pre-trained and multi-task learning models for malware detection approaches to obtain high accuracy and which the best approach if we have a standard benchmark dataset. We discuss the issues and the challenges in malware detection using DL classifiers by reviewing the effectiveness of these DL classifiers and their inability to explain their decisions and actions to DL developers presenting the need to use Explainable Machine Learning (XAI) or Interpretable Machine Learning (IML) programs. Additionally, we discuss the impact of adversarial attacks on deep learning models, negatively affecting their generalization capabilities and resulting in poor performance on unseen data. We believe there is a need to train and test the effectiveness and efficiency of the current state-of-the-art deep learning models on different malware datasets. We examine eight popular DL approaches on various datasets. This survey will help researchers develop a general understanding of malware recognition using deep learning.

摘要: 恶意软件(Malware)的检测和分类问题是一项复杂的任务，目前还没有完美的方法。还有很多工作要做。与大多数其他研究领域不同，很难找到用于检测恶意软件的标准基准。本文旨在研究深度学习在MacOS、Windows、iOS、Android和Linux上的恶意软件检测方面的最新进展，通过研究深度学习在文本和图像分类中的应用，使用预训练和多任务学习模型来获得高精度的恶意软件检测方法，以及如果我们有标准的基准数据集，哪种方法是最好的方法。我们讨论了使用DL分类器检测恶意软件的问题和挑战，方法是回顾这些DL分类器的有效性，以及它们无法向DL开发人员解释它们的决策和操作，并提出需要使用可解释机器学习(XAI)或可解释机器学习(IML)程序。此外，我们还讨论了敌意攻击对深度学习模型的影响，对其泛化能力产生了负面影响，并导致在不可见数据上的性能较差。我们认为有必要对当前最先进的深度学习模型在不同恶意软件数据集上的有效性和效率进行培训和测试。我们在不同的数据集上检查了八种流行的数据挖掘方法。这项调查将帮助研究人员利用深度学习对恶意软件识别有一个大致的理解。



## **21. Accurate and Scalable Detection and Investigation of Cyber Persistence Threats**

对网络持久性威胁进行准确且可扩展的检测和调查 cs.CR

16 pages

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18832v1) [paper-pdf](http://arxiv.org/pdf/2407.18832v1)

**Authors**: Qi Liu, Muhammad Shoaib, Mati Ur Rehman, Kaibin Bao, Veit Hagenmeyer, Wajih Ul Hassan

**Abstract**: In Advanced Persistent Threat (APT) attacks, achieving stealthy persistence within target systems is often crucial for an attacker's success. This persistence allows adversaries to maintain prolonged access, often evading detection mechanisms. Recognizing its pivotal role in the APT lifecycle, this paper introduces Cyber Persistence Detector (CPD), a novel system dedicated to detecting cyber persistence through provenance analytics. CPD is founded on the insight that persistent operations typically manifest in two phases: the "persistence setup" and the subsequent "persistence execution". By causally relating these phases, we enhance our ability to detect persistent threats. First, CPD discerns setups signaling an impending persistent threat and then traces processes linked to remote connections to identify persistence execution activities. A key feature of our system is the introduction of pseudo-dependency edges (pseudo-edges), which effectively connect these disjoint phases using data provenance analysis, and expert-guided edges, which enable faster tracing and reduced log size. These edges empower us to detect persistence threats accurately and efficiently. Moreover, we propose a novel alert triage algorithm that further reduces false positives associated with persistence threats. Evaluations conducted on well-known datasets demonstrate that our system reduces the average false positive rate by 93% compared to state-of-the-art methods.

摘要: 在高级持久威胁(APT)攻击中，在目标系统内实现隐形持久性通常是攻击者成功的关键。这种持久性使攻击者能够保持长时间的访问，通常可以避开检测机制。认识到其在APT生命周期中的关键作用，本文引入了网络持久性检测器(CPD)，这是一个致力于通过来源分析来检测网络持久性的新系统。CPD建立在持久化操作通常表现为两个阶段的洞察力之上：“持久化设置”和随后的“持久化执行”。通过将这些阶段因果联系起来，我们增强了检测持续威胁的能力。首先，CPD识别发出即将到来的持久威胁的设置，然后跟踪链接到远程连接的进程，以识别持久执行活动。我们系统的一个关键特征是引入了伪依赖边(伪边)，它使用数据来源分析有效地连接了这些不相交的阶段，以及专家指导的边，从而实现了更快的跟踪和减少日志大小。这些边缘使我们能够准确高效地检测持久性威胁。此外，我们还提出了一种新的警报分类算法，进一步减少了与持久性威胁相关的误报。在知名数据集上进行的评估表明，与最先进的方法相比，我们的系统平均假阳性率降低了93%。



## **22. Frosty: Bringing strong liveness guarantees to the Snow family of consensus protocols**

Frosty：为Snow家族的共识协议带来强大的活力保证 cs.DC

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2404.14250v4) [paper-pdf](http://arxiv.org/pdf/2404.14250v4)

**Authors**: Aaron Buchwald, Stephen Buttolph, Andrew Lewis-Pye, Patrick O'Grady, Kevin Sekniqi

**Abstract**: Snowman is the consensus protocol implemented by the Avalanche blockchain and is part of the Snow family of protocols, first introduced through the original Avalanche leaderless consensus protocol. A major advantage of Snowman is that each consensus decision only requires an expected constant communication overhead per processor in the `common' case that the protocol is not under substantial Byzantine attack, i.e. it provides a solution to the scalability problem which ensures that the expected communication overhead per processor is independent of the total number of processors $n$ during normal operation. This is the key property that would enable a consensus protocol to scale to 10,000 or more independent validators (i.e. processors). On the other hand, the two following concerns have remained:   (1) Providing formal proofs of consistency for Snowman has presented a formidable challenge.   (2) Liveness attacks exist in the case that a Byzantine adversary controls more than $O(\sqrt{n})$ processors, slowing termination to more than a logarithmic number of steps.   In this paper, we address the two issues above. We consider a Byzantine adversary that controls at most $f<n/5$ processors. First, we provide a simple proof of consistency for Snowman. Then we supplement Snowman with a `liveness module' that can be triggered in the case that a substantial adversary launches a liveness attack, and which guarantees liveness in this event by temporarily forgoing the communication complexity advantages of Snowman, but without sacrificing these low communication complexity advantages during normal operation.

摘要: 雪人是雪崩区块链实施的共识协议，是雪诺协议家族的一部分，最初是通过最初的雪崩无领导共识协议引入的。Snowman的一个主要优势是，在协议没有受到实质性拜占庭攻击的情况下，每个协商一致的决定只需要每个处理器预期的恒定通信开销，即它提供了对可伸缩性问题的解决方案，该解决方案确保在正常操作期间每个处理器的预期通信开销与处理器总数$n$无关。这是使共识协议能够扩展到10,000个或更多独立验证器(即处理器)的关键属性。另一方面，以下两个问题仍然存在：(1)为雪人提供一致性的正式证据是一个巨大的挑战。(2)当拜占庭敌手控制超过$O(\Sqrt{n})$个处理器时，存在活性攻击，从而将终止速度减慢到超过对数步数。在本文中，我们解决了上述两个问题。我们考虑一个拜占庭对手，它至多控制$f<n/5$处理器。首先，我们为雪人提供了一个简单的一致性证明。然后，我们给Snowman增加了一个活跃度模块，该模块可以在强大的对手发起活跃度攻击的情况下触发，并通过暂时放弃Snowman的通信复杂性优势来保证在这种情况下的活跃性，但在正常运行时不会牺牲这些低通信复杂性的优势。



## **23. Embodied Laser Attack:Leveraging Scene Priors to Achieve Agent-based Robust Non-contact Attacks**

准激光攻击：利用场景先验实现基于代理的鲁棒非接触式攻击 cs.CV

9 pages, 7 figures, Accepted by ACM MM 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2312.09554v3) [paper-pdf](http://arxiv.org/pdf/2312.09554v3)

**Authors**: Yitong Sun, Yao Huang, Xingxing Wei

**Abstract**: As physical adversarial attacks become extensively applied in unearthing the potential risk of security-critical scenarios, especially in dynamic scenarios, their vulnerability to environmental variations has also been brought to light. The non-robust nature of physical adversarial attack methods brings less-than-stable performance consequently. Although methods such as EOT have enhanced the robustness of traditional contact attacks like adversarial patches, they fall short in practicality and concealment within dynamic environments such as traffic scenarios. Meanwhile, non-contact laser attacks, while offering enhanced adaptability, face constraints due to a limited optimization space for their attributes, rendering EOT less effective. This limitation underscores the necessity for developing a new strategy to augment the robustness of such practices. To address these issues, this paper introduces the Embodied Laser Attack (ELA), a novel framework that leverages the embodied intelligence paradigm of Perception-Decision-Control to dynamically tailor non-contact laser attacks. For the perception module, given the challenge of simulating the victim's view by full-image transformation, ELA has innovatively developed a local perspective transformation network, based on the intrinsic prior knowledge of traffic scenes and enables effective and efficient estimation. For the decision and control module, ELA trains an attack agent with data-driven reinforcement learning instead of adopting time-consuming heuristic algorithms, making it capable of instantaneously determining a valid attack strategy with the perceived information by well-designed rewards, which is then conducted by a controllable laser emitter. Experimentally, we apply our framework to diverse traffic scenarios both in the digital and physical world, verifying the effectiveness of our method under dynamic successive scenes.

摘要: 随着物理对抗性攻击被广泛应用于挖掘安全关键场景的潜在风险，特别是在动态场景中，它们对环境变化的脆弱性也暴露了出来。因此，物理对抗性攻击方法的非健壮性带来了不稳定的性能。虽然EOT等方法增强了传统接触攻击(如对抗性补丁)的健壮性，但在交通场景等动态环境中缺乏实用性和隐蔽性。同时，非接触式激光攻击虽然提供了增强的适应性，但由于其属性的优化空间有限而面临约束，使得EOT的效率较低。这一局限性突出表明，有必要制定一项新战略，以加强这类做法的稳健性。为了解决这些问题，本文引入了嵌入式激光攻击(ELA)，这是一种利用感知-决策-控制的嵌入式智能范式来动态定制非接触式激光攻击的框架。对于感知模块，针对通过全图像变换模拟受害者视角的挑战，ELA基于交通场景的固有先验知识，创新性地开发了局部透视变换网络，并实现了有效和高效的估计。对于决策与控制模块，ELA用数据驱动的强化学习来训练攻击代理，而不是采用耗时的启发式算法，使其能够通过精心设计的奖励，根据感知的信息即时确定有效的攻击策略，然后由可控的激光发射器进行。实验中，我们将我们的框架应用于数字和物理世界中不同的交通场景，验证了我们的方法在动态连续场景下的有效性。



## **24. Adversarial Robustification via Text-to-Image Diffusion Models**

通过文本到图像扩散模型的对抗性Robusification cs.CV

Code is available at https://github.com/ChoiDae1/robustify-T2I

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18658v1) [paper-pdf](http://arxiv.org/pdf/2407.18658v1)

**Authors**: Daewon Choi, Jongheon Jeong, Huiwon Jang, Jinwoo Shin

**Abstract**: Adversarial robustness has been conventionally believed as a challenging property to encode for neural networks, requiring plenty of training data. In the recent paradigm of adopting off-the-shelf models, however, access to their training data is often infeasible or not practical, while most of such models are not originally trained concerning adversarial robustness. In this paper, we develop a scalable and model-agnostic solution to achieve adversarial robustness without using any data. Our intuition is to view recent text-to-image diffusion models as "adaptable" denoisers that can be optimized to specify target tasks. Based on this, we propose: (a) to initiate a denoise-and-classify pipeline that offers provable guarantees against adversarial attacks, and (b) to leverage a few synthetic reference images generated from the text-to-image model that enables novel adaptation schemes. Our experiments show that our data-free scheme applied to the pre-trained CLIP could improve the (provable) adversarial robustness of its diverse zero-shot classification derivatives (while maintaining their accuracy), significantly surpassing prior approaches that utilize the full training data. Not only for CLIP, we also demonstrate that our framework is easily applicable for robustifying other visual classifiers efficiently.

摘要: 传统上，对抗健壮性被认为是神经网络编码的一种具有挑战性的特性，需要大量的训练数据。然而，在最近采用现成模型的范例中，获取其训练数据往往是不可行或不实用的，而大多数这种模型最初并没有接受过关于对手稳健性的培训。在本文中，我们开发了一个可扩展的与模型无关的解决方案，以在不使用任何数据的情况下实现对手健壮性。我们的直觉是将最近的文本到图像扩散模型视为可以优化以指定目标任务的“适应性”消噪器。基于此，我们建议：(A)启动去噪和分类管道，提供针对对手攻击的可证明的保证，以及(B)利用从文本到图像模型生成的几个合成参考图像，从而实现新的适应方案。我们的实验表明，我们的无数据方案应用于预先训练的剪辑，可以提高其各种零射击分类导数的(可证明的)对抗健壮性(同时保持其准确性)，显著超过以往利用全部训练数据的方法。不仅对于CLIP，我们还证明了我们的框架可以很容易地适用于高效地增强其他视觉分类器的鲁棒性。



## **25. Robust VAEs via Generating Process of Noise Augmented Data**

通过生成噪音增强数据的过程实现稳健的VAE cs.LG

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18632v1) [paper-pdf](http://arxiv.org/pdf/2407.18632v1)

**Authors**: Hiroo Irobe, Wataru Aoki, Kimihiro Yamazaki, Yuhui Zhang, Takumi Nakagawa, Hiroki Waida, Yuichiro Wada, Takafumi Kanamori

**Abstract**: Advancing defensive mechanisms against adversarial attacks in generative models is a critical research topic in machine learning. Our study focuses on a specific type of generative models - Variational Auto-Encoders (VAEs). Contrary to common beliefs and existing literature which suggest that noise injection towards training data can make models more robust, our preliminary experiments revealed that naive usage of noise augmentation technique did not substantially improve VAE robustness. In fact, it even degraded the quality of learned representations, making VAEs more susceptible to adversarial perturbations. This paper introduces a novel framework that enhances robustness by regularizing the latent space divergence between original and noise-augmented data. Through incorporating a paired probabilistic prior into the standard variational lower bound, our method significantly boosts defense against adversarial attacks. Our empirical evaluations demonstrate that this approach, termed Robust Augmented Variational Auto-ENcoder (RAVEN), yields superior performance in resisting adversarial inputs on widely-recognized benchmark datasets.

摘要: 在产生式模型中提出对抗攻击的防御机制是机器学习中的一个重要研究课题。我们的研究集中在一种特定类型的产生式模型-变分自动编码器(VAE)。与通常的信念和现有文献表明向训练数据注入噪声可以使模型更稳健相反，我们的初步实验表明，幼稚地使用噪声增强技术并没有显著提高VAE的稳健性。事实上，它甚至降低了学习表征的质量，使VAE更容易受到对抗性干扰。本文介绍了一种新的框架，它通过正则化原始数据和噪声增强数据之间的潜在空间发散来增强稳健性。通过在标准变分下界中引入成对概率先验，我们的方法显著增强了对对手攻击的防御。我们的实验评估表明，这种方法被称为稳健的增广变分自动编码器(RAVEN)，在抵抗广泛认可的基准数据集上的敌意输入方面具有优越的性能。



## **26. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

DistriBlock：通过利用输出分布的特征来识别对抗性音频样本 cs.SD

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2305.17000v6) [paper-pdf](http://arxiv.org/pdf/2305.17000v6)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.

摘要: 敌意攻击可以误导自动语音识别(ASR)系统预测任意目标文本，从而构成明显的安全威胁。为了防止此类攻击，我们提出了DistriBlock，这是一种适用于任何ASR系统的有效检测策略，它预测每个时间步输出令牌上的概率分布。我们测量了该分布的一组特征：输出概率的中位数、最大值和最小值，分布的熵，以及关于后续时间步分布的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性数据和恶意数据观察到的特征，我们应用二进制分类器，包括简单的基于阈值的分类、这种分类器的集成和神经网络。通过对不同的ASR系统和语言数据集的广泛分析，我们证明了该方法的最高性能，在干净和有噪声的数据下，接收器操作特征曲线下的平均面积分别为99%和97%。为了评估我们方法的健壮性，我们证明了可以绕过DistriBlock的自适应攻击示例的噪声要大得多，这使得它们更容易通过过滤来检测，并为保持系统的健壮性创造了另一种途径。



## **27. Unveiling Privacy Vulnerabilities: Investigating the Role of Structure in Graph Data**

揭露隐私漏洞：调查结构在图形数据中的作用 cs.LG

In KDD'24; with full appendix

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18564v1) [paper-pdf](http://arxiv.org/pdf/2407.18564v1)

**Authors**: Hanyang Yuan, Jiarong Xu, Cong Wang, Ziqi Yang, Chunping Wang, Keting Yin, Yang Yang

**Abstract**: The public sharing of user information opens the door for adversaries to infer private data, leading to privacy breaches and facilitating malicious activities. While numerous studies have concentrated on privacy leakage via public user attributes, the threats associated with the exposure of user relationships, particularly through network structure, are often neglected. This study aims to fill this critical gap by advancing the understanding and protection against privacy risks emanating from network structure, moving beyond direct connections with neighbors to include the broader implications of indirect network structural patterns. To achieve this, we first investigate the problem of Graph Privacy Leakage via Structure (GPS), and introduce a novel measure, the Generalized Homophily Ratio, to quantify the various mechanisms contributing to privacy breach risks in GPS. Based on this insight, we develop a novel graph private attribute inference attack, which acts as a pivotal tool for evaluating the potential for privacy leakage through network structures under worst-case scenarios. To protect users' private data from such vulnerabilities, we propose a graph data publishing method incorporating a learnable graph sampling technique, effectively transforming the original graph into a privacy-preserving version. Extensive experiments demonstrate that our attack model poses a significant threat to user privacy, and our graph data publishing method successfully achieves the optimal privacy-utility trade-off compared to baselines.

摘要: 用户信息的公开共享为攻击者推断私人数据打开了大门，导致隐私被侵犯，并为恶意活动提供便利。虽然许多研究都集中在通过公共用户属性泄露隐私，但与暴露用户关系相关的威胁，特别是通过网络结构，往往被忽视。这项研究旨在通过促进对网络结构产生的隐私风险的理解和保护来填补这一关键空白，超越与邻居的直接连接，包括间接网络结构模式的更广泛影响。为此，我们首先研究了通过结构的图隐私泄露问题，并引入了一种新的度量方法--广义同伦比来量化GPS中导致隐私泄露风险的各种机制。基于这一观点，我们开发了一种新的图私有属性推理攻击，该攻击可以作为评估最坏情况下通过网络结构泄露隐私的可能性的关键工具。为了保护用户的私有数据免受此类漏洞的侵害，我们提出了一种结合了可学习的图采样技术的图数据发布方法，有效地将原始图转换为隐私保护版本。大量的实验表明，我们的攻击模型对用户隐私构成了严重的威胁，并且我们的图数据发布方法成功地实现了相对于基线的最优隐私效用权衡。



## **28. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

Janus界面：大型语言模型中的微调如何放大隐私风险 cs.CR

This work has been accepted by CCS 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2310.15469v3) [paper-pdf](http://arxiv.org/pdf/2310.15469v3)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, Haixu Tang

**Abstract**: The rapid advancements of large language models (LLMs) have raised public concerns about the privacy leakage of personally identifiable information (PII) within their extensive training datasets. Recent studies have demonstrated that an adversary could extract highly sensitive privacy data from the training data of LLMs with carefully designed prompts. However, these attacks suffer from the model's tendency to hallucinate and catastrophic forgetting (CF) in the pre-training stage, rendering the veracity of divulged PIIs negligible. In our research, we propose a novel attack, Janus, which exploits the fine-tuning interface to recover forgotten PIIs from the pre-training data in LLMs. We formalize the privacy leakage problem in LLMs and explain why forgotten PIIs can be recovered through empirical analysis on open-source language models. Based upon these insights, we evaluate the performance of Janus on both open-source language models and two latest LLMs, i.e., GPT-3.5-Turbo and LLaMA-2-7b. Our experiment results show that Janus amplifies the privacy risks by over 10 times in comparison with the baseline and significantly outperforms the state-of-the-art privacy extraction attacks including prefix attacks and in-context learning (ICL). Furthermore, our analysis validates that existing fine-tuning APIs provided by OpenAI and Azure AI Studio are susceptible to our Janus attack, allowing an adversary to conduct such an attack at a low cost.

摘要: 大型语言模型(LLM)的快速发展引起了公众对其广泛训练数据集中个人身份信息(PII)隐私泄露的担忧。最近的研究表明，攻击者可以通过精心设计的提示从LLMS的训练数据中提取高度敏感的隐私数据。然而，这些攻击受到模型在预训练阶段的幻觉和灾难性遗忘(CF)的倾向的影响，使得泄露的PII的真实性可以忽略不计。在我们的研究中，我们提出了一种新的攻击，Janus，它利用微调接口从LLMS的训练前数据中恢复被遗忘的PII。我们形式化地描述了LLMS中的隐私泄露问题，并通过对开源语言模型的实证分析解释了为什么被遗忘的PII可以恢复。基于这些见解，我们评估了Janus在开源语言模型和两个最新的LLMS上的性能，即GPT-3.5-Turbo和Llama-2-7b。我们的实验结果表明，Janus将隐私风险放大了10倍以上，并且显著优于目前最先进的隐私提取攻击，包括前缀攻击和上下文中学习(ICL)。此外，我们的分析验证了OpenAI和Azure AI Studio提供的现有微调API容易受到我们的Janus攻击，允许对手以低成本进行此类攻击。



## **29. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

Dysca：评估LVLM感知能力的动态和可扩展基准 cs.CV

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2406.18849v2) [paper-pdf](http://arxiv.org/pdf/2406.18849v2)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 8 advanced open-source LVLMs with 10 checkpoints are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released in \url{https://github.com/Benchmark-Dysca/Dysca}.

摘要: 目前，人们已经提出了许多基准来评估大型视觉语言模型的感知能力。然而，大多数基准测试通过从现有数据集中选择图像来进行问题，从而导致潜在的数据泄漏。此外，这些基准只关注真实感风格的图像和干净的场景来评估LVLMS，而对多风格化的图像和噪声场景没有进行探索。为了应对这些挑战，我们提出了一种动态的、可扩展的基准测试DYSCA，用于利用合成图像来评估LVLMS。具体地说，我们利用稳定扩散并设计了一种基于规则的方法来动态生成新的图像、问题和相应的答案。我们考虑了51种图像风格，并在20个子任务中评估了感知能力。此外，我们还在4个场景(即廉洁、腐败、打印攻击和对抗性攻击)和3个问题类型(即多项选择、对错和自由形式)下进行了评估。多亏了生成性范式，Dysca成为了一个可伸缩的基准，可以轻松添加新的子任务和场景。在Dysca上对8个具有10个检查点的高级开源LVLMS进行了评估，揭示了当前LVLMS的缺陷。该基准测试在\url{https://github.com/Benchmark-Dysca/Dysca}.中发布



## **30. Machine Unlearning using a Multi-GAN based Model**

使用基于多GAN的模型的机器去学习 cs.LG

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18467v1) [paper-pdf](http://arxiv.org/pdf/2407.18467v1)

**Authors**: Amartya Hatua, Trung T. Nguyen, Andrew H. Sung

**Abstract**: This article presents a new machine unlearning approach that utilizes multiple Generative Adversarial Network (GAN) based models. The proposed method comprises two phases: i) data reorganization in which synthetic data using the GAN model is introduced with inverted class labels of the forget datasets, and ii) fine-tuning the pre-trained model. The GAN models consist of two pairs of generators and discriminators. The generator discriminator pairs generate synthetic data for the retain and forget datasets. Then, a pre-trained model is utilized to get the class labels of the synthetic datasets. The class labels of synthetic and original forget datasets are inverted. Finally, all combined datasets are used to fine-tune the pre-trained model to get the unlearned model. We have performed the experiments on the CIFAR-10 dataset and tested the unlearned models using Membership Inference Attacks (MIA). The inverted class labels procedure and synthetically generated data help to acquire valuable information that enables the model to outperform state-of-the-art models and other standard unlearning classifiers.

摘要: 提出了一种基于多生成性对抗网络(GAN)模型的机器遗忘方法。该方法包括两个阶段：i)数据重组，其中使用GAN模型的合成数据被引入带有倒排的忘记数据集的类别标签；ii)微调预先训练的模型。GaN模型由两对生成器和鉴别器组成。生成器鉴别器对为保留和忘记数据集生成合成数据。然后，利用预先训练好的模型得到合成数据集的类标签。对合成的和原始的忘记数据集的分类标签进行倒置。最后，使用所有组合的数据集来微调预先训练的模型，以获得未学习的模型。我们在CIFAR-10数据集上进行了实验，并使用隶属度推理攻击(MIA)对未学习模型进行了测试。倒置的分类标签程序和合成生成的数据有助于获取有价值的信息，使该模型的表现优于最先进的模型和其他标准的遗忘分类器。



## **31. Disrupting Diffusion: Token-Level Attention Erasure Attack against Diffusion-based Customization**

扰乱扩散：针对基于扩散的定制的代币级注意力擦除攻击 cs.CV

Accepted by ACM MM2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2405.20584v2) [paper-pdf](http://arxiv.org/pdf/2405.20584v2)

**Authors**: Yisu Liu, Jinyang An, Wanqian Zhang, Dayan Wu, Jingzi Gu, Zheng Lin, Weiping Wang

**Abstract**: With the development of diffusion-based customization methods like DreamBooth, individuals now have access to train the models that can generate their personalized images. Despite the convenience, malicious users have misused these techniques to create fake images, thereby triggering a privacy security crisis. In light of this, proactive adversarial attacks are proposed to protect users against customization. The adversarial examples are trained to distort the customization model's outputs and thus block the misuse. In this paper, we propose DisDiff (Disrupting Diffusion), a novel adversarial attack method to disrupt the diffusion model outputs. We first delve into the intrinsic image-text relationships, well-known as cross-attention, and empirically find that the subject-identifier token plays an important role in guiding image generation. Thus, we propose the Cross-Attention Erasure module to explicitly "erase" the indicated attention maps and disrupt the text guidance. Besides,we analyze the influence of the sampling process of the diffusion model on Projected Gradient Descent (PGD) attack and introduce a novel Merit Sampling Scheduler to adaptively modulate the perturbation updating amplitude in a step-aware manner. Our DisDiff outperforms the state-of-the-art methods by 12.75% of FDFR scores and 7.25% of ISM scores across two facial benchmarks and two commonly used prompts on average.

摘要: 随着像DreamBooth这样基于扩散的定制方法的发展，个人现在可以训练能够生成他们个性化图像的模型。尽管很方便，但恶意用户滥用这些技术创造了虚假图像，从而引发了隐私安全危机。有鉴于此，提出了主动对抗性攻击来保护用户免受定制。对抗性的例子被训练来扭曲定制模型的输出，从而阻止误用。在本文中，我们提出了一种新的对抗性攻击方法DisDiff(破坏扩散)来破坏扩散模型的输出。我们首先深入研究了内在的图文关系，即众所周知的交叉注意，并经验地发现，主体-标识符号在引导图像生成方面发挥着重要作用。因此，我们提出了交叉注意擦除模块来显式地“擦除”所指示的注意地图并扰乱文本引导。此外，我们还分析了扩散模型的采样过程对投影梯度下降(PGD)攻击的影响，并引入了一种新的优点采样调度器来以步长感知的方式自适应地调制扰动更新幅度。在两个面部基准和两个常用提示上，我们的DisDiff平均比最先进的方法高出12.75%的FDFR分数和7.25%的ISM分数。



## **32. Regret-Optimal Defense Against Stealthy Adversaries: A System Level Approach**

针对潜行对手的遗憾最佳防御：系统级方法 eess.SY

Accepted, IEEE Conference on Decision and Control (CDC), 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18448v1) [paper-pdf](http://arxiv.org/pdf/2407.18448v1)

**Authors**: Hiroyasu Tsukamoto, Joudi Hajar, Soon-Jo Chung, Fred Y. Hadaegh

**Abstract**: Modern control designs in robotics, aerospace, and cyber-physical systems heavily depend on real-world data obtained through the system outputs. In the face of system faults and malicious attacks, however, these outputs can be compromised to misrepresent some portion of the system information that critically affects their secure and trustworthy operation. In this paper, we introduce a novel regret-optimal control framework for designing controllers that render a linear system robust against stealthy attacks, including sensor and actuator attacks, as well as external disturbances. In particular, we establish (a) a convex optimization-based system metric to quantify the regret with the worst-case stealthy attack (the true performance minus the optimal performance in hindsight with the knowledge of the stealthy attack), which improves and adaptively interpolates $\mathcal{H}_2$ and $\mathcal{H}_{\infty}$ norms in the presence of stealthy adversaries, (b) an optimization problem for minimizing the regret of 1 expressed in the system level parameterization, which is useful for its localized and distributed implementation in large-scale systems, and (c) a rank-constrained optimization problem (i.e., optimization with a convex objective subject to convex constraints and rank constraints) equivalent to the optimization problem of (b). Finally, we conduct a numerical simulation which showcases the effectiveness of our approach.

摘要: 机器人、航空航天和计算机物理系统中的现代控制设计在很大程度上依赖于通过系统输出获得的真实世界数据。然而，面对系统故障和恶意攻击，这些输出可能会受到损害，从而歪曲系统信息的某些部分，从而严重影响其安全和可信的操作。在本文中，我们介绍了一种新的后悔最优控制框架，用于设计控制器，使线性系统对隐身攻击，包括传感器和执行器攻击，以及外部干扰具有鲁棒性。具体地，我们建立了(A)基于凸优化的系统度量来量化最坏情况下的隐蔽攻击的遗憾(真实性能减去具有隐形攻击知识的后见之明的最优性能)，它改进并自适应地内插在隐身对手存在的情况下的$\数学{H}_2$和$\数学{H}_(Inty)$范数；(B)以系统级参数化表示的最小化1的遗憾的优化问题，这对于其在大规模系统中的局部和分布式实现是有用的；以及(C)秩约束优化问题(即，具有凸约束和秩约束的凸目标最优化)等价于(B)的最优化问题。最后，我们进行了数值模拟，验证了该方法的有效性。



## **33. Human-Interpretable Adversarial Prompt Attack on Large Language Models with Situational Context**

具有情境上下文的大型语言模型的人类可解释对抗提示攻击 cs.CL

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.14644v2) [paper-pdf](http://arxiv.org/pdf/2407.14644v2)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous research on testing the vulnerabilities in Large Language Models (LLMs) using adversarial attacks has primarily focused on nonsensical prompt injections, which are easily detected upon manual or automated review (e.g., via byte entropy). However, the exploration of innocuous human-understandable malicious prompts augmented with adversarial injections remains limited. In this research, we explore converting a nonsensical suffix attack into a sensible prompt via a situation-driven contextual re-writing. This allows us to show suffix conversion without any gradients, using only LLMs to perform the attacks, and thus better understand the scope of possible risks. We combine an independent, meaningful adversarial insertion and situations derived from movies to check if this can trick an LLM. The situations are extracted from the IMDB dataset, and prompts are defined following a few-shot chain-of-thought prompting. Our approach demonstrates that a successful situation-driven attack can be executed on both open-source and proprietary LLMs. We find that across many LLMs, as few as 1 attempt produces an attack and that these attacks transfer between LLMs.

摘要: 之前关于使用对抗性攻击测试大型语言模型(LLM)中的漏洞的研究主要集中在无意义的提示注入上，这些注入很容易通过手动或自动审查(例如，通过字节熵)检测到。然而，通过恶意注入增强无害的人类可理解的恶意提示的探索仍然有限。在这项研究中，我们探索通过情景驱动的语境重写将无意义的后缀攻击转化为合理的提示。这使我们能够显示没有任何梯度的后缀转换，仅使用LLM来执行攻击，从而更好地了解可能风险的范围。我们结合了一个独立的、有意义的敌意插入和来自电影的情况来检查这是否可以欺骗LLM。情况是从IMDB数据集中提取的，提示是在几个镜头的思维链提示之后定义的。我们的方法表明，成功的情境驱动攻击可以在开源和专有LLM上执行。我们发现，在许多LLM中，只有1次尝试就会产生攻击，并且这些攻击会在LLM之间传输。



## **34. Sparse vs Contiguous Adversarial Pixel Perturbations in Multimodal Models: An Empirical Analysis**

稀疏与连续多峰模型中的对抗像素扰动：实证分析 cs.CV

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.18251v1) [paper-pdf](http://arxiv.org/pdf/2407.18251v1)

**Authors**: Cristian-Alexandru Botocan, Raphael Meier, Ljiljana Dolamic

**Abstract**: Assessing the robustness of multimodal models against adversarial examples is an important aspect for the safety of its users. We craft L0-norm perturbation attacks on the preprocessed input images. We launch them in a black-box setup against four multimodal models and two unimodal DNNs, considering both targeted and untargeted misclassification. Our attacks target less than 0.04% of perturbed image area and integrate different spatial positioning of perturbed pixels: sparse positioning and pixels arranged in different contiguous shapes (row, column, diagonal, and patch). To the best of our knowledge, we are the first to assess the robustness of three state-of-the-art multimodal models (ALIGN, AltCLIP, GroupViT) against different sparse and contiguous pixel distribution perturbations. The obtained results indicate that unimodal DNNs are more robust than multimodal models. Furthermore, models using CNN-based Image Encoder are more vulnerable than models with ViT - for untargeted attacks, we obtain a 99% success rate by perturbing less than 0.02% of the image area.

摘要: 评估多通道模型对敌意例子的稳健性是保证其使用者安全的一个重要方面。我们对经过预处理的输入图像进行L0范数扰动攻击。我们在黑盒设置中针对四个多模式模型和两个单峰DNN启动，同时考虑了目标和非目标错误分类。我们的攻击目标是不超过0.04%的扰动图像区域，并整合了扰动像素的不同空间位置：稀疏定位和以不同的连续形状(行、列、对角线和面片)排列的像素。据我们所知，我们首次评估了三种最先进的多模式模型(ALIGN、AltCLIP、GroupViT)在不同稀疏和连续像素分布扰动下的稳健性。结果表明，单模DNN比多模DNN具有更好的鲁棒性。此外，使用基于CNN的图像编码器的模型比使用VIT的模型更容易受到攻击-对于非目标攻击，我们通过扰动不到0.02%的图像区域获得99%的成功率。



## **35. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

杰基尔博士和海德先生：法学硕士的两面 cs.CR

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2312.03853v4) [paper-pdf](http://arxiv.org/pdf/2312.03853v4)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Gemini (and, to some extent, Bing chat) by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Gemini. We also introduce several ways of activating such adversarial personas, which show that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.

摘要: 最近，我们看到大型语言模型(LLM)的使用有所增加，特别是在聊天机器人助手等应用程序中。实施了安全机制和专门的培训程序，以防止这些助理做出不当反应。在这项工作中，我们绕过了ChatGPT和Gemini(在某种程度上，Bing聊天)的这些措施，让他们模仿具有与诚实的助手不一致的个性特征的复杂人物角色。我们首先为这些角色创建精致的传记，然后在与相同的聊天机器人的新会话中使用。然后，我们的对话遵循角色扮演的风格，以引发被禁止的回应。使用人物角色，我们展示了实际上提供了被禁止的响应，使得获得未经授权的、非法的或有害的信息成为可能。这项工作表明，通过使用敌对的人物角色，一个人可以克服ChatGPT和Gemini提出的安全机制。我们还介绍了几种激活这种敌对角色的方法，这表明这两个聊天机器人都容易受到这种攻击。在相同的原则下，我们引入了两个防御措施，推动该模型解释可信任的个性，并使其对此类攻击更加健壮。



## **36. RIDA: A Robust Attack Framework on Incomplete Graphs**

RIDA：一个针对不完整图的稳健攻击框架 cs.LG

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.18170v1) [paper-pdf](http://arxiv.org/pdf/2407.18170v1)

**Authors**: Jianke Yu, Hanchen Wang, Chen Chen, Xiaoyang Wang, Wenjie Zhang, Ying Zhang

**Abstract**: Graph Neural Networks (GNNs) are vital in data science but are increasingly susceptible to adversarial attacks. To help researchers develop more robust GNN models, it's essential to focus on designing strong attack models as foundational benchmarks and guiding references. Among adversarial attacks, gray-box poisoning attacks are noteworthy due to their effectiveness and fewer constraints. These attacks exploit GNNs' need for retraining on updated data, thereby impacting their performance by perturbing these datasets. However, current research overlooks the real-world scenario of incomplete graphs.To address this gap, we introduce the Robust Incomplete Deep Attack Framework (RIDA). It is the first algorithm for robust gray-box poisoning attacks on incomplete graphs. The approach innovatively aggregates distant vertex information and ensures powerful data utilization.Extensive tests against 9 SOTA baselines on 3 real-world datasets demonstrate RIDA's superiority in handling incompleteness and high attack performance on the incomplete graph.

摘要: 图神经网络(GNN)在数据科学中至关重要，但越来越容易受到对手攻击。为了帮助研究人员开发更健壮的GNN模型，有必要将重点放在设计强大的攻击模型作为基础基准和指导参考。在对抗性攻击中，灰箱中毒攻击由于其有效性和较少的约束而值得注意。这些攻击利用了GNN对更新数据进行再培训的需要，从而通过扰乱这些数据集来影响其性能。然而，目前的研究忽略了现实世界中不完整图形的场景，为了解决这一差距，我们引入了健壮的不完整深度攻击框架(RIDA)。这是第一个针对不完备图的稳健灰盒中毒攻击的算法。该方法创新性地聚合了距离较远的顶点信息，确保了强大的数据利用率，并在3个真实数据集上对9条SOTA基线进行了扩展测试，验证了RIDA在处理不完全图的不完备性和高攻击性能方面的优势。



## **37. Understanding the Security Benefits and Overheads of Emerging Industry Solutions to DRAM Read Disturbance**

了解新兴行业解决方案的安全优势和管理费用针对内存读取干扰 cs.CR

To appear in DRAMSec 2024

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2406.19094v2) [paper-pdf](http://arxiv.org/pdf/2406.19094v2)

**Authors**: Oğuzhan Canpolat, A. Giray Yağlıkçı, Geraldo F. Oliveira, Ataberk Olgun, Oğuz Ergin, Onur Mutlu

**Abstract**: We present the first rigorous security, performance, energy, and cost analyses of the state-of-the-art on-DRAM-die read disturbance mitigation method, Per Row Activation Counting (PRAC), described in JEDEC DDR5 specification's April 2024 update. Unlike prior state-of-the-art that advises the memory controller to periodically issue refresh management (RFM) commands, which provides the DRAM chip with time to perform refreshes, PRAC introduces a new back-off signal. PRAC's back-off signal propagates from the DRAM chip to the memory controller and forces the memory controller to 1) stop serving requests and 2) issue RFM commands. As a result, RFM commands are issued when needed as opposed to periodically, reducing RFM's overheads. We analyze PRAC in four steps. First, we define an adversarial access pattern that represents the worst-case for PRAC's security. Second, we investigate PRAC's configurations and security implications. Our analyses show that PRAC can be configured for secure operation as long as no bitflip occurs before accessing a memory location 10 times. Third, we evaluate the performance impact of PRAC and compare it against prior works using Ramulator 2.0. Our analysis shows that while PRAC incurs less than 13% performance overhead for today's DRAM chips, its performance overheads can reach up to 94% for future DRAM chips that are more vulnerable to read disturbance bitflips. Fourth, we define an availability adversarial access pattern that exacerbates PRAC's performance overhead to perform a memory performance attack, demonstrating that such an adversarial pattern can hog up to 94% of DRAM throughput and degrade system throughput by up to 95%. We discuss PRAC's implications on future systems and foreshadow future research directions. To aid future research, we open-source our implementations and scripts at https://github.com/CMU-SAFARI/ramulator2.

摘要: 我们首次对JEDEC DDR5规范2024年4月更新中描述的最先进的片上DRAM读取干扰缓解方法-每行激活计数(PRAC)-进行了严格的安全、性能、能量和成本分析。与建议存储器控制器定期发出刷新管理(RFM)命令(为DRAM芯片提供执行刷新的时间)的现有技术不同，PRAC引入了新的退避信号。PRAC的退避信号从DRAM芯片传播到存储器控制器，并迫使存储器控制器1)停止服务请求和2)发出RFM命令。因此，RFM命令在需要时发出，而不是定期发出，从而减少了RFM的管理费用。我们分四个步骤对PRAC进行分析。首先，我们定义了一种对抗性访问模式，它代表了对PRAC安全的最坏情况。其次，我们调查了PRAC的配置和安全影响。我们的分析表明，只要在访问一个存储单元10次之前没有发生位翻转，就可以将PRAC配置为安全操作。第三，我们评估了PRAC对性能的影响，并将其与使用Ramuler2.0的前人工作进行了比较。我们的分析表明，虽然PRAC对今天的DRAM芯片产生的性能开销不到13%，但对于更容易受到读取干扰位翻转的未来DRAM芯片，其性能开销可能高达94%。第四，我们定义了一种可用性对抗性访问模式，它加剧了PRAC执行内存性能攻击的性能开销，证明了这种对抗性模式可以占用高达94%的DRAM吞吐量，并使系统吞吐量降低高达95%。我们讨论了PRAC对未来系统的影响，并预示了未来的研究方向。为了帮助未来的研究，我们在https://github.com/CMU-SAFARI/ramulator2.上开放了我们的实现和脚本



## **38. Chernoff Information as a Privacy Constraint for Adversarial Classification**

作为对抗性分类的隐私约束的删除信息 cs.IT

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2403.10307v2) [paper-pdf](http://arxiv.org/pdf/2403.10307v2)

**Authors**: Ayşe Ünsal, Melek Önen

**Abstract**: This work inspects a privacy metric based on Chernoff information, \textit{Chernoff differential privacy}, due to its significance in characterization of the optimal classifier's performance. Adversarial classification, as any other classification problem is built around minimization of the (average or correct detection) probability of error in deciding on either of the classes in the case of binary classification. Unlike the classical hypothesis testing problem, where the false alarm and mis-detection probabilities are handled separately resulting in an asymmetric behavior of the best error exponent, in this work, we focus on the Bayesian setting and characterize the relationship between the best error exponent of the average error probability and $\varepsilon\textrm{-}$differential privacy \cite{D06}. Accordingly, we re-derive Chernoff differential privacy in terms of $\varepsilon\textrm{-}$differential privacy using the Radon-Nikodym derivative and show that it satisfies the composition property for sequential composition. Subsequently, we present numerical evaluation results, which demonstrates that Chernoff information outperforms Kullback-Leibler divergence as a function of the privacy parameter $\varepsilon$, the impact of the adversary's attack and global sensitivity for the problem of adversarial classification in Laplace mechanisms.

摘要: 基于Chernoff信息的隐私度量对抗性分类，因为任何其他分类问题都建立在最小化(平均或正确检测)错误概率的基础上，在二进制分类的情况下，决定其中一个类别的错误概率。与经典假设检验问题不同，在经典假设检验问题中，虚警概率和误检概率被分开处理，导致最佳错误指数的非对称行为。在该工作中，我们关注贝叶斯设置，并刻画了平均错误概率的最佳错误指数与差分隐私{D06}之间的关系。相应地，我们利用Radon-Nikodym导数将Chernoff差分隐私重新推导为$varepsilon差分隐私，并证明它满足序列合成的合成性质。随后，我们给出了数值评估结果，结果表明，Chernoff信息优于Kullback-Leibler发散，它是隐私参数$varepsilon$、对手攻击的影响和全局敏感度的函数。



## **39. Is the Digital Forensics and Incident Response Pipeline Ready for Text-Based Threats in LLM Era?**

数字取证和事件响应管道是否准备好应对LLM时代的基于文本的威胁？ cs.CR

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17870v1) [paper-pdf](http://arxiv.org/pdf/2407.17870v1)

**Authors**: Avanti Bhandarkar, Ronald Wilson, Anushka Swarup, Mengdi Zhu, Damon Woodard

**Abstract**: In the era of generative AI, the widespread adoption of Neural Text Generators (NTGs) presents new cybersecurity challenges, particularly within the realms of Digital Forensics and Incident Response (DFIR). These challenges primarily involve the detection and attribution of sources behind advanced attacks like spearphishing and disinformation campaigns. As NTGs evolve, the task of distinguishing between human and NTG-authored texts becomes critically complex. This paper rigorously evaluates the DFIR pipeline tailored for text-based security systems, specifically focusing on the challenges of detecting and attributing authorship of NTG-authored texts. By introducing a novel human-NTG co-authorship text attack, termed CS-ACT, our study uncovers significant vulnerabilities in traditional DFIR methodologies, highlighting discrepancies between ideal scenarios and real-world conditions. Utilizing 14 diverse datasets and 43 unique NTGs, up to the latest GPT-4, our research identifies substantial vulnerabilities in the forensic profiling phase, particularly in attributing authorship to NTGs. Our comprehensive evaluation points to factors such as model sophistication and the lack of distinctive style within NTGs as significant contributors for these vulnerabilities. Our findings underscore the necessity for more sophisticated and adaptable strategies, such as incorporating adversarial learning, stylizing NTGs, and implementing hierarchical attribution through the mapping of NTG lineages to enhance source attribution. This sets the stage for future research and the development of more resilient text-based security systems.

摘要: 在生成式人工智能时代，神经文本生成器(NTGs)的广泛采用带来了新的网络安全挑战，特别是在数字取证和事件响应(DFIR)领域。这些挑战主要涉及对鱼叉式网络钓鱼和虚假信息运动等高级攻击背后的来源进行检测和归类。随着NTG的发展，区分人类和NTG创作的文本的任务变得极其复杂。本文严格评估了为基于文本的安全系统量身定做的DFIR管道，特别关注了NTG创作的文本的作者身份检测和归属方面的挑战。通过引入一种名为CS-ACT的新型人-NTG合作文本攻击，我们的研究揭示了传统DFIR方法中的重大漏洞，突出了理想场景和现实世界条件之间的差异。利用14个不同的数据集和43个独特的NTGs，直到最新的GPT-4，我们的研究发现了法医侧写阶段的重大漏洞，特别是在将作者归因于NTGs方面。我们的综合评估指出，模型的复杂性和NTG内部缺乏独特的风格等因素是导致这些漏洞的重要因素。我们的发现强调了更复杂和适应性更强的策略的必要性，例如纳入对抗性学习，风格化的NTGs，以及通过NTG谱系的映射来实现分层归因以增强来源归因。这为未来的研究和开发更具弹性的基于文本的安全系统奠定了基础。



## **40. Domain Generalized Recaptured Screen Image Identification Using SWIN Transformer**

使用SWIN Transformer的域广义重捕获屏幕图像识别 cs.CV

11 pages, 10 figures, 9 tables

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17170v2) [paper-pdf](http://arxiv.org/pdf/2407.17170v2)

**Authors**: Preeti Mehta, Aman Sagar, Suchi Kumari

**Abstract**: An increasing number of classification approaches have been developed to address the issue of image rebroadcast and recapturing, a standard attack strategy in insurance frauds, face spoofing, and video piracy. However, most of them neglected scale variations and domain generalization scenarios, performing poorly in instances involving domain shifts, typically made worse by inter-domain and cross-domain scale variances. To overcome these issues, we propose a cascaded data augmentation and SWIN transformer domain generalization framework (DAST-DG) in the current research work Initially, we examine the disparity in dataset representation. A feature generator is trained to make authentic images from various domains indistinguishable. This process is then applied to recaptured images, creating a dual adversarial learning setup. Extensive experiments demonstrate that our approach is practical and surpasses state-of-the-art methods across different databases. Our model achieves an accuracy of approximately 82\% with a precision of 95\% on high-variance datasets.

摘要: 已经开发了越来越多的分类方法来解决图像重播和重新捕获的问题，这是保险欺诈、面部欺骗和视频盗版中的一种标准攻击策略。然而，它们中的大多数忽略了尺度变化和域泛化情景，在涉及域移动的情况下表现不佳，通常由于域间和跨域的尺度差异而变得更糟。为了克服这些问题，我们提出了一个级联数据增强和Swin变换器域泛化框架(DAST-DG)。在当前的研究工作中，我们首先检查了数据集表示上的差异。特征生成器被训练成使来自不同领域的真实图像无法区分。然后，这个过程被应用于重新捕获的图像，创建了一个双重对抗性学习设置。大量的实验表明，我们的方法是实用的，并且在不同的数据库上超过了最先进的方法。我们的模型在高方差数据集上达到了约82的精度和95的精度。



## **41. A Unified Understanding of Adversarial Vulnerability Regarding Unimodal Models and Vision-Language Pre-training Models**

统一理解关于单峰模型和视觉语言预训练模型的对抗脆弱性 cs.CV

14 pages, 9 figures, published in ACMMM2024(oral)

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17797v1) [paper-pdf](http://arxiv.org/pdf/2407.17797v1)

**Authors**: Haonan Zheng, Xinyang Deng, Wen Jiang, Wenrui Li

**Abstract**: With Vision-Language Pre-training (VLP) models demonstrating powerful multimodal interaction capabilities, the application scenarios of neural networks are no longer confined to unimodal domains but have expanded to more complex multimodal V+L downstream tasks. The security vulnerabilities of unimodal models have been extensively examined, whereas those of VLP models remain challenging. We note that in CV models, the understanding of images comes from annotated information, while VLP models are designed to learn image representations directly from raw text. Motivated by this discrepancy, we developed the Feature Guidance Attack (FGA), a novel method that uses text representations to direct the perturbation of clean images, resulting in the generation of adversarial images. FGA is orthogonal to many advanced attack strategies in the unimodal domain, facilitating the direct application of rich research findings from the unimodal to the multimodal scenario. By appropriately introducing text attack into FGA, we construct Feature Guidance with Text Attack (FGA-T). Through the interaction of attacking two modalities, FGA-T achieves superior attack effects against VLP models. Moreover, incorporating data augmentation and momentum mechanisms significantly improves the black-box transferability of FGA-T. Our method demonstrates stable and effective attack capabilities across various datasets, downstream tasks, and both black-box and white-box settings, offering a unified baseline for exploring the robustness of VLP models.

摘要: 随着视觉语言预训练模型显示出强大的多通道交互能力，神经网络的应用场景不再局限于单通道领域，而是扩展到更复杂的多通道V+L下游任务。单峰模型的安全漏洞已经被广泛研究，而VLP模型的安全漏洞仍然具有挑战性。我们注意到，在CV模型中，对图像的理解来自于注释信息，而VLP模型被设计为直接从原始文本学习图像表示。基于这种差异，我们提出了特征引导攻击(FGA)，这是一种新的方法，它使用文本表示来引导干净图像的扰动，从而产生对抗性图像。FGA与单模领域的许多高级攻击策略是正交的，便于将丰富的研究成果直接应用于从单模到多模的场景。通过在模糊遗传算法中适当地引入文本攻击，构造了基于文本攻击的特征引导算法(FGA-T)。通过两种攻击模式的交互作用，FGA-T对VLP模型取得了优越的攻击效果。此外，结合数据增强和动量机制显著提高了FGA-T的黑盒可转移性。我们的方法在各种数据集、下游任务以及黑盒和白盒设置上展示了稳定和有效的攻击能力，为探索VLP模型的稳健性提供了统一的基线。



## **42. Exploring Semantic Perturbations on Grover**

探索Grover的语义扰动 cs.LG

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2302.00509v2) [paper-pdf](http://arxiv.org/pdf/2302.00509v2)

**Authors**: Ziqing Ji, Pranav Kulkarni, Marko Neskovic, Kevin Nolan, Yan Xu

**Abstract**: With news and information being as easy to access as they currently are, it is more important than ever to ensure that people are not mislead by what they read. Recently, the rise of neural fake news (AI-generated fake news) and its demonstrated effectiveness at fooling humans has prompted the development of models to detect it. One such model is the Grover model, which can both detect neural fake news to prevent it, and generate it to demonstrate how a model could be misused to fool human readers. In this work we explore the Grover model's fake news detection capabilities by performing targeted attacks through perturbations on input news articles. Through this we test Grover's resilience to these adversarial attacks and expose some potential vulnerabilities which should be addressed in further iterations to ensure it can detect all types of fake news accurately.

摘要: 随着新闻和信息像现在一样容易获取，确保人们不被所读内容误导比以往任何时候都更加重要。最近，神经假新闻（人工智能生成的假新闻）的兴起及其在欺骗人类方面所表现出的有效性促使了检测它的模型的开发。其中一个模型是Grover模型，它既可以检测神经假新闻以防止它，又可以生成它来演示模型如何被滥用来欺骗人类读者。在这项工作中，我们通过对输入新闻文章的扰动进行有针对性的攻击来探索Grover模型的假新闻检测能力。通过此，我们测试了Grover对这些对抗攻击的弹性，并暴露了一些潜在的漏洞，这些漏洞应该在进一步的迭代中解决，以确保它能够准确地检测所有类型的假新闻。



## **43. Explaining the Model, Protecting Your Data: Revealing and Mitigating the Data Privacy Risks of Post-Hoc Model Explanations via Membership Inference**

解释模型，保护您的数据：通过会员资格推断揭示和缓解事后模型简化的数据隐私风险 cs.CR

ICML 2024 Workshop on the Next Generation of AI Safety

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17663v1) [paper-pdf](http://arxiv.org/pdf/2407.17663v1)

**Authors**: Catherine Huang, Martin Pawelczyk, Himabindu Lakkaraju

**Abstract**: Predictive machine learning models are becoming increasingly deployed in high-stakes contexts involving sensitive personal data; in these contexts, there is a trade-off between model explainability and data privacy. In this work, we push the boundaries of this trade-off: with a focus on foundation models for image classification fine-tuning, we reveal unforeseen privacy risks of post-hoc model explanations and subsequently offer mitigation strategies for such risks. First, we construct VAR-LRT and L1/L2-LRT, two new membership inference attacks based on feature attribution explanations that are significantly more successful than existing explanation-leveraging attacks, particularly in the low false-positive rate regime that allows an adversary to identify specific training set members with confidence. Second, we find empirically that optimized differentially private fine-tuning substantially diminishes the success of the aforementioned attacks, while maintaining high model accuracy. We carry out a systematic empirical investigation of our 2 new attacks with 5 vision transformer architectures, 5 benchmark datasets, 4 state-of-the-art post-hoc explanation methods, and 4 privacy strength settings.

摘要: 预测性机器学习模型越来越多地部署在涉及敏感个人数据的高风险环境中；在这些环境中，模型的可解释性和数据隐私之间存在权衡。在这项工作中，我们突破了这种权衡的界限：将重点放在图像分类微调的基础模型上，揭示后自组织模型解释的不可预见的隐私风险，并随后提供此类风险的缓解策略。首先，我们构造了VAR-LRT和L1/L2-LRT，这是两种新的基于特征属性解释的成员推理攻击，它们比现有的解释杠杆攻击要成功得多，特别是在允许对手自信地识别特定训练集成员的低假阳性率机制下。其次，我们从经验上发现，在保持较高模型精度的同时，优化的差分私有微调显著降低了上述攻击的成功率。我们使用5个视觉转换器架构、5个基准数据集、4个最先进的后自组织解释方法和4个隐私强度设置对我们的2个新攻击进行了系统的经验研究。



## **44. Revising the Problem of Partial Labels from the Perspective of CNNs' Robustness**

从CNN的稳健性角度修正部分标签问题 cs.CV

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17630v1) [paper-pdf](http://arxiv.org/pdf/2407.17630v1)

**Authors**: Xin Zhang, Yuqi Song, Wyatt McCurdy, Xiaofeng Wang, Fei Zuo

**Abstract**: Convolutional neural networks (CNNs) have gained increasing popularity and versatility in recent decades, finding applications in diverse domains. These remarkable achievements are greatly attributed to the support of extensive datasets with precise labels. However, annotating image datasets is intricate and complex, particularly in the case of multi-label datasets. Hence, the concept of partial-label setting has been proposed to reduce annotation costs, and numerous corresponding solutions have been introduced. The evaluation methods for these existing solutions have been primarily based on accuracy. That is, their performance is assessed by their predictive accuracy on the test set. However, we insist that such an evaluation is insufficient and one-sided. On one hand, since the quality of the test set has not been evaluated, the assessment results are unreliable. On the other hand, the partial-label problem may also be raised by undergoing adversarial attacks. Therefore, incorporating robustness into the evaluation system is crucial. For this purpose, we first propose two attack models to generate multiple partial-label datasets with varying degrees of label missing rates. Subsequently, we introduce a lightweight partial-label solution using pseudo-labeling techniques and a designed loss function. Then, we employ D-Score to analyze both the proposed and existing methods to determine whether they can enhance robustness while improving accuracy. Extensive experimental results demonstrate that while certain methods may improve accuracy, the enhancement in robustness is not significant, and in some cases, it even diminishes.

摘要: 卷积神经网络(CNN)在近几十年来得到了越来越广泛的应用，在不同的领域得到了广泛的应用。这些显著的成就在很大程度上归功于具有精确标签的大量数据集的支持。然而，标注图像数据集是复杂和复杂的，特别是在多标签数据集的情况下。因此，为了降低标注代价，人们提出了部分标注设置的概念，并提出了许多相应的解决方案。对这些现有解决方案的评估方法主要是基于准确性。也就是说，他们的表现是通过他们在测试集上的预测准确性来评估的。然而，我们坚持认为，这样的评估是不充分和片面的。一方面，由于测试集的质量没有得到评估，评估结果不可靠。另一方面，部分标签问题也可能通过经历对抗性攻击而引起。因此，将稳健性纳入评估体系至关重要。为此，我们首先提出了两种攻击模型来生成具有不同程度标签缺失率的多个部分标签数据集。随后，我们介绍了一种使用伪标记技术和设计的损失函数的轻量级部分标记解决方案。然后，我们使用D-SCORE对提出的方法和现有的方法进行分析，以确定它们是否可以在提高准确率的同时增强稳健性。大量的实验结果表明，虽然某些方法可以提高准确率，但在稳健性方面的增强并不显著，在某些情况下，甚至会减弱。



## **45. Fluent Student-Teacher Redteaming**

流利的师生红团队 cs.CL

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17447v1) [paper-pdf](http://arxiv.org/pdf/2407.17447v1)

**Authors**: T. Ben Thompson, Michael Sklar

**Abstract**: Many publicly available language models have been safety tuned to reduce the likelihood of toxic or liability-inducing text. Users or security analysts attempt to jailbreak or redteam these models with adversarial prompts which cause compliance with requests. One attack method is to apply discrete optimization techniques to the prompt. However, the resulting attack strings are often gibberish text, easily filtered by defenders due to high measured perplexity, and may fail for unseen tasks and/or well-tuned models. In this work, we improve existing algorithms (primarily GCG and BEAST) to develop powerful and fluent attacks on safety-tuned models like Llama-2 and Phi-3. Our technique centers around a new distillation-based approach that encourages the victim model to emulate a toxified finetune, either in terms of output probabilities or internal activations. To encourage human-fluent attacks, we add a multi-model perplexity penalty and a repetition penalty to the objective. We also enhance optimizer strength by allowing token insertions, token swaps, and token deletions and by using longer attack sequences. The resulting process is able to reliably jailbreak the most difficult target models with prompts that appear similar to human-written prompts. On Advbench we achieve attack success rates $>93$% for Llama-2-7B, Llama-3-8B, and Vicuna-7B, while maintaining model-measured perplexity $<33$; we achieve $95$% attack success for Phi-3, though with higher perplexity. We also find a universally-optimized single fluent prompt that induces $>88$% compliance on previously unseen tasks across Llama-2-7B, Phi-3-mini and Vicuna-7B and transfers to other black-box models.

摘要: 许多公开提供的语言模型都经过了安全调整，以减少有毒或导致责任的文本的可能性。用户或安全分析师试图用敌意提示对这些模型进行越狱或编辑，从而导致遵守请求。一种攻击方法是对提示应用离散优化技术。然而，由此产生的攻击字符串通常是胡言乱语的文本，由于高度测量的困惑，很容易被防御者过滤，并且可能无法完成看不见的任务和/或良好调整的模型。在这项工作中，我们改进了现有的算法(主要是GCG和BEAST)，以开发针对Llama-2和Phi-3等安全调整模型的强大而流畅的攻击。我们的技术以一种新的基于蒸馏的方法为中心，该方法鼓励受害者模型在输出概率或内部激活方面模仿中毒的微调。为了鼓励人类流利的攻击，我们在目标上增加了多模式困惑惩罚和重复惩罚。我们还通过允许令牌插入、令牌交换和令牌删除以及使用更长的攻击序列来增强优化器的强度。由此产生的过程能够可靠地用看起来类似于人写的提示的提示越狱最困难的目标模型。在Advbench上，我们实现了骆驼-2-7B、骆驼-3-8B和维库纳-7B的攻击成功率$>93$%，同时保持了模型测量的困惑$<33$；我们为Phi-3实现了$95$%的攻击成功率，尽管困惑程度更高。我们还发现了一个普遍优化的单一流畅提示，在Llama-2-7B、Phi-3-mini和Vicuna-7B上导致以前未见过的任务的合规性>88$%，并转移到其他黑盒型号。



## **46. Physical Adversarial Attack on Monocular Depth Estimation via Shape-Varying Patches**

通过形状变化贴片对单眼深度估计的物理对抗攻击 cs.CV

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17312v1) [paper-pdf](http://arxiv.org/pdf/2407.17312v1)

**Authors**: Chenxing Zhao, Yang Li, Shihao Wu, Wenyi Tan, Shuangju Zhou, Quan Pan

**Abstract**: Adversarial attacks against monocular depth estimation (MDE) systems pose significant challenges, particularly in safety-critical applications such as autonomous driving. Existing patch-based adversarial attacks for MDE are confined to the vicinity of the patch, making it difficult to affect the entire target. To address this limitation, we propose a physics-based adversarial attack on monocular depth estimation, employing a framework called Attack with Shape-Varying Patches (ASP), aiming to optimize patch content, shape, and position to maximize effectiveness. We introduce various mask shapes, including quadrilateral, rectangular, and circular masks, to enhance the flexibility and efficiency of the attack. Furthermore, we propose a new loss function to extend the influence of the patch beyond the overlapping regions. Experimental results demonstrate that our attack method generates an average depth error of 18 meters on the target car with a patch area of 1/9, affecting over 98\% of the target area.

摘要: 针对单目深度估计(MDE)系统的对抗性攻击带来了巨大的挑战，特别是在自动驾驶等安全关键应用中。现有的针对MDE的基于补丁的对抗性攻击仅限于补丁附近，难以影响整个目标。针对这一局限性，我们提出了一种基于物理的对抗性单眼深度估计攻击方法，采用了一种称为形状变化补丁攻击(ASP)的框架，旨在优化补丁的内容、形状和位置以最大化效果。我们引入了各种掩码形状，包括四边形、矩形和圆形掩码，以增强攻击的灵活性和效率。此外，我们还提出了一种新的损失函数，将斑块的影响扩展到重叠区域之外。实验结果表明，我们的攻击方法对目标车的平均深度误差为18m，补丁面积为1/9，影响了98%以上的目标区域。



## **47. Learning to Transform Dynamically for Better Adversarial Transferability**

学习动态转型以获得更好的对抗可移植性 cs.CV

accepted as a poster in CVPR 2024

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2405.14077v2) [paper-pdf](http://arxiv.org/pdf/2405.14077v2)

**Authors**: Rongyi Zhu, Zeliang Zhang, Susan Liang, Zhuo Liu, Chenliang Xu

**Abstract**: Adversarial examples, crafted by adding perturbations imperceptible to humans, can deceive neural networks. Recent studies identify the adversarial transferability across various models, \textit{i.e.}, the cross-model attack ability of adversarial samples. To enhance such adversarial transferability, existing input transformation-based methods diversify input data with transformation augmentation. However, their effectiveness is limited by the finite number of available transformations. In our study, we introduce a novel approach named Learning to Transform (L2T). L2T increases the diversity of transformed images by selecting the optimal combination of operations from a pool of candidates, consequently improving adversarial transferability. We conceptualize the selection of optimal transformation combinations as a trajectory optimization problem and employ a reinforcement learning strategy to effectively solve the problem. Comprehensive experiments on the ImageNet dataset, as well as practical tests with Google Vision and GPT-4V, reveal that L2T surpasses current methodologies in enhancing adversarial transferability, thereby confirming its effectiveness and practical significance. The code is available at https://github.com/RongyiZhu/L2T.

摘要: 通过添加人类察觉不到的扰动而精心制作的对抗性例子可以欺骗神经网络。最近的研究发现了各种模型之间的对抗性转移，即对抗性样本的跨模型攻击能力。为了增强这种对抗性的可转移性，现有的基于输入变换的方法通过变换增强来使输入数据多样化。然而，它们的有效性受到可用变换数量有限的限制。在我们的研究中，我们引入了一种名为学习转化(L2T)的新方法。L2T通过从候选集合中选择最优的操作组合来增加变换图像的多样性，从而提高了对抗性转移。我们将最优变换组合的选择概念化为一个轨迹优化问题，并采用强化学习策略来有效地解决该问题。在ImageNet数据集上的综合实验以及与Google Vision和GPT-4V的实际测试表明，L2T在增强对抗性可转移性方面优于现有方法，从而证实了其有效性和现实意义。代码可在https://github.com/RongyiZhu/L2T.上获得



## **48. When AI Defeats Password Deception! A Deep Learning Framework to Distinguish Passwords and Honeywords**

当人工智能击败密码欺骗！区分密码和蜜语的深度学习框架 cs.CR

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.16964v1) [paper-pdf](http://arxiv.org/pdf/2407.16964v1)

**Authors**: Jimmy Dani, Brandon McCulloh, Nitesh Saxena

**Abstract**: "Honeywords" have emerged as a promising defense mechanism for detecting data breaches and foiling offline dictionary attacks (ODA) by deceiving attackers with false passwords. In this paper, we propose PassFilter, a novel deep learning (DL) based attack framework, fundamental in its ability to identify passwords from a set of sweetwords associated with a user account, effectively challenging a variety of honeywords generation techniques (HGTs). The DL model in PassFilter is trained with a set of previously collected or adversarially generated passwords and honeywords, and carefully orchestrated to predict whether a sweetword is the password or a honeyword. Our model can compromise the security of state-of-the-art, heuristics-based, and representation learning-based HGTs proposed by Dionysiou et al. Specifically, our analysis with nine publicly available password datasets shows that PassFilter significantly outperforms the baseline random guessing success rate of 5%, achieving 6.10% to 52.78% on the 1st guessing attempt, considering 20 sweetwords per account. This success rate rapidly increases with additional login attempts before account lock-outs, often allowed on many real-world online services to maintain reasonable usability. For example, it ranges from 41.78% to 96.80% for five attempts, and from 72.87% to 99.00% for ten attempts, compared to 25% and 50% random guessing, respectively. We also examined PassFilter against general-purpose language models used for honeyword generation, like those proposed by Yu et al. These honeywords also proved vulnerable to our attack, with success rates of 14.19% for 1st guessing attempt, increasing to 30.23%, 41.70%, and 63.10% after 3rd, 5th, and 10th guessing attempts, respectively. Our findings demonstrate the effectiveness of DL model deployed in PassFilter in breaching state-of-the-art HGTs and compromising password security based on ODA.

摘要: “蜜字”已经成为一种很有前途的防御机制，可以通过用虚假密码欺骗攻击者来检测数据泄露和挫败离线词典攻击(Oda)。在本文中，我们提出了一种新的基于深度学习的攻击框架PassFilter，其基本特征是能够从与用户帐户关联的一组甜言蜜语中识别密码，有效地挑战了各种蜜语生成技术(HGT)。PassFilter中的DL模型使用一组先前收集的或恶意生成的密码和蜜字进行训练，并精心编排以预测甜言蜜语是密码还是蜜语。我们的模型可能会危及Dionsiou等人提出的最新的、基于启发式的和基于表示学习的HGT的安全性。具体地说，我们对9个公开可用的密码数据集的分析表明，PassFilter的性能显著高于5%的基线随机猜测成功率，考虑到每个帐户20个甜言蜜语，第一次猜测的成功率为6.10%到52.78%。这一成功率随着帐户锁定之前的额外登录尝试而迅速增加，这在许多现实世界的在线服务中通常是允许的，以保持合理的可用性。例如，5次尝试的命中率从41.78%到96.80%，10次尝试的命中率从72.87%到99.00%，而随机猜测的命中率分别为25%和50%。我们还将PassFilter与用于蜜词生成的通用语言模型进行了对比，如Yu等人提出的模型。这些蜜语也容易受到我们的攻击，第一次猜测的成功率为14.19%，第三次、第五次和第十次的猜测成功率分别增加到30.23%、41.70%和63.10%。我们的发现证明了在PassFilter中部署的DL模型在破解最新的HGTS和基于oda的口令安全性方面的有效性。



## **49. RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content**

RigorLLM：针对不需要内容的大型语言模型的弹性护栏 cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2403.13031v2) [paper-pdf](http://arxiv.org/pdf/2403.13031v2)

**Authors**: Zhuowen Yuan, Zidi Xiong, Yi Zeng, Ning Yu, Ruoxi Jia, Dawn Song, Bo Li

**Abstract**: Recent advancements in Large Language Models (LLMs) have showcased remarkable capabilities across various tasks in different domains. However, the emergence of biases and the potential for generating harmful content in LLMs, particularly under malicious inputs, pose significant challenges. Current mitigation strategies, while effective, are not resilient under adversarial attacks. This paper introduces Resilient Guardrails for Large Language Models (RigorLLM), a novel framework designed to efficiently and effectively moderate harmful and unsafe inputs and outputs for LLMs. By employing a multi-faceted approach that includes energy-based training data augmentation through Langevin dynamics, optimizing a safe suffix for inputs via minimax optimization, and integrating a fusion-based model combining robust KNN with LLMs based on our data augmentation, RigorLLM offers a robust solution to harmful content moderation. Our experimental evaluations demonstrate that RigorLLM not only outperforms existing baselines like OpenAI API and Perspective API in detecting harmful content but also exhibits unparalleled resilience to jailbreaking attacks. The innovative use of constrained optimization and a fusion-based guardrail approach represents a significant step forward in developing more secure and reliable LLMs, setting a new standard for content moderation frameworks in the face of evolving digital threats.

摘要: 大型语言模型(LLM)的最新进展展示了跨越不同领域的各种任务的显著能力。然而，偏见的出现和在低成本管理中产生有害内容的可能性，特别是在恶意投入下，构成了重大挑战。目前的缓解战略虽然有效，但在对抗性攻击下缺乏弹性。本文介绍了用于大型语言模型的弹性护栏(RigorLLM)，这是一个新的框架，旨在高效和有效地控制LLM中有害和不安全的输入和输出。通过采用多方面的方法，包括通过朗之万动力学基于能量的训练数据增强，通过极小极大优化优化输入的安全后缀，以及基于我们的数据增强将稳健的KNN与LLMS相结合的基于融合的模型，RigorLLM为有害内容适度提供了稳健的解决方案。我们的实验评估表明，RigorLLM不仅在检测有害内容方面优于OpenAI API和透视API等现有基线，而且对越狱攻击表现出无与伦比的弹性。约束优化和基于融合的护栏方法的创新使用代表着在开发更安全可靠的LLMS方面向前迈出的重要一步，为面对不断变化的数字威胁的内容审查框架设定了新的标准。



## **50. RedAgent: Red Teaming Large Language Models with Context-aware Autonomous Language Agent**

RedAgent：Red将大型语言模型与上下文感知自治语言代理结合起来 cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16667v1) [paper-pdf](http://arxiv.org/pdf/2407.16667v1)

**Authors**: Huiyu Xu, Wenhui Zhang, Zhibo Wang, Feng Xiao, Rui Zheng, Yunhe Feng, Zhongjie Ba, Kui Ren

**Abstract**: Recently, advanced Large Language Models (LLMs) such as GPT-4 have been integrated into many real-world applications like Code Copilot. These applications have significantly expanded the attack surface of LLMs, exposing them to a variety of threats. Among them, jailbreak attacks that induce toxic responses through jailbreak prompts have raised critical safety concerns. To identify these threats, a growing number of red teaming approaches simulate potential adversarial scenarios by crafting jailbreak prompts to test the target LLM. However, existing red teaming methods do not consider the unique vulnerabilities of LLM in different scenarios, making it difficult to adjust the jailbreak prompts to find context-specific vulnerabilities. Meanwhile, these methods are limited to refining jailbreak templates using a few mutation operations, lacking the automation and scalability to adapt to different scenarios. To enable context-aware and efficient red teaming, we abstract and model existing attacks into a coherent concept called "jailbreak strategy" and propose a multi-agent LLM system named RedAgent that leverages these strategies to generate context-aware jailbreak prompts. By self-reflecting on contextual feedback in an additional memory buffer, RedAgent continuously learns how to leverage these strategies to achieve effective jailbreaks in specific contexts. Extensive experiments demonstrate that our system can jailbreak most black-box LLMs in just five queries, improving the efficiency of existing red teaming methods by two times. Additionally, RedAgent can jailbreak customized LLM applications more efficiently. By generating context-aware jailbreak prompts towards applications on GPTs, we discover 60 severe vulnerabilities of these real-world applications with only two queries per vulnerability. We have reported all found issues and communicated with OpenAI and Meta for bug fixes.

摘要: 最近，GPT-4等高级大型语言模型(LLM)已集成到许多实际应用程序中，如Code Copilot。这些应用程序显著扩大了LLMS的攻击面，使它们暴露在各种威胁之下。其中，通过越狱提示引发有毒反应的越狱攻击引发了严重的安全问题。为了识别这些威胁，越来越多的红色团队方法通过精心编制越狱提示来测试目标LLM，以模拟潜在的敌对场景。然而，现有的红色团队方法没有考虑LLM在不同场景下的独特漏洞，很难调整越狱提示来发现上下文特定的漏洞。同时，这些方法仅限于使用少量的变异操作来提炼越狱模板，缺乏适应不同场景的自动化和可扩展性。为了实现上下文感知和高效的红色团队，我们将现有的攻击抽象并建模为一个连贯的概念，称为越狱策略，并提出了一个名为RedAgent的多代理LLM系统，该系统利用这些策略来生成上下文感知越狱提示。通过对额外内存缓冲区中的上下文反馈进行自我反思，RedAgent不断学习如何利用这些策略在特定上下文中实现有效的越狱。大量的实验表明，我们的系统可以在短短五次查询中破解大部分黑盒LLM，将现有的红色团队方法的效率提高了两倍。此外，RedAgent可以更高效地越狱定制的LLM应用程序。通过向GPT上的应用程序生成上下文感知越狱提示，我们发现了这些现实世界应用程序的60个严重漏洞，每个漏洞只有两个查询。我们已经报告了所有发现的问题，并与OpenAI和Meta进行了沟通以修复错误。



