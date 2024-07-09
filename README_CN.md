# Latest Adversarial Attack Papers
**update at 2024-07-09 10:23:40**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Improving Alignment and Robustness with Circuit Breakers**

改善断路器的对准和稳健性 cs.LG

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2406.04313v3) [paper-pdf](http://arxiv.org/pdf/2406.04313v3)

**Authors**: Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks

**Abstract**: AI systems can take harmful actions and are highly vulnerable to adversarial attacks. We present an approach, inspired by recent advances in representation engineering, that interrupts the models as they respond with harmful outputs with "circuit breakers." Existing techniques aimed at improving alignment, such as refusal training, are often bypassed. Techniques such as adversarial training try to plug these holes by countering specific attacks. As an alternative to refusal training and adversarial training, circuit-breaking directly controls the representations that are responsible for harmful outputs in the first place. Our technique can be applied to both text-only and multimodal language models to prevent the generation of harmful outputs without sacrificing utility -- even in the presence of powerful unseen attacks. Notably, while adversarial robustness in standalone image recognition remains an open challenge, circuit breakers allow the larger multimodal system to reliably withstand image "hijacks" that aim to produce harmful content. Finally, we extend our approach to AI agents, demonstrating considerable reductions in the rate of harmful actions when they are under attack. Our approach represents a significant step forward in the development of reliable safeguards to harmful behavior and adversarial attacks.

摘要: 人工智能系统可能采取有害行动，并且非常容易受到对抗性攻击。我们提出了一种方法，灵感来自于最近在表示工程方面的进展，该方法中断了模型，因为它们用“断路器”来响应有害的输出。旨在改善一致性的现有技术，如拒绝训练，经常被绕过。对抗性训练等技术试图通过反击特定攻击来堵塞这些漏洞。作为拒绝训练和对抗性训练的另一种选择，断路直接控制首先要对有害输出负责的陈述。我们的技术可以应用于纯文本和多模式语言模型，在不牺牲效用的情况下防止产生有害输出-即使在存在强大的看不见的攻击的情况下也是如此。值得注意的是，虽然独立图像识别中的对抗性健壮性仍然是一个开放的挑战，但断路器允许更大的多模式系统可靠地经受住旨在产生有害内容的图像“劫持”。最后，我们将我们的方法扩展到人工智能代理，表明当他们受到攻击时，有害行动的比率大大降低。我们的方法代表着在发展对有害行为和敌对攻击的可靠保障方面向前迈出了重要的一步。



## **2. Adaptive and robust watermark against model extraction attack**

抗模型提取攻击的自适应鲁棒水印 cs.CR

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2405.02365v2) [paper-pdf](http://arxiv.org/pdf/2405.02365v2)

**Authors**: Kaiyi Pang

**Abstract**: Large language models (LLMs) demonstrate general intelligence across a variety of machine learning tasks, thereby enhancing the commercial value of their intellectual property (IP). To protect this IP, model owners typically allow user access only in a black-box manner, however, adversaries can still utilize model extraction attacks to steal the model intelligence encoded in model generation. Watermarking technology offers a promising solution for defending against such attacks by embedding unique identifiers into the model-generated content. However, existing watermarking methods often compromise the quality of generated content due to heuristic alterations and lack robust mechanisms to counteract adversarial strategies, thus limiting their practicality in real-world scenarios. In this paper, we introduce an adaptive and robust watermarking method (named ModelShield) to protect the IP of LLMs. Our method incorporates a self-watermarking mechanism that allows LLMs to autonomously insert watermarks into their generated content to avoid the degradation of model content. We also propose a robust watermark detection mechanism capable of effectively identifying watermark signals under the interference of varying adversarial strategies. Besides, ModelShield is a plug-and-play method that does not require additional model training, enhancing its applicability in LLM deployments. Extensive evaluations on two real-world datasets and three LLMs demonstrate that our method surpasses existing methods in terms of defense effectiveness and robustness while significantly reducing the degradation of watermarking on the model-generated content.

摘要: 大型语言模型(LLM)在各种机器学习任务中展示了一般智能，从而提高了其知识产权(IP)的商业价值。为了保护这个IP，模型所有者通常只允许用户以黑盒方式访问，但是，攻击者仍然可以利用模型提取攻击来窃取模型生成中编码的模型情报。水印技术通过在模型生成的内容中嵌入唯一标识符，为防御此类攻击提供了一种很有前途的解决方案。然而，现有的水印方法往往会由于启发式修改而影响生成内容的质量，并且缺乏强大的机制来对抗对抗性策略，从而限制了它们在现实世界场景中的实用性。本文提出了一种自适应的稳健水印算法(ModelShield)来保护LLMS的IP地址。我们的方法结合了一种自水印机制，允许LLM自主地在其生成的内容中插入水印，以避免模型内容的降级。我们还提出了一种稳健的水印检测机制，能够在不同的对抗策略的干扰下有效地识别水印信号。此外，ModelShield是一种即插即用的方法，不需要额外的模型培训，增强了其在LLM部署中的适用性。在两个真实数据集和三个LLM上的广泛评估表明，我们的方法在防御有效性和稳健性方面优于现有方法，同时显着降低了水印对模型生成内容的退化。



## **3. Multi-View Black-Box Physical Attacks on Infrared Pedestrian Detectors Using Adversarial Infrared Grid**

使用对抗红外网格对红外行人探测器进行多视图黑匣子物理攻击 cs.CV

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2407.01168v2) [paper-pdf](http://arxiv.org/pdf/2407.01168v2)

**Authors**: Kalibinuer Tiliwalidi, Chengyin Hu, Weiwen Shi

**Abstract**: While extensive research exists on physical adversarial attacks within the visible spectrum, studies on such techniques in the infrared spectrum are limited. Infrared object detectors are vital in modern technological applications but are susceptible to adversarial attacks, posing significant security threats. Previous studies using physical perturbations like light bulb arrays and aerogels for white-box attacks, or hot and cold patches for black-box attacks, have proven impractical or limited in multi-view support. To address these issues, we propose the Adversarial Infrared Grid (AdvGrid), which models perturbations in a grid format and uses a genetic algorithm for black-box optimization. These perturbations are cyclically applied to various parts of a pedestrian's clothing to facilitate multi-view black-box physical attacks on infrared pedestrian detectors. Extensive experiments validate AdvGrid's effectiveness, stealthiness, and robustness. The method achieves attack success rates of 80.00\% in digital environments and 91.86\% in physical environments, outperforming baseline methods. Additionally, the average attack success rate exceeds 50\% against mainstream detectors, demonstrating AdvGrid's robustness. Our analyses include ablation studies, transfer attacks, and adversarial defenses, confirming the method's superiority.

摘要: 虽然在可见光光谱内对物理对抗攻击已有广泛的研究，但在红外光谱中对这类技术的研究有限。红外目标探测器在现代技术应用中至关重要，但容易受到对抗性攻击，构成重大安全威胁。以前的研究证明，使用物理扰动，如灯泡阵列和气凝胶进行白盒攻击，或使用冷热补丁进行黑盒攻击，都被证明是不切实际的，或者在多视角支持方面受到限制。为了解决这些问题，我们提出了对抗性红外网格(AdvGrid)，它以网格的形式对扰动进行建模，并使用遗传算法进行黑盒优化。这些扰动被循环应用于行人衣服的不同部分，以促进对红外行人探测器的多视角黑匣子物理攻击。大量实验验证了AdvGrid的有效性、隐蔽性和健壮性。该方法在数字环境下的攻击成功率为80.00%，在物理环境下的攻击成功率为91.86%，优于基准攻击方法。此外，对主流检测器的平均攻击成功率超过50%，显示了AdvGrid的健壮性。我们的分析包括烧蚀研究、转移攻击和对抗性防御，证实了该方法的优越性。



## **4. Malicious Agent Detection for Robust Multi-Agent Collaborative Perception**

用于鲁棒多代理协作感知的恶意代理检测 cs.CR

Accepted by IROS 2024

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2310.11901v2) [paper-pdf](http://arxiv.org/pdf/2310.11901v2)

**Authors**: Yangheng Zhao, Zhen Xiang, Sheng Yin, Xianghe Pang, Siheng Chen, Yanfeng Wang

**Abstract**: Recently, multi-agent collaborative (MAC) perception has been proposed and outperformed the traditional single-agent perception in many applications, such as autonomous driving. However, MAC perception is more vulnerable to adversarial attacks than single-agent perception due to the information exchange. The attacker can easily degrade the performance of a victim agent by sending harmful information from a malicious agent nearby. In this paper, we extend adversarial attacks to an important perception task -- MAC object detection, where generic defenses such as adversarial training are no longer effective against these attacks. More importantly, we propose Malicious Agent Detection (MADE), a reactive defense specific to MAC perception that can be deployed by each agent to accurately detect and then remove any potential malicious agent in its local collaboration network. In particular, MADE inspects each agent in the network independently using a semi-supervised anomaly detector based on a double-hypothesis test with the Benjamini-Hochberg procedure to control the false positive rate of the inference. For the two hypothesis tests, we propose a match loss statistic and a collaborative reconstruction loss statistic, respectively, both based on the consistency between the agent to be inspected and the ego agent where our detector is deployed. We conduct comprehensive evaluations on a benchmark 3D dataset V2X-sim and a real-road dataset DAIR-V2X and show that with the protection of MADE, the drops in the average precision compared with the best-case "oracle" defender against our attack are merely 1.28% and 0.34%, respectively, much lower than 8.92% and 10.00% for adversarial training, respectively.

摘要: 近年来，多智能体协作(MAC)感知被提出，并在许多应用中优于传统的单智能体感知，如自主驾驶。然而，由于信息的交换，MAC感知比单代理感知更容易受到敌意攻击。攻击者可以很容易地通过从附近的恶意代理发送有害信息来降低受害者代理的性能。在本文中，我们将对抗性攻击扩展到一项重要的感知任务--MAC对象检测，在这种情况下，对抗性训练等一般防御手段不再有效地对抗这些攻击。更重要的是，我们提出了恶意代理检测(Made)，这是一种针对MAC感知的反应性防御，可以由每个代理部署以准确检测并随后删除其本地协作网络中的任何潜在恶意代理。特别地，Made使用基于双假设检验的半监督异常检测器独立地检查网络中的每个代理，并结合Benjamini-Hochberg过程来控制推理的误检率。对于这两种假设检验，我们分别提出了一个匹配损失统计量和一个协作重建损失统计量，这两个统计量都是基于待检查代理和部署检测器的自我代理之间的一致性。我们在基准3D数据集V2X-SIM和真实道路数据集DAIR-V2X上进行了综合评估，结果表明，在Made的保护下，与最佳情况下的Oracle防御者相比，对抗我们的攻击的平均精度分别下降了1.28%和0.34%，远低于对抗性训练的8.92%和10.00%。



## **5. Formal Verification of Object Detection**

对象检测的形式化验证 cs.CV

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2407.01295v2) [paper-pdf](http://arxiv.org/pdf/2407.01295v2)

**Authors**: Avraham Raviv, Yizhak Y. Elboher, Michelle Aluf-Medina, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep Neural Networks (DNNs) are ubiquitous in real-world applications, yet they remain vulnerable to errors and adversarial attacks. This work tackles the challenge of applying formal verification to ensure the safety of computer vision models, extending verification beyond image classification to object detection. We propose a general formulation for certifying the robustness of object detection models using formal verification and outline implementation strategies compatible with state-of-the-art verification tools. Our approach enables the application of these tools, originally designed for verifying classification models, to object detection. We define various attacks for object detection, illustrating the diverse ways adversarial inputs can compromise neural network outputs. Our experiments, conducted on several common datasets and networks, reveal potential errors in object detection models, highlighting system vulnerabilities and emphasizing the need for expanding formal verification to these new domains. This work paves the way for further research in integrating formal verification across a broader range of computer vision applications.

摘要: 深度神经网络(DNN)在实际应用中无处不在，但它们仍然容易受到错误和对手攻击。这项工作解决了应用形式化验证来确保计算机视觉模型的安全性的挑战，将验证从图像分类扩展到目标检测。我们提出了使用形式化验证来证明目标检测模型的健壮性的一般公式，并概述了与最先进的验证工具兼容的实现策略。我们的方法使得这些最初设计用于验证分类模型的工具能够应用于目标检测。我们定义了用于目标检测的各种攻击，说明了敌意输入可以损害神经网络输出的不同方式。我们在几个常见的数据集和网络上进行的实验，揭示了对象检测模型中的潜在错误，突出了系统漏洞，并强调了将正式验证扩展到这些新领域的必要性。这项工作为在更广泛的计算机视觉应用中整合形式验证的进一步研究铺平了道路。



## **6. Exploring the Adversarial Capabilities of Large Language Models**

探索大型语言模型的对抗能力 cs.AI

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2402.09132v4) [paper-pdf](http://arxiv.org/pdf/2402.09132v4)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.

摘要: 大型语言模型因其强大的语言生成能力而引起了广泛的关注，为工业和研究提供了巨大的潜力。虽然之前的研究已经深入研究了LLMS的安全和隐私问题，但这些模型在多大程度上可以表现出敌对行为，仍然很大程度上还没有被探索。针对这一差距，我们调查了常见的公开可用的LLM是否具有固有的能力来扰乱文本样本以愚弄安全措施，即所谓的对抗性示例攻击。更具体地说，我们调查LLM是否天生就能够从良性样本中制作敌意示例，以愚弄现有的安全Rail。我们的实验集中在仇恨语音检测上，实验表明，LLMS成功地发现了敌意扰动，有效地破坏了仇恨语音检测系统。我们的发现对依赖LLMS的(半)自治系统具有重大影响，突显了它们与现有系统和安全措施相互作用的潜在挑战。



## **7. Improving Adversarial Transferability of Vision-Language Pre-training Models through Collaborative Multimodal Interaction**

通过协作多模式交互提高视觉语言预训练模型的对抗性可移植性 cs.CV

This work won first place in CVPR 2024 Workshop Challenge: Black-box  Adversarial Attacks on Vision Foundation Models

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2403.10883v2) [paper-pdf](http://arxiv.org/pdf/2403.10883v2)

**Authors**: Jiyuan Fu, Zhaoyu Chen, Kaixun Jiang, Haijing Guo, Jiafeng Wang, Shuyong Gao, Wenqiang Zhang

**Abstract**: Despite the substantial advancements in Vision-Language Pre-training (VLP) models, their susceptibility to adversarial attacks poses a significant challenge. Existing work rarely studies the transferability of attacks on VLP models, resulting in a substantial performance gap from white-box attacks. We observe that prior work overlooks the interaction mechanisms between modalities, which plays a crucial role in understanding the intricacies of VLP models. In response, we propose a novel attack, called Collaborative Multimodal Interaction Attack (CMI-Attack), leveraging modality interaction through embedding guidance and interaction enhancement. Specifically, attacking text at the embedding level while preserving semantics, as well as utilizing interaction image gradients to enhance constraints on perturbations of texts and images. Significantly, in the image-text retrieval task on Flickr30K dataset, CMI-Attack raises the transfer success rates from ALBEF to TCL, $\text{CLIP}_{\text{ViT}}$ and $\text{CLIP}_{\text{CNN}}$ by 8.11%-16.75% over state-of-the-art methods. Moreover, CMI-Attack also demonstrates superior performance in cross-task generalization scenarios. Our work addresses the underexplored realm of transfer attacks on VLP models, shedding light on the importance of modality interaction for enhanced adversarial robustness.

摘要: 尽管视觉语言预训练(VLP)模式有了很大的进步，但它们对对手攻击的敏感性构成了一个巨大的挑战。现有的工作很少研究攻击对VLP模型的可转移性，导致与白盒攻击相比性能有很大的差距。我们注意到，以前的工作忽略了通道之间的相互作用机制，这在理解VLP模型的复杂性方面起着至关重要的作用。对此，我们提出了一种新的攻击方法，称为协作多模式交互攻击(CMI-Attack)，通过嵌入引导和交互增强来利用通道交互。具体地说，在保留语义的同时在嵌入层攻击文本，以及利用交互图像梯度来增强对文本和图像扰动的约束。值得注意的是，在Flickr30K数据集的图文检索任务中，CMI-Attack将从ALBEF到TCL、$\Text{Clip}_{\Text{Vit}}$和$\Text{Clip}_{\Text{CNN}}$的传输成功率比最先进的方法提高了8.11%-16.75%。此外，CMI-Attack在跨任务泛化场景中也表现出了优越的性能。我们的工作解决了VLP模型上未被探索的传输攻击领域，揭示了通道交互对于增强对手健壮性的重要性。



## **8. A Survey of Fragile Model Watermarking**

脆弱模型水印综述 cs.CR

Submitted Signal Processing

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2406.04809v4) [paper-pdf](http://arxiv.org/pdf/2406.04809v4)

**Authors**: Zhenzhe Gao, Yu Cheng, Zhaoxia Yin

**Abstract**: Model fragile watermarking, inspired by both the field of adversarial attacks on neural networks and traditional multimedia fragile watermarking, has gradually emerged as a potent tool for detecting tampering, and has witnessed rapid development in recent years. Unlike robust watermarks, which are widely used for identifying model copyrights, fragile watermarks for models are designed to identify whether models have been subjected to unexpected alterations such as backdoors, poisoning, compression, among others. These alterations can pose unknown risks to model users, such as misidentifying stop signs as speed limit signs in classic autonomous driving scenarios. This paper provides an overview of the relevant work in the field of model fragile watermarking since its inception, categorizing them and revealing the developmental trajectory of the field, thus offering a comprehensive survey for future endeavors in model fragile watermarking.

摘要: 模型脆弱水印受到神经网络对抗攻击领域和传统多媒体脆弱水印的启发，逐渐成为检测篡改的有力工具，并在近年来得到了快速发展。与广泛用于识别模型版权的稳健水印不同，模型的脆弱水印旨在识别模型是否遭受了意外更改，例如后门、中毒、压缩等。这些更改可能会给模型用户带来未知的风险，例如在经典自动驾驶场景中将停车标志误识别为限速标志。本文概述了模型脆弱水印领域自诞生以来的相关工作，对其进行了分类，揭示了该领域的发展轨迹，从而为模型脆弱水印的未来工作提供了全面的综述。



## **9. To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now**

生成还是不生成？安全驱动的未学习扩散模型仍然很容易生成不安全的图像.现在 cs.CV

Accepted by ECCV'24. Codes are available at  https://github.com/OPTML-Group/Diffusion-MU-Attack

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2310.11868v4) [paper-pdf](http://arxiv.org/pdf/2310.11868v4)

**Authors**: Yimeng Zhang, Jinghan Jia, Xin Chen, Aochuan Chen, Yihua Zhang, Jiancheng Liu, Ke Ding, Sijia Liu

**Abstract**: The recent advances in diffusion models (DMs) have revolutionized the generation of realistic and complex images. However, these models also introduce potential safety hazards, such as producing harmful content and infringing data copyrights. Despite the development of safety-driven unlearning techniques to counteract these challenges, doubts about their efficacy persist. To tackle this issue, we introduce an evaluation framework that leverages adversarial prompts to discern the trustworthiness of these safety-driven DMs after they have undergone the process of unlearning harmful concepts. Specifically, we investigated the adversarial robustness of DMs, assessed by adversarial prompts, when eliminating unwanted concepts, styles, and objects. We develop an effective and efficient adversarial prompt generation approach for DMs, termed UnlearnDiffAtk. This method capitalizes on the intrinsic classification abilities of DMs to simplify the creation of adversarial prompts, thereby eliminating the need for auxiliary classification or diffusion models. Through extensive benchmarking, we evaluate the robustness of widely-used safety-driven unlearned DMs (i.e., DMs after unlearning undesirable concepts, styles, or objects) across a variety of tasks. Our results demonstrate the effectiveness and efficiency merits of UnlearnDiffAtk over the state-of-the-art adversarial prompt generation method and reveal the lack of robustness of current safetydriven unlearning techniques when applied to DMs. Codes are available at https://github.com/OPTML-Group/Diffusion-MU-Attack. WARNING: There exist AI generations that may be offensive in nature.

摘要: 扩散模型的最新进展使逼真和复杂图像的生成发生了革命性的变化。然而，这些模式也带来了潜在的安全隐患，如产生有害内容和侵犯数据著作权。尽管发展了安全驱动的遗忘技术来应对这些挑战，但对其有效性的怀疑依然存在。为了解决这个问题，我们引入了一个评估框架，利用对抗性提示，在这些以安全为导向的DM经历了忘记有害概念的过程后，识别他们的可信度。具体地说，我们研究了DM在消除不需要的概念、风格和对象时，通过对抗性提示评估的对抗性健壮性。本文提出了一种高效的敌意提示生成方法，称为UnlearnDiffAtk。这种方法利用DM的内在分类能力来简化对抗性提示的创建，从而消除了对辅助分类或扩散模型的需要。通过广泛的基准测试，我们评估了广泛使用的安全驱动的未学习DM(即在忘记不需要的概念、风格或对象后的DM)在各种任务中的健壮性。实验结果证明了UnlearnDiffAtk算法相对于最新的对抗性提示生成方法的有效性和高效性，并揭示了当前安全驱动的遗忘技术在应用于决策支持系统时存在的健壮性不足。有关代码，请访问https://github.com/OPTML-Group/Diffusion-MU-Attack.警告：存在可能具有攻击性的人工智能世代。



## **10. Rethinking Targeted Adversarial Attacks For Neural Machine Translation**

重新思考神经机器翻译的有针对性的对抗攻击 cs.CL

5 pages, 2 figures, accepted by ICASSP 2024

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2407.05319v1) [paper-pdf](http://arxiv.org/pdf/2407.05319v1)

**Authors**: Junjie Wu, Lemao Liu, Wei Bi, Dit-Yan Yeung

**Abstract**: Targeted adversarial attacks are widely used to evaluate the robustness of neural machine translation systems. Unfortunately, this paper first identifies a critical issue in the existing settings of NMT targeted adversarial attacks, where their attacking results are largely overestimated. To this end, this paper presents a new setting for NMT targeted adversarial attacks that could lead to reliable attacking results. Under the new setting, it then proposes a Targeted Word Gradient adversarial Attack (TWGA) method to craft adversarial examples. Experimental results demonstrate that our proposed setting could provide faithful attacking results for targeted adversarial attacks on NMT systems, and the proposed TWGA method can effectively attack such victim NMT systems. In-depth analyses on a large-scale dataset further illustrate some valuable findings. 1 Our code and data are available at https://github.com/wujunjie1998/TWGA.

摘要: 有针对性的对抗攻击被广泛用于评估神经机器翻译系统的鲁棒性。不幸的是，本文首先指出了NMT有针对性的对抗攻击现有环境中的一个关键问题，即它们的攻击结果在很大程度上被高估了。为此，本文为NMT有针对性的对抗攻击提供了一种新设置，可以带来可靠的攻击结果。在新的设置下，它随后提出了一种有针对性的词梯度对抗攻击（TWGA）方法来制作对抗示例。实验结果表明，我们提出的设置可以为针对NMT系统的有针对性的对抗攻击提供可靠的攻击结果，并且提出的TWGA方法可以有效地攻击此类受害NMT系统。对大规模数据集的深入分析进一步说明了一些有价值的发现。1我们的代码和数据可在https://github.com/wujunjie1998/TWGA上获取。



## **11. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models**

TrojanRAG：检索增强生成可以成为大型语言模型中的后门驱动程序 cs.CR

19 pages, 14 figures, 4 tables

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2405.13401v4) [paper-pdf](http://arxiv.org/pdf/2405.13401v4)

**Authors**: Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Large language models (LLMs) have raised concerns about potential security threats despite performing significantly in Natural Language Processing (NLP). Backdoor attacks initially verified that LLM is doing substantial harm at all stages, but the cost and robustness have been criticized. Attacking LLMs is inherently risky in security review, while prohibitively expensive. Besides, the continuous iteration of LLMs will degrade the robustness of backdoors. In this paper, we propose TrojanRAG, which employs a joint backdoor attack in the Retrieval-Augmented Generation, thereby manipulating LLMs in universal attack scenarios. Specifically, the adversary constructs elaborate target contexts and trigger sets. Multiple pairs of backdoor shortcuts are orthogonally optimized by contrastive learning, thus constraining the triggering conditions to a parameter subspace to improve the matching. To improve the recall of the RAG for the target contexts, we introduce a knowledge graph to construct structured data to achieve hard matching at a fine-grained level. Moreover, we normalize the backdoor scenarios in LLMs to analyze the real harm caused by backdoors from both attackers' and users' perspectives and further verify whether the context is a favorable tool for jailbreaking models. Extensive experimental results on truthfulness, language understanding, and harmfulness show that TrojanRAG exhibits versatility threats while maintaining retrieval capabilities on normal queries.

摘要: 尽管大型语言模型(LLM)在自然语言处理(NLP)中表现出色，但仍引发了人们对潜在安全威胁的担忧。后门攻击最初证实了LLM在所有阶段都在造成实质性的危害，但其成本和健壮性受到了批评。在安全审查中，攻击LLMS固有的风险，同时代价高得令人望而却步。此外，LLMS的连续迭代会降低后门的健壮性。在本文中，我们提出了TrojanRAG，它在检索-增强生成中使用联合后门攻击，从而在通用攻击场景下操纵LLMS。具体地说，对手构建了精心设计的目标上下文和触发集。通过对比学习对多对后门捷径进行正交化优化，从而将触发条件约束到一个参数子空间以提高匹配性。为了提高RAG对目标上下文的查全率，我们引入了知识图来构建结构化数据，以实现细粒度的硬匹配。此外，我们对LLMS中的后门场景进行了规范化，从攻击者和用户的角度分析了后门造成的真实危害，并进一步验证了上下文是否为越狱模型的有利工具。在真实性、语言理解和危害性方面的大量实验结果表明，TrojanRAG在保持对正常查询的检索能力的同时，表现出通用性威胁。



## **12. FedCG: Leverage Conditional GAN for Protecting Privacy and Maintaining Competitive Performance in Federated Learning**

FedCG：利用有条件GAN在联邦学习中保护隐私并保持竞争绩效 cs.LG

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2111.08211v3) [paper-pdf](http://arxiv.org/pdf/2111.08211v3)

**Authors**: Yuezhou Wu, Yan Kang, Jiahuan Luo, Yuanqin He, Qiang Yang

**Abstract**: Federated learning (FL) aims to protect data privacy by enabling clients to build machine learning models collaboratively without sharing their private data. Recent works demonstrate that information exchanged during FL is subject to gradient-based privacy attacks, and consequently, a variety of privacy-preserving methods have been adopted to thwart such attacks. However, these defensive methods either introduce orders of magnitude more computational and communication overheads (e.g., with homomorphic encryption) or incur substantial model performance losses in terms of prediction accuracy (e.g., with differential privacy). In this work, we propose $\textsc{FedCG}$, a novel federated learning method that leverages conditional generative adversarial networks to achieve high-level privacy protection while still maintaining competitive model performance. $\textsc{FedCG}$ decomposes each client's local network into a private extractor and a public classifier and keeps the extractor local to protect privacy. Instead of exposing extractors, $\textsc{FedCG}$ shares clients' generators with the server for aggregating clients' shared knowledge, aiming to enhance the performance of each client's local networks. Extensive experiments demonstrate that $\textsc{FedCG}$ can achieve competitive model performance compared with FL baselines, and privacy analysis shows that $\textsc{FedCG}$ has a high-level privacy-preserving capability. Code is available at https://github.com/yankang18/FedCG

摘要: 联合学习(FL)旨在通过使客户能够在不共享他们的私人数据的情况下协作地建立机器学习模型来保护数据隐私。最近的研究表明，在外语学习过程中交换的信息会受到基于梯度的隐私攻击，因此，人们已经采取了各种隐私保护方法来阻止这种攻击。然而，这些防御方法要么引入更多数量级的计算和通信开销(例如，使用同态加密)，要么在预测精度方面导致显著的模型性能损失(例如，使用差分隐私)。在这项工作中，我们提出了一种新的联邦学习方法，它利用条件生成对抗网络来实现高级别的隐私保护，同时又保持了竞争模型的性能。$\Textsc{FedCG}$将每个客户端的本地网络分解为私有提取程序和公共分类器，并将提取程序保留在本地以保护隐私。$\extsc{FedCG}$不公开提取程序，而是与服务器共享客户端的生成器，以聚合客户端共享的知识，旨在增强每个客户端的本地网络的性能。大量实验表明，与FL基线相比，$\extsc{FedCG}$具有与FL基线相当的模型性能，隐私分析表明，$\extsc{FedCG}$具有较高的隐私保护能力。代码可在https://github.com/yankang18/FedCG上找到



## **13. A Novel Bifurcation Method for Observation Perturbation Attacks on Reinforcement Learning Agents: Load Altering Attacks on a Cyber Physical Power System**

一种用于对强化学习代理进行观察扰动攻击的新型分歧方法：对网络物理电力系统的负载改变攻击 cs.LG

12 pages, 5 figures

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2407.05182v1) [paper-pdf](http://arxiv.org/pdf/2407.05182v1)

**Authors**: Kiernan Broda-Milian, Ranwa Al-Mallah, Hanane Dagdougui

**Abstract**: Components of cyber physical systems, which affect real-world processes, are often exposed to the internet. Replacing conventional control methods with Deep Reinforcement Learning (DRL) in energy systems is an active area of research, as these systems become increasingly complex with the advent of renewable energy sources and the desire to improve their efficiency. Artificial Neural Networks (ANN) are vulnerable to specific perturbations of their inputs or features, called adversarial examples. These perturbations are difficult to detect when properly regularized, but have significant effects on the ANN's output. Because DRL uses ANN to map optimal actions to observations, they are similarly vulnerable to adversarial examples. This work proposes a novel attack technique for continuous control using Group Difference Logits loss with a bifurcation layer. By combining aspects of targeted and untargeted attacks, the attack significantly increases the impact compared to an untargeted attack, with drastically smaller distortions than an optimally targeted attack. We demonstrate the impacts of powerful gradient-based attacks in a realistic smart energy environment, show how the impacts change with different DRL agents and training procedures, and use statistical and time-series analysis to evaluate attacks' stealth. The results show that adversarial attacks can have significant impacts on DRL controllers, and constraining an attack's perturbations makes it difficult to detect. However, certain DRL architectures are far more robust, and robust training methods can further reduce the impact.

摘要: 影响真实世界进程的网络物理系统的组件经常暴露在互联网上。在能源系统中用深度强化学习(DRL)取代传统的控制方法是一个活跃的研究领域，因为随着可再生能源的出现和提高其效率的愿望，这些系统变得越来越复杂。人工神经网络(ANN)容易受到其输入或特征的特定扰动，称为对抗性示例。当适当地正则化时，这些扰动很难被检测到，但会对神经网络的输出产生重大影响。由于DRL使用人工神经网络将最优动作映射到观测值，因此它们同样容易受到对手例子的攻击。本文提出了一种新的基于分组差值Logits损失的连续控制攻击技术。通过结合目标攻击和非目标攻击的各个方面，与非目标攻击相比，该攻击显著增加了影响，扭曲程度比最佳目标攻击小得多。我们展示了强大的基于梯度的攻击在现实的智能能源环境中的影响，展示了影响如何随不同的DRL代理和训练过程而变化，并使用统计和时间序列分析来评估攻击的隐蔽性。结果表明，对抗性攻击可以对DRL控制器产生显著影响，并且限制攻击的扰动使得检测变得困难。然而，某些DRL架构要健壮得多，健壮的训练方法可以进一步降低影响。



## **14. Robust Skin Color Driven Privacy Preserving Face Recognition via Function Secret Sharing**

通过功能秘密共享的鲁棒肤色驱动的隐私保护面部识别 cs.CV

Accepted at ICIP2024

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2407.05045v1) [paper-pdf](http://arxiv.org/pdf/2407.05045v1)

**Authors**: Dong Han, Yufan Jiang, Yong Li, Ricardo Mendes, Joachim Denzler

**Abstract**: In this work, we leverage the pure skin color patch from the face image as the additional information to train an auxiliary skin color feature extractor and face recognition model in parallel to improve performance of state-of-the-art (SOTA) privacy-preserving face recognition (PPFR) systems. Our solution is robust against black-box attacking and well-established generative adversarial network (GAN) based image restoration. We analyze the potential risk in previous work, where the proposed cosine similarity computation might directly leak the protected precomputed embedding stored on the server side. We propose a Function Secret Sharing (FSS) based face embedding comparison protocol without any intermediate result leakage. In addition, we show in experiments that the proposed protocol is more efficient compared to the Secret Sharing (SS) based protocol.

摘要: 在这项工作中，我们利用面部图像中的纯肤色补丁作为附加信息来并行训练辅助肤色特征提取器和面部识别模型，以提高最新技术水平（SOTA）隐私保护面部识别（PPFR）系统的性能。我们的解决方案对于黑匣子攻击和基于成熟的生成对抗网络（GAN）的图像恢复具有鲁棒性。我们分析了之前工作中的潜在风险，其中提出的cos相似度计算可能会直接泄露存储在服务器端的受保护的预计算嵌入。我们提出了一种基于功能秘密共享（FSG）的人脸嵌入比较协议，不会出现任何中间结果泄露。此外，我们在实验中表明，与基于秘密共享（SS）的协议相比，所提出的协议更有效。



## **15. Certified Zeroth-order Black-Box Defense with Robust UNet Denoiser**

经过认证的零阶黑匣子防御，具有强大的UNet降噪器 cs.CV

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2304.06430v2) [paper-pdf](http://arxiv.org/pdf/2304.06430v2)

**Authors**: Astha Verma, A V Subramanyam, Siddhesh Bangar, Naman Lal, Rajiv Ratn Shah, Shin'ichi Satoh

**Abstract**: Certified defense methods against adversarial perturbations have been recently investigated in the black-box setting with a zeroth-order (ZO) perspective. However, these methods suffer from high model variance with low performance on high-dimensional datasets due to the ineffective design of the denoiser and are limited in their utilization of ZO techniques. To this end, we propose a certified ZO preprocessing technique for removing adversarial perturbations from the attacked image in the black-box setting using only model queries. We propose a robust UNet denoiser (RDUNet) that ensures the robustness of black-box models trained on high-dimensional datasets. We propose a novel black-box denoised smoothing (DS) defense mechanism, ZO-RUDS, by prepending our RDUNet to the black-box model, ensuring black-box defense. We further propose ZO-AE-RUDS in which RDUNet followed by autoencoder (AE) is prepended to the black-box model. We perform extensive experiments on four classification datasets, CIFAR-10, CIFAR-10, Tiny Imagenet, STL-10, and the MNIST dataset for image reconstruction tasks. Our proposed defense methods ZO-RUDS and ZO-AE-RUDS beat SOTA with a huge margin of $35\%$ and $9\%$, for low dimensional (CIFAR-10) and with a margin of $20.61\%$ and $23.51\%$ for high-dimensional (STL-10) datasets, respectively.

摘要: 针对对抗性扰动的认证防御方法最近在零阶(ZO)视角的黑盒环境中被研究。然而，由于去噪器的设计不合理，这些方法在高维数据集上存在模型方差大、性能低的问题，限制了ZO技术的应用。为此，我们提出了一种经过验证的ZO预处理技术，用于在仅使用模型查询的情况下从黑盒环境中去除攻击图像中的对抗性扰动。我们提出了一种稳健的UNET去噪器(RDUNet)，以确保在高维数据集上训练的黑盒模型的稳健性。提出了一种新的黑盒去噪平滑防御机制ZO-RUDS，将RDUNet加入到黑盒模型中，保证了黑盒防御。我们进一步提出了ZO-AE-RUDS，其中RDUNet后跟自动编码器(AE)优先于黑盒模型。我们在CIFAR-10、CIFAR-10、Tiny Imagenet、STL-10和MNIST四个分类数据集上进行了大量的实验，用于图像重建任务。我们提出的防御方法ZO-RUDS和ZO-AE-RUDS对于低维数据集(CIFAR-10)分别以35美元和9美元的巨大优势击败了SOTA，而对于高维数据集(STL-10)分别以20.61美元和23.51美元的优势击败了SOTA。



## **16. PAC-Bayesian Adversarially Robust Generalization Bounds for Graph Neural Network**

图神经网络的Pac-Bayesian对抗鲁棒广义界 stat.ML

38pages

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2402.04038v2) [paper-pdf](http://arxiv.org/pdf/2402.04038v2)

**Authors**: Tan Sun, Junhong Lin

**Abstract**: Graph neural networks (GNNs) have gained popularity for various graph-related tasks. However, similar to deep neural networks, GNNs are also vulnerable to adversarial attacks. Empirical studies have shown that adversarially robust generalization has a pivotal role in establishing effective defense algorithms against adversarial attacks. In this paper, we contribute by providing adversarially robust generalization bounds for two kinds of popular GNNs, graph convolutional network (GCN) and message passing graph neural network, using the PAC-Bayesian framework. Our result reveals that spectral norm of the diffusion matrix on the graph and spectral norm of the weights as well as the perturbation factor govern the robust generalization bounds of both models. Our bounds are nontrivial generalizations of the results developed in (Liao et al., 2020) from the standard setting to adversarial setting while avoiding exponential dependence of the maximum node degree. As corollaries, we derive better PAC-Bayesian robust generalization bounds for GCN in the standard setting, which improve the bounds in (Liao et al., 2020) by avoiding exponential dependence on the maximum node degree.

摘要: 图神经网络(GNN)在各种与图相关的任务中得到了广泛的应用。然而，与深度神经网络类似，GNN也容易受到敌意攻击。经验研究表明，对抗性健壮性泛化在建立有效的防御算法抵抗对抗性攻击方面起着关键作用。在本文中，我们利用PAC-贝叶斯框架，为两种流行的GNN，图卷积网络(GCN)和消息传递图神经网络(GCN)提供了相对健壮的泛化界。我们的结果表明，图上扩散矩阵的谱范数和权值的谱范数以及扰动因子控制着两个模型的鲁棒推广界。我们的界是(Liao等人，2020)中发展的结果从标准环境到对抗环境的非平凡推广，同时避免了最大节点度的指数依赖。作为推论，我们得到了GCN在标准设置下更好的PAC-贝叶斯鲁棒推广界，通过避免对最大节点度的指数依赖，改进了(Liao等人，2020)的界。



## **17. Defensive Reconfigurable Intelligent Surface (D-RIS) Based on Non-Reciprocal Channel Links**

基于非互惠通道链接的防御性可重新配置智能表面（D-RIS） eess.SP

14 pages, journal paper

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2407.04905v1) [paper-pdf](http://arxiv.org/pdf/2407.04905v1)

**Authors**: Kun Chen-Hu, Petar Popovski

**Abstract**: A reconfigurable intelligent surface (RIS) is commonly made of low-cost passive and reflective meta-materials with excellent beam steering capabilities. It is applied to enhance wireless communication systems as a customizable signal reflector. However, RIS can also be adversely employed to disrupt the existing communication systems by introducing new types of vulnerability to the physical layer. We consider the \emph{RIS-In-The-Middle (RITM) attack}, in which an adversary uses RIS to jeopardize the direct channel between two transceivers by providing an alternative one with higher signal quality. This adversary can eavesdrop on all exchanged data by the legitimate users, but also perform a false data injection to the receiver. This work devises anti-attack techniques based on a non-reciprocal channel produced by a defensive RIS (D-RIS). The proposed precoding and combining methods and the channel estimation procedure for a non-reciprocal link are effective against potential adversaries while keeping the existing advantages of the RIS. We analyse the robustness of the system against attacks in terms of achievable secrecy rate and probability of detecting fake data. We believe that this defensive role of RIS can be a basis for new protocols and algorithms in the area.

摘要: 可重构智能表面(RIS)通常由低成本的被动和反射超材料制成，具有良好的波控能力。它作为一种可定制的信号反射器应用于增强无线通信系统。然而，RIS也可以被用来通过向物理层引入新类型的漏洞来扰乱现有的通信系统。我们考虑了中间RIS(RIS-in-the-Medium，RITM)攻击，在该攻击中，敌手使用RIS来危害两个收发信机之间的直接信道，从而提供一个具有更高信号质量的替代收发信机。此敌手可以窃听合法用户交换的所有数据，但也可以向接收方执行虚假数据注入。该工作设计了基于防御RIS(D-RIS)产生的非互易信道的抗攻击技术。所提出的预编码和合并方法以及针对非互易链路的信道估计过程在保持RIS的现有优势的同时有效地对抗潜在的对手。我们从可达到的保密率和检测到虚假数据的概率两个方面分析了系统对攻击的稳健性。我们相信，RIS的这种防御角色可以作为该领域新协议和算法的基础。



## **18. Late Breaking Results: Fortifying Neural Networks: Safeguarding Against Adversarial Attacks with Stochastic Computing**

最新突破性成果：强化神经网络：利用随机计算防范对抗攻击 cs.CR

3 pages, 1 figure, 2 tables

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04861v1) [paper-pdf](http://arxiv.org/pdf/2407.04861v1)

**Authors**: Faeze S. Banitaba, Sercan Aygun, M. Hassan Najafi

**Abstract**: In neural network (NN) security, safeguarding model integrity and resilience against adversarial attacks has become paramount. This study investigates the application of stochastic computing (SC) as a novel mechanism to fortify NN models. The primary objective is to assess the efficacy of SC to mitigate the deleterious impact of attacks on NN results. Through a series of rigorous experiments and evaluations, we explore the resilience of NNs employing SC when subjected to adversarial attacks. Our findings reveal that SC introduces a robust layer of defense, significantly reducing the susceptibility of networks to attack-induced alterations in their outcomes. This research contributes novel insights into the development of more secure and reliable NN systems, essential for applications in sensitive domains where data integrity is of utmost concern.

摘要: 在神经网络（NN）安全中，保护模型完整性和抵御对抗攻击的弹性已变得至关重要。本研究探讨了随机计算（SC）作为强化神经网络模型的新型机制的应用。主要目标是评估SC减轻攻击对NN结果的有害影响的有效性。通过一系列严格的实验和评估，我们探索了使用SC的NN在遭受对抗性攻击时的弹性。我们的研究结果表明，SC引入了强大的防御层，显着降低了网络对攻击诱导的结果改变的敏感性。这项研究为开发更安全、更可靠的神经网络系统提供了新的见解，这对于数据完整性最为关注的敏感领域的应用至关重要。



## **19. Where have you been? A Study of Privacy Risk for Point-of-Interest Recommendation**

你去哪儿了？兴趣点推荐的隐私风险研究 cs.LG

18 pages

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2310.18606v2) [paper-pdf](http://arxiv.org/pdf/2310.18606v2)

**Authors**: Kunlin Cai, Jinghuai Zhang, Zhiqing Hong, Will Shand, Guang Wang, Desheng Zhang, Jianfeng Chi, Yuan Tian

**Abstract**: As location-based services (LBS) have grown in popularity, more human mobility data has been collected. The collected data can be used to build machine learning (ML) models for LBS to enhance their performance and improve overall experience for users. However, the convenience comes with the risk of privacy leakage since this type of data might contain sensitive information related to user identities, such as home/work locations. Prior work focuses on protecting mobility data privacy during transmission or prior to release, lacking the privacy risk evaluation of mobility data-based ML models. To better understand and quantify the privacy leakage in mobility data-based ML models, we design a privacy attack suite containing data extraction and membership inference attacks tailored for point-of-interest (POI) recommendation models, one of the most widely used mobility data-based ML models. These attacks in our attack suite assume different adversary knowledge and aim to extract different types of sensitive information from mobility data, providing a holistic privacy risk assessment for POI recommendation models. Our experimental evaluation using two real-world mobility datasets demonstrates that current POI recommendation models are vulnerable to our attacks. We also present unique findings to understand what types of mobility data are more susceptible to privacy attacks. Finally, we evaluate defenses against these attacks and highlight future directions and challenges. Our attack suite is released at https://github.com/KunlinChoi/POIPrivacy.

摘要: 随着基于位置的服务(LBS)越来越受欢迎，人们收集了更多的人类移动性数据。收集到的数据可用于为LBS构建机器学习(ML)模型，以增强其性能并改善用户的整体体验。然而，随之而来的是隐私泄露的风险，因为这种类型的数据可能包含与用户身份相关的敏感信息，如家庭/工作地点。以往的工作主要集中在移动数据传输过程中或发布前的隐私保护上，缺乏对基于移动数据的ML模型的隐私风险评估。为了更好地理解和量化基于移动数据的ML模型中的隐私泄漏，我们设计了一个隐私攻击套件，其中包含针对兴趣点(POI)推荐模型的数据提取和成员关系推理攻击，该模型是应用最广泛的基于移动数据的ML模型之一。我们攻击套件中的这些攻击假设了不同的对手知识，旨在从移动数据中提取不同类型的敏感信息，为POI推荐模型提供全面的隐私风险评估。我们使用两个真实的移动数据集进行的实验评估表明，当前的POI推荐模型容易受到我们的攻击。我们还提出了独特的发现，以了解哪些类型的移动数据更容易受到隐私攻击。最后，我们评估了针对这些攻击的防御措施，并强调了未来的方向和挑战。我们的攻击套件在https://github.com/KunlinChoi/POIPrivacy.发布



## **20. CosPGD: an efficient white-box adversarial attack for pixel-wise prediction tasks**

CosPVD：针对像素预测任务的高效白盒对抗攻击 cs.CV

Accepted at 41st International Conference on Machine Learning (ICML),  2024

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2302.02213v3) [paper-pdf](http://arxiv.org/pdf/2302.02213v3)

**Authors**: Shashank Agnihotri, Steffen Jung, Margret Keuper

**Abstract**: While neural networks allow highly accurate predictions in many tasks, their lack of robustness towards even slight input perturbations often hampers their deployment. Adversarial attacks such as the seminal projected gradient descent (PGD) offer an effective means to evaluate a model's robustness and dedicated solutions have been proposed for attacks on semantic segmentation or optical flow estimation. While they attempt to increase the attack's efficiency, a further objective is to balance its effect, so that it acts on the entire image domain instead of isolated point-wise predictions. This often comes at the cost of optimization stability and thus efficiency. Here, we propose CosPGD, an attack that encourages more balanced errors over the entire image domain while increasing the attack's overall efficiency. To this end, CosPGD leverages a simple alignment score computed from any pixel-wise prediction and its target to scale the loss in a smooth and fully differentiable way. It leads to efficient evaluations of a model's robustness for semantic segmentation as well as regression models (such as optical flow, disparity estimation, or image restoration), and it allows it to outperform the previous SotA attack on semantic segmentation. We provide code for the CosPGD algorithm and example usage at https://github.com/shashankskagnihotri/cospgd.

摘要: 虽然神经网络允许在许多任务中进行高度准确的预测，但它们对即使是轻微的输入扰动缺乏稳健性，往往会阻碍它们的部署。诸如种子投影梯度下降(PGD)这样的对抗性攻击为评估模型的稳健性提供了一种有效的手段，针对语义分割或光流估计的攻击已经提出了专门的解决方案。虽然他们试图提高攻击的效率，但另一个目标是平衡其影响，使其作用于整个图像域，而不是孤立的逐点预测。这通常是以优化、稳定性和效率为代价的。在这里，我们提出了CosPGD，这是一种在提高攻击整体效率的同时，鼓励在整个图像域上更平衡错误的攻击。为此，CosPGD利用根据任何像素预测及其目标计算的简单对齐分数，以平滑和完全可区分的方式衡量损失。它可以有效地评估模型对语义分割和回归模型(如光流、视差估计或图像恢复)的稳健性，并使其性能优于以前的SOTA语义分割攻击。我们提供了CosPGD算法的代码和https://github.com/shashankskagnihotri/cospgd.上的示例使用



## **21. On Evaluating The Performance of Watermarked Machine-Generated Texts Under Adversarial Attacks**

关于评估带有水印的机器生成文本在对抗性攻击下的性能 cs.CR

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04794v1) [paper-pdf](http://arxiv.org/pdf/2407.04794v1)

**Authors**: Zesen Liu, Tianshuo Cong, Xinlei He, Qi Li

**Abstract**: Large Language Models (LLMs) excel in various applications, including text generation and complex tasks. However, the misuse of LLMs raises concerns about the authenticity and ethical implications of the content they produce, such as deepfake news, academic fraud, and copyright infringement. Watermarking techniques, which embed identifiable markers in machine-generated text, offer a promising solution to these issues by allowing for content verification and origin tracing. Unfortunately, the robustness of current LLM watermarking schemes under potential watermark removal attacks has not been comprehensively explored.   In this paper, to fill this gap, we first systematically comb the mainstream watermarking schemes and removal attacks on machine-generated texts, and then we categorize them into pre-text (before text generation) and post-text (after text generation) classes so that we can conduct diversified analyses. In our experiments, we evaluate eight watermarks (five pre-text, three post-text) and twelve attacks (two pre-text, ten post-text) across 87 scenarios. Evaluation results indicate that (1) KGW and Exponential watermarks offer high text quality and watermark retention but remain vulnerable to most attacks; (2) Post-text attacks are found to be more efficient and practical than pre-text attacks; (3) Pre-text watermarks are generally more imperceptible, as they do not alter text fluency, unlike post-text watermarks; (4) Additionally, combined attack methods can significantly increase effectiveness, highlighting the need for more robust watermarking solutions. Our study underscores the vulnerabilities of current techniques and the necessity for developing more resilient schemes.

摘要: 大型语言模型(LLM)在各种应用中表现出色，包括文本生成和复杂任务。然而，LLMS的滥用引发了人们对它们产生的内容的真实性和伦理影响的担忧，例如深度假新闻、学术欺诈和侵犯版权。在机器生成的文本中嵌入可识别标记的水印技术，通过允许内容验证和来源追踪，为这些问题提供了一种有前途的解决方案。遗憾的是，目前的LLM水印方案在潜在的水印去除攻击下的稳健性还没有得到全面的研究。为了填补这一空白，本文首先对主流的机器生成文本水印算法和去除攻击进行了系统的梳理，然后将其分为前文本类(文本生成前)和后文本类(文本生成后)，以便进行多样化的分析。在我们的实验中，我们评估了87个场景中的8个水印(5个前置文本，3个后置文本)和12个攻击(2个前置文本，10个后置文本)。评估结果表明：(1)KGW和指数水印具有高的文本质量和水印保留率，但仍然容易受到大多数攻击；(2)后文本攻击被发现比前文本攻击更有效和实用；(3)前文本水印通常更不可察觉，因为它们不像后文本水印那样改变文本的流畅性；(4)此外，组合攻击方法可以显著提高攻击效果，突出了对更健壮的水印解决方案的需求。我们的研究强调了当前技术的脆弱性，以及开发更具弹性的方案的必要性。



## **22. Remembering Everything Makes You Vulnerable: A Limelight on Machine Unlearning for Personalized Healthcare Sector**

记住一切让你变得脆弱：个性化医疗保健领域机器学习的聚光灯 cs.LG

15 Pages, Exploring unlearning techniques on ECG Classifier

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04589v1) [paper-pdf](http://arxiv.org/pdf/2407.04589v1)

**Authors**: Ahan Chatterjee, Sai Anirudh Aryasomayajula, Rajat Chaudhari, Subhajit Paul, Vishwa Mohan Singh

**Abstract**: As the prevalence of data-driven technologies in healthcare continues to rise, concerns regarding data privacy and security become increasingly paramount. This thesis aims to address the vulnerability of personalized healthcare models, particularly in the context of ECG monitoring, to adversarial attacks that compromise patient privacy. We propose an approach termed "Machine Unlearning" to mitigate the impact of exposed data points on machine learning models, thereby enhancing model robustness against adversarial attacks while preserving individual privacy. Specifically, we investigate the efficacy of Machine Unlearning in the context of personalized ECG monitoring, utilizing a dataset of clinical ECG recordings. Our methodology involves training a deep neural classifier on ECG data and fine-tuning the model for individual patients. We demonstrate the susceptibility of fine-tuned models to adversarial attacks, such as the Fast Gradient Sign Method (FGSM), which can exploit additional data points in personalized models. To address this vulnerability, we propose a Machine Unlearning algorithm that selectively removes sensitive data points from fine-tuned models, effectively enhancing model resilience against adversarial manipulation. Experimental results demonstrate the effectiveness of our approach in mitigating the impact of adversarial attacks while maintaining the pre-trained model accuracy.

摘要: 随着数据驱动技术在医疗保健中的普及程度持续上升，对数据隐私和安全的担忧变得越来越重要。这篇论文旨在解决个性化医疗模型的脆弱性，特别是在心电监测的背景下，面对危及患者隐私的对抗性攻击。我们提出了一种称为“机器遗忘”的方法来缓解暴露数据点对机器学习模型的影响，从而在保护个人隐私的同时增强了模型对对手攻击的稳健性。具体地说，我们利用临床心电记录的数据集，在个性化心电监测的背景下，研究了机器遗忘的有效性。我们的方法包括对心电数据训练深度神经分类器，并针对个别患者微调模型。我们证明了微调模型对敌意攻击的敏感性，例如快速梯度符号方法(FGSM)，它可以利用个性化模型中的额外数据点。为了解决这一漏洞，我们提出了一种机器遗忘算法，该算法选择性地从微调的模型中移除敏感数据点，有效地增强了模型对对手操纵的弹性。实验结果表明，该方法在保持预先训练好的模型精度的同时，有效地抑制了对抗性攻击的影响。



## **23. Certifiable Black-Box Attacks with Randomized Adversarial Examples: Breaking Defenses with Provable Confidence**

具有随机对抗示例的可认证黑匣子攻击：具有可证明的信心突破防御 cs.LG

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2304.04343v2) [paper-pdf](http://arxiv.org/pdf/2304.04343v2)

**Authors**: Hanbin Hong, Xinyu Zhang, Binghui Wang, Zhongjie Ba, Yuan Hong

**Abstract**: Black-box adversarial attacks have shown strong potential to subvert machine learning models. Existing black-box attacks craft adversarial examples by iteratively querying the target model and/or leveraging the transferability of a local surrogate model. Recently, such attacks can be effectively mitigated by state-of-the-art (SOTA) defenses, e.g., detection via the pattern of sequential queries, or injecting noise into the model. To our best knowledge, we take the first step to study a new paradigm of black-box attacks with provable guarantees -- certifiable black-box attacks that can guarantee the attack success probability (ASP) of adversarial examples before querying over the target model. This new black-box attack unveils significant vulnerabilities of machine learning models, compared to traditional empirical black-box attacks, e.g., breaking strong SOTA defenses with provable confidence, constructing a space of (infinite) adversarial examples with high ASP, and the ASP of the generated adversarial examples is theoretically guaranteed without verification/queries over the target model. Specifically, we establish a novel theoretical foundation for ensuring the ASP of the black-box attack with randomized adversarial examples (AEs). Then, we propose several novel techniques to craft the randomized AEs while reducing the perturbation size for better imperceptibility. Finally, we have comprehensively evaluated the certifiable black-box attacks on the CIFAR10/100, ImageNet, and LibriSpeech datasets, while benchmarking with 16 SOTA empirical black-box attacks, against various SOTA defenses in the domains of computer vision and speech recognition. Both theoretical and experimental results have validated the significance of the proposed attack.

摘要: 黑盒对抗性攻击已经显示出极大的潜力来颠覆机器学习模型。现有的黑盒攻击通过迭代查询目标模型和/或利用本地代理模型的可转移性来手工创建对抗性示例。最近，这种攻击可以通过最先进的(SOTA)防御措施有效地缓解，例如，通过顺序查询的模式进行检测，或者向模型中注入噪声。据我们所知，我们首先研究了一种新的具有可证明保证的黑盒攻击范型--可证明黑盒攻击，它可以在查询目标模型之前保证对手实例的攻击成功概率(ASP)。这种新的黑盒攻击揭示了机器学习模型的显著弱点，与传统的经验黑盒攻击相比，例如以可证明的置信度打破强大的Sota防御，构建具有高ASP的(无限)对抗性实例空间，并且在不对目标模型进行验证/查询的情况下，生成的对抗性实例的ASP在理论上得到了保证。具体地说，我们为用随机对抗性例子(AES)确保黑盒攻击的ASP奠定了新的理论基础。然后，我们提出了几种新的技术来制作随机化的动态平衡，同时减小扰动的大小以获得更好的不可见性。最后，我们综合评估了针对CIFAR10/100、ImageNet和LibriSpeech数据集的可认证黑盒攻击，同时以16个SOTA经验黑盒攻击为基准，针对计算机视觉和语音识别领域的各种SOTA防御进行了比较。理论和实验结果都验证了该攻击的重要性。



## **24. Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack**

使用局部自适应对抗色彩攻击保护艺术品免受神经风格转移 cs.CV

9 pages, 5 figures, 4 tables

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2401.09673v3) [paper-pdf](http://arxiv.org/pdf/2401.09673v3)

**Authors**: Zhongliang Guo, Junhao Dong, Yifei Qian, Kaixuan Wang, Weiye Li, Ziheng Guo, Yuheng Wang, Yanli Li, Ognjen Arandjelović, Lei Fang

**Abstract**: Neural style transfer (NST) generates new images by combining the style of one image with the content of another. However, unauthorized NST can exploit artwork, raising concerns about artists' rights and motivating the development of proactive protection methods. We propose Locally Adaptive Adversarial Color Attack (LAACA), empowering artists to protect their artwork from unauthorized style transfer by processing before public release. By delving into the intricacies of human visual perception and the role of different frequency components, our method strategically introduces frequency-adaptive perturbations in the image. These perturbations significantly degrade the generation quality of NST while maintaining an acceptable level of visual change in the original image, ensuring that potential infringers are discouraged from using the protected artworks, because of its bad NST generation quality. Additionally, existing metrics often overlook the importance of color fidelity in evaluating color-mattered tasks, such as the quality of NST-generated images, which is crucial in the context of artistic works. To comprehensively assess the color-mattered tasks, we propose the Adversarial Color Distance Metric (ACDM), designed to quantify the color difference of images pre- and post-manipulations. Experimental results confirm that attacking NST using LAACA results in visually inferior style transfer, and the ACDM can efficiently measure color-mattered tasks. By providing artists with a tool to safeguard their intellectual property, our work relieves the socio-technical challenges posed by the misuse of NST in the art community.

摘要: 神经样式转换(NST)通过将一幅图像的样式与另一幅图像的内容相结合来生成新图像。然而，未经授权的NST可以利用艺术品，这引发了人们对艺术家权利的担忧，并推动了主动保护方法的发展。我们提出了局部自适应对抗性色彩攻击(LAACA)，授权艺术家通过在公开发布之前进行处理来保护他们的作品免受未经授权的样式转移。通过深入研究人类视觉感知的复杂性和不同频率分量的作用，我们的方法战略性地在图像中引入了频率自适应扰动。这些干扰显著降低了NST的生成质量，同时保持了原始图像中可接受的视觉变化水平，确保了潜在侵权者因其糟糕的NST生成质量而不愿使用受保护的艺术品。此外，现有的衡量标准往往忽略了颜色保真度在评估与颜色有关的任务时的重要性，例如NST生成的图像的质量，这在艺术作品的背景下是至关重要的。为了全面评估颜色相关任务，我们提出了对抗性颜色距离度量(ACDM)，旨在量化图像处理前后的颜色差异。实验结果证实，使用LAACA攻击NST会导致视觉劣势风格迁移，ACDM可以有效地测量颜色相关任务。通过为艺术家提供保护其知识产权的工具，我们的工作缓解了艺术界滥用NST带来的社会技术挑战。



## **25. Distributed Black-box Attack: Do Not Overestimate Black-box Attacks**

分布式黑匣子攻击：不要高估黑匣子攻击 cs.LG

8 pages, 10 figures

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2210.16371v4) [paper-pdf](http://arxiv.org/pdf/2210.16371v4)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: Black-box adversarial attacks can fool image classifiers into misclassifying images without requiring access to model structure and weights. Recent studies have reported attack success rates of over 95% with less than 1,000 queries. The question then arises of whether black-box attacks have become a real threat against IoT devices that rely on cloud APIs to achieve image classification. To shed some light on this, note that prior research has primarily focused on increasing the success rate and reducing the number of queries. However, another crucial factor for black-box attacks against cloud APIs is the time required to perform the attack. This paper applies black-box attacks directly to cloud APIs rather than to local models, thereby avoiding mistakes made in prior research that applied the perturbation before image encoding and pre-processing. Further, we exploit load balancing to enable distributed black-box attacks that can reduce the attack time by a factor of about five for both local search and gradient estimation methods.

摘要: 黑盒对抗性攻击可以欺骗图像分类器对图像进行错误分类，而不需要访问模型结构和权重。最近的研究报告说，在不到1,000个查询的情况下，攻击成功率超过95%。随之而来的问题是，黑盒攻击是否已经成为对依赖云API实现图像分类的物联网设备的真正威胁。为了阐明这一点，请注意，以前的研究主要集中在提高成功率和减少查询数量上。然而，针对云API的黑盒攻击的另一个关键因素是执行攻击所需的时间。本文将黑盒攻击直接应用于云API而不是本地模型，从而避免了以往研究中在图像编码和预处理之前应用扰动的错误。此外，我们利用负载平衡来实现分布式黑盒攻击，对于局部搜索和梯度估计方法，可以将攻击时间减少约五倍。



## **26. Controlling Whisper: Universal Acoustic Adversarial Attacks to Control Speech Foundation Models**

控制耳语：控制语音基础模型的通用声学对抗攻击 cs.SD

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04482v1) [paper-pdf](http://arxiv.org/pdf/2407.04482v1)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Speech enabled foundation models, either in the form of flexible speech recognition based systems or audio-prompted large language models (LLMs), are becoming increasingly popular. One of the interesting aspects of these models is their ability to perform tasks other than automatic speech recognition (ASR) using an appropriate prompt. For example, the OpenAI Whisper model can perform both speech transcription and speech translation. With the development of audio-prompted LLMs there is the potential for even greater control options. In this work we demonstrate that with this greater flexibility the systems can be susceptible to model-control adversarial attacks. Without any access to the model prompt it is possible to modify the behaviour of the system by appropriately changing the audio input. To illustrate this risk, we demonstrate that it is possible to prepend a short universal adversarial acoustic segment to any input speech signal to override the prompt setting of an ASR foundation model. Specifically, we successfully use a universal adversarial acoustic segment to control Whisper to always perform speech translation, despite being set to perform speech transcription. Overall, this work demonstrates a new form of adversarial attack on multi-tasking speech enabled foundation models that needs to be considered prior to the deployment of this form of model.

摘要: 以灵活的基于语音识别的系统或音频提示的大型语言模型(LLM)的形式启用语音的基础模型正变得越来越受欢迎。这些模型的一个有趣方面是，它们能够使用适当的提示执行自动语音识别(ASR)以外的任务。例如，OpenAI Whisper模型可以执行语音转录和语音翻译。随着音频提示LLMS的发展，有可能出现更大的控制选项。在这项工作中，我们证明了有了这种更大的灵活性，系统可以容易受到模型控制的对抗性攻击。在不访问模型提示的情况下，可以通过适当地改变音频输入来修改系统的行为。为了说明这一风险，我们证明了有可能在任何输入语音信号之前添加一个简短的通用对抗性声学片段，以覆盖ASR基础模型的提示设置。具体地说，我们成功地使用了一个通用的对抗性声学段来控制Whisper始终执行语音翻译，尽管被设置为执行语音转录。总体而言，这项工作展示了一种对多任务语音启用的基础模型的新形式的对抗性攻击，在部署这种形式的模型之前需要考虑这种形式。



## **27. Self-Supervised Representation Learning for Adversarial Attack Detection**

对抗性攻击检测的自我监督表示学习 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04382v1) [paper-pdf](http://arxiv.org/pdf/2407.04382v1)

**Authors**: Yi Li, Plamen Angelov, Neeraj Suri

**Abstract**: Supervised learning-based adversarial attack detection methods rely on a large number of labeled data and suffer significant performance degradation when applying the trained model to new domains. In this paper, we propose a self-supervised representation learning framework for the adversarial attack detection task to address this drawback. Firstly, we map the pixels of augmented input images into an embedding space. Then, we employ the prototype-wise contrastive estimation loss to cluster prototypes as latent variables. Additionally, drawing inspiration from the concept of memory banks, we introduce a discrimination bank to distinguish and learn representations for each individual instance that shares the same or a similar prototype, establishing a connection between instances and their associated prototypes. We propose a parallel axial-attention (PAA)-based encoder to facilitate the training process by parallel training over height- and width-axis of attention maps. Experimental results show that, compared to various benchmark self-supervised vision learning models and supervised adversarial attack detection methods, the proposed model achieves state-of-the-art performance on the adversarial attack detection task across a wide range of images.

摘要: 基于有监督学习的敌意攻击检测方法依赖于大量的标记数据，在将训练好的模型应用于新的领域时，性能会显著下降。针对这一缺陷，本文提出了一种用于敌方攻击检测任务的自监督表示学习框架。首先，将增强后的输入图像的像素映射到嵌入空间。然后，我们使用原型的对比估计损失来聚类原型作为潜在变量。此外，受内存库概念的启发，我们引入了一个鉴别库来区分和学习共享相同或相似原型的每个实例的表示，在实例及其相关原型之间建立了联系。我们提出了一种基于并行轴注意(PAA)的编码器，通过在高度和宽度注意轴地图上进行并行训练来简化训练过程。实验结果表明，与各种基准的自监督视觉学习模型和有监督的对抗性攻击检测方法相比，该模型在大范围图像的对抗性攻击检测任务上取得了最好的性能。



## **28. Jailbreak Attacks and Defenses Against Large Language Models: A Survey**

针对大型语言模型的越狱攻击和防御：调查 cs.CR

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04295v1) [paper-pdf](http://arxiv.org/pdf/2407.04295v1)

**Authors**: Sibo Yi, Yule Liu, Zhen Sun, Tianshuo Cong, Xinlei He, Jiaxing Song, Ke Xu, Qi Li

**Abstract**: Large Language Models (LLMs) have performed exceptionally in various text-generative tasks, including question answering, translation, code completion, etc. However, the over-assistance of LLMs has raised the challenge of "jailbreaking", which induces the model to generate malicious responses against the usage policy and society by designing adversarial prompts. With the emergence of jailbreak attack methods exploiting different vulnerabilities in LLMs, the corresponding safety alignment measures are also evolving. In this paper, we propose a comprehensive and detailed taxonomy of jailbreak attack and defense methods. For instance, the attack methods are divided into black-box and white-box attacks based on the transparency of the target model. Meanwhile, we classify defense methods into prompt-level and model-level defenses. Additionally, we further subdivide these attack and defense methods into distinct sub-classes and present a coherent diagram illustrating their relationships. We also conduct an investigation into the current evaluation methods and compare them from different perspectives. Our findings aim to inspire future research and practical implementations in safeguarding LLMs against adversarial attacks. Above all, although jailbreak remains a significant concern within the community, we believe that our work enhances the understanding of this domain and provides a foundation for developing more secure LLMs.

摘要: 大型语言模型(LLMS)在问答、翻译、代码补全等文本生成任务中表现出色。然而，LLMS的过度协助带来了越狱的挑战，这导致该模型通过设计敌意提示来生成针对使用策略和社会的恶意响应。随着利用LLMS中不同漏洞的越狱攻击方法的出现，相应的安全对齐措施也在不断发展。在本文中，我们提出了一个全面和详细的分类越狱攻防方法。例如，根据目标模型的透明性，将攻击方法分为黑盒攻击和白盒攻击。同时，我们将防御方法分为提示级防御和模型级防御。此外，我们还将这些攻击和防御方法进一步细分为不同的子类，并提供了一个连贯的图来说明它们之间的关系。我们还对现有的评估方法进行了调查，并从不同的角度对它们进行了比较。我们的发现旨在启发未来在保护LLM免受对手攻击方面的研究和实际实现。最重要的是，尽管越狱在社区中仍然是一个重要的问题，但我们相信我们的工作增进了对这个领域的了解，并为开发更安全的LLM提供了基础。



## **29. FlowMur: A Stealthy and Practical Audio Backdoor Attack with Limited Knowledge**

FlowMur：知识有限的隐秘而实用的音频后门攻击 cs.CR

To appear at lEEE Symposium on Security & Privacy (Oakland) 2024

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2312.09665v2) [paper-pdf](http://arxiv.org/pdf/2312.09665v2)

**Authors**: Jiahe Lan, Jie Wang, Baochen Yan, Zheng Yan, Elisa Bertino

**Abstract**: Speech recognition systems driven by DNNs have revolutionized human-computer interaction through voice interfaces, which significantly facilitate our daily lives. However, the growing popularity of these systems also raises special concerns on their security, particularly regarding backdoor attacks. A backdoor attack inserts one or more hidden backdoors into a DNN model during its training process, such that it does not affect the model's performance on benign inputs, but forces the model to produce an adversary-desired output if a specific trigger is present in the model input. Despite the initial success of current audio backdoor attacks, they suffer from the following limitations: (i) Most of them require sufficient knowledge, which limits their widespread adoption. (ii) They are not stealthy enough, thus easy to be detected by humans. (iii) Most of them cannot attack live speech, reducing their practicality. To address these problems, in this paper, we propose FlowMur, a stealthy and practical audio backdoor attack that can be launched with limited knowledge. FlowMur constructs an auxiliary dataset and a surrogate model to augment adversary knowledge. To achieve dynamicity, it formulates trigger generation as an optimization problem and optimizes the trigger over different attachment positions. To enhance stealthiness, we propose an adaptive data poisoning method according to Signal-to-Noise Ratio (SNR). Furthermore, ambient noise is incorporated into the process of trigger generation and data poisoning to make FlowMur robust to ambient noise and improve its practicality. Extensive experiments conducted on two datasets demonstrate that FlowMur achieves high attack performance in both digital and physical settings while remaining resilient to state-of-the-art defenses. In particular, a human study confirms that triggers generated by FlowMur are not easily detected by participants.

摘要: 由DNN驱动的语音识别系统通过语音接口使人机交互发生了革命性的变化，极大地方便了我们的日常生活。然而，这些系统越来越受欢迎，也引发了对其安全性的特别关注，特别是关于后门攻击的问题。后门攻击在其训练过程中将一个或多个隐藏的后门插入到DNN模型中，使得它不会影响模型在良性输入上的性能，但如果模型输入中存在特定触发器，则迫使模型产生对手期望的输出。尽管目前的音频后门攻击取得了初步的成功，但它们受到以下限制：(I)大多数音频后门攻击需要足够的知识，这限制了它们的广泛采用。(Ii)它们的隐蔽性不够，因此很容易被人发现。(3)他们中的大多数不能攻击现场演讲，降低了他们的实用性。为了解决这些问题，在本文中，我们提出了FlowMur，一种隐蔽而实用的音频后门攻击，可以在有限的知识下发起。FlowMur构建了一个辅助数据集和一个代理模型来增强对手的知识。为了实现动态化，它将触发器的生成描述为一个优化问题，并在不同的附着位置上对触发器进行优化。为了提高隐蔽性，提出了一种基于信噪比的自适应数据毒化方法。此外，在触发产生和数据毒化过程中引入了环境噪声，使FlowMur对环境噪声具有较强的鲁棒性，提高了其实用性。在两个数据集上进行的广泛实验表明，FlowMur在数字和物理环境中都实现了高攻击性能，同时对最先进的防御保持了弹性。特别是，一项人类研究证实，FlowMur产生的触发因素不容易被参与者检测到。



## **30. Defending Jailbreak Prompts via In-Context Adversarial Game**

通过上下文对抗游戏为越狱辩护 cs.LG

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2402.13148v2) [paper-pdf](http://arxiv.org/pdf/2402.13148v2)

**Authors**: Yujun Zhou, Yufei Han, Haomin Zhuang, Kehan Guo, Zhenwen Liang, Hongyan Bao, Xiangliang Zhang

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities across diverse applications. However, concerns regarding their security, particularly the vulnerability to jailbreak attacks, persist. Drawing inspiration from adversarial training in deep learning and LLM agent learning processes, we introduce the In-Context Adversarial Game (ICAG) for defending against jailbreaks without the need for fine-tuning. ICAG leverages agent learning to conduct an adversarial game, aiming to dynamically extend knowledge to defend against jailbreaks. Unlike traditional methods that rely on static datasets, ICAG employs an iterative process to enhance both the defense and attack agents. This continuous improvement process strengthens defenses against newly generated jailbreak prompts. Our empirical studies affirm ICAG's efficacy, where LLMs safeguarded by ICAG exhibit significantly reduced jailbreak success rates across various attack scenarios. Moreover, ICAG demonstrates remarkable transferability to other LLMs, indicating its potential as a versatile defense mechanism.

摘要: 大型语言模型(LLM)在不同的应用程序中展示了卓越的功能。然而，对他们的安全，特别是对越狱攻击的脆弱性的担忧依然存在。从深度学习和LLM代理学习过程中的对抗性训练中获得灵感，我们引入了无需微调的上下文对抗性游戏(ICAG)来防御越狱。ICAG利用代理学习进行对抗性游戏，旨在动态扩展知识来防御越狱。与依赖静态数据集的传统方法不同，ICAG采用迭代过程来增强防御和攻击代理。这一不断改进的过程加强了对新生成的越狱提示的防御。我们的经验研究肯定了ICAG的有效性，在不同的攻击场景中，由ICAG保护的LLM显示出显著降低的越狱成功率。此外，ICAG表现出显著的可转移性，表明其作为一种多功能防御机制的潜力。



## **31. Securing Multi-turn Conversational Language Models Against Distributed Backdoor Triggers**

保护多轮对话语言模型免受分布式后门触发器的影响 cs.CL

Submitted to EMNLP 2024

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.04151v1) [paper-pdf](http://arxiv.org/pdf/2407.04151v1)

**Authors**: Terry Tong, Jiashu Xu, Qin Liu, Muhao Chen

**Abstract**: The security of multi-turn conversational large language models (LLMs) is understudied despite it being one of the most popular LLM utilization. Specifically, LLMs are vulnerable to data poisoning backdoor attacks, where an adversary manipulates the training data to cause the model to output malicious responses to predefined triggers. Specific to the multi-turn dialogue setting, LLMs are at the risk of even more harmful and stealthy backdoor attacks where the backdoor triggers may span across multiple utterances, giving lee-way to context-driven attacks. In this paper, we explore a novel distributed backdoor trigger attack that serves to be an extra tool in an adversary's toolbox that can interface with other single-turn attack strategies in a plug and play manner. Results on two representative defense mechanisms indicate that distributed backdoor triggers are robust against existing defense strategies which are designed for single-turn user-model interactions, motivating us to propose a new defense strategy for the multi-turn dialogue setting that is more challenging. To this end, we also explore a novel contrastive decoding based defense that is able to mitigate the backdoor with a low computational tradeoff.

摘要: 多话轮会话大语言模型(LLMS)是目前应用最广泛的大语言模型之一，但其安全性仍未得到充分的研究。具体地说，LLM容易受到数据中毒后门攻击，在这种攻击中，敌手操纵训练数据，使模型输出对预定义触发器的恶意响应。具体到多轮对话设置，LLM面临着更具危害性和更隐蔽的后门攻击的风险，其中后门触发可能跨越多个话语，使Lee-way成为上下文驱动的攻击。在本文中，我们探索了一种新型的分布式后门触发攻击，它是对手工具箱中的一个额外工具，可以以即插即用的方式与其他单轮攻击策略对接。在两种典型防御机制上的结果表明，分布式后门触发器对现有的针对单回合用户-模型交互的防御策略具有较强的健壮性，这促使我们提出了一种新的针对更具挑战性的多回合对话环境的防御策略。为此，我们还探索了一种新的基于对比解码的防御方案，该方案能够以较低的计算代价缓解后门问题。



## **32. A Survey on Adversarial Contention Resolution**

对抗性竞争解决方法研究 cs.DC

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2403.03876v2) [paper-pdf](http://arxiv.org/pdf/2403.03876v2)

**Authors**: Ioana Banicescu, Trisha Chakraborty, Seth Gilbert, Maxwell Young

**Abstract**: Contention resolution addresses the challenge of coordinating access by multiple processes to a shared resource such as memory, disk storage, or a communication channel. Originally spurred by challenges in database systems and bus networks, contention resolution has endured as an important abstraction for resource sharing, despite decades of technological change. Here, we survey the literature on resolving worst-case contention, where the number of processes and the time at which each process may start seeking access to the resource is dictated by an adversary. We highlight the evolution of contention resolution, where new concerns -- such as security, quality of service, and energy efficiency -- are motivated by modern systems. These efforts have yielded insights into the limits of randomized and deterministic approaches, as well as the impact of different model assumptions such as global clock synchronization, knowledge of the number of processors, feedback from access attempts, and attacks on the availability of the shared resource.

摘要: 争用解决方案解决了协调多个进程对共享资源(如内存、磁盘存储或通信通道)的访问的挑战。争用解决最初是由数据库系统和总线网络中的挑战推动的，尽管经历了几十年的技术变革，但它作为资源共享的一个重要抽象概念一直存在。在这里，我们回顾了关于解决最坏情况争用的文献，在这种情况下，进程的数量和每个进程可能开始寻求访问资源的时间由对手决定。我们重点介绍争用解决方案的演变，其中新的关注点--如安全性、服务质量和能源效率--是由现代系统驱动的。这些努力使人们深入了解了随机化和确定性方法的局限性，以及不同模型假设的影响，如全球时钟同步、处理器数量的知识、访问尝试的反馈以及对共享资源可用性的攻击。



## **33. Optimizing a-DCF for Spoofing-Robust Speaker Verification**

优化a-DCF以实现欺骗稳健的说话人验证 eess.AS

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.04034v1) [paper-pdf](http://arxiv.org/pdf/2407.04034v1)

**Authors**: Oğuzhan Kurnaz, Jagabandhu Mishra, Tomi H. Kinnunen, Cemal Hanilçi

**Abstract**: Automatic speaker verification (ASV) systems are vulnerable to spoofing attacks such as text-to-speech. In this study, we propose a novel spoofing-robust ASV back-end classifier, optimized directly for the recently introduced, architecture-agnostic detection cost function (a-DCF). We combine a-DCF and binary cross-entropy (BCE) losses to optimize the network weights, combined by a novel, straightforward detection threshold optimization technique. Experiments on the ASVspoof2019 database demonstrate considerable improvement over the baseline optimized using BCE only (from minimum a-DCF of 0.1445 to 0.1254), representing 13% relative improvement. These initial promising results demonstrate that it is possible to adjust an ASV system to find appropriate balance across the contradicting aims of user convenience and security against adversaries.

摘要: 自动说话人验证（ASV）系统容易受到文本转语音等欺骗攻击。在这项研究中，我们提出了一种新型的欺骗鲁棒ASV后台分类器，该分类器直接针对最近引入的、与架构无关的检测成本函数（a-DCF）进行了优化。我们结合a-DCF和二进制交叉熵（BCE）损失来优化网络权重，并结合一种新颖、简单的检测阈值优化技术。ASVspoof 2019数据库的实验表明，与仅使用BCE优化的基线相比，有了相当大的改进（从最小a-DCF 0.1445到0.1254），相对改进了13%。这些初步有希望的结果表明，调整ASV系统是可能的，以在用户便利性和针对对手的安全性这两个相互矛盾的目标之间找到适当的平衡。



## **34. Mitigating Low-Frequency Bias: Feature Recalibration and Frequency Attention Regularization for Adversarial Robustness**

缓解低频偏见：对抗稳健性的特征重新校准和频率注意力规则化 cs.CV

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.04016v1) [paper-pdf](http://arxiv.org/pdf/2407.04016v1)

**Authors**: Kejia Zhang, Juanjuan Weng, Yuanzheng Cai, Zhiming Luo, Shaozi Li

**Abstract**: Ensuring the robustness of computer vision models against adversarial attacks is a significant and long-lasting objective. Motivated by adversarial attacks, researchers have devoted considerable efforts to enhancing model robustness by adversarial training (AT). However, we observe that while AT improves the models' robustness against adversarial perturbations, it fails to improve their ability to effectively extract features across all frequency components. Each frequency component contains distinct types of crucial information: low-frequency features provide fundamental structural insights, while high-frequency features capture intricate details and textures. In particular, AT tends to neglect the reliance on susceptible high-frequency features. This low-frequency bias impedes the model's ability to effectively leverage the potentially meaningful semantic information present in high-frequency features. This paper proposes a novel module called High-Frequency Feature Disentanglement and Recalibration (HFDR), which separates features into high-frequency and low-frequency components and recalibrates the high-frequency feature to capture latent useful semantics. Additionally, we introduce frequency attention regularization to magnitude the model's extraction of different frequency features and mitigate low-frequency bias during AT. Extensive experiments showcase the immense potential and superiority of our approach in resisting various white-box attacks, transfer attacks, and showcasing strong generalization capabilities.

摘要: 确保计算机视觉模型对敌方攻击的健壮性是一个重要而持久的目标。在对抗性攻击的激励下，研究人员通过对抗性训练(AT)来增强模型的稳健性。然而，我们观察到，虽然AT提高了模型对对抗性扰动的稳健性，但它并没有提高它们有效地提取所有频率分量的特征的能力。每个频率分量包含不同类型的关键信息：低频特征提供基本的结构洞察，而高频特征捕获复杂的细节和纹理。特别是，AT倾向于忽略对易感高频特征的依赖。这种低频偏差阻碍了模型有效地利用高频特征中存在的潜在有意义的语义信息的能力。提出了一种新的高频特征解缠与重校准模块(HFDR)，该模块将特征分解为高频成分和低频成分，并对高频成分进行重校准以获取潜在的有用语义。此外，我们还引入了频率注意正则化来放大模型对不同频率特征的提取，并减少了AT过程中的低频偏差。广泛的实验表明，该方法在抵抗各种白盒攻击、传输攻击和表现出强大的泛化能力方面具有巨大的潜力和优势。



## **35. TrackPGD: A White-box Attack using Binary Masks against Robust Transformer Trackers**

TrackPVD：使用二进制屏蔽针对稳健的Transformer Trackers的白盒攻击 cs.CV

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.03946v1) [paper-pdf](http://arxiv.org/pdf/2407.03946v1)

**Authors**: Fatemeh Nourilenjan Nokabadi, Yann Batiste Pequignot, Jean-Francois Lalonde, Christian Gagné

**Abstract**: Object trackers with transformer backbones have achieved robust performance on visual object tracking datasets. However, the adversarial robustness of these trackers has not been well studied in the literature. Due to the backbone differences, the adversarial white-box attacks proposed for object tracking are not transferable to all types of trackers. For instance, transformer trackers such as MixFormerM still function well after black-box attacks, especially in predicting the object binary masks. We are proposing a novel white-box attack named TrackPGD, which relies on the predicted object binary mask to attack the robust transformer trackers. That new attack focuses on annotation masks by adapting the well-known SegPGD segmentation attack, allowing to successfully conduct the white-box attack on trackers relying on transformer backbones. The experimental results indicate that the TrackPGD is able to effectively attack transformer-based trackers such as MixFormerM, OSTrackSTS, and TransT-SEG on several tracking datasets.

摘要: 具有变压器主干的对象跟踪器在视觉对象跟踪数据集上取得了稳健的性能。然而，这些跟踪器的对抗健壮性在文献中还没有得到很好的研究。由于主干的差异，提出的用于目标跟踪的对抗性白盒攻击并不适用于所有类型的跟踪器。例如，在黑盒攻击后，诸如MixFormerM之类的转换器跟踪器仍然可以很好地工作，特别是在预测对象的二进制掩码方面。我们提出了一种新的白盒攻击方法TrackPGD，它依靠预测对象的二进制掩码来攻击健壮的变压器跟踪器。该新攻击通过采用著名的SegPGD分段攻击，将重点放在注释掩码上，从而能够成功地对依赖于变压器主干的跟踪器进行白盒攻击。实验结果表明，TrackPGD能够在多个跟踪数据集上有效地攻击MixFormerM、OSTrackSTS和TransT-SEG等基于变压器的跟踪器。



## **36. EvolBA: Evolutionary Boundary Attack under Hard-label Black Box condition**

EvolBA：硬标签黑匣子条件下的进化边界攻击 cs.CV

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.02248v2) [paper-pdf](http://arxiv.org/pdf/2407.02248v2)

**Authors**: Ayane Tajima, Satoshi Ono

**Abstract**: Research has shown that deep neural networks (DNNs) have vulnerabilities that can lead to the misrecognition of Adversarial Examples (AEs) with specifically designed perturbations. Various adversarial attack methods have been proposed to detect vulnerabilities under hard-label black box (HL-BB) conditions in the absence of loss gradients and confidence scores.However, these methods fall into local solutions because they search only local regions of the search space. Therefore, this study proposes an adversarial attack method named EvolBA to generate AEs using Covariance Matrix Adaptation Evolution Strategy (CMA-ES) under the HL-BB condition, where only a class label predicted by the target DNN model is available. Inspired by formula-driven supervised learning, the proposed method introduces domain-independent operators for the initialization process and a jump that enhances search exploration. Experimental results confirmed that the proposed method could determine AEs with smaller perturbations than previous methods in images where the previous methods have difficulty.

摘要: 研究表明，深度神经网络(DNN)存在漏洞，可能会导致对经过特殊设计的扰动的对抗性示例(AE)的错误识别。针对硬标签黑盒(HL-BB)环境下不存在损失梯度和置信度的漏洞检测问题，提出了多种对抗性攻击方法，但这些方法只搜索搜索空间的局部区域，容易陷入局部解.因此，本文提出了一种基于协方差矩阵自适应进化策略(CMA-ES)的对抗性攻击方法EvolBA，用于在目标DNN模型预测的类别标签不可用的HL-BB条件下生成AEs。受公式驱动的监督学习的启发，该方法在初始化过程中引入了领域无关的算子，并引入了一个跳跃来增强搜索探索。实验结果表明，该方法能够以较小的扰动确定图像中的声学效应，克服了以往方法的不足。



## **37. Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment**

法学硕士作为法官稳健吗？调查零射击LLM评估中的普遍对抗攻击 cs.CL

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2402.14016v2) [paper-pdf](http://arxiv.org/pdf/2402.14016v2)

**Authors**: Vyas Raina, Adian Liusie, Mark Gales

**Abstract**: Large Language Models (LLMs) are powerful zero-shot assessors used in real-world situations such as assessing written exams and benchmarking systems. Despite these critical applications, no existing work has analyzed the vulnerability of judge-LLMs to adversarial manipulation. This work presents the first study on the adversarial robustness of assessment LLMs, where we demonstrate that short universal adversarial phrases can be concatenated to deceive judge LLMs to predict inflated scores. Since adversaries may not know or have access to the judge-LLMs, we propose a simple surrogate attack where a surrogate model is first attacked, and the learned attack phrase then transferred to unknown judge-LLMs. We propose a practical algorithm to determine the short universal attack phrases and demonstrate that when transferred to unseen models, scores can be drastically inflated such that irrespective of the assessed text, maximum scores are predicted. It is found that judge-LLMs are significantly more susceptible to these adversarial attacks when used for absolute scoring, as opposed to comparative assessment. Our findings raise concerns on the reliability of LLM-as-a-judge methods, and emphasize the importance of addressing vulnerabilities in LLM assessment methods before deployment in high-stakes real-world scenarios.

摘要: 大型语言模型(LLM)是用于评估笔试和基准系统等真实世界情况的强大的零分评价器。尽管有这些关键的应用，但现有的工作还没有分析JUSTER-LLMS对对抗操纵的脆弱性。这项工作首次研究了评估LLMS的对抗稳健性，其中我们证明了短的通用对抗性短语可以被串联起来欺骗裁判LLMS来预测夸大的分数。由于敌手可能不知道或无法访问裁判LLMS，我们提出了一种简单的代理攻击，首先攻击代理模型，然后将学习到的攻击短语传输到未知的JUSTER-LLMS。我们提出了一个实用的算法来确定简短的通用攻击短语，并证明了当转移到看不见的模型时，分数可以被大幅夸大，以至于无论评估的文本是什么，都可以预测最高分数。研究发现，与比较评估相比，JUSICE-LLM在用于绝对评分时更容易受到这些对抗性攻击。我们的发现引起了人们对LLM作为判断方法的可靠性的担忧，并强调了在高风险的现实世界场景中部署之前解决LLM评估方法中的漏洞的重要性。



## **38. DART: Deep Adversarial Automated Red Teaming for LLM Safety**

DART：深度对抗自动化红色团队，确保LLM安全 cs.CR

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.03876v1) [paper-pdf](http://arxiv.org/pdf/2407.03876v1)

**Authors**: Bojian Jiang, Yi Jing, Tianhao Shen, Qing Yang, Deyi Xiong

**Abstract**: Manual Red teaming is a commonly-used method to identify vulnerabilities in large language models (LLMs), which, is costly and unscalable. In contrast, automated red teaming uses a Red LLM to automatically generate adversarial prompts to the Target LLM, offering a scalable way for safety vulnerability detection. However, the difficulty of building a powerful automated Red LLM lies in the fact that the safety vulnerabilities of the Target LLM are dynamically changing with the evolution of the Target LLM. To mitigate this issue, we propose a Deep Adversarial Automated Red Teaming (DART) framework in which the Red LLM and Target LLM are deeply and dynamically interacting with each other in an iterative manner. In each iteration, in order to generate successful attacks as many as possible, the Red LLM not only takes into account the responses from the Target LLM, but also adversarially adjust its attacking directions by monitoring the global diversity of generated attacks across multiple iterations. Simultaneously, to explore dynamically changing safety vulnerabilities of the Target LLM, we allow the Target LLM to enhance its safety via an active learning based data selection mechanism. Experimential results demonstrate that DART significantly reduces the safety risk of the target LLM. For human evaluation on Anthropic Harmless dataset, compared to the instruction-tuning target LLM, DART eliminates the violation risks by 53.4\%. We will release the datasets and codes of DART soon.

摘要: 手动Red Teaming是一种常用的识别大型语言模型(LLM)漏洞的方法，该方法代价高昂且不可扩展。相比之下，自动红色团队使用Red LLM自动生成针对Target LLM的敌意提示，为安全漏洞检测提供了一种可扩展的方法。然而，构建功能强大的自动Red LLM的难点在于，目标LLM的安全漏洞随着目标LLM的演化而动态变化。为了缓解这一问题，我们提出了一个深度对抗性自动红团队(DART)框架，在该框架中，Red LLM和Target LLM以迭代的方式进行深度和动态的交互。在每一次迭代中，为了生成尽可能多的成功攻击，Red LLM不仅考虑目标LLM的响应，而且通过监控生成的攻击在多个迭代中的全局多样性来相反地调整其攻击方向。同时，为了探索动态变化的安全漏洞，我们允许目标LLM通过一种基于主动学习的数据选择机制来增强其安全性。实验结果表明，DART显著降低了目标LLM的安全风险。对于人类无害数据集的人工评估，与指令调优目标LLM相比，DART消除了53.4%的违规风险。我们将很快公布DART的数据集和代码。



## **39. RobQuNNs: A Methodology for Robust Quanvolutional Neural Networks against Adversarial Attacks**

RobQuNNs：一种对抗攻击的鲁棒量子卷积神经网络方法 quant-ph

6 pages, 3 figures

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.03875v1) [paper-pdf](http://arxiv.org/pdf/2407.03875v1)

**Authors**: Walid El Maouaki, Alberto Marchisio, Taoufik Said, Muhammad Shafique, Mohamed Bennai

**Abstract**: Recent advancements in quantum computing have led to the emergence of hybrid quantum neural networks, such as Quanvolutional Neural Networks (QuNNs), which integrate quantum and classical layers. While the susceptibility of classical neural networks to adversarial attacks is well-documented, the impact on QuNNs remains less understood. This study introduces RobQuNN, a new methodology to enhance the robustness of QuNNs against adversarial attacks, utilizing quantum circuit expressibility and entanglement capability alongside different adversarial strategies. Additionally, the study investigates the transferability of adversarial examples between classical and quantum models using RobQuNN, enhancing our understanding of cross-model vulnerabilities and pointing to new directions in quantum cybersecurity. The findings reveal that QuNNs exhibit up to 60\% higher robustness compared to classical networks for the MNIST dataset, particularly at low levels of perturbation. This underscores the potential of quantum approaches in improving security defenses. In addition, RobQuNN revealed that QuNN does not exhibit enhanced resistance or susceptibility to cross-model adversarial examples regardless of the quantum circuit architecture.

摘要: 量子计算的最新进展导致了混合量子神经网络的出现，例如将量子层和经典层结合在一起的量子卷积神经网络(Quanvolutive Neurn Networks，QuNNS)。虽然经典神经网络对敌意攻击的敏感性已经得到了很好的证明，但对量子神经网络的影响仍然知之甚少。本文介绍了一种新的方法RobQuNN，该方法利用量子电路的表达能力和纠缠能力，结合不同的对抗策略，增强了量子神经网络对敌意攻击的健壮性。此外，该研究还利用RobQuNN研究了经典模型和量子模型之间敌意例子的可转移性，增强了我们对跨模型漏洞的理解，并为量子网络安全指明了新的方向。结果表明，对于MNIST数据集，量子神经网络表现出比经典网络更高的稳健性，特别是在低扰动水平下。这突显了量子方法在改善安全防御方面的潜力。此外，RobQuNN揭示，无论量子电路架构如何，QuNN都不会对跨模型对手示例表现出增强的抵抗力或敏感度。



## **40. Adversarial Robustness of VAEs across Intersectional Subgroups**

跨交叉亚组VAE的对抗稳健性 cs.LG

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.03864v1) [paper-pdf](http://arxiv.org/pdf/2407.03864v1)

**Authors**: Chethan Krishnamurthy Ramanaik, Arjun Roy, Eirini Ntoutsi

**Abstract**: Despite advancements in Autoencoders (AEs) for tasks like dimensionality reduction, representation learning and data generation, they remain vulnerable to adversarial attacks. Variational Autoencoders (VAEs), with their probabilistic approach to disentangling latent spaces, show stronger resistance to such perturbations compared to deterministic AEs; however, their resilience against adversarial inputs is still a concern. This study evaluates the robustness of VAEs against non-targeted adversarial attacks by optimizing minimal sample-specific perturbations to cause maximal damage across diverse demographic subgroups (combinations of age and gender). We investigate two questions: whether there are robustness disparities among subgroups, and what factors contribute to these disparities, such as data scarcity and representation entanglement. Our findings reveal that robustness disparities exist but are not always correlated with the size of the subgroup. By using downstream gender and age classifiers and examining latent embeddings, we highlight the vulnerability of subgroups like older women, who are prone to misclassification due to adversarial perturbations pushing their representations toward those of other subgroups.

摘要: 尽管自动编码器(AE)在降维、表示学习和数据生成等任务中取得了进步，但它们仍然容易受到对手的攻击。变分自动编码器(VAE)以其概率方法分离潜在空间，与确定性自动编码器相比，表现出更强的抗扰动能力；然而，它们对对手输入的弹性仍然是一个令人担忧的问题。这项研究通过优化最小样本特定扰动以在不同的人口统计亚组(年龄和性别组合)上造成最大损害，来评估VAE对非目标对抗性攻击的稳健性。我们调查了两个问题：子组之间是否存在稳健性差异，以及数据稀缺和表征纠缠等因素对这些差异的影响。我们的发现表明稳健性差异是存在的，但并不总是与子组的大小相关。通过使用下游的性别和年龄分类器并检查潜在嵌入，我们强调了像老年女性这样的子组的脆弱性，由于对抗性的扰动将她们的表征推向其他子组，她们容易被错误分类。



## **41. True image construction in quantum-secured single-pixel imaging under spoofing attack**

欺骗攻击下量子安全单像素成像中的真实图像构建 quant-ph

10 pages, 6 figures

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2312.03465v3) [paper-pdf](http://arxiv.org/pdf/2312.03465v3)

**Authors**: Jaesung Heo, Taek Jeong, Nam Hun Park, Yonggi Jo

**Abstract**: In this paper, we introduce a quantum-secured single-pixel imaging (QS-SPI) technique designed to withstand spoofing attacks, wherein adversaries attempt to deceive imaging systems with fake signals. Unlike previous quantum-secured protocols that impose a threshold error rate limiting their operation, even with the existence of true signals, our approach not only identifies spoofing attacks but also facilitates the reconstruction of a true image. Our method involves the analysis of a specific mode correlation of a photon-pair, which is independent of the mode used for image construction, to check security. Through this analysis, we can identify both the targeted image region by the attack and the type of spoofing attack, enabling reconstruction of the true image. A proof-of-principle demonstration employing polarization-correlation of a photon-pair is provided, showcasing successful image reconstruction even under the condition of spoofing signals 2000 times stronger than the true signals. We expect our approach to be applied to quantum-secured signal processing such as quantum target detection or ranging.

摘要: 在这篇文章中，我们介绍了一种量子安全单像素成像(QS-SPI)技术，旨在抵抗欺骗攻击，其中攻击者试图用虚假信号欺骗成像系统。与以前的量子安全协议不同，即使在真实信号存在的情况下，我们的方法也会施加阈值错误率来限制它们的操作，我们的方法不仅可以识别欺骗攻击，还可以帮助重建真实图像。我们的方法包括分析与用于图像构建的模式无关的光子对的特定模式相关性，以检查安全性。通过这种分析，我们可以根据攻击和欺骗攻击的类型来识别目标图像区域，从而能够重建出真实的图像。提供了使用光子对的偏振相关的原理证明演示，展示了即使在欺骗信号比真实信号强2000倍的情况下也成功地重建图像。我们希望将我们的方法应用于量子安全信号处理，如量子目标检测或测距。



## **42. On the Adversarial Robustness of Learning-based Image Compression Against Rate-Distortion Attacks**

基于学习的图像压缩对率失真攻击的对抗鲁棒性 eess.IV

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2405.07717v2) [paper-pdf](http://arxiv.org/pdf/2405.07717v2)

**Authors**: Chenhao Wu, Qingbo Wu, Haoran Wei, Shuai Chen, Lei Wang, King Ngi Ngan, Fanman Meng, Hongliang Li

**Abstract**: Despite demonstrating superior rate-distortion (RD) performance, learning-based image compression (LIC) algorithms have been found to be vulnerable to malicious perturbations in recent studies. However, the adversarial attacks considered in existing literature remain divergent from real-world scenarios, both in terms of the attack direction and bitrate. Additionally, existing methods focus solely on empirical observations of the model vulnerability, neglecting to identify the origin of it. These limitations hinder the comprehensive investigation and in-depth understanding of the adversarial robustness of LIC algorithms. To address the aforementioned issues, this paper considers the arbitrary nature of the attack direction and the uncontrollable compression ratio faced by adversaries, and presents two practical rate-distortion attack paradigms, i.e., Specific-ratio Rate-Distortion Attack (SRDA) and Agnostic-ratio Rate-Distortion Attack (ARDA). Using the performance variations as indicators, we evaluate the adversarial robustness of eight predominant LIC algorithms against diverse attacks. Furthermore, we propose two novel analytical tools for in-depth analysis, i.e., Entropy Causal Intervention and Layer-wise Distance Magnify Ratio, and reveal that hyperprior significantly increases the bitrate and Inverse Generalized Divisive Normalization (IGDN) significantly amplifies input perturbations when under attack. Lastly, we examine the efficacy of adversarial training and introduce the use of online updating for defense. By comparing their advantages and disadvantages, we provide a reference for constructing more robust LIC algorithms against the rate-distortion attacks.

摘要: 尽管基于学习的图像压缩(LIC)算法表现出优异的率失真(RD)性能，但在最近的研究中发现LIC算法容易受到恶意干扰。然而，现有文献中考虑的对抗性攻击在攻击方向和比特率方面仍然与真实世界的场景不同。此外，现有的方法只关注对模型漏洞的经验观察，而忽略了识别其来源。这些局限性阻碍了对LIC算法对抗性稳健性的全面研究和深入理解。针对上述问题，本文考虑了攻击方向的随意性和敌手面临的不可控的压缩比，提出了两种实用的率失真攻击范式，即比比率率失真攻击(SRDA)和不可知率失真攻击(ARDA)。以性能变化为指标，评估了8种主流LIC算法对抗不同攻击的健壮性。此外，我们还提出了两种新的深入分析工具，即熵因果干预和逐层距离放大比，结果表明，超先验显著提高了比特率，而逆广义分裂归一化(IGDN)显著放大了攻击时的输入扰动。最后，我们考察了对抗性训练的有效性，并介绍了在线更新防御的使用。通过比较它们的优缺点，为构造更健壮的抗率失真攻击的LIC算法提供参考。



## **43. Charging Ahead: A Hierarchical Adversarial Framework for Counteracting Advanced Cyber Threats in EV Charging Stations**

提前充电：用于对抗电动汽车充电站高级网络威胁的分层对抗框架 cs.CR

Accepted for IEEE VTC Spring 2024

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.03729v1) [paper-pdf](http://arxiv.org/pdf/2407.03729v1)

**Authors**: Mohammed Al-Mehdhar, Abdullatif Albaseer, Mohamed Abdallah, Ala Al-Fuqaha

**Abstract**: The increasing popularity of electric vehicles (EVs) necessitates robust defenses against sophisticated cyber threats. A significant challenge arises when EVs intentionally provide false information to gain higher charging priority, potentially causing grid instability. While various approaches have been proposed in existing literature to address this issue, they often overlook the possibility of attackers using advanced techniques like deep reinforcement learning (DRL) or other complex deep learning methods to achieve such attacks. In response to this, this paper introduces a hierarchical adversarial framework using DRL (HADRL), which effectively detects stealthy cyberattacks on EV charging stations, especially those leading to denial of charging. Our approach includes a dual approach, where the first scheme leverages DRL to develop advanced and stealthy attack methods that can bypass basic intrusion detection systems (IDS). Second, we implement a DRL-based scheme within the IDS at EV charging stations, aiming to detect and counter these sophisticated attacks. This scheme is trained with datasets created from the first scheme, resulting in a robust and efficient IDS. We evaluated the effectiveness of our framework against the recent literature approaches, and the results show that our IDS can accurately detect deceptive EVs with a low false alarm rate, even when confronted with attacks not represented in the training dataset.

摘要: 电动汽车(EVS)的日益流行需要强大的防御措施来抵御复杂的网络威胁。当电动汽车故意提供虚假信息以获得更高的充电优先级时，一个重大挑战就会出现，这可能会导致电网不稳定。虽然现有文献中提出了各种方法来解决这个问题，但它们往往忽略了攻击者使用深度强化学习(DRL)等高级技术或其他复杂的深度学习方法来实现此类攻击的可能性。针对这一问题，本文提出了一种基于DRL的层次化攻击框架(HADRL)，该框架能够有效地检测到针对电动汽车充电站的隐蔽网络攻击，特别是那些导致拒绝充电的攻击。我们的方法包括一种双重方法，其中第一种方案利用DRL来开发高级的隐形攻击方法，可以绕过基本的入侵检测系统(IDS)。其次，我们在电动汽车充电站的入侵检测系统中实现了一个基于DRL的方案，旨在检测和对抗这些复杂的攻击。该方案利用第一种方案生成的数据集进行训练，得到了一个健壮而高效的入侵检测系统。我们对比最近的文献方法对我们的框架的有效性进行了评估，结果表明，我们的入侵检测系统能够以较低的误警率准确地检测出欺骗性电动汽车，即使面对训练数据集中没有表示的攻击也是如此。



## **44. Pseudorandom Permutations from Random Reversible Circuits**

随机可逆电路的伪随机排列 cs.CC

v2: added references and comparison to subsequent work, removed claim  in previous Section 7.3 with error in proof

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2404.14648v2) [paper-pdf](http://arxiv.org/pdf/2404.14648v2)

**Authors**: William He, Ryan O'Donnell

**Abstract**: We study pseudorandomness properties of permutations on $\{0,1\}^n$ computed by random circuits made from reversible $3$-bit gates (permutations on $\{0,1\}^3$). Our main result is that a random circuit of depth $n \cdot \tilde{O}(k^2)$, with each layer consisting of $\approx n/3$ random gates in a fixed nearest-neighbor architecture, yields almost $k$-wise independent permutations. The main technical component is showing that the Markov chain on $k$-tuples of $n$-bit strings induced by a single random $3$-bit nearest-neighbor gate has spectral gap at least $1/n \cdot \tilde{O}(k)$. This improves on the original work of Gowers [Gowers96], who showed a gap of $1/\mathrm{poly}(n,k)$ for one random gate (with non-neighboring inputs); and, on subsequent work [HMMR05,BH08] improving the gap to $\Omega(1/n^2k)$ in the same setting.   From the perspective of cryptography, our result can be seen as a particularly simple/practical block cipher construction that gives provable statistical security against attackers with access to $k$~input-output pairs within few rounds. We also show that the Luby--Rackoff construction of pseudorandom permutations from pseudorandom functions can be implemented with reversible circuits. From this, we make progress on the complexity of the Minimum Reversible Circuit Size Problem (MRCSP), showing that block ciphers of fixed polynomial size are computationally secure against arbitrary polynomial-time adversaries, assuming the existence of one-way functions (OWFs).

摘要: 我们研究了由可逆$3$位门($0，1^3$上的置换)构成的随机电路计算的$0，1^n上置换的伪随机性。我们的主要结果是，一个深度为$n\cot\tide{O}(k^2)$的随机电路，每一层由固定最近邻体系结构中的$\约n/3$随机门组成，产生几乎$k$方向的独立排列。主要的技术内容是证明了由单个随机的$3$比特最近邻门产生的$n$比特串的$k$-元组上的马尔可夫链至少有$1/n\cdot\tilde{O}(K)$。这比Gowers[Gowers96]的原始工作有所改进，Gowers[Gowers96]对一个随机门(具有非相邻输入)显示了$1/\mathm{pol}(n，k)$的差距；在随后的工作[HMMR05，BH08]中，在相同设置下将差距改进为$\Omega(1/n^2k)$。从密码学的角度来看，我们的结果可以看作是一种特别简单实用的分组密码构造，它提供了针对在几轮内访问$k$~输入输出对的攻击者的可证明的统计安全性。我们还证明了伪随机函数的伪随机置换的Luby-Rackoff构造可以用可逆电路实现。由此，我们在最小可逆电路大小问题(MRCSP)的复杂性方面取得了进展，表明在假设存在单向函数(OWF)的情况下，固定多项式大小的分组密码在计算上是安全的，可以抵抗任意多项式时间的攻击者。



## **45. Jailbreaking Black Box Large Language Models in Twenty Queries**

二十分钟内越狱黑匣子大型语言模型 cs.LG

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2310.08419v3) [paper-pdf](http://arxiv.org/pdf/2310.08419v3)

**Authors**: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong

**Abstract**: There is growing interest in ensuring that large language models (LLMs) align with human values. However, the alignment of such models is vulnerable to adversarial jailbreaks, which coax LLMs into overriding their safety guardrails. The identification of these vulnerabilities is therefore instrumental in understanding inherent weaknesses and preventing future misuse. To this end, we propose Prompt Automatic Iterative Refinement (PAIR), an algorithm that generates semantic jailbreaks with only black-box access to an LLM. PAIR -- which is inspired by social engineering attacks -- uses an attacker LLM to automatically generate jailbreaks for a separate targeted LLM without human intervention. In this way, the attacker LLM iteratively queries the target LLM to update and refine a candidate jailbreak. Empirically, PAIR often requires fewer than twenty queries to produce a jailbreak, which is orders of magnitude more efficient than existing algorithms. PAIR also achieves competitive jailbreaking success rates and transferability on open and closed-source LLMs, including GPT-3.5/4, Vicuna, and Gemini.

摘要: 人们对确保大型语言模型(LLM)与人类价值观保持一致的兴趣与日俱增。然而，这类模型的调整很容易受到对抗性越狱的影响，这会诱使低收入国家凌驾于他们的安全护栏之上。因此，确定这些漏洞有助于了解固有的弱点并防止今后的滥用。为此，我们提出了即时自动迭代求精(Pair)，这是一种仅通过黑盒访问LLM来生成语义越狱的算法。Pair受到社会工程攻击的启发，它使用攻击者LLM自动为单独的目标LLM生成越狱，而无需人工干预。通过这种方式，攻击者LLM迭代地查询目标LLM以更新和改进候选越狱。根据经验，Pair通常只需要不到20次查询就可以产生越狱，这比现有算法的效率高出几个数量级。Pair还在开放和封闭源代码的LLM上实现了具有竞争力的越狱成功率和可转移性，包括GPT-3.5/4、维库纳和双子座。



## **46. JailBreakV-28K: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

JailBreakV-28 K：评估多模式大型语言模型对抗越狱攻击的稳健性的基准 cs.CR

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2404.03027v3) [paper-pdf](http://arxiv.org/pdf/2404.03027v3)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.

摘要: 随着多模式大型语言模型(MLLMS)的快速发展，保护这些模型不受恶意输入的影响，同时使它们与人类的价值观保持一致，已经成为一项关键的挑战。在本文中，我们研究了一个重要而未被探索的问题，即成功越狱大语言模型(LLMS)的技术是否可以在越狱MLLM中同样有效。为了探讨这一问题，我们引入了JailBreakV-28K，这是一个开创性的基准测试，旨在评估LLM越狱技术到MLLM的可转移性，从而评估MLLMS对各种越狱攻击的健壮性。利用本文提出的包含2,000个恶意查询的数据集，我们使用针对LLMS的高级越狱攻击生成了20,000个基于文本的越狱提示，以及来自最近MLLMS越狱攻击的8,000个基于图像的越狱输入，我们的综合数据集包括来自各种对抗场景的28,000个测试用例。我们对10个开源MLLMS的评估显示，对于从LLMS转移的攻击，攻击成功率(ASR)非常高，这突显了MLLMS中源于其文本处理能力的一个严重漏洞。我们的发现强调了未来研究的迫切需要，以解决MLLMS中从文本和视觉输入的对齐漏洞。



## **47. On Large Language Models in National Security Applications**

国家安全应用中的大型语言模型 cs.CR

20 pages

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03453v1) [paper-pdf](http://arxiv.org/pdf/2407.03453v1)

**Authors**: William N. Caballero, Phillip R. Jenkins

**Abstract**: The overwhelming success of GPT-4 in early 2023 highlighted the transformative potential of large language models (LLMs) across various sectors, including national security. This article explores the implications of LLM integration within national security contexts, analyzing their potential to revolutionize information processing, decision-making, and operational efficiency. Whereas LLMs offer substantial benefits, such as automating tasks and enhancing data analysis, they also pose significant risks, including hallucinations, data privacy concerns, and vulnerability to adversarial attacks. Through their coupling with decision-theoretic principles and Bayesian reasoning, LLMs can significantly improve decision-making processes within national security organizations. Namely, LLMs can facilitate the transition from data to actionable decisions, enabling decision-makers to quickly receive and distill available information with less manpower. Current applications within the US Department of Defense and beyond are explored, e.g., the USAF's use of LLMs for wargaming and automatic summarization, that illustrate their potential to streamline operations and support decision-making. However, these applications necessitate rigorous safeguards to ensure accuracy and reliability. The broader implications of LLM integration extend to strategic planning, international relations, and the broader geopolitical landscape, with adversarial nations leveraging LLMs for disinformation and cyber operations, emphasizing the need for robust countermeasures. Despite exhibiting "sparks" of artificial general intelligence, LLMs are best suited for supporting roles rather than leading strategic decisions. Their use in training and wargaming can provide valuable insights and personalized learning experiences for military personnel, thereby improving operational readiness.

摘要: 2023年初GPT-4的压倒性成功突显了大型语言模型(LLM)在包括国家安全在内的各个部门的变革潜力。本文探讨了国家安全背景下LLM集成的含义，分析了它们为信息处理、决策和操作效率带来革命性变化的潜力。虽然LLMS提供了大量的好处，如自动化任务和增强数据分析，但它们也带来了重大风险，包括幻觉、数据隐私问题和易受对手攻击。通过与决策理论原则和贝叶斯推理相结合，LLMS可以显著改善国家安全组织内的决策过程。也就是说，低成本管理系统可以促进从数据到可操作决策的转变，使决策者能够以更少的人力快速接收和提取可用的信息。探讨了美国国防部内外目前的应用，例如，美国空军使用LLM进行战争游戏和自动摘要，说明了它们在简化操作和支持决策方面的潜力。然而，这些应用需要严格的安全措施来确保准确性和可靠性。LLM一体化的更广泛影响延伸到战略规划、国际关系和更广泛的地缘政治格局，敌对国家利用LLM进行虚假信息和网络行动，强调需要强有力的对策。尽管LLM展现出人工智能的“火花”，但它们最适合扮演辅助角色，而不是领导战略决策。它们在训练和战争游戏中的使用可以为军事人员提供有价值的见解和个性化的学习经验，从而提高作战准备。



## **48. Correlated Privacy Mechanisms for Differentially Private Distributed Mean Estimation**

用于差异私有分布均值估计的相关隐私机制 cs.IT

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03289v1) [paper-pdf](http://arxiv.org/pdf/2407.03289v1)

**Authors**: Sajani Vithana, Viveck R. Cadambe, Flavio P. Calmon, Haewon Jeong

**Abstract**: Differentially private distributed mean estimation (DP-DME) is a fundamental building block in privacy-preserving federated learning, where a central server estimates the mean of $d$-dimensional vectors held by $n$ users while ensuring $(\epsilon,\delta)$-DP. Local differential privacy (LDP) and distributed DP with secure aggregation (SecAgg) are the most common notions of DP used in DP-DME settings with an untrusted server. LDP provides strong resilience to dropouts, colluding users, and malicious server attacks, but suffers from poor utility. In contrast, SecAgg-based DP-DME achieves an $O(n)$ utility gain over LDP in DME, but requires increased communication and computation overheads and complex multi-round protocols to handle dropouts and malicious attacks. In this work, we propose CorDP-DME, a novel DP-DME mechanism that spans the gap between DME with LDP and distributed DP, offering a favorable balance between utility and resilience to dropout and collusion. CorDP-DME is based on correlated Gaussian noise, ensuring DP without the perfect conditional privacy guarantees of SecAgg-based approaches. We provide an information-theoretic analysis of CorDP-DME, and derive theoretical guarantees for utility under any given privacy parameters and dropout/colluding user thresholds. Our results demonstrate that (anti) correlated Gaussian DP mechanisms can significantly improve utility in mean estimation tasks compared to LDP -- even in adversarial settings -- while maintaining better resilience to dropouts and attacks compared to distributed DP.

摘要: 差分私有分布平均估计(DP-DME)是保护隐私的联合学习的基本构件，其中中央服务器估计$n$用户所持有的$d$维向量的平均值，同时确保$(？，？)$-DP。本地差异隐私(LDP)和带安全聚合的分布式DP(SecAgg)是DP-DME设置中使用不受信任服务器的最常见概念。LDP对辍学、串通用户和恶意服务器攻击具有很强的弹性，但实用性较差。相比之下，基于SecAgg的DP-DME在DME中比LDP获得$O(N)$效用收益，但需要更多的通信和计算开销以及复杂的多轮协议来处理丢弃和恶意攻击。在这项工作中，我们提出了一种新的DP-DME机制CorDP-DME，它跨越了DME与LDP和分布式DP之间的差距，在实用性和抗丢弃和共谋能力之间提供了良好的平衡。CorDP-DME基于相关的高斯噪声，在没有基于SecAgg的方法的完美条件隐私保证的情况下确保DP。我们对CorDP-DME进行了信息论分析，并推导出在任何给定的隐私参数和丢弃/合谋用户阈值下的效用的理论保证。我们的结果表明，与LDP相比，(反)相关的高斯DP机制可以显著提高均值估计任务的实用性--即使在对抗环境中--同时与分布式DP相比，保持更好的对丢弃和攻击的弹性。



## **49. Self-Evaluation as a Defense Against Adversarial Attacks on LLMs**

自我评估作为对LLM的对抗攻击的防御 cs.LG

8 pages, 7 figures

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03234v1) [paper-pdf](http://arxiv.org/pdf/2407.03234v1)

**Authors**: Hannah Brown, Leon Lin, Kenji Kawaguchi, Michael Shieh

**Abstract**: When LLMs are deployed in sensitive, human-facing settings, it is crucial that they do not output unsafe, biased, or privacy-violating outputs. For this reason, models are both trained and instructed to refuse to answer unsafe prompts such as "Tell me how to build a bomb." We find that, despite these safeguards, it is possible to break model defenses simply by appending a space to the end of a model's input. In a study of eight open-source models, we demonstrate that this acts as a strong enough attack to cause the majority of models to generate harmful outputs with very high success rates. We examine the causes of this behavior, finding that the contexts in which single spaces occur in tokenized training data encourage models to generate lists when prompted, overriding training signals to refuse to answer unsafe requests. Our findings underscore the fragile state of current model alignment and promote the importance of developing more robust alignment methods. Code and data will be made available at https://github.com/Linlt-leon/Adversarial-Alignments.

摘要: 当LLM部署在敏感的、面向人的环境中时，至关重要的是它们不输出不安全、有偏见或违反隐私的输出。出于这个原因，模特们既接受了培训，又被指示拒绝回答不安全的提示，比如“告诉我如何制造炸弹。”我们发现，尽管有这些保障措施，但只需在模型输入的末尾添加一个空格，就可以打破模型的防御。在对八个开源模型的研究中，我们证明了这是一种足够强大的攻击，足以导致大多数模型产生非常高的成功率的有害输出。我们研究了这种行为的原因，发现在标记化的训练数据中出现单个空格的上下文鼓励模型在得到提示时生成列表，从而覆盖拒绝回答不安全请求的训练信号。我们的发现强调了当前模型比对的脆弱状态，并促进了开发更稳健的比对方法的重要性。代码和数据将在https://github.com/Linlt-leon/Adversarial-Alignments.上提供



## **50. Venomancer: Towards Imperceptible and Target-on-Demand Backdoor Attacks in Federated Learning**

毒液杀手：联邦学习中的不可感知和按需定向后门攻击 cs.CV

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03144v1) [paper-pdf](http://arxiv.org/pdf/2407.03144v1)

**Authors**: Son Nguyen, Thinh Nguyen, Khoa Doan, Kok-Seng Wong

**Abstract**: Federated Learning (FL) is a distributed machine learning approach that maintains data privacy by training on decentralized data sources. Similar to centralized machine learning, FL is also susceptible to backdoor attacks. Most backdoor attacks in FL assume a predefined target class and require control over a large number of clients or knowledge of benign clients' information. Furthermore, they are not imperceptible and are easily detected by human inspection due to clear artifacts left on the poison data. To overcome these challenges, we propose Venomancer, an effective backdoor attack that is imperceptible and allows target-on-demand. Specifically, imperceptibility is achieved by using a visual loss function to make the poison data visually indistinguishable from the original data. Target-on-demand property allows the attacker to choose arbitrary target classes via conditional adversarial training. Additionally, experiments showed that the method is robust against state-of-the-art defenses such as Norm Clipping, Weak DP, Krum, and Multi-Krum. The source code is available at https://anonymous.4open.science/r/Venomancer-3426.

摘要: 联合学习(FL)是一种分布式机器学习方法，通过对分散的数据源进行训练来维护数据隐私。与集中式机器学习类似，FL也容易受到后门攻击。FL中的大多数后门攻击假设一个预定义的目标类，并需要控制大量客户端或了解良性客户端的信息。此外，由于毒物数据上留下了明显的伪影，它们并不是不可察觉的，并且很容易被人类检查发现。为了克服这些挑战，我们提出了毒液杀手，这是一种有效的后门攻击，可以潜移默化，并允许按需锁定目标。具体地说，不可感知性是通过使用视觉损失函数来实现的，以使有毒数据在视觉上与原始数据不可区分。按需目标属性允许攻击者通过有条件的对抗性训练选择任意目标类。此外，实验表明，该方法对诸如Norm裁剪、弱DP、Krum和多Krum等最先进的防御方法具有很强的鲁棒性。源代码可在https://anonymous.4open.science/r/Venomancer-3426.上找到



