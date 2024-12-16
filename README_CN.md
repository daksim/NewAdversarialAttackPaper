# Latest Adversarial Attack Papers
**update at 2024-12-16 09:59:59**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. A Semi Black-Box Adversarial Bit-Flip Attack with Limited DNN Model Information**

DNN模型信息有限的半黑匣子对抗位翻转攻击 cs.CR

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2412.09450v1) [paper-pdf](http://arxiv.org/pdf/2412.09450v1)

**Authors**: Behnam Ghavami, Mani Sadati, Mohammad Shahidzadeh, Lesley Shannon, Steve Wilton

**Abstract**: Despite the rising prevalence of deep neural networks (DNNs) in cyber-physical systems, their vulnerability to adversarial bit-flip attacks (BFAs) is a noteworthy concern. This paper proposes B3FA, a semi-black-box BFA-based parameter attack on DNNs, assuming the adversary has limited knowledge about the model. We consider practical scenarios often feature a more restricted threat model for real-world systems, contrasting with the typical BFA models that presuppose the adversary's full access to a network's inputs and parameters. The introduced bit-flip approach utilizes a magnitude-based ranking method and a statistical re-construction technique to identify the vulnerable bits. We demonstrate the effectiveness of B3FA on several DNN models in a semi-black-box setting. For example, B3FA could drop the accuracy of a MobileNetV2 from 69.84% to 9% with only 20 bit-flips in a real-world setting.

摘要: 尽管深度神经网络（DNN）在网络物理系统中的普及率越来越高，但它们对对抗性位翻转攻击（BFA）的脆弱性是一个值得关注的问题。本文提出了B3 FA，这是一种对DNN的基于BFA的半黑匣子参数攻击，假设对手对该模型的了解有限。我们认为，实际场景通常具有现实世界系统更受限制的威胁模型，与假设对手完全访问网络输入和参数的典型BFA模型形成鲜明对比。引入的位翻转方法利用基于幅度的排名方法和统计重建技术来识别脆弱位。我们在半黑匣子环境中在几个DNN模型上展示了B3 FA的有效性。例如，在现实环境中，B3 FA只需20个位翻转即可将MobileNetV2的准确率从69.84%降低到9%。



## **2. On the Robustness of Kolmogorov-Arnold Networks: An Adversarial Perspective**

论科尔莫戈洛夫-阿诺德网络的鲁棒性：对抗的视角 cs.CV

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2408.13809v2) [paper-pdf](http://arxiv.org/pdf/2408.13809v2)

**Authors**: Tal Alter, Raz Lapid, Moshe Sipper

**Abstract**: Kolmogorov-Arnold Networks (KANs) have recently emerged as a novel approach to function approximation, demonstrating remarkable potential in various domains. Despite their theoretical promise, the robustness of KANs under adversarial conditions has yet to be thoroughly examined. In this paper we explore the adversarial robustness of KANs, with a particular focus on image classification tasks. We assess the performance of KANs against standard white box and black-box adversarial attacks, comparing their resilience to that of established neural network architectures. Our experimental evaluation encompasses a variety of standard image classification benchmark datasets and investigates both fully connected and convolutional neural network architectures, of three sizes: small, medium, and large. We conclude that small- and medium-sized KANs (either fully connected or convolutional) are not consistently more robust than their standard counterparts, but that large-sized KANs are, by and large, more robust. This comprehensive evaluation of KANs in adversarial scenarios offers the first in-depth analysis of KAN security, laying the groundwork for future research in this emerging field.

摘要: Kolmogorov-Arnold网络(KANS)是最近出现的一种新的函数逼近方法，在各个领域显示出巨大的潜力。尽管它们在理论上有希望，但KANS在对抗条件下的健壮性尚未得到彻底的检验。在这篇文章中，我们探索了KANS的对抗稳健性，特别关注图像分类任务。我们评估了人工神经网络对标准白盒和黑盒对抗攻击的性能，比较了它们与已建立的神经网络结构的弹性。我们的实验评估涵盖了各种标准图像分类基准数据集，并研究了三种规模的完全连接和卷积神经网络结构：小型、中型和大型。我们的结论是，小型和中型KAN(无论是完全连接的还是卷积的)并不总是比它们的标准对应产品更健壮，但总的来说，大型KAN更健壮。这项对对抗性情景下的KANS的全面评估提供了第一次对KAN安全性的深入分析，为这一新兴领域的未来研究奠定了基础。



## **3. FedAA: A Reinforcement Learning Perspective on Adaptive Aggregation for Fair and Robust Federated Learning**

FedAA：从强化学习角度探讨公平和稳健的联邦学习的自适应聚合 cs.LG

AAAI 2025

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2402.05541v2) [paper-pdf](http://arxiv.org/pdf/2402.05541v2)

**Authors**: Jialuo He, Wei Chen, Xiaojin Zhang

**Abstract**: Federated Learning (FL) has emerged as a promising approach for privacy-preserving model training across decentralized devices. However, it faces challenges such as statistical heterogeneity and susceptibility to adversarial attacks, which can impact model robustness and fairness. Personalized FL attempts to provide some relief by customizing models for individual clients. However, it falls short in addressing server-side aggregation vulnerabilities. We introduce a novel method called \textbf{FedAA}, which optimizes client contributions via \textbf{A}daptive \textbf{A}ggregation to enhance model robustness against malicious clients and ensure fairness across participants in non-identically distributed settings. To achieve this goal, we propose an approach involving a Deep Deterministic Policy Gradient-based algorithm for continuous control of aggregation weights, an innovative client selection method based on model parameter distances, and a reward mechanism guided by validation set performance. Empirically, extensive experiments demonstrate that, in terms of robustness, \textbf{FedAA} outperforms the state-of-the-art methods, while maintaining comparable levels of fairness, offering a promising solution to build resilient and fair federated systems. Our code is available at https://github.com/Gp1g/FedAA.

摘要: 联合学习(FL)已经成为跨分散设备进行隐私保护模型培训的一种很有前途的方法。然而，它面临着诸如统计异构性和对对手攻击的敏感性等挑战，这可能会影响模型的稳健性和公平性。个性化FL试图通过为个人客户定制模型来提供一些缓解。然而，它在解决服务器端聚合漏洞方面做得不够。提出了一种新的方法-.为了实现这一目标，我们提出了一种基于深度确定性策略梯度的聚合权重连续控制算法，一种基于模型参数距离的创新客户选择方法，以及一种以验证集性能为导向的奖励机制。大量的实验表明，在健壮性方面，Fedaa的性能优于最先进的方法，同时保持了相当的公平性，为构建具有弹性和公平的联邦系统提供了一种有前途的解决方案。我们的代码可以在https://github.com/Gp1g/FedAA.上找到



## **4. On the Generation and Removal of Speaker Adversarial Perturbation for Voice-Privacy Protection**

语音隐私保护中说话人对抗性扰动的产生和消除 cs.SD

6 pages, 3 figures, published to IEEE SLT Workshop 2024

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2412.09195v1) [paper-pdf](http://arxiv.org/pdf/2412.09195v1)

**Authors**: Chenyang Guo, Liping Chen, Zhuhai Li, Kong Aik Lee, Zhen-Hua Ling, Wu Guo

**Abstract**: Neural networks are commonly known to be vulnerable to adversarial attacks mounted through subtle perturbation on the input data. Recent development in voice-privacy protection has shown the positive use cases of the same technique to conceal speaker's voice attribute with additive perturbation signal generated by an adversarial network. This paper examines the reversibility property where an entity generating the adversarial perturbations is authorized to remove them and restore original speech (e.g., the speaker him/herself). A similar technique could also be used by an investigator to deanonymize a voice-protected speech to restore criminals' identities in security and forensic analysis. In this setting, the perturbation generative module is assumed to be known in the removal process. To this end, a joint training of perturbation generation and removal modules is proposed. Experimental results on the LibriSpeech dataset demonstrated that the subtle perturbations added to the original speech can be predicted from the anonymized speech while achieving the goal of privacy protection. By removing these perturbations from the anonymized sample, the original speech can be restored. Audio samples can be found in \url{https://voiceprivacy.github.io/Perturbation-Generation-Removal/}.

摘要: 众所周知，神经网络容易受到通过对输入数据进行微妙扰动而发起的对抗性攻击。语音隐私保护方面的最新进展表明，利用对抗性网络产生的加性扰动信号来隐藏说话人的语音属性的相同技术的积极使用案例。本文考察了可逆性，其中产生对抗性扰动的实体被授权移除它们并恢复原始语音(例如说话人他/她自己)。调查员也可以使用类似的技术对受声音保护的语音进行去匿名化，以在安全和法医分析中恢复罪犯的身份。在该设置中，假设在移除过程中已知扰动生成模块。为此，提出了一种扰动产生和消除模块的联合训练。在LibriSpeech数据集上的实验结果表明，在达到隐私保护的目的的同时，可以从匿名语音中预测到添加到原始语音中的细微扰动。通过从匿名样本中去除这些扰动，可以恢复原始语音。音频样本可在\url{https://voiceprivacy.github.io/Perturbation-Generation-Removal/}.中找到



## **5. Evaluating Adversarial Attacks on Traffic Sign Classifiers beyond Standard Baselines**

评估对超出标准基线的交通标志分类器的对抗攻击 cs.CV

Accepted for publication at ICMLA 2024

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2412.09150v1) [paper-pdf](http://arxiv.org/pdf/2412.09150v1)

**Authors**: Svetlana Pavlitska, Leopold Müller, J. Marius Zöllner

**Abstract**: Adversarial attacks on traffic sign classification models were among the first successfully tried in the real world. Since then, the research in this area has been mainly restricted to repeating baseline models, such as LISA-CNN or GTSRB-CNN, and similar experiment settings, including white and black patches on traffic signs. In this work, we decouple model architectures from the datasets and evaluate on further generic models to make a fair comparison. Furthermore, we compare two attack settings, inconspicuous and visible, which are usually regarded without direct comparison. Our results show that standard baselines like LISA-CNN or GTSRB-CNN are significantly more susceptible than the generic ones. We, therefore, suggest evaluating new attacks on a broader spectrum of baselines in the future. Our code is available at \url{https://github.com/KASTEL-MobilityLab/attacks-on-traffic-sign-recognition/}.

摘要: 对交通标志分类模型的对抗攻击是在现实世界中首次成功尝试的攻击之一。从那时起，该领域的研究主要局限于重复基线模型，例如LISA-CNN或GTSRB-CNN，以及类似的实验环境，包括交通标志上的白色和黑色斑块。在这项工作中，我们将模型架构与数据集分离，并评估进一步的通用模型，以进行公平的比较。此外，我们还比较了两种攻击设置，即不引人注目的和可见的，这两种设置通常被认为是没有直接比较的。我们的结果表明，LISA-CNN或GTSRB-CNN等标准基线明显比通用基线更容易受到影响。因此，我们建议未来在更广泛的基线上评估新的攻击。我们的代码可在\url{https：//github.com/KASTEL-MobilityLab/attacks-on-traffic-SIGNITION/}上获取。



## **6. Unlearning or Concealment? A Critical Analysis and Evaluation Metrics for Unlearning in Diffusion Models**

忘记还是隐瞒？扩散模型中放弃学习的批判性分析和评估工具包 cs.LG

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2409.05668v2) [paper-pdf](http://arxiv.org/pdf/2409.05668v2)

**Authors**: Aakash Sen Sharma, Niladri Sarkar, Vikram Chundawat, Ankur A Mali, Murari Mandal

**Abstract**: Recent research has seen significant interest in methods for concept removal and targeted forgetting in text-to-image diffusion models. In this paper, we conduct a comprehensive white-box analysis showing the vulnerabilities in existing diffusion model unlearning methods. We show that existing unlearning methods lead to decoupling of the targeted concepts (meant to be forgotten) for the corresponding prompts. This is concealment and not actual forgetting, which was the original goal. This paper presents a rigorous theoretical and empirical examination of five commonly used techniques for unlearning in diffusion models, while showing their potential weaknesses. We introduce two new evaluation metrics: Concept Retrieval Score (\textbf{CRS}) and Concept Confidence Score (\textbf{CCS}). These metrics are based on a successful adversarial attack setup that can recover \textit{forgotten} concepts from unlearned diffusion models. \textbf{CRS} measures the similarity between the latent representations of the unlearned and fully trained models after unlearning. It reports the extent of retrieval of the \textit{forgotten} concepts with increasing amount of guidance. CCS quantifies the confidence of the model in assigning the target concept to the manipulated data. It reports the probability of the \textit{unlearned} model's generations to be aligned with the original domain knowledge with increasing amount of guidance. The \textbf{CCS} and \textbf{CRS} enable a more robust evaluation of concept erasure methods. Evaluating existing five state-of-the-art methods with our metrics, reveal significant shortcomings in their ability to truly \textit{unlearn}. Source Code: \color{blue}{https://respailab.github.io/unlearning-or-concealment}

摘要: 最近的研究对文本到图像扩散模型中的概念移除和目标遗忘的方法产生了浓厚的兴趣。在本文中，我们进行了全面的白盒分析，显示了现有扩散模型遗忘方法的漏洞。我们表明，现有的遗忘方法导致了对应提示的目标概念(意在被遗忘)的分离。这是隐藏，而不是真正的遗忘，这是最初的目标。本文对扩散模型中五种常用的遗忘技术进行了严格的理论和实证检验，同时指出了它们的潜在弱点。我们引入了两个新的评价指标：概念检索得分(Textbf{ccs})和概念置信度得分(Textbf{ccs})。这些度量基于成功的对抗性攻击设置，该设置可以从未学习的扩散模型中恢复{忘记}概念。文本bf{crs}度量未学习的模型的潜在表示与去学习后的完全训练的模型之间的相似性。它报告了检索被遗忘的概念的程度，并提供了越来越多的指导。CCS在将目标概念分配给被操纵的数据时量化模型的置信度。它报告了随着指导量的增加，模型生成与原始领域知识保持一致的概率。使用\extbf{ccs}和\extbf{crs}可以对概念擦除方法进行更可靠的评估。使用我们的指标评估现有的五种最先进的方法，发现它们在真正\文本{取消学习}的能力方面存在重大缺陷。源代码：\color{blue}{https://respailab.github.io/unlearning-or-concealment}



## **7. Deep Learning Model Security: Threats and Defenses**

深度学习模型安全性：威胁和防御 cs.CR

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2412.08969v1) [paper-pdf](http://arxiv.org/pdf/2412.08969v1)

**Authors**: Tianyang Wang, Ziqian Bi, Yichao Zhang, Ming Liu, Weiche Hsieh, Pohsun Feng, Lawrence K. Q. Yan, Yizhu Wen, Benji Peng, Junyu Liu, Keyu Chen, Sen Zhang, Ming Li, Chuanqi Jiang, Xinyuan Song, Junjie Yang, Bowen Jing, Jintao Ren, Junhao Song, Hong-Ming Tseng, Silin Chen, Yunze Wang, Chia Xin Liang, Jiawei Xu, Xuanhe Pan, Jinlang Wang, Qian Niu

**Abstract**: Deep learning has transformed AI applications but faces critical security challenges, including adversarial attacks, data poisoning, model theft, and privacy leakage. This survey examines these vulnerabilities, detailing their mechanisms and impact on model integrity and confidentiality. Practical implementations, including adversarial examples, label flipping, and backdoor attacks, are explored alongside defenses such as adversarial training, differential privacy, and federated learning, highlighting their strengths and limitations.   Advanced methods like contrastive and self-supervised learning are presented for enhancing robustness. The survey concludes with future directions, emphasizing automated defenses, zero-trust architectures, and the security challenges of large AI models. A balanced approach to performance and security is essential for developing reliable deep learning systems.

摘要: 深度学习改变了人工智能应用程序，但面临着关键的安全挑战，包括对抗性攻击、数据中毒、模型盗窃和隐私泄露。本调查检查了这些漏洞，详细介绍了它们的机制以及对模型完整性和机密性的影响。实践实现，包括对抗性示例、标签翻转和后门攻击，与对抗性训练、差异隐私和联邦学习等防御措施一起进行了探讨，强调了它们的优点和局限性。   提出了对比学习和自我监督学习等先进方法来增强鲁棒性。该调查得出了未来的方向，强调自动化防御、零信任架构以及大型人工智能模型的安全挑战。平衡的性能和安全方法对于开发可靠的深度学习系统至关重要。



## **8. Respect the model: Fine-grained and Robust Explanation with Sharing Ratio Decomposition**

尊重模型：通过共享比分解进行细粒度且稳健的解释 cs.CV

To be published in ICLR 2024

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2402.03348v2) [paper-pdf](http://arxiv.org/pdf/2402.03348v2)

**Authors**: Sangyu Han, Yearim Kim, Nojun Kwak

**Abstract**: The truthfulness of existing explanation methods in authentically elucidating the underlying model's decision-making process has been questioned. Existing methods have deviated from faithfully representing the model, thus susceptible to adversarial attacks. To address this, we propose a novel eXplainable AI (XAI) method called SRD (Sharing Ratio Decomposition), which sincerely reflects the model's inference process, resulting in significantly enhanced robustness in our explanations. Different from the conventional emphasis on the neuronal level, we adopt a vector perspective to consider the intricate nonlinear interactions between filters. We also introduce an interesting observation termed Activation-Pattern-Only Prediction (APOP), letting us emphasize the importance of inactive neurons and redefine relevance encapsulating all relevant information including both active and inactive neurons. Our method, SRD, allows for the recursive decomposition of a Pointwise Feature Vector (PFV), providing a high-resolution Effective Receptive Field (ERF) at any layer.

摘要: 现有的解释方法在真实地阐明潜在模型的决策过程方面的真实性受到了质疑。现有的方法已经偏离了忠实地表示模型，因此容易受到对抗性攻击。针对这一问题，我们提出了一种新的可解释AI(XAI)方法SRD(Share Ratio Decomacy)，该方法真实地反映了模型的推理过程，显著增强了解释的健壮性。不同于传统的对神经元层次的强调，我们采用了向量的视角来考虑过滤器之间复杂的非线性相互作用。我们还引入了一种有趣的观察，称为仅激活模式预测(APOP)，它让我们强调非活动神经元的重要性，并重新定义相关性，封装了所有相关信息，包括活动和非活动神经元。我们的方法，SRD，允许递归分解点特征向量(PFV)，在任何层提供高分辨率的有效感受野(ERF)。



## **9. Flexible Physical Camouflage Generation Based on a Differential Approach**

基于差异方法的灵活物理服装生成 cs.CV

**SubmitDate**: 2024-12-12    [abs](http://arxiv.org/abs/2402.13575v3) [paper-pdf](http://arxiv.org/pdf/2402.13575v3)

**Authors**: Yang Li, Wenyi Tan, Tingrui Wang, Xinkai Liang, Quan Pan

**Abstract**: This study introduces a novel approach to neural rendering, specifically tailored for adversarial camouflage, within an extensive 3D rendering framework. Our method, named FPA, goes beyond traditional techniques by faithfully simulating lighting conditions and material variations, ensuring a nuanced and realistic representation of textures on a 3D target. To achieve this, we employ a generative approach that learns adversarial patterns from a diffusion model. This involves incorporating a specially designed adversarial loss and covert constraint loss to guarantee the adversarial and covert nature of the camouflage in the physical world. Furthermore, we showcase the effectiveness of the proposed camouflage in sticker mode, demonstrating its ability to cover the target without compromising adversarial information. Through empirical and physical experiments, FPA exhibits strong performance in terms of attack success rate and transferability. Additionally, the designed sticker-mode camouflage, coupled with a concealment constraint, adapts to the environment, yielding diverse styles of texture. Our findings highlight the versatility and efficacy of the FPA approach in adversarial camouflage applications.

摘要: 这项研究介绍了一种新的神经渲染方法，专门为对抗性伪装而定制，在一个广泛的3D渲染框架内。我们的方法名为FPA，超越了传统技术，忠实地模拟了照明条件和材质变化，确保了3D目标上纹理的细微差别和逼真表示。为了实现这一点，我们采用了一种生成性方法，从扩散模型中学习对抗性模式。这涉及到包括特别设计的对抗性损失和隐蔽约束损失，以保证物理世界中伪装的对抗性和隐蔽性。此外，我们在贴纸模式下展示了所提出的伪装的有效性，展示了其在不损害敌方信息的情况下覆盖目标的能力。通过实验和物理实验，FPA在攻击成功率和可转移性方面表现出很强的性能。此外，设计的贴纸模式伪装，加上隐藏限制，适应环境，产生不同风格的纹理。我们的发现突出了FPA方法在对抗性伪装应用中的多功能性和有效性。



## **10. AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization**

AICAttack：基于注意力的优化的对抗性图像字幕攻击 cs.CV

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2402.11940v4) [paper-pdf](http://arxiv.org/pdf/2402.11940v4)

**Authors**: Jiyao Li, Mingze Ni, Yifei Dong, Tianqing Zhu, Wei Liu

**Abstract**: Recent advances in deep learning research have shown remarkable achievements across many tasks in computer vision (CV) and natural language processing (NLP). At the intersection of CV and NLP is the problem of image captioning, where the related models' robustness against adversarial attacks has not been well studied. This paper presents a novel adversarial attack strategy, AICAttack (Attention-based Image Captioning Attack), designed to attack image captioning models through subtle perturbations on images. Operating within a black-box attack scenario, our algorithm requires no access to the target model's architecture, parameters, or gradient information. We introduce an attention-based candidate selection mechanism that identifies the optimal pixels to attack, followed by a customised differential evolution method to optimise the perturbations of pixels' RGB values. We demonstrate AICAttack's effectiveness through extensive experiments on benchmark datasets against multiple victim models. The experimental results demonstrate that our method outperforms current leading-edge techniques by achieving consistently higher attack success rates.

摘要: 近年来，深度学习研究在计算机视觉(CV)和自然语言处理(NLP)领域取得了令人瞩目的成就。在CV和NLP的交叉点是图像字幕问题，相关模型对敌意攻击的稳健性还没有得到很好的研究。提出了一种新的对抗性攻击策略AICAttack(基于注意力的图像字幕攻击)，旨在通过对图像进行微妙的扰动来攻击图像字幕模型。我们的算法在黑盒攻击场景中运行，不需要访问目标模型的体系结构、参数或梯度信息。我们引入了一种基于注意力的候选选择机制来确定要攻击的最佳像素，然后使用定制的差异进化方法来优化像素RGB值的扰动。通过在基准数据集上针对多个受害者模型的大量实验，我们展示了AICAttack的有效性。实验结果表明，我们的方法比目前的前沿技术具有更高的攻击成功率。



## **11. Exploiting the Index Gradients for Optimization-Based Jailbreaking on Large Language Models**

利用索引要素对大型语言模型进行基于优化的越狱 cs.CL

13 pages,2 figures, accepted by The 31st International Conference on  Computational Linguistics

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08615v1) [paper-pdf](http://arxiv.org/pdf/2412.08615v1)

**Authors**: Jiahui Li, Yongchang Hao, Haoyu Xu, Xing Wang, Yu Hong

**Abstract**: Despite the advancements in training Large Language Models (LLMs) with alignment techniques to enhance the safety of generated content, these models remain susceptible to jailbreak, an adversarial attack method that exposes security vulnerabilities in LLMs. Notably, the Greedy Coordinate Gradient (GCG) method has demonstrated the ability to automatically generate adversarial suffixes that jailbreak state-of-the-art LLMs. However, the optimization process involved in GCG is highly time-consuming, rendering the jailbreaking pipeline inefficient. In this paper, we investigate the process of GCG and identify an issue of Indirect Effect, the key bottleneck of the GCG optimization. To this end, we propose the Model Attack Gradient Index GCG (MAGIC), that addresses the Indirect Effect by exploiting the gradient information of the suffix tokens, thereby accelerating the procedure by having less computation and fewer iterations. Our experiments on AdvBench show that MAGIC achieves up to a 1.5x speedup, while maintaining Attack Success Rates (ASR) on par or even higher than other baselines. Our MAGIC achieved an ASR of 74% on the Llama-2 and an ASR of 54% when conducting transfer attacks on GPT-3.5. Code is available at https://github.com/jiah-li/magic.

摘要: 尽管在使用对齐技术训练大型语言模型(LLM)以增强生成内容的安全性方面取得了进展，但这些模型仍然容易受到越狱的影响，这是一种暴露LLM安全漏洞的对抗性攻击方法。值得注意的是，贪婪坐标梯度(GCG)方法已经展示了自动生成敌意后缀的能力，这些后缀是越狱最先进的LLM。然而，GCG涉及的优化过程非常耗时，使得越狱管道效率低下。在本文中，我们研究了GCG的过程，并找出了间接影响的问题，这是GCG优化的关键瓶颈。为此，我们提出了模型攻击梯度索引GCG(MAGIC)，它通过利用后缀标记的梯度信息来解决间接影响，从而以更少的计算量和更少的迭代来加速过程。我们在AdvBtch上的实验表明，Magic在保持攻击成功率(ASR)与其他基线相当甚至更高的情况下，实现了高达1.5倍的加速。我们的魔法在骆驼-2上达到了74%的ASR，当对GPT-3.5进行传输攻击时ASR达到54%。代码可在https://github.com/jiah-li/magic.上找到



## **12. AdvWave: Stealthy Adversarial Jailbreak Attack against Large Audio-Language Models**

AdvWave：针对大型音频语言模型的隐形对抗越狱攻击 cs.SD

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08608v1) [paper-pdf](http://arxiv.org/pdf/2412.08608v1)

**Authors**: Mintong Kang, Chejian Xu, Bo Li

**Abstract**: Recent advancements in large audio-language models (LALMs) have enabled speech-based user interactions, significantly enhancing user experience and accelerating the deployment of LALMs in real-world applications. However, ensuring the safety of LALMs is crucial to prevent risky outputs that may raise societal concerns or violate AI regulations. Despite the importance of this issue, research on jailbreaking LALMs remains limited due to their recent emergence and the additional technical challenges they present compared to attacks on DNN-based audio models. Specifically, the audio encoders in LALMs, which involve discretization operations, often lead to gradient shattering, hindering the effectiveness of attacks relying on gradient-based optimizations. The behavioral variability of LALMs further complicates the identification of effective (adversarial) optimization targets. Moreover, enforcing stealthiness constraints on adversarial audio waveforms introduces a reduced, non-convex feasible solution space, further intensifying the challenges of the optimization process. To overcome these challenges, we develop AdvWave, the first jailbreak framework against LALMs. We propose a dual-phase optimization method that addresses gradient shattering, enabling effective end-to-end gradient-based optimization. Additionally, we develop an adaptive adversarial target search algorithm that dynamically adjusts the adversarial optimization target based on the response patterns of LALMs for specific queries. To ensure that adversarial audio remains perceptually natural to human listeners, we design a classifier-guided optimization approach that generates adversarial noise resembling common urban sounds. Extensive evaluations on multiple advanced LALMs demonstrate that AdvWave outperforms baseline methods, achieving a 40% higher average jailbreak attack success rate.

摘要: 大型音频语言模型(LALM)的最新进展实现了基于语音的用户交互，显著增强了用户体验，并加快了LALM在现实世界应用中的部署。然而，确保LALM的安全对于防止可能引发社会担忧或违反人工智能法规的高风险输出至关重要。尽管这一问题很重要，但由于最近出现的LALM以及与攻击基于DNN的音频模型相比带来的额外技术挑战，对越狱LALM的研究仍然有限。具体地说，LALMS中的音频编码器涉及离散化操作，经常会导致梯度破碎，阻碍了依赖于基于梯度优化的攻击的有效性。LALMS的行为变异性使有效(对抗性)优化目标的识别变得更加复杂。此外，对敌方音频波形实施隐蔽性约束会引入一个缩减的非凸可行解空间，从而进一步加剧了优化过程的挑战。为了克服这些挑战，我们开发了第一个针对LALMS的越狱框架AdvWave。我们提出了一种双阶段优化方法，解决了梯度破碎问题，实现了有效的端到端基于梯度的优化。此外，我们还开发了一种自适应对抗性目标搜索算法，该算法根据LALMS对特定查询的响应模式动态调整对抗性优化目标。为了确保对抗性音频对人类听众来说保持感知上的自然，我们设计了一种分类器引导的优化方法，该方法产生类似于常见城市声音的对抗性噪声。对多个高级LALM的广泛评估表明，AdvWave的性能优于基准方法，实现了40%的平均越狱攻击成功率。



## **13. Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts**

彩虹团队：开放式一代的多元化对抗预言 cs.CL

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2402.16822v3) [paper-pdf](http://arxiv.org/pdf/2402.16822v3)

**Authors**: Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H. Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, Tim Rocktäschel, Roberta Raileanu

**Abstract**: As large language models (LLMs) become increasingly prevalent across many real-world applications, understanding and enhancing their robustness to adversarial attacks is of paramount importance. Existing methods for identifying adversarial prompts tend to focus on specific domains, lack diversity, or require extensive human annotations. To address these limitations, we present Rainbow Teaming, a novel black-box approach for producing a diverse collection of adversarial prompts. Rainbow Teaming casts adversarial prompt generation as a quality-diversity problem and uses open-ended search to generate prompts that are both effective and diverse. Focusing on the safety domain, we use Rainbow Teaming to target various state-of-the-art LLMs, including the Llama 2 and Llama 3 models. Our approach reveals hundreds of effective adversarial prompts, with an attack success rate exceeding 90% across all tested models. Furthermore, we demonstrate that prompts generated by Rainbow Teaming are highly transferable and that fine-tuning models with synthetic data generated by our method significantly enhances their safety without sacrificing general performance or helpfulness. We additionally explore the versatility of Rainbow Teaming by applying it to question answering and cybersecurity, showcasing its potential to drive robust open-ended self-improvement in a wide range of applications.

摘要: 随着大型语言模型(LLM)在许多真实世界的应用中变得越来越普遍，理解和增强它们对对手攻击的健壮性是至关重要的。现有的识别对抗性提示的方法往往集中在特定的领域，缺乏多样性，或者需要大量的人工注释。为了解决这些局限性，我们提出了彩虹分组，这是一种新的黑盒方法，用于产生多样化的对抗性提示集合。彩虹团队将敌意提示生成视为质量多样性问题，并使用开放式搜索来生成既有效又多样化的提示。专注于安全领域，我们使用彩虹团队瞄准各种最先进的LLM，包括Llama 2和Llama 3型号。我们的方法揭示了数百个有效的对抗性提示，在所有测试模型上的攻击成功率超过90%。此外，我们证明了彩虹组合生成的提示具有很高的可转移性，并且使用我们的方法生成的合成数据对模型进行微调显著地增强了它们的安全性，而不会牺牲一般性能或帮助。我们还探讨了彩虹团队的多功能性，将其应用于问题回答和网络安全，展示了其在广泛应用中推动强大的开放式自我改进的潜力。



## **14. Grimm: A Plug-and-Play Perturbation Rectifier for Graph Neural Networks Defending against Poisoning Attacks**

Grimm：用于图神经网络防御中毒攻击的即插即用微扰矫正器 cs.LG

19 pages, 13 figures

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08555v1) [paper-pdf](http://arxiv.org/pdf/2412.08555v1)

**Authors**: Ao Liu, Wenshan Li, Beibei Li, Wengang Ma, Tao Li, Pan Zhou

**Abstract**: End-to-end training with global optimization have popularized graph neural networks (GNNs) for node classification, yet inadvertently introduced vulnerabilities to adversarial edge-perturbing attacks. Adversaries can exploit the inherent opened interfaces of GNNs' input and output, perturbing critical edges and thus manipulating the classification results. Current defenses, due to their persistent utilization of global-optimization-based end-to-end training schemes, inherently encapsulate the vulnerabilities of GNNs. This is specifically evidenced in their inability to defend against targeted secondary attacks. In this paper, we propose the Graph Agent Network (GAgN) to address the aforementioned vulnerabilities of GNNs. GAgN is a graph-structured agent network in which each node is designed as an 1-hop-view agent. Through the decentralized interactions between agents, they can learn to infer global perceptions to perform tasks including inferring embeddings, degrees and neighbor relationships for given nodes. This empowers nodes to filtering adversarial edges while carrying out classification tasks. Furthermore, agents' limited view prevents malicious messages from propagating globally in GAgN, thereby resisting global-optimization-based secondary attacks. We prove that single-hidden-layer multilayer perceptrons (MLPs) are theoretically sufficient to achieve these functionalities. Experimental results show that GAgN effectively implements all its intended capabilities and, compared to state-of-the-art defenses, achieves optimal classification accuracy on the perturbed datasets.

摘要: 具有全局优化的端到端训练普及了图神经网络(GNN)用于节点分类，但无意中引入了对敌意边缘扰动攻击的脆弱性。攻击者可以利用GNN输入和输出固有的开放接口，扰乱关键边缘，从而操纵分类结果。目前的防御措施由于持续使用基于全局优化的端到端培训方案，固有地封装了GNN的脆弱性。这一点具体表现在他们无法防御有针对性的二次攻击。在本文中，我们提出了图代理网络(GagN)来解决GNN的上述漏洞。GAGN是一个图结构的代理网络，其中每个节点被设计为一个1跳视图代理。通过代理之间的分散交互，它们可以学习推断全局感知来执行任务，包括推断给定节点的嵌入度、度数和邻居关系。这使节点能够在执行分类任务时过滤敌意边缘。此外，代理的有限视点防止恶意消息在GAGN中全局传播，从而抵抗基于全局优化的二次攻击。我们证明了单隐层多层感知器(MLP)理论上足以实现这些功能。实验结果表明，GAGN有效地实现了其预期的所有功能，并且与现有的防御措施相比，在扰动数据集上获得了最优的分类精度。



## **15. Robust Deep Reinforcement Learning Through Adversarial Attacks and Training : A Survey**

通过对抗性攻击和训练进行稳健的深度强化学习：一项调查 cs.LG

61 pages, 17 figues, 1 table

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2403.00420v2) [paper-pdf](http://arxiv.org/pdf/2403.00420v2)

**Authors**: Lucas Schott, Josephine Delas, Hatem Hajri, Elies Gherbi, Reda Yaich, Nora Boulahia-Cuppens, Frederic Cuppens, Sylvain Lamprier

**Abstract**: Deep Reinforcement Learning (DRL) is a subfield of machine learning for training autonomous agents that take sequential actions across complex environments. Despite its significant performance in well-known environments, it remains susceptible to minor condition variations, raising concerns about its reliability in real-world applications. To improve usability, DRL must demonstrate trustworthiness and robustness. A way to improve the robustness of DRL to unknown changes in the environmental conditions and possible perturbations is through Adversarial Training, by training the agent against well-suited adversarial attacks on the observations and the dynamics of the environment. Addressing this critical issue, our work presents an in-depth analysis of contemporary adversarial attack and training methodologies, systematically categorizing them and comparing their objectives and operational mechanisms.

摘要: 深度强化学习（DRL）是机器学习的一个子领域，用于训练在复杂环境中采取顺序动作的自主代理。尽管它在众所周知的环境中表现出色，但它仍然容易受到微小的条件变化的影响，这引发了人们对其在现实应用中可靠性的担忧。为了提高可用性，DRL必须证明可信性和稳健性。提高DRL对环境条件未知变化和可能扰动的稳健性的一种方法是通过对抗训练，通过训练代理人免受对观察和环境动态的适当对抗攻击。针对这一关键问题，我们的工作对当代对抗性攻击和训练方法进行了深入分析，对其进行系统性分类并比较其目标和操作机制。



## **16. Adversarial Purification by Consistency-aware Latent Space Optimization on Data Manifolds**

通过一致性感知的数据集合体潜在空间优化进行对抗性净化 cs.LG

17 pages, 8 figures

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08394v1) [paper-pdf](http://arxiv.org/pdf/2412.08394v1)

**Authors**: Shuhai Zhang, Jiahao Yang, Hui Luo, Jie Chen, Li Wang, Feng Liu, Bo Han, Mingkui Tan

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial samples crafted by adding imperceptible perturbations to clean data, potentially leading to incorrect and dangerous predictions. Adversarial purification has been an effective means to improve DNNs robustness by removing these perturbations before feeding the data into the model. However, it faces significant challenges in preserving key structural and semantic information of data, as the imperceptible nature of adversarial perturbations makes it hard to avoid over-correcting, which can destroy important information and degrade model performance. In this paper, we break away from traditional adversarial purification methods by focusing on the clean data manifold. To this end, we reveal that samples generated by a well-trained generative model are close to clean ones but far from adversarial ones. Leveraging this insight, we propose Consistency Model-based Adversarial Purification (CMAP), which optimizes vectors within the latent space of a pre-trained consistency model to generate samples for restoring clean data. Specifically, 1) we propose a \textit{Perceptual consistency restoration} mechanism by minimizing the discrepancy between generated samples and input samples in both pixel and perceptual spaces. 2) To maintain the optimized latent vectors within the valid data manifold, we introduce a \textit{Latent distribution consistency constraint} strategy to align generated samples with the clean data distribution. 3) We also apply a \textit{Latent vector consistency prediction} scheme via an ensemble approach to enhance prediction reliability. CMAP fundamentally addresses adversarial perturbations at their source, providing a robust purification. Extensive experiments on CIFAR-10 and ImageNet-100 show that our CMAP significantly enhances robustness against strong adversarial attacks while preserving high natural accuracy.

摘要: 深度神经网络(DNN)很容易受到敌意样本的攻击，这些样本是通过在干净的数据中添加不可察觉的扰动而制作的，可能会导致错误和危险的预测。对抗性净化是通过在将数据输入模型之前消除这些扰动来提高DNN稳健性的有效手段。然而，它在保存数据的关键结构和语义信息方面面临着巨大的挑战，因为对抗性扰动的不可察觉性质使得它很难避免过度校正，这可能会破坏重要信息并降低模型的性能。在本文中，我们摆脱了传统的对抗性净化方法，将重点放在干净的数据流形上。为此，我们揭示了由训练有素的生成模型生成的样本接近干净的样本，但远离对抗性的样本。利用这一观点，我们提出了基于一致性模型的对抗净化算法(CMAP)，它在预先训练的一致性模型的潜在空间内优化向量，以生成用于恢复干净数据的样本。具体地说，1)通过最小化像素空间和感知空间中生成样本和输入样本之间的差异，我们提出了一种文本一致性恢复机制。2)为了在有效数据流形内保持最优的潜在向量，我们引入了一种潜在分布一致性约束策略来将生成的样本与干净的数据分布对齐。3)我们还通过集成方法应用了潜在向量一致性预测方案来提高预测的可靠性。Cmap从根本上解决了对抗性扰动的根源，提供了强大的净化。在CIFAR-10和ImageNet-100上的大量实验表明，我们的CMAP在保持高自然准确率的同时，显著增强了对强对手攻击的稳健性。



## **17. Graph Agent Network: Empowering Nodes with Inference Capabilities for Adversarial Resilience**

图代理网络：赋予节点推理能力以对抗复原力 cs.LG

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2306.06909v5) [paper-pdf](http://arxiv.org/pdf/2306.06909v5)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Guangquan Xu, Pan Zhou, Wengang Ma, Hanyuan Huang

**Abstract**: End-to-end training with global optimization have popularized graph neural networks (GNNs) for node classification, yet inadvertently introduced vulnerabilities to adversarial edge-perturbing attacks. Adversaries can exploit the inherent opened interfaces of GNNs' input and output, perturbing critical edges and thus manipulating the classification results. Current defenses, due to their persistent utilization of global-optimization-based end-to-end training schemes, inherently encapsulate the vulnerabilities of GNNs. This is specifically evidenced in their inability to defend against targeted secondary attacks. In this paper, we propose the Graph Agent Network (GAgN) to address the aforementioned vulnerabilities of GNNs. GAgN is a graph-structured agent network in which each node is designed as an 1-hop-view agent. Through the decentralized interactions between agents, they can learn to infer global perceptions to perform tasks including inferring embeddings, degrees and neighbor relationships for given nodes. This empowers nodes to filtering adversarial edges while carrying out classification tasks. Furthermore, agents' limited view prevents malicious messages from propagating globally in GAgN, thereby resisting global-optimization-based secondary attacks. We prove that single-hidden-layer multilayer perceptrons (MLPs) are theoretically sufficient to achieve these functionalities. Experimental results show that GAgN effectively implements all its intended capabilities and, compared to state-of-the-art defenses, achieves optimal classification accuracy on the perturbed datasets.

摘要: 具有全局优化的端到端训练普及了图神经网络(GNN)用于节点分类，但无意中引入了对敌意边缘扰动攻击的脆弱性。攻击者可以利用GNN输入和输出固有的开放接口，扰乱关键边缘，从而操纵分类结果。目前的防御措施由于持续使用基于全局优化的端到端培训方案，固有地封装了GNN的脆弱性。这一点具体表现在他们无法防御有针对性的二次攻击。在本文中，我们提出了图代理网络(GagN)来解决GNN的上述漏洞。GAGN是一个图结构的代理网络，其中每个节点被设计为一个1跳视图代理。通过代理之间的分散交互，它们可以学习推断全局感知来执行任务，包括推断给定节点的嵌入度、度数和邻居关系。这使节点能够在执行分类任务时过滤敌意边缘。此外，代理的有限视点防止恶意消息在GAGN中全局传播，从而抵抗基于全局优化的二次攻击。我们证明了单隐层多层感知器(MLP)理论上足以实现这些功能。实验结果表明，GAGN有效地实现了其预期的所有功能，并且与现有的防御措施相比，在扰动数据集上获得了最优的分类精度。



## **18. How Does the Smoothness Approximation Method Facilitate Generalization for Federated Adversarial Learning?**

光滑度逼近方法如何促进联邦对抗学习的推广？ cs.LG

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08282v1) [paper-pdf](http://arxiv.org/pdf/2412.08282v1)

**Authors**: Wenjun Ding, Ying An, Lixing Chen, Shichao Kan, Fan Wu, Zhe Qu

**Abstract**: Federated Adversarial Learning (FAL) is a robust framework for resisting adversarial attacks on federated learning. Although some FAL studies have developed efficient algorithms, they primarily focus on convergence performance and overlook generalization. Generalization is crucial for evaluating algorithm performance on unseen data. However, generalization analysis is more challenging due to non-smooth adversarial loss functions. A common approach to addressing this issue is to leverage smoothness approximation. In this paper, we develop algorithm stability measures to evaluate the generalization performance of two popular FAL algorithms: \textit{Vanilla FAL (VFAL)} and {\it Slack FAL (SFAL)}, using three different smooth approximation methods: 1) \textit{Surrogate Smoothness Approximation (SSA)}, (2) \textit{Randomized Smoothness Approximation (RSA)}, and (3) \textit{Over-Parameterized Smoothness Approximation (OPSA)}. Based on our in-depth analysis, we answer the question of how to properly set the smoothness approximation method to mitigate generalization error in FAL. Moreover, we identify RSA as the most effective method for reducing generalization error. In highly data-heterogeneous scenarios, we also recommend employing SFAL to mitigate the deterioration of generalization performance caused by heterogeneity. Based on our theoretical results, we provide insights to help develop more efficient FAL algorithms, such as designing new metrics and dynamic aggregation rules to mitigate heterogeneity.

摘要: 联合对抗学习(FAL)是一种用于抵抗对联合学习的敌意攻击的健壮框架。虽然一些FAL研究已经开发出了有效的算法，但它们主要集中在收敛性能上，而忽略了泛化。泛化是在未知数据上评估算法性能的关键。然而，由于对抗性损失函数的非光滑性，泛化分析具有更大的挑战性。解决此问题的一种常见方法是利用平滑近似。本文利用3种不同的光滑逼近方法：1)替代平滑逼近(SSA)}、(2)随机平滑逼近(RSA)}和(3)过参数平滑逼近(OPSA)}，对两种常见的FAL算法在深入分析的基础上，我们回答了如何合理地设置光滑度逼近方法来减小FAL中的泛化误差的问题。此外，我们认为RSA是减少泛化误差的最有效方法。在数据高度异构性的场景中，我们还建议使用SFAL来缓解异构性导致的泛化性能下降。基于我们的理论结果，我们提供了一些见解来帮助开发更高效的FAL算法，例如设计新的度量和动态聚集规则来缓解异构性。



## **19. DG-Mamba: Robust and Efficient Dynamic Graph Structure Learning with Selective State Space Models**

DG-Mamba：使用选择性状态空间模型稳健高效的动态图结构学习 cs.LG

Accepted by the Main Technical Track of the 39th Annual AAAI  Conference on Artificial Intelligence (AAAI-2025)

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2412.08160v2) [paper-pdf](http://arxiv.org/pdf/2412.08160v2)

**Authors**: Haonan Yuan, Qingyun Sun, Zhaonan Wang, Xingcheng Fu, Cheng Ji, Yongjian Wang, Bo Jin, Jianxin Li

**Abstract**: Dynamic graphs exhibit intertwined spatio-temporal evolutionary patterns, widely existing in the real world. Nevertheless, the structure incompleteness, noise, and redundancy result in poor robustness for Dynamic Graph Neural Networks (DGNNs). Dynamic Graph Structure Learning (DGSL) offers a promising way to optimize graph structures. However, aside from encountering unacceptable quadratic complexity, it overly relies on heuristic priors, making it hard to discover underlying predictive patterns. How to efficiently refine the dynamic structures, capture intrinsic dependencies, and learn robust representations, remains under-explored. In this work, we propose the novel DG-Mamba, a robust and efficient Dynamic Graph structure learning framework with the Selective State Space Models (Mamba). To accelerate the spatio-temporal structure learning, we propose a kernelized dynamic message-passing operator that reduces the quadratic time complexity to linear. To capture global intrinsic dynamics, we establish the dynamic graph as a self-contained system with State Space Model. By discretizing the system states with the cross-snapshot graph adjacency, we enable the long-distance dependencies capturing with the selective snapshot scan. To endow learned dynamic structures more expressive with informativeness, we propose the self-supervised Principle of Relevant Information for DGSL to regularize the most relevant yet least redundant information, enhancing global robustness. Extensive experiments demonstrate the superiority of the robustness and efficiency of our DG-Mamba compared with the state-of-the-art baselines against adversarial attacks.

摘要: 动态图形表现出交织在一起的时空演化模式，广泛存在于现实世界中。然而，动态图神经网络的结构不完备性、噪声和冗余性导致其健壮性较差。动态图结构学习(DGSL)为优化图结构提供了一种很有前途的方法。然而，除了遇到不可接受的二次型复杂性外，它还过度依赖启发式先验，使得发现潜在的预测模式变得困难。如何有效地提炼动态结构，捕获内在依赖关系，并学习健壮的表示，仍未得到探索。在这项工作中，我们提出了一种新颖的DG-MAMBA，这是一种基于选择状态空间模型(MAMBA)的健壮而高效的动态图结构学习框架。为了加速时空结构的学习，我们提出了一种核化的动态消息传递算子，将二次时间复杂度降为线性。为了捕捉全局内在动力学，我们利用状态空间模型将动态图建立为一个自包含系统。通过使用交叉快照图邻接关系对系统状态进行离散化，实现了选择性快照扫描的远程依赖捕获。为了使学习到的动态结构具有更强的信息性，我们提出了DGSL的相关信息自监督原则，将相关程度最高但冗余最少的信息正则化，增强了全局鲁棒性。大量的实验证明了DG-MAMBA算法的健壮性和高效性，与目前最先进的对抗攻击基线算法相比具有更好的性能。



## **20. Antelope: Potent and Concealed Jailbreak Attack Strategy**

羚羊：有力且隐蔽的越狱攻击策略 cs.CR

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08156v1) [paper-pdf](http://arxiv.org/pdf/2412.08156v1)

**Authors**: Xin Zhao, Xiaojun Chen, Haoyu Gao

**Abstract**: Due to the remarkable generative potential of diffusion-based models, numerous researches have investigated jailbreak attacks targeting these frameworks. A particularly concerning threat within image models is the generation of Not-Safe-for-Work (NSFW) content. Despite the implementation of security filters, numerous efforts continue to explore ways to circumvent these safeguards. Current attack methodologies primarily encompass adversarial prompt engineering or concept obfuscation, yet they frequently suffer from slow search efficiency, conspicuous attack characteristics and poor alignment with targets. To overcome these challenges, we propose Antelope, a more robust and covert jailbreak attack strategy designed to expose security vulnerabilities inherent in generative models. Specifically, Antelope leverages the confusion of sensitive concepts with similar ones, facilitates searches in the semantically adjacent space of these related concepts and aligns them with the target imagery, thereby generating sensitive images that are consistent with the target and capable of evading detection. Besides, we successfully exploit the transferability of model-based attacks to penetrate online black-box services. Experimental evaluations demonstrate that Antelope outperforms existing baselines across multiple defensive mechanisms, underscoring its efficacy and versatility.

摘要: 由于基于扩散的模型具有显著的生成潜力，许多研究已经研究了针对这些框架的越狱攻击。图像模型中一个特别令人担忧的威胁是不安全工作(NSFW)内容的生成。尽管实施了安全过滤器，但许多努力仍在继续探索规避这些保障措施的方法。目前的攻击方法主要包括对抗性的提示工程或概念混淆，但它们往往存在搜索效率低、攻击特征明显以及与目标对准不良的问题。为了克服这些挑战，我们提出了Antelope，这是一种更强大和隐蔽的越狱攻击策略，旨在暴露生成式模型中固有的安全漏洞。具体地说，Antelope利用敏感概念与相似概念的混淆，促进在这些相关概念的语义相邻空间中进行搜索，并将它们与目标图像对齐，从而生成与目标一致并能够躲避检测的敏感图像。此外，我们还成功地利用了基于模型的攻击的可转移性来渗透在线黑盒服务。实验评估表明，Antelope在多种防御机制上的表现优于现有基线，突显了其有效性和多功能性。



## **21. Doubly-Universal Adversarial Perturbations: Deceiving Vision-Language Models Across Both Images and Text with a Single Perturbation**

双重普遍对抗性扰动：通过单一扰动欺骗图像和文本的视觉语言模型 cs.CV

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08108v1) [paper-pdf](http://arxiv.org/pdf/2412.08108v1)

**Authors**: Hee-Seon Kim, Minbeom Kim, Changick Kim

**Abstract**: Large Vision-Language Models (VLMs) have demonstrated remarkable performance across multimodal tasks by integrating vision encoders with large language models (LLMs). However, these models remain vulnerable to adversarial attacks. Among such attacks, Universal Adversarial Perturbations (UAPs) are especially powerful, as a single optimized perturbation can mislead the model across various input images. In this work, we introduce a novel UAP specifically designed for VLMs: the Doubly-Universal Adversarial Perturbation (Doubly-UAP), capable of universally deceiving VLMs across both image and text inputs. To successfully disrupt the vision encoder's fundamental process, we analyze the core components of the attention mechanism. After identifying value vectors in the middle-to-late layers as the most vulnerable, we optimize Doubly-UAP in a label-free manner with a frozen model. Despite being developed as a black-box to the LLM, Doubly-UAP achieves high attack success rates on VLMs, consistently outperforming baseline methods across vision-language tasks. Extensive ablation studies and analyses further demonstrate the robustness of Doubly-UAP and provide insights into how it influences internal attention mechanisms.

摘要: 大视觉语言模型(VLM)通过将视觉编码器与大语言模型(LLM)相结合，在多通道任务中表现出了显著的性能。然而，这些模型仍然容易受到对手的攻击。在这些攻击中，通用对抗性扰动(UAP)尤其强大，因为单个优化的扰动可以在不同的输入图像上误导模型。在这项工作中，我们介绍了一种新的专门针对VLMS设计的UAP：双重通用对抗性摄动(Double-Universal Aversarial微扰，Double-UAP)，能够在图像和文本输入之间普遍欺骗VLMS。为了成功地扰乱视觉编码器的基本过程，我们分析了注意机制的核心组件。在确定中后期价值向量最易受攻击后，我们使用冻结模型以无标签的方式对Double-UAP进行优化。尽管被开发为LLM的黑匣子，Double-UAP在VLM上实现了高攻击成功率，在视觉语言任务中始终优于基线方法。广泛的消融研究和分析进一步证明了Double-UAP的健壮性，并提供了对其如何影响内部注意机制的见解。



## **22. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

时间序列预测大型语言模型中的对抗漏洞 cs.LG

11 pages, 5 figures

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08099v1) [paper-pdf](http://arxiv.org/pdf/2412.08099v1)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in the field of time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like TimeGPT and LLM-Time with GPT-3.5, GPT-4, LLaMa, and Mistral, show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications.

摘要: 大型语言模型最近在时间序列预测领域显示出巨大的潜力，在处理复杂的时间数据方面提供了令人印象深刻的能力。然而，它们在实际应用中的健壮性和可靠性仍然没有得到充分的研究，特别是关于它们对对手攻击的敏感性。本文提出了一种基于LLM的时间序列预测的对抗性攻击框架。通过使用无梯度和黑盒优化方法，我们产生了最小但高效的扰动，这些扰动显著降低了跨多个数据集和LLM体系结构的预测精度。我们的实验，包括使用GPT-3.5、GPT-4、LLAMA和Mistral的TimeGPT和LLM-Time模型，表明对抗性攻击导致的性能降级比随机噪声严重得多，并证明了我们的攻击在不同LLM上的广泛有效性。这些结果强调了低层管理在时间序列预测中的关键弱点，强调了需要强大的防御机制来确保其在实际应用中的可靠部署。



## **23. What You See Is Not Always What You Get: An Empirical Study of Code Comprehension by Large Language Models**

你所看到的并不总是你所得到的：大型语言模型对代码理解的实证研究 cs.SE

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08098v1) [paper-pdf](http://arxiv.org/pdf/2412.08098v1)

**Authors**: Bangshuo Zhu, Jiawen Wen, Huaming Chen

**Abstract**: Recent studies have demonstrated outstanding capabilities of large language models (LLMs) in software engineering domain, covering numerous tasks such as code generation and comprehension. While the benefit of LLMs for coding task is well noted, it is perceived that LLMs are vulnerable to adversarial attacks. In this paper, we study the specific LLM vulnerability to imperceptible character attacks, a type of prompt-injection attack that uses special characters to befuddle an LLM whilst keeping the attack hidden to human eyes. We devise four categories of attacks and investigate their effects on the performance outcomes of tasks relating to code analysis and code comprehension. Two generations of ChatGPT are included to evaluate the impact of advancements made to contemporary models. Our experimental design consisted of comparing perturbed and unperturbed code snippets and evaluating two performance outcomes, which are model confidence using log probabilities of response, and correctness of response. We conclude that earlier version of ChatGPT exhibits a strong negative linear correlation between the amount of perturbation and the performance outcomes, while the recent ChatGPT presents a strong negative correlation between the presence of perturbation and performance outcomes, but no valid correlational relationship between perturbation budget and performance outcomes. We anticipate this work contributes to an in-depth understanding of leveraging LLMs for coding tasks. It is suggested future research should delve into how to create LLMs that can return a correct response even if the prompt exhibits perturbations.

摘要: 最近的研究表明，大型语言模型(LLM)在软件工程领域具有卓越的能力，涵盖了代码生成和理解等众多任务。虽然LLMS对于编码任务的好处是众所周知的，但人们认为LLMS容易受到对手的攻击。在本文中，我们研究了特定的LLM对隐蔽字符攻击的脆弱性，这是一种使用特殊字符来迷惑LLM的即时注入攻击，同时将攻击隐藏在人眼之外。我们设计了四类攻击，并调查了它们对与代码分析和代码理解相关的任务的性能结果的影响。两代ChatGPT被包括在内，以评估当代模型所取得的进步的影响。我们的实验设计包括比较扰动和未扰动的代码片段，并评估两个性能结果，即使用响应的对数概率的模型置信度和响应的正确性。我们的结论是，早期版本的ChatGPT表现出扰动量与绩效结果之间的强负相关，而最近版本的ChatGPT呈现出扰动的存在与绩效结果之间的强负相关，但扰动预算与绩效结果之间没有有效的相关关系。我们预计这项工作有助于深入理解利用LLM进行编码任务。有人建议，未来的研究应该深入研究如何创建即使在提示显示扰动的情况下也能返回正确响应的LLM。



## **24. Plentiful Jailbreaks with String Compositions**

弦乐作品丰富越狱 cs.CL

NeurIPS SoLaR Workshop 2024

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2411.01084v3) [paper-pdf](http://arxiv.org/pdf/2411.01084v3)

**Authors**: Brian R. Y. Huang

**Abstract**: Large language models (LLMs) remain vulnerable to a slew of adversarial attacks and jailbreaking methods. One common approach employed by white-hat attackers, or red-teamers, is to process model inputs and outputs using string-level obfuscations, which can include leetspeak, rotary ciphers, Base64, ASCII, and more. Our work extends these encoding-based attacks by unifying them in a framework of invertible string transformations. With invertibility, we can devise arbitrary string compositions, defined as sequences of transformations, that we can encode and decode end-to-end programmatically. We devise a automated best-of-n attack that samples from a combinatorially large number of string compositions. Our jailbreaks obtain competitive attack success rates on several leading frontier models when evaluated on HarmBench, highlighting that encoding-based attacks remain a persistent vulnerability even in advanced LLMs.

摘要: 大型语言模型（LLM）仍然容易受到一系列对抗攻击和越狱方法的影响。白帽攻击者或红团队使用的一种常见方法是使用字符串级混淆处理模型输入和输出，其中可以包括leetspeak、旋转密码、Base 64、ASC等。我们的工作通过将这些基于编码的攻击统一到可逆字符串转换的框架中来扩展它们。通过可逆性，我们可以设计任意的字符串组合，定义为转换序列，我们可以通过编程方式进行端到端编码和解码。我们设计了一种自动化的n中最佳攻击，该攻击从组合上大量的字符串组合中进行采样。在HarmBench上进行评估时，我们的越狱在几个领先的前沿模型上获得了有竞争力的攻击成功率，这凸显了即使在高级LLM中，基于编码的攻击仍然是一个持久的漏洞。



## **25. DynamicPAE: Generating Scene-Aware Physical Adversarial Examples in Real-Time**

DynamicPoker：实时生成场景感知物理对抗示例 cs.CV

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08053v1) [paper-pdf](http://arxiv.org/pdf/2412.08053v1)

**Authors**: Jin Hu, Xianglong Liu, Jiakai Wang, Junkai Zhang, Xianqi Yang, Haotong Qin, Yuqing Ma, Ke Xu

**Abstract**: Physical adversarial examples (PAEs) are regarded as "whistle-blowers" of real-world risks in deep-learning applications. However, current PAE generation studies show limited adaptive attacking ability to diverse and varying scenes. The key challenges in generating dynamic PAEs are exploring their patterns under noisy gradient feedback and adapting the attack to agnostic scenario natures. To address the problems, we present DynamicPAE, the first generative framework that enables scene-aware real-time physical attacks beyond static attacks. Specifically, to train the dynamic PAE generator under noisy gradient feedback, we introduce the residual-driven sample trajectory guidance technique, which redefines the training task to break the limited feedback information restriction that leads to the degeneracy problem. Intuitively, it allows the gradient feedback to be passed to the generator through a low-noise auxiliary task, thereby guiding the optimization away from degenerate solutions and facilitating a more comprehensive and stable exploration of feasible PAEs. To adapt the generator to agnostic scenario natures, we introduce the context-aligned scene expectation simulation process, consisting of the conditional-uncertainty-aligned data module and the skewness-aligned objective re-weighting module. The former enhances robustness in the context of incomplete observation by employing a conditional probabilistic model for domain randomization, while the latter facilitates consistent stealth control across different attack targets by automatically reweighting losses based on the skewness indicator. Extensive digital and physical evaluations demonstrate the superior attack performance of DynamicPAE, attaining a 1.95 $\times$ boost (65.55% average AP drop under attack) on representative object detectors (e.g., Yolo-v8) over state-of-the-art static PAE generating methods.

摘要: 物理对抗性例子(PAE)被认为是深度学习应用中真实世界风险的“告密者”。然而，目前的PAE代研究表明，对不同场景的自适应攻击能力有限。生成动态PAE的关键挑战是在噪声梯度反馈下探索它们的模式，并使攻击适应不可知的场景性质。为了解决这些问题，我们提出了DynamicPAE，这是第一个生成性框架，它能够在静态攻击之外实现场景感知的实时物理攻击。具体地说，为了在噪声梯度反馈下训练动态PAE产生器，我们引入了残差驱动样本轨迹制导技术，重新定义了训练任务，打破了导致退化问题的有限反馈信息限制。直观地说，它允许通过低噪声辅助任务将梯度反馈传递给生成器，从而引导优化远离退化解，并有助于更全面和稳定地探索可行的PAE。为了使生成器适应不可知的场景性质，我们引入了上下文对齐的场景期望模拟过程，由条件不确定性对齐的数据模块和偏度对齐的目标重加权模块组成。前者通过使用区域随机化的条件概率模型来增强不完全观测环境下的鲁棒性，而后者通过基于偏度指标自动重新加权损失来促进对不同攻击目标的一致隐身控制。广泛的数字和物理评估表明，DynamicPAE具有优越的攻击性能，与最先进的静态PAE生成方法相比，在典型对象检测器(如Yolo-V8)上获得了1.95美元\倍的$提升(攻击下平均AP下降65.55%)。



## **26. GLL: A Differentiable Graph Learning Layer for Neural Networks**

GLL：神经网络的可区分图学习层 cs.LG

44 pages, 11 figures. Preprint. Submitted to the Journal of Machine  Learning Research

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08016v1) [paper-pdf](http://arxiv.org/pdf/2412.08016v1)

**Authors**: Jason Brown, Bohan Chen, Harris Hardiman-Mostow, Jeff Calder, Andrea L. Bertozzi

**Abstract**: Standard deep learning architectures used for classification generate label predictions with a projection head and softmax activation function. Although successful, these methods fail to leverage the relational information between samples in the batch for generating label predictions. In recent works, graph-based learning techniques, namely Laplace learning, have been heuristically combined with neural networks for both supervised and semi-supervised learning (SSL) tasks. However, prior works approximate the gradient of the loss function with respect to the graph learning algorithm or decouple the processes; end-to-end integration with neural networks is not achieved. In this work, we derive backpropagation equations, via the adjoint method, for inclusion of a general family of graph learning layers into a neural network. This allows us to precisely integrate graph Laplacian-based label propagation into a neural network layer, replacing a projection head and softmax activation function for classification tasks. Using this new framework, our experimental results demonstrate smooth label transitions across data, improved robustness to adversarial attacks, improved generalization, and improved training dynamics compared to the standard softmax-based approach.

摘要: 用于分类的标准深度学习体系结构生成具有投影头和Softmax激活函数的标签预测。尽管这些方法成功了，但它们无法利用批次中样本之间的关系信息来生成标签预测。在最近的工作中，基于图的学习技术，即拉普拉斯学习，已经被启发式地与神经网络相结合，用于监督和半监督学习(SSL)任务。然而，已有的工作相对于图学习算法近似损失函数的梯度或将过程解耦，无法实现与神经网络的端到端集成。在这项工作中，我们推导出反向传播方程，通过伴随方法，将一个一般的图学习层族包含到神经网络中。这使我们能够精确地将基于图的拉普拉斯标记传播集成到神经网络层，取代用于分类任务的投影头和Softmax激活函数。使用这个新的框架，我们的实验结果表明，与标准的基于Softmax的方法相比，我们的实验结果显示出跨数据的平滑标签转换，提高了对对手攻击的健壮性，改善了泛化，并改善了训练动态。



## **27. MAGIC: Mastering Physical Adversarial Generation in Context through Collaborative LLM Agents**

MAGIC：通过协作LLM代理掌握上下文中的物理对抗生成 cs.CV

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.08014v1) [paper-pdf](http://arxiv.org/pdf/2412.08014v1)

**Authors**: Yun Xing, Nhat Chung, Jie Zhang, Yue Cao, Ivor Tsang, Yang Liu, Lei Ma, Qing Guo

**Abstract**: Physical adversarial attacks in driving scenarios can expose critical vulnerabilities in visual perception models. However, developing such attacks remains challenging due to diverse real-world backgrounds and the requirement for maintaining visual naturality. Building upon this challenge, we reformulate physical adversarial attacks as a one-shot patch-generation problem. Our approach generates adversarial patches through a deep generative model that considers the specific scene context, enabling direct physical deployment in matching environments. The primary challenge lies in simultaneously achieving two objectives: generating adversarial patches that effectively mislead object detection systems while determining contextually appropriate placement within the scene. We propose MAGIC (Mastering Physical Adversarial Generation In Context), a novel framework powered by multi-modal LLM agents to address these challenges. MAGIC automatically understands scene context and orchestrates adversarial patch generation through the synergistic interaction of language and vision capabilities. MAGIC orchestrates three specialized LLM agents: The adv-patch generation agent (GAgent) masters the creation of deceptive patches through strategic prompt engineering for text-to-image models. The adv-patch deployment agent (DAgent) ensures contextual coherence by determining optimal placement strategies based on scene understanding. The self-examination agent (EAgent) completes this trilogy by providing critical oversight and iterative refinement of both processes. We validate our method on both digital and physical level, \ie, nuImage and manually captured real scenes, where both statistical and visual results prove that our MAGIC is powerful and effectively for attacking wide-used object detection systems.

摘要: 驾驶场景中的物理对抗性攻击可以暴露视觉感知模型中的关键漏洞。然而，由于不同的现实世界背景和对保持视觉自然性的要求，开发这样的攻击仍然具有挑战性。在这一挑战的基础上，我们将物理对抗性攻击重新定义为一次性补丁生成问题。我们的方法通过深度生成模型生成对抗性补丁，该模型考虑了特定的场景上下文，支持在匹配环境中直接物理部署。主要的挑战在于同时实现两个目标：生成有效误导目标检测系统的对抗性补丁，同时确定场景中的上下文适当位置。我们提出了MAGIC(掌握上下文中的物理对手生成)，这是一个由多模式LLM代理支持的新框架来应对这些挑战。Magic自动理解场景背景，并通过语言和视觉能力的协同交互来协调对抗性补丁的生成。Magic协调了三个专门的LLM代理：adv-patch生成代理(Gagent)通过针对文本到图像模型的战略提示工程掌握了欺骗性补丁的创建。Adv-patch部署代理(DAgent)通过基于场景理解确定最优放置策略来确保上下文一致性。自我检查代理(EAgent)通过提供对两个过程的关键监督和迭代细化来完成这三部曲。我们在数字和物理两个层面上验证了我们的方法，即NuImage和人工捕获的真实场景，其中统计和视觉结果都证明了我们的魔力对于攻击广泛使用的目标检测系统是强大和有效的。



## **28. Enhancing Remote Adversarial Patch Attacks on Face Detectors with Tiling and Scaling**

通过拼贴和缩放增强对面部检测器的远程对抗补丁攻击 cs.CV

Accepted and Presented at APSIPA ASC 2024

**SubmitDate**: 2024-12-11    [abs](http://arxiv.org/abs/2412.07996v1) [paper-pdf](http://arxiv.org/pdf/2412.07996v1)

**Authors**: Masora Okano, Koichi Ito, Masakatsu Nishigaki, Tetsushi Ohki

**Abstract**: This paper discusses the attack feasibility of Remote Adversarial Patch (RAP) targeting face detectors. The RAP that targets face detectors is similar to the RAP that targets general object detectors, but the former has multiple issues in the attack process the latter does not. (1) It is possible to detect objects of various scales. In particular, the area of small objects that are convolved during feature extraction by CNN is small,so the area that affects the inference results is also small. (2) It is a two-class classification, so there is a large gap in characteristics between the classes. This makes it difficult to attack the inference results by directing them to a different class. In this paper, we propose a new patch placement method and loss function for each problem. The patches targeting the proposed face detector showed superior detection obstruct effects compared to the patches targeting the general object detector.

摘要: 本文讨论了针对人脸检测器的远程对抗补丁（RAP）攻击的可行性。针对面部检测器的RAP与针对一般对象检测器的RAP类似，但前者在攻击过程中存在多个问题，而后者则没有。(1)可以检测各种规模的物体。特别是，CNN在特征提取过程中卷积的小对象的面积很小，因此影响推断结果的面积也很小。(2)它是两类分类，因此类之间的特征差距很大。这使得很难通过将推理结果引导到不同的类来攻击推理结果。本文针对每个问题提出了一种新的补丁放置方法和损失函数。与针对通用对象检测器的贴片相比，针对拟议面部检测器的贴片表现出更好的检测阻碍效应。



## **29. PBP: Post-training Backdoor Purification for Malware Classifiers**

PBP：恶意软件分类器的培训后后门净化 cs.LG

Accepted at NDSS 2025

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.03441v3) [paper-pdf](http://arxiv.org/pdf/2412.03441v3)

**Authors**: Dung Thuy Nguyen, Ngoc N. Tran, Taylor T. Johnson, Kevin Leach

**Abstract**: In recent years, the rise of machine learning (ML) in cybersecurity has brought new challenges, including the increasing threat of backdoor poisoning attacks on ML malware classifiers. For instance, adversaries could inject malicious samples into public malware repositories, contaminating the training data and potentially misclassifying malware by the ML model. Current countermeasures predominantly focus on detecting poisoned samples by leveraging disagreements within the outputs of a diverse set of ensemble models on training data points. However, these methods are not suitable for scenarios where Machine Learning-as-a-Service (MLaaS) is used or when users aim to remove backdoors from a model after it has been trained. Addressing this scenario, we introduce PBP, a post-training defense for malware classifiers that mitigates various types of backdoor embeddings without assuming any specific backdoor embedding mechanism. Our method exploits the influence of backdoor attacks on the activation distribution of neural networks, independent of the trigger-embedding method. In the presence of a backdoor attack, the activation distribution of each layer is distorted into a mixture of distributions. By regulating the statistics of the batch normalization layers, we can guide a backdoored model to perform similarly to a clean one. Our method demonstrates substantial advantages over several state-of-the-art methods, as evidenced by experiments on two datasets, two types of backdoor methods, and various attack configurations. Notably, our approach requires only a small portion of the training data -- only 1\% -- to purify the backdoor and reduce the attack success rate from 100\% to almost 0\%, a 100-fold improvement over the baseline methods. Our code is available at \url{https://github.com/judydnguyen/pbp-backdoor-purification-official}.

摘要: 近年来，机器学习(ML)在网络安全领域的兴起带来了新的挑战，包括对ML恶意软件分类器进行后门中毒攻击的威胁越来越大。例如，攻击者可以将恶意样本注入公共恶意软件存储库中，污染训练数据，并可能根据ML模型对恶意软件进行错误分类。目前的对策主要集中在通过利用关于训练数据点的一组不同集合模型的输出中的不一致来检测有毒样本。然而，这些方法不适用于使用机器学习即服务(MLaaS)的场景，或者用户希望在模型经过训练后删除后门的场景。针对这种情况，我们引入了PBP，这是一种针对恶意软件分类器的训练后防御，它可以减少各种类型的后门嵌入，而不需要假设任何特定的后门嵌入机制。我们的方法利用了后门攻击对神经网络激活分布的影响，独立于触发器嵌入方法。在后门攻击存在的情况下，每一层的激活分布被扭曲为混合分布。通过调整批处理归一化层的统计信息，我们可以引导回溯模型以类似于干净模型的方式执行。在两个数据集、两种类型的后门方法和不同的攻击配置上的实验证明，我们的方法比几种最先进的方法显示出了实质性的优势。值得注意的是，我们的方法只需要一小部分训练数据--只需要1\%--来净化后门，并将攻击成功率从100\%降低到几乎0\%，比基线方法提高了100倍。我们的代码可以在\url{https://github.com/judydnguyen/pbp-backdoor-purification-official}.上找到



## **30. DeMem: Privacy-Enhanced Robust Adversarial Learning via De-Memorization**

DeMem：通过去伪化的隐私增强鲁棒对抗学习 cs.LG

10 pages

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.05767v2) [paper-pdf](http://arxiv.org/pdf/2412.05767v2)

**Authors**: Xiaoyu Luo, Qiongxiu Li

**Abstract**: Adversarial robustness, the ability of a model to withstand manipulated inputs that cause errors, is essential for ensuring the trustworthiness of machine learning models in real-world applications. However, previous studies have shown that enhancing adversarial robustness through adversarial training increases vulnerability to privacy attacks. While differential privacy can mitigate these attacks, it often compromises robustness against both natural and adversarial samples. Our analysis reveals that differential privacy disproportionately impacts low-risk samples, causing an unintended performance drop. To address this, we propose DeMem, which selectively targets high-risk samples, achieving a better balance between privacy protection and model robustness. DeMem is versatile and can be seamlessly integrated into various adversarial training techniques. Extensive evaluations across multiple training methods and datasets demonstrate that DeMem significantly reduces privacy leakage while maintaining robustness against both natural and adversarial samples. These results confirm DeMem's effectiveness and broad applicability in enhancing privacy without compromising robustness.

摘要: 对抗性健壮性，即模型承受导致错误的操纵输入的能力，对于确保机器学习模型在现实世界应用中的可信性至关重要。然而，先前的研究表明，通过对抗性训练来增强对抗性的健壮性会增加对隐私攻击的脆弱性。虽然差异隐私可以缓解这些攻击，但它通常会损害对自然样本和对手样本的稳健性。我们的分析显示，差异隐私对低风险样本的影响不成比例，导致意外的性能下降。为了解决这一问题，我们提出了DeMem，它选择性地针对高风险样本，在隐私保护和模型稳健性之间实现了更好的平衡。DeMem是多才多艺的，可以无缝地整合到各种对抗性训练技术中。对多种训练方法和数据集的广泛评估表明，DeMem显著减少了隐私泄露，同时保持了对自然样本和对手样本的健壮性。这些结果证实了DeMem在增强隐私而不影响健壮性方面的有效性和广泛的适用性。



## **31. Defending Against Neural Network Model Inversion Attacks via Data Poisoning**

通过数据中毒防御神经网络模型倒置攻击 cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07575v1) [paper-pdf](http://arxiv.org/pdf/2412.07575v1)

**Authors**: Shuai Zhou, Dayong Ye, Tianqing Zhu, Wanlei Zhou

**Abstract**: Model inversion attacks pose a significant privacy threat to machine learning models by reconstructing sensitive data from their outputs. While various defenses have been proposed to counteract these attacks, they often come at the cost of the classifier's utility, thus creating a challenging trade-off between privacy protection and model utility. Moreover, most existing defenses require retraining the classifier for enhanced robustness, which is impractical for large-scale, well-established models. This paper introduces a novel defense mechanism to better balance privacy and utility, particularly against adversaries who employ a machine learning model (i.e., inversion model) to reconstruct private data. Drawing inspiration from data poisoning attacks, which can compromise the performance of machine learning models, we propose a strategy that leverages data poisoning to contaminate the training data of inversion models, thereby preventing model inversion attacks.   Two defense methods are presented. The first, termed label-preserving poisoning attacks for all output vectors (LPA), involves subtle perturbations to all output vectors while preserving their labels. Our findings demonstrate that these minor perturbations, introduced through a data poisoning approach, significantly increase the difficulty of data reconstruction without compromising the utility of the classifier. Subsequently, we introduce a second method, label-flipping poisoning for partial output vectors (LFP), which selectively perturbs a small subset of output vectors and alters their labels during the process. Empirical results indicate that LPA is notably effective, outperforming the current state-of-the-art defenses. Our data poisoning-based defense provides a new retraining-free defense paradigm that preserves the victim classifier's utility.

摘要: 模型反转攻击通过从输出中重建敏感数据，对机器学习模型构成了严重的隐私威胁。虽然已经提出了各种防御措施来对抗这些攻击，但它们往往是以分类器的效用为代价的，因此在隐私保护和模型效用之间创建了一个具有挑战性的权衡。此外，大多数现有的防御措施需要重新训练分类器以增强稳健性，这对于大规模、成熟的模型来说是不切实际的。本文介绍了一种新的防御机制，以更好地平衡隐私和效用，特别是针对使用机器学习模型(即倒置模型)重建私人数据的攻击者。从影响机器学习模型性能的数据中毒攻击中得到启发，提出了一种利用数据中毒来污染倒置模型训练数据的策略，从而防止模型倒置攻击。提出了两种防御方法。第一种称为对所有输出向量的标签保持毒化攻击(LPA)，它涉及对所有输出向量的微妙扰动，同时保持它们的标签。我们的发现表明，这些通过数据中毒方法引入的微小扰动显著增加了数据重建的难度，而不会影响分类器的实用性。随后，我们介绍了第二种方法，部分输出向量的标签翻转中毒(LFP)，它选择性地扰动一小部分输出向量，并在过程中改变它们的标签。经验结果表明，LPA非常有效，表现优于目前最先进的防御措施。我们基于数据中毒的防御提供了一种新的无需再培训的防御范例，保留了受害者分类器的实用性。



## **32. Adaptive Epsilon Adversarial Training for Robust Gravitational Wave Parameter Estimation Using Normalizing Flows**

使用正规化流进行鲁棒引力波参数估计的自适应Episodes对抗训练 cs.LG

7 pages, 9 figures

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07559v1) [paper-pdf](http://arxiv.org/pdf/2412.07559v1)

**Authors**: Yiqian Yang, Xihua Zhu, Fan Zhang

**Abstract**: Adversarial training with Normalizing Flow (NF) models is an emerging research area aimed at improving model robustness through adversarial samples. In this study, we focus on applying adversarial training to NF models for gravitational wave parameter estimation. We propose an adaptive epsilon method for Fast Gradient Sign Method (FGSM) adversarial training, which dynamically adjusts perturbation strengths based on gradient magnitudes using logarithmic scaling. Our hybrid architecture, combining ResNet and Inverse Autoregressive Flow, reduces the Negative Log Likelihood (NLL) loss by 47\% under FGSM attacks compared to the baseline model, while maintaining an NLL of 4.2 on clean data (only 5\% higher than the baseline). For perturbation strengths between 0.01 and 0.1, our model achieves an average NLL of 5.8, outperforming both fixed-epsilon (NLL: 6.7) and progressive-epsilon (NLL: 7.2) methods. Under stronger Projected Gradient Descent attacks with perturbation strength of 0.05, our model maintains an NLL of 6.4, demonstrating superior robustness while avoiding catastrophic overfitting.

摘要: 利用归一化流量模型进行对抗性训练是一个新兴的研究领域，其目的是通过对抗性样本提高模型的稳健性。在这项研究中，我们将对抗性训练应用到引力波参数估计的神经网络模型中。提出了一种用于快速梯度符号法(FGSM)对抗训练的自适应epsilon方法，该方法利用对数尺度根据梯度大小动态调整扰动强度。我们的混合架构结合了ResNet和反向自回归流，与基线模型相比，在FGSM攻击下，负对数似然(NLL)损失降低了47%，而对于干净的数据，NLL保持在4.2(仅比基线高5%)。当微扰强度在0.01到0.1之间时，我们的模型的平均NLL为5.8，优于固定epsilon(NLL：6.7)和渐进epsilon(NLL：7.2)方法。在扰动强度为0.05的较强投影梯度下降攻击下，我们的模型保持了6.4的NLL，在避免灾难性过拟合的同时表现出了优越的稳健性。



## **33. Quantifying the Prediction Uncertainty of Machine Learning Models for Individual Data**

量化个体数据机器学习模型的预测不确定性 cs.LG

PHD thesis

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07520v1) [paper-pdf](http://arxiv.org/pdf/2412.07520v1)

**Authors**: Koby Bibas

**Abstract**: Machine learning models have exhibited exceptional results in various domains. The most prevalent approach for learning is the empirical risk minimizer (ERM), which adapts the model's weights to reduce the loss on a training set and subsequently leverages these weights to predict the label for new test data. Nonetheless, ERM makes the assumption that the test distribution is similar to the training distribution, which may not always hold in real-world situations. In contrast, the predictive normalized maximum likelihood (pNML) was proposed as a min-max solution for the individual setting where no assumptions are made on the distribution of the tested input. This study investigates pNML's learnability for linear regression and neural networks, and demonstrates that pNML can improve the performance and robustness of these models on various tasks. Moreover, the pNML provides an accurate confidence measure for its output, showcasing state-of-the-art results for out-of-distribution detection, resistance to adversarial attacks, and active learning.

摘要: 机器学习模型在各个领域都显示出了出众的结果。最流行的学习方法是经验风险最小化(ERM)，它通过调整模型的权重来减少训练集上的损失，然后利用这些权重来预测新测试数据的标签。尽管如此，ERM假设测试分布类似于训练分布，这在现实世界中可能并不总是成立的。相反，预测归一化最大似然(PNML)被提出作为对测试输入的分布不作任何假设的个人设置的最小-最大解。本研究考察了PNML对线性回归和神经网络的学习能力，并证明了PNML可以提高这些模型在各种任务上的性能和稳健性。此外，PNML为其输出提供了准确的置信度度量，展示了在分布外检测、抵抗对手攻击和主动学习方面的最先进结果。



## **34. AHSG: Adversarial Attacks on High-level Semantics in Graph Neural Networks**

AHSG：对图神经网络中高级语义的对抗性攻击 cs.LG

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07468v1) [paper-pdf](http://arxiv.org/pdf/2412.07468v1)

**Authors**: Kai Yuan, Xiaobing Pei, Haoran Yang

**Abstract**: Graph Neural Networks (GNNs) have garnered significant interest among researchers due to their impressive performance in graph learning tasks. However, like other deep neural networks, GNNs are also vulnerable to adversarial attacks. In existing adversarial attack methods for GNNs, the metric between the attacked graph and the original graph is usually the attack budget or a measure of global graph properties. However, we have found that it is possible to generate attack graphs that disrupt the primary semantics even within these constraints. To address this problem, we propose a Adversarial Attacks on High-level Semantics in Graph Neural Networks (AHSG), which is a graph structure attack model that ensures the retention of primary semantics. The latent representations of each node can extract rich semantic information by applying convolutional operations on graph data. These representations contain both task-relevant primary semantic information and task-irrelevant secondary semantic information. The latent representations of same-class nodes with the same primary semantics can fulfill the objective of modifying secondary semantics while preserving the primary semantics. Finally, the latent representations with attack effects is mapped to an attack graph using Projected Gradient Descent (PGD) algorithm. By attacking graph deep learning models with some advanced defense strategies, we validate that AHSG has superior attack effectiveness compared to other attack methods. Additionally, we employ Contextual Stochastic Block Models (CSBMs) as a proxy for the primary semantics to detect the attacked graph, confirming that AHSG almost does not disrupt the original primary semantics of the graph.

摘要: 图形神经网络(GNN)因其在图形学习任务中的出色表现而引起了研究者的极大兴趣。然而，像其他深度神经网络一样，GNN也容易受到对手的攻击。在现有的GNN对抗攻击方法中，攻击图和原始图之间的度量通常是攻击预算或全局图性质的度量。然而，我们发现，即使在这些约束下，也可能生成破坏主要语义的攻击图。针对这一问题，我们提出了一种基于图神经网络高级语义的对抗性攻击(AHSG)，它是一种图结构攻击模型，保证了基本语义的保留。通过对图数据进行卷积运算，每个节点的潜在表示可以提取丰富的语义信息。这些表征既包含与任务相关的初级语义信息，也包含与任务无关的次要语义信息。具有相同初级语义的同类节点的潜在表示可以在保持初级语义的同时达到修改次级语义的目的。最后，使用投影梯度下降(PGD)算法将具有攻击效果的潜在表示映射到攻击图。通过攻击图的深度学习模型和一些先进的防御策略，验证了AHSG与其他攻击方法相比具有更好的攻击效果。此外，我们使用上下文随机块模型(CSBM)作为主要语义的代理来检测被攻击的图，证实了AHSG几乎没有破坏图的原始主要语义。



## **35. Addressing Key Challenges of Adversarial Attacks and Defenses in the Tabular Domain: A Methodological Framework for Coherence and Consistency**

应对表格领域对抗性攻击和防御的关键挑战：一致性和一致性的方法论框架 cs.LG

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07326v1) [paper-pdf](http://arxiv.org/pdf/2412.07326v1)

**Authors**: Yael Itzhakev, Amit Giloni, Yuval Elovici, Asaf Shabtai

**Abstract**: Machine learning models trained on tabular data are vulnerable to adversarial attacks, even in realistic scenarios where attackers have access only to the model's outputs. Researchers evaluate such attacks by considering metrics like success rate, perturbation magnitude, and query count. However, unlike other data domains, the tabular domain contains complex interdependencies among features, presenting a unique aspect that should be evaluated: the need for the attack to generate coherent samples and ensure feature consistency for indistinguishability. Currently, there is no established methodology for evaluating adversarial samples based on these criteria. In this paper, we address this gap by proposing new evaluation criteria tailored for tabular attacks' quality; we defined anomaly-based framework to assess the distinguishability of adversarial samples and utilize the SHAP explainability technique to identify inconsistencies in the model's decision-making process caused by adversarial samples. These criteria could form the basis for potential detection methods and be integrated into established evaluation metrics for assessing attack's quality Additionally, we introduce a novel technique for perturbing dependent features while maintaining coherence and feature consistency within the sample. We compare different attacks' strategies, examining black-box query-based attacks and transferability-based gradient attacks across four target models. Our experiments, conducted on benchmark tabular datasets, reveal significant differences between the examined attacks' strategies in terms of the attacker's risk and effort and the attacks' quality. The findings provide valuable insights on the strengths, limitations, and trade-offs of various adversarial attacks in the tabular domain, laying a foundation for future research on attacks and defense development.

摘要: 根据表格数据训练的机器学习模型容易受到敌意攻击，即使在攻击者只能访问模型输出的现实情况下也是如此。研究人员通过考虑成功率、扰乱程度和查询计数等指标来评估此类攻击。然而，与其他数据域不同的是，表格域包含各种特征之间的复杂相互依赖关系，这提出了一个应加以评估的独特方面：需要攻击生成一致的样本，并确保特征的一致性，从而实现不可区分。目前，还没有根据这些标准评估敌方样本的既定方法。在本文中，我们通过提出新的针对列表攻击质量的评估标准来解决这一问题；我们定义了基于异常的框架来评估对抗性样本的可区分性，并利用Shap可解释性技术来识别由对抗性样本导致的模型决策过程中的不一致。这些标准可以作为潜在检测方法的基础，并被集成到已建立的评估攻击质量的评估指标中。此外，我们引入了一种新的技术来扰动依赖特征，同时保持样本内的一致性和特征一致性。我们比较了不同攻击的策略，考察了基于黑盒查询的攻击和基于可转移性的梯度攻击在四个目标模型上的表现。我们在基准表格数据集上进行的实验表明，被检查的攻击策略在攻击者的风险和努力以及攻击质量方面存在显著差异。这些发现为表格领域中各种对抗性攻击的优势、局限性和权衡提供了有价值的见解，为未来攻击和防御发展的研究奠定了基础。



## **36. Backdoor Attacks against No-Reference Image Quality Assessment Models via A Scalable Trigger**

通过可扩展触发器对无参考图像质量评估模型进行后门攻击 cs.CV

Accept by AAAI 2025

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07277v1) [paper-pdf](http://arxiv.org/pdf/2412.07277v1)

**Authors**: Yi Yu, Song Xia, Xun Lin, Wenhan Yang, Shijian Lu, Yap-peng Tan, Alex Kot

**Abstract**: No-Reference Image Quality Assessment (NR-IQA), responsible for assessing the quality of a single input image without using any reference, plays a critical role in evaluating and optimizing computer vision systems, e.g., low-light enhancement. Recent research indicates that NR-IQA models are susceptible to adversarial attacks, which can significantly alter predicted scores with visually imperceptible perturbations. Despite revealing vulnerabilities, these attack methods have limitations, including high computational demands, untargeted manipulation, limited practical utility in white-box scenarios, and reduced effectiveness in black-box scenarios. To address these challenges, we shift our focus to another significant threat and present a novel poisoning-based backdoor attack against NR-IQA (BAIQA), allowing the attacker to manipulate the IQA model's output to any desired target value by simply adjusting a scaling coefficient $\alpha$ for the trigger. We propose to inject the trigger in the discrete cosine transform (DCT) domain to improve the local invariance of the trigger for countering trigger diminishment in NR-IQA models due to widely adopted data augmentations. Furthermore, the universal adversarial perturbations (UAP) in the DCT space are designed as the trigger, to increase IQA model susceptibility to manipulation and improve attack effectiveness. In addition to the heuristic method for poison-label BAIQA (P-BAIQA), we explore the design of clean-label BAIQA (C-BAIQA), focusing on $\alpha$ sampling and image data refinement, driven by theoretical insights we reveal. Extensive experiments on diverse datasets and various NR-IQA models demonstrate the effectiveness of our attacks. Code will be released at https://github.com/yuyi-sd/BAIQA.

摘要: 无参考图像质量评估(NR-IQA)负责在不使用任何参考图像的情况下评估单个输入图像的质量，在评估和优化计算机视觉系统(如微光增强)中起着至关重要的作用。最近的研究表明，NR-IQA模型容易受到对抗性攻击，这种攻击会在视觉上不可察觉的扰动下显著改变预测分数。尽管暴露出漏洞，但这些攻击方法都有局限性，包括计算要求高、无针对性操作、在白盒场景中实际效用有限，以及在黑盒场景中有效性降低。为了应对这些挑战，我们将重点转移到另一个重要的威胁上，并提出了一种针对NR-IQA的基于中毒的后门攻击(BAIQA)，允许攻击者通过简单地调整触发器的缩放系数$\α$来操纵IQA模型的输出到任何期望的目标值。我们提出在离散余弦变换(DCT)域中注入触发器以改善触发器的局部不变性，以对抗由于广泛采用的数据增强而导致的NR-IQA模型中的触发器衰减。此外，设计了DCT空间中的通用对抗摄动(UAP)作为触发器，以增加IQA模型对操纵的敏感度，提高攻击效率。除了有毒标签BAIQA的启发式方法(P-BAIQA)外，我们还探索了清洁标签BAIQA(C-BAIQA)的设计，重点是$\α$采样和图像数据精化，这是我们揭示的理论见解的驱动。在不同的数据集和不同的NR-IQA模型上的大量实验证明了我们的攻击的有效性。代码将在https://github.com/yuyi-sd/BAIQA.上发布



## **37. A Generative Victim Model for Segmentation**

用于分割的生成受害者模型 cs.CV

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07274v1) [paper-pdf](http://arxiv.org/pdf/2412.07274v1)

**Authors**: Aixuan Li, Jing Zhang, Jiawei Shi, Yiran Zhong, Yuchao Dai

**Abstract**: We find that the well-trained victim models (VMs), against which the attacks are generated, serve as fundamental prerequisites for adversarial attacks, i.e. a segmentation VM is needed to generate attacks for segmentation. In this context, the victim model is assumed to be robust to achieve effective adversarial perturbation generation. Instead of focusing on improving the robustness of the task-specific victim models, we shift our attention to image generation. From an image generation perspective, we derive a novel VM for segmentation, aiming to generate adversarial perturbations for segmentation tasks without requiring models explicitly designed for image segmentation. Our approach to adversarial attack generation diverges from conventional white-box or black-box attacks, offering a fresh outlook on adversarial attack strategies. Experiments show that our attack method is able to generate effective adversarial attacks with good transferability.

摘要: 我们发现，攻击所针对的训练有素的受害者模型（VMs）是对抗性攻击的基本先决条件，即需要分段虚拟机来生成分段攻击。在这种情况下，假设受害者模型是稳健的，能够实现有效的对抗扰动生成。我们不再专注于提高特定任务受害者模型的稳健性，而是将注意力转移到图像生成上。从图像生成的角度来看，我们推导出一种新型的用于分割的虚拟机，旨在为分割任务生成对抗性扰动，而不需要为图像分割明确设计的模型。我们的对抗性攻击生成方法与传统的白盒或黑匣子攻击不同，为对抗性攻击策略提供了全新的视角。实验表明，我们的攻击方法能够产生有效的对抗攻击，具有良好的可移植性。



## **38. CapGen:An Environment-Adaptive Generator of Adversarial Patches**

CapGen：环境适应性对抗补丁生成器 cs.CV

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07253v1) [paper-pdf](http://arxiv.org/pdf/2412.07253v1)

**Authors**: Chaoqun Li, Zhuodong Liu, Huanqian Yan, Hang Su

**Abstract**: Adversarial patches, often used to provide physical stealth protection for critical assets and assess perception algorithm robustness, usually neglect the need for visual harmony with the background environment, making them easily noticeable. Moreover, existing methods primarily concentrate on improving attack performance, disregarding the intricate dynamics of adversarial patch elements. In this work, we introduce the Camouflaged Adversarial Pattern Generator (CAPGen), a novel approach that leverages specific base colors from the surrounding environment to produce patches that seamlessly blend with their background for superior visual stealthiness while maintaining robust adversarial performance. We delve into the influence of both patterns (i.e., color-agnostic texture information) and colors on the effectiveness of attacks facilitated by patches, discovering that patterns exert a more pronounced effect on performance than colors. Based on these findings, we propose a rapid generation strategy for adversarial patches. This involves updating the colors of high-performance adversarial patches to align with those of the new environment, ensuring visual stealthiness without compromising adversarial impact. This paper is the first to comprehensively examine the roles played by patterns and colors in the context of adversarial patches.

摘要: 对抗性补丁通常用于为关键资产提供物理隐身保护，并评估感知算法的健壮性，通常忽略了与背景环境视觉和谐的需要，使它们很容易被注意到。此外，现有的方法主要集中在提高攻击性能，而忽略了对抗性补丁元素的复杂动态。在这项工作中，我们介绍了伪装对抗模式生成器(CAPGen)，这是一种新的方法，利用周围环境中特定的基色来产生与背景无缝混合的补丁，以实现卓越的视觉隐蔽性，同时保持稳健的对抗性能。我们深入研究了图案(即与颜色无关的纹理信息)和颜色对补丁攻击有效性的影响，发现图案比颜色对性能的影响更显著。基于这些发现，我们提出了一种对抗性补丁的快速生成策略。这包括更新高性能对抗性补丁的颜色以与新环境的颜色保持一致，确保视觉隐蔽性而不影响对抗性影响。本文首次全面探讨了图案和色彩在对抗性斑块中所起的作用。



## **39. Adversarial Filtering Based Evasion and Backdoor Attacks to EEG-Based Brain-Computer Interfaces**

基于对抗过滤的规避和后门攻击基于脑电的脑机接口 cs.HC

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07231v1) [paper-pdf](http://arxiv.org/pdf/2412.07231v1)

**Authors**: Lubin Meng, Xue Jiang, Xiaoqing Chen, Wenzhong Liu, Hanbin Luo, Dongrui Wu

**Abstract**: A brain-computer interface (BCI) enables direct communication between the brain and an external device. Electroencephalogram (EEG) is a common input signal for BCIs, due to its convenience and low cost. Most research on EEG-based BCIs focuses on the accurate decoding of EEG signals, while ignoring their security. Recent studies have shown that machine learning models in BCIs are vulnerable to adversarial attacks. This paper proposes adversarial filtering based evasion and backdoor attacks to EEG-based BCIs, which are very easy to implement. Experiments on three datasets from different BCI paradigms demonstrated the effectiveness of our proposed attack approaches. To our knowledge, this is the first study on adversarial filtering for EEG-based BCIs, raising a new security concern and calling for more attention on the security of BCIs.

摘要: 脑机接口（BCI）实现大脑和外部设备之间的直接通信。由于其方便性和低成本，脑电波（EEG）是BCI的常见输入信号。大多数关于基于脑电的BCI的研究都集中在脑电信号的准确解码上，而忽视了其安全性。最近的研究表明，BCI中的机器学习模型容易受到对抗攻击。本文提出了对基于脑电的BCI的基于对抗过滤的规避和后门攻击，这些攻击非常容易实现。对来自不同BCI范式的三个数据集的实验证明了我们提出的攻击方法的有效性。据我们所知，这是第一项关于基于脑电的BCI对抗过滤的研究，提出了新的安全问题，并呼吁人们更加关注BCI的安全性。



## **40. A Parametric Approach to Adversarial Augmentation for Cross-Domain Iris Presentation Attack Detection**

跨域虹膜呈现攻击检测的对抗增强参数方法 cs.CV

IEEE/CVF Winter Conference on Applications of Computer Vision (WACV),  2025

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07199v1) [paper-pdf](http://arxiv.org/pdf/2412.07199v1)

**Authors**: Debasmita Pal, Redwan Sony, Arun Ross

**Abstract**: Iris-based biometric systems are vulnerable to presentation attacks (PAs), where adversaries present physical artifacts (e.g., printed iris images, textured contact lenses) to defeat the system. This has led to the development of various presentation attack detection (PAD) algorithms, which typically perform well in intra-domain settings. However, they often struggle to generalize effectively in cross-domain scenarios, where training and testing employ different sensors, PA instruments, and datasets. In this work, we use adversarial training samples of both bonafide irides and PAs to improve the cross-domain performance of a PAD classifier. The novelty of our approach lies in leveraging transformation parameters from classical data augmentation schemes (e.g., translation, rotation) to generate adversarial samples. We achieve this through a convolutional autoencoder, ADV-GEN, that inputs original training samples along with a set of geometric and photometric transformations. The transformation parameters act as regularization variables, guiding ADV-GEN to generate adversarial samples in a constrained search space. Experiments conducted on the LivDet-Iris 2017 database, comprising four datasets, and the LivDet-Iris 2020 dataset, demonstrate the efficacy of our proposed method. The code is available at https://github.com/iPRoBe-lab/ADV-GEN-IrisPAD.

摘要: 基于虹膜的生物识别系统容易受到呈现攻击(PAS)，即攻击者呈现物理伪像(例如，打印的虹膜图像、纹理隐形眼镜)来击败系统。这导致了各种呈现攻击检测(PAD)算法的发展，这些算法通常在域内设置中执行得很好。然而，在训练和测试使用不同的传感器、PA仪器和数据集的跨域场景中，它们往往难以有效地推广。在这项工作中，我们使用真实虹膜和PAS的对抗性训练样本来提高PAD分类器的跨域性能。我们方法的创新之处在于利用经典数据增强方案(如平移、旋转)中的变换参数来生成对抗性样本。我们通过卷积自动编码器ADV-Gen实现这一点，该编码器输入原始训练样本以及一组几何和光度变换。变换参数作为正则化变量，指导ADV-Gen在受限的搜索空间中生成对抗性样本。在包含四个数据集的LivDet-Iris 2017数据库和LivDet-Iris 2020数据集上进行的实验证明了我们所提出的方法的有效性。代码可在https://github.com/iPRoBe-lab/ADV-GEN-IrisPAD.上获得



## **41. PrisonBreak: Jailbreaking Large Language Models with Fewer Than Twenty-Five Targeted Bit-flips**

Prison Break：越狱大型语言模型，目标位翻转少于25个 cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07192v1) [paper-pdf](http://arxiv.org/pdf/2412.07192v1)

**Authors**: Zachary Coalson, Jeonghyun Woo, Shiyang Chen, Yu Sun, Lishan Yang, Prashant Nair, Bo Fang, Sanghyun Hong

**Abstract**: We introduce a new class of attacks on commercial-scale (human-aligned) language models that induce jailbreaking through targeted bitwise corruptions in model parameters. Our adversary can jailbreak billion-parameter language models with fewer than 25 bit-flips in all cases$-$and as few as 5 in some$-$using up to 40$\times$ less bit-flips than existing attacks on computer vision models at least 100$\times$ smaller. Unlike prompt-based jailbreaks, our attack renders these models in memory 'uncensored' at runtime, allowing them to generate harmful responses without any input modifications. Our attack algorithm efficiently identifies target bits to flip, offering up to 20$\times$ more computational efficiency than previous methods. This makes it practical for language models with billions of parameters. We show an end-to-end exploitation of our attack using software-induced fault injection, Rowhammer (RH). Our work examines 56 DRAM RH profiles from DDR4 and LPDDR4X devices with different RH vulnerabilities. We show that our attack can reliably induce jailbreaking in systems similar to those affected by prior bit-flip attacks. Moreover, our approach remains effective even against highly RH-secure systems (e.g., 46$\times$ more secure than previously tested systems). Our analyses further reveal that: (1) models with less post-training alignment require fewer bit flips to jailbreak; (2) certain model components, such as value projection layers, are substantially more vulnerable than others; and (3) our method is mechanistically different than existing jailbreaks. Our findings highlight a pressing, practical threat to the language model ecosystem and underscore the need for research to protect these models from bit-flip attacks.

摘要: 我们在商业规模(人类对齐的)语言模型上引入了一类新的攻击，这些攻击通过模型参数中有针对性的逐位破坏来诱导越狱。我们的对手可以用不到25个比特翻转的语言模型越狱，所有情况下都不到25个比特翻转，在一些$-$中只有5个，使用多达40个比特翻转，比对计算机视觉模型的现有攻击少至少100$\×$。与基于提示的越狱不同，我们的攻击在运行时将这些模型呈现在内存中，不受审查，允许它们在不修改任何输入的情况下生成有害的响应。我们的攻击算法有效地识别要翻转的目标比特，比以前的方法提供了高达20美元\倍的计算效率。这使得它适用于具有数十亿个参数的语言模型。我们使用软件诱导的故障注入Rowhammer(RH)展示了对我们的攻击的端到端攻击。我们的工作检查了来自具有不同RH漏洞的DDR4和LPDDR4X设备的56个DRAM RH配置文件。我们证明了我们的攻击可以可靠地在类似于先前受比特翻转攻击影响的系统中诱导越狱。此外，我们的方法即使对高度RH安全的系统也是有效的(例如，比之前测试的系统安全46美元\倍)。我们的分析进一步表明：(1)训练后对齐较少的模型需要较少的比特翻转越狱；(2)某些模型组件，如值投影层，比其他组件更容易受到攻击；(3)我们的方法与现有的越狱方法在机械上不同。我们的发现突显了语言模型生态系统面临的紧迫、实际的威胁，并强调了研究保护这些模型免受比特翻转攻击的必要性。



## **42. dSTAR: Straggler Tolerant and Byzantine Resilient Distributed SGD**

dSTAR：容忍落后和拜占庭弹性分布式新元 cs.DC

15 pages

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07151v1) [paper-pdf](http://arxiv.org/pdf/2412.07151v1)

**Authors**: Jiahe Yan, Pratik Chaudhari, Leonard Kleinrock

**Abstract**: Distributed model training needs to be adapted to challenges such as the straggler effect and Byzantine attacks. When coordinating the training process with multiple computing nodes, ensuring timely and reliable gradient aggregation amidst network and system malfunctions is essential. To tackle these issues, we propose \textit{dSTAR}, a lightweight and efficient approach for distributed stochastic gradient descent (SGD) that enhances robustness and convergence. \textit{dSTAR} selectively aggregates gradients by collecting updates from the first \(k\) workers to respond, filtering them based on deviations calculated using an ensemble median. This method not only mitigates the impact of stragglers but also fortifies the model against Byzantine adversaries. We theoretically establish that \textit{dSTAR} is (\(\alpha, f\))-Byzantine resilient and achieves a linear convergence rate. Empirical evaluations across various scenarios demonstrate that \textit{dSTAR} consistently maintains high accuracy, outperforming other Byzantine-resilient methods that often suffer up to a 40-50\% accuracy drop under attack. Our results highlight \textit{dSTAR} as a robust solution for training models in distributed environments prone to both straggler delays and Byzantine faults.

摘要: 分布式模型训练需要适应诸如掉队效应和拜占庭攻击等挑战。在与多个计算节点协调训练过程时，确保在网络和系统故障中及时可靠地进行梯度聚合是至关重要的。为了解决这些问题，我们提出了一种轻量级、高效的分布式随机梯度下降(SGD)方法，该方法增强了鲁棒性和收敛能力。通过收集来自第一(K)个工作人员的更新进行响应，根据使用集合中值计算的偏差对其进行过滤，从而有选择地聚合梯度。这种方法不仅减轻了掉队的影响，而且增强了该模型对抗拜占庭式对手的能力。我们从理论上证明了Texttit{dSTAR}是((α，f))-拜占庭弹性的，并且达到了线性收敛速度。对不同场景的经验评估表明，该方法始终保持较高的准确率，优于其他拜占庭弹性方法，后者在攻击下的准确率通常会下降40%-50%。我们的结果突出表明，在容易出现分散延迟和拜占庭故障的分布式环境中，文本{dSTAR}是一种稳健的模型训练解决方案。



## **43. Defensive Dual Masking for Robust Adversarial Defense**

防御性双重掩蔽实现强大的对抗性防御 cs.CL

First version

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07078v1) [paper-pdf](http://arxiv.org/pdf/2412.07078v1)

**Authors**: Wangli Yang, Jie Yang, Yi Guo, Johan Barthelemy

**Abstract**: The field of textual adversarial defenses has gained considerable attention in recent years due to the increasing vulnerability of natural language processing (NLP) models to adversarial attacks, which exploit subtle perturbations in input text to deceive models. This paper introduces the Defensive Dual Masking (DDM) algorithm, a novel approach designed to enhance model robustness against such attacks. DDM utilizes a unique adversarial training strategy where [MASK] tokens are strategically inserted into training samples to prepare the model to handle adversarial perturbations more effectively. During inference, potentially adversarial tokens are dynamically replaced with [MASK] tokens to neutralize potential threats while preserving the core semantics of the input. The theoretical foundation of our approach is explored, demonstrating how the selective masking mechanism strengthens the model's ability to identify and mitigate adversarial manipulations. Our empirical evaluation across a diverse set of benchmark datasets and attack mechanisms consistently shows that DDM outperforms state-of-the-art defense techniques, improving model accuracy and robustness. Moreover, when applied to Large Language Models (LLMs), DDM also enhances their resilience to adversarial attacks, providing a scalable defense mechanism for large-scale NLP applications.

摘要: 近年来，由于自然语言处理(NLP)模型越来越容易受到敌意攻击，利用输入文本中的细微扰动来欺骗模型，文本对抗防御领域受到了相当大的关注。介绍了防御性双重掩蔽(DDM)算法，这是一种新的方法，旨在增强模型对此类攻击的稳健性。DDM利用一种独特的对抗性训练策略，其中[MASK]标记被战略性地插入到训练样本中，以准备模型以更有效地处理对抗性扰动。在推理过程中，潜在的敌意令牌被动态地替换为[掩码]令牌，以中和潜在的威胁，同时保留输入的核心语义。探讨了我们方法的理论基础，展示了选择性掩蔽机制如何增强模型识别和缓解对手操纵的能力。我们对不同的基准数据集和攻击机制进行的经验评估一致表明，DDM的性能优于最先进的防御技术，提高了模型的准确性和稳健性。此外，当应用于大型语言模型时，DDM还增强了它们对对手攻击的韧性，为大规模NLP应用提供了一种可扩展的防御机制。



## **44. Dense Cross-Connected Ensemble Convolutional Neural Networks for Enhanced Model Robustness**

密集交叉连接卷积神经网络增强模型鲁棒性 cs.CV

6 pages, 1 figure

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.07022v1) [paper-pdf](http://arxiv.org/pdf/2412.07022v1)

**Authors**: Longwei Wang, Xueqian Li, Zheng Zhang

**Abstract**: The resilience of convolutional neural networks against input variations and adversarial attacks remains a significant challenge in image recognition tasks. Motivated by the need for more robust and reliable image recognition systems, we propose the Dense Cross-Connected Ensemble Convolutional Neural Network (DCC-ECNN). This novel architecture integrates the dense connectivity principle of DenseNet with the ensemble learning strategy, incorporating intermediate cross-connections between different DenseNet paths to facilitate extensive feature sharing and integration. The DCC-ECNN architecture leverages DenseNet's efficient parameter usage and depth while benefiting from the robustness of ensemble learning, ensuring a richer and more resilient feature representation.

摘要: 卷积神经网络对输入变化和对抗攻击的弹性仍然是图像识别任务中的一个重大挑战。出于对更稳健、更可靠的图像识别系统的需求，我们提出了密集交叉连接卷积神经网络（DCC-ECNN）。这种新颖的架构将DenseNet的密集连接原则与集成学习策略集成在一起，合并了不同DenseNet路径之间的中间交叉连接，以促进广泛的特征共享和集成。DCC-ECNN架构利用DenseNet的高效参数使用和深度，同时受益于集成学习的稳健性，确保更丰富、更有弹性的特征表示。



## **45. Fiat-Shamir for Proofs Lacks a Proof Even in the Presence of Shared Entanglement**

即使存在共同纠缠，菲亚特-沙米尔的证据也缺乏证据 quant-ph

58 pages, 4 figures; accepted in Quantum

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2204.02265v5) [paper-pdf](http://arxiv.org/pdf/2204.02265v5)

**Authors**: Frédéric Dupuis, Philippe Lamontagne, Louis Salvail

**Abstract**: We explore the cryptographic power of arbitrary shared physical resources. The most general such resource is access to a fresh entangled quantum state at the outset of each protocol execution. We call this the Common Reference Quantum State (CRQS) model, in analogy to the well-known Common Reference String (CRS). The CRQS model is a natural generalization of the CRS model but appears to be more powerful: in the two-party setting, a CRQS can sometimes exhibit properties associated with a Random Oracle queried once by measuring a maximally entangled state in one of many mutually unbiased bases. We formalize this notion as a Weak One-Time Random Oracle (WOTRO), where we only ask of the $m$-bit output to have some randomness when conditioned on the $n$-bit input.   We show that when $n-m\in\omega(\lg n)$, any protocol for WOTRO in the CRQS model can be attacked by an (inefficient) adversary. Moreover, our adversary is efficiently simulatable, which rules out the possibility of proving the computational security of a scheme by a fully black-box reduction to a cryptographic game assumption. On the other hand, we introduce a non-game quantum assumption for hash functions that implies WOTRO in the CRQS model (where the CRQS consists only of EPR pairs). We first build a statistically secure WOTRO protocol where $m=n$, then hash the output.   The impossibility of WOTRO has the following consequences. First, we show the fully-black-box impossibility of a quantum Fiat-Shamir transform, extending the impossibility result of Bitansky et al. (TCC 2013) to the CRQS model. Second, we show a fully-black-box impossibility result for a strenghtened version of quantum lightning (Zhandry, Eurocrypt 2019) where quantum bolts have an additional parameter that cannot be changed without generating new bolts. Our results also apply to $2$-message protocols in the plain model.

摘要: 我们探索任意共享物理资源的加密能力。最常见的这类资源是在每个协议执行开始时访问新的纠缠量子态。我们称之为公共参考量子态(CRQS)模型，类似于众所周知的公共参考弦(CRS)。CRQS模型是CRS模型的自然推广，但似乎更强大：在两方设置中，CRQS有时可以通过测量许多相互无偏的碱基之一中的最大纠缠态来展示与查询一次的随机Oracle相关联的属性。我们将这个概念形式化为弱一次性随机Oracle(WOTRO)，其中我们只要求$m$位的输出在以$n$位输入为条件时具有一定的随机性。我们证明了当$n-m\in\omega(\lg n)$时，CRQS模型中用于WOTRO的任何协议都可以被(低效的)攻击者攻击。此外，我们的对手是高效可模拟的，这排除了通过完全黑箱简化为密码博弈假设来证明方案的计算安全性的可能性。另一方面，我们对散列函数引入了一个非博弈量子假设，该假设意味着CRQS模型(其中CRQS只由EPR对组成)中的WOTRO。我们首先构建一个统计安全的WOTRO协议，其中$m=n$，然后对输出进行散列。WOTRO的不可能性会产生以下后果。首先，我们证明了量子Fiat-Shamir变换的全黑箱不可能性，推广了Bitansky等人的不可能结果。(TCC 2013)到CRQS模式。其次，我们证明了量子闪电的增强版本(Zhandry，Eurocrypt 2019)的一个完全黑箱不可能的结果，其中量子闪电有一个额外的参数，如果不产生新的闪电，这个参数就不能改变。我们的结果也适用于普通模型中的$2$-消息协议。



## **46. WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs**

WildGuard：针对LLC安全风险、越狱和拒绝的开放式一站式审核工具 cs.CL

NeurIPS 2024 Camera Ready. First two authors contributed equally.  Third and fourth authors contributed equally

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2406.18495v3) [paper-pdf](http://arxiv.org/pdf/2406.18495v3)

**Authors**: Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, Nouha Dziri

**Abstract**: We introduce WildGuard -- an open, light-weight moderation tool for LLM safety that achieves three goals: (1) identifying malicious intent in user prompts, (2) detecting safety risks of model responses, and (3) determining model refusal rate. Together, WildGuard serves the increasing needs for automatic safety moderation and evaluation of LLM interactions, providing a one-stop tool with enhanced accuracy and broad coverage across 13 risk categories. While existing open moderation tools such as Llama-Guard2 score reasonably well in classifying straightforward model interactions, they lag far behind a prompted GPT-4, especially in identifying adversarial jailbreaks and in evaluating models' refusals, a key measure for evaluating safety behaviors in model responses.   To address these challenges, we construct WildGuardMix, a large-scale and carefully balanced multi-task safety moderation dataset with 92K labeled examples that cover vanilla (direct) prompts and adversarial jailbreaks, paired with various refusal and compliance responses. WildGuardMix is a combination of WildGuardTrain, the training data of WildGuard, and WildGuardTest, a high-quality human-annotated moderation test set with 5K labeled items covering broad risk scenarios. Through extensive evaluations on WildGuardTest and ten existing public benchmarks, we show that WildGuard establishes state-of-the-art performance in open-source safety moderation across all the three tasks compared to ten strong existing open-source moderation models (e.g., up to 26.4% improvement on refusal detection). Importantly, WildGuard matches and sometimes exceeds GPT-4 performance (e.g., up to 3.9% improvement on prompt harmfulness identification). WildGuard serves as a highly effective safety moderator in an LLM interface, reducing the success rate of jailbreak attacks from 79.8% to 2.4%.

摘要: 我们介绍了WildGuard--一个开放的、轻量级的LLM安全防御工具，它实现了三个目标：(1)识别用户提示中的恶意意图，(2)检测模型响应的安全风险，(3)确定模型拒绝率。综合起来，WildGuard可满足日益增长的自动安全审核和评估LLM交互作用的需求，提供了一种一站式工具，具有更高的准确性和广泛的覆盖范围，涵盖13个风险类别。虽然现有的开放式审核工具，如Llama-Guard2，在对直接的模型交互进行分类方面得分相当好，但它们远远落后于GPT-4，特别是在识别对抗性越狱和评估模型拒绝方面，这是评估模型响应中安全行为的关键指标。为了应对这些挑战，我们构建了WildGuardMix，这是一个大规模的、仔细平衡的多任务安全缓和数据集，具有92K标记的示例，涵盖普通(直接)提示和对抗性越狱，并与各种拒绝和合规响应配对。WildGuardMix是WildGuard的训练数据WildGuardTrain和WildGuardTest的组合，WildGuardTest是一种高质量的人工注释适度测试集，具有覆盖广泛风险情景的5K标签项目。通过对WildGuardTest和十个现有公共基准的广泛评估，我们表明WildGuard在所有三个任务中建立了开源安全适度的最先进性能，而不是现有的十个强大的开源适度模型(例如，拒绝检测方面高达26.4%的改进)。重要的是，WildGuard的性能与GPT-4相当，有时甚至超过GPT-4(例如，在及时识别危害性方面最高提高3.9%)。WildGuard在LLM界面中充当高效的安全调节器，将越狱攻击的成功率从79.8%降低到2.4%。



## **47. Take Fake as Real: Realistic-like Robust Black-box Adversarial Attack to Evade AIGC Detection**

以假为真：类似现实的鲁棒黑匣子对抗攻击以逃避AIGC检测 cs.CV

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06727v1) [paper-pdf](http://arxiv.org/pdf/2412.06727v1)

**Authors**: Caiyun Xie, Dengpan Ye, Yunming Zhang, Long Tang, Yunna Lv, Jiacheng Deng, Jiawei Song

**Abstract**: The security of AI-generated content (AIGC) detection based on GANs and diffusion models is closely related to the credibility of multimedia content. Malicious adversarial attacks can evade these developing AIGC detection. However, most existing adversarial attacks focus only on GAN-generated facial images detection, struggle to be effective on multi-class natural images and diffusion-based detectors, and exhibit poor invisibility. To fill this gap, we first conduct an in-depth analysis of the vulnerability of AIGC detectors and discover the feature that detectors vary in vulnerability to different post-processing. Then, considering the uncertainty of detectors in real-world scenarios, and based on the discovery, we propose a Realistic-like Robust Black-box Adversarial attack (R$^2$BA) with post-processing fusion optimization. Unlike typical perturbations, R$^2$BA uses real-world post-processing, i.e., Gaussian blur, JPEG compression, Gaussian noise and light spot to generate adversarial examples. Specifically, we use a stochastic particle swarm algorithm with inertia decay to optimize post-processing fusion intensity and explore the detector's decision boundary. Guided by the detector's fake probability, R$^2$BA enhances/weakens the detector-vulnerable/detector-robust post-processing intensity to strike a balance between adversariality and invisibility. Extensive experiments on popular/commercial AIGC detectors and datasets demonstrate that R$^2$BA exhibits impressive anti-detection performance, excellent invisibility, and strong robustness in GAN-based and diffusion-based cases. Compared to state-of-the-art white-box and black-box attacks, R$^2$BA shows significant improvements of 15% and 21% in anti-detection performance under the original and robust scenario respectively, offering valuable insights for the security of AIGC detection in real-world applications.

摘要: 基于遗传算法和扩散模型的人工智能生成内容(AIGC)检测的安全性与多媒体内容的可信度密切相关。恶意对抗性攻击可以逃避这些正在开发的AIGC检测。然而，现有的对抗性攻击大多只针对GaN生成的人脸图像检测，难以对多类自然图像和基于扩散的检测器有效，并且表现出较差的不可见性。为了填补这一空白，我们首先对AIGC检测器的脆弱性进行了深入的分析，发现了检测器对不同后处理的脆弱性不同的特点。然后，考虑到检测器在实际场景中的不确定性，基于这一发现，我们提出了一种具有后处理融合优化的逼真的稳健黑盒对抗攻击(R$^2$BA)。与典型的扰动不同，R$^2$BA使用真实世界的后处理，即高斯模糊、JPEG压缩、高斯噪声和光斑来生成对抗性示例。具体地说，我们使用了一种带惯性衰减的随机粒子群算法来优化后处理融合强度，并探索了检测器的决策边界。在检测器伪概率的指导下，R$^2$BA增强/削弱了检测器易受攻击/检测器健壮的后处理强度，以在对抗性和不可见性之间取得平衡。在流行的/商用AIGC探测器和数据集上的大量实验表明，R$^2$BA在基于GaN和基于扩散的情况下具有令人印象深刻的抗检测性能、出色的不可见性和强大的稳健性。与最新的白盒和黑盒攻击相比，R$^2$BA在原始场景和健壮场景下的抗检测性能分别提高了15%和21%，为实际应用中AIGC检测的安全性提供了有价值的见解。



## **48. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

越多越好（大多数）：关于联邦图神经网络中的后门攻击 cs.CR

15 pages, 13 figures

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2202.03195v6) [paper-pdf](http://arxiv.org/pdf/2202.03195v6)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstract**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for a different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against one defense. We find that both attacks are robust against the investigated defense, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.

摘要: 图神经网络是一类基于深度学习的图域信息处理方法。由于其优越的学习复杂图形数据表示的能力，GNN最近已成为一种广泛使用的图形分析方法。然而，由于隐私问题和监管限制，集中式GNN可能很难适用于数据敏感的情况。联合学习(FL)是一种新兴的技术，是为保护隐私而开发的，当多个参与方需要协作训练共享的全球模型时。虽然一些研究工作已经将FL用于训练GNN(Federated GNN)，但还没有关于其对后门攻击的健壮性的研究。本文通过在联邦GNN中实施两种类型的后门攻击来弥合这一差距：集中式后门攻击(CBA)和分布式后门攻击(DBA)。我们的实验表明，DBA攻击的成功率几乎在所有评估案例中都高于CBA。对于CBA，即使敌方的训练集嵌入了全局触发器，所有局部触发器的攻击成功率也与全局触发器相似。为了进一步探索联邦GNN中两种后门攻击的特性，我们评估了不同客户端数量、触发器大小、中毒强度和触发器密度下的攻击性能。此外，我们还探讨了DBA和CBA对一种防御的健壮性。我们发现，这两种攻击对所调查的防御都是健壮的，因此有必要将联邦GNN中的后门攻击视为一种需要自定义防御的新威胁。



## **49. Vulnerability, Where Art Thou? An Investigation of Vulnerability Management in Android Smartphone Chipsets**

脆弱，你在哪里？Android智能手机芯片组漏洞管理调查 cs.CR

Accepted by Network and Distributed System Security (NDSS) Symposium  2025

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06556v1) [paper-pdf](http://arxiv.org/pdf/2412.06556v1)

**Authors**: Daniel Klischies, Philipp Mackensen, Veelasha Moonsamy

**Abstract**: Vulnerabilities in Android smartphone chipsets have severe consequences, as recent real-world attacks have demonstrated that adversaries can leverage vulnerabilities to execute arbitrary code or exfiltrate confidential information. Despite the far-reaching impact of such attacks, the lifecycle of chipset vulnerabilities has yet to be investigated, with existing papers primarily investigating vulnerabilities in the Android operating system. This paper provides a comprehensive and empirical study of the current state of smartphone chipset vulnerability management within the Android ecosystem. For the first time, we create a unified knowledge base of 3,676 chipset vulnerabilities affecting 437 chipset models from all four major chipset manufacturers, combined with 6,866 smartphone models. Our analysis revealed that the same vulnerabilities are often included in multiple generations of chipsets, providing novel empirical evidence that vulnerabilities are inherited through multiple chipset generations. Furthermore, we demonstrate that the commonly accepted 90-day responsible vulnerability disclosure period is seldom adhered to. We find that a single vulnerability often affects hundreds to thousands of different smartphone models, for which update availability is, as we show, often unclear or heavily delayed. Leveraging the new insights gained from our empirical analysis, we recommend several changes that chipset manufacturers can implement to improve the security posture of their products. At the same time, our knowledge base enables academic researchers to conduct more representative evaluations of smartphone chipsets, accurately assess the impact of vulnerabilities they discover, and identify avenues for future research.

摘要: Android智能手机芯片组中的漏洞具有严重后果，因为最近的现实世界攻击表明，攻击者可以利用漏洞执行任意代码或泄露机密信息。尽管此类攻击影响深远，但芯片组漏洞的生命周期尚未调查，现有论文主要调查Android操作系统中的漏洞。本文对Android生态系统中智能手机芯片组漏洞管理的现状进行了全面的实证研究。我们首次创建了一个统一的知识库，其中包含影响所有四大芯片组制造商的437个芯片组型号的3,676个芯片组漏洞，以及6,866个智能手机型号。我们的分析显示，相同的漏洞通常包含在多代芯片组中，这为漏洞通过多代芯片组遗传提供了新的经验证据。此外，我们还证明，通常接受的90天负责任的漏洞披露期限很少得到遵守。我们发现，一个单一的漏洞通常会影响数百到数千种不同的智能手机型号，正如我们所显示的那样，这些型号的更新通常不清楚或严重延迟。利用我们从经验分析中获得的新见解，我们建议芯片组制造商可以实施的几项更改，以改善其产品的安全状况。同时，我们的知识库使学术研究人员能够对智能手机芯片组进行更具代表性的评估，准确评估他们发现的漏洞的影响，并确定未来研究的途径。



## **50. Flexible and Scalable Deep Dendritic Spiking Neural Networks with Multiple Nonlinear Branching**

具有多个非线性分支的灵活可扩展深度树枝状尖峰神经网络 cs.NE

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06355v1) [paper-pdf](http://arxiv.org/pdf/2412.06355v1)

**Authors**: Yifan Huang, Wei Fang, Zhengyu Ma, Guoqi Li, Yonghong Tian

**Abstract**: Recent advances in spiking neural networks (SNNs) have a predominant focus on network architectures, while relatively little attention has been paid to the underlying neuron model. The point neuron models, a cornerstone of deep SNNs, pose a bottleneck on the network-level expressivity since they depict somatic dynamics only. In contrast, the multi-compartment models in neuroscience offer remarkable expressivity by introducing dendritic morphology and dynamics, but remain underexplored in deep learning due to their unaffordable computational cost and inflexibility. To combine the advantages of both sides for a flexible, efficient yet more powerful model, we propose the dendritic spiking neuron (DendSN) incorporating multiple dendritic branches with nonlinear dynamics. Compared to the point spiking neurons, DendSN exhibits significantly higher expressivity. DendSN's flexibility enables its seamless integration into diverse deep SNN architectures. To accelerate dendritic SNNs (DendSNNs), we parallelize dendritic state updates across time steps, and develop Triton kernels for GPU-level acceleration. As a result, we can construct large-scale DendSNNs with depth comparable to their point SNN counterparts. Next, we comprehensively evaluate DendSNNs' performance on various demanding tasks. By modulating dendritic branch strengths using a context signal, catastrophic forgetting of DendSNNs is substantially mitigated. Moreover, DendSNNs demonstrate enhanced robustness against noise and adversarial attacks compared to point SNNs, and excel in few-shot learning settings. Our work firstly demonstrates the possibility of training bio-plausible dendritic SNNs with depths and scales comparable to traditional point SNNs, and reveals superior expressivity and robustness of reduced dendritic neuron models in deep learning, thereby offering a fresh perspective on advancing neural network design.

摘要: 尖峰神经网络(SNN)的最新进展主要集中在网络结构上，而对潜在的神经元模型的关注相对较少。点神经元模型是深层次SNN的基石，但由于其仅描述体细胞动力学，对网络层次的表现力构成了瓶颈。相比之下，神经科学中的多室模型通过引入树突形态和动力学提供了显著的表现力，但由于其负担不起的计算成本和灵活性，在深度学习中仍然没有得到充分的探索。为了结合两者的优点建立一个灵活、高效且功能更强大的模型，我们提出了一种结合了多个树枝和非线性动力学的树状突起神经元模型(DendSN)。与点刺神经元相比，DendSN的表达显著增强。DendSN的灵活性使其能够无缝集成到各种深度SNN架构中。为了加速树状SNN(DendSNN)，我们跨时间步长并行更新树状状态，并开发用于GPU级加速的Triton核。因此，我们可以构建大规模的DendSNN，其深度可与它们的点SNN相媲美。接下来，我们将全面评估DendSNns在各种高要求任务上的表现。通过使用上下文信号调制树枝分支强度，大大减轻了树突状SNN的灾难性遗忘。此外，与点SNN相比，DendSNN表现出对噪声和敌意攻击的更强的稳健性，并且在少镜头学习环境中表现出色。我们的工作首次证明了训练生物可信的树突状神经网络的可能性，其深度和规模与传统的点状神经网络相当，并揭示了简化的树突状神经元模型在深度学习中的优越表达能力和健壮性，从而为进一步的神经网络设计提供了一个新的视角。



