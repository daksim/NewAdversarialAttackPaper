# Latest Adversarial Attack Papers
**update at 2024-12-18 10:31:16**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adaptive Epsilon Adversarial Training for Robust Gravitational Wave Parameter Estimation Using Normalizing Flows**

使用正规化流进行鲁棒引力波参数估计的自适应Episodes对抗训练 cs.LG

Due to new experimental results to add to the paper, this version no  longer accurately reflects the current state of our research. Therefore, we  are withdrawing the paper while further experiments are conducted. We will  submit a new version in the future. We apologize for any inconvenience this  may cause

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.07559v2) [paper-pdf](http://arxiv.org/pdf/2412.07559v2)

**Authors**: Yiqian Yang, Xihua Zhu, Fan Zhang

**Abstract**: Adversarial training with Normalizing Flow (NF) models is an emerging research area aimed at improving model robustness through adversarial samples. In this study, we focus on applying adversarial training to NF models for gravitational wave parameter estimation. We propose an adaptive epsilon method for Fast Gradient Sign Method (FGSM) adversarial training, which dynamically adjusts perturbation strengths based on gradient magnitudes using logarithmic scaling. Our hybrid architecture, combining ResNet and Inverse Autoregressive Flow, reduces the Negative Log Likelihood (NLL) loss by 47\% under FGSM attacks compared to the baseline model, while maintaining an NLL of 4.2 on clean data (only 5\% higher than the baseline). For perturbation strengths between 0.01 and 0.1, our model achieves an average NLL of 5.8, outperforming both fixed-epsilon (NLL: 6.7) and progressive-epsilon (NLL: 7.2) methods. Under stronger Projected Gradient Descent attacks with perturbation strength of 0.05, our model maintains an NLL of 6.4, demonstrating superior robustness while avoiding catastrophic overfitting.

摘要: 利用归一化流量模型进行对抗性训练是一个新兴的研究领域，其目的是通过对抗性样本提高模型的稳健性。在这项研究中，我们将对抗性训练应用到引力波参数估计的神经网络模型中。提出了一种用于快速梯度符号法(FGSM)对抗训练的自适应epsilon方法，该方法利用对数尺度根据梯度大小动态调整扰动强度。我们的混合架构结合了ResNet和反向自回归流，与基线模型相比，在FGSM攻击下，负对数似然(NLL)损失降低了47%，而对于干净的数据，NLL保持在4.2(仅比基线高5%)。当微扰强度在0.01到0.1之间时，我们的模型的平均NLL为5.8，优于固定epsilon(NLL：6.7)和渐进epsilon(NLL：7.2)方法。在扰动强度为0.05的较强投影梯度下降攻击下，我们的模型保持了6.4的NLL，在避免灾难性过拟合的同时表现出了优越的稳健性。



## **2. PROSAC: Provably Safe Certification for Machine Learning Models under Adversarial Attacks**

PROSAC：对抗性攻击下的机器学习模型可证明安全的认证 cs.LG

Accepted to AAAI2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2402.02629v2) [paper-pdf](http://arxiv.org/pdf/2402.02629v2)

**Authors**: Chen Feng, Ziquan Liu, Zhuo Zhi, Ilija Bogunovic, Carsten Gerner-Beuerle, Miguel Rodrigues

**Abstract**: It is widely known that state-of-the-art machine learning models, including vision and language models, can be seriously compromised by adversarial perturbations. It is therefore increasingly relevant to develop capabilities to certify their performance in the presence of the most effective adversarial attacks. Our paper offers a new approach to certify the performance of machine learning models in the presence of adversarial attacks with population level risk guarantees. In particular, we introduce the notion of $(\alpha,\zeta)$-safe machine learning model. We propose a hypothesis testing procedure, based on the availability of a calibration set, to derive statistical guarantees providing that the probability of declaring that the adversarial (population) risk of a machine learning model is less than $\alpha$ (i.e. the model is safe), while the model is in fact unsafe (i.e. the model adversarial population risk is higher than $\alpha$), is less than $\zeta$. We also propose Bayesian optimization algorithms to determine efficiently whether a machine learning model is $(\alpha,\zeta)$-safe in the presence of an adversarial attack, along with statistical guarantees. We apply our framework to a range of machine learning models - including various sizes of vision Transformer (ViT) and ResNet models - impaired by a variety of adversarial attacks, such as PGDAttack, MomentumAttack, GenAttack and BanditAttack, to illustrate the operation of our approach. Importantly, we show that ViT's are generally more robust to adversarial attacks than ResNets, and large models are generally more robust than smaller models. Our approach goes beyond existing empirical adversarial risk-based certification guarantees. It formulates rigorous (and provable) performance guarantees that can be used to satisfy regulatory requirements mandating the use of state-of-the-art technical tools.

摘要: 众所周知，最先进的机器学习模型，包括视觉和语言模型，可能会受到对抗性扰动的严重影响。因此，越来越有必要发展能力，以证明它们在最有效的对抗性攻击下的表现。本文提供了一种新的方法来证明机器学习模型在种群水平风险保证的对抗性攻击下的性能。特别地，我们引入了$(\α，\Zeta)$-安全机器学习模型的概念。我们提出了一种假设检验程序，基于校准集的可用性来获得统计保证，假设宣布一个机器学习模型的对抗(总体)风险小于$\α$(即该模型是安全的)，而该模型实际上是不安全的(即该模型的对抗总体风险高于$\α$)的概率小于$\Zeta$。我们还提出了贝叶斯优化算法来有效地确定机器学习模型在存在对抗性攻击的情况下是否$(\α，\Zeta)$安全，并提供统计保证。我们将我们的框架应用于一系列机器学习模型，包括各种大小的视觉转换器(VIT)和ResNet模型，这些模型被各种敌意攻击所破坏，如PGDAttack、MomentumAttack、GenAttack和BanditAttack，以说明我们方法的操作。重要的是，我们发现VIT通常比ResNet对对手攻击更健壮，大模型通常比小模型更健壮。我们的方法超越了现有的经验对抗性、基于风险的认证保证。它制定了严格的(和可证明的)性能保证，可用于满足要求使用最先进技术工具的监管要求。



## **3. Deep Learning for Resilient Adversarial Decision Fusion in Byzantine Networks**

深度学习用于拜占庭网络中弹性对抗决策融合 cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12739v1) [paper-pdf](http://arxiv.org/pdf/2412.12739v1)

**Authors**: Kassem Kallas

**Abstract**: This paper introduces a deep learning-based framework for resilient decision fusion in adversarial multi-sensor networks, providing a unified mathematical setup that encompasses diverse scenarios, including varying Byzantine node proportions, synchronized and unsynchronized attacks, unbalanced priors, adaptive strategies, and Markovian states. Unlike traditional methods, which depend on explicit parameter tuning and are limited by scenario-specific assumptions, the proposed approach employs a deep neural network trained on a globally constructed dataset to generalize across all cases without requiring adaptation. Extensive simulations validate the method's robustness, achieving superior accuracy, minimal error probability, and scalability compared to state-of-the-art techniques, while ensuring computational efficiency for real-time applications. This unified framework demonstrates the potential of deep learning to revolutionize decision fusion by addressing the challenges posed by Byzantine nodes in dynamic adversarial environments.

摘要: 提出了一种基于深度学习的对抗性多传感器网络弹性决策融合框架，提供了一个统一的数学模型，涵盖了拜占庭节点比例变化、同步和非同步攻击、不平衡先验、自适应策略和马尔可夫状态等多种场景。与依赖于显式参数调整的传统方法不同，该方法使用在全局构造的数据集上训练的深度神经网络来对所有情况进行泛化，而不需要自适应。大量的仿真验证了该方法的稳健性，与最先进的技术相比，实现了更高的精确度、最小的错误概率和可扩展性，同时确保了实时应用的计算效率。这一统一的框架展示了深度学习通过解决动态对抗环境中拜占庭节点带来的挑战来彻底改变决策融合的潜力。



## **4. On the Impact of Hard Adversarial Instances on Overfitting in Adversarial Training**

论硬对抗预设对对抗训练中过度配合的影响 cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2112.07324v2) [paper-pdf](http://arxiv.org/pdf/2112.07324v2)

**Authors**: Chen Liu, Zhichao Huang, Mathieu Salzmann, Tong Zhang, Sabine Süsstrunk

**Abstract**: Adversarial training is a popular method to robustify models against adversarial attacks. However, it exhibits much more severe overfitting than training on clean inputs. In this work, we investigate this phenomenon from the perspective of training instances, i.e., training input-target pairs. Based on a quantitative metric measuring the relative difficulty of an instance in the training set, we analyze the model's behavior on training instances of different difficulty levels. This lets us demonstrate that the decay in generalization performance of adversarial training is a result of fitting hard adversarial instances. We theoretically verify our observations for both linear and general nonlinear models, proving that models trained on hard instances have worse generalization performance than ones trained on easy instances, and that this generalization gap increases with the size of the adversarial budget. Finally, we investigate solutions to mitigate adversarial overfitting in several scenarios, including fast adversarial training and fine-tuning a pretrained model with additional data. Our results demonstrate that using training data adaptively improves the model's robustness.

摘要: 对抗性训练是一种流行的方法，用来增强模型抵御对抗性攻击的能力。然而，它表现出比清洁投入培训严重得多的过度适应。在这项工作中，我们从训练实例的角度来研究这一现象，即训练输入-目标对。基于一个量化度量实例在训练集中的相对难度，我们分析了该模型在不同难度级别的训练实例上的行为。这使我们能够证明，对抗性训练的泛化性能的下降是拟合困难的对抗性实例的结果。我们从理论上验证了我们对线性和一般非线性模型的观察结果，证明了在硬实例上训练的模型比在简单实例上训练的模型具有更差的泛化性能，并且这种泛化差距随着对抗预算的大小而增大。最后，我们研究了在几种情况下缓解对抗过度匹配的解决方案，包括快速的对抗训练和用额外的数据微调预先训练的模型。结果表明，训练数据的自适应使用提高了模型的稳健性。



## **5. Building Gradient Bridges: Label Leakage from Restricted Gradient Sharing in Federated Learning**

构建梯度桥：联邦学习中受限制的梯度共享造成的标签泄漏 cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12640v1) [paper-pdf](http://arxiv.org/pdf/2412.12640v1)

**Authors**: Rui Zhang, Ka-Ho Chow, Ping Li

**Abstract**: The growing concern over data privacy, the benefits of utilizing data from diverse sources for model training, and the proliferation of networked devices with enhanced computational capabilities have all contributed to the rise of federated learning (FL). The clients in FL collaborate to train a global model by uploading gradients computed on their private datasets without collecting raw data. However, a new attack surface has emerged from gradient sharing, where adversaries can restore the label distribution of a victim's private data by analyzing the obtained gradients. To mitigate this privacy leakage, existing lightweight defenses restrict the sharing of gradients, such as encrypting the final-layer gradients or locally updating the parameters within. In this paper, we introduce a novel attack called Gradient Bridge (GDBR) that recovers the label distribution of training data from the limited gradient information shared in FL. GDBR explores the relationship between the layer-wise gradients, tracks the flow of gradients, and analytically derives the batch training labels. Extensive experiments show that GDBR can accurately recover more than 80% of labels in various FL settings. GDBR highlights the inadequacy of restricted gradient sharing-based defenses and calls for the design of effective defense schemes in FL.

摘要: 对数据隐私的日益关注，利用来自不同来源的数据进行模型训练的好处，以及具有增强计算能力的联网设备的激增，所有这些都促进了联合学习(FL)的兴起。FL中的客户通过上传在其私有数据集上计算的梯度来协作训练全局模型，而不收集原始数据。然而，从梯度共享中出现了一个新的攻击面，攻击者可以通过分析获得的梯度来恢复受害者私人数据的标签分布。为了缓解这种隐私泄露，现有的轻量级防御限制了渐变的共享，例如加密最终层渐变或本地更新其中的参数。本文提出了一种新的攻击方法--梯度桥(GDBR)，它从FL共享的有限的梯度信息中恢复训练数据的标签分布。GDBR探索逐层梯度之间的关系，跟踪梯度的流动，并解析地得出批量训练标签。大量实验表明，在各种FL环境下，GDBR可以准确地恢复80%以上的标签。GDBR强调了基于受限梯度共享的防御的不足，并呼吁在FL中设计有效的防御方案。



## **6. Improving the Transferability of 3D Point Cloud Attack via Spectral-aware Admix and Optimization Designs**

通过光谱感知的混合和优化设计提高3D点云攻击的可转移性 cs.CV

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12626v1) [paper-pdf](http://arxiv.org/pdf/2412.12626v1)

**Authors**: Shiyu Hu, Daizong Liu, Wei Hu

**Abstract**: Deep learning models for point clouds have shown to be vulnerable to adversarial attacks, which have received increasing attention in various safety-critical applications such as autonomous driving, robotics, and surveillance. Existing 3D attackers generally design various attack strategies in the white-box setting, requiring the prior knowledge of 3D model details. However, real-world 3D applications are in the black-box setting, where we can only acquire the outputs of the target classifier. Although few recent works try to explore the black-box attack, they still achieve limited attack success rates (ASR). To alleviate this issue, this paper focuses on attacking the 3D models in a transfer-based black-box setting, where we first carefully design adversarial examples in a white-box surrogate model and then transfer them to attack other black-box victim models. Specifically, we propose a novel Spectral-aware Admix with Augmented Optimization method (SAAO) to improve the adversarial transferability. In particular, since traditional Admix strategy are deployed in the 2D domain that adds pixel-wise images for perturbing, we can not directly follow it to merge point clouds in coordinate domain as it will destroy the geometric shapes. Therefore, we design spectral-aware fusion that performs Graph Fourier Transform (GFT) to get spectral features of the point clouds and add them in the spectral domain. Afterward, we run a few steps with spectral-aware weighted Admix to select better optimization paths as well as to adjust corresponding learning weights. At last, we run more steps to generate adversarial spectral feature along the optimization path and perform Inverse-GFT on the adversarial spectral feature to obtain the adversarial example in the data domain. Experiments show that our SAAO achieves better transferability compared to existing 3D attack methods.

摘要: 点云的深度学习模型容易受到敌意攻击，在自动驾驶、机器人和监控等各种安全关键应用中受到越来越多的关注。现有的3D攻击者一般在白盒环境下设计各种攻击策略，需要对3D模型细节的先验知识。然而，现实世界中的3D应用程序处于黑盒设置中，我们只能获取目标分类器的输出。虽然最近很少有文献对黑盒攻击进行研究，但它们的攻击成功率(ASR)仍然有限。为了缓解这一问题，本文重点攻击基于转移的黑箱环境中的3D模型，首先在白箱代理模型中精心设计对抗性实例，然后将它们转移到攻击其他黑箱受害者模型。具体地说，我们提出了一种新的频谱感知广告混合增强优化方法(SAAO)来提高对抗可转移性。特别是，由于传统的AdMix策略是在二维域中添加像素级的图像进行扰动，因此不能直接跟随它在坐标域中合并点云，因为这会破坏几何形状。因此，我们设计了光谱感知融合，通过图形傅里叶变换(GFT)来获取点云的光谱特征，并将其添加到谱域中。然后，我们使用谱感知加权AdMix运行几个步骤来选择更好的优化路径以及调整相应的学习权重。最后，我们沿着优化路径运行更多的步骤来生成对抗性谱特征，并对对抗性谱特征进行逆GFT，得到数据域中的对抗性实例。实验表明，与现有的3D攻击方法相比，我们的SAAO实现了更好的可移植性。



## **7. Jailbreaking? One Step Is Enough!**

越狱？一步就够了！ cs.CL

17 pages

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12621v1) [paper-pdf](http://arxiv.org/pdf/2412.12621v1)

**Authors**: Weixiong Zheng, Peijian Zeng, Yiwei Li, Hongyan Wu, Nankai Lin, Junhao Chen, Aimin Yang, Yongmei Zhou

**Abstract**: Large language models (LLMs) excel in various tasks but remain vulnerable to jailbreak attacks, where adversaries manipulate prompts to generate harmful outputs. Examining jailbreak prompts helps uncover the shortcomings of LLMs. However, current jailbreak methods and the target model's defenses are engaged in an independent and adversarial process, resulting in the need for frequent attack iterations and redesigning attacks for different models. To address these gaps, we propose a Reverse Embedded Defense Attack (REDA) mechanism that disguises the attack intention as the "defense". intention against harmful content. Specifically, REDA starts from the target response, guiding the model to embed harmful content within its defensive measures, thereby relegating harmful content to a secondary role and making the model believe it is performing a defensive task. The attacking model considers that it is guiding the target model to deal with harmful content, while the target model thinks it is performing a defensive task, creating an illusion of cooperation between the two. Additionally, to enhance the model's confidence and guidance in "defensive" intentions, we adopt in-context learning (ICL) with a small number of attack examples and construct a corresponding dataset of attack examples. Extensive evaluations demonstrate that the REDA method enables cross-model attacks without the need to redesign attack strategies for different models, enables successful jailbreak in one iteration, and outperforms existing methods on both open-source and closed-source models.

摘要: 大型语言模型(LLM)在各种任务中表现出色，但仍然容易受到越狱攻击的攻击，在越狱攻击中，对手操纵提示以生成有害的输出。检查越狱提示有助于发现LLMS的缺点。然而，当前的越狱方法和目标模型的防御都是独立的和对抗性的过程，导致需要频繁的攻击迭代和针对不同模型的重新设计攻击。针对这些漏洞，我们提出了一种反向嵌入防御攻击(REDA)机制，将攻击意图伪装成“防御”。针对有害内容的意图。具体地说，Reda从目标响应开始，引导模型在其防御措施中嵌入有害内容，从而将有害内容降级为次要角色，并使模型相信它正在执行防御任务。攻击模型认为它是在引导目标模型处理有害内容，而目标模型则认为它是在执行防御任务，制造了两者合作的错觉。此外，为了增强模型对“防御”意图的可信度和指导性，我们采用了上下文中学习(ICL)的方法，并结合少量攻击实例构建了相应的攻击实例数据集。广泛的评估表明，REDA方法支持跨模型攻击，不需要针对不同的模型重新设计攻击策略，一次迭代即可成功越狱，并且在开源和闭源模型上的性能都优于现有方法。



## **8. WaterPark: A Robustness Assessment of Language Model Watermarking**

WaterPark：语言模型水印的稳健性评估 cs.CR

22 pages

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2411.13425v2) [paper-pdf](http://arxiv.org/pdf/2411.13425v2)

**Authors**: Jiacheng Liang, Zian Wang, Lauren Hong, Shouling Ji, Ting Wang

**Abstract**: Various watermarking methods (``watermarkers'') have been proposed to identify LLM-generated texts; yet, due to the lack of unified evaluation platforms, many critical questions remain under-explored: i) What are the strengths/limitations of various watermarkers, especially their attack robustness? ii) How do various design choices impact their robustness? iii) How to optimally operate watermarkers in adversarial environments? To fill this gap, we systematize existing LLM watermarkers and watermark removal attacks, mapping out their design spaces. We then develop WaterPark, a unified platform that integrates 10 state-of-the-art watermarkers and 12 representative attacks. More importantly, by leveraging WaterPark, we conduct a comprehensive assessment of existing watermarkers, unveiling the impact of various design choices on their attack robustness. We further explore the best practices to operate watermarkers in adversarial environments. We believe our study sheds light on current LLM watermarking techniques while WaterPark serves as a valuable testbed to facilitate future research.

摘要: 已经提出了各种水印方法来识别LLM生成的文本；然而，由于缺乏统一的评估平台，许多关键问题仍然没有得到充分的研究：i)各种水印的优点/局限性是什么，特别是它们的攻击稳健性？Ii)各种设计选择对其健壮性有何影响？三)如何在对抗性环境中以最佳方式使用水印？为了填补这一空白，我们对现有的LLM水印和水印移除攻击进行了系统化，规划了它们的设计空间。然后我们开发了Water Park，这是一个统一的平台，集成了10个最先进的水印和12个具有代表性的攻击。更重要的是，通过利用水上公园，我们对现有的水印进行了全面的评估，揭示了各种设计选择对其攻击健壮性的影响。我们进一步探索在对抗性环境中操作水印的最佳实践。我们相信我们的研究对当前的LLM数字水印技术有一定的启发作用，同时也为以后的研究提供了一个有价值的实验平台。



## **9. Attack On Prompt: Backdoor Attack in Prompt-Based Continual Learning**

攻击提示：基于预算的持续学习中的后门攻击 cs.LG

Accepted to AAAI 2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2406.19753v2) [paper-pdf](http://arxiv.org/pdf/2406.19753v2)

**Authors**: Trang Nguyen, Anh Tran, Nhat Ho

**Abstract**: Prompt-based approaches offer a cutting-edge solution to data privacy issues in continual learning, particularly in scenarios involving multiple data suppliers where long-term storage of private user data is prohibited. Despite delivering state-of-the-art performance, its impressive remembering capability can become a double-edged sword, raising security concerns as it might inadvertently retain poisoned knowledge injected during learning from private user data. Following this insight, in this paper, we expose continual learning to a potential threat: backdoor attack, which drives the model to follow a desired adversarial target whenever a specific trigger is present while still performing normally on clean samples. We highlight three critical challenges in executing backdoor attacks on incremental learners and propose corresponding solutions: (1) \emph{Transferability}: We employ a surrogate dataset and manipulate prompt selection to transfer backdoor knowledge to data from other suppliers; (2) \emph{Resiliency}: We simulate static and dynamic states of the victim to ensure the backdoor trigger remains robust during intense incremental learning processes; and (3) \emph{Authenticity}: We apply binary cross-entropy loss as an anti-cheating factor to prevent the backdoor trigger from devolving into adversarial noise. Extensive experiments across various benchmark datasets and continual learners validate our continual backdoor framework, achieving up to $100\%$ attack success rate, with further ablation studies confirming our contributions' effectiveness.

摘要: 基于提示的方法为持续学习中的数据隐私问题提供了一种尖端解决方案，特别是在涉及多个数据供应商的场景中，禁止长期存储私人用户数据。尽管提供了最先进的性能，但其令人印象深刻的记忆能力可能会成为一把双刃剑，这引发了安全问题，因为它可能会无意中保留在从私人用户数据学习过程中注入的有毒知识。根据这一见解，在本文中，我们将持续学习暴露于一个潜在的威胁：后门攻击，它驱动模型在出现特定触发时跟踪期望的对手目标，同时仍然在干净的样本上正常运行。我们强调了对增量学习者执行后门攻击的三个关键挑战并提出了相应的解决方案：(1)\emph{可传递性}：我们使用代理数据集并操纵提示选择来将后门知识传输到其他供应商的数据；(2)\emph{弹性}：我们模拟受害者的静态和动态，以确保后门触发在激烈的增量学习过程中保持健壮；以及(3)\emph{真实性}：我们应用二进制交叉熵损失作为反作弊因子，以防止后门触发演变为对抗性噪声。在各种基准数据集和不断学习的人中进行的广泛实验验证了我们的持续后门框架，实现了高达100美元的攻击成功率，进一步的消融研究证实了我们的贡献的有效性。



## **10. Do Parameters Reveal More than Loss for Membership Inference?**

参数揭示的不仅仅是会员推断的损失吗？ cs.LG

Accepted to Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2406.11544v3) [paper-pdf](http://arxiv.org/pdf/2406.11544v3)

**Authors**: Anshuman Suri, Xiao Zhang, David Evans

**Abstract**: Membership inference attacks are used as a key tool for disclosure auditing. They aim to infer whether an individual record was used to train a model. While such evaluations are useful to demonstrate risk, they are computationally expensive and often make strong assumptions about potential adversaries' access to models and training environments, and thus do not provide tight bounds on leakage from potential attacks. We show how prior claims around black-box access being sufficient for optimal membership inference do not hold for stochastic gradient descent, and that optimal membership inference indeed requires white-box access. Our theoretical results lead to a new white-box inference attack, IHA (Inverse Hessian Attack), that explicitly uses model parameters by taking advantage of computing inverse-Hessian vector products. Our results show that both auditors and adversaries may be able to benefit from access to model parameters, and we advocate for further research into white-box methods for membership inference.

摘要: 成员关系推断攻击被用作信息披露审计的关键工具。他们的目的是推断个人记录是否被用来训练模型。虽然这样的评估有助于显示风险，但它们的计算成本很高，而且通常会对潜在对手访问模型和训练环境做出强有力的假设，因此不会对潜在攻击的泄漏提供严格的限制。我们证明了关于黑盒访问的关于最优成员关系推理的先前声明如何不适用于随机梯度下降，而最优成员关系推理确实需要白盒访问。我们的理论结果导致了一种新的白盒推理攻击，IHA(逆向Hessian攻击)，它通过计算逆向Hessian向量积来显式地使用模型参数。我们的结果表明，审计师和对手都可能从访问模型参数中受益，我们主张进一步研究成员关系推理的白盒方法。



## **11. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

大型语言模型能否提高图神经网络的对抗鲁棒性？ cs.LG

accepted by KDD2025

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2408.08685v2) [paper-pdf](http://arxiv.org/pdf/2408.08685v2)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial attacks, especially for topology perturbations, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attacks. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph. The source code can be found in https://github.com/zhongjian-zhang/LLM4RGNN.

摘要: 图神经网络(GNN)容易受到敌意攻击，尤其是对拓扑扰动的攻击，许多提高GNN健壮性的方法受到了广泛的关注。最近，我们目睹了大型语言模型(LLM)的巨大成功，这导致许多人探索LLM在GNN上的巨大潜力。然而，它们主要集中在通过利用LLMS来增强节点特征来提高GNN的性能。因此，我们问：GNN的健壮性是否也会随着LLMS强大的理解和推理能力而得到增强？通过给出实验结果，我们发现，尽管LLMS可以提高GNN的健壮性，但其准确率仍然平均下降23.1%，这意味着GNN仍然非常容易受到拓扑攻击。因此，另一个问题是如何扩展LLMS在图对抗健壮性方面的能力。本文提出了一种基于LLM的稳健图结构推理框架LLM4RGNN，该框架将GPT-4的推理能力抽象为用于识别恶意边的局部LLM和用于发现丢失重要边的基于LLM的边预测器，以恢复稳健的图结构。大量的实验表明，LLM4RGNN在不同的GNN上一致地提高了健壮性。即使在某些扰动比增加到40%的情况下，GNN的精度仍然好于干净图形上的精度。源代码可以在https://github.com/zhongjian-zhang/LLM4RGNN.中找到



## **12. Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script**

对抗性文本的人在循环生成：以藏传文字为例 cs.CL

Review Version; Submitted to NAACL 2025 Demo Track

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12478v1) [paper-pdf](http://arxiv.org/pdf/2412.12478v1)

**Authors**: Xi Cao, Yuan Sun, Jiajun Li, Quzong Gesang, Nuo Qun, Tashi Nyima

**Abstract**: DNN-based language models perform excellently on various tasks, but even SOTA LLMs are susceptible to textual adversarial attacks. Adversarial texts play crucial roles in multiple subfields of NLP. However, current research has the following issues. (1) Most textual adversarial attack methods target rich-resourced languages. How do we generate adversarial texts for less-studied languages? (2) Most textual adversarial attack methods are prone to generating invalid or ambiguous adversarial texts. How do we construct high-quality adversarial robustness benchmarks? (3) New language models may be immune to part of previously generated adversarial texts. How do we update adversarial robustness benchmarks? To address the above issues, we introduce HITL-GAT, a system based on a general approach to human-in-the-loop generation of adversarial texts. HITL-GAT contains four stages in one pipeline: victim model construction, adversarial example generation, high-quality benchmark construction, and adversarial robustness evaluation. Additionally, we utilize HITL-GAT to make a case study on Tibetan script which can be a reference for the adversarial research of other less-studied languages.

摘要: 基于DNN的语言模型在各种任务中表现出色，但即使是Sota LLM也容易受到文本攻击。对抗性语篇在自然语言处理的多个子领域发挥着至关重要的作用。然而，目前的研究存在以下问题。(1)大多数文本对抗性攻击方法针对的是资源丰富的语言。如何为较少研究的语言生成对抗性文本？(2)大多数文本对抗性攻击方法容易产生无效或歧义的对抗性文本。我们如何构建高质量的对抗性健壮性基准？(3)新的语言模型可能对先前生成的部分对抗性文本免疫。我们如何更新对手健壮性基准？为了解决上述问题，我们引入了HITL-GAT，这是一个基于人在环中生成对抗性文本的通用方法的系统。HITL-GAT在一条流水线上包括四个阶段：受害者模型构建、对手实例生成、高质量基准构建和对手健壮性评估。此外，我们还利用HITL-GAT对藏文进行了实例研究，对其他研究较少的语言的对抗性研究具有一定的借鉴意义。



## **13. Architectural Patterns for Designing Quantum Artificial Intelligence Systems**

设计量子人工智能系统的架构模式 cs.SE

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2411.10487v3) [paper-pdf](http://arxiv.org/pdf/2411.10487v3)

**Authors**: Mykhailo Klymenko, Thong Hoang, Xiwei Xu, Zhenchang Xing, Muhammad Usman, Qinghua Lu, Liming Zhu

**Abstract**: Utilising quantum computing technology to enhance artificial intelligence systems is expected to improve training and inference times, increase robustness against noise and adversarial attacks, and reduce the number of parameters without compromising accuracy. However, moving beyond proof-of-concept or simulations to develop practical applications of these systems while ensuring high software quality faces significant challenges due to the limitations of quantum hardware and the underdeveloped knowledge base in software engineering for such systems. In this work, we have conducted a systematic mapping study to identify the challenges and solutions associated with the software architecture of quantum-enhanced artificial intelligence systems. The results of the systematic mapping study reveal several architectural patterns that describe how quantum components can be integrated into inference engines, as well as middleware patterns that facilitate communication between classical and quantum components. Each pattern realises a trade-off between various software quality attributes, such as efficiency, scalability, trainability, simplicity, portability, and deployability. The outcomes of this work have been compiled into a catalogue of architectural patterns.

摘要: 利用量子计算技术来增强人工智能系统，预计将改善训练和推理时间，提高对噪音和对手攻击的稳健性，并在不影响准确性的情况下减少参数数量。然而，由于量子硬件的限制和此类系统的软件工程知识库的不发达，超越概念验证或模拟来开发这些系统的实际应用，同时确保高软件质量面临着重大挑战。在这项工作中，我们进行了系统的映射研究，以确定与量子增强型人工智能系统的软件体系结构相关的挑战和解决方案。系统映射研究的结果揭示了几种描述量子组件如何集成到推理引擎中的架构模式，以及促进经典组件和量子组件之间通信的中间件模式。每个模式都实现了各种软件质量属性之间的权衡，例如效率、可伸缩性、可训练性、简单性、可移植性和可部署性。这项工作的成果已被汇编成建筑模式目录。



## **14. Adversarially robust generalization theory via Jacobian regularization for deep neural networks**

深度神经网络通过Jacobian正规化的对抗鲁棒概括理论 stat.ML

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.12449v1) [paper-pdf](http://arxiv.org/pdf/2412.12449v1)

**Authors**: Dongya Wu, Xin Li

**Abstract**: Powerful deep neural networks are vulnerable to adversarial attacks. To obtain adversarially robust models, researchers have separately developed adversarial training and Jacobian regularization techniques. There are abundant theoretical and empirical studies for adversarial training, but theoretical foundations for Jacobian regularization are still lacking. In this study, we show that Jacobian regularization is closely related to adversarial training in that $\ell_{2}$ or $\ell_{1}$ Jacobian regularized loss serves as an approximate upper bound on the adversarially robust loss under $\ell_{2}$ or $\ell_{\infty}$ adversarial attack respectively. Further, we establish the robust generalization gap for Jacobian regularized risk minimizer via bounding the Rademacher complexity of both the standard loss function class and Jacobian regularization function class. Our theoretical results indicate that the norms of Jacobian are related to both standard and robust generalization. We also perform experiments on MNIST data classification to demonstrate that Jacobian regularized risk minimization indeed serves as a surrogate for adversarially robust risk minimization, and that reducing the norms of Jacobian can improve both standard and robust generalization. This study promotes both theoretical and empirical understandings to adversarially robust generalization via Jacobian regularization.

摘要: 强大的深度神经网络很容易受到敌意攻击。为了获得对抗性稳健的模型，研究人员分别开发了对抗性训练和雅可比正则化技术。对抗性训练已经有了丰富的理论和实证研究，但雅可比正则化的理论基础还很缺乏。本文证明了雅可比正则化与对抗训练密切相关，即雅可比正则化损失分别作为对抗性攻击下对抗性稳健损失的近似上界。进一步，我们通过对标准损失函数类和雅可比正则化函数类的Rademacher复杂性的界，建立了雅可比正则化风险最小化的鲁棒推广间隙。我们的理论结果表明，雅可比的范数既与标准推广有关，也与稳健推广有关。我们还在MNIST数据分类上进行了实验，证明了雅可比正则化风险最小化确实可以作为对抗性稳健风险最小化的替代，并且降低雅可比范数可以提高标准泛化和稳健泛化。本研究通过雅可比正则化，促进了对逆稳健泛化的理论和经验的理解。



## **15. Quantum Adversarial Machine Learning and Defense Strategies: Challenges and Opportunities**

量子对抗机器学习和防御策略：挑战和机遇 quant-ph

24 pages, 9 figures, 12 tables

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.12373v1) [paper-pdf](http://arxiv.org/pdf/2412.12373v1)

**Authors**: Eric Yocam, Anthony Rizi, Mahesh Kamepalli, Varghese Vaidyan, Yong Wang, Gurcan Comert

**Abstract**: As quantum computing continues to advance, the development of quantum-secure neural networks is crucial to prevent adversarial attacks. This paper proposes three quantum-secure design principles: (1) using post-quantum cryptography, (2) employing quantum-resistant neural network architectures, and (3) ensuring transparent and accountable development and deployment. These principles are supported by various quantum strategies, including quantum data anonymization, quantum-resistant neural networks, and quantum encryption. The paper also identifies open issues in quantum security, privacy, and trust, and recommends exploring adaptive adversarial attacks and auto adversarial attacks as future directions. The proposed design principles and recommendations provide guidance for developing quantum-secure neural networks, ensuring the integrity and reliability of machine learning models in the quantum era.

摘要: 随着量子计算的不断发展，量子安全神经网络的发展对于防止对抗攻击至关重要。本文提出了三个量子安全设计原则：（1）使用后量子密码学，（2）采用抗量子神经网络架构，（3）确保透明和负责任的开发和部署。这些原则得到各种量子策略的支持，包括量子数据匿名化、量子抵抗神经网络和量子加密。该论文还指出了量子安全、隐私和信任方面的未决问题，并建议探索自适应对抗攻击和自动对抗攻击作为未来的方向。提出的设计原则和建议为开发量子安全神经网络提供了指导，确保量子时代机器学习模型的完整性和可靠性。



## **16. Multi-Robot Target Tracking with Sensing and Communication Danger Zones**

具有传感和通信危险区的多机器人目标跟踪 cs.RO

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2404.07880v3) [paper-pdf](http://arxiv.org/pdf/2404.07880v3)

**Authors**: Jiazhen Liu, Peihan Li, Yuwei Wu, Gaurav S. Sukhatme, Vijay Kumar, Lifeng Zhou

**Abstract**: Multi-robot target tracking finds extensive applications in different scenarios, such as environmental surveillance and wildfire management, which require the robustness of the practical deployment of multi-robot systems in uncertain and dangerous environments. Traditional approaches often focus on the performance of tracking accuracy with no modeling and assumption of the environments, neglecting potential environmental hazards which result in system failures in real-world deployments. To address this challenge, we investigate multi-robot target tracking in the adversarial environment considering sensing and communication attacks with uncertainty. We design specific strategies to avoid different danger zones and proposed a multi-agent tracking framework under the perilous environment. We approximate the probabilistic constraints and formulate practical optimization strategies to address computational challenges efficiently. We evaluate the performance of our proposed methods in simulations to demonstrate the ability of robots to adjust their risk-aware behaviors under different levels of environmental uncertainty and risk confidence. The proposed method is further validated via real-world robot experiments where a team of drones successfully track dynamic ground robots while being risk-aware of the sensing and/or communication danger zones.

摘要: 多机器人目标跟踪在环境监测、野火管理等不同场景中有着广泛的应用，这就要求多机器人系统在不确定和危险环境中的实际部署具有很强的鲁棒性。传统的方法往往只关注跟踪精度的性能，没有对环境进行建模和假设，而忽略了实际部署中可能导致系统故障的环境危害。为了应对这一挑战，我们研究了在具有不确定性的感知和通信攻击的对抗性环境中的多机器人目标跟踪。设计了避开不同危险区域的具体策略，提出了危险环境下的多智能体跟踪框架。我们对概率约束进行近似，并制定实用的优化策略来有效地应对计算挑战。我们在仿真中评估了我们提出的方法的性能，以展示机器人在不同的环境不确定性和风险置信度下调整其风险意识行为的能力。通过真实世界的机器人实验进一步验证了所提出的方法，其中一组无人机成功地跟踪了动态的地面机器人，同时意识到了传感和/或通信危险区域的风险。



## **17. Adversarial Attacks on Large Language Models in Medicine**

医学中对大型语言模型的对抗攻击 cs.AI

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2406.12259v3) [paper-pdf](http://arxiv.org/pdf/2406.12259v3)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.

摘要: 将大型语言模型(LLM)集成到医疗保健应用程序中，在医疗诊断、治疗建议和患者护理方面提供了有希望的进步。然而，LLMS对对抗性攻击的敏感性构成了一个重大威胁，可能会在微妙的医疗环境中导致有害后果。本研究调查了LLMS在三个医疗任务中对两种类型的对抗性攻击的脆弱性。利用真实世界的患者数据，我们证明了开源和专有LLM都容易受到跨多个任务的操纵。这项研究进一步表明，特定领域的任务在模型微调中需要比一般领域任务更多的对抗性数据才能有效地执行攻击，特别是对于能力更强的模型。我们发现，虽然整合对抗性数据并不会显著降低医学基准上的整体模型性能，但它确实会导致微调模型权重的显著变化，这表明了一条检测和对抗模型攻击的潜在路径。这项研究强调了迫切需要强有力的安全措施和开发防御机制来保护医疗应用中的低成本管理，以确保其在医疗保健环境中的安全和有效部署。



## **18. Robust Synthetic Data-Driven Detection of Living-Off-the-Land Reverse Shells**

陆地生活反向壳的稳健综合数据驱动检测 cs.CR

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2402.18329v2) [paper-pdf](http://arxiv.org/pdf/2402.18329v2)

**Authors**: Dmitrijs Trizna, Luca Demetrio, Battista Biggio, Fabio Roli

**Abstract**: Living-off-the-land (LOTL) techniques pose a significant challenge to security operations, exploiting legitimate tools to execute malicious commands that evade traditional detection methods. To address this, we present a robust augmentation framework for cyber defense systems as Security Information and Event Management (SIEM) solutions, enabling the detection of LOTL attacks such as reverse shells through machine learning. Leveraging real-world threat intelligence and adversarial training, our framework synthesizes diverse malicious datasets while preserving the variability of legitimate activity, ensuring high accuracy and low false-positive rates. We validate our approach through extensive experiments on enterprise-scale datasets, achieving a 90\% improvement in detection rates over non-augmented baselines at an industry-grade False Positive Rate (FPR) of $10^{-5}$. We define black-box data-driven attacks that successfully evade unprotected models, and develop defenses to mitigate them, producing adversarially robust variants of ML models. Ethical considerations are central to this work; we discuss safeguards for synthetic data generation and the responsible release of pre-trained models across four best performing architectures, including both adversarially and regularly trained variants: https://huggingface.co/dtrizna/quasarnix. Furthermore, we provide a malicious LOTL dataset containing over 1 million augmented attack variants to enable reproducible research and community collaboration: https://huggingface.co/datasets/dtrizna/QuasarNix. This work offers a reproducible, scalable, and production-ready defense against evolving LOTL threats.

摘要: 谋生(LOTL)技术对安全操作构成了重大挑战，它们利用合法工具执行恶意命令，从而规避了传统的检测方法。为了解决这个问题，我们提出了一个强大的网络防御系统增强框架作为安全信息和事件管理(SIEM)解决方案，使能够通过机器学习检测LOTL攻击，如反向外壳。利用真实世界的威胁情报和对抗训练，我们的框架综合了不同的恶意数据集，同时保留了合法活动的可变性，确保了高准确性和低假阳性率。我们通过在企业级数据集上的广泛实验验证了我们的方法，在行业级假阳性率(FPR)为10^{-5}$的情况下，检测率比非增强基线提高了90%。我们定义了成功逃避不受保护的模型的黑盒数据驱动攻击，并开发防御措施来缓解它们，产生对手健壮的ML模型变体。伦理方面的考虑是这项工作的核心；我们讨论了合成数据生成的保障措施，以及在四个性能最好的体系结构中负责任地发布预先训练的模型，包括对抗性和常规训练的变体：https://huggingface.co/dtrizna/quasarnix.此外，我们还提供了一个包含100多万个扩展攻击变体的恶意LOTL数据集，以支持可重现的研究和社区协作：https://huggingface.co/datasets/dtrizna/QuasarNix.这项工作提供了针对不断演变的LOTL威胁的可复制、可扩展和生产就绪的防御。



## **19. Sonar-based Deep Learning in Underwater Robotics: Overview, Robustness and Challenges**

水下机器人中基于声纳的深度学习：概述、稳健性和挑战 cs.RO

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.11840v1) [paper-pdf](http://arxiv.org/pdf/2412.11840v1)

**Authors**: Martin Aubard, Ana Madureira, Luís Teixeira, José Pinto

**Abstract**: With the growing interest in underwater exploration and monitoring, Autonomous Underwater Vehicles (AUVs) have become essential. The recent interest in onboard Deep Learning (DL) has advanced real-time environmental interaction capabilities relying on efficient and accurate vision-based DL models. However, the predominant use of sonar in underwater environments, characterized by limited training data and inherent noise, poses challenges to model robustness. This autonomy improvement raises safety concerns for deploying such models during underwater operations, potentially leading to hazardous situations. This paper aims to provide the first comprehensive overview of sonar-based DL under the scope of robustness. It studies sonar-based DL perception task models, such as classification, object detection, segmentation, and SLAM. Furthermore, the paper systematizes sonar-based state-of-the-art datasets, simulators, and robustness methods such as neural network verification, out-of-distribution, and adversarial attacks. This paper highlights the lack of robustness in sonar-based DL research and suggests future research pathways, notably establishing a baseline sonar-based dataset and bridging the simulation-to-reality gap.

摘要: 随着人们对水下探测和监测的兴趣与日俱增，自主水下机器人(AUV)已经成为必不可少的工具。最近对车载深度学习(DL)的兴趣依赖于高效和准确的基于视觉的DL模型，具有先进的实时环境交互能力。然而，声纳在水下环境中的主要应用具有训练数据有限和固有噪声的特点，这给模型的稳健性带来了挑战。这种自主性的改进增加了在水下作业期间部署此类模型的安全问题，可能会导致危险情况。本文的目的是在稳健性的范围内，首次对基于声纳的数字水声通信进行全面的综述。研究了基于声纳的目标识别任务模型，如分类、目标检测、分割、SLAM等。此外，本文还系统化了基于声纳的最先进的数据集、模拟器和稳健性方法，如神经网络验证、分布外分布和敌方攻击。本文强调了基于声纳的数字图书馆研究缺乏稳健性，并提出了未来的研究路径，特别是建立基于声纳的基线数据集和弥合模拟与现实之间的差距。



## **20. Transferable Adversarial Face Attack with Text Controlled Attribute**

具有文本控制属性的可转移对抗面部攻击 cs.CV

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.11735v1) [paper-pdf](http://arxiv.org/pdf/2412.11735v1)

**Authors**: Wenyun Li, Zheng Zhang, Xiangyuan Lan, Dongmei Jiang

**Abstract**: Traditional adversarial attacks typically produce adversarial examples under norm-constrained conditions, whereas unrestricted adversarial examples are free-form with semantically meaningful perturbations. Current unrestricted adversarial impersonation attacks exhibit limited control over adversarial face attributes and often suffer from low transferability. In this paper, we propose a novel Text Controlled Attribute Attack (TCA$^2$) to generate photorealistic adversarial impersonation faces guided by natural language. Specifically, the category-level personal softmax vector is employed to precisely guide the impersonation attacks. Additionally, we propose both data and model augmentation strategies to achieve transferable attacks on unknown target models. Finally, a generative model, \textit{i.e}, Style-GAN, is utilized to synthesize impersonated faces with desired attributes. Extensive experiments on two high-resolution face recognition datasets validate that our TCA$^2$ method can generate natural text-guided adversarial impersonation faces with high transferability. We also evaluate our method on real-world face recognition systems, \textit{i.e}, Face++ and Aliyun, further demonstrating the practical potential of our approach.

摘要: 传统的对抗性攻击通常在范数受限的条件下产生对抗性示例，而不受限制的对抗性示例是自由形式的，具有语义意义的扰动。当前不受限制的对抗性模仿攻击对对抗性面孔属性的控制有限，并且往往存在可转移性低的问题。本文提出了一种新的文本控制属性攻击(TCA$^2$)，用于生成自然语言引导下的照片真实感对抗性模仿人脸。具体地说，采用类别级的个人Softmax向量来精确地指导模仿攻击。此外，我们还提出了数据和模型扩充策略来实现对未知目标模型的可转移攻击。最后，利用一个生成模型在两个高分辨率人脸识别数据集上的大量实验验证了我们的TCA$^2$方法能够生成具有很高可转移性的自然文本引导的对抗性模拟人脸。我们还在真实的人脸识别系统



## **21. Against All Odds: Overcoming Typology, Script, and Language Confusion in Multilingual Embedding Inversion Attacks**

克服一切困难：克服多语言嵌入倒置攻击中的类型学、脚本和语言混乱 cs.CL

11 pages, 4 figures, 7 tables

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2408.11749v2) [paper-pdf](http://arxiv.org/pdf/2408.11749v2)

**Authors**: Yiyi Chen, Russa Biswas, Heather Lent, Johannes Bjerva

**Abstract**: Large Language Models (LLMs) are susceptible to malicious influence by cyber attackers through intrusions such as adversarial, backdoor, and embedding inversion attacks. In response, the burgeoning field of LLM Security aims to study and defend against such threats. Thus far, the majority of works in this area have focused on monolingual English models, however, emerging research suggests that multilingual LLMs may be more vulnerable to various attacks than their monolingual counterparts. While previous work has investigated embedding inversion over a small subset of European languages, it is challenging to extrapolate these findings to languages from different linguistic families and with differing scripts. To this end, we explore the security of multilingual LLMs in the context of embedding inversion attacks and investigate cross-lingual and cross-script inversion across 20 languages, spanning over 8 language families and 12 scripts. Our findings indicate that languages written in Arabic script and Cyrillic script are particularly vulnerable to embedding inversion, as are languages within the Indo-Aryan language family. We further observe that inversion models tend to suffer from language confusion, sometimes greatly reducing the efficacy of an attack. Accordingly, we systematically explore this bottleneck for inversion models, uncovering predictable patterns which could be leveraged by attackers. Ultimately, this study aims to further the field's understanding of the outstanding security vulnerabilities facing multilingual LLMs and raise awareness for the languages most at risk of negative impact from these attacks.

摘要: 大型语言模型(LLM)容易受到网络攻击者通过对抗性、后门和嵌入反转攻击等入侵的恶意影响。作为回应，LLM Security这个新兴领域的目标是研究和防御此类威胁。到目前为止，这一领域的研究大多集中在单语英语模型上，然而，新的研究表明，多语种的LLM可能比单语的LLM更容易受到各种攻击。虽然以前的工作已经研究了在一小部分欧洲语言上嵌入倒置，但将这些发现外推到来自不同语系和不同脚本的语言是具有挑战性的。为此，我们在嵌入倒置攻击的情况下探索了多语言LLMS的安全性，并研究了跨语言和跨脚本的跨语言和跨脚本倒置，涉及8个语系和12个脚本。我们的发现表明，用阿拉伯文字和西里尔文字书写的语言特别容易嵌入倒置，印度-雅利安语系的语言也是如此。我们进一步观察到，倒置模型往往受到语言混乱的影响，有时会极大地降低攻击的有效性。因此，我们系统地探索了倒置模型的这一瓶颈，揭示了可被攻击者利用的可预测模式。最终，这项研究旨在加深外地对多语种土地管理系统面临的突出安全漏洞的了解，并提高对最有可能受到这些攻击的负面影响的语言的认识。



## **22. Take Fake as Real: Realistic-like Robust Black-box Adversarial Attack to Evade AIGC Detection**

以假为真：类似现实的鲁棒黑匣子对抗攻击以逃避AIGC检测 cs.CV

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.06727v2) [paper-pdf](http://arxiv.org/pdf/2412.06727v2)

**Authors**: Caiyun Xie, Dengpan Ye, Yunming Zhang, Long Tang, Yunna Lv, Jiacheng Deng, Jiawei Song

**Abstract**: The security of AI-generated content (AIGC) detection is crucial for ensuring multimedia content credibility. To enhance detector security, research on adversarial attacks has become essential. However, most existing adversarial attacks focus only on GAN-generated facial images detection, struggle to be effective on multi-class natural images and diffusion-based detectors, and exhibit poor invisibility. To fill this gap, we first conduct an in-depth analysis of the vulnerability of AIGC detectors and discover the feature that detectors vary in vulnerability to different post-processing. Then, considering that the detector is agnostic in real-world scenarios and given this discovery, we propose a Realistic-like Robust Black-box Adversarial attack (R$^2$BA) with post-processing fusion optimization. Unlike typical perturbations, R$^2$BA uses real-world post-processing, i.e., Gaussian blur, JPEG compression, Gaussian noise and light spot to generate adversarial examples. Specifically, we use a stochastic particle swarm algorithm with inertia decay to optimize post-processing fusion intensity and explore the detector's decision boundary. Guided by the detector's fake probability, R$^2$BA enhances/weakens the detector-vulnerable/detector-robust post-processing intensity to strike a balance between adversariality and invisibility. Extensive experiments on popular/commercial AIGC detectors and datasets demonstrate that R$^2$BA exhibits impressive anti-detection performance, excellent invisibility, and strong robustness in GAN-based and diffusion-based cases. Compared to state-of-the-art white-box and black-box attacks, R$^2$BA shows significant improvements of 15\%--72\% and 21\%--47\% in anti-detection performance under the original and robust scenario respectively, offering valuable insights for the security of AIGC detection in real-world applications.

摘要: 人工智能生成内容(AIGC)检测的安全性是保证多媒体内容可信度的关键。为了提高探测器的安全性，对抗性攻击的研究变得至关重要。然而，现有的对抗性攻击大多只针对GaN生成的人脸图像检测，难以对多类自然图像和基于扩散的检测器有效，并且表现出较差的不可见性。为了填补这一空白，我们首先对AIGC检测器的脆弱性进行了深入的分析，发现了检测器对不同后处理的脆弱性不同的特点。然后，考虑到检测器在实际场景中是不可知的，并根据这一发现，我们提出了一种具有后处理融合优化的逼真的健壮黑盒对抗攻击(R$^2$BA)。与典型的扰动不同，R$^2$BA使用真实世界的后处理，即高斯模糊、JPEG压缩、高斯噪声和光斑来生成对抗性示例。具体地说，我们使用了一种带惯性衰减的随机粒子群算法来优化后处理融合强度，并探索了检测器的决策边界。在检测器伪概率的指导下，R$^2$BA增强/削弱了检测器易受攻击/检测器健壮的后处理强度，以在对抗性和不可见性之间取得平衡。在流行的/商用AIGC探测器和数据集上的大量实验表明，R$^2$BA在基于GaN和基于扩散的情况下具有令人印象深刻的抗检测性能、出色的不可见性和强大的稳健性。与最新的白盒和黑盒攻击相比，R$^2$BA在原始场景和健壮场景下的抗检测性能分别提高了15-72和21-47，为实际应用中AIGC检测的安全性提供了有价值的见解。



## **23. PriPHiT: Privacy-Preserving Hierarchical Training of Deep Neural Networks**

PriPhiT：深度神经网络的隐私保护分层训练 cs.CV

21 pages, 19 figures, 11 tables

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2408.05092v2) [paper-pdf](http://arxiv.org/pdf/2408.05092v2)

**Authors**: Yamin Sepehri, Pedram Pad, Pascal Frossard, L. Andrea Dunbar

**Abstract**: The training phase of deep neural networks requires substantial resources and as such is often performed on cloud servers. However, this raises privacy concerns when the training dataset contains sensitive content, e.g., facial or medical images. In this work, we propose a method to perform the training phase of a deep learning model on both an edge device and a cloud server that prevents sensitive content being transmitted to the cloud while retaining the desired information. The proposed privacy-preserving method uses adversarial early exits to suppress the sensitive content at the edge and transmits the task-relevant information to the cloud. This approach incorporates noise addition during the training phase to provide a differential privacy guarantee. We extensively test our method on different facial and medical datasets with diverse attributes using various deep learning architectures, showcasing its outstanding performance. We also demonstrate the effectiveness of privacy preservation through successful defenses against different white-box, deep and GAN-based reconstruction attacks. This approach is designed for resource-constrained edge devices, ensuring minimal memory usage and computational overhead.

摘要: 深度神经网络的训练阶段需要大量资源，因此通常在云服务器上执行。然而，当训练数据集包含敏感内容时，这会引起隐私问题，例如面部或医学图像。在这项工作中，我们提出了一种方法，在边缘设备和云服务器上执行深度学习模型的训练阶段，以防止敏感数据被传输到云中，同时保留所需的信息。提出的隐私保护方法使用对抗性的提前退出来抑制边缘敏感内容，并将与任务相关的信息传输到云中。这种方法在训练阶段加入了噪声，以提供不同的隐私保证。我们使用各种深度学习架构，在具有不同属性的不同面部和医学数据集上广泛测试了我们的方法，展示了其出色的性能。我们还通过成功防御不同的白盒攻击、深度攻击和基于GAN的重建攻击来展示隐私保护的有效性。此方法专为资源受限的边缘设备设计，可确保最小的内存使用和计算开销。



## **24. Towards Adversarial Robustness of Model-Level Mixture-of-Experts Architectures for Semantic Segmentation**

面向语义分割的模型级专家混合架构的对抗鲁棒性 cs.CV

Accepted for publication at ICMLA 2024

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.11608v1) [paper-pdf](http://arxiv.org/pdf/2412.11608v1)

**Authors**: Svetlana Pavlitska, Enrico Eisen, J. Marius Zöllner

**Abstract**: Vulnerability to adversarial attacks is a well-known deficiency of deep neural networks. Larger networks are generally more robust, and ensembling is one method to increase adversarial robustness: each model's weaknesses are compensated by the strengths of others. While an ensemble uses a deterministic rule to combine model outputs, a mixture of experts (MoE) includes an additional learnable gating component that predicts weights for the outputs of the expert models, thus determining their contributions to the final prediction. MoEs have been shown to outperform ensembles on specific tasks, yet their susceptibility to adversarial attacks has not been studied yet. In this work, we evaluate the adversarial vulnerability of MoEs for semantic segmentation of urban and highway traffic scenes. We show that MoEs are, in most cases, more robust to per-instance and universal white-box adversarial attacks and can better withstand transfer attacks. Our code is available at \url{https://github.com/KASTEL-MobilityLab/mixtures-of-experts/}.

摘要: 对敌意攻击的脆弱性是深度神经网络的一个众所周知的缺陷。更大的网络通常更健壮，而集成是增加对手健壮性的一种方法：每个模型的弱点被其他模型的优点所弥补。虽然集合使用确定性规则来组合模型输出，但专家混合(MOE)包括额外的可学习选通组件，该组件预测专家模型的输出的权重，从而确定它们对最终预测的贡献。MOE已被证明在特定任务中表现优于合奏，但它们对对手攻击的敏感性尚未被研究。在这项工作中，我们评估了MOE在城市和公路交通场景语义分割中的对抗脆弱性。我们表明，在大多数情况下，MOE对每个实例和通用白盒对抗攻击具有更强的健壮性，并且能够更好地抵御传输攻击。我们的代码可以在\url{https://github.com/KASTEL-MobilityLab/mixtures-of-experts/}.上找到



## **25. Towards Efficient Training and Evaluation of Robust Models against $l_0$ Bounded Adversarial Perturbations**

针对1_0美元有界对抗性扰动的稳健模型的有效训练和评估 cs.LG

Accepted by ICML2024

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2405.05075v2) [paper-pdf](http://arxiv.org/pdf/2405.05075v2)

**Authors**: Xuyang Zhong, Yixiao Huang, Chen Liu

**Abstract**: This work studies sparse adversarial perturbations bounded by $l_0$ norm. We propose a white-box PGD-like attack method named sparse-PGD to effectively and efficiently generate such perturbations. Furthermore, we combine sparse-PGD with a black-box attack to comprehensively and more reliably evaluate the models' robustness against $l_0$ bounded adversarial perturbations. Moreover, the efficiency of sparse-PGD enables us to conduct adversarial training to build robust models against sparse perturbations. Extensive experiments demonstrate that our proposed attack algorithm exhibits strong performance in different scenarios. More importantly, compared with other robust models, our adversarially trained model demonstrates state-of-the-art robustness against various sparse attacks. Codes are available at https://github.com/CityU-MLO/sPGD.

摘要: 这项工作研究了以$l_0$规范为界的稀疏对抗扰动。我们提出了一种名为sparse-PVD的白盒类PGD攻击方法，以有效且高效地生成此类扰动。此外，我们将稀疏PVD与黑匣子攻击相结合，以全面、更可靠地评估模型对1_0美元有界对抗扰动的鲁棒性。此外，稀疏PVD的效率使我们能够进行对抗训练，以构建针对稀疏扰动的稳健模型。大量实验表明，我们提出的攻击算法在不同场景下表现出强大的性能。更重要的是，与其他稳健模型相比，我们的对抗训练模型表现出了针对各种稀疏攻击的最新稳健性。代码可访问https://github.com/CityU-MLO/sPGD。



## **26. DG-Mamba: Robust and Efficient Dynamic Graph Structure Learning with Selective State Space Models**

DG-Mamba：使用选择性状态空间模型稳健高效的动态图结构学习 cs.LG

Accepted by the Main Technical Track of the 39th Annual AAAI  Conference on Artificial Intelligence (AAAI-2025)

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.08160v3) [paper-pdf](http://arxiv.org/pdf/2412.08160v3)

**Authors**: Haonan Yuan, Qingyun Sun, Zhaonan Wang, Xingcheng Fu, Cheng Ji, Yongjian Wang, Bo Jin, Jianxin Li

**Abstract**: Dynamic graphs exhibit intertwined spatio-temporal evolutionary patterns, widely existing in the real world. Nevertheless, the structure incompleteness, noise, and redundancy result in poor robustness for Dynamic Graph Neural Networks (DGNNs). Dynamic Graph Structure Learning (DGSL) offers a promising way to optimize graph structures. However, aside from encountering unacceptable quadratic complexity, it overly relies on heuristic priors, making it hard to discover underlying predictive patterns. How to efficiently refine the dynamic structures, capture intrinsic dependencies, and learn robust representations, remains under-explored. In this work, we propose the novel DG-Mamba, a robust and efficient Dynamic Graph structure learning framework with the Selective State Space Models (Mamba). To accelerate the spatio-temporal structure learning, we propose a kernelized dynamic message-passing operator that reduces the quadratic time complexity to linear. To capture global intrinsic dynamics, we establish the dynamic graph as a self-contained system with State Space Model. By discretizing the system states with the cross-snapshot graph adjacency, we enable the long-distance dependencies capturing with the selective snapshot scan. To endow learned dynamic structures more expressive with informativeness, we propose the self-supervised Principle of Relevant Information for DGSL to regularize the most relevant yet least redundant information, enhancing global robustness. Extensive experiments demonstrate the superiority of the robustness and efficiency of our DG-Mamba compared with the state-of-the-art baselines against adversarial attacks.

摘要: 动态图形表现出交织在一起的时空演化模式，广泛存在于现实世界中。然而，动态图神经网络的结构不完备性、噪声和冗余性导致其健壮性较差。动态图结构学习(DGSL)为优化图结构提供了一种很有前途的方法。然而，除了遇到不可接受的二次型复杂性外，它还过度依赖启发式先验，使得发现潜在的预测模式变得困难。如何有效地提炼动态结构，捕获内在依赖关系，并学习健壮的表示，仍未得到探索。在这项工作中，我们提出了一种新颖的DG-MAMBA，这是一种基于选择状态空间模型(MAMBA)的健壮而高效的动态图结构学习框架。为了加速时空结构的学习，我们提出了一种核化的动态消息传递算子，将二次时间复杂度降为线性。为了捕捉全局内在动力学，我们利用状态空间模型将动态图建立为一个自包含系统。通过使用交叉快照图邻接关系对系统状态进行离散化，实现了选择性快照扫描的远程依赖捕获。为了使学习到的动态结构具有更强的信息性，我们提出了DGSL的相关信息自监督原则，将相关程度最高但冗余最少的信息正则化，增强了全局鲁棒性。大量的实验证明了DG-MAMBA算法的健壮性和高效性，与目前最先进的对抗攻击基线算法相比具有更好的性能。



## **27. Enhancing Robustness in Incremental Learning with Adversarial Training**

通过对抗训练增强增量学习的鲁棒性 cs.CV

Accepted to AAAI 2025

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2312.03289v3) [paper-pdf](http://arxiv.org/pdf/2312.03289v3)

**Authors**: Seungju Cho, Hongsin Lee, Changick Kim

**Abstract**: Adversarial training is one of the most effective approaches against adversarial attacks. However, adversarial training has primarily been studied in scenarios where data for all classes is provided, with limited research conducted in the context of incremental learning where knowledge is introduced sequentially. In this study, we investigate Adversarially Robust Class Incremental Learning (ARCIL), which deals with adversarial robustness in incremental learning. We first explore a series of baselines that integrate incremental learning with existing adversarial training methods, finding that they lead to conflicts between acquiring new knowledge and retaining past knowledge. Furthermore, we discover that training new knowledge causes the disappearance of a key characteristic in robust models: a flat loss landscape in input space. To address such issues, we propose a novel and robust baseline for ARCIL, named \textbf{FL}atness-preserving \textbf{A}dversarial \textbf{I}ncremental learning for \textbf{R}obustness (\textbf{FLAIR}). Experimental results demonstrate that FLAIR significantly outperforms other baselines. To the best of our knowledge, we are the first to comprehensively investigate the baselines, challenges, and solutions for ARCIL, which we believe represents a significant advance toward achieving real-world robustness. Codes are available at \url{https://github.com/HongsinLee/FLAIR}.

摘要: 对抗性训练是对抗对抗性攻击最有效的方法之一。然而，对抗性训练主要是在提供所有班级的数据的情况下进行的，在循序渐进地引入知识的增量学习背景下进行的研究有限。在这项研究中，我们研究了对抗性稳健类增量学习(ARCIL)，它涉及增量学习中的对抗性稳健性。我们首先探索了一系列将增量学习与现有对抗性训练方法相结合的基线，发现它们导致了获取新知识和保留过去知识之间的冲突。此外，我们发现训练新知识会导致稳健模型中一个关键特征的消失：输入空间中的平坦损失景象。为了解决这些问题，我们提出了一种新颖而健壮的ARCIL基线，命名为保持恒定度的文本bf{A}动态文本bf{i}增量学习(extbf{FLAIR})。实验结果表明，FLAIR的性能明显优于其他基线。据我们所知，我们是第一个全面调查ARCIL的基线、挑战和解决方案的公司，我们认为这代表着实现现实世界健壮性的重大进步。代码可在\url{https://github.com/HongsinLee/FLAIR}.



## **28. UIBDiffusion: Universal Imperceptible Backdoor Attack for Diffusion Models**

UIB扩散：扩散模型的普遍不可感知后门攻击 cs.CR

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.11441v1) [paper-pdf](http://arxiv.org/pdf/2412.11441v1)

**Authors**: Yuning Han, Bingyin Zhao, Rui Chu, Feng Luo, Biplab Sikdar, Yingjie Lao

**Abstract**: Recent studies show that diffusion models (DMs) are vulnerable to backdoor attacks. Existing backdoor attacks impose unconcealed triggers (e.g., a gray box and eyeglasses) that contain evident patterns, rendering remarkable attack effects yet easy detection upon human inspection and defensive algorithms. While it is possible to improve stealthiness by reducing the strength of the backdoor, doing so can significantly compromise its generality and effectiveness. In this paper, we propose UIBDiffusion, the universal imperceptible backdoor attack for diffusion models, which allows us to achieve superior attack and generation performance while evading state-of-the-art defenses. We propose a novel trigger generation approach based on universal adversarial perturbations (UAPs) and reveal that such perturbations, which are initially devised for fooling pre-trained discriminative models, can be adapted as potent imperceptible backdoor triggers for DMs. We evaluate UIBDiffusion on multiple types of DMs with different kinds of samplers across various datasets and targets. Experimental results demonstrate that UIBDiffusion brings three advantages: 1) Universality, the imperceptible trigger is universal (i.e., image and model agnostic) where a single trigger is effective to any images and all diffusion models with different samplers; 2) Utility, it achieves comparable generation quality (e.g., FID) and even better attack success rate (i.e., ASR) at low poison rates compared to the prior works; and 3) Undetectability, UIBDiffusion is plausible to human perception and can bypass Elijah and TERD, the SOTA defenses against backdoors for DMs. We will release our backdoor triggers and code.

摘要: 最近的研究表明，扩散模型(DM)容易受到后门攻击。现有的后门攻击施加了包含明显模式的隐藏触发器(例如，灰色盒子和眼镜)，使攻击效果显著，但很容易检测到人工检查和防御算法。虽然可以通过降低后门的强度来提高隐蔽性，但这样做会显著影响后门的通用性和有效性。在本文中，我们提出了一种针对扩散模型的通用不可感知后门攻击--UIB扩散，它使我们能够在避开最先进的防御的同时获得优越的攻击和生成性能。我们提出了一种新的基于通用对抗性扰动(UAP)的触发器生成方法，并揭示了这种扰动最初是为愚弄预训练的区分模型而设计的，现在可以被改装成用于DM的有效的不可感知的后门触发器。我们使用不同类型的采样器在不同的数据集和目标上评估了UIB在多种类型的DM上的扩散。实验结果表明：1)通用性，隐形触发器具有普适性(即，图像和模型无关)，其中单个触发器对任何图像和具有不同采样器的所有扩散模型有效；2)实用性，它在较低的毒害率下获得了与已有工作相当的生成质量(例如，FID)和更好的攻击成功率(例如，ASR)；以及3)不可检测性，UIB扩散对人类感知是可信的，并且可以绕过Elijah和Terd，SOTA对DM的后门防御。我们将发布我们的后门触发器和代码。



## **29. Exploiting the Index Gradients for Optimization-Based Jailbreaking on Large Language Models**

利用索引要素对大型语言模型进行基于优化的越狱 cs.CL

13 pages,2 figures, accepted by COLING 2025

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.08615v2) [paper-pdf](http://arxiv.org/pdf/2412.08615v2)

**Authors**: Jiahui Li, Yongchang Hao, Haoyu Xu, Xing Wang, Yu Hong

**Abstract**: Despite the advancements in training Large Language Models (LLMs) with alignment techniques to enhance the safety of generated content, these models remain susceptible to jailbreak, an adversarial attack method that exposes security vulnerabilities in LLMs. Notably, the Greedy Coordinate Gradient (GCG) method has demonstrated the ability to automatically generate adversarial suffixes that jailbreak state-of-the-art LLMs. However, the optimization process involved in GCG is highly time-consuming, rendering the jailbreaking pipeline inefficient. In this paper, we investigate the process of GCG and identify an issue of Indirect Effect, the key bottleneck of the GCG optimization. To this end, we propose the Model Attack Gradient Index GCG (MAGIC), that addresses the Indirect Effect by exploiting the gradient information of the suffix tokens, thereby accelerating the procedure by having less computation and fewer iterations. Our experiments on AdvBench show that MAGIC achieves up to a 1.5x speedup, while maintaining Attack Success Rates (ASR) on par or even higher than other baselines. Our MAGIC achieved an ASR of 74% on the Llama-2 and an ASR of 54% when conducting transfer attacks on GPT-3.5. Code is available at https://github.com/jiah-li/magic.

摘要: 尽管在使用对齐技术训练大型语言模型(LLM)以增强生成内容的安全性方面取得了进展，但这些模型仍然容易受到越狱的影响，这是一种暴露LLM安全漏洞的对抗性攻击方法。值得注意的是，贪婪坐标梯度(GCG)方法已经展示了自动生成敌意后缀的能力，这些后缀是越狱最先进的LLM。然而，GCG涉及的优化过程非常耗时，使得越狱管道效率低下。在本文中，我们研究了GCG的过程，并找出了间接影响的问题，这是GCG优化的关键瓶颈。为此，我们提出了模型攻击梯度索引GCG(MAGIC)，它通过利用后缀标记的梯度信息来解决间接影响，从而以更少的计算量和更少的迭代来加速过程。我们在AdvBtch上的实验表明，Magic在保持攻击成功率(ASR)与其他基线相当甚至更高的情况下，实现了高达1.5倍的加速。我们的魔法在骆驼-2上达到了74%的ASR，当对GPT-3.5进行传输攻击时ASR达到54%。代码可在https://github.com/jiah-li/magic.上找到



## **30. Deep Learning Model Security: Threats and Defenses**

深度学习模型安全性：威胁和防御 cs.CR

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.08969v2) [paper-pdf](http://arxiv.org/pdf/2412.08969v2)

**Authors**: Tianyang Wang, Ziqian Bi, Yichao Zhang, Ming Liu, Weiche Hsieh, Pohsun Feng, Lawrence K. Q. Yan, Yizhu Wen, Benji Peng, Junyu Liu, Keyu Chen, Sen Zhang, Ming Li, Chuanqi Jiang, Xinyuan Song, Junjie Yang, Bowen Jing, Jintao Ren, Junhao Song, Hong-Ming Tseng, Silin Chen, Yunze Wang, Chia Xin Liang, Jiawei Xu, Xuanhe Pan, Jinlang Wang, Qian Niu

**Abstract**: Deep learning has transformed AI applications but faces critical security challenges, including adversarial attacks, data poisoning, model theft, and privacy leakage. This survey examines these vulnerabilities, detailing their mechanisms and impact on model integrity and confidentiality. Practical implementations, including adversarial examples, label flipping, and backdoor attacks, are explored alongside defenses such as adversarial training, differential privacy, and federated learning, highlighting their strengths and limitations.   Advanced methods like contrastive and self-supervised learning are presented for enhancing robustness. The survey concludes with future directions, emphasizing automated defenses, zero-trust architectures, and the security challenges of large AI models. A balanced approach to performance and security is essential for developing reliable deep learning systems.

摘要: 深度学习改变了人工智能应用程序，但面临着关键的安全挑战，包括对抗性攻击、数据中毒、模型盗窃和隐私泄露。本调查检查了这些漏洞，详细介绍了它们的机制以及对模型完整性和机密性的影响。实践实现，包括对抗性示例、标签翻转和后门攻击，与对抗性训练、差异隐私和联邦学习等防御措施一起进行了探讨，强调了它们的优点和局限性。   提出了对比学习和自我监督学习等先进方法来增强鲁棒性。该调查得出了未来的方向，强调自动化防御、零信任架构以及大型人工智能模型的安全挑战。平衡的性能和安全方法对于开发可靠的深度学习系统至关重要。



## **31. A Comprehensive Review of Adversarial Attacks on Machine Learning**

机器学习对抗性攻击的全面回顾 cs.CR

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.11384v1) [paper-pdf](http://arxiv.org/pdf/2412.11384v1)

**Authors**: Syed Quiser Ahmed, Bharathi Vokkaliga Ganesh, Sathyanarayana Sampath Kumar, Prakhar Mishra, Ravi Anand, Bhanuteja Akurathi

**Abstract**: This research provides a comprehensive overview of adversarial attacks on AI and ML models, exploring various attack types, techniques, and their potential harms. We also delve into the business implications, mitigation strategies, and future research directions. To gain practical insights, we employ the Adversarial Robustness Toolbox (ART) [1] library to simulate these attacks on real-world use cases, such as self-driving cars. Our goal is to inform practitioners and researchers about the challenges and opportunities in defending AI systems against adversarial threats. By providing a comprehensive comparison of different attack methods, we aim to contribute to the development of more robust and secure AI systems.

摘要: 这项研究全面概述了对人工智能和ML模型的对抗性攻击，探索了各种攻击类型、技术及其潜在危害。我们还深入研究了业务影响、缓解策略和未来的研究方向。为了获得实际见解，我们使用对抗鲁棒性搜索器（ART）[1]库来模拟对现实世界用例（例如自动驾驶汽车）的这些攻击。我们的目标是让从业者和研究人员了解保护人工智能系统免受对抗威胁的挑战和机遇。通过提供不同攻击方法的全面比较，我们的目标是为开发更强大、更安全的人工智能系统做出贡献。



## **32. Comprehensive Survey on Adversarial Examples in Cybersecurity: Impacts, Challenges, and Mitigation Strategies**

网络安全中的对抗示例全面调查：影响、挑战和缓解策略 cs.CR

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2412.12217v1) [paper-pdf](http://arxiv.org/pdf/2412.12217v1)

**Authors**: Li Li

**Abstract**: Deep learning (DL) has significantly transformed cybersecurity, enabling advancements in malware detection, botnet identification, intrusion detection, user authentication, and encrypted traffic analysis. However, the rise of adversarial examples (AE) poses a critical challenge to the robustness and reliability of DL-based systems. These subtle, crafted perturbations can deceive models, leading to severe consequences like misclassification and system vulnerabilities. This paper provides a comprehensive review of the impact of AE attacks on key cybersecurity applications, highlighting both their theoretical and practical implications. We systematically examine the methods used to generate adversarial examples, their specific effects across various domains, and the inherent trade-offs attackers face between efficacy and resource efficiency. Additionally, we explore recent advancements in defense mechanisms, including gradient masking, adversarial training, and detection techniques, evaluating their potential to enhance model resilience. By summarizing cutting-edge research, this study aims to bridge the gap between adversarial research and practical security applications, offering insights to fortify the adoption of DL solutions in cybersecurity.

摘要: 深度学习(DL)显著改变了网络安全，实现了恶意软件检测、僵尸网络识别、入侵检测、用户身份验证和加密流量分析方面的进步。然而，对抗性实例(AE)的兴起对基于DL的系统的健壮性和可靠性提出了严峻的挑战。这些细微的、精心设计的扰动可能会欺骗模型，导致错误分类和系统漏洞等严重后果。本文全面回顾了AE攻击对关键网络安全应用的影响，强调了它们的理论和实践意义。我们系统地研究了用于生成对抗性示例的方法、它们在不同领域中的具体效果，以及攻击者在有效性和资源效率之间所面临的内在权衡。此外，我们还探讨了防御机制的最新进展，包括梯度掩蔽、对抗性训练和检测技术，评估了它们增强模型弹性的潜力。通过总结前沿研究，本研究旨在弥合对抗性研究和实际安全应用之间的差距，为加强在网络安全中采用DL解决方案提供见解。



## **33. Failures to Find Transferable Image Jailbreaks Between Vision-Language Models**

未能在视觉语言模型之间找到可传输的图像越狱 cs.CL

NeurIPS 2024 Workshops: RBFM (Best Paper), Frontiers in AdvML (Oral),  Red Teaming GenAI (Oral), SoLaR (Spotlight), SATA

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2407.15211v2) [paper-pdf](http://arxiv.org/pdf/2407.15211v2)

**Authors**: Rylan Schaeffer, Dan Valentine, Luke Bailey, James Chua, Cristóbal Eyzaguirre, Zane Durante, Joe Benton, Brando Miranda, Henry Sleight, John Hughes, Rajashree Agrawal, Mrinank Sharma, Scott Emmons, Sanmi Koyejo, Ethan Perez

**Abstract**: The integration of new modalities into frontier AI systems offers exciting capabilities, but also increases the possibility such systems can be adversarially manipulated in undesirable ways. In this work, we focus on a popular class of vision-language models (VLMs) that generate text outputs conditioned on visual and textual inputs. We conducted a large-scale empirical study to assess the transferability of gradient-based universal image ``jailbreaks" using a diverse set of over 40 open-parameter VLMs, including 18 new VLMs that we publicly release. Overall, we find that transferable gradient-based image jailbreaks are extremely difficult to obtain. When an image jailbreak is optimized against a single VLM or against an ensemble of VLMs, the jailbreak successfully jailbreaks the attacked VLM(s), but exhibits little-to-no transfer to any other VLMs; transfer is not affected by whether the attacked and target VLMs possess matching vision backbones or language models, whether the language model underwent instruction-following and/or safety-alignment training, or many other factors. Only two settings display partially successful transfer: between identically-pretrained and identically-initialized VLMs with slightly different VLM training data, and between different training checkpoints of a single VLM. Leveraging these results, we then demonstrate that transfer can be significantly improved against a specific target VLM by attacking larger ensembles of ``highly-similar" VLMs. These results stand in stark contrast to existing evidence of universal and transferable text jailbreaks against language models and transferable adversarial attacks against image classifiers, suggesting that VLMs may be more robust to gradient-based transfer attacks.

摘要: 将新的模式集成到前沿人工智能系统中提供了令人兴奋的能力，但也增加了此类系统被以不受欢迎的方式进行相反操作的可能性。在这项工作中，我们专注于一类流行的视觉语言模型(VLM)，它们生成以视觉和文本输入为条件的文本输出。我们进行了一项大规模的实证研究，以评估基于梯度的通用图像“越狱”的可转移性，使用了40多个开放参数VLM的不同集合，其中包括我们公开发布的18个新的VLM。总体而言，我们发现基于梯度的可转移越狱图像非常难以获得。当针对单个VLM或一组VLM优化图像越狱时，越狱成功地越狱了被攻击的VLM(S)，但很少或根本不转移到任何其他VLM；转移不受攻击和目标VLM是否具有匹配的视觉主干或语言模型、语言模型是否经过指令遵循和/或安全对齐培训或许多其他因素的影响。只有两个设置显示部分成功的传输：在具有略微不同的VLM训练数据的相同预训练和相同初始化的VLM之间，以及在单个VLM的不同训练检查点之间。利用这些结果，我们随后证明了针对特定目标VLm的转移可以通过攻击更大的“高度相似的”VLM集合来显著改善。这些结果与针对语言模型的普遍和可传输的文本越狱以及针对图像分类器的可传输的对抗性攻击的现有证据形成了鲜明对比，这表明VLM可能对基于梯度的传输攻击更健壮。



## **34. Dissecting Adversarial Robustness of Multimodal LM Agents**

剖析多模式LM代理的对抗鲁棒性 cs.LG

Oral presentation at NeurIPS 2024 Open-World Agents Workshop

**SubmitDate**: 2024-12-16    [abs](http://arxiv.org/abs/2406.12814v2) [paper-pdf](http://arxiv.org/pdf/2406.12814v2)

**Authors**: Chen Henry Wu, Rishi Shah, Jing Yu Koh, Ruslan Salakhutdinov, Daniel Fried, Aditi Raghunathan

**Abstract**: As language models (LMs) are used to build autonomous agents in real environments, ensuring their adversarial robustness becomes a critical challenge. Unlike chatbots, agents are compound systems with multiple components, which existing LM safety evaluations do not adequately address. To bridge this gap, we manually create 200 targeted adversarial tasks and evaluation functions in a realistic threat model on top of VisualWebArena, a real environment for web-based agents. In order to systematically examine the robustness of various multimodal we agents, we propose the Agent Robustness Evaluation (ARE) framework. ARE views the agent as a graph showing the flow of intermediate outputs between components and decomposes robustness as the flow of adversarial information on the graph. First, we find that we can successfully break a range of the latest agents that use black-box frontier LLMs, including those that perform reflection and tree-search. With imperceptible perturbations to a single product image (less than 5% of total web page pixels), an attacker can hijack these agents to execute targeted adversarial goals with success rates up to 67%. We also use ARE to rigorously evaluate how the robustness changes as new components are added. We find that new components that typically improve benign performance can open up new vulnerabilities and harm robustness. An attacker can compromise the evaluator used by the reflexion agent and the value function of the tree search agent, which increases the attack success relatively by 15% and 20%. Our data and code for attacks, defenses, and evaluation are available at https://github.com/ChenWu98/agent-attack

摘要: 随着语言模型(LMS)被用来在真实环境中构建自治代理，确保它们的对抗健壮性成为一个关键挑战。与聊天机器人不同，代理是具有多个组件的复合系统，现有的LM安全评估不能充分解决这一问题。为了弥合这一差距，我们在基于Web的代理的真实环境VisualWebArena的基础上，在现实威胁模型中手动创建了200个有针对性的对抗性任务和评估函数。为了系统地考察各种多通道WE智能体的健壮性，我们提出了智能体健壮性评估(ARE)框架。ARE将代理视为显示组件之间的中间输出流的图，并将健壮性分解为图上的对抗性信息流。首先，我们发现我们可以成功地破解一系列使用黑箱边界LLM的最新代理，包括那些执行反射和树搜索的代理。通过对单个产品图像的不可察觉的干扰(不到网页总像素的5%)，攻击者可以劫持这些代理以执行目标明确的对抗性目标，成功率高达67%。我们还使用ARS来严格评估在添加新组件时健壮性如何变化。我们发现，通常可以提高良性性能的新组件可能会打开新的漏洞，损害健壮性。攻击者可以折衷反射代理使用的赋值器和树搜索代理的值函数，从而使攻击成功率相对提高15%和20%。我们用于攻击、防御和评估的数据和代码可在https://github.com/ChenWu98/agent-attack获得



## **35. Finding a Wolf in Sheep's Clothing: Combating Adversarial Text-To-Image Prompts with Text Summarization**

披着羊皮找狼：用文本摘要对抗对抗文本到图像的对抗性冲突 cs.CR

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.12212v1) [paper-pdf](http://arxiv.org/pdf/2412.12212v1)

**Authors**: Portia Cooper, Harshita Narnoli, Mihai Surdeanu

**Abstract**: Text-to-image models are vulnerable to the stepwise "Divide-and-Conquer Attack" (DACA) that utilize a large language model to obfuscate inappropriate content in prompts by wrapping sensitive text in a benign narrative. To mitigate stepwise DACA attacks, we propose a two-layer method involving text summarization followed by binary classification. We assembled the Adversarial Text-to-Image Prompt (ATTIP) dataset ($N=940$), which contained DACA-obfuscated and non-obfuscated prompts. From the ATTIP dataset, we created two summarized versions: one generated by a small encoder model and the other by a large language model. Then, we used an encoder classifier and a GPT-4o classifier to perform content moderation on the summarized and unsummarized prompts. When compared with a classifier that operated over the unsummarized data, our method improved F1 score performance by 31%. Further, the highest recorded F1 score achieved (98%) was produced by the encoder classifier on a summarized ATTIP variant. This study indicates that pre-classification text summarization can inoculate content detection models against stepwise DACA obfuscations.

摘要: 文本到图像模型容易受到分而治之攻击(DACA)的攻击，这种攻击利用大型语言模型通过将敏感文本包装在良性叙事中来混淆提示中的不适当内容。为了缓解分步式DACA攻击，我们提出了一种文本摘要和二进制分类相结合的两层方法。我们汇编了对抗性文本到图像提示(ATTIP)数据集($N=940$)，其中包含DACA模糊和非模糊提示。从ATTIP数据集，我们创建了两个汇总版本：一个由小型编码器模型生成，另一个由大型语言模型生成。然后，我们使用编码器分类器和GPT-40分类器对摘要和未摘要的提示进行内容审核。与处理未汇总数据的分类器相比，我们的方法将F1得分性能提高了31%。此外，获得的最高记录F1分数(98%)是由编码员在总结的ATTIP变体上产生的。这项研究表明，预分类文本摘要可以为内容检测模型接种针对逐步DACA混淆的疫苗。



## **36. Unpacking the Resilience of SNLI Contradiction Examples to Attacks**

解开SNLI矛盾示例对攻击的弹性 cs.CL

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.11172v1) [paper-pdf](http://arxiv.org/pdf/2412.11172v1)

**Authors**: Chetan Verma, Archit Agarwal

**Abstract**: Pre-trained models excel on NLI benchmarks like SNLI and MultiNLI, but their true language understanding remains uncertain. Models trained only on hypotheses and labels achieve high accuracy, indicating reliance on dataset biases and spurious correlations. To explore this issue, we applied the Universal Adversarial Attack to examine the model's vulnerabilities. Our analysis revealed substantial drops in accuracy for the entailment and neutral classes, whereas the contradiction class exhibited a smaller decline. Fine-tuning the model on an augmented dataset with adversarial examples restored its performance to near-baseline levels for both the standard and challenge sets. Our findings highlight the value of adversarial triggers in identifying spurious correlations and improving robustness while providing insights into the resilience of the contradiction class to adversarial attacks.

摘要: 预训练的模型在SNLI和MultiNLI等NLI基准上表现出色，但它们真正的语言理解仍然不确定。仅根据假设和标签训练的模型可以实现高准确性，这表明依赖于数据集偏差和虚假相关性。为了探索这个问题，我们应用了通用对抗攻击来检查该模型的漏洞。我们的分析显示，蕴含类和中立类的准确性大幅下降，而矛盾类的准确性下降较小。在具有对抗性示例的增强数据集上对模型进行微调，将其性能恢复到标准集和挑战集的接近基线水平。我们的研究结果强调了对抗触发器在识别虚假相关性和提高稳健性方面的价值，同时深入了解矛盾类对对抗攻击的弹性。



## **37. PGD-Imp: Rethinking and Unleashing Potential of Classic PGD with Dual Strategies for Imperceptible Adversarial Attacks**

PGD-Imp：通过双重策略重新思考和释放经典PVD的潜力，以应对难以感知的对抗攻击 cs.LG

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.11168v1) [paper-pdf](http://arxiv.org/pdf/2412.11168v1)

**Authors**: Jin Li, Zitong Yu, Ziqiang He, Z. Jane Wang, Xiangui Kang

**Abstract**: Imperceptible adversarial attacks have recently attracted increasing research interests. Existing methods typically incorporate external modules or loss terms other than a simple $l_p$-norm into the attack process to achieve imperceptibility, while we argue that such additional designs may not be necessary. In this paper, we rethink the essence of imperceptible attacks and propose two simple yet effective strategies to unleash the potential of PGD, the common and classical attack, for imperceptibility from an optimization perspective. Specifically, the Dynamic Step Size is introduced to find the optimal solution with minimal attack cost towards the decision boundary of the attacked model, and the Adaptive Early Stop strategy is adopted to reduce the redundant strength of adversarial perturbations to the minimum level. The proposed PGD-Imperceptible (PGD-Imp) attack achieves state-of-the-art results in imperceptible adversarial attacks for both untargeted and targeted scenarios. When performing untargeted attacks against ResNet-50, PGD-Imp attains 100$\%$ (+0.3$\%$) ASR, 0.89 (-1.76) $l_2$ distance, and 52.93 (+9.2) PSNR with 57s (-371s) running time, significantly outperforming existing methods.

摘要: 潜伏的敌意攻击最近吸引了越来越多的研究兴趣。现有的方法通常在攻击过程中加入外部模块或损失项，而不是简单的$L_p$-范数来实现不可感知性，而我们认为这样的额外设计可能不是必要的。本文从优化的角度重新思考了不可察觉攻击的本质，并提出了两种简单而有效的策略来释放PGD攻击--普通攻击和经典攻击--的不可感知性。具体地，引入动态步长在攻击模型的决策边界附近寻找攻击代价最小的最优解，并采用自适应提前停止策略将敌方扰动的冗余强度降至最小。建议的PGD-Imp(PGD-Imp)攻击在非目标场景和目标场景中都实现了最先进的不可感知对手攻击。在对ResNet-50进行非定向攻击时，PGD-Imp在57s(-371s)的运行时间内获得了100$(+0.3$)ASR，0.89(-1.76)$L_2$距离和52.93(+9.2)PSNR，显著优于现有方法。



## **38. The Superalignment of Superhuman Intelligence with Large Language Models**

超人智能与大型语言模型的超级对齐 cs.CL

Under review of Science China

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.11145v1) [paper-pdf](http://arxiv.org/pdf/2412.11145v1)

**Authors**: Minlie Huang, Yingkang Wang, Shiyao Cui, Pei Ke, Jie Tang

**Abstract**: We have witnessed superhuman intelligence thanks to the fast development of large language models and multimodal language models. As the application of such superhuman models becomes more and more common, a critical question rises here: how can we ensure superhuman models are still safe, reliable and aligned well to human values? In this position paper, we discuss the concept of superalignment from the learning perspective to answer this question by outlining the learning paradigm shift from large-scale pretraining, supervised fine-tuning, to alignment training. We define superalignment as designing effective and efficient alignment algorithms to learn from noisy-labeled data (point-wise samples or pair-wise preference data) in a scalable way when the task becomes very complex for human experts to annotate and the model is stronger than human experts. We highlight some key research problems in superalignment, namely, weak-to-strong generalization, scalable oversight, and evaluation. We then present a conceptual framework for superalignment, which consists of three modules: an attacker which generates adversary queries trying to expose the weaknesses of a learner model; a learner which will refine itself by learning from scalable feedbacks generated by a critic model along with minimal human experts; and a critic which generates critics or explanations for a given query-response pair, with a target of improving the learner by criticizing. We discuss some important research problems in each component of this framework and highlight some interesting research ideas that are closely related to our proposed framework, for instance, self-alignment, self-play, self-refinement, and more. Last, we highlight some future research directions for superalignment, including identification of new emergent risks and multi-dimensional alignment.

摘要: 由于大型语言模型和多模式语言模型的快速发展，我们见证了超人的智能。随着这种超人模型的应用变得越来越普遍，一个关键的问题出现了：我们如何确保超人模型仍然安全、可靠，并与人类的价值观保持良好一致？在这份立场文件中，我们从学习的角度讨论了超匹配的概念，通过概述学习范式从大规模预训练、有监督的微调到对齐训练的转变来回答这个问题。我们将超比对定义为设计有效和高效的比对算法，当任务对于人类专家来说变得非常复杂并且模型比人类专家更强时，以可扩展的方式从噪声标记的数据(点状样本或成对偏好数据)中学习。我们强调了超比对中的一些关键研究问题，即从弱到强的泛化、可扩展的监督和评估。然后，我们提出了一个超对齐的概念框架，它由三个模块组成：攻击者，生成敌意查询，试图揭露学习者模型的弱点；学习者，将通过与最少的人类专家一起从批评者模型生成的可伸缩反馈中学习来改进自己；批评者，为给定的查询-响应对生成批评者或解释，目标是通过批评来改进学习者。我们讨论了该框架每个组成部分中的一些重要研究问题，并突出了与我们提出的框架密切相关的一些有趣的研究想法，例如自我调整、自我发挥、自我完善等。最后，我们指出了超配准未来的研究方向，包括识别新出现的风险和多维配对。



## **39. Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models**

通过扩散模型高效生成视觉语言模型的有针对性且可转移的对抗示例 cs.CV

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2404.10335v4) [paper-pdf](http://arxiv.org/pdf/2404.10335v4)

**Authors**: Qi Guo, Shanmin Pang, Xiaojun Jia, Yang Liu, Qing Guo

**Abstract**: Adversarial attacks, particularly \textbf{targeted} transfer-based attacks, can be used to assess the adversarial robustness of large visual-language models (VLMs), allowing for a more thorough examination of potential security flaws before deployment. However, previous transfer-based adversarial attacks incur high costs due to high iteration counts and complex method structure. Furthermore, due to the unnaturalness of adversarial semantics, the generated adversarial examples have low transferability. These issues limit the utility of existing methods for assessing robustness. To address these issues, we propose AdvDiffVLM, which uses diffusion models to generate natural, unrestricted and targeted adversarial examples via score matching. Specifically, AdvDiffVLM uses Adaptive Ensemble Gradient Estimation to modify the score during the diffusion model's reverse generation process, ensuring that the produced adversarial examples have natural adversarial targeted semantics, which improves their transferability. Simultaneously, to improve the quality of adversarial examples, we use the GradCAM-guided Mask method to disperse adversarial semantics throughout the image rather than concentrating them in a single area. Finally, AdvDiffVLM embeds more target semantics into adversarial examples after multiple iterations. Experimental results show that our method generates adversarial examples 5x to 10x faster than state-of-the-art transfer-based adversarial attacks while maintaining higher quality adversarial examples. Furthermore, compared to previous transfer-based adversarial attacks, the adversarial examples generated by our method have better transferability. Notably, AdvDiffVLM can successfully attack a variety of commercial VLMs in a black-box environment, including GPT-4V.

摘要: 对抗性攻击，特别是基于传输的对抗性攻击，可用于评估大型视觉语言模型(VLM)的对抗性健壮性，从而允许在部署之前更彻底地检查潜在的安全漏洞。然而，以往基于转移的对抗性攻击由于迭代次数多、方法结构复杂，代价较高。此外，由于对抗性语义的非自然性，生成的对抗性实例可转移性较低。这些问题限制了现有稳健性评估方法的实用性。为了解决这些问题，我们提出了AdvDiffVLM，它使用扩散模型通过得分匹配来生成自然的、不受限制的和有针对性的对抗性实例。具体地说，AdvDiffVLM在扩散模型的反向生成过程中使用自适应集成梯度估计来修改分数，确保生成的对抗性实例具有自然对抗性目标语义，从而提高了它们的可转移性。同时，为了提高对抗性实例的质量，我们使用了GradCAM引导的掩码方法，将对抗性语义分散在整个图像中，而不是将它们集中在单个区域。最后，在多次迭代后，AdvDiffVLM将更多的目标语义嵌入到对抗性实例中。实验结果表明，在保持较高质量的对抗性实例的同时，我们的方法生成对抗性实例的速度比最新的基于传输的对抗性攻击快5倍到10倍。此外，与以往基于转移的对抗性攻击相比，该方法生成的对抗性实例具有更好的可转移性。值得注意的是，AdvDiffVLM可以在黑盒环境中成功攻击各种商业VLM，包括GPT-4V。



## **40. Impact of Adversarial Attacks on Deep Learning Model Explainability**

对抗性攻击对深度学习模型解释性的影响 cs.LG

29 pages with reference included, submitted to a journal

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.11119v1) [paper-pdf](http://arxiv.org/pdf/2412.11119v1)

**Authors**: Gazi Nazia Nur, Mohammad Ahnaf Sadat

**Abstract**: In this paper, we investigate the impact of adversarial attacks on the explainability of deep learning models, which are commonly criticized for their black-box nature despite their capacity for autonomous feature extraction. This black-box nature can affect the perceived trustworthiness of these models. To address this, explainability techniques such as GradCAM, SmoothGrad, and LIME have been developed to clarify model decision-making processes. Our research focuses on the robustness of these explanations when models are subjected to adversarial attacks, specifically those involving subtle image perturbations that are imperceptible to humans but can significantly mislead models. For this, we utilize attack methods like the Fast Gradient Sign Method (FGSM) and the Basic Iterative Method (BIM) and observe their effects on model accuracy and explanations. The results reveal a substantial decline in model accuracy, with accuracies dropping from 89.94% to 58.73% and 45.50% under FGSM and BIM attacks, respectively. Despite these declines in accuracy, the explanation of the models measured by metrics such as Intersection over Union (IoU) and Root Mean Square Error (RMSE) shows negligible changes, suggesting that these metrics may not be sensitive enough to detect the presence of adversarial perturbations.

摘要: 在本文中，我们研究了对抗性攻击对深度学习模型可解释性的影响，尽管深度学习模型具有自主特征提取的能力，但它们通常因其黑箱性质而受到批评。这种黑箱性质可能会影响这些模型的可信度。为了解决这个问题，已经开发了可解释性技术，如GradCAM、SmoothGrad和LIME，以澄清模型决策过程。我们的研究重点是当模型受到敌意攻击时，这些解释的稳健性，特别是那些涉及人类无法察觉但可能显著误导模型的微妙图像扰动。为此，我们使用了快速梯度符号方法(FGSM)和基本迭代方法(BIM)等攻击方法，并观察了它们对模型精度和解释的影响。结果表明，模型的准确率大幅下降，在FGSM和BIM攻击下，准确率分别从89.94%下降到58.73%和45.50%。尽管准确率有所下降，但通过联合交集(IOU)和均方根误差(RMSE)等指标衡量的模型的解释显示出可以忽略不计的变化，这表明这些指标可能不够敏感，无法检测到对抗性扰动的存在。



## **41. Simulate and Eliminate: Revoke Backdoors for Generative Large Language Models**

模拟和消除：撤销生成性大型语言模型的后门 cs.CR

To appear at AAAI 2025

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2405.07667v2) [paper-pdf](http://arxiv.org/pdf/2405.07667v2)

**Authors**: Haoran Li, Yulin Chen, Zihao Zheng, Qi Hu, Chunkit Chan, Heshan Liu, Yangqiu Song

**Abstract**: With rapid advances, generative large language models (LLMs) dominate various Natural Language Processing (NLP) tasks from understanding to reasoning. Yet, language models' inherent vulnerabilities may be exacerbated due to increased accessibility and unrestricted model training on massive data. A malicious adversary may publish poisoned data online and conduct backdoor attacks on the victim LLMs pre-trained on the poisoned data. Backdoored LLMs behave innocuously for normal queries and generate harmful responses when the backdoor trigger is activated. Despite significant efforts paid to LLMs' safety issues, LLMs are still struggling against backdoor attacks. As Anthropic recently revealed, existing safety training strategies, including supervised fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), fail to revoke the backdoors once the LLM is backdoored during the pre-training stage. In this paper, we present Simulate and Eliminate (SANDE) to erase the undesired backdoored mappings for generative LLMs. We initially propose Overwrite Supervised Fine-tuning (OSFT) for effective backdoor removal when the trigger is known. Then, to handle scenarios where trigger patterns are unknown, we integrate OSFT into our two-stage framework, SANDE. Unlike other works that assume access to cleanly trained models, our safety-enhanced LLMs are able to revoke backdoors without any reference. Consequently, our safety-enhanced LLMs no longer produce targeted responses when the backdoor triggers are activated. We conduct comprehensive experiments to show that our proposed SANDE is effective against backdoor attacks while bringing minimal harm to LLMs' powerful capability.

摘要: 随着研究的深入，从理解到推理的各种自然语言处理任务都被生成性大语言模型(LLMS)所支配。然而，语言模型的固有脆弱性可能会由于海量数据的可获得性增加和不受限制的模型训练而加剧。恶意对手可能会在网上发布有毒数据，并对受害者LLM进行后门攻击，这些LLM预先训练了有毒数据。后门LLM在正常查询中的行为是无害的，并在激活后门触发器时生成有害的响应。尽管在LLMS的安全问题上付出了巨大的努力，但LLMS仍在努力应对后门攻击。正如人类最近揭示的那样，现有的安全培训策略，包括监督微调(SFT)和从人类反馈的强化学习(RLHF)，一旦LLM在培训前阶段后退，就无法取消后门。在这篇文章中，我们提出了模拟和消除(SANDE)来消除生成式LLMS中不需要的回溯映射。我们最初提出了覆盖监督精调(OSFT)，用于在已知触发器的情况下有效地删除后门。然后，为了处理触发模式未知的场景，我们将OSFT集成到我们的两阶段框架Sande中。与其他假定可以访问训练有素的模型的作品不同，我们的安全增强型LLM能够在没有任何参考的情况下撤销后门。因此，当后门触发器被激活时，我们的安全增强型LLMS不再产生目标响应。我们进行了全面的实验，以表明我们提出的SANDE能够有效地抵抗后门攻击，同时对LLMS的强大能力造成的损害最小。



## **42. Learning Robust and Privacy-Preserving Representations via Information Theory**

通过信息理论学习稳健且隐私保护的表示 cs.LG

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2412.11066v1) [paper-pdf](http://arxiv.org/pdf/2412.11066v1)

**Authors**: Binghui Zhang, Sayedeh Leila Noorbakhsh, Yun Dong, Yuan Hong, Binghui Wang

**Abstract**: Machine learning models are vulnerable to both security attacks (e.g., adversarial examples) and privacy attacks (e.g., private attribute inference). We take the first step to mitigate both the security and privacy attacks, and maintain task utility as well. Particularly, we propose an information-theoretic framework to achieve the goals through the lens of representation learning, i.e., learning representations that are robust to both adversarial examples and attribute inference adversaries. We also derive novel theoretical results under our framework, e.g., the inherent trade-off between adversarial robustness/utility and attribute privacy, and guaranteed attribute privacy leakage against attribute inference adversaries.

摘要: 机器学习模型容易受到这两种安全攻击（例如，对抗性例子）和隐私攻击（例如，私人属性推断）。我们迈出的第一步是减轻安全和隐私攻击，并维护任务实用性。特别是，我们提出了一个信息论框架来通过表示学习的视角实现目标，即学习对对抗性示例和属性推断对手都稳健的表示。我们还在我们的框架下得出了新颖的理论结果，例如，对抗稳健性/效用和属性隐私之间的固有权衡，以及针对属性推断对手的保证属性隐私泄露。



## **43. HTS-Attack: Heuristic Token Search for Jailbreaking Text-to-Image Models**

HTS-Attack：启发式代币搜索越狱的文本到图像模型 cs.CV

**SubmitDate**: 2024-12-15    [abs](http://arxiv.org/abs/2408.13896v3) [paper-pdf](http://arxiv.org/pdf/2408.13896v3)

**Authors**: Sensen Gao, Xiaojun Jia, Yihao Huang, Ranjie Duan, Jindong Gu, Yang Bai, Yang Liu, Qing Guo

**Abstract**: Text-to-Image(T2I) models have achieved remarkable success in image generation and editing, yet these models still have many potential issues, particularly in generating inappropriate or Not-Safe-For-Work(NSFW) content. Strengthening attacks and uncovering such vulnerabilities can advance the development of reliable and practical T2I models. Most of the previous works treat T2I models as white-box systems, using gradient optimization to generate adversarial prompts. However, accessing the model's gradient is often impossible in real-world scenarios. Moreover, existing defense methods, those using gradient masking, are designed to prevent attackers from obtaining accurate gradient information. While several black-box jailbreak attacks have been explored, they achieve the limited performance of jailbreaking T2I models due to difficulties associated with optimization in discrete spaces. To address this, we propose HTS-Attack, a heuristic token search attack method. HTS-Attack begins with an initialization that removes sensitive tokens, followed by a heuristic search where high-performing candidates are recombined and mutated. This process generates a new pool of candidates, and the optimal adversarial prompt is updated based on their effectiveness. By incorporating both optimal and suboptimal candidates, HTS-Attack avoids local optima and improves robustness in bypassing defenses. Extensive experiments validate the effectiveness of our method in attacking the latest prompt checkers, post-hoc image checkers, securely trained T2I models, and online commercial models.

摘要: 文本到图像(T2I)模型在图像生成和编辑方面取得了显著的成功，但这些模型仍然存在许多潜在的问题，特别是在生成不适当或不安全的工作内容(NSFW)方面。加强攻击并发现此类漏洞可以促进可靠和实用的T2I模型的发展。以前的工作大多将T2I模型视为白盒系统，使用梯度优化来生成对抗性提示。然而，在现实世界的场景中，访问模型的渐变通常是不可能的。此外，现有的防御方法，即使用梯度掩码的方法，旨在防止攻击者获得准确的梯度信息。虽然已经探索了几种黑盒越狱攻击，但由于与离散空间中的优化相关的困难，它们实现了越狱T2I模型的有限性能。针对这一问题，我们提出了一种启发式令牌搜索攻击方法HTS-Attack。HTS攻击以删除敏感令牌的初始化开始，然后进行启发式搜索，对高性能的候选进行重组和变异。这个过程产生一个新的候选者池，并根据他们的有效性更新最优的对抗性提示。通过结合最优和次优候选，HTS-Attack避免了局部最优，并提高了绕过防御的健壮性。大量的实验验证了该方法在攻击最新的提示检查器、后自组织图像检查器、安全训练的T2I模型和在线商业模型方面的有效性。



## **44. Identification of Path Congestion Status for Network Performance Tomography using Deep Spatial-Temporal Learning**

使用深度时空学习识别网络性能断层扫描的路径拥挤状态 cs.NI

**SubmitDate**: 2024-12-14    [abs](http://arxiv.org/abs/2412.10762v1) [paper-pdf](http://arxiv.org/pdf/2412.10762v1)

**Authors**: Chengze Du, Zhiwei Yu, Xiangyu Wang

**Abstract**: Network tomography plays a crucial role in assessing the operational status of internal links within networks through end-to-end path-level measurements, independently of cooperation from the network infrastructure. However, the accuracy of performance inference in internal network links heavily relies on comprehensive end-to-end path performance data. Most network tomography algorithms employ conventional threshold-based methods to identify congestion along paths, while these methods encounter limitations stemming from network complexities, resulting in inaccuracies such as misidentifying abnormal links and overlooking congestion attacks, thereby impeding algorithm performance. This paper introduces the concept of Additive Congestion Status to address these challenges effectively. Using a framework that combines Adversarial Autoencoders (AAE) with Long Short-Term Memory (LSTM) networks, this approach robustly categorizes (as uncongested, single-congested, or multiple-congested) and quantifies (regarding the number of congested links) the Additive Congestion Status. Leveraging prior path information and capturing spatio-temporal characteristics of probing flows, this method significantly enhances the localization of congested links and the inference of link performance compared to conventional network tomography algorithms, as demonstrated through experimental evaluations.

摘要: 网络断层成像在通过端到端路径级测量来评估网络内内部链路的运行状态方面发挥着至关重要的作用，独立于网络基础设施的合作。然而，内部网络链路中性能推断的准确性在很大程度上依赖于全面的端到端路径性能数据。大多数网络层析成像算法使用传统的基于阈值的方法来识别路径上的拥塞，而这些方法由于网络的复杂性而受到限制，导致错误识别异常链路和忽略拥塞攻击等不准确的情况，从而影响算法的性能。为了有效地应对这些挑战，本文引入了加性拥塞状态的概念。使用对抗性自动编码器(AAE)和长短期记忆(LSTM)网络相结合的框架，该方法稳健地分类(非拥塞、单拥塞或多拥塞)并量化(关于拥塞链路的数量)加性拥塞状态。实验结果表明，与传统的网络层析成像算法相比，该方法利用先验路径信息并捕获探测流的时空特征，显著提高了拥塞链路的定位和链路性能的推断。



## **45. On Effects of Steering Latent Representation for Large Language Model Unlearning**

论引导潜在表示对大型语言模型取消学习的影响 cs.CL

Accepted at AAAI-25 Main Technical Track

**SubmitDate**: 2024-12-14    [abs](http://arxiv.org/abs/2408.06223v2) [paper-pdf](http://arxiv.org/pdf/2408.06223v2)

**Authors**: Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue

**Abstract**: Representation Misdirection for Unlearning (RMU), which steers model representation in the intermediate layer to a target random representation, is an effective method for large language model (LLM) unlearning. Despite its high performance, the underlying cause and explanation remain underexplored. In this paper, we theoretically demonstrate that steering forget representations in the intermediate layer reduces token confidence, causing LLMs to generate wrong or nonsense responses. We investigate how the coefficient influences the alignment of forget-sample representations with the random direction and hint at the optimal coefficient values for effective unlearning across different network layers. We show that RMU unlearned models are robust against adversarial jailbreak attacks. Furthermore, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs. To resolve this drawback, we propose Adaptive RMU -- a simple yet effective alternative method that makes unlearning effective with most layers. Extensive experiments demonstrate that Adaptive RMU significantly improves the unlearning performance compared to prior art while incurring no additional computational cost.

摘要: 遗忘表征误导(RMU)是一种有效的大语言模型遗忘方法，它将中间层的模型表征引导到目标随机表征。尽管其表现良好，但其根本原因和解释仍未得到充分研究。在本文中，我们从理论上证明了中间层中的转向遗忘表征降低了令牌置信度，从而导致LLM产生错误或无意义的响应。我们研究了系数如何影响遗忘样本表示与随机方向的对齐，并提示了跨不同网络层有效遗忘的最优系数值。我们证明了RMU未学习模型对敌意越狱攻击是健壮的。此外，我们的实证分析表明，当RMU应用于LLMS的中后期时，其有效性较差。为了解决这一缺陷，我们提出了自适应RMU--一种简单但有效的替代方法，使遗忘在大多数层都有效。大量实验表明，与现有技术相比，自适应RMU在不增加额外计算代价的情况下，显著改善了遗忘性能。



## **46. RAT: Adversarial Attacks on Deep Reinforcement Agents for Targeted Behaviors**

RAT：针对目标行为的深度强化代理的对抗性攻击 cs.LG

Accepted by AAAI 2025

**SubmitDate**: 2024-12-14    [abs](http://arxiv.org/abs/2412.10713v1) [paper-pdf](http://arxiv.org/pdf/2412.10713v1)

**Authors**: Fengshuo Bai, Runze Liu, Yali Du, Ying Wen, Yaodong Yang

**Abstract**: Evaluating deep reinforcement learning (DRL) agents against targeted behavior attacks is critical for assessing their robustness. These attacks aim to manipulate the victim into specific behaviors that align with the attacker's objectives, often bypassing traditional reward-based defenses. Prior methods have primarily focused on reducing cumulative rewards; however, rewards are typically too generic to capture complex safety requirements effectively. As a result, focusing solely on reward reduction can lead to suboptimal attack strategies, particularly in safety-critical scenarios where more precise behavior manipulation is needed. To address these challenges, we propose RAT, a method designed for universal, targeted behavior attacks. RAT trains an intention policy that is explicitly aligned with human preferences, serving as a precise behavioral target for the adversary. Concurrently, an adversary manipulates the victim's policy to follow this target behavior. To enhance the effectiveness of these attacks, RAT dynamically adjusts the state occupancy measure within the replay buffer, allowing for more controlled and effective behavior manipulation. Our empirical results on robotic simulation tasks demonstrate that RAT outperforms existing adversarial attack algorithms in inducing specific behaviors. Additionally, RAT shows promise in improving agent robustness, leading to more resilient policies. We further validate RAT by guiding Decision Transformer agents to adopt behaviors aligned with human preferences in various MuJoCo tasks, demonstrating its effectiveness across diverse tasks.

摘要: 评估深度强化学习(DRL)代理抵抗目标行为攻击是评估其稳健性的关键。这些攻击旨在操纵受害者做出与攻击者目标一致的特定行为，通常绕过传统的基于奖励的防御。以前的方法主要集中在减少累积奖励；然而，奖励通常过于笼统，无法有效地满足复杂的安全要求。因此，只关注奖励减少可能会导致次优攻击策略，特别是在需要更精确的行为控制的安全关键场景中。为了应对这些挑战，我们提出了RAT，这是一种为通用的、有针对性的行为攻击而设计的方法。RAT训练一种明确与人类偏好相一致的意图策略，作为对手的精确行为目标。同时，敌手操纵受害者的策略以遵循此目标行为。为了增强这些攻击的有效性，RAT动态调整重放缓冲区内的状态占用度量，从而允许更可控和有效的行为操作。我们在机器人模拟任务上的实验结果表明，RAT在诱导特定行为方面优于现有的对抗性攻击算法。此外，RAT在改善代理健壮性方面显示出希望，从而导致更具弹性的策略。我们通过指导决策转换器代理在各种MuJoCo任务中采用符合人类偏好的行为来进一步验证RAT，展示了它在不同任务中的有效性。



## **47. BinarySelect to Improve Accessibility of Black-Box Attack Research**

Binaryselect提高黑匣子攻击研究的可访问性 cs.CR

Accepted to COLING 2025, 17 pages, 5 figures, 11 tables

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2412.10617v1) [paper-pdf](http://arxiv.org/pdf/2412.10617v1)

**Authors**: Shatarupa Ghosh, Jonathan Rusert

**Abstract**: Adversarial text attack research is useful for testing the robustness of NLP models, however, the rise of transformers has greatly increased the time required to test attacks. Especially when researchers do not have access to adequate resources (e.g. GPUs). This can hinder attack research, as modifying one example for an attack can require hundreds of queries to a model, especially for black-box attacks. Often these attacks remove one token at a time to find the ideal one to change, requiring $n$ queries (the length of the text) right away. We propose a more efficient selection method called BinarySelect which combines binary search and attack selection methods to greatly reduce the number of queries needed to find a token. We find that BinarySelect only needs $\text{log}_2(n) * 2$ queries to find the first token compared to $n$ queries. We also test BinarySelect in an attack setting against 5 classifiers across 3 datasets and find a viable tradeoff between number of queries saved and attack effectiveness. For example, on the Yelp dataset, the number of queries is reduced by 32% (72 less) with a drop in attack effectiveness of only 5 points. We believe that BinarySelect can help future researchers study adversarial attacks and black-box problems more efficiently and opens the door for researchers with access to less resources.

摘要: 对抗性文本攻击的研究对于测试NLP模型的稳健性是有用的，然而，转换器的兴起大大增加了测试攻击所需的时间。特别是当研究人员无法获得足够的资源(例如图形处理器)时。这可能会阻碍攻击研究，因为为攻击修改一个示例可能需要对一个模型进行数百次查询，特别是对于黑盒攻击。通常，这些攻击一次删除一个令牌，以找到要更改的理想令牌，需要立即执行$n$查询(文本的长度)。我们提出了一种更有效的选择方法BinarySelect，它结合了二进制搜索和攻击选择方法，大大减少了查找令牌所需的查询次数。我们发现BinarySelect只需要$\Text{log}_2(N)*2$查询就可以找到第一个令牌，而不是$n$查询。我们还针对3个数据集上的5个分类器在攻击环境中测试了BinarySelect，并在节省的查询数量和攻击效率之间找到了可行的折衷。例如，在Yelp数据集上，查询次数减少了32%(减少了72)，攻击效率仅下降了5个点。我们相信，BinarySelect可以帮助未来的研究人员更有效地研究对抗性攻击和黑盒问题，并为研究人员打开获取更少资源的大门。



## **48. Client-Side Patching against Backdoor Attacks in Federated Learning**

客户端修补联邦学习中的后门攻击 cs.CR

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2412.10605v1) [paper-pdf](http://arxiv.org/pdf/2412.10605v1)

**Authors**: Borja Molina Coronado

**Abstract**: Federated learning is a versatile framework for training models in decentralized environments. However, the trust placed in clients makes federated learning vulnerable to backdoor attacks launched by malicious participants. While many defenses have been proposed, they often fail short when facing heterogeneous data distributions among participating clients. In this paper, we propose a novel defense mechanism for federated learning systems designed to mitigate backdoor attacks on the clients-side. Our approach leverages adversarial learning techniques and model patching to neutralize the impact of backdoor attacks. Through extensive experiments on the MNIST and Fashion-MNIST datasets, we demonstrate that our defense effectively reduces backdoor accuracy, outperforming existing state-of-the-art defenses, such as LFighter, FLAME, and RoseAgg, in i.i.d. and non-i.i.d. scenarios, while maintaining competitive or superior accuracy on clean data.

摘要: 联邦学习是去中心化环境中训练模型的通用框架。然而，对客户的信任使得联邦学习容易受到恶意参与者发起的后门攻击。虽然已经提出了许多防御措施，但当面临参与客户端之间的异类数据分布时，它们往往会失败。在本文中，我们提出了一种新型的联邦学习系统防御机制，旨在减轻客户端的后门攻击。我们的方法利用对抗学习技术和模型修补来抵消后门攻击的影响。通过对MNIST和Fashion-MNIST数据集的广泛实验，我们证明我们的防御有效地降低了后门准确性，优于现有的最先进防御，例如LFighter、FLAME和RoseAgg，i.i. d。和非i.i.d.场景，同时在干净数据上保持有竞争力或卓越的准确性。



## **49. Crosstalk-induced Side Channel Threats in Multi-Tenant NISQ Computers**

多租户NISQ计算机中的串话引发的侧通道威胁 cs.ET

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2412.10507v1) [paper-pdf](http://arxiv.org/pdf/2412.10507v1)

**Authors**: Navnil Choudhury, Chaithanya Naik Mude, Sanjay Das, Preetham Chandra Tikkireddi, Swamit Tannu, Kanad Basu

**Abstract**: As quantum computing rapidly advances, its near-term applications are becoming increasingly evident. However, the high cost and under-utilization of quantum resources are prompting a shift from single-user to multi-user access models. In a multi-tenant environment, where multiple users share one quantum computer, protecting user confidentiality becomes crucial. The varied uses of quantum computers increase the risk that sensitive data encoded by one user could be compromised by others, rendering the protection of data integrity and confidentiality essential. In the evolving quantum computing landscape, it is imperative to study these security challenges within the scope of realistic threat model assumptions, wherein an adversarial user can mount practical attacks without relying on any heightened privileges afforded by physical access to a quantum computer or rogue cloud services. In this paper, we demonstrate the potential of crosstalk as an attack vector for the first time on a Noisy Intermediate Scale Quantum (NISQ) machine, that an adversarial user can exploit within a multi-tenant quantum computing model. The proposed side-channel attack is conducted with minimal and realistic adversarial privileges, with the overarching aim of uncovering the quantum algorithm being executed by a victim. Crosstalk signatures are used to estimate the presence of CNOT gates in the victim circuit, and subsequently, this information is encoded and classified by a graph-based learning model to identify the victim quantum algorithm. When evaluated on up to 336 benchmark circuits, our attack framework is found to be able to unveil the victim's quantum algorithm with up to 85.7\% accuracy.

摘要: 随着量子计算的快速发展，其近期应用正变得越来越明显。然而，量子资源的高成本和未充分利用正在推动从单用户访问模式向多用户访问模式的转变。在多租户环境中，多个用户共享一台量子计算机，保护用户机密性变得至关重要。量子计算机的各种用途增加了一个用户编码的敏感数据可能被其他用户泄露的风险，这使得保护数据完整性和机密性变得至关重要。在不断发展的量子计算环境中，必须在现实威胁模型假设的范围内研究这些安全挑战，在这种假设下，敌意用户可以发动实际攻击，而不需要依赖物理访问量子计算机或流氓云服务所提供的任何高级特权。在本文中，我们首次证明了串扰作为攻击向量的潜力，在噪声中的中间尺度量子(NISQ)机器上，敌对用户可以在多租户量子计算模型中利用该机器。拟议的旁路攻击是以最低限度和现实的对手特权进行的，首要目标是破解受害者正在执行的量子算法。串扰特征被用来估计受害者电路中CNOT门的存在，随后，该信息被基于图的学习模型编码和分类以识别受害者量子算法。当在多达336个基准电路上进行评估时，我们发现我们的攻击框架能够以高达85.7%的准确率揭示受害者的量子算法。



## **50. MOREL: Enhancing Adversarial Robustness through Multi-Objective Representation Learning**

MOREL：通过多目标表示学习增强对抗鲁棒性 cs.LG

**SubmitDate**: 2024-12-13    [abs](http://arxiv.org/abs/2410.01697v3) [paper-pdf](http://arxiv.org/pdf/2410.01697v3)

**Authors**: Sedjro Salomon Hotegni, Sebastian Peitz

**Abstract**: Extensive research has shown that deep neural networks (DNNs) are vulnerable to slight adversarial perturbations$-$small changes to the input data that appear insignificant but cause the model to produce drastically different outputs. In addition to augmenting training data with adversarial examples generated from a specific attack method, most of the current defense strategies necessitate modifying the original model architecture components to improve robustness or performing test-time data purification to handle adversarial attacks. In this work, we demonstrate that strong feature representation learning during training can significantly enhance the original model's robustness. We propose MOREL, a multi-objective feature representation learning approach, encouraging classification models to produce similar features for inputs within the same class, despite perturbations. Our training method involves an embedding space where cosine similarity loss and multi-positive contrastive loss are used to align natural and adversarial features from the model encoder and ensure tight clustering. Concurrently, the classifier is motivated to achieve accurate predictions. Through extensive experiments, we demonstrate that our approach significantly enhances the robustness of DNNs against white-box and black-box adversarial attacks, outperforming other methods that similarly require no architectural changes or test-time data purification. Our code is available at https://github.com/salomonhotegni/MOREL

摘要: 广泛的研究表明，深度神经网络(DNN)容易受到输入数据的微小对抗性扰动，这些微小的变化看起来微不足道，但会导致模型产生截然不同的输出。除了使用特定攻击方法生成的对抗性样本来扩充训练数据外，当前的大多数防御策略都需要修改原始模型体系结构组件以提高健壮性，或者执行测试时间数据净化来处理对抗性攻击。在这项工作中，我们证明了在训练过程中的强特征表示学习可以显著增强原始模型的稳健性。我们提出了MOREL，一种多目标特征表示学习方法，鼓励分类模型为同一类内的输入产生相似的特征，尽管存在扰动。我们的训练方法涉及一个嵌入空间，在该空间中使用余弦相似损失和多正对比损失来对齐模型编码器中的自然特征和对抗性特征，并确保紧密的聚类。同时，分类器的动机是实现准确的预测。通过大量的实验，我们证明了我们的方法显著提高了DNN对白盒和黑盒攻击的健壮性，优于其他同样不需要改变体系结构或净化测试时间数据的方法。我们的代码可以在https://github.com/salomonhotegni/MOREL上找到



