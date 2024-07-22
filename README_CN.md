# Latest Adversarial Attack Papers
**update at 2024-07-22 09:46:47**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Multi-Attribute Vision Transformers are Efficient and Robust Learners**

多属性视觉变形者是高效且稳健的学习者 cs.CV

Accepted at IEEE ICIP 2024. arXiv admin note: text overlap with  arXiv:2207.08677 by other authors

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2402.08070v2) [paper-pdf](http://arxiv.org/pdf/2402.08070v2)

**Authors**: Hanan Gani, Nada Saadi, Noor Hussein, Karthik Nandakumar

**Abstract**: Since their inception, Vision Transformers (ViTs) have emerged as a compelling alternative to Convolutional Neural Networks (CNNs) across a wide spectrum of tasks. ViTs exhibit notable characteristics, including global attention, resilience against occlusions, and adaptability to distribution shifts. One underexplored aspect of ViTs is their potential for multi-attribute learning, referring to their ability to simultaneously grasp multiple attribute-related tasks. In this paper, we delve into the multi-attribute learning capability of ViTs, presenting a straightforward yet effective strategy for training various attributes through a single ViT network as distinct tasks. We assess the resilience of multi-attribute ViTs against adversarial attacks and compare their performance against ViTs designed for single attributes. Moreover, we further evaluate the robustness of multi-attribute ViTs against a recent transformer based attack called Patch-Fool. Our empirical findings on the CelebA dataset provide validation for our assertion. Our code is available at https://github.com/hananshafi/MTL-ViT

摘要: 自诞生以来，视觉变压器(VITS)已经成为卷积神经网络(CNN)在广泛任务范围内的一种引人注目的替代方案。VITS表现出显著的特征，包括全球注意力、对闭塞的弹性和对分布变化的适应性。VITS的一个未被开发的方面是其多属性学习的潜力，指的是它们同时掌握多个与属性相关的任务的能力。在本文中，我们深入研究了VITS的多属性学习能力，提出了一种简单而有效的策略，通过单个VIT网络将各种属性作为不同的任务进行训练。我们评估了多属性VITS抵抗敌意攻击的能力，并与单属性VITS的性能进行了比较。此外，我们进一步评估了多属性VITS对最近一种称为Patch-Fool的基于变压器的攻击的健壮性。我们在CelebA数据集上的经验发现为我们的断言提供了验证。我们的代码可以在https://github.com/hananshafi/MTL-ViT上找到



## **2. SlowPerception: Physical-World Latency Attack against Visual Perception in Autonomous Driving**

慢感知：自动驾驶中对视觉感知的物理世界延迟攻击 cs.CV

This submission was made without all contributors' consent

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2406.05800v2) [paper-pdf](http://arxiv.org/pdf/2406.05800v2)

**Authors**: Chen Ma, Ningfei Wang, Zhengyu Zhao, Qi Alfred Chen, Chao Shen

**Abstract**: Autonomous Driving (AD) systems critically depend on visual perception for real-time object detection and multiple object tracking (MOT) to ensure safe driving. However, high latency in these visual perception components can lead to significant safety risks, such as vehicle collisions. While previous research has extensively explored latency attacks within the digital realm, translating these methods effectively to the physical world presents challenges. For instance, existing attacks rely on perturbations that are unrealistic or impractical for AD, such as adversarial perturbations affecting areas like the sky, or requiring large patches that obscure most of a camera's view, thus making them impossible to be conducted effectively in the real world.   In this paper, we introduce SlowPerception, the first physical-world latency attack against AD perception, via generating projector-based universal perturbations. SlowPerception strategically creates numerous phantom objects on various surfaces in the environment, significantly increasing the computational load of Non-Maximum Suppression (NMS) and MOT, thereby inducing substantial latency. Our SlowPerception achieves second-level latency in physical-world settings, with an average latency of 2.5 seconds across different AD perception systems, scenarios, and hardware configurations. This performance significantly outperforms existing state-of-the-art latency attacks. Additionally, we conduct AD system-level impact assessments, such as vehicle collisions, using industry-grade AD systems with production-grade AD simulators with a 97% average rate. We hope that our analyses can inspire further research in this critical domain, enhancing the robustness of AD systems against emerging vulnerabilities.

摘要: 自动驾驶(AD)系统在很大程度上依赖于视觉感知进行实时目标检测和多目标跟踪(MOT)来确保安全驾驶。然而，这些视觉感知组件的高延迟可能会导致重大安全风险，如车辆碰撞。虽然之前的研究已经广泛地探索了数字领域内的延迟攻击，但将这些方法有效地转换到物理世界是一项挑战。例如，现有的攻击依赖于对AD来说不现实或不切实际的扰动，例如影响天空等区域的对抗性扰动，或者需要遮挡大部分摄像机视野的大补丁，从而使它们不可能在现实世界中有效地进行。在本文中，我们通过产生基于投影仪的普遍扰动，引入了第一个针对AD感知的物理世界延迟攻击SlowPercept。SlowPercept战略性地在环境中的不同表面创建大量幻影对象，显著增加非最大抑制(NMS)和MOT的计算负荷，从而导致显著的延迟。我们的SlowPercept在物理世界设置中实现了二级延迟，跨不同AD感知系统、场景和硬件配置的平均延迟为2.5秒。这一性能大大超过了现有最先进的延迟攻击。此外，我们使用带有生产级AD模拟器的工业级AD系统进行AD系统级影响评估，例如车辆碰撞，平均比率为97%。我们希望我们的分析能够启发这一关键领域的进一步研究，增强AD系统对新出现的漏洞的健壮性。



## **3. Do Parameters Reveal More than Loss for Membership Inference?**

参数揭示的不仅仅是会员推断的损失吗？ cs.LG

Accepted at High-dimensional Learning Dynamics (HiLD) Workshop, ICML  2024

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2406.11544v2) [paper-pdf](http://arxiv.org/pdf/2406.11544v2)

**Authors**: Anshuman Suri, Xiao Zhang, David Evans

**Abstract**: Membership inference attacks aim to infer whether an individual record was used to train a model, serving as a key tool for disclosure auditing. While such evaluations are useful to demonstrate risk, they are computationally expensive and often make strong assumptions about potential adversaries' access to models and training environments, and thus do not provide very tight bounds on leakage from potential attacks. We show how prior claims around black-box access being sufficient for optimal membership inference do not hold for most useful settings such as stochastic gradient descent, and that optimal membership inference indeed requires white-box access. We validate our findings with a new white-box inference attack IHA (Inverse Hessian Attack) that explicitly uses model parameters by taking advantage of computing inverse-Hessian vector products. Our results show that both audits and adversaries may be able to benefit from access to model parameters, and we advocate for further research into white-box methods for membership privacy auditing.

摘要: 成员资格推断攻击旨在推断个人记录是否被用来训练模型，作为披露审计的关键工具。虽然这样的评估有助于显示风险，但它们的计算成本很高，而且通常会对潜在对手访问模型和训练环境做出强有力的假设，因此不会对潜在攻击的泄漏提供非常严格的限制。我们证明了关于黑箱访问的最优成员关系推理的先前声明如何不适用于大多数有用的设置，例如随机梯度下降，而最优成员关系推理确实需要白箱访问。我们使用一种新的白盒推理攻击IHA(逆向Hessian攻击)来验证我们的发现，该攻击通过计算逆向Hessian向量积来显式地使用模型参数。我们的结果表明，审计和对手都可以从访问模型参数中受益，我们主张进一步研究成员隐私审计的白盒方法。



## **4. Does Refusal Training in LLMs Generalize to the Past Tense?**

LLM中的拒绝培训是否适用于过去时态？ cs.CL

Update in v2: Claude-3.5 Sonnet and GPT-4o mini. We provide code and  jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.11969v2) [paper-pdf](http://arxiv.org/pdf/2407.11969v2)

**Authors**: Maksym Andriushchenko, Nicolas Flammarion

**Abstract**: Refusal training is widely used to prevent LLMs from generating harmful, undesirable, or illegal outputs. We reveal a curious generalization gap in the current refusal training approaches: simply reformulating a harmful request in the past tense (e.g., "How to make a Molotov cocktail?" to "How did people make a Molotov cocktail?") is often sufficient to jailbreak many state-of-the-art LLMs. We systematically evaluate this method on Llama-3 8B, Claude-3.5 Sonnet, GPT-3.5 Turbo, Gemma-2 9B, Phi-3-Mini, GPT-4o mini, GPT-4o, and R2D2 models using GPT-3.5 Turbo as a reformulation model. For example, the success rate of this simple attack on GPT-4o increases from 1% using direct requests to 88% using 20 past tense reformulation attempts on harmful requests from JailbreakBench with GPT-4 as a jailbreak judge. Interestingly, we also find that reformulations in the future tense are less effective, suggesting that refusal guardrails tend to consider past historical questions more benign than hypothetical future questions. Moreover, our experiments on fine-tuning GPT-3.5 Turbo show that defending against past reformulations is feasible when past tense examples are explicitly included in the fine-tuning data. Overall, our findings highlight that the widely used alignment techniques -- such as SFT, RLHF, and adversarial training -- employed to align the studied models can be brittle and do not always generalize as intended. We provide code and jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense.

摘要: 拒绝训练被广泛用于防止LLMS产生有害、不受欢迎或非法的输出。我们揭示了当前拒绝训练方法中一个奇怪的概括缺口：简单地用过去时重新表达一个有害的请求(例如，“如何调制燃烧鸡尾酒？”“人们是如何调制燃烧鸡尾酒的？”)通常足以越狱许多最先进的LLM。我们以GPT-3.5 Turbo为改写模型，对Llama-3 8B、Claude-3.5十四行诗、GPT-3.5 Turbo、Gema-2 9B、Phi-3-Mini、GPT-40 mini、GPT-40和R2D2模型进行了系统的评估。例如，对GPT-4o的这种简单攻击的成功率从使用直接请求的1%增加到使用20次过去时态重组尝试的88%，这些尝试使用GPT-4作为越狱法官的JailBreakB边的有害请求。有趣的是，我们还发现，未来时的重述没有那么有效，这表明拒绝障碍倾向于考虑过去的历史问题，而不是假设的未来问题。此外，我们在微调GPT-3.5Turbo上的实验表明，当微调数据中明确包含过去时态示例时，防御过去的重新公式是可行的。总体而言，我们的发现强调了广泛使用的对齐技术--如SFT、RLHF和对抗性训练--用于对所研究的模型进行对齐可能是脆弱的，并且并不总是像预期的那样泛化。我们在https://github.com/tml-epfl/llm-past-tense.上提供代码和越狱文物



## **5. Watermark Smoothing Attacks against Language Models**

针对语言模型的水印平滑攻击 cs.LG

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.14206v1) [paper-pdf](http://arxiv.org/pdf/2407.14206v1)

**Authors**: Hongyan Chang, Hamed Hassani, Reza Shokri

**Abstract**: Watermarking is a technique used to embed a hidden signal in the probability distribution of text generated by large language models (LLMs), enabling attribution of the text to the originating model. We introduce smoothing attacks and show that existing watermarking methods are not robust against minor modifications of text. An adversary can use weaker language models to smooth out the distribution perturbations caused by watermarks without significantly compromising the quality of the generated text. The modified text resulting from the smoothing attack remains close to the distribution of text that the original model (without watermark) would have produced. Our attack reveals a fundamental limitation of a wide range of watermarking techniques.

摘要: 水印是一种用于将隐藏信号嵌入大型语言模型（LLM）生成的文本的概率分布中的技术，从而将文本归因于原始模型。我们引入了平滑攻击，并表明现有的水印方法对文本的微小修改并不鲁棒。对手可以使用较弱的语言模型来平滑水印引起的分布扰动，而不会显着损害生成文本的质量。平滑攻击产生的修改文本仍然接近原始模型（没有水印）产生的文本分布。我们的攻击揭示了广泛水印技术的根本局限性。



## **6. MVPatch: More Vivid Patch for Adversarial Camouflaged Attacks on Object Detectors in the Physical World**

MVpatch：针对物理世界中物体检测器的对抗性伪装攻击的更多生动补丁 cs.CR

16 pages, 8 figures. This work has been submitted to the IEEE for  possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2312.17431v3) [paper-pdf](http://arxiv.org/pdf/2312.17431v3)

**Authors**: Zheng Zhou, Hongbo Zhao, Ju Liu, Qiaosheng Zhang, Liwei Geng, Shuchang Lyu, Wenquan Feng

**Abstract**: Recent studies have shown that Adversarial Patches (APs) can effectively manipulate object detection models. However, the conspicuous patterns often associated with these patches tend to attract human attention, posing a significant challenge. Existing research has primarily focused on enhancing attack efficacy in the physical domain while often neglecting the optimization of stealthiness and transferability. Furthermore, applying APs in real-world scenarios faces major challenges related to transferability, stealthiness, and practicality. To address these challenges, we introduce generalization theory into the context of APs, enabling our iterative process to simultaneously enhance transferability and refine visual correlation with realistic images. We propose a Dual-Perception-Based Framework (DPBF) to generate the More Vivid Patch (MVPatch), which enhances transferability, stealthiness, and practicality. The DPBF integrates two key components: the Model-Perception-Based Module (MPBM) and the Human-Perception-Based Module (HPBM), along with regularization terms. The MPBM employs ensemble strategy to reduce object confidence scores across multiple detectors, thereby improving AP transferability with robust theoretical support. Concurrently, the HPBM introduces a lightweight method for achieving visual similarity, creating natural and inconspicuous adversarial patches without relying on additional generative models. The regularization terms further enhance the practicality of the generated APs in the physical domain. Additionally, we introduce naturalness and transferability scores to provide an unbiased assessment of APs. Extensive experimental validation demonstrates that MVPatch achieves superior transferability and a natural appearance in both digital and physical domains, underscoring its effectiveness and stealthiness.

摘要: 最近的研究表明，对抗性补丁(AP)可以有效地操纵目标检测模型。然而，与这些斑块相关的明显图案往往会吸引人类的注意，构成了一个重大的挑战。现有的研究主要集中在提高物理领域的攻击效能，而往往忽略了隐蔽性和可转移性的优化。此外，在真实场景中应用AP面临着与可转移性、隐蔽性和实用性相关的重大挑战。为了应对这些挑战，我们将泛化理论引入到AP的上下文中，使我们的迭代过程能够同时增强可转移性和精细化与真实图像的视觉相关性。我们提出了一种基于双重感知的框架(DPBF)来生成更生动的补丁(MVPatch)，从而增强了可转移性、隐蔽性和实用性。DPBF集成了两个关键组件：基于模型感知的模块(MPBM)和基于人的感知的模块(HPBM)，以及正则化项。MPBM使用集成策略来降低多个检测器上的对象置信度分数，从而在稳健的理论支持下提高AP的可转移性。同时，HPBM引入了一种轻量级的方法来实现视觉相似性，创建自然的和不明显的对抗性补丁，而不依赖于额外的生成模型。正则化项进一步增强了所生成的AP在物理域中的实用性。此外，我们引入了自然性和可转移性分数，以提供对AP的公正评估。广泛的实验验证表明，MVPatch在数字和物理领域都实现了卓越的可转移性和自然外观，突显了其有效性和隐蔽性。



## **7. Adversarial Examples in the Physical World: A Survey**

物理世界中的对抗例子：调查 cs.CV

Adversarial examples, physical-world scenarios, attacks and defenses

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2311.01473v2) [paper-pdf](http://arxiv.org/pdf/2311.01473v2)

**Authors**: Jiakai Wang, Xianglong Liu, Jin Hu, Donghua Wang, Siyang Wu, Tingsong Jiang, Yuanfang Guo, Aishan Liu, Aishan Liu, Jiantao Zhou

**Abstract**: Deep neural networks (DNNs) have demonstrated high vulnerability to adversarial examples, raising broad security concerns about their applications. Besides the attacks in the digital world, the practical implications of adversarial examples in the physical world present significant challenges and safety concerns. However, current research on physical adversarial examples (PAEs) lacks a comprehensive understanding of their unique characteristics, leading to limited significance and understanding. In this paper, we address this gap by thoroughly examining the characteristics of PAEs within a practical workflow encompassing training, manufacturing, and re-sampling processes. By analyzing the links between physical adversarial attacks, we identify manufacturing and re-sampling as the primary sources of distinct attributes and particularities in PAEs. Leveraging this knowledge, we develop a comprehensive analysis and classification framework for PAEs based on their specific characteristics, covering over 100 studies on physical-world adversarial examples. Furthermore, we investigate defense strategies against PAEs and identify open challenges and opportunities for future research. We aim to provide a fresh, thorough, and systematic understanding of PAEs, thereby promoting the development of robust adversarial learning and its application in open-world scenarios to provide the community with a continuously updated list of physical world adversarial sample resources, including papers, code, \etc, within the proposed framework

摘要: 深度神经网络(DNN)对敌意例子表现出很高的脆弱性，引起了人们对其应用的广泛安全担忧。除了数字世界中的攻击，物理世界中敌意例子的实际影响也带来了重大挑战和安全问题。然而，目前对身体对抗例子(PAE)的研究缺乏对其独特特征的全面了解，导致其意义和理解有限。在本文中，我们通过彻底检查包括培训、制造和重新采样过程在内的实际工作流程中的PAE的特征来解决这一差距。通过分析物理对抗性攻击之间的联系，我们确定制造和重采样是PAE中不同属性和特殊性的主要来源。利用这些知识，我们根据PAE的具体特征开发了一个全面的分析和分类框架，涵盖了100多个物理世界对抗性例子的研究。此外，我们还研究了针对PAE的防御策略，并确定了未来研究的开放挑战和机会。我们的目标是提供一个新的，彻底的和系统的了解，从而促进发展强大的对抗性学习及其在开放世界情景中的应用，以在拟议的框架内为社区提供持续更新的物理世界对抗性样本资源列表，包括论文、代码等



## **8. Resilient Consensus Sustained Collaboratively**

通过协作维持弹性共识 cs.CR

15 pages, 7 figures

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2302.02325v5) [paper-pdf](http://arxiv.org/pdf/2302.02325v5)

**Authors**: Junchao Chen, Suyash Gupta, Alberto Sonnino, Lefteris Kokoris-Kogias, Mohammad Sadoghi

**Abstract**: Decentralized systems built around blockchain technology promise clients an immutable ledger. They add a transaction to the ledger after it undergoes consensus among the replicas that run a Proof-of-Stake (PoS) or Byzantine Fault-Tolerant (BFT) consensus protocol. Unfortunately, these protocols face a long-range attack where an adversary having access to the private keys of the replicas can rewrite the ledger. One solution is forcing each committed block from these protocols to undergo another consensus, Proof-of-Work(PoW) consensus; PoW protocol leads to wastage of computational resources as miners compete to solve complex puzzles. In this paper, we present the design of our Power-of-Collaboration (PoC) protocol, which guards existing PoS/BFT blockchains against long-range attacks and requires miners to collaborate rather than compete. PoC guarantees fairness and accountability and only marginally degrades the throughput of the underlying system.

摘要: 围绕区块链技术构建的去中心化系统向客户承诺一个不可改变的分类帐。在运行权益证明（PoS）或拜占庭过失容忍（BFT）共识协议的副本之间达成共识后，他们将交易添加到分类帐中。不幸的是，这些协议面临着远程攻击，其中可以访问副本的私有密钥的对手可以重写分类帐。一种解决方案是迫使这些协议中的每个已提交的块经历另一种共识，即工作量证明（PoW）共识;随着矿工竞相解决复杂难题，PoW协议导致计算资源的浪费。在本文中，我们介绍了协作力量（NPS）协议的设计，该协议保护现有PoS/BFT区块链免受远程攻击，并要求矿工进行合作而不是竞争。收件箱保证公平性和问责制，并且只略微降低底层系统的吞吐量。



## **9. Personalized Privacy Protection Mask Against Unauthorized Facial Recognition**

针对未经授权的面部识别的个性化隐私保护面具 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.13975v1) [paper-pdf](http://arxiv.org/pdf/2407.13975v1)

**Authors**: Ka-Ho Chow, Sihao Hu, Tiansheng Huang, Ling Liu

**Abstract**: Face recognition (FR) can be abused for privacy intrusion. Governments, private companies, or even individual attackers can collect facial images by web scraping to build an FR system identifying human faces without their consent. This paper introduces Chameleon, which learns to generate a user-centric personalized privacy protection mask, coined as P3-Mask, to protect facial images against unauthorized FR with three salient features. First, we use a cross-image optimization to generate one P3-Mask for each user instead of tailoring facial perturbation for each facial image of a user. It enables efficient and instant protection even for users with limited computing resources. Second, we incorporate a perceptibility optimization to preserve the visual quality of the protected facial images. Third, we strengthen the robustness of P3-Mask against unknown FR models by integrating focal diversity-optimized ensemble learning into the mask generation process. Extensive experiments on two benchmark datasets show that Chameleon outperforms three state-of-the-art methods with instant protection and minimal degradation of image quality. Furthermore, Chameleon enables cost-effective FR authorization using the P3-Mask as a personalized de-obfuscation key, and it demonstrates high resilience against adaptive adversaries.

摘要: 人脸识别(FR)可能被滥用来侵犯隐私。政府、私人公司，甚至个人攻击者都可以通过网络抓取来收集面部图像，以建立一个识别人脸的FR系统，而不需要他们的同意。本文介绍了变色龙，它学习生成一个以用户为中心的个性化隐私保护面具，称为P3-面具，以保护面部图像免受未经授权的FR，具有三个显著特征。首先，我们使用交叉图像优化为每个用户生成一个P3-面具，而不是为每个用户的人脸图像定制面部扰动。它可以为计算资源有限的用户提供高效、即时的保护。其次，我们加入了感知性优化，以保持受保护的面部图像的视觉质量。第三，通过在模板生成过程中集成焦点分集优化的集成学习来增强P3-MASK对未知FR模型的稳健性。在两个基准数据集上的广泛实验表明，变色龙的性能优于三种最先进的方法，具有即时保护和最小的图像质量退化。此外，变色龙能够使用P3-MASK作为个性化的去模糊密钥来实现经济高效的FR授权，并且它对自适应对手表现出高弹性。



## **10. A Closer Look at GAN Priors: Exploiting Intermediate Features for Enhanced Model Inversion Attacks**

仔细研究GAN先验：利用中间功能进行增强模型倒置攻击 cs.CV

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13863v1) [paper-pdf](http://arxiv.org/pdf/2407.13863v1)

**Authors**: Yixiang Qiu, Hao Fang, Hongyao Yu, Bin Chen, MeiKang Qiu, Shu-Tao Xia

**Abstract**: Model Inversion (MI) attacks aim to reconstruct privacy-sensitive training data from released models by utilizing output information, raising extensive concerns about the security of Deep Neural Networks (DNNs). Recent advances in generative adversarial networks (GANs) have contributed significantly to the improved performance of MI attacks due to their powerful ability to generate realistic images with high fidelity and appropriate semantics. However, previous MI attacks have solely disclosed private information in the latent space of GAN priors, limiting their semantic extraction and transferability across multiple target models and datasets. To address this challenge, we propose a novel method, Intermediate Features enhanced Generative Model Inversion (IF-GMI), which disassembles the GAN structure and exploits features between intermediate blocks. This allows us to extend the optimization space from latent code to intermediate features with enhanced expressive capabilities. To prevent GAN priors from generating unrealistic images, we apply a L1 ball constraint to the optimization process. Experiments on multiple benchmarks demonstrate that our method significantly outperforms previous approaches and achieves state-of-the-art results under various settings, especially in the out-of-distribution (OOD) scenario. Our code is available at: https://github.com/final-solution/IF-GMI

摘要: 模型反转(MI)攻击的目的是利用输出信息从已发布的模型中重建隐私敏感的训练数据，这引起了人们对深度神经网络(DNN)安全性的广泛关注。生成性对抗网络(GANS)的最新进展为MI攻击的性能改进做出了重要贡献，因为它们能够生成高保真和适当语义的真实图像。然而，以往的MI攻击只在GaN先验的潜在空间中泄露隐私信息，限制了它们的语义提取和跨多个目标模型和数据集的可传输性。为了解决这一挑战，我们提出了一种新的方法，中间特征增强的生成性模型反转(IF-GMI)，它分解GaN结构并利用中间块之间的特征。这允许我们将优化空间从潜在代码扩展到具有增强表达能力的中间功能。为了防止GaN先验数据产生不真实的图像，我们在优化过程中应用了L1球约束。在多个基准测试上的实验表明，我们的方法显著优于以前的方法，并在各种设置下获得了最先进的结果，特别是在分布外(OOD)的情况下。我们的代码请访问：https://github.com/final-solution/IF-GMI



## **11. Jailbreaking Black Box Large Language Models in Twenty Queries**

二十分钟内越狱黑匣子大型语言模型 cs.LG

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2310.08419v4) [paper-pdf](http://arxiv.org/pdf/2310.08419v4)

**Authors**: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong

**Abstract**: There is growing interest in ensuring that large language models (LLMs) align with human values. However, the alignment of such models is vulnerable to adversarial jailbreaks, which coax LLMs into overriding their safety guardrails. The identification of these vulnerabilities is therefore instrumental in understanding inherent weaknesses and preventing future misuse. To this end, we propose Prompt Automatic Iterative Refinement (PAIR), an algorithm that generates semantic jailbreaks with only black-box access to an LLM. PAIR -- which is inspired by social engineering attacks -- uses an attacker LLM to automatically generate jailbreaks for a separate targeted LLM without human intervention. In this way, the attacker LLM iteratively queries the target LLM to update and refine a candidate jailbreak. Empirically, PAIR often requires fewer than twenty queries to produce a jailbreak, which is orders of magnitude more efficient than existing algorithms. PAIR also achieves competitive jailbreaking success rates and transferability on open and closed-source LLMs, including GPT-3.5/4, Vicuna, and Gemini.

摘要: 人们对确保大型语言模型(LLM)与人类价值观保持一致的兴趣与日俱增。然而，这类模型的调整很容易受到对抗性越狱的影响，这会诱使低收入国家凌驾于他们的安全护栏之上。因此，确定这些漏洞有助于了解固有的弱点并防止今后的滥用。为此，我们提出了即时自动迭代求精(Pair)，这是一种仅通过黑盒访问LLM来生成语义越狱的算法。Pair受到社会工程攻击的启发，它使用攻击者LLM自动为单独的目标LLM生成越狱，而无需人工干预。通过这种方式，攻击者LLM迭代地查询目标LLM以更新和改进候选越狱。根据经验，Pair通常只需要不到20次查询就可以产生越狱，这比现有算法的效率高出几个数量级。Pair还在开放和封闭源代码的LLM上实现了具有竞争力的越狱成功率和可转移性，包括GPT-3.5/4、维库纳和双子座。



## **12. Black-Box Opinion Manipulation Attacks to Retrieval-Augmented Generation of Large Language Models**

对大型语言模型检索增强生成的黑匣子观点操纵攻击 cs.CL

10 pages, 3 figures, under review

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13757v1) [paper-pdf](http://arxiv.org/pdf/2407.13757v1)

**Authors**: Zhuo Chen, Jiawei Liu, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) is applied to solve hallucination problems and real-time constraints of large language models, but it also induces vulnerabilities against retrieval corruption attacks. Existing research mainly explores the unreliability of RAG in white-box and closed-domain QA tasks. In this paper, we aim to reveal the vulnerabilities of Retrieval-Enhanced Generative (RAG) models when faced with black-box attacks for opinion manipulation. We explore the impact of such attacks on user cognition and decision-making, providing new insight to enhance the reliability and security of RAG models. We manipulate the ranking results of the retrieval model in RAG with instruction and use these results as data to train a surrogate model. By employing adversarial retrieval attack methods to the surrogate model, black-box transfer attacks on RAG are further realized. Experiments conducted on opinion datasets across multiple topics show that the proposed attack strategy can significantly alter the opinion polarity of the content generated by RAG. This demonstrates the model's vulnerability and, more importantly, reveals the potential negative impact on user cognition and decision-making, making it easier to mislead users into accepting incorrect or biased information.

摘要: 检索增强生成(RAG)被应用于解决大型语言模型的幻觉问题和实时约束，但它也导致了对检索破坏攻击的脆弱性。已有研究主要探讨RAG在白盒和封闭域QA任务中的不可靠性。在本文中，我们旨在揭示检索增强生成(RAG)模型在面对意见操纵黑盒攻击时的脆弱性。我们探讨了此类攻击对用户认知和决策的影响，为提高RAG模型的可靠性和安全性提供了新的见解。我们通过指令对检索模型在RAG中的排序结果进行操作，并将这些结果作为数据来训练代理模型。通过对代理模型采用对抗性检索攻击方法，进一步实现了对RAG的黑箱转移攻击。在多个主题的观点数据集上进行的实验表明，所提出的攻击策略可以显著改变RAG生成的内容的观点极性。这表明了该模型的脆弱性，更重要的是，揭示了对用户认知和决策的潜在负面影响，使其更容易误导用户接受不正确或有偏见的信息。



## **13. Cross-Task Attack: A Self-Supervision Generative Framework Based on Attention Shift**

跨任务攻击：基于注意力转移的自我监督生成框架 cs.CV

Has been accepted by IJCNN2024

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13700v1) [paper-pdf](http://arxiv.org/pdf/2407.13700v1)

**Authors**: Qingyuan Zeng, Yunpeng Gong, Min Jiang

**Abstract**: Studying adversarial attacks on artificial intelligence (AI) systems helps discover model shortcomings, enabling the construction of a more robust system. Most existing adversarial attack methods only concentrate on single-task single-model or single-task cross-model scenarios, overlooking the multi-task characteristic of artificial intelligence systems. As a result, most of the existing attacks do not pose a practical threat to a comprehensive and collaborative AI system. However, implementing cross-task attacks is highly demanding and challenging due to the difficulty in obtaining the real labels of different tasks for the same picture and harmonizing the loss functions across different tasks. To address this issue, we propose a self-supervised Cross-Task Attack framework (CTA), which utilizes co-attention and anti-attention maps to generate cross-task adversarial perturbation. Specifically, the co-attention map reflects the area to which different visual task models pay attention, while the anti-attention map reflects the area that different visual task models neglect. CTA generates cross-task perturbations by shifting the attention area of samples away from the co-attention map and closer to the anti-attention map. We conduct extensive experiments on multiple vision tasks and the experimental results confirm the effectiveness of the proposed design for adversarial attacks.

摘要: 研究对人工智能(AI)系统的对抗性攻击有助于发现模型的缺陷，使构建更健壮的系统成为可能。现有的对抗性攻击方法大多只关注单任务单模型或单任务跨模型场景，忽略了人工智能系统的多任务特性。因此，现有的大多数攻击都不会对一个全面、协同的AI系统构成实际威胁。然而，由于难以获得同一图片的不同任务的真实标签以及协调不同任务之间的损失函数，实施跨任务攻击的要求和挑战性很高。针对这一问题，我们提出了一种自监督的跨任务攻击框架(CTA)，该框架利用共同注意图和反注意图来产生跨任务的敌意扰动。具体而言，共注意图反映了不同视觉任务模型注意的区域，反注意图反映了不同视觉任务模型忽视的区域。CTA通过将样本的注意区域从共同注意图转移到更接近反注意图的位置来产生跨任务扰动。我们在多个视觉任务上进行了大量的实验，实验结果证实了所提出的设计对于对抗性攻击的有效性。



## **14. Prover-Verifier Games improve legibility of LLM outputs**

证明者-验证者游戏提高了LLM输出的清晰度 cs.CL

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13692v1) [paper-pdf](http://arxiv.org/pdf/2407.13692v1)

**Authors**: Jan Hendrik Kirchner, Yining Chen, Harri Edwards, Jan Leike, Nat McAleese, Yuri Burda

**Abstract**: One way to increase confidence in the outputs of Large Language Models (LLMs) is to support them with reasoning that is clear and easy to check -- a property we call legibility. We study legibility in the context of solving grade-school math problems and show that optimizing chain-of-thought solutions only for answer correctness can make them less legible. To mitigate the loss in legibility, we propose a training algorithm inspired by Prover-Verifier Game from Anil et al. (2021). Our algorithm iteratively trains small verifiers to predict solution correctness, "helpful" provers to produce correct solutions that the verifier accepts, and "sneaky" provers to produce incorrect solutions that fool the verifier. We find that the helpful prover's accuracy and the verifier's robustness to adversarial attacks increase over the course of training. Furthermore, we show that legibility training transfers to time-constrained humans tasked with verifying solution correctness. Over course of LLM training human accuracy increases when checking the helpful prover's solutions, and decreases when checking the sneaky prover's solutions. Hence, training for checkability by small verifiers is a plausible technique for increasing output legibility. Our results suggest legibility training against small verifiers as a practical avenue for increasing legibility of large LLMs to humans, and thus could help with alignment of superhuman models.

摘要: 增加对大型语言模型(LLM)输出的信心的一种方法是用清晰且易于检查的推理来支持它们--我们称之为易读性。我们在解决小学数学问题的背景下研究了易读性，并表明只为了答案的正确性而优化思维链解决方案会降低它们的易读性。为了减少易读性的损失，我们提出了一种受Anil等人的Prover-Verator游戏启发的训练算法。(2021年)。我们的算法迭代地训练小的验证者来预测解决方案的正确性，“有帮助的”验证者来产生验证者接受的正确的解决方案，而“偷偷摸摸”的验证者产生愚弄验证者的不正确的解决方案。我们发现，随着训练过程的进行，有益证明者的准确率和验证者对敌意攻击的健壮性都有所提高。此外，我们还表明，易读性训练转移到负责验证解决方案正确性的时间受限的人身上。在LLM训练过程中，当检查有用的证明者的解时，人类的准确率提高，而当检查偷偷摸摸的证明者的解时，人类的准确率降低。因此，由小型验证员进行可校验性培训是提高输出清晰度的一种可行的技术。我们的结果表明，针对小验证者的易读性训练是提高大型LLM对人类易读性的实用途径，因此可能有助于超人模型的对齐。



## **15. Beyond Dropout: Robust Convolutional Neural Networks Based on Local Feature Masking**

Beyond Dropout：基于局部特征掩蔽的鲁棒卷积神经网络 cs.CV

It has been accepted by IJCNN 2024

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13646v1) [paper-pdf](http://arxiv.org/pdf/2407.13646v1)

**Authors**: Yunpeng Gong, Chuangliang Zhang, Yongjie Hou, Lifei Chen, Min Jiang

**Abstract**: In the contemporary of deep learning, where models often grapple with the challenge of simultaneously achieving robustness against adversarial attacks and strong generalization capabilities, this study introduces an innovative Local Feature Masking (LFM) strategy aimed at fortifying the performance of Convolutional Neural Networks (CNNs) on both fronts. During the training phase, we strategically incorporate random feature masking in the shallow layers of CNNs, effectively alleviating overfitting issues, thereby enhancing the model's generalization ability and bolstering its resilience to adversarial attacks. LFM compels the network to adapt by leveraging remaining features to compensate for the absence of certain semantic features, nurturing a more elastic feature learning mechanism. The efficacy of LFM is substantiated through a series of quantitative and qualitative assessments, collectively showcasing a consistent and significant improvement in CNN's generalization ability and resistance against adversarial attacks--a phenomenon not observed in current and prior methodologies. The seamless integration of LFM into established CNN frameworks underscores its potential to advance both generalization and adversarial robustness within the deep learning paradigm. Through comprehensive experiments, including robust person re-identification baseline generalization experiments and adversarial attack experiments, we demonstrate the substantial enhancements offered by LFM in addressing the aforementioned challenges. This contribution represents a noteworthy stride in advancing robust neural network architectures.

摘要: 在深度学习的当代，模型经常面临同时获得对对手攻击的鲁棒性和强大的泛化能力的挑战，该研究引入了一种创新的局部特征掩蔽(LFM)策略，旨在增强卷积神经网络(CNN)在这两个方面的性能。在训练阶段，我们战略性地将随机特征掩蔽引入到CNN的浅层，有效地缓解了过拟合问题，从而增强了模型的泛化能力，增强了模型对对手攻击的韧性。LFM通过利用剩余特征来补偿某些语义特征的缺失来迫使网络适应，从而培育出更灵活的特征学习机制。LFM的有效性通过一系列定量和定性评估得到证实，这些评估共同表明CNN的泛化能力和对对抗性攻击的抵抗力得到了持续和显著的改善--这是目前和以前的方法中没有观察到的现象。LFM无缝集成到已建立的CNN框架中，突显了它在深度学习范式中提高泛化和对抗性稳健性的潜力。通过全面的实验，包括稳健的人重新识别基线泛化实验和对抗性攻击实验，我们展示了LFM在应对上述挑战方面所提供的实质性增强。这一贡献代表着在推进健壮的神经网络结构方面迈出了值得注意的一步。



## **16. Not Just Change the Labels, Learn the Features: Watermarking Deep Neural Networks with Multi-View Data**

不仅仅改变标签，学习功能：使用多视图数据对深度神经网络进行水印 cs.CR

ECCV 2024

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2403.10663v2) [paper-pdf](http://arxiv.org/pdf/2403.10663v2)

**Authors**: Yuxuan Li, Sarthak Kumar Maharana, Yunhui Guo

**Abstract**: With the increasing prevalence of Machine Learning as a Service (MLaaS) platforms, there is a growing focus on deep neural network (DNN) watermarking techniques. These methods are used to facilitate the verification of ownership for a target DNN model to protect intellectual property. One of the most widely employed watermarking techniques involves embedding a trigger set into the source model. Unfortunately, existing methodologies based on trigger sets are still susceptible to functionality-stealing attacks, potentially enabling adversaries to steal the functionality of the source model without a reliable means of verifying ownership. In this paper, we first introduce a novel perspective on trigger set-based watermarking methods from a feature learning perspective. Specifically, we demonstrate that by selecting data exhibiting multiple features, also referred to as \emph{multi-view data}, it becomes feasible to effectively defend functionality stealing attacks. Based on this perspective, we introduce a novel watermarking technique based on Multi-view dATa, called MAT, for efficiently embedding watermarks within DNNs. This approach involves constructing a trigger set with multi-view data and incorporating a simple feature-based regularization method for training the source model. We validate our method across various benchmarks and demonstrate its efficacy in defending against model extraction attacks, surpassing relevant baselines by a significant margin. The code is available at: \href{https://github.com/liyuxuan-github/MAT}{https://github.com/liyuxuan-github/MAT}.

摘要: 随着机器学习即服务(MLaaS)平台的日益普及，深度神经网络(DNN)水印技术受到越来越多的关注。这些方法用于帮助验证目标DNN模型的所有权，以保护知识产权。最广泛使用的水印技术之一涉及将触发集嵌入到源模型中。遗憾的是，基于触发器集的现有方法仍然容易受到功能窃取攻击，这可能使攻击者能够在没有可靠的验证所有权的方法的情况下窃取源模型的功能。本文首先从特征学习的角度介绍了一种基于触发集的数字水印方法。具体地说，我们证明了通过选择表现出多个特征的数据，也称为\MPH(多视图数据)，可以有效地防御功能窃取攻击。基于这一观点，我们提出了一种新的基于多视点数据的水印技术，称为MAT，用于在DNN中有效地嵌入水印。该方法包括利用多视点数据构造触发集，并结合简单的基于特征的正则化方法来训练源模型。我们在不同的基准上验证了我们的方法，并证明了它在防御模型提取攻击方面的有效性，远远超过了相关的基线。代码可从以下网址获得：\href{https://github.com/liyuxuan-github/MAT}{https://github.com/liyuxuan-github/MAT}.



## **17. Distributionally and Adversarially Robust Logistic Regression via Intersecting Wasserstein Balls**

通过交叉Wasserstein Balls进行分布和反向稳健逻辑回归 math.OC

34 pages, 3 color figures, under review at a conference

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13625v1) [paper-pdf](http://arxiv.org/pdf/2407.13625v1)

**Authors**: Aras Selvi, Eleonora Kreacic, Mohsen Ghassemi, Vamsi Potluru, Tucker Balch, Manuela Veloso

**Abstract**: Empirical risk minimization often fails to provide robustness against adversarial attacks in test data, causing poor out-of-sample performance. Adversarially robust optimization (ARO) has thus emerged as the de facto standard for obtaining models that hedge against such attacks. However, while these models are robust against adversarial attacks, they tend to suffer severely from overfitting. To address this issue for logistic regression, we study the Wasserstein distributionally robust (DR) counterpart of ARO and show that this problem admits a tractable reformulation. Furthermore, we develop a framework to reduce the conservatism of this problem by utilizing an auxiliary dataset (e.g., synthetic, external, or out-of-domain data), whenever available, with instances independently sampled from a nonidentical but related ground truth. In particular, we intersect the ambiguity set of the DR problem with another Wasserstein ambiguity set that is built using the auxiliary dataset. We analyze the properties of the underlying optimization problem, develop efficient solution algorithms, and demonstrate that the proposed method consistently outperforms benchmark approaches on real-world datasets.

摘要: 经验风险最小化往往不能在测试数据中提供对对手攻击的健壮性，从而导致样本外性能较差。相反，稳健优化(ARO)因此已经成为获得对冲此类攻击的模型的事实上的标准。然而，尽管这些模型对对手攻击很健壮，但它们往往会受到过度拟合的严重影响。为了解决Logistic回归的这个问题，我们研究了ARO的Wasserstein分布健壮性(DR)对应的问题，并表明这个问题允许一个易于处理的重新公式。此外，我们开发了一个框架来通过利用辅助数据集(例如，合成的、外部的或域外的数据)来减少这个问题的保守性，只要有可用的辅助数据集，实例就独立地从不相同但相关的基本事实中采样。特别是，我们将DR问题的歧义集与使用辅助数据集构建的另一个Wasserstein歧义集相交。我们分析了底层优化问题的性质，开发了有效的求解算法，并证明了所提出的方法在真实数据集上的性能一致优于基准方法。



## **18. VeriQR: A Robustness Verification Tool for Quantum Machine Learning Models**

VeriQR：量子机器学习模型的稳健性验证工具 quant-ph

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13533v1) [paper-pdf](http://arxiv.org/pdf/2407.13533v1)

**Authors**: Yanling Lin, Ji Guan, Wang Fang, Mingsheng Ying, Zhaofeng Su

**Abstract**: Adversarial noise attacks present a significant threat to quantum machine learning (QML) models, similar to their classical counterparts. This is especially true in the current Noisy Intermediate-Scale Quantum era, where noise is unavoidable. Therefore, it is essential to ensure the robustness of QML models before their deployment. To address this challenge, we introduce \textit{VeriQR}, the first tool designed specifically for formally verifying and improving the robustness of QML models, to the best of our knowledge. This tool mimics real-world quantum hardware's noisy impacts by incorporating random noise to formally validate a QML model's robustness. \textit{VeriQR} supports exact (sound and complete) algorithms for both local and global robustness verification. For enhanced efficiency, it implements an under-approximate (complete) algorithm and a tensor network-based algorithm to verify local and global robustness, respectively. As a formal verification tool, \textit{VeriQR} can detect adversarial examples and utilize them for further analysis and to enhance the local robustness through adversarial training, as demonstrated by experiments on real-world quantum machine learning models. Moreover, it permits users to incorporate customized noise. Based on this feature, we assess \textit{VeriQR} using various real-world examples, and experimental outcomes confirm that the addition of specific quantum noise can enhance the global robustness of QML models. These processes are made accessible through a user-friendly graphical interface provided by \textit{VeriQR}, catering to general users without requiring a deep understanding of the counter-intuitive probabilistic nature of quantum computing.

摘要: 对抗噪声攻击对量子机器学习(QML)模型构成了重大威胁，类似于它们的经典对应模型。尤其是在当前嘈杂的中尺度量子时代，噪音是不可避免的。因此，在QML模型部署之前，确保其健壮性是至关重要的。为了应对这一挑战，据我们所知，我们引入了第一个专门为正式验证和改进QML模型的健壮性而设计的工具\textit{VeriQR}。该工具通过加入随机噪声来正式验证QML模型的稳健性，从而模拟现实世界量子硬件的噪声影响。\textit{VeriQR}支持针对本地和全局健壮性验证的精确(声音和完整)算法。为了提高效率，它分别采用了欠近似(完全)算法和基于张量网络的算法来验证局部和全局的稳健性。在真实量子机器学习模型上的实验表明，作为一种形式化的验证工具，该工具可以检测敌意实例，并利用它们进行进一步的分析，并通过对抗性训练来增强局部稳健性。此外，它还允许用户加入定制的噪音。基于这一特征，我们使用各种真实世界的例子来评估文本{VeriQR}，实验结果证实了特定量子噪声的加入可以增强QML模型的全局稳健性。这些过程可以通过一个用户友好的图形界面进行访问，满足普通用户的需要，而不需要深入了解量子计算的反直觉概率性质。



## **19. Correlation inference attacks against machine learning models**

针对机器学习模型的相关推理攻击 cs.LG

Published in Science Advances. This version contains both the main  paper and supplementary material. There are minor editorial differences  between this version and the published version. The first two authors  contributed equally

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2112.08806v4) [paper-pdf](http://arxiv.org/pdf/2112.08806v4)

**Authors**: Ana-Maria Creţu, Florent Guépin, Yves-Alexandre de Montjoye

**Abstract**: Despite machine learning models being widely used today, the relationship between a model and its training dataset is not well understood. We explore correlation inference attacks, whether and when a model leaks information about the correlations between the input variables of its training dataset. We first propose a model-less attack, where an adversary exploits the spherical parametrization of correlation matrices alone to make an informed guess. Second, we propose a model-based attack, where an adversary exploits black-box model access to infer the correlations using minimal and realistic assumptions. Third, we evaluate our attacks against logistic regression and multilayer perceptron models on three tabular datasets and show the models to leak correlations. We finally show how extracted correlations can be used as building blocks for attribute inference attacks and enable weaker adversaries. Our results raise fundamental questions on what a model does and should remember from its training set.

摘要: 尽管机器学习模型在今天得到了广泛的应用，但模型与其训练数据集之间的关系并未得到很好的理解。我们探讨了关联推理攻击，即模型是否以及何时泄露了有关其训练数据集的输入变量之间的相关性的信息。我们首先提出了一种无模型攻击，其中敌手仅利用相关矩阵的球面参数化来进行知情猜测。其次，我们提出了一种基于模型的攻击，其中对手利用黑盒模型访问来使用最小和现实的假设来推断相关性。第三，我们在三个表格数据集上评估了我们对Logistic回归模型和多层感知器模型的攻击，并显示了模型的泄漏相关性。最后，我们展示了如何将提取的相关性用作属性推理攻击的构建块，并使较弱的对手能够进行攻击。我们的结果提出了一些基本问题，即模型从其训练集中做了什么，以及应该记住什么。



## **20. NeuroPlug: Plugging Side-Channel Leaks in NPUs using Space Filling Curves**

NeuroPlug：使用空间填充曲线堵住NPU中的侧通道泄漏 cs.CR

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13383v1) [paper-pdf](http://arxiv.org/pdf/2407.13383v1)

**Authors**: Nivedita Shrivastava, Smruti R. Sarangi

**Abstract**: Securing deep neural networks (DNNs) from side-channel attacks is an important problem as of today, given the substantial investment of time and resources in acquiring the raw data and training complex models. All published countermeasures (CMs) add noise N to a signal X (parameter of interest such as the net memory traffic that is leaked). The adversary observes X+N ; we shall show that it is easy to filter this noise out using targeted measurements, statistical analyses and different kinds of reasonably-assumed side information. We present a novel CM NeuroPlug that is immune to these attack methodologies mainly because we use a different formulation CX + N . We introduce a multiplicative variable C that naturally arises from feature map compression; it plays a key role in obfuscating the parameters of interest. Our approach is based on mapping all the computations to a 1-D space filling curve and then performing a sequence of tiling, compression and binning-based obfuscation operations. We follow up with proposing a theoretical framework based on Mellin transforms that allows us to accurately quantify the size of the search space as a function of the noise we add and the side information that an adversary possesses. The security guarantees provided by NeuroPlug are validated using a battery of statistical and information theory-based tests. We also demonstrate a substantial performance enhancement of 15% compared to the closest competing work.

摘要: 考虑到获取原始数据和训练复杂模型需要大量的时间和资源投入，保护深度神经网络(DNN)免受旁路攻击是当今的一个重要问题。所有发布的对策(CM)都将噪声N添加到信号X(感兴趣的参数，例如泄漏的网络内存流量)。对手观察X+N；我们将展示使用有针对性的测量、统计分析和各种合理假设的辅助信息来过滤这种噪声是很容易的。我们提出了一种新的CM NeuroPlug，它对这些攻击方法免疫，主要是因为我们使用了不同的公式CX+N。我们引入了一个自然产生于特征地图压缩的乘法变量C；它在混淆感兴趣的参数方面起着关键作用。我们的方法是将所有的计算映射到一维空间填充曲线上，然后执行一系列基于平铺、压缩和装箱的混淆操作。接下来，我们提出了一个基于梅林变换的理论框架，该框架允许我们准确地量化搜索空间的大小，作为我们添加的噪声和对手拥有的辅助信息的函数。NeuroPlug提供的安全保证使用了一系列基于统计和信息论的测试进行了验证。与最接近的竞争对手相比，我们还展示了15%的显著性能提升。



## **21. AgentDojo: A Dynamic Environment to Evaluate Attacks and Defenses for LLM Agents**

AgentDojo：评估LLM代理攻击和防御的动态环境 cs.CR

Updated version after fixing a bug in the Llama implementation and  updating the travel suite

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2406.13352v2) [paper-pdf](http://arxiv.org/pdf/2406.13352v2)

**Authors**: Edoardo Debenedetti, Jie Zhang, Mislav Balunović, Luca Beurer-Kellner, Marc Fischer, Florian Tramèr

**Abstract**: AI agents aim to solve complex tasks by combining text-based reasoning with external tool calls. Unfortunately, AI agents are vulnerable to prompt injection attacks where data returned by external tools hijacks the agent to execute malicious tasks. To measure the adversarial robustness of AI agents, we introduce AgentDojo, an evaluation framework for agents that execute tools over untrusted data. To capture the evolving nature of attacks and defenses, AgentDojo is not a static test suite, but rather an extensible environment for designing and evaluating new agent tasks, defenses, and adaptive attacks. We populate the environment with 97 realistic tasks (e.g., managing an email client, navigating an e-banking website, or making travel bookings), 629 security test cases, and various attack and defense paradigms from the literature. We find that AgentDojo poses a challenge for both attacks and defenses: state-of-the-art LLMs fail at many tasks (even in the absence of attacks), and existing prompt injection attacks break some security properties but not all. We hope that AgentDojo can foster research on new design principles for AI agents that solve common tasks in a reliable and robust manner. We release the code for AgentDojo at https://github.com/ethz-spylab/agentdojo.

摘要: 人工智能代理旨在通过将基于文本的推理与外部工具调用相结合来解决复杂任务。不幸的是，人工智能代理容易受到提示注入攻击，外部工具返回的数据劫持代理执行恶意任务。为了衡量AI代理的对抗健壮性，我们引入了AgentDojo，一个针对在不可信数据上执行工具的代理的评估框架。为了捕捉攻击和防御不断演变的本质，AgentDojo不是一个静态测试套件，而是一个可扩展的环境，用于设计和评估新的代理任务、防御和适应性攻击。我们在环境中填充了97项现实任务(例如，管理电子邮件客户端、浏览电子银行网站或预订旅行)、629个安全测试用例以及文献中的各种攻击和防御范例。我们发现AgentDojo对攻击和防御都构成了挑战：最先进的LLM在许多任务中失败(即使在没有攻击的情况下也是如此)，并且现有的即时注入攻击破坏了一些安全属性，但不是全部。我们希望AgentDojo能够促进对AI代理新设计原则的研究，这些原则能够以可靠和健壮的方式解决常见任务。我们在https://github.com/ethz-spylab/agentdojo.上发布了AgentDojo的代码



## **22. Benchmarking Robust Self-Supervised Learning Across Diverse Downstream Tasks**

在不同的下游任务中对稳健的自我监督学习进行基准测试 cs.CV

Accepted at the ICML 2024 Workshop on Foundation Models in the Wild

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.12588v2) [paper-pdf](http://arxiv.org/pdf/2407.12588v2)

**Authors**: Antoni Kowalczuk, Jan Dubiński, Atiyeh Ashari Ghomi, Yi Sui, George Stein, Jiapeng Wu, Jesse C. Cresswell, Franziska Boenisch, Adam Dziedzic

**Abstract**: Large-scale vision models have become integral in many applications due to their unprecedented performance and versatility across downstream tasks. However, the robustness of these foundation models has primarily been explored for a single task, namely image classification. The vulnerability of other common vision tasks, such as semantic segmentation and depth estimation, remains largely unknown. We present a comprehensive empirical evaluation of the adversarial robustness of self-supervised vision encoders across multiple downstream tasks. Our attacks operate in the encoder embedding space and at the downstream task output level. In both cases, current state-of-the-art adversarial fine-tuning techniques tested only for classification significantly degrade clean and robust performance on other tasks. Since the purpose of a foundation model is to cater to multiple applications at once, our findings reveal the need to enhance encoder robustness more broadly. Our code is available at ${github.com/layer6ai-labs/ssl-robustness}$.

摘要: 大规模视觉模型已经成为许多应用中不可或缺的一部分，因为它们具有前所未有的性能和跨下游任务的多功能性。然而，这些基础模型的稳健性主要是针对单个任务来探索的，即图像分类。其他常见的视觉任务，如语义分割和深度估计，其脆弱性在很大程度上仍然未知。我们提出了一个全面的经验评估的对抗性自监督视觉编码器跨越多个下游任务。我们的攻击在编码器嵌入空间和下游任务输出级别进行。在这两种情况下，当前最先进的对抗性微调技术仅针对分类进行测试，显著降低了其他任务的干净和健壮的性能。由于基础模型的目的是同时迎合多个应用，我们的研究结果揭示了更广泛地增强编码器健壮性的必要性。我们的代码可以在${githorb.com/layer6ai-Labs/ssl-rostness}$中找到。



## **23. Enhancing TinyML Security: Study of Adversarial Attack Transferability**

增强TinyML安全性：对抗性攻击可转移性研究 cs.CR

Accepted and presented at tinyML Foundation EMEA Innovation Forum  2024

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.11599v2) [paper-pdf](http://arxiv.org/pdf/2407.11599v2)

**Authors**: Parin Shah, Yuvaraj Govindarajulu, Pavan Kulkarni, Manojkumar Parmar

**Abstract**: The recent strides in artificial intelligence (AI) and machine learning (ML) have propelled the rise of TinyML, a paradigm enabling AI computations at the edge without dependence on cloud connections. While TinyML offers real-time data analysis and swift responses critical for diverse applications, its devices' intrinsic resource limitations expose them to security risks. This research delves into the adversarial vulnerabilities of AI models on resource-constrained embedded hardware, with a focus on Model Extraction and Evasion Attacks. Our findings reveal that adversarial attacks from powerful host machines could be transferred to smaller, less secure devices like ESP32 and Raspberry Pi. This illustrates that adversarial attacks could be extended to tiny devices, underscoring vulnerabilities, and emphasizing the necessity for reinforced security measures in TinyML deployments. This exploration enhances the comprehension of security challenges in TinyML and offers insights for safeguarding sensitive data and ensuring device dependability in AI-powered edge computing settings.

摘要: 人工智能(AI)和机器学习(ML)最近的进步推动了TinyML的崛起，TinyML是一种能够在边缘进行AI计算的范式，而不依赖于云连接。虽然TinyML提供对各种应用至关重要的实时数据分析和快速响应，但其设备固有的资源限制使它们面临安全风险。该研究深入研究了资源受限嵌入式硬件上人工智能模型的对抗性漏洞，重点研究了模型提取和规避攻击。我们的发现表明，来自强大主机的对抗性攻击可能会转移到较小、安全性较低的设备上，如ESP32和Raspberry PI。这表明敌意攻击可以扩展到微型设备，这突显了漏洞，并强调了在TinyML部署中加强安全措施的必要性。这一探索增强了对TinyML安全挑战的理解，并为在人工智能支持的边缘计算环境中保护敏感数据和确保设备可靠性提供了见解。



## **24. Compressed models are NOT miniature versions of large models**

压缩模型不是大型模型的微型版本 cs.LG

Accepted at the 33rd ACM International Conference on Information and  Knowledge Management (CIKM 2024) for the Short Research Paper track, 5 pages

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13174v1) [paper-pdf](http://arxiv.org/pdf/2407.13174v1)

**Authors**: Rohit Raj Rai, Rishant Pal, Amit Awekar

**Abstract**: Large neural models are often compressed before deployment. Model compression is necessary for many practical reasons, such as inference latency, memory footprint, and energy consumption. Compressed models are assumed to be miniature versions of corresponding large neural models. However, we question this belief in our work. We compare compressed models with corresponding large neural models using four model characteristics: prediction errors, data representation, data distribution, and vulnerability to adversarial attack. We perform experiments using the BERT-large model and its five compressed versions. For all four model characteristics, compressed models significantly differ from the BERT-large model. Even among compressed models, they differ from each other on all four model characteristics. Apart from the expected loss in model performance, there are major side effects of using compressed models to replace large neural models.

摘要: 大型神经模型通常在部署之前进行压缩。由于许多实际原因，例如推理延迟、内存占用和能源消耗，模型压缩是必要的。压缩模型被假设是相应大型神经模型的微型版本。然而，我们质疑我们工作中的这种信念。我们使用四个模型特征将压缩模型与相应的大型神经模型进行比较：预测误差、数据表示、数据分布和对抗攻击的脆弱性。我们使用BERT大模型及其五个压缩版本进行实验。对于所有四个模型特征，压缩模型与BERT大模型显着不同。即使在压缩模型中，它们在所有四个模型特征上也存在差异。除了模型性能的预期损失外，使用压缩模型来取代大型神经模型还存在重大副作用。



## **25. ToDA: Target-oriented Diffusion Attacker against Recommendation System**

ToDA：针对推荐系统的目标导向扩散攻击者 cs.CR

under-review

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2401.12578v3) [paper-pdf](http://arxiv.org/pdf/2401.12578v3)

**Authors**: Xiaohao Liu, Zhulin Tao, Ting Jiang, He Chang, Yunshan Ma, Yinwei Wei, Xiang Wang

**Abstract**: Recommendation systems (RS) have become indispensable tools for web services to address information overload, thus enhancing user experiences and bolstering platforms' revenues. However, with their increasing ubiquity, security concerns have also emerged. As the public accessibility of RS, they are susceptible to specific malicious attacks where adversaries can manipulate user profiles, leading to biased recommendations. Recent research often integrates additional modules using generative models to craft these deceptive user profiles, ensuring them are imperceptible while causing the intended harm. Albeit their efficacy, these models face challenges of unstable training and the exploration-exploitation dilemma, which can lead to suboptimal results. In this paper, we pioneer to investigate the potential of diffusion models (DMs), for shilling attacks. Specifically, we propose a novel Target-oriented Diffusion Attack model (ToDA). It incorporates a pre-trained autoencoder that transforms user profiles into a high dimensional space, paired with a Latent Diffusion Attacker (LDA)-the core component of ToDA. LDA introduces noise into the profiles within this latent space, adeptly steering the approximation towards targeted items through cross-attention mechanisms. The global horizon, implemented by a bipartite graph, is involved in LDA and derived from the encoded user profile feature. This makes LDA possible to extend the generation outwards the on-processing user feature itself, and bridges the gap between diffused user features and target item features. Extensive experiments compared to several SOTA baselines demonstrate ToDA's effectiveness. Specific studies exploit the elaborative design of ToDA and underscore the potency of advanced generative models in such contexts.

摘要: 推荐系统(RS)已经成为Web服务解决信息过载的不可或缺的工具，从而增强了用户体验并增加了平台的收入。然而，随着它们越来越普遍，安全问题也出现了。由于RS的公共可访问性，它们容易受到特定的恶意攻击，攻击者可以操纵用户配置文件，导致有偏见的推荐。最近的研究经常使用生成性模型集成额外的模块来制作这些欺骗性的用户配置文件，确保它们在造成预期伤害的同时是不可察觉的。尽管这些模式很有效，但它们面临着训练不稳定和勘探-开采困境的挑战，这可能导致不太理想的结果。在本文中，我们率先研究了扩散模型(DM)对先令攻击的可能性。具体来说，我们提出了一种新的面向目标的扩散攻击模型(Toda)。它结合了一个预先训练的自动编码器，可以将用户配置文件转换到高维空间，并与Toda的核心组件潜在扩散攻击者(LDA)配对。LDA将噪声引入到这个潜在空间内的轮廓中，通过交叉注意机制熟练地将近似引导到目标项目。全局地平线由二部图实现，涉及LDA，并从编码的用户简档特征中派生出来。这使得LDA有可能将生成向外扩展到正在处理的用户功能本身，并弥合扩散的用户功能和目标项目功能之间的差距。与几个SOTA基线相比的广泛实验证明了Toda的有效性。具体的研究利用了户田的精心设计，并强调了高级生成模式在这种背景下的效力。



## **26. PG-Attack: A Precision-Guided Adversarial Attack Framework Against Vision Foundation Models for Autonomous Driving**

PG-Attack：针对自动驾驶视觉基金会模型的精确引导对抗攻击框架 cs.MM

First-Place in the CVPR 2024 Workshop Challenge: Black-box  Adversarial Attacks on Vision Foundation Models

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13111v1) [paper-pdf](http://arxiv.org/pdf/2407.13111v1)

**Authors**: Jiyuan Fu, Zhaoyu Chen, Kaixun Jiang, Haijing Guo, Shuyong Gao, Wenqiang Zhang

**Abstract**: Vision foundation models are increasingly employed in autonomous driving systems due to their advanced capabilities. However, these models are susceptible to adversarial attacks, posing significant risks to the reliability and safety of autonomous vehicles. Adversaries can exploit these vulnerabilities to manipulate the vehicle's perception of its surroundings, leading to erroneous decisions and potentially catastrophic consequences. To address this challenge, we propose a novel Precision-Guided Adversarial Attack (PG-Attack) framework that combines two techniques: Precision Mask Perturbation Attack (PMP-Attack) and Deceptive Text Patch Attack (DTP-Attack). PMP-Attack precisely targets the attack region to minimize the overall perturbation while maximizing its impact on the target object's representation in the model's feature space. DTP-Attack introduces deceptive text patches that disrupt the model's understanding of the scene, further enhancing the attack's effectiveness. Our experiments demonstrate that PG-Attack successfully deceives a variety of advanced multi-modal large models, including GPT-4V, Qwen-VL, and imp-V1. Additionally, we won First-Place in the CVPR 2024 Workshop Challenge: Black-box Adversarial Attacks on Vision Foundation Models and codes are available at https://github.com/fuhaha824/PG-Attack.

摘要: 由于其先进的性能，视觉基础模型越来越多地被用于自动驾驶系统。然而，这些车型容易受到对抗性攻击，对自动驾驶车辆的可靠性和安全性构成重大风险。对手可以利用这些漏洞来操纵车辆对周围环境的感知，从而导致错误的决定和潜在的灾难性后果。为了应对这一挑战，我们提出了一种新的精确制导攻击框架(PG-Attack)，该框架结合了两种技术：精确掩码扰动攻击(PMP-Attack)和欺骗性文本补丁攻击(DTP-Attack)。PMP-Attack精确定位攻击区域，以最小化整体扰动，同时最大化其对目标对象在模型特征空间中的表示的影响。DTP-攻击引入了欺骗性的文本补丁，扰乱了模型对场景的理解，进一步增强了攻击的有效性。实验表明，PG-Attack成功地欺骗了包括GPT-4V、QWEN-VL和IMP-V1在内的多种先进的多模式大型模型。此外，我们还在CVPR2024研讨会挑战赛中获得了第一名：针对Vision Foundation模型和代码的黑盒攻击可在https://github.com/fuhaha824/PG-Attack.上获得



## **27. Krait: A Backdoor Attack Against Graph Prompt Tuning**

Krait：针对图形提示调整的后门攻击 cs.LG

Previously submitted to CCS on 04/29

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13068v1) [paper-pdf](http://arxiv.org/pdf/2407.13068v1)

**Authors**: Ying Song, Rita Singh, Balaji Palanisamy

**Abstract**: Graph prompt tuning has emerged as a promising paradigm to effectively transfer general graph knowledge from pre-trained models to various downstream tasks, particularly in few-shot contexts. However, its susceptibility to backdoor attacks, where adversaries insert triggers to manipulate outcomes, raises a critical concern. We conduct the first study to investigate such vulnerability, revealing that backdoors can disguise benign graph prompts, thus evading detection. We introduce Krait, a novel graph prompt backdoor. Specifically, we propose a simple yet effective model-agnostic metric called label non-uniformity homophily to select poisoned candidates, significantly reducing computational complexity. To accommodate diverse attack scenarios and advanced attack types, we design three customizable trigger generation methods to craft prompts as triggers. We propose a novel centroid similarity-based loss function to optimize prompt tuning for attack effectiveness and stealthiness. Experiments on four real-world graphs demonstrate that Krait can efficiently embed triggers to merely 0.15% to 2% of training nodes, achieving high attack success rates without sacrificing clean accuracy. Notably, in one-to-one and all-to-one attacks, Krait can achieve 100% attack success rates by poisoning as few as 2 and 22 nodes, respectively. Our experiments further show that Krait remains potent across different transfer cases, attack types, and graph neural network backbones. Additionally, Krait can be successfully extended to the black-box setting, posing more severe threats. Finally, we analyze why Krait can evade both classical and state-of-the-art defenses, and provide practical insights for detecting and mitigating this class of attacks.

摘要: 图形提示调优已经成为一种很有前途的范例，可以有效地将一般的图形知识从预先训练的模型转移到各种下游任务，特别是在少镜头的情况下。然而，它容易受到后门攻击，即对手插入触发器来操纵结果，这引发了一个严重的担忧。我们进行了第一次研究来调查这种漏洞，揭示了后门可以伪装良性图形提示，从而逃避检测。我们介绍了Krait，一个新颖的图形提示后门。具体地说，我们提出了一种简单而有效的模型不可知性度量，称为标签非一致性同态来选择中毒候选，显著降低了计算复杂度。为了适应不同的攻击场景和高级攻击类型，我们设计了三种可定制的触发器生成方法来制作提示作为触发器。我们提出了一种新的基于质心相似度的损失函数来优化攻击有效性和隐蔽性的即时调整。在四个真实世界图上的实验表明，Krait可以有效地将触发器嵌入到仅0.15%到2%的训练节点中，在不牺牲干净精度的情况下获得高攻击成功率。值得注意的是，在一对一和全对一攻击中，Krait可以通过分别毒化仅2个和22个节点来实现100%的攻击成功率。我们的实验进一步表明，Krait在不同的传输案例、攻击类型和图神经网络主干上仍然有效。此外，Krait可以成功地扩展到黑箱设置，构成更严重的威胁。最后，我们分析了为什么Krait可以同时躲避经典和最新的防御，并为检测和缓解这类攻击提供了实用的见解。



## **28. Deep Generative Attacks and Countermeasures for Data-Driven Offline Signature Verification**

数据驱动离线签名验证的深度生成攻击和对策 cs.CV

Ten pages, 6 figures, 1 table, Signature verification, Deep  generative models, attacks, generative attack explainability, data-driven  verification system

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2312.00987v2) [paper-pdf](http://arxiv.org/pdf/2312.00987v2)

**Authors**: An Ngo, Rajesh Kumar, Phuong Cao

**Abstract**: This study investigates the vulnerabilities of data-driven offline signature verification (DASV) systems to generative attacks and proposes robust countermeasures. Specifically, we explore the efficacy of Variational Autoencoders (VAEs) and Conditional Generative Adversarial Networks (CGANs) in creating deceptive signatures that challenge DASV systems. Using the Structural Similarity Index (SSIM) to evaluate the quality of forged signatures, we assess their impact on DASV systems built with Xception, ResNet152V2, and DenseNet201 architectures. Initial results showed False Accept Rates (FARs) ranging from 0% to 5.47% across all models and datasets. However, exposure to synthetic signatures significantly increased FARs, with rates ranging from 19.12% to 61.64%. The proposed countermeasure, i.e., retraining the models with real + synthetic datasets, was very effective, reducing FARs between 0% and 0.99%. These findings emphasize the necessity of investigating vulnerabilities in security systems like DASV and reinforce the role of generative methods in enhancing the security of data-driven systems.

摘要: 研究了数据驱动的离线签名验证(DASV)系统对生成性攻击的脆弱性，并提出了稳健的对策。具体地说，我们探索了变分自动编码器(VAE)和条件生成对抗网络(CGAN)在创建挑战DASV系统的欺骗性签名方面的有效性。使用结构相似性指数(SSIM)来评估伪造签名的质量，评估了它们对使用Xception、ResNet152V2和DenseNet201架构构建的DASV系统的影响。初步结果显示，所有模型和数据集的错误接受率(FAR)从0%到5.47%不等。然而，接触合成签名显著增加了FAR，比率从19.12%到61.64%不等。所提出的对策，即用真实+合成数据集重新训练模型是非常有效的，将FAR降低了0%至0.99%。这些发现强调了调查DASV等安全系统中漏洞的必要性，并加强了生成性方法在增强数据驱动系统安全性方面的作用。



## **29. Investigating Adversarial Vulnerability and Implicit Bias through Frequency Analysis**

通过频率分析调查对抗脆弱性和隐性偏见 cs.LG

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2305.15203v2) [paper-pdf](http://arxiv.org/pdf/2305.15203v2)

**Authors**: Lorenzo Basile, Nikos Karantzas, Alberto D'Onofrio, Luca Bortolussi, Alex Rodriguez, Fabio Anselmi

**Abstract**: Despite their impressive performance in classification tasks, neural networks are known to be vulnerable to adversarial attacks, subtle perturbations of the input data designed to deceive the model. In this work, we investigate the relation between these perturbations and the implicit bias of neural networks trained with gradient-based algorithms. To this end, we analyse the network's implicit bias through the lens of the Fourier transform. Specifically, we identify the minimal and most critical frequencies necessary for accurate classification or misclassification respectively for each input image and its adversarially perturbed version, and uncover the correlation among those. To this end, among other methods, we use a newly introduced technique capable of detecting non-linear correlations between high-dimensional datasets. Our results provide empirical evidence that the network bias in Fourier space and the target frequencies of adversarial attacks are highly correlated and suggest new potential strategies for adversarial defence.

摘要: 尽管神经网络在分类任务中的表现令人印象深刻，但众所周知，它很容易受到对抗性攻击，即输入数据的微妙扰动，目的是欺骗模型。在这项工作中，我们研究了这些扰动与用基于梯度的算法训练的神经网络的隐偏差之间的关系。为此，我们通过傅里叶变换的透镜分析了网络的隐含偏差。具体地说，我们分别为每个输入图像及其相反的扰动版本识别准确分类或误分类所需的最小和最关键频率，并揭示这些频率之间的相关性。为此，在其他方法中，我们使用了一种新引入的技术，能够检测高维数据集之间的非线性相关性。我们的结果提供了经验证据，证明了傅立叶空间中的网络偏差与敌方攻击的目标频率高度相关，并为敌方防御提供了新的潜在策略。



## **30. Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models**

静音低语：对语音基础模型的通用声学对抗攻击 cs.CL

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2405.06134v2) [paper-pdf](http://arxiv.org/pdf/2405.06134v2)

**Authors**: Vyas Raina, Rao Ma, Charles McGhee, Kate Knill, Mark Gales

**Abstract**: Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as $\texttt{<|endoftext|>}$, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's $\texttt{<|endoftext|>}$ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively `muting' the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97\% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to `muting' adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.

摘要: 像Whisper这样的大型语音基础模型的最新发展导致它们在许多自动语音识别(ASR)应用中被广泛使用。这些系统在其词汇表中加入了“特殊标记”，如$\exttt{<|endoftext|>}$，以指导其语言生成过程。然而，我们证明了这些令牌可以被敌意攻击利用来操纵模型的行为。我们提出了一种简单而有效的方法来学习Whisper的$\exttt{}$标记的通用声学实现，当该标记被预先添加到任何语音信号时，鼓励模型忽略语音而只转录特殊的标记，从而有效地抑制了模型。我们的实验表明，相同的、通用的0.64秒的对抗性音频片段可以成功地使目标Whisper ASR模型在97%以上的语音样本上静音。此外，我们发现这种普遍的对抗性音频片段经常转移到新的数据集和任务。总体而言，这项工作证明了Whisper模型对“静音”对手攻击的脆弱性，在现实世界中，这种攻击既可以带来风险，也可以带来潜在的好处：例如，攻击可以用来绕过语音调节系统，或者反过来，攻击也可以用来保护私人语音数据。



## **31. Similarity of Neural Architectures using Adversarial Attack Transferability**

使用对抗攻击可转移性的神经架构相似性 cs.LG

ECCV 2024; 35pages, 2.56MB

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2210.11407v4) [paper-pdf](http://arxiv.org/pdf/2210.11407v4)

**Authors**: Jaehui Hwang, Dongyoon Han, Byeongho Heo, Song Park, Sanghyuk Chun, Jong-Seok Lee

**Abstract**: In recent years, many deep neural architectures have been developed for image classification. Whether they are similar or dissimilar and what factors contribute to their (dis)similarities remains curious. To address this question, we aim to design a quantitative and scalable similarity measure between neural architectures. We propose Similarity by Attack Transferability (SAT) from the observation that adversarial attack transferability contains information related to input gradients and decision boundaries widely used to understand model behaviors. We conduct a large-scale analysis on 69 state-of-the-art ImageNet classifiers using our proposed similarity function to answer the question. Moreover, we observe neural architecture-related phenomena using model similarity that model diversity can lead to better performance on model ensembles and knowledge distillation under specific conditions. Our results provide insights into why developing diverse neural architectures with distinct components is necessary.

摘要: 近年来，已经发展了许多用于图像分类的深层神经结构。它们是相似的还是不相似的，以及是什么因素导致了它们(不同)的相似之处，仍然令人好奇。为了解决这个问题，我们的目标是设计一种量化的、可伸缩的神经结构之间的相似性度量。基于对抗性攻击的可移动性包含与输入梯度和决策边界有关的信息，被广泛用于理解模型行为，我们提出了攻击可转移性相似性(SAT)。我们使用我们提出的相似度函数对69个最先进的ImageNet分类器进行了大规模的分析。此外，我们使用模型相似性来观察与神经结构相关的现象，即在特定条件下，模型多样性可以在模型集成和知识提取方面带来更好的性能。我们的结果为为什么开发具有不同组件的不同神经架构提供了洞察力。



## **32. Open-Vocabulary Object Detectors: Robustness Challenges under Distribution Shifts**

开放词汇对象检测器：分布转移下的鲁棒性挑战 cs.CV

14 + 3 single column pages

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2405.14874v3) [paper-pdf](http://arxiv.org/pdf/2405.14874v3)

**Authors**: Prakash Chandra Chhipa, Kanjar De, Meenakshi Subhash Chippa, Rajkumar Saini, Marcus Liwicki

**Abstract**: The challenge of Out-Of-Distribution (OOD) robustness remains a critical hurdle towards deploying deep vision models. Vision-Language Models (VLMs) have recently achieved groundbreaking results. VLM-based open-vocabulary object detection extends the capabilities of traditional object detection frameworks, enabling the recognition and classification of objects beyond predefined categories. Investigating OOD robustness in recent open-vocabulary object detection is essential to increase the trustworthiness of these models. This study presents a comprehensive robustness evaluation of the zero-shot capabilities of three recent open-vocabulary (OV) foundation object detection models: OWL-ViT, YOLO World, and Grounding DINO. Experiments carried out on the robustness benchmarks COCO-O, COCO-DC, and COCO-C encompassing distribution shifts due to information loss, corruption, adversarial attacks, and geometrical deformation, highlighting the challenges of the model's robustness to foster the research for achieving robustness. Source code shall be made available to the research community on GitHub.

摘要: 分布外(OOD)稳健性的挑战仍然是部署深度视觉模型的关键障碍。视觉语言模型(VLM)最近取得了突破性的成果。基于VLM的开放词汇表目标检测扩展了传统目标检测框架的能力，支持对超出预定义类别的目标进行识别和分类。研究开放词汇对象检测中的面向对象设计的健壮性对于提高这些模型的可信性是至关重要的。这项研究对最近三种开放词汇表(OV)基础目标检测模型的零射能力进行了全面的稳健性评估：OWL-VIT、YOLO World和接地Dino。在稳健性基准COCO-O、COCO-DC和COCO-C上进行的实验涵盖了由于信息丢失、损坏、对抗性攻击和几何变形而导致的分布偏移，突显了模型稳健性的挑战，以促进实现稳健性的研究。应在GitHub上向研究社区提供源代码。



## **33. Preventing Catastrophic Overfitting in Fast Adversarial Training: A Bi-level Optimization Perspective**

防止快速对抗训练中的灾难性过度匹配：双层优化的角度 cs.LG

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.12443v1) [paper-pdf](http://arxiv.org/pdf/2407.12443v1)

**Authors**: Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Adversarial training (AT) has become an effective defense method against adversarial examples (AEs) and it is typically framed as a bi-level optimization problem. Among various AT methods, fast AT (FAT), which employs a single-step attack strategy to guide the training process, can achieve good robustness against adversarial attacks at a low cost. However, FAT methods suffer from the catastrophic overfitting problem, especially on complex tasks or with large-parameter models. In this work, we propose a FAT method termed FGSM-PCO, which mitigates catastrophic overfitting by averting the collapse of the inner optimization problem in the bi-level optimization process. FGSM-PCO generates current-stage AEs from the historical AEs and incorporates them into the training process using an adaptive mechanism. This mechanism determines an appropriate fusion ratio according to the performance of the AEs on the training model. Coupled with a loss function tailored to the training framework, FGSM-PCO can alleviate catastrophic overfitting and help the recovery of an overfitted model to effective training. We evaluate our algorithm across three models and three datasets to validate its effectiveness. Comparative empirical studies against other FAT algorithms demonstrate that our proposed method effectively addresses unresolved overfitting issues in existing algorithms.

摘要: 对抗性训练(AT)已经成为对抗对抗性范例(AEs)的一种有效的防御方法，它通常被描述为一个双层优化问题。在众多的AT方法中，FAST AT(FAT)采用单步攻击策略来指导训练过程，能够以较低的代价获得对对手攻击的良好健壮性。然而，FAT方法存在灾难性的过拟合问题，特别是在处理复杂任务或大参数模型时。在这项工作中，我们提出了一种称为FGSM-PCO的FAT方法，它通过避免双层优化过程中内部优化问题的崩溃来减轻灾难性的过拟合。FGSM-PCO从历史的AE生成当前阶段的AE，并使用自适应机制将它们合并到训练过程中。该机制根据AEs在训练模型上的表现来确定合适的融合比例。再加上为训练框架量身定做的损失函数，FGSM-PCO可以缓解灾难性的过拟合，并帮助过度拟合的模型恢复到有效的训练。我们在三个模型和三个数据集上对我们的算法进行了评估，以验证其有效性。与其他FAT算法的对比实验表明，我们提出的方法有效地解决了现有算法中尚未解决的过拟合问题。



## **34. DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks**

发送者：针对补丁攻击的基于扩散的对抗防御 cs.CV

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2306.09124v4) [paper-pdf](http://arxiv.org/pdf/2306.09124v4)

**Authors**: Caixin Kang, Yinpeng Dong, Zhengyi Wang, Shouwei Ruan, Yubo Chen, Hang Su, Xingxing Wei

**Abstract**: Adversarial attacks, particularly patch attacks, pose significant threats to the robustness and reliability of deep learning models. Developing reliable defenses against patch attacks is crucial for real-world applications. This paper introduces DIFFender, a novel defense framework that harnesses the capabilities of a text-guided diffusion model to combat patch attacks. Central to our approach is the discovery of the Adversarial Anomaly Perception (AAP) phenomenon, which empowers the diffusion model to detect and localize adversarial patches through the analysis of distributional discrepancies. DIFFender integrates dual tasks of patch localization and restoration within a single diffusion model framework, utilizing their close interaction to enhance defense efficacy. Moreover, DIFFender utilizes vision-language pre-training coupled with an efficient few-shot prompt-tuning algorithm, which streamlines the adaptation of the pre-trained diffusion model to defense tasks, thus eliminating the need for extensive retraining. Our comprehensive evaluation spans image classification and face recognition tasks, extending to real-world scenarios, where DIFFender shows good robustness against adversarial attacks. The versatility and generalizability of DIFFender are evident across a variety of settings, classifiers, and attack methodologies, marking an advancement in adversarial patch defense strategies.

摘要: 对抗性攻击，尤其是补丁攻击，对深度学习模型的健壮性和可靠性构成了严重的威胁。开发针对补丁攻击的可靠防御对于现实世界的应用程序至关重要。本文介绍了一种新的防御框架DIFFender，它利用文本引导扩散模型的能力来对抗补丁攻击。我们方法的核心是发现了对抗性异常感知(AAP)现象，这使得扩散模型能够通过分析分布差异来检测和定位对抗性补丁。DIFFender在一个扩散模型框架内集成了补丁定位和恢复的双重任务，利用它们的密切交互来提高防御效率。此外，DIFFender利用视觉语言预训练和高效的少镜头提示调整算法，简化了预先训练的扩散模型对防御任务的适应，从而消除了对广泛再训练的需要。我们的综合评估涵盖图像分类和人脸识别任务，并扩展到真实场景，在这些场景中，DIFFender显示出对对手攻击的良好稳健性。DIFFender的多功能性和通用性在各种设置、分类器和攻击方法中都很明显，标志着对抗性补丁防御策略的进步。



## **35. Bribe & Fork: Cheap Bribing Attacks via Forking Threat**

贿赂与叉子：通过叉子威胁进行廉价贿赂攻击 cs.CR

This is a full version of the paper Bribe & Fork: Cheap Bribing  Attacks via Forking Threat which was accepted to AFT'24

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2402.01363v2) [paper-pdf](http://arxiv.org/pdf/2402.01363v2)

**Authors**: Zeta Avarikioti, Paweł Kędzior, Tomasz Lizurej, Tomasz Michalak

**Abstract**: In this work, we reexamine the vulnerability of Payment Channel Networks (PCNs) to bribing attacks, where an adversary incentivizes blockchain miners to deliberately ignore a specific transaction to undermine the punishment mechanism of PCNs. While previous studies have posited a prohibitive cost for such attacks, we show that this cost may be dramatically reduced (to approximately \$125), thereby increasing the likelihood of these attacks. To this end, we introduce Bribe & Fork, a modified bribing attack that leverages the threat of a so-called feather fork which we analyze with a novel formal model for the mining game with forking. We empirically analyze historical data of some real-world blockchain implementations to evaluate the scale of this cost reduction. Our findings shed more light on the potential vulnerability of PCNs and highlight the need for robust solutions.

摘要: 在这项工作中，我们重新审视了支付渠道网络（PCE）对贿赂攻击的脆弱性，即对手激励区块链矿工故意忽略特定交易，以破坏PCE的惩罚机制。虽然之前的研究假设此类攻击的成本过高，但我们表明，这一成本可能会大幅降低（约为125日元），从而增加这些攻击的可能性。为此，我们引入了贿赂和叉子，这是一种改进的贿赂攻击，利用了所谓的羽毛叉的威胁，我们使用带有分叉的采矿游戏的新颖形式模型来分析该攻击。我们通过经验分析一些现实世界区块链实施的历史数据，以评估成本降低的规模。我们的研究结果进一步揭示了多学科网络的潜在脆弱性，并强调了对强大解决方案的需求。



## **36. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

重新审视视觉语言模型的对抗鲁棒性：多模式视角 cs.CV

16 pages, 14 figures

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2404.19287v2) [paper-pdf](http://arxiv.org/pdf/2404.19287v2)

**Authors**: Wanqi Zhou, Shuanghao Bai, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP have shown impressive generalization performance across various downstream tasks, yet they remain vulnerable to adversarial attacks. While prior research has primarily concentrated on improving the adversarial robustness of image encoders to guard against attacks on images, the exploration of text-based and multimodal attacks has largely been overlooked. In this work, we initiate the first known and comprehensive effort to study adapting vision-language models for adversarial robustness under the multimodal attack. Firstly, we introduce a multimodal attack strategy and investigate the impact of different attacks. We then propose a multimodal contrastive adversarial training loss, aligning the clean and adversarial text embeddings with the adversarial and clean visual features, to enhance the adversarial robustness of both image and text encoders of CLIP. Extensive experiments on 15 datasets across two tasks demonstrate that our method significantly improves the adversarial robustness of CLIP. Interestingly, we find that the model fine-tuned against multimodal adversarial attacks exhibits greater robustness than its counterpart fine-tuned solely against image-based attacks, even in the context of image attacks, which may open up new possibilities for enhancing the security of VLMs.

摘要: 像CLIP这样的预先训练的视觉语言模型(VLM)在各种下游任务中表现出令人印象深刻的泛化性能，但它们仍然容易受到对手的攻击。虽然以前的研究主要集中在提高图像编码器的对抗健壮性以防止对图像的攻击，但对基于文本的和多模式攻击的探索在很大程度上被忽视了。在这项工作中，我们启动了第一个已知和全面的努力，以研究适应视觉语言模型的对手在多模式攻击下的稳健性。首先，我们介绍了一种多模式攻击策略，并研究了不同攻击的影响。然后，我们提出了一种多模式对抗性训练损失，将干净和对抗性的文本嵌入与对抗性和干净的视觉特征相结合，以增强CLIP图像和文本编码者的对抗性健壮性。在两个任务的15个数据集上的大量实验表明，我们的方法显著地提高了CLIP的对抗健壮性。有趣的是，我们发现，与仅针对基于图像的攻击进行微调的模型相比，针对多模式攻击进行微调的模型表现出更强的稳健性，甚至在图像攻击的背景下也是如此，这可能为增强VLM的安全性开辟新的可能性。



## **37. Augmented Neural Fine-Tuning for Efficient Backdoor Purification**

增强神经微调以实现高效后门净化 cs.CV

Accepted to ECCV 2024

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.10052v2) [paper-pdf](http://arxiv.org/pdf/2407.10052v2)

**Authors**: Nazmul Karim, Abdullah Al Arafat, Umar Khalid, Zhishan Guo, Nazanin Rahnavard

**Abstract**: Recent studies have revealed the vulnerability of deep neural networks (DNNs) to various backdoor attacks, where the behavior of DNNs can be compromised by utilizing certain types of triggers or poisoning mechanisms. State-of-the-art (SOTA) defenses employ too-sophisticated mechanisms that require either a computationally expensive adversarial search module for reverse-engineering the trigger distribution or an over-sensitive hyper-parameter selection module. Moreover, they offer sub-par performance in challenging scenarios, e.g., limited validation data and strong attacks. In this paper, we propose Neural mask Fine-Tuning (NFT) with an aim to optimally re-organize the neuron activities in a way that the effect of the backdoor is removed. Utilizing a simple data augmentation like MixUp, NFT relaxes the trigger synthesis process and eliminates the requirement of the adversarial search module. Our study further reveals that direct weight fine-tuning under limited validation data results in poor post-purification clean test accuracy, primarily due to overfitting issue. To overcome this, we propose to fine-tune neural masks instead of model weights. In addition, a mask regularizer has been devised to further mitigate the model drift during the purification process. The distinct characteristics of NFT render it highly efficient in both runtime and sample usage, as it can remove the backdoor even when a single sample is available from each class. We validate the effectiveness of NFT through extensive experiments covering the tasks of image classification, object detection, video action recognition, 3D point cloud, and natural language processing. We evaluate our method against 14 different attacks (LIRA, WaNet, etc.) on 11 benchmark data sets such as ImageNet, UCF101, Pascal VOC, ModelNet, OpenSubtitles2012, etc.

摘要: 最近的研究揭示了深度神经网络(DNN)对各种后门攻击的脆弱性，其中DNN的行为可以通过利用某些类型的触发或中毒机制来危害。最先进的(SOTA)防御使用了过于复杂的机制，需要计算昂贵的对抗性搜索模块来对触发分布进行反向工程，或者需要过于敏感的超参数选择模块。此外，它们在具有挑战性的场景中提供了低于平均水平的性能，例如有限的验证数据和强大的攻击。在本文中，我们提出了神经掩码微调(NFT)，目的是以一种消除后门影响的方式来优化重组神经元的活动。利用简单的数据增强，如混合，NFT放松了触发器合成过程，并消除了对敌方搜索模块的要求。我们的研究进一步表明，在有限的验证数据下直接权重微调会导致净化后清洁测试的准确性较差，这主要是由于过度拟合问题。为了克服这一点，我们建议微调神经掩模而不是模型权重。此外，还设计了一种掩膜正则化算法，以进一步缓解纯化过程中的模型漂移。NFT的独特特性使得它在运行时和样本使用方面都非常高效，因为即使每个类只有一个样本可用，它也可以删除后门。通过在图像分类、目标检测、视频动作识别、三维点云和自然语言处理等方面的大量实验，验证了NFT的有效性。我们针对14种不同的攻击(Lira、WaNet等)对我们的方法进行了评估。基于ImageNet、UCF101、Pascal VOC、ModelNet、OpenSubtitles2012等11个基准数据集。



## **38. Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks**

具有对抗性攻击的文本到图像生成中的不对称偏差 cs.LG

camera-ready version

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2312.14440v3) [paper-pdf](http://arxiv.org/pdf/2312.14440v3)

**Authors**: Haz Sameen Shahgir, Xianghao Kong, Greg Ver Steeg, Yue Dong

**Abstract**: The widespread use of Text-to-Image (T2I) models in content generation requires careful examination of their safety, including their robustness to adversarial attacks. Despite extensive research on adversarial attacks, the reasons for their effectiveness remain underexplored. This paper presents an empirical study on adversarial attacks against T2I models, focusing on analyzing factors associated with attack success rates (ASR). We introduce a new attack objective - entity swapping using adversarial suffixes and two gradient-based attack algorithms. Human and automatic evaluations reveal the asymmetric nature of ASRs on entity swap: for example, it is easier to replace "human" with "robot" in the prompt "a human dancing in the rain." with an adversarial suffix, but the reverse replacement is significantly harder. We further propose probing metrics to establish indicative signals from the model's beliefs to the adversarial ASR. We identify conditions that result in a success probability of 60% for adversarial attacks and others where this likelihood drops below 5%.

摘要: 文本到图像(T2I)模型在内容生成中的广泛使用需要仔细检查它们的安全性，包括它们对对手攻击的健壮性。尽管对对抗性攻击进行了广泛的研究，但其有效性的原因仍未得到充分探讨。本文对T2I模型的对抗性攻击进行了实证研究，重点分析了影响攻击成功率的因素。提出了一种新的攻击目标实体交换算法，利用对抗性后缀和两种基于梯度的攻击算法。人工评估和自动评估揭示了ASR在实体交换上的不对称性质：例如，在提示符“a Human in the雨中跳舞”中，更容易将“Human”替换为“bot”。使用对抗性后缀，但反向替换要困难得多。我们进一步提出了探测度量来建立从模型信念到对抗性ASR的指示性信号。我们确定了对抗性攻击成功概率为60%的条件，以及其他可能性降至5%以下的条件。



## **39. Any Target Can be Offense: Adversarial Example Generation via Generalized Latent Infection**

任何目标都可能是攻击性的：通过普遍潜伏感染生成对抗性示例 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.12292v1) [paper-pdf](http://arxiv.org/pdf/2407.12292v1)

**Authors**: Youheng Sun, Shengming Yuan, Xuanhan Wang, Lianli Gao, Jingkuan Song

**Abstract**: Targeted adversarial attack, which aims to mislead a model to recognize any image as a target object by imperceptible perturbations, has become a mainstream tool for vulnerability assessment of deep neural networks (DNNs). Since existing targeted attackers only learn to attack known target classes, they cannot generalize well to unknown classes. To tackle this issue, we propose $\bf{G}$eneralized $\bf{A}$dversarial attac$\bf{KER}$ ($\bf{GAKer}$), which is able to construct adversarial examples to any target class. The core idea behind GAKer is to craft a latently infected representation during adversarial example generation. To this end, the extracted latent representations of the target object are first injected into intermediate features of an input image in an adversarial generator. Then, the generator is optimized to ensure visual consistency with the input image while being close to the target object in the feature space. Since the GAKer is class-agnostic yet model-agnostic, it can be regarded as a general tool that not only reveals the vulnerability of more DNNs but also identifies deficiencies of DNNs in a wider range of classes. Extensive experiments have demonstrated the effectiveness of our proposed method in generating adversarial examples for both known and unknown classes. Notably, compared with other generative methods, our method achieves an approximately $14.13\%$ higher attack success rate for unknown classes and an approximately $4.23\%$ higher success rate for known classes. Our code is available in https://github.com/VL-Group/GAKer.

摘要: 目标对抗攻击旨在通过不可察觉的扰动来误导模型将任何图像识别为目标对象，已成为深度神经网络脆弱性评估的主流工具。由于现有的目标攻击者只学习攻击已知的目标类，因此他们不能很好地泛化到未知的类。为了解决这个问题，我们提出了推广的$\bf{G}$推广的$\bf{A}$dversarialattac$\bf{Ker}$($\bf{GAKer}$)，它能够构造对任何目标类的对抗性例子。GAKer背后的核心思想是在敌意示例生成期间创建一个潜伏感染的表示。为此，首先在对抗性生成器中将提取的目标对象的潜在表示注入到输入图像的中间特征中。然后，对生成器进行优化，以确保与输入图像的视觉一致性，同时在特征空间中接近目标对象。由于GAKer是类不可知的，也是模型不可知的，它可以被视为一个通用工具，不仅可以揭示更多DNN的脆弱性，还可以在更大范围的类中识别DNN的缺陷。大量的实验表明，该方法在生成已知和未知类别的对抗性样本方面是有效的。值得注意的是，与其他产生式方法相比，该方法对未知类的攻击成功率约高14.13美元，对已知类的攻击成功率约高4.23美元。我们的代码在https://github.com/VL-Group/GAKer.中可用



## **40. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

越狱长凳：越狱大型语言模型的开放鲁棒性基准 cs.CR

JailbreakBench v1.0: more attack artifacts, more test-time defenses,  a more accurate jailbreak judge (Llama-3-70B with a custom prompt), a larger  dataset of human preferences for selecting a jailbreak judge (300 examples),  an over-refusal evaluation dataset (100 benign/borderline behaviors), a  semantic refusal judge based on Llama-3-8B

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2404.01318v4) [paper-pdf](http://arxiv.org/pdf/2404.01318v4)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work (Zou et al., 2023; Mazeika et al., 2023, 2024) -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak，这是一个开源基准，具有以下组件：(1)一个不断发展的最先进对手提示的储存库，我们称之为越狱人工制品；(2)一个包括100种行为的越狱数据集--既有原始的，也有源自先前工作的(邹某等人，2023年；Mazeika等人，2023年，2024年)--与开放人工智能的使用政策保持一致；(3)https://github.com/JailbreakBench/jailbreakbench的标准化评估框架，包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)https://jailbreakbench.github.io/的排行榜，跟踪各种LLM的攻击和防御性能。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。



## **41. Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models**

迈向稳健的语义分割模型的可靠评估和快速训练 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2306.12941v2) [paper-pdf](http://arxiv.org/pdf/2306.12941v2)

**Authors**: Francesco Croce, Naman D Singh, Matthias Hein

**Abstract**: Adversarial robustness has been studied extensively in image classification, especially for the $\ell_\infty$-threat model, but significantly less so for related tasks such as object detection and semantic segmentation, where attacks turn out to be a much harder optimization problem than for image classification. We propose several problem-specific novel attacks minimizing different metrics in accuracy and mIoU. The ensemble of our attacks, SEA, shows that existing attacks severely overestimate the robustness of semantic segmentation models. Surprisingly, existing attempts of adversarial training for semantic segmentation models turn out to be weak or even completely non-robust. We investigate why previous adaptations of adversarial training to semantic segmentation failed and show how recently proposed robust ImageNet backbones can be used to obtain adversarially robust semantic segmentation models with up to six times less training time for PASCAL-VOC and the more challenging ADE20k. The associated code and robust models are available at https://github.com/nmndeep/robust-segmentation

摘要: 对抗性稳健性在图像分类中已经得到了广泛的研究，特别是对于威胁模型，但对于目标检测和语义分割等相关任务的研究要少得多，因为在这些任务中，攻击被证明是一个比图像分类更困难的优化问题。我们提出了几种针对特定问题的新攻击，它们在准确率和Miou上最小化不同的度量。我们的SEA攻击集成表明，现有的攻击严重高估了语义分割模型的稳健性。令人惊讶的是，现有的针对语义分割模型的对抗性训练尝试被证明是弱的，甚至是完全不稳健的。我们调查了以前对抗性训练对语义分割的适应失败的原因，并展示了最近提出的健壮ImageNet主干如何用于获得对抗性健壮的语义分割模型，而Pascal-VOC和更具挑战性的ADE20k的训练时间最多减少了六分之一。相关代码和健壮模型可在https://github.com/nmndeep/robust-segmentation上获得



## **42. Variational Randomized Smoothing for Sample-Wise Adversarial Robustness**

样本对抗鲁棒性的变分随机平滑 cs.LG

20 pages, under preparation

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11844v1) [paper-pdf](http://arxiv.org/pdf/2407.11844v1)

**Authors**: Ryo Hase, Ye Wang, Toshiaki Koike-Akino, Jing Liu, Kieran Parsons

**Abstract**: Randomized smoothing is a defensive technique to achieve enhanced robustness against adversarial examples which are small input perturbations that degrade the performance of neural network models. Conventional randomized smoothing adds random noise with a fixed noise level for every input sample to smooth out adversarial perturbations. This paper proposes a new variational framework that uses a per-sample noise level suitable for each input by introducing a noise level selector. Our experimental results demonstrate enhancement of empirical robustness against adversarial attacks. We also provide and analyze the certified robustness for our sample-wise smoothing method.

摘要: 随机平滑是一种防御性技术，旨在针对对抗性示例实现增强的鲁棒性，这些示例是会降低神经网络模型性能的小输入扰动。传统的随机平滑会为每个输入样本添加具有固定噪音水平的随机噪音，以消除对抗性扰动。本文提出了一种新的变分框架，该框架通过引入噪音水平选择器来使用适合每个输入的每样本噪音水平。我们的实验结果表明，针对对抗性攻击的经验鲁棒性增强。我们还提供并分析了我们的样本平滑方法的认证稳健性。



## **43. Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models**

通过对基于LLM的排队模型的对抗攻击探索决策级的鲁棒性 cs.MM

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2405.19802v3) [paper-pdf](http://arxiv.org/pdf/2405.19802v3)

**Authors**: Shuyuan Liu, Jiawei Chen, Shouwei Ruan, Hang Su, Zhaoxia Yin

**Abstract**: Embodied intelligence empowers agents with a profound sense of perception, enabling them to respond in a manner closely aligned with real-world situations. Large Language Models (LLMs) delve into language instructions with depth, serving a crucial role in generating plans for intricate tasks. Thus, LLM-based embodied models further enhance the agent's capacity to comprehend and process information. However, this amalgamation also ushers in new challenges in the pursuit of heightened intelligence. Specifically, attackers can manipulate LLMs to produce irrelevant or even malicious outputs by altering their prompts. Confronted with this challenge, we observe a notable absence of multi-modal datasets essential for comprehensively evaluating the robustness of LLM-based embodied models. Consequently, we construct the Embodied Intelligent Robot Attack Dataset (EIRAD), tailored specifically for robustness evaluation. Additionally, two attack strategies are devised, including untargeted attacks and targeted attacks, to effectively simulate a range of diverse attack scenarios. At the same time, during the attack process, to more accurately ascertain whether our method is successful in attacking the LLM-based embodied model, we devise a new attack success evaluation method utilizing the BLIP2 model. Recognizing the time and cost-intensive nature of the GCG algorithm in attacks, we devise a scheme for prompt suffix initialization based on various target tasks, thus expediting the convergence process. Experimental results demonstrate that our method exhibits a superior attack success rate when targeting LLM-based embodied models, indicating a lower level of decision-level robustness in these models.

摘要: 具身智能使特工具有深刻的感知力，使他们能够以与现实世界情况密切一致的方式做出反应。大型语言模型(LLM)深入研究语言指令，在为复杂任务制定计划方面发挥着至关重要的作用。因此，基于LLM的具体化模型进一步增强了代理理解和处理信息的能力。然而，这种融合也带来了追求高智商的新挑战。具体地说，攻击者可以通过更改提示来操纵LLMS生成无关甚至恶意的输出。面对这一挑战，我们注意到明显缺乏全面评估基于LLM的体现模型的稳健性所必需的多模式数据集。因此，我们构建了专门为健壮性评估量身定做的具体化智能机器人攻击数据集(Eirad)。此外，设计了两种攻击策略，包括非定向攻击和定向攻击，以有效地模拟一系列不同的攻击场景。同时，在攻击过程中，为了更准确地确定我们的方法在攻击基于LLM的体现模型上是否成功，我们设计了一种新的利用BLIP2模型的攻击成功评估方法。考虑到GCG算法在攻击中的时间和成本密集性，我们设计了一种基于不同目标任务的快速后缀初始化方案，从而加快了收敛过程。实验结果表明，我们的方法在攻击基于LLM的具体模型时表现出了较高的攻击成功率，表明这些模型具有较低的决策级健壮性。



## **44. Relaxing Graph Transformers for Adversarial Attacks**

对抗攻击的放松图变形器 cs.LG

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11764v1) [paper-pdf](http://arxiv.org/pdf/2407.11764v1)

**Authors**: Philipp Foth, Lukas Gosch, Simon Geisler, Leo Schwinn, Stephan Günnemann

**Abstract**: Existing studies have shown that Graph Neural Networks (GNNs) are vulnerable to adversarial attacks. Even though Graph Transformers (GTs) surpassed Message-Passing GNNs on several benchmarks, their adversarial robustness properties are unexplored. However, attacking GTs is challenging due to their Positional Encodings (PEs) and special attention mechanisms which can be difficult to differentiate. We overcome these challenges by targeting three representative architectures based on (1) random-walk PEs, (2) pair-wise-shortest-path PEs, and (3) spectral PEs - and propose the first adaptive attacks for GTs. We leverage our attacks to evaluate robustness to (a) structure perturbations on node classification; and (b) node injection attacks for (fake-news) graph classification. Our evaluation reveals that they can be catastrophically fragile and underlines our work's importance and the necessity for adaptive attacks.

摘要: 现有的研究表明，图形神经网络（GNN）容易受到对抗攻击。尽管图形变形器（GT）在多个基准测试中超过了消息传递GNN，但其对抗鲁棒性属性尚未被探索。然而，攻击GT具有挑战性，因为它们的位置编码（PE）和特殊注意机制很难区分。我们通过针对基于（1）随机游走PE、（2）成对最短路径PE和（3）频谱PE的三种代表性架构来克服这些挑战，并提出了针对GT的第一次自适应攻击。我们利用我们的攻击来评估对（a）节点分类的结构扰动的稳健性;和（b）对（假新闻）图分类的节点注入攻击。我们的评估表明，它们可能是灾难性的脆弱性，并强调了我们工作的重要性和自适应攻击的必要性。



## **45. AEMIM: Adversarial Examples Meet Masked Image Modeling**

AEIM：对抗性示例与掩蔽图像建模相结合 cs.CV

Under review of International Journal of Computer Vision (IJCV)

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11537v1) [paper-pdf](http://arxiv.org/pdf/2407.11537v1)

**Authors**: Wenzhao Xiang, Chang Liu, Hang Su, Hongyang Yu

**Abstract**: Masked image modeling (MIM) has gained significant traction for its remarkable prowess in representation learning. As an alternative to the traditional approach, the reconstruction from corrupted images has recently emerged as a promising pretext task. However, the regular corrupted images are generated using generic generators, often lacking relevance to the specific reconstruction task involved in pre-training. Hence, reconstruction from regular corrupted images cannot ensure the difficulty of the pretext task, potentially leading to a performance decline. Moreover, generating corrupted images might introduce an extra generator, resulting in a notable computational burden. To address these issues, we propose to incorporate adversarial examples into masked image modeling, as the new reconstruction targets. Adversarial examples, generated online using only the trained models, can directly aim to disrupt tasks associated with pre-training. Therefore, the incorporation not only elevates the level of challenge in reconstruction but also enhances efficiency, contributing to the acquisition of superior representations by the model. In particular, we introduce a novel auxiliary pretext task that reconstructs the adversarial examples corresponding to the original images. We also devise an innovative adversarial attack to craft more suitable adversarial examples for MIM pre-training. It is noted that our method is not restricted to specific model architectures and MIM strategies, rendering it an adaptable plug-in capable of enhancing all MIM methods. Experimental findings substantiate the remarkable capability of our approach in amplifying the generalization and robustness of existing MIM methods. Notably, our method surpasses the performance of baselines on various tasks, including ImageNet, its variants, and other downstream tasks.

摘要: 蒙面图像建模(MIM)因其在表征学习方面的卓越能力而获得了巨大的吸引力。作为传统方法的替代方法，从损坏的图像中重建图像最近已经成为一项很有前途的借口任务。然而，常规的损坏图像是使用通用生成器生成的，通常与预训练中涉及的特定重建任务缺乏相关性。因此，从常规损坏的图像重建不能确保借口任务的难度，这可能会导致性能下降。此外，生成损坏的图像可能会引入额外的生成器，从而导致显著的计算负担。为了解决这些问题，我们建议将对抗性例子引入到掩蔽图像建模中，作为新的重建目标。仅使用训练过的模型在线生成的对抗性例子可以直接旨在扰乱与预训练相关的任务。因此，合并不仅提高了重建中的挑战水平，而且提高了效率，有助于通过模型获得更好的表示。特别是，我们引入了一种新的辅助借口任务，该任务重建与原始图像相对应的对抗性示例。我们还设计了一种创新的对抗性攻击，为MIM预训练制作更合适的对抗性范例。值得注意的是，我们的方法并不局限于特定的模型体系结构和MIM策略，使其成为一个能够增强所有MIM方法的适应性插件。实验结果证明了该方法在增强现有MIM方法的泛化和健壮性方面具有显著的能力。值得注意的是，我们的方法超过了各种任务的基线性能，包括ImageNet、其变体和其他下游任务。



## **46. Learning on Graphs with Large Language Models(LLMs): A Deep Dive into Model Robustness**

使用大型语言模型（LLM）学习图形：深入研究模型稳健性 cs.LG

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.12068v1) [paper-pdf](http://arxiv.org/pdf/2407.12068v1)

**Authors**: Kai Guo, Zewen Liu, Zhikai Chen, Hongzhi Wen, Wei Jin, Jiliang Tang, Yi Chang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing tasks. Recently, several LLMs-based pipelines have been developed to enhance learning on graphs with text attributes, showcasing promising performance. However, graphs are well-known to be susceptible to adversarial attacks and it remains unclear whether LLMs exhibit robustness in learning on graphs. To address this gap, our work aims to explore the potential of LLMs in the context of adversarial attacks on graphs. Specifically, we investigate the robustness against graph structural and textual perturbations in terms of two dimensions: LLMs-as-Enhancers and LLMs-as-Predictors. Through extensive experiments, we find that, compared to shallow models, both LLMs-as-Enhancers and LLMs-as-Predictors offer superior robustness against structural and textual attacks.Based on these findings, we carried out additional analyses to investigate the underlying causes. Furthermore, we have made our benchmark library openly available to facilitate quick and fair evaluations, and to encourage ongoing innovative research in this field.

摘要: 大型语言模型(LLM)在各种自然语言处理任务中表现出了显著的性能。最近，已经开发了几个基于LLMS的管道来增强对具有文本属性的图形的学习，展示了良好的性能。然而，众所周知，图是容易受到敌意攻击的，而且目前还不清楚LLM是否表现出关于图的学习的健壮性。为了解决这一差距，我们的工作旨在探索LLMS在对抗性攻击图的背景下的潜力。具体地说，我们从两个维度考察了对图结构和文本扰动的稳健性：作为增强器的LLMS和作为预测者的LLMS。通过广泛的实验，我们发现，与浅层模型相比，LLMS-as-Enhancer和LLMS-as-Predicator对结构和文本攻击都具有更好的稳健性。基于这些发现，我们进行了额外的分析，以探讨潜在的原因。此外，我们开放了我们的基准图书馆，以促进快速和公平的评估，并鼓励这一领域正在进行的创新研究。



## **47. Boosting the Transferability of Adversarial Attacks with Global Momentum Initialization**

利用全球动量指标提高对抗性攻击的可转移性 cs.CV

Accepted by Expert Systems with Applications (ESWA)

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2211.11236v3) [paper-pdf](http://arxiv.org/pdf/2211.11236v3)

**Authors**: Jiafeng Wang, Zhaoyu Chen, Kaixun Jiang, Dingkang Yang, Lingyi Hong, Pinxue Guo, Haijing Guo, Wenqiang Zhang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial examples, which are crafted by adding human-imperceptible perturbations to the benign inputs. Simultaneously, adversarial examples exhibit transferability across models, enabling practical black-box attacks. However, existing methods are still incapable of achieving the desired transfer attack performance. In this work, focusing on gradient optimization and consistency, we analyse the gradient elimination phenomenon as well as the local momentum optimum dilemma. To tackle these challenges, we introduce Global Momentum Initialization (GI), providing global momentum knowledge to mitigate gradient elimination. Specifically, we perform gradient pre-convergence before the attack and a global search during this stage. GI seamlessly integrates with existing transfer methods, significantly improving the success rate of transfer attacks by an average of 6.4% under various advanced defense mechanisms compared to the state-of-the-art method. Ultimately, GI demonstrates strong transferability in both image and video attack domains. Particularly, when attacking advanced defense methods in the image domain, it achieves an average attack success rate of 95.4%. The code is available at $\href{https://github.com/Omenzychen/Global-Momentum-Initialization}{https://github.com/Omenzychen/Global-Momentum-Initialization}$.

摘要: 深度神经网络(DNN)很容易受到敌意例子的影响，这些例子是通过在良性输入中添加人类无法察觉的扰动来构建的。同时，敌意例子展示了跨模型的可转移性，从而实现了实际的黑盒攻击。然而，现有的方法仍然不能达到期望的传输攻击性能。在这项工作中，我们以梯度优化和一致性为重点，分析了梯度消除现象以及局部动量最优困境。为了应对这些挑战，我们引入了全局动量初始化(GI)，提供了全局动量知识来缓解梯度消除。具体地说，我们在攻击前执行梯度预收敛，并在此阶段进行全局搜索。GI与现有的传输方式无缝集成，在各种先进的防御机制下，相比最先进的方法，传输攻击成功率平均提升了6.4%。最终，GI在图像和视频攻击领域都表现出了很强的可转移性。特别是在攻击图像域的先进防御手段时，平均攻击成功率达到95.4%。代码可在$\href{https://github.com/Omenzychen/Global-Momentum-Initialization}{https://github.com/Omenzychen/Global-Momentum-Initialization}$.上获得



## **48. Investigating Imperceptibility of Adversarial Attacks on Tabular Data: An Empirical Analysis**

调查表格数据对抗性攻击的不可感知性：实证分析 cs.LG

33 pages

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11463v1) [paper-pdf](http://arxiv.org/pdf/2407.11463v1)

**Authors**: Zhipeng He, Chun Ouyang, Laith Alzubaidi, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks are a potential threat to machine learning models, as they can cause the model to make incorrect predictions by introducing imperceptible perturbations to the input data. While extensively studied in unstructured data like images, their application to structured data like tabular data presents unique challenges due to the heterogeneity and intricate feature interdependencies of tabular data. Imperceptibility in tabular data involves preserving data integrity while potentially causing misclassification, underscoring the need for tailored imperceptibility criteria for tabular data. However, there is currently a lack of standardised metrics for assessing adversarial attacks specifically targeted at tabular data. To address this gap, we derive a set of properties for evaluating the imperceptibility of adversarial attacks on tabular data. These properties are defined to capture seven perspectives of perturbed data: proximity to original inputs, sparsity of alterations, deviation to datapoints in the original dataset, sensitivity of altering sensitive features, immutability of perturbation, feasibility of perturbed values and intricate feature interdepencies among tabular features. Furthermore, we conduct both quantitative empirical evaluation and case-based qualitative examples analysis for seven properties. The evaluation reveals a trade-off between attack success and imperceptibility, particularly concerning proximity, sensitivity, and deviation. Although no evaluated attacks can achieve optimal effectiveness and imperceptibility simultaneously, unbounded attacks prove to be more promised for tabular data in crafting imperceptible adversarial examples. The study also highlights the limitation of evaluated algorithms in controlling sparsity effectively. We suggest incorporating a sparsity metric in future attack design to regulate the number of perturbed features.

摘要: 对抗性攻击是对机器学习模型的潜在威胁，因为它们可以通过向输入数据引入不可察觉的扰动来导致模型做出不正确的预测。虽然它们在图像等非结构化数据中得到了广泛的研究，但由于表格数据的异构性和复杂的特征相互依赖关系，它们在表格数据等结构化数据中的应用面临着独特的挑战。表格数据的不可察觉涉及在可能造成错误分类的同时保持数据的完整性，强调需要为表格数据制定专门的不可察觉标准。然而，目前缺乏评估专门针对表格数据的对抗性攻击的标准化指标。为了弥补这一差距，我们推导了一组用于评估对抗性攻击对表格数据的不可感知性的性质。这些属性被定义为捕捉扰动数据的七个角度：接近原始输入、改变的稀疏性、与原始数据集中的数据点的偏差、改变敏感特征的敏感度、扰动的不变性、扰动值的可行性以及表格特征之间复杂的特征相互依赖。此外，我们还对七个属性进行了定量的实证评估和基于案例的定性实例分析。评估揭示了攻击成功和不可察觉之间的权衡，特别是在接近、敏感度和偏差方面。虽然没有经过评估的攻击可以同时达到最优的有效性和不可见性，但无界攻击被证明在制作不可察觉的对抗性例子时更有希望获得表格数据。该研究还强调了被评估算法在有效控制稀疏性方面的局限性。我们建议在未来的攻击设计中加入稀疏性度量，以规范受干扰特征的数量。



## **49. PromptRobust: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

Entrobust：评估大型语言模型在对抗性预测上的稳健性 cs.CL

Technical report; code is at:  https://github.com/microsoft/promptbench

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2306.04528v5) [paper-pdf](http://arxiv.org/pdf/2306.04528v5)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Yue Zhang, Neil Zhenqiang Gong, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptRobust, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. The adversarial prompts, crafted to mimic plausible user errors like typos or synonyms, aim to evaluate how slight deviations can affect LLM outcomes while maintaining semantic integrity. These prompts are then employed in diverse tasks including sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4,788 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets. Our findings demonstrate that contemporary LLMs are not robust to adversarial prompts. Furthermore, we present a comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users.

摘要: 学术界和工业界对大型语言模型(LLM)的依赖日益增加，这就要求我们必须全面了解它们对提示的稳健性。为了响应这一关键需求，我们引入了PromptRobust，这是一个健壮性基准，旨在衡量LLMS对敌意提示的弹性。这项研究使用了过多的对抗性文本攻击，目标是多个层面的提示：字符、单词、句子和语义。这些对抗性提示旨在模仿打字或同义词等看似合理的用户错误，旨在评估微小的偏差如何在保持语义完整性的同时影响LLM结果。然后，这些提示被用于各种任务，包括情感分析、自然语言推理、阅读理解、机器翻译和数学解题。我们的研究产生了4788个对抗性提示，仔细评估了8个任务和13个数据集。我们的研究结果表明，当代的LLM对敌意提示并不健壮。此外，我们给出了一个全面的分析，以理解即时健壮性及其可转移性背后的奥秘。然后，我们提供了有洞察力的健壮性分析和实用的即时撰写建议，这对研究人员和日常用户都是有益的。



## **50. Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD**

学生看起来很相似：DP-Singapore中的敏感性经常被高估 cs.LG

published in 33rd USENIX Security Symposium

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2307.00310v3) [paper-pdf](http://arxiv.org/pdf/2307.00310v3)

**Authors**: Anvith Thudi, Hengrui Jia, Casey Meehan, Ilia Shumailov, Nicolas Papernot

**Abstract**: Differentially private stochastic gradient descent (DP-SGD) is the canonical approach to private deep learning. While the current privacy analysis of DP-SGD is known to be tight in some settings, several empirical results suggest that models trained on common benchmark datasets leak significantly less privacy for many datapoints. Yet, despite past attempts, a rigorous explanation for why this is the case has not been reached. Is it because there exist tighter privacy upper bounds when restricted to these dataset settings, or are our attacks not strong enough for certain datapoints? In this paper, we provide the first per-instance (i.e., ``data-dependent") DP analysis of DP-SGD. Our analysis captures the intuition that points with similar neighbors in the dataset enjoy better data-dependent privacy than outliers. Formally, this is done by modifying the per-step privacy analysis of DP-SGD to introduce a dependence on the distribution of model updates computed from a training dataset. We further develop a new composition theorem to effectively use this new per-step analysis to reason about an entire training run. Put all together, our evaluation shows that this novel DP-SGD analysis allows us to now formally show that DP-SGD leaks significantly less privacy for many datapoints (when trained on common benchmarks) than the current data-independent guarantee. This implies privacy attacks will necessarily fail against many datapoints if the adversary does not have sufficient control over the possible training datasets.

摘要: 差分私人随机梯度下降(DP-SGD)是私人深度学习的典型方法。虽然目前DP-SGD的隐私分析在某些情况下是严格的，但一些经验结果表明，在公共基准数据集上训练的模型对于许多数据点来说泄露的隐私要少得多。然而，尽管过去曾尝试过，但对于为什么会出现这种情况，还没有达成一个严格的解释。是因为限制到这些数据集设置时存在更严格的隐私上限，还是因为我们的攻击对某些数据点不够强大？在这篇文章中，我们提供了DP-SGD的第一个逐实例(即“数据依赖”)DP分析。我们的分析抓住了这样一种直觉，即数据集中具有相似邻居的点比离群值享有更好的数据依赖隐私。形式上，这是通过修改DP-SGD的每一步隐私分析来实现的，以引入对从训练数据集计算的模型更新的分布的依赖。我们进一步开发了一个新的合成定理，以有效地使用这个新的逐步分析来推理整个训练运行。综上所述，我们的评估表明，这种新颖的DP-SGD分析允许我们现在正式地表明，DP-SGD对于许多数据点(当根据公共基准进行训练时)的隐私泄露显著低于当前的数据独立保证。这意味着如果对手对可能的训练数据集没有足够的控制，针对许多数据点的隐私攻击必然会失败。



