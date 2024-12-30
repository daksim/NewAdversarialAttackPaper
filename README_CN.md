# Latest Adversarial Attack Papers
**update at 2024-12-30 10:07:18**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. An Engorgio Prompt Makes Large Language Model Babble on**

Engorgio提示让大型语言模型胡言乱语 cs.CR

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2412.19394v1) [paper-pdf](http://arxiv.org/pdf/2412.19394v1)

**Authors**: Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Han Qiu, Tianwei Zhang, Hao Wang, Hewu Li, Qi Li, Chao Zhang, Ke Xu

**Abstract**: Auto-regressive large language models (LLMs) have yielded impressive performance in many real-world tasks. However, the new paradigm of these LLMs also exposes novel threats. In this paper, we explore their vulnerability to inference cost attacks, where a malicious user crafts Engorgio prompts to intentionally increase the computation cost and latency of the inference process. We design Engorgio, a novel methodology, to efficiently generate adversarial Engorgio prompts to affect the target LLM's service availability. Engorgio has the following two technical contributions. (1) We employ a parameterized distribution to track LLMs' prediction trajectory. (2) Targeting the auto-regressive nature of LLMs' inference process, we propose novel loss functions to stably suppress the appearance of the <EOS> token, whose occurrence will interrupt the LLM's generation process. We conduct extensive experiments on 13 open-sourced LLMs with parameters ranging from 125M to 30B. The results show that Engorgio prompts can successfully induce LLMs to generate abnormally long outputs (i.e., roughly 2-13$\times$ longer to reach 90%+ of the output length limit) in a white-box scenario and our real-world experiment demonstrates Engergio's threat to LLM service with limited computing resources. The code is accessible at https://github.com/jianshuod/Engorgio-prompt.

摘要: 自回归大型语言模型(LLM)在许多实际任务中取得了令人印象深刻的性能。然而，这些LLM的新范式也暴露了新的威胁。在本文中，我们探讨了它们对推理成本攻击的脆弱性，在这种攻击中，恶意用户手工制作Engorgio会提示故意增加推理过程的计算成本和延迟。我们设计了一种新的方法Engorgio来有效地生成对抗性Engorgio提示，以影响目标LLM的服务可用性。Engorgio有以下两个技术贡献。(1)采用一种参数分布来跟踪LLMS的预测轨迹。(2)针对LLMS推理过程的自回归特性，提出了一种新的损失函数来稳定地抑制<EOS>令牌的出现，它的出现会中断LLM的生成过程。我们在13个开源的LLM上进行了广泛的实验，参数从125M到30B不等。结果表明，在白盒场景中，Engorgio提示能够成功地诱导LLMS产生异常长的输出(即大约2-13$\x$以达到输出长度限制的90%以上)，并且我们的真实世界实验证明了Engergio在有限计算资源的情况下对LLM服务的威胁。该代码可在https://github.com/jianshuod/Engorgio-prompt.上访问



## **2. Quantum-Inspired Weight-Constrained Neural Network: Reducing Variable Numbers by 100x Compared to Standard Neural Networks**

量子启发的权重约束神经网络：与标准神经网络相比将变量数减少100倍 quant-ph

13 pages, 5 figures. Comments are welcome

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19355v1) [paper-pdf](http://arxiv.org/pdf/2412.19355v1)

**Authors**: Shaozhi Li, M Sabbir Salek, Binayyak Roy, Yao Wang, Mashrur Chowdhury

**Abstract**: Although quantum machine learning has shown great promise, the practical application of quantum computers remains constrained in the noisy intermediate-scale quantum era. To take advantage of quantum machine learning, we investigate the underlying mathematical principles of these quantum models and adapt them to classical machine learning frameworks. Specifically, we develop a classical weight-constrained neural network that generates weights based on quantum-inspired insights. We find that this approach can reduce the number of variables in a classical neural network by a factor of 135 while preserving its learnability. In addition, we develop a dropout method to enhance the robustness of quantum machine learning models, which are highly susceptible to adversarial attacks. This technique can also be applied to improve the adversarial resilience of the classical weight-constrained neural network, which is essential for industry applications, such as self-driving vehicles. Our work offers a novel approach to reduce the complexity of large classical neural networks, addressing a critical challenge in machine learning.

摘要: 尽管量子机器学习显示出了巨大的前景，但在嘈杂的中等规模量子时代，量子计算机的实际应用仍然受到限制。为了利用量子机器学习的优势，我们研究了这些量子模型的基本数学原理，并将它们适应于经典的机器学习框架。具体地说，我们开发了一个经典的权重约束神经网络，它基于量子启发的见解生成权重。我们发现，这种方法可以将经典神经网络中的变量数量减少135倍，同时保持其可学习性。此外，我们开发了一种丢弃方法来增强量子机器学习模型的健壮性，这些模型对对手攻击非常敏感。该技术还可以用于提高经典的权值约束神经网络的对抗能力，这对于自动驾驶汽车等工业应用是必不可少的。我们的工作提供了一种新的方法来降低大型经典神经网络的复杂性，解决了机器学习中的一个关键挑战。



## **3. Federated Hybrid Training and Self-Adversarial Distillation: Towards Robust Edge Networks**

联合混合训练和自对抗蒸馏：迈向稳健的边缘网络 cs.CV

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19354v1) [paper-pdf](http://arxiv.org/pdf/2412.19354v1)

**Authors**: Yu Qiao, Apurba Adhikary, Kitae Kim, Eui-Nam Huh, Zhu Han, Choong Seon Hong

**Abstract**: Federated learning (FL) is a distributed training technology that enhances data privacy in mobile edge networks by allowing data owners to collaborate without transmitting raw data to the edge server. However, data heterogeneity and adversarial attacks pose challenges to develop an unbiased and robust global model for edge deployment. To address this, we propose Federated hyBrid Adversarial training and self-adversarial disTillation (FedBAT), a new framework designed to improve both robustness and generalization of the global model. FedBAT seamlessly integrates hybrid adversarial training and self-adversarial distillation into the conventional FL framework from data augmentation and feature distillation perspectives. From a data augmentation perspective, we propose hybrid adversarial training to defend against adversarial attacks by balancing accuracy and robustness through a weighted combination of standard and adversarial training. From a feature distillation perspective, we introduce a novel augmentation-invariant adversarial distillation method that aligns local adversarial features of augmented images with their corresponding unbiased global clean features. This alignment can effectively mitigate bias from data heterogeneity while enhancing both the robustness and generalization of the global model. Extensive experimental results across multiple datasets demonstrate that FedBAT yields comparable or superior performance gains in improving robustness while maintaining accuracy compared to several baselines.

摘要: 联合学习(FL)是一种分布式训练技术，它允许数据所有者在不向边缘服务器传输原始数据的情况下进行协作，从而增强移动边缘网络中的数据隐私。然而，数据异构性和对抗性攻击给开发无偏见和健壮的全球EDGE部署模型带来了挑战。为了解决这一问题，我们提出了联邦混合对抗训练和自我对抗蒸馏(FedBAT)，这是一个新的框架，旨在提高全局模型的健壮性和泛化能力。FedBAT从数据增强和特征提取的角度，将混合对抗性训练和自我对抗性提炼无缝地集成到传统的FL框架中。从数据增强的角度，我们提出了混合对抗性训练，通过标准训练和对抗性训练的加权组合来平衡精确度和稳健性来防御对抗性攻击。从特征提取的角度，我们提出了一种新的增强不变对抗提取方法，该方法将增强图像的局部对抗特征与其相应的无偏全局清洁特征对齐。这种对齐可以有效地减少数据异质性带来的偏差，同时增强全局模型的稳健性和泛化能力。在多个数据集上的广泛实验结果表明，与几个基线相比，FedBAT在提高稳健性同时保持准确性方面获得了类似或更好的性能收益。



## **4. xSRL: Safety-Aware Explainable Reinforcement Learning -- Safety as a Product of Explainability**

xSRL：安全意识的可解释强化学习--安全作为可解释性的产物 cs.AI

Accepted to 24th International Conference on Autonomous Agents and  Multiagent Systems (AAMAS 2025)

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19311v1) [paper-pdf](http://arxiv.org/pdf/2412.19311v1)

**Authors**: Risal Shahriar Shefin, Md Asifur Rahman, Thai Le, Sarra Alqahtani

**Abstract**: Reinforcement learning (RL) has shown great promise in simulated environments, such as games, where failures have minimal consequences. However, the deployment of RL agents in real-world systems such as autonomous vehicles, robotics, UAVs, and medical devices demands a higher level of safety and transparency, particularly when facing adversarial threats. Safe RL algorithms have been developed to address these concerns by optimizing both task performance and safety constraints. However, errors are inevitable, and when they occur, it is essential that the RL agents can also explain their actions to human operators. This makes trust in the safety mechanisms of RL systems crucial for effective deployment. Explainability plays a key role in building this trust by providing clear, actionable insights into the agent's decision-making process, ensuring that safety-critical decisions are well understood. While machine learning (ML) has seen significant advances in interpretability and visualization, explainability methods for RL remain limited. Current tools fail to address the dynamic, sequential nature of RL and its needs to balance task performance with safety constraints over time. The re-purposing of traditional ML methods, such as saliency maps, is inadequate for safety-critical RL applications where mistakes can result in severe consequences. To bridge this gap, we propose xSRL, a framework that integrates both local and global explanations to provide a comprehensive understanding of RL agents' behavior. xSRL also enables developers to identify policy vulnerabilities through adversarial attacks, offering tools to debug and patch agents without retraining. Our experiments and user studies demonstrate xSRL's effectiveness in increasing safety in RL systems, making them more reliable and trustworthy for real-world deployment. Code is available at https://github.com/risal-shefin/xSRL.

摘要: 强化学习(RL)在模拟环境中显示了巨大的前景，例如游戏，在这些环境中，失败的后果最小。然而，在自动驾驶车辆、机器人、无人机和医疗设备等现实世界系统中部署RL代理需要更高水平的安全性和透明度，特别是在面临对手威胁的情况下。安全RL算法已经被开发出来，通过优化任务性能和安全约束来解决这些问题。然而，错误是不可避免的，当它们发生时，RL代理也可以向人类操作员解释他们的行为是至关重要的。这使得对RL系统安全机制的信任对有效部署至关重要。可解释性在建立这种信任方面发挥了关键作用，它为代理人的决策过程提供了清晰、可操作的见解，确保了对安全至关重要的决策得到很好的理解。虽然机器学习(ML)在可解释性和可视化方面取得了重大进展，但用于RL的可解释性方法仍然有限。目前的工具不能解决RL的动态、连续的性质，以及它需要随着时间的推移平衡任务性能和安全约束。传统ML方法的再利用，如显著图，对于安全关键的RL应用是不够的，因为错误可能会导致严重的后果。为了弥合这一差距，我们提出了xSRL，一个整合了局部和全局解释的框架，以提供对RL代理行为的全面理解。XSRL还使开发人员能够通过对抗性攻击识别策略漏洞，提供工具来调试和修补代理，而无需重新培训。我们的实验和用户研究证明了xSRL在提高RL系统安全性方面的有效性，使它们在现实世界的部署中更加可靠和值得信赖。代码可在https://github.com/risal-shefin/xSRL.上找到



## **5. Game-Theoretically Secure Distributed Protocols for Fair Allocation in Coalitional Games**

联盟游戏中公平分配的游戏理论安全分布式协议 cs.GT

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19192v1) [paper-pdf](http://arxiv.org/pdf/2412.19192v1)

**Authors**: T-H. Hubert Chan, Qipeng Kuang, Quan Xue

**Abstract**: We consider game-theoretically secure distributed protocols for coalition games that approximate the Shapley value with small multiplicative error. Since all known existing approximation algorithms for the Shapley value are randomized, it is a challenge to design efficient distributed protocols among mutually distrusted players when there is no central authority to generate unbiased randomness. The game-theoretic notion of maximin security has been proposed to offer guarantees to an honest player's reward even if all other players are susceptible to an adversary.   Permutation sampling is often used in approximation algorithms for the Shapley value. A previous work in 1994 by Zlotkin et al. proposed a simple constant-round distributed permutation generation protocol based on commitment scheme, but it is vulnerable to rushing attacks. The protocol, however, can detect such attacks.   In this work, we model the limited resources of an adversary by a violation budget that determines how many times it can perform such detectable attacks. Therefore, by repeating the number of permutation samples, an honest player's reward can be guaranteed to be close to its Shapley value. We explore both high probability and expected maximin security. We obtain an upper bound on the number of permutation samples for high probability maximin security, even with an unknown violation budget. Furthermore, we establish a matching lower bound for the weaker notion of expected maximin security in specific permutation generation protocols. We have also performed experiments on both synthetic and real data to empirically verify our results.

摘要: 我们考虑在小乘法误差下近似Shapley值的联盟博弈的博弈论安全分布式协议。由于所有已知的Shapley值的近似算法都是随机化的，在没有中央权威机构来产生无偏随机性的情况下，在相互不信任的参与者之间设计有效的分布式协议是一个挑战。博弈论的最大限度安全的概念被提出，以保证诚实的玩家的回报，即使所有其他玩家都容易受到对手的影响。在Shapley值的近似算法中，通常使用置换采样。Zlotkin等人在1994年进行的前一项工作。提出了一种简单的基于承诺方案的恒轮分布式置换生成协议，但该协议容易受到冲刺攻击。然而，该协议可以检测到此类攻击。在这项工作中，我们通过违规预算来模拟对手的有限资源，该预算决定了对手可以执行这种可检测到的攻击的次数。因此，通过重复排列样本的数量，可以保证诚实玩家的奖励接近其Shapley值。我们探讨了高概率安全性和期望最大安全性。我们得到了高概率最大化安全性的置换样本数目的上界，即使在未知的违规预算下也是如此。此外，对于特定置换生成协议中较弱的期望最大安全性概念，我们建立了匹配的下界。我们还在合成数据和真实数据上进行了实验，以经验地验证我们的结果。



## **6. TSCheater: Generating High-Quality Tibetan Adversarial Texts via Visual Similarity**

TSCheater：通过视觉相似性生成高质量的西藏对抗文本 cs.CL

Camera-Ready Version; Accepted at ICASSP 2025

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.02371v3) [paper-pdf](http://arxiv.org/pdf/2412.02371v3)

**Authors**: Xi Cao, Quzong Gesang, Yuan Sun, Nuo Qun, Tashi Nyima

**Abstract**: Language models based on deep neural networks are vulnerable to textual adversarial attacks. While rich-resource languages like English are receiving focused attention, Tibetan, a cross-border language, is gradually being studied due to its abundant ancient literature and critical language strategy. Currently, there are several Tibetan adversarial text generation methods, but they do not fully consider the textual features of Tibetan script and overestimate the quality of generated adversarial texts. To address this issue, we propose a novel Tibetan adversarial text generation method called TSCheater, which considers the characteristic of Tibetan encoding and the feature that visually similar syllables have similar semantics. This method can also be transferred to other abugidas, such as Devanagari script. We utilize a self-constructed Tibetan syllable visual similarity database called TSVSDB to generate substitution candidates and adopt a greedy algorithm-based scoring mechanism to determine substitution order. After that, we conduct the method on eight victim language models. Experimentally, TSCheater outperforms existing methods in attack effectiveness, perturbation magnitude, semantic similarity, visual similarity, and human acceptance. Finally, we construct the first Tibetan adversarial robustness evaluation benchmark called AdvTS, which is generated by existing methods and proofread by humans.

摘要: 基于深度神经网络的语言模型容易受到文本攻击。在英语等资源丰富的语言受到关注的同时，藏语这一跨境语言也因其丰富的古代文献和批评的语言策略而逐渐被研究。目前，有几种藏文对抗性文本生成方法，但它们没有充分考虑藏文的文本特征，高估了生成的对抗性文本的质量。针对这一问题，我们提出了一种新的藏文对抗性文本生成方法TSCheater，该方法考虑了藏文编码的特点和视觉上相似音节具有相似语义的特点。这种方法也可以移植到其他ABUGIDAS，如天成文书。利用自行构建的藏文音节视觉相似度数据库TSVSDB生成替换候选，并采用基于贪婪算法的评分机制确定替换顺序。之后，我们在八个受害者语言模型上进行了该方法。实验结果表明，TSCheater在攻击效果、扰动幅度、语义相似度、视觉相似度和人类接受度等方面均优于现有方法。最后，我们构建了第一个藏文对手健壮性评估基准ADVTS，该基准由现有方法生成并由人工校对。



## **7. DiffPatch: Generating Customizable Adversarial Patches using Diffusion Model**

迪夫补丁：使用扩散模型生成可定制的对抗补丁 cs.CV

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.01440v2) [paper-pdf](http://arxiv.org/pdf/2412.01440v2)

**Authors**: Zhixiang Wang, Guangnan Ye, Xiaosen Wang, Siheng Chen, Zhibo Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Physical adversarial patches printed on clothing can easily allow individuals to evade person detectors. However, most existing adversarial patch generation methods prioritize attack effectiveness over stealthiness, resulting in patches that are aesthetically unpleasing. Although existing methods using generative adversarial networks or diffusion models can produce more natural-looking patches, they often struggle to balance stealthiness with attack effectiveness and lack flexibility for user customization. To address these challenges, we propose a novel diffusion-based customizable patch generation framework termed DiffPatch, specifically tailored for creating naturalistic and customizable adversarial patches. Our approach enables users to utilize a reference image as the source, rather than starting from random noise, and incorporates masks to craft naturalistic patches of various shapes, not limited to squares. To prevent the original semantics from being lost during the diffusion process, we employ Null-text inversion to map random noise samples to a single input image and generate patches through Incomplete Diffusion Optimization (IDO). Notably, while maintaining a natural appearance, our method achieves a comparable attack performance to state-of-the-art non-naturalistic patches when using similarly sized attacks. Using DiffPatch, we have created a physical adversarial T-shirt dataset, AdvPatch-1K, specifically targeting YOLOv5s. This dataset includes over a thousand images across diverse scenarios, validating the effectiveness of our attack in real-world environments. Moreover, it provides a valuable resource for future research.

摘要: 衣服上印有敌意的物理补丁可以很容易地让个人躲避个人探测器。然而，大多数现有的对抗性补丁生成方法将攻击效率置于隐蔽性之上，导致生成的补丁在美学上令人不快。虽然现有的方法使用生成性对抗网络或扩散模型可以产生看起来更自然的补丁，但它们往往难以平衡隐蔽性和攻击有效性，并且缺乏用户定制的灵活性。为了应对这些挑战，我们提出了一种新的基于扩散的可定制补丁生成框架DiffPatch，该框架专门用于创建自然的和可定制的对抗性补丁。我们的方法使用户能够利用参考图像作为源，而不是从随机噪声开始，并结合蒙版来制作各种形状的自然斑块，而不限于正方形。为了避免在扩散过程中丢失原始语义，我们使用空文本反转将随机噪声样本映射到单一输入图像，并通过不完全扩散优化(IDO)生成斑块。值得注意的是，在保持自然外观的同时，我们的方法在使用类似大小的攻击时，实现了与最先进的非自然主义补丁相当的攻击性能。使用DiffPatch，我们已经创建了一个物理对手T恤数据集AdvPatch-1K，专门针对YOLOv5。该数据集包括1000多张不同场景的图像，验证了我们的攻击在真实环境中的有效性。此外，它还为今后的研究提供了宝贵的资源。



## **8. Provable Robust Saliency-based Explanations**

可证明的稳健基于显着性的解释 cs.LG

Accepted to NeurIPS 2024

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2212.14106v4) [paper-pdf](http://arxiv.org/pdf/2212.14106v4)

**Authors**: Chao Chen, Chenghua Guo, Rufeng Chen, Guixiang Ma, Ming Zeng, Xiangwen Liao, Xi Zhang, Sihong Xie

**Abstract**: To foster trust in machine learning models, explanations must be faithful and stable for consistent insights. Existing relevant works rely on the $\ell_p$ distance for stability assessment, which diverges from human perception. Besides, existing adversarial training (AT) associated with intensive computations may lead to an arms race. To address these challenges, we introduce a novel metric to assess the stability of top-$k$ salient features. We introduce R2ET which trains for stable explanation by efficient and effective regularizer, and analyze R2ET by multi-objective optimization to prove numerical and statistical stability of explanations. Moreover, theoretical connections between R2ET and certified robustness justify R2ET's stability in all attacks. Extensive experiments across various data modalities and model architectures show that R2ET achieves superior stability against stealthy attacks, and generalizes effectively across different explanation methods.

摘要: 为了促进对机器学习模型的信任，解释必须忠实且稳定，以获得一致的见解。现有的相关作品依赖于$\ell_p$距离进行稳定性评估，这与人类的感知存在分歧。此外，与密集计算相关的现有对抗训练（AT）可能会导致军备竞赛。为了应对这些挑战，我们引入了一种新颖的指标来评估顶级$k$显着特征的稳定性。我们引入R2 ET，通过高效且有效的正规化器训练稳定的解释，并通过多目标优化分析R2 ET，以证明解释的数字和统计稳定性。此外，R2 ET和认证稳健性之间的理论联系证明了R2 ET在所有攻击中的稳定性。跨各种数据模式和模型架构的广泛实验表明，R2 ET针对隐形攻击实现了卓越的稳定性，并在不同的解释方法中有效推广。



## **9. Imperceptible Adversarial Attacks on Point Clouds Guided by Point-to-Surface Field**

点到表面场引导下的点云不可感知的对抗攻击 cs.CV

Accepted by ICASSP 2025

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19015v1) [paper-pdf](http://arxiv.org/pdf/2412.19015v1)

**Authors**: Keke Tang, Weiyao Ke, Weilong Peng, Xiaofei Wang, Ziyong Du, Zhize Wu, Peican Zhu, Zhihong Tian

**Abstract**: Adversarial attacks on point clouds are crucial for assessing and improving the adversarial robustness of 3D deep learning models. Traditional solutions strictly limit point displacement during attacks, making it challenging to balance imperceptibility with adversarial effectiveness. In this paper, we attribute the inadequate imperceptibility of adversarial attacks on point clouds to deviations from the underlying surface. To address this, we introduce a novel point-to-surface (P2S) field that adjusts adversarial perturbation directions by dragging points back to their original underlying surface. Specifically, we use a denoising network to learn the gradient field of the logarithmic density function encoding the shape's surface, and apply a distance-aware adjustment to perturbation directions during attacks, thereby enhancing imperceptibility. Extensive experiments show that adversarial attacks guided by our P2S field are more imperceptible, outperforming state-of-the-art methods.

摘要: 对点云的对抗攻击对于评估和提高3D深度学习模型的对抗稳健性至关重要。传统的解决方案严格限制攻击期间的点位移，使得平衡不可感知性与对抗有效性变得具有挑战性。在本文中，我们将点云对抗攻击的不可感知性不足归因于与底层表面的偏差。为了解决这个问题，我们引入了一种新型的点到面（P2 S）场，该场通过将点拖回其原始底层表面来调整对抗扰动方向。具体来说，我们使用去噪网络来学习编码形状表面的log密度函数的梯度场，并在攻击期间对扰动方向进行距离感知调整，从而增强不可感知性。大量实验表明，由我们的P2S领域引导的对抗攻击更难以察觉，性能优于最先进的方法。



## **10. Bridging Interpretability and Robustness Using LIME-Guided Model Refinement**

使用LIME引导的模型细化来弥合可解释性和鲁棒性 cs.LG

10 pages, 15 figures

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18952v1) [paper-pdf](http://arxiv.org/pdf/2412.18952v1)

**Authors**: Navid Nayyem, Abdullah Rakin, Longwei Wang

**Abstract**: This paper explores the intricate relationship between interpretability and robustness in deep learning models. Despite their remarkable performance across various tasks, deep learning models often exhibit critical vulnerabilities, including susceptibility to adversarial attacks, over-reliance on spurious correlations, and a lack of transparency in their decision-making processes. To address these limitations, we propose a novel framework that leverages Local Interpretable Model-Agnostic Explanations (LIME) to systematically enhance model robustness. By identifying and mitigating the influence of irrelevant or misleading features, our approach iteratively refines the model, penalizing reliance on these features during training. Empirical evaluations on multiple benchmark datasets demonstrate that LIME-guided refinement not only improves interpretability but also significantly enhances resistance to adversarial perturbations and generalization to out-of-distribution data.

摘要: 本文探讨了深度学习模型中可解释性和稳健性之间的复杂关系。尽管深度学习模型在各种任务中表现出色，但它们往往表现出严重的漏洞，包括容易受到对抗攻击、过度依赖虚假相关性以及决策过程缺乏透明度。为了解决这些限制，我们提出了一种新颖的框架，该框架利用本地可解释模型不可知解释（LIME）来系统性地增强模型稳健性。通过识别和减轻不相关或误导性特征的影响，我们的方法迭代地完善模型，惩罚训练期间对这些特征的依赖。对多个基准数据集的经验评估表明，LIME引导的细化不仅提高了可解释性，而且显着增强了对对抗性扰动的抵抗力和对非分布数据的概括。



## **11. Improving Integrated Gradient-based Transferable Adversarial Examples by Refining the Integration Path**

通过完善集成路径改进基于集成对象的可转移对抗示例 cs.CR

Accepted by AAAI 2025

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18844v1) [paper-pdf](http://arxiv.org/pdf/2412.18844v1)

**Authors**: Yuchen Ren, Zhengyu Zhao, Chenhao Lin, Bo Yang, Lu Zhou, Zhe Liu, Chao Shen

**Abstract**: Transferable adversarial examples are known to cause threats in practical, black-box attack scenarios. A notable approach to improving transferability is using integrated gradients (IG), originally developed for model interpretability. In this paper, we find that existing IG-based attacks have limited transferability due to their naive adoption of IG in model interpretability. To address this limitation, we focus on the IG integration path and refine it in three aspects: multiplicity, monotonicity, and diversity, supported by theoretical analyses. We propose the Multiple Monotonic Diversified Integrated Gradients (MuMoDIG) attack, which can generate highly transferable adversarial examples on different CNN and ViT models and defenses. Experiments validate that MuMoDIG outperforms the latest IG-based attack by up to 37.3\% and other state-of-the-art attacks by 8.4\%. In general, our study reveals that migrating established techniques to improve transferability may require non-trivial efforts. Code is available at \url{https://github.com/RYC-98/MuMoDIG}.

摘要: 众所周知，可转移的对抗性示例在实际的黑盒攻击场景中会造成威胁。提高可转移性的一个值得注意的方法是使用集成梯度(IG)，它最初是为模型的可解释性而开发的。在本文中，我们发现现有的基于IG的攻击由于在模型可解释性方面对IG的天真采用而具有有限的可转移性。为了解决这一局限性，我们聚焦于IG整合路径，并在理论分析的支持下，从多样性、单调性和多样性三个方面对其进行了提炼。提出了多重单调多元集成梯度(MuMoDIG)攻击，该攻击可以在不同的CNN和VIT模型和防御上产生高度可移植的敌意实例。实验证明，MuMoDIG的性能比最新的基于IG的攻击高出37.3\%，比其他最新的攻击高出8.4\%。总体而言，我们的研究表明，移植已有的技术以提高可转移性可能需要付出巨大的努力。代码位于\url{https://github.com/RYC-98/MuMoDIG}.



## **12. Distortion-Aware Adversarial Attacks on Bounding Boxes of Object Detectors**

对物体检测器边界盒的失真感知对抗攻击 cs.CV

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18815v1) [paper-pdf](http://arxiv.org/pdf/2412.18815v1)

**Authors**: Pham Phuc, Son Vuong, Khang Nguyen, Tuan Dang

**Abstract**: Deep learning-based object detection has become ubiquitous in the last decade due to its high accuracy in many real-world applications. With this growing trend, these models are interested in being attacked by adversaries, with most of the results being on classifiers, which do not match the context of practical object detection. In this work, we propose a novel method to fool object detectors, expose the vulnerability of state-of-the-art detectors, and promote later works to build more robust detectors to adversarial examples. Our method aims to generate adversarial images by perturbing object confidence scores during training, which is crucial in predicting confidence for each class in the testing phase. Herein, we provide a more intuitive technique to embed additive noises based on detected objects' masks and the training loss with distortion control over the original image by leveraging the gradient of iterative images. To verify the proposed method, we perform adversarial attacks against different object detectors, including the most recent state-of-the-art models like YOLOv8, Faster R-CNN, RetinaNet, and Swin Transformer. We also evaluate our technique on MS COCO 2017 and PASCAL VOC 2012 datasets and analyze the trade-off between success attack rate and image distortion. Our experiments show that the achievable success attack rate is up to $100$\% and up to $98$\% when performing white-box and black-box attacks, respectively. The source code and relevant documentation for this work are available at the following link: https://github.com/anonymous20210106/attack_detector

摘要: 在过去的十年里，基于深度学习的目标检测已经变得无处不在，因为它在许多实际应用中都具有很高的准确率。随着这一趋势的发展，这些模型对受到对手的攻击感兴趣，大多数结果都是基于分类器的，这与实际目标检测的上下文不匹配。在这项工作中，我们提出了一种新的方法来愚弄对象检测器，揭露现有检测器的脆弱性，并将后来的工作推广到构建更健壮的检测器。我们的方法旨在通过在训练过程中扰动对象置信度分数来生成对抗性图像，这对于在测试阶段预测每个类别的置信度是至关重要的。在这里，我们提供了一种更直观的技术，通过利用迭代图像的梯度来嵌入基于检测对象的掩模和训练损失的加性噪声，并对原始图像进行失真控制。为了验证所提出的方法，我们对不同的对象检测器进行了对抗性攻击，包括最新的模型，如YOLOv8，FASTER R-CNN，RetinaNet和Swin Transformer。我们还在MS Coco 2017和Pascal VOC 2012数据集上对我们的技术进行了评估，并分析了攻击成功率和图像失真之间的权衡。实验表明，白盒攻击和黑盒攻击的成功率分别达到100美元和98美元。这项工作的源代码和相关文档可在以下链接中找到：https://github.com/anonymous20210106/attack_detector



## **13. Protective Perturbations against Unauthorized Data Usage in Diffusion-based Image Generation**

基于扩散的图像生成中针对未经授权的数据使用的保护性扰动 cs.CV

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18791v1) [paper-pdf](http://arxiv.org/pdf/2412.18791v1)

**Authors**: Sen Peng, Jijia Yang, Mingyue Wang, Jianfei He, Xiaohua Jia

**Abstract**: Diffusion-based text-to-image models have shown immense potential for various image-related tasks. However, despite their prominence and popularity, customizing these models using unauthorized data also brings serious privacy and intellectual property issues. Existing methods introduce protective perturbations based on adversarial attacks, which are applied to the customization samples. In this systematization of knowledge, we present a comprehensive survey of protective perturbation methods designed to prevent unauthorized data usage in diffusion-based image generation. We establish the threat model and categorize the downstream tasks relevant to these methods, providing a detailed analysis of their designs. We also propose a completed evaluation framework for these perturbation techniques, aiming to advance research in this field.

摘要: 基于扩散的文本到图像模型在各种图像相关任务中表现出了巨大的潜力。然而，尽管它们引人注目且受欢迎，但使用未经授权的数据定制这些模型也带来了严重的隐私和知识产权问题。现有方法引入基于对抗攻击的保护性扰动，并应用于定制样本。在知识的系统化中，我们对旨在防止在基于扩散的图像生成中未经授权使用数据的保护性扰动方法进行了全面调查。我们建立威胁模型并对与这些方法相关的下游任务进行分类，并对其设计进行详细分析。我们还为这些扰动技术提出了一个完整的评估框架，旨在推进该领域的研究。



## **14. Attack-in-the-Chain: Bootstrapping Large Language Models for Attacks Against Black-box Neural Ranking Models**

链中攻击：引导大型语言模型来攻击黑匣子神经排名模型 cs.IR

Accepted by AAAI25

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18770v1) [paper-pdf](http://arxiv.org/pdf/2412.18770v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Neural ranking models (NRMs) have been shown to be highly effective in terms of retrieval performance. Unfortunately, they have also displayed a higher degree of sensitivity to attacks than previous generation models. To help expose and address this lack of robustness, we introduce a novel ranking attack framework named Attack-in-the-Chain, which tracks interactions between large language models (LLMs) and NRMs based on chain-of-thought (CoT) prompting to generate adversarial examples under black-box settings. Our approach starts by identifying anchor documents with higher ranking positions than the target document as nodes in the reasoning chain. We then dynamically assign the number of perturbation words to each node and prompt LLMs to execute attacks. Finally, we verify the attack performance of all nodes at each reasoning step and proceed to generate the next reasoning step. Empirical results on two web search benchmarks show the effectiveness of our method.

摘要: 神经排名模型（NRM）已被证明在检索性能方面非常有效。不幸的是，它们还表现出比前一代模型更高的攻击敏感性。为了帮助揭露和解决这种缺乏稳健性的问题，我们引入了一种名为Chain Attack-in-the-Chain的新型排名攻击框架，该框架基于思想链（CoT）来跟踪大型语言模型（LLM）和NRM之间的交互，以在黑匣子设置下生成对抗性示例。我们的方法首先将排名位置高于目标文档的锚文档识别为推理链中的节点。然后，我们动态地为每个节点分配扰动字的数量，并提示LLM执行攻击。最后，我们在每个推理步骤中验证所有节点的攻击性能，并继续生成下一个推理步骤。两个网络搜索基准的经验结果表明了我们方法的有效性。



## **15. Token Highlighter: Inspecting and Mitigating Jailbreak Prompts for Large Language Models**

Token Highliter：检查和缓解大型语言模型的越狱承诺 cs.CR

Accepted by AAAI 2025. Project page:  https://huggingface.co/spaces/TrustSafeAI/Token-Highlighter

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18171v2) [paper-pdf](http://arxiv.org/pdf/2412.18171v2)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into services such as ChatGPT to provide responses to user queries. To mitigate potential harm and prevent misuse, there have been concerted efforts to align the LLMs with human values and legal compliance by incorporating various techniques, such as Reinforcement Learning from Human Feedback (RLHF), into the training of the LLMs. However, recent research has exposed that even aligned LLMs are susceptible to adversarial manipulations known as Jailbreak Attacks. To address this challenge, this paper proposes a method called Token Highlighter to inspect and mitigate the potential jailbreak threats in the user query. Token Highlighter introduced a concept called Affirmation Loss to measure the LLM's willingness to answer the user query. It then uses the gradient of Affirmation Loss for each token in the user query to locate the jailbreak-critical tokens. Further, Token Highlighter exploits our proposed Soft Removal technique to mitigate the jailbreak effects of critical tokens via shrinking their token embeddings. Experimental results on two aligned LLMs (LLaMA-2 and Vicuna-V1.5) demonstrate that the proposed method can effectively defend against a variety of Jailbreak Attacks while maintaining competent performance on benign questions of the AlpacaEval benchmark. In addition, Token Highlighter is a cost-effective and interpretable defense because it only needs to query the protected LLM once to compute the Affirmation Loss and can highlight the critical tokens upon refusal.

摘要: 大型语言模型(LLM)越来越多地被集成到ChatGPT等服务中，以提供对用户查询的响应。为减少潜在危害和防止滥用，已作出协调一致的努力，通过将从人类反馈中强化学习(RLHF)等各种技术纳入LLMS的培训，使LLMS与人的价值观和法律合规保持一致。然而，最近的研究表明，即使是对准的LLM也容易受到称为越狱攻击的对抗性操纵的影响。为了应对这一挑战，本文提出了一种称为令牌荧光的方法来检测和缓解用户查询中潜在的越狱威胁。令牌亮点引入了一个名为肯定损失的概念，以衡量LLM回答用户问题的意愿。然后，它使用用户查询中每个令牌的确认损失梯度来定位越狱关键令牌。此外，令牌荧光利用我们提出的软删除技术，通过缩小关键令牌的令牌嵌入来缓解关键令牌的越狱影响。在两个对齐的LLMS(Llama-2和Vicuna-V1.5)上的实验结果表明，该方法可以有效地防御各种越狱攻击，同时保持在AlpacaEval基准测试的良性问题上的良好性能。此外，令牌加亮器是一种经济高效且可解释的防御方案，因为它只需查询受保护的LLM一次即可计算肯定损失，并且可以在拒绝时突出显示关键令牌。



## **16. Evaluating the Adversarial Robustness of Detection Transformers**

评估检测转换器的对抗鲁棒性 cs.CV

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18718v1) [paper-pdf](http://arxiv.org/pdf/2412.18718v1)

**Authors**: Amirhossein Nazeri, Chunheng Zhao, Pierluigi Pisu

**Abstract**: Robust object detection is critical for autonomous driving and mobile robotics, where accurate detection of vehicles, pedestrians, and obstacles is essential for ensuring safety. Despite the advancements in object detection transformers (DETRs), their robustness against adversarial attacks remains underexplored. This paper presents a comprehensive evaluation of DETR model and its variants under both white-box and black-box adversarial attacks, using the MS-COCO and KITTI datasets to cover general and autonomous driving scenarios. We extend prominent white-box attack methods (FGSM, PGD, and CW) to assess DETR vulnerability, demonstrating that DETR models are significantly susceptible to adversarial attacks, similar to traditional CNN-based detectors. Our extensive transferability analysis reveals high intra-network transferability among DETR variants, but limited cross-network transferability to CNN-based models. Additionally, we propose a novel untargeted attack designed specifically for DETR, exploiting its intermediate loss functions to induce misclassification with minimal perturbations. Visualizations of self-attention feature maps provide insights into how adversarial attacks affect the internal representations of DETR models. These findings reveal critical vulnerabilities in detection transformers under standard adversarial attacks, emphasizing the need for future research to enhance the robustness of transformer-based object detectors in safety-critical applications.

摘要: 稳健的目标检测对于自动驾驶和移动机器人至关重要，在这些领域，对车辆、行人和障碍物的准确检测对于确保安全至关重要。尽管目标检测转换器(DETR)有了很大的进步，但它们对对手攻击的健壮性仍然没有得到充分的研究。本文使用MS-COCO和KITTI数据集对白盒和黑盒对抗攻击下的DETR模型及其变体进行了综合评估，以涵盖一般和自动驾驶场景。我们扩展了著名的白盒攻击方法(FGSM、PGD和CW)来评估DETR漏洞，表明DETR模型与传统的基于CNN的检测器相似，非常容易受到对抗性攻击。我们广泛的可转移性分析表明，DETR变体之间的网络内可转移性很高，但对基于CNN的模型的跨网络可转移性有限。此外，我们还提出了一种新的针对DETR的非目标攻击，利用其中间损失函数以最小的扰动来诱导错误分类。自我注意特征图的可视化提供了对对抗性攻击如何影响DETR模型的内部表示的洞察。这些发现揭示了标准对抗性攻击下检测变压器的关键漏洞，强调了未来研究的必要性，以增强基于变压器的对象检测器在安全关键应用中的稳健性。



## **17. SurvAttack: Black-Box Attack On Survival Models through Ontology-Informed EHR Perturbation**

SurvAttack：通过基于实体的EHR扰动对生存模型进行黑匣子攻击 cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18706v1) [paper-pdf](http://arxiv.org/pdf/2412.18706v1)

**Authors**: Mohsen Nayebi Kerdabadi, Arya Hadizadeh Moghaddam, Bin Liu, Mei Liu, Zijun Yao

**Abstract**: Survival analysis (SA) models have been widely studied in mining electronic health records (EHRs), particularly in forecasting the risk of critical conditions for prioritizing high-risk patients. However, their vulnerability to adversarial attacks is much less explored in the literature. Developing black-box perturbation algorithms and evaluating their impact on state-of-the-art survival models brings two benefits to medical applications. First, it can effectively evaluate the robustness of models in pre-deployment testing. Also, exploring how subtle perturbations would result in significantly different outcomes can provide counterfactual insights into the clinical interpretation of model prediction. In this work, we introduce SurvAttack, a novel black-box adversarial attack framework leveraging subtle clinically compatible, and semantically consistent perturbations on longitudinal EHRs to degrade survival models' predictive performance. We specifically develop a greedy algorithm to manipulate medical codes with various adversarial actions throughout a patient's medical history. Then, these adversarial actions are prioritized using a composite scoring strategy based on multi-aspect perturbation quality, including saliency, perturbation stealthiness, and clinical meaningfulness. The proposed adversarial EHR perturbation algorithm is then used in an efficient SA-specific strategy to attack a survival model when estimating the temporal ranking of survival urgency for patients. To demonstrate the significance of our work, we conduct extensive experiments, including baseline comparisons, explainability analysis, and case studies. The experimental results affirm our research's effectiveness in illustrating the vulnerabilities of patient survival models, model interpretation, and ultimately contributing to healthcare quality.

摘要: 生存分析(SA)模型在挖掘电子健康记录(EHR)中得到了广泛的研究，特别是在预测危重疾病的风险以优先处理高危患者方面。然而，它们在对抗性攻击中的脆弱性在文献中很少被探讨。开发黑盒扰动算法并评估它们对最先进的生存模型的影响为医学应用带来了两个好处。首先，它可以在部署前测试中有效地评估模型的稳健性。此外，探索细微的扰动如何导致显著不同的结果，可以为模型预测的临床解释提供反事实的见解。在这项工作中，我们引入了SurvAttack，一个新的黑盒对抗性攻击框架，利用对纵向EHR的微妙的临床兼容和语义一致的扰动来降低生存模型的预测性能。我们专门开发了一种贪婪的算法来操纵医疗代码，在患者的病史上采取各种敌对行动。然后，使用基于多方面扰动质量(包括显著程度、扰动隐蔽性和临床意义)的综合评分策略对这些对抗性动作进行优先排序。然后，将所提出的对抗性EHR扰动算法用于有效的SA特定策略中，在估计患者生存紧迫性的时间排序时攻击生存模型。为了证明我们工作的重要性，我们进行了广泛的实验，包括基线比较、可解释性分析和案例研究。实验结果肯定了我们的研究在阐明患者生存模型的脆弱性、模型解释以及最终对医疗质量做出贡献方面的有效性。



## **18. Adversarial Attack Against Images Classification based on Generative Adversarial Networks**

基于生成对抗网络的图像分类对抗攻击 cs.CV

7 pages, 6 figures

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.16662v2) [paper-pdf](http://arxiv.org/pdf/2412.16662v2)

**Authors**: Yahe Yang

**Abstract**: Adversarial attacks on image classification systems have always been an important problem in the field of machine learning, and generative adversarial networks (GANs), as popular models in the field of image generation, have been widely used in various novel scenarios due to their powerful generative capabilities. However, with the popularity of generative adversarial networks, the misuse of fake image technology has raised a series of security problems, such as malicious tampering with other people's photos and videos, and invasion of personal privacy. Inspired by the generative adversarial networks, this work proposes a novel adversarial attack method, aiming to gain insight into the weaknesses of the image classification system and improve its anti-attack ability. Specifically, the generative adversarial networks are used to generate adversarial samples with small perturbations but enough to affect the decision-making of the classifier, and the adversarial samples are generated through the adversarial learning of the training generator and the classifier. From extensive experiment analysis, we evaluate the effectiveness of the method on a classical image classification dataset, and the results show that our model successfully deceives a variety of advanced classifiers while maintaining the naturalness of adversarial samples.

摘要: 针对图像分类系统的对抗性攻击一直是机器学习领域的一个重要问题，而生成性对抗性网络(GANS)作为图像生成领域的热门模型，由于其强大的生成能力而被广泛应用于各种新颖的场景中。然而，随着生成性对抗网络的流行，虚假图像技术的滥用引发了一系列安全问题，如恶意篡改他人照片和视频、侵犯个人隐私等。受生成式对抗性网络的启发，本文提出了一种新颖的对抗性攻击方法，旨在洞察图像分类系统的弱点，提高其抗攻击能力。具体地说，生成式对抗性网络用于生成扰动较小但足以影响分类器决策的对抗性样本，并通过训练器和分类器的对抗性学习来生成对抗性样本。通过大量的实验分析，我们在一个经典的图像分类数据集上对该方法的有效性进行了评估，结果表明，我们的模型成功地欺骗了各种高级分类器，同时保持了对抗性样本的自然性。



## **19. An Empirical Analysis of Federated Learning Models Subject to Label-Flipping Adversarial Attack**

受标签翻转对抗攻击的联邦学习模型的实证分析 cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18507v1) [paper-pdf](http://arxiv.org/pdf/2412.18507v1)

**Authors**: Kunal Bhatnagar, Sagana Chattanathan, Angela Dang, Bhargav Eranki, Ronnit Rana, Charan Sridhar, Siddharth Vedam, Angie Yao, Mark Stamp

**Abstract**: In this paper, we empirically analyze adversarial attacks on selected federated learning models. The specific learning models considered are Multinominal Logistic Regression (MLR), Support Vector Classifier (SVC), Multilayer Perceptron (MLP), Convolution Neural Network (CNN), %Recurrent Neural Network (RNN), Random Forest, XGBoost, and Long Short-Term Memory (LSTM). For each model, we simulate label-flipping attacks, experimenting extensively with 10 federated clients and 100 federated clients. We vary the percentage of adversarial clients from 10% to 100% and, simultaneously, the percentage of labels flipped by each adversarial client is also varied from 10% to 100%. Among other results, we find that models differ in their inherent robustness to the two vectors in our label-flipping attack, i.e., the percentage of adversarial clients, and the percentage of labels flipped by each adversarial client. We discuss the potential practical implications of our results.

摘要: 在本文中，我们实证分析了对选定联邦学习模型的对抗攻击。考虑的具体学习模型是多项逻辑回归（MLR）、支持载体分类器（SRC）、多层感知器（MLP）、卷积神经网络（CNN）、%回归神经网络（RNN）、随机森林、XGboost和长短期记忆（LSTM）。对于每个模型，我们模拟标签翻转攻击，对10个联邦客户端和100个联邦客户端进行了广泛实验。我们将敌对客户的百分比从10%到100%不等，同时，每个敌对客户翻转的标签百分比也从10%到100%不等。除其他结果外，我们发现模型对标签翻转攻击中的两个载体的固有鲁棒性有所不同，即敌对客户的百分比，以及每个敌对客户翻转的标签百分比。我们讨论了结果的潜在实际影响。



## **20. Prompted Contextual Vectors for Spear-Phishing Detection**

用于鱼叉钓鱼检测的预定上下文载体 cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2402.08309v3) [paper-pdf](http://arxiv.org/pdf/2402.08309v3)

**Authors**: Daniel Nahmias, Gal Engelberg, Dan Klein, Asaf Shabtai

**Abstract**: Spear-phishing attacks present a significant security challenge, with large language models (LLMs) escalating the threat by generating convincing emails and facilitating target reconnaissance. To address this, we propose a detection approach based on a novel document vectorization method that utilizes an ensemble of LLMs to create representation vectors. By prompting LLMs to reason and respond to human-crafted questions, we quantify the presence of common persuasion principles in the email's content, producing prompted contextual document vectors for a downstream supervised machine learning model. We evaluate our method using a unique dataset generated by a proprietary system that automates target reconnaissance and spear-phishing email creation. Our method achieves a 91\% F1 score in identifying LLM-generated spear-phishing emails, with the training set comprising only traditional phishing and benign emails. Key contributions include a novel document vectorization method utilizing LLM reasoning, a publicly available dataset of high-quality spear-phishing emails, and the demonstrated effectiveness of our method in detecting such emails. This methodology can be utilized for various document classification tasks, particularly in adversarial problem domains.

摘要: 鱼叉式网络钓鱼攻击是一个重大的安全挑战，大型语言模型(LLM)通过生成令人信服的电子邮件和促进目标侦察来升级威胁。针对这一问题，我们提出了一种基于一种新的文档矢量化方法的检测方法，该方法利用一组LLM来创建表示向量。通过促使LLM对人类提出的问题进行推理和回应，我们量化了电子邮件内容中常见说服原则的存在，为下游有监督的机器学习模型生成了提示的上下文文档向量。我们使用由专有系统生成的唯一数据集来评估我们的方法，该系统自动执行目标侦察和鱼叉式网络钓鱼电子邮件创建。我们的方法在识别LLM生成的鱼叉式钓鱼邮件方面取得了91%的F1分数，训练集仅包括传统钓鱼邮件和良性电子邮件。主要贡献包括一种利用LLM推理的新的文档矢量化方法，一个公开可用的高质量鱼叉式钓鱼电子邮件数据集，以及我们的方法在检测此类电子邮件方面的有效性。这种方法可用于各种文档分类任务，特别是在对抗性问题领域。



## **21. Unveiling the Threat of Fraud Gangs to Graph Neural Networks: Multi-Target Graph Injection Attacks against GNN-Based Fraud Detectors**

揭露欺诈团伙对图神经网络的威胁：针对基于GNN的欺诈检测器的多目标图注入攻击 cs.LG

19 pages, 5 figures, 12 tables, The 39th AAAI Conference on  Artificial Intelligence (AAAI 2025)

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18370v1) [paper-pdf](http://arxiv.org/pdf/2412.18370v1)

**Authors**: Jinhyeok Choi, Heehyeon Kim, Joyce Jiyoung Whang

**Abstract**: Graph neural networks (GNNs) have emerged as an effective tool for fraud detection, identifying fraudulent users, and uncovering malicious behaviors. However, attacks against GNN-based fraud detectors and their risks have rarely been studied, thereby leaving potential threats unaddressed. Recent findings suggest that frauds are increasingly organized as gangs or groups. In this work, we design attack scenarios where fraud gangs aim to make their fraud nodes misclassified as benign by camouflaging their illicit activities in collusion. Based on these scenarios, we study adversarial attacks against GNN-based fraud detectors by simulating attacks of fraud gangs in three real-world fraud cases: spam reviews, fake news, and medical insurance frauds. We define these attacks as multi-target graph injection attacks and propose MonTi, a transformer-based Multi-target one-Time graph injection attack model. MonTi simultaneously generates attributes and edges of all attack nodes with a transformer encoder, capturing interdependencies between attributes and edges more effectively than most existing graph injection attack methods that generate these elements sequentially. Additionally, MonTi adaptively allocates the degree budget for each attack node to explore diverse injection structures involving target, candidate, and attack nodes, unlike existing methods that fix the degree budget across all attack nodes. Experiments show that MonTi outperforms the state-of-the-art graph injection attack methods on five real-world graphs.

摘要: 图神经网络(GNN)已经成为检测欺诈、识别欺诈用户和揭露恶意行为的有效工具。然而，对基于GNN的欺诈探测器的攻击及其风险很少被研究，从而使潜在的威胁得不到解决。最近的发现表明，诈骗越来越多地被组织成帮派或团体。在这项工作中，我们设计了攻击场景，其中欺诈团伙的目标是通过伪装他们在串通中的非法活动来使他们的欺诈节点错误地被归类为良性的。基于这些场景，我们通过模拟三个真实世界的欺诈案例：垃圾邮件评论、假新闻和医疗保险欺诈，研究了针对基于GNN的欺诈检测器的对抗性攻击。我们将这些攻击定义为多目标图注入攻击，并提出了一种基于变压器的多目标一次性图注入攻击模型MONTI。Monti使用转换器编码器同时生成所有攻击节点的属性和边，比大多数现有的按顺序生成这些元素的图注入攻击方法更有效地捕获属性和边之间的相互依赖关系。此外，与现有方法固定所有攻击节点的度预算不同，Monti自适应地为每个攻击节点分配度预算，以探索涉及目标、候选和攻击节点的不同注入结构。实验表明，Monti在五个真实图上的性能优于目前最先进的图注入攻击方法。



## **22. Hypergraph Attacks via Injecting Homogeneous Nodes into Elite Hyperedges**

通过将同质节点注入精英超文本攻击 cs.LG

9 pages, The 39th Annual AAAI Conference on Artificial  Intelligence(2025)

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18365v1) [paper-pdf](http://arxiv.org/pdf/2412.18365v1)

**Authors**: Meixia He, Peican Zhu, Keke Tang, Yangming Guo

**Abstract**: Recent studies have shown that Hypergraph Neural Networks (HGNNs) are vulnerable to adversarial attacks. Existing approaches focus on hypergraph modification attacks guided by gradients, overlooking node spanning in the hypergraph and the group identity of hyperedges, thereby resulting in limited attack performance and detectable attacks. In this manuscript, we present a novel framework, i.e., Hypergraph Attacks via Injecting Homogeneous Nodes into Elite Hyperedges (IE-Attack), to tackle these challenges. Initially, utilizing the node spanning in the hypergraph, we propose the elite hyperedges sampler to identify hyperedges to be injected. Subsequently, a node generator utilizing Kernel Density Estimation (KDE) is proposed to generate the homogeneous node with the group identity of hyperedges. Finally, by injecting the homogeneous node into elite hyperedges, IE-Attack improves the attack performance and enhances the imperceptibility of attacks. Extensive experiments are conducted on five authentic datasets to validate the effectiveness of IE-Attack and the corresponding superiority to state-of-the-art methods.

摘要: 最近的研究表明，超图神经网络(HGNN)容易受到敌意攻击。现有的攻击方法主要关注梯度引导的超图修改攻击，忽略了超图中节点的生成和超边的群标识，从而导致攻击性能有限和攻击可检测。在这篇文章中，我们提出了一种新的框架，即通过向精英超边注入同质节点的超图攻击(IE-Attack)来应对这些挑战。首先，利用超图中的节点生成特性，提出了精英超边采样器来识别待注入的超边。在此基础上，提出了一种基于核密度估计(KDE)的节点生成器来生成具有超边群同一性的同质节点。最后，IE-Attack通过在精英超边中注入同质节点，改善了攻击性能，增强了攻击的隐蔽性。在五个真实的数据集上进行了大量的实验，以验证IE攻击的有效性及其相对于最新方法的优越性。



## **23. Level Up with ML Vulnerability Identification: Leveraging Domain Constraints in Feature Space for Robust Android Malware Detection**

利用ML漏洞识别升级：利用特征空间中的域约束进行稳健的Android恶意软件检测 cs.LG

The paper was accepted by ACM Transactions on Privacy and Security on  2 December 2024

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2205.15128v4) [paper-pdf](http://arxiv.org/pdf/2205.15128v4)

**Authors**: Hamid Bostani, Zhengyu Zhao, Zhuoran Liu, Veelasha Moonsamy

**Abstract**: Machine Learning (ML) promises to enhance the efficacy of Android Malware Detection (AMD); however, ML models are vulnerable to realistic evasion attacks--crafting realizable Adversarial Examples (AEs) that satisfy Android malware domain constraints. To eliminate ML vulnerabilities, defenders aim to identify susceptible regions in the feature space where ML models are prone to deception. The primary approach to identifying vulnerable regions involves investigating realizable AEs, but generating these feasible apps poses a challenge. For instance, previous work has relied on generating either feature-space norm-bounded AEs or problem-space realizable AEs in adversarial hardening. The former is efficient but lacks full coverage of vulnerable regions while the latter can uncover these regions by satisfying domain constraints but is known to be time-consuming. To address these limitations, we propose an approach to facilitate the identification of vulnerable regions. Specifically, we introduce a new interpretation of Android domain constraints in the feature space, followed by a novel technique that learns them. Our empirical evaluations across various evasion attacks indicate effective detection of AEs using learned domain constraints, with an average of 89.6%. Furthermore, extensive experiments on different Android malware detectors demonstrate that utilizing our learned domain constraints in Adversarial Training (AT) outperforms other AT-based defenses that rely on norm-bounded AEs or state-of-the-art non-uniform perturbations. Finally, we show that retraining a malware detector with a wide variety of feature-space realizable AEs results in a 77.9% robustness improvement against realizable AEs generated by unknown problem-space transformations, with up to 70x faster training than using problem-space realizable AEs.

摘要: 机器学习(ML)有望提高Android恶意软件检测(AMD)的效率；然而，ML模型容易受到现实的逃避攻击--制作满足Android恶意软件领域约束的可实现的对手示例(AE)。为了消除ML漏洞，防御者的目标是识别特征空间中ML模型容易被欺骗的敏感区域。识别易受攻击地区的主要方法包括调查可实现的企业实体，但生成这些可行的应用程序会带来挑战。例如，以前的工作依赖于在对抗性强化中产生特征空间范数有界的实体或问题空间可实现的实体。前者是有效的，但缺乏对脆弱区域的完全覆盖，而后者可以通过满足域约束来发现这些区域，但众所周知是耗时的。为了解决这些限制，我们提出了一种便于识别脆弱区域的方法。具体地说，我们在特征空间中引入了对Android域约束的新解释，随后采用了一种新的技术来学习它们。我们对各种规避攻击的实验评估表明，使用学习的域约束可以有效地检测到AEs，平均检测准确率为89.6%。此外，在不同的Android恶意软件检测器上的大量实验表明，在对抗训练(AT)中利用我们学习的域约束的性能优于其他基于AT的防御系统，这些防御系统依赖于范数有界的AE或最新的非均匀扰动。最后，我们展示了用各种各样的特征空间可实现的AE来重新训练恶意软件检测器，对于未知问题空间变换产生的可实现的AE，健壮性提高了77.9%，训练速度比使用问题空间可实现的AE快70倍。



## **24. ErasableMask: A Robust and Erasable Privacy Protection Scheme against Black-box Face Recognition Models**

ErasableMass：针对黑匣子人脸识别模型的稳健且可擦除的隐私保护方案 cs.CV

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.17038v2) [paper-pdf](http://arxiv.org/pdf/2412.17038v2)

**Authors**: Sipeng Shen, Yunming Zhang, Dengpan Ye, Xiuwen Shi, Long Tang, Haoran Duan, Jiacheng Deng, Ziyi Liu

**Abstract**: While face recognition (FR) models have brought remarkable convenience in face verification and identification, they also pose substantial privacy risks to the public. Existing facial privacy protection schemes usually adopt adversarial examples to disrupt face verification of FR models. However, these schemes often suffer from weak transferability against black-box FR models and permanently damage the identifiable information that cannot fulfill the requirements of authorized operations such as forensics and authentication. To address these limitations, we propose ErasableMask, a robust and erasable privacy protection scheme against black-box FR models. Specifically, via rethinking the inherent relationship between surrogate FR models, ErasableMask introduces a novel meta-auxiliary attack, which boosts black-box transferability by learning more general features in a stable and balancing optimization strategy. It also offers a perturbation erasion mechanism that supports the erasion of semantic perturbations in protected face without degrading image quality. To further improve performance, ErasableMask employs a curriculum learning strategy to mitigate optimization conflicts between adversarial attack and perturbation erasion. Extensive experiments on the CelebA-HQ and FFHQ datasets demonstrate that ErasableMask achieves the state-of-the-art performance in transferability, achieving over 72% confidence on average in commercial FR systems. Moreover, ErasableMask also exhibits outstanding perturbation erasion performance, achieving over 90% erasion success rate.

摘要: 虽然人脸识别(FR)模型在人脸验证和识别方面带来了显著的便利，但它们也给公众带来了巨大的隐私风险。现有的人脸隐私保护方案通常采用对抗性的例子来干扰FR模型的人脸验证。然而，这些方案往往对黑盒FR模型的可转移性较弱，并且永久性地破坏了不能满足取证和认证等授权操作要求的可识别信息。为了解决这些局限性，我们提出了一种针对黑盒FR模型的健壮且可擦除的隐私保护方案--可擦除掩码。具体地说，通过重新考虑代理FR模型之间的内在联系，ErasableMASK引入了一种新的元辅助攻击，该攻击通过学习稳定平衡的优化策略中的更多一般特征来提高黑盒的可转移性。它还提供了一种扰动消除机制，支持在不降低图像质量的情况下消除受保护人脸的语义扰动。为了进一步提高性能，ErasableMASK采用了课程学习策略来缓解对抗性攻击和扰动擦除之间的优化冲突。在CelebA-HQ和FFHQ数据集上的广泛实验表明，可擦除掩码在可转移性方面达到了最先进的性能，在商业FR系统中平均达到72%以上的置信度。此外，可擦除掩模还表现出出色的扰动擦除性能，擦除成功率达到90%以上。



## **25. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

大型语言模型能否提高图神经网络的对抗鲁棒性？ cs.LG

accepted by KDD 2025

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2408.08685v3) [paper-pdf](http://arxiv.org/pdf/2408.08685v3)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial attacks, especially for topology perturbations, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attacks. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph. The source code can be found in https://github.com/zhongjian-zhang/LLM4RGNN.

摘要: 图神经网络(GNN)容易受到敌意攻击，尤其是对拓扑扰动的攻击，许多提高GNN健壮性的方法受到了广泛的关注。最近，我们目睹了大型语言模型(LLM)的巨大成功，这导致许多人探索LLM在GNN上的巨大潜力。然而，它们主要集中在通过利用LLMS来增强节点特征来提高GNN的性能。因此，我们问：GNN的健壮性是否也会随着LLMS强大的理解和推理能力而得到增强？通过给出实验结果，我们发现，尽管LLMS可以提高GNN的健壮性，但其准确率仍然平均下降23.1%，这意味着GNN仍然非常容易受到拓扑攻击。因此，另一个问题是如何扩展LLMS在图对抗健壮性方面的能力。本文提出了一种基于LLM的稳健图结构推理框架LLM4RGNN，该框架将GPT-4的推理能力抽象为用于识别恶意边的局部LLM和用于发现丢失重要边的基于LLM的边预测器，以恢复稳健的图结构。大量的实验表明，LLM4RGNN在不同的GNN上一致地提高了健壮性。即使在某些扰动比增加到40%的情况下，GNN的精度仍然好于干净图形上的精度。源代码可以在https://github.com/zhongjian-zhang/LLM4RGNN.中找到



## **26. On the Effectiveness of Adversarial Training on Malware Classifiers**

恶意软件分类器对抗训练的有效性 cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18218v1) [paper-pdf](http://arxiv.org/pdf/2412.18218v1)

**Authors**: Hamid Bostani, Jacopo Cortellazzi, Daniel Arp, Fabio Pierazzi, Veelasha Moonsamy, Lorenzo Cavallaro

**Abstract**: Adversarial Training (AT) has been widely applied to harden learning-based classifiers against adversarial evasive attacks. However, its effectiveness in identifying and strengthening vulnerable areas of the model's decision space while maintaining high performance on clean data of malware classifiers remains an under-explored area. In this context, the robustness that AT achieves has often been assessed against unrealistic or weak adversarial attacks, which negatively affect performance on clean data and are arguably no longer threats. Previous work seems to suggest robustness is a task-dependent property of AT. We instead argue it is a more complex problem that requires exploring AT and the intertwined roles played by certain factors within data, feature representations, classifiers, and robust optimization settings, as well as proper evaluation factors, such as the realism of evasion attacks, to gain a true sense of AT's effectiveness. In our paper, we address this gap by systematically exploring the role such factors have in hardening malware classifiers through AT. Contrary to recent prior work, a key observation of our research and extensive experiments confirm the hypotheses that all such factors influence the actual effectiveness of AT, as demonstrated by the varying degrees of success from our empirical analysis. We identify five evaluation pitfalls that affect state-of-the-art studies and summarize our insights in ten takeaways to draw promising research directions toward better understanding the factors' settings under which adversarial training works at best.

摘要: 对抗性训练(AT)已被广泛应用于强化基于学习的分类器抵抗对抗性回避攻击。然而，它在识别和加强模型决策空间的易受攻击区域方面的有效性，同时在恶意软件分类器的干净数据上保持高性能，仍然是一个探索不足的领域。在这种情况下，AT实现的健壮性经常被评估以对抗不现实或弱的对手攻击，这些攻击对干净数据的性能产生负面影响，并且可以说不再是威胁。前人的研究似乎表明稳健性是任务依赖的AT特性。相反，我们认为这是一个更复杂的问题，需要探索AT以及数据、特征表示、分类器和稳健优化设置中的某些因素所扮演的相互交织的角色，以及适当的评估因素，如规避攻击的真实性，以获得对AT有效性的真实感觉。在我们的论文中，我们通过系统地探索这些因素在通过AT强化恶意软件分类器中所起的作用来解决这一差距。与最近的工作相反，我们对研究的关键观察和广泛的实验证实了这样的假设，即所有这些因素都会影响AT的实际有效性，正如我们的实证分析所显示的不同程度的成功所证明的那样。我们找出了影响最先进研究的五个评估陷阱，并总结了我们在十个方面的见解，以得出有希望的研究方向，以更好地理解对抗性训练最好发挥作用的因素设置。



## **27. Robustness-aware Automatic Prompt Optimization**

具有鲁棒性的自动提示优化 cs.CL

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18196v1) [paper-pdf](http://arxiv.org/pdf/2412.18196v1)

**Authors**: Zeru Shi, Zhenting Wang, Yongye Su, Weidi Luo, Fan Yang, Yongfeng Zhang

**Abstract**: The performance of Large Language Models (LLMs) is based on the quality of the prompts and the semantic and structural integrity information of the input data. However, current prompt generation methods primarily focus on generating prompts for clean input data, often overlooking the impact of perturbed inputs on prompt performance. To address this limitation, we propose BATprompt (By Adversarial Training prompt), a novel method for prompt generation designed to withstand input perturbations (such as typos in the input). Inspired by adversarial training techniques, BATprompt demonstrates strong performance on a variety of perturbed tasks through a two-step process: adversarial perturbation and iterative optimization on unperturbed input via LLM. Unlike conventional adversarial attack methods, BATprompt avoids reliance on real gradients or model parameters. Instead, it leverages the advanced reasoning, language understanding and self reflection capabilities of LLMs to simulate gradients, guiding the generation of adversarial perturbations and optimizing prompt performance. In our experiments, we evaluate BATprompt on multiple datasets across both language understanding and generation tasks. The results indicate that BATprompt outperforms existing prompt generation methods, delivering superior robustness and performance under diverse perturbation scenarios.

摘要: 大语言模型的性能取决于提示的质量以及输入数据的语义和结构完整性信息。然而，目前的提示生成方法主要集中于为干净的输入数据生成提示，往往忽略了输入干扰对提示性能的影响。为了解决这一局限性，我们提出了一种新的提示生成方法BATprint(通过对抗性训练提示)，该方法旨在抵抗输入扰动(如输入中的打字错误)。受到对抗性训练技术的启发，通过两步过程：对抗性扰动和通过LLM对不受扰动的输入进行迭代优化，BATprint在各种扰动任务上表现出了强大的性能。与传统的对抗性攻击方法不同，BATprint避免了对真实梯度或模型参数的依赖。相反，它利用LLMS的高级推理、语言理解和自我反思能力来模拟梯度，指导生成对抗性扰动并优化提示性能。在我们的实验中，我们在语言理解和生成任务的多个数据集上对BATprint进行了评估。结果表明，BATprint的性能优于现有的提示生成方法，在不同的扰动场景下都具有较好的健壮性和性能。



## **28. Sparse-PGD: A Unified Framework for Sparse Adversarial Perturbations Generation**

稀疏对抗扰动生成的统一框架 cs.LG

Extended version. Codes are available at  https://github.com/CityU-MLO/sPGD

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2405.05075v3) [paper-pdf](http://arxiv.org/pdf/2405.05075v3)

**Authors**: Xuyang Zhong, Chen Liu

**Abstract**: This work studies sparse adversarial perturbations, including both unstructured and structured ones. We propose a framework based on a white-box PGD-like attack method named Sparse-PGD to effectively and efficiently generate such perturbations. Furthermore, we combine Sparse-PGD with a black-box attack to comprehensively and more reliably evaluate the models' robustness against unstructured and structured sparse adversarial perturbations. Moreover, the efficiency of Sparse-PGD enables us to conduct adversarial training to build robust models against various sparse perturbations. Extensive experiments demonstrate that our proposed attack algorithm exhibits strong performance in different scenarios. More importantly, compared with other robust models, our adversarially trained model demonstrates state-of-the-art robustness against various sparse attacks.

摘要: 这项工作研究了稀疏的对抗性扰动，包括非结构化和结构化的扰动。我们提出了一个基于类似白盒PGD攻击方法的框架，名为Sparse-PVD，以有效且高效地生成此类扰动。此外，我们将Sparse-PGDD与黑匣子攻击相结合，以全面、更可靠地评估模型对非结构化和结构化稀疏对抗扰动的鲁棒性。此外，Sparse-PVD的效率使我们能够进行对抗训练，以针对各种稀疏扰动构建稳健的模型。大量实验表明，我们提出的攻击算法在不同场景下表现出强大的性能。更重要的是，与其他稳健模型相比，我们的对抗训练模型表现出了针对各种稀疏攻击的最新稳健性。



## **29. AEIOU: A Unified Defense Framework against NSFW Prompts in Text-to-Image Models**

AEIOU：针对文本到图像模型中NSFW格式的统一防御框架 cs.CR

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18123v1) [paper-pdf](http://arxiv.org/pdf/2412.18123v1)

**Authors**: Yiming Wang, Jiahao Chen, Qingming Li, Xing Yang, Shouling Ji

**Abstract**: As text-to-image (T2I) models continue to advance and gain widespread adoption, their associated safety issues are becoming increasingly prominent. Malicious users often exploit these models to generate Not-Safe-for-Work (NSFW) images using harmful or adversarial prompts, highlighting the critical need for robust safeguards to ensure the integrity and compliance of model outputs. Current internal safeguards frequently degrade image quality, while external detection methods often suffer from low accuracy and inefficiency.   In this paper, we introduce AEIOU, a defense framework that is Adaptable, Efficient, Interpretable, Optimizable, and Unified against NSFW prompts in T2I models. AEIOU extracts NSFW features from the hidden states of the model's text encoder, utilizing the separable nature of these features to detect NSFW prompts. The detection process is efficient, requiring minimal inference time. AEIOU also offers real-time interpretation of results and supports optimization through data augmentation techniques. The framework is versatile, accommodating various T2I architectures. Our extensive experiments show that AEIOU significantly outperforms both commercial and open-source moderation tools, achieving over 95% accuracy across all datasets and improving efficiency by at least tenfold. It effectively counters adaptive attacks and excels in few-shot and multi-label scenarios.

摘要: 随着文本到图像(T2I)模型的不断发展和广泛采用，其相关的安全问题也变得越来越突出。恶意用户经常利用这些模型，使用有害或敌对的提示生成不安全工作(NSFW)图像，突显出迫切需要强有力的保障措施，以确保模型输出的完整性和合规性。目前的内部保护措施经常会降低图像质量，而外部检测方法往往存在精度低和效率低的问题。在本文中，我们介绍了一种针对T2I模型中NSFW提示的适应性、高效、可解释、可优化和统一的防御框架AEIOU。AEIOU从模型的文本编码器的隐藏状态中提取NSFW特征，利用这些特征的可分离性来检测NSFW提示。检测过程是高效的，所需的推理时间最短。AEIOU还提供对结果的实时解释，并通过数据增强技术支持优化。该框架是通用的，可以容纳各种T2I架构。我们的广泛实验表明，AEIOU的性能明显优于商业和开源审核工具，在所有数据集上实现了95%以上的准确率，并将效率提高了至少10倍。它有效地对抗自适应攻击，并在少镜头和多标签场景中表现出色。



## **30. A Tunable Despeckling Neural Network Stabilized via Diffusion Equation**

通过扩散方程稳定的可调谐降斑神经网络 cs.CV

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2411.15921v2) [paper-pdf](http://arxiv.org/pdf/2411.15921v2)

**Authors**: Yi Ran, Zhichang Guo, Jia Li, Yao Li, Martin Burger, Boying Wu

**Abstract**: The removal of multiplicative Gamma noise is a critical research area in the application of synthetic aperture radar (SAR) imaging, where neural networks serve as a potent tool. However, real-world data often diverges from theoretical models, exhibiting various disturbances, which makes the neural network less effective. Adversarial attacks can be used as a criterion for judging the adaptability of neural networks to real data, since adversarial attacks can find the most extreme perturbations that make neural networks ineffective. In this work, the diffusion equation is designed as a regularization block to provide sufficient regularity to the whole neural network, due to its spontaneous dissipative nature. We propose a tunable, regularized neural network framework that unrolls a shallow denoising neural network block and a diffusion regularity block into a single network for end-to-end training. The linear heat equation, known for its inherent smoothness and low-pass filtering properties, is adopted as the diffusion regularization block. In our model, a single time step hyperparameter governs the smoothness of the outputs and can be adjusted dynamically, significantly enhancing flexibility. The stability and convergence of our model are theoretically proven. Experimental results demonstrate that the proposed model effectively eliminates high-frequency oscillations induced by adversarial attacks. Finally, the proposed model is benchmarked against several state-of-the-art denoising methods on simulated images, adversarial samples, and real SAR images, achieving superior performance in both quantitative and visual evaluations.

摘要: 乘性伽马噪声的去除是合成孔径雷达(SAR)成像应用中的一个关键研究领域，而神经网络是其中一个强有力的工具。然而，现实世界的数据经常与理论模型背道而驰，表现出各种干扰，这使得神经网络的效率较低。对抗性攻击可以用来作为判断神经网络对真实数据适应性的标准，因为对抗性攻击可以找到使神经网络无效的最极端的扰动。在这项工作中，由于扩散方程的自发耗散性质，它被设计成一个正则化块来为整个神经网络提供足够的正则性。我们提出了一种可调的正则化神经网络框架，将浅层去噪神经网络块和扩散规则块展开成单个网络进行端到端的训练。以其固有的光滑性和低通滤波特性而闻名的线性热方程被用作扩散正则化块。在我们的模型中，一个时间步长的超参数控制输出的平稳性，并且可以动态调整，显著增强了灵活性。从理论上证明了该模型的稳定性和收敛性。实验结果表明，该模型有效地消除了对抗性攻击引起的高频振荡。最后，在模拟图像、对抗性样本和真实SAR图像上对所提出的模型进行了基准测试，取得了较好的定量和视觉评估性能。



## **31. Large Language Model Safety: A Holistic Survey**

大型语言模型安全性：整体调查 cs.AI

158 pages, 18 figures

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17686v1) [paper-pdf](http://arxiv.org/pdf/2412.17686v1)

**Authors**: Dan Shi, Tianhao Shen, Yufei Huang, Zhigen Li, Yongqi Leng, Renren Jin, Chuang Liu, Xinwei Wu, Zishan Guo, Linhao Yu, Ling Shi, Bojian Jiang, Deyi Xiong

**Abstract**: The rapid development and deployment of large language models (LLMs) have introduced a new frontier in artificial intelligence, marked by unprecedented capabilities in natural language understanding and generation. However, the increasing integration of these models into critical applications raises substantial safety concerns, necessitating a thorough examination of their potential risks and associated mitigation strategies.   This survey provides a comprehensive overview of the current landscape of LLM safety, covering four major categories: value misalignment, robustness to adversarial attacks, misuse, and autonomous AI risks. In addition to the comprehensive review of the mitigation methodologies and evaluation resources on these four aspects, we further explore four topics related to LLM safety: the safety implications of LLM agents, the role of interpretability in enhancing LLM safety, the technology roadmaps proposed and abided by a list of AI companies and institutes for LLM safety, and AI governance aimed at LLM safety with discussions on international cooperation, policy proposals, and prospective regulatory directions.   Our findings underscore the necessity for a proactive, multifaceted approach to LLM safety, emphasizing the integration of technical solutions, ethical considerations, and robust governance frameworks. This survey is intended to serve as a foundational resource for academy researchers, industry practitioners, and policymakers, offering insights into the challenges and opportunities associated with the safe integration of LLMs into society. Ultimately, it seeks to contribute to the safe and beneficial development of LLMs, aligning with the overarching goal of harnessing AI for societal advancement and well-being. A curated list of related papers has been publicly available at https://github.com/tjunlp-lab/Awesome-LLM-Safety-Papers.

摘要: 大型语言模型的快速开发和部署为人工智能带来了一个新的前沿，其标志是在自然语言理解和生成方面具有前所未有的能力。然而，这些模型越来越多地集成到关键应用程序中，引发了大量的安全问题，需要彻底检查它们的潜在风险和相关的缓解策略。这项调查全面概述了LLM安全的现状，包括四个主要类别：价值错位、对对手攻击的健壮性、误用和自主AI风险。除了对这四个方面的缓解方法和评估资源进行全面审查外，我们还进一步探讨了与LLM安全相关的四个主题：LLM制剂的安全影响、可解释性在增强LLM安全方面的作用、一系列人工智能公司和机构为LLM安全提出并遵守的技术路线图，以及旨在实现LLM安全的人工智能治理，并就国际合作、政策建议和未来监管方向进行了讨论。我们的发现强调了对LLM安全采取积极、多方面方法的必要性，强调将技术解决方案、伦理考虑和强大的治理框架整合在一起。这项调查旨在为学院研究人员、行业从业者和政策制定者提供基础性资源，为低收入国家安全融入社会带来的挑战和机遇提供洞察力。最终，它寻求为低土地管理的安全和有益的发展做出贡献，与利用人工智能促进社会进步和福祉的总体目标保持一致。相关论文的精选名单已在https://github.com/tjunlp-lab/Awesome-LLM-Safety-Papers.上公开提供



## **32. Emerging Security Challenges of Large Language Models**

大型语言模型新出现的安全挑战 cs.CR

A version of this appeared in the larger Dagstuhl seminar 23431  report (https://doi.org/10.4230/DagRep.13.10.90)

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17614v1) [paper-pdf](http://arxiv.org/pdf/2412.17614v1)

**Authors**: Herve Debar, Sven Dietrich, Pavel Laskov, Emil C. Lupu, Eirini Ntoutsi

**Abstract**: Large language models (LLMs) have achieved record adoption in a short period of time across many different sectors including high importance areas such as education [4] and healthcare [23]. LLMs are open-ended models trained on diverse data without being tailored for specific downstream tasks, enabling broad applicability across various domains. They are commonly used for text generation, but also widely used to assist with code generation [3], and even analysis of security information, as Microsoft Security Copilot demonstrates [18]. Traditional Machine Learning (ML) models are vulnerable to adversarial attacks [9]. So the concerns on the potential security implications of such wide scale adoption of LLMs have led to the creation of this working group on the security of LLMs. During the Dagstuhl seminar on "Network Attack Detection and Defense - AI-Powered Threats and Responses", the working group discussions focused on the vulnerability of LLMs to adversarial attacks, rather than their potential use in generating malware or enabling cyberattacks. Although we note the potential threat represented by the latter, the role of the LLMs in such uses is mostly as an accelerator for development, similar to what it is in benign use. To make the analysis more specific, the working group employed ChatGPT as a concrete example of an LLM and addressed the following points, which also form the structure of this report: 1. How do LLMs differ in vulnerabilities from traditional ML models? 2. What are the attack objectives in LLMs? 3. How complex it is to assess the risks posed by the vulnerabilities of LLMs? 4. What is the supply chain in LLMs, how data flow in and out of systems and what are the security implications? We conclude with an overview of open challenges and outlook.

摘要: 大型语言模型(LLM)在短时间内在许多不同的领域获得了创纪录的采用，包括教育[4]和医疗保健[23]等高度重要的领域。LLM是基于不同数据进行培训的开放式模型，无需为特定的下游任务量身定做，从而实现了跨不同领域的广泛适用性。它们通常用于文本生成，但也广泛用于协助代码生成[3]，甚至分析安全信息，正如Microsoft Security Copilot演示的那样[18]。传统的机器学习(ML)模型容易受到对抗性攻击[9]。因此，出于对大规模采用小岛屿发展中国家可能产生的安全影响的关切，设立了小岛屿发展中国家安全问题工作组。在达格斯图尔关于“网络攻击检测和防御--人工智能支持的威胁和反应”的研讨会期间，工作组讨论了低收入管理系统对对抗性攻击的脆弱性，而不是它们在生成恶意软件或支持网络攻击方面的潜在用途。尽管我们注意到后者所代表的潜在威胁，但小岛屿发展中国家在这种用途中的作用主要是作为发展的加速器，类似于它在良性使用中的作用。为了使分析更具体，工作组使用ChatGPT作为LLM的具体例子，并讨论了以下几点，这些点也构成了本报告的结构：1.LLMS与传统的ML模型在漏洞方面有何不同？2.LLMS中的攻击目标是什么？3.评估LLMS漏洞带来的风险有多复杂？4.LLMS中的供应链是什么，数据如何进出系统，以及安全影响是什么？最后，我们对开放的挑战和前景进行了概述。



## **33. Retention Score: Quantifying Jailbreak Risks for Vision Language Models**

保留分数：量化视觉语言模型的越狱风险 cs.AI

14 pages, 8 figures, AAAI 2025

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17544v1) [paper-pdf](http://arxiv.org/pdf/2412.17544v1)

**Authors**: Zaitang Li, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: The emergence of Vision-Language Models (VLMs) is a significant advancement in integrating computer vision with Large Language Models (LLMs) to enhance multi-modal machine learning capabilities. However, this progress has also made VLMs vulnerable to sophisticated adversarial attacks, raising concerns about their reliability. The objective of this paper is to assess the resilience of VLMs against jailbreak attacks that can compromise model safety compliance and result in harmful outputs. To evaluate a VLM's ability to maintain its robustness against adversarial input perturbations, we propose a novel metric called the \textbf{Retention Score}. Retention Score is a multi-modal evaluation metric that includes Retention-I and Retention-T scores for quantifying jailbreak risks in visual and textual components of VLMs. Our process involves generating synthetic image-text pairs using a conditional diffusion model. These pairs are then predicted for toxicity score by a VLM alongside a toxicity judgment classifier. By calculating the margin in toxicity scores, we can quantify the robustness of the VLM in an attack-agnostic manner. Our work has four main contributions. First, we prove that Retention Score can serve as a certified robustness metric. Second, we demonstrate that most VLMs with visual components are less robust against jailbreak attacks than the corresponding plain VLMs. Additionally, we evaluate black-box VLM APIs and find that the security settings in Google Gemini significantly affect the score and robustness. Moreover, the robustness of GPT4V is similar to the medium settings of Gemini. Finally, our approach offers a time-efficient alternative to existing adversarial attack methods and provides consistent model robustness rankings when evaluated on VLMs including MiniGPT-4, InstructBLIP, and LLaVA.

摘要: 视觉语言模型(VLMS)的出现是将计算机视觉与大型语言模型(LLM)相结合以增强多模式机器学习能力的一个重大进步。然而，这一进展也使VLM容易受到复杂的对抗性攻击，这引发了人们对其可靠性的担忧。本文的目的是评估VLM对越狱攻击的恢复能力，这些攻击可能会损害模型安全合规性并导致有害输出。为了评估VLM对敌意输入扰动保持健壮性的能力，我们提出了一种新的度量，称为\extbf{保留分数}。保留分数是一种多模式评估指标，包括用于量化VLM视觉和文本部分越狱风险的保留-I和保留-T分数。我们的过程包括使用条件扩散模型生成合成图文对。然后，由VLM和毒性判断分类器一起预测这些对的毒性分数。通过计算毒性分数的差值，我们可以以一种攻击不可知的方式来量化VLM的健壮性。我们的工作有四个主要贡献。首先，我们证明了保留分数可以作为认证的稳健性度量。其次，我们证明了大多数具有可视组件的VLM对越狱攻击的健壮性不如相应的普通VLM。此外，我们对黑盒VLMAPI进行了评估，发现Google Gemini中的安全设置对分数和健壮性有显著影响。此外，GPT4V的健壮性与双子座的中等设置相似。最后，我们的方法为现有的对抗性攻击方法提供了一种省时的替代方法，并在包括MiniGPT-4、InstructBLIP和LLaVA的VLM上进行了评估，提供了一致的模型健壮性排名。



## **34. Gröbner Basis Cryptanalysis of Ciminion and Hydra**

格罗布纳基础对西米尼恩和海德拉的密码分析 cs.CR

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2405.05040v2) [paper-pdf](http://arxiv.org/pdf/2405.05040v2)

**Authors**: Matthias Johann Steiner

**Abstract**: Ciminion and Hydra are two recently introduced symmetric key Pseudo-Random Functions for Multi-Party Computation applications. For efficiency both primitives utilize quadratic permutations at round level. Therefore, polynomial system solving-based attacks pose a serious threat to these primitives. For Ciminion, we construct a quadratic degree reverse lexicographic (DRL) Gr\"obner basis for the iterated polynomial model via linear transformations. With the Gr\"obner basis we can simplify cryptanalysis since we do not need to impose genericity assumptions anymore to derive complexity estimations. For Hydra, with the help of a computer algebra program like SageMath we construct a DRL Gr\"obner basis for the iterated model via linear transformations and a linear change of coordinates. In the Hydra proposal it was claimed that $r_\mathcal{H} = 31$ rounds are sufficient to provide $128$ bits of security against Gr\"obner basis attacks for an ideal adversary with $\omega = 2$. However, via our Hydra Gr\"obner basis standard term order conversion to a lexicographic (LEX) Gr\"obner basis requires just $126$ bits with $\omega = 2$. Moreover, via a dedicated polynomial system solving technique up to $r_\mathcal{H} = 33$ rounds can be attacked below $128$ bits for an ideal adversary.

摘要: Ciminion和Hydra是最近推出的两个用于多方计算应用的对称密钥伪随机函数。为了提高效率，两个基元都在循环水平上使用二次置换。因此，基于多项式系统求解的攻击对这些原语构成了严重威胁。对于Ciminion，我们通过线性变换为迭代多项式模型构造了一个二次逆词典(DRL)Gr‘obner基，利用这个Gr’obner基，我们可以简化密码分析，因为我们不再需要强加一般性假设来推导复杂性估计。对于Hydra，借助于SageMath这样的计算机代数程序，我们通过线性变换和线性坐标变化，为迭代模型构造了一个DRL Grobner基.在Hydra的方案中，声称$r_\mathcal{H}=31$轮足以为$omega=2$的理想对手提供$128$bit的Gr‘obner基攻击安全.然而，通过我们的Hydra Gr\“obner基础”将标准术语顺序转换为词典(Lex)Gr\“obner基础只需要$126$位，且$\omega=2$。此外，通过一种专门的多项式系统求解技术，高达$r_\数学{H}=33$的轮数可以被攻击到低于$128$比特的理想对手。



## **35. Ensembler: Protect Collaborative Inference Privacy from Model Inversion Attack via Selective Ensemble**

Ensembler：通过选择性Ensemble保护协作推理隐私免受模型倒置攻击 cs.CR

in submission

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2401.10859v2) [paper-pdf](http://arxiv.org/pdf/2401.10859v2)

**Authors**: Dancheng Liu, Chenhui Xu, Jiajie Li, Amir Nassereldine, Jinjun Xiong

**Abstract**: For collaborative inference through a cloud computing platform, it is sometimes essential for the client to shield its sensitive information from the cloud provider. In this paper, we introduce Ensembler, an extensible framework designed to substantially increase the difficulty of conducting model inversion attacks by adversarial parties. Ensembler leverages selective model ensemble on the adversarial server to obfuscate the reconstruction of the client's private information. Our experiments demonstrate that Ensembler can effectively shield input images from reconstruction attacks, even when the client only retains one layer of the network locally. Ensembler significantly outperforms baseline methods by up to 43.5% in structural similarity while only incurring 4.8% time overhead during inference.

摘要: 对于通过云计算平台进行协作推理，客户端有时必须保护其敏感信息免受云提供商的侵害。在本文中，我们引入了Ensembler，这是一个可扩展框架，旨在大幅增加对抗方进行模型倒置攻击的难度。Ensembler利用对抗服务器上的选择性模型集成来模糊客户端私人信息的重建。我们的实验表明，即使客户端仅在本地保留网络的一层，Ensembler也可以有效地保护输入图像免受重建攻击。Ensembler在结构相似性方面显着优于基线方法，高达43.5%，而推理过程中仅产生4.8%的时间负担。



## **36. SEAS: Self-Evolving Adversarial Safety Optimization for Large Language Models**

SEAS：大型语言模型的自进化对抗安全优化 cs.CL

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2408.02632v2) [paper-pdf](http://arxiv.org/pdf/2408.02632v2)

**Authors**: Muxi Diao, Rumei Li, Shiyang Liu, Guogang Liao, Jingang Wang, Xunliang Cai, Weiran Xu

**Abstract**: As large language models (LLMs) continue to advance in capability and influence, ensuring their security and preventing harmful outputs has become crucial. A promising approach to address these concerns involves training models to automatically generate adversarial prompts for red teaming. However, the evolving subtlety of vulnerabilities in LLMs challenges the effectiveness of current adversarial methods, which struggle to specifically target and explore the weaknesses of these models. To tackle these challenges, we introduce the $\mathbf{S}\text{elf-}\mathbf{E}\text{volving }\mathbf{A}\text{dversarial }\mathbf{S}\text{afety }\mathbf{(SEAS)}$ optimization framework, which enhances security by leveraging data generated by the model itself. SEAS operates through three iterative stages: Initialization, Attack, and Adversarial Optimization, refining both the Red Team and Target models to improve robustness and safety. This framework reduces reliance on manual testing and significantly enhances the security capabilities of LLMs. Our contributions include a novel adversarial framework, a comprehensive safety dataset, and after three iterations, the Target model achieves a security level comparable to GPT-4, while the Red Team model shows a marked increase in attack success rate (ASR) against advanced models. Our code and datasets are released at https://SEAS-LLM.github.io/.

摘要: 随着大型语言模型在能力和影响力方面的不断进步，确保它们的安全和防止有害输出变得至关重要。解决这些担忧的一个有希望的方法是建立训练模型，为红色团队自动生成对抗性提示。然而，LLMS中不断演变的漏洞的微妙之处挑战了当前对抗性方法的有效性，这些方法难以具体针对和探索这些模型的弱点。为了应对这些挑战，我们引入了$\mathbf{S}\Text{ELF-}\mathbf{E}\Text{volving}\mathbf{A}\Text{dversarial}\mathbf{S}\Text{afty}\mathbf{(SEA)}$优化框架，该框架通过利用模型本身生成的数据来增强安全性。SEA经历了三个迭代阶段：初始化、攻击和对抗性优化，完善了Red Team和Target模型，以提高健壮性和安全性。该框架减少了对手动测试的依赖，显著增强了LLMS的安全能力。我们的贡献包括一个新的对抗性框架，一个全面的安全数据集，经过三次迭代，Target模型达到了与GPT-4相当的安全级别，而Red Team模型显示出相对于高级模型在攻击成功率(ASR)方面的显著提高。我们的代码和数据集在https://SEAS-LLM.github.io/.上发布



## **37. The Superalignment of Superhuman Intelligence with Large Language Models**

超人智能与大型语言模型的超级对齐 cs.CL

Under review of Science China

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.11145v2) [paper-pdf](http://arxiv.org/pdf/2412.11145v2)

**Authors**: Minlie Huang, Yingkang Wang, Shiyao Cui, Pei Ke, Jie Tang

**Abstract**: We have witnessed superhuman intelligence thanks to the fast development of large language models and multimodal language models. As the application of such superhuman models becomes more and more popular, a critical question arises here: how can we ensure superhuman models are still safe, reliable and aligned well to human values? In this position paper, we discuss the concept of superalignment from the learning perspective to answer this question by outlining the learning paradigm shift from large-scale pretraining, supervised fine-tuning, to alignment training. We define superalignment as designing effective and efficient alignment algorithms to learn from noisy-labeled data (point-wise samples or pair-wise preference data) in a scalable way when the task becomes very complex for human experts to annotate and the model is stronger than human experts. We highlight some key research problems in superalignment, namely, weak-to-strong generalization, scalable oversight, and evaluation. We then present a conceptual framework for superalignment, which consists of three modules: an attacker which generates adversary queries trying to expose the weaknesses of a learner model; a learner which will refine itself by learning from scalable feedbacks generated by a critic model along with minimal human experts; and a critic which generates critics or explanations for a given query-response pair, with a target of improving the learner by criticizing. We discuss some important research problems in each component of this framework and highlight some interesting research ideas that are closely related to our proposed framework, for instance, self-alignment, self-play, self-refinement, and more. Last, we highlight some future research directions for superalignment, including identification of new emergent risks and multi-dimensional alignment.

摘要: 由于大型语言模型和多模式语言模型的快速发展，我们见证了超人的智能。随着这种超人模型的应用变得越来越普遍，一个关键的问题出现了：我们如何确保超人模型仍然安全、可靠，并与人类的价值观保持良好一致？在这份立场文件中，我们从学习的角度讨论了超匹配的概念，通过概述学习范式从大规模预训练、有监督的微调到对齐训练的转变来回答这个问题。我们将超比对定义为设计有效和高效的比对算法，当任务对于人类专家来说变得非常复杂并且模型比人类专家更强时，以可扩展的方式从噪声标记的数据(点状样本或成对偏好数据)中学习。我们强调了超比对中的一些关键研究问题，即从弱到强的泛化、可扩展的监督和评估。然后，我们提出了一个超对齐的概念框架，它由三个模块组成：攻击者，生成敌意查询，试图揭露学习者模型的弱点；学习者，将通过与最少的人类专家一起从批评者模型生成的可伸缩反馈中学习来改进自己；批评者，为给定的查询-响应对生成批评者或解释，目标是通过批评来改进学习者。我们讨论了该框架每个组成部分中的一些重要研究问题，并突出了与我们提出的框架密切相关的一些有趣的研究想法，例如自我调整、自我发挥、自我完善等。最后，我们指出了超配准未来的研究方向，包括识别新出现的风险和多维配对。



## **38. DynamicPAE: Generating Scene-Aware Physical Adversarial Examples in Real-Time**

DynamicPoker：实时生成场景感知物理对抗示例 cs.CV

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.08053v2) [paper-pdf](http://arxiv.org/pdf/2412.08053v2)

**Authors**: Jin Hu, Xianglong Liu, Jiakai Wang, Junkai Zhang, Xianqi Yang, Haotong Qin, Yuqing Ma, Ke Xu

**Abstract**: Physical adversarial examples (PAEs) are regarded as "whistle-blowers" of real-world risks in deep-learning applications. However, current PAE generation studies show limited adaptive attacking ability to diverse and varying scenes. The key challenges in generating dynamic PAEs are exploring their patterns under noisy gradient feedback and adapting the attack to agnostic scenario natures. To address the problems, we present DynamicPAE, the first generative framework that enables scene-aware real-time physical attacks beyond static attacks. Specifically, to train the dynamic PAE generator under noisy gradient feedback, we introduce the residual-driven sample trajectory guidance technique, which redefines the training task to break the limited feedback information restriction that leads to the degeneracy problem. Intuitively, it allows the gradient feedback to be passed to the generator through a low-noise auxiliary task, thereby guiding the optimization away from degenerate solutions and facilitating a more comprehensive and stable exploration of feasible PAEs. To adapt the generator to agnostic scenario natures, we introduce the context-aligned scene expectation simulation process, consisting of the conditional-uncertainty-aligned data module and the skewness-aligned objective re-weighting module. The former enhances robustness in the context of incomplete observation by employing a conditional probabilistic model for domain randomization, while the latter facilitates consistent stealth control across different attack targets by automatically reweighting losses based on the skewness indicator. Extensive digital and physical evaluations demonstrate the superior attack performance of DynamicPAE, attaining a 1.95 $\times$ boost (65.55% average AP drop under attack) on representative object detectors (e.g., Yolo-v8) over state-of-the-art static PAE generating methods.

摘要: 物理对抗性例子(PAE)被认为是深度学习应用中真实世界风险的“告密者”。然而，目前的PAE代研究表明，对不同场景的自适应攻击能力有限。生成动态PAE的关键挑战是在噪声梯度反馈下探索它们的模式，并使攻击适应不可知的场景性质。为了解决这些问题，我们提出了DynamicPAE，这是第一个生成性框架，它能够在静态攻击之外实现场景感知的实时物理攻击。具体地说，为了在噪声梯度反馈下训练动态PAE产生器，我们引入了残差驱动样本轨迹制导技术，重新定义了训练任务，打破了导致退化问题的有限反馈信息限制。直观地说，它允许通过低噪声辅助任务将梯度反馈传递给生成器，从而引导优化远离退化解，并有助于更全面和稳定地探索可行的PAE。为了使生成器适应不可知的场景性质，我们引入了上下文对齐的场景期望模拟过程，由条件不确定性对齐的数据模块和偏度对齐的目标重加权模块组成。前者通过使用区域随机化的条件概率模型来增强不完全观测环境下的鲁棒性，而后者通过基于偏度指标自动重新加权损失来促进对不同攻击目标的一致隐身控制。广泛的数字和物理评估表明，DynamicPAE具有优越的攻击性能，与最先进的静态PAE生成方法相比，在典型对象检测器(如Yolo-V8)上获得了1.95美元\倍的$提升(攻击下平均AP下降65.55%)。



## **39. Robustness of Large Language Models Against Adversarial Attacks**

大型语言模型对抗对抗攻击的鲁棒性 cs.CL

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.17011v1) [paper-pdf](http://arxiv.org/pdf/2412.17011v1)

**Authors**: Yiyi Tao, Yixian Shen, Hang Zhang, Yanxin Shen, Lun Wang, Chuanqi Shi, Shaoshuai Du

**Abstract**: The increasing deployment of Large Language Models (LLMs) in various applications necessitates a rigorous evaluation of their robustness against adversarial attacks. In this paper, we present a comprehensive study on the robustness of GPT LLM family. We employ two distinct evaluation methods to assess their resilience. The first method introduce character-level text attack in input prompts, testing the models on three sentiment classification datasets: StanfordNLP/IMDB, Yelp Reviews, and SST-2. The second method involves using jailbreak prompts to challenge the safety mechanisms of the LLMs. Our experiments reveal significant variations in the robustness of these models, demonstrating their varying degrees of vulnerability to both character-level and semantic-level adversarial attacks. These findings underscore the necessity for improved adversarial training and enhanced safety mechanisms to bolster the robustness of LLMs.

摘要: 大型语言模型（LLM）在各种应用程序中的部署越来越多，需要严格评估其对抗性攻击的稳健性。本文对GPT LLM家族的稳健性进行了全面的研究。我们采用两种不同的评估方法来评估其弹性。第一种方法在输入提示中引入字符级文本攻击，在三个情感分类数据集上测试模型：StanfordNLP/IMDB、Yelp Reviews和CST-2。第二种方法涉及使用越狱提示来挑战LLM的安全机制。我们的实验揭示了这些模型的稳健性存在显着差异，证明了它们对字符级和语义级对抗攻击的脆弱性程度不同。这些发现强调了改进对抗培训和增强安全机制以增强LLM稳健性的必要性。



## **40. Breaking Barriers in Physical-World Adversarial Examples: Improving Robustness and Transferability via Robust Feature**

打破物理世界对抗示例中的障碍：通过稳健特征提高稳健性和可移植性 cs.CV

Accepted by AAAI2025

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.16958v1) [paper-pdf](http://arxiv.org/pdf/2412.16958v1)

**Authors**: Yichen Wang, Yuxuan Chou, Ziqi Zhou, Hangtao Zhang, Wei Wan, Shengshan Hu, Minghui Li

**Abstract**: As deep neural networks (DNNs) are widely applied in the physical world, many researches are focusing on physical-world adversarial examples (PAEs), which introduce perturbations to inputs and cause the model's incorrect outputs. However, existing PAEs face two challenges: unsatisfactory attack performance (i.e., poor transferability and insufficient robustness to environment conditions), and difficulty in balancing attack effectiveness with stealthiness, where better attack effectiveness often makes PAEs more perceptible.   In this paper, we explore a novel perturbation-based method to overcome the challenges. For the first challenge, we introduce a strategy Deceptive RF injection based on robust features (RFs) that are predictive, robust to perturbations, and consistent across different models. Specifically, it improves the transferability and robustness of PAEs by covering RFs of other classes onto the predictive features in clean images. For the second challenge, we introduce another strategy Adversarial Semantic Pattern Minimization, which removes most perturbations and retains only essential adversarial patterns in AEsBased on the two strategies, we design our method Robust Feature Coverage Attack (RFCoA), comprising Robust Feature Disentanglement and Adversarial Feature Fusion. In the first stage, we extract target class RFs in feature space. In the second stage, we use attention-based feature fusion to overlay these RFs onto predictive features of clean images and remove unnecessary perturbations. Experiments show our method's superior transferability, robustness, and stealthiness compared to existing state-of-the-art methods. Additionally, our method's effectiveness can extend to Large Vision-Language Models (LVLMs), indicating its potential applicability to more complex tasks.

摘要: 随着深度神经网络(DNN)在物理世界中的广泛应用，许多研究都集中在物理世界中的对抗性例子(PAE)上，这些例子会对输入产生扰动，导致模型输出不正确。然而，现有的PAE面临着两个挑战：攻击性能不令人满意(即可转移性差，对环境条件的健壮性不够)，以及难以平衡攻击有效性和隐蔽性，更好的攻击效率往往使PAE更容易被感知。在本文中，我们探索了一种新的基于扰动的方法来克服这些挑战。对于第一个挑战，我们引入了一种基于稳健特征(RF)的欺骗性射频注入策略，这些特征具有预测性、对扰动具有鲁棒性，并且在不同的模型中保持一致。具体地说，它通过将其他类的RF覆盖到干净图像中的预测特征来提高PAE的可转移性和稳健性。对于第二个挑战，我们引入了另一种对抗性语义模式最小化策略，该策略去除了大部分扰动，只保留了AEss中的基本对抗性模式。在这两种策略的基础上，我们设计了一种鲁棒特征覆盖攻击(RFCoA)方法，包括健壮特征解缠和对抗性特征融合。在第一阶段，我们在特征空间中提取目标类RFS。在第二阶段，我们使用基于注意力的特征融合将这些RF叠加到干净图像的预测特征上，并去除不必要的扰动。实验表明，与现有最先进的方法相比，我们的方法具有更好的可转移性、健壮性和隐蔽性。此外，我们的方法的有效性可以扩展到大型视觉语言模型(LVLM)，这表明它对更复杂的任务具有潜在的适用性。



## **41. NumbOD: A Spatial-Frequency Fusion Attack Against Object Detectors**

NumbOD：针对目标检测器的空频融合攻击 cs.CV

Accepted by AAAI 2025

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.16955v1) [paper-pdf](http://arxiv.org/pdf/2412.16955v1)

**Authors**: Ziqi Zhou, Bowen Li, Yufei Song, Zhifei Yu, Shengshan Hu, Wei Wan, Leo Yu Zhang, Dezhong Yao, Hai Jin

**Abstract**: With the advancement of deep learning, object detectors (ODs) with various architectures have achieved significant success in complex scenarios like autonomous driving. Previous adversarial attacks against ODs have been focused on designing customized attacks targeting their specific structures (e.g., NMS and RPN), yielding some results but simultaneously constraining their scalability. Moreover, most efforts against ODs stem from image-level attacks originally designed for classification tasks, resulting in redundant computations and disturbances in object-irrelevant areas (e.g., background). Consequently, how to design a model-agnostic efficient attack to comprehensively evaluate the vulnerabilities of ODs remains challenging and unresolved. In this paper, we propose NumbOD, a brand-new spatial-frequency fusion attack against various ODs, aimed at disrupting object detection within images. We directly leverage the features output by the OD without relying on its internal structures to craft adversarial examples. Specifically, we first design a dual-track attack target selection strategy to select high-quality bounding boxes from OD outputs for targeting. Subsequently, we employ directional perturbations to shift and compress predicted boxes and change classification results to deceive ODs. Additionally, we focus on manipulating the high-frequency components of images to confuse ODs' attention on critical objects, thereby enhancing the attack efficiency. Our extensive experiments on nine ODs and two datasets show that NumbOD achieves powerful attack performance and high stealthiness.

摘要: 随着深度学习的发展，各种结构的对象检测器在自动驾驶等复杂场景中取得了巨大的成功。以往针对OD的对抗性攻击都集中在针对其特定结构(如NMS和RPN)设计定制攻击，取得了一些效果，但同时也限制了其可扩展性。此外，大多数针对OD的努力源于最初为分类任务设计的图像级攻击，导致与对象无关的区域(例如背景)的冗余计算和干扰。因此，如何设计一种模型不可知的高效攻击来全面评估入侵检测的脆弱性仍然是一个挑战和悬而未决的问题。在本文中，我们提出了一种全新的针对各种OD的空频融合攻击，旨在扰乱图像中的目标检测。我们直接利用OD输出的功能，而不依赖其内部结构来创建对抗性的例子。具体地说，我们首先设计了一种双轨攻击目标选择策略，从OD输出中选择高质量的边界框进行目标定位。随后，我们使用方向扰动来移动和压缩预测框，并改变分类结果来欺骗OD。此外，我们还利用图像的高频成分来混淆OD对关键目标的注意力，从而提高了攻击效率。我们在9个OD和2个数据集上的大量实验表明，NumbOD具有强大的攻击性能和高隐蔽性。



## **42. Preventing Non-intrusive Load Monitoring Privacy Invasion: A Precise Adversarial Attack Scheme for Networked Smart Meters**

防止非侵入性负载监控隐私入侵：针对网络智能电表的精确对抗攻击方案 cs.CR

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.16893v1) [paper-pdf](http://arxiv.org/pdf/2412.16893v1)

**Authors**: Jialing He, Jiacheng Wang, Ning Wang, Shangwei Guo, Liehuang Zhu, Dusit Niyato, Tao Xiang

**Abstract**: Smart grid, through networked smart meters employing the non-intrusive load monitoring (NILM) technique, can considerably discern the usage patterns of residential appliances. However, this technique also incurs privacy leakage. To address this issue, we propose an innovative scheme based on adversarial attack in this paper. The scheme effectively prevents NILM models from violating appliance-level privacy, while also ensuring accurate billing calculation for users. To achieve this objective, we overcome two primary challenges. First, as NILM models fall under the category of time-series regression models, direct application of traditional adversarial attacks designed for classification tasks is not feasible. To tackle this issue, we formulate a novel adversarial attack problem tailored specifically for NILM and providing a theoretical foundation for utilizing the Jacobian of the NILM model to generate imperceptible perturbations. Leveraging the Jacobian, our scheme can produce perturbations, which effectively misleads the signal prediction of NILM models to safeguard users' appliance-level privacy. The second challenge pertains to fundamental utility requirements, where existing adversarial attack schemes struggle to achieve accurate billing calculation for users. To handle this problem, we introduce an additional constraint, mandating that the sum of added perturbations within a billing period must be precisely zero. Experimental validation on real-world power datasets REDD and UK-DALE demonstrates the efficacy of our proposed solutions, which can significantly amplify the discrepancy between the output of the targeted NILM model and the actual power signal of appliances, and enable accurate billing at the same time. Additionally, our solutions exhibit transferability, making the generated perturbation signal from one target model applicable to other diverse NILM models.

摘要: 智能电网通过采用非侵入式负荷监测(NILM)技术的联网智能电表，可以相当程度地识别家用电器的使用模式。然而，这种技术也会导致隐私泄露。针对这一问题，本文提出了一种基于对抗性攻击的创新方案。该方案有效地防止了NILM模型侵犯设备级隐私，同时还确保了用户准确的计费计算。为了实现这一目标，我们克服了两个主要挑战。首先，由于NILM模型属于时间序列回归模型的范畴，直接应用传统的针对分类任务的对抗性攻击是不可行的。为了解决这一问题，我们提出了一种新的针对NILM的对抗性攻击问题，为利用NILM模型的雅可比产生不可察觉的扰动提供了理论基础。利用雅可比矩阵，我们的方案可以产生扰动，从而有效地误导NILM模型的信号预测，以保护用户的家用电器级别的隐私。第二个挑战与基本的效用要求有关，现有的对抗性攻击方案难以为用户实现准确的计费计算。为了处理这个问题，我们引入了一个额外的约束，要求在一个计费周期内添加的扰动之和必须正好为零。在真实电力数据集REDD和UK-Dale上的实验验证表明，我们提出的解决方案是有效的，可以显著放大目标NILM模型的输出与家电实际电力信号之间的差异，同时实现准确的计费。此外，我们的解表现出可移植性，使得从一个目标模型产生的微扰信号适用于其他不同的NILM模型。



## **43. Towards More Robust Retrieval-Augmented Generation: Evaluating RAG Under Adversarial Poisoning Attacks**

迈向更稳健的检索增强生成：在对抗性中毒攻击下评估RAG cs.IR

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16708v1) [paper-pdf](http://arxiv.org/pdf/2412.16708v1)

**Authors**: Jinyan Su, Jin Peng Zhou, Zhengxin Zhang, Preslav Nakov, Claire Cardie

**Abstract**: Retrieval-Augmented Generation (RAG) systems have emerged as a promising solution to mitigate LLM hallucinations and enhance their performance in knowledge-intensive domains. However, these systems are vulnerable to adversarial poisoning attacks, where malicious passages injected into retrieval databases can mislead the model into generating factually incorrect outputs. In this paper, we investigate both the retrieval and the generation components of RAG systems to understand how to enhance their robustness against such attacks. From the retrieval perspective, we analyze why and how the adversarial contexts are retrieved and assess how the quality of the retrieved passages impacts downstream generation. From a generation perspective, we evaluate whether LLMs' advanced critical thinking and internal knowledge capabilities can be leveraged to mitigate the impact of adversarial contexts, i.e., using skeptical prompting as a self-defense mechanism. Our experiments and findings provide actionable insights into designing safer and more resilient retrieval-augmented frameworks, paving the way for their reliable deployment in real-world applications.

摘要: 提取-增强生成(RAG)系统已经成为缓解LLM幻觉和提高其在知识密集型领域的表现的一种有前途的解决方案。然而，这些系统容易受到对抗性中毒攻击，在这种攻击中，注入检索数据库的恶意段落可能会误导模型生成实际不正确的输出。在本文中，我们研究了RAG系统的检索组件和生成组件，以了解如何增强其对此类攻击的健壮性。从提取的角度，我们分析了为什么以及如何提取对抗性语境，并评估了所检索的段落的质量如何影响下游生成。从一代人的角度，我们评估了LLMS的高级批判性思维和内部知识能力是否可以被用来减轻对手环境的影响，即使用怀疑提示作为一种自卫机制。我们的实验和发现为设计更安全、更具弹性的检索增强框架提供了可行的见解，为它们在现实世界的应用程序中的可靠部署铺平了道路。



## **44. PB-UAP: Hybrid Universal Adversarial Attack For Image Segmentation**

PB-UAP：图像分割的混合通用对抗攻击 cs.CV

Accepted by ICASSP 2025

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16651v1) [paper-pdf](http://arxiv.org/pdf/2412.16651v1)

**Authors**: Yufei Song, Ziqi Zhou, Minghui Li, Xianlong Wang, Menghao Deng, Wei Wan, Shengshan Hu, Leo Yu Zhang

**Abstract**: With the rapid advancement of deep learning, the model robustness has become a significant research hotspot, \ie, adversarial attacks on deep neural networks. Existing works primarily focus on image classification tasks, aiming to alter the model's predicted labels. Due to the output complexity and deeper network architectures, research on adversarial examples for segmentation models is still limited, particularly for universal adversarial perturbations. In this paper, we propose a novel universal adversarial attack method designed for segmentation models, which includes dual feature separation and low-frequency scattering modules. The two modules guide the training of adversarial examples in the pixel and frequency space, respectively. Experiments demonstrate that our method achieves high attack success rates surpassing the state-of-the-art methods, and exhibits strong transferability across different models.

摘要: 随着深度学习的快速发展，模型鲁棒性已成为一个重要的研究热点，即对深度神经网络的对抗性攻击。现有的工作主要集中在图像分类任务上，旨在改变模型的预测标签。由于输出复杂性和更深层次的网络架构，对分段模型对抗性示例的研究仍然有限，特别是对于普遍对抗性扰动。本文提出了一种针对分割模型设计的新型通用对抗攻击方法，其中包括双重特征分离和低频散射模块。这两个模块分别指导像素和频率空间中对抗性示例的训练。实验表明，我们的方法比最先进的方法具有更高的攻击成功率，并且在不同模型之间表现出很强的可移植性。



## **45. POEX: Policy Executable Embodied AI Jailbreak Attacks**

POEX：政策可执行性许可人工智能越狱攻击 cs.RO

Homepage: https://poex-eai-jailbreak.github.io/

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16633v1) [paper-pdf](http://arxiv.org/pdf/2412.16633v1)

**Authors**: Xuancun Lu, Zhengxian Huang, Xinfeng Li, Xiaoyu ji, Wenyuan Xu

**Abstract**: The integration of large language models (LLMs) into the planning module of Embodied Artificial Intelligence (Embodied AI) systems has greatly enhanced their ability to translate complex user instructions into executable policies. In this paper, we demystified how traditional LLM jailbreak attacks behave in the Embodied AI context. We conducted a comprehensive safety analysis of the LLM-based planning module of embodied AI systems against jailbreak attacks. Using the carefully crafted Harmful-RLbench, we accessed 20 open-source and proprietary LLMs under traditional jailbreak attacks, and highlighted two key challenges when adopting the prior jailbreak techniques to embodied AI contexts: (1) The harmful text output by LLMs does not necessarily induce harmful policies in Embodied AI context, and (2) even we can generate harmful policies, we have to guarantee they are executable in practice. To overcome those challenges, we propose Policy Executable (POEX) jailbreak attacks, where harmful instructions and optimized suffixes are injected into LLM-based planning modules, leading embodied AI to perform harmful actions in both simulated and physical environments. Our approach involves constraining adversarial suffixes to evade detection and fine-tuning a policy evaluater to improve the executability of harmful policies. We conducted extensive experiments on both a robotic arm embodied AI platform and simulators, to validate the attack and policy success rates on 136 harmful instructions from Harmful-RLbench. Our findings expose serious safety vulnerabilities in LLM-based planning modules, including the ability of POEX to be transferred across models. Finally, we propose mitigation strategies, such as safety-constrained prompts, pre- and post-planning checks, to address these vulnerabilities and ensure the safe deployment of embodied AI in real-world settings.

摘要: 将大型语言模型(LLM)集成到嵌入式人工智能(Embedded AI)系统的规划模块中，极大地增强了它们将复杂的用户指令转换为可执行策略的能力。在这篇文章中，我们揭开了传统的LLM越狱攻击在具体的人工智能上下文中的行为。我们对基于LLM的具体化人工智能系统抗越狱攻击规划模块进行了全面的安全分析。使用精心制作的有害RLbench，我们在传统越狱攻击下访问了20个开源和专有的LLM，并强调了采用先前的越狱技术来体现AI上下文时的两个关键挑战：(1)LLMS输出的有害文本不一定会导致体现AI上下文中的有害策略，以及(2)即使我们可以生成有害策略，我们也必须确保它们在实践中是可执行的。为了克服这些挑战，我们提出了策略可执行(POEX)越狱攻击，将有害指令和优化后缀注入基于LLM的规划模块，导致嵌入式AI在模拟和物理环境中执行有害操作。我们的方法包括限制敌意后缀以逃避检测，以及微调策略评估器以提高有害策略的可执行性。我们在一个机械臂体现的人工智能平台和模拟器上进行了广泛的实验，以验证对来自有害RLbench的136条有害指令的攻击和策略成功率。我们的发现暴露了基于LLM的计划模块中的严重安全漏洞，包括POEX跨模型传输的能力。最后，我们提出了缓解策略，如安全约束提示，规划前和规划后检查，以应对这些漏洞，并确保体现的人工智能在现实世界中的安全部署。



## **46. Automated Progressive Red Teaming**

自动化渐进式红色团队 cs.CR

Accepted by COLING 2025

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2407.03876v3) [paper-pdf](http://arxiv.org/pdf/2407.03876v3)

**Authors**: Bojian Jiang, Yi Jing, Tianhao Shen, Tong Wu, Qing Yang, Deyi Xiong

**Abstract**: Ensuring the safety of large language models (LLMs) is paramount, yet identifying potential vulnerabilities is challenging. While manual red teaming is effective, it is time-consuming, costly and lacks scalability. Automated red teaming (ART) offers a more cost-effective alternative, automatically generating adversarial prompts to expose LLM vulnerabilities. However, in current ART efforts, a robust framework is absent, which explicitly frames red teaming as an effectively learnable task. To address this gap, we propose Automated Progressive Red Teaming (APRT) as an effectively learnable framework. APRT leverages three core modules: an Intention Expanding LLM that generates diverse initial attack samples, an Intention Hiding LLM that crafts deceptive prompts, and an Evil Maker to manage prompt diversity and filter ineffective samples. The three modules collectively and progressively explore and exploit LLM vulnerabilities through multi-round interactions. In addition to the framework, we further propose a novel indicator, Attack Effectiveness Rate (AER) to mitigate the limitations of existing evaluation metrics. By measuring the likelihood of eliciting unsafe but seemingly helpful responses, AER aligns closely with human evaluations. Extensive experiments with both automatic and human evaluations, demonstrate the effectiveness of ARPT across both open- and closed-source LLMs. Specifically, APRT effectively elicits 54% unsafe yet useful responses from Meta's Llama-3-8B-Instruct, 50% from GPT-4o (API access), and 39% from Claude-3.5 (API access), showcasing its robust attack capability and transferability across LLMs (especially from open-source LLMs to closed-source LLMs).

摘要: 确保大型语言模型(LLM)的安全是最重要的，但识别潜在的漏洞是具有挑战性的。虽然手动红色团队是有效的，但它耗时、成本高，而且缺乏可扩展性。自动红色团队(ART)提供了一种更具成本效益的替代方案，可自动生成敌意提示以暴露LLM漏洞。然而，在目前的艺术努力中，缺乏一个强大的框架，它明确地将红色团队作为一项有效的可学习任务。为了弥补这一差距，我们提出了自动渐进红色团队(APRT)作为一种有效的可学习框架。APRT利用三个核心模块：用于生成不同初始攻击样本的意图扩展LLM，用于制作欺骗性提示的意图隐藏LLM，以及用于管理提示多样性和过滤无效样本的邪恶制造者。这三个模块通过多轮交互共同逐步探索和利用LLM漏洞。除了该框架外，我们进一步提出了一个新的指标--攻击效率(AER)，以缓解现有评估指标的局限性。通过衡量引发不安全但似乎有帮助的反应的可能性，AER与人类的评估密切一致。自动和人工评估的广泛实验证明了ARPT在开放源码和封闭源码LLM中的有效性。具体地说，APRT有效地从Meta的Llama-3-8B-Indict、GPT-40(API访问)和Claude-3.5(API访问)中引发了54%的不安全但有用的响应，展示了其强大的攻击能力和跨LLM(特别是从开源LLM到闭源LLM)的可转移性。



## **47. PGD-Imp: Rethinking and Unleashing Potential of Classic PGD with Dual Strategies for Imperceptible Adversarial Attacks**

PGD-Imp：通过双重策略重新思考和释放经典PVD的潜力，以应对难以感知的对抗攻击 cs.LG

accepted by ICASSP 2025

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.11168v2) [paper-pdf](http://arxiv.org/pdf/2412.11168v2)

**Authors**: Jin Li, Zitong Yu, Ziqiang He, Z. Jane Wang, Xiangui Kang

**Abstract**: Imperceptible adversarial attacks have recently attracted increasing research interests. Existing methods typically incorporate external modules or loss terms other than a simple $l_p$-norm into the attack process to achieve imperceptibility, while we argue that such additional designs may not be necessary. In this paper, we rethink the essence of imperceptible attacks and propose two simple yet effective strategies to unleash the potential of PGD, the common and classical attack, for imperceptibility from an optimization perspective. Specifically, the Dynamic Step Size is introduced to find the optimal solution with minimal attack cost towards the decision boundary of the attacked model, and the Adaptive Early Stop strategy is adopted to reduce the redundant strength of adversarial perturbations to the minimum level. The proposed PGD-Imperceptible (PGD-Imp) attack achieves state-of-the-art results in imperceptible adversarial attacks for both untargeted and targeted scenarios. When performing untargeted attacks against ResNet-50, PGD-Imp attains 100$\%$ (+0.3$\%$) ASR, 0.89 (-1.76) $l_2$ distance, and 52.93 (+9.2) PSNR with 57s (-371s) running time, significantly outperforming existing methods.

摘要: 潜伏的敌意攻击最近吸引了越来越多的研究兴趣。现有的方法通常在攻击过程中加入外部模块或损失项，而不是简单的$L_p$-范数来实现不可感知性，而我们认为这样的额外设计可能不是必要的。本文从优化的角度重新思考了不可察觉攻击的本质，并提出了两种简单而有效的策略来释放PGD攻击--普通攻击和经典攻击--的不可感知性。具体地，引入动态步长在攻击模型的决策边界附近寻找攻击代价最小的最优解，并采用自适应提前停止策略将敌方扰动的冗余强度降至最小。建议的PGD-Imp(PGD-Imp)攻击在非目标场景和目标场景中都实现了最先进的不可感知对手攻击。在对ResNet-50进行非定向攻击时，PGD-Imp在57s(-371s)的运行时间内获得了100$(+0.3$)ASR，0.89(-1.76)$L_2$距离和52.93(+9.2)PSNR，显著优于现有方法。



## **48. WiP: Deception-in-Depth Using Multiple Layers of Deception**

WiP：使用多层欺骗进行深度欺骗 cs.CR

Presented at HoTSoS 2024

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16430v1) [paper-pdf](http://arxiv.org/pdf/2412.16430v1)

**Authors**: Jason Landsborough, Neil C. Rowe, Thuy D. Nguyen, Sunny Fugate

**Abstract**: Deception is being increasingly explored as a cyberdefense strategy to protect operational systems. We are studying implementation of deception-in-depth strategies with initially three logical layers: network, host, and data. We draw ideas from military deception, network orchestration, software deception, file deception, fake honeypots, and moving-target defenses. We are building a prototype representing our ideas and will be testing it in several adversarial environments. We hope to show that deploying a broad range of deception techniques can be more effective in protecting systems than deploying single techniques. Unlike traditional deception methods that try to encourage active engagement from attackers to collect intelligence, we focus on deceptions that can be used on real machines to discourage attacks.

摘要: 欺骗作为一种保护操作系统的网络防御策略正在被越来越多地探索。我们正在研究深度欺骗策略的实施，最初分为三个逻辑层：网络、主机和数据。我们从军事欺骗、网络编排、软件欺骗、文件欺骗、假蜜罐和移动目标防御中汲取灵感。我们正在构建一个代表我们想法的原型，并将在几个对抗环境中对其进行测试。我们希望证明，部署广泛的欺骗技术比部署单一技术在保护系统方面更有效。与试图鼓励攻击者积极参与收集情报的传统欺骗方法不同，我们专注于可在真实机器上使用以阻止攻击的欺骗方法。



## **49. Chain-of-Scrutiny: Detecting Backdoor Attacks for Large Language Models**

审查链：检测大型语言模型的后门攻击 cs.CR

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2406.05948v2) [paper-pdf](http://arxiv.org/pdf/2406.05948v2)

**Authors**: Xi Li, Yusen Zhang, Renze Lou, Chen Wu, Jiaqi Wang

**Abstract**: Large Language Models (LLMs), especially those accessed via APIs, have demonstrated impressive capabilities across various domains. However, users without technical expertise often turn to (untrustworthy) third-party services, such as prompt engineering, to enhance their LLM experience, creating vulnerabilities to adversarial threats like backdoor attacks. Backdoor-compromised LLMs generate malicious outputs to users when inputs contain specific "triggers" set by attackers. Traditional defense strategies, originally designed for small-scale models, are impractical for API-accessible LLMs due to limited model access, high computational costs, and data requirements. To address these limitations, we propose Chain-of-Scrutiny (CoS) which leverages LLMs' unique reasoning abilities to mitigate backdoor attacks. It guides the LLM to generate reasoning steps for a given input and scrutinizes for consistency with the final output -- any inconsistencies indicating a potential attack. It is well-suited for the popular API-only LLM deployments, enabling detection at minimal cost and with little data. User-friendly and driven by natural language, it allows non-experts to perform the defense independently while maintaining transparency. We validate the effectiveness of CoS through extensive experiments on various tasks and LLMs, with results showing greater benefits for more powerful LLMs.

摘要: 大型语言模型(LLM)，特别是那些通过API访问的模型，已经在各个领域展示了令人印象深刻的能力。然而，没有技术专业知识的用户通常会求助于(不值得信任的)第三方服务，如提示工程，以增强他们的LLM体验，从而对后门攻击等对手威胁造成漏洞。当输入包含攻击者设置的特定“触发器”时，受后门攻击的LLM会向用户生成恶意输出。传统的防御策略最初是为小规模模型设计的，由于模型访问有限、计算成本高和数据要求高，对于API可访问的LLM来说是不切实际的。为了解决这些局限性，我们提出了审查链(CoS)，它利用LLMS的独特推理能力来减少后门攻击。它指导LLM为给定的输入生成推理步骤，并仔细检查与最终输出的一致性--任何指示潜在攻击的不一致。它非常适合流行的纯API LLM部署，能够以最低的成本和很少的数据进行检测。它用户友好，由自然语言驱动，允许非专家独立进行辩护，同时保持透明度。我们通过在不同任务和LLM上的大量实验验证了CoS的有效性，结果表明，更强大的LLM具有更大的好处。



## **50. EMPRA: Embedding Perturbation Rank Attack against Neural Ranking Models**

EMPRA：针对神经排名模型的嵌入扰动排名攻击 cs.IR

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.16382v1) [paper-pdf](http://arxiv.org/pdf/2412.16382v1)

**Authors**: Amin Bigdeli, Negar Arabzadeh, Ebrahim Bagheri, Charles L. A. Clarke

**Abstract**: Recent research has shown that neural information retrieval techniques may be susceptible to adversarial attacks. Adversarial attacks seek to manipulate the ranking of documents, with the intention of exposing users to targeted content. In this paper, we introduce the Embedding Perturbation Rank Attack (EMPRA) method, a novel approach designed to perform adversarial attacks on black-box Neural Ranking Models (NRMs). EMPRA manipulates sentence-level embeddings, guiding them towards pertinent context related to the query while preserving semantic integrity. This process generates adversarial texts that seamlessly integrate with the original content and remain imperceptible to humans. Our extensive evaluation conducted on the widely-used MS MARCO V1 passage collection demonstrate the effectiveness of EMPRA against a wide range of state-of-the-art baselines in promoting a specific set of target documents within a given ranked results. Specifically, EMPRA successfully achieves a re-ranking of almost 96% of target documents originally ranked between 51-100 to rank within the top 10. Furthermore, EMPRA does not depend on surrogate models for adversarial text generation, enhancing its robustness against different NRMs in realistic settings.

摘要: 最近的研究表明，神经信息检索技术可能容易受到对抗性攻击。敌意攻击试图操纵文档的排名，目的是让用户接触到有针对性的内容。本文介绍了一种新的针对黑盒神经网络排名模型(NRM)的对抗性攻击方法--嵌入扰动等级攻击方法。EMPRA操作语句级别的嵌入，引导它们指向与查询相关的上下文，同时保持语义完整性。这一过程产生了与原始内容无缝集成的对抗性文本，并且对人类来说仍然是不可感知的。我们对广泛使用的MS Marco V1文章集进行了广泛的评估，证明了EMPRA在推广给定排名结果中的一组特定目标文档方面相对于广泛的最先进基线的有效性。具体地说，EMPRA成功地实现了几乎96%的目标文档的重新排序，这些文档最初的排名在51-100之间，进入前10名。此外，EMPRA不依赖于生成敌意文本的代理模型，增强了它在现实环境中对不同NRM的稳健性。



