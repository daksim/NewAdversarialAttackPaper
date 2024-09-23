# Latest Adversarial Attack Papers
**update at 2024-09-23 11:26:14**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Neurosymbolic Conformal Classification**

神经符号保形分类 cs.LG

10 pages, 0 figures. arXiv admin note: text overlap with  arXiv:2404.08404

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13585v1) [paper-pdf](http://arxiv.org/pdf/2409.13585v1)

**Authors**: Arthur Ledaguenel, Céline Hudelot, Mostepha Khouadjia

**Abstract**: The last decades have seen a drastic improvement of Machine Learning (ML), mainly driven by Deep Learning (DL). However, despite the resounding successes of ML in many domains, the impossibility to provide guarantees of conformity and the fragility of ML systems (faced with distribution shifts, adversarial attacks, etc.) have prevented the design of trustworthy AI systems. Several research paths have been investigated to mitigate this fragility and provide some guarantees regarding the behavior of ML systems, among which are neurosymbolic AI and conformal prediction. Neurosymbolic artificial intelligence is a growing field of research aiming to combine neural network learning capabilities with the reasoning abilities of symbolic systems. One of the objective of this hybridization can be to provide theoritical guarantees that the output of the system will comply with some prior knowledge. Conformal prediction is a set of techniques that enable to take into account the uncertainty of ML systems by transforming the unique prediction into a set of predictions, called a confidence set. Interestingly, this comes with statistical guarantees regarding the presence of the true label inside the confidence set. Both approaches are distribution-free and model-agnostic. In this paper, we see how these two approaches can complement one another. We introduce several neurosymbolic conformal prediction techniques and explore their different characteristics (size of confidence sets, computational complexity, etc.).

摘要: 在过去的几十年里，机器学习(ML)有了巨大的进步，这主要是由深度学习(DL)推动的。然而，尽管ML在许多领域取得了巨大成功，但无法保证一致性和ML系统的脆弱性(面临分布变化、对抗性攻击等)。阻碍了值得信赖的人工智能系统的设计。为了缓解这种脆弱性，并为ML系统的行为提供一些保证，人们已经研究了几条途径，其中包括神经符号人工智能和保形预测。神经符号人工智能是一个不断发展的研究领域，旨在将神经网络的学习能力与符号系统的推理能力结合起来。这种杂交的目标之一可以是提供理论上的保证，即系统的输出将符合某些先验知识。保角预测是一组技术，它通过将唯一的预测转换为一组预测，称为置信度集，从而能够考虑ML系统的不确定性。有趣的是，这伴随着关于置信度集中存在真实标签的统计保证。这两种方法都不依赖于分发，也不依赖于模型。在这篇文章中，我们看到了这两种方法是如何相互补充的。我们介绍了几种神经符号保形预测技术，并探讨了它们的不同特点(置信集的大小、计算复杂度等)。



## **2. Efficient Visualization of Neural Networks with Generative Models and Adversarial Perturbations**

具有生成模型和对抗性扰动的神经网络的有效可视化 cs.CV

4 pages, 3 figures

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13559v1) [paper-pdf](http://arxiv.org/pdf/2409.13559v1)

**Authors**: Athanasios Karagounis

**Abstract**: This paper presents a novel approach for deep visualization via a generative network, offering an improvement over existing methods. Our model simplifies the architecture by reducing the number of networks used, requiring only a generator and a discriminator, as opposed to the multiple networks traditionally involved. Additionally, our model requires less prior training knowledge and uses a non-adversarial training process, where the discriminator acts as a guide rather than a competitor to the generator. The core contribution of this work is its ability to generate detailed visualization images that align with specific class labels. Our model incorporates a unique skip-connection-inspired block design, which enhances label-directed image generation by propagating class information across multiple layers. Furthermore, we explore how these generated visualizations can be utilized as adversarial examples, effectively fooling classification networks with minimal perceptible modifications to the original images. Experimental results demonstrate that our method outperforms traditional adversarial example generation techniques in both targeted and non-targeted attacks, achieving up to a 94.5% fooling rate with minimal perturbation. This work bridges the gap between visualization methods and adversarial examples, proposing that fooling rate could serve as a quantitative measure for evaluating visualization quality. The insights from this study provide a new perspective on the interpretability of neural networks and their vulnerabilities to adversarial attacks.

摘要: 本文提出了一种新的基于产生式网络的深度可视化方法，对现有方法进行了改进。我们的模型通过减少使用的网络数量简化了体系结构，只需要一个生成器和一个鉴别器，而不是传统上涉及的多个网络。此外，我们的模型需要较少的先验训练知识，并使用非对抗性训练过程，其中鉴别器充当生成器的指南而不是竞争者。这项工作的核心贡献是它能够生成与特定类标签一致的详细可视化图像。我们的模型结合了一种独特的跳跃连接启发块设计，通过在多个层上传播类信息来增强标签导向图像的生成。此外，我们还探讨了如何利用这些生成的可视化作为对抗性的例子，有效地愚弄分类网络，对原始图像进行最小可感知的修改。实验结果表明，在目标攻击和非目标攻击中，我们的方法都优于传统的敌意样本生成技术，在最小扰动的情况下，可以达到94.5%的愚弄率。这项工作弥合了可视化方法和对抗性例子之间的差距，提出了傻瓜率可以作为评估可视化质量的量化指标。这项研究的洞察力为神经网络的可解释性及其对抗攻击的脆弱性提供了一个新的视角。



## **3. Deterministic versus stochastic dynamical classifiers: opposing random adversarial attacks with noise**

确定性与随机动态分类器：对抗带有噪音的随机对抗攻击 cs.LG

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13470v1) [paper-pdf](http://arxiv.org/pdf/2409.13470v1)

**Authors**: Lorenzo Chicchi, Duccio Fanelli, Diego Febbe, Lorenzo Buffoni, Francesca Di Patti, Lorenzo Giambagli, Raffele Marino

**Abstract**: The Continuous-Variable Firing Rate (CVFR) model, widely used in neuroscience to describe the intertangled dynamics of excitatory biological neurons, is here trained and tested as a veritable dynamically assisted classifier. To this end the model is supplied with a set of planted attractors which are self-consistently embedded in the inter-nodes coupling matrix, via its spectral decomposition. Learning to classify amounts to sculp the basin of attraction of the imposed equilibria, directing different items towards the corresponding destination target, which reflects the class of respective pertinence. A stochastic variant of the CVFR model is also studied and found to be robust to aversarial random attacks, which corrupt the items to be classified. This remarkable finding is one of the very many surprising effects which arise when noise and dynamical attributes are made to mutually resonate.

摘要: 连续可变放电率（CVFR）模型广泛用于神经科学，用于描述兴奋性生物神经元的相互交织的动力学，在这里作为名副其实的动态辅助分类器进行训练和测试。为此，该模型配备了一组种植吸引子，这些吸引子通过其谱分解自一致地嵌入到节点间耦合矩阵中。学习分类相当于雕刻强加均衡的吸引力盆地，将不同的物品引导到相应的目的地目标，这反映了各自相关性的类别。还研究了CVFR模型的一种随机变体，发现它对敌对随机攻击具有鲁棒性，这些攻击会破坏要分类的项目。这一非凡的发现是当噪音和动态属性相互共振时产生的众多令人惊讶的影响之一。



## **4. Celtibero: Robust Layered Aggregation for Federated Learning**

Celtibero：用于联邦学习的稳健分层聚合 cs.CR

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2408.14240v2) [paper-pdf](http://arxiv.org/pdf/2408.14240v2)

**Authors**: Borja Molina-Coronado

**Abstract**: Federated Learning (FL) is an innovative approach to distributed machine learning. While FL offers significant privacy advantages, it also faces security challenges, particularly from poisoning attacks where adversaries deliberately manipulate local model updates to degrade model performance or introduce hidden backdoors. Existing defenses against these attacks have been shown to be effective when the data on the nodes is identically and independently distributed (i.i.d.), but they often fail under less restrictive, non-i.i.d data conditions. To overcome these limitations, we introduce Celtibero, a novel defense mechanism that integrates layered aggregation to enhance robustness against adversarial manipulation. Through extensive experiments on the MNIST and IMDB datasets, we demonstrate that Celtibero consistently achieves high main task accuracy (MTA) while maintaining minimal attack success rates (ASR) across a range of untargeted and targeted poisoning attacks. Our results highlight the superiority of Celtibero over existing defenses such as FL-Defender, LFighter, and FLAME, establishing it as a highly effective solution for securing federated learning systems against sophisticated poisoning attacks.

摘要: 联合学习(FL)是分布式机器学习的一种创新方法。虽然FL提供了显著的隐私优势，但它也面临着安全挑战，特别是来自毒化攻击的挑战，即攻击者故意操纵本地模型更新以降低模型性能或引入隐藏后门。当节点上的数据相同且独立分布(I.I.D.)时，现有的针对这些攻击的防御已被证明是有效的，但在限制较少的非I.I.D.数据条件下，它们通常会失败。为了克服这些局限性，我们引入了Celtibero，这是一种新的防御机制，它集成了分层聚合来增强对对手操纵的健壮性。通过在MNIST和IMDB数据集上的广泛实验，我们证明了Celtibero在一系列非目标和目标中毒攻击中始终实现了高的主任务准确率(MTA)，同时保持了最低的攻击成功率(ASR)。我们的结果突出了Celtibero相对于FL-Defender、LFighter和FAME等现有防御系统的优势，使其成为保护联邦学习系统免受复杂中毒攻击的高效解决方案。



## **5. Enhancing Transferability of Adversarial Attacks with GE-AdvGAN+: A Comprehensive Framework for Gradient Editing**

利用GE-AdvGAN+增强对抗性攻击的可转移性：梯度编辑的综合框架 cs.AI

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2408.12673v3) [paper-pdf](http://arxiv.org/pdf/2408.12673v3)

**Authors**: Zhibo Jin, Jiayu Zhang, Zhiyu Zhu, Chenyu Zhang, Jiahao Huang, Jianlong Zhou, Fang Chen

**Abstract**: Transferable adversarial attacks pose significant threats to deep neural networks, particularly in black-box scenarios where internal model information is inaccessible. Studying adversarial attack methods helps advance the performance of defense mechanisms and explore model vulnerabilities. These methods can uncover and exploit weaknesses in models, promoting the development of more robust architectures. However, current methods for transferable attacks often come with substantial computational costs, limiting their deployment and application, especially in edge computing scenarios. Adversarial generative models, such as Generative Adversarial Networks (GANs), are characterized by their ability to generate samples without the need for retraining after an initial training phase. GE-AdvGAN, a recent method for transferable adversarial attacks, is based on this principle. In this paper, we propose a novel general framework for gradient editing-based transferable attacks, named GE-AdvGAN+, which integrates nearly all mainstream attack methods to enhance transferability while significantly reducing computational resource consumption. Our experiments demonstrate the compatibility and effectiveness of our framework. Compared to the baseline AdvGAN, our best-performing method, GE-AdvGAN++, achieves an average ASR improvement of 47.8. Additionally, it surpasses the latest competing algorithm, GE-AdvGAN, with an average ASR increase of 5.9. The framework also exhibits enhanced computational efficiency, achieving 2217.7 FPS, outperforming traditional methods such as BIM and MI-FGSM. The implementation code for our GE-AdvGAN+ framework is available at https://github.com/GEAdvGANP

摘要: 可转移的敌意攻击对深度神经网络构成重大威胁，特别是在内部模型信息不可访问的黑盒场景中。研究对抗性攻击方法有助于提高防御机制的性能，探索模型漏洞。这些方法可以发现和利用模型中的弱点，从而促进更健壮的体系结构的开发。然而，当前的可转移攻击方法往往伴随着巨大的计算成本，限制了它们的部署和应用，特别是在边缘计算场景中。对抗性生成模型，如生成性对抗性网络(GANS)，其特点是能够在初始训练阶段后生成样本，而不需要重新训练。GE-AdvGAN是一种新的可转移的对抗性攻击方法，它基于这一原理。本文提出了一种新颖的基于梯度编辑的可转移攻击通用框架GE-AdvGAN+，该框架集成了几乎所有的主流攻击方法，在提高可转移性的同时显著降低了计算资源消耗。我们的实验证明了该框架的兼容性和有效性。与基准的AdvGAN相比，我们性能最好的方法GE-AdvGAN++实现了平均47.8的ASR改进。此外，它还超过了最新的竞争算法GE-AdvGAN，平均ASR提高了5.9。该框架还表现出更高的计算效率，达到2217.7 FPS，优于传统的BIM和MI-FGSM方法。我们GE-AdvGan+框架的实现代码可在https://github.com/GEAdvGANP上获得



## **6. Relationship between Uncertainty in DNNs and Adversarial Attacks**

DNN的不确定性与对抗攻击之间的关系 cs.LG

review

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13232v1) [paper-pdf](http://arxiv.org/pdf/2409.13232v1)

**Authors**: Abigail Adeniran, Adewale Adeyemo

**Abstract**: Deep Neural Networks (DNNs) have achieved state of the art results and even outperformed human accuracy in many challenging tasks, leading to DNNs adoption in a variety of fields including natural language processing, pattern recognition, prediction, and control optimization. However, DNNs are accompanied by uncertainty about their results, causing them to predict an outcome that is either incorrect or outside of a certain level of confidence. These uncertainties stem from model or data constraints, which could be exacerbated by adversarial attacks. Adversarial attacks aim to provide perturbed input to DNNs, causing the DNN to make incorrect predictions or increase model uncertainty. In this review, we explore the relationship between DNN uncertainty and adversarial attacks, emphasizing how adversarial attacks might raise DNN uncertainty.

摘要: 深度神经网络（DNN）已实现最先进的结果，甚至在许多具有挑战性的任务中超过了人类的准确性，导致DNN在自然语言处理、模式识别、预测和控制优化等各个领域得到采用。然而，DNN伴随着结果的不确定性，导致它们预测的结果要么不正确，要么超出一定置信水平。这些不确定性源于模型或数据限制，对抗性攻击可能会加剧这种限制。对抗性攻击旨在向DNN提供受干扰的输入，导致DNN做出错误的预测或增加模型的不确定性。在这篇评论中，我们探讨了DNN不确定性和对抗性攻击之间的关系，强调了对抗性攻击如何提高DNN不确定性。



## **7. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

面对威胁的操纵：评估端到端视觉语言动作模型中的身体脆弱性 cs.CV

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13174v1) [paper-pdf](http://arxiv.org/pdf/2409.13174v1)

**Authors**: Hao Cheng, Erjia Xiao, Chengyuan Yu, Zhao Yao, Jiahang Cao, Qiang Zhang, Jiaxu Wang, Mengshu Sun, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical security threats.

摘要: 最近，在多模式大语言模型(MLLM)的推动下，视觉语言动作模型(VLAM)被提出以在机器人操作任务的开放词汇场景中实现更好的性能。由于操作任务涉及与物理世界的直接交互，因此确保该任务执行过程中的健壮性和安全性始终是一个非常关键的问题。本文通过综合当前MLLMS的安全研究现状和物理世界中操纵任务的具体应用场景，对VLAMS在面临潜在物理威胁的情况下进行综合评估。具体地说，我们提出了物理脆弱性评估管道(PVEP)，它可以结合尽可能多的视觉通道物理威胁来评估VLAMS的物理健壮性。PVEP中的物理威胁具体包括分发外、基于排版的视觉提示和敌意补丁攻击。通过比较VLAM在受到攻击前后的性能波动，我们提供了VLAM如何应对不同的物理安全威胁的概括性的文本bf{文本{分析}}。



## **8. Hidden Activations Are Not Enough: A General Approach to Neural Network Predictions**

隐藏的激活还不够：神经网络预测的通用方法 cs.LG

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13163v1) [paper-pdf](http://arxiv.org/pdf/2409.13163v1)

**Authors**: Samuel Leblanc, Aiky Rasolomanana, Marco Armenta

**Abstract**: We introduce a novel mathematical framework for analyzing neural networks using tools from quiver representation theory. This framework enables us to quantify the similarity between a new data sample and the training data, as perceived by the neural network. By leveraging the induced quiver representation of a data sample, we capture more information than traditional hidden layer outputs. This quiver representation abstracts away the complexity of the computations of the forward pass into a single matrix, allowing us to employ simple geometric and statistical arguments in a matrix space to study neural network predictions. Our mathematical results are architecture-agnostic and task-agnostic, making them broadly applicable. As proof of concept experiments, we apply our results for the MNIST and FashionMNIST datasets on the problem of detecting adversarial examples on different MLP architectures and several adversarial attack methods. Our experiments can be reproduced with our \href{https://github.com/MarcoArmenta/Hidden-Activations-are-not-Enough}{publicly available repository}.

摘要: 我们介绍了一个新的数学框架来分析神经网络使用箭图表示理论的工具。这个框架使我们能够量化新数据样本和训练数据之间的相似性，就像神经网络所感知的那样。通过利用数据样本的诱导抖动表示，我们捕获了比传统隐含层输出更多的信息。这种箭图表示将前向传递计算的复杂性抽象到单个矩阵中，允许我们在矩阵空间中使用简单的几何和统计论点来研究神经网络预测。我们的数学结果是体系结构不可知和任务不可知的，使它们具有广泛的适用性。作为概念验证实验，我们在MNIST和FashionMNIST数据集上应用我们的结果来检测不同MLP体系结构和几种对抗性攻击方法上的对抗性实例。我们的实验可以用我们的\href{https://github.com/MarcoArmenta/Hidden-Activations-are-not-Enough}{publicly Available存储库重现。



## **9. FedAT: Federated Adversarial Training for Distributed Insider Threat Detection**

FedAT：分布式内部威胁检测的联合对抗训练 cs.CR

10 pages, 7 figures

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.13083v1) [paper-pdf](http://arxiv.org/pdf/2409.13083v1)

**Authors**: R G Gayathri, Atul Sajjanhar, Md Palash Uddin, Yong Xiang

**Abstract**: Insider threats usually occur from within the workplace, where the attacker is an entity closely associated with the organization. The sequence of actions the entities take on the resources to which they have access rights allows us to identify the insiders. Insider Threat Detection (ITD) using Machine Learning (ML)-based approaches gained attention in the last few years. However, most techniques employed centralized ML methods to perform such an ITD. Organizations operating from multiple locations cannot contribute to the centralized models as the data is generated from various locations. In particular, the user behavior data, which is the primary source of ITD, cannot be shared among the locations due to privacy concerns. Additionally, the data distributed across various locations result in extreme class imbalance due to the rarity of attacks. Federated Learning (FL), a distributed data modeling paradigm, gained much interest recently. However, FL-enabled ITD is not yet explored, and it still needs research to study the significant issues of its implementation in practical settings. As such, our work investigates an FL-enabled multiclass ITD paradigm that considers non-Independent and Identically Distributed (non-IID) data distribution to detect insider threats from different locations (clients) of an organization. Specifically, we propose a Federated Adversarial Training (FedAT) approach using a generative model to alleviate the extreme data skewness arising from the non-IID data distribution among the clients. Besides, we propose to utilize a Self-normalized Neural Network-based Multi-Layer Perceptron (SNN-MLP) model to improve ITD. We perform comprehensive experiments and compare the results with the benchmarks to manifest the enhanced performance of the proposed FedATdriven ITD scheme.

摘要: 内部威胁通常发生在工作场所内部，攻击者是与组织密切相关的实体。实体对其拥有访问权限的资源采取的操作顺序使我们能够识别内部人员。使用基于机器学习(ML)方法的内部威胁检测(ITD)在过去几年中得到了关注。然而，大多数技术使用集中式ML方法来执行这样的ITD。由于数据是从多个位置生成的，因此在多个位置运营的组织不能对集中化模型做出贡献。特别是，由于隐私问题，作为ITD的主要来源的用户行为数据不能在不同地点之间共享。此外，由于攻击的罕见，分布在不同位置的数据导致了极端的类不平衡。联邦学习(FL)是一种分布式数据建模范式，近年来受到了极大的关注。然而，启用外语的信息技术开发还没有被探索，它仍然需要研究，以研究其在实际环境中实施的重大问题。因此，我们的工作研究了启用FL的多类ITD范例，该范例考虑非独立且相同分布(Non-IID)的数据分布，以检测来自组织不同位置(客户端)的内部威胁。具体地说，我们提出了一种使用产生式模型的联合对手训练(FedAT)方法，以缓解非IID数据在客户端之间分布时产生的极端数据偏斜。此外，我们还提出了一种基于自归一化神经网络的多层感知器(SNN-MLP)模型来改进ITD。我们进行了全面的实验，并将结果与基准测试结果进行了比较，以证明所提出的FedATDriven ITD方案具有更好的性能。



## **10. Defending against Reverse Preference Attacks is Difficult**

防御反向偏好攻击很困难 cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12914v1) [paper-pdf](http://arxiv.org/pdf/2409.12914v1)

**Authors**: Domenic Rosati, Giles Edkins, Harsh Raj, David Atanasov, Subhabrata Majumdar, Janarthanan Rajendran, Frank Rudzicz, Hassan Sajjad

**Abstract**: While there has been progress towards aligning Large Language Models (LLMs) with human values and ensuring safe behaviour at inference time, safety-aligned LLMs are known to be vulnerable to training-time attacks such as supervised fine-tuning (SFT) on harmful datasets. In this paper, we ask if LLMs are vulnerable to adversarial reinforcement learning. Motivated by this goal, we propose Reverse Preference Attacks (RPA), a class of attacks to make LLMs learn harmful behavior using adversarial reward during reinforcement learning from human feedback (RLHF). RPAs expose a critical safety gap of safety-aligned LLMs in RL settings: they easily explore the harmful text generation policies to optimize adversarial reward. To protect against RPAs, we explore a host of mitigation strategies. Leveraging Constrained Markov-Decision Processes, we adapt a number of mechanisms to defend against harmful fine-tuning attacks into the RL setting. Our experiments show that ``online" defenses that are based on the idea of minimizing the negative log likelihood of refusals -- with the defender having control of the loss function -- can effectively protect LLMs against RPAs. However, trying to defend model weights using ``offline" defenses that operate under the assumption that the defender has no control over the loss function are less effective in the face of RPAs. These findings show that attacks done using RL can be used to successfully undo safety alignment in open-weight LLMs and use them for malicious purposes.

摘要: 虽然在使大型语言模型(LLM)与人类价值观保持一致并确保推理时的安全行为方面取得了进展，但众所周知，与安全保持一致的LLM容易受到训练时的攻击，例如对有害数据集的监督微调(SFT)。在本文中，我们询问LLMS是否容易受到对抗性强化学习的影响。基于这一目标，我们提出了反向偏好攻击(RPA)，这是一类在人类反馈强化学习(RLHF)过程中利用对抗性奖励使LLM学习有害行为的攻击。RPA暴露了RL环境中安全对齐的LLM的一个关键安全漏洞：它们很容易探索有害文本生成策略，以优化对抗性奖励。为了防范RPA，我们探索了一系列缓解策略。利用受限的马尔可夫决策过程，我们采用了一些机制来防御RL设置中的有害微调攻击。我们的实验表明，基于最小化拒绝的负对数可能性的思想的“在线”防御--防御者控制着损失函数--可以有效地保护LLM免受RPA的攻击。然而，试图使用在防御者无法控制损失函数的假设下运行的“离线”防御来捍卫模型权重，在面对RPA时效果较差。这些发现表明，使用RL进行的攻击可以成功地取消开放重量LLM中的安全对齐，并将其用于恶意目的。



## **11. VCAT: Vulnerability-aware and Curiosity-driven Adversarial Training for Enhancing Autonomous Vehicle Robustness**

VCAT：脆弱性感知和好奇心驱动的对抗培训，以增强自动驾驶车辆的稳健性 cs.LG

7 pages, 5 figures, conference

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12997v1) [paper-pdf](http://arxiv.org/pdf/2409.12997v1)

**Authors**: Xuan Cai, Zhiyong Cui, Xuesong Bai, Ruimin Ke, Zhenshu Ma, Haiyang Yu, Yilong Ren

**Abstract**: Autonomous vehicles (AVs) face significant threats to their safe operation in complex traffic environments. Adversarial training has emerged as an effective method of enabling AVs to preemptively fortify their robustness against malicious attacks. Train an attacker using an adversarial policy, allowing the AV to learn robust driving through interaction with this attacker. However, adversarial policies in existing methodologies often get stuck in a loop of overexploiting established vulnerabilities, resulting in poor improvement for AVs. To overcome the limitations, we introduce a pioneering framework termed Vulnerability-aware and Curiosity-driven Adversarial Training (VCAT). Specifically, during the traffic vehicle attacker training phase, a surrogate network is employed to fit the value function of the AV victim, providing dense information about the victim's inherent vulnerabilities. Subsequently, random network distillation is used to characterize the novelty of the environment, constructing an intrinsic reward to guide the attacker in exploring unexplored territories. In the victim defense training phase, the AV is trained in critical scenarios in which the pretrained attacker is positioned around the victim to generate attack behaviors. Experimental results revealed that the training methodology provided by VCAT significantly improved the robust control capabilities of learning-based AVs, outperforming both conventional training modalities and alternative reinforcement learning counterparts, with a marked reduction in crash rates. The code is available at https://github.com/caixxuan/VCAT.

摘要: 自动驾驶汽车(AVs)在复杂的交通环境中的安全运行面临着巨大的威胁。对抗性训练已经成为一种有效的方法，使AVs能够先发制人地增强其对恶意攻击的健壮性。使用对抗性策略训练攻击者，允许反病毒通过与该攻击者的交互学习健壮的驾驶。然而，现有方法中的对抗性策略经常陷入过度利用已建立的漏洞的循环中，导致对AV的改进很差。为了克服这些限制，我们引入了一个开创性的框架，称为漏洞感知和好奇心驱动的对手训练(VCAT)。具体地说，在交通车辆攻击者训练阶段，使用代理网络来拟合反病毒受害者的价值函数，提供关于受害者固有漏洞的密集信息。随后，随机网络蒸馏被用来表征环境的新颖性，构建一种内在的奖励来引导攻击者探索未知领域。在受害者防御训练阶段，反病毒在关键场景中进行训练，在该场景中，预先训练的攻击者被定位在受害者周围以产生攻击行为。实验结果表明，VCAT提供的训练方法显著提高了基于学习的自动驾驶系统的鲁棒控制能力，表现优于传统训练模式和替代强化学习模式，并显著降低了撞车率。代码可在https://github.com/caixxuan/VCAT.上获得



## **12. Boosting Certified Robustness for Time Series Classification with Efficient Self-Ensemble**

通过高效的自我整合增强时间序列分类的认证鲁棒性 cs.LG

6 figures, 4 tables, 10 pages

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.02802v3) [paper-pdf](http://arxiv.org/pdf/2409.02802v3)

**Authors**: Chang Dong, Zhengyang Li, Liangwei Zheng, Weitong Chen, Wei Emma Zhang

**Abstract**: Recently, the issue of adversarial robustness in the time series domain has garnered significant attention. However, the available defense mechanisms remain limited, with adversarial training being the predominant approach, though it does not provide theoretical guarantees. Randomized Smoothing has emerged as a standout method due to its ability to certify a provable lower bound on robustness radius under $\ell_p$-ball attacks. Recognizing its success, research in the time series domain has started focusing on these aspects. However, existing research predominantly focuses on time series forecasting, or under the non-$\ell_p$ robustness in statistic feature augmentation for time series classification~(TSC). Our review found that Randomized Smoothing performs modestly in TSC, struggling to provide effective assurances on datasets with poor robustness. Therefore, we propose a self-ensemble method to enhance the lower bound of the probability confidence of predicted labels by reducing the variance of classification margins, thereby certifying a larger radius. This approach also addresses the computational overhead issue of Deep Ensemble~(DE) while remaining competitive and, in some cases, outperforming it in terms of robustness. Both theoretical analysis and experimental results validate the effectiveness of our method, demonstrating superior performance in robustness testing compared to baseline approaches.

摘要: 最近，时间序列域中的对抗性稳健性问题引起了人们的广泛关注。然而，现有的防御机制仍然有限，对抗性训练是主要的方法，尽管它不提供理论上的保证。由于随机化平滑方法能够证明在$ell_p$-ball攻击下的健壮性半径的一个可证明的下界，所以它已经成为一种优秀的方法。认识到它的成功，时间序列领域的研究已经开始集中在这些方面。然而，现有的研究主要集中在时间序列预测，或在统计特征增强对时间序列分类具有非埃尔p稳健性的情况下。我们的综述发现，随机平滑在TSC中表现平平，难以对稳健性较差的数据集提供有效的保证。因此，我们提出了一种自集成方法，通过减小分类裕度的方差来提高预测标签的概率置信度下界，从而证明更大的半径。这种方法还解决了深层集成~(DE)的计算开销问题，同时保持了竞争力，在某些情况下，在健壮性方面优于它。理论分析和实验结果都验证了该方法的有效性，在稳健性测试中表现出了优于基线方法的性能。



## **13. Deep generative models as an adversarial attack strategy for tabular machine learning**

深度生成模型作为表格机器学习的对抗攻击策略 cs.LG

Accepted at ICMLC 2024 (International Conference on Machine Learning  and Cybernetics)

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12642v1) [paper-pdf](http://arxiv.org/pdf/2409.12642v1)

**Authors**: Salijona Dyrmishi, Mihaela Cătălina Stoian, Eleonora Giunchiglia, Maxime Cordy

**Abstract**: Deep Generative Models (DGMs) have found application in computer vision for generating adversarial examples to test the robustness of machine learning (ML) systems. Extending these adversarial techniques to tabular ML presents unique challenges due to the distinct nature of tabular data and the necessity to preserve domain constraints in adversarial examples. In this paper, we adapt four popular tabular DGMs into adversarial DGMs (AdvDGMs) and evaluate their effectiveness in generating realistic adversarial examples that conform to domain constraints.

摘要: 深度生成模型（DGM）已在计算机视觉中应用，用于生成对抗性示例来测试机器学习（ML）系统的稳健性。由于表格数据的独特性质以及在对抗性示例中保留域约束的必要性，将这些对抗性技术扩展到表格ML带来了独特的挑战。在本文中，我们将四种流行的表格式DGM调整为对抗性DGM（AdvDGM），并评估它们在生成符合领域约束的现实对抗性示例方面的有效性。



## **14. Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective**

探索共享扩散模型中的隐私和公平风险：对抗的视角 cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2402.18607v3) [paper-pdf](http://arxiv.org/pdf/2402.18607v3)

**Authors**: Xinjian Luo, Yangfan Jiang, Fei Wei, Yuncheng Wu, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Diffusion models have recently gained significant attention in both academia and industry due to their impressive generative performance in terms of both sampling quality and distribution coverage. Accordingly, proposals are made for sharing pre-trained diffusion models across different organizations, as a way of improving data utilization while enhancing privacy protection by avoiding sharing private data directly. However, the potential risks associated with such an approach have not been comprehensively examined.   In this paper, we take an adversarial perspective to investigate the potential privacy and fairness risks associated with the sharing of diffusion models. Specifically, we investigate the circumstances in which one party (the sharer) trains a diffusion model using private data and provides another party (the receiver) black-box access to the pre-trained model for downstream tasks. We demonstrate that the sharer can execute fairness poisoning attacks to undermine the receiver's downstream models by manipulating the training data distribution of the diffusion model. Meanwhile, the receiver can perform property inference attacks to reveal the distribution of sensitive features in the sharer's dataset. Our experiments conducted on real-world datasets demonstrate remarkable attack performance on different types of diffusion models, which highlights the critical importance of robust data auditing and privacy protection protocols in pertinent applications.

摘要: 扩散模型由于其在抽样质量和分布复盖率方面令人印象深刻的生成性能，最近在学术界和工业界都得到了极大的关注。因此，提出了在不同组织之间共享预先培训的传播模式的建议，以此作为提高数据利用率的一种方式，同时通过避免直接共享私人数据来加强隐私保护。然而，与这种方法相关的潜在风险尚未得到全面审查。在这篇文章中，我们采取对抗性的视角来调查与共享扩散模型相关的潜在的隐私和公平风险。具体地说，我们调查了一方(共享者)使用私有数据训练扩散模型，并为另一方(接收者)提供对下游任务的预训练模型的黑箱访问的情况。我们证明了共享者可以通过操纵扩散模型的训练数据分布来执行公平毒化攻击来破坏接收者的下游模型。同时，接收者可以执行属性推断攻击，以揭示共享者数据集中敏感特征的分布。我们在真实数据集上进行的实验表明，在不同类型的扩散模型上具有显著的攻击性能，这突显了健壮的数据审计和隐私保护协议在相关应用中的关键重要性。



## **15. Adversarial Attack for Explanation Robustness of Rationalization Models**

对合理化模型解释稳健性的对抗攻击 cs.CL

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2408.10795v3) [paper-pdf](http://arxiv.org/pdf/2408.10795v3)

**Authors**: Yuankai Zhang, Lingxiao Kong, Haozhao Wang, Ruixuan Li, Jun Wang, Yuhua Li, Wei Liu

**Abstract**: Rationalization models, which select a subset of input text as rationale-crucial for humans to understand and trust predictions-have recently emerged as a prominent research area in eXplainable Artificial Intelligence. However, most of previous studies mainly focus on improving the quality of the rationale, ignoring its robustness to malicious attack. Specifically, whether the rationalization models can still generate high-quality rationale under the adversarial attack remains unknown. To explore this, this paper proposes UAT2E, which aims to undermine the explainability of rationalization models without altering their predictions, thereby eliciting distrust in these models from human users. UAT2E employs the gradient-based search on triggers and then inserts them into the original input to conduct both the non-target and target attack. Experimental results on five datasets reveal the vulnerability of rationalization models in terms of explanation, where they tend to select more meaningless tokens under attacks. Based on this, we make a series of recommendations for improving rationalization models in terms of explanation.

摘要: 合理化模型选择输入文本的一个子集作为理论基础--这对人类理解和信任预测至关重要--最近已成为可解释人工智能的一个重要研究领域。然而，以往的研究大多侧重于提高理论基础的质量，而忽略了其对恶意攻击的健壮性。具体地说，在对抗性攻击下，合理化模型是否仍能产生高质量的推理仍是未知的。为了探索这一点，本文提出了UAT2E，其目的是在不改变其预测的情况下削弱合理化模型的可解释性，从而引起人类用户对这些模型的不信任。UAT2E在触发器上采用基于梯度的搜索，然后将它们插入到原始输入中，以进行非目标攻击和目标攻击。在五个数据集上的实验结果揭示了合理化模型在解释方面的脆弱性，在攻击下，它们倾向于选择更多无意义的标记。在此基础上，本文从解释的角度提出了一系列改进合理化模型的建议。



## **16. Designing an attack-defense game: how to increase robustness of financial transaction models via a competition**

设计攻守游戏：如何通过竞争提高金融交易模型的稳健性 cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2308.11406v3) [paper-pdf](http://arxiv.org/pdf/2308.11406v3)

**Authors**: Alexey Zaytsev, Maria Kovaleva, Alex Natekin, Evgeni Vorsin, Valerii Smirnov, Georgii Smirnov, Oleg Sidorshin, Alexander Senin, Alexander Dudin, Dmitry Berestnev

**Abstract**: Banks routinely use neural networks to make decisions. While these models offer higher accuracy, they are susceptible to adversarial attacks, a risk often overlooked in the context of event sequences, particularly sequences of financial transactions, as most works consider computer vision and NLP modalities.   We propose a thorough approach to studying these risks: a novel type of competition that allows a realistic and detailed investigation of problems in financial transaction data. The participants directly oppose each other, proposing attacks and defenses -- so they are examined in close-to-real-life conditions.   The paper outlines our unique competition structure with direct opposition of participants, presents results for several different top submissions, and analyzes the competition results. We also introduce a new open dataset featuring financial transactions with credit default labels, enhancing the scope for practical research and development.

摘要: 银行经常使用神经网络来做出决策。虽然这些模型提供了更高的准确性，但它们很容易受到对抗攻击，这一风险在事件序列（尤其是金融交易序列）的背景下经常被忽视，因为大多数作品都考虑计算机视觉和NLP模式。   我们提出了一种彻底的方法来研究这些风险：一种新型竞争，可以对金融交易数据中的问题进行现实而详细的调查。参与者直接相互反对，提出攻击和防御--因此他们在接近现实生活的条件下受到审查。   该论文概述了我们独特的竞争结构，参与者直接反对，列出了几种不同的顶级提交的结果，并分析了竞争结果。我们还引入了一个新的开放数据集，以带有信用违约标签的金融交易为特色，扩大了实践研究和开发的范围。



## **17. Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems**

Magmaw：对基于机器学习的无线通信系统的模式不可知的对抗攻击 cs.CR

Accepted at NDSS 2025

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2311.00207v2) [paper-pdf](http://arxiv.org/pdf/2311.00207v2)

**Authors**: Jung-Woo Chang, Ke Sun, Nasimeh Heydaribeni, Seira Hidano, Xinyu Zhang, Farinaz Koushanfar

**Abstract**: Machine Learning (ML) has been instrumental in enabling joint transceiver optimization by merging all physical layer blocks of the end-to-end wireless communication systems. Although there have been a number of adversarial attacks on ML-based wireless systems, the existing methods do not provide a comprehensive view including multi-modality of the source data, common physical layer protocols, and wireless domain constraints. This paper proposes Magmaw, a novel wireless attack methodology capable of generating universal adversarial perturbations for any multimodal signal transmitted over a wireless channel. We further introduce new objectives for adversarial attacks on downstream applications. We adopt the widely-used defenses to verify the resilience of Magmaw. For proof-of-concept evaluation, we build a real-time wireless attack platform using a software-defined radio system. Experimental results demonstrate that Magmaw causes significant performance degradation even in the presence of strong defense mechanisms. Furthermore, we validate the performance of Magmaw in two case studies: encrypted communication channel and channel modality-based ML model.

摘要: 机器学习(ML)通过合并端到端无线通信系统的所有物理层块，在实现联合收发器优化方面发挥了重要作用。尽管已经有一些针对基于ML的无线系统的对抗性攻击，但现有的方法不能提供包括源数据的多模态、公共物理层协议和无线域限制在内的全面视角。本文提出了一种新的无线攻击方法Magmaw，它能够对无线信道上传输的任何多模信号产生通用的对抗性扰动。我们进一步引入了针对下游应用程序的对抗性攻击的新目标。我们采用了广泛使用的防御措施来验证Magmaw的弹性。对于概念验证评估，我们使用软件定义的无线电系统构建了一个实时无线攻击平台。实验结果表明，即使在强防御机制存在的情况下，Magmaw也会导致性能显著下降。此外，我们在加密通信通道和基于通道通道的ML模型两个案例中验证了MAGMAW的性能。



## **18. TEAM: Temporal Adversarial Examples Attack Model against Network Intrusion Detection System Applied to RNN**

TEAM：应用于RNN的针对网络入侵检测系统的时间对抗示例攻击模型 cs.CR

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12472v1) [paper-pdf](http://arxiv.org/pdf/2409.12472v1)

**Authors**: Ziyi Liu, Dengpan Ye, Long Tang, Yunming Zhang, Jiacheng Deng

**Abstract**: With the development of artificial intelligence, neural networks play a key role in network intrusion detection systems (NIDS). Despite the tremendous advantages, neural networks are susceptible to adversarial attacks. To improve the reliability of NIDS, many research has been conducted and plenty of solutions have been proposed. However, the existing solutions rarely consider the adversarial attacks against recurrent neural networks (RNN) with time steps, which would greatly affect the application of NIDS in real world. Therefore, we first propose a novel RNN adversarial attack model based on feature reconstruction called \textbf{T}emporal adversarial \textbf{E}xamples \textbf{A}ttack \textbf{M}odel \textbf{(TEAM)}, which applied to time series data and reveals the potential connection between adversarial and time steps in RNN. That is, the past adversarial examples within the same time steps can trigger further attacks on current or future original examples. Moreover, TEAM leverages Time Dilation (TD) to effectively mitigates the effect of temporal among adversarial examples within the same time steps. Experimental results show that in most attack categories, TEAM improves the misjudgment rate of NIDS on both black and white boxes, making the misjudgment rate reach more than 96.68%. Meanwhile, the maximum increase in the misjudgment rate of the NIDS for subsequent original samples exceeds 95.57%.

摘要: 随着人工智能的发展，神经网络在网络入侵检测系统中发挥着关键作用。尽管神经网络具有巨大的优势，但它很容易受到对手的攻击。为了提高网络入侵检测系统的可靠性，人们进行了大量的研究，并提出了大量的解决方案。然而，现有的解决方案很少考虑对带时间步长的递归神经网络的对抗性攻击，这将极大地影响网络入侵检测系统在现实世界中的应用。为此，我们首先提出了一种新的基于特征重构的RNN对抗性攻击模型也就是说，在相同的时间步骤内的过去的对抗性例子可以触发对当前或未来的原始例子的进一步攻击。此外，团队利用时间膨胀(TD)来有效地缓解相同时间步长内的对抗性例子之间的时间效应。实验结果表明，在大多数攻击类别中，Team都提高了黑盒和白盒的误判率，使误判率达到96.68%以上。同时，网络入侵检测系统对后续原始样本的误判率最大增幅超过95.57%。



## **19. Object-fabrication Targeted Attack for Object Detection**

物体制造用于物体检测的定向攻击 cs.CV

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2212.06431v3) [paper-pdf](http://arxiv.org/pdf/2212.06431v3)

**Authors**: Xuchong Zhang, Changfeng Sun, Haoliang Han, Hongbin Sun

**Abstract**: Recent studies have demonstrated that object detection networks are usually vulnerable to adversarial examples. Generally, adversarial attacks for object detection can be categorized into targeted and untargeted attacks. Compared with untargeted attacks, targeted attacks present greater challenges and all existing targeted attack methods launch the attack by misleading detectors to mislabel the detected object as a specific wrong label. However, since these methods must depend on the presence of the detected objects within the victim image, they suffer from limitations in attack scenarios and attack success rates. In this paper, we propose a targeted feature space attack method that can mislead detectors to `fabricate' extra designated objects regardless of whether the victim image contains objects or not. Specifically, we introduce a guided image to extract coarse-grained features of the target objects and design an innovative dual attention mechanism to filter out the critical features of the target objects efficiently. The attack performance of the proposed method is evaluated on MS COCO and BDD100K datasets with FasterRCNN and YOLOv5. Evaluation results indicate that the proposed targeted feature space attack method shows significant improvements in terms of image-specific, universality, and generalization attack performance, compared with the previous targeted attack for object detection.

摘要: 最近的研究表明，目标检测网络通常容易受到敌意例子的攻击。通常，用于目标检测的对抗性攻击可以分为目标攻击和非目标攻击。与非定向攻击相比，定向攻击提出了更大的挑战，现有的所有定向攻击方法都是通过误导检测器将检测到的对象错误地标记为特定的错误标签来发起攻击。然而，由于这些方法必须依赖于受害者图像中检测到的对象的存在，它们在攻击场景和攻击成功率方面受到限制。在本文中，我们提出了一种目标特征空间攻击方法，该方法可以误导检测器‘捏造’额外的指定对象，而不管受害者图像中是否包含对象。具体地说，我们引入引导图像来提取目标对象的粗粒度特征，并设计了一种创新的双重注意机制来高效地过滤出目标对象的关键特征。使用FasterRCNN和YOLOv5对该方法在MS COCO和BDD100K数据集上的攻击性能进行了评估。评估结果表明，与已有的目标检测的目标攻击方法相比，本文提出的目标特征空间攻击方法在图像专用性、通用性和泛化性能方面都有明显的提高。



## **20. Towards Physically-Realizable Adversarial Attacks in Embodied Vision Navigation**

视觉导航中实现物理可实现的对抗攻击 cs.CV

8 pages, 6 figures, submitted to the 2025 IEEE International  Conference on Robotics & Automation (ICRA)

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.10071v2) [paper-pdf](http://arxiv.org/pdf/2409.10071v2)

**Authors**: Meng Chen, Jiawei Tu, Chao Qi, Yonghao Dang, Feng Zhou, Wei Wei, Jianqin Yin

**Abstract**: The deployment of embodied navigation agents in safety-critical environments raises concerns about their vulnerability to adversarial attacks on deep neural networks. However, current attack methods often lack practicality due to challenges in transitioning from the digital to the physical world, while existing physical attacks for object detection fail to achieve both multi-view effectiveness and naturalness. To address this, we propose a practical attack method for embodied navigation by attaching adversarial patches with learnable textures and opacity to objects. Specifically, to ensure effectiveness across varying viewpoints, we employ a multi-view optimization strategy based on object-aware sampling, which uses feedback from the navigation model to optimize the patch's texture. To make the patch inconspicuous to human observers, we introduce a two-stage opacity optimization mechanism, where opacity is refined after texture optimization. Experimental results show our adversarial patches reduce navigation success rates by about 40%, outperforming previous methods in practicality, effectiveness, and naturalness. Code is available at: [https://github.com/chen37058/Physical-Attacks-in-Embodied-Navigation].

摘要: 在安全关键环境中部署具体化导航代理引起了人们对它们在深层神经网络上易受敌意攻击的担忧。然而，由于从数字世界向物理世界过渡的挑战，现有的攻击方法往往缺乏实用性，而现有的针对目标检测的物理攻击无法达到多视角的有效性和自然性。为了解决这一问题，我们提出了一种实用的具身导航攻击方法，通过将具有可学习纹理和不透明度的敌意补丁附加到对象上。具体地说，为了确保不同视点的有效性，我们采用了一种基于对象感知采样的多视点优化策略，该策略利用导航模型的反馈来优化面片的纹理。为了使面片不易被人察觉，我们引入了一种两阶段不透明度优化机制，在纹理优化后对不透明度进行细化。实验结果表明，我们的对抗性补丁使导航成功率降低了约40%，在实用性、有效性和自然性方面都优于以往的方法。代码可从以下网址获得：[https://github.com/chen37058/Physical-Attacks-in-Embodied-Navigation].



## **21. Typography Leads Semantic Diversifying: Amplifying Adversarial Transferability across Multimodal Large Language Models**

字体设计引领语义多元化：增强多模式大型语言模型之间的对抗性可移植性 cs.CV

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2405.20090v2) [paper-pdf](http://arxiv.org/pdf/2405.20090v2)

**Authors**: Hao Cheng, Erjia Xiao, Jiayan Yang, Jiahang Cao, Le Yang, Jize Zhang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Following the advent of the Artificial Intelligence (AI) era of large models, Multimodal Large Language Models (MLLMs) with the ability to understand cross-modal interactions between vision and text have attracted wide attention. Adversarial examples with human-imperceptible perturbation are shown to possess a characteristic known as transferability, which means that a perturbation generated by one model could also mislead another different model. Augmenting the diversity in input data is one of the most significant methods for enhancing adversarial transferability. This method has been certified as a way to significantly enlarge the threat impact under black-box conditions. Research works also demonstrate that MLLMs can be exploited to generate adversarial examples in the white-box scenario. However, the adversarial transferability of such perturbations is quite limited, failing to achieve effective black-box attacks across different models. In this paper, we propose the Typographic-based Semantic Transfer Attack (TSTA), which is inspired by: (1) MLLMs tend to process semantic-level information; (2) Typographic Attack could effectively distract the visual information captured by MLLMs. In the scenarios of Harmful Word Insertion and Important Information Protection, our TSTA demonstrates superior performance.

摘要: 随着大模型人工智能时代的到来，能够理解视觉和文本之间跨通道交互的多通道大语言模型引起了人们的广泛关注。具有人类不可察觉的扰动的对抗性例子具有被称为可转移性的特征，这意味着一个模型产生的扰动也可能误导另一个不同的模型。增加输入数据的多样性是增强对抗性转移的最重要的方法之一。这种方法已被证明是一种在黑箱条件下显著扩大威胁影响的方法。研究工作还表明，在白盒情况下，MLLMS可以被用来生成对抗性示例。然而，此类扰动的对抗性可转移性相当有限，无法实现跨不同模型的有效黑盒攻击。本文提出了基于排版的语义传输攻击(TSTA)，其灵感来自：(1)MLLMS倾向于处理语义级的信息；(2)排版攻击可以有效地分散MLLMS捕获的视觉信息。在有害词语插入和重要信息保护的场景中，我们的TSTA表现出了卓越的性能。



## **22. ITPatch: An Invisible and Triggered Physical Adversarial Patch against Traffic Sign Recognition**

ITpatch：针对交通标志识别的隐形且触发的物理对抗补丁 cs.CV

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12394v1) [paper-pdf](http://arxiv.org/pdf/2409.12394v1)

**Authors**: Shuai Yuan, Hongwei Li, Xingshuo Han, Guowen Xu, Wenbo Jiang, Tao Ni, Qingchuan Zhao, Yuguang Fang

**Abstract**: Physical adversarial patches have emerged as a key adversarial attack to cause misclassification of traffic sign recognition (TSR) systems in the real world. However, existing adversarial patches have poor stealthiness and attack all vehicles indiscriminately once deployed. In this paper, we introduce an invisible and triggered physical adversarial patch (ITPatch) with a novel attack vector, i.e., fluorescent ink, to advance the state-of-the-art. It applies carefully designed fluorescent perturbations to a target sign, an attacker can later trigger a fluorescent effect using invisible ultraviolet light, causing the TSR system to misclassify the sign and potentially resulting in traffic accidents. We conducted a comprehensive evaluation to investigate the effectiveness of ITPatch, which shows a success rate of 98.31% in low-light conditions. Furthermore, our attack successfully bypasses five popular defenses and achieves a success rate of 96.72%.

摘要: 物理对抗补丁已成为导致现实世界中交通标志识别（TSB）系统错误分类的关键对抗攻击。然而，现有的对抗补丁的隐蔽性较差，一旦部署，就会不加区别地攻击所有车辆。在本文中，我们引入了一种具有新型攻击载体的隐形触发物理对抗补丁（ITpatch），即，荧光墨水，推进最新技术水平。它将精心设计的荧光扰动应用于目标标志，攻击者随后可以使用不可见的紫外光触发荧光效应，导致TSB系统错误分类标志，并可能导致交通事故。我们进行了全面的评估来调查ITpatch的有效性，结果显示在弱光条件下的成功率为98.31%。此外，我们的攻击成功绕过了五种流行防御，成功率达到96.72%。



## **23. Enhancing 3D Robotic Vision Robustness by Minimizing Adversarial Mutual Information through a Curriculum Training Approach**

通过课程培训方法最大限度地减少对抗互信息，增强3D机器人视觉的稳健性 cs.CV

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12379v1) [paper-pdf](http://arxiv.org/pdf/2409.12379v1)

**Authors**: Nastaran Darabi, Dinithi Jayasuriya, Devashri Naik, Theja Tulabandhula, Amit Ranjan Trivedi

**Abstract**: Adversarial attacks exploit vulnerabilities in a model's decision boundaries through small, carefully crafted perturbations that lead to significant mispredictions. In 3D vision, the high dimensionality and sparsity of data greatly expand the attack surface, making 3D vision particularly vulnerable for safety-critical robotics. To enhance 3D vision's adversarial robustness, we propose a training objective that simultaneously minimizes prediction loss and mutual information (MI) under adversarial perturbations to contain the upper bound of misprediction errors. This approach simplifies handling adversarial examples compared to conventional methods, which require explicit searching and training on adversarial samples. However, minimizing prediction loss conflicts with minimizing MI, leading to reduced robustness and catastrophic forgetting. To address this, we integrate curriculum advisors in the training setup that gradually introduce adversarial objectives to balance training and prevent models from being overwhelmed by difficult cases early in the process. The advisors also enhance robustness by encouraging training on diverse MI examples through entropy regularizers. We evaluated our method on ModelNet40 and KITTI using PointNet, DGCNN, SECOND, and PointTransformers, achieving 2-5% accuracy gains on ModelNet40 and a 5-10% mAP improvement in object detection. Our code is publicly available at https://github.com/nstrndrbi/Mine-N-Learn.

摘要: 对抗性攻击利用模型决策边界中的漏洞，通过精心设计的小扰动导致严重的错误预测。在3D视觉中，数据的高维性和稀疏性极大地扩大了攻击面，使得3D视觉特别容易受到安全关键型机器人的攻击。为了增强3D视觉的对抗鲁棒性，我们提出了一种训练目标，该目标同时最小化对抗扰动下的预测损失和互信息(MI)，以遏制误预测误差的上界。与传统方法相比，这种方法简化了对对抗性样本的处理，因为传统方法需要对对抗性样本进行明确的搜索和训练。然而，最小化预测损失与最小化MI相冲突，导致健壮性降低和灾难性遗忘。为了解决这个问题，我们在培训设置中整合了课程顾问，逐步引入对抗性目标，以平衡培训，并防止模型在过程早期被困难的案例淹没。顾问还通过鼓励通过熵正则化对不同的MI示例进行培训来增强稳健性。我们在ModelNet40和Kitti上使用PointNet、DGCNN、Second和PointTransformers对我们的方法进行了评估，在ModelNet40上获得了2%-5%的准确率改进，在目标检测方面获得了5%-10%的MAP改进。我们的代码在https://github.com/nstrndrbi/Mine-N-Learn.上公开提供



## **24. AirGapAgent: Protecting Privacy-Conscious Conversational Agents**

AirGapAgent：保护有隐私意识的对话代理人 cs.CR

at CCS'24

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2405.05175v2) [paper-pdf](http://arxiv.org/pdf/2405.05175v2)

**Authors**: Eugene Bagdasarian, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, Daniel Ramage

**Abstract**: The growing use of large language model (LLM)-based conversational agents to manage sensitive user data raises significant privacy concerns. While these agents excel at understanding and acting on context, this capability can be exploited by malicious actors. We introduce a novel threat model where adversarial third-party apps manipulate the context of interaction to trick LLM-based agents into revealing private information not relevant to the task at hand.   Grounded in the framework of contextual integrity, we introduce AirGapAgent, a privacy-conscious agent designed to prevent unintended data leakage by restricting the agent's access to only the data necessary for a specific task. Extensive experiments using Gemini, GPT, and Mistral models as agents validate our approach's effectiveness in mitigating this form of context hijacking while maintaining core agent functionality. For example, we show that a single-query context hijacking attack on a Gemini Ultra agent reduces its ability to protect user data from 94% to 45%, while an AirGapAgent achieves 97% protection, rendering the same attack ineffective.

摘要: 越来越多地使用基于大型语言模型(LLM)的会话代理来管理敏感用户数据，这引发了严重的隐私问题。虽然这些代理擅长理解上下文并根据上下文执行操作，但这种能力可能会被恶意行为者利用。我们引入了一种新的威胁模型，在该模型中，敌意的第三方应用程序操纵交互的上下文，以欺骗基于LLM的代理泄露与手头任务无关的私人信息。基于上下文完整性的框架，我们引入了AirGapAgent，这是一个具有隐私意识的代理，旨在通过限制代理仅访问特定任务所需的数据来防止意外的数据泄露。使用Gemini、GPT和Mistral模型作为代理的大量实验验证了我们的方法在保持核心代理功能的同时缓解这种形式的上下文劫持的有效性。例如，我们表明，对Gemini Ultra代理的单查询上下文劫持攻击将其保护用户数据的能力从94%降低到45%，而AirGapAgent实现了97%的保护，使得相同的攻击无效。



## **25. LookAhead: Preventing DeFi Attacks via Unveiling Adversarial Contracts**

展望未来：通过揭露对抗性合同来防止DeFi攻击 cs.CR

21 pages, 7 figures

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2401.07261v4) [paper-pdf](http://arxiv.org/pdf/2401.07261v4)

**Authors**: Shoupeng Ren, Lipeng He, Tianyu Tu, Di Wu, Jian Liu, Kui Ren, Chun Chen

**Abstract**: Decentralized Finance (DeFi) incidents stemming from the exploitation of smart contract vulnerabilities have culminated in financial damages exceeding 3 billion US dollars. Existing defense mechanisms typically focus on detecting and reacting to malicious transactions executed by attackers that target victim contracts. However, with the emergence of private transaction pools where transactions are sent directly to miners without first appearing in public mempools, current detection tools face significant challenges in identifying attack activities effectively.   Based on the fact that most attack logic rely on deploying one or more intermediate smart contracts as supporting components to the exploitation of victim contracts, in this paper, we propose a new direction for detecting DeFi attacks that focuses on identifying adversarial contracts instead of adversarial transactions. Our approach allows us to leverage common attack patterns, code semantics and intrinsic characteristics found in malicious smart contracts to build the LookAhead system based on Machine Learning (ML) classifiers and a transformer model that is able to effectively distinguish adversarial contracts from benign ones, and make just-in-time predictions of potential zero-day attacks. Our contributions are three-fold: First, we construct a comprehensive dataset consisting of features extracted and constructed from recent contracts deployed on the Ethereum and BSC blockchains. Secondly, we design a condensed representation of smart contract programs called Pruned Semantic-Control Flow Tokenization (PSCFT) and use it to train a combination of ML models that understand the behaviour of malicious codes based on function calls, control flows and other pattern-conforming features. Lastly, we provide the complete implementation of LookAhead and the evaluation of its performance metrics for detecting adversarial contracts.

摘要: 因利用智能合同漏洞而引发的去中心化金融(Defi)事件已造成超过30亿美元的经济损失。现有的防御机制通常专注于检测和响应攻击者执行的针对受害者合同的恶意交易。然而，随着私人交易池的出现，交易直接发送给矿工，而不是首先出现在公共记忆池中，当前的检测工具在有效识别攻击活动方面面临重大挑战。基于大多数攻击逻辑依赖于部署一个或多个中间智能合约作为攻击受害者合约的支持组件的事实，本文提出了一种新的检测Defi攻击的方向，该方向侧重于识别对手合约而不是对手交易。我们的方法允许我们利用恶意智能合同中发现的常见攻击模式、代码语义和内在特征来构建基于机器学习(ML)分类器和转换器模型的前瞻性系统，该系统能够有效区分敌意合同和良性合同，并及时预测潜在的零日攻击。我们的贡献有三个方面：首先，我们构建了一个全面的数据集，其中包含从Etherum和BSC区块链上部署的最近合同中提取和构建的特征。其次，我们设计了智能合同程序的精简表示，称为剪枝语义控制流令牌化(PSCFT)，并使用它来训练ML模型的组合，这些模型基于函数调用、控制流和其他符合模式的特征来理解恶意代码的行为。最后，我们给出了LookHead的完整实现，并对其用于检测敌对合同的性能度量进行了评估。



## **26. Bi-objective trail-planning for a robot team orienteering in a hazardous environment**

危险环境中机器人队定向运动的双目标路径规划 cs.RO

v0.0

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.12114v1) [paper-pdf](http://arxiv.org/pdf/2409.12114v1)

**Authors**: Cory M. Simon, Jeffrey Richley, Lucas Overbey, Darleen Perez-Lavin

**Abstract**: Teams of mobile [aerial, ground, or aquatic] robots have applications in resource delivery, patrolling, information-gathering, agriculture, forest fire fighting, chemical plume source localization and mapping, and search-and-rescue. Robot teams traversing hazardous environments -- with e.g. rough terrain or seas, strong winds, or adversaries capable of attacking or capturing robots -- should plan and coordinate their trails in consideration of risks of disablement, destruction, or capture. Specifically, the robots should take the safest trails, coordinate their trails to cooperatively achieve the team-level objective with robustness to robot failures, and balance the reward from visiting locations against risks of robot losses. Herein, we consider bi-objective trail-planning for a mobile team of robots orienteering in a hazardous environment. The hazardous environment is abstracted as a directed graph whose arcs, when traversed by a robot, present known probabilities of survival. Each node of the graph offers a reward to the team if visited by a robot (which e.g. delivers a good to or images the node). We wish to search for the Pareto-optimal robot-team trail plans that maximize two [conflicting] team objectives: the expected (i) team reward and (ii) number of robots that survive the mission. A human decision-maker can then select trail plans that balance, according to their values, reward and robot survival. We implement ant colony optimization, guided by heuristics, to search for the Pareto-optimal set of robot team trail plans. As a case study, we illustrate with an information-gathering mission in an art museum.

摘要: 移动[空中、地面或水上]机器人团队在资源输送、巡逻、信息收集、农业、森林灭火、化学烟雾源定位和测绘以及搜救中有应用。机器人团队穿越危险环境--例如崎岖的地形或海洋、强风或有能力攻击或捕获机器人的敌人--应考虑到致残、破坏或捕获的风险来规划和协调他们的路径。具体地说，机器人应该选择最安全的路径，协调它们的路径，以协作地实现团队级目标，并对机器人故障具有鲁棒性，并在访问地点的回报和机器人损失的风险之间进行平衡。在这里，我们考虑在危险环境中定向的移动机器人团队的双目标路径规划。危险环境被抽象为一个有向图，当机器人穿过该图的弧线时，表示已知的生存概率。如果被机器人访问，图中的每个节点向团队提供奖励(例如，机器人向节点递送商品或为节点成像)。我们希望寻找帕累托最优的机器人团队路径计划，以最大化两个[相互冲突的]团队目标：预期的(I)团队奖励和(Ii)在任务中幸存下来的机器人数量。然后，人类决策者可以根据他们的价值观、奖励和机器人生存来选择平衡的试验计划。在启发式算法的指导下，采用蚁群算法来搜索机器人团队路径规划的帕累托最优集合。作为一个案例研究，我们以一家美术馆的信息收集任务为例进行说明。



## **27. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

从ML到LLM：评估网络钓鱼网页检测模型对抗对抗攻击的稳健性 cs.CR

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2407.20361v2) [paper-pdf](http://arxiv.org/pdf/2407.20361v2)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of two existing models, Stack model and Phishpedia, in classifying PhishOracle-generated adversarial phishing webpages. Additionally, we study a commercial large language model, Gemini Pro Vision, in the context of adversarial attacks. We conduct a user study to determine whether PhishOracle-generated adversarial phishing webpages deceive users. Our findings reveal that many PhishOracle-generated phishing webpages evade current phishing webpage detection models and deceive users, but Gemini Pro Vision is robust to the attack. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources are publicly available on GitHub.

摘要: 网络钓鱼攻击试图欺骗用户窃取敏感信息，构成重大的网络安全威胁。机器学习(ML)和深度学习(DL)的进步导致了许多钓鱼网页检测解决方案的发展，但这些模型仍然容易受到对手攻击。评估它们对敌意网络钓鱼网页的健壮性是至关重要的。现有工具包含为有限数量的品牌预先设计的钓鱼网页的数据集，并且在钓鱼功能方面缺乏多样性。为了应对这些挑战，我们开发了PhishOracle，这是一个通过在合法网页中嵌入不同的钓鱼功能来生成敌意钓鱼网页的工具。我们评估了现有的两种模型Stack模型和Phishpedia模型对PhishOracle生成的敌意钓鱼网页进行分类的稳健性。此外，我们研究了一个商业大型语言模型，Gemini Pro Vision，在对抗性攻击的背景下。我们进行了一项用户研究，以确定PhishOracle生成的敌意钓鱼网页是否欺骗了用户。我们的研究结果显示，许多PhishOracle生成的钓鱼网页逃避了当前的钓鱼网页检测模型并欺骗用户，但Gemini Pro Vision对攻击具有健壮性。我们还开发了PhishOracle Web应用程序，允许用户输入合法的URL，选择相关的网络钓鱼功能并生成相应的网络钓鱼网页。所有资源都在GitHub上公开提供。



## **28. Adversarial attacks on neural networks through canonical Riemannian foliations**

通过典型的Riemann叶联对神经网络的对抗攻击 stat.ML

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2203.00922v3) [paper-pdf](http://arxiv.org/pdf/2203.00922v3)

**Authors**: Eliot Tron, Nicolas Couellan, Stéphane Puechmorel

**Abstract**: Deep learning models are known to be vulnerable to adversarial attacks. Adversarial learning is therefore becoming a crucial task. We propose a new vision on neural network robustness using Riemannian geometry and foliation theory. The idea is illustrated by creating a new adversarial attack that takes into account the curvature of the data space. This new adversarial attack, called the two-step spectral attack is a piece-wise linear approximation of a geodesic in the data space. The data space is treated as a (degenerate) Riemannian manifold equipped with the pullback of the Fisher Information Metric (FIM) of the neural network. In most cases, this metric is only semi-definite and its kernel becomes a central object to study. A canonical foliation is derived from this kernel. The curvature of transverse leaves gives the appropriate correction to get a two-step approximation of the geodesic and hence a new efficient adversarial attack. The method is first illustrated on a 2D toy example in order to visualize the neural network foliation and the corresponding attacks. Next, we report numerical results on the MNIST and CIFAR10 datasets with the proposed technique and state of the art attacks presented in Zhao et al. (2019) (OSSA) and Croce et al. (2020) (AutoAttack). The result show that the proposed attack is more efficient at all levels of available budget for the attack (norm of the attack), confirming that the curvature of the transverse neural network FIM foliation plays an important role in the robustness of neural networks. The main objective and interest of this study is to provide a mathematical understanding of the geometrical issues at play in the data space when constructing efficient attacks on neural networks.

摘要: 众所周知，深度学习模型容易受到对手的攻击。因此，对抗性学习正成为一项至关重要的任务。利用黎曼几何和分层理论，我们对神经网络的稳健性提出了新的看法。通过创建一种新的考虑数据空间曲率的对抗性攻击来说明这一想法。这种新的敌意攻击被称为两步谱攻击，它是数据空间中测地线的分段线性近似。数据空间被视为一个(退化的)黎曼流形，带有神经网络的Fisher信息度量(FIM)的回撤。在大多数情况下，这个度量只是半定的，它的核心成为研究的中心对象。一个典型的叶理是从这个核派生出来的。横叶的曲率给出了适当的修正，得到了测地线的两步近似，从而得到了一种新的有效的对抗性攻击。为了可视化神经网络的分层和相应的攻击，首先以2D玩具为例说明了该方法。接下来，我们报告了在MNIST和CIFAR10数据集上的数值结果，以及赵等人提出的技术和最新攻击。(2019)(Ossa)和Croce等人。(2020)(AutoAttack)。结果表明，该攻击在攻击的所有可用预算级别(攻击范数)上都是有效的，证实了横向神经网络FIM分层的曲率对神经网络的健壮性起着重要作用。这项研究的主要目的和兴趣是在构造对神经网络的有效攻击时，提供对数据空间中所起作用的几何问题的数学理解。



## **29. Secure Control Systems for Autonomous Quadrotors against Cyber-Attacks**

针对网络攻击的自主四螺旋桨安全控制系统 cs.RO

The paper is based on an undergraduate thesis and is not intended for  publication in a journal

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.11897v1) [paper-pdf](http://arxiv.org/pdf/2409.11897v1)

**Authors**: Samuel Belkadi

**Abstract**: The problem of safety for robotic systems has been extensively studied. However, little attention has been given to security issues for three-dimensional systems, such as quadrotors. Malicious adversaries can compromise robot sensors and communication networks, causing incidents, achieving illegal objectives, or even injuring people. This study first designs an intelligent control system for autonomous quadrotors. Then, it investigates the problems of optimal false data injection attack scheduling and countermeasure design for unmanned aerial vehicles. Using a state-of-the-art deep learning-based approach, an optimal false data injection attack scheme is proposed to deteriorate a quadrotor's tracking performance with limited attack energy. Subsequently, an optimal tracking control strategy is learned to mitigate attacks and recover the quadrotor's tracking performance. We base our work on Agilicious, a state-of-the-art quadrotor recently deployed for autonomous settings. This paper is the first in the United Kingdom to deploy this quadrotor and implement reinforcement learning on its platform. Therefore, to promote easy reproducibility with minimal engineering overhead, we further provide (1) a comprehensive breakdown of this quadrotor, including software stacks and hardware alternatives; (2) a detailed reinforcement-learning framework to train autonomous controllers on Agilicious agents; and (3) a new open-source environment that builds upon PyFlyt for future reinforcement learning research on Agilicious platforms. Both simulated and real-world experiments are conducted to show the effectiveness of the proposed frameworks in section 5.2.

摘要: 机器人系统的安全问题已经得到了广泛的研究。然而，三维系统的安全问题却很少受到关注，比如四旋翼飞行器。恶意攻击者可以破坏机器人传感器和通信网络，引发事件，实现非法目标，甚至伤害人。本研究首先设计了自主四旋翼飞行器的智能控制系统。然后，研究了无人机最优虚假数据注入攻击调度和对抗设计问题。利用最新的基于深度学习的方法，提出了一种在攻击能量有限的情况下恶化四旋翼跟踪性能的最优虚假数据注入攻击方案。随后，学习了一种最优跟踪控制策略，以减轻攻击并恢复四旋翼的跟踪性能。我们的工作基于Agilous，这是一款最先进的四旋翼飞机，最近部署在自主环境中。本文在英国首次部署了这种四旋翼，并在其平台上实现了强化学习。因此，为了以最小的工程开销促进易重复性，我们进一步提供(1)这个四旋翼的全面细分，包括软件堆栈和硬件替代；(2)详细的强化学习框架，以培训Agilous代理上的自主控制器；以及(3)一个新的开源环境，该环境构建在PyFlyt基础上，用于未来在Agilous平台上的强化学习研究。仿真和真实世界的实验都表明了5.2节中提出的框架的有效性。



## **30. NPAT Null-Space Projected Adversarial Training Towards Zero Deterioration**

NMat零空间投影对抗训练实现零恶化 cs.LG

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.11754v1) [paper-pdf](http://arxiv.org/pdf/2409.11754v1)

**Authors**: Hanyi Hu, Qiao Han, Kui Chen, Yao Yang

**Abstract**: To mitigate the susceptibility of neural networks to adversarial attacks, adversarial training has emerged as a prevalent and effective defense strategy. Intrinsically, this countermeasure incurs a trade-off, as it sacrifices the model's accuracy in processing normal samples. To reconcile the trade-off, we pioneer the incorporation of null-space projection into adversarial training and propose two innovative Null-space Projection based Adversarial Training(NPAT) algorithms tackling sample generation and gradient optimization, named Null-space Projected Data Augmentation (NPDA) and Null-space Projected Gradient Descent (NPGD), to search for an overarching optimal solutions, which enhance robustness with almost zero deterioration in generalization performance. Adversarial samples and perturbations are constrained within the null-space of the decision boundary utilizing a closed-form null-space projector, effectively mitigating threat of attack stemming from unreliable features. Subsequently, we conducted experiments on the CIFAR10 and SVHN datasets and reveal that our methodology can seamlessly combine with adversarial training methods and obtain comparable robustness while keeping generalization close to a high-accuracy model.

摘要: 为了减轻神经网络对对抗性攻击的敏感性，对抗性训练已经成为一种普遍而有效的防御策略。本质上，这种对策需要权衡取舍，因为它牺牲了模型在处理正常样本时的准确性。为了协调两者之间的权衡，我们将零空间投影引入对抗性训练中，提出了两种新的基于零空间投影的对抗性训练(NPAT)算法，即零空间投影数据增强(NPDA)和零空间投影梯度下降(NPGD)算法，以寻求在泛化性能几乎为零恶化的情况下增强鲁棒性的零空间投影梯度下降(NPGD)算法。利用封闭的零空间投影仪将对抗性样本和扰动限制在决策边界的零空间内，有效地减轻了来自不可靠特征的攻击威胁。随后，我们在CIFAR10和SVHN数据集上进行了实验，结果表明，我们的方法可以与对抗性训练方法无缝结合，在保持泛化接近高精度模型的情况下获得相当的鲁棒性。



## **31. Hard-Label Cryptanalytic Extraction of Neural Network Models**

神经网络模型的硬标签密码分析提取 cs.CR

Accepted by Asiacrypt 2024

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.11646v1) [paper-pdf](http://arxiv.org/pdf/2409.11646v1)

**Authors**: Yi Chen, Xiaoyang Dong, Jian Guo, Yantian Shen, Anyu Wang, Xiaoyun Wang

**Abstract**: The machine learning problem of extracting neural network parameters has been proposed for nearly three decades. Functionally equivalent extraction is a crucial goal for research on this problem. When the adversary has access to the raw output of neural networks, various attacks, including those presented at CRYPTO 2020 and EUROCRYPT 2024, have successfully achieved this goal. However, this goal is not achieved when neural networks operate under a hard-label setting where the raw output is inaccessible.   In this paper, we propose the first attack that theoretically achieves functionally equivalent extraction under the hard-label setting, which applies to ReLU neural networks. The effectiveness of our attack is validated through practical experiments on a wide range of ReLU neural networks, including neural networks trained on two real benchmarking datasets (MNIST, CIFAR10) widely used in computer vision. For a neural network consisting of $10^5$ parameters, our attack only requires several hours on a single core.

摘要: 提取神经网络参数的机器学习问题已经提出了近三十年。功能等价抽取是这一问题研究的一个重要目标。当对手可以访问神经网络的原始输出时，各种攻击，包括在加密2020和欧洲加密2024上提出的攻击，都成功地实现了这一目标。然而，当神经网络在原始输出不可访问的硬标签设置下运行时，这一目标无法实现。在本文中，我们提出了第一个攻击，该攻击在理论上实现了硬标签设置下的函数等价提取，适用于RELU神经网络。通过在广泛的RELU神经网络上的实际实验，包括在两个在计算机视觉中广泛使用的真实基准数据集(MNIST，CIFAR10)上训练的神经网络，验证了该攻击的有效性。对于一个由$10^5$参数组成的神经网络，我们的攻击只需要在一个核上几个小时。



## **32. Image Hijacks: Adversarial Images can Control Generative Models at Runtime**

图像劫持：对抗图像可以随时控制生成模型 cs.LG

Project page at https://image-hijacks.github.io

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2309.00236v4) [paper-pdf](http://arxiv.org/pdf/2309.00236v4)

**Authors**: Luke Bailey, Euan Ong, Stuart Russell, Scott Emmons

**Abstract**: Are foundation models secure against malicious actors? In this work, we focus on the image input to a vision-language model (VLM). We discover image hijacks, adversarial images that control the behaviour of VLMs at inference time, and introduce the general Behaviour Matching algorithm for training image hijacks. From this, we derive the Prompt Matching method, allowing us to train hijacks matching the behaviour of an arbitrary user-defined text prompt (e.g. 'the Eiffel Tower is now located in Rome') using a generic, off-the-shelf dataset unrelated to our choice of prompt. We use Behaviour Matching to craft hijacks for four types of attack, forcing VLMs to generate outputs of the adversary's choice, leak information from their context window, override their safety training, and believe false statements. We study these attacks against LLaVA, a state-of-the-art VLM based on CLIP and LLaMA-2, and find that all attack types achieve a success rate of over 80%. Moreover, our attacks are automated and require only small image perturbations.

摘要: 基础模型针对恶意行为者是否安全？在这项工作中，我们关注的是图像输入到视觉语言模型(VLM)。我们发现了图像劫持，即在推理时控制VLM行为的敌意图像，并介绍了用于训练图像劫持的通用行为匹配算法。从这里，我们得到了提示匹配方法，允许我们使用与我们选择的提示无关的通用现成数据集来训练与任意用户定义的文本提示(例如‘埃菲尔铁塔现在位于罗马’)的行为匹配的劫持者。我们使用行为匹配来为四种类型的攻击制作劫持，迫使VLM生成对手选择的输出，从他们的上下文窗口泄露信息，覆盖他们的安全培训，并相信虚假陈述。我们对基于CLIP和LLAMA-2的最新VLM LLaVA进行了研究，发现所有类型的攻击都达到了80%以上的成功率。此外，我们的攻击是自动化的，只需要很小的图像扰动。



## **33. Golden Ratio Search: A Low-Power Adversarial Attack for Deep Learning based Modulation Classification**

黄金比率搜索：针对基于深度学习的调制分类的低功耗对抗攻击 cs.CR

5 pages, 1 figure, 3 tables

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2409.11454v1) [paper-pdf](http://arxiv.org/pdf/2409.11454v1)

**Authors**: Deepsayan Sadhukhan, Nitin Priyadarshini Shankar, Sheetal Kalyani

**Abstract**: We propose a minimal power white box adversarial attack for Deep Learning based Automatic Modulation Classification (AMC). The proposed attack uses the Golden Ratio Search (GRS) method to find powerful attacks with minimal power. We evaluate the efficacy of the proposed method by comparing it with existing adversarial attack approaches. Additionally, we test the robustness of the proposed attack against various state-of-the-art architectures, including defense mechanisms such as adversarial training, binarization, and ensemble methods. Experimental results demonstrate that the proposed attack is powerful, requires minimal power, and can be generated in less time, significantly challenging the resilience of current AMC methods.

摘要: 我们针对基于深度学习的自动调制分类（AMC）提出了一种最小功率白盒对抗攻击。拟议的攻击使用黄金比例搜索（GRS）方法来以最小的功率找到强大的攻击。我们通过与现有的对抗攻击方法进行比较来评估所提出方法的有效性。此外，我们还测试了针对各种最先进架构的拟议攻击的稳健性，包括对抗性训练、二进制化和集成方法等防御机制。实验结果表明，提出的攻击强大，所需功率最小，并且可以在更短的时间内产生，极大地挑战了当前AMC方法的弹性。



## **34. EIA: Environmental Injection Attack on Generalist Web Agents for Privacy Leakage**

EIA：针对多面手网络代理隐私泄露的环境注入攻击 cs.CR

24 pages

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2409.11295v1) [paper-pdf](http://arxiv.org/pdf/2409.11295v1)

**Authors**: Zeyi Liao, Lingbo Mo, Chejian Xu, Mintong Kang, Jiawei Zhang, Chaowei Xiao, Yuan Tian, Bo Li, Huan Sun

**Abstract**: Generalist web agents have evolved rapidly and demonstrated remarkable potential. However, there are unprecedented safety risks associated with these them, which are nearly unexplored so far. In this work, we aim to narrow this gap by conducting the first study on the privacy risks of generalist web agents in adversarial environments. First, we present a threat model that discusses the adversarial targets, constraints, and attack scenarios. Particularly, we consider two types of adversarial targets: stealing users' specific personally identifiable information (PII) or stealing the entire user request. To achieve these objectives, we propose a novel attack method, termed Environmental Injection Attack (EIA). This attack injects malicious content designed to adapt well to different environments where the agents operate, causing them to perform unintended actions. This work instantiates EIA specifically for the privacy scenario. It inserts malicious web elements alongside persuasive instructions that mislead web agents into leaking private information, and can further leverage CSS and JavaScript features to remain stealthy. We collect 177 actions steps that involve diverse PII categories on realistic websites from the Mind2Web dataset, and conduct extensive experiments using one of the most capable generalist web agent frameworks to date, SeeAct. The results demonstrate that EIA achieves up to 70% ASR in stealing users' specific PII. Stealing full user requests is more challenging, but a relaxed version of EIA can still achieve 16% ASR. Despite these concerning results, it is important to note that the attack can still be detectable through careful human inspection, highlighting a trade-off between high autonomy and security. This leads to our detailed discussion on the efficacy of EIA under different levels of human supervision as well as implications on defenses for generalist web agents.

摘要: 多面手网络代理发展迅速，并显示出非凡的潜力。然而，它们存在着前所未有的安全风险，到目前为止几乎没有人探索过。在这项工作中，我们旨在通过对对抗环境中通才网络代理的隐私风险进行第一次研究来缩小这一差距。首先，我们提出了一个威胁模型，该模型讨论了对抗性目标、约束和攻击场景。具体地说，我们考虑了两种类型的对抗目标：窃取用户特定的个人身份信息(PII)或窃取整个用户请求。为了实现这些目标，我们提出了一种新的攻击方法，称为环境注入攻击(EIA)。此攻击注入恶意内容，旨在很好地适应代理程序运行的不同环境，导致它们执行意外操作。这项工作专门为隐私场景实例化了EIA。它将恶意的网络元素与具有说服力的指令一起插入，误导网络代理泄露私人信息，并可以进一步利用CSS和JavaScript功能来保持隐蔽性。我们从Mind2Web数据集中收集了177个动作步骤，涉及现实网站上的不同PII类别，并使用迄今最有能力的通用Web代理框架之一SeeAct进行了广泛的实验。结果表明，在窃取用户特定PII时，EIA的ASR高达70%。窃取完整的用户请求更具挑战性，但宽松版本的EIA仍可实现16%的ASR。尽管有这些令人担忧的结果，但必须指出的是，通过仔细的人工检查仍然可以检测到攻击，这突显了高度自治和安全之间的权衡。这导致了我们详细讨论了在不同级别的人类监督下的EIA的有效性，以及对多面手网络代理的防御的影响。



## **35. Backdoor Attacks in Peer-to-Peer Federated Learning**

点对点联邦学习中的后门攻击 cs.LG

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2301.09732v4) [paper-pdf](http://arxiv.org/pdf/2301.09732v4)

**Authors**: Georgios Syros, Gokberk Yar, Simona Boboila, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Most machine learning applications rely on centralized learning processes, opening up the risk of exposure of their training datasets. While federated learning (FL) mitigates to some extent these privacy risks, it relies on a trusted aggregation server for training a shared global model. Recently, new distributed learning architectures based on Peer-to-Peer Federated Learning (P2PFL) offer advantages in terms of both privacy and reliability. Still, their resilience to poisoning attacks during training has not been investigated. In this paper, we propose new backdoor attacks for P2PFL that leverage structural graph properties to select the malicious nodes, and achieve high attack success, while remaining stealthy. We evaluate our attacks under various realistic conditions, including multiple graph topologies, limited adversarial visibility of the network, and clients with non-IID data. Finally, we show the limitations of existing defenses adapted from FL and design a new defense that successfully mitigates the backdoor attacks, without an impact on model accuracy.

摘要: 大多数机器学习应用程序依赖于集中的学习过程，从而打开了暴露其训练数据集的风险。虽然联合学习(FL)在一定程度上缓解了这些隐私风险，但它依赖于可信的聚合服务器来训练共享的全局模型。近年来，基于对等联合学习(P2P-to-Peer Federated Learning，简称P2PFL)的新型分布式学习体系结构在保密性和可靠性方面都具有优势。尽管如此，他们在训练期间对中毒攻击的抵抗力还没有得到调查。本文提出了一种新的针对P2P PFL的后门攻击，利用结构图的性质来选择恶意节点，在保持隐蔽性的同时获得较高的攻击成功率。我们在各种现实条件下评估我们的攻击，包括多个图拓扑、有限的网络敌意可见性以及具有非IID数据的客户端。最后，我们指出了现有防御方案的局限性，并设计了一种新的防御方案，在不影响模型精度的情况下，成功地缓解了后门攻击。



## **36. A Survey of Machine Unlearning**

机器学习研究 cs.LG

extend the survey with more recent published work and add more  discussions

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2209.02299v6) [paper-pdf](http://arxiv.org/pdf/2209.02299v6)

**Authors**: Thanh Tam Nguyen, Thanh Trung Huynh, Zhao Ren, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen

**Abstract**: Today, computer systems hold large amounts of personal data. Yet while such an abundance of data allows breakthroughs in artificial intelligence, and especially machine learning (ML), its existence can be a threat to user privacy, and it can weaken the bonds of trust between humans and AI. Recent regulations now require that, on request, private information about a user must be removed from both computer systems and from ML models, i.e. ``the right to be forgotten''). While removing data from back-end databases should be straightforward, it is not sufficient in the AI context as ML models often `remember' the old data. Contemporary adversarial attacks on trained models have proven that we can learn whether an instance or an attribute belonged to the training data. This phenomenon calls for a new paradigm, namely machine unlearning, to make ML models forget about particular data. It turns out that recent works on machine unlearning have not been able to completely solve the problem due to the lack of common frameworks and resources. Therefore, this paper aspires to present a comprehensive examination of machine unlearning's concepts, scenarios, methods, and applications. Specifically, as a category collection of cutting-edge studies, the intention behind this article is to serve as a comprehensive resource for researchers and practitioners seeking an introduction to machine unlearning and its formulations, design criteria, removal requests, algorithms, and applications. In addition, we aim to highlight the key findings, current trends, and new research areas that have not yet featured the use of machine unlearning but could benefit greatly from it. We hope this survey serves as a valuable resource for ML researchers and those seeking to innovate privacy technologies. Our resources are publicly available at https://github.com/tamlhp/awesome-machine-unlearning.

摘要: 今天，计算机系统保存着大量的个人数据。然而，尽管如此丰富的数据使人工智能，特别是机器学习(ML)取得了突破，但它的存在可能会对用户隐私构成威胁，并可能削弱人类与人工智能之间的信任纽带。最近的法规现在要求，根据请求，必须从计算机系统和ML模型中删除关于用户的私人信息，即“被遗忘权”)。虽然从后端数据库中删除数据应该是直接的，但在人工智能上下文中这是不够的，因为ML模型经常‘记住’旧数据。当代针对训练模型的对抗性攻击已经证明，我们可以学习到一个实例或一个属性是否属于训练数据。这种现象呼唤一种新的范式，即机器遗忘，以使ML模型忘记特定的数据。事实证明，由于缺乏通用的框架和资源，最近关于机器遗忘的研究并不能完全解决这个问题。因此，本文致力于对机器遗忘的概念、场景、方法和应用进行全面的考察。具体地说，作为尖端研究的类别集合，本文背后的目的是为寻求介绍机器遗忘及其公式、设计标准、移除请求、算法和应用的研究人员和从业者提供全面的资源。此外，我们的目标是强调关键的发现、当前的趋势和新的研究领域，这些领域还没有使用机器遗忘，但可以从中受益匪浅。我们希望这项调查对ML研究人员和那些寻求创新隐私技术的人来说是一个有价值的资源。我们的资源可在https://github.com/tamlhp/awesome-machine-unlearning.上公开获取



## **37. Remote Keylogging Attacks in Multi-user VR Applications**

多用户VR应用程序中的远程键盘记录攻击 cs.CR

Accepted for Usenix 2024

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2405.14036v2) [paper-pdf](http://arxiv.org/pdf/2405.14036v2)

**Authors**: Zihao Su, Kunlin Cai, Reuben Beeler, Lukas Dresel, Allan Garcia, Ilya Grishchenko, Yuan Tian, Christopher Kruegel, Giovanni Vigna

**Abstract**: As Virtual Reality (VR) applications grow in popularity, they have bridged distances and brought users closer together. However, with this growth, there have been increasing concerns about security and privacy, especially related to the motion data used to create immersive experiences. In this study, we highlight a significant security threat in multi-user VR applications, which are applications that allow multiple users to interact with each other in the same virtual space. Specifically, we propose a remote attack that utilizes the avatar rendering information collected from an adversary's game clients to extract user-typed secrets like credit card information, passwords, or private conversations. We do this by (1) extracting motion data from network packets, and (2) mapping motion data to keystroke entries. We conducted a user study to verify the attack's effectiveness, in which our attack successfully inferred 97.62% of the keystrokes. Besides, we performed an additional experiment to underline that our attack is practical, confirming its effectiveness even when (1) there are multiple users in a room, and (2) the attacker cannot see the victims. Moreover, we replicated our proposed attack on four applications to demonstrate the generalizability of the attack. Lastly, we proposed a defense against the attack, which has been implemented by major players in the VR industry. These results underscore the severity of the vulnerability and its potential impact on millions of VR social platform users.

摘要: 随着虚拟现实(VR)应用越来越受欢迎，它们弥合了距离，拉近了用户之间的距离。然而，随着这种增长，人们对安全和隐私的担忧也越来越多，特别是与用于创建身临其境体验的运动数据有关。在这项研究中，我们强调了多用户VR应用中的一个重大安全威胁，即允许多个用户在同一虚拟空间中相互交互的应用。具体地说，我们提出了一种远程攻击，它利用从对手的游戏客户端收集的化身渲染信息来提取用户键入的秘密，如信用卡信息、密码或私人对话。我们通过(1)从网络分组中提取运动数据，以及(2)将运动数据映射到击键条目来实现这一点。我们进行了用户研究来验证攻击的有效性，其中我们的攻击成功推断了97.62%的击键。此外，我们还执行了一个额外的实验，以强调我们的攻击是实用的，即使在(1)一个房间有多个用户，以及(2)攻击者看不到受害者的情况下，也证实了它的有效性。此外，我们在四个应用程序上复制了我们提出的攻击，以证明该攻击的泛化能力。最后，我们提出了针对攻击的防御方案，并已被VR行业的主要参与者实施。这些结果突显了该漏洞的严重性及其对数百万VR社交平台用户的潜在影响。



## **38. An Anti-disguise Authentication System Using the First Impression of Avatar in Metaverse**

利用虚拟宇宙阿凡达第一印象的反伪装认证系统 cs.CR

19 pages, 16 figures

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2409.10850v1) [paper-pdf](http://arxiv.org/pdf/2409.10850v1)

**Authors**: Zhenyong Zhang, Kedi Yang, Youliang Tian, Jianfeng Ma

**Abstract**: Metaverse is a vast virtual world parallel to the physical world, where the user acts as an avatar to enjoy various services that break through the temporal and spatial limitations of the physical world. Metaverse allows users to create arbitrary digital appearances as their own avatars by which an adversary may disguise his/her avatar to fraud others. In this paper, we propose an anti-disguise authentication method that draws on the idea of the first impression from the physical world to recognize an old friend. Specifically, the first meeting scenario in the metaverse is stored and recalled to help the authentication between avatars. To prevent the adversary from replacing and forging the first impression, we construct a chameleon-based signcryption mechanism and design a ciphertext authentication protocol to ensure the public verifiability of encrypted identities. The security analysis shows that the proposed signcryption mechanism meets not only the security requirement but also the public verifiability. Besides, the ciphertext authentication protocol has the capability of defending against the replacing and forging attacks on the first impression. Extensive experiments show that the proposed avatar authentication system is able to achieve anti-disguise authentication at a low storage consumption on the blockchain.

摘要: Metverse是一个与物理世界平行的广阔虚拟世界，用户在其中扮演化身，享受各种突破物理世界时空限制的服务。Metverse允许用户创建任意的数字外观作为他们自己的化身，对手可以利用这些化身来伪装他/她的化身以欺骗他人。在本文中，我们提出了一种反伪装认证方法，该方法借鉴了物理世界第一印象的思想来识别老朋友。具体地说，存储和调用虚拟世界中的第一个会议场景，以帮助在化身之间进行身份验证。为了防止攻击者替换和伪造第一印象，我们构造了一个基于变色龙的签密机制，并设计了一个密文认证协议来确保加密身份的公开可验证性。安全性分析表明，该签密机制不仅满足安全性要求，而且具有公开可验证性。此外，密文认证协议还具有抵抗替换攻击和伪造第一印象攻击的能力。大量实验表明，所提出的头像认证系统能够在区块链上以较低的存储消耗实现反伪装认证。



## **39. Weak Superimposed Codes of Improved Asymptotic Rate and Their Randomized Construction**

改进渐进率的弱叠加码及其随机构造 cs.IT

6 pages, accepted for presentation at the 2022 IEEE International  Symposium on Information Theory (ISIT)

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2409.10511v1) [paper-pdf](http://arxiv.org/pdf/2409.10511v1)

**Authors**: Yu Tsunoda, Yuichiro Fujiwara

**Abstract**: Weak superimposed codes are combinatorial structures related closely to generalized cover-free families, superimposed codes, and disjunct matrices in that they are only required to satisfy similar but less stringent conditions. This class of codes may also be seen as a stricter variant of what are known as locally thin families in combinatorics. Originally, weak superimposed codes were introduced in the context of multimedia content protection against illegal distribution of copies under the assumption that a coalition of malicious users may employ the averaging attack with adversarial noise. As in many other kinds of codes in information theory, it is of interest and importance in the study of weak superimposed codes to find the highest achievable rate in the asymptotic regime and give an efficient construction that produces an infinite sequence of codes that achieve it. Here, we prove a tighter lower bound than the sharpest known one on the rate of optimal weak superimposed codes and give a polynomial-time randomized construction algorithm for codes that asymptotically attain our improved bound with high probability. Our probabilistic approach is versatile and applicable to many other related codes and arrays.

摘要: 弱叠加码是一种与广义无覆盖族、叠加码和析取矩阵密切相关的组合结构，它们只需要满足相似但不那么严格的条件。这类代码也可以被视为组合数学中所知的局部瘦族的更严格的变体。最初，在防止非法分发副本的多媒体内容保护的环境中引入了弱叠加代码，假设恶意用户的联盟可能采用带有对抗性噪声的平均攻击。正如信息论中的许多其他类型的码一样，在弱重叠码的研究中，寻找渐近状态下的最高可达速率并给出一种有效的构造以产生实现它的无限序列是很有意义和重要的。在这里，我们证明了最优弱叠加码码率的一个比已知的最强下界更紧的下界，并给出了一个多项式时间的随机构造算法，它以很高的概率渐近地达到我们的改进界。我们的概率方法是通用的，并适用于许多其他相关的代码和数组。



## **40. Assessing biomedical knowledge robustness in large language models by query-efficient sampling attacks**

通过查询高效抽样攻击评估大型语言模型中生物医学知识的稳健性 cs.CL

28 pages incl. appendix, updated version

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2402.10527v2) [paper-pdf](http://arxiv.org/pdf/2402.10527v2)

**Authors**: R. Patrick Xian, Alex J. Lee, Satvik Lolla, Vincent Wang, Qiming Cui, Russell Ro, Reza Abbasi-Asl

**Abstract**: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. Understanding model vulnerabilities in high-stakes and knowledge-intensive tasks is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples (i.e. adversarial entities) in natural language processing tasks raises questions about their potential impact on the knowledge robustness of pre-trained and finetuned LLMs in high-stakes and specialized domains. We examined the use of type-consistent entity substitution as a template for collecting adversarial entities for billion-parameter LLMs with biomedical knowledge. To this end, we developed an embedding-space attack based on powerscaled distance-weighted sampling to assess the robustness of their biomedical knowledge with a low query budget and controllable coverage. Our method has favorable query efficiency and scaling over alternative approaches based on random sampling and blackbox gradient-guided search, which we demonstrated for adversarial distractor generation in biomedical question answering. Subsequent failure mode analysis uncovered two regimes of adversarial entities on the attack surface with distinct characteristics and we showed that entity substitution attacks can manipulate token-wise Shapley value explanations, which become deceptive in this setting. Our approach complements standard evaluations for high-capacity models and the results highlight the brittleness of domain knowledge in LLMs.

摘要: 大型语言模型(LLM)中参数领域知识的不断深入推动了它们在现实世界应用程序中的快速部署。了解高风险和知识密集型任务中的模型脆弱性对于量化模型预测的可信度和规范其使用至关重要。最近在自然语言处理任务中发现了命名实体作为对抗性实例(即对抗性实体)，这引发了人们对高风险和专门领域中预先训练和精细调整的LLM知识稳健性的潜在影响的问题。我们研究了使用类型一致的实体替换作为收集具有生物医学知识的10亿参数LLM的对抗性实体的模板。为此，我们提出了一种基于加权距离加权抽样的嵌入空间攻击方法，以较低的查询预算和可控的覆盖率来评估他们的生物医学知识的稳健性。与基于随机抽样和黑盒梯度引导搜索的方法相比，我们的方法具有良好的查询效率和伸缩性，并在生物医学问答中的对抗性干扰项生成中得到了验证。随后的失效模式分析揭示了攻击面上具有不同特征的两种对抗实体的机制，我们表明实体替换攻击可以操纵令人信服的Shapley值解释，在这种情况下，这种解释变得具有欺骗性。我们的方法补充了对大容量模型的标准评估，结果突出了领域知识在LLMS中的脆性。



## **41. Deep Reinforcement Learning for Autonomous Cyber Operations: A Survey**

自主网络运营的深度强化学习：调查 cs.LG

89 pages, 14 figures, 4 tables

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2310.07745v2) [paper-pdf](http://arxiv.org/pdf/2310.07745v2)

**Authors**: Gregory Palmer, Chris Parry, Daniel J. B. Harrold, Chris Willis

**Abstract**: The rapid increase in the number of cyber-attacks in recent years raises the need for principled methods for defending networks against malicious actors. Deep reinforcement learning (DRL) has emerged as a promising approach for mitigating these attacks. However, while DRL has shown much potential for cyber defence, numerous challenges must be overcome before DRL can be applied to autonomous cyber operations (ACO) at scale. Principled methods are required for environments that confront learners with very high-dimensional state spaces, large multi-discrete action spaces, and adversarial learning. Recent works have reported success in solving these problems individually. There have also been impressive engineering efforts towards solving all three for real-time strategy games. However, applying DRL to the full ACO problem remains an open challenge. Here, we survey the relevant DRL literature and conceptualize an idealised ACO-DRL agent. We provide: i.) A summary of the domain properties that define the ACO problem; ii.) A comprehensive comparison of current ACO environments used for benchmarking DRL approaches; iii.) An overview of state-of-the-art approaches for scaling DRL to domains that confront learners with the curse of dimensionality, and; iv.) A survey and critique of current methods for limiting the exploitability of agents within adversarial settings from the perspective of ACO. We conclude with open research questions that we hope will motivate future directions for researchers and practitioners working on ACO.

摘要: 近年来，网络攻击数量的迅速增加增加了对保护网络免受恶意行为侵害的原则性方法的需求。深度强化学习(DRL)已成为缓解这些攻击的一种很有前途的方法。然而，尽管DRL在网络防御方面显示出了很大的潜力，但在DRL能够大规模应用于自主网络作战(ACO)之前，必须克服许多挑战。对于学习者面对高维状态空间、大的多离散动作空间和对抗性学习的环境，需要有原则性的方法。最近的研究报告成功地单独解决了这些问题。也有令人印象深刻的工程努力，以解决所有这三个实时战略游戏。然而，将DRL应用于整个蚁群优化问题仍然是一个开放的挑战。在这里，我们回顾了相关的DRL文献，并概念化了一个理想的ACO-DRL试剂。我们提供：i.)定义ACO问题的域属性摘要；ii.)对当前用于基准DRL方法的ACO环境进行了全面比较；三.)概述将DRL扩展到学习者面临维度诅咒的领域的最新方法，以及；i.)从蚁群算法的角度对当前在对抗性环境中限制代理的可利用性的方法进行了调查和评论。我们以开放的研究问题结束，我们希望这些问题将激励从事ACO工作的研究人员和从业者未来的方向。



## **42. Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents**

机器对抗RAG：用阻止器文档干扰检索增强生成 cs.CR

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.05870v2) [paper-pdf](http://arxiv.org/pdf/2406.05870v2)

**Authors**: Avital Shafran, Roei Schuster, Vitaly Shmatikov

**Abstract**: Retrieval-augmented generation (RAG) systems respond to queries by retrieving relevant documents from a knowledge database, then generating an answer by applying an LLM to the retrieved documents. We demonstrate that RAG systems that operate on databases with untrusted content are vulnerable to a new class of denial-of-service attacks we call jamming. An adversary can add a single ``blocker'' document to the database that will be retrieved in response to a specific query and result in the RAG system not answering this query - ostensibly because it lacks the information or because the answer is unsafe.   We describe and measure the efficacy of several methods for generating blocker documents, including a new method based on black-box optimization. This method (1) does not rely on instruction injection, (2) does not require the adversary to know the embedding or LLM used by the target RAG system, and (3) does not use an auxiliary LLM to generate blocker documents.   We evaluate jamming attacks on several LLMs and embeddings and demonstrate that the existing safety metrics for LLMs do not capture their vulnerability to jamming. We then discuss defenses against blocker documents.

摘要: 检索-增强生成(RAG)系统通过从知识数据库中检索相关文档，然后通过将LLM应用于所检索的文档来生成答案来响应查询。我们证明，在含有不可信内容的数据库上运行的RAG系统容易受到一种新的拒绝服务攻击，我们称之为干扰。敌手可以向数据库添加一个“拦截器”文档，该文档将响应于特定查询而被检索，并导致RAG系统不回答该查询--表面上是因为它缺乏信息或因为答案不安全。我们描述并测试了几种生成拦截器文档的方法的有效性，其中包括一种基于黑盒优化的新方法。该方法(1)不依赖于指令注入，(2)不要求对手知道目标RAG系统使用的嵌入或LLM，以及(3)不使用辅助LLM来生成拦截器文档。我们评估了几个LLM和嵌入上的干扰攻击，并证明了现有的LLM安全度量没有捕捉到它们对干扰的脆弱性。然后我们讨论针对拦截器文档的防御。



## **43. Towards Evaluating the Robustness of Visual State Space Models**

评估视觉状态空间模型的稳健性 cs.CV

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.09407v2) [paper-pdf](http://arxiv.org/pdf/2406.09407v2)

**Authors**: Hashmat Shadab Malik, Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar, Fahad Shahbaz Khan, Salman Khan

**Abstract**: Vision State Space Models (VSSMs), a novel architecture that combines the strengths of recurrent neural networks and latent variable models, have demonstrated remarkable performance in visual perception tasks by efficiently capturing long-range dependencies and modeling complex visual dynamics. However, their robustness under natural and adversarial perturbations remains a critical concern. In this work, we present a comprehensive evaluation of VSSMs' robustness under various perturbation scenarios, including occlusions, image structure, common corruptions, and adversarial attacks, and compare their performance to well-established architectures such as transformers and Convolutional Neural Networks. Furthermore, we investigate the resilience of VSSMs to object-background compositional changes on sophisticated benchmarks designed to test model performance in complex visual scenes. We also assess their robustness on object detection and segmentation tasks using corrupted datasets that mimic real-world scenarios. To gain a deeper understanding of VSSMs' adversarial robustness, we conduct a frequency-based analysis of adversarial attacks, evaluating their performance against low-frequency and high-frequency perturbations. Our findings highlight the strengths and limitations of VSSMs in handling complex visual corruptions, offering valuable insights for future research. Our code and models will be available at https://github.com/HashmatShadab/MambaRobustness.

摘要: 视觉状态空间模型(VSSMS)是一种结合了递归神经网络和潜变量模型优点的新型结构，通过有效地捕捉长距离依赖关系和建模复杂的视觉动力学，在视觉感知任务中表现出了显著的性能。然而，它们在自然和对抗性扰动下的稳健性仍然是一个严重的问题。在这项工作中，我们对VSSM在各种扰动场景下的健壮性进行了全面的评估，包括遮挡、图像结构、常见的腐败和敌对攻击，并将它们的性能与成熟的架构，如变压器和卷积神经网络进行了比较。此外，我们在复杂的基准测试中考察了VSSM对对象-背景成分变化的弹性，该基准旨在测试复杂视觉场景中的模型性能。我们还使用模拟真实世界场景的损坏数据集评估了它们在对象检测和分割任务中的稳健性。为了更深入地了解VSSM的对抗稳健性，我们对对抗攻击进行了基于频率的分析，评估了它们对低频和高频扰动的性能。我们的发现突出了VSSM在处理复杂视觉腐败方面的优势和局限性，为未来的研究提供了有价值的见解。我们的代码和模型将在https://github.com/HashmatShadab/MambaRobustness.上提供



## **44. Multi-agent Attacks for Black-box Social Recommendations**

针对黑匣子社交推荐的多代理攻击 cs.SI

Accepted by ACM TOIS

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2311.07127v4) [paper-pdf](http://arxiv.org/pdf/2311.07127v4)

**Authors**: Shijie Wang, Wenqi Fan, Xiao-yong Wei, Xiaowei Mei, Shanru Lin, Qing Li

**Abstract**: The rise of online social networks has facilitated the evolution of social recommender systems, which incorporate social relations to enhance users' decision-making process. With the great success of Graph Neural Networks (GNNs) in learning node representations, GNN-based social recommendations have been widely studied to model user-item interactions and user-user social relations simultaneously. Despite their great successes, recent studies have shown that these advanced recommender systems are highly vulnerable to adversarial attacks, in which attackers can inject well-designed fake user profiles to disrupt recommendation performances. While most existing studies mainly focus on argeted attacks to promote target items on vanilla recommender systems, untargeted attacks to degrade the overall prediction performance are less explored on social recommendations under a black-box scenario. To perform untargeted attacks on social recommender systems, attackers can construct malicious social relationships for fake users to enhance the attack performance. However, the coordination of social relations and item profiles is challenging for attacking black-box social recommendations. To address this limitation, we first conduct several preliminary studies to demonstrate the effectiveness of cross-community connections and cold-start items in degrading recommendations performance. Specifically, we propose a novel framework MultiAttack based on multi-agent reinforcement learning to coordinate the generation of cold-start item profiles and cross-community social relations for conducting untargeted attacks on black-box social recommendations. Comprehensive experiments on various real-world datasets demonstrate the effectiveness of our proposed attacking framework under the black-box setting.

摘要: 在线社交网络的兴起促进了社交推荐系统的发展，社交推荐系统整合了社会关系，以增强用户的决策过程。随着图神经网络(GNN)在学习节点表示方面的巨大成功，基于GNN的社交推荐被广泛研究以同时建模用户-项目交互和用户-用户社会关系。尽管它们取得了巨大的成功，但最近的研究表明，这些先进的推荐系统非常容易受到对手攻击，攻击者可以注入精心设计的虚假用户配置文件来破坏推荐性能。虽然现有的研究主要集中在香草推荐系统上为推广目标项而进行的有针对性的攻击，但在黑盒场景下的社交推荐中，降低整体预测性能的非目标攻击的研究较少。为了对社交推荐系统进行无针对性的攻击，攻击者可以为虚假用户构建恶意的社交关系，以提高攻击性能。然而，社交关系和项目简介的协调对于攻击黑箱社交推荐是具有挑战性的。为了解决这一局限性，我们首先进行了几项初步研究，以证明跨社区联系和冷启动项目在降低推荐性能方面的有效性。具体地说，我们提出了一种基于多智能体强化学习的新型框架MultiAttack，用于协调冷启动项目配置文件的生成和跨社区社会关系的生成，以对黑盒社交推荐进行无针对性的攻击。在各种真实数据集上的综合实验证明了我们提出的攻击框架在黑盒环境下的有效性。



## **45. Towards Adversarial Robustness And Backdoor Mitigation in SSL**

SSL中的对抗稳健性和后门缓解 cs.CV

8 pages, 2 figures

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2403.15918v3) [paper-pdf](http://arxiv.org/pdf/2403.15918v3)

**Authors**: Aryan Satpathy, Nilaksh Singh, Dhruva Rajwade, Somesh Kumar

**Abstract**: Self-Supervised Learning (SSL) has shown great promise in learning representations from unlabeled data. The power of learning representations without the need for human annotations has made SSL a widely used technique in real-world problems. However, SSL methods have recently been shown to be vulnerable to backdoor attacks, where the learned model can be exploited by adversaries to manipulate the learned representations, either through tampering the training data distribution, or via modifying the model itself. This work aims to address defending against backdoor attacks in SSL, where the adversary has access to a realistic fraction of the SSL training data, and no access to the model. We use novel methods that are computationally efficient as well as generalizable across different problem settings. We also investigate the adversarial robustness of SSL models when trained with our method, and show insights into increased robustness in SSL via frequency domain augmentations. We demonstrate the effectiveness of our method on a variety of SSL benchmarks, and show that our method is able to mitigate backdoor attacks while maintaining high performance on downstream tasks. Code for our work is available at github.com/Aryan-Satpathy/Backdoor

摘要: 自监督学习(SSL)在从未标记数据中学习表示方面显示出巨大的前景。无需人工注释即可学习表示的能力使SSL成为实际问题中广泛使用的技术。然而，最近已证明SSL方法容易受到后门攻击，攻击者可以通过篡改训练数据分发或通过修改模型本身来利用学习的模型来操纵学习的表示。这项工作旨在解决针对SSL中的后门攻击的防御，在这种攻击中，攻击者可以访问真实的一小部分SSL训练数据，而不能访问模型。我们使用新的方法，这些方法在计算上是有效的，并且可以在不同的问题设置中推广。我们还研究了使用我们的方法训练的SSL模型的对抗稳健性，并展示了通过频域增强来增强SSL的稳健性的见解。我们在不同的SSL基准测试上证明了我们的方法的有效性，并表明我们的方法能够在保持下游任务的高性能的同时减少后门攻击。我们工作的代码可在githorb.com/aryan-Satthy/Backdoor上找到



## **46. Trading Devil: Robust backdoor attack via Stochastic investment models and Bayesian approach**

交易魔鬼：通过随机投资模型和Bayesian方法进行强有力的后门攻击 cs.CR

(Last update!, a constructive comment from arxiv led to this latest  update ) Stochastic investment models and a Bayesian approach to better  modeling of uncertainty : adversarial machine learning or Stochastic market.  arXiv admin note: substantial text overlap with arXiv:2402.05967 (see this  link to the paper by : Orson Mengara)

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.10719v4) [paper-pdf](http://arxiv.org/pdf/2406.10719v4)

**Authors**: Orson Mengara

**Abstract**: With the growing use of voice-activated systems and speech recognition technologies, the danger of backdoor attacks on audio data has grown significantly. This research looks at a specific type of attack, known as a Stochastic investment-based backdoor attack (MarketBack), in which adversaries strategically manipulate the stylistic properties of audio to fool speech recognition systems. The security and integrity of machine learning models are seriously threatened by backdoor attacks, in order to maintain the reliability of audio applications and systems, the identification of such attacks becomes crucial in the context of audio data. Experimental results demonstrated that MarketBack is feasible to achieve an average attack success rate close to 100% in seven victim models when poisoning less than 1% of the training data.

摘要: 随着语音激活系统和语音识别技术的日益广泛使用，对音频数据进行后门攻击的危险显着增加。这项研究着眼于一种特定类型的攻击，称为基于随机投资的后门攻击（MarketBack），其中对手战略性地操纵音频的风格属性来愚弄语音识别系统。机器学习模型的安全性和完整性受到后门攻击的严重威胁，为了维护音频应用和系统的可靠性，识别此类攻击在音频数据环境中变得至关重要。实验结果表明，当毒害少于1%的训练数据时，MarketBack可以在7个受害者模型中实现接近100%的平均攻击成功率。



## **47. Exact Recovery Guarantees for Parameterized Non-linear System Identification Problem under Adversarial Attacks**

对抗攻击下参数化非线性系统识别问题的精确恢复保证 math.OC

33 pages

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2409.00276v2) [paper-pdf](http://arxiv.org/pdf/2409.00276v2)

**Authors**: Haixiang Zhang, Baturalp Yalcin, Javad Lavaei, Eduardo D. Sontag

**Abstract**: In this work, we study the system identification problem for parameterized non-linear systems using basis functions under adversarial attacks. Motivated by the LASSO-type estimators, we analyze the exact recovery property of a non-smooth estimator, which is generated by solving an embedded $\ell_1$-loss minimization problem. First, we derive necessary and sufficient conditions for the well-specifiedness of the estimator and the uniqueness of global solutions to the underlying optimization problem. Next, we provide exact recovery guarantees for the estimator under two different scenarios of boundedness and Lipschitz continuity of the basis functions. The non-asymptotic exact recovery is guaranteed with high probability, even when there are more severely corrupted data than clean data. Finally, we numerically illustrate the validity of our theory. This is the first study on the sample complexity analysis of a non-smooth estimator for the non-linear system identification problem.

摘要: 在这项工作中，我们研究了对抗攻击下使用基函数的参数化非线性系统的系统识别问题。受LANSO型估计器的激励，我们分析了非光滑估计器的精确恢复性质，该估计器是通过解决嵌入的$\ell_1 $-损失最小化问题而生成的。首先，我们推导出估计量的良好指定性和基本优化问题的全局解的唯一性的充要条件。接下来，我们在基函数的有界性和Lipschitz连续性两种不同场景下为估计器提供精确的恢复保证。即使存在比干净数据更严重的损坏数据，也能以高概率保证非渐进精确恢复。最后，我们用数字说明了我们理论的有效性。这是首次对非线性系统识别问题的非光滑估计器的样本复杂性分析进行研究。



## **48. LLM Whisperer: An Inconspicuous Attack to Bias LLM Responses**

LLM Whisperer：对LLM偏见回应的不起眼攻击 cs.CR

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.04755v2) [paper-pdf](http://arxiv.org/pdf/2406.04755v2)

**Authors**: Weiran Lin, Anna Gerchanovsky, Omer Akgul, Lujo Bauer, Matt Fredrikson, Zifan Wang

**Abstract**: Writing effective prompts for large language models (LLM) can be unintuitive and burdensome. In response, services that optimize or suggest prompts have emerged. While such services can reduce user effort, they also introduce a risk: the prompt provider can subtly manipulate prompts to produce heavily biased LLM responses. In this work, we show that subtle synonym replacements in prompts can increase the likelihood (by a difference up to 78%) that LLMs mention a target concept (e.g., a brand, political party, nation). We substantiate our observations through a user study, showing our adversarially perturbed prompts 1) are indistinguishable from unaltered prompts by humans, 2) push LLMs to recommend target concepts more often, and 3) make users more likely to notice target concepts, all without arousing suspicion. The practicality of this attack has the potential to undermine user autonomy. Among other measures, we recommend implementing warnings against using prompts from untrusted parties.

摘要: 为大型语言模型(LLM)编写有效的提示可能是不直观和繁琐的。作为回应，优化或建议提示的服务应运而生。虽然这类服务可以减少用户的工作，但它们也带来了风险：提示提供商可能会巧妙地操纵提示，以产生严重偏见的LLM响应。在这项工作中，我们表明，提示中微妙的同义词替换可以增加LLMS提到目标概念(例如，品牌、政党、国家)的可能性(差异高达78%)。我们通过一项用户研究证实了我们的观察结果，表明我们受到敌意干扰的提示1)与人类未更改的提示无法区分，2)推动LLMS更频繁地推荐目标概念，3)使用户更有可能注意到目标概念，所有这些都不会引起怀疑。这种攻击的实用性有可能破坏用户的自主性。在其他措施中，我们建议实施警告，以防止使用来自不受信任方的提示。



## **49. Revisiting Physical-World Adversarial Attack on Traffic Sign Recognition: A Commercial Systems Perspective**

重新审视对交通标志识别的物理世界对抗攻击：商业系统的角度 cs.CR

Accepted by NDSS 2025

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.09860v1) [paper-pdf](http://arxiv.org/pdf/2409.09860v1)

**Authors**: Ningfei Wang, Shaoyuan Xie, Takami Sato, Yunpeng Luo, Kaidi Xu, Qi Alfred Chen

**Abstract**: Traffic Sign Recognition (TSR) is crucial for safe and correct driving automation. Recent works revealed a general vulnerability of TSR models to physical-world adversarial attacks, which can be low-cost, highly deployable, and capable of causing severe attack effects such as hiding a critical traffic sign or spoofing a fake one. However, so far existing works generally only considered evaluating the attack effects on academic TSR models, leaving the impacts of such attacks on real-world commercial TSR systems largely unclear. In this paper, we conduct the first large-scale measurement of physical-world adversarial attacks against commercial TSR systems. Our testing results reveal that it is possible for existing attack works from academia to have highly reliable (100\%) attack success against certain commercial TSR system functionality, but such attack capabilities are not generalizable, leading to much lower-than-expected attack success rates overall. We find that one potential major factor is a spatial memorization design that commonly exists in today's commercial TSR systems. We design new attack success metrics that can mathematically model the impacts of such design on the TSR system-level attack success, and use them to revisit existing attacks. Through these efforts, we uncover 7 novel observations, some of which directly challenge the observations or claims in prior works due to the introduction of the new metrics.

摘要: 交通标志识别(TSR)对于安全、正确的驾驶自动化至关重要。最近的工作揭示了TSR模型对物理世界对抗性攻击的普遍脆弱性，这些攻击可以是低成本的，高度可部署的，并且能够造成严重的攻击效果，例如隐藏关键交通标志或欺骗假交通标志。然而，到目前为止，现有的工作一般只考虑评估攻击对学术TSR模型的影响，而对现实世界商业TSR系统的影响很大程度上是未知的。在本文中，我们首次进行了针对商业TSR系统的物理世界对抗性攻击的大规模测量。我们的测试结果表明，学术界现有的攻击工作有可能对某些商用TSR系统功能具有高可靠性(100%)的攻击成功，但这种攻击能力不是通用的，导致总体攻击成功率远低于预期。我们发现一个潜在的主要因素是空间记忆设计，这种设计普遍存在于今天的商业TSR系统中。我们设计了新的攻击成功度量，可以对这种设计对TSR系统级攻击成功的影响进行数学建模，并使用它们来重新审视现有的攻击。通过这些努力，我们发现了7个新颖的观察结果，其中一些由于新度量的引入直接挑战了先前工作中的观察或主张。



## **50. Federated Learning in Adversarial Environments: Testbed Design and Poisoning Resilience in Cybersecurity**

对抗环境中的联邦学习：网络安全中的测试床设计和毒害韧性 cs.CR

7 pages, 4 figures

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.09794v1) [paper-pdf](http://arxiv.org/pdf/2409.09794v1)

**Authors**: Hao Jian Huang, Bekzod Iskandarov, Mizanur Rahman, Hakan T. Otal, M. Abdullah Canbaz

**Abstract**: This paper presents the design and implementation of a Federated Learning (FL) testbed, focusing on its application in cybersecurity and evaluating its resilience against poisoning attacks. Federated Learning allows multiple clients to collaboratively train a global model while keeping their data decentralized, addressing critical needs for data privacy and security, particularly in sensitive fields like cybersecurity. Our testbed, built using the Flower framework, facilitates experimentation with various FL frameworks, assessing their performance, scalability, and ease of integration. Through a case study on federated intrusion detection systems, we demonstrate the testbed's capabilities in detecting anomalies and securing critical infrastructure without exposing sensitive network data. Comprehensive poisoning tests, targeting both model and data integrity, evaluate the system's robustness under adversarial conditions. Our results show that while federated learning enhances data privacy and distributed learning, it remains vulnerable to poisoning attacks, which must be mitigated to ensure its reliability in real-world applications.

摘要: 本文介绍了联邦学习(FL)测试平台的设计与实现，重点研究了其在网络安全中的应用，并对其抗中毒攻击的能力进行了评估。联合学习允许多个客户协作培训全球模型，同时保持他们的数据分散，满足数据隐私和安全的关键需求，特别是在网络安全等敏感领域。我们的试验台使用Flower框架构建，促进了各种FL框架的实验，评估了它们的性能、可扩展性和集成简易性。通过一个联合入侵检测系统的案例研究，我们展示了测试床在检测异常和保护关键基础设施方面的能力，而不会暴露敏感的网络数据。针对模型和数据完整性的全面中毒测试，评估系统在对抗条件下的健壮性。我们的结果表明，尽管联合学习增强了数据隐私和分布式学习，但它仍然容易受到中毒攻击，必须加以缓解，以确保其在现实世界应用中的可靠性。



