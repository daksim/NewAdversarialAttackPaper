# Latest Adversarial Attack Papers
**update at 2024-02-21 10:59:58**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Unfolding Local Growth Rate Estimates for (Almost) Perfect Adversarial Detection**

关于(几乎)完美敌意检测的局部增长率估计 cs.CV

accepted at VISAPP23

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2212.06776v4) [paper-pdf](http://arxiv.org/pdf/2212.06776v4)

**Authors**: Peter Lorenz, Margret Keuper, Janis Keuper

**Abstract**: Convolutional neural networks (CNN) define the state-of-the-art solution on many perceptual tasks. However, current CNN approaches largely remain vulnerable against adversarial perturbations of the input that have been crafted specifically to fool the system while being quasi-imperceptible to the human eye. In recent years, various approaches have been proposed to defend CNNs against such attacks, for example by model hardening or by adding explicit defence mechanisms. Thereby, a small "detector" is included in the network and trained on the binary classification task of distinguishing genuine data from data containing adversarial perturbations. In this work, we propose a simple and light-weight detector, which leverages recent findings on the relation between networks' local intrinsic dimensionality (LID) and adversarial attacks. Based on a re-interpretation of the LID measure and several simple adaptations, we surpass the state-of-the-art on adversarial detection by a significant margin and reach almost perfect results in terms of F1-score for several networks and datasets. Sources available at: https://github.com/adverML/multiLID

摘要: 卷积神经网络(CNN)定义了许多感知任务的最先进的解决方案。然而，目前的CNN方法在很大程度上仍然容易受到输入的对抗性扰动，这些扰动是专门为愚弄系统而设计的，而人眼几乎察觉不到。近年来，已经提出了各种方法来保护CNN免受此类攻击，例如通过模型硬化或通过添加显式防御机制。因此，在网络中包括一个小的“检测器”，并在区分真实数据和包含对抗性扰动的数据的二进制分类任务上进行训练。在这项工作中，我们提出了一个简单而轻量级的检测器，它利用了最近关于网络的局部固有维度(LID)与对手攻击之间关系的研究结果。基于对LID度量的重新解释和几个简单的适应，我们在对手检测方面远远超过了最先进的水平，并在几个网络和数据集的F1得分方面取得了几乎完美的结果。资料来源：https://github.com/adverML/multiLID



## **2. Is RobustBench/AutoAttack a suitable Benchmark for Adversarial Robustness?**

RobustBtch/AutoAttack是衡量对手健壮性的合适基准吗？ cs.CV

AAAI-22 AdvML Workshop

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2112.01601v4) [paper-pdf](http://arxiv.org/pdf/2112.01601v4)

**Authors**: Peter Lorenz, Dominik Strassel, Margret Keuper, Janis Keuper

**Abstract**: Recently, RobustBench (Croce et al. 2020) has become a widely recognized benchmark for the adversarial robustness of image classification networks. In its most commonly reported sub-task, RobustBench evaluates and ranks the adversarial robustness of trained neural networks on CIFAR10 under AutoAttack (Croce and Hein 2020b) with l-inf perturbations limited to eps = 8/255. With leading scores of the currently best performing models of around 60% of the baseline, it is fair to characterize this benchmark to be quite challenging. Despite its general acceptance in recent literature, we aim to foster discussion about the suitability of RobustBench as a key indicator for robustness which could be generalized to practical applications. Our line of argumentation against this is two-fold and supported by excessive experiments presented in this paper: We argue that I) the alternation of data by AutoAttack with l-inf, eps = 8/255 is unrealistically strong, resulting in close to perfect detection rates of adversarial samples even by simple detection algorithms and human observers. We also show that other attack methods are much harder to detect while achieving similar success rates. II) That results on low-resolution data sets like CIFAR10 do not generalize well to higher resolution images as gradient-based attacks appear to become even more detectable with increasing resolutions.

摘要: 最近，RobustBench（Croce et al. 2020）已成为图像分类网络对抗鲁棒性的广泛认可的基准。在其最常报告的子任务中，RobustBench在AutoAttack（Croce和Hein 2020 b）下评估并排名CIFAR 10上训练的神经网络的对抗鲁棒性，其中l-inf扰动限制为eps = 8/255。由于目前表现最好的模型的领先分数约为基线的60%，因此将此基准描述为相当具有挑战性是公平的。尽管它在最近的文献中被普遍接受，我们的目标是促进讨论RobustBench作为鲁棒性的关键指标，可以推广到实际应用的适用性。我们对此的论证是双重的，并得到了本文中大量实验的支持：我们认为I）AutoAttack使用l-inf，eps = 8/255对数据进行的更改是不切实际的，即使通过简单的检测算法和人类观察者，也会导致对抗样本的接近完美的检测率。我们还表明，其他攻击方法更难检测，同时实现类似的成功率。II）CIFAR 10等低分辨率数据集的结果不能很好地推广到更高分辨率的图像，因为随着分辨率的提高，基于梯度的攻击似乎变得更容易检测。



## **3. Detecting AutoAttack Perturbations in the Frequency Domain**

在频域中检测AutoAttack扰动 cs.CV

accepted at ICML 2021 workshop for robustness

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2111.08785v3) [paper-pdf](http://arxiv.org/pdf/2111.08785v3)

**Authors**: Peter Lorenz, Paula Harder, Dominik Strassel, Margret Keuper, Janis Keuper

**Abstract**: Recently, adversarial attacks on image classification networks by the AutoAttack (Croce and Hein, 2020b) framework have drawn a lot of attention. While AutoAttack has shown a very high attack success rate, most defense approaches are focusing on network hardening and robustness enhancements, like adversarial training. This way, the currently best-reported method can withstand about 66% of adversarial examples on CIFAR10. In this paper, we investigate the spatial and frequency domain properties of AutoAttack and propose an alternative defense. Instead of hardening a network, we detect adversarial attacks during inference, rejecting manipulated inputs. Based on a rather simple and fast analysis in the frequency domain, we introduce two different detection algorithms. First, a black box detector that only operates on the input images and achieves a detection accuracy of 100% on the AutoAttack CIFAR10 benchmark and 99.3% on ImageNet, for epsilon = 8/255 in both cases. Second, a whitebox detector using an analysis of CNN feature maps, leading to a detection rate of also 100% and 98.7% on the same benchmarks.

摘要: 最近，AutoAttack(Croce and Hein，2020b)框架对图像分类网络的敌意攻击引起了人们的极大关注。虽然AutoAttack显示出非常高的攻击成功率，但大多数防御方法都专注于网络加固和健壮性增强，如对抗性训练。这样，目前报道最好的方法可以承受CIFAR10上约66%的对抗性例子。在本文中，我们研究了AutoAttack的空间和频域特性，并提出了一种替代防御方案。我们不是强化网络，而是在推理过程中检测敌意攻击，拒绝被操纵的输入。在对频域进行较为简单快速的分析的基础上，介绍了两种不同的检测算法。首先，黑匣子探测器只对输入图像进行操作，在AutoAttack CIFAR10基准上实现了100%的检测准确率，在ImageNet上实现了99.3%的检测准确率，对于epsilon=8/255这两种情况。其次，白盒检测器使用CNN特征地图的分析，导致在相同的基准上的检测率也是100%和98.7%。



## **4. AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization**

AICAttack：基于注意力优化的对抗性图像字幕攻击 cs.CV

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.11940v2) [paper-pdf](http://arxiv.org/pdf/2402.11940v2)

**Authors**: Jiyao Li, Mingze Ni, Yifei Dong, Tianqing Zhu, Wei Liu

**Abstract**: Recent advances in deep learning research have shown remarkable achievements across many tasks in computer vision (CV) and natural language processing (NLP). At the intersection of CV and NLP is the problem of image captioning, where the related models' robustness against adversarial attacks has not been well studied. In this paper, we present a novel adversarial attack strategy, which we call AICAttack (Attention-based Image Captioning Attack), designed to attack image captioning models through subtle perturbations on images. Operating within a black-box attack scenario, our algorithm requires no access to the target model's architecture, parameters, or gradient information. We introduce an attention-based candidate selection mechanism that identifies the optimal pixels to attack, followed by Differential Evolution (DE) for perturbing pixels' RGB values. We demonstrate AICAttack's effectiveness through extensive experiments on benchmark datasets with multiple victim models. The experimental results demonstrate that our method surpasses current leading-edge techniques by effectively distributing the alignment and semantics of words in the output.

摘要: 近年来，深度学习研究在计算机视觉(CV)和自然语言处理(NLP)领域取得了令人瞩目的成就。在CV和NLP的交叉点是图像字幕问题，相关模型对敌意攻击的稳健性还没有得到很好的研究。本文提出了一种新的对抗性攻击策略AICAttack(基于注意力的图像字幕攻击)，旨在通过对图像的细微扰动来攻击图像字幕模型。我们的算法在黑盒攻击场景中运行，不需要访问目标模型的体系结构、参数或梯度信息。我们引入了一种基于注意力的候选选择机制来确定要攻击的最佳像素，然后采用差分进化(DE)来扰动像素的RGB值。通过在具有多个受害者模型的基准数据集上的大量实验，我们展示了AICAttack的有效性。实验结果表明，该方法能够有效地将词的对齐和语义分布在输出结果中，超过了当前的前沿技术。



## **5. Bounding Reconstruction Attack Success of Adversaries Without Data Priors**

无数据先验对手的边界重构攻击成功 cs.LG

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.12861v1) [paper-pdf](http://arxiv.org/pdf/2402.12861v1)

**Authors**: Alexander Ziller, Anneliese Riess, Kristian Schwethelm, Tamara T. Mueller, Daniel Rueckert, Georgios Kaissis

**Abstract**: Reconstruction attacks on machine learning (ML) models pose a strong risk of leakage of sensitive data. In specific contexts, an adversary can (almost) perfectly reconstruct training data samples from a trained model using the model's gradients. When training ML models with differential privacy (DP), formal upper bounds on the success of such reconstruction attacks can be provided. So far, these bounds have been formulated under worst-case assumptions that might not hold high realistic practicality. In this work, we provide formal upper bounds on reconstruction success under realistic adversarial settings against ML models trained with DP and support these bounds with empirical results. With this, we show that in realistic scenarios, (a) the expected reconstruction success can be bounded appropriately in different contexts and by different metrics, which (b) allows for a more educated choice of a privacy parameter.

摘要: 针对机器学习(ML)模型的重构攻击具有很强的敏感数据泄露风险。在特定的环境中，对手可以(几乎)使用模型的梯度从训练的模型中完美地重建训练数据样本。当训练具有差异隐私(DP)的ML模型时，可以提供此类重构攻击成功的形式上界。到目前为止，这些界限是在最糟糕的假设下制定的，这些假设可能不具有很高的现实实用性。在这项工作中，我们提供了在现实对抗性环境下，针对使用DP训练的ML模型的重建成功的形式上界，并用经验结果支持这些上界。在此基础上，我们证明了在现实场景中，(A)期望的重建成功可以在不同的上下文和不同的度量下适当地限定，这(B)允许更明智地选择隐私参数。



## **6. SWAP: Sparse Entropic Wasserstein Regression for Robust Network Pruning**

SWAP：用于稳健网络剪枝的稀疏熵Wasserstein回归 cs.AI

Published as a conference paper at ICLR 2024

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2310.04918v4) [paper-pdf](http://arxiv.org/pdf/2310.04918v4)

**Authors**: Lei You, Hei Victor Cheng

**Abstract**: This study addresses the challenge of inaccurate gradients in computing the empirical Fisher Information Matrix during neural network pruning. We introduce SWAP, a formulation of Entropic Wasserstein regression (EWR) for pruning, capitalizing on the geometric properties of the optimal transport problem. The ``swap'' of the commonly used linear regression with the EWR in optimization is analytically demonstrated to offer noise mitigation effects by incorporating neighborhood interpolation across data points with only marginal additional computational cost. The unique strength of SWAP is its intrinsic ability to balance noise reduction and covariance information preservation effectively. Extensive experiments performed on various networks and datasets show comparable performance of SWAP with state-of-the-art (SoTA) network pruning algorithms. Our proposed method outperforms the SoTA when the network size or the target sparsity is large, the gain is even larger with the existence of noisy gradients, possibly from noisy data, analog memory, or adversarial attacks. Notably, our proposed method achieves a gain of 6% improvement in accuracy and 8% improvement in testing loss for MobileNetV1 with less than one-fourth of the network parameters remaining.

摘要: 这项研究解决了在神经网络修剪过程中计算经验Fisher信息矩阵时梯度不准确的挑战。利用最优运输问题的几何性质，我们引入了SWAP，这是一种用于修剪的熵Wasserstein回归(EWR)公式。分析表明，在优化中常用的线性回归与EWR的“交换”，通过在仅有边际额外计算成本的情况下结合数据点之间的邻域内插来提供噪声缓解效果。SWAP的独特优势在于其内在的有效平衡噪声抑制和协方差信息保存的能力。在不同网络和数据集上进行的大量实验表明，交换算法的性能与最新的网络剪枝算法(SOTA)相当。当网络规模或目标稀疏度较大时，我们提出的方法的性能优于SOTA，当存在噪声梯度时，增益甚至更大，可能来自噪声数据、模拟记忆或敌对攻击。值得注意的是，我们提出的方法在剩余不到四分之一的网络参数的情况下，对MobileNetV1实现了6%的准确率提高和8%的测试损失改善。



## **7. Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems**

理解和缓解Vec2Text对密集检索系统的威胁 cs.IR

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.12784v1) [paper-pdf](http://arxiv.org/pdf/2402.12784v1)

**Authors**: Shengyao Zhuang, Bevan Koopman, Xiaoran Chu, Guido Zuccon

**Abstract**: The introduction of Vec2Text, a technique for inverting text embeddings, has raised serious privacy concerns within dense retrieval systems utilizing text embeddings, including those provided by OpenAI and Cohere. This threat comes from the ability for a malicious attacker with access to text embeddings to reconstruct the original text.   In this paper, we investigate various aspects of embedding models that could influence the recoverability of text using Vec2Text. Our exploration involves factors such as distance metrics, pooling functions, bottleneck pre-training, training with noise addition, embedding quantization, and embedding dimensions -- aspects not previously addressed in the original Vec2Text paper. Through a thorough analysis of these factors, our aim is to gain a deeper understanding of the critical elements impacting the trade-offs between text recoverability and retrieval effectiveness in dense retrieval systems. This analysis provides valuable insights for practitioners involved in designing privacy-aware dense retrieval systems. Additionally, we propose a straightforward fix for embedding transformation that ensures equal ranking effectiveness while mitigating the risk of text recoverability.   Furthermore, we extend the application of Vec2Text to the separate task of corpus poisoning, where, theoretically, Vec2Text presents a more potent threat compared to previous attack methods. Notably, Vec2Text does not require access to the dense retriever's model parameters and can efficiently generate numerous adversarial passages.   In summary, this study highlights the potential threat posed by Vec2Text to existing dense retrieval systems, while also presenting effective methods to patch and strengthen such systems against such risks.

摘要: Vec2Text是一种用于倒排文本嵌入的技术，它的引入在使用文本嵌入的密集检索系统中引发了严重的隐私问题，包括OpenAI和Cohere提供的那些。这一威胁来自于恶意攻击者能够访问文本嵌入来重建原始文本。在本文中，我们使用Vec2Text研究了影响文本可恢复性的嵌入模型的各个方面。我们的探索涉及诸如距离度量、池化函数、瓶颈预训练、添加噪声的训练、嵌入量化和嵌入维度等因素--这些方面在最初的Vec2Text论文中没有涉及。通过对这些因素的深入分析，我们的目的是更深入地了解在密集检索系统中影响文本可恢复性和检索效率之间权衡的关键因素。这一分析为参与设计隐私感知密集检索系统的从业者提供了有价值的见解。此外，我们还提出了一种简单明了的嵌入转换的解决方案，它可以确保同等的排名效率，同时降低文本可恢复性的风险。此外，我们将Vec2Text的应用扩展到语料库中毒的单独任务中，理论上，Vec2Text比以前的攻击方法具有更强大的威胁。值得注意的是，Vec2Text不需要访问密集检索器的模型参数，并且可以高效地生成大量对抗性段落。总之，这项研究强调了Vec2Text对现有密集检索系统构成的潜在威胁，同时也提出了有效的方法来修补和加强此类系统以应对此类风险。



## **8. Revisiting the Information Capacity of Neural Network Watermarks: Upper Bound Estimation and Beyond**

重新审视神经网络水印的信息容量：上界估计和超越 cs.CR

Accepted by AAAI 2024

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.12720v1) [paper-pdf](http://arxiv.org/pdf/2402.12720v1)

**Authors**: Fangqi Li, Haodong Zhao, Wei Du, Shilin Wang

**Abstract**: To trace the copyright of deep neural networks, an owner can embed its identity information into its model as a watermark. The capacity of the watermark quantify the maximal volume of information that can be verified from the watermarked model. Current studies on capacity focus on the ownership verification accuracy under ordinary removal attacks and fail to capture the relationship between robustness and fidelity. This paper studies the capacity of deep neural network watermarks from an information theoretical perspective. We propose a new definition of deep neural network watermark capacity analogous to channel capacity, analyze its properties, and design an algorithm that yields a tight estimation of its upper bound under adversarial overwriting. We also propose a universal non-invasive method to secure the transmission of the identity message beyond capacity by multiple rounds of ownership verification. Our observations provide evidence for neural network owners and defenders that are curious about the tradeoff between the integrity of their ownership and the performance degradation of their products.

摘要: 为了追踪深度神经网络的版权，所有者可以将其身份信息作为水印嵌入到其模型中。水印的容量量化了可以从水印模型中验证的最大信息量。目前对容量的研究主要集中在普通删除攻击下的所有权验证准确性，而没有捕捉到健壮性和保真度之间的关系。从信息论的角度对深度神经网络水印的容量进行了研究。提出了一种类似于信道容量的深度神经网络水印容量的新定义，分析了它的性质，并设计了一种算法，在对抗覆盖的情况下给出了它的上界的严格估计。我们还提出了一种通用的非侵入性方法，通过多轮所有权验证来保护超出容量的身份消息的传输。我们的观察为神经网络所有者和捍卫者提供了证据，他们对其所有权的完整性和其产品的性能降级之间的权衡感到好奇。



## **9. Beyond Worst-case Attacks: Robust RL with Adaptive Defense via Non-dominated Policies**

超越最坏情况的攻击：强大的RL，通过非支配策略进行自适应防御 cs.LG

International Conference on Learning Representations (ICLR) 2024,  spotlight

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.12673v1) [paper-pdf](http://arxiv.org/pdf/2402.12673v1)

**Authors**: Xiangyu Liu, Chenghao Deng, Yanchao Sun, Yongyuan Liang, Furong Huang

**Abstract**: In light of the burgeoning success of reinforcement learning (RL) in diverse real-world applications, considerable focus has been directed towards ensuring RL policies are robust to adversarial attacks during test time. Current approaches largely revolve around solving a minimax problem to prepare for potential worst-case scenarios. While effective against strong attacks, these methods often compromise performance in the absence of attacks or the presence of only weak attacks. To address this, we study policy robustness under the well-accepted state-adversarial attack model, extending our focus beyond only worst-case attacks. We first formalize this task at test time as a regret minimization problem and establish its intrinsic hardness in achieving sublinear regret when the baseline policy is from a general continuous policy class, $\Pi$. This finding prompts us to \textit{refine} the baseline policy class $\Pi$ prior to test time, aiming for efficient adaptation within a finite policy class $\Tilde{\Pi}$, which can resort to an adversarial bandit subroutine. In light of the importance of a small, finite $\Tilde{\Pi}$, we propose a novel training-time algorithm to iteratively discover \textit{non-dominated policies}, forming a near-optimal and minimal $\Tilde{\Pi}$, thereby ensuring both robustness and test-time efficiency. Empirical validation on the Mujoco corroborates the superiority of our approach in terms of natural and robust performance, as well as adaptability to various attack scenarios.

摘要: 鉴于强化学习(RL)在各种现实世界应用中的迅速成功，人们已经将相当大的注意力集中在确保RL策略在测试期间对对手攻击是健壮的。目前的方法主要围绕着解决极小极大问题来为潜在的最坏情况做准备。虽然这些方法对强攻击有效，但在没有攻击或仅存在弱攻击的情况下，这些方法往往会损害性能。为了解决这一问题，我们研究了广为接受的状态对抗攻击模型下的策略健壮性，将我们的重点扩展到仅最坏情况下的攻击。我们首先在测试时将这一任务形式化为后悔最小化问题，并建立了当基线策略来自一般连续策略类$\PI$时实现次线性后悔的内在难度。这一发现促使我们在测试时间之前对基线策略类$\pI$进行改进，目标是在有限的策略类$\tilde{\pI}$内进行有效的适应，这可以求助于对抗性的Bandit子例程。考虑到小的、有限的Tilde{PI}的重要性，我们提出了一种新的训练时间算法来迭代地发现文本{非支配策略}，从而形成一个接近最优且最小的Tilde{PI}，从而保证了健壮性和测试时间的效率。在Mujoco上的经验验证证实了我们的方法在自然和健壮的性能以及对各种攻击场景的适应性方面的优越性。



## **10. Citadel: Enclaves with Microarchitectural Isolation and Secure Shared Memory on a Speculative Out-of-Order Processor**

Citadel：推测乱序处理器上具有微体系结构隔离和安全共享内存的飞地 cs.CR

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2306.14882v3) [paper-pdf](http://arxiv.org/pdf/2306.14882v3)

**Authors**: Jules Drean, Miguel Gomez-Garcia, Fisher Jepsen, Thomas Bourgeat, Srinivas Devadas

**Abstract**: Enclaves or Trusted Execution Environments are trusted-hardware primitives that make it possible to isolate and protect a sensitive program from an untrusted operating system. Unfortunately, almost all existing enclave platforms are vulnerable to microarchitectural side channels and transient execution attacks, and the one academic proposal that is not does not allow programs to interact with the outside world. We present Citadel, to our knowledge, the first enclave platform with microarchitectural isolation to run realistic secure programs on a speculative out-of-order multicore processor. We show how to leverage hardware/software co-design to enable shared memory between an enclave and an untrusted operating system while preventing speculative transmitters between the enclave and a potential adversary. We then evaluate our secure baseline and present further mechanisms to achieve reasonable performance for out-of-the-box programs. Our multicore processor runs on an FPGA and boots untrusted Linux from which users can securely launch and interact with enclaves. To demonstrate our platform capabilities, we run a private inference enclave that embed a small neural network trained on MNIST. A remote user can remotely attest the enclave integrity, perform key exchange and send encrypted input for secure evaluation. We open-source our end-to-end hardware and software infrastructure, hoping to spark more research and bridge the gap between conceptual proposals and FPGA prototypes.

摘要: 飞地或受信任的执行环境是受信任的硬件原语，它使隔离和保护敏感程序免受不受信任的操作系统的攻击成为可能。不幸的是，几乎所有现有的Enclave平台都容易受到微体系结构侧通道和瞬时执行攻击的攻击，而唯一不允许程序与外部世界交互的学术建议是不允许的。据我们所知，Citadel是第一个具有微体系结构隔离的Enclave平台，可以在推测无序的多核处理器上运行现实的安全程序。我们展示了如何利用硬件/软件协同设计来实现飞地和不可信操作系统之间的共享内存，同时防止飞地和潜在对手之间的投机性传输。然后，我们评估我们的安全基准，并提出进一步的机制，以实现开箱即用程序的合理性能。我们的多核处理器在FPGA上运行，并启动不受信任的Linux，用户可以从该Linux安全地启动Enclaves并与其交互。为了展示我们的平台能力，我们运行了一个私人推理飞地，其中嵌入了一个针对MNIST训练的小型神经网络。远程用户可以远程证明飞地的完整性、执行密钥交换并发送加密输入以进行安全评估。我们将我们的端到端硬件和软件基础设施开源，希望引发更多研究，并弥合概念提案和FPGA原型之间的差距。



## **11. Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!**

模拟失调：大型语言模型的安全校准可能会适得其反！ cs.CL

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12343v1) [paper-pdf](http://arxiv.org/pdf/2402.12343v1)

**Authors**: Zhanhui Zhou, Jie Liu, Zhichen Dong, Jiaheng Liu, Chao Yang, Wanli Ouyang, Yu Qiao

**Abstract**: Large language models (LLMs) need to undergo safety alignment to ensure safe conversations with humans. However, in this work, we introduce an inference-time attack framework, demonstrating that safety alignment can also unintentionally facilitate harmful outcomes under adversarial manipulation. This framework, named Emulated Disalignment (ED), adversely combines a pair of open-source pre-trained and safety-aligned language models in the output space to produce a harmful language model without any training. Our experiments with ED across three datasets and four model families (Llama-1, Llama-2, Mistral, and Alpaca) show that ED doubles the harmfulness of pre-trained models and outperforms strong baselines, achieving the highest harmful rate in 43 out of 48 evaluation subsets by a large margin. Crucially, our findings highlight the importance of reevaluating the practice of open-sourcing language models even after safety alignment.

摘要: 大型语言模型(LLM)需要经过安全调整，以确保与人类的安全对话。然而，在这项工作中，我们引入了一个推理时间攻击框架，证明了安全对齐也可以在无意中促进对抗性操纵下的有害结果。这个名为仿真失调(ED)的框架在输出空间中反向组合了两个开放源码的预训练和安全对齐的语言模型，在没有任何训练的情况下产生了有害的语言模型。我们在三个数据集和四个模型家族(骆驼-1、骆驼-2、米斯特拉尔和羊驼)上使用ED进行的实验表明，ED的危害性是预训练模型的两倍，并且性能优于强基线，在48个评估子集中的43个子集上获得了最高的伤害率。至关重要的是，我们的发现强调了重新评估开源语言模型实践的重要性，即使在安全调整之后也是如此。



## **12. An Adversarial Approach to Evaluating the Robustness of Event Identification Models**

事件识别模型稳健性评估的对抗性方法 eess.SY

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12338v1) [paper-pdf](http://arxiv.org/pdf/2402.12338v1)

**Authors**: Obai Bahwal, Oliver Kosut, Lalitha Sankar

**Abstract**: Intelligent machine learning approaches are finding active use for event detection and identification that allow real-time situational awareness. Yet, such machine learning algorithms have been shown to be susceptible to adversarial attacks on the incoming telemetry data. This paper considers a physics-based modal decomposition method to extract features for event classification and focuses on interpretable classifiers including logistic regression and gradient boosting to distinguish two types of events: load loss and generation loss. The resulting classifiers are then tested against an adversarial algorithm to evaluate their robustness. The adversarial attack is tested in two settings: the white box setting, wherein the attacker knows exactly the classification model; and the gray box setting, wherein the attacker has access to historical data from the same network as was used to train the classifier, but does not know the classification model. Thorough experiments on the synthetic South Carolina 500-bus system highlight that a relatively simpler model such as logistic regression is more susceptible to adversarial attacks than gradient boosting.

摘要: 智能机器学习方法正在积极应用于事件检测和识别，从而实现实时的态势感知。然而，这种机器学习算法已被证明容易受到对传入遥测数据的敌意攻击。本文考虑了一种基于物理的模式分解方法来提取事件分类的特征，并重点使用Logistic回归和梯度提升等可解释分类器来区分两种类型的事件：负荷损失和发电损失。然后，将得到的分类器与对抗性算法进行测试，以评估它们的稳健性。在两种设置中测试对抗性攻击：白盒设置，其中攻击者确切地知道分类模型；以及灰盒设置，其中攻击者可以访问来自用于训练分类器的相同网络的历史数据，但不知道分类模型。在合成的南卡罗来纳州500母线系统上进行的彻底实验表明，相对简单的模型，如Logistic回归，比梯度助推更容易受到对抗性攻击。



## **13. Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models**

Robust CLIP：用于强健大型视觉语言模型的视觉嵌入的无监督对抗性微调 cs.LG

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12336v1) [paper-pdf](http://arxiv.org/pdf/2402.12336v1)

**Authors**: Christian Schlarmann, Naman Deep Singh, Francesco Croce, Matthias Hein

**Abstract**: Multi-modal foundation models like OpenFlamingo, LLaVA, and GPT-4 are increasingly used for various real-world tasks. Prior work has shown that these models are highly vulnerable to adversarial attacks on the vision modality. These attacks can be leveraged to spread fake information or defraud users, and thus pose a significant risk, which makes the robustness of large multi-modal foundation models a pressing problem. The CLIP model, or one of its variants, is used as a frozen vision encoder in many vision-language models (VLMs), e.g. LLaVA and OpenFlamingo. We propose an unsupervised adversarial fine-tuning scheme to obtain a robust CLIP vision encoder, which yields robustness on all vision down-stream tasks (VLMs, zero-shot classification) that rely on CLIP. In particular, we show that stealth-attacks on users of VLMs by a malicious third party providing manipulated images are no longer possible once one replaces the original CLIP model with our robust one. No retraining or fine-tuning of the VLM is required. The code and robust models are available at https://github.com/chs20/RobustVLM

摘要: OpenFlamingo、LLaVA和GPT-4等多模态基础模型越来越多地用于各种现实任务。先前的工作表明，这些模型非常容易受到视觉模态的对抗性攻击。这些攻击可以被用来传播虚假信息或欺骗用户，从而构成重大风险，这使得大型多模态基础模型的鲁棒性成为一个紧迫的问题。CLIP模型或其变体之一在许多视觉语言模型（VLM）中用作冻结视觉编码器，例如LLaVA和OpenFlamingo。我们提出了一个无监督的对抗微调方案，以获得一个强大的CLIP视觉编码器，它产生依赖于CLIP的所有视觉下游任务（VLMs，零拍摄分类）的鲁棒性。特别是，我们表明，一旦用我们强大的CLIP模型替换了原始CLIP模型，恶意第三方提供操纵图像对VLM用户的隐形攻击就不再可能了。无需对VLM进行再培训或微调。代码和鲁棒模型可在https://github.com/chs20/RobustVLM上获得



## **14. Query-Based Adversarial Prompt Generation**

基于查询的对抗性提示生成 cs.CL

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12329v1) [paper-pdf](http://arxiv.org/pdf/2402.12329v1)

**Authors**: Jonathan Hayase, Ema Borevkovic, Nicholas Carlini, Florian Tramèr, Milad Nasr

**Abstract**: Recent work has shown it is possible to construct adversarial examples that cause an aligned language model to emit harmful strings or perform harmful behavior. Existing attacks work either in the white-box setting (with full access to the model weights), or through transferability: the phenomenon that adversarial examples crafted on one model often remain effective on other models. We improve on prior work with a query-based attack that leverages API access to a remote language model to construct adversarial examples that cause the model to emit harmful strings with (much) higher probability than with transfer-only attacks. We validate our attack on GPT-3.5 and OpenAI's safety classifier; we can cause GPT-3.5 to emit harmful strings that current transfer attacks fail at, and we can evade the safety classifier with nearly 100% probability.

摘要: 最近的研究表明，有可能构建对抗性的例子，导致对齐的语言模型发出有害的字符串或执行有害的行为。现有攻击要么在白盒设置(完全访问模型权重)下工作，要么通过可转移性工作：在一个模型上制作的对抗性示例通常在其他模型上仍然有效的现象。我们改进了以前的基于查询的攻击，该攻击利用对远程语言模型的API访问来构建敌意示例，这些示例导致该模型以(远远)比仅传输攻击更高的概率发出有害字符串。我们在GPT-3.5和OpenAI的安全分类器上验证了我们的攻击，我们可以使GPT-3.5发出当前传输攻击失败的有害字符串，并且我们可以近100%的概率避开安全分类器。



## **15. Attacks on Node Attributes in Graph Neural Networks**

图神经网络中节点属性的攻击 cs.SI

Accepted to AAAI 2024 AICS workshop

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12426v1) [paper-pdf](http://arxiv.org/pdf/2402.12426v1)

**Authors**: Ying Xu, Michael Lanier, Anindya Sarkar, Yevgeniy Vorobeychik

**Abstract**: Graphs are commonly used to model complex networks prevalent in modern social media and literacy applications. Our research investigates the vulnerability of these graphs through the application of feature based adversarial attacks, focusing on both decision-time attacks and poisoning attacks. In contrast to state-of-the-art models like Net Attack and Meta Attack, which target node attributes and graph structure, our study specifically targets node attributes. For our analysis, we utilized the text dataset Hellaswag and graph datasets Cora and CiteSeer, providing a diverse basis for evaluation. Our findings indicate that decision-time attacks using Projected Gradient Descent (PGD) are more potent compared to poisoning attacks that employ Mean Node Embeddings and Graph Contrastive Learning strategies. This provides insights for graph data security, pinpointing where graph-based models are most vulnerable and thereby informing the development of stronger defense mechanisms against such attacks.

摘要: 图通常用于对现代社交媒体和识字应用中普遍存在的复杂网络进行建模。我们的研究通过应用基于特征的对抗性攻击来研究这些图的脆弱性，重点针对决策时间攻击和中毒攻击。与网络攻击和元攻击等以节点属性和图结构为目标的最新模型不同，我们的研究专门针对节点属性。在我们的分析中，我们使用了文本数据集Hellaswag和图形数据集Cora和CiteSeer，为评估提供了多样化的基础。我们的发现表明，与使用均值节点嵌入和图对比学习策略的中毒攻击相比，使用投影梯度下降(PGD)的决策时间攻击更有效。这为图形数据安全提供了洞察力，准确地指出了基于图形的模型最易受攻击的位置，从而为开发针对此类攻击的更强大的防御机制提供了信息。



## **16. Can AI-Generated Text be Reliably Detected?**

能否可靠地检测到人工智能生成的文本？ cs.CL

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2303.11156v3) [paper-pdf](http://arxiv.org/pdf/2303.11156v3)

**Authors**: Vinu Sankar Sadasivan, Aounon Kumar, Sriram Balasubramanian, Wenxiao Wang, Soheil Feizi

**Abstract**: The unregulated use of LLMs can potentially lead to malicious consequences such as plagiarism, generating fake news, spamming, etc. Therefore, reliable detection of AI-generated text can be critical to ensure the responsible use of LLMs. Recent works attempt to tackle this problem either using certain model signatures present in the generated text outputs or by applying watermarking techniques that imprint specific patterns onto them. In this paper, we show that these detectors are not reliable in practical scenarios. In particular, we develop a recursive paraphrasing attack to apply on AI text, which can break a whole range of detectors, including the ones using the watermarking schemes as well as neural network-based detectors, zero-shot classifiers, and retrieval-based detectors. Our experiments include passages around 300 tokens in length, showing the sensitivity of the detectors even in the case of relatively long passages. We also observe that our recursive paraphrasing only degrades text quality slightly, measured via human studies, and metrics such as perplexity scores and accuracy on text benchmarks. Additionally, we show that even LLMs protected by watermarking schemes can be vulnerable against spoofing attacks aimed to mislead detectors to classify human-written text as AI-generated, potentially causing reputational damages to the developers. In particular, we show that an adversary can infer hidden AI text signatures of the LLM outputs without having white-box access to the detection method. Finally, we provide a theoretical connection between the AUROC of the best possible detector and the Total Variation distance between human and AI text distributions that can be used to study the fundamental hardness of the reliable detection problem for advanced language models. Our code is publicly available at https://github.com/vinusankars/Reliability-of-AI-text-detectors.

摘要: LLM的不受监管的使用可能会导致恶意后果，如剽窃，生成假新闻，垃圾邮件等，因此，对AI生成的文本进行可靠检测对于确保负责任地使用LLM至关重要。最近的作品试图解决这个问题，要么使用某些模型签名中存在的生成的文本输出，或通过应用水印技术，印记特定的模式到他们。在本文中，我们表明，这些检测器在实际情况下是不可靠的。特别是，我们开发了一种递归释义攻击来应用于AI文本，它可以破坏整个范围的检测器，包括使用水印方案以及基于神经网络的检测器，零触发分类器和基于检索的检测器。我们的实验包括长度约300个标记的通道，即使在相对较长的通道的情况下，也显示了检测器的灵敏度。我们还观察到，我们的递归释义只会轻微降低文本质量，通过人类研究和指标，如困惑分数和文本基准的准确性。此外，我们还发现，即使是受水印方案保护的LLM也容易受到欺骗攻击，这些攻击旨在误导检测器将人类编写的文本分类为AI生成的文本，从而可能对开发人员造成声誉损害。特别是，我们证明了对手可以推断出LLM输出的隐藏AI文本签名，而无需白盒访问检测方法。最后，我们提供了最佳检测器的AUROC与人类和AI文本分布之间的总变差距离之间的理论联系，可用于研究高级语言模型可靠检测问题的基本难度。我们的代码可在https://github.com/vinusankars/Reliability-of-AI-text-detectors上公开获取。



## **17. On the Byzantine-Resilience of Distillation-Based Federated Learning**

基于蒸馏的联邦学习的拜占庭弹性研究 cs.LG

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12265v1) [paper-pdf](http://arxiv.org/pdf/2402.12265v1)

**Authors**: Christophe Roux, Max Zimmer, Sebastian Pokutta

**Abstract**: Federated Learning (FL) algorithms using Knowledge Distillation (KD) have received increasing attention due to their favorable properties with respect to privacy, non-i.i.d. data and communication cost. These methods depart from transmitting model parameters and, instead, communicate information about a learning task by sharing predictions on a public dataset. In this work, we study the performance of such approaches in the byzantine setting, where a subset of the clients act in an adversarial manner aiming to disrupt the learning process. We show that KD-based FL algorithms are remarkably resilient and analyze how byzantine clients can influence the learning process compared to Federated Averaging. Based on these insights, we introduce two new byzantine attacks and demonstrate that they are effective against prior byzantine-resilient methods. Additionally, we propose FilterExp, a novel method designed to enhance the byzantine resilience of KD-based FL algorithms and demonstrate its efficacy. Finally, we provide a general method to make attacks harder to detect, improving their effectiveness.

摘要: 基于知识蒸馏(KD)的联合学习(FL)算法因其在隐私、非I.I.D.等方面的良好特性而受到越来越多的关注。数据和通信成本。这些方法不同于传输模型参数，而是通过共享公共数据集上的预测来传递关于学习任务的信息。在这项工作中，我们研究了这些方法在拜占庭环境下的性能，在拜占庭环境中，客户的子集以对抗性的方式行动，旨在扰乱学习过程。我们证明了基于KD的FL算法具有显著的弹性，并分析了与联合平均相比，拜占庭客户端如何影响学习过程。基于这些见解，我们引入了两个新的拜占庭攻击，并证明了它们对以前的拜占庭弹性方法是有效的。此外，我们还提出了一种新的方法FilterExp，该方法旨在增强基于KD的FL算法的拜占庭恢复能力，并证明了其有效性。最后，我们提供了一种使攻击更难检测的通用方法，从而提高了攻击的有效性。



## **18. Amplifying Training Data Exposure through Fine-Tuning with Pseudo-Labeled Memberships**

通过伪标记模型的微调来扩大训练数据的暴露 cs.CL

20 pages, 6 figures, 15 tables

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12189v1) [paper-pdf](http://arxiv.org/pdf/2402.12189v1)

**Authors**: Myung Gyo Oh, Hong Eun Ahn, Leo Hyun Park, Taekyoung Kwon

**Abstract**: Neural language models (LMs) are vulnerable to training data extraction attacks due to data memorization. This paper introduces a novel attack scenario wherein an attacker adversarially fine-tunes pre-trained LMs to amplify the exposure of the original training data. This strategy differs from prior studies by aiming to intensify the LM's retention of its pre-training dataset. To achieve this, the attacker needs to collect generated texts that are closely aligned with the pre-training data. However, without knowledge of the actual dataset, quantifying the amount of pre-training data within generated texts is challenging. To address this, we propose the use of pseudo-labels for these generated texts, leveraging membership approximations indicated by machine-generated probabilities from the target LM. We subsequently fine-tune the LM to favor generations with higher likelihoods of originating from the pre-training data, based on their membership probabilities. Our empirical findings indicate a remarkable outcome: LMs with over 1B parameters exhibit a four to eight-fold increase in training data exposure. We discuss potential mitigations and suggest future research directions.

摘要: 神经语言模型(LMS)由于数据的记忆，容易受到训练数据提取攻击。本文介绍了一种新的攻击场景，其中攻击者对抗性地微调预先训练的最小均方根来放大原始训练数据的暴露。这一策略与以前的研究不同，目的是加强LM对其训练前数据集的保留。为了实现这一点，攻击者需要收集与预训练数据紧密一致的生成文本。然而，在没有实际数据集的情况下，量化生成文本中的预训练数据量是具有挑战性的。为了解决这个问题，我们建议对这些生成的文本使用伪标签，利用来自目标LM的机器生成的概率所指示的隶属度近似。随后，我们基于成员概率对LM进行微调，以支持来自预培训数据的可能性更高的世代。我们的经验发现表明了一个显著的结果：参数超过1B的最小二乘法显示出训练数据暴露的四到八倍的增加。我们讨论了潜在的缓解措施，并提出了未来的研究方向。



## **19. Adversarial Feature Alignment: Balancing Robustness and Accuracy in Deep Learning via Adversarial Training**

对抗性特征对齐：通过对抗性训练在深度学习中平衡稳健性和准确性 cs.CV

19 pages, 5 figures, 16 tables, 2 algorithms

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12187v1) [paper-pdf](http://arxiv.org/pdf/2402.12187v1)

**Authors**: Leo Hyun Park, Jaeuk Kim, Myung Gyo Oh, Jaewoo Park, Taekyoung Kwon

**Abstract**: Deep learning models continue to advance in accuracy, yet they remain vulnerable to adversarial attacks, which often lead to the misclassification of adversarial examples. Adversarial training is used to mitigate this problem by increasing robustness against these attacks. However, this approach typically reduces a model's standard accuracy on clean, non-adversarial samples. The necessity for deep learning models to balance both robustness and accuracy for security is obvious, but achieving this balance remains challenging, and the underlying reasons are yet to be clarified. This paper proposes a novel adversarial training method called Adversarial Feature Alignment (AFA), to address these problems. Our research unveils an intriguing insight: misalignment within the feature space often leads to misclassification, regardless of whether the samples are benign or adversarial. AFA mitigates this risk by employing a novel optimization algorithm based on contrastive learning to alleviate potential feature misalignment. Through our evaluations, we demonstrate the superior performance of AFA. The baseline AFA delivers higher robust accuracy than previous adversarial contrastive learning methods while minimizing the drop in clean accuracy to 1.86% and 8.91% on CIFAR10 and CIFAR100, respectively, in comparison to cross-entropy. We also show that joint optimization of AFA and TRADES, accompanied by data augmentation using a recent diffusion model, achieves state-of-the-art accuracy and robustness.

摘要: 深度学习模型的准确性不断提高，但它们仍然容易受到对抗性攻击，这往往会导致对抗性例子的错误分类。对抗性训练被用来通过增加对这些攻击的健壮性来缓解这个问题。然而，这种方法通常会降低模型在干净、非对抗性样本上的标准精度。深度学习模型显然有必要平衡安全性的稳健性和准确性，但实现这一平衡仍然具有挑战性，其根本原因尚待澄清。针对这些问题，提出了一种新的对抗性训练方法--对抗性特征对齐(AFA)。我们的研究揭示了一个耐人寻味的见解：无论样本是良性的还是对抗性的，特征空间内的未对齐往往会导致错误分类。AFA通过采用一种新的基于对比学习的优化算法来缓解潜在的特征未对齐，从而降低了这种风险。通过我们的评估，我们证明了AFA的优越性能。与以前的对抗性对比学习方法相比，基准AFA提供了更高的鲁棒性准确率，同时将CIFAR10和CIFAR100上的清洁准确率下降最小，分别为1.86%和8.91%。我们还表明，AFA和TRADS的联合优化，伴随着使用最近的扩散模型的数据增强，实现了最先进的准确性和稳健性。



## **20. Dynamic Graph Information Bottleneck**

动态图信息瓶颈 cs.LG

Accepted by the research tracks of The Web Conference 2024 (WWW 2024)

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.06716v2) [paper-pdf](http://arxiv.org/pdf/2402.06716v2)

**Authors**: Haonan Yuan, Qingyun Sun, Xingcheng Fu, Cheng Ji, Jianxin Li

**Abstract**: Dynamic Graphs widely exist in the real world, which carry complicated spatial and temporal feature patterns, challenging their representation learning. Dynamic Graph Neural Networks (DGNNs) have shown impressive predictive abilities by exploiting the intrinsic dynamics. However, DGNNs exhibit limited robustness, prone to adversarial attacks. This paper presents the novel Dynamic Graph Information Bottleneck (DGIB) framework to learn robust and discriminative representations. Leveraged by the Information Bottleneck (IB) principle, we first propose the expected optimal representations should satisfy the Minimal-Sufficient-Consensual (MSC) Condition. To compress redundant as well as conserve meritorious information into latent representation, DGIB iteratively directs and refines the structural and feature information flow passing through graph snapshots. To meet the MSC Condition, we decompose the overall IB objectives into DGIB$_{MS}$ and DGIB$_C$, in which the DGIB$_{MS}$ channel aims to learn the minimal and sufficient representations, with the DGIB$_{MS}$ channel guarantees the predictive consensus. Extensive experiments on real-world and synthetic dynamic graph datasets demonstrate the superior robustness of DGIB against adversarial attacks compared with state-of-the-art baselines in the link prediction task. To the best of our knowledge, DGIB is the first work to learn robust representations of dynamic graphs grounded in the information-theoretic IB principle.

摘要: 动态图形广泛存在于现实世界中，携带着复杂的时空特征模式，对其表示学习提出了挑战。动态图神经网络(DGNN)利用其内在的动力学特性，表现出了令人印象深刻的预测能力。然而，DGNN表现出有限的健壮性，容易受到对抗性攻击。本文提出了一种新的动态图信息瓶颈(DGIB)框架来学习稳健和区分的表示。利用信息瓶颈(IB)原理，我们首先提出期望的最优表示应该满足最小-充分-一致(MSC)条件。为了压缩冗余信息并将有价值的信息保存到潜在表示中，DGIB迭代地指导和细化通过图快照传递的结构和特征信息流。为了满足MSC条件，我们将整体IB目标分解为DGIB${MS}$和DGIB$_C$，其中DGIB${MS}$通道旨在学习最小且充分的表示，而DGIB${MS}$通道保证预测共识。在真实世界和合成动态图数据集上的大量实验表明，与链接预测任务中的最新基线相比，DGIB对对手攻击具有更好的稳健性。据我们所知，DGIB是第一个基于信息论IB原理学习动态图的稳健表示的工作。



## **21. Self-Guided Robust Graph Structure Refinement**

自引导鲁棒图结构求精 cs.LG

This paper has been accepted by TheWebConf 2024

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.11837v1) [paper-pdf](http://arxiv.org/pdf/2402.11837v1)

**Authors**: Yeonjun In, Kanghoon Yoon, Kibum Kim, Kijung Shin, Chanyoung Park

**Abstract**: Recent studies have revealed that GNNs are vulnerable to adversarial attacks. To defend against such attacks, robust graph structure refinement (GSR) methods aim at minimizing the effect of adversarial edges based on node features, graph structure, or external information. However, we have discovered that existing GSR methods are limited by narrowassumptions, such as assuming clean node features, moderate structural attacks, and the availability of external clean graphs, resulting in the restricted applicability in real-world scenarios. In this paper, we propose a self-guided GSR framework (SG-GSR), which utilizes a clean sub-graph found within the given attacked graph itself. Furthermore, we propose a novel graph augmentation and a group-training strategy to handle the two technical challenges in the clean sub-graph extraction: 1) loss of structural information, and 2) imbalanced node degree distribution. Extensive experiments demonstrate the effectiveness of SG-GSR under various scenarios including non-targeted attacks, targeted attacks, feature attacks, e-commerce fraud, and noisy node labels. Our code is available at https://github.com/yeonjun-in/torch-SG-GSR.

摘要: 最近的研究表明，GNN很容易受到对抗性攻击。为了防御这类攻击，基于节点特征、图结构或外部信息的健壮图结构精化(GSR)方法旨在最小化敌方边的影响。然而，我们发现现有的GSR方法受到狭隘假设的限制，如假设节点特征干净、结构攻击适度、外部干净图可用等，导致其在现实场景中的适用性受到限制。在本文中，我们提出了一种自导GSR框架(SG-GSR)，它利用了在给定的攻击图本身中发现的干净的子图。此外，我们还提出了一种新的图增强和分组训练策略来解决清洁子图提取中的两个技术挑战：1)结构信息的丢失；2)结点度分布的不平衡。大量实验验证了SG-GSR在非定向攻击、定向攻击、特征攻击、电子商务欺诈、节点标签噪声等场景下的有效性。我们的代码可以在https://github.com/yeonjun-in/torch-SG-GSR.上找到



## **22. On the Safety Concerns of Deploying LLMs/VLMs in Robotics: Highlighting the Risks and Vulnerabilities**

在机器人中部署LLMS/VLM的安全问题：突出风险和漏洞 cs.RO

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.10340v2) [paper-pdf](http://arxiv.org/pdf/2402.10340v2)

**Authors**: Xiyang Wu, Ruiqi Xian, Tianrui Guan, Jing Liang, Souradip Chakraborty, Fuxiao Liu, Brian Sadler, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: In this paper, we highlight the critical issues of robustness and safety associated with integrating large language models (LLMs) and vision-language models (VLMs) into robotics applications. Recent works have focused on using LLMs and VLMs to improve the performance of robotics tasks, such as manipulation, navigation, etc. However, such integration can introduce significant vulnerabilities, in terms of their susceptibility to adversarial attacks due to the language models, potentially leading to catastrophic consequences. By examining recent works at the interface of LLMs/VLMs and robotics, we show that it is easy to manipulate or misguide the robot's actions, leading to safety hazards. We define and provide examples of several plausible adversarial attacks, and conduct experiments on three prominent robot frameworks integrated with a language model, including KnowNo VIMA, and Instruct2Act, to assess their susceptibility to these attacks. Our empirical findings reveal a striking vulnerability of LLM/VLM-robot integrated systems: simple adversarial attacks can significantly undermine the effectiveness of LLM/VLM-robot integrated systems. Specifically, our data demonstrate an average performance deterioration of 21.2% under prompt attacks and a more alarming 30.2% under perception attacks. These results underscore the critical need for robust countermeasures to ensure the safe and reliable deployment of the advanced LLM/VLM-based robotic systems.

摘要: 在这篇文章中，我们强调了与将大语言模型(LLM)和视觉语言模型(VLM)集成到机器人应用中相关的健壮性和安全性的关键问题。最近的工作集中在使用LLMS和VLM来提高机器人任务的性能，如操纵、导航等。然而，这种集成可能会引入显著的漏洞，因为它们容易由于语言模型而受到对手攻击，可能会导致灾难性的后果。通过对LLMS/VLMS与机器人接口的最新研究，我们发现很容易操纵或误导机器人的动作，从而导致安全隐患。我们定义并提供了几种可能的对抗性攻击的例子，并在三个与语言模型集成的著名机器人框架上进行了实验，包括KnowNo Vima和Instruct2Act，以评估它们对这些攻击的敏感度。我们的实验结果揭示了LLM/VLM-机器人集成系统的一个显著漏洞：简单的对抗性攻击会显著削弱LLM/VLM-机器人集成系统的有效性。具体地说，我们的数据显示，在即时攻击下，平均性能下降21.2%，在感知攻击下，更令人震惊的是30.2%。这些结果突出表明，迫切需要强有力的对策，以确保安全可靠地部署先进的基于LLM/VLM的机器人系统。



## **23. SAGMAN: Stability Analysis of Graph Neural Networks on the Manifolds**

Sagman：流形上图神经网络的稳定性分析 cs.LG

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.08653v2) [paper-pdf](http://arxiv.org/pdf/2402.08653v2)

**Authors**: Wuxinlin Cheng, Chenhui Deng, Ali Aghdaei, Zhiru Zhang, Zhuo Feng

**Abstract**: Modern graph neural networks (GNNs) can be sensitive to changes in the input graph structure and node features, potentially resulting in unpredictable behavior and degraded performance. In this work, we introduce a spectral framework known as SAGMAN for examining the stability of GNNs. This framework assesses the distance distortions that arise from the nonlinear mappings of GNNs between the input and output manifolds: when two nearby nodes on the input manifold are mapped (through a GNN model) to two distant ones on the output manifold, it implies a large distance distortion and thus a poor GNN stability. We propose a distance-preserving graph dimension reduction (GDR) approach that utilizes spectral graph embedding and probabilistic graphical models (PGMs) to create low-dimensional input/output graph-based manifolds for meaningful stability analysis. Our empirical evaluations show that SAGMAN effectively assesses the stability of each node when subjected to various edge or feature perturbations, offering a scalable approach for evaluating the stability of GNNs, extending to applications within recommendation systems. Furthermore, we illustrate its utility in downstream tasks, notably in enhancing GNN stability and facilitating adversarial targeted attacks.

摘要: 现代图神经网络(GNN)对输入图结构和节点特征的变化很敏感，可能导致不可预测的行为和性能下降。在这项工作中，我们引入了一个称为Sagman的光谱框架来检查GNN的稳定性。该框架评估了输入和输出流形之间GNN之间的非线性映射引起的距离失真：当输入流形上的两个邻近节点(通过GNN模型)映射到输出流形上的两个相距较远的节点时，这意味着较大的距离失真，因此GNN稳定性较差。我们提出了一种保持距离的图降维方法(GDR)，该方法利用谱图嵌入和概率图模型(PGMS)来创建基于低维输入/输出图的流形，用于有意义的稳定性分析。实验结果表明，Sagman算法能够有效地评估每个节点在受到各种边缘或特征扰动时的稳定性，为评估GNN的稳定性提供了一种可扩展的方法，并扩展到推荐系统中的应用。此外，我们还说明了它在下游任务中的效用，特别是在增强GNN稳定性和促进对抗性定向攻击方面。



## **24. The Effectiveness of Random Forgetting for Robust Generalization**

随机遗忘对稳健泛化的有效性 cs.LG

Published as a conference paper at ICLR 2024

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2402.11733v1) [paper-pdf](http://arxiv.org/pdf/2402.11733v1)

**Authors**: Vijaya Raghavan T Ramkumar, Bahram Zonooz, Elahe Arani

**Abstract**: Deep neural networks are susceptible to adversarial attacks, which can compromise their performance and accuracy. Adversarial Training (AT) has emerged as a popular approach for protecting neural networks against such attacks. However, a key challenge of AT is robust overfitting, where the network's robust performance on test data deteriorates with further training, thus hindering generalization. Motivated by the concept of active forgetting in the brain, we introduce a novel learning paradigm called "Forget to Mitigate Overfitting (FOMO)". FOMO alternates between the forgetting phase, which randomly forgets a subset of weights and regulates the model's information through weight reinitialization, and the relearning phase, which emphasizes learning generalizable features. Our experiments on benchmark datasets and adversarial attacks show that FOMO alleviates robust overfitting by significantly reducing the gap between the best and last robust test accuracy while improving the state-of-the-art robustness. Furthermore, FOMO provides a better trade-off between standard and robust accuracy, outperforming baseline adversarial methods. Finally, our framework is robust to AutoAttacks and increases generalization in many real-world scenarios.

摘要: 深度神经网络容易受到敌意攻击，这可能会影响其性能和准确性。对抗训练(AT)已成为保护神经网络免受此类攻击的一种流行方法。然而，AT的一个关键挑战是稳健过拟合，即网络在测试数据上的稳健性能随着进一步的训练而恶化，从而阻碍泛化。在大脑主动遗忘概念的启发下，我们引入了一种新的学习范式--忘记缓解过度匹配(FOMO)。FOMO在遗忘阶段和重新学习阶段之间交替，遗忘阶段随机忘记权重的子集，并通过权重重新初始化来调整模型的信息，重新学习阶段强调学习可推广的特征。我们在基准数据集和敌意攻击上的实验表明，FOMO通过显著缩小最佳和最后稳健测试精度之间的差距来缓解稳健过拟合，同时提高了最新的稳健性。此外，FOMO在标准准确度和稳健准确度之间提供了更好的权衡，表现优于基线对抗性方法。最后，我们的框架对AutoAttack是健壮的，并增加了在许多真实世界场景中的泛化。



## **25. OUTFOX: LLM-Generated Essay Detection Through In-Context Learning with Adversarially Generated Examples**

Outfox：基于上下文学习的LLM生成的文章检测与恶意生成的示例 cs.CL

AAAI 2024 camera ready. Code and dataset available at  https://github.com/ryuryukke/OUTFOX

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2307.11729v3) [paper-pdf](http://arxiv.org/pdf/2307.11729v3)

**Authors**: Ryuto Koike, Masahiro Kaneko, Naoaki Okazaki

**Abstract**: Large Language Models (LLMs) have achieved human-level fluency in text generation, making it difficult to distinguish between human-written and LLM-generated texts. This poses a growing risk of misuse of LLMs and demands the development of detectors to identify LLM-generated texts. However, existing detectors lack robustness against attacks: they degrade detection accuracy by simply paraphrasing LLM-generated texts. Furthermore, a malicious user might attempt to deliberately evade the detectors based on detection results, but this has not been assumed in previous studies. In this paper, we propose OUTFOX, a framework that improves the robustness of LLM-generated-text detectors by allowing both the detector and the attacker to consider each other's output. In this framework, the attacker uses the detector's prediction labels as examples for in-context learning and adversarially generates essays that are harder to detect, while the detector uses the adversarially generated essays as examples for in-context learning to learn to detect essays from a strong attacker. Experiments in the domain of student essays show that the proposed detector improves the detection performance on the attacker-generated texts by up to +41.3 points F1-score. Furthermore, the proposed detector shows a state-of-the-art detection performance: up to 96.9 points F1-score, beating existing detectors on non-attacked texts. Finally, the proposed attacker drastically degrades the performance of detectors by up to -57.0 points F1-score, massively outperforming the baseline paraphrasing method for evading detection.

摘要: 大型语言模型(LLM)在文本生成方面达到了人类水平的流畅性，使得区分人类编写的文本和LLM生成的文本变得困难。这带来了滥用LLMS的越来越大的风险，并要求开发检测器来识别LLM生成的文本。然而，现有的检测器缺乏对攻击的稳健性：它们通过简单地解释LLM生成的文本来降低检测精度。此外，恶意用户可能试图根据检测结果故意躲避检测器，但在之前的研究中没有假设这一点。在本文中，我们提出了Outfox框架，它通过允许检测器和攻击者考虑彼此的输出来提高LLM生成的文本检测器的健壮性。在该框架中，攻击者使用检测器的预测标签作为上下文中学习的示例，并恶意生成更难检测的文章，而检测器使用恶意生成的文章作为上下文中学习的示例，以学习检测来自强大攻击者的文章。在学生作文领域的实验表明，该检测器对攻击者生成的文本的检测性能最高可提高41.3分F1-Score。此外，提出的检测器具有最先进的检测性能：高达96.9分的F1分数，在非攻击文本上击败了现有的检测器。最后，提出的攻击者极大地降低了检测器的性能，最高可达-57.0分F1-Score，大大超过了用于逃避检测的基线改述方法。



## **26. Evaluating Adversarial Robustness of Low dose CT Recovery**

评估低剂量CT恢复的对抗稳健性 eess.IV

MIDL 2023

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2402.11557v1) [paper-pdf](http://arxiv.org/pdf/2402.11557v1)

**Authors**: Kanchana Vaishnavi Gandikota, Paramanand Chandramouli, Hannah Droege, Michael Moeller

**Abstract**: Low dose computed tomography (CT) acquisition using reduced radiation or sparse angle measurements is recommended to decrease the harmful effects of X-ray radiation. Recent works successfully apply deep networks to the problem of low dose CT recovery on bench-mark datasets. However, their robustness needs a thorough evaluation before use in clinical settings. In this work, we evaluate the robustness of different deep learning approaches and classical methods for CT recovery. We show that deep networks, including model-based networks encouraging data consistency, are more susceptible to untargeted attacks. Surprisingly, we observe that data consistency is not heavily affected even for these poor quality reconstructions, motivating the need for better regularization for the networks. We demonstrate the feasibility of universal attacks and study attack transferability across different methods. We analyze robustness to attacks causing localized changes in clinically relevant regions. Both classical approaches and deep networks are affected by such attacks leading to changes in the visual appearance of localized lesions, for extremely small perturbations. As the resulting reconstructions have high data consistency with the original measurements, these localized attacks can be used to explore the solution space of the CT recovery problem.

摘要: 建议使用减少辐射或稀疏角测量的低剂量计算机断层扫描(CT)来减少X射线辐射的有害影响。最近的工作成功地将深度网络应用于基准数据集上的低剂量CT恢复问题。然而，在临床使用之前，需要对它们的稳健性进行彻底的评估。在这项工作中，我们评估了不同的深度学习方法和经典的CT恢复方法的稳健性。我们发现深层网络，包括鼓励数据一致性的基于模型的网络，更容易受到非目标攻击。令人惊讶的是，我们观察到，即使对于这些质量较差的重建，数据一致性也没有受到严重影响，这促使了对网络进行更好的正规化的需要。我们论证了通用攻击的可行性，并研究了不同方法的攻击可转移性。我们分析了对引起临床相关区域局部变化的攻击的稳健性。经典方法和深层网络都会受到此类攻击的影响，对于极小的扰动，这些攻击会导致局部病变的视觉外观发生变化。由于得到的重建结果与原始测量数据具有很高的一致性，这些局部化攻击可以用来探索CT恢复问题的解空间。



## **27. Measuring Privacy Loss in Distributed Spatio-Temporal Data**

分布式时空数据中隐私损失的度量 cs.CR

Chrome PDF viewer might not display Figures 3 and 4 properly

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2402.11526v1) [paper-pdf](http://arxiv.org/pdf/2402.11526v1)

**Authors**: Tatsuki Koga, Casey Meehan, Kamalika Chaudhuri

**Abstract**: Statistics about traffic flow and people's movement gathered from multiple geographical locations in a distributed manner are the driving force powering many applications, such as traffic prediction, demand prediction, and restaurant occupancy reports. However, these statistics are often based on sensitive location data of people, and hence privacy has to be preserved while releasing them. The standard way to do this is via differential privacy, which guarantees a form of rigorous, worst-case, person-level privacy. In this work, motivated by several counter-intuitive features of differential privacy in distributed location applications, we propose an alternative privacy loss against location reconstruction attacks by an informed adversary. Our experiments on real and synthetic data demonstrate that our privacy loss better reflects our intuitions on individual privacy violation in the distributed spatio-temporal setting.

摘要: 从多个地理位置以分布式方式收集的关于交通流量和人员流动的统计数据是许多应用程序的驱动力，例如交通预测、需求预测和餐厅入住率报告。然而，这些统计数据往往基于人们的敏感位置数据，因此在发布这些数据的同时必须保护隐私。做到这一点的标准方法是通过差异隐私，这保证了一种形式的严格的、最坏的情况下的个人级别的隐私。在这项工作中，受分布式位置应用中差异隐私的几个违反直觉的特征的启发，我们提出了一种针对知情对手的位置重构攻击的隐私损失替代方案。我们在真实和合成数据上的实验表明，我们的隐私损失更好地反映了我们在分布式时空环境中对个人隐私侵犯的直觉。



## **28. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

披着羊皮的狼：广义嵌套越狱提示可以轻松愚弄大型语言模型 cs.CL

Pre-print, code is available at https://github.com/NJUNLP/ReNeLLM

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2311.08268v2) [paper-pdf](http://arxiv.org/pdf/2311.08268v2)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, compromising generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.

摘要: 大型语言模型（LLM），如ChatGPT和GPT-4，旨在提供有用和安全的响应。然而，被称为“越狱”的对抗性提示可以规避安全措施，导致LLM生成潜在的有害内容。探索越狱提示可以帮助更好地揭示LLM的弱点，并进一步引导我们保护它们。不幸的是，现有的越狱方法要么遭受复杂的手动设计，要么需要在其他白盒模型上进行优化，从而影响泛化或效率。本文将越狱提示攻击归纳为两个方面：（1）提示重写和（2）场景嵌套。在此基础上，我们提出了ReNeLLM，一个自动框架，利用LLM本身来生成有效的越狱提示。大量的实验表明，与现有的基线相比，ReNeLLM显着提高了攻击成功率，同时大大降低了时间成本。我们的研究也揭示了目前的防御方法在保护LLM方面的不足。最后，从即时执行优先级的角度分析了LLM防御的失效，并提出了相应的防御策略。我们希望我们的研究能够促进学术界和LLM开发人员提供更安全，更受监管的LLM。该代码可在https://github.com/NJUNLP/ReNeLLM上获得。



## **29. Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information**

基于复杂度测度和上下文信息的令牌级对抗提示检测 cs.CL

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2311.11509v3) [paper-pdf](http://arxiv.org/pdf/2311.11509v3)

**Authors**: Zhengmian Hu, Gang Wu, Saayan Mitra, Ruiyi Zhang, Tong Sun, Heng Huang, Viswanathan Swaminathan

**Abstract**: In recent years, Large Language Models (LLM) have emerged as pivotal tools in various applications. However, these models are susceptible to adversarial prompt attacks, where attackers can carefully curate input strings that mislead LLMs into generating incorrect or undesired outputs. Previous work has revealed that with relatively simple yet effective attacks based on discrete optimization, it is possible to generate adversarial prompts that bypass moderation and alignment of the models. This vulnerability to adversarial prompts underscores a significant concern regarding the robustness and reliability of LLMs. Our work aims to address this concern by introducing a novel approach to detecting adversarial prompts at a token level, leveraging the LLM's capability to predict the next token's probability. We measure the degree of the model's perplexity, where tokens predicted with high probability are considered normal, and those exhibiting high perplexity are flagged as adversarial. Additionaly, our method also integrates context understanding by incorporating neighboring token information to encourage the detection of contiguous adversarial prompt sequences. To this end, we design two algorithms for adversarial prompt detection: one based on optimization techniques and another on Probabilistic Graphical Models (PGM). Both methods are equipped with efficient solving methods, ensuring efficient adversarial prompt detection. Our token-level detection result can be visualized as heatmap overlays on the text sequence, allowing for a clearer and more intuitive representation of which part of the text may contain adversarial prompts.

摘要: 近年来，大型语言模型(LLM)已经成为各种应用中的关键工具。然而，这些模型容易受到敌意提示攻击，攻击者可以仔细策划误导LLM生成不正确或不想要的输出的输入字符串。以前的工作已经表明，通过基于离散优化的相对简单但有效的攻击，有可能生成绕过模型的缓和和对齐的对抗性提示。这种对敌意提示的脆弱性突出了人们对LLMS的健壮性和可靠性的严重关切。我们的工作旨在通过引入一种新的方法来检测令牌级别的敌意提示，利用LLM预测下一个令牌的概率的能力来解决这一问题。我们测量了模型的困惑程度，其中高概率预测的标记被认为是正常的，而那些表现出高困惑的标记被标记为对抗性的。此外，我们的方法还通过结合邻近的令牌信息来整合上下文理解，以鼓励检测连续的对抗性提示序列。为此，我们设计了两种对抗性提示检测算法：一种基于优化技术，另一种基于概率图模型(PGM)。这两种方法都配备了高效的解决方法，确保了高效的对抗性及时检测。我们的令牌级检测结果可以可视化为覆盖在文本序列上的热图，从而允许更清晰、更直观地表示文本的哪一部分可能包含对抗性提示。



## **30. A Curious Case of Searching for the Correlation between Training Data and Adversarial Robustness of Transformer Textual Models**

寻找训练数据与变压器文本模型对抗稳健性之间的相关性的一个新奇案例 cs.LG

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2402.11469v1) [paper-pdf](http://arxiv.org/pdf/2402.11469v1)

**Authors**: Cuong Dang, Dung D. Le, Thai Le

**Abstract**: Existing works have shown that fine-tuned textual transformer models achieve state-of-the-art prediction performances but are also vulnerable to adversarial text perturbations. Traditional adversarial evaluation is often done \textit{only after} fine-tuning the models and ignoring the training data. In this paper, we want to prove that there is also a strong correlation between training data and model robustness. To this end, we extract 13 different features representing a wide range of input fine-tuning corpora properties and use them to predict the adversarial robustness of the fine-tuned models. Focusing mostly on encoder-only transformer models BERT and RoBERTa with additional results for BART, ELECTRA and GPT2, we provide diverse evidence to support our argument. First, empirical analyses show that (a) extracted features can be used with a lightweight classifier such as Random Forest to effectively predict the attack success rate and (b) features with the most influence on the model robustness have a clear correlation with the robustness. Second, our framework can be used as a fast and effective additional tool for robustness evaluation since it (a) saves 30x-193x runtime compared to the traditional technique, (b) is transferable across models, (c) can be used under adversarial training, and (d) robust to statistical randomness. Our code will be publicly available.

摘要: 已有的工作表明，微调的文本变换模型取得了最先进的预测性能，但也容易受到对抗性文本扰动的影响。传统的对抗性评估往往是在对模型进行微调而忽略训练数据之后才进行的。在本文中，我们想要证明训练数据和模型稳健性之间也存在很强的相关性。为此，我们提取了13个不同的特征，代表了广泛的输入微调语料库属性，并使用它们来预测微调模型的对抗性健壮性。主要关注仅编码器的变压器模型BART和Roberta，以及BART、ELECTRA和GPT2的其他结果，我们提供了不同的证据来支持我们的论点。首先，实证分析表明：(A)提取的特征可以与随机森林等轻量级分类器一起有效地预测攻击成功率；(B)对模型稳健性影响最大的特征与模型的稳健性有明显的相关性。其次，我们的框架可以作为快速有效的额外工具用于健壮性评估，因为它(A)比传统技术节省30-193倍的运行时间，(B)可以跨模型转移，(C)可以在对抗性训练下使用，以及(D)对统计随机性具有健壮性。我们的代码将公开可用。



## **31. VoltSchemer: Use Voltage Noise to Manipulate Your Wireless Charger**

VoltSchemer：使用电压噪声操纵您的无线充电器 cs.CR

This paper has been accepted by the 33rd USENIX Security Symposium

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2402.11423v1) [paper-pdf](http://arxiv.org/pdf/2402.11423v1)

**Authors**: Zihao Zhan, Yirui Yang, Haoqi Shan, Hanqiu Wang, Yier Jin, Shuo Wang

**Abstract**: Wireless charging is becoming an increasingly popular charging solution in portable electronic products for a more convenient and safer charging experience than conventional wired charging. However, our research identified new vulnerabilities in wireless charging systems, making them susceptible to intentional electromagnetic interference. These vulnerabilities facilitate a set of novel attack vectors, enabling adversaries to manipulate the charger and perform a series of attacks.   In this paper, we propose VoltSchemer, a set of innovative attacks that grant attackers control over commercial-off-the-shelf wireless chargers merely by modulating the voltage from the power supply. These attacks represent the first of its kind, exploiting voltage noises from the power supply to manipulate wireless chargers without necessitating any malicious modifications to the chargers themselves. The significant threats imposed by VoltSchemer are substantiated by three practical attacks, where a charger can be manipulated to: control voice assistants via inaudible voice commands, damage devices being charged through overcharging or overheating, and bypass Qi-standard specified foreign-object-detection mechanism to damage valuable items exposed to intense magnetic fields.   We demonstrate the effectiveness and practicality of the VoltSchemer attacks with successful attacks on 9 top-selling COTS wireless chargers. Furthermore, we discuss the security implications of our findings and suggest possible countermeasures to mitigate potential threats.

摘要: 无线充电正成为便携式电子产品中越来越受欢迎的充电解决方案，以获得比传统有线充电更方便和更安全的充电体验。然而，我们的研究发现了无线充电系统中的新漏洞，使它们容易受到有意的电磁干扰。这些漏洞促进了一系列新颖的攻击载体，使对手能够操纵充电器并执行一系列攻击。   在本文中，我们提出了VoltSchemer，这是一组创新的攻击，攻击者只需通过调制电源电压就可以控制商用现成的无线充电器。这些攻击代表了此类攻击的第一次，利用电源的电压噪声来操纵无线充电器，而不需要对充电器本身进行任何恶意修改。VoltSchemer造成的重大威胁通过三种实际攻击得到证实，其中充电器可以被操纵：通过听不见的语音命令控制语音助手，通过过度充电或过热损坏正在充电的设备，以及绕过Qi标准指定的异物检测机制，以损坏暴露在强磁场中的贵重物品。   我们证明了VoltSchemer攻击的有效性和实用性，成功攻击了9个最畅销的COTS无线充电器。此外，我们讨论了我们的研究结果的安全影响，并提出可能的对策，以减轻潜在的威胁。



## **32. Effective Prompt Extraction from Language Models**

从语言模型中有效地提取提示 cs.CL

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2307.06865v2) [paper-pdf](http://arxiv.org/pdf/2307.06865v2)

**Authors**: Yiming Zhang, Nicholas Carlini, Daphne Ippolito

**Abstract**: The text generated by large language models is commonly controlled by prompting, where a prompt prepended to a user's query guides the model's output. The prompts used by companies to guide their models are often treated as secrets, to be hidden from the user making the query. They have even been treated as commodities to be bought and sold. However, anecdotal reports have shown adversarial users employing prompt extraction attacks to recover these prompts. In this paper, we present a framework for systematically measuring the effectiveness of these attacks. In experiments with 3 different sources of prompts and 11 underlying large language models, we find that simple text-based attacks can in fact reveal prompts with high probability. Our framework determines with high precision whether an extracted prompt is the actual secret prompt, rather than a model hallucination. Prompt extraction experiments on real systems such as Bing Chat and ChatGPT suggest that system prompts can be revealed by an adversary despite existing defenses in place.

摘要: 大型语言模型生成的文本通常由提示控制，其中用户查询前的提示指导模型的输出。公司用来指导他们的模型的提示通常被视为秘密，对进行查询的用户隐藏。他们甚至被当作商品来买卖。然而，轶事报告显示，敌对用户使用提示提取攻击来恢复这些提示。在本文中，我们提出了一个框架，系统地衡量这些攻击的有效性。在3个不同的提示源和11个底层大型语言模型的实验中，我们发现简单的基于文本的攻击实际上可以以很高的概率揭示提示。我们的框架以高精度确定提取的提示是否是实际的秘密提示，而不是模型幻觉。在Bing Chat和ChatGPT等真实系统上进行的提示提取实验表明，尽管存在现有的防御措施，但系统提示仍可能被对手泄露。



## **33. DALA: A Distribution-Aware LoRA-Based Adversarial Attack against Language Models**

Dala：一种基于分布感知LORA的语言模型对抗性攻击 cs.CL

First two authors contribute equally

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2311.08598v2) [paper-pdf](http://arxiv.org/pdf/2311.08598v2)

**Authors**: Yibo Wang, Xiangjue Dong, James Caverlee, Philip S. Yu

**Abstract**: Language models (LMs) can be manipulated by adversarial attacks, which introduce subtle perturbations to input data. While recent attack methods can achieve a relatively high attack success rate (ASR), we've observed that the generated adversarial examples have a different data distribution compared with the original examples. Specifically, these adversarial examples exhibit reduced confidence levels and greater divergence from the training data distribution. Consequently, they are easy to detect using straightforward detection methods, diminishing the efficacy of such attacks. To address this issue, we propose a Distribution-Aware LoRA-based Adversarial Attack (DALA) method. DALA considers distribution shifts of adversarial examples to improve the attack's effectiveness under detection methods. We further design a novel evaluation metric, the Non-detectable Attack Success Rate (NASR), which integrates both ASR and detectability for the attack task. We conduct experiments on four widely used datasets to validate the attack effectiveness and transferability of adversarial examples generated by DALA against both the white-box BERT-base model and the black-box LLaMA2-7b model. Our codes are available at https://anonymous.4open.science/r/DALA-A16D/.

摘要: 语言模型(LMS)可以被敌意攻击所操纵，这种攻击会给输入数据带来微妙的扰动。虽然目前的攻击方法可以达到相对较高的攻击成功率(ASR)，但我们观察到生成的敌意示例与原始示例相比具有不同的数据分布。具体地说，这些对抗性的例子表现出更低的置信度和与训练数据分布更大的背离。因此，使用直接的检测方法很容易检测到它们，从而降低了此类攻击的有效性。为了解决这个问题，我们提出了一种基于分布感知LORA的对抗性攻击(DALA)方法。Dala考虑了对抗性样本的分布偏移，以提高检测方法下的攻击有效性。在此基础上，我们进一步设计了一种新的评价指标--不可检测攻击成功率(NASR)，它综合了攻击任务的ASR和可检测性。我们在四个广泛使用的数据集上进行了实验，以验证DALA生成的对抗性实例在白盒Bert-base模型和黑盒LLaMA2-7b模型下的攻击有效性和可转移性。我们的代码可在https://anonymous.4open.science/r/DALA-A16D/.上获得



## **34. On the Evaluation of User Privacy in Deep Neural Networks using Timing Side Channel**

基于时序侧通道的深度神经网络用户隐私评估研究 cs.CR

15 pages, 20 figures

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2208.01113v3) [paper-pdf](http://arxiv.org/pdf/2208.01113v3)

**Authors**: Shubhi Shukla, Manaar Alam, Sarani Bhattacharya, Debdeep Mukhopadhyay, Pabitra Mitra

**Abstract**: Recent Deep Learning (DL) advancements in solving complex real-world tasks have led to its widespread adoption in practical applications. However, this opportunity comes with significant underlying risks, as many of these models rely on privacy-sensitive data for training in a variety of applications, making them an overly-exposed threat surface for privacy violations. Furthermore, the widespread use of cloud-based Machine-Learning-as-a-Service (MLaaS) for its robust infrastructure support has broadened the threat surface to include a variety of remote side-channel attacks. In this paper, we first identify and report a novel data-dependent timing side-channel leakage (termed Class Leakage) in DL implementations originating from non-constant time branching operation in a widely used DL framework PyTorch. We further demonstrate a practical inference-time attack where an adversary with user privilege and hard-label black-box access to an MLaaS can exploit Class Leakage to compromise the privacy of MLaaS users. DL models are vulnerable to Membership Inference Attack (MIA), where an adversary's objective is to deduce whether any particular data has been used while training the model. In this paper, as a separate case study, we demonstrate that a DL model secured with differential privacy (a popular countermeasure against MIA) is still vulnerable to MIA against an adversary exploiting Class Leakage. We develop an easy-to-implement countermeasure by making a constant-time branching operation that alleviates the Class Leakage and also aids in mitigating MIA. We have chosen two standard benchmarking image classification datasets, CIFAR-10 and CIFAR-100 to train five state-of-the-art pre-trained DL models, over two different computing environments having Intel Xeon and Intel i7 processors to validate our approach.

摘要: 最近深度学习（DL）在解决复杂的现实世界任务方面取得的进展使其在实际应用中得到了广泛采用。然而，这一机会伴随着重大的潜在风险，因为许多模型依赖于隐私敏感数据来进行各种应用程序的训练，使其成为隐私侵犯的过度暴露的威胁表面。此外，基于云的机器学习即服务（MLaaS）的广泛使用为其强大的基础设施提供了支持，这扩大了威胁面，包括各种远程侧通道攻击。在本文中，我们首先确定并报告了一种新的数据相关的定时侧信道泄漏（称为类泄漏）在DL实现起源于非恒定时间分支操作在一个广泛使用的DL框架PyTorch。我们进一步展示了一个实际的推理时间攻击，其中具有用户权限和硬标签黑盒访问MLaaS的对手可以利用类泄漏来损害MLaaS用户的隐私。DL模型容易受到成员推理攻击（MIA），其中对手的目标是推断在训练模型时是否使用了任何特定数据。在本文中，作为一个单独的案例研究，我们证明了一个DL模型与差分隐私（一种流行的对抗MIA的对策）安全仍然容易受到MIA对利用类泄漏的对手。我们开发了一个易于实现的对策，使一个恒定的时间分支操作，消除类泄漏，也有助于减轻MIA。我们选择了两个标准的基准图像分类数据集CIFAR-10和CIFAR-100来训练五个最先进的预训练DL模型，在两个不同的计算环境中使用英特尔至强和英特尔i7处理器来验证我们的方法。



## **35. Maintaining Adversarial Robustness in Continuous Learning**

在持续学习中保持对手的健壮性 cs.LG

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2402.11196v1) [paper-pdf](http://arxiv.org/pdf/2402.11196v1)

**Authors**: Xiaolei Ru, Xiaowei Cao, Zijia Liu, Jack Murdoch Moore, Xin-Ya Zhang, Xia Zhu, Wenjia Wei, Gang Yan

**Abstract**: Adversarial robustness is essential for security and reliability of machine learning systems. However, the adversarial robustness gained by sophisticated defense algorithms is easily erased as the neural network evolves to learn new tasks. This vulnerability can be addressed by fostering a novel capability for neural networks, termed continual robust learning, which focuses on both the (classification) performance and adversarial robustness on previous tasks during continuous learning. To achieve continuous robust learning, we propose an approach called Double Gradient Projection that projects the gradients for weight updates orthogonally onto two crucial subspaces -- one for stabilizing the smoothed sample gradients and another for stabilizing the final outputs of the neural network. The experimental results on four benchmarks demonstrate that the proposed approach effectively maintains continuous robustness against strong adversarial attacks, outperforming the baselines formed by combining the existing defense strategies and continual learning methods.

摘要: 对抗鲁棒性对于机器学习系统的安全性和可靠性至关重要。然而，复杂的防御算法所获得的对抗鲁棒性很容易随着神经网络的发展而学习新的任务。这种脆弱性可以通过培养神经网络的新能力来解决，称为持续鲁棒学习，它专注于在持续学习期间对先前任务的（分类）性能和对抗鲁棒性。为了实现连续的鲁棒学习，我们提出了一种称为双梯度投影的方法，该方法将权重更新的梯度正交投影到两个关键子空间上-一个用于稳定平滑的样本梯度，另一个用于稳定神经网络的最终输出。在4个基准测试上的实验结果表明，该方法有效地保持了对强对抗攻击的持续鲁棒性，优于现有防御策略和持续学习方法相结合形成的基线。



## **36. Quantization Aware Attack: Enhancing Transferable Adversarial Attacks by Model Quantization**

量化感知攻击：利用模型量化增强可转移对抗性攻击 cs.CR

Accepted by IEEE Transactions on Information Forensics and Security  in 2024

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2305.05875v3) [paper-pdf](http://arxiv.org/pdf/2305.05875v3)

**Authors**: Yulong Yang, Chenhao Lin, Qian Li, Zhengyu Zhao, Haoran Fan, Dawei Zhou, Nannan Wang, Tongliang Liu, Chao Shen

**Abstract**: Quantized neural networks (QNNs) have received increasing attention in resource-constrained scenarios due to their exceptional generalizability. However, their robustness against realistic black-box adversarial attacks has not been extensively studied. In this scenario, adversarial transferability is pursued across QNNs with different quantization bitwidths, which particularly involve unknown architectures and defense methods. Previous studies claim that transferability is difficult to achieve across QNNs with different bitwidths on the condition that they share the same architecture. However, we discover that under different architectures, transferability can be largely improved by using a QNN quantized with an extremely low bitwidth as the substitute model. We further improve the attack transferability by proposing \textit{quantization aware attack} (QAA), which fine-tunes a QNN substitute model with a multiple-bitwidth training objective. In particular, we demonstrate that QAA addresses the two issues that are commonly known to hinder transferability: 1) quantization shifts and 2) gradient misalignments. Extensive experimental results validate the high transferability of the QAA to diverse target models. For instance, when adopting the ResNet-34 substitute model on ImageNet, QAA outperforms the current best attack in attacking standardly trained DNNs, adversarially trained DNNs, and QNNs with varied bitwidths by 4.3\% $\sim$ 20.9\%, 8.7\% $\sim$ 15.5\%, and 2.6\% $\sim$ 31.1\% (absolute), respectively. In addition, QAA is efficient since it only takes one epoch for fine-tuning. In the end, we empirically explain the effectiveness of QAA from the view of the loss landscape. Our code is available at https://github.com/yyl-github-1896/QAA/

摘要: 量化神经网络(QNN)由于具有良好的泛化能力，在资源受限的情况下受到越来越多的关注。然而，它们对现实黑盒攻击的稳健性还没有得到广泛的研究。在这种情况下，不同量化位宽的QNN之间追求对抗性的可转移性，这尤其涉及未知的体系结构和防御方法。以前的研究表明，如果QNN共享相同的体系结构，则很难实现不同位宽的QNN之间的可转移性。然而，我们发现，在不同的体系结构下，使用具有极低位宽的量化的QNN作为替代模型，可以大大提高可转移性。为了进一步提高攻击的可转移性，提出了一种基于多位宽训练目标的量化感知攻击(QAA)，对QNN替换模型进行微调。特别是，我们证明了QAA解决了两个众所周知的阻碍可转移性的问题：1)量化位移和2)梯度失调。大量的实验结果验证了QAA对不同目标模型的高可移植性。例如，当在ImageNet上采用ResNet-34替换模型时，QAA在攻击标准训练的DNN、对抗性训练的DNN和不同位宽的QNN时，分别比当前最好的攻击性能高4.3$SIM$20.9\%、8.7$\SIM$15.5\%和2.6$\SIM$31.1\%(绝对值)。此外，QAA是高效的，因为它只需要一个时期来进行微调。最后，我们从损失的角度对QAA的有效性进行了实证解释。我们的代码可以在https://github.com/yyl-github-1896/QAA/上找到



## **37. Adversarial Illusions in Multi-Modal Embeddings**

多模态嵌入中的对抗性错觉 cs.CR

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2308.11804v3) [paper-pdf](http://arxiv.org/pdf/2308.11804v3)

**Authors**: Tingwei Zhang, Rishi Jha, Eugene Bagdasaryan, Vitaly Shmatikov

**Abstract**: Multi-modal embeddings encode texts, images, sounds, videos, etc., into a single embedding space, aligning representations across different modalities (e.g., associate an image of a dog with a barking sound). In this paper, we show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an image or a sound, an adversary can perturb it to make its embedding close to an arbitrary, adversary-chosen input in another modality.   These attacks are cross-modal and targeted: the adversary is free to align any image and any sound with any target of his choice. Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks and modalities, enabling a wholesale compromise of current and future downstream tasks and modalities not available to the adversary. Using ImageBind and AudioCLIP embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, zero-shot classification, and audio retrieval.   We investigate transferability of illusions across different embeddings and develop a black-box version of our method that we use to demonstrate the first adversarial alignment attack on Amazon's commercial, proprietary Titan embedding. Finally, we analyze countermeasures and evasion attacks.

摘要: 多模式嵌入将文本、图像、声音、视频等编码到单个嵌入空间中，跨不同模式对齐表示(例如，将狗的图像与叫声相关联)。在这篇文章中，我们证明了多模式嵌入可能容易受到一种我们称为“对抗错觉”的攻击。在给定图像或声音的情况下，敌手可以对其进行干扰，使其嵌入到另一种形式中，接近对手选择的任意输入。这些攻击是跨模式的和有针对性的：对手可以自由地将任何图像和任何声音与他选择的任何目标对齐。对抗性错觉利用嵌入空间中的邻近性，因此对下游任务和模式是不可知的，从而使对手无法获得的当前和未来下游任务和模式能够大规模妥协。使用ImageBind和AudioCLIP嵌入，我们演示了在不知道特定下游任务的情况下生成的恶意对齐输入如何误导图像生成、文本生成、零镜头分类和音频检索。我们调查了错觉在不同嵌入中的可转移性，并开发了我们方法的黑盒版本，用于演示对亚马逊商业、专有的Titan嵌入的第一次敌意对齐攻击。最后，分析了相应的对策和规避攻击。



## **38. Token-Ensemble Text Generation: On Attacking the Automatic AI-Generated Text Detection**

令牌集成文本生成：对人工智能生成文本自动检测的攻击 cs.CL

Submitted to ACL 2024

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2402.11167v1) [paper-pdf](http://arxiv.org/pdf/2402.11167v1)

**Authors**: Fan Huang, Haewoon Kwak, Jisun An

**Abstract**: The robustness of AI-content detection models against cultivated attacks (e.g., paraphrasing or word switching) remains a significant concern. This study proposes a novel token-ensemble generation strategy to challenge the robustness of current AI-content detection approaches. We explore the ensemble attack strategy by completing the prompt with the next token generated from random candidate LLMs. We find the token-ensemble approach significantly drops the performance of AI-content detection models (The code and test sets will be released). Our findings reveal that token-ensemble generation poses a vital challenge to current detection models and underlines the need for advancing detection technologies to counter sophisticated adversarial strategies.

摘要: 人工智能内容检测模型对人工培养的攻击(例如，转述或单词切换)的稳健性仍然是一个重要的问题。本文提出了一种新的令牌集成生成策略来挑战现有人工智能内容检测方法的健壮性。通过使用随机候选LLM生成的下一个令牌来完成提示，我们探索了集成攻击策略。我们发现令牌集成方法显著降低了AI内容检测模型的性能(代码和测试集将被发布)。我们的发现表明，令牌集成生成对当前的检测模型构成了至关重要的挑战，并强调了需要先进的检测技术来对抗复杂的对手策略。



## **39. DART: A Principled Approach to Adversarially Robust Unsupervised Domain Adaptation**

DART：对抗性稳健无监督领域自适应的原则性方法 cs.LG

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.11120v1) [paper-pdf](http://arxiv.org/pdf/2402.11120v1)

**Authors**: Yunjuan Wang, Hussein Hazimeh, Natalia Ponomareva, Alexey Kurakin, Ibrahim Hammoud, Raman Arora

**Abstract**: Distribution shifts and adversarial examples are two major challenges for deploying machine learning models. While these challenges have been studied individually, their combination is an important topic that remains relatively under-explored. In this work, we study the problem of adversarial robustness under a common setting of distribution shift - unsupervised domain adaptation (UDA). Specifically, given a labeled source domain $D_S$ and an unlabeled target domain $D_T$ with related but different distributions, the goal is to obtain an adversarially robust model for $D_T$. The absence of target domain labels poses a unique challenge, as conventional adversarial robustness defenses cannot be directly applied to $D_T$. To address this challenge, we first establish a generalization bound for the adversarial target loss, which consists of (i) terms related to the loss on the data, and (ii) a measure of worst-case domain divergence. Motivated by this bound, we develop a novel unified defense framework called Divergence Aware adveRsarial Training (DART), which can be used in conjunction with a variety of standard UDA methods; e.g., DANN [Ganin and Lempitsky, 2015]. DART is applicable to general threat models, including the popular $\ell_p$-norm model, and does not require heuristic regularizers or architectural changes. We also release DomainRobust: a testbed for evaluating robustness of UDA models to adversarial attacks. DomainRobust consists of 4 multi-domain benchmark datasets (with 46 source-target pairs) and 7 meta-algorithms with a total of 11 variants. Our large-scale experiments demonstrate that on average, DART significantly enhances model robustness on all benchmarks compared to the state of the art, while maintaining competitive standard accuracy. The relative improvement in robustness from DART reaches up to 29.2% on the source-target domain pairs considered.

摘要: 分布转移和敌对例子是部署机器学习模型的两大挑战。虽然已经单独研究了这些挑战，但它们的结合是一个相对较少探索的重要课题。在这项工作中，我们研究了一种常见的分布平移背景下的对手健壮性问题--无监督域自适应(UDA)。具体地说，给定一个带标签的源域$D_S$和一个具有相关但不同分布的未标记的目标域$D_T$，目标是得到一个关于$D_T$的对抗性稳健模型。缺乏目标域标签构成了一个独特的挑战，因为传统的对抗性健壮性防御不能直接应用于$D_T$。为了应对这一挑战，我们首先建立了对抗性目标损失的泛化界限，它由(I)与数据损失相关的项和(Ii)最坏情况域分歧的度量组成。在这一界限的推动下，我们开发了一种新颖的统一防御框架，称为分歧感知对抗训练(DART)，它可以与各种标准的UDA方法结合使用，例如Dann[Ganin和Lempitsky，2015]。DART适用于一般威胁模型，包括流行的$\ell_p$-Norm模型，并且不需要启发式正则化程序或架构更改。我们还发布了DomainRobust：一个用于评估UDA模型对对手攻击的健壮性的测试床。DomainRobust由4个多域基准数据集(46个源-目标对)和7个元算法组成，总共有11个变体。我们的大规模实验表明，与最新水平相比，DART在所有基准上显著增强了模型的稳健性，同时保持了具有竞争力的标准精度。在所考虑的源-目标域对上，DART在健壮性方面的相对改进高达29.2%。



## **40. VQAttack: Transferable Adversarial Attacks on Visual Question Answering via Pre-trained Models**

VQAttack：基于预训练模型的可转移敌意视觉问答攻击 cs.CV

AAAI 2024, 11 pages

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.11083v1) [paper-pdf](http://arxiv.org/pdf/2402.11083v1)

**Authors**: Ziyi Yin, Muchao Ye, Tianrong Zhang, Jiaqi Wang, Han Liu, Jinghui Chen, Ting Wang, Fenglong Ma

**Abstract**: Visual Question Answering (VQA) is a fundamental task in computer vision and natural language process fields. Although the ``pre-training & finetuning'' learning paradigm significantly improves the VQA performance, the adversarial robustness of such a learning paradigm has not been explored. In this paper, we delve into a new problem: using a pre-trained multimodal source model to create adversarial image-text pairs and then transferring them to attack the target VQA models. Correspondingly, we propose a novel VQAttack model, which can iteratively generate both image and text perturbations with the designed modules: the large language model (LLM)-enhanced image attack and the cross-modal joint attack module. At each iteration, the LLM-enhanced image attack module first optimizes the latent representation-based loss to generate feature-level image perturbations. Then it incorporates an LLM to further enhance the image perturbations by optimizing the designed masked answer anti-recovery loss. The cross-modal joint attack module will be triggered at a specific iteration, which updates the image and text perturbations sequentially. Notably, the text perturbation updates are based on both the learned gradients in the word embedding space and word synonym-based substitution. Experimental results on two VQA datasets with five validated models demonstrate the effectiveness of the proposed VQAttack in the transferable attack setting, compared with state-of-the-art baselines. This work reveals a significant blind spot in the ``pre-training & fine-tuning'' paradigm on VQA tasks. Source codes will be released.

摘要: 视觉问答是计算机视觉和自然语言处理领域的一项基本任务。虽然“预训练和精调”学习范式显著提高了VQA成绩，但这种学习范式的对抗稳健性还没有被探索过。本文深入研究了一个新的问题：使用预先训练好的多模源模型来生成对抗性图文对，然后将它们转移到攻击目标的VQA模型。相应地，我们提出了一种新的VQAttack模型，该模型可以迭代地产生图像和文本扰动，并设计了两个模块：大语言模型(LLM)增强的图像攻击和跨模式联合攻击模块。在每一次迭代中，LLM增强的图像攻击模块首先优化基于潜在表示的损失，以产生特征级的图像扰动。然后，通过优化设计的抗恢复损失的蒙版答案，引入LLM来进一步增强图像扰动。跨模式联合攻击模块将在特定迭代时触发，该迭代将按顺序更新图像和文本扰动。值得注意的是，文本扰动更新基于单词嵌入空间中的学习梯度和基于单词同义词的替换。在两个VQA数据集和5个已验证模型上的实验结果表明，该算法在可转移攻击环境下具有较好的性能。这项工作揭示了VQA任务“预培训和微调”范式中的一个重大盲点。源代码将会公布。



## **41. The AI Security Pyramid of Pain**

人工智能安全的痛苦金字塔 cs.CR

SPIE DCS 2024

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.11082v1) [paper-pdf](http://arxiv.org/pdf/2402.11082v1)

**Authors**: Chris M. Ward, Josh Harguess, Julia Tao, Daniel Christman, Paul Spicer, Mike Tan

**Abstract**: We introduce the AI Security Pyramid of Pain, a framework that adapts the cybersecurity Pyramid of Pain to categorize and prioritize AI-specific threats. This framework provides a structured approach to understanding and addressing various levels of AI threats. Starting at the base, the pyramid emphasizes Data Integrity, which is essential for the accuracy and reliability of datasets and AI models, including their weights and parameters. Ensuring data integrity is crucial, as it underpins the effectiveness of all AI-driven decisions and operations. The next level, AI System Performance, focuses on MLOps-driven metrics such as model drift, accuracy, and false positive rates. These metrics are crucial for detecting potential security breaches, allowing for early intervention and maintenance of AI system integrity. Advancing further, the pyramid addresses the threat posed by Adversarial Tools, identifying and neutralizing tools used by adversaries to target AI systems. This layer is key to staying ahead of evolving attack methodologies. At the Adversarial Input layer, the framework addresses the detection and mitigation of inputs designed to deceive or exploit AI models. This includes techniques like adversarial patterns and prompt injection attacks, which are increasingly used in sophisticated attacks on AI systems. Data Provenance is the next critical layer, ensuring the authenticity and lineage of data and models. This layer is pivotal in preventing the use of compromised or biased data in AI systems. At the apex is the tactics, techniques, and procedures (TTPs) layer, dealing with the most complex and challenging aspects of AI security. This involves a deep understanding and strategic approach to counter advanced AI-targeted attacks, requiring comprehensive knowledge and planning.

摘要: 我们引入了人工智能安全金字塔of Pain，这是一个适应网络安全金字塔的框架，用于对特定于人工智能的威胁进行分类和优先排序。这个框架提供了一种结构化的方法来理解和解决各种级别的人工智能威胁。从基础开始，金字塔强调数据完整性，这对于数据集和人工智能模型的准确性和可靠性至关重要，包括它们的权重和参数。确保数据完整性至关重要，因为它支撑着所有人工智能驱动的决策和操作的有效性。下一个级别，AI系统性能，重点关注MLOPS驱动的指标，如模型漂移、准确性和误检率。这些指标对于检测潜在的安全漏洞至关重要，允许及早干预和维护人工智能系统的完整性。进一步推进，金字塔解决了对抗性工具构成的威胁，识别和中和了对手使用的针对人工智能系统的工具。这一层是保持领先于不断发展的攻击方法的关键。在对抗性输入层，该框架解决了对旨在欺骗或利用人工智能模型的输入的检测和缓解。这包括对抗性模式和即时注入攻击等技术，这些技术越来越多地用于对人工智能系统的复杂攻击。数据来源是下一个关键层，确保数据和模型的真实性和系列性。这一层在防止在人工智能系统中使用受危害或有偏见的数据方面至关重要。顶端是战术、技术和程序(TTP)层，处理人工智能安全中最复杂和最具挑战性的方面。这需要一种深刻的理解和战略方法来对抗先进的人工智能目标攻击，需要全面的知识和规划。



## **42. QDoor: Exploiting Approximate Synthesis for Backdoor Attacks in Quantum Neural Networks**

QDoor：利用近似合成实现量子神经网络中的后门攻击 quant-ph

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2307.09529v2) [paper-pdf](http://arxiv.org/pdf/2307.09529v2)

**Authors**: Cheng Chu, Fan Chen, Philip Richerme, Lei Jiang

**Abstract**: Quantum neural networks (QNNs) succeed in object recognition, natural language processing, and financial analysis. To maximize the accuracy of a QNN on a Noisy Intermediate Scale Quantum (NISQ) computer, approximate synthesis modifies the QNN circuit by reducing error-prone 2-qubit quantum gates. The success of QNNs motivates adversaries to attack QNNs via backdoors. However, na\"ively transplanting backdoors designed for classical neural networks to QNNs yields only low attack success rate, due to the noises and approximate synthesis on NISQ computers. Prior quantum circuit-based backdoors cannot selectively attack some inputs or work with all types of encoding layers of a QNN circuit. Moreover, it is easy to detect both transplanted and circuit-based backdoors in a QNN.   In this paper, we propose a novel and stealthy backdoor attack, QDoor, to achieve high attack success rate in approximately-synthesized QNN circuits by weaponizing unitary differences between uncompiled QNNs and their synthesized counterparts. QDoor trains a QNN behaving normally for all inputs with and without a trigger. However, after approximate synthesis, the QNN circuit always predicts any inputs with a trigger to a predefined class while still acts normally for benign inputs. Compared to prior backdoor attacks, QDoor improves the attack success rate by $13\times$ and the clean data accuracy by $65\%$ on average. Furthermore, prior backdoor detection techniques cannot find QDoor attacks in uncompiled QNN circuits.

摘要: 量子神经网络(QNN)在目标识别、自然语言处理和金融分析等方面取得了成功。为了在噪声中尺度量子(NISQ)计算机上最大限度地提高QNN的精度，近似综合通过减少容易出错的2量子位量子门来修改QNN电路。QNN的成功激发了对手通过后门攻击QNN的动机。然而，由于NISQ计算机上的噪声和近似综合，将为经典神经网络设计的后门自然地移植到QNN上只会产生较低的攻击成功率。现有的基于量子电路的后门不能选择性地攻击一些输入或与QNN电路的所有类型的编码层一起工作。此外，在QNN中很容易检测到移植的后门和基于电路的后门。在本文中，我们提出了一种新颖的隐身后门攻击QDoor，通过武器化未编译的QNN与其合成的QNN之间的么正差异，在近似合成的QNN电路中实现高攻击成功率。QDoor训练一个QNN，使其在有或没有触发器的所有输入上都能正常工作。然而，在近似综合之后，QNN电路总是预测具有预定义类别的触发器的任何输入，同时对于良性输入仍然正常工作。与以往的后门攻击相比，QDoor的攻击成功率平均提高了13倍，干净数据的准确率平均提高了65美元。此外，现有的后门检测技术无法在未编译的QNN电路中发现QDoor攻击。



## **43. Decorrelative Network Architecture for Robust Electrocardiogram Classification**

用于稳健心电分类的解相关网络结构 cs.LG

24 pages, 7 figures

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2207.09031v4) [paper-pdf](http://arxiv.org/pdf/2207.09031v4)

**Authors**: Christopher Wiedeman, Ge Wang

**Abstract**: Artificial intelligence has made great progress in medical data analysis, but the lack of robustness and trustworthiness has kept these methods from being widely deployed. As it is not possible to train networks that are accurate in all scenarios, models must recognize situations where they cannot operate confidently. Bayesian deep learning methods sample the model parameter space to estimate uncertainty, but these parameters are often subject to the same vulnerabilities, which can be exploited by adversarial attacks. We propose a novel ensemble approach based on feature decorrelation and Fourier partitioning for teaching networks diverse complementary features, reducing the chance of perturbation-based fooling. We test our approach on single and multi-channel electrocardiogram classification, and adapt adversarial training and DVERGE into the Bayesian ensemble framework for comparison. Our results indicate that the combination of decorrelation and Fourier partitioning generally maintains performance on unperturbed data while demonstrating superior robustness and uncertainty estimation on projected gradient descent and smooth adversarial attacks of various magnitudes. Furthermore, our approach does not require expensive optimization with adversarial samples, adding much less compute to the training process than adversarial training or DVERGE. These methods can be applied to other tasks for more robust and trustworthy models.

摘要: 人工智能在医疗数据分析方面取得了很大进展，但缺乏健壮性和可信性，阻碍了这些方法的广泛部署。由于不可能训练出在所有情况下都准确的网络，因此模型必须认识到它们不能自信地运行的情况。贝叶斯深度学习方法对模型参数空间进行采样以估计不确定性，但这些参数经常受到相同的漏洞的影响，这可能被对抗性攻击所利用。我们提出了一种新的基于特征去相关和傅立叶划分的集成方法，用于训练网络中不同的互补特征，减少了基于扰动的愚弄的机会。我们在单通道和多通道心电分类上测试了我们的方法，并将对抗性训练和DVERGE应用到贝叶斯集成框架中进行比较。我们的结果表明，解相关和傅立叶划分的组合在保持对未受干扰的数据的性能的同时，对投影梯度下降和平滑的不同幅度的敌意攻击表现出了更好的稳健性和不确定性估计。此外，我们的方法不需要使用对抗性样本进行昂贵的优化，与对抗性训练或DVERGE相比，增加的训练过程的计算量要少得多。这些方法可以应用于其他任务，以获得更健壮和可信的模型。



## **44. TernaryVote: Differentially Private, Communication Efficient, and Byzantine Resilient Distributed Optimization on Heterogeneous Data**

TernaryVote：差异化私有、高效通信和拜占庭弹性的异质数据分布式优化 cs.LG

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.10816v1) [paper-pdf](http://arxiv.org/pdf/2402.10816v1)

**Authors**: Richeng Jin, Yujie Gu, Kai Yue, Xiaofan He, Zhaoyang Zhang, Huaiyu Dai

**Abstract**: Distributed training of deep neural networks faces three critical challenges: privacy preservation, communication efficiency, and robustness to fault and adversarial behaviors. Although significant research efforts have been devoted to addressing these challenges independently, their synthesis remains less explored. In this paper, we propose TernaryVote, which combines a ternary compressor and the majority vote mechanism to realize differential privacy, gradient compression, and Byzantine resilience simultaneously. We theoretically quantify the privacy guarantee through the lens of the emerging f-differential privacy (DP) and the Byzantine resilience of the proposed algorithm. Particularly, in terms of privacy guarantees, compared to the existing sign-based approach StoSign, the proposed method improves the dimension dependence on the gradient size and enjoys privacy amplification by mini-batch sampling while ensuring a comparable convergence rate. We also prove that TernaryVote is robust when less than 50% of workers are blind attackers, which matches that of SIGNSGD with majority vote. Extensive experimental results validate the effectiveness of the proposed algorithm.

摘要: 深度神经网络的分布式训练面临三个关键挑战：隐私保护、通信效率以及对错误和敌对行为的健壮性。虽然已有大量的研究工作致力于独立应对这些挑战，但对它们的合成探索仍然较少。在本文中，我们提出了TernaryVote，它结合了一个三值压缩器和多数投票机制，同时实现了差分隐私、梯度压缩和拜占庭弹性。我们从新出现的f-差分隐私(DP)和算法的拜占庭弹性两个角度对隐私保障进行了理论上的量化。特别是，在隐私保证方面，与现有的基于符号的方法StoSign相比，该方法改善了维度对梯度大小的依赖，并在保证相当的收敛速度的情况下，通过小批量采样获得隐私放大。我们还证明了当只有不到50%的工作人员是盲人攻击者时，TernaryVote是健壮的，这与SIGNSGD的大多数投票结果相匹配。大量的实验结果验证了该算法的有效性。



## **45. Uncertainty, Calibration, and Membership Inference Attacks: An Information-Theoretic Perspective**

不确定性、校准和成员推理攻击：信息论视角 cs.IT

27 pages, 13 figures

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.10686v1) [paper-pdf](http://arxiv.org/pdf/2402.10686v1)

**Authors**: Meiyi Zhu, Caili Guo, Chunyan Feng, Osvaldo Simeone

**Abstract**: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the state-of-the-art likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in which an adaptive prediction set is produced as in conformal prediction. We derive bounds on the advantage of an MIA adversary with the aim of offering insights into the impact of uncertainty and calibration on the effectiveness of MIAs. Simulation results demonstrate that the derived analytical bounds predict well the effectiveness of MIAs.

摘要: 在成员关系推理攻击(MIA)中，攻击者利用典型机器学习模型表现出的过度自信来确定是否使用特定数据点来训练目标模型。在本文中，我们在信息论框架内分析了最新的似然比攻击(LIRA)的性能，该框架允许研究真实数据生成过程中的任意不确定性的影响、有限训练数据集引起的认知不确定性的影响以及目标模型的校准水平。我们比较了三种不同的设置，其中攻击者从目标模型接收信息递减的反馈：置信度向量(CV)披露，其中输出概率向量被释放；真实标签置信度(TLC)披露，其中模型仅提供分配给真实标签的概率；以及决策集(DS)披露，其中产生与保形预测相同的自适应预测集。我们得出了MIA对手的优势界限，目的是对不确定性和校准对MIA有效性的影响提供见解。仿真结果表明，推导出的解析界很好地预测了MIA的有效性。



## **46. Zero-shot sampling of adversarial entities in biomedical question answering**

生物医学问答中对抗性实体的零抽样 cs.CL

20 pages incl. appendix, under review

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.10527v1) [paper-pdf](http://arxiv.org/pdf/2402.10527v1)

**Authors**: R. Patrick Xian, Alex J. Lee, Vincent Wang, Qiming Cui, Russell Ro, Reza Abbasi-Asl

**Abstract**: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. In high-stakes and knowledge-intensive tasks, understanding model vulnerabilities is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples in natural language processing tasks raises questions about their potential guises in other settings. Here, we propose a powerscaled distance-weighted sampling scheme in embedding space to discover diverse adversarial entities as distractors. We demonstrate its advantage over random sampling in adversarial question answering on biomedical topics. Our approach enables the exploration of different regions on the attack surface, which reveals two regimes of adversarial entities that markedly differ in their characteristics. Moreover, we show that the attacks successfully manipulate token-wise Shapley value explanations, which become deceptive in the adversarial setting. Our investigations illustrate the brittleness of domain knowledge in LLMs and reveal a shortcoming of standard evaluations for high-capacity models.

摘要: 大型语言模型(LLM)中参数领域知识的不断深入推动了它们在现实世界应用程序中的快速部署。在高风险和知识密集型任务中，了解模型漏洞对于量化模型预测的可信度和规范其使用至关重要。最近在自然语言处理任务中发现了命名实体作为对抗性例子，这引发了人们对它们在其他环境中潜在伪装的质疑。在这里，我们提出了一种嵌入空间中的加权距离加权抽样方案，以发现不同的敌意实体作为分心者。在生物医学主题的对抗性问答中，我们展示了它比随机抽样的优势。我们的方法能够探索攻击面上的不同区域，这揭示了两个在特征上明显不同的敌对实体制度。此外，我们还证明了攻击成功地操纵了令牌Shapley值解释，这在对抗性环境下变得具有欺骗性。我们的研究表明了LLMS中领域知识的脆性，并揭示了大容量模型的标准评估的缺陷。



## **47. Benchmarking Transferable Adversarial Attacks**

标杆可转移的对抗性攻击 cs.CV

Accepted by NDSS 2024 Workshop

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.00418v3) [paper-pdf](http://arxiv.org/pdf/2402.00418v3)

**Authors**: Zhibo Jin, Jiayu Zhang, Zhiyu Zhu, Huaming Chen

**Abstract**: The robustness of deep learning models against adversarial attacks remains a pivotal concern. This study presents, for the first time, an exhaustive review of the transferability aspect of adversarial attacks. It systematically categorizes and critically evaluates various methodologies developed to augment the transferability of adversarial attacks. This study encompasses a spectrum of techniques, including Generative Structure, Semantic Similarity, Gradient Editing, Target Modification, and Ensemble Approach. Concurrently, this paper introduces a benchmark framework \textit{TAA-Bench}, integrating ten leading methodologies for adversarial attack transferability, thereby providing a standardized and systematic platform for comparative analysis across diverse model architectures. Through comprehensive scrutiny, we delineate the efficacy and constraints of each method, shedding light on their underlying operational principles and practical utility. This review endeavors to be a quintessential resource for both scholars and practitioners in the field, charting the complex terrain of adversarial transferability and setting a foundation for future explorations in this vital sector. The associated codebase is accessible at: https://github.com/KxPlaug/TAA-Bench

摘要: 深度学习模型对敌意攻击的稳健性仍然是一个关键问题。这项研究首次对对抗性攻击的可转移性进行了详尽的回顾。它系统地分类和批判性地评价了为加强对抗性攻击的可转移性而开发的各种方法。这项研究涵盖了一系列技术，包括生成结构、语义相似性、梯度编辑、目标修改和集成方法。同时，本文引入了一个基准框架\TAA-BENCH，集成了十种主流的对抗性攻击可转移性方法，从而为跨不同模型体系结构的比较分析提供了一个标准化和系统化的平台。通过全面的审查，我们描述了每种方法的有效性和制约因素，揭示了它们潜在的操作原理和实用价值。本综述努力成为该领域学者和实践者的典型资源，描绘了对抗性可转移性的复杂地形，并为未来在这一重要领域的探索奠定了基础。相关的代码库可在以下网址访问：https://github.com/KxPlaug/TAA-Bench



## **48. Rethinking Adversarial Policies: A Generalized Attack Formulation and Provable Defense in RL**

对抗性策略的再思考：RL中的广义攻击公式和可证明防御 cs.LG

International Conference on Learning Representations (ICLR) 2024

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2305.17342v3) [paper-pdf](http://arxiv.org/pdf/2305.17342v3)

**Authors**: Xiangyu Liu, Souradip Chakraborty, Yanchao Sun, Furong Huang

**Abstract**: Most existing works focus on direct perturbations to the victim's state/action or the underlying transition dynamics to demonstrate the vulnerability of reinforcement learning agents to adversarial attacks. However, such direct manipulations may not be always realizable. In this paper, we consider a multi-agent setting where a well-trained victim agent $\nu$ is exploited by an attacker controlling another agent $\alpha$ with an \textit{adversarial policy}. Previous models do not account for the possibility that the attacker may only have partial control over $\alpha$ or that the attack may produce easily detectable "abnormal" behaviors. Furthermore, there is a lack of provably efficient defenses against these adversarial policies. To address these limitations, we introduce a generalized attack framework that has the flexibility to model to what extent the adversary is able to control the agent, and allows the attacker to regulate the state distribution shift and produce stealthier adversarial policies. Moreover, we offer a provably efficient defense with polynomial convergence to the most robust victim policy through adversarial training with timescale separation. This stands in sharp contrast to supervised learning, where adversarial training typically provides only \textit{empirical} defenses. Using the Robosumo competition experiments, we show that our generalized attack formulation results in much stealthier adversarial policies when maintaining the same winning rate as baselines. Additionally, our adversarial training approach yields stable learning dynamics and less exploitable victim policies.

摘要: 现有的大多数工作都集中在对受害者状态/动作的直接扰动或潜在的转换动力学上，以证明强化学习代理在对抗性攻击下的脆弱性。然而，这样的直接操纵并不总是可以实现的。在本文中，我们考虑了一个多智能体环境，其中训练有素的受害者智能体$\nu$被攻击者利用，攻击者用对抗策略控制另一个智能体$\α$。以前的模型没有考虑这样一种可能性，即攻击者可能只对$\Alpha$进行了部分控制，或者攻击可能会产生容易检测到的“异常”行为。此外，缺乏针对这些对抗性政策的被证明有效的防御。为了解决这些局限性，我们引入了一个通用的攻击框架，该框架可以灵活地建模对手能够控制代理的程度，并允许攻击者调节状态分布变化并产生更隐蔽的对抗性策略。此外，通过时间尺度分离的对抗性训练，我们提供了一个多项式收敛到最健壮的受害者策略的可证明的有效防御。这与监督学习形成了鲜明的对比，在监督学习中，对抗性训练通常只提供经验上的防御。使用Robosumo竞争实验，我们表明，当保持相同的胜率作为基线时，我们的广义攻击公式导致了更隐蔽的对抗策略。此外，我们的对抗性训练方法产生了稳定的学习动力和较少可利用的受害者政策。



## **49. PPR: Enhancing Dodging Attacks while Maintaining Impersonation Attacks on Face Recognition Systems**

PPR：增强躲避攻击，同时保持对人脸识别系统的模仿攻击 cs.CV

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2401.08903v2) [paper-pdf](http://arxiv.org/pdf/2401.08903v2)

**Authors**: Fengfan Zhou, Heifei Ling, Bangjie Yin, Hui Zheng

**Abstract**: Adversarial Attacks on Face Recognition (FR) encompass two types: impersonation attacks and evasion attacks. We observe that achieving a successful impersonation attack on FR does not necessarily ensure a successful dodging attack on FR in the black-box setting. Introducing a novel attack method named Pre-training Pruning Restoration Attack (PPR), we aim to enhance the performance of dodging attacks whilst avoiding the degradation of impersonation attacks. Our method employs adversarial example pruning, enabling a portion of adversarial perturbations to be set to zero, while tending to maintain the attack performance. By utilizing adversarial example pruning, we can prune the pre-trained adversarial examples and selectively free up certain adversarial perturbations. Thereafter, we embed adversarial perturbations in the pruned area, which enhances the dodging performance of the adversarial face examples. The effectiveness of our proposed attack method is demonstrated through our experimental results, showcasing its superior performance.

摘要: 针对人脸识别(FR)的敌意攻击包括两种类型：模仿攻击和逃避攻击。我们观察到，在黑盒环境下，成功地实现对FR的模仿攻击并不一定确保对FR的成功躲避攻击。引入一种新的攻击方法--预训练剪枝恢复攻击(PPR)，旨在提高躲避攻击的性能，同时避免冒充攻击的降级。该方法采用对抗性样本剪枝，在保持攻击性能的同时，使一部分对抗性扰动被设置为零。通过利用对抗性实例修剪，我们可以修剪预先训练的对抗性实例，并选择性地释放某些对抗性扰动。之后，我们在剪枝区域嵌入对抗性扰动，提高了对抗性人脸样例的躲避性能。实验结果表明，本文提出的攻击方法是有效的，表现出了优越的性能。



## **50. Quantum-Inspired Analysis of Neural Network Vulnerabilities: The Role of Conjugate Variables in System Attacks**

神经网络脆弱性的量子分析：共轭变量在系统攻击中的作用 cs.LG

13 pages, 3 figures

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.10983v1) [paper-pdf](http://arxiv.org/pdf/2402.10983v1)

**Authors**: Jun-Jie Zhang, Deyu Meng

**Abstract**: Neural networks demonstrate inherent vulnerability to small, non-random perturbations, emerging as adversarial attacks. Such attacks, born from the gradient of the loss function relative to the input, are discerned as input conjugates, revealing a systemic fragility within the network structure. Intriguingly, a mathematical congruence manifests between this mechanism and the quantum physics' uncertainty principle, casting light on a hitherto unanticipated interdisciplinarity. This inherent susceptibility within neural network systems is generally intrinsic, highlighting not only the innate vulnerability of these networks but also suggesting potential advancements in the interdisciplinary area for understanding these black-box networks.

摘要: 神经网络对小的、非随机的扰动表现出内在的脆弱性，出现了对抗性攻击。这种攻击产生于损失函数相对于输入的梯度，被识别为输入共轭，揭示了网络结构中的系统脆弱性。有趣的是，这一机制和量子物理学的测不准原理在数学上是一致的，揭示了一种迄今未曾预料到的跨学科。神经网络系统中的这种固有敏感性通常是固有的，这不仅突显了这些网络的固有脆弱性，而且表明了在理解这些黑盒网络的跨学科领域的潜在进步。



