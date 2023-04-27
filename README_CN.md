# Latest Adversarial Attack Papers
**update at 2023-04-27 10:22:00**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Learning Robust Deep Equilibrium Models**

学习稳健的深度均衡模型 cs.LG

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.12707v2) [paper-pdf](http://arxiv.org/pdf/2304.12707v2)

**Authors**: Haoyu Chu, Shikui Wei, Ting Liu, Yao Zhao

**Abstract**: Deep equilibrium (DEQ) models have emerged as a promising class of implicit layer models in deep learning, which abandon traditional depth by solving for the fixed points of a single nonlinear layer. Despite their success, the stability of the fixed points for these models remains poorly understood. Recently, Lyapunov theory has been applied to Neural ODEs, another type of implicit layer model, to confer adversarial robustness. By considering DEQ models as nonlinear dynamic systems, we propose a robust DEQ model named LyaDEQ with guaranteed provable stability via Lyapunov theory. The crux of our method is ensuring the fixed points of the DEQ models are Lyapunov stable, which enables the LyaDEQ models to resist minor initial perturbations. To avoid poor adversarial defense due to Lyapunov-stable fixed points being located near each other, we add an orthogonal fully connected layer after the Lyapunov stability module to separate different fixed points. We evaluate LyaDEQ models on several widely used datasets under well-known adversarial attacks, and experimental results demonstrate significant improvement in robustness. Furthermore, we show that the LyaDEQ model can be combined with other defense methods, such as adversarial training, to achieve even better adversarial robustness.

摘要: 深度平衡(DEQ)模型是深度学习中一类很有前途的隐层模型，它通过求解单个非线性层的不动点来抛弃传统的深度模型。尽管它们取得了成功，但这些模型的固定点的稳定性仍然知之甚少。最近，Lyapunov理论被应用于另一种类型的隐含层模型--神经常微分方程组，以赋予对手健壮性。将DEQ模型视为非线性动态系统，利用Lyapunov理论，提出了一种具有可证明稳定性的鲁棒DEQ模型LyaDEQ。我们方法的关键是确保DEQ模型的不动点是Lyapunov稳定的，这使得LyaDEQ模型能够抵抗微小的初始扰动。为了避免Lyapunov稳定不动点位置较近造成的对抗性差，我们在Lyapunov稳定模后增加了一个正交全连通层来分离不同的不动点。我们在几个广泛使用的数据集上对LyaDEQ模型进行了评估，实验结果表明，LyaDEQ模型在稳健性方面有了显著的提高。此外，我们还证明了LyaDEQ模型可以与其他防御方法相结合，例如对抗训练，以获得更好的对抗健壮性。



## **2. One-vs-the-Rest Loss to Focus on Important Samples in Adversarial Training**

在对抗性训练中专注于重要样本的一对一损失 cs.LG

ICML2023, 26 pages, 19 figures

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2207.10283v3) [paper-pdf](http://arxiv.org/pdf/2207.10283v3)

**Authors**: Sekitoshi Kanai, Shin'ya Yamaguchi, Masanori Yamada, Hiroshi Takahashi, Kentaro Ohno, Yasutoshi Ida

**Abstract**: This paper proposes a new loss function for adversarial training. Since adversarial training has difficulties, e.g., necessity of high model capacity, focusing on important data points by weighting cross-entropy loss has attracted much attention. However, they are vulnerable to sophisticated attacks, e.g., Auto-Attack. This paper experimentally reveals that the cause of their vulnerability is their small margins between logits for the true label and the other labels. Since neural networks classify the data points based on the logits, logit margins should be large enough to avoid flipping the largest logit by the attacks. Importance-aware methods do not increase logit margins of important samples but decrease those of less-important samples compared with cross-entropy loss. To increase logit margins of important samples, we propose switching one-vs-the-rest loss (SOVR), which switches from cross-entropy to one-vs-the-rest loss for important samples that have small logit margins. We prove that one-vs-the-rest loss increases logit margins two times larger than the weighted cross-entropy loss for a simple problem. We experimentally confirm that SOVR increases logit margins of important samples unlike existing methods and achieves better robustness against Auto-Attack than importance-aware methods.

摘要: 本文提出了一种新的对抗性训练损失函数。由于对抗性训练的难度很大，例如需要较高的模型容量，因此通过加权交叉熵损失来关注重要数据点的方法引起了人们的广泛关注。然而，它们很容易受到复杂的攻击，例如自动攻击。本文通过实验揭示了它们易受攻击的原因是真实标签的对数与其他标签的对数之间的差值很小。由于神经网络根据Logit对数据点进行分类，因此Logit边际应该足够大，以避免因攻击而翻转最大的Logit。与交叉熵损失相比，重要性感知方法不会增加重要样本的Logit裕度，但会减少不太重要的样本的Logit裕度。为了提高重要样本的Logit裕度，我们提出了切换一对一损失(SOVR)，对于Logit裕度较小的重要样本，它从交叉熵切换为一对休息损失。我们证明，对于一个简单的问题，一对一的损失比加权的交叉熵损失增加了两倍的Logit边际。实验证实，与现有方法相比，SOVR提高了重要样本的Logit裕度，并且比重要性感知方法获得了更好的对自动攻击的健壮性。



## **3. Improving Adversarial Transferability by Intermediate-level Perturbation Decay**

利用中层扰动衰减提高对手的可转换性 cs.LG

Revision of ICML '23 submission for better clarity

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13410v1) [paper-pdf](http://arxiv.org/pdf/2304.13410v1)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: Intermediate-level attacks that attempt to perturb feature representations following an adversarial direction drastically have shown favorable performance in crafting transferable adversarial examples. Existing methods in this category are normally formulated with two separate stages, where a directional guide is required to be determined at first and the scalar projection of the intermediate-level perturbation onto the directional guide is enlarged thereafter. The obtained perturbation deviates from the guide inevitably in the feature space, and it is revealed in this paper that such a deviation may lead to sub-optimal attack. To address this issue, we develop a novel intermediate-level method that crafts adversarial examples within a single stage of optimization. In particular, the proposed method, named intermediate-level perturbation decay (ILPD), encourages the intermediate-level perturbation to be in an effective adversarial direction and to possess a great magnitude simultaneously. In-depth discussion verifies the effectiveness of our method. Experimental results show that it outperforms state-of-the-arts by large margins in attacking various victim models on ImageNet (+10.07% on average) and CIFAR-10 (+3.88% on average). Our code is at https://github.com/qizhangli/ILPD-attack.

摘要: 中级攻击试图按照对抗性方向彻底扰乱特征表示，在制作可转移的对抗性示例方面表现出了良好的性能。现有的这类方法通常分为两个不同的阶段，首先需要确定一个方向导轨，然后放大中层摄动在该方向导轨上的标量投影。所得到的扰动在特征空间中不可避免地偏离了导引，本文揭示了这种偏离可能导致次优攻击。为了解决这个问题，我们开发了一种新的中级方法，该方法在单个优化阶段内创建对抗性示例。特别是，所提出的方法，称为中层扰动衰变(ILPD)，它鼓励中层扰动朝着有效的对抗性方向发展，同时具有较大的幅度。通过深入讨论，验证了该方法的有效性。实验结果表明，在ImageNet(平均+10.07%)和CIFAR-10(平均+3.88%)上攻击各种受害者模型时，该算法的性能明显优于最新的攻击模型。我们的代码在https://github.com/qizhangli/ILPD-attack.



## **4. Blockchain-based Access Control for Secure Smart Industry Management Systems**

基于区块链的安全智能工业管理系统访问控制 cs.CR

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13379v1) [paper-pdf](http://arxiv.org/pdf/2304.13379v1)

**Authors**: Aditya Pribadi Kalapaaking, Ibrahim Khalil, Mohammad Saidur Rahman, Abdelaziz Bouras

**Abstract**: Smart manufacturing systems involve a large number of interconnected devices resulting in massive data generation. Cloud computing technology has recently gained increasing attention in smart manufacturing systems for facilitating cost-effective service provisioning and massive data management. In a cloud-based manufacturing system, ensuring authorized access to the data is crucial. A cloud platform is operated under a single authority. Hence, a cloud platform is prone to a single point of failure and vulnerable to adversaries. An internal or external adversary can easily modify users' access to allow unauthorized users to access the data. This paper proposes a role-based access control to prevent modification attacks by leveraging blockchain and smart contracts in a cloud-based smart manufacturing system. The role-based access control is developed to determine users' roles and rights in smart contracts. The smart contracts are then deployed to the private blockchain network. We evaluate our solution by utilizing Ethereum private blockchain network to deploy the smart contract. The experimental results demonstrate the feasibility and evaluation of the proposed framework's performance.

摘要: 智能制造系统涉及大量互联设备，产生了海量数据。云计算技术最近在智能制造系统中获得了越来越多的关注，以促进经济高效的服务提供和海量数据管理。在基于云的制造系统中，确保授权访问数据至关重要。云平台是在单一授权下运行的。因此，云平台容易出现单点故障，容易受到对手的攻击。内部或外部对手可以很容易地修改用户的访问权限，以允许未经授权的用户访问数据。在基于云的智能制造系统中，通过利用区块链和智能契约，提出了一种基于角色的访问控制来防止修改攻击。基于角色的访问控制是为了确定用户在智能合同中的角色和权限。然后，智能合同被部署到私有区块链网络。我们通过利用以太私有区块链网络部署智能合同来评估我们的解决方案。实验结果证明了该框架的可行性和性能评估。



## **5. Blockchain-based Federated Learning with SMPC Model Verification Against Poisoning Attack for Healthcare Systems**

基于区块链的联合学习与SMPC模型验证的医疗系统抗中毒攻击 cs.CR

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13360v1) [paper-pdf](http://arxiv.org/pdf/2304.13360v1)

**Authors**: Aditya Pribadi Kalapaaking, Ibrahim Khalil, Xun Yi

**Abstract**: Due to the rising awareness of privacy and security in machine learning applications, federated learning (FL) has received widespread attention and applied to several areas, e.g., intelligence healthcare systems, IoT-based industries, and smart cities. FL enables clients to train a global model collaboratively without accessing their local training data. However, the current FL schemes are vulnerable to adversarial attacks. Its architecture makes detecting and defending against malicious model updates difficult. In addition, most recent studies to detect FL from malicious updates while maintaining the model's privacy have not been sufficiently explored. This paper proposed blockchain-based federated learning with SMPC model verification against poisoning attacks for healthcare systems. First, we check the machine learning model from the FL participants through an encrypted inference process and remove the compromised model. Once the participants' local models have been verified, the models are sent to the blockchain node to be securely aggregated. We conducted several experiments with different medical datasets to evaluate our proposed framework.

摘要: 由于机器学习应用中隐私和安全意识的提高，联合学习(FL)受到了广泛的关注，并应用于智能医疗系统、基于物联网的行业和智能城市等领域。FL使客户能够协作地训练全局模型，而无需访问其本地训练数据。然而，当前的FL方案容易受到对手的攻击。其体系结构使得检测和防御恶意模型更新变得困难。此外，最近关于在保护模型隐私的同时从恶意更新中检测FL的研究还没有得到充分的探索。针对医疗系统的中毒攻击，提出了基于区块链的联合学习和SMPC模型验证。首先，我们通过加密的推理过程检查FL参与者的机器学习模型，并删除被攻破的模型。一旦参与者的本地模型经过验证，模型就会被发送到区块链节点进行安全聚合。我们使用不同的医学数据集进行了几个实验来评估我们提出的框架。



## **6. On the Risks of Stealing the Decoding Algorithms of Language Models**

论窃取语言模型译码算法的风险 cs.LG

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2303.04729v3) [paper-pdf](http://arxiv.org/pdf/2303.04729v3)

**Authors**: Ali Naseh, Kalpesh Krishna, Mohit Iyyer, Amir Houmansadr

**Abstract**: A key component of generating text from modern language models (LM) is the selection and tuning of decoding algorithms. These algorithms determine how to generate text from the internal probability distribution generated by the LM. The process of choosing a decoding algorithm and tuning its hyperparameters takes significant time, manual effort, and computation, and it also requires extensive human evaluation. Therefore, the identity and hyperparameters of such decoding algorithms are considered to be extremely valuable to their owners. In this work, we show, for the first time, that an adversary with typical API access to an LM can steal the type and hyperparameters of its decoding algorithms at very low monetary costs. Our attack is effective against popular LMs used in text generation APIs, including GPT-2 and GPT-3. We demonstrate the feasibility of stealing such information with only a few dollars, e.g., $\$0.8$, $\$1$, $\$4$, and $\$40$ for the four versions of GPT-3.

摘要: 从现代语言模型(LM)生成文本的一个关键组件是解码算法的选择和调整。这些算法确定如何从LM生成的内部概率分布生成文本。选择解码算法和调整其超参数的过程需要大量的时间、人工和计算，还需要广泛的人工评估。因此，这种译码算法的恒等式和超参数被认为对它们的所有者非常有价值。在这项工作中，我们首次证明，具有典型API访问权限的攻击者可以以非常低的金钱成本窃取其解码算法的类型和超参数。我们的攻击对文本生成API中使用的流行LMS有效，包括GPT-2和GPT-3。我们证明了只需几美元即可窃取此类信息的可行性，例如，对于GPT-3的四个版本，仅需$0.8$、$1$、$4$和$40$。



## **7. SHIELD: Thwarting Code Authorship Attribution**

盾牌：挫败代码作者归属 cs.CR

12 pages, 13 figures

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13255v1) [paper-pdf](http://arxiv.org/pdf/2304.13255v1)

**Authors**: Mohammed Abuhamad, Changhun Jung, David Mohaisen, DaeHun Nyang

**Abstract**: Authorship attribution has become increasingly accurate, posing a serious privacy risk for programmers who wish to remain anonymous. In this paper, we introduce SHIELD to examine the robustness of different code authorship attribution approaches against adversarial code examples. We define four attacks on attribution techniques, which include targeted and non-targeted attacks, and realize them using adversarial code perturbation. We experiment with a dataset of 200 programmers from the Google Code Jam competition to validate our methods targeting six state-of-the-art authorship attribution methods that adopt a variety of techniques for extracting authorship traits from source-code, including RNN, CNN, and code stylometry. Our experiments demonstrate the vulnerability of current authorship attribution methods against adversarial attacks. For the non-targeted attack, our experiments demonstrate the vulnerability of current authorship attribution methods against the attack with an attack success rate exceeds 98.5\% accompanied by a degradation of the identification confidence that exceeds 13\%. For the targeted attacks, we show the possibility of impersonating a programmer using targeted-adversarial perturbations with a success rate ranging from 66\% to 88\% for different authorship attribution techniques under several adversarial scenarios.

摘要: 作者身份的归属变得越来越准确，这对希望保持匿名的程序员构成了严重的隐私风险。在本文中，我们引入Shield来检验不同代码作者归属方法对敌意代码示例的稳健性。我们定义了四种基于归因技术的攻击，包括定向攻击和非定向攻击，并使用对抗性代码扰动来实现它们。我们使用来自Google Code Jam竞赛的200名程序员的数据集来验证我们的方法，目标是六种最先进的作者归属方法，这些方法采用了各种技术从源代码中提取作者特征，包括RNN、CNN和代码样式法。我们的实验证明了现有的作者归属方法在抵抗敌意攻击时的脆弱性。对于非目标攻击，我们的实验证明了现有作者归属方法对攻击的脆弱性，攻击成功率超过98.5%，同时身份识别置信度下降超过13%。对于有针对性的攻击，我们证明了使用有针对性的对抗性扰动来模拟程序员的可能性，在几种对抗性场景下，对于不同的作者归属技术，成功率从66\%到88\%不等。



## **8. Generating Adversarial Examples with Task Oriented Multi-Objective Optimization**

面向任务的多目标优化生成对抗性实例 cs.LG

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13229v1) [paper-pdf](http://arxiv.org/pdf/2304.13229v1)

**Authors**: Anh Bui, Trung Le, He Zhao, Quan Tran, Paul Montague, Dinh Phung

**Abstract**: Deep learning models, even the-state-of-the-art ones, are highly vulnerable to adversarial examples. Adversarial training is one of the most efficient methods to improve the model's robustness. The key factor for the success of adversarial training is the capability to generate qualified and divergent adversarial examples which satisfy some objectives/goals (e.g., finding adversarial examples that maximize the model losses for simultaneously attacking multiple models). Therefore, multi-objective optimization (MOO) is a natural tool for adversarial example generation to achieve multiple objectives/goals simultaneously. However, we observe that a naive application of MOO tends to maximize all objectives/goals equally, without caring if an objective/goal has been achieved yet. This leads to useless effort to further improve the goal-achieved tasks, while putting less focus on the goal-unachieved tasks. In this paper, we propose \emph{Task Oriented MOO} to address this issue, in the context where we can explicitly define the goal achievement for a task. Our principle is to only maintain the goal-achieved tasks, while letting the optimizer spend more effort on improving the goal-unachieved tasks. We conduct comprehensive experiments for our Task Oriented MOO on various adversarial example generation schemes. The experimental results firmly demonstrate the merit of our proposed approach. Our code is available at \url{https://github.com/tuananhbui89/TAMOO}.

摘要: 深度学习模型，即使是最先进的模型，也非常容易受到对抗性例子的影响。对抗性训练是提高模型稳健性的最有效方法之一。对抗性训练成功的关键因素是生成满足某些目标/目标的合格的和不同的对抗性范例的能力(例如，找到同时攻击多个模型的最大化模型损失的对抗性范例)。因此，多目标优化(MOO)是敌方实例生成同时实现多个目标/目标的一种自然工具。然而，我们观察到，天真地应用MoO往往会平等地最大化所有目标/目标，而不关心某个目标/目标是否已经实现。这导致进一步改进目标已完成任务的努力是徒劳的，而对目标未完成任务的关注较少。在本文中，我们提出了在明确定义任务的目标实现的情况下，解决这一问题的方法。我们的原则是只维护已实现目标的任务，而让优化器将更多精力花在改进未实现目标的任务上。我们对我们的面向任务的MOO在不同的对抗性实例生成方案上进行了全面的实验。实验结果有力地证明了该方法的优点。我们的代码可在\url{https://github.com/tuananhbui89/TAMOO}.



## **9. Uncovering the Representation of Spiking Neural Networks Trained with Surrogate Gradient**

揭示用代理梯度训练的尖峰神经网络的表示 cs.LG

Published in Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2023-04-25    [abs](http://arxiv.org/abs/2304.13098v1) [paper-pdf](http://arxiv.org/pdf/2304.13098v1)

**Authors**: Yuhang Li, Youngeun Kim, Hyoungseob Park, Priyadarshini Panda

**Abstract**: Spiking Neural Networks (SNNs) are recognized as the candidate for the next-generation neural networks due to their bio-plausibility and energy efficiency. Recently, researchers have demonstrated that SNNs are able to achieve nearly state-of-the-art performance in image recognition tasks using surrogate gradient training. However, some essential questions exist pertaining to SNNs that are little studied: Do SNNs trained with surrogate gradient learn different representations from traditional Artificial Neural Networks (ANNs)? Does the time dimension in SNNs provide unique representation power? In this paper, we aim to answer these questions by conducting a representation similarity analysis between SNNs and ANNs using Centered Kernel Alignment (CKA). We start by analyzing the spatial dimension of the networks, including both the width and the depth. Furthermore, our analysis of residual connections shows that SNNs learn a periodic pattern, which rectifies the representations in SNNs to be ANN-like. We additionally investigate the effect of the time dimension on SNN representation, finding that deeper layers encourage more dynamics along the time dimension. We also investigate the impact of input data such as event-stream data and adversarial attacks. Our work uncovers a host of new findings of representations in SNNs. We hope this work will inspire future research to fully comprehend the representation power of SNNs. Code is released at https://github.com/Intelligent-Computing-Lab-Yale/SNNCKA.

摘要: 尖峰神经网络(SNN)因其生物合理性和能量效率而被认为是下一代神经网络的候选网络。最近，研究人员已经证明，在使用代理梯度训练的图像识别任务中，SNN能够获得几乎最先进的性能。然而，与SNN相关的一些基本问题很少被研究：用代理梯度训练的SNN是否学习不同于传统人工神经网络(ANN)的表示？SNN中的时间维度是否提供了独特的表征能力？在本文中，我们旨在通过使用中心核对齐(CKA)对SNN和ANN之间的表示相似性进行分析来回答这些问题。我们首先分析网络的空间维度，包括宽度和深度。此外，我们对剩余连接的分析表明，SNN学习一个周期性的模式，这将SNN中的表示纠正为类似ANN的表示。此外，我们还研究了时间维度对SNN表示的影响，发现更深的层促进了沿时间维度的更多动力学。我们还研究了输入数据的影响，如事件流数据和对抗性攻击。我们的工作揭示了SNN中表征的一系列新发现。我们希望这项工作将启发未来的研究，以充分理解SNN的表征能力。代码在https://github.com/Intelligent-Computing-Lab-Yale/SNNCKA.上发布



## **10. Improving Robustness Against Adversarial Attacks with Deeply Quantized Neural Networks**

利用深度量化神经网络提高对抗攻击的稳健性 cs.LG

Accepted at IJCNN 2023. 8 pages, 5 figures

**SubmitDate**: 2023-04-25    [abs](http://arxiv.org/abs/2304.12829v1) [paper-pdf](http://arxiv.org/pdf/2304.12829v1)

**Authors**: Ferheen Ayaz, Idris Zakariyya, José Cano, Sye Loong Keoh, Jeremy Singer, Danilo Pau, Mounia Kharbouche-Harrari

**Abstract**: Reducing the memory footprint of Machine Learning (ML) models, particularly Deep Neural Networks (DNNs), is essential to enable their deployment into resource-constrained tiny devices. However, a disadvantage of DNN models is their vulnerability to adversarial attacks, as they can be fooled by adding slight perturbations to the inputs. Therefore, the challenge is how to create accurate, robust, and tiny DNN models deployable on resource-constrained embedded devices. This paper reports the results of devising a tiny DNN model, robust to adversarial black and white box attacks, trained with an automatic quantizationaware training framework, i.e. QKeras, with deep quantization loss accounted in the learning loop, thereby making the designed DNNs more accurate for deployment on tiny devices. We investigated how QKeras and an adversarial robustness technique, Jacobian Regularization (JR), can provide a co-optimization strategy by exploiting the DNN topology and the per layer JR approach to produce robust yet tiny deeply quantized DNN models. As a result, a new DNN model implementing this cooptimization strategy was conceived, developed and tested on three datasets containing both images and audio inputs, as well as compared its performance with existing benchmarks against various white-box and black-box attacks. Experimental results demonstrated that on average our proposed DNN model resulted in 8.3% and 79.5% higher accuracy than MLCommons/Tiny benchmarks in the presence of white-box and black-box attacks on the CIFAR-10 image dataset and a subset of the Google Speech Commands audio dataset respectively. It was also 6.5% more accurate for black-box attacks on the SVHN image dataset.

摘要: 减少机器学习(ML)模型的内存占用，特别是深度神经网络(DNN)，对于使其能够部署到资源受限的微型设备是至关重要的。然而，DNN模型的一个缺点是它们容易受到对抗性攻击，因为它们可以通过在输入中添加轻微的扰动来愚弄它们。因此，面临的挑战是如何创建可在资源受限的嵌入式设备上部署的准确、健壮和微小的DNN模型。设计了一种对黑白盒攻击具有较强鲁棒性的微型DNN模型，该模型使用一个自动量化感知训练框架QKera进行训练，并在学习循环中考虑了深度量化损失，从而使所设计的DNN更适合于部署在微型设备上。我们研究了QKera和一种对抗性健壮性技术雅可比正则化(JR)如何通过利用DNN拓扑和逐层JR方法来提供联合优化策略来产生健壮但微小的深度量化的DNN模型。因此，在三个同时包含图像和音频输入的数据集上构思、开发和测试了一个新的DNN模型，并将其与现有的针对各种白盒和黑盒攻击的基准测试进行了比较。实验结果表明，在CIFAR-10图像数据集和Google语音命令音频数据子集上分别存在白盒和黑盒攻击的情况下，我们提出的DNN模型的准确率比MLCommons/Tiny基准分别高8.3%和79.5%。对于SVHN图像数据集的黑盒攻击，它的准确率也提高了6.5%。



## **11. RobCaps: Evaluating the Robustness of Capsule Networks against Affine Transformations and Adversarial Attacks**

RobCaps：评估胶囊网络对仿射变换和对手攻击的稳健性 cs.LG

To appear at the 2023 International Joint Conference on Neural  Networks (IJCNN), Queensland, Australia, June 2023

**SubmitDate**: 2023-04-25    [abs](http://arxiv.org/abs/2304.03973v2) [paper-pdf](http://arxiv.org/pdf/2304.03973v2)

**Authors**: Alberto Marchisio, Antonio De Marco, Alessio Colucci, Maurizio Martina, Muhammad Shafique

**Abstract**: Capsule Networks (CapsNets) are able to hierarchically preserve the pose relationships between multiple objects for image classification tasks. Other than achieving high accuracy, another relevant factor in deploying CapsNets in safety-critical applications is the robustness against input transformations and malicious adversarial attacks.   In this paper, we systematically analyze and evaluate different factors affecting the robustness of CapsNets, compared to traditional Convolutional Neural Networks (CNNs). Towards a comprehensive comparison, we test two CapsNet models and two CNN models on the MNIST, GTSRB, and CIFAR10 datasets, as well as on the affine-transformed versions of such datasets. With a thorough analysis, we show which properties of these architectures better contribute to increasing the robustness and their limitations. Overall, CapsNets achieve better robustness against adversarial examples and affine transformations, compared to a traditional CNN with a similar number of parameters. Similar conclusions have been derived for deeper versions of CapsNets and CNNs. Moreover, our results unleash a key finding that the dynamic routing does not contribute much to improving the CapsNets' robustness. Indeed, the main generalization contribution is due to the hierarchical feature learning through capsules.

摘要: 胶囊网络(CapsNets)能够在图像分类任务中分层地保持多个对象之间的姿势关系。在安全关键型应用程序中部署CapsNet的另一个相关因素是对输入转换和恶意对手攻击的稳健性。本文系统地分析和评价了影响CapsNets健壮性的各种因素，并与传统卷积神经网络(CNN)进行了比较。为了进行全面的比较，我们在MNIST、GTSRB和CIFAR10数据集以及这些数据集的仿射变换版本上测试了两个CapsNet模型和两个CNN模型。通过深入的分析，我们展示了这些体系结构的哪些属性更有助于提高健壮性及其局限性。总体而言，与具有类似参数的传统CNN相比，CapsNets在对抗对手示例和仿射变换方面实现了更好的健壮性。对于CapsNet和CNN的更深版本，也得出了类似的结论。此外，我们的结果揭示了一个关键发现，即动态路由对提高CapsNet的健壮性没有太大帮助。事实上，主要的泛化贡献是由于通过胶囊进行的分层特征学习。



## **12. Evaluating Adversarial Robustness on Document Image Classification**

文档图像分类中的对抗健壮性评价 cs.CV

The 17th International Conference on Document Analysis and  Recognition

**SubmitDate**: 2023-04-24    [abs](http://arxiv.org/abs/2304.12486v1) [paper-pdf](http://arxiv.org/pdf/2304.12486v1)

**Authors**: Timothée Fronteau, Arnaud Paran, Aymen Shabou

**Abstract**: Adversarial attacks and defenses have gained increasing interest on computer vision systems in recent years, but as of today, most investigations are limited to images. However, many artificial intelligence models actually handle documentary data, which is very different from real world images. Hence, in this work, we try to apply the adversarial attack philosophy on documentary and natural data and to protect models against such attacks. We focus our work on untargeted gradient-based, transfer-based and score-based attacks and evaluate the impact of adversarial training, JPEG input compression and grey-scale input transformation on the robustness of ResNet50 and EfficientNetB0 model architectures. To the best of our knowledge, no such work has been conducted by the community in order to study the impact of these attacks on the document image classification task.

摘要: 近年来，对抗性攻击和防御对计算机视觉系统产生了越来越大的兴趣，但截至目前，大多数调查仅限于图像。然而，许多人工智能模型实际上处理的是纪实数据，这与现实世界的图像有很大不同。因此，在这项工作中，我们试图将对抗性攻击的理念应用于文献和自然数据，并保护模型免受此类攻击。我们的工作集中在基于非目标梯度、基于转移和基于分数的攻击上，并评估了对抗性训练、JPEG输入压缩和灰度输入变换对ResNet50和EfficientNetB0模型架构的健壮性的影响。据我们所知，社区还没有进行过这样的工作，以研究这些攻击对文档图像分类任务的影响。



## **13. StratDef: Strategic Defense Against Adversarial Attacks in ML-based Malware Detection**

StratDef：基于ML的恶意软件检测中对抗攻击的战略防御 cs.LG

**SubmitDate**: 2023-04-24    [abs](http://arxiv.org/abs/2202.07568v6) [paper-pdf](http://arxiv.org/pdf/2202.07568v6)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image recognition domain. The ML-based malware detection domain has received less attention despite its importance. Moreover, most work exploring these defenses has focused on several methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system based on a moving target defense approach. We overcome challenges related to the systematic construction, selection, and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker while minimizing critical aspects in the adversarial ML domain, like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, of the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.

摘要: 多年来，针对机器学习模型的敌意攻击防御的研究大多集中在图像识别领域。基于ML的恶意软件检测领域尽管很重要，但受到的关注较少。此外，大多数探索这些防御措施的工作都集中在几种方法上，但在应用这些方法时没有策略。本文介绍了一种基于移动目标防御方法的战略防御系统StratDef。我们克服了与模型的系统构建、选择和战略使用相关的挑战，以最大限度地提高对手的稳健性。StratDef动态和战略性地选择最佳模型，以增加攻击者的不确定性，同时最小化敌对ML领域的关键方面，如攻击可转移性。我们提供了针对恶意软件检测的机器学习的首次全面防御评估，其中我们的威胁模型探索了不同级别的威胁、攻击者知识、能力和攻击强度。我们表明，即使在面临最大的对手威胁时，StratDef也比其他防御系统表现得更好。我们还表明，在现有的防御系统中，只有少数经过对抗性训练的模型提供了比仅仅使用普通模型更好的保护，但仍然优于StratDef。



## **14. On Adversarial Robustness of Point Cloud Semantic Segmentation**

点云语义分割的对抗性研究 cs.CV

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2112.05871v4) [paper-pdf](http://arxiv.org/pdf/2112.05871v4)

**Authors**: Jiacen Xu, Zhe Zhou, Boyuan Feng, Yufei Ding, Zhou Li

**Abstract**: Recent research efforts on 3D point cloud semantic segmentation (PCSS) have achieved outstanding performance by adopting neural networks. However, the robustness of these complex models have not been systematically analyzed. Given that PCSS has been applied in many safety-critical applications like autonomous driving, it is important to fill this knowledge gap, especially, how these models are affected under adversarial samples. As such, we present a comparative study of PCSS robustness. First, we formally define the attacker's objective under performance degradation and object hiding. Then, we develop new attack by whether to bound the norm. We evaluate different attack options on two datasets and three PCSS models. We found all the models are vulnerable and attacking point color is more effective. With this study, we call the attention of the research community to develop new approaches to harden PCSS models.

摘要: 近年来，基于神经网络的三维点云语义分割(PCSS)的研究取得了显著的效果。然而，这些复杂模型的稳健性还没有得到系统的分析。鉴于PCSS已经被应用于许多安全关键应用，如自动驾驶，填补这一知识空白是很重要的，特别是这些模型在对抗性样本下是如何受到影响的。因此，我们提出了PCSS稳健性的比较研究。首先，在性能下降和对象隐藏的情况下，形式化地定义了攻击者的目标。然后，我们根据是否绑定规范来开发新的攻击。我们在两个数据集和三个PCSS模型上评估了不同的攻击方案。我们发现所有的模型都是易受攻击的，攻击点颜色更有效。通过这项研究，我们呼吁研究界关注开发新的方法来强化PCSS模型。



## **15. Evading DeepFake Detectors via Adversarial Statistical Consistency**

利用对抗性统计一致性规避DeepFake检测器 cs.CV

Accepted by CVPR 2023

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2304.11670v1) [paper-pdf](http://arxiv.org/pdf/2304.11670v1)

**Authors**: Yang Hou, Qing Guo, Yihao Huang, Xiaofei Xie, Lei Ma, Jianjun Zhao

**Abstract**: In recent years, as various realistic face forgery techniques known as DeepFake improves by leaps and bounds,more and more DeepFake detection techniques have been proposed. These methods typically rely on detecting statistical differences between natural (i.e., real) and DeepFakegenerated images in both spatial and frequency domains. In this work, we propose to explicitly minimize the statistical differences to evade state-of-the-art DeepFake detectors. To this end, we propose a statistical consistency attack (StatAttack) against DeepFake detectors, which contains two main parts. First, we select several statistical-sensitive natural degradations (i.e., exposure, blur, and noise) and add them to the fake images in an adversarial way. Second, we find that the statistical differences between natural and DeepFake images are positively associated with the distribution shifting between the two kinds of images, and we propose to use a distribution-aware loss to guide the optimization of different degradations. As a result, the feature distributions of generated adversarial examples is close to the natural images.Furthermore, we extend the StatAttack to a more powerful version, MStatAttack, where we extend the single-layer degradation to multi-layer degradations sequentially and use the loss to tune the combination weights jointly. Comprehensive experimental results on four spatial-based detectors and two frequency-based detectors with four datasets demonstrate the effectiveness of our proposed attack method in both white-box and black-box settings.

摘要: 近年来，随着各种真实感人脸伪造技术DeepFake的突飞猛进，越来越多的DeepFake检测技术被提出。这些方法通常依赖于在空间域和频域中检测自然(即，真实)和深度错误生成的图像之间的统计差异。在这项工作中，我们建议显式地最小化统计差异，以避开最新的DeepFake检测器。为此，我们提出了一种针对DeepFake检测器的统计一致性攻击(StatAttack)，主要包括两个部分。首先，我们选择了几种统计敏感的自然退化(即曝光、模糊和噪声)，并将它们以对抗性的方式添加到虚假图像中。其次，我们发现自然图像和DeepFake图像之间的统计差异与两种图像之间的分布漂移呈正相关，并提出使用分布感知损失来指导不同降质的优化。将StatAttack扩展到一个更强大的版本MStatAttack，将单层退化扩展到多层退化，并利用损失联合调整组合权值。在四个基于空间的检测器和两个基于频率的检测器上对四个数据集的综合实验结果表明，该攻击方法在白盒和黑盒环境下都是有效的。



## **16. Partial-Information, Longitudinal Cyber Attacks on LiDAR in Autonomous Vehicles**

自主车载激光雷达的部分信息、纵向网络攻击 cs.CR

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2303.03470v2) [paper-pdf](http://arxiv.org/pdf/2303.03470v2)

**Authors**: R. Spencer Hallyburton, Qingzhao Zhang, Z. Morley Mao, Miroslav Pajic

**Abstract**: What happens to an autonomous vehicle (AV) if its data are adversarially compromised? Prior security studies have addressed this question through mostly unrealistic threat models, with limited practical relevance, such as white-box adversarial learning or nanometer-scale laser aiming and spoofing. With growing evidence that cyber threats pose real, imminent danger to AVs and cyber-physical systems (CPS) in general, we present and evaluate a novel AV threat model: a cyber-level attacker capable of disrupting sensor data but lacking any situational awareness. We demonstrate that even though the attacker has minimal knowledge and only access to raw data from a single sensor (i.e., LiDAR), she can design several attacks that critically compromise perception and tracking in multi-sensor AVs. To mitigate vulnerabilities and advance secure architectures in AVs, we introduce two improvements for security-aware fusion: a probabilistic data-asymmetry monitor and a scalable track-to-track fusion of 3D LiDAR and monocular detections (T2T-3DLM); we demonstrate that the approaches significantly reduce attack effectiveness. To support objective safety and security evaluations in AVs, we release our security evaluation platform, AVsec, which is built on security-relevant metrics to benchmark AVs on gold-standard longitudinal AV datasets and AV simulators.

摘要: 如果自动驾驶汽车(AV)的数据被相反地泄露，会发生什么？以前的安全研究大多是通过不现实的威胁模型来解决这个问题，实际意义有限，例如白盒对抗性学习或纳米级激光瞄准和欺骗。随着越来越多的证据表明网络威胁对反病毒和网络物理系统(CP)构成真实、紧迫的威胁，我们提出并评估了一种新的反病毒威胁模型：能够破坏传感器数据但缺乏任何态势感知的网络级攻击者。我们证明，即使攻击者只有很少的知识并且只能访问来自单个传感器(即LiDAR)的原始数据，她也可以设计几种严重危害多传感器AVs感知和跟踪的攻击。为了缓解AVS中的漏洞和推进安全体系结构，我们引入了两个安全感知融合的改进：概率数据不对称监测器和可扩展的3D LiDAR和单目检测的航迹到航迹融合(T2T-3DLM)；我们证明这两种方法显著降低了攻击效率。为了支持AVS的客观安全和安保评估，我们发布了我们的安全评估平台AVSEC，该平台构建在与安全相关的指标基础上，以黄金标准的纵向AV数据集和AV模拟器为基准。



## **17. Disco Intelligent Reflecting Surfaces: Active Channel Aging for Fully-Passive Jamming Attacks**

DISCO智能反射面：用于完全无源干扰攻击的主动通道老化 eess.SP

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2302.00415v2) [paper-pdf](http://arxiv.org/pdf/2302.00415v2)

**Authors**: Huan Huang, Ying Zhang, Hongliang Zhang, Yi Cai, A. Lee Swindlehurst, Zhu Han

**Abstract**: Due to the open communications environment in wireless channels, wireless networks are vulnerable to jamming attacks. However, existing approaches for jamming rely on knowledge of the legitimate users' (LUs') channels, extra jamming power, or both. To raise concerns about the potential threats posed by illegitimate intelligent reflecting surfaces (IRSs), we propose an alternative method to launch jamming attacks on LUs without either LU channel state information (CSI) or jamming power. The proposed approach employs an adversarial IRS with random phase shifts, referred to as a "disco" IRS (DIRS), that acts like a "disco ball" to actively age the LUs' channels. Such active channel aging (ACA) interference can be used to launch jamming attacks on multi-user multiple-input single-output (MU-MISO) systems. The proposed DIRS-based fully-passive jammer (FPJ) can jam LUs with no additional jamming power or knowledge of the LU CSI, and it can not be mitigated by classical anti-jamming approaches. A theoretical analysis of the proposed DIRS-based FPJ that provides an evaluation of the DIRS-based jamming attacks is derived. Based on this detailed theoretical analysis, some unique properties of the proposed DIRS-based FPJ can be obtained. Furthermore, a design example of the proposed DIRS-based FPJ based on one-bit quantization of the IRS phases is demonstrated to be sufficient for implementing the jamming attack. In addition, numerical results are provided to show the effectiveness of the derived theoretical analysis and the jamming impact of the proposed DIRS-based FPJ.

摘要: 由于无线信道的开放通信环境，无线网络很容易受到干扰攻击。然而，现有的干扰方法依赖于对合法用户(LU)的信道、额外干扰功率或两者的了解。为了引起人们对非法智能反射面(IRS)潜在威胁的关注，我们提出了一种在没有LU信道状态信息(CSI)或干扰功率的情况下对LU发起干扰攻击的替代方法。所提出的方法采用具有随机相移的对抗性IRS，被称为“迪斯科”IRS(DIRS)，其作用类似于“迪斯科球”来主动老化LU的频道。这种主动信道老化(ACA)干扰可用于对多用户多输入单输出(MU-MISO)系统发起干扰攻击。所提出的基于DIRS的全无源干扰机在不增加干扰功率和不知道逻辑单元CSI的情况下，可以对逻辑单元进行干扰，而且经典的干扰方法不能对其进行抑制。对提出的基于DIRS的干扰攻击进行了理论分析，为基于DIRS的干扰攻击提供了评估。在详细的理论分析的基础上，可以得到基于DIRS的FPJ的一些独特的性质。最后，给出了一个基于IRS相位1比特量化的基于DIRS的FPJ的设计实例，证明了该设计对于实现干扰攻击是足够的。数值结果表明了理论分析的有效性和所提出的基于DIRS的干扰效果。



## **18. StyLess: Boosting the Transferability of Adversarial Examples**

StyLess：提高对抗性例子的可转移性 cs.CV

CVPR 2023

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2304.11579v1) [paper-pdf](http://arxiv.org/pdf/2304.11579v1)

**Authors**: Kaisheng Liang, Bin Xiao

**Abstract**: Adversarial attacks can mislead deep neural networks (DNNs) by adding imperceptible perturbations to benign examples. The attack transferability enables adversarial examples to attack black-box DNNs with unknown architectures or parameters, which poses threats to many real-world applications. We find that existing transferable attacks do not distinguish between style and content features during optimization, limiting their attack transferability. To improve attack transferability, we propose a novel attack method called style-less perturbation (StyLess). Specifically, instead of using a vanilla network as the surrogate model, we advocate using stylized networks, which encode different style features by perturbing an adaptive instance normalization. Our method can prevent adversarial examples from using non-robust style features and help generate transferable perturbations. Comprehensive experiments show that our method can significantly improve the transferability of adversarial examples. Furthermore, our approach is generic and can outperform state-of-the-art transferable attacks when combined with other attack techniques.

摘要: 对抗性攻击可以通过在良性示例中添加不可察觉的扰动来误导深度神经网络(DNN)。攻击的可转移性使得恶意例子能够攻击结构或参数未知的黑盒DNN，这对许多现实世界的应用构成了威胁。我们发现，现有的可转移攻击在优化过程中没有区分风格和内容特征，限制了它们的攻击可转移性。为了提高攻击的可转移性，我们提出了一种新的攻击方法--StyLess攻击方法。具体地说，与其使用普通网络作为代理模型，我们主张使用风格化网络，它通过扰乱自适应实例规范化来编码不同的样式特征。我们的方法可以防止敌意例子使用非健壮的风格特征，并有助于产生可转移的扰动。综合实验表明，该方法能显著提高对抗性实例的可转移性。此外，我们的方法是通用的，当与其他攻击技术相结合时，可以胜过最先进的可转移攻击。



## **19. QuMoS: A Framework for Preserving Security of Quantum Machine Learning Model**

QUMOS：一种量子机器学习模型的安全保护框架 quant-ph

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2304.11511v1) [paper-pdf](http://arxiv.org/pdf/2304.11511v1)

**Authors**: Zhepeng Wang, Jinyang Li, Zhirui Hu, Blake Gage, Elizabeth Iwasawa, Weiwen Jiang

**Abstract**: Security has always been a critical issue in machine learning (ML) applications. Due to the high cost of model training -- such as collecting relevant samples, labeling data, and consuming computing power -- model-stealing attack is one of the most fundamental but vitally important issues. When it comes to quantum computing, such a quantum machine learning (QML) model-stealing attack also exists and it is even more severe because the traditional encryption method can hardly be directly applied to quantum computation. On the other hand, due to the limited quantum computing resources, the monetary cost of training QML model can be even higher than classical ones in the near term. Therefore, a well-tuned QML model developed by a company can be delegated to a quantum cloud provider as a service to be used by ordinary users. In this case, the QML model will be leaked if the cloud provider is under attack. To address such a problem, we propose a novel framework, namely QuMoS, to preserve model security. Instead of applying encryption algorithms, we propose to distribute the QML model to multiple physically isolated quantum cloud providers. As such, even if the adversary in one provider can obtain a partial model, the information of the full model is maintained in the QML service company. Although promising, we observed an arbitrary model design under distributed settings cannot provide model security. We further developed a reinforcement learning-based security engine, which can automatically optimize the model design under the distributed setting, such that a good trade-off between model performance and security can be made. Experimental results on four datasets show that the model design proposed by QuMoS can achieve a close accuracy to the model designed with neural architecture search under centralized settings while providing the highest security than the baselines.

摘要: 安全性一直是机器学习(ML)应用中的一个关键问题。由于模型训练的成本很高--如收集相关样本、标记数据和消耗计算能力--模型窃取攻击是最基本但至关重要的问题之一。当涉及到量子计算时，这样的量子机器学习(QML)模型窃取攻击也存在，而且由于传统的加密方法很难直接应用于量子计算，这种攻击更加严重。另一方面，由于量子计算资源有限，短期内训练QML模型的货币成本可能甚至高于经典模型。因此，一家公司开发的调谐良好的QML模型可以委托给量子云提供商作为普通用户使用的服务。在这种情况下，如果云提供商受到攻击，QML模型将被泄露。为了解决这个问题，我们提出了一个新的框架，即QUMOS，来保护模型的安全性。我们建议将QML模型分发给多个物理上独立的量子云提供商，而不是应用加密算法。因此，即使一个提供商中的对手可以获得部分模型，也在QML服务公司中维护完整模型的信息。尽管前景看好，但我们观察到分布式环境下的任意模型设计不能提供模型安全性。我们进一步开发了一个基于强化学习的安全引擎，该引擎可以在分布式环境下自动优化模型设计，从而在模型性能和安全性之间取得良好的权衡。在四个数据集上的实验结果表明，Qumos提出的模型设计在提供比基线更高的安全性的同时，能够在集中式设置下获得与神经结构搜索设计的模型接近的精度。



## **20. PatchCensor: Patch Robustness Certification for Transformers via Exhaustive Testing**

补丁检查器：通过穷举测试为变压器提供补丁健壮性认证 cs.CV

This paper has been accepted by ACM Transactions on Software  Engineering and Methodology (TOSEM'23) in "Continuous Special Section: AI and  SE." Please include TOSEM for any citations

**SubmitDate**: 2023-04-22    [abs](http://arxiv.org/abs/2111.10481v3) [paper-pdf](http://arxiv.org/pdf/2111.10481v3)

**Authors**: Yuheng Huang, Lei Ma, Yuanchun Li

**Abstract**: Vision Transformer (ViT) is known to be highly nonlinear like other classical neural networks and could be easily fooled by both natural and adversarial patch perturbations. This limitation could pose a threat to the deployment of ViT in the real industrial environment, especially in safety-critical scenarios. In this work, we propose PatchCensor, aiming to certify the patch robustness of ViT by applying exhaustive testing. We try to provide a provable guarantee by considering the worst patch attack scenarios. Unlike empirical defenses against adversarial patches that may be adaptively breached, certified robust approaches can provide a certified accuracy against arbitrary attacks under certain conditions. However, existing robustness certifications are mostly based on robust training, which often requires substantial training efforts and the sacrifice of model performance on normal samples. To bridge the gap, PatchCensor seeks to improve the robustness of the whole system by detecting abnormal inputs instead of training a robust model and asking it to give reliable results for every input, which may inevitably compromise accuracy. Specifically, each input is tested by voting over multiple inferences with different mutated attention masks, where at least one inference is guaranteed to exclude the abnormal patch. This can be seen as complete-coverage testing, which could provide a statistical guarantee on inference at the test time. Our comprehensive evaluation demonstrates that PatchCensor is able to achieve high certified accuracy (e.g. 67.1% on ImageNet for 2%-pixel adversarial patches), significantly outperforming state-of-the-art techniques while achieving similar clean accuracy (81.8% on ImageNet). Meanwhile, our technique also supports flexible configurations to handle different adversarial patch sizes (up to 25%) by simply changing the masking strategy.

摘要: 众所周知，视觉转换器(VIT)像其他经典神经网络一样是高度非线性的，很容易被自然和敌对的补丁扰动所愚弄。这一限制可能会对VIT在实际工业环境中的部署构成威胁，特别是在安全关键的情况下。在这项工作中，我们提出了补丁检查器，旨在通过应用穷举测试来证明VIT的补丁健壮性。我们试图通过考虑最糟糕的补丁攻击场景来提供可证明的保证。与针对可能被适应性破坏的对手补丁的经验防御不同，经过认证的稳健方法可以在某些条件下提供经过认证的准确性，以抵御任意攻击。然而，现有的稳健性认证大多基于健壮性训练，这往往需要大量的训练努力和牺牲正常样本上的模型性能。为了弥合这一差距，补丁检查器试图通过检测异常输入来提高整个系统的稳健性，而不是训练一个健壮的模型，并要求它为每一个输入提供可靠的结果，这可能不可避免地损害精度。具体地说，通过对具有不同突变注意掩码的多个推理进行投票来测试每一输入，其中至少一个推理被保证排除异常补丁。这可以看作是完全覆盖测试，它可以为测试时的推理提供统计保证。我们的综合评估表明，PatchComtor能够达到很高的认证准确率(例如，ImageNet上2%像素的恶意补丁的准确率为67.1%)，显著优于最先进的技术，同时获得类似的干净准确率(ImageNet上的81.8%)。同时，我们的技术还支持灵活的配置，通过简单地更改掩码策略来处理不同的对手补丁大小(高达25%)。



## **21. Universal Adversarial Backdoor Attacks to Fool Vertical Federated Learning in Cloud-Edge Collaboration**

云-边缘协作中对愚人垂直联合学习的普遍对抗性后门攻击 cs.LG

14 pages, 7 figures

**SubmitDate**: 2023-04-22    [abs](http://arxiv.org/abs/2304.11432v1) [paper-pdf](http://arxiv.org/pdf/2304.11432v1)

**Authors**: Peng Chen, Xin Du, Zhihui Lu, Hongfeng Chai

**Abstract**: Vertical federated learning (VFL) is a cloud-edge collaboration paradigm that enables edge nodes, comprising resource-constrained Internet of Things (IoT) devices, to cooperatively train artificial intelligence (AI) models while retaining their data locally. This paradigm facilitates improved privacy and security for edges and IoT devices, making VFL an essential component of Artificial Intelligence of Things (AIoT) systems. Nevertheless, the partitioned structure of VFL can be exploited by adversaries to inject a backdoor, enabling them to manipulate the VFL predictions. In this paper, we aim to investigate the vulnerability of VFL in the context of binary classification tasks. To this end, we define a threat model for backdoor attacks in VFL and introduce a universal adversarial backdoor (UAB) attack to poison the predictions of VFL. The UAB attack, consisting of universal trigger generation and clean-label backdoor injection, is incorporated during the VFL training at specific iterations. This is achieved by alternately optimizing the universal trigger and model parameters of VFL sub-problems. Our work distinguishes itself from existing studies on designing backdoor attacks for VFL, as those require the knowledge of auxiliary information not accessible within the split VFL architecture. In contrast, our approach does not necessitate any additional data to execute the attack. On the LendingClub and Zhongyuan datasets, our approach surpasses existing state-of-the-art methods, achieving up to 100\% backdoor task performance while maintaining the main task performance. Our results in this paper make a major advance to revealing the hidden backdoor risks of VFL, hence paving the way for the future development of secure AIoT.

摘要: 垂直联合学习(VFL)是一种云-边缘协作模式，使包括资源受限的物联网(IoT)设备的边缘节点能够协作训练人工智能(AI)模型，同时将其数据保留在本地。这种模式有助于提高边缘和物联网设备的私密性和安全性，使VFL成为人工物联网(AIoT)系统的重要组件。然而，VFL的分割结构可以被攻击者利用来注入后门，使他们能够操纵VFL预测。在本文中，我们旨在研究VFL在二进制分类任务环境中的脆弱性。为此，我们定义了VFL中后门攻击的威胁模型，并引入了一种通用的对抗性后门(UAB)攻击来毒化VFL的预测。UAB攻击由通用触发器生成和干净标签后门注入组成，在特定迭代的VFL训练期间被纳入。这是通过交替优化VFL子问题的通用触发器和模型参数来实现的。我们的工作不同于现有的为VFL设计后门攻击的研究，因为这些研究需要在分离的VFL体系结构中无法访问的辅助信息的知识。相比之下，我们的方法不需要任何额外的数据来执行攻击。在LendingClub和中原数据集上，我们的方法超过了现有的最先进的方法，在保持主任务性能的同时获得了高达100%的后门任务性能。本文的研究结果在揭示VFL隐藏的后门风险方面取得了重大进展，从而为安全AIoT的未来发展铺平了道路。



## **22. Detecting Adversarial Faces Using Only Real Face Self-Perturbations**

仅利用真实人脸自扰动检测敌方人脸 cs.CV

IJCAI2023

**SubmitDate**: 2023-04-22    [abs](http://arxiv.org/abs/2304.11359v1) [paper-pdf](http://arxiv.org/pdf/2304.11359v1)

**Authors**: Qian Wang, Yongqin Xian, Hefei Ling, Jinyuan Zhang, Xiaorui Lin, Ping Li, Jiazhong Chen, Ning Yu

**Abstract**: Adversarial attacks aim to disturb the functionality of a target system by adding specific noise to the input samples, bringing potential threats to security and robustness when applied to facial recognition systems. Although existing defense techniques achieve high accuracy in detecting some specific adversarial faces (adv-faces), new attack methods especially GAN-based attacks with completely different noise patterns circumvent them and reach a higher attack success rate. Even worse, existing techniques require attack data before implementing the defense, making it impractical to defend newly emerging attacks that are unseen to defenders. In this paper, we investigate the intrinsic generality of adv-faces and propose to generate pseudo adv-faces by perturbing real faces with three heuristically designed noise patterns. We are the first to train an adv-face detector using only real faces and their self-perturbations, agnostic to victim facial recognition systems, and agnostic to unseen attacks. By regarding adv-faces as out-of-distribution data, we then naturally introduce a novel cascaded system for adv-face detection, which consists of training data self-perturbations, decision boundary regularization, and a max-pooling-based binary classifier focusing on abnormal local color aberrations. Experiments conducted on LFW and CelebA-HQ datasets with eight gradient-based and two GAN-based attacks validate that our method generalizes to a variety of unseen adversarial attacks.

摘要: 敌意攻击的目的是通过在输入样本中添加特定的噪声来扰乱目标系统的功能，从而在应用于面部识别系统时给安全性和健壮性带来潜在威胁。虽然现有的防御技术对某些特定的敌方人脸(adv-Faces)的检测准确率很高，但新的攻击方法，特别是具有完全不同噪声模式的基于GAN的攻击，可以绕过它们，达到更高的攻击成功率。更糟糕的是，现有技术在实施防御之前需要攻击数据，因此防御防御者看不到的新出现的攻击是不切实际的。在本文中，我们研究了广告人脸的内在共性，并提出了通过用三种启发式设计的噪声模式来扰动真实人脸来生成伪广告人脸的方法。我们是第一个只使用真实人脸及其自身扰动来训练Adv-Face检测器的人，对受害者面部识别系统不可知，对看不见的攻击不可知。通过将广告人脸看作非分布数据，我们自然地提出了一种新的级联广告人脸检测系统，该系统包括训练数据自扰动、决策边界正则化和基于最大池的二进制分类器，该分类器针对异常局部颜色偏差。在LFW和CelebA-HQ数据集上对8个基于梯度的攻击和2个基于GAN的攻击进行了实验，验证了我们的方法适用于各种看不见的对抗性攻击。



## **23. MAWSEO: Adversarial Wiki Search Poisoning for Illicit Online Promotion**

MAWSEO：恶意维基搜索投毒进行非法在线推广 cs.CR

**SubmitDate**: 2023-04-22    [abs](http://arxiv.org/abs/2304.11300v1) [paper-pdf](http://arxiv.org/pdf/2304.11300v1)

**Authors**: Zilong Lin, Zhengyi Li, Xiaojing Liao, XiaoFeng Wang, Xiaozhong Liu

**Abstract**: As a prominent instance of vandalism edits, Wiki search poisoning for illicit promotion is a cybercrime in which the adversary aims at editing Wiki articles to promote illicit businesses through Wiki search results of relevant queries. In this paper, we report a study that, for the first time, shows that such stealthy blackhat SEO on Wiki can be automated. Our technique, called MAWSEO, employs adversarial revisions to achieve real-world cybercriminal objectives, including rank boosting, vandalism detection evasion, topic relevancy, semantic consistency, user awareness (but not alarming) of promotional content, etc. Our evaluation and user study demonstrate that MAWSEO is able to effectively and efficiently generate adversarial vandalism edits, which can bypass state-of-the-art built-in Wiki vandalism detectors, and also get promotional content through to Wiki users without triggering their alarms. In addition, we investigated potential defense, including coherence based detection and adversarial training of vandalism detection, against our attack in the Wiki ecosystem.

摘要: 作为破坏编辑的一个突出例子，非法推广的维基搜索中毒是一种网络犯罪，对手旨在编辑维基文章，通过相关查询的维基搜索结果来推广非法业务。在这篇文章中，我们报告了一项研究，首次表明维基上这种隐蔽的黑帽SEO可以自动化。我们的技术，称为MAWSEO，使用对抗性修订来实现现实世界的网络犯罪目标，包括排名提升、破坏检测规避、主题相关性、语义一致性、用户对促销内容的感知(但不令人震惊)等。我们的评估和用户研究表明，MAWSEO能够有效和高效地生成对抗性破坏编辑，这可以绕过最先进的内置维基破坏检测器，还可以在不触发维基用户警报的情况下将宣传内容传递给维基用户。此外，我们调查了针对我们在维基生态系统中的攻击的潜在防御，包括基于一致性的检测和恶意检测的对抗性培训。



## **24. Individual Fairness in Bayesian Neural Networks**

贝叶斯神经网络中的个体公平性 cs.LG

**SubmitDate**: 2023-04-21    [abs](http://arxiv.org/abs/2304.10828v1) [paper-pdf](http://arxiv.org/pdf/2304.10828v1)

**Authors**: Alice Doherty, Matthew Wicker, Luca Laurenti, Andrea Patane

**Abstract**: We study Individual Fairness (IF) for Bayesian neural networks (BNNs). Specifically, we consider the $\epsilon$-$\delta$-individual fairness notion, which requires that, for any pair of input points that are $\epsilon$-similar according to a given similarity metrics, the output of the BNN is within a given tolerance $\delta>0.$ We leverage bounds on statistical sampling over the input space and the relationship between adversarial robustness and individual fairness to derive a framework for the systematic estimation of $\epsilon$-$\delta$-IF, designing Fair-FGSM and Fair-PGD as global,fairness-aware extensions to gradient-based attacks for BNNs. We empirically study IF of a variety of approximately inferred BNNs with different architectures on fairness benchmarks, and compare against deterministic models learnt using frequentist techniques. Interestingly, we find that BNNs trained by means of approximate Bayesian inference consistently tend to be markedly more individually fair than their deterministic counterparts.

摘要: 研究了贝叶斯神经网络的个体公平性。具体地说，我们考虑了个体公平性的概念，它要求对于根据给定的相似性度量是$\epon$-相似的任何一对输入点，BNN的输出都在给定的容差$\Delta>0的范围内。$我们利用输入空间上的统计抽样的界以及对手健壮性和个体公平性之间的关系来推导出系统估计$-$\Delta$-IF的框架，并将公平-FGSM和公平-PGD设计为基于梯度攻击的全局公平感知扩展。我们对各种不同结构的近似推断BNN的公平性基准进行了实证研究，并与使用频域技术学习的确定性模型进行了比较。有趣的是，我们发现通过近似贝叶斯推理训练的BNN一致倾向于比它们的确定性对应物更个别地更公平。



## **25. Reliable Representations Make A Stronger Defender: Unsupervised Structure Refinement for Robust GNN**

可靠的表示使防御者更强大：健壮GNN的无监督结构求精 cs.LG

Accepted in KDD2022

**SubmitDate**: 2023-04-21    [abs](http://arxiv.org/abs/2207.00012v4) [paper-pdf](http://arxiv.org/pdf/2207.00012v4)

**Authors**: Kuan Li, Yang Liu, Xiang Ao, Jianfeng Chi, Jinghua Feng, Hao Yang, Qing He

**Abstract**: Benefiting from the message passing mechanism, Graph Neural Networks (GNNs) have been successful on flourish tasks over graph data. However, recent studies have shown that attackers can catastrophically degrade the performance of GNNs by maliciously modifying the graph structure. A straightforward solution to remedy this issue is to model the edge weights by learning a metric function between pairwise representations of two end nodes, which attempts to assign low weights to adversarial edges. The existing methods use either raw features or representations learned by supervised GNNs to model the edge weights. However, both strategies are faced with some immediate problems: raw features cannot represent various properties of nodes (e.g., structure information), and representations learned by supervised GNN may suffer from the poor performance of the classifier on the poisoned graph. We need representations that carry both feature information and as mush correct structure information as possible and are insensitive to structural perturbations. To this end, we propose an unsupervised pipeline, named STABLE, to optimize the graph structure. Finally, we input the well-refined graph into a downstream classifier. For this part, we design an advanced GCN that significantly enhances the robustness of vanilla GCN without increasing the time complexity. Extensive experiments on four real-world graph benchmarks demonstrate that STABLE outperforms the state-of-the-art methods and successfully defends against various attacks.

摘要: 得益于消息传递机制，图神经网络(GNN)已经成功地处理了大量的图数据任务。然而，最近的研究表明，攻击者可以通过恶意修改图结构来灾难性地降低GNN的性能。解决这一问题的一个直接解决方案是通过学习两个末端节点的成对表示之间的度量函数来对边权重进行建模，该度量函数试图为对抗性边分配较低的权重。现有的方法要么使用原始特征，要么使用由监督GNN学习的表示来对边权重进行建模。然而，这两种策略都面临着一些迫在眉睫的问题：原始特征不能表示节点的各种属性(例如结构信息)，而有监督GNN学习的表示可能会受到有毒图上分类器性能较差的影响。我们需要既携带特征信息又尽可能正确的结构信息并对结构扰动不敏感的表示法。为此，我们提出了一种名为STRATE的无监督流水线来优化图的结构。最后，我们将精化后的图输入到下游分类器中。对于这一部分，我们设计了一种改进的GCN，它在不增加时间复杂度的情况下显著增强了普通GCN的健壮性。在四个真实图形基准上的大量实验表明，STRATE的性能优于最先进的方法，并成功地防御了各种攻击。



## **26. Denial-of-Service or Fine-Grained Control: Towards Flexible Model Poisoning Attacks on Federated Learning**

拒绝服务或细粒度控制：对联邦学习的灵活模型中毒攻击 cs.LG

This paper has been accepted by the 32st International Joint  Conference on Artificial Intelligence (IJCAI-23, Main Track)

**SubmitDate**: 2023-04-21    [abs](http://arxiv.org/abs/2304.10783v1) [paper-pdf](http://arxiv.org/pdf/2304.10783v1)

**Authors**: Hangtao Zhang, Zeming Yao, Leo Yu Zhang, Shengshan Hu, Chao Chen, Alan Liew, Zhetao Li

**Abstract**: Federated learning (FL) is vulnerable to poisoning attacks, where adversaries corrupt the global aggregation results and cause denial-of-service (DoS). Unlike recent model poisoning attacks that optimize the amplitude of malicious perturbations along certain prescribed directions to cause DoS, we propose a Flexible Model Poisoning Attack (FMPA) that can achieve versatile attack goals. We consider a practical threat scenario where no extra knowledge about the FL system (e.g., aggregation rules or updates on benign devices) is available to adversaries. FMPA exploits the global historical information to construct an estimator that predicts the next round of the global model as a benign reference. It then fine-tunes the reference model to obtain the desired poisoned model with low accuracy and small perturbations. Besides the goal of causing DoS, FMPA can be naturally extended to launch a fine-grained controllable attack, making it possible to precisely reduce the global accuracy. Armed with precise control, malicious FL service providers can gain advantages over their competitors without getting noticed, hence opening a new attack surface in FL other than DoS. Even for the purpose of DoS, experiments show that FMPA significantly decreases the global accuracy, outperforming six state-of-the-art attacks.

摘要: 联合学习(FL)容易受到中毒攻击，攻击者破坏全局聚合结果并导致拒绝服务(DoS)。与目前的模型中毒攻击不同的是，我们提出了一种灵活的模型中毒攻击(FMPA)，它可以实现多种攻击目标。我们考虑了一个实际的威胁场景，其中没有关于FL系统的额外知识(例如，聚合规则或良性设备上的更新)可供攻击者使用。FMPA利用全球历史信息来构建一个估计器，该估计器预测下一轮全球模型作为良性参考。然后，它微调参考模型，以获得所需的低精度和小扰动的中毒模型。除了造成DoS的目标外，FMPA还可以自然地扩展到发起细粒度的可控攻击，从而有可能精准地降低全局精度。拥有精确控制的恶意FL服务提供商可以在不被注意的情况下获得相对于竞争对手的优势，从而打开了FL除DoS之外的新攻击面。实验表明，即使是在DoS攻击的情况下，FMPA也显著降低了全局的准确率，性能超过了六种最先进的攻击。



## **27. Interpretable and Robust AI in EEG Systems: A Survey**

可解释和稳健的人工智能在脑电系统中的研究进展 eess.SP

**SubmitDate**: 2023-04-21    [abs](http://arxiv.org/abs/2304.10755v1) [paper-pdf](http://arxiv.org/pdf/2304.10755v1)

**Authors**: Xinliang Zhou, Chenyu Liu, Liming Zhai, Ziyu Jia, Cuntai Guan, Yang Liu

**Abstract**: The close coupling of artificial intelligence (AI) and electroencephalography (EEG) has substantially advanced human-computer interaction (HCI) technologies in the AI era. Different from traditional EEG systems, the interpretability and robustness of AI-based EEG systems are becoming particularly crucial. The interpretability clarifies the inner working mechanisms of AI models and thus can gain the trust of users. The robustness reflects the AI's reliability against attacks and perturbations, which is essential for sensitive and fragile EEG signals. Thus the interpretability and robustness of AI in EEG systems have attracted increasing attention, and their research has achieved great progress recently. However, there is still no survey covering recent advances in this field. In this paper, we present the first comprehensive survey and summarize the interpretable and robust AI techniques for EEG systems. Specifically, we first propose a taxonomy of interpretability by characterizing it into three types: backpropagation, perturbation, and inherently interpretable methods. Then we classify the robustness mechanisms into four classes: noise and artifacts, human variability, data acquisition instability, and adversarial attacks. Finally, we identify several critical and unresolved challenges for interpretable and robust AI in EEG systems and further discuss their future directions.

摘要: 人工智能(AI)和脑电(EEG)的紧密结合在AI时代极大地推动了人机交互(HCI)技术的进步。与传统的脑电系统不同，基于人工智能的脑电系统的可解释性和稳健性变得尤为关键。这种可解释性阐明了人工智能模型的内部工作机制，从而可以赢得用户的信任。健壮性反映了人工智能对攻击和扰动的可靠性，这对于敏感和脆弱的脑电信号是必不可少的。因此，人工智能在脑电系统中的可解释性和稳健性受到越来越多的关注，近年来其研究取得了很大进展。然而，目前还没有关于这一领域最新进展的调查。在本文中，我们介绍了第一个全面的综述，并总结了可解释的和健壮的脑电系统人工智能技术。具体地说，我们首先提出了一种可解释性的分类，将其描述为三种类型：反向传播方法、扰动方法和内在可解释方法。然后，我们将健壮性机制分为四类：噪声和伪影、人类可变性、数据获取不稳定性和对抗性攻击。最后，我们确定了脑电系统中可解释的和健壮的人工智能面临的几个关键和尚未解决的挑战，并进一步讨论了它们的未来发展方向。



## **28. Fooling Thermal Infrared Detectors in Physical World**

在现实世界中愚弄热红外探测器 cs.CV

**SubmitDate**: 2023-04-21    [abs](http://arxiv.org/abs/2304.10712v1) [paper-pdf](http://arxiv.org/pdf/2304.10712v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Infrared imaging systems have a vast array of potential applications in pedestrian detection and autonomous driving, and their safety performance is of great concern. However, few studies have explored the safety of infrared imaging systems in real-world settings. Previous research has used physical perturbations such as small bulbs and thermal "QR codes" to attack infrared imaging detectors, but such methods are highly visible and lack stealthiness. Other researchers have used hot and cold blocks to deceive infrared imaging detectors, but this method is limited in its ability to execute attacks from various angles. To address these shortcomings, we propose a novel physical attack called adversarial infrared blocks (AdvIB). By optimizing the physical parameters of the adversarial infrared blocks, this method can execute a stealthy black-box attack on thermal imaging system from various angles. We evaluate the proposed method based on its effectiveness, stealthiness, and robustness. Our physical tests show that the proposed method achieves a success rate of over 80% under most distance and angle conditions, validating its effectiveness. For stealthiness, our method involves attaching the adversarial infrared block to the inside of clothing, enhancing its stealthiness. Additionally, we test the proposed method on advanced detectors, and experimental results demonstrate an average attack success rate of 51.2%, proving its robustness. Overall, our proposed AdvIB method offers a promising avenue for conducting stealthy, effective and robust black-box attacks on thermal imaging system, with potential implications for real-world safety and security applications.

摘要: 红外成像系统在行人检测和自动驾驶方面有着广泛的潜在应用，其安全性能备受关注。然而，很少有研究探讨红外成像系统在现实世界中的安全性。之前的研究曾使用小灯泡和热二维码等物理扰动来攻击红外成像探测器，但这种方法具有高度可见性和隐蔽性。其他研究人员也曾使用冷热块来欺骗红外成像探测器，但这种方法在从不同角度执行攻击的能力方面受到限制。为了克服这些缺陷，我们提出了一种新的物理攻击，称为对抗性红外线拦截(AdvIB)。该方法通过优化敌方红外块的物理参数，可以从多个角度对热成像系统进行隐身黑匣子攻击。我们从有效性、隐蔽性和稳健性三个方面对提出的方法进行了评估。我们的物理测试表明，该方法在大多数距离和角度条件下都达到了80%以上的成功率，验证了其有效性。对于隐蔽性，我们的方法是将敌对的红外线块安装在衣服的内部，增强其隐蔽性。此外，我们在高级检测器上测试了该方法，实验结果表明该方法的平均攻击成功率为51.2%，证明了该方法的健壮性。总体而言，我们提出的AdvIB方法为对热成像系统进行隐蔽、有效和健壮的黑盒攻击提供了一条很有前途的途径，对现实世界的安全和安保应用具有潜在的影响。



## **29. VenoMave: Targeted Poisoning Against Speech Recognition**

VenoMave：针对语音识别的定向中毒 cs.SD

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2010.10682v3) [paper-pdf](http://arxiv.org/pdf/2010.10682v3)

**Authors**: Hojjat Aghakhani, Lea Schönherr, Thorsten Eisenhofer, Dorothea Kolossa, Thorsten Holz, Christopher Kruegel, Giovanni Vigna

**Abstract**: Despite remarkable improvements, automatic speech recognition is susceptible to adversarial perturbations. Compared to standard machine learning architectures, these attacks are significantly more challenging, especially since the inputs to a speech recognition system are time series that contain both acoustic and linguistic properties of speech. Extracting all recognition-relevant information requires more complex pipelines and an ensemble of specialized components. Consequently, an attacker needs to consider the entire pipeline. In this paper, we present VENOMAVE, the first training-time poisoning attack against speech recognition. Similar to the predominantly studied evasion attacks, we pursue the same goal: leading the system to an incorrect and attacker-chosen transcription of a target audio waveform. In contrast to evasion attacks, however, we assume that the attacker can only manipulate a small part of the training data without altering the target audio waveform at runtime. We evaluate our attack on two datasets: TIDIGITS and Speech Commands. When poisoning less than 0.17% of the dataset, VENOMAVE achieves attack success rates of more than 80.0%, without access to the victim's network architecture or hyperparameters. In a more realistic scenario, when the target audio waveform is played over the air in different rooms, VENOMAVE maintains a success rate of up to 73.3%. Finally, VENOMAVE achieves an attack transferability rate of 36.4% between two different model architectures.

摘要: 尽管有了显著的改进，自动语音识别仍然容易受到对抗性干扰的影响。与标准的机器学习体系结构相比，这些攻击更具挑战性，特别是因为语音识别系统的输入是同时包含语音的声学和语言属性的时间序列。提取所有与识别相关的信息需要更复杂的管道和一系列专门的组件。因此，攻击者需要考虑整个管道。本文提出了第一个针对语音识别的训练时间中毒攻击方法VENOMAVE。与主要研究的规避攻击类似，我们追求相同的目标：将系统引导到攻击者选择的目标音频波形的错误转录。然而，与逃避攻击不同的是，我们假设攻击者只能操纵一小部分训练数据，而不会在运行时改变目标音频波形。我们在两个数据集：TIDIGITS和语音命令上评估了我们的攻击。当毒化少于0.17%的数据集时，VENOMAVE实现了80.0%以上的攻击成功率，而无需访问受害者的网络架构或超级参数。在更真实的场景中，当目标音频波形在不同房间的空中播放时，VENOMAVE保持高达73.3%的成功率。最后，VENOMAVE在两种不同的模型架构之间达到了36.4%的攻击可转移率。



## **30. Get Rid Of Your Trail: Remotely Erasing Backdoors in Federated Learning**

摆脱你的痕迹：远程清除联合学习中的后门 cs.LG

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10638v1) [paper-pdf](http://arxiv.org/pdf/2304.10638v1)

**Authors**: Manaar Alam, Hithem Lamri, Michail Maniatakos

**Abstract**: Federated Learning (FL) enables collaborative deep learning training across multiple participants without exposing sensitive personal data. However, the distributed nature of FL and the unvetted participants' data makes it vulnerable to backdoor attacks. In these attacks, adversaries inject malicious functionality into the centralized model during training, leading to intentional misclassifications for specific adversary-chosen inputs. While previous research has demonstrated successful injections of persistent backdoors in FL, the persistence also poses a challenge, as their existence in the centralized model can prompt the central aggregation server to take preventive measures to penalize the adversaries. Therefore, this paper proposes a methodology that enables adversaries to effectively remove backdoors from the centralized model upon achieving their objectives or upon suspicion of possible detection. The proposed approach extends the concept of machine unlearning and presents strategies to preserve the performance of the centralized model and simultaneously prevent over-unlearning of information unrelated to backdoor patterns, making the adversaries stealthy while removing backdoors. To the best of our knowledge, this is the first work that explores machine unlearning in FL to remove backdoors to the benefit of adversaries. Exhaustive evaluation considering image classification scenarios demonstrates the efficacy of the proposed method in efficient backdoor removal from the centralized model, injected by state-of-the-art attacks across multiple configurations.

摘要: 联合学习(FL)支持多个参与者之间的协作深度学习培训，而不会暴露敏感的个人数据。然而，FL的分布式性质和未经审查的参与者的数据使其容易受到后门攻击。在这些攻击中，攻击者在训练期间向集中式模型注入恶意功能，导致故意对特定对手选择的输入进行错误分类。虽然之前的研究已经证明在FL中成功地注入了持久性后门，但持久性也构成了一个挑战，因为它们存在于集中模式中可以促使中央聚合服务器采取预防措施来惩罚对手。因此，本文提出了一种方法，使攻击者能够在达到他们的目标或怀疑可能被检测到时，有效地从集中式模型中删除后门。该方法扩展了机器遗忘的概念，并提出了保持集中式模型性能的策略，同时防止了与后门模式无关的信息的过度遗忘，使对手在移除后门的同时具有隐蔽性。据我们所知，这是第一个探索机器遗忘在FL中的工作，以消除后门，使对手受益。考虑图像分类场景的详尽评估证明了所提出的方法在从集中式模型中高效移除后门的有效性，该模型通过跨多个配置的最先进的攻击注入。



## **31. SoK: Let the Privacy Games Begin! A Unified Treatment of Data Inference Privacy in Machine Learning**

索克：让隐私游戏开始吧！机器学习中数据推理隐私的统一处理 cs.LG

20 pages, to appear in 2023 IEEE Symposium on Security and Privacy

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2212.10986v2) [paper-pdf](http://arxiv.org/pdf/2212.10986v2)

**Authors**: Ahmed Salem, Giovanni Cherubin, David Evans, Boris Köpf, Andrew Paverd, Anshuman Suri, Shruti Tople, Santiago Zanella-Béguelin

**Abstract**: Deploying machine learning models in production may allow adversaries to infer sensitive information about training data. There is a vast literature analyzing different types of inference risks, ranging from membership inference to reconstruction attacks. Inspired by the success of games (i.e., probabilistic experiments) to study security properties in cryptography, some authors describe privacy inference risks in machine learning using a similar game-based style. However, adversary capabilities and goals are often stated in subtly different ways from one presentation to the other, which makes it hard to relate and compose results. In this paper, we present a game-based framework to systematize the body of knowledge on privacy inference risks in machine learning. We use this framework to (1) provide a unifying structure for definitions of inference risks, (2) formally establish known relations among definitions, and (3) to uncover hitherto unknown relations that would have been difficult to spot otherwise.

摘要: 在生产中部署机器学习模型可能会允许攻击者推断有关训练数据的敏感信息。有大量的文献分析了不同类型的推理风险，从成员关系推理到重构攻击。受游戏(即概率实验)在密码学中研究安全属性的成功启发，一些作者使用类似的基于游戏的风格描述了机器学习中的隐私推理风险。然而，对手的能力和目标往往在不同的演示文稿中以微妙的不同方式表达，这使得很难联系和撰写结果。在本文中，我们提出了一个基于博弈的框架来系统化机器学习中隐私推理风险的知识体系。我们使用这个框架来(1)为推理风险的定义提供一个统一的结构，(2)正式地建立定义之间的已知关系，以及(3)揭示否则很难发现的迄今未知的关系。



## **32. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

越多越好(多数)：联邦图神经网络中的后门攻击 cs.CR

15 pages, 13 figures

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2202.03195v5) [paper-pdf](http://arxiv.org/pdf/2202.03195v5)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstract**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for a different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against one defense. We find that both attacks are robust against the investigated defense, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.

摘要: 图神经网络是一类基于深度学习的图域信息处理方法。由于其优越的学习复杂图形数据表示的能力，GNN最近已成为一种广泛使用的图形分析方法。然而，由于隐私问题和监管限制，集中式GNN可能很难适用于数据敏感的情况。联合学习(FL)是一种新兴的技术，是为保护隐私而开发的，当多个参与方需要协作训练共享的全球模型时。虽然一些研究工作已经将FL用于训练GNN(Federated GNN)，但还没有关于其对后门攻击的健壮性的研究。本文通过在联邦GNN中实施两种类型的后门攻击来弥合这一差距：集中式后门攻击(CBA)和分布式后门攻击(DBA)。我们的实验表明，DBA攻击的成功率几乎在所有评估案例中都高于CBA。对于CBA，即使敌方的训练集嵌入了全局触发器，所有局部触发器的攻击成功率也与全局触发器相似。为了进一步探索联邦GNN中两种后门攻击的特性，我们评估了不同客户端数量、触发器大小、中毒强度和触发器密度下的攻击性能。此外，我们还探讨了DBA和CBA对一种防御的健壮性。我们发现，这两种攻击对所调查的防御都是健壮的，因此有必要将联邦GNN中的后门攻击视为一种需要自定义防御的新威胁。



## **33. Byzantine-Resilient Learning Beyond Gradients: Distributing Evolutionary Search**

超越梯度的拜占庭弹性学习：分布式进化搜索 cs.DC

10 pages, 4 listings, 2 theorems

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.13540v1) [paper-pdf](http://arxiv.org/pdf/2304.13540v1)

**Authors**: Andrei Kucharavy, Matteo Monti, Rachid Guerraoui, Ljiljana Dolamic

**Abstract**: Modern machine learning (ML) models are capable of impressive performances. However, their prowess is not due only to the improvements in their architecture and training algorithms but also to a drastic increase in computational power used to train them.   Such a drastic increase led to a growing interest in distributed ML, which in turn made worker failures and adversarial attacks an increasingly pressing concern. While distributed byzantine resilient algorithms have been proposed in a differentiable setting, none exist in a gradient-free setting.   The goal of this work is to address this shortcoming. For that, we introduce a more general definition of byzantine-resilience in ML - the \textit{model-consensus}, that extends the definition of the classical distributed consensus. We then leverage this definition to show that a general class of gradient-free ML algorithms - ($1,\lambda$)-Evolutionary Search - can be combined with classical distributed consensus algorithms to generate gradient-free byzantine-resilient distributed learning algorithms. We provide proofs and pseudo-code for two specific cases - the Total Order Broadcast and proof-of-work leader election.

摘要: 现代机器学习(ML)模型具有令人印象深刻的性能。然而，他们的能力不仅归功于他们在体系结构和训练算法方面的改进，也归功于用于训练他们的计算能力的急剧增加。这种急剧的增长导致了人们对分布式ML的兴趣与日俱增，这反过来又使工人失败和对抗性攻击成为一个日益紧迫的问题。虽然分布式拜占庭弹性算法已经在可微环境下被提出，但没有一种算法在无梯度环境下存在。这项工作的目标就是解决这一缺陷。为此，我们在ML中引入了拜占庭弹性的一个更一般的定义--模型一致性，它扩展了经典的分布式一致性的定义。然后，我们利用这个定义证明了一类通用的无梯度ML算法--($1，\lambda$)-进化搜索-可以与经典的分布式共识算法相结合，生成无梯度的拜占庭弹性分布式学习算法。我们提供了两个具体案例的证明和伪代码-总订单广播和工作证明领导人选举。



## **34. Certified Adversarial Robustness Within Multiple Perturbation Bounds**

多扰动界下的认证对抗稳健性 cs.LG

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10446v1) [paper-pdf](http://arxiv.org/pdf/2304.10446v1)

**Authors**: Soumalya Nandi, Sravanti Addepalli, Harsh Rangwani, R. Venkatesh Babu

**Abstract**: Randomized smoothing (RS) is a well known certified defense against adversarial attacks, which creates a smoothed classifier by predicting the most likely class under random noise perturbations of inputs during inference. While initial work focused on robustness to $\ell_2$ norm perturbations using noise sampled from a Gaussian distribution, subsequent works have shown that different noise distributions can result in robustness to other $\ell_p$ norm bounds as well. In general, a specific noise distribution is optimal for defending against a given $\ell_p$ norm based attack. In this work, we aim to improve the certified adversarial robustness against multiple perturbation bounds simultaneously. Towards this, we firstly present a novel \textit{certification scheme}, that effectively combines the certificates obtained using different noise distributions to obtain optimal results against multiple perturbation bounds. We further propose a novel \textit{training noise distribution} along with a \textit{regularized training scheme} to improve the certification within both $\ell_1$ and $\ell_2$ perturbation norms simultaneously. Contrary to prior works, we compare the certified robustness of different training algorithms across the same natural (clean) accuracy, rather than across fixed noise levels used for training and certification. We also empirically invalidate the argument that training and certifying the classifier with the same amount of noise gives the best results. The proposed approach achieves improvements on the ACR (Average Certified Radius) metric across both $\ell_1$ and $\ell_2$ perturbation bounds.

摘要: 随机平滑(RS)是一种著名的对抗攻击的认证防御方法，它通过在推理过程中输入的随机噪声扰动下预测最可能的类别来创建平滑的分类器。虽然最初的工作集中于使用从高斯分布采样的噪声对$\ell_2$范数扰动的鲁棒性，但随后的工作表明，不同的噪声分布也可以导致对其他$\ell_p$范数界的鲁棒性。一般来说，特定的噪声分布对于防御给定的基于$\ell_p$范数的攻击是最优的。在这项工作中，我们的目标是同时提高认证对手对多个扰动界的稳健性。为此，我们首先提出了一种新的认证方案，该方案有效地结合了使用不同噪声分布获得的证书，从而在多个扰动界下获得最优结果。我们进一步提出了一种新的训练噪声分布和一种正则化训练方案，以同时改进在$1和$2扰动范数下的证明。与以前的工作相反，我们在相同的自然(干净)精度范围内比较不同训练算法的认证稳健性，而不是在用于训练和认证的固定噪声水平上进行比较。我们还从经验上证明了训练和认证具有相同噪声量的分类器可以得到最好的结果的论点。所提出的方法在$\ell_1$和$\ell_2$摄动界上实现了对ACR(平均认证半径)度量的改进。



## **35. An Analysis of the Completion Time of the BB84 Protocol**

对BB84议定书完成时间的分析 cs.PF

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10218v1) [paper-pdf](http://arxiv.org/pdf/2304.10218v1)

**Authors**: Sounak Kar, Jean-Yves Le Boudec

**Abstract**: The BB84 QKD protocol is based on the idea that the sender and the receiver can reconcile a certain fraction of the teleported qubits to detect eavesdropping or noise and decode the rest to use as a private key. Under the present hardware infrastructure, decoherence of quantum states poses a significant challenge to performing perfect or efficient teleportation, meaning that a teleportation-based protocol must be run multiple times to observe success. Thus, performance analyses of such protocols usually consider the completion time, i.e., the time until success, rather than the duration of a single attempt. Moreover, due to decoherence, the success of an attempt is in general dependent on the duration of individual phases of that attempt, as quantum states must wait in memory while the success or failure of a generation phase is communicated to the relevant parties. In this work, we do a performance analysis of the completion time of the BB84 protocol in a setting where the sender and the receiver are connected via a single quantum repeater and the only quantum channel between them does not see any adversarial attack. Assuming certain distributional forms for the generation and communication phases of teleportation, we provide a method to compute the MGF of the completion time and subsequently derive an estimate of the CDF and a bound on the tail probability. This result helps us gauge the (tail) behaviour of the completion time in terms of the parameters characterising the elementary phases of teleportation, without having to run the protocol multiple times. We also provide an efficient simulation scheme to generate the completion time, which relies on expressing the completion time in terms of aggregated teleportation times. We numerically compare our approach with a full-scale simulation and observe good agreement between them.

摘要: BB84量子密钥分发协议基于这样的思想，即发送者和接收者可以协调传送的量子比特的特定部分，以检测窃听或噪声，并解码其余的量子比特作为私钥。在目前的硬件基础设施下，量子态的退相干对执行完美或有效的隐形传态构成了巨大的挑战，这意味着基于隐形传态的协议必须多次运行才能观察到成功。因此，此类协议的性能分析通常考虑完成时间，即成功之前的时间，而不是单次尝试的持续时间。此外，由于退相干，尝试的成功通常取决于尝试的各个阶段的持续时间，因为当生成阶段的成功或失败被传递给相关方时，量子态必须在存储器中等待。在这项工作中，我们对BB84协议的完成时间进行了性能分析，在发送方和接收方通过单个量子中继器连接并且它们之间唯一的量子信道没有任何敌意攻击的情况下。假设隐形传态的产生和通信阶段有一定的分布形式，我们提供了一种计算完成时间的MGF的方法，并由此得到了CDF的估计和尾概率的界。这一结果帮助我们根据表征隐形传态的基本阶段的参数来衡量完成时间的(尾部)行为，而不必多次运行协议。我们还提供了一种高效的仿真方案来生成完成时间，该方案依赖于将完成时间表示为聚合隐形传态时间。我们将我们的方法与全尺寸模拟进行了数值比较，并观察到它们之间有很好的一致性。



## **36. Quantum-secure message authentication via blind-unforgeability**

基于盲不可伪造性的量子安全消息认证 quant-ph

37 pages, v4: Erratum added. We removed a result that had an error in  its proof

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/1803.03761v4) [paper-pdf](http://arxiv.org/pdf/1803.03761v4)

**Authors**: Gorjan Alagic, Christian Majenz, Alexander Russell, Fang Song

**Abstract**: Formulating and designing authentication of classical messages in the presence of adversaries with quantum query access has been a longstanding challenge, as the familiar classical notions of unforgeability do not directly translate into meaningful notions in the quantum setting. A particular difficulty is how to fairly capture the notion of "predicting an unqueried value" when the adversary can query in quantum superposition.   We propose a natural definition of unforgeability against quantum adversaries called blind unforgeability. This notion defines a function to be predictable if there exists an adversary who can use "partially blinded" oracle access to predict values in the blinded region. We support the proposal with a number of technical results. We begin by establishing that the notion coincides with EUF-CMA in the classical setting and go on to demonstrate that the notion is satisfied by a number of simple guiding examples, such as random functions and quantum-query-secure pseudorandom functions. We then show the suitability of blind unforgeability for supporting canonical constructions and reductions. We prove that the "hash-and-MAC" paradigm and the Lamport one-time digital signature scheme are indeed unforgeable according to the definition. To support our analysis, we additionally define and study a new variety of quantum-secure hash functions called Bernoulli-preserving.   Finally, we demonstrate that blind unforgeability is stronger than a previous definition of Boneh and Zhandry [EUROCRYPT '13, CRYPTO '13] in the sense that we can construct an explicit function family which is forgeable by an attack that is recognized by blind-unforgeability, yet satisfies the definition by Boneh and Zhandry.

摘要: 在具有量子查询访问的攻击者在场的情况下，制定和设计经典消息的认证一直是一个长期存在的挑战，因为熟悉的不可伪造性的经典概念不能直接转化为在量子环境中有意义的概念。一个特别的困难是，当对手可以在量子叠加中进行查询时，如何公平地捕捉到“预测一个未被质疑的值”的概念。针对量子对手，我们提出了不可伪造性的自然定义，称为盲不可伪造性。这个概念定义了一个函数是可预测的，如果存在一个对手，该对手可以使用“部分盲目的”先知访问来预测盲区中的值。我们用一系列技术成果支持这项提议。我们首先建立了这个概念在经典设置下与EUF-CMA重合，然后通过一些简单的指导性例子，例如随机函数和量子查询安全伪随机函数，证明了这个概念是满足的。然后，我们证明了盲不可伪造性对于支持规范构造和约简的适用性。根据定义，我们证明了Hash-and-MAC范例和Lamport一次性数字签名方案确实是不可伪造的。为了支持我们的分析，我们还定义和研究了一种新的量子安全散列函数，称为伯努利保持。最后，我们证明了盲不可伪造性比Boneh和Zhandry[Eurocrypt‘13，Crypto’13]的定义更强，因为我们可以构造一个显式函数族，该函数族可以被盲不可伪造性识别的攻击伪造，但满足Boneh和Zhandry的定义。



## **37. Diversifying the High-level Features for better Adversarial Transferability**

使高级功能多样化，以实现更好的对手可转换性 cs.CV

15 pages

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10136v1) [paper-pdf](http://arxiv.org/pdf/2304.10136v1)

**Authors**: Zhiyuan Wang, Zeliang Zhang, Siyuan Liang, Xiaosen Wang

**Abstract**: Given the great threat of adversarial attacks against Deep Neural Networks (DNNs), numerous works have been proposed to boost transferability to attack real-world applications. However, existing attacks often utilize advanced gradient calculation or input transformation but ignore the white-box model. Inspired by the fact that DNNs are over-parameterized for superior performance, we propose diversifying the high-level features (DHF) for more transferable adversarial examples. In particular, DHF perturbs the high-level features by randomly transforming the high-level features and mixing them with the feature of benign samples when calculating the gradient at each iteration. Due to the redundancy of parameters, such transformation does not affect the classification performance but helps identify the invariant features across different models, leading to much better transferability. Empirical evaluations on ImageNet dataset show that DHF could effectively improve the transferability of existing momentum-based attacks. Incorporated into the input transformation-based attacks, DHF generates more transferable adversarial examples and outperforms the baselines with a clear margin when attacking several defense models, showing its generalization to various attacks and high effectiveness for boosting transferability.

摘要: 鉴于针对深度神经网络的敌意攻击的巨大威胁，人们已经提出了许多工作来提高可转移性以攻击真实世界的应用。然而，现有的攻击往往利用先进的梯度计算或输入变换，而忽略了白盒模型。受DNN过度参数化以获得卓越性能这一事实的启发，我们建议将高级特征(DHF)多样化，以获得更多可转移的对抗性示例。特别是，DHF在每次迭代计算梯度时，通过随机变换高层特征并将其与良性样本的特征混合来扰动高层特征。由于参数的冗余性，这种变换不会影响分类性能，但有助于识别不同模型之间的不变特征，从而产生更好的可移植性。在ImageNet数据集上的实验评估表明，DHF能够有效地提高现有动量攻击的可转移性。DHF结合到基于输入变换的攻击中，生成了更多可转移的对抗性实例，在攻击多种防御模型时以明显的优势超过基线，显示了其对各种攻击的通用性和提高可转移性的高效性。



## **38. Towards the Universal Defense for Query-Based Audio Adversarial Attacks**

面向基于查询的音频攻击的通用防御 eess.AS

Submitted to Cybersecurity journal

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10088v1) [paper-pdf](http://arxiv.org/pdf/2304.10088v1)

**Authors**: Feng Guo, Zheng Sun, Yuxuan Chen, Lei Ju

**Abstract**: Recently, studies show that deep learning-based automatic speech recognition (ASR) systems are vulnerable to adversarial examples (AEs), which add a small amount of noise to the original audio examples. These AE attacks pose new challenges to deep learning security and have raised significant concerns about deploying ASR systems and devices. The existing defense methods are either limited in application or only defend on results, but not on process. In this work, we propose a novel method to infer the adversary intent and discover audio adversarial examples based on the AEs generation process. The insight of this method is based on the observation: many existing audio AE attacks utilize query-based methods, which means the adversary must send continuous and similar queries to target ASR models during the audio AE generation process. Inspired by this observation, We propose a memory mechanism by adopting audio fingerprint technology to analyze the similarity of the current query with a certain length of memory query. Thus, we can identify when a sequence of queries appears to be suspectable to generate audio AEs. Through extensive evaluation on four state-of-the-art audio AE attacks, we demonstrate that on average our defense identify the adversary intent with over 90% accuracy. With careful regard for robustness evaluations, we also analyze our proposed defense and its strength to withstand two adaptive attacks. Finally, our scheme is available out-of-the-box and directly compatible with any ensemble of ASR defense models to uncover audio AE attacks effectively without model retraining.

摘要: 最近的研究表明，基于深度学习的自动语音识别(ASR)系统容易受到对抗性样本(AES)的攻击，这些样本会在原始音频样本中添加少量噪声。这些AE攻击对深度学习安全提出了新的挑战，并引发了对部署ASR系统和设备的重大担忧。现有的防御方法要么局限于应用，要么只针对结果进行防御，而不是针对过程进行防御。在这项工作中，我们提出了一种新的方法来推断对手的意图，并发现音频对抗性实例的基础上，AES的生成过程。这种方法的洞察力是基于观察到的：现有的许多音频AE攻击都使用基于查询的方法，这意味着在音频AE生成过程中，攻击者必须向目标ASR模型发送连续的相似查询。受此启发，我们提出了一种采用音频指纹技术的记忆机制来分析当前查询与一定长度的记忆查询的相似度。因此，我们可以识别查询序列何时看起来可疑以生成音频AE。通过对四种最先进的音频AE攻击的广泛评估，我们证明了我们的防御平均识别对手意图的准确率超过90%。在仔细考虑健壮性评估的同时，我们还分析了我们提出的防御方案及其抵抗两种自适应攻击的能力。最后，我们的方案是开箱即用的，并且直接与任何ASR防御模型集成兼容，以有效地发现音频AE攻击，而不需要重新训练模型。



## **39. A Search-Based Testing Approach for Deep Reinforcement Learning Agents**

一种基于搜索的深度强化学习代理测试方法 cs.SE

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2206.07813v3) [paper-pdf](http://arxiv.org/pdf/2206.07813v3)

**Authors**: Amirhossein Zolfagharian, Manel Abdellatif, Lionel Briand, Mojtaba Bagherzadeh, Ramesh S

**Abstract**: Deep Reinforcement Learning (DRL) algorithms have been increasingly employed during the last decade to solve various decision-making problems such as autonomous driving and robotics. However, these algorithms have faced great challenges when deployed in safety-critical environments since they often exhibit erroneous behaviors that can lead to potentially critical errors. One way to assess the safety of DRL agents is to test them to detect possible faults leading to critical failures during their execution. This raises the question of how we can efficiently test DRL policies to ensure their correctness and adherence to safety requirements. Most existing works on testing DRL agents use adversarial attacks that perturb states or actions of the agent. However, such attacks often lead to unrealistic states of the environment. Their main goal is to test the robustness of DRL agents rather than testing the compliance of agents' policies with respect to requirements. Due to the huge state space of DRL environments, the high cost of test execution, and the black-box nature of DRL algorithms, the exhaustive testing of DRL agents is impossible. In this paper, we propose a Search-based Testing Approach of Reinforcement Learning Agents (STARLA) to test the policy of a DRL agent by effectively searching for failing executions of the agent within a limited testing budget. We use machine learning models and a dedicated genetic algorithm to narrow the search towards faulty episodes. We apply STARLA on Deep-Q-Learning agents which are widely used as benchmarks and show that it significantly outperforms Random Testing by detecting more faults related to the agent's policy. We also investigate how to extract rules that characterize faulty episodes of the DRL agent using our search results. Such rules can be used to understand the conditions under which the agent fails and thus assess its deployment risks.

摘要: 在过去的十年中，深度强化学习(DRL)算法被越来越多地用于解决各种决策问题，如自动驾驶和机器人技术。然而，当这些算法部署在安全关键环境中时，它们面临着巨大的挑战，因为它们经常表现出可能导致潜在关键错误的错误行为。评估DRL代理安全性的一种方法是对其进行测试，以检测在其执行期间可能导致严重故障的故障。这提出了一个问题，即我们如何有效地测试DRL政策，以确保它们的正确性和对安全要求的遵守。大多数现有的测试DRL代理的工作使用对抗性攻击，扰乱代理的状态或动作。然而，这样的攻击往往会导致不切实际的环境状况。他们的主要目标是测试DRL代理的健壮性，而不是测试代理策略与需求的符合性。由于DRL环境的状态空间巨大、测试执行成本高以及DRL算法的黑箱性质，对DRL代理进行穷举测试是不可能的。在本文中，我们提出了一种基于搜索的强化学习代理测试方法(STARLA)，通过在有限的测试预算内有效地搜索代理的失败执行来测试DRL代理的策略。我们使用机器学习模型和专门的遗传算法将搜索范围缩小到故障剧集。我们将Starla应用于被广泛用作基准测试的Deep-Q-Learning代理上，结果表明它在检测到更多与代理策略相关的错误方面明显优于随机测试。我们还研究了如何使用搜索结果提取表征DRL代理故障情节的规则。此类规则可用于了解代理发生故障的条件，从而评估其部署风险。



## **40. Quantifying the Preferential Direction of the Model Gradient in Adversarial Training With Projected Gradient Descent**

用投影梯度下降法量化对抗性训练中模型梯度的优先方向 stat.ML

This paper was published in Pattern Recognition

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2009.04709v5) [paper-pdf](http://arxiv.org/pdf/2009.04709v5)

**Authors**: Ricardo Bigolin Lanfredi, Joyce D. Schroeder, Tolga Tasdizen

**Abstract**: Adversarial training, especially projected gradient descent (PGD), has proven to be a successful approach for improving robustness against adversarial attacks. After adversarial training, gradients of models with respect to their inputs have a preferential direction. However, the direction of alignment is not mathematically well established, making it difficult to evaluate quantitatively. We propose a novel definition of this direction as the direction of the vector pointing toward the closest point of the support of the closest inaccurate class in decision space. To evaluate the alignment with this direction after adversarial training, we apply a metric that uses generative adversarial networks to produce the smallest residual needed to change the class present in the image. We show that PGD-trained models have a higher alignment than the baseline according to our definition, that our metric presents higher alignment values than a competing metric formulation, and that enforcing this alignment increases the robustness of models.

摘要: 对抗性训练，特别是投影梯度下降(PGD)，已被证明是提高对抗攻击的稳健性的一种成功方法。经过对抗性训练后，模型相对于其输入的梯度具有优先的方向。然而，对齐的方向在数学上没有很好的确定，因此很难进行定量评估。我们提出了一种新的方向定义，即向量指向决策空间中最接近的不准确类的支持度的最近点的方向。为了在对抗性训练后评估与这一方向的一致性，我们应用了一种度量，该度量使用生成性对抗性网络来产生改变图像中存在的类别所需的最小残差。我们表明，根据我们的定义，PGD训练的模型具有比基线更高的比对，我们的指标比竞争指标公式提供了更高的比对值，并且强制执行这种比对提高了模型的稳健性。



## **41. Jedi: Entropy-based Localization and Removal of Adversarial Patches**

绝地：基于熵的敌方补丁定位与移除 cs.CR

9 pages, 11 figures. To appear in CVPR 2023

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10029v1) [paper-pdf](http://arxiv.org/pdf/2304.10029v1)

**Authors**: Bilel Tarchoun, Anouar Ben Khalifa, Mohamed Ali Mahjoub, Nael Abu-Ghazaleh, Ihsen Alouani

**Abstract**: Real-world adversarial physical patches were shown to be successful in compromising state-of-the-art models in a variety of computer vision applications. Existing defenses that are based on either input gradient or features analysis have been compromised by recent GAN-based attacks that generate naturalistic patches. In this paper, we propose Jedi, a new defense against adversarial patches that is resilient to realistic patch attacks. Jedi tackles the patch localization problem from an information theory perspective; leverages two new ideas: (1) it improves the identification of potential patch regions using entropy analysis: we show that the entropy of adversarial patches is high, even in naturalistic patches; and (2) it improves the localization of adversarial patches, using an autoencoder that is able to complete patch regions from high entropy kernels. Jedi achieves high-precision adversarial patch localization, which we show is critical to successfully repair the images. Since Jedi relies on an input entropy analysis, it is model-agnostic, and can be applied on pre-trained off-the-shelf models without changes to the training or inference of the protected models. Jedi detects on average 90% of adversarial patches across different benchmarks and recovers up to 94% of successful patch attacks (Compared to 75% and 65% for LGS and Jujutsu, respectively).

摘要: 现实世界中的对抗性物理补丁被证明在各种计算机视觉应用中成功地折衷了最先进的模型。现有的基于输入梯度或特征分析的防御已经被最近基于GaN的攻击所破坏，这些攻击产生了自然主义的补丁。在这篇文章中，我们提出了一种新的防御对手补丁的绝地，它对现实的补丁攻击具有弹性。绝地从信息论的角度解决了补丁定位问题；利用了两个新的想法：(1)它利用熵分析改进了潜在补丁区域的识别：我们证明了对抗性补丁的熵很高，即使在自然补丁中也是如此；(2)它改进了对抗性补丁的定位，使用了能够从高熵内核完成补丁区域的自动编码器。绝地实现了高精度的对抗性补丁定位，我们证明这是成功修复图像的关键。由于绝地依赖于输入熵分析，它与模型无关，可以应用于预先训练的现成模型，而不需要改变受保护模型的训练或推断。绝地平均可以通过不同的基准检测90%的敌方补丁，并恢复高达94%的成功补丁攻击(相比之下，LGS和Jujutsu的这一比例分别为75%和65%)。



## **42. Fundamental Limitations of Alignment in Large Language Models**

大型语言模型中对齐的基本限制 cs.CL

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.11082v1) [paper-pdf](http://arxiv.org/pdf/2304.11082v1)

**Authors**: Yotam Wolf, Noam Wies, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback increase the LLM's proneness to being prompted into the undesired behaviors. Moreover, we include the notion of personas in our BEB framework, and find that behaviors which are generally very unlikely to be exhibited by the model can be brought to the front by prompting the model to behave as specific persona. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.

摘要: 开发与人类交互的语言模型的一个重要方面是使他们的行为对人类用户有用而无害。这通常是通过调整模型来实现的，这种方式增强了期望的行为，抑制了不期望的行为，这一过程称为对齐。在本文中，我们提出了一种名为行为期望界限(BEB)的理论方法，它允许我们正式地研究大型语言模型中对齐的几个固有特征和限制。重要的是，我们证明了对于模型表现出的有限概率的任何行为，都存在可以触发模型输出该行为的提示，其概率随着提示的长度增加而增加。这意味着，任何减弱不受欢迎的行为但不能完全消除它的对准过程，在对抗提示攻击时都是不安全的。此外，我们的框架暗示了一种机制，通过这种机制，领先的对齐方法，如来自人类反馈的强化学习，增加了LLM被提示进入不希望看到的行为的倾向。此外，我们在我们的BEB框架中包括了人物角色的概念，并发现通过促使模型表现为特定的人物角色，通常不太可能在模型中表现的行为可以被带到前面。这一理论结果正在由所谓的当代“聊天GPT越狱”大规模实验证明，在这种情况下，敌对用户通过触发LLM充当恶意角色来欺骗LLM打破其对齐护栏。我们的结果暴露了LLM对齐方面的根本限制，并将设计可靠的机制以确保人工智能安全的必要性放在了首位。



## **43. GREAT Score: Global Robustness Evaluation of Adversarial Perturbation using Generative Models**

高分：使用生成模型对对抗性扰动进行全局稳健性评估 cs.LG

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09875v1) [paper-pdf](http://arxiv.org/pdf/2304.09875v1)

**Authors**: Li Zaitang, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Current studies on adversarial robustness mainly focus on aggregating local robustness results from a set of data samples to evaluate and rank different models. However, the local statistics may not well represent the true global robustness of the underlying unknown data distribution. To address this challenge, this paper makes the first attempt to present a new framework, called GREAT Score , for global robustness evaluation of adversarial perturbation using generative models. Formally, GREAT Score carries the physical meaning of a global statistic capturing a mean certified attack-proof perturbation level over all samples drawn from a generative model. For finite-sample evaluation, we also derive a probabilistic guarantee on the sample complexity and the difference between the sample mean and the true mean. GREAT Score has several advantages: (1) Robustness evaluations using GREAT Score are efficient and scalable to large models, by sparing the need of running adversarial attacks. In particular, we show high correlation and significantly reduced computation cost of GREAT Score when compared to the attack-based model ranking on RobustBench (Croce,et. al. 2021). (2) The use of generative models facilitates the approximation of the unknown data distribution. In our ablation study with different generative adversarial networks (GANs), we observe consistency between global robustness evaluation and the quality of GANs. (3) GREAT Score can be used for remote auditing of privacy-sensitive black-box models, as demonstrated by our robustness evaluation on several online facial recognition services.

摘要: 目前关于对抗稳健性的研究主要集中在从一组数据样本中聚集局部稳健性结果来评估和排序不同的模型。然而，局部统计可能不能很好地代表潜在未知数据分布的真实全局稳健性。为了应对这一挑战，本文首次尝试提出了一种新的框架，称为Great Score，用于利用产生式模型评估对抗扰动的全局稳健性。在形式上，高分具有全球统计的物理意义，该统计捕获来自生成模型的所有样本的平均经认证的防攻击扰动水平。对于有限样本评价，我们还得到了样本复杂度和样本均值与真均值之差的概率保证。Great Score有几个优点：(1)使用Great Score进行健壮性评估是高效的，并且可以扩展到大型模型，因为它避免了运行对抗性攻击的需要。特别是，与基于攻击的模型排名相比，我们表现出了高度的相关性和显著的降低了计算开销。艾尔2021年)。(2)生成模型的使用有利于未知数据分布的近似。在我们对不同生成对抗网络(GANS)的消融研究中，我们观察到全局健壮性评估与GANS质量之间的一致性。(3)Great Score可以用于隐私敏感的黑盒模型的远程审计，我们在几种在线人脸识别服务上的健壮性评估证明了这一点。



## **44. Experimental Certification of Quantum Transmission via Bell's Theorem**

用贝尔定理进行量子传输的实验验证 quant-ph

34 pages, 14 figures

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09605v1) [paper-pdf](http://arxiv.org/pdf/2304.09605v1)

**Authors**: Simon Neves, Laura dos Santos Martins, Verena Yacoub, Pascal Lefebvre, Ivan Supic, Damian Markham, Eleni Diamanti

**Abstract**: Quantum transmission links are central elements in essentially all implementations of quantum information protocols. Emerging progress in quantum technologies involving such links needs to be accompanied by appropriate certification tools. In adversarial scenarios, a certification method can be vulnerable to attacks if too much trust is placed on the underlying system. Here, we propose a protocol in a device independent framework, which allows for the certification of practical quantum transmission links in scenarios where minimal assumptions are made about the functioning of the certification setup. In particular, we take unavoidable transmission losses into account by modeling the link as a completely-positive trace-decreasing map. We also, crucially, remove the assumption of independent and identically distributed samples, which is known to be incompatible with adversarial settings. Finally, in view of the use of the certified transmitted states for follow-up applications, our protocol moves beyond certification of the channel to allow us to estimate the quality of the transmitted state itself. To illustrate the practical relevance and the feasibility of our protocol with currently available technology we provide an experimental implementation based on a state-of-the-art polarization entangled photon pair source in a Sagnac configuration and analyze its robustness for realistic losses and errors.

摘要: 量子传输链路是几乎所有量子信息协议实现的核心要素。涉及这种联系的量子技术的新进展需要伴随着适当的认证工具。在对抗性场景中，如果对底层系统的信任过高，则认证方法可能容易受到攻击。在这里，我们提出了一个独立于设备的框架中的协议，它允许在对认证设置的功能做出最小假设的情况下对实际量子传输链路进行认证。特别地，我们通过将链路建模为完全正迹递减映射来考虑不可避免的传输损耗。至关重要的是，我们还取消了独立和同分布样本的假设，这是已知与对抗性设置不兼容的。最后，考虑到后续应用对认证的传输状态的使用，我们的协议超越了对信道的认证，允许我们估计传输状态本身的质量。为了说明我们的协议与现有技术的实际相关性和可行性，我们提供了一个基于Sagnac结构中最先进的偏振纠缠光子对源的实验实现，并分析了其对现实损失和错误的稳健性。



## **45. Masked Language Model Based Textual Adversarial Example Detection**

基于掩蔽语言模型的文本对抗性实例检测 cs.CR

13 pages,3 figures

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.08767v2) [paper-pdf](http://arxiv.org/pdf/2304.08767v2)

**Authors**: Xiaomei Zhang, Zhaoxi Zhang, Qi Zhong, Xufei Zheng, Yanjun Zhang, Shengshan Hu, Leo Yu Zhang

**Abstract**: Adversarial attacks are a serious threat to the reliable deployment of machine learning models in safety-critical applications. They can misguide current models to predict incorrectly by slightly modifying the inputs. Recently, substantial work has shown that adversarial examples tend to deviate from the underlying data manifold of normal examples, whereas pre-trained masked language models can fit the manifold of normal NLP data. To explore how to use the masked language model in adversarial detection, we propose a novel textual adversarial example detection method, namely Masked Language Model-based Detection (MLMD), which can produce clearly distinguishable signals between normal examples and adversarial examples by exploring the changes in manifolds induced by the masked language model. MLMD features a plug and play usage (i.e., no need to retrain the victim model) for adversarial defense and it is agnostic to classification tasks, victim model's architectures, and to-be-defended attack methods. We evaluate MLMD on various benchmark textual datasets, widely studied machine learning models, and state-of-the-art (SOTA) adversarial attacks (in total $3*4*4 = 48$ settings). Experimental results show that MLMD can achieve strong performance, with detection accuracy up to 0.984, 0.967, and 0.901 on AG-NEWS, IMDB, and SST-2 datasets, respectively. Additionally, MLMD is superior, or at least comparable to, the SOTA detection defenses in detection accuracy and F1 score. Among many defenses based on the off-manifold assumption of adversarial examples, this work offers a new angle for capturing the manifold change. The code for this work is openly accessible at \url{https://github.com/mlmddetection/MLMDdetection}.

摘要: 对抗性攻击对机器学习模型在安全关键型应用中的可靠部署构成严重威胁。他们可以通过稍微修改输入来误导当前的模型进行错误的预测。最近的大量工作表明，对抗性例子往往偏离正常例子的底层数据流形，而预先训练的掩蔽语言模型可以适应正常NLP数据的流形。为了探索掩蔽语言模型在对抗性检测中的应用，我们提出了一种新的文本对抗性实例检测方法，即基于掩蔽语言模型的检测方法(MLMD)，该方法通过研究掩蔽语言模型引起的流形变化来产生能够清晰区分正常例子和对抗性例子的信号。MLMD具有即插即用的特点(即，不需要重新训练受害者模型)用于对抗防御，并且它与分类任务、受害者模型的体系结构和要防御的攻击方法无关。我们在各种基准文本数据集、广泛研究的机器学习模型和最先进的(SOTA)对手攻击(总计$3*4*4=48$设置)上评估MLMD。实验结果表明，该算法在AG-NEWS、IMDB和SST-2数据集上的检测准确率分别达到0.984、0.967和0.901。此外，MLMD在检测精度和F1得分方面优于SOTA检测防御，或至少与SOTA检测防御相当。在许多基于对抗性例子的非流形假设的防御中，这项工作为捕捉流形变化提供了一个新的角度。这项工作的代码可以在\url{https://github.com/mlmddetection/MLMDdetection}.上公开访问



## **46. Understanding Overfitting in Adversarial Training via Kernel Regression**

用核回归方法理解对抗性训练中的过度适应 stat.ML

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.06326v2) [paper-pdf](http://arxiv.org/pdf/2304.06326v2)

**Authors**: Teng Zhang, Kang Li

**Abstract**: Adversarial training and data augmentation with noise are widely adopted techniques to enhance the performance of neural networks. This paper investigates adversarial training and data augmentation with noise in the context of regularized regression in a reproducing kernel Hilbert space (RKHS). We establish the limiting formula for these techniques as the attack and noise size, as well as the regularization parameter, tend to zero. Based on this limiting formula, we analyze specific scenarios and demonstrate that, without appropriate regularization, these two methods may have larger generalization error and Lipschitz constant than standard kernel regression. However, by selecting the appropriate regularization parameter, these two methods can outperform standard kernel regression and achieve smaller generalization error and Lipschitz constant. These findings support the empirical observations that adversarial training can lead to overfitting, and appropriate regularization methods, such as early stopping, can alleviate this issue.

摘要: 对抗性训练和带噪声的数据增强是提高神经网络性能的广泛采用的技术。在再生核Hilbert空间(RKHS)的正则化回归背景下，研究了带噪声的对抗性训练和数据增强问题。当攻击和噪声大小以及正则化参数趋于零时，我们建立了这些技术的极限公式。基于这一极限公式，我们分析了具体的情形，并证明了在没有适当的正则化的情况下，这两种方法可能具有比标准核回归更大的泛化误差和Lipschitz常数。然而，通过选择合适的正则化参数，这两种方法都可以获得比标准核回归更好的性能，并获得更小的泛化误差和Lipschitz常数。这些发现支持了经验观察，即对抗性训练会导致过度适应，而适当的正规化方法，如提前停止，可以缓解这一问题。



## **47. Secure Split Learning against Property Inference, Data Reconstruction, and Feature Space Hijacking Attacks**

抗属性推理、数据重构和特征空间劫持攻击的安全分裂学习 cs.LG

23 pages

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09515v1) [paper-pdf](http://arxiv.org/pdf/2304.09515v1)

**Authors**: Yunlong Mao, Zexi Xin, Zhenyu Li, Jue Hong, Qingyou Yang, Sheng Zhong

**Abstract**: Split learning of deep neural networks (SplitNN) has provided a promising solution to learning jointly for the mutual interest of a guest and a host, which may come from different backgrounds, holding features partitioned vertically. However, SplitNN creates a new attack surface for the adversarial participant, holding back its practical use in the real world. By investigating the adversarial effects of highly threatening attacks, including property inference, data reconstruction, and feature hijacking attacks, we identify the underlying vulnerability of SplitNN and propose a countermeasure. To prevent potential threats and ensure the learning guarantees of SplitNN, we design a privacy-preserving tunnel for information exchange between the guest and the host. The intuition is to perturb the propagation of knowledge in each direction with a controllable unified solution. To this end, we propose a new activation function named R3eLU, transferring private smashed data and partial loss into randomized responses in forward and backward propagations, respectively. We give the first attempt to secure split learning against three threatening attacks and present a fine-grained privacy budget allocation scheme. The analysis proves that our privacy-preserving SplitNN solution provides a tight privacy budget, while the experimental results show that our solution performs better than existing solutions in most cases and achieves a good tradeoff between defense and model usability.

摘要: 深度神经网络的分裂学习(SplitNN)为来自不同背景、拥有垂直分割特征的宾主共同兴趣的联合学习提供了一种很有前途的解决方案。然而，SplitNN为对手参与者创造了一个新的攻击面，阻碍了其在现实世界中的实际应用。通过研究高威胁性攻击的对抗性，包括属性推断、数据重构和特征劫持攻击，我们识别了SplitNN的潜在脆弱性，并提出了对策。为了防止潜在的威胁，并保证SplitNN的学习保证，我们设计了一条隐私保护隧道，用于客户和主机之间的信息交换。直觉是用一种可控的统一解决方案扰乱知识在各个方向的传播。为此，我们提出了一种新的激活函数R3eLU，在前向传播和后向传播中分别将私有粉碎数据和部分丢失转移到随机响应中。我们首次尝试保护分裂学习不受三种威胁攻击，并提出了一种细粒度的隐私预算分配方案。分析证明，我们的隐私保护SplitNN方案提供了较少的隐私预算，而实验结果表明，我们的方案在大多数情况下都比现有的方案性能更好，并在防御性和模型可用性之间取得了良好的折衷。



## **48. Maybenot: A Framework for Traffic Analysis Defenses**

Maybenot：一种流量分析防御框架 cs.CR

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09510v1) [paper-pdf](http://arxiv.org/pdf/2304.09510v1)

**Authors**: Tobias Pulls

**Abstract**: End-to-end encryption is a powerful tool for protecting the privacy of Internet users. Together with the increasing use of technologies such as Tor, VPNs, and encrypted messaging, it is becoming increasingly difficult for network adversaries to monitor and censor Internet traffic. One remaining avenue for adversaries is traffic analysis: the analysis of patterns in encrypted traffic to infer information about the users and their activities. Recent improvements using deep learning have made traffic analysis attacks more effective than ever before.   We present Maybenot, a framework for traffic analysis defenses. Maybenot is designed to be easy to use and integrate into existing end-to-end encrypted protocols. It is implemented in the Rust programming language as a crate (library), together with a simulator to further the development of defenses. Defenses in Maybenot are expressed as probabilistic state machines that schedule actions to inject padding or block outgoing traffic. Maybenot is an evolution from the Tor Circuit Padding Framework by Perry and Kadianakis, designed to support a wide range of protocols and use cases.

摘要: 端到端加密是保护互联网用户隐私的有力工具。随着ToR、VPN和加密消息等技术的使用越来越多，网络对手监控和审查互联网流量变得越来越困难。攻击者剩下的另一个途径是流量分析：分析加密流量中的模式，以推断有关用户及其活动的信息。最近使用深度学习的改进使流量分析攻击比以往任何时候都更有效。我们提出了一个用于流量分析防御的框架--Maybenot。Maybenot被设计为易于使用并集成到现有的端到端加密协议中。它在Rust编程语言中被实现为一个板条箱(库)，以及一个模拟器来进一步开发防御。Maybenot中的防御被表示为概率状态机，该状态机调度注入填充或阻止传出流量的动作。Maybenot是由Perry和Kadianakis提出的Tor电路填充框架的演变，旨在支持广泛的协议和用例。



## **49. Wavelets Beat Monkeys at Adversarial Robustness**

小波在对抗健壮性上击败猴子 cs.LG

Machine Learning and the Physical Sciences Workshop, NeurIPS 2022

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09403v1) [paper-pdf](http://arxiv.org/pdf/2304.09403v1)

**Authors**: Jingtong Su, Julia Kempe

**Abstract**: Research on improving the robustness of neural networks to adversarial noise - imperceptible malicious perturbations of the data - has received significant attention. The currently uncontested state-of-the-art defense to obtain robust deep neural networks is Adversarial Training (AT), but it consumes significantly more resources compared to standard training and trades off accuracy for robustness. An inspiring recent work [Dapello et al.] aims to bring neurobiological tools to the question: How can we develop Neural Nets that robustly generalize like human vision? [Dapello et al.] design a network structure with a neural hidden first layer that mimics the primate primary visual cortex (V1), followed by a back-end structure adapted from current CNN vision models. It seems to achieve non-trivial adversarial robustness on standard vision benchmarks when tested on small perturbations. Here we revisit this biologically inspired work, and ask whether a principled parameter-free representation with inspiration from physics is able to achieve the same goal. We discover that the wavelet scattering transform can replace the complex V1-cortex and simple uniform Gaussian noise can take the role of neural stochasticity, to achieve adversarial robustness. In extensive experiments on the CIFAR-10 benchmark with adaptive adversarial attacks we show that: 1) Robustness of VOneBlock architectures is relatively weak (though non-zero) when the strength of the adversarial attack radius is set to commonly used benchmarks. 2) Replacing the front-end VOneBlock by an off-the-shelf parameter-free Scatternet followed by simple uniform Gaussian noise can achieve much more substantial adversarial robustness without adversarial training. Our work shows how physically inspired structures yield new insights into robustness that were previously only thought possible by meticulously mimicking the human cortex.

摘要: 提高神经网络对敌意噪声--数据的不可察觉的恶意扰动--的稳健性的研究受到了极大的关注。目前无可争议的获得稳健深度神经网络的最先进的防御是对抗性训练(AT)，但与标准训练相比，它消耗的资源明显更多，并以精度换取健壮性。最近一部鼓舞人心的作品[Dapello等人]旨在将神经生物学工具引入这个问题：我们如何才能开发出像人类视觉一样具有强大泛化能力的神经网络？[Dapello等人]设计一个网络结构，该网络结构具有一个模仿灵长类初级视觉皮质(V1)的神经隐藏第一层，然后是一个改编自当前CNN视觉模型的后端结构。当在小扰动上测试时，它似乎在标准视觉基准上实现了非平凡的对抗性健壮性。在这里，我们重新审视这项受生物启发的工作，并询问受物理学启发的原则性无参数表示法是否能够实现同样的目标。我们发现，小波散射变换可以代替复杂的V1皮层，而简单的均匀高斯噪声可以起到神经随机性的作用，达到对抗的稳健性。在CIFAR-10基准上进行的大量自适应攻击实验表明：1)当敌方攻击半径设置为常用基准时，VOneBlock架构的健壮性相对较弱(尽管非零值)。2)用无参数散射网和简单的均匀高斯噪声代替前端的VOneBlock，无需对抗性训练即可获得更强的对抗性健壮性。我们的工作展示了受物理启发的结构如何产生对健壮性的新见解，而这在以前只被认为是通过精心模仿人类皮质才可能实现的。



## **50. CodeAttack: Code-Based Adversarial Attacks for Pre-trained Programming Language Models**

CodeAttack：针对预先训练的编程语言模型的基于代码的对抗性攻击 cs.CL

AAAI Conference on Artificial Intelligence (AAAI) 2023

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2206.00052v3) [paper-pdf](http://arxiv.org/pdf/2206.00052v3)

**Authors**: Akshita Jha, Chandan K. Reddy

**Abstract**: Pre-trained programming language (PL) models (such as CodeT5, CodeBERT, GraphCodeBERT, etc.,) have the potential to automate software engineering tasks involving code understanding and code generation. However, these models operate in the natural channel of code, i.e., they are primarily concerned with the human understanding of the code. They are not robust to changes in the input and thus, are potentially susceptible to adversarial attacks in the natural channel. We propose, CodeAttack, a simple yet effective black-box attack model that uses code structure to generate effective, efficient, and imperceptible adversarial code samples and demonstrates the vulnerabilities of the state-of-the-art PL models to code-specific adversarial attacks. We evaluate the transferability of CodeAttack on several code-code (translation and repair) and code-NL (summarization) tasks across different programming languages. CodeAttack outperforms state-of-the-art adversarial NLP attack models to achieve the best overall drop in performance while being more efficient, imperceptible, consistent, and fluent. The code can be found at https://github.com/reddy-lab-code-research/CodeAttack.

摘要: 预先训练的编程语言(PL)模型(如CodeT5、CodeBERT、GraphCodeBERT等)有可能自动化涉及代码理解和代码生成的软件工程任务。然而，这些模型在代码的自然通道中运行，即它们主要关注人类对代码的理解。它们对输入的变化不是很健壮，因此，在自然通道中可能容易受到对抗性攻击。我们提出了一个简单而有效的黑盒攻击模型CodeAttack，它使用代码结构来生成有效、高效和不可察觉的对抗性代码样本，并展示了最新的PL模型对代码特定的对抗性攻击的脆弱性。我们评估了CodeAttack在几个代码-代码(翻译和修复)和代码-NL(摘要)任务上跨不同编程语言的可移植性。CodeAttack超越了最先进的对抗性NLP攻击模型，在更高效、更隐蔽、更一致和更流畅的同时，实现了最佳的整体性能下降。代码可在https://github.com/reddy-lab-code-research/CodeAttack.上找到



