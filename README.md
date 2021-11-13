# NewAdversarialAttackPaper
## **Poisoning Knowledge Graph Embeddings via Relation Inference Patterns**

通过关系推理模式毒化知识图嵌入 cs.LG

Joint Conference of the 59th Annual Meeting of the Association for  Computational Linguistics and the 11th International Joint Conference on  Natural Language Processing (ACL-IJCNLP 2021)

**SubmitDate**: 2021-11-11    [paper-pdf](http://arxiv.org/pdf/2111.06345v1)

**Authors**: Peru Bhardwaj, John Kelleher, Luca Costabello, Declan O'Sullivan

**Abstracts**: We study the problem of generating data poisoning attacks against Knowledge Graph Embedding (KGE) models for the task of link prediction in knowledge graphs. To poison KGE models, we propose to exploit their inductive abilities which are captured through the relationship patterns like symmetry, inversion and composition in the knowledge graph. Specifically, to degrade the model's prediction confidence on target facts, we propose to improve the model's prediction confidence on a set of decoy facts. Thus, we craft adversarial additions that can improve the model's prediction confidence on decoy facts through different inference patterns. Our experiments demonstrate that the proposed poisoning attacks outperform state-of-art baselines on four KGE models for two publicly available datasets. We also find that the symmetry pattern based attacks generalize across all model-dataset combinations which indicates the sensitivity of KGE models to this pattern.

摘要: 研究了针对知识图中链接预测任务的知识图嵌入(KGE)模型产生数据中毒攻击的问题。为了毒化KGE模型，我们提出开发KGE模型的归纳能力，这些能力是通过知识图中的对称性、反转和合成等关系模式捕捉到的。具体地说，为了降低模型对目标事实的预测置信度，我们提出了提高模型对一组诱饵事实的预测置信度的方法。因此，我们设计了对抗性的加法，通过不同的推理模式来提高模型对诱饵事实的预测置信度。我们的实验表明，提出的中毒攻击在两个公开可用的数据集的四个KGE模型上的性能优于最新的基线。我们还发现，基于对称模式的攻击在所有模型-数据集组合中都是通用的，这表明了KGE模型对该模式的敏感性。



## **Trustworthy Medical Segmentation with Uncertainty Estimation**

基于不确定性估计的可信医学分割 eess.IV

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05978v1)

**Authors**: Giuseppina Carannante, Dimah Dera, Nidhal C. Bouaynaya, Rasool Ghulam, Hassan M. Fathallah-Shaykh

**Abstracts**: Deep Learning (DL) holds great promise in reshaping the healthcare systems given its precision, efficiency, and objectivity. However, the brittleness of DL models to noisy and out-of-distribution inputs is ailing their deployment in the clinic. Most systems produce point estimates without further information about model uncertainty or confidence. This paper introduces a new Bayesian deep learning framework for uncertainty quantification in segmentation neural networks, specifically encoder-decoder architectures. The proposed framework uses the first-order Taylor series approximation to propagate and learn the first two moments (mean and covariance) of the distribution of the model parameters given the training data by maximizing the evidence lower bound. The output consists of two maps: the segmented image and the uncertainty map of the segmentation. The uncertainty in the segmentation decisions is captured by the covariance matrix of the predictive distribution. We evaluate the proposed framework on medical image segmentation data from Magnetic Resonances Imaging and Computed Tomography scans. Our experiments on multiple benchmark datasets demonstrate that the proposed framework is more robust to noise and adversarial attacks as compared to state-of-the-art segmentation models. Moreover, the uncertainty map of the proposed framework associates low confidence (or equivalently high uncertainty) to patches in the test input images that are corrupted with noise, artifacts or adversarial attacks. Thus, the model can self-assess its segmentation decisions when it makes an erroneous prediction or misses part of the segmentation structures, e.g., tumor, by presenting higher values in the uncertainty map.

摘要: 深度学习(DL)由于其精确性、效率和客观性，在重塑医疗系统方面有着巨大的希望。然而，DL模型对噪声和非分布输入的脆性阻碍了它们在临床上的部署。大多数系统在没有关于模型不确定性或置信度的进一步信息的情况下产生点估计。本文介绍了一种新的贝叶斯深度学习框架，用于分段神经网络中的不确定性量化，特别是编解码器的体系结构。该框架使用一阶泰勒级数近似，通过最大化证据下界来传播和学习给定训练数据的模型参数分布的前两个矩(均值和协方差)。输出由两幅图组成：分割后的图像和分割的不确定性图。通过预测分布的协方差矩阵来捕捉分割决策中的不确定性。我们在磁共振成像和计算机断层扫描的医学图像分割数据上对所提出的框架进行了评估。我们在多个基准数据集上的实验表明，与现有的分割模型相比，该框架对噪声和敌意攻击具有更强的鲁棒性。此外，该框架的不确定性图将低置信度(或相当于高不确定性)与测试输入图像中被噪声、伪影或敌意攻击破坏的补丁相关联。因此，当模型做出错误的预测或通过在不确定性图中呈现更高的值来错过部分分割结构(例如，肿瘤)时，该模型可以自我评估其分割决策。



## **Robust Learning via Ensemble Density Propagation in Deep Neural Networks**

基于集成密度传播的深度神经网络鲁棒学习 cs.LG

submitted to 2020 IEEE International Workshop on Machine Learning for  Signal Processing

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05953v1)

**Authors**: Giuseppina Carannante, Dimah Dera, Ghulam Rasool, Nidhal C. Bouaynaya, Lyudmila Mihaylova

**Abstracts**: Learning in uncertain, noisy, or adversarial environments is a challenging task for deep neural networks (DNNs). We propose a new theoretically grounded and efficient approach for robust learning that builds upon Bayesian estimation and Variational Inference. We formulate the problem of density propagation through layers of a DNN and solve it using an Ensemble Density Propagation (EnDP) scheme. The EnDP approach allows us to propagate moments of the variational probability distribution across the layers of a Bayesian DNN, enabling the estimation of the mean and covariance of the predictive distribution at the output of the model. Our experiments using MNIST and CIFAR-10 datasets show a significant improvement in the robustness of the trained models to random noise and adversarial attacks.

摘要: 对于深度神经网络(DNNs)来说，在不确定、噪声或敌对环境中学习是一项具有挑战性的任务。在贝叶斯估计和变分推理的基础上，提出了一种新的具有理论基础的、高效的鲁棒学习方法。我们用集合密度传播(ENDP)方案描述了DNN各层间的密度传播问题，并对其进行了求解。ENDP方法允许我们在贝叶斯DNN的各层之间传播变分概率分布的矩，从而能够在模型的输出处估计预测分布的均值和协方差。我们使用MNIST和CIFAR-10数据集进行的实验表明，训练后的模型对随机噪声和敌意攻击的鲁棒性有了显着的提高。



## **Sparse Adversarial Video Attacks with Spatial Transformations**

基于空间变换的稀疏对抗性视频攻击 cs.CV

The short version of this work will appear in the BMVC 2021  conference

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05468v1)

**Authors**: Ronghui Mu, Wenjie Ruan, Leandro Soriano Marcolino, Qiang Ni

**Abstracts**: In recent years, a significant amount of research efforts concentrated on adversarial attacks on images, while adversarial video attacks have seldom been explored. We propose an adversarial attack strategy on videos, called DeepSAVA. Our model includes both additive perturbation and spatial transformation by a unified optimisation framework, where the structural similarity index (SSIM) measure is adopted to measure the adversarial distance. We design an effective and novel optimisation scheme which alternatively utilizes Bayesian optimisation to identify the most influential frame in a video and Stochastic gradient descent (SGD) based optimisation to produce both additive and spatial-transformed perturbations. Doing so enables DeepSAVA to perform a very sparse attack on videos for maintaining human imperceptibility while still achieving state-of-the-art performance in terms of both attack success rate and adversarial transferability. Our intensive experiments on various types of deep neural networks and video datasets confirm the superiority of DeepSAVA.

摘要: 近年来，大量的研究工作集中在图像的对抗性攻击上，而对抗性视频攻击的研究很少。我们提出了一种针对视频的对抗性攻击策略，称为DeepSAVA。我们的模型通过一个统一的优化框架同时包括加性扰动和空间变换，其中采用结构相似指数(SSIM)度量对抗距离。我们设计了一种有效和新颖的优化方案，它交替使用贝叶斯优化来识别视频中最有影响力的帧，以及基于随机梯度下降(SGD)的优化来产生加性和空间变换的扰动。这样做使DeepSAVA能够对视频执行非常稀疏的攻击，以保持人的不可感知性，同时在攻击成功率和对手可转移性方面仍获得最先进的性能。我们在不同类型的深度神经网络和视频数据集上的密集实验证实了DeepSAVA的优越性。



## **Are Transformers More Robust Than CNNs?**

变形金刚比CNN更健壮吗？ cs.CV

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05464v1)

**Authors**: Yutong Bai, Jieru Mei, Alan Yuille, Cihang Xie

**Abstracts**: Transformer emerges as a powerful tool for visual recognition. In addition to demonstrating competitive performance on a broad range of visual benchmarks, recent works also argue that Transformers are much more robust than Convolutions Neural Networks (CNNs). Nonetheless, surprisingly, we find these conclusions are drawn from unfair experimental settings, where Transformers and CNNs are compared at different scales and are applied with distinct training frameworks. In this paper, we aim to provide the first fair & in-depth comparisons between Transformers and CNNs, focusing on robustness evaluations.   With our unified training setup, we first challenge the previous belief that Transformers outshine CNNs when measuring adversarial robustness. More surprisingly, we find CNNs can easily be as robust as Transformers on defending against adversarial attacks, if they properly adopt Transformers' training recipes. While regarding generalization on out-of-distribution samples, we show pre-training on (external) large-scale datasets is not a fundamental request for enabling Transformers to achieve better performance than CNNs. Moreover, our ablations suggest such stronger generalization is largely benefited by the Transformer's self-attention-like architectures per se, rather than by other training setups. We hope this work can help the community better understand and benchmark the robustness of Transformers and CNNs. The code and models are publicly available at https://github.com/ytongbai/ViTs-vs-CNNs.

摘要: 变压器作为一种强有力的视觉识别工具应运而生。除了展示好胜在广泛的可视基准上的性能外，最近的研究还认为，变形金刚比卷积神经网络(CNN)更健壮。然而，令人惊讶的是，我们发现这些结论是从不公平的实验环境中得出的，在这些实验环境中，变形金刚和CNN在不同的尺度上进行了比较，并应用了不同的训练框架。在本文中，我们的目标是提供变压器和CNN之间的第一次公平和深入的比较，重点是鲁棒性评估。有了我们的统一训练设置，我们首先挑战了以前的信念，即在衡量对手的健壮性时，变形金刚优于CNN。更令人惊讶的是，我们发现，如果CNN恰当地采用了变形金刚的训练食谱，它们在防御对手攻击方面可以很容易地像变形金刚一样健壮。虽然关于分布外样本的泛化，我们表明(外部)大规模数据集的预训练并不是使Transformers获得比CNN更好的性能的基本要求。此外，我们的消融表明，这种更强的概括性在很大程度上得益于变形金刚的自我关注式架构本身，而不是其他培训设置。我们希望这项工作可以帮助社区更好地理解Transformers和CNN的健壮性，并对其进行基准测试。代码和模型可在https://github.com/ytongbai/ViTs-vs-CNNs.上公开获得



## **Statistical Perspectives on Reliability of Artificial Intelligence Systems**

人工智能系统可靠性的统计透视 cs.SE

40 pages

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05391v1)

**Authors**: Yili Hong, Jiayi Lian, Li Xu, Jie Min, Yueyao Wang, Laura J. Freeman, Xinwei Deng

**Abstracts**: Artificial intelligence (AI) systems have become increasingly popular in many areas. Nevertheless, AI technologies are still in their developing stages, and many issues need to be addressed. Among those, the reliability of AI systems needs to be demonstrated so that the AI systems can be used with confidence by the general public. In this paper, we provide statistical perspectives on the reliability of AI systems. Different from other considerations, the reliability of AI systems focuses on the time dimension. That is, the system can perform its designed functionality for the intended period. We introduce a so-called SMART statistical framework for AI reliability research, which includes five components: Structure of the system, Metrics of reliability, Analysis of failure causes, Reliability assessment, and Test planning. We review traditional methods in reliability data analysis and software reliability, and discuss how those existing methods can be transformed for reliability modeling and assessment of AI systems. We also describe recent developments in modeling and analysis of AI reliability and outline statistical research challenges in this area, including out-of-distribution detection, the effect of the training set, adversarial attacks, model accuracy, and uncertainty quantification, and discuss how those topics can be related to AI reliability, with illustrative examples. Finally, we discuss data collection and test planning for AI reliability assessment and how to improve system designs for higher AI reliability. The paper closes with some concluding remarks.

摘要: 人工智能(AI)系统在许多领域变得越来越受欢迎。然而，人工智能技术仍处于发展阶段，许多问题需要解决。其中，需要证明人工智能系统的可靠性，以便普通公众可以放心地使用人工智能系统。在这篇文章中，我们提供了关于人工智能系统可靠性的统计观点。与其他考虑因素不同，人工智能系统的可靠性侧重于时间维度。也就是说，系统可以在预期的时间段内执行其设计的功能。介绍了一种用于人工智能可靠性研究的智能统计框架，该框架包括五个组成部分：系统结构、可靠性度量、故障原因分析、可靠性评估和测试规划。我们回顾了可靠性数据分析和软件可靠性的传统方法，并讨论了如何将这些现有方法转化为人工智能系统的可靠性建模和评估。我们还描述了人工智能可靠性建模和分析的最新进展，并概述了该领域的统计研究挑战，包括分布失调检测、训练集的影响、对抗性攻击、模型精度和不确定性量化，并用说明性例子讨论了这些主题如何与人工智能可靠性相关。最后，我们讨论了人工智能可靠性评估的数据收集和测试规划，以及如何改进系统设计以提高人工智能可靠性。论文最后以一些结束语结束。



## **Membership Inference Attacks Against Self-supervised Speech Models**

针对自监督语音模型的隶属度推理攻击 cs.CR

Submitted to ICASSP 2022. Source code available at  https://github.com/RayTzeng/s3m-membership-inference

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05113v1)

**Authors**: Wei-Cheng Tseng, Wei-Tsung Kao, Hung-yi Lee

**Abstracts**: Recently, adapting the idea of self-supervised learning (SSL) on continuous speech has started gaining attention. SSL models pre-trained on a huge amount of unlabeled audio can generate general-purpose representations that benefit a wide variety of speech processing tasks. Despite their ubiquitous deployment, however, the potential privacy risks of these models have not been well investigated. In this paper, we present the first privacy analysis on several SSL speech models using Membership Inference Attacks (MIA) under black-box access. The experiment results show that these pre-trained models are vulnerable to MIA and prone to membership information leakage with high adversarial advantage scores in both utterance-level and speaker-level. Furthermore, we also conduct several ablation studies to understand the factors that contribute to the success of MIA.

摘要: 最近，将自我监督学习(SSL)的思想应用于连续语音的研究开始受到关注。在大量未标记音频上预先训练的SSL模型可以生成有利于各种语音处理任务的通用表示。然而，尽管它们的部署无处不在，但这些模型的潜在隐私风险还没有得到很好的调查。本文首次对几种SSL语音模型在黑盒访问下使用成员推理攻击(MIA)进行了隐私分析。实验结果表明，这些预训练模型在话语级和说话人级都有较高的对手优势得分，容易受到MIA的影响，容易泄露成员信息。此外，我们还进行了几项消融研究，以了解导致MIA成功的因素。



## **A Statistical Difference Reduction Method for Escaping Backdoor Detection**

一种逃避后门检测的统计减差方法 cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05077v1)

**Authors**: Pengfei Xia, Hongjing Niu, Ziqiang Li, Bin Li

**Abstracts**: Recent studies show that Deep Neural Networks (DNNs) are vulnerable to backdoor attacks. An infected model behaves normally on benign inputs, whereas its prediction will be forced to an attack-specific target on adversarial data. Several detection methods have been developed to distinguish inputs to defend against such attacks. The common hypothesis that these defenses rely on is that there are large statistical differences between the latent representations of clean and adversarial inputs extracted by the infected model. However, although it is important, comprehensive research on whether the hypothesis must be true is lacking. In this paper, we focus on it and study the following relevant questions: 1) What are the properties of the statistical differences? 2) How to effectively reduce them without harming the attack intensity? 3) What impact does this reduction have on difference-based defenses? Our work is carried out on the three questions. First, by introducing the Maximum Mean Discrepancy (MMD) as the metric, we identify that the statistical differences of multi-level representations are all large, not just the highest level. Then, we propose a Statistical Difference Reduction Method (SDRM) by adding a multi-level MMD constraint to the loss function during training a backdoor model to effectively reduce the differences. Last, three typical difference-based detection methods are examined. The F1 scores of these defenses drop from 90%-100% on the regularly trained backdoor models to 60%-70% on the models trained with SDRM on all two datasets, four model architectures, and four attack methods. The results indicate that the proposed method can be used to enhance existing attacks to escape backdoor detection algorithms.

摘要: 最近的研究表明，深度神经网络(DNNs)很容易受到后门攻击。被感染的模型在良性输入上表现正常，而它的预测将被迫在对抗性数据上针对攻击特定的目标。已经开发了几种检测方法来区分输入以防御此类攻击。这些防御所依赖的共同假设是，由感染模型提取的干净和敌对输入的潜在表示之间存在很大的统计差异。然而，尽管这很重要，但关于这一假设是否一定是真的缺乏全面的研究。本文针对这一问题进行了研究：1)统计差异的性质是什么？2)如何在不影响攻击强度的情况下有效地降低统计差异？3)这种减少对基于差异的防御有什么影响？(2)如何在不影响攻击强度的情况下有效地减少统计差异？3)这种减少对基于差异的防御有什么影响？我们的工作就是围绕这三个问题展开的。首先，通过引入最大平均差异(MMD)作为度量，我们发现多级表示的统计差异都很大，而不仅仅是最高级别。然后，在后门模型训练过程中，通过在损失函数中加入多级MMD约束，提出了一种统计差值缩减方法(SDRM)，有效地减小了差值。最后，分析了三种典型的基于差分的检测方法。在所有两个数据集、四个模型体系结构和四种攻击方法上，这些防御的F1得分从定期训练的后门模型的90%-100%下降到使用SDRM训练的模型的60%-70%。实验结果表明，该方法可用于增强现有的逃避后门检测算法的攻击。



## **Tightening the Approximation Error of Adversarial Risk with Auto Loss Function Search**

用自动损失函数搜索法缩小对抗性风险的逼近误差 cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05063v1)

**Authors**: Pengfei Xia, Ziqiang Li, Bin Li

**Abstracts**: Numerous studies have demonstrated that deep neural networks are easily misled by adversarial examples. Effectively evaluating the adversarial robustness of a model is important for its deployment in practical applications. Currently, a common type of evaluation is to approximate the adversarial risk of a model as a robustness indicator by constructing malicious instances and executing attacks. Unfortunately, there is an error (gap) between the approximate value and the true value. Previous studies manually design attack methods to achieve a smaller error, which is inefficient and may miss a better solution. In this paper, we establish the tightening of the approximation error as an optimization problem and try to solve it with an algorithm. More specifically, we first analyze that replacing the non-convex and discontinuous 0-1 loss with a surrogate loss, a necessary compromise in calculating the approximation, is one of the main reasons for the error. Then we propose AutoLoss-AR, the first method for searching loss functions for tightening the approximation error of adversarial risk. Extensive experiments are conducted in multiple settings. The results demonstrate the effectiveness of the proposed method: the best-discovered loss functions outperform the handcrafted baseline by 0.9%-2.9% and 0.7%-2.0% on MNIST and CIFAR-10, respectively. Besides, we also verify that the searched losses can be transferred to other settings and explore why they are better than the baseline by visualizing the local loss landscape.

摘要: 大量研究表明，深度神经网络很容易被对抗性例子所误导。有效地评估模型的对抗健壮性对于其在实际应用中的部署具有重要意义。目前，一种常见的评估方法是通过构建恶意实例和执行攻击来近似模型的敌意风险作为健壮性指标。不幸的是，近似值和真实值之间存在误差(差距)。以往的研究都是通过手工设计攻击方法来实现较小的错误，效率较低，可能会错过更好的解决方案。本文将逼近误差的收紧问题建立为优化问题，并尝试用算法求解。更具体地说，我们首先分析了用替代损失代替非凸的、不连续的0-1损失是造成误差的主要原因之一，这是计算近似时的一种必要的折衷。在此基础上，提出了第一种搜索损失函数的方法AutoLoss-AR，以减小对手风险的逼近误差。在多个环境中进行了广泛的实验。结果证明了该方法的有效性：在MNIST和CIFAR-10上，最好发现的损失函数的性能分别比手工制作的基线高0.9%-2.9%和0.7%-2.0%。此外，我们还验证了搜索到的损失可以转移到其他设置，并通过可视化本地损失情况来探索为什么它们比基线更好。



## **Bayesian Framework for Gradient Leakage**

梯度泄漏的贝叶斯框架 cs.LG

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04706v1)

**Authors**: Mislav Balunović, Dimitar I. Dimitrov, Robin Staab, Martin Vechev

**Abstracts**: Federated learning is an established method for training machine learning models without sharing training data. However, recent work has shown that it cannot guarantee data privacy as shared gradients can still leak sensitive information. To formalize the problem of gradient leakage, we propose a theoretical framework that enables, for the first time, analysis of the Bayes optimal adversary phrased as an optimization problem. We demonstrate that existing leakage attacks can be seen as approximations of this optimal adversary with different assumptions on the probability distributions of the input data and gradients. Our experiments confirm the effectiveness of the Bayes optimal adversary when it has knowledge of the underlying distribution. Further, our experimental evaluation shows that several existing heuristic defenses are not effective against stronger attacks, especially early in the training process. Thus, our findings indicate that the construction of more effective defenses and their evaluation remains an open problem.

摘要: 联合学习是一种在不共享训练数据的情况下训练机器学习模型的既定方法。然而，最近的研究表明，它不能保证数据隐私，因为共享梯度仍然可能泄露敏感信息。为了形式化梯度泄漏问题，我们提出了一个理论框架，该框架首次能够将贝叶斯最优对手表述为优化问题进行分析。我们证明了现有的泄漏攻击可以看作是对输入数据的概率分布和梯度的不同假设的最优对手的近似。我们的实验证实了贝叶斯最优对手在知道潜在分布的情况下的有效性。此外，我们的实验评估表明，现有的几种启发式防御方法对更强的攻击并不有效，特别是在训练过程的早期。因此，我们的研究结果表明，构建更有效的防御体系及其评估仍然是一个悬而未决的问题。



## **HAPSSA: Holistic Approach to PDF Malware Detection Using Signal and Statistical Analysis**

HAPSSA：基于信号和统计分析的PDF恶意软件整体检测方法 cs.CR

Submitted version - MILCOM 2021 IEEE Military Communications  Conference

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04703v1)

**Authors**: Tajuddin Manhar Mohammed, Lakshmanan Nataraj, Satish Chikkagoudar, Shivkumar Chandrasekaran, B. S. Manjunath

**Abstracts**: Malicious PDF documents present a serious threat to various security organizations that require modern threat intelligence platforms to effectively analyze and characterize the identity and behavior of PDF malware. State-of-the-art approaches use machine learning (ML) to learn features that characterize PDF malware. However, ML models are often susceptible to evasion attacks, in which an adversary obfuscates the malware code to avoid being detected by an Antivirus. In this paper, we derive a simple yet effective holistic approach to PDF malware detection that leverages signal and statistical analysis of malware binaries. This includes combining orthogonal feature space models from various static and dynamic malware detection methods to enable generalized robustness when faced with code obfuscations. Using a dataset of nearly 30,000 PDF files containing both malware and benign samples, we show that our holistic approach maintains a high detection rate (99.92%) of PDF malware and even detects new malicious files created by simple methods that remove the obfuscation conducted by malware authors to hide their malware, which are undetected by most antiviruses.

摘要: 恶意PDF文档对各种安全组织构成严重威胁，这些组织需要现代威胁情报平台来有效地分析和表征PDF恶意软件的身份和行为。最先进的方法使用机器学习(ML)来学习PDF恶意软件的特征。然而，ML模型经常容易受到规避攻击，在这种攻击中，敌手混淆恶意软件代码以避免被防病毒程序检测到。在本文中，我们推导了一种简单而有效的整体PDF恶意软件检测方法，该方法利用恶意软件二进制文件的信号和统计分析。这包括组合来自各种静电的正交特征空间模型和动态恶意软件检测方法，以在面临代码混淆时实现普遍的鲁棒性。使用包含近30,000个包含恶意软件和良性样本的PDF文件的数据集，我们显示，我们的整体方法保持了较高的PDF恶意软件检测率(99.92%)，甚至可以检测到通过简单方法创建的新恶意文件，这些新的恶意文件消除了恶意软件作者为隐藏恶意软件而进行的混淆，而这些文件是大多数防病毒软件无法检测到的。



## **DeepSteal: Advanced Model Extractions Leveraging Efficient Weight Stealing in Memories**

DeepSteal：高级模型提取，利用记忆中有效的重量窃取 cs.CR

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04625v1)

**Authors**: Adnan Siraj Rakin, Md Hafizul Islam Chowdhuryy, Fan Yao, Deliang Fan

**Abstracts**: Recent advancements of Deep Neural Networks (DNNs) have seen widespread deployment in multiple security-sensitive domains. The need of resource-intensive training and use of valuable domain-specific training data have made these models a top intellectual property (IP) for model owners. One of the major threats to the DNN privacy is model extraction attacks where adversaries attempt to steal sensitive information in DNN models. Recent studies show hardware-based side channel attacks can reveal internal knowledge about DNN models (e.g., model architectures) However, to date, existing attacks cannot extract detailed model parameters (e.g., weights/biases). In this work, for the first time, we propose an advanced model extraction attack framework DeepSteal that effectively steals DNN weights with the aid of memory side-channel attack. Our proposed DeepSteal comprises two key stages. Firstly, we develop a new weight bit information extraction method, called HammerLeak, through adopting the rowhammer based hardware fault technique as the information leakage vector. HammerLeak leverages several novel system-level techniques tailed for DNN applications to enable fast and efficient weight stealing. Secondly, we propose a novel substitute model training algorithm with Mean Clustering weight penalty, which leverages the partial leaked bit information effectively and generates a substitute prototype of the target victim model. We evaluate this substitute model extraction method on three popular image datasets (e.g., CIFAR-10/100/GTSRB) and four DNN architectures (e.g., ResNet-18/34/Wide-ResNet/VGG-11). The extracted substitute model has successfully achieved more than 90 % test accuracy on deep residual networks for the CIFAR-10 dataset. Moreover, our extracted substitute model could also generate effective adversarial input samples to fool the victim model.

摘要: 近年来，深度神经网络(DNNs)在多个安全敏感领域得到了广泛的应用。对资源密集型培训的需求和对有价值的特定领域培训数据的使用已使这些模型成为模型所有者的最高知识产权(IP)。DNN隐私面临的主要威胁之一是模型提取攻击，即攻击者试图窃取DNN模型中的敏感信息。最近的研究表明，基于硬件的侧信道攻击可以揭示DNN模型(例如，模型体系结构)的内部知识，然而，到目前为止，现有的攻击不能提取详细的模型参数(例如，权重/偏差)。在这项工作中，我们首次提出了一个高级模型提取攻击框架DeepSteal，该框架可以借助记忆边信道攻击有效地窃取DNN权重。我们建议的DeepSteal包括两个关键阶段。首先，通过采用基于Rowhammer的硬件故障技术作为信息泄漏向量，提出了一种新的加权比特信息提取方法HammerLeak。HammerLeak利用针对DNN应用的几种新颖的系统级技术来实现快速高效的重量盗窃。其次，提出了一种基于均值聚类权重惩罚的替身模型训练算法，该算法有效地利用了部分泄露的比特信息，生成了目标受害者模型的替身原型。我们在三个流行的图像数据集(如CIFAR10/10 0/GTSRB)和四个数字近邻结构(如Resnet-18/34/Wide-Resnet/VGG-11)上对该替身模型提取方法进行了评估。所提取的替身模型在CIFAR-10数据集上的深层残差网络上的测试准确率已成功达到90%以上。此外，我们提取的替身模型还可以生成有效的敌意输入样本来愚弄受害者模型。



## **Robust and Information-theoretically Safe Bias Classifier against Adversarial Attacks**

抗敌意攻击的稳健且信息理论安全的偏向分类器 cs.LG

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04404v1)

**Authors**: Lijia Yu, Xiao-Shan Gao

**Abstracts**: In this paper, the bias classifier is introduced, that is, the bias part of a DNN with Relu as the activation function is used as a classifier. The work is motivated by the fact that the bias part is a piecewise constant function with zero gradient and hence cannot be directly attacked by gradient-based methods to generate adversaries such as FGSM. The existence of the bias classifier is proved an effective training method for the bias classifier is proposed. It is proved that by adding a proper random first-degree part to the bias classifier, an information-theoretically safe classifier against the original-model gradient-based attack is obtained in the sense that the attack generates a totally random direction for generating adversaries. This seems to be the first time that the concept of information-theoretically safe classifier is proposed. Several attack methods for the bias classifier are proposed and numerical experiments are used to show that the bias classifier is more robust than DNNs against these attacks in most cases.

摘要: 本文介绍了偏向分类器，即以RELU为激活函数的DNN的偏向部分作为分类器。该工作的动机在于偏差部分是零梯度的分段常数函数，因此不能被基于梯度的方法直接攻击来生成诸如FGSM之类的对手。证明了偏向分类器的存在性，提出了一种有效的偏向分类器训练方法。证明了通过在偏向分类器中增加适当的随机一阶部分，在攻击产生一个完全随机的攻击方向的意义下，得到了一个针对原始模型梯度攻击的信息论安全的分类器。这似乎是首次提出信息理论安全分类器的概念。提出了几种针对偏向分类器的攻击方法，并通过数值实验表明，在大多数情况下，偏向分类器比DNNs对这些攻击具有更强的鲁棒性。



## **Get a Model! Model Hijacking Attack Against Machine Learning Models**

找个模特来！针对机器学习模型的模型劫持攻击 cs.CR

To Appear in NDSS 2022

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04394v1)

**Authors**: Ahmed Salem, Michael Backes, Yang Zhang

**Abstracts**: Machine learning (ML) has established itself as a cornerstone for various critical applications ranging from autonomous driving to authentication systems. However, with this increasing adoption rate of machine learning models, multiple attacks have emerged. One class of such attacks is training time attack, whereby an adversary executes their attack before or during the machine learning model training. In this work, we propose a new training time attack against computer vision based machine learning models, namely model hijacking attack. The adversary aims to hijack a target model to execute a different task than its original one without the model owner noticing. Model hijacking can cause accountability and security risks since a hijacked model owner can be framed for having their model offering illegal or unethical services. Model hijacking attacks are launched in the same way as existing data poisoning attacks. However, one requirement of the model hijacking attack is to be stealthy, i.e., the data samples used to hijack the target model should look similar to the model's original training dataset. To this end, we propose two different model hijacking attacks, namely Chameleon and Adverse Chameleon, based on a novel encoder-decoder style ML model, namely the Camouflager. Our evaluation shows that both of our model hijacking attacks achieve a high attack success rate, with a negligible drop in model utility.

摘要: 机器学习(ML)已经成为从自动驾驶到身份验证系统等各种关键应用的基石。然而，随着机器学习模型采用率的不断提高，出现了多种攻击。这种攻击的一类是训练时间攻击，由此对手在机器学习模型训练之前或期间执行他们的攻击。在这项工作中，我们提出了一种新的针对基于计算机视觉的机器学习模型的训练时间攻击，即模型劫持攻击。敌手的目标是劫持目标模型，以便在模型所有者不察觉的情况下执行与其原始任务不同的任务。劫持模特可能会导致责任和安全风险，因为被劫持的模特所有者可能会因为让他们的模特提供非法或不道德的服务而被陷害。模型劫持攻击的发起方式与现有的数据中毒攻击方式相同。然而，模型劫持攻击的一个要求是隐蔽性，即用于劫持目标模型的数据样本应该与模型的原始训练数据集相似。为此，我们基于一种新的编解码器风格的ML模型，即伪装器，提出了两种不同模型的劫持攻击，即变色龙攻击和逆变色龙攻击。我们的评估表明，我们的两种模型劫持攻击都达到了很高的攻击成功率，而模型效用的下降可以忽略不计。



## **Geometrically Adaptive Dictionary Attack on Face Recognition**

人脸识别中的几何自适应字典攻击 cs.CV

Accepted at WACV 2022

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04371v1)

**Authors**: Junyoung Byun, Hyojun Go, Changick Kim

**Abstracts**: CNN-based face recognition models have brought remarkable performance improvement, but they are vulnerable to adversarial perturbations. Recent studies have shown that adversaries can fool the models even if they can only access the models' hard-label output. However, since many queries are needed to find imperceptible adversarial noise, reducing the number of queries is crucial for these attacks. In this paper, we point out two limitations of existing decision-based black-box attacks. We observe that they waste queries for background noise optimization, and they do not take advantage of adversarial perturbations generated for other images. We exploit 3D face alignment to overcome these limitations and propose a general strategy for query-efficient black-box attacks on face recognition named Geometrically Adaptive Dictionary Attack (GADA). Our core idea is to create an adversarial perturbation in the UV texture map and project it onto the face in the image. It greatly improves query efficiency by limiting the perturbation search space to the facial area and effectively recycling previous perturbations. We apply the GADA strategy to two existing attack methods and show overwhelming performance improvement in the experiments on the LFW and CPLFW datasets. Furthermore, we also present a novel attack strategy that can circumvent query similarity-based stateful detection that identifies the process of query-based black-box attacks.

摘要: 基于CNN的人脸识别模型带来了显著的性能提升，但它们容易受到对手的干扰。最近的研究表明，即使对手只能访问模型的硬标签输出，他们也可以愚弄模型。然而，由于需要大量的查询来发现不可察觉的对抗性噪声，因此减少查询的数量对这些攻击至关重要。在本文中，我们指出了现有基于决策的黑盒攻击的两个局限性。我们观察到，它们将查询浪费在背景噪声优化上，并且它们没有利用为其他图像生成的对抗性扰动。我们利用三维人脸对齐来克服这些限制，并提出了一种通用的人脸识别黑盒攻击策略，称为几何自适应字典攻击(GADA)。我们的核心想法是在UV纹理贴图中创建对抗性扰动，并将其投影到图像中的脸部。通过将扰动搜索空间限制在人脸区域，并有效地循环使用先前的扰动，极大地提高了查询效率。我们将GADA策略应用于现有的两种攻击方法，在LFW和CPLFW数据集上的实验表明，GADA策略的性能有了显著的提高。此外，我们还提出了一种新的攻击策略，可以规避基于查询相似度的状态检测，识别基于查询的黑盒攻击过程。



## **Characterizing the adversarial vulnerability of speech self-supervised learning**

语音自监督学习的对抗性脆弱性表征 cs.SD

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04330v1)

**Authors**: Haibin Wu, Bo Zheng, Xu Li, Xixin Wu, Hung-yi Lee, Helen Meng

**Abstracts**: A leaderboard named Speech processing Universal PERformance Benchmark (SUPERB), which aims at benchmarking the performance of a shared self-supervised learning (SSL) speech model across various downstream speech tasks with minimal modification of architectures and small amount of data, has fueled the research for speech representation learning. The SUPERB demonstrates speech SSL upstream models improve the performance of various downstream tasks through just minimal adaptation. As the paradigm of the self-supervised learning upstream model followed by downstream tasks arouses more attention in the speech community, characterizing the adversarial robustness of such paradigm is of high priority. In this paper, we make the first attempt to investigate the adversarial vulnerability of such paradigm under the attacks from both zero-knowledge adversaries and limited-knowledge adversaries. The experimental results illustrate that the paradigm proposed by SUPERB is seriously vulnerable to limited-knowledge adversaries, and the attacks generated by zero-knowledge adversaries are with transferability. The XAB test verifies the imperceptibility of crafted adversarial attacks.

摘要: 一个名为语音处理通用性能基准(SUBB)的排行榜推动了语音表示学习的研究，该基准测试旨在以最小的体系结构和少量的数据对共享的自监督学习(SSL)语音模型在各种下游语音任务中的性能进行基准测试。出色的演示了语音SSL上行模型通过最小程度的适配提高了各种下行任务的性能。随着上游自我监督学习模型和下游任务的范式越来越受到语言学界的关注，表征这种范式的对抗性鲁棒性是当务之急。在本文中，我们首次尝试研究了该范式在零知识和有限知识两种攻击下的攻击脆弱性。实验结果表明，Superb提出的范式对有限知识的攻击具有很强的脆弱性，零知识攻击产生的攻击具有可移植性。Xab测试验证精心设计的敌意攻击的隐蔽性。



## **Graph Robustness Benchmark: Benchmarking the Adversarial Robustness of Graph Machine Learning**

图健壮性基准：对图机器学习的对抗性健壮性进行基准测试 cs.LG

21 pages, 12 figures, NeurIPS 2021 Datasets and Benchmarks Track

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04314v1)

**Authors**: Qinkai Zheng, Xu Zou, Yuxiao Dong, Yukuo Cen, Da Yin, Jiarong Xu, Yang Yang, Jie Tang

**Abstracts**: Adversarial attacks on graphs have posed a major threat to the robustness of graph machine learning (GML) models. Naturally, there is an ever-escalating arms race between attackers and defenders. However, the strategies behind both sides are often not fairly compared under the same and realistic conditions. To bridge this gap, we present the Graph Robustness Benchmark (GRB) with the goal of providing a scalable, unified, modular, and reproducible evaluation for the adversarial robustness of GML models. GRB standardizes the process of attacks and defenses by 1) developing scalable and diverse datasets, 2) modularizing the attack and defense implementations, and 3) unifying the evaluation protocol in refined scenarios. By leveraging the GRB pipeline, the end-users can focus on the development of robust GML models with automated data processing and experimental evaluations. To support open and reproducible research on graph adversarial learning, GRB also hosts public leaderboards across different scenarios. As a starting point, we conduct extensive experiments to benchmark baseline techniques. GRB is open-source and welcomes contributions from the community. Datasets, codes, leaderboards are available at https://cogdl.ai/grb/home.

摘要: 图的敌意攻击已经成为图机器学习(GML)模型健壮性的主要威胁。当然，攻击者和防御者之间的军备竞赛不断升级。然而，在相同的现实条件下，双方背后的战略往往是不公平的比较。为了弥补这一差距，我们提出了图健壮性基准(GRB)，目的是为GML模型的对抗健壮性提供一个可扩展的、统一的、模块化的和可重现的评估。GRB通过1)开发可扩展和多样化的数据集，2)将攻击和防御实现模块化，3)在细化的场景中统一评估协议，从而标准化了攻击和防御的过程。通过利用GRB管道，最终用户可以专注于开发具有自动数据处理和实验评估功能的健壮GML模型。为了支持关于图形对抗性学习的开放和可重复的研究，GRB还在不同的场景中主持公共排行榜。作为起点，我们进行了广泛的实验来对基线技术进行基准测试。GRB是开源的，欢迎来自社区的贡献。有关数据集、代码和排行榜的信息，请访问https://cogdl.ai/grb/home.



## **Defense Against Explanation Manipulation**

对解释操纵的防御 cs.LG

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04303v1)

**Authors**: Ruixiang Tang, Ninghao Liu, Fan Yang, Na Zou, Xia Hu

**Abstracts**: Explainable machine learning attracts increasing attention as it improves transparency of models, which is helpful for machine learning to be trusted in real applications. However, explanation methods have recently been demonstrated to be vulnerable to manipulation, where we can easily change a model's explanation while keeping its prediction constant. To tackle this problem, some efforts have been paid to use more stable explanation methods or to change model configurations. In this work, we tackle the problem from the training perspective, and propose a new training scheme called Adversarial Training on EXplanations (ATEX) to improve the internal explanation stability of a model regardless of the specific explanation method being applied. Instead of directly specifying explanation values over data instances, ATEX only puts requirement on model predictions which avoids involving second-order derivatives in optimization. As a further discussion, we also find that explanation stability is closely related to another property of the model, i.e., the risk of being exposed to adversarial attack. Through experiments, besides showing that ATEX improves model robustness against manipulation targeting explanation, it also brings additional benefits including smoothing explanations and improving the efficacy of adversarial training if applied to the model.

摘要: 可解释机器学习由于提高了模型的透明性而受到越来越多的关注，这有助于机器学习在实际应用中得到信任。然而，最近已经证明解释方法容易受到操纵，在这些方法中，我们可以很容易地改变模型的解释，同时保持其预测不变。为了解决撞击的这一问题，已经做出了一些努力，使用更稳定的解释方法或改变模型配置。在这项工作中，我们从训练的角度对这一问题进行了撞击研究，并提出了一种新的训练方案，称为对抗性解释训练(ATEX)，以提高模型的内部解释稳定性，而不考虑具体的解释方法。ATEX没有直接指定数据实例上的解释值，而是只对模型预测提出了要求，避免了优化中涉及二阶导数的问题。作为进一步的讨论，我们还发现解释稳定性与模型的另一个性质，即暴露于敌意攻击的风险密切相关。通过实验表明，ATEX除了提高了模型对操作目标解释的鲁棒性外，如果将其应用到模型中，还可以带来平滑解释和提高对抗性训练效果等额外的好处。



## **Generative Dynamic Patch Attack**

生成式动态补丁攻击 cs.CV

Published as a conference paper at BMVC 2021

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04266v1)

**Authors**: Xiang Li, Shihao Ji

**Abstracts**: Adversarial patch attack is a family of attack algorithms that perturb a part of image to fool a deep neural network model. Existing patch attacks mostly consider injecting adversarial patches at input-agnostic locations: either a predefined location or a random location. This attack setup may be sufficient for attack but has considerable limitations when using it for adversarial training. Thus, robust models trained with existing patch attacks cannot effectively defend other adversarial attacks. In this paper, we first propose an end-to-end patch attack algorithm, Generative Dynamic Patch Attack (GDPA), which generates both patch pattern and patch location adversarially for each input image. We show that GDPA is a generic attack framework that can produce dynamic/static and visible/invisible patches with a few configuration changes. Secondly, GDPA can be readily integrated for adversarial training to improve model robustness to various adversarial attacks. Extensive experiments on VGGFace, Traffic Sign and ImageNet show that GDPA achieves higher attack success rates than state-of-the-art patch attacks, while adversarially trained model with GDPA demonstrates superior robustness to adversarial patch attacks than competing methods. Our source code can be found at https://github.com/lxuniverse/gdpa.

摘要: 对抗性补丁攻击是一系列攻击算法，通过扰动图像的一部分来欺骗深层神经网络模型。现有的补丁攻击大多考虑在与输入无关的位置(预定义位置或随机位置)注入敌意补丁。这种攻击设置对于攻击来说可能是足够的，但在用于对抗性训练时有相当大的限制。因此，用现有补丁攻击训练的鲁棒模型不能有效防御其他对抗性攻击。本文首先提出了一种端到端的补丁攻击算法--生成性动态补丁攻击(GDPA)，该算法对每幅输入图像分别生成补丁模式和补丁位置。我们证明了GDPA是一个通用的攻击框架，只需少量的配置更改，就可以生成动态/静电和可见/不可见的补丁。其次，GDPA可以很容易地集成到对抗性训练中，以提高模型对各种对抗性攻击的鲁棒性。在VGGFace、交通标志和ImageNet上的大量实验表明，GDPA比最新的补丁攻击具有更高的攻击成功率，而带有GDPA的对抗性训练模型对敌意补丁攻击表现出比竞争方法更好的鲁棒性。我们的源代码可以在https://github.com/lxuniverse/gdpa.上找到



## **Natural Adversarial Objects**

自然对抗性客体 cs.CV

**SubmitDate**: 2021-11-07    [paper-pdf](http://arxiv.org/pdf/2111.04204v1)

**Authors**: Felix Lau, Nishant Subramani, Sasha Harrison, Aerin Kim, Elliot Branson, Rosanne Liu

**Abstracts**: Although state-of-the-art object detection methods have shown compelling performance, models often are not robust to adversarial attacks and out-of-distribution data. We introduce a new dataset, Natural Adversarial Objects (NAO), to evaluate the robustness of object detection models. NAO contains 7,934 images and 9,943 objects that are unmodified and representative of real-world scenarios, but cause state-of-the-art detection models to misclassify with high confidence. The mean average precision (mAP) of EfficientDet-D7 drops 74.5% when evaluated on NAO compared to the standard MSCOCO validation set.   Moreover, by comparing a variety of object detection architectures, we find that better performance on MSCOCO validation set does not necessarily translate to better performance on NAO, suggesting that robustness cannot be simply achieved by training a more accurate model.   We further investigate why examples in NAO are difficult to detect and classify. Experiments of shuffling image patches reveal that models are overly sensitive to local texture. Additionally, using integrated gradients and background replacement, we find that the detection model is reliant on pixel information within the bounding box, and insensitive to the background context when predicting class labels. NAO can be downloaded at https://drive.google.com/drive/folders/15P8sOWoJku6SSEiHLEts86ORfytGezi8.

摘要: 虽然最先进的目标检测方法已经显示出令人信服的性能，但模型通常对敌意攻击和分布外的数据并不健壮。我们引入了一个新的数据集--自然对抗性对象(NAO)来评估目标检测模型的健壮性。NAO包含7934张图像和9943个对象，这些图像和对象未经修改，可以代表真实世界的场景，但会导致最先进的检测模型高度可信地错误分类。与标准MSCOCO验证集相比，在NAO上评估EfficientDet-D7的平均平均精度(MAP)下降了74.5%。此外，通过比较各种目标检测体系结构，我们发现在MSCOCO验证集上更好的性能并不一定转化为在NAO上更好的性能，这表明鲁棒性不能简单地通过训练更精确的模型来实现。我们进一步调查了为什么NAO中的例子很难检测和分类。混洗图像块的实验表明，模型对局部纹理过于敏感。此外，通过使用集成梯度和背景替换，我们发现该检测模型依赖于边界框内的像素信息，并且在预测类别标签时对背景上下文不敏感。NAO可从https://drive.google.com/drive/folders/15P8sOWoJku6SSEiHLEts86ORfytGezi8.下载



## **Reconstructing Training Data from Diverse ML Models by Ensemble Inversion**

基于集成反演的不同ML模型训练数据重构 cs.LG

9 pages, 8 figures, WACV 2022

**SubmitDate**: 2021-11-05    [paper-pdf](http://arxiv.org/pdf/2111.03702v1)

**Authors**: Qian Wang, Daniel Kurz

**Abstracts**: Model Inversion (MI), in which an adversary abuses access to a trained Machine Learning (ML) model attempting to infer sensitive information about its original training data, has attracted increasing research attention. During MI, the trained model under attack (MUA) is usually frozen and used to guide the training of a generator, such as a Generative Adversarial Network (GAN), to reconstruct the distribution of the original training data of that model. This might cause leakage of original training samples, and if successful, the privacy of dataset subjects will be at risk if the training data contains Personally Identifiable Information (PII). Therefore, an in-depth investigation of the potentials of MI techniques is crucial for the development of corresponding defense techniques. High-quality reconstruction of training data based on a single model is challenging. However, existing MI literature does not explore targeting multiple models jointly, which may provide additional information and diverse perspectives to the adversary.   We propose the ensemble inversion technique that estimates the distribution of original training data by training a generator constrained by an ensemble (or set) of trained models with shared subjects or entities. This technique leads to noticeable improvements of the quality of the generated samples with distinguishable features of the dataset entities compared to MI of a single ML model. We achieve high quality results without any dataset and show how utilizing an auxiliary dataset that's similar to the presumed training data improves the results. The impact of model diversity in the ensemble is thoroughly investigated and additional constraints are utilized to encourage sharp predictions and high activations for the reconstructed samples, leading to more accurate reconstruction of training images.

摘要: 模型反转(MI)是指敌手滥用对经过训练的机器学习(ML)模型的访问，试图推断关于其原始训练数据的敏感信息，已引起越来越多的研究关注。在MI过程中，训练的攻击下模型(MUA)通常被冻结，并用于指导生成器(如生成性对抗网络)的训练，以重构该模型的原始训练数据的分布。这可能会导致原始训练样本的泄漏，如果成功，如果训练数据包含个人身份信息(PII)，则数据集对象的隐私将面临风险。因此，深入研究MI技术的潜力对于发展相应的防御技术至关重要。基于单一模型的高质量训练数据重建具有挑战性。然而，现有的MI文献没有探索联合瞄准多个模型，这可能会为对手提供额外的信息和不同的视角。我们提出了集成反演技术，该技术通过训练一个生成器来估计原始训练数据的分布，该生成器受具有共享主题或实体的训练模型的集成(或集合)约束。与单个ML模型的MI相比，该技术导致具有数据集实体的可区分特征的生成样本的质量显著改善。我们在没有任何数据集的情况下实现了高质量的结果，并展示了如何利用与假定的训练数据相似的辅助数据集来改善结果。深入研究了集成中模型多样性的影响，并利用附加约束来鼓励对重建样本的精确预测和高激活，从而导致更准确的训练图像重建。



## **A Unified Game-Theoretic Interpretation of Adversarial Robustness**

对抗性稳健性的统一博弈论解释 cs.LG

the previous version is arXiv:2103.07364, but I mistakenly apply a  new ID for the paper

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.03536v2)

**Authors**: Jie Ren, Die Zhang, Yisen Wang, Lu Chen, Zhanpeng Zhou, Yiting Chen, Xu Cheng, Xin Wang, Meng Zhou, Jie Shi, Quanshi Zhang

**Abstracts**: This paper provides a unified view to explain different adversarial attacks and defense methods, \emph{i.e.} the view of multi-order interactions between input variables of DNNs. Based on the multi-order interaction, we discover that adversarial attacks mainly affect high-order interactions to fool the DNN. Furthermore, we find that the robustness of adversarially trained DNNs comes from category-specific low-order interactions. Our findings provide a potential method to unify adversarial perturbations and robustness, which can explain the existing defense methods in a principle way. Besides, our findings also make a revision of previous inaccurate understanding of the shape bias of adversarially learned features.

摘要: 本文提供了一个统一的视角来解释不同的对抗性攻击和防御方法，即DNNs的输入变量之间的多阶交互的观点。基于多阶交互，我们发现对抗性攻击主要影响高阶交互来欺骗DNN。此外，我们发现对抗性训练的DNN的鲁棒性来自于特定类别的低阶交互。我们的发现为统一对抗性扰动和鲁棒性提供了一种潜在的方法，可以对现有的防御方法进行原则性的解释。此外，我们的发现还修正了以往对对抗性习得特征的形状偏向的不准确理解。



## **Visualizing the Emergence of Intermediate Visual Patterns in DNNs**

在DNNs中可视化中间视觉模式的出现 cs.CV

**SubmitDate**: 2021-11-05    [paper-pdf](http://arxiv.org/pdf/2111.03505v1)

**Authors**: Mingjie Li, Shaobo Wang, Quanshi Zhang

**Abstracts**: This paper proposes a method to visualize the discrimination power of intermediate-layer visual patterns encoded by a DNN. Specifically, we visualize (1) how the DNN gradually learns regional visual patterns in each intermediate layer during the training process, and (2) the effects of the DNN using non-discriminative patterns in low layers to construct disciminative patterns in middle/high layers through the forward propagation. Based on our visualization method, we can quantify knowledge points (i.e., the number of discriminative visual patterns) learned by the DNN to evaluate the representation capacity of the DNN. Furthermore, this method also provides new insights into signal-processing behaviors of existing deep-learning techniques, such as adversarial attacks and knowledge distillation.

摘要: 提出了一种将DNN编码的中间层视觉模式的识别力可视化的方法。具体地说，我们可视化了(1)DNN如何在训练过程中逐渐学习各中间层的区域视觉模式，以及(2)DNN在低层使用非区分模式通过前向传播构建中高层区分模式的效果。基于我们的可视化方法，我们可以量化DNN学习的知识点(即区分视觉模式的数量)来评估DNN的表示能力。此外，该方法还为现有深度学习技术(如对抗性攻击和知识提取)的信号处理行为提供了新的见解。



## **Adversarial Attacks on Knowledge Graph Embeddings via Instance Attribution Methods**

基于实例属性方法的知识图嵌入对抗性攻击 cs.LG

2021 Conference on Empirical Methods in Natural Language Processing  (EMNLP 2021)

**SubmitDate**: 2021-11-04    [paper-pdf](http://arxiv.org/pdf/2111.03120v1)

**Authors**: Peru Bhardwaj, John Kelleher, Luca Costabello, Declan O'Sullivan

**Abstracts**: Despite the widespread use of Knowledge Graph Embeddings (KGE), little is known about the security vulnerabilities that might disrupt their intended behaviour. We study data poisoning attacks against KGE models for link prediction. These attacks craft adversarial additions or deletions at training time to cause model failure at test time. To select adversarial deletions, we propose to use the model-agnostic instance attribution methods from Interpretable Machine Learning, which identify the training instances that are most influential to a neural model's predictions on test instances. We use these influential triples as adversarial deletions. We further propose a heuristic method to replace one of the two entities in each influential triple to generate adversarial additions. Our experiments show that the proposed strategies outperform the state-of-art data poisoning attacks on KGE models and improve the MRR degradation due to the attacks by up to 62% over the baselines.

摘要: 尽管KGE(Knowledge Graph Embedding，知识图嵌入)被广泛使用，但人们对可能破坏其预期行为的安全漏洞知之甚少。我们研究了针对链接预测的KGE模型的数据中毒攻击。这些攻击在训练时精心设计敌意的添加或删除，从而在测试时导致模型失败。为了选择对抗性删除，我们建议使用可解释机器学习中的与模型无关的实例属性方法，该方法识别对神经模型对测试实例的预测影响最大的训练实例。我们使用这些有影响力的三元组作为对抗性删除。我们进一步提出了一种启发式方法来替换每个有影响力的三元组中的两个实体中的一个，以生成对抗性加法。我们的实验表明，所提出的策略比现有的针对KGE模型的数据中毒攻击具有更好的性能，并且使由于攻击而导致的MRR降级在基线上提高了高达62%。



## **Scanflow: A multi-graph framework for Machine Learning workflow management, supervision, and debugging**

Scanflow：一个用于机器学习工作流管理、监督和调试的多图框架 cs.LG

**SubmitDate**: 2021-11-04    [paper-pdf](http://arxiv.org/pdf/2111.03003v1)

**Authors**: Gusseppe Bravo-Rocca, Peini Liu, Jordi Guitart, Ajay Dholakia, David Ellison, Jeffrey Falkanger, Miroslav Hodak

**Abstracts**: Machine Learning (ML) is more than just training models, the whole workflow must be considered. Once deployed, a ML model needs to be watched and constantly supervised and debugged to guarantee its validity and robustness in unexpected situations. Debugging in ML aims to identify (and address) the model weaknesses in not trivial contexts. Several techniques have been proposed to identify different types of model weaknesses, such as bias in classification, model decay, adversarial attacks, etc., yet there is not a generic framework that allows them to work in a collaborative, modular, portable, iterative way and, more importantly, flexible enough to allow both human- and machine-driven techniques. In this paper, we propose a novel containerized directed graph framework to support and accelerate end-to-end ML workflow management, supervision, and debugging. The framework allows defining and deploying ML workflows in containers, tracking their metadata, checking their behavior in production, and improving the models by using both learned and human-provided knowledge. We demonstrate these capabilities by integrating in the framework two hybrid systems to detect data drift distribution which identify the samples that are far from the latent space of the original distribution, ask for human intervention, and whether retrain the model or wrap it with a filter to remove the noise of corrupted data at inference time. We test these systems on MNIST-C, CIFAR-10-C, and FashionMNIST-C datasets, obtaining promising accuracy results with the help of human involvement.

摘要: 机器学习(ML)不仅仅是训练模型，还必须考虑整个工作流程。一旦部署，就需要监视ML模型，并不断地对其进行监督和调试，以确保其在意外情况下的有效性和健壮性。ML中的调试旨在识别(并解决)在不平凡的上下文中的模型弱点。已经提出了几种技术来识别不同类型的模型弱点，例如分类偏差、模型衰减、对抗性攻击等，但是还没有一个通用的框架允许它们以协作、模块化、可移植、迭代的方式工作，更重要的是，足够灵活地允许人和机器驱动的技术。本文提出了一种新的容器化有向图框架来支持和加速端到端ML工作流管理、监督和调试。该框架允许在容器中定义和部署ML工作流，跟踪它们的元数据，检查它们在生产中的行为，并通过使用学习到的知识和人工提供的知识来改进模型。我们通过在框架中集成两个混合系统来检测数据漂移分布来展示这些能力，这两个系统识别远离原始分布潜在空间的样本，要求人工干预，以及是重新训练模型还是用过滤包裹模型，以在推理时消除损坏数据的噪声。我们在MNIST-C、CIFAR-10-C和FashionMNIST-C数据集上测试了这些系统，在人工参与的帮助下获得了令人满意的准确性结果。



## **Attacking Deep Reinforcement Learning-Based Traffic Signal Control Systems with Colluding Vehicles**

用合谋车辆攻击基于深度强化学习的交通信号控制系统 cs.LG

**SubmitDate**: 2021-11-04    [paper-pdf](http://arxiv.org/pdf/2111.02845v1)

**Authors**: Ao Qu, Yihong Tang, Wei Ma

**Abstracts**: The rapid advancements of Internet of Things (IoT) and artificial intelligence (AI) have catalyzed the development of adaptive traffic signal control systems (ATCS) for smart cities. In particular, deep reinforcement learning (DRL) methods produce the state-of-the-art performance and have great potentials for practical applications. In the existing DRL-based ATCS, the controlled signals collect traffic state information from nearby vehicles, and then optimal actions (e.g., switching phases) can be determined based on the collected information. The DRL models fully "trust" that vehicles are sending the true information to the signals, making the ATCS vulnerable to adversarial attacks with falsified information. In view of this, this paper first time formulates a novel task in which a group of vehicles can cooperatively send falsified information to "cheat" DRL-based ATCS in order to save their total travel time. To solve the proposed task, we develop CollusionVeh, a generic and effective vehicle-colluding framework composed of a road situation encoder, a vehicle interpreter, and a communication mechanism. We employ our method to attack established DRL-based ATCS and demonstrate that the total travel time for the colluding vehicles can be significantly reduced with a reasonable number of learning episodes, and the colluding effect will decrease if the number of colluding vehicles increases. Additionally, insights and suggestions for the real-world deployment of DRL-based ATCS are provided. The research outcomes could help improve the reliability and robustness of the ATCS and better protect the smart mobility systems.

摘要: 物联网(IoT)和人工智能(AI)的快速发展促进了智能城市自适应交通信号控制系统(ATCS)的发展。尤其是深度强化学习(DRL)方法具有最先进的性能和巨大的实际应用潜力。在现有的基于DRL的ATCS中，受控信号收集附近车辆的交通状态信息，然后可以基于收集的信息来确定最优动作(例如，切换相位)。DRL模型完全“信任”车辆正在向信号发送真实的信息，使得ATCS容易受到带有伪造信息的敌意攻击。有鉴于此，本文首次提出了一种新颖的任务，即一组车辆可以协同发送伪造信息来“欺骗”基于DRL的ATC，以节省它们的总行程时间。为了解决这一问题，我们开发了CollusionVeh，这是一个通用的、有效的车辆共谋框架，由路况编码器、车辆解释器和通信机制组成。我们利用我们的方法对已建立的基于DRL的ATCS进行攻击，并证明了在合理的学习场景数下，合谋车辆的总行驶时间可以显著减少，并且合谋效应随着合谋车辆数量的增加而降低。此外，还为基于DRL的ATCS的实际部署提供了见解和建议。研究成果有助于提高ATCS的可靠性和鲁棒性，更好地保护智能移动系统。



## **Adversarial Attacks on Graph Classification via Bayesian Optimisation**

基于贝叶斯优化的图分类对抗性攻击 stat.ML

NeurIPS 2021. 11 pages, 8 figures, 2 tables (24 pages, 17 figures, 8  tables including references and appendices)

**SubmitDate**: 2021-11-04    [paper-pdf](http://arxiv.org/pdf/2111.02842v1)

**Authors**: Xingchen Wan, Henry Kenlay, Binxin Ru, Arno Blaas, Michael A. Osborne, Xiaowen Dong

**Abstracts**: Graph neural networks, a popular class of models effective in a wide range of graph-based learning tasks, have been shown to be vulnerable to adversarial attacks. While the majority of the literature focuses on such vulnerability in node-level classification tasks, little effort has been dedicated to analysing adversarial attacks on graph-level classification, an important problem with numerous real-life applications such as biochemistry and social network analysis. The few existing methods often require unrealistic setups, such as access to internal information of the victim models, or an impractically-large number of queries. We present a novel Bayesian optimisation-based attack method for graph classification models. Our method is black-box, query-efficient and parsimonious with respect to the perturbation applied. We empirically validate the effectiveness and flexibility of the proposed method on a wide range of graph classification tasks involving varying graph properties, constraints and modes of attack. Finally, we analyse common interpretable patterns behind the adversarial samples produced, which may shed further light on the adversarial robustness of graph classification models.

摘要: 图神经网络是一类在广泛的基于图的学习任务中有效的流行模型，已被证明容易受到敌意攻击。虽然大多数文献集中在节点级分类任务中的此类漏洞，但很少有人致力于分析对图级分类的敌意攻击，这是许多现实应用(如生物化学和社会网络分析)中的一个重要问题。现有的少数方法通常需要不切实际的设置，例如访问受害者模型的内部信息，或者不切实际地进行大量查询。提出了一种新的基于贝叶斯优化的图分类模型攻击方法。我们的方法是黑箱的，查询效率高，并且相对于所应用的扰动是简约的。我们通过实验验证了该方法在涉及不同的图属性、约束和攻击模式的广泛的图分类任务上的有效性和灵活性。最后，我们分析了产生的对抗性样本背后常见的可解释模式，这可能进一步揭示图分类模型的对抗性鲁棒性。



## **Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models**

对抗性胶水：语言模型健壮性评估的多任务基准 cs.CL

Oral Presentation in NeurIPS 2021 (Datasets and Benchmarks Track). 24  pages, 4 figures, 12 tables

**SubmitDate**: 2021-11-04    [paper-pdf](http://arxiv.org/pdf/2111.02840v1)

**Authors**: Boxin Wang, Chejian Xu, Shuohang Wang, Zhe Gan, Yu Cheng, Jianfeng Gao, Ahmed Hassan Awadallah, Bo Li

**Abstracts**: Large-scale pre-trained language models have achieved tremendous success across a wide range of natural language understanding (NLU) tasks, even surpassing human performance. However, recent studies reveal that the robustness of these models can be challenged by carefully crafted textual adversarial examples. While several individual datasets have been proposed to evaluate model robustness, a principled and comprehensive benchmark is still missing. In this paper, we present Adversarial GLUE (AdvGLUE), a new multi-task benchmark to quantitatively and thoroughly explore and evaluate the vulnerabilities of modern large-scale language models under various types of adversarial attacks. In particular, we systematically apply 14 textual adversarial attack methods to GLUE tasks to construct AdvGLUE, which is further validated by humans for reliable annotations. Our findings are summarized as follows. (i) Most existing adversarial attack algorithms are prone to generating invalid or ambiguous adversarial examples, with around 90% of them either changing the original semantic meanings or misleading human annotators as well. Therefore, we perform a careful filtering process to curate a high-quality benchmark. (ii) All the language models and robust training methods we tested perform poorly on AdvGLUE, with scores lagging far behind the benign accuracy. We hope our work will motivate the development of new adversarial attacks that are more stealthy and semantic-preserving, as well as new robust language models against sophisticated adversarial attacks. AdvGLUE is available at https://adversarialglue.github.io.

摘要: 大规模的预训练语言模型在广泛的自然语言理解(NLU)任务中取得了巨大的成功，甚至超过了人类的表现。然而，最近的研究表明，这些模型的稳健性可能会受到精心设计的文本对抗性例子的挑战。虽然已经提出了几个单独的数据集来评估模型的稳健性，但仍然缺乏一个原则性和综合性的基准。本文提出了一种新的多任务基准--对抗性粘合剂(AdvGLUE)，用以定量、深入地研究和评估现代大规模语言模型在各种类型的对抗性攻击下的脆弱性。特别是，我们系统地应用了14种文本对抗性攻击方法来粘合任务来构建AdvGLUE，并进一步验证了该方法的可靠性。我们的发现总结如下。(I)现有的对抗性攻击算法大多容易产生无效或歧义的对抗性示例，其中90%左右的算法要么改变了原有的语义，要么误导了人类的注释者。因此，我们执行仔细的筛选过程来策划一个高质量的基准。(Ii)我们测试的所有语言模型和稳健训练方法在AdvGLUE上的表现都很差，分数远远落后于良性准确率。我们希望我们的工作将促进更隐蔽性和语义保持的新的对抗性攻击的发展，以及针对复杂的对抗性攻击的新的健壮语言模型的开发。有关AdvGLUE的信息，请访问https://adversarialglue.github.io.。



## **HoneyCar: A Framework to Configure Honeypot Vulnerabilities on the Internet of Vehicles**

HoneyCar：车联网蜜罐漏洞配置框架 cs.CR

**SubmitDate**: 2021-11-03    [paper-pdf](http://arxiv.org/pdf/2111.02364v1)

**Authors**: Sakshyam Panda, Stefan Rass, Sotiris Moschoyiannis, Kaitai Liang, George Loukas, Emmanouil Panaousis

**Abstracts**: The Internet of Vehicles (IoV), whereby interconnected vehicles communicate with each other and with road infrastructure on a common network, has promising socio-economic benefits but also poses new cyber-physical threats. Data on vehicular attackers can be realistically gathered through cyber threat intelligence using systems like honeypots. Admittedly, configuring honeypots introduces a trade-off between the level of honeypot-attacker interactions and any incurred overheads and costs for implementing and monitoring these honeypots. We argue that effective deception can be achieved through strategically configuring the honeypots to represent components of the IoV and engage attackers to collect cyber threat intelligence. In this paper, we present HoneyCar, a novel decision support framework for honeypot deception in IoV. HoneyCar builds upon a repository of known vulnerabilities of the autonomous and connected vehicles found in the Common Vulnerabilities and Exposure (CVE) data within the National Vulnerability Database (NVD) to compute optimal honeypot configuration strategies. By taking a game-theoretic approach, we model the adversarial interaction as a repeated imperfect-information zero-sum game in which the IoV network administrator chooses a set of vulnerabilities to offer in a honeypot and a strategic attacker chooses a vulnerability of the IoV to exploit under uncertainty. Our investigation is substantiated by examining two different versions of the game, with and without the re-configuration cost to empower the network administrator to determine optimal honeypot configurations. We evaluate HoneyCar in a realistic use case to support decision makers with determining optimal honeypot configuration strategies for strategic deployment in IoV.

摘要: 车联网(IoV)使互联的车辆相互通信，并在共同的网络上与道路基础设施进行通信，具有良好的社会经济效益，但也构成了新的网络-物理威胁。车辆攻击者的数据可以通过使用蜜罐等系统的网络威胁情报现实地收集到。诚然，配置蜜罐会在蜜罐-攻击者交互的级别与实现和监控这些蜜罐所产生的任何管理费用和成本之间进行权衡。我们认为，通过战略性地配置蜜罐来表示IoV的组件，并让攻击者参与收集网络威胁情报，可以实现有效的欺骗。本文提出了一种新的物联网蜜罐欺骗决策支持框架--HoneyCar。HoneyCar建立在国家漏洞数据库(NVD)的通用漏洞和暴露(CVE)数据中找到的自主和连接车辆的已知漏洞存储库，以计算最佳蜜罐配置策略。通过采用博弈论的方法，我们将敌方交互建模为一个重复的不完全信息零和博弈，其中IoV网络管理员选择一组漏洞提供给蜜罐，而策略性攻击者选择IoV的一个漏洞在不确定的情况下进行攻击。我们的调查通过检查游戏的两个不同版本来证实，在有和没有重新配置成本的情况下，使网络管理员能够确定最优的蜜罐配置。我们在实际使用案例中对HoneyCar进行评估，以帮助决策者确定IoV中战略部署的最佳蜜罐配置策略。



## **LTD: Low Temperature Distillation for Robust Adversarial Training**

LTD：低温蒸馏进行强有力的对抗性训练 cs.CV

**SubmitDate**: 2021-11-03    [paper-pdf](http://arxiv.org/pdf/2111.02331v1)

**Authors**: Erh-Chung Chen, Che-Rung Lee

**Abstracts**: Adversarial training has been widely used to enhance the robustness of the neural network models against adversarial attacks. However, there still a notable gap between the nature accuracy and the robust accuracy. We found one of the reasons is the commonly used labels, one-hot vectors, hinder the learning process for image recognition. In this paper, we proposed a method, called Low Temperature Distillation (LTD), which is based on the knowledge distillation framework to generate the desired soft labels. Unlike the previous work, LTD uses relatively low temperature in the teacher model, and employs different, but fixed, temperatures for the teacher model and the student model. Moreover, we have investigated the methods to synergize the use of nature data and adversarial ones in LTD. Experimental results show that without extra unlabeled data, the proposed method combined with the previous work can achieve 57.72\% and 30.36\% robust accuracy on CIFAR-10 and CIFAR-100 dataset respectively, which is about 1.21\% improvement of the state-of-the-art methods in average.

摘要: 对抗性训练已被广泛应用于增强神经网络模型抵抗对抗性攻击的鲁棒性。然而，自然精度与稳健精度之间仍有较大差距。我们发现其中一个原因是常用的标签，一个热点向量，阻碍了图像识别的学习过程。本文提出了一种基于知识蒸馏框架生成所需软标签的方法，称为低温蒸馏(LTD)。与以前的工作不同，LTD在教师模型中使用相对较低的温度，并为教师模型和学生模型使用不同但固定的温度。此外，我们还研究了在LTD协同使用自然数据和对抗性数据的方法。实验结果表明，该方法在不需要额外未标注数据的情况下，在CIFAR-10和CIFAR-100数据集上分别达到了57.72和30.36的鲁棒准确率，比现有方法平均提高了约1.21%。



## **Multi-Glimpse Network: A Robust and Efficient Classification Architecture based on Recurrent Downsampled Attention**

多瞥网络：一种基于循环下采样注意力的鲁棒高效分类体系结构 cs.CV

**SubmitDate**: 2021-11-03    [paper-pdf](http://arxiv.org/pdf/2111.02018v1)

**Authors**: Sia Huat Tan, Runpei Dong, Kaisheng Ma

**Abstracts**: Most feedforward convolutional neural networks spend roughly the same efforts for each pixel. Yet human visual recognition is an interaction between eye movements and spatial attention, which we will have several glimpses of an object in different regions. Inspired by this observation, we propose an end-to-end trainable Multi-Glimpse Network (MGNet) which aims to tackle the challenges of high computation and the lack of robustness based on recurrent downsampled attention mechanism. Specifically, MGNet sequentially selects task-relevant regions of an image to focus on and then adaptively combines all collected information for the final prediction. MGNet expresses strong resistance against adversarial attacks and common corruptions with less computation. Also, MGNet is inherently more interpretable as it explicitly informs us where it focuses during each iteration. Our experiments on ImageNet100 demonstrate the potential of recurrent downsampled attention mechanisms to improve a single feedforward manner. For example, MGNet improves 4.76% accuracy on average in common corruptions with only 36.9% computational cost. Moreover, while the baseline incurs an accuracy drop to 7.6%, MGNet manages to maintain 44.2% accuracy in the same PGD attack strength with ResNet-50 backbone. Our code is available at https://github.com/siahuat0727/MGNet.

摘要: 大多数前馈卷积神经网络对于每个像素花费大致相同的努力。然而，人类的视觉识别是眼球运动和空间注意力之间的相互作用，我们会对不同地区的一个物体有几次瞥见。受此启发，我们提出了一种端到端可训练的多瞥网络，旨在解决撞击中基于递归下采样注意机制的高计算量和健壮性不足的挑战。具体地说，MGNet按顺序选择图像中与任务相关的区域进行聚焦，然后自适应地组合所有收集的信息以进行最终预测。MGNet以较少的计算量对敌意攻击和常见的腐败表现出很强的抵抗力。此外，MGNet本质上更易于解释，因为它在每次迭代中都会明确地告诉我们它关注的位置。我们在ImageNet100上的实验证明了循环下采样注意机制改善单一前馈方式的潜力。例如，MGNet在普通腐败中平均提高4.76%的准确率，而计算代价仅为36.9%。此外，当基线导致准确率下降到7.6%时，MGNet设法在与ResNet-50主干相同的PGD攻击强度下保持44.2%的准确率。我们的代码可在https://github.com/siahuat0727/MGNet.获得



## **Pareto Adversarial Robustness: Balancing Spatial Robustness and Sensitivity-based Robustness**

帕累托对抗稳健性：平衡空间稳健性和基于敏感度的稳健性 cs.LG

**SubmitDate**: 2021-11-03    [paper-pdf](http://arxiv.org/pdf/2111.01996v1)

**Authors**: Ke Sun, Mingjie Li, Zhouchen Lin

**Abstracts**: Adversarial robustness, which mainly contains sensitivity-based robustness and spatial robustness, plays an integral part in the robust generalization. In this paper, we endeavor to design strategies to achieve universal adversarial robustness. To hit this target, we firstly investigate the less-studied spatial robustness and then integrate existing spatial robustness methods by incorporating both local and global spatial vulnerability into one spatial attack and adversarial training. Based on this exploration, we further present a comprehensive relationship between natural accuracy, sensitivity-based and different spatial robustness, supported by the strong evidence from the perspective of robust representation. More importantly, in order to balance these mutual impacts of different robustness into one unified framework, we incorporate \textit{Pareto criterion} into the adversarial robustness analysis, yielding a novel strategy called \textit{Pareto Adversarial Training} towards universal robustness. The resulting Pareto front, the set of optimal solutions, provides the set of optimal balance among natural accuracy and different adversarial robustness, shedding light on solutions towards universal robustness in the future. To the best of our knowledge, we are the first to consider the universal adversarial robustness via multi-objective optimization.

摘要: 对抗鲁棒性是鲁棒泛化的重要组成部分，主要包括基于敏感度的鲁棒性和空间鲁棒性。在本文中，我们努力设计策略来实现普遍的对抗健壮性。为了达到这一目标，我们首先研究了较少研究的空间鲁棒性，然后通过将局部和全局空间脆弱性合并到一个空间攻击和对抗性训练中来整合现有的空间鲁棒性方法。在此基础上，进一步提出了自然精度、基于敏感度和不同空间稳健性之间的综合关系，并从稳健表示的角度提供了有力的证据支持。更重要的是，为了将不同健壮性的相互影响平衡到一个统一的框架中，我们将Texttit{Pareto准则}引入到对抗健壮性分析中，提出了一种新的通用性健壮性策略--Texttit(Pareto对抗性训练)。由此得到的帕累托前沿，即最优解的集合，提供了自然精确度和不同对手鲁棒性之间的最佳平衡，从而揭示了未来通用性健壮性的解决方案。据我们所知，我们是第一个通过多目标优化来考虑普遍的对抗健壮性的。



## **Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks**

基于黑盒随机搜索的对抗性攻击搜索分布元学习 cs.LG

accepted at NeurIPS 2021

**SubmitDate**: 2021-11-02    [paper-pdf](http://arxiv.org/pdf/2111.01714v1)

**Authors**: Maksym Yatsura, Jan Hendrik Metzen, Matthias Hein

**Abstracts**: Adversarial attacks based on randomized search schemes have obtained state-of-the-art results in black-box robustness evaluation recently. However, as we demonstrate in this work, their efficiency in different query budget regimes depends on manual design and heuristic tuning of the underlying proposal distributions. We study how this issue can be addressed by adapting the proposal distribution online based on the information obtained during the attack. We consider Square Attack, which is a state-of-the-art score-based black-box attack, and demonstrate how its performance can be improved by a learned controller that adjusts the parameters of the proposal distribution online during the attack. We train the controller using gradient-based end-to-end training on a CIFAR10 model with white box access. We demonstrate that plugging the learned controller into the attack consistently improves its black-box robustness estimate in different query regimes by up to 20% for a wide range of different models with black-box access. We further show that the learned adaptation principle transfers well to the other data distributions such as CIFAR100 or ImageNet and to the targeted attack setting.

摘要: 近年来，基于随机搜索方案的对抗性攻击在黑盒健壮性评估方面取得了最新的研究成果。然而，正如我们在这项工作中演示的那样，它们在不同查询预算机制中的效率取决于对底层提案分布的手动设计和启发式调优。我们研究了如何根据攻击期间获得的信息在线调整建议分发来解决这个问题。我们考虑Square攻击，这是一种最先进的基于分数的黑盒攻击，并展示了如何通过学习控制器在攻击期间在线调整建议分布的参数来提高其性能。我们在带有白盒访问的CIFAR10模型上使用基于梯度的端到端训练来训练控制器。我们证明，对于具有黑盒访问的大范围不同模型，在不同的查询机制下，将学习控制器插入攻击可持续提高其黑盒健壮性估计高达20%。我们进一步表明，学习的适应原则很好地移植到其他数据分布，如CIFAR100或ImageNet，以及目标攻击设置。



## **HydraText: Multi-objective Optimization for Adversarial Textual Attack**

HydraText：对抗性文本攻击的多目标优化 cs.CL

**SubmitDate**: 2021-11-02    [paper-pdf](http://arxiv.org/pdf/2111.01528v1)

**Authors**: Shengcai Liu, Ning Lu, Cheng Chen, Chao Qian, Ke Tang

**Abstracts**: The field of adversarial textual attack has significantly grown over the last years, where the commonly considered objective is to craft adversarial examples that can successfully fool the target models. However, the imperceptibility of attacks, which is also an essential objective, is often left out by previous studies. In this work, we advocate considering both objectives at the same time, and propose a novel multi-optimization approach (dubbed HydraText) with provable performance guarantee to achieve successful attacks with high imperceptibility. We demonstrate the efficacy of HydraText through extensive experiments under both score-based and decision-based settings, involving five modern NLP models across five benchmark datasets. In comparison to existing state-of-the-art attacks, HydraText consistently achieves simultaneously higher success rates, lower modification rates, and higher semantic similarity to the original texts. A human evaluation study shows that the adversarial examples crafted by HydraText maintain validity and naturality well. Finally, these examples also exhibit good transferability and can bring notable robustness improvement to the target models by adversarial training.

摘要: 对抗性文本攻击领域在过去几年中有了显著的增长，通常认为的目标是制作能够成功愚弄目标模型的对抗性示例。然而，攻击的隐蔽性也是一个重要的目标，但以往的研究往往忽略了这一点。在这项工作中，我们提倡同时考虑这两个目标，并提出了一种新颖的多重优化方法(称为HydraText)，该方法具有可证明的性能保证，以实现高隐蔽性的成功攻击。我们通过在基于分数和基于决策的设置下的大量实验，在五个基准数据集上涉及五个现代NLP模型，证明了HydraText的有效性。与现有最先进的攻击相比，HydraText始终同时实现更高的成功率、更低的修改率和与原始文本更高的语义相似度。一项人类评价研究表明，HydraText制作的对抗性例子保持了很好的有效性和自然性。最后，这些例子也表现出良好的可移植性，通过对抗性训练可以显著提高目标模型的鲁棒性。



## **Knowledge Cross-Distillation for Membership Privacy**

面向会员隐私的知识交叉蒸馏 cs.CR

Under Review

**SubmitDate**: 2021-11-02    [paper-pdf](http://arxiv.org/pdf/2111.01363v1)

**Authors**: Rishav Chourasia, Batnyam Enkhtaivan, Kunihiro Ito, Junki Mori, Isamu Teranishi, Hikaru Tsuchida

**Abstracts**: A membership inference attack (MIA) poses privacy risks on the training data of a machine learning model. With an MIA, an attacker guesses if the target data are a member of the training dataset. The state-of-the-art defense against MIAs, distillation for membership privacy (DMP), requires not only private data to protect but a large amount of unlabeled public data. However, in certain privacy-sensitive domains, such as medical and financial, the availability of public data is not obvious. Moreover, a trivial method to generate the public data by using generative adversarial networks significantly decreases the model accuracy, as reported by the authors of DMP. To overcome this problem, we propose a novel defense against MIAs using knowledge distillation without requiring public data. Our experiments show that the privacy protection and accuracy of our defense are comparable with those of DMP for the benchmark tabular datasets used in MIA researches, Purchase100 and Texas100, and our defense has much better privacy-utility trade-off than those of the existing defenses without using public data for image dataset CIFAR10.

摘要: 成员关系推理攻击(MIA)会给机器学习模型的训练数据带来隐私风险。使用MIA，攻击者可以猜测目标数据是否为训练数据集的成员。针对MIA的最先进的防御措施，即会员隐私蒸馏(DMP)，不仅需要保护私人数据，还需要大量未标记的公共数据。然而，在某些隐私敏感领域，如医疗和金融，公开数据的可用性并不明显。此外，正如DMP的作者所报告的那样，使用生成性对抗网络来生成公共数据的琐碎方法显著降低了模型的准确性。为了克服这一问题，我们提出了一种新的防御MIA的方法，该方法使用知识蒸馏而不需要公开数据。我们的实验表明，对于MIA研究中使用的基准表格数据集，我们的防御方案的隐私保护和准确性与DMP相当，并且我们的防御方案在隐私效用方面比现有的防御方案具有更好的隐私效用权衡，而不使用公共数据的图像数据集CIFAR10的情况下，我们的防御方案具有更好的隐私效用权衡。



## **ZeBRA: Precisely Destroying Neural Networks with Zero-Data Based Repeated Bit Flip Attack**

Zebra：基于零数据重复位翻转攻击精确摧毁神经网络 cs.LG

14 pages, 3 figures, 5 tables, Accepted at British Machine Vision  Conference (BMVC) 2021

**SubmitDate**: 2021-11-01    [paper-pdf](http://arxiv.org/pdf/2111.01080v1)

**Authors**: Dahoon Park, Kon-Woo Kwon, Sunghoon Im, Jaeha Kung

**Abstracts**: In this paper, we present Zero-data Based Repeated bit flip Attack (ZeBRA) that precisely destroys deep neural networks (DNNs) by synthesizing its own attack datasets. Many prior works on adversarial weight attack require not only the weight parameters, but also the training or test dataset in searching vulnerable bits to be attacked. We propose to synthesize the attack dataset, named distilled target data, by utilizing the statistics of batch normalization layers in the victim DNN model. Equipped with the distilled target data, our ZeBRA algorithm can search vulnerable bits in the model without accessing training or test dataset. Thus, our approach makes the adversarial weight attack more fatal to the security of DNNs. Our experimental results show that 2.0x (CIFAR-10) and 1.6x (ImageNet) less number of bit flips are required on average to destroy DNNs compared to the previous attack method. Our code is available at https://github. com/pdh930105/ZeBRA.

摘要: 本文提出了一种基于零数据的重复位翻转攻击(Zebra)，它通过合成自己的攻击数据集来精确地破坏深度神经网络(DNNs)。以往许多关于对抗性权重攻击的工作不仅需要权重参数，还需要训练或测试数据集来搜索易受攻击的部位。我们提出利用受害者DNN模型中的批归一化层的统计信息来合成攻击数据集，称为提取的目标数据。有了提取的目标数据，我们的斑马算法可以在不访问训练或测试数据集的情况下搜索模型中的易受攻击的位。因此，我们的方法使得敌意加权攻击对DNNs的安全性更加致命。我们的实验结果表明，与以前的攻击方法相比，破坏DNN平均需要减少2.0倍(CIFAR-10)和1.6倍(ImageNet)的比特翻转次数。我们的代码可在https://github.获得com/pdh930105/zebra。



## **Robustness of deep learning algorithms in astronomy -- galaxy morphology studies**

深度学习算法在天文学中的稳健性--星系形态学研究 astro-ph.GA

Accepted in: Fourth Workshop on Machine Learning and the Physical  Sciences (35th Conference on Neural Information Processing Systems;  NeurIPS2021); final version

**SubmitDate**: 2021-11-02    [paper-pdf](http://arxiv.org/pdf/2111.00961v2)

**Authors**: A. Ćiprijanović, D. Kafkes, G. N. Perdue, K. Pedro, G. Snyder, F. J. Sánchez, S. Madireddy, S. M. Wild, B. Nord

**Abstracts**: Deep learning models are being increasingly adopted in wide array of scientific domains, especially to handle high-dimensionality and volume of the scientific data. However, these models tend to be brittle due to their complexity and overparametrization, especially to the inadvertent adversarial perturbations that can appear due to common image processing such as compression or blurring that are often seen with real scientific data. It is crucial to understand this brittleness and develop models robust to these adversarial perturbations. To this end, we study the effect of observational noise from the exposure time, as well as the worst case scenario of a one-pixel attack as a proxy for compression or telescope errors on performance of ResNet18 trained to distinguish between galaxies of different morphologies in LSST mock data. We also explore how domain adaptation techniques can help improve model robustness in case of this type of naturally occurring attacks and help scientists build more trustworthy and stable models.

摘要: 深度学习模型正被越来越多的科学领域所采用，特别是在处理高维和海量的科学数据方面。然而，由于它们的复杂性和过度参数化，这些模型往往是脆弱的，特别是由于在真实科学数据中经常看到的常见图像处理(例如压缩或模糊)可能会出现无意中的对抗性扰动。理解这种脆弱性并开发出对这些对抗性扰动具有健壮性的模型是至关重要的。为此，我们研究了来自曝光时间的观测噪声的影响，以及单像素攻击作为压缩或望远镜误差的替代对ResNet18性能的最坏情况的影响，所训练的ResNet18在LSST模拟数据中区分不同形态的星系。我们还探讨了领域自适应技术如何在这种自然发生的攻击情况下帮助提高模型的健壮性，并帮助科学家建立更可靠和更稳定的模型。



## **A Frequency Perspective of Adversarial Robustness**

对抗性稳健性的频率透视 cs.CV

**SubmitDate**: 2021-10-26    [paper-pdf](http://arxiv.org/pdf/2111.00861v1)

**Authors**: Shishira R Maiya, Max Ehrlich, Vatsal Agarwal, Ser-Nam Lim, Tom Goldstein, Abhinav Shrivastava

**Abstracts**: Adversarial examples pose a unique challenge for deep learning systems. Despite recent advances in both attacks and defenses, there is still a lack of clarity and consensus in the community about the true nature and underlying properties of adversarial examples. A deep understanding of these examples can provide new insights towards the development of more effective attacks and defenses. Driven by the common misconception that adversarial examples are high-frequency noise, we present a frequency-based understanding of adversarial examples, supported by theoretical and empirical findings. Our analysis shows that adversarial examples are neither in high-frequency nor in low-frequency components, but are simply dataset dependent. Particularly, we highlight the glaring disparities between models trained on CIFAR-10 and ImageNet-derived datasets. Utilizing this framework, we analyze many intriguing properties of training robust models with frequency constraints, and propose a frequency-based explanation for the commonly observed accuracy vs. robustness trade-off.

摘要: 对抗性的例子给深度学习系统带来了独特的挑战。尽管最近在攻击和防御方面都取得了进展，但对于对抗性例子的真实性质和潜在属性，社会上仍然缺乏清晰度和共识。深入理解这些例子可以为开发更有效的攻击和防御提供新的见解。由于普遍认为对抗性例子是高频噪声的误解，我们提出了基于频率的对抗性例子的理解，并得到了理论和实证结果的支持。我们的分析表明，对抗性示例既不在高频成分中，也不在低频成分中，而只是简单地依赖于数据集。特别是，我们强调了在CIFAR-10和ImageNet派生的数据集上训练的模型之间的明显差异。利用该框架，我们分析了具有频率约束的训练鲁棒模型的许多有趣的性质，并对通常观察到的精度与鲁棒性之间的权衡提出了一种基于频率的解释。



## **Graph Structural Attack by Spectral Distance**

基于谱距离的图结构攻击 cs.LG

**SubmitDate**: 2021-11-03    [paper-pdf](http://arxiv.org/pdf/2111.00684v2)

**Authors**: Lu Lin, Ethan Blaser, Hongning Wang

**Abstracts**: Graph Convolutional Networks (GCNs) have fueled a surge of interest due to their superior performance on graph learning tasks, but are also shown vulnerability to adversarial attacks. In this paper, an effective graph structural attack is investigated to disrupt graph spectral filters in the Fourier domain. We define the spectral distance based on the eigenvalues of graph Laplacian to measure the disruption of spectral filters. We then generate edge perturbations by simultaneously maximizing a task-specific attack objective and the proposed spectral distance. The experiments demonstrate remarkable effectiveness of the proposed attack in the white-box setting at both training and test time. Our qualitative analysis shows the connection between the attack behavior and the imposed changes on the spectral distribution, which provides empirical evidence that maximizing spectral distance is an effective manner to change the structural property of graphs in the spatial domain and perturb the frequency components in the Fourier domain.

摘要: 图卷积网络(GCNS)由于其在图学习任务上的优异性能引起了人们的极大兴趣，但同时也显示出易受敌意攻击的缺点。本文研究了一种有效的图结构攻击，以破坏傅立叶域中的图谱滤波器。我们根据图的拉普拉斯特征值定义了谱距离来度量谱滤波器的破坏程度。然后，我们通过同时最大化特定于任务的攻击目标和建议的谱距离来产生边缘扰动。实验表明，在白盒环境下，该攻击无论在训练时间还是测试时间都具有显着的有效性。我们的定性分析揭示了攻击行为与谱分布变化之间的关系，这为最大化谱距离是改变图在空间域的结构性质和扰动傅立叶域的频率分量的一种有效方式提供了经验证据。



## **An Actor-Critic Method for Simulation-Based Optimization**

一种基于仿真优化的参与者-批评者方法 cs.LG

**SubmitDate**: 2021-10-31    [paper-pdf](http://arxiv.org/pdf/2111.00435v1)

**Authors**: Kuo Li, Qing-Shan Jia, Jiaqi Yan

**Abstracts**: We focus on a simulation-based optimization problem of choosing the best design from the feasible space. Although the simulation model can be queried with finite samples, its internal processing rule cannot be utilized in the optimization process. We formulate the sampling process as a policy searching problem and give a solution from the perspective of Reinforcement Learning (RL). Concretely, Actor-Critic (AC) framework is applied, where the Actor serves as a surrogate model to predict the performance on unknown designs, whereas the actor encodes the sampling policy to be optimized. We design the updating rule and propose two algorithms for the cases where the feasible spaces are continuous and discrete respectively. Some experiments are designed to validate the effectiveness of proposed algorithms, including two toy examples, which intuitively explain the algorithms, and two more complex tasks, i.e., adversarial attack task and RL task, which validate the effectiveness in large-scale problems. The results show that the proposed algorithms can successfully deal with these problems. Especially note that in the RL task, our methods give a new perspective to robot control by treating the task as a simulation model and solving it by optimizing the policy generating process, while existing works commonly optimize the policy itself directly.

摘要: 重点研究了从可行空间中选择最优设计的基于仿真的优化问题。虽然仿真模型可以用有限样本进行查询，但在优化过程中不能利用其内部处理规则。我们将抽样过程描述为一个策略搜索问题，并从强化学习(RL)的角度给出了解决方案。具体地，采用了Actor-Critic(AC)框架，其中Actor作为代理模型来预测未知设计的性能，而Actor对要优化的采样策略进行编码。针对可行空间为连续和离散的情况，设计了更新规则，并提出了两种算法。设计了一些实验来验证算法的有效性，包括两个玩具示例，直观地解释了算法，以及两个更复杂的任务，即对抗性攻击任务和RL任务，验证了算法在大规模问题中的有效性。实验结果表明，本文提出的算法能够很好地解决这些问题。特别要注意的是，在RL任务中，我们的方法将任务作为一个仿真模型，通过优化策略生成过程来求解，从而为机器人控制提供了一个新的视角，而现有的工作通常是直接优化策略本身。



## **Efficient passive membership inference attack in federated learning**

联邦学习中高效的被动成员推理攻击 cs.LG

Accepted as a poster in NeurIPS 2021 PriML workshop

**SubmitDate**: 2021-10-31    [paper-pdf](http://arxiv.org/pdf/2111.00430v1)

**Authors**: Oualid Zari, Chuan Xu, Giovanni Neglia

**Abstracts**: In cross-device federated learning (FL) setting, clients such as mobiles cooperate with the server to train a global machine learning model, while maintaining their data locally. However, recent work shows that client's private information can still be disclosed to an adversary who just eavesdrops the messages exchanged between the client and the server. For example, the adversary can infer whether the client owns a specific data instance, which is called a passive membership inference attack. In this paper, we propose a new passive inference attack that requires much less computation power and memory than existing methods. Our empirical results show that our attack achieves a higher accuracy on CIFAR100 dataset (more than $4$ percentage points) with three orders of magnitude less memory space and five orders of magnitude less calculations.

摘要: 在跨设备联合学习(FL)设置中，诸如移动设备之类的客户端与服务器协作来训练全局机器学习模型，同时在本地维护它们的数据。然而，最近的研究表明，客户端的私人信息仍然可以泄露给仅仅窃听客户端和服务器之间交换的消息的对手。例如，对手可以推断客户端是否拥有特定的数据实例，这称为被动成员关系推理攻击。在本文中，我们提出了一种新的被动推理攻击，它比现有的方法需要更少的计算能力和内存。实验结果表明，我们的攻击在CIFAR100数据集上达到了更高的准确率(超过4美元百分点)，内存空间减少了3个数量级，计算量减少了5个数量级。



## **AdvCodeMix: Adversarial Attack on Code-Mixed Data**

AdvCodeMix：混合代码数据的对抗性攻击 cs.CL

Accepted to CODS-COMAD 2022

**SubmitDate**: 2021-10-30    [paper-pdf](http://arxiv.org/pdf/2111.00350v1)

**Authors**: Sourya Dipta Das, Ayan Basak, Soumil Mandal, Dipankar Das

**Abstracts**: Research on adversarial attacks are becoming widely popular in the recent years. One of the unexplored areas where prior research is lacking is the effect of adversarial attacks on code-mixed data. Therefore, in the present work, we have explained the first generalized framework on text perturbation to attack code-mixed classification models in a black-box setting. We rely on various perturbation techniques that preserve the semantic structures of the sentences and also obscure the attacks from the perception of a human user. The present methodology leverages the importance of a token to decide where to attack by employing various perturbation strategies. We test our strategies on various sentiment classification models trained on Bengali-English and Hindi-English code-mixed datasets, and reduce their F1-scores by nearly 51 % and 53 % respectively, which can be further reduced if a larger number of tokens are perturbed in a given sentence.

摘要: 对抗性攻击的研究在近几年得到了广泛的重视。先前缺乏研究的未探索领域之一是对抗性攻击对代码混合数据的影响。因此，在目前的工作中，我们解释了第一个针对文本扰动的通用框架，用于攻击黑盒环境下的代码混合分类模型。我们依赖于各种扰动技术，这些技术保留了句子的语义结构，同时也从人类用户的感知中掩盖了攻击。本方法利用令牌的重要性通过采用各种扰动策略来决定攻击的位置。我们在孟加拉-英语和印地-英语代码混合数据集上训练的各种情感分类模型上测试了我们的策略，它们的F1-得分分别降低了近51%和53%，如果给定句子中扰动了大量的标记，还可以进一步降低F1-得分。



## **Get Fooled for the Right Reason: Improving Adversarial Robustness through a Teacher-guided Curriculum Learning Approach**

为了正确的理由而被愚弄：通过教师指导的课程学习方法提高对手的健壮性 cs.LG

16 pages, 9 figures, Accepted at NeurIPS 2021, Code at  https://github.com/sowgali/Get-Fooled-for-the-Right-Reason

**SubmitDate**: 2021-10-30    [paper-pdf](http://arxiv.org/pdf/2111.00295v1)

**Authors**: Anindya Sarkar, Anirban Sarkar, Sowrya Gali, Vineeth N Balasubramanian

**Abstracts**: Current SOTA adversarially robust models are mostly based on adversarial training (AT) and differ only by some regularizers either at inner maximization or outer minimization steps. Being repetitive in nature during the inner maximization step, they take a huge time to train. We propose a non-iterative method that enforces the following ideas during training. Attribution maps are more aligned to the actual object in the image for adversarially robust models compared to naturally trained models. Also, the allowed set of pixels to perturb an image (that changes model decision) should be restricted to the object pixels only, which reduces the attack strength by limiting the attack space. Our method achieves significant performance gains with a little extra effort (10-20%) over existing AT models and outperforms all other methods in terms of adversarial as well as natural accuracy. We have performed extensive experimentation with CIFAR-10, CIFAR-100, and TinyImageNet datasets and reported results against many popular strong adversarial attacks to prove the effectiveness of our method.

摘要: 目前的SOTA对抗性鲁棒模型大多基于对抗性训练(AT)，仅在内极大化或外极小化阶段有一些正则化的不同之处。在内心最大化的过程中，他们本质上是重复的，需要花费大量的时间来训练。我们提出了一种非迭代方法，在训练过程中实施了以下思想。与自然训练的模型相比，对于相反的鲁棒模型，属性图更符合图像中的实际对象。此外，允许扰乱图像的像素集(改变模型决策)应该仅限于对象像素，这通过限制攻击空间来降低攻击强度。与已有的AT模型相比，我们的方法以较小的额外工作量(10-20%)获得了显着的性能提升，并且在对抗性和自然准确率方面优于所有其他方法。我们在CIFAR-10、CIFAR-100和TinyImageNet数据集上进行了广泛的实验，并报告了对许多流行的强对手攻击的结果，以证明我们的方法的有效性。



## **Improving the quality of generative models through Smirnov transformation**

利用Smirnov变换提高产生式模型的质量 cs.LG

28 pages, 16 Figures, 4 Tables

**SubmitDate**: 2021-10-29    [paper-pdf](http://arxiv.org/pdf/2110.15914v1)

**Authors**: Ángel González-Prieto, Alberto Mozo, Sandra Gómez-Canaval, Edgar Talavera

**Abstracts**: Solving the convergence issues of Generative Adversarial Networks (GANs) is one of the most outstanding problems in generative models. In this work, we propose a novel activation function to be used as output of the generator agent. This activation function is based on the Smirnov probabilistic transformation and it is specifically designed to improve the quality of the generated data. In sharp contrast with previous works, our activation function provides a more general approach that deals not only with the replication of categorical variables but with any type of data distribution (continuous or discrete). Moreover, our activation function is derivable and therefore, it can be seamlessly integrated in the backpropagation computations during the GAN training processes. To validate this approach, we evaluate our proposal against two different data sets: a) an artificially rendered data set containing a mixture of discrete and continuous variables, and b) a real data set of flow-based network traffic data containing both normal connections and cryptomining attacks. To evaluate the fidelity of the generated data, we analyze both their results in terms of quality measures of statistical nature and also regarding the use of these synthetic data to feed a nested machine learning-based classifier. The experimental results evince a clear outperformance of the GAN network tuned with this new activation function with respect to both a na\"ive mean-based generator and a standard GAN. The quality of the data is so high that the generated data can fully substitute real data for training the nested classifier without a fall in the obtained accuracy. This result encourages the use of GANs to produce high-quality synthetic data that are applicable in scenarios in which data privacy must be guaranteed.

摘要: 解决产生式对抗性网络(GANS)的收敛问题是产生式模型中最突出的问题之一。在这项工作中，我们提出了一种新的激活函数作为生成器Agent的输出。该激活函数基于斯米尔诺夫概率变换，它是专门为提高生成数据的质量而设计的。与前人的工作形成鲜明对比的是，我们的激活函数提供了一种更通用的方法，不仅可以处理分类变量的复制，而且可以处理任何类型的数据分布(连续或离散)。此外，我们的激活函数是可导的，因此它可以无缝地集成到GaN训练过程中的反向传播计算中。为了验证该方法，我们在两个不同的数据集上评估了我们的建议：a)包含离散变量和连续变量的混合的人工渲染数据集，以及b)包含正常连接和密码攻击的基于流的网络流量数据的真实数据集。为了评估生成数据的保真度，我们根据统计性质的质量度量以及使用这些合成数据来馈送基于机器学习的嵌套分类器来分析它们的结果。实验结果表明，与基于朴素均值的生成器和标准的GAN网络相比，使用这种新的激活函数调整的GAN网络具有明显的优异性能，数据质量非常高，生成的数据可以充分利用真实数据来训练嵌套分类器，而不会降低获得的精度，这一结果鼓励使用GANS生成高质量的合成数据，这些数据适用于必须保证数据保密性的场景。这一结果鼓励了GANS用于生成高质量的合成数据，这些数据适用于数据隐私必须得到保证的场景。这一结果鼓励了GANS用于生成高质量的合成数据，这些数据适用于必须保证数据隐私的场景。



## **Attacking Video Recognition Models with Bullet-Screen Comments**

用弹幕评论攻击视频识别模型 cs.CV

**SubmitDate**: 2021-10-29    [paper-pdf](http://arxiv.org/pdf/2110.15629v1)

**Authors**: Kai Chen, Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstracts**: Recent research has demonstrated that Deep Neural Networks (DNNs) are vulnerable to adversarial patches which introducing perceptible but localized changes to the input. Nevertheless, existing approaches have focused on generating adversarial patches on images, their counterparts in videos have been less explored. Compared with images, attacking videos is much more challenging as it needs to consider not only spatial cues but also temporal cues. To close this gap, we introduce a novel adversarial attack in this paper, the bullet-screen comment (BSC) attack, which attacks video recognition models with BSCs. Specifically, adversarial BSCs are generated with a Reinforcement Learning (RL) framework, where the environment is set as the target model and the agent plays the role of selecting the position and transparency of each BSC. By continuously querying the target models and receiving feedback, the agent gradually adjusts its selection strategies in order to achieve a high fooling rate with non-overlapping BSCs. As BSCs can be regarded as a kind of meaningful patch, adding it to a clean video will not affect people' s understanding of the video content, nor will arouse people' s suspicion. We conduct extensive experiments to verify the effectiveness of the proposed method. On both UCF-101 and HMDB-51 datasets, our BSC attack method can achieve about 90\% fooling rate when attack three mainstream video recognition models, while only occluding \textless 8\% areas in the video.

摘要: 最近的研究表明，深度神经网络(DNNs)很容易受到敌意补丁的攻击，这些补丁会给输入带来可感知的局部变化。然而，现有的方法主要集中在生成图像上的对抗性补丁，而对视频中的对应补丁的研究较少。与图像相比，攻击视频更具挑战性，因为它不仅需要考虑空间线索，还需要考虑时间线索。为了缩小这一差距，本文引入了一种新的对抗性攻击，即弹幕评论(BSC)攻击，它利用弹幕评论攻击视频识别模型。具体地说，利用强化学习(RL)框架生成对抗性BSC，其中环境被设置为目标模型，Agent扮演选择每个BSC的位置和透明度的角色。通过不断查询目标模型并接收反馈，Agent逐渐调整其选择策略，以获得不重叠的BSC的较高愚弄率。由于BSCS可以看作是一种有意义的补丁，将其添加到干净的视频中不会影响人们对视频内容的理解，也不会引起人们的怀疑。为了验证该方法的有效性，我们进行了大量的实验。在UCF-101和HMDB-51两个数据集上，我们的BSC攻击方法在攻击三种主流视频识别模型时，仅对视频中的8个无遮挡区域进行攻击，可以达到90%左右的蒙骗率。



## **Bridge the Gap Between CV and NLP! A Gradient-based Textual Adversarial Attack Framework**

弥合简历和NLP之间的鸿沟！一种基于梯度的文本对抗性攻击框架 cs.CL

Work on progress

**SubmitDate**: 2021-10-28    [paper-pdf](http://arxiv.org/pdf/2110.15317v1)

**Authors**: Lifan Yuan, Yichi Zhang, Yangyi Chen, Wei Wei

**Abstracts**: Despite great success on many machine learning tasks, deep neural networks are still vulnerable to adversarial samples. While gradient-based adversarial attack methods are well-explored in the field of computer vision, it is impractical to directly apply them in natural language processing due to the discrete nature of text. To bridge this gap, we propose a general framework to adapt existing gradient-based methods to craft textual adversarial samples. In this framework, gradient-based continuous perturbations are added to the embedding layer and are amplified in the forward propagation process. Then the final perturbed latent representations are decoded with a mask language model head to obtain potential adversarial samples. In this paper, we instantiate our framework with \textbf{T}extual \textbf{P}rojected \textbf{G}radient \textbf{D}escent (\textbf{TPGD}). We conduct comprehensive experiments to evaluate our framework by performing transfer black-box attacks on BERT, RoBERTa and ALBERT on three benchmark datasets. Experimental results demonstrate our method achieves an overall better performance and produces more fluent and grammatical adversarial samples compared to strong baseline methods. All the code and data will be made public.

摘要: 尽管深度神经网络在许多机器学习任务中取得了巨大的成功，但它仍然容易受到敌意样本的影响。虽然基于梯度的对抗性攻击方法在计算机视觉领域得到了很好的探索，但由于文本的离散性，将其直接应用于自然语言处理是不切实际的。为了弥补这一差距，我们提出了一个通用框架，以适应现有的基于梯度的方法来制作文本对抗性样本。在该框架中，基于梯度的连续扰动被添加到嵌入层，并在前向传播过程中被放大。然后用掩码语言模型头部对最终扰动的潜在表示进行解码，得到潜在的对抗性样本。在本文中，我们用\textbf{T}extual\textbf{P}rojected\textbf{G}Radient\textbf{D}light(\textbf{tpgd})实例化我们的框架。我们通过在三个基准数据集上对Bert、Roberta和Albert进行传输黑盒攻击，对我们的框架进行了全面的测试。实验结果表明，与强基线方法相比，我们的方法取得了总体上更好的性能，生成了更流畅、更具语法意义的对抗性样本。所有的代码和数据都将公之于众。



## **The magnitude vector of images**

图像的幅值矢量 cs.LG

15 pages, 8 figures

**SubmitDate**: 2021-10-28    [paper-pdf](http://arxiv.org/pdf/2110.15188v1)

**Authors**: Michael F. Adamer, Leslie O'Bray, Edward De Brouwer, Bastian Rieck, Karsten Borgwardt

**Abstracts**: The magnitude of a finite metric space is a recently-introduced invariant quantity. Despite beneficial theoretical and practical properties, such as a general utility for outlier detection, and a close connection to Laplace radial basis kernels, magnitude has received little attention by the machine learning community so far. In this work, we investigate the properties of magnitude on individual images, with each image forming its own metric space. We show that the known properties of outlier detection translate to edge detection in images and we give supporting theoretical justifications. In addition, we provide a proof of concept of its utility by using a novel magnitude layer to defend against adversarial attacks. Since naive magnitude calculations may be computationally prohibitive, we introduce an algorithm that leverages the regular structure of images to dramatically reduce the computational cost.

摘要: 有限度量空间的量级是最近引入的不变量。尽管有一些有益的理论和实践性质，例如用于异常值检测的通用工具，以及与Laplace径向基核的密切联系，但到目前为止，量值几乎没有受到机器学习界的关注。在这项工作中，我们研究了单个图像的量值性质，每个图像形成了自己的度量空间。我们证明了孤立点检测的已知性质转化为图像中的边缘检测，并给出了支持的理论证明。此外，我们通过使用一种新的幅值层来防御对手攻击，给出了它的实用性的概念证明。由于朴素的幅值计算可能在计算上是令人望而却步的，我们引入了一种算法，该算法利用图像的规则结构来极大地降低计算成本。



## **Authentication Attacks on Projection-based Cancelable Biometric Schemes**

对基于投影的可取消生物特征识别方案的认证攻击 cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2021-10-28    [paper-pdf](http://arxiv.org/pdf/2110.15163v1)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.

摘要: 可取消生物测定方案旨在通过将诸如密码、存储的秘密或盐等用户特定令牌与生物测定数据相结合来生成安全的生物测定模板。这种类型的变换被构造为生物测定变换与特征提取算法的组合。可撤销生物特征识别方案的安全性要求涉及模板的不可逆性、不可链接性和可撤销性，而不损失比较的准确性。虽然最近几个方案在这些要求方面受到攻击，但据我们所知，这种组合物的完全可逆性以产生碰撞的生物测定特征，特别是呈现攻击，从未被证明。本文利用整数线性规划(ILP)和二次约束二次规划(QCQP)对传统的可取消方案进行了形式化描述。解决这些优化问题允许对手稍微更改其指纹图像，以便冒充任何个人。此外，在更严重的情况下，可以同时冒充几个人。



## **Adversarial Robustness in Multi-Task Learning: Promises and Illusions**

多任务学习中的对抗性稳健性：承诺与幻想 cs.LG

**SubmitDate**: 2021-10-26    [paper-pdf](http://arxiv.org/pdf/2110.15053v1)

**Authors**: Salah Ghamizi, Maxime Cordy, Mike Papadakis, Yves Le Traon

**Abstracts**: Vulnerability to adversarial attacks is a well-known weakness of Deep Neural networks. While most of the studies focus on single-task neural networks with computer vision datasets, very little research has considered complex multi-task models that are common in real applications. In this paper, we evaluate the design choices that impact the robustness of multi-task deep learning networks. We provide evidence that blindly adding auxiliary tasks, or weighing the tasks provides a false sense of robustness. Thereby, we tone down the claim made by previous research and study the different factors which may affect robustness. In particular, we show that the choice of the task to incorporate in the loss function are important factors that can be leveraged to yield more robust models.

摘要: 对敌意攻击的脆弱性是深度神经网络的一个众所周知的弱点。虽然大多数研究集中在具有计算机视觉数据集的单任务神经网络，但很少有研究考虑实际应用中常见的复杂多任务模型。在本文中，我们评估了影响多任务深度学习网络健壮性的设计选择。我们提供的证据表明，盲目添加辅助任务或对任务进行加权会带来一种错误的健壮感。因此，我们淡化了以往研究的结论，并研究了可能影响稳健性的不同因素。特别地，我们表明，选择要纳入损失函数的任务是可以用来产生更健壮的模型的重要因素。



## **AEVA: Black-box Backdoor Detection Using Adversarial Extreme Value Analysis**

AEVA：基于对抗性极值分析的黑盒后门检测 cs.LG

**SubmitDate**: 2021-10-29    [paper-pdf](http://arxiv.org/pdf/2110.14880v2)

**Authors**: Junfeng Guo, Ang Li, Cong Liu

**Abstracts**: Deep neural networks (DNNs) are proved to be vulnerable against backdoor attacks. A backdoor is often embedded in the target DNNs through injecting a backdoor trigger into training examples, which can cause the target DNNs misclassify an input attached with the backdoor trigger. Existing backdoor detection methods often require the access to the original poisoned training data, the parameters of the target DNNs, or the predictive confidence for each given input, which are impractical in many real-world applications, e.g., on-device deployed DNNs. We address the black-box hard-label backdoor detection problem where the DNN is fully black-box and only its final output label is accessible. We approach this problem from the optimization perspective and show that the objective of backdoor detection is bounded by an adversarial objective. Further theoretical and empirical studies reveal that this adversarial objective leads to a solution with highly skewed distribution; a singularity is often observed in the adversarial map of a backdoor-infected example, which we call the adversarial singularity phenomenon. Based on this observation, we propose the adversarial extreme value analysis(AEVA) to detect backdoors in black-box neural networks. AEVA is based on an extreme value analysis of the adversarial map, computed from the monte-carlo gradient estimation. Evidenced by extensive experiments across multiple popular tasks and backdoor attacks, our approach is shown effective in detecting backdoor attacks under the black-box hard-label scenarios.

摘要: 深度神经网络(DNNs)被证明是易受后门攻击的。通过将后门触发器注入到训练示例中，通常将后门嵌入到目标DNN中，这可能导致目标DNN对与后门触发器附加的输入进行错误分类。现有的后门检测方法通常需要访问原始有毒训练数据、目标DNN的参数或每个给定输入的预测置信度，这在许多真实世界应用中是不切实际的，例如在设备上部署的DNN。我们解决了黑盒硬标签后门检测问题，其中DNN是完全黑盒的，并且只有其最终输出标签是可访问的。我们从优化的角度来研究这个问题，并证明了后门检测的目标是由一个对抗性目标限定的。进一步的理论和实证研究表明，这种对抗性目标导致了一个具有高度偏态分布的解决方案；在一个被后门感染的例子的对抗性地图中经常观察到一个奇点，我们称之为对抗性奇点现象。基于这一观察，我们提出了对抗性极值分析(AEVA)来检测黑盒神经网络中的后门。AEVA是基于对敌方地图的极值分析，通过蒙特卡洛梯度估计计算出来的。通过对多个流行任务和后门攻击的大量实验证明，我们的方法在黑盒硬标签场景下检测后门攻击是有效的。



