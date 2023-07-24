# Latest Adversarial Attack Papers
**update at 2023-07-24 11:33:42**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. OUTFOX: LLM-generated Essay Detection through In-context Learning with Adversarially Generated Examples**

Outfox：基于上下文学习的LLM生成的文章检测与恶意生成的示例 cs.CL

**SubmitDate**: 2023-07-21    [abs](http://arxiv.org/abs/2307.11729v1) [paper-pdf](http://arxiv.org/pdf/2307.11729v1)

**Authors**: Ryuto Koike, Masahiro Kaneko, Naoaki Okazaki

**Abstract**: Large Language Models (LLMs) have achieved human-level fluency in text generation, making it difficult to distinguish between human-written and LLM-generated texts. This poses a growing risk of misuse of LLMs and demands the development of detectors to identify LLM-generated texts. However, existing detectors degrade detection accuracy by simply paraphrasing LLM-generated texts. Furthermore, the effectiveness of these detectors in real-life situations, such as when students use LLMs for writing homework assignments (e.g., essays) and quickly learn how to evade these detectors, has not been explored. In this paper, we propose OUTFOX, a novel framework that improves the robustness of LLM-generated-text detectors by allowing both the detector and the attacker to consider each other's output and apply this to the domain of student essays. In our framework, the attacker uses the detector's prediction labels as examples for in-context learning and adversarially generates essays that are harder to detect. While the detector uses the adversarially generated essays as examples for in-context learning to learn to detect essays from a strong attacker. Our experiments show that our proposed detector learned in-context from the attacker improves the detection performance on the attacked dataset by up to +41.3 point F1-score. While our proposed attacker can drastically degrade the performance of the detector by up to -57.0 point F1-score compared to the paraphrasing method.

摘要: 大型语言模型(LLM)在文本生成方面达到了人类水平的流畅性，使得区分人类编写的文本和LLM生成的文本变得困难。这带来了滥用LLMS的越来越大的风险，并要求开发检测器来识别LLM生成的文本。然而，现有的检测器通过简单地解释LLM生成的文本来降低检测精度。此外，这些检测器在现实生活中的有效性还没有被探索过，例如当学生使用LLMS来写作业(例如，论文)并迅速学习如何逃避这些检测器时。在本文中，我们提出了一种新的框架Outfox，它通过允许检测器和攻击者考虑彼此的输出来提高LLM生成的文本检测器的健壮性，并将其应用到学生作文领域。在我们的框架中，攻击者使用检测器的预测标签作为上下文学习的示例，并恶意生成更难检测的文章。而检测器使用恶意生成的文章作为上下文学习的示例，以学习检测来自强大攻击者的文章。我们的实验表明，从攻击者那里学习的上下文中学习的检测器在攻击数据集上的检测性能提高了高达41.3点F1-Score。而我们提出的攻击者可以大幅降低检测器的性能，与改述方法相比，最高可达-57.0点F1分数。



## **2. (Ab)using Images and Sounds for Indirect Instruction Injection in Multi-Modal LLMs**

(AB)在多模式LLMS中使用图像和声音进行间接指令注入 cs.CR

**SubmitDate**: 2023-07-21    [abs](http://arxiv.org/abs/2307.10490v2) [paper-pdf](http://arxiv.org/pdf/2307.10490v2)

**Authors**: Eugene Bagdasaryan, Tsung-Yin Hsieh, Ben Nassi, Vitaly Shmatikov

**Abstract**: We demonstrate how images and sounds can be used for indirect prompt and instruction injection in multi-modal LLMs. An attacker generates an adversarial perturbation corresponding to the prompt and blends it into an image or audio recording. When the user asks the (unmodified, benign) model about the perturbed image or audio, the perturbation steers the model to output the attacker-chosen text and/or make the subsequent dialog follow the attacker's instruction. We illustrate this attack with several proof-of-concept examples targeting LLaVa and PandaGPT.

摘要: 我们演示了图像和声音如何在多模式LLMS中用于间接提示和指令注入。攻击者生成与提示相对应的对抗性扰动，并将其混合到图像或音频记录中。当用户询问(未修改的、良性的)模型有关受干扰的图像或音频时，扰动引导模型输出攻击者选择的文本和/或使后续对话遵循攻击者的指令。我们用几个针对LLaVa和PandaGPT的概念验证示例来说明这种攻击。



## **3. Fast Adaptive Test-Time Defense with Robust Features**

具有健壮性的快速自适应测试时间防御 cs.LG

**SubmitDate**: 2023-07-21    [abs](http://arxiv.org/abs/2307.11672v1) [paper-pdf](http://arxiv.org/pdf/2307.11672v1)

**Authors**: Anurag Singh, Mahalakshmi Sabanayagam, Krikamol Muandet, Debarghya Ghoshdastidar

**Abstract**: Adaptive test-time defenses are used to improve the robustness of deep neural networks to adversarial examples. However, existing methods significantly increase the inference time due to additional optimization on the model parameters or the input at test time. In this work, we propose a novel adaptive test-time defense strategy that is easy to integrate with any existing (robust) training procedure without additional test-time computation. Based on the notion of robustness of features that we present, the key idea is to project the trained models to the most robust feature space, thereby reducing the vulnerability to adversarial attacks in non-robust directions. We theoretically show that the top eigenspace of the feature matrix are more robust for a generalized additive model and support our argument for a large width neural network with the Neural Tangent Kernel (NTK) equivalence. We conduct extensive experiments on CIFAR-10 and CIFAR-100 datasets for several robustness benchmarks, including the state-of-the-art methods in RobustBench, and observe that the proposed method outperforms existing adaptive test-time defenses at much lower computation costs.

摘要: 使用自适应测试时间防御来提高深度神经网络对敌意例子的鲁棒性。然而，现有的方法由于在测试时对模型参数或输入进行了额外的优化，从而大大增加了推理时间。在这项工作中，我们提出了一种新的自适应测试时间防御策略，该策略易于与任何现有的(鲁棒)训练过程集成，而不需要额外的测试时间计算。基于我们提出的特征的稳健性的概念，其关键思想是将训练好的模型投影到最健壮的特征空间，从而降低在非健壮方向上对敌方攻击的脆弱性。我们从理论上证明了特征矩阵的顶特征空间对于广义加性模型更具稳健性，并支持我们关于具有神经切核(NTK)等价的大宽度神经网络的论点。我们在CIFAR-10和CIFAR-100数据集上对几个健壮性基准测试进行了广泛的实验，包括RobustBitch中的最新方法，并观察到所提出的方法以更低的计算代价获得了比现有的自适应测试时间防御方法更好的性能。



## **4. Improving Viewpoint Robustness for Visual Recognition via Adversarial Training**

通过对抗性训练提高视觉识别的视点稳健性 cs.CV

14 pages, 12 figures. arXiv admin note: substantial text overlap with  arXiv:2307.10235

**SubmitDate**: 2023-07-21    [abs](http://arxiv.org/abs/2307.11528v1) [paper-pdf](http://arxiv.org/pdf/2307.11528v1)

**Authors**: Shouwei Ruan, Yinpeng Dong, Hang Su, Jianteng Peng, Ning Chen, Xingxing Wei

**Abstract**: Viewpoint invariance remains challenging for visual recognition in the 3D world, as altering the viewing directions can significantly impact predictions for the same object. While substantial efforts have been dedicated to making neural networks invariant to 2D image translations and rotations, viewpoint invariance is rarely investigated. Motivated by the success of adversarial training in enhancing model robustness, we propose Viewpoint-Invariant Adversarial Training (VIAT) to improve the viewpoint robustness of image classifiers. Regarding viewpoint transformation as an attack, we formulate VIAT as a minimax optimization problem, where the inner maximization characterizes diverse adversarial viewpoints by learning a Gaussian mixture distribution based on the proposed attack method GMVFool. The outer minimization obtains a viewpoint-invariant classifier by minimizing the expected loss over the worst-case viewpoint distributions that can share the same one for different objects within the same category. Based on GMVFool, we contribute a large-scale dataset called ImageNet-V+ to benchmark viewpoint robustness. Experimental results show that VIAT significantly improves the viewpoint robustness of various image classifiers based on the diversity of adversarial viewpoints generated by GMVFool. Furthermore, we propose ViewRS, a certified viewpoint robustness method that provides a certified radius and accuracy to demonstrate the effectiveness of VIAT from the theoretical perspective.

摘要: 视点不变性对3D世界中的视觉识别仍然具有挑战性，因为改变观察方向可以显著影响对同一对象的预测。虽然人们致力于使神经网络对2D图像平移和旋转具有不变性，但很少有人研究视点不变性。基于对抗性训练在增强模型稳健性方面的成功，我们提出了视点不变对抗性训练(VAT)来提高图像分类器的视点稳健性。将视点变换视为一种攻击，将VAT问题描述为一个极小极大优化问题，其中内极大化通过学习基于GMVFool的高斯混合分布来刻画不同的敌方视点。外部最小化通过最小化最坏情况下的视点分布的期望损失来获得视点不变的分类器，所述最坏情况的视点分布可以对于相同类别中的不同对象共享相同的视点分布。基于GMVFool，我们贡献了一个称为ImageNet-V+的大规模数据集来测试视点稳健性。实验结果表明，基于GMVFool生成的对抗性视点的多样性，VIAT显著提高了各种图像分类器的视点稳健性。此外，我们提出了VIERS，这是一种验证的视点稳健性方法，它提供了验证的半径和精度，从理论上证明了VAT的有效性。



## **5. Improving Transferability of Adversarial Examples via Bayesian Attacks**

通过贝叶斯攻击提高对抗性实例的可转移性 cs.LG

**SubmitDate**: 2023-07-21    [abs](http://arxiv.org/abs/2307.11334v1) [paper-pdf](http://arxiv.org/pdf/2307.11334v1)

**Authors**: Qizhang Li, Yiwen Guo, Xiaochen Yang, Wangmeng Zuo, Hao Chen

**Abstract**: This paper presents a substantial extension of our work published at ICLR. Our ICLR work advocated for enhancing transferability in adversarial examples by incorporating a Bayesian formulation into model parameters, which effectively emulates the ensemble of infinitely many deep neural networks, while, in this paper, we introduce a novel extension by incorporating the Bayesian formulation into the model input as well, enabling the joint diversification of both the model input and model parameters. Our empirical findings demonstrate that: 1) the combination of Bayesian formulations for both the model input and model parameters yields significant improvements in transferability; 2) by introducing advanced approximations of the posterior distribution over the model input, adversarial transferability achieves further enhancement, surpassing all state-of-the-arts when attacking without model fine-tuning. Moreover, we propose a principled approach to fine-tune model parameters in such an extended Bayesian formulation. The derived optimization objective inherently encourages flat minima in the parameter space and input space. Extensive experiments demonstrate that our method achieves a new state-of-the-art on transfer-based attacks, improving the average success rate on ImageNet and CIFAR-10 by 19.14% and 2.08%, respectively, when comparing with our ICLR basic Bayesian method. We will make our code publicly available.

摘要: 这篇论文是对我们在ICLR上发表的工作的实质性扩展。我们的ICLR工作主张通过将贝叶斯公式纳入模型参数来增强对抗性例子的可转移性，这有效地模拟了无限多个深度神经网络的集成，而在本文中，我们引入了一种新的扩展，将贝叶斯公式也纳入到模型输入中，使得模型输入和模型参数能够联合多样化。我们的实证结果表明：1)结合模型输入和模型参数的贝叶斯公式显著提高了可转移性；2)通过引入对模型输入的后验分布的高级近似，对手的可转移性得到了进一步的增强，在不进行模型微调的情况下，超过了所有最新的攻击。此外，我们还提出了一种在这种扩展贝叶斯公式中微调模型参数的原则性方法。所导出的优化目标内在地鼓励参数空间和输入空间中的平坦极小。大量实验表明，与ICLR基本贝叶斯方法相比，我们的方法在基于传输的攻击上达到了新的水平，在ImageNet和CIFAR-10上的平均成功率分别提高了19.14%和2.08%。我们将公开我们的代码。



## **6. Epsilon*: Privacy Metric for Machine Learning Models**

Epsilon*：机器学习模型的隐私度量 cs.LG

**SubmitDate**: 2023-07-21    [abs](http://arxiv.org/abs/2307.11280v1) [paper-pdf](http://arxiv.org/pdf/2307.11280v1)

**Authors**: Diana M. Negoescu, Humberto Gonzalez, Saad Eddin Al Orjany, Jilei Yang, Yuliia Lut, Rahul Tandra, Xiaowen Zhang, Xinyi Zheng, Zach Douglas, Vidita Nolkha, Parvez Ahammad, Gennady Samorodnitsky

**Abstract**: We introduce Epsilon*, a new privacy metric for measuring the privacy risk of a single model instance prior to, during, or after deployment of privacy mitigation strategies. The metric does not require access to the training data sampling or model training algorithm. Epsilon* is a function of true positive and false positive rates in a hypothesis test used by an adversary in a membership inference attack. We distinguish between quantifying the privacy loss of a trained model instance and quantifying the privacy loss of the training mechanism which produces this model instance. Existing approaches in the privacy auditing literature provide lower bounds for the latter, while our metric provides a lower bound for the former by relying on an (${\epsilon}$,${\delta}$)-type of quantification of the privacy of the trained model instance. We establish a relationship between these lower bounds and show how to implement Epsilon* to avoid numerical and noise amplification instability. We further show in experiments on benchmark public data sets that Epsilon* is sensitive to privacy risk mitigation by training with differential privacy (DP), where the value of Epsilon* is reduced by up to 800% compared to the Epsilon* values of non-DP trained baseline models. This metric allows privacy auditors to be independent of model owners, and enables all decision-makers to visualize the privacy-utility landscape to make informed decisions regarding the trade-offs between model privacy and utility.

摘要: 我们引入了Epsilon*，这是一种新的隐私度量标准，用于在部署隐私缓解策略之前、期间或之后衡量单个模型实例的隐私风险。该指标不需要访问训练数据采样或模型训练算法。Epsilon*是对手在成员关系推断攻击中使用的假设检验中真阳性和假阳性率的函数。我们区分量化训练模型实例的隐私损失和量化产生该模型实例的训练机制的隐私损失。隐私审计文献中的现有方法为后者提供了下界，而我们的度量通过依赖于($\epsilon}$，${\Delta}$)类型的训练模型实例的隐私量化来为前者提供下界。我们建立了这些下界之间的关系，并展示了如何实现Epsilon*以避免数值和噪声放大的不稳定性。我们在基准公共数据集上的实验进一步表明，Epsilon*通过使用差异隐私(DP)进行训练对隐私风险缓解非常敏感，其中Epsilon*的值与未使用DP训练的基线模型的Epsilon*值相比降低了800%。这一指标允许隐私审核员独立于模型所有者，并使所有决策者能够可视化隐私-效用环境，以便就模型隐私和效用之间的权衡做出明智的决策。



## **7. Preprocessors Matter! Realistic Decision-Based Attacks on Machine Learning Systems**

预处理器很重要！基于现实决策的机器学习系统攻击 cs.CR

ICML 2023. Code can be found at  https://github.com/google-research/preprocessor-aware-black-box-attack

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2210.03297v2) [paper-pdf](http://arxiv.org/pdf/2210.03297v2)

**Authors**: Chawin Sitawarin, Florian Tramèr, Nicholas Carlini

**Abstract**: Decision-based attacks construct adversarial examples against a machine learning (ML) model by making only hard-label queries. These attacks have mainly been applied directly to standalone neural networks. However, in practice, ML models are just one component of a larger learning system. We find that by adding a single preprocessor in front of a classifier, state-of-the-art query-based attacks are up to 7$\times$ less effective at attacking a prediction pipeline than at attacking the model alone. We explain this discrepancy by the fact that most preprocessors introduce some notion of invariance to the input space. Hence, attacks that are unaware of this invariance inevitably waste a large number of queries to re-discover or overcome it. We, therefore, develop techniques to (i) reverse-engineer the preprocessor and then (ii) use this extracted information to attack the end-to-end system. Our preprocessors extraction method requires only a few hundred queries, and our preprocessor-aware attacks recover the same efficacy as when attacking the model alone. The code can be found at https://github.com/google-research/preprocessor-aware-black-box-attack.

摘要: 基于决策的攻击通过只进行硬标签查询来构建对抗机器学习(ML)模型的例子。这些攻击主要直接应用于独立的神经网络。然而，在实践中，ML模型只是更大的学习系统的一个组件。我们发现，通过在分类器前面添加单个预处理器，最新的基于查询的攻击在攻击预测流水线时的效率比单独攻击模型低7倍。我们解释这种差异的原因是，大多数预处理器都在输入空间引入了某种不变性的概念。因此，没有意识到这种不变性的攻击不可避免地会浪费大量查询来重新发现或克服它。因此，我们开发了一些技术来(I)对预处理器进行逆向工程，然后(Ii)使用这些提取的信息来攻击端到端系统。我们的预处理器提取方法只需要几百次查询，并且我们的预处理器感知攻击恢复了与单独攻击模型时相同的效率。代码可在https://github.com/google-research/preprocessor-aware-black-box-attack.上找到



## **8. Frequency Domain Adversarial Training for Robust Volumetric Medical Segmentation**

用于健壮体积医学分割的频域对抗性训练 eess.IV

This paper has been accepted in MICCAI 2023 conference

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2307.07269v2) [paper-pdf](http://arxiv.org/pdf/2307.07269v2)

**Authors**: Asif Hanif, Muzammal Naseer, Salman Khan, Mubarak Shah, Fahad Shahbaz Khan

**Abstract**: It is imperative to ensure the robustness of deep learning models in critical applications such as, healthcare. While recent advances in deep learning have improved the performance of volumetric medical image segmentation models, these models cannot be deployed for real-world applications immediately due to their vulnerability to adversarial attacks. We present a 3D frequency domain adversarial attack for volumetric medical image segmentation models and demonstrate its advantages over conventional input or voxel domain attacks. Using our proposed attack, we introduce a novel frequency domain adversarial training approach for optimizing a robust model against voxel and frequency domain attacks. Moreover, we propose frequency consistency loss to regulate our frequency domain adversarial training that achieves a better tradeoff between model's performance on clean and adversarial samples. Code is publicly available at https://github.com/asif-hanif/vafa.

摘要: 确保深度学习模型在医疗保健等关键应用中的健壮性是当务之急。虽然深度学习的最新进展提高了体积医学图像分割模型的性能，但由于它们容易受到对手攻击，这些模型不能立即应用于现实世界。提出了一种对体医学图像分割模型进行3D频域攻击的方法，并证明了其相对于传统的输入域或体素域攻击的优势。利用我们提出的攻击，我们引入了一种新的频域对抗训练方法来优化抗体素和频域攻击的稳健模型。此外，我们还提出了频率一致性损失来调整我们的频域对抗训练，从而在干净样本和对抗样本上的模型性能之间实现了更好的折衷。代码可在https://github.com/asif-hanif/vafa.上公开获得



## **9. PATROL: Privacy-Oriented Pruning for Collaborative Inference Against Model Inversion Attacks**

PATROL：针对模型反转攻击的协同推理面向隐私剪枝 cs.LG

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2307.10981v1) [paper-pdf](http://arxiv.org/pdf/2307.10981v1)

**Authors**: Shiwei Ding, Lan Zhang, Miao Pan, Xiaoyong Yuan

**Abstract**: Collaborative inference has been a promising solution to enable resource-constrained edge devices to perform inference using state-of-the-art deep neural networks (DNNs). In collaborative inference, the edge device first feeds the input to a partial DNN locally and then uploads the intermediate result to the cloud to complete the inference. However, recent research indicates model inversion attacks (MIAs) can reconstruct input data from intermediate results, posing serious privacy concerns for collaborative inference. Existing perturbation and cryptography techniques are inefficient and unreliable in defending against MIAs while performing accurate inference. This paper provides a viable solution, named PATROL, which develops privacy-oriented pruning to balance privacy, efficiency, and utility of collaborative inference. PATROL takes advantage of the fact that later layers in a DNN can extract more task-specific features. Given limited local resources for collaborative inference, PATROL intends to deploy more layers at the edge based on pruning techniques to enforce task-specific features for inference and reduce task-irrelevant but sensitive features for privacy preservation. To achieve privacy-oriented pruning, PATROL introduces two key components: Lipschitz regularization and adversarial reconstruction training, which increase the reconstruction errors by reducing the stability of MIAs and enhance the target inference model by adversarial training, respectively.

摘要: 协作推理是一种很有前途的解决方案，它使资源受限的边缘设备能够使用最先进的深度神经网络(DNN)进行推理。在协同推理中，边缘设备首先将输入反馈到本地的部分DNN，然后将中间结果上传到云中完成推理。然而，最近的研究表明，模型反转攻击(MIA)可以从中间结果重建输入数据，这给协作推理带来了严重的隐私问题。现有的微扰和密码技术在防御MIA的同时执行准确的推理是低效和不可靠的。本文提出了一个可行的解决方案，称为PATR，它发展了面向隐私的剪枝，以平衡协作推理的私密性、效率和效用。PATROL利用了DNN中较晚的层可以提取更多特定于任务的特征这一事实。考虑到用于协作推理的本地资源有限，PATR打算基于剪枝技术在边缘部署更多层，以强制执行特定于任务的特征进行推理，并减少与任务无关但敏感的特征以保护隐私。为了实现面向隐私的剪枝，PATR引入了两个关键部分：Lipschitz正则化和对抗性重建训练，它们分别通过降低MIA的稳定性来增加重建误差，并通过对抗性训练来增强目标推理模型。



## **10. Risk-optimized Outlier Removal for Robust Point Cloud Classification**

用于稳健点云分类的风险优化的离群点去除 cs.CV

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2307.10875v1) [paper-pdf](http://arxiv.org/pdf/2307.10875v1)

**Authors**: Xinke Li, Junchi Lu

**Abstract**: The popularity of point cloud deep models for safety-critical purposes has increased, but the reliability and security of these models can be compromised by intentional or naturally occurring point cloud noise. To combat this issue, we present a novel point cloud outlier removal method called PointCVaR, which empowers standard-trained models to eliminate additional outliers and restore the data. Our approach begins by conducting attribution analysis to determine the influence of each point on the model output, which we refer to as point risk. We then optimize the process of filtering high-risk points using Conditional Value at Risk (CVaR) as the objective. The rationale for this approach is based on the observation that noise points in point clouds tend to cluster in the tail of the risk distribution, with a low frequency but a high level of risk, resulting in significant interference with classification results. Despite requiring no additional training effort, our method produces exceptional results in various removal-and-classification experiments for noisy point clouds, which are corrupted by random noise, adversarial noise, and backdoor trigger noise. Impressively, it achieves 87% accuracy in defense against the backdoor attack by removing triggers. Overall, the proposed PointCVaR effectively eliminates noise points and enhances point cloud classification, making it a promising plug-in module for various models in different scenarios.

摘要: 用于安全关键目的的点云深度模型越来越流行，但这些模型的可靠性和安全性可能会受到有意或自然发生的点云噪声的影响。为了解决这一问题，我们提出了一种新的点云离群点去除方法，称为PointCVaR，它允许标准训练的模型消除额外的离群点并恢复数据。我们的方法首先进行归因分析，以确定每个点对模型输出的影响，我们称之为点风险。然后，我们以条件风险价值(CVaR)为目标，优化了筛选高风险点的流程。这种方法的基本原理是基于这样的观察，即点云中的噪声点往往聚集在风险分布的尾部，频率低但风险水平高，导致对分类结果的显著干扰。尽管不需要额外的训练，但我们的方法在各种噪声点云的去除和分类实验中取得了优异的效果，这些噪声点云受到随机噪声、对抗性噪声和后门触发噪声的破坏。令人印象深刻的是，它通过删除触发器实现了87%的后门攻击防御准确率。总体而言，提出的PointCVaR有效地消除了噪声点，增强了点云分类，使其成为一种很有前途的插件模块，适用于不同场景下的各种模型。



## **11. Adversarial attacks for mixtures of classifiers**

针对混合分类器的对抗性攻击 cs.LG

7 pages + 4 pages of appendix. 5 figures in main text

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2307.10788v1) [paper-pdf](http://arxiv.org/pdf/2307.10788v1)

**Authors**: Lucas Gnecco Heredia, Benjamin Negrevergne, Yann Chevaleyre

**Abstract**: Mixtures of classifiers (a.k.a. randomized ensembles) have been proposed as a way to improve robustness against adversarial attacks. However, it has been shown that existing attacks are not well suited for this kind of classifiers. In this paper, we discuss the problem of attacking a mixture in a principled way and introduce two desirable properties of attacks based on a geometrical analysis of the problem (effectiveness and maximality). We then show that existing attacks do not meet both of these properties. Finally, we introduce a new attack called lattice climber attack with theoretical guarantees on the binary linear setting, and we demonstrate its performance by conducting experiments on synthetic and real datasets.

摘要: 分类器的混合(也称为随机化集成)已被提出为提高对对手攻击的稳健性的一种方式。然而，已有的攻击已被证明不能很好地适用于这类分类器。在本文中，我们讨论了以原则性方式攻击混合物的问题，并在对问题进行几何分析的基础上引入了攻击的两个理想性质(有效性和极大性)。然后，我们证明了现有的攻击并不同时满足这两个性质。最后，我们介绍了一种在二元线性设置下具有理论保证的新攻击--格型攀爬攻击，并通过在合成数据集和真实数据集上的实验验证了其性能。



## **12. Friendly Noise against Adversarial Noise: A Powerful Defense against Data Poisoning Attacks**

友善噪声对抗敌意噪声：数据中毒攻击的有力防御 cs.CR

Code available at: https://github.com/tianyu139/friendly-noise

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2208.10224v4) [paper-pdf](http://arxiv.org/pdf/2208.10224v4)

**Authors**: Tian Yu Liu, Yu Yang, Baharan Mirzasoleiman

**Abstract**: A powerful category of (invisible) data poisoning attacks modify a subset of training examples by small adversarial perturbations to change the prediction of certain test-time data. Existing defense mechanisms are not desirable to deploy in practice, as they often either drastically harm the generalization performance, or are attack-specific, and prohibitively slow to apply. Here, we propose a simple but highly effective approach that unlike existing methods breaks various types of invisible poisoning attacks with the slightest drop in the generalization performance. We make the key observation that attacks introduce local sharp regions of high training loss, which when minimized, results in learning the adversarial perturbations and makes the attack successful. To break poisoning attacks, our key idea is to alleviate the sharp loss regions introduced by poisons. To do so, our approach comprises two components: an optimized friendly noise that is generated to maximally perturb examples without degrading the performance, and a randomly varying noise component. The combination of both components builds a very light-weight but extremely effective defense against the most powerful triggerless targeted and hidden-trigger backdoor poisoning attacks, including Gradient Matching, Bulls-eye Polytope, and Sleeper Agent. We show that our friendly noise is transferable to other architectures, and adaptive attacks cannot break our defense due to its random noise component. Our code is available at: https://github.com/tianyu139/friendly-noise

摘要: 一种强大的(不可见)数据中毒攻击通过小的对抗性扰动来修改训练样本的子集，以改变对某些测试时间数据的预测。现有的防御机制在实践中并不可取，因为它们通常要么严重损害泛化性能，要么是特定于攻击的，应用速度慢得令人望而却步。在这里，我们提出了一种简单而高效的方法，不同于现有的方法，它以最小的泛化性能下降来破解各种类型的隐形中毒攻击。我们的关键观察是，攻击引入了高训练损失的局部尖锐区域，当训练损失最小化时，导致学习对手的扰动，使攻击成功。要打破毒物攻击，我们的关键思想是减轻毒物引入的急剧损失区域。为此，我们的方法包括两个组件：一个优化的友好噪声，它被生成以在不降低性能的情况下最大限度地扰动示例，以及一个随机变化的噪声组件。这两个组件的组合构建了一个非常轻但极其有效的防御系统，以抵御最强大的无触发器定向和隐藏触发器后门中毒攻击，包括梯度匹配、公牛眼多面体和睡眠代理。我们证明了我们的友好噪声是可以转移到其他体系结构的，而自适应攻击由于其随机噪声成分而不能破坏我们的防御。我们的代码请访问：https://github.com/tianyu139/friendly-noise



## **13. A Holistic Assessment of the Reliability of Machine Learning Systems**

机器学习系统可靠性的整体评估 cs.LG

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2307.10586v1) [paper-pdf](http://arxiv.org/pdf/2307.10586v1)

**Authors**: Anthony Corso, David Karamadian, Romeo Valentin, Mary Cooper, Mykel J. Kochenderfer

**Abstract**: As machine learning (ML) systems increasingly permeate high-stakes settings such as healthcare, transportation, military, and national security, concerns regarding their reliability have emerged. Despite notable progress, the performance of these systems can significantly diminish due to adversarial attacks or environmental changes, leading to overconfident predictions, failures to detect input faults, and an inability to generalize in unexpected scenarios. This paper proposes a holistic assessment methodology for the reliability of ML systems. Our framework evaluates five key properties: in-distribution accuracy, distribution-shift robustness, adversarial robustness, calibration, and out-of-distribution detection. A reliability score is also introduced and used to assess the overall system reliability. To provide insights into the performance of different algorithmic approaches, we identify and categorize state-of-the-art techniques, then evaluate a selection on real-world tasks using our proposed reliability metrics and reliability score. Our analysis of over 500 models reveals that designing for one metric does not necessarily constrain others but certain algorithmic techniques can improve reliability across multiple metrics simultaneously. This study contributes to a more comprehensive understanding of ML reliability and provides a roadmap for future research and development.

摘要: 随着机器学习(ML)系统越来越多地渗透到医疗、交通、军事和国家安全等高风险环境中，人们对其可靠性的担忧已经出现。尽管取得了显著的进展，但由于敌意攻击或环境变化，这些系统的性能可能会显著下降，导致预测过于自信，无法检测输入故障，并且无法在意外情况下进行泛化。本文提出了一种ML系统可靠性的整体评估方法。我们的框架评估了五个关键属性：分布内准确性、分布移位稳健性、对抗性健壮性、校准和分布外检测。还引入了可靠性分数，并用来评估系统的整体可靠性。为了深入了解不同算法方法的性能，我们对最先进的技术进行识别和分类，然后使用我们提出的可靠性度量和可靠性分数来评估对现实世界任务的选择。我们对500多个模型的分析表明，为一个指标设计并不一定约束其他指标，但某些算法技术可以同时提高多个指标的可靠性。这项研究有助于更全面地了解ML的可靠性，并为未来的研究和开发提供了路线图。



## **14. FACADE: A Framework for Adversarial Circuit Anomaly Detection and Evaluation**

Facade：一种对抗性巡回法庭异常检测与评估框架 cs.LG

Accepted as BlueSky Poster at 2023 ICML AdvML Workshop

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2307.10563v1) [paper-pdf](http://arxiv.org/pdf/2307.10563v1)

**Authors**: Dhruv Pai, Andres Carranza, Rylan Schaeffer, Arnuv Tandon, Sanmi Koyejo

**Abstract**: We present FACADE, a novel probabilistic and geometric framework designed for unsupervised mechanistic anomaly detection in deep neural networks. Its primary goal is advancing the understanding and mitigation of adversarial attacks. FACADE aims to generate probabilistic distributions over circuits, which provide critical insights to their contribution to changes in the manifold properties of pseudo-classes, or high-dimensional modes in activation space, yielding a powerful tool for uncovering and combating adversarial attacks. Our approach seeks to improve model robustness, enhance scalable model oversight, and demonstrates promising applications in real-world deployment settings.

摘要: 提出了一种新的概率几何框架FAADE，用于深度神经网络中的无监督机械异常检测。它的主要目标是促进对对抗性攻击的理解和缓解。Facade的目标是在电路上生成概率分布，这些分布为伪类或激活空间中高维模式的流形属性的变化提供了关键的见解，为发现和对抗对手攻击提供了一个强大的工具。我们的方法旨在提高模型的健壮性，增强可伸缩的模型监督，并展示了在真实世界部署环境中的良好应用。



## **15. Shared Adversarial Unlearning: Backdoor Mitigation by Unlearning Shared Adversarial Examples**

共享对抗性遗忘：通过忘却共享对抗性实例的后门缓解 cs.LG

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2307.10562v1) [paper-pdf](http://arxiv.org/pdf/2307.10562v1)

**Authors**: Shaokui Wei, Mingda Zhang, Hongyuan Zha, Baoyuan Wu

**Abstract**: Backdoor attacks are serious security threats to machine learning models where an adversary can inject poisoned samples into the training set, causing a backdoored model which predicts poisoned samples with particular triggers to particular target classes, while behaving normally on benign samples. In this paper, we explore the task of purifying a backdoored model using a small clean dataset. By establishing the connection between backdoor risk and adversarial risk, we derive a novel upper bound for backdoor risk, which mainly captures the risk on the shared adversarial examples (SAEs) between the backdoored model and the purified model. This upper bound further suggests a novel bi-level optimization problem for mitigating backdoor using adversarial training techniques. To solve it, we propose Shared Adversarial Unlearning (SAU). Specifically, SAU first generates SAEs, and then, unlearns the generated SAEs such that they are either correctly classified by the purified model and/or differently classified by the two models, such that the backdoor effect in the backdoored model will be mitigated in the purified model. Experiments on various benchmark datasets and network architectures show that our proposed method achieves state-of-the-art performance for backdoor defense.

摘要: 后门攻击是对机器学习模型的严重安全威胁，在机器学习模型中，攻击者可以将有毒样本注入训练集，导致后门模型预测有毒样本具有特定触发到特定目标类，而在良性样本上正常操作。在这篇文章中，我们探索了使用一个小的干净的数据集来净化一个回溯模型的任务。通过建立后门风险和对手风险之间的联系，我们得到了一个新的后门风险上界，它主要捕捉后门模型和提纯模型之间共享的对抗性实例(SAE)上的风险。这个上界进一步提出了一个新的双层优化问题，用于使用对抗性训练技术来缓解后门问题。为了解决这一问题，我们提出了共享对抗性遗忘学习(SAU)。具体地说，SAU首先生成SAE，然后取消学习所生成的SAE，使得它们被净化模型正确地分类和/或被两个模型不同地分类，从而在净化模型中将减轻后门效应。在不同的基准数据集和网络架构上的实验表明，我们提出的方法在后门防御方面取得了最好的性能。



## **16. It Is All About Data: A Survey on the Effects of Data on Adversarial Robustness**

这一切都与数据有关：数据对对手健壮性影响的调查 cs.LG

51 pages, 25 figures, under review

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2303.09767v2) [paper-pdf](http://arxiv.org/pdf/2303.09767v2)

**Authors**: Peiyu Xiong, Michael Tegegn, Jaskeerat Singh Sarin, Shubhraneel Pal, Julia Rubin

**Abstract**: Adversarial examples are inputs to machine learning models that an attacker has intentionally designed to confuse the model into making a mistake. Such examples pose a serious threat to the applicability of machine-learning-based systems, especially in life- and safety-critical domains. To address this problem, the area of adversarial robustness investigates mechanisms behind adversarial attacks and defenses against these attacks. This survey reviews a particular subset of this literature that focuses on investigating properties of training data in the context of model robustness under evasion attacks. It first summarizes the main properties of data leading to adversarial vulnerability. It then discusses guidelines and techniques for improving adversarial robustness by enhancing the data representation and learning procedures, as well as techniques for estimating robustness guarantees given particular data. Finally, it discusses gaps of knowledge and promising future research directions in this area.

摘要: 对抗性的例子是机器学习模型的输入，攻击者故意设计这些模型来混淆模型，使其出错。这些例子对基于机器学习的系统的适用性构成了严重威胁，特别是在生命和安全关键领域。为了解决这个问题，对抗性稳健性领域调查了对抗性攻击背后的机制和对这些攻击的防御。这项调查回顾了这篇文献的一个特定子集，重点是在逃避攻击下的模型稳健性背景下调查训练数据的属性。它首先总结了导致对抗性漏洞的数据的主要属性。然后讨论了通过加强数据表示和学习过程来提高对手稳健性的指导方针和技术，以及在给定特定数据的情况下估计稳健性保证的技术。最后，讨论了该领域的知识差距和未来的研究方向。



## **17. MultiRobustBench: Benchmarking Robustness Against Multiple Attacks**

MultiRobustBch：针对多个攻击的健壮性基准 cs.LG

ICML 2023

**SubmitDate**: 2023-07-20    [abs](http://arxiv.org/abs/2302.10980v3) [paper-pdf](http://arxiv.org/pdf/2302.10980v3)

**Authors**: Sihui Dai, Saeed Mahloujifar, Chong Xiang, Vikash Sehwag, Pin-Yu Chen, Prateek Mittal

**Abstract**: The bulk of existing research in defending against adversarial examples focuses on defending against a single (typically bounded Lp-norm) attack, but for a practical setting, machine learning (ML) models should be robust to a wide variety of attacks. In this paper, we present the first unified framework for considering multiple attacks against ML models. Our framework is able to model different levels of learner's knowledge about the test-time adversary, allowing us to model robustness against unforeseen attacks and robustness against unions of attacks. Using our framework, we present the first leaderboard, MultiRobustBench, for benchmarking multiattack evaluation which captures performance across attack types and attack strengths. We evaluate the performance of 16 defended models for robustness against a set of 9 different attack types, including Lp-based threat models, spatial transformations, and color changes, at 20 different attack strengths (180 attacks total). Additionally, we analyze the state of current defenses against multiple attacks. Our analysis shows that while existing defenses have made progress in terms of average robustness across the set of attacks used, robustness against the worst-case attack is still a big open problem as all existing models perform worse than random guessing.

摘要: 现有的大量研究集中于防御单一(通常有界的Lp范数)攻击，但对于实际环境，机器学习(ML)模型应该对各种攻击具有健壮性。在这篇文章中，我们提出了第一个考虑针对ML模型的多重攻击的统一框架。我们的框架能够对学习者关于测试时间对手的不同级别的知识进行建模，使我们能够建模对意外攻击的健壮性和对攻击组合的健壮性。使用我们的框架，我们提出了第一个排行榜，MultiRobustBch，用于对多攻击进行基准评估，该评估捕获了攻击类型和攻击强度的性能。我们评估了16种防御模型在20种不同攻击强度(总共180次攻击)下对9种不同攻击类型的健壮性，包括基于LP的威胁模型、空间变换和颜色变化。此外，我们还分析了当前对多种攻击的防御状态。我们的分析表明，尽管现有防御在使用的一组攻击的平均健壮性方面取得了进展，但对最坏情况攻击的健壮性仍然是一个巨大的开放问题，因为所有现有模型的表现都不如随机猜测。



## **18. Invariant Aggregator for Defending against Federated Backdoor Attacks**

用于防御联合后门攻击的不变聚合器 cs.LG

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2210.01834v2) [paper-pdf](http://arxiv.org/pdf/2210.01834v2)

**Authors**: Xiaoyang Wang, Dimitrios Dimitriadis, Sanmi Koyejo, Shruti Tople

**Abstract**: Federated learning is gaining popularity as it enables training high-utility models across several clients without directly sharing their private data. As a downside, the federated setting makes the model vulnerable to various adversarial attacks in the presence of malicious clients. Despite the theoretical and empirical success in defending against attacks that aim to degrade models' utility, defense against backdoor attacks that increase model accuracy on backdoor samples exclusively without hurting the utility on other samples remains challenging. To this end, we first analyze the vulnerability of federated learning to backdoor attacks over a flat loss landscape which is common for well-designed neural networks such as Resnet [He et al., 2015] but is often overlooked by previous works. Over a flat loss landscape, misleading federated learning models to exclusively benefit malicious clients with backdoor samples do not require a significant difference between malicious and benign client-wise updates, making existing defenses insufficient. In contrast, we propose an invariant aggregator that redirects the aggregated update to invariant directions that are generally useful via selectively masking out the gradient elements that favor few and possibly malicious clients regardless of the difference magnitude. Theoretical results suggest that our approach provably mitigates backdoor attacks over both flat and sharp loss landscapes. Empirical results on three datasets with different modalities and varying numbers of clients further demonstrate that our approach mitigates a broad class of backdoor attacks with a negligible cost on the model utility.

摘要: 联合学习越来越受欢迎，因为它能够在几个客户之间培训高实用模型，而无需直接共享他们的私人数据。缺点是，联合设置使模型在存在恶意客户端的情况下容易受到各种敌意攻击。尽管在防御旨在降低模型效用的攻击方面取得了理论和经验上的成功，但针对后门攻击的防御仍然具有挑战性，这种攻击只能提高后门样本的模型精度，而不会损害其他样本的效用。为此，我们首先分析了联合学习在平坦损失场景下对后门攻击的脆弱性，这在RESNET等设计良好的神经网络中很常见[他等人，2015]，但经常被以前的工作忽视。在平坦的损失环境中，误导性的联合学习模型通过后门样本专门使恶意客户端受益，不需要在恶意客户端更新和良性客户端更新之间存在显著差异，从而使现有防御措施不足。相反，我们提出了一个不变聚集器，它通过选择性地屏蔽有利于少数客户端和可能是恶意客户端的梯度元素，将聚合的更新重定向到通常有用的不变方向，而不考虑差异的大小。理论结果表明，我们的方法可以有效地减少对平坦和尖锐损失场景的后门攻击。在三个具有不同模式和不同客户端数量的数据集上的实验结果进一步表明，我们的方法以可以忽略不计的模型效用代价缓解了广泛类别的后门攻击。



## **19. Rethinking Backdoor Attacks**

重新思考后门攻击 cs.CR

ICML 2023

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2307.10163v1) [paper-pdf](http://arxiv.org/pdf/2307.10163v1)

**Authors**: Alaa Khaddaj, Guillaume Leclerc, Aleksandar Makelov, Kristian Georgiev, Hadi Salman, Andrew Ilyas, Aleksander Madry

**Abstract**: In a backdoor attack, an adversary inserts maliciously constructed backdoor examples into a training set to make the resulting model vulnerable to manipulation. Defending against such attacks typically involves viewing these inserted examples as outliers in the training set and using techniques from robust statistics to detect and remove them.   In this work, we present a different approach to the backdoor attack problem. Specifically, we show that without structural information about the training data distribution, backdoor attacks are indistinguishable from naturally-occurring features in the data--and thus impossible to "detect" in a general sense. Then, guided by this observation, we revisit existing defenses against backdoor attacks and characterize the (often latent) assumptions they make and on which they depend. Finally, we explore an alternative perspective on backdoor attacks: one that assumes these attacks correspond to the strongest feature in the training data. Under this assumption (which we make formal) we develop a new primitive for detecting backdoor attacks. Our primitive naturally gives rise to a detection algorithm that comes with theoretical guarantees and is effective in practice.

摘要: 在后门攻击中，对手将恶意构建的后门示例插入训练集，从而使生成的模型容易受到操纵。对此类攻击的防御通常涉及将这些插入的示例视为训练集中的异常值，并使用稳健统计中的技术来检测和删除它们。在这项工作中，我们提出了一种不同的方法来解决后门攻击问题。具体地说，我们表明，如果没有关于训练数据分布的结构信息，后门攻击与数据中自然出现的特征是无法区分的--因此不可能在一般意义上“检测”。然后，在这一观察的指导下，我们重新审视针对后门攻击的现有防御措施，并描述它们所做的和它们所依赖的(通常是潜在的)假设。最后，我们探索了关于后门攻击的另一种观点：假设这些攻击对应于训练数据中最强的特征。在这个假设下(我们将其形式化)，我们开发了一个用于检测后门攻击的新原语。我们的原语自然地产生了一种具有理论保证并在实践中有效的检测算法。



## **20. I See Dead People: Gray-Box Adversarial Attack on Image-To-Text Models**

我看到死人：对图像到文本模型的灰箱对抗性攻击 cs.CV

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2306.07591v3) [paper-pdf](http://arxiv.org/pdf/2306.07591v3)

**Authors**: Raz Lapid, Moshe Sipper

**Abstract**: Modern image-to-text systems typically adopt the encoder-decoder framework, which comprises two main components: an image encoder, responsible for extracting image features, and a transformer-based decoder, used for generating captions. Taking inspiration from the analysis of neural networks' robustness against adversarial perturbations, we propose a novel gray-box algorithm for creating adversarial examples in image-to-text models. Unlike image classification tasks that have a finite set of class labels, finding visually similar adversarial examples in an image-to-text task poses greater challenges because the captioning system allows for a virtually infinite space of possible captions. In this paper, we present a gray-box adversarial attack on image-to-text, both untargeted and targeted. We formulate the process of discovering adversarial perturbations as an optimization problem that uses only the image-encoder component, meaning the proposed attack is language-model agnostic. Through experiments conducted on the ViT-GPT2 model, which is the most-used image-to-text model in Hugging Face, and the Flickr30k dataset, we demonstrate that our proposed attack successfully generates visually similar adversarial examples, both with untargeted and targeted captions. Notably, our attack operates in a gray-box manner, requiring no knowledge about the decoder module. We also show that our attacks fool the popular open-source platform Hugging Face.

摘要: 现代图像到文本系统通常采用编解码器框架，该框架包括两个主要组件：负责提取图像特征的图像编码器和用于生成字幕的基于转换器的解码器。从神经网络对对抗性扰动的鲁棒性分析中得到启发，我们提出了一种新的灰盒算法，用于在图像到文本模型中创建对抗性示例。与具有有限类别标签集的图像分类任务不同，在图像到文本的任务中找到视觉上相似的对抗性例子带来了更大的挑战，因为字幕系统允许可能的字幕的几乎无限空间。在本文中，我们提出了一种针对图像到文本的灰盒对抗性攻击，包括无目标攻击和目标攻击。我们将发现敌意扰动的过程描述为一个只使用图像编码器组件的优化问题，这意味着所提出的攻击是语言模型不可知的。通过在拥抱脸中最常用的图文转换模型VIT-GPT2模型和Flickr30k数据集上的实验，我们证明了我们的攻击成功地生成了视觉上相似的对抗性例子，无论是无目标字幕还是有目标字幕。值得注意的是，我们的攻击以灰盒方式运行，不需要了解解码器模块。我们还表明，我们的攻击愚弄了流行的开源平台拥抱脸。



## **21. Why Does Little Robustness Help? Understanding Adversarial Transferability From Surrogate Training**

为什么小健壮性会有帮助？从替补训练看对手的转换性 cs.LG

Accepted by IEEE Symposium on Security and Privacy (Oakland) 2024; 21  pages, 12 figures, 13 tables

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2307.07873v2) [paper-pdf](http://arxiv.org/pdf/2307.07873v2)

**Authors**: Yechao Zhang, Shengshan Hu, Leo Yu Zhang, Junyu Shi, Minghui Li, Xiaogeng Liu, Wei Wan, Hai Jin

**Abstract**: Adversarial examples (AEs) for DNNs have been shown to be transferable: AEs that successfully fool white-box surrogate models can also deceive other black-box models with different architectures. Although a bunch of empirical studies have provided guidance on generating highly transferable AEs, many of these findings lack explanations and even lead to inconsistent advice. In this paper, we take a further step towards understanding adversarial transferability, with a particular focus on surrogate aspects. Starting from the intriguing little robustness phenomenon, where models adversarially trained with mildly perturbed adversarial samples can serve as better surrogates, we attribute it to a trade-off between two predominant factors: model smoothness and gradient similarity. Our investigations focus on their joint effects, rather than their separate correlations with transferability. Through a series of theoretical and empirical analyses, we conjecture that the data distribution shift in adversarial training explains the degradation of gradient similarity. Building on these insights, we explore the impacts of data augmentation and gradient regularization on transferability and identify that the trade-off generally exists in the various training mechanisms, thus building a comprehensive blueprint for the regulation mechanism behind transferability. Finally, we provide a general route for constructing better surrogates to boost transferability which optimizes both model smoothness and gradient similarity simultaneously, e.g., the combination of input gradient regularization and sharpness-aware minimization (SAM), validated by extensive experiments. In summary, we call for attention to the united impacts of these two factors for launching effective transfer attacks, rather than optimizing one while ignoring the other, and emphasize the crucial role of manipulating surrogate models.

摘要: DNN的对抗性例子(AE)已被证明是可移植的：成功欺骗白盒代理模型的AES也可以欺骗其他具有不同体系结构的黑盒模型。尽管大量的实证研究为生成高度可转移的企业实体提供了指导，但其中许多发现缺乏解释，甚至导致了不一致的建议。在这篇文章中，我们进一步了解对手的可转移性，特别关注代理方面。从有趣的小鲁棒性现象开始，我们将其归因于两个主要因素之间的权衡：模型的光滑性和梯度相似性。我们的研究重点是它们的联合影响，而不是它们与可转让性的单独关联。通过一系列的理论和实证分析，我们推测对抗性训练中的数据分布转移解释了梯度相似度的下降。基于这些见解，我们探讨了数据扩充和梯度规则化对可转移性的影响，并确定了各种培训机制中普遍存在的权衡，从而构建了可转移性背后的监管机制的全面蓝图。最后，我们提供了一条同时优化模型光滑性和梯度相似性的构造更好的代理以提高可转移性的一般路线，例如输入梯度正则化和锐度感知最小化(SAM)的组合，并通过大量的实验进行了验证。总之，我们呼吁注意这两个因素对发动有效转移攻击的联合影响，而不是优化一个而忽略另一个，并强调操纵代理模型的关键作用。



## **22. Fix your downsampling ASAP! Be natively more robust via Aliasing and Spectral Artifact free Pooling**

尽快解决您的下采样问题！通过混叠和无频谱伪影池实现更强大的性能 cs.CV

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2307.09804v1) [paper-pdf](http://arxiv.org/pdf/2307.09804v1)

**Authors**: Julia Grabinski, Janis Keuper, Margret Keuper

**Abstract**: Convolutional neural networks encode images through a sequence of convolutions, normalizations and non-linearities as well as downsampling operations into potentially strong semantic embeddings. Yet, previous work showed that even slight mistakes during sampling, leading to aliasing, can be directly attributed to the networks' lack in robustness. To address such issues and facilitate simpler and faster adversarial training, [12] recently proposed FLC pooling, a method for provably alias-free downsampling - in theory. In this work, we conduct a further analysis through the lens of signal processing and find that such current pooling methods, which address aliasing in the frequency domain, are still prone to spectral leakage artifacts. Hence, we propose aliasing and spectral artifact-free pooling, short ASAP. While only introducing a few modifications to FLC pooling, networks using ASAP as downsampling method exhibit higher native robustness against common corruptions, a property that FLC pooling was missing. ASAP also increases native robustness against adversarial attacks on high and low resolution data while maintaining similar clean accuracy or even outperforming the baseline.

摘要: 卷积神经网络通过一系列卷积、归一化和非线性以及下采样操作将图像编码为潜在的强语义嵌入。然而，以前的工作表明，即使是采样过程中的微小错误，导致混叠，也可以直接归因于网络缺乏健壮性。为了解决这些问题并促进更简单、更快的对抗训练，[12]最近提出了FLC Pooling，一种理论上可证明的无混叠下采样方法。在这项工作中，我们通过信号处理的镜头进行了进一步的分析，发现目前这种在频域解决混叠的池化方法仍然容易出现频谱泄漏伪影。因此，我们提出了无混叠和无频谱伪影的池化方法，简称为越快越好。虽然只对FLC池进行了一些修改，但使用ASAP作为下采样方法的网络对常见损坏表现出更高的本机健壮性，这是FLC池所缺少的特性。ASAP还提高了针对高分辨率和低分辨率数据的敌意攻击的本地健壮性，同时保持了类似的干净准确性，甚至超过了基准。



## **23. Improving the Transferability of Adversarial Attacks on Face Recognition with Beneficial Perturbation Feature Augmentation**

利用有益扰动特征增强提高人脸识别中敌意攻击的可转移性 cs.CV

\c{opyright} 2023 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2210.16117v4) [paper-pdf](http://arxiv.org/pdf/2210.16117v4)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Zongyi Li, Ping Li

**Abstract**: Face recognition (FR) models can be easily fooled by adversarial examples, which are crafted by adding imperceptible perturbations on benign face images. The existence of adversarial face examples poses a great threat to the security of society. In order to build a more sustainable digital nation, in this paper, we improve the transferability of adversarial face examples to expose more blind spots of existing FR models. Though generating hard samples has shown its effectiveness in improving the generalization of models in training tasks, the effectiveness of utilizing this idea to improve the transferability of adversarial face examples remains unexplored. To this end, based on the property of hard samples and the symmetry between training tasks and adversarial attack tasks, we propose the concept of hard models, which have similar effects as hard samples for adversarial attack tasks. Utilizing the concept of hard models, we propose a novel attack method called Beneficial Perturbation Feature Augmentation Attack (BPFA), which reduces the overfitting of adversarial examples to surrogate FR models by constantly generating new hard models to craft the adversarial examples. Specifically, in the backpropagation, BPFA records the gradients on pre-selected feature maps and uses the gradient on the input image to craft the adversarial example. In the next forward propagation, BPFA leverages the recorded gradients to add beneficial perturbations on their corresponding feature maps to increase the loss. Extensive experiments demonstrate that BPFA can significantly boost the transferability of adversarial attacks on FR.

摘要: 人脸识别(FR)模型很容易被敌意的例子所愚弄，这些例子是通过在良性的人脸图像上添加难以察觉的扰动来构建的。敌对面孔的存在对社会安全构成了极大的威胁。为了建设一个更可持续的数字国家，本文通过提高对抗性人脸样本的可转移性来暴露现有FR模型的更多盲点。尽管硬样本的生成在训练任务中提高了模型的泛化能力，但利用硬样本来提高对抗性人脸样本的可转移性的有效性仍未被探讨。为此，基于硬样本的性质和训练任务与对抗性攻击任务之间的对称性，我们提出了硬模型的概念，对于对抗性攻击任务，硬模型具有类似硬样本的效果。利用硬模型的概念，我们提出了一种新的攻击方法，称为有益扰动特征增强攻击(BPFA)，该方法通过不断生成新的硬模型来构造对抗性实例，从而减少了对抗性实例对替代FR模型的过度拟合。具体地说，在反向传播中，BPFA记录预先选择的特征地图上的梯度，并使用输入图像上的梯度来制作对抗性例子。在下一次前向传播中，BPFA利用记录的梯度在其相应的特征图上添加有益的扰动以增加损失。大量实验表明，BPFA能够显著提高对抗性攻击对FR的可转移性。



## **24. Making Substitute Models More Bayesian Can Enhance Transferability of Adversarial Examples**

使替代模型更具贝叶斯性质可以增强对抗性例子的可转移性 cs.LG

Accepted by ICLR 2023, fix typos

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2302.05086v3) [paper-pdf](http://arxiv.org/pdf/2302.05086v3)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: The transferability of adversarial examples across deep neural networks (DNNs) is the crux of many black-box attacks. Many prior efforts have been devoted to improving the transferability via increasing the diversity in inputs of some substitute models. In this paper, by contrast, we opt for the diversity in substitute models and advocate to attack a Bayesian model for achieving desirable transferability. Deriving from the Bayesian formulation, we develop a principled strategy for possible finetuning, which can be combined with many off-the-shelf Gaussian posterior approximations over DNN parameters. Extensive experiments have been conducted to verify the effectiveness of our method, on common benchmark datasets, and the results demonstrate that our method outperforms recent state-of-the-arts by large margins (roughly 19% absolute increase in average attack success rate on ImageNet), and, by combining with these recent methods, further performance gain can be obtained. Our code: https://github.com/qizhangli/MoreBayesian-attack.

摘要: 恶意例子在深度神经网络(DNN)之间的可转移性是许多黑盒攻击的症结所在。先前的许多努力都致力于通过增加一些替代模型的投入的多样性来提高可转移性。相反，在本文中，我们选择了替代模型的多样性，并主张攻击贝叶斯模型，以实现理想的可转移性。从贝叶斯公式出发，我们开发了一种可能的精调的原则性策略，该策略可以与许多关于DNN参数的现成的高斯后验近似相结合。在常见的基准数据集上进行了广泛的实验来验证我们的方法的有效性，结果表明我们的方法的性能比最近的最新技术有很大的差距(在ImageNet上的平均攻击成功率大约绝对提高了19%)，并且通过结合这些最新的方法，可以获得进一步的性能提升。我们的代码：https://github.com/qizhangli/MoreBayesian-attack.



## **25. Reinforcing POD based model reduction techniques in reaction-diffusion complex networks using stochastic filtering and pattern recognition**

基于随机滤波和模式识别的反应扩散复杂网络模型降阶技术 cs.CE

19 pages, 6 figures

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2307.09762v1) [paper-pdf](http://arxiv.org/pdf/2307.09762v1)

**Authors**: Abhishek Ajayakumar, Soumyendu Raha

**Abstract**: Complex networks are used to model many real-world systems. However, the dimensionality of these systems can make them challenging to analyze. Dimensionality reduction techniques like POD can be used in such cases. However, these models are susceptible to perturbations in the input data. We propose an algorithmic framework that combines techniques from pattern recognition (PR) and stochastic filtering theory to enhance the output of such models. The results of our study show that our method can improve the accuracy of the surrogate model under perturbed inputs. Deep Neural Networks (DNNs) are susceptible to adversarial attacks. However, recent research has revealed that neural Ordinary Differential Equations (ODEs) exhibit robustness in specific applications. We benchmark our algorithmic framework with a Neural ODE-based approach as a reference.

摘要: 复杂网络被用来对许多真实世界的系统进行建模。然而，这些系统的维度可能会使它们难以分析。在这种情况下可以使用像POD这样的降维技术。然而，这些模型容易受到输入数据中的扰动的影响。我们提出了一种结合模式识别(PR)和随机滤波理论的算法框架，以增强此类模型的输出。研究结果表明，在扰动输入下，我们的方法可以提高代理模型的精度。深度神经网络(DNN)容易受到敌意攻击。然而，最近的研究表明，神经常微分方程(ODE)在特定的应用中表现出了健壮性。我们以基于神经节点的方法作为参考，对我们的算法框架进行基准测试。



## **26. Unified Adversarial Patch for Cross-modal Attacks in the Physical World**

物理世界中跨模式攻击的统一对抗性补丁 cs.CV

10 pages, 8 figures, accepted by ICCV2023

**SubmitDate**: 2023-07-19    [abs](http://arxiv.org/abs/2307.07859v2) [paper-pdf](http://arxiv.org/pdf/2307.07859v2)

**Authors**: Xingxing Wei, Yao Huang, Yitong Sun, Jie Yu

**Abstract**: Recently, physical adversarial attacks have been presented to evade DNNs-based object detectors. To ensure the security, many scenarios are simultaneously deployed with visible sensors and infrared sensors, leading to the failures of these single-modal physical attacks. To show the potential risks under such scenes, we propose a unified adversarial patch to perform cross-modal physical attacks, i.e., fooling visible and infrared object detectors at the same time via a single patch. Considering different imaging mechanisms of visible and infrared sensors, our work focuses on modeling the shapes of adversarial patches, which can be captured in different modalities when they change. To this end, we design a novel boundary-limited shape optimization to achieve the compact and smooth shapes, and thus they can be easily implemented in the physical world. In addition, to balance the fooling degree between visible detector and infrared detector during the optimization process, we propose a score-aware iterative evaluation, which can guide the adversarial patch to iteratively reduce the predicted scores of the multi-modal sensors. We finally test our method against the one-stage detector: YOLOv3 and the two-stage detector: Faster RCNN. Results show that our unified patch achieves an Attack Success Rate (ASR) of 73.33% and 69.17%, respectively. More importantly, we verify the effective attacks in the physical world when visible and infrared sensors shoot the objects under various settings like different angles, distances, postures, and scenes.

摘要: 最近，为了躲避基于DNN的对象检测器，已经出现了物理对抗性攻击。为了确保安全，多个场景同时部署了可见光传感器和红外传感器，导致这些单模物理攻击失败。为了显示这种场景下的潜在风险，我们提出了一个统一的对抗性补丁来执行跨模式的物理攻击，即通过一个补丁同时欺骗可见光和红外目标探测器。考虑到可见光和红外传感器不同的成像机理，我们的工作重点是对敌方斑块的形状进行建模，当它们发生变化时，可以以不同的方式捕获这些斑块。为此，我们设计了一种新颖的边界受限形状优化算法，实现了形状的紧凑和光滑，从而可以很容易地在物理世界中实现。此外，为了在优化过程中平衡可见光探测器和红外探测器之间的愚弄程度，我们提出了一种分数感知迭代评估方法，可以引导敌方补丁迭代降低多模式传感器的预测分数。最后，我们用单级检测器YOLOv3和两级检测器FASTER RCNN测试了我们的方法。结果表明，统一补丁的攻击成功率分别为73.33%和69.17%。更重要的是，我们验证了当可见光和红外传感器在不同的角度、距离、姿势和场景等各种设置下拍摄对象时，在物理世界中的有效攻击。



## **27. VISER: A Tractable Solution Concept for Games with Information Asymmetry**

Viser：一种易处理的信息不对称对策解概念 cs.GT

17 pages, 6 figures

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2307.09652v1) [paper-pdf](http://arxiv.org/pdf/2307.09652v1)

**Authors**: Jeremy McMahan, Young Wu, Yudong Chen, Xiaojin Zhu, Qiaomin Xie

**Abstract**: Many real-world games suffer from information asymmetry: one player is only aware of their own payoffs while the other player has the full game information. Examples include the critical domain of security games and adversarial multi-agent reinforcement learning. Information asymmetry renders traditional solution concepts such as Strong Stackelberg Equilibrium (SSE) and Robust-Optimization Equilibrium (ROE) inoperative. We propose a novel solution concept called VISER (Victim Is Secure, Exploiter best-Responds). VISER enables an external observer to predict the outcome of such games. In particular, for security applications, VISER allows the victim to better defend itself while characterizing the most damaging attacks available to the attacker. We show that each player's VISER strategy can be computed independently in polynomial time using linear programming (LP). We also extend VISER to its Markov-perfect counterpart for Markov games, which can be solved efficiently using a series of LPs.

摘要: 许多现实世界的游戏都存在信息不对称的问题：一个玩家只知道自己的收益，而另一个玩家拥有完整的游戏信息。例子包括安全游戏的关键领域和对抗性多智能体强化学习。信息不对称使得传统的解概念，如强Stackelberg均衡(SSE)和稳健优化均衡(ROE)不再适用。我们提出了一种新的解决方案概念，称为VISER(受害者是安全的，剥削者是最佳响应)。Viser使外部观察者能够预测这类游戏的结果。特别是，对于安全应用程序，VISER允许受害者更好地自卫，同时确定攻击者可以使用的最具破坏性的攻击的特征。我们证明了利用线性规划(LP)可以在多项式时间内独立地计算出每个参与者的Viser策略。我们还将VISER推广到马尔可夫对策的马尔可夫完美对应，这可以用一系列LP有效地求解。



## **28. Dead Man's PLC: Towards Viable Cyber Extortion for Operational Technology**

死人的PLC：为作战技术走向可行的网络勒索 cs.CR

13 pages, 19 figures

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2307.09549v1) [paper-pdf](http://arxiv.org/pdf/2307.09549v1)

**Authors**: Richard Derbyshire, Benjamin Green, Charl van der Walt, David Hutchison

**Abstract**: For decades, operational technology (OT) has enjoyed the luxury of being suitably inaccessible so as to experience directly targeted cyber attacks from only the most advanced and well-resourced adversaries. However, security via obscurity cannot last forever, and indeed a shift is happening whereby less advanced adversaries are showing an appetite for targeting OT. With this shift in adversary demographics, there will likely also be a shift in attack goals, from clandestine process degradation and espionage to overt cyber extortion (Cy-X). The consensus from OT cyber security practitioners suggests that, even if encryption-based Cy-X techniques were launched against OT assets, typical recovery practices designed for engineering processes would provide adequate resilience. In response, this paper introduces Dead Man's PLC (DM-PLC), a pragmatic step towards viable OT Cy-X that acknowledges and weaponises the resilience processes typically encountered. Using only existing functionality, DM-PLC considers an entire environment as the entity under ransom, whereby all assets constantly poll one another to ensure the attack remains untampered, treating any deviations as a detonation trigger akin to a Dead Man's switch. A proof of concept of DM-PLC is implemented and evaluated on an academically peer reviewed and industry validated OT testbed to demonstrate its malicious efficacy.

摘要: 几十年来，作战技术(OT)一直享受着适当地不可访问的奢侈，以便只体验来自最先进和资源充足的对手的直接定向网络攻击。然而，默默无闻的安全不可能永远持续下去，事实上，一种转变正在发生，不那么先进的对手表现出了以OT为目标的胃口。随着对手人口结构的这种转变，攻击目标可能也会发生变化，从秘密过程降级和间谍活动转向公开的网络勒索(Cy-X)。OT网络安全从业人员的共识表明，即使针对OT资产推出了基于加密的Cy-X技术，为工程流程设计的典型恢复做法也将提供足够的弹性。作为回应，本文介绍了死人PLC(DM-PLC)，这是朝着可行的OT Cy-X迈出的务实一步，它承认并武器化了通常遇到的弹性过程。DM-PLC仅使用现有功能，将整个环境视为赎金下的实体，所有资产不断轮询彼此，以确保攻击保持不被篡改，将任何偏差视为类似于死人开关的引爆触发器。DM-PLC的概念验证在学术同行评审和行业验证的OT测试床上实现和评估，以证明其恶意效果。



## **29. Untargeted Near-collision Attacks in Biometric Recognition**

生物特征识别中的无目标近碰撞攻击 cs.CR

Addition of results and correction of typos

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2304.01580v3) [paper-pdf](http://arxiv.org/pdf/2304.01580v3)

**Authors**: Axel Durbet, Paul-Marie Grollemund, Kevin Thiry-Atighehchi

**Abstract**: A biometric recognition system can operate in two distinct modes, identification or verification. In the first mode, the system recognizes an individual by searching the enrolled templates of all the users for a match. In the second mode, the system validates a user's identity claim by comparing the fresh provided template with the enrolled template. The biometric transformation schemes usually produce binary templates that are better handled by cryptographic schemes, and the comparison is based on a distance that leaks information about the similarities between two biometric templates. Both the experimentally determined false match rate and false non-match rate through recognition threshold adjustment define the recognition accuracy, and hence the security of the system. To the best of our knowledge, few works provide a formal treatment of the security under minimum leakage of information, i.e., the binary outcome of a comparison with a threshold. In this paper, we rely on probabilistic modelling to quantify the security strength of binary templates. We investigate the influence of template size, database size and threshold on the probability of having a near-collision. We highlight several untargeted attacks on biometric systems considering naive and adaptive adversaries. Interestingly, these attacks can be launched both online and offline and, both in the identification mode and in the verification mode. We discuss the choice of parameters through the generic presented attacks.

摘要: 生物识别系统可以在两种截然不同的模式下工作，即识别或验证。在第一种模式中，系统通过在所有用户的注册模板中搜索匹配项来识别个人。在第二种模式中，系统通过将新提供的模板与注册的模板进行比较来验证用户的身份声明。生物特征转换方案通常产生由加密方案更好地处理的二进制模板，并且比较基于泄露关于两个生物特征模板之间的相似性的信息的距离。实验确定的误匹配率和通过调整识别阈值确定的误不匹配率都定义了识别精度，从而决定了系统的安全性。就我们所知，很少有文献在信息泄露最小的情况下提供安全的形式处理，即与阈值比较的二进制结果。在本文中，我们依赖于概率建模来量化二进制模板的安全强度。我们研究了模板大小、数据库大小和阈值对近碰撞概率的影响。我们重点介绍了几种针对生物识别系统的非定向攻击，考虑到了天真和自适应的对手。有趣的是，这些攻击既可以在线上也可以离线发起，也可以在识别模式和验证模式下发起。我们通过一般提出的攻击讨论参数的选择。



## **30. Mitigating Intersection Attacks in Anonymous Microblogging**

减轻匿名微博中的交叉攻击 cs.CR

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2307.09069v1) [paper-pdf](http://arxiv.org/pdf/2307.09069v1)

**Authors**: Sarah Abdelwahab Gaballah, Thanh Hoang Long Nguyen, Lamya Abdullah, Ephraim Zimmer, Max Mühlhäuser

**Abstract**: Anonymous microblogging systems are known to be vulnerable to intersection attacks due to network churn. An adversary that monitors all communications can leverage the churn to learn who is publishing what with increasing confidence over time. In this paper, we propose a protocol for mitigating intersection attacks in anonymous microblogging systems by grouping users into anonymity sets based on similarities in their publishing behavior. The protocol provides a configurable communication schedule for users in each set to manage the inevitable trade-off between latency and bandwidth overhead. In our evaluation, we use real-world datasets from two popular microblogging platforms, Twitter and Reddit, to simulate user publishing behavior. The results demonstrate that the protocol can protect users against intersection attacks at low bandwidth overhead when the users adhere to communication schedules. In addition, the protocol can sustain a slow degradation in the size of the anonymity set over time under various churn rates.

摘要: 众所周知，匿名微博系统由于网络波动而容易受到交叉攻击。监视所有通信的对手可以利用流失来了解谁在发布什么，并且随着时间的推移越来越有信心。本文提出了一种基于用户发布行为相似性将用户分组为匿名集合的协议，用于缓解匿名微博系统中的交叉攻击。该协议为每个集合中的用户提供可配置的通信调度，以管理延迟和带宽开销之间不可避免的权衡。在我们的评估中，我们使用了来自两个流行的微博平台Twitter和Reddit的真实数据集来模拟用户发布行为。结果表明，该协议在用户遵守通信调度的情况下，能够以较低的带宽开销保护用户免受交集攻击。此外，该协议可以在不同的流失率下，随着时间的推移，保持匿名性大小的缓慢下降。



## **31. FedDefender: Client-Side Attack-Tolerant Federated Learning**

FedDefender：客户端容忍攻击的联合学习 cs.CR

KDD'23 research track accepted

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2307.09048v1) [paper-pdf](http://arxiv.org/pdf/2307.09048v1)

**Authors**: Sungwon Park, Sungwon Han, Fangzhao Wu, Sundong Kim, Bin Zhu, Xing Xie, Meeyoung Cha

**Abstract**: Federated learning enables learning from decentralized data sources without compromising privacy, which makes it a crucial technique. However, it is vulnerable to model poisoning attacks, where malicious clients interfere with the training process. Previous defense mechanisms have focused on the server-side by using careful model aggregation, but this may not be effective when the data is not identically distributed or when attackers can access the information of benign clients. In this paper, we propose a new defense mechanism that focuses on the client-side, called FedDefender, to help benign clients train robust local models and avoid the adverse impact of malicious model updates from attackers, even when a server-side defense cannot identify or remove adversaries. Our method consists of two main components: (1) attack-tolerant local meta update and (2) attack-tolerant global knowledge distillation. These components are used to find noise-resilient model parameters while accurately extracting knowledge from a potentially corrupted global model. Our client-side defense strategy has a flexible structure and can work in conjunction with any existing server-side strategies. Evaluations of real-world scenarios across multiple datasets show that the proposed method enhances the robustness of federated learning against model poisoning attacks.

摘要: 联合学习能够在不损害隐私的情况下从分散的数据源进行学习，这使其成为一项关键技术。然而，它很容易受到模型中毒攻击，在这种攻击中，恶意客户端会干扰训练过程。以前的防御机制主要集中在服务器端，使用仔细的模型聚合，但当数据不同分布或攻击者可以访问良性客户端的信息时，这可能不会有效。在本文中，我们提出了一种新的专注于客户端的防御机制FedDefender，以帮助良性客户端训练健壮的本地模型，并避免攻击者恶意更新模型的不利影响，即使在服务器端防御无法识别或删除对手的情况下也是如此。该方法由两个主要部分组成：(1)容忍攻击的局部元更新和(2)容忍攻击的全局知识提取。这些组件用于找到抗噪模型参数，同时准确地从可能损坏的全局模型中提取知识。我们的客户端防御策略具有灵活的结构，可以与任何现有的服务器端策略协同工作。对多个数据集的真实场景的评估表明，该方法增强了联合学习对模型中毒攻击的健壮性。



## **32. Discretization-based ensemble model for robust learning in IoT**

物联网中基于离散化的稳健学习集成模型 cs.LG

15 pages

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2307.08955v1) [paper-pdf](http://arxiv.org/pdf/2307.08955v1)

**Authors**: Anahita Namvar, Chandra Thapa, Salil S. Kanhere

**Abstract**: IoT device identification is the process of recognizing and verifying connected IoT devices to the network. This is an essential process for ensuring that only authorized devices can access the network, and it is necessary for network management and maintenance. In recent years, machine learning models have been used widely for automating the process of identifying devices in the network. However, these models are vulnerable to adversarial attacks that can compromise their accuracy and effectiveness. To better secure device identification models, discretization techniques enable reduction in the sensitivity of machine learning models to adversarial attacks contributing to the stability and reliability of the model. On the other hand, Ensemble methods combine multiple heterogeneous models to reduce the impact of remaining noise or errors in the model. Therefore, in this paper, we integrate discretization techniques and ensemble methods and examine it on model robustness against adversarial attacks. In other words, we propose a discretization-based ensemble stacking technique to improve the security of our ML models. We evaluate the performance of different ML-based IoT device identification models against white box and black box attacks using a real-world dataset comprised of network traffic from 28 IoT devices. We demonstrate that the proposed method enables robustness to the models for IoT device identification.

摘要: 物联网设备识别是识别和验证连接到网络的物联网设备的过程。这是确保只有授权设备才能访问网络的基本过程，也是网络管理和维护所必需的。近年来，机器学习模型被广泛用于自动识别网络中的设备的过程。然而，这些模型容易受到敌意攻击，从而影响其准确性和有效性。为了更好地保护设备识别模型，离散化技术能够降低机器学习模型对敌对攻击的敏感度，有助于提高模型的稳定性和可靠性。另一方面，集成方法将多个异质模型组合在一起，以减少模型中剩余噪声或误差的影响。因此，在本文中，我们将离散化技术和集成方法相结合，并检验其对模型抵抗对手攻击的稳健性。换句话说，我们提出了一种基于离散化的集成堆叠技术来提高ML模型的安全性。我们使用由28台物联网设备的网络流量组成的真实数据集，评估了不同的基于ML的物联网设备识别模型对抗白盒和黑盒攻击的性能。我们证明了所提出的方法能够对物联网设备识别的模型具有稳健性。



## **33. On the Robustness of Split Learning against Adversarial Attacks**

关于分裂学习对敌意攻击的稳健性 cs.LG

accepted by ECAI 2023, camera-ready version

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2307.07916v2) [paper-pdf](http://arxiv.org/pdf/2307.07916v2)

**Authors**: Mingyuan Fan, Cen Chen, Chengyu Wang, Wenmeng Zhou, Jun Huang

**Abstract**: Split learning enables collaborative deep learning model training while preserving data privacy and model security by avoiding direct sharing of raw data and model details (i.e., sever and clients only hold partial sub-networks and exchange intermediate computations). However, existing research has mainly focused on examining its reliability for privacy protection, with little investigation into model security. Specifically, by exploring full models, attackers can launch adversarial attacks, and split learning can mitigate this severe threat by only disclosing part of models to untrusted servers.This paper aims to evaluate the robustness of split learning against adversarial attacks, particularly in the most challenging setting where untrusted servers only have access to the intermediate layers of the model.Existing adversarial attacks mostly focus on the centralized setting instead of the collaborative setting, thus, to better evaluate the robustness of split learning, we develop a tailored attack called SPADV, which comprises two stages: 1) shadow model training that addresses the issue of lacking part of the model and 2) local adversarial attack that produces adversarial examples to evaluate.The first stage only requires a few unlabeled non-IID data, and, in the second stage, SPADV perturbs the intermediate output of natural samples to craft the adversarial ones. The overall cost of the proposed attack process is relatively low, yet the empirical attack effectiveness is significantly high, demonstrating the surprising vulnerability of split learning to adversarial attacks.

摘要: 分裂学习通过避免直接共享原始数据和模型细节(即服务器和客户端只持有部分子网络并交换中间计算)，实现了协作式深度学习模型训练，同时保护了数据隐私和模型安全性。然而，现有的研究主要集中在考察其在隐私保护方面的可靠性，对模型安全性的研究很少。通过探索完整的模型，攻击者可以发起敌意攻击，而分裂学习可以通过只将部分模型泄露给不可信的服务器来缓解这一严重威胁。本文旨在评估分裂学习对敌意攻击的稳健性，特别是在最具挑战性的环境中，不可信的服务器只能访问模型的中间层。现有的对抗性攻击大多集中在集中环境而不是协作环境，因此，为了更好地评估分裂学习的稳健性，我们开发了一种称为SPADV的定制攻击。它包括两个阶段：1)影子模型训练，解决模型缺乏部分的问题；2)局部对抗性攻击，产生对抗性样本进行评估；第一阶段，只需要少量未标记的非IID数据；第二阶段，SPADV扰动自然样本的中间输出来制作对抗性样本。所提出的攻击过程的总体成本相对较低，但经验攻击效率显著高，表明分裂学习在对抗性攻击中具有令人惊讶的脆弱性。



## **34. Hiding Visual Information via Obfuscating Adversarial Perturbations**

通过混淆敌意扰动隐藏视觉信息 cs.CV

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2209.15304v3) [paper-pdf](http://arxiv.org/pdf/2209.15304v3)

**Authors**: Zhigang Su, Dawei Zhou, Decheng Liu, Nannan Wang, Zhen Wang, Xinbo Gao

**Abstract**: Growing leakage and misuse of visual information raise security and privacy concerns, which promotes the development of information protection. Existing adversarial perturbations-based methods mainly focus on the de-identification against deep learning models. However, the inherent visual information of the data has not been well protected. In this work, inspired by the Type-I adversarial attack, we propose an adversarial visual information hiding method to protect the visual privacy of data. Specifically, the method generates obfuscating adversarial perturbations to obscure the visual information of the data. Meanwhile, it maintains the hidden objectives to be correctly predicted by models. In addition, our method does not modify the parameters of the applied model, which makes it flexible for different scenarios. Experimental results on the recognition and classification tasks demonstrate that the proposed method can effectively hide visual information and hardly affect the performances of models. The code is available in the supplementary material.

摘要: 日益增长的视觉信息泄露和滥用引发了人们对安全和隐私的担忧，这推动了信息保护的发展。现有的基于对抗性扰动的方法主要集中在针对深度学习模型的去识别。然而，数据固有的视觉信息并没有得到很好的保护。在这项工作中，受Type-I对抗攻击的启发，我们提出了一种对抗性视觉信息隐藏方法来保护数据的视觉隐私。具体地说，该方法产生模糊的对抗性扰动以模糊数据的可视信息。同时，保持模型对隐含目标的正确预测。此外，我们的方法不修改应用模型的参数，这使得它可以灵活地适应不同的场景。在识别和分类任务上的实验结果表明，该方法能够有效地隐藏视觉信息，且几乎不影响模型的性能。该代码可在补充材料中找到。



## **35. Unstoppable Attack: Label-Only Model Inversion via Conditional Diffusion Model**

不可阻挡的攻击：基于条件扩散模型的仅标签模型反演 cs.AI

11 pages, 6 figures, 2 tables

**SubmitDate**: 2023-07-18    [abs](http://arxiv.org/abs/2307.08424v2) [paper-pdf](http://arxiv.org/pdf/2307.08424v2)

**Authors**: Rongke Liu

**Abstract**: Model inversion attacks (MIAs) are aimed at recovering private data from a target model's training set, which poses a threat to the privacy of deep learning models. MIAs primarily focus on the white-box scenario where the attacker has full access to the structure and parameters of the target model. However, practical applications are black-box, it is not easy for adversaries to obtain model-related parameters, and various models only output predicted labels. Existing black-box MIAs primarily focused on designing the optimization strategy, and the generative model is only migrated from the GAN used in white-box MIA. Our research is the pioneering study of feasible attack models in label-only black-box scenarios, to the best of our knowledge.   In this paper, we develop a novel method of MIA using the conditional diffusion model to recover the precise sample of the target without any extra optimization, as long as the target model outputs the label. Two primary techniques are introduced to execute the attack. Firstly, select an auxiliary dataset that is relevant to the target model task, and the labels predicted by the target model are used as conditions to guide the training process. Secondly, target labels and random standard normally distributed noise are input into the trained conditional diffusion model, generating target samples with pre-defined guidance strength. We then filter out the most robust and representative samples. Furthermore, we propose for the first time to use Learned Perceptual Image Patch Similarity (LPIPS) as one of the evaluation metrics for MIA, with systematic quantitative and qualitative evaluation in terms of attack accuracy, realism, and similarity. Experimental results show that this method can generate similar and accurate data to the target without optimization and outperforms generators of previous approaches in the label-only scenario.

摘要: 模型反转攻击(MIA)旨在从目标模型的训练集中恢复隐私数据，这对深度学习模型的隐私构成了威胁。MIA主要关注白盒情况，其中攻击者具有对目标模型的结构和参数的完全访问权限。然而，实际应用都是黑箱问题，攻击者不容易获得模型相关参数，而且各种模型只输出预测标签。已有的黑盒MIA主要侧重于优化策略的设计，生成模型仅从白盒MIA中使用的GaN移植而来。据我们所知，我们的研究是对只有标签的黑盒场景中可行攻击模型的开创性研究。在本文中，我们提出了一种新的MIA方法，该方法使用条件扩散模型来恢复目标的精确样本，而不需要任何额外的优化，只要目标模型输出标签即可。两种主要技术被引入来执行该攻击。首先选择与目标模型任务相关的辅助数据集，以目标模型预测的标签作为条件指导训练过程。其次，将目标标签和随机标准正态分布噪声输入到训练好的条件扩散模型中，生成具有预定制导强度的目标样本。然后我们过滤出最健壮和最具代表性的样本。此外，我们还首次提出将学习感知图像块相似度(LPIPS)作为MIA的评价指标之一，从攻击的准确性、真实性和相似性三个方面进行了系统的定量和定性评价。实验结果表明，该方法可以在不经过优化的情况下生成与目标相似且准确的数据，并且在仅有标签的情况下性能优于以往方法的生成器。



## **36. Exploring the Unprecedented Privacy Risks of the Metaverse**

探索Metverse前所未有的隐私风险 cs.CR

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2207.13176v3) [paper-pdf](http://arxiv.org/pdf/2207.13176v3)

**Authors**: Vivek Nair, Gonzalo Munilla Garrido, Dawn Song

**Abstract**: Thirty study participants playtested an innocent-looking "escape room" game in virtual reality (VR). Behind the scenes, an adversarial program had accurately inferred over 25 personal data attributes, from anthropometrics like height and wingspan to demographics like age and gender, within just a few minutes of gameplay. As notoriously data-hungry companies become increasingly involved in VR development, this experimental scenario may soon represent a typical VR user experience. While virtual telepresence applications (and the so-called "metaverse") have recently received increased attention and investment from major tech firms, these environments remain relatively under-studied from a security and privacy standpoint. In this work, we illustrate how VR attackers can covertly ascertain dozens of personal data attributes from seemingly-anonymous users of popular metaverse applications like VRChat. These attackers can be as simple as other VR users without special privilege, and the potential scale and scope of this data collection far exceed what is feasible within traditional mobile and web applications. We aim to shed light on the unique privacy risks of the metaverse, and provide the first holistic framework for understanding intrusive data harvesting attacks in these emerging VR ecosystems.

摘要: 30名研究参与者在虚拟现实(VR)中玩了一个看起来很无辜的“逃生室”游戏。在幕后，一个对抗性的程序在玩游戏的短短几分钟内就准确地推断出了25个人的数据属性，从身高和翼展等人体测量数据到年龄和性别等人口统计数据。随着以渴望数据著称的公司越来越多地参与到VR开发中来，这种实验场景可能很快就会代表一种典型的VR用户体验。虽然虚拟远程呈现应用(以及所谓的虚拟现实)最近得到了主要科技公司越来越多的关注和投资，但从安全和隐私的角度来看，这些环境的研究仍然相对较少。在这项工作中，我们展示了VR攻击者如何从VRChat等流行虚拟世界应用程序的看似匿名的用户那里秘密确定数十个个人数据属性。这些攻击者可以像其他没有特殊权限的VR用户一样简单，而且这种数据收集的潜在规模和范围远远超出了传统移动和网络应用程序中的可行范围。我们的目标是阐明虚拟世界独特的隐私风险，并提供第一个整体框架，以了解这些新兴的虚拟现实生态系统中的侵入性数据收集攻击。



## **37. TorMult: Introducing a Novel Tor Bandwidth Inflation Attack**

TorMult：引入一种新的ToR带宽膨胀攻击 cs.CR

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2307.08550v1) [paper-pdf](http://arxiv.org/pdf/2307.08550v1)

**Authors**: Christoph Sendner, Jasper Stang, Alexandra Dmitrienko, Raveen Wijewickrama, Murtuza Jadliwala

**Abstract**: The Tor network is the most prominent system for providing anonymous communication to web users, with a daily user base of 2 million users. However, since its inception, it has been constantly targeted by various traffic fingerprinting and correlation attacks aiming at deanonymizing its users. A critical requirement for these attacks is to attract as much user traffic to adversarial relays as possible, which is typically accomplished by means of bandwidth inflation attacks. This paper proposes a new inflation attack vector in Tor, referred to as TorMult, which enables inflation of measured bandwidth. The underlying attack technique exploits resource sharing among Tor relay nodes and employs a cluster of attacker-controlled relays with coordinated resource allocation within the cluster to deceive bandwidth measurers into believing that each relay node in the cluster possesses ample resources. We propose two attack variants, C-TorMult and D-TorMult, and test both versions in a private Tor test network. Our evaluation demonstrates that an attacker can inflate the measured bandwidth by a factor close to n using C-TorMult and nearly half n*N using D-TorMult, where n is the size of the cluster hosted on one server and N is the number of servers. Furthermore, our theoretical analysis reveals that gaining control over half of the Tor network's traffic can be achieved by employing just 10 dedicated servers with a cluster size of 109 relays running the TorMult attack, each with a bandwidth of 100MB/s. The problem is further exacerbated by the fact that Tor not only allows resource sharing but, according to recent reports, even promotes it.

摘要: Tor网络是向网络用户提供匿名通信的最突出的系统，每天有200万用户。然而，自其成立以来，它一直是各种旨在解除用户匿名的流量指纹和关联攻击的目标。这些攻击的一个关键要求是将尽可能多的用户流量吸引到对抗性中继，这通常是通过带宽膨胀攻击实现的。本文提出了一种新的ToR膨胀攻击向量，称为TorMult，它支持测量带宽的膨胀。底层攻击技术利用ToR中继节点之间的资源共享，使用一群攻击者控制的中继群，并在群内协调资源分配，以欺骗带宽测量者相信群中的每个中继节点拥有充足的资源。我们提出了两种攻击变体C-TorMult和D-TorMult，并在专用ToR测试网络上对这两种版本进行了测试。我们的评估表明，攻击者可以使用C-TorMult将测量的带宽夸大接近n的系数，使用D-TorMult将测量的带宽夸大近一半n*N，其中n是一台服务器上托管的集群的大小，N是服务器的数量。此外，我们的理论分析表明，只需使用10台专用服务器，集群大小为109个运行TorMult攻击的中继站，每个中继站的带宽为100MB/S，就可以控制Tor网络一半的流量。根据最近的报道，Tor不仅允许资源共享，甚至促进资源共享，这进一步加剧了问题。



## **38. FocusedCleaner: Sanitizing Poisoned Graphs for Robust GNN-based Node Classification**

FocusedCleaner：用于基于GNN的健壮节点分类的毒图清理 cs.LG

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2210.13815v2) [paper-pdf](http://arxiv.org/pdf/2210.13815v2)

**Authors**: Yulin Zhu, Liang Tong, Gaolei Li, Xiapu Luo, Kai Zhou

**Abstract**: Graph Neural Networks (GNNs) are vulnerable to data poisoning attacks, which will generate a poisoned graph as the input to the GNN models. We present FocusedCleaner as a poisoned graph sanitizer to effectively identify the poison injected by attackers. Specifically, FocusedCleaner provides a sanitation framework consisting of two modules: bi-level structural learning and victim node detection. In particular, the structural learning module will reverse the attack process to steadily sanitize the graph while the detection module provides ``the focus" -- a narrowed and more accurate search region -- to structural learning. These two modules will operate in iterations and reinforce each other to sanitize a poisoned graph step by step. As an important application, we show that the adversarial robustness of GNNs trained over the sanitized graph for the node classification task is significantly improved. Extensive experiments demonstrate that FocusedCleaner outperforms the state-of-the-art baselines both on poisoned graph sanitation and improving robustness.

摘要: 图神经网络(GNN)容易受到数据中毒攻击，这种攻击会产生一个有毒的图作为GNN模型的输入。我们将FocusedCleaner作为一个有毒的图形消毒器来有效地识别攻击者注入的毒药。具体地说，FocusedCleaner提供了一个由两个模块组成的卫生框架：双层结构学习和受害者节点检测。特别是，结构学习模块将逆转攻击过程，以稳定地对图进行杀毒，而检测模块将为结构学习提供`焦点‘--一个更窄、更准确的搜索区域--。这两个模块将迭代运行，相互加强，逐步对中毒图进行杀毒。作为一个重要的应用，我们证明了针对节点分类任务，在杀毒图上训练的GNN的对抗健壮性得到了显著提高。大量实验表明，FocusedCleaner在毒图卫生和提高健壮性方面都优于最先进的基线。



## **39. Tracking Fringe and Coordinated Activity on Twitter Leading Up To the US Capitol Attack**

追踪美国国会大厦袭击前推特上的边缘和协调活动 cs.SI

11 pages (including references), 8 figures, 1 table. Accepted at The  18th International AAAI Conference on Web and Social Media

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2302.04450v2) [paper-pdf](http://arxiv.org/pdf/2302.04450v2)

**Authors**: Vishnuprasad Padinjaredath Suresh, Gianluca Nogara, Felipe Cardoso, Stefano Cresci, Silvia Giordano, Luca Luceri

**Abstract**: The aftermath of the 2020 US Presidential Election witnessed an unprecedented attack on the democratic values of the country through the violent insurrection at Capitol Hill on January 6th, 2021. The attack was fueled by the proliferation of conspiracy theories and misleading claims about the integrity of the election pushed by political elites and fringe communities on social media. In this study, we explore the evolution of fringe content and conspiracy theories on Twitter in the seven months leading up to the Capitol attack. We examine the suspicious coordinated activity carried out by users sharing fringe content, finding evidence of common adversarial manipulation techniques ranging from targeted amplification to manufactured consensus. Further, we map out the temporal evolution of, and the relationship between, fringe and conspiracy theories, which eventually coalesced into the rhetoric of a stolen election, with the hashtag #stopthesteal, alongside QAnon-related narratives. Our findings further highlight how social media platforms offer fertile ground for the widespread proliferation of conspiracies during major societal events, which can potentially lead to offline coordinated actions and organized violence.

摘要: 在2020年美国总统选举的余波中，2021年1月6日发生在国会山的暴力起义，见证了美国民主价值观受到前所未有的攻击。政治精英和边缘社区在社交媒体上推动的关于选举完整性的误导性言论和阴谋论的扩散，助长了这次袭击。在这项研究中，我们探索了推特上边缘内容和阴谋论在国会大厦袭击前的七个月里的演变。我们检查了共享边缘内容的用户进行的可疑的协调活动，发现了常见的对抗性操纵技术的证据，范围从定向放大到制造共识。此外，我们绘制了边缘理论和阴谋论的时间演变以及它们之间的关系，这些理论最终结合成了一场被盗选举的修辞，标签为#Stop thesteal，以及与QAnon相关的叙述。我们的发现进一步突显了社交媒体平台如何为重大社会活动期间阴谋的广泛扩散提供了肥沃的土壤，这可能会导致线下协调行动和有组织的暴力。



## **40. Adversarial Self-Attack Defense and Spatial-Temporal Relation Mining for Visible-Infrared Video Person Re-Identification**

基于对抗性自攻击防御和时空关系挖掘的可见光-红外视频人员再识别 cs.CV

11 pages,8 figures

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2307.03903v2) [paper-pdf](http://arxiv.org/pdf/2307.03903v2)

**Authors**: Huafeng Li, Le Xu, Yafei Zhang, Dapeng Tao, Zhengtao Yu

**Abstract**: In visible-infrared video person re-identification (re-ID), extracting features not affected by complex scenes (such as modality, camera views, pedestrian pose, background, etc.) changes, and mining and utilizing motion information are the keys to solving cross-modal pedestrian identity matching. To this end, the paper proposes a new visible-infrared video person re-ID method from a novel perspective, i.e., adversarial self-attack defense and spatial-temporal relation mining. In this work, the changes of views, posture, background and modal discrepancy are considered as the main factors that cause the perturbations of person identity features. Such interference information contained in the training samples is used as an adversarial perturbation. It performs adversarial attacks on the re-ID model during the training to make the model more robust to these unfavorable factors. The attack from the adversarial perturbation is introduced by activating the interference information contained in the input samples without generating adversarial samples, and it can be thus called adversarial self-attack. This design allows adversarial attack and defense to be integrated into one framework. This paper further proposes a spatial-temporal information-guided feature representation network to use the information in video sequences. The network cannot only extract the information contained in the video-frame sequences but also use the relation of the local information in space to guide the network to extract more robust features. The proposed method exhibits compelling performance on large-scale cross-modality video datasets. The source code of the proposed method will be released at https://github.com/lhf12278/xxx.

摘要: 在可见光-红外视频人重新识别(Re-ID)中，提取不受复杂场景(如通道、摄像机视角、行人姿势、背景等)影响的特征。变化，以及运动信息的挖掘和利用是解决跨模式行人身份匹配的关键。为此，本文从一个新的角度提出了一种新的可见光-红外视频人身份识别方法，即对抗性自攻击防御和时空关系挖掘。在本工作中，视角、姿态、背景和模式差异的变化被认为是导致身份特征扰动的主要因素。包含在训练样本中的这种干扰信息被用作对抗性扰动。在训练过程中对Re-ID模型进行对抗性攻击，使模型对这些不利因素具有更强的鲁棒性。来自对抗性扰动的攻击是通过激活输入样本中包含的干扰信息而不产生对抗性样本来引入的，因此可以称为对抗性自攻击。这种设计允许将对抗性攻击和防御集成到一个框架中。本文进一步提出了一种时空信息制导的特征表示网络来利用视频序列中的信息。该网络不仅可以提取视频帧序列中包含的信息，而且可以利用局部信息在空间上的关系来指导网络提取更鲁棒的特征。该方法在大规模跨通道视频数据集上表现出了令人信服的性能。建议的方法的源代码将在https://github.com/lhf12278/xxx.上发布



## **41. A Machine Learning based Empirical Evaluation of Cyber Threat Actors High Level Attack Patterns over Low level Attack Patterns in Attributing Attacks**

基于机器学习的网络威胁行为人高级别攻击模式对低级别攻击模式归因的经验评估 cs.CR

20 pages

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2307.10252v1) [paper-pdf](http://arxiv.org/pdf/2307.10252v1)

**Authors**: Umara Noor, Sawera Shahid, Rimsha Kanwal, Zahid Rashid

**Abstract**: Cyber threat attribution is the process of identifying the actor of an attack incident in cyberspace. An accurate and timely threat attribution plays an important role in deterring future attacks by applying appropriate and timely defense mechanisms. Manual analysis of attack patterns gathered by honeypot deployments, intrusion detection systems, firewalls, and via trace-back procedures is still the preferred method of security analysts for cyber threat attribution. Such attack patterns are low-level Indicators of Compromise (IOC). They represent Tactics, Techniques, Procedures (TTP), and software tools used by the adversaries in their campaigns. The adversaries rarely re-use them. They can also be manipulated, resulting in false and unfair attribution. To empirically evaluate and compare the effectiveness of both kinds of IOC, there are two problems that need to be addressed. The first problem is that in recent research works, the ineffectiveness of low-level IOC for cyber threat attribution has been discussed intuitively. An empirical evaluation for the measure of the effectiveness of low-level IOC based on a real-world dataset is missing. The second problem is that the available dataset for high-level IOC has a single instance for each predictive class label that cannot be used directly for training machine learning models. To address these problems in this research work, we empirically evaluate the effectiveness of low-level IOC based on a real-world dataset that is specifically built for comparative analysis with high-level IOC. The experimental results show that the high-level IOC trained models effectively attribute cyberattacks with an accuracy of 95% as compared to the low-level IOC trained models where accuracy is 40%.

摘要: 网络威胁归因是识别网络空间攻击事件行为人的过程。准确和及时的威胁归因通过应用适当和及时的防御机制在威慑未来的攻击中发挥着重要作用。对蜜罐部署、入侵检测系统、防火墙和通过回溯程序收集的攻击模式进行手动分析仍然是安全分析师确定网络威胁归属的首选方法。此类攻击模式是低级危害指示器(IoC)。它们代表对手在其战役中使用的战术、技术、程序(TTP)和软件工具。对手很少再使用它们。它们也可能被操纵，导致错误和不公平的归属。要对两种IOC的有效性进行实证评估和比较，有两个问题需要解决。第一个问题是，在最近的研究工作中，人们直观地讨论了低级别IOC对网络威胁归因的无效。对基于真实世界数据集的低水平IOC有效性的衡量缺乏经验评估。第二个问题是，高层IOC的可用数据集对于每个预测类标签都有一个实例，不能直接用于训练机器学习模型。为了解决这项研究工作中的这些问题，我们基于一个专门为与高级IOC进行比较分析而构建的真实数据集，对低级别IOC的有效性进行了实证评估。实验结果表明，高级别IOC训练模型对网络攻击的有效属性准确率为95%，而低级别IOC训练模型的准确率为40%。



## **42. Analyzing the Impact of Adversarial Examples on Explainable Machine Learning**

对抗性例子对可解释机器学习的影响分析 cs.LG

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2307.08327v1) [paper-pdf](http://arxiv.org/pdf/2307.08327v1)

**Authors**: Prathyusha Devabhakthini, Sasmita Parida, Raj Mani Shukla, Suvendu Chandan Nayak

**Abstract**: Adversarial attacks are a type of attack on machine learning models where an attacker deliberately modifies the inputs to cause the model to make incorrect predictions. Adversarial attacks can have serious consequences, particularly in applications such as autonomous vehicles, medical diagnosis, and security systems. Work on the vulnerability of deep learning models to adversarial attacks has shown that it is very easy to make samples that make a model predict things that it doesn't want to. In this work, we analyze the impact of model interpretability due to adversarial attacks on text classification problems. We develop an ML-based classification model for text data. Then, we introduce the adversarial perturbations on the text data to understand the classification performance after the attack. Subsequently, we analyze and interpret the model's explainability before and after the attack

摘要: 对抗性攻击是对机器学习模型的一种攻击，攻击者故意修改输入，使模型做出错误的预测。对抗性攻击可能会产生严重的后果，特别是在自动驾驶汽车、医疗诊断和安全系统等应用中。关于深度学习模型对敌意攻击脆弱性的研究表明，很容易制作样本，让模型预测它不想预测的事情。在这项工作中，我们分析了对抗性攻击导致的模型可解释性对文本分类问题的影响。提出了一种基于ML的文本数据分类模型。然后，我们引入对文本数据的对抗性扰动，以了解攻击后的分类性能。随后，我们分析和解释了模型在攻击前后的可解释性



## **43. Adversarial Attacks on Traffic Sign Recognition: A Survey**

交通标志识别对抗性攻击研究综述 cs.CV

Accepted for publication at ICECCME2023

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2307.08278v1) [paper-pdf](http://arxiv.org/pdf/2307.08278v1)

**Authors**: Svetlana Pavlitska, Nico Lambing, J. Marius Zöllner

**Abstract**: Traffic sign recognition is an essential component of perception in autonomous vehicles, which is currently performed almost exclusively with deep neural networks (DNNs). However, DNNs are known to be vulnerable to adversarial attacks. Several previous works have demonstrated the feasibility of adversarial attacks on traffic sign recognition models. Traffic signs are particularly promising for adversarial attack research due to the ease of performing real-world attacks using printed signs or stickers. In this work, we survey existing works performing either digital or real-world attacks on traffic sign detection and classification models. We provide an overview of the latest advancements and highlight the existing research areas that require further investigation.

摘要: 交通标志识别是自动驾驶车辆感知的重要组成部分，目前几乎完全使用深度神经网络(DNN)进行识别。然而，众所周知，DNN很容易受到敌意攻击。前人的一些工作已经证明了对抗性攻击交通标志识别模型的可行性。交通标志在对抗性攻击研究中特别有前景，因为它很容易使用打印的标志或贴纸进行现实世界的攻击。在这项工作中，我们调查了现有的对交通标志检测和分类模型进行数字或真实攻击的工作。我们提供了最新进展的概述，并强调了需要进一步研究的现有研究领域。



## **44. Towards Stealthy Backdoor Attacks against Speech Recognition via Elements of Sound**

利用声音元素对语音识别进行隐秘的后门攻击 cs.SD

13 pages

**SubmitDate**: 2023-07-17    [abs](http://arxiv.org/abs/2307.08208v1) [paper-pdf](http://arxiv.org/pdf/2307.08208v1)

**Authors**: Hanbo Cai, Pengcheng Zhang, Hai Dong, Yan Xiao, Stefanos Koffas, Yiming Li

**Abstract**: Deep neural networks (DNNs) have been widely and successfully adopted and deployed in various applications of speech recognition. Recently, a few works revealed that these models are vulnerable to backdoor attacks, where the adversaries can implant malicious prediction behaviors into victim models by poisoning their training process. In this paper, we revisit poison-only backdoor attacks against speech recognition. We reveal that existing methods are not stealthy since their trigger patterns are perceptible to humans or machine detection. This limitation is mostly because their trigger patterns are simple noises or separable and distinctive clips. Motivated by these findings, we propose to exploit elements of sound ($e.g.$, pitch and timbre) to design more stealthy yet effective poison-only backdoor attacks. Specifically, we insert a short-duration high-pitched signal as the trigger and increase the pitch of remaining audio clips to `mask' it for designing stealthy pitch-based triggers. We manipulate timbre features of victim audios to design the stealthy timbre-based attack and design a voiceprint selection module to facilitate the multi-backdoor attack. Our attacks can generate more `natural' poisoned samples and therefore are more stealthy. Extensive experiments are conducted on benchmark datasets, which verify the effectiveness of our attacks under different settings ($e.g.$, all-to-one, all-to-all, clean-label, physical, and multi-backdoor settings) and their stealthiness. The code for reproducing main experiments are available at \url{https://github.com/HanboCai/BadSpeech_SoE}.

摘要: 深度神经网络(DNN)已经被广泛和成功地应用于语音识别的各种应用中。最近，一些工作表明这些模型容易受到后门攻击，攻击者可以通过毒化受害者模型的训练过程将恶意预测行为植入到受害者模型中。在这篇文章中，我们重新审视了针对语音识别的纯有毒后门攻击。我们发现，现有的方法并不是隐蔽的，因为它们的触发模式是人类或机器检测到的。这一限制主要是因为它们的触发模式是简单的噪音或可分离和独特的剪辑。受这些发现的启发，我们建议利用声音元素(例如$、音调和音色)来设计更隐蔽但有效的仅限毒药的后门攻击。具体地说，我们插入一个持续时间短的高音调信号作为触发器，并增加剩余音频片段的音调，以便设计基于音调的隐身触发器。我们利用受害者音频的音色特征来设计基于音色的隐身攻击，并设计了声纹选择模块来支持多后门攻击。我们的攻击可以产生更多“天然”的有毒样本，因此更隐蔽。在基准数据集上进行了广泛的实验，验证了我们的攻击在不同设置(如$、All-to-One、All-to-All、干净标签、物理和多后门设置)下的有效性及其隐蔽性。重现主要实验的代码可在\url{https://github.com/HanboCai/BadSpeech_SoE}.上找到



## **45. Certifying Model Accuracy under Distribution Shifts**

分布漂移下的模型精度验证 cs.LG

**SubmitDate**: 2023-07-16    [abs](http://arxiv.org/abs/2201.12440v3) [paper-pdf](http://arxiv.org/pdf/2201.12440v3)

**Authors**: Aounon Kumar, Alexander Levine, Tom Goldstein, Soheil Feizi

**Abstract**: Certified robustness in machine learning has primarily focused on adversarial perturbations of the input with a fixed attack budget for each point in the data distribution. In this work, we present provable robustness guarantees on the accuracy of a model under bounded Wasserstein shifts of the data distribution. We show that a simple procedure that randomizes the input of the model within a transformation space is provably robust to distributional shifts under the transformation. Our framework allows the datum-specific perturbation size to vary across different points in the input distribution and is general enough to include fixed-sized perturbations as well. Our certificates produce guaranteed lower bounds on the performance of the model for any (natural or adversarial) shift of the input distribution within a Wasserstein ball around the original distribution. We apply our technique to: (i) certify robustness against natural (non-adversarial) transformations of images such as color shifts, hue shifts and changes in brightness and saturation, (ii) certify robustness against adversarial shifts of the input distribution, and (iii) show provable lower bounds (hardness results) on the performance of models trained on so-called "unlearnable" datasets that have been poisoned to interfere with model training.

摘要: 机器学习中已证明的稳健性主要集中在输入的对抗性扰动上，对数据分布中的每个点都有固定的攻击预算。在这项工作中，我们给出了在数据分布的有界Wasserstein位移下模型精度的可证明的稳健性保证。我们证明了在变换空间内随机化模型输入的简单过程对变换下的分布位移是被证明是健壮的。我们的框架允许特定于基准的扰动大小在输入分布的不同点上变化，并且足够普遍以包括固定大小的扰动。我们的证书为输入分布在Wasserstein球中围绕原始分布的任何(自然或对抗性)移动产生了模型性能的保证下限。我们将我们的技术应用于：(I)证明对图像的自然(非对抗性)变换的稳健性，例如颜色漂移、色调漂移以及亮度和饱和度的变化，(Ii)验证对输入分布的对抗性漂移的稳健性，以及(Iii)显示在已被毒害到干扰模型训练的所谓的“不可学习”数据集上训练的模型的性能的可证明的下界(困难结果)。



## **46. Training Socially Aligned Language Models in Simulated Human Society**

在模拟人类社会中训练社会一致的语言模型 cs.CL

Code, data, and models can be downloaded via  https://github.com/agi-templar/Stable-Alignment

**SubmitDate**: 2023-07-16    [abs](http://arxiv.org/abs/2305.16960v2) [paper-pdf](http://arxiv.org/pdf/2305.16960v2)

**Authors**: Ruibo Liu, Ruixin Yang, Chenyan Jia, Ge Zhang, Denny Zhou, Andrew M. Dai, Diyi Yang, Soroush Vosoughi

**Abstract**: Social alignment in AI systems aims to ensure that these models behave according to established societal values. However, unlike humans, who derive consensus on value judgments through social interaction, current language models (LMs) are trained to rigidly replicate their training corpus in isolation, leading to subpar generalization in unfamiliar scenarios and vulnerability to adversarial attacks. This work presents a novel training paradigm that permits LMs to learn from simulated social interactions. In comparison to existing methodologies, our approach is considerably more scalable and efficient, demonstrating superior performance in alignment benchmarks and human evaluations. This paradigm shift in the training of LMs brings us a step closer to developing AI systems that can robustly and accurately reflect societal norms and values.

摘要: 人工智能系统中的社会一致性旨在确保这些模型的行为符合既定的社会价值观。然而，与通过社交互动就价值判断达成共识的人类不同，当前的语言模型(LMS)被训练成孤立地僵硬地复制他们的训练语料库，导致在不熟悉的场景中的泛化能力不佳，并且容易受到对手攻击。这项工作提出了一种新的训练范式，允许LMS从模拟的社会互动中学习。与现有方法相比，我们的方法可伸缩性更强，效率更高，在比对基准和人工评估方面表现出卓越的性能。LMS培训的这种范式转变使我们离开发能够有力而准确地反映社会规范和价值观的人工智能系统又近了一步。



## **47. A First Order Meta Stackelberg Method for Robust Federated Learning**

一种用于鲁棒联邦学习的一阶Meta Stackelberg方法 cs.LG

Accepted to ICML 2023 Workshop on The 2nd New Frontiers In  Adversarial Machine Learning. Associated technical report arXiv:2306.13273

**SubmitDate**: 2023-07-16    [abs](http://arxiv.org/abs/2306.13800v3) [paper-pdf](http://arxiv.org/pdf/2306.13800v3)

**Authors**: Yunian Pan, Tao Li, Henger Li, Tianyi Xu, Zizhan Zheng, Quanyan Zhu

**Abstract**: Previous research has shown that federated learning (FL) systems are exposed to an array of security risks. Despite the proposal of several defensive strategies, they tend to be non-adaptive and specific to certain types of attacks, rendering them ineffective against unpredictable or adaptive threats. This work models adversarial federated learning as a Bayesian Stackelberg Markov game (BSMG) to capture the defender's incomplete information of various attack types. We propose meta-Stackelberg learning (meta-SL), a provably efficient meta-learning algorithm, to solve the equilibrium strategy in BSMG, leading to an adaptable FL defense. We demonstrate that meta-SL converges to the first-order $\varepsilon$-equilibrium point in $O(\varepsilon^{-2})$ gradient iterations, with $O(\varepsilon^{-4})$ samples needed per iteration, matching the state of the art. Empirical evidence indicates that our meta-Stackelberg framework performs exceptionally well against potent model poisoning and backdoor attacks of an uncertain nature.

摘要: 先前的研究表明，联合学习(FL)系统面临一系列安全风险。尽管提出了几种防御战略，但它们往往是非适应性的，并且特定于某些类型的攻击，使得它们对不可预测或适应性威胁无效。该工作将对抗性联邦学习建模为贝叶斯Stackelberg马尔可夫博弈(BSMG)，以捕获防御者各种攻击类型的不完全信息。我们提出了元Stackelberg学习算法(META-SL)来解决BSMG中的均衡策略，从而得到一种自适应的FL防御。我们证明了META-SL在$O(varepsilon^{-2})$梯度迭代中收敛到一阶$varepsilon$-均衡点，每次迭代需要$O(varepsilon^{-4})$样本，与现有技术相匹配。经验证据表明，我们的Meta-Stackelberg框架在对抗强大的模型中毒和不确定性质的后门攻击时表现得非常好。



## **48. Diffusion to Confusion: Naturalistic Adversarial Patch Generation Based on Diffusion Model for Object Detector**

扩散到混乱：基于扩散模型的目标检测器自然主义对抗性补丁生成 cs.CV

**SubmitDate**: 2023-07-16    [abs](http://arxiv.org/abs/2307.08076v1) [paper-pdf](http://arxiv.org/pdf/2307.08076v1)

**Authors**: Shuo-Yen Lin, Ernie Chu, Che-Hsien Lin, Jun-Cheng Chen, Jia-Ching Wang

**Abstract**: Many physical adversarial patch generation methods are widely proposed to protect personal privacy from malicious monitoring using object detectors. However, they usually fail to generate satisfactory patch images in terms of both stealthiness and attack performance without making huge efforts on careful hyperparameter tuning. To address this issue, we propose a novel naturalistic adversarial patch generation method based on the diffusion models (DM). Through sampling the optimal image from the DM model pretrained upon natural images, it allows us to stably craft high-quality and naturalistic physical adversarial patches to humans without suffering from serious mode collapse problems as other deep generative models. To the best of our knowledge, we are the first to propose DM-based naturalistic adversarial patch generation for object detectors. With extensive quantitative, qualitative, and subjective experiments, the results demonstrate the effectiveness of the proposed approach to generate better-quality and more naturalistic adversarial patches while achieving acceptable attack performance than other state-of-the-art patch generation methods. We also show various generation trade-offs under different conditions.

摘要: 许多物理敌意补丁生成方法被广泛提出，以保护个人隐私免受使用对象检测器的恶意监控。然而，如果不对超参数进行仔细的调整，它们通常无法在隐蔽性和攻击性能方面生成令人满意的补丁图像。针对这一问题，我们提出了一种新的基于扩散模型(DM)的自然主义对抗性补丁生成方法。通过从自然图像上预先训练的DM模型中采样最优图像，它允许我们稳定地创建高质量和自然的物理对抗性补丁，而不会像其他深度生成模型那样遭受严重的模式崩溃问题。据我们所知，我们是第一个提出基于DM的自然对抗性补丁生成的目标检测器。通过大量的定量、定性和主观实验，结果表明，该方法在生成质量更好、更具自然感的对抗性补丁的同时，与其他最先进的补丁生成方法相比，具有可接受的攻击性能。我们还展示了在不同条件下的各种代际权衡。



## **49. How to choose your best allies for a transferable attack?**

如何为可转移的攻击选择最好的盟友？ cs.CR

ICCV 2023

**SubmitDate**: 2023-07-16    [abs](http://arxiv.org/abs/2304.02312v2) [paper-pdf](http://arxiv.org/pdf/2304.02312v2)

**Authors**: Thibault Maho, Seyed-Mohsen Moosavi-Dezfooli, Teddy Furon

**Abstract**: The transferability of adversarial examples is a key issue in the security of deep neural networks. The possibility of an adversarial example crafted for a source model fooling another targeted model makes the threat of adversarial attacks more realistic. Measuring transferability is a crucial problem, but the Attack Success Rate alone does not provide a sound evaluation. This paper proposes a new methodology for evaluating transferability by putting distortion in a central position. This new tool shows that transferable attacks may perform far worse than a black box attack if the attacker randomly picks the source model. To address this issue, we propose a new selection mechanism, called FiT, which aims at choosing the best source model with only a few preliminary queries to the target. Our experimental results show that FiT is highly effective at selecting the best source model for multiple scenarios such as single-model attacks, ensemble-model attacks and multiple attacks (Code available at: https://github.com/t-maho/transferability_measure_fit).

摘要: 对抗性样本的可转移性是深层神经网络安全的一个关键问题。为源模型制作的对抗性示例欺骗另一个目标模型的可能性使对抗性攻击的威胁变得更加现实。衡量可转移性是一个关键问题，但仅凭攻击成功率并不能提供合理的评估。本文提出了一种将失真放在中心位置来评价可转移性的新方法。这个新工具表明，如果攻击者随机选择源模型，可转移攻击的性能可能比黑盒攻击差得多。为了解决这个问题，我们提出了一种新的选择机制，称为FIT，它旨在通过对目标的几个初步查询来选择最优源模型。我们的实验结果表明，对于单模型攻击、集成模型攻击和多攻击等多种场景，FIT在选择最佳源模型方面非常有效(代码可在：https://github.com/t-maho/transferability_measure_fit).



## **50. Byzantine-Robust Distributed Online Learning: Taming Adversarial Participants in An Adversarial Environment**

拜占庭-稳健的分布式在线学习：在对抗性环境中驯服对抗性参与者 cs.LG

**SubmitDate**: 2023-07-16    [abs](http://arxiv.org/abs/2307.07980v1) [paper-pdf](http://arxiv.org/pdf/2307.07980v1)

**Authors**: Xingrong Dong, Zhaoxian Wu, Qing Ling, Zhi Tian

**Abstract**: This paper studies distributed online learning under Byzantine attacks. The performance of an online learning algorithm is often characterized by (adversarial) regret, which evaluates the quality of one-step-ahead decision-making when an environment provides adversarial losses, and a sublinear bound is preferred. But we prove that, even with a class of state-of-the-art robust aggregation rules, in an adversarial environment and in the presence of Byzantine participants, distributed online gradient descent can only achieve a linear adversarial regret bound, which is tight. This is the inevitable consequence of Byzantine attacks, even though we can control the constant of the linear adversarial regret to a reasonable level. Interestingly, when the environment is not fully adversarial so that the losses of the honest participants are i.i.d. (independent and identically distributed), we show that sublinear stochastic regret, in contrast to the aforementioned adversarial regret, is possible. We develop a Byzantine-robust distributed online momentum algorithm to attain such a sublinear stochastic regret bound. Extensive numerical experiments corroborate our theoretical analysis.

摘要: 本文研究拜占庭攻击下的分布式在线学习。在线学习算法的性能通常以(对抗性)后悔为特征，当环境提供对抗性损失时，该算法评估领先一步的决策的质量，并且次线性界是首选的。但我们证明了，即使在一类最新的稳健聚集规则下，在对抗性环境下，在拜占庭参与者在场的情况下，分布式在线梯度下降也只能达到线性对抗性遗憾界，这是紧的。这是拜占庭攻击的必然结果，即使我们可以将线性对抗性后悔的常量控制在合理的水平。有趣的是，当环境不是完全对抗性的时候，诚实的参与者的损失是I.I.D.(独立同分布)，我们证明了与前面提到的对抗性后悔相比，次线性随机后悔是可能的。我们开发了一种拜占庭稳健的分布式在线动量算法来获得这样的次线性随机后悔界。大量的数值实验证实了我们的理论分析。



