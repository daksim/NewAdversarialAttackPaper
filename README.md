# NewAdversarialAttackPaper
## **1. Trustworthy Medical Segmentation with Uncertainty Estimation**

基于不确定性估计的可信医学分割 eess.IV

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05978v1)

**Authors**: Giuseppina Carannante, Dimah Dera, Nidhal C. Bouaynaya, Rasool Ghulam, Hassan M. Fathallah-Shaykh

**Abstracts**: Deep Learning (DL) holds great promise in reshaping the healthcare systems given its precision, efficiency, and objectivity. However, the brittleness of DL models to noisy and out-of-distribution inputs is ailing their deployment in the clinic. Most systems produce point estimates without further information about model uncertainty or confidence. This paper introduces a new Bayesian deep learning framework for uncertainty quantification in segmentation neural networks, specifically encoder-decoder architectures. The proposed framework uses the first-order Taylor series approximation to propagate and learn the first two moments (mean and covariance) of the distribution of the model parameters given the training data by maximizing the evidence lower bound. The output consists of two maps: the segmented image and the uncertainty map of the segmentation. The uncertainty in the segmentation decisions is captured by the covariance matrix of the predictive distribution. We evaluate the proposed framework on medical image segmentation data from Magnetic Resonances Imaging and Computed Tomography scans. Our experiments on multiple benchmark datasets demonstrate that the proposed framework is more robust to noise and adversarial attacks as compared to state-of-the-art segmentation models. Moreover, the uncertainty map of the proposed framework associates low confidence (or equivalently high uncertainty) to patches in the test input images that are corrupted with noise, artifacts or adversarial attacks. Thus, the model can self-assess its segmentation decisions when it makes an erroneous prediction or misses part of the segmentation structures, e.g., tumor, by presenting higher values in the uncertainty map.

摘要: 深度学习(DL)由于其精确性、效率和客观性，在重塑医疗系统方面有着巨大的希望。然而，DL模型对噪声和非分布输入的脆性阻碍了它们在临床上的部署。大多数系统在没有关于模型不确定性或置信度的进一步信息的情况下产生点估计。本文介绍了一种新的贝叶斯深度学习框架，用于分段神经网络中的不确定性量化，特别是编解码器的体系结构。该框架使用一阶泰勒级数近似，通过最大化证据下界来传播和学习给定训练数据的模型参数分布的前两个矩(均值和协方差)。输出由两幅图组成：分割后的图像和分割的不确定性图。通过预测分布的协方差矩阵来捕捉分割决策中的不确定性。我们在磁共振成像和计算机断层扫描的医学图像分割数据上对所提出的框架进行了评估。我们在多个基准数据集上的实验表明，与现有的分割模型相比，该框架对噪声和敌意攻击具有更强的鲁棒性。此外，该框架的不确定性图将低置信度(或相当于高不确定性)与测试输入图像中被噪声、伪影或敌意攻击破坏的补丁相关联。因此，当模型做出错误的预测或通过在不确定性图中呈现更高的值来错过部分分割结构(例如，肿瘤)时，该模型可以自我评估其分割决策。



## **2. Robust Learning via Ensemble Density Propagation in Deep Neural Networks**

基于集成密度传播的深度神经网络鲁棒学习 cs.LG

submitted to 2020 IEEE International Workshop on Machine Learning for  Signal Processing

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05953v1)

**Authors**: Giuseppina Carannante, Dimah Dera, Ghulam Rasool, Nidhal C. Bouaynaya, Lyudmila Mihaylova

**Abstracts**: Learning in uncertain, noisy, or adversarial environments is a challenging task for deep neural networks (DNNs). We propose a new theoretically grounded and efficient approach for robust learning that builds upon Bayesian estimation and Variational Inference. We formulate the problem of density propagation through layers of a DNN and solve it using an Ensemble Density Propagation (EnDP) scheme. The EnDP approach allows us to propagate moments of the variational probability distribution across the layers of a Bayesian DNN, enabling the estimation of the mean and covariance of the predictive distribution at the output of the model. Our experiments using MNIST and CIFAR-10 datasets show a significant improvement in the robustness of the trained models to random noise and adversarial attacks.

摘要: 对于深度神经网络(DNNs)来说，在不确定、噪声或敌对环境中学习是一项具有挑战性的任务。在贝叶斯估计和变分推理的基础上，提出了一种新的具有理论基础的、高效的鲁棒学习方法。我们用集合密度传播(ENDP)方案描述了DNN各层间的密度传播问题，并对其进行了求解。ENDP方法允许我们在贝叶斯DNN的各层之间传播变分概率分布的矩，从而能够在模型的输出处估计预测分布的均值和协方差。我们使用MNIST和CIFAR-10数据集进行的实验表明，训练后的模型对随机噪声和敌意攻击的鲁棒性有了显着的提高。



## **3. Sparse Adversarial Video Attacks with Spatial Transformations**

基于空间变换的稀疏对抗性视频攻击 cs.CV

The short version of this work will appear in the BMVC 2021  conference

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05468v1)

**Authors**: Ronghui Mu, Wenjie Ruan, Leandro Soriano Marcolino, Qiang Ni

**Abstracts**: In recent years, a significant amount of research efforts concentrated on adversarial attacks on images, while adversarial video attacks have seldom been explored. We propose an adversarial attack strategy on videos, called DeepSAVA. Our model includes both additive perturbation and spatial transformation by a unified optimisation framework, where the structural similarity index (SSIM) measure is adopted to measure the adversarial distance. We design an effective and novel optimisation scheme which alternatively utilizes Bayesian optimisation to identify the most influential frame in a video and Stochastic gradient descent (SGD) based optimisation to produce both additive and spatial-transformed perturbations. Doing so enables DeepSAVA to perform a very sparse attack on videos for maintaining human imperceptibility while still achieving state-of-the-art performance in terms of both attack success rate and adversarial transferability. Our intensive experiments on various types of deep neural networks and video datasets confirm the superiority of DeepSAVA.

摘要: 近年来，大量的研究工作集中在图像的对抗性攻击上，而对抗性视频攻击的研究很少。我们提出了一种针对视频的对抗性攻击策略，称为DeepSAVA。我们的模型通过一个统一的优化框架同时包括加性扰动和空间变换，其中采用结构相似指数(SSIM)度量对抗距离。我们设计了一种有效和新颖的优化方案，它交替使用贝叶斯优化来识别视频中最有影响力的帧，以及基于随机梯度下降(SGD)的优化来产生加性和空间变换的扰动。这样做使DeepSAVA能够对视频执行非常稀疏的攻击，以保持人的不可感知性，同时在攻击成功率和对手可转移性方面仍获得最先进的性能。我们在不同类型的深度神经网络和视频数据集上的密集实验证实了DeepSAVA的优越性。



## **4. A Statistical Difference Reduction Method for Escaping Backdoor Detection**

一种逃避后门检测的统计减差方法 cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05077v1)

**Authors**: Pengfei Xia, Hongjing Niu, Ziqiang Li, Bin Li

**Abstracts**: Recent studies show that Deep Neural Networks (DNNs) are vulnerable to backdoor attacks. An infected model behaves normally on benign inputs, whereas its prediction will be forced to an attack-specific target on adversarial data. Several detection methods have been developed to distinguish inputs to defend against such attacks. The common hypothesis that these defenses rely on is that there are large statistical differences between the latent representations of clean and adversarial inputs extracted by the infected model. However, although it is important, comprehensive research on whether the hypothesis must be true is lacking. In this paper, we focus on it and study the following relevant questions: 1) What are the properties of the statistical differences? 2) How to effectively reduce them without harming the attack intensity? 3) What impact does this reduction have on difference-based defenses? Our work is carried out on the three questions. First, by introducing the Maximum Mean Discrepancy (MMD) as the metric, we identify that the statistical differences of multi-level representations are all large, not just the highest level. Then, we propose a Statistical Difference Reduction Method (SDRM) by adding a multi-level MMD constraint to the loss function during training a backdoor model to effectively reduce the differences. Last, three typical difference-based detection methods are examined. The F1 scores of these defenses drop from 90%-100% on the regularly trained backdoor models to 60%-70% on the models trained with SDRM on all two datasets, four model architectures, and four attack methods. The results indicate that the proposed method can be used to enhance existing attacks to escape backdoor detection algorithms.

摘要: 最近的研究表明，深度神经网络(DNNs)很容易受到后门攻击。被感染的模型在良性输入上表现正常，而它的预测将被迫在对抗性数据上针对攻击特定的目标。已经开发了几种检测方法来区分输入以防御此类攻击。这些防御所依赖的共同假设是，由感染模型提取的干净和敌对输入的潜在表示之间存在很大的统计差异。然而，尽管这很重要，但关于这一假设是否一定是真的缺乏全面的研究。本文针对这一问题进行了研究：1)统计差异的性质是什么？2)如何在不影响攻击强度的情况下有效地降低统计差异？3)这种减少对基于差异的防御有什么影响？(2)如何在不影响攻击强度的情况下有效地减少统计差异？3)这种减少对基于差异的防御有什么影响？我们的工作就是围绕这三个问题展开的。首先，通过引入最大平均差异(MMD)作为度量，我们发现多级表示的统计差异都很大，而不仅仅是最高级别。然后，在后门模型训练过程中，通过在损失函数中加入多级MMD约束，提出了一种统计差值缩减方法(SDRM)，有效地减小了差值。最后，分析了三种典型的基于差分的检测方法。在所有两个数据集、四个模型体系结构和四种攻击方法上，这些防御的F1得分从定期训练的后门模型的90%-100%下降到使用SDRM训练的模型的60%-70%。实验结果表明，该方法可用于增强现有的逃避后门检测算法的攻击。



## **5. Tightening the Approximation Error of Adversarial Risk with Auto Loss Function Search**

用自动损失函数搜索法缩小对抗性风险的逼近误差 cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05063v1)

**Authors**: Pengfei Xia, Ziqiang Li, Bin Li

**Abstracts**: Numerous studies have demonstrated that deep neural networks are easily misled by adversarial examples. Effectively evaluating the adversarial robustness of a model is important for its deployment in practical applications. Currently, a common type of evaluation is to approximate the adversarial risk of a model as a robustness indicator by constructing malicious instances and executing attacks. Unfortunately, there is an error (gap) between the approximate value and the true value. Previous studies manually design attack methods to achieve a smaller error, which is inefficient and may miss a better solution. In this paper, we establish the tightening of the approximation error as an optimization problem and try to solve it with an algorithm. More specifically, we first analyze that replacing the non-convex and discontinuous 0-1 loss with a surrogate loss, a necessary compromise in calculating the approximation, is one of the main reasons for the error. Then we propose AutoLoss-AR, the first method for searching loss functions for tightening the approximation error of adversarial risk. Extensive experiments are conducted in multiple settings. The results demonstrate the effectiveness of the proposed method: the best-discovered loss functions outperform the handcrafted baseline by 0.9%-2.9% and 0.7%-2.0% on MNIST and CIFAR-10, respectively. Besides, we also verify that the searched losses can be transferred to other settings and explore why they are better than the baseline by visualizing the local loss landscape.

摘要: 大量研究表明，深度神经网络很容易被对抗性例子所误导。有效地评估模型的对抗健壮性对于其在实际应用中的部署具有重要意义。目前，一种常见的评估方法是通过构建恶意实例和执行攻击来近似模型的敌意风险作为健壮性指标。不幸的是，近似值和真实值之间存在误差(差距)。以往的研究都是通过手工设计攻击方法来实现较小的错误，效率较低，可能会错过更好的解决方案。本文将逼近误差的收紧问题建立为优化问题，并尝试用算法求解。更具体地说，我们首先分析了用替代损失代替非凸的、不连续的0-1损失是造成误差的主要原因之一，这是计算近似时的一种必要的折衷。在此基础上，提出了第一种搜索损失函数的方法AutoLoss-AR，以减小对手风险的逼近误差。在多个环境中进行了广泛的实验。结果证明了该方法的有效性：在MNIST和CIFAR-10上，最好发现的损失函数的性能分别比手工制作的基线高0.9%-2.9%和0.7%-2.0%。此外，我们还验证了搜索到的损失可以转移到其他设置，并通过可视化本地损失情况来探索为什么它们比基线更好。



## **6. DeepSteal: Advanced Model Extractions Leveraging Efficient Weight Stealing in Memories**

DeepSteal：高级模型提取，利用记忆中有效的重量窃取 cs.CR

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04625v1)

**Authors**: Adnan Siraj Rakin, Md Hafizul Islam Chowdhuryy, Fan Yao, Deliang Fan

**Abstracts**: Recent advancements of Deep Neural Networks (DNNs) have seen widespread deployment in multiple security-sensitive domains. The need of resource-intensive training and use of valuable domain-specific training data have made these models a top intellectual property (IP) for model owners. One of the major threats to the DNN privacy is model extraction attacks where adversaries attempt to steal sensitive information in DNN models. Recent studies show hardware-based side channel attacks can reveal internal knowledge about DNN models (e.g., model architectures) However, to date, existing attacks cannot extract detailed model parameters (e.g., weights/biases). In this work, for the first time, we propose an advanced model extraction attack framework DeepSteal that effectively steals DNN weights with the aid of memory side-channel attack. Our proposed DeepSteal comprises two key stages. Firstly, we develop a new weight bit information extraction method, called HammerLeak, through adopting the rowhammer based hardware fault technique as the information leakage vector. HammerLeak leverages several novel system-level techniques tailed for DNN applications to enable fast and efficient weight stealing. Secondly, we propose a novel substitute model training algorithm with Mean Clustering weight penalty, which leverages the partial leaked bit information effectively and generates a substitute prototype of the target victim model. We evaluate this substitute model extraction method on three popular image datasets (e.g., CIFAR-10/100/GTSRB) and four DNN architectures (e.g., ResNet-18/34/Wide-ResNet/VGG-11). The extracted substitute model has successfully achieved more than 90 % test accuracy on deep residual networks for the CIFAR-10 dataset. Moreover, our extracted substitute model could also generate effective adversarial input samples to fool the victim model.

摘要: 近年来，深度神经网络(DNNs)在多个安全敏感领域得到了广泛的应用。对资源密集型培训的需求和对有价值的特定领域培训数据的使用已使这些模型成为模型所有者的最高知识产权(IP)。DNN隐私面临的主要威胁之一是模型提取攻击，即攻击者试图窃取DNN模型中的敏感信息。最近的研究表明，基于硬件的侧信道攻击可以揭示DNN模型(例如，模型体系结构)的内部知识，然而，到目前为止，现有的攻击不能提取详细的模型参数(例如，权重/偏差)。在这项工作中，我们首次提出了一个高级模型提取攻击框架DeepSteal，该框架可以借助记忆边信道攻击有效地窃取DNN权重。我们建议的DeepSteal包括两个关键阶段。首先，通过采用基于Rowhammer的硬件故障技术作为信息泄漏向量，提出了一种新的加权比特信息提取方法HammerLeak。HammerLeak利用针对DNN应用的几种新颖的系统级技术来实现快速高效的重量盗窃。其次，提出了一种基于均值聚类权重惩罚的替身模型训练算法，该算法有效地利用了部分泄露的比特信息，生成了目标受害者模型的替身原型。我们在三个流行的图像数据集(如CIFAR10/10 0/GTSRB)和四个数字近邻结构(如Resnet-18/34/Wide-Resnet/VGG-11)上对该替身模型提取方法进行了评估。所提取的替身模型在CIFAR-10数据集上的深层残差网络上的测试准确率已成功达到90%以上。此外，我们提取的替身模型还可以生成有效的敌意输入样本来愚弄受害者模型。



## **7. Generative Dynamic Patch Attack**

生成式动态补丁攻击 cs.CV

Published as a conference paper at BMVC 2021

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04266v1)

**Authors**: Xiang Li, Shihao Ji

**Abstracts**: Adversarial patch attack is a family of attack algorithms that perturb a part of image to fool a deep neural network model. Existing patch attacks mostly consider injecting adversarial patches at input-agnostic locations: either a predefined location or a random location. This attack setup may be sufficient for attack but has considerable limitations when using it for adversarial training. Thus, robust models trained with existing patch attacks cannot effectively defend other adversarial attacks. In this paper, we first propose an end-to-end patch attack algorithm, Generative Dynamic Patch Attack (GDPA), which generates both patch pattern and patch location adversarially for each input image. We show that GDPA is a generic attack framework that can produce dynamic/static and visible/invisible patches with a few configuration changes. Secondly, GDPA can be readily integrated for adversarial training to improve model robustness to various adversarial attacks. Extensive experiments on VGGFace, Traffic Sign and ImageNet show that GDPA achieves higher attack success rates than state-of-the-art patch attacks, while adversarially trained model with GDPA demonstrates superior robustness to adversarial patch attacks than competing methods. Our source code can be found at https://github.com/lxuniverse/gdpa.

摘要: 对抗性补丁攻击是一系列攻击算法，通过扰动图像的一部分来欺骗深层神经网络模型。现有的补丁攻击大多考虑在与输入无关的位置(预定义位置或随机位置)注入敌意补丁。这种攻击设置对于攻击来说可能是足够的，但在用于对抗性训练时有相当大的限制。因此，用现有补丁攻击训练的鲁棒模型不能有效防御其他对抗性攻击。本文首先提出了一种端到端的补丁攻击算法--生成性动态补丁攻击(GDPA)，该算法对每幅输入图像分别生成补丁模式和补丁位置。我们证明了GDPA是一个通用的攻击框架，只需少量的配置更改，就可以生成动态/静电和可见/不可见的补丁。其次，GDPA可以很容易地集成到对抗性训练中，以提高模型对各种对抗性攻击的鲁棒性。在VGGFace、交通标志和ImageNet上的大量实验表明，GDPA比最新的补丁攻击具有更高的攻击成功率，而带有GDPA的对抗性训练模型对敌意补丁攻击表现出比竞争方法更好的鲁棒性。我们的源代码可以在https://github.com/lxuniverse/gdpa.上找到



## **8. Visualizing the Emergence of Intermediate Visual Patterns in DNNs**

在DNNs中可视化中间视觉模式的出现 cs.CV

**SubmitDate**: 2021-11-05    [paper-pdf](http://arxiv.org/pdf/2111.03505v1)

**Authors**: Mingjie Li, Shaobo Wang, Quanshi Zhang

**Abstracts**: This paper proposes a method to visualize the discrimination power of intermediate-layer visual patterns encoded by a DNN. Specifically, we visualize (1) how the DNN gradually learns regional visual patterns in each intermediate layer during the training process, and (2) the effects of the DNN using non-discriminative patterns in low layers to construct disciminative patterns in middle/high layers through the forward propagation. Based on our visualization method, we can quantify knowledge points (i.e., the number of discriminative visual patterns) learned by the DNN to evaluate the representation capacity of the DNN. Furthermore, this method also provides new insights into signal-processing behaviors of existing deep-learning techniques, such as adversarial attacks and knowledge distillation.

摘要: 提出了一种将DNN编码的中间层视觉模式的识别力可视化的方法。具体地说，我们可视化了(1)DNN如何在训练过程中逐渐学习各中间层的区域视觉模式，以及(2)DNN在低层使用非区分模式通过前向传播构建中高层区分模式的效果。基于我们的可视化方法，我们可以量化DNN学习的知识点(即区分视觉模式的数量)来评估DNN的表示能力。此外，该方法还为现有深度学习技术(如对抗性攻击和知识提取)的信号处理行为提供了新的见解。



## **9. Attacking Deep Reinforcement Learning-Based Traffic Signal Control Systems with Colluding Vehicles**

用合谋车辆攻击基于深度强化学习的交通信号控制系统 cs.LG

**SubmitDate**: 2021-11-04    [paper-pdf](http://arxiv.org/pdf/2111.02845v1)

**Authors**: Ao Qu, Yihong Tang, Wei Ma

**Abstracts**: The rapid advancements of Internet of Things (IoT) and artificial intelligence (AI) have catalyzed the development of adaptive traffic signal control systems (ATCS) for smart cities. In particular, deep reinforcement learning (DRL) methods produce the state-of-the-art performance and have great potentials for practical applications. In the existing DRL-based ATCS, the controlled signals collect traffic state information from nearby vehicles, and then optimal actions (e.g., switching phases) can be determined based on the collected information. The DRL models fully "trust" that vehicles are sending the true information to the signals, making the ATCS vulnerable to adversarial attacks with falsified information. In view of this, this paper first time formulates a novel task in which a group of vehicles can cooperatively send falsified information to "cheat" DRL-based ATCS in order to save their total travel time. To solve the proposed task, we develop CollusionVeh, a generic and effective vehicle-colluding framework composed of a road situation encoder, a vehicle interpreter, and a communication mechanism. We employ our method to attack established DRL-based ATCS and demonstrate that the total travel time for the colluding vehicles can be significantly reduced with a reasonable number of learning episodes, and the colluding effect will decrease if the number of colluding vehicles increases. Additionally, insights and suggestions for the real-world deployment of DRL-based ATCS are provided. The research outcomes could help improve the reliability and robustness of the ATCS and better protect the smart mobility systems.

摘要: 物联网(IoT)和人工智能(AI)的快速发展促进了智能城市自适应交通信号控制系统(ATCS)的发展。尤其是深度强化学习(DRL)方法具有最先进的性能和巨大的实际应用潜力。在现有的基于DRL的ATCS中，受控信号收集附近车辆的交通状态信息，然后可以基于收集的信息来确定最优动作(例如，切换相位)。DRL模型完全“信任”车辆正在向信号发送真实的信息，使得ATCS容易受到带有伪造信息的敌意攻击。有鉴于此，本文首次提出了一种新颖的任务，即一组车辆可以协同发送伪造信息来“欺骗”基于DRL的ATC，以节省它们的总行程时间。为了解决这一问题，我们开发了CollusionVeh，这是一个通用的、有效的车辆共谋框架，由路况编码器、车辆解释器和通信机制组成。我们利用我们的方法对已建立的基于DRL的ATCS进行攻击，并证明了在合理的学习场景数下，合谋车辆的总行驶时间可以显著减少，并且合谋效应随着合谋车辆数量的增加而降低。此外，还为基于DRL的ATCS的实际部署提供了见解和建议。研究成果有助于提高ATCS的可靠性和鲁棒性，更好地保护智能移动系统。



## **10. ZeBRA: Precisely Destroying Neural Networks with Zero-Data Based Repeated Bit Flip Attack**

Zebra：基于零数据重复位翻转攻击精确摧毁神经网络 cs.LG

14 pages, 3 figures, 5 tables, Accepted at British Machine Vision  Conference (BMVC) 2021

**SubmitDate**: 2021-11-01    [paper-pdf](http://arxiv.org/pdf/2111.01080v1)

**Authors**: Dahoon Park, Kon-Woo Kwon, Sunghoon Im, Jaeha Kung

**Abstracts**: In this paper, we present Zero-data Based Repeated bit flip Attack (ZeBRA) that precisely destroys deep neural networks (DNNs) by synthesizing its own attack datasets. Many prior works on adversarial weight attack require not only the weight parameters, but also the training or test dataset in searching vulnerable bits to be attacked. We propose to synthesize the attack dataset, named distilled target data, by utilizing the statistics of batch normalization layers in the victim DNN model. Equipped with the distilled target data, our ZeBRA algorithm can search vulnerable bits in the model without accessing training or test dataset. Thus, our approach makes the adversarial weight attack more fatal to the security of DNNs. Our experimental results show that 2.0x (CIFAR-10) and 1.6x (ImageNet) less number of bit flips are required on average to destroy DNNs compared to the previous attack method. Our code is available at https://github. com/pdh930105/ZeBRA.

摘要: 本文提出了一种基于零数据的重复位翻转攻击(Zebra)，它通过合成自己的攻击数据集来精确地破坏深度神经网络(DNNs)。以往许多关于对抗性权重攻击的工作不仅需要权重参数，还需要训练或测试数据集来搜索易受攻击的部位。我们提出利用受害者DNN模型中的批归一化层的统计信息来合成攻击数据集，称为提取的目标数据。有了提取的目标数据，我们的斑马算法可以在不访问训练或测试数据集的情况下搜索模型中的易受攻击的位。因此，我们的方法使得敌意加权攻击对DNNs的安全性更加致命。我们的实验结果表明，与以前的攻击方法相比，破坏DNN平均需要减少2.0倍(CIFAR-10)和1.6倍(ImageNet)的比特翻转次数。我们的代码可在https://github.获得com/pdh930105/zebra。



## **11. Robustness of deep learning algorithms in astronomy -- galaxy morphology studies**

深度学习算法在天文学中的稳健性--星系形态学研究 astro-ph.GA

Accepted in: Fourth Workshop on Machine Learning and the Physical  Sciences (35th Conference on Neural Information Processing Systems;  NeurIPS2021); final version

**SubmitDate**: 2021-11-02    [paper-pdf](http://arxiv.org/pdf/2111.00961v2)

**Authors**: A. Ćiprijanović, D. Kafkes, G. N. Perdue, K. Pedro, G. Snyder, F. J. Sánchez, S. Madireddy, S. M. Wild, B. Nord

**Abstracts**: Deep learning models are being increasingly adopted in wide array of scientific domains, especially to handle high-dimensionality and volume of the scientific data. However, these models tend to be brittle due to their complexity and overparametrization, especially to the inadvertent adversarial perturbations that can appear due to common image processing such as compression or blurring that are often seen with real scientific data. It is crucial to understand this brittleness and develop models robust to these adversarial perturbations. To this end, we study the effect of observational noise from the exposure time, as well as the worst case scenario of a one-pixel attack as a proxy for compression or telescope errors on performance of ResNet18 trained to distinguish between galaxies of different morphologies in LSST mock data. We also explore how domain adaptation techniques can help improve model robustness in case of this type of naturally occurring attacks and help scientists build more trustworthy and stable models.

摘要: 深度学习模型正被越来越多的科学领域所采用，特别是在处理高维和海量的科学数据方面。然而，由于它们的复杂性和过度参数化，这些模型往往是脆弱的，特别是由于在真实科学数据中经常看到的常见图像处理(例如压缩或模糊)可能会出现无意中的对抗性扰动。理解这种脆弱性并开发出对这些对抗性扰动具有健壮性的模型是至关重要的。为此，我们研究了来自曝光时间的观测噪声的影响，以及单像素攻击作为压缩或望远镜误差的替代对ResNet18性能的最坏情况的影响，所训练的ResNet18在LSST模拟数据中区分不同形态的星系。我们还探讨了领域自适应技术如何在这种自然发生的攻击情况下帮助提高模型的健壮性，并帮助科学家建立更可靠和更稳定的模型。



## **12. A Frequency Perspective of Adversarial Robustness**

对抗性稳健性的频率透视 cs.CV

**SubmitDate**: 2021-10-26    [paper-pdf](http://arxiv.org/pdf/2111.00861v1)

**Authors**: Shishira R Maiya, Max Ehrlich, Vatsal Agarwal, Ser-Nam Lim, Tom Goldstein, Abhinav Shrivastava

**Abstracts**: Adversarial examples pose a unique challenge for deep learning systems. Despite recent advances in both attacks and defenses, there is still a lack of clarity and consensus in the community about the true nature and underlying properties of adversarial examples. A deep understanding of these examples can provide new insights towards the development of more effective attacks and defenses. Driven by the common misconception that adversarial examples are high-frequency noise, we present a frequency-based understanding of adversarial examples, supported by theoretical and empirical findings. Our analysis shows that adversarial examples are neither in high-frequency nor in low-frequency components, but are simply dataset dependent. Particularly, we highlight the glaring disparities between models trained on CIFAR-10 and ImageNet-derived datasets. Utilizing this framework, we analyze many intriguing properties of training robust models with frequency constraints, and propose a frequency-based explanation for the commonly observed accuracy vs. robustness trade-off.

摘要: 对抗性的例子给深度学习系统带来了独特的挑战。尽管最近在攻击和防御方面都取得了进展，但对于对抗性例子的真实性质和潜在属性，社会上仍然缺乏清晰度和共识。深入理解这些例子可以为开发更有效的攻击和防御提供新的见解。由于普遍认为对抗性例子是高频噪声的误解，我们提出了基于频率的对抗性例子的理解，并得到了理论和实证结果的支持。我们的分析表明，对抗性示例既不在高频成分中，也不在低频成分中，而只是简单地依赖于数据集。特别是，我们强调了在CIFAR-10和ImageNet派生的数据集上训练的模型之间的明显差异。利用该框架，我们分析了具有频率约束的训练鲁棒模型的许多有趣的性质，并对通常观察到的精度与鲁棒性之间的权衡提出了一种基于频率的解释。



## **13. Attacking Video Recognition Models with Bullet-Screen Comments**

用弹幕评论攻击视频识别模型 cs.CV

**SubmitDate**: 2021-10-29    [paper-pdf](http://arxiv.org/pdf/2110.15629v1)

**Authors**: Kai Chen, Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstracts**: Recent research has demonstrated that Deep Neural Networks (DNNs) are vulnerable to adversarial patches which introducing perceptible but localized changes to the input. Nevertheless, existing approaches have focused on generating adversarial patches on images, their counterparts in videos have been less explored. Compared with images, attacking videos is much more challenging as it needs to consider not only spatial cues but also temporal cues. To close this gap, we introduce a novel adversarial attack in this paper, the bullet-screen comment (BSC) attack, which attacks video recognition models with BSCs. Specifically, adversarial BSCs are generated with a Reinforcement Learning (RL) framework, where the environment is set as the target model and the agent plays the role of selecting the position and transparency of each BSC. By continuously querying the target models and receiving feedback, the agent gradually adjusts its selection strategies in order to achieve a high fooling rate with non-overlapping BSCs. As BSCs can be regarded as a kind of meaningful patch, adding it to a clean video will not affect people' s understanding of the video content, nor will arouse people' s suspicion. We conduct extensive experiments to verify the effectiveness of the proposed method. On both UCF-101 and HMDB-51 datasets, our BSC attack method can achieve about 90\% fooling rate when attack three mainstream video recognition models, while only occluding \textless 8\% areas in the video.

摘要: 最近的研究表明，深度神经网络(DNNs)很容易受到敌意补丁的攻击，这些补丁会给输入带来可感知的局部变化。然而，现有的方法主要集中在生成图像上的对抗性补丁，而对视频中的对应补丁的研究较少。与图像相比，攻击视频更具挑战性，因为它不仅需要考虑空间线索，还需要考虑时间线索。为了缩小这一差距，本文引入了一种新的对抗性攻击，即弹幕评论(BSC)攻击，它利用弹幕评论攻击视频识别模型。具体地说，利用强化学习(RL)框架生成对抗性BSC，其中环境被设置为目标模型，Agent扮演选择每个BSC的位置和透明度的角色。通过不断查询目标模型并接收反馈，Agent逐渐调整其选择策略，以获得不重叠的BSC的较高愚弄率。由于BSCS可以看作是一种有意义的补丁，将其添加到干净的视频中不会影响人们对视频内容的理解，也不会引起人们的怀疑。为了验证该方法的有效性，我们进行了大量的实验。在UCF-101和HMDB-51两个数据集上，我们的BSC攻击方法在攻击三种主流视频识别模型时，仅对视频中的8个无遮挡区域进行攻击，可以达到90%左右的蒙骗率。



## **14. Bridge the Gap Between CV and NLP! A Gradient-based Textual Adversarial Attack Framework**

弥合简历和NLP之间的鸿沟！一种基于梯度的文本对抗性攻击框架 cs.CL

Work on progress

**SubmitDate**: 2021-10-28    [paper-pdf](http://arxiv.org/pdf/2110.15317v1)

**Authors**: Lifan Yuan, Yichi Zhang, Yangyi Chen, Wei Wei

**Abstracts**: Despite great success on many machine learning tasks, deep neural networks are still vulnerable to adversarial samples. While gradient-based adversarial attack methods are well-explored in the field of computer vision, it is impractical to directly apply them in natural language processing due to the discrete nature of text. To bridge this gap, we propose a general framework to adapt existing gradient-based methods to craft textual adversarial samples. In this framework, gradient-based continuous perturbations are added to the embedding layer and are amplified in the forward propagation process. Then the final perturbed latent representations are decoded with a mask language model head to obtain potential adversarial samples. In this paper, we instantiate our framework with \textbf{T}extual \textbf{P}rojected \textbf{G}radient \textbf{D}escent (\textbf{TPGD}). We conduct comprehensive experiments to evaluate our framework by performing transfer black-box attacks on BERT, RoBERTa and ALBERT on three benchmark datasets. Experimental results demonstrate our method achieves an overall better performance and produces more fluent and grammatical adversarial samples compared to strong baseline methods. All the code and data will be made public.

摘要: 尽管深度神经网络在许多机器学习任务中取得了巨大的成功，但它仍然容易受到敌意样本的影响。虽然基于梯度的对抗性攻击方法在计算机视觉领域得到了很好的探索，但由于文本的离散性，将其直接应用于自然语言处理是不切实际的。为了弥补这一差距，我们提出了一个通用框架，以适应现有的基于梯度的方法来制作文本对抗性样本。在该框架中，基于梯度的连续扰动被添加到嵌入层，并在前向传播过程中被放大。然后用掩码语言模型头部对最终扰动的潜在表示进行解码，得到潜在的对抗性样本。在本文中，我们用\textbf{T}extual\textbf{P}rojected\textbf{G}Radient\textbf{D}light(\textbf{tpgd})实例化我们的框架。我们通过在三个基准数据集上对Bert、Roberta和Albert进行传输黑盒攻击，对我们的框架进行了全面的测试。实验结果表明，与强基线方法相比，我们的方法取得了总体上更好的性能，生成了更流畅、更具语法意义的对抗性样本。所有的代码和数据都将公之于众。



## **15. Adversarial Robustness in Multi-Task Learning: Promises and Illusions**

多任务学习中的对抗性稳健性：承诺与幻想 cs.LG

**SubmitDate**: 2021-10-26    [paper-pdf](http://arxiv.org/pdf/2110.15053v1)

**Authors**: Salah Ghamizi, Maxime Cordy, Mike Papadakis, Yves Le Traon

**Abstracts**: Vulnerability to adversarial attacks is a well-known weakness of Deep Neural networks. While most of the studies focus on single-task neural networks with computer vision datasets, very little research has considered complex multi-task models that are common in real applications. In this paper, we evaluate the design choices that impact the robustness of multi-task deep learning networks. We provide evidence that blindly adding auxiliary tasks, or weighing the tasks provides a false sense of robustness. Thereby, we tone down the claim made by previous research and study the different factors which may affect robustness. In particular, we show that the choice of the task to incorporate in the loss function are important factors that can be leveraged to yield more robust models.

摘要: 对敌意攻击的脆弱性是深度神经网络的一个众所周知的弱点。虽然大多数研究集中在具有计算机视觉数据集的单任务神经网络，但很少有研究考虑实际应用中常见的复杂多任务模型。在本文中，我们评估了影响多任务深度学习网络健壮性的设计选择。我们提供的证据表明，盲目添加辅助任务或对任务进行加权会带来一种错误的健壮感。因此，我们淡化了以往研究的结论，并研究了可能影响稳健性的不同因素。特别地，我们表明，选择要纳入损失函数的任务是可以用来产生更健壮的模型的重要因素。



## **16. AEVA: Black-box Backdoor Detection Using Adversarial Extreme Value Analysis**

AEVA：基于对抗性极值分析的黑盒后门检测 cs.LG

**SubmitDate**: 2021-10-29    [paper-pdf](http://arxiv.org/pdf/2110.14880v2)

**Authors**: Junfeng Guo, Ang Li, Cong Liu

**Abstracts**: Deep neural networks (DNNs) are proved to be vulnerable against backdoor attacks. A backdoor is often embedded in the target DNNs through injecting a backdoor trigger into training examples, which can cause the target DNNs misclassify an input attached with the backdoor trigger. Existing backdoor detection methods often require the access to the original poisoned training data, the parameters of the target DNNs, or the predictive confidence for each given input, which are impractical in many real-world applications, e.g., on-device deployed DNNs. We address the black-box hard-label backdoor detection problem where the DNN is fully black-box and only its final output label is accessible. We approach this problem from the optimization perspective and show that the objective of backdoor detection is bounded by an adversarial objective. Further theoretical and empirical studies reveal that this adversarial objective leads to a solution with highly skewed distribution; a singularity is often observed in the adversarial map of a backdoor-infected example, which we call the adversarial singularity phenomenon. Based on this observation, we propose the adversarial extreme value analysis(AEVA) to detect backdoors in black-box neural networks. AEVA is based on an extreme value analysis of the adversarial map, computed from the monte-carlo gradient estimation. Evidenced by extensive experiments across multiple popular tasks and backdoor attacks, our approach is shown effective in detecting backdoor attacks under the black-box hard-label scenarios.

摘要: 深度神经网络(DNNs)被证明是易受后门攻击的。通过将后门触发器注入到训练示例中，通常将后门嵌入到目标DNN中，这可能导致目标DNN对与后门触发器附加的输入进行错误分类。现有的后门检测方法通常需要访问原始有毒训练数据、目标DNN的参数或每个给定输入的预测置信度，这在许多真实世界应用中是不切实际的，例如在设备上部署的DNN。我们解决了黑盒硬标签后门检测问题，其中DNN是完全黑盒的，并且只有其最终输出标签是可访问的。我们从优化的角度来研究这个问题，并证明了后门检测的目标是由一个对抗性目标限定的。进一步的理论和实证研究表明，这种对抗性目标导致了一个具有高度偏态分布的解决方案；在一个被后门感染的例子的对抗性地图中经常观察到一个奇点，我们称之为对抗性奇点现象。基于这一观察，我们提出了对抗性极值分析(AEVA)来检测黑盒神经网络中的后门。AEVA是基于对敌方地图的极值分析，通过蒙特卡洛梯度估计计算出来的。通过对多个流行任务和后门攻击的大量实验证明，我们的方法在黑盒硬标签场景下检测后门攻击是有效的。



## **17. Evaluating Deep Learning Models and Adversarial Attacks on Accelerometer-Based Gesture Authentication**

基于加速度计的手势认证深度学习模型和攻击评估 cs.CR

**SubmitDate**: 2021-10-03    [paper-pdf](http://arxiv.org/pdf/2110.14597v1)

**Authors**: Elliu Huang, Fabio Di Troia, Mark Stamp

**Abstracts**: Gesture-based authentication has emerged as a non-intrusive, effective means of authenticating users on mobile devices. Typically, such authentication techniques have relied on classical machine learning techniques, but recently, deep learning techniques have been applied this problem. Although prior research has shown that deep learning models are vulnerable to adversarial attacks, relatively little research has been done in the adversarial domain for behavioral biometrics. In this research, we collect tri-axial accelerometer gesture data (TAGD) from 46 users and perform classification experiments with both classical machine learning and deep learning models. Specifically, we train and test support vector machines (SVM) and convolutional neural networks (CNN). We then consider a realistic adversarial attack, where we assume the attacker has access to real users' TAGD data, but not the authentication model. We use a deep convolutional generative adversarial network (DC-GAN) to create adversarial samples, and we show that our deep learning model is surprisingly robust to such an attack scenario.

摘要: 基于手势的身份验证已经成为一种在移动设备上对用户进行身份验证的非侵入性的有效手段。通常，这样的认证技术依赖于经典的机器学习技术，但最近，深度学习技术已经应用到这个问题上。虽然先前的研究表明深度学习模型容易受到敌意攻击，但在对抗性领域行为生物特征识别方面的研究相对较少。在这项研究中，我们收集了46个用户的三轴加速度计手势数据(TAGD)，并用经典机器学习模型和深度学习模型进行了分类实验。具体地说，我们训练和测试了支持向量机(SVM)和卷积神经网络(CNN)。然后，我们考虑一个现实的对抗性攻击，其中我们假设攻击者可以访问真实用户的TAGD数据，但不能访问身份验证模型。我们使用深度卷积生成对抗性网络(DC-GAN)来创建对抗性样本，并且我们表明我们的深度学习模型对于这样的攻击场景具有出人意料的健壮性。



## **18. Drawing Robust Scratch Tickets: Subnetworks with Inborn Robustness Are Found within Randomly Initialized Networks**

绘制健壮的暂存券：在随机初始化的网络中发现具有天生健壮性的子网 cs.LG

Accepted at NeurIPS 2021

**SubmitDate**: 2021-11-06    [paper-pdf](http://arxiv.org/pdf/2110.14068v2)

**Authors**: Yonggan Fu, Qixuan Yu, Yang Zhang, Shang Wu, Xu Ouyang, David Cox, Yingyan Lin

**Abstracts**: Deep Neural Networks (DNNs) are known to be vulnerable to adversarial attacks, i.e., an imperceptible perturbation to the input can mislead DNNs trained on clean images into making erroneous predictions. To tackle this, adversarial training is currently the most effective defense method, by augmenting the training set with adversarial samples generated on the fly. Interestingly, we discover for the first time that there exist subnetworks with inborn robustness, matching or surpassing the robust accuracy of the adversarially trained networks with comparable model sizes, within randomly initialized networks without any model training, indicating that adversarial training on model weights is not indispensable towards adversarial robustness. We name such subnetworks Robust Scratch Tickets (RSTs), which are also by nature efficient. Distinct from the popular lottery ticket hypothesis, neither the original dense networks nor the identified RSTs need to be trained. To validate and understand this fascinating finding, we further conduct extensive experiments to study the existence and properties of RSTs under different models, datasets, sparsity patterns, and attacks, drawing insights regarding the relationship between DNNs' robustness and their initialization/overparameterization. Furthermore, we identify the poor adversarial transferability between RSTs of different sparsity ratios drawn from the same randomly initialized dense network, and propose a Random RST Switch (R2S) technique, which randomly switches between different RSTs, as a novel defense method built on top of RSTs. We believe our findings about RSTs have opened up a new perspective to study model robustness and extend the lottery ticket hypothesis.

摘要: 深度神经网络(DNNs)很容易受到敌意攻击，即输入的不知不觉的扰动会误导训练在干净图像上的DNN做出错误的预测。对撞击来说，对抗性训练是目前最有效的防御方法，通过使用飞翔上生成的对抗性样本来扩大训练集。有趣的是，我们首次发现，在没有任何模型训练的随机初始化网络中，存在具有天生鲁棒性的子网络，其鲁棒性精度与具有相似模型大小的对抗性训练网络相当或超过，这表明对抗性模型权重的训练对于对抗性鲁棒性来说并不是必不可少的。我们将这样的子网命名为健壮的暂存票(RST)，它本质上也是有效的。与流行的彩票假设不同，原始的密集网络和识别出的RST都不需要训练。为了验证和理解这一有趣的发现，我们进一步进行了大量的实验，研究了不同模型、数据集、稀疏模式和攻击下RST的存在和性质，得出了DNNs的健壮性与其初始化/过参数化之间的关系。此外，我们还发现了来自同一随机初始化密集网络的不同稀疏比的RST之间的对抗性较差，并提出了一种在RST之上随机切换的随机RST切换(R2S)技术，作为一种新的防御方法。我们相信，我们关于RST的发现为研究模型的稳健性和扩展彩票假说开辟了一个新的视角。



## **19. Disrupting Deep Uncertainty Estimation Without Harming Accuracy**

在不损害准确性的情况下中断深度不确定性估计 cs.LG

To be published in NeurIPS 2021

**SubmitDate**: 2021-10-26    [paper-pdf](http://arxiv.org/pdf/2110.13741v1)

**Authors**: Ido Galil, Ran El-Yaniv

**Abstracts**: Deep neural networks (DNNs) have proven to be powerful predictors and are widely used for various tasks. Credible uncertainty estimation of their predictions, however, is crucial for their deployment in many risk-sensitive applications. In this paper we present a novel and simple attack, which unlike adversarial attacks, does not cause incorrect predictions but instead cripples the network's capacity for uncertainty estimation. The result is that after the attack, the DNN is more confident of its incorrect predictions than about its correct ones without having its accuracy reduced. We present two versions of the attack. The first scenario focuses on a black-box regime (where the attacker has no knowledge of the target network) and the second scenario attacks a white-box setting. The proposed attack is only required to be of minuscule magnitude for its perturbations to cause severe uncertainty estimation damage, with larger magnitudes resulting in completely unusable uncertainty estimations. We demonstrate successful attacks on three of the most popular uncertainty estimation methods: the vanilla softmax score, Deep Ensembles and MC-Dropout. Additionally, we show an attack on SelectiveNet, the selective classification architecture. We test the proposed attack on several contemporary architectures such as MobileNetV2 and EfficientNetB0, all trained to classify ImageNet.

摘要: 深度神经网络(DNNs)已被证明是一种强大的预测器，并被广泛应用于各种任务。然而，对他们的预测进行可信的不确定性估计，对于他们在许多风险敏感的应用中的部署至关重要。本文提出了一种新颖而简单的攻击，与对抗性攻击不同，它不会导致错误的预测，而是削弱了网络的不确定性估计能力。其结果是，在攻击之后，DNN对其错误的预测比对其正确的预测更有信心，而不会降低其准确性。我们呈现两个版本的攻击。第一个场景侧重于黑盒机制(攻击者不知道目标网络)，第二个场景攻击白盒设置。所提出的攻击只需要极小的震级，其摄动就会造成严重的不确定性估计损害，而较大的震级会导致完全不可用的不确定性估计。我们展示了对三种最流行的不确定性估计方法的成功攻击：Vanilla Softmax评分、深度集成和MC-Dropout。此外，我们还展示了对选择性分类体系结构SelectiveNet的攻击。我们在几种当代架构(如MobileNetV2和EfficientNetB0)上测试了所提出的攻击，所有这些架构都被训练为对ImageNet进行分类。



## **20. Stable Neural ODE with Lyapunov-Stable Equilibrium Points for Defending Against Adversarial Attacks**

具有Lyapunov稳定平衡点的防御敌意攻击的稳定神经微分方程组 cs.LG

**SubmitDate**: 2021-10-25    [paper-pdf](http://arxiv.org/pdf/2110.12976v1)

**Authors**: Qiyu Kang, Yang Song, Qinxu Ding, Wee Peng Tay

**Abstracts**: Deep neural networks (DNNs) are well-known to be vulnerable to adversarial attacks, where malicious human-imperceptible perturbations are included in the input to the deep network to fool it into making a wrong classification. Recent studies have demonstrated that neural Ordinary Differential Equations (ODEs) are intrinsically more robust against adversarial attacks compared to vanilla DNNs. In this work, we propose a stable neural ODE with Lyapunov-stable equilibrium points for defending against adversarial attacks (SODEF). By ensuring that the equilibrium points of the ODE solution used as part of SODEF is Lyapunov-stable, the ODE solution for an input with a small perturbation converges to the same solution as the unperturbed input. We provide theoretical results that give insights into the stability of SODEF as well as the choice of regularizers to ensure its stability. Our analysis suggests that our proposed regularizers force the extracted feature points to be within a neighborhood of the Lyapunov-stable equilibrium points of the ODE. SODEF is compatible with many defense methods and can be applied to any neural network's final regressor layer to enhance its stability against adversarial attacks.

摘要: 众所周知，深层神经网络(DNNs)容易受到敌意攻击，在深层网络的输入中包含恶意的人类无法察觉的扰动，以欺骗它进行错误的分类。最近的研究表明，与普通的DNN相比，神经常微分方程(ODE)在本质上对敌意攻击具有更强的鲁棒性。在这项工作中，我们提出了一种具有Lyapunov稳定平衡点的稳定神经微分方程组(SODEF)来防御对手攻击。通过保证作为SODEF的一部分的常微分方程组的平衡点是Lyapunov稳定的，具有小扰动的输入的常微分方程组收敛到与未摄动输入相同的解。我们提供了理论结果，为SODEF的稳定性以及确保其稳定性的正则化因子的选择提供了见解。我们的分析表明，我们提出的正则化方法迫使提取的特征点在常微分方程的Lyapunov稳定平衡点的邻域内。SODEF与许多防御方法兼容，可以应用于任何神经网络的最终回归层，以增强其对对手攻击的稳定性。



## **21. Generating Watermarked Adversarial Texts**

生成带水印的敌意文本 cs.CR

https://scholar.google.com/citations?user=IdiF7M0AAAAJ&hl=en

**SubmitDate**: 2021-10-25    [paper-pdf](http://arxiv.org/pdf/2110.12948v1)

**Authors**: Mingjie Li, Hanzhou Wu, Xinpeng Zhang

**Abstracts**: Adversarial example generation has been a hot spot in recent years because it can cause deep neural networks (DNNs) to misclassify the generated adversarial examples, which reveals the vulnerability of DNNs, motivating us to find good solutions to improve the robustness of DNN models. Due to the extensiveness and high liquidity of natural language over the social networks, various natural language based adversarial attack algorithms have been proposed in the literature. These algorithms generate adversarial text examples with high semantic quality. However, the generated adversarial text examples may be maliciously or illegally used. In order to tackle with this problem, we present a general framework for generating watermarked adversarial text examples. For each word in a given text, a set of candidate words are determined to ensure that all the words in the set can be used to either carry secret bits or facilitate the construction of adversarial example. By applying a word-level adversarial text generation algorithm, the watermarked adversarial text example can be finally generated. Experiments show that the adversarial text examples generated by the proposed method not only successfully fool advanced DNN models, but also carry a watermark that can effectively verify the ownership and trace the source of the adversarial examples. Moreover, the watermark can still survive after attacked with adversarial example generation algorithms, which has shown the applicability and superiority.

摘要: 对抗性实例生成是近年来的一个研究热点，因为它会导致深层神经网络(DNNs)对生成的对抗性实例进行误分类，从而暴露了DNN模型的脆弱性，促使我们寻找好的解决方案来提高DNN模型的健壮性。由于自然语言在社会网络上的广泛性和高流动性，文献中提出了各种基于自然语言的对抗性攻击算法。这些算法生成语义质量较高的对抗性文本实例。然而，生成的敌意文本示例可能被恶意或非法使用。为了解决撞击中的这个问题，我们给出了一个生成带水印的敌意文本示例的通用框架。对于给定文本中的每个词，确定候选词的集合，以确保该集合中的所有词都可以用于携带秘密比特或便于构建对抗性示例。应用词级对抗性文本生成算法，最终生成带水印的对抗性文本实例。实验表明，该方法生成的敌意文本实例不仅成功地欺骗了高级DNN模型，而且还携带了水印，可以有效地验证敌意实例的所有权并追踪其来源。此外，利用对抗性实例生成算法，水印在遭受攻击后仍能存活，显示了其适用性和优越性。



## **22. Alignment Attention by Matching Key and Query Distributions**

通过匹配键和查询分布注意对齐 cs.LG

NeurIPS 2021; Our code is publicly available at  https://github.com/szhang42/alignment_attention

**SubmitDate**: 2021-10-25    [paper-pdf](http://arxiv.org/pdf/2110.12567v1)

**Authors**: Shujian Zhang, Xinjie Fan, Huangjie Zheng, Korawat Tanwisuth, Mingyuan Zhou

**Abstracts**: The neural attention mechanism has been incorporated into deep neural networks to achieve state-of-the-art performance in various domains. Most such models use multi-head self-attention which is appealing for the ability to attend to information from different perspectives. This paper introduces alignment attention that explicitly encourages self-attention to match the distributions of the key and query within each head. The resulting alignment attention networks can be optimized as an unsupervised regularization in the existing attention framework. It is simple to convert any models with self-attention, including pre-trained ones, to the proposed alignment attention. On a variety of language understanding tasks, we show the effectiveness of our method in accuracy, uncertainty estimation, generalization across domains, and robustness to adversarial attacks. We further demonstrate the general applicability of our approach on graph attention and visual question answering, showing the great potential of incorporating our alignment method into various attention-related tasks.

摘要: 神经注意机制已经被融入到深度神经网络中，以在各个领域实现最先进的性能。大多数这样的模型使用多头自我注意，这吸引了人们从不同角度关注信息的能力。本文引入了对齐注意，它明确地鼓励自我注意，以匹配每个头部中的键和查询的分布。由此产生的对齐注意网络可以在现有的注意框架中作为一种无监督的正则化进行优化。将任何具有自我注意的模型(包括预先训练的模型)转换为建议的对齐注意是很简单的。在不同的语言理解任务上，我们展示了我们的方法在准确性、不确定性估计、跨域泛化和对敌意攻击的鲁棒性方面的有效性。我们进一步展示了我们的方法在图形注意和视觉问题回答上的普遍适用性，显示了将我们的对齐方法整合到各种与注意相关的任务中的巨大潜力。



## **23. Towards A Conceptually Simple Defensive Approach for Few-shot classifiers Against Adversarial Support Samples**

针对对抗性支持样本的少射分类器概念上的简单防御方法 cs.LG

arXiv admin note: text overlap with arXiv:2012.06330

**SubmitDate**: 2021-10-24    [paper-pdf](http://arxiv.org/pdf/2110.12357v1)

**Authors**: Yi Xiang Marcus Tan, Penny Chong, Jiamei Sun, Ngai-man Cheung, Yuval Elovici, Alexander Binder

**Abstracts**: Few-shot classifiers have been shown to exhibit promising results in use cases where user-provided labels are scarce. These models are able to learn to predict novel classes simply by training on a non-overlapping set of classes. This can be largely attributed to the differences in their mechanisms as compared to conventional deep networks. However, this also offers new opportunities for novel attackers to induce integrity attacks against such models, which are not present in other machine learning setups. In this work, we aim to close this gap by studying a conceptually simple approach to defend few-shot classifiers against adversarial attacks. More specifically, we propose a simple attack-agnostic detection method, using the concept of self-similarity and filtering, to flag out adversarial support sets which destroy the understanding of a victim classifier for a certain class. Our extended evaluation on the miniImagenet (MI) and CUB datasets exhibit good attack detection performance, across three different few-shot classifiers and across different attack strengths, beating baselines. Our observed results allow our approach to establishing itself as a strong detection method for support set poisoning attacks. We also show that our approach constitutes a generalizable concept, as it can be paired with other filtering functions. Finally, we provide an analysis of our results when we vary two components found in our detection approach.

摘要: 已经证明，在用户提供的标签稀缺的情况下，少射分类器显示出有希望的结果。这些模型能够简单地通过对一组不重叠的类进行训练来学习预测新类。这在很大程度上可以归因于它们与传统的深层网络在机制上的不同。然而，这也为新的攻击者提供了新的机会来诱导针对这些模型的完整性攻击，这在其他机器学习设置中是不存在的。在这项工作中，我们的目标是通过研究一种概念上简单的方法来保护少射分类器免受对手攻击，从而缩小这一差距。更具体地说，我们提出了一种简单的攻击不可知性检测方法，利用自相似和过滤的概念来剔除破坏受害者分类器对某一类的理解的敌意支持集。我们在MiniImagenet(MI)和CUB数据集上的扩展评估显示，在三种不同的少发分类器和不同攻击强度、超过基线的情况下，都显示出良好的攻击检测性能。我们的观察结果使我们的方法成为支持集中毒攻击的一种强有力的检测方法。我们还表明，我们的方法构成了一个可推广的概念，因为它可以与其他过滤函数配对。最后，当我们改变检测方法中发现的两个组件时，我们对结果进行了分析。



## **24. ADC: Adversarial attacks against object Detection that evade Context consistency checks**

ADC：逃避上下文一致性检查的针对对象检测的对抗性攻击 cs.CV

WCAV'22 Acceptted

**SubmitDate**: 2021-10-24    [paper-pdf](http://arxiv.org/pdf/2110.12321v1)

**Authors**: Mingjun Yin, Shasha Li, Chengyu Song, M. Salman Asif, Amit K. Roy-Chowdhury, Srikanth V. Krishnamurthy

**Abstracts**: Deep Neural Networks (DNNs) have been shown to be vulnerable to adversarial examples, which are slightly perturbed input images which lead DNNs to make wrong predictions. To protect from such examples, various defense strategies have been proposed. A very recent defense strategy for detecting adversarial examples, that has been shown to be robust to current attacks, is to check for intrinsic context consistencies in the input data, where context refers to various relationships (e.g., object-to-object co-occurrence relationships) in images. In this paper, we show that even context consistency checks can be brittle to properly crafted adversarial examples and to the best of our knowledge, we are the first to do so. Specifically, we propose an adaptive framework to generate examples that subvert such defenses, namely, Adversarial attacks against object Detection that evade Context consistency checks (ADC). In ADC, we formulate a joint optimization problem which has two attack goals, viz., (i) fooling the object detector and (ii) evading the context consistency check system, at the same time. Experiments on both PASCAL VOC and MS COCO datasets show that examples generated with ADC fool the object detector with a success rate of over 85% in most cases, and at the same time evade the recently proposed context consistency checks, with a bypassing rate of over 80% in most cases. Our results suggest that how to robustly model context and check its consistency, is still an open problem.

摘要: 深度神经网络(DNNs)被证明容易受到敌意例子的攻击，这些例子是轻微扰动的输入图像，导致DNNs做出错误的预测。为了保护自己不受此类例子的伤害，各种防御策略应运而生。用于检测已被证明对当前攻击是稳健的敌意示例的最新防御策略是检查输入数据中的内在上下文一致性，其中上下文指的是图像中的各种关系(例如，对象对对象的同现关系)。在这篇文章中，我们表明，即使是上下文一致性检查对于适当制作的敌意示例也可能是脆弱的，据我们所知，我们是第一个这样做的人。具体地说，我们提出了一个自适应的框架来生成颠覆这种防御的示例，即针对对象检测的逃避上下文一致性检查(ADC)的对抗性攻击。在ADC中，我们构造了一个具有两个攻击目标的联合优化问题，即(I)欺骗对象检测器和(Ii)同时逃避上下文一致性检查系统。在Pascal VOC和MS Coco数据集上的实验表明，ADC生成的示例在大多数情况下欺骗了对象检测器，成功率超过85%，同时避开了最近提出的上下文一致性检查，大多数情况下旁路率超过80%。我们的结果表明，如何健壮地建模上下文并检查其一致性，仍然是一个悬而未决的问题。



## **25. HIRE-SNN: Harnessing the Inherent Robustness of Energy-Efficient Deep Spiking Neural Networks by Training with Crafted Input Noise**

HIRE-SNN：通过使用精心设计的输入噪声进行训练来利用能效深度尖峰神经网络的固有健壮性 cs.CV

10 pages, 11 figures, 7 tables, International Conference on Computer  Vision

**SubmitDate**: 2021-10-06    [paper-pdf](http://arxiv.org/pdf/2110.11417v1)

**Authors**: Souvik Kundu, Massoud Pedram, Peter A. Beerel

**Abstracts**: Low-latency deep spiking neural networks (SNNs) have become a promising alternative to conventional artificial neural networks (ANNs) because of their potential for increased energy efficiency on event-driven neuromorphic hardware. Neural networks, including SNNs, however, are subject to various adversarial attacks and must be trained to remain resilient against such attacks for many applications. Nevertheless, due to prohibitively high training costs associated with SNNs, analysis, and optimization of deep SNNs under various adversarial attacks have been largely overlooked. In this paper, we first present a detailed analysis of the inherent robustness of low-latency SNNs against popular gradient-based attacks, namely fast gradient sign method (FGSM) and projected gradient descent (PGD). Motivated by this analysis, to harness the model robustness against these attacks we present an SNN training algorithm that uses crafted input noise and incurs no additional training time. To evaluate the merits of our algorithm, we conducted extensive experiments with variants of VGG and ResNet on both CIFAR-10 and CIFAR-100 datasets. Compared to standard trained direct input SNNs, our trained models yield improved classification accuracy of up to 13.7% and 10.1% on FGSM and PGD attack-generated images, respectively, with negligible loss in clean image accuracy. Our models also outperform inherently robust SNNs trained on rate-coded inputs with improved or similar classification performance on attack-generated images while having up to 25x and 4.6x lower latency and computation energy, respectively.

摘要: 低延迟深度尖峰神经网络(SNNs)由于其在事件驱动的神经形态硬件上提高能量效率的潜力，已成为传统人工神经网络(ANN)的一种有前途的替代方案。然而，包括SNN在内的神经网络会受到各种对抗性攻击，必须经过训练才能在许多应用中保持对此类攻击的弹性。然而，由于与SNN相关的训练成本高得令人望而却步，在各种对抗性攻击下对深度SNN的分析和优化在很大程度上被忽视了。本文首先详细分析了低延迟SNN对常用的基于梯度的攻击，即快速梯度符号法(FGSM)和投影梯度下降法(PGD)的固有鲁棒性。在这种分析的基础上，为了利用模型对这些攻击的鲁棒性，我们提出了一种SNN训练算法，该算法使用精心制作的输入噪声，并且不会产生额外的训练时间。为了评估我们算法的优点，我们在CIFAR-10和CIFAR-100数据集上用VGG和ResNet的变体进行了广泛的实验。与标准训练的直接输入SNN相比，我们训练的模型对FGSM和PGD攻击生成的图像的分类准确率分别提高了13.7%和10.1%，而清晰图像的准确率几乎可以忽略不计。我们的模型还优于在速率编码输入上训练的固有鲁棒SNN，在攻击生成的图像上的分类性能得到了改善或类似，而延迟和计算能量分别降低了25倍和4.6倍。



