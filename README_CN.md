# Latest Adversarial Attack Papers
**update at 2022-07-19 06:31:22**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Demystifying the Adversarial Robustness of Random Transformation Defenses**

揭开随机变换防御对抗健壮性的神秘面纱 cs.CR

ICML 2022 (short presentation), AAAI 2022 AdvML Workshop (best paper,  oral presentation)

**SubmitDate**: 2022-07-15    [paper-pdf](http://arxiv.org/pdf/2207.03574v2)

**Authors**: Chawin Sitawarin, Zachary Golan-Strieb, David Wagner

**Abstracts**: Neural networks' lack of robustness against attacks raises concerns in security-sensitive settings such as autonomous vehicles. While many countermeasures may look promising, only a few withstand rigorous evaluation. Defenses using random transformations (RT) have shown impressive results, particularly BaRT (Raff et al., 2019) on ImageNet. However, this type of defense has not been rigorously evaluated, leaving its robustness properties poorly understood. Their stochastic properties make evaluation more challenging and render many proposed attacks on deterministic models inapplicable. First, we show that the BPDA attack (Athalye et al., 2018a) used in BaRT's evaluation is ineffective and likely overestimates its robustness. We then attempt to construct the strongest possible RT defense through the informed selection of transformations and Bayesian optimization for tuning their parameters. Furthermore, we create the strongest possible attack to evaluate our RT defense. Our new attack vastly outperforms the baseline, reducing the accuracy by 83% compared to the 19% reduction by the commonly used EoT attack ($4.3\times$ improvement). Our result indicates that the RT defense on the Imagenette dataset (a ten-class subset of ImageNet) is not robust against adversarial examples. Extending the study further, we use our new attack to adversarially train RT defense (called AdvRT), resulting in a large robustness gain. Code is available at https://github.com/wagner-group/demystify-random-transform.

摘要: 神经网络对攻击缺乏稳健性，这在自动驾驶汽车等安全敏感环境中引发了担忧。尽管许多对策看起来很有希望，但只有少数几项经得起严格的评估。使用随机变换(RT)的防御已经显示出令人印象深刻的结果，特别是Bart(Raff等人，2019年)在ImageNet上。然而，这种类型的防御没有得到严格的评估，使得人们对其健壮性属性知之甚少。它们的随机性使评估变得更具挑战性，并使许多已提出的对确定性模型的攻击不适用。首先，我们证明了BART评估中使用的BPDA攻击(Athalye等人，2018a)是无效的，并且可能高估了它的健壮性。然后，我们试图通过对变换的知情选择和贝叶斯优化来调整它们的参数，从而构建尽可能强的RT防御。此外，我们创建了尽可能强的攻击来评估我们的RT防御。我们的新攻击大大超过了基准，与常用的EoT攻击相比，准确率降低了83%($4.3倍$改进)。我们的结果表明，在Imagenette数据集(ImageNet的十类子集)上的RT防御对敌意示例不是健壮的。进一步扩展研究，我们使用我们的新攻击来恶意训练RT防御(称为AdvRT)，从而获得了很大的健壮性收益。代码可在https://github.com/wagner-group/demystify-random-transform.上找到



## **2. CC-Fuzz: Genetic algorithm-based fuzzing for stress testing congestion control algorithms**

CC-Fuzz：基于遗传算法的模糊压力测试拥塞控制算法 cs.NI

This version was submitted to Hotnets 2022

**SubmitDate**: 2022-07-15    [paper-pdf](http://arxiv.org/pdf/2207.07300v1)

**Authors**: Devdeep Ray, Srinivasan Seshan

**Abstracts**: Congestion control research has experienced a significant increase in interest in the past few years, with many purpose-built algorithms being designed with the needs of specific applications in mind. These algorithms undergo limited testing before being deployed on the Internet, where they interact with other congestion control algorithms and run across a variety of network conditions. This often results in unforeseen performance issues in the wild due to algorithmic inadequacies or implementation bugs, and these issues are often hard to identify since packet traces are not available.   In this paper, we present CC-Fuzz, an automated congestion control testing framework that uses a genetic search algorithm in order to stress test congestion control algorithms by generating adversarial network traces and traffic patterns. Initial results using this approach are promising - CC-Fuzz automatically found a bug in BBR that causes it to stall permanently, and is able to automatically discover the well-known low-rate TCP attack, among other things.

摘要: 在过去的几年中，拥塞控制研究的兴趣显著增加，许多专门设计的算法都是考虑到特定应用的需求而设计的。这些算法在部署到Internet之前经过有限的测试，在Internet上它们与其他拥塞控制算法交互，并在各种网络条件下运行。这通常会由于算法不足或实现错误而导致无法预见的性能问题，而且这些问题通常很难识别，因为数据包跟踪不可用。在本文中，我们提出了一个自动化拥塞控制测试框架CC-Fuzz，它使用遗传搜索算法，通过生成敌对的网络轨迹和流量模式来对拥塞控制算法进行压力测试。使用这种方法的初步结果是有希望的-CC-Fuzz自动发现BBR中的一个错误，导致它永久停止，并能够自动发现众所周知的低速率TCP攻击等。



## **3. PASS: Parameters Audit-based Secure and Fair Federated Learning Scheme against Free Rider**

PASS：基于参数审计的反搭便车安全公平联邦学习方案 cs.CR

8 pages, 5 figures, 3 tables

**SubmitDate**: 2022-07-15    [paper-pdf](http://arxiv.org/pdf/2207.07292v1)

**Authors**: Jianhua Wang

**Abstracts**: Federated Learning (FL) as a secure distributed learning frame gains interest in Internet of Things (IoT) due to its capability of protecting private data of participants. However, traditional FL systems are vulnerable to attacks such as Free-Rider (FR) attack, which causes not only unfairness but also privacy leakage and inferior performance to FL systems. The existing defense mechanisms against FR attacks only concern the scenarios where the adversaries declare less than 50% of the total amount of clients. Moreover, they lose effectiveness in resisting selfish FR (SFR) attacks. In this paper, we propose a Parameter Audit-based Secure and fair federated learning Scheme (PASS) against FR attacks. The PASS has the following key features: (a) works well in the scenario where adversaries are more than 50% of the total amount of clients; (b) is effective in countering anonymous FR attacks and SFR attacks; (c) prevents from privacy leakage without accuracy loss. Extensive experimental results verify the data protecting capability in mean square error against privacy leakage and reveal the effectiveness of PASS in terms of a higher defense success rate and lower false positive rate against anonymous SFR attacks. Note in addition, PASS produces no effect on FL accuracy when there is no FR adversary.

摘要: 联邦学习(FL)作为一种安全的分布式学习框架，因其能够保护参与者的隐私数据而受到物联网(IoT)的关注。然而，传统的FL系统容易受到搭便车(FR)攻击等攻击，这不仅会造成不公平，而且还会泄露隐私，降低FL系统的性能。现有的FR攻击防御机制只针对对手申报的客户端总数不到50%的场景。此外，它们在抵抗自私FR(SFR)攻击方面也失去了效力。针对FR攻击，提出了一种基于参数审计的安全公平的联邦学习方案(PASS)。PASS具有以下主要特点：(A)在对手占客户端总数50%以上的场景下工作良好；(B)有效对抗匿名FR攻击和SFR攻击；(C)在不损失准确性的情况下防止隐私泄露。大量的实验结果验证了PASS在均方误差下对隐私泄露的保护能力，揭示了PASS对匿名SFR攻击具有较高的防御成功率和较低的误检率。另外，在没有FR对手的情况下，传球不会影响FL的准确性。



## **4. Lipschitz Bound Analysis of Neural Networks**

神经网络的Lipschitz界分析 cs.LG

5 pages, 7 figures

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.07232v1)

**Authors**: Sarosij Bose

**Abstracts**: Lipschitz Bound Estimation is an effective method of regularizing deep neural networks to make them robust against adversarial attacks. This is useful in a variety of applications ranging from reinforcement learning to autonomous systems. In this paper, we highlight the significant gap in obtaining a non-trivial Lipschitz bound certificate for Convolutional Neural Networks (CNNs) and empirically support it with extensive graphical analysis. We also show that unrolling Convolutional layers or Toeplitz matrices can be employed to convert Convolutional Neural Networks (CNNs) to a Fully Connected Network. Further, we propose a simple algorithm to show the existing 20x-50x gap in a particular data distribution between the actual lipschitz constant and the obtained tight bound. We also ran sets of thorough experiments on various network architectures and benchmark them on datasets like MNIST and CIFAR-10. All these proposals are supported by extensive testing, graphs, histograms and comparative analysis.

摘要: Lipschitz界估计是一种有效的正则化深度神经网络的方法，使其对敌意攻击具有较强的鲁棒性。这在从强化学习到自主系统的各种应用中都很有用。在这篇文章中，我们强调了在获得卷积神经网络(CNN)的非平凡Lipschitz界证书方面的显著差距，并用广泛的图形分析对其进行了经验支持。我们还证明了卷积层展开或Toeplitz矩阵可用于将卷积神经网络(CNN)转换为完全连通网络。此外，我们提出了一个简单的算法来显示在特定数据分布中实际的Lipschitz常数和得到的紧界之间存在的20x-50x的差距。我们还在各种网络体系结构上运行了一组彻底的实验，并在MNIST和CIFAR-10等数据集上对它们进行了基准测试。所有这些建议都得到了广泛的测试、图表、直方图和比较分析的支持。



## **5. Multi-Agent Deep Reinforcement Learning-Driven Mitigation of Adverse Effects of Cyber-Attacks on Electric Vehicle Charging Station**

基于多智能体深度强化学习的电动汽车充电站网络攻击缓解 eess.SY

Submitted to IEEE Transactions on Smart Grids

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.07041v1)

**Authors**: M. Basnet, MH Ali

**Abstracts**: An electric vehicle charging station (EVCS) infrastructure is the backbone of transportation electrification. However, the EVCS has myriads of exploitable vulnerabilities in software, hardware, supply chain, and incumbent legacy technologies such as network, communication, and control. These standalone or networked EVCS open up large attack surfaces for the local or state-funded adversaries. The state-of-the-art approaches are not agile and intelligent enough to defend against and mitigate advanced persistent threats (APT). We propose the data-driven model-free distributed intelligence based on multiagent Deep Reinforcement Learning (MADRL)-- Twin Delayed Deep Deterministic Policy Gradient (TD3) -- that efficiently learns the control policy to mitigate the cyberattacks on the controllers of EVCS. Also, we have proposed two additional mitigation methods: the manual/Bruteforce mitigation and the controller clone-based mitigation. The attack model considers the APT designed to malfunction the duty cycles of the EVCS controllers with Type-I low-frequency attack and Type-II constant attack. The proposed model restores the EVCS operation under threat incidence in any/all controllers by correcting the control signals generated by the legacy controllers. Also, the TD3 algorithm provides higher granularity by learning nonlinear control policies as compared to the other two mitigation methods. Index Terms: Cyberattack, Deep Reinforcement Learning(DRL), Electric Vehicle Charging Station, Mitigation.

摘要: 电动汽车充电站(EVCS)基础设施是交通电气化的支柱。然而，EVCS在软件、硬件、供应链和现有的遗留技术(如网络、通信和控制)中存在无数可利用的漏洞。这些独立或联网的EVCS为当地或国家资助的对手打开了巨大的攻击面。最先进的方法不够灵活和智能，无法防御和缓解高级持续威胁(APT)。提出了一种基于多智能体深度强化学习(MADRL)的无模型数据驱动的分布式智能模型--双延迟深度确定性策略梯度(TD3)，它能有效地学习控制策略以减轻对EVCS控制器的网络攻击。此外，我们还提出了两种额外的缓解方法：手动/暴力缓解和基于控制器克隆的缓解。该攻击模型考虑了APT，该APT被设计为在I型低频攻击和II型恒定攻击下使EVCS控制器的占空比发生故障。该模型通过校正传统控制器产生的控制信号，恢复了任意/所有控制器在威胁发生时的EVCS操作。此外，与其他两种缓解方法相比，TD3算法通过学习非线性控制策略提供了更高的粒度。索引词：网络攻击，深度强化学习，电动汽车充电站，缓解。



## **6. Adversarial Attacks on Monocular Pose Estimation**

针对单目位姿估计的对抗性攻击 cs.CV

Accepted at the 2022 IEEE/RSJ International Conference on Intelligent  Robots and Systems (IROS 2022)

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.07032v1)

**Authors**: Hemang Chawla, Arnav Varma, Elahe Arani, Bahram Zonooz

**Abstracts**: Advances in deep learning have resulted in steady progress in computer vision with improved accuracy on tasks such as object detection and semantic segmentation. Nevertheless, deep neural networks are vulnerable to adversarial attacks, thus presenting a challenge in reliable deployment. Two of the prominent tasks in 3D scene-understanding for robotics and advanced drive assistance systems are monocular depth and pose estimation, often learned together in an unsupervised manner. While studies evaluating the impact of adversarial attacks on monocular depth estimation exist, a systematic demonstration and analysis of adversarial perturbations against pose estimation are lacking. We show how additive imperceptible perturbations can not only change predictions to increase the trajectory drift but also catastrophically alter its geometry. We also study the relation between adversarial perturbations targeting monocular depth and pose estimation networks, as well as the transferability of perturbations to other networks with different architectures and losses. Our experiments show how the generated perturbations lead to notable errors in relative rotation and translation predictions and elucidate vulnerabilities of the networks.

摘要: 深度学习的进步导致了计算机视觉的稳步发展，提高了目标检测和语义分割等任务的准确性。然而，深度神经网络很容易受到敌意攻击，因此在可靠部署方面提出了挑战。在机器人和先进驾驶辅助系统的3D场景理解中，两项突出的任务是单目深度和姿势估计，它们通常是在无人监督的方式下一起学习的。虽然已有研究评估对抗性攻击对单目深度估计的影响，但缺乏针对姿态估计的对抗性扰动的系统论证和分析。我们展示了加性不可察觉的扰动不仅可以改变预测以增加轨迹漂移，而且还可以灾难性地改变其几何形状。我们还研究了针对单目深度的对抗扰动与姿态估计网络之间的关系，以及扰动在具有不同结构和损失的其他网络中的可传递性。我们的实验显示了产生的扰动如何导致相对旋转和平移预测的显著错误，并阐明了网络的脆弱性。



## **7. Susceptibility of Continual Learning Against Adversarial Attacks**

持续学习对敌意攻击的敏感性 cs.LG

18 pages, 13 figures

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.05225v3)

**Authors**: Hikmat Khan, Pir Masoom Shah, Syed Farhan Alam Zaidi, Saif ul Islam

**Abstracts**: The recent advances in continual (incremental or lifelong) learning have concentrated on the prevention of forgetting that can lead to catastrophic consequences, but there are two outstanding challenges that must be addressed. The first is the evaluation of the robustness of the proposed methods. The second is ensuring the security of learned tasks remains largely unexplored. This paper presents a comprehensive study of the susceptibility of the continually learned tasks (including both current and previously learned tasks) that are vulnerable to forgetting. Such vulnerability of tasks against adversarial attacks raises profound issues in data integrity and privacy. We consider all three scenarios (i.e, task-incremental leaning, domain-incremental learning and class-incremental learning) of continual learning and explore three regularization-based experiments, three replay-based experiments, and one hybrid technique based on the reply and exemplar approach. We examine the robustness of these methods. In particular, we consider cases where we demonstrate that any class belonging to the current or previously learned tasks is prone to misclassification. Our observations, we identify potential limitations in continual learning approaches against adversarial attacks. Our empirical study recommends that the research community consider the robustness of the proposed continual learning approaches and invest extensive efforts in mitigating catastrophic forgetting.

摘要: 持续(增量或终身)学习的最新进展集中在防止可能导致灾难性后果的遗忘上，但有两个突出的挑战必须解决。首先是对所提出方法的稳健性进行评估。第二，确保学习任务的安全性在很大程度上仍未得到探索。本文对易被遗忘的持续学习任务(包括当前学习任务和先前学习任务)的易感性进行了全面的研究。任务对对手攻击的这种脆弱性引发了数据完整性和隐私方面的严重问题。我们考虑了持续学习的三种情景(任务增量式学习、领域增量式学习和班级增量式学习)，探索了三种基于正则化的实验、三种基于回放的实验以及一种基于回复和样例的混合技术。我们检验了这些方法的稳健性。特别是，我们考虑了这样的情况，即我们证明属于当前或以前学习的任务的任何类都容易发生错误分类。根据我们的观察，我们确定了针对敌意攻击的持续学习方法的潜在局限性。我们的实证研究建议研究界考虑所提出的持续学习方法的稳健性，并投入广泛的努力来缓解灾难性遗忘。



## **8. Adversarial Examples for Model-Based Control: A Sensitivity Analysis**

基于模型控制的对抗性实例：敏感度分析 eess.SY

Submission to the 58th Annual Allerton Conference on Communication,  Control, and Computing

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.06982v1)

**Authors**: Po-han Li, Ufuk Topcu, Sandeep P. Chinchali

**Abstracts**: We propose a method to attack controllers that rely on external timeseries forecasts as task parameters. An adversary can manipulate the costs, states, and actions of the controllers by forging the timeseries, in this case perturbing the real timeseries. Since the controllers often encode safety requirements or energy limits in their costs and constraints, we refer to such manipulation as an adversarial attack. We show that different attacks on model-based controllers can increase control costs, activate constraints, or even make the control optimization problem infeasible. We use the linear quadratic regulator and convex model predictive controllers as examples of how adversarial attacks succeed and demonstrate the impact of adversarial attacks on a battery storage control task for power grid operators. As a result, our method increases control cost by $8500\%$ and energy constraints by $13\%$ on real electricity demand timeseries.

摘要: 我们提出了一种攻击依赖外部时间序列预测作为任务参数的控制器的方法。对手可以通过伪造时间序列来操纵控制器的成本、状态和操作，在这种情况下，会扰乱真实的时间序列。由于控制器经常在其成本和约束中编码安全要求或能量限制，我们将这种操纵称为对抗性攻击。我们证明了对基于模型的控制器的不同攻击会增加控制成本，激活约束，甚至使控制优化问题变得不可行。我们使用线性二次型调节器和凸模型预测控制器作为对抗性攻击如何成功的例子，并展示了对抗性攻击对电网运营商电池存储控制任务的影响。结果表明，该方法使实际电力需求时间序列的控制成本增加了8500美元，能源约束增加了13美元。



## **9. RSD-GAN: Regularized Sobolev Defense GAN Against Speech-to-Text Adversarial Attacks**

RSD-GAN：正规化Sobolev防御GAN防止语音到文本的对抗性攻击 cs.SD

Paper submitted to IEEE Signal Processing Letters Journal

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.06858v1)

**Authors**: Mohammad Esmaeilpour, Nourhene Chaalia, Patrick Cardinal

**Abstracts**: This paper introduces a new synthesis-based defense algorithm for counteracting with a varieties of adversarial attacks developed for challenging the performance of the cutting-edge speech-to-text transcription systems. Our algorithm implements a Sobolev-based GAN and proposes a novel regularizer for effectively controlling over the functionality of the entire generative model, particularly the discriminator network during training. Our achieved results upon carrying out numerous experiments on the victim DeepSpeech, Kaldi, and Lingvo speech transcription systems corroborate the remarkable performance of our defense approach against a comprehensive range of targeted and non-targeted adversarial attacks.

摘要: 本文介绍了一种新的基于合成的防御算法，用于对抗各种针对尖端语音到文本转录系统性能的挑战而开发的对抗性攻击。我们的算法实现了一种基于Sobolev的GAN，并提出了一种新的正则化算法来有效地控制整个生成模型的功能，特别是在训练过程中的鉴别器网络。我们在受害者DeepSpeech、Kaldi和Lingvo语音转录系统上进行的大量实验所取得的结果证实了我们的防御方法在应对全面的定向和非定向对手攻击方面的卓越表现。



## **10. AGIC: Approximate Gradient Inversion Attack on Federated Learning**

AGIC：联邦学习中的近似梯度反转攻击 cs.LG

This paper is accepted at the 41st International Symposium on  Reliable Distributed Systems (SRDS 2022)

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2204.13784v3)

**Authors**: Jin Xu, Chi Hong, Jiyue Huang, Lydia Y. Chen, Jérémie Decouchant

**Abstracts**: Federated learning is a private-by-design distributed learning paradigm where clients train local models on their own data before a central server aggregates their local updates to compute a global model. Depending on the aggregation method used, the local updates are either the gradients or the weights of local learning models. Recent reconstruction attacks apply a gradient inversion optimization on the gradient update of a single minibatch to reconstruct the private data used by clients during training. As the state-of-the-art reconstruction attacks solely focus on single update, realistic adversarial scenarios are overlooked, such as observation across multiple updates and updates trained from multiple mini-batches. A few studies consider a more challenging adversarial scenario where only model updates based on multiple mini-batches are observable, and resort to computationally expensive simulation to untangle the underlying samples for each local step. In this paper, we propose AGIC, a novel Approximate Gradient Inversion Attack that efficiently and effectively reconstructs images from both model or gradient updates, and across multiple epochs. In a nutshell, AGIC (i) approximates gradient updates of used training samples from model updates to avoid costly simulation procedures, (ii) leverages gradient/model updates collected from multiple epochs, and (iii) assigns increasing weights to layers with respect to the neural network structure for reconstruction quality. We extensively evaluate AGIC on three datasets, CIFAR-10, CIFAR-100 and ImageNet. Our results show that AGIC increases the peak signal-to-noise ratio (PSNR) by up to 50% compared to two representative state-of-the-art gradient inversion attacks. Furthermore, AGIC is faster than the state-of-the-art simulation based attack, e.g., it is 5x faster when attacking FedAvg with 8 local steps in between model updates.

摘要: 联合学习是一种私人设计的分布式学习范例，其中客户端在中央服务器聚合其本地更新以计算全局模型之前，根据自己的数据训练本地模型。根据所使用的聚合方法，局部更新要么是局部学习模型的梯度，要么是局部学习模型的权重。最近的重建攻击将梯度倒置优化应用于单个小批量的梯度更新，以重建客户在训练期间使用的私有数据。由于最新的重建攻击只关注单个更新，因此忽略了现实的对抗性场景，例如跨多个更新的观察和从多个小批次训练的更新。一些研究考虑了一种更具挑战性的对抗性场景，其中只能观察到基于多个小批次的模型更新，并求助于计算代价高昂的模拟来解开每个局部步骤的潜在样本。在本文中，我们提出了AGIC，一种新的近似梯度反转攻击，它可以高效地从模型或梯度更新中重建图像，并跨越多个历元。简而言之，AGIC(I)根据模型更新近似使用的训练样本的梯度更新以避免昂贵的模拟过程，(Ii)利用从多个历元收集的梯度/模型更新，以及(Iii)为重建质量向层分配相对于神经网络结构的不断增加的权重。我们在三个数据集CIFAR-10、CIFAR-100和ImageNet上对AGIC进行了广泛的评估。实验结果表明，与两种典型的梯度反转攻击相比，AGIC的峰值信噪比(PSNR)提高了50%。此外，AGIC比最先进的基于模拟的攻击速度更快，例如，在模型更新之间有8个本地步骤的情况下，攻击FedAvg的速度要快5倍。



## **11. Superclass Adversarial Attack**

超类对抗性攻击 cs.CV

ICML Workshop 2022 on Adversarial Machine Learning Frontiers

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2205.14629v2)

**Authors**: Soichiro Kumano, Hiroshi Kera, Toshihiko Yamasaki

**Abstracts**: Adversarial attacks have only focused on changing the predictions of the classifier, but their danger greatly depends on how the class is mistaken. For example, when an automatic driving system mistakes a Persian cat for a Siamese cat, it is hardly a problem. However, if it mistakes a cat for a 120km/h minimum speed sign, serious problems can arise. As a stepping stone to more threatening adversarial attacks, we consider the superclass adversarial attack, which causes misclassification of not only fine classes, but also superclasses. We conducted the first comprehensive analysis of superclass adversarial attacks (an existing and 19 new methods) in terms of accuracy, speed, and stability, and identified several strategies to achieve better performance. Although this study is aimed at superclass misclassification, the findings can be applied to other problem settings involving multiple classes, such as top-k and multi-label classification attacks.

摘要: 对抗性攻击只专注于改变分类器的预测，但它们的危险在很大程度上取决于类的错误程度。例如，当自动驾驶系统将波斯猫误认为暹罗猫时，这几乎不是问题。然而，如果它把猫错当成120公里/小时的最低速度标志，可能会出现严重的问题。作为更具威胁性的对抗性攻击的垫脚石，我们认为超类对抗性攻击不仅会导致细类的错误分类，而且会导致超类的错误分类。我们首次对超类对抗性攻击(现有方法和19种新方法)在准确性、速度和稳定性方面进行了全面分析，并确定了几种实现更好性能的策略。虽然这项研究是针对超类错误分类的，但研究结果也适用于其他涉及多类的问题，如top-k和多标签分类攻击。



## **12. Adversarially-Aware Robust Object Detector**

对抗性感知的鲁棒目标检测器 cs.CV

ECCV2022 oral paper

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.06202v2)

**Authors**: Ziyi Dong, Pengxu Wei, Liang Lin

**Abstracts**: Object detection, as a fundamental computer vision task, has achieved a remarkable progress with the emergence of deep neural networks. Nevertheless, few works explore the adversarial robustness of object detectors to resist adversarial attacks for practical applications in various real-world scenarios. Detectors have been greatly challenged by unnoticeable perturbation, with sharp performance drop on clean images and extremely poor performance on adversarial images. In this work, we empirically explore the model training for adversarial robustness in object detection, which greatly attributes to the conflict between learning clean images and adversarial images. To mitigate this issue, we propose a Robust Detector (RobustDet) based on adversarially-aware convolution to disentangle gradients for model learning on clean and adversarial images. RobustDet also employs the Adversarial Image Discriminator (AID) and Consistent Features with Reconstruction (CFR) to ensure a reliable robustness. Extensive experiments on PASCAL VOC and MS-COCO demonstrate that our model effectively disentangles gradients and significantly enhances the detection robustness with maintaining the detection ability on clean images.

摘要: 随着深度神经网络的出现，目标检测作为一项基本的计算机视觉任务已经取得了显著的进展。然而，很少有研究探讨对象检测器在各种真实场景中的实际应用中抵抗对手攻击的对抗性健壮性。检测器受到了不可察觉的扰动的极大挑战，在干净图像上的性能急剧下降，在对抗性图像上的性能极差。在这项工作中，我们经验性地探索了目标检测中对抗鲁棒性的模型训练，这在很大程度上归因于学习干净图像和对抗图像之间的冲突。为了缓解这一问题，我们提出了一种基于对抗性感知卷积的稳健检测器(RobustDet)，用于在干净图像和对抗性图像上进行模型学习。RobustDet还采用了对抗性图像鉴别器(AID)和重建一致特征(CFR)，以确保可靠的健壮性。在PASCAL、VOC和MS-COCO上的大量实验表明，该模型在保持对干净图像的检测能力的同时，有效地解开了梯度的纠缠，显著提高了检测的鲁棒性。



## **13. PIAT: Physics Informed Adversarial Training for Solving Partial Differential Equations**

PIAT：解偏微分方程解的物理对抗性训练 cs.LG

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.06647v1)

**Authors**: Simin Shekarpaz, Mohammad Azizmalayeri, Mohammad Hossein Rohban

**Abstracts**: In this paper, we propose the physics informed adversarial training (PIAT) of neural networks for solving nonlinear differential equations (NDE). It is well-known that the standard training of neural networks results in non-smooth functions. Adversarial training (AT) is an established defense mechanism against adversarial attacks, which could also help in making the solution smooth. AT include augmenting the training mini-batch with a perturbation that makes the network output mismatch the desired output adversarially. Unlike formal AT, which relies only on the training data, here we encode the governing physical laws in the form of nonlinear differential equations using automatic differentiation in the adversarial network architecture. We compare PIAT with PINN to indicate the effectiveness of our method in solving NDEs for up to 10 dimensions. Moreover, we propose weight decay and Gaussian smoothing to demonstrate the PIAT advantages. The code repository is available at https://github.com/rohban-lab/PIAT.

摘要: 本文提出了求解非线性微分方程组的神经网络的物理知情对抗训练(PIAT)方法。众所周知，神经网络的标准训练会导致函数的非光滑。对抗性训练(AT)是一种针对对抗性攻击的既定防御机制，它也有助于使解决方案顺利进行。AT包括用使网络输出与期望输出相反地失配的扰动来扩充训练小批量。与仅依赖训练数据的形式AT不同，我们在对抗性网络结构中使用自动微分将支配物理定律编码为非线性微分方程组的形式。我们将PIAT和Pinn进行了比较，以表明我们的方法在求解高达10维的非定常方程方面的有效性。此外，我们提出了权重衰减和高斯平滑来展示PIAT的优势。代码存储库可在https://github.com/rohban-lab/PIAT.上找到



## **14. Exploring Adversarial Attacks and Defenses in Vision Transformers trained with DINO**

探索与恐龙一起训练的视觉变形金刚的对抗性攻击和防御 cs.CV

Workshop paper accepted at AdvML Frontiers (ICML 2022)

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2206.06761v3)

**Authors**: Javier Rando, Nasib Naimi, Thomas Baumann, Max Mathys

**Abstracts**: This work conducts the first analysis on the robustness against adversarial attacks on self-supervised Vision Transformers trained using DINO. First, we evaluate whether features learned through self-supervision are more robust to adversarial attacks than those emerging from supervised learning. Then, we present properties arising for attacks in the latent space. Finally, we evaluate whether three well-known defense strategies can increase adversarial robustness in downstream tasks by only fine-tuning the classification head to provide robustness even in view of limited compute resources. These defense strategies are: Adversarial Training, Ensemble Adversarial Training and Ensemble of Specialized Networks.

摘要: 本文首次对使用Dino训练的自监督视觉转换器的抗敌意攻击能力进行了分析。首先，我们评估通过自我监督学习的特征是否比通过监督学习获得的特征对对手攻击更健壮。然后，我们给出了潜在空间中攻击产生的性质。最后，我们评估了三种著名的防御策略是否能够在下游任务中通过微调分类头来提高对手的健壮性，即使在计算资源有限的情况下也是如此。这些防御策略是：对抗性训练、系列性对抗性训练和专业网络系列化。



## **15. Interactive Machine Learning: A State of the Art Review**

交互式机器学习：最新进展 cs.LG

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.06196v1)

**Authors**: Natnael A. Wondimu, Cédric Buche, Ubbo Visser

**Abstracts**: Machine learning has proved useful in many software disciplines, including computer vision, speech and audio processing, natural language processing, robotics and some other fields. However, its applicability has been significantly hampered due its black-box nature and significant resource consumption. Performance is achieved at the expense of enormous computational resource and usually compromising the robustness and trustworthiness of the model. Recent researches have been identifying a lack of interactivity as the prime source of these machine learning problems. Consequently, interactive machine learning (iML) has acquired increased attention of researchers on account of its human-in-the-loop modality and relatively efficient resource utilization. Thereby, a state-of-the-art review of interactive machine learning plays a vital role in easing the effort toward building human-centred models. In this paper, we provide a comprehensive analysis of the state-of-the-art of iML. We analyze salient research works using merit-oriented and application/task oriented mixed taxonomy. We use a bottom-up clustering approach to generate a taxonomy of iML research works. Research works on adversarial black-box attacks and corresponding iML based defense system, exploratory machine learning, resource constrained learning, and iML performance evaluation are analyzed under their corresponding theme in our merit-oriented taxonomy. We have further classified these research works into technical and sectoral categories. Finally, research opportunities that we believe are inspiring for future work in iML are discussed thoroughly.

摘要: 机器学习已被证明在许多软件学科中都很有用，包括计算机视觉、语音和音频处理、自然语言处理、机器人学和其他一些领域。然而，由于其黑箱性质和巨大的资源消耗，其适用性受到了极大的阻碍。性能的实现是以牺牲巨大的计算资源为代价的，并且通常会损害模型的健壮性和可信性。最近的研究已经确定缺乏互动性是这些机器学习问题的主要来源。因此，交互式机器学习(IML)以其人在环中的方式和相对高效的资源利用而受到越来越多的研究者的关注。因此，对交互式机器学习的最新回顾在减轻建立以人为中心的模型的努力方面发挥着至关重要的作用。在本文中，我们对iML的最新发展进行了全面的分析。我们使用面向价值的和面向应用/任务的混合分类来分析重要的研究作品。我们使用自下而上的聚类方法来生成iML研究作品的分类。在我们的价值导向分类法中，对抗性黑盒攻击和相应的基于iML的防御系统、探索性机器学习、资源受限学习和iML性能评估的研究工作都在相应的主题下进行了分析。我们进一步将这些研究工作分为技术和部门两个类别。最后，深入讨论了我们认为对iML未来工作有启发的研究机会。



## **16. On the Robustness of Bayesian Neural Networks to Adversarial Attacks**

贝叶斯神经网络对敌方攻击的稳健性研究 cs.LG

arXiv admin note: text overlap with arXiv:2002.04359

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.06154v1)

**Authors**: Luca Bortolussi, Ginevra Carbone, Luca Laurenti, Andrea Patane, Guido Sanguinetti, Matthew Wicker

**Abstracts**: Vulnerability to adversarial attacks is one of the principal hurdles to the adoption of deep learning in safety-critical applications. Despite significant efforts, both practical and theoretical, training deep learning models robust to adversarial attacks is still an open problem. In this paper, we analyse the geometry of adversarial attacks in the large-data, overparameterized limit for Bayesian Neural Networks (BNNs). We show that, in the limit, vulnerability to gradient-based attacks arises as a result of degeneracy in the data distribution, i.e., when the data lies on a lower-dimensional submanifold of the ambient space. As a direct consequence, we demonstrate that in this limit BNN posteriors are robust to gradient-based adversarial attacks. Crucially, we prove that the expected gradient of the loss with respect to the BNN posterior distribution is vanishing, even when each neural network sampled from the posterior is vulnerable to gradient-based attacks. Experimental results on the MNIST, Fashion MNIST, and half moons datasets, representing the finite data regime, with BNNs trained with Hamiltonian Monte Carlo and Variational Inference, support this line of arguments, showing that BNNs can display both high accuracy on clean data and robustness to both gradient-based and gradient-free based adversarial attacks.

摘要: 对敌意攻击的脆弱性是在安全关键应用中采用深度学习的主要障碍之一。尽管在实践和理论上都做了大量的努力，但训练对对手攻击稳健的深度学习模型仍然是一个悬而未决的问题。本文分析了贝叶斯神经网络(BNN)在大数据、过参数限制下的攻击几何。我们证明，在极限情况下，由于数据分布的退化，即当数据位于环境空间的低维子流形上时，对基于梯度的攻击的脆弱性出现。作为一个直接的推论，我们证明了在这个极限下，BNN后验网络对基于梯度的敌意攻击是稳健的。重要的是，我们证明了损失相对于BNN后验分布的期望梯度是零的，即使从后验采样的每个神经网络都容易受到基于梯度的攻击。在代表有限数据区的MNIST、Fashion MNIST和半月数据集上的实验结果支持这一论点，BNN采用哈密顿蒙特卡罗和变分推理进行训练，表明BNN在干净数据上具有很高的准确率，并且对基于梯度和基于无梯度的敌意攻击都具有很好的鲁棒性。



## **17. Neural Network Robustness as a Verification Property: A Principled Case Study**

作为验证属性的神经网络健壮性：原则性案例研究 cs.LG

11 pages, CAV 2022

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2104.01396v2)

**Authors**: Marco Casadio, Ekaterina Komendantskaya, Matthew L. Daggitt, Wen Kokke, Guy Katz, Guy Amir, Idan Refaeli

**Abstracts**: Neural networks are very successful at detecting patterns in noisy data, and have become the technology of choice in many fields. However, their usefulness is hampered by their susceptibility to adversarial attacks. Recently, many methods for measuring and improving a network's robustness to adversarial perturbations have been proposed, and this growing body of research has given rise to numerous explicit or implicit notions of robustness. Connections between these notions are often subtle, and a systematic comparison between them is missing in the literature. In this paper we begin addressing this gap, by setting up general principles for the empirical analysis and evaluation of a network's robustness as a mathematical property - during the network's training phase, its verification, and after its deployment. We then apply these principles and conduct a case study that showcases the practical benefits of our general approach.

摘要: 神经网络在检测噪声数据中的模式方面非常成功，已经成为许多领域的首选技术。然而，由于它们容易受到对抗性攻击，它们的有用性受到了阻碍。最近，已经提出了许多方法来衡量和提高网络对敌意干扰的稳健性，并且这一不断增长的研究已经产生了许多显式或隐式的健壮性概念。这些概念之间的联系往往很微妙，文献中也没有对它们进行系统的比较。在本文中，我们开始解决这一差距，通过建立一般原则，将网络的稳健性作为一种数学属性进行经验分析和评估--在网络的训练阶段、验证阶段和部署之后。然后，我们应用这些原则并进行案例研究，展示我们一般方法的实际好处。



## **18. Perturbation Inactivation Based Adversarial Defense for Face Recognition**

基于扰动失活的人脸识别对抗性防御 cs.CV

Accepted by IEEE Transactions on Information Forensics & Security  (T-IFS)

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.06035v1)

**Authors**: Min Ren, Yuhao Zhu, Yunlong Wang, Zhenan Sun

**Abstracts**: Deep learning-based face recognition models are vulnerable to adversarial attacks. To curb these attacks, most defense methods aim to improve the robustness of recognition models against adversarial perturbations. However, the generalization capacities of these methods are quite limited. In practice, they are still vulnerable to unseen adversarial attacks. Deep learning models are fairly robust to general perturbations, such as Gaussian noises. A straightforward approach is to inactivate the adversarial perturbations so that they can be easily handled as general perturbations. In this paper, a plug-and-play adversarial defense method, named perturbation inactivation (PIN), is proposed to inactivate adversarial perturbations for adversarial defense. We discover that the perturbations in different subspaces have different influences on the recognition model. There should be a subspace, called the immune space, in which the perturbations have fewer adverse impacts on the recognition model than in other subspaces. Hence, our method estimates the immune space and inactivates the adversarial perturbations by restricting them to this subspace. The proposed method can be generalized to unseen adversarial perturbations since it does not rely on a specific kind of adversarial attack method. This approach not only outperforms several state-of-the-art adversarial defense methods but also demonstrates a superior generalization capacity through exhaustive experiments. Moreover, the proposed method can be successfully applied to four commercial APIs without additional training, indicating that it can be easily generalized to existing face recognition systems. The source code is available at https://github.com/RenMin1991/Perturbation-Inactivate

摘要: 基于深度学习的人脸识别模型容易受到敌意攻击。为了遏制这些攻击，大多数防御方法的目的是提高识别模型对对手扰动的稳健性。然而，这些方法的泛化能力相当有限。在实践中，他们仍然容易受到看不见的对手攻击。深度学习模型对一般扰动具有较强的鲁棒性，如高斯噪声。一种简单的方法是停用对抗性扰动，这样它们就可以很容易地作为一般扰动来处理。本文提出了一种即插即用的对抗防御方法，称为扰动失活(PIN)，用于灭活对抗防御中的对抗扰动。我们发现，不同子空间中的扰动对识别模型有不同的影响。应该有一个称为免疫空间的子空间，在这个子空间中，扰动对识别模型的不利影响比在其他子空间中要小。因此，我们的方法估计免疫空间，并通过将对抗性扰动限制在此子空间来使其失活。由于该方法不依赖于一种特定的对抗性攻击方法，因此可以推广到不可见的对抗性扰动。通过详尽的实验证明，该方法不仅比目前最先进的对抗性防御方法有更好的性能，而且具有更好的泛化能力。此外，该方法可以成功地应用于四个商业API，而无需额外的训练，这表明它可以很容易地推广到现有的人脸识别系统中。源代码可在https://github.com/RenMin1991/Perturbation-Inactivate上找到



## **19. BadHash: Invisible Backdoor Attacks against Deep Hashing with Clean Label**

BadHash：使用Clean Label对深度哈希进行隐形后门攻击 cs.CV

This paper has been accepted by the 30th ACM International Conference  on Multimedia (MM '22, October 10--14, 2022, Lisboa, Portugal)

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.00278v3)

**Authors**: Shengshan Hu, Ziqi Zhou, Yechao Zhang, Leo Yu Zhang, Yifeng Zheng, Yuanyuan HE, Hai Jin

**Abstracts**: Due to its powerful feature learning capability and high efficiency, deep hashing has achieved great success in large-scale image retrieval. Meanwhile, extensive works have demonstrated that deep neural networks (DNNs) are susceptible to adversarial examples, and exploring adversarial attack against deep hashing has attracted many research efforts. Nevertheless, backdoor attack, another famous threat to DNNs, has not been studied for deep hashing yet. Although various backdoor attacks have been proposed in the field of image classification, existing approaches failed to realize a truly imperceptive backdoor attack that enjoys invisible triggers and clean label setting simultaneously, and they also cannot meet the intrinsic demand of image retrieval backdoor. In this paper, we propose BadHash, the first generative-based imperceptible backdoor attack against deep hashing, which can effectively generate invisible and input-specific poisoned images with clean label. Specifically, we first propose a new conditional generative adversarial network (cGAN) pipeline to effectively generate poisoned samples. For any given benign image, it seeks to generate a natural-looking poisoned counterpart with a unique invisible trigger. In order to improve the attack effectiveness, we introduce a label-based contrastive learning network LabCLN to exploit the semantic characteristics of different labels, which are subsequently used for confusing and misleading the target model to learn the embedded trigger. We finally explore the mechanism of backdoor attacks on image retrieval in the hash space. Extensive experiments on multiple benchmark datasets verify that BadHash can generate imperceptible poisoned samples with strong attack ability and transferability over state-of-the-art deep hashing schemes.

摘要: 深度哈希法由于其强大的特征学习能力和高效的检索效率，在大规模图像检索中取得了巨大的成功。同时，大量的研究表明，深度神经网络(DNN)容易受到敌意例子的影响，探索针对深度散列的敌意攻击吸引了许多研究努力。然而，DNNS的另一个著名威胁--后门攻击，还没有被研究过深度散列。虽然在图像分类领域已经提出了各种各样的后门攻击，但现有的方法未能实现真正的隐蔽的、同时具有不可见触发器和干净标签设置的后门攻击，也不能满足图像检索的内在需求。本文提出了BadHash，这是第一个基于生成性的针对深度哈希的不可察觉的后门攻击，它可以有效地生成标签清晰的不可见和输入特定的有毒图像。具体地说，我们首先提出了一种新的条件生成对抗网络(CGAN)管道来有效地生成有毒样本。对于任何给定的良性形象，它都试图生成一个看起来自然、有毒的形象，并带有独特的无形触发器。为了提高攻击的有效性，我们引入了一个基于标签的对比学习网络LabCLN来利用不同标签的语义特征，这些语义特征被用来混淆和误导目标模型学习嵌入的触发器。最后，我们探讨了哈希空间中后门攻击对图像检索的影响机制。在多个基准数据集上的大量实验证明，BadHash可以生成不可察觉的有毒样本，具有很强的攻击能力和可转移性，优于最先进的深度哈希方案。



## **20. Physical Backdoor Attacks to Lane Detection Systems in Autonomous Driving**

自动驾驶中车道检测系统的物理后门攻击 cs.CV

Accepted by ACM MultiMedia 2022

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2203.00858v2)

**Authors**: Xingshuo Han, Guowen Xu, Yuan Zhou, Xuehuan Yang, Jiwei Li, Tianwei Zhang

**Abstracts**: Modern autonomous vehicles adopt state-of-the-art DNN models to interpret the sensor data and perceive the environment. However, DNN models are vulnerable to different types of adversarial attacks, which pose significant risks to the security and safety of the vehicles and passengers. One prominent threat is the backdoor attack, where the adversary can compromise the DNN model by poisoning the training samples. Although lots of effort has been devoted to the investigation of the backdoor attack to conventional computer vision tasks, its practicality and applicability to the autonomous driving scenario is rarely explored, especially in the physical world.   In this paper, we target the lane detection system, which is an indispensable module for many autonomous driving tasks, e.g., navigation, lane switching. We design and realize the first physical backdoor attacks to such system. Our attacks are comprehensively effective against different types of lane detection algorithms. Specifically, we introduce two attack methodologies (poison-annotation and clean-annotation) to generate poisoned samples. With those samples, the trained lane detection model will be infected with the backdoor, and can be activated by common objects (e.g., traffic cones) to make wrong detections, leading the vehicle to drive off the road or onto the opposite lane. Extensive evaluations on public datasets and physical autonomous vehicles demonstrate that our backdoor attacks are effective, stealthy and robust against various defense solutions. Our codes and experimental videos can be found in https://sites.google.com/view/lane-detection-attack/lda.

摘要: 现代自动驾驶汽车采用最先进的DNN模型来解释传感器数据和感知环境。然而，DNN模型容易受到不同类型的对抗性攻击，这对车辆和乘客的安全和安全构成了重大风险。一个突出的威胁是后门攻击，在这种攻击中，对手可以通过毒化训练样本来危害DNN模型。虽然对传统计算机视觉任务的后门攻击已经投入了大量的精力来研究，但很少有人探索它在自动驾驶场景中的实用性和适用性，特别是在物理世界中。本文以车道检测系统为研究对象，车道检测系统是导航、车道切换等自动驾驶任务中不可缺少的模块。我们设计并实现了对此类系统的第一次物理后门攻击。我们的攻击对不同类型的车道检测算法都是全面有效的。具体地说，我们引入了两种攻击方法(毒注解和干净注解)来生成中毒样本。利用这些样本，训练后的车道检测模型将被后门感染，并可能被常见对象(如交通锥体)激活以进行错误检测，导致车辆驶离道路或进入对面车道。对公共数据集和物理自动驾驶车辆的广泛评估表明，我们的后门攻击针对各种防御解决方案是有效的、隐蔽的和健壮的。我们的代码和实验视频可在https://sites.google.com/view/lane-detection-attack/lda.中找到



## **21. PatchZero: Defending against Adversarial Patch Attacks by Detecting and Zeroing the Patch**

PatchZero：通过检测和归零补丁来防御敌意补丁攻击 cs.CV

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.01795v2)

**Authors**: Ke Xu, Yao Xiao, Zhaoheng Zheng, Kaijie Cai, Ram Nevatia

**Abstracts**: Adversarial patch attacks mislead neural networks by injecting adversarial pixels within a local region. Patch attacks can be highly effective in a variety of tasks and physically realizable via attachment (e.g. a sticker) to the real-world objects. Despite the diversity in attack patterns, adversarial patches tend to be highly textured and different in appearance from natural images. We exploit this property and present PatchZero, a general defense pipeline against white-box adversarial patches without retraining the downstream classifier or detector. Specifically, our defense detects adversaries at the pixel-level and "zeros out" the patch region by repainting with mean pixel values. We further design a two-stage adversarial training scheme to defend against the stronger adaptive attacks. PatchZero achieves SOTA defense performance on the image classification (ImageNet, RESISC45), object detection (PASCAL VOC), and video classification (UCF101) tasks with little degradation in benign performance. In addition, PatchZero transfers to different patch shapes and attack types.

摘要: 对抗性补丁攻击通过在局部区域内注入对抗性像素来误导神经网络。补丁攻击可以在各种任务中非常有效，并且可以通过附着(例如贴纸)到真实世界的对象来物理实现。尽管攻击模式多种多样，但敌方补丁往往纹理丰富，外观与自然图像不同。我们利用这一特性，提出了PatchZero，一种针对白盒恶意补丁的通用防御管道，而不需要重新训练下游的分类器或检测器。具体地说，我们的防御在像素级检测对手，并通过使用平均像素值重新绘制来对补丁区域进行“清零”。我们进一步设计了一种两阶段对抗性训练方案，以抵御更强的适应性攻击。PatchZero在图像分类(ImageNet，RESISC45)、目标检测(Pascal VOC)和视频分类(UCF101)任务上实现了SOTA防御性能，性能良好，性能几乎没有下降。此外，PatchZero还可以转换为不同的补丁形状和攻击类型。



## **22. Game of Trojans: A Submodular Byzantine Approach**

特洛伊木马游戏：一种拜占庭式的子模块方法 cs.LG

Submitted to GameSec 2022

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.05937v1)

**Authors**: Dinuka Sahabandu, Arezoo Rajabi, Luyao Niu, Bo Li, Bhaskar Ramasubramanian, Radha Poovendran

**Abstracts**: Machine learning models in the wild have been shown to be vulnerable to Trojan attacks during training. Although many detection mechanisms have been proposed, strong adaptive attackers have been shown to be effective against them. In this paper, we aim to answer the questions considering an intelligent and adaptive adversary: (i) What is the minimal amount of instances required to be Trojaned by a strong attacker? and (ii) Is it possible for such an attacker to bypass strong detection mechanisms?   We provide an analytical characterization of adversarial capability and strategic interactions between the adversary and detection mechanism that take place in such models. We characterize adversary capability in terms of the fraction of the input dataset that can be embedded with a Trojan trigger. We show that the loss function has a submodular structure, which leads to the design of computationally efficient algorithms to determine this fraction with provable bounds on optimality. We propose a Submodular Trojan algorithm to determine the minimal fraction of samples to inject a Trojan trigger. To evade detection of the Trojaned model, we model strategic interactions between the adversary and Trojan detection mechanism as a two-player game. We show that the adversary wins the game with probability one, thus bypassing detection. We establish this by proving that output probability distributions of a Trojan model and a clean model are identical when following the Min-Max (MM) Trojan algorithm.   We perform extensive evaluations of our algorithms on MNIST, CIFAR-10, and EuroSAT datasets. The results show that (i) with Submodular Trojan algorithm, the adversary needs to embed a Trojan trigger into a very small fraction of samples to achieve high accuracy on both Trojan and clean samples, and (ii) the MM Trojan algorithm yields a trained Trojan model that evades detection with probability 1.

摘要: 野外的机器学习模型已被证明在训练期间容易受到特洛伊木马的攻击。虽然已经提出了许多检测机制，但强自适应攻击者被证明对它们是有效的。在本文中，我们的目标是考虑一个智能和自适应的对手来回答以下问题：(I)强攻击者需要木马的最小实例数量是多少？以及(Ii)这样的攻击者是否有可能绕过强大的检测机制？我们提供了发生在这样的模型中的对手能力和对手与检测机制之间的战略交互的分析特征。我们根据可以嵌入特洛伊木马触发器的输入数据集的比例来表征攻击者的能力。我们证明了损失函数具有子模结构，这导致设计了计算效率高的算法来确定具有可证明的最优界的分数。我们提出了一种子模块木马算法来确定注入木马触发器的最小样本比例。为了逃避木马模型的检测，我们将对手和木马检测机制之间的战略交互建模为两人博弈。我们证明了对手以概率1赢得比赛，从而绕过了检测。我们证明了当遵循Min-Max(MM)木马算法时，木马模型和CLEAN模型的输出概率分布是相同的。我们在MNIST、CIFAR-10和EuroSAT数据集上对我们的算法进行了广泛的评估。结果表明：(I)利用子模块木马算法，攻击者需要在很小一部分样本中嵌入木马触发器，以获得对木马和干净样本的高精度；(Ii)MM木马算法生成一个以1的概率逃避检测的训练有素的木马模型。



## **23. A Word is Worth A Thousand Dollars: Adversarial Attack on Tweets Fools Stock Predictions**

一句话抵得上一千美元：对推特傻瓜股预测的敌意攻击 cs.CR

NAACL short paper, github: https://github.com/yonxie/AdvFinTweet

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2205.01094v3)

**Authors**: Yong Xie, Dakuo Wang, Pin-Yu Chen, Jinjun Xiong, Sijia Liu, Sanmi Koyejo

**Abstracts**: More and more investors and machine learning models rely on social media (e.g., Twitter and Reddit) to gather real-time information and sentiment to predict stock price movements. Although text-based models are known to be vulnerable to adversarial attacks, whether stock prediction models have similar vulnerability is underexplored. In this paper, we experiment with a variety of adversarial attack configurations to fool three stock prediction victim models. We address the task of adversarial generation by solving combinatorial optimization problems with semantics and budget constraints. Our results show that the proposed attack method can achieve consistent success rates and cause significant monetary loss in trading simulation by simply concatenating a perturbed but semantically similar tweet.

摘要: 越来越多的投资者和机器学习模型依赖社交媒体(如Twitter和Reddit)来收集实时信息和情绪，以预测股价走势。尽管众所周知，基于文本的模型容易受到对手攻击，但股票预测模型是否也有类似的脆弱性，还没有得到充分的探讨。在本文中，我们实验了各种对抗性攻击配置，以愚弄三个股票预测受害者模型。我们通过求解具有语义和预算约束的组合优化问题来解决对抗性生成问题。我们的结果表明，该攻击方法可以获得一致的成功率，并在交易模拟中通过简单地连接一条扰动但语义相似的推文来造成巨大的金钱损失。



## **24. Practical Attacks on Machine Learning: A Case Study on Adversarial Windows Malware**

对机器学习的实用攻击：恶意Windows恶意软件的案例研究 cs.CR

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05548v1)

**Authors**: Luca Demetrio, Battista Biggio, Fabio Roli

**Abstracts**: While machine learning is vulnerable to adversarial examples, it still lacks systematic procedures and tools for evaluating its security in different application contexts. In this article, we discuss how to develop automated and scalable security evaluations of machine learning using practical attacks, reporting a use case on Windows malware detection.

摘要: 虽然机器学习很容易受到敌意例子的影响，但它仍然缺乏系统的程序和工具来评估其在不同应用环境中的安全性。在本文中，我们讨论了如何使用实际攻击来开发自动化和可扩展的机器学习安全评估，并报告了一个Windows恶意软件检测的用例。



## **25. Improving the Robustness and Generalization of Deep Neural Network with Confidence Threshold Reduction**

降低置信度阈值提高深度神经网络的鲁棒性和泛化能力 cs.LG

Under review

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2206.00913v2)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstracts**: Deep neural networks are easily attacked by imperceptible perturbation. Presently, adversarial training (AT) is the most effective method to enhance the robustness of the model against adversarial examples. However, because adversarial training solved a min-max value problem, in comparison with natural training, the robustness and generalization are contradictory, i.e., the robustness improvement of the model will decrease the generalization of the model. To address this issue, in this paper, a new concept, namely confidence threshold (CT), is introduced and the reducing of the confidence threshold, known as confidence threshold reduction (CTR), is proven to improve both the generalization and robustness of the model. Specifically, to reduce the CT for natural training (i.e., for natural training with CTR), we propose a mask-guided divergence loss function (MDL) consisting of a cross-entropy loss term and an orthogonal term. The empirical and theoretical analysis demonstrates that the MDL loss improves the robustness and generalization of the model simultaneously for natural training. However, the model robustness improvement of natural training with CTR is not comparable to that of adversarial training. Therefore, for adversarial training, we propose a standard deviation loss function (STD), which minimizes the difference in the probabilities of the wrong categories, to reduce the CT by being integrated into the loss function of adversarial training. The empirical and theoretical analysis demonstrates that the STD based loss function can further improve the robustness of the adversarially trained model on basis of guaranteeing the changeless or slight improvement of the natural accuracy.

摘要: 深层神经网络很容易受到不可察觉的扰动的攻击。目前，对抗性训练(AT)是提高模型对对抗性例子的稳健性的最有效方法。然而，由于对抗性训练解决了最小-最大值问题，与自然训练相比，鲁棒性和泛化是矛盾的，即模型的稳健性提高会降低模型的泛化能力。针对这一问题，本文引入了置信度阈值的概念，并证明了置信度阈值的降低既能提高模型的泛化能力，又能提高模型的鲁棒性。具体地说，为了减少自然训练(即具有CTR的自然训练)的CT，我们提出了一种掩模引导的发散损失函数(MDL)，该函数由交叉熵损失项和正交项组成。实验和理论分析表明，对于自然训练，MDL损失同时提高了模型的鲁棒性和泛化能力。然而，CTR自然训练对模型稳健性的改善与对抗性训练不可同日而语。因此，对于对抗性训练，我们提出了一种标准偏差损失函数(STD)，它最小化了错误类别概率的差异，通过将其整合到对抗性训练的损失函数中来降低CT。实证和理论分析表明，基于STD的损失函数可以在保证自然精度不变或略有提高的基础上，进一步提高对抗性训练模型的稳健性。



## **26. Adversarial Robustness Assessment of NeuroEvolution Approaches**

神经进化方法的对抗性稳健性评估 cs.NE

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05451v1)

**Authors**: Inês Valentim, Nuno Lourenço, Nuno Antunes

**Abstracts**: NeuroEvolution automates the generation of Artificial Neural Networks through the application of techniques from Evolutionary Computation. The main goal of these approaches is to build models that maximize predictive performance, sometimes with an additional objective of minimizing computational complexity. Although the evolved models achieve competitive results performance-wise, their robustness to adversarial examples, which becomes a concern in security-critical scenarios, has received limited attention. In this paper, we evaluate the adversarial robustness of models found by two prominent NeuroEvolution approaches on the CIFAR-10 image classification task: DENSER and NSGA-Net. Since the models are publicly available, we consider white-box untargeted attacks, where the perturbations are bounded by either the L2 or the Linfinity-norm. Similarly to manually-designed networks, our results show that when the evolved models are attacked with iterative methods, their accuracy usually drops to, or close to, zero under both distance metrics. The DENSER model is an exception to this trend, showing some resistance under the L2 threat model, where its accuracy only drops from 93.70% to 18.10% even with iterative attacks. Additionally, we analyzed the impact of pre-processing applied to the data before the first layer of the network. Our observations suggest that some of these techniques can exacerbate the perturbations added to the original inputs, potentially harming robustness. Thus, this choice should not be neglected when automatically designing networks for applications where adversarial attacks are prone to occur.

摘要: 神经进化通过应用进化计算的技术自动生成人工神经网络。这些方法的主要目标是构建最大限度提高预测性能的模型，有时还有最小化计算复杂性的额外目标。虽然进化模型在性能方面达到了竞争的结果，但它们对敌意示例的健壮性受到了有限的关注，这在安全关键场景中成为一个令人担忧的问题。在本文中，我们评估了两种重要的神经进化方法在CIFAR-10图像分类任务中发现的模型的对抗性健壮性：Denser和NSGA-Net。由于模型是公开可用的，我们考虑白盒非目标攻击，其中扰动由L2或Linfinity范数有界。与人工设计的网络类似，我们的结果表明，当进化模型受到迭代方法的攻击时，在两种距离度量下，它们的精度通常下降到或接近于零。密度更高的模型是这一趋势的一个例外，在L2威胁模型下显示出一些阻力，即使在迭代攻击的情况下，其准确率也只从93.70%下降到18.10%。此外，我们还分析了在网络第一层之前应用预处理对数据的影响。我们的观察表明，其中一些技术可能会加剧添加到原始输入的扰动，潜在地损害稳健性。因此，在为容易发生对抗性攻击的应用程序自动设计网络时，这一选择不应被忽视。



## **27. A Security-aware and LUT-based CAD Flow for the Physical Synthesis of eASICs**

一种安全感知的基于查找表的eASIC物理综合CAD流程 cs.CR

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05413v1)

**Authors**: Zain UlAbideen, Tiago Diadami Perez, Mayler Martins, Samuel Pagliarini

**Abstracts**: Numerous threats are associated with the globalized integrated circuit (IC) supply chain, such as piracy, reverse engineering, overproduction, and malicious logic insertion. Many obfuscation approaches have been proposed to mitigate these threats by preventing an adversary from fully understanding the IC (or parts of it). The use of reconfigurable elements inside an IC is a known obfuscation technique, either as a coarse grain reconfigurable block (i.e., eFPGA) or as a fine grain element (i.e., FPGA-like look-up tables). This paper presents a security-aware CAD flow that is LUT-based yet still compatible with the standard cell based physical synthesis flow. More precisely, our CAD flow explores the FPGA-ASIC design space and produces heavily obfuscated designs where only small portions of the logic resemble an ASIC. Therefore, we term this specialized solution an "embedded ASIC" (eASIC). Nevertheless, even for heavily LUT-dominated designs, our proposed decomposition and pin swapping algorithms allow for performance gains that enable performance levels that only ASICs would otherwise achieve. On the security side, we have developed novel template-based attacks and also applied existing attacks, both oracle-free and oracle-based. Our security analysis revealed that the obfuscation rate for an SHA-256 study case should be at least 45% for withstanding traditional attacks and at least 80% for withstanding template-based attacks. When the 80\% obfuscated SHA-256 design is physically implemented, it achieves a remarkable frequency of 368MHz in a 65nm commercial technology, whereas its FPGA implementation (in a superior technology) achieves only 77MHz.

摘要: 与全球化集成电路(IC)供应链相关的许多威胁，如盗版、逆向工程、生产过剩和恶意逻辑插入。已经提出了许多模糊方法来通过阻止对手完全理解IC(或其部分)来缓解这些威胁。在IC内使用可重构元件是一种已知的混淆技术，或者作为粗粒度可重构块(即，eFPGA)，或者作为细粒度元件(即，类似于FPGA的查找表)。本文提出了一种安全感知的CAD流程，该流程是基于LUT的，但仍然与基于标准单元的物理综合流程兼容。更准确地说，我们的CAD流程探索了FPGA-ASIC设计空间，并产生了高度混淆的设计，其中只有一小部分逻辑类似于ASIC。因此，我们将这种专门的解决方案称为“嵌入式ASIC”(EASIC)。然而，即使对于以查找表为主的设计，我们建议的分解和管脚交换算法也可以实现性能提升，从而实现只有ASIC才能达到的性能水平。在安全方面，我们开发了新的基于模板的攻击，并应用了现有的攻击，包括无Oracle攻击和基于Oracle的攻击。我们的安全分析显示，对于SHA-256研究案例，对于抵抗传统攻击至少应该是45%，对于基于模板的攻击至少应该是80%。在实际实现80型混淆SHA-256设计时，它在65 nm的商用工艺下达到了368 MHz的显著频率，而它的FPGA实现(在更高的工艺下)只达到了77 MHz。



## **28. Frequency Domain Model Augmentation for Adversarial Attack**

对抗性攻击的频域模型增强 cs.CV

Accepted by ECCV 2022

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05382v1)

**Authors**: Yuyang Long, Qilong Zhang, Boheng Zeng, Lianli Gao, Xianglong Liu, Jian Zhang, Jingkuan Song

**Abstracts**: For black-box attacks, the gap between the substitute model and the victim model is usually large, which manifests as a weak attack performance. Motivated by the observation that the transferability of adversarial examples can be improved by attacking diverse models simultaneously, model augmentation methods which simulate different models by using transformed images are proposed. However, existing transformations for spatial domain do not translate to significantly diverse augmented models. To tackle this issue, we propose a novel spectrum simulation attack to craft more transferable adversarial examples against both normally trained and defense models. Specifically, we apply a spectrum transformation to the input and thus perform the model augmentation in the frequency domain. We theoretically prove that the transformation derived from frequency domain leads to a diverse spectrum saliency map, an indicator we proposed to reflect the diversity of substitute models. Notably, our method can be generally combined with existing attacks. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our method, \textit{e.g.}, attacking nine state-of-the-art defense models with an average success rate of \textbf{95.4\%}. Our code is available in \url{https://github.com/yuyang-long/SSA}.

摘要: 对于黑盒攻击，替换模型与受害者模型之间的差距通常较大，表现为攻击性能较弱。基于同时攻击不同模型可以提高对抗性实例的可转移性这一观察结果，提出了利用变换后的图像模拟不同模型的模型增强方法。然而，现有的空间域变换并不能转化为显著不同的增强模型。为了解决这个问题，我们提出了一种新颖的频谱模拟攻击，以针对正常训练的模型和防御模型创建更多可转移的对抗性示例。具体地说，我们对输入应用频谱变换，从而在频域中执行模型增强。我们从理论上证明了从频域得到的变换导致了不同的频谱显著图，这是我们提出的反映替代模型多样性的一个指标。值得注意的是，我们的方法通常可以与现有攻击相结合。在ImageNet数据集上的大量实验证明了该方法的有效性，该方法攻击了9个最先进的防御模型，平均成功率为\textbf{95.4\}。我们的代码位于\url{https://github.com/yuyang-long/SSA}.



## **29. Bi-fidelity Evolutionary Multiobjective Search for Adversarially Robust Deep Neural Architectures**

双保真进化多目标搜索逆鲁棒深神经网络结构 cs.LG

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05321v1)

**Authors**: Jia Liu, Ran Cheng, Yaochu Jin

**Abstracts**: Deep neural networks have been found vulnerable to adversarial attacks, thus raising potentially concerns in security-sensitive contexts. To address this problem, recent research has investigated the adversarial robustness of deep neural networks from the architectural point of view. However, searching for architectures of deep neural networks is computationally expensive, particularly when coupled with adversarial training process. To meet the above challenge, this paper proposes a bi-fidelity multiobjective neural architecture search approach. First, we formulate the NAS problem for enhancing adversarial robustness of deep neural networks into a multiobjective optimization problem. Specifically, in addition to a low-fidelity performance predictor as the first objective, we leverage an auxiliary-objective -- the value of which is the output of a surrogate model trained with high-fidelity evaluations. Secondly, we reduce the computational cost by combining three performance estimation methods, i.e., parameter sharing, low-fidelity evaluation, and surrogate-based predictor. The effectiveness of the proposed approach is confirmed by extensive experiments conducted on CIFAR-10, CIFAR-100 and SVHN datasets.

摘要: 深度神经网络被发现容易受到敌意攻击，因此在安全敏感的环境中引发了潜在的担忧。为了解决这个问题，最近的研究从体系结构的角度研究了深度神经网络的对抗健壮性。然而，寻找深度神经网络的结构在计算上是昂贵的，特别是当与对抗性训练过程相结合时。为了应对上述挑战，本文提出了一种双保真多目标神经结构搜索方法。首先，我们将增强深层神经网络对抗健壮性的NAS问题转化为一个多目标优化问题。具体地说，除了作为第一个目标的低保真性能预测器之外，我们还利用一个辅助目标--其值是用高保真评估训练的代理模型的输出。其次，通过结合参数共享、低保真评估和基于代理的预测器三种性能估计方法来降低计算代价。在CIFAR-10、CIFAR-100和SVHN数据集上进行的大量实验证实了该方法的有效性。



## **30. Multitask Learning from Augmented Auxiliary Data for Improving Speech Emotion Recognition**

基于增强辅助数据的多任务学习改进语音情感识别 cs.SD

Under review IEEE Transactions on Affective Computing

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05298v1)

**Authors**: Siddique Latif, Rajib Rana, Sara Khalifa, Raja Jurdak, Björn W. Schuller

**Abstracts**: Despite the recent progress in speech emotion recognition (SER), state-of-the-art systems lack generalisation across different conditions. A key underlying reason for poor generalisation is the scarcity of emotion datasets, which is a significant roadblock to designing robust machine learning (ML) models. Recent works in SER focus on utilising multitask learning (MTL) methods to improve generalisation by learning shared representations. However, most of these studies propose MTL solutions with the requirement of meta labels for auxiliary tasks, which limits the training of SER systems. This paper proposes an MTL framework (MTL-AUG) that learns generalised representations from augmented data. We utilise augmentation-type classification and unsupervised reconstruction as auxiliary tasks, which allow training SER systems on augmented data without requiring any meta labels for auxiliary tasks. The semi-supervised nature of MTL-AUG allows for the exploitation of the abundant unlabelled data to further boost the performance of SER. We comprehensively evaluate the proposed framework in the following settings: (1) within corpus, (2) cross-corpus and cross-language, (3) noisy speech, (4) and adversarial attacks. Our evaluations using the widely used IEMOCAP, MSP-IMPROV, and EMODB datasets show improved results compared to existing state-of-the-art methods.

摘要: 尽管最近在语音情感识别(SER)方面取得了进展，但最先进的系统缺乏对不同条件的通用性。泛化能力差的一个关键根本原因是情感数据集的稀缺，这是设计健壮的机器学习(ML)模型的一个重要障碍。SER最近的工作集中在利用多任务学习(MTL)方法通过学习共享表示来提高泛化能力。然而，这些研究大多提出了辅助任务需要元标签的MTL解决方案，这限制了SER系统的训练。提出了一个从扩充数据中学习泛化表示的MTL框架(MTL-AUG)。我们使用增强型分类和无监督重建作为辅助任务，允许在增强型数据上训练SER系统，而不需要任何辅助任务的元标签。MTL-AUG的半监督性质允许利用丰富的未标记数据来进一步提高SER的性能。我们在以下几个方面对该框架进行了综合评估：(1)在语料库内，(2)跨语料库和跨语言，(3)噪声语音，(4)和对抗性攻击。我们使用广泛使用的IEMOCAP、MSP-Improv和EMODB数据集进行的评估显示，与现有最先进的方法相比，结果有所改善。



## **31. "Why do so?" -- A Practical Perspective on Machine Learning Security**

“为什么要这样做？”--机器学习安全的实践视角 cs.LG

under submission - 18 pages, 3 tables and 4 figures. Long version of  the paper accepted at: New Frontiers of Adversarial Machine Learning@ICML

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.05164v1)

**Authors**: Kathrin Grosse, Lukas Bieringer, Tarek Richard Besold, Battista Biggio, Katharina Krombholz

**Abstracts**: Despite the large body of academic work on machine learning security, little is known about the occurrence of attacks on machine learning systems in the wild. In this paper, we report on a quantitative study with 139 industrial practitioners. We analyze attack occurrence and concern and evaluate statistical hypotheses on factors influencing threat perception and exposure. Our results shed light on real-world attacks on deployed machine learning. On the organizational level, while we find no predictors for threat exposure in our sample, the amount of implement defenses depends on exposure to threats or expected likelihood to become a target. We also provide a detailed analysis of practitioners' replies on the relevance of individual machine learning attacks, unveiling complex concerns like unreliable decision making, business information leakage, and bias introduction into models. Finally, we find that on the individual level, prior knowledge about machine learning security influences threat perception. Our work paves the way for more research about adversarial machine learning in practice, but yields also insights for regulation and auditing.

摘要: 尽管有大量关于机器学习安全的学术工作，但人们对野外发生的针对机器学习系统的攻击知之甚少。在本文中，我们报告了一项对139名工业从业者的定量研究。我们分析了攻击的发生和关注，并对影响威胁感知和暴露的因素进行了统计假设评估。我们的结果揭示了对部署的机器学习的真实世界攻击。在组织层面上，虽然我们在样本中没有发现威胁暴露的预测因素，但实施防御的数量取决于威胁暴露或成为目标的预期可能性。我们还提供了对从业者对单个机器学习攻击相关性的回复的详细分析，揭示了不可靠的决策、商业信息泄露和模型中的偏见引入等复杂问题。最后，我们发现在个体层面上，关于机器学习安全的先验知识会影响威胁感知。我们的工作为在实践中对对抗性机器学习进行更多的研究铺平了道路，但也为监管和审计提供了见解。



## **32. LQG Reference Tracking with Safety and Reachability Guarantees under Unknown False Data Injection Attacks**

未知虚假数据注入攻击下具有安全性和可达性保证的LQG参考跟踪 eess.SY

13 pages, 4 figures, extended version of a Transactions on Automatic  Control paper

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2103.00387v2)

**Authors**: Zhouchi Li, Luyao Niu, Andrew Clark

**Abstracts**: We investigate a linear quadratic Gaussian (LQG) tracking problem with safety and reachability constraints in the presence of an adversary who mounts an FDI attack on an unknown set of sensors. For each possible set of compromised sensors, we maintain a state estimator disregarding the sensors in that set, and calculate the optimal LQG control input at each time based on this estimate. We propose a control policy which constrains the control input to lie within a fixed distance of the optimal control input corresponding to each state estimate. The control input is obtained at each time step by solving a quadratically constrained quadratic program (QCQP). We prove that our policy can achieve a desired probability of safety and reachability using the barrier certificate method. Our control policy is evaluated via a numerical case study.

摘要: 研究了一类具有安全性和可达性约束的线性二次型高斯(LQG)跟踪问题，其中敌手对未知传感器集发起了FDI攻击。对于每一组可能的受损传感器，我们维护一个状态估计器，而不考虑该集合中的传感器，并基于该估计计算每次的最优LQG控制输入。我们提出了一种控制策略，该策略将控制输入限制在与每个状态估计对应的最优控制输入的固定距离内。在每个时间步通过求解二次约束二次规划(QCQP)来获得控制输入。我们使用屏障证书方法证明了我们的策略可以达到期望的安全和可达性概率。我们的控制策略是通过一个数值案例研究来评估的。



## **33. Towards Effective Multi-Label Recognition Attacks via Knowledge Graph Consistency**

基于知识图一致性的高效多标签识别攻击 cs.CV

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.05137v1)

**Authors**: Hassan Mahmood, Ehsan Elhamifar

**Abstracts**: Many real-world applications of image recognition require multi-label learning, whose goal is to find all labels in an image. Thus, robustness of such systems to adversarial image perturbations is extremely important. However, despite a large body of recent research on adversarial attacks, the scope of the existing works is mainly limited to the multi-class setting, where each image contains a single label. We show that the naive extensions of multi-class attacks to the multi-label setting lead to violating label relationships, modeled by a knowledge graph, and can be detected using a consistency verification scheme. Therefore, we propose a graph-consistent multi-label attack framework, which searches for small image perturbations that lead to misclassifying a desired target set while respecting label hierarchies. By extensive experiments on two datasets and using several multi-label recognition models, we show that our method generates extremely successful attacks that, unlike naive multi-label perturbations, can produce model predictions consistent with the knowledge graph.

摘要: 现实世界中的许多图像识别应用都需要多标签学习，其目标是找到图像中的所有标签。因此，这类系统对对抗性图像扰动的稳健性是极其重要的。然而，尽管最近对对抗性攻击进行了大量的研究，但现有的工作范围主要局限于多类背景下，其中每幅图像包含一个单一的标签。我们证明了多类攻击对多标签设置的天真扩展导致了违反标签关系，用知识图来建模，并且可以使用一致性验证方案来检测。因此，我们提出了一种图一致的多标签攻击框架，该框架在尊重标签层次的同时，搜索导致期望目标集错误分类的微小图像扰动。通过在两个数据集上的广泛实验和使用几个多标签识别模型，我们的方法产生了非常成功的攻击，不同于朴素的多标签扰动，我们可以产生与知识图一致的模型预测。



## **34. RUSH: Robust Contrastive Learning via Randomized Smoothing**

RASH：随机平滑的稳健对比学习 cs.LG

12 pages, 2 figures

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.05127v1)

**Authors**: Yijiang Pang, Boyang Liu, Jiayu Zhou

**Abstracts**: Recently, adversarial training has been incorporated in self-supervised contrastive pre-training to augment label efficiency with exciting adversarial robustness. However, the robustness came at a cost of expensive adversarial training. In this paper, we show a surprising fact that contrastive pre-training has an interesting yet implicit connection with robustness, and such natural robustness in the pre trained representation enables us to design a powerful robust algorithm against adversarial attacks, RUSH, that combines the standard contrastive pre-training and randomized smoothing. It boosts both standard accuracy and robust accuracy, and significantly reduces training costs as compared with adversarial training. We use extensive empirical studies to show that the proposed RUSH outperforms robust classifiers from adversarial training, by a significant margin on common benchmarks (CIFAR-10, CIFAR-100, and STL-10) under first-order attacks. In particular, under $\ell_{\infty}$-norm perturbations of size 8/255 PGD attack on CIFAR-10, our model using ResNet-18 as backbone reached 77.8% robust accuracy and 87.9% standard accuracy. Our work has an improvement of over 15% in robust accuracy and a slight improvement in standard accuracy, compared to the state-of-the-arts.

摘要: 最近，对抗性训练被结合到自我监督的对比预训练中，以增强标记的效率和令人兴奋的对抗性健壮性。然而，这种健壮性是以昂贵的对抗性训练为代价的。在这篇文章中，我们展示了一个令人惊讶的事实，即对比预训练与稳健性有着有趣而隐含的联系，而预训练表示中的这种自然的稳健性使我们能够设计出一种结合了标准的对比预训练和随机平滑的强大的抗对手攻击的鲁棒算法RASH。与对抗性训练相比，它同时提高了标准准确率和稳健准确率，并显著降低了训练成本。我们使用广泛的实证研究表明，在一阶攻击下，所提出的冲刺算法在常见基准(CIFAR-10、CIFAR-100和STL-10)上的性能明显优于来自对手训练的稳健分类器。特别是，在CIFAR-10遭受8/255 PGD攻击时，以ResNet-18为主干的模型达到了77.8%的稳健准确率和87.9%的标准准确率。与最新水平相比，我们的工作在稳健精度上提高了15%以上，在标准精度上略有提高。



## **35. Physical Passive Patch Adversarial Attacks on Visual Odometry Systems**

对视觉里程计系统的物理被动补丁敌意攻击 cs.CV

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.05729v1)

**Authors**: Yaniv Nemcovsky, Matan Yaakoby, Alex M. Bronstein, Chaim Baskin

**Abstracts**: Deep neural networks are known to be susceptible to adversarial perturbations -- small perturbations that alter the output of the network and exist under strict norm limitations. While such perturbations are usually discussed as tailored to a specific input, a universal perturbation can be constructed to alter the model's output on a set of inputs. Universal perturbations present a more realistic case of adversarial attacks, as awareness of the model's exact input is not required. In addition, the universal attack setting raises the subject of generalization to unseen data, where given a set of inputs, the universal perturbations aim to alter the model's output on out-of-sample data. In this work, we study physical passive patch adversarial attacks on visual odometry-based autonomous navigation systems. A visual odometry system aims to infer the relative camera motion between two corresponding viewpoints, and is frequently used by vision-based autonomous navigation systems to estimate their state. For such navigation systems, a patch adversarial perturbation poses a severe security issue, as it can be used to mislead a system onto some collision course. To the best of our knowledge, we show for the first time that the error margin of a visual odometry model can be significantly increased by deploying patch adversarial attacks in the scene. We provide evaluation on synthetic closed-loop drone navigation data and demonstrate that a comparable vulnerability exists in real data. A reference implementation of the proposed method and the reported experiments is provided at https://github.com/patchadversarialattacks/patchadversarialattacks.

摘要: 众所周知，深度神经网络容易受到对抗性扰动的影响--微小的扰动会改变网络的输出，并在严格的范数限制下存在。虽然这样的扰动通常被讨论为针对特定的输入而量身定做的，但是可以构造一个普遍的扰动来改变模型在一组输入上的输出。普遍摄动提供了一种更现实的对抗性攻击情况，因为不需要知道模型的确切输入。此外，通用攻击设置提出了对不可见数据的泛化主题，在给定一组输入的情况下，通用扰动的目的是改变模型对样本外数据的输出。在这项工作中，我们研究了基于视觉里程计的自主导航系统的物理被动补丁对抗性攻击。视觉里程计系统旨在推断两个相应视点之间的相机相对运动，经常被基于视觉的自主导航系统用来估计它们的状态。对于这样的导航系统，补丁对抗性扰动构成了一个严重的安全问题，因为它可能被用来误导系统进入某些碰撞路线。据我们所知，我们首次证明了在场景中部署补丁对抗性攻击可以显著增加视觉里程计模型的误差。我们对合成的无人机闭环导航数据进行了评估，并证明了真实数据中存在类似的漏洞。在https://github.com/patchadversarialattacks/patchadversarialattacks.上提供了所提出的方法和报告的实验的参考实现



## **36. Risk assessment and optimal allocation of security measures under stealthy false data injection attacks**

隐蔽虚假数据注入攻击下的风险评估与安全措施优化配置 eess.SY

Accepted for publication at 6th IEEE Conference on Control Technology  and Applications (CCTA). arXiv admin note: substantial text overlap with  arXiv:2106.07071

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.04860v1)

**Authors**: Sribalaji C. Anand, André M. H. Teixeira, Anders Ahlén

**Abstracts**: This paper firstly addresses the problem of risk assessment under false data injection attacks on uncertain control systems. We consider an adversary with complete system knowledge, injecting stealthy false data into an uncertain control system. We then use the Value-at-Risk to characterize the risk associated with the attack impact caused by the adversary. The worst-case attack impact is characterized by the recently proposed output-to-output gain. We observe that the risk assessment problem corresponds to an infinite non-convex robust optimization problem. To this end, we use dissipative system theory and the scenario approach to approximate the risk-assessment problem into a convex problem and also provide probabilistic certificates on approximation. Secondly, we consider the problem of security measure allocation. We consider an operator with a constraint on the security budget. Under this constraint, we propose an algorithm to optimally allocate the security measures using the calculated risk such that the resulting Value-at-risk is minimized. Finally, we illustrate the results through a numerical example. The numerical example also illustrates that the security allocation using the Value-at-risk, and the impact on the nominal system may have different outcomes: thereby depicting the benefit of using risk metrics.

摘要: 本文首先研究了不确定控制系统在虚假数据注入攻击下的风险评估问题。我们考虑一个拥有完整系统知识的对手，向一个不确定的控制系统注入隐蔽的虚假数据。然后，我们使用风险值来表征与对手造成的攻击影响相关的风险。最坏情况的攻击影响的特征是最近提出的输出到输出的增益。我们观察到风险评估问题对应于一个无穷大的非凸稳健优化问题。为此，我们使用耗散系统理论和情景方法将风险评估问题近似化为一个凸问题，并给出了近似的概率证书。其次，我们考虑了安全措施分配问题。我们考虑一个对安全预算有限制的运营商。在此约束下，我们提出了一种算法，利用计算出的风险对安全措施进行最优分配，使得得到的风险值最小。最后，通过一个数值算例对结果进行了说明。数值例子还表明，使用在险价值的证券分配和对名义系统的影响可能会有不同的结果：从而描绘了使用风险度量的好处。



## **37. Statistical Detection of Adversarial examples in Blockchain-based Federated Forest In-vehicle Network Intrusion Detection Systems**

基于区块链的联邦森林车载网络入侵检测系统中恶意实例的统计检测 cs.CR

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.04843v1)

**Authors**: Ibrahim Aliyu, Selinde van Engelenburg, Muhammed Bashir Muazu, Jinsul Kim, Chang Gyoon Lim

**Abstracts**: The internet-of-Vehicle (IoV) can facilitate seamless connectivity between connected vehicles (CV), autonomous vehicles (AV), and other IoV entities. Intrusion Detection Systems (IDSs) for IoV networks can rely on machine learning (ML) to protect the in-vehicle network from cyber-attacks. Blockchain-based Federated Forests (BFFs) could be used to train ML models based on data from IoV entities while protecting the confidentiality of the data and reducing the risks of tampering with the data. However, ML models created this way are still vulnerable to evasion, poisoning, and exploratory attacks using adversarial examples. This paper investigates the impact of various possible adversarial examples on the BFF-IDS. We proposed integrating a statistical detector to detect and extract unknown adversarial samples. By including the unknown detected samples into the dataset of the detector, we augment the BFF-IDS with an additional model to detect original known attacks and the new adversarial inputs. The statistical adversarial detector confidently detected adversarial examples at the sample size of 50 and 100 input samples. Furthermore, the augmented BFF-IDS (BFF-IDS(AUG)) successfully mitigates the adversarial examples with more than 96% accuracy. With this approach, the model will continue to be augmented in a sandbox whenever an adversarial sample is detected and subsequently adopt the BFF-IDS(AUG) as the active security model. Consequently, the proposed integration of the statistical adversarial detector and the subsequent augmentation of the BFF-IDS with detected adversarial samples provides a sustainable security framework against adversarial examples and other unknown attacks.

摘要: 车联网(IoV)可以促进互联车辆(CV)、自动驾驶车辆(AV)和其他IoV实体之间的无缝连接。IoV网络的入侵检测系统(IDS)可以依靠机器学习(ML)来保护车载网络免受网络攻击。基于区块链的联合森林(BFR)可用于基于来自IoV实体的数据训练ML模型，同时保护数据的机密性并降低篡改数据的风险。然而，以这种方式创建的ML模型仍然容易受到逃避、中毒和使用对抗性示例的探索性攻击。本文研究了各种可能的对抗性例子对BFF-IDS的影响。我们提出集成一个统计检测器来检测和提取未知对手样本。通过将未知的检测样本加入到检测器的数据集中，我们在BFF-IDS中增加了一个额外的模型来检测原始的已知攻击和新的敌意输入。统计敌意检测器在50个和100个输入样本的样本大小下自信地检测到对抗性例子。此外，扩展的BFF-IDS(BFF-IDS(AUG))成功地减少了对抗性实例，准确率超过96%。通过这种方法，只要检测到敌意样本，该模型就会继续在沙箱中进行扩充，并随后采用BFF-IDS(AUG)作为主动安全模型。因此，拟议整合敌意统计探测器，并随后利用检测到的敌意样本加强生物多样性框架--入侵检测系统，这提供了一个可持续的安全框架，可抵御敌意例子和其他未知攻击。



## **38. Physical Attack on Monocular Depth Estimation with Optimal Adversarial Patches**

基于最优对抗性斑块的单目深度估计的物理攻击 cs.CV

ECCV2022

**SubmitDate**: 2022-07-11    [paper-pdf](http://arxiv.org/pdf/2207.04718v1)

**Authors**: Zhiyuan Cheng, James Liang, Hongjun Choi, Guanhong Tao, Zhiwen Cao, Dongfang Liu, Xiangyu Zhang

**Abstracts**: Deep learning has substantially boosted the performance of Monocular Depth Estimation (MDE), a critical component in fully vision-based autonomous driving (AD) systems (e.g., Tesla and Toyota). In this work, we develop an attack against learning-based MDE. In particular, we use an optimization-based method to systematically generate stealthy physical-object-oriented adversarial patches to attack depth estimation. We balance the stealth and effectiveness of our attack with object-oriented adversarial design, sensitive region localization, and natural style camouflage. Using real-world driving scenarios, we evaluate our attack on concurrent MDE models and a representative downstream task for AD (i.e., 3D object detection). Experimental results show that our method can generate stealthy, effective, and robust adversarial patches for different target objects and models and achieves more than 6 meters mean depth estimation error and 93% attack success rate (ASR) in object detection with a patch of 1/9 of the vehicle's rear area. Field tests on three different driving routes with a real vehicle indicate that we cause over 6 meters mean depth estimation error and reduce the object detection rate from 90.70% to 5.16% in continuous video frames.

摘要: 深度学习大大提高了单目深度估计(MDE)的性能，单目深度估计是完全基于视觉的自动驾驶(AD)系统(例如特斯拉和丰田)中的关键组件。在这项工作中，我们开发了一个针对基于学习的MDE的攻击。特别是，我们使用了一种基于优化的方法来系统地生成隐身的面向物理对象的对抗性补丁来进行攻击深度估计。我们在攻击的隐蔽性和有效性与面向对象的对抗性设计、敏感区域定位和自然风格伪装之间取得平衡。使用真实的驾驶场景，我们评估了我们对并发MDE模型的攻击以及一个具有代表性的AD下游任务(即3D对象检测)。实验结果表明，该方法能够针对不同的目标对象和模型生成隐身、高效、健壮的对抗性补丁，在目标检测中以1/9的车尾面积实现6m以上的平均深度估计误差和93%的攻击成功率。对三条不同行驶路线的实际车辆进行的现场测试表明，在连续视频帧中，该算法会产生超过6m的平均深度估计误差，目标检测率从90.70%下降到5.16%。



## **39. Universal Adversarial Examples in Remote Sensing: Methodology and Benchmark**

遥感领域的普遍对抗性实例：方法论和基准 cs.CV

**SubmitDate**: 2022-07-10    [paper-pdf](http://arxiv.org/pdf/2202.07054v2)

**Authors**: Yonghao Xu, Pedram Ghamisi

**Abstracts**: Deep neural networks have achieved great success in many important remote sensing tasks. Nevertheless, their vulnerability to adversarial examples should not be neglected. In this study, we systematically analyze the universal adversarial examples in remote sensing data for the first time, without any knowledge from the victim model. Specifically, we propose a novel black-box adversarial attack method, namely Mixup-Attack, and its simple variant Mixcut-Attack, for remote sensing data. The key idea of the proposed methods is to find common vulnerabilities among different networks by attacking the features in the shallow layer of a given surrogate model. Despite their simplicity, the proposed methods can generate transferable adversarial examples that deceive most of the state-of-the-art deep neural networks in both scene classification and semantic segmentation tasks with high success rates. We further provide the generated universal adversarial examples in the dataset named UAE-RS, which is the first dataset that provides black-box adversarial samples in the remote sensing field. We hope UAE-RS may serve as a benchmark that helps researchers to design deep neural networks with strong resistance toward adversarial attacks in the remote sensing field. Codes and the UAE-RS dataset are available online (https://github.com/YonghaoXu/UAE-RS).

摘要: 深度神经网络在许多重要的遥感任务中取得了巨大的成功。然而，不应忽视它们在对抗性例子面前的脆弱性。在本研究中，我们首次在没有任何受害者模型知识的情况下，系统地分析了遥感数据中的通用对抗性实例。具体来说，我们提出了一种新的针对遥感数据的黑盒对抗攻击方法，即Mixup攻击及其简单的变种MixCut攻击。提出的方法的核心思想是通过攻击给定代理模型浅层的特征来发现不同网络之间的共同漏洞。尽管方法简单，但在场景分类和语义分割任务中，所提出的方法可以生成可转移的对抗性样本，欺骗了大多数最新的深度神经网络，并且成功率很高。此外，我们还在名为UAE-RS的数据集上提供了生成的通用对抗性实例，这是遥感领域中第一个提供黑盒对抗性样本的数据集。我们希望UAE-RS可以作为一个基准，帮助研究人员设计出对遥感领域的敌意攻击具有很强抵抗能力的深度神经网络。代码和阿联酋-RS数据集可在网上获得(https://github.com/YonghaoXu/UAE-RS).



## **40. Visual explanation of black-box model: Similarity Difference and Uniqueness (SIDU) method**

黑盒模型的可视化解释：相似性、差异性和唯一性(SIDU)方法 cs.CV

**SubmitDate**: 2022-07-10    [paper-pdf](http://arxiv.org/pdf/2101.10710v2)

**Authors**: Satya M. Muddamsetty, Mohammad N. S. Jahromi, Andreea E. Ciontos, Laura M. Fenoy, Thomas B. Moeslund

**Abstracts**: Explainable Artificial Intelligence (XAI) has in recent years become a well-suited framework to generate human understandable explanations of "black-box" models. In this paper, a novel XAI visual explanation algorithm known as the Similarity Difference and Uniqueness (SIDU) method that can effectively localize entire object regions responsible for prediction is presented in full detail. The SIDU algorithm robustness and effectiveness is analyzed through various computational and human subject experiments. In particular, the SIDU algorithm is assessed using three different types of evaluations (Application, Human and Functionally-Grounded) to demonstrate its superior performance. The robustness of SIDU is further studied in the presence of adversarial attack on "black-box" models to better understand its performance. Our code is available at: https://github.com/satyamahesh84/SIDU_XAI_CODE.

摘要: 近年来，可解释人工智能(XAI)已经成为一个非常适合生成人类可理解的“黑盒”模型解释的框架。提出了一种新的XAI视觉解释算法--相似性差值唯一性(SIDU)算法，该算法能够有效地定位预测的整个目标区域。通过各种计算和人体实验，分析了SIDU算法的健壮性和有效性。特别是，SIDU算法使用三种不同类型的评估(应用评估、人工评估和基于功能的评估)进行了评估，以展示其优越的性能。为了更好地理解SIDU的性能，进一步研究了SIDU在黑盒模型受到敌意攻击的情况下的稳健性。我们的代码请访问：https://github.com/satyamahesh84/SIDU_XAI_CODE.



## **41. Fooling Partial Dependence via Data Poisoning**

通过数据中毒愚弄部分依赖 cs.LG

Accepted at ECML PKDD 2022

**SubmitDate**: 2022-07-10    [paper-pdf](http://arxiv.org/pdf/2105.12837v3)

**Authors**: Hubert Baniecki, Wojciech Kretowicz, Przemyslaw Biecek

**Abstracts**: Many methods have been developed to understand complex predictive models and high expectations are placed on post-hoc model explainability. It turns out that such explanations are not robust nor trustworthy, and they can be fooled. This paper presents techniques for attacking Partial Dependence (plots, profiles, PDP), which are among the most popular methods of explaining any predictive model trained on tabular data. We showcase that PD can be manipulated in an adversarial manner, which is alarming, especially in financial or medical applications where auditability became a must-have trait supporting black-box machine learning. The fooling is performed via poisoning the data to bend and shift explanations in the desired direction using genetic and gradient algorithms. We believe this to be the first work using a genetic algorithm for manipulating explanations, which is transferable as it generalizes both ways: in a model-agnostic and an explanation-agnostic manner.

摘要: 已经开发了许多方法来理解复杂的预测模型，并对后自组织模型的可解释性寄予了很高的期望。事实证明，这样的解释既不可靠，也不可信，它们可能会被愚弄。这篇文章介绍了攻击部分相关性(曲线图、轮廓、PDP)的技术，这些技术是解释任何基于表格数据训练的预测模型的最流行的方法之一。我们展示了PD可以被以对抗的方式操纵，这是令人震惊的，特别是在金融或医疗应用中，可审计性成为支持黑盒机器学习的必备特征。这种愚弄是通过使用遗传和梯度算法毒化数据来弯曲和改变所需方向的解释来实现的。我们认为这是第一个使用遗传算法来操纵解释的工作，这是可以转移的，因为它概括了两种方式：以模型不可知论和解释不可知论的方式。



## **42. Adversarial Framework with Certified Robustness for Time-Series Domain via Statistical Features**

基于统计特征的时间序列域认证稳健性对抗框架 cs.LG

Published at Journal of Artificial Intelligence Research

**SubmitDate**: 2022-07-09    [paper-pdf](http://arxiv.org/pdf/2207.04307v1)

**Authors**: Taha Belkhouja, Janardhan Rao Doppa

**Abstracts**: Time-series data arises in many real-world applications (e.g., mobile health) and deep neural networks (DNNs) have shown great success in solving them. Despite their success, little is known about their robustness to adversarial attacks. In this paper, we propose a novel adversarial framework referred to as Time-Series Attacks via STATistical Features (TSA-STAT)}. To address the unique challenges of time-series domain, TSA-STAT employs constraints on statistical features of the time-series data to construct adversarial examples. Optimized polynomial transformations are used to create attacks that are more effective (in terms of successfully fooling DNNs) than those based on additive perturbations. We also provide certified bounds on the norm of the statistical features for constructing adversarial examples. Our experiments on diverse real-world benchmark datasets show the effectiveness of TSA-STAT in fooling DNNs for time-series domain and in improving their robustness. The source code of TSA-STAT algorithms is available at https://github.com/tahabelkhouja/Time-Series-Attacks-via-STATistical-Features

摘要: 时间序列数据出现在许多现实世界的应用中(例如移动医疗)，而深度神经网络(DNN)在解决这些问题上取得了巨大的成功。尽管它们取得了成功，但人们对它们对对手攻击的健壮性知之甚少。在本文中，我们提出了一种新的对抗性框架，称为基于统计特征的时间序列攻击(TSA-STAT)。为了解决时间序列领域的独特挑战，TSA-STAT使用了对时间序列数据的统计特征的约束来构造对抗性例子。优化的多项式变换用于创建比基于加性扰动的攻击更有效的攻击(就成功愚弄DNN而言)。我们还提供了用于构造对抗性例子的统计特征范数的确定界。我们在不同的真实基准数据集上的实验表明，TSA-STAT在时间序列域欺骗DNN和提高它们的稳健性方面是有效的。TSA-STAT算法的源代码可在https://github.com/tahabelkhouja/Time-Series-Attacks-via-STATistical-Features上找到



## **43. Federated Learning with Quantum Secure Aggregation**

基于量子安全聚合的联合学习 quant-ph

**SubmitDate**: 2022-07-09    [paper-pdf](http://arxiv.org/pdf/2207.07444v1)

**Authors**: Yichi Zhang, Chao Zhang, Cai Zhang, Lixin Fan, Bei Zeng, Qiang Yang

**Abstracts**: This article illustrates a novel Quantum Secure Aggregation (QSA) scheme that is designed to provide highly secure and efficient aggregation of local model parameters for federated learning. The scheme is secure in protecting private model parameters from being disclosed to semi-honest attackers by utilizing quantum bits i.e. qubits to represent model parameters. The proposed security mechanism ensures that any attempts to eavesdrop private model parameters can be immediately detected and stopped. The scheme is also efficient in terms of the low computational complexity of transmitting and aggregating model parameters through entangled qubits. Benefits of the proposed QSA scheme are showcased in a horizontal federated learning setting in which both a centralized and decentralized architectures are taken into account. It was empirically demonstrated that the proposed QSA can be readily applied to aggregate different types of local models including logistic regression (LR), convolutional neural networks (CNN) as well as quantum neural network (QNN), indicating the versatility of the QSA scheme. Performances of global models are improved to various extents with respect to local models obtained by individual participants, while no private model parameters are disclosed to semi-honest adversaries.

摘要: 本文阐述了一种新的量子安全聚合(QSA)方案，该方案旨在为联合学习提供高度安全和高效的本地模型参数聚合。该方案利用量子比特即量子比特来表示模型参数，从而保护私有模型参数不被泄露给半诚实攻击者。建议的安全机制确保可以立即检测和阻止任何窃听私有模型参数的尝试。该方案在通过纠缠量子比特传输和聚合模型参数的计算复杂度方面也是有效的。在同时考虑集中式和分散式结构的水平联合学习环境中，展示了所提出的QSA方案的优点。实验证明，所提出的QSA方案可以很容易地应用于聚集不同类型的局部模型，包括Logistic回归(LR)、卷积神经网络(CNN)以及量子神经网络(QNN)，这表明了QSA方案的通用性。相对于单个参与者获得的局部模型，全局模型的性能得到了不同程度的提高，而不向半诚实的对手透露任何私人模型参数。



## **44. Not all broken defenses are equal: The dead angles of adversarial accuracy**

并不是所有破碎的防守都是平等的：对手准确性的死角 cs.LG

**SubmitDate**: 2022-07-08    [paper-pdf](http://arxiv.org/pdf/2207.04129v1)

**Authors**: Raphael Olivier, Bhiksha Raj

**Abstracts**: Robustness to adversarial attack is typically evaluated with adversarial accuracy. This metric is however too coarse to properly capture all robustness properties of machine learning models. Many defenses, when evaluated against a strong attack, do not provide accuracy improvements while still contributing partially to adversarial robustness. Popular certification methods suffer from the same issue, as they provide a lower bound to accuracy. To capture finer robustness properties we propose a new metric for L2 robustness, adversarial angular sparsity, which partially answers the question "how many adversarial examples are there around an input". We demonstrate its usefulness by evaluating both "strong" and "weak" defenses. We show that some state-of-the-art defenses, delivering very similar accuracy, can have very different sparsity on the inputs that they are not robust on. We also show that some weak defenses actually decrease robustness, while others strengthen it in a measure that accuracy cannot capture. These differences are predictive of how useful such defenses can become when combined with adversarial training.

摘要: 对敌方攻击的稳健性通常用敌方准确度来评估。然而，这个度量太粗糙了，无法正确地捕获机器学习模型的所有健壮性属性。许多防御在评估强大的攻击时，并不能提高准确率，同时仍能部分提高对手的健壮性。流行的认证方法也存在同样的问题，因为它们提供了准确度的下限。为了获得更好的稳健性，我们提出了一种新的二语稳健性度量--对抗性角度稀疏性，它部分地回答了“一个输入周围有多少个对抗性例子”的问题。我们通过评估“强”和“弱”防御来证明它的有效性。我们表明，一些最先进的防御，提供非常类似的准确性，可以在输入上具有非常不同的稀疏性，而它们的输入并不健壮。我们还表明，一些薄弱的防御实际上降低了稳健性，而另一些则以精确度无法捕捉的方式加强了稳健性。这些差异预示着，当这种防御与对抗性训练相结合时，会变得多么有用。



## **45. Neighbors From Hell: Voltage Attacks Against Deep Learning Accelerators on Multi-Tenant FPGAs**

地狱邻居：针对多租户现场可编程门阵列上深度学习加速器的电压攻击 cs.CR

Published in the 2020 proceedings of the International Conference of  Field-Programmable Technology (ICFPT)

**SubmitDate**: 2022-07-08    [paper-pdf](http://arxiv.org/pdf/2012.07242v2)

**Authors**: Andrew Boutros, Mathew Hall, Nicolas Papernot, Vaughn Betz

**Abstracts**: Field-programmable gate arrays (FPGAs) are becoming widely used accelerators for a myriad of datacenter applications due to their flexibility and energy efficiency. Among these applications, FPGAs have shown promising results in accelerating low-latency real-time deep learning (DL) inference, which is becoming an indispensable component of many end-user applications. With the emerging research direction towards virtualized cloud FPGAs that can be shared by multiple users, the security aspect of FPGA-based DL accelerators requires careful consideration. In this work, we evaluate the security of DL accelerators against voltage-based integrity attacks in a multitenant FPGA scenario. We first demonstrate the feasibility of such attacks on a state-of-the-art Stratix 10 card using different attacker circuits that are logically and physically isolated in a separate attacker role, and cannot be flagged as malicious circuits by conventional bitstream checkers. We show that aggressive clock gating, an effective power-saving technique, can also be a potential security threat in modern FPGAs. Then, we carry out the attack on a DL accelerator running ImageNet classification in the victim role to evaluate the inherent resilience of DL models against timing faults induced by the adversary. We find that even when using the strongest attacker circuit, the prediction accuracy of the DL accelerator is not compromised when running at its safe operating frequency. Furthermore, we can achieve 1.18-1.31x higher inference performance by over-clocking the DL accelerator without affecting its prediction accuracy.

摘要: 现场可编程门阵列(现场可编程门阵列)由于其灵活性和能效，正成为各种数据中心应用的广泛使用的加速器。在这些应用中，现场可编程门阵列在加速低延迟实时深度学习(DL)推理方面表现出了良好的效果，这正成为许多最终用户应用程序中不可或缺的组件。随着可由多个用户共享的虚拟云FPGA的研究方向的出现，基于FPGA的DL加速器的安全方面需要仔细考虑。在这项工作中，我们评估了在多租户现场可编程门阵列场景中，DL加速器抵抗基于电压的完整性攻击的安全性。我们首先证明了使用不同的攻击电路对最先进的Stratix 10卡进行此类攻击的可行性，这些攻击电路在逻辑上和物理上隔离在单独的攻击者角色中，并且不能被传统的比特流检查器标记为恶意电路。我们发现，积极的时钟门控技术是一种有效的节能技术，也可能成为现代现场可编程门阵列中的潜在安全威胁。然后，我们以受害者角色对运行ImageNet分类的DL加速器进行攻击，以评估DL模型对对手导致的时序错误的内在弹性。我们发现，即使在使用最强攻击电路的情况下，DL加速器在其安全工作频率下运行时的预测精度也不会受到影响。此外，通过对DL加速器超频，我们可以在不影响其预测精度的情况下获得1.18-1.31倍的高推理性能。



## **46. Defense Against Multi-target Trojan Attacks**

防御多目标木马攻击 cs.CV

**SubmitDate**: 2022-07-08    [paper-pdf](http://arxiv.org/pdf/2207.03895v1)

**Authors**: Haripriya Harikumar, Santu Rana, Kien Do, Sunil Gupta, Wei Zong, Willy Susilo, Svetha Venkastesh

**Abstracts**: Adversarial attacks on deep learning-based models pose a significant threat to the current AI infrastructure. Among them, Trojan attacks are the hardest to defend against. In this paper, we first introduce a variation of the Badnet kind of attacks that introduces Trojan backdoors to multiple target classes and allows triggers to be placed anywhere in the image. The former makes it more potent and the latter makes it extremely easy to carry out the attack in the physical space. The state-of-the-art Trojan detection methods fail with this threat model. To defend against this attack, we first introduce a trigger reverse-engineering mechanism that uses multiple images to recover a variety of potential triggers. We then propose a detection mechanism by measuring the transferability of such recovered triggers. A Trojan trigger will have very high transferability i.e. they make other images also go to the same class. We study many practical advantages of our attack method and then demonstrate the detection performance using a variety of image datasets. The experimental results show the superior detection performance of our method over the state-of-the-arts.

摘要: 对基于深度学习的模型的对抗性攻击对当前的人工智能基础设施构成了重大威胁。其中，木马攻击是最难防御的。在本文中，我们首先介绍了Badnet类型的攻击的一个变体，该攻击将特洛伊木马后门引入多个目标类，并允许在图像中的任何位置放置触发器。前者使其更具威力，而后者使在物理空间进行攻击变得极其容易。最先进的特洛伊木马检测方法在此威胁模型下失败。为了防御这种攻击，我们首先引入了一种触发反向工程机制，该机制使用多个图像来恢复各种潜在的触发。然后，我们提出了一种通过测量这些恢复的触发器的可转移性来检测的机制。特洛伊木马触发器将具有非常高的可转移性，即它们使其他图像也进入同一类。我们研究了该攻击方法的许多实用优势，并使用各种图像数据集演示了该方法的检测性能。实验结果表明，该方法具有较好的检测性能。



## **47. Towards Effective and Robust Neural Trojan Defenses via Input Filtering**

通过输入过滤实现高效、健壮的神经木马防御 cs.CR

Accepted to ECCV 2022

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2202.12154v4)

**Authors**: Kien Do, Haripriya Harikumar, Hung Le, Dung Nguyen, Truyen Tran, Santu Rana, Dang Nguyen, Willy Susilo, Svetha Venkatesh

**Abstracts**: Trojan attacks on deep neural networks are both dangerous and surreptitious. Over the past few years, Trojan attacks have advanced from using only a single input-agnostic trigger and targeting only one class to using multiple, input-specific triggers and targeting multiple classes. However, Trojan defenses have not caught up with this development. Most defense methods still make inadequate assumptions about Trojan triggers and target classes, thus, can be easily circumvented by modern Trojan attacks. To deal with this problem, we propose two novel "filtering" defenses called Variational Input Filtering (VIF) and Adversarial Input Filtering (AIF) which leverage lossy data compression and adversarial learning respectively to effectively purify potential Trojan triggers in the input at run time without making assumptions about the number of triggers/target classes or the input dependence property of triggers. In addition, we introduce a new defense mechanism called "Filtering-then-Contrasting" (FtC) which helps avoid the drop in classification accuracy on clean data caused by "filtering", and combine it with VIF/AIF to derive new defenses of this kind. Extensive experimental results and ablation studies show that our proposed defenses significantly outperform well-known baseline defenses in mitigating five advanced Trojan attacks including two recent state-of-the-art while being quite robust to small amounts of training data and large-norm triggers.

摘要: 特洛伊木马对深层神经网络的攻击既危险又隐蔽。在过去的几年中，特洛伊木马攻击已经从只使用一个与输入无关的触发器和只针对一个类发展到使用多个特定于输入的触发器和目标多个类。然而，特洛伊木马防御并没有跟上这一发展。大多数防御方法仍然对木马的触发器和目标类别做了不充分的假设，因此很容易被现代木马攻击所规避。针对这一问题，我们提出了两种新的“过滤”防御机制，称为变输入过滤(VIF)和对抗输入过滤(AIF)，它们分别利用有损数据压缩和对抗学习在运行时有效地净化输入中潜在的特洛伊木马触发器，而不需要假设触发器/目标类的数量或触发器的输入依赖属性。此外，我们还引入了一种新的防御机制，称为“过滤-然后-对比”(FTC)，它有助于避免“过滤”导致对干净数据的分类精度的下降，并将其与VIF/AIF相结合来派生出这种新的防御机制。广泛的实验结果和烧蚀研究表明，我们提出的防御方案在缓解五种高级特洛伊木马攻击(包括两种最新的木马攻击)方面明显优于众所周知的基线防御方案，同时对少量训练数据和大范数触发事件具有相当的健壮性。



## **48. On the Relationship Between Adversarial Robustness and Decision Region in Deep Neural Network**

深度神经网络中敌方稳健性与决策区域的关系 cs.LG

14 pages

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2207.03400v1)

**Authors**: Seongjin Park, Haedong Jeong, Giyoung Jeon, Jaesik Choi

**Abstracts**: In general, Deep Neural Networks (DNNs) are evaluated by the generalization performance measured on unseen data excluded from the training phase. Along with the development of DNNs, the generalization performance converges to the state-of-the-art and it becomes difficult to evaluate DNNs solely based on this metric. The robustness against adversarial attack has been used as an additional metric to evaluate DNNs by measuring their vulnerability. However, few studies have been performed to analyze the adversarial robustness in terms of the geometry in DNNs. In this work, we perform an empirical study to analyze the internal properties of DNNs that affect model robustness under adversarial attacks. In particular, we propose the novel concept of the Populated Region Set (PRS), where training samples are populated more frequently, to represent the internal properties of DNNs in a practical setting. From systematic experiments with the proposed concept, we provide empirical evidence to validate that a low PRS ratio has a strong relationship with the adversarial robustness of DNNs. We also devise PRS regularizer leveraging the characteristics of PRS to improve the adversarial robustness without adversarial training.

摘要: 通常，深度神经网络(DNN)是通过在训练阶段排除的未知数据上测量的泛化性能来评估的。随着DNN的发展，泛化性能趋于最新水平，单纯基于这一指标来评价DNN变得越来越困难。对敌意攻击的健壮性已被用作通过测量DNN的脆弱性来评估DNN的额外度量。然而，很少有研究从DNN的几何角度来分析敌手的稳健性。在这项工作中，我们进行了一项实证研究，以分析在敌意攻击下影响模型稳健性的DNN的内部属性。特别是，我们提出了填充训练样本频率更高的填充区域集合(Prs)的新概念，以表示实际环境中DNN的内在属性。通过对所提出概念的系统实验，我们提供了经验证据来验证较低的粗糙集比率与DNN的对抗健壮性有很强的关系。利用粗糙集的特点，设计了粗糙集正则化算法，在不需要对手训练的情况下提高了对手的健壮性。



## **49. Federated Robustness Propagation: Sharing Robustness in Heterogeneous Federated Learning**

联邦健壮性传播：异质联邦学习中的健壮性共享 cs.LG

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2106.10196v2)

**Authors**: Junyuan Hong, Haotao Wang, Zhangyang Wang, Jiayu Zhou

**Abstracts**: Federated learning (FL) emerges as a popular distributed learning schema that learns a model from a set of participating users without sharing raw data. One major challenge of FL comes with heterogeneous users, who may have distributionally different (or non-iid) data and varying computation resources. As federated users would use the model for prediction, they often demand the trained model to be robust against malicious attackers at test time. Whereas adversarial training (AT) provides a sound solution for centralized learning, extending its usage for federated users has imposed significant challenges, as many users may have very limited training data and tight computational budgets, to afford the data-hungry and costly AT. In this paper, we study a novel FL strategy: propagating adversarial robustness from rich-resource users that can afford AT, to those with poor resources that cannot afford it, during federated learning. We show that existing FL techniques cannot be effectively integrated with the strategy to propagate robustness among non-iid users and propose an efficient propagation approach by the proper use of batch-normalization. We demonstrate the rationality and effectiveness of our method through extensive experiments. Especially, the proposed method is shown to grant federated models remarkable robustness even when only a small portion of users afford AT during learning. Source code will be released.

摘要: 联合学习(FL)是一种流行的分布式学习模式，它从一组参与的用户那里学习模型，而不共享原始数据。FL的一个主要挑战来自不同的用户，他们可能具有分布不同(或非IID)的数据和不同的计算资源。由于联合用户将使用该模型进行预测，因此他们经常要求训练后的模型在测试时对恶意攻击者具有健壮性。虽然对抗训练(AT)为集中学习提供了一种合理的解决方案，但扩展其对联合用户的使用带来了巨大的挑战，因为许多用户可能具有非常有限的训练数据和紧张的计算预算，以负担数据匮乏和成本高昂的AT。在本文中，我们研究了一种新的FL策略：在联合学习过程中，将对抗健壮性从有能力负担AT的资源丰富的用户传播到资源贫乏的用户。我们证明了现有的FL技术不能有效地与在非IID用户之间传播健壮性的策略相结合，并通过适当地使用批处理归一化来提出一种有效的传播方法。通过大量的实验验证了该方法的合理性和有效性。特别是，在只有一小部分用户在学习过程中提供AT的情况下，所提出的方法被证明具有显著的健壮性。源代码将会公布。



## **50. SYNFI: Pre-Silicon Fault Analysis of an Open-Source Secure Element**

SYNFI：一种开源安全元件的硅前故障分析 cs.CR

**SubmitDate**: 2022-07-07    [paper-pdf](http://arxiv.org/pdf/2205.04775v2)

**Authors**: Pascal Nasahl, Miguel Osorio, Pirmin Vogel, Michael Schaffner, Timothy Trippel, Dominic Rizzo, Stefan Mangard

**Abstracts**: Fault attacks are active, physical attacks that an adversary can leverage to alter the control-flow of embedded devices to gain access to sensitive information or bypass protection mechanisms. Due to the severity of these attacks, manufacturers deploy hardware-based fault defenses into security-critical systems, such as secure elements. The development of these countermeasures is a challenging task due to the complex interplay of circuit components and because contemporary design automation tools tend to optimize inserted structures away, thereby defeating their purpose. Hence, it is critical that such countermeasures are rigorously verified post-synthesis. As classical functional verification techniques fall short of assessing the effectiveness of countermeasures, developers have to resort to methods capable of injecting faults in a simulation testbench or into a physical chip. However, developing test sequences to inject faults in simulation is an error-prone task and performing fault attacks on a chip requires specialized equipment and is incredibly time-consuming. To that end, this paper introduces SYNFI, a formal pre-silicon fault verification framework that operates on synthesized netlists. SYNFI can be used to analyze the general effect of faults on the input-output relationship in a circuit and its fault countermeasures, and thus enables hardware designers to assess and verify the effectiveness of embedded countermeasures in a systematic and semi-automatic way. To demonstrate that SYNFI is capable of handling unmodified, industry-grade netlists synthesized with commercial and open tools, we analyze OpenTitan, the first open-source secure element. In our analysis, we identified critical security weaknesses in the unprotected AES block, developed targeted countermeasures, reassessed their security, and contributed these countermeasures back to the OpenTitan repository.

摘要: 故障攻击是一种主动的物理攻击，攻击者可以利用这些攻击来改变嵌入式设备的控制流，从而获得对敏感信息的访问权限或绕过保护机制。由于这些攻击的严重性，制造商将基于硬件的故障防御部署到安全关键系统中，例如安全元件。这些对策的开发是一项具有挑战性的任务，因为电路元件之间的复杂相互作用，以及现代设计自动化工具倾向于优化插入的结构，从而违背了它们的目的。因此，至关重要的是，这些对策在合成后得到严格验证。由于传统的功能验证技术无法评估对策的有效性，开发人员不得不求助于能够在模拟测试台或物理芯片中注入故障的方法。然而，开发测试序列以在模拟中注入故障是一项容易出错的任务，在芯片上执行故障攻击需要专门的设备，并且非常耗时。为此，本文引入了SYNFI，这是一个运行在合成网表上的形式化的预硅故障验证框架。SYNFI可以用来分析故障对电路输入输出关系的一般影响及其故障对策，从而使硬件设计者能够以系统和半自动的方式评估和验证嵌入式对策的有效性。为了证明SYNFI能够处理使用商业和开放工具合成的未经修改的工业级网表，我们分析了第一个开源安全元素OpenTitan。在我们的分析中，我们确定了未受保护的AES块中的关键安全漏洞，开发了有针对性的对策，重新评估了它们的安全性，并将这些对策贡献给了OpenTitan存储库。



