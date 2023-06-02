# Latest Adversarial Attack Papers
**update at 2023-06-02 11:57:32**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Intriguing Properties of Text-guided Diffusion Models**

文本引导扩散模型的有趣性质 cs.CV

Code will be available at: https://github.com/qihao067/SAGE/

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2306.00974v1) [paper-pdf](http://arxiv.org/pdf/2306.00974v1)

**Authors**: Qihao Liu, Adam Kortylewski, Yutong Bai, Song Bai, Alan Yuille

**Abstract**: Text-guided diffusion models (TDMs) are widely applied but can fail unexpectedly. Common failures include: (i) natural-looking text prompts generating images with the wrong content, or (ii) different random samples of the latent variables that generate vastly different, and even unrelated, outputs despite being conditioned on the same text prompt. In this work, we aim to study and understand the failure modes of TDMs in more detail. To achieve this, we propose SAGE, an adversarial attack on TDMs that uses image classifiers as surrogate loss functions, to search over the discrete prompt space and the high-dimensional latent space of TDMs to automatically discover unexpected behaviors and failure cases in the image generation. We make several technical contributions to ensure that SAGE finds failure cases of the diffusion model, rather than the classifier, and verify this in a human study. Our study reveals four intriguing properties of TDMs that have not been systematically studied before: (1) We find a variety of natural text prompts producing images that fail to capture the semantics of input texts. We categorize these failures into ten distinct types based on the underlying causes. (2) We find samples in the latent space (which are not outliers) that lead to distorted images independent of the text prompt, suggesting that parts of the latent space are not well-structured. (3) We also find latent samples that lead to natural-looking images which are unrelated to the text prompt, implying a potential misalignment between the latent and prompt spaces. (4) By appending a single adversarial token embedding to an input prompt we can generate a variety of specified target objects, while only minimally affecting the CLIP score. This demonstrates the fragility of language representations and raises potential safety concerns.

摘要: 文本引导扩散模型(TDM)被广泛应用，但可能会意外失败。常见的故障包括：(I)看起来自然的文本提示生成具有错误内容的图像，或(Ii)潜在变量的不同随机样本，尽管以相同的文本提示为条件，但生成的输出却截然不同，甚至是无关的。在这项工作中，我们旨在更详细地研究和理解TDM的故障模式。为此，我们提出了一种针对TDMS的对抗性攻击方法SAGE，它使用图像分类器作为代理损失函数，在TDMS的离散提示空间和高维潜在空间中进行搜索，自动发现图像生成中的意外行为和失败案例。我们做出了几项技术贡献，以确保SAGE找到扩散模型的故障案例，而不是分类器，并在人体研究中验证这一点。我们的研究揭示了TDM的四个以前没有被系统研究的有趣的性质：(1)我们发现各种自然的文本提示产生的图像无法捕捉到输入文本的语义。根据根本原因，我们将这些故障分为十种不同的类型。(2)我们在潜在空间(不是离群点)中发现了与文本提示无关的导致失真图像的样本，这表明潜在空间的部分结构不是良好的。(3)我们还发现潜在样本导致了与文本提示无关的看起来自然的图像，这意味着潜在空间和提示空间之间存在潜在的错位。(4)通过在输入提示中添加单个对抗性令牌，我们可以生成各种指定的目标对象，而对片段得分的影响最小。这表明了语言表达的脆弱性，并引发了潜在的安全问题。



## **2. Adversarial Robustness in Unsupervised Machine Learning: A Systematic Review**

无监督机器学习中的对抗性稳健性：系统评价 cs.LG

38 pages, 11 figures

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2306.00687v1) [paper-pdf](http://arxiv.org/pdf/2306.00687v1)

**Authors**: Mathias Lundteigen Mohus, Jinyue Li

**Abstract**: As the adoption of machine learning models increases, ensuring robust models against adversarial attacks is increasingly important. With unsupervised machine learning gaining more attention, ensuring it is robust against attacks is vital. This paper conducts a systematic literature review on the robustness of unsupervised learning, collecting 86 papers. Our results show that most research focuses on privacy attacks, which have effective defenses; however, many attacks lack effective and general defensive measures. Based on the results, we formulate a model on the properties of an attack on unsupervised learning, contributing to future research by providing a model to use.

摘要: 随着机器学习模型的采用越来越多，确保稳健的模型抵御对手攻击变得越来越重要。随着无监督机器学习获得越来越多的关注，确保其对攻击的健壮性至关重要。本文对非监督学习的稳健性进行了系统的文献综述，共收集了86篇论文。我们的研究结果表明，大多数研究集中在隐私攻击上，这些攻击都有有效的防御措施，但许多攻击缺乏有效和通用的防御措施。在此基础上，我们建立了一个关于无监督学习攻击性质的模型，为以后的研究提供了一个可供使用的模型。



## **3. Byzantine-Robust Clustered Federated Learning**

拜占庭--稳健的分簇联合学习 stat.ML

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2306.00638v1) [paper-pdf](http://arxiv.org/pdf/2306.00638v1)

**Authors**: Zhixu Tao, Kun Yang, Sanjeev R. Kulkarni

**Abstract**: This paper focuses on the problem of adversarial attacks from Byzantine machines in a Federated Learning setting where non-Byzantine machines can be partitioned into disjoint clusters. In this setting, non-Byzantine machines in the same cluster have the same underlying data distribution, and different clusters of non-Byzantine machines have different learning tasks. Byzantine machines can adversarially attack any cluster and disturb the training process on clusters they attack. In the presence of Byzantine machines, the goal of our work is to identify cluster membership of non-Byzantine machines and optimize the models learned by each cluster. We adopt the Iterative Federated Clustering Algorithm (IFCA) framework of Ghosh et al. (2020) to alternatively estimate cluster membership and optimize models. In order to make this framework robust against adversarial attacks from Byzantine machines, we use coordinate-wise trimmed mean and coordinate-wise median aggregation methods used by Yin et al. (2018). Specifically, we propose a new Byzantine-Robust Iterative Federated Clustering Algorithm to improve on the results in Ghosh et al. (2019). We prove a convergence rate for this algorithm for strongly convex loss functions. We compare our convergence rate with the convergence rate of an existing algorithm, and we demonstrate the performance of our algorithm on simulated data.

摘要: 本文主要研究联邦学习环境下拜占庭机器的敌意攻击问题，其中非拜占庭机器可以划分为不相交的簇。在此设置中，同一集群中的非拜占庭机器具有相同的底层数据分布，而不同的非拜占庭机器集群具有不同的学习任务。拜占庭机器可以恶意攻击任何集群，并扰乱它们攻击的集群的训练过程。在拜占庭机器存在的情况下，我们的工作的目标是识别非拜占庭机器的簇成员资格，并优化每个簇学习的模型。我们采用了Ghosh等人的迭代联邦聚类算法(IFCA)框架。(2020)交替估计集群成员资格和优化模型。为了使该框架对拜占庭机器的敌意攻击具有健壮性，我们使用了Yen等人使用的坐标修剪均值和坐标中值聚合方法。(2018)。具体地说，我们提出了一种新的拜占庭-稳健迭代联邦聚类算法来改进Ghosh等人的结果。(2019年)。对于强凸损失函数，我们证明了该算法的收敛速度。我们将我们的收敛速度与现有算法的收敛速度进行了比较，并在模拟数据上展示了我们的算法的性能。



## **4. Spying on the Spy: Security Analysis of Hidden Cameras**

监视间谍：隐藏式摄像机的安全性分析 cs.CR

19 pages. Conference: NSS 2023: 17th International Conference on  Network and System Security

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2306.00610v1) [paper-pdf](http://arxiv.org/pdf/2306.00610v1)

**Authors**: Samuel Herodotou, Feng Hao

**Abstract**: Hidden cameras, also called spy cameras, are surveillance tools commonly used to spy on people without their knowledge. Whilst previous studies largely focused on investigating the detection of such a camera and the privacy implications, the security of the camera itself has received limited attention. Compared with ordinary IP cameras, spy cameras are normally sold in bulk at cheap prices and are ubiquitously deployed in hidden places within homes and workplaces. A security compromise of these cameras can have severe consequences. In this paper, we analyse a generic IP camera module, which has been packaged and re-branded for sale by several spy camera vendors. The module is controlled by mobile phone apps. By analysing the Android app and the traffic data, we reverse-engineered the security design of the whole system, including the module's Linux OS environment, the file structure, the authentication mechanism, the session management, and the communication with a remote server. Serious vulnerabilities have been identified in every component. Combined together, they allow an adversary to take complete control of a spy camera from anywhere over the Internet, enabling arbitrary code execution. This is possible even if the camera is behind a firewall. All that an adversary needs to launch an attack is the camera's serial number, which users sometimes unknowingly share in online reviews. We responsibly disclosed our findings to the manufacturer. Whilst the manufacturer acknowledged our work, they showed no intention to fix the problems. Patching or recalling the affected cameras is infeasible due to complexities in the supply chain. However, it is prudent to assume that bad actors have already been exploiting these flaws. We provide details of the identified vulnerabilities in order to raise public awareness, especially on the grave danger of disclosing a spy camera's serial number.

摘要: 隐藏摄像头，也被称为间谍摄像头，是通常用来在人们不知情的情况下监视他们的监视工具。虽然之前的研究主要集中在调查此类摄像头的检测及其对隐私的影响，但摄像头本身的安全性受到的关注有限。与普通IP摄像头相比，间谍摄像头通常以低廉的价格批量销售，并无处不在地部署在家庭和工作场所的隐蔽场所。这些摄像头的安全漏洞可能会产生严重的后果。在本文中，我们分析了一个通用的IP摄像机模块，它已经被几个间谍摄像机供应商包装并重新命名以供销售。该模块由手机应用程序控制。通过分析Android应用程序和流量数据，对整个系统的安全设计进行了逆向工程，包括模块的Linux操作系统环境、文件结构、认证机制、会话管理以及与远程服务器的通信。在每个组件中都发现了严重的漏洞。它们结合在一起，允许对手从互联网上的任何地方完全控制间谍摄像头，从而实现任意代码执行。即使摄像机位于防火墙之后，这也是可能的。对手发动攻击所需要的只是摄像头的序列号，用户有时会在不知情的情况下在在线评论中分享这个序列号。我们负责任地向制造商透露了我们的发现。虽然制造商承认了我们的工作，但他们没有表现出解决问题的意图。由于供应链的复杂性，修补或召回受影响的摄像头是不可行的。然而，谨慎的假设是，不良行为者已经在利用这些缺陷。我们提供已识别漏洞的详细信息，以提高公众的认识，特别是对泄露间谍摄像头序列号的严重危险的认识。



## **5. Does Black-box Attribute Inference Attacks on Graph Neural Networks Constitute Privacy Risk?**

对图神经网络的黑盒属性推理攻击是否构成隐私风险？ cs.LG

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2306.00578v1) [paper-pdf](http://arxiv.org/pdf/2306.00578v1)

**Authors**: Iyiola E. Olatunji, Anmar Hizber, Oliver Sihlovec, Megha Khosla

**Abstract**: Graph neural networks (GNNs) have shown promising results on real-life datasets and applications, including healthcare, finance, and education. However, recent studies have shown that GNNs are highly vulnerable to attacks such as membership inference attack and link reconstruction attack. Surprisingly, attribute inference attacks has received little attention. In this paper, we initiate the first investigation into attribute inference attack where an attacker aims to infer the sensitive user attributes based on her public or non-sensitive attributes. We ask the question whether black-box attribute inference attack constitutes a significant privacy risk for graph-structured data and their corresponding GNN model. We take a systematic approach to launch the attacks by varying the adversarial knowledge and assumptions. Our findings reveal that when an attacker has black-box access to the target model, GNNs generally do not reveal significantly more information compared to missing value estimation techniques. Code is available.

摘要: 图形神经网络(GNN)在现实生活数据集和应用方面显示了良好的结果，包括医疗保健、金融和教育。然而，最近的研究表明，GNN非常容易受到成员推理攻击和链路重建攻击等攻击。令人惊讶的是，属性推理攻击几乎没有受到关注。在本文中，我们首次对属性推理攻击进行了研究，攻击者的目标是根据用户的公共属性或非敏感属性来推断敏感用户属性。我们提出的问题是，黑盒属性推理攻击是否对图结构数据及其对应的GNN模型构成了显著的隐私风险。我们采取系统化的方法，通过改变对手的知识和假设来发动攻击。我们的研究结果表明，当攻击者拥有对目标模型的黑盒访问权限时，GNN通常不会透露比未命中值估计技术更多的信息。代码可用。



## **6. Constructing Semantics-Aware Adversarial Examples with Probabilistic Perspective**

从概率的角度构建语义感知的对抗性实例 stat.ML

17 pages, 14 figures

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2306.00353v1) [paper-pdf](http://arxiv.org/pdf/2306.00353v1)

**Authors**: Andi Zhang, Damon Wischik

**Abstract**: In this study, we introduce a novel, probabilistic viewpoint on adversarial examples, achieved through box-constrained Langevin Monte Carlo (LMC). Proceeding from this perspective, we develop an innovative approach for generating semantics-aware adversarial examples in a principled manner. This methodology transcends the restriction imposed by geometric distance, instead opting for semantic constraints. Our approach empowers individuals to incorporate their personal comprehension of semantics into the model. Through human evaluation, we validate that our semantics-aware adversarial examples maintain their inherent meaning. Experimental findings on the MNIST and SVHN datasets demonstrate that our semantics-aware adversarial examples can effectively circumvent robust adversarial training methods tailored for traditional adversarial attacks.

摘要: 在这项研究中，我们介绍了一种新的对抗性例子的概率观点，通过盒约束的朗之万蒙特卡罗(LMC)实现。从这个角度出发，我们开发了一种创新的方法，以原则性的方式生成语义感知的对抗性实例。这种方法超越了几何距离的限制，而是选择了语义约束。我们的方法使个人能够将他们对语义的个人理解融入到模型中。通过人类的评估，我们验证了我们的语义感知的对抗性例子保持了它们的内在含义。在MNIST和SVHN数据集上的实验结果表明，我们的语义感知的对抗性例子可以有效地绕过针对传统对抗性攻击定制的健壮的对抗性训练方法。



## **7. CALICO: Self-Supervised Camera-LiDAR Contrastive Pre-training for BEV Perception**

Calico：BEV感知的自我监控相机-LiDAR对比预训练 cs.CV

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2306.00349v1) [paper-pdf](http://arxiv.org/pdf/2306.00349v1)

**Authors**: Jiachen Sun, Haizhong Zheng, Qingzhao Zhang, Atul Prakash, Z. Morley Mao, Chaowei Xiao

**Abstract**: Perception is crucial in the realm of autonomous driving systems, where bird's eye view (BEV)-based architectures have recently reached state-of-the-art performance. The desirability of self-supervised representation learning stems from the expensive and laborious process of annotating 2D and 3D data. Although previous research has investigated pretraining methods for both LiDAR and camera-based 3D object detection, a unified pretraining framework for multimodal BEV perception is missing. In this study, we introduce CALICO, a novel framework that applies contrastive objectives to both LiDAR and camera backbones. Specifically, CALICO incorporates two stages: point-region contrast (PRC) and region-aware distillation (RAD). PRC better balances the region- and scene-level representation learning on the LiDAR modality and offers significant performance improvement compared to existing methods. RAD effectively achieves contrastive distillation on our self-trained teacher model. CALICO's efficacy is substantiated by extensive evaluations on 3D object detection and BEV map segmentation tasks, where it delivers significant performance improvements. Notably, CALICO outperforms the baseline method by 10.5% and 8.6% on NDS and mAP. Moreover, CALICO boosts the robustness of multimodal 3D object detection against adversarial attacks and corruption. Additionally, our framework can be tailored to different backbones and heads, positioning it as a promising approach for multimodal BEV perception.

摘要: 在自动驾驶系统领域，感知是至关重要的，基于鸟瞰(Bev)的架构最近达到了最先进的性能。自监督表示学习的可取性源于对2D和3D数据进行标注的昂贵且费力的过程。虽然以前的研究已经研究了LiDAR和基于摄像机的3D目标检测的预训练方法，但缺乏一个统一的多模式Bev感知的预训练框架。在这项研究中，我们介绍了Calico，这是一个新的框架，它将对比目标应用于LiDAR和相机主干。具体地说，Calico包含两个阶段：点区域对比(PRC)和区域感知蒸馏(RAD)。PRC在LiDAR通道上更好地平衡了区域和场景级别的表示学习，并与现有方法相比提供了显著的性能改进。RAD有效地实现了对我们自学教师模式的对比提炼。Calico的有效性通过对3D对象检测和Bev地图分割任务的广泛评估得到证实，在这些任务中，Calico提供了显著的性能改进。值得注意的是，Calico在NDS和MAP上分别比基线方法高出10.5%和8.6%。此外，Calico增强了多模式3D对象检测针对对手攻击和损坏的健壮性。此外，我们的框架可以为不同的主干和头部量身定做，将其定位为一种有前途的多模式Bev感知方法。



## **8. Improving Adversarial Robustness by Putting More Regularizations on Less Robust Samples**

通过对稳健性较差的样本进行更多的规则化来提高对手的稳健性 stat.ML

Accepted in ICML 2023

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2206.03353v4) [paper-pdf](http://arxiv.org/pdf/2206.03353v4)

**Authors**: Dongyoon Yang, Insung Kong, Yongdai Kim

**Abstract**: Adversarial training, which is to enhance robustness against adversarial attacks, has received much attention because it is easy to generate human-imperceptible perturbations of data to deceive a given deep neural network. In this paper, we propose a new adversarial training algorithm that is theoretically well motivated and empirically superior to other existing algorithms. A novel feature of the proposed algorithm is to apply more regularization to data vulnerable to adversarial attacks than other existing regularization algorithms do. Theoretically, we show that our algorithm can be understood as an algorithm of minimizing the regularized empirical risk motivated from a newly derived upper bound of the robust risk. Numerical experiments illustrate that our proposed algorithm improves the generalization (accuracy on examples) and robustness (accuracy on adversarial attacks) simultaneously to achieve the state-of-the-art performance.

摘要: 对抗性训练是为了提高对抗攻击的稳健性，因为它很容易产生人类无法察觉的数据扰动来欺骗给定的深度神经网络。在本文中，我们提出了一种新的对抗性训练算法，该算法在理论上动机良好，在经验上优于其他现有的算法。与现有的正则化算法相比，该算法的一个新特点是对易受敌意攻击的数据进行了更多的正则化。理论上，我们的算法可以理解为最小化正则化经验风险的算法，该正则化经验风险是由新导出的稳健风险上界引起的。数值实验表明，我们提出的算法同时提高了泛化(例题准确率)和稳健性(对抗性攻击准确率)，达到了最好的性能。



## **9. Adversarial-Aware Deep Learning System based on a Secondary Classical Machine Learning Verification Approach**

基于二次经典机器学习验证方法的对抗性深度学习系统 cs.CR

17 pages, 3 figures

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2306.00314v1) [paper-pdf](http://arxiv.org/pdf/2306.00314v1)

**Authors**: Mohammed Alkhowaiter, Hisham Kholidy, Mnassar Alyami, Abdulmajeed Alghamdi, Cliff Zou

**Abstract**: Deep learning models have been used in creating various effective image classification applications. However, they are vulnerable to adversarial attacks that seek to misguide the models into predicting incorrect classes. Our study of major adversarial attack models shows that they all specifically target and exploit the neural networking structures in their designs. This understanding makes us develop a hypothesis that most classical machine learning models, such as Random Forest (RF), are immune to adversarial attack models because they do not rely on neural network design at all. Our experimental study of classical machine learning models against popular adversarial attacks supports this hypothesis. Based on this hypothesis, we propose a new adversarial-aware deep learning system by using a classical machine learning model as the secondary verification system to complement the primary deep learning model in image classification. Although the secondary classical machine learning model has less accurate output, it is only used for verification purposes, which does not impact the output accuracy of the primary deep learning model, and at the same time, can effectively detect an adversarial attack when a clear mismatch occurs. Our experiments based on CIFAR-100 dataset show that our proposed approach outperforms current state-of-the-art adversarial defense systems.

摘要: 深度学习模型已被用于创建各种有效的图像分类应用程序。然而，它们很容易受到对手攻击，这些攻击试图误导模型预测不正确的类别。我们对主要对手攻击模型的研究表明，它们都在设计中专门针对和利用神经网络结构。这种理解使我们提出了一个假设，即大多数经典的机器学习模型，如随机森林(RF)，对对手攻击模型是免疫的，因为它们根本不依赖于神经网络设计。我们对经典机器学习模型对抗流行的敌意攻击的实验研究支持这一假设。基于这一假设，我们提出了一种新的对抗性深度学习系统，该系统使用一个经典的机器学习模型作为辅助验证系统来补充图像分类中的主要深度学习模型。虽然次经典机器学习模型的输出精度较低，但仅用于验证目的，不影响主深度学习模型的输出精度，同时，当出现明显的失配时，可以有效地检测到对抗性攻击。基于CIFAR-100数据集的实验表明，我们提出的方法比目前最先进的对抗性防御系统具有更好的性能。



## **10. Near Optimal Adversarial Attack on UCB Bandits**

对UCB土匪的近最优敌意攻击 cs.LG

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2008.09312v5) [paper-pdf](http://arxiv.org/pdf/2008.09312v5)

**Authors**: Shiliang Zuo

**Abstract**: I study a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. I propose a novel attack strategy that manipulates a learner employing the UCB algorithm into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\widehat{O}(\sqrt{\log T})$, where $T$ is the number of rounds. I also prove the first lower bound on the cumulative attack cost. The lower bound matches the upper bound up to $O(\log \log T)$ factors, showing the proposed attack strategy to be near optimal.

摘要: 我研究了一个随机多臂强盗问题，其中报酬受到对抗性腐败的影响。提出了一种新的攻击策略，该策略利用UCB算法操纵学习者拉出一些非最优目标臂$T-o(T)$次，累积代价为$\widehat{O}(\Sqrt{\log T})$，其中$T$是轮数。我还证明了累积攻击成本的第一个下限。下界与最高可达$O(\LOG\LOG T)$因子的上界匹配，表明所提出的攻击策略接近最优。



## **11. Deception by Omission: Using Adversarial Missingness to Poison Causal Structure Learning**

遗漏欺骗：利用对抗性缺失毒化因果结构学习 cs.LG

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2305.20043v1) [paper-pdf](http://arxiv.org/pdf/2305.20043v1)

**Authors**: Deniz Koyuncu, Alex Gittens, Bülent Yener, Moti Yung

**Abstract**: Inference of causal structures from observational data is a key component of causal machine learning; in practice, this data may be incompletely observed. Prior work has demonstrated that adversarial perturbations of completely observed training data may be used to force the learning of inaccurate causal structural models (SCMs). However, when the data can be audited for correctness (e.g., it is crytographically signed by its source), this adversarial mechanism is invalidated. This work introduces a novel attack methodology wherein the adversary deceptively omits a portion of the true training data to bias the learned causal structures in a desired manner. Theoretically sound attack mechanisms are derived for the case of arbitrary SCMs, and a sample-efficient learning-based heuristic is given for Gaussian SCMs. Experimental validation of these approaches on real and synthetic data sets demonstrates the effectiveness of adversarial missingness attacks at deceiving popular causal structure learning algorithms.

摘要: 从观测数据推断因果结构是因果机器学习的关键组成部分；在实践中，这些数据可能不完全观察到。先前的工作已经证明，完全观察到的训练数据的对抗性扰动可能被用来强迫学习不准确的因果结构模型(SCM)。然而，当可以审计数据的正确性时(例如，数据是由其来源进行加密签名的)，则该对抗机制无效。这项工作介绍了一种新的攻击方法，其中对手欺骗性地省略了真实训练数据的一部分，以期望的方式偏向学习的因果结构。从理论上推导了任意SCM的攻击机制，并针对高斯SCM给出了一种样本有效的基于学习的启发式攻击方法。在真实和合成数据集上的实验验证表明，对抗性错位攻击在欺骗流行的因果结构学习算法方面是有效的。



## **12. IB-RAR: Information Bottleneck as Regularizer for Adversarial Robustness**

IB-RAR：作为对抗健壮性调节器的信息瓶颈 cs.LG

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2302.10896v2) [paper-pdf](http://arxiv.org/pdf/2302.10896v2)

**Authors**: Xiaoyun Xu, Guilherme Perin, Stjepan Picek

**Abstract**: In this paper, we propose a novel method, IB-RAR, which uses Information Bottleneck (IB) to strengthen adversarial robustness for both adversarial training and non-adversarial-trained methods. We first use the IB theory to build regularizers as learning objectives in the loss function. Then, we filter out unnecessary features of intermediate representation according to their mutual information (MI) with labels, as the network trained with IB provides easily distinguishable MI for its features. Experimental results show that our method can be naturally combined with adversarial training and provides consistently better accuracy on new adversarial examples. Our method improves the accuracy by an average of 3.07% against five adversarial attacks for the VGG16 network, trained with three adversarial training benchmarks and the CIFAR-10 dataset. In addition, our method also provides good robustness for undefended methods, such as training with cross-entropy loss only. Finally, in the absence of adversarial training, the VGG16 network trained using our method and the CIFAR-10 dataset reaches an accuracy of 35.86% against PGD examples, while using all layers reaches 25.61% accuracy.

摘要: 在本文中，我们提出了一种新的方法IB-RAR，它利用信息瓶颈(IB)来增强对抗性训练和非对抗性训练方法的对抗性健壮性。在损失函数中，我们首先使用IB理论来构造正则化子作为学习目标。然后，我们根据中间表示的带标签的互信息过滤掉不必要的特征，因为用带标签的互信息训练的网络为其特征提供了容易区分的互信息。实验结果表明，我们的方法可以自然地与对抗性训练相结合，并在新的对抗性样本上提供了一致更好的准确率。我们的方法对VGG16网络的五个对抗性攻击平均提高了3.07%的准确率，并使用三个对抗性训练基准和CIFAR-10数据集进行了训练。此外，我们的方法也为非防御方法提供了良好的稳健性，例如仅使用交叉熵损失进行训练。最后，在没有对抗性训练的情况下，使用我们的方法和CIFAR-10数据集训练的VGG16网络对PGD实例的准确率达到了35.86%，而使用所有层的准确率达到了25.61%。



## **13. Graph-based methods coupled with specific distributional distances for adversarial attack detection**

基于图的结合特定分布距离的攻击检测方法 cs.LG

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2306.00042v1) [paper-pdf](http://arxiv.org/pdf/2306.00042v1)

**Authors**: Dwight Nwaigwe, Lucrezia Carboni, Martial Mermillod, Sophie Achard, Michel Dojat

**Abstract**: Artificial neural networks are prone to being fooled by carefully perturbed inputs which cause an egregious misclassification. These \textit{adversarial} attacks have been the focus of extensive research. Likewise, there has been an abundance of research in ways to detect and defend against them. We introduce a novel approach of detection and interpretation of adversarial attacks from a graph perspective. For an image, benign or adversarial, we study how a neural network's architecture can induce an associated graph. We study this graph and introduce specific measures used to predict and interpret adversarial attacks. We show that graphs-based approaches help to investigate the inner workings of adversarial attacks.

摘要: 人工神经网络很容易被精心扰动的输入所愚弄，这会导致严重的错误分类。这些对抗性攻击一直是广泛研究的焦点。同样，在检测和防御它们的方法方面也进行了大量的研究。我们从图的角度介绍了一种新的检测和解释对抗性攻击的方法。对于一幅图像，无论是良性的还是对抗性的，我们研究了神经网络的体系结构是如何诱导出关联图的。我们研究了该图，并介绍了用于预测和解释对抗性攻击的具体方法。我们表明，基于图的方法有助于研究对抗性攻击的内部工作原理。



## **14. UKP-SQuARE: An Interactive Tool for Teaching Question Answering**

UKP-Square：一种交互式问答教学工具 cs.CL

Accepted by BEA workshop, ACL2023

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2305.19748v1) [paper-pdf](http://arxiv.org/pdf/2305.19748v1)

**Authors**: Haishuo Fang, Haritz Puerto, Iryna Gurevych

**Abstract**: The exponential growth of question answering (QA) has made it an indispensable topic in any Natural Language Processing (NLP) course. Additionally, the breadth of QA derived from this exponential growth makes it an ideal scenario for teaching related NLP topics such as information retrieval, explainability, and adversarial attacks among others. In this paper, we introduce UKP-SQuARE as a platform for QA education. This platform provides an interactive environment where students can run, compare, and analyze various QA models from different perspectives, such as general behavior, explainability, and robustness. Therefore, students can get a first-hand experience in different QA techniques during the class. Thanks to this, we propose a learner-centered approach for QA education in which students proactively learn theoretical concepts and acquire problem-solving skills through interactive exploration, experimentation, and practical assignments, rather than solely relying on traditional lectures. To evaluate the effectiveness of UKP-SQuARE in teaching scenarios, we adopted it in a postgraduate NLP course and surveyed the students after the course. Their positive feedback shows the platform's effectiveness in their course and invites a wider adoption.

摘要: 问答的指数级增长使其成为任何自然语言处理(NLP)课程中不可缺少的话题。此外，从这种指数增长中获得的QA的广度使其成为教授相关NLP主题的理想场景，如信息检索、可解释性和对抗性攻击等。在本文中，我们介绍了UKP-Square作为QA教育的平台。该平台提供了一个交互环境，学生可以从不同的角度运行、比较和分析各种QA模型，如一般行为、可解释性和健壮性。因此，学生可以在课堂上获得不同QA技术的第一手体验。有鉴于此，我们提出了以学习者为中心的QA教育方法，学生通过互动探索、实验和实践作业来主动学习理论概念并获得解决问题的技能，而不是仅仅依靠传统的讲授。为了评估UKP-Square在教学情景中的有效性，我们在一门研究生的NLP课程中采用了UKP-Square，并在课程结束后对学生进行了调查。他们的积极反馈显示了该平台在他们的课程中的有效性，并吸引了更广泛的采用。



## **15. Adversarial Detection: Attacking Object Detection in Real Time**

对抗性检测：攻击目标的实时检测 cs.AI

Accepted by IEEE Intelligent Vehicle Symposium, 2023

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2209.01962v5) [paper-pdf](http://arxiv.org/pdf/2209.01962v5)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: Intelligent robots rely on object detection models to perceive the environment. Following advances in deep learning security it has been revealed that object detection models are vulnerable to adversarial attacks. However, prior research primarily focuses on attacking static images or offline videos. Therefore, it is still unclear if such attacks could jeopardize real-world robotic applications in dynamic environments. This paper bridges this gap by presenting the first real-time online attack against object detection models. We devise three attacks that fabricate bounding boxes for nonexistent objects at desired locations. The attacks achieve a success rate of about 90% within about 20 iterations. The demo video is available at https://youtu.be/zJZ1aNlXsMU.

摘要: 智能机器人依靠物体检测模型来感知环境。随着深度学习安全性的进步，人们发现目标检测模型容易受到敌意攻击。然而，以往的研究主要集中在攻击静态图像或离线视频上。因此，目前尚不清楚此类攻击是否会危及动态环境中真实世界的机器人应用。本文通过提出第一个针对目标检测的实时在线攻击模型来弥补这一差距。我们设计了三种攻击，在所需位置为不存在的对象制造边界框。这些攻击在大约20次迭代内实现了约90%的成功率。该演示视频可在https://youtu.be/zJZ1aNlXsMU.上查看



## **16. Adversarial Driving: Attacking End-to-End Autonomous Driving**

对抗性驾驶：攻击型端到端自动驾驶 cs.CV

Accepted by IEEE Intelligent Vehicle Symposium, 2023

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2103.09151v7) [paper-pdf](http://arxiv.org/pdf/2103.09151v7)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: As research in deep neural networks advances, deep convolutional networks become promising for autonomous driving tasks. In particular, there is an emerging trend of employing end-to-end neural network models for autonomous driving. However, previous research has shown that deep neural network classifiers are vulnerable to adversarial attacks. While for regression tasks, the effect of adversarial attacks is not as well understood. In this research, we devise two white-box targeted attacks against end-to-end autonomous driving models. Our attacks manipulate the behavior of the autonomous driving system by perturbing the input image. In an average of 800 attacks with the same attack strength (epsilon=1), the image-specific and image-agnostic attack deviates the steering angle from the original output by 0.478 and 0.111, respectively, which is much stronger than random noises that only perturbs the steering angle by 0.002 (The steering angle ranges from [-1, 1]). Both attacks can be initiated in real-time on CPUs without employing GPUs. Demo video: https://youtu.be/I0i8uN2oOP0.

摘要: 随着深度神经网络研究的深入，深度卷积网络在自动驾驶任务中变得很有前途。特别是，使用端到端神经网络模型进行自动驾驶是一种新兴的趋势。然而，以往的研究表明，深度神经网络分类器容易受到敌意攻击。而对于回归任务，对抗性攻击的效果并没有被很好地理解。在本研究中，我们设计了两种针对端到端自动驾驶模型的白盒针对性攻击。我们的攻击通过干扰输入图像来操纵自动驾驶系统的行为。在相同攻击强度(epsilon=1)的平均800次攻击中，图像特定攻击和图像无关攻击使转向角与原始输出的偏差分别为0.478和0.111，远远强于仅扰动转向角0.002的随机噪声(转向角范围为[-1，1])。这两种攻击都可以在不使用GPU的情况下在CPU上实时发起。演示视频：https://youtu.be/I0i8uN2oOP0.



## **17. IBP Regularization for Verified Adversarial Robustness via Branch-and-Bound**

基于分枝定界的IBP正则化算法 cs.LG

ICML 2022 Workshop on Formal Verification of Machine Learning

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2206.14772v2) [paper-pdf](http://arxiv.org/pdf/2206.14772v2)

**Authors**: Alessandro De Palma, Rudy Bunel, Krishnamurthy Dvijotham, M. Pawan Kumar, Robert Stanforth

**Abstract**: Recent works have tried to increase the verifiability of adversarially trained networks by running the attacks over domains larger than the original perturbations and adding various regularization terms to the objective. However, these algorithms either underperform or require complex and expensive stage-wise training procedures, hindering their practical applicability. We present IBP-R, a novel verified training algorithm that is both simple and effective. IBP-R induces network verifiability by coupling adversarial attacks on enlarged domains with a regularization term, based on inexpensive interval bound propagation, that minimizes the gap between the non-convex verification problem and its approximations. By leveraging recent branch-and-bound frameworks, we show that IBP-R obtains state-of-the-art verified robustness-accuracy trade-offs for small perturbations on CIFAR-10 while training significantly faster than relevant previous work. Additionally, we present UPB, a novel branching strategy that, relying on a simple heuristic based on $\beta$-CROWN, reduces the cost of state-of-the-art branching algorithms while yielding splits of comparable quality.

摘要: 最近的工作试图通过在比原始扰动更大的域上运行攻击并在目标中添加各种正则化项来增加恶意训练网络的可验证性。然而，这些算法要么表现不佳，要么需要复杂而昂贵的阶段性训练过程，从而阻碍了它们的实际适用性。提出了一种简单有效的新的验证训练算法IBP-R。IBP-R通过将扩展域上的敌意攻击与基于廉价区间界传播的正则化项相结合来诱导网络可验证性，从而最小化非凸验证问题与其近似问题之间的差距。通过利用最近的分支定界框架，我们表明IBP-R在CIFAR-10上的小扰动下获得了经过验证的最先进的健壮性和准确性折衷，同时训练速度比相关以前的工作快得多。此外，我们提出了一种新的分支策略UPB，它依赖于基于$\beta$-Crown的简单启发式算法，在产生类似质量的分裂的同时，降低了最先进的分支算法的成本。



## **18. Adversarial Clean Label Backdoor Attacks and Defenses on Text Classification Systems**

文本分类系统的对抗性Clean Label后门攻击与防御 cs.CL

RepL4NLP 2023 at ACL 2023

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2305.19607v1) [paper-pdf](http://arxiv.org/pdf/2305.19607v1)

**Authors**: Ashim Gupta, Amrith Krishna

**Abstract**: Clean-label (CL) attack is a form of data poisoning attack where an adversary modifies only the textual input of the training data, without requiring access to the labeling function. CL attacks are relatively unexplored in NLP, as compared to label flipping (LF) attacks, where the latter additionally requires access to the labeling function as well. While CL attacks are more resilient to data sanitization and manual relabeling methods than LF attacks, they often demand as high as ten times the poisoning budget than LF attacks. In this work, we first introduce an Adversarial Clean Label attack which can adversarially perturb in-class training examples for poisoning the training set. We then show that an adversary can significantly bring down the data requirements for a CL attack, using the aforementioned approach, to as low as 20% of the data otherwise required. We then systematically benchmark and analyze a number of defense methods, for both LF and CL attacks, some previously employed solely for LF attacks in the textual domain and others adapted from computer vision. We find that text-specific defenses greatly vary in their effectiveness depending on their properties.

摘要: 干净标签(CL)攻击是数据中毒攻击的一种形式，其中对手只修改训练数据的文本输入，而不需要访问标签功能。与标签翻转(LF)攻击相比，CL攻击在NLP中相对未被探索，后者还需要访问标签功能。虽然CL攻击比LF攻击对数据清理和手动重新标记方法更具弹性，但它们通常需要比LF攻击高达10倍的中毒预算。在这项工作中，我们首先介绍了一种对抗性的Clean Label攻击，它可以对抗性地扰乱类内训练样本以毒化训练集。然后，我们展示了对手可以使用前述方法将CL攻击的数据需求显著降低到其他情况下所需数据的20%。然后，我们系统地对一些防御方法进行了基准测试和分析，这些方法既适用于低频攻击，也适用于CL攻击，其中一些以前仅用于文本域中的低频攻击，另一些则改编自计算机视觉。我们发现，特定于文本的防御在有效性上有很大的差异，这取决于它们的性质。



## **19. Exploring the Vulnerabilities of Machine Learning and Quantum Machine Learning to Adversarial Attacks using a Malware Dataset: A Comparative Analysis**

利用恶意软件数据集探讨机器学习和量子机器学习对敌意攻击的脆弱性：比较分析 cs.LG

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2305.19593v1) [paper-pdf](http://arxiv.org/pdf/2305.19593v1)

**Authors**: Mst Shapna Akter, Hossain Shahriar, Iysa Iqbal, MD Hossain, M. A. Karim, Victor Clincy, Razvan Voicu

**Abstract**: The burgeoning fields of machine learning (ML) and quantum machine learning (QML) have shown remarkable potential in tackling complex problems across various domains. However, their susceptibility to adversarial attacks raises concerns when deploying these systems in security sensitive applications. In this study, we present a comparative analysis of the vulnerability of ML and QML models, specifically conventional neural networks (NN) and quantum neural networks (QNN), to adversarial attacks using a malware dataset. We utilize a software supply chain attack dataset known as ClaMP and develop two distinct models for QNN and NN, employing Pennylane for quantum implementations and TensorFlow and Keras for traditional implementations. Our methodology involves crafting adversarial samples by introducing random noise to a small portion of the dataset and evaluating the impact on the models performance using accuracy, precision, recall, and F1 score metrics. Based on our observations, both ML and QML models exhibit vulnerability to adversarial attacks. While the QNNs accuracy decreases more significantly compared to the NN after the attack, it demonstrates better performance in terms of precision and recall, indicating higher resilience in detecting true positives under adversarial conditions. We also find that adversarial samples crafted for one model type can impair the performance of the other, highlighting the need for robust defense mechanisms. Our study serves as a foundation for future research focused on enhancing the security and resilience of ML and QML models, particularly QNN, given its recent advancements. A more extensive range of experiments will be conducted to better understand the performance and robustness of both models in the face of adversarial attacks.

摘要: 机器学习(ML)和量子机器学习(QML)这两个新兴领域在处理不同领域的复杂问题方面显示出了巨大的潜力。然而，当在安全敏感应用程序中部署这些系统时，它们容易受到对手攻击，这引起了人们的担忧。在这项研究中，我们比较分析了ML和QML模型，特别是传统神经网络(NN)和量子神经网络(QNN)在使用恶意软件数据集进行攻击时的脆弱性。我们利用一个名为CLAMP的软件供应链攻击数据集，为QNN和NN开发了两个不同的模型，使用Pennylane进行量子实现，使用TensorFlow和Kera进行传统实现。我们的方法包括通过向数据集的一小部分引入随机噪声来制作敌意样本，并使用准确性、精确度、召回率和F1分数度量来评估对模型性能的影响。根据我们的观察，ML和QML模型都表现出对对手攻击的脆弱性。虽然QNN的准确率在攻击后比NN下降得更明显，但它在准确率和召回率方面表现出了更好的性能，表明在对抗条件下检测真正阳性的能力更强。我们还发现，为一种模型类型制作的对抗性样本可能会损害另一种模型的性能，这突显了强大防御机制的必要性。我们的研究为未来的研究奠定了基础，这些研究的重点是提高ML和QML模型的安全性和弹性，特别是QNN，因为它最近取得了进展。将进行更广泛的实验，以更好地了解这两种模型在面对对手攻击时的性能和稳健性。



## **20. Incremental Randomized Smoothing Certification**

增量式随机平滑证明 cs.LG

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2305.19521v1) [paper-pdf](http://arxiv.org/pdf/2305.19521v1)

**Authors**: Shubham Ugare, Tarun Suresh, Debangshu Banerjee, Gagandeep Singh, Sasa Misailovic

**Abstract**: Randomized smoothing-based certification is an effective approach for obtaining robustness certificates of deep neural networks (DNNs) against adversarial attacks. This method constructs a smoothed DNN model and certifies its robustness through statistical sampling, but it is computationally expensive, especially when certifying with a large number of samples. Furthermore, when the smoothed model is modified (e.g., quantized or pruned), certification guarantees may not hold for the modified DNN, and recertifying from scratch can be prohibitively expensive.   We present the first approach for incremental robustness certification for randomized smoothing, IRS. We show how to reuse the certification guarantees for the original smoothed model to certify an approximated model with very few samples. IRS significantly reduces the computational cost of certifying modified DNNs while maintaining strong robustness guarantees. We experimentally demonstrate the effectiveness of our approach, showing up to 3x certification speedup over the certification that applies randomized smoothing of the approximate model from scratch.

摘要: 基于随机平滑的认证是获得深层神经网络抗攻击健壮性证书的有效方法。该方法构造了一个平滑的DNN模型，并通过统计抽样验证了其稳健性，但其计算量很大，特别是在需要大量样本的情况下。此外，当修改平滑的模型(例如，量化或修剪)时，认证保证可能不适用于修改的DNN，并且从头开始重新认证可能昂贵得令人望而却步。我们提出了第一种随机平滑的增量式稳健性证明方法--IRS。我们展示了如何重用对原始平滑模型的证明保证来证明具有很少样本的近似模型。IRS在保持较强的健壮性保证的同时，显著降低了证明修改的DNN的计算代价。我们在实验中展示了我们方法的有效性，与从头开始应用随机平滑近似模型的认证相比，认证加速高达3倍。



## **21. Pareto Regret Analyses in Multi-objective Multi-armed Bandit**

多目标多臂匪徒的帕累托悔恨分析 cs.LG

19 pages; accepted at ICML 2023 and to be published in Proceedings of  Machine Learning Research (PMLR)

**SubmitDate**: 2023-05-30    [abs](http://arxiv.org/abs/2212.00884v2) [paper-pdf](http://arxiv.org/pdf/2212.00884v2)

**Authors**: Mengfan Xu, Diego Klabjan

**Abstract**: We study Pareto optimality in multi-objective multi-armed bandit by providing a formulation of adversarial multi-objective multi-armed bandit and defining its Pareto regrets that can be applied to both stochastic and adversarial settings. The regrets do not rely on any scalarization functions and reflect Pareto optimality compared to scalarized regrets. We also present new algorithms assuming both with and without prior information of the multi-objective multi-armed bandit setting. The algorithms are shown optimal in adversarial settings and nearly optimal up to a logarithmic factor in stochastic settings simultaneously by our established upper bounds and lower bounds on Pareto regrets. Moreover, the lower bound analyses show that the new regrets are consistent with the existing Pareto regret for stochastic settings and extend an adversarial attack mechanism from bandit to the multi-objective one.

摘要: 研究了多目标多臂土匪的Pareto最优性，给出了一个对抗性多目标多臂土匪模型，并定义了它的Pareto遗憾，它既适用于随机环境，也适用于对抗性环境。遗憾不依赖于任何标量化函数，并且与标量化的遗憾相比，反映了帕累托最优。我们还提出了新的算法，假设有和没有多目标多臂匪徒设置的先验信息。通过我们建立的帕累托遗憾的上界和下界，这些算法在对抗性环境中被证明是最优的，在随机环境中几乎是最优的，直到对数倍。此外，下界分析表明，新的遗憾与现有的随机设置下的帕累托遗憾是一致的，并将对抗性攻击机制从强盗扩展到多目标机制。



## **22. Adversarial Attacks on Online Learning to Rank with Stochastic Click Models**

基于随机点击模型的在线学习排名的对抗性攻击 cs.LG

**SubmitDate**: 2023-05-30    [abs](http://arxiv.org/abs/2305.19218v1) [paper-pdf](http://arxiv.org/pdf/2305.19218v1)

**Authors**: Zichen Wang, Rishab Balasubramanian, Hui Yuan, Chenyu Song, Mengdi Wang, Huazheng Wang

**Abstract**: We propose the first study of adversarial attacks on online learning to rank. The goal of the adversary is to misguide the online learning to rank algorithm to place the target item on top of the ranking list linear times to time horizon $T$ with a sublinear attack cost. We propose generalized list poisoning attacks that perturb the ranking list presented to the user. This strategy can efficiently attack any no-regret ranker in general stochastic click models. Furthermore, we propose a click poisoning-based strategy named attack-then-quit that can efficiently attack two representative OLTR algorithms for stochastic click models. We theoretically analyze the success and cost upper bound of the two proposed methods. Experimental results based on synthetic and real-world data further validate the effectiveness and cost-efficiency of the proposed attack strategies.

摘要: 我们首次提出了对抗性攻击对在线学习进行排名的研究。对手的目标是误导在线学习排名算法，以次线性攻击成本将目标项目放在排名列表的顶部线性时间到时间范围$T$。我们提出了广义列表中毒攻击，扰乱了呈现给用户的排名列表。该策略可以有效地攻击一般随机点击模型中的任何无遗憾排名者。在此基础上，提出了一种基于点击中毒的攻击退出策略，该策略可以有效地攻击随机点击模型中的两种典型的OLTR算法。我们从理论上分析了这两种方法的成功率和成本上限。基于人工数据和真实数据的实验结果进一步验证了所提出的攻击策略的有效性和成本效益。



## **23. Learning Robust Kernel Ensembles with Kernel Average Pooling**

利用核平均池化学习稳健核集成 cs.LG

**SubmitDate**: 2023-05-30    [abs](http://arxiv.org/abs/2210.00062v2) [paper-pdf](http://arxiv.org/pdf/2210.00062v2)

**Authors**: Pouya Bashivan, Adam Ibrahim, Amirozhan Dehghani, Yifei Ren

**Abstract**: Model ensembles have long been used in machine learning to reduce the variance in individual model predictions, making them more robust to input perturbations. Pseudo-ensemble methods like dropout have also been commonly used in deep learning models to improve generalization. However, the application of these techniques to improve neural networks' robustness against input perturbations remains underexplored. We introduce Kernel Average Pooling (KAP), a neural network building block that applies the mean filter along the kernel dimension of the layer activation tensor. We show that ensembles of kernels with similar functionality naturally emerge in convolutional neural networks equipped with KAP and trained with backpropagation. Moreover, we show that when trained on inputs perturbed with additive Gaussian noise, KAP models are remarkably robust against various forms of adversarial attacks. Empirical evaluations on CIFAR10, CIFAR100, TinyImagenet, and Imagenet datasets show substantial improvements in robustness against strong adversarial attacks such as AutoAttack without training on any adversarial examples.

摘要: 长期以来，模型集成一直被用于机器学习，以减少单个模型预测中的方差，使它们对输入扰动更具鲁棒性。丢弃等伪集合法也被广泛用于深度学习模型中，以提高泛化能力。然而，应用这些技术来提高神经网络对输入扰动的稳健性仍然没有得到充分的探索。我们介绍了核平均池(KAP)，这是一种神经网络构建块，它沿层激活张量的核维度应用平均过滤器。我们证明了具有相似功能的核函数集成自然地出现在配备KAP并用反向传播训练的卷积神经网络中。此外，我们还表明，当对带有加性高斯噪声扰动的输入进行训练时，KAP模型对各种形式的对抗性攻击具有显著的鲁棒性。在CIFAR10、CIFAR100、TinyImagenet和Imagenet数据集上的经验评估表明，在没有对任何对抗性示例进行训练的情况下，对AutoAttack等强对抗性攻击的稳健性有了显著的改善。



## **24. Does CLIP Know My Face?**

小夹子认得我的脸吗？ cs.LG

15 pages, 6 figures

**SubmitDate**: 2023-05-30    [abs](http://arxiv.org/abs/2209.07341v3) [paper-pdf](http://arxiv.org/pdf/2209.07341v3)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Manuel Brack, Felix Friedrich, Patrick Schramowski, Kristian Kersting

**Abstract**: With the rise of deep learning in various applications, privacy concerns around the protection of training data has become a critical area of research. Whereas prior studies have focused on privacy risks in single-modal models, we introduce a novel method to assess privacy for multi-modal models, specifically vision-language models like CLIP. The proposed Identity Inference Attack (IDIA) reveals whether an individual was included in the training data by querying the model with images of the same person. Letting the model choose from a wide variety of possible text labels, the model reveals whether it recognizes the person and, therefore, was used for training. Our large-scale experiments on CLIP demonstrate that individuals used for training can be identified with very high accuracy. We confirm that the model has learned to associate names with depicted individuals, implying the existence of sensitive information that can be extracted by adversaries. Our results highlight the need for stronger privacy protection in large-scale models and suggest that IDIAs can be used to prove the unauthorized use of data for training and to enforce privacy laws.

摘要: 随着深度学习在各种应用中的兴起，围绕训练数据保护的隐私问题已经成为一个关键的研究领域。鉴于以往的研究主要集中于单通道模型中的隐私风险，我们引入了一种新的方法来评估多通道模型的隐私，特别是像CLIP这样的视觉语言模型。提出的身份推断攻击(IDIA)通过用同一人的图像查询模型来揭示该人是否包括在训练数据中。让模型从各种各样的可能的文本标签中进行选择，该模型显示它是否识别出这个人，因此，它被用于训练。我们在CLIP上的大规模实验表明，用于训练的个体可以非常准确地识别。我们确认，该模型已经学会了将姓名与所描述的个人相关联，这意味着存在可被对手提取的敏感信息。我们的结果强调了在大规模模型中加强隐私保护的必要性，并建议可以使用IDIA来证明未经授权使用数据进行培训和执行隐私法。



## **25. Pseudo-Siamese Network based Timbre-reserved Black-box Adversarial Attack in Speaker Identification**

说话人辨认中基于伪暹罗网络的保留音色黑盒对抗攻击 cs.SD

5 pages

**SubmitDate**: 2023-05-30    [abs](http://arxiv.org/abs/2305.19020v1) [paper-pdf](http://arxiv.org/pdf/2305.19020v1)

**Authors**: Qing Wang, Jixun Yao, Ziqian Wang, Pengcheng Guo, Lei Xie

**Abstract**: In this study, we propose a timbre-reserved adversarial attack approach for speaker identification (SID) to not only exploit the weakness of the SID model but also preserve the timbre of the target speaker in a black-box attack setting. Particularly, we generate timbre-reserved fake audio by adding an adversarial constraint during the training of the voice conversion model. Then, we leverage a pseudo-Siamese network architecture to learn from the black-box SID model constraining both intrinsic similarity and structural similarity simultaneously. The intrinsic similarity loss is to learn an intrinsic invariance, while the structural similarity loss is to ensure that the substitute SID model shares a similar decision boundary to the fixed black-box SID model. The substitute model can be used as a proxy to generate timbre-reserved fake audio for attacking. Experimental results on the Audio Deepfake Detection (ADD) challenge dataset indicate that the attack success rate of our proposed approach yields up to 60.58% and 55.38% in the white-box and black-box scenarios, respectively, and can deceive both human beings and machines.

摘要: 在本研究中，我们提出一种保留音色的对抗性攻击方法(SID)，既能利用SID模型的弱点，又能在黑盒攻击环境下保持目标说话人的音色。具体地说，在语音转换模型的训练过程中，通过添加对抗性约束来生成保留音色的伪音频。然后，我们利用伪暹罗网络体系结构来学习同时约束本征相似性和结构相似性的黑盒SID模型。本质相似损失是为了学习固有不变性，而结构相似损失是为了确保替代SID模型与固定黑盒SID模型共享相似的决策边界。该替代模型可以作为生成预留音质的伪音频的代理，用于攻击。在音频深伪检测(ADD)挑战数据集上的实验结果表明，该方法在白盒和黑盒场景下的攻击成功率分别达到60.58%和55.38%，可以同时欺骗人类和机器。



## **26. Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start**

低水平收缩的两层优化：无热启动的最优样本复杂性 stat.ML

Published in JMLR. Code at https://github.com/CSML-IIT-UCL/bioptexps

**SubmitDate**: 2023-05-30    [abs](http://arxiv.org/abs/2202.03397v3) [paper-pdf](http://arxiv.org/pdf/2202.03397v3)

**Authors**: Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

**Abstract**: We analyse a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, equilibrium models, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower-level problem, i.e. they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. However, there are situations, e.g., meta learning and equilibrium models, in which the warm-start procedure is not well-suited or ineffective. In this work we show that without warm-start, it is still possible to achieve order-wise (near) optimal sample complexity. In particular, we propose a simple method which uses (stochastic) fixed point iterations at the lower-level and projected inexact gradient descent at the upper-level, that reaches an $\epsilon$-stationary point using $O(\epsilon^{-2})$ and $\tilde{O}(\epsilon^{-1})$ samples for the stochastic and the deterministic setting, respectively. Finally, compared to methods using warm-start, our approach yields a simpler analysis that does not need to study the coupled interactions between the upper-level and lower-level iterates.

摘要: 我们分析了一类一般的两层问题，其中上层问题在于光滑目标函数的最小化，下层问题是寻找光滑压缩映射的不动点。这类问题包括元学习、均衡模型、超参数优化和数据中毒攻击等实例。最近的一些工作提出了暖启动下层问题的算法，即使用先前的下层近似解作为下层求解器的起始点。这种热启动过程允许人们在随机和确定性设置下改善样本复杂性，在某些情况下实现顺序最优的样本复杂性。然而，在一些情况下，例如元学习和平衡模型，热启动程序不是很适合或无效的。在这项工作中，我们证明了在没有热启动的情况下，仍然有可能达到阶次(接近)最优的样本复杂性。特别地，我们提出了一种简单的方法，它在下层使用(随机)不动点迭代，在上层使用投影的不精确梯度下降，在随机和确定设置下分别使用$O(epsilon^{-2})$和$tilde{O}(epsilon^{-1})$样本达到$-epsilon$-固定点。最后，与使用热启动的方法相比，我们的方法产生了更简单的分析，不需要研究上层和下层迭代之间的耦合作用。



## **27. Ring Signature from Bonsai Tree: How to Preserve the Long-Term Anonymity**

盆景树上的环签名：如何保持长期匿名性 cs.CR

**SubmitDate**: 2023-05-30    [abs](http://arxiv.org/abs/2305.16135v2) [paper-pdf](http://arxiv.org/pdf/2305.16135v2)

**Authors**: Mingxing Hu, Yunhong Zhou

**Abstract**: Signer-anonymity is the central feature of ring signatures, which enable a user to sign messages on behalf of an arbitrary set of users, called the ring, without revealing exactly which member of the ring actually generated the signature. Strong and long-term signer-anonymity is a reassuring guarantee for users who are hesitant to leak a secret, especially if the consequences of identification are dire in certain scenarios such as whistleblowing. The notion of \textit{unconditional anonymity}, which protects signer-anonymity even against an infinitely powerful adversary, is considered for ring signatures that aim to achieve long-term signer-anonymity. However, the existing lattice-based works that consider the unconditional anonymity notion did not strictly capture the security requirements imposed in practice, this leads to a realistic attack on signer-anonymity.   In this paper, we present a realistic attack on the unconditional anonymity of ring signatures, and formalize the unconditional anonymity model to strictly capture it. We then propose a lattice-based ring signature construction with unconditional anonymity by leveraging bonsai tree mechanism. Finally, we prove the security in the standard model and demonstrate the unconditional anonymity through both theoretical proof and practical experiments.

摘要: 签名者匿名性是环签名的核心特征，它使用户能够代表称为环的任意一组用户对消息进行签名，而不会确切地透露环中的哪个成员实际生成了签名。强大和长期的签名者匿名性对于不愿泄露机密的用户来说是一种令人放心的保证，特别是如果在某些情况下身份识别的后果是可怕的，比如告密。无条件匿名的概念可以保护签名者的匿名性，甚至可以保护签名者的匿名性，即使是针对无限强大的对手，被认为是为了实现长期签名者的匿名。然而，现有的基于格的工作考虑了无条件匿名性的概念，没有严格地捕捉到实践中强加的安全需求，这导致了对签名者匿名性的现实攻击。本文对环签名的无条件匿名性提出了一种现实的攻击，并将无条件匿名性模型形式化以严格捕获它。然后利用盆景树机制，提出了一种无条件匿名性的格型环签名方案。最后，通过理论证明和实际实验证明了标准模型下的安全性和无条件匿名性。



## **28. Exploring the Unprecedented Privacy Risks of the Metaverse**

探索Metverse前所未有的隐私风险 cs.CR

**SubmitDate**: 2023-05-30    [abs](http://arxiv.org/abs/2207.13176v2) [paper-pdf](http://arxiv.org/pdf/2207.13176v2)

**Authors**: Vivek Nair, Gonzalo Munilla Garrido, Dawn Song

**Abstract**: Thirty study participants playtested an innocent-looking "escape room" game in virtual reality (VR). Behind the scenes, an adversarial program had accurately inferred over 25 personal data attributes, from anthropometrics like height and wingspan to demographics like age and gender, within just a few minutes of gameplay. As notoriously data-hungry companies become increasingly involved in VR development, this experimental scenario may soon represent a typical VR user experience. While virtual telepresence applications (and the so-called "metaverse") have recently received increased attention and investment from major tech firms, these environments remain relatively under-studied from a security and privacy standpoint. In this work, we illustrate how VR attackers can covertly ascertain dozens of personal data attributes from seemingly-anonymous users of popular metaverse applications like VRChat. These attackers can be as simple as other VR users without special privilege, and the potential scale and scope of this data collection far exceed what is feasible within traditional mobile and web applications. We aim to shed light on the unique privacy risks of the metaverse, and provide the first holistic framework for understanding intrusive data harvesting attacks in these emerging VR ecosystems.

摘要: 30名研究参与者在虚拟现实(VR)中玩了一个看起来很无辜的“逃生室”游戏。在幕后，一个对抗性的程序在玩游戏的短短几分钟内就准确地推断出了25个人的数据属性，从身高和翼展等人体测量数据到年龄和性别等人口统计数据。随着以渴望数据著称的公司越来越多地参与到VR开发中来，这种实验场景可能很快就会代表一种典型的VR用户体验。虽然虚拟远程呈现应用(以及所谓的虚拟现实)最近得到了主要科技公司越来越多的关注和投资，但从安全和隐私的角度来看，这些环境的研究仍然相对较少。在这项工作中，我们展示了VR攻击者如何从VRChat等流行虚拟世界应用程序的看似匿名的用户那里秘密确定数十个个人数据属性。这些攻击者可以像其他没有特殊权限的VR用户一样简单，而且这种数据收集的潜在规模和范围远远超出了传统移动和网络应用程序中的可行范围。我们的目标是阐明虚拟世界独特的隐私风险，并提供第一个整体框架，以了解这些新兴的虚拟现实生态系统中的侵入性数据收集攻击。



## **29. UMD: Unsupervised Model Detection for X2X Backdoor Attacks**

UMD：X2X后门攻击的无监督模型检测 cs.LG

ICML 2023

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.18651v1) [paper-pdf](http://arxiv.org/pdf/2305.18651v1)

**Authors**: Zhen Xiang, Zidi Xiong, Bo Li

**Abstract**: Backdoor (Trojan) attack is a common threat to deep neural networks, where samples from one or more source classes embedded with a backdoor trigger will be misclassified to adversarial target classes. Existing methods for detecting whether a classifier is backdoor attacked are mostly designed for attacks with a single adversarial target (e.g., all-to-one attack). To the best of our knowledge, without supervision, no existing methods can effectively address the more general X2X attack with an arbitrary number of source classes, each paired with an arbitrary target class. In this paper, we propose UMD, the first Unsupervised Model Detection method that effectively detects X2X backdoor attacks via a joint inference of the adversarial (source, target) class pairs. In particular, we first define a novel transferability statistic to measure and select a subset of putative backdoor class pairs based on a proposed clustering approach. Then, these selected class pairs are jointly assessed based on an aggregation of their reverse-engineered trigger size for detection inference, using a robust and unsupervised anomaly detector we proposed. We conduct comprehensive evaluations on CIFAR-10, GTSRB, and Imagenette dataset, and show that our unsupervised UMD outperforms SOTA detectors (even with supervision) by 17%, 4%, and 8%, respectively, in terms of the detection accuracy against diverse X2X attacks. We also show the strong detection performance of UMD against several strong adaptive attacks.

摘要: 后门(特洛伊木马)攻击是深度神经网络的常见威胁，来自嵌入后门触发器的一个或多个源类的样本将被错误分类为对抗性目标类。现有的检测分类器是否被后门攻击的方法大多是针对单个敌对目标的攻击而设计的(例如，All-to-One攻击)。就我们所知，在没有监督的情况下，没有任何现有方法可以有效地应对具有任意数量的源类的更通用的X2X攻击，每个源类都与任意的目标类配对。在本文中，我们提出了第一种无监督模型检测方法UMD，它通过联合推理对手(源、目标)类对来有效地检测X2X后门攻击。特别是，我们首先定义了一种新的可转移性统计量来度量和选择基于所提出的聚类方法的假定的后门类对的子集。然后，使用我们提出的健壮和无监督的异常检测器，基于它们的反向工程触发大小的聚集来联合评估这些选择的类对以用于检测推理。我们在CIFAR-10、GTSRB和Imagenette数据集上进行了综合评估，结果表明，在对各种X2X攻击的检测准确率方面，我们的无监督UMD分别比SOTA检测器(即使有监督)提高了17%、4%和8%。我们还展示了UMD对几种强自适应攻击的强检测性能。



## **30. Securing Cloud File Systems using Shielded Execution**

使用屏蔽执行保护云文件系统 cs.CR

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.18639v1) [paper-pdf](http://arxiv.org/pdf/2305.18639v1)

**Authors**: Quinn Burke, Yohan Beugin, Blaine Hoak, Rachel King, Eric Pauley, Ryan Sheatsley, Mingli Yu, Ting He, Thomas La Porta, Patrick McDaniel

**Abstract**: Cloud file systems offer organizations a scalable and reliable file storage solution. However, cloud file systems have become prime targets for adversaries, and traditional designs are not equipped to protect organizations against the myriad of attacks that may be initiated by a malicious cloud provider, co-tenant, or end-client. Recently proposed designs leveraging cryptographic techniques and trusted execution environments (TEEs) still force organizations to make undesirable trade-offs, consequently leading to either security, functional, or performance limitations. In this paper, we introduce TFS, a cloud file system that leverages the security capabilities provided by TEEs to bootstrap new security protocols that meet real-world security, functional, and performance requirements. Through extensive security and performance analyses, we show that TFS can ensure stronger security guarantees while still providing practical utility and performance w.r.t. state-of-the-art systems; compared to the widely-used NFS, TFS achieves up to 2.1X speedups across micro-benchmarks and incurs <1X overhead for most macro-benchmark workloads. TFS demonstrates that organizations need not sacrifice file system security to embrace the functional and performance advantages of outsourcing.

摘要: 云文件系统为组织提供了可扩展且可靠的文件存储解决方案。然而，云文件系统已成为对手的主要目标，传统设计无法保护组织免受恶意云提供商、联合租户或终端客户端可能发起的无数攻击。最近提出的利用加密技术和可信执行环境(TEE)的设计仍然迫使组织做出不希望看到的权衡，从而导致安全、功能或性能限制。在本文中，我们介绍了TFS，这是一个云文件系统，它利用TES提供的安全功能来引导新的安全协议，以满足现实世界的安全、功能和性能要求。通过广泛的安全和性能分析，我们证明了TFS在保证更强的安全保证的同时，仍然提供了实用的实用性和性能。最先进的系统；与广泛使用的NFS相比，TFS在微观基准测试中实现了高达2.1倍的加速，而大多数宏观基准测试工作负载的开销不到1倍。TFS证明，组织不需要牺牲文件系统安全性来享受外包的功能和性能优势。



## **31. Exploiting Explainability to Design Adversarial Attacks and Evaluate Attack Resilience in Hate-Speech Detection Models**

利用可理解性在仇恨言语检测模型中设计对抗性攻击和评估攻击弹性 cs.CL

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.18585v1) [paper-pdf](http://arxiv.org/pdf/2305.18585v1)

**Authors**: Pranath Reddy Kumbam, Sohaib Uddin Syed, Prashanth Thamminedi, Suhas Harish, Ian Perera, Bonnie J. Dorr

**Abstract**: The advent of social media has given rise to numerous ethical challenges, with hate speech among the most significant concerns. Researchers are attempting to tackle this problem by leveraging hate-speech detection and employing language models to automatically moderate content and promote civil discourse. Unfortunately, recent studies have revealed that hate-speech detection systems can be misled by adversarial attacks, raising concerns about their resilience. While previous research has separately addressed the robustness of these models under adversarial attacks and their interpretability, there has been no comprehensive study exploring their intersection. The novelty of our work lies in combining these two critical aspects, leveraging interpretability to identify potential vulnerabilities and enabling the design of targeted adversarial attacks. We present a comprehensive and comparative analysis of adversarial robustness exhibited by various hate-speech detection models. Our study evaluates the resilience of these models against adversarial attacks using explainability techniques. To gain insights into the models' decision-making processes, we employ the Local Interpretable Model-agnostic Explanations (LIME) framework. Based on the explainability results obtained by LIME, we devise and execute targeted attacks on the text by leveraging the TextAttack tool. Our findings enhance the understanding of the vulnerabilities and strengths exhibited by state-of-the-art hate-speech detection models. This work underscores the importance of incorporating explainability in the development and evaluation of such models to enhance their resilience against adversarial attacks. Ultimately, this work paves the way for creating more robust and reliable hate-speech detection systems, fostering safer online environments and promoting ethical discourse on social media platforms.

摘要: 社交媒体的出现引发了许多道德挑战，仇恨言论是最重要的担忧之一。研究人员正试图通过利用仇恨言论检测和使用语言模型来自动调节内容和促进公民话语来解决这个问题。不幸的是，最近的研究表明，仇恨言论检测系统可能会被敌意攻击误导，这引发了人们对其弹性的担忧。虽然以前的研究已经分别讨论了这些模型在对抗性攻击下的稳健性及其可解释性，但还没有全面的研究来探索它们的交集。我们工作的新颖性在于将这两个关键方面结合起来，利用可解释性来确定潜在的漏洞，并能够设计有针对性的对抗性攻击。我们对各种仇恨言语检测模型所表现出的敌意稳健性进行了全面和比较的分析。我们的研究使用可解释性技术评估了这些模型对对手攻击的恢复能力。为了深入了解模型的决策过程，我们采用了局部可解释模型不可知性解释(LIME)框架。基于LIME获得的可解释性结果，我们利用TextAttack工具设计并执行了针对文本的有针对性的攻击。我们的发现加强了对最先进的仇恨言语检测模型所表现出的脆弱性和优势的理解。这项工作强调了将可解释性纳入这类模型的发展和评价的重要性，以增强它们对敌方攻击的复原力。归根结底，这项工作为创建更强大和可靠的仇恨言论检测系统、培育更安全的在线环境和在社交媒体平台上促进道德话语铺平了道路。



## **32. Robust Lipschitz Bandits to Adversarial Corruptions**

对抗腐败的强健Lipschitz Bandits cs.LG

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.18543v1) [paper-pdf](http://arxiv.org/pdf/2305.18543v1)

**Authors**: Yue Kang, Cho-Jui Hsieh, Thomas C. M. Lee

**Abstract**: Lipschitz bandit is a variant of stochastic bandits that deals with a continuous arm set defined on a metric space, where the reward function is subject to a Lipschitz constraint. In this paper, we introduce a new problem of Lipschitz bandits in the presence of adversarial corruptions where an adaptive adversary corrupts the stochastic rewards up to a total budget $C$. The budget is measured by the sum of corruption levels across the time horizon $T$. We consider both weak and strong adversaries, where the weak adversary is unaware of the current action before the attack, while the strong one can observe it. Our work presents the first line of robust Lipschitz bandit algorithms that can achieve sub-linear regret under both types of adversary, even when the total budget of corruption $C$ is unrevealed to the agent. We provide a lower bound under each type of adversary, and show that our algorithm is optimal under the strong case. Finally, we conduct experiments to illustrate the effectiveness of our algorithms against two classic kinds of attacks.

摘要: Lipschitz Bandit是随机强盗的变种，它处理定义在度量空间上的连续臂集，其中奖励函数受Lipschitz约束。在这篇文章中，我们引入了一个新的存在对抗性腐败的Lipschitz强盗问题，其中一个自适应的敌对者破坏了总预算为$C$的随机报酬。预算是通过整个时间范围内腐败程度的总和新台币来衡量的。我们既考虑弱对手，也考虑强对手，其中弱对手在攻击前不知道当前的行动，而强对手可以观察到。我们的工作提出了第一行稳健的Lipschitz强盗算法，即使在代理未透露腐败$C$的总预算时，该算法也可以在这两种类型的对手下实现次线性遗憾。在每种类型的对手下，我们给出了一个下界，并证明了我们的算法在强情形下是最优的。最后，通过实验验证了该算法对两种经典攻击的有效性。



## **33. BITE: Textual Backdoor Attacks with Iterative Trigger Injection**

BITE：使用迭代触发器注入的文本后门攻击 cs.CL

Accepted to ACL 2023

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2205.12700v3) [paper-pdf](http://arxiv.org/pdf/2205.12700v3)

**Authors**: Jun Yan, Vansh Gupta, Xiang Ren

**Abstract**: Backdoor attacks have become an emerging threat to NLP systems. By providing poisoned training data, the adversary can embed a "backdoor" into the victim model, which allows input instances satisfying certain textual patterns (e.g., containing a keyword) to be predicted as a target label of the adversary's choice. In this paper, we demonstrate that it is possible to design a backdoor attack that is both stealthy (i.e., hard to notice) and effective (i.e., has a high attack success rate). We propose BITE, a backdoor attack that poisons the training data to establish strong correlations between the target label and a set of "trigger words". These trigger words are iteratively identified and injected into the target-label instances through natural word-level perturbations. The poisoned training data instruct the victim model to predict the target label on inputs containing trigger words, forming the backdoor. Experiments on four text classification datasets show that our proposed attack is significantly more effective than baseline methods while maintaining decent stealthiness, raising alarm on the usage of untrusted training data. We further propose a defense method named DeBITE based on potential trigger word removal, which outperforms existing methods in defending against BITE and generalizes well to handling other backdoor attacks.

摘要: 后门攻击已成为对NLP系统的新威胁。通过提供有毒的训练数据，敌手可以在受害者模型中嵌入“后门”，这允许满足某些文本模式(例如，包含关键字)的输入实例被预测为敌手选择的目标标签。在这篇文章中，我们证明了设计一种既隐蔽(即难以察觉)又有效(即具有高攻击成功率)的后门攻击是可能的。我们提出了BITE，一种毒化训练数据的后门攻击，以建立目标标签和一组“触发词”之间的强关联。这些触发词被迭代地识别并通过自然的词级扰动注入到目标标签实例中。有毒的训练数据指示受害者模型预测包含触发词的输入上的目标标签，从而形成后门。在四个文本分类数据集上的实验表明，我们提出的攻击方法比基准方法更有效，同时保持了良好的隐蔽性，并对不可信训练数据的使用发出了警报。在此基础上，提出了一种基于潜在触发字移除的防御方法DeBITE，该方法在防御BITE攻击方面优于已有方法，并能很好地推广到应对其他后门攻击。



## **34. Fundamental Limitations of Alignment in Large Language Models**

大型语言模型中对齐的基本限制 cs.CL

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2304.11082v2) [paper-pdf](http://arxiv.org/pdf/2304.11082v2)

**Authors**: Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback increase the LLM's proneness to being prompted into the undesired behaviors. Moreover, we include the notion of personas in our BEB framework, and find that behaviors which are generally very unlikely to be exhibited by the model can be brought to the front by prompting the model to behave as specific persona. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.

摘要: 开发与人类交互的语言模型的一个重要方面是使他们的行为对人类用户有用而无害。这通常是通过调整模型来实现的，这种方式增强了期望的行为，抑制了不期望的行为，这一过程称为对齐。在本文中，我们提出了一种名为行为期望界限(BEB)的理论方法，它允许我们正式地研究大型语言模型中对齐的几个固有特征和限制。重要的是，我们证明了对于模型表现出的有限概率的任何行为，都存在可以触发模型输出该行为的提示，其概率随着提示的长度增加而增加。这意味着，任何减弱不受欢迎的行为但不能完全消除它的对准过程，在对抗提示攻击时都是不安全的。此外，我们的框架暗示了一种机制，通过这种机制，领先的对齐方法，如来自人类反馈的强化学习，增加了LLM被提示进入不希望看到的行为的倾向。此外，我们在我们的BEB框架中包括了人物角色的概念，并发现通过促使模型表现为特定的人物角色，通常不太可能在模型中表现的行为可以被带到前面。这一理论结果正在由所谓的当代“聊天GPT越狱”大规模实验证明，在这种情况下，敌对用户通过触发LLM充当恶意角色来欺骗LLM打破其对齐护栏。我们的结果暴露了LLM对齐方面的根本限制，并将设计可靠的机制以确保人工智能安全的必要性放在了首位。



## **35. From Adversarial Arms Race to Model-centric Evaluation: Motivating a Unified Automatic Robustness Evaluation Framework**

从对抗性军备竞赛到以模型为中心的评估：激励统一的自动健壮性评估框架 cs.CL

Accepted to Findings of ACL 2023

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.18503v1) [paper-pdf](http://arxiv.org/pdf/2305.18503v1)

**Authors**: Yangyi Chen, Hongcheng Gao, Ganqu Cui, Lifan Yuan, Dehan Kong, Hanlu Wu, Ning Shi, Bo Yuan, Longtao Huang, Hui Xue, Zhiyuan Liu, Maosong Sun, Heng Ji

**Abstract**: Textual adversarial attacks can discover models' weaknesses by adding semantic-preserved but misleading perturbations to the inputs. The long-lasting adversarial attack-and-defense arms race in Natural Language Processing (NLP) is algorithm-centric, providing valuable techniques for automatic robustness evaluation. However, the existing practice of robustness evaluation may exhibit issues of incomprehensive evaluation, impractical evaluation protocol, and invalid adversarial samples. In this paper, we aim to set up a unified automatic robustness evaluation framework, shifting towards model-centric evaluation to further exploit the advantages of adversarial attacks. To address the above challenges, we first determine robustness evaluation dimensions based on model capabilities and specify the reasonable algorithm to generate adversarial samples for each dimension. Then we establish the evaluation protocol, including evaluation settings and metrics, under realistic demands. Finally, we use the perturbation degree of adversarial samples to control the sample validity. We implement a toolkit RobTest that realizes our automatic robustness evaluation framework. In our experiments, we conduct a robustness evaluation of RoBERTa models to demonstrate the effectiveness of our evaluation framework, and further show the rationality of each component in the framework. The code will be made public at \url{https://github.com/thunlp/RobTest}.

摘要: 文本对抗性攻击可以通过向输入中添加语义保留但具有误导性的扰动来发现模型的弱点。自然语言处理(NLP)中持久的对抗性攻防军备竞赛以算法为中心，为自动健壮性评估提供了有价值的技术。然而，现有的健壮性评估实践可能存在评估不全面、评估协议不实用、对手样本无效等问题。在本文中，我们的目标是建立一个统一的自动健壮性评估框架，转向以模型为中心的评估，以进一步发挥对抗攻击的优势。为了应对上述挑战，我们首先根据模型能力确定健壮性评估维度，并指定合理的算法为每个维度生成对抗性样本。然后根据实际需求，制定了评估方案，包括评估设置和评估指标。最后，利用对抗性样本的扰动程度来控制样本的有效性。我们实现了一个工具包RobTest，它实现了我们的自动健壮性评估框架。在我们的实验中，我们对Roberta模型进行了健壮性评估，以验证评估框架的有效性，并进一步说明了框架中各个组件的合理性。代码将在\url{https://github.com/thunlp/RobTest}.}上公布



## **36. Fourier Analysis on Robustness of Graph Convolutional Neural Networks for Skeleton-based Action Recognition**

基于骨架的图卷积神经网络鲁棒性的傅立叶分析 cs.CV

17 pages, 13 figures

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.17939v1) [paper-pdf](http://arxiv.org/pdf/2305.17939v1)

**Authors**: Nariki Tanaka, Hiroshi Kera, Kazuhiko Kawamoto

**Abstract**: Using Fourier analysis, we explore the robustness and vulnerability of graph convolutional neural networks (GCNs) for skeleton-based action recognition. We adopt a joint Fourier transform (JFT), a combination of the graph Fourier transform (GFT) and the discrete Fourier transform (DFT), to examine the robustness of adversarially-trained GCNs against adversarial attacks and common corruptions. Experimental results with the NTU RGB+D dataset reveal that adversarial training does not introduce a robustness trade-off between adversarial attacks and low-frequency perturbations, which typically occurs during image classification based on convolutional neural networks. This finding indicates that adversarial training is a practical approach to enhancing robustness against adversarial attacks and common corruptions in skeleton-based action recognition. Furthermore, we find that the Fourier approach cannot explain vulnerability against skeletal part occlusion corruption, which highlights its limitations. These findings extend our understanding of the robustness of GCNs, potentially guiding the development of more robust learning methods for skeleton-based action recognition.

摘要: 利用傅立叶分析，我们研究了基于骨架的动作识别的图卷积神经网络(GCNS)的稳健性和脆弱性。我们采用联合傅里叶变换(JFT)，即图傅里叶变换(GFT)和离散傅立叶变换(DFT)的组合，来检验经过对抗性训练的GCNS对敌意攻击和常见腐败的健壮性。在NTU RGB+D数据集上的实验结果表明，对抗性训练不会在对抗性攻击和低频扰动之间引入稳健性权衡，而这通常发生在基于卷积神经网络的图像分类中。这一发现表明，在基于骨架的动作识别中，对抗性训练是一种增强对对抗性攻击和常见腐败的稳健性的实用方法。此外，我们发现傅立叶方法不能解释对骨骼部分遮挡破坏的脆弱性，这突出了它的局限性。这些发现扩展了我们对GCNS健壮性的理解，潜在地指导了基于骨骼的动作识别的更健壮的学习方法的发展。



## **37. Membership Inference Attacks against Language Models via Neighbourhood Comparison**

基于邻域比较的语言模型隶属度推理攻击 cs.CL

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.18462v1) [paper-pdf](http://arxiv.org/pdf/2305.18462v1)

**Authors**: Justus Mattern, Fatemehsadat Mireshghallah, Zhijing Jin, Bernhard Schölkopf, Mrinmaya Sachan, Taylor Berg-Kirkpatrick

**Abstract**: Membership Inference attacks (MIAs) aim to predict whether a data sample was present in the training data of a machine learning model or not, and are widely used for assessing the privacy risks of language models. Most existing attacks rely on the observation that models tend to assign higher probabilities to their training samples than non-training points. However, simple thresholding of the model score in isolation tends to lead to high false-positive rates as it does not account for the intrinsic complexity of a sample. Recent work has demonstrated that reference-based attacks which compare model scores to those obtained from a reference model trained on similar data can substantially improve the performance of MIAs. However, in order to train reference models, attacks of this kind make the strong and arguably unrealistic assumption that an adversary has access to samples closely resembling the original training data. Therefore, we investigate their performance in more realistic scenarios and find that they are highly fragile in relation to the data distribution used to train reference models. To investigate whether this fragility provides a layer of safety, we propose and evaluate neighbourhood attacks, which compare model scores for a given sample to scores of synthetically generated neighbour texts and therefore eliminate the need for access to the training data distribution. We show that, in addition to being competitive with reference-based attacks that have perfect knowledge about the training data distribution, our attack clearly outperforms existing reference-free attacks as well as reference-based attacks with imperfect knowledge, which demonstrates the need for a reevaluation of the threat model of adversarial attacks.

摘要: 成员关系推理攻击(MIA)旨在预测数据样本是否存在于机器学习模型的训练数据中，被广泛用于评估语言模型的隐私风险。大多数现有的攻击都依赖于这样的观察，即模型倾向于为其训练样本分配比非训练点更高的概率。然而，孤立地对模型分数进行简单的阈值处理往往会导致高的假阳性率，因为它没有考虑到样本的内在复杂性。最近的工作表明，基于参考的攻击将模型得分与根据相似数据训练的参考模型获得的得分进行比较，可以显著提高MIA的性能。然而，为了训练参考模型，这类攻击做出了强有力的、可以说是不切实际的假设，即对手可以获得与原始训练数据非常相似的样本。因此，我们在更现实的场景中调查了它们的性能，发现它们相对于用于训练参考模型的数据分布来说是非常脆弱的。为了调查这种脆弱性是否提供了一层安全，我们提出并评估了邻居攻击，该攻击将给定样本的模型分数与数十个合成生成的邻居文本进行比较，从而消除了访问训练数据分布的需要。我们表明，除了与对训练数据分布有完善了解的基于引用的攻击相比，该攻击的性能明显优于现有的无引用攻击和具有不完全知识的基于引用的攻击，这表明需要对对抗性攻击的威胁模型进行重新评估。



## **38. NaturalFinger: Generating Natural Fingerprint with Generative Adversarial Networks**

NaturalFinger：利用产生式对抗网络生成自然指纹 cs.CV

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.17868v1) [paper-pdf](http://arxiv.org/pdf/2305.17868v1)

**Authors**: Kang Yang, Kunhao Lai

**Abstract**: Deep neural network (DNN) models have become a critical asset of the model owner as training them requires a large amount of resource (i.e. labeled data). Therefore, many fingerprinting schemes have been proposed to safeguard the intellectual property (IP) of the model owner against model extraction and illegal redistribution. However, previous schemes adopt unnatural images as the fingerprint, such as adversarial examples and noisy images, which can be easily perceived and rejected by the adversary. In this paper, we propose NaturalFinger which generates natural fingerprint with generative adversarial networks (GANs). Besides, our proposed NaturalFinger fingerprints the decision difference areas rather than the decision boundary, which is more robust. The application of GAN not only allows us to generate more imperceptible samples, but also enables us to generate unrestricted samples to explore the decision boundary.To demonstrate the effectiveness of our fingerprint approach, we evaluate our approach against four model modification attacks including adversarial training and two model extraction attacks. Experiments show that our approach achieves 0.91 ARUC value on the FingerBench dataset (154 models), exceeding the optimal baseline (MetaV) over 17\%.

摘要: 深度神经网络(DNN)模型已经成为模型所有者的重要资产，因为训练它们需要大量的资源(即标记数据)。因此，许多指纹方案被提出来保护模型所有者的知识产权(IP)，以防止模型提取和非法再分发。然而，以往的方案采用非自然的图像作为指纹，如敌意图像和噪声图像，这些图像很容易被攻击者感知和拒绝。在本文中，我们提出了利用生成性对抗网络(GANS)生成自然指纹的NaturalFinger。此外，我们提出的NaturalFinger指纹提取决策差异区而不是决策边界，从而更健壮。该方法不仅可以生成更多的隐蔽样本，还可以生成不受限制的样本来探索决策边界。为了验证指纹方法的有效性，我们对包括对抗性训练和两个模型提取攻击在内的四种模型修改攻击进行了评估。实验表明，该方法在FingerB边数据集(154个模型)上达到了0.91ARUC值，超过了最优基线(MetaV)17。



## **39. NOTABLE: Transferable Backdoor Attacks Against Prompt-based NLP Models**

值得注意：针对基于提示的NLP模型的可转移后门攻击 cs.CL

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.17826v1) [paper-pdf](http://arxiv.org/pdf/2305.17826v1)

**Authors**: Kai Mei, Zheng Li, Zhenting Wang, Yang Zhang, Shiqing Ma

**Abstract**: Prompt-based learning is vulnerable to backdoor attacks. Existing backdoor attacks against prompt-based models consider injecting backdoors into the entire embedding layers or word embedding vectors. Such attacks can be easily affected by retraining on downstream tasks and with different prompting strategies, limiting the transferability of backdoor attacks. In this work, we propose transferable backdoor attacks against prompt-based models, called NOTABLE, which is independent of downstream tasks and prompting strategies. Specifically, NOTABLE injects backdoors into the encoders of PLMs by utilizing an adaptive verbalizer to bind triggers to specific words (i.e., anchors). It activates the backdoor by pasting input with triggers to reach adversary-desired anchors, achieving independence from downstream tasks and prompting strategies. We conduct experiments on six NLP tasks, three popular models, and three prompting strategies. Empirical results show that NOTABLE achieves superior attack performance (i.e., attack success rate over 90% on all the datasets), and outperforms two state-of-the-art baselines. Evaluations on three defenses show the robustness of NOTABLE. Our code can be found at https://github.com/RU-System-Software-and-Security/Notable.

摘要: 基于提示的学习很容易受到后门攻击。现有针对基于提示的模型的后门攻击考虑向整个嵌入层或单词嵌入向量中注入后门。这类攻击很容易受到下游任务再培训和不同提示策略的影响，限制了后门攻击的可转移性。在这项工作中，我们提出了独立于下游任务和提示策略的针对提示模型的可转移后门攻击，称为显著模型。具体地说，值得注意的是，通过利用自适应动词器将触发器绑定到特定单词(即锚)，将后门注入PLM的编码器。它通过粘贴带有触发器的输入来激活后门，以到达对手想要的锚，实现对下游任务和提示策略的独立。我们在六个自然语言处理任务、三个流行模式和三个提示策略上进行了实验。实验结果表明，该算法取得了较好的攻击性能(即在所有数据集上的攻击成功率都在90%以上)，并超过了两个最先进的基线。对三种防御措施的评估表明，该算法具有较好的稳健性。我们的代码可以在https://github.com/RU-System-Software-and-Security/Notable.上找到



## **40. DiffProtect: Generate Adversarial Examples with Diffusion Models for Facial Privacy Protection**

DiffProtect：使用扩散模型生成用于面部隐私保护的敌意示例 cs.CV

Code will be available at https://github.com/joellliu/DiffProtect/

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.13625v2) [paper-pdf](http://arxiv.org/pdf/2305.13625v2)

**Authors**: Jiang Liu, Chun Pong Lau, Rama Chellappa

**Abstract**: The increasingly pervasive facial recognition (FR) systems raise serious concerns about personal privacy, especially for billions of users who have publicly shared their photos on social media. Several attempts have been made to protect individuals from being identified by unauthorized FR systems utilizing adversarial attacks to generate encrypted face images. However, existing methods suffer from poor visual quality or low attack success rates, which limit their utility. Recently, diffusion models have achieved tremendous success in image generation. In this work, we ask: can diffusion models be used to generate adversarial examples to improve both visual quality and attack performance? We propose DiffProtect, which utilizes a diffusion autoencoder to generate semantically meaningful perturbations on FR systems. Extensive experiments demonstrate that DiffProtect produces more natural-looking encrypted images than state-of-the-art methods while achieving significantly higher attack success rates, e.g., 24.5% and 25.1% absolute improvements on the CelebA-HQ and FFHQ datasets.

摘要: 日益普及的面部识别(FR)系统引发了对个人隐私的严重担忧，特别是对数十亿在社交媒体上公开分享照片的用户来说。已经进行了几次尝试，以保护个人不被未经授权的FR系统利用敌意攻击来生成加密的面部图像来识别。然而，现有的方法存在视觉质量差或攻击成功率低的问题，这限制了它们的实用性。近年来，扩散模型在图像生成方面取得了巨大的成功。在这项工作中，我们问：扩散模型能否被用来生成对抗性例子，以提高视觉质量和攻击性能？我们提出了DiffProtect，它利用扩散自动编码器在FR系统上产生语义上有意义的扰动。大量实验表明，与最先进的方法相比，DiffProtect生成的加密图像看起来更自然，同时实现了显著更高的攻击成功率，例如，在CelebA-HQ和FFHQ数据集上的绝对改进了24.5%和25.1%。



## **41. Amplification trojan network: Attack deep neural networks by amplifying their inherent weakness**

放大特洛伊木马网络：通过放大深层神经网络的固有弱点来攻击它们 cs.CR

Published Sep 2022 in Neurocomputing

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.17688v1) [paper-pdf](http://arxiv.org/pdf/2305.17688v1)

**Authors**: Zhanhao Hu, Jun Zhu, Bo Zhang, Xiaolin Hu

**Abstract**: Recent works found that deep neural networks (DNNs) can be fooled by adversarial examples, which are crafted by adding adversarial noise on clean inputs. The accuracy of DNNs on adversarial examples will decrease as the magnitude of the adversarial noise increase. In this study, we show that DNNs can be also fooled when the noise is very small under certain circumstances. This new type of attack is called Amplification Trojan Attack (ATAttack). Specifically, we use a trojan network to transform the inputs before sending them to the target DNN. This trojan network serves as an amplifier to amplify the inherent weakness of the target DNN. The target DNN, which is infected by the trojan network, performs normally on clean data while being more vulnerable to adversarial examples. Since it only transforms the inputs, the trojan network can hide in DNN-based pipelines, e.g. by infecting the pre-processing procedure of the inputs before sending them to the DNNs. This new type of threat should be considered in developing safe DNNs.

摘要: 最近的工作发现，深度神经网络(DNN)可以被敌意例子愚弄，这些例子是通过在干净的输入上添加对抗性噪声来构建的。DNN对对抗性样本的准确率会随着对抗性噪声的增加而降低。在这项研究中，我们证明了在某些情况下，当噪声非常小时，DNN也可以被愚弄。这种新型攻击被称为放大特洛伊木马攻击(ATAttack)。具体地说，我们使用特洛伊木马网络在将输入发送到目标DNN之前对其进行转换。此特洛伊木马网络充当放大器，放大目标DNN的固有弱点。被特洛伊木马网络感染的目标DNN在干净的数据上正常执行，但更容易受到敌意示例的攻击。由于它只转换输入，特洛伊木马网络可以隐藏在基于DNN的管道中，例如通过在将输入发送到DNN之前感染输入的预处理过程。在开发安全的DNN时，应该考虑这种新型的威胁。



## **42. Threat Models over Space and Time: A Case Study of E2EE Messaging Applications**

空间和时间上的威胁模型：E2EE消息传递应用程序的案例研究 cs.CR

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2301.05653v2) [paper-pdf](http://arxiv.org/pdf/2301.05653v2)

**Authors**: Partha Das Chowdhury, Maria Sameen, Jenny Blessing, Nicholas Boucher, Joseph Gardiner, Tom Burrows, Ross Anderson, Awais Rashid

**Abstract**: Threat modelling is foundational to secure systems engineering and should be done in consideration of the context within which systems operate. On the other hand, the continuous evolution of both the technical sophistication of threats and the system attack surface is an inescapable reality. In this work, we explore the extent to which real-world systems engineering reflects the changing threat context. To this end we examine the desktop clients of six widely used end-to-end-encrypted mobile messaging applications to understand the extent to which they adjusted their threat model over space (when enabling clients on new platforms, such as desktop clients) and time (as new threats emerged). We experimented with short-lived adversarial access against these desktop clients and analyzed the results with respect to two popular threat elicitation frameworks, STRIDE and LINDDUN. The results demonstrate that system designers need to both recognise the threats in the evolving context within which systems operate and, more importantly, to mitigate them by rescoping trust boundaries in a manner that those within the administrative boundary cannot violate security and privacy properties. Such a nuanced understanding of trust boundary scopes and their relationship with administrative boundaries allows for better administration of shared components, including securing them with safe defaults.

摘要: 威胁建模是安全系统工程的基础，应该考虑到系统运行的环境。另一方面，威胁的技术复杂性和系统攻击面的不断演变是一个不可避免的现实。在这项工作中，我们探索现实世界系统工程反映不断变化的威胁背景的程度。为此，我们研究了六个广泛使用的端到端加密移动消息传递应用程序的桌面客户端，以了解它们在空间(当在新平台上启用客户端时)和时间(当出现新威胁时)对其威胁模型进行调整的程度。我们对这些桌面客户端进行了短暂的恶意访问试验，并分析了两个流行的威胁诱导框架STRIDE和LINDDUN的结果。结果表明，系统设计者需要在系统运行的不断变化的环境中认识到威胁，更重要的是，通过以行政边界内的信任边界不能违反安全和隐私属性的方式重新应对信任边界来缓解这些威胁。对信任边界作用域及其与管理边界的关系的这种细致入微的理解允许更好地管理共享组件，包括使用安全缺省值保护它们。



## **43. A Synergistic Framework Leveraging Autoencoders and Generative Adversarial Networks for the Synthesis of Computational Fluid Dynamics Results in Aerofoil Aerodynamics**

利用自动编码器和生成性对抗网络综合翼型空气动力学计算流体力学结果的协同框架 physics.flu-dyn

9 pages, 11 figures

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.18386v1) [paper-pdf](http://arxiv.org/pdf/2305.18386v1)

**Authors**: Tanishk Nandal, Vaibhav Fulara, Raj Kumar Singh

**Abstract**: In the realm of computational fluid dynamics (CFD), accurate prediction of aerodynamic behaviour plays a pivotal role in aerofoil design and optimization. This study proposes a novel approach that synergistically combines autoencoders and Generative Adversarial Networks (GANs) for the purpose of generating CFD results. Our innovative framework harnesses the intrinsic capabilities of autoencoders to encode aerofoil geometries into a compressed and informative 20-length vector representation. Subsequently, a conditional GAN network adeptly translates this vector into precise pressure-distribution plots, accounting for fixed wind velocity, angle of attack, and turbulence level specifications. The training process utilizes a meticulously curated dataset acquired from JavaFoil software, encompassing a comprehensive range of aerofoil geometries. The proposed approach exhibits profound potential in reducing the time and costs associated with aerodynamic prediction, enabling efficient evaluation of aerofoil performance. The findings contribute to the advancement of computational techniques in fluid dynamics and pave the way for enhanced design and optimization processes in aerodynamics.

摘要: 在计算流体力学(CFD)领域，气动性能的准确预测在翼型设计和优化中起着至关重要的作用。这项研究提出了一种新的方法，它协同结合自动编码器和生成性对抗网络(GANS)来生成CFD结果。我们的创新框架利用自动编码器的内在能力，将机翼几何图形编码为压缩的20长度矢量表示。随后，条件GAN网络巧妙地将该向量转换为精确的压力分布图，考虑到固定的风速、攻角和湍流级别规格。培训过程利用了从JavaFoil软件获得的经过精心挑选的数据集，包括全面的机翼几何形状。所提出的方法在减少与气动预测相关的时间和成本方面显示出巨大的潜力，使得能够有效地评估翼型的性能。这些发现有助于流体力学计算技术的进步，并为空气动力学的改进设计和优化过程铺平了道路。



## **44. Backdoor Attacks Against Incremental Learners: An Empirical Evaluation Study**

针对渐进式学习者的后门攻击：一项实证评估研究 cs.CR

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.18384v1) [paper-pdf](http://arxiv.org/pdf/2305.18384v1)

**Authors**: Yiqi Zhong, Xianming Liu, Deming Zhai, Junjun Jiang, Xiangyang Ji

**Abstract**: Large amounts of incremental learning algorithms have been proposed to alleviate the catastrophic forgetting issue arises while dealing with sequential data on a time series. However, the adversarial robustness of incremental learners has not been widely verified, leaving potential security risks. Specifically, for poisoning-based backdoor attacks, we argue that the nature of streaming data in IL provides great convenience to the adversary by creating the possibility of distributed and cross-task attacks -- an adversary can affect \textbf{any unknown} previous or subsequent task by data poisoning \textbf{at any time or time series} with extremely small amount of backdoor samples injected (e.g., $0.1\%$ based on our observations). To attract the attention of the research community, in this paper, we empirically reveal the high vulnerability of 11 typical incremental learners against poisoning-based backdoor attack on 3 learning scenarios, especially the cross-task generalization effect of backdoor knowledge, while the poison ratios range from $5\%$ to as low as $0.1\%$. Finally, the defense mechanism based on activation clustering is found to be effective in detecting our trigger pattern to mitigate potential security risks.

摘要: 为了缓解在处理时间序列上的连续数据时出现的灾难性遗忘问题，已经提出了大量的增量学习算法。然而，增量学习的对抗健壮性还没有得到广泛的验证，留下了潜在的安全隐患。具体地说，对于基于中毒的后门攻击，我们认为IL中的流数据的性质为对手提供了极大的便利，因为它创造了分布式和跨任务攻击的可能性--对手可以通过在任何时间或时间序列注入极少量的后门样本(例如，根据我们的观察，0.1美元)来影响\extbf{任何未知}之前或之后的任务。为了引起研究界的注意，本文实证揭示了11名典型增量学习者在3种学习场景下对中毒后门攻击的高度脆弱性，特别是后门知识的跨任务泛化效应，中毒比从5美元到0.1美元不等。最后，基于激活聚类的防御机制有效地检测了触发模式，降低了潜在的安全风险。



## **45. BadLabel: A Robust Perspective on Evaluating and Enhancing Label-noise Learning**

BadLabel：评估和加强标签噪声学习的稳健视角 cs.LG

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.18377v1) [paper-pdf](http://arxiv.org/pdf/2305.18377v1)

**Authors**: Jingfeng Zhang, Bo Song, Haohan Wang, Bo Han, Tongliang Liu, Lei Liu, Masashi Sugiyama

**Abstract**: Label-noise learning (LNL) aims to increase the model's generalization given training data with noisy labels. To facilitate practical LNL algorithms, researchers have proposed different label noise types, ranging from class-conditional to instance-dependent noises. In this paper, we introduce a novel label noise type called BadLabel, which can significantly degrade the performance of existing LNL algorithms by a large margin. BadLabel is crafted based on the label-flipping attack against standard classification, where specific samples are selected and their labels are flipped to other labels so that the loss values of clean and noisy labels become indistinguishable. To address the challenge posed by BadLabel, we further propose a robust LNL method that perturbs the labels in an adversarial manner at each epoch to make the loss values of clean and noisy labels again distinguishable. Once we select a small set of (mostly) clean labeled data, we can apply the techniques of semi-supervised learning to train the model accurately. Empirically, our experimental results demonstrate that existing LNL algorithms are vulnerable to the newly introduced BadLabel noise type, while our proposed robust LNL method can effectively improve the generalization performance of the model under various types of label noise. The new dataset of noisy labels and the source codes of robust LNL algorithms are available at https://github.com/zjfheart/BadLabels.

摘要: 标签噪声学习(LNL)的目的是在给定含有噪声标签的训练数据的情况下提高模型的泛化能力。为了便于实用的LNL算法，研究人员提出了不同的标签噪声类型，从类条件噪声到实例相关噪声。本文引入了一种新的标签噪声类型BadLabel，它可以显著降低现有LNL算法的性能。BadLabel是基于针对标准分类的标签翻转攻击而构建的，在标准分类中，选择特定的样本，并将其标签翻转到其他标签，从而使干净和噪声标签的损失值变得无法区分。为了应对BadLabel带来的挑战，我们进一步提出了一种稳健的LNL方法，该方法在每个历元以对抗性的方式扰动标签，使干净标签和噪声标签的丢失值再次可区分。一旦我们选择了一小部分(大部分)干净的标签数据，我们就可以应用半监督学习技术来准确地训练模型。实验结果表明，现有的LNL算法容易受到新引入的BadLabel噪声类型的影响，而本文提出的稳健LNL方法可以有效地提高模型在各种类型标签噪声下的泛化性能。新的噪声标注数据集和健壮的LNL算法的源代码可在https://github.com/zjfheart/BadLabels.获得



## **46. SneakyPrompt: Evaluating Robustness of Text-to-image Generative Models' Safety Filters**

SneakyPrompt：评估文本到图像生成模型的安全过滤器的健壮性 cs.LG

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.12082v2) [paper-pdf](http://arxiv.org/pdf/2305.12082v2)

**Authors**: Yuchen Yang, Bo Hui, Haolin Yuan, Neil Gong, Yinzhi Cao

**Abstract**: Text-to-image generative models such as Stable Diffusion and DALL$\cdot$E 2 have attracted much attention since their publication due to their wide application in the real world. One challenging problem of text-to-image generative models is the generation of Not-Safe-for-Work (NSFW) content, e.g., those related to violence and adult. Therefore, a common practice is to deploy a so-called safety filter, which blocks NSFW content based on either text or image features. Prior works have studied the possible bypass of such safety filters. However, existing works are largely manual and specific to Stable Diffusion's official safety filter. Moreover, the bypass ratio of Stable Diffusion's safety filter is as low as 23.51% based on our evaluation.   In this paper, we propose the first automated attack framework, called SneakyPrompt, to evaluate the robustness of real-world safety filters in state-of-the-art text-to-image generative models. Our key insight is to search for alternative tokens in a prompt that generates NSFW images so that the generated prompt (called an adversarial prompt) bypasses existing safety filters. Specifically, SneakyPrompt utilizes reinforcement learning (RL) to guide an agent with positive rewards on semantic similarity and bypass success.   Our evaluation shows that SneakyPrompt successfully generated NSFW content using an online model DALL$\cdot$E 2 with its default, closed-box safety filter enabled. At the same time, we also deploy several open-source state-of-the-art safety filters on a Stable Diffusion model and show that SneakyPrompt not only successfully generates NSFW content, but also outperforms existing adversarial attacks in terms of the number of queries and image qualities.

摘要: 从文本到图像的生成模型，如稳定扩散模型和Dall$\CDOT$E2模型自问世以来，由于其在现实世界中的广泛应用而引起了人们的广泛关注。文本到图像生成模型的一个具有挑战性的问题是生成非安全工作(NSFW)内容，例如与暴力和成人有关的内容。因此，一种常见的做法是部署所谓的安全过滤器，即根据文本或图像特征阻止NSFW内容。以前的工作已经研究了这种安全过滤器的可能旁路。然而，现有的工作主要是手动的，专门针对稳定扩散的官方安全过滤器。此外，根据我们的评估，稳定扩散安全过滤器的旁路比低至23.51%。在本文中，我们提出了第一个自动攻击框架，称为SneakyPrompt，用于评估最新的文本到图像生成模型中现实世界安全过滤器的稳健性。我们的主要见解是在生成NSFW图像的提示中搜索替代令牌，以便生成的提示(称为对抗性提示)绕过现有的安全过滤器。具体地说，SneakyPrompt利用强化学习(RL)来指导代理在语义相似性方面获得积极回报，并绕过成功。我们的评估表明，SneakyPrompt成功地使用在线模型DALL$\CDOT$E 2生成了NSFW内容，并启用了默认的闭箱安全过滤器。同时，我们还在一个稳定的扩散模型上部署了几个开源的最先进的安全过滤器，并表明SneakyPrompt不仅成功地生成了NSFW内容，而且在查询数量和图像质量方面都优于现有的对抗性攻击。



## **47. Tubes Among Us: Analog Attack on Automatic Speaker Identification**

我们之间的管道：对自动说话人识别的模拟攻击 cs.LG

Published at USENIX Security 2023  https://www.usenix.org/conference/usenixsecurity23/presentation/ahmed

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2202.02751v2) [paper-pdf](http://arxiv.org/pdf/2202.02751v2)

**Authors**: Shimaa Ahmed, Yash Wani, Ali Shahin Shamsabadi, Mohammad Yaghini, Ilia Shumailov, Nicolas Papernot, Kassem Fawaz

**Abstract**: Recent years have seen a surge in the popularity of acoustics-enabled personal devices powered by machine learning. Yet, machine learning has proven to be vulnerable to adversarial examples. A large number of modern systems protect themselves against such attacks by targeting artificiality, i.e., they deploy mechanisms to detect the lack of human involvement in generating the adversarial examples. However, these defenses implicitly assume that humans are incapable of producing meaningful and targeted adversarial examples. In this paper, we show that this base assumption is wrong. In particular, we demonstrate that for tasks like speaker identification, a human is capable of producing analog adversarial examples directly with little cost and supervision: by simply speaking through a tube, an adversary reliably impersonates other speakers in eyes of ML models for speaker identification. Our findings extend to a range of other acoustic-biometric tasks such as liveness detection, bringing into question their use in security-critical settings in real life, such as phone banking.

摘要: 近年来，由机器学习驱动的声学个人设备的受欢迎程度激增。然而，事实证明，机器学习很容易受到对抗性例子的影响。大量现代系统通过瞄准人为攻击来保护自己免受此类攻击，即，它们部署机制来检测在生成对抗性例子时缺乏人的参与。然而，这些防御隐含地假设人类没有能力制造有意义和有针对性的对抗性例子。在本文中，我们证明了这个基本假设是错误的。特别是，我们证明了对于像说话人识别这样的任务，人类能够在几乎不需要成本和监督的情况下直接产生模拟的对抗性例子：通过简单地通过管道说话，对手可靠地在ML模型的眼中模仿其他说话人进行说话人识别。我们的发现延伸到了其他一系列声学-生物识别任务，如活体检测，这让人质疑它们在现实生活中对安全至关重要的环境中的使用，比如电话银行。



## **48. PowerGAN: A Machine Learning Approach for Power Side-Channel Attack on Compute-in-Memory Accelerators**

PowerGAN：一种针对内存计算加速器电源侧通道攻击的机器学习方法 cs.CR

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2304.11056v2) [paper-pdf](http://arxiv.org/pdf/2304.11056v2)

**Authors**: Ziyu Wang, Yuting Wu, Yongmo Park, Sangmin Yoo, Xinxin Wang, Jason K. Eshraghian, Wei D. Lu

**Abstract**: Analog compute-in-memory (CIM) systems are promising for deep neural network (DNN) inference acceleration due to their energy efficiency and high throughput. However, as the use of DNNs expands, protecting user input privacy has become increasingly important. In this paper, we identify a potential security vulnerability wherein an adversary can reconstruct the user's private input data from a power side-channel attack, under proper data acquisition and pre-processing, even without knowledge of the DNN model. We further demonstrate a machine learning-based attack approach using a generative adversarial network (GAN) to enhance the data reconstruction. Our results show that the attack methodology is effective in reconstructing user inputs from analog CIM accelerator power leakage, even at large noise levels and after countermeasures are applied. Specifically, we demonstrate the efficacy of our approach on an example of U-Net inference chip for brain tumor detection, and show the original magnetic resonance imaging (MRI) medical images can be successfully reconstructed even at a noise-level of 20% standard deviation of the maximum power signal value. Our study highlights a potential security vulnerability in analog CIM accelerators and raises awareness of using GAN to breach user privacy in such systems.

摘要: 模拟内存计算(CIM)系统由于其高能量效率和高吞吐量，在深度神经网络(DNN)推理加速中具有广阔的应用前景。然而，随着DNN的使用范围扩大，保护用户输入隐私变得越来越重要。在本文中，我们发现了一个潜在的安全漏洞，在该漏洞中，攻击者即使在不了解DNN模型的情况下，也可以在适当的数据获取和预处理下，从功率侧通道攻击中重构用户的私人输入数据。我们进一步展示了一种基于机器学习的攻击方法，使用生成性对抗网络(GAN)来增强数据重构。我们的结果表明，即使在大噪声水平和采取对策后，该攻击方法也能有效地从模拟CIM加速器功率泄漏中恢复用户输入。具体地说，我们在U-Net推理芯片用于脑肿瘤检测的例子上验证了该方法的有效性，并表明即使在最大功率信号值的20%标准差的噪声水平下，也可以成功地重建原始磁共振成像(MRI)医学图像。我们的研究突出了模拟CIM加速器中的潜在安全漏洞，并提高了人们对在此类系统中使用GAN来侵犯用户隐私的认识。



## **49. Two Heads are Better than One: Towards Better Adversarial Robustness by Combining Transduction and Rejection**

两个头比一个头好：通过结合转导和拒绝来实现更好的对手稳健性 cs.LG

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.17528v1) [paper-pdf](http://arxiv.org/pdf/2305.17528v1)

**Authors**: Nils Palumbo, Yang Guo, Xi Wu, Jiefeng Chen, Yingyu Liang, Somesh Jha

**Abstract**: Both transduction and rejection have emerged as important techniques for defending against adversarial perturbations. A recent work by Tram\`er showed that, in the rejection-only case (no transduction), a strong rejection-solution can be turned into a strong (but computationally inefficient) non-rejection solution. This detector-to-classifier reduction has been mostly applied to give evidence that certain claims of strong selective-model solutions are susceptible, leaving the benefits of rejection unclear. On the other hand, a recent work by Goldwasser et al. showed that rejection combined with transduction can give provable guarantees (for certain problems) that cannot be achieved otherwise. Nevertheless, under recent strong adversarial attacks (GMSA, which has been shown to be much more effective than AutoAttack against transduction), Goldwasser et al.'s work was shown to have low performance in a practical deep-learning setting. In this paper, we take a step towards realizing the promise of transduction+rejection in more realistic scenarios. Theoretically, we show that a novel application of Tram\`er's classifier-to-detector technique in the transductive setting can give significantly improved sample-complexity for robust generalization. While our theoretical construction is computationally inefficient, it guides us to identify an efficient transductive algorithm to learn a selective model. Extensive experiments using state of the art attacks (AutoAttack, GMSA) show that our solutions provide significantly better robust accuracy.

摘要: 转导和拒绝都已成为防御敌意干扰的重要技术。Tramer最近的一项工作表明，在只有拒绝的情况下(没有转导)，强拒绝解可以变成强(但计算效率低)的非拒绝解。这种探测器到分类器的减少主要是为了提供证据，证明某些声称的强选择性模型解决方案是敏感的，留下了拒绝的好处不清楚。另一方面，Goldwasser等人最近的一项工作。表明拒绝和转导相结合可以提供可证明的保证(对于某些问题)，而不是通过其他方式无法实现的。然而，在最近强大的对手攻击下(GMSA，已被证明比AutoAttack对抗转导要有效得多)，Goldwasser等人的工作被证明在实际的深度学习环境中表现不佳。在这篇论文中，我们朝着在更现实的情景中实现转导+拒绝的承诺迈出了一步。理论上，我们证明了Tram‘er的分类器到检测器技术在换能式环境中的一种新的应用可以显著改善样本复杂度以实现稳健的泛化。虽然我们的理论构建在计算上效率低下，但它指导我们识别一个有效的换能式算法来学习一个选择的模型。使用最先进的攻击(AutoAttack，GMSA)进行的广泛实验表明，我们的解决方案提供了显著更好的健壮性准确性。



## **50. Backdooring Neural Code Search**

回溯神经编码搜索 cs.SE

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.17506v1) [paper-pdf](http://arxiv.org/pdf/2305.17506v1)

**Authors**: Weisong Sun, Yuchen Chen, Guanhong Tao, Chunrong Fang, Xiangyu Zhang, Quanjun Zhang, Bin Luo

**Abstract**: Reusing off-the-shelf code snippets from online repositories is a common practice, which significantly enhances the productivity of software developers. To find desired code snippets, developers resort to code search engines through natural language queries. Neural code search models are hence behind many such engines. These models are based on deep learning and gain substantial attention due to their impressive performance. However, the security aspect of these models is rarely studied. Particularly, an adversary can inject a backdoor in neural code search models, which return buggy or even vulnerable code with security/privacy issues. This may impact the downstream software (e.g., stock trading systems and autonomous driving) and cause financial loss and/or life-threatening incidents. In this paper, we demonstrate such attacks are feasible and can be quite stealthy. By simply modifying one variable/function name, the attacker can make buggy/vulnerable code rank in the top 11%. Our attack BADCODE features a special trigger generation and injection procedure, making the attack more effective and stealthy. The evaluation is conducted on two neural code search models and the results show our attack outperforms baselines by 60%. Our user study demonstrates that our attack is more stealthy than the baseline by two times based on the F1 score.

摘要: 重用在线存储库中的现成代码片段是一种常见的做法，这显著提高了软件开发人员的工作效率。为了找到所需的代码片段，开发人员通过自然语言查询求助于代码搜索引擎。因此，神经代码搜索模型是许多此类引擎的幕后推手。这些模型是基于深度学习的，由于其令人印象深刻的性能而获得了大量关注。然而，这些模型的安全性方面的研究很少。特别是，攻击者可以在神经代码搜索模型中注入后门，该模型返回带有安全/隐私问题的错误代码，甚至是易受攻击的代码。这可能会影响下游软件(例如股票交易系统和自动驾驶)，并导致经济损失和/或危及生命的事件。在这篇文章中，我们证明了这种攻击是可行的，并且可以相当隐蔽。只需修改一个变量/函数名称，攻击者就可以使有错误/易受攻击的代码排在前11%。我们的攻击BADCODE具有特殊的触发生成和注入过程，使攻击更有效和隐蔽。在两个神经编码搜索模型上进行了评估，结果表明我们的攻击性能比基线高60%。我们的用户研究表明，基于F1比分，我们的攻击比基线更隐蔽两倍。



