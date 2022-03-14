# Latest Adversarial Attack Papers
**update at 2022-03-15 06:31:57**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. On Adversarial Robustness of Trajectory Prediction for Autonomous Vehicles**

自主车辆轨迹预测的对抗鲁棒性研究 cs.CV

13 pages, 13 figures, accepted by CVPR 2022

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2201.05057v2)

**Authors**: Qingzhao Zhang, Shengtuo Hu, Jiachen Sun, Qi Alfred Chen, Z. Morley Mao

**Abstracts**: Trajectory prediction is a critical component for autonomous vehicles (AVs) to perform safe planning and navigation. However, few studies have analyzed the adversarial robustness of trajectory prediction or investigated whether the worst-case prediction can still lead to safe planning. To bridge this gap, we study the adversarial robustness of trajectory prediction models by proposing a new adversarial attack that perturbs normal vehicle trajectories to maximize the prediction error. Our experiments on three models and three datasets show that the adversarial prediction increases the prediction error by more than 150%. Our case studies show that if an adversary drives a vehicle close to the target AV following the adversarial trajectory, the AV may make an inaccurate prediction and even make unsafe driving decisions. We also explore possible mitigation techniques via data augmentation and trajectory smoothing. The implementation is open source at https://github.com/zqzqz/AdvTrajectoryPrediction.

摘要: 轨迹预测是自动驾驶车辆进行安全规划和导航的重要组成部分。然而，很少有研究分析弹道预测的对抗稳健性，或研究最坏情况的预测是否仍能导致安全规划。为了弥补这一差距，我们研究了轨迹预测模型的对抗性，提出了一种新的对抗性攻击，通过扰动正常的车辆轨迹来最大化预测误差。在三个模型和三个数据集上的实验表明，对抗性预测使预测误差增加了150%以上。我们的案例研究表明，如果对手沿着敌对的轨迹驾驶车辆接近目标AV，AV可能会做出不准确的预测，甚至做出不安全的驾驶决策。我们还通过数据增强和轨迹平滑来探索可能的缓解技术。该实现在https://github.com/zqzqz/AdvTrajectoryPrediction.上是开源的



## **2. Sparse Black-box Video Attack with Reinforcement Learning**

基于强化学习的稀疏黑盒视频攻击 cs.CV

Accepted at IJCV 2022

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2001.03754v3)

**Authors**: Xingxing Wei, Huanqian Yan, Bo Li

**Abstracts**: Adversarial attacks on video recognition models have been explored recently. However, most existing works treat each video frame equally and ignore their temporal interactions. To overcome this drawback, a few methods try to select some key frames and then perform attacks based on them. Unfortunately, their selection strategy is independent of the attacking step, therefore the resulting performance is limited. Instead, we argue the frame selection phase is closely relevant with the attacking phase. The key frames should be adjusted according to the attacking results. For that, we formulate the black-box video attacks into a Reinforcement Learning (RL) framework. Specifically, the environment in RL is set as the recognition model, and the agent in RL plays the role of frame selecting. By continuously querying the recognition models and receiving the attacking feedback, the agent gradually adjusts its frame selection strategy and adversarial perturbations become smaller and smaller. We conduct a series of experiments with two mainstream video recognition models: C3D and LRCN on the public UCF-101 and HMDB-51 datasets. The results demonstrate that the proposed method can significantly reduce the adversarial perturbations with efficient query times.

摘要: 最近，针对视频识别模型的对抗性攻击已经被探索出来。然而，现有的大多数工作都将每个视频帧一视同仁地对待，而忽略了它们之间的时间交互。为了克服这一缺点，有几种方法试图选择一些关键帧，然后根据这些关键帧进行攻击。不幸的是，它们的选择策略与攻击步骤无关，因此所产生的性能是有限的。相反，我们认为帧选择阶段与攻击阶段密切相关。应根据攻击结果调整关键帧。为此，我们将黑盒视频攻击描述为强化学习(RL)框架。具体地说，RL中的环境被设置为识别模型，RL中的Agent起到框架选择的作用。通过不断查询识别模型和接收攻击反馈，Agent逐渐调整其帧选择策略，敌方扰动变得越来越小。我们在公开的UCF-101和HMDB-51数据集上用两种主流的视频识别模型C3D和LRCN进行了一系列的实验。实验结果表明，该方法可以有效地减少对抗性扰动，提高查询效率。



## **3. Block-Sparse Adversarial Attack to Fool Transformer-Based Text Classifiers**

对基于愚人转换器的文本分类器的挡路稀疏敌意攻击 cs.CL

ICASSP 2022, Code available at:  https://github.com/sssadrizadeh/transformer-text-classifier-attack

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2203.05948v1)

**Authors**: Sahar Sadrizadeh, Ljiljana Dolamic, Pascal Frossard

**Abstracts**: Recently, it has been shown that, in spite of the significant performance of deep neural networks in different fields, those are vulnerable to adversarial examples. In this paper, we propose a gradient-based adversarial attack against transformer-based text classifiers. The adversarial perturbation in our method is imposed to be block-sparse so that the resultant adversarial example differs from the original sentence in only a few words. Due to the discrete nature of textual data, we perform gradient projection to find the minimizer of our proposed optimization problem. Experimental results demonstrate that, while our adversarial attack maintains the semantics of the sentence, it can reduce the accuracy of GPT-2 to less than 5% on different datasets (AG News, MNLI, and Yelp Reviews). Furthermore, the block-sparsity constraint of the proposed optimization problem results in small perturbations in the adversarial example.

摘要: 最近的研究表明，尽管深度神经网络在不同的领域有着显著的性能，但它们很容易受到敌意例子的影响。本文针对基于变换的文本分类器提出了一种基于梯度的对抗性攻击方法。我们方法中的对抗性扰动被强加为挡路稀疏的，这样得到的对抗性示例与原始句子只有几个字的不同。由于文本数据的离散性，我们使用梯度投影来寻找我们所提出的优化问题的最小值。实验结果表明，我们的对抗性攻击在保持句子语义的同时，可以将GPT-2在不同数据集(AG News、MNLI和Yelp Reviews)上的准确率降低到5%以下。此外，所提出的优化问题的挡路稀疏性约束导致了对抗性例子中的微小扰动。



## **4. Reinforcement Learning for Linear Quadratic Control is Vulnerable Under Cost Manipulation**

线性二次控制的强化学习在成本操纵下易受攻击 eess.SY

This paper is yet to be peer-reviewed

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2203.05774v1)

**Authors**: Yunhan Huang, Quanyan Zhu

**Abstracts**: In this work, we study the deception of a Linear-Quadratic-Gaussian (LQG) agent by manipulating the cost signals. We show that a small falsification on the cost parameters will only lead to a bounded change in the optimal policy and the bound is linear on the amount of falsification the attacker can apply on the cost parameters. We propose an attack model where the goal of the attacker is to mislead the agent into learning a `nefarious' policy with intended falsification on the cost parameters. We formulate the attack's problem as an optimization problem, which is proved to be convex, and developed necessary and sufficient conditions to check the achievability of the attacker's goal.   We showcase the adversarial manipulation on two types of LQG learners: the batch RL learner and the other is the adaptive dynamic programming (ADP) learner. Our results demonstrate that with only 2.296% of falsification on the cost data, the attacker misleads the batch RL into learning the 'nefarious' policy that leads the vehicle to a dangerous position. The attacker can also gradually trick the ADP learner into learning the same `nefarious' policy by consistently feeding the learner a falsified cost signal that stays close to the true cost signal. The aim of the paper is to raise people's awareness of the security threats faced by RL-enabled control systems.

摘要: 在这项工作中，我们通过操纵费用信号来研究线性二次高斯(LQG)代理的欺骗行为。我们证明了对代价参数的微小篡改只会导致最优策略有界的改变，并且攻击者可以对代价参数应用的伪造量是线性的。我们提出了一个攻击模型，其中攻击者的目标是误导代理学习具有故意篡改成本参数的“邪恶”策略。我们将攻击问题描述为一个优化问题，证明了该优化问题是凸的，并给出了检验攻击者目标可达性的充要条件。我们展示了在两种类型的LQG学习器上的对抗操作：批量RL学习器和自适应动态规划(ADP)学习器。我们的结果表明，由于只有2.296%的成本数据被篡改，攻击者误导批次RL学习将车辆引向危险位置的“邪恶”策略。攻击者还可以通过始终如一地向学习者提供接近真实成本信号的伪造成本信号，逐渐欺骗ADP学习者学习相同的“邪恶”策略。本文的目的是提高人们对启用RL的控制系统所面临的安全威胁的认识。



## **5. Single Loop Gaussian Homotopy Method for Non-convex Optimization**

求解非凸优化问题的单圈高斯同伦方法 math.OC

45 pages

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2203.05717v1)

**Authors**: Hidenori Iwakiri, Yuhang Wang, Shinji Ito, Akiko Takeda

**Abstracts**: The Gaussian homotopy (GH) method is a popular approach to finding better local minima for non-convex optimization problems by gradually changing the problem to be solved from a simple one to the original target one. Existing GH-based methods consisting of a double loop structure incur high computational costs, which may limit their potential for practical application. We propose a novel single loop framework for GH methods (SLGH) for both deterministic and stochastic settings. For those applications in which the convolution calculation required to build a GH function is difficult, we present zeroth-order SLGH algorithms with gradient-free oracles. The convergence rate of (zeroth-order) SLGH depends on the decreasing speed of a smoothing hyperparameter, and when the hyperparameter is chosen appropriately, it becomes consistent with the convergence rate of (zeroth-order) gradient descent. In experiments that included artificial highly non-convex examples and black-box adversarial attacks, we have demonstrated that our algorithms converge much faster than an existing double loop GH method while outperforming gradient descent-based methods in terms of finding a better solution.

摘要: 高斯同伦(GH)方法是一种流行的寻找非凸优化问题局部极小值的方法，它将待求解问题从简单问题逐步转化为原始目标问题。现有的基于双环结构的GH方法计算量大，限制了其实际应用的潜力。我们提出了一种新的单环GH方法框架(SLGH)，既适用于确定性环境，也适用于随机环境。对于构造GH函数所需的卷积计算困难的应用，我们提出了具有无梯度预言的零阶SLGH算法。(零阶)SLGH的收敛速度取决于平滑超参数的下降速度，当超参数选择适当时，收敛速度与(零阶)梯度下降的收敛速度一致。在包含人工高度非凸集和黑盒攻击的实验中，我们证明了我们的算法比现有的双环GH方法收敛速度快得多，并且在找到更好的解方面优于基于梯度下降的方法。



## **6. Formalizing and Estimating Distribution Inference Risks**

配电推理风险的形式化与估计 cs.LG

Update: New version with more theoretical results and a deeper  exploration of results. We noted some discrepancies in our experiments on the  CelebA dataset and re-ran all of our experiments for this dataset, updating  Table 1 and Figures 2c, 3b, 4, 7a, and 8a in the process. These did not  substantially impact our results, and our conclusions and observations in  trends remain unchanged

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2109.06024v5)

**Authors**: Anshuman Suri, David Evans

**Abstracts**: Distribution inference, sometimes called property inference, infers statistical properties about a training set from access to a model trained on that data. Distribution inference attacks can pose serious risks when models are trained on private data, but are difficult to distinguish from the intrinsic purpose of statistical machine learning -- namely, to produce models that capture statistical properties about a distribution. Motivated by Yeom et al.'s membership inference framework, we propose a formal definition of distribution inference attacks that is general enough to describe a broad class of attacks distinguishing between possible training distributions. We show how our definition captures previous ratio-based property inference attacks as well as new kinds of attack including revealing the average node degree or clustering coefficient of a training graph. To understand distribution inference risks, we introduce a metric that quantifies observed leakage by relating it to the leakage that would occur if samples from the training distribution were provided directly to the adversary. We report on a series of experiments across a range of different distributions using both novel black-box attacks and improved versions of the state-of-the-art white-box attacks. Our results show that inexpensive attacks are often as effective as expensive meta-classifier attacks, and that there are surprising asymmetries in the effectiveness of attacks.

摘要: 分布推理，有时被称为属性推理，从对基于该数据训练的模型的访问中推断出关于训练集的统计属性。当模型基于私有数据进行训练时，分布推断攻击可能会带来严重的风险，但很难与统计机器学习的内在目的区分开来--即，生成捕获有关分布的统计属性的模型。在Yeom等人的成员关系推理框架的启发下，我们提出了分布推理攻击的形式化定义，该定义足够通用，可以描述区分可能的训练分布的广泛的攻击类别。我们展示了我们的定义如何捕获以前的基于比率的属性推理攻击，以及新的攻击类型，包括揭示训练图的平均节点度或聚类系数。为了了解分布推理风险，我们引入了一个度量，通过将观察到的泄漏与训练分布的样本直接提供给对手时将发生的泄漏联系起来，对观察到的泄漏进行量化。我们报告了使用新颖的黑盒攻击和最先进的白盒攻击的改进版本在一系列不同的发行版上进行的一系列实验。我们的结果表明，廉价的攻击通常与昂贵的元分类器攻击一样有效，并且攻击的有效性存在惊人的不对称性。



## **7. TraSw: Tracklet-Switch Adversarial Attacks against Multi-Object Tracking**

TraSw：针对多目标跟踪的Tracklet-Switch敌意攻击 cs.CV

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2111.08954v2)

**Authors**: Delv Lin, Qi Chen, Chengyu Zhou, Kun He

**Abstracts**: Multi-Object Tracking (MOT) has achieved aggressive progress and derives many excellent deep learning models. However, the robustness of the trackers is rarely studied, and it is challenging to attack the MOT system since its mature association algorithms are designed to be robust against errors during the tracking. In this work, we analyze the vulnerability of popular pedestrian MOT trackers and propose a novel adversarial attack method called Tracklet-Switch (TraSw) against the complete tracking pipeline of MOT. TraSw can fool the advanced deep trackers (i.e., FairMOT and ByteTrack) to fail to track the targets in the subsequent frames by attacking very few frames. Experiments on the MOT-Challenge datasets (i.e., 2DMOT15, MOT17, and MOT20) show that TraSw can achieve an extraordinarily high success rate of over 95% by attacking only four frames on average. To our knowledge, this is the first work on the adversarial attack against pedestrian MOT trackers. The code is available at https://github.com/DerryHub/FairMOT-attack .

摘要: 多目标跟踪(MOT)取得了突破性的进展，衍生出了许多优秀的深度学习模型。然而，很少有人研究跟踪器的鲁棒性，而且由于其成熟的关联算法被设计成对跟踪过程中的错误具有鲁棒性，因此对MOT系统的攻击是具有挑战性的。在这项工作中，我们分析了流行的行人MOT跟踪器的脆弱性，并提出了一种新的针对MOT完整跟踪管道的对抗性攻击方法Tracklet-Switch(TraSw)。TraSw可以通过攻击很少的帧来欺骗高级深度跟踪器(即FairMOT和ByteTrack)，使其无法跟踪后续帧中的目标。在MOT-Challenger数据集(2DMOT15、MOT17和MOT20)上的实验表明，TraSw平均只攻击4帧，可以达到95%以上的超高成功率。据我们所知，这是针对行人MOT跟踪器的首次对抗性攻击。代码可在https://github.com/DerryHub/FairMOT-attack上获得。



## **8. SoK: On the Semantic AI Security in Autonomous Driving**

SOK：关于自动驾驶中的语义人工智能安全 cs.CR

Project website: https://sites.google.com/view/cav-sec/pass

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2203.05314v1)

**Authors**: Junjie Shen, Ningfei Wang, Ziwen Wan, Yunpeng Luo, Takami Sato, Zhisheng Hu, Xinyang Zhang, Shengjian Guo, Zhenyu Zhong, Kang Li, Ziming Zhao, Chunming Qiao, Qi Alfred Chen

**Abstracts**: Autonomous Driving (AD) systems rely on AI components to make safety and correct driving decisions. Unfortunately, today's AI algorithms are known to be generally vulnerable to adversarial attacks. However, for such AI component-level vulnerabilities to be semantically impactful at the system level, it needs to address non-trivial semantic gaps both (1) from the system-level attack input spaces to those at AI component level, and (2) from AI component-level attack impacts to those at the system level. In this paper, we define such research space as semantic AI security as opposed to generic AI security. Over the past 5 years, increasingly more research works are performed to tackle such semantic AI security challenges in AD context, which has started to show an exponential growth trend.   In this paper, we perform the first systematization of knowledge of such growing semantic AD AI security research space. In total, we collect and analyze 53 such papers, and systematically taxonomize them based on research aspects critical for the security field. We summarize 6 most substantial scientific gaps observed based on quantitative comparisons both vertically among existing AD AI security works and horizontally with security works from closely-related domains. With these, we are able to provide insights and potential future directions not only at the design level, but also at the research goal, methodology, and community levels. To address the most critical scientific methodology-level gap, we take the initiative to develop an open-source, uniform, and extensible system-driven evaluation platform, named PASS, for the semantic AD AI security research community. We also use our implemented platform prototype to showcase the capabilities and benefits of such a platform using representative semantic AD AI attacks.

摘要: 自动驾驶(AD)系统依赖人工智能组件来做出安全和正确的驾驶决策。不幸的是，众所周知，今天的人工智能算法通常容易受到对手的攻击。然而，要使这种AI组件级别的漏洞在系统级别产生语义影响，它需要解决以下两个方面的重要语义差距：(1)从系统级别的攻击输入空间到AI组件级别的输入空间，以及(2)从AI组件级别的攻击影响到系统级别的影响。在本文中，我们将这样的研究空间定义为语义人工智能安全，而不是一般的人工智能安全。在过去的5年里，越来越多的研究工作对撞击这样的语义AI在AD环境下的安全挑战进行了研究，并开始呈现指数增长的趋势。在本文中，我们首次对这种不断增长的语义AD AI安全研究空间的知识进行了系统化。我们总共收集和分析了53篇这样的论文，并根据对安全领域至关重要的研究方面对它们进行了系统的分类。我们总结了基于定量比较观察到的6个最实质性的科学差距，这6个差距既包括现有AD AI安全作品之间的纵向比较，也包括与密切相关领域的安全作品的横向比较。有了这些，我们不仅能够在设计层面上，而且在研究目标、方法和社区层面上提供洞察力和潜在的未来方向。为了解决最关键的科学方法论层面的差距，我们主动开发了一个开源的、统一的、可扩展的系统驱动的评估平台，名为PASS，用于语义AD AI安全研究社区。我们还使用我们实现的平台原型来展示这样一个使用典型语义AD AI攻击的平台的能力和好处。



## **9. Adversarial Attacks on Machinery Fault Diagnosis**

机械故障诊断中的对抗性攻击 cs.CR

5 pages, 5 figures. Submitted to Interspeech 2022

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2110.02498v2)

**Authors**: Jiahao Chen, Diqun Yan

**Abstracts**: Despite the great progress of neural network-based (NN-based) machinery fault diagnosis methods, their robustness has been largely neglected, for they can be easily fooled through adding imperceptible perturbation to the input. For fault diagnosis problems, in this paper, we reformulate various adversarial attacks and intensively investigate them under untargeted and targeted conditions. Experimental results on six typical NN-based models show that accuracies of the models are greatly reduced by adding small perturbations. We further propose a simple, efficient and universal scheme to protect the victim models. This work provides an in-depth look at adversarial examples of machinery vibration signals for developing protection methods against adversarial attack and improving the robustness of NN-based models.

摘要: 尽管基于神经网络(NN)的机械故障诊断方法有了很大的进步，但它们的鲁棒性很大程度上被忽略了，因为它们很容易通过在输入中添加不可察觉的扰动而被愚弄。针对故障诊断问题，本文对各种对抗性攻击进行了重新定义，并在无目标和有目标的情况下对其进行了深入的研究。在6个典型的神经网络模型上的实验结果表明，加入小扰动会大大降低模型的精度。在此基础上，提出了一种简单、高效、通用的受害者模型保护方案。这项工作对机械振动信号的对抗性实例进行了深入的研究，以开发针对对抗性攻击的保护方法，并提高基于神经网络的模型的鲁棒性。



## **10. Clustering Label Inference Attack against Practical Split Learning**

针对实用分裂学习的聚类标签推理攻击 cs.LG

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2203.05222v1)

**Authors**: Junlin Liu, Xinchen Lyu

**Abstracts**: Split learning is deemed as a promising paradigm for privacy-preserving distributed learning, where the learning model can be cut into multiple portions to be trained at the participants collaboratively. The participants only exchange the intermediate learning results at the cut layer, including smashed data via forward-pass (i.e., features extracted from the raw data) and gradients during backward-propagation.Understanding the security performance of split learning is critical for various privacy-sensitive applications.With the emphasis on private labels, this paper proposes a passive clustering label inference attack for practical split learning. The adversary (either clients or servers) can accurately retrieve the private labels by collecting the exchanged gradients and smashed data.We mathematically analyse potential label leakages in split learning and propose the cosine and Euclidean similarity measurements for clustering attack. Experimental results validate that the proposed approach is scalable and robust under different settings (e.g., cut layer positions, epochs, and batch sizes) for practical split learning.The adversary can still achieve accurate predictions, even when differential privacy and gradient compression are adopted for label protections.

摘要: 分裂学习被认为是一种很有前途的隐私保护分布式学习范例，它可以将学习模型分割成多个部分，在参与者处进行协作训练。参与者只在切割层交换中间学习结果，包括前向传递的粉碎数据(即从原始数据中提取的特征)和后向传播过程中的梯度，了解分裂学习的安全性能对于各种隐私敏感应用至关重要，该文以私有标签为重点，提出了一种用于实际分裂学习的被动聚类标签推理攻击。通过收集交换的梯度和粉碎的数据，攻击者(无论是客户端还是服务器)都可以准确地恢复私有标签，对分裂学习中潜在的标签泄漏进行了数学分析，并提出了基于余弦和欧几里德相似度量的聚类攻击方法。实验结果表明，该方法在不同环境(如切割层位置、历元、批次大小等)下具有较好的扩展性和鲁棒性，即使在采用差分隐私和梯度压缩进行标签保护的情况下，对手仍能获得准确的预测。



## **11. Membership Privacy Protection for Image Translation Models via Adversarial Knowledge Distillation**

基于对抗性知识提取的图像翻译模型成员隐私保护 cs.CV

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2203.05212v1)

**Authors**: Saeed Ranjbar Alvar, Lanjun Wang, Jian Pei, Yong Zhang

**Abstracts**: Image-to-image translation models are shown to be vulnerable to the Membership Inference Attack (MIA), in which the adversary's goal is to identify whether a sample is used to train the model or not. With daily increasing applications based on image-to-image translation models, it is crucial to protect the privacy of these models against MIAs.   We propose adversarial knowledge distillation (AKD) as a defense method against MIAs for image-to-image translation models. The proposed method protects the privacy of the training samples by improving the generalizability of the model. We conduct experiments on the image-to-image translation models and show that AKD achieves the state-of-the-art utility-privacy tradeoff by reducing the attack performance up to 38.9% compared with the regular training model at the cost of a slight drop in the quality of the generated output images. The experimental results also indicate that the models trained by AKD generalize better than the regular training models. Furthermore, compared with existing defense methods, the results show that at the same privacy protection level, image translation models trained by AKD generate outputs with higher quality; while at the same quality of outputs, AKD enhances the privacy protection over 30%.

摘要: 图像到图像的翻译模型容易受到成员关系推理攻击(MIA)的攻击，在MIA攻击中，对手的目标是识别是否使用样本来训练模型。随着基于图像到图像翻译模型的应用日益增多，保护这些模型的隐私免受MIA攻击变得至关重要。针对图像到图像翻译模型，我们提出了对抗性知识蒸馏(AKD)作为一种防御MIA的方法。该方法通过提高模型的泛化能力来保护训练样本的私密性。我们对图像到图像的转换模型进行了实验，结果表明，AKD在生成图像质量略有下降的情况下，与常规训练模型相比，攻击性能降低了38.9%，达到了最先进的效用-隐私折衷。实验结果还表明，AKD训练的模型比常规训练模型具有更好的泛化能力。此外，与现有的防御方法相比，实验结果表明，在相同的隐私保护水平下，AKD训练的图像翻译模型生成的输出质量更高，而在相同的输出质量下，AKD对隐私的保护提高了30%以上。



## **12. Practical Evaluation of Adversarial Robustness via Adaptive Auto Attack**

基于自适应自动攻击的对手健壮性实用评估 cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2203.05154v1)

**Authors**: Ye Liu, Yaya Cheng, Lianli Gao, Xianglong Liu, Qilong Zhang, Jingkuan Song

**Abstracts**: Defense models against adversarial attacks have grown significantly, but the lack of practical evaluation methods has hindered progress. Evaluation can be defined as looking for defense models' lower bound of robustness given a budget number of iterations and a test dataset. A practical evaluation method should be convenient (i.e., parameter-free), efficient (i.e., fewer iterations) and reliable (i.e., approaching the lower bound of robustness). Towards this target, we propose a parameter-free Adaptive Auto Attack (A$^3$) evaluation method which addresses the efficiency and reliability in a test-time-training fashion. Specifically, by observing that adversarial examples to a specific defense model follow some regularities in their starting points, we design an Adaptive Direction Initialization strategy to speed up the evaluation. Furthermore, to approach the lower bound of robustness under the budget number of iterations, we propose an online statistics-based discarding strategy that automatically identifies and abandons hard-to-attack images. Extensive experiments demonstrate the effectiveness of our A$^3$. Particularly, we apply A$^3$ to nearly 50 widely-used defense models. By consuming much fewer iterations than existing methods, i.e., $1/10$ on average (10$\times$ speed up), we achieve lower robust accuracy in all cases. Notably, we won $\textbf{first place}$ out of 1681 teams in CVPR 2021 White-box Adversarial Attacks on Defense Models competitions with this method. Code is available at: $\href{https://github.com/liuye6666/adaptive_auto_attack}{https://github.com/liuye6666/adaptive\_auto\_attack}$

摘要: 对抗对手攻击的防御模型已经显著增长，但缺乏实用的评估方法阻碍了进展。评估可以定义为在给定预算迭代次数和测试数据集的情况下寻找防御模型的健壮性下限。一种实用的评估方法应该是方便(即无参数)、高效(即迭代次数较少)和可靠(即接近鲁棒性的下界)。针对这一目标，我们提出了一种无参数的自适应自动攻击(A$^3$)评估方法，该方法以测试时间训练的方式来解决效率和可靠性问题。具体地说，通过观察特定防御模型的对抗性示例在起始点遵循一定的规律，我们设计了一种自适应方向初始化策略来加快评估速度。此外，为了在预算迭代次数下逼近鲁棒性的下界，我们提出了一种基于在线统计的丢弃策略，自动识别和丢弃不易攻击的图像。广泛的实验证明了我们的澳元^3元的有效性。特别是，我们将澳元^3美元应用于近50种广泛使用的防御模型。通过比现有方法消耗更少的迭代次数，即平均$1/10$(10$\倍$加速)，我们在所有情况下都获得了较低的鲁棒精度。值得注意的是，我们用这种方法赢得了CVPR 2021年白盒对抗性攻击防御模型比赛1681支队伍中的$\textbf{第一名}$。代码可在以下网址获得：$\href{https://github.com/liuye6666/adaptive_auto_attack}{https://github.com/liuye6666/adaptive\_auto\_attack}$



## **13. Frequency-driven Imperceptible Adversarial Attack on Semantic Similarity**

基于频率驱动的语义相似度潜伏攻击 cs.CV

10 pages, 7 figure, CVPR 2022 conference

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2203.05151v1)

**Authors**: Cheng Luo, Qinliang Lin, Weicheng Xie, Bizhu Wu, Jinheng Xie, Linlin Shen

**Abstracts**: Current adversarial attack research reveals the vulnerability of learning-based classifiers against carefully crafted perturbations. However, most existing attack methods have inherent limitations in cross-dataset generalization as they rely on a classification layer with a closed set of categories. Furthermore, the perturbations generated by these methods may appear in regions easily perceptible to the human visual system (HVS). To circumvent the former problem, we propose a novel algorithm that attacks semantic similarity on feature representations. In this way, we are able to fool classifiers without limiting attacks to a specific dataset. For imperceptibility, we introduce the low-frequency constraint to limit perturbations within high-frequency components, ensuring perceptual similarity between adversarial examples and originals. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) and three public online platforms indicate that our attack can yield misleading and transferable adversarial examples across architectures and datasets. Additionally, visualization results and quantitative performance (in terms of four different metrics) show that the proposed algorithm generates more imperceptible perturbations than the state-of-the-art methods. Code is made available at.

摘要: 目前的对抗性攻击研究揭示了基于学习的分类器对精心设计的扰动的脆弱性。然而，现有的大多数攻击方法在跨数据集泛化方面都有固有的局限性，因为它们依赖于具有封闭类别集的分类层。此外，由这些方法产生的扰动可能出现在人类视觉系统(HVS)容易察觉的区域。针对上述问题，我们提出了一种攻击特征表示语义相似度的新算法。通过这种方式，我们能够愚弄分类器，而不会将攻击限制在特定的数据集。对于不可感知性，我们引入了低频约束来限制高频分量内的扰动，以确保对抗性示例与原始示例之间的感知相似性。在三个数据集(CIFAR-10、CIFAR-100和ImageNet-1K)和三个公共在线平台上的广泛实验表明，我们的攻击可以产生跨体系结构和数据集的误导性和可转移的敌意示例。此外，可视化结果和量化性能(根据四个不同的度量)表明，该算法比现有的方法产生更多的不可察觉的扰动。代码可在以下位置获得：。



## **14. Controllable Evaluation and Generation of Physical Adversarial Patch on Face Recognition**

人脸识别中物理对抗性补丁的可控评价与生成 cs.CV

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2203.04623v2)

**Authors**: Xiao Yang, Yinpeng Dong, Tianyu Pang, Zihao Xiao, Hang Su, Jun Zhu

**Abstracts**: Recent studies have revealed the vulnerability of face recognition models against physical adversarial patches, which raises security concerns about the deployed face recognition systems. However, it is still challenging to ensure the reproducibility for most attack algorithms under complex physical conditions, which leads to the lack of a systematic evaluation of the existing methods. It is therefore imperative to develop a framework that can enable a comprehensive evaluation of the vulnerability of face recognition in the physical world. To this end, we propose to simulate the complex transformations of faces in the physical world via 3D-face modeling, which serves as a digital counterpart of physical faces. The generic framework allows us to control different face variations and physical conditions to conduct reproducible evaluations comprehensively. With this digital simulator, we further propose a Face3DAdv method considering the 3D face transformations and realistic physical variations. Extensive experiments validate that Face3DAdv can significantly improve the effectiveness of diverse physically realizable adversarial patches in both simulated and physical environments, against various white-box and black-box face recognition models.

摘要: 最近的研究揭示了人脸识别模型对物理对手补丁的脆弱性，这引发了人们对部署的人脸识别系统的安全担忧。然而，大多数攻击算法在复杂物理条件下的可重复性仍然是具有挑战性的，这导致对现有方法缺乏系统的评估。因此，当务之急是制定一个框架，使之能够全面评估现实世界中人脸识别的脆弱性。为此，我们建议通过3D人脸建模来模拟人脸在物理世界中的复杂变换，3D人脸建模是物理人脸的数字对应。通用框架允许我们控制不同的脸部变化和身体条件，以进行全面的可重复性评估。在此数字模拟器的基础上，我们进一步提出了一种考虑3D人脸变换和真实物理变化的Face3DAdv方法。大量实验证明，Face3DAdv能够显著提高各种物理可实现的对抗性补丁在模拟和物理环境中对抗各种白盒和黑盒人脸识别模型的有效性。



## **15. Security of quantum key distribution from generalised entropy accumulation**

广义熵积累下量子密钥分配的安全性 quant-ph

32 pages

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.04993v1)

**Authors**: Tony Metger, Renato Renner

**Abstracts**: The goal of quantum key distribution (QKD) is to establish a secure key between two parties connected by an insecure quantum channel. To use a QKD protocol in practice, one has to prove that it is secure against general attacks: even if an adversary performs a complicated attack involving all of the rounds of the protocol, they cannot gain useful information about the key. A much simpler task is to prove security against collective attacks, where the adversary is assumed to behave the same in each round. Using a recently developed information-theoretic tool called generalised entropy accumulation, we show that for a very broad class of QKD protocols, security against collective attacks implies security against general attacks. Compared to existing techniques such as the quantum de Finetti theorem or a previous version of entropy accumulation, our result can be applied much more broadly and easily: it does not require special assumptions on the protocol such as symmetry or a Markov property between rounds, its bounds are independent of the dimension of the underlying Hilbert space, and it can be applied to prepare-and-measure protocols directly without switching to an entanglement-based version.

摘要: 量子密钥分发(QKD)的目标是在通过不安全的量子信道连接的双方之间建立安全密钥。要在实践中使用QKD协议，必须证明它对一般攻击是安全的：即使对手执行了涉及协议所有轮的复杂攻击，他们也无法获得有关密钥的有用信息。一个简单得多的任务是证明针对集体攻击的安全性，假设对手在每一轮中的行为都是一样的。使用最近开发的称为广义熵积累的信息论工具，我们证明了对于非常广泛的一类QKD协议，针对集体攻击的安全性意味着针对一般攻击的安全性。与量子de Finetti定理或前一版本的熵积累等现有技术相比，我们的结果可以更广泛和更容易地应用：它不需要对协议进行特殊的假设，例如轮间的对称性或马尔可夫性质，它的界限与底层Hilbert空间的维数无关，并且它可以直接应用于制备和测量协议，而不需要切换到基于纠缠的版本。



## **16. Physics-aware Complex-valued Adversarial Machine Learning in Reconfigurable Diffractive All-optical Neural Network**

可重构衍射全光神经网络中的物理感知复值对抗性机器学习 cs.ET

34 pages, 4 figures

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.06055v1)

**Authors**: Ruiyang Chen, Yingjie Li, Minhan Lou, Jichao Fan, Yingheng Tang, Berardi Sensale-Rodriguez, Cunxi Yu, Weilu Gao

**Abstracts**: Diffractive optical neural networks have shown promising advantages over electronic circuits for accelerating modern machine learning (ML) algorithms. However, it is challenging to achieve fully programmable all-optical implementation and rapid hardware deployment. Furthermore, understanding the threat of adversarial ML in such system becomes crucial for real-world applications, which remains unexplored. Here, we demonstrate a large-scale, cost-effective, complex-valued, and reconfigurable diffractive all-optical neural networks system in the visible range based on cascaded transmissive twisted nematic liquid crystal spatial light modulators. With the assist of categorical reparameterization, we create a physics-aware training framework for the fast and accurate deployment of computer-trained models onto optical hardware. Furthermore, we theoretically analyze and experimentally demonstrate physics-aware adversarial attacks onto the system, which are generated from a complex-valued gradient-based algorithm. The detailed adversarial robustness comparison with conventional multiple layer perceptrons and convolutional neural networks features a distinct statistical adversarial property in diffractive optical neural networks. Our full stack of software and hardware provides new opportunities of employing diffractive optics in a variety of ML tasks and enabling the research on optical adversarial ML.

摘要: 与电子电路相比，衍射光学神经网络在加速现代机器学习(ML)算法方面显示出了巨大的优势。然而，要实现完全可编程的全光实现和快速的硬件部署是具有挑战性的。此外，了解敌意ML在这样的系统中的威胁对于现实世界的应用来说是至关重要的，这一点仍然有待探索。在这里，我们展示了一种基于级联透射式扭曲向列相液晶空间光调制器的大规模、高性价比、复值和可重构的可见光衍射全光神经网络系统。在分类重参数化的帮助下，我们创建了一个物理感知训练框架，用于快速准确地将计算机训练的模型部署到光学硬件上。此外，我们对基于复值梯度算法产生的物理感知的敌意攻击进行了理论分析和实验演示。与传统的多层感知器和卷积神经网络相比，衍射光学神经网络具有明显的统计对抗性。我们的全套软件和硬件提供了在各种ML任务中使用衍射光学的新机会，并使光学对抗性ML的研究成为可能。



## **17. Reverse Engineering $\ell_p$ attacks: A block-sparse optimization approach with recovery guarantees**

逆向工程$\ell_p$攻击：一种具有恢复保证的挡路稀疏优化方法 cs.LG

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.04886v1)

**Authors**: Darshan Thaker, Paris Giampouras, René Vidal

**Abstracts**: Deep neural network-based classifiers have been shown to be vulnerable to imperceptible perturbations to their input, such as $\ell_p$-bounded norm adversarial attacks. This has motivated the development of many defense methods, which are then broken by new attacks, and so on. This paper focuses on a different but related problem of reverse engineering adversarial attacks. Specifically, given an attacked signal, we study conditions under which one can determine the type of attack ($\ell_1$, $\ell_2$ or $\ell_\infty$) and recover the clean signal. We pose this problem as a block-sparse recovery problem, where both the signal and the attack are assumed to lie in a union of subspaces that includes one subspace per class and one subspace per attack type. We derive geometric conditions on the subspaces under which any attacked signal can be decomposed as the sum of a clean signal plus an attack. In addition, by determining the subspaces that contain the signal and the attack, we can also classify the signal and determine the attack type. Experiments on digit and face classification demonstrate the effectiveness of the proposed approach.

摘要: 基于深度神经网络的分类器很容易受到不可察觉的输入扰动，例如$\ellp$-有界范数的对抗性攻击。这推动了许多防御方法的发展，然后这些方法被新的攻击所打破，等等。本文关注的是一个不同但又相关的逆向工程对抗性攻击问题。具体地说，在给定攻击信号的情况下，我们研究了确定攻击类型($\ell_1$、$\ell_2$或$\ell_\infty$)并恢复干净信号的条件。我们把这个问题归结为一个挡路稀疏恢复问题，这里假设信号和攻击都位于一个子空间的并中，每个类包含一个子空间，每个攻击类型包含一个子空间。我们导出了子空间上的几何条件，在这些条件下，任何被攻击的信号都可以分解为一个干净的信号加一个攻击的和。此外，通过确定包含信号和攻击的子空间，还可以对信号进行分类，确定攻击类型。数字和人脸分类实验证明了该方法的有效性。



## **18. Defending Black-box Skeleton-based Human Activity Classifiers**

基于黑盒骨架防御的人类活动分类器 cs.CV

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.04713v1)

**Authors**: He Wang, Yunfeng Diao, Zichang Tan, Guodong Guo

**Abstracts**: Deep learning has been regarded as the `go to' solution for many tasks today, but its intrinsic vulnerability to malicious attacks has become a major concern. The vulnerability is affected by a variety of factors including models, tasks, data, and attackers. Consequently, methods such as Adversarial Training and Randomized Smoothing have been proposed to tackle the problem in a wide range of applications. In this paper, we investigate skeleton-based Human Activity Recognition, which is an important type of time-series data but under-explored in defense against attacks. Our method is featured by (1) a new Bayesian Energy-based formulation of robust discriminative classifiers, (2) a new parameterization of the adversarial sample manifold of actions, and (3) a new post-train Bayesian treatment on both the adversarial samples and the classifier. We name our framework Bayesian Energy-based Adversarial Training or BEAT. BEAT is straightforward but elegant, which turns vulnerable black-box classifiers into robust ones without sacrificing accuracy. It demonstrates surprising and universal effectiveness across a wide range of action classifiers and datasets, under various attacks.

摘要: 深度学习已被认为是当今许多任务的“去”解决方案，但其固有的易受恶意攻击的脆弱性已成为人们主要关注的问题。该漏洞受多种因素影响，包括模型、任务、数据和攻击者。因此，对抗性训练和随机化平滑等方法被提出，以解决撞击的这一问题，并得到了广泛的应用。在本文中，我们研究了基于骨架的人类活动识别，这是一种重要的时间序列数据类型，但在防御攻击方面还没有得到充分的探索。我们的方法的特点是(1)新的基于贝叶斯能量的鲁棒判别分类器公式，(2)新的对抗性样本动作流形的参数化，(3)对对抗性样本和分类器的新的训练后贝叶斯处理。我们将我们的框架命名为基于贝叶斯能量的对抗性训练或BEAT。BEAT是直截了当但优雅的，它将脆弱的黑匣子分类器变成了健壮的分类器，而不会牺牲准确性。在各种攻击下，它在广泛的动作分类器和数据集上展示了令人惊讶的和普遍的有效性。



## **19. Robust Federated Learning Against Adversarial Attacks for Speech Emotion Recognition**

语音情感识别中抗敌意攻击的鲁棒联合学习 cs.SD

11 pages, 6 figures, 3 tables

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.04696v1)

**Authors**: Yi Chang, Sofiane Laridi, Zhao Ren, Gregory Palmer, Björn W. Schuller, Marco Fisichella

**Abstracts**: Due to the development of machine learning and speech processing, speech emotion recognition has been a popular research topic in recent years. However, the speech data cannot be protected when it is uploaded and processed on servers in the internet-of-things applications of speech emotion recognition. Furthermore, deep neural networks have proven to be vulnerable to human-indistinguishable adversarial perturbations. The adversarial attacks generated from the perturbations may result in deep neural networks wrongly predicting the emotional states. We propose a novel federated adversarial learning framework for protecting both data and deep neural networks. The proposed framework consists of i) federated learning for data privacy, and ii) adversarial training at the training stage and randomisation at the testing stage for model robustness. The experiments show that our proposed framework can effectively protect the speech data locally and improve the model robustness against a series of adversarial attacks.

摘要: 近年来，随着机器学习和语音处理技术的发展，语音情感识别成为一个热门的研究课题。然而，在语音情感识别的物联网应用中，当语音数据被上传并在服务器上处理时，语音数据不能得到保护。此外，深层神经网络已被证明容易受到人类无法区分的敌意干扰。由扰动产生的对抗性攻击可能会导致深层神经网络错误地预测情绪状态。我们提出了一种新的联合对抗性学习框架，用于保护数据和深度神经网络。该框架包括：i)数据隐私的联合学习；ii)训练阶段的对抗性训练和模型鲁棒性测试阶段的随机化。实验表明，该框架能有效地保护语音数据的局部安全，提高了模型对一系列对抗性攻击的鲁棒性。



## **20. Shadows can be Dangerous: Stealthy and Effective Physical-world Adversarial Attack by Natural Phenomenon**

阴影可能是危险的：自然现象对物理世界的隐秘而有效的对抗性攻击 cs.CV

This paper has been accepted by CVPR2022. Code:  https://github.com/hncszyq/ShadowAttack

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.03818v2)

**Authors**: Yiqi Zhong, Xianming Liu, Deming Zhai, Junjun Jiang, Xiangyang Ji

**Abstracts**: Estimating the risk level of adversarial examples is essential for safely deploying machine learning models in the real world. One popular approach for physical-world attacks is to adopt the "sticker-pasting" strategy, which however suffers from some limitations, including difficulties in access to the target or printing by valid colors. A new type of non-invasive attacks emerged recently, which attempt to cast perturbation onto the target by optics based tools, such as laser beam and projector. However, the added optical patterns are artificial but not natural. Thus, they are still conspicuous and attention-grabbed, and can be easily noticed by humans. In this paper, we study a new type of optical adversarial examples, in which the perturbations are generated by a very common natural phenomenon, shadow, to achieve naturalistic and stealthy physical-world adversarial attack under the black-box setting. We extensively evaluate the effectiveness of this new attack on both simulated and real-world environments. Experimental results on traffic sign recognition demonstrate that our algorithm can generate adversarial examples effectively, reaching 98.23% and 90.47% success rates on LISA and GTSRB test sets respectively, while continuously misleading a moving camera over 95% of the time in real-world scenarios. We also offer discussions about the limitations and the defense mechanism of this attack.

摘要: 估计对抗性示例的风险水平对于在现实世界中安全地部署机器学习模型是至关重要的。物理世界攻击的一种流行方法是采用“粘贴”策略，但该策略受到一些限制，包括难以接近目标或以有效颜色打印。最近出现了一种新型的非侵入性攻击，它试图通过激光束和投影仪等基于光学的工具对目标进行摄动。然而，添加的光学图案是人造的，但不是自然的。因此，它们仍然是引人注目和引人注目的，很容易被人类注意到。本文研究了一种新的光学对抗实例，其中的扰动是由一种非常常见的自然现象--阴影产生的，从而在黑盒环境下实现了自然主义的、隐身的物理世界对抗攻击。我们广泛评估了这种新攻击在模拟和真实环境中的有效性。在交通标志识别上的实验结果表明，该算法能够有效地生成对抗性样本，在LISA和GTSRB测试集上的成功率分别达到98.23%和90.47%，而在真实场景中，95%以上的时间都能连续误导移动的摄像机。我们还讨论了这种攻击的局限性和防御机制。



## **21. Practical No-box Adversarial Attacks with Training-free Hybrid Image Transformation**

一种实用的免训练混合图像变换的无盒对抗性攻击 cs.CV

This is the revision (the previous version rated 8,8,5,4 in ICLR2022,  where 8 denotes "accept, good paper"), which has been further polished and  added many new experiments

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.04607v1)

**Authors**: Qilong Zhang, Chaoning Zhang, Chaoqun Li, Jingkuan Song, Lianli Gao, Heng Tao Shen

**Abstracts**: In recent years, the adversarial vulnerability of deep neural networks (DNNs) has raised increasing attention. Among all the threat models, no-box attacks are the most practical but extremely challenging since they neither rely on any knowledge of the target model or similar substitute model, nor access the dataset for training a new substitute model. Although a recent method has attempted such an attack in a loose sense, its performance is not good enough and computational overhead of training is expensive. In this paper, we move a step forward and show the existence of a \textbf{training-free} adversarial perturbation under the no-box threat model, which can be successfully used to attack different DNNs in real-time. Motivated by our observation that high-frequency component (HFC) domains in low-level features and plays a crucial role in classification, we attack an image mainly by manipulating its frequency components. Specifically, the perturbation is manipulated by suppression of the original HFC and adding of noisy HFC. We empirically and experimentally analyze the requirements of effective noisy HFC and show that it should be regionally homogeneous, repeating and dense. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our proposed no-box method. It attacks ten well-known models with a success rate of \textbf{98.13\%} on average, which outperforms state-of-the-art no-box attacks by \textbf{29.39\%}. Furthermore, our method is even competitive to mainstream transfer-based black-box attacks.

摘要: 近年来，深度神经网络(DNNs)的敌意脆弱性引起了越来越多的关注。在所有的威胁模型中，非盒子攻击是最实用但极具挑战性的，因为它们既不依赖于任何目标模型或类似替身模型的知识，也不需要访问数据集来训练新的替身模型。虽然最近的一种方法尝试了松散意义上的这种攻击，但其性能不够好，并且训练的计算开销很高。在本文中，我们进一步证明了在非盒子威胁模型下存在一个{无需训练}的对抗性扰动，该扰动可以成功地用于实时攻击不同的DNN。由于我们观察到高频分量(HFC)域存在于低层特征中，并且在分类中起着至关重要的作用，我们主要通过对图像的频率分量进行操作来攻击图像。具体地说，通过抑制原始HFC和添加噪声HFC来操纵该扰动。我们从经验和实验上分析了有效的噪声HFC的要求，指出它应该是区域均匀的、重复的和密集的。在ImageNet数据集上的大量实验证明了该方法的有效性。它攻击10个著名的模型，平均成功率为\textbf{98.13\%}，比最先进的非盒子攻击性能高\textbf{29.39\%}。此外，我们的方法甚至是好胜来主流的基于转账的黑匣子攻击。



## **22. The Dangerous Combo: Fileless Malware and Cryptojacking**

危险的组合：无文件恶意软件和密码劫持 cs.CR

9 Pages - Accepted to be published in SoutheastCon 2022 IEEE Region 3  Technical, Professional, and Student Conference. Mobile, Alabama, USA. Mar  31st to Apr 03rd 2022. https://ieeesoutheastcon.org/

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.03175v2)

**Authors**: Said Varlioglu, Nelly Elsayed, Zag ElSayed, Murat Ozer

**Abstracts**: Fileless malware and cryptojacking attacks have appeared independently as the new alarming threats in 2017. After 2020, fileless attacks have been devastating for victim organizations with low-observable characteristics. Also, the amount of unauthorized cryptocurrency mining has increased after 2019. Adversaries have started to merge these two different cyberattacks to gain more invisibility and profit under "Fileless Cryptojacking." This paper aims to provide a literature review in academic papers and industry reports for this new threat. Additionally, we present a new threat hunting-oriented DFIR approach with the best practices derived from field experience as well as the literature. Last, this paper reviews the fundamentals of the fileless threat that can also help ransomware researchers examine similar patterns.

摘要: 无文件恶意软件和密码劫持攻击已经独立出现，成为2017年新的令人担忧的威胁。2020年后，无文件攻击对具有低可察觉特征的受害者组织来说是毁灭性的。此外，2019年之后，未经授权的加密货币挖掘量有所增加。对手已经开始合并这两种不同的网络攻击，以在“无文件密码劫持”下获得更多的隐蔽性和利润。本文旨在对有关这一新威胁的学术论文和行业报告中的文献进行综述。此外，我们提出了一种新的面向威胁追捕的DFIR方法，该方法结合了来自现场经验和文献的最佳实践。最后，本文回顾了无文件威胁的基本原理，它也可以帮助勒索软件研究人员检查类似的模式。



## **23. Targeted Attack on Deep RL-based Autonomous Driving with Learned Visual Patterns**

基于学习视觉模式的深度RL自主驾驶目标攻击 cs.LG

7 pages, 4 figures; Accepted at ICRA 2022

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2109.07723v2)

**Authors**: Prasanth Buddareddygari, Travis Zhang, Yezhou Yang, Yi Ren

**Abstracts**: Recent studies demonstrated the vulnerability of control policies learned through deep reinforcement learning against adversarial attacks, raising concerns about the application of such models to risk-sensitive tasks such as autonomous driving. Threat models for these demonstrations are limited to (1) targeted attacks through real-time manipulation of the agent's observation, and (2) untargeted attacks through manipulation of the physical environment. The former assumes full access to the agent's states/observations at all times, while the latter has no control over attack outcomes. This paper investigates the feasibility of targeted attacks through visually learned patterns placed on physical objects in the environment, a threat model that combines the practicality and effectiveness of the existing ones. Through analysis, we demonstrate that a pre-trained policy can be hijacked within a time window, e.g., performing an unintended self-parking, when an adversarial object is present. To enable the attack, we adopt an assumption that the dynamics of both the environment and the agent can be learned by the attacker. Lastly, we empirically show the effectiveness of the proposed attack on different driving scenarios, perform a location robustness test, and study the tradeoff between the attack strength and its effectiveness. Code is available at https://github.com/ASU-APG/Targeted-Physical-Adversarial-Attacks-on-AD

摘要: 最近的研究表明，通过深度强化学习学习的控制策略在对抗对手攻击时存在脆弱性，这引发了人们对此类模型应用于自动驾驶等风险敏感任务的担忧。这些演示的威胁模型仅限于(1)通过实时操纵代理的观察进行的定向攻击，以及(2)通过操纵物理环境进行的非定向攻击。前者假定始终完全访问代理的状态/观察，而后者无法控制攻击结果。通过将视觉学习模式放置在环境中的物理对象上来研究定向攻击的可行性，这是一种结合了现有威胁模型的实用性和有效性的威胁模型。通过分析，我们证明了当存在敌对对象时，预先训练的策略可以在一个时间窗口内被劫持，例如，执行非故意的自停。为了启用攻击，我们采用了一个假设，即攻击者可以学习环境和代理的动态。最后，通过实验验证了所提出的攻击在不同驾驶场景下的有效性，并进行了位置鲁棒性测试，研究了攻击强度与有效性之间的权衡关系。代码可在https://github.com/ASU-APG/Targeted-Physical-Adversarial-Attacks-on-AD上获得



## **24. Machine Learning in NextG Networks via Generative Adversarial Networks**

基于产生式对抗网络的NextG网络机器学习 cs.LG

47 pages, 7 figures, 12 tables

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.04453v1)

**Authors**: Ender Ayanoglu, Kemal Davaslioglu, Yalin E. Sagduyu

**Abstracts**: Generative Adversarial Networks (GANs) are Machine Learning (ML) algorithms that have the ability to address competitive resource allocation problems together with detection and mitigation of anomalous behavior. In this paper, we investigate their use in next-generation (NextG) communications within the context of cognitive networks to address i) spectrum sharing, ii) detecting anomalies, and iii) mitigating security attacks. GANs have the following advantages. First, they can learn and synthesize field data, which can be costly, time consuming, and nonrepeatable. Second, they enable pre-training classifiers by using semi-supervised data. Third, they facilitate increased resolution. Fourth, they enable the recovery of corrupted bits in the spectrum. The paper provides the basics of GANs, a comparative discussion on different kinds of GANs, performance measures for GANs in computer vision and image processing as well as wireless applications, a number of datasets for wireless applications, performance measures for general classifiers, a survey of the literature on GANs for i)-iii) above, and future research directions. As a use case of GAN for NextG communications, we show that a GAN can be effectively applied for anomaly detection in signal classification (e.g., user authentication) outperforming another state-of-the-art ML technique such as an autoencoder.

摘要: 生成性对抗网络(GAN)是一种机器学习(ML)算法，具有解决好胜资源分配问题以及检测和缓解异常行为的能力。在本文中，我们研究了它们在认知网络环境下在下一代(NextG)通信中的应用，以解决i)频谱共享、ii)检测异常和iii)缓解安全攻击的问题。GAN具有以下优势。首先，他们可以学习和合成现场数据，这可能是昂贵、耗时和不可重复的。其次，它们通过使用半监督数据来启用预训练分类器。第三，它们有助于提高分辨率。第四，它们能够恢复频谱中的损坏比特。本文介绍了遗传算法的基本知识，不同类型的遗传算法的比较讨论，遗传算法在计算机视觉和图像处理以及无线应用中的性能度量，一些无线应用的数据集，通用分类器的性能度量，上述(I)-(Iii))关于遗传算法的文献综述，以及未来的研究方向。作为GAN在NextG通信中的应用，我们证明了GAN可以有效地应用于信号分类中的异常检测(例如，用户认证)，其性能优于另一种先进的ML技术，例如自动编码器。



## **25. DeepSE-WF: Unified Security Estimation for Website Fingerprinting Defenses**

DeepSE-WF：网站指纹防御的统一安全评估 cs.CR

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.04428v1)

**Authors**: Alexander Veicht, Cedric Renggli, Diogo Barradas

**Abstracts**: Website fingerprinting (WF) attacks, usually conducted with the help of a machine learning-based classifier, enable a network eavesdropper to pinpoint which web page a user is accessing through the inspection of traffic patterns. These attacks have been shown to succeed even when users browse the Internet through encrypted tunnels, e.g., through Tor or VPNs. To assess the security of new defenses against WF attacks, recent works have proposed feature-dependent theoretical frameworks that estimate the Bayes error of an adversary's features set or the mutual information leaked by manually-crafted features. Unfortunately, as state-of-the-art WF attacks increasingly rely on deep learning and latent feature spaces, security estimations based on simpler (and less informative) manually-crafted features can no longer be trusted to assess the potential success of a WF adversary in defeating such defenses. In this work, we propose DeepSE-WF, a novel WF security estimation framework that leverages specialized kNN-based estimators to produce Bayes error and mutual information estimates from learned latent feature spaces, thus bridging the gap between current WF attacks and security estimation methods. Our evaluation reveals that DeepSE-WF produces tighter security estimates than previous frameworks, reducing the required computational resources to output security estimations by one order of magnitude.

摘要: 通常在基于机器学习的分类器的帮助下进行的网站指纹(WF)攻击使网络窃听者能够通过检查流量模式来确定用户正在访问哪个网页。已经证明，即使当用户通过加密隧道(例如，通过ToR或VPN)浏览因特网时，这些攻击也是成功的。为了评估针对WF攻击的新防御措施的安全性，最近的工作提出了基于特征的理论框架，该框架估计对手的特征集的贝叶斯误差或人工创建的特征所泄露的互信息。不幸的是，随着最先进的WF攻击越来越依赖深度学习和潜在特征空间，基于更简单(且信息量更少)的手动创建的特征的安全评估不再值得信任，无法再评估WF对手在击败此类防御方面的潜在成功。在这项工作中，我们提出了DeepSE-WF，一个新的WF安全估计框架，它利用专门的基于KNN的估计器，从学习的潜在特征空间中产生贝叶斯误差和互信息估计，从而弥合了当前WF攻击和安全估计方法之间的差距。我们的评估显示，DeepSE-WF比以前的框架产生了更严格的安全估计，将输出安全估计所需的计算资源减少了一个数量级。



## **26. Disrupting Adversarial Transferability in Deep Neural Networks**

深度神经网络中破坏敌意可转移性的研究 cs.LG

20 pages, 13 figures

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2108.12492v2)

**Authors**: Christopher Wiedeman, Ge Wang

**Abstracts**: Adversarial attack transferability is well-recognized in deep learning. Prior work has partially explained transferability by recognizing common adversarial subspaces and correlations between decision boundaries, but little is known beyond this. We propose that transferability between seemingly different models is due to a high linear correlation between the feature sets that different networks extract. In other words, two models trained on the same task that are distant in the parameter space likely extract features in the same fashion, just with trivial affine transformations between the latent spaces. Furthermore, we show how applying a feature correlation loss, which decorrelates the extracted features in a latent space, can reduce the transferability of adversarial attacks between models, suggesting that the models complete tasks in semantically different ways. Finally, we propose a Dual Neck Autoencoder (DNA), which leverages this feature correlation loss to create two meaningfully different encodings of input information with reduced transferability.

摘要: 对抗性攻击的可转移性在深度学习中得到了广泛的认可。以前的工作已经通过识别共同的敌对子空间和决策边界之间的相关性部分解释了可转移性，但除此之外知之甚少。我们认为，在看似不同的模型之间的可转移性是由于不同网络提取的特征集之间具有高度的线性相关性。换句话说，在参数空间中相距较远的两个在同一任务上训练的模型很可能以相同的方式提取特征，只是在潜在空间之间进行了琐碎的仿射变换。此外，我们还展示了应用特征相关性损失(在潜在空间中去关联提取的特征)如何降低模型之间的对抗性攻击的可传递性，这表明两个模型以不同的语义方式完成任务。最后，我们提出了一种双颈自动编码器(DNA)，它利用这种特征相关性损失来创建两种有意义的不同的输入信息编码，降低了可传输性。



## **27. RAPTEE: Leveraging trusted execution environments for Byzantine-tolerant peer sampling services**

RAPTEE：为拜占庭容忍的对等采样服务利用可信执行环境 cs.DC

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.04258v1)

**Authors**: Matthieu Pigaglio, Joachim Bruneau-Queyreix, David Bromberg, Davide Frey, Etienne Rivière, Laurent Réveillère

**Abstracts**: Peer sampling is a first-class abstraction used in distributed systems for overlay management and information dissemination. The goal of peer sampling is to continuously build and refresh a partial and local view of the full membership of a dynamic, large-scale distributed system. Malicious nodes under the control of an adversary may aim at being over-represented in the views of correct nodes, increasing their impact on the proper operation of protocols built over peer sampling. State-of-the-art Byzantine resilient peer sampling protocols reduce this bias as long as Byzantines are not overly present. This paper studies the benefits brought to the resilience of peer sampling services when considering that a small portion of trusted nodes can run code whose authenticity and integrity can be assessed within a trusted execution environment, and specifically Intel's software guard extensions technology (SGX). We present RAPTEE, a protocol that builds and leverages trusted gossip-based communications to hamper an adversary's ability to increase its system-wide representation in the views of all nodes. We apply RAPTEE to BRAHMS, the most resilient peer sampling protocol to date. Experiments with 10,000 nodes show that with only 1% of SGX-capable devices, RAPTEE can reduce the proportion of Byzantine IDs in the view of honest nodes by up to 17% when the system contains 10% of Byzantine nodes. In addition, the security guarantees of RAPTEE hold even in the presence of a powerful attacker attempting to identify trusted nodes and injecting view-poisoned trusted nodes.

摘要: 对等抽样是分布式系统中用于覆盖管理和信息分发的一级抽象。对等抽样的目标是持续构建和刷新动态、大规模分布式系统的完整成员的局部和局部视图。在对手控制下的恶意节点可能旨在在正确节点的视图中被过度表示，从而增加它们对建立在对等采样之上的协议的正确操作的影响。只要拜占庭人不过度存在，最先进的拜占庭弹性对等采样协议就会减少这种偏见。本文研究了当考虑到一小部分可信节点可以在可信执行环境(特别是Intel的软件保护扩展技术(SGX))中运行其真实性和完整性可以被评估的代码时，给对等采样服务的弹性带来的好处。我们提出了RAPTEE，这是一种协议，它建立并利用基于可信八卦的通信来阻碍对手在所有节点的视图中增加其系统范围表示的能力。我们将RAPTEE应用于Brahms，这是迄今为止最具弹性的对等采样协议。在10000个节点上的实验表明，当系统包含10%的拜占庭节点时，RAPTEE可以在仅使用1%的SGX功能的设备的情况下，将拜占庭ID在诚实节点中的比例降低高达17%。此外，即使强大的攻击者试图识别受信任节点并注入视图中毒的受信任节点，RAPTEE的安全保证仍然有效。



## **28. Towards Effective and Robust Neural Trojan Defenses via Input Filtering**

通过输入过滤实现高效、健壮的神经木马防御 cs.CR

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2202.12154v3)

**Authors**: Kien Do, Haripriya Harikumar, Hung Le, Dung Nguyen, Truyen Tran, Santu Rana, Dang Nguyen, Willy Susilo, Svetha Venkatesh

**Abstracts**: Trojan attacks on deep neural networks are both dangerous and surreptitious. Over the past few years, Trojan attacks have advanced from using only a single input-agnostic trigger and targeting only one class to using multiple, input-specific triggers and targeting multiple classes. However, Trojan defenses have not caught up with this development. Most defense methods still make out-of-date assumptions about Trojan triggers and target classes, thus, can be easily circumvented by modern Trojan attacks. To deal with this problem, we propose two novel "filtering" defenses called Variational Input Filtering (VIF) and Adversarial Input Filtering (AIF) which leverage lossy data compression and adversarial learning respectively to effectively purify all potential Trojan triggers in the input at run time without making assumptions about the number of triggers/target classes or the input dependence property of triggers. In addition, we introduce a new defense mechanism called "Filtering-then-Contrasting" (FtC) which helps avoid the drop in classification accuracy on clean data caused by "filtering", and combine it with VIF/AIF to derive new defenses of this kind. Extensive experimental results and ablation studies show that our proposed defenses significantly outperform well-known baseline defenses in mitigating five advanced Trojan attacks including two recent state-of-the-art while being quite robust to small amounts of training data and large-norm triggers.

摘要: 特洛伊木马对深层神经网络的攻击既危险又隐蔽。在过去的几年中，特洛伊木马攻击已经从只使用一个与输入无关的触发器和只针对一个类发展到使用多个特定于输入的触发器和以多个类为目标。然而，特洛伊木马防御并没有跟上这一发展。大多数防御方法仍然对木马触发器和目标类做出过时的假设，因此很容易被现代木马攻击所规避。为了解决这一问题，我们提出了两种新的“过滤”防御机制，称为变量输入过滤(VIF)和对抗性输入过滤(AIF)，它们分别利用有损数据压缩和对抗性学习在运行时有效地净化输入中所有潜在的木马触发器，而不需要假设触发器/目标类的数量或触发器的输入依赖属性。此外，我们引入了一种新的防御机制，称为“过滤-然后-对比”(FTC)，它有助于避免“过滤”导致的对干净数据分类精度的下降，并将其与VIF/AIF相结合来派生出新的防御机制。广泛的实验结果和烧蚀研究表明，我们提出的防御方案在缓解五种高级木马攻击(包括最近的两种)方面明显优于众所周知的基线防御方案，同时对少量训练数据和大范数触发事件具有相当的鲁棒性。



## **29. Adaptative Perturbation Patterns: Realistic Adversarial Learning for Robust NIDS**

自适应扰动模式：鲁棒NIDS的现实对抗性学习 cs.CR

16 pages, 6 tables, 8 figures, Future Internet journal

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.04234v1)

**Authors**: João Vitorino, Nuno Oliveira, Isabel Praça

**Abstracts**: Adversarial attacks pose a major threat to machine learning and to the systems that rely on it. Nonetheless, adversarial examples cannot be freely generated for domains with tabular data, such as cybersecurity. This work establishes the fundamental constraint levels required to achieve realism and introduces the Adaptative Perturbation Pattern Method (A2PM) to fulfill these constraints in a gray-box setting. A2PM relies on pattern sequences that are independently adapted to the characteristics of each class to create valid and coherent data perturbations. The developed method was evaluated in a cybersecurity case study with two scenarios: Enterprise and Internet of Things (IoT) networks. Multilayer Perceptron (MLP) and Random Forest (RF) classifiers were created with regular and adversarial training, using the CIC-IDS2017 and IoT-23 datasets. In each scenario, targeted and untargeted attacks were performed against the classifiers, and the generated examples were compared with the original network traffic flows to assess their realism. The obtained results demonstrate that A2PM provides a time efficient generation of realistic adversarial examples, which can be advantageous for both adversarial training and attacks.

摘要: 对抗性攻击对机器学习和依赖机器学习的系统构成了重大威胁。尽管如此，对于具有表格数据的域(如网络安全)，不能免费生成敌意示例。这项工作建立了达到真实感所需的基本约束水平，并引入了自适应扰动模式方法(A2PM)来满足灰箱设置中的这些约束。A2PM依赖于独立适应每个类别的特征的模式序列来创建有效且一致的数据扰动。开发的方法在两个场景的网络安全案例研究中进行了评估：企业和物联网(IoT)网络。使用CIC-IDS2017和IoT-23数据集，通过定期和对抗性训练创建了多层感知器(MLP)和随机森林(RF)分类器。在每个场景中，对分类器执行目标攻击和非目标攻击，并将生成的示例与原始网络流量进行比较，以评估其真实性。所获得的结果表明，A2 PM提供了一种时间效率高的真实对抗性示例的生成，这对于对抗性训练和攻击都是有利的。



## **30. Robustly-reliable learners under poisoning attacks**

中毒攻击下健壮可靠的学习者 cs.LG

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.04160v1)

**Authors**: Maria-Florina Balcan, Avrim Blum, Steve Hanneke, Dravyansh Sharma

**Abstracts**: Data poisoning attacks, in which an adversary corrupts a training set with the goal of inducing specific desired mistakes, have raised substantial concern: even just the possibility of such an attack can make a user no longer trust the results of a learning system. In this work, we show how to achieve strong robustness guarantees in the face of such attacks across multiple axes.   We provide robustly-reliable predictions, in which the predicted label is guaranteed to be correct so long as the adversary has not exceeded a given corruption budget, even in the presence of instance targeted attacks, where the adversary knows the test example in advance and aims to cause a specific failure on that example. Our guarantees are substantially stronger than those in prior approaches, which were only able to provide certificates that the prediction of the learning algorithm does not change, as opposed to certifying that the prediction is correct, as we are able to achieve in our work. Remarkably, we provide a complete characterization of learnability in this setting, in particular, nearly-tight matching upper and lower bounds on the region that can be certified, as well as efficient algorithms for computing this region given an ERM oracle. Moreover, for the case of linear separators over logconcave distributions, we provide efficient truly polynomial time algorithms (i.e., non-oracle algorithms) for such robustly-reliable predictions.   We also extend these results to the active setting where the algorithm adaptively asks for labels of specific informative examples, and the difficulty is that the adversary might even be adaptive to this interaction, as well as to the agnostic learning setting where there is no perfect classifier even over the uncorrupted data.

摘要: 数据中毒攻击，即对手破坏训练集，目的是诱导特定的预期错误，已经引起了极大的担忧：即使是这种攻击的可能性也会让用户不再信任学习系统的结果。在这项工作中，我们展示了如何在面对这种跨越多个轴的攻击时实现强鲁棒性保证。我们提供鲁棒可靠的预测，其中只要对手没有超过给定的腐败预算，即使在存在实例目标攻击的情况下，预测的标签也被保证是正确的，其中对手提前知道测试示例并旨在导致该示例上的特定失败。我们的保证比以前的方法要强得多，以前的方法只能提供学习算法的预测不变的证书，而不是像我们在工作中能够实现的那样，证明预测是正确的。值得注意的是，在这种情况下，我们给出了可学习性的完整刻画，特别是在可证明的区域的上下界几乎紧匹配的情况下，以及在给定ERM预言的情况下计算该区域的高效算法。此外，对于对数凹分布上线性分隔符的情况，我们为这种鲁棒可靠的预测提供了有效的真多项式时间算法(即非Oracle算法)。我们还将这些结果扩展到主动设置，其中算法自适应地要求特定信息示例的标签，困难在于对手甚至可能适应这种交互，以及即使在未被破坏的数据上也没有完美分类器的不可知学习设置。



## **31. Adversarial Texture for Fooling Person Detectors in the Physical World**

物理世界中愚人探测器的对抗性纹理 cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.03373v2)

**Authors**: Zhanhao Hu, Siyuan Huang, Xiaopei Zhu, Xiaolin Hu, Fuchun Sun, Bo Zhang

**Abstracts**: Nowadays, cameras equipped with AI systems can capture and analyze images to detect people automatically. However, the AI system can make mistakes when receiving deliberately designed patterns in the real world, i.e., physical adversarial examples. Prior works have shown that it is possible to print adversarial patches on clothes to evade DNN-based person detectors. However, these adversarial examples could have catastrophic drops in the attack success rate when the viewing angle (i.e., the camera's angle towards the object) changes. To perform a multi-angle attack, we propose Adversarial Texture (AdvTexture). AdvTexture can cover clothes with arbitrary shapes so that people wearing such clothes can hide from person detectors from different viewing angles. We propose a generative method, named Toroidal-Cropping-based Expandable Generative Attack (TC-EGA), to craft AdvTexture with repetitive structures. We printed several pieces of cloth with AdvTexure and then made T-shirts, skirts, and dresses in the physical world. Experiments showed that these clothes could fool person detectors in the physical world.

摘要: 如今，配备人工智能系统的摄像头可以捕捉和分析图像，自动检测人。然而，当接收到现实世界中故意设计的模式时，人工智能系统可能会出错，即物理对抗性示例。以前的工作已经表明，可以在衣服上打印敌意补丁来躲避基于DNN的人检测器。然而，当视角(即相机朝向对象的角度)改变时，这些对抗性的例子可能会使攻击成功率灾难性地下降。为了进行多角度攻击，我们提出了对抗性纹理(AdvTexture)。AdvTexture可以覆盖任意形状的衣服，这样穿着这种衣服的人就可以从不同的视角躲避人的探测器。提出了一种基于环形裁剪的可扩展生成攻击方法(TC-EGA)来制作具有重复结构的AdvTexture。我们用AdvTexure打印了几块布，然后在现实世界中制作了T恤、裙子和连衣裙。实验表明，这些衣服可以愚弄物理世界中的人体探测器。



## **32. Shape-invariant 3D Adversarial Point Clouds**

形状不变的三维对抗性点云 cs.CV

Accepted at CVPR 2022

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.04041v1)

**Authors**: Qidong Huang, Xiaoyi Dong, Dongdong Chen, Hang Zhou, Weiming Zhang, Nenghai Yu

**Abstracts**: Adversary and invisibility are two fundamental but conflict characters of adversarial perturbations. Previous adversarial attacks on 3D point cloud recognition have often been criticized for their noticeable point outliers, since they just involve an "implicit constrain" like global distance loss in the time-consuming optimization to limit the generated noise. While point cloud is a highly structured data format, it is hard to metric and constrain its perturbation with a simple loss properly. In this paper, we propose a novel Point-Cloud Sensitivity Map to boost both the efficiency and imperceptibility of point perturbations. This map reveals the vulnerability of point cloud recognition models when encountering shape-invariant adversarial noises. These noises are designed along the shape surface with an "explicit constrain" instead of extra distance loss. Specifically, we first apply a reversible coordinate transformation on each point of the point cloud input, to reduce one degree of point freedom and limit its movement on the tangent plane. Then we calculate the best attacking direction with the gradients of the transformed point cloud obtained on the white-box model. Finally we assign each point with a non-negative score to construct the sensitivity map, which benefits both white-box adversarial invisibility and black-box query-efficiency extended in our work. Extensive evaluations prove that our method can achieve the superior performance on various point cloud recognition models, with its satisfying adversarial imperceptibility and strong resistance to different point cloud defense settings. Our code is available at: https://github.com/shikiw/SI-Adv.

摘要: 对抗性和隐蔽性是对抗性扰动的两个基本但又相互冲突的特征。以前针对3D点云识别的敌意攻击经常因为其明显的点离群值而受到批评，因为它们只是在耗时的优化过程中涉及诸如全局距离损失这样的“隐式约束”，以限制生成的噪声。虽然点云是一种高度结构化的数据格式，但是很难用简单的损失来度量和约束它的扰动。在本文中，我们提出了一种新的点云敏感度图，以提高点扰动的效率和隐蔽性。这张地图揭示了点云识别模型在遇到形状不变的对抗性噪声时的脆弱性。这些噪波是沿着形状表面设计的，带有“显式约束”，而不是额外的距离损失。具体地说，我们首先对点云输入的每个点应用可逆坐标变换，以减少一个点自由度并限制其在切面上的移动。然后利用白盒模型得到的变换后的点云梯度计算最佳攻击方向。最后，我们给每个点分配一个非负分数来构造敏感度图，这样既有利于白盒对抗不可见性，也有利于提高黑盒查询效率。广泛的评测表明，该方法在各种点云识别模型上都能取得较好的性能，具有令人满意的对抗性和对不同点云防御设置的较强抵抗力。我们的代码可从以下网址获得：https://github.com/shikiw/SI-Adv.



## **33. ART-Point: Improving Rotation Robustness of Point Cloud Classifiers via Adversarial Rotation**

ART-Point：通过对抗性轮换提高点云分类器的旋转稳健性 cs.CV

CVPR 2022

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2203.03888v1)

**Authors**: Robin Wang, Yibo Yang, Dacheng Tao

**Abstracts**: Point cloud classifiers with rotation robustness have been widely discussed in the 3D deep learning community. Most proposed methods either use rotation invariant descriptors as inputs or try to design rotation equivariant networks. However, robust models generated by these methods have limited performance under clean aligned datasets due to modifications on the original classifiers or input space. In this study, for the first time, we show that the rotation robustness of point cloud classifiers can also be acquired via adversarial training with better performance on both rotated and clean datasets. Specifically, our proposed framework named ART-Point regards the rotation of the point cloud as an attack and improves rotation robustness by training the classifier on inputs with Adversarial RoTations. We contribute an axis-wise rotation attack that uses back-propagated gradients of the pre-trained model to effectively find the adversarial rotations. To avoid model over-fitting on adversarial inputs, we construct rotation pools that leverage the transferability of adversarial rotations among samples to increase the diversity of training data. Moreover, we propose a fast one-step optimization to efficiently reach the final robust model. Experiments show that our proposed rotation attack achieves a high success rate and ART-Point can be used on most existing classifiers to improve the rotation robustness while showing better performance on clean datasets than state-of-the-art methods.

摘要: 具有旋转鲁棒性的点云分类器在三维深度学习领域得到了广泛的讨论。大多数提出的方法要么使用旋转不变描述符作为输入，要么尝试设计旋转等变网络。然而，由于对原始分类器或输入空间的修改，这些方法生成的鲁棒模型在干净的对齐数据集上的性能有限。在这项研究中，我们首次表明，点云分类器的旋转鲁棒性也可以通过对抗性训练获得，在旋转数据集和清洁数据集上都具有更好的性能。具体地说，我们提出的ART-Point框架将点云的旋转视为一种攻击，并通过对具有对抗性旋转的输入训练分类器来提高旋转的鲁棒性。我们提出了一种轴向旋转攻击，它使用预训练模型的反向传播梯度来有效地找到对抗性旋转。为了避免对抗性输入的模型过拟合，我们构建了轮转池，利用对抗性轮换在样本之间的可传递性来增加训练数据的多样性。此外，我们还提出了一种快速的一步优化方法来高效地得到最终的鲁棒模型。实验表明，我们提出的旋转攻击取得了很高的成功率，ART-Point可以在现有的大多数分类器上使用，以提高旋转的鲁棒性，同时在干净的数据集上表现出比现有方法更好的性能。



## **34. Submodularity-based False Data Injection Attack Scheme in Multi-agent Dynamical Systems**

多智能体动态系统中基于子模块性的虚假数据注入攻击方案 math.DS

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2201.06017v2)

**Authors**: Xiaoyu Luo, Chengcheng Zhao, Chongrong Fang, Jianping He

**Abstracts**: Consensus in multi-agent dynamical systems is prone to be sabotaged by the adversary, which has attracted much attention due to its key role in broad applications. In this paper, we study a new false data injection (FDI) attack design problem, where the adversary with limited capability aims to select a subset of agents and manipulate their local multi-dimensional states to maximize the consensus convergence error. We first formulate the FDI attack design problem as a combinatorial optimization problem and prove it is NP-hard. Then, based on the submodularity optimization theory, we show the convergence error is a submodular function of the set of the compromised agents, which satisfies the property of diminishing marginal returns. In other words, the benefit of adding an extra agent to the compromised set decreases as that set becomes larger. With this property, we exploit the greedy scheme to find the optimal compromised agent set that can produce the maximum convergence error when adding one extra agent to that set each time. Thus, the FDI attack set selection algorithms are developed to obtain the near-optimal subset of the compromised agents. Furthermore, we derive the analytical suboptimality bounds and the worst-case running time under the proposed algorithms. Extensive simulation results are conducted to show the effectiveness of the proposed algorithm.

摘要: 多智能体动态系统中的共识容易受到对手的破坏，因其在广泛应用中的关键作用而备受关注。本文研究了一类新的虚假数据注入(FDI)攻击设计问题，其中能力有限的敌手的目标是选择一个Agent子集并操纵其局部多维状态以最大化共识收敛误差。我们首先将FDI攻击设计问题描述为一个组合优化问题，并证明了它是NP难的。然后，基于子模优化理论，证明了收敛误差是折衷智能体集合的子模函数，满足边际收益递减的性质。换句话说，向受危害的集合添加额外代理的好处随着该集合变得更大而降低。利用这一性质，我们利用贪婪方案来寻找最优的折衷智能体集合，该集合在每次额外增加一个智能体的情况下可以产生最大的收敛误差。因此，开发了FDI攻击集选择算法，以获得受攻击代理的近优子集。此外，我们还给出了所提出算法的分析次优界和最坏情况下的运行时间。大量的仿真结果表明了该算法的有效性。



## **35. Adversarial Attacks in Cooperative AI**

协作式人工智能中的对抗性攻击 cs.LG

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2111.14833v3)

**Authors**: Ted Fujimoto, Arthur Paul Pedersen

**Abstracts**: Single-agent reinforcement learning algorithms in a multi-agent environment are inadequate for fostering cooperation. If intelligent agents are to interact and work together to solve complex problems, methods that counter non-cooperative behavior are needed to facilitate the training of multiple agents. This is the goal of cooperative AI. Recent research in adversarial machine learning, however, shows that models (e.g., image classifiers) can be easily deceived into making inferior decisions. Meanwhile, an important line of research in cooperative AI has focused on introducing algorithmic improvements that accelerate learning of optimally cooperative behavior. We argue that prominent methods of cooperative AI are exposed to weaknesses analogous to those studied in prior machine learning research. More specifically, we show that three algorithms inspired by human-like social intelligence are, in principle, vulnerable to attacks that exploit weaknesses introduced by cooperative AI's algorithmic improvements and report experimental findings that illustrate how these vulnerabilities can be exploited in practice.

摘要: 多智能体环境中的单智能体强化学习算法不能很好地促进协作。如果智能Agent要交互并共同工作来解决复杂问题，就需要针对不合作行为的方法，以便于多个Agent的训练。这是合作AI的目标。然而，最近在对抗性机器学习方面的研究表明，模型(例如，图像分类器)很容易被欺骗，从而做出较差的决策。同时，合作人工智能的一条重要研究方向是引入算法改进，以加速最优合作行为的学习。我们认为，合作人工智能的突出方法暴露了与先前机器学习研究中所研究的类似的弱点。更具体地说，我们证明了三种受类人类社会智能启发的算法原则上容易受到攻击，这些攻击利用合作AI的算法改进引入的弱点，并报告了实验结果，说明了如何在实践中利用这些漏洞。



## **36. Taxonomy of Machine Learning Safety: A Survey and Primer**

机器学习安全分类学：综述与入门读本 cs.LG

**SubmitDate**: 2022-03-08    [paper-pdf](http://arxiv.org/pdf/2106.04823v2)

**Authors**: Sina Mohseni, Haotao Wang, Zhiding Yu, Chaowei Xiao, Zhangyang Wang, Jay Yadawa

**Abstracts**: The open-world deployment of Machine Learning (ML) algorithms in safety-critical applications such as autonomous vehicles needs to address a variety of ML vulnerabilities such as interpretability, verifiability, and performance limitations. Research explores different approaches to improve ML dependability by proposing new models and training techniques to reduce generalization error, achieve domain adaptation, and detect outlier examples and adversarial attacks. However, there is a missing connection between ongoing ML research and well-established safety principles. In this paper, we present a structured and comprehensive review of ML techniques to improve the dependability of ML algorithms in uncontrolled open-world settings. From this review, we propose the Taxonomy of ML Safety that maps state-of-the-art ML techniques to key engineering safety strategies. Our taxonomy of ML safety presents a safety-oriented categorization of ML techniques to provide guidance for improving dependability of the ML design and development. The proposed taxonomy can serve as a safety checklist to aid designers in improving coverage and diversity of safety strategies employed in any given ML system.

摘要: 机器学习(ML)算法在自动驾驶汽车等安全关键型应用中的开放世界部署需要解决各种ML漏洞，如可解释性、可验证性和性能限制。研究探索了通过提出新的模型和训练技术来提高ML可靠性的不同方法，以减少泛化错误，实现领域自适应，并检测离群点示例和敌意攻击。然而，正在进行的ML研究和公认的安全原则之间缺少联系。本文对ML技术进行了系统全面的综述，以提高ML算法在非受控开放环境下的可靠性。从这篇综述中，我们提出了ML安全分类法，它将最先进的ML技术映射到关键的工程安全策略。我们的ML安全分类法对ML技术进行了面向安全的分类，为提高ML设计和开发的可靠性提供指导。建议的分类可以作为安全检查表，帮助设计者提高在任何给定ML系统中采用的安全策略的复盖率和多样性。



## **37. Defending Graph Convolutional Networks against Dynamic Graph Perturbations via Bayesian Self-supervision**

基于贝叶斯自监督的图卷积网络抗动态图扰动 cs.LG

The paper is accepted by AAAI 2022

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2203.03762v1)

**Authors**: Jun Zhuang, Mohammad Al Hasan

**Abstracts**: In recent years, plentiful evidence illustrates that Graph Convolutional Networks (GCNs) achieve extraordinary accomplishments on the node classification task. However, GCNs may be vulnerable to adversarial attacks on label-scarce dynamic graphs. Many existing works aim to strengthen the robustness of GCNs; for instance, adversarial training is used to shield GCNs against malicious perturbations. However, these works fail on dynamic graphs for which label scarcity is a pressing issue. To overcome label scarcity, self-training attempts to iteratively assign pseudo-labels to highly confident unlabeled nodes but such attempts may suffer serious degradation under dynamic graph perturbations. In this paper, we generalize noisy supervision as a kind of self-supervised learning method and then propose a novel Bayesian self-supervision model, namely GraphSS, to address the issue. Extensive experiments demonstrate that GraphSS can not only affirmatively alert the perturbations on dynamic graphs but also effectively recover the prediction of a node classifier when the graph is under such perturbations. These two advantages prove to be generalized over three classic GCNs across five public graph datasets.

摘要: 近年来，大量的证据表明，图卷积网络(GCNS)在节点分类任务中取得了非凡的成就。然而，在标签稀缺的动态图上，GCNS可能容易受到敌意攻击。许多现有的工作都是为了增强GCNS的健壮性，例如，使用对抗性训练来保护GCNS免受恶意干扰。然而，这些工作在动态图上失败了，因为对于动态图来说，标签稀缺是一个紧迫的问题。为了克服标签稀缺性，自训练试图迭代地将伪标签分配给高度自信的无标签节点，但是这种尝试在动态图扰动下可能遭受严重降级。本文将噪声监督推广为一种自监督学习方法，并提出了一种新的贝叶斯自监督模型GraphSS来解决这一问题。大量实验表明，GraphSS不仅能肯定地告警动态图上的扰动，而且当图受到扰动时，还能有效地恢复节点分类器的预测。事实证明，这两个优势在五个公共图数据集上的三个经典GCN上得到了推广。



## **38. Art-Attack: Black-Box Adversarial Attack via Evolutionary Art**

艺术攻击：通过进化艺术进行的黑箱对抗性攻击 cs.CR

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2203.04405v1)

**Authors**: Phoenix Williams, Ke Li

**Abstracts**: Deep neural networks (DNNs) have achieved state-of-the-art performance in many tasks but have shown extreme vulnerabilities to attacks generated by adversarial examples. Many works go with a white-box attack that assumes total access to the targeted model including its architecture and gradients. A more realistic assumption is the black-box scenario where an attacker only has access to the targeted model by querying some input and observing its predicted class probabilities. Different from most prevalent black-box attacks that make use of substitute models or gradient estimation, this paper proposes a gradient-free attack by using a concept of evolutionary art to generate adversarial examples that iteratively evolves a set of overlapping transparent shapes. To evaluate the effectiveness of our proposed method, we attack three state-of-the-art image classification models trained on the CIFAR-10 dataset in a targeted manner. We conduct a parameter study outlining the impact the number and type of shapes have on the proposed attack's performance. In comparison to state-of-the-art black-box attacks, our attack is more effective at generating adversarial examples and achieves a higher attack success rate on all three baseline models.

摘要: 深度神经网络(DNNs)在许多任务中取得了最先进的性能，但对敌意例子产生的攻击表现出极大的脆弱性。许多作品都采用白盒攻击，假定可以完全访问目标模型，包括其体系结构和渐变。一个更现实的假设是黑盒场景，在黑盒场景中，攻击者只能通过查询一些输入并观察其预测的类别概率来访问目标模型。与目前流行的基于替身模型或梯度估计的黑盒攻击不同，该文利用进化艺术的概念，提出了一种无梯度攻击，通过迭代进化出一组重叠透明形状的对抗性示例。为了评估我们提出的方法的有效性，我们有针对性地攻击了三个在CIFAR-10数据集上训练的最先进的图像分类模型。我们进行了一项参数研究，概述了形状的数量和类型对拟议攻击性能的影响。与最先进的黑盒攻击相比，我们的攻击在生成对抗性示例方面更有效，并且在所有三个基线模型上都实现了更高的攻击成功率。



## **39. Uncertify: Attacks Against Neural Network Certification**

未认证：针对神经网络认证的攻击 cs.LG

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2108.11299v2)

**Authors**: Tobias Lorenz, Marta Kwiatkowska, Mario Fritz

**Abstracts**: Certifiers for neural networks have made great progress towards provable robustness guarantees against evasion attacks using adversarial examples. However, introducing certifiers into deep learning systems also opens up new attack vectors, which need to be considered before deployment. In this work, we conduct the first systematic analysis of training-time attacks against certifiers in practical application pipelines, identifying new threat vectors that can be exploited to degrade the overall system. Using these insights, we design two backdoor attacks against network certifiers, which can drastically reduce certified robustness. For example, adding 1% poisoned data points during training is sufficient to reduce certified robustness by up to 95 percentage points, effectively rendering the certifier useless. We analyze how such novel attacks can compromise the overall system's integrity or availability. Our extensive experiments across multiple datasets, model architectures, and certifiers demonstrate the wide applicability of these attacks. A first investigation into potential defenses shows that current approaches are insufficient to mitigate the issue, highlighting the need for new, more specific solutions.

摘要: 神经网络的认证器已经取得了很大的进展，可以用敌意的例子来证明对规避攻击的鲁棒性保证。然而，将认证器引入深度学习系统也会打开新的攻击向量，在部署之前需要考虑这些攻击向量。在这项工作中，我们首次对实际应用管道中针对认证器的训练时间攻击进行了系统分析，识别出可以用来降低整个系统性能的新威胁向量。利用这些见解，我们设计了两种针对网络认证器的后门攻击，这两种攻击会极大地降低认证的健壮性。例如，在训练期间添加1%的有毒数据点就足以将认证的健壮性降低高达95个百分点，从而有效地使认证器无用。我们分析了这种新颖的攻击如何危害整个系统的完整性或可用性。我们在多个数据集、模型架构和认证器上的广泛实验证明了这些攻击的广泛适用性。对潜在防御措施的首次调查显示，目前的方法不足以缓解这一问题，这突显了需要新的、更具体的解决方案。



## **40. Searching for Robust Neural Architectures via Comprehensive and Reliable Evaluation**

通过综合可靠的评估寻找健壮的神经结构 cs.LG

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2203.03128v1)

**Authors**: Jialiang Sun, Tingsong Jiang, Chao Li, Weien Zhou, Xiaoya Zhang, Wen Yao, Xiaoqian Chen

**Abstracts**: Neural architecture search (NAS) could help search for robust network architectures, where defining robustness evaluation metrics is the important procedure. However, current robustness evaluations in NAS are not sufficiently comprehensive and reliable. In particular, the common practice only considers adversarial noise and quantified metrics such as the Jacobian matrix, whereas, some studies indicated that the models are also vulnerable to other types of noises such as natural noise. In addition, existing methods taking adversarial noise as the evaluation just use the robust accuracy of the FGSM or PGD, but these adversarial attacks could not provide the adequately reliable evaluation, leading to the vulnerability of the models under stronger attacks. To alleviate the above problems, we propose a novel framework, called Auto Adversarial Attack and Defense (AAAD), where we employ neural architecture search methods, and four types of robustness evaluations are considered, including adversarial noise, natural noise, system noise and quantified metrics, thereby assisting in finding more robust architectures. Also, among the adversarial noise, we use the composite adversarial attack obtained by random search as the new metric to evaluate the robustness of the model architectures. The empirical results on the CIFAR10 dataset show that the searched efficient attack could help find more robust architectures.

摘要: 神经体系结构搜索(NAS)可以帮助搜索健壮的网络结构，其中定义健壮性评估度量是重要的步骤。然而，目前NAS中的健壮性评估还不够全面和可靠。特别是，通常的做法只考虑对抗性噪声和量化的度量，如雅可比矩阵，而一些研究表明，模型也容易受到其他类型的噪声，如自然噪声的影响。另外，现有的以对抗性噪声为评价指标的方法仅仅使用了FGSM或PGD的鲁棒准确度，但这些对抗性攻击不能提供足够可靠的评价，导致模型在较强攻击下的脆弱性。为了缓解上述问题，我们提出了一种新的框架，称为自动对抗攻击和防御(AAAD)，其中我们使用了神经结构搜索方法，并考虑了四种类型的健壮性评估，包括对抗噪声、自然噪声、系统噪声和量化度量，从而帮助发现更健壮的体系结构。另外，在敌意噪声中，我们使用随机搜索得到的复合敌意攻击作为新的度量来评估模型体系的健壮性。在CIFAR10数据集上的实验结果表明，搜索到的高效攻击可以帮助发现更健壮的体系结构。



## **41. Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer**

保护面部隐私：通过风格稳健的化妆传输生成敌意身份面具 cs.CV

Accepted by CVPR2022, NOT the camera-ready version

**SubmitDate**: 2022-03-07    [paper-pdf](http://arxiv.org/pdf/2203.03121v1)

**Authors**: Shengshan Hu, Xiaogeng Liu, Yechao Zhang, Minghui Li, Leo Yu Zhang, Hai Jin, Libing Wu

**Abstracts**: While deep face recognition (FR) systems have shown amazing performance in identification and verification, they also arouse privacy concerns for their excessive surveillance on users, especially for public face images widely spread on social networks. Recently, some studies adopt adversarial examples to protect photos from being identified by unauthorized face recognition systems. However, existing methods of generating adversarial face images suffer from many limitations, such as awkward visual, white-box setting, weak transferability, making them difficult to be applied to protect face privacy in reality. In this paper, we propose adversarial makeup transfer GAN (AMT-GAN), a novel face protection method aiming at constructing adversarial face images that preserve stronger black-box transferability and better visual quality simultaneously. AMT-GAN leverages generative adversarial networks (GAN) to synthesize adversarial face images with makeup transferred from reference images. In particular, we introduce a new regularization module along with a joint training strategy to reconcile the conflicts between the adversarial noises and the cycle consistence loss in makeup transfer, achieving a desirable balance between the attack strength and visual changes. Extensive experiments verify that compared with state of the arts, AMT-GAN can not only preserve a comfortable visual quality, but also achieve a higher attack success rate over commercial FR APIs, including Face++, Aliyun, and Microsoft.

摘要: 深度人脸识别(FR)系统在身份识别和验证方面表现出惊人性能的同时，也因其对用户的过度监控而引起隐私问题，特别是对社交网络上广泛传播的公共人脸图像。近年来，一些研究采用对抗性例子来保护照片不被未经授权的人脸识别系统识别。然而，现有的生成敌意人脸图像的方法存在着视觉效果不佳、白盒设置、可移植性差等诸多局限性，难以应用于现实中的人脸隐私保护。本文提出了一种新的人脸保护方法--对抗性化妆转移GAN(AMT-GAN)，其目的是在构建对抗性人脸图像的同时保持较强的黑盒可传递性和较好的视觉质量。AMT-GAN利用生成性对抗性网络(GAN)来合成带有参考图像化妆的对抗性人脸图像。特别是，我们引入了一种新的正则化模型和联合训练策略来协调化妆转移中对抗性噪声和循环一致性损失之间的冲突，实现了攻击强度和视觉变化之间的理想平衡。大量实验证明，与现有技术相比，AMT-GAN不仅能保持舒适的视觉质量，而且比Face++、阿里云、微软等商用FR API具有更高的攻击成功率。



## **42. Can You Hear It? Backdoor Attacks via Ultrasonic Triggers**

你能听到吗？通过超声波触发器进行后门攻击 cs.CR

**SubmitDate**: 2022-03-06    [paper-pdf](http://arxiv.org/pdf/2107.14569v3)

**Authors**: Stefanos Koffas, Jing Xu, Mauro Conti, Stjepan Picek

**Abstracts**: This work explores backdoor attacks for automatic speech recognition systems where we inject inaudible triggers. By doing so, we make the backdoor attack challenging to detect for legitimate users, and thus, potentially more dangerous. We conduct experiments on two versions of a speech dataset and three neural networks and explore the performance of our attack concerning the duration, position, and type of the trigger. Our results indicate that less than 1% of poisoned data is sufficient to deploy a backdoor attack and reach a 100% attack success rate. We observed that short, non-continuous triggers result in highly successful attacks. However, since our trigger is inaudible, it can be as long as possible without raising any suspicions making the attack more effective. Finally, we conducted our attack in actual hardware and saw that an adversary could manipulate inference in an Android application by playing the inaudible trigger over the air.

摘要: 这项工作探索了自动语音识别系统的后门攻击，在这些系统中，我们注入了可听得见的触发器。通过这样做，我们使后门攻击对于合法用户来说更难检测，因此，潜在地更危险。我们在两个版本的语音数据集和三个神经网络上进行了实验，并探讨了我们的攻击在触发持续时间、位置和类型方面的性能。我们的结果表明，只有不到1%的有毒数据足以部署后门攻击，并且达到100%的攻击成功率。我们观察到短的、非连续的触发器会导致非常成功的攻击。然而，由于我们的扳机是听不见的，它可以尽可能长，而不会引起任何怀疑，从而使攻击更有效。最后，我们在实际的硬件上进行了攻击，看到对手可以通过空中播放听不见的触发器来操纵Android应用程序中的推理。



## **43. An Improved Genetic Algorithm and Its Application in Neural Network Adversarial Attack**

一种改进的遗传算法及其在神经网络攻击中的应用 cs.NE

18 pages, 9 figures, 9 tables and 23 References

**SubmitDate**: 2022-03-06    [paper-pdf](http://arxiv.org/pdf/2110.01818v5)

**Authors**: Dingming Yang, Zeyu Yu, Hongqiang Yuan, Yanrong Cui

**Abstracts**: The choice of crossover and mutation strategies plays a crucial role in the searchability, convergence efficiency and precision of genetic algorithms. In this paper, a novel improved genetic algorithm is proposed by improving the crossover and mutation operation of the simple genetic algorithm, and it is verified by 15 test functions. The qualitative results show that, compared with three other mainstream swarm intelligence optimization algorithms, the algorithm can not only improve the global search ability, convergence efficiency and precision, but also increase the success rate of convergence to the optimal value under the same experimental conditions. The quantitative results show that the algorithm performs superiorly in 13 of the 15 tested functions. The Wilcoxon rank-sum test was used for statistical evaluation, showing the significant advantage of the algorithm at $95\%$ confidence intervals. Finally, the algorithm is applied to neural network adversarial attacks. The applied results show that the method does not need the structure and parameter information inside the neural network model, and it can obtain the adversarial samples with high confidence in a brief time just by the classification and confidence information output from the neural network.

摘要: 交叉和变异策略的选择对遗传算法的可搜索性、收敛效率和精度起着至关重要的作用。通过对简单遗传算法交叉和变异操作的改进，提出了一种新的改进遗传算法，并通过15个测试函数进行了验证。定性结果表明，与其他三种主流群体智能优化算法相比，该算法在相同的实验条件下，不仅提高了全局搜索能力、收敛效率和精度，而且提高了收敛到最优值的成功率。定量结果表明，该算法在15个测试函数中有13个表现较好。采用Wilcoxon秩和检验进行统计评价，结果表明该算法在95美元置信区间内具有显著优势。最后，将该算法应用于神经网络对抗性攻击。应用结果表明，该方法不需要神经网络模型内部的结构和参数信息，仅根据神经网络输出的分类和置信度信息，即可在短时间内获得高置信度的对抗性样本。



## **44. Finding Dynamics Preserving Adversarial Winning Tickets**

寻找动态保存的对抗性中奖彩票 cs.LG

Accepted by AISTATS2022

**SubmitDate**: 2022-03-06    [paper-pdf](http://arxiv.org/pdf/2202.06488v3)

**Authors**: Xupeng Shi, Pengfei Zheng, A. Adam Ding, Yuan Gao, Weizhong Zhang

**Abstracts**: Modern deep neural networks (DNNs) are vulnerable to adversarial attacks and adversarial training has been shown to be a promising method for improving the adversarial robustness of DNNs. Pruning methods have been considered in adversarial context to reduce model capacity and improve adversarial robustness simultaneously in training. Existing adversarial pruning methods generally mimic the classical pruning methods for natural training, which follow the three-stage 'training-pruning-fine-tuning' pipelines. We observe that such pruning methods do not necessarily preserve the dynamics of dense networks, making it potentially hard to be fine-tuned to compensate the accuracy degradation in pruning. Based on recent works of \textit{Neural Tangent Kernel} (NTK), we systematically study the dynamics of adversarial training and prove the existence of trainable sparse sub-network at initialization which can be trained to be adversarial robust from scratch. This theoretically verifies the \textit{lottery ticket hypothesis} in adversarial context and we refer such sub-network structure as \textit{Adversarial Winning Ticket} (AWT). We also show empirical evidences that AWT preserves the dynamics of adversarial training and achieve equal performance as dense adversarial training.

摘要: 现代深层神经网络(DNNs)容易受到敌意攻击，对抗性训练已被证明是提高DNN对抗性鲁棒性的一种很有前途的方法。在训练过程中，考虑了对抗性环境下的剪枝方法，在减少模型容量的同时提高对抗性鲁棒性。现有的对抗性剪枝方法一般是模仿经典的自然训练剪枝方法，遵循“训练-剪枝-微调”三阶段的流水线。我们观察到，这样的剪枝方法并不一定保持密集网络的动态，使得它可能很难被微调来补偿剪枝过程中的精度下降。基于神经切核(NTK)的最新工作，系统地研究了对抗性训练的动力学，证明了在初始化时存在可训练的稀疏子网络，它可以从头开始训练为对抗性健壮性网络。这从理论上验证了对抗性环境下的\text{彩票假设}，我们将这种子网络结构称为\text{对抗性中票}(AWT)。我们还展示了经验证据，AWT保持了对抗性训练的动态性，并获得了与密集对抗性训练相同的性能。



## **45. aaeCAPTCHA: The Design and Implementation of Audio Adversarial CAPTCHA**

AaeCAPTCHA：音频对抗性验证码的设计与实现 cs.CR

Accepted at 7th IEEE European Symposium on Security and Privacy  (EuroS&P 2022)

**SubmitDate**: 2022-03-05    [paper-pdf](http://arxiv.org/pdf/2203.02735v1)

**Authors**: Md Imran Hossen, Xiali Hei

**Abstracts**: CAPTCHAs are designed to prevent malicious bot programs from abusing websites. Most online service providers deploy audio CAPTCHAs as an alternative to text and image CAPTCHAs for visually impaired users. However, prior research investigating the security of audio CAPTCHAs found them highly vulnerable to automated attacks using Automatic Speech Recognition (ASR) systems. To improve the robustness of audio CAPTCHAs against automated abuses, we present the design and implementation of an audio adversarial CAPTCHA (aaeCAPTCHA) system in this paper. The aaeCAPTCHA system exploits audio adversarial examples as CAPTCHAs to prevent the ASR systems from automatically solving them. Furthermore, we conducted a rigorous security evaluation of our new audio CAPTCHA design against five state-of-the-art DNN-based ASR systems and three commercial Speech-to-Text (STT) services. Our experimental evaluations demonstrate that aaeCAPTCHA is highly secure against these speech recognition technologies, even when the attacker has complete knowledge of the current attacks against audio adversarial examples. We also conducted a usability evaluation of the proof-of-concept implementation of the aaeCAPTCHA scheme. Our results show that it achieves high robustness at a moderate usability cost compared to normal audio CAPTCHAs. Finally, our extensive analysis highlights that aaeCAPTCHA can significantly enhance the security and robustness of traditional audio CAPTCHA systems while maintaining similar usability.

摘要: 验证码旨在防止恶意的僵尸程序滥用网站。大多数在线服务提供商为视障用户部署音频验证码作为文本和图像验证码的替代方案。然而，先前调查音频验证码安全性的研究发现，它们非常容易受到使用自动语音识别(ASR)系统的自动攻击。为了提高音频验证码对自动化滥用的健壮性，本文设计并实现了一个音频对抗性验证码系统(AaeCAPTCHA)。aaeCAPTCHA系统利用音频对抗性示例作为验证码来防止ASR系统自动求解它们。此外，我们针对五个基于DNN的最先进的ASR系统和三个商业语音到文本(STT)服务对我们的新音频验证码设计进行了严格的安全评估。我们的实验评估表明，即使攻击者完全了解当前针对音频对抗性示例的攻击，aaeCAPTCHA对这些语音识别技术也是高度安全的。我们还对aaeCAPTCHA方案的概念验证实现进行了可用性评估。实验结果表明，与普通的音频验证码相比，该算法以适中的可用性代价实现了较高的鲁棒性。最后，我们的广泛分析强调，aaeCAPTCHA可以显著增强传统音频验证码系统的安全性和健壮性，同时保持类似的可用性。



## **46. Generating Out of Distribution Adversarial Attack using Latent Space Poisoning**

利用潜在空间毒化产生分布外的敌意攻击 cs.CV

IEEE SPL 2021

**SubmitDate**: 2022-03-05    [paper-pdf](http://arxiv.org/pdf/2012.05027v2)

**Authors**: Ujjwal Upadhyay, Prerana Mukherjee

**Abstracts**: Traditional adversarial attacks rely upon the perturbations generated by gradients from the network which are generally safeguarded by gradient guided search to provide an adversarial counterpart to the network. In this paper, we propose a novel mechanism of generating adversarial examples where the actual image is not corrupted rather its latent space representation is utilized to tamper with the inherent structure of the image while maintaining the perceptual quality intact and to act as legitimate data samples. As opposed to gradient-based attacks, the latent space poisoning exploits the inclination of classifiers to model the independent and identical distribution of the training dataset and tricks it by producing out of distribution samples. We train a disentangled variational autoencoder (beta-VAE) to model the data in latent space and then we add noise perturbations using a class-conditioned distribution function to the latent space under the constraint that it is misclassified to the target label. Our empirical results on MNIST, SVHN, and CelebA dataset validate that the generated adversarial examples can easily fool robust l_0, l_2, l_inf norm classifiers designed using provably robust defense mechanisms.

摘要: 传统的对抗性攻击依赖于由来自网络的梯度产生的扰动，这些扰动通常由梯度引导搜索来保护，以提供网络的对应物。在本文中，我们提出了一种新的生成敌意示例的机制，其中实际图像没有被破坏，而是利用其潜在空间表示来篡改图像的内在结构，同时保持感知质量不变，并作为合法的数据样本。与基于梯度的攻击不同，潜在空间中毒利用分类器的倾向性来建模训练数据集的独立且相同的分布，并通过产生分布外的样本来欺骗它。我们训练一个解缠变分自动编码器(β-VAE)来对潜在空间中的数据建模，然后在误分类为目标标签的约束下，使用分类条件分布函数向潜在空间添加噪声扰动。我们在MNIST、SVHN和CelebA数据集上的实验结果验证了所生成的敌意示例可以很容易地欺骗使用可证明鲁棒防御机制设计的鲁棒l_0、l_2、l_inf范数分类器。



## **47. Adversarial samples for deep monocular 6D object pose estimation**

用于深部单目6维目标姿态估计的对抗性样本 cs.CV

15 pages

**SubmitDate**: 2022-03-05    [paper-pdf](http://arxiv.org/pdf/2203.00302v2)

**Authors**: Jinlai Zhang, Weiming Li, Shuang Liang, Hao Wang, Jihong Zhu

**Abstracts**: Estimating 6D object pose from an RGB image is important for many real-world applications such as autonomous driving and robotic grasping. Recent deep learning models have achieved significant progress on this task but their robustness received little research attention. In this work, for the first time, we study adversarial samples that can fool deep learning models with imperceptible perturbations to input image. In particular, we propose a Unified 6D pose estimation Attack, namely U6DA, which can successfully attack several state-of-the-art (SOTA) deep learning models for 6D pose estimation. The key idea of our U6DA is to fool the models to predict wrong results for object instance localization and shape that are essential for correct 6D pose estimation. Specifically, we explore a transfer-based black-box attack to 6D pose estimation. We design the U6DA loss to guide the generation of adversarial examples, the loss aims to shift the segmentation attention map away from its original position. We show that the generated adversarial samples are not only effective for direct 6D pose estimation models, but also are able to attack two-stage models regardless of their robust RANSAC modules. Extensive experiments were conducted to demonstrate the effectiveness, transferability, and anti-defense capability of our U6DA on large-scale public benchmarks. We also introduce a new U6DA-Linemod dataset for robustness study of the 6D pose estimation task. Our codes and dataset will be available at \url{https://github.com/cuge1995/U6DA}.

摘要: 从RGB图像估计6D物体姿态对于许多现实世界的应用非常重要，例如自动驾驶和机器人抓取。最近的深度学习模型在这方面已经取得了显着的进展，但是它们的鲁棒性却没有得到足够的研究。在这项工作中，我们首次研究了可以欺骗深度学习模型的对抗性样本，并对输入图像进行了潜移默化的扰动。特别地，我们提出了一种统一的6D位姿估计攻击，即U6DA，它可以成功地攻击几种用于6D位姿估计的SOTA深度学习模型。我们的U6DA的关键思想是愚弄模型来预测错误的对象实例定位和形状结果，这是正确的6D姿势估计所必需的。具体地说，我们探索了一种基于传输的黑盒攻击来进行6D位姿估计。我们设计了U6DA丢失来指导对抗性示例的生成，该丢失的目的是将分割注意图从原来的位置移开。结果表明，生成的对抗性样本不仅对直接6D姿态估计模型有效，而且无论其RANSAC模型是否具有鲁棒性，都能够攻击两阶段模型。在大型公共基准上进行了广泛的实验，以验证我们的U6DA的有效性、可转移性和反防御能力。我们还介绍了一个新的U6DA-Linemod数据集，用于6D位姿估计任务的鲁棒性研究。我们的代码和数据集将在\url{https://github.com/cuge1995/U6DA}.



## **48. Training privacy-preserving video analytics pipelines by suppressing features that reveal information about private attributes**

通过抑制泄露关于私有属性的信息的特征来训练保护隐私的视频分析管道 cs.CV

**SubmitDate**: 2022-03-05    [paper-pdf](http://arxiv.org/pdf/2203.02635v1)

**Authors**: Chau Yi Li, Andrea Cavallaro

**Abstracts**: Deep neural networks are increasingly deployed for scene analytics, including to evaluate the attention and reaction of people exposed to out-of-home advertisements. However, the features extracted by a deep neural network that was trained to predict a specific, consensual attribute (e.g. emotion) may also encode and thus reveal information about private, protected attributes (e.g. age or gender). In this work, we focus on such leakage of private information at inference time. We consider an adversary with access to the features extracted by the layers of a deployed neural network and use these features to predict private attributes. To prevent the success of such an attack, we modify the training of the network using a confusion loss that encourages the extraction of features that make it difficult for the adversary to accurately predict private attributes. We validate this training approach on image-based tasks using a publicly available dataset. Results show that, compared to the original network, the proposed PrivateNet can reduce the leakage of private information of a state-of-the-art emotion recognition classifier by 2.88% for gender and by 13.06% for age group, with a minimal effect on task accuracy.

摘要: 深度神经网络越来越多地用于场景分析，包括评估接触户外广告的人的注意力和反应。然而，由被训练来预测特定的、一致的属性(例如，情感)的深度神经网络提取的特征也可以编码并且因此揭示关于私人的、受保护的属性(例如，年龄或性别)的信息。在这项工作中，我们关注的是推理时隐私信息的泄露。我们考虑一个可以访问由部署的神经网络的各层提取的特征的对手，并使用这些特征来预测私有属性。为了防止此类攻击成功，我们使用念力损失修改了网络的训练，该损失鼓励提取使对手难以准确预测私人属性的特征。我们使用公开可用的数据集在基于图像的任务上验证了这种训练方法。实验结果表明，与原网络相比，本文提出的PrivateNet能够将最先进的情感识别分类器的隐私信息泄露减少2.88%(性别)和13.06%(年龄组)，并且对任务准确率的影响最小。



## **49. Optimal Clock Synchronization with Signatures**

带签名的最佳时钟同步 cs.DC

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2203.02553v1)

**Authors**: Christoph Lenzen, Julian Loss

**Abstracts**: Cryptographic signatures can be used to increase the resilience of distributed systems against adversarial attacks, by increasing the number of faulty parties that can be tolerated. While this is well-studied for consensus, it has been underexplored in the context of fault-tolerant clock synchronization, even in fully connected systems. Here, the honest parties of an $n$-node system are required to compute output clocks of small skew (i.e., maximum phase offset) despite local clock rates varying between $1$ and $\vartheta>1$, end-to-end communication delays varying between $d-u$ and $d$, and the interference from malicious parties. So far, it is only known that clock pulses of skew $d$ can be generated with (trivially optimal) resilience of $\lceil n/2\rceil-1$ (PODC `19), improving over the tight bound of $\lceil n/3\rceil-1$ holding without signatures for \emph{any} skew bound (STOC `84, PODC `85). Since typically $d\gg u$ and $\vartheta-1\ll 1$, this is far from the lower bound of $u+(\vartheta-1)d$ that applies even in the fault-free case (IPL `01).   We prove matching upper and lower bounds of $\Theta(u+(\vartheta-1)d)$ on the skew for the resilience range from $\lceil n/3\rceil$ to $\lceil n/2\rceil-1$. The algorithm showing the upper bound is, under the assumption that the adversary cannot forge signatures, deterministic. The lower bound holds even if clocks are initially perfectly synchronized, message delays between honest nodes are known, $\vartheta$ is arbitrarily close to one, and the synchronization algorithm is randomized. This has crucial implications for network designers that seek to leverage signatures for providing more robust time. In contrast to the setting without signatures, they must ensure that an attacker cannot easily bypass the lower bound on the delay on links with a faulty endpoint.

摘要: 通过增加可容忍的故障方的数量，可以使用密码签名来提高分布式系统对敌意攻击的恢复能力。虽然这是为了达成共识而研究得很好的，但在容错时钟同步的背景下，甚至在完全连接的系统中，这一点也没有得到充分的探索。这里，尽管本地时钟速率在$1$和$\vartheta>1$之间变化，端到端通信延迟在$d-u$和$d$之间变化，以及来自恶意方的干扰，但$n$节点系统的诚实方被要求计算小偏差(即最大相位偏移)的输出时钟。到目前为止，只知道歪斜$d$的时钟脉冲可以产生$\lceil n/2\rceil$(PODC`19)的弹性(最优)，改进了没有签名的$\lceil n/3\rceil$保持的紧凑界限(STEC`84，PODC‘85)。由于通常为$d\gg u$和$\vartheta-1\ll 1$，即使在无故障的情况下(IPL‘01)，这也远不是适用于$u+(\vartheta-1)d$的下限。我们证明了$theta(u+(vartheta-1)d)$在$lceil n/3\rceil$到$lceil n/2\rceil-1$的斜斜度上的上下界是匹配的。在假设对手不能伪造签名的情况下，给出上限的算法是确定性的。即使时钟最初是完全同步的，诚实节点之间的消息延迟是已知的，$\vartheta$任意接近于1，并且同步算法是随机的，这个下限也是成立的。这对于寻求利用签名来提供更可靠时间的网络设计人员具有重要意义。与没有签名的设置相比，它们必须确保攻击者不能轻松绕过具有故障端点的链路的延迟下限。



## **50. Medical Aegis: Robust adversarial protectors for medical images**

医学宙斯盾：强大的医学图像对抗性保护者 cs.CV

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2111.10969v4)

**Authors**: Qingsong Yao, Zecheng He, S. Kevin Zhou

**Abstracts**: Deep neural network based medical image systems are vulnerable to adversarial examples. Many defense mechanisms have been proposed in the literature, however, the existing defenses assume a passive attacker who knows little about the defense system and does not change the attack strategy according to the defense. Recent works have shown that a strong adaptive attack, where an attacker is assumed to have full knowledge about the defense system, can easily bypass the existing defenses. In this paper, we propose a novel adversarial example defense system called Medical Aegis. To the best of our knowledge, Medical Aegis is the first defense in the literature that successfully addresses the strong adaptive adversarial example attacks to medical images. Medical Aegis boasts two-tier protectors: The first tier of Cushion weakens the adversarial manipulation capability of an attack by removing its high-frequency components, yet posing a minimal effect on classification performance of the original image; the second tier of Shield learns a set of per-class DNN models to predict the logits of the protected model. Deviation from the Shield's prediction indicates adversarial examples. Shield is inspired by the observations in our stress tests that there exist robust trails in the shallow layers of a DNN model, which the adaptive attacks can hardly destruct. Experimental results show that the proposed defense accurately detects adaptive attacks, with negligible overhead for model inference.

摘要: 基于深度神经网络的医学图像系统容易受到敌意例子的攻击。文献中提出了很多防御机制，但现有的防御机制都假设攻击者是被动的，对防御系统知之甚少，不会根据防御情况改变攻击策略。最近的研究表明，强自适应攻击(假设攻击者完全了解防御系统)可以很容易地绕过现有的防御系统。在本文中，我们提出了一种新的对抗性实例防御系统--医学宙斯盾(Medical Aegis)。据我们所知，医学宙斯盾是文献中第一个成功解决了对医学图像的强适应性敌意攻击的防御方案。医学宙斯盾拥有两层保护器：第一层缓冲通过移除攻击的高频分量来削弱攻击的敌意操纵能力，但对原始图像的分类性能影响最小；第二层Shield学习一组按类的DNN模型来预测受保护模型的逻辑。与盾牌的预测背道而驰表明了敌对的例子。Shield的灵感来自于我们在压力测试中观察到的DNN模型的浅层存在健壮的踪迹，而自适应攻击很难破坏这些踪迹。实验结果表明，该防御方法能够准确检测出自适应攻击，而模型推理开销可以忽略不计。



