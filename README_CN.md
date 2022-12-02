# Latest Adversarial Attack Papers
**update at 2022-12-02 19:32:26**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Purifier: Defending Data Inference Attacks via Transforming Confidence Scores**

净化器：通过变换置信度分数防御数据推理攻击 cs.LG

accepted by AAAI 2023

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2212.00612v1) [paper-pdf](http://arxiv.org/pdf/2212.00612v1)

**Authors**: Ziqi Yang, Lijin Wang, Da Yang, Jie Wan, Ziming Zhao, Ee-Chien Chang, Fan Zhang, Kui Ren

**Abstract**: Neural networks are susceptible to data inference attacks such as the membership inference attack, the adversarial model inversion attack and the attribute inference attack, where the attacker could infer useful information such as the membership, the reconstruction or the sensitive attributes of a data sample from the confidence scores predicted by the target classifier. In this paper, we propose a method, namely PURIFIER, to defend against membership inference attacks. It transforms the confidence score vectors predicted by the target classifier and makes purified confidence scores indistinguishable in individual shape, statistical distribution and prediction label between members and non-members. The experimental results show that PURIFIER helps defend membership inference attacks with high effectiveness and efficiency, outperforming previous defense methods, and also incurs negligible utility loss. Besides, our further experiments show that PURIFIER is also effective in defending adversarial model inversion attacks and attribute inference attacks. For example, the inversion error is raised about 4+ times on the Facescrub530 classifier, and the attribute inference accuracy drops significantly when PURIFIER is deployed in our experiment.

摘要: 神经网络容易受到数据推理攻击，如隶属度推理攻击、对抗性模型反转攻击和属性推理攻击，攻击者可以从目标分类器预测的置信度分数推断出数据样本的隶属度、重构或敏感属性等有用信息。在本文中，我们提出了一种防御成员关系推理攻击的方法，即净化器。它对目标分类器预测的置信度向量进行变换，使得纯化后的置信度在个体形状、统计分布和成员与非成员之间的预测标签上无法区分。实验结果表明，该算法能够有效、高效地防御成员关系推理攻击，性能优于以往的防御方法，而效用损失可以忽略不计。此外，我们进一步的实验表明，净化器在防御对抗性模型反转攻击和属性推理攻击方面也是有效的。例如，在FacescRub530分类器上，倒置错误提高了大约4倍以上，当我们的实验中部署了净化器时，属性推理的准确率显著下降。



## **2. PointCA: Evaluating the Robustness of 3D Point Cloud Completion Models Against Adversarial Examples**

PointCA：评估3D点云补全模型对敌方示例的稳健性 cs.CV

Accepted by the 37th AAAI Conference on Artificial Intelligence  (AAAI-23)

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2211.12294v2) [paper-pdf](http://arxiv.org/pdf/2211.12294v2)

**Authors**: Shengshan Hu, Junwei Zhang, Wei Liu, Junhui Hou, Minghui Li, Leo Yu Zhang, Hai Jin, Lichao Sun

**Abstract**: Point cloud completion, as the upstream procedure of 3D recognition and segmentation, has become an essential part of many tasks such as navigation and scene understanding. While various point cloud completion models have demonstrated their powerful capabilities, their robustness against adversarial attacks, which have been proven to be fatally malicious towards deep neural networks, remains unknown. In addition, existing attack approaches towards point cloud classifiers cannot be applied to the completion models due to different output forms and attack purposes. In order to evaluate the robustness of the completion models, we propose PointCA, the first adversarial attack against 3D point cloud completion models. PointCA can generate adversarial point clouds that maintain high similarity with the original ones, while being completed as another object with totally different semantic information. Specifically, we minimize the representation discrepancy between the adversarial example and the target point set to jointly explore the adversarial point clouds in the geometry space and the feature space. Furthermore, to launch a stealthier attack, we innovatively employ the neighbourhood density information to tailor the perturbation constraint, leading to geometry-aware and distribution-adaptive modifications for each point. Extensive experiments against different premier point cloud completion networks show that PointCA can cause a performance degradation from 77.9% to 16.7%, with the structure chamfer distance kept below 0.01. We conclude that existing completion models are severely vulnerable to adversarial examples, and state-of-the-art defenses for point cloud classification will be partially invalid when applied to incomplete and uneven point cloud data.

摘要: 点云补全作为三维识别和分割的上游步骤，已经成为导航、场景理解等许多任务的重要组成部分。虽然各种点云补全模型已经展示了它们强大的能力，但它们对对手攻击的健壮性仍然未知，这些攻击已被证明是对深度神经网络的致命恶意攻击。此外，由于不同的输出形式和攻击目的，现有的针对点云分类器的攻击方法不能应用于完成模型。为了评估补全模型的健壮性，我们提出了针对三维点云补全模型的第一次对抗性攻击PointCA。PointCA可以生成与原始点云保持高度相似的对抗性点云，同时作为另一个对象完成，具有完全不同的语义信息。具体地说，我们最小化对抗性样本和目标点集之间的表示差异，共同探索几何空间和特征空间中的对抗性点云。此外，为了发动更隐蔽的攻击，我们创新性地使用邻域密度信息来定制扰动约束，导致对每个点的几何感知和分布自适应修改。在不同初始点云完成网络上的大量实验表明，在结构倒角距离保持在0.01以下的情况下，PointCA可以导致性能从77.9%下降到16.7%。我们得出的结论是，现有的完备化模型非常容易受到对手例子的攻击，并且最新的点云分类方法在应用于不完整和不均匀的点云数据时将部分无效。



## **3. Defending Black-box Skeleton-based Human Activity Classifiers**

基于黑盒骨架防御的人类活动分类器 cs.CV

Accepted in AAAI 2023

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2203.04713v5) [paper-pdf](http://arxiv.org/pdf/2203.04713v5)

**Authors**: He Wang, Yunfeng Diao, Zichang Tan, Guodong Guo

**Abstract**: Skeletal motions have been heavily replied upon for human activity recognition (HAR). Recently, a universal vulnerability of skeleton-based HAR has been identified across a variety of classifiers and data, calling for mitigation. To this end, we propose the first black-box defense method for skeleton-based HAR to our best knowledge. Our method is featured by full Bayesian treatments of the clean data, the adversaries and the classifier, leading to (1) a new Bayesian Energy-based formulation of robust discriminative classifiers, (2) a new adversary sampling scheme based on natural motion manifolds, and (3) a new post-train Bayesian strategy for black-box defense. We name our framework Bayesian Energy-based Adversarial Training or BEAT. BEAT is straightforward but elegant, which turns vulnerable black-box classifiers into robust ones without sacrificing accuracy. It demonstrates surprising and universal effectiveness across a wide range of skeletal HAR classifiers and datasets, under various attacks. Code is available at https://github.com/realcrane/RobustActionRecogniser.

摘要: 骨骼运动在人类活动识别(HAR)中得到了广泛的应用。最近，基于骨架的HAR在各种分类器和数据中发现了一个普遍的漏洞，需要缓解。为此，我们提出了第一种基于骨架的HAR黑盒防御方法。我们的方法的特点是对干净数据、对手和分类器进行全面的贝叶斯处理，导致(1)新的基于贝叶斯能量的稳健判别分类器的形成，(2)基于自然运动流形的新的对手采样方案，(3)新的训练后贝叶斯策略用于黑盒防御。我们将我们的框架命名为基于贝叶斯能量的对抗性训练或BEAT。BEAT是简单但优雅的，它将脆弱的黑匣子分类器变成了健壮的分类器，而不会牺牲准确性。在各种攻击下，它在广泛的骨架HAR分类器和数据集上展示了令人惊讶的和普遍的有效性。代码可在https://github.com/realcrane/RobustActionRecogniser.上找到



## **4. All You Need Is Hashing: Defending Against Data Reconstruction Attack in Vertical Federated Learning**

您所需要的就是散列：在垂直联合学习中防御数据重建攻击 cs.CR

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2212.00325v1) [paper-pdf](http://arxiv.org/pdf/2212.00325v1)

**Authors**: Pengyu Qiu, Xuhong Zhang, Shouling Ji, Yuwen Pu, Ting Wang

**Abstract**: Vertical federated learning is a trending solution for multi-party collaboration in training machine learning models. Industrial frameworks adopt secure multi-party computation methods such as homomorphic encryption to guarantee data security and privacy. However, a line of work has revealed that there are still leakage risks in VFL. The leakage is caused by the correlation between the intermediate representations and the raw data. Due to the powerful approximation ability of deep neural networks, an adversary can capture the correlation precisely and reconstruct the data. To deal with the threat of the data reconstruction attack, we propose a hashing-based VFL framework, called \textit{HashVFL}, to cut off the reversibility directly. The one-way nature of hashing allows our framework to block all attempts to recover data from hash codes. However, integrating hashing also brings some challenges, e.g., the loss of information. This paper proposes and addresses three challenges to integrating hashing: learnability, bit balance, and consistency. Experimental results demonstrate \textit{HashVFL}'s efficiency in keeping the main task's performance and defending against data reconstruction attacks. Furthermore, we also analyze its potential value in detecting abnormal inputs. In addition, we conduct extensive experiments to prove \textit{HashVFL}'s generalization in various settings. In summary, \textit{HashVFL} provides a new perspective on protecting multi-party's data security and privacy in VFL. We hope our study can attract more researchers to expand the application domains of \textit{HashVFL}.

摘要: 垂直联合学习是训练机器学习模型中多方协作的一种趋势解决方案。产业框架采用同态加密等安全多方计算方法，保障数据安全和隐私。然而，有一项工作透露，VFL仍存在泄漏风险。泄漏是由中间表示法和原始数据之间的相关性造成的。由于深度神经网络的强大逼近能力，对手可以准确地捕获相关性并重建数据。为了应对数据重构攻击的威胁，我们提出了一种基于哈希的VFL框架，称为\textit{HashVFL}，直接切断可逆性。哈希的单向特性允许我们的框架阻止所有从哈希码恢复数据的尝试。然而，整合散列也带来了一些挑战，例如信息的丢失。本文提出并解决了整合散列的三个挑战：可学习性、位平衡和一致性。实验结果证明了该算法在保持主任务性能和抵御数据重构攻击方面的有效性，并分析了其在异常输入检测方面的潜在价值。此外，我们还通过大量的实验证明了该算法在不同环境下的泛化能力。综上所述，该算法为保护VFL中多方的数据安全和隐私提供了一个新的视角。我们希望我们的研究能够吸引更多的研究人员来拓展该算法的应用领域。



## **5. Overcoming the Convex Relaxation Barrier for Neural Network Verification via Nonconvex Low-Rank Semidefinite Relaxations**

用非凸低阶半定松弛克服神经网络验证的凸松弛障碍 cs.LG

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2211.17244v1) [paper-pdf](http://arxiv.org/pdf/2211.17244v1)

**Authors**: Hong-Ming Chiu, Richard Y. Zhang

**Abstract**: To rigorously certify the robustness of neural networks to adversarial perturbations, most state-of-the-art techniques rely on a triangle-shaped linear programming (LP) relaxation of the ReLU activation. While the LP relaxation is exact for a single neuron, recent results suggest that it faces an inherent "convex relaxation barrier" as additional activations are added, and as the attack budget is increased. In this paper, we propose a nonconvex relaxation for the ReLU relaxation, based on a low-rank restriction of a semidefinite programming (SDP) relaxation. We show that the nonconvex relaxation has a similar complexity to the LP relaxation, but enjoys improved tightness that is comparable to the much more expensive SDP relaxation. Despite nonconvexity, we prove that the verification problem satisfies constraint qualification, and therefore a Riemannian staircase approach is guaranteed to compute a near-globally optimal solution in polynomial time. Our experiments provide evidence that our nonconvex relaxation almost completely overcome the "convex relaxation barrier" faced by the LP relaxation.

摘要: 为了严格证明神经网络对对抗性扰动的稳健性，大多数最先进的技术依赖于REU激活的三角形线性规划(LP)松弛。虽然LP松弛对于单个神经元来说是精确的，但最近的结果表明，随着额外的激活增加和攻击预算的增加，它面临着固有的“凸松弛障碍”。本文基于半定规划(SDP)松弛的低阶限制，提出了RELU松弛的非凸松弛。我们证明了非凸松弛具有与LP松弛相似的复杂性，但具有与更昂贵的SDP松弛相当的紧性。尽管非凸性，我们证明了验证问题满足约束限定，从而保证了黎曼阶梯方法在多项式时间内计算出近全局最优解。我们的实验证明，我们的非凸松弛几乎完全克服了LP松弛所面临的“凸松弛障碍”。



## **6. Differentially Private ADMM-Based Distributed Discrete Optimal Transport for Resource Allocation**

基于差分私有ADMM的分布式离散资源优化传输 cs.SI

6 pages, 4 images, 1 algorithm, IEEE GLOBECOMM 2022

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2211.17070v1) [paper-pdf](http://arxiv.org/pdf/2211.17070v1)

**Authors**: Jason Hughes, Juntao Chen

**Abstract**: Optimal transport (OT) is a framework that can guide the design of efficient resource allocation strategies in a network of multiple sources and targets. To ease the computational complexity of large-scale transport design, we first develop a distributed algorithm based on the alternating direction method of multipliers (ADMM). However, such a distributed algorithm is vulnerable to sensitive information leakage when an attacker intercepts the transport decisions communicated between nodes during the distributed ADMM updates. To this end, we propose a privacy-preserving distributed mechanism based on output variable perturbation by adding appropriate randomness to each node's decision before it is shared with other corresponding nodes at each update instance. We show that the developed scheme is differentially private, which prevents the adversary from inferring the node's confidential information even knowing the transport decisions. Finally, we corroborate the effectiveness of the devised algorithm through case studies.

摘要: 最优传输(OT)是一个框架，可以指导在多个源和目标的网络中设计有效的资源分配策略。为了降低大规模运输设计的计算复杂性，我们首先提出了一种基于交替方向乘子法的分布式算法。然而，当攻击者在分布式ADMM更新期间截获节点之间通信的传输决策时，这样的分布式算法容易受到敏感信息的泄漏。为此，我们提出了一种基于输出变量扰动的隐私保护分布式机制，在每个更新时刻与其他对应节点共享之前，为每个节点的决策添加适当的随机性。我们证明了所提出的方案是差分私密性的，即使知道传输决策，也可以防止对手推断节点的机密信息。最后，通过案例分析验证了所设计算法的有效性。



## **7. A Systematic Evaluation of Node Embedding Robustness**

节点嵌入健壮性的系统评估 cs.LG

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2209.08064v3) [paper-pdf](http://arxiv.org/pdf/2209.08064v3)

**Authors**: Alexandru Mara, Jefrey Lijffijt, Stephan Günnemann, Tijl De Bie

**Abstract**: Node embedding methods map network nodes to low dimensional vectors that can be subsequently used in a variety of downstream prediction tasks. The popularity of these methods has grown significantly in recent years, yet, their robustness to perturbations of the input data is still poorly understood. In this paper, we assess the empirical robustness of node embedding models to random and adversarial poisoning attacks. Our systematic evaluation covers representative embedding methods based on Skip-Gram, matrix factorization, and deep neural networks. We compare edge addition, deletion and rewiring attacks computed using network properties as well as node labels. We also investigate the performance of popular node classification attack baselines that assume full knowledge of the node labels. We report qualitative results via embedding visualization and quantitative results in terms of downstream node classification and network reconstruction performances. We find that node classification results are impacted more than network reconstruction ones, that degree-based and label-based attacks are on average the most damaging and that label heterophily can strongly influence attack performance.

摘要: 节点嵌入方法将网络节点映射到可随后用于各种下行预测任务的低维向量。近年来，这些方法的普及率显著提高，然而，人们对它们对输入数据扰动的稳健性仍然知之甚少。在本文中，我们评估了节点嵌入模型对随机和对抗性中毒攻击的经验稳健性。我们的系统评价涵盖了基于Skip-Gram的典型嵌入方法、矩阵分解和深度神经网络。我们比较了使用网络属性和节点标签计算的边添加、删除和重新布线攻击。我们还研究了假设完全知道节点标签的流行的节点分类攻击基线的性能。我们通过嵌入可视化和定量结果来报告下游节点分类和网络重构性能方面的定性结果。我们发现，节点分类结果受到的影响比网络重构的影响更大，基于度和基于标签的攻击平均破坏性最大，标签异质性对攻击性能有很强的影响。



## **8. Adaptive adversarial training method for improving multi-scale GAN based on generalization bound theory**

基于泛化界理论的改进多尺度GAN的自适应对抗性训练方法 cs.CV

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2211.16791v1) [paper-pdf](http://arxiv.org/pdf/2211.16791v1)

**Authors**: Jing Tang, Bo Tao, Zeyu Gong, Zhouping Yin

**Abstract**: In recent years, multi-scale generative adversarial networks (GANs) have been proposed to build generalized image processing models based on single sample. Constraining on the sample size, multi-scale GANs have much difficulty converging to the global optimum, which ultimately leads to limitations in their capabilities. In this paper, we pioneered the introduction of PAC-Bayes generalized bound theory into the training analysis of specific models under different adversarial training methods, which can obtain a non-vacuous upper bound on the generalization error for the specified multi-scale GAN structure. Based on the drastic changes we found of the generalization error bound under different adversarial attacks and different training states, we proposed an adaptive training method which can greatly improve the image manipulation ability of multi-scale GANs. The final experimental results show that our adaptive training method in this paper has greatly contributed to the improvement of the quality of the images generated by multi-scale GANs on several image manipulation tasks. In particular, for the image super-resolution restoration task, the multi-scale GAN model trained by the proposed method achieves a 100% reduction in natural image quality evaluator (NIQE) and a 60% reduction in root mean squared error (RMSE), which is better than many models trained on large-scale datasets.

摘要: 近年来，多尺度生成对抗网络(GANS)被提出用来建立基于单样本的广义图像处理模型。多尺度遗传算法受样本量的限制，很难收敛到全局最优解，这最终导致它们的能力受到限制。在本文中，我们首次将PAC-Bayes广义界理论引入到不同对抗性训练方法下特定模型的训练分析中，可以得到特定多尺度GaN结构的泛化误差的非空上界。基于不同敌方攻击和不同训练状态下泛化误差界的剧烈变化，提出了一种自适应训练方法，可以极大地提高多尺度遗传算法的图像处理能力。最后的实验结果表明，本文提出的自适应训练方法对多尺度GANS算法在多个图像处理任务中生成的图像质量有很大的改善作用。特别是，对于图像超分辨率恢复任务，由该方法训练的多尺度GAN模型在自然图像质量评价(NIQE)和均方根误差(RMSE)方面取得了100%的降低，优于在大规模数据集上训练的许多模型。



## **9. GARNET: Reduced-Rank Topology Learning for Robust and Scalable Graph Neural Networks**

Garnet：强健可扩展图神经网络的降阶拓扑学习 cs.LG

Published as a conference paper at LoG 2020

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2201.12741v3) [paper-pdf](http://arxiv.org/pdf/2201.12741v3)

**Authors**: Chenhui Deng, Xiuyu Li, Zhuo Feng, Zhiru Zhang

**Abstract**: Graph neural networks (GNNs) have been increasingly deployed in various applications that involve learning on non-Euclidean data. However, recent studies show that GNNs are vulnerable to graph adversarial attacks. Although there are several defense methods to improve GNN robustness by eliminating adversarial components, they may also impair the underlying clean graph structure that contributes to GNN training. In addition, few of those defense models can scale to large graphs due to their high computational complexity and memory usage. In this paper, we propose GARNET, a scalable spectral method to boost the adversarial robustness of GNN models. GARNET first leverages weighted spectral embedding to construct a base graph, which is not only resistant to adversarial attacks but also contains critical (clean) graph structure for GNN training. Next, GARNET further refines the base graph by pruning additional uncritical edges based on probabilistic graphical model. GARNET has been evaluated on various datasets, including a large graph with millions of nodes. Our extensive experiment results show that GARNET achieves adversarial accuracy improvement and runtime speedup over state-of-the-art GNN (defense) models by up to 13.27% and 14.7x, respectively.

摘要: 图形神经网络(GNN)已被越来越多地应用于涉及非欧几里德数据学习的各种应用中。然而，最近的研究表明，GNN容易受到图的对抗性攻击。虽然有几种防御方法可以通过消除敌对组件来提高GNN的健壮性，但它们也可能损害有助于GNN训练的底层干净的图形结构。此外，这些防御模型中很少有能够扩展到大型图形的，因为它们的计算复杂性和内存使用量很高。在本文中，我们提出了Garnet，一种可伸缩的谱方法来提高GNN模型的对抗健壮性。Garnet First利用加权谱嵌入来构造基图，该基图不仅能抵抗敌方攻击，而且还包含GNN训练所需的关键(干净)图结构。接下来，Garnet基于概率图模型，通过剪枝额外的非关键边来进一步精化基图。石榴石已经在各种数据集上进行了评估，包括一个包含数百万个节点的大型图表。我们的大量实验结果表明，与现有的GNN(防御)模型相比，Garnet的对抗准确率提高了13.27%，运行时加速比提高了14.7倍。



## **10. Sludge for Good: Slowing and Imposing Costs on Cyber Attackers**

泥浆向善：减缓网络攻击者的速度并增加其成本 cs.CR

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16626v1) [paper-pdf](http://arxiv.org/pdf/2211.16626v1)

**Authors**: Josiah Dykstra, Kelly Shortridge, Jamie Met, Douglas Hough

**Abstract**: Choice architecture describes the design by which choices are presented to people. Nudges are an aspect intended to make "good" outcomes easy, such as using password meters to encourage strong passwords. Sludge, on the contrary, is friction that raises the transaction cost and is often seen as a negative to users. Turning this concept around, we propose applying sludge for positive cybersecurity outcomes by using it offensively to consume attackers' time and other resources.   To date, most cyber defenses have been designed to be optimally strong and effective and prohibit or eliminate attackers as quickly as possible. Our complimentary approach is to also deploy defenses that seek to maximize the consumption of the attackers' time and other resources while causing as little damage as possible to the victim. This is consistent with zero trust and similar mindsets which assume breach. The Sludge Strategy introduces cost-imposing cyber defense by strategically deploying friction for attackers before, during, and after an attack using deception and authentic design features. We present the characteristics of effective sludge, and show a continuum from light to heavy sludge. We describe the quantitative and qualitative costs to attackers and offer practical considerations for deploying sludge in practice. Finally, we examine real-world examples of U.S. government operations to frustrate and impose cost on cyber adversaries.

摘要: 选择体系结构描述了将选择呈现给人们的设计。轻推是一个旨在使“好”结果变得容易的方面，例如使用密码计量器来鼓励强密码。相反，淤泥是增加交易成本的摩擦，通常被视为对用户不利。扭转这一概念，我们建议将污泥应用于积极的网络安全结果，通过攻击性地使用它来消耗攻击者的时间和其他资源。到目前为止，大多数网络防御都被设计成最强大和有效的，并尽可能快地禁止或消除攻击者。我们的互补方法是部署防御措施，寻求最大限度地消耗攻击者的时间和其他资源，同时尽可能减少对受害者的损害。这与零信任和假设违反的类似心态是一致的。通过使用欺骗和可信的设计功能，在攻击之前、期间和之后战略性地为攻击者部署摩擦，Send Strategy引入了成本高昂的网络防御。我们介绍了有效污泥的特性，并显示了从轻到重的连续统。我们描述了攻击者的定量和定性成本，并为在实践中部署污泥提供了实际考虑。最后，我们考察了美国政府挫败网络对手并将代价强加于他们的真实世界的例子。



## **11. Synthesizing Attack-Aware Control and Active Sensing Strategies under Reactive Sensor Attacks**

反应式传感器攻击下的攻击感知控制与主动感知综合策略 math.OC

7 pages, 3 figure, 1 table, 1 algorithm

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2204.01584v2) [paper-pdf](http://arxiv.org/pdf/2204.01584v2)

**Authors**: Sumukha Udupa, Abhishek N. Kulkarni, Shuo Han, Nandi O. Leslie, Charles A. Kamhoua, Jie Fu

**Abstract**: We consider the probabilistic planning problem for a defender (P1) who can jointly query the sensors and take control actions to reach a set of goal states while being aware of possible sensor attacks by an adversary (P2) who has perfect observations. To synthesize a provably-correct, attack-aware joint control and active sensing strategy for P1, we construct a stochastic game on graph with augmented states that include the actual game state (known only to the attacker), the belief of the defender about the game state (constructed by the attacker based on his knowledge of defender's observations). We present an algorithm to compute a belief-based, randomized strategy for P1 to ensure satisfying the reachability objective with probability one, under the worst-case sensor attack carried out by an informed P2. We prove the correctness of the algorithm and illustrate using an example.

摘要: 我们考虑了防御者(P1)的概率规划问题，该防御者可以联合查询传感器并采取控制行动以达到一组目标状态，同时知道具有完美观察的对手(P2)可能的传感器攻击。为了综合P1的可证明正确的、攻击感知的联合控制和主动感知策略，我们在图上构造了一个带有增广状态的随机游戏，其中包括实际的游戏状态(只有攻击者知道)、防御者对游戏状态的信念(由攻击者根据他对防御者观察的知识构建的)。我们提出了一种算法来计算基于信任的随机策略，以确保在由知情的P2执行的最坏情况的传感器攻击下，以概率1满足可达性目标。证明了该算法的正确性，并用实例进行了说明。



## **12. Improving Adversarial Robustness with Self-Paced Hard-Class Pair Reweighting**

自定步长硬类对重加权提高对手健壮性 cs.CV

AAAI-23

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2210.15068v2) [paper-pdf](http://arxiv.org/pdf/2210.15068v2)

**Authors**: Pengyue Hou, Jie Han, Xingyu Li

**Abstract**: Deep Neural Networks are vulnerable to adversarial attacks. Among many defense strategies, adversarial training with untargeted attacks is one of the most effective methods. Theoretically, adversarial perturbation in untargeted attacks can be added along arbitrary directions and the predicted labels of untargeted attacks should be unpredictable. However, we find that the naturally imbalanced inter-class semantic similarity makes those hard-class pairs become virtual targets of each other. This study investigates the impact of such closely-coupled classes on adversarial attacks and develops a self-paced reweighting strategy in adversarial training accordingly. Specifically, we propose to upweight hard-class pair losses in model optimization, which prompts learning discriminative features from hard classes. We further incorporate a term to quantify hard-class pair consistency in adversarial training, which greatly boosts model robustness. Extensive experiments show that the proposed adversarial training method achieves superior robustness performance over state-of-the-art defenses against a wide range of adversarial attacks.

摘要: 深度神经网络很容易受到敌意攻击。在众多的防守策略中，非靶向攻击的对抗性训练是最有效的方法之一。从理论上讲，非目标攻击中的对抗性扰动可以沿任意方向添加，并且非目标攻击的预测标签应该是不可预测的。然而，我们发现，自然不平衡的类间语义相似度使得这些硬类对成为彼此的虚拟目标。本研究调查了这种紧密耦合的课程对对抗性攻击的影响，并相应地在对抗性训练中开发了一种自定步调重权重策略。具体地说，我们建议在模型优化中增加硬类对损失的权重，从而促进从硬类中学习区分特征。在对抗性训练中，我们进一步引入了一个术语来量化硬类对一致性，这大大提高了模型的稳健性。大量的实验表明，所提出的对抗性训练方法在对抗大范围的对抗性攻击时获得了比最先进的防御方法更好的健壮性性能。



## **13. Resilient Risk based Adaptive Authentication and Authorization (RAD-AA) Framework**

基于弹性风险的自适应身份验证和授权(RAD-AA)框架 cs.CR

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2208.02592v3) [paper-pdf](http://arxiv.org/pdf/2208.02592v3)

**Authors**: Jaimandeep Singh, Chintan Patel, Naveen Kumar Chaudhary

**Abstract**: In recent cyber attacks, credential theft has emerged as one of the primary vectors of gaining entry into the system. Once attacker(s) have a foothold in the system, they use various techniques including token manipulation to elevate the privileges and access protected resources. This makes authentication and token based authorization a critical component for a secure and resilient cyber system. In this paper we discuss the design considerations for such a secure and resilient authentication and authorization framework capable of self-adapting based on the risk scores and trust profiles. We compare this design with the existing standards such as OAuth 2.0, OpenID Connect and SAML 2.0. We then study popular threat models such as STRIDE and PASTA and summarize the resilience of the proposed architecture against common and relevant threat vectors. We call this framework as Resilient Risk based Adaptive Authentication and Authorization (RAD-AA). The proposed framework excessively increases the cost for an adversary to launch and sustain any cyber attack and provides much-needed strength to critical infrastructure. We also discuss the machine learning (ML) approach for the adaptive engine to accurately classify transactions and arrive at risk scores.

摘要: 在最近的网络攻击中，凭据盗窃已成为进入系统的主要载体之一。一旦攻击者在系统中站稳脚跟，他们就会使用包括令牌操作在内的各种技术来提升权限并访问受保护的资源。这使得身份验证和基于令牌的授权成为安全和有弹性的网络系统的关键组件。在本文中，我们讨论了这样一个安全的、具有弹性的认证和授权框架的设计考虑，该框架能够基于风险分数和信任配置文件自适应。我们将该设计与OAuth 2.0、OpenID Connect和SAML 2.0等现有标准进行了比较。然后，我们研究了流行的威胁模型，如STRIDE和PASA，并总结了所提出的体系结构对常见和相关威胁向量的恢复能力。我们将此框架称为基于弹性风险的自适应身份验证和授权(RAD-AA)。拟议的框架过度增加了对手发动和维持任何网络攻击的成本，并为关键基础设施提供了亟需的力量。我们还讨论了机器学习(ML)方法，使自适应引擎能够准确地对交易进行分类，并得出风险分数。



## **14. Ada3Diff: Defending against 3D Adversarial Point Clouds via Adaptive Diffusion**

Ada3Diff：通过自适应扩散防御3D对抗性点云 cs.CV

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16247v1) [paper-pdf](http://arxiv.org/pdf/2211.16247v1)

**Authors**: Kui Zhang, Hang Zhou, Jie Zhang, Qidong Huang, Weiming Zhang, Nenghai Yu

**Abstract**: Deep 3D point cloud models are sensitive to adversarial attacks, which poses threats to safety-critical applications such as autonomous driving. Robust training and defend-by-denoise are typical strategies for defending adversarial perturbations, including adversarial training and statistical filtering, respectively. However, they either induce massive computational overhead or rely heavily upon specified noise priors, limiting generalized robustness against attacks of all kinds. This paper introduces a new defense mechanism based on denoising diffusion models that can adaptively remove diverse noises with a tailored intensity estimator. Specifically, we first estimate adversarial distortions by calculating the distance of the points to their neighborhood best-fit plane. Depending on the distortion degree, we choose specific diffusion time steps for the input point cloud and perform the forward diffusion to disrupt potential adversarial shifts. Then we conduct the reverse denoising process to restore the disrupted point cloud back to a clean distribution. This approach enables effective defense against adaptive attacks with varying noise budgets, achieving accentuated robustness of existing 3D deep recognition models.

摘要: 深度3D点云模型对对抗性攻击很敏感，这会对自动驾驶等安全关键型应用程序构成威胁。稳健训练和降噪防御分别是防御对抗性扰动的典型策略，包括对抗性训练和统计过滤。然而，它们要么导致巨大的计算开销，要么严重依赖于特定的噪声先验，限制了对所有类型的攻击的普遍健壮性。本文介绍了一种新的基于去噪扩散模型的防御机制，该机制可以通过一个定制的强度估计器自适应地去除各种噪声。具体地说，我们首先通过计算点到其邻域最佳拟合平面的距离来估计对抗性失真。根据失真程度，我们为输入点云选择特定的扩散时间步长，并执行正向扩散来扰乱潜在的对抗性偏移。然后，我们进行反向去噪过程，将中断的点云恢复到干净的分布。这种方法能够有效地防御具有不同噪声预算的自适应攻击，实现了现有3D深度识别模型的增强稳健性。



## **15. Defending Adversarial Attacks on Deep Learning Based Power Allocation in Massive MIMO Using Denoising Autoencoders**

利用去噪自动编码器防御大规模MIMO中基于深度学习的功率分配的敌意攻击 eess.SP

This work is currently under review for publication

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.15365v2) [paper-pdf](http://arxiv.org/pdf/2211.15365v2)

**Authors**: Rajeev Sahay, Minjun Zhang, David J. Love, Christopher G. Brinton

**Abstract**: Recent work has advocated for the use of deep learning to perform power allocation in the downlink of massive MIMO (maMIMO) networks. Yet, such deep learning models are vulnerable to adversarial attacks. In the context of maMIMO power allocation, adversarial attacks refer to the injection of subtle perturbations into the deep learning model's input, during inference (i.e., the adversarial perturbation is injected into inputs during deployment after the model has been trained) that are specifically crafted to force the trained regression model to output an infeasible power allocation solution. In this work, we develop an autoencoder-based mitigation technique, which allows deep learning-based power allocation models to operate in the presence of adversaries without requiring retraining. Specifically, we develop a denoising autoencoder (DAE), which learns a mapping between potentially perturbed data and its corresponding unperturbed input. We test our defense across multiple attacks and in multiple threat models and demonstrate its ability to (i) mitigate the effects of adversarial attacks on power allocation networks using two common precoding schemes, (ii) outperform previously proposed benchmarks for mitigating regression-based adversarial attacks on maMIMO networks, (iii) retain accurate performance in the absence of an attack, and (iv) operate with low computational overhead.

摘要: 最近的工作主张使用深度学习在大规模MIMO(MaMIMO)网络的下行链路中执行功率分配。然而，这种深度学习模型很容易受到对手的攻击。在maMIMO功率分配的上下文中，对抗性攻击是指在推理期间将微妙的扰动注入深度学习模型的输入(即，在模型被训练后在部署期间将对抗性扰动注入输入)，这是专门定制的，以迫使训练的回归模型输出不可行的功率分配解。在这项工作中，我们开发了一种基于自动编码器的缓解技术，该技术允许基于深度学习的功率分配模型在对手存在的情况下运行，而不需要重新训练。具体地说，我们开发了一个去噪自动编码器(DAE)，它学习潜在扰动数据与其对应的未扰动输入之间的映射。我们在多个攻击和多个威胁模型中测试了我们的防御，并证明了它的能力：(I)使用两种常见的预编码方案缓解对抗性攻击对功率分配网络的影响；(Ii)优于先前提出的针对maMIMO网络的基于回归的对抗性攻击基准；(Iii)在没有攻击的情况下保持准确的性能；以及(Iv)以较低的计算开销运行。



## **16. Quantization-aware Interval Bound Propagation for Training Certifiably Robust Quantized Neural Networks**

量化感知区间界传播用于训练可证明稳健的量化神经网络 cs.LG

Accepted at AAAI 2023

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16187v1) [paper-pdf](http://arxiv.org/pdf/2211.16187v1)

**Authors**: Mathias Lechner, Đorđe Žikelić, Krishnendu Chatterjee, Thomas A. Henzinger, Daniela Rus

**Abstract**: We study the problem of training and certifying adversarially robust quantized neural networks (QNNs). Quantization is a technique for making neural networks more efficient by running them using low-bit integer arithmetic and is therefore commonly adopted in industry. Recent work has shown that floating-point neural networks that have been verified to be robust can become vulnerable to adversarial attacks after quantization, and certification of the quantized representation is necessary to guarantee robustness. In this work, we present quantization-aware interval bound propagation (QA-IBP), a novel method for training robust QNNs. Inspired by advances in robust learning of non-quantized networks, our training algorithm computes the gradient of an abstract representation of the actual network. Unlike existing approaches, our method can handle the discrete semantics of QNNs. Based on QA-IBP, we also develop a complete verification procedure for verifying the adversarial robustness of QNNs, which is guaranteed to terminate and produce a correct answer. Compared to existing approaches, the key advantage of our verification procedure is that it runs entirely on GPU or other accelerator devices. We demonstrate experimentally that our approach significantly outperforms existing methods and establish the new state-of-the-art for training and certifying the robustness of QNNs.

摘要: 研究了反向稳健量化神经网络(QNN)的训练和证明问题。量化是一种通过使用低位整数算法运行神经网络来提高神经网络效率的技术，因此在工业中被广泛采用。最近的工作表明，已被验证为健壮性的浮点神经网络在量化后容易受到敌意攻击，而量化表示的证明是保证健壮性的必要条件。在这项工作中，我们提出了量化感知区间界限传播(QA-IBP)，这是一种新的训练稳健QNN的方法。受非量化网络稳健学习的启发，我们的训练算法计算实际网络的抽象表示的梯度。与现有方法不同，我们的方法可以处理QNN的离散语义。在QA-IBP的基础上，我们还开发了一个完整的验证过程来验证QNN的对抗健壮性，该验证过程保证了QNN的终止和产生正确的答案。与现有的方法相比，我们的验证过程的关键优势是它完全运行在GPU或其他加速器设备上。实验表明，我们的方法明显优于现有的方法，并为QNN的训练和证明的健壮性建立了新的技术水平。



## **17. Understanding and Enhancing Robustness of Concept-based Models**

理解和增强基于概念的模型的健壮性 cs.LG

Accepted at AAAI 2023. Extended Version

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16080v1) [paper-pdf](http://arxiv.org/pdf/2211.16080v1)

**Authors**: Sanchit Sinha, Mengdi Huai, Jianhui Sun, Aidong Zhang

**Abstract**: Rising usage of deep neural networks to perform decision making in critical applications like medical diagnosis and financial analysis have raised concerns regarding their reliability and trustworthiness. As automated systems become more mainstream, it is important their decisions be transparent, reliable and understandable by humans for better trust and confidence. To this effect, concept-based models such as Concept Bottleneck Models (CBMs) and Self-Explaining Neural Networks (SENN) have been proposed which constrain the latent space of a model to represent high level concepts easily understood by domain experts in the field. Although concept-based models promise a good approach to both increasing explainability and reliability, it is yet to be shown if they demonstrate robustness and output consistent concepts under systematic perturbations to their inputs. To better understand performance of concept-based models on curated malicious samples, in this paper, we aim to study their robustness to adversarial perturbations, which are also known as the imperceptible changes to the input data that are crafted by an attacker to fool a well-learned concept-based model. Specifically, we first propose and analyze different malicious attacks to evaluate the security vulnerability of concept based models. Subsequently, we propose a potential general adversarial training-based defense mechanism to increase robustness of these systems to the proposed malicious attacks. Extensive experiments on one synthetic and two real-world datasets demonstrate the effectiveness of the proposed attacks and the defense approach.

摘要: 在医疗诊断和金融分析等关键应用中，越来越多的人使用深度神经网络进行决策，这引发了人们对它们的可靠性和可信度的担忧。随着自动化系统变得越来越主流，为了获得更好的信任和信心，重要的是它们的决策要透明、可靠，并能被人类理解。为此，人们提出了基于概念的模型，如概念瓶颈模型(CBMS)和自解释神经网络(SENN)，它们限制了模型的潜在空间，以表示领域专家容易理解的高级概念。尽管基于概念的模型承诺了一种提高可解释性和可靠性的好方法，但它们是否在输入受到系统扰动时表现出稳健性和输出一致的概念还有待证明。为了更好地理解基于概念的模型在经过精选的恶意样本上的性能，在本文中，我们旨在研究它们对对手扰动的鲁棒性，这种扰动也称为输入数据的不可察觉变化，这些变化是攻击者精心制作的，目的是愚弄一个学习良好的基于概念的模型。具体地说，我们首先提出并分析了不同的恶意攻击来评估基于概念的模型的安全漏洞。随后，我们提出了一种潜在的基于一般对抗性训练的防御机制，以增强这些系统对所提出的恶意攻击的健壮性。在一个合成数据集和两个真实数据集上的大量实验证明了所提出的攻击和防御方法的有效性。



## **18. Model Extraction Attack against Self-supervised Speech Models**

针对自监督语音模型的模型提取攻击 cs.SD

Submitted to ICASSP 2023

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16044v1) [paper-pdf](http://arxiv.org/pdf/2211.16044v1)

**Authors**: Tsu-Yuan Hsu, Chen-An Li, Tung-Yu Wu, Hung-yi Lee

**Abstract**: Self-supervised learning (SSL) speech models generate meaningful representations of given clips and achieve incredible performance across various downstream tasks. Model extraction attack (MEA) often refers to an adversary stealing the functionality of the victim model with only query access. In this work, we study the MEA problem against SSL speech model with a small number of queries. We propose a two-stage framework to extract the model. In the first stage, SSL is conducted on the large-scale unlabeled corpus to pre-train a small speech model. Secondly, we actively sample a small portion of clips from the unlabeled corpus and query the target model with these clips to acquire their representations as labels for the small model's second-stage training. Experiment results show that our sampling methods can effectively extract the target model without knowing any information about its model architecture.

摘要: 自监督学习(SSL)语音模型生成给定片段的有意义的表示，并在各种下游任务中获得令人难以置信的性能。模型提取攻击(MEA)通常是指攻击者仅通过查询访问来窃取受害者模型的功能。在这项工作中，我们研究了带有少量查询的SSL语音模型的MEA问题。我们提出了一个两阶段的模型提取框架。在第一阶段，对大规模的未标注语料库进行SSL，以预先训练一个小的语音模型。其次，我们从未标注的语料库中主动采样一小部分片段，并用这些片段查询目标模型，以获得它们的表示作为小模型第二阶段训练的标签。实验结果表明，我们的采样方法可以在不知道目标模型结构的情况下有效地提取目标模型。



## **19. AdvMask: A Sparse Adversarial Attack Based Data Augmentation Method for Image Classification**

AdvMASK：一种基于稀疏对抗性攻击的图像分类数据增强方法 cs.CV

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16040v1) [paper-pdf](http://arxiv.org/pdf/2211.16040v1)

**Authors**: Suorong Yang, Jinqiao Li, Jian Zhao, Furao Shen

**Abstract**: Data augmentation is a widely used technique for enhancing the generalization ability of convolutional neural networks (CNNs) in image classification tasks. Occlusion is a critical factor that affects on the generalization ability of image classification models. In order to generate new samples, existing data augmentation methods based on information deletion simulate occluded samples by randomly removing some areas in the images. However, those methods cannot delete areas of the images according to their structural features of the images. To solve those problems, we propose a novel data augmentation method, AdvMask, for image classification tasks. Instead of randomly removing areas in the images, AdvMask obtains the key points that have the greatest influence on the classification results via an end-to-end sparse adversarial attack module. Therefore, we can find the most sensitive points of the classification results without considering the diversity of various image appearance and shapes of the object of interest. In addition, a data augmentation module is employed to generate structured masks based on the key points, thus forcing the CNN classification models to seek other relevant content when the most discriminative content is hidden. AdvMask can effectively improve the performance of classification models in the testing process. The experimental results on various datasets and CNN models verify that the proposed method outperforms other previous data augmentation methods in image classification tasks.

摘要: 数据增强是一种广泛使用的增强卷积神经网络(CNN)泛化能力的技术。遮挡是影响图像分类模型泛化能力的关键因素。为了生成新的样本，现有的基于信息删除的数据增强方法通过随机去除图像中的某些区域来模拟遮挡样本。然而，这些方法不能根据图像的结构特征来删除图像区域。为了解决这些问题，我们提出了一种新的数据增强方法AdvMASK，用于图像分类任务。该算法通过端到端稀疏对抗攻击模块获取对分类结果影响最大的关键点，而不是随机去除图像中的区域。因此，我们可以在不考虑感兴趣对象的各种图像外观和形状的多样性的情况下，找到分类结果中最敏感的点。此外，使用数据增强模块根据关键点生成结构化掩码，从而迫使CNN分类模型在最具区别性的内容被隐藏时寻找其他相关内容。在测试过程中，AdvMASK可以有效地提高分类模型的性能。在不同数据集和CNN模型上的实验结果验证了该方法在图像分类任务中的性能优于以往的其他数据增强方法。



## **20. Interpretations Cannot Be Trusted: Stealthy and Effective Adversarial Perturbations against Interpretable Deep Learning**

解释不可信：对可解释深度学习的隐秘而有效的对抗性干扰 cs.CR

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.15926v1) [paper-pdf](http://arxiv.org/pdf/2211.15926v1)

**Authors**: Eldor Abdukhamidov, Mohammed Abuhamad, Simon S. Woo, Eric Chan-Tin, Tamer Abuhmed

**Abstract**: Deep learning methods have gained increased attention in various applications due to their outstanding performance. For exploring how this high performance relates to the proper use of data artifacts and the accurate problem formulation of a given task, interpretation models have become a crucial component in developing deep learning-based systems. Interpretation models enable the understanding of the inner workings of deep learning models and offer a sense of security in detecting the misuse of artifacts in the input data. Similar to prediction models, interpretation models are also susceptible to adversarial inputs. This work introduces two attacks, AdvEdge and AdvEdge$^{+}$, that deceive both the target deep learning model and the coupled interpretation model. We assess the effectiveness of proposed attacks against two deep learning model architectures coupled with four interpretation models that represent different categories of interpretation models. Our experiments include the attack implementation using various attack frameworks. We also explore the potential countermeasures against such attacks. Our analysis shows the effectiveness of our attacks in terms of deceiving the deep learning models and their interpreters, and highlights insights to improve and circumvent the attacks.

摘要: 深度学习方法由于其优异的性能在各种应用中得到了越来越多的关注。为了探索这种高性能如何与数据人工制品的正确使用和给定任务的准确问题表达有关，解释模型已经成为开发基于深度学习的系统的关键组成部分。解释模型能够理解深度学习模型的内部工作原理，并在检测输入数据中的伪像误用时提供一种安全感。与预测模型类似，解释模型也容易受到对抗性输入的影响。本文介绍了两种欺骗目标深度学习模型和耦合解释模型的攻击方法：AdvEdge和AdvEdge。我们评估了针对两个深度学习模型体系结构和代表不同类别解释模型的四个解释模型的攻击的有效性。我们的实验包括使用各种攻击框架的攻击实现。我们还探讨了针对此类攻击的潜在对策。我们的分析显示了我们的攻击在欺骗深度学习模型及其解释器方面的有效性，并强调了改进和规避攻击的见解。



## **21. Training Time Adversarial Attack Aiming the Vulnerability of Continual Learning**

针对持续学习脆弱性的训练时间对抗性攻击 cs.LG

Accepted at NeurIPS 2022 ML Safety Workshop

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.15875v1) [paper-pdf](http://arxiv.org/pdf/2211.15875v1)

**Authors**: Gyojin Han, Jaehyun Choi, Hyeong Gwon Hong, Junmo Kim

**Abstract**: Generally, regularization-based continual learning models limit access to the previous task data to imitate the real-world setting which has memory and privacy issues. However, this introduces a problem in these models by not being able to track the performance on each task. In other words, current continual learning methods are vulnerable to attacks done on the previous task. We demonstrate the vulnerability of regularization-based continual learning methods by presenting simple task-specific training time adversarial attack that can be used in the learning process of a new task. Training data generated by the proposed attack causes performance degradation on a specific task targeted by the attacker. Experiment results justify the vulnerability proposed in this paper and demonstrate the importance of developing continual learning models that are robust to adversarial attack.

摘要: 通常，基于正则化的持续学习模型限制对先前任务数据的访问，以模拟存在记忆和隐私问题的真实世界环境。然而，这在这些模型中引入了一个问题，因为无法跟踪每个任务的性能。换句话说，当前的持续学习方法很容易受到对前一任务的攻击。我们通过简单的任务特定训练时间的对抗性攻击来展示基于正则化的持续学习方法的脆弱性，这些攻击可以用于新任务的学习过程。建议的攻击生成的训练数据会导致攻击者针对的特定任务的性能下降。实验结果验证了本文提出的脆弱性，并证明了开发对对手攻击具有健壮性的持续学习模型的重要性。



## **22. How Important are Good Method Names in Neural Code Generation? A Model Robustness Perspective**

好的方法名在神经代码生成中有多重要？模型稳健性视角 cs.SE

UNDER REVIEW

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.15844v1) [paper-pdf](http://arxiv.org/pdf/2211.15844v1)

**Authors**: Guang Yang, Yu Zhou, Wenhua Yang, Tao Yue, Xiang Chen, Taolue Chen

**Abstract**: Pre-trained code generation models (PCGMs) have been widely applied in neural code generation which can generate executable code from functional descriptions in natural languages, possibly together with signatures. Despite substantial performance improvement of PCGMs, the role of method names in neural code generation has not been thoroughly investigated. In this paper, we study and demonstrate the potential of benefiting from method names to enhance the performance of PCGMs, from a model robustness perspective. Specifically, we propose a novel approach, named RADAR (neuRAl coDe generAtor Robustifier). RADAR consists of two components: RADAR-Attack and RADAR-Defense. The former attacks a PCGM by generating adversarial method names as part of the input, which are semantic and visual similar to the original input, but may trick the PCGM to generate completely unrelated code snippets. As a countermeasure to such attacks, RADAR-Defense synthesizes a new method name from the functional description and supplies it to the PCGM. Evaluation results show that RADAR-Attack can, e.g., reduce the CodeBLEU of generated code by 19.72% to 38.74% in three state-of-the-art PCGMs (i.e., CodeGPT, PLBART, and CodeT5). Moreover, RADAR-Defense is able to reinstate the performance of PCGMs with synthesized method names. These results highlight the importance of good method names in neural code generation and implicate the benefits of studying model robustness in software engineering.

摘要: 预先训练的代码生成模型(PCGM)已被广泛应用于神经代码生成中，它可以从自然语言的功能描述中生成可执行代码，并可能与签名一起生成。尽管PCGM的性能有了很大的提高，但方法名称在神经代码生成中的作用还没有得到彻底的研究。在本文中，我们从模型稳健性的角度，研究和论证了受益于方法名称来提高PCGM性能的潜力。具体地说，我们提出了一种新的方法，称为RADAR(神经编码生成器Robustifier)。雷达由两部分组成：雷达攻击和雷达防御。前者通过生成敌对的方法名称作为输入的一部分来攻击PCGM，这些名称在语义和视觉上类似于原始输入，但可能会欺骗PCGM生成完全不相关的代码片段。作为对这类攻击的对策，雷达防御从功能描述中合成了一个新的方法名称，并将其提供给PCGM。评估结果表明，雷达攻击可以在三种最先进的PCGM(即CodeGPT、PLBART和CodeT5)中将生成代码的CodeBLEU减少19.72%到38.74%。此外，雷达防御能够用合成的方法名称恢复PCGM的性能。这些结果突出了好的方法名称在神经代码生成中的重要性，并暗示了在软件工程中研究模型健壮性的好处。



## **23. Attack on Unfair ToS Clause Detection: A Case Study using Universal Adversarial Triggers**

对不公平ToS子句检测的攻击：使用通用对抗触发器的案例研究 cs.CL

Accepted at NLLP@EMNLP2022

**SubmitDate**: 2022-11-28    [abs](http://arxiv.org/abs/2211.15556v1) [paper-pdf](http://arxiv.org/pdf/2211.15556v1)

**Authors**: Shanshan Xu, Irina Broda, Rashid Haddad, Marco Negrini, Matthias Grabmair

**Abstract**: Recent work has demonstrated that natural language processing techniques can support consumer protection by automatically detecting unfair clauses in the Terms of Service (ToS) Agreement. This work demonstrates that transformer-based ToS analysis systems are vulnerable to adversarial attacks. We conduct experiments attacking an unfair-clause detector with universal adversarial triggers. Experiments show that a minor perturbation of the text can considerably reduce the detection performance. Moreover, to measure the detectability of the triggers, we conduct a detailed human evaluation study by collecting both answer accuracy and response time from the participants. The results show that the naturalness of the triggers remains key to tricking readers.

摘要: 最近的工作表明，自然语言处理技术可以通过自动检测服务条款(ToS)中的不公平条款来支持消费者保护。这项工作表明，基于变压器的ToS分析系统容易受到敌意攻击。我们进行了攻击具有通用对抗性触发器的不公平条款检测器的实验。实验表明，文本的微小扰动会显著降低检测性能。此外，为了衡量触发因素的可检测性，我们通过收集参与者的回答准确率和响应时间进行了详细的人类评估研究。结果表明，诱因的自然性仍然是欺骗读者的关键。



## **24. Adversarial Detection by Approximation of Ensemble Boundary**

基于集合边界逼近的对抗性检测 cs.LG

6 pages, 3 figures, 5 tables

**SubmitDate**: 2022-11-28    [abs](http://arxiv.org/abs/2211.10227v2) [paper-pdf](http://arxiv.org/pdf/2211.10227v2)

**Authors**: T. Windeatt

**Abstract**: A spectral approximation of a Boolean function is proposed for approximating the decision boundary of an ensemble of Deep Neural Networks (DNNs) solving two-class pattern recognition problems. The Walsh combination of relatively weak DNN classifiers is shown experimentally to be capable of detecting adversarial attacks. By observing the difference in Walsh coefficient approximation between clean and adversarial images, it appears that transferability of attack may be used for detection. Approximating the decision boundary may also aid in understanding the learning and transferability properties of DNNs. While the experiments here use images, the proposed approach of modelling two-class ensemble decision boundaries could in principle be applied to any application area.

摘要: 提出一种布尔函数的谱逼近方法，用于逼近求解两类模式识别问题的深度神经网络(DNN)集成的决策边界。实验表明，相对较弱的DNN分类器的Walsh组合能够检测到对抗性攻击。通过观察干净图像和敌意图像在沃尔什系数逼近上的差异，可以看出攻击的可转移性可以用于检测。近似决策边界也有助于理解DNN的学习和可转移性。虽然这里的实验使用的是图像，但所提出的建模两类集合决策边界的方法原则上可以应用于任何应用领域。



## **25. DeepSE-WF: Unified Security Estimation for Website Fingerprinting Defenses**

DeepSE-WF：网站指纹防御的统一安全评估 cs.CR

Major revision - added experiments with new dataset and alternative  neural network architectures for estimating the BER

**SubmitDate**: 2022-11-28    [abs](http://arxiv.org/abs/2203.04428v2) [paper-pdf](http://arxiv.org/pdf/2203.04428v2)

**Authors**: Alexander Veicht, Cedric Renggli, Diogo Barradas

**Abstract**: Website fingerprinting (WF) attacks, usually conducted with the help of a machine learning-based classifier, enable a network eavesdropper to pinpoint which web page a user is accessing through the inspection of traffic patterns. These attacks have been shown to succeed even when users browse the Internet through encrypted tunnels, e.g., through Tor or VPNs. To assess the security of new defenses against WF attacks, recent works have proposed feature-dependent theoretical frameworks that estimate the Bayes error of an adversary's features set or the mutual information leaked by manually-crafted features. Unfortunately, as state-of-the-art WF attacks increasingly rely on deep learning and latent feature spaces, security estimations based on simpler (and less informative) manually-crafted features can no longer be trusted to assess the potential success of a WF adversary in defeating such defenses. In this work, we propose DeepSE-WF, a novel WF security estimation framework that leverages specialized kNN-based estimators to produce Bayes error and mutual information estimates from learned latent feature spaces, thus bridging the gap between current WF attacks and security estimation methods. Our evaluation reveals that DeepSE-WF produces tighter security estimates than previous frameworks, reducing the required computational resources to output security estimations by one order of magnitude.

摘要: 网站指纹(WF)攻击通常在基于机器学习的分类器的帮助下进行，使网络窃听者能够通过检查流量模式来确定用户正在访问哪个网页。这些攻击已经被证明是成功的，即使当用户通过加密隧道浏览因特网时，例如通过ToR或VPN。为了评估针对WF攻击的新防御措施的安全性，最近的工作提出了基于特征的理论框架，该框架估计对手的特征集的贝叶斯误差或手动构建的特征泄漏的互信息。不幸的是，随着最先进的WF攻击越来越依赖深度学习和潜在特征空间，基于更简单(且信息量更少)的手动创建的特征的安全估计不再被信任，以评估WF对手在击败此类防御方面的潜在成功。在这项工作中，我们提出了DeepSE-WF，一个新的WF安全估计框架，它利用专门的基于KNN的估计器，从学习的潜在特征空间产生贝叶斯误差和互信息估计，从而弥合了当前WF攻击和安全估计方法之间的差距。我们的评估显示，DeepSE-WF比以前的框架产生更严格的安全估计，将输出安全估计所需的计算资源减少了一个数量级。



## **26. Security Analysis of the Consumer Remote SIM Provisioning Protocol**

消费者远程SIM配置协议的安全性分析 cs.CR

33 pages, 8 figures, Associated ProVerif model files located at  https://github.com/peltona/rsp_model

**SubmitDate**: 2022-11-28    [abs](http://arxiv.org/abs/2211.15323v1) [paper-pdf](http://arxiv.org/pdf/2211.15323v1)

**Authors**: Abu Shohel Ahmed, Aleksi Peltonen, Mohit Sethi, Tuomas Aura

**Abstract**: Remote SIM provisioning (RSP) for consumer devices is the protocol specified by the GSM Association for downloading SIM profiles into a secure element in a mobile device. The process is commonly known as eSIM, and it is expected to replace removable SIM cards. The security of the protocol is critical because the profile includes the credentials with which the mobile device will authenticate to the mobile network. In this paper, we present a formal security analysis of the consumer RSP protocol. We model the multi-party protocol in applied pi calculus, define formal security goals, and verify them in ProVerif. The analysis shows that the consumer RSP protocol protects against a network adversary when all the intended participants are honest. However, we also model the protocol in realistic partial compromise scenarios where the adversary controls a legitimate participant or communication channel. The security failures in the partial compromise scenarios reveal weaknesses in the protocol design. The most important observation is that the security of RSP depends unnecessarily on it being encapsulated in a TLS tunnel. Also, the lack of pre-established identifiers means that a compromised download server anywhere in the world or a compromised secure element can be used for attacks against RSP between honest participants. Additionally, the lack of reliable methods for verifying user intent can lead to serious security failures. Based on the findings, we recommend practical improvements to RSP implementations, to future versions of the specification, and to mobile operator processes to increase the robustness of eSIM security.

摘要: 用于消费者设备的远程SIM供应(RSP)是由GSM协会指定的用于将SIM简档下载到移动设备中的安全元件的协议。这一过程通常被称为eSIM卡，预计将取代可拆卸的SIM卡。协议的安全性是至关重要的，因为简档包括移动设备将用来向移动网络进行认证的凭证。本文对消费者RSP协议进行了形式化的安全性分析。我们用pi演算对多方协议进行了建模，定义了形式化的安全目标，并在ProVerif中进行了验证。分析表明，当所有预期参与者都是诚实的时，消费者RSP协议可以防御网络对手。然而，我们也在现实的部分妥协场景中对协议进行建模，其中对手控制合法的参与者或通信通道。部分妥协场景中的安全故障揭示了协议设计中的弱点。最重要的观察是，RSP的安全性不必要地依赖于它被封装在TLS隧道中。此外，缺乏预先建立的标识符意味着世界上任何地方的受攻击的下载服务器或受攻击的安全元素都可以用于在诚实的参与者之间对RSP进行攻击。此外，缺乏可靠的方法来验证用户意图可能会导致严重的安全故障。基于这些发现，我们建议对RSP实施、该规范的未来版本以及移动运营商流程进行实际改进，以增加eSIM安全的健壮性。



## **27. Adversarial Artifact Detection in EEG-Based Brain-Computer Interfaces**

基于脑电的脑机接口对抗性伪影检测 cs.CR

**SubmitDate**: 2022-11-28    [abs](http://arxiv.org/abs/2212.00727v1) [paper-pdf](http://arxiv.org/pdf/2212.00727v1)

**Authors**: Xiaoqing Chen, Dongrui Wu

**Abstract**: Machine learning has achieved great success in electroencephalogram (EEG) based brain-computer interfaces (BCIs). Most existing BCI research focused on improving its accuracy, but few had considered its security. Recent studies, however, have shown that EEG-based BCIs are vulnerable to adversarial attacks, where small perturbations added to the input can cause misclassification. Detection of adversarial examples is crucial to both the understanding of this phenomenon and the defense. This paper, for the first time, explores adversarial detection in EEG-based BCIs. Experiments on two EEG datasets using three convolutional neural networks were performed to verify the performances of multiple detection approaches. We showed that both white-box and black-box attacks can be detected, and the former are easier to detect.

摘要: 机器学习在基于脑电(EEG)的脑机接口(BCI)领域取得了巨大的成功。现有的大多数脑机接口研究都集中在提高其准确性上，但很少考虑其安全性。然而，最近的研究表明，基于脑电的脑机接口很容易受到对抗性攻击，在输入中添加微小的扰动可能会导致错误分类。对抗性例子的检测对于理解这一现象和进行辩护都是至关重要的。本文首次探讨了基于脑电的脑机接口中的对抗性检测问题。利用三种卷积神经网络在两个脑电数据集上进行了实验，验证了多种检测方法的性能。我们证明了白盒攻击和黑盒攻击都可以被检测到，并且前者更容易检测到。



## **28. Rethinking the Number of Shots in Robust Model-Agnostic Meta-Learning**

健壮模型不可知元学习中镜头数的再思考 cs.CV

**SubmitDate**: 2022-11-28    [abs](http://arxiv.org/abs/2211.15180v1) [paper-pdf](http://arxiv.org/pdf/2211.15180v1)

**Authors**: Xiaoyue Duan, Guoliang Kang, Runqi Wang, Shumin Han, Song Xue, Tian Wang, Baochang Zhang

**Abstract**: Robust Model-Agnostic Meta-Learning (MAML) is usually adopted to train a meta-model which may fast adapt to novel classes with only a few exemplars and meanwhile remain robust to adversarial attacks. The conventional solution for robust MAML is to introduce robustness-promoting regularization during meta-training stage. With such a regularization, previous robust MAML methods simply follow the typical MAML practice that the number of training shots should match with the number of test shots to achieve an optimal adaptation performance. However, although the robustness can be largely improved, previous methods sacrifice clean accuracy a lot. In this paper, we observe that introducing robustness-promoting regularization into MAML reduces the intrinsic dimension of clean sample features, which results in a lower capacity of clean representations. This may explain why the clean accuracy of previous robust MAML methods drops severely. Based on this observation, we propose a simple strategy, i.e., increasing the number of training shots, to mitigate the loss of intrinsic dimension caused by robustness-promoting regularization. Though simple, our method remarkably improves the clean accuracy of MAML without much loss of robustness, producing a robust yet accurate model. Extensive experiments demonstrate that our method outperforms prior arts in achieving a better trade-off between accuracy and robustness. Besides, we observe that our method is less sensitive to the number of fine-tuning steps during meta-training, which allows for a reduced number of fine-tuning steps to improve training efficiency.

摘要: 稳健的模型不可知元学习(MAML)通常被用来训练一个元模型，它可以快速适应新的类别，只需要很少的样本，同时对对手攻击保持健壮。传统的稳健MAML方法是在元训练阶段引入增强稳健性的正则化方法。有了这样的正则化，以前的稳健MAML方法只是遵循典型的MAML实践，即训练镜头的数量应该与测试镜头的数量相匹配，以实现最佳的自适应性能。然而，虽然稳健性可以得到很大的提高，但以前的方法牺牲了很大的精度。在本文中，我们观察到，在MAML中引入增强稳健性的正则化降低了干净样本特征的固有维度，从而导致了干净表示能力的降低。这可能解释了为什么以前健壮的MAML方法的清洁精度严重下降。基于这一观察结果，我们提出了一种简单的策略，即增加训练镜头的数量，以缓解增强稳健性的正则化带来的固有维度的损失。虽然简单，但我们的方法显著提高了MAML的清洁精度，而不会损失太多的稳健性，产生了一个健壮而准确的模型。大量的实验表明，我们的方法在准确性和稳健性之间取得了更好的折衷，优于现有技术。此外，我们观察到我们的方法对元训练过程中的微调步骤数量不那么敏感，从而减少了微调步骤的数量，从而提高了训练效率。



## **29. Adversarial Attack on Radar-based Environment Perception Systems**

雷达环境感知系统的敌意攻击 cs.CR

**SubmitDate**: 2022-11-28    [abs](http://arxiv.org/abs/2211.01112v2) [paper-pdf](http://arxiv.org/pdf/2211.01112v2)

**Authors**: Amira Guesmi, Ihsen Alouani

**Abstract**: Due to their robustness to degraded capturing conditions, radars are widely used for environment perception, which is a critical task in applications like autonomous vehicles. More specifically, Ultra-Wide Band (UWB) radars are particularly efficient for short range settings as they carry rich information on the environment. Recent UWB-based systems rely on Machine Learning (ML) to exploit the rich signature of these sensors. However, ML classifiers are susceptible to adversarial examples, which are created from raw data to fool the classifier such that it assigns the input to the wrong class. These attacks represent a serious threat to systems integrity, especially for safety-critical applications. In this work, we present a new adversarial attack on UWB radars in which an adversary injects adversarial radio noise in the wireless channel to cause an obstacle recognition failure. First, based on signals collected in real-life environment, we show that conventional attacks fail to generate robust noise under realistic conditions. We propose a-RNA, i.e., Adversarial Radio Noise Attack to overcome these issues. Specifically, a-RNA generates an adversarial noise that is efficient without synchronization between the input signal and the noise. Moreover, a-RNA generated noise is, by-design, robust against pre-processing countermeasures such as filtering-based defenses. Moreover, in addition to the undetectability objective by limiting the noise magnitude budget, a-RNA is also efficient in the presence of sophisticated defenses in the spectral domain by introducing a frequency budget. We believe this work should alert about potentially critical implementations of adversarial attacks on radar systems that should be taken seriously.

摘要: 由于雷达对恶劣的捕获条件具有较强的鲁棒性，被广泛用于环境感知，这是自动驾驶汽车等应用中的一项关键任务。更具体地说，超宽带(UWB)雷达对于短距离设置特别有效，因为它们携带丰富的环境信息。最近基于超宽带的系统依赖于机器学习(ML)来利用这些传感器的丰富特征。然而，ML分类器很容易受到敌意示例的影响，这些示例是从原始数据创建的，目的是愚弄分类器，使其将输入分配给错误的类。这些攻击是对系统完整性的严重威胁，尤其是对于安全关键型应用程序。在这项工作中，我们提出了一种新的针对UWB雷达的对抗性攻击，在该攻击中，敌方在无线信道中注入对抗性无线电噪声以导致障碍识别失败。首先，基于实际环境中采集的信号，我们证明了传统的攻击在现实条件下不能产生稳健的噪声。为了克服这些问题，我们提出了a-RNA，即对抗性无线电噪声攻击。具体地说，a-RNA产生对抗性噪声，该噪声在输入信号和噪声之间没有同步的情况下是有效的。此外，根据设计，a-RNA产生的噪声对诸如基于过滤的防御等预处理对策是健壮的。此外，除了通过限制噪声幅度预算的不可检测性目标之外，a-RNA还通过引入频率预算在频谱域中存在复杂的防御时也是有效的。我们认为，这项工作应警惕对雷达系统实施对抗性攻击的潜在关键行动，应予以认真对待。



## **30. Imperceptible Adversarial Attack via Invertible Neural Networks**

基于逆神经网络的潜伏性敌意攻击 cs.CV

**SubmitDate**: 2022-11-28    [abs](http://arxiv.org/abs/2211.15030v1) [paper-pdf](http://arxiv.org/pdf/2211.15030v1)

**Authors**: Zihan Chen, Ziyue Wang, Junjie Huang, Wentao Zhao, Xiao Liu, Dejian Guan

**Abstract**: Adding perturbations via utilizing auxiliary gradient information or discarding existing details of the benign images are two common approaches for generating adversarial examples. Though visual imperceptibility is the desired property of adversarial examples, conventional adversarial attacks still generate traceable adversarial perturbations. In this paper, we introduce a novel Adversarial Attack via Invertible Neural Networks (AdvINN) method to produce robust and imperceptible adversarial examples. Specifically, AdvINN fully takes advantage of the information preservation property of Invertible Neural Networks and thereby generates adversarial examples by simultaneously adding class-specific semantic information of the target class and dropping discriminant information of the original class. Extensive experiments on CIFAR-10, CIFAR-100, and ImageNet-1K demonstrate that the proposed AdvINN method can produce less imperceptible adversarial images than the state-of-the-art methods and AdvINN yields more robust adversarial examples with high confidence compared to other adversarial attacks.

摘要: 通过利用辅助梯度信息添加扰动或丢弃良性图像的现有细节是生成对抗性示例的两种常见方法。虽然视觉不可感知性是对抗性例子的理想属性，但传统的对抗性攻击仍然产生可追踪的对抗性扰动。在本文中，我们介绍了一种新的基于可逆神经网络(AdvINN)的对抗性攻击方法，以产生健壮且不可察觉的对抗性示例。具体而言，AdvINN充分利用了可逆神经网络的信息保持性，通过同时添加目标类的类特定语义信息和丢弃原类的判别信息来生成对抗性实例。在CIFAR-10、CIFAR-100和ImageNet-1K上的大量实验表明，所提出的AdvINN方法可以产生比现有方法更少的不可察觉的对抗性图像，并且与其他对抗性攻击相比，AdvINN产生更健壮的对抗性例子和更高的置信度。



## **31. Adversarial Rademacher Complexity of Deep Neural Networks**

深度神经网络的对抗性Rademacher复杂性 cs.LG

**SubmitDate**: 2022-11-27    [abs](http://arxiv.org/abs/2211.14966v1) [paper-pdf](http://arxiv.org/pdf/2211.14966v1)

**Authors**: Jiancong Xiao, Yanbo Fan, Ruoyu Sun, Zhi-Quan Luo

**Abstract**: Deep neural networks are vulnerable to adversarial attacks. Ideally, a robust model shall perform well on both the perturbed training data and the unseen perturbed test data. It is found empirically that fitting perturbed training data is not hard, but generalizing to perturbed test data is quite difficult. To better understand adversarial generalization, it is of great interest to study the adversarial Rademacher complexity (ARC) of deep neural networks. However, how to bound ARC in multi-layers cases is largely unclear due to the difficulty of analyzing adversarial loss in the definition of ARC. There have been two types of attempts of ARC. One is to provide the upper bound of ARC in linear and one-hidden layer cases. However, these approaches seem hard to extend to multi-layer cases. Another is to modify the adversarial loss and provide upper bounds of Rademacher complexity on such surrogate loss in multi-layer cases. However, such variants of Rademacher complexity are not guaranteed to be bounds for meaningful robust generalization gaps (RGG). In this paper, we provide a solution to this unsolved problem. Specifically, we provide the first bound of adversarial Rademacher complexity of deep neural networks. Our approach is based on covering numbers. We provide a method to handle the robustify function classes of DNNs such that we can calculate the covering numbers. Finally, we provide experiments to study the empirical implication of our bounds and provide an analysis of poor adversarial generalization.

摘要: 深度神经网络很容易受到敌意攻击。理想情况下，稳健模型应该在扰动的训练数据和看不见的扰动的测试数据上都能很好地执行。经验发现，对扰动训练数据进行拟合并不难，但将其推广到扰动测试数据却相当困难。为了更好地理解对抗性泛化，研究深层神经网络的对抗性Rademacher复杂性(ARC)是非常有意义的。然而，由于ARC的定义很难分析对抗性损失，在多层情况下如何界定ARC在很大程度上是不清楚的。ARC有两种类型的尝试。一种是在线性和单隐层情况下给出ARC的上界。然而，这些方法似乎很难扩展到多层情况。二是对对抗性损失进行修正，给出了多层情形下代理损失的Rademacher复杂性上界。然而，Rademacher复杂性的这种变体并不保证是有意义的健壮泛化差距(RGG)的界限。在本文中，我们为这一悬而未决的问题提供了一个解决方案。具体地说，我们给出了深层神经网络对抗性Rademacher复杂性的第一界。我们的方法是基于覆盖数字。我们提供了一种方法来处理DNN的Robutify函数类，以便计算覆盖数。最后，我们提供了实验来研究我们的界限的经验含义，并提供了对对抗性较差的泛化的分析。



## **32. Foiling Explanations in Deep Neural Networks**

深度神经网络中的模糊解释 cs.CV

**SubmitDate**: 2022-11-27    [abs](http://arxiv.org/abs/2211.14860v1) [paper-pdf](http://arxiv.org/pdf/2211.14860v1)

**Authors**: Snir Vitrack Tamam, Raz Lapid, Moshe Sipper

**Abstract**: Deep neural networks (DNNs) have greatly impacted numerous fields over the past decade. Yet despite exhibiting superb performance over many problems, their black-box nature still poses a significant challenge with respect to explainability. Indeed, explainable artificial intelligence (XAI) is crucial in several fields, wherein the answer alone -- sans a reasoning of how said answer was derived -- is of little value. This paper uncovers a troubling property of explanation methods for image-based DNNs: by making small visual changes to the input image -- hardly influencing the network's output -- we demonstrate how explanations may be arbitrarily manipulated through the use of evolution strategies. Our novel algorithm, AttaXAI, a model-agnostic, adversarial attack on XAI algorithms, only requires access to the output logits of a classifier and to the explanation map; these weak assumptions render our approach highly useful where real-world models and data are concerned. We compare our method's performance on two benchmark datasets -- CIFAR100 and ImageNet -- using four different pretrained deep-learning models: VGG16-CIFAR100, VGG16-ImageNet, MobileNet-CIFAR100, and Inception-v3-ImageNet. We find that the XAI methods can be manipulated without the use of gradients or other model internals. Our novel algorithm is successfully able to manipulate an image in a manner imperceptible to the human eye, such that the XAI method outputs a specific explanation map. To our knowledge, this is the first such method in a black-box setting, and we believe it has significant value where explainability is desired, required, or legally mandatory.

摘要: 在过去的十年中，深度神经网络(DNN)对众多领域产生了巨大的影响。然而，尽管在许多问题上表现出了出色的表现，但它们的黑匣子性质仍然在可解释性方面构成了一个重大挑战。事实上，可解释人工智能(XAI)在几个领域都是至关重要的，在这些领域中，答案本身--不考虑答案是如何得出的--几乎没有价值。本文揭示了基于图像的DNN解释方法的一个令人不安的特性：通过对输入图像进行微小的视觉改变--几乎不影响网络的输出--我们演示了如何通过使用进化策略来任意操纵解释。我们的新算法AttaXAI是对XAI算法的一种与模型无关的对抗性攻击，它只需要访问分类器的输出日志和解释地图；这些弱假设使得我们的方法在涉及真实世界的模型和数据时非常有用。我们使用四个不同的预训练深度学习模型：VGG16-CIFAR100、VGG16-ImageNet、MobileNet-CIFAR100和Inception-v3-ImageNet，在两个基准数据集CIFAR100和ImageNet上比较了我们的方法的性能。我们发现，XAI方法可以在不使用梯度或其他模型内部的情况下进行操作。我们的新算法能够成功地以人眼看不到的方式操作图像，从而XAI方法输出特定的解释地图。据我们所知，这是黑盒环境中第一个这样的方法，我们相信它在需要可解释性、要求可解释性或法律强制性的地方具有重要价值。



## **33. Traditional Classification Neural Networks are Good Generators: They are Competitive with DDPMs and GANs**

传统的分类神经网络是很好的生成器：它们与DDPM和GANS具有竞争力 cs.CV

This paper has 29 pages with 22 figures, including rich supplementary  information

**SubmitDate**: 2022-11-27    [abs](http://arxiv.org/abs/2211.14794v1) [paper-pdf](http://arxiv.org/pdf/2211.14794v1)

**Authors**: Guangrun Wang, Philip H. S. Torr

**Abstract**: Classifiers and generators have long been separated. We break down this separation and showcase that conventional neural network classifiers can generate high-quality images of a large number of categories, being comparable to the state-of-the-art generative models (e.g., DDPMs and GANs). We achieve this by computing the partial derivative of the classification loss function with respect to the input to optimize the input to produce an image. Since it is widely known that directly optimizing the inputs is similar to targeted adversarial attacks incapable of generating human-meaningful images, we propose a mask-based stochastic reconstruction module to make the gradients semantic-aware to synthesize plausible images. We further propose a progressive-resolution technique to guarantee fidelity, which produces photorealistic images. Furthermore, we introduce a distance metric loss and a non-trivial distribution loss to ensure classification neural networks can synthesize diverse and high-fidelity images. Using traditional neural network classifiers, we can generate good-quality images of 256$\times$256 resolution on ImageNet. Intriguingly, our method is also applicable to text-to-image generation by regarding image-text foundation models as generalized classifiers.   Proving that classifiers have learned the data distribution and are ready for image generation has far-reaching implications, for classifiers are much easier to train than generative models like DDPMs and GANs. We don't even need to train classification models because tons of public ones are available for download. Also, this holds great potential for the interpretability and robustness of classifiers.

摘要: 分类器和生成器长期以来一直是分开的。我们打破了这种分离，并展示了传统的神经网络分类器可以生成大量类别的高质量图像，可与最先进的生成模型(例如，DDPM和GAN)相媲美。我们通过计算分类损失函数相对于输入的偏导数来实现这一点，以优化输入以产生图像。由于众所周知，直接优化输入类似于无法生成对人类有意义的图像的定向对抗性攻击，我们提出了一种基于掩模的随机重建模型，使梯度能够感知语义，从而合成可信图像。我们进一步提出了一种渐进分辨率技术来保证保真度，从而产生照片级真实感图像。此外，我们还引入了距离度量损失和非平凡分布损失，以确保分类神经网络能够合成各种高保真图像。使用传统的神经网络分类器，我们可以在ImageNet上生成256美元\x 256美元分辨率的高质量图像。有趣的是，我们的方法也适用于文本到图像的生成，因为我们将图像-文本基础模型视为广义分类器。证明分类器已经学习了数据分布并准备好生成图像具有深远的意义，因为分类器比DDPM和Gans等生成模型更容易训练。我们甚至不需要训练分类模型，因为有大量的公共模型可供下载。此外，这对分类器的可解释性和健壮性具有很大的潜力。



## **34. Plausible Adversarial Attacks on Direct Parameter Inference Models in Astrophysics**

天体物理中对直接参数推理模型的貌似对抗性攻击 astro-ph.CO

Accepted submission to Machine Learning and the Physical Sciences  workshop, NeurIPS 2022

**SubmitDate**: 2022-11-27    [abs](http://arxiv.org/abs/2211.14788v1) [paper-pdf](http://arxiv.org/pdf/2211.14788v1)

**Authors**: Benjamin Horowitz, Peter Melchior

**Abstract**: In this abstract we explore the possibility of introducing biases in physical parameter inference models from adversarial-type attacks. In particular, we inject small amplitude systematics into inputs to a mixture density networks tasked with inferring cosmological parameters from observed data. The systematics are constructed analogously to white-box adversarial attacks. We find that the analysis network can be tricked into spurious detection of new physics in cases where standard cosmological estimators would be insensitive. This calls into question the robustness of such networks and their utility for reliably detecting new physics.

摘要: 在这篇摘要中，我们探索了在对抗性攻击的物理参数推理模型中引入偏差的可能性。特别是，我们将小幅度系统学注入到混合密度网络的输入中，该网络的任务是从观测数据推断宇宙学参数。该系统的结构类似于白盒对抗性攻击。我们发现，在标准宇宙学估计器不敏感的情况下，分析网络可以被骗到新物理的虚假检测中。这让人质疑这种网络的健壮性以及它们对可靠地检测新物理的效用。



## **35. Game Theoretic Mixed Experts for Combinational Adversarial Machine Learning**

组合对抗性机器学习的博弈论混合专家 cs.LG

21 pages, 6 figures

**SubmitDate**: 2022-11-26    [abs](http://arxiv.org/abs/2211.14669v1) [paper-pdf](http://arxiv.org/pdf/2211.14669v1)

**Authors**: Ethan Rathbun, Kaleel Mahmood, Sohaib Ahmad, Caiwen Ding, Marten van Dijk

**Abstract**: Recent advances in adversarial machine learning have shown that defenses considered to be robust are actually susceptible to adversarial attacks which are specifically tailored to target their weaknesses. These defenses include Barrage of Random Transforms (BaRT), Friendly Adversarial Training (FAT), Trash is Treasure (TiT) and ensemble models made up of Vision Transformers (ViTs), Big Transfer models and Spiking Neural Networks (SNNs). A natural question arises: how can one best leverage a combination of adversarial defenses to thwart such attacks? In this paper, we provide a game-theoretic framework for ensemble adversarial attacks and defenses which answers this question. In addition to our framework we produce the first adversarial defense transferability study to further motivate a need for combinational defenses utilizing a diverse set of defense architectures. Our framework is called Game theoretic Mixed Experts (GaME) and is designed to find the Mixed-Nash strategy for a defender when facing an attacker employing compositional adversarial attacks. We show that this framework creates an ensemble of defenses with greater robustness than multiple state-of-the-art, single-model defenses in addition to combinational defenses with uniform probability distributions. Overall, our framework and analyses advance the field of adversarial machine learning by yielding new insights into compositional attack and defense formulations.

摘要: 对抗性机器学习的最新进展表明，被认为是健壮的防御实际上容易受到针对其弱点而专门定制的对抗性攻击。这些防御包括随机变换弹幕(BART)、友好对手训练(FAT)、垃圾就是宝藏(TIT)以及由视觉变形金刚(VITS)、大转移模型和尖峰神经网络(SNN)组成的集成模型。一个自然的问题出现了：如何才能最好地利用对抗性防御的组合来挫败这种攻击？在这篇文章中，我们提供了一个关于集成对抗性攻防的博弈论框架，它回答了这个问题。除了我们的框架外，我们还制作了第一个对抗性防御可转移性研究，以进一步激发利用不同的防御体系结构进行组合防御的需求。我们的框架被称为博弈论混合专家(GAME)，旨在找到防御者在面对采用成分对抗攻击的攻击者时的混合纳什策略。我们表明，除了具有均匀概率分布的组合防御之外，该框架还创建了比多个最先进的单一模型防御具有更强稳健性的防御集成。总体而言，我们的框架和分析通过对组合攻击和防御公式产生新的见解，促进了对抗性机器学习领域的发展。



## **36. Minimax Problems with Coupled Linear Constraints: Computational Complexity, Duality and Solution Methods**

耦合线性约束的极大极小问题：计算复杂性、对偶性和求解方法 math.OC

**SubmitDate**: 2022-11-26    [abs](http://arxiv.org/abs/2110.11210v2) [paper-pdf](http://arxiv.org/pdf/2110.11210v2)

**Authors**: Ioannis Tsaknakis, Mingyi Hong, Shuzhong Zhang

**Abstract**: In this work we study a special minimax problem where there are linear constraints that couple both the minimization and maximization decision variables. The problem is a generalization of the traditional saddle point problem (which does not have the coupling constraint), and it finds applications in wireless communication, game theory, transportation, just to name a few. We show that the considered problem is challenging, in the sense that it violates the classical max-min inequality, and that it is NP-hard even under very strong assumptions (e.g., when the objective is strongly convex-strongly concave). We then develop a duality theory for it, and analyze conditions under which the duality gap becomes zero. Finally, we study a class of stationary solutions defined based on the dual problem, and evaluate their practical performance in an application on adversarial attacks on network flow problems.

摘要: 在这项工作中，我们研究了一个特殊的极小极大问题，其中存在耦合最小化和最大化决策变量的线性约束。该问题是传统鞍点问题(不含耦合约束)的推广，在无线通信、博弈论、交通运输等领域有着广泛的应用。我们证明了所考虑的问题是具有挑战性的，因为它违反了经典的最大-最小不等式，并且即使在非常强的假设下(例如，当目标是强凸-强凹的时候)它也是NP-难的。然后，我们发展了它的对偶理论，并分析了对偶差距为零的条件。最后，我们研究了一类基于对偶问题的平稳解，并评价了它们在对抗网络流攻击问题中的实际应用性能。



## **37. SegPGD: An Effective and Efficient Adversarial Attack for Evaluating and Boosting Segmentation Robustness**

SegPGD：一种评估和提高分割健壮性的高效对抗性攻击 cs.CV

**SubmitDate**: 2022-11-25    [abs](http://arxiv.org/abs/2207.12391v2) [paper-pdf](http://arxiv.org/pdf/2207.12391v2)

**Authors**: Jindong Gu, Hengshuang Zhao, Volker Tresp, Philip Torr

**Abstract**: Deep neural network-based image classifications are vulnerable to adversarial perturbations. The image classifications can be easily fooled by adding artificial small and imperceptible perturbations to input images. As one of the most effective defense strategies, adversarial training was proposed to address the vulnerability of classification models, where the adversarial examples are created and injected into training data during training. The attack and defense of classification models have been intensively studied in past years. Semantic segmentation, as an extension of classifications, has also received great attention recently. Recent work shows a large number of attack iterations are required to create effective adversarial examples to fool segmentation models. The observation makes both robustness evaluation and adversarial training on segmentation models challenging. In this work, we propose an effective and efficient segmentation attack method, dubbed SegPGD. Besides, we provide a convergence analysis to show the proposed SegPGD can create more effective adversarial examples than PGD under the same number of attack iterations. Furthermore, we propose to apply our SegPGD as the underlying attack method for segmentation adversarial training. Since SegPGD can create more effective adversarial examples, the adversarial training with our SegPGD can boost the robustness of segmentation models. Our proposals are also verified with experiments on popular Segmentation model architectures and standard segmentation datasets.

摘要: 基于深度神经网络的图像分类容易受到对抗性扰动的影响。通过在输入图像中添加人为的微小和不可察觉的扰动，可以很容易地欺骗图像分类。对抗性训练作为最有效的防御策略之一，被提出用来解决分类模型的脆弱性，即在训练过程中创建对抗性实例并注入训练数据。分类模型的攻防问题在过去的几年里得到了广泛的研究。语义切分作为分类的延伸，近年来也受到了极大的关注。最近的工作表明，需要大量的攻击迭代来创建有效的对抗性示例来愚弄分段模型。这种观察结果使得分割模型的健壮性评估和对抗性训练都具有挑战性。在这项工作中，我们提出了一种有效且高效的分段攻击方法，称为SegPGD。此外，我们还进行了收敛分析，结果表明，在相同的攻击迭代次数下，所提出的SegPGD算法能够生成比PGD算法更有效的攻击实例。此外，我们建议将我们的SegPGD作为分割对手训练的底层攻击方法。由于SegPGD可以生成更有效的对抗性实例，因此使用我们的SegPGD进行对抗性训练可以提高分割模型的稳健性。在流行的分割模型体系结构和标准分割数据集上的实验也验证了我们的建议。



## **38. Beyond Smoothing: Unsupervised Graph Representation Learning with Edge Heterophily Discriminating**

超越平滑：边缘异质性判别的无监督图表示学习 cs.LG

14 pages, 7 tables, 6 figures, accepted by AAAI 2023

**SubmitDate**: 2022-11-25    [abs](http://arxiv.org/abs/2211.14065v1) [paper-pdf](http://arxiv.org/pdf/2211.14065v1)

**Authors**: Yixin Liu, Yizhen Zheng, Daokun Zhang, Vincent CS Lee, Shirui Pan

**Abstract**: Unsupervised graph representation learning (UGRL) has drawn increasing research attention and achieved promising results in several graph analytic tasks. Relying on the homophily assumption, existing UGRL methods tend to smooth the learned node representations along all edges, ignoring the existence of heterophilic edges that connect nodes with distinct attributes. As a result, current methods are hard to generalize to heterophilic graphs where dissimilar nodes are widely connected, and also vulnerable to adversarial attacks. To address this issue, we propose a novel unsupervised Graph Representation learning method with Edge hEterophily discriminaTing (GREET) which learns representations by discriminating and leveraging homophilic edges and heterophilic edges. To distinguish two types of edges, we build an edge discriminator that infers edge homophily/heterophily from feature and structure information. We train the edge discriminator in an unsupervised way through minimizing the crafted pivot-anchored ranking loss, with randomly sampled node pairs acting as pivots. Node representations are learned through contrasting the dual-channel encodings obtained from the discriminated homophilic and heterophilic edges. With an effective interplaying scheme, edge discriminating and representation learning can mutually boost each other during the training phase. We conducted extensive experiments on 14 benchmark datasets and multiple learning scenarios to demonstrate the superiority of GREET.

摘要: 无监督图表示学习(UGRL)已经引起了越来越多的研究关注，并在一些图分析任务中取得了可喜的结果。基于同质性假设，现有的UGRL方法倾向于沿着所有边平滑学习的节点表示，而忽略了连接具有不同属性的节点的异嗜边的存在。因此，现有的方法很难推广到异嗜图，其中不同的节点被广泛连接，并且容易受到对手的攻击。针对这一问题，我们提出了一种新的无监督边异嗜性判别图表示学习方法(GREET)，该方法通过区分并利用同嗜性边和异嗜性边来学习表示。为了区分两种类型的边缘，我们构造了一个边缘鉴别器，它根据特征和结构信息推断边缘的同质性/异质性。我们通过最小化定制的枢轴锚定排序损失，以随机抽样的节点对作为枢轴，以无监督的方式训练边缘鉴别器。通过对比从区分的同亲边和异亲边获得的双通道编码来学习节点表示。在有效的交互作用下，边缘识别和表征学习可以在训练阶段相互促进。我们在14个基准数据集和多个学习场景上进行了广泛的实验，验证了GREET的优越性。



## **39. Cross-Domain Ensemble Distillation for Domain Generalization**

面向领域泛化的跨域集成精馏 cs.CV

Accepted to ECCV 2022. Code is available at  http://github.com/leekyungmoon/XDED

**SubmitDate**: 2022-11-25    [abs](http://arxiv.org/abs/2211.14058v1) [paper-pdf](http://arxiv.org/pdf/2211.14058v1)

**Authors**: Kyungmoon Lee, Sungyeon Kim, Suha Kwak

**Abstract**: Domain generalization is the task of learning models that generalize to unseen target domains. We propose a simple yet effective method for domain generalization, named cross-domain ensemble distillation (XDED), that learns domain-invariant features while encouraging the model to converge to flat minima, which recently turned out to be a sufficient condition for domain generalization. To this end, our method generates an ensemble of the output logits from training data with the same label but from different domains and then penalizes each output for the mismatch with the ensemble. Also, we present a de-stylization technique that standardizes features to encourage the model to produce style-consistent predictions even in an arbitrary target domain. Our method greatly improves generalization capability in public benchmarks for cross-domain image classification, cross-dataset person re-ID, and cross-dataset semantic segmentation. Moreover, we show that models learned by our method are robust against adversarial attacks and image corruptions.

摘要: 领域泛化是学习模型的任务，这些模型泛化到看不见的目标领域。我们提出了一种简单而有效的领域泛化方法，称为跨域集成蒸馏(XDED)，它学习领域不变的特征，同时鼓励模型收敛到平坦极小点，这是最近被证明是领域泛化的充分条件。为此，我们的方法从具有相同标签但来自不同域的训练数据生成输出逻辑集合，然后对与该集合不匹配的每个输出进行惩罚。此外，我们还提出了一种去样式化技术，该技术对特征进行标准化，以鼓励模型即使在任意目标领域中也能产生风格一致的预测。我们的方法大大提高了跨域图像分类、跨数据集Person Re-ID和跨数据集语义分割的公共基准测试的泛化能力。此外，我们还证明了由我们的方法学习的模型对敌意攻击和图像损坏具有较强的鲁棒性。



## **40. Cross-Quality LFW: A Database for Analyzing Cross-Resolution Image Face Recognition in Unconstrained Environments**

Cross-Quality LFW：一个分析不受约束环境下的跨分辨率图像人脸识别的数据库 cs.CV

9 pages, 4 figures, 2 tables

**SubmitDate**: 2022-11-25    [abs](http://arxiv.org/abs/2108.10290v3) [paper-pdf](http://arxiv.org/pdf/2108.10290v3)

**Authors**: Martin Knoche, Stefan Hörmann, Gerhard Rigoll

**Abstract**: Real-world face recognition applications often deal with suboptimal image quality or resolution due to different capturing conditions such as various subject-to-camera distances, poor camera settings, or motion blur. This characteristic has an unignorable effect on performance. Recent cross-resolution face recognition approaches used simple, arbitrary, and unrealistic down- and up-scaling techniques to measure robustness against real-world edge-cases in image quality. Thus, we propose a new standardized benchmark dataset and evaluation protocol derived from the famous Labeled Faces in the Wild (LFW). In contrast to previous derivatives, which focus on pose, age, similarity, and adversarial attacks, our Cross-Quality Labeled Faces in the Wild (XQLFW) maximizes the quality difference. It contains only more realistic synthetically degraded images when necessary. Our proposed dataset is then used to further investigate the influence of image quality on several state-of-the-art approaches. With XQLFW, we show that these models perform differently in cross-quality cases, and hence, the generalizing capability is not accurately predicted by their performance on LFW. Additionally, we report baseline accuracy with recent deep learning models explicitly trained for cross-resolution applications and evaluate the susceptibility to image quality. To encourage further research in cross-resolution face recognition and incite the assessment of image quality robustness, we publish the database and code for evaluation.

摘要: 现实世界中的人脸识别应用程序通常会处理由于拍摄条件不同而导致的次优图像质量或分辨率，例如不同的拍摄对象到摄像机的距离、较差的摄像机设置或运动模糊。这一特征对性能有不可忽视的影响。最近的跨分辨率人脸识别方法使用简单、任意和不切实际的上下缩放技术来衡量图像质量对真实世界边缘情况的鲁棒性。因此，我们提出了一种新的标准化基准数据集和评估协议，该协议源于著名的野外标记人脸(LFW)。与之前关注姿势、年龄、相似性和敌意攻击的衍生品不同，我们的Cross-Quality Label Faces in the Wild(XQLFW)最大化了质量差异。它仅在必要时包含更逼真的合成降级图像。我们提出的数据集被用来进一步研究图像质量对几种最先进方法的影响。对于XQLFW，我们表明这些模型在交叉质量情况下的性能不同，因此，它们在LFW上的性能并不能准确地预测泛化能力。此外，我们报告了最近为跨分辨率应用而明确训练的深度学习模型的基线准确性，并评估了对图像质量的敏感性。为了鼓励对跨分辨率人脸识别的进一步研究，并鼓励对图像质量稳健性的评估，我们公布了用于评估的数据库和代码。



## **41. Let Graph be the Go Board: Gradient-free Node Injection Attack for Graph Neural Networks via Reinforcement Learning**

图为棋盘：基于强化学习的图神经网络无梯度节点注入攻击 cs.LG

AAAI 2023. v2: update acknowledgement section. arXiv admin note:  substantial text overlap with arXiv:2202.09389

**SubmitDate**: 2022-11-25    [abs](http://arxiv.org/abs/2211.10782v2) [paper-pdf](http://arxiv.org/pdf/2211.10782v2)

**Authors**: Mingxuan Ju, Yujie Fan, Chuxu Zhang, Yanfang Ye

**Abstract**: Graph Neural Networks (GNNs) have drawn significant attentions over the years and been broadly applied to essential applications requiring solid robustness or vigorous security standards, such as product recommendation and user behavior modeling. Under these scenarios, exploiting GNN's vulnerabilities and further downgrading its performance become extremely incentive for adversaries. Previous attackers mainly focus on structural perturbations or node injections to the existing graphs, guided by gradients from the surrogate models. Although they deliver promising results, several limitations still exist. For the structural perturbation attack, to launch a proposed attack, adversaries need to manipulate the existing graph topology, which is impractical in most circumstances. Whereas for the node injection attack, though being more practical, current approaches require training surrogate models to simulate a white-box setting, which results in significant performance downgrade when the surrogate architecture diverges from the actual victim model. To bridge these gaps, in this paper, we study the problem of black-box node injection attack, without training a potentially misleading surrogate model. Specifically, we model the node injection attack as a Markov decision process and propose Gradient-free Graph Advantage Actor Critic, namely G2A2C, a reinforcement learning framework in the fashion of advantage actor critic. By directly querying the victim model, G2A2C learns to inject highly malicious nodes with extremely limited attacking budgets, while maintaining a similar node feature distribution. Through our comprehensive experiments over eight acknowledged benchmark datasets with different characteristics, we demonstrate the superior performance of our proposed G2A2C over the existing state-of-the-art attackers. Source code is publicly available at: https://github.com/jumxglhf/G2A2C}.

摘要: 多年来，图神经网络(GNN)引起了人们的广泛关注，并被广泛应用于需要可靠的健壮性或严格的安全标准的重要应用，如产品推荐和用户行为建模。在这些场景下，利用GNN的漏洞并进一步降低其性能成为对手的极大诱因。以前的攻击者主要集中在结构扰动或对现有图的节点注入上，由代理模型的梯度引导。尽管它们带来了令人振奋的结果，但仍然存在一些限制。对于结构扰动攻击，要发起拟议的攻击，攻击者需要操纵现有的图拓扑，这在大多数情况下是不切实际的。而对于节点注入攻击，目前的方法虽然更加实用，但需要训练代理模型来模拟白盒设置，当代理体系结构偏离实际受害者模型时，这会导致性能显著下降。为了弥补这些差距，在本文中，我们研究了黑盒节点注入攻击问题，而不需要训练一个潜在的误导性代理模型。具体地说，我们将节点注入攻击建模为马尔可夫决策过程，提出了无梯度图Advantage Actor Critic，即G2A2C，一种基于Advantage Actor Critic的强化学习框架。通过直接查询受害者模型，G2A2C学习以极其有限的攻击预算注入高度恶意的节点，同时保持类似的节点特征分布。通过我们在八个不同特征的公认基准数据集上的综合实验，我们证明了我们提出的G2A2C比现有的最先进的攻击者具有更好的性能。源代码可在以下网址公开获得：https://github.com/jumxglhf/G2A2C}.



## **42. SAGA: Spectral Adversarial Geometric Attack on 3D Meshes**

SAGA：3D网格上的光谱对抗几何攻击 cs.CV

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2211.13775v1) [paper-pdf](http://arxiv.org/pdf/2211.13775v1)

**Authors**: Tomer Stolik, Itai Lang, Shai Avidan

**Abstract**: A triangular mesh is one of the most popular 3D data representations. As such, the deployment of deep neural networks for mesh processing is widely spread and is increasingly attracting more attention. However, neural networks are prone to adversarial attacks, where carefully crafted inputs impair the model's functionality. The need to explore these vulnerabilities is a fundamental factor in the future development of 3D-based applications. Recently, mesh attacks were studied on the semantic level, where classifiers are misled to produce wrong predictions. Nevertheless, mesh surfaces possess complex geometric attributes beyond their semantic meaning, and their analysis often includes the need to encode and reconstruct the geometry of the shape.   We propose a novel framework for a geometric adversarial attack on a 3D mesh autoencoder. In this setting, an adversarial input mesh deceives the autoencoder by forcing it to reconstruct a different geometric shape at its output. The malicious input is produced by perturbing a clean shape in the spectral domain. Our method leverages the spectral decomposition of the mesh along with additional mesh-related properties to obtain visually credible results that consider the delicacy of surface distortions. Our code is publicly available at https://github.com/StolikTomer/SAGA.

摘要: 三角网格是最流行的3D数据表示形式之一。因此，深度神经网络在网格处理中的应用得到了广泛的应用，并日益引起人们的关注。然而，神经网络容易受到对抗性攻击，精心设计的输入会损害模型的功能。探索这些漏洞的需要是未来基于3D的应用程序开发的一个基本因素。最近，网格攻击被研究在语义层面上，其中分类器被误导以产生错误的预测。然而，网格曲面具有超出其语义含义的复杂几何属性，其分析通常包括需要对形状的几何进行编码和重建。提出了一种针对3D网格自动编码器的几何对抗攻击的新框架。在此设置中，敌意输入网格通过迫使自动编码器在其输出端重建不同的几何形状来欺骗自动编码器。恶意输入是通过干扰谱域中的干净形状而产生的。我们的方法利用网格的频谱分解以及其他与网格相关的属性来获得视觉上可信的结果，该结果考虑了表面扭曲的敏感性。我们的代码在https://github.com/StolikTomer/SAGA.上公开提供



## **43. Backdoor Attack and Defense in Federated Generative Adversarial Network-based Medical Image Synthesis**

基于联邦生成对抗网络的医学图像合成后门攻击与防御 cs.CV

25 pages, 7 figures. arXiv admin note: text overlap with  arXiv:2207.00762

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2210.10886v2) [paper-pdf](http://arxiv.org/pdf/2210.10886v2)

**Authors**: Ruinan Jin, Xiaoxiao Li

**Abstract**: Deep Learning-based image synthesis techniques have been applied in healthcare research for generating medical images to support open research and augment medical datasets. Training generative adversarial neural networks (GANs) usually require large amounts of training data. Federated learning (FL) provides a way of training a central model using distributed data while keeping raw data locally. However, given that the FL server cannot access the raw data, it is vulnerable to backdoor attacks, an adversarial by poisoning training data. Most backdoor attack strategies focus on classification models and centralized domains. It is still an open question if the existing backdoor attacks can affect GAN training and, if so, how to defend against the attack in the FL setting. In this work, we investigate the overlooked issue of backdoor attacks in federated GANs (FedGANs). The success of this attack is subsequently determined to be the result of some local discriminators overfitting the poisoned data and corrupting the local GAN equilibrium, which then further contaminates other clients when averaging the generator's parameters and yields high generator loss. Therefore, we proposed FedDetect, an efficient and effective way of defending against the backdoor attack in the FL setting, which allows the server to detect the client's adversarial behavior based on their losses and block the malicious clients. Our extensive experiments on two medical datasets with different modalities demonstrate the backdoor attack on FedGANs can result in synthetic images with low fidelity. After detecting and suppressing the detected malicious clients using the proposed defense strategy, we show that FedGANs can synthesize high-quality medical datasets (with labels) for data augmentation to improve classification models' performance.

摘要: 基于深度学习的图像合成技术已被应用于医疗保健研究中，以生成医学图像以支持开放研究和扩充医学数据集。生成性对抗神经网络的训练通常需要大量的训练数据。联合学习(FL)提供了一种在本地保留原始数据的同时使用分布式数据训练中央模型的方法。然而，鉴于FL服务器无法访问原始数据，它很容易受到后门攻击，这是通过毒化训练数据而产生的对抗性攻击。大多数后门攻击策略侧重于分类模型和集中域。现有的后门攻击是否会影响GAN的训练，如果是的话，在FL环境下如何防御攻击，仍然是一个悬而未决的问题。在这项工作中，我们研究了联邦GAN(FedGAN)中被忽视的后门攻击问题。这种攻击的成功随后被确定为一些本地鉴别器过度拟合有毒数据并破坏本地GaN平衡的结果，这随后在平均发电机参数时进一步污染其他客户端，并产生高发电机损耗。因此，我们提出了FedDetect，这是一种在FL环境下有效防御后门攻击的方法，它允许服务器根据客户端的损失来检测客户端的敌对行为，并阻止恶意客户端。我们在两个不同模式的医学数据集上的广泛实验表明，对FedGan的后门攻击可以导致合成图像的低保真度。在使用该防御策略检测和抑制检测到的恶意客户端后，我们证明了FedGans能够合成高质量的医学数据集(带标签)用于数据增强，从而提高分类模型的性能。



## **44. Enhancing Targeted Attack Transferability via Diversified Weight Pruning**

通过不同的权重剪枝提高目标攻击的可转移性 cs.CV

8 pages + Appendix

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2208.08677v2) [paper-pdf](http://arxiv.org/pdf/2208.08677v2)

**Authors**: Hung-Jui Wang, Yu-Yu Wu, Shang-Tse Chen

**Abstract**: Malicious attackers can generate targeted adversarial examples by imposing tiny noises, forcing neural networks to produce specific incorrect outputs. With cross-model transferability, network models remain vulnerable even in black-box settings. Recent studies have shown the effectiveness of ensemble-based methods in generating transferable adversarial examples. To further enhance transferability, model augmentation methods aim to produce more networks participating in the ensemble. However, existing model augmentation methods are only proven effective in untargeted attacks. In this work, we propose Diversified Weight Pruning (DWP), a novel model augmentation technique for generating transferable targeted attacks. DWP leverages the weight pruning method commonly used in model compression. Compared with prior work, DWP protects necessary connections and ensures the diversity of the pruned models simultaneously, which we show are crucial for targeted transferability. Experiments on the ImageNet-compatible dataset under various and more challenging scenarios confirm the effectiveness: transferring to adversarially trained models, Non-CNN architectures, and Google Cloud Vision. The results show that our proposed DWP improves the targeted attack success rates with up to $10.1$%, $6.6$%, and $7.0$% on the combination of state-of-the-art methods, respectively. The source code will be made available after acceptance.

摘要: 恶意攻击者可以通过施加微小噪音来生成有针对性的对抗性示例，迫使神经网络生成特定的错误输出。由于具有跨模型的可转移性，网络模型即使在黑盒设置中也仍然容易受到攻击。最近的研究表明，基于集成的方法在生成可转移的对抗性例子方面是有效的。为了进一步提高可转移性，模型扩充方法旨在产生更多参与集成的网络。然而，现有的模型增强方法仅在非目标攻击中被证明是有效的。在这项工作中，我们提出了一种新的模型增强技术--DiversifiedWeight Puning(DWP)，用于生成可转移的目标攻击。DWP利用模型压缩中常用的权重修剪方法。与以前的工作相比，DWP保护了必要的连接，同时确保了剪枝模型的多样性，我们表明这对于定向转移至关重要。在各种更具挑战性的场景下对ImageNet兼容数据集的实验证实了该方法的有效性：转换到经过对抗性训练的模型、非CNN架构和Google Cloud Vision。结果表明，与最新的攻击方法相结合，我们提出的DWP分别提高了10.1$%、6.6$%和7.0$%的目标攻击成功率。源代码将在验收后提供。



## **45. Tracking Dataset IP Use in Deep Neural Networks**

跟踪数据集IP在深度神经网络中的应用 cs.CR

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2211.13535v1) [paper-pdf](http://arxiv.org/pdf/2211.13535v1)

**Authors**: Seonhye Park, Alsharif Abuadbba, Shuo Wang, Kristen Moore, Yansong Gao, Hyoungshick Kim, Surya Nepal

**Abstract**: Training highly performant deep neural networks (DNNs) typically requires the collection of a massive dataset and the use of powerful computing resources. Therefore, unauthorized redistribution of private pre-trained DNNs may cause severe economic loss for model owners. For protecting the ownership of DNN models, DNN watermarking schemes have been proposed by embedding secret information in a DNN model and verifying its presence for model ownership. However, existing DNN watermarking schemes compromise the model utility and are vulnerable to watermark removal attacks because a model is modified with a watermark. Alternatively, a new approach dubbed DEEPJUDGE was introduced to measure the similarity between a suspect model and a victim model without modifying the victim model. However, DEEPJUDGE would only be designed to detect the case where a suspect model's architecture is the same as a victim model's. In this work, we propose a novel DNN fingerprinting technique dubbed DEEPTASTER to prevent a new attack scenario in which a victim's data is stolen to build a suspect model. DEEPTASTER can effectively detect such data theft attacks even when a suspect model's architecture differs from a victim model's. To achieve this goal, DEEPTASTER generates a few adversarial images with perturbations, transforms them into the Fourier frequency domain, and uses the transformed images to identify the dataset used in a suspect model. The intuition is that those adversarial images can be used to capture the characteristics of DNNs built on a specific dataset. We evaluated the detection accuracy of DEEPTASTER on three datasets with three model architectures under various attack scenarios, including transfer learning, pruning, fine-tuning, and data augmentation. Overall, DEEPTASTER achieves a balanced accuracy of 94.95%, which is significantly better than 61.11% achieved by DEEPJUDGE in the same settings.

摘要: 训练高性能的深度神经网络(DNN)通常需要收集大量数据并使用强大的计算资源。因此，未经授权重新分发私人预先训练的DNN可能会给模型所有者造成严重的经济损失。为了保护DNN模型的所有权，已经提出了DNN水印方案，该方案通过在DNN模型中嵌入秘密信息并验证其是否存在来实现模型所有权。然而，现有的DNN水印方案损害了模型的实用性，并且容易受到水印去除攻击，因为模型是用水印修改的。或者，引入了一种称为DEEPJUDGE的新方法来衡量嫌疑人模型和受害者模型之间的相似性，而不需要修改受害者模型。然而，DEEPJUDGE只被设计用于检测可疑模型与受害者模型的体系结构相同的情况。在这项工作中，我们提出了一种名为DEEPTASTER的新型DNN指纹识别技术，以防止新的攻击场景，即受害者的数据被窃取来构建嫌疑人模型。DEEPTASTER即使在可疑模型的体系结构与受害者模型不同的情况下也能有效地检测到这样的数据窃取攻击。为了实现这一目标，DEEPTASTER生成一些带有扰动的对抗性图像，将它们变换到傅立叶频域，并使用变换后的图像来识别可疑模型中使用的数据集。人们的直觉是，这些对抗性图像可以用来捕捉建立在特定数据集上的DNN的特征。在转移学习、剪枝、微调和数据增强等不同攻击场景下，使用三种模型架构对DEEPTASTER在三个数据集上的检测精度进行了评估。总体而言，DEEPTASTER达到了94.95%的均衡准确率，明显好于DEEPJUDGE在相同设置下实现的61.11%。



## **46. Reliability and Robustness analysis of Machine Learning based Phishing URL Detectors**

基于机器学习的钓鱼URL检测器的可靠性和稳健性分析 cs.CR

Accepted in Transactions of Dependable and Secure Computing  (SI-Reliability and Robustness in AI-Based Cybersecurity Solutions)

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2005.08454v3) [paper-pdf](http://arxiv.org/pdf/2005.08454v3)

**Authors**: Bushra Sabir, M. Ali Babar, Raj Gaire, Alsharif Abuadbba

**Abstract**: ML-based Phishing URL (MLPU) detectors serve as the first level of defence to protect users and organisations from being victims of phishing attacks. Lately, few studies have launched successful adversarial attacks against specific MLPU detectors raising questions about their practical reliability and usage. Nevertheless, the robustness of these systems has not been extensively investigated. Therefore, the security vulnerabilities of these systems, in general, remain primarily unknown which calls for testing the robustness of these systems. In this article, we have proposed a methodology to investigate the reliability and robustness of 50 representative state-of-the-art MLPU models. Firstly, we have proposed a cost-effective Adversarial URL generator URLBUG that created an Adversarial URL dataset. Subsequently, we reproduced 50 MLPU (traditional ML and Deep learning) systems and recorded their baseline performance. Lastly, we tested the considered MLPU systems on Adversarial Dataset and analyzed their robustness and reliability using box plots and heat maps. Our results showed that the generated adversarial URLs have valid syntax and can be registered at a median annual price of \$11.99. Out of 13\% of the already registered adversarial URLs, 63.94\% were used for malicious purposes. Moreover, the considered MLPU models Matthew Correlation Coefficient (MCC) dropped from a median 0.92 to 0.02 when tested against $Adv_\mathrm{data}$, indicating that the baseline MLPU models are unreliable in their current form. Further, our findings identified several security vulnerabilities of these systems and provided future directions for researchers to design dependable and secure MLPU systems.

摘要: 基于ML的钓鱼URL(MLPU)检测器是保护用户和组织免受钓鱼攻击的第一级防御。最近，很少有研究针对特定的MLPU检测器发起成功的对抗性攻击，这引发了对其实际可靠性和使用的质疑。然而，这些系统的稳健性还没有得到广泛的研究。因此，这些系统的安全漏洞总体上仍然是未知的，这就需要测试这些系统的健壮性。在本文中，我们提出了一种方法来调查50个最具代表性的MLPU模型的可靠性和稳健性。首先，我们提出了一个高性价比的敌意URL生成器URLBUG，它创建了一个敌意URL数据集。随后，我们复制了50个MLPU(传统ML和深度学习)系统，并记录了它们的基线性能。最后，我们在敌意数据集上对所考虑的MLPU系统进行了测试，并使用盒图和热图分析了它们的健壮性和可靠性。我们的结果表明，生成的恶意URL具有有效的语法，并且可以以11.99美元的中位数年价格注册。在已注册的13个恶意URL中，有63.94个被用于恶意目的。此外，所考虑的MLPU模型马修相关系数(MCC)从中位数0.92下降到0.02，表明基线MLPU模型目前的形式是不可靠的。此外，我们的发现发现了这些系统的几个安全漏洞，并为研究人员设计可靠和安全的MLPU系统提供了未来的方向。



## **47. Explainable and Safe Reinforcement Learning for Autonomous Air Mobility**

可解释且安全的自主空中机动强化学习 cs.LG

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2211.13474v1) [paper-pdf](http://arxiv.org/pdf/2211.13474v1)

**Authors**: Lei Wang, Hongyu Yang, Yi Lin, Suwan Yin, Yuankai Wu

**Abstract**: Increasing traffic demands, higher levels of automation, and communication enhancements provide novel design opportunities for future air traffic controllers (ATCs). This article presents a novel deep reinforcement learning (DRL) controller to aid conflict resolution for autonomous free flight. Although DRL has achieved important advancements in this field, the existing works pay little attention to the explainability and safety issues related to DRL controllers, particularly the safety under adversarial attacks. To address those two issues, we design a fully explainable DRL framework wherein we: 1) decompose the coupled Q value learning model into a safety-awareness and efficiency (reach the target) one; and 2) use information from surrounding intruders as inputs, eliminating the needs of central controllers. In our simulated experiments, we show that by decoupling the safety-awareness and efficiency, we can exceed performance on free flight control tasks while dramatically improving explainability on practical. In addition, the safety Q learning module provides rich information about the safety situation of environments. To study the safety under adversarial attacks, we additionally propose an adversarial attack strategy that can impose both safety-oriented and efficiency-oriented attacks. The adversarial aims to minimize safety/efficiency by only attacking the agent at a few time steps. In the experiments, our attack strategy increases as many collisions as the uniform attack (i.e., attacking at every time step) by only attacking the agent four times less often, which provide insights into the capabilities and restrictions of the DRL in future ATC designs. The source code is publicly available at https://github.com/WLeiiiii/Gym-ATC-Attack-Project.

摘要: 日益增长的交通需求、更高水平的自动化和通信的增强为未来的空中交通管制员(ATC)提供了新的设计机会。本文提出了一种新的深度强化学习(DRL)控制器来辅助自主自由飞行中的冲突消解。虽然DRL在这一领域已经取得了重要的进展，但现有的工作很少关注与DRL控制器相关的可解释性和安全性问题，特别是在对抗攻击下的安全性。为了解决这两个问题，我们设计了一个完全可解释的DRL框架，其中：1)将耦合的Q值学习模型分解为安全意识和效率(达到目标)模型；2)使用来自周围入侵者的信息作为输入，消除了中央控制器的需求。在我们的模拟实验中，我们表明，通过将安全意识和效率解耦，我们可以在自由飞行控制任务上超越性能，同时在实际应用中显著提高可解释性。此外，安全Q学习模块提供了关于环境安全状况的丰富信息。为了研究对抗性攻击下的安全性，我们还提出了一种对抗性攻击策略，它既可以实施面向安全的攻击，也可以实施面向效率的攻击。对抗性的目的是通过只在几个时间步攻击代理来最小化安全/效率。在实验中，我们的攻击策略增加了与统一攻击(即在每个时间步攻击)相同的冲突次数，仅减少了对代理的攻击次数的四倍，这为未来ATC设计中DRL的能力和限制提供了深入的见解。源代码可在https://github.com/WLeiiiii/Gym-ATC-Attack-Project.上公开获得



## **48. Dikaios: Privacy Auditing of Algorithmic Fairness via Attribute Inference Attacks**

Dikaios：基于属性推理攻击的算法公平性隐私审计 cs.CR

The paper's results and conclusions underwent significant changes.  The updated paper can be found at arXiv:2211.10209

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2202.02242v2) [paper-pdf](http://arxiv.org/pdf/2202.02242v2)

**Authors**: Jan Aalmoes, Vasisht Duddu, Antoine Boutet

**Abstract**: Machine learning (ML) models have been deployed for high-stakes applications. Due to class imbalance in the sensitive attribute observed in the datasets, ML models are unfair on minority subgroups identified by a sensitive attribute, such as race and sex. In-processing fairness algorithms ensure model predictions are independent of sensitive attribute. Furthermore, ML models are vulnerable to attribute inference attacks where an adversary can identify the values of sensitive attribute by exploiting their distinguishable model predictions. Despite privacy and fairness being important pillars of trustworthy ML, the privacy risk introduced by fairness algorithms with respect to attribute leakage has not been studied. We identify attribute inference attacks as an effective measure for auditing blackbox fairness algorithms to enable model builder to account for privacy and fairness in the model design. We proposed Dikaios, a privacy auditing tool for fairness algorithms for model builders which leveraged a new effective attribute inference attack that account for the class imbalance in sensitive attributes through an adaptive prediction threshold. We evaluated Dikaios to perform a privacy audit of two in-processing fairness algorithms over five datasets. We show that our attribute inference attacks with adaptive prediction threshold significantly outperform prior attacks. We highlighted the limitations of in-processing fairness algorithms to ensure indistinguishable predictions across different values of sensitive attributes. Indeed, the attribute privacy risk of these in-processing fairness schemes is highly variable according to the proportion of the sensitive attributes in the dataset. This unpredictable effect of fairness mechanisms on the attribute privacy risk is an important limitation on their utilization which has to be accounted by the model builder.

摘要: 机器学习(ML)模型已被部署用于高风险应用程序。由于在数据集中观察到的敏感属性的类别不平衡，ML模型对由敏感属性识别的少数群体是不公平的，例如种族和性别。处理中的公平算法确保模型预测独立于敏感属性。此外，ML模型容易受到属性推理攻击，攻击者可以利用敏感属性的可区分模型预测来识别敏感属性的值。尽管隐私和公平是可信ML的重要支柱，但公平算法在属性泄漏方面引入的隐私风险尚未被研究。我们将属性推理攻击作为审计黑盒公平算法的一种有效手段，使模型构建者能够在模型设计中考虑隐私和公平。我们提出了Dikaios，这是一个针对模型构建者公平算法的隐私审计工具，它利用了一种新的有效的属性推理攻击，通过自适应预测阈值来解释敏感属性中的类不平衡。我们评估了Dikaios在五个数据集上对两个正在处理的公平算法进行隐私审计的能力。实验结果表明，基于自适应预测阈值的属性推理攻击性能明显优于已有的攻击。我们强调了处理中公平算法的局限性，以确保对敏感属性的不同值进行不可区分的预测。事实上，这些正在处理的公平方案的属性隐私风险根据敏感属性在数据集中的比例而变化很大。公平机制对属性隐私风险的这种不可预测的影响是对其使用的一个重要限制，这必须由模型构建者考虑。



## **49. Blackbox Attacks via Surrogate Ensemble Search**

通过代理集成搜索进行黑盒攻击 cs.LG

Our code is available at https://github.com/CSIPlab/BASES

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2208.03610v2) [paper-pdf](http://arxiv.org/pdf/2208.03610v2)

**Authors**: Zikui Cai, Chengyu Song, Srikanth Krishnamurthy, Amit Roy-Chowdhury, M. Salman Asif

**Abstract**: Blackbox adversarial attacks can be categorized into transfer- and query-based attacks. Transfer methods do not require any feedback from the victim model, but provide lower success rates compared to query-based methods. Query attacks often require a large number of queries for success. To achieve the best of both approaches, recent efforts have tried to combine them, but still require hundreds of queries to achieve high success rates (especially for targeted attacks). In this paper, we propose a novel method for Blackbox Attacks via Surrogate Ensemble Search (BASES) that can generate highly successful blackbox attacks using an extremely small number of queries. We first define a perturbation machine that generates a perturbed image by minimizing a weighted loss function over a fixed set of surrogate models. To generate an attack for a given victim model, we search over the weights in the loss function using queries generated by the perturbation machine. Since the dimension of the search space is small (same as the number of surrogate models), the search requires a small number of queries. We demonstrate that our proposed method achieves better success rate with at least 30x fewer queries compared to state-of-the-art methods on different image classifiers trained with ImageNet. In particular, our method requires as few as 3 queries per image (on average) to achieve more than a 90% success rate for targeted attacks and 1-2 queries per image for over a 99% success rate for untargeted attacks. Our method is also effective on Google Cloud Vision API and achieved a 91% untargeted attack success rate with 2.9 queries per image. We also show that the perturbations generated by our proposed method are highly transferable and can be adopted for hard-label blackbox attacks. We also show effectiveness of BASES for hiding attacks on object detectors.

摘要: 黑盒对抗性攻击可分为基于传输的攻击和基于查询的攻击。传输方法不需要来自受害者模型的任何反馈，但与基于查询的方法相比，提供了更低的成功率。查询攻击通常需要大量查询才能成功。为了达到这两种方法的最佳效果，最近的努力试图将它们结合起来，但仍然需要数百次查询才能获得高成功率(特别是针对有针对性的攻击)。本文提出了一种新的基于代理集成搜索(BASS)的黑盒攻击方法，该方法可以用极少的查询生成非常成功的黑盒攻击。我们首先定义了一种微扰机，它通过最小化一组固定代理模型上的加权损失函数来生成扰动图像。为了针对给定的受害者模型生成攻击，我们使用由扰动机器生成的查询来搜索损失函数中的权重。由于搜索空间的维度很小(与代理模型的数量相同)，因此搜索需要少量的查询。实验结果表明，与现有的基于ImageNet训练的不同图像分类器的方法相比，本文提出的方法在至少30倍的查询次数下获得了更好的分类成功率。特别是，我们的方法只需每幅图像3次查询(平均)即可实现90%以上的定向攻击成功率，以及每幅图像1-2次查询的非定向攻击成功率99%以上。我们的方法在Google Cloud Vision API上也是有效的，在每张图片2.9个查询的情况下，获得了91%的非定向攻击成功率。我们还证明了我们提出的方法产生的扰动具有很高的可转移性，可以用于硬标签黑盒攻击。我们还展示了用于隐藏对对象检测器的攻击的基础的有效性。



## **50. Modelling Direct Messaging Networks with Multiple Recipients for Cyber Deception**

具有多个接收者的网络欺骗直接消息传递网络建模 cs.CR

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2111.11932v2) [paper-pdf](http://arxiv.org/pdf/2111.11932v2)

**Authors**: Kristen Moore, Cody J. Christopher, David Liebowitz, Surya Nepal, Renee Selvey

**Abstract**: Cyber deception is emerging as a promising approach to defending networks and systems against attackers and data thieves. However, despite being relatively cheap to deploy, the generation of realistic content at scale is very costly, due to the fact that rich, interactive deceptive technologies are largely hand-crafted. With recent improvements in Machine Learning, we now have the opportunity to bring scale and automation to the creation of realistic and enticing simulated content. In this work, we propose a framework to automate the generation of email and instant messaging-style group communications at scale. Such messaging platforms within organisations contain a lot of valuable information inside private communications and document attachments, making them an enticing target for an adversary. We address two key aspects of simulating this type of system: modelling when and with whom participants communicate, and generating topical, multi-party text to populate simulated conversation threads. We present the LogNormMix-Net Temporal Point Process as an approach to the first of these, building upon the intensity-free modeling approach of Shchur et al. to create a generative model for unicast and multi-cast communications. We demonstrate the use of fine-tuned, pre-trained language models to generate convincing multi-party conversation threads. A live email server is simulated by uniting our LogNormMix-Net TPP (to generate the communication timestamp, sender and recipients) with the language model, which generates the contents of the multi-party email threads. We evaluate the generated content with respect to a number of realism-based properties, that encourage a model to learn to generate content that will engage the attention of an adversary to achieve a deception outcome.

摘要: 网络欺骗正在成为保护网络和系统免受攻击者和数据窃贼攻击的一种很有前途的方法。然而，尽管部署成本相对较低，但大规模生成逼真内容的成本非常高，因为丰富的交互式欺骗性技术主要是手工制作的。随着机器学习最近的改进，我们现在有机会将规模化和自动化带到创建逼真和诱人的模拟内容的过程中。在这项工作中，我们提出了一个框架，以自动生成大规模的电子邮件和即时消息风格的群组通信。组织内部的此类消息传递平台在私人通信和文档附件中包含大量有价值的信息，使它们成为诱人的对手攻击目标。我们解决了模拟这种类型的系统的两个关键方面：模拟参与者何时以及与谁交流，以及生成主题多方文本以填充模拟的对话线索。我们提出了LogNormMix-net时点过程作为第一种方法，建立在Shchur等人的无强度建模方法的基础上。创建单播和多播通信的生成性模型。我们演示了如何使用经过微调、预先训练的语言模型来生成令人信服的多方对话线索。通过将我们的LogNormMix-Net TPP(生成通信时间戳、发送者和接收者)与语言模型相结合来模拟实时电子邮件服务器，语言模型生成多方电子邮件线程的内容。我们根据一些基于现实主义的属性来评估生成的内容，这些属性鼓励模型学习生成将吸引对手注意力的内容，以实现欺骗结果。



