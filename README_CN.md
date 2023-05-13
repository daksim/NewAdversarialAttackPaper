# Latest Adversarial Attack Papers
**update at 2023-05-13 15:24:10**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. A theoretical basis for Blockchain Extractable Value**

区块链可提取价值的理论基础 cs.CR

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2302.02154v2) [paper-pdf](http://arxiv.org/pdf/2302.02154v2)

**Authors**: Massimo Bartoletti, Roberto Zunino

**Abstract**: Extractable Value refers to a wide class of economic attacks to public blockchains, where adversaries with the power to reorder, drop or insert transactions in a block can "extract" value from smart contracts. Empirical research has shown that mainstream protocols, like e.g. decentralized exchanges, are massively targeted by these attacks, with detrimental effects on their users and on the blockchain network. Despite the growing impact of these attacks in the real world, theoretical foundations are still missing. We propose a formal theory of Extractable Value, based on a general, abstract model of blockchains and smart contracts. Our theory is the basis for proofs of security against Extractable Value attacks.

摘要: 可提取价值指的是对公共区块链的一大类经济攻击，在这些攻击中，有能力在区块中重新排序、丢弃或插入交易的对手可以从智能合约中“提取”价值。经验研究表明，主流协议，如分散交换，是这些攻击的大规模目标，对其用户和区块链网络造成有害影响。尽管这些袭击在现实世界中的影响越来越大，但理论基础仍然缺乏。基于区块链和智能合约的一般抽象模型，我们提出了可提取价值的形式理论。我们的理论是针对可提取值攻击的安全性证明的基础。



## **2. Improving Hyperspectral Adversarial Robustness Under Multiple Attacks**

提高多重攻击下的高光谱对抗健壮性 cs.LG

6 pages, 2 figures, 1 table, 1 algorithm

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2210.16346v4) [paper-pdf](http://arxiv.org/pdf/2210.16346v4)

**Authors**: Nicholas Soucy, Salimeh Yasaei Sekeh

**Abstract**: Semantic segmentation models classifying hyperspectral images (HSI) are vulnerable to adversarial examples. Traditional approaches to adversarial robustness focus on training or retraining a single network on attacked data, however, in the presence of multiple attacks these approaches decrease in performance compared to networks trained individually on each attack. To combat this issue we propose an Adversarial Discriminator Ensemble Network (ADE-Net) which focuses on attack type detection and adversarial robustness under a unified model to preserve per data-type weight optimally while robustifiying the overall network. In the proposed method, a discriminator network is used to separate data by attack type into their specific attack-expert ensemble network.

摘要: 对高光谱图像进行分类的语义分割模型容易受到敌意例子的影响。传统的对抗稳健性方法侧重于针对受攻击的数据训练或重新训练单个网络，然而，在存在多个攻击的情况下，与针对每个攻击单独训练的网络相比，这些方法的性能会下降。为了解决这个问题，我们提出了一种对抗性鉴别集成网络(ADE-Net)，它在统一的模型下关注攻击类型的检测和对抗性的健壮性，以便在使整个网络稳健的同时最优地保持每种数据类型的权重。在该方法中，利用鉴别器网络根据攻击类型将数据分离到其特定的攻击专家集成网络中。



## **3. Run-Off Election: Improved Provable Defense against Data Poisoning Attacks**

决选：改进了针对数据中毒攻击的可证明防御 cs.LG

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2302.02300v2) [paper-pdf](http://arxiv.org/pdf/2302.02300v2)

**Authors**: Keivan Rezaei, Kiarash Banihashem, Atoosa Chegini, Soheil Feizi

**Abstract**: In data poisoning attacks, an adversary tries to change a model's prediction by adding, modifying, or removing samples in the training data. Recently, ensemble-based approaches for obtaining provable defenses against data poisoning have been proposed where predictions are done by taking a majority vote across multiple base models. In this work, we show that merely considering the majority vote in ensemble defenses is wasteful as it does not effectively utilize available information in the logits layers of the base models. Instead, we propose Run-Off Election (ROE), a novel aggregation method based on a two-round election across the base models: In the first round, models vote for their preferred class and then a second, Run-Off election is held between the top two classes in the first round. Based on this approach, we propose DPA+ROE and FA+ROE defense methods based on Deep Partition Aggregation (DPA) and Finite Aggregation (FA) approaches from prior work. We evaluate our methods on MNIST, CIFAR-10, and GTSRB and obtain improvements in certified accuracy by up to 3%-4%. Also, by applying ROE on a boosted version of DPA, we gain improvements around 12%-27% comparing to the current state-of-the-art, establishing a new state-of-the-art in (pointwise) certified robustness against data poisoning. In many cases, our approach outperforms the state-of-the-art, even when using 32 times less computational power.

摘要: 在数据中毒攻击中，对手试图通过添加、修改或删除训练数据中的样本来更改模型的预测。最近，已经提出了基于集成的方法来获得针对数据中毒的可证明防御，其中预测是通过在多个基础模型上获得多数票来完成的。在这项工作中，我们表明，仅仅在集成防御中考虑多数投票是浪费的，因为它没有有效地利用基本模型的Logits层中的可用信息。相反，我们提出了决选选举(ROE)，这是一种基于基础模型之间的两轮选举的新型聚合方法：在第一轮中，模型投票选择他们喜欢的类，然后在第一轮中前两个类之间举行第二次决选。在此基础上，提出了基于深度划分聚集(DPA)和有限聚集(FA)的DPA+ROE和FA+ROE防御方法。我们在MNIST、CIFAR-10和GTSRB上对我们的方法进行了评估，并在认证的准确性方面获得了高达3%-4%的改进。此外，通过在增强版本的DPA上应用ROE，与当前最先进的版本相比，我们获得了约12%-27%的改进，从而建立了针对数据中毒的(按点)经认证的新的最先进的健壮性。在许多情况下，我们的方法优于最先进的方法，即使在使用32倍的计算能力时也是如此。



## **4. Untargeted Near-collision Attacks in Biometric Recognition**

生物特征识别中的无目标近碰撞攻击 cs.CR

Addition of results and correction of typos

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2304.01580v2) [paper-pdf](http://arxiv.org/pdf/2304.01580v2)

**Authors**: Axel Durbet, Paul-Marie Grollemund, Kevin Thiry-Atighehchi

**Abstract**: A biometric recognition system can operate in two distinct modes, identification or verification. In the first mode, the system recognizes an individual by searching the enrolled templates of all the users for a match. In the second mode, the system validates a user's identity claim by comparing the fresh provided template with the enrolled template. The biometric transformation schemes usually produce binary templates that are better handled by cryptographic schemes, and the comparison is based on a distance that leaks information about the similarities between two biometric templates. Both the experimentally determined false match rate and false non-match rate through recognition threshold adjustment define the recognition accuracy, and hence the security of the system. To the best of our knowledge, few works provide a formal treatment of the security under minimum leakage of information, i.e., the binary outcome of a comparison with a threshold. In this paper, we rely on probabilistic modelling to quantify the security strength of binary templates. We investigate the influence of template size, database size and threshold on the probability of having a near-collision. We highlight several untargeted attacks on biometric systems considering naive and adaptive adversaries. Interestingly, these attacks can be launched both online and offline and, both in the identification mode and in the verification mode. We discuss the choice of parameters through the generic presented attacks.

摘要: 生物识别系统可以在两种截然不同的模式下工作，即识别或验证。在第一种模式中，系统通过在所有用户的注册模板中搜索匹配项来识别个人。在第二种模式中，系统通过将新提供的模板与注册的模板进行比较来验证用户的身份声明。生物特征转换方案通常产生由加密方案更好地处理的二进制模板，并且比较基于泄露关于两个生物特征模板之间的相似性的信息的距离。实验确定的误匹配率和通过调整识别阈值确定的误不匹配率都定义了识别精度，从而决定了系统的安全性。就我们所知，很少有文献在信息泄露最小的情况下提供安全的形式处理，即与阈值比较的二进制结果。在本文中，我们依赖于概率建模来量化二进制模板的安全强度。我们研究了模板大小、数据库大小和阈值对近碰撞概率的影响。我们重点介绍了几种针对生物识别系统的非定向攻击，考虑到了天真和自适应的对手。有趣的是，这些攻击既可以在线上也可以离线发起，也可以在识别模式和验证模式下发起。我们通过一般提出的攻击讨论参数的选择。



## **5. Distracting Downpour: Adversarial Weather Attacks for Motion Estimation**

分散注意力的倾盆大雨：运动估计的对抗性天气攻击 cs.CV

This work is a direct extension of our extended abstract from  arXiv:2210.11242

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.06716v1) [paper-pdf](http://arxiv.org/pdf/2305.06716v1)

**Authors**: Jenny Schmalfuss, Lukas Mehl, Andrés Bruhn

**Abstract**: Current adversarial attacks on motion estimation, or optical flow, optimize small per-pixel perturbations, which are unlikely to appear in the real world. In contrast, adverse weather conditions constitute a much more realistic threat scenario. Hence, in this work, we present a novel attack on motion estimation that exploits adversarially optimized particles to mimic weather effects like snowflakes, rain streaks or fog clouds. At the core of our attack framework is a differentiable particle rendering system that integrates particles (i) consistently over multiple time steps (ii) into the 3D space (iii) with a photo-realistic appearance. Through optimization, we obtain adversarial weather that significantly impacts the motion estimation. Surprisingly, methods that previously showed good robustness towards small per-pixel perturbations are particularly vulnerable to adversarial weather. At the same time, augmenting the training with non-optimized weather increases a method's robustness towards weather effects and improves generalizability at almost no additional cost.

摘要: 目前对运动估计或光流的敌意攻击，优化了每像素的小扰动，这在现实世界中不太可能出现。相比之下，不利的天气条件构成了更现实的威胁情景。因此，在这项工作中，我们提出了一种新颖的攻击运动估计的方法，该方法利用反向优化的粒子来模拟雪花、雨带或雾云等天气效果。在我们的攻击框架的核心是一个可区分的粒子渲染系统，它以照片般的外观将粒子(I)在多个时间步骤(Ii)一致地集成到3D空间(Iii)中。通过优化，得到对运动估计有显著影响的对抗性天气。令人惊讶的是，以前对每像素微小扰动表现出良好稳健性的方法特别容易受到恶劣天气的影响。同时，在不增加额外代价的情况下，用非优化的天气来增加训练，增加了方法对天气影响的鲁棒性，并提高了泛化能力。



## **6. Beyond the Model: Data Pre-processing Attack to Deep Learning Models in Android Apps**

超越模型：Android应用程序中对深度学习模型的数据预处理攻击 cs.CR

Accepted to AsiaCCS WorkShop on Secure and Trustworthy Deep Learning  Systems (SecTL 2023)

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.03963v2) [paper-pdf](http://arxiv.org/pdf/2305.03963v2)

**Authors**: Ye Sang, Yujin Huang, Shuo Huang, Helei Cui

**Abstract**: The increasing popularity of deep learning (DL) models and the advantages of computing, including low latency and bandwidth savings on smartphones, have led to the emergence of intelligent mobile applications, also known as DL apps, in recent years. However, this technological development has also given rise to several security concerns, including adversarial examples, model stealing, and data poisoning issues. Existing works on attacks and countermeasures for on-device DL models have primarily focused on the models themselves. However, scant attention has been paid to the impact of data processing disturbance on the model inference. This knowledge disparity highlights the need for additional research to fully comprehend and address security issues related to data processing for on-device models. In this paper, we introduce a data processing-based attacks against real-world DL apps. In particular, our attack could influence the performance and latency of the model without affecting the operation of a DL app. To demonstrate the effectiveness of our attack, we carry out an empirical study on 517 real-world DL apps collected from Google Play. Among 320 apps utilizing MLkit, we find that 81.56\% of them can be successfully attacked.   The results emphasize the importance of DL app developers being aware of and taking actions to secure on-device models from the perspective of data processing.

摘要: 近年来，深度学习模型的日益流行以及计算的优势，包括智能手机上的低延迟和带宽节省，导致了智能移动应用程序的出现，也被称为深度学习应用程序。然而，这种技术的发展也引起了一些安全问题，包括对抗性例子、模型窃取和数据中毒问题。现有的针对设备上DL模型的攻击和对策的研究主要集中在模型本身。然而，数据处理干扰对模型推理的影响还没有引起足够的重视。这种知识差距突出表明，需要进行更多的研究，以充分理解和解决与设备上模型的数据处理相关的安全问题。在本文中，我们介绍了一种基于数据处理的针对现实世界数字图书馆应用程序的攻击。特别是，我们的攻击可能会影响模型的性能和延迟，而不会影响DL应用的操作。为了证明我们攻击的有效性，我们对从Google Play收集的517个真实数字图书馆应用程序进行了实证研究。在320个使用MLkit的应用中，我们发现81.56%的应用可以被成功攻击。这些结果强调了数字图书馆应用程序开发人员从数据处理的角度意识到并采取行动保护设备上模型的重要性。



## **7. On the Robustness of Graph Neural Diffusion to Topology Perturbations**

关于图神经扩散对拓扑扰动的稳健性 cs.LG

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2209.07754v2) [paper-pdf](http://arxiv.org/pdf/2209.07754v2)

**Authors**: Yang Song, Qiyu Kang, Sijie Wang, Zhao Kai, Wee Peng Tay

**Abstract**: Neural diffusion on graphs is a novel class of graph neural networks that has attracted increasing attention recently. The capability of graph neural partial differential equations (PDEs) in addressing common hurdles of graph neural networks (GNNs), such as the problems of over-smoothing and bottlenecks, has been investigated but not their robustness to adversarial attacks. In this work, we explore the robustness properties of graph neural PDEs. We empirically demonstrate that graph neural PDEs are intrinsically more robust against topology perturbation as compared to other GNNs. We provide insights into this phenomenon by exploiting the stability of the heat semigroup under graph topology perturbations. We discuss various graph diffusion operators and relate them to existing graph neural PDEs. Furthermore, we propose a general graph neural PDE framework based on which a new class of robust GNNs can be defined. We verify that the new model achieves comparable state-of-the-art performance on several benchmark datasets.

摘要: 图上的神经扩散是一类新的图神经网络，近年来受到越来越多的关注。图神经偏微分方程组(PDE)在解决图神经网络(GNN)的常见障碍(如过光滑和瓶颈问题)方面的能力已被研究，但其对对手攻击的稳健性尚未得到研究。在这项工作中，我们研究了图神经偏微分方程的稳健性。我们的经验证明，与其他GNN相比，图神经PDE在本质上对拓扑扰动具有更强的鲁棒性。通过利用图的拓扑扰动下热半群的稳定性，我们提供了对这一现象的见解。我们讨论了各种图扩散算子，并将它们与现有的图神经偏微分方程联系起来。此外，我们还提出了一个通用的图神经偏微分方程框架，基于该框架可以定义一类新的健壮GNN。我们在几个基准数据集上验证了新模型取得了相当于最先进的性能。



## **8. Prevention of shoulder-surfing attacks using shifting condition using digraph substitution rules**

基于有向图替换规则的移位条件防止冲浪攻击 cs.CR

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.06549v1) [paper-pdf](http://arxiv.org/pdf/2305.06549v1)

**Authors**: Amanul Islam, Fazidah Othman, Nazmus Sakib, Hafiz Md. Hasan Babu

**Abstract**: Graphical passwords are implemented as an alternative scheme to replace alphanumeric passwords to help users to memorize their password. However, most of the graphical password systems are vulnerable to shoulder-surfing attack due to the usage of the visual interface. In this research, a method that uses shifting condition with digraph substitution rules is proposed to address shoulder-surfing attack problem. The proposed algorithm uses both password images and decoy images throughout the user authentication procedure to confuse adversaries from obtaining the password images via direct observation or watching from a recorded session. The pass-images generated by this suggested algorithm are random and can only be generated if the algorithm is fully understood. As a result, adversaries will have no clue to obtain the right password images to log in. A user study was undertaken to assess the proposed method's effectiveness to avoid shoulder-surfing attacks. The results of the user study indicate that the proposed approach can withstand shoulder-surfing attacks (both direct observation and video recording method).The proposed method was tested and the results showed that it is able to resist shoulder-surfing and frequency of occurrence analysis attacks. Moreover, the experience gained in this research can be pervaded the gap on the realm of knowledge of the graphical password.

摘要: 图形密码作为替代字母数字密码的替代方案来实施，以帮助用户记住他们的密码。然而，由于可视化界面的使用，大多数图形化密码系统都容易受到肩部冲浪攻击。针对肩部冲浪攻击问题，提出了一种基于有向图替换规则的移位条件攻击方法。该算法在用户认证过程中同时使用口令图像和诱骗图像，以迷惑攻击者通过直接观察或从记录的会话中观看来获得口令图像。该算法生成的通道图像是随机的，只有在充分理解该算法的情况下才能生成。因此，攻击者将没有任何线索来获取正确的密码图像来登录。进行了一项用户研究，以评估所提出的方法在避免肩部冲浪攻击方面的有效性。用户研究结果表明，该方法能够抵抗直接观察法和录像法的肩部冲浪攻击，并对该方法进行了测试，结果表明该方法能够抵抗肩部冲浪和频度分析攻击。此外，在本研究中获得的经验可以填补图形密码知识领域的空白。



## **9. Inter-frame Accelerate Attack against Video Interpolation Models**

针对视频插补模型的帧间加速攻击 cs.CV

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.06540v1) [paper-pdf](http://arxiv.org/pdf/2305.06540v1)

**Authors**: Junpei Liao, Zhikai Chen, Liang Yi, Wenyuan Yang, Baoyuan Wu, Xiaochun Cao

**Abstract**: Deep learning based video frame interpolation (VIF) method, aiming to synthesis the intermediate frames to enhance video quality, have been highly developed in the past few years. This paper investigates the adversarial robustness of VIF models. We apply adversarial attacks to VIF models and find that the VIF models are very vulnerable to adversarial examples. To improve attack efficiency, we suggest to make full use of the property of video frame interpolation task. The intuition is that the gap between adjacent frames would be small, leading to the corresponding adversarial perturbations being similar as well. Then we propose a novel attack method named Inter-frame Accelerate Attack (IAA) that initializes the perturbation as the perturbation for the previous adjacent frame and reduces the number of attack iterations. It is shown that our method can improve attack efficiency greatly while achieving comparable attack performance with traditional methods. Besides, we also extend our method to video recognition models which are higher level vision tasks and achieves great attack efficiency.

摘要: 基于深度学习的视频帧内插方法(VIF)旨在合成中间帧以提高视频质量，在过去的几年中得到了很大的发展。本文研究了VIF模型的对抗稳健性。我们将对抗性攻击应用于VIF模型，发现VIF模型非常容易受到对抗性例子的攻击。为了提高攻击效率，我们建议充分利用视频帧内插任务的特性。直觉是，相邻帧之间的间隙会很小，导致相应的对抗性扰动也是相似的。然后，我们提出了一种新的攻击方法--帧间加速攻击(IAA)，该方法将扰动初始化为对前一相邻帧的扰动，并减少了攻击迭代的次数。实验结果表明，该方法在取得与传统方法相当的攻击性能的同时，大大提高了攻击效率。此外，我们还将我们的方法扩展到视频识别模型，这些模型是较高级别的视觉任务，具有很高的攻击效率。



## **10. Improving Adversarial Robustness via Joint Classification and Multiple Explicit Detection Classes**

联合分类和多个显式检测类提高敌方鲁棒性 cs.CV

20 pages, 6 figures

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2210.14410v2) [paper-pdf](http://arxiv.org/pdf/2210.14410v2)

**Authors**: Sina Baharlouei, Fatemeh Sheikholeslami, Meisam Razaviyayn, Zico Kolter

**Abstract**: This work concerns the development of deep networks that are certifiably robust to adversarial attacks. Joint robust classification-detection was recently introduced as a certified defense mechanism, where adversarial examples are either correctly classified or assigned to the "abstain" class. In this work, we show that such a provable framework can benefit by extension to networks with multiple explicit abstain classes, where the adversarial examples are adaptively assigned to those. We show that naively adding multiple abstain classes can lead to "model degeneracy", then we propose a regularization approach and a training method to counter this degeneracy by promoting full use of the multiple abstain classes. Our experiments demonstrate that the proposed approach consistently achieves favorable standard vs. robust verified accuracy tradeoffs, outperforming state-of-the-art algorithms for various choices of number of abstain classes.

摘要: 这项工作涉及到深度网络的发展，这些网络对对手攻击具有可证明的健壮性。联合稳健分类-检测是最近引入的一种认证防御机制，在这种机制中，对抗性例子要么被正确分类，要么被分配到“弃权”类别。在这项工作中，我们表明这样一个可证明的框架可以通过扩展到具有多个显式弃权类的网络而受益，其中对抗性示例被自适应地分配给那些显式弃权类。我们证明了简单地添加多个弃权类会导致“模型退化”，然后我们提出了一种正则化方法和一种训练方法，通过促进多个弃权类的充分利用来克服这种退化。我们的实验表明，该方法一致地达到了良好的标准和健壮的验证精度折衷，在不同数量的弃权类的选择上优于最新的算法。



## **11. Towards Adversarial-Resilient Deep Neural Networks for False Data Injection Attack Detection in Power Grids**

用于电网虚假数据注入攻击检测的对抗性深度神经网络 cs.CR

This paper has been accepted by the the 32nd International Conference  on Computer Communications and Networks (ICCCN 2023)

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2102.09057v2) [paper-pdf](http://arxiv.org/pdf/2102.09057v2)

**Authors**: Jiangnan Li, Yingyuan Yang, Jinyuan Stella Sun, Kevin Tomsovic, Hairong Qi

**Abstract**: False data injection attacks (FDIAs) pose a significant security threat to power system state estimation. To detect such attacks, recent studies have proposed machine learning (ML) techniques, particularly deep neural networks (DNNs). However, most of these methods fail to account for the risk posed by adversarial measurements, which can compromise the reliability of DNNs in various ML applications. In this paper, we present a DNN-based FDIA detection approach that is resilient to adversarial attacks. We first analyze several adversarial defense mechanisms used in computer vision and show their inherent limitations in FDIA detection. We then propose an adversarial-resilient DNN detection framework for FDIA that incorporates random input padding in both the training and inference phases. Our simulations, based on an IEEE standard power system, demonstrate that this framework significantly reduces the effectiveness of adversarial attacks while having a negligible impact on the DNNs' detection performance.

摘要: 虚假数据注入攻击(FDIA)对电力系统状态估计造成了严重的安全威胁。为了检测此类攻击，最近的研究提出了机器学习(ML)技术，特别是深度神经网络(DNN)。然而，这些方法中的大多数都没有考虑到对抗性测量所带来的风险，这可能会损害DNN在各种ML应用中的可靠性。在本文中，我们提出了一种基于DNN的对敌方攻击具有弹性的FDIA检测方法。我们首先分析了计算机视觉中使用的几种对抗性防御机制，并指出了它们在FDIA检测中的固有局限性。然后，我们提出了一种用于FDIA的对抗性DNN检测框架，该框架在训练和推理阶段都加入了随机输入填充。基于IEEE标准电力系统的仿真表明，该框架显著降低了对抗性攻击的有效性，而对DNN的检测性能影响可以忽略不计。



## **12. Invisible Backdoor Attack with Dynamic Triggers against Person Re-identification**

利用动态触发器对个人重新身份进行隐形后门攻击 cs.CV

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2211.10933v2) [paper-pdf](http://arxiv.org/pdf/2211.10933v2)

**Authors**: Wenli Sun, Xinyang Jiang, Shuguang Dou, Dongsheng Li, Duoqian Miao, Cheng Deng, Cairong Zhao

**Abstract**: In recent years, person Re-identification (ReID) has rapidly progressed with wide real-world applications, but also poses significant risks of adversarial attacks. In this paper, we focus on the backdoor attack on deep ReID models. Existing backdoor attack methods follow an all-to-one or all-to-all attack scenario, where all the target classes in the test set have already been seen in the training set. However, ReID is a much more complex fine-grained open-set recognition problem, where the identities in the test set are not contained in the training set. Thus, previous backdoor attack methods for classification are not applicable for ReID. To ameliorate this issue, we propose a novel backdoor attack on deep ReID under a new all-to-unknown scenario, called Dynamic Triggers Invisible Backdoor Attack (DT-IBA). Instead of learning fixed triggers for the target classes from the training set, DT-IBA can dynamically generate new triggers for any unknown identities. Specifically, an identity hashing network is proposed to first extract target identity information from a reference image, which is then injected into the benign images by image steganography. We extensively validate the effectiveness and stealthiness of the proposed attack on benchmark datasets, and evaluate the effectiveness of several defense methods against our attack.

摘要: 近年来，身份识别技术发展迅速，在实际应用中得到了广泛的应用，但同时也带来了巨大的对抗性攻击风险。本文主要研究对深度Reid模型的后门攻击。现有的后门攻击方法遵循All-to-One或All-to-All攻击方案，其中测试集中的所有目标类都已在训练集中看到。然而，REID是一个更复杂的细粒度开集识别问题，其中测试集中的身份不包含在训练集中。因此，以前用于分类的后门攻击方法不适用于REID。为了改善这一问题，我们提出了一种新的全未知场景下对深度Reid的后门攻击，称为动态触发器不可见后门攻击(DT-IBA)。DT-IBA不需要从训练集中学习目标类的固定触发器，而是可以为任何未知身份动态生成新的触发器。具体地说，提出了一种身份散列网络，首先从参考图像中提取目标身份信息，然后通过图像隐写将这些身份信息注入到良性图像中。我们在基准数据集上广泛验证了提出的攻击的有效性和隐蔽性，并评估了几种防御方法对我们的攻击的有效性。



## **13. The Robustness of Computer Vision Models against Common Corruptions: a Survey**

计算机视觉模型对常见腐败的稳健性研究综述 cs.CV

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2305.06024v1) [paper-pdf](http://arxiv.org/pdf/2305.06024v1)

**Authors**: Shunxin Wang, Raymond Veldhuis, Nicola Strisciuglio

**Abstract**: The performance of computer vision models is susceptible to unexpected changes in input images when deployed in real scenarios. These changes are referred to as common corruptions. While they can hinder the applicability of computer vision models in real-world scenarios, they are not always considered as a testbed for model generalization and robustness. In this survey, we present a comprehensive and systematic overview of methods that improve corruption robustness of computer vision models. Unlike existing surveys that focus on adversarial attacks and label noise, we cover extensively the study of robustness to common corruptions that can occur when deploying computer vision models to work in practical applications. We describe different types of image corruption and provide the definition of corruption robustness. We then introduce relevant evaluation metrics and benchmark datasets. We categorize methods into four groups. We also cover indirect methods that show improvements in generalization and may improve corruption robustness as a byproduct. We report benchmark results collected from the literature and find that they are not evaluated in a unified manner, making it difficult to compare and analyze. We thus built a unified benchmark framework to obtain directly comparable results on benchmark datasets. Furthermore, we evaluate relevant backbone networks pre-trained on ImageNet using our framework, providing an overview of the base corruption robustness of existing models to help choose appropriate backbones for computer vision tasks. We identify that developing methods to handle a wide range of corruptions and efficiently learn with limited data and computational resources is crucial for future development. Additionally, we highlight the need for further investigation into the relationship among corruption robustness, OOD generalization, and shortcut learning.

摘要: 当计算机视觉模型部署在真实场景中时，其性能很容易受到输入图像中意外变化的影响。这些变化被称为常见的腐败。虽然它们会阻碍计算机视觉模型在现实世界场景中的适用性，但它们并不总是被视为模型泛化和健壮性的试验台。在这次调查中，我们全面和系统地概述了提高计算机视觉模型的腐败稳健性的方法。与专注于对抗性攻击和标签噪声的现有调查不同，我们广泛涵盖了对在实际应用中部署计算机视觉模型时可能发生的常见腐败的稳健性研究。我们描述了不同类型的图像损坏，并给出了损坏稳健性的定义。然后我们介绍了相关的评估指标和基准数据集。我们将方法分为四类。我们还介绍了间接方法，这些方法显示了泛化方面的改进，并可能作为副产品提高腐败健壮性。我们报告了从文献中收集的基准结果，发现它们没有以统一的方式进行评估，这使得比较和分析变得困难。因此，我们建立了一个统一的基准框架，以获得基准数据集的直接可比结果。此外，我们使用我们的框架评估了在ImageNet上预先训练的相关骨干网络，提供了现有模型的基本腐败稳健性的概述，以帮助选择合适的骨干网络来执行计算机视觉任务。我们认识到，开发方法来处理广泛的腐败问题，并利用有限的数据和计算资源有效地学习，对未来的发展至关重要。此外，我们强调有必要进一步调查腐败稳健性、面向对象设计泛化和快捷学习之间的关系。



## **14. Robust multi-agent coordination via evolutionary generation of auxiliary adversarial attackers**

通过进化生成辅助对抗性攻击者实现健壮的多智能体协作 cs.MA

In: Proceedings of the 37th AAAI Conference on Artificial  Intelligence (AAAI'23), 2023

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2305.05909v1) [paper-pdf](http://arxiv.org/pdf/2305.05909v1)

**Authors**: Lei Yuan, Zi-Qian Zhang, Ke Xue, Hao Yin, Feng Chen, Cong Guan, Li-He Li, Chao Qian, Yang Yu

**Abstract**: Cooperative multi-agent reinforcement learning (CMARL) has shown to be promising for many real-world applications. Previous works mainly focus on improving coordination ability via solving MARL-specific challenges (e.g., non-stationarity, credit assignment, scalability), but ignore the policy perturbation issue when testing in a different environment. This issue hasn't been considered in problem formulation or efficient algorithm design. To address this issue, we firstly model the problem as a limited policy adversary Dec-POMDP (LPA-Dec-POMDP), where some coordinators from a team might accidentally and unpredictably encounter a limited number of malicious action attacks, but the regular coordinators still strive for the intended goal. Then, we propose Robust Multi-Agent Coordination via Evolutionary Generation of Auxiliary Adversarial Attackers (ROMANCE), which enables the trained policy to encounter diversified and strong auxiliary adversarial attacks during training, thus achieving high robustness under various policy perturbations. Concretely, to avoid the ego-system overfitting to a specific attacker, we maintain a set of attackers, which is optimized to guarantee the attackers high attacking quality and behavior diversity. The goal of quality is to minimize the ego-system coordination effect, and a novel diversity regularizer based on sparse action is applied to diversify the behaviors among attackers. The ego-system is then paired with a population of attackers selected from the maintained attacker set, and alternately trained against the constantly evolving attackers. Extensive experiments on multiple scenarios from SMAC indicate our ROMANCE provides comparable or better robustness and generalization ability than other baselines.

摘要: 协作多智能体强化学习(CMARL)已被证明在许多实际应用中具有广阔的应用前景。以往的工作主要集中在通过解决MAIL特有的挑战(如非平稳性、信用分配、可扩展性)来提高协调能力，而忽略了在不同环境中测试时的策略扰动问题。这个问题在问题描述和有效的算法设计中都没有考虑到。为了解决这个问题，我们首先将问题建模为有限策略对手DEC-POMDP(LPA-DEC-POMDP)，其中团队中的一些协调者可能意外地和不可预测地遇到有限数量的恶意行为攻击，但常规协调者仍然努力实现预期的目标。在此基础上，提出了基于辅助对抗性攻击进化生成的稳健多智能体协调算法(ROMANCE)，使训练后的策略在训练过程中能够遇到多样化且强的辅助对抗性攻击，从而在各种策略扰动下具有较高的鲁棒性。具体地说，为了避免自我系统对特定攻击者的过度匹配，我们维护了一组攻击者，并对其进行了优化，以保证攻击者的高攻击质量和行为多样性。质量的目标是最小化自我-系统协调效应，并采用一种新的基于稀疏动作的多样性正则化算法来使攻击者的行为多样化。然后，自我系统与从维护的攻击者集合中选择的一群攻击者配对，并交替地针对不断演变的攻击者进行训练。在SMAC的多个场景上的大量实验表明，我们的Romance提供了与其他基线相当或更好的健壮性和泛化能力。



## **15. RNNS: Representation Nearest Neighbor Search Black-Box Attack on Code Models**

RNNS：代码模型上的表示最近邻搜索黑盒攻击 cs.CR

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2305.05896v1) [paper-pdf](http://arxiv.org/pdf/2305.05896v1)

**Authors**: Jie Zhang, Wei Ma, Qiang Hu, Xiaofei Xie, Yves Le Traon, Yang Liu

**Abstract**: Pre-trained code models are mainly evaluated using the in-distribution test data. The robustness of models, i.e., the ability to handle hard unseen data, still lacks evaluation. In this paper, we propose a novel search-based black-box adversarial attack guided by model behaviours for pre-trained programming language models, named Representation Nearest Neighbor Search(RNNS), to evaluate the robustness of Pre-trained PL models. Unlike other black-box adversarial attacks, RNNS uses the model-change signal to guide the search in the space of the variable names collected from real-world projects. Specifically, RNNS contains two main steps, 1) indicate which variable (attack position location) we should attack based on model uncertainty, and 2) search which adversarial tokens we should use for variable renaming according to the model behaviour observations. We evaluate RNNS on 6 code tasks (e.g., clone detection), 3 programming languages (Java, Python, and C), and 3 pre-trained code models: CodeBERT, GraphCodeBERT, and CodeT5. The results demonstrate that RNNS outperforms the state-of-the-art black-box attacking methods (MHM and ALERT) in terms of attack success rate (ASR) and query times (QT). The perturbation of generated adversarial examples from RNNS is smaller than the baselines with respect to the number of replaced variables and the variable length change. Our experiments also show that RNNS is efficient in attacking the defended models and is useful for adversarial training.

摘要: 预先训练的代码模型主要使用分发内测试数据进行评估。模型的稳健性，即处理硬的看不见的数据的能力，仍然缺乏评估。针对预先训练的程序设计语言模型，提出了一种以模型行为为导向的基于搜索的黑盒对抗攻击方法--表示最近邻搜索算法(RNNS)，以评估预先训练的程序设计语言模型的健壮性。与其他黑盒对抗性攻击不同，RNNS使用模型更改信号来指导在从现实世界项目中收集的变量名称空间中的搜索。具体地说，RNNS包含两个主要步骤，1)根据模型的不确定性指示我们应该攻击哪个变量(攻击位置)，2)根据模型行为观察寻找应该使用哪些敌意标记进行变量重命名。我们在6个代码任务(例如克隆检测)、3种编程语言(Java、Python和C)以及3种预先训练的代码模型上对RNNS进行了评估：CodeBERT、GraphCodeBERT和CodeT5。结果表明，RNNS在攻击成功率(ASR)和查询次数(Qt)方面均优于目前最先进的黑盒攻击方法(MHM和ALERT)。从RNNS生成的对抗性样本在替换变量的数量和可变长度变化方面的扰动小于基线。我们的实验还表明，RNNS在攻击防御模型方面是有效的，并且对于对抗性训练是有用的。



## **16. Quantization Aware Attack: Enhancing the Transferability of Adversarial Attacks across Target Models with Different Quantization Bitwidths**

量化感知攻击：提高敌意攻击在不同量化位宽的目标模型上的可转移性 cs.CR

9 pages

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2305.05875v1) [paper-pdf](http://arxiv.org/pdf/2305.05875v1)

**Authors**: Yulong Yang, Chenhao Lin, Qian Li, Chao Shen, Dawei Zhou, Nannan Wang, Tongliang Liu

**Abstract**: Quantized Neural Networks (QNNs) receive increasing attention in resource-constrained scenarios because of their excellent generalization abilities, but their robustness under realistic black-box adversarial attacks has not been deeply studied, in which the adversary requires to improve the attack capability across target models with unknown quantization bitwidths. One major challenge is that adversarial examples transfer poorly against QNNs with unknown bitwidths because of the quantization shift and gradient misalignment issues. This paper proposes the Quantization Aware Attack to enhance the attack transferability by making the substitute model ``aware of'' the target of attacking models with multiple bitwidths. Specifically, we design a training objective with multiple bitwidths to align the gradient of the substitute model with the target model with different bitwidths and thus mitigate the negative effect of the above two issues. We conduct comprehensive evaluations by performing multiple transfer-based attacks on standard models and defense models with different architectures and quantization bitwidths. Experimental results show that QAA significantly improves the adversarial transferability of the state-of-the-art attacks by 3.4%-20.9% against normally trained models and 3.7%-13.4% against adversarially trained models on average.

摘要: 量化神经网络(QNN)以其良好的泛化能力在资源受限的场景中受到越来越多的关注，但其在现实黑盒攻击下的健壮性还没有得到深入的研究，在现实的黑盒攻击中，对手要求提高对未知量化比特目标模型的攻击能力。一个主要的挑战是，由于量化漂移和梯度对齐问题，对抗性例子在比特宽度未知的QNN上的传输效果很差。为了提高攻击的可转移性，提出了量化感知攻击，通过使替换模型“感知”到多个比特攻击模型的目标。具体地说，我们设计了一个多位宽的训练目标，将替换模型的梯度与不同位宽的目标模型对齐，从而缓解了上述两个问题的负面影响。通过对具有不同体系结构和量化位宽的标准模型和防御模型进行多次基于传输的攻击，进行综合评估。实验结果表明，QAA显著提高了最新攻击的对抗性，对正常训练的模型平均提高了3.4%-20.9%，对对抗性训练的模型平均提高了3.7%-13.4%。



## **17. VSMask: Defending Against Voice Synthesis Attack via Real-Time Predictive Perturbation**

VSMAsk：利用实时预测扰动防御语音合成攻击 cs.SD

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05736v1) [paper-pdf](http://arxiv.org/pdf/2305.05736v1)

**Authors**: Yuanda Wang, Hanqing Guo, Guangjing Wang, Bocheng Chen, Qiben Yan

**Abstract**: Deep learning based voice synthesis technology generates artificial human-like speeches, which has been used in deepfakes or identity theft attacks. Existing defense mechanisms inject subtle adversarial perturbations into the raw speech audios to mislead the voice synthesis models. However, optimizing the adversarial perturbation not only consumes substantial computation time, but it also requires the availability of entire speech. Therefore, they are not suitable for protecting live speech streams, such as voice messages or online meetings. In this paper, we propose VSMask, a real-time protection mechanism against voice synthesis attacks. Different from offline protection schemes, VSMask leverages a predictive neural network to forecast the most effective perturbation for the upcoming streaming speech. VSMask introduces a universal perturbation tailored for arbitrary speech input to shield a real-time speech in its entirety. To minimize the audio distortion within the protected speech, we implement a weight-based perturbation constraint to reduce the perceptibility of the added perturbation. We comprehensively evaluate VSMask protection performance under different scenarios. The experimental results indicate that VSMask can effectively defend against 3 popular voice synthesis models. None of the synthetic voice could deceive the speaker verification models or human ears with VSMask protection. In a physical world experiment, we demonstrate that VSMask successfully safeguards the real-time speech by injecting the perturbation over the air.

摘要: 基于深度学习的语音合成技术生成的人工语音已被用于深度假冒或身份盗窃攻击。现有的防御机制在原始语音音频中注入微妙的对抗性扰动，以误导语音合成模型。然而，优化对抗性扰动不仅需要消耗大量的计算时间，而且还需要整个语音的可用性。因此，它们不适合保护实时语音流，例如语音消息或在线会议。本文提出了一种针对语音合成攻击的实时防护机制VSMASK。与离线保护方案不同，VSMAsk利用预测神经网络来预测即将到来的流传输语音的最有效扰动。VSMask引入了一种为任意语音输入量身定做的通用扰动，以完整地屏蔽实时语音。为了最大限度地减少受保护语音中的音频失真，我们实现了基于权重的扰动约束来降低附加扰动的可感知性。我们综合评估了不同场景下的VSMASK保护性能。实验结果表明，VSMASK能够有效防御3种流行的语音合成模型。任何合成语音都无法欺骗说话人验证模型或具有VSMASK保护的人耳。在物理世界的实验中，我们演示了VSMAsk通过在空中注入扰动来成功地保护实时语音。



## **18. Using Anomaly Detection to Detect Poisoning Attacks in Federated Learning Applications**

在联合学习应用中使用异常检测来检测中毒攻击 cs.LG

We will updated this article soon

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2207.08486v2) [paper-pdf](http://arxiv.org/pdf/2207.08486v2)

**Authors**: Ali Raza, Shujun Li, Kim-Phuc Tran, Ludovic Koehl

**Abstract**: Adversarial attacks such as poisoning attacks have attracted the attention of many machine learning researchers. Traditionally, poisoning attacks attempt to inject adversarial training data in order to manipulate the trained model. In federated learning (FL), data poisoning attacks can be generalized to model poisoning attacks, which cannot be detected by simpler methods due to the lack of access to local training data by the detector. State-of-the-art poisoning attack detection methods for FL have various weaknesses, e.g., the number of attackers has to be known or not high enough, working with i.i.d. data only, and high computational complexity. To overcome above weaknesses, we propose a novel framework for detecting poisoning attacks in FL, which employs a reference model based on a public dataset and an auditor model to detect malicious updates. We implemented a detector based on the proposed framework and using a one-class support vector machine (OC-SVM), which reaches the lowest possible computational complexity O(K) where K is the number of clients. We evaluated our detector's performance against state-of-the-art (SOTA) poisoning attacks for two typical applications of FL: electrocardiograph (ECG) classification and human activity recognition (HAR). Our experimental results validated the performance of our detector over other SOTA detection methods.

摘要: 中毒攻击等对抗性攻击引起了许多机器学习研究人员的关注。传统上，中毒攻击试图注入对抗性的训练数据，以操纵训练的模型。在联邦学习中，数据中毒攻击可以被概括为模型中毒攻击，但由于检测器无法访问本地训练数据，因此无法用更简单的方法检测到中毒攻击。目前针对FL的中毒攻击检测方法有很多缺点，例如，攻击者的数量必须已知或不够高，与I.I.D.配合使用。仅限数据，且计算复杂性高。为了克服上述缺陷，我们提出了一种新的FL中毒攻击检测框架，该框架使用基于公共数据集的参考模型和审计者模型来检测恶意更新。我们基于提出的框架实现了一个检测器，并使用了单类支持向量机(OC-SVM)，它达到了最低的计算复杂度O(K)，其中K是客户端的数量。我们针对FL的两个典型应用：心电图分类和人类活动识别(HAR)，评估了我们的检测器对最先进的(SOTA)中毒攻击的性能。我们的实验结果验证了我们的检测器相对于其他SOTA检测方法的性能。



## **19. Improving Adversarial Transferability via Intermediate-level Perturbation Decay**

通过中层扰动衰减提高对手的可转换性 cs.LG

Revision of ICML '23 submission for better clarity

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2304.13410v2) [paper-pdf](http://arxiv.org/pdf/2304.13410v2)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: Intermediate-level attacks that attempt to perturb feature representations following an adversarial direction drastically have shown favorable performance in crafting transferable adversarial examples. Existing methods in this category are normally formulated with two separate stages, where a directional guide is required to be determined at first and the scalar projection of the intermediate-level perturbation onto the directional guide is enlarged thereafter. The obtained perturbation deviates from the guide inevitably in the feature space, and it is revealed in this paper that such a deviation may lead to sub-optimal attack. To address this issue, we develop a novel intermediate-level method that crafts adversarial examples within a single stage of optimization. In particular, the proposed method, named intermediate-level perturbation decay (ILPD), encourages the intermediate-level perturbation to be in an effective adversarial direction and to possess a great magnitude simultaneously. In-depth discussion verifies the effectiveness of our method. Experimental results show that it outperforms state-of-the-arts by large margins in attacking various victim models on ImageNet (+10.07% on average) and CIFAR-10 (+3.88% on average). Our code is at https://github.com/qizhangli/ILPD-attack.

摘要: 中级攻击试图按照对抗性方向彻底扰乱特征表示，在制作可转移的对抗性示例方面表现出了良好的性能。现有的这类方法通常分为两个不同的阶段，首先需要确定一个方向导轨，然后放大中层摄动在该方向导轨上的标量投影。所得到的扰动在特征空间中不可避免地偏离了导引，本文揭示了这种偏离可能导致次优攻击。为了解决这个问题，我们开发了一种新的中级方法，该方法在单个优化阶段内创建对抗性示例。特别是，所提出的方法，称为中层扰动衰变(ILPD)，它鼓励中层扰动朝着有效的对抗性方向发展，同时具有较大的幅度。通过深入讨论，验证了该方法的有效性。实验结果表明，在ImageNet(平均+10.07%)和CIFAR-10(平均+3.88%)上攻击各种受害者模型时，该算法的性能明显优于最新的攻击模型。我们的代码在https://github.com/qizhangli/ILPD-attack.



## **20. Turning Privacy-preserving Mechanisms against Federated Learning**

将隐私保护机制转向联合学习 cs.LG

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05355v1) [paper-pdf](http://arxiv.org/pdf/2305.05355v1)

**Authors**: Marco Arazzi, Mauro Conti, Antonino Nocera, Stjepan Picek

**Abstract**: Recently, researchers have successfully employed Graph Neural Networks (GNNs) to build enhanced recommender systems due to their capability to learn patterns from the interaction between involved entities. In addition, previous studies have investigated federated learning as the main solution to enable a native privacy-preserving mechanism for the construction of global GNN models without collecting sensitive data into a single computation unit. Still, privacy issues may arise as the analysis of local model updates produced by the federated clients can return information related to sensitive local data. For this reason, experts proposed solutions that combine federated learning with Differential Privacy strategies and community-driven approaches, which involve combining data from neighbor clients to make the individual local updates less dependent on local sensitive data. In this paper, we identify a crucial security flaw in such a configuration, and we design an attack capable of deceiving state-of-the-art defenses for federated learning. The proposed attack includes two operating modes, the first one focusing on convergence inhibition (Adversarial Mode), and the second one aiming at building a deceptive rating injection on the global federated model (Backdoor Mode). The experimental results show the effectiveness of our attack in both its modes, returning on average 60% performance detriment in all the tests on Adversarial Mode and fully effective backdoors in 93% of cases for the tests performed on Backdoor Mode.

摘要: 最近，研究人员已经成功地使用图神经网络(GNN)来构建增强的推荐系统，这是因为它们能够从相关实体之间的交互中学习模式。此外，以前的研究已经将联合学习作为主要解决方案，以实现在不将敏感数据收集到单个计算单元的情况下构建全局GNN模型的本地隐私保护机制。尽管如此，隐私问题可能会出现，因为对联合客户端生成的本地模型更新的分析可能会返回与敏感本地数据相关的信息。为此，专家们提出了将联合学习与差异隐私策略和社区驱动方法相结合的解决方案，其中包括合并来自邻居客户端的数据，以减少个别本地更新对本地敏感数据的依赖。在本文中，我们确定了这种配置中的一个关键安全漏洞，并设计了一个能够欺骗联邦学习的最新防御的攻击。提出的攻击包括两种工作模式，第一种集中在收敛抑制(对抗性模式)，第二种旨在在全球联邦模型上建立欺骗性评级注入(后门模式)。实验结果表明，我们的攻击在两种模式下都是有效的，在对抗性模式下的所有测试中平均返回60%的性能损失，在后门模式上执行的测试中，93%的情况下完全有效的后门程序。



## **21. Data Protection and Security Issues With Network Error Logging**

网络错误记录的数据保护和安全问题 cs.CR

Accepted for SECRYPT'23

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05343v1) [paper-pdf](http://arxiv.org/pdf/2305.05343v1)

**Authors**: Libor Polčák, Kamil Jeřábek

**Abstract**: Network Error Logging helps web server operators detect operational problems in real-time to provide fast and reliable services. This paper analyses Network Error Logging from two angles. Firstly, this paper overviews Network Error Logging from the data protection view. The ePrivacy Directive requires consent for non-essential access to the end devices. Nevertheless, the Network Error Logging design does not allow limiting the tracking to consenting users. Other issues lay in GDPR requirements for transparency and the obligations in the contract between controllers and processors of personal data. Secondly, this paper explains Network Error Logging exploitations to deploy long-time trackers to the victim devices. Even though users should be able to disable Network Error Logging, it is not clear how to do so. Web server operators can mitigate the attack by configuring servers to preventively remove policies that adversaries might have added.

摘要: 网络错误记录帮助Web服务器操作员实时检测运行问题，以提供快速可靠的服务。本文从两个角度对网络错误记录进行了分析。首先，本文从数据保护的角度对网络错误记录进行了综述。电子隐私指令要求对终端设备进行非必要访问的同意。然而，网络错误记录设计不允许将跟踪限制到同意的用户。其他问题包括GDPR对透明度的要求以及个人数据管制员和处理者之间合同中的义务。其次，本文解释了利用网络错误记录漏洞将长期跟踪器部署到受攻击设备。尽管用户应该能够禁用网络错误记录，但不清楚如何这样做。Web服务器运营商可以通过配置服务器以预防性地删除攻击者可能添加的策略来缓解攻击。



## **22. Attack Named Entity Recognition by Entity Boundary Interference**

利用实体边界干扰攻击命名实体识别 cs.CL

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05253v1) [paper-pdf](http://arxiv.org/pdf/2305.05253v1)

**Authors**: Yifei Yang, Hongqiu Wu, Hai Zhao

**Abstract**: Named Entity Recognition (NER) is a cornerstone NLP task while its robustness has been given little attention. This paper rethinks the principles of NER attacks derived from sentence classification, as they can easily violate the label consistency between the original and adversarial NER examples. This is due to the fine-grained nature of NER, as even minor word changes in the sentence can result in the emergence or mutation of any entities, resulting in invalid adversarial examples. To this end, we propose a novel one-word modification NER attack based on a key insight, NER models are always vulnerable to the boundary position of an entity to make their decision. We thus strategically insert a new boundary into the sentence and trigger the Entity Boundary Interference that the victim model makes the wrong prediction either on this boundary word or on other words in the sentence. We call this attack Virtual Boundary Attack (ViBA), which is shown to be remarkably effective when attacking both English and Chinese models with a 70%-90% attack success rate on state-of-the-art language models (e.g. RoBERTa, DeBERTa) and also significantly faster than previous methods.

摘要: 命名实体识别(NER)是一项基础性的自然语言处理任务，但其健壮性却鲜有人关注。本文重新思考了基于句子分类的NER攻击的原理，因为它们很容易破坏原始例子和对抗性例子之间的标签一致性。这是由于NER的细粒度性质，因为即使句子中的微小单词变化也可能导致任何实体的出现或突变，从而导致无效的对抗性例子。为此，我们提出了一种新颖的基于关键洞察力的单字修正NER攻击，NER模型总是容易受到实体边界位置的影响而做出决策。因此，我们策略性地在句子中插入新的边界，并触发实体边界干扰，即受害者模型对句子中的该边界词或其他词做出错误预测。我们将这种攻击称为虚拟边界攻击(VIBA)，该攻击在对最先进的语言模型(如Roberta，DeBERTa)的攻击成功率为70%-90%的情况下，对英文和中文模型的攻击都非常有效，而且攻击速度也明显快于以前的方法。



## **23. Generating Phishing Attacks using ChatGPT**

使用ChatGPT生成网络钓鱼攻击 cs.CR

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05133v1) [paper-pdf](http://arxiv.org/pdf/2305.05133v1)

**Authors**: Sayak Saha Roy, Krishna Vamsi Naragam, Shirin Nilizadeh

**Abstract**: The ability of ChatGPT to generate human-like responses and understand context has made it a popular tool for conversational agents, content creation, data analysis, and research and innovation. However, its effectiveness and ease of accessibility makes it a prime target for generating malicious content, such as phishing attacks, that can put users at risk. In this work, we identify several malicious prompts that can be provided to ChatGPT to generate functional phishing websites. Through an iterative approach, we find that these phishing websites can be made to imitate popular brands and emulate several evasive tactics that have been known to avoid detection by anti-phishing entities. These attacks can be generated using vanilla ChatGPT without the need of any prior adversarial exploits (jailbreaking).

摘要: ChatGPT生成类似人类的响应并理解上下文的能力使其成为对话代理、内容创建、数据分析以及研究和创新的流行工具。然而，它的有效性和易访问性使其成为生成恶意内容的主要目标，例如网络钓鱼攻击，这些内容可能会将用户置于风险之中。在这项工作中，我们识别了几个可以提供给ChatGPT以生成功能性钓鱼网站的恶意提示。通过迭代的方法，我们发现可以让这些钓鱼网站模仿流行品牌，并模仿几种已知的规避策略，以避免被反钓鱼实体发现。这些攻击可以使用普通的ChatGPT生成，而不需要任何先前的对抗性攻击(越狱)。



## **24. Communication-Robust Multi-Agent Learning by Adaptable Auxiliary Multi-Agent Adversary Generation**

基于自适应辅助多智能体对手生成的通信健壮多智能体学习 cs.LG

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05116v1) [paper-pdf](http://arxiv.org/pdf/2305.05116v1)

**Authors**: Lei Yuan, Feng Chen, Zhongzhang Zhang, Yang Yu

**Abstract**: Communication can promote coordination in cooperative Multi-Agent Reinforcement Learning (MARL). Nowadays, existing works mainly focus on improving the communication efficiency of agents, neglecting that real-world communication is much more challenging as there may exist noise or potential attackers. Thus the robustness of the communication-based policies becomes an emergent and severe issue that needs more exploration. In this paper, we posit that the ego system trained with auxiliary adversaries may handle this limitation and propose an adaptable method of Multi-Agent Auxiliary Adversaries Generation for robust Communication, dubbed MA3C, to obtain a robust communication-based policy. In specific, we introduce a novel message-attacking approach that models the learning of the auxiliary attacker as a cooperative problem under a shared goal to minimize the coordination ability of the ego system, with which every information channel may suffer from distinct message attacks. Furthermore, as naive adversarial training may impede the generalization ability of the ego system, we design an attacker population generation approach based on evolutionary learning. Finally, the ego system is paired with an attacker population and then alternatively trained against the continuously evolving attackers to improve its robustness, meaning that both the ego system and the attackers are adaptable. Extensive experiments on multiple benchmarks indicate that our proposed MA3C provides comparable or better robustness and generalization ability than other baselines.

摘要: 在协作多智能体强化学习(MAIL)中，通信可以促进协作。目前，已有的研究主要集中在提高智能体的通信效率上，而忽略了现实世界中可能存在噪声或潜在攻击者的情况下，通信更具挑战性。因此，基于通信的策略的健壮性成为一个迫切而严峻的问题，需要进一步探讨。在本文中，我们假设用辅助对手训练的EGO系统可以处理这一局限性，并提出了一种用于稳健通信的自适应多智能体辅助对手生成方法MA3C，以获得基于通信的健壮策略。具体地说，我们引入了一种新的消息攻击方法，将辅助攻击者的学习建模为一个共享目标下的合作问题，以最小化EGO系统的协调能力，在这种情况下，每个信息通道都可能遭受不同的消息攻击。此外，由于天真的对抗性训练可能会阻碍EGO系统的泛化能力，我们设计了一种基于进化学习的攻击种群生成方法。最后，EGO系统与攻击者群体配对，然后交替地针对不断进化的攻击者进行训练，以提高其健壮性，这意味着EGO系统和攻击者都是自适应的。在多个基准上的大量实验表明，我们提出的MA3C提供了与其他基准相当或更好的稳健性和泛化能力。



## **25. Escaping saddle points in zeroth-order optimization: the power of two-point estimators**

零阶最优化中的鞍点逃逸：两点估计的能力 math.OC

To appear at ICML 2023

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2209.13555v3) [paper-pdf](http://arxiv.org/pdf/2209.13555v3)

**Authors**: Zhaolin Ren, Yujie Tang, Na Li

**Abstract**: Two-point zeroth order methods are important in many applications of zeroth-order optimization, such as robotics, wind farms, power systems, online optimization, and adversarial robustness to black-box attacks in deep neural networks, where the problem may be high-dimensional and/or time-varying. Most problems in these applications are nonconvex and contain saddle points. While existing works have shown that zeroth-order methods utilizing $\Omega(d)$ function valuations per iteration (with $d$ denoting the problem dimension) can escape saddle points efficiently, it remains an open question if zeroth-order methods based on two-point estimators can escape saddle points. In this paper, we show that by adding an appropriate isotropic perturbation at each iteration, a zeroth-order algorithm based on $2m$ (for any $1 \leq m \leq d$) function evaluations per iteration can not only find $\epsilon$-second order stationary points polynomially fast, but do so using only $\tilde{O}\left(\frac{d}{m\epsilon^{2}\bar{\psi}}\right)$ function evaluations, where $\bar{\psi} \geq \tilde{\Omega}\left(\sqrt{\epsilon}\right)$ is a parameter capturing the extent to which the function of interest exhibits the strict saddle property.

摘要: 两点零阶方法在零阶优化的许多应用中都是重要的，例如机器人、风电场、电力系统、在线优化以及深层神经网络中对黑盒攻击的对抗鲁棒性，这些问题可能是高维的和/或时变的。这些应用中的大多数问题都是非凸的，并且包含鞍点。虽然已有的工作表明，利用每次迭代的$\Omega(D)$函数赋值(其中$d$表示问题的维度)的零级方法可以有效地逃离鞍点，但基于两点估计的零级方法是否能够逃离鞍点仍然是一个悬而未决的问题。本文证明了，通过在每次迭代中加入适当的各向同性扰动，基于每一次迭代的$2m$(对于任意$1\leq m\leq d$)函数求值的零阶算法不仅可以多项式地快速地找到$-二阶驻点，而且只使用$\tilde{O}\left(\frac{d}{m\epsilon^{2}\bar{\psi}}\right)$函数求值，其中，$\bar{\psi}\geq\tilde{\omega}\Left(\Sqrt{\epsilon}\right)$是捕获感兴趣函数展现严格鞍形属性的程度的参数。



## **26. Less is More: Removing Text-regions Improves CLIP Training Efficiency and Robustness**

少即是多：去除文本区域可以提高剪辑训练的效率和健壮性 cs.CV

10 pages, 8 figures

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.05095v1) [paper-pdf](http://arxiv.org/pdf/2305.05095v1)

**Authors**: Liangliang Cao, Bowen Zhang, Chen Chen, Yinfei Yang, Xianzhi Du, Wencong Zhang, Zhiyun Lu, Yantao Zheng

**Abstract**: The CLIP (Contrastive Language-Image Pre-training) model and its variants are becoming the de facto backbone in many applications. However, training a CLIP model from hundreds of millions of image-text pairs can be prohibitively expensive. Furthermore, the conventional CLIP model doesn't differentiate between the visual semantics and meaning of text regions embedded in images. This can lead to non-robustness when the text in the embedded region doesn't match the image's visual appearance. In this paper, we discuss two effective approaches to improve the efficiency and robustness of CLIP training: (1) augmenting the training dataset while maintaining the same number of optimization steps, and (2) filtering out samples that contain text regions in the image. By doing so, we significantly improve the classification and retrieval accuracy on public benchmarks like ImageNet and CoCo. Filtering out images with text regions also protects the model from typographic attacks. To verify this, we build a new dataset named ImageNet with Adversarial Text Regions (ImageNet-Attr). Our filter-based CLIP model demonstrates a top-1 accuracy of 68.78\%, outperforming previous models whose accuracy was all below 50\%.

摘要: CLIP(对比语言-图像预训练)模型及其变体正在成为许多应用中事实上的支柱。然而，从数以亿计的图像-文本对中训练剪辑模型的成本可能高得令人望而却步。此外，传统的剪辑模型不区分嵌入在图像中的文本区域的视觉语义和含义。当嵌入区域中的文本与图像的视觉外观不匹配时，这可能会导致不稳定。在本文中，我们讨论了两种有效的方法来提高剪辑训练的效率和稳健性：(1)在保持相同的优化步数的情况下扩大训练数据集；(2)过滤掉图像中包含文本区域的样本。通过这样做，我们在ImageNet和CoCo等公共基准上显著提高了分类和检索的准确性。过滤掉带有文本区域的图像还可以保护模型免受排版攻击。为了验证这一点，我们构建了一个名为ImageNet的新数据集，其中包含敌对文本区域(ImageNet-Attr)。我们的基于过滤器的剪辑模型的TOP-1精度为68.78\%，超过了以前的精度都在50\%以下的模型。



## **27. Distributed Detection over Blockchain-aided Internet of Things in the Presence of Attacks**

存在攻击的区块链辅助物联网分布式检测 cs.CR

16 pages, 4 figures. This work has been submitted to the IEEE TIFS

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.05070v1) [paper-pdf](http://arxiv.org/pdf/2305.05070v1)

**Authors**: Yiming Jiang, Jiangfan Zhang

**Abstract**: Distributed detection over a blockchain-aided Internet of Things (BIoT) network in the presence of attacks is considered, where the integrated blockchain is employed to secure data exchanges over the BIoT as well as data storage at the agents of the BIoT. We consider a general adversary model where attackers jointly exploit the vulnerability of IoT devices and that of the blockchain employed in the BIoT. The optimal attacking strategy which minimizes the Kullback-Leibler divergence is pursued. It can be shown that this optimization problem is nonconvex, and hence it is generally intractable to find the globally optimal solution to such a problem. To overcome this issue, we first propose a relaxation method that can convert the original nonconvex optimization problem into a convex optimization problem, and then the analytic expression for the optimal solution to the relaxed convex optimization problem is derived. The optimal value of the relaxed convex optimization problem provides a detection performance guarantee for the BIoT in the presence of attacks. In addition, we develop a coordinate descent algorithm which is based on a capped water-filling method to solve the relaxed convex optimization problem, and moreover, we show that the convergence of the proposed coordinate descent algorithm can be guaranteed.

摘要: 考虑了在存在攻击的情况下通过区块链辅助的物联网(Biot)网络进行分布式检测，其中使用集成的区块链来保护Biot上的数据交换以及Biot代理处的数据存储。我们考虑了一个一般的对手模型，在该模型中，攻击者联合利用物联网设备和Biot中采用的区块链的漏洞。寻求使Kullback-Leibler发散最小的最优攻击策略。可以看出，该优化问题是非凸的，因此寻找此类问题的全局最优解通常是困难的。为了克服这个问题，我们首先提出了一种松弛方法，可以将原来的非凸优化问题转化为凸优化问题，然后推导出松弛凸优化问题的最优解的解析表达式。松弛凸优化问题的最优值为Biot在存在攻击时的检测性能提供了保证。此外，我们还提出了一种基于封顶注水方法的坐标下降算法来求解松弛凸优化问题，并证明了该算法的收敛是有保证的。



## **28. A Survey on AI/ML-Driven Intrusion and Misbehavior Detection in Networked Autonomous Systems: Techniques, Challenges and Opportunities**

AI/ML驱动的网络自治系统入侵与行为检测研究综述：技术、挑战与机遇 cs.NI

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.05040v1) [paper-pdf](http://arxiv.org/pdf/2305.05040v1)

**Authors**: Opeyemi Ajibuwa, Bechir Hamdaoui, Attila A. Yavuz

**Abstract**: AI/ML-based intrusion detection systems (IDSs) and misbehavior detection systems (MDSs) have shown great potential in identifying anomalies in the network traffic of networked autonomous systems. Despite the vast research efforts, practical deployments of such systems in the real world have been limited. Although the safety-critical nature of autonomous systems and the vulnerability of learning-based techniques to adversarial attacks are among the potential reasons, the lack of objective evaluation and feasibility assessment metrics is one key reason behind the limited adoption of these systems in practical settings. This survey aims to address the aforementioned limitation by presenting an in-depth analysis of AI/ML-based IDSs/MDSs and establishing baseline metrics relevant to networked autonomous systems. Furthermore, this work thoroughly surveys recent studies in this domain, highlighting the evaluation metrics and gaps in the current literature. It also presents key findings derived from our analysis of the surveyed papers and proposes guidelines for providing AI/ML-based IDS/MDS solution approaches suitable for vehicular network applications. Our work provides researchers and practitioners with the needed tools to evaluate the feasibility of AI/ML-based IDS/MDS techniques in real-world settings, with the aim of facilitating the practical adoption of such techniques in emerging autonomous vehicular systems.

摘要: 基于AI/ML的入侵检测系统和行为异常检测系统在识别网络自治系统的网络流量异常方面显示出了巨大的潜力。尽管付出了巨大的研究努力，但这类系统在现实世界中的实际部署一直有限。虽然自主系统的安全关键性质和基于学习的技术对对手攻击的脆弱性是潜在原因之一，但缺乏客观评估和可行性评估衡量标准是这些系统在实际环境中采用有限的一个关键原因。本调查旨在通过深入分析基于AI/ML的入侵检测系统/分布式检测系统并建立与联网自治系统相关的基线度量来解决上述限制。此外，这项工作深入地综述了这一领域的最新研究，突出了当前文献中的评价指标和差距。它还介绍了我们对调查论文的分析得出的主要发现，并提出了适用于车载网络应用的基于AI/ML的入侵检测/MDS解决方案方法的指导方针。我们的工作为研究人员和实践者提供了必要的工具来评估基于AI/ML的入侵检测/MDS技术在现实世界中的可行性，目的是促进此类技术在新兴的自主车辆系统中的实际采用。



## **29. White-Box Multi-Objective Adversarial Attack on Dialogue Generation**

对话生成的白盒多目标对抗性攻击 cs.CL

ACL 2023 main conference long paper

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.03655v2) [paper-pdf](http://arxiv.org/pdf/2305.03655v2)

**Authors**: Yufei Li, Zexin Li, Yingfan Gao, Cong Liu

**Abstract**: Pre-trained transformers are popular in state-of-the-art dialogue generation (DG) systems. Such language models are, however, vulnerable to various adversarial samples as studied in traditional tasks such as text classification, which inspires our curiosity about their robustness in DG systems. One main challenge of attacking DG models is that perturbations on the current sentence can hardly degrade the response accuracy because the unchanged chat histories are also considered for decision-making. Instead of merely pursuing pitfalls of performance metrics such as BLEU, ROUGE, we observe that crafting adversarial samples to force longer generation outputs benefits attack effectiveness -- the generated responses are typically irrelevant, lengthy, and repetitive. To this end, we propose a white-box multi-objective attack method called DGSlow. Specifically, DGSlow balances two objectives -- generation accuracy and length, via a gradient-based multi-objective optimizer and applies an adaptive searching mechanism to iteratively craft adversarial samples with only a few modifications. Comprehensive experiments on four benchmark datasets demonstrate that DGSlow could significantly degrade state-of-the-art DG models with a higher success rate than traditional accuracy-based methods. Besides, our crafted sentences also exhibit strong transferability in attacking other models.

摘要: 预先培训的变压器在最先进的对话生成(DG)系统中很受欢迎。然而，这类语言模型容易受到文本分类等传统任务中研究的各种对抗性样本的影响，这引发了我们对它们在DG系统中的健壮性的好奇。攻击DG模型的一个主要挑战是，当前句子的扰动几乎不会降低响应精度，因为没有变化的聊天历史也被考虑用于决策。而不是仅仅追求性能指标的陷阱，如BLEU，Rouge，我们观察到精心制作敌意样本来迫使更长的世代输出有利于攻击效率-生成的响应通常是无关的、冗长的和重复的。为此，我们提出了一种白盒多目标攻击方法DGSlow。具体地说，DGSlow通过基于梯度的多目标优化器来平衡生成精度和长度这两个目标，并应用自适应搜索机制来迭代地创建只需少量修改的对抗性样本。在四个基准数据集上的综合实验表明，DGSlow可以显著降低最新的DG模型，并且比传统的基于精度的方法具有更高的成功率。此外，我们制作的句子在攻击其他模型时也表现出很强的可转移性。



## **30. Understanding Noise-Augmented Training for Randomized Smoothing**

理解随机平滑的噪声增强训练 cs.LG

Transactions on Machine Learning Research, 2023

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.04746v1) [paper-pdf](http://arxiv.org/pdf/2305.04746v1)

**Authors**: Ambar Pal, Jeremias Sulam

**Abstract**: Randomized smoothing is a technique for providing provable robustness guarantees against adversarial attacks while making minimal assumptions about a classifier. This method relies on taking a majority vote of any base classifier over multiple noise-perturbed inputs to obtain a smoothed classifier, and it remains the tool of choice to certify deep and complex neural network models. Nonetheless, non-trivial performance of such smoothed classifier crucially depends on the base model being trained on noise-augmented data, i.e., on a smoothed input distribution. While widely adopted in practice, it is still unclear how this noisy training of the base classifier precisely affects the risk of the robust smoothed classifier, leading to heuristics and tricks that are poorly understood. In this work we analyze these trade-offs theoretically in a binary classification setting, proving that these common observations are not universal. We show that, without making stronger distributional assumptions, no benefit can be expected from predictors trained with noise-augmentation, and we further characterize distributions where such benefit is obtained. Our analysis has direct implications to the practical deployment of randomized smoothing, and we illustrate some of these via experiments on CIFAR-10 and MNIST, as well as on synthetic datasets.

摘要: 随机化平滑是一种在对分类器做出最小假设的同时提供针对对手攻击的可证明的稳健性保证的技术。这种方法依赖于在多个受噪声干扰的输入上取得任何基本分类器的多数票来获得平滑的分类器，并且它仍然是验证深度和复杂神经网络模型的首选工具。尽管如此，这种平滑分类器的非平凡性能关键取决于基本模型是基于噪声增强的数据来训练的，即基于平滑的输入分布。虽然在实践中被广泛采用，但仍然不清楚基分类器的这种噪声训练如何准确地影响稳健平滑分类器的风险，从而导致启发式算法和技巧鲜为人知。在这项工作中，我们在二进制分类的背景下从理论上分析了这些权衡，证明了这些常见的观察结果并不普遍。我们证明，如果不做更强的分布假设，则不能期望从经过噪声增强训练的预报器中获益，并且我们进一步刻画了获得这种益处的分布。我们的分析对随机平滑的实际部署有直接的影响，我们通过在CIFAR-10和MNIST上以及在合成数据集上的实验来说明其中的一些。



## **31. Evaluating Impact of User-Cluster Targeted Attacks in Matrix Factorisation Recommenders**

在矩阵分解推荐器中评估用户聚类定向攻击的影响 cs.IR

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.04694v1) [paper-pdf](http://arxiv.org/pdf/2305.04694v1)

**Authors**: Sulthana Shams, Douglas Leith

**Abstract**: In practice, users of a Recommender System (RS) fall into a few clusters based on their preferences. In this work, we conduct a systematic study on user-cluster targeted data poisoning attacks on Matrix Factorisation (MF) based RS, where an adversary injects fake users with falsely crafted user-item feedback to promote an item to a specific user cluster. We analyse how user and item feature matrices change after data poisoning attacks and identify the factors that influence the effectiveness of the attack on these feature matrices. We demonstrate that the adversary can easily target specific user clusters with minimal effort and that some items are more susceptible to attacks than others. Our theoretical analysis has been validated by the experimental results obtained from two real-world datasets. Our observations from the study could serve as a motivating point to design a more robust RS.

摘要: 在实践中，推荐系统(RS)的用户根据他们的偏好分为几个集群。在这项工作中，我们对基于矩阵分解(MF)的RS中针对用户簇的定向数据中毒攻击进行了系统的研究。在该攻击中，敌手向虚假用户注入虚假的用户项反馈，以将项推送到特定的用户簇。我们分析了数据中毒攻击后用户和项目特征矩阵的变化，并确定了影响这些特征矩阵攻击有效性的因素。我们证明了敌手可以很容易地以最小的努力瞄准特定的用户集群，并且一些项目比其他项目更容易受到攻击。两个真实数据集的实验结果验证了我们的理论分析。我们从这项研究中观察到的结果可以作为设计更健壮的RS的激励点。



## **32. StyleAdv: Meta Style Adversarial Training for Cross-Domain Few-Shot Learning**

StyleAdv：跨域少发学习的元式对抗性训练 cs.CV

accepted by CVPR 2023

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2302.09309v2) [paper-pdf](http://arxiv.org/pdf/2302.09309v2)

**Authors**: Yuqian Fu, Yu Xie, Yanwei Fu, Yu-Gang Jiang

**Abstract**: Cross-Domain Few-Shot Learning (CD-FSL) is a recently emerging task that tackles few-shot learning across different domains. It aims at transferring prior knowledge learned on the source dataset to novel target datasets. The CD-FSL task is especially challenged by the huge domain gap between different datasets. Critically, such a domain gap actually comes from the changes of visual styles, and wave-SAN empirically shows that spanning the style distribution of the source data helps alleviate this issue. However, wave-SAN simply swaps styles of two images. Such a vanilla operation makes the generated styles ``real'' and ``easy'', which still fall into the original set of the source styles. Thus, inspired by vanilla adversarial learning, a novel model-agnostic meta Style Adversarial training (StyleAdv) method together with a novel style adversarial attack method is proposed for CD-FSL. Particularly, our style attack method synthesizes both ``virtual'' and ``hard'' adversarial styles for model training. This is achieved by perturbing the original style with the signed style gradients. By continually attacking styles and forcing the model to recognize these challenging adversarial styles, our model is gradually robust to the visual styles, thus boosting the generalization ability for novel target datasets. Besides the typical CNN-based backbone, we also employ our StyleAdv method on large-scale pretrained vision transformer. Extensive experiments conducted on eight various target datasets show the effectiveness of our method. Whether built upon ResNet or ViT, we achieve the new state of the art for CD-FSL. Code is available at https://github.com/lovelyqian/StyleAdv-CDFSL.

摘要: 跨域少镜头学习(CD-FSL)是近年来出现的一项研究课题，旨在解决不同领域的少镜头学习问题。它的目的是将在源数据集上学习的先验知识转移到新的目标数据集。CD-FSL任务尤其受到不同数据集之间巨大的域差距的挑战。关键的是，这样的领域差距实际上来自视觉风格的变化，而WAVE-SAN经验表明，跨越源数据的风格分布有助于缓解这一问题。然而，WAVE-SAN只是简单地交换两个图像的样式。这样的普通操作使生成的样式“真实”和“容易”，它们仍然属于原始的源样式集。因此，受传统对抗学习的启发，提出了一种新的模型--不可知元风格对抗训练方法(StyleAdv)和一种新风格的对抗攻击方法。具体地说，我们的风格攻击方法为模型训练综合了“虚拟”和“硬”两种对抗性风格。这是通过用签名的样式渐变扰乱原始样式来实现的。通过不断攻击风格并迫使模型识别这些具有挑战性的对抗性风格，我们的模型逐渐对视觉风格具有健壮性，从而增强了对新目标数据集的泛化能力。除了典型的基于CNN的主干，我们还将我们的StyleAdv方法应用于大规模的预训练视觉转换器。在8个不同的目标数据集上进行的大量实验表明了该方法的有效性。无论是建立在ResNet还是VIT之上，我们都达到了CD-FSL的最新技术水平。代码可在https://github.com/lovelyqian/StyleAdv-CDFSL.上找到



## **33. Recent Advances in Reliable Deep Graph Learning: Inherent Noise, Distribution Shift, and Adversarial Attack**

可靠深度图学习的最新进展：固有噪声、分布漂移和敌意攻击 cs.LG

Preprint. 9 pages, 2 figures

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2202.07114v2) [paper-pdf](http://arxiv.org/pdf/2202.07114v2)

**Authors**: Jintang Li, Bingzhe Wu, Chengbin Hou, Guoji Fu, Yatao Bian, Liang Chen, Junzhou Huang, Zibin Zheng

**Abstract**: Deep graph learning (DGL) has achieved remarkable progress in both business and scientific areas ranging from finance and e-commerce to drug and advanced material discovery. Despite the progress, applying DGL to real-world applications faces a series of reliability threats including inherent noise, distribution shift, and adversarial attacks. This survey aims to provide a comprehensive review of recent advances for improving the reliability of DGL algorithms against the above threats. In contrast to prior related surveys which mainly focus on adversarial attacks and defense, our survey covers more reliability-related aspects of DGL, i.e., inherent noise and distribution shift. Additionally, we discuss the relationships among above aspects and highlight some important issues to be explored in future research.

摘要: 深度图学习(DGL)在从金融和电子商务到药物和先进材料发现的商业和科学领域都取得了显着的进展。尽管取得了进展，但将DGL应用于现实世界的应用程序面临着一系列可靠性威胁，包括固有的噪声、分布偏移和对手攻击。本调查旨在全面回顾在提高DGL算法针对上述威胁的可靠性方面的最新进展。与以往主要关注对抗性攻击和防御的相关调查不同，我们的调查涵盖了DGL更多与可靠性相关的方面，即固有噪声和分布漂移。此外，我们还讨论了上述几个方面之间的关系，并指出了未来研究中需要探索的一些重要问题。



## **34. Toward Adversarial Training on Contextualized Language Representation**

语境化语言表征的对抗性训练 cs.CL

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.04557v1) [paper-pdf](http://arxiv.org/pdf/2305.04557v1)

**Authors**: Hongqiu Wu, Yongxiang Liu, Hanwen Shi, Hai Zhao, Min Zhang

**Abstract**: Beyond the success story of adversarial training (AT) in the recent text domain on top of pre-trained language models (PLMs), our empirical study showcases the inconsistent gains from AT on some tasks, e.g. commonsense reasoning, named entity recognition. This paper investigates AT from the perspective of the contextualized language representation outputted by PLM encoders. We find the current AT attacks lean to generate sub-optimal adversarial examples that can fool the decoder part but have a minor effect on the encoder. However, we find it necessary to effectively deviate the latter one to allow AT to gain. Based on the observation, we propose simple yet effective \textit{Contextualized representation-Adversarial Training} (CreAT), in which the attack is explicitly optimized to deviate the contextualized representation of the encoder. It allows a global optimization of adversarial examples that can fool the entire model. We also find CreAT gives rise to a better direction to optimize the adversarial examples, to let them less sensitive to hyperparameters. Compared to AT, CreAT produces consistent performance gains on a wider range of tasks and is proven to be more effective for language pre-training where only the encoder part is kept for downstream tasks. We achieve the new state-of-the-art performances on a series of challenging benchmarks, e.g. AdvGLUE (59.1 $ \rightarrow $ 61.1), HellaSWAG (93.0 $ \rightarrow $ 94.9), ANLI (68.1 $ \rightarrow $ 69.3).

摘要: 除了最近文本领域在预训练语言模型(PLM)之上的对抗性训练(AT)的成功案例外，我们的实证研究还展示了在一些任务上，例如常识推理、命名实体识别，对抗性训练(AT)的不一致收获。本文从PLM编码者输出的语境化语言表征的角度对自动翻译进行研究。我们发现，当前的AT攻击倾向于生成次优的对抗性示例，这些示例可以愚弄解码器部分，但对编码器的影响很小。然而，我们发现有必要有效地偏离后者，以使AT获得收益。在此基础上，我们提出了简单而有效的文本化表示-对抗性训练(CREAT)，其中攻击被显式地优化以偏离编码者的上下文表示。它允许对可以愚弄整个模型的对抗性例子进行全局优化。我们还发现CREAT给出了一个更好的方向来优化对抗性例子，让它们对超参数不那么敏感。与AT相比，CREAT在更广泛的任务范围内产生了一致的性能提升，并且被证明在语言预训练中更有效，因为只有编码器部分被保留用于后续任务。我们在一系列具有挑战性的基准上实现了新的最先进的表现，例如AdvGLUE(59.1$\right tarrow$61.1)，HellaSWAG(93.0$\right tarrow$94.9)，Anli(68.1$\right tarrow$69.3)。



## **35. Privacy-preserving Adversarial Facial Features**

保护隐私的敌意面部特征 cs.CV

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.05391v1) [paper-pdf](http://arxiv.org/pdf/2305.05391v1)

**Authors**: Zhibo Wang, He Wang, Shuaifan Jin, Wenwen Zhang, Jiahui Hu, Yan Wang, Peng Sun, Wei Yuan, Kaixin Liu, Kui Ren

**Abstract**: Face recognition service providers protect face privacy by extracting compact and discriminative facial features (representations) from images, and storing the facial features for real-time recognition. However, such features can still be exploited to recover the appearance of the original face by building a reconstruction network. Although several privacy-preserving methods have been proposed, the enhancement of face privacy protection is at the expense of accuracy degradation. In this paper, we propose an adversarial features-based face privacy protection (AdvFace) approach to generate privacy-preserving adversarial features, which can disrupt the mapping from adversarial features to facial images to defend against reconstruction attacks. To this end, we design a shadow model which simulates the attackers' behavior to capture the mapping function from facial features to images and generate adversarial latent noise to disrupt the mapping. The adversarial features rather than the original features are stored in the server's database to prevent leaked features from exposing facial information. Moreover, the AdvFace requires no changes to the face recognition network and can be implemented as a privacy-enhancing plugin in deployed face recognition systems. Extensive experimental results demonstrate that AdvFace outperforms the state-of-the-art face privacy-preserving methods in defending against reconstruction attacks while maintaining face recognition accuracy.

摘要: 人脸识别服务提供商通过从图像中提取紧凑和区别性的面部特征(表示)，并存储面部特征以供实时识别，从而保护面部隐私。然而，这些特征仍然可以通过构建重建网络来恢复原始人脸的外观。虽然已经提出了几种隐私保护方法，但增强人脸隐私保护的代价是准确性下降。本文提出了一种基于对抗特征的人脸隐私保护方法(AdvFace)来生成隐私保护的对抗特征，该方法可以破坏对抗特征到人脸图像的映射，从而防止重构攻击。为此，我们设计了一种模拟攻击者行为的阴影模型来捕捉人脸特征到图像的映射函数，并产生对抗性的潜在噪声来破坏映射。敌意特征而不是原始特征存储在服务器的数据库中，以防止泄露的特征暴露面部信息。此外，AdvFace不需要改变人脸识别网络，可以作为已部署的人脸识别系统中的隐私增强插件来实现。大量的实验结果表明，AdvFace在抵抗重构攻击的同时，在保持人脸识别准确率方面优于最先进的人脸隐私保护方法。



## **36. Attack-SAM: Towards Attacking Segment Anything Model With Adversarial Examples**

攻击-SAM：以对手为例的攻击分段Anything模型 cs.CV

The first work to attack Segment Anything Model with adversarial  examples

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.00866v2) [paper-pdf](http://arxiv.org/pdf/2305.00866v2)

**Authors**: Chenshuang Zhang, Chaoning Zhang, Taegoo Kang, Donghun Kim, Sung-Ho Bae, In So Kweon

**Abstract**: Segment Anything Model (SAM) has attracted significant attention recently, due to its impressive performance on various downstream tasks in a zero-short manner. Computer vision (CV) area might follow the natural language processing (NLP) area to embark on a path from task-specific vision models toward foundation models. However, deep vision models are widely recognized as vulnerable to adversarial examples, which fool the model to make wrong predictions with imperceptible perturbation. Such vulnerability to adversarial attacks causes serious concerns when applying deep models to security-sensitive applications. Therefore, it is critical to know whether the vision foundation model SAM can also be fooled by adversarial attacks. To the best of our knowledge, our work is the first of its kind to conduct a comprehensive investigation on how to attack SAM with adversarial examples. With the basic attack goal set to mask removal, we investigate the adversarial robustness of SAM in the full white-box setting and transfer-based black-box settings. Beyond the basic goal of mask removal, we further investigate and find that it is possible to generate any desired mask by the adversarial attack.

摘要: 分段任意模型(SAM)最近受到了极大的关注，因为它在各种下游任务上以零-短的方式表现出令人印象深刻的性能。计算机视觉(CV)领域可能会跟随自然语言处理(NLP)领域，走上一条从特定于任务的视觉模型到基础模型的道路。然而，深度视觉模型被广泛认为容易受到敌意例子的影响，这些例子愚弄了模型，使其在不知不觉中做出了错误的预测。在将深度模型应用于安全敏感应用程序时，此类易受敌意攻击的漏洞会引起严重关注。因此，了解VISION基础模型SAM是否也会被对抗性攻击愚弄是至关重要的。据我们所知，我们的工作是第一次对如何用对抗性例子攻击SAM进行全面调查。以去除掩码为基本攻击目标，研究了SAM在完全白盒设置和基于传输的黑盒设置下的对抗健壮性。除了掩码去除的基本目标之外，我们进一步研究发现，通过对抗性攻击可以产生任何想要的掩码。



## **37. Location Privacy Threats and Protections in Future Vehicular Networks: A Comprehensive Review**

未来车载网络中位置隐私的威胁与保护综述 cs.CR

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.04503v1) [paper-pdf](http://arxiv.org/pdf/2305.04503v1)

**Authors**: Baihe Ma, Xu Wang, Xiaojie Lin, Yanna Jiang, Caijun Sun, Zhe Wang, Guangsheng Yu, Ying He, Wei Ni, Ren Ping Liu

**Abstract**: Location privacy is critical in vehicular networks, where drivers' trajectories and personal information can be exposed, allowing adversaries to launch data and physical attacks that threaten drivers' safety and personal security. This survey reviews comprehensively different localization techniques, including widely used ones like sensing infrastructure-based, optical vision-based, and cellular radio-based localization, and identifies inadequately addressed location privacy concerns. We classify Location Privacy Preserving Mechanisms (LPPMs) into user-side, server-side, and user-server-interface-based, and evaluate their effectiveness. Our analysis shows that the user-server-interface-based LPPMs have received insufficient attention in the literature, despite their paramount importance in vehicular networks. Further, we examine methods for balancing data utility and privacy protection for existing LPPMs in vehicular networks and highlight emerging challenges from future upper-layer location privacy attacks, wireless technologies, and network convergences. By providing insights into the relationship between localization techniques and location privacy, and evaluating the effectiveness of different LPPMs, this survey can help inform the development of future LPPMs in vehicular networks.

摘要: 位置隐私在车载网络中至关重要，在车载网络中，司机的轨迹和个人信息可能会被暴露，从而允许对手发动威胁司机安全和人身安全的数据和物理攻击。这项调查全面回顾了不同的定位技术，包括广泛使用的基于传感基础设施的定位技术、基于光学视觉的定位技术和基于蜂窝无线电的定位技术，并确定了没有充分解决位置隐私问题的问题。我们将位置隐私保护机制(LPPM)分为基于用户端、基于服务器端和基于用户-服务器接口，并对其有效性进行了评估。我们的分析表明，基于用户-服务器接口的LPPM在文献中没有得到足够的关注，尽管它们在车载网络中非常重要。此外，我们还研究了在车载网络中平衡现有LPPM的数据效用和隐私保护的方法，并强调了未来上层位置隐私攻击、无线技术和网络融合带来的新挑战。通过深入了解定位技术和位置隐私之间的关系，以及评估不同LPPM的有效性，这项调查有助于为未来LPPM在车载网络中的发展提供信息。



## **38. Adversarial Examples Detection with Enhanced Image Difference Features based on Local Histogram Equalization**

基于局部直方图均衡的增强图像差分特征敌例检测 cs.CV

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.04436v1) [paper-pdf](http://arxiv.org/pdf/2305.04436v1)

**Authors**: Zhaoxia Yin, Shaowei Zhu, Hang Su, Jianteng Peng, Wanli Lyu, Bin Luo

**Abstract**: Deep Neural Networks (DNNs) have recently made significant progress in many fields. However, studies have shown that DNNs are vulnerable to adversarial examples, where imperceptible perturbations can greatly mislead DNNs even if the full underlying model parameters are not accessible. Various defense methods have been proposed, such as feature compression and gradient masking. However, numerous studies have proven that previous methods create detection or defense against certain attacks, which renders the method ineffective in the face of the latest unknown attack methods. The invisibility of adversarial perturbations is one of the evaluation indicators for adversarial example attacks, which also means that the difference in the local correlation of high-frequency information in adversarial examples and normal examples can be used as an effective feature to distinguish the two. Therefore, we propose an adversarial example detection framework based on a high-frequency information enhancement strategy, which can effectively extract and amplify the feature differences between adversarial examples and normal examples. Experimental results show that the feature augmentation module can be combined with existing detection models in a modular way under this framework. Improve the detector's performance and reduce the deployment cost without modifying the existing detection model.

摘要: 深度神经网络(DNN)近年来在许多领域都取得了重大进展。然而，研究表明，DNN很容易受到敌意例子的影响，在这些例子中，即使无法获得完整的底层模型参数，不可察觉的扰动也会极大地误导DNN。人们已经提出了各种防御方法，如特征压缩和梯度掩蔽。然而，大量的研究已经证明，以前的方法是针对某些攻击进行检测或防御，这使得该方法在面对最新的未知攻击方法时效率低下。对抗性扰动的不可见性是对抗性范例攻击的评价指标之一，这也意味着对抗性范例与正常范例高频信息局部相关性的差异可以作为区分两者的有效特征。因此，我们提出了一种基于高频信息增强策略的对抗性范例检测框架，能够有效地提取和放大对抗性范例与正常范例之间的特征差异。实验结果表明，在该框架下，特征增强模块可以与现有的检测模型以模块化的方式结合起来。在不修改现有检测模型的情况下，提高了检测器的性能，降低了部署成本。



## **39. Are Synonym Substitution Attacks Really Synonym Substitution Attacks?**

同义词替换攻击真的是同义词替换攻击吗？ cs.CL

Findings in ACL 2023. Major revisions compared with previous versions  are made to incorporate the reviewers' suggestions. The modifications made  are listed in Appendix A

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2210.02844v3) [paper-pdf](http://arxiv.org/pdf/2210.02844v3)

**Authors**: Cheng-Han Chiang, Hung-yi Lee

**Abstract**: In this paper, we explore the following question: Are synonym substitution attacks really synonym substitution attacks (SSAs)? We approach this question by examining how SSAs replace words in the original sentence and show that there are still unresolved obstacles that make current SSAs generate invalid adversarial samples. We reveal that four widely used word substitution methods generate a large fraction of invalid substitution words that are ungrammatical or do not preserve the original sentence's semantics. Next, we show that the semantic and grammatical constraints used in SSAs for detecting invalid word replacements are highly insufficient in detecting invalid adversarial samples.

摘要: 本文探讨了以下问题：同义词替换攻击真的是同义词替换攻击吗？我们通过审查SSA如何替换原始句子中的单词来处理这个问题，并表明仍然存在尚未解决的障碍，使当前的SSA生成无效的对抗性样本。我们发现，四种广泛使用的词替换方法产生了很大一部分无效替换词，这些词不符合语法或没有保留原始句子的语义。其次，我们证明了SSA中用于检测无效单词替换的语义和语法约束在检测无效对抗性样本方面严重不足。



## **40. Reactive Perturbation Defocusing for Textual Adversarial Defense**

文本对抗防御中的反应性扰动散焦 cs.CL

**SubmitDate**: 2023-05-06    [abs](http://arxiv.org/abs/2305.04067v1) [paper-pdf](http://arxiv.org/pdf/2305.04067v1)

**Authors**: Heng Yang, Ke Li

**Abstract**: Recent studies have shown that large pre-trained language models are vulnerable to adversarial attacks. Existing methods attempt to reconstruct the adversarial examples. However, these methods usually have limited performance in defense against adversarial examples, while also negatively impacting the performance on natural examples. To overcome this problem, we propose a method called Reactive Perturbation Defocusing (RPD). RPD uses an adversarial detector to identify adversarial examples and reduce false defenses on natural examples. Instead of reconstructing the adversaries, RPD injects safe perturbations into adversarial examples to distract the objective models from the malicious perturbations. Our experiments on three datasets, two objective models, and various adversarial attacks show that our proposed framework successfully repairs up to approximately 97% of correctly identified adversarial examples with only about a 2% performance decrease on natural examples. We also provide a demo of adversarial detection and repair based on our work.

摘要: 最近的研究表明，大型预先训练的语言模型容易受到对抗性攻击。现有的方法试图重建对抗性的例子。然而，这些方法通常在防御对抗性实例时性能有限，同时也对自然实例的性能产生负面影响。为了克服这个问题，我们提出了一种称为反应性微扰散焦(RPD)的方法。RPD使用对抗性检测器来识别对抗性实例，并减少对自然实例的错误防御。RPD不是重构对手，而是将安全扰动注入到敌意实例中，以分散目标模型对恶意扰动的注意力。我们在三个数据集、两个目标模型和各种对抗性攻击上的实验表明，我们的框架成功地修复了大约97%的正确识别的对抗性例子，而对自然例子的性能只有大约2%的下降。我们还提供了一个基于我们的工作的对抗性检测和修复的演示。



## **41. TPC: Transformation-Specific Smoothing for Point Cloud Models**

TPC：点云模型的变换特定平滑 cs.CV

Accepted as a conference paper at ICML 2022

**SubmitDate**: 2023-05-06    [abs](http://arxiv.org/abs/2201.12733v5) [paper-pdf](http://arxiv.org/pdf/2201.12733v5)

**Authors**: Wenda Chu, Linyi Li, Bo Li

**Abstract**: Point cloud models with neural network architectures have achieved great success and have been widely used in safety-critical applications, such as Lidar-based recognition systems in autonomous vehicles. However, such models are shown vulnerable to adversarial attacks which aim to apply stealthy semantic transformations such as rotation and tapering to mislead model predictions. In this paper, we propose a transformation-specific smoothing framework TPC, which provides tight and scalable robustness guarantees for point cloud models against semantic transformation attacks. We first categorize common 3D transformations into three categories: additive (e.g., shearing), composable (e.g., rotation), and indirectly composable (e.g., tapering), and we present generic robustness certification strategies for all categories respectively. We then specify unique certification protocols for a range of specific semantic transformations and their compositions. Extensive experiments on several common 3D transformations show that TPC significantly outperforms the state of the art. For example, our framework boosts the certified accuracy against twisting transformation along z-axis (within 20$^\circ$) from 20.3$\%$ to 83.8$\%$. Codes and models are available at https://github.com/chuwd19/Point-Cloud-Smoothing.

摘要: 神经网络结构的点云模型已经取得了巨大的成功，并在安全关键应用中得到了广泛的应用，例如自动车辆中基于激光雷达的识别系统。然而，这类模型被证明容易受到敌意攻击，这些攻击旨在应用诸如旋转和缩减等隐蔽的语义转换来误导模型预测。在本文中，我们提出了一个特定于变换的平滑框架TPC，它为点云模型抵抗语义变换攻击提供了紧凑和可伸缩的健壮性保证。我们首先将常见的3D变换分为三类：加性变换(如剪切)、可合成变换(如旋转变换)和间接合成变换(如锥化变换)，并分别针对所有类别提出了通用的健壮性认证策略。然后，我们为一系列特定的语义转换及其组合指定唯一的认证协议。对几种常见的3D变换进行的大量实验表明，TPC的性能明显优于最先进的技术。例如，我们的框架将z轴(在20$^\cic$内)抗扭曲变换的认证精度从20.3$\$提高到83.8$\$。有关代码和型号，请访问https://github.com/chuwd19/Point-Cloud-Smoothing.



## **42. Towards Prompt-robust Face Privacy Protection via Adversarial Decoupling Augmentation Framework**

基于对抗性解耦增强框架的快速稳健人脸隐私保护 cs.CV

8 pages, 6 figures

**SubmitDate**: 2023-05-06    [abs](http://arxiv.org/abs/2305.03980v1) [paper-pdf](http://arxiv.org/pdf/2305.03980v1)

**Authors**: Ruijia Wu, Yuhang Wang, Huafeng Shi, Zhipeng Yu, Yichao Wu, Ding Liang

**Abstract**: Denoising diffusion models have shown remarkable potential in various generation tasks. The open-source large-scale text-to-image model, Stable Diffusion, becomes prevalent as it can generate realistic artistic or facial images with personalization through fine-tuning on a limited number of new samples. However, this has raised privacy concerns as adversaries can acquire facial images online and fine-tune text-to-image models for malicious editing, leading to baseless scandals, defamation, and disruption to victims' lives. Prior research efforts have focused on deriving adversarial loss from conventional training processes for facial privacy protection through adversarial perturbations. However, existing algorithms face two issues: 1) they neglect the image-text fusion module, which is the vital module of text-to-image diffusion models, and 2) their defensive performance is unstable against different attacker prompts. In this paper, we propose the Adversarial Decoupling Augmentation Framework (ADAF), addressing these issues by targeting the image-text fusion module to enhance the defensive performance of facial privacy protection algorithms. ADAF introduces multi-level text-related augmentations for defense stability against various attacker prompts. Concretely, considering the vision, text, and common unit space, we propose Vision-Adversarial Loss, Prompt-Robust Augmentation, and Attention-Decoupling Loss. Extensive experiments on CelebA-HQ and VGGFace2 demonstrate ADAF's promising performance, surpassing existing algorithms.

摘要: 去噪扩散模型在各种发电任务中显示出显著的潜力。开源的大规模文本到图像模型稳定扩散，因为它可以通过对有限数量的新样本进行微调来生成具有个性化的逼真艺术或面部图像，因此变得流行起来。然而，这引发了隐私问题，因为攻击者可以在线获取面部图像，并微调文本到图像的模型以进行恶意编辑，从而导致毫无根据的丑闻、诽谤和对受害者生活的破坏。以前的研究工作集中在通过对抗性扰动从传统的面部隐私保护训练过程中获得对抗性损失。然而，现有的算法面临着两个问题：1)它们忽略了文本到图像扩散模型中的关键模块--图文融合模块；2)它们对不同的攻击者提示的防御性能不稳定。在本文中，我们提出了对抗性解耦增强框架(ADAF)，通过针对图文融合模块来解决这些问题，以增强人脸隐私保护算法的防御性能。ADAF引入了与文本相关的多级别增强，以针对各种攻击者提示提供防御稳定性。具体地说，考虑到视觉、文本和公共单元空间，我们提出了视觉对抗性损失、即时稳健增强和注意分离损失。在CelebA-HQ和VGGFace2上的大量实验证明了ADAF的良好性能，超过了现有的算法。



## **43. Evading Watermark based Detection of AI-Generated Content**

基于规避水印的人工智能生成内容检测 cs.LG

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2305.03807v1) [paper-pdf](http://arxiv.org/pdf/2305.03807v1)

**Authors**: Zhengyuan Jiang, Jinghuai Zhang, Neil Zhenqiang Gong

**Abstract**: A generative AI model -- such as DALL-E, Stable Diffusion, and ChatGPT -- can generate extremely realistic-looking content, posing growing challenges to the authenticity of information. To address the challenges, watermark has been leveraged to detect AI-generated content. Specifically, a watermark is embedded into an AI-generated content before it is released. A content is detected as AI-generated if a similar watermark can be decoded from it. In this work, we perform a systematic study on the robustness of such watermark-based AI-generated content detection. We focus on AI-generated images. Our work shows that an attacker can post-process an AI-generated watermarked image via adding a small, human-imperceptible perturbation to it, such that the post-processed AI-generated image evades detection while maintaining its visual quality. We demonstrate the effectiveness of our attack both theoretically and empirically. Moreover, to evade detection, our adversarial post-processing method adds much smaller perturbations to the AI-generated images and thus better maintain their visual quality than existing popular image post-processing methods such as JPEG compression, Gaussian blur, and Brightness/Contrast. Our work demonstrates the insufficiency of existing watermark-based detection of AI-generated content, highlighting the urgent needs of new detection methods.

摘要: 生成性AI模型--如Dall-E、稳定扩散和ChatGPT--可以生成极其逼真的内容，对信息的真实性提出了越来越大的挑战。为了应对这些挑战，水印被用来检测人工智能生成的内容。具体地说，水印在发布之前被嵌入到人工智能生成的内容中。如果可以从内容中解码类似的水印，则该内容被检测为人工智能生成的内容。在这项工作中，我们对这种基于水印的人工智能内容检测的稳健性进行了系统的研究。我们专注于人工智能生成的图像。我们的工作表明，攻击者可以通过在人工智能生成的水印图像上添加一个人类无法察觉的小扰动来对其进行后处理，从而在保持其视觉质量的同时逃避检测。我们从理论上和经验上证明了我们的攻击的有效性。此外，为了逃避检测，我们的对抗性后处理方法对人工智能生成的图像添加了更小的扰动，从而比现有的流行的图像后处理方法，如JPEG压缩、高斯模糊和亮度/对比度，更好地保持了它们的视觉质量。我们的工作证明了现有基于水印的人工智能生成内容检测的不足，突出了对新检测方法的迫切需求。



## **44. Verifiable Learning for Robust Tree Ensembles**

用于稳健树集成的可验证学习 cs.LG

17 pages, 3 figures

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2305.03626v1) [paper-pdf](http://arxiv.org/pdf/2305.03626v1)

**Authors**: Stefano Calzavara, Lorenzo Cazzaro, Giulio Ermanno Pibiri, Nicola Prezza

**Abstract**: Verifying the robustness of machine learning models against evasion attacks at test time is an important research problem. Unfortunately, prior work established that this problem is NP-hard for decision tree ensembles, hence bound to be intractable for specific inputs. In this paper, we identify a restricted class of decision tree ensembles, called large-spread ensembles, which admit a security verification algorithm running in polynomial time. We then propose a new approach called verifiable learning, which advocates the training of such restricted model classes which are amenable for efficient verification. We show the benefits of this idea by designing a new training algorithm that automatically learns a large-spread decision tree ensemble from labelled data, thus enabling its security verification in polynomial time. Experimental results on publicly available datasets confirm that large-spread ensembles trained using our algorithm can be verified in a matter of seconds, using standard commercial hardware. Moreover, large-spread ensembles are more robust than traditional ensembles against evasion attacks, while incurring in just a relatively small loss of accuracy in the non-adversarial setting.

摘要: 验证机器学习模型在测试时对逃避攻击的稳健性是一个重要的研究问题。不幸的是，以前的工作确定了这个问题对于决策树集成来说是NP-Hard的，因此对于特定的输入必然是棘手的。在本文中，我们识别了一类受限的决策树集成，称为大分布集成，它允许安全验证算法在多项式时间内运行。然后，我们提出了一种新的方法，称为可验证学习，它主张训练这样的受限模型类，这些模型类适合于有效的验证。我们通过设计一种新的训练算法，从标记数据中自动学习大规模决策树集成，从而在多项式时间内实现其安全性验证，从而展示了这种思想的好处。在公开可用的数据集上的实验结果证实，使用我们的算法训练的大范围集成可以在几秒钟内使用标准商业硬件进行验证。此外，大范围的合奏比传统的合奏更能抵御逃避攻击，而在非对抗性的设置中，只会造成相对较小的准确性损失。



## **45. Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection**

与您签约的目标不同：使用间接提示注入损害真实世界的LLM集成应用程序 cs.CR

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2302.12173v2) [paper-pdf](http://arxiv.org/pdf/2302.12173v2)

**Authors**: Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, Mario Fritz

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into various applications. The functionalities of recent LLMs can be flexibly modulated via natural language prompts. This renders them susceptible to targeted adversarial prompting, e.g., Prompt Injection (PI) attacks enable attackers to override original instructions and employed controls. So far, it was assumed that the user is directly prompting the LLM. But, what if it is not the user prompting? We argue that LLM-Integrated Applications blur the line between data and instructions. We reveal new attack vectors, using Indirect Prompt Injection, that enable adversaries to remotely (without a direct interface) exploit LLM-integrated applications by strategically injecting prompts into data likely to be retrieved. We derive a comprehensive taxonomy from a computer security perspective to systematically investigate impacts and vulnerabilities, including data theft, worming, information ecosystem contamination, and other novel security risks. We demonstrate our attacks' practical viability against both real-world systems, such as Bing's GPT-4 powered Chat and code-completion engines, and synthetic applications built on GPT-4. We show how processing retrieved prompts can act as arbitrary code execution, manipulate the application's functionality, and control how and if other APIs are called. Despite the increasing integration and reliance on LLMs, effective mitigations of these emerging threats are currently lacking. By raising awareness of these vulnerabilities and providing key insights into their implications, we aim to promote the safe and responsible deployment of these powerful models and the development of robust defenses that protect users and systems from potential attacks.

摘要: 大型语言模型(LLM)越来越多地被集成到各种应用程序中。最近的LLMS的功能可以通过自然语言提示灵活地调节。这使得它们容易受到有针对性的对抗性提示，例如，提示注入(PI)攻击使攻击者能够覆盖原始指令和采用的控制。到目前为止，假设用户是在直接提示LLM。但是，如果不是用户提示呢？我们认为，LLM集成应用程序模糊了数据和指令之间的界限。我们使用间接提示注入揭示了新的攻击载体，使攻击者能够通过战略性地向可能被检索的数据注入提示来远程(无需直接接口)利用LLM集成的应用程序。我们从计算机安全的角度得出了一个全面的分类，以系统地调查影响和漏洞，包括数据窃取、蠕虫、信息生态系统污染和其他新的安全风险。我们展示了我们的攻击在现实世界系统上的实际可行性，例如Bing的GPT-4聊天和代码完成引擎，以及基于GPT-4的合成应用程序。我们展示了处理检索到的提示如何充当任意代码执行、操纵应用程序的功能以及控制调用其他API的方式和是否调用。尽管对小岛屿发展中国家的整合和依赖日益增加，但目前缺乏对这些新出现的威胁的有效缓解。通过提高对这些漏洞的认识并提供对其影响的关键见解，我们的目标是促进安全和负责任地部署这些强大的模型，并开发强大的防御措施，以保护用户和系统免受潜在攻击。



## **46. Exploring the Connection between Robust and Generative Models**

探索健壮性模型和生成性模型之间的联系 cs.LG

technical report, 6 pages, 6 figures

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2304.04033v2) [paper-pdf](http://arxiv.org/pdf/2304.04033v2)

**Authors**: Senad Beadini, Iacopo Masi

**Abstract**: We offer a study that connects robust discriminative classifiers trained with adversarial training (AT) with generative modeling in the form of Energy-based Models (EBM). We do so by decomposing the loss of a discriminative classifier and showing that the discriminative model is also aware of the input data density. Though a common assumption is that adversarial points leave the manifold of the input data, our study finds out that, surprisingly, untargeted adversarial points in the input space are very likely under the generative model hidden inside the discriminative classifier -- have low energy in the EBM. We present two evidence: untargeted attacks are even more likely than the natural data and their likelihood increases as the attack strength increases. This allows us to easily detect them and craft a novel attack called High-Energy PGD that fools the classifier yet has energy similar to the data set.

摘要: 我们提供了一项研究，将经过对抗性训练(AT)训练的稳健区分分类器与基于能量的模型(EBM)形式的生成性建模相结合。我们通过分解判别分类器的损失来做到这一点，并表明判别模型也知道输入数据的密度。虽然一个普遍的假设是敌对点离开了输入数据的流形，但我们的研究发现，令人惊讶的是，在隐藏在判别分类器中的生成模型下，输入空间中的非目标对抗性点很可能在EBM中具有低能量。我们提出了两个证据：非目标攻击的可能性甚至比自然数据更高，并且随着攻击强度的增加，它们的可能性也会增加。这使我们能够轻松地检测到它们，并创建一种名为高能PGD的新型攻击，它愚弄了分类器，但具有与数据集相似的能量。



## **47. Boosting Adversarial Transferability via Fusing Logits of Top-1 Decomposed Feature**

融合Top-1分解特征的Logit提高对手的可转移性 cs.CV

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2305.01361v2) [paper-pdf](http://arxiv.org/pdf/2305.01361v2)

**Authors**: Juanjuan Weng, Zhiming Luo, Dazhen Lin, Shaozi Li, Zhun Zhong

**Abstract**: Recent research has shown that Deep Neural Networks (DNNs) are highly vulnerable to adversarial samples, which are highly transferable and can be used to attack other unknown black-box models. To improve the transferability of adversarial samples, several feature-based adversarial attack methods have been proposed to disrupt neuron activation in the middle layers. However, current state-of-the-art feature-based attack methods typically require additional computation costs for estimating the importance of neurons. To address this challenge, we propose a Singular Value Decomposition (SVD)-based feature-level attack method. Our approach is inspired by the discovery that eigenvectors associated with the larger singular values decomposed from the middle layer features exhibit superior generalization and attention properties. Specifically, we conduct the attack by retaining the decomposed Top-1 singular value-associated feature for computing the output logits, which are then combined with the original logits to optimize adversarial examples. Our extensive experimental results verify the effectiveness of our proposed method, which can be easily integrated into various baselines to significantly enhance the transferability of adversarial samples for disturbing normally trained CNNs and advanced defense strategies. The source code of this study is available at \textcolor{blue}{\href{https://anonymous.4open.science/r/SVD-SSA-13BF/README.md}{Link}}.

摘要: 最近的研究表明，深度神经网络非常容易受到敌意样本的攻击，这些样本具有很高的可传递性，可以用来攻击其他未知的黑盒模型。为了提高对抗性样本的可转移性，已经提出了几种基于特征的对抗性攻击方法来破坏中间层神经元的激活。然而，当前最先进的基于特征的攻击方法通常需要额外的计算成本来估计神经元的重要性。为了应对这一挑战，我们提出了一种基于奇异值分解(SVD)的特征级攻击方法。我们的方法是受到这样的发现的启发，即与从中间层特征分解的较大奇异值相关的特征向量具有更好的泛化和注意特性。具体地说，我们通过保留分解后的Top-1奇异值关联特征来计算输出逻辑，然后将其与原始逻辑相结合来优化对抗性实例，从而进行攻击。大量的实验结果验证了该方法的有效性，该方法可以很容易地集成到不同的基线中，显著提高对手样本干扰正常训练的CNN和高级防御策略的可转移性。这项研究的源代码可在\textcolor{blue}{\href{https://anonymous.4open.science/r/SVD-SSA-13BF/README.md}{Link}}.上获得



## **48. Diagnostics for Deep Neural Networks with Automated Copy/Paste Attacks**

具有自动复制/粘贴攻击的深度神经网络的诊断 cs.LG

Best paper award at the NeurIPS 2022 ML Safety Workshop --  https://neurips2022.mlsafety.org/

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2211.10024v3) [paper-pdf](http://arxiv.org/pdf/2211.10024v3)

**Authors**: Stephen Casper, Kaivalya Hariharan, Dylan Hadfield-Menell

**Abstract**: This paper considers the problem of helping humans exercise scalable oversight over deep neural networks (DNNs). Adversarial examples can be useful by helping to reveal weaknesses in DNNs, but they can be difficult to interpret or draw actionable conclusions from. Some previous works have proposed using human-interpretable adversarial attacks including copy/paste attacks in which one natural image pasted into another causes an unexpected misclassification. We build on these with two contributions. First, we introduce Search for Natural Adversarial Features Using Embeddings (SNAFUE) which offers a fully automated method for finding copy/paste attacks. Second, we use SNAFUE to red team an ImageNet classifier. We reproduce copy/paste attacks from previous works and find hundreds of other easily-describable vulnerabilities, all without a human in the loop. Code is available at https://github.com/thestephencasper/snafue

摘要: 本文研究了帮助人类对深度神经网络进行可扩展监督的问题。对抗性的例子可以通过帮助揭示DNN中的弱点而有用，但它们可能很难解释或从中得出可操作的结论。以前的一些工作已经提出使用人类可解释的对抗性攻击，包括复制/粘贴攻击，在这种攻击中，一幅自然图像粘贴到另一幅图像中会导致意外的错误分类。我们在这些基础上做出了两项贡献。首先，我们介绍了使用嵌入搜索自然对抗性特征(SNAFUE)，它提供了一种全自动的方法来发现复制/粘贴攻击。其次，我们使用SNAFUE对ImageNet分类器进行分组。我们复制了以前的作品中的复制/粘贴攻击，并发现了数百个其他容易描述的漏洞，所有这些都没有人参与。代码可在https://github.com/thestephencasper/snafue上找到



## **49. Efficient Adversarial Contrastive Learning via Robustness-Aware Coreset Selection**

基于鲁棒性感知CoReset选择的高效对抗性对比学习 cs.LG

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2302.03857v3) [paper-pdf](http://arxiv.org/pdf/2302.03857v3)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL) does not require expensive data annotations but outputs a robust representation that withstands adversarial attacks and also generalizes to a wide range of downstream tasks. However, ACL needs tremendous running time to generate the adversarial variants of all training data, which limits its scalability to large datasets. To speed up ACL, this paper proposes a robustness-aware coreset selection (RCS) method. RCS does not require label information and searches for an informative subset that minimizes a representational divergence, which is the distance of the representation between natural data and their virtual adversarial variants. The vanilla solution of RCS via traversing all possible subsets is computationally prohibitive. Therefore, we theoretically transform RCS into a surrogate problem of submodular maximization, of which the greedy search is an efficient solution with an optimality guarantee for the original problem. Empirically, our comprehensive results corroborate that RCS can speed up ACL by a large margin without significantly hurting the robustness transferability. Notably, to the best of our knowledge, we are the first to conduct ACL efficiently on the large-scale ImageNet-1K dataset to obtain an effective robust representation via RCS.

摘要: 对抗性对比学习(ACL)不需要昂贵的数据标注，但输出了一种稳健的表示，可以抵抗对抗性攻击，并适用于广泛的下游任务。然而，ACL需要大量的运行时间来生成所有训练数据的对抗性变体，这限制了其在大数据集上的可扩展性。为了提高访问控制列表的速度，提出了一种健壮性感知的核心重置选择(RCS)方法。RCS不需要标签信息，并且搜索最小化表示分歧的信息子集，表示分歧是自然数据和它们的虚拟对抗性变体之间的表示距离。通过遍历所有可能子集的RCS的香草解在计算上是令人望而却步的。因此，我们从理论上将RCS问题转化为子模最大化的代理问题，其中贪婪搜索是原问题的最优性保证的有效解。实验结果表明，RCS在不影响健壮性和可转移性的前提下，可以大幅度地提高ACL的速度。值得注意的是，据我们所知，我们是第一个在大规模ImageNet-1K数据集上高效地进行ACL的人，通过RCS获得了有效的健壮表示。



## **50. Single Node Injection Label Specificity Attack on Graph Neural Networks via Reinforcement Learning**

基于强化学习的图神经网络单节点注入标签专用性攻击 cs.LG

**SubmitDate**: 2023-05-04    [abs](http://arxiv.org/abs/2305.02901v1) [paper-pdf](http://arxiv.org/pdf/2305.02901v1)

**Authors**: Dayuan Chen, Jian Zhang, Yuqian Lv, Jinhuan Wang, Hongjie Ni, Shanqing Yu, Zhen Wang, Qi Xuan

**Abstract**: Graph neural networks (GNNs) have achieved remarkable success in various real-world applications. However, recent studies highlight the vulnerability of GNNs to malicious perturbations. Previous adversaries primarily focus on graph modifications or node injections to existing graphs, yielding promising results but with notable limitations. Graph modification attack~(GMA) requires manipulation of the original graph, which is often impractical, while graph injection attack~(GIA) necessitates training a surrogate model in the black-box setting, leading to significant performance degradation due to divergence between the surrogate architecture and the actual victim model. Furthermore, most methods concentrate on a single attack goal and lack a generalizable adversary to develop distinct attack strategies for diverse goals, thus limiting precise control over victim model behavior in real-world scenarios. To address these issues, we present a gradient-free generalizable adversary that injects a single malicious node to manipulate the classification result of a target node in the black-box evasion setting. We propose Gradient-free Generalizable Single Node Injection Attack, namely G$^2$-SNIA, a reinforcement learning framework employing Proximal Policy Optimization. By directly querying the victim model, G$^2$-SNIA learns patterns from exploration to achieve diverse attack goals with extremely limited attack budgets. Through comprehensive experiments over three acknowledged benchmark datasets and four prominent GNNs in the most challenging and realistic scenario, we demonstrate the superior performance of our proposed G$^2$-SNIA over the existing state-of-the-art baselines. Moreover, by comparing G$^2$-SNIA with multiple white-box evasion baselines, we confirm its capacity to generate solutions comparable to those of the best adversaries.

摘要: 图神经网络(GNN)在各种实际应用中取得了显著的成功。然而，最近的研究强调了GNN对恶意扰动的脆弱性。以前的对手主要集中在修改图或向现有图注入节点，产生了有希望的结果，但具有显著的局限性。图修改攻击~(GMA)需要对原始图进行操作，这往往是不切实际的，而图注入攻击~(GIA)需要在黑盒环境下训练代理模型，由于代理体系结构与实际受害者模型之间的差异，导致性能显著下降。此外，大多数方法集中在单个攻击目标上，缺乏一个可概括的对手来针对不同的目标制定不同的攻击策略，从而限制了对现实场景中受害者模型行为的精确控制。为了解决这些问题，我们提出了一种无梯度泛化攻击，在黑盒规避环境下注入单个恶意节点来操纵目标节点的分类结果。本文提出了一种无梯度泛化单节点注入攻击，即G$^2$-SNIA，这是一种基于近邻策略优化的强化学习框架。通过直接查询受害者模型，G$^2$-SNIA从探索中学习模式，以极其有限的攻击预算实现不同的攻击目标。通过在最具挑战性和最现实的场景中对三个公认的基准数据集和四个重要的GNN进行全面的实验，我们证明了我们提出的G$^2$-SNIA比现有的最先进的基线具有更好的性能。此外，通过将G$^2$-SNIA与多个白盒规避基线进行比较，我们证实了它产生与最好的对手相当的解的能力。



