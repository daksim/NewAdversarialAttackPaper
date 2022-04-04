# Latest Adversarial Attack Papers
**update at 2022-04-05 06:31:30**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. FrequencyLowCut Pooling -- Plug & Play against Catastrophic Overfitting**

FredencyLowCut池--针对灾难性过拟合的即插即用 cs.CV

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2204.00491v1)

**Authors**: Julia Grabinski, Steffen Jung, Janis Keuper, Margret Keuper

**Abstracts**: Over the last years, Convolutional Neural Networks (CNNs) have been the dominating neural architecture in a wide range of computer vision tasks. From an image and signal processing point of view, this success might be a bit surprising as the inherent spatial pyramid design of most CNNs is apparently violating basic signal processing laws, i.e. Sampling Theorem in their down-sampling operations. However, since poor sampling appeared not to affect model accuracy, this issue has been broadly neglected until model robustness started to receive more attention. Recent work [17] in the context of adversarial attacks and distribution shifts, showed after all, that there is a strong correlation between the vulnerability of CNNs and aliasing artifacts induced by poor down-sampling operations. This paper builds on these findings and introduces an aliasing free down-sampling operation which can easily be plugged into any CNN architecture: FrequencyLowCut pooling. Our experiments show, that in combination with simple and fast FGSM adversarial training, our hyper-parameter free operator significantly improves model robustness and avoids catastrophic overfitting.

摘要: 在过去的几年里，卷积神经网络(CNN)已经成为在广泛的计算机视觉任务中占主导地位的神经结构。从图像和信号处理的角度来看，这一成功可能有点令人惊讶，因为大多数CNN固有的空间金字塔设计显然违反了基本的信号处理定律，即下采样操作中的采样定理。然而，由于较差的采样似乎不会影响模型的精度，所以这个问题一直被广泛忽视，直到模型的稳健性开始受到更多的关注。最近的工作[17]在对抗性攻击和分布转移的背景下，毕竟表明在CNN的脆弱性和糟糕的下采样操作引起的混叠伪像之间存在很强的相关性。本文以这些发现为基础，介绍了一种无混叠的下采样操作，该操作可以很容易地插入到任何CNN架构中：FrequencyLowCut池。我们的实验表明，结合简单快速的FGSM对抗性训练，我们的超参数自由算子显著地提高了模型的稳健性，并避免了灾难性的过拟合。



## **2. Sensor Data Validation and Driving Safety in Autonomous Driving Systems**

自动驾驶系统中的传感器数据验证与驾驶安全 cs.CV

PhD Thesis, City University of Hong Kong

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2203.16130v2)

**Authors**: Jindi Zhang

**Abstracts**: Autonomous driving technology has drawn a lot of attention due to its fast development and extremely high commercial values. The recent technological leap of autonomous driving can be primarily attributed to the progress in the environment perception. Good environment perception provides accurate high-level environment information which is essential for autonomous vehicles to make safe and precise driving decisions and strategies. Moreover, such progress in accurate environment perception would not be possible without deep learning models and advanced onboard sensors, such as optical sensors (LiDARs and cameras), radars, GPS. However, the advanced sensors and deep learning models are prone to recently invented attack methods. For example, LiDARs and cameras can be compromised by optical attacks, and deep learning models can be attacked by adversarial examples. The attacks on advanced sensors and deep learning models can largely impact the accuracy of the environment perception, posing great threats to the safety and security of autonomous vehicles. In this thesis, we study the detection methods against the attacks on onboard sensors and the linkage between attacked deep learning models and driving safety for autonomous vehicles. To detect the attacks, redundant data sources can be exploited, since information distortions caused by attacks in victim sensor data result in inconsistency with the information from other redundant sources. To study the linkage between attacked deep learning models and driving safety...

摘要: 自动驾驶技术因其快速发展和极高的商业价值而备受关注。最近自动驾驶的技术飞跃主要归功于环境感知的进步。良好的环境感知提供了准确的高层环境信息，这对自动驾驶车辆做出安全、准确的驾驶决策和策略至关重要。此外，如果没有深度学习模型和先进的车载传感器，如光学传感器(激光雷达和照相机)、雷达、全球定位系统，准确的环境感知方面的进展是不可能的。然而，先进的传感器和深度学习模型很容易受到最近发明的攻击方法的影响。例如，激光雷达和摄像头可能会受到光学攻击，深度学习模型可能会受到对抗性例子的攻击。对先进传感器和深度学习模型的攻击会在很大程度上影响环境感知的准确性，对自动驾驶车辆的安全构成极大威胁。在本文中，我们研究了针对车载传感器攻击的检测方法，以及被攻击的深度学习模型与自主车辆驾驶安全之间的联系。为了检测攻击，可以利用冗余数据源，因为攻击导致受害者传感器数据中的信息失真导致与来自其他冗余源的信息不一致。为了研究被攻击的深度学习模型和驾驶安全之间的联系...



## **3. Multi-Expert Adversarial Attack Detection in Person Re-identification Using Context Inconsistency**

基于上下文不一致的人重识别中的多专家对抗攻击检测 cs.CV

Accepted at IEEE ICCV 2021

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2108.09891v2)

**Authors**: Xueping Wang, Shasha Li, Min Liu, Yaonan Wang, Amit K. Roy-Chowdhury

**Abstracts**: The success of deep neural networks (DNNs) has promoted the widespread applications of person re-identification (ReID). However, ReID systems inherit the vulnerability of DNNs to malicious attacks of visually inconspicuous adversarial perturbations. Detection of adversarial attacks is, therefore, a fundamental requirement for robust ReID systems. In this work, we propose a Multi-Expert Adversarial Attack Detection (MEAAD) approach to achieve this goal by checking context inconsistency, which is suitable for any DNN-based ReID systems. Specifically, three kinds of context inconsistencies caused by adversarial attacks are employed to learn a detector for distinguishing the perturbed examples, i.e., a) the embedding distances between a perturbed query person image and its top-K retrievals are generally larger than those between a benign query image and its top-K retrievals, b) the embedding distances among the top-K retrievals of a perturbed query image are larger than those of a benign query image, c) the top-K retrievals of a benign query image obtained with multiple expert ReID models tend to be consistent, which is not preserved when attacks are present. Extensive experiments on the Market1501 and DukeMTMC-ReID datasets show that, as the first adversarial attack detection approach for ReID, MEAAD effectively detects various adversarial attacks and achieves high ROC-AUC (over 97.5%).

摘要: 深度神经网络(DNN)的成功促进了人的再识别(ReID)的广泛应用。然而，REID系统继承了DNN对视觉上不明显的对抗性扰动的恶意攻击的脆弱性。因此，对敌意攻击的检测是健壮的Reid系统的基本要求。在这项工作中，我们提出了一种多专家对抗攻击检测(MEAAD)方法，通过检查上下文不一致性来实现这一目标，该方法适用于任何基于DNN的REID系统。具体地说，利用对抗性攻击引起的三种上下文不一致来学习用于区分扰动示例的检测器，即a)扰动查询人图像与其top-K检索之间的嵌入距离通常大于良性查询图像与其top-K检索之间的嵌入距离，b)扰动查询图像的top-K检索之间的嵌入距离大于良性查询图像的嵌入距离，c)用多个专家Reid模型获得的良性查询图像的top-K检索趋于一致，当存在攻击时不被保存。在Market1501和DukeMTMC-Reid数据集上的大量实验表明，MEAAD作为REID的第一种对抗性攻击检测方法，有效地检测了各种对抗性攻击，并获得了高ROC-AUC(97.5%以上)。



## **4. Effect of Balancing Data Using Synthetic Data on the Performance of Machine Learning Classifiers for Intrusion Detection in Computer Networks**

计算机网络入侵检测中数据均衡对机器学习分类器性能的影响 cs.LG

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2204.00144v1)

**Authors**: Ayesha S. Dina, A. B. Siddique, D. Manivannan

**Abstracts**: Attacks on computer networks have increased significantly in recent days, due in part to the availability of sophisticated tools for launching such attacks as well as thriving underground cyber-crime economy to support it. Over the past several years, researchers in academia and industry used machine learning (ML) techniques to design and implement Intrusion Detection Systems (IDSes) for computer networks. Many of these researchers used datasets collected by various organizations to train ML models for predicting intrusions. In many of the datasets used in such systems, data are imbalanced (i.e., not all classes have equal amount of samples). With unbalanced data, the predictive models developed using ML algorithms may produce unsatisfactory classifiers which would affect accuracy in predicting intrusions. Traditionally, researchers used over-sampling and under-sampling for balancing data in datasets to overcome this problem. In this work, in addition to over-sampling, we also use a synthetic data generation method, called Conditional Generative Adversarial Network (CTGAN), to balance data and study their effect on various ML classifiers. To the best of our knowledge, no one else has used CTGAN to generate synthetic samples to balance intrusion detection datasets. Based on extensive experiments using a widely used dataset NSL-KDD, we found that training ML models on dataset balanced with synthetic samples generated by CTGAN increased prediction accuracy by up to $8\%$, compared to training the same ML models over unbalanced data. Our experiments also show that the accuracy of some ML models trained over data balanced with random over-sampling decline compared to the same ML models trained over unbalanced data.

摘要: 最近几天，针对计算机网络的攻击显著增加，部分原因是可以使用复杂的工具来发动此类攻击，以及蓬勃发展的地下网络犯罪经济为其提供支持。在过去的几年里，学术界和工业界的研究人员使用机器学习(ML)技术来设计和实现计算机网络的入侵检测系统(IDSS)。这些研究人员中的许多人使用不同组织收集的数据集来训练ML模型以预测入侵。在这种系统中使用的许多数据集中，数据是不平衡的(即，不是所有类别都具有相同数量的样本量)。在数据不平衡的情况下，使用ML算法开发的预测模型可能会产生不令人满意的分类器，这将影响预测入侵的准确性。传统上，研究人员使用过采样和欠采样来平衡数据集中的数据，以克服这一问题。在这项工作中，除了过采样，我们还使用了一种称为条件生成对抗网络(CTGAN)的合成数据生成方法来平衡数据并研究它们对各种ML分类器的影响。据我们所知，还没有人使用CTGAN来生成合成样本来平衡入侵检测数据集。基于广泛使用的数据集NSL-KDD的大量实验，我们发现，与在不平衡数据上训练相同的ML模型相比，在CTGAN生成的合成样本平衡的数据集上训练ML模型可以将预测精度提高高达8美元。我们的实验还表明，与在非平衡数据上训练的相同ML模型相比，在均衡数据上训练的某些ML模型的准确率有所下降。



## **5. Reverse Engineering of Imperceptible Adversarial Image Perturbations**

不可感知的对抗性图像扰动的逆向工程 cs.CV

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2203.14145v2)

**Authors**: Yifan Gong, Yuguang Yao, Yize Li, Yimeng Zhang, Xiaoming Liu, Xue Lin, Sijia Liu

**Abstracts**: It has been well recognized that neural network based image classifiers are easily fooled by images with tiny perturbations crafted by an adversary. There has been a vast volume of research to generate and defend such adversarial attacks. However, the following problem is left unexplored: How to reverse-engineer adversarial perturbations from an adversarial image? This leads to a new adversarial learning paradigm--Reverse Engineering of Deceptions (RED). If successful, RED allows us to estimate adversarial perturbations and recover the original images. However, carefully crafted, tiny adversarial perturbations are difficult to recover by optimizing a unilateral RED objective. For example, the pure image denoising method may overfit to minimizing the reconstruction error but hardly preserve the classification properties of the true adversarial perturbations. To tackle this challenge, we formalize the RED problem and identify a set of principles crucial to the RED approach design. Particularly, we find that prediction alignment and proper data augmentation (in terms of spatial transformations) are two criteria to achieve a generalizable RED approach. By integrating these RED principles with image denoising, we propose a new Class-Discriminative Denoising based RED framework, termed CDD-RED. Extensive experiments demonstrate the effectiveness of CDD-RED under different evaluation metrics (ranging from the pixel-level, prediction-level to the attribution-level alignment) and a variety of attack generation methods (e.g., FGSM, PGD, CW, AutoAttack, and adaptive attacks).

摘要: 众所周知，基于神经网络的图像分类器很容易被对手制作的带有微小扰动的图像所愚弄。已经有大量的研究来产生和防御这种对抗性攻击。然而，以下问题仍未得到探索：如何从对抗性图像中逆向设计对抗性扰动？这导致了一种新的对抗性学习范式--欺骗的逆向工程(RED)。如果成功，RED允许我们估计敌方干扰并恢复原始图像。然而，精心设计的微小对抗性干扰很难通过优化单边红色目标来恢复。例如，纯图像去噪方法可能过于适合最小化重建误差，但很难保持真实对抗性扰动的分类性质。为了应对这一挑战，我们将RED问题形式化，并确定一组对RED方法设计至关重要的原则。特别是，我们发现预测对齐和适当的数据增强(在空间变换方面)是实现可推广的RED方法的两个标准。通过将这些RED原理与图像去噪相结合，我们提出了一种新的基于类别区分的RED去噪框架，称为CDD-RED。大量的实验证明了CDD-RED在不同的评估指标(从像素级、预测级到属性级对齐)和各种攻击生成方法(如FGSM、PGD、CW、AutoAttack和自适应攻击)下的有效性。



## **6. Scalable Whitebox Attacks on Tree-based Models**

基于树模型的可伸缩白盒攻击 stat.ML

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2204.00103v1)

**Authors**: Giuseppe Castiglione, Gavin Ding, Masoud Hashemi, Christopher Srinivasa, Ga Wu

**Abstracts**: Adversarial robustness is one of the essential safety criteria for guaranteeing the reliability of machine learning models. While various adversarial robustness testing approaches were introduced in the last decade, we note that most of them are incompatible with non-differentiable models such as tree ensembles. Since tree ensembles are widely used in industry, this reveals a crucial gap between adversarial robustness research and practical applications. This paper proposes a novel whitebox adversarial robustness testing approach for tree ensemble models. Concretely, the proposed approach smooths the tree ensembles through temperature controlled sigmoid functions, which enables gradient descent-based adversarial attacks. By leveraging sampling and the log-derivative trick, the proposed approach can scale up to testing tasks that were previously unmanageable. We compare the approach against both random perturbations and blackbox approaches on multiple public datasets (and corresponding models). Our results show that the proposed method can 1) successfully reveal the adversarial vulnerability of tree ensemble models without causing computational pressure for testing and 2) flexibly balance the search performance and time complexity to meet various testing criteria.

摘要: 对抗稳健性是保证机器学习模型可靠性的基本安全准则之一。虽然在过去的十年中引入了各种对抗性健壮性测试方法，但我们注意到其中大多数方法与不可微模型(如树集成)不兼容。由于树形集成在工业中的广泛应用，这揭示了对抗性稳健性研究与实际应用之间的关键差距。提出了一种新的树集成模型白盒对抗健壮性测试方法。具体地说，该方法通过温度控制的Sigmoid函数来平滑树集合，从而实现基于梯度下降的对抗性攻击。通过利用采样和对数导数技巧，建议的方法可以扩展到测试以前无法管理的任务。我们在多个公共数据集(以及相应的模型)上将该方法与随机扰动和黑盒方法进行了比较。结果表明，该方法能够在不增加测试计算压力的情况下，1)成功地揭示树集成模型的对抗性漏洞，2)灵活地平衡搜索性能和时间复杂度，以满足不同的测试标准。



## **7. Parallel Proof-of-Work with Concrete Bounds**

具有具体界限的并行工作证明 cs.CR

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2204.00034v1)

**Authors**: Patrik Keller, Rainer Böhme

**Abstracts**: Authorization is challenging in distributed systems that cannot rely on the identification of nodes. Proof-of-work offers an alternative gate-keeping mechanism, but its probabilistic nature is incompatible with conventional security definitions. Recent related work establishes concrete bounds for the failure probability of Bitcoin's sequential proof-of-work mechanism. We propose a family of state replication protocols using parallel proof-of-work. Our bottom-up design from an agreement sub-protocol allows us to give concrete bounds for the failure probability in adversarial synchronous networks. After the typical interval of 10 minutes, parallel proof-of-work offers two orders of magnitude more security than sequential proof-of-work. This means that state updates can be sufficiently secure to support commits after one block (i.e., after 10 minutes), removing the risk of double-spending in many applications. We offer guidance on the optimal choice of parameters for a wide range of network and attacker assumptions. Simulations show that the proposed construction is robust against violations of design assumptions.

摘要: 在不能依赖节点标识的分布式系统中，授权是具有挑战性的。工作证明提供了一种替代的把关机制，但其概率性质与传统的安全定义不兼容。最近的相关工作为比特币的序贯验证机制的失效概率建立了具体的界限。我们提出了一类使用并行工作证明的状态复制协议。我们从协议子协议开始的自下而上的设计允许我们给出对抗性同步网络中故障概率的具体界。在典型的10分钟间隔之后，并行工作证明提供的安全性比顺序工作证明高两个数量级。这意味着状态更新可以足够安全，以支持在一个数据块(即10分钟之后)后提交，从而消除了许多应用程序中重复支出的风险。我们为各种网络和攻击者假设提供参数最佳选择的指导。仿真结果表明，所提出的结构对违反设计假设具有较强的鲁棒性。



## **8. Truth Serum: Poisoning Machine Learning Models to Reveal Their Secrets**

真相血清：毒化机器学习模型以揭示其秘密 cs.CR

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2204.00032v1)

**Authors**: Florian Tramèr, Reza Shokri, Ayrton San Joaquin, Hoang Le, Matthew Jagielski, Sanghyun Hong, Nicholas Carlini

**Abstracts**: We introduce a new class of attacks on machine learning models. We show that an adversary who can poison a training dataset can cause models trained on this dataset to leak significant private details of training points belonging to other parties. Our active inference attacks connect two independent lines of work targeting the integrity and privacy of machine learning training data.   Our attacks are effective across membership inference, attribute inference, and data extraction. For example, our targeted attacks can poison <0.1% of the training dataset to boost the performance of inference attacks by 1 to 2 orders of magnitude. Further, an adversary who controls a significant fraction of the training data (e.g., 50%) can launch untargeted attacks that enable 8x more precise inference on all other users' otherwise-private data points.   Our results cast doubts on the relevance of cryptographic privacy guarantees in multiparty computation protocols for machine learning, if parties can arbitrarily select their share of training data.

摘要: 我们在机器学习模型上引入了一类新的攻击。我们表明，可以毒化训练数据集的对手可以导致在该数据集上训练的模型泄露属于其他方的训练点的重要私人细节。我们的主动推理攻击将两个独立的工作线连接在一起，目标是机器学习训练数据的完整性和隐私。我们的攻击在成员关系推理、属性推理和数据提取方面都是有效的。例如，我们的有针对性的攻击可以毒化<0.1%的训练数据集，将推理攻击的性能提高1到2个数量级。此外，控制很大一部分训练数据(例如50%)的对手可以发起无目标攻击，从而能够对所有其他用户的其他私有数据点进行8倍的精确推断。我们的结果对用于机器学习的多方计算协议中的密码隐私保证的相关性提出了怀疑，如果各方可以任意选择他们在训练数据中的份额。



## **9. Training strategy for a lightweight countermeasure model for automatic speaker verification**

一种轻量级说话人自动确认对策模型的训练策略 cs.SD

ASVspoof2021

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.17031v1)

**Authors**: Yen-Lun Liao, Xuanjun Chen, Chung-Che Wang, Jyh-Shing Roger Jang

**Abstracts**: The countermeasure (CM) model is developed to protect Automatic Speaker Verification (ASV) systems from spoof attacks and prevent resulting personal information leakage. Based on practicality and security considerations, the CM model is usually deployed on edge devices, which have more limited computing resources and storage space than cloud- based systems. This work proposes training strategies for a lightweight CM model for ASV, using generalized end- to-end (GE2E) pre-training and adversarial fine-tuning to improve performance, and applying knowledge distillation (KD) to reduce the size of the CM model. In the evalua- tion phase of the ASVspoof 2021 Logical Access task, the lightweight ResNetSE model reaches min t-DCF 0.2695 and EER 3.54%. Compared to the teacher model, the lightweight student model only uses 22.5% of parameters and 21.1% of multiply and accumulate operands of the teacher model.

摘要: 对策(CM)模型是为了保护自动说话人验证(ASV)系统免受欺骗攻击并防止由此导致的个人信息泄露而开发的。基于实用性和安全性考虑，CM模型通常部署在边缘设备上，与基于云的系统相比，边缘设备的计算资源和存储空间更有限。提出了一种面向ASV的轻量级CM模型的训练策略，使用通用端到端(GE2E)预训练和对抗性微调来提高性能，并应用知识蒸馏(KD)来减小CM模型的规模。在ASVspoof2021逻辑访问任务的评估阶段，轻量级ResNetSE模型达到了最小t-DCF值0.2695和EER3.54%.与教师模型相比，轻量级学生模型仅使用了教师模型22.5%的参数和21.1%的乘加操作数。



## **10. Improving Adversarial Transferability via Neuron Attribution-Based Attacks**

通过基于神经元属性的攻击提高对手的可转换性 cs.LG

CVPR 2022

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2204.00008v1)

**Authors**: Jianping Zhang, Weibin Wu, Jen-tse Huang, Yizhan Huang, Wenxuan Wang, Yuxin Su, Michael R. Lyu

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to adversarial examples. It is thus imperative to devise effective attack algorithms to identify the deficiencies of DNNs beforehand in security-sensitive applications. To efficiently tackle the black-box setting where the target model's particulars are unknown, feature-level transfer-based attacks propose to contaminate the intermediate feature outputs of local models, and then directly employ the crafted adversarial samples to attack the target model. Due to the transferability of features, feature-level attacks have shown promise in synthesizing more transferable adversarial samples. However, existing feature-level attacks generally employ inaccurate neuron importance estimations, which deteriorates their transferability. To overcome such pitfalls, in this paper, we propose the Neuron Attribution-based Attack (NAA), which conducts feature-level attacks with more accurate neuron importance estimations. Specifically, we first completely attribute a model's output to each neuron in a middle layer. We then derive an approximation scheme of neuron attribution to tremendously reduce the computation overhead. Finally, we weight neurons based on their attribution results and launch feature-level attacks. Extensive experiments confirm the superiority of our approach to the state-of-the-art benchmarks.

摘要: 深度神经网络(DNN)很容易受到敌意例子的攻击。因此，设计有效的攻击算法来预先识别DNN在安全敏感应用中的缺陷是当务之急。为了有效地处理目标模型细节未知的黑箱环境，基于特征级转移的攻击提出了污染局部模型的中间特征输出，然后直接利用精心制作的敌意样本来攻击目标模型。由于特征的可转移性，特征级攻击在合成更多可转移的对手样本方面显示出了希望。然而，现有的特征级攻击一般采用不准确的神经元重要性估计，这降低了它们的可转移性。为了克服这些缺陷，本文提出了基于神经元属性的攻击(NAA)，它通过更准确的神经元重要性估计来进行特征级别的攻击。具体地说，我们首先将模型的输出完全归因于中间层的每个神经元。然后，我们推导了神经元属性的近似方案，极大地减少了计算开销。最后，我们根据神经元的属性结果对其进行加权，并发起特征级别的攻击。广泛的实验证实了我们的方法对最先进的基准的优越性。



## **11. Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond**

对抗对抗性攻击的稳健除雨：综合基准分析及进一步研究 cs.CV

10 pages, 6 figures, to appear in CVPR 2022

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.16931v1)

**Authors**: Yi Yu, Wenhan Yang, Yap-Peng Tan, Alex C. Kot

**Abstracts**: Rain removal aims to remove rain streaks from images/videos and reduce the disruptive effects caused by rain. It not only enhances image/video visibility but also allows many computer vision algorithms to function properly. This paper makes the first attempt to conduct a comprehensive study on the robustness of deep learning-based rain removal methods against adversarial attacks. Our study shows that, when the image/video is highly degraded, rain removal methods are more vulnerable to the adversarial attacks as small distortions/perturbations become less noticeable or detectable. In this paper, we first present a comprehensive empirical evaluation of various methods at different levels of attacks and with various losses/targets to generate the perturbations from the perspective of human perception and machine analysis tasks. A systematic evaluation of key modules in existing methods is performed in terms of their robustness against adversarial attacks. From the insights of our analysis, we construct a more robust deraining method by integrating these effective modules. Finally, we examine various types of adversarial attacks that are specific to deraining problems and their effects on both human and machine vision tasks, including 1) rain region attacks, adding perturbations only in the rain regions to make the perturbations in the attacked rain images less visible; 2) object-sensitive attacks, adding perturbations only in regions near the given objects. Code is available at https://github.com/yuyi-sd/Robust_Rain_Removal.

摘要: 除雨的目的是去除图像/视频中的雨纹，减少降雨造成的干扰。它不仅增强了图像/视频的可见性，还允许许多计算机视觉算法正常工作。本文首次尝试对基于深度学习的降雨方法对敌方攻击的稳健性进行了全面的研究。我们的研究表明，当图像/视频高度退化时，随着微小的失真/扰动变得不那么明显或可检测到，雨滴去除方法更容易受到敌意攻击。在本文中，我们首先从人的感知和机器分析任务的角度，对不同攻击级别和不同损失/目标的各种方法进行了全面的经验评估，以产生扰动。对现有方法中的关键模块进行了系统的评估，评估了它们对对手攻击的健壮性。根据我们的分析，我们通过整合这些有效的模块来构造一个更健壮的去噪方法。最后，我们研究了针对去重问题的各种类型的对抗性攻击及其对人类和机器视觉任务的影响，包括1)雨区域攻击，仅在雨区域添加扰动，以使被攻击的雨图像中的扰动不那么明显；2)对象敏感攻击，仅在给定对象附近的区域添加扰动。代码可在https://github.com/yuyi-sd/Robust_Rain_Removal.上找到



## **12. Assessing the risk of re-identification arising from an attack on anonymised data**

评估匿名数据遭受攻击后重新识别的风险 cs.LG

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.16921v1)

**Authors**: Anna Antoniou, Giacomo Dossena, Julia MacMillan, Steven Hamblin, David Clifton, Paula Petrone

**Abstracts**: Objective: The use of routinely-acquired medical data for research purposes requires the protection of patient confidentiality via data anonymisation. The objective of this work is to calculate the risk of re-identification arising from a malicious attack to an anonymised dataset, as described below. Methods: We first present an analytical means of estimating the probability of re-identification of a single patient in a k-anonymised dataset of Electronic Health Record (EHR) data. Second, we generalize this solution to obtain the probability of multiple patients being re-identified. We provide synthetic validation via Monte Carlo simulations to illustrate the accuracy of the estimates obtained. Results: The proposed analytical framework for risk estimation provides re-identification probabilities that are in agreement with those provided by simulation in a number of scenarios. Our work is limited by conservative assumptions which inflate the re-identification probability. Discussion: Our estimates show that the re-identification probability increases with the proportion of the dataset maliciously obtained and that it has an inverse relationship with the equivalence class size. Our recursive approach extends the applicability domain to the general case of a multi-patient re-identification attack in an arbitrary k-anonymisation scheme. Conclusion: We prescribe a systematic way to parametrize the k-anonymisation process based on a pre-determined re-identification probability. We observed that the benefits of a reduced re-identification risk that come with increasing k-size may not be worth the reduction in data granularity when one is considering benchmarking the re-identification probability on the size of the portion of the dataset maliciously obtained by the adversary.

摘要: 目的：将常规获取的医疗数据用于研究目的需要通过数据匿名化保护患者的机密性。这项工作的目标是计算恶意攻击引起的对匿名数据集的重新识别风险，如下所述。方法：我们首先提出了一种分析方法来估计电子健康记录(EHR)数据的k匿名数据集中单个患者重新识别的可能性。其次，我们推广这一解决方案，以获得多个患者被重新识别的概率。我们通过蒙特卡罗模拟提供了综合验证，以说明所获得的估计的准确性。结果：建议的风险评估分析框架提供的重新识别概率与模拟在许多情况下提供的概率一致。我们的工作受到保守假设的限制，这些假设夸大了重新识别的概率。讨论：我们的估计表明，重新识别的概率随着恶意获得的数据集的比例而增加，并且与等价类的大小成反比。我们的递归方法将适用范围扩展到任意k-匿名化方案中的多患者重新识别攻击的一般情况。结论：我们规定了一种系统的方法，基于预先确定的重新识别概率来对k-匿名化过程进行参数化。我们观察到，当考虑根据对手恶意获得的部分数据集的大小来对重新识别概率进行基准测试时，随着k大小的增加而来的重新识别风险降低的好处可能不值得减少数据粒度。



## **13. Attack Impact Evaluation by Exact Convexification through State Space Augmentation**

基于状态空间增强的精确凸化攻击效果评估 eess.SY

8 pages

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.16803v1)

**Authors**: Hampei Sasahara, Takashi Tanaka, Henrik Sandberg

**Abstracts**: We address the attack impact evaluation problem for control system security. We formulate the problem as a Markov decision process with a temporally joint chance constraint that forces the adversary to avoid being detected throughout the considered time period. Owing to the joint constraint, the optimal control policy depends not only on the current state but also on the entire history, which leads to the explosion of the search space and makes the problem generally intractable. It is shown that whether an alarm has been triggered or not, in addition to the current state is sufficient for specifying the optimal decision at each time step. Augmentation of the information to the state space induces an equivalent convex optimization problem, which is tractable using standard solvers.

摘要: 我们解决了控制系统安全的攻击影响评估问题。我们将问题描述为一个具有时间联合机会约束的马尔可夫决策过程，迫使对手在所考虑的时间段内避免被发现。由于联合约束，最优控制策略不仅依赖于当前状态，还依赖于整个历史，这导致搜索空间的爆炸性，使问题普遍难以解决。结果表明，除了当前状态外，警报是否已被触发足以确定每个时间步的最优决策。将信息扩充到状态空间将导致等价的凸优化问题，使用标准求解器可以很容易地处理该问题。



## **14. The Block-based Mobile PDE Systems Are Not Secure -- Experimental Attacks**

基于分组的移动PDE系统不安全--实验性攻击 cs.CR

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2203.16349v2)

**Authors**: Niusen Chen, Bo Chen, Weisong Shi

**Abstracts**: Nowadays, mobile devices have been used broadly to store and process sensitive data. To ensure confidentiality of the sensitive data, Full Disk Encryption (FDE) is often integrated in mainstream mobile operating systems like Android and iOS. FDE however cannot defend against coercive attacks in which the adversary can force the device owner to disclose the decryption key. To combat the coercive attacks, Plausibly Deniable Encryption (PDE) is leveraged to plausibly deny the very existence of sensitive data. However, most of the existing PDE systems for mobile devices are deployed at the block layer and suffer from deniability compromises.   Having observed that none of existing works in the literature have experimentally demonstrated the aforementioned compromises, our work bridges this gap by experimentally confirming the deniability compromises of the block-layer mobile PDE systems. We have built a mobile device testbed, which consists of a host computing device and a flash storage device. Additionally, we have deployed both the hidden volume PDE and the steganographic file system at the block layer of the testbed and performed disk forensics to assess potential compromises on the raw NAND flash. Our experimental results confirm it is indeed possible for the adversary to compromise the block-layer PDE systems by accessing the raw NAND flash in practice. We also discuss potential issues when performing such attacks in real world.

摘要: 如今，移动设备已被广泛用于存储和处理敏感数据。为了确保敏感数据的机密性，Android和iOS等主流移动操作系统经常集成全盘加密(FDE)。然而，FDE无法抵御强制攻击，在这种攻击中，对手可以迫使设备所有者披露解密密钥。为了对抗强制攻击，可信可否认加密(PDE)被用来可信地否认敏感数据的存在。然而，大多数现有的移动设备PDE系统都部署在块层，并受到不可否认性妥协的影响。在观察到现有的文献中没有一项工作在实验上证明了上述妥协之后，我们的工作通过实验确认了块层移动PDE系统的否认妥协来弥合了这一差距。我们搭建了一个移动设备试验台，它由主机计算设备和闪存设备组成。此外，我们在测试床的块层部署了隐藏卷PDE和隐写文件系统，并执行了磁盘取证以评估对原始NAND闪存的潜在危害。我们的实验结果证实了攻击者在实践中确实有可能通过访问原始的NAND闪存来危害块层PDE系统。我们还讨论了在现实世界中执行此类攻击时的潜在问题。



## **15. Well-classified Examples are Underestimated in Classification with Deep Neural Networks**

深度神经网络在分类中低估分类好的样本 cs.LG

Accepted by AAAI 2022; 17 pages, 11 figures, 13 tables

**SubmitDate**: 2022-03-31    [paper-pdf](http://arxiv.org/pdf/2110.06537v5)

**Authors**: Guangxiang Zhao, Wenkai Yang, Xuancheng Ren, Lei Li, Yunfang Wu, Xu Sun

**Abstracts**: The conventional wisdom behind learning deep classification models is to focus on bad-classified examples and ignore well-classified examples that are far from the decision boundary. For instance, when training with cross-entropy loss, examples with higher likelihoods (i.e., well-classified examples) contribute smaller gradients in back-propagation. However, we theoretically show that this common practice hinders representation learning, energy optimization, and margin growth. To counteract this deficiency, we propose to reward well-classified examples with additive bonuses to revive their contribution to the learning process. This counterexample theoretically addresses these three issues. We empirically support this claim by directly verifying the theoretical results or significant performance improvement with our counterexample on diverse tasks, including image classification, graph classification, and machine translation. Furthermore, this paper shows that we can deal with complex scenarios, such as imbalanced classification, OOD detection, and applications under adversarial attacks because our idea can solve these three issues. Code is available at: https://github.com/lancopku/well-classified-examples-are-underestimated.

摘要: 学习深度分类模型背后的传统智慧是专注于分类不好的例子，而忽略远离决策边界的分类良好的例子。例如，当使用交叉熵损失进行训练时，具有较高似然的示例(即，分类良好的示例)在反向传播中贡献较小的梯度。然而，我们从理论上表明，这种常见的做法阻碍了表示学习、能量优化和利润率增长。为了弥补这一不足，我们建议用额外的奖金奖励分类良好的例子，以恢复他们对学习过程的贡献。这个反例从理论上解决了这三个问题。我们通过直接验证理论结果或通过我们的反例在包括图像分类、图形分类和机器翻译在内的不同任务上的显著性能改进来经验地支持这一论断。此外，本文还表明，我们的思想可以解决这三个问题，因此我们可以处理复杂的场景，如不平衡分类、面向对象的检测和对手攻击下的应用。代码可从以下网址获得：https://github.com/lancopku/well-classified-examples-are-underestimated.



## **16. Example-based Explanations with Adversarial Attacks for Respiratory Sound Analysis**

呼吸音分析中对抗性攻击的实例解释 cs.SD

Submitted to INTERSPEECH 2022

**SubmitDate**: 2022-03-30    [paper-pdf](http://arxiv.org/pdf/2203.16141v1)

**Authors**: Yi Chang, Zhao Ren, Thanh Tam Nguyen, Wolfgang Nejdl, Björn W. Schuller

**Abstracts**: Respiratory sound classification is an important tool for remote screening of respiratory-related diseases such as pneumonia, asthma, and COVID-19. To facilitate the interpretability of classification results, especially ones based on deep learning, many explanation methods have been proposed using prototypes. However, existing explanation techniques often assume that the data is non-biased and the prediction results can be explained by a set of prototypical examples. In this work, we develop a unified example-based explanation method for selecting both representative data (prototypes) and outliers (criticisms). In particular, we propose a novel application of adversarial attacks to generate an explanation spectrum of data instances via an iterative fast gradient sign method. Such unified explanation can avoid over-generalisation and bias by allowing human experts to assess the model mistakes case by case. We performed a wide range of quantitative and qualitative evaluations to show that our approach generates effective and understandable explanation and is robust with many deep learning models

摘要: 呼吸音分类是远程筛查肺炎、哮喘和新冠肺炎等呼吸系统相关疾病的重要工具。为了便于分类结果的可解释性，特别是基于深度学习的分类结果，已经提出了许多使用原型的解释方法。然而，现有的解释技术往往假设数据是无偏的，预测结果可以通过一组典型例子来解释。在这项工作中，我们开发了一个统一的基于实例的解释方法，用于选择代表性数据(原型)和离群值(批评)。特别是，我们提出了一种新的对抗性攻击的应用，通过迭代快速梯度符号方法来生成数据实例的解释谱。这种统一的解释允许人类专家逐一评估模型错误，从而避免过度概括和偏见。我们进行了广泛的定量和定性评估，表明我们的方法产生了有效和可理解的解释，并对许多深度学习模型具有健壮性



## **17. Fooling the primate brain with minimal, targeted image manipulation**

通过最小的、有针对性的图像处理来愚弄灵长类动物的大脑 q-bio.NC

**SubmitDate**: 2022-03-30    [paper-pdf](http://arxiv.org/pdf/2011.05623v3)

**Authors**: Li Yuan, Will Xiao, Giorgia Dellaferrera, Gabriel Kreiman, Francis E. H. Tay, Jiashi Feng, Margaret S. Livingstone

**Abstracts**: Artificial neural networks (ANNs) are considered the current best models of biological vision. ANNs are the best predictors of neural activity in the ventral stream; moreover, recent work has demonstrated that ANN models fitted to neuronal activity can guide the synthesis of images that drive pre-specified response patterns in small neuronal populations. Despite the success in predicting and steering firing activity, these results have not been connected with perceptual or behavioral changes. Here we propose an array of methods for creating minimal, targeted image perturbations that lead to changes in both neuronal activity and perception as reflected in behavior. We generated 'deceptive images' of human faces, monkey faces, and noise patterns so that they are perceived as a different, pre-specified target category, and measured both monkey neuronal responses and human behavior to these images. We found several effective methods for changing primate visual categorization that required much smaller image change compared to untargeted noise. Our work shares the same goal with adversarial attack, namely the manipulation of images with minimal, targeted noise that leads ANN models to misclassify the images. Our results represent a valuable step in quantifying and characterizing the differences in perturbation robustness of biological and artificial vision.

摘要: 人工神经网络(ANN)被认为是目前最好的生物视觉模型。神经网络是腹侧神经流中神经活动的最佳预测因子；此外，最近的工作表明，适合于神经元活动的神经网络模型可以指导图像的合成，这些图像驱动了小神经元群体中预先指定的反应模式。尽管在预测和指导射击活动方面取得了成功，但这些结果并没有与感知或行为变化联系在一起。在这里，我们提出了一系列方法来创建最小的、有针对性的图像扰动，这些扰动导致神经活动和感知的变化，反映在行为上。我们生成了人脸、猴子脸和噪音模式的“欺骗性图像”，以便它们被视为不同的、预先指定的目标类别，并测量了猴子对这些图像的神经元反应和人类行为。我们发现了几种有效的方法来改变灵长类动物的视觉分类，与非目标噪声相比，这些方法需要的图像改变要小得多。我们的工作与对抗性攻击有着相同的目标，即以最小的目标噪声操纵图像，从而导致ANN模型对图像进行错误分类。我们的结果在量化和表征生物视觉和人工视觉在扰动稳健性方面的差异方面迈出了有价值的一步。



## **18. StyleFool: Fooling Video Classification Systems via Style Transfer**

StyleFool：通过样式转换愚弄视频分类系统 cs.CV

18 pages, 7 figures

**SubmitDate**: 2022-03-30    [paper-pdf](http://arxiv.org/pdf/2203.16000v1)

**Authors**: Yuxin Cao, Xi Xiao, Ruoxi Sun, Derui Wang, Minhui Xue, Sheng Wen

**Abstracts**: Video classification systems are vulnerable to adversarial attacks, which can create severe security problems in video verification. Current black-box attacks need a large number of queries to succeed, resulting in high computational overhead in the process of attack. On the other hand, attacks with restricted perturbations are ineffective against defenses such as denoising or adversarial training. In this paper, we focus on unrestricted perturbations and propose StyleFool, a black-box video adversarial attack via style transfer to fool the video classification system. StyleFool first utilizes color theme proximity to select the best style image, which helps avoid unnatural details in the stylized videos. Meanwhile, the target class confidence is additionally considered in targeted attack to influence the output distribution of the classifier by moving the stylized video closer to or even across the decision boundary. A gradient-free method is then employed to further optimize the adversarial perturbation. We carry out extensive experiments to evaluate StyleFool on two standard datasets, UCF-101 and HMDB-51. The experimental results suggest that StyleFool outperforms the state-of-the-art adversarial attacks in terms of both number of queries and robustness against existing defenses. We identify that 50% of the stylized videos in untargeted attack do not need any query since they can already fool the video classification model. Furthermore, we evaluate the indistinguishability through a user study to show that the adversarial samples of StyleFool look imperceptible to human eyes, despite unrestricted perturbations.

摘要: 视频分类系统容易受到敌意攻击，这会给视频验证带来严重的安全问题。当前的黑盒攻击需要大量的查询才能成功，导致攻击过程中的计算开销很高。另一方面，受限扰动的攻击对诸如去噪或对抗性训练等防御措施无效。本文针对无限制扰动，提出了StyleFool，一种通过风格转移来欺骗视频分类系统的黑盒视频对抗性攻击。StyleFool首先利用颜色主题贴近度来选择最佳风格的图像，这有助于避免风格化视频中不自然的细节。同时，在目标攻击中还考虑了目标类置信度，通过将风格化视频移动到更接近甚至跨越决策边界的位置来影响分类器的输出分布。然后采用无梯度方法进一步优化对抗性扰动。我们在两个标准数据集UCF-101和HMDB-51上进行了大量的实验来评估StyleFool。实验结果表明，StyleFool在查询次数和对现有防御的健壮性方面都优于最先进的对抗性攻击。我们发现，在非定向攻击中，50%的风格化视频不需要任何查询，因为它们已经可以欺骗视频分类模型。此外，我们通过用户研究对StyleFool的不可区分性进行了评估，以表明StyleFool的敌意样本在人眼看来是不可察觉的，尽管存在无限的扰动。



## **19. Recent improvements of ASR models in the face of adversarial attacks**

面对对抗性攻击的ASR模型的最新改进 cs.CR

Submitted to Interspeech 2022

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.16536v1)

**Authors**: Raphael Olivier, Bhiksha Raj

**Abstracts**: Like many other tasks involving neural networks, Speech Recognition models are vulnerable to adversarial attacks. However recent research has pointed out differences between attacks and defenses on ASR models compared to image models. Improving the robustness of ASR models requires a paradigm shift from evaluating attacks on one or a few models to a systemic approach in evaluation. We lay the ground for such research by evaluating on various architectures a representative set of adversarial attacks: targeted and untargeted, optimization and speech processing-based, white-box, black-box and targeted attacks. Our results show that the relative strengths of different attack algorithms vary considerably when changing the model architecture, and that the results of some attacks are not to be blindly trusted. They also indicate that training choices such as self-supervised pretraining can significantly impact robustness by enabling transferable perturbations. We release our source code as a package that should help future research in evaluating their attacks and defenses.

摘要: 像涉及神经网络的许多其他任务一样，语音识别模型容易受到敌意攻击。然而，最近的研究指出，与图像模型相比，ASR模型在攻击和防御方面存在差异。要提高ASR模型的稳健性，需要从评估针对一个或几个模型的攻击转变为评估的系统性方法。我们通过在不同的体系结构上评估一组具有代表性的对抗性攻击：目标攻击和非目标攻击、基于优化和语音处理的攻击、白盒攻击、黑盒攻击和目标攻击，为这类研究奠定了基础。结果表明，随着模型结构的改变，不同攻击算法的相对强度有很大差异，某些攻击的结果不能盲目信任。它们还表明，自我监督预训练等训练选择可以通过实现可转移的扰动来显著影响稳健性。我们将我们的源代码作为一个包发布，这应该有助于未来的研究评估他们的攻击和防御。



## **20. NICGSlowDown: Evaluating the Efficiency Robustness of Neural Image Caption Generation Models**

NICGSlowDown：评估神经图像字幕生成模型的效率和稳健性 cs.CV

This paper is accepted at CVPR2022

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.15859v1)

**Authors**: Simin Chen, Zihe Song, Mirazul Haque, Cong Liu, Wei Yang

**Abstracts**: Neural image caption generation (NICG) models have received massive attention from the research community due to their excellent performance in visual understanding. Existing work focuses on improving NICG model accuracy while efficiency is less explored. However, many real-world applications require real-time feedback, which highly relies on the efficiency of NICG models. Recent research observed that the efficiency of NICG models could vary for different inputs. This observation brings in a new attack surface of NICG models, i.e., An adversary might be able to slightly change inputs to cause the NICG models to consume more computational resources. To further understand such efficiency-oriented threats, we propose a new attack approach, NICGSlowDown, to evaluate the efficiency robustness of NICG models. Our experimental results show that NICGSlowDown can generate images with human-unnoticeable perturbations that will increase the NICG model latency up to 483.86%. We hope this research could raise the community's concern about the efficiency robustness of NICG models.

摘要: 神经图像字幕生成(NICG)模型因其在视觉理解方面的优异性能而受到研究界的广泛关注。现有的工作主要集中在提高NICG模型的精度上，而对效率的研究较少。然而，许多现实世界的应用需要实时反馈，这高度依赖于NICG模型的效率。最近的研究发现，对于不同的投入，NICG模型的效率可能会有所不同。这一观察结果为NICG模型带来了一个新的攻击面，即对手可能能够稍微改变输入以导致NICG模型消耗更多的计算资源。为了进一步理解这种面向效率的威胁，我们提出了一种新的攻击方法NICGSlowDown来评估NICG模型的效率健壮性。我们的实验结果表明，NICGSlowDown可以生成具有人类不可察觉的扰动的图像，这将使NICG模型的延迟增加483.86%。我们希望这项研究能引起社会各界对NICG模型的效率和稳健性的关注。



## **21. Characterizing the adversarial vulnerability of speech self-supervised learning**

表征语音自监督学习的对抗性脆弱性 cs.SD

Accepted by ICASSP 2022

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2111.04330v2)

**Authors**: Haibin Wu, Bo Zheng, Xu Li, Xixin Wu, Hung-yi Lee, Helen Meng

**Abstracts**: A leaderboard named Speech processing Universal PERformance Benchmark (SUPERB), which aims at benchmarking the performance of a shared self-supervised learning (SSL) speech model across various downstream speech tasks with minimal modification of architectures and small amount of data, has fueled the research for speech representation learning. The SUPERB demonstrates speech SSL upstream models improve the performance of various downstream tasks through just minimal adaptation. As the paradigm of the self-supervised learning upstream model followed by downstream tasks arouses more attention in the speech community, characterizing the adversarial robustness of such paradigm is of high priority. In this paper, we make the first attempt to investigate the adversarial vulnerability of such paradigm under the attacks from both zero-knowledge adversaries and limited-knowledge adversaries. The experimental results illustrate that the paradigm proposed by SUPERB is seriously vulnerable to limited-knowledge adversaries, and the attacks generated by zero-knowledge adversaries are with transferability. The XAB test verifies the imperceptibility of crafted adversarial attacks.

摘要: 一个名为语音处理通用性能基准(SUBB)的排行榜推动了语音表示学习的研究，该基准测试旨在以最小的体系结构和少量数据对共享的自我监督学习(SSLE)语音模型在各种下游语音任务中的性能进行基准测试。这一出色的演示表明，语音SSL上行模型仅通过最小的适配就可以提高各种下游任务的性能。随着上游自主学习模型和下游任务的学习范式在语音界引起了更多的关注，表征这种范式的对抗性稳健性是当务之急。在本文中，我们首次尝试研究了这种范式在零知识和有限知识两种攻击下的攻击脆弱性。实验结果表明，Superb提出的范式对有限知识攻击者具有很强的脆弱性，而零知识攻击者产生的攻击具有可移植性。Xab测试验证了精心设计的敌意攻击的隐蔽性。



## **22. Adaptative Perturbation Patterns: Realistic Adversarial Learning for Robust Intrusion Detection**

自适应扰动模式：用于稳健入侵检测的现实对抗性学习 cs.CR

18 pages, 6 tables, 10 figures, Future Internet journal

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.04234v2)

**Authors**: João Vitorino, Nuno Oliveira, Isabel Praça

**Abstracts**: Adversarial attacks pose a major threat to machine learning and to the systems that rely on it. In the cybersecurity domain, adversarial cyber-attack examples capable of evading detection are especially concerning. Nonetheless, an example generated for a domain with tabular data must be realistic within that domain. This work establishes the fundamental constraint levels required to achieve realism and introduces the Adaptative Perturbation Pattern Method (A2PM) to fulfill these constraints in a gray-box setting. A2PM relies on pattern sequences that are independently adapted to the characteristics of each class to create valid and coherent data perturbations. The proposed method was evaluated in a cybersecurity case study with two scenarios: Enterprise and Internet of Things (IoT) networks. Multilayer Perceptron (MLP) and Random Forest (RF) classifiers were created with regular and adversarial training, using the CIC-IDS2017 and IoT-23 datasets. In each scenario, targeted and untargeted attacks were performed against the classifiers, and the generated examples were compared with the original network traffic flows to assess their realism. The obtained results demonstrate that A2PM provides a scalable generation of realistic adversarial examples, which can be advantageous for both adversarial training and attacks.

摘要: 对抗性攻击对机器学习和依赖机器学习的系统构成了重大威胁。在网络安全领域，能够逃避检测的敌意网络攻击实例尤其令人担忧。尽管如此，为包含表格数据的域生成的示例在该域中必须是真实的。这项工作建立了实现真实感所需的基本约束水平，并引入了自适应扰动模式方法(A2PM)来满足灰箱设置中的这些约束。A2PM依赖于独立适应每一类的特征的模式序列来创建有效且一致的数据扰动。该方法在一个网络安全案例研究中进行了评估，其中包含两个场景：企业和物联网(IoT)网络。使用CIC-IDS2017和IoT-23数据集，通过定期和对抗性训练创建了多层感知器(MLP)和随机森林(RF)分类器。在每个场景中，对分类器执行目标攻击和非目标攻击，并将生成的示例与原始网络流量进行比较，以评估其真实性。所获得的结果表明，A2PM提供了可扩展的真实对抗性实例生成，这对于对抗性训练和攻击都是有利的。



## **23. Exploring Frequency Adversarial Attacks for Face Forgery Detection**

基于频率对抗攻击的人脸伪造检测方法研究 cs.CV

Accepted by CVPR2022

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.15674v1)

**Authors**: Shuai Jia, Chao Ma, Taiping Yao, Bangjie Yin, Shouhong Ding, Xiaokang Yang

**Abstracts**: Various facial manipulation techniques have drawn serious public concerns in morality, security, and privacy. Although existing face forgery classifiers achieve promising performance on detecting fake images, these methods are vulnerable to adversarial examples with injected imperceptible perturbations on the pixels. Meanwhile, many face forgery detectors always utilize the frequency diversity between real and fake faces as a crucial clue. In this paper, instead of injecting adversarial perturbations into the spatial domain, we propose a frequency adversarial attack method against face forgery detectors. Concretely, we apply discrete cosine transform (DCT) on the input images and introduce a fusion module to capture the salient region of adversary in the frequency domain. Compared with existing adversarial attacks (e.g. FGSM, PGD) in the spatial domain, our method is more imperceptible to human observers and does not degrade the visual quality of the original images. Moreover, inspired by the idea of meta-learning, we also propose a hybrid adversarial attack that performs attacks in both the spatial and frequency domains. Extensive experiments indicate that the proposed method fools not only the spatial-based detectors but also the state-of-the-art frequency-based detectors effectively. In addition, the proposed frequency attack enhances the transferability across face forgery detectors as black-box attacks.

摘要: 各种面部操纵技术在道德、安全和隐私方面引起了公众的严重关注。虽然现有的人脸伪造分类器在检测虚假图像方面取得了良好的性能，但这些方法很容易受到像素上注入不可察觉扰动的敌对示例的影响。与此同时，许多人脸伪造检测器总是利用真假人脸之间的频率差异作为关键线索。在本文中，我们提出了一种针对人脸伪造检测器的频率对抗攻击方法，而不是在空间域中注入对抗扰动。具体地，对输入图像进行离散余弦变换(DCT)，并引入融合模块在频域中捕捉对手的显著区域。与已有的空域对抗性攻击(如FGSM、PGD)相比，我们的方法对人类观察者来说更不易察觉，并且不会降低原始图像的视觉质量。此外，受元学习思想的启发，我们还提出了一种在空域和频域执行攻击的混合对抗性攻击。大量实验表明，该方法不仅能有效地欺骗基于空间的检测器，而且能有效地欺骗最新的基于频率的检测器。此外，提出的频率攻击增强了黑盒攻击在人脸伪造检测器之间的可传递性。



## **24. Adaptive Image Transformations for Transfer-based Adversarial Attack**

基于传输的对抗性攻击中的自适应图像变换 cs.CV

33 pages, 7 figures, 10 tables

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2111.13844v2)

**Authors**: Zheng Yuan, Jie Zhang, Shiguang Shan

**Abstracts**: Adversarial attacks provide a good way to study the robustness of deep learning models. One category of methods in transfer-based black-box attack utilizes several image transformation operations to improve the transferability of adversarial examples, which is effective, but fails to take the specific characteristic of the input image into consideration. In this work, we propose a novel architecture, called Adaptive Image Transformation Learner (AITL), which incorporates different image transformation operations into a unified framework to further improve the transferability of adversarial examples. Unlike the fixed combinational transformations used in existing works, our elaborately designed transformation learner adaptively selects the most effective combination of image transformations specific to the input image. Extensive experiments on ImageNet demonstrate that our method significantly improves the attack success rates on both normally trained models and defense models under various settings.

摘要: 对抗性攻击为研究深度学习模型的稳健性提供了一种很好的方法。一类基于转移的黑盒攻击方法利用多幅图像变换操作来提高对抗性样本的可转移性，这种方法是有效的，但没有考虑到输入图像的具体特征。在这项工作中，我们提出了一种新的体系结构，称为自适应图像变换学习器(AITL)，它将不同的图像变换操作整合到一个统一的框架中，以进一步提高对抗性例子的可转移性。与现有工作中使用的固定组合变换不同，我们精心设计的变换学习器自适应地选择特定于输入图像的最有效的图像变换组合。在ImageNet上的大量实验表明，该方法在正常训练模型和防御模型上的攻击成功率在各种设置下都有显著提高。



## **25. Treatment Learning Transformer for Noisy Image Classification**

用于含噪图像分类的治疗学习转换器 cs.CV

Preprint. The first version was finished in May 2018

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.15529v1)

**Authors**: Chao-Han Huck Yang, I-Te Danny Hung, Yi-Chieh Liu, Pin-Yu Chen

**Abstracts**: Current top-notch deep learning (DL) based vision models are primarily based on exploring and exploiting the inherent correlations between training data samples and their associated labels. However, a known practical challenge is their degraded performance against "noisy" data, induced by different circumstances such as spurious correlations, irrelevant contexts, domain shift, and adversarial attacks. In this work, we incorporate this binary information of "existence of noise" as treatment into image classification tasks to improve prediction accuracy by jointly estimating their treatment effects. Motivated from causal variational inference, we propose a transformer-based architecture, Treatment Learning Transformer (TLT), that uses a latent generative model to estimate robust feature representations from current observational input for noise image classification. Depending on the estimated noise level (modeled as a binary treatment factor), TLT assigns the corresponding inference network trained by the designed causal loss for prediction. We also create new noisy image datasets incorporating a wide range of noise factors (e.g., object masking, style transfer, and adversarial perturbation) for performance benchmarking. The superior performance of TLT in noisy image classification is further validated by several refutation evaluation metrics. As a by-product, TLT also improves visual salience methods for perceiving noisy images.

摘要: 目前基于深度学习的视觉模型主要是基于探索和利用训练数据样本及其关联标签之间的内在相关性。然而，一个已知的实际挑战是，它们针对不同环境(如伪相关性、无关上下文、域转移和敌意攻击)引起的抗噪声数据的性能下降。在这项工作中，我们将“噪声的存在”这一二值信息作为处理，引入到图像分类任务中，通过联合估计它们的处理效果来提高预测精度。受因果变分推理的启发，我们提出了一种基于变换的结构-处理学习转换器(TLT)，它使用一个潜在的生成模型来估计当前观测输入的稳健特征表示，以用于噪声图像分类。根据估计的噪声水平(建模为二进制处理因子)，TLT分配由设计的因果损失训练的相应的推理网络用于预测。我们还创建了新的噪声图像数据集，其中包含了广泛的噪声因素(例如，对象掩蔽、样式转移和对抗性扰动)，用于性能基准测试。几种反驳评价指标进一步验证了TLT在含噪图像分类中的优越性能。作为一个副产品，TLT还改进了感知噪声图像的视觉显著方法。



## **26. Spotting adversarial samples for speaker verification by neural vocoders**

利用神经声码器识别用于说话人确认的敌意样本 cs.SD

Accepted by ICASSP 2022

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2107.00309v3)

**Authors**: Haibin Wu, Po-chun Hsu, Ji Gao, Shanshan Zhang, Shen Huang, Jian Kang, Zhiyong Wu, Helen Meng, Hung-yi Lee

**Abstracts**: Automatic speaker verification (ASV), one of the most important technology for biometric identification, has been widely adopted in security-critical applications. However, ASV is seriously vulnerable to recently emerged adversarial attacks, yet effective countermeasures against them are limited. In this paper, we adopt neural vocoders to spot adversarial samples for ASV. We use the neural vocoder to re-synthesize audio and find that the difference between the ASV scores for the original and re-synthesized audio is a good indicator for discrimination between genuine and adversarial samples. This effort is, to the best of our knowledge, among the first to pursue such a technical direction for detecting time-domain adversarial samples for ASV, and hence there is a lack of established baselines for comparison. Consequently, we implement the Griffin-Lim algorithm as the detection baseline. The proposed approach achieves effective detection performance that outperforms the baselines in all the settings. We also show that the neural vocoder adopted in the detection framework is dataset-independent. Our codes will be made open-source for future works to do fair comparison.

摘要: 自动说话人验证(ASV)是生物特征识别的重要技术之一，在安全关键应用中得到了广泛的应用。然而，ASV在最近出现的对抗性攻击中非常脆弱，但针对它们的有效对策有限。在本文中，我们采用神经声码器来识别ASV的对抗性样本。我们使用神经声码器对音频进行重新合成，发现原始音频和重新合成音频的ASV分数之间的差异是区分真实和敌对样本的一个很好的指标。据我们所知，这项工作是为检测ASV的时间域对手样本而采取的最早的技术方向之一，因此缺乏用于比较的既定基线。因此，我们实现了Griffin-Lim算法作为检测基线。所提出的方法在所有设置下都取得了优于基线的有效检测性能。我们还证明了检测框架中采用的神经声码器是与数据集无关的。我们的代码将是开源的，以供将来的工作做公平的比较。



## **27. Mel Frequency Spectral Domain Defenses against Adversarial Attacks on Speech Recognition Systems**

针对语音识别系统的敌意攻击的MEL频域防御 eess.AS

This paper is 5 pages long and was submitted to Interspeech 2022

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.15283v1)

**Authors**: Nicholas Mehlman, Anirudh Sreeram, Raghuveer Peri, Shrikanth Narayanan

**Abstracts**: A variety of recent works have looked into defenses for deep neural networks against adversarial attacks particularly within the image processing domain. Speech processing applications such as automatic speech recognition (ASR) are increasingly relying on deep learning models, and so are also prone to adversarial attacks. However, many of the defenses explored for ASR simply adapt the image-domain defenses, which may not provide optimal robustness. This paper explores speech specific defenses using the mel spectral domain, and introduces a novel defense method called 'mel domain noise flooding' (MDNF). MDNF applies additive noise to the mel spectrogram of a speech utterance prior to re-synthesising the audio signal. We test the defenses against strong white-box adversarial attacks such as projected gradient descent (PGD) and Carlini-Wagner (CW) attacks, and show better robustness compared to a randomized smoothing baseline across strong threat models.

摘要: 最近的各种工作都着眼于深度神经网络对对手攻击的防御，特别是在图像处理领域。自动语音识别(ASR)等语音处理应用越来越依赖深度学习模型，因此也容易受到对抗性攻击。然而，为ASR探索的许多防御措施只是简单地采用图像域防御措施，这可能不能提供最佳的稳健性。本文探讨了基于Mel谱域的语音特定防御方法，并提出了一种新的防御方法--“MEL域噪声泛洪”(MDNF)。MDNF在重新合成音频信号之前将加性噪波应用于语音发声的MEL谱图。我们测试了它们对投影梯度下降(PGD)和Carlini-Wagner(CW)攻击等强白盒攻击的防御能力，并与跨强威胁模型的随机平滑基线相比显示出更好的鲁棒性。



## **28. Robust Structured Declarative Classifiers for 3D Point Clouds: Defending Adversarial Attacks with Implicit Gradients**

用于3D点云的稳健结构化声明性分类器：用隐式梯度防御对抗性攻击 cs.CV

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.15245v1)

**Authors**: Kaidong Li, Ziming Zhang, Cuncong Zhong, Guanghui Wang

**Abstracts**: Deep neural networks for 3D point cloud classification, such as PointNet, have been demonstrated to be vulnerable to adversarial attacks. Current adversarial defenders often learn to denoise the (attacked) point clouds by reconstruction, and then feed them to the classifiers as input. In contrast to the literature, we propose a family of robust structured declarative classifiers for point cloud classification, where the internal constrained optimization mechanism can effectively defend adversarial attacks through implicit gradients. Such classifiers can be formulated using a bilevel optimization framework. We further propose an effective and efficient instantiation of our approach, namely, Lattice Point Classifier (LPC), based on structured sparse coding in the permutohedral lattice and 2D convolutional neural networks (CNNs) that is end-to-end trainable. We demonstrate state-of-the-art robust point cloud classification performance on ModelNet40 and ScanNet under seven different attackers. For instance, we achieve 89.51% and 83.16% test accuracy on each dataset under the recent JGBA attacker that outperforms DUP-Net and IF-Defense with PointNet by ~70%. Demo code is available at https://zhang-vislab.github.io.

摘要: 深度神经网络用于三维点云分类，例如PointNet，已经被证明容易受到对手的攻击。当前的敌方防御者通常学习通过重建来对(受攻击的)点云进行去噪，然后将它们作为输入提供给分类器。与文献不同，我们提出了一类稳健的结构化声明式分类器用于点云分类，其中内部约束优化机制可以通过隐式梯度有效地防御敌意攻击。这样的分类器可以使用双层优化框架来表示。在此基础上，提出了一种基于置换面体格子结构稀疏编码的格点分类器(LPC)和端到端可训练的二维卷积神经网络(CNN)。我们在ModelNet40和ScanNet上演示了在七种不同的攻击者下最先进的健壮的点云分类性能。例如，在最近的JGBA攻击下，我们在每个数据集上分别获得了89.51%和83.16%的测试准确率，比使用PointNet的DUP-Net和IF-Defense的测试准确率高出约70%。演示代码可在https://zhang-vislab.github.io.上获得



## **29. Zero-Query Transfer Attacks on Context-Aware Object Detectors**

上下文感知对象检测器的零查询传输攻击 cs.CV

CVPR 2022 Accepted

**SubmitDate**: 2022-03-29    [paper-pdf](http://arxiv.org/pdf/2203.15230v1)

**Authors**: Zikui Cai, Shantanu Rane, Alejandro E. Brito, Chengyu Song, Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury, M. Salman Asif

**Abstracts**: Adversarial attacks perturb images such that a deep neural network produces incorrect classification results. A promising approach to defend against adversarial attacks on natural multi-object scenes is to impose a context-consistency check, wherein, if the detected objects are not consistent with an appropriately defined context, then an attack is suspected. Stronger attacks are needed to fool such context-aware detectors. We present the first approach for generating context-consistent adversarial attacks that can evade the context-consistency check of black-box object detectors operating on complex, natural scenes. Unlike many black-box attacks that perform repeated attempts and open themselves to detection, we assume a "zero-query" setting, where the attacker has no knowledge of the classification decisions of the victim system. First, we derive multiple attack plans that assign incorrect labels to victim objects in a context-consistent manner. Then we design and use a novel data structure that we call the perturbation success probability matrix, which enables us to filter the attack plans and choose the one most likely to succeed. This final attack plan is implemented using a perturbation-bounded adversarial attack algorithm. We compare our zero-query attack against a few-query scheme that repeatedly checks if the victim system is fooled. We also compare against state-of-the-art context-agnostic attacks. Against a context-aware defense, the fooling rate of our zero-query approach is significantly higher than context-agnostic approaches and higher than that achievable with up to three rounds of the few-query scheme.

摘要: 对抗性攻击会扰乱图像，以至于深度神经网络会产生错误的分类结果。在自然多目标场景中防御敌意攻击的一种很有前途的方法是实施上下文一致性检查，其中如果检测到的对象与适当定义的上下文不一致，则怀疑存在攻击。需要更强大的攻击来愚弄这些情景感知检测器。我们提出了第一种生成上下文一致的对抗性攻击的方法，该方法可以逃避操作在复杂的自然场景上的黑盒对象检测器的上下文一致性检查。与许多重复尝试并开放自身以供检测的黑盒攻击不同，我们假定为“零查询”设置，其中攻击者不知道受害者系统的分类决策。首先，我们导出了多个攻击计划，这些攻击计划以上下文一致的方式为受害者对象分配了错误的标签。然后，我们设计并使用了一种新的数据结构，称为扰动成功概率矩阵，它使我们能够过滤攻击计划并选择最有可能成功的攻击计划。该最终攻击计划是使用扰动受限的对抗性攻击算法来实现的。我们将我们的零查询攻击与重复检查受害者系统是否被愚弄的几个查询方案进行了比较。我们还与最先进的上下文不可知攻击进行了比较。与上下文感知防御相比，我们的零查询方法的愚弄率显著高于上下文不可知方法，并且高于最多三轮少查询方案所能达到的水平。



## **30. A Robust Phased Elimination Algorithm for Corruption-Tolerant Gaussian Process Bandits**

一种稳健的容忍腐败高斯过程带的阶段性消除算法 stat.ML

Added references

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2202.01850v2)

**Authors**: Ilija Bogunovic, Zihan Li, Andreas Krause, Jonathan Scarlett

**Abstracts**: We consider the sequential optimization of an unknown, continuous, and expensive to evaluate reward function, from noisy and adversarially corrupted observed rewards. When the corruption attacks are subject to a suitable budget $C$ and the function lives in a Reproducing Kernel Hilbert Space (RKHS), the problem can be posed as corrupted Gaussian process (GP) bandit optimization. We propose a novel robust elimination-type algorithm that runs in epochs, combines exploration with infrequent switching to select a small subset of actions, and plays each action for multiple time instants. Our algorithm, Robust GP Phased Elimination (RGP-PE), successfully balances robustness to corruptions with exploration and exploitation such that its performance degrades minimally in the presence (or absence) of adversarial corruptions. When $T$ is the number of samples and $\gamma_T$ is the maximal information gain, the corruption-dependent term in our regret bound is $O(C \gamma_T^{3/2})$, which is significantly tighter than the existing $O(C \sqrt{T \gamma_T})$ for several commonly-considered kernels. We perform the first empirical study of robustness in the corrupted GP bandit setting, and show that our algorithm is robust against a variety of adversarial attacks.

摘要: 我们考虑了一个未知的、连续的和昂贵的奖励函数的序列优化问题，这些奖励函数来自于噪声和恶意破坏的观测奖励。当腐败攻击受到适当的预算$C$并且函数位于再生核-希尔伯特空间(RKHS)中时，问题可以被假设为被破坏的高斯过程(GP)强盗优化。我们提出了一种新的健壮的消去型算法，该算法运行在纪元上，结合探索和不频繁的切换来选择一小部分动作，并在多个时刻播放每个动作。我们的算法，稳健GP分阶段消除(RGP-PE)，成功地平衡了对腐败的稳健性与探索和利用，使得它的性能在存在(或不存在)对抗腐败的情况下降幅最小。当$T$是样本数，$\Gamma_T$是最大信息增益时，我们遗憾界中的腐败依赖项是$O(C\Gamma_T^{3/2})$，这比现有的几个常用核函数的$O(C\Sqrt{T\Gamma_T})$要紧得多。我们首次在被破坏的GP盗贼环境下进行了健壮性的实验研究，并证明了我们的算法对各种敌意攻击具有健壮性。



## **31. Neurosymbolic hybrid approach to driver collision warning**

驾驶员碰撞预警的神经符号混合方法 cs.CV

SPIE Defense and Commercial Sensing 2022

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2203.15076v1)

**Authors**: Kyongsik Yun, Thomas Lu, Alexander Huyen, Patrick Hammer, Pei Wang

**Abstracts**: There are two main algorithmic approaches to autonomous driving systems: (1) An end-to-end system in which a single deep neural network learns to map sensory input directly into appropriate warning and driving responses. (2) A mediated hybrid recognition system in which a system is created by combining independent modules that detect each semantic feature. While some researchers believe that deep learning can solve any problem, others believe that a more engineered and symbolic approach is needed to cope with complex environments with less data. Deep learning alone has achieved state-of-the-art results in many areas, from complex gameplay to predicting protein structures. In particular, in image classification and recognition, deep learning models have achieved accuracies as high as humans. But sometimes it can be very difficult to debug if the deep learning model doesn't work. Deep learning models can be vulnerable and are very sensitive to changes in data distribution. Generalization can be problematic. It's usually hard to prove why it works or doesn't. Deep learning models can also be vulnerable to adversarial attacks. Here, we combine deep learning-based object recognition and tracking with an adaptive neurosymbolic network agent, called the Non-Axiomatic Reasoning System (NARS), that can adapt to its environment by building concepts based on perceptual sequences. We achieved an improved intersection-over-union (IOU) object recognition performance of 0.65 in the adaptive retraining model compared to IOU 0.31 in the COCO data pre-trained model. We improved the object detection limits using RADAR sensors in a simulated environment, and demonstrated the weaving car detection capability by combining deep learning-based object detection and tracking with a neurosymbolic model.

摘要: 自动驾驶系统有两种主要的算法方法：(1)端到端系统，其中单个深层神经网络学习将感觉输入直接映射到适当的警告和驾驶响应。(2)中介混合识别系统，其中通过组合检测每个语义特征的独立模块来创建系统。虽然一些研究人员认为深度学习可以解决任何问题，但另一些人认为，需要一种更具工程化和符号化的方法来应对数据较少的复杂环境。仅深度学习就在许多领域取得了最先进的结果，从复杂的游戏到预测蛋白质结构。特别是，在图像分类和识别中，深度学习模型已经达到了与人类一样高的准确率。但有时，如果深度学习模式不起作用，调试可能会非常困难。深度学习模型可能很脆弱，并且对数据分布的变化非常敏感。泛化可能是有问题的。通常很难证明它为什么有效或无效。深度学习模型也很容易受到对手的攻击。在这里，我们将基于深度学习的目标识别和跟踪与自适应神经符号网络代理相结合，称为非公理推理系统(NARS)，它可以通过基于感知序列构建概念来适应其环境。与COCO数据预训练模型中的IOU 0.31相比，自适应再训练模型的IOU识别性能提高了0.65。在模拟环境中改进了雷达传感器对目标的检测极限，并将基于深度学习的目标检测与跟踪与神经符号模型相结合，展示了编织小车的检测能力。



## **32. Poisoning and Backdooring Contrastive Learning**

中毒与倒退对比学习 cs.LG

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2106.09667v2)

**Authors**: Nicholas Carlini, Andreas Terzis

**Abstracts**: Multimodal contrastive learning methods like CLIP train on noisy and uncurated training datasets. This is cheaper than labeling datasets manually, and even improves out-of-distribution robustness. We show that this practice makes backdoor and poisoning attacks a significant threat. By poisoning just 0.01% of a dataset (e.g., just 300 images of the 3 million-example Conceptual Captions dataset), we can cause the model to misclassify test images by overlaying a small patch. Targeted poisoning attacks, whereby the model misclassifies a particular test input with an adversarially-desired label, are even easier requiring control of 0.0001% of the dataset (e.g., just three out of the 3 million images). Our attacks call into question whether training on noisy and uncurated Internet scrapes is desirable.

摘要: 多模式对比学习方法，如CLIP，在噪声和未经过处理的训练数据集上进行训练。这比手动标记数据集更便宜，甚至提高了分布外的健壮性。我们表明，这种做法使后门和中毒攻击成为一种重大威胁。通过只毒化0.01%的数据集(例如，仅300万个概念字幕数据集的300张图像)，我们可以通过叠加一个小补丁来导致模型对测试图像进行错误分类。有针对性的中毒攻击，即模型将特定的测试输入与对手希望的标签进行错误分类，甚至更容易，需要控制0.0001的数据集(例如，仅控制300万张图像中的3张)。我们的攻击让人质疑，对嘈杂和未经管理的互联网擦伤进行培训是否可取。



## **33. Boosting Black-Box Adversarial Attacks with Meta Learning**

利用元学习增强黑箱对抗攻击 cs.LG

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2203.14607v1)

**Authors**: Junjie Fu, Jian Sun, Gang Wang

**Abstracts**: Deep neural networks (DNNs) have achieved remarkable success in diverse fields. However, it has been demonstrated that DNNs are very vulnerable to adversarial examples even in black-box settings. A large number of black-box attack methods have been proposed to in the literature. However, those methods usually suffer from low success rates and large query counts, which cannot fully satisfy practical purposes. In this paper, we propose a hybrid attack method which trains meta adversarial perturbations (MAPs) on surrogate models and performs black-box attacks by estimating gradients of the models. Our method uses the meta adversarial perturbation as an initialization and subsequently trains any black-box attack method for several epochs. Furthermore, the MAPs enjoy favorable transferability and universality, in the sense that they can be employed to boost performance of other black-box adversarial attack methods. Extensive experiments demonstrate that our method can not only improve the attack success rates, but also reduces the number of queries compared to other methods.

摘要: 深度神经网络(DNN)在各个领域都取得了显著的成功。然而，已经证明，即使在黑盒环境中，DNN也非常容易受到敌意示例的攻击。文献中提出了大量的黑盒攻击方法。然而，这些方法通常存在成功率低、查询次数多等问题，不能完全满足实际应用的需要。在本文中，我们提出了一种混合攻击方法，该方法在代理模型上训练元对抗扰动(MAP)，并通过估计模型的梯度来执行黑盒攻击。我们的方法使用元对抗扰动作为初始化，并随后在几个时期训练任何黑盒攻击方法。此外，映射具有良好的可转移性和普适性，可以用来提高其他黑盒对抗攻击方法的性能。大量实验表明，与其他方法相比，该方法不仅可以提高攻击成功率，而且可以减少查询次数。



## **34. Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer**

保护面部隐私：通过风格稳健的化妆传输生成敌意身份面具 cs.CV

Accepted by CVPR2022. Code is available at  https://github.com/CGCL-codes/AMT-GAN

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2203.03121v2)

**Authors**: Shengshan Hu, Xiaogeng Liu, Yechao Zhang, Minghui Li, Leo Yu Zhang, Hai Jin, Libing Wu

**Abstracts**: While deep face recognition (FR) systems have shown amazing performance in identification and verification, they also arouse privacy concerns for their excessive surveillance on users, especially for public face images widely spread on social networks. Recently, some studies adopt adversarial examples to protect photos from being identified by unauthorized face recognition systems. However, existing methods of generating adversarial face images suffer from many limitations, such as awkward visual, white-box setting, weak transferability, making them difficult to be applied to protect face privacy in reality. In this paper, we propose adversarial makeup transfer GAN (AMT-GAN), a novel face protection method aiming at constructing adversarial face images that preserve stronger black-box transferability and better visual quality simultaneously. AMT-GAN leverages generative adversarial networks (GAN) to synthesize adversarial face images with makeup transferred from reference images. In particular, we introduce a new regularization module along with a joint training strategy to reconcile the conflicts between the adversarial noises and the cycle consistence loss in makeup transfer, achieving a desirable balance between the attack strength and visual changes. Extensive experiments verify that compared with state of the arts, AMT-GAN can not only preserve a comfortable visual quality, but also achieve a higher attack success rate over commercial FR APIs, including Face++, Aliyun, and Microsoft.

摘要: 虽然深度人脸识别系统在身份识别和验证方面表现出了惊人的性能，但它们也因为对用户的过度监控而引起了隐私问题，特别是对社交网络上广泛传播的公共人脸图像。最近，一些研究采用对抗性的例子来保护照片不被未经授权的人脸识别系统识别。然而，现有的生成敌意人脸图像的方法存在着视觉上的笨拙、白盒设置、可移植性较弱等诸多局限性，难以应用于现实中的人脸隐私保护。在本文中，我们提出了一种新的人脸保护方法--对抗性化妆转移GAN(AMT-GAN)，该方法旨在构建对抗性人脸图像，同时保持较强的黑盒可转移性和较好的视觉质量。AMT-GAN利用生成性对抗性网络(GAN)来合成带有参考图像化妆的对抗性人脸图像。特别是，我们引入了一种新的正则化模型和一种联合训练策略来协调化妆转移中对抗性噪声和循环一致性损失之间的冲突，实现了攻击强度和视觉变化之间的理想平衡。广泛的实验证明，与现有技术相比，AMT-GAN不仅可以保持舒适的视觉质量，而且比Face++、阿里云、微软等商用FR API具有更高的攻击成功率。



## **35. Practical Evaluation of Adversarial Robustness via Adaptive Auto Attack**

自适应自动攻击对敌方健壮性的实用评估 cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2203.05154v3)

**Authors**: Ye Liu, Yaya Cheng, Lianli Gao, Xianglong Liu, Qilong Zhang, Jingkuan Song

**Abstracts**: Defense models against adversarial attacks have grown significantly, but the lack of practical evaluation methods has hindered progress. Evaluation can be defined as looking for defense models' lower bound of robustness given a budget number of iterations and a test dataset. A practical evaluation method should be convenient (i.e., parameter-free), efficient (i.e., fewer iterations) and reliable (i.e., approaching the lower bound of robustness). Towards this target, we propose a parameter-free Adaptive Auto Attack (A$^3$) evaluation method which addresses the efficiency and reliability in a test-time-training fashion. Specifically, by observing that adversarial examples to a specific defense model follow some regularities in their starting points, we design an Adaptive Direction Initialization strategy to speed up the evaluation. Furthermore, to approach the lower bound of robustness under the budget number of iterations, we propose an online statistics-based discarding strategy that automatically identifies and abandons hard-to-attack images. Extensive experiments demonstrate the effectiveness of our A$^3$. Particularly, we apply A$^3$ to nearly 50 widely-used defense models. By consuming much fewer iterations than existing methods, i.e., $1/10$ on average (10$\times$ speed up), we achieve lower robust accuracy in all cases. Notably, we won $\textbf{first place}$ out of 1681 teams in CVPR 2021 White-box Adversarial Attacks on Defense Models competitions with this method. Code is available at: $\href{https://github.com/liuye6666/adaptive_auto_attack}{https://github.com/liuye6666/adaptive\_auto\_attack}$

摘要: 针对对抗性攻击的防御模型显著增长，但缺乏实用的评估方法阻碍了进展。评估可以被定义为在给定预算迭代次数和测试数据集的情况下寻找防御模型的健壮性下限。一种实用的评估方法应该是方便的(即无参数的)、有效的(即迭代次数较少的)和可靠的(即接近稳健性的下界)。针对这一目标，我们提出了一种无参数的自适应自动攻击评估方法(A$^3$)，该方法以测试时间训练的方式来解决效率和可靠性问题。具体地说，通过观察特定防御模型的对抗性样本在起点上遵循一定的规律，我们设计了一种自适应方向初始化策略来加快评估速度。此外，为了在预算迭代次数下逼近健壮性的下界，我们提出了一种基于在线统计的丢弃策略，自动识别和丢弃不易攻击的图像。广泛的实验证明了我们的澳元^3元的有效性。特别是，我们将澳元^3美元应用于近50种广泛使用的防御模型。通过比现有方法消耗更少的迭代次数，即平均$1/10$(10$\倍$加速)，我们在所有情况下都获得较低的健壮性精度。值得注意的是，我们用这种方法在CVPR 2021白盒对抗性攻击防御模型比赛中赢得了1681支队伍中的$\extbf{第一名}$。代码可从以下网址获得：$\href{https://github.com/liuye6666/adaptive_auto_attack}{https://github.com/liuye6666/adaptive\_auto\_attack}$



## **36. Essential Features: Content-Adaptive Pixel Discretization to Improve Model Robustness to Adaptive Adversarial Attacks**

基本特征：内容自适应像素离散化，以提高模型对自适应攻击的稳健性 cs.CV

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2012.01699v3)

**Authors**: Ryan Feng, Wu-chi Feng, Atul Prakash

**Abstracts**: Preprocessing defenses such as pixel discretization are appealing to remove adversarial attacks due to their simplicity. However, they have been shown to be ineffective except on simple datasets such as MNIST. We hypothesize that existing discretization approaches failed because using a fixed codebook for the entire dataset limits their ability to balance image representation and codeword separability. We propose a per-image adaptive preprocessing defense called Essential Features, which first applies adaptive blurring to push perturbed pixel values back to their original value and then discretizes the image to an image-adaptive codebook to reduce the color space. Essential Features thus constrains the attack space by forcing the adversary to perturb large regions both locally and color-wise for its effects to survive the preprocessing. Against adaptive attacks, we find that our approach increases the $L_2$ and $L_\infty$ robustness on higher resolution datasets.

摘要: 像像素离散化这样的预处理防御由于其简单性而被用来消除对抗性攻击。然而，它们已经被证明是无效的，除非在MNIST这样的简单数据集上。我们假设现有的离散化方法失败了，因为对整个数据集使用固定的码本限制了它们平衡图像表示和码字可分性的能力。我们提出了一种称为基本特征的逐图像自适应预处理防御方法，它首先应用自适应模糊将扰动的像素值恢复到其原始值，然后将图像离散为图像自适应码本以减少颜色空间。因此，基本特征通过迫使对手在局部和颜色上扰乱大片区域来限制攻击空间，以使其效果在预处理过程中幸存下来。对于自适应攻击，我们发现我们的方法在更高分辨率的数据集上提高了$L_2$和$L_INFTY$的稳健性。



## **37. Adversarial Representation Sharing: A Quantitative and Secure Collaborative Learning Framework**

对抗性表征共享：一种量化、安全的协作学习框架 cs.CR

**SubmitDate**: 2022-03-27    [paper-pdf](http://arxiv.org/pdf/2203.14299v1)

**Authors**: Jikun Chen, Feng Qiang, Na Ruan

**Abstracts**: The performance of deep learning models highly depends on the amount of training data. It is common practice for today's data holders to merge their datasets and train models collaboratively, which yet poses a threat to data privacy. Different from existing methods such as secure multi-party computation (MPC) and federated learning (FL), we find representation learning has unique advantages in collaborative learning due to the lower communication overhead and task-independency. However, data representations face the threat of model inversion attacks. In this article, we formally define the collaborative learning scenario, and quantify data utility and privacy. Then we present ARS, a collaborative learning framework wherein users share representations of data to train models, and add imperceptible adversarial noise to data representations against reconstruction or attribute extraction attacks. By evaluating ARS in different contexts, we demonstrate that our mechanism is effective against model inversion attacks, and achieves a balance between privacy and utility. The ARS framework has wide applicability. First, ARS is valid for various data types, not limited to images. Second, data representations shared by users can be utilized in different tasks. Third, the framework can be easily extended to the vertical data partitioning scenario.

摘要: 深度学习模型的性能在很大程度上取决于训练数据量。对于今天的数据持有者来说，合并他们的数据集和协作训练模型是一种常见的做法，但这对数据隐私构成了威胁。不同于现有的安全多方计算(MPC)和联合学习(FL)等方法，我们发现表征学习在协作学习中具有独特的优势，因为它具有较低的通信开销和任务无关性。然而，数据表示面临着模型反转攻击的威胁。在本文中，我们正式定义了协作学习场景，并量化了数据效用和隐私。然后，我们提出了一种协作学习框架ARS，在该框架中，用户共享数据表示以训练模型，并在数据表示中添加不可察觉的对抗性噪声以抵抗重构或属性提取攻击。通过在不同环境下对ARS的评估，我们证明了该机制对模型反转攻击是有效的，并在隐私和效用之间取得了平衡。ARS框架具有广泛的适用性。首先，ARS适用于各种数据类型，而不限于图像。其次，用户共享的数据表示可以在不同的任务中使用。第三，该框架可以很容易地扩展到垂直数据分区场景。



## **38. Rebuild and Ensemble: Exploring Defense Against Text Adversaries**

重建与整合：探索对文本对手的防御 cs.CL

work in progress

**SubmitDate**: 2022-03-27    [paper-pdf](http://arxiv.org/pdf/2203.14207v1)

**Authors**: Linyang Li, Demin Song, Jiehang Zeng, Ruotian Ma, Xipeng Qiu

**Abstracts**: Adversarial attacks can mislead strong neural models; as such, in NLP tasks, substitution-based attacks are difficult to defend. Current defense methods usually assume that the substitution candidates are accessible, which cannot be widely applied against adversarial attacks unless knowing the mechanism of the attacks. In this paper, we propose a \textbf{Rebuild and Ensemble} Framework to defend against adversarial attacks in texts without knowing the candidates. We propose a rebuild mechanism to train a robust model and ensemble the rebuilt texts during inference to achieve good adversarial defense results. Experiments show that our method can improve accuracy under the current strong attack methods.

摘要: 对抗性攻击会误导强大的神经模型；因此，在NLP任务中，基于替换的攻击很难防御。目前的防御方法通常假设替换候选者是可访问的，除非了解攻击的机制，否则不能广泛应用于对抗攻击。在本文中，我们提出了一个在不知道候选者的情况下防御文本中的敌意攻击的文本重建与集成框架。我们提出了一种重建机制来训练一个健壮的模型，并在推理过程中对重建的文本进行集成，以达到良好的对抗防御效果。实验表明，在现有的强攻击方法下，我们的方法能够提高准确率。



## **39. HINT: Hierarchical Neuron Concept Explainer**

提示：层次化神经元概念解释器 cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-27    [paper-pdf](http://arxiv.org/pdf/2203.14196v1)

**Authors**: Andong Wang, Wei-Ning Lee, Xiaojuan Qi

**Abstracts**: To interpret deep networks, one main approach is to associate neurons with human-understandable concepts. However, existing methods often ignore the inherent relationships of different concepts (e.g., dog and cat both belong to animals), and thus lose the chance to explain neurons responsible for higher-level concepts (e.g., animal). In this paper, we study hierarchical concepts inspired by the hierarchical cognition process of human beings. To this end, we propose HIerarchical Neuron concepT explainer (HINT) to effectively build bidirectional associations between neurons and hierarchical concepts in a low-cost and scalable manner. HINT enables us to systematically and quantitatively study whether and how the implicit hierarchical relationships of concepts are embedded into neurons, such as identifying collaborative neurons responsible to one concept and multimodal neurons for different concepts, at different semantic levels from concrete concepts (e.g., dog) to more abstract ones (e.g., animal). Finally, we verify the faithfulness of the associations using Weakly Supervised Object Localization, and demonstrate its applicability in various tasks such as discovering saliency regions and explaining adversarial attacks. Code is available on https://github.com/AntonotnaWang/HINT.

摘要: 要解释深层网络，一种主要的方法是将神经元与人类可以理解的概念联系起来。然而，现有的方法往往忽略了不同概念之间的内在联系(例如，狗和猫都属于动物)，从而失去了解释负责更高层次概念(例如，动物)的神经元的机会。受人类层次化认知过程的启发，本文研究了层次化概念。为此，我们提出了层次化神经元概念解释器(HINT)，以低成本和可扩展的方式有效地建立神经元和层次化概念之间的双向关联。提示使我们能够系统和定量地研究概念的隐含层次关系是否以及如何嵌入到神经元中，例如识别负责一个概念的协作神经元和负责不同概念的多通道神经元，从具体的概念(如狗)到更抽象的概念(如动物)，在不同的语义水平上识别。最后，我们使用弱监督对象定位验证了关联的可信性，并证明了它在发现显著区域和解释对抗性攻击等各种任务中的适用性。代码可在https://github.com/AntonotnaWang/HINT.上找到



## **40. How to Robustify Black-Box ML Models? A Zeroth-Order Optimization Perspective**

如何将黑箱ML模型规模化？零阶最优化视角 cs.LG

Accepted as ICLR'22 Spotlight Paper

**SubmitDate**: 2022-03-27    [paper-pdf](http://arxiv.org/pdf/2203.14195v1)

**Authors**: Yimeng Zhang, Yuguang Yao, Jinghan Jia, Jinfeng Yi, Mingyi Hong, Shiyu Chang, Sijia Liu

**Abstracts**: The lack of adversarial robustness has been recognized as an important issue for state-of-the-art machine learning (ML) models, e.g., deep neural networks (DNNs). Thereby, robustifying ML models against adversarial attacks is now a major focus of research. However, nearly all existing defense methods, particularly for robust training, made the white-box assumption that the defender has the access to the details of an ML model (or its surrogate alternatives if available), e.g., its architectures and parameters. Beyond existing works, in this paper we aim to address the problem of black-box defense: How to robustify a black-box model using just input queries and output feedback? Such a problem arises in practical scenarios, where the owner of the predictive model is reluctant to share model information in order to preserve privacy. To this end, we propose a general notion of defensive operation that can be applied to black-box models, and design it through the lens of denoised smoothing (DS), a first-order (FO) certified defense technique. To allow the design of merely using model queries, we further integrate DS with the zeroth-order (gradient-free) optimization. However, a direct implementation of zeroth-order (ZO) optimization suffers a high variance of gradient estimates, and thus leads to ineffective defense. To tackle this problem, we next propose to prepend an autoencoder (AE) to a given (black-box) model so that DS can be trained using variance-reduced ZO optimization. We term the eventual defense as ZO-AE-DS. In practice, we empirically show that ZO-AE- DS can achieve improved accuracy, certified robustness, and query complexity over existing baselines. And the effectiveness of our approach is justified under both image classification and image reconstruction tasks. Codes are available at https://github.com/damon-demon/Black-Box-Defense.

摘要: 缺乏对抗性已经被认为是最先进的机器学习(ML)模型的一个重要问题，例如，深度神经网络(DNN)。因此，增强ML模型的抗敌意攻击能力是目前研究的重点。然而，几乎所有现有的防御方法，特别是对于稳健训练，都建立在白盒假设下，即防御者可以访问ML模型(或其替代方案，如果可用)的细节，例如其体系结构和参数。在已有工作的基础上，本文旨在解决黑盒防御问题：如何仅使用输入查询和输出反馈来增强黑盒模型的健壮性？这样的问题出现在实际场景中，其中预测模型的所有者不愿共享模型信息以保护隐私。为此，我们提出了适用于黑盒模型的防御操作的一般概念，并通过一阶(FO)认证的防御技术去噪平滑(DS)的透镜来设计它。为了允许只使用模型查询的设计，我们进一步将DS与零阶(无梯度)优化相结合。然而，直接实现零阶(ZO)优化会遇到梯度估计的高方差，从而导致无效防御。为了解决这个问题，我们接下来建议在给定的(黑盒)模型中预先设置一个自动编码器，以便可以使用方差减少的ZO优化来训练DS。我们称最终的防御为ZO-AE-DS。在实践中，我们的经验表明，ZO-AE-DS可以在现有基线上获得更高的准确率、经过验证的健壮性和查询复杂性。并在图像分类和图像重建任务中验证了该方法的有效性。有关代码，请访问https://github.com/damon-demon/Black-Box-Defense.



## **41. PiDAn: A Coherence Optimization Approach for Backdoor Attack Detection and Mitigation in Deep Neural Networks**

PIDAN：一种深度神经网络后门攻击检测与防御的一致性优化方法 cs.LG

**SubmitDate**: 2022-03-26    [paper-pdf](http://arxiv.org/pdf/2203.09289v2)

**Authors**: Yue Wang, Wenqing Li, Esha Sarkar, Muhammad Shafique, Michail Maniatakos, Saif Eddin Jabari

**Abstracts**: Backdoor attacks impose a new threat in Deep Neural Networks (DNNs), where a backdoor is inserted into the neural network by poisoning the training dataset, misclassifying inputs that contain the adversary trigger. The major challenge for defending against these attacks is that only the attacker knows the secret trigger and the target class. The problem is further exacerbated by the recent introduction of "Hidden Triggers", where the triggers are carefully fused into the input, bypassing detection by human inspection and causing backdoor identification through anomaly detection to fail. To defend against such imperceptible attacks, in this work we systematically analyze how representations, i.e., the set of neuron activations for a given DNN when using the training data as inputs, are affected by backdoor attacks. We propose PiDAn, an algorithm based on coherence optimization purifying the poisoned data. Our analysis shows that representations of poisoned data and authentic data in the target class are still embedded in different linear subspaces, which implies that they show different coherence with some latent spaces. Based on this observation, the proposed PiDAn algorithm learns a sample-wise weight vector to maximize the projected coherence of weighted samples, where we demonstrate that the learned weight vector has a natural "grouping effect" and is distinguishable between authentic data and poisoned data. This enables the systematic detection and mitigation of backdoor attacks. Based on our theoretical analysis and experimental results, we demonstrate the effectiveness of PiDAn in defending against backdoor attacks that use different settings of poisoned samples on GTSRB and ILSVRC2012 datasets. Our PiDAn algorithm can detect more than 90% infected classes and identify 95% poisoned samples.

摘要: 后门攻击给深度神经网络(DNNS)带来了新的威胁，通过毒化训练数据集，错误分类包含对手触发器的输入，将后门插入到神经网络中。防御这些攻击的主要挑战是只有攻击者知道秘密触发器和目标类别。最近引入的“隐藏触发器”进一步加剧了这一问题，即触发器被小心地融合到输入中，绕过人工检查的检测，并导致通过异常检测进行的后门识别失败。为了防御这种不可察觉的攻击，我们系统地分析了后门攻击如何影响表示，即使用训练数据作为输入时给定DNN的神经元激活集。提出了一种基于相干优化的毒化数据净化算法PIDAN。我们的分析表明，有毒数据和真实数据在目标类中的表示仍然嵌入不同的线性子空间，这意味着它们与一些潜在空间表现出不同的一致性。基于这一观察，提出的PIDAN算法学习样本权重向量来最大化加权样本的投影一致性，其中我们证明了学习的权重向量具有自然的分组效应，并且可以区分真实数据和有毒数据。这使得能够系统地检测和缓解后门攻击。在理论分析和实验结果的基础上，我们在GTSRB和ILSVRC2012数据集上验证了PIDAN对使用不同设置的有毒样本的后门攻击的有效性。我们的Pidan算法可以检测到90%以上的感染类别，并识别出95%的中毒样本。



## **42. A Survey of Robust Adversarial Training in Pattern Recognition: Fundamental, Theory, and Methodologies**

模式识别中稳健的对抗性训练：基础、理论和方法综述 cs.CV

**SubmitDate**: 2022-03-26    [paper-pdf](http://arxiv.org/pdf/2203.14046v1)

**Authors**: Zhuang Qian, Kaizhu Huang, Qiu-Feng Wang, Xu-Yao Zhang

**Abstracts**: In the last a few decades, deep neural networks have achieved remarkable success in machine learning, computer vision, and pattern recognition. Recent studies however show that neural networks (both shallow and deep) may be easily fooled by certain imperceptibly perturbed input samples called adversarial examples. Such security vulnerability has resulted in a large body of research in recent years because real-world threats could be introduced due to vast applications of neural networks. To address the robustness issue to adversarial examples particularly in pattern recognition, robust adversarial training has become one mainstream. Various ideas, methods, and applications have boomed in the field. Yet, a deep understanding of adversarial training including characteristics, interpretations, theories, and connections among different models has still remained elusive. In this paper, we present a comprehensive survey trying to offer a systematic and structured investigation on robust adversarial training in pattern recognition. We start with fundamentals including definition, notations, and properties of adversarial examples. We then introduce a unified theoretical framework for defending against adversarial samples - robust adversarial training with visualizations and interpretations on why adversarial training can lead to model robustness. Connections will be also established between adversarial training and other traditional learning theories. After that, we summarize, review, and discuss various methodologies with adversarial attack and defense/training algorithms in a structured way. Finally, we present analysis, outlook, and remarks of adversarial training.

摘要: 在过去的几十年里，深度神经网络在机器学习、计算机视觉和模式识别方面取得了显著的成功。然而，最近的研究表明，神经网络(无论是浅层的还是深层的)可能很容易被某些被称为对抗性例子的潜意识扰动的输入样本所愚弄。近年来，这种安全漏洞导致了大量的研究，因为神经网络的广泛应用可能会带来现实世界的威胁。为了解决对抗实例的稳健性问题，特别是在模式识别中，稳健的对抗训练已经成为一种主流。各种思想、方法和应用在该领域蓬勃发展。然而，对对抗性训练的深入理解，包括特征、解释、理论以及不同模式之间的联系，仍然是难以捉摸的。在这篇文章中，我们提供了一个全面的调查，试图提供一个系统的和结构化的研究在模式识别中的稳健对手训练。我们从基础知识开始，包括对抗性例子的定义、符号和性质。然后，我们介绍了一个统一的理论框架来防御对抗样本-稳健的对抗训练，可视化和解释为什么对抗训练可以导致模型稳健性。对抗性训练和其他传统学习理论之间也将建立联系。然后，我们以结构化的方式总结、回顾和讨论了各种对抗性攻击和防御/训练算法的方法。最后，我们对对抗性训练进行了分析、展望和评论。



## **43. Canary Extraction in Natural Language Understanding Models**

自然语言理解模型中的金丝雀提取 cs.CL

Accepted to ACL 2022, Main Conference

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13920v1)

**Authors**: Rahil Parikh, Christophe Dupuy, Rahul Gupta

**Abstracts**: Natural Language Understanding (NLU) models can be trained on sensitive information such as phone numbers, zip-codes etc. Recent literature has focused on Model Inversion Attacks (ModIvA) that can extract training data from model parameters. In this work, we present a version of such an attack by extracting canaries inserted in NLU training data. In the attack, an adversary with open-box access to the model reconstructs the canaries contained in the model's training set. We evaluate our approach by performing text completion on canaries and demonstrate that by using the prefix (non-sensitive) tokens of the canary, we can generate the full canary. As an example, our attack is able to reconstruct a four digit code in the training dataset of the NLU model with a probability of 0.5 in its best configuration. As countermeasures, we identify several defense mechanisms that, when combined, effectively eliminate the risk of ModIvA in our experiments.

摘要: 自然语言理解(NLU)模型可以针对电话号码、邮政编码等敏感信息进行训练。最近的文献集中在模型反转攻击(MODIVA)上，它可以从模型参数中提取训练数据。在这项工作中，我们通过提取插入到NLU训练数据中的金丝雀来呈现这种攻击的一个版本。在攻击中，拥有模型开箱访问权限的对手重新构建了模型训练集中包含的金丝雀。我们通过对金丝雀执行文本补全来评估我们的方法，并演示了通过使用金丝雀的前缀(非敏感)标记，我们可以生成完整的金丝雀。例如，我们的攻击能够在NLU模型的训练数据集中以0.5的概率在其最佳配置下重建四位数代码。作为对策，我们确定了几种防御机制，当它们结合在一起时，在我们的实验中有效地消除了MODIVA的风险。



## **44. Improving robustness of jet tagging algorithms with adversarial training**

利用对抗性训练提高JET标签算法的稳健性 physics.data-an

14 pages, 11 figures, 2 tables

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13890v1)

**Authors**: Annika Stein, Xavier Coubez, Spandan Mondal, Andrzej Novak, Alexander Schmidt

**Abstracts**: Deep learning is a standard tool in the field of high-energy physics, facilitating considerable sensitivity enhancements for numerous analysis strategies. In particular, in identification of physics objects, such as jet flavor tagging, complex neural network architectures play a major role. However, these methods are reliant on accurate simulations. Mismodeling can lead to non-negligible differences in performance in data that need to be measured and calibrated against. We investigate the classifier response to input data with injected mismodelings and probe the vulnerability of flavor tagging algorithms via application of adversarial attacks. Subsequently, we present an adversarial training strategy that mitigates the impact of such simulated attacks and improves the classifier robustness. We examine the relationship between performance and vulnerability and show that this method constitutes a promising approach to reduce the vulnerability to poor modeling.

摘要: 深度学习是高能物理领域的标准工具，可大大提高许多分析策略的灵敏度。特别是，在喷气香精等物理对象的识别中，复杂的神经网络结构扮演着重要的角色。然而，这些方法依赖于准确的模拟。错误的建模可能会导致需要测量和校准的数据的性能出现不可忽略的差异。我们研究了分类器对带有注入误建模的输入数据的响应，并通过应用对抗性攻击来探索味道标注算法的脆弱性。随后，我们提出了一种对抗性训练策略，减轻了这类模拟攻击的影响，提高了分类器的稳健性。我们研究了性能和脆弱性之间的关系，并表明该方法是一种很有前途的方法，可以降低因建模不当而造成的脆弱性。



## **45. Origins of Low-dimensional Adversarial Perturbations**

低维对抗性扰动的起源 stat.ML

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13779v1)

**Authors**: Elvis Dohmatob, Chuan Guo, Morgane Goibert

**Abstracts**: In this note, we initiate a rigorous study of the phenomenon of low-dimensional adversarial perturbations in classification. These are adversarial perturbations wherein, unlike the classical setting, the attacker's search is limited to a low-dimensional subspace of the feature space. The goal is to fool the classifier into flipping its decision on a nonzero fraction of inputs from a designated class, upon the addition of perturbations from a subspace chosen by the attacker and fixed once and for all. It is desirable that the dimension $k$ of the subspace be much smaller than the dimension $d$ of the feature space, while the norm of the perturbations should be negligible compared to the norm of a typical data point. In this work, we consider binary classification models under very general regularity conditions, which are verified by certain feedforward neural networks (e.g., with sufficiently smooth, or else ReLU activation function), and compute analytical lower-bounds for the fooling rate of any subspace. These bounds explicitly highlight the dependence that the fooling rate has on the margin of the model (i.e., the ratio of the output to its $L_2$-norm of its gradient at a test point), and on the alignment of the given subspace with the gradients of the model w.r.t. inputs. Our results provide a theoretical explanation for the recent success of heuristic methods for efficiently generating low-dimensional adversarial perturbations. Moreover, our theoretical results are confirmed by experiments.

摘要: 在这篇笔记中，我们开始了对分类中低维对抗性扰动现象的严格研究。这些是对抗性扰动，其中，与经典设置不同，攻击者的搜索被限制在特征空间的低维子空间。其目的是愚弄分类器，在添加来自攻击者选择的并一劳永逸地修复的子空间的扰动时，翻转其对指定类别输入的非零分数的决定。希望子空间的维度$k$比特征空间的维度$d$小得多，而与典型数据点的范数相比，扰动的范数应该是可以忽略的。在这项工作中，我们考虑了在非常一般的正则性条件下的二分类模型，并用某些前向神经网络(例如，具有足够光滑的激活函数)进行了验证，并计算了任意子空间的愚弄率的解析下界。这些界明确地突出了愚弄率对模型边际(即，输出与其在测试点的梯度的$L_2$-范数的比率)以及给定子空间与模型的梯度对齐的依赖性。投入。我们的结果为启发式方法最近成功地产生低维对抗性扰动提供了理论上的解释。此外，我们的理论结果也得到了实验的证实。



## **46. Understanding and Increasing Efficiency of Frank-Wolfe Adversarial Training**

理解和提高弗兰克-沃尔夫对抗性训练的效率 cs.LG

IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR  2022. Preliminary version ICML 2021 Adversarial Machine Learning Workshop.  Code: https://github.com/TheoT1/FW-AT-Adapt

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2012.12368v5)

**Authors**: Theodoros Tsiligkaridis, Jay Roberts

**Abstracts**: Deep neural networks are easily fooled by small perturbations known as adversarial attacks. Adversarial Training (AT) is a technique that approximately solves a robust optimization problem to minimize the worst-case loss and is widely regarded as the most effective defense. Due to the high computation time for generating strong adversarial examples in the AT process, single-step approaches have been proposed to reduce training time. However, these methods suffer from catastrophic overfitting where adversarial accuracy drops during training, and although improvements have been proposed, they increase training time and robustness is far from that of multi-step AT. We develop a theoretical framework for adversarial training with FW optimization (FW-AT) that reveals a geometric connection between the loss landscape and the $\ell_2$ distortion of $\ell_\infty$ FW attacks. We analytically show that high distortion of FW attacks is equivalent to small gradient variation along the attack path. It is then experimentally demonstrated on various deep neural network architectures that $\ell_\infty$ attacks against robust models achieve near maximal distortion, while standard networks have lower distortion. It is experimentally shown that catastrophic overfitting is strongly correlated with low distortion of FW attacks. This mathematical transparency differentiates FW from Projected Gradient Descent (PGD) optimization. To demonstrate the utility of our theoretical framework we develop FW-AT-Adapt, a novel adversarial training algorithm which uses a simple distortion measure to adapt the number of attack steps during training to increase efficiency without compromising robustness. FW-AT-Adapt provides training time on par with single-step fast AT methods and closes the gap between fast AT methods and multi-step PGD-AT with minimal loss in adversarial accuracy in white-box and black-box settings.

摘要: 深度神经网络很容易被被称为对抗性攻击的小扰动所愚弄。对抗性训练(AT)是一种近似地解决稳健优化问题以最小化最坏情况损失的技术，被广泛认为是最有效的防御方法。由于在AT过程中生成强对抗性样本需要很高的计算时间，因此提出了一种单步方法来减少训练时间。然而，这些方法在训练过程中存在灾难性的过拟合问题，其中对抗性精度会下降，虽然已经提出了改进，但它们增加了训练时间，并且鲁棒性远远低于多步AT。我们开发了一个基于FW优化的对抗性训练理论框架(FW-AT)，该框架揭示了损失情况与$\ell_inty$FW攻击的$\ell_2$失真之间的几何关系。我们分析表明，FW攻击的高失真等价于攻击路径上的小梯度变化。在不同深度神经网络结构上的实验结果表明，对健壮模型的攻击可以获得接近最大的失真，而标准网络的失真较小。实验表明，灾难性过拟合与FW攻击的低失真密切相关。这种数学透明度将FW与投影渐变下降(PGD)优化区分开来。为了证明我们的理论框架的有效性，我们开发了一种新的对抗性训练算法FW-AT-Adapt，它使用一个简单的失真度量来调整训练过程中的攻击步数，以在不影响健壮性的情况下提高效率。FW-AT-Adapt提供与单步快速AT方法相当的训练时间，并在白盒和黑盒设置下以最小的对手精度损失缩小了快速AT方法和多步PGD-AT方法之间的差距。



## **47. Give Me Your Attention: Dot-Product Attention Considered Harmful for Adversarial Patch Robustness**

给我注意：点积注意被认为对对手补丁的健壮性有害 cs.CV

to be published in IEEE/CVF Conference on Computer Vision and Pattern  Recognition 2022, CVPR22

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13639v1)

**Authors**: Giulio Lovisotto, Nicole Finnie, Mauricio Munoz, Chaithanya Kumar Mummadi, Jan Hendrik Metzen

**Abstracts**: Neural architectures based on attention such as vision transformers are revolutionizing image recognition. Their main benefit is that attention allows reasoning about all parts of a scene jointly. In this paper, we show how the global reasoning of (scaled) dot-product attention can be the source of a major vulnerability when confronted with adversarial patch attacks. We provide a theoretical understanding of this vulnerability and relate it to an adversary's ability to misdirect the attention of all queries to a single key token under the control of the adversarial patch. We propose novel adversarial objectives for crafting adversarial patches which target this vulnerability explicitly. We show the effectiveness of the proposed patch attacks on popular image classification (ViTs and DeiTs) and object detection models (DETR). We find that adversarial patches occupying 0.5% of the input can lead to robust accuracies as low as 0% for ViT on ImageNet, and reduce the mAP of DETR on MS COCO to less than 3%.

摘要: 基于注意力的神经结构，如视觉转换器，正在给图像识别带来革命性的变化。它们的主要好处是，注意力允许对场景的所有部分进行联合推理。在这篇文章中，我们展示了在面对敌意补丁攻击时，(按比例)点积注意力的全局推理如何成为主要漏洞的来源。我们提供了对该漏洞的理论理解，并将其与对手将所有查询的注意力误导到对手补丁控制下的单个密钥令牌的能力相关联。我们提出了新的对抗性目标，用于制作明确针对此漏洞的对抗性补丁。我们展示了所提出的补丁攻击在流行的图像分类(VITS和DeITS)和目标检测模型(DETR)上的有效性。我们发现，敌意补丁占输入的0.5%可以导致VIT在ImageNet上的稳健准确率低至0%，并将DETR在MS CoCo上的MAP降低到3%以下。



## **48. Adversarial Bone Length Attack on Action Recognition**

动作识别的对抗性骨长攻击 cs.CV

12 pages, 8 figures, accepted to AAAI2022

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2109.05830v2)

**Authors**: Nariki Tanaka, Hiroshi Kera, Kazuhiko Kawamoto

**Abstracts**: Skeleton-based action recognition models have recently been shown to be vulnerable to adversarial attacks. Compared to adversarial attacks on images, perturbations to skeletons are typically bounded to a lower dimension of approximately 100 per frame. This lower-dimensional setting makes it more difficult to generate imperceptible perturbations. Existing attacks resolve this by exploiting the temporal structure of the skeleton motion so that the perturbation dimension increases to thousands. In this paper, we show that adversarial attacks can be performed on skeleton-based action recognition models, even in a significantly low-dimensional setting without any temporal manipulation. Specifically, we restrict the perturbations to the lengths of the skeleton's bones, which allows an adversary to manipulate only approximately 30 effective dimensions. We conducted experiments on the NTU RGB+D and HDM05 datasets and demonstrate that the proposed attack successfully deceived models with sometimes greater than 90% success rate by small perturbations. Furthermore, we discovered an interesting phenomenon: in our low-dimensional setting, the adversarial training with the bone length attack shares a similar property with data augmentation, and it not only improves the adversarial robustness but also improves the classification accuracy on the original data. This is an interesting counterexample of the trade-off between adversarial robustness and clean accuracy, which has been widely observed in studies on adversarial training in the high-dimensional regime.

摘要: 基于骨架的动作识别模型最近被证明容易受到对手攻击。与对图像的敌意攻击相比，对骨骼的扰动通常被限制在大约每帧100个维度的较低维度。这种较低维度的设置使产生难以察觉的微扰变得更加困难。现有的攻击通过利用骨骼运动的时间结构来解决这个问题，从而使扰动维度增加到数千。在本文中，我们证明了对抗性攻击可以在基于骨架的动作识别模型上执行，即使在显著低维的环境中也不需要任何时间处理。具体地说，我们将扰动限制在骨骼的长度上，这使得对手只能操纵大约30个有效维度。我们在NTU、RGB+D和HDM05数据集上进行了实验，结果表明，该攻击通过微小的扰动成功地欺骗了模型，有时成功率超过90%。此外，我们还发现了一个有趣的现象：在我们的低维环境下，带有骨长攻击的对抗性训练与数据增强具有相似的性质，它不仅提高了对抗性的健壮性，而且提高了对原始数据的分类精度。这是一个有趣的反例，说明了对抗性稳健性和清晰准确性之间的权衡，这在高维体制下对抗性训练的研究中得到了广泛的观察。



## **49. Improving Adversarial Transferability with Spatial Momentum**

利用空间动量提高对抗性转移能力 cs.CV

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13479v1)

**Authors**: Guoqiu Wang, Xingxing Wei, Huanqian Yan

**Abstracts**: Deep Neural Networks (DNN) are vulnerable to adversarial examples. Although many adversarial attack methods achieve satisfactory attack success rates under the white-box setting, they usually show poor transferability when attacking other DNN models. Momentum-based attack (MI-FGSM) is one effective method to improve transferability. It integrates the momentum term into the iterative process, which can stabilize the update directions by adding the gradients' temporal correlation for each pixel. We argue that only this temporal momentum is not enough, the gradients from the spatial domain within an image, i.e. gradients from the context pixels centered on the target pixel are also important to the stabilization. For that, in this paper, we propose a novel method named Spatial Momentum Iterative FGSM Attack (SMI-FGSM), which introduces the mechanism of momentum accumulation from temporal domain to spatial domain by considering the context gradient information from different regions within the image. SMI-FGSM is then integrated with MI-FGSM to simultaneously stabilize the gradients' update direction from both the temporal and spatial domain. The final method is called SM$^2$I-FGSM. Extensive experiments are conducted on the ImageNet dataset and results show that SM$^2$I-FGSM indeed further enhances the transferability. It achieves the best transferability success rate for multiple mainstream undefended and defended models, which outperforms the state-of-the-art methods by a large margin.

摘要: 深度神经网络(DNN)很容易受到敌意例子的影响。虽然许多对抗性攻击方法在白盒设置下取得了令人满意的攻击成功率，但它们在攻击其他DNN模型时往往表现出较差的可移植性。基于动量的攻击(MI-FGSM)是提高可转移性的一种有效方法。该算法在迭代过程中引入动量项，通过对每个像素增加梯度的时间相关性来稳定更新方向。我们认为，仅有这种时间动量是不够的，图像中来自空间域的梯度，即来自以目标像素为中心的上下文像素的梯度，对于稳定也是重要的。为此，本文提出了一种新的方法--空间动量迭代FGSM攻击(SMI-FGSM)，该方法通过考虑图像内不同区域的上下文梯度信息，引入了从时间域到空间域的动量积累机制。然后将SMI-FGSM与MI-FGSM相结合，从时间域和空间域同时稳定梯度的更新方向。最后一种方法称为SM$^2$I-FGSM。在ImageNet数据集上进行了大量的实验，结果表明SM$^2$I-FGSM确实进一步提高了可转移性。它在多个主流的无防御和有防御的模型上实现了最好的可转移性成功率，远远超过了最先进的方法。



## **50. Trojan Horse Training for Breaking Defenses against Backdoor Attacks in Deep Learning**

深度学习中突破后门攻击防御的特洛伊木马训练 cs.CR

Submitted to conference

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.15506v1)

**Authors**: Arezoo Rajabi, Bhaskar Ramasubramanian, Radha Poovendran

**Abstracts**: Machine learning (ML) models that use deep neural networks are vulnerable to backdoor attacks. Such attacks involve the insertion of a (hidden) trigger by an adversary. As a consequence, any input that contains the trigger will cause the neural network to misclassify the input to a (single) target class, while classifying other inputs without a trigger correctly. ML models that contain a backdoor are called Trojan models. Backdoors can have severe consequences in safety-critical cyber and cyber physical systems when only the outputs of the model are available. Defense mechanisms have been developed and illustrated to be able to distinguish between outputs from a Trojan model and a non-Trojan model in the case of a single-target backdoor attack with accuracy > 96 percent. Understanding the limitations of a defense mechanism requires the construction of examples where the mechanism fails. Current single-target backdoor attacks require one trigger per target class. We introduce a new, more general attack that will enable a single trigger to result in misclassification to more than one target class. Such a misclassification will depend on the true (actual) class that the input belongs to. We term this category of attacks multi-target backdoor attacks. We demonstrate that a Trojan model with either a single-target or multi-target trigger can be trained so that the accuracy of a defense mechanism that seeks to distinguish between outputs coming from a Trojan and a non-Trojan model will be reduced. Our approach uses the non-Trojan model as a teacher for the Trojan model and solves a min-max optimization problem between the Trojan model and defense mechanism. Empirical evaluations demonstrate that our training procedure reduces the accuracy of a state-of-the-art defense mechanism from >96 to 0 percent.

摘要: 使用深度神经网络的机器学习(ML)模型容易受到后门攻击。此类攻击涉及对手插入(隐藏的)触发器。因此，任何包含触发器的输入都会导致神经网络将输入错误地分类到(单个)目标类，而对没有触发器的其他输入进行正确分类。包含后门的ML模型称为特洛伊木马模型。如果只有模型的输出可用，那么在安全关键的网络和网络物理系统中，后门可能会产生严重后果。已开发和说明的防御机制能够在单目标后门攻击的情况下区分来自特洛伊木马模型和非特洛伊木马模型的输出，准确率超过96%。要理解防御机制的局限性，就需要在机制失效的地方举例说明。当前的单目标后门攻击需要每个目标类一个触发器。我们引入了一种新的、更通用的攻击，它将使单个触发器能够导致错误分类到多个目标类别。这种错误分类将取决于输入所属的真实(实际)类。我们称这类攻击为多目标后门攻击。我们证明了具有单目标或多目标触发器的特洛伊木马模型可以被训练，因此试图区分来自特洛伊木马和非特洛伊木马模型的输出的防御机制的准确性将会降低。该方法使用非木马模型作为木马模型的教师，解决了木马模型和防御机制之间的最小-最大优化问题。经验评估表明，我们的训练程序将最先进的防御机制的准确率从>96%降低到0%。



