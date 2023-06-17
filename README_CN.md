# Latest Adversarial Attack Papers
**update at 2023-06-17 15:38:19**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Inroads into Autonomous Network Defence using Explained Reinforcement Learning**

基于解释强化学习的自主网络防御研究 cs.CR

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2306.09318v1) [paper-pdf](http://arxiv.org/pdf/2306.09318v1)

**Authors**: Myles Foley, Mia Wang, Zoe M, Chris Hicks, Vasilios Mavroudis

**Abstract**: Computer network defence is a complicated task that has necessitated a high degree of human involvement. However, with recent advancements in machine learning, fully autonomous network defence is becoming increasingly plausible. This paper introduces an end-to-end methodology for studying attack strategies, designing defence agents and explaining their operation. First, using state diagrams, we visualise adversarial behaviour to gain insight about potential points of intervention and inform the design of our defensive models. We opt to use a set of deep reinforcement learning agents trained on different parts of the task and organised in a shallow hierarchy. Our evaluation shows that the resulting design achieves a substantial performance improvement compared to prior work. Finally, to better investigate the decision-making process of our agents, we complete our analysis with a feature ablation and importance study.

摘要: 计算机网络防御是一项复杂的任务，需要高度的人工参与。然而，随着最近机器学习的进步，完全自主的网络防御正变得越来越有可能。本文介绍了一种端到端的方法，用于研究攻击策略、设计防御代理并解释它们的操作。首先，我们使用状态图将敌对行为形象化，以洞察潜在的干预点，并为我们的防御模型的设计提供信息。我们选择使用一组针对任务不同部分进行培训的深度强化学习代理，并以浅层次进行组织。我们的评估结果表明，与以前的工作相比，所得到的设计实现了显著的性能改进。最后，为了更好地研究我们的代理的决策过程，我们用特征消融和重要性研究来完成我们的分析。



## **2. Adversarial Cheap Talk**

对抗性的低级谈资 cs.LG

To be published at ICML 2023. Project video and code are available at  https://sites.google.com/view/adversarial-cheap-talk

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2211.11030v2) [paper-pdf](http://arxiv.org/pdf/2211.11030v2)

**Authors**: Chris Lu, Timon Willi, Alistair Letcher, Jakob Foerster

**Abstract**: Adversarial attacks in reinforcement learning (RL) often assume highly-privileged access to the victim's parameters, environment, or data. Instead, this paper proposes a novel adversarial setting called a Cheap Talk MDP in which an Adversary can merely append deterministic messages to the Victim's observation, resulting in a minimal range of influence. The Adversary cannot occlude ground truth, influence underlying environment dynamics or reward signals, introduce non-stationarity, add stochasticity, see the Victim's actions, or access their parameters. Additionally, we present a simple meta-learning algorithm called Adversarial Cheap Talk (ACT) to train Adversaries in this setting. We demonstrate that an Adversary trained with ACT still significantly influences the Victim's training and testing performance, despite the highly constrained setting. Affecting train-time performance reveals a new attack vector and provides insight into the success and failure modes of existing RL algorithms. More specifically, we show that an ACT Adversary is capable of harming performance by interfering with the learner's function approximation, or instead helping the Victim's performance by outputting useful features. Finally, we show that an ACT Adversary can manipulate messages during train-time to directly and arbitrarily control the Victim at test-time. Project video and code are available at https://sites.google.com/view/adversarial-cheap-talk

摘要: 强化学习(RL)中的对抗性攻击通常假定具有访问受害者参数、环境或数据的高度特权。相反，本文提出了一种新的对抗性环境，称为廉价谈话MDP，在该环境中，对手只需将确定性消息附加到受害者的观察中，从而产生最小的影响范围。敌手不能掩盖基本事实、影响潜在环境动态或奖励信号、引入非平稳性、增加随机性、看到受害者的行为或获取他们的参数。此外，我们还提出了一个简单的元学习算法，称为对抗性廉价谈话(ACT)，以在这种情况下训练对手。我们证明，尽管在高度受限的环境下，接受过ACT训练的对手仍然会显著影响受害者的训练和测试表现。影响训练时间性能揭示了新的攻击向量，并提供了对现有RL算法的成功和失败模式的洞察。更具体地说，我们证明了ACT对手能够通过干扰学习者的函数逼近来损害性能，或者相反地通过输出有用的特征来帮助受害者的性能。最后，我们证明了ACT攻击者可以在训练时间内操纵消息，从而在测试时间直接任意控制受害者。项目视频和代码可在https://sites.google.com/view/adversarial-cheap-talk上获得



## **3. DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks in the Physical World**

DIFFender：物理世界中基于扩散的对抗性防御补丁攻击 cs.CV

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2306.09124v1) [paper-pdf](http://arxiv.org/pdf/2306.09124v1)

**Authors**: Caixin Kang, Yinpeng Dong, Zhengyi Wang, Shouwei Ruan, Hang Su, Xingxing Wei

**Abstract**: Adversarial attacks in the physical world, particularly patch attacks, pose significant threats to the robustness and reliability of deep learning models. Developing reliable defenses against patch attacks is crucial for real-world applications, yet current research in this area is severely lacking. In this paper, we propose DIFFender, a novel defense method that leverages the pre-trained diffusion model to perform both localization and defense against potential adversarial patch attacks. DIFFender is designed as a pipeline consisting of two main stages: patch localization and restoration. In the localization stage, we exploit the intriguing properties of a diffusion model to effectively identify the locations of adversarial patches. In the restoration stage, we employ a text-guided diffusion model to eliminate adversarial regions in the image while preserving the integrity of the visual content. Additionally, we design a few-shot prompt-tuning algorithm to facilitate simple and efficient tuning, enabling the learned representations to easily transfer to downstream tasks, which optimize two stages jointly. We conduct extensive experiments on image classification and face recognition to demonstrate that DIFFender exhibits superior robustness under strong adaptive attacks and generalizes well across various scenarios, diverse classifiers, and multiple attack methods.

摘要: 物理世界中的对抗性攻击，特别是补丁攻击，对深度学习模型的健壮性和可靠性构成了严重威胁。开发针对补丁攻击的可靠防御对于现实世界的应用至关重要，但目前这一领域的研究严重缺乏。在本文中，我们提出了一种新的防御方法DIFFender，它利用预先训练的扩散模型来定位和防御潜在的敌意补丁攻击。DIFFender被设计为一个由两个主要阶段组成的管道：补丁定位和恢复。在本地化阶段，我们利用扩散模型的有趣性质来有效地识别敌方补丁的位置。在恢复阶段，我们使用文本引导的扩散模型来消除图像中的对抗性区域，同时保持视觉内容的完整性。此外，我们设计了几个镜头的提示调整算法，以便于简单有效的调整，使学习到的表示可以很容易地转移到下游任务，共同优化两个阶段。我们在图像分类和人脸识别上进行了大量的实验，证明了DIFFender在强自适应攻击下表现出了良好的鲁棒性，并且能够很好地适用于各种场景、不同的分类器和多种攻击方法。



## **4. The Effect of Length on Key Fingerprint Verification Security and Usability**

长度对密钥指纹验证安全性和可用性的影响 cs.CR

Accepted to International Conference on Availability, Reliability and  Security (ARES 2023)

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2306.04574v2) [paper-pdf](http://arxiv.org/pdf/2306.04574v2)

**Authors**: Dan Turner, Siamak F. Shahandashti, Helen Petrie

**Abstract**: In applications such as end-to-end encrypted instant messaging, secure email, and device pairing, users need to compare key fingerprints to detect impersonation and adversary-in-the-middle attacks. Key fingerprints are usually computed as truncated hashes of each party's view of the channel keys, encoded as an alphanumeric or numeric string, and compared out-of-band, e.g. manually, to detect any inconsistencies. Previous work has extensively studied the usability of various verification strategies and encoding formats, however, the exact effect of key fingerprint length on the security and usability of key fingerprint verification has not been rigorously investigated. We present a 162-participant study on the effect of numeric key fingerprint length on comparison time and error rate. While the results confirm some widely-held intuitions such as general comparison times and errors increasing significantly with length, a closer look reveals interesting nuances. The significant rise in comparison time only occurs when highly similar fingerprints are compared, and comparison time remains relatively constant otherwise. On errors, our results clearly distinguish between security non-critical errors that remain low irrespective of length and security critical errors that significantly rise, especially at higher fingerprint lengths. A noteworthy implication of this latter result is that Signal/WhatsApp key fingerprints provide a considerably lower level of security than usually assumed.

摘要: 在端到端加密即时消息、安全电子邮件和设备配对等应用中，用户需要比较密钥指纹来检测模仿和中间人攻击。密钥指纹通常被计算为每一方的频道密钥视图的截断散列，被编码为字母数字或数字字符串，并例如手动地进行带外比较以检测任何不一致。以往的工作已经广泛地研究了各种验证策略和编码格式的可用性，但还没有严格地研究密钥指纹长度对密钥指纹验证的安全性和可用性的确切影响。我们对162名参与者进行了一项关于数字密钥指纹长度对比较时间和错误率的影响的研究。虽然结果证实了一些普遍存在的直觉，如一般的比较时间和误差随着长度的增加而显著增加，但仔细观察会发现有趣的细微差别。只有当比较高度相似的指纹时，比较时间才会显著增加，否则比较时间保持相对恒定。在错误方面，我们的结果清楚地区分了无论长度如何都保持较低的安全非关键错误和显著上升的安全关键错误，特别是在较长的指纹长度时。后一种结果的一个值得注意的含义是，Signal/WhatsApp密钥指纹提供的安全级别比通常假设的要低得多。



## **5. Community Detection Attack against Collaborative Learning-based Recommender Systems**

基于协作学习的推荐系统的社区检测攻击 cs.IR

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2306.08929v1) [paper-pdf](http://arxiv.org/pdf/2306.08929v1)

**Authors**: Yacine Belal, Sonia Ben Mokhtar, Mohamed Maouche, Anthony Simonet-Boulogne

**Abstract**: Collaborative-learning based recommender systems emerged following the success of collaborative learning techniques such as Federated Learning (FL) and Gossip Learning (GL). In these systems, users participate in the training of a recommender system while keeping their history of consumed items on their devices. While these solutions seemed appealing for preserving the privacy of the participants at a first glance, recent studies have shown that collaborative learning can be vulnerable to a variety of privacy attacks. In this paper we propose a novel privacy attack called Community Detection Attack (CDA), which allows an adversary to discover the members of a community based on a set of items of her choice (e.g., discovering users interested in LGBT content). Through experiments on three real recommendation datasets and by using two state-of-the-art recommendation models, we assess the sensitivity of an FL-based recommender system as well as two flavors of Gossip Learning-based recommender systems to CDA. Results show that on all models and all datasets, the FL setting is more vulnerable to CDA than Gossip settings. We further evaluated two off-the-shelf mitigation strategies, namely differential privacy (DP) and a share less policy, which consists in sharing a subset of model parameters. Results show a better privacy-utility trade-off for the share less policy compared to DP especially in the Gossip setting.

摘要: 基于协作学习的推荐系统是在联邦学习(FL)和八卦学习(GL)等协作学习技术成功之后应运而生的。在这些系统中，用户参与推荐系统的培训，同时在他们的设备上保存他们的消费项目的历史。虽然这些解决方案乍一看似乎在保护参与者的隐私方面很有吸引力，但最近的研究表明，协作学习可能容易受到各种隐私攻击。在本文中，我们提出了一种新的隐私攻击，称为社区检测攻击(CDA)，它允许攻击者根据她选择的一组项目来发现社区成员(例如，发现对LGBT内容感兴趣的用户)。通过在三个真实推荐数据集上的实验，使用两种最新的推荐模型，我们评估了一个基于FL的推荐系统以及两种基于八卦学习的推荐系统对CDA的敏感度。结果表明，在所有模型和所有数据集上，FL设置比八卦设置更容易受到CDA的影响。我们进一步评估了两种现成的缓解策略，即差异隐私(DP)策略和共享较少策略，该策略包括共享模型参数的子集。结果表明，与DP相比，共享更少的策略具有更好的隐私效用权衡，尤其是在八卦环境下。



## **6. MalProtect: Stateful Defense Against Adversarial Query Attacks in ML-based Malware Detection**

MalProtect：基于ML的恶意软件检测中对抗恶意查询攻击的状态防御 cs.LG

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2302.10739v2) [paper-pdf](http://arxiv.org/pdf/2302.10739v2)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: ML models are known to be vulnerable to adversarial query attacks. In these attacks, queries are iteratively perturbed towards a particular class without any knowledge of the target model besides its output. The prevalence of remotely-hosted ML classification models and Machine-Learning-as-a-Service platforms means that query attacks pose a real threat to the security of these systems. To deal with this, stateful defenses have been proposed to detect query attacks and prevent the generation of adversarial examples by monitoring and analyzing the sequence of queries received by the system. Several stateful defenses have been proposed in recent years. However, these defenses rely solely on similarity or out-of-distribution detection methods that may be effective in other domains. In the malware detection domain, the methods to generate adversarial examples are inherently different, and therefore we find that such detection mechanisms are significantly less effective. Hence, in this paper, we present MalProtect, which is a stateful defense against query attacks in the malware detection domain. MalProtect uses several threat indicators to detect attacks. Our results show that it reduces the evasion rate of adversarial query attacks by 80+\% in Android and Windows malware, across a range of attacker scenarios. In the first evaluation of its kind, we show that MalProtect outperforms prior stateful defenses, especially under the peak adversarial threat.

摘要: 众所周知，ML模型容易受到敌意查询攻击。在这些攻击中，查询被迭代地扰动到特定的类，除了其输出之外，不知道目标模型。远程托管的ML分类模型和机器学习即服务平台的流行意味着查询攻击对这些系统的安全构成了真正的威胁。为了解决这个问题，已经提出了状态防御来检测查询攻击，并通过监控和分析系统接收到的查询序列来防止敌对实例的生成。近年来，有人提出了几项有状态的辩护。然而，这些防御完全依赖于可能在其他领域有效的相似性或分布外检测方法。在恶意软件检测领域，生成恶意示例的方法本质上是不同的，因此我们发现这种检测机制的有效性显著降低。因此，在本文中，我们提出了MalProtect，它是恶意软件检测领域中针对查询攻击的一种状态防御。MalProtect使用多个威胁指示器来检测攻击。我们的结果表明，在各种攻击场景下，该算法将Android和Windows恶意软件中恶意查询攻击的逃避率降低了80%+\%。在该类型的第一次评估中，我们表明MalProtect的性能优于先前的状态防御，特别是在峰值敌意威胁下。



## **7. Augment then Smooth: Reconciling Differential Privacy with Certified Robustness**

先增强后平滑：使差异隐私与认证的健壮性相协调 cs.LG

25 pages, 19 figures

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08656v1) [paper-pdf](http://arxiv.org/pdf/2306.08656v1)

**Authors**: Jiapeng Wu, Atiyeh Ashari Ghomi, David Glukhov, Jesse C. Cresswell, Franziska Boenisch, Nicolas Papernot

**Abstract**: Machine learning models are susceptible to a variety of attacks that can erode trust in their deployment. These threats include attacks against the privacy of training data and adversarial examples that jeopardize model accuracy. Differential privacy and randomized smoothing are effective defenses that provide certifiable guarantees for each of these threats, however, it is not well understood how implementing either defense impacts the other. In this work, we argue that it is possible to achieve both privacy guarantees and certified robustness simultaneously. We provide a framework called DP-CERT for integrating certified robustness through randomized smoothing into differentially private model training. For instance, compared to differentially private stochastic gradient descent on CIFAR10, DP-CERT leads to a 12-fold increase in certified accuracy and a 10-fold increase in the average certified radius at the expense of a drop in accuracy of 1.2%. Through in-depth per-sample metric analysis, we show that the certified radius correlates with the local Lipschitz constant and smoothness of the loss surface. This provides a new way to diagnose when private models will fail to be robust.

摘要: 机器学习模型容易受到各种攻击，这些攻击可能会侵蚀对其部署的信任。这些威胁包括对训练数据隐私的攻击，以及危及模型准确性的敌意例子。差异隐私和随机平滑是为这些威胁中的每一种提供可证明的保证的有效防御措施，然而，实施这两种防御措施对另一种威胁的影响还不是很清楚。在这项工作中，我们认为可以同时实现隐私保证和认证的健壮性。我们提供了一个称为DP-CERT的框架，用于将通过随机平滑验证的稳健性集成到不同的私有模型训练中。例如，与CIFAR10上的差分私有随机梯度下降相比，DP-CERT的认证精度提高了12倍，平均认证半径增加了10倍，但精度下降了1.2%。通过深入的逐样本度量分析，我们发现认证半径与损失曲面的局部Lipschitz常数和光滑度相关。这提供了一种新的方法来诊断何时私人车型将不再健壮。



## **8. A Unified Framework of Graph Information Bottleneck for Robustness and Membership Privacy**

面向健壮性和成员隐私的图信息瓶颈统一框架 cs.LG

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08604v1) [paper-pdf](http://arxiv.org/pdf/2306.08604v1)

**Authors**: Enyan Dai, Limeng Cui, Zhengyang Wang, Xianfeng Tang, Yinghan Wang, Monica Cheng, Bing Yin, Suhang Wang

**Abstract**: Graph Neural Networks (GNNs) have achieved great success in modeling graph-structured data. However, recent works show that GNNs are vulnerable to adversarial attacks which can fool the GNN model to make desired predictions of the attacker. In addition, training data of GNNs can be leaked under membership inference attacks. This largely hinders the adoption of GNNs in high-stake domains such as e-commerce, finance and bioinformatics. Though investigations have been made in conducting robust predictions and protecting membership privacy, they generally fail to simultaneously consider the robustness and membership privacy. Therefore, in this work, we study a novel problem of developing robust and membership privacy-preserving GNNs. Our analysis shows that Information Bottleneck (IB) can help filter out noisy information and regularize the predictions on labeled samples, which can benefit robustness and membership privacy. However, structural noises and lack of labels in node classification challenge the deployment of IB on graph-structured data. To mitigate these issues, we propose a novel graph information bottleneck framework that can alleviate structural noises with neighbor bottleneck. Pseudo labels are also incorporated in the optimization to minimize the gap between the predictions on the labeled set and unlabeled set for membership privacy. Extensive experiments on real-world datasets demonstrate that our method can give robust predictions and simultaneously preserve membership privacy.

摘要: 图神经网络(GNN)在图结构数据建模方面取得了巨大的成功。然而，最近的研究表明，GNN很容易受到敌意攻击，这些攻击可以欺骗GNN模型做出所需的攻击者预测。此外，在成员关系推理攻击下，GNN的训练数据可能会被泄露。这在很大程度上阻碍了在电子商务、金融和生物信息学等高风险领域采用GNN。虽然已经在稳健预测和保护成员隐私方面进行了研究，但他们通常没有同时考虑稳健性和成员隐私。因此，在这项工作中，我们研究了一个新的问题，即开发健壮的、保护成员隐私的GNN。我们的分析表明，信息瓶颈(IB)可以帮助过滤噪声信息，并使对标记样本的预测正规化，这有利于稳健性和成员隐私。然而，结构噪声和节点分类中标签的缺乏对图结构数据上的IB的部署提出了挑战。为了缓解这些问题，我们提出了一种新的图信息瓶颈框架，该框架可以缓解带有邻居瓶颈的结构噪声。在优化过程中还加入了伪标签，以最大限度地减少对已标记集合和未标记集合的预测之间的差距，从而保证成员隐私。在真实数据集上的大量实验表明，我们的方法可以给出稳健的预测，同时保护成员隐私。



## **9. Tight Certification of Adversarially Trained Neural Networks via Nonconvex Low-Rank Semidefinite Relaxations**

基于非凸低阶半正定松弛的对抗性训练神经网络的紧认证 cs.LG

ICML 2023

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2211.17244v3) [paper-pdf](http://arxiv.org/pdf/2211.17244v3)

**Authors**: Hong-Ming Chiu, Richard Y. Zhang

**Abstract**: Adversarial training is well-known to produce high-quality neural network models that are empirically robust against adversarial perturbations. Nevertheless, once a model has been adversarially trained, one often desires a certification that the model is truly robust against all future attacks. Unfortunately, when faced with adversarially trained models, all existing approaches have significant trouble making certifications that are strong enough to be practically useful. Linear programming (LP) techniques in particular face a "convex relaxation barrier" that prevent them from making high-quality certifications, even after refinement with mixed-integer linear programming (MILP) and branch-and-bound (BnB) techniques. In this paper, we propose a nonconvex certification technique, based on a low-rank restriction of a semidefinite programming (SDP) relaxation. The nonconvex relaxation makes strong certifications comparable to much more expensive SDP methods, while optimizing over dramatically fewer variables comparable to much weaker LP methods. Despite nonconvexity, we show how off-the-shelf local optimization algorithms can be used to achieve and to certify global optimality in polynomial time. Our experiments find that the nonconvex relaxation almost completely closes the gap towards exact certification of adversarially trained models.

摘要: 众所周知，对抗性训练可以产生高质量的神经网络模型，这些模型对对抗性扰动具有经验上的健壮性。然而，一旦一个模型经过对抗性的训练，人们往往希望得到一个证明，证明该模型对未来的所有攻击都是真正健壮的。不幸的是，当面对对手训练的模型时，所有现有的方法都在制作强大到足以实用的证书方面存在重大问题。线性规划(LP)技术尤其面临着一种“凸松弛障碍”，即使在使用混合整数线性规划(MILP)和分支定界(BNB)技术进行了改进之后，也无法进行高质量的认证。本文提出了一种基于半定规划(SDP)松弛的低阶限制的非凸证明技术。非凸松弛使强认证可与昂贵得多的SDP方法相媲美，同时优化的变量可比弱得多的线性规划方法少得多。尽管非凸性，我们展示了如何使用现成的局部优化算法来在多项式时间内实现和证明全局最优性。我们的实验发现，非凸松弛几乎完全弥合了对对抗性训练模型进行精确验证的差距。



## **10. Reliable Evaluation of Adversarial Transferability**

对抗性转移性的可靠评估 cs.CV

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08565v1) [paper-pdf](http://arxiv.org/pdf/2306.08565v1)

**Authors**: Wenqian Yu, Jindong Gu, Zhijiang Li, Philip Torr

**Abstract**: Adversarial examples (AEs) with small adversarial perturbations can mislead deep neural networks (DNNs) into wrong predictions. The AEs created on one DNN can also fool another DNN. Over the last few years, the transferability of AEs has garnered significant attention as it is a crucial property for facilitating black-box attacks. Many approaches have been proposed to improve adversarial transferability. However, they are mainly verified across different convolutional neural network (CNN) architectures, which is not a reliable evaluation since all CNNs share some similar architectural biases. In this work, we re-evaluate 12 representative transferability-enhancing attack methods where we test on 18 popular models from 4 types of neural networks. Our reevaluation revealed that the adversarial transferability is often overestimated, and there is no single AE that can be transferred to all popular models. The transferability rank of previous attacking methods changes when under our comprehensive evaluation. Based on our analysis, we propose a reliable benchmark including three evaluation protocols. Adversarial transferability on our new benchmark is extremely low, which further confirms the overestimation of adversarial transferability. We release our benchmark at https://adv-trans-eval.github.io to facilitate future research, which includes code, model checkpoints, and evaluation protocols.

摘要: 具有小的对抗性扰动的对抗性示例(AE)可能会将深度神经网络(DNN)误导到错误的预测中。在一个DNN上创建的AE也可以欺骗另一个DNN。在过去的几年里，AE的可转移性引起了人们的极大关注，因为它是促进黑盒攻击的关键属性。已经提出了许多方法来提高对抗性转移能力。然而，它们主要是在不同的卷积神经网络(CNN)结构上进行验证的，这不是一个可靠的评估，因为所有的卷积神经网络都有一些相似的结构偏差。在这项工作中，我们重新评估了12种具有代表性的可转移性增强攻击方法，并在4种神经网络的18个流行模型上进行了测试。我们的重新评估表明，对抗性的可转移性经常被高估，并且没有一个单一的AE可以转移到所有流行的模型。在我们的综合评估下，以往进攻方法的可转换性排名发生了变化。基于我们的分析，我们提出了一个可靠的基准测试，包括三个评估协议。我们新基准的对抗性可转让性极低，这进一步证实了对对抗性可转移性的高估。我们在https://adv-trans-eval.github.io上发布我们的基准测试，以促进未来的研究，其中包括代码、模型检查点和评估协议。



## **11. LMD: A Learnable Mask Network to Detect Adversarial Examples for Speaker Verification**

LMD：一种用于说话人确认的可学习掩码网络 eess.AS

13 pages, 9 figures

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2211.00825v2) [paper-pdf](http://arxiv.org/pdf/2211.00825v2)

**Authors**: Xing Chen, Jie Wang, Xiao-Lei Zhang, Wei-Qiang Zhang, Kunde Yang

**Abstract**: Although the security of automatic speaker verification (ASV) is seriously threatened by recently emerged adversarial attacks, there have been some countermeasures to alleviate the threat. However, many defense approaches not only require the prior knowledge of the attackers but also possess weak interpretability. To address this issue, in this paper, we propose an attacker-independent and interpretable method, named learnable mask detector (LMD), to separate adversarial examples from the genuine ones. It utilizes score variation as an indicator to detect adversarial examples, where the score variation is the absolute discrepancy between the ASV scores of an original audio recording and its transformed audio synthesized from its masked complex spectrogram. A core component of the score variation detector is to generate the masked spectrogram by a neural network. The neural network needs only genuine examples for training, which makes it an attacker-independent approach. Its interpretability lies that the neural network is trained to minimize the score variation of the targeted ASV, and maximize the number of the masked spectrogram bins of the genuine training examples. Its foundation is based on the observation that, masking out the vast majority of the spectrogram bins with little speaker information will inevitably introduce a large score variation to the adversarial example, and a small score variation to the genuine example. Experimental results with 12 attackers and two representative ASV systems show that our proposed method outperforms five state-of-the-art baselines. The extensive experimental results can also be a benchmark for the detection-based ASV defenses.

摘要: 尽管自动说话人确认(ASV)的安全性受到最近出现的敌意攻击的严重威胁，但已经有一些对策来缓解这种威胁。然而，许多防御方法不仅需要攻击者的先验知识，而且具有较弱的可解释性。针对这一问题，本文提出了一种独立于攻击者且可解释的方法，称为可学习掩码检测器(LMD)，用于区分敌意实例和真实实例。它利用分数变化作为检测敌意例子的指标，其中分数变化是原始音频记录的ASV分数与从其掩蔽的复谱图合成的变换音频之间的绝对差异。分数变化检测器的核心部件是通过神经网络生成被屏蔽的谱图。神经网络只需要真实的样本进行训练，这使它成为一种独立于攻击者的方法。它的可解释性在于神经网络被训练来最小化目标ASV的分数差异，并最大化真实训练样本的掩蔽谱图库的数量。它的基础是观察到，在几乎没有说话人信息的情况下掩蔽绝大多数的语谱库将不可避免地给对抗性例子带来大的分数变化，而对真实的例子带来小的分数变化。对12个攻击者和两个有代表性的ASV系统的实验结果表明，我们提出的方法的性能超过了五个最先进的基线。广泛的实验结果也可以作为基于检测的ASV防御的基准。



## **12. COVER: A Heuristic Greedy Adversarial Attack on Prompt-based Learning in Language Models**

封面：对语言模型中基于提示的学习的启发式贪婪对抗性攻击 cs.CL

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.05659v2) [paper-pdf](http://arxiv.org/pdf/2306.05659v2)

**Authors**: Zihao Tan, Qingliang Chen, Wenbin Zhu, Yongjian Huang

**Abstract**: Prompt-based learning has been proved to be an effective way in pre-trained language models (PLMs), especially in low-resource scenarios like few-shot settings. However, the trustworthiness of PLMs is of paramount significance and potential vulnerabilities have been shown in prompt-based templates that could mislead the predictions of language models, causing serious security concerns. In this paper, we will shed light on some vulnerabilities of PLMs, by proposing a prompt-based adversarial attack on manual templates in black box scenarios. First of all, we design character-level and word-level heuristic approaches to break manual templates separately. Then we present a greedy algorithm for the attack based on the above heuristic destructive approaches. Finally, we evaluate our approach with the classification tasks on three variants of BERT series models and eight datasets. And comprehensive experimental results justify the effectiveness of our approach in terms of attack success rate and attack speed. Further experimental studies indicate that our proposed method also displays good capabilities in scenarios with varying shot counts, template lengths and query counts, exhibiting good generalizability.

摘要: 基于提示的学习已被证明是预训练语言模型(PLM)中的一种有效方法，特别是在资源较少的场景中，如少镜头场景。然而，PLM的可信性至关重要，基于提示的模板中已经显示出潜在的漏洞，这些漏洞可能会误导语言模型的预测，导致严重的安全问题。在本文中，我们将通过在黑盒场景中对人工模板提出一种基于提示的对抗性攻击来揭示PLM的一些漏洞。首先，我们分别设计了字字级和词级启发式方法来打破人工模板。在此基础上，提出了一种基于上述启发式破坏性方法的贪婪算法。最后，我们在BERT系列模型的三个变种和八个数据集上对我们的方法进行了评估。综合实验结果从攻击成功率和攻击速度两个方面验证了该方法的有效性。进一步的实验研究表明，该方法在镜头数、模板长度和查询次数不同的场景中也表现出了良好的性能，表现出良好的泛化能力。



## **13. A Relaxed Optimization Approach for Adversarial Attacks against Neural Machine Translation Models**

神经机器翻译模型对抗性攻击的一种松弛优化方法 cs.CL

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08492v1) [paper-pdf](http://arxiv.org/pdf/2306.08492v1)

**Authors**: Sahar Sadrizadeh, Clément Barbier, Ljiljana Dolamic, Pascal Frossard

**Abstract**: In this paper, we propose an optimization-based adversarial attack against Neural Machine Translation (NMT) models. First, we propose an optimization problem to generate adversarial examples that are semantically similar to the original sentences but destroy the translation generated by the target NMT model. This optimization problem is discrete, and we propose a continuous relaxation to solve it. With this relaxation, we find a probability distribution for each token in the adversarial example, and then we can generate multiple adversarial examples by sampling from these distributions. Experimental results show that our attack significantly degrades the translation quality of multiple NMT models while maintaining the semantic similarity between the original and adversarial sentences. Furthermore, our attack outperforms the baselines in terms of success rate, similarity preservation, effect on translation quality, and token error rate. Finally, we propose a black-box extension of our attack by sampling from an optimized probability distribution for a reference model whose gradients are accessible.

摘要: 本文针对神经机器翻译(NMT)模型提出了一种基于优化的敌意攻击方法。首先，我们提出了一个优化问题，以生成与原始句子语义相似但破坏目标NMT模型生成的翻译的对抗性实例。这个优化问题是离散的，我们提出了一种连续松弛法来解决它。通过这种松弛，我们找到了对抗性实例中每个令牌的概率分布，然后我们可以从这些分布中采样来生成多个对抗性实例。实验结果表明，我们的攻击在保持原始句子和对抗性句子之间的语义相似性的同时，显著降低了多个自然机器翻译模型的翻译质量。此外，我们的攻击在成功率、相似性保持、对翻译质量的影响和令牌错误率方面都优于基线。最后，我们提出了我们的攻击的一个黑盒扩展，通过从一个梯度可访问的参考模型的优化概率分布中采样来实现。



## **14. X-Detect: Explainable Adversarial Patch Detection for Object Detectors in Retail**

X-Detect：零售业目标检测器的可解释敌意补丁检测 cs.CV

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08422v1) [paper-pdf](http://arxiv.org/pdf/2306.08422v1)

**Authors**: Omer Hofman, Amit Giloni, Yarin Hayun, Ikuya Morikawa, Toshiya Shimizu, Yuval Elovici, Asaf Shabtai

**Abstract**: Object detection models, which are widely used in various domains (such as retail), have been shown to be vulnerable to adversarial attacks. Existing methods for detecting adversarial attacks on object detectors have had difficulty detecting new real-life attacks. We present X-Detect, a novel adversarial patch detector that can: i) detect adversarial samples in real time, allowing the defender to take preventive action; ii) provide explanations for the alerts raised to support the defender's decision-making process, and iii) handle unfamiliar threats in the form of new attacks. Given a new scene, X-Detect uses an ensemble of explainable-by-design detectors that utilize object extraction, scene manipulation, and feature transformation techniques to determine whether an alert needs to be raised. X-Detect was evaluated in both the physical and digital space using five different attack scenarios (including adaptive attacks) and the COCO dataset and our new Superstore dataset. The physical evaluation was performed using a smart shopping cart setup in real-world settings and included 17 adversarial patch attacks recorded in 1,700 adversarial videos. The results showed that X-Detect outperforms the state-of-the-art methods in distinguishing between benign and adversarial scenes for all attack scenarios while maintaining a 0% FPR (no false alarms) and providing actionable explanations for the alerts raised. A demo is available.

摘要: 目标检测模型被广泛应用于各个领域(如零售)，已被证明容易受到对手攻击。现有的用于检测对象检测器上的敌意攻击的方法已经很难检测到新的现实生活中的攻击。我们提出了X-Detect，这是一种新型的对抗性补丁检测器，它可以：i)实时检测对手样本，允许防御者采取预防措施；ii)为支持防御者决策过程而发出的警报提供解释；iii)处理新攻击形式的陌生威胁。给定一个新场景，X-Detect使用一组可通过设计解释的检测器，这些检测器利用对象提取、场景操作和特征转换技术来确定是否需要发出警报。X-Detect在物理和数字空间中使用五种不同的攻击场景(包括自适应攻击)以及Coco数据集和我们新的Superstore数据集进行了评估。物理评估是使用真实世界设置中的智能购物车进行的，包括1700个对抗性视频中记录的17个对抗性补丁攻击。结果表明，X-Detect在区分所有攻击场景的良性和敌意场景方面优于最先进的方法，同时保持0%的FPR(无错误警报)，并为发出的警报提供可行的解释。现已提供演示。



## **15. Global-Local Processing in Convolutional Neural Networks**

卷积神经网络中的全局-局部处理 cs.CV

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08336v1) [paper-pdf](http://arxiv.org/pdf/2306.08336v1)

**Authors**: Zahra Rezvani, Soroor Shekarizeh, Mohammad Sabokrou

**Abstract**: Convolutional Neural Networks (CNNs) have achieved outstanding performance on image processing challenges. Actually, CNNs imitate the typically developed human brain structures at the micro-level (Artificial neurons). At the same time, they distance themselves from imitating natural visual perception in humans at the macro architectures (high-level cognition). Recently it has been investigated that CNNs are highly biased toward local features and fail to detect the global aspects of their input. Nevertheless, the literature offers limited clues on this problem. To this end, we propose a simple yet effective solution inspired by the unconscious behavior of the human pupil. We devise a simple module called Global Advantage Stream (GAS) to learn and capture the holistic features of input samples (i.e., the global features). Then, the GAS features were combined with a CNN network as a plug-and-play component called the Global/Local Processing (GLP) model. The experimental results confirm that this stream improves the accuracy with an insignificant additional computational/temporal load and makes the network more robust to adversarial attacks. Furthermore, investigating the interpretation of the model shows that it learns a more holistic representation similar to the perceptual system of healthy humans

摘要: 卷积神经网络(CNN)在图像处理方面取得了优异的性能。实际上，CNN在微观层面上模仿了典型的人类大脑结构(人工神经元)。与此同时，他们在宏观架构(高级认知)上与模仿人类的自然视觉知觉保持距离。最近的研究表明，CNN高度偏向于局部特征，并且无法检测其输入的全局方面。然而，文献对这个问题提供的线索有限。为此，我们提出了一种简单而有效的解决方案，灵感来自于人类的无意识行为。我们设计了一个名为Global Advantage Stream(GAS)的简单模块来学习和捕获输入样本的整体特征(即全局特征)。然后，将GAS功能与CNN网络组合为称为全局/本地处理(GLP)模型的即插即用组件。实验结果证实，该流在不增加计算/时间开销的情况下提高了准确率，并使网络对对手攻击具有更强的鲁棒性。此外，研究该模型的解释表明，它学习了一种更整体的表示，类似于健康人类的感知系统



## **16. On the Robustness of Latent Diffusion Models**

关于潜在扩散模型的稳健性 cs.CV

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08257v1) [paper-pdf](http://arxiv.org/pdf/2306.08257v1)

**Authors**: Jianping Zhang, Zhuoer Xu, Shiwen Cui, Changhua Meng, Weibin Wu, Michael R. Lyu

**Abstract**: Latent diffusion models achieve state-of-the-art performance on a variety of generative tasks, such as image synthesis and image editing. However, the robustness of latent diffusion models is not well studied. Previous works only focus on the adversarial attacks against the encoder or the output image under white-box settings, regardless of the denoising process. Therefore, in this paper, we aim to analyze the robustness of latent diffusion models more thoroughly. We first study the influence of the components inside latent diffusion models on their white-box robustness. In addition to white-box scenarios, we evaluate the black-box robustness of latent diffusion models via transfer attacks, where we consider both prompt-transfer and model-transfer settings and possible defense mechanisms. However, all these explorations need a comprehensive benchmark dataset, which is missing in the literature. Therefore, to facilitate the research of the robustness of latent diffusion models, we propose two automatic dataset construction pipelines for two kinds of image editing models and release the whole dataset. Our code and dataset are available at \url{https://github.com/jpzhang1810/LDM-Robustness}.

摘要: 潜在扩散模型在各种生成性任务中实现了最先进的性能，例如图像合成和图像编辑。然而，潜扩散模型的稳健性还没有得到很好的研究。以往的工作只关注白盒环境下对编码器或输出图像的敌意攻击，而没有考虑去噪过程。因此，在本文中，我们旨在更深入地分析潜在扩散模型的稳健性。我们首先研究了潜扩散模型中各分量对其白盒稳健性的影响。除了白盒场景外，我们还通过传输攻击评估了潜在扩散模型的黑盒稳健性，其中我们同时考虑了提示传输和模型传输设置以及可能的防御机制。然而，所有这些探索都需要一个全面的基准数据集，而文献中缺少这一数据集。因此，为了便于研究潜在扩散模型的健壮性，我们针对两种图像编辑模型提出了两种数据集自动构建流水线，并发布了整个数据集。我们的代码和数据集可在\url{https://github.com/jpzhang1810/LDM-Robustness}.上获得



## **17. CARSO: Counter-Adversarial Recall of Synthetic Observations**

卡索：合成观察的反对抗性召回 cs.CV

20 pages, 5 figures, 10 tables; Update: removed visual artifacts from  some figures, fixed typos/capitalisation, typographic/pagination improvements

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.06081v2) [paper-pdf](http://arxiv.org/pdf/2306.06081v2)

**Authors**: Emanuele Ballarin, Alessio Ansuini, Luca Bortolussi

**Abstract**: In this paper, we propose a novel adversarial defence mechanism for image classification -- CARSO -- inspired by cues from cognitive neuroscience. The method is synergistically complementary to adversarial training and relies on knowledge of the internal representation of the attacked classifier. Exploiting a generative model for adversarial purification, conditioned on such representation, it samples reconstructions of inputs to be finally classified. Experimental evaluation by a well-established benchmark of varied, strong adaptive attacks, across diverse image datasets and classifier architectures, shows that CARSO is able to defend the classifier significantly better than state-of-the-art adversarial training alone -- with a tolerable clean accuracy toll. Furthermore, the defensive architecture succeeds in effectively shielding itself from unforeseen threats, and end-to-end attacks adapted to fool stochastic defences. Code and pre-trained models are available at https://github.com/emaballarin/CARSO .

摘要: 在这篇文章中，我们提出了一种新的图像分类对抗性防御机制--CARSO--受认知神经科学的启发。该方法是对抗性训练的协同补充，并依赖于被攻击分类器的内部表示的知识。利用生成模型进行对抗性净化，在这种表示的条件下，对待最终分类的输入的重构进行采样。通过对不同图像数据集和分类器体系结构的各种强大自适应攻击的成熟基准进行的实验评估表明，CARSO能够比仅使用最先进的对手训练更好地防御分类器--并且具有可容忍的干净准确性代价。此外，防御体系结构成功地有效地保护自己免受不可预见的威胁，以及适合愚弄随机防御的端到端攻击。代码和预先培训的模型可在https://github.com/emaballarin/CARSO上找到。



## **18. White-Box Adversarial Policies in Deep Reinforcement Learning**

深度强化学习中的白盒对抗策略 cs.AI

Code is available at  https://github.com/thestephencasper/white_box_rarl

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2209.02167v2) [paper-pdf](http://arxiv.org/pdf/2209.02167v2)

**Authors**: Stephen Casper, Taylor Killian, Gabriel Kreiman, Dylan Hadfield-Menell

**Abstract**: In reinforcement learning (RL), adversarial policies can be developed by training an adversarial agent to minimize a target agent's rewards. Prior work has studied black-box versions of these attacks where the adversary only observes the world state and treats the target agent as any other part of the environment. However, this does not take into account additional structure in the problem. In this work, we take inspiration from the literature on white-box attacks to train more effective adversarial policies. We study white-box adversarial policies and show that having access to a target agent's internal state can be useful for identifying its vulnerabilities. We make two contributions. (1) We introduce white-box adversarial policies where an attacker observes both a target's internal state and the world state at each timestep. We formulate ways of using these policies to attack agents in 2-player games and text-generating language models. (2) We demonstrate that these policies can achieve higher initial and asymptotic performance against a target agent than black-box controls. Code is available at https://github.com/thestephencasper/lm_white_box_attacks

摘要: 在强化学习(RL)中，可以通过训练对抗代理来制定对抗策略，以最小化目标代理的回报。以前的工作已经研究了这些攻击的黑盒版本，其中对手只观察世界状态，并将目标代理视为环境的任何其他部分。然而，这没有考虑到问题中的额外结构。在这项工作中，我们从白盒攻击的文献中获得灵感，以训练更有效的对抗策略。我们研究了白盒对抗策略，并表明访问目标代理的内部状态有助于识别其漏洞。我们做出了两项贡献。(1)我们引入了白盒对抗策略，其中攻击者在每个时间步同时观察目标的内部状态和世界状态。我们制定了使用这些策略攻击双人游戏中的代理和文本生成语言模型的方法。(2)我们证明了这些策略可以获得比黑箱控制更高的初始性能和针对目标代理的渐近性能。代码可在https://github.com/thestephencasper/lm_white_box_attacks上找到



## **19. Class Attribute Inference Attacks: Inferring Sensitive Class Information by Diffusion-Based Attribute Manipulations**

类别属性推断攻击：通过基于扩散的属性操作推断敏感类别信息 cs.LG

46 pages, 37 figures, 5 tables

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2303.09289v2) [paper-pdf](http://arxiv.org/pdf/2303.09289v2)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Felix Friedrich, Manuel Brack, Patrick Schramowski, Kristian Kersting

**Abstract**: Neural network-based image classifiers are powerful tools for computer vision tasks, but they inadvertently reveal sensitive attribute information about their classes, raising concerns about their privacy. To investigate this privacy leakage, we introduce the first Class Attribute Inference Attack (CAIA), which leverages recent advances in text-to-image synthesis to infer sensitive attributes of individual classes in a black-box setting, while remaining competitive with related white-box attacks. Our extensive experiments in the face recognition domain show that CAIA can accurately infer undisclosed sensitive attributes, such as an individual's hair color, gender, and racial appearance, which are not part of the training labels. Interestingly, we demonstrate that adversarial robust models are even more vulnerable to such privacy leakage than standard models, indicating that a trade-off between robustness and privacy exists.

摘要: 基于神经网络的图像分类器是计算机视觉任务的强大工具，但它们无意中泄露了有关其类别的敏感属性信息，引发了对其隐私的担忧。为了调查这种隐私泄露，我们引入了第一类属性推理攻击(CAIA)，它利用文本到图像合成的最新进展来推断黑盒环境中个别类的敏感属性，同时保持与相关白盒攻击的竞争力。我们在人脸识别领域的广泛实验表明，CAIA可以准确地推断出未披露的敏感属性，如个人的头发颜色、性别和种族外观，这些属性不属于训练标签的一部分。有趣的是，我们证明了对抗性稳健模型比标准模型更容易受到这种隐私泄露的影响，这表明存在稳健性和隐私之间的权衡。



## **20. Finite Gaussian Neurons: Defending against adversarial attacks by making neural networks say "I don't know"**

有限高斯神经元：通过让神经网络说出“我不知道”来防御敌意攻击 cs.LG

PhD thesis

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07796v1) [paper-pdf](http://arxiv.org/pdf/2306.07796v1)

**Authors**: Felix Grezes

**Abstract**: Since 2014, artificial neural networks have been known to be vulnerable to adversarial attacks, which can fool the network into producing wrong or nonsensical outputs by making humanly imperceptible alterations to inputs. While defenses against adversarial attacks have been proposed, they usually involve retraining a new neural network from scratch, a costly task. In this work, I introduce the Finite Gaussian Neuron (FGN), a novel neuron architecture for artificial neural networks. My works aims to: - easily convert existing models to Finite Gaussian Neuron architecture, - while preserving the existing model's behavior on real data, - and offering resistance against adversarial attacks. I show that converted and retrained Finite Gaussian Neural Networks (FGNN) always have lower confidence (i.e., are not overconfident) in their predictions over randomized and Fast Gradient Sign Method adversarial images when compared to classical neural networks, while maintaining high accuracy and confidence over real MNIST images. To further validate the capacity of Finite Gaussian Neurons to protect from adversarial attacks, I compare the behavior of FGNs to that of Bayesian Neural Networks against both randomized and adversarial images, and show how the behavior of the two architectures differs. Finally I show some limitations of the FGN models by testing them on the more complex SPEECHCOMMANDS task, against the stronger Carlini-Wagner and Projected Gradient Descent adversarial attacks.

摘要: 自2014年以来，人工神经网络一直被认为容易受到对抗性攻击，这些攻击可以通过对输入进行人类无法察觉的改变来愚弄网络产生错误或毫无意义的输出。虽然有人提出了防御对手攻击的建议，但它们通常涉及从头开始重新训练新的神经网络，这是一项代价高昂的任务。在这项工作中，我介绍了有限高斯神经元(FGN)，一种新的人工神经网络的神经元结构。我的工作目标是：-轻松地将现有模型转换为有限的高斯神经元架构-同时保留现有模型在真实数据上的行为-并提供对对手攻击的抵抗。结果表明，与经典神经网络相比，经过转换和再训练的有限高斯神经网络(FGNN)在对随机和快速梯度符号法对手图像的预测中总是具有较低的置信度(即不过分自信)，而在真实MNIST图像上保持较高的精度和置信度。为了进一步验证有限高斯神经元抵御敌意攻击的能力，我比较了FGNs和贝叶斯神经网络在随机图像和敌意图像下的行为，并展示了这两种体系结构的行为如何不同。最后，我通过在更复杂的SPEECHCOMMANDS任务中测试FGN模型的一些局限性，对抗更强大的Carlini-Wagner和预测的梯度下降对手攻击。



## **21. Area is all you need: repeatable elements make stronger adversarial attacks**

面积就是你所需要的：可重复的元素构成更强的对抗性攻击 cs.CV

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07768v1) [paper-pdf](http://arxiv.org/pdf/2306.07768v1)

**Authors**: Dillon Niederhut

**Abstract**: Over the last decade, deep neural networks have achieved state of the art in computer vision tasks. These models, however, are susceptible to unusual inputs, known as adversarial examples, that cause them to misclassify or otherwise fail to detect objects. Here, we provide evidence that the increasing success of adversarial attacks is primarily due to increasing their size. We then demonstrate a method for generating the largest possible adversarial patch by building a adversarial pattern out of repeatable elements. This approach achieves a new state of the art in evading detection by YOLOv2 and YOLOv3. Finally, we present an experiment that fails to replicate the prior success of several attacks published in this field, and end with some comments on testing and reproducibility.

摘要: 在过去的十年里，深度神经网络在计算机视觉任务中达到了最先进的水平。然而，这些模型很容易受到异常输入的影响，这些输入被称为对抗性示例，导致它们错误分类或无法检测到对象。在这里，我们提供的证据表明，对抗性攻击的日益成功主要是由于其规模的增加。然后，我们演示了一种通过从可重复元素中构建对抗性模式来生成可能最大的对抗性补丁的方法。该方法在躲避YOLOv2和YOLOv3的检测方面达到了新的技术水平。最后，我们给出了一个未能复制该领域已发表的几种攻击的先前成功的实验，并以对测试和重复性的一些评论结束。



## **22. PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

PromptBitch：评估大型语言模型在对抗性提示下的稳健性 cs.CL

Technical report; 23 pages; code is at:  https://github.com/microsoft/promptbench

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.04528v2) [paper-pdf](http://arxiv.org/pdf/2306.04528v2)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Neil Zhenqiang Gong, Yue Zhang, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptBench, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. These prompts are then employed in diverse tasks, such as sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4,032 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets, with 567,084 test samples in total. Our findings demonstrate that contemporary LLMs are vulnerable to adversarial prompts. Furthermore, we present comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users. We make our code, prompts, and methodologies to generate adversarial prompts publicly accessible, thereby enabling and encouraging collaborative exploration in this pivotal field: https://github.com/microsoft/promptbench.

摘要: 学术界和工业界对大型语言模型(LLM)的依赖日益增加，这就要求我们必须全面了解它们对提示的稳健性。为了响应这一关键需求，我们引入了PromptBtch，这是一个健壮性基准，旨在衡量LLMS对敌意提示的弹性。这项研究使用了过多的对抗性文本攻击，目标是多个层面的提示：字符、单词、句子和语义。这些提示随后被用于不同的任务，如情感分析、自然语言推理、阅读理解、机器翻译和数学解题。我们的研究产生了4032个对抗性提示，仔细评估了8个任务和13个数据集，总共有567,084个测试样本。我们的研究结果表明，当代的LLM容易受到对抗性提示的影响。此外，我们还提供了全面的分析，以了解即时健壮性及其可转移性背后的奥秘。然后，我们提供了有洞察力的健壮性分析和实用的即时撰写建议，这对研究人员和日常用户都是有益的。我们将生成对抗性提示的代码、提示和方法公之于众，从而支持并鼓励在这个关键领域进行协作探索：https://github.com/microsoft/promptbench.



## **23. Privacy Inference-Empowered Stealthy Backdoor Attack on Federated Learning under Non-IID Scenarios**

隐私推理--非IID场景下联合学习的隐形后门攻击 cs.LG

It can be accepted IJCNN

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.08011v1) [paper-pdf](http://arxiv.org/pdf/2306.08011v1)

**Authors**: Haochen Mei, Gaolei Li, Jun Wu, Longfei Zheng

**Abstract**: Federated learning (FL) naturally faces the problem of data heterogeneity in real-world scenarios, but this is often overlooked by studies on FL security and privacy. On the one hand, the effectiveness of backdoor attacks on FL may drop significantly under non-IID scenarios. On the other hand, malicious clients may steal private data through privacy inference attacks. Therefore, it is necessary to have a comprehensive perspective of data heterogeneity, backdoor, and privacy inference. In this paper, we propose a novel privacy inference-empowered stealthy backdoor attack (PI-SBA) scheme for FL under non-IID scenarios. Firstly, a diverse data reconstruction mechanism based on generative adversarial networks (GANs) is proposed to produce a supplementary dataset, which can improve the attacker's local data distribution and support more sophisticated strategies for backdoor attacks. Based on this, we design a source-specified backdoor learning (SSBL) strategy as a demonstration, allowing the adversary to arbitrarily specify which classes are susceptible to the backdoor trigger. Since the PI-SBA has an independent poisoned data synthesis process, it can be integrated into existing backdoor attacks to improve their effectiveness and stealthiness in non-IID scenarios. Extensive experiments based on MNIST, CIFAR10 and Youtube Aligned Face datasets demonstrate that the proposed PI-SBA scheme is effective in non-IID FL and stealthy against state-of-the-art defense methods.

摘要: 联邦学习(FL)自然会面临现实场景中数据异构性的问题，但这一点往往被FL安全和隐私方面的研究所忽视。一方面，在非IID场景下，对FL的后门攻击效果可能会大幅下降。另一方面，恶意客户端可能会通过隐私推理攻击窃取隐私数据。因此，有必要对数据异构性、后门和隐私推断有一个全面的视角。提出了一种新的基于隐私推理的隐蔽后门攻击方案(PI-SBA)，用于非IID场景下的FL攻击。首先，提出了一种基于产生式对抗网络(GANS)的多样化数据重构机制，生成一个补充数据集，改善攻击者的局部数据分布，支持更复杂的后门攻击策略。在此基础上，我们设计了一种来源指定的后门学习(SSBL)策略作为演示，允许攻击者任意指定哪些类容易受到后门触发器的影响。由于PI-SBA具有独立的有毒数据合成过程，因此可以将其集成到现有的后门攻击中，以提高其在非IID场景中的有效性和隐蔽性。基于MNIST、CIFAR10和YouTube对齐人脸数据集的大量实验表明，所提出的PI-SBA算法在非IID人脸识别中是有效的，并且对现有的防御方法具有较好的隐蔽性。



## **24. Malafide: a novel adversarial convolutive noise attack against deepfake and spoofing detection systems**

恶意：一种新的对抗深度假冒和欺骗检测系统的对抗性卷积噪声攻击 eess.AS

Accepted at INTERSPEECH 2023

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07655v1) [paper-pdf](http://arxiv.org/pdf/2306.07655v1)

**Authors**: Michele Panariello, Wanying Ge, Hemlata Tak, Massimiliano Todisco, Nicholas Evans

**Abstract**: We present Malafide, a universal adversarial attack against automatic speaker verification (ASV) spoofing countermeasures (CMs). By introducing convolutional noise using an optimised linear time-invariant filter, Malafide attacks can be used to compromise CM reliability while preserving other speech attributes such as quality and the speaker's voice. In contrast to other adversarial attacks proposed recently, Malafide filters are optimised independently of the input utterance and duration, are tuned instead to the underlying spoofing attack, and require the optimisation of only a small number of filter coefficients. Even so, they degrade CM performance estimates by an order of magnitude, even in black-box settings, and can also be configured to overcome integrated CM and ASV subsystems. Integrated solutions that use self-supervised learning CMs, however, are more robust, under both black-box and white-box settings.

摘要: 我们提出了一种针对自动说话人验证(ASV)欺骗对策(CMS)的通用对抗性攻击--恶意攻击。通过使用优化的线性时不变滤波器引入卷积噪声，恶意攻击可以用来损害CM的可靠性，同时保留其他语音属性，如质量和说话人的声音。与最近提出的其他敌意攻击不同，恶意过滤器独立于输入发音和持续时间进行优化，而是根据潜在的欺骗攻击进行调整，并且只需要优化少量的过滤器系数。即便如此，即使在黑盒设置中，它们也会将CM性能估计降低一个数量级，并且还可以配置为克服集成的CM和ASV子系统。然而，使用自我监督学习CMS的集成解决方案在黑盒和白盒设置下都更健壮。



## **25. DHBE: Data-free Holistic Backdoor Erasing in Deep Neural Networks via Restricted Adversarial Distillation**

DHBE：基于受限对抗性蒸馏的深度神经网络无数据整体后门擦除 cs.LG

It has been accepted by asiaccs

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.08009v1) [paper-pdf](http://arxiv.org/pdf/2306.08009v1)

**Authors**: Zhicong Yan, Shenghong Li, Ruijie Zhao, Yuan Tian, Yuanyuan Zhao

**Abstract**: Backdoor attacks have emerged as an urgent threat to Deep Neural Networks (DNNs), where victim DNNs are furtively implanted with malicious neurons that could be triggered by the adversary. To defend against backdoor attacks, many works establish a staged pipeline to remove backdoors from victim DNNs: inspecting, locating, and erasing. However, in a scenario where a few clean data can be accessible, such pipeline is fragile and cannot erase backdoors completely without sacrificing model accuracy. To address this issue, in this paper, we propose a novel data-free holistic backdoor erasing (DHBE) framework. Instead of the staged pipeline, the DHBE treats the backdoor erasing task as a unified adversarial procedure, which seeks equilibrium between two different competing processes: distillation and backdoor regularization. In distillation, the backdoored DNN is distilled into a proxy model, transferring its knowledge about clean data, yet backdoors are simultaneously transferred. In backdoor regularization, the proxy model is holistically regularized to prevent from infecting any possible backdoor transferred from distillation. These two processes jointly proceed with data-free adversarial optimization until a clean, high-accuracy proxy model is obtained. With the novel adversarial design, our framework demonstrates its superiority in three aspects: 1) minimal detriment to model accuracy, 2) high tolerance for hyperparameters, and 3) no demand for clean data. Extensive experiments on various backdoor attacks and datasets are performed to verify the effectiveness of the proposed framework. Code is available at \url{https://github.com/yanzhicong/DHBE}

摘要: 后门攻击已经成为对深度神经网络(DNN)的紧迫威胁，受害者DNN被秘密植入可能由对手触发的恶意神经元。为了防御后门攻击，许多工作建立了一个分阶段的管道来删除受害者DNN的后门：检查、定位和擦除。然而，在可以访问少数干净数据的情况下，这样的管道是脆弱的，无法在不牺牲模型精度的情况下完全擦除后门。针对这一问题，本文提出了一种新的无数据整体后门擦除(DHBE)框架。与阶段性管道不同，DHBE将后门擦除任务视为统一的对抗性程序，寻求两个不同竞争过程之间的平衡：蒸馏和后门正规化。在蒸馏过程中，后置的DNN被蒸馏成代理模型，传递其关于干净数据的知识，但同时传递后门。在后门正规化中，代理模型被整体正规化，以防止感染从蒸馏转移的任何可能的后门。这两个过程共同进行无数据的对抗性优化，直到获得干净、高精度的代理模型。通过新的对抗性设计，我们的框架在三个方面显示了其优越性：1)对模型精度的损害最小，2)对超参数的容忍度高，3)不需要干净的数据。在各种后门攻击和数据集上进行了大量的实验，以验证该框架的有效性。代码位于\url{https://github.com/yanzhicong/DHBE}



## **26. A Hypergraph-Based Machine Learning Ensemble Network Intrusion Detection System**

一种基于超图的机器学习集成网络入侵检测系统 cs.CR

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2211.03933v2) [paper-pdf](http://arxiv.org/pdf/2211.03933v2)

**Authors**: Zong-Zhi Lin, Thomas D. Pike, Mark M. Bailey, Nathaniel D. Bastian

**Abstract**: Network intrusion detection systems (NIDS) to detect malicious attacks continue to meet challenges. NIDS are often developed offline while they face auto-generated port scan infiltration attempts, resulting in a significant time lag from adversarial adaption to NIDS response. To address these challenges, we use hypergraphs focused on internet protocol addresses and destination ports to capture evolving patterns of port scan attacks. The derived set of hypergraph-based metrics are then used to train an ensemble machine learning (ML) based NIDS that allows for real-time adaption in monitoring and detecting port scanning activities, other types of attacks, and adversarial intrusions at high accuracy, precision and recall performances. This ML adapting NIDS was developed through the combination of (1) intrusion examples, (2) NIDS update rules, (3) attack threshold choices to trigger NIDS retraining requests, and (4) a production environment with no prior knowledge of the nature of network traffic. 40 scenarios were auto-generated to evaluate the ML ensemble NIDS comprising three tree-based models. The resulting ML Ensemble NIDS was extended and evaluated with the CIC-IDS2017 dataset. Results show that under the model settings of an Update-ALL-NIDS rule (specifically retrain and update all the three models upon the same NIDS retraining request) the proposed ML ensemble NIDS evolved intelligently and produced the best results with nearly 100% detection performance throughout the simulation.

摘要: 网络入侵检测系统(NID)检测恶意攻击的能力不断受到挑战。当NID面临自动生成的端口扫描渗透尝试时，它们通常是离线开发的，导致从对手适应到NIDS响应有很大的时间延迟。为了应对这些挑战，我们使用聚焦于互联网协议地址和目标端口的超图来捕获端口扫描攻击的演变模式。然后，使用导出的基于超图的度量集合来训练基于机器学习(ML)的集成网络入侵检测系统，该集成机器学习系统允许实时适应监视和检测端口扫描活动、其他类型的攻击以及高准确度、精确度和召回性能的敌对入侵。通过(1)入侵实例、(2)网络入侵检测系统更新规则、(3)用于触发网络入侵检测系统再训练请求的攻击阈值选择和(4)不事先知道网络流量性质的生产环境的组合，开发了该ML适应网络入侵检测系统。自动生成了40个场景来评估包含三个基于树的模型的ML集成网络入侵检测系统。使用CIC-IDS2017数据集对所得到的ML集成网络入侵检测系统进行了扩展和评估。结果表明，在更新-全部-网络入侵检测系统规则的模型设置下(根据同一网络入侵检测系统的重新训练请求，具体地对三个模型进行重新训练和更新)，所提出的最大似然集成网络入侵检测系统以智能方式进化，并在整个仿真过程中产生最好的结果，检测性能接近100%。



## **27. Extracting Cloud-based Model with Prior Knowledge**

基于先验知识的云模型提取 cs.CR

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.04192v4) [paper-pdf](http://arxiv.org/pdf/2306.04192v4)

**Authors**: Shiqian Zhao, Kangjie Chen, Meng Hao, Jian Zhang, Guowen Xu, Hongwei Li, Tianwei Zhang

**Abstract**: Machine Learning-as-a-Service, a pay-as-you-go business pattern, is widely accepted by third-party users and developers. However, the open inference APIs may be utilized by malicious customers to conduct model extraction attacks, i.e., attackers can replicate a cloud-based black-box model merely via querying malicious examples. Existing model extraction attacks mainly depend on the posterior knowledge (i.e., predictions of query samples) from Oracle. Thus, they either require high query overhead to simulate the decision boundary, or suffer from generalization errors and overfitting problems due to query budget limitations. To mitigate it, this work proposes an efficient model extraction attack based on prior knowledge for the first time. The insight is that prior knowledge of unlabeled proxy datasets is conducive to the search for the decision boundary (e.g., informative samples). Specifically, we leverage self-supervised learning including autoencoder and contrastive learning to pre-compile the prior knowledge of the proxy dataset into the feature extractor of the substitute model. Then we adopt entropy to measure and sample the most informative examples to query the target model. Our design leverages both prior and posterior knowledge to extract the model and thus eliminates generalizability errors and overfitting problems. We conduct extensive experiments on open APIs like Traffic Recognition, Flower Recognition, Moderation Recognition, and NSFW Recognition from real-world platforms, Azure and Clarifai. The experimental results demonstrate the effectiveness and efficiency of our attack. For example, our attack achieves 95.1% fidelity with merely 1.8K queries (cost 2.16$) on the NSFW Recognition API. Also, the adversarial examples generated with our substitute model have better transferability than others, which reveals that our scheme is more conducive to downstream attacks.

摘要: 机器学习即服务是一种现收现付的商业模式，被第三方用户和开发人员广泛接受。然而，开放推理API可能被恶意客户利用来进行模型提取攻击，即攻击者仅通过查询恶意示例就可以复制基于云的黑盒模型。现有的模型提取攻击主要依赖于Oracle的后验知识(即查询样本的预测)。因此，它们要么需要很高的查询开销来模拟决策边界，要么由于查询预算的限制而存在泛化错误和过适应问题。针对这一问题，本文首次提出了一种基于先验知识的高效模型提取攻击方法。结论是，未标记的代理数据集的先验知识有助于搜索决策边界(例如，信息样本)。具体地说，我们利用包括自动编码器和对比学习在内的自监督学习来将代理数据集的先验知识预编译到替代模型的特征提取器中。然后利用信息熵对最具信息量的实例进行度量和采样，以查询目标模型。我们的设计利用先验和后验知识来提取模型，从而消除了泛化误差和过拟合问题。我们对来自真实平台Azure和Clarifai的流量识别、花卉识别、适度识别和NSFW识别等开放API进行了广泛的实验。实验结果证明了该攻击的有效性和高效性。例如，我们的攻击在NSFW识别API上仅用1.8K个查询(成本2.16美元)就达到了95.1%的保真度。此外，我们的替代模型生成的敌意例子具有更好的可移植性，这表明我们的方案更有利于下游攻击。



## **28. I See Dead People: Gray-Box Adversarial Attack on Image-To-Text Models**

我看到死人：对图像到文本模型的灰箱对抗性攻击 cs.CV

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07591v1) [paper-pdf](http://arxiv.org/pdf/2306.07591v1)

**Authors**: Raz Lapid, Moshe Sipper

**Abstract**: Modern image-to-text systems typically adopt the encoder-decoder framework, which comprises two main components: an image encoder, responsible for extracting image features, and a transformer-based decoder, used for generating captions. Taking inspiration from the analysis of neural networks' robustness against adversarial perturbations, we propose a novel gray-box algorithm for creating adversarial examples in image-to-text models. Unlike image classification tasks that have a finite set of class labels, finding visually similar adversarial examples in an image-to-text task poses greater challenges because the captioning system allows for a virtually infinite space of possible captions. In this paper, we present a gray-box adversarial attack on image-to-text, both untargeted and targeted. We formulate the process of discovering adversarial perturbations as an optimization problem that uses only the image-encoder component, meaning the proposed attack is language-model agnostic. Through experiments conducted on the ViT-GPT2 model, which is the most-used image-to-text model in Hugging Face, and the Flickr30k dataset, we demonstrate that our proposed attack successfully generates visually similar adversarial examples, both with untargeted and targeted captions. Notably, our attack operates in a gray-box manner, requiring no knowledge about the decoder module. We also show that our attacks fool the popular open-source platform Hugging Face.

摘要: 现代图像到文本系统通常采用编解码器框架，该框架包括两个主要组件：负责提取图像特征的图像编码器和用于生成字幕的基于转换器的解码器。从神经网络对对抗性扰动的鲁棒性分析中得到启发，我们提出了一种新的灰盒算法，用于在图像到文本模型中创建对抗性示例。与具有有限类别标签集的图像分类任务不同，在图像到文本的任务中找到视觉上相似的对抗性例子带来了更大的挑战，因为字幕系统允许可能的字幕的几乎无限空间。在本文中，我们提出了一种针对图像到文本的灰盒对抗性攻击，包括无目标攻击和目标攻击。我们将发现敌意扰动的过程描述为一个只使用图像编码器组件的优化问题，这意味着所提出的攻击是语言模型不可知的。通过在拥抱脸中最常用的图文转换模型VIT-GPT2模型和Flickr30k数据集上的实验，我们证明了我们的攻击成功地生成了视觉上相似的对抗性例子，无论是无目标字幕还是有目标字幕。值得注意的是，我们的攻击以灰盒方式运行，不需要了解解码器模块。我们还表明，我们的攻击愚弄了流行的开源平台拥抱脸。



## **29. How Secure is Your Website? A Comprehensive Investigation on CAPTCHA Providers and Solving Services**

您的网站有多安全？验证码提供商和解决方案服务的全面调查 cs.CR

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07543v1) [paper-pdf](http://arxiv.org/pdf/2306.07543v1)

**Authors**: Rui Jin, Lin Huang, Jikang Duan, Wei Zhao, Yong Liao, Pengyuan Zhou

**Abstract**: Completely Automated Public Turing Test To Tell Computers and Humans Apart (CAPTCHA) has been implemented on many websites to identify between harmful automated bots and legitimate users. However, the revenue generated by the bots has turned circumventing CAPTCHAs into a lucrative business. Although earlier studies provided information about text-based CAPTCHAs and the associated CAPTCHA-solving services, a lot has changed in the past decade regarding content, suppliers, and solvers of CAPTCHA. We have conducted a comprehensive investigation of the latest third-party CAPTCHA providers and CAPTCHA-solving services' attacks. We dug into the details of CAPTCHA-As-a-Service and the latest CAPTCHA-solving services and carried out adversarial experiments on CAPTCHAs and CAPTCHA solvers. The experiment results show a worrying fact: most latest CAPTCHAs are vulnerable to both human solvers and automated solvers. New CAPTCHAs based on hard AI problems and behavior analysis are needed to stop CAPTCHA solvers.

摘要: 区分计算机和人类的全自动公共图灵测试(CAPTCHA)已经在许多网站上实施，以识别有害的自动机器人和合法用户。然而，机器人产生的收入已经把绕过验证码变成了一项有利可图的业务。尽管早期的研究提供了有关基于文本的验证码和相关验证码解析服务的信息，但在过去十年中，验证码的内容、供应商和解算器发生了很大变化。我们对最新的第三方验证码提供商和验证码解决服务的攻击进行了全面调查。我们深入研究了验证码即服务和最新的验证码解算服务的细节，并对验证码和验证码解算器进行了对抗性实验。实验结果显示了一个令人担忧的事实：大多数最新的验证码都容易受到人工求解器和自动求解器的攻击。需要基于硬AI问题和行为分析的新验证码来停止验证码解算器。



## **30. Adversarial Attacks on the Interpretation of Neuron Activation Maximization**

对抗性攻击对神经元激活最大化的解释 cs.LG

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.07397v1) [paper-pdf](http://arxiv.org/pdf/2306.07397v1)

**Authors**: Geraldin Nanfack, Alexander Fulleringer, Jonathan Marty, Michael Eickenberg, Eugene Belilovsky

**Abstract**: The internal functional behavior of trained Deep Neural Networks is notoriously difficult to interpret. Activation-maximization approaches are one set of techniques used to interpret and analyze trained deep-learning models. These consist in finding inputs that maximally activate a given neuron or feature map. These inputs can be selected from a data set or obtained by optimization. However, interpretability methods may be subject to being deceived. In this work, we consider the concept of an adversary manipulating a model for the purpose of deceiving the interpretation. We propose an optimization framework for performing this manipulation and demonstrate a number of ways that popular activation-maximization interpretation techniques associated with CNNs can be manipulated to change the interpretations, shedding light on the reliability of these methods.

摘要: 众所周知，经过训练的深度神经网络的内部功能行为很难解释。激活最大化方法是一套用于解释和分析训练有素的深度学习模型的技术。这包括寻找最大限度地激活给定神经元或特征映射的输入。这些输入可以从数据集中选择或通过优化获得。然而，可解释性方法可能会受到欺骗。在这项工作中，我们考虑了对手操纵模型以欺骗解释的概念。我们提出了一个执行这种操作的优化框架，并展示了与CNN相关的流行的激活最大化解释技术可以被操作以改变解释的一些方法，从而揭示了这些方法的可靠性。



## **31. Gaussian Membership Inference Privacy**

高斯隶属度推理隐私性 cs.LG

The first two authors contributed equally

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.07273v1) [paper-pdf](http://arxiv.org/pdf/2306.07273v1)

**Authors**: Tobias Leemann, Martin Pawelczyk, Gjergji Kasneci

**Abstract**: We propose a new privacy notion called $f$-Membership Inference Privacy ($f$-MIP), which explicitly considers the capabilities of realistic adversaries under the membership inference attack threat model. By doing so $f$-MIP offers interpretable privacy guarantees and improved utility (e.g., better classification accuracy). Our novel theoretical analysis of likelihood ratio-based membership inference attacks on noisy stochastic gradient descent (SGD) results in a parametric family of $f$-MIP guarantees that we refer to as $\mu$-Gaussian Membership Inference Privacy ($\mu$-GMIP). Our analysis additionally yields an analytical membership inference attack that offers distinct advantages over previous approaches. First, unlike existing methods, our attack does not require training hundreds of shadow models to approximate the likelihood ratio. Second, our analytical attack enables straightforward auditing of our privacy notion $f$-MIP. Finally, our analysis emphasizes the importance of various factors, such as hyperparameters (e.g., batch size, number of model parameters) and data specific characteristics in controlling an attacker's success in reliably inferring a given point's membership to the training set. We demonstrate the effectiveness of our method on models trained across vision and tabular datasets.

摘要: 我们提出了一种新的隐私概念$f$-MIP($f$-MIP)，它显式地考虑了现实对手在成员关系推理攻击威胁模型下的能力。通过这样做，$f$-MIP提供了可解释的隐私保证和改进的实用性(例如，更好的分类准确性)。我们对噪声随机梯度下降(SGD)上基于似然比的成员推理攻击进行了新的理论分析，得到了一个由$f$-MIP保证组成的参数族，我们称之为$\MU$-高斯成员关系推理隐私($\MU$-GMIP)。我们的分析还产生了一种分析性成员关系推理攻击，与以前的方法相比具有明显的优势。首先，与现有方法不同，我们的攻击不需要训练数百个阴影模型来逼近似然比。其次，我们的分析攻击使我们能够直接审计我们的隐私概念$f$-mip。最后，我们的分析强调了各种因素的重要性，例如超参数(例如，批次大小、模型参数的数量)和数据特定特征，以控制攻击者在可靠地推断给定点的训练集的成员资格方面的成功。我们在视觉和表格数据集上训练的模型上展示了我们方法的有效性。



## **32. When Vision Fails: Text Attacks Against ViT and OCR**

当视觉失败：针对VIT和OCR的文本攻击 cs.CR

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.07033v1) [paper-pdf](http://arxiv.org/pdf/2306.07033v1)

**Authors**: Nicholas Boucher, Jenny Blessing, Ilia Shumailov, Ross Anderson, Nicolas Papernot

**Abstract**: While text-based machine learning models that operate on visual inputs of rendered text have become robust against a wide range of existing attacks, we show that they are still vulnerable to visual adversarial examples encoded as text. We use the Unicode functionality of combining diacritical marks to manipulate encoded text so that small visual perturbations appear when the text is rendered. We show how a genetic algorithm can be used to generate visual adversarial examples in a black-box setting, and conduct a user study to establish that the model-fooling adversarial examples do not affect human comprehension. We demonstrate the effectiveness of these attacks in the real world by creating adversarial examples against production models published by Facebook, Microsoft, IBM, and Google.

摘要: 虽然基于文本的机器学习模型对呈现文本的可视输入已经变得对广泛的现有攻击具有健壮性，但我们表明它们仍然容易受到编码为文本的可视对抗性示例的攻击。我们使用组合变音符号的Unicode功能来操作编码文本，以便在呈现文本时出现小的视觉干扰。我们展示了如何使用遗传算法在黑盒环境中生成可视对抗性示例，并进行了用户研究以确定愚弄模型的对抗性示例不会影响人类的理解。我们通过创建针对Facebook、Microsoft、IBM和Google发布的生产模型的对抗性示例，在现实世界中演示了这些攻击的有效性。



## **33. A Linear Reconstruction Approach for Attribute Inference Attacks against Synthetic Data**

一种针对合成数据的属性推理攻击的线性重构方法 cs.LG

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2301.10053v2) [paper-pdf](http://arxiv.org/pdf/2301.10053v2)

**Authors**: Meenatchi Sundaram Muthu Selva Annamalai, Andrea Gadotti, Luc Rocher

**Abstract**: Recent advances in synthetic data generation (SDG) have been hailed as a solution to the difficult problem of sharing sensitive data while protecting privacy. SDG aims to learn statistical properties of real data in order to generate "artificial" data that are structurally and statistically similar to sensitive data. However, prior research suggests that inference attacks on synthetic data can undermine privacy, but only for specific outlier records. In this work, we introduce a new attribute inference attack against synthetic data. The attack is based on linear reconstruction methods for aggregate statistics, which target all records in the dataset, not only outliers. We evaluate our attack on state-of-the-art SDG algorithms, including Probabilistic Graphical Models, Generative Adversarial Networks, and recent differentially private SDG mechanisms. By defining a formal privacy game, we show that our attack can be highly accurate even on arbitrary records, and that this is the result of individual information leakage (as opposed to population-level inference). We then systematically evaluate the tradeoff between protecting privacy and preserving statistical utility. Our findings suggest that current SDG methods cannot consistently provide sufficient privacy protection against inference attacks while retaining reasonable utility. The best method evaluated, a differentially private SDG mechanism, can provide both protection against inference attacks and reasonable utility, but only in very specific settings. Lastly, we show that releasing a larger number of synthetic records can improve utility but at the cost of making attacks far more effective.

摘要: 合成数据生成(SDG)的最新进展被誉为在保护隐私的同时共享敏感数据这一难题的解决方案。SDG旨在学习真实数据的统计属性，以便生成在结构和统计上与敏感数据相似的“人造”数据。然而，先前的研究表明，对合成数据的推理攻击可能会破坏隐私，但仅限于特定的离群值记录。在这项工作中，我们引入了一种新的针对合成数据的属性推理攻击。该攻击基于聚合统计的线性重建方法，其目标是数据集中的所有记录，而不仅仅是离群值。我们评估了我们对最先进的SDG算法的攻击，包括概率图形模型、生成性对手网络和最近的差异私有SDG机制。通过定义一个正式的隐私游戏，我们证明了我们的攻击即使在任意记录上也可以非常准确，并且这是个人信息泄露的结果(与总体级别的推断相反)。然后，我们系统地评估了保护隐私和保护统计效用之间的权衡。我们的发现表明，现有的SDG方法在保持合理效用的同时，不能始终如一地提供足够的隐私保护来抵御推理攻击。评估的最佳方法是一种不同的私有SDG机制，它可以提供对推理攻击的保护和合理的实用程序，但只能在非常特定的环境中提供。最后，我们表明，发布更多的合成记录可以提高实用性，但代价是使攻击更加有效。



## **34. How robust accuracy suffers from certified training with convex relaxations**

带凸松弛的认证训练对稳健精度的影响 cs.LG

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.06995v1) [paper-pdf](http://arxiv.org/pdf/2306.06995v1)

**Authors**: Piersilvio De Bartolomeis, Jacob Clarysse, Amartya Sanyal, Fanny Yang

**Abstract**: Adversarial attacks pose significant threats to deploying state-of-the-art classifiers in safety-critical applications. Two classes of methods have emerged to address this issue: empirical defences and certified defences. Although certified defences come with robustness guarantees, empirical defences such as adversarial training enjoy much higher popularity among practitioners. In this paper, we systematically compare the standard and robust error of these two robust training paradigms across multiple computer vision tasks. We show that in most tasks and for both $\mathscr{l}_\infty$-ball and $\mathscr{l}_2$-ball threat models, certified training with convex relaxations suffers from worse standard and robust error than adversarial training. We further explore how the error gap between certified and adversarial training depends on the threat model and the data distribution. In particular, besides the perturbation budget, we identify as important factors the shape of the perturbation set and the implicit margin of the data distribution. We support our arguments with extensive ablations on both synthetic and image datasets.

摘要: 对抗性攻击对在安全关键型应用中部署最先进的分类器构成了重大威胁。解决这一问题的方法有两类：经验性辩护和证明性辩护。虽然认证的防御具有健壮性保证，但经验防御，如对抗性训练，在从业者中享有更高的受欢迎程度。在本文中，我们系统地比较了这两种稳健训练范例在多个计算机视觉任务中的标准误差和稳健误差。结果表明，在大多数任务中，对于$\mathscr{L}_inty$-ball和$\mathscr{L}_2$-ball威胁模型，带凸松弛的认证训练的标准误差和稳健误差都比对抗性训练差。我们进一步探讨认证训练和对抗性训练之间的错误差距如何取决于威胁模型和数据分布。特别是，除了扰动预算之外，我们还将扰动集合的形状和数据分布的隐含裕度确定为重要因素。我们通过在合成数据集和图像数据集上进行广泛的消融来支持我们的论点。



## **35. Backdooring Neural Code Search**

回溯神经编码搜索 cs.SE

Accepted to the 61st Annual Meeting of the Association for  Computational Linguistics (ACL 2023)

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2305.17506v2) [paper-pdf](http://arxiv.org/pdf/2305.17506v2)

**Authors**: Weisong Sun, Yuchen Chen, Guanhong Tao, Chunrong Fang, Xiangyu Zhang, Quanjun Zhang, Bin Luo

**Abstract**: Reusing off-the-shelf code snippets from online repositories is a common practice, which significantly enhances the productivity of software developers. To find desired code snippets, developers resort to code search engines through natural language queries. Neural code search models are hence behind many such engines. These models are based on deep learning and gain substantial attention due to their impressive performance. However, the security aspect of these models is rarely studied. Particularly, an adversary can inject a backdoor in neural code search models, which return buggy or even vulnerable code with security/privacy issues. This may impact the downstream software (e.g., stock trading systems and autonomous driving) and cause financial loss and/or life-threatening incidents. In this paper, we demonstrate such attacks are feasible and can be quite stealthy. By simply modifying one variable/function name, the attacker can make buggy/vulnerable code rank in the top 11%. Our attack BADCODE features a special trigger generation and injection procedure, making the attack more effective and stealthy. The evaluation is conducted on two neural code search models and the results show our attack outperforms baselines by 60%. Our user study demonstrates that our attack is more stealthy than the baseline by two times based on the F1 score.

摘要: 重用在线存储库中的现成代码片段是一种常见的做法，这显著提高了软件开发人员的工作效率。为了找到所需的代码片段，开发人员通过自然语言查询求助于代码搜索引擎。因此，神经代码搜索模型是许多此类引擎的幕后推手。这些模型是基于深度学习的，由于其令人印象深刻的性能而获得了大量关注。然而，这些模型的安全性方面的研究很少。特别是，攻击者可以在神经代码搜索模型中注入后门，该模型返回带有安全/隐私问题的错误代码，甚至是易受攻击的代码。这可能会影响下游软件(例如股票交易系统和自动驾驶)，并导致经济损失和/或危及生命的事件。在这篇文章中，我们证明了这种攻击是可行的，并且可以相当隐蔽。只需修改一个变量/函数名称，攻击者就可以使有错误/易受攻击的代码排在前11%。我们的攻击BADCODE具有特殊的触发生成和注入过程，使攻击更有效和隐蔽。在两个神经编码搜索模型上进行了评估，结果表明我们的攻击性能比基线高60%。我们的用户研究表明，基于F1比分，我们的攻击比基线更隐蔽两倍。



## **36. Graph Agent Network: Empowering Nodes with Decentralized Communications Capabilities for Adversarial Resilience**

图代理网络：赋予节点分散的通信能力以提高对抗能力 cs.LG

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.06909v1) [paper-pdf](http://arxiv.org/pdf/2306.06909v1)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Hanyuan Huang, Guangquan Xu, Pan Zhou

**Abstract**: End-to-end training with global optimization have popularized graph neural networks (GNNs) for node classification, yet inadvertently introduced vulnerabilities to adversarial edge-perturbing attacks. Adversaries can exploit the inherent opened interfaces of GNNs' input and output, perturbing critical edges and thus manipulating the classification results. Current defenses, due to their persistent utilization of global-optimization-based end-to-end training schemes, inherently encapsulate the vulnerabilities of GNNs. This is specifically evidenced in their inability to defend against targeted secondary attacks. In this paper, we propose the Graph Agent Network (GAgN) to address the aforementioned vulnerabilities of GNNs. GAgN is a graph-structured agent network in which each node is designed as an 1-hop-view agent. Through the decentralized interactions between agents, they can learn to infer global perceptions to perform tasks including inferring embeddings, degrees and neighbor relationships for given nodes. This empowers nodes to filtering adversarial edges while carrying out classification tasks. Furthermore, agents' limited view prevents malicious messages from propagating globally in GAgN, thereby resisting global-optimization-based secondary attacks. We prove that single-hidden-layer multilayer perceptrons (MLPs) are theoretically sufficient to achieve these functionalities. Experimental results show that GAgN effectively implements all its intended capabilities and, compared to state-of-the-art defenses, achieves optimal classification accuracy on the perturbed datasets.

摘要: 具有全局优化的端到端训练普及了图神经网络(GNN)用于节点分类，但无意中引入了对敌意边缘扰动攻击的脆弱性。攻击者可以利用GNN输入和输出固有的开放接口，扰乱关键边缘，从而操纵分类结果。目前的防御措施由于持续使用基于全局优化的端到端培训方案，固有地封装了GNN的脆弱性。这一点具体表现在他们无法防御有针对性的二次攻击。在本文中，我们提出了图代理网络(GagN)来解决GNN的上述漏洞。GAGN是一个图结构的代理网络，其中每个节点被设计为一个1跳视图代理。通过代理之间的分散交互，它们可以学习推断全局感知来执行任务，包括推断给定节点的嵌入度、度数和邻居关系。这使节点能够在执行分类任务时过滤敌意边缘。此外，代理的有限视点防止恶意消息在GAGN中全局传播，从而抵抗基于全局优化的二次攻击。我们证明了单隐层多层感知器(MLP)理论上足以实现这些功能。实验结果表明，GAGN有效地实现了其预期的所有功能，并且与现有的防御措施相比，在扰动数据集上获得了最优的分类精度。



## **37. GAN-CAN: A Novel Attack to Behavior-Based Driver Authentication Systems**

GAN-CAN：一种针对基于行为的驾驶员身份认证系统的新型攻击 cs.CR

16 pages, 6 figures

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.05923v2) [paper-pdf](http://arxiv.org/pdf/2306.05923v2)

**Authors**: Emad Efatinasab, Francesco Marchiori, Denis Donadel, Alessandro Brighente, Mauro Conti

**Abstract**: For many years, car keys have been the sole mean of authentication in vehicles. Whether the access control process is physical or wireless, entrusting the ownership of a vehicle to a single token is prone to stealing attempts. For this reason, many researchers started developing behavior-based authentication systems. By collecting data in a moving vehicle, Deep Learning (DL) models can recognize patterns in the data and identify drivers based on their driving behavior. This can be used as an anti-theft system, as a thief would exhibit a different driving style compared to the vehicle owner's. However, the assumption that an attacker cannot replicate the legitimate driver behavior falls under certain conditions.   In this paper, we propose GAN-CAN, the first attack capable of fooling state-of-the-art behavior-based driver authentication systems in a vehicle. Based on the adversary's knowledge, we propose different GAN-CAN implementations. Our attack leverages the lack of security in the Controller Area Network (CAN) to inject suitably designed time-series data to mimic the legitimate driver. Our design of the malicious time series results from the combination of different Generative Adversarial Networks (GANs) and our study on the safety importance of the injected values during the attack. We tested GAN-CAN in an improved version of the most efficient driver behavior-based authentication model in the literature. We prove that our attack can fool it with an attack success rate of up to 0.99. We show how an attacker, without prior knowledge of the authentication system, can steal a car by deploying GAN-CAN in an off-the-shelf system in under 22 minutes.

摘要: 多年来，汽车钥匙一直是车辆身份验证的唯一手段。无论访问控制过程是物理的还是无线的，将车辆的所有权委托给单个令牌都容易发生窃取尝试。为此，许多研究人员开始开发基于行为的身份认证系统。通过收集移动车辆的数据，深度学习(DL)模型可以识别数据中的模式，并根据司机的驾驶行为识别司机。这可以用作防盗系统，因为与车主相比，小偷会表现出不同的驾驶风格。然而，在某些情况下，攻击者无法复制合法司机行为的假设是成立的。在本文中，我们提出了GAN-CAN，这是第一个能够欺骗车辆中最先进的基于行为的驾驶员身份验证系统的攻击。基于对手的知识，我们提出了不同的GAN-CAN实现方案。我们的攻击利用控制器区域网络(CAN)缺乏安全性来注入适当设计的时间序列数据，以模仿合法的驱动程序。我们的恶意时间序列的设计是不同生成性对抗网络(GANS)组合的结果，也是我们对攻击期间注入的值的安全重要性的研究的结果。我们在文献中最有效的基于司机行为的身份验证模型的改进版本中测试了GAN-CAN。我们证明了我们的攻击可以欺骗它，攻击成功率高达0.99。我们展示了攻击者如何在不了解身份验证系统的情况下，在不到22分钟的时间内通过在现成系统中部署GAN-CAN来窃取一辆汽车。



## **38. Trustworthy Artificial Intelligence Framework for Proactive Detection and Risk Explanation of Cyber Attacks in Smart Grid**

智能电网网络攻击主动检测与风险解释的可信人工智能框架 cs.CR

Submitted for peer review

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.07993v1) [paper-pdf](http://arxiv.org/pdf/2306.07993v1)

**Authors**: Md. Shirajum Munir, Sachin Shetty, Danda B. Rawat

**Abstract**: The rapid growth of distributed energy resources (DERs), such as renewable energy sources, generators, consumers, and prosumers in the smart grid infrastructure, poses significant cybersecurity and trust challenges to the grid controller. Consequently, it is crucial to identify adversarial tactics and measure the strength of the attacker's DER. To enable a trustworthy smart grid controller, this work investigates a trustworthy artificial intelligence (AI) mechanism for proactive identification and explanation of the cyber risk caused by the control/status message of DERs. Thus, proposing and developing a trustworthy AI framework to facilitate the deployment of any AI algorithms for detecting potential cyber threats and analyzing root causes based on Shapley value interpretation while dynamically quantifying the risk of an attack based on Ward's minimum variance formula. The experiment with a state-of-the-art dataset establishes the proposed framework as a trustworthy AI by fulfilling the capabilities of reliability, fairness, explainability, transparency, reproducibility, and accountability.

摘要: 分布式能源(DER)的快速增长，如智能电网基础设施中的可再生能源、发电机、消费者和消费者，给电网控制器带来了巨大的网络安全和信任挑战。因此，确定敌方战术并衡量攻击者的实力至关重要。为了实现可信赖的智能电网控制器，本工作研究了一种可信赖的人工智能(AI)机制，用于主动识别和解释由DER的控制/状态消息引起的网络风险。因此，提出并开发了一个可信的人工智能框架，以促进任何人工智能算法的部署，以检测潜在的网络威胁并基于Shapley值解释分析根本原因，同时根据Ward的最小方差公式动态量化攻击风险。使用最先进的数据集进行的实验通过实现可靠性、公平性、可解释性、透明度、重复性和问责制的能力，将所提出的框架建立为值得信赖的人工智能。



## **39. Asymptotically Optimal Adversarial Strategies for the Probability Estimation Framework**

概率估计框架的渐近最优对抗策略 quant-ph

54 pages

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2306.06802v1) [paper-pdf](http://arxiv.org/pdf/2306.06802v1)

**Authors**: Soumyadip Patra, Peter Bierhorst

**Abstract**: The Probability Estimation Framework involves direct estimation of the probability of occurrences of outcomes conditioned on measurement settings and side information. It is a powerful tool for certifying randomness in quantum non-locality experiments. In this paper, we present a self-contained proof of the asymptotic optimality of the method. Our approach refines earlier results to allow a better characterisation of optimal adversarial attacks on the protocol. We apply these results to the (2,2,2) Bell scenario, obtaining an analytic characterisation of the optimal adversarial attacks bound by no-signalling principles, while also demonstrating the asymptotic robustness of the PEF method to deviations from expected experimental behaviour. We also study extensions of the analysis to quantum-limited adversaries in the (2,2,2) Bell scenario and no-signalling adversaries in higher $(n,m,k)$ Bell scenarios.

摘要: 概率估计框架涉及根据测量设置和辅助信息直接估计结果发生的概率。它是验证量子非定域性实验中随机性的有力工具。本文给出了该方法渐近最优性的一个完备证明。我们的方法改进了早期的结果，以便更好地描述对协议的最优敌意攻击。我们将这些结果应用于(2，2，2)Bell情形，得到了受无信令原理约束的最优对抗攻击的解析刻画，同时也证明了PEF方法对偏离预期实验行为的渐近鲁棒性。我们还研究了在(2，2，2)Bell情形下的量子受限对手和在更高的$(n，m，k)$Bell情形下的无信号对手的分析的扩展。



## **40. Adversarial Reconnaissance Mitigation and Modeling**

对抗性侦察消解与建模 cs.CR

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2306.06769v1) [paper-pdf](http://arxiv.org/pdf/2306.06769v1)

**Authors**: Shanto Roy, Nazia Sharmin, Mohammad Sujan Miah, Jaime C Acosta, Christopher Kiekintveld, Aron Laszka

**Abstract**: Adversarial reconnaissance is a crucial step in sophisticated cyber-attacks as it enables threat actors to find the weakest points of otherwise well-defended systems. To thwart reconnaissance, defenders can employ cyber deception techniques, such as deploying honeypots. In recent years, researchers have made great strides in developing game-theoretic models to find optimal deception strategies. However, most of these game-theoretic models build on relatively simple models of adversarial reconnaissance -- even though reconnaissance should be a focus point as the very purpose of deception is to thwart reconnaissance. In this paper, we first discuss effective cyber reconnaissance mitigation techniques including deception strategies and beyond. Then we provide a review of the literature on deception games from the perspective of modeling adversarial reconnaissance, highlighting key aspects of reconnaissance that have not been adequately captured in prior work. We then describe a probability-theory based model of the adversaries' belief formation and illustrate using numerical examples that this model can capture key aspects of adversarial reconnaissance. We believe that our review and belief model can serve as a stepping stone for developing more realistic and practical deception games.

摘要: 对抗性侦察是复杂网络攻击的关键一步，因为它使威胁参与者能够找到原本防御良好的系统的最薄弱环节。为了阻止侦察，防御者可以使用网络欺骗技术，例如部署蜜罐。近年来，研究人员在开发博弈论模型以寻找最优欺骗策略方面取得了长足的进步。然而，这些博弈论模型大多建立在相对简单的对抗性侦察模型之上--尽管侦察应该是重点，因为欺骗的目的就是挫败侦察。在本文中，我们首先讨论了有效的网络侦察缓解技术，包括欺骗策略等。然后，我们从建模对抗侦察的角度对欺骗游戏的文献进行了回顾，强调了侦察的关键方面，这些方面在以前的工作中没有被充分捕获。然后，我们描述了一个基于概率论的敌方信念形成模型，并用数值例子说明了该模型能够捕捉敌方侦察的关键方面。我们相信，我们的复习和信念模型可以作为开发更现实和实用的欺骗游戏的垫脚石。



## **41. Securing Visually-Aware Recommender Systems: An Adversarial Image Reconstruction and Detection Framework**

保护视觉感知推荐系统：一种对抗性图像重建与检测框架 cs.CV

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2306.07992v1) [paper-pdf](http://arxiv.org/pdf/2306.07992v1)

**Authors**: Minglei Yin, Bin Liu, Neil Zhenqiang Gong, Xin Li

**Abstract**: With rich visual data, such as images, becoming readily associated with items, visually-aware recommendation systems (VARS) have been widely used in different applications. Recent studies have shown that VARS are vulnerable to item-image adversarial attacks, which add human-imperceptible perturbations to the clean images associated with those items. Attacks on VARS pose new security challenges to a wide range of applications such as e-Commerce and social networks where VARS are widely used. How to secure VARS from such adversarial attacks becomes a critical problem. Currently, there is still a lack of systematic study on how to design secure defense strategies against visual attacks on VARS. In this paper, we attempt to fill this gap by proposing an adversarial image reconstruction and detection framework to secure VARS. Our proposed method can simultaneously (1) secure VARS from adversarial attacks characterized by local perturbations by image reconstruction based on global vision transformers; and (2) accurately detect adversarial examples using a novel contrastive learning approach. Meanwhile, our framework is designed to be used as both a filter and a detector so that they can be jointly trained to improve the flexibility of our defense strategy to a variety of attacks and VARS models. We have conducted extensive experimental studies with two popular attack methods (FGSM and PGD). Our experimental results on two real-world datasets show that our defense strategy against visual attacks is effective and outperforms existing methods on different attacks. Moreover, our method can detect adversarial examples with high accuracy.

摘要: 随着图像等丰富的视觉数据变得容易与物品相关联，视觉感知推荐系统(VAR)已被广泛应用于不同的应用中。最近的研究表明，VAR容易受到物品图像对抗性攻击，这些攻击会给与这些物品相关的干净图像添加人类无法察觉的扰动。对var的攻击给广泛使用var的电子商务和社交网络等广泛的应用程序带来了新的安全挑战。如何保护VAR免受这种敌意攻击成为一个关键问题。目前，针对VaR的视觉攻击如何设计安全的防御策略还缺乏系统的研究。在本文中，我们试图通过提出一种对抗性图像重建和检测框架来保护VARS来填补这一空白。我们提出的方法可以同时(1)通过基于全局视觉变换的图像重建来保护VAR免受以局部扰动为特征的对抗性攻击；(2)使用一种新的对比学习方法来准确地检测对抗性示例。同时，我们的框架被设计成同时用作过滤器和检测器，以便它们可以被联合训练，以提高我们的防御策略对各种攻击和VARS模型的灵活性。我们对两种流行的攻击方法(FGSM和PGD)进行了广泛的实验研究。我们在两个真实数据集上的实验结果表明，我们的防御策略对视觉攻击是有效的，并且在不同攻击下的性能优于现有的方法。此外，我们的方法能够以较高的准确率检测出对抗性实例。



## **42. Neural Architecture Design and Robustness: A Dataset**

神经结构设计与稳健性：一个数据集 cs.LG

ICLR 2023; project page: http://robustness.vision/

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2306.06712v1) [paper-pdf](http://arxiv.org/pdf/2306.06712v1)

**Authors**: Steffen Jung, Jovita Lukasik, Margret Keuper

**Abstract**: Deep learning models have proven to be successful in a wide range of machine learning tasks. Yet, they are often highly sensitive to perturbations on the input data which can lead to incorrect decisions with high confidence, hampering their deployment for practical use-cases. Thus, finding architectures that are (more) robust against perturbations has received much attention in recent years. Just like the search for well-performing architectures in terms of clean accuracy, this usually involves a tedious trial-and-error process with one additional challenge: the evaluation of a network's robustness is significantly more expensive than its evaluation for clean accuracy. Thus, the aim of this paper is to facilitate better streamlined research on architectural design choices with respect to their impact on robustness as well as, for example, the evaluation of surrogate measures for robustness. We therefore borrow one of the most commonly considered search spaces for neural architecture search for image classification, NAS-Bench-201, which contains a manageable size of 6466 non-isomorphic network designs. We evaluate all these networks on a range of common adversarial attacks and corruption types and introduce a database on neural architecture design and robustness evaluations. We further present three exemplary use cases of this dataset, in which we (i) benchmark robustness measurements based on Jacobian and Hessian matrices for their robustness predictability, (ii) perform neural architecture search on robust accuracies, and (iii) provide an initial analysis of how architectural design choices affect robustness. We find that carefully crafting the topology of a network can have substantial impact on its robustness, where networks with the same parameter count range in mean adversarial robust accuracy from 20%-41%. Code and data is available at http://robustness.vision/.

摘要: 深度学习模型已被证明在广泛的机器学习任务中是成功的。然而，它们往往对输入数据的扰动高度敏感，这可能会导致高置信度的错误决策，阻碍它们在实际用例中的部署。因此，寻找对扰动(更)健壮的体系结构在最近几年受到了极大的关注。就像在干净准确性方面寻找表现良好的体系结构一样，这通常涉及一个乏味的反复试验过程，还有一个额外的挑战：评估网络的稳健性比评估其干净准确性的成本要高得多。因此，本文的目的是促进关于建筑设计选择对稳健性的影响的更好的简化研究，以及例如，对稳健性替代措施的评估。因此，我们借用了用于图像分类的神经结构搜索最常用的搜索空间之一NAS-BENCH-201，它包含6466个非同构网络设计的可管理大小。我们对所有这些网络在一系列常见的对抗性攻击和破坏类型上进行了评估，并引入了一个关于神经体系结构设计和健壮性评估的数据库。我们进一步给出了这个数据集的三个典型用例，其中我们(I)基于雅可比矩阵和海森矩阵对健壮性度量进行基准测试，以确定其健壮性可预测性，(Ii)对健壮性精度执行神经体系结构搜索，以及(Iii)提供体系结构设计选择如何影响健壮性的初步分析。我们发现，精心设计网络的拓扑结构可以对其健壮性产生重大影响，其中具有相同参数的网络的平均对抗健壮性准确率在20%-41%之间。代码和数据可在http://robustness.vision/.上获得



## **43. EvadeDroid: A Practical Evasion Attack on Machine Learning for Black-box Android Malware Detection**

EvadeDroid：一种实用的机器学习黑盒Android恶意软件检测规避攻击 cs.LG

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2110.03301v3) [paper-pdf](http://arxiv.org/pdf/2110.03301v3)

**Authors**: Hamid Bostani, Veelasha Moonsamy

**Abstract**: Over the last decade, researchers have extensively explored the vulnerabilities of Android malware detectors to adversarial examples through the development of evasion attacks; however, the practicality of these attacks in real-world scenarios remains arguable. The majority of studies have assumed attackers know the details of the target classifiers used for malware detection, while in reality, malicious actors have limited access to the target classifiers. This paper introduces EvadeDroid, a practical decision-based adversarial attack designed to effectively evade black-box Android malware detectors in real-world scenarios. In addition to generating real-world adversarial malware, the proposed evasion attack can also preserve the functionality of the original malware applications (apps). EvadeDroid constructs a collection of functionality-preserving transformations derived from benign donors that share opcode-level similarity with malware apps by leveraging an n-gram-based approach. These transformations are then used to morph malware instances into benign ones via an iterative and incremental manipulation strategy. The proposed manipulation technique is a novel, query-efficient optimization algorithm that can find and inject optimal sequences of transformations into malware apps. Our empirical evaluation demonstrates the efficacy of EvadeDroid under soft- and hard-label attacks. Furthermore, EvadeDroid exhibits the capability to generate real-world adversarial examples that can effectively evade a wide range of black-box ML-based malware detectors with minimal query requirements. Finally, we show that the proposed problem-space adversarial attack is able to preserve its stealthiness against five popular commercial antiviruses, thus demonstrating its feasibility in the real world.

摘要: 在过去的十年里，研究人员通过规避攻击的开发，广泛探索了Android恶意软件检测器对敌意例子的漏洞；然而，这些攻击在现实世界场景中的实用性仍然存在争议。大多数研究都假设攻击者知道用于恶意软件检测的目标分类器的详细信息，而实际上，恶意行为者对目标分类器的访问权限有限。本文介绍了EvadeDroid，一种实用的基于决策的对抗性攻击，旨在有效地躲避现实场景中的黑盒Android恶意软件检测。除了生成真实世界的敌意恶意软件外，拟议的逃避攻击还可以保留原始恶意软件应用程序(APP)的功能。EvadeDroid构建了一组保留功能的转换，这些转换来自良性捐赠者，通过利用基于n-gram的方法与恶意软件应用程序共享操作码级相似性。然后，这些转换被用于通过迭代和增量操作策略将恶意软件实例变形为良性实例。提出的操纵技术是一种新颖的、查询高效的优化算法，可以找到最优的转换序列并将其注入恶意软件应用程序。我们的经验评估证明了EvadeDroid在软标签和硬标签攻击下的有效性。此外，EvadeDroid展示了生成真实世界敌意示例的能力，这些示例可以有效地躲避各种基于黑盒ML的恶意软件检测器，而查询要求最低。最后，我们证明了所提出的问题空间对抗攻击能够对五种流行的商业反病毒保持其隐蔽性，从而证明了其在现实世界中的可行性。



## **44. Level Up with RealAEs: Leveraging Domain Constraints in Feature Space to Strengthen Robustness of Android Malware Detection**

与RealEs并驾齐驱：利用特征空间中的域约束增强Android恶意软件检测的健壮性 cs.LG

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2205.15128v3) [paper-pdf](http://arxiv.org/pdf/2205.15128v3)

**Authors**: Hamid Bostani, Zhengyu Zhao, Zhuoran Liu, Veelasha Moonsamy

**Abstract**: The vulnerability to adversarial examples remains one major obstacle for Machine Learning (ML)-based Android malware detection. Realistic attacks in the Android malware domain create Realizable Adversarial Examples (RealAEs), i.e., AEs that satisfy the domain constraints of Android malware. Recent studies have shown that using such RealAEs in Adversarial Training (AT) is more effective in defending against realistic attacks than using unrealizable AEs (unRealAEs). This is because RealAEs allow defenders to explore certain pockets in the feature space that are vulnerable to realistic attacks. However, existing defenses commonly generate RealAEs in the problem space, which is known to be time-consuming and impractical for AT. In this paper, we propose to generate RealAEs in the feature space, leading to a simpler and more efficient solution. Our approach is driven by a novel interpretation of Android domain constraints in the feature space. More concretely, our defense first learns feature-space domain constraints by extracting meaningful feature dependencies from data and then applies them to generating feature-space RealAEs during AT. Extensive experiments on DREBIN, a well-known Android malware detector, demonstrate that our new defense outperforms not only unRealAE-based AT but also the state-of-the-art defense that relies on non-uniform perturbations. We further validate the ability of our learned feature-space domain constraints in representing Android malware properties by showing that our feature-space domain constraints can help distinguish RealAEs from unRealAEs.

摘要: 恶意示例的漏洞仍然是基于机器学习(ML)的Android恶意软件检测的主要障碍。Android恶意软件领域中的现实攻击创建了可实现的对抗性实例(RealAE)，即满足Android恶意软件的域约束的AE。最近的研究表明，在对抗训练(AT)中使用这种真实AEs比使用不可实现AEs(UnRealAEs)更有效地防御现实攻击。这是因为RealEs允许防御者探索特征空间中易受现实攻击的某些口袋。然而，现有的防御通常会在问题空间中生成RealEs，这对于AT来说是耗时和不切实际的。在本文中，我们建议在特征空间中生成RealAE，从而得到一个更简单、更有效的解决方案。我们的方法是由特征空间中对Android领域约束的一种新解释驱动的。更具体地说，我们的防御首先通过从数据中提取有意义的特征依赖来学习特征空间域约束，然后将它们应用于在AT过程中生成特征空间RealAE。在著名的Android恶意软件检测器Drebin上的广泛实验表明，我们的新防御不仅优于基于UnRealAE的AT，而且优于依赖非均匀扰动的最先进防御。我们进一步验证了我们学习的特征空间域约束在表示Android恶意软件属性方面的能力，方法是展示我们的特征空间域约束可以帮助区分真正的恶意软件和非真实的恶意软件。



## **45. Attacking Cooperative Multi-Agent Reinforcement Learning by Adversarial Minority Influence**

对抗性少数影响攻击协作式多智能体强化学习 cs.LG

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2302.03322v2) [paper-pdf](http://arxiv.org/pdf/2302.03322v2)

**Authors**: Simin Li, Jun Guo, Jingqiao Xiu, Pu Feng, Xin Yu, Aishan Liu, Wenjun Wu, Xianglong Liu

**Abstract**: This study probes the vulnerabilities of cooperative multi-agent reinforcement learning (c-MARL) under adversarial attacks, a critical determinant of c-MARL's worst-case performance prior to real-world implementation. Current observation-based attacks, constrained by white-box assumptions, overlook c-MARL's complex multi-agent interactions and cooperative objectives, resulting in impractical and limited attack capabilities. To address these shortcomes, we propose Adversarial Minority Influence (AMI), a practical and strong for c-MARL. AMI is a practical black-box attack and can be launched without knowing victim parameters. AMI is also strong by considering the complex multi-agent interaction and the cooperative goal of agents, enabling a single adversarial agent to unilaterally misleads majority victims to form targeted worst-case cooperation. This mirrors minority influence phenomena in social psychology. To achieve maximum deviation in victim policies under complex agent-wise interactions, our unilateral attack aims to characterize and maximize the impact of the adversary on the victims. This is achieved by adapting a unilateral agent-wise relation metric derived from mutual information, thereby mitigating the adverse effects of victim influence on the adversary. To lead the victims into a jointly detrimental scenario, our targeted attack deceives victims into a long-term, cooperatively harmful situation by guiding each victim towards a specific target, determined through a trial-and-error process executed by a reinforcement learning agent. Through AMI, we achieve the first successful attack against real-world robot swarms and effectively fool agents in simulated environments into collectively worst-case scenarios, including Starcraft II and Multi-agent Mujoco. The source code and demonstrations can be found at: https://github.com/DIG-Beihang/AMI.

摘要: 该研究探讨了协作多智能体强化学习(c-Marl)在对抗攻击下的脆弱性，这是c-Marl在现实世界实现之前最差情况性能的关键决定因素。目前的基于观测的攻击受白盒假设的约束，忽略了c-Marl复杂的多智能体交互和合作目标，导致攻击能力不切实际和有限。针对这些不足，我们提出了一种实用而强大的c-Marl算法--对抗性少数影响算法。AMI是一种实用的黑盒攻击，可以在不知道受害者参数的情况下启动。通过考虑复杂的多智能体相互作用和智能体的合作目标，使单一对抗智能体能够单方面误导大多数受害者形成有针对性的最坏情况合作，AMI也很强大。这反映了社会心理学中的小众影响现象。为了在复杂的智能体相互作用下实现受害者政策的最大偏差，我们的单边攻击旨在刻画和最大化对手对受害者的影响。这是通过采用来自互信息的单边代理关系度量来实现的，从而减轻了受害者影响对对手的不利影响。为了将受害者引导到共同有害的情景中，我们的有针对性的攻击通过引导每个受害者指向特定的目标，将受害者欺骗到长期的、合作有害的情况中，该特定目标是通过由强化学习代理执行的反复试验过程确定的。通过AMI，我们实现了对真实世界机器人群的第一次成功攻击，并有效地将模拟环境中的代理愚弄到了集体最坏的情况下，包括星际争霸II和多代理Mujoco。源代码和演示可在以下网址找到：https://github.com/DIG-Beihang/AMI.



## **46. Defense Against Adversarial Attacks on Audio DeepFake Detection**

音频DeepFake检测中的敌意攻击防御 cs.SD

Accepted to INTERSPEECH 2023

**SubmitDate**: 2023-06-10    [abs](http://arxiv.org/abs/2212.14597v2) [paper-pdf](http://arxiv.org/pdf/2212.14597v2)

**Authors**: Piotr Kawa, Marcin Plata, Piotr Syga

**Abstract**: Audio DeepFakes (DF) are artificially generated utterances created using deep learning, with the primary aim of fooling the listeners in a highly convincing manner. Their quality is sufficient to pose a severe threat in terms of security and privacy, including the reliability of news or defamation. Multiple neural network-based methods to detect generated speech have been proposed to prevent the threats. In this work, we cover the topic of adversarial attacks, which decrease the performance of detectors by adding superficial (difficult to spot by a human) changes to input data. Our contribution contains evaluating the robustness of 3 detection architectures against adversarial attacks in two scenarios (white-box and using transferability) and enhancing it later by using adversarial training performed by our novel adaptive training. Moreover, one of the investigated architectures is RawNet3, which, to the best of our knowledge, we adapted for the first time to DeepFake detection.

摘要: Audio DeepFake(DF)是使用深度学习创建的人工生成的话语，其主要目的是以高度令人信服的方式愚弄听众。它们的质量足以在安全和隐私方面构成严重威胁，包括新闻或诽谤的可靠性。为了防止这种威胁，已经提出了多种基于神经网络的生成语音检测方法。在这项工作中，我们讨论了对抗性攻击的主题，它通过向输入数据添加表面(难以被人发现)的更改来降低检测器的性能。我们的贡献包括评估三种检测体系结构在两种场景(白盒和使用可转移性)下对敌意攻击的健壮性，并在以后通过使用我们新的自适应训练执行的对抗性训练来增强它。此外，其中一个被研究的架构是RawNet3，据我们所知，我们第一次将其应用于DeepFake检测。



## **47. The Defense of Networked Targets in General Lotto games**

普通彩票游戏中网络目标的防御 cs.GT

**SubmitDate**: 2023-06-10    [abs](http://arxiv.org/abs/2306.06485v1) [paper-pdf](http://arxiv.org/pdf/2306.06485v1)

**Authors**: Adel Aghajan, Keith Paarporn, Jason R. Marden

**Abstract**: Ensuring the security of networked systems is a significant problem, considering the susceptibility of modern infrastructures and technologies to adversarial interference. A central component of this problem is how defensive resources should be allocated to mitigate the severity of potential attacks on the system. In this paper, we consider this in the context of a General Lotto game, where a defender and attacker deploys resources on the nodes of a network, and the objective is to secure as many links as possible. The defender secures a link only if it out-competes the attacker on both of its associated nodes. For bipartite networks, we completely characterize equilibrium payoffs and strategies for both the defender and attacker. Surprisingly, the resulting payoffs are the same for any bipartite graph. On arbitrary network structures, we provide lower and upper bounds on the defender's max-min value. Notably, the equilibrium payoff from bipartite networks serves as the lower bound. These results suggest that more connected networks are easier to defend against attacks. We confirm these findings with simulations that compute deterministic allocation strategies on large random networks. This also highlights the importance of randomization in the equilibrium strategies.

摘要: 考虑到现代基础设施和技术对敌方干扰的敏感性，确保联网系统的安全是一个重大问题。这个问题的一个核心部分是应该如何分配防御资源，以减轻对系统的潜在攻击的严重性。在本文中，我们在一般彩票博弈的背景下考虑这一问题，其中防御者和攻击者在网络的节点上部署资源，目标是确保尽可能多的链路。只有当防御者在两个相关节点上都胜过攻击者时，防御者才能确保链路的安全。对于二部网络，我们完全刻画了防御者和攻击者的均衡收益和策略。令人惊讶的是，由此产生的收益对于任何二部图都是相同的。在任意网络结构下，我们给出了防御者的最大最小值的上下界。值得注意的是，二部网络的均衡收益是下限。这些结果表明，连接越紧密的网络越容易抵御攻击。我们通过在大型随机网络上计算确定性分配策略的模拟来证实这些发现。这也突显了随机化在均衡战略中的重要性。



## **48. NeRFool: Uncovering the Vulnerability of Generalizable Neural Radiance Fields against Adversarial Perturbations**

NeRFool：发现可推广的神经辐射场对抗敌方扰动的脆弱性 cs.CV

Accepted by ICML 2023

**SubmitDate**: 2023-06-10    [abs](http://arxiv.org/abs/2306.06359v1) [paper-pdf](http://arxiv.org/pdf/2306.06359v1)

**Authors**: Yonggan Fu, Ye Yuan, Souvik Kundu, Shang Wu, Shunyao Zhang, Yingyan Lin

**Abstract**: Generalizable Neural Radiance Fields (GNeRF) are one of the most promising real-world solutions for novel view synthesis, thanks to their cross-scene generalization capability and thus the possibility of instant rendering on new scenes. While adversarial robustness is essential for real-world applications, little study has been devoted to understanding its implication on GNeRF. We hypothesize that because GNeRF is implemented by conditioning on the source views from new scenes, which are often acquired from the Internet or third-party providers, there are potential new security concerns regarding its real-world applications. Meanwhile, existing understanding and solutions for neural networks' adversarial robustness may not be applicable to GNeRF, due to its 3D nature and uniquely diverse operations. To this end, we present NeRFool, which to the best of our knowledge is the first work that sets out to understand the adversarial robustness of GNeRF. Specifically, NeRFool unveils the vulnerability patterns and important insights regarding GNeRF's adversarial robustness. Built upon the above insights gained from NeRFool, we further develop NeRFool+, which integrates two techniques capable of effectively attacking GNeRF across a wide range of target views, and provide guidelines for defending against our proposed attacks. We believe that our NeRFool/NeRFool+ lays the initial foundation for future innovations in developing robust real-world GNeRF solutions. Our codes are available at: https://github.com/GATECH-EIC/NeRFool.

摘要: 可概括神经辐射场(GNeRF)是现实世界中最有前途的新型视点合成解决方案之一，这要归功于它们的跨场景泛化能力，从而可以在新场景上进行即时渲染。虽然对抗的稳健性对于现实世界的应用是必不可少的，但很少有研究致力于了解其对GNeRF的影响。我们假设，由于GNeRF是通过对来自新场景的源视图进行条件处理来实现的，这些场景通常是从互联网或第三方提供商获得的，因此其现实世界的应用程序存在潜在的新的安全问题。同时，由于GNeRF的3D性质和独特的多样性操作，现有对神经网络对抗性稳健性的理解和解决方案可能不适用于GNeRF。为此，我们提出了NeRFool，据我们所知，这是第一个开始了解GNeRF的对手健壮性的工作。具体地说，NeRFool揭示了关于GNeRF的对手健壮性的漏洞模式和重要见解。基于以上从NeRFool获得的见解，我们进一步开发了NeRFool+，它集成了两种能够在广泛的目标视图中有效攻击GNeRF的技术，并为防御我们提出的攻击提供了指导方针。我们相信，我们的NeRFool/NeRFool+为未来在开发强大的现实世界GNeRF解决方案方面的创新奠定了初步基础。我们的代码请访问：https://github.com/GATECH-EIC/NeRFool.



## **49. Differentially private sliced inverse regression in the federated paradigm**

联邦范例中的差分私有切片逆回归 stat.ME

**SubmitDate**: 2023-06-10    [abs](http://arxiv.org/abs/2306.06324v1) [paper-pdf](http://arxiv.org/pdf/2306.06324v1)

**Authors**: Shuaida He, Jiarui Zhang, Xin Chen

**Abstract**: We extend the celebrated sliced inverse regression to address the challenges of decentralized data, prioritizing privacy and communication efficiency. Our approach, federated sliced inverse regression (FSIR), facilitates collaborative estimation of the sufficient dimension reduction subspace among multiple clients, solely sharing local estimates to protect sensitive datasets from exposure. To guard against potential adversary attacks, FSIR further employs diverse perturbation strategies, including a novel multivariate Gaussian mechanism that guarantees differential privacy at a low cost of statistical accuracy. Additionally, FSIR naturally incorporates a collaborative variable screening step, enabling effective handling of high-dimensional client data. Theoretical properties of FSIR are established for both low-dimensional and high-dimensional settings, supported by extensive numerical experiments and real data analysis.

摘要: 我们扩展了著名的切片反向回归，以应对分散数据、优先考虑隐私和通信效率的挑战。我们的方法，联合切片逆回归(FSIR)，促进了多个客户之间对足够降维空间的协作估计，只共享局部估计来保护敏感数据集免受暴露。为了防止潜在的对手攻击，FSIR进一步采用了多种扰动策略，包括一种新颖的多变量高斯机制，该机制以较低的统计准确性为代价保证了差分隐私。此外，FSIR自然包含协作可变筛选步骤，从而能够有效处理高维客户数据。在大量的数值实验和实际数据分析的支持下，建立了低维和高维环境下FSIR的理论性质。



## **50. The Certification Paradox: Certifications Admit Better Attacks**

认证悖论：认证允许更好的攻击 cs.LG

16 pages, 6 figures

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2302.04379v2) [paper-pdf](http://arxiv.org/pdf/2302.04379v2)

**Authors**: Andrew C. Cullen, Shijie Liu, Paul Montague, Sarah M. Erfani, Benjamin I. P. Rubinstein

**Abstract**: In guaranteeing that no adversarial examples exist within a bounded region, certification mechanisms play an important role in demonstrating the robustness of neural networks. In this work we ask: Could certifications have any unintended consequences, through exposing additional information about certified models? We answer this question in the affirmative, demonstrating that certifications not only measure model robustness but also present a new attack surface. We propose \emph{Certification Aware Attacks}, that produce smaller adversarial perturbations more than twice as frequently as any prior approach, when launched against certified models. Our attacks achieve an up to $34\%$ reduction in the median perturbation norm (comparing target and attack instances), while requiring $90 \%$ less computational time than approaches like PGD. That our attacks achieve such significant reductions in perturbation size and computational cost highlights an apparent paradox in deploying certification mechanisms. We end the paper with a discussion of how these risks could potentially be mitigated.

摘要: 在保证有界区域内不存在对抗性实例方面，认证机制在证明神经网络的健壮性方面发挥了重要作用。在这项工作中，我们问：通过暴露有关认证模型的更多信息，认证是否会产生任何意想不到的后果？我们对这个问题的回答是肯定的，证明了认证不仅衡量了模型的稳健性，而且还提供了一个新的攻击面。我们提出了认证感知攻击，当针对认证模型发起攻击时，产生较小的对抗性扰动的频率是之前任何方法的两倍以上。我们的攻击使中值扰动范数(比较目标和攻击实例)减少了高达34美元，而与PGD等方法相比，所需计算时间减少了90美元。我们的攻击在扰动大小和计算成本方面实现了如此显著的减少，突显了部署认证机制的一个明显的悖论。我们在文章的最后讨论了如何潜在地减轻这些风险。



