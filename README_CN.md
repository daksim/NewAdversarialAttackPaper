# Latest Adversarial Attack Papers
**update at 2022-04-21 06:31:35**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. A Brief Survey on Deep Learning Based Data Hiding**

基于深度学习的数据隐藏研究综述 cs.CR

v2: reorganize some sections and add several new papers published in  2021~2022

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2103.01607v2)

**Authors**: Chaoning Zhang, Chenguo Lin, Philipp Benz, Kejiang Chen, Weiming Zhang, In So Kweon

**Abstracts**: Data hiding is the art of concealing messages with limited perceptual changes. Recently, deep learning has enriched it from various perspectives with significant progress. In this work, we conduct a brief yet comprehensive review of existing literature for deep learning based data hiding (deep hiding) by first classifying it according to three essential properties (i.e., capacity, security and robustness), and outline three commonly used architectures. Based on this, we summarize specific strategies for different applications of data hiding, including basic hiding, steganography, watermarking and light field messaging. Finally, further insight into deep hiding is provided by incorporating the perspective of adversarial attack.

摘要: 数据隐藏是通过有限的感知变化来隐藏消息的艺术。最近，深度学习从多个角度丰富了它，取得了重大进展。在这项工作中，我们对现有的基于深度学习的数据隐藏(深度隐藏)进行了简要而全面的回顾，首先根据三个基本属性(即容量、安全性和健壮性)对其进行分类，并概述了三种常用的体系结构。在此基础上，总结了针对不同应用的数据隐藏的具体策略，包括基本隐藏、隐写、水印和光场消息。最后，通过结合对抗性攻击的视角，对深层隐藏提供了进一步的洞察。



## **2. Jacobian Ensembles Improve Robustness Trade-offs to Adversarial Attacks**

雅可比集合提高了对抗攻击的稳健性权衡 cs.LG

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.08726v1)

**Authors**: Kenneth T. Co, David Martinez-Rego, Zhongyuan Hau, Emil C. Lupu

**Abstracts**: Deep neural networks have become an integral part of our software infrastructure and are being deployed in many widely-used and safety-critical applications. However, their integration into many systems also brings with it the vulnerability to test time attacks in the form of Universal Adversarial Perturbations (UAPs). UAPs are a class of perturbations that when applied to any input causes model misclassification. Although there is an ongoing effort to defend models against these adversarial attacks, it is often difficult to reconcile the trade-offs in model accuracy and robustness to adversarial attacks. Jacobian regularization has been shown to improve the robustness of models against UAPs, whilst model ensembles have been widely adopted to improve both predictive performance and model robustness. In this work, we propose a novel approach, Jacobian Ensembles-a combination of Jacobian regularization and model ensembles to significantly increase the robustness against UAPs whilst maintaining or improving model accuracy. Our results show that Jacobian Ensembles achieves previously unseen levels of accuracy and robustness, greatly improving over previous methods that tend to skew towards only either accuracy or robustness.

摘要: 深度神经网络已成为我们软件基础设施的组成部分，并被部署在许多广泛使用和安全关键的应用程序中。然而，它们与许多系统的集成也带来了测试通用对抗扰动(UAP)形式的时间攻击的脆弱性。UAP是一类扰动，当应用于任何输入时，都会导致模型错误分类。尽管人们一直在努力保护模型免受这些对抗性攻击，但通常很难在模型精确度和对对抗性攻击的稳健性之间进行权衡。雅可比正则化已被证明可以提高模型对UAP的稳健性，而模型集成已被广泛采用来提高预测性能和模型稳健性。在这项工作中，我们提出了一种新的方法，雅可比集成-雅可比正则化和模型集成的组合，在保持或改善模型精度的同时，显著增强了对UAP的鲁棒性。我们的结果表明，雅可比集成达到了前所未有的精度和稳健性水平，大大改进了以前的方法，这些方法倾向于只偏向精度或稳健性。



## **3. Topology and geometry of data manifold in deep learning**

深度学习中数据流形的拓扑和几何 cs.LG

12 pages, 15 figures

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.08624v1)

**Authors**: German Magai, Anton Ayzenberg

**Abstracts**: Despite significant advances in the field of deep learning in applications to various fields, explaining the inner processes of deep learning models remains an important and open question. The purpose of this article is to describe and substantiate the geometric and topological view of the learning process of neural networks. Our attention is focused on the internal representation of neural networks and on the dynamics of changes in the topology and geometry of the data manifold on different layers. We also propose a method for assessing the generalizing ability of neural networks based on topological descriptors. In this paper, we use the concepts of topological data analysis and intrinsic dimension, and we present a wide range of experiments on different datasets and different configurations of convolutional neural network architectures. In addition, we consider the issue of the geometry of adversarial attacks in the classification task and spoofing attacks on face recognition systems. Our work is a contribution to the development of an important area of explainable and interpretable AI through the example of computer vision.

摘要: 尽管深度学习领域在各个领域的应用取得了重大进展，但解释深度学习模型的内部过程仍然是一个重要而开放的问题。本文的目的是描述和充实神经网络学习过程的几何和拓扑观。我们的注意力集中在神经网络的内部表示以及不同层上数据流形的拓扑和几何变化的动力学上。提出了一种基于拓扑描述子的神经网络泛化能力评估方法。在本文中，我们使用了拓扑数据分析和内在维的概念，并在不同的数据集和不同结构的卷积神经网络结构上进行了广泛的实验。此外，我们还考虑了分类任务中敌意攻击的几何问题和对人脸识别系统的欺骗攻击。我们的工作是对通过计算机视觉的例子来发展可解释和可解释的人工智能的一个重要领域的贡献。



## **4. Poisons that are learned faster are more effective**

学得越快的毒药越有效 cs.LG

8 pages, 4 figures. Accepted to CVPR 2022 Art of Robustness Workshop

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.08615v1)

**Authors**: Pedro Sandoval-Segura, Vasu Singla, Liam Fowl, Jonas Geiping, Micah Goldblum, David Jacobs, Tom Goldstein

**Abstracts**: Imperceptible poisoning attacks on entire datasets have recently been touted as methods for protecting data privacy. However, among a number of defenses preventing the practical use of these techniques, early-stopping stands out as a simple, yet effective defense. To gauge poisons' vulnerability to early-stopping, we benchmark error-minimizing, error-maximizing, and synthetic poisons in terms of peak test accuracy over 100 epochs and make a number of surprising observations. First, we find that poisons that reach a low training loss faster have lower peak test accuracy. Second, we find that a current state-of-the-art error-maximizing poison is 7 times less effective when poison training is stopped at epoch 8. Third, we find that stronger, more transferable adversarial attacks do not make stronger poisons. We advocate for evaluating poisons in terms of peak test accuracy.

摘要: 对整个数据集的潜伏中毒攻击最近被吹捧为保护数据隐私的方法。然而，在阻止这些技术实际使用的许多防御措施中，提前停止是一种简单而有效的防御措施。为了衡量毒药对提前停止的脆弱性，我们根据100个纪元的峰值测试精度对误差最小化、误差最大化和合成毒药进行了基准测试，并进行了许多令人惊讶的观察。首先，我们发现毒药达到低训练损失的速度越快，峰值测试精度就越低。其次，我们发现当毒药训练在纪元8停止时，当前最先进的最大化错误的毒药的有效性降低了7倍。第三，我们发现更强、更具转移性的对抗性攻击不会产生更强的毒药。我们主张根据峰值测试的准确性来评估毒物。



## **5. Metamorphic Testing-based Adversarial Attack to Fool Deepfake Detectors**

基于变形测试的对愚人深伪检测器的攻击 cs.CV

paper submitted to 26TH International Conference on Pattern  Recognition (ICPR2022)

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.08612v1)

**Authors**: Nyee Thoang Lim, Meng Yi Kuan, Muxin Pu, Mei Kuan Lim, Chun Yong Chong

**Abstracts**: Deepfakes utilise Artificial Intelligence (AI) techniques to create synthetic media where the likeness of one person is replaced with another. There are growing concerns that deepfakes can be maliciously used to create misleading and harmful digital contents. As deepfakes become more common, there is a dire need for deepfake detection technology to help spot deepfake media. Present deepfake detection models are able to achieve outstanding accuracy (>90%). However, most of them are limited to within-dataset scenario, where the same dataset is used for training and testing. Most models do not generalise well enough in cross-dataset scenario, where models are tested on unseen datasets from another source. Furthermore, state-of-the-art deepfake detection models rely on neural network-based classification models that are known to be vulnerable to adversarial attacks. Motivated by the need for a robust deepfake detection model, this study adapts metamorphic testing (MT) principles to help identify potential factors that could influence the robustness of the examined model, while overcoming the test oracle problem in this domain. Metamorphic testing is specifically chosen as the testing technique as it fits our demand to address learning-based system testing with probabilistic outcomes from largely black-box components, based on potentially large input domains. We performed our evaluations on MesoInception-4 and TwoStreamNet models, which are the state-of-the-art deepfake detection models. This study identified makeup application as an adversarial attack that could fool deepfake detectors. Our experimental results demonstrate that both the MesoInception-4 and TwoStreamNet models degrade in their performance by up to 30\% when the input data is perturbed with makeup.

摘要: Deepfakes利用人工智能(AI)技术来创建合成媒体，其中一个人的肖像被另一个人取代。越来越多的人担心，深度假货可能被恶意用于创建误导性和有害的数字内容。随着深度假变得越来越普遍，迫切需要深度假检测技术来帮助识别深度假媒体。现有的深度伪检测模型能够达到显著的准确率(>90%)。然而，它们中的大多数仅限于数据集内的场景，其中相同的数据集用于训练和测试。大多数模型在跨数据集情况下不能很好地泛化，在这种情况下，模型是在来自另一个来源的不可见的数据集上进行测试的。此外，最先进的深度伪检测模型依赖于基于神经网络的分类模型，这些模型已知容易受到对手攻击。出于对稳健深度伪检测模型的需求，本研究采用变形测试(MT)原理来帮助识别可能影响被检查模型的稳健性的潜在因素，同时克服了该领域的测试预言问题。变形测试被特别选为测试技术，因为它符合我们的需求，以解决基于学习的系统测试，其结果主要来自黑盒组件，基于潜在的大输入域。我们对目前最先进的深度伪检测模型MesoInception-4和TwoStreamNet模型进行了评估。这项研究发现，化妆应用是一种对抗性攻击，可以愚弄深度假货检测器。实验结果表明，当输入数据受到置乱干扰时，两种模型的性能都下降了30%。



## **6. A Comprehensive Survey on Trustworthy Graph Neural Networks: Privacy, Robustness, Fairness, and Explainability**

可信图神经网络研究综述：私密性、稳健性、公平性和可解释性 cs.LG

**SubmitDate**: 2022-04-18    [paper-pdf](http://arxiv.org/pdf/2204.08570v1)

**Authors**: Enyan Dai, Tianxiang Zhao, Huaisheng Zhu, Junjie Xu, Zhimeng Guo, Hui Liu, Jiliang Tang, Suhang Wang

**Abstracts**: Graph Neural Networks (GNNs) have made rapid developments in the recent years. Due to their great ability in modeling graph-structured data, GNNs are vastly used in various applications, including high-stakes scenarios such as financial analysis, traffic predictions, and drug discovery. Despite their great potential in benefiting humans in the real world, recent study shows that GNNs can leak private information, are vulnerable to adversarial attacks, can inherit and magnify societal bias from training data and lack interpretability, which have risk of causing unintentional harm to the users and society. For example, existing works demonstrate that attackers can fool the GNNs to give the outcome they desire with unnoticeable perturbation on training graph. GNNs trained on social networks may embed the discrimination in their decision process, strengthening the undesirable societal bias. Consequently, trustworthy GNNs in various aspects are emerging to prevent the harm from GNN models and increase the users' trust in GNNs. In this paper, we give a comprehensive survey of GNNs in the computational aspects of privacy, robustness, fairness, and explainability. For each aspect, we give the taxonomy of the related methods and formulate the general frameworks for the multiple categories of trustworthy GNNs. We also discuss the future research directions of each aspect and connections between these aspects to help achieve trustworthiness.

摘要: 近年来，图形神经网络(GNN)得到了迅速发展。由于其强大的图结构数据建模能力，GNN被广泛应用于各种应用中，包括金融分析、交通预测和药物发现等高风险场景。尽管GNN在现实世界中具有造福人类的巨大潜力，但最近的研究表明，GNN可能会泄露私人信息，容易受到对手攻击，会继承和放大来自训练数据的社会偏见，并且缺乏可解释性，这有可能对用户和社会造成无意的伤害。例如，现有的工作表明，攻击者可以欺骗GNN给出他们想要的结果，而训练图上的扰动并不明显。在社交网络上培训的GNN可能会在其决策过程中嵌入歧视，强化不受欢迎的社会偏见。因此，各个方面的可信GNN应运而生，以防止GNN模型的危害，增加用户对GNN的信任。本文从私密性、健壮性、公平性和可解释性等方面对GNN进行了全面的综述。对于每个方面，我们给出了相关方法的分类，并制定了多个类别的可信GNN的一般框架。我们还讨论了各个方面的未来研究方向以及这些方面之间的联系，以帮助实现可信性。



## **7. Optimal Layered Defense For Site Protection**

站点保护的最优分层防御 cs.OH

**SubmitDate**: 2022-04-18    [paper-pdf](http://arxiv.org/pdf/2204.08961v1)

**Authors**: Tsvetan Asamov, Emre Yamangil, Endre Boros, Paul Kantor, Fred Roberts

**Abstracts**: We present a model for layered security with applications to the protection of sites such as stadiums or large gathering places. We formulate the problem as one of maximizing the capture of illegal contraband. The objective function is indefinite and only limited information can be gained when the problem is solved by standard convex optimization methods. In order to solve the model, we develop a dynamic programming approach, and study its convergence properties. Additionally, we formulate a version of the problem aimed at addressing intelligent adversaries who can adjust their direction of attack as they observe changes in the site security. Furthermore, we also develop a method for the solution of the latter model. Finally, we perform computational experiments to demonstrate the use of our methods.

摘要: 我们提出了一种分层安全模型，并将其应用于体育场馆或大型集会场所等场所的保护。我们把这个问题表述为最大限度地捕获非法违禁品的问题。用标准的凸优化方法求解时，目标函数是不确定的，只能得到有限的信息。为了求解该模型，我们提出了一种动态规划方法，并研究了它的收敛性质。此外，我们制定了一个版本的问题，旨在解决智能对手谁可以调整他们的攻击方向，因为他们观察到网站安全的变化。此外，我们还发展了一种求解后一种模型的方法。最后，我们进行了计算实验，以演示我们的方法的使用。



## **8. Revisiting the Adversarial Robustness-Accuracy Tradeoff in Robot Learning**

机器人学习中对抗性稳健性与准确性权衡的再认识 cs.RO

**SubmitDate**: 2022-04-15    [paper-pdf](http://arxiv.org/pdf/2204.07373v1)

**Authors**: Mathias Lechner, Alexander Amini, Daniela Rus, Thomas A. Henzinger

**Abstracts**: Adversarial training (i.e., training on adversarially perturbed input data) is a well-studied method for making neural networks robust to potential adversarial attacks during inference. However, the improved robustness does not come for free but rather is accompanied by a decrease in overall model accuracy and performance. Recent work has shown that, in practical robot learning applications, the effects of adversarial training do not pose a fair trade-off but inflict a net loss when measured in holistic robot performance. This work revisits the robustness-accuracy trade-off in robot learning by systematically analyzing if recent advances in robust training methods and theory in conjunction with adversarial robot learning can make adversarial training suitable for real-world robot applications. We evaluate a wide variety of robot learning tasks ranging from autonomous driving in a high-fidelity environment amenable to sim-to-real deployment, to mobile robot gesture recognition. Our results demonstrate that, while these techniques make incremental improvements on the trade-off on a relative scale, the negative side-effects caused by adversarial training still outweigh the improvements by an order of magnitude. We conclude that more substantial advances in robust learning methods are necessary before they can benefit robot learning tasks in practice.

摘要: 对抗性训练(即对对抗性扰动的输入数据进行训练)是一种研究得很好的方法，可以使神经网络在推理过程中对潜在的对抗性攻击具有健壮性。然而，稳健性的提高并不是免费的，而是伴随着总体模型精度和性能的下降。最近的工作表明，在实际的机器人学习应用中，对抗性训练的效果并不构成公平的权衡，而是在衡量整体机器人性能时造成净损失。这项工作通过系统地分析稳健训练方法和理论的最新进展以及对抗性机器人学习是否可以使对抗性训练适用于现实世界的机器人应用，重新审视了机器人学习中的稳健性和精确度之间的权衡。我们评估了各种机器人学习任务，从高保真环境中的自动驾驶到模拟真实的部署，再到移动机器人手势识别。我们的结果表明，虽然这些技术在相对规模上对权衡做出了增量改进，但对抗性训练造成的负面副作用仍然比改进多一个数量级。我们的结论是，在健壮学习方法能够在实践中有益于机器人学习任务之前，需要更多实质性的进步。



## **9. Can You Spot the Chameleon? Adversarially Camouflaging Images from Co-Salient Object Detection**

你能认出变色龙吗？对抗伪装图像以防止共显著目标检测 cs.CV

Accepted to CVPR 2022

**SubmitDate**: 2022-04-18    [paper-pdf](http://arxiv.org/pdf/2009.09258v5)

**Authors**: Ruijun Gao, Qing Guo, Felix Juefei-Xu, Hongkai Yu, Huazhu Fu, Wei Feng, Yang Liu, Song Wang

**Abstracts**: Co-salient object detection (CoSOD) has recently achieved significant progress and played a key role in retrieval-related tasks. However, it inevitably poses an entirely new safety and security issue, i.e., highly personal and sensitive content can potentially be extracting by powerful CoSOD methods. In this paper, we address this problem from the perspective of adversarial attacks and identify a novel task: adversarial co-saliency attack. Specially, given an image selected from a group of images containing some common and salient objects, we aim to generate an adversarial version that can mislead CoSOD methods to predict incorrect co-salient regions. Note that, compared with general white-box adversarial attacks for classification, this new task faces two additional challenges: (1) low success rate due to the diverse appearance of images in the group; (2) low transferability across CoSOD methods due to the considerable difference between CoSOD pipelines. To address these challenges, we propose the very first black-box joint adversarial exposure and noise attack (Jadena), where we jointly and locally tune the exposure and additive perturbations of the image according to a newly designed high-feature-level contrast-sensitive loss function. Our method, without any information on the state-of-the-art CoSOD methods, leads to significant performance degradation on various co-saliency detection datasets and makes the co-salient objects undetectable. This can have strong practical benefits in properly securing the large number of personal photos currently shared on the Internet. Moreover, our method is potential to be utilized as a metric for evaluating the robustness of CoSOD methods.

摘要: 共显著目标检测(CoSOD)近年来取得了重大进展，在检索相关任务中发挥了关键作用。然而，它不可避免地提出了一个全新的安全问题，即高度个人和敏感的内容可能会被强大的CoSOD方法提取出来。在本文中，我们从对抗性攻击的角度来解决这个问题，并提出了一种新的任务：对抗性共显攻击。特别是，给定一幅从一组包含一些常见和显著对象的图像中选择的图像，我们的目标是生成一个对抗性版本，该版本可能会误导CoSOD方法预测错误的共同显著区域。注意到，与一般的白盒对抗性分类攻击相比，这项新任务面临着两个额外的挑战：(1)由于组中图像的多样性，成功率较低；(2)由于CoSOD管道之间的巨大差异，CoSOD方法之间的可传输性较低。为了应对这些挑战，我们提出了第一个黑盒联合对抗性曝光和噪声攻击(Jadena)，其中我们根据新设计的高特征级别对比度敏感损失函数来联合和局部地调整图像的曝光和加性扰动。我们的方法在没有关于最新CoSOD方法的任何信息的情况下，导致在各种共显著检测数据集上的性能显著下降，并且使得共显著对象不可检测。这对妥善保护目前在互联网上共享的大量个人照片具有很大的实际好处。此外，我们的方法有可能被用作评估CoSOD方法的稳健性的一个度量。



## **10. Robotic and Generative Adversarial Attacks in Offline Writer-independent Signature Verification**

离线作者无关签名验证中的机器人和生成性对抗攻击 cs.RO

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2204.07246v1)

**Authors**: Jordan J. Bird

**Abstracts**: This study explores how robots and generative approaches can be used to mount successful false-acceptance adversarial attacks on signature verification systems. Initially, a convolutional neural network topology and data augmentation strategy are explored and tuned, producing an 87.12% accurate model for the verification of 2,640 human signatures. Two robots are then tasked with forging 50 signatures, where 25 are used for the verification attack, and the remaining 25 are used for tuning of the model to defend against them. Adversarial attacks on the system show that there exists an information security risk; the Line-us robotic arm can fool the system 24% of the time and the iDraw 2.0 robot 32% of the time. A conditional GAN finds similar success, with around 30% forged signatures misclassified as genuine. Following fine-tune transfer learning of robotic and generative data, adversarial attacks are reduced below the model threshold by both robots and the GAN. It is observed that tuning the model reduces the risk of attack by robots to 8% and 12%, and that conditional generative adversarial attacks can be reduced to 4% when 25 images are presented and 5% when 1000 images are presented.

摘要: 这项研究探索了如何使用机器人和生成性方法来对签名验证系统发起成功的虚假接受对抗性攻击。首先，探索和调整了卷积神经网络的拓扑结构和数据增强策略，产生了一个87.12%的模型，用于验证2640个人的签名。然后，两个机器人的任务是伪造50个签名，其中25个用于验证攻击，其余25个用于调整模型以防御它们。对该系统的对抗性攻击表明，存在信息安全风险；Line-us机械臂可以在24%的时间内欺骗系统，iDraw 2.0机器人可以在32%的时间内欺骗系统。有条件的GAN发现了类似的成功，大约30%的伪造签名被错误归类为真签名。在对机器人和生成性数据进行微调传递学习后，机器人和GAN都将对抗性攻击降低到模型阈值以下。实验结果表明，调整模型后，机器人的攻击风险分别降低到8%和12%，当呈现25幅图像时，条件生成性对抗攻击可以降低到4%，当呈现1000幅图像时，条件生成性对抗攻击可以降低到5%。



## **11. ExPLoit: Extracting Private Labels in Split Learning**

漏洞：在分裂学习中提取私有标签 cs.CR

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2112.01299v2)

**Authors**: Sanjay Kariyappa, Moinuddin K Qureshi

**Abstracts**: Split learning is a popular technique used for vertical federated learning (VFL), where the goal is to jointly train a model on the private input and label data held by two parties. This technique uses a split-model, trained end-to-end, by exchanging the intermediate representations (IR) of the inputs and gradients of the IR between the two parties. We propose ExPLoit - a label-leakage attack that allows an adversarial input-owner to extract the private labels of the label-owner during split-learning. ExPLoit frames the attack as a supervised learning problem by using a novel loss function that combines gradient-matching and several regularization terms developed using key properties of the dataset and models. Our evaluations show that ExPLoit can uncover the private labels with near-perfect accuracy of up to 99.96%. Our findings underscore the need for better training techniques for VFL.

摘要: 分裂学习是垂直联合学习(VFL)中一种流行的技术，其目标是联合训练一个关于双方持有的私有输入和标签数据的模型。该技术通过在双方之间交换IR的输入和梯度的中间表示(IR)来使用端到端训练的拆分模型。我们提出了利用攻击-一种标签泄漏攻击，允许敌意输入所有者在分裂学习过程中提取标签所有者的私有标签。利用攻击通过使用一种新的损失函数将攻击帧化为有监督的学习问题，该损失函数结合了梯度匹配和利用数据集和模型的关键属性开发的几个正则化项。我们的评估表明，利用漏洞可以发现私人标签的近乎完美的准确率高达99.96%。我们的发现强调了为VFL提供更好的训练技术的必要性。



## **12. From Environmental Sound Representation to Robustness of 2D CNN Models Against Adversarial Attacks**

从环境声音表示到2D CNN模型对敌方攻击的稳健性 cs.SD

32 pages, Preprint Submitted to Journal of Applied Acoustics. arXiv  admin note: substantial text overlap with arXiv:2007.13703

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2204.07018v1)

**Authors**: Mohammad Esmaeilpour, Patrick Cardinal, Alessandro Lameiras Koerich

**Abstracts**: This paper investigates the impact of different standard environmental sound representations (spectrograms) on the recognition performance and adversarial attack robustness of a victim residual convolutional neural network, namely ResNet-18. Our main motivation for focusing on such a front-end classifier rather than other complex architectures is balancing recognition accuracy and the total number of training parameters. Herein, we measure the impact of different settings required for generating more informative Mel-frequency cepstral coefficient (MFCC), short-time Fourier transform (STFT), and discrete wavelet transform (DWT) representations on our front-end model. This measurement involves comparing the classification performance over the adversarial robustness. We demonstrate an inverse relationship between recognition accuracy and model robustness against six benchmarking attack algorithms on the balance of average budgets allocated by the adversary and the attack cost. Moreover, our experimental results have shown that while the ResNet-18 model trained on DWT spectrograms achieves a high recognition accuracy, attacking this model is relatively more costly for the adversary than other 2D representations. We also report some results on different convolutional neural network architectures such as ResNet-34, ResNet-56, AlexNet, and GoogLeNet, SB-CNN, and LSTM-based.

摘要: 研究了不同标准环境声音表示(谱图)对受害者残差卷积神经网络ResNet-18识别性能和对抗攻击稳健性的影响。我们关注这样的前端分类器而不是其他复杂的体系结构的主要动机是在识别精度和训练参数总数之间取得平衡。在这里，我们测量了在我们的前端模型上生成更多信息的梅尔频率倒谱系数(MFCC)、短时傅立叶变换(STFT)和离散小波变换(DWT)表示所需的不同设置的影响。这种测量包括比较分类性能与对手健壮性。在平衡对手分配的平均预算和攻击成本的基础上，我们证明了识别准确率与模型对六种基准攻击算法的稳健性成反比关系。此外，我们的实验结果表明，虽然基于DWT谱图训练的ResNet-18模型达到了较高的识别精度，但攻击该模型的代价相对较高。我们还报告了不同卷积神经网络结构的一些结果，例如ResNet-34、ResNet-56、AlexNet和GoogLeNet、SB-CNN和基于LSTM的结构。



## **13. Finding MNEMON: Reviving Memories of Node Embeddings**

寻找Mnemon：唤醒节点嵌入的记忆 cs.LG

To Appear in the 29th ACM Conference on Computer and Communications  Security (CCS), November 7-11, 2022

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2204.06963v1)

**Authors**: Yun Shen, Yufei Han, Zhikun Zhang, Min Chen, Ting Yu, Michael Backes, Yang Zhang, Gianluca Stringhini

**Abstracts**: Previous security research efforts orbiting around graphs have been exclusively focusing on either (de-)anonymizing the graphs or understanding the security and privacy issues of graph neural networks. Little attention has been paid to understand the privacy risks of integrating the output from graph embedding models (e.g., node embeddings) with complex downstream machine learning pipelines. In this paper, we fill this gap and propose a novel model-agnostic graph recovery attack that exploits the implicit graph structural information preserved in the embeddings of graph nodes. We show that an adversary can recover edges with decent accuracy by only gaining access to the node embedding matrix of the original graph without interactions with the node embedding models. We demonstrate the effectiveness and applicability of our graph recovery attack through extensive experiments.

摘要: 以前围绕图的安全研究一直专注于图的(去)匿名化或理解图神经网络的安全和隐私问题。很少有人注意到将图嵌入模型(例如，节点嵌入)的输出与复杂的下游机器学习管道集成的隐私风险。在本文中，我们填补了这一空白，并提出了一种新的模型不可知图恢复攻击，该攻击利用了图节点嵌入中保留的隐含的图结构信息。我们证明了敌手只需访问原始图的节点嵌入矩阵，而不需要与节点嵌入模型交互，就能以相当高的精度恢复边。我们通过大量的实验证明了我们的图恢复攻击的有效性和适用性。



## **14. Arbitrarily Varying Wiretap Channels with Non-Causal Side Information at the Jammer**

干扰机具有非因果边信息的任意变化的窃听信道 cs.IT

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2001.03035v4)

**Authors**: Carsten Rudolf Janda, Moritz Wiese, Eduard A. Jorswieck, Holger Boche

**Abstracts**: Secure communication in a potentially malicious environment becomes more and more important. The arbitrarily varying wiretap channel (AVWC) provides information theoretical bounds on how much information can be exchanged even in the presence of an active attacker. If the active attacker has non-causal side information, situations in which a legitimate communication system has been hacked, can be modeled. We investigate the AVWC with non-causal side information at the jammer for the case that there exists a best channel to the eavesdropper. Non-causal side information means that the transmitted codeword is known to an active adversary before it is transmitted. By considering the maximum error criterion, we allow also messages to be known at the jammer before the corresponding codeword is transmitted. A single letter formula for the common randomness secrecy capacity is derived. Additionally, we provide a single letter formula for the common randomness secrecy capacity, for the cases that the channel to the eavesdropper is strongly degraded, strongly noisier, or strongly less capable with respect to the main channel. Furthermore, we compare our results to the random code secrecy capacity for the cases of maximum error criterion but without non-causal side information at the jammer, maximum error criterion with non-causal side information of the messages at the jammer, and the case of average error criterion without non-causal side information at the jammer.

摘要: 在潜在的恶意环境中进行安全通信变得越来越重要。任意变化的窃听通道(AVWC)提供了信息理论界限，即即使在活动攻击者在场的情况下也可以交换多少信息。如果主动攻击者具有非因果的辅助信息，则可以模拟合法通信系统被黑客攻击的情况。在干扰机存在最佳通道的情况下，我们研究了干扰机侧信息为非因果的AVWC。非因果辅助信息意味着所传输的码字在被传输之前为活跃的敌手所知。通过考虑最大差错准则，我们还允许在发送相应码字之前在干扰器处知道消息。给出了常见随机性保密容量的单字母公式。此外，对于到窃听者的信道相对于主信道是强退化、强噪声或强弱能力的情况，我们还给出了公共随机性保密容量的单字母公式。此外，我们还将我们的结果与干扰机无非因果边信息的最大差错准则、干扰机有非因果边信息的最大差错准则以及干扰机无非因果边信息的平均差错准则的情况下的随机码保密容量进行了比较。



## **15. Improving Adversarial Transferability with Gradient Refining**

利用梯度精化提高对手的可转移性 cs.CV

Accepted at CVPR 2021 Workshop on Adversarial Machine Learning in  Real-World Computer Vision Systems and Online Challenges. The extension  vision of this paper, please refer to arxiv:2203.13479

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2105.04834v3)

**Authors**: Guoqiu Wang, Huanqian Yan, Ying Guo, Xingxing Wei

**Abstracts**: Deep neural networks are vulnerable to adversarial examples, which are crafted by adding human-imperceptible perturbations to original images. Most existing adversarial attack methods achieve nearly 100% attack success rates under the white-box setting, but only achieve relatively low attack success rates under the black-box setting. To improve the transferability of adversarial examples for the black-box setting, several methods have been proposed, e.g., input diversity, translation-invariant attack, and momentum-based attack. In this paper, we propose a method named Gradient Refining, which can further improve the adversarial transferability by correcting useless gradients introduced by input diversity through multiple transformations. Our method is generally applicable to many gradient-based attack methods combined with input diversity. Extensive experiments are conducted on the ImageNet dataset and our method can achieve an average transfer success rate of 82.07% for three different models under single-model setting, which outperforms the other state-of-the-art methods by a large margin of 6.0% averagely. And we have applied the proposed method to the competition CVPR 2021 Unrestricted Adversarial Attacks on ImageNet organized by Alibaba and won the second place in attack success rates among 1558 teams.

摘要: 深度神经网络很容易受到敌意示例的攻击，这些示例是通过在原始图像中添加人类无法察觉的扰动来构建的。现有的对抗性攻击方法大多在白盒环境下攻击成功率接近100%，而在黑盒环境下攻击成功率相对较低。为了提高黑盒环境下敌意例子的可转移性，人们提出了输入多样性、平移不变攻击和动量攻击等方法。本文提出了一种梯度精化方法，通过多次变换修正输入分集引入的无用梯度，进一步提高了算法的对抗性可转移性。该方法普遍适用于多种结合输入分集的基于梯度的攻击方法。在ImageNet数据集上进行了大量的实验，在单一模型设置下，我们的方法对三种不同模型的平均传输成功率达到了82.07%，比其他最先进的方法平均提高了6.0%。并将提出的方法应用于阿里巴巴组织的CVPR 2021无限对抗性攻击ImageNet比赛中，获得了1558支球队攻击成功率的第二名。



## **16. Real-time Adversarial Perturbations against Deep Reinforcement Learning Policies: Attacks and Defenses**

深度强化学习策略的实时对抗性扰动：攻击与防御 cs.LG

13 pages, 6 figures

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2106.08746v3)

**Authors**: Buse G. A. Tekgul, Shelly Wang, Samuel Marchal, N. Asokan

**Abstracts**: Recent work has shown that deep reinforcement learning (DRL) policies are vulnerable to adversarial perturbations. Adversaries can mislead policies of DRL agents by perturbing the state of the environment observed by the agents. Existing attacks are feasible in principle but face challenges in practice, either by being too slow to fool DRL policies in real time or by modifying past observations stored in the agent's memory. We show that using the Universal Adversarial Perturbation (UAP) method to compute perturbations, independent of the individual inputs to which they are applied to, can fool DRL policies effectively and in real time. We describe three such attack variants. Via an extensive evaluation using three Atari 2600 games, we show that our attacks are effective, as they fully degrade the performance of three different DRL agents (up to 100%, even when the $l_\infty$ bound on the perturbation is as small as 0.01). It is faster compared to the response time (0.6ms on average) of different DRL policies, and considerably faster than prior attacks using adversarial perturbations (1.8ms on average). We also show that our attack technique is efficient, incurring an online computational cost of 0.027ms on average. Using two further tasks involving robotic movement, we confirm that our results generalize to more complex DRL tasks. Furthermore, we demonstrate that the effectiveness of known defenses diminishes against universal perturbations. We propose an effective technique that detects all known adversarial perturbations against DRL policies, including all the universal perturbations presented in this paper.

摘要: 最近的工作表明，深度强化学习(DRL)策略容易受到对抗性扰动的影响。攻击者可以通过扰乱代理观察到的环境状态来误导DRL代理的策略。现有的攻击在原则上是可行的，但在实践中面临挑战，要么太慢，无法实时愚弄DRL策略，要么修改存储在代理内存中的过去观察。我们表明，使用通用对抗扰动(UAP)方法来计算扰动，而与应用扰动的个体输入无关，可以有效且实时地愚弄DRL策略。我们描述了三种这样的攻击变体。通过使用三款Atari 2600游戏进行的广泛评估，我们表明我们的攻击是有效的，因为它们完全降低了三种不同DRL代理的性能(高达100%，即使在扰动上的$l_\infty$约束小到0.01)。它比不同DRL策略的响应时间(平均0.6ms)更快，并且比以前使用对抗性扰动的攻击(平均1.8ms)要快得多。我们还证明了我们的攻击技术是有效的，平均在线计算成本为0.027ms。使用另外两个涉及机器人移动的任务，我们确认我们的结果推广到更复杂的DRL任务。此外，我们还证明了已知防御措施对普遍扰动的有效性会降低。我们提出了一种有效的技术来检测所有已知的针对DRL策略的对抗性扰动，包括本文提出的所有通用扰动。



## **17. Overparameterized Linear Regression under Adversarial Attacks**

对抗性攻击下的超参数线性回归 stat.ML

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2204.06274v1)

**Authors**: Antônio H. Ribeiro, Thomas B. Schön

**Abstracts**: As machine learning models start to be used in critical applications, their vulnerabilities and brittleness become a pressing concern. Adversarial attacks are a popular framework for studying these vulnerabilities. In this work, we study the error of linear regression in the face of adversarial attacks. We provide bounds of the error in terms of the traditional risk and the parameter norm and show how these bounds can be leveraged and make it possible to use analysis from non-adversarial setups to study the adversarial risk. The usefulness of these results is illustrated by shedding light on whether or not overparameterized linear models can be adversarially robust. We show that adding features to linear models might be either a source of additional robustness or brittleness. We show that these differences appear due to scaling and how the $\ell_1$ and $\ell_2$ norms of random projections concentrate. We also show how the reformulation we propose allows for solving adversarial training as a convex optimization problem. This is then used as a tool to study how adversarial training and other regularization methods might affect the robustness of the estimated models.

摘要: 随着机器学习模型开始在关键应用中使用，它们的脆弱性和脆性成为一个紧迫的问题。对抗性攻击是研究这些漏洞的流行框架。在这项工作中，我们研究了线性回归在面对对手攻击时的误差。我们给出了关于传统风险和参数范数的误差的界，并说明了如何利用这些界来利用这些界，使得从非对抗性设置的分析来研究对抗性风险成为可能。这些结果的有用之处在于揭示了过度参数化线性模型是否具有相反的稳健性。我们表明，向线性模型添加特征可能是额外的稳健性或脆性的来源。我们证明了这些差异是由于尺度以及随机投影的$\ell_1$和$\ell_2$范数是如何集中的。我们还展示了我们提出的重新公式如何允许将对抗性训练作为一个凸优化问题来解决。然后将其用作研究对抗性训练和其他正则化方法如何影响估计模型的稳健性的工具。



## **18. Towards A Critical Evaluation of Robustness for Deep Learning Backdoor Countermeasures**

深度学习后门对策的稳健性评测 cs.CR

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2204.06273v1)

**Authors**: Huming Qiu, Hua Ma, Zhi Zhang, Alsharif Abuadbba, Wei Kang, Anmin Fu, Yansong Gao

**Abstracts**: Since Deep Learning (DL) backdoor attacks have been revealed as one of the most insidious adversarial attacks, a number of countermeasures have been developed with certain assumptions defined in their respective threat models. However, the robustness of these countermeasures is inadvertently ignored, which can introduce severe consequences, e.g., a countermeasure can be misused and result in a false implication of backdoor detection.   For the first time, we critically examine the robustness of existing backdoor countermeasures with an initial focus on three influential model-inspection ones that are Neural Cleanse (S&P'19), ABS (CCS'19), and MNTD (S&P'21). Although the three countermeasures claim that they work well under their respective threat models, they have inherent unexplored non-robust cases depending on factors such as given tasks, model architectures, datasets, and defense hyper-parameter, which are \textit{not even rooted from delicate adaptive attacks}. We demonstrate how to trivially bypass them aligned with their respective threat models by simply varying aforementioned factors. Particularly, for each defense, formal proofs or empirical studies are used to reveal its two non-robust cases where it is not as robust as it claims or expects, especially the recent MNTD. This work highlights the necessity of thoroughly evaluating the robustness of backdoor countermeasures to avoid their misleading security implications in unknown non-robust cases.

摘要: 由于深度学习(DL)后门攻击已被发现是最隐蔽的敌意攻击之一，因此已经开发了一些对策，并在各自的威胁模型中定义了某些假设。然而，这些对策的稳健性被无意中忽视了，这可能会带来严重的后果，例如，对策可能被误用，并导致后门检测的错误含义。我们首次批判性地检验了现有后门对策的稳健性，最初集中在三个有影响力的模型检查对策上，它们是神经清洗(S&P‘19)、ABS(CCS’19)和MNTD(S&P‘21)。虽然这三种对策声称它们在各自的威胁模型下都工作得很好，但它们固有的非稳健性情况取决于给定的任务、模型体系结构、数据集和防御超参数等因素，而这些因素甚至不是源于脆弱的自适应攻击}。我们将演示如何通过简单地改变上述因素来绕过它们，使其与各自的威胁模型保持一致。特别是，对于每一种辩护，形式证明或实证研究都被用来揭示它的两种不稳健的情况，其中它并不像它声称或期望的那样稳健，特别是最近的MNTD。这项工作强调了彻底评估后门对策的稳健性的必要性，以避免在未知的非稳健性情况下它们具有误导性的安全影响。



## **19. Towards Practical Robustness Analysis for DNNs based on PAC-Model Learning**

基于PAC模型学习的DNN实用健壮性分析 cs.LG

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2101.10102v2)

**Authors**: Renjue Li, Pengfei Yang, Cheng-Chao Huang, Youcheng Sun, Bai Xue, Lijun Zhang

**Abstracts**: To analyse local robustness properties of deep neural networks (DNNs), we present a practical framework from a model learning perspective. Based on black-box model learning with scenario optimisation, we abstract the local behaviour of a DNN via an affine model with the probably approximately correct (PAC) guarantee. From the learned model, we can infer the corresponding PAC-model robustness property. The innovation of our work is the integration of model learning into PAC robustness analysis: that is, we construct a PAC guarantee on the model level instead of sample distribution, which induces a more faithful and accurate robustness evaluation. This is in contrast to existing statistical methods without model learning. We implement our method in a prototypical tool named DeepPAC. As a black-box method, DeepPAC is scalable and efficient, especially when DNNs have complex structures or high-dimensional inputs. We extensively evaluate DeepPAC, with 4 baselines (using formal verification, statistical methods, testing and adversarial attack) and 20 DNN models across 3 datasets, including MNIST, CIFAR-10, and ImageNet. It is shown that DeepPAC outperforms the state-of-the-art statistical method PROVERO, and it achieves more practical robustness analysis than the formal verification tool ERAN. Also, its results are consistent with existing DNN testing work like DeepGini.

摘要: 为了分析深度神经网络(DNN)的局部稳健性，从模型学习的角度提出了一个实用的框架。基于场景优化的黑盒模型学习，我们通过仿射模型在可能近似正确(PAC)的保证下抽象DNN的局部行为。从学习的模型中，我们可以推断出相应的PAC模型的稳健性。我们工作的创新之处在于将模型学习融入到PAC稳健性分析中：即在模型级别而不是样本分布上构造PAC保证，从而得到更真实和准确的稳健性评估。这与没有模型学习的现有统计方法形成了鲜明对比。我们在一个原型工具DeepPAC中实现了我们的方法。作为一种黑盒方法，DeepPAC具有可扩展性和高效率，特别是当DNN具有复杂的结构或高维输入时。我们对DeepPAC进行了广泛的评估，使用了4个基线(使用正式验证、统计方法、测试和对抗性攻击)和20个DNN模型，涉及MNIST、CIFAR-10和ImageNet等3个数据集。结果表明，DeepPAC的性能优于目前最先进的统计方法PROVERO，并且比形式化验证工具ERAN实现了更实用的健壮性分析。此外，它的结果与DeepGini等现有的DNN测试工作是一致的。



## **20. Stealing Malware Classifiers and AVs at Low False Positive Conditions**

在低误报条件下窃取恶意软件分类器和反病毒软件 cs.CR

12 pages, 8 figures, 6 tables. Under review

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2204.06241v1)

**Authors**: Maria Rigaki, Sebastian Garcia

**Abstracts**: Model stealing attacks have been successfully used in many machine learning domains, but there is little understanding of how these attacks work in the malware detection domain. Malware detection and, in general, security domains have very strong requirements of low false positive rates (FPR). However, these requirements are not the primary focus of the existing model stealing literature. Stealing attacks create surrogate models that perform similarly to a target model using a limited amount of queries to the target. The first stage of this study is the evaluation of active learning model stealing attacks against publicly available stand-alone machine learning malware classifiers and antivirus products (AVs). We propose a new neural network architecture for surrogate models that outperforms the existing state of the art on low FPR conditions. The surrogates were evaluated on their agreement with the targeted models. Good surrogates of the stand-alone classifiers were created with up to 99% agreement with the target models, using less than 4% of the original training dataset size. Good AV surrogates were also possible to train, but with a lower agreement. The second stage used the best surrogates as well as the target models to generate adversarial malware using the MAB framework to test stand-alone models and AVs (offline and online). Results showed that surrogate models could generate adversarial samples that evade the targets but are less successful than the targets themselves. Using surrogates, however, is a necessity for attackers, given that attacks against AVs are extremely time-consuming and easily detected when the AVs are connected to the internet.

摘要: 模型窃取攻击已经成功地应用于许多机器学习领域，但对于这些攻击在恶意软件检测领域的工作原理却知之甚少。通常，恶意软件检测和安全域对低误报比率(FPR)有非常强烈的要求。然而，这些要求并不是现有窃取文献模型的主要关注点。窃取攻击创建代理模型，该代理模型使用对目标的有限数量的查询来执行类似于目标模型的操作。本研究的第一阶段是评估针对公开可用的独立机器学习恶意软件分类器和反病毒产品(AV)的主动学习模型窃取攻击。我们提出了一种新的神经网络体系结构，用于代理模型，在低FPR条件下性能优于现有技术。根据其与目标模型的一致性对代用品进行评估。使用不到原始训练数据集大小的4%，创建了独立分类器的良好代理，与目标模型的一致性高达99%。好的反病毒代言人也有可能接受培训，但协议的一致性较低。第二阶段使用最好的代理以及目标模型来生成敌意恶意软件，使用MAB框架测试独立模型和AVs(离线和在线)。结果表明，代理模型可以生成避开目标的对抗性样本，但不如目标本身那么成功。然而，使用代理对于攻击者来说是必要的，因为针对AVs的攻击非常耗时，并且在AVs连接到互联网时很容易被检测到。



## **21. Liuer Mihou: A Practical Framework for Generating and Evaluating Grey-box Adversarial Attacks against NIDS**

六二密侯：一种实用的生成和评估针对NIDS的灰盒攻击的框架 cs.CR

16 pages, 8 figures, planning on submitting to ACM CCS 2022

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2204.06113v1)

**Authors**: Ke He, Dan Dongseong Kim, Jing Sun, Jeong Do Yoo, Young Hun Lee, Huy Kang Kim

**Abstracts**: Due to its high expressiveness and speed, Deep Learning (DL) has become an increasingly popular choice as the detection algorithm for Network-based Intrusion Detection Systems (NIDSes). Unfortunately, DL algorithms are vulnerable to adversarial examples that inject imperceptible modifications to the input and cause the DL algorithm to misclassify the input. Existing adversarial attacks in the NIDS domain often manipulate the traffic features directly, which hold no practical significance because traffic features cannot be replayed in a real network. It remains a research challenge to generate practical and evasive adversarial attacks.   This paper presents the Liuer Mihou attack that generates practical and replayable adversarial network packets that can bypass anomaly-based NIDS deployed in the Internet of Things (IoT) networks. The core idea behind Liuer Mihou is to exploit adversarial transferability and generate adversarial packets on a surrogate NIDS constrained by predefined mutation operations to ensure practicality. We objectively analyse the evasiveness of Liuer Mihou against four ML-based algorithms (LOF, OCSVM, RRCF, and SOM) and the state-of-the-art NIDS, Kitsune. From the results of our experiment, we gain valuable insights into necessary conditions on the adversarial transferability of anomaly detection algorithms. Going beyond a theoretical setting, we replay the adversarial attack in a real IoT testbed to examine the practicality of Liuer Mihou. Furthermore, we demonstrate that existing feature-level adversarial defence cannot defend against Liuer Mihou and constructively criticise the limitations of feature-level adversarial defences.

摘要: 深度学习以其高的表现力和速度成为基于网络的入侵检测系统(NIDSS)的检测算法之一。不幸的是，DL算法容易受到敌意示例的攻击，这些示例向输入注入难以察觉的修改，并导致DL算法错误地对输入进行分类。现有的NIDS域中的对抗性攻击往往直接操纵流量特征，由于流量特征不能在真实网络中再现，因此没有实际意义。产生实用的和躲避的对抗性攻击仍然是一个研究挑战。针对物联网网络中部署的基于异常的网络入侵检测系统，提出了一种能够生成实用的、可重放的敌意网络数据包的六二米侯攻击方法。六儿后的核心思想是利用敌意可转移性，在预定义的变异操作约束下的代理网络入侵检测系统上生成对抗性的报文，以确保实用性。我们客观地分析了六儿密侯对四种基于ML的算法(LOF、OCSVM、RRCF和SOM)以及最新的网络入侵检测系统Kitsune的规避能力。从实验结果中，我们对异常检测算法的对抗性转移的必要条件得到了有价值的见解。超越理论设置，我们在真实的IoT试验台上重播对抗性攻击，以检验六儿后的实用性。此外，我们证明了现有的特征级对抗性防御不能抵抗六二密侯，并建设性地批评了特征级对抗性防御的局限性。



## **22. Optimal Membership Inference Bounds for Adaptive Composition of Sampled Gaussian Mechanisms**

采样高斯机构自适应组合的最优成员推理界 cs.CR

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2204.06106v1)

**Authors**: Saeed Mahloujifar, Alexandre Sablayrolles, Graham Cormode, Somesh Jha

**Abstracts**: Given a trained model and a data sample, membership-inference (MI) attacks predict whether the sample was in the model's training set. A common countermeasure against MI attacks is to utilize differential privacy (DP) during model training to mask the presence of individual examples. While this use of DP is a principled approach to limit the efficacy of MI attacks, there is a gap between the bounds provided by DP and the empirical performance of MI attacks. In this paper, we derive bounds for the \textit{advantage} of an adversary mounting a MI attack, and demonstrate tightness for the widely-used Gaussian mechanism. We further show bounds on the \textit{confidence} of MI attacks. Our bounds are much stronger than those obtained by DP analysis. For example, analyzing a setting of DP-SGD with $\epsilon=4$ would obtain an upper bound on the advantage of $\approx0.36$ based on our analyses, while getting bound of $\approx 0.97$ using the analysis of previous work that convert $\epsilon$ to membership inference bounds.   Finally, using our analysis, we provide MI metrics for models trained on CIFAR10 dataset. To the best of our knowledge, our analysis provides the state-of-the-art membership inference bounds for the privacy.

摘要: 在给定训练模型和数据样本的情况下，成员推理(MI)攻击预测样本是否在模型的训练集中。针对MI攻击的一种常见对策是在模型训练期间利用差异隐私(DP)来掩盖单个示例的存在。虽然使用DP是一种原则性的方法来限制MI攻击的有效性，但是DP提供的界限和MI攻击的经验性能之间存在差距。在这篇文章中，我们得到了敌手发起MI攻击的优势的界，并证明了广泛使用的Gauss机制的紧性。我们进一步给出了MI攻击的置信度的界。我们的边界比DP分析得到的边界要强得多。例如，根据我们的分析，分析具有$\epsilon=4$的DP-SGD的设置将得到$\约0.36$的优势的上界，而使用将$\epsilon$转换为成员推理界的前人工作的分析，得到$\约0.97$的上界。最后，利用我们的分析，我们提供了在CIFAR10数据集上训练的模型的MI度量。据我们所知，我们的分析为隐私提供了最先进的成员关系推断界限。



## **23. Membership Inference Attacks From First Principles**

从第一性原理出发的成员推理攻击 cs.CR

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2112.03570v2)

**Authors**: Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramer

**Abstracts**: A membership inference attack allows an adversary to query a trained machine learning model to predict whether or not a particular example was contained in the model's training dataset. These attacks are currently evaluated using average-case "accuracy" metrics that fail to characterize whether the attack can confidently identify any members of the training set. We argue that attacks should instead be evaluated by computing their true-positive rate at low (e.g., <0.1%) false-positive rates, and find most prior attacks perform poorly when evaluated in this way. To address this we develop a Likelihood Ratio Attack (LiRA) that carefully combines multiple ideas from the literature. Our attack is 10x more powerful at low false-positive rates, and also strictly dominates prior attacks on existing metrics.

摘要: 成员关系推理攻击允许对手查询经过训练的机器学习模型，以预测特定示例是否包含在该模型的训练数据集中。目前，这些攻击是使用平均案例“准确性”度量来评估的，该度量无法确定攻击是否可以自信地识别训练集的任何成员。我们认为，应该通过在较低的(例如<0.1%)假阳性率下计算真阳性率来评估攻击，并发现大多数先前的攻击在以这种方式评估时表现很差。为了解决这个问题，我们开发了一种似然比攻击(LIRA)，它仔细地结合了文献中的多种想法。我们的攻击在低假阳性率下的威力要高出10倍，而且还严格控制了之前对现有指标的攻击。



## **24. Rate Coding or Direct Coding: Which One is Better for Accurate, Robust, and Energy-efficient Spiking Neural Networks?**

速率编码和直接编码：哪种编码更适合准确、健壮和节能的尖峰神经网络？ cs.NE

Accepted to ICASSP2022

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2202.03133v2)

**Authors**: Youngeun Kim, Hyoungseob Park, Abhishek Moitra, Abhiroop Bhattacharjee, Yeshwanth Venkatesha, Priyadarshini Panda

**Abstracts**: Recent Spiking Neural Networks (SNNs) works focus on an image classification task, therefore various coding techniques have been proposed to convert an image into temporal binary spikes. Among them, rate coding and direct coding are regarded as prospective candidates for building a practical SNN system as they show state-of-the-art performance on large-scale datasets. Despite their usage, there is little attention to comparing these two coding schemes in a fair manner. In this paper, we conduct a comprehensive analysis of the two codings from three perspectives: accuracy, adversarial robustness, and energy-efficiency. First, we compare the performance of two coding techniques with various architectures and datasets. Then, we measure the robustness of the coding techniques on two adversarial attack methods. Finally, we compare the energy-efficiency of two coding schemes on a digital hardware platform. Our results show that direct coding can achieve better accuracy especially for a small number of timesteps. In contrast, rate coding shows better robustness to adversarial attacks owing to the non-differentiable spike generation process. Rate coding also yields higher energy-efficiency than direct coding which requires multi-bit precision for the first layer. Our study explores the characteristics of two codings, which is an important design consideration for building SNNs. The code is made available at https://github.com/Intelligent-Computing-Lab-Yale/Rate-vs-Direct.

摘要: 最近的尖峰神经网络(SNN)专注于图像分类任务，因此已经提出了各种编码技术来将图像转换为时间二进制尖峰。其中，码率编码和直接编码由于在大规模数据集上表现出最先进的性能，被认为是构建实用SNN系统的潜在候选者。尽管使用了这两种编码方案，但很少有人注意以公平的方式比较这两种编码方案。在本文中，我们从准确性、对手健壮性和能量效率三个角度对这两种编码进行了全面的分析。首先，我们比较了两种编码技术在不同架构和不同数据集下的性能。然后，我们测量了编码技术在两种对抗性攻击方法上的健壮性。最后，在数字硬件平台上对两种编码方案的能量效率进行了比较。我们的结果表明，直接编码可以获得更好的精度，特别是在少量时间步长的情况下。相反，由于不可微的尖峰产生过程，速率编码表现出更好的对对手攻击的稳健性。速率编码还产生比直接编码更高的能量效率，直接编码需要第一层的多比特精度。我们的研究探索了两种编码的特点，这是构建SNN的重要设计考虑因素。代码可在https://github.com/Intelligent-Computing-Lab-Yale/Rate-vs-Direct.上获得



## **25. Masked Faces with Faced Masks**

戴着面具的蒙面人 cs.CV

8 pages

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2201.06427v2)

**Authors**: Jiayi Zhu, Qing Guo, Felix Juefei-Xu, Yihao Huang, Yang Liu, Geguang Pu

**Abstracts**: Modern face recognition systems (FRS) still fall short when the subjects are wearing facial masks, a common theme in the age of respiratory pandemics. An intuitive partial remedy is to add a mask detector to flag any masked faces so that the FRS can act accordingly for those low-confidence masked faces. In this work, we set out to investigate the potential vulnerability of such FRS equipped with a mask detector, on large-scale masked faces, which might trigger a serious risk, e.g., letting a suspect evade the FRS where both facial identity and mask are undetected. As existing face recognizers and mask detectors have high performance in their respective tasks, it is significantly challenging to simultaneously fool them and preserve the transferability of the attack. We formulate the new task as the generation of realistic & adversarial-faced mask and make three main contributions: First, we study the naive Delanunay-based masking method (DM) to simulate the process of wearing a faced mask that is cropped from a template image, which reveals the main challenges of this new task. Second, we further equip the DM with the adversarial noise attack and propose the adversarial noise Delaunay-based masking method (AdvNoise-DM) that can fool the face recognition and mask detection effectively but make the face less natural. Third, we propose the adversarial filtering Delaunay-based masking method denoted as MF2M by employing the adversarial filtering for AdvNoise-DM and obtain more natural faces. With the above efforts, the final version not only leads to significant performance deterioration of the state-of-the-art (SOTA) deep learning-based FRS, but also remains undetected by the SOTA facial mask detector, thus successfully fooling both systems at the same time.

摘要: 当受试者戴着口罩时，现代人脸识别系统(FRS)仍然不足，这在呼吸道大流行的时代是一个常见的主题。一种直观的部分补救方法是添加一个掩模检测器来标记任何掩蔽的人脸，以便FRS可以对这些低置信度的掩蔽人脸采取相应的行动。在这项工作中，我们开始调查这种配备了面具检测器的FRS在大规模蒙面人脸上的潜在脆弱性，这可能会引发严重的风险，例如，让嫌疑人逃避FRS，其中面部身份和面具都没有被检测到。由于现有的人脸识别器和面具检测器在各自的任务中具有很高的性能，因此要同时欺骗它们并保持攻击的可转移性是非常具有挑战性的。首先，我们研究了基于朴素Delanunay的掩蔽方法(DM)来模拟从模板图像中裁剪出来的人脸面具的佩戴过程，揭示了这一新任务的主要挑战。其次，进一步为DM提供了对抗性噪声攻击，并提出了基于对抗性噪声Delaunay的掩蔽方法(AdvNoise-DM)，该方法可以有效地欺骗人脸识别和掩模检测，但会使人脸变得不自然。第三，针对AdvNoise-DM，提出了一种基于对抗过滤的Delaunay掩蔽方法MF2M，得到了更自然的人脸。在上述努力下，最终版本不仅导致基于最先进的(SOTA)深度学习的FRS的性能显著恶化，而且仍然没有被SOTA面膜检测器检测到，从而成功地同时愚弄了两个系统。



## **26. Automated Attacker Synthesis for Distributed Protocols**

分布式协议的自动攻击者综合 cs.CR

24 pages, 15 figures

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2004.01220v4)

**Authors**: Max von Hippel, Cole Vick, Stavros Tripakis, Cristina Nita-Rotaru

**Abstracts**: Distributed protocols should be robust to both benign malfunction (e.g. packet loss or delay) and attacks (e.g. message replay) from internal or external adversaries. In this paper we take a formal approach to the automated synthesis of attackers, i.e. adversarial processes that can cause the protocol to malfunction. Specifically, given a formal threat model capturing the distributed protocol model and network topology, as well as the placement, goals, and interface (inputs and outputs) of potential attackers, we automatically synthesize an attacker. We formalize four attacker synthesis problems - across attackers that always succeed versus those that sometimes fail, and attackers that attack forever versus those that do not - and we propose algorithmic solutions to two of them. We report on a prototype implementation called KORG and its application to TCP as a case-study. Our experiments show that KORG can automatically generate well-known attacks for TCP within seconds or minutes.

摘要: 分布式协议应该对来自内部或外部对手的良性故障(例如，分组丢失或延迟)和攻击(例如，消息重放)具有健壮性。在本文中，我们采用了一种形式化的方法来自动合成攻击者，即可能导致协议故障的对抗性过程。具体地说，给定一个捕获分布式协议模型和网络拓扑以及潜在攻击者的位置、目标和接口(输入和输出)的正式威胁模型，我们将自动合成攻击者。我们将四个攻击者合成问题形式化--总是成功的攻击者与有时失败的攻击者，以及永远攻击的攻击者与不成功的攻击者--并针对其中两个问题提出了算法解决方案。我们报告了一个称为KORG的原型实现以及它在TCP中的应用作为案例研究。我们的实验表明，KORG可以在几秒或几分钟内自动生成针对TCP的知名攻击。



## **27. Catch Me If You Can: Blackbox Adversarial Attacks on Automatic Speech Recognition using Frequency Masking**

抓住我：使用频率掩蔽对自动语音识别进行黑箱对抗性攻击 cs.SD

11 pages, 7 figures and 3 tables

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2112.01821v2)

**Authors**: Xiaoliang Wu, Ajitha Rajan

**Abstracts**: Automatic speech recognition (ASR) models are prevalent, particularly in applications for voice navigation and voice control of domestic appliances. The computational core of ASRs are deep neural networks (DNNs) that have been shown to be susceptible to adversarial perturbations; easily misused by attackers to generate malicious outputs. To help test the security and robustnesss of ASRS, we propose techniques that generate blackbox (agnostic to the DNN), untargeted adversarial attacks that are portable across ASRs. This is in contrast to existing work that focuses on whitebox targeted attacks that are time consuming and lack portability.   Our techniques generate adversarial attacks that have no human audible difference by manipulating the audio signal using a psychoacoustic model that maintains the audio perturbations below the thresholds of human perception. We evaluate portability and effectiveness of our techniques using three popular ASRs and two input audio datasets using the metrics - Word Error Rate (WER) of output transcription, Similarity to original audio, attack Success Rate on different ASRs and Detection score by a defense system. We found our adversarial attacks were portable across ASRs, not easily detected by a state-of-the-art defense system, and had significant difference in output transcriptions while sounding similar to original audio.

摘要: 自动语音识别(ASR)模型非常普遍，特别是在家用电器的语音导航和语音控制应用中。ASR的计算核心是深度神经网络(DNN)，已被证明容易受到对手扰动的影响；很容易被攻击者误用来生成恶意输出。为了帮助测试ASR的安全性和健壮性，我们提出了生成黑盒(与DNN无关)的技术，这是一种可跨ASR移植的无目标对抗性攻击。这与现有的专注于白盒目标攻击的工作形成了鲜明对比，这些攻击既耗时又缺乏可移植性。我们的技术通过使用将音频扰动保持在人类感知阈值以下的心理声学模型来操纵音频信号，从而产生没有人类听觉差异的对抗性攻击。我们使用三个流行的ASR和两个输入音频数据集，使用输出转录的错误率(WER)、与原始音频的相似性、对不同ASR的攻击成功率和防御系统的检测分数来评估我们的技术的可移植性和有效性。我们发现，我们的敌意攻击可以跨ASR进行移植，不容易被最先进的防御系统检测到，而且在输出转录方面有显著差异，但听起来与原始音频相似。



## **28. Staircase Sign Method for Boosting Adversarial Attacks**

一种加强对抗性攻击的阶梯标记法 cs.CV

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2104.09722v2)

**Authors**: Qilong Zhang, Xiaosu Zhu, Jingkuan Song, Lianli Gao, Heng Tao Shen

**Abstracts**: Crafting adversarial examples for the transfer-based attack is challenging and remains a research hot spot. Currently, such attack methods are based on the hypothesis that the substitute model and the victim model learn similar decision boundaries, and they conventionally apply Sign Method (SM) to manipulate the gradient as the resultant perturbation. Although SM is efficient, it only extracts the sign of gradient units but ignores their value difference, which inevitably leads to a deviation. Therefore, we propose a novel Staircase Sign Method (S$^2$M) to alleviate this issue, thus boosting attacks. Technically, our method heuristically divides the gradient sign into several segments according to the values of the gradient units, and then assigns each segment with a staircase weight for better crafting adversarial perturbation. As a result, our adversarial examples perform better in both white-box and black-box manner without being more visible. Since S$^2$M just manipulates the resultant gradient, our method can be generally integrated into the family of FGSM algorithms, and the computational overhead is negligible. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our proposed methods, which significantly improve the transferability (i.e., on average, \textbf{5.1\%} for normally trained models and \textbf{12.8\%} for adversarially trained defenses). Our code is available at \url{https://github.com/qilong-zhang/Staircase-sign-method}.

摘要: 为基于转移的攻击制作敌意例子是具有挑战性的，也是一个研究热点。目前，这种攻击方法是基于替换模型和受害者模型学习相似的决策边界的假设，并且它们通常应用符号方法(SM)来操纵梯度作为结果扰动。虽然SM方法是有效的，但它只提取梯度单元的符号，而忽略了它们的值差异，这不可避免地会导致偏差。因此，我们提出了一种新的阶梯符号方法(S$^2$M)来缓解这个问题，从而增强了攻击。从技术上讲，我们的方法根据梯度单元的值启发式地将梯度符号分成几个段，然后为每个段分配阶梯权重，以便更好地制作对抗扰动。因此，我们的对抗性例子在白盒和黑盒方式下都表现得更好，而不是更明显。由于S$^2$M只是对合成的梯度进行操作，因此我们的方法一般可以集成到FGSM算法家族中，并且计算开销可以忽略不计。在ImageNet数据集上的大量实验表明，我们提出的方法是有效的，显著提高了可转移性(即，对于正常训练的模型，平均为extbf{5.1}，对于经过相反训练的防御，平均为\extbf{12.8})。我们的代码可以在\url{https://github.com/qilong-zhang/Staircase-sign-method}.上找到



## **29. A survey in Adversarial Defences and Robustness in NLP**

自然语言处理中的对抗性防御和稳健性研究综述 cs.CL

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2203.06414v2)

**Authors**: Shreya Goyal, Sumanth Doddapaneni, Mitesh M. Khapra, Balaraman Ravindran

**Abstracts**: In recent years, it has been seen that deep neural networks are lacking robustness and are likely to break in case of adversarial perturbations in input data. Strong adversarial attacks are proposed by various authors for computer vision and Natural Language Processing (NLP). As a counter-effort, several defense mechanisms are also proposed to save these networks from failing. In contrast with image data, generating adversarial attacks and defending these models is not easy in NLP because of the discrete nature of the text data. However, numerous methods for adversarial defense are proposed of late, for different NLP tasks such as text classification, named entity recognition, natural language inferencing, etc. These methods are not just used for defending neural networks from adversarial attacks, but also used as a regularization mechanism during training, saving the model from overfitting. The proposed survey is an attempt to review different methods proposed for adversarial defenses in NLP in the recent past by proposing a novel taxonomy. This survey also highlights the fragility of the advanced deep neural networks in NLP and the challenges in defending them.

摘要: 近年来，人们已经看到，深度神经网络缺乏健壮性，在输入数据发生对抗性扰动的情况下很容易崩溃。强对抗性攻击是计算机视觉和自然语言处理领域的研究热点。作为应对措施，还提出了几种防御机制，以避免这些网络出现故障。与图像数据相比，由于文本数据的离散性，在自然语言处理中生成对抗性攻击并对这些模型进行防御并不容易。然而，最近针对文本分类、命名实体识别、自然语言推理等不同的NLP任务提出了许多对抗性防御方法，这些方法不仅用于保护神经网络免受对抗性攻击，而且在训练过程中作为一种正则化机制，避免了模型的过度拟合。这项拟议的调查试图通过提出一种新的分类法来回顾最近在NLP中提出的不同的对抗性防御方法。这项调查还突显了NLP中先进的深度神经网络的脆弱性以及在保护它们方面的挑战。



## **30. Generalizing Adversarial Explanations with Grad-CAM**

基于Grad-CAM的对抗性解释泛化 cs.CV

Accepted in CVPRw ArtofRobustness workshop

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.05427v1)

**Authors**: Tanmay Chakraborty, Utkarsh Trehan, Khawla Mallat, Jean-Luc Dugelay

**Abstracts**: Gradient-weighted Class Activation Mapping (Grad- CAM), is an example-based explanation method that provides a gradient activation heat map as an explanation for Convolution Neural Network (CNN) models. The drawback of this method is that it cannot be used to generalize CNN behaviour. In this paper, we present a novel method that extends Grad-CAM from example-based explanations to a method for explaining global model behaviour. This is achieved by introducing two new metrics, (i) Mean Observed Dissimilarity (MOD) and (ii) Variation in Dissimilarity (VID), for model generalization. These metrics are computed by comparing a Normalized Inverted Structural Similarity Index (NISSIM) metric of the Grad-CAM generated heatmap for samples from the original test set and samples from the adversarial test set. For our experiment, we study adversarial attacks on deep models such as VGG16, ResNet50, and ResNet101, and wide models such as InceptionNetv3 and XceptionNet using Fast Gradient Sign Method (FGSM). We then compute the metrics MOD and VID for the automatic face recognition (AFR) use case with the VGGFace2 dataset. We observe a consistent shift in the region highlighted in the Grad-CAM heatmap, reflecting its participation to the decision making, across all models under adversarial attacks. The proposed method can be used to understand adversarial attacks and explain the behaviour of black box CNN models for image analysis.

摘要: 梯度加权类激活映射(Grad-CAM)是一种基于实例的解释方法，它提供了一个梯度激活热图作为对卷积神经网络(CNN)模型的解释。这种方法的缺点是，它不能用来概括CNN的行为。在本文中，我们提出了一种新的方法，将Grad-CAM从基于实例的解释扩展为一种解释全局模型行为的方法。这是通过引入两个新的度量来实现的，(I)平均观察相异度(MOD)和(Ii)相异度变化(VID)，用于模型泛化。这些度量是通过比较Grad-CAM为原始测试集中的样本和对手测试集中的样本生成的热图的归一化反向结构相似性指数(Nissim)度量来计算的。在我们的实验中，我们使用快速梯度符号方法(FGSM)研究了对VGG16、ResNet50和ResNet101等深层模型以及InceptionNetv3和XceptionNet等宽模型的敌意攻击。然后，我们使用VGGFace2数据集计算自动人脸识别(AFR)用例的度量MOD和VID。我们观察到，在Grad-CAM热图中突出显示的区域出现了持续的变化，反映了其参与决策的情况，涵盖了所有受到敌对攻击的模型。该方法可用于理解对抗性攻击和解释用于图像分析的黑盒CNN模型的行为。



## **31. Segmentation-Consistent Probabilistic Lesion Counting**

分割一致的概率病变计数 eess.IV

Accepted at Medical Imaging with Deep Learning (MIDL) 2022

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.05276v1)

**Authors**: Julien Schroeter, Chelsea Myers-Colet, Douglas L Arnold, Tal Arbel

**Abstracts**: Lesion counts are important indicators of disease severity, patient prognosis, and treatment efficacy, yet counting as a task in medical imaging is often overlooked in favor of segmentation. This work introduces a novel continuously differentiable function that maps lesion segmentation predictions to lesion count probability distributions in a consistent manner. The proposed end-to-end approach--which consists of voxel clustering, lesion-level voxel probability aggregation, and Poisson-binomial counting--is non-parametric and thus offers a robust and consistent way to augment lesion segmentation models with post hoc counting capabilities. Experiments on Gadolinium-enhancing lesion counting demonstrate that our method outputs accurate and well-calibrated count distributions that capture meaningful uncertainty information. They also reveal that our model is suitable for multi-task learning of lesion segmentation, is efficient in low data regimes, and is robust to adversarial attacks.

摘要: 病灶计数是疾病严重程度、患者预后和治疗效果的重要指标，但在医学成像中，作为一项任务的计数往往被忽视，而有利于分割。这项工作引入了一种新的连续可微函数，它以一致的方式将病变分割预测映射到病变计数概率分布。所提出的端到端方法--包括体素聚类、病变级体素概率聚合和泊松二项计数--是非参数的，因此提供了一种稳健且一致的方法来增强具有后自组织计数能力的病变分割模型。对Gd增强病变计数的实验表明，我们的方法输出准确且校准良好的计数分布，捕捉到有意义的不确定信息。结果还表明，该模型适用于病变分割的多任务学习，在低数据量环境下是有效的，并且对敌意攻击具有较强的鲁棒性。



## **32. Exploring the Universal Vulnerability of Prompt-based Learning Paradigm**

探索基于提示的学习范式的普遍脆弱性 cs.CL

Accepted to Findings of NAACL 2022

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.05239v1)

**Authors**: Lei Xu, Yangyi Chen, Ganqu Cui, Hongcheng Gao, Zhiyuan Liu

**Abstracts**: Prompt-based learning paradigm bridges the gap between pre-training and fine-tuning, and works effectively under the few-shot setting. However, we find that this learning paradigm inherits the vulnerability from the pre-training stage, where model predictions can be misled by inserting certain triggers into the text. In this paper, we explore this universal vulnerability by either injecting backdoor triggers or searching for adversarial triggers on pre-trained language models using only plain text. In both scenarios, we demonstrate that our triggers can totally control or severely decrease the performance of prompt-based models fine-tuned on arbitrary downstream tasks, reflecting the universal vulnerability of the prompt-based learning paradigm. Further experiments show that adversarial triggers have good transferability among language models. We also find conventional fine-tuning models are not vulnerable to adversarial triggers constructed from pre-trained language models. We conclude by proposing a potential solution to mitigate our attack methods. Code and data are publicly available at https://github.com/leix28/prompt-universal-vulnerability

摘要: 基于提示的学习范式在预训练和微调之间架起了一座桥梁，并在少数情况下有效地工作。然而，我们发现这种学习范式继承了预训练阶段的脆弱性，在预训练阶段，通过在文本中插入某些触发器可能会误导模型预测。在本文中，我们通过注入后门触发器或仅使用纯文本在预先训练的语言模型上搜索敌意触发器来探索这一普遍漏洞。在这两种情况下，我们的触发器可以完全控制或严重降低基于提示的模型对任意下游任务进行微调的性能，反映了基于提示的学习范式的普遍脆弱性。进一步的实验表明，对抗性触发词在语言模型之间具有良好的可移植性。我们还发现，传统的微调模型不容易受到从预先训练的语言模型构建的对抗性触发的影响。最后，我们提出了一个潜在的解决方案来减轻我们的攻击方法。代码和数据可在https://github.com/leix28/prompt-universal-vulnerability上公开获得



## **33. Analysis of a blockchain protocol based on LDPC codes**

一种基于LDPC码的区块链协议分析 cs.CR

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2202.07265v2)

**Authors**: Massimo Battaglioni, Paolo Santini, Giulia Rafaiani, Franco Chiaraluce, Marco Baldi

**Abstracts**: In a blockchain Data Availability Attack (DAA), a malicious node publishes a block header but withholds part of the block, which contains invalid transactions. Honest full nodes, which can download and store the full blockchain, are aware that some data are not available but they have no formal way to prove it to light nodes, i.e., nodes that have limited resources and are not able to access the whole blockchain data. A common solution to counter these attacks exploits linear error correcting codes to encode the block content. A recent protocol, called SPAR, employs coded Merkle trees and low-density parity-check (LDPC) codes to counter DAAs. We show that the protocol is less secure than expected, owing to a redefinition of the adversarial success probability.

摘要: 在区块链数据可用性攻击(DAA)中，恶意节点发布块标头，但保留包含无效事务的部分块。可以下载并存储完整区块链的诚实全节点，知道有些数据不可用，但没有正式的方法向轻节点证明，即资源有限、无法访问整个区块链数据的节点。对抗这些攻击的常见解决方案使用线性纠错码来编码块内容。最近的一种称为SPAR的协议使用编码Merkle树和低密度奇偶校验(LDPC)码来对抗DAA。我们表明，由于重新定义了对抗性成功概率，该协议的安全性低于预期。



## **34. Measuring and Mitigating the Risk of IP Reuse on Public Clouds**

衡量和降低公共云上IP重用的风险 cs.CR

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.05122v1)

**Authors**: Eric Pauley, Ryan Sheatsley, Blaine Hoak, Quinn Burke, Yohan Beugin, Patrick McDaniel

**Abstracts**: Public clouds provide scalable and cost-efficient computing through resource sharing. However, moving from traditional on-premises service management to clouds introduces new challenges; failure to correctly provision, maintain, or decommission elastic services can lead to functional failure and vulnerability to attack. In this paper, we explore a broad class of attacks on clouds which we refer to as cloud squatting. In a cloud squatting attack, an adversary allocates resources in the cloud (e.g., IP addresses) and thereafter leverages latent configuration to exploit prior tenants. To measure and categorize cloud squatting we deployed a custom Internet telescope within the Amazon Web Services us-east-1 region. Using this apparatus, we deployed over 3 million servers receiving 1.5 million unique IP addresses (56% of the available pool) over 101 days beginning in March of 2021. We identified 4 classes of cloud services, 7 classes of third-party services, and DNS as sources of exploitable latent configurations. We discovered that exploitable configurations were both common and in many cases extremely dangerous; we received over 5 million cloud messages, many containing sensitive data such as financial transactions, GPS location, and PII. Within the 7 classes of third-party services, we identified dozens of exploitable software systems spanning hundreds of servers (e.g., databases, caches, mobile applications, and web services). Lastly, we identified 5446 exploitable domains spanning 231 eTLDs-including 105 in the top 10,000 and 23 in the top 1000 popular domains. Through tenant disclosures we have identified several root causes, including (a) a lack of organizational controls, (b) poor service hygiene, and (c) failure to follow best practices. We conclude with a discussion of the space of possible mitigations and describe the mitigations to be deployed by Amazon in response to this study.

摘要: 公共云通过资源共享提供可扩展且经济高效的计算。然而，从传统的本地服务管理转移到云带来了新的挑战；未能正确调配、维护或停用弹性服务可能会导致功能故障和易受攻击。在本文中，我们探索了一大类针对云的攻击，我们称之为云蹲攻击。在云蹲守攻击中，对手在云中分配资源(例如，IP地址)，然后利用潜在配置来利用先前的租户。为了测量和分类云蹲点，我们在亚马逊网络服务US-East-1地区部署了一个定制的互联网望远镜。使用此设备，我们部署了300多万台服务器，从2021年3月开始，在101天内接收150万个唯一IP地址(占可用池的56%)。我们确定了4类云服务、7类第三方服务和DNS作为可利用的潜在配置来源。我们发现，可利用的配置很常见，而且在许多情况下极其危险；我们收到了500多万条云消息，其中许多包含金融交易、GPS位置和PII等敏感数据。在7类第三方服务中，我们确定了跨越数百台服务器(例如数据库、缓存、移动应用程序和Web服务)的数十个可利用的软件系统。最后，我们确定了覆盖231个eTLD的5446个可利用域名-其中105个在前10,000个域名中，23个在前1000个热门域名中。通过对租户的披露，我们确定了几个根本原因，包括(A)缺乏组织控制，(B)糟糕的服务卫生，以及(C)未能遵循最佳做法。最后，我们讨论了可能的缓解措施的空间，并描述了Amazon将针对这项研究部署的缓解措施。



## **35. Anti-Adversarially Manipulated Attributions for Weakly Supervised Semantic Segmentation and Object Localization**

用于弱监督语义分割和对象定位的反恶意操纵属性 cs.CV

IEEE TPAMI, 2022

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.04890v1)

**Authors**: Jungbeom Lee, Eunji Kim, Jisoo Mok, Sungroh Yoon

**Abstracts**: Obtaining accurate pixel-level localization from class labels is a crucial process in weakly supervised semantic segmentation and object localization. Attribution maps from a trained classifier are widely used to provide pixel-level localization, but their focus tends to be restricted to a small discriminative region of the target object. An AdvCAM is an attribution map of an image that is manipulated to increase the classification score produced by a classifier before the final softmax or sigmoid layer. This manipulation is realized in an anti-adversarial manner, so that the original image is perturbed along pixel gradients in directions opposite to those used in an adversarial attack. This process enhances non-discriminative yet class-relevant features, which make an insufficient contribution to previous attribution maps, so that the resulting AdvCAM identifies more regions of the target object. In addition, we introduce a new regularization procedure that inhibits the incorrect attribution of regions unrelated to the target object and the excessive concentration of attributions on a small region of the target object. Our method achieves a new state-of-the-art performance in weakly and semi-supervised semantic segmentation, on both the PASCAL VOC 2012 and MS COCO 2014 datasets. In weakly supervised object localization, it achieves a new state-of-the-art performance on the CUB-200-2011 and ImageNet-1K datasets.

摘要: 从类标签中获得准确的像素级定位是弱监督语义分割和目标定位中的关键步骤。来自训练好的分类器的属性图被广泛用于提供像素级定位，但它们的焦点往往被限制在目标对象的一个小的区分区域。AdvCAM是图像的属性图，其被处理以在最终的Softmax或Sigmoid层之前增加由分类器产生的分类分数。这种操作是以反对抗性的方式实现的，使得原始图像沿着与对抗性攻击中使用的方向相反的像素梯度被扰动。该过程增强了对先前属性图贡献不足的非歧视但与类相关的特征，从而所产生的AdvCAM识别目标对象的更多区域。此外，我们引入了一种新的正则化过程，该过程抑制了与目标对象无关的区域的错误归属以及目标对象的小区域属性的过度集中。在PASCAL VOC 2012和MS Coco 2014数据集上，我们的方法在弱监督和半监督语义分割方面取得了新的最先进的性能。在弱监督目标定位方面，它在CUB-200-2011和ImageNet-1K数据集上取得了最新的性能。



## **36. Adversarial Robustness of Deep Sensor Fusion Models**

深度传感器融合模型的对抗稳健性 cs.CV

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2006.13192v3)

**Authors**: Shaojie Wang, Tong Wu, Ayan Chakrabarti, Yevgeniy Vorobeychik

**Abstracts**: We experimentally study the robustness of deep camera-LiDAR fusion architectures for 2D object detection in autonomous driving. First, we find that the fusion model is usually both more accurate, and more robust against single-source attacks than single-sensor deep neural networks. Furthermore, we show that without adversarial training, early fusion is more robust than late fusion, whereas the two perform similarly after adversarial training. However, we note that single-channel adversarial training of deep fusion is often detrimental even to robustness. Moreover, we observe cross-channel externalities, where single-channel adversarial training reduces robustness to attacks on the other channel. Additionally, we observe that the choice of adversarial model in adversarial training is critical: using attacks restricted to cars' bounding boxes is more effective in adversarial training and exhibits less significant cross-channel externalities. Finally, we find that joint-channel adversarial training helps mitigate many of the issues above, but does not significantly boost adversarial robustness.

摘要: 实验研究了深度摄像机-LiDAR融合结构在自动驾驶中检测2D目标的稳健性。首先，我们发现融合模型通常比单传感器深度神经网络更准确，并且对单源攻击具有更强的鲁棒性。此外，我们还表明，在没有对抗性训练的情况下，早期融合比后期融合更稳健，而在对抗性训练后，两者的表现相似。然而，我们注意到，深度融合的单通道对抗性训练往往甚至对健壮性有害。此外，我们观察到了跨通道外部性，其中单通道对抗性训练降低了对另一通道攻击的稳健性。此外，我们观察到，在对抗性训练中选择对抗性模型是至关重要的：在对抗性训练中，使用仅限于汽车包围盒的攻击更有效，并且表现出较少的跨通道外部性。最后，我们发现联合通道对抗性训练有助于缓解上述许多问题，但并不显著提高对抗性健壮性。



## **37. Measuring the False Sense of Security**

测量虚假的安全感 cs.LG

**SubmitDate**: 2022-04-10    [paper-pdf](http://arxiv.org/pdf/2204.04778v1)

**Authors**: Carlos Gomes

**Abstracts**: Recently, several papers have demonstrated how widespread gradient masking is amongst proposed adversarial defenses. Defenses that rely on this phenomenon are considered failed, and can easily be broken. Despite this, there has been little investigation into ways of measuring the phenomenon of gradient masking and enabling comparisons of its extent amongst different networks. In this work, we investigate gradient masking under the lens of its mensurability, departing from the idea that it is a binary phenomenon. We propose and motivate several metrics for it, performing extensive empirical tests on defenses suspected of exhibiting different degrees of gradient masking. These are computationally cheaper than strong attacks, enable comparisons between models, and do not require the large time investment of tailor-made attacks for specific models. Our results reveal metrics that are successful in measuring the extent of gradient masking across different networks

摘要: 最近，几篇论文已经证明了梯度掩蔽在所提出的对抗防御中是如何广泛存在的。依赖这种现象的防御被认为是失败的，很容易被打破。尽管如此，关于如何测量梯度掩蔽现象并能够在不同网络之间比较其程度的研究很少。在这项工作中，我们从梯度掩蔽是一种二元现象的观点出发，研究了它的可测性透镜下的梯度掩蔽。我们为它提出并激励了几个衡量标准，对被怀疑表现出不同程度的梯度掩蔽的防御进行了广泛的经验测试。这些攻击在计算上比强攻击更便宜，可以在模型之间进行比较，并且不需要为特定模型定制攻击的大量时间投资。我们的结果揭示了成功测量不同网络之间的梯度掩蔽程度的度量标准



## **38. Analysis of Power-Oriented Fault Injection Attacks on Spiking Neural Networks**

尖峰神经网络面向能量的故障注入攻击分析 cs.AI

Design, Automation and Test in Europe Conference (DATE) 2022

**SubmitDate**: 2022-04-10    [paper-pdf](http://arxiv.org/pdf/2204.04768v1)

**Authors**: Karthikeyan Nagarajan, Junde Li, Sina Sayyah Ensan, Mohammad Nasim Imtiaz Khan, Sachhidh Kannan, Swaroop Ghosh

**Abstracts**: Spiking Neural Networks (SNN) are quickly gaining traction as a viable alternative to Deep Neural Networks (DNN). In comparison to DNNs, SNNs are more computationally powerful and provide superior energy efficiency. SNNs, while exciting at first appearance, contain security-sensitive assets (e.g., neuron threshold voltage) and vulnerabilities (e.g., sensitivity of classification accuracy to neuron threshold voltage change) that adversaries can exploit. We investigate global fault injection attacks by employing external power supplies and laser-induced local power glitches to corrupt crucial training parameters such as spike amplitude and neuron's membrane threshold potential on SNNs developed using common analog neurons. We also evaluate the impact of power-based attacks on individual SNN layers for 0% (i.e., no attack) to 100% (i.e., whole layer under attack). We investigate the impact of the attacks on digit classification tasks and find that in the worst-case scenario, classification accuracy is reduced by 85.65%. We also propose defenses e.g., a robust current driver design that is immune to power-oriented attacks, improved circuit sizing of neuron components to reduce/recover the adversarial accuracy degradation at the cost of negligible area and 25% power overhead. We also present a dummy neuron-based voltage fault injection detection system with 1% power and area overhead.

摘要: 尖峰神经网络(SNN)作为深度神经网络(DNN)的一种可行的替代方案正在迅速获得发展。与DNN相比，SNN的计算能力更强，并提供更高的能源效率。SNN虽然乍看上去令人兴奋，但包含对安全敏感的资产(例如，神经元阈值电压)和漏洞(例如，分类精度对神经元阈值电压变化的敏感性)，攻击者可以利用这些漏洞。我们通过使用外部电源和激光诱导的局部功率毛刺来破坏使用普通模拟神经元开发的SNN上的关键训练参数，如棘波幅度和神经元的膜阈值电位，来调查全局故障注入攻击。我们还评估了基于能量的攻击对单个SNN层的影响，从0%(即没有攻击)到100%(即整个层受到攻击)。我们研究了攻击对数字分类任务的影响，发现在最坏的情况下，分类准确率下降了85.65%。我们还提出了防御措施，例如，稳健的电流驱动器设计，它不受面向功率的攻击，改进了神经元组件的电路大小，以可忽略的面积和25%的功率开销为代价来减少/恢复对抗性精度的下降。我们还提出了一个基于虚拟神经元的电压故障注入检测系统，该系统具有1%的功率和面积开销。



## **39. "That Is a Suspicious Reaction!": Interpreting Logits Variation to Detect NLP Adversarial Attacks**

“这是一个可疑的反应！”：解读Logits变量以检测NLP对手攻击 cs.AI

ACL 2022

**SubmitDate**: 2022-04-10    [paper-pdf](http://arxiv.org/pdf/2204.04636v1)

**Authors**: Edoardo Mosca, Shreyash Agarwal, Javier Rando-Ramirez, Georg Groh

**Abstracts**: Adversarial attacks are a major challenge faced by current machine learning research. These purposely crafted inputs fool even the most advanced models, precluding their deployment in safety-critical applications. Extensive research in computer vision has been carried to develop reliable defense strategies. However, the same issue remains less explored in natural language processing. Our work presents a model-agnostic detector of adversarial text examples. The approach identifies patterns in the logits of the target classifier when perturbing the input text. The proposed detector improves the current state-of-the-art performance in recognizing adversarial inputs and exhibits strong generalization capabilities across different NLP models, datasets, and word-level attacks.

摘要: 对抗性攻击是当前机器学习研究面临的一大挑战。这些刻意制作的输入甚至欺骗了最先进的型号，使它们无法部署在安全关键应用程序中。为了制定可靠的防御策略，人们在计算机视觉方面进行了广泛的研究。然而，在自然语言处理中，同样的问题仍然被较少地探讨。我们的工作提出了一个模型不可知的对抗性文本例子的检测器。该方法在干扰输入文本时识别目标分类器的逻辑中的模式。提出的检测器提高了当前在识别敌意输入方面的最新性能，并在不同的NLP模型、数据集和词级攻击中显示出强大的泛化能力。



## **40. LTD: Low Temperature Distillation for Robust Adversarial Training**

LTD：低温蒸馏用于强大的对抗性训练 cs.CV

**SubmitDate**: 2022-04-10    [paper-pdf](http://arxiv.org/pdf/2111.02331v2)

**Authors**: Erh-Chung Chen, Che-Rung Lee

**Abstracts**: Adversarial training has been widely used to enhance the robustness of the neural network models against adversarial attacks. However, there still a notable gap between the nature accuracy and the robust accuracy. We found one of the reasons is the commonly used labels, one-hot vectors, hinder the learning process for image recognition. In this paper, we proposed a method, called Low Temperature Distillation (LTD), which is based on the knowledge distillation framework to generate the desired soft labels. Unlike the previous work, LTD uses relatively low temperature in the teacher model, and employs different, but fixed, temperatures for the teacher model and the student model. Moreover, we have investigated the methods to synergize the use of nature data and adversarial ones in LTD. Experimental results show that without extra unlabeled data, the proposed method combined with the previous work can achieve 57.72\% and 30.36\% robust accuracy on CIFAR-10 and CIFAR-100 dataset respectively, which is about 1.21\% improvement of the state-of-the-art methods in average.

摘要: 对抗性训练已被广泛应用于增强神经网络模型对对抗性攻击的鲁棒性。然而，自然精度与稳健精度之间仍存在着显著的差距。我们发现，其中一个原因是常用的标签，一个热点向量，阻碍了图像识别的学习过程。本文提出了一种基于知识蒸馏框架来生成所需软标签的方法，称为低温蒸馏(LTD)。与以前的工作不同，LTD在教师模型中使用相对较低的温度，并为教师模型和学生模型使用不同但固定的温度。此外，我们还研究了在LTD协同使用自然数据和对抗性数据的方法。实验结果表明，在不增加额外未标注数据的情况下，该方法在CIFAR-10和CIFAR-100数据集上分别达到了57.72和30.36的稳健准确率，平均比现有方法提高了1.21倍。



## **41. Understanding, Detecting, and Separating Out-of-Distribution Samples and Adversarial Samples in Text Classification**

文本分类中非分布样本和敌意样本的理解、检测和分离 cs.CL

Preprint. Work in progress

**SubmitDate**: 2022-04-09    [paper-pdf](http://arxiv.org/pdf/2204.04458v1)

**Authors**: Cheng-Han Chiang, Hung-yi Lee

**Abstracts**: In this paper, we study the differences and commonalities between statistically out-of-distribution (OOD) samples and adversarial (Adv) samples, both of which hurting a text classification model's performance. We conduct analyses to compare the two types of anomalies (OOD and Adv samples) with the in-distribution (ID) ones from three aspects: the input features, the hidden representations in each layer of the model, and the output probability distributions of the classifier. We find that OOD samples expose their aberration starting from the first layer, while the abnormalities of Adv samples do not emerge until the deeper layers of the model. We also illustrate that the models' output probabilities for Adv samples tend to be more unconfident. Based on our observations, we propose a simple method to separate ID, OOD, and Adv samples using the hidden representations and output probabilities of the model. On multiple combinations of ID, OOD datasets, and Adv attacks, our proposed method shows exceptional results on distinguishing ID, OOD, and Adv samples.

摘要: 本文研究了统计分布(OOD)样本和对抗性(ADV)样本之间的差异和共同点，这两种样本都影响了文本分类模型的性能。我们从输入特征、模型每一层的隐含表示和分类器的输出概率分布三个方面对两类异常(OOD和ADV样本)和非分布异常(ID)进行了分析比较。我们发现，OOD样本从第一层开始暴露出它们的异常，而ADV样本的异常直到模型的更深层才出现。我们还说明，对于ADV样本，模型的输出概率往往更不可信。基于我们的观察，我们提出了一种简单的方法，利用模型的隐含表示和输出概率来分离ID、OOD和ADV样本。在ID、OOD数据集和ADV攻击的多种组合上，我们提出的方法在区分ID、OOD和ADV样本方面表现出了出色的结果。



## **42. PatchCleanser: Certifiably Robust Defense against Adversarial Patches for Any Image Classifier**

PatchCleanser：针对任何图像分类器的恶意补丁的可靠防御 cs.CV

USENIX Security Symposium 2022; extended technical report

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2108.09135v2)

**Authors**: Chong Xiang, Saeed Mahloujifar, Prateek Mittal

**Abstracts**: The adversarial patch attack against image classification models aims to inject adversarially crafted pixels within a restricted image region (i.e., a patch) for inducing model misclassification. This attack can be realized in the physical world by printing and attaching the patch to the victim object; thus, it imposes a real-world threat to computer vision systems. To counter this threat, we design PatchCleanser as a certifiably robust defense against adversarial patches. In PatchCleanser, we perform two rounds of pixel masking on the input image to neutralize the effect of the adversarial patch. This image-space operation makes PatchCleanser compatible with any state-of-the-art image classifier for achieving high accuracy. Furthermore, we can prove that PatchCleanser will always predict the correct class labels on certain images against any adaptive white-box attacker within our threat model, achieving certified robustness. We extensively evaluate PatchCleanser on the ImageNet, ImageNette, CIFAR-10, CIFAR-100, SVHN, and Flowers-102 datasets and demonstrate that our defense achieves similar clean accuracy as state-of-the-art classification models and also significantly improves certified robustness from prior works. Remarkably, PatchCleanser achieves 83.9% top-1 clean accuracy and 62.1% top-1 certified robust accuracy against a 2%-pixel square patch anywhere on the image for the 1000-class ImageNet dataset.

摘要: 针对图像分类模型的对抗性补丁攻击的目的是在受限的图像区域(即补丁)内注入恶意创建的像素，以导致模型误分类。这种攻击可以通过将补丁打印并附加到受害者对象上在物理世界中实现；因此，它对计算机视觉系统构成了现实世界的威胁。为了应对这种威胁，我们将PatchCleanser设计为针对恶意补丁的可靠可靠防御。在PatchCleanser中，我们对输入图像执行两轮像素掩蔽，以中和对手补丁的影响。这种图像空间操作使PatchCleanser与任何最先进的图像分类器兼容，以实现高精度。此外，我们可以证明PatchCleanser将始终预测特定图像上的正确类别标签，以对抗我们威胁模型中的任何自适应白盒攻击者，从而实现经过验证的健壮性。我们在ImageNet、ImageNette、CIFAR-10、CIFAR-100、SVHN和Flowers-102数据集上对PatchCleanser进行了广泛的评估，并展示了我们的防御实现了与最先进的分类模型类似的干净准确性，并显著提高了先前工作中经过认证的稳健性。值得注意的是，对于1000级ImageNet数据集，PatchCleanser针对图像上任何位置2%像素的正方形补丁实现了83.9%的TOP-1清洁准确率和62.1%的TOP-1认证的稳健准确率。



## **43. Path Defense in Dynamic Defender-Attacker Blotto Games (dDAB) with Limited Information**

有限信息动态防御者-攻击者Blotto博弈(DDAB)中的路径防御 cs.GT

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.04176v1)

**Authors**: Austin K. Chen, Bryce L. Ferguson, Daigo Shishika, Michael Dorothy, Jason R. Marden, George J. Pappas, Vijay Kumar

**Abstracts**: We consider a path guarding problem in dynamic Defender-Attacker Blotto games (dDAB), where a team of robots must defend a path in a graph against adversarial agents. Multi-robot systems are particularly well suited to this application, as recent work has shown the effectiveness of these systems in related areas such as perimeter defense and surveillance. When designing a defender policy that guarantees the defense of a path, information about the adversary and the environment can be helpful and may reduce the number of resources required by the defender to achieve a sufficient level of security. In this work, we characterize the necessary and sufficient number of assets needed to guarantee the defense of a shortest path between two nodes in dDAB games when the defender can only detect assets within $k$-hops of a shortest path. By characterizing the relationship between sensing horizon and required resources, we show that increasing the sensing capability of the defender greatly reduces the number of defender assets needed to defend the path.

摘要: 我们考虑了动态防御者-攻击者Blotto博弈(DDAB)中的路径保护问题，其中一组机器人必须防御图中的一条路径以对抗对手代理。多机器人系统特别适合这一应用，因为最近的研究表明，这些系统在周边防御和监视等相关领域是有效的。在设计保证路径防御的防御方策略时，有关对手和环境的信息可能会有所帮助，并且可以减少防御方实现足够安全级别所需的资源数量。在这项工作中，我们刻画了当防御者只能检测到最短路径$k$-跳内的资产时，保证dDAB博弈中两个节点之间最短路径的防御所需的必要且足够数量的资产。通过描述感知范围和所需资源之间的关系，我们表明，增加防御者的感知能力可以极大地减少防御路径所需的防御者资产的数量。



## **44. DAD: Data-free Adversarial Defense at Test Time**

DAD：测试时的无数据对抗性防御 cs.LG

WACV 2022. Project page: https://sites.google.com/view/dad-wacv22

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.01568v2)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Anirban Chakraborty

**Abstracts**: Deep models are highly susceptible to adversarial attacks. Such attacks are carefully crafted imperceptible noises that can fool the network and can cause severe consequences when deployed. To encounter them, the model requires training data for adversarial training or explicit regularization-based techniques. However, privacy has become an important concern, restricting access to only trained models but not the training data (e.g. biometric data). Also, data curation is expensive and companies may have proprietary rights over it. To handle such situations, we propose a completely novel problem of 'test-time adversarial defense in absence of training data and even their statistics'. We solve it in two stages: a) detection and b) correction of adversarial samples. Our adversarial sample detection framework is initially trained on arbitrary data and is subsequently adapted to the unlabelled test data through unsupervised domain adaptation. We further correct the predictions on detected adversarial samples by transforming them in Fourier domain and obtaining their low frequency component at our proposed suitable radius for model prediction. We demonstrate the efficacy of our proposed technique via extensive experiments against several adversarial attacks and for different model architectures and datasets. For a non-robust Resnet-18 model pre-trained on CIFAR-10, our detection method correctly identifies 91.42% adversaries. Also, we significantly improve the adversarial accuracy from 0% to 37.37% with a minimal drop of 0.02% in clean accuracy on state-of-the-art 'Auto Attack' without having to retrain the model.

摘要: 深度模型非常容易受到对抗性攻击。这类攻击是精心设计的难以察觉的噪音，可以愚弄网络，在部署时可能会造成严重后果。为了遇到它们，该模型需要用于对抗性训练的训练数据或明确的基于正则化的技术。然而，隐私已经成为一个重要的问题，只限制对训练模型的访问，而不限制对训练数据(例如生物特征数据)的访问。此外，数据管理成本高昂，公司可能对其拥有专有权。为了处理这种情况，我们提出了一个全新的问题，即在没有训练数据甚至其统计数据的情况下进行测试时间对抗性防御。我们分两个阶段来解决这个问题：a)对手样本的检测和b)对手样本的校正。我们的对抗性样本检测框架首先在任意数据上进行训练，然后通过无监督的领域自适应来适应未标记的测试数据。通过对检测到的敌意样本进行傅立叶变换，并在我们提出的适合模型预测的半径处获得它们的低频分量，进一步修正了预测。我们通过针对几种对抗性攻击以及针对不同模型架构和数据集的广泛实验，证明了我们所提出的技术的有效性。对于在CIFAR-10上预先训练的非健壮RESNET-18模型，我们的检测方法正确识别了91.42%的对手。此外，在不需要重新训练模型的情况下，我们显著地将对手准确率从0%提高到37.37%，而对最先进的自动攻击的干净准确率最小下降了0.02%。



## **45. Training strategy for a lightweight countermeasure model for automatic speaker verification**

一种轻量级说话人自动确认对策模型的训练策略 cs.SD

ASVspoof2021

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2203.17031v3)

**Authors**: Yen-Lun Liao, Xuanjun Chen, Chung-Che Wang, Jyh-Shing Roger Jang

**Abstracts**: The countermeasure (CM) model is developed to protect Automatic Speaker Verification (ASV) systems from spoof attacks and prevent resulting personal information leakage. Based on practicality and security considerations, the CM model is usually deployed on edge devices, which have more limited computing resources and storage space than cloud-based systems. This work proposes training strategies for a lightweight CM model for ASV, using generalized end-to-end (GE2E) pre-training and adversarial fine-tuning to improve performance, and applying knowledge distillation (KD) to reduce the size of the CM model. In the evaluation phase of the ASVspoof 2021 Logical Access task, the lightweight ResNetSE model reaches min t-DCF 0.2695 and EER 3.54%. Compared to the teacher model, the lightweight student model only uses 22.5% of parameters and 21.1% of multiply and accumulate operands of the teacher model.

摘要: 对策(CM)模型是为了保护自动说话人验证(ASV)系统免受欺骗攻击并防止由此导致的个人信息泄露而开发的。基于实用性和安全性考虑，CM模型通常部署在边缘设备上，与基于云的系统相比，边缘设备的计算资源和存储空间更有限。提出了一种面向ASV的轻量级CM模型的训练策略，使用通用端到端(GE2E)预训练和对抗性微调来提高性能，并应用知识蒸馏(KD)来减小CM模型的规模。在ASVspoof2021逻辑访问任务的评估阶段，轻量级ResNetSE模型达到了最小t-DCF值0.2695和EER3.54%.与教师模型相比，轻量级学生模型仅使用了教师模型22.5%的参数和21.1%的乘加操作数。



## **46. Spinning Language Models: Risks of Propaganda-As-A-Service and Countermeasures**

旋转的语言模型：宣传即服务的风险和对策 cs.CR

IEEE S&P 2022. arXiv admin note: text overlap with arXiv:2107.10443

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2112.05224v2)

**Authors**: Eugene Bagdasaryan, Vitaly Shmatikov

**Abstracts**: We investigate a new threat to neural sequence-to-sequence (seq2seq) models: training-time attacks that cause models to "spin" their outputs so as to support an adversary-chosen sentiment or point of view -- but only when the input contains adversary-chosen trigger words. For example, a spinned summarization model outputs positive summaries of any text that mentions the name of some individual or organization.   Model spinning introduces a "meta-backdoor" into a model. Whereas conventional backdoors cause models to produce incorrect outputs on inputs with the trigger, outputs of spinned models preserve context and maintain standard accuracy metrics, yet also satisfy a meta-task chosen by the adversary.   Model spinning enables propaganda-as-a-service, where propaganda is defined as biased speech. An adversary can create customized language models that produce desired spins for chosen triggers, then deploy these models to generate disinformation (a platform attack), or else inject them into ML training pipelines (a supply-chain attack), transferring malicious functionality to downstream models trained by victims.   To demonstrate the feasibility of model spinning, we develop a new backdooring technique. It stacks an adversarial meta-task onto a seq2seq model, backpropagates the desired meta-task output to points in the word-embedding space we call "pseudo-words," and uses pseudo-words to shift the entire output distribution of the seq2seq model. We evaluate this attack on language generation, summarization, and translation models with different triggers and meta-tasks such as sentiment, toxicity, and entailment. Spinned models largely maintain their accuracy metrics (ROUGE and BLEU) while shifting their outputs to satisfy the adversary's meta-task. We also show that, in the case of a supply-chain attack, the spin functionality transfers to downstream models.

摘要: 我们调查了对神经序列到序列(Seq2seq)模型的一种新威胁：训练时间攻击，该攻击使模型的输出“旋转”以支持对手选择的情绪或观点--但仅当输入包含对手选择的触发词时。例如，旋转摘要模型输出提及某个个人或组织名称的任何文本的正面摘要。模型旋转在模型中引入了“元后门”。传统的后门会导致模型在带有触发器的输入上产生不正确的输出，而旋转模型的输出保留了上下文并保持了标准的准确性度量，但也满足了对手选择的元任务。模型旋转使宣传成为一种服务，其中宣传被定义为有偏见的言论。对手可以创建自定义语言模型，为选定的触发器生成所需的旋转，然后部署这些模型以生成虚假信息(平台攻击)，或者将它们注入ML训练管道(供应链攻击)，将恶意功能转移到受害者训练的下游模型。为了论证模型旋转的可行性，我们开发了一种新的回溯技术。它将一个对抗性元任务堆叠到seq2seq模型上，将所需的元任务输出反向传播到我们称为“伪词”的单词嵌入空间中的点，并使用伪词来移动seq2seq模型的整个输出分布。我们用不同的触发因素和元任务，如情感、毒性和蕴涵来评估这种对语言生成、摘要和翻译模型的攻击。旋转模型在很大程度上保持了它们的精度指标(Rouge和BLEU)，同时改变了它们的输出以满足对手的元任务。我们还表明，在供应链攻击的情况下，自旋功能转移到下游模型。



## **47. Backdoor Attack against NLP models with Robustness-Aware Perturbation defense**

对具有鲁棒性感知扰动防御的NLP模型的后门攻击 cs.CR

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.05758v1)

**Authors**: Shaik Mohammed Maqsood, Viveros Manuela Ceron, Addluri GowthamKrishna

**Abstracts**: Backdoor attack intends to embed hidden backdoor into deep neural networks (DNNs), such that the attacked model performs well on benign samples, whereas its prediction will be maliciously changed if the hidden backdoor is activated by the attacker defined trigger. This threat could happen when the training process is not fully controlled, such as training on third-party data-sets or adopting third-party models. There has been a lot of research and different methods to defend such type of backdoor attacks, one being robustness-aware perturbation-based defense method. This method mainly exploits big gap of robustness between poisoned and clean samples. In our work, we break this defense by controlling the robustness gap between poisoned and clean samples using adversarial training step.

摘要: 后门攻击的目的是将隐藏的后门嵌入到深度神经网络(DNN)中，使得攻击模型在良性样本上表现良好，而如果隐藏的后门被攻击者定义的触发器激活，则其预测将被恶意更改。当培训过程没有得到完全控制时，例如在第三方数据集上进行培训或采用第三方模型时，可能会出现这种威胁。已经有很多研究和不同的方法来防御这种类型的后门攻击，其中一种是基于健壮性感知扰动的防御方法。该方法主要利用了有毒样本和干净样本之间存在的较大的稳健性差距。在我们的工作中，我们通过使用对抗性训练步骤来控制有毒样本和干净样本之间的稳健性差距，从而打破了这种防御。



## **48. Defense against Adversarial Attacks on Hybrid Speech Recognition using Joint Adversarial Fine-tuning with Denoiser**

基于联合对抗性微调和去噪的混合语音识别抗敌意攻击 eess.AS

Submitted to Interspeech 2022

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.03851v1)

**Authors**: Sonal Joshi, Saurabh Kataria, Yiwen Shao, Piotr Zelasko, Jesus Villalba, Sanjeev Khudanpur, Najim Dehak

**Abstracts**: Adversarial attacks are a threat to automatic speech recognition (ASR) systems, and it becomes imperative to propose defenses to protect them. In this paper, we perform experiments to show that K2 conformer hybrid ASR is strongly affected by white-box adversarial attacks. We propose three defenses--denoiser pre-processor, adversarially fine-tuning ASR model, and adversarially fine-tuning joint model of ASR and denoiser. Our evaluation shows denoiser pre-processor (trained on offline adversarial examples) fails to defend against adaptive white-box attacks. However, adversarially fine-tuning the denoiser using a tandem model of denoiser and ASR offers more robustness. We evaluate two variants of this defense--one updating parameters of both models and the second keeping ASR frozen. The joint model offers a mean absolute decrease of 19.3\% ground truth (GT) WER with reference to baseline against fast gradient sign method (FGSM) attacks with different $L_\infty$ norms. The joint model with frozen ASR parameters gives the best defense against projected gradient descent (PGD) with 7 iterations, yielding a mean absolute increase of 22.3\% GT WER with reference to baseline; and against PGD with 500 iterations, yielding a mean absolute decrease of 45.08\% GT WER and an increase of 68.05\% adversarial target WER.

摘要: 敌意攻击是对自动语音识别(ASR)系统的一种威胁，提出防御措施势在必行。在本文中，我们通过实验证明K2一致性混合ASR受到白盒对抗攻击的强烈影响。我们提出了三个防御措施--去噪预处理器、反向微调ASR模型、反向微调ASR和去噪联合模型。我们的评估表明，去噪预处理器(针对离线对手示例进行训练)无法防御自适应白盒攻击。然而，相反地，使用去噪器和ASR的串联模型来微调去噪器可提供更强的稳健性。我们评估了这种防御的两种变体--一种是更新两个模型的参数，另一种是保持ASR不变。该联合模型对于具有不同$L_inty$范数的快速梯度符号法(FGSM)攻击，相对于基线平均绝对减少了19.3%的地面真实(GT)WER。采用冻结ASR参数的联合模型对7次迭代的投影梯度下降(PGD)提供了最好的防御，相对于基线平均绝对增加了2 2.3 GT WER，对5 0 0次的PGD给出了最好的防御，平均绝对减少4 5.0 8 GT WER，增加了68.0 5个目标WER。



## **49. AdvEst: Adversarial Perturbation Estimation to Classify and Detect Adversarial Attacks against Speaker Identification**

AdvEst：对抗性扰动估计分类检测针对说话人识别的对抗性攻击 eess.AS

Submitted to InterSpeech 2022

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.03848v1)

**Authors**: Sonal Joshi, Saurabh Kataria, Jesus Villalba, Najim Dehak

**Abstracts**: Adversarial attacks pose a severe security threat to the state-of-the-art speaker identification systems, thereby making it vital to propose countermeasures against them. Building on our previous work that used representation learning to classify and detect adversarial attacks, we propose an improvement to it using AdvEst, a method to estimate adversarial perturbation. First, we prove our claim that training the representation learning network using adversarial perturbations as opposed to adversarial examples (consisting of the combination of clean signal and adversarial perturbation) is beneficial because it eliminates nuisance information. At inference time, we use a time-domain denoiser to estimate the adversarial perturbations from adversarial examples. Using our improved representation learning approach to obtain attack embeddings (signatures), we evaluate their performance for three applications: known attack classification, attack verification, and unknown attack detection. We show that common attacks in the literature (Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), Carlini-Wagner (CW) with different Lp threat models) can be classified with an accuracy of ~96%. We also detect unknown attacks with an equal error rate (EER) of ~9%, which is absolute improvement of ~12% from our previous work.

摘要: 对抗性攻击对最先进的说话人识别系统构成了严重的安全威胁，因此提出针对它们的对策是至关重要的。在利用表征学习对敌方攻击进行分类和检测的基础上，我们提出了一种基于AdvEst的改进方法，该方法是一种估计对抗性扰动的方法。首先，我们证明了我们的主张，即使用对抗性扰动而不是对抗性示例(由干净的信号和对抗性扰动的组合组成)来训练表示学习网络是有益的，因为它消除了滋扰信息。在推理时，我们使用一个时间域去噪器来估计对抗性样本中的对抗性扰动。使用改进的表示学习方法获得攻击嵌入(签名)，我们评估了它们在三个应用中的性能：已知攻击分类、攻击验证和未知攻击检测。我们表明，文献中常见的攻击(快速梯度符号法(FGSM)、投影梯度下降法(PGD)、Carlini-Wagner(CW)和不同的LP威胁模型)可以被分类，准确率为96%。我们还检测未知攻击，等错误率(EER)为~9%，比我们以前的工作绝对提高了~12%。



## **50. Using Multiple Self-Supervised Tasks Improves Model Robustness**

使用多个自监督任务可提高模型的稳健性 cs.CV

Accepted to ICLR 2022 Workshop on PAIR^2Struct: Privacy,  Accountability, Interpretability, Robustness, Reasoning on Structured Data

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2204.03714v1)

**Authors**: Matthew Lawhon, Chengzhi Mao, Junfeng Yang

**Abstracts**: Deep networks achieve state-of-the-art performance on computer vision tasks, yet they fail under adversarial attacks that are imperceptible to humans. In this paper, we propose a novel defense that can dynamically adapt the input using the intrinsic structure from multiple self-supervised tasks. By simultaneously using many self-supervised tasks, our defense avoids over-fitting the adapted image to one specific self-supervised task and restores more intrinsic structure in the image compared to a single self-supervised task approach. Our approach further improves robustness and clean accuracy significantly compared to the state-of-the-art single task self-supervised defense. Our work is the first to connect multiple self-supervised tasks to robustness, and suggests that we can achieve better robustness with more intrinsic signal from visual data.

摘要: 深度网络在计算机视觉任务中实现了最先进的性能，但它们在人类无法察觉的敌意攻击下失败了。在本文中，我们提出了一种新的防御机制，它可以利用多个自监督任务的内在结构来动态调整输入。通过同时使用多个自监督任务，我们的防御方法避免了将适应的图像过度匹配到一个特定的自监督任务，并且与单一的自监督任务方法相比，恢复了图像中更多的内在结构。与最先进的单任务自我监督防御相比，我们的方法进一步提高了健壮性和干净的准确性。我们的工作首次将多个自监督任务与稳健性联系起来，并表明我们可以通过从视觉数据中获得更多的内在信号来获得更好的稳健性。



