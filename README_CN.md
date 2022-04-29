# Latest Adversarial Attack Papers
**update at 2022-04-30 06:31:27**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. UNBUS: Uncertainty-aware Deep Botnet Detection System in Presence of Perturbed Samples**

UNBUS：存在扰动样本的不确定性感知深度僵尸网络检测系统 cs.CR

8 pages, 5 figures, 5 Tables

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.09502v2)

**Authors**: Rahim Taheri

**Abstracts**: A rising number of botnet families have been successfully detected using deep learning architectures. While the variety of attacks increases, these architectures should become more robust against attacks. They have been proven to be very sensitive to small but well constructed perturbations in the input. Botnet detection requires extremely low false-positive rates (FPR), which are not commonly attainable in contemporary deep learning. Attackers try to increase the FPRs by making poisoned samples. The majority of recent research has focused on the use of model loss functions to build adversarial examples and robust models. In this paper, two LSTM-based classification algorithms for botnet classification with an accuracy higher than 98% are presented. Then, the adversarial attack is proposed, which reduces the accuracy to about 30%. Then, by examining the methods for computing the uncertainty, the defense method is proposed to increase the accuracy to about 70%. By using the deep ensemble and stochastic weight averaging quantification methods it has been investigated the uncertainty of the accuracy in the proposed methods.

摘要: 使用深度学习体系结构已成功检测到越来越多的僵尸网络家族。随着攻击种类的增加，这些体系结构应该变得更强大，以抵御攻击。事实证明，它们对输入中的微小但构造良好的扰动非常敏感。僵尸网络检测需要极低的假阳性率(FPR)，这在当代深度学习中是不常见的。攻击者试图通过制作有毒样本来增加FPR。最近的大多数研究都集中在使用模型损失函数来构建对抗性例子和稳健模型。本文提出了两种基于LSTM的僵尸网络分类算法，分类正确率高于98%。然后，提出了对抗性攻击，使准确率降低到30%左右。然后，通过研究不确定度的计算方法，提出了将准确度提高到70%左右的防御方法。通过使用深度集成和随机加权平均量化方法，对所提出方法的精度的不确定度进行了研究。



## **2. Deepfake Forensics via An Adversarial Game**

通过对抗性游戏进行深度假冒取证 cs.CV

Accepted by IEEE Transactions on Image Processing; 13 pages, 4  figures

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2103.13567v2)

**Authors**: Zhi Wang, Yiwen Guo, Wangmeng Zuo

**Abstracts**: With the progress in AI-based facial forgery (i.e., deepfake), people are increasingly concerned about its abuse. Albeit effort has been made for training classification (also known as deepfake detection) models to recognize such forgeries, existing models suffer from poor generalization to unseen forgery technologies and high sensitivity to changes in image/video quality. In this paper, we advocate adversarial training for improving the generalization ability to both unseen facial forgeries and unseen image/video qualities. We believe training with samples that are adversarially crafted to attack the classification models improves the generalization ability considerably. Considering that AI-based face manipulation often leads to high-frequency artifacts that can be easily spotted by models yet difficult to generalize, we further propose a new adversarial training method that attempts to blur out these specific artifacts, by introducing pixel-wise Gaussian blurring models. With adversarial training, the classification models are forced to learn more discriminative and generalizable features, and the effectiveness of our method can be verified by plenty of empirical evidence. Our code will be made publicly available.

摘要: 随着基于人工智能的人脸伪造(即深度假)的发展，人们越来越关注它的滥用。尽管已经努力训练分类(也称为深度伪检测)模型来识别此类伪造物，但现有模型对不可见的伪造物技术的泛化能力差，并且对图像/视频质量的变化高度敏感。在本文中，我们提倡对抗性训练，以提高对看不见的人脸伪造和看不见的图像/视频质量的泛化能力。我们相信，用恶意设计的样本来攻击分类模型的训练大大提高了泛化能力。考虑到基于人工智能的人脸操作往往会导致高频伪影，这些伪影很容易被模型发现，但很难推广，我们进一步提出了一种新的对抗性训练方法，试图通过引入像素级的高斯模糊模型来模糊这些特定的伪影。通过对抗性训练，迫使分类模型学习更具区分性和泛化能力的特征，并通过大量的经验证据验证了该方法的有效性。我们的代码将公开可用。



## **3. Adversarial Fine-tune with Dynamically Regulated Adversary**

动态调整对手的对抗性微调 cs.LG

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.13232v1)

**Authors**: Pengyue Hou, Ming Zhou, Jie Han, Petr Musilek, Xingyu Li

**Abstracts**: Adversarial training is an effective method to boost model robustness to malicious, adversarial attacks. However, such improvement in model robustness often leads to a significant sacrifice of standard performance on clean images. In many real-world applications such as health diagnosis and autonomous surgical robotics, the standard performance is more valued over model robustness against such extremely malicious attacks. This leads to the question: To what extent we can boost model robustness without sacrificing standard performance? This work tackles this problem and proposes a simple yet effective transfer learning-based adversarial training strategy that disentangles the negative effects of adversarial samples on model's standard performance. In addition, we introduce a training-friendly adversarial attack algorithm, which facilitates the boost of adversarial robustness without introducing significant training complexity. Extensive experimentation indicates that the proposed method outperforms previous adversarial training algorithms towards the target: to improve model robustness while preserving model's standard performance on clean data.

摘要: 对抗性训练是提高模型对恶意、对抗性攻击稳健性的有效方法。然而，这种模型稳健性的改进经常导致在干净图像上的标准性能的显著牺牲。在许多真实世界的应用中，例如健康诊断和自主手术机器人，对于这种极端恶意的攻击，标准性能比模型健壮性更受重视。这就引出了一个问题：在不牺牲标准性能的情况下，我们可以在多大程度上提高模型的健壮性？针对这一问题，提出了一种简单而有效的基于迁移学习的对抗性训练策略，消除了对抗性样本对模型标准性能的负面影响。此外，我们还引入了一种训练友好的对抗性攻击算法，该算法在不引入显著训练复杂度的情况下，有助于提高对抗性攻击的健壮性。大量实验表明，该方法优于以往对抗性训练算法的目标：在保持模型在干净数据上的标准性能的同时，提高模型的稳健性。



## **4. An Adversarial Attack Analysis on Malicious Advertisement URL Detection Framework**

恶意广告URL检测框架的对抗性攻击分析 cs.LG

13

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.13172v1)

**Authors**: Ehsan Nowroozi, Abhishek, Mohammadreza Mohammadi, Mauro Conti

**Abstracts**: Malicious advertisement URLs pose a security risk since they are the source of cyber-attacks, and the need to address this issue is growing in both industry and academia. Generally, the attacker delivers an attack vector to the user by means of an email, an advertisement link or any other means of communication and directs them to a malicious website to steal sensitive information and to defraud them. Existing malicious URL detection techniques are limited and to handle unseen features as well as generalize to test data. In this study, we extract a novel set of lexical and web-scrapped features and employ machine learning technique to set up system for fraudulent advertisement URLs detection. The combination set of six different kinds of features precisely overcome the obfuscation in fraudulent URL classification. Based on different statistical properties, we use twelve different formatted datasets for detection, prediction and classification task. We extend our prediction analysis for mismatched and unlabelled datasets. For this framework, we analyze the performance of four machine learning techniques: Random Forest, Gradient Boost, XGBoost and AdaBoost in the detection part. With our proposed method, we can achieve a false negative rate as low as 0.0037 while maintaining high accuracy of 99.63%. Moreover, we devise a novel unsupervised technique for data clustering using K- Means algorithm for the visual analysis. This paper analyses the vulnerability of decision tree-based models using the limited knowledge attack scenario. We considered the exploratory attack and implemented Zeroth Order Optimization adversarial attack on the detection models.

摘要: 恶意广告URL构成了安全风险，因为它们是网络攻击的来源，而且在工业界和学术界，解决这一问题的需求都在不断增长。通常，攻击者通过电子邮件、广告链接或任何其他通信方式向用户发送攻击矢量，并将他们定向到恶意网站，以窃取敏感信息并诈骗他们。现有的恶意URL检测技术在处理看不见的功能以及泛化测试数据方面都是有限的。在这项研究中，我们提取了一组新颖的词汇和网页废弃特征，并利用机器学习技术建立了欺诈性广告URL检测系统。六种不同特征的组合集合恰好克服了欺诈性URL分类中的混淆。基于不同的统计特性，我们使用了12个不同格式的数据集进行检测、预测和分类任务。我们将我们的预测分析扩展到不匹配和未标记的数据集。在检测部分，分析了四种机器学习技术：随机森林、梯度增强、XGBoost和AdaBoost的性能。该方法在保持99.63%的准确率的同时，假阴性率可低至0.0037。此外，我们设计了一种新的无监督数据聚类技术，使用K-Means算法进行可视化分析。分析了基于决策树的模型在有限知识攻击场景下的脆弱性。考虑了探索性攻击，在检测模型上实现了零阶优化对抗性攻击。



## **5. SSR-GNNs: Stroke-based Sketch Representation with Graph Neural Networks**

SSR-GNNS：基于图形神经网络的笔画表示 cs.CV

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.13153v1)

**Authors**: Sheng Cheng, Yi Ren, Yezhou Yang

**Abstracts**: This paper follows cognitive studies to investigate a graph representation for sketches, where the information of strokes, i.e., parts of a sketch, are encoded on vertices and information of inter-stroke on edges. The resultant graph representation facilitates the training of a Graph Neural Networks for classification tasks, and achieves accuracy and robustness comparable to the state-of-the-art against translation and rotation attacks, as well as stronger attacks on graph vertices and topologies, i.e., modifications and addition of strokes, all without resorting to adversarial training. Prior studies on sketches, e.g., graph transformers, encode control points of stroke on vertices, which are not invariant to spatial transformations. In contrary, we encode vertices and edges using pairwise distances among control points to achieve invariance. Compared with existing generative sketch model for one-shot classification, our method does not rely on run-time statistical inference. Lastly, the proposed representation enables generation of novel sketches that are structurally similar to while separable from the existing dataset.

摘要: 在认知研究的基础上，对素描的图形表示进行了研究，其中笔画的信息，即草图的部分，在顶点上编码，边上的笔画间的信息编码。所得到的图表示促进了图神经网络的分类任务的训练，并且获得了与最新技术相媲美的针对平移和旋转攻击的准确性和稳健性，以及对图顶点和拓扑的更强攻击，即修改和添加笔划，所有这些都不求助于对抗性训练。以往对草图的研究，例如图形转换器，对顶点上的笔划控制点进行编码，而这些控制点并不是空间变换的不变性。相反，我们使用控制点之间的成对距离对顶点和边进行编码，以实现不变性。与现有的一次分类生成式草图模型相比，该方法不依赖于运行时的统计推理。最后，所提出的表示法能够生成在结构上与现有数据集相似但可与现有数据集分开的新草图。



## **6. Defending Against Person Hiding Adversarial Patch Attack with a Universal White Frame**

用通用白框防御隐藏敌方补丁攻击的人 cs.CV

Submitted by NeurIPS 2021 with response letter to the anonymous  reviewers' comments

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.13004v1)

**Authors**: Youngjoon Yu, Hong Joo Lee, Hakmin Lee, Yong Man Ro

**Abstracts**: Object detection has attracted great attention in the computer vision area and has emerged as an indispensable component in many vision systems. In the era of deep learning, many high-performance object detection networks have been proposed. Although these detection networks show high performance, they are vulnerable to adversarial patch attacks. Changing the pixels in a restricted region can easily fool the detection network in the physical world. In particular, person-hiding attacks are emerging as a serious problem in many safety-critical applications such as autonomous driving and surveillance systems. Although it is necessary to defend against an adversarial patch attack, very few efforts have been dedicated to defending against person-hiding attacks. To tackle the problem, in this paper, we propose a novel defense strategy that mitigates a person-hiding attack by optimizing defense patterns, while previous methods optimize the model. In the proposed method, a frame-shaped pattern called a 'universal white frame' (UWF) is optimized and placed on the outside of the image. To defend against adversarial patch attacks, UWF should have three properties (i) suppressing the effect of the adversarial patch, (ii) maintaining its original prediction, and (iii) applicable regardless of images. To satisfy the aforementioned properties, we propose a novel pattern optimization algorithm that can defend against the adversarial patch. Through comprehensive experiments, we demonstrate that the proposed method effectively defends against the adversarial patch attack.

摘要: 目标检测在计算机视觉领域引起了极大的关注，已经成为许多视觉系统中不可或缺的组成部分。在深度学习时代，已经提出了许多高性能的目标检测网络。虽然这些检测网络表现出高性能，但它们很容易受到对抗性补丁攻击。更改受限区域中的像素可以很容易地欺骗物理世界中的检测网络。特别是，在自动驾驶和监控系统等许多安全关键应用中，藏人攻击正在成为一个严重的问题。尽管防御对抗性补丁攻击是必要的，但很少有人致力于防御人员隐藏攻击。针对这一问题，本文提出了一种新的防御策略，该策略通过优化防御模式来缓解人员躲藏攻击，而以往的方法则对该模型进行了优化。在所提出的方法中，一个被称为“通用白框”(UWF)的框形图案被优化并放置在图像的外部。为了防御对抗性补丁攻击，UWF应该具有三个性质(I)抑制对抗性补丁的影响，(Ii)保持其原始预测，以及(Iii)适用于任何图像。为了满足上述性质，我们提出了一种新的模式优化算法，该算法能够防御恶意补丁。通过综合实验，我们证明了该方法能够有效地防御敌意补丁攻击。



## **7. The MeVer DeepFake Detection Service: Lessons Learnt from Developing and Deploying in the Wild**

Mever DeepFake检测服务：从野外开发和部署中吸取的教训 cs.CV

10 pages, 6 figures

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.12816v1)

**Authors**: Spyridon Baxevanakis, Giorgos Kordopatis-Zilos, Panagiotis Galopoulos, Lazaros Apostolidis, Killian Levacher, Ipek B. Schlicht, Denis Teyssou, Ioannis Kompatsiaris, Symeon Papadopoulos

**Abstracts**: Enabled by recent improvements in generation methodologies, DeepFakes have become mainstream due to their increasingly better visual quality, the increase in easy-to-use generation tools and the rapid dissemination through social media. This fact poses a severe threat to our societies with the potential to erode social cohesion and influence our democracies. To mitigate the threat, numerous DeepFake detection schemes have been introduced in the literature but very few provide a web service that can be used in the wild. In this paper, we introduce the MeVer DeepFake detection service, a web service detecting deep learning manipulations in images and video. We present the design and implementation of the proposed processing pipeline that involves a model ensemble scheme, and we endow the service with a model card for transparency. Experimental results show that our service performs robustly on the three benchmark datasets while being vulnerable to Adversarial Attacks. Finally, we outline our experience and lessons learned when deploying a research system into production in the hopes that it will be useful to other academic and industry teams.

摘要: 由于最近在生成方法上的改进，DeepFake已经成为主流，因为它们的视觉质量越来越好，易于使用的生成工具的增加，以及通过社交媒体的快速传播。这一事实对我们的社会构成严重威胁，有可能侵蚀社会凝聚力并影响我们的民主国家。为了减轻威胁，文献中已经引入了许多DeepFake检测方案，但很少提供可以在野外使用的Web服务。在本文中，我们介绍了Mever DeepFake检测服务，这是一个检测图像和视频中的深度学习操作的Web服务。我们给出了所提出的处理流水线的设计和实现，该流水线涉及模型集成方案，并且我们为服务赋予模型卡以实现透明性。实验结果表明，我们的服务在三个基准数据集上表现出很好的性能，但很容易受到对手攻击。最后，我们概述了我们在将研究系统部署到生产中时的经验和教训，希望它对其他学术和行业团队有用。



## **8. Improving the Transferability of Adversarial Examples with Restructure Embedded Patches**

利用重构嵌入补丁提高对抗性实例的可转移性 cs.CV

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.12680v1)

**Authors**: Huipeng Zhou, Yu-an Tan, Yajie Wang, Haoran Lyu, Shangbo Wu, Yuanzhang Li

**Abstracts**: Vision transformers (ViTs) have demonstrated impressive performance in various computer vision tasks. However, the adversarial examples generated by ViTs are challenging to transfer to other networks with different structures. Recent attack methods do not consider the specificity of ViTs architecture and self-attention mechanism, which leads to poor transferability of the generated adversarial samples by ViTs. We attack the unique self-attention mechanism in ViTs by restructuring the embedded patches of the input. The restructured embedded patches enable the self-attention mechanism to obtain more diverse patches connections and help ViTs keep regions of interest on the object. Therefore, we propose an attack method against the unique self-attention mechanism in ViTs, called Self-Attention Patches Restructure (SAPR). Our method is simple to implement yet efficient and applicable to any self-attention based network and gradient transferability-based attack methods. We evaluate attack transferability on black-box models with different structures. The result show that our method generates adversarial examples on white-box ViTs with higher transferability and higher image quality. Our research advances the development of black-box transfer attacks on ViTs and demonstrates the feasibility of using white-box ViTs to attack other black-box models.

摘要: 视觉转换器(VITS)在各种计算机视觉任务中表现出令人印象深刻的性能。然而，VITS生成的对抗性例子很难转移到其他具有不同结构的网络上。现有的攻击方法没有考虑VITS体系结构和自我注意机制的特殊性，导致VITS生成的攻击样本可移植性较差。我们通过重组输入的嵌入补丁来攻击VITS中独特的自我注意机制。重构后的嵌入贴片使自我注意机制能够获得更多样化的贴片连接，并帮助VITS保持对象上的感兴趣区域。因此，我们提出了一种针对VITS中独特的自我注意机制的攻击方法，称为自我注意补丁重构(SAPR)。该方法实现简单，效率高，适用于任何基于自我注意的网络攻击方法和基于梯度转移的攻击方法。我们在不同结构的黑盒模型上评估了攻击的可转移性。实验结果表明，该方法在白盒VITS上生成的对抗性样本具有较高的可移植性和较高的图像质量。我们的研究推动了针对VITS的黑盒传输攻击的发展，并论证了利用白盒VITS攻击其他黑盒模型的可行性。



## **9. Data Bootstrapping Approaches to Improve Low Resource Abusive Language Detection for Indic Languages**

改进印度语低资源滥用语言检测的数据自举方法 cs.CL

Accepted at HT '22: 33rd ACM Conference on Hypertext and Social Media

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12543v1)

**Authors**: Mithun Das, Somnath Banerjee, Animesh Mukherjee

**Abstracts**: Abusive language is a growing concern in many social media platforms. Repeated exposure to abusive speech has created physiological effects on the target users. Thus, the problem of abusive language should be addressed in all forms for online peace and safety. While extensive research exists in abusive speech detection, most studies focus on English. Recently, many smearing incidents have occurred in India, which provoked diverse forms of abusive speech in online space in various languages based on the geographic location. Therefore it is essential to deal with such malicious content. In this paper, to bridge the gap, we demonstrate a large-scale analysis of multilingual abusive speech in Indic languages. We examine different interlingual transfer mechanisms and observe the performance of various multilingual models for abusive speech detection for eight different Indic languages. We also experiment to show how robust these models are on adversarial attacks. Finally, we conduct an in-depth error analysis by looking into the models' misclassified posts across various settings. We have made our code and models public for other researchers.

摘要: 在许多社交媒体平台上，辱骂语言日益受到关注。反复暴露在辱骂言语中对目标用户造成了生理影响。因此，为了网络和平与安全，应该以各种形式解决辱骂语言的问题。虽然在辱骂语音检测方面已经有了广泛的研究，但大多数研究都集中在英语上。最近，印度发生了多起诽谤事件，引发了基于地理位置的各种语言在网络空间发表不同形式的辱骂言论。因此，对此类恶意内容的处理至关重要。为了弥补这一差距，我们对印度语中的多语种辱骂言语进行了大规模的分析。我们考察了不同的语际迁移机制，并观察了针对八种不同印度语的滥用语音检测的各种多语言模型的性能。我们还进行了实验，以表明这些模型在对抗攻击时的健壮性。最后，我们通过查看模型在不同环境下错误分类的帖子进行了深入的错误分析。我们已经向其他研究人员公开了我们的代码和模型。



## **10. Restricted Black-box Adversarial Attack Against DeepFake Face Swapping**

针对DeepFake脸部交换的受限黑盒对抗性攻击 cs.CV

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12347v1)

**Authors**: Junhao Dong, Yuan Wang, Jianhuang Lai, Xiaohua Xie

**Abstracts**: DeepFake face swapping presents a significant threat to online security and social media, which can replace the source face in an arbitrary photo/video with the target face of an entirely different person. In order to prevent this fraud, some researchers have begun to study the adversarial methods against DeepFake or face manipulation. However, existing works focus on the white-box setting or the black-box setting driven by abundant queries, which severely limits the practical application of these methods. To tackle this problem, we introduce a practical adversarial attack that does not require any queries to the facial image forgery model. Our method is built on a substitute model persuing for face reconstruction and then transfers adversarial examples from the substitute model directly to inaccessible black-box DeepFake models. Specially, we propose the Transferable Cycle Adversary Generative Adversarial Network (TCA-GAN) to construct the adversarial perturbation for disrupting unknown DeepFake systems. We also present a novel post-regularization module for enhancing the transferability of generated adversarial examples. To comprehensively measure the effectiveness of our approaches, we construct a challenging benchmark of DeepFake adversarial attacks for future development. Extensive experiments impressively show that the proposed adversarial attack method makes the visual quality of DeepFake face images plummet so that they are easier to be detected by humans and algorithms. Moreover, we demonstrate that the proposed algorithm can be generalized to offer face image protection against various face translation methods.

摘要: DeepFake人脸交换对在线安全和社交媒体构成了重大威胁，可以将任意照片/视频中的源脸替换为完全不同的人的目标脸。为了防止这种欺诈，一些研究人员已经开始研究对抗DeepFake或Face操纵的方法。然而，现有的工作主要集中在白盒设置或大量查询驱动的黑盒设置上，这严重限制了这些方法的实际应用。为了解决这个问题，我们引入了一种实用的对抗性攻击，该攻击不需要对人脸图像伪造模型进行任何查询。我们的方法建立在一个试图进行人脸重建的替代模型上，然后将敌对样本从替代模型直接转移到不可访问的黑盒DeepFake模型上。特别地，我们提出了可转移循环对抗性生成对抗性网络(TCA-GAN)来构造针对未知DeepFake系统的对抗性扰动。我们还提出了一种新的后正则化模块来增强生成的对抗性实例的可转移性。为了全面衡量我们的方法的有效性，我们为未来的发展构建了一个具有挑战性的DeepFake对抗性攻击基准。大量实验表明，所提出的对抗性攻击方法使得DeepFake人脸图像的视觉质量直线下降，更容易被人类和算法检测到。此外，我们还证明了所提出的算法可以推广到针对各种人脸转换方法提供人脸图像保护。



## **11. Boosting Adversarial Transferability of MLP-Mixer**

提高MLP-Mixer的对抗转移性 cs.CV

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12204v1)

**Authors**: Haoran Lyu, Yajie Wang, Yu-an Tan, Huipeng Zhou, Yuhang Zhao, Quanxin Zhang

**Abstracts**: The security of models based on new architectures such as MLP-Mixer and ViTs needs to be studied urgently. However, most of the current researches are mainly aimed at the adversarial attack against ViTs, and there is still relatively little adversarial work on MLP-mixer. We propose an adversarial attack method against MLP-Mixer called Maxwell's demon Attack (MA). MA breaks the channel-mixing and token-mixing mechanism of MLP-Mixer by controlling the part input of MLP-Mixer's each Mixer layer, and disturbs MLP-Mixer to obtain the main information of images. Our method can mask the part input of the Mixer layer, avoid overfitting of the adversarial examples to the source model, and improve the transferability of cross-architecture. Extensive experimental evaluation demonstrates the effectiveness and superior performance of the proposed MA. Our method can be easily combined with existing methods and can improve the transferability by up to 38.0% on MLP-based ResMLP. Adversarial examples produced by our method on MLP-Mixer are able to exceed the transferability of adversarial examples produced using DenseNet against CNNs. To the best of our knowledge, we are the first work to study adversarial transferability of MLP-Mixer.

摘要: 基于MLP-Mixer和VITS等新型体系结构的模型的安全性亟待研究。然而，目前的研究大多是针对VITS的对抗性攻击，针对MLP-Mixer的对抗性研究还相对较少。提出了一种针对MLP-Mixer的对抗性攻击方法，称为麦克斯韦恶魔攻击(MA)。MA通过控制MLP-Mixer各混合层的部分输入，打破了MLP-Mixer的通道混合和令牌混合机制，干扰MLP-Mixer获取图像的主要信息。该方法屏蔽了混合层的部分输入，避免了对抗性实例对源模型的过度拟合，提高了跨体系结构的可移植性。大量的实验评估表明了该算法的有效性和优越的性能。我们的方法可以很容易地与现有的方法相结合，在基于MLP的ResMLP上可以提高高达38.0%的可转移性。我们的方法在MLP-Mixer上生成的对抗性实例能够超过DenseNet针对CNN生成的对抗性实例的可转移性。据我们所知，我们是第一个研究MLP-Mixer对抗性转移的工作。



## **12. Mixed Strategies for Security Games with General Defending Requirements**

具有一般防御要求的安全博弈的混合策略 cs.GT

Accepted by IJCAI-2022

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12158v1)

**Authors**: Rufan Bai, Haoxing Lin, Xinyu Yang, Xiaowei Wu, Minming Li, Weijia Jia

**Abstracts**: The Stackelberg security game is played between a defender and an attacker, where the defender needs to allocate a limited amount of resources to multiple targets in order to minimize the loss due to adversarial attack by the attacker. While allowing targets to have different values, classic settings often assume uniform requirements to defend the targets. This enables existing results that study mixed strategies (randomized allocation algorithms) to adopt a compact representation of the mixed strategies.   In this work, we initiate the study of mixed strategies for the security games in which the targets can have different defending requirements. In contrast to the case of uniform defending requirement, for which an optimal mixed strategy can be computed efficiently, we show that computing the optimal mixed strategy is NP-hard for the general defending requirements setting. However, we show that strong upper and lower bounds for the optimal mixed strategy defending result can be derived. We propose an efficient close-to-optimal Patching algorithm that computes mixed strategies that use only few pure strategies. We also study the setting when the game is played on a network and resource sharing is enabled between neighboring targets. Our experimental results demonstrate the effectiveness of our algorithm in several large real-world datasets.

摘要: Stackelberg安全博弈是在防御者和攻击者之间进行的，防御者需要将有限的资源分配给多个目标，以便将攻击者的对抗性攻击造成的损失降至最低。虽然允许目标具有不同的值，但经典设置通常假定保护目标的要求是统一的。这使得研究混合策略(随机分配算法)的现有结果能够采用混合策略的紧凑表示。在这项工作中，我们发起了安全博弈的混合策略的研究，其中目标可以有不同的防御需求。与统一防御需求情况下可以有效计算最优混合策略的情况相比，对于一般防御需求设置，计算最优混合策略是NP难的。然而，我们证明了最优混合策略防御结果的上界和下界是强的。我们提出了一种高效的接近最优的修补算法，该算法只使用很少的纯策略来计算混合策略。我们还研究了当游戏在网络上进行并且相邻目标之间实现资源共享时的设置。我们的实验结果证明了我们的算法在几个大型真实数据集上的有效性。



## **13. Source-independent quantum random number generator against detector blinding attacks**

抗探测器盲攻击的源无关量子随机数发生器 quant-ph

14 pages, 7 figures, 6 tables, comments are welcome

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12156v1)

**Authors**: Wen-Bo Liu, Yu-Shuo Lu, Yao Fu, Si-Cheng Huang, Ze-Jie Yin, Kun Jiang, Hua-Lei Yin, Zeng-Bing Chen

**Abstracts**: Randomness, mainly in the form of random numbers, is the fundamental prerequisite for the security of many cryptographic tasks. Quantum randomness can be extracted even if adversaries are fully aware of the protocol and even control the randomness source. However, an adversary can further manipulate the randomness via detector blinding attacks, which are a hacking attack suffered by protocols with trusted detectors. Here, by treating no-click events as valid error events, we propose a quantum random number generation protocol that can simultaneously address source vulnerability and ferocious detector blinding attacks. The method can be extended to high-dimensional random number generation. We experimentally demonstrate the ability of our protocol to generate random numbers for two-dimensional measurement with a generation speed of 0.515 Mbps, which is two orders of magnitude higher than that of device-independent protocols that can address both issues of imperfect sources and imperfect detectors.

摘要: 随机性，主要是随机数的形式，是许多密码任务安全的基本前提。即使攻击者完全知道该协议，甚至控制了随机性来源，也可以提取量子随机性。然而，攻击者可以通过检测器盲化攻击进一步操纵随机性，这是具有可信检测器的协议遭受的黑客攻击。这里，通过将无点击事件视为有效的错误事件，我们提出了一种量子随机数生成协议，该协议可以同时应对源漏洞和猛烈的检测器盲攻击。该方法可以推广到高维随机数的生成。我们通过实验证明了该协议能够生成用于二维测量的随机数，生成速度为0.515 Mbps，比设备无关的协议高出两个数量级，后者既可以解决不完善的源问题，也可以解决不完善的检测器问题。



## **14. Self-recoverable Adversarial Examples: A New Effective Protection Mechanism in Social Networks**

可自我恢复的敌意例子：一种新的有效的社交网络保护机制 cs.CV

13 pages, 11 figures

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12050v1)

**Authors**: Jiawei Zhang, Jinwei Wang, Hao Wang, Xiangyang Luo

**Abstracts**: Malicious intelligent algorithms greatly threaten the security of social users' privacy by detecting and analyzing the uploaded photos to social network platforms. The destruction to DNNs brought by the adversarial attack sparks the potential that adversarial examples serve as a new protection mechanism for privacy security in social networks. However, the existing adversarial example does not have recoverability for serving as an effective protection mechanism. To address this issue, we propose a recoverable generative adversarial network to generate self-recoverable adversarial examples. By modeling the adversarial attack and recovery as a united task, our method can minimize the error of the recovered examples while maximizing the attack ability, resulting in better recoverability of adversarial examples. To further boost the recoverability of these examples, we exploit a dimension reducer to optimize the distribution of adversarial perturbation. The experimental results prove that the adversarial examples generated by the proposed method present superior recoverability, attack ability, and robustness on different datasets and network architectures, which ensure its effectiveness as a protection mechanism in social networks.

摘要: 恶意智能算法通过检测和分析上传到社交网络平台的照片，极大地威胁到社交用户的隐私安全。敌意攻击对DNN的破坏引发了敌意例子作为一种新的社交网络隐私安全保护机制的潜力。然而，现有的对抗性范例作为一种有效的保护机制，并不具有可恢复性。为了解决这个问题，我们提出了一个可恢复的生成性对抗性网络来生成可自我恢复的对抗性实例。通过将对抗性攻击和恢复建模为一个联合任务，该方法可以在最大化攻击能力的同时最小化恢复样本的误差，从而使对抗性样本具有更好的可恢复性。为了进一步提高这些例子的可恢复性，我们开发了一个降维器来优化对抗性扰动的分布。实验结果表明，该方法生成的恶意实例在不同的数据集和网络体系结构下具有良好的可恢复性、攻击性和健壮性，是一种有效的社交网络保护机制。



## **15. Can Rationalization Improve Robustness?**

合理化能提高健壮性吗？ cs.CL

Accepted to NAACL 2022

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2204.11790v1)

**Authors**: Howard Chen, Jacqueline He, Karthik Narasimhan, Danqi Chen

**Abstracts**: A growing line of work has investigated the development of neural NLP models that can produce rationales--subsets of input that can explain their model predictions. In this paper, we ask whether such rationale models can also provide robustness to adversarial attacks in addition to their interpretable nature. Since these models need to first generate rationales ("rationalizer") before making predictions ("predictor"), they have the potential to ignore noise or adversarially added text by simply masking it out of the generated rationale. To this end, we systematically generate various types of 'AddText' attacks for both token and sentence-level rationalization tasks, and perform an extensive empirical evaluation of state-of-the-art rationale models across five different tasks. Our experiments reveal that the rationale models show the promise to improve robustness, while they struggle in certain scenarios--when the rationalizer is sensitive to positional bias or lexical choices of attack text. Further, leveraging human rationale as supervision does not always translate to better performance. Our study is a first step towards exploring the interplay between interpretability and robustness in the rationalize-then-predict framework.

摘要: 越来越多的工作研究了神经NLP模型的发展，这种模型可以产生原理--输入的子集可以解释他们的模型预测。在本文中，我们询问这些基本模型除了具有可解释的性质外，是否还可以提供对对手攻击的稳健性。由于这些模型在做出预测(“预测者”)之前需要首先生成理由(“理性器”)，因此它们有可能忽略噪声或相反添加的文本，只需将其从生成的理由中掩盖出来。为此，我们系统地为标记和句子级合理化任务生成了各种类型的AddText攻击，并在五个不同的任务中对最先进的理性模型进行了广泛的经验评估。我们的实验表明，当理性器对位置偏差或攻击文本的词汇选择敏感时，基本模型显示出提高稳健性的前景，而它们在某些场景中却举步维艰。此外，利用人的理性作为监督并不总是能转化为更好的业绩。我们的研究是探索在合理化-然后预测框架中可解释性和稳健性之间的相互作用的第一步。



## **16. Discovering Exfiltration Paths Using Reinforcement Learning with Attack Graphs**

基于攻击图强化学习的渗出路径发现 cs.CR

The 5th IEEE Conference on Dependable and Secure Computing (IEEE DSC  2022)

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2201.12416v2)

**Authors**: Tyler Cody, Abdul Rahman, Christopher Redino, Lanxiao Huang, Ryan Clark, Akshay Kakkar, Deepak Kushwaha, Paul Park, Peter Beling, Edward Bowen

**Abstracts**: Reinforcement learning (RL), in conjunction with attack graphs and cyber terrain, are used to develop reward and state associated with determination of optimal paths for exfiltration of data in enterprise networks. This work builds on previous crown jewels (CJ) identification that focused on the target goal of computing optimal paths that adversaries may traverse toward compromising CJs or hosts within their proximity. This work inverts the previous CJ approach based on the assumption that data has been stolen and now must be quietly exfiltrated from the network. RL is utilized to support the development of a reward function based on the identification of those paths where adversaries desire reduced detection. Results demonstrate promising performance for a sizable network environment.

摘要: 强化学习(RL)与攻击图和网络地形相结合，用于开发与确定企业网络中数据泄漏的最佳路径相关的奖励和状态。这项工作建立在以前的皇冠宝石(CJ)识别的基础上，该识别专注于计算对手可能穿过的最优路径的目标，以危害其邻近的CJ或主机。这项工作颠覆了之前CJ的方法，该方法基于数据已被窃取，现在必须从网络中悄悄渗出的假设。利用RL来支持基于识别其中对手希望减少检测的那些路径的奖励函数的开发。结果表明，在相当大的网络环境中具有良好的性能。



## **17. Reconstructing Training Data with Informed Adversaries**

利用知情对手重建训练数据 cs.CR

Published at "2022 IEEE Symposium on Security and Privacy (SP)"

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2201.04845v2)

**Authors**: Borja Balle, Giovanni Cherubin, Jamie Hayes

**Abstracts**: Given access to a machine learning model, can an adversary reconstruct the model's training data? This work studies this question from the lens of a powerful informed adversary who knows all the training data points except one. By instantiating concrete attacks, we show it is feasible to reconstruct the remaining data point in this stringent threat model. For convex models (e.g. logistic regression), reconstruction attacks are simple and can be derived in closed-form. For more general models (e.g. neural networks), we propose an attack strategy based on training a reconstructor network that receives as input the weights of the model under attack and produces as output the target data point. We demonstrate the effectiveness of our attack on image classifiers trained on MNIST and CIFAR-10, and systematically investigate which factors of standard machine learning pipelines affect reconstruction success. Finally, we theoretically investigate what amount of differential privacy suffices to mitigate reconstruction attacks by informed adversaries. Our work provides an effective reconstruction attack that model developers can use to assess memorization of individual points in general settings beyond those considered in previous works (e.g. generative language models or access to training gradients); it shows that standard models have the capacity to store enough information to enable high-fidelity reconstruction of training data points; and it demonstrates that differential privacy can successfully mitigate such attacks in a parameter regime where utility degradation is minimal.

摘要: 在获得机器学习模型的情况下，对手能否重建模型的训练数据？这项工作从一个强大的知情对手的角度来研究这个问题，他知道除一个以外的所有训练数据点。通过实例化具体攻击，我们证明了在这个严格的威胁模型中重构剩余数据点是可行的。对于凸模型(例如Logistic回归)，重构攻击很简单，并且可以以闭合形式推导出来。对于更一般的模型(如神经网络)，我们提出了一种基于训练重构器网络的攻击策略，该重建器网络接收被攻击模型的权重作为输入，并产生目标数据点作为输出。我们在MNIST和CIFAR-10上训练的图像分类器上验证了我们的攻击的有效性，并系统地研究了标准机器学习管道中哪些因素影响重建成功。最后，我们从理论上研究了多少差异隐私足以缓解知情攻击者的重构攻击。我们的工作提供了一种有效的重建攻击，模型开发者可以用它来评估在一般环境下对单个点的记忆，而不是以前的工作中考虑的那些(例如，生成性语言模型或对训练梯度的访问)；它表明标准模型有能力存储足够的信息来实现训练数据点的高保真重建；并且它证明了在效用降级最小的参数机制中，差分隐私可以成功地缓解这种攻击。



## **18. A Simple Structure For Building A Robust Model**

一种用于建立稳健模型的简单结构 cs.CV

10 pages, 3 figures, 4 tables

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2204.11596v1)

**Authors**: Xiao Tan, JingBo Gao, Ruolin Li

**Abstracts**: As deep learning applications, especially programs of computer vision, are increasingly deployed in our lives, we have to think more urgently about the security of these applications.One effective way to improve the security of deep learning models is to perform adversarial training, which allows the model to be compatible with samples that are deliberately created for use in attacking the model.Based on this, we propose a simple architecture to build a model with a certain degree of robustness, which improves the robustness of the trained network by adding an adversarial sample detection network for cooperative training.At the same time, we design a new data sampling strategy that incorporates multiple existing attacks, allowing the model to adapt to many different adversarial attacks with a single training.We conducted some experiments to test the effectiveness of this design based on Cifar10 dataset, and the results indicate that it has some degree of positive effect on the robustness of the model.Our code could be found at https://github.com/dowdyboy/simple_structure_for_robust_model.

摘要: 随着深度学习应用，特别是计算机视觉应用的日益广泛，我们不得不更加迫切地考虑这些应用的安全性。提高深度学习模型安全性的有效方法之一是进行对抗性训练，使模型与故意创建的用于攻击模型的样本相兼容。在此基础上，提出了一种简单的架构来构建具有一定健壮性的模型，通过增加对抗性样本检测网络来进行协作训练，从而提高了训练网络的健壮性。同时，我们设计了一种新的数据采样策略，该策略融合了多种现有的攻击，在Cifar10数据集上进行了实验，测试结果表明，该设计对模型的健壮性有一定的积极作用。我们的代码可以在https://github.com/dowdyboy/simple_structure_for_robust_model.上找到



## **19. Dominating Vertical Collaborative Learning Systems**

主导垂直协作学习系统 cs.CR

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2201.02775v2)

**Authors**: Qi Pang, Yuanyuan Yuan, Shuai Wang

**Abstracts**: Vertical collaborative learning system also known as vertical federated learning (VFL) system has recently become prominent as a concept to process data distributed across many individual sources without the need to centralize it. Multiple participants collaboratively train models based on their local data in a privacy-preserving manner. To date, VFL has become a de facto solution to securely learn a model among organizations, allowing knowledge to be shared without compromising privacy of any individual organizations.   Despite the prosperous development of VFL systems, we find that certain inputs of a participant, named adversarial dominating inputs (ADIs), can dominate the joint inference towards the direction of the adversary's will and force other (victim) participants to make negligible contributions, losing rewards that are usually offered regarding the importance of their contributions in collaborative learning scenarios.   We conduct a systematic study on ADIs by first proving their existence in typical VFL systems. We then propose gradient-based methods to synthesize ADIs of various formats and exploit common VFL systems. We further launch greybox fuzz testing, guided by the resiliency score of "victim" participants, to perturb adversary-controlled inputs and systematically explore the VFL attack surface in a privacy-preserving manner. We conduct an in-depth study on the influence of critical parameters and settings in synthesizing ADIs. Our study reveals new VFL attack opportunities, promoting the identification of unknown threats before breaches and building more secure VFL systems.

摘要: 垂直协作学习系统，也被称为垂直联合学习(VFL)系统，作为一种概念，它可以处理分布在多个独立数据源的数据，而不需要集中这些数据。多个参与者以隐私保护的方式基于他们的本地数据协作训练模型。到目前为止，VFL已经成为在组织之间安全地学习模式的事实上的解决方案，允许在不损害任何个人组织隐私的情况下共享知识。尽管VFL系统的蓬勃发展，我们发现参与者的某些输入，称为对抗性主导输入(ADI)，可以主导朝着对手意愿方向的联合推理，并迫使其他(受害者)参与者做出可以忽略不计的贡献，失去通常提供的关于他们在协作学习场景中贡献的重要性的奖励。我们首先通过证明ADI在典型的VFL系统中的存在来对ADI进行系统的研究。然后，我们提出了基于梯度的方法来合成各种格式的ADI，并开发了常见的VFL系统。我们进一步推出灰盒模糊测试，以“受害者”参与者的弹性分数为指导，扰乱对手控制的输入，并以保护隐私的方式系统地探索VFL攻击面。我们深入研究了关键参数和设置对ADI合成的影响。我们的研究揭示了新的VFL攻击机会，促进了在入侵之前识别未知威胁，并建立了更安全的VFL系统。



## **20. Real or Virtual: A Video Conferencing Background Manipulation-Detection System**

真实还是虚拟：一种视频会议背景操纵检测系统 cs.CV

34 pages. arXiv admin note: text overlap with arXiv:2106.15130

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2204.11853v1)

**Authors**: Ehsan Nowroozi, Yassine Mekdad, Mauro Conti, Simone Milani, Selcuk Uluagac, Berrin Yanikoglu

**Abstracts**: Recently, the popularity and wide use of the last-generation video conferencing technologies created an exponential growth in its market size. Such technology allows participants in different geographic regions to have a virtual face-to-face meeting. Additionally, it enables users to employ a virtual background to conceal their own environment due to privacy concerns or to reduce distractions, particularly in professional settings. Nevertheless, in scenarios where the users should not hide their actual locations, they may mislead other participants by claiming their virtual background as a real one. Therefore, it is crucial to develop tools and strategies to detect the authenticity of the considered virtual background. In this paper, we present a detection strategy to distinguish between real and virtual video conferencing user backgrounds. We demonstrate that our detector is robust against two attack scenarios. The first scenario considers the case where the detector is unaware about the attacks and inn the second scenario, we make the detector aware of the adversarial attacks, which we refer to Adversarial Multimedia Forensics (i.e, the forensically-edited frames are included in the training set). Given the lack of publicly available dataset of virtual and real backgrounds for video conferencing, we created our own dataset and made them publicly available [1]. Then, we demonstrate the robustness of our detector against different adversarial attacks that the adversary considers. Ultimately, our detector's performance is significant against the CRSPAM1372 [2] features, and post-processing operations such as geometric transformations with different quality factors that the attacker may choose. Moreover, our performance results shows that we can perfectly identify a real from a virtual background with an accuracy of 99.80%.

摘要: 最近，上一代视频会议技术的普及和广泛使用导致其市场规模呈指数级增长。这种技术允许不同地理区域的参与者进行虚拟面对面的会议。此外，它使用户能够使用虚拟背景来隐藏他们自己的环境，因为隐私问题或减少分心，特别是在专业环境中。然而，在用户不应该隐藏他们的实际位置的情况下，他们可能会误导其他参与者，声称他们的虚拟背景是真实的。因此，开发工具和策略来检测所考虑的虚拟背景的真实性是至关重要的。本文提出了一种区分真实和虚拟视频会议用户背景的检测策略。我们证明了我们的检测器对两种攻击场景都是健壮的。第一种情况考虑了检测器不知道攻击的情况，在第二种情况下，我们让检测器知道对抗性攻击，我们称之为对抗性多媒体取证(即，经取证编辑的帧包括在训练集中)。由于缺乏公开可用的视频会议虚拟和真实背景的数据集，我们创建了自己的数据集并将其公开[1]。然后，我们证明了我们的检测器对对手所考虑的不同的对手攻击具有健壮性。归根结底，我们的检测器相对于CRSPAM1372[2]功能和后处理操作(如攻击者可能选择的具有不同质量因子的几何变换)的性能是显著的。此外，我们的性能结果表明，我们可以很好地识别真实和虚拟的背景，准确率为99.80%。



## **21. Improving Deep Learning Model Robustness Against Adversarial Attack by Increasing the Network Capacity**

通过增加网络容量提高深度学习模型对敌意攻击的稳健性 cs.LG

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.11357v1)

**Authors**: Marco Marchetti, Edmond S. L. Ho

**Abstracts**: Nowadays, we are more and more reliant on Deep Learning (DL) models and thus it is essential to safeguard the security of these systems. This paper explores the security issues in Deep Learning and analyses, through the use of experiments, the way forward to build more resilient models. Experiments are conducted to identify the strengths and weaknesses of a new approach to improve the robustness of DL models against adversarial attacks. The results show improvements and new ideas that can be used as recommendations for researchers and practitioners to create increasingly better DL algorithms.

摘要: 如今，我们越来越依赖深度学习模型，因此保障这些系统的安全至关重要。本文探讨了深度学习中的安全问题，并通过实验分析了构建更具弹性模型的前进方向。通过实验确定了一种新方法的优点和缺点，以提高DL模型对对手攻击的稳健性。研究结果显示了一些改进和新的想法，可以作为研究人员和实践者创建越来越好的DL算法的建议。



## **22. A Comprehensive Test Pattern Generation Approach Exploiting SAT Attack for Logic Locking**

一种利用SAT攻击进行逻辑锁定的综合测试码生成方法 cs.CR

10 pages, 8 figures

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.11307v1)

**Authors**: Yadi Zhong, Ujjwal Guin

**Abstracts**: The need for reducing manufacturing defect escape in today's safety-critical applications requires increased fault coverage. However, generating a test set using commercial automatic test pattern generation (ATPG) tools that lead to zero-defect escape is still an open problem. It is challenging to detect all stuck-at faults to reach 100% fault coverage. In parallel, the hardware security community has been actively involved in developing solutions for logic locking to prevent IP piracy. Locks (e.g., XOR gates) are inserted in different locations of the netlist so that an adversary cannot determine the secret key. Unfortunately, the Boolean satisfiability (SAT) based attack, introduced in [1], can break different logic locking schemes in minutes. In this paper, we propose a novel test pattern generation approach using the powerful SAT attack on logic locking. A stuck-at fault is modeled as a locked gate with a secret key. Our modeling of stuck-at faults preserves the property of fault activation and propagation. We show that the input pattern that determines the key is a test for the stuck-at fault. We propose two different approaches for test pattern generation. First, a single stuck-at fault is targeted, and a corresponding locked circuit with one key bit is created. This approach generates one test pattern per fault. Second, we consider a group of faults and convert the circuit to its locked version with multiple key bits. The inputs obtained from the SAT tool are the test set for detecting this group of faults. Our approach is able to find test patterns for hard-to-detect faults that were previously failed in commercial ATPG tools. The proposed test pattern generation approach can efficiently detect redundant faults present in a circuit. We demonstrate the effectiveness of the approach on ITC'99 benchmarks. The results show that we can achieve a perfect fault coverage reaching 100%.

摘要: 在当今的安全关键应用中，减少制造缺陷逃逸的需要需要增加故障覆盖率。然而，使用商业自动测试模式生成(ATPG)工具生成测试集以实现零缺陷逃逸仍然是一个未解决的问题。要检测所有固定故障以达到100%的故障覆盖率是具有挑战性的。与此同时，硬件安全界一直积极参与开发逻辑锁定解决方案，以防止知识产权盗版。锁(例如，异或门)被插入网表的不同位置，使得对手不能确定密钥。不幸的是，在[1]中引入的基于布尔可满足性(SAT)的攻击可以在几分钟内破解不同的逻辑锁定方案。在本文中，我们提出了一种新的测试模式生成方法，该方法利用了对逻辑锁的强大SAT攻击。一个顽固的错误被建模为一扇锁着的门和一把密钥。我们对固定故障的建模保留了故障激活和传播的性质。我们证明了决定关键字的输入模式是对固定错误的测试。我们提出了两种不同的测试模式生成方法。首先，针对单个固定故障，创建具有一个密钥位的相应锁定电路。该方法为每个故障生成一个测试模式。其次，我们考虑一组故障，并将电路转换为具有多个密钥位的锁定版本。从SAT工具获得的输入是用于检测这组故障的测试集。我们的方法能够为以前在商业ATPG工具中失败的难以检测的故障找到测试模式。提出的测试码生成方法可以有效地检测电路中存在的冗余故障。我们在ITC‘99基准上证明了该方法的有效性。结果表明，我们可以达到100%的完美故障覆盖率。



## **23. Dictionary Attacks on Speaker Verification**

针对说话人确认的词典攻击 cs.SD

Manuscript and supplement, currently under review

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.11304v1)

**Authors**: Mirko Marras, Pawel Korus, Anubhav Jain, Nasir Memon

**Abstracts**: In this paper, we propose dictionary attacks against speaker verification - a novel attack vector that aims to match a large fraction of speaker population by chance. We introduce a generic formulation of the attack that can be used with various speech representations and threat models. The attacker uses adversarial optimization to maximize raw similarity of speaker embeddings between a seed speech sample and a proxy population. The resulting master voice successfully matches a non-trivial fraction of people in an unknown population. Adversarial waveforms obtained with our approach can match on average 69% of females and 38% of males enrolled in the target system at a strict decision threshold calibrated to yield false alarm rate of 1%. By using the attack with a black-box voice cloning system, we obtain master voices that are effective in the most challenging conditions and transferable between speaker encoders. We also show that, combined with multiple attempts, this attack opens even more to serious issues on the security of these systems.

摘要: 在本文中，我们提出了针对说话人验证的词典攻击，这是一种新的攻击向量，旨在随机匹配大部分说话人群体。我们介绍了一种可用于各种语音表示和威胁模型的攻击的通用公式。攻击者使用对抗性优化来最大化种子语音样本和代理群体之间说话人嵌入的原始相似性。由此产生的主音成功地匹配了未知人群中的一小部分人。使用我们的方法获得的对抗性波形可以在严格的判决阈值下与目标系统中登记的平均69%的女性和38%的男性匹配，该阈值被校准为产生1%的错误警报率。通过使用黑匣子语音克隆系统进行攻击，我们获得了在最具挑战性的条件下有效的主音，并且可以在说话人编码者之间传输。我们还表明，与多次尝试相结合，这种攻击会给这些系统的安全带来更严重的问题。



## **24. Sampling-based Fast Gradient Rescaling Method for Highly Transferable Adversarial Attacks**

一种基于采样的高可转移对抗性攻击快速梯度重缩放方法 cs.CV

The writing and experiment of the article need to be further  strengthened

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.02887v2)

**Authors**: Xu Han, Anmin Liu, Yifeng Xiong, Yanbo Fan, Kun He

**Abstracts**: Deep neural networks have shown to be very vulnerable to adversarial examples crafted by adding human-imperceptible perturbations to benign inputs. After achieving impressive attack success rates in the white-box setting, more focus is shifted to black-box attacks. In either case, the common gradient-based approaches generally use the $sign$ function to generate perturbations at the end of the process. However, only a few works pay attention to the limitation of the $sign$ function. Deviation between the original gradient and the generated noises may lead to inaccurate gradient update estimation and suboptimal solutions for adversarial transferability, which is crucial for black-box attacks. To address this issue, we propose a Sampling-based Fast Gradient Rescaling Method (S-FGRM) to improve the transferability of the crafted adversarial examples. Specifically, we use data rescaling to substitute the inefficient $sign$ function in gradient-based attacks without extra computational cost. We also propose a Depth First Sampling method to eliminate the fluctuation of rescaling and stabilize the gradient update. Our method can be used in any gradient-based optimizations and is extensible to be integrated with various input transformation or ensemble methods for further improving the adversarial transferability. Extensive experiments on the standard ImageNet dataset show that our S-FGRM could significantly boost the transferability of gradient-based attacks and outperform the state-of-the-art baselines.

摘要: 深度神经网络已被证明非常容易受到敌意例子的攻击，这些例子是通过在良性输入中添加人类无法察觉的扰动来构建的。在白盒攻击中取得了令人印象深刻的攻击成功率之后，更多的注意力转移到了黑盒攻击上。在这两种情况下，常见的基于梯度的方法通常使用$SIGN$函数在过程结束时生成扰动。然而，只有少数著作注意到$SIGN$函数的局限性。原始梯度与产生的噪声之间的偏差可能会导致不准确的梯度更新估计和对抗性转移的次优解，这是黑盒攻击的关键。针对这一问题，我们提出了一种基于采样的快速梯度重缩放方法(S-FGRM)来提高恶意例子的可转移性。具体地说，在基于梯度的攻击中，我们使用数据重缩放来代替低效的$sign$函数，而不需要额外的计算代价。我们还提出了深度优先采样的方法，消除了重缩放的波动，稳定了梯度更新。我们的方法可以用于任何基于梯度的优化，并且可以扩展到与各种输入变换或集成方法相集成，以进一步提高对抗性转移。在标准ImageNet数据集上的大量实验表明，我们的S-FGRM可以显著提高基于梯度的攻击的可转移性，并优于最新的基线。



## **25. Eliminating Backdoor Triggers for Deep Neural Networks Using Attention Relation Graph Distillation**

基于注意力关系图提取的深度神经网络后门触发器剔除 cs.LG

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.09975v2)

**Authors**: Jun Xia, Ting Wang, Jiepin Ding, Xian Wei, Mingsong Chen

**Abstracts**: Due to the prosperity of Artificial Intelligence (AI) techniques, more and more backdoors are designed by adversaries to attack Deep Neural Networks (DNNs).Although the state-of-the-art method Neural Attention Distillation (NAD) can effectively erase backdoor triggers from DNNs, it still suffers from non-negligible Attack Success Rate (ASR) together with lowered classification ACCuracy (ACC), since NAD focuses on backdoor defense using attention features (i.e., attention maps) of the same order. In this paper, we introduce a novel backdoor defense framework named Attention Relation Graph Distillation (ARGD), which fully explores the correlation among attention features with different orders using our proposed Attention Relation Graphs (ARGs). Based on the alignment of ARGs between both teacher and student models during knowledge distillation, ARGD can eradicate more backdoor triggers than NAD. Comprehensive experimental results show that, against six latest backdoor attacks, ARGD outperforms NAD by up to 94.85% reduction in ASR, while ACC can be improved by up to 3.23%.

摘要: 由于人工智能(AI)技术的蓬勃发展，越来越多的对手设计了后门来攻击深度神经网络(DNN)，尽管目前最先进的方法神经注意力蒸馏(NAD)可以有效地清除DNN中的后门触发，但由于NAD侧重于利用同阶的注意特征(即注意力地图)进行后门防御，因此仍然存在不可忽视的攻击成功率(ASR)和较低的分类精度(ACC)。本文介绍了一种新的后门防御框架--注意关系图蒸馏(ARGD)，它充分利用我们提出的注意关系图(ARGs)来探索不同阶次的注意特征之间的相关性。基于知识提炼过程中教师和学生模型之间的ARG对齐，ARGD比NAD能够消除更多的后门触发。综合实验结果表明，对于最近的6次后门攻击，ARGD在ASR上比NAD降低了94.85%，而ACC则提高了3.23%。



## **26. Stochastic Variance Reduced Ensemble Adversarial Attack for Boosting the Adversarial Transferability**

提高对抗性转移能力的随机方差降低集成对抗性攻击 cs.LG

11 pages, 6 figures, accepted by CVPR 2022

**SubmitDate**: 2022-04-23    [paper-pdf](http://arxiv.org/pdf/2111.10752v2)

**Authors**: Yifeng Xiong, Jiadong Lin, Min Zhang, John E. Hopcroft, Kun He

**Abstracts**: The black-box adversarial attack has attracted impressive attention for its practical use in the field of deep learning security. Meanwhile, it is very challenging as there is no access to the network architecture or internal weights of the target model. Based on the hypothesis that if an example remains adversarial for multiple models, then it is more likely to transfer the attack capability to other models, the ensemble-based adversarial attack methods are efficient and widely used for black-box attacks. However, ways of ensemble attack are rather less investigated, and existing ensemble attacks simply fuse the outputs of all the models evenly. In this work, we treat the iterative ensemble attack as a stochastic gradient descent optimization process, in which the variance of the gradients on different models may lead to poor local optima. To this end, we propose a novel attack method called the stochastic variance reduced ensemble (SVRE) attack, which could reduce the gradient variance of the ensemble models and take full advantage of the ensemble attack. Empirical results on the standard ImageNet dataset demonstrate that the proposed method could boost the adversarial transferability and outperforms existing ensemble attacks significantly. Code is available at https://github.com/JHL-HUST/SVRE.

摘要: 黑盒对抗性攻击因其在深度学习安全领域的实际应用而备受关注。同时，这是非常具有挑战性的，因为无法访问目标模型的网络体系结构或内部权重。基于这样的假设，如果一个例子在多个模型上保持对抗性，那么它更有可能将攻击能力转移到其他模型上，基于集成的对抗性攻击方法是有效的，并被广泛应用于黑盒攻击。然而，集成攻击方法的研究相对较少，现有的集成攻击只是将所有模型的输出均匀地融合在一起。在本文中，我们将迭代集成攻击视为一个随机梯度下降优化过程，其中不同模型上的梯度变化可能导致局部最优解较差。为此，我们提出了一种新的攻击方法，称为随机方差减少集成(SVRE)攻击，它可以降低集成模型的梯度方差，并充分利用集成攻击的优势。在标准ImageNet数据集上的实验结果表明，该方法可以提高对抗性可转移性，并显著优于现有的集成攻击。代码可在https://github.com/JHL-HUST/SVRE.上找到



## **27. Certifiably Robust Variational Autoencoders**

可证明稳健性的变分自动编码器 stat.ML

12 pages and appendix

**SubmitDate**: 2022-04-23    [paper-pdf](http://arxiv.org/pdf/2102.07559v3)

**Authors**: Ben Barrett, Alexander Camuto, Matthew Willetts, Tom Rainforth

**Abstracts**: We introduce an approach for training Variational Autoencoders (VAEs) that are certifiably robust to adversarial attack. Specifically, we first derive actionable bounds on the minimal size of an input perturbation required to change a VAE's reconstruction by more than an allowed amount, with these bounds depending on certain key parameters such as the Lipschitz constants of the encoder and decoder. We then show how these parameters can be controlled, thereby providing a mechanism to ensure \textit{a priori} that a VAE will attain a desired level of robustness. Moreover, we extend this to a complete practical approach for training such VAEs to ensure our criteria are met. Critically, our method allows one to specify a desired level of robustness \emph{upfront} and then train a VAE that is guaranteed to achieve this robustness. We further demonstrate that these Lipschitz--constrained VAEs are more robust to attack than standard VAEs in practice.

摘要: 我们介绍了一种训练变分自动编码器(VAE)的方法，该方法对对手攻击具有可证明的健壮性。具体地说，我们首先推导出将VAE的重建改变超过允许量所需的输入扰动的最小大小的可操作界，这些界取决于某些关键参数，例如编码器和解码器的Lipschitz常数。然后，我们将展示如何控制这些参数，从而提供一种机制来确保VAE将达到所需的健壮性级别。此外，我们将此扩展为培训此类VAE的完整实用方法，以确保符合我们的标准。关键是，我们的方法允许指定期望的健壮性级别，然后训练保证实现该健壮性的VAE。在实践中，我们进一步证明了这些Lipschitz约束的VAE比标准VAE具有更强的抗攻击能力。



## **28. Smart App Attack: Hacking Deep Learning Models in Android Apps**

智能应用程序攻击：入侵Android应用程序中的深度学习模型 cs.LG

Accepted to IEEE Transactions on Information Forensics and Security.  This is a preprint version, the copyright belongs to The Institute of  Electrical and Electronics Engineers

**SubmitDate**: 2022-04-23    [paper-pdf](http://arxiv.org/pdf/2204.11075v1)

**Authors**: Yujin Huang, Chunyang Chen

**Abstracts**: On-device deep learning is rapidly gaining popularity in mobile applications. Compared to offloading deep learning from smartphones to the cloud, on-device deep learning enables offline model inference while preserving user privacy. However, such mechanisms inevitably store models on users' smartphones and may invite adversarial attacks as they are accessible to attackers. Due to the characteristic of the on-device model, most existing adversarial attacks cannot be directly applied for on-device models. In this paper, we introduce a grey-box adversarial attack framework to hack on-device models by crafting highly similar binary classification models based on identified transfer learning approaches and pre-trained models from TensorFlow Hub. We evaluate the attack effectiveness and generality in terms of four different settings including pre-trained models, datasets, transfer learning approaches and adversarial attack algorithms. The results demonstrate that the proposed attacks remain effective regardless of different settings, and significantly outperform state-of-the-art baselines. We further conduct an empirical study on real-world deep learning mobile apps collected from Google Play. Among 53 apps adopting transfer learning, we find that 71.7\% of them can be successfully attacked, which includes popular ones in medicine, automation, and finance categories with critical usage scenarios. The results call for the awareness and actions of deep learning mobile app developers to secure the on-device models. The code of this work is available at https://github.com/Jinxhy/SmartAppAttack

摘要: 设备上的深度学习在移动应用程序中迅速流行起来。与将深度学习从智能手机转移到云相比，设备上的深度学习支持离线模型推理，同时保护用户隐私。然而，这种机制不可避免地将模型存储在用户的智能手机上，并可能招致对抗性攻击，因为攻击者可以访问这些模型。由于设备上模型的特点，现有的大多数对抗性攻击不能直接应用于设备上模型。在本文中，我们引入了一种灰盒对抗性攻击框架，通过基于识别的迁移学习方法和TensorFlow Hub的预训练模型构建高度相似的二进制分类模型来破解设备上的模型。我们从预先训练的模型、数据集、迁移学习方法和对抗性攻击算法四个不同的设置来评估攻击的有效性和通用性。结果表明，所提出的攻击无论在不同的设置下都保持有效，并且显著优于最先进的基线。我们进一步对从Google Play收集的真实世界深度学习移动应用程序进行了实证研究。在53个采用迁移学习的应用中，我们发现其中71.7%的应用可以被成功攻击，其中包括医学、自动化、金融等具有关键使用场景的热门应用。这一结果呼吁深度学习移动应用开发者的意识和行动，以确保设备模型的安全。这项工作的代码可以在https://github.com/Jinxhy/SmartAppAttack上找到



## **29. Towards Data-Free Model Stealing in a Hard Label Setting**

在硬标签设置中走向无数据模型窃取 cs.CR

CVPR 2022, Project Page: https://sites.google.com/view/dfms-hl

**SubmitDate**: 2022-04-23    [paper-pdf](http://arxiv.org/pdf/2204.11022v1)

**Authors**: Sunandini Sanyal, Sravanti Addepalli, R. Venkatesh Babu

**Abstracts**: Machine learning models deployed as a service (MLaaS) are susceptible to model stealing attacks, where an adversary attempts to steal the model within a restricted access framework. While existing attacks demonstrate near-perfect clone-model performance using softmax predictions of the classification network, most of the APIs allow access to only the top-1 labels. In this work, we show that it is indeed possible to steal Machine Learning models by accessing only top-1 predictions (Hard Label setting) as well, without access to model gradients (Black-Box setting) or even the training dataset (Data-Free setting) within a low query budget. We propose a novel GAN-based framework that trains the student and generator in tandem to steal the model effectively while overcoming the challenge of the hard label setting by utilizing gradients of the clone network as a proxy to the victim's gradients. We propose to overcome the large query costs associated with a typical Data-Free setting by utilizing publicly available (potentially unrelated) datasets as a weak image prior. We additionally show that even in the absence of such data, it is possible to achieve state-of-the-art results within a low query budget using synthetically crafted samples. We are the first to demonstrate the scalability of Model Stealing in a restricted access setting on a 100 class dataset as well.

摘要: 部署为服务的机器学习模型(MLaaS)容易受到模型窃取攻击，在这种攻击中，对手试图在受限访问框架内窃取模型。虽然现有攻击使用Softmax分类网络预测展示了近乎完美的克隆模型性能，但大多数API仅允许访问前1个标签。在这项工作中，我们表明，通过只访问TOP-1预测(硬标签设置)，而不访问模型梯度(黑盒设置)，甚至在低查询预算内访问训练数据集(无数据设置)，确实有可能窃取机器学习模型。我们提出了一种新的基于GAN的框架，它通过利用克隆网络的梯度作为受害者梯度的代理来训练学生和生成器一起有效地窃取模型，同时克服了硬标签设置的挑战。我们建议通过利用公开可用的(潜在无关的)数据集作为弱图像先验来克服与典型的无数据设置相关联的大量查询成本。此外，我们还表明，即使在没有这样的数据的情况下，也可以使用人工合成的样本在较低的查询预算内获得最先进的结果。我们也是第一个在100类数据集上的受限访问设置中演示模型窃取的可扩展性的。



## **30. GFCL: A GRU-based Federated Continual Learning Framework against Adversarial Attacks in IoV**

GFCL：一种基于GRU的联合持续学习框架 cs.LG

11 pages, 12 figures, 3 tables; This paper has been submitted to IEEE  Internet of Things Journal

**SubmitDate**: 2022-04-23    [paper-pdf](http://arxiv.org/pdf/2204.11010v1)

**Authors**: Anum Talpur, Mohan Gurusamy

**Abstracts**: The integration of ML in 5G-based Internet of Vehicles (IoV) networks has enabled intelligent transportation and smart traffic management. Nonetheless, the security against adversarial attacks is also increasingly becoming a challenging task. Specifically, Deep Reinforcement Learning (DRL) is one of the widely used ML designs in IoV applications. The standard ML security techniques are not effective in DRL where the algorithm learns to solve sequential decision-making through continuous interaction with the environment, and the environment is time-varying, dynamic, and mobile. In this paper, we propose a Gated Recurrent Unit (GRU)-based federated continual learning (GFCL) anomaly detection framework against adversarial attacks in IoV. The objective is to present a lightweight and scalable framework that learns and detects the illegitimate behavior without having a-priori training dataset consisting of attack samples. We use GRU to predict a future data sequence to analyze and detect illegitimate behavior from vehicles in a federated learning-based distributed manner. We investigate the performance of our framework using real-world vehicle mobility traces. The results demonstrate the effectiveness of our proposed solution for different performance metrics.

摘要: ML在基于5G的车联网(IoV)网络中的整合实现了智能交通和智能交通管理。然而，对抗攻击的安全也日益成为一项具有挑战性的任务。其中，深度强化学习(DRL)是IoV应用中广泛使用的ML设计之一。标准的ML安全技术在DRL中并不有效，在DRL中，算法通过与环境的持续交互来学习解决顺序决策，并且环境是时变的、动态的和移动的。提出了一种基于门控递归单元(GRU)的联合连续学习(GFCL)异常检测框架，用于对抗IoV中的敌意攻击。其目的是提供一个轻量级和可扩展的框架，在没有包含攻击样本的先验训练数据集的情况下学习和检测非法行为。我们使用GRU来预测未来的数据序列，以基于联合学习的分布式方式来分析和检测车辆的非法行为。我们使用真实世界的车辆移动轨迹来研究我们的框架的性能。结果表明，本文提出的解决方案对于不同的性能指标是有效的。



## **31. A Tale of Two Models: Constructing Evasive Attacks on Edge Models**

两个模型的故事：构造对边模型的规避攻击 cs.CR

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2204.10933v1)

**Authors**: Wei Hao, Aahil Awatramani, Jiayang Hu, Chengzhi Mao, Pin-Chun Chen, Eyal Cidon, Asaf Cidon, Junfeng Yang

**Abstracts**: Full-precision deep learning models are typically too large or costly to deploy on edge devices. To accommodate to the limited hardware resources, models are adapted to the edge using various edge-adaptation techniques, such as quantization and pruning. While such techniques may have a negligible impact on top-line accuracy, the adapted models exhibit subtle differences in output compared to the original model from which they are derived. In this paper, we introduce a new evasive attack, DIVA, that exploits these differences in edge adaptation, by adding adversarial noise to input data that maximizes the output difference between the original and adapted model. Such an attack is particularly dangerous, because the malicious input will trick the adapted model running on the edge, but will be virtually undetectable by the original model, which typically serves as the authoritative model version, used for validation, debugging and retraining. We compare DIVA to a state-of-the-art attack, PGD, and show that DIVA is only 1.7-3.6% worse on attacking the adapted model but 1.9-4.2 times more likely not to be detected by the the original model under a whitebox and semi-blackbox setting, compared to PGD.

摘要: 全精度深度学习模型通常太大或太昂贵，无法在边缘设备上部署。为了适应有限的硬件资源，模型使用各种边缘自适应技术来适应边缘，如量化和剪枝。虽然这类技术对顶线精度的影响可能微乎其微，但与原始模型相比，改装后的模型在输出方面表现出了细微的差异。在本文中，我们介绍了一种新的规避攻击，DIVA，它利用边缘自适应的这些差异，通过在输入数据中添加对抗性噪声来最大化原始模型和自适应模型之间的输出差异。这种攻击特别危险，因为恶意输入将欺骗在边缘运行的适应模型，但原始模型实际上无法检测到，原始模型通常用作权威模型版本，用于验证、调试和再培训。我们将DIVA与最先进的攻击pgd进行了比较，结果表明，与pgd相比，在白盒和半黑盒设置下，DIVA在攻击适应的模型上只差1.7%-3.6%，但不被原始模型检测的可能性高1.9-4.2倍。



## **32. How Sampling Impacts the Robustness of Stochastic Neural Networks**

抽样如何影响随机神经网络的稳健性 cs.LG

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2204.10839v1)

**Authors**: Sina Däubener, Asja Fischer

**Abstracts**: Stochastic neural networks (SNNs) are random functions and predictions are gained by averaging over multiple realizations of this random function. Consequently, an adversarial attack is calculated based on one set of samples and applied to the prediction defined by another set of samples. In this paper we analyze robustness in this setting by deriving a sufficient condition for the given prediction process to be robust against the calculated attack. This allows us to identify the factors that lead to an increased robustness of SNNs and helps to explain the impact of the variance and the amount of samples. Among other things, our theoretical analysis gives insights into (i) why increasing the amount of samples drawn for the estimation of adversarial examples increases the attack's strength, (ii) why decreasing sample size during inference hardly influences the robustness, and (iii) why a higher prediction variance between realizations relates to a higher robustness. We verify the validity of our theoretical findings by an extensive empirical analysis.

摘要: 随机神经网络(SNN)是随机函数，预测是通过对该随机函数的多个实现进行平均来获得的。因此，基于一组样本计算对抗性攻击，并将其应用于由另一组样本定义的预测。在本文中，我们通过推导出给定预测过程对计算攻击具有健壮性的一个充分条件来分析这种情况下的稳健性。这使我们能够确定导致SNN稳健性增强的因素，并有助于解释方差和样本量的影响。在其他方面，我们的理论分析揭示了(I)为什么增加用于估计对抗性例子的样本量会增加攻击的强度，(Ii)为什么在推理过程中减少样本大小几乎不会影响稳健性，以及(Iii)为什么实现之间的预测方差越大，就会有越高的稳健性。我们通过广泛的实证分析验证了我们的理论发现的有效性。



## **33. Defending Black-box Skeleton-based Human Activity Classifiers**

基于黑盒骨架防御的人类活动分类器 cs.CV

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2203.04713v2)

**Authors**: He Wang, Yunfeng Diao, Zichang Tan, Guodong Guo

**Abstracts**: Deep learning has been regarded as the `go to' solution for many tasks today, but its intrinsic vulnerability to malicious attacks has become a major concern. The vulnerability is affected by a variety of factors including models, tasks, data, and attackers. Consequently, methods such as Adversarial Training and Randomized Smoothing have been proposed to tackle the problem in a wide range of applications. In this paper, we investigate skeleton-based Human Activity Recognition, which is an important type of time-series data but under-explored in defense against attacks. Our method is featured by (1) a new Bayesian Energy-based formulation of robust discriminative classifiers, (2) a new parameterization of the adversarial sample manifold of actions, and (3) a new post-train Bayesian treatment on both the adversarial samples and the classifier. We name our framework Bayesian Energy-based Adversarial Training or BEAT. BEAT is straightforward but elegant, which turns vulnerable black-box classifiers into robust ones without sacrificing accuracy. It demonstrates surprising and universal effectiveness across a wide range of action classifiers and datasets, under various attacks.

摘要: 深度学习一直被认为是当今许多任务的“首选”解决方案，但其固有的易受恶意攻击的脆弱性已成为一个主要问题。该漏洞受到多种因素的影响，包括模型、任务、数据和攻击者。因此，提出了对抗性训练和随机平滑等方法来解决这一问题，并得到了广泛的应用。在本文中，我们研究了基于骨架的人类活动识别，这是一种重要的时间序列数据类型，但在防御攻击方面还没有得到充分的探索。我们的方法的特点是(1)新的基于贝叶斯能量的稳健判别分类器的公式，(2)对抗性样本动作流形的新的参数化，以及(3)对对抗性样本和分类器的新的训练后贝叶斯处理。我们将我们的框架命名为基于贝叶斯能量的对抗性训练或BEAT。BEAT是简单但优雅的，它将脆弱的黑匣子分类器变成了健壮的分类器，而不会牺牲准确性。它在各种攻击下，在广泛的动作分类器和数据集上展示了令人惊讶和普遍的有效性。



## **34. Enhancing the Transferability via Feature-Momentum Adversarial Attack**

通过特征-动量对抗性攻击增强可转移性 cs.CV

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2204.10606v1)

**Authors**: Xianglong, Yuezun Li, Haipeng Qu, Junyu Dong

**Abstracts**: Transferable adversarial attack has drawn increasing attention due to their practical threaten to real-world applications. In particular, the feature-level adversarial attack is one recent branch that can enhance the transferability via disturbing the intermediate features. The existing methods usually create a guidance map for features, where the value indicates the importance of the corresponding feature element and then employs an iterative algorithm to disrupt the features accordingly. However, the guidance map is fixed in existing methods, which can not consistently reflect the behavior of networks as the image is changed during iteration. In this paper, we describe a new method called Feature-Momentum Adversarial Attack (FMAA) to further improve transferability. The key idea of our method is that we estimate a guidance map dynamically at each iteration using momentum to effectively disturb the category-relevant features. Extensive experiments demonstrate that our method significantly outperforms other state-of-the-art methods by a large margin on different target models.

摘要: 可转移敌意攻击由于其对现实世界应用的实际威胁而受到越来越多的关注。特别是，特征级对抗性攻击是最近的一个分支，它可以通过干扰中间特征来增强可转移性。现有的方法通常为特征创建一个导引地图，其中的值表示对应的特征元素的重要性，然后采用迭代算法对特征进行相应的破坏。然而，现有方法中导航地图是固定的，当图像在迭代过程中发生变化时，不能一致地反映网络的行为。在本文中，我们描述了一种新的方法，称为特征-动量对抗攻击(FMAA)，以进一步提高可转移性。该方法的核心思想是在每一次迭代中使用动量来动态估计导航图，以有效地干扰与类别相关的特征。大量的实验表明，在不同的目标模型上，我们的方法比其他最先进的方法有很大的优势。



## **35. Data-Efficient Backdoor Attacks**

数据高效的后门攻击 cs.CV

Accepted to IJCAI 2022 Long Oral

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2204.12281v1)

**Authors**: Pengfei Xia, Ziqiang Li, Wei Zhang, Bin Li

**Abstracts**: Recent studies have proven that deep neural networks are vulnerable to backdoor attacks. Specifically, by mixing a small number of poisoned samples into the training set, the behavior of the trained model can be maliciously controlled. Existing attack methods construct such adversaries by randomly selecting some clean data from the benign set and then embedding a trigger into them. However, this selection strategy ignores the fact that each poisoned sample contributes inequally to the backdoor injection, which reduces the efficiency of poisoning. In this paper, we formulate improving the poisoned data efficiency by the selection as an optimization problem and propose a Filtering-and-Updating Strategy (FUS) to solve it. The experimental results on CIFAR-10 and ImageNet-10 indicate that the proposed method is effective: the same attack success rate can be achieved with only 47% to 75% of the poisoned sample volume compared to the random selection strategy. More importantly, the adversaries selected according to one setting can generalize well to other settings, exhibiting strong transferability.

摘要: 最近的研究证明，深度神经网络很容易受到后门攻击。具体地说，通过将少量有毒样本混合到训练集中，可以恶意控制训练模型的行为。现有的攻击方法通过从良性集合中随机选择一些干净的数据，然后在其中嵌入触发器来构建这样的攻击者。然而，这种选择策略忽略了这样一个事实，即每个有毒样本对后门注入的贡献是不相等的，这降低了中毒的效率。在本文中，我们将通过选择来提高有毒数据效率的问题描述为一个优化问题，并提出了一种过滤和更新策略(FUS)来解决该问题。在CIFAR-10和ImageNet-10上的实验结果表明，该方法是有效的：与随机选择策略相比，只需47%~75%的中毒样本量即可获得相同的攻击成功率。更重要的是，根据一种设置选择的对手可以很好地推广到其他设置，表现出很强的可转移性。



## **36. Improving the Robustness of Adversarial Attacks Using an Affine-Invariant Gradient Estimator**

利用仿射不变梯度估计提高敌方攻击的稳健性 cs.CV

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2109.05820v2)

**Authors**: Wenzhao Xiang, Hang Su, Chang Liu, Yandong Guo, Shibao Zheng

**Abstracts**: As designers of artificial intelligence try to outwit hackers, both sides continue to hone in on AI's inherent vulnerabilities. Designed and trained from certain statistical distributions of data, AI's deep neural networks (DNNs) remain vulnerable to deceptive inputs that violate a DNN's statistical, predictive assumptions. Before being fed into a neural network, however, most existing adversarial examples cannot maintain malicious functionality when applied to an affine transformation. For practical purposes, maintaining that malicious functionality serves as an important measure of the robustness of adversarial attacks. To help DNNs learn to defend themselves more thoroughly against attacks, we propose an affine-invariant adversarial attack, which can consistently produce more robust adversarial examples over affine transformations. For efficiency, we propose to disentangle current affine-transformation strategies from the Euclidean geometry coordinate plane with its geometric translations, rotations and dilations; we reformulate the latter two in polar coordinates. Afterwards, we construct an affine-invariant gradient estimator by convolving the gradient at the original image with derived kernels, which can be integrated with any gradient-based attack methods. Extensive experiments on ImageNet, including some experiments under physical condition, demonstrate that our method can significantly improve the affine invariance of adversarial examples and, as a byproduct, improve the transferability of adversarial examples, compared with alternative state-of-the-art methods.

摘要: 在人工智能设计者试图智取黑客的同时，双方都在继续钻研人工智能固有的弱点。人工智能的深度神经网络(DNN)是根据数据的某些统计分布设计和训练的，它仍然容易受到违反DNN统计预测假设的欺骗性输入的影响。然而，在输入到神经网络之前，大多数现有的对抗性例子在应用于仿射变换时无法保持恶意功能。出于实际目的，保持恶意功能是衡量敌意攻击健壮性的重要指标。为了帮助DNN学习更彻底地防御攻击，我们提出了一种仿射不变的对抗性攻击，它可以一致地产生比仿射变换更健壮的对抗性例子。为了提高效率，我们建议将当前的仿射变换策略从欧几里德几何坐标平面及其几何平移、旋转和伸缩中分离出来；我们将后两者重新表述在极坐标中。然后，我们通过将原始图像上的梯度与派生核进行卷积来构造仿射不变梯度估计器，该估计器可以与任何基于梯度的攻击方法相结合。在ImageNet上的大量实验，包括一些物理条件下的实验，表明我们的方法可以显著地提高对抗性例子的仿射不变性，并且作为副产品，与其他最新的方法相比，提高了对抗性例子的可转移性。



## **37. Real-Time Detectors for Digital and Physical Adversarial Inputs to Perception Systems**

感知系统的数字和物理敌方输入的实时检测器 cs.CV

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2002.09792v2)

**Authors**: Yiannis Kantaros, Taylor Carpenter, Kaustubh Sridhar, Yahan Yang, Insup Lee, James Weimer

**Abstracts**: Deep neural network (DNN) models have proven to be vulnerable to adversarial digital and physical attacks. In this paper, we propose a novel attack- and dataset-agnostic and real-time detector for both types of adversarial inputs to DNN-based perception systems. In particular, the proposed detector relies on the observation that adversarial images are sensitive to certain label-invariant transformations. Specifically, to determine if an image has been adversarially manipulated, the proposed detector checks if the output of the target classifier on a given input image changes significantly after feeding it a transformed version of the image under investigation. Moreover, we show that the proposed detector is computationally-light both at runtime and design-time which makes it suitable for real-time applications that may also involve large-scale image domains. To highlight this, we demonstrate the efficiency of the proposed detector on ImageNet, a task that is computationally challenging for the majority of relevant defenses, and on physically attacked traffic signs that may be encountered in real-time autonomy applications. Finally, we propose the first adversarial dataset, called AdvNet that includes both clean and physical traffic sign images. Our extensive comparative experiments on the MNIST, CIFAR10, ImageNet, and AdvNet datasets show that VisionGuard outperforms existing defenses in terms of scalability and detection performance. We have also evaluated the proposed detector on field test data obtained on a moving vehicle equipped with a perception-based DNN being under attack.

摘要: 深度神经网络(DNN)模型已被证明容易受到敌意的数字和物理攻击。在本文中，我们提出了一种新的攻击和数据集不可知的实时检测器，用于基于DNN的感知系统的两种类型的敌意输入。特别是，所提出的检测器依赖于观察到的对抗性图像对某些标签不变变换是敏感的。具体地说，为了确定图像是否被恶意操纵，所提出的检测器检查在向目标分类器提供被调查图像的变换版本后，目标分类器在给定输入图像上的输出是否发生显著变化。此外，我们证明了所提出的检测器在运行时和设计时都是计算轻量级的，这使得它适合于也可能涉及大规模图像域的实时应用。为了突出这一点，我们在ImageNet上展示了所提出的检测器的效率，对于大多数相关防御来说，这是一项计算上具有挑战性的任务，以及在实时自主应用中可能遇到的物理攻击的交通标志上。最后，我们提出了第一个对抗性数据集，称为AdvNet，它包括干净的和物理的交通标志图像。我们在MNIST、CIFAR10、ImageNet和AdvNet数据集上的广泛比较实验表明，VisionGuard在可扩展性和检测性能方面优于现有的防御系统。我们还根据现场测试数据对所提出的检测器进行了评估，该检测器是在一辆安装了基于感知的DNN的移动车辆上被攻击的。



## **38. Adversarial Contrastive Learning by Permuting Cluster Assignments**

基于置换类分配的对抗性对比学习 cs.LG

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.10314v1)

**Authors**: Muntasir Wahed, Afrina Tabassum, Ismini Lourentzou

**Abstracts**: Contrastive learning has gained popularity as an effective self-supervised representation learning technique. Several research directions improve traditional contrastive approaches, e.g., prototypical contrastive methods better capture the semantic similarity among instances and reduce the computational burden by considering cluster prototypes or cluster assignments, while adversarial instance-wise contrastive methods improve robustness against a variety of attacks. To the best of our knowledge, no prior work jointly considers robustness, cluster-wise semantic similarity and computational efficiency. In this work, we propose SwARo, an adversarial contrastive framework that incorporates cluster assignment permutations to generate representative adversarial samples. We evaluate SwARo on multiple benchmark datasets and against various white-box and black-box attacks, obtaining consistent improvements over state-of-the-art baselines.

摘要: 对比学习作为一种有效的自我监督表征学习技术已经得到了广泛的应用。一些研究方向改进了传统的对比方法，如原型对比方法更好地捕捉实例之间的语义相似性，并通过考虑簇原型或簇分配来减少计算负担，而对抗性实例对比方法提高了对各种攻击的健壮性。就我们所知，以前的工作没有同时考虑稳健性、聚类语义相似度和计算效率。在这项工作中，我们提出了SwARo，这是一个对抗性对比框架，它结合了簇分配排列来生成具有代表性的对抗性样本。我们在多个基准数据集上对SwARo进行评估，并针对各种白盒和黑盒攻击进行评估，在最先进的基线上获得持续的改进。



## **39. A Mask-Based Adversarial Defense Scheme**

一种基于面具的对抗性防御方案 cs.LG

7 pages

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.11837v1)

**Authors**: Weizhen Xu, Chenyi Zhang, Fangzhen Zhao, Liangda Fang

**Abstracts**: Adversarial attacks hamper the functionality and accuracy of Deep Neural Networks (DNNs) by meddling with subtle perturbations to their inputs.In this work, we propose a new Mask-based Adversarial Defense scheme (MAD) for DNNs to mitigate the negative effect from adversarial attacks. To be precise, our method promotes the robustness of a DNN by randomly masking a portion of potential adversarial images, and as a result, the %classification result output of the DNN becomes more tolerant to minor input perturbations. Compared with existing adversarial defense techniques, our method does not need any additional denoising structure, nor any change to a DNN's design. We have tested this approach on a collection of DNN models for a variety of data sets, and the experimental results confirm that the proposed method can effectively improve the defense abilities of the DNNs against all of the tested adversarial attack methods. In certain scenarios, the DNN models trained with MAD have improved classification accuracy by as much as 20% to 90% compared to the original models that are given adversarial inputs.

摘要: 对抗性攻击通过干扰深层神经网络(DNNS)的输入干扰其功能和准确性，提出了一种新的基于掩码的DNN对抗性防御方案(MAD)，以缓解对抗性攻击带来的负面影响。准确地说，我们的方法通过随机掩蔽一部分潜在的敌意图像来提高DNN的稳健性，从而使DNN的分类结果输出对微小的输入扰动具有更强的容错性。与现有的对抗性防御技术相比，该方法不需要任何额外的去噪结构，也不需要对DNN的设计进行任何改变。我们在各种数据集的DNN模型上对该方法进行了测试，实验结果证实，该方法能够有效地提高DNN对所有测试的对抗性攻击的防御能力。在某些场景中，与给出对抗性输入的原始模型相比，用MAD训练的DNN模型的分类准确率提高了20%到90%。



## **40. Robustness of Machine Learning Models Beyond Adversarial Attacks**

对抗攻击下机器学习模型的稳健性 cs.LG

25 pages, 7 figures

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.10046v1)

**Authors**: Sebastian Scher, Andreas Trügler

**Abstracts**: Correctly quantifying the robustness of machine learning models is a central aspect in judging their suitability for specific tasks, and thus, ultimately, for generating trust in the models. We show that the widely used concept of adversarial robustness and closely related metrics based on counterfactuals are not necessarily valid metrics for determining the robustness of ML models against perturbations that occur "naturally", outside specific adversarial attack scenarios. Additionally, we argue that generic robustness metrics in principle are insufficient for determining real-world-robustness. Instead we propose a flexible approach that models possible perturbations in input data individually for each application. This is then combined with a probabilistic approach that computes the likelihood that a real-world perturbation will change a prediction, thus giving quantitative information of the robustness of the trained machine learning model. The method does not require access to the internals of the classifier and thus in principle works for any black-box model. It is, however, based on Monte-Carlo sampling and thus only suited for input spaces with small dimensions. We illustrate our approach on two dataset, as well as on analytically solvable cases. Finally, we discuss ideas on how real-world robustness could be computed or estimated in high-dimensional input spaces.

摘要: 正确量化机器学习模型的稳健性是判断它们是否适合特定任务的一个中心方面，从而最终产生对模型的信任。我们表明，广泛使用的对抗性健壮性概念和基于反事实的密切相关的度量标准，并不一定是确定ML模型对特定对抗性攻击场景外的“自然”扰动的健壮性的有效度量。此外，我们认为一般的健壮性度量原则上不足以确定真实世界的健壮性。相反，我们提出了一种灵活的方法，为每个应用程序分别建模输入数据中可能的扰动。然后，将其与计算真实世界扰动将改变预测的可能性的概率方法相结合，从而给出训练后的机器学习模型的稳健性的定量信息。该方法不需要访问分类器的内部，因此原则上适用于任何黑盒模型。然而，它是基于蒙特卡罗抽样的，因此只适用于小维度的输入空间。我们在两个数据集上以及在分析可解的情况下说明了我们的方法。最后，我们讨论了如何在高维输入空间中计算或估计真实世界的稳健性。



## **41. Is Neuron Coverage Needed to Make Person Detection More Robust?**

需要神经元覆盖才能使人检测更可靠吗？ cs.CV

Accepted for publication at CVPR 2022 TCV workshop

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.10027v1)

**Authors**: Svetlana Pavlitskaya, Şiyar Yıkmış, J. Marius Zöllner

**Abstracts**: The growing use of deep neural networks (DNNs) in safety- and security-critical areas like autonomous driving raises the need for their systematic testing. Coverage-guided testing (CGT) is an approach that applies mutation or fuzzing according to a predefined coverage metric to find inputs that cause misbehavior. With the introduction of a neuron coverage metric, CGT has also recently been applied to DNNs. In this work, we apply CGT to the task of person detection in crowded scenes. The proposed pipeline uses YOLOv3 for person detection and includes finding DNN bugs via sampling and mutation, and subsequent DNN retraining on the updated training set. To be a bug, we require a mutated image to cause a significant performance drop compared to a clean input. In accordance with the CGT, we also consider an additional requirement of increased coverage in the bug definition. In order to explore several types of robustness, our approach includes natural image transformations, corruptions, and adversarial examples generated with the Daedalus attack. The proposed framework has uncovered several thousand cases of incorrect DNN behavior. The relative change in mAP performance of the retrained models reached on average between 26.21\% and 64.24\% for different robustness types. However, we have found no evidence that the investigated coverage metrics can be advantageously used to improve robustness.

摘要: 深度神经网络(DNN)在自动驾驶等安全和安保关键领域的应用越来越多，这增加了对其进行系统测试的必要性。覆盖引导测试(CGT)是一种根据预定义的覆盖度量应用突变或模糊来发现导致错误行为的输入的方法。随着神经元覆盖度量的引入，CGT最近也被应用于DNN。在这项工作中，我们将CGT应用于拥挤场景中的人检测任务。拟议的管道使用YOLOv3进行人员检测，包括通过采样和突变发现DNN错误，以及随后在更新的训练集上对DNN进行再培训。要成为一个错误，我们需要一个突变的图像来导致与干净的输入相比显著的性能下降。根据CGT，我们还考虑在错误定义中增加覆盖范围的额外要求。为了探索几种类型的健壮性，我们的方法包括自然图像转换、损坏和由Daedalus攻击生成的敌意示例。所提出的框架已经发现了数千例不正确的DNN行为。对于不同的稳健性类型，重新训练的模型的MAP性能的相对变化平均在26.21~64.24之间。然而，我们没有发现证据表明所调查的覆盖度量可以被有利地用于提高稳健性。



## **42. On the Certified Robustness for Ensemble Models and Beyond**

关于系综模型及以后模型的认证稳健性 cs.LG

ICLR 2022. 51 pages, 10 pages for main text. Forum and code:  https://openreview.net/forum?id=tUa4REjGjTf

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2107.10873v2)

**Authors**: Zhuolin Yang, Linyi Li, Xiaojun Xu, Bhavya Kailkhura, Tao Xie, Bo Li

**Abstracts**: Recent studies show that deep neural networks (DNN) are vulnerable to adversarial examples, which aim to mislead DNNs by adding perturbations with small magnitude. To defend against such attacks, both empirical and theoretical defense approaches have been extensively studied for a single ML model. In this work, we aim to analyze and provide the certified robustness for ensemble ML models, together with the sufficient and necessary conditions of robustness for different ensemble protocols. Although ensemble models are shown more robust than a single model empirically; surprisingly, we find that in terms of the certified robustness the standard ensemble models only achieve marginal improvement compared to a single model. Thus, to explore the conditions that guarantee to provide certifiably robust ensemble ML models, we first prove that diversified gradient and large confidence margin are sufficient and necessary conditions for certifiably robust ensemble models under the model-smoothness assumption. We then provide the bounded model-smoothness analysis based on the proposed Ensemble-before-Smoothing strategy. We also prove that an ensemble model can always achieve higher certified robustness than a single base model under mild conditions. Inspired by the theoretical findings, we propose the lightweight Diversity Regularized Training (DRT) to train certifiably robust ensemble ML models. Extensive experiments show that our DRT enhanced ensembles can consistently achieve higher certified robustness than existing single and ensemble ML models, demonstrating the state-of-the-art certified L2-robustness on MNIST, CIFAR-10, and ImageNet datasets.

摘要: 最近的研究表明，深度神经网络(DNN)很容易受到敌意例子的影响，这些例子旨在通过添加小幅度的扰动来误导DNN。为了防御这样的攻击，针对单个ML模型，已经广泛地研究了经验和理论防御方法。在这项工作中，我们旨在分析和提供集成ML模型的证明的稳健性，以及对于不同的集成协议的健壮性的充要条件。虽然从经验上看，集成模型比单个模型更稳健，但令人惊讶的是，我们发现标准集成模型在验证的稳健性方面，与单个模型相比只有轻微的改善。因此，为了探索保证提供可证明鲁棒的集成ML模型的条件，我们首先证明了在模型光滑性假设下，多样化的梯度和大的置信度是可证明鲁棒的集成模型的充要条件。然后，基于所提出的先集成后平滑策略，给出了有界模型光滑性分析。我们还证明了在温和的条件下，集成模型总是可以获得比单一基础模型更高的认证稳健性。受理论研究的启发，我们提出了轻量级多样性正则化训练(DRT)来训练可证明稳健的ML集成模型。广泛的实验表明，我们的DRT增强型集成可以持续实现比现有单一和集成ML模型更高的认证稳健性，在MNIST、CIFAR-10和ImageNet数据集上展示了最先进的认证L2稳健性。



## **43. Fast AdvProp**

Fast AdvProp cs.CV

ICLR 2022 camera ready version

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.09838v1)

**Authors**: Jieru Mei, Yucheng Han, Yutong Bai, Yixiao Zhang, Yingwei Li, Xianhang Li, Alan Yuille, Cihang Xie

**Abstracts**: Adversarial Propagation (AdvProp) is an effective way to improve recognition models, leveraging adversarial examples. Nonetheless, AdvProp suffers from the extremely slow training speed, mainly because: a) extra forward and backward passes are required for generating adversarial examples; b) both original samples and their adversarial counterparts are used for training (i.e., 2$\times$ data). In this paper, we introduce Fast AdvProp, which aggressively revamps AdvProp's costly training components, rendering the method nearly as cheap as the vanilla training. Specifically, our modifications in Fast AdvProp are guided by the hypothesis that disentangled learning with adversarial examples is the key for performance improvements, while other training recipes (e.g., paired clean and adversarial training samples, multi-step adversarial attackers) could be largely simplified.   Our empirical results show that, compared to the vanilla training baseline, Fast AdvProp is able to further model performance on a spectrum of visual benchmarks, without incurring extra training cost. Additionally, our ablations find Fast AdvProp scales better if larger models are used, is compatible with existing data augmentation methods (i.e., Mixup and CutMix), and can be easily adapted to other recognition tasks like object detection. The code is available here: https://github.com/meijieru/fast_advprop.

摘要: 对抗性传播(AdvProp)是利用对抗性例子改进识别模型的一种有效方法。然而，AdvProp的训练速度非常慢，主要是因为：a)需要额外的向前和向后传递来生成对抗性示例；b)原始样本和它们的对应物都用于训练(即2$\x$数据)。在本文中，我们介绍了Fast AdvProp，它积极地改造了AdvProp昂贵的训练组件，使该方法几乎与普通训练一样便宜。具体地说，我们在Fast AdvProp中的修改是在这样一个假设的指导下进行的，即与对抗性例子的分离学习是性能提高的关键，而其他训练食谱(例如，成对的干净和对抗性训练样本、多步骤对抗性攻击者)可以在很大程度上被简化。我们的实验结果表明，与普通的训练基准相比，Fast AdvProp能够在不产生额外训练成本的情况下，在一系列视觉基准上进一步建模性能。此外，我们的烧蚀发现，如果使用更大的模型，Fast AdvProp的规模会更好，与现有的数据增强方法(即Mixup和CutMix)兼容，并且可以很容易地适应其他识别任务，如目标检测。代码可在此处获得：https://github.com/meijieru/fast_advprop.



## **44. GUARD: Graph Universal Adversarial Defense**

后卫：GRAPH通用对抗性防御 cs.LG

Code is publicly available at https://github.com/EdisonLeeeee/GUARD

**SubmitDate**: 2022-04-20    [paper-pdf](http://arxiv.org/pdf/2204.09803v1)

**Authors**: Jintang Li, Jie Liao, Ruofan Wu, Liang Chen, Changhua Meng, Zibin Zheng, Weiqiang Wang

**Abstracts**: Recently, graph convolutional networks (GCNs) have shown to be vulnerable to small adversarial perturbations, which becomes a severe threat and largely limits their applications in security-critical scenarios. To mitigate such a threat, considerable research efforts have been devoted to increasing the robustness of GCNs against adversarial attacks. However, current approaches for defense are typically designed for the whole graph and consider the global performance, posing challenges in protecting important local nodes from stronger adversarial targeted attacks. In this work, we present a simple yet effective method, named \textbf{\underline{G}}raph \textbf{\underline{U}}niversal \textbf{\underline{A}}dve\textbf{\underline{R}}sarial \textbf{\underline{D}}efense (GUARD). Unlike previous works, GUARD protects each individual node from attacks with a universal defensive patch, which is generated once and can be applied to any node (node-agnostic) in a graph. Extensive experiments on four benchmark datasets demonstrate that our method significantly improves robustness for several established GCNs against multiple adversarial attacks and outperforms existing adversarial defense methods by large margins. Our code is publicly available at https://github.com/EdisonLeeeee/GUARD.

摘要: 最近，图卷积网络(GCNS)被证明容易受到微小的敌意扰动，这成为一种严重的威胁，并在很大程度上限制了它们在安全关键场景中的应用。为了减轻这种威胁，人们投入了大量的研究努力来提高GCNS对对手攻击的健壮性。然而，当前的防御方法通常是为整个图设计的，并考虑了全局性能，这给保护重要的局部节点免受更强的对抗性目标攻击带来了挑战。在这项工作中，我们提出了一种简单而有效的方法，命名为\Textbf{\下划线{G}}RAPH\Textbf{\下划线{U}}通用\textbf{\underline{A}}dve\textbf{\underline{R}}sarial\Textbf{\下划线{D}}保护。与以前的工作不同，Guard使用一个通用的防御补丁来保护每个单独的节点免受攻击，该补丁只生成一次，可以应用于图中的任何节点(与节点无关)。在四个基准数据集上的大量实验表明，我们的方法显著提高了几个已建立的GCN对多个对手攻击的稳健性，并大大超过了现有的对抗性防御方法。我们的代码在https://github.com/EdisonLeeeee/GUARD.上公开提供



## **45. Backdooring Explainable Machine Learning**

Backdoding可解释机器学习 cs.CR

**SubmitDate**: 2022-04-20    [paper-pdf](http://arxiv.org/pdf/2204.09498v1)

**Authors**: Maximilian Noppel, Lukas Peter, Christian Wressnegger

**Abstracts**: Explainable machine learning holds great potential for analyzing and understanding learning-based systems. These methods can, however, be manipulated to present unfaithful explanations, giving rise to powerful and stealthy adversaries. In this paper, we demonstrate blinding attacks that can fully disguise an ongoing attack against the machine learning model. Similar to neural backdoors, we modify the model's prediction upon trigger presence but simultaneously also fool the provided explanation. This enables an adversary to hide the presence of the trigger or point the explanation to entirely different portions of the input, throwing a red herring. We analyze different manifestations of such attacks for different explanation types in the image domain, before we resume to conduct a red-herring attack against malware classification.

摘要: 可解释机器学习在分析和理解基于学习的系统方面具有巨大的潜力。然而，这些方法可以被操纵来提供不可信的解释，从而产生强大而隐蔽的对手。在本文中，我们演示了盲攻击，可以完全掩盖对机器学习模型的正在进行的攻击。与神经后门类似，我们根据触发器的存在修改模型的预测，但同时也欺骗了所提供的解释。这使得对手可以隐藏触发器的存在，或者将解释指向输入的完全不同的部分，从而转移注意力。我们分析了这类攻击在图像域中针对不同解释类型的不同表现，然后继续进行针对恶意软件分类的转移话题攻击。



## **46. Adversarial Scratches: Deployable Attacks to CNN Classifiers**

对抗性抓痕：对CNN分类器的可部署攻击 cs.LG

This paper stems from 'Scratch that! An Evolution-based Adversarial  Attack against Neural Networks' for which an arXiv preprint is available at  arXiv:1912.02316. Further studies led to a complete overhaul of the work,  resulting in this paper. This work was submitted for review in Pattern  Recognition (Elsevier)

**SubmitDate**: 2022-04-20    [paper-pdf](http://arxiv.org/pdf/2204.09397v1)

**Authors**: Loris Giulivi, Malhar Jere, Loris Rossi, Farinaz Koushanfar, Gabriela Ciocarlie, Briland Hitaj, Giacomo Boracchi

**Abstracts**: A growing body of work has shown that deep neural networks are susceptible to adversarial examples. These take the form of small perturbations applied to the model's input which lead to incorrect predictions. Unfortunately, most literature focuses on visually imperceivable perturbations to be applied to digital images that often are, by design, impossible to be deployed to physical targets. We present Adversarial Scratches: a novel L0 black-box attack, which takes the form of scratches in images, and which possesses much greater deployability than other state-of-the-art attacks. Adversarial Scratches leverage B\'ezier Curves to reduce the dimension of the search space and possibly constrain the attack to a specific location. We test Adversarial Scratches in several scenarios, including a publicly available API and images of traffic signs. Results show that, often, our attack achieves higher fooling rate than other deployable state-of-the-art methods, while requiring significantly fewer queries and modifying very few pixels.

摘要: 越来越多的研究表明，深度神经网络很容易受到敌意例子的影响。这些采用的形式是应用于模型输入的小扰动，从而导致不正确的预测。不幸的是，大多数文献关注的是应用于数字图像的视觉上不可察觉的扰动，而根据设计，数字图像通常不可能被部署到物理目标上。我们提出了对抗性划痕：一种新颖的L0黑盒攻击，它采用图像划痕的形式，并且比其他最先进的攻击具有更大的可部署性。对抗性划痕利用B‘ezier曲线来减少搜索空间的维度，并可能将攻击限制在特定位置。我们在几个场景中测试了对抗性划痕，包括公开可用的API和交通标志图像。结果表明，我们的攻击通常比其他可部署的最先进方法获得更高的愚骗率，同时需要的查询和修改的像素也非常少。



## **47. You Are What You Write: Preserving Privacy in the Era of Large Language Models**

你写什么，你就是什么：在大型语言模型时代保护隐私 cs.CL

**SubmitDate**: 2022-04-20    [paper-pdf](http://arxiv.org/pdf/2204.09391v1)

**Authors**: Richard Plant, Valerio Giuffrida, Dimitra Gkatzia

**Abstracts**: Large scale adoption of large language models has introduced a new era of convenient knowledge transfer for a slew of natural language processing tasks. However, these models also run the risk of undermining user trust by exposing unwanted information about the data subjects, which may be extracted by a malicious party, e.g. through adversarial attacks. We present an empirical investigation into the extent of the personal information encoded into pre-trained representations by a range of popular models, and we show a positive correlation between the complexity of a model, the amount of data used in pre-training, and data leakage. In this paper, we present the first wide coverage evaluation and comparison of some of the most popular privacy-preserving algorithms, on a large, multi-lingual dataset on sentiment analysis annotated with demographic information (location, age and gender). The results show since larger and more complex models are more prone to leaking private information, use of privacy-preserving methods is highly desirable. We also find that highly privacy-preserving technologies like differential privacy (DP) can have serious model utility effects, which can be ameliorated using hybrid or metric-DP techniques.

摘要: 大型语言模型的大规模采用为一系列自然语言处理任务引入了一个方便的知识转移的新时代。然而，这些模型也存在通过暴露关于数据主体的不想要的信息来破坏用户信任的风险，这些信息可能由恶意方提取，例如通过对抗性攻击。我们通过一系列流行的模型对个人信息编码到预训练表示中的程度进行了实证研究，结果表明模型的复杂性、预训练中使用的数据量和数据泄露之间存在正相关关系。在本文中，我们首次对一些最流行的隐私保护算法进行了广泛的评估和比较，这些算法是在一个带有人口统计信息(位置、年龄和性别)的大型多语言情感分析数据集上进行的。结果表明，由于更大、更复杂的模型更容易泄露私人信息，因此使用隐私保护方法是非常可取的。我们还发现，像差分隐私(DP)这样的高度隐私保护技术可能会产生严重的模型效用效应，这可以使用混合或度量DP技术来改进。



## **48. Identifying Near-Optimal Single-Shot Attacks on ICSs with Limited Process Knowledge**

利用有限的流程知识识别ICSS上的近最佳单发攻击 cs.CR

This paper has been accepted at Applied Cryptography and Network  Security (ACNS) 2022

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.09106v1)

**Authors**: Herson Esquivel-Vargas, John Henry Castellanos, Marco Caselli, Nils Ole Tippenhauer, Andreas Peter

**Abstracts**: Industrial Control Systems (ICSs) rely on insecure protocols and devices to monitor and operate critical infrastructure. Prior work has demonstrated that powerful attackers with detailed system knowledge can manipulate exchanged sensor data to deteriorate performance of the process, even leading to full shutdowns of plants. Identifying those attacks requires iterating over all possible sensor values, and running detailed system simulation or analysis to identify optimal attacks. That setup allows adversaries to identify attacks that are most impactful when applied on the system for the first time, before the system operators become aware of the manipulations.   In this work, we investigate if constrained attackers without detailed system knowledge and simulators can identify comparable attacks. In particular, the attacker only requires abstract knowledge on general information flow in the plant, instead of precise algorithms, operating parameters, process models, or simulators. We propose an approach that allows single-shot attacks, i.e., near-optimal attacks that are reliably shutting down a system on the first try. The approach is applied and validated on two use cases, and demonstrated to achieve comparable results to prior work, which relied on detailed system information and simulations.

摘要: 工业控制系统(ICSS)依赖不安全的协议和设备来监控和运行关键基础设施。先前的工作已经证明，拥有详细系统知识的强大攻击者可以操纵交换的传感器数据，从而降低过程的性能，甚至导致工厂完全关闭。识别这些攻击需要迭代所有可能的传感器值，并运行详细的系统模拟或分析以确定最佳攻击。这种设置允许攻击者在系统操作员意识到操纵之前，识别首次应用于系统时最具影响力的攻击。在这项工作中，我们调查在没有详细系统知识和模拟器的情况下，受限攻击者是否能够识别类似的攻击。特别是，攻击者只需要工厂中一般信息流的抽象知识，而不是精确的算法、操作参数、过程模型或模拟器。我们提出了一种允许单发攻击的方法，即在第一次尝试时可靠地关闭系统的近最佳攻击。该方法在两个用例上进行了应用和验证，并证明了其结果与之前依赖于详细系统信息和模拟的工作类似。



## **49. Indiscriminate Data Poisoning Attacks on Neural Networks**

对神经网络的不分青红皂白的数据中毒攻击 cs.LG

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.09092v1)

**Authors**: Yiwei Lu, Gautam Kamath, Yaoliang Yu

**Abstracts**: Data poisoning attacks, in which a malicious adversary aims to influence a model by injecting "poisoned" data into the training process, have attracted significant recent attention. In this work, we take a closer look at existing poisoning attacks and connect them with old and new algorithms for solving sequential Stackelberg games. By choosing an appropriate loss function for the attacker and optimizing with algorithms that exploit second-order information, we design poisoning attacks that are effective on neural networks. We present efficient implementations that exploit modern auto-differentiation packages and allow simultaneous and coordinated generation of tens of thousands of poisoned points, in contrast to existing methods that generate poisoned points one by one. We further perform extensive experiments that empirically explore the effect of data poisoning attacks on deep neural networks.

摘要: 数据中毒攻击是指恶意对手通过在训练过程中注入“有毒”数据来影响模型的攻击，最近引起了极大的关注。在这项工作中，我们仔细研究现有的中毒攻击，并将它们与解决连续Stackelberg博弈的旧算法和新算法联系起来。通过为攻击者选择合适的损失函数，并利用二阶信息的算法进行优化，设计出对神经网络有效的中毒攻击。我们提供了利用现代自动区分包并允许同时和协调地生成数万个毒点的高效实现，与逐个生成毒点的现有方法形成对比。我们进一步进行了大量的实验，经验地探索了数据中毒攻击对深度神经网络的影响。



## **50. A Brief Survey on Deep Learning Based Data Hiding**

基于深度学习的数据隐藏研究综述 cs.CR

v2: reorganize some sections and add several new papers published in  2021~2022

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2103.01607v2)

**Authors**: Chaoning Zhang, Chenguo Lin, Philipp Benz, Kejiang Chen, Weiming Zhang, In So Kweon

**Abstracts**: Data hiding is the art of concealing messages with limited perceptual changes. Recently, deep learning has enriched it from various perspectives with significant progress. In this work, we conduct a brief yet comprehensive review of existing literature for deep learning based data hiding (deep hiding) by first classifying it according to three essential properties (i.e., capacity, security and robustness), and outline three commonly used architectures. Based on this, we summarize specific strategies for different applications of data hiding, including basic hiding, steganography, watermarking and light field messaging. Finally, further insight into deep hiding is provided by incorporating the perspective of adversarial attack.

摘要: 数据隐藏是通过有限的感知变化来隐藏消息的艺术。最近，深度学习从多个角度丰富了它，取得了重大进展。在这项工作中，我们对现有的基于深度学习的数据隐藏(深度隐藏)进行了简要而全面的回顾，首先根据三个基本属性(即容量、安全性和健壮性)对其进行分类，并概述了三种常用的体系结构。在此基础上，总结了针对不同应用的数据隐藏的具体策略，包括基本隐藏、隐写、水印和光场消息。最后，通过结合对抗性攻击的视角，对深层隐藏提供了进一步的洞察。



