# Latest Adversarial Attack Papers
**update at 2022-10-31 17:11:58**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Universalization of any adversarial attack using very few test examples**

使用极少的测试示例实现任何对抗性攻击的通用化 cs.LG

Appeared in ACM CODS-COMAD 2022 (Research Track)

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2005.08632v2) [paper-pdf](http://arxiv.org/pdf/2005.08632v2)

**Authors**: Sandesh Kamath, Amit Deshpande, K V Subrahmanyam, Vineeth N Balasubramanian

**Abstract**: Deep learning models are known to be vulnerable not only to input-dependent adversarial attacks but also to input-agnostic or universal adversarial attacks. Dezfooli et al. \cite{Dezfooli17,Dezfooli17anal} construct universal adversarial attack on a given model by looking at a large number of training data points and the geometry of the decision boundary near them. Subsequent work \cite{Khrulkov18} constructs universal attack by looking only at test examples and intermediate layers of the given model. In this paper, we propose a simple universalization technique to take any input-dependent adversarial attack and construct a universal attack by only looking at very few adversarial test examples. We do not require details of the given model and have negligible computational overhead for universalization. We theoretically justify our universalization technique by a spectral property common to many input-dependent adversarial perturbations, e.g., gradients, Fast Gradient Sign Method (FGSM) and DeepFool. Using matrix concentration inequalities and spectral perturbation bounds, we show that the top singular vector of input-dependent adversarial directions on a small test sample gives an effective and simple universal adversarial attack. For VGG16 and VGG19 models trained on ImageNet, our simple universalization of Gradient, FGSM, and DeepFool perturbations using a test sample of 64 images gives fooling rates comparable to state-of-the-art universal attacks \cite{Dezfooli17,Khrulkov18} for reasonable norms of perturbation. Code available at https://github.com/ksandeshk/svd-uap .

摘要: 众所周知，深度学习模型不仅容易受到依赖输入的对抗性攻击，而且还容易受到输入不可知的或普遍的对抗性攻击。德兹戈尼等人。{Dezfooli17，Dezfooli17anal}通过查看大量的训练数据点和它们附近的决策边界的几何形状来构造对给定模型的通用对抗性攻击。后续工作{Khrulkov18}通过只关注给定模型的测试用例和中间层来构造通用攻击。在本文中，我们提出了一种简单的普适化技术，可以接受任何依赖于输入的对抗性攻击，并且只需查看极少的对抗性测试实例就可以构建通用攻击。我们不需要给定模型的细节，并且通用性的计算开销可以忽略不计。我们从理论上证明了我们的普适化技术是由许多依赖于输入的对抗性扰动所共有的谱性质来证明的，例如梯度、快速梯度符号方法(FGSM)和DeepFool。利用矩阵集中不等式和谱摄动界，我们证明了在小样本上依赖于输入的对抗方向的顶部奇异向量给出了一种有效且简单的通用对抗攻击。对于在ImageNet上训练的VGG16和VGG19模型，我们使用64幅图像的测试样本对梯度、FGSM和DeepFool扰动的简单通用化提供了与最先进的通用攻击相当的愚弄率\引用{Dezfooli17，Khrulkov18}的合理扰动规范。代码可在https://github.com/ksandeshk/svd-uap上找到。



## **2. Local Model Reconstruction Attacks in Federated Learning and their Uses**

联合学习中的局部模型重构攻击及其应用 cs.LG

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2210.16205v1) [paper-pdf](http://arxiv.org/pdf/2210.16205v1)

**Authors**: Ilias Driouich, Chuan Xu, Giovanni Neglia, Frederic Giroire, Eoin Thomas

**Abstract**: In this paper, we initiate the study of local model reconstruction attacks for federated learning, where a honest-but-curious adversary eavesdrops the messages exchanged between a targeted client and the server, and then reconstructs the local/personalized model of the victim. The local model reconstruction attack allows the adversary to trigger other classical attacks in a more effective way, since the local model only depends on the client's data and can leak more private information than the global model learned by the server. Additionally, we propose a novel model-based attribute inference attack in federated learning leveraging the local model reconstruction attack. We provide an analytical lower-bound for this attribute inference attack. Empirical results using real world datasets confirm that our local reconstruction attack works well for both regression and classification tasks. Moreover, we benchmark our novel attribute inference attack against the state-of-the-art attacks in federated learning. Our attack results in higher reconstruction accuracy especially when the clients' datasets are heterogeneous. Our work provides a new angle for designing powerful and explainable attacks to effectively quantify the privacy risk in FL.

摘要: 在本文中，我们发起了联合学习的本地模型重建攻击的研究，其中诚实但好奇的攻击者窃听目标客户端和服务器之间交换的消息，然后重建受害者的本地/个性化模型。本地模型重构攻击允许攻击者以更有效的方式触发其他经典攻击，因为本地模型仅依赖于客户端的数据，并且可以比服务器学习的全局模型泄露更多的私人信息。此外，利用局部模型重构攻击，提出了一种新的联邦学习中基于模型的属性推理攻击。我们给出了这种属性推理攻击的一个分析下界。使用真实世界数据集的实验结果证实，我们的局部重建攻击对于回归和分类任务都很好地工作。此外，我们还对联邦学习中最新的属性推理攻击进行了基准测试。我们的攻击导致了更高的重建精度，特别是当客户的数据集是异质的时候。我们的工作为设计强大的、可解释的攻击以有效量化FL中的隐私风险提供了一个新的角度。



## **3. Improving Transferability of Adversarial Examples on Face Recognition with Beneficial Perturbation Feature Augmentation**

利用有益扰动特征增强提高人脸识别中对抗性样本的可转移性 cs.CV

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2210.16117v1) [paper-pdf](http://arxiv.org/pdf/2210.16117v1)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Zongyi Li, Qian Wang

**Abstract**: Face recognition (FR) models can be easily fooled by adversarial examples, which are crafted by adding imperceptible perturbations on benign face images. To improve the transferability of adversarial examples on FR models, we propose a novel attack method called Beneficial Perturbation Feature Augmentation Attack (BPFA), which reduces the overfitting of the adversarial examples to surrogate FR models by the adversarial strategy. Specifically, in the backpropagation step, BPFA records the gradients on pre-selected features and uses the gradient on the input image to craft adversarial perturbation to be added on the input image. In the next forward propagation step, BPFA leverages the recorded gradients to add perturbations(i.e., beneficial perturbations) that can be pitted against the adversarial perturbation added on the input image on their corresponding features. The above two steps are repeated until the last backpropagation step before the maximum number of iterations is reached. The optimization process of the adversarial perturbation added on the input image and the optimization process of the beneficial perturbations added on the features correspond to a minimax two-player game. Extensive experiments demonstrate that BPFA outperforms the state-of-the-art gradient-based adversarial attacks on FR.

摘要: 人脸识别(FR)模型很容易被敌意的例子所愚弄，这些例子是通过在良性的人脸图像上添加难以察觉的扰动来构建的。为了提高对抗实例在FR模型上的可转移性，我们提出了一种新的攻击方法，称为有益扰动特征增强攻击(BPFA)，它通过对抗策略减少了对抗实例对替代FR模型的过度拟合。具体地说，在反向传播步骤中，BPFA记录预先选择的特征的梯度，并使用输入图像上的梯度来构造要添加到输入图像上的对抗性扰动。在下一前向传播步骤中，BPFA利用记录的梯度来添加扰动(即，有益扰动)，该扰动可以相对于添加在输入图像上的对抗性扰动添加到其相应的特征上。重复上述两个步骤，直到达到最大迭代次数之前的最后一个反向传播步骤。添加在输入图像上的对抗性扰动的优化过程和添加在特征上的有益扰动的优化过程对应于极小极大两人博弈。大量实验表明，BPFA的性能优于目前最先进的基于梯度的对抗性FR攻击。



## **4. Watermarking Graph Neural Networks based on Backdoor Attacks**

基于后门攻击的数字水印图神经网络 cs.LG

18 pages, 9 figures

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2110.11024v4) [paper-pdf](http://arxiv.org/pdf/2110.11024v4)

**Authors**: Jing Xu, Stefanos Koffas, Oguzhan Ersoy, Stjepan Picek

**Abstract**: Graph Neural Networks (GNNs) have achieved promising performance in various real-world applications. Building a powerful GNN model is not a trivial task, as it requires a large amount of training data, powerful computing resources, and human expertise in fine-tuning the model. Moreover, with the development of adversarial attacks, e.g., model stealing attacks, GNNs raise challenges to model authentication. To avoid copyright infringement on GNNs, verifying the ownership of the GNN models is necessary.   This paper presents a watermarking framework for GNNs for both graph and node classification tasks. We 1) design two strategies to generate watermarked data for the graph classification task and one for the node classification task, 2) embed the watermark into the host model through training to obtain the watermarked GNN model, and 3) verify the ownership of the suspicious model in a black-box setting. The experiments show that our framework can verify the ownership of GNN models with a very high probability (up to $99\%$) for both tasks. Finally, we experimentally show that our watermarking approach is robust against a state-of-the-art model extraction technique and four state-of-the-art defenses against backdoor attacks.

摘要: 图神经网络(GNN)在各种实际应用中取得了良好的性能。构建一个强大的GNN模型不是一项简单的任务，因为它需要大量的训练数据、强大的计算资源和微调模型的人力专业知识。此外，随着敌意攻击的发展，例如模型窃取攻击，GNN对模型认证提出了挑战。为了避免对GNN的版权侵权，有必要核实GNN模型的所有权。本文提出了一种适用于图和节点分类任务的GNN水印框架。我们设计了两种策略来为图分类任务和节点分类任务生成水印数据，2)通过训练将水印嵌入到宿主模型中，得到带水印的GNN模型，3)在黑盒环境下验证可疑模型的所有权。实验表明，我们的框架能够以很高的概率(高达99美元)验证这两个任务的GNN模型的所有权。最后，我们的实验表明，我们的水印方法对于一种最先进的模型提取技术和四种最先进的后门攻击防御方法是健壮的。



## **5. RoChBert: Towards Robust BERT Fine-tuning for Chinese**

RoChBert：为中文走向稳健的BERT微调 cs.CL

Accepted by Findings of EMNLP 2022

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2210.15944v1) [paper-pdf](http://arxiv.org/pdf/2210.15944v1)

**Authors**: Zihan Zhang, Jinfeng Li, Ning Shi, Bo Yuan, Xiangyu Liu, Rong Zhang, Hui Xue, Donghong Sun, Chao Zhang

**Abstract**: Despite of the superb performance on a wide range of tasks, pre-trained language models (e.g., BERT) have been proved vulnerable to adversarial texts. In this paper, we present RoChBERT, a framework to build more Robust BERT-based models by utilizing a more comprehensive adversarial graph to fuse Chinese phonetic and glyph features into pre-trained representations during fine-tuning. Inspired by curriculum learning, we further propose to augment the training dataset with adversarial texts in combination with intermediate samples. Extensive experiments demonstrate that RoChBERT outperforms previous methods in significant ways: (i) robust -- RoChBERT greatly improves the model robustness without sacrificing accuracy on benign texts. Specifically, the defense lowers the success rates of unlimited and limited attacks by 59.43% and 39.33% respectively, while remaining accuracy of 93.30%; (ii) flexible -- RoChBERT can easily extend to various language models to solve different downstream tasks with excellent performance; and (iii) efficient -- RoChBERT can be directly applied to the fine-tuning stage without pre-training language model from scratch, and the proposed data augmentation method is also low-cost.

摘要: 尽管在广泛的任务中表现出色，但预先训练的语言模型(如BERT)已被证明容易受到敌意文本的攻击。在本文中，我们提出了RoChBERT，这是一个框架，通过在微调过程中利用更全面的对抗性图将汉语语音和字形特征融合到预先训练的表示中来建立更健壮的基于ERT的模型。受课程学习的启发，我们进一步提出用对抗性文本结合中间样本来扩充训练数据集。大量的实验表明，RoChBERT在以下几个方面明显优于以往的方法：(I)健壮性--RoChBERT在不牺牲对良性文本的准确性的情况下，大大提高了模型的健壮性。具体来说，防御使无限和有限攻击的成功率分别降低了59.43%和39.33%，同时保持了93.30%的准确率；(Ii)灵活的-RoChBERT可以很容易地扩展到各种语言模型，以优异的性能解决不同的下游任务；(Iii)高效-RoChBERT可以直接应用到微调阶段，而不需要从零开始预先训练语言模型，并且所提出的数据增强方法也是低成本的。



## **6. DICTION: DynamIC robusT whIte bOx watermarkiNg scheme**

动态稳健白盒水印方案 cs.CR

18 pages, 5 figures, PrePrint

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15745v1) [paper-pdf](http://arxiv.org/pdf/2210.15745v1)

**Authors**: Reda Bellafqira, Gouenou Coatrieux

**Abstract**: Deep neural network (DNN) watermarking is a suitable method for protecting the ownership of deep learning (DL) models derived from computationally intensive processes and painstakingly compiled and annotated datasets. It secretly embeds an identifier (watermark) within the model, which can be retrieved by the owner to prove ownership. In this paper, we first provide a unified framework for white box DNN watermarking schemes. It includes current state-of-the art methods outlining their theoretical inter-connections. In second, we introduce DICTION, a new white-box Dynamic Robust watermarking scheme, we derived from this framework. Its main originality stands on a generative adversarial network (GAN) strategy where the watermark extraction function is a DNN trained as a GAN discriminator, and the target model to watermark as a GAN generator taking a GAN latent space as trigger set input. DICTION can be seen as a generalization of DeepSigns which, to the best of knowledge, is the only other Dynamic white-box watermarking scheme from the literature. Experiments conducted on the same model test set as Deepsigns demonstrate that our scheme achieves much better performance. Especially, and contrarily to DeepSigns, with DICTION one can increase the watermark capacity while preserving at best the model accuracy and ensuring simultaneously a strong robustness against a wide range of watermark removal and detection attacks.

摘要: 深度神经网络(DNN)水印是一种适合于保护深度学习(DL)模型所有权的方法，该模型源于计算密集型过程和精心编译和注释的数据集。它在模型中秘密嵌入一个标识符(水印)，所有者可以检索该标识符以证明所有权。本文首先给出了白盒DNN数字水印方案的统一框架。它包括当前最先进的方法，概述了它们理论上的相互联系。其次，介绍了一种新的白盒动态鲁棒水印方案--WICH，它是由该框架衍生出来的。它的主要创新之处在于一种生成性对抗网络(GAN)策略，其中水印提取函数是训练成GAN鉴别器的DNN，目标模型是以GAN潜在空间作为触发集输入的GAN生成器。就目前所知，DeepSigns是文献中唯一的动态白盒水印方案，它可以被视为DeepSigns的推广。在与DeepDesign相同的模型测试集上进行的实验表明，我们的方案取得了更好的性能。特别是，与DeepSigns相反，使用该算法可以在最好地保持模型精度的同时增加水印容量，并同时确保对广泛的水印去除和检测攻击具有很强的鲁棒性。



## **7. TAD: Transfer Learning-based Multi-Adversarial Detection of Evasion Attacks against Network Intrusion Detection Systems**

TAD：基于转移学习的网络入侵检测系统逃避攻击的多对手检测 cs.CR

This is a preprint of an already published journal paper

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15700v1) [paper-pdf](http://arxiv.org/pdf/2210.15700v1)

**Authors**: Islam Debicha, Richard Bauwens, Thibault Debatty, Jean-Michel Dricot, Tayeb Kenaza, Wim Mees

**Abstract**: Nowadays, intrusion detection systems based on deep learning deliver state-of-the-art performance. However, recent research has shown that specially crafted perturbations, called adversarial examples, are capable of significantly reducing the performance of these intrusion detection systems. The objective of this paper is to design an efficient transfer learning-based adversarial detector and then to assess the effectiveness of using multiple strategically placed adversarial detectors compared to a single adversarial detector for intrusion detection systems. In our experiments, we implement existing state-of-the-art models for intrusion detection. We then attack those models with a set of chosen evasion attacks. In an attempt to detect those adversarial attacks, we design and implement multiple transfer learning-based adversarial detectors, each receiving a subset of the information passed through the IDS. By combining their respective decisions, we illustrate that combining multiple detectors can further improve the detectability of adversarial traffic compared to a single detector in the case of a parallel IDS design.

摘要: 如今，基于深度学习的入侵检测系统提供了最先进的性能。然而，最近的研究表明，精心设计的扰动，称为对抗性示例，能够显著降低这些入侵检测系统的性能。本文的目的是设计一种高效的基于转移学习的敌意检测器，并在此基础上评估在入侵检测系统中使用多个策略放置的敌意检测器与使用单个敌意检测器的有效性。在我们的实验中，我们实现了现有的最先进的入侵检测模型。然后，我们用一系列有选择的规避攻击来攻击这些模型。为了检测这些敌意攻击，我们设计并实现了多个基于转移学习的敌意检测器，每个检测器接收通过入侵检测系统传递的信息的一个子集。通过结合它们各自的决策，我们说明了在并行入侵检测系统设计的情况下，与单一检测器相比，组合多个检测器可以进一步提高敌意流量的可检测性。



## **8. Learning Location from Shared Elevation Profiles in Fitness Apps: A Privacy Perspective**

从隐私的角度从健身应用程序中的共享高程配置文件学习位置 cs.CR

16 pages, 12 figures, 10 tables; accepted for publication in IEEE  Transactions on Mobile Computing (October 2022). arXiv admin note:  substantial text overlap with arXiv:1910.09041

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15529v1) [paper-pdf](http://arxiv.org/pdf/2210.15529v1)

**Authors**: Ulku Meteriz-Yildiran, Necip Fazil Yildiran, Joongheon Kim, David Mohaisen

**Abstract**: The extensive use of smartphones and wearable devices has facilitated many useful applications. For example, with Global Positioning System (GPS)-equipped smart and wearable devices, many applications can gather, process, and share rich metadata, such as geolocation, trajectories, elevation, and time. For example, fitness applications, such as Runkeeper and Strava, utilize the information for activity tracking and have recently witnessed a boom in popularity. Those fitness tracker applications have their own web platforms and allow users to share activities on such platforms or even with other social network platforms. To preserve the privacy of users while allowing sharing, several of those platforms may allow users to disclose partial information, such as the elevation profile for an activity, which supposedly would not leak the location of the users. In this work, and as a cautionary tale, we create a proof of concept where we examine the extent to which elevation profiles can be used to predict the location of users. To tackle this problem, we devise three plausible threat settings under which the city or borough of the targets can be predicted. Those threat settings define the amount of information available to the adversary to launch the prediction attacks. Establishing that simple features of elevation profiles, e.g., spectral features, are insufficient, we devise both natural language processing (NLP)-inspired text-like representation and computer vision-inspired image-like representation of elevation profiles, and we convert the problem at hand into text and image classification problem. We use both traditional machine learning- and deep learning-based techniques and achieve a prediction success rate ranging from 59.59\% to 99.80\%. The findings are alarming, highlighting that sharing elevation information may have significant location privacy risks.

摘要: 智能手机和可穿戴设备的广泛使用促进了许多有用的应用。例如，使用配备全球定位系统(GPS)的智能可穿戴设备，许多应用程序可以收集、处理和共享丰富的元数据，如地理位置、轨迹、高程和时间。例如，RunKeeper和Strava等健身应用程序利用这些信息进行活动跟踪，最近见证了这种应用程序的流行。这些健身跟踪应用程序有自己的网络平台，允许用户在这些平台上甚至与其他社交网络平台分享活动。为了在允许分享的同时保护用户的隐私，其中几个平台可能会允许用户披露部分信息，比如活动的海拔概况，这应该不会泄露用户的位置。在这项工作中，作为一个警示故事，我们创建了一个概念证明，其中我们检查了高程分布可以在多大程度上用于预测用户的位置。为了解决这个问题，我们设计了三个可信的威胁设置，在这些设置下可以预测目标的城市或行政区。这些威胁设置定义了对手可用来发动预测攻击的信息量。建立了高程剖面的简单特征，例如光谱特征是不够的，我们设计了受自然语言处理(NLP)启发的类似文本的高程剖面表示和受计算机视觉启发的类似图像的高程剖面表示，并将手头的问题转化为文本和图像分类问题。我们同时使用了传统的机器学习和基于深度学习的技术，预测成功率从59.59到99.80%不等。这些发现令人担忧，突显出共享高程信息可能会带来重大的位置隐私风险。



## **9. An Analysis of Robustness of Non-Lipschitz Networks**

非Lipschitz网络的稳健性分析 cs.LG

42 pages, 9 figures

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2010.06154v3) [paper-pdf](http://arxiv.org/pdf/2010.06154v3)

**Authors**: Maria-Florina Balcan, Avrim Blum, Dravyansh Sharma, Hongyang Zhang

**Abstract**: Despite significant advances, deep networks remain highly susceptible to adversarial attack. One fundamental challenge is that small input perturbations can often produce large movements in the network's final-layer feature space. In this paper, we define an attack model that abstracts this challenge, to help understand its intrinsic properties. In our model, the adversary may move data an arbitrary distance in feature space but only in random low-dimensional subspaces. We prove such adversaries can be quite powerful: defeating any algorithm that must classify any input it is given. However, by allowing the algorithm to abstain on unusual inputs, we show such adversaries can be overcome when classes are reasonably well-separated in feature space. We further provide strong theoretical guarantees for setting algorithm parameters to optimize over accuracy-abstention trade-offs using data-driven methods. Our results provide new robustness guarantees for nearest-neighbor style algorithms, and also have application to contrastive learning, where we empirically demonstrate the ability of such algorithms to obtain high robust accuracy with low abstention rates. Our model is also motivated by strategic classification, where entities being classified aim to manipulate their observable features to produce a preferred classification, and we provide new insights into that area as well.

摘要: 尽管取得了重大进展，深度网络仍然极易受到对手的攻击。一个基本的挑战是，小的输入扰动通常会在网络的最后一层特征空间中产生大的运动。在本文中，我们定义了一个抽象这一挑战的攻击模型，以帮助理解其内在属性。在我们的模型中，敌手可以将数据在特征空间中移动任意距离，但只能在随机低维子空间中移动。我们证明了这样的对手可以是相当强大的：击败任何必须对给定的任何输入进行分类的算法。然而，通过允许算法在不寻常的输入上弃权，我们证明了当类在特征空间中合理地分离时，这样的对手可以被克服。我们进一步为使用数据驱动方法设置算法参数以优化过度精确度权衡提供了有力的理论保证。我们的结果为最近邻式算法提供了新的稳健性保证，并在对比学习中也得到了应用，我们的经验证明了这种算法能够在较低的弃权率下获得较高的鲁棒性精度。我们的模型也受到战略分类的推动，在战略分类中，被分类的实体旨在操纵它们的可观察特征来产生首选的分类，我们也提供了对该领域的新见解。



## **10. LeNo: Adversarial Robust Salient Object Detection Networks with Learnable Noise**

Leno：具有可学习噪声的对抗性鲁棒显著目标检测网络 cs.CV

8 pages, 5 figures, submitted to AAAI

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15392v1) [paper-pdf](http://arxiv.org/pdf/2210.15392v1)

**Authors**: He Tang, He Wang

**Abstract**: Pixel-wise predction with deep neural network has become an effective paradigm for salient object detection (SOD) and achieved remakable performance. However, very few SOD models are robust against adversarial attacks which are visually imperceptible for human visual attention. The previous work robust salient object detection against adversarial attacks (ROSA) shuffles the pre-segmented superpixels and then refines the coarse saliency map by the densely connected CRF. Different from ROSA that rely on various pre- and post-processings, this paper proposes a light-weight Learnble Noise (LeNo) to against adversarial attacks for SOD models. LeNo preserves accuracy of SOD models on both adversarial and clean images, as well as inference speed. In general, LeNo consists of a simple shallow noise and noise estimation that embedded in the encoder and decoder of arbitrary SOD networks respectively. Inspired by the center prior of human visual attention mechanism, we initialize the shallow noise with a cross-shaped gaussian distribution for better defense against adversarial attacks. Instead of adding additional network components for post-processing, the proposed noise estimation modifies only one channel of the decoder. With the deeply-supervised noise-decoupled training on state-of-the-art RGB and RGB-D SOD networks, LeNo outperforms previous works not only on adversarial images but also clean images, which contributes stronger robustness for SOD.

摘要: 基于深度神经网络的像素预测已成为显著目标检测的一种有效范例，并取得了较好的性能。然而，很少有SOD模型对人类视觉上不可察觉的对抗性攻击具有健壮性。针对敌意攻击的稳健显著目标检测(ROSA)算法首先对预分割的超像素进行置乱处理，然后利用稠密连接的CRF函数对粗略显著图进行细化。不同于ROSA依赖于各种前后处理，本文提出了一种轻量级可学习噪声(Leno)来抵抗对SOD模型的敌意攻击。Leno保持了SOD模型在对抗性图像和干净图像上的准确性，以及推理速度。一般来说，LENO由简单的浅层噪声和噪声估计组成，分别嵌入到任意SOD网络的编码器和译码中。受人类视觉注意机制中心先验的启发，我们用十字形高斯分布对浅层噪声进行初始化，以更好地防御对手的攻击。所提出的噪声估计只需修改解码器的一个通道，而不是为后处理增加额外的网络组件。通过在最新的RGB和RGB-D SOD网络上进行深度监督的噪声解耦训练，Leno不仅在对抗性图像上而且在干净的图像上都优于以往的工作，这为SOD提供了更强的稳健性。



## **11. Isometric 3D Adversarial Examples in the Physical World**

物理世界中的等距3D对抗性例子 cs.CV

NeurIPS 2022

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15291v1) [paper-pdf](http://arxiv.org/pdf/2210.15291v1)

**Authors**: Yibo Miao, Yinpeng Dong, Jun Zhu, Xiao-Shan Gao

**Abstract**: 3D deep learning models are shown to be as vulnerable to adversarial examples as 2D models. However, existing attack methods are still far from stealthy and suffer from severe performance degradation in the physical world. Although 3D data is highly structured, it is difficult to bound the perturbations with simple metrics in the Euclidean space. In this paper, we propose a novel $\epsilon$-isometric ($\epsilon$-ISO) attack to generate natural and robust 3D adversarial examples in the physical world by considering the geometric properties of 3D objects and the invariance to physical transformations. For naturalness, we constrain the adversarial example to be $\epsilon$-isometric to the original one by adopting the Gaussian curvature as a surrogate metric guaranteed by a theoretical analysis. For invariance to physical transformations, we propose a maxima over transformation (MaxOT) method that actively searches for the most harmful transformations rather than random ones to make the generated adversarial example more robust in the physical world. Experiments on typical point cloud recognition models validate that our approach can significantly improve the attack success rate and naturalness of the generated 3D adversarial examples than the state-of-the-art attack methods.

摘要: 研究表明，3D深度学习模型与2D模型一样容易受到敌意例子的影响。然而，现有的攻击方法还远远不是隐身的，在物理世界中还存在严重的性能下降。虽然三维数据是高度结构化的，但在欧氏空间中很难用简单的度量来约束扰动。考虑到三维物体的几何特性和对物理变换的不变性，提出了一种新的三维等距($-ISO)攻击，用于在物理世界中生成自然的和健壮的3D对抗实例。对于自然度，我们通过采用高斯曲率作为理论分析所保证的替代度量来约束对抗性实例与原始实例的-等距。对于物理变换的不变性，我们提出了一种最大值过变换(MaxOT)方法，该方法主动地搜索最有害的变换而不是随机的变换，以使生成的对抗性实例在物理世界中更健壮。在典型点云识别模型上的实验证明，与现有的攻击方法相比，该方法可以显著提高生成的3D对抗性实例的攻击成功率和自然度。



## **12. TASA: Deceiving Question Answering Models by Twin Answer Sentences Attack**

TASA：利用双答句攻击欺骗问答模型 cs.CL

Accepted by EMNLP 2022 (long), 9 pages main + 2 pages references + 7  pages appendix

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15221v1) [paper-pdf](http://arxiv.org/pdf/2210.15221v1)

**Authors**: Yu Cao, Dianqi Li, Meng Fang, Tianyi Zhou, Jun Gao, Yibing Zhan, Dacheng Tao

**Abstract**: We present Twin Answer Sentences Attack (TASA), an adversarial attack method for question answering (QA) models that produces fluent and grammatical adversarial contexts while maintaining gold answers. Despite phenomenal progress on general adversarial attacks, few works have investigated the vulnerability and attack specifically for QA models. In this work, we first explore the biases in the existing models and discover that they mainly rely on keyword matching between the question and context, and ignore the relevant contextual relations for answer prediction. Based on two biases above, TASA attacks the target model in two folds: (1) lowering the model's confidence on the gold answer with a perturbed answer sentence; (2) misguiding the model towards a wrong answer with a distracting answer sentence. Equipped with designed beam search and filtering methods, TASA can generate more effective attacks than existing textual attack methods while sustaining the quality of contexts, in extensive experiments on five QA datasets and human evaluations.

摘要: 我们提出了一种针对问答(QA)模型的对抗性攻击方法Twin Answer语句攻击(TASA)，该方法在保持黄金答案的同时产生流畅的语法对抗性上下文。尽管在一般对抗性攻击方面取得了显著的进展，但很少有研究专门针对QA模型的脆弱性和攻击。在这项工作中，我们首先探讨了现有模型中的偏差，发现它们主要依赖于问题和上下文之间的关键字匹配，而忽略了相关的上下文关系来进行答案预测。基于以上两个偏差，TASA从两个方面对目标模型进行攻击：(1)用扰动答案句降低模型对黄金答案的置信度；(2)用令人分心的答案句将模型误导向错误答案。在五个QA数据集和人工评估的广泛实验中，TASA配备了设计的波束搜索和过滤方法，在保持上下文质量的情况下，可以产生比现有文本攻击方法更有效的攻击。



## **13. V-Cloak: Intelligibility-, Naturalness- & Timbre-Preserving Real-Time Voice Anonymization**

V-Cloak：可理解性、自然性和保留音色的实时语音匿名化 cs.SD

Accepted by USENIX Security Symposium 2023

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15140v1) [paper-pdf](http://arxiv.org/pdf/2210.15140v1)

**Authors**: Jiangyi Deng, Fei Teng, Yanjiao Chen, Xiaofu Chen, Zhaohui Wang, Wenyuan Xu

**Abstract**: Voice data generated on instant messaging or social media applications contains unique user voiceprints that may be abused by malicious adversaries for identity inference or identity theft. Existing voice anonymization techniques, e.g., signal processing and voice conversion/synthesis, suffer from degradation of perceptual quality. In this paper, we develop a voice anonymization system, named V-Cloak, which attains real-time voice anonymization while preserving the intelligibility, naturalness and timbre of the audio. Our designed anonymizer features a one-shot generative model that modulates the features of the original audio at different frequency levels. We train the anonymizer with a carefully-designed loss function. Apart from the anonymity loss, we further incorporate the intelligibility loss and the psychoacoustics-based naturalness loss. The anonymizer can realize untargeted and targeted anonymization to achieve the anonymity goals of unidentifiability and unlinkability.   We have conducted extensive experiments on four datasets, i.e., LibriSpeech (English), AISHELL (Chinese), CommonVoice (French) and CommonVoice (Italian), five Automatic Speaker Verification (ASV) systems (including two DNN-based, two statistical and one commercial ASV), and eleven Automatic Speech Recognition (ASR) systems (for different languages). Experiment results confirm that V-Cloak outperforms five baselines in terms of anonymity performance. We also demonstrate that V-Cloak trained only on the VoxCeleb1 dataset against ECAPA-TDNN ASV and DeepSpeech2 ASR has transferable anonymity against other ASVs and cross-language intelligibility for other ASRs. Furthermore, we verify the robustness of V-Cloak against various de-noising techniques and adaptive attacks. Hopefully, V-Cloak may provide a cloak for us in a prism world.

摘要: 即时消息或社交媒体应用程序上生成的语音数据包含独特的用户声纹，恶意攻击者可能会利用这些声纹进行身份推断或身份窃取。现有的语音匿名化技术，例如信号处理和语音转换/合成，存在感知质量下降的问题。在本文中，我们开发了一个语音匿名系统V-Cloak，它在保持音频的可理解性、自然度和音色的同时，实现了实时的语音匿名。我们设计的匿名器具有一次性生成模型，可以在不同的频率水平上调制原始音频的特征。我们用精心设计的损失函数来训练匿名者。除了匿名性损失外，我们还进一步引入了可理解性损失和基于心理声学的自然度损失。匿名者可以实现无定向和定向匿名化，达到不可识别和不可链接的匿名性目标。我们在LibriSpeech(英语)、AISHELL(中文)、CommonVoice(法语)和CommonVoice(意大利语)四个数据集上进行了广泛的实验，五个自动说话人确认(ASV)系统(包括两个基于DNN的、两个统计的和一个商业的ASV)和11个自动语音识别(ASR)系统(针对不同的语言)。实验结果表明，V-Cloak在匿名性方面优于5条Baseline。我们还证明了仅在VoxCeleb1数据集上针对ECAPA-TDNN ASV和DeepSpeech2 ASR进行训练的V-Cloak具有针对其他ASV的可传递匿名性和针对其他ASR的跨语言可理解性。此外，我们还验证了V-Cloak对各种去噪技术和自适应攻击的稳健性。希望V-Cloak可以为我们在棱镜世界中提供一件斗篷。



## **14. Adaptive Test-Time Defense with the Manifold Hypothesis**

流形假设下的自适应测试时间防御 cs.LG

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.14404v2) [paper-pdf](http://arxiv.org/pdf/2210.14404v2)

**Authors**: Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard Hartley, Peter Tu

**Abstract**: In this work, we formulate a novel framework of adversarial robustness using the manifold hypothesis. Our framework provides sufficient conditions for defending against adversarial examples. We develop a test-time defense method with our formulation and variational inference. The developed approach combines manifold learning with the Bayesian framework to provide adversarial robustness without the need for adversarial training. We show that our proposed approach can provide adversarial robustness even if attackers are aware of existence of test-time defense. In additions, our approach can also serve as a test-time defense mechanism for variational autoencoders.

摘要: 在这项工作中，我们使用流形假设建立了一个新的对抗健壮性框架。我们的框架为防御对抗性例子提供了充分的条件。利用我们的公式和变分推理，我们开发了一种测试时间防御方法。该方法将流形学习与贝叶斯框架相结合，在不需要对抗性训练的情况下提供对抗性健壮性。我们证明，即使攻击者知道测试时间防御的存在，我们所提出的方法也可以提供对抗健壮性。此外，我们的方法还可以作为可变自动编码器的测试时间防御机制。



## **15. Improving Adversarial Robustness with Self-Paced Hard-Class Pair Reweighting**

自定步长硬类对重加权提高对手健壮性 cs.CV

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.15068v1) [paper-pdf](http://arxiv.org/pdf/2210.15068v1)

**Authors**: Pengyue Hou, Jie Han, Xingyu Li

**Abstract**: Deep Neural Networks are vulnerable to adversarial attacks. Among many defense strategies, adversarial training with untargeted attacks is one of the most recognized methods. Theoretically, the predicted labels of untargeted attacks should be unpredictable and uniformly-distributed overall false classes. However, we find that the naturally imbalanced inter-class semantic similarity makes those hard-class pairs to become the virtual targets of each other. This study investigates the impact of such closely-coupled classes on adversarial attacks and develops a self-paced reweighting strategy in adversarial training accordingly. Specifically, we propose to upweight hard-class pair loss in model optimization, which prompts learning discriminative features from hard classes. We further incorporate a term to quantify hard-class pair consistency in adversarial training, which greatly boost model robustness. Extensive experiments show that the proposed adversarial training method achieves superior robustness performance over state-of-the-art defenses against a wide range of adversarial attacks.

摘要: 深度神经网络很容易受到敌意攻击。在众多的防御策略中，非靶向攻击的对抗性训练是最受认可的方法之一。从理论上讲，非目标攻击的预测标签应该是不可预测的，且总体上均匀分布的伪类。然而，我们发现，自然不平衡的类间语义相似度使得这些硬类对成为彼此的虚拟目标。本研究调查了这种紧密耦合的课程对对抗性攻击的影响，并相应地在对抗性训练中开发了一种自定步调重权重策略。具体地说，我们提出了在模型优化中增加硬类对损失的权重，从而促进了从硬类中学习区分特征。在对抗性训练中，我们进一步引入了一个术语来量化硬类对一致性，这大大提高了模型的稳健性。大量的实验表明，所提出的对抗性训练方法在对抗广泛的对抗性攻击时获得了比最先进的防御方法更好的健壮性性能。



## **16. Using Deception in Markov Game to Understand Adversarial Behaviors through a Capture-The-Flag Environment**

利用马尔可夫博弈中的欺骗来理解捕获旗帜环境中的敌方行为 cs.GT

Accepted at GameSec 2022

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.15011v1) [paper-pdf](http://arxiv.org/pdf/2210.15011v1)

**Authors**: Siddhant Bhambri, Purv Chauhan, Frederico Araujo, Adam Doupé, Subbarao Kambhampati

**Abstract**: Identifying the actual adversarial threat against a system vulnerability has been a long-standing challenge for cybersecurity research. To determine an optimal strategy for the defender, game-theoretic based decision models have been widely used to simulate the real-world attacker-defender scenarios while taking the defender's constraints into consideration. In this work, we focus on understanding human attacker behaviors in order to optimize the defender's strategy. To achieve this goal, we model attacker-defender engagements as Markov Games and search for their Bayesian Stackelberg Equilibrium. We validate our modeling approach and report our empirical findings using a Capture-The-Flag (CTF) setup, and we conduct user studies on adversaries with varying skill-levels. Our studies show that application-level deceptions are an optimal mitigation strategy against targeted attacks -- outperforming classic cyber-defensive maneuvers, such as patching or blocking network requests. We use this result to further hypothesize over the attacker's behaviors when trapped in an embedded honeypot environment and present a detailed analysis of the same.

摘要: 识别针对系统漏洞的实际对手威胁一直是网络安全研究的长期挑战。为了确定防御者的最优策略，基于博弈论的决策模型被广泛用于模拟现实世界中的攻防场景，同时考虑了防御者的约束。在这项工作中，我们重点了解人类攻击者的行为，以便优化防御者的策略。为了实现这一目标，我们将攻防双方的交战建模为马尔可夫博弈，并寻找他们的贝叶斯Stackelberg均衡。我们验证了我们的建模方法，并使用捕获旗帜(CTF)设置报告了我们的经验结果，并对具有不同技能水平的对手进行了用户研究。我们的研究表明，应用程序级别的欺骗是针对目标攻击的最佳缓解策略--性能优于修补或阻止网络请求等传统的网络防御策略。我们利用这一结果进一步假设攻击者在被困在嵌入式蜜罐环境中时的行为，并对此进行了详细的分析。



## **17. Model-Free Prediction of Adversarial Drop Points in 3D Point Clouds**

三维点云中对抗性滴点的无模式预报 cs.CV

10 pages, 6 figures

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14164v2) [paper-pdf](http://arxiv.org/pdf/2210.14164v2)

**Authors**: Hanieh Naderi, Chinthaka Dinesh, Ivan V. Bajic, Shohreh Kasaei

**Abstract**: Adversarial attacks pose serious challenges for deep neural network (DNN)-based analysis of various input signals. In the case of 3D point clouds, methods have been developed to identify points that play a key role in the network decision, and these become crucial in generating existing adversarial attacks. For example, a saliency map approach is a popular method for identifying adversarial drop points, whose removal would significantly impact the network decision. Generally, methods for identifying adversarial points rely on the deep model itself in order to determine which points are critically important for the model's decision. This paper aims to provide a novel viewpoint on this problem, in which adversarial points can be predicted independently of the model. To this end, we define 14 point cloud features and use multiple linear regression to examine whether these features can be used for model-free adversarial point prediction, and which combination of features is best suited for this purpose. Experiments show that a suitable combination of features is able to predict adversarial points of three different networks -- PointNet, PointNet++, and DGCNN -- significantly better than a random guess. The results also provide further insight into DNNs for point cloud analysis, by showing which features play key roles in their decision-making process.

摘要: 对抗性攻击对基于深度神经网络(DNN)的各种输入信号分析提出了严峻的挑战。在3D点云的情况下，已经开发出方法来识别在网络决策中起关键作用的点，并且这些点在生成现有的对抗性攻击时变得至关重要。例如，显著图方法是一种流行的识别对抗性丢弃点的方法，其移除将显著影响网络决策。通常，识别敌对点的方法依赖于深度模型本身，以确定哪些点对模型的决策至关重要。本文旨在为这一问题提供一种新的观点，即可以独立于模型来预测敌对点。为此，我们定义了14个点云特征，并使用多元线性回归来检验这些特征是否可以用于无模型对抗点预测，以及哪种特征组合最适合于此目的。实验表明，适当的特征组合能够预测三种不同网络--PointNet、PointNet++和DGCNN的敌对点--明显好于随机猜测。通过显示哪些特征在其决策过程中起关键作用，结果还提供了对用于点云分析的DNN的进一步洞察。



## **18. Disentangled Text Representation Learning with Information-Theoretic Perspective for Adversarial Robustness**

基于信息论视角的解缠文本表征学习的对抗性 cs.CL

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14957v1) [paper-pdf](http://arxiv.org/pdf/2210.14957v1)

**Authors**: Jiahao Zhao, Wenji Mao

**Abstract**: Adversarial vulnerability remains a major obstacle to constructing reliable NLP systems. When imperceptible perturbations are added to raw input text, the performance of a deep learning model may drop dramatically under attacks. Recent work argues the adversarial vulnerability of the model is caused by the non-robust features in supervised training. Thus in this paper, we tackle the adversarial robustness challenge from the view of disentangled representation learning, which is able to explicitly disentangle robust and non-robust features in text. Specifically, inspired by the variation of information (VI) in information theory, we derive a disentangled learning objective composed of mutual information to represent both the semantic representativeness of latent embeddings and differentiation of robust and non-robust features. On the basis of this, we design a disentangled learning network to estimate these mutual information. Experiments on text classification and entailment tasks show that our method significantly outperforms the representative methods under adversarial attacks, indicating that discarding non-robust features is critical for improving adversarial robustness.

摘要: 对抗性漏洞仍然是构建可靠的自然语言处理系统的主要障碍。当向原始输入文本添加不可察觉的扰动时，深度学习模型的性能可能会在攻击下显著下降。最近的工作认为，该模型的对抗性漏洞是由监督训练中的非稳健特征造成的。因此，在本文中，我们从解缠表示学习的角度来解决对抗性的健壮性挑战，它能够显式地解开文本中的健壮和非健壮特征。具体地说，受信息论中信息变化(VI)的启发，我们提出了一个由互信息组成的解缠学习目标，以表示潜在嵌入的语义代表性以及稳健和非稳健特征的区分。在此基础上，我们设计了一个解缠学习网络来估计这些互信息。在文本分类和蕴涵任务上的实验表明，我们的方法在对抗攻击下的性能明显优于典型的方法，这表明丢弃非健壮性特征对于提高对抗攻击的稳健性至关重要。



## **19. On the Versatile Uses of Partial Distance Correlation in Deep Learning**

偏距离相关在深度学习中的广泛应用 cs.CV

This paper has been selected as best paper award for ECCV 2022!

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2207.09684v2) [paper-pdf](http://arxiv.org/pdf/2207.09684v2)

**Authors**: Xingjian Zhen, Zihang Meng, Rudrasis Chakraborty, Vikas Singh

**Abstract**: Comparing the functional behavior of neural network models, whether it is a single network over time or two (or more networks) during or post-training, is an essential step in understanding what they are learning (and what they are not), and for identifying strategies for regularization or efficiency improvements. Despite recent progress, e.g., comparing vision transformers to CNNs, systematic comparison of function, especially across different networks, remains difficult and is often carried out layer by layer. Approaches such as canonical correlation analysis (CCA) are applicable in principle, but have been sparingly used so far. In this paper, we revisit a (less widely known) from statistics, called distance correlation (and its partial variant), designed to evaluate correlation between feature spaces of different dimensions. We describe the steps necessary to carry out its deployment for large scale models -- this opens the door to a surprising array of applications ranging from conditioning one deep model w.r.t. another, learning disentangled representations as well as optimizing diverse models that would directly be more robust to adversarial attacks. Our experiments suggest a versatile regularizer (or constraint) with many advantages, which avoids some of the common difficulties one faces in such analyses. Code is at https://github.com/zhenxingjian/Partial_Distance_Correlation.

摘要: 比较神经网络模型的功能行为，无论是随着时间的推移是单个网络还是在训练期间或训练后的两个(或更多)网络，对于了解它们正在学习什么(以及它们不是什么)以及确定正规化或效率改进的策略是至关重要的一步。尽管最近取得了进展，例如，将视觉转换器与CNN进行了比较，但系统地比较功能，特别是跨不同网络的功能，仍然很困难，而且往往是逐层进行的。典型相关分析(CCA)等方法在原则上是适用的，但到目前为止一直很少使用。在本文中，我们回顾了统计学中的一个(不太广为人知的)方法，称为距离相关(及其部分变量)，旨在评估不同维度的特征空间之间的相关性。我们描述了在大规模模型中实施其部署所需的步骤--这为一系列令人惊讶的应用打开了大门，从调节一个深度模型到W.r.t。另一种是，学习分离的表示以及优化多样化的模型，这些模型将直接对对手攻击更健壮。我们的实验提出了一种具有许多优点的通用正则化(或约束)方法，它避免了人们在此类分析中面临的一些常见困难。代码在https://github.com/zhenxingjian/Partial_Distance_Correlation.上



## **20. Identifying Threats, Cybercrime and Digital Forensic Opportunities in Smart City Infrastructure via Threat Modeling**

通过威胁建模识别智能城市基础设施中的威胁、网络犯罪和数字取证机会 cs.CR

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14692v1) [paper-pdf](http://arxiv.org/pdf/2210.14692v1)

**Authors**: Yee Ching Tok, Sudipta Chattopadhyay

**Abstract**: Technological advances have enabled multiple countries to consider implementing Smart City Infrastructure to provide in-depth insights into different data points and enhance the lives of citizens. Unfortunately, these new technological implementations also entice adversaries and cybercriminals to execute cyber-attacks and commit criminal acts on these modern infrastructures. Given the borderless nature of cyber attacks, varying levels of understanding of smart city infrastructure and ongoing investigation workloads, law enforcement agencies and investigators would be hard-pressed to respond to these kinds of cybercrime. Without an investigative capability by investigators, these smart infrastructures could become new targets favored by cybercriminals.   To address the challenges faced by investigators, we propose a common definition of smart city infrastructure. Based on the definition, we utilize the STRIDE threat modeling methodology and the Microsoft Threat Modeling Tool to identify threats present in the infrastructure and create a threat model which can be further customized or extended by interested parties. Next, we map offences, possible evidence sources and types of threats identified to help investigators understand what crimes could have been committed and what evidence would be required in their investigation work. Finally, noting that Smart City Infrastructure investigations would be a global multi-faceted challenge, we discuss technical and legal opportunities in digital forensics on Smart City Infrastructure.

摘要: 技术进步使多个国家能够考虑实施智慧城市基础设施，以深入了解不同的数据点，并改善公民的生活。不幸的是，这些新的技术实施也引诱对手和网络罪犯对这些现代基础设施进行网络攻击和犯罪行为。鉴于网络攻击的无边界性质、对智能城市基础设施的不同程度的了解以及正在进行的调查工作量，执法机构和调查人员将很难对此类网络犯罪做出回应。如果调查人员没有调查能力，这些智能基础设施可能会成为网络犯罪分子青睐的新目标。为了应对调查人员面临的挑战，我们提出了智能城市基础设施的共同定义。在定义的基础上，我们利用STRIDE威胁建模方法和Microsoft威胁建模工具来识别基础设施中存在的威胁，并创建可由感兴趣的各方进一步定制或扩展的威胁模型。接下来，我们将绘制罪行、可能的证据来源和已确定的威胁类型的地图，以帮助调查人员了解哪些罪行可能发生，以及在调查工作中需要哪些证据。最后，注意到智能城市基础设施调查将是一项全球多方面的挑战，我们讨论了智能城市基础设施数字取证的技术和法律机会。



## **21. Certified Robustness in Federated Learning**

联合学习中的认证稳健性 cs.LG

Accepted at Workshop on Federated Learning: Recent Advances and New  Challenges, NeurIPS 2022

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2206.02535v2) [paper-pdf](http://arxiv.org/pdf/2206.02535v2)

**Authors**: Motasem Alfarra, Juan C. Pérez, Egor Shulgin, Peter Richtárik, Bernard Ghanem

**Abstract**: Federated learning has recently gained significant attention and popularity due to its effectiveness in training machine learning models on distributed data privately. However, as in the single-node supervised learning setup, models trained in federated learning suffer from vulnerability to imperceptible input transformations known as adversarial attacks, questioning their deployment in security-related applications. In this work, we study the interplay between federated training, personalization, and certified robustness. In particular, we deploy randomized smoothing, a widely-used and scalable certification method, to certify deep networks trained on a federated setup against input perturbations and transformations. We find that the simple federated averaging technique is effective in building not only more accurate, but also more certifiably-robust models, compared to training solely on local data. We further analyze personalization, a popular technique in federated training that increases the model's bias towards local data, on robustness. We show several advantages of personalization over both~(that is, only training on local data and federated training) in building more robust models with faster training. Finally, we explore the robustness of mixtures of global and local~(i.e. personalized) models, and find that the robustness of local models degrades as they diverge from the global model

摘要: 由于联邦学习在训练分布式数据上的机器学习模型方面的有效性，它最近获得了极大的关注和普及。然而，与单节点监督学习设置中一样，在联合学习中训练的模型容易受到称为对抗性攻击的不可察觉的输入转换的影响，从而质疑其在安全相关应用中的部署。在这项工作中，我们研究了联合训练、个性化和经过认证的健壮性之间的相互作用。特别是，我们采用了随机化平滑，这是一种广泛使用和可扩展的认证方法，用于认证在联合设置上训练的深层网络不受输入扰动和转换的影响。我们发现，与仅基于本地数据进行训练相比，简单的联合平均技术不仅在建立更准确的模型方面是有效的，而且在可证明的健壮性方面也更有效。我们进一步分析了个性化，这是联合训练中的一种流行技术，它增加了模型对本地数据的偏差，并对稳健性进行了分析。我们展示了个性化比这两者(即只在本地数据上训练和联合训练)在建立更健壮的模型和更快的训练方面的几个优势。最后，我们研究了全局模型和局部模型(即个性化模型)的混合模型的稳健性，发现局部模型的稳健性随着偏离全局模型而降低



## **22. Short Paper: Static and Microarchitectural ML-Based Approaches For Detecting Spectre Vulnerabilities and Attacks**

短文：基于静态和微体系结构ML的检测Spectre漏洞和攻击的方法 cs.CR

5 pages, 2 figures. Accepted to the Hardware and Architectural  Support for Security and Privacy (HASP'22), in conjunction with the 55th  IEEE/ACM International Symposium on Microarchitecture (MICRO'22)

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14452v1) [paper-pdf](http://arxiv.org/pdf/2210.14452v1)

**Authors**: Chidera Biringa, Gaspard Baye, Gökhan Kul

**Abstract**: Spectre intrusions exploit speculative execution design vulnerabilities in modern processors. The attacks violate the principles of isolation in programs to gain unauthorized private user information. Current state-of-the-art detection techniques utilize micro-architectural features or vulnerable speculative code to detect these threats. However, these techniques are insufficient as Spectre attacks have proven to be more stealthy with recently discovered variants that bypass current mitigation mechanisms. Side-channels generate distinct patterns in processor cache, and sensitive information leakage is dependent on source code vulnerable to Spectre attacks, where an adversary uses these vulnerabilities, such as branch prediction, which causes a data breach. Previous studies predominantly approach the detection of Spectre attacks using the microarchitectural analysis, a reactive approach. Hence, in this paper, we present the first comprehensive evaluation of static and microarchitectural analysis-assisted machine learning approaches to detect Spectre vulnerable code snippets (preventive) and Spectre attacks (reactive). We evaluate the performance trade-offs in employing classifiers for detecting Spectre vulnerabilities and attacks.

摘要: 幽灵入侵利用现代处理器中的推测性执行设计漏洞。这些攻击违反了程序中的隔离原则，以获取未经授权的私人用户信息。当前最先进的检测技术利用微体系结构特征或易受攻击的推测代码来检测这些威胁。然而，这些技术是不够的，因为Spectre攻击已被证明是更隐蔽的，最近发现的变体绕过了当前的缓解机制。侧通道在处理器缓存中生成不同的模式，敏感信息泄漏依赖于易受Spectre攻击的源代码，其中对手使用这些漏洞，如分支预测，从而导致数据泄露。以前的研究主要是使用微体系结构分析来检测Spectre攻击，这是一种反应性方法。因此，在本文中，我们首次对静态和微体系结构分析辅助的机器学习方法进行了全面评估，以检测Spectre易受攻击的代码片段(预防性的)和Spectre攻击(反应性的)。我们评估了使用分类器来检测Spectre漏洞和攻击时的性能权衡。



## **23. LP-BFGS attack: An adversarial attack based on the Hessian with limited pixels**

LP-BFGS攻击：一种基于有限像素黑森的对抗性攻击 cs.CR

5 pages, 4 figures

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.15446v1) [paper-pdf](http://arxiv.org/pdf/2210.15446v1)

**Authors**: Jiebao Zhang, Wenhua Qian, Rencan Nie, Jinde Cao, Dan Xu

**Abstract**: Deep neural networks are vulnerable to adversarial attacks. Most white-box attacks are based on the gradient of models to the input. Since the computation and memory budget, adversarial attacks based on the Hessian information are not paid enough attention. In this work, we study the attack performance and computation cost of the attack method based on the Hessian with a limited perturbation pixel number. Specifically, we propose the Limited Pixel BFGS (LP-BFGS) attack method by incorporating the BFGS algorithm. Some pixels are selected as perturbation pixels by the Integrated Gradient algorithm, which are regarded as optimization variables of the LP-BFGS attack. Experimental results across different networks and datasets with various perturbation pixel numbers demonstrate our approach has a comparable attack with an acceptable computation compared with existing solutions.

摘要: 深度神经网络很容易受到敌意攻击。大多数白盒攻击都是基于模型对输入的梯度。由于计算和内存预算的限制，基于黑森信息的对抗性攻击没有引起足够的重视。在这项工作中，我们研究了有限扰动像素数的基于Hessian的攻击方法的攻击性能和计算代价。具体地说，我们结合有限像素BFGS算法，提出了LP-BFGS攻击方法。综合梯度算法选取部分像素点作为扰动像素点，作为LP-BFGS攻击的优化变量。在具有不同扰动像素数的不同网络和数据集上的实验结果表明，该方法具有与现有解决方案相当的攻击能力和可接受的计算量。



## **24. Improving Adversarial Robustness via Joint Classification and Multiple Explicit Detection Classes**

联合分类和多个显式检测类提高敌方鲁棒性 cs.CV

21 pages, 6 figures

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14410v1) [paper-pdf](http://arxiv.org/pdf/2210.14410v1)

**Authors**: Sina Baharlouei, Fatemeh Sheikholeslami, Meisam Razaviyayn, Zico Kolter

**Abstract**: This work concerns the development of deep networks that are certifiably robust to adversarial attacks. Joint robust classification-detection was recently introduced as a certified defense mechanism, where adversarial examples are either correctly classified or assigned to the "abstain" class. In this work, we show that such a provable framework can benefit by extension to networks with multiple explicit abstain classes, where the adversarial examples are adaptively assigned to those. We show that naively adding multiple abstain classes can lead to "model degeneracy", then we propose a regularization approach and a training method to counter this degeneracy by promoting full use of the multiple abstain classes. Our experiments demonstrate that the proposed approach consistently achieves favorable standard vs. robust verified accuracy tradeoffs, outperforming state-of-the-art algorithms for various choices of number of abstain classes.

摘要: 这项工作涉及到深度网络的发展，这些网络对对手攻击具有可证明的健壮性。联合稳健分类-检测是最近引入的一种认证防御机制，在这种机制中，对抗性例子要么被正确分类，要么被分配到“弃权”类别。在这项工作中，我们表明这样一个可证明的框架可以通过扩展到具有多个显式弃权类的网络而受益，其中对抗性示例被自适应地分配给那些显式弃权类。我们证明了简单地添加多个弃权类会导致“模型退化”，然后我们提出了一种正则化方法和一种训练方法，通过促进多个弃权类的充分利用来克服这种退化。我们的实验表明，该方法一致地达到了良好的标准和健壮的验证精度折衷，在不同数量的弃权类的选择上优于最新的算法。



## **25. Robustness of Locally Differentially Private Graph Analysis Against Poisoning**

局部差分私有图分析对中毒的稳健性 cs.CR

22 pages, 6 figures

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14376v1) [paper-pdf](http://arxiv.org/pdf/2210.14376v1)

**Authors**: Jacob Imola, Amrita Roy Chowdhury, Kamalika Chaudhuri

**Abstract**: Locally differentially private (LDP) graph analysis allows private analysis on a graph that is distributed across multiple users. However, such computations are vulnerable to data poisoning attacks where an adversary can skew the results by submitting malformed data. In this paper, we formally study the impact of poisoning attacks for graph degree estimation protocols under LDP. We make two key technical contributions. First, we observe LDP makes a protocol more vulnerable to poisoning -- the impact of poisoning is worse when the adversary can directly poison their (noisy) responses, rather than their input data. Second, we observe that graph data is naturally redundant -- every edge is shared between two users. Leveraging this data redundancy, we design robust degree estimation protocols under LDP that can significantly reduce the impact of data poisoning and compute degree estimates with high accuracy. We evaluate our proposed robust degree estimation protocols under poisoning attacks on real-world datasets to demonstrate their efficacy in practice.

摘要: 局部差分私有(LDP)图分析允许对分布在多个用户之间的图进行私有分析。然而，这种计算很容易受到数据中毒攻击，对手可以通过提交格式错误的数据来扭曲结果。本文形式化地研究了毒化攻击对LDP下图度估计协议的影响。我们做出了两项关键的技术贡献。首先，我们观察到LDP使协议更容易中毒--当对手可以直接中毒他们的(噪声)响应，而不是他们的输入数据时，中毒的影响更严重。其次，我们观察到图形数据自然是冗余的--每条边都在两个用户之间共享。利用这种数据冗余性，我们在LDP下设计了稳健的度估计协议，可以显著降低数据中毒的影响，并能高精度地计算度估计。我们在真实数据集上的中毒攻击下对我们提出的稳健程度估计协议进行了评估，以证明其在实践中的有效性。



## **26. Accelerating Certified Robustness Training via Knowledge Transfer**

通过知识转移加快认证健壮性培训 cs.LG

NeurIPS '22 Camera Ready version (with appendix)

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14283v1) [paper-pdf](http://arxiv.org/pdf/2210.14283v1)

**Authors**: Pratik Vaishnavi, Kevin Eykholt, Amir Rahmati

**Abstract**: Training deep neural network classifiers that are certifiably robust against adversarial attacks is critical to ensuring the security and reliability of AI-controlled systems. Although numerous state-of-the-art certified training methods have been developed, they are computationally expensive and scale poorly with respect to both dataset and network complexity. Widespread usage of certified training is further hindered by the fact that periodic retraining is necessary to incorporate new data and network improvements. In this paper, we propose Certified Robustness Transfer (CRT), a general-purpose framework for reducing the computational overhead of any certifiably robust training method through knowledge transfer. Given a robust teacher, our framework uses a novel training loss to transfer the teacher's robustness to the student. We provide theoretical and empirical validation of CRT. Our experiments on CIFAR-10 show that CRT speeds up certified robustness training by $8 \times$ on average across three different architecture generations while achieving comparable robustness to state-of-the-art methods. We also show that CRT can scale to large-scale datasets like ImageNet.

摘要: 训练深度神经网络分类器对于确保人工智能控制系统的安全性和可靠性至关重要。尽管已经开发了许多最先进的认证训练方法，但它们的计算成本很高，并且在数据集和网络复杂性方面可伸缩性较差。定期再培训对于纳入新的数据和网络改进是必要的，这进一步阻碍了认证培训的广泛使用。在本文中，我们提出了认证健壮性转移(CRT)，这是一个通用的框架，通过知识转移来减少任何可证明健壮性训练方法的计算开销。假设有一位健壮的教师，我们的框架使用了一种新的训练损失来将教师的健壮性传递给学生。我们提供了CRT的理论和经验验证。我们在CIFAR-10上的实验表明，CRT在三代不同的体系结构上平均将经过认证的健壮性训练速度提高了8倍，同时获得了与最先进方法相当的健壮性。我们还展示了CRT可以扩展到像ImageNet这样的大规模数据集。



## **27. Similarity between Units of Natural Language: The Transition from Coarse to Fine Estimation**

自然语言单位之间的相似性：从粗略到精细的过渡 cs.CL

PhD thesis

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14275v1) [paper-pdf](http://arxiv.org/pdf/2210.14275v1)

**Authors**: Wenchuan Mu

**Abstract**: Capturing the similarities between human language units is crucial for explaining how humans associate different objects, and therefore its computation has received extensive attention, research, and applications. With the ever-increasing amount of information around us, calculating similarity becomes increasingly complex, especially in many cases, such as legal or medical affairs, measuring similarity requires extra care and precision, as small acts within a language unit can have significant real-world effects. My research goal in this thesis is to develop regression models that account for similarities between language units in a more refined way.   Computation of similarity has come a long way, but approaches to debugging the measures are often based on continually fitting human judgment values. To this end, my goal is to develop an algorithm that precisely catches loopholes in a similarity calculation. Furthermore, most methods have vague definitions of the similarities they compute and are often difficult to interpret. The proposed framework addresses both shortcomings. It constantly improves the model through catching different loopholes. In addition, every refinement of the model provides a reasonable explanation. The regression model introduced in this thesis is called progressively refined similarity computation, which combines attack testing with adversarial training. The similarity regression model of this thesis achieves state-of-the-art performance in handling edge cases.

摘要: 捕捉人类语言单位之间的相似性对于解释人类如何关联不同的对象至关重要，因此其计算得到了广泛的关注、研究和应用。随着我们周围信息量的不断增加，计算相似度变得越来越复杂，特别是在许多情况下，如法律或医疗事务，计算相似度需要格外小心和精确，因为一个语言单位内的小行为可能会产生重大的现实世界影响。我在这篇论文中的研究目标是建立回归模型，以更精细的方式解释语言单位之间的相似性。相似性的计算已经走了很长一段路，但调试这些测量的方法通常是基于不断拟合人类判断值。为此，我的目标是开发一种算法，准确地捕捉相似性计算中的漏洞。此外，大多数方法对它们计算的相似性有模糊的定义，而且往往很难解释。拟议的框架解决了这两个缺点。它通过捕捉不同的漏洞来不断改进模型。此外，模型的每一次细化都提供了合理的解释。本文所介绍的回归模型称为递进精化相似度计算，它将攻击测试和对抗性训练相结合。本文的相似度回归模型在处理边缘情况方面达到了最好的性能。



## **28. Leveraging the Verifier's Dilemma to Double Spend in Bitcoin**

利用验证者的两难境地加倍投入比特币 cs.CR

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14072v1) [paper-pdf](http://arxiv.org/pdf/2210.14072v1)

**Authors**: Tong Cao, Jérémie Decouchant, Jiangshan Yu

**Abstract**: We describe and analyze perishing mining, a novel block-withholding mining strategy that lures profit-driven miners away from doing useful work on the public chain by releasing block headers from a privately maintained chain. We then introduce the dual private chain (DPC) attack, where an adversary that aims at double spending increases its success rate by intermittently dedicating part of its hash power to perishing mining. We detail the DPC attack's Markov decision process, evaluate its double spending success rate using Monte Carlo simulations. We show that the DPC attack lowers Bitcoin's security bound in the presence of profit-driven miners that do not wait to validate the transactions of a block before mining on it.

摘要: 我们描述和分析了正在灭亡的挖掘，这是一种新的块扣留挖掘策略，通过从私人维护的链中释放块头来引诱受利润驱动的矿工远离在公共链上做有用的工作。然后，我们介绍了双重私有链(DPC)攻击，在这种攻击中，一个旨在加倍支出的对手通过断断续续地将其部分散列能力用于消灭挖掘来提高其成功率。详细描述了DPC攻击的马尔可夫决策过程，并利用蒙特卡罗模拟对其双开销成功率进行了评估。我们表明，在利润驱动的矿工在场的情况下，DPC攻击降低了比特币的安全界限，这些矿工在挖掘比特币之前不会等待验证区块的交易。



## **29. A White-Box Adversarial Attack Against a Digital Twin**

一种针对数字双胞胎的白盒对抗性攻击 cs.CR

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14018v1) [paper-pdf](http://arxiv.org/pdf/2210.14018v1)

**Authors**: Wilson Patterson, Ivan Fernandez, Subash Neupane, Milan Parmar, Sudip Mittal, Shahram Rahimi

**Abstract**: Recent research has shown that Machine Learning/Deep Learning (ML/DL) models are particularly vulnerable to adversarial perturbations, which are small changes made to the input data in order to fool a machine learning classifier. The Digital Twin, which is typically described as consisting of a physical entity, a virtual counterpart, and the data connections in between, is increasingly being investigated as a means of improving the performance of physical entities by leveraging computational techniques, which are enabled by the virtual counterpart. This paper explores the susceptibility of Digital Twin (DT), a virtual model designed to accurately reflect a physical object using ML/DL classifiers that operate as Cyber Physical Systems (CPS), to adversarial attacks. As a proof of concept, we first formulate a DT of a vehicular system using a deep neural network architecture and then utilize it to launch an adversarial attack. We attack the DT model by perturbing the input to the trained model and show how easily the model can be broken with white-box attacks.

摘要: 最近的研究表明，机器学习/深度学习(ML/DL)模型特别容易受到对抗性扰动的影响，这些扰动是为了愚弄机器学习分类器而对输入数据进行的微小更改。数字双胞胎通常被描述为由物理实体、虚拟对应物和它们之间的数据连接组成，越来越多的人将其作为一种通过利用由虚拟对等体实现的计算技术来改善物理实体的性能的手段来进行研究。数字孪生(DT)是一种虚拟模型，它使用作为网络物理系统(CPS)的ML/DL分类器来准确地反映物理对象，本文探讨了DT对对手攻击的敏感性。作为概念验证，我们首先使用深度神经网络体系结构来建立车载系统的DT，然后利用它来发起对抗性攻击。我们通过扰动训练模型的输入来攻击DT模型，并展示了该模型可以多么容易地被白盒攻击打破。



## **30. Causal Information Bottleneck Boosts Adversarial Robustness of Deep Neural Network**

因果信息瓶颈增强深度神经网络的对抗健壮性 cs.LG

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14229v1) [paper-pdf](http://arxiv.org/pdf/2210.14229v1)

**Authors**: Huan Hua, Jun Yan, Xi Fang, Weiquan Huang, Huilin Yin, Wancheng Ge

**Abstract**: The information bottleneck (IB) method is a feasible defense solution against adversarial attacks in deep learning. However, this method suffers from the spurious correlation, which leads to the limitation of its further improvement of adversarial robustness. In this paper, we incorporate the causal inference into the IB framework to alleviate such a problem. Specifically, we divide the features obtained by the IB method into robust features (content information) and non-robust features (style information) via the instrumental variables to estimate the causal effects. With the utilization of such a framework, the influence of non-robust features could be mitigated to strengthen the adversarial robustness. We make an analysis of the effectiveness of our proposed method. The extensive experiments in MNIST, FashionMNIST, and CIFAR-10 show that our method exhibits the considerable robustness against multiple adversarial attacks. Our code would be released.

摘要: 信息瓶颈方法是深度学习中对抗攻击的一种可行的防御方案。然而，该方法存在伪相关问题，限制了其进一步提高对抗健壮性。在本文中，我们将因果推理引入到IB框架中来缓解这一问题。具体地说，我们通过工具变量将IB方法得到的特征划分为稳健特征(内容信息)和非稳健特征(风格信息)来估计因果效应。利用该框架可以减少非稳健特征的影响，增强对抗的稳健性。并对该方法的有效性进行了分析。在MNIST、FashionMNIST和CIFAR-10上的大量实验表明，我们的方法对多个对手攻击具有相当大的鲁棒性。我们的代码就会被发布。



## **31. CalFAT: Calibrated Federated Adversarial Training with Label Skewness**

卡尔法特：带有标签偏斜度的校准联合对抗性训练 cs.LG

Accepted to the Conference on the Advances in Neural Information  Processing Systems (NeurIPS) 2022

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2205.14926v2) [paper-pdf](http://arxiv.org/pdf/2205.14926v2)

**Authors**: Chen Chen, Yuchen Liu, Xingjun Ma, Lingjuan Lyu

**Abstract**: Recent studies have shown that, like traditional machine learning, federated learning (FL) is also vulnerable to adversarial attacks. To improve the adversarial robustness of FL, federated adversarial training (FAT) methods have been proposed to apply adversarial training locally before global aggregation. Although these methods demonstrate promising results on independent identically distributed (IID) data, they suffer from training instability on non-IID data with label skewness, resulting in degraded natural accuracy. This tends to hinder the application of FAT in real-world applications where the label distribution across the clients is often skewed. In this paper, we study the problem of FAT under label skewness, and reveal one root cause of the training instability and natural accuracy degradation issues: skewed labels lead to non-identical class probabilities and heterogeneous local models. We then propose a Calibrated FAT (CalFAT) approach to tackle the instability issue by calibrating the logits adaptively to balance the classes. We show both theoretically and empirically that the optimization of CalFAT leads to homogeneous local models across the clients and better convergence points.

摘要: 最近的研究表明，与传统的机器学习一样，联邦学习(FL)也容易受到对手攻击。为了提高FL的对抗健壮性，联合对抗训练(FAT)方法被提出在全局聚集之前局部应用对抗训练。虽然这些方法在独立同分布(IID)数据上显示了良好的结果，但它们在具有标签偏斜的非IID数据上存在训练不稳定性，导致自然精度降低。这往往会阻碍FAT在实际应用中的应用，在现实应用中，跨客户端的标签分布通常是不对称的。本文研究了标签倾斜下的FAT问题，揭示了训练不稳定和自然精度下降的一个根本原因：倾斜的标签会导致类别概率不同和局部模型的异构性。然后，我们提出了一种校准FAT(CALFAT)方法来解决不稳定性问题，方法是自适应地校准逻辑以平衡类别。我们从理论和经验两个方面证明了CALFAT算法的优化可以得到跨客户的同质局部模型和更好的收敛点。



## **32. FocusedCleaner: Sanitizing Poisoned Graphs for Robust GNN-based Node Classification**

FocusedCleaner：用于基于GNN的健壮节点分类的毒图清理 cs.LG

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.13815v1) [paper-pdf](http://arxiv.org/pdf/2210.13815v1)

**Authors**: Yulin Zhu, Liang Tong, Kai Zhou

**Abstract**: Recently, a lot of research attention has been devoted to exploring Web security, a most representative topic is the adversarial robustness of graph mining algorithms. Especially, a widely deployed adversarial attacks formulation is the graph manipulation attacks by modifying the relational data to mislead the Graph Neural Networks' (GNNs) predictions. Naturally, an intrinsic question one would ask is whether we can accurately identify the manipulations over graphs - we term this problem as poisoned graph sanitation. In this paper, we present FocusedCleaner, a poisoned graph sanitation framework consisting of two modules: bi-level structural learning and victim node detection. In particular, the structural learning module will reserve the attack process to steadily sanitize the graph while the detection module provides the "focus" - a narrowed and more accurate search region - to structural learning. These two modules will operate in iterations and reinforce each other to sanitize a poisoned graph step by step. Extensive experiments demonstrate that FocusedCleaner outperforms the state-of-the-art baselines both on poisoned graph sanitation and improving robustness.

摘要: 近年来，很多研究都致力于Web安全的探索，图挖掘算法的对抗健壮性就是一个最具代表性的话题。特别是，一种被广泛应用的对抗性攻击方案是通过修改关系数据来误导图神经网络(GNN)预测的图操纵攻击。自然，人们会问一个内在的问题是，我们是否能准确地识别对图的操纵-我们将这个问题称为有毒的图卫生。在本文中，我们提出了FocusedCleaner，一个由两个模块组成的有毒图健康框架：双层结构学习和受害者节点检测。特别是，结构学习模块将保留攻击过程，以稳定地对图进行杀菌，而检测模块则为结构学习提供“焦点”--一个更窄、更准确的搜索区域。这两个模块将在迭代中运行，并相互加强，逐步清理有毒的图形。广泛的实验表明，FocusedCleaner在有毒的图形卫生和提高健壮性方面都优于最先进的基线。



## **33. Flexible Android Malware Detection Model based on Generative Adversarial Networks with Code Tensor**

基于码张量生成对抗网络的灵活Android恶意软件检测模型 cs.CR

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14225v1) [paper-pdf](http://arxiv.org/pdf/2210.14225v1)

**Authors**: Zhao Yang, Fengyang Deng, Linxi Han

**Abstract**: The behavior of malware threats is gradually increasing, heightened the need for malware detection. However, existing malware detection methods only target at the existing malicious samples, the detection of fresh malicious code and variants of malicious code is limited. In this paper, we propose a novel scheme that detects malware and its variants efficiently. Based on the idea of the generative adversarial networks (GANs), we obtain the `true' sample distribution that satisfies the characteristics of the real malware, use them to deceive the discriminator, thus achieve the defense against malicious code attacks and improve malware detection. Firstly, a new Android malware APK to image texture feature extraction segmentation method is proposed, which is called segment self-growing texture segmentation algorithm. Secondly, tensor singular value decomposition (tSVD) based on the low-tubal rank transforms malicious features with different sizes into a fixed third-order tensor uniformly, which is entered into the neural network for training and learning. Finally, a flexible Android malware detection model based on GANs with code tensor (MTFD-GANs) is proposed. Experiments show that the proposed model can generally surpass the traditional malware detection model, with a maximum improvement efficiency of 41.6\%. At the same time, the newly generated samples of the GANs generator greatly enrich the sample diversity. And retraining malware detector can effectively improve the detection efficiency and robustness of traditional models.

摘要: 恶意软件威胁的行为正在逐渐增加，这加剧了对恶意软件检测的需求。然而，现有的恶意软件检测方法仅针对已有的恶意样本，对新的恶意代码和恶意代码变体的检测有限。本文提出了一种有效检测恶意软件及其变种的新方案。基于生成式对抗网络的思想，我们得到了满足真实恶意软件特征的“真”样本分布，并利用它们来欺骗鉴别器，从而实现了对恶意代码攻击的防御，提高了恶意软件的检测能力。首先，提出了一种新的Android恶意软件APK对图像纹理特征提取的分割方法--分段自增长纹理分割算法。其次，基于低管阶的张量奇异值分解(TSVD)将不同大小的恶意特征统一变换为固定的三阶张量，并输入神经网络进行训练和学习。最后，提出了一种基于编码张量遗传算法的Android恶意软件检测模型(MTFD-GANS)。实验表明，该模型总体上可以超过传统的恶意软件检测模型，最大改进效率为41.6%。同时，Gans生成器新生成的样本极大地丰富了样本的多样性。对恶意软件检测器进行再训练，可以有效提高传统模型的检测效率和鲁棒性。



## **34. Differential Evolution based Dual Adversarial Camouflage: Fooling Human Eyes and Object Detectors**

基于差异进化的双重对抗性伪装：愚弄人眼和目标探测器 cs.CV

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.08870v2) [paper-pdf](http://arxiv.org/pdf/2210.08870v2)

**Authors**: Jialiang Sun, Tingsong Jiang, Wen Yao, Donghua Wang, Xiaoqian Chen

**Abstract**: Recent studies reveal that deep neural network (DNN) based object detectors are vulnerable to adversarial attacks in the form of adding the perturbation to the images, leading to the wrong output of object detectors. Most current existing works focus on generating perturbed images, also called adversarial examples, to fool object detectors. Though the generated adversarial examples themselves can remain a certain naturalness, most of them can still be easily observed by human eyes, which limits their further application in the real world. To alleviate this problem, we propose a differential evolution based dual adversarial camouflage (DE_DAC) method, composed of two stages to fool human eyes and object detectors simultaneously. Specifically, we try to obtain the camouflage texture, which can be rendered over the surface of the object. In the first stage, we optimize the global texture to minimize the discrepancy between the rendered object and the scene images, making human eyes difficult to distinguish. In the second stage, we design three loss functions to optimize the local texture, making object detectors ineffective. In addition, we introduce the differential evolution algorithm to search for the near-optimal areas of the object to attack, improving the adversarial performance under certain attack area limitations. Besides, we also study the performance of adaptive DE_DAC, which can be adapted to the environment. Experiments show that our proposed method could obtain a good trade-off between the fooling human eyes and object detectors under multiple specific scenes and objects.

摘要: 最近的研究表明，基于深度神经网络(DNN)的目标检测器容易受到敌意攻击，其形式是向图像添加扰动，导致目标检测器的输出错误。目前大多数现有的工作都集中在生成扰动图像，也称为对抗性示例，以愚弄对象检测器。尽管生成的对抗性例子本身可以保持一定的自然度，但其中大部分仍然很容易被人眼观察到，这限制了它们在现实世界中的进一步应用。为了缓解这一问题，我们提出了一种基于差异进化的双重对抗伪装(DE_DAC)方法，该方法由两个阶段组成，同时欺骗人眼和目标检测器。具体地说，我们试图获得伪装纹理，它可以在对象的表面上渲染。在第一阶段，我们对全局纹理进行优化，最小化绘制对象和场景图像之间的差异，使人眼难以辨别。在第二阶段，我们设计了三个损失函数来优化局部纹理，使得目标检测失效。此外，我们还引入了差分进化算法来搜索攻击对象的近最优区域，提高了在一定攻击区域限制下的对抗性能。此外，我们还研究了适应环境的自适应DE_DAC的性能。实验表明，在多个特定场景和目标的情况下，我们提出的方法可以在愚弄人眼和目标检测器之间取得良好的折衷。



## **35. Musings on the HashGraph Protocol: Its Security and Its Limitations**

对哈希图协议的思考：其安全性及其局限性 cs.CR

30 pages, 16 figures

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.13682v1) [paper-pdf](http://arxiv.org/pdf/2210.13682v1)

**Authors**: Vinesh Sridhar, Erica Blum, Jonathan Katz

**Abstract**: The HashGraph Protocol is a Byzantine fault tolerant atomic broadcast protocol. Its novel use of locally stored metadata allows parties to recover a consistent ordering of their log just by examining their local data, removing the need for a voting protocol. Our paper's first contribution is to present a rewritten proof of security for the HashGraph Protocol that follows the consistency and liveness paradigm used in the atomic broadcast literature. In our second contribution, we show a novel adversarial strategy that stalls the protocol from committing data to the log for an expected exponential number of rounds. This proves tight the exponential upper bound conjectured in the original paper. We believe that our proof of security will make it easier to compare HashGraph with other atomic broadcast protocols and to incorporate its ideas into new constructions. We also believe that our attack might inspire more research into similar attacks for other DAG-based atomic broadcast protocols.

摘要: 哈希图协议是一种拜占庭容错原子广播协议。它新颖地使用了本地存储的元数据，允许各方仅通过检查其本地数据就可以恢复其日志的一致顺序，从而消除了对投票协议的需要。我们的论文的第一个贡献是为HashGraph协议提供了一个重写的安全性证明，它遵循原子广播文献中使用的一致性和活跃性范例。在我们的第二个贡献中，我们展示了一种新的对抗性策略，该策略使协议无法将数据提交到日志中，达到预期的指数轮数。这证明了原论文中所猜想的指数上界是紧的。我们相信，我们的安全性证明将使我们更容易将HashGraph与其他原子广播协议进行比较，并将其思想融入新的构造中。我们还认为，我们的攻击可能会激发更多对其他基于DAG的原子广播协议的类似攻击的研究。



## **36. Analyzing Privacy Leakage in Machine Learning via Multiple Hypothesis Testing: A Lesson From Fano**

用多重假设检验分析机器学习中的隐私泄露--来自Fano的经验 cs.LG

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.13662v1) [paper-pdf](http://arxiv.org/pdf/2210.13662v1)

**Authors**: Chuan Guo, Alexandre Sablayrolles, Maziar Sanjabi

**Abstract**: Differential privacy (DP) is by far the most widely accepted framework for mitigating privacy risks in machine learning. However, exactly how small the privacy parameter $\epsilon$ needs to be to protect against certain privacy risks in practice is still not well-understood. In this work, we study data reconstruction attacks for discrete data and analyze it under the framework of multiple hypothesis testing. We utilize different variants of the celebrated Fano's inequality to derive upper bounds on the inferential power of a data reconstruction adversary when the model is trained differentially privately. Importantly, we show that if the underlying private data takes values from a set of size $M$, then the target privacy parameter $\epsilon$ can be $O(\log M)$ before the adversary gains significant inferential power. Our analysis offers theoretical evidence for the empirical effectiveness of DP against data reconstruction attacks even at relatively large values of $\epsilon$.

摘要: 差异隐私(DP)是迄今为止被最广泛接受的减轻机器学习中隐私风险的框架。然而，在实践中，隐私参数$\epsilon$需要多小才能防止某些隐私风险仍然没有得到很好的理解。在本工作中，我们研究了离散数据的数据重构攻击，并在多重假设检验的框架下对其进行了分析。我们利用著名的Fano不等式的不同变体来推导数据重建对手在模型被私人差分训练时的推理能力的上界。重要的是，我们证明了如果底层私有数据取自一组大小为$M$的值，则在对手获得显著的推理能力之前，目标隐私参数$\epsilon$可以是$O(\log M)$。我们的分析为DP抵抗数据重建攻击的经验有效性提供了理论证据，即使在相对较大的$\epsilon$的情况下也是如此。



## **37. SpacePhish: The Evasion-space of Adversarial Attacks against Phishing Website Detectors using Machine Learning**

SpacePhish：利用机器学习对钓鱼网站检测器进行敌意攻击的规避空间 cs.CR

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.13660v1) [paper-pdf](http://arxiv.org/pdf/2210.13660v1)

**Authors**: Giovanni Apruzzese, Mauro Conti, Ying Yuan

**Abstract**: Existing literature on adversarial Machine Learning (ML) focuses either on showing attacks that break every ML model, or defenses that withstand most attacks. Unfortunately, little consideration is given to the actual \textit{cost} of the attack or the defense. Moreover, adversarial samples are often crafted in the "feature-space", making the corresponding evaluations of questionable value. Simply put, the current situation does not allow to estimate the actual threat posed by adversarial attacks, leading to a lack of secure ML systems.   We aim to clarify such confusion in this paper. By considering the application of ML for Phishing Website Detection (PWD), we formalize the "evasion-space" in which an adversarial perturbation can be introduced to fool a ML-PWD -- demonstrating that even perturbations in the "feature-space" are useful. Then, we propose a realistic threat model describing evasion attacks against ML-PWD that are cheap to stage, and hence intrinsically more attractive for real phishers. Finally, we perform the first statistically validated assessment of state-of-the-art ML-PWD against 12 evasion attacks. Our evaluation shows (i) the true efficacy of evasion attempts that are more likely to occur; and (ii) the impact of perturbations crafted in different evasion-spaces. Our realistic evasion attempts induce a statistically significant degradation (3-10% at $p\!<$0.05), and their cheap cost makes them a subtle threat. Notably, however, some ML-PWD are immune to our most realistic attacks ($p$=0.22). Our contribution paves the way for a much needed re-assessment of adversarial attacks against ML systems for cybersecurity.

摘要: 现有的关于对抗性机器学习(ML)的文献要么专注于展示打破每个ML模型的攻击，要么专注于抵御大多数攻击的防御。不幸的是，很少考虑攻击或防御的实际成本。此外，对抗性样本往往是在“特征空间”中制作的，使得相应的评估价值值得怀疑。简单地说，目前的情况不允许估计对抗性攻击构成的实际威胁，导致缺乏安全的ML系统。我们的目的是在这篇论文中澄清这种混淆。通过考虑ML在钓鱼网站检测(PWD)中的应用，我们形式化了“规避空间”，在该空间中可以引入敌意扰动来愚弄ML-PWD--表明即使在“特征空间”中的扰动也是有用的。然后，我们提出了一个真实的威胁模型，描述了针对ML-PWD的逃避攻击，这些攻击的实施成本很低，因此本质上对真正的网络钓鱼者更具吸引力。最后，我们对最先进的ML-PWD进行了第一次统计验证评估，以对抗12次逃避攻击。我们的评估显示了(I)更有可能发生的逃避尝试的真实效果；以及(Ii)在不同的逃避空间中制造的扰动的影响。我们的现实规避尝试导致了统计上显著的下降(3%-10%，在$p<$0.05)，而且它们的廉价成本使它们成为一个微妙的威胁。然而，值得注意的是，一些ML-PWD对我们最现实的攻击($p$=0.22)是免疫的。我们的贡献为重新评估针对ML网络安全系统的对抗性攻击铺平了道路。



## **38. On the Robustness of Dataset Inference**

关于数据集推理的稳健性 cs.LG

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.13631v1) [paper-pdf](http://arxiv.org/pdf/2210.13631v1)

**Authors**: Sebastian Szyller, Rui Zhang, Jian Liu, N. Asokan

**Abstract**: Machine learning (ML) models are costly to train as they can require a significant amount of data, computational resources and technical expertise. Thus, they constitute valuable intellectual property that needs protection from adversaries wanting to steal them. Ownership verification techniques allow the victims of model stealing attacks to demonstrate that a suspect model was in fact stolen from theirs. Although a number of ownership verification techniques based on watermarking or fingerprinting have been proposed, most of them fall short either in terms of security guarantees (well-equipped adversaries can evade verification) or computational cost. A fingerprinting technique introduced at ICLR '21, Dataset Inference (DI), has been shown to offer better robustness and efficiency than prior methods. The authors of DI provided a correctness proof for linear (suspect) models. However, in the same setting, we prove that DI suffers from high false positives (FPs) -- it can incorrectly identify an independent model trained with non-overlapping data from the same distribution as stolen. We further prove that DI also triggers FPs in realistic, non-linear suspect models. We then confirm empirically that DI leads to FPs, with high confidence. Second, we show that DI also suffers from false negatives (FNs) -- an adversary can fool DI by regularising a stolen model's decision boundaries using adversarial training, thereby leading to an FN. To this end, we demonstrate that DI fails to identify a model adversarially trained from a stolen dataset -- the setting where DI is the hardest to evade. Finally, we discuss the implications of our findings, the viability of fingerprinting-based ownership verification in general, and suggest directions for future work.

摘要: 机器学习(ML)模型的训练成本很高，因为它们可能需要大量的数据、计算资源和技术专长。因此，它们构成了宝贵的知识产权，需要保护，不受想要窃取它们的对手的攻击。所有权验证技术允许模型盗窃攻击的受害者证明可疑模型实际上是从他们的模型中被盗的。虽然已经提出了一些基于水印或指纹的所有权验证技术，但它们大多在安全保证(装备良好的攻击者可以逃避验证)或计算代价方面存在不足。在ICLR‘21上引入的一种指纹技术，数据集推理(DI)，已经被证明比以前的方法提供了更好的稳健性和效率。DI的作者为线性(可疑)模型提供了正确性证明。然而，在相同的设置中，我们证明了DI存在高误报(FP)--它可能错误地识别使用来自相同分布的非重叠数据训练的独立模型作为被盗。我们进一步证明，在现实的、非线性的可疑模型中，依赖注入也会触发FP。然后，我们以很高的置信度从经验上证实了DI会导致FP。其次，我们证明了DI也存在假阴性(FN)--对手可以通过使用对抗性训练来调整被盗模型的决策边界来愚弄DI，从而导致FN。为此，我们演示了DI无法识别从窃取的数据集中恶意训练的模型--DI最难逃避的设置。最后，我们讨论了我们的发现的含义，基于指纹的所有权验证总体上的可行性，并对未来的工作提出了方向。



## **39. Deep VULMAN: A Deep Reinforcement Learning-Enabled Cyber Vulnerability Management Framework**

Deep VULMAN：一种深度强化学习的网络漏洞管理框架 cs.AI

12 pages, 3 figures

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2208.02369v2) [paper-pdf](http://arxiv.org/pdf/2208.02369v2)

**Authors**: Soumyadeep Hore, Ankit Shah, Nathaniel D. Bastian

**Abstract**: Cyber vulnerability management is a critical function of a cybersecurity operations center (CSOC) that helps protect organizations against cyber-attacks on their computer and network systems. Adversaries hold an asymmetric advantage over the CSOC, as the number of deficiencies in these systems is increasing at a significantly higher rate compared to the expansion rate of the security teams to mitigate them in a resource-constrained environment. The current approaches are deterministic and one-time decision-making methods, which do not consider future uncertainties when prioritizing and selecting vulnerabilities for mitigation. These approaches are also constrained by the sub-optimal distribution of resources, providing no flexibility to adjust their response to fluctuations in vulnerability arrivals. We propose a novel framework, Deep VULMAN, consisting of a deep reinforcement learning agent and an integer programming method to fill this gap in the cyber vulnerability management process. Our sequential decision-making framework, first, determines the near-optimal amount of resources to be allocated for mitigation under uncertainty for a given system state and then determines the optimal set of prioritized vulnerability instances for mitigation. Our proposed framework outperforms the current methods in prioritizing the selection of important organization-specific vulnerabilities, on both simulated and real-world vulnerability data, observed over a one-year period.

摘要: 网络漏洞管理是网络安全运营中心(CSOC)的一项重要职能，有助于保护组织免受对其计算机和网络系统的网络攻击。与CSOC相比，对手拥有不对称的优势，因为与安全团队的扩张率相比，这些系统中的缺陷数量正在以显著更高的速度增加，以在资源受限的环境中缓解这些缺陷。目前的方法是确定性和一次性决策方法，在确定和选择要缓解的脆弱性时，不考虑未来的不确定性。这些办法还受到资源分配次优的限制，无法灵活地调整其对脆弱抵达人数波动的反应。我们提出了一种新的框架--Deep VULMAN，它由深度强化学习代理和整数规划方法组成，以填补网络漏洞管理过程中的这一空白。我们的顺序决策框架首先确定在给定系统状态下的不确定性情况下为缓解而分配的接近最优的资源量，然后确定用于缓解的最优优先级漏洞实例集。我们提出的框架在优先选择重要的特定于组织的漏洞方面优于目前的方法，该方法基于模拟和真实世界的漏洞数据，在一年的时间内观察到。



## **40. Probabilistic Categorical Adversarial Attack & Adversarial Training**

概率分类对抗性攻击与对抗性训练 cs.LG

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.09364v2) [paper-pdf](http://arxiv.org/pdf/2210.09364v2)

**Authors**: Pengfei He, Han Xu, Jie Ren, Yuxuan Wan, Zitao Liu, Jiliang Tang

**Abstract**: The existence of adversarial examples brings huge concern for people to apply Deep Neural Networks (DNNs) in safety-critical tasks. However, how to generate adversarial examples with categorical data is an important problem but lack of extensive exploration. Previously established methods leverage greedy search method, which can be very time-consuming to conduct successful attack. This also limits the development of adversarial training and potential defenses for categorical data. To tackle this problem, we propose Probabilistic Categorical Adversarial Attack (PCAA), which transfers the discrete optimization problem to a continuous problem that can be solved efficiently by Projected Gradient Descent. In our paper, we theoretically analyze its optimality and time complexity to demonstrate its significant advantage over current greedy based attacks. Moreover, based on our attack, we propose an efficient adversarial training framework. Through a comprehensive empirical study, we justify the effectiveness of our proposed attack and defense algorithms.

摘要: 对抗性实例的存在给深度神经网络在安全关键任务中的应用带来了极大的关注。然而，如何利用分类数据生成对抗性实例是一个重要的问题，但缺乏广泛的探索。以前建立的方法利用贪婪搜索方法，进行成功的攻击可能非常耗时。这也限制了对抗性训练的发展和对分类数据的潜在防御。为了解决这个问题，我们提出了概率分类对抗性攻击(PCAA)，它将离散的优化问题转化为一个连续的问题，可以用投影梯度下降法有效地解决。在本文中，我们从理论上分析了它的最优性和时间复杂性，以证明它相对于现有的基于贪婪的攻击具有显著的优势。此外，基于我们的攻击，我们提出了一个有效的对抗性训练框架。通过全面的实证研究，验证了本文提出的攻防算法的有效性。



## **41. Driver Locations Harvesting Attack on pRide**

司机位置收割对Pride的攻击 cs.CR

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.13263v1) [paper-pdf](http://arxiv.org/pdf/2210.13263v1)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstract**: Privacy preservation in Ride-Hailing Services (RHS) is intended to protect privacy of drivers and riders. pRide, published in IEEE Trans. Vehicular Technology 2021, is a prediction based privacy-preserving RHS protocol to match riders with an optimum driver. In the protocol, the Service Provider (SP) homomorphically computes Euclidean distances between encrypted locations of drivers and rider. Rider selects an optimum driver using decrypted distances augmented by a new-ride-emergence prediction. To improve the effectiveness of driver selection, the paper proposes an enhanced version where each driver gives encrypted distances to each corner of her grid. To thwart a rider from using these distances to launch an inference attack, the SP blinds these distances before sharing them with the rider. In this work, we propose a passive attack where an honest-but-curious adversary rider who makes a single ride request and receives the blinded distances from SP can recover the constants used to blind the distances. Using the unblinded distances, rider to driver distance and Google Nearest Road API, the adversary can obtain the precise locations of responding drivers. We conduct experiments with random on-road driver locations for four different cities. Our experiments show that we can determine the precise locations of at least 80% of the drivers participating in the enhanced pRide protocol.

摘要: 网约车服务(RHS)中的隐私保护旨在保护司机和乘客的隐私。Pride，发表在IEEE Trans上。Vehicular Technology 2021是一种基于预测的隐私保护RHS协议，用于将乘客与最佳司机进行匹配。在该协议中，服务提供商(SP)同态地计算司机和乘客的加密位置之间的欧几里德距离。骑手使用解密的距离选择最优的司机，并增加了一个新的乘车出现预测。为了提高驾驶员选择的有效性，本文提出了一种增强版本，每个驾驶员给出了到其网格每个角落的加密距离。为了阻止骑手使用这些距离来发动推理攻击，SP在与骑手共享这些距离之前会先隐藏这些距离。在这项工作中，我们提出了一种被动攻击，在这种攻击中，诚实但好奇的敌方骑手发出一个骑行请求，并从SP接收到盲距离，就可以恢复用于盲距离的常量。使用非盲目距离、骑手到司机的距离和谷歌最近道路API，对手可以获得回应司机的准确位置。我们对四个不同城市的随机道路司机位置进行了实验。我们的实验表明，我们可以确定至少80%参与增强PROID协议的司机的准确位置。



## **42. SealClub: Computer-aided Paper Document Authentication**

SealClub：计算机辅助纸质文档认证 cs.CR

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.07884v2) [paper-pdf](http://arxiv.org/pdf/2210.07884v2)

**Authors**: Martín Ochoa, Jorge Toro-Pozo, David Basin

**Abstract**: Digital authentication is a mature field, offering a range of solutions with rigorous mathematical guarantees. Nevertheless, paper documents, where cryptographic techniques are not directly applicable, are still widely utilized due to usability and legal reasons. We propose a novel approach to authenticating paper documents using smartphones by taking short videos of them. Our solution combines cryptographic and image comparison techniques to detect and highlight subtle semantic-changing attacks on rich documents, containing text and graphics, that could go unnoticed by humans. We rigorously analyze our approach, proving that it is secure against strong adversaries capable of compromising different system components. We also measure its accuracy empirically on a set of 128 videos of paper documents, half containing subtle forgeries. Our algorithm finds all forgeries accurately (no false alarms) after analyzing 5.13 frames on average (corresponding to 1.28 seconds of video). Highlighted regions are large enough to be visible to users, but small enough to precisely locate forgeries. Thus, our approach provides a promising way for users to authenticate paper documents using conventional smartphones under realistic conditions.

摘要: 数字身份验证是一个成熟的领域，提供了一系列具有严格数学保证的解决方案。然而，由于可用性和法律原因，在加密技术不直接适用的情况下，纸质文档仍然被广泛使用。我们提出了一种通过拍摄短视频来使用智能手机对纸质文档进行身份验证的新方法。我们的解决方案结合了加密和图像比较技术，以检测和突出对包含文本和图形的丰富文档的细微语义变化攻击，这些攻击可能不会被人类注意到。我们严格分析了我们的方法，证明了它是安全的，可以抵御能够危害不同系统组件的强大对手。我们还在一组128个纸质文档的视频上对其准确性进行了经验性的测量，其中一半包含微妙的伪造。该算法在平均分析5.13帧(对应于1.28秒的视频)后，准确地发现了所有的伪造(没有虚警)。突出显示的区域足够大，用户可以看到，但也足够小，可以精确定位假货。因此，我们的方法为用户在现实条件下使用传统的智能手机认证纸质文档提供了一种很有前途的方法。



## **43. Sardino: Ultra-Fast Dynamic Ensemble for Secure Visual Sensing at Mobile Edge**

Sardino：移动边缘安全视觉感知的超快动态合奏 cs.CV

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2204.08189v3) [paper-pdf](http://arxiv.org/pdf/2204.08189v3)

**Authors**: Qun Song, Zhenyu Yan, Wenjie Luo, Rui Tan

**Abstract**: Adversarial example attack endangers the mobile edge systems such as vehicles and drones that adopt deep neural networks for visual sensing. This paper presents {\em Sardino}, an active and dynamic defense approach that renews the inference ensemble at run time to develop security against the adaptive adversary who tries to exfiltrate the ensemble and construct the corresponding effective adversarial examples. By applying consistency check and data fusion on the ensemble's predictions, Sardino can detect and thwart adversarial inputs. Compared with the training-based ensemble renewal, we use HyperNet to achieve {\em one million times} acceleration and per-frame ensemble renewal that presents the highest level of difficulty to the prerequisite exfiltration attacks. We design a run-time planner that maximizes the ensemble size in favor of security while maintaining the processing frame rate. Beyond adversarial examples, Sardino can also address the issue of out-of-distribution inputs effectively. This paper presents extensive evaluation of Sardino's performance in counteracting adversarial examples and applies it to build a real-time car-borne traffic sign recognition system. Live on-road tests show the built system's effectiveness in maintaining frame rate and detecting out-of-distribution inputs due to the false positives of a preceding YOLO-based traffic sign detector.

摘要: 对抗性示例攻击危及采用深度神经网络进行视觉传感的移动边缘系统，如车辆和无人机。提出了一种主动的、动态的防御方法{em Sardino}，该方法在运行时更新推理集成，以提高安全性，防止自适应对手试图渗透集成并构造相应的有效对抗实例。通过对合奏的预测应用一致性检查和数据融合，萨迪诺可以检测和挫败敌方的输入。与基于训练的集成更新相比，我们使用HyperNet实现了加速和每帧集成更新，这对先决条件渗透攻击呈现出最高的难度。我们设计了一个运行时规划器，在保持处理帧速率的同时最大化集成大小以利于安全性。除了敌对的例子，萨迪诺还可以有效地解决分配外投入的问题。本文对Sardino在对抗敌意例子方面的表现进行了广泛的评估，并将其应用于构建一个实时车载交通标志识别系统。现场道路测试表明，所建立的系统在保持帧速率和检测由于先前基于YOLO的交通标志检测器的错误阳性而导致的不分布输入方面是有效的。



## **44. Ares: A System-Oriented Wargame Framework for Adversarial ML**

ARES：一种面向系统的对抗性ML战争游戏框架 cs.LG

Presented at the DLS Workshop at S&P 2022

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.12952v1) [paper-pdf](http://arxiv.org/pdf/2210.12952v1)

**Authors**: Farhan Ahmed, Pratik Vaishnavi, Kevin Eykholt, Amir Rahmati

**Abstract**: Since the discovery of adversarial attacks against machine learning models nearly a decade ago, research on adversarial machine learning has rapidly evolved into an eternal war between defenders, who seek to increase the robustness of ML models against adversarial attacks, and adversaries, who seek to develop better attacks capable of weakening or defeating these defenses. This domain, however, has found little buy-in from ML practitioners, who are neither overtly concerned about these attacks affecting their systems in the real world nor are willing to trade off the accuracy of their models in pursuit of robustness against these attacks.   In this paper, we motivate the design and implementation of Ares, an evaluation framework for adversarial ML that allows researchers to explore attacks and defenses in a realistic wargame-like environment. Ares frames the conflict between the attacker and defender as two agents in a reinforcement learning environment with opposing objectives. This allows the introduction of system-level evaluation metrics such as time to failure and evaluation of complex strategies such as moving target defenses. We provide the results of our initial exploration involving a white-box attacker against an adversarially trained defender.

摘要: 自从近十年前发现了针对机器学习模型的对抗性攻击以来，对抗性机器学习的研究迅速演变为防御者和对手之间的一场永恒的战争。防御者试图增加ML模型对对抗性攻击的健壮性，而对手试图开发能够削弱或击败这些防御的更好的攻击。然而，这个领域几乎没有得到ML从业者的认可，他们既不公开担心这些攻击会影响他们在现实世界中的系统，也不愿意牺牲他们模型的准确性来追求对这些攻击的健壮性。在本文中，我们推动了ARES的设计和实现，这是一个针对对抗性ML的评估框架，允许研究人员在现实的类似战争游戏的环境中探索攻击和防御。阿瑞斯将攻击者和防御者之间的冲突框架为具有相反目标的强化学习环境中的两个代理。这允许引入系统级评估指标，如故障发生时间，以及评估复杂战略，如移动目标防御。我们提供了我们的初步探索的结果，涉及一个白盒攻击者对一个对手训练的后卫。



## **45. Backdoor Attacks in Federated Learning by Rare Embeddings and Gradient Ensembling**

基于稀有嵌入和梯度集成的联合学习后门攻击 cs.LG

Accepted to EMNLP 2022, 9 pages and Appendix

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2204.14017v2) [paper-pdf](http://arxiv.org/pdf/2204.14017v2)

**Authors**: KiYoon Yoo, Nojun Kwak

**Abstract**: Recent advances in federated learning have demonstrated its promising capability to learn on decentralized datasets. However, a considerable amount of work has raised concerns due to the potential risks of adversaries participating in the framework to poison the global model for an adversarial purpose. This paper investigates the feasibility of model poisoning for backdoor attacks through rare word embeddings of NLP models. In text classification, less than 1% of adversary clients suffices to manipulate the model output without any drop in the performance on clean sentences. For a less complex dataset, a mere 0.1% of adversary clients is enough to poison the global model effectively. We also propose a technique specialized in the federated learning scheme called Gradient Ensemble, which enhances the backdoor performance in all our experimental settings.

摘要: 联邦学习的最新进展已经证明了它在分散的数据集上学习的前景。然而，由于参与该框架的对手出于对抗目的而破坏全球模式的潜在风险，大量工作引起了关注。通过对自然语言处理模型的稀有词嵌入，研究了模型中毒用于后门攻击的可行性。在文本分类中，只有不到1%的敌意客户端足以在不降低干净句子性能的情况下操纵模型输出。对于不太复杂的数据集，仅0.1%的恶意客户端就足以有效地毒化全球模型。我们还提出了一种专门用于联邦学习方案的技术，称为梯度集成，它在所有实验设置中都提高了后门性能。



## **46. TextHacker: Learning based Hybrid Local Search Algorithm for Text Hard-label Adversarial Attack**

TextHacker：用于文本硬标签攻击的基于学习的混合局部搜索算法 cs.CL

Accepted by EMNLP 2022 Findings, Code is available at  https://github.com/JHL-HUST/TextHacker

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2201.08193v2) [paper-pdf](http://arxiv.org/pdf/2201.08193v2)

**Authors**: Zhen Yu, Xiaosen Wang, Wanxiang Che, Kun He

**Abstract**: Existing textual adversarial attacks usually utilize the gradient or prediction confidence to generate adversarial examples, making it hard to be deployed in real-world applications. To this end, we consider a rarely investigated but more rigorous setting, namely hard-label attack, in which the attacker can only access the prediction label. In particular, we find we can learn the importance of different words via the change on prediction label caused by word substitutions on the adversarial examples. Based on this observation, we propose a novel adversarial attack, termed Text Hard-label attacker (TextHacker). TextHacker randomly perturbs lots of words to craft an adversarial example. Then, TextHacker adopts a hybrid local search algorithm with the estimation of word importance from the attack history to minimize the adversarial perturbation. Extensive evaluations for text classification and textual entailment show that TextHacker significantly outperforms existing hard-label attacks regarding the attack performance as well as adversary quality.

摘要: 现有的文本对抗性攻击通常利用梯度或预测置信度来生成对抗性实例，这使得它很难应用于实际应用中。为此，我们考虑了一种很少被研究但更严格的环境，即硬标签攻击，在这种攻击中，攻击者只能访问预测标签。特别是，我们发现我们可以通过对抗性例子上的单词替换引起的预测标签的变化来了解不同单词的重要性。基于此，我们提出了一种新的对抗性攻击，称为文本硬标签攻击者(TextHacker)。TextHacker随机扰乱大量单词来制作一个对抗性的例子。然后，TextHacker采用了一种混合局部搜索算法，并从攻击历史中估计单词的重要性，以最小化对手的扰动。对文本分类和文本蕴涵的广泛评估表明，TextHacker在攻击性能和对手质量方面都明显优于现有的硬标签攻击。



## **47. A Secure Design Pattern Approach Toward Tackling Lateral-Injection Attacks**

一种应对横向注入攻击的安全设计模式方法 cs.CR

4 pages, 3 figures. Accepted to The 15th IEEE International  Conference on Security of Information and Networks (SIN)

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.12877v1) [paper-pdf](http://arxiv.org/pdf/2210.12877v1)

**Authors**: Chidera Biringa, Gökhan Kul

**Abstract**: Software weaknesses that create attack surfaces for adversarial exploits, such as lateral SQL injection (LSQLi) attacks, are usually introduced during the design phase of software development. Security design patterns are sometimes applied to tackle these weaknesses. However, due to the stealthy nature of lateral-based attacks, employing traditional security patterns to address these threats is insufficient. Hence, we present SEAL, a secure design that extrapolates architectural, design, and implementation abstraction levels to delegate security strategies toward tackling LSQLi attacks. We evaluated SEAL using case study software, where we assumed the role of an adversary and injected several attack vectors tasked with compromising the confidentiality and integrity of its database. Our evaluation of SEAL demonstrated its capacity to address LSQLi attacks.

摘要: 在软件开发的设计阶段，通常会引入为对抗性攻击(如横向SQL注入(LSQLi)攻击)创建攻击面的软件弱点。有时会应用安全设计模式来解决这些弱点。然而，由于基于侧向的攻击的隐蔽性，采用传统的安全模式来应对这些威胁是不够的。因此，我们提出了SEAL，这是一种安全设计，它推断出体系结构、设计和实现抽象级别，以委派安全策略来应对LSQLi攻击。我们使用案例研究软件评估了海豹突击队，我们扮演了一个对手的角色，并注入了几个攻击载体，任务是危及其数据库的机密性和完整性。我们对海豹突击队的评估表明，它有能力应对LSQLi攻击。



## **48. TAPE: Assessing Few-shot Russian Language Understanding**

录像带：评估不太可能的俄语理解 cs.CL

Accepted to EMNLP 2022 Findings

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.12813v1) [paper-pdf](http://arxiv.org/pdf/2210.12813v1)

**Authors**: Ekaterina Taktasheva, Tatiana Shavrina, Alena Fenogenova, Denis Shevelev, Nadezhda Katricheva, Maria Tikhonova, Albina Akhmetgareeva, Oleg Zinkevich, Anastasiia Bashmakova, Svetlana Iordanskaia, Alena Spiridonova, Valentina Kurenshchikova, Ekaterina Artemova, Vladislav Mikhailov

**Abstract**: Recent advances in zero-shot and few-shot learning have shown promise for a scope of research and practical purposes. However, this fast-growing area lacks standardized evaluation suites for non-English languages, hindering progress outside the Anglo-centric paradigm. To address this line of research, we propose TAPE (Text Attack and Perturbation Evaluation), a novel benchmark that includes six more complex NLU tasks for Russian, covering multi-hop reasoning, ethical concepts, logic and commonsense knowledge. The TAPE's design focuses on systematic zero-shot and few-shot NLU evaluation: (i) linguistic-oriented adversarial attacks and perturbations for analyzing robustness, and (ii) subpopulations for nuanced interpretation. The detailed analysis of testing the autoregressive baselines indicates that simple spelling-based perturbations affect the performance the most, while paraphrasing the input has a more negligible effect. At the same time, the results demonstrate a significant gap between the neural and human baselines for most tasks. We publicly release TAPE (tape-benchmark.com) to foster research on robust LMs that can generalize to new tasks when little to no supervision is available.

摘要: 零射击和少射击学习的最新进展显示出了研究范围和实用目的的前景。然而，这个快速发展的领域缺乏针对非英语语言的标准化评估套件，阻碍了以英语为中心的范式之外的进步。针对这一研究方向，我们提出了TAPE(文本攻击和扰动评估)，这是一个新的基准测试，包括六个更复杂的俄语自然语言理解任务，涵盖了多跳推理、伦理概念、逻辑和常识知识。这盘磁带的设计侧重于系统的零镜头和少镜头NLU评估：(I)面向语言的对抗性攻击和扰动，用于分析稳健性；(Ii)亚群，用于细微差别的解释。对自回归基线测试的详细分析表明，基于拼写的简单扰动对成绩的影响最大，而释义输入的影响较小。与此同时，结果表明，在大多数任务中，神经基线和人类基线之间存在着显著的差距。我们公开发布磁带(Tape-Benchmark.com)，以促进对健壮的LMS的研究，这些LMS可以在几乎没有监督的情况下推广到新任务。



## **49. Adversarial Pretraining of Self-Supervised Deep Networks: Past, Present and Future**

自监督深度网络的对抗性预训练：过去、现在和未来 cs.LG

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.13463v1) [paper-pdf](http://arxiv.org/pdf/2210.13463v1)

**Authors**: Guo-Jun Qi, Mubarak Shah

**Abstract**: In this paper, we review adversarial pretraining of self-supervised deep networks including both convolutional neural networks and vision transformers. Unlike the adversarial training with access to labeled examples, adversarial pretraining is complicated as it only has access to unlabeled examples. To incorporate adversaries into pretraining models on either input or feature level, we find that existing approaches are largely categorized into two groups: memory-free instance-wise attacks imposing worst-case perturbations on individual examples, and memory-based adversaries shared across examples over iterations. In particular, we review several representative adversarial pretraining models based on Contrastive Learning (CL) and Masked Image Modeling (MIM), respectively, two popular self-supervised pretraining methods in literature. We also review miscellaneous issues about computing overheads, input-/feature-level adversaries, as well as other adversarial pretraining approaches beyond the above two groups. Finally, we discuss emerging trends and future directions about the relations between adversarial and cooperative pretraining, unifying adversarial CL and MIM pretraining, and the trade-off between accuracy and robustness in adversarial pretraining.

摘要: 本文回顾了自监督深度网络的对抗性预训练，包括卷积神经网络和视觉转换器。与获得标记样本的对抗性训练不同，对抗性预训练是复杂的，因为它只能访问未标记的样本。为了将对手纳入到输入或特征级别的预训练模型中，我们发现现有的方法主要分为两类：无记忆的实例攻击对单个实例施加最坏情况的扰动，以及基于记忆的对手在迭代过程中跨实例共享。特别是，我们回顾了几种代表性的基于对比学习(CL)和掩蔽图像建模(MIM)的对抗性预训练模型，这两种方法是文献中流行的两种自我监督预训练方法。我们还审查了有关计算管理费用、输入/特征级别的对手以及以上两组以外的其他对抗性预训练方法的杂项问题。最后，我们讨论了对抗性预训练和合作预训练之间的关系，对抗性CL和MIM预训练的统一，以及对抗性预训练中准确性和稳健性之间的权衡等方面的发展趋势和未来方向。



## **50. GANI: Global Attacks on Graph Neural Networks via Imperceptible Node Injections**

GANI：基于不可察觉节点注入的图神经网络全局攻击 cs.LG

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.12598v1) [paper-pdf](http://arxiv.org/pdf/2210.12598v1)

**Authors**: Junyuan Fang, Haixian Wen, Jiajing Wu, Qi Xuan, Zibin Zheng, Chi K. Tse

**Abstract**: Graph neural networks (GNNs) have found successful applications in various graph-related tasks. However, recent studies have shown that many GNNs are vulnerable to adversarial attacks. In a vast majority of existing studies, adversarial attacks on GNNs are launched via direct modification of the original graph such as adding/removing links, which may not be applicable in practice. In this paper, we focus on a realistic attack operation via injecting fake nodes. The proposed Global Attack strategy via Node Injection (GANI) is designed under the comprehensive consideration of an unnoticeable perturbation setting from both structure and feature domains. Specifically, to make the node injections as imperceptible and effective as possible, we propose a sampling operation to determine the degree of the newly injected nodes, and then generate features and select neighbors for these injected nodes based on the statistical information of features and evolutionary perturbations obtained from a genetic algorithm, respectively. In particular, the proposed feature generation mechanism is suitable for both binary and continuous node features. Extensive experimental results on benchmark datasets against both general and defended GNNs show strong attack performance of GANI. Moreover, the imperceptibility analyses also demonstrate that GANI achieves a relatively unnoticeable injection on benchmark datasets.

摘要: 图神经网络(GNN)在各种与图相关的任务中得到了成功的应用。然而，最近的研究表明，许多GNN容易受到对抗性攻击。在现有的绝大多数研究中，对GNN的对抗性攻击是通过直接修改原始图形来发起的，例如添加/删除链接，这在实践中可能并不适用。在本文中，我们关注的是一种通过注入伪节点进行的真实攻击操作。通过节点注入的全局攻击策略(GANI)是在综合考虑结构域和特征域中不可察觉的扰动设置的基础上设计的。具体地说，为了使节点注入尽可能隐蔽和有效，我们提出了一种抽样操作来确定新注入节点的程度，然后根据遗传算法获得的特征统计信息和进化扰动分别为这些注入节点生成特征和选择邻居。特别是，所提出的特征生成机制既适用于二进制节点特征，也适用于连续节点特征。在基准数据集上对一般GNN和防御GNN的大量实验结果表明，GANI具有很强的攻击性能。此外，不可感知性分析还表明，GANI在基准数据集上实现了相对不明显的注入。



