# Latest Adversarial Attack Papers
**update at 2022-10-06 16:36:25**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Dynamical systems' based neural networks**

基于动力系统的神经网络 cs.LG

25 pages with 2 appendices

**SubmitDate**: 2022-10-05    [abs](http://arxiv.org/abs/2210.02373v1) [paper-pdf](http://arxiv.org/pdf/2210.02373v1)

**Authors**: Elena Celledoni, Davide Murari, Brynjulf Owren, Carola-Bibiane Schönlieb, Ferdia Sherry

**Abstract**: Neural networks have gained much interest because of their effectiveness in many applications. However, their mathematical properties are generally not well understood. If there is some underlying geometric structure inherent to the data or to the function to approximate, it is often desirable to take this into account in the design of the neural network. In this work, we start with a non-autonomous ODE and build neural networks using a suitable, structure-preserving, numerical time-discretisation. The structure of the neural network is then inferred from the properties of the ODE vector field. Besides injecting more structure into the network architectures, this modelling procedure allows a better theoretical understanding of their behaviour. We present two universal approximation results and demonstrate how to impose some particular properties on the neural networks. A particular focus is on 1-Lipschitz architectures including layers that are not 1-Lipschitz. These networks are expressive and robust against adversarial attacks, as shown for the CIFAR-10 dataset.

摘要: 神经网络因其在许多应用中的有效性而引起人们的极大兴趣。然而，它们的数学性质通常没有被很好地理解。如果数据或函数具有某种内在的几何结构，那么在设计神经网络时往往需要考虑到这一点。在这项工作中，我们从一个非自治的常微分方程开始，并使用适当的、结构保持的、数值时间离散化来建立神经网络。然后从ODE向量场的特性推断出神经网络的结构。除了向网络架构注入更多结构外，此建模过程还允许更好地从理论上理解它们的行为。我们给出了两个普遍的逼近结果，并演示了如何对神经网络施加一些特殊性质。一个特别的焦点是1-Lipschitz体系结构，包括不是1-Lipschitz的层。正如CIFAR-10数据集所显示的那样，这些网络具有很强的表现力和对对手攻击的健壮性。



## **2. Can Adversarial Training Be Manipulated By Non-Robust Features?**

对抗性训练能被非强健特征操纵吗？ cs.LG

NeurIPS 2022

**SubmitDate**: 2022-10-05    [abs](http://arxiv.org/abs/2201.13329v3) [paper-pdf](http://arxiv.org/pdf/2201.13329v3)

**Authors**: Lue Tao, Lei Feng, Hongxin Wei, Jinfeng Yi, Sheng-Jun Huang, Songcan Chen

**Abstract**: Adversarial training, originally designed to resist test-time adversarial examples, has shown to be promising in mitigating training-time availability attacks. This defense ability, however, is challenged in this paper. We identify a novel threat model named stability attacks, which aims to hinder robust availability by slightly manipulating the training data. Under this threat, we show that adversarial training using a conventional defense budget $\epsilon$ provably fails to provide test robustness in a simple statistical setting, where the non-robust features of the training data can be reinforced by $\epsilon$-bounded perturbation. Further, we analyze the necessity of enlarging the defense budget to counter stability attacks. Finally, comprehensive experiments demonstrate that stability attacks are harmful on benchmark datasets, and thus the adaptive defense is necessary to maintain robustness. Our code is available at https://github.com/TLMichael/Hypocritical-Perturbation.

摘要: 对抗性训练最初是为了抵抗测试时间对抗性的例子，已经被证明在减轻训练时间可用性攻击方面很有希望。然而，这种防御能力在本文中受到了挑战。我们识别了一种名为稳定性攻击的新威胁模型，该模型旨在通过稍微操纵训练数据来阻碍健壮性可用性。在这种威胁下，我们证明了在简单的统计设置下，使用常规国防预算的对抗性训练不能提供测试稳健性，其中训练数据的非稳健性特征可以通过有界扰动来加强。进一步，我们分析了增加国防预算以对抗稳定性攻击的必要性。最后，综合实验表明，稳定性攻击对基准数据集是有害的，因此需要自适应防御来保持稳健性。我们的代码可以在https://github.com/TLMichael/Hypocritical-Perturbation.上找到



## **3. Image Masking for Robust Self-Supervised Monocular Depth Estimation**

用于稳健自监督单目深度估计的图像掩蔽算法 cs.CV

**SubmitDate**: 2022-10-05    [abs](http://arxiv.org/abs/2210.02357v1) [paper-pdf](http://arxiv.org/pdf/2210.02357v1)

**Authors**: Hemang Chawla, Kishaan Jeeveswaran, Elahe Arani, Bahram Zonooz

**Abstract**: Self-supervised monocular depth estimation is a salient task for 3D scene understanding. Learned jointly with monocular ego-motion estimation, several methods have been proposed to predict accurate pixel-wise depth without using labeled data. Nevertheless, these methods focus on improving performance under ideal conditions without natural or digital corruptions. A general absence of occlusions is assumed even for object-specific depth estimation. These methods are also vulnerable to adversarial attacks, which is a pertinent concern for their reliable deployment on robots and autonomous driving systems. We propose MIMDepth, a method that adapts masked image modeling (MIM) for self-supervised monocular depth estimation. While MIM has been used to learn generalizable features during pre-training, we show how it could be adapted for direct training of monocular depth estimation. Our experiments show that MIMDepth is more robust to noise, blur, weather conditions, digital artifacts, occlusions, as well as untargeted and targeted adversarial attacks.

摘要: 自监督单目深度估计是三维场景理解的一项重要任务。与单目自我运动估计联合学习，已经提出了几种方法来预测准确的像素深度，而不使用标记数据。然而，这些方法侧重于在没有自然或数字损坏的理想条件下提高性能。即使对于特定于对象的深度估计，也假定一般不存在遮挡。这些方法也容易受到对手攻击，这是一个相关的担忧，因为它们可靠地部署在机器人和自动驾驶系统上。提出了一种自监督单目深度估计的MIMDepth方法。虽然MIM已经被用来在预训练中学习泛化特征，但我们展示了它如何适用于单眼深度估计的直接训练。我们的实验表明，MIMDepth对噪声、模糊、天气条件、数字伪影、遮挡以及非定向和定向的对抗性攻击具有更强的鲁棒性。



## **4. Jitter Does Matter: Adapting Gaze Estimation to New Domains**

抖动确实很重要：使凝视估计适应新的领域 cs.CV

7 pages, 5 figures

**SubmitDate**: 2022-10-05    [abs](http://arxiv.org/abs/2210.02082v1) [paper-pdf](http://arxiv.org/pdf/2210.02082v1)

**Authors**: Ruicong Liu, Yiwei Bao, Mingjie Xu, Haofei Wang, Yunfei Liu, Feng Lu

**Abstract**: Deep neural networks have demonstrated superior performance on appearance-based gaze estimation tasks. However, due to variations in person, illuminations, and background, performance degrades dramatically when applying the model to a new domain. In this paper, we discover an interesting gaze jitter phenomenon in cross-domain gaze estimation, i.e., the gaze predictions of two similar images can be severely deviated in target domain. This is closely related to cross-domain gaze estimation tasks, but surprisingly, it has not been noticed yet previously. Therefore, we innovatively propose to utilize the gaze jitter to analyze and optimize the gaze domain adaptation task. We find that the high-frequency component (HFC) is an important factor that leads to jitter. Based on this discovery, we add high-frequency components to input images using the adversarial attack and employ contrastive learning to encourage the model to obtain similar representations between original and perturbed data, which reduces the impacts of HFC. We evaluate the proposed method on four cross-domain gaze estimation tasks, and experimental results demonstrate that it significantly reduces the gaze jitter and improves the gaze estimation performance in target domains.

摘要: 深度神经网络在基于外表的视线估计任务中表现出了优越的性能。但是，由于人、照明和背景的不同，当将模型应用到新的领域时，性能会显著降低。在本文中，我们发现了跨域凝视估计中一个有趣的凝视抖动现象，即两幅相似图像的凝视预测在目标域可能会严重偏离。这与跨域凝视估计任务密切相关，但令人惊讶的是，之前还没有人注意到这一点。因此，我们创新性地提出利用视线抖动来分析和优化视域自适应任务。我们发现高频分量(HFC)是导致抖动的一个重要因素。基于这一发现，我们在对抗性攻击的输入图像中加入高频分量，并使用对比学习来鼓励模型获得原始数据和扰动数据之间的相似表示，从而减少HFC的影响。我们在四个跨域的视线估计任务上对该方法进行了评估，实验结果表明，该方法显著降低了视线抖动，提高了目标域的视线估计性能。



## **5. Natural Color Fool: Towards Boosting Black-box Unrestricted Attacks**

本色愚人：加强黑盒不受限制的攻击 cs.CV

NeurIPS 2022

**SubmitDate**: 2022-10-05    [abs](http://arxiv.org/abs/2210.02041v1) [paper-pdf](http://arxiv.org/pdf/2210.02041v1)

**Authors**: Shengming Yuan, Qilong Zhang, Lianli Gao, Yaya Cheng, Jingkuan Song

**Abstract**: Unrestricted color attacks, which manipulate semantically meaningful color of an image, have shown their stealthiness and success in fooling both human eyes and deep neural networks. However, current works usually sacrifice the flexibility of the uncontrolled setting to ensure the naturalness of adversarial examples. As a result, the black-box attack performance of these methods is limited. To boost transferability of adversarial examples without damaging image quality, we propose a novel Natural Color Fool (NCF) which is guided by realistic color distributions sampled from a publicly available dataset and optimized by our neighborhood search and initialization reset. By conducting extensive experiments and visualizations, we convincingly demonstrate the effectiveness of our proposed method. Notably, on average, results show that our NCF can outperform state-of-the-art approaches by 15.0%$\sim$32.9% for fooling normally trained models and 10.0%$\sim$25.3% for evading defense methods. Our code is available at https://github.com/ylhz/Natural-Color-Fool.

摘要: 不受限制的颜色攻击，即操纵图像的语义有意义的颜色，已经显示出它们的隐蔽性和成功地愚弄了人眼和深层神经网络。然而，目前的作品往往牺牲了不受控制的场景的灵活性，以确保对抗性例子的自然性。因此，这些方法的黑盒攻击性能是有限的。为了在不损害图像质量的情况下提高对抗性样本的可转移性，我们提出了一种新的自然颜色愚人算法(NCF)，它以从公开数据集中采集的真实颜色分布为指导，并通过邻域搜索和初始化重置进行优化。通过大量的实验和可视化实验，我们令人信服地证明了我们所提出的方法的有效性。值得注意的是，平均而言，我们的NCF在愚弄正常训练的模型方面比最先进的方法高出15.0%$\sim$32.9%，在逃避防御方法方面高出10.0%$\sim$25.3%。我们的代码可以在https://github.com/ylhz/Natural-Color-Fool.上找到



## **6. Towards Adversarially Robust Deepfake Detection: An Ensemble Approach**

面向对抗性强健的深伪检测：集成方法 cs.LG

**SubmitDate**: 2022-10-05    [abs](http://arxiv.org/abs/2202.05687v2) [paper-pdf](http://arxiv.org/pdf/2202.05687v2)

**Authors**: Ashish Hooda, Neal Mangaokar, Ryan Feng, Kassem Fawaz, Somesh Jha, Atul Prakash

**Abstract**: Detecting deepfakes remains an open problem. Current detection methods fail against an adversary who adds imperceptible adversarial perturbations to the deepfake to evade detection. We propose Disjoint Deepfake Detection (D3), a deepfake detector designed to improve adversarial robustness beyond de facto solutions such as adversarial training. D3 uses an ensemble of models over disjoint subsets of the frequency spectrum to significantly improve robustness. Our key insight is to leverage a redundancy in the frequency domain and apply a saliency partitioning technique to disjointly distribute frequency components across multiple models. We formally prove that these disjoint ensembles lead to a reduction in the dimensionality of the input subspace where adversarial deepfakes lie. We then empirically validate the D3 method against white-box attacks and black-box attacks and find that D3 significantly outperforms existing state-of-the-art defenses applied to deepfake detection.

摘要: 检测深度假货仍然是一个悬而未决的问题。当前的检测方法无法对抗将不可察觉的对抗性扰动添加到深度伪码以逃避检测的对手。我们提出了不相交的深伪检测(D3)，这是一种深伪检测器，旨在提高对抗性训练等实际解决方案之外的对抗性健壮性。D3使用频谱的不相交子集上的模型集合来显著提高稳健性。我们的主要见解是利用频域中的冗余，并应用显著分区技术来在多个模型之间分散分布频率分量。我们正式证明了这些不相交的集成导致了对抗性深伪所在的输入子空间的降维。然后，我们通过实验验证了D3方法对白盒攻击和黑盒攻击的抵抗能力，并发现D3方法的性能明显优于现有的应用于深度伪检测的最先进的防御方法。



## **7. Robust Fair Clustering: A Novel Fairness Attack and Defense Framework**

稳健公平聚类：一种新的公平攻防框架 cs.LG

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01953v1) [paper-pdf](http://arxiv.org/pdf/2210.01953v1)

**Authors**: Anshuman Chhabra, Peizhao Li, Prasant Mohapatra, Hongfu Liu

**Abstract**: Clustering algorithms are widely used in many societal resource allocation applications, such as loan approvals and candidate recruitment, among others, and hence, biased or unfair model outputs can adversely impact individuals that rely on these applications. To this end, many fair clustering approaches have been recently proposed to counteract this issue. Due to the potential for significant harm, it is essential to ensure that fair clustering algorithms provide consistently fair outputs even under adversarial influence. However, fair clustering algorithms have not been studied from an adversarial attack perspective. In contrast to previous research, we seek to bridge this gap and conduct a robustness analysis against fair clustering by proposing a novel black-box fairness attack. Through comprehensive experiments, we find that state-of-the-art models are highly susceptible to our attack as it can reduce their fairness performance significantly. Finally, we propose Consensus Fair Clustering (CFC), the first robust fair clustering approach that transforms consensus clustering into a fair graph partitioning problem, and iteratively learns to generate fair cluster outputs. Experimentally, we observe that CFC is highly robust to the proposed attack and is thus a truly robust fair clustering alternative.

摘要: 聚类算法广泛应用于许多社会资源分配应用中，如贷款审批和候选人招聘等，因此，有偏见或不公平的模型输出可能会对依赖这些应用的个人产生不利影响。为此，最近提出了许多公平的聚类方法来解决这个问题。由于潜在的重大危害，确保公平的聚类算法即使在敌意影响下也能提供一致的公平输出是至关重要的。然而，公平分簇算法还没有从对抗攻击的角度进行研究。与以往的研究不同，我们试图弥补这一差距，并通过提出一种新的黑盒公平攻击来进行针对公平聚类的健壮性分析。通过全面的实验，我们发现最新的模型非常容易受到我们的攻击，因为它会显著降低它们的公平性能。最后，我们提出了共识公平聚类(CFC)，这是第一种将共识聚类转化为公平图划分问题的稳健公平聚类方法，并迭代地学习生成公平聚类输出。在实验上，我们观察到cfc对所提出的攻击具有高度的健壮性，因此是一种真正健壮的公平集群替代方案。



## **8. On the Robustness of Deep Clustering Models: Adversarial Attacks and Defenses**

深度聚类模型的稳健性：对抗性攻击与防御 cs.LG

Accepted to the 36th Conference on Neural Information Processing  Systems (NeurIPS 2022)

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01940v1) [paper-pdf](http://arxiv.org/pdf/2210.01940v1)

**Authors**: Anshuman Chhabra, Ashwin Sekhari, Prasant Mohapatra

**Abstract**: Clustering models constitute a class of unsupervised machine learning methods which are used in a number of application pipelines, and play a vital role in modern data science. With recent advancements in deep learning -- deep clustering models have emerged as the current state-of-the-art over traditional clustering approaches, especially for high-dimensional image datasets. While traditional clustering approaches have been analyzed from a robustness perspective, no prior work has investigated adversarial attacks and robustness for deep clustering models in a principled manner. To bridge this gap, we propose a blackbox attack using Generative Adversarial Networks (GANs) where the adversary does not know which deep clustering model is being used, but can query it for outputs. We analyze our attack against multiple state-of-the-art deep clustering models and real-world datasets, and find that it is highly successful. We then employ some natural unsupervised defense approaches, but find that these are unable to mitigate our attack. Finally, we attack Face++, a production-level face clustering API service, and find that we can significantly reduce its performance as well. Through this work, we thus aim to motivate the need for truly robust deep clustering models.

摘要: 聚类模型构成了一类无监督的机器学习方法，广泛应用于多种应用领域，在现代数据科学中发挥着重要的作用。随着深度学习的最新进展--深度聚类模型已经成为当前最先进的聚类方法，特别是对于高维图像数据集。虽然传统的聚类方法已经从稳健性的角度进行了分析，但以前的工作还没有原则性地研究深度聚类模型的对抗性攻击和稳健性。为了弥补这一差距，我们提出了一种使用生成性对抗网络(GANS)的黑盒攻击，其中对手不知道使用了哪种深度聚类模型，但可以查询它的输出。我们分析了我们对多个最先进的深度聚类模型和真实世界数据集的攻击，发现它非常成功。然后，我们使用一些自然的无人监督的防御方法，但发现这些方法无法减轻我们的攻击。最后，我们对生产级别的人脸集群API服务Face++进行了测试，发现我们也可以显著降低其性能。因此，通过这项工作，我们的目标是激发对真正健壮的深度聚类模型的需求。



## **9. Revisiting and Advancing Fast Adversarial Training Through The Lens of Bi-Level Optimization**

用双层优化镜头重温和推进快速对抗性训练 cs.LG

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2112.12376v6) [paper-pdf](http://arxiv.org/pdf/2112.12376v6)

**Authors**: Yihua Zhang, Guanhua Zhang, Prashant Khanduri, Mingyi Hong, Shiyu Chang, Sijia Liu

**Abstract**: Adversarial training (AT) is a widely recognized defense mechanism to gain the robustness of deep neural networks against adversarial attacks. It is built on min-max optimization (MMO), where the minimizer (i.e., defender) seeks a robust model to minimize the worst-case training loss in the presence of adversarial examples crafted by the maximizer (i.e., attacker). However, the conventional MMO method makes AT hard to scale. Thus, Fast-AT (Wong et al., 2020) and other recent algorithms attempt to simplify MMO by replacing its maximization step with the single gradient sign-based attack generation step. Although easy to implement, Fast-AT lacks theoretical guarantees, and its empirical performance is unsatisfactory due to the issue of robust catastrophic overfitting when training with strong adversaries. In this paper, we advance Fast-AT from the fresh perspective of bi-level optimization (BLO). We first show that the commonly-used Fast-AT is equivalent to using a stochastic gradient algorithm to solve a linearized BLO problem involving a sign operation. However, the discrete nature of the sign operation makes it difficult to understand the algorithm performance. Inspired by BLO, we design and analyze a new set of robust training algorithms termed Fast Bi-level AT (Fast-BAT), which effectively defends sign-based projected gradient descent (PGD) attacks without using any gradient sign method or explicit robust regularization. In practice, we show our method yields substantial robustness improvements over baselines across multiple models and datasets. Codes are available at https://github.com/OPTML-Group/Fast-BAT.

摘要: 对抗训练(AT)是一种公认的防御机制，用来增强深度神经网络对对手攻击的健壮性。它建立在最小-最大优化(MMO)的基础上，其中最小化器(即防御者)寻求一个健壮的模型来最小化最坏情况下的训练损失，其中存在由最大化者(即攻击者)制作的对抗性例子。然而，传统的MMO方法使AT难以规模化。因此，Fast-AT(Wong等人，2020)和其他最近的算法试图通过将其最大化步骤替换为基于单一梯度符号的攻击生成步骤来简化MMO。尽管FAST-AT易于实现，但它缺乏理论保证，而且在与强大对手进行训练时，由于存在稳健的灾难性过拟合问题，其经验性能也不尽如人意。本文从双层优化(BLO)的新视角提出了FAST-AT算法。我们首先证明了常用的Fast-AT算法等价于使用随机梯度算法来求解包含符号运算的线性化BLO问题。然而，符号运算的离散性使得很难理解算法的性能。受BLO的启发，我们设计并分析了一组新的稳健训练算法--快速双水平AT(Fast-BAT)，该算法无需使用任何梯度符号方法或显式稳健正则化，即可有效地防御基于符号的投影梯度下降(PGD)攻击。在实践中，我们证明了我们的方法在多个模型和数据集的基线上产生了实质性的稳健性改进。有关代码，请访问https://github.com/OPTML-Group/Fast-BAT.



## **10. Invariant Aggregator for Defending Federated Backdoor Attacks**

用于防御联合后门攻击的不变聚合器 cs.LG

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01834v1) [paper-pdf](http://arxiv.org/pdf/2210.01834v1)

**Authors**: Xiaoyang Wang, Dimitrios Dimitriadis, Sanmi Koyejo, Shruti Tople

**Abstract**: Federated learning is gaining popularity as it enables training of high-utility models across several clients without directly sharing their private data. As a downside, the federated setting makes the model vulnerable to various adversarial attacks in the presence of malicious clients. Specifically, an adversary can perform backdoor attacks to control model predictions via poisoning the training dataset with a trigger. In this work, we propose a mitigation for backdoor attacks in a federated learning setup. Our solution forces the model optimization trajectory to focus on the invariant directions that are generally useful for utility and avoid selecting directions that favor few and possibly malicious clients. Concretely, we consider the sign consistency of the pseudo-gradient (the client update) as an estimation of the invariance. Following this, our approach performs dimension-wise filtering to remove pseudo-gradient elements with low sign consistency. Then, a robust mean estimator eliminates outliers among the remaining dimensions. Our theoretical analysis further shows the necessity of the defense combination and illustrates how our proposed solution defends the federated learning model. Empirical results on three datasets with different modalities and varying number of clients show that our approach mitigates backdoor attacks with a negligible cost on the model utility.

摘要: 联合学习越来越受欢迎，因为它允许在几个客户之间培训高实用模型，而无需直接共享他们的私人数据。缺点是，联合设置使模型在存在恶意客户端的情况下容易受到各种敌意攻击。具体地说，对手可以执行后门攻击，通过使用触发器毒化训练数据集来控制模型预测。在这项工作中，我们提出了一种在联邦学习设置中缓解后门攻击的方法。我们的解决方案迫使模型优化轨迹集中在通常对实用程序有用的不变方向上，避免选择有利于少数甚至可能是恶意客户端的方向。具体地说，我们考虑伪梯度(客户更新)的符号相合性作为不变性的估计。在此之后，我们的方法进行维度过滤，以去除符号一致性较低的伪梯度元素。然后，稳健的均值估计器消除剩余维度中的异常值。我们的理论分析进一步说明了防御组合的必要性，并说明了我们提出的解决方案如何为联邦学习模型辩护。在三个不同模式和不同客户端数量的数据集上的实验结果表明，我们的方法以可以忽略不计的模型效用代价来缓解后门攻击。



## **11. Data Leakage in Tabular Federated Learning**

表格化联合学习中的数据泄漏 cs.LG

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01785v1) [paper-pdf](http://arxiv.org/pdf/2210.01785v1)

**Authors**: Mark Vero, Mislav Balunović, Dimitar I. Dimitrov, Martin Vechev

**Abstract**: While federated learning (FL) promises to preserve privacy in distributed training of deep learning models, recent work in the image and NLP domains showed that training updates leak private data of participating clients. At the same time, most high-stakes applications of FL (e.g., legal and financial) use tabular data. Compared to the NLP and image domains, reconstruction of tabular data poses several unique challenges: (i) categorical features introduce a significantly more difficult mixed discrete-continuous optimization problem, (ii) the mix of categorical and continuous features causes high variance in the final reconstructions, and (iii) structured data makes it difficult for the adversary to judge reconstruction quality. In this work, we tackle these challenges and propose the first comprehensive reconstruction attack on tabular data, called TabLeak. TabLeak is based on three key ingredients: (i) a softmax structural prior, implicitly converting the mixed discrete-continuous optimization problem into an easier fully continuous one, (ii) a way to reduce the variance of our reconstructions through a pooled ensembling scheme exploiting the structure of tabular data, and (iii) an entropy measure which can successfully assess reconstruction quality. Our experimental evaluation demonstrates the effectiveness of TabLeak, reaching a state-of-the-art on four popular tabular datasets. For instance, on the Adult dataset, we improve attack accuracy by 10% compared to the baseline on the practically relevant batch size of 32 and further obtain non-trivial reconstructions for batch sizes as large as 128. Our findings are important as they show that performing FL on tabular data, which often poses high privacy risks, is highly vulnerable.

摘要: 虽然联合学习(FL)承诺在深度学习模型的分布式训练中保护隐私，但最近在图像和NLP领域的研究表明，训练更新泄露了参与的客户的私人数据。与此同时，大多数高风险的FL应用程序(例如，法律和金融应用程序)使用表格数据。与自然语言处理和图像域相比，表格数据的重建提出了几个独特的挑战：(I)分类特征引入了一个明显更困难的离散-连续混合优化问题，(Ii)分类特征和连续特征的混合导致最终重建的高方差，以及(Iii)结构化数据使对手难以判断重建质量。在这项工作中，我们解决了这些挑战，并提出了第一个全面的表格数据重建攻击，称为TabLeak。TabLeak基于三个关键因素：(I)Softmax结构先验，隐式地将离散-连续混合优化问题转换为更容易的完全连续优化问题；(Ii)通过利用表格数据的结构的池化集成方案来减少重建的方差；以及(Iii)可以成功评估重建质量的熵度量。我们的实验评估证明了TabLeak的有效性，在四个流行的表格数据集上达到了最先进的水平。例如，在成人数据集上，与实际相关的批处理大小为32的基线相比，我们将攻击准确率提高了10%，并进一步获得了批处理大小为128的非平凡重建。我们的发现很重要，因为它们表明在表格数据上执行FL是非常容易受到攻击的，这通常会带来很高的隐私风险。



## **12. Adversarial Attack on Attackers: Post-Process to Mitigate Black-Box Score-Based Query Attacks**

对攻击者的对抗性攻击：减轻基于黑盒分数的查询攻击的后处理 cs.LG

accepted by NeurIPS 2022

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2205.12134v2) [paper-pdf](http://arxiv.org/pdf/2205.12134v2)

**Authors**: Sizhe Chen, Zhehao Huang, Qinghua Tao, Yingwen Wu, Cihang Xie, Xiaolin Huang

**Abstract**: The score-based query attacks (SQAs) pose practical threats to deep neural networks by crafting adversarial perturbations within dozens of queries, only using the model's output scores. Nonetheless, we note that if the loss trend of the outputs is slightly perturbed, SQAs could be easily misled and thereby become much less effective. Following this idea, we propose a novel defense, namely Adversarial Attack on Attackers (AAA), to confound SQAs towards incorrect attack directions by slightly modifying the output logits. In this way, (1) SQAs are prevented regardless of the model's worst-case robustness; (2) the original model predictions are hardly changed, i.e., no degradation on clean accuracy; (3) the calibration of confidence scores can be improved simultaneously. Extensive experiments are provided to verify the above advantages. For example, by setting $\ell_\infty=8/255$ on CIFAR-10, our proposed AAA helps WideResNet-28 secure 80.59% accuracy under Square attack (2500 queries), while the best prior defense (i.e., adversarial training) only attains 67.44%. Since AAA attacks SQA's general greedy strategy, such advantages of AAA over 8 defenses can be consistently observed on 8 CIFAR-10/ImageNet models under 6 SQAs, using different attack targets, bounds, norms, losses, and strategies. Moreover, AAA calibrates better without hurting the accuracy. Our code is available at https://github.com/Sizhe-Chen/AAA.

摘要: 基于分数的查询攻击(SQA)通过在数十个查询中精心设计敌意扰动，仅使用模型的输出分数，对深度神经网络构成实际威胁。尽管如此，我们注意到，如果产出的损失趋势受到轻微干扰，质量保证人员很容易受到误导，从而变得不那么有效。根据这一思想，我们提出了一种新的防御方法，即对攻击者的对抗性攻击(AAA)，通过略微修改输出日志来迷惑SQA对错误的攻击方向。这样，(1)无论模型在最坏情况下的稳健性如何，都可以防止SQA；(2)原始模型预测几乎不会改变，即不会降低干净的精度；(3)置信度得分的校准可以同时得到改善。通过大量的实验验证了上述优点。例如，通过在CIFAR-10上设置$\ell_\inty=8/255$，我们提出的AAA在Square攻击(2500个查询)下帮助WideResNet-28确保80.59%的准确率，而最好的先前防御(即对抗性训练)仅达到67.44%。由于AAA攻击SQA的一般贪婪策略，因此在6个SQA下的8个CIFAR-10/ImageNet模型上，使用不同的攻击目标、边界、规范、损失和策略，可以一致地观察到AAA相对于8个防御的优势。此外，AAA在不影响精度的情况下校准得更好。我们的代码可以在https://github.com/Sizhe-Chen/AAA.上找到



## **13. Causal Intervention-based Prompt Debiasing for Event Argument Extraction**

基于因果干预的事件论元提取提示去偏 cs.CL

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01561v1) [paper-pdf](http://arxiv.org/pdf/2210.01561v1)

**Authors**: Jiaju Lin, Jie Zhou, Qin Chen

**Abstract**: Prompt-based methods have become increasingly popular among information extraction tasks, especially in low-data scenarios. By formatting a finetune task into a pre-training objective, prompt-based methods resolve the data scarce problem effectively. However, seldom do previous research investigate the discrepancy among different prompt formulating strategies. In this work, we compare two kinds of prompts, name-based prompt and ontology-base prompt, and reveal how ontology-base prompt methods exceed its counterpart in zero-shot event argument extraction (EAE) . Furthermore, we analyse the potential risk in ontology-base prompts via a causal view and propose a debias method by causal intervention. Experiments on two benchmarks demonstrate that modified by our debias method, the baseline model becomes both more effective and robust, with significant improvement in the resistance to adversarial attacks.

摘要: 基于提示的方法在信息提取任务中变得越来越流行，特别是在低数据场景中。基于提示的方法通过将精调任务格式化为预训练目标，有效地解决了数据稀缺的问题。然而，以往的研究很少考察不同的即时形成策略之间的差异。在这项工作中，我们比较了两种提示，基于名称的提示和基于本体的提示，揭示了基于本体的提示方法在零命中事件参数提取(EAE)方面的优势。此外，我们从因果关系的角度分析了基于本体的提示中的潜在风险，并提出了一种基于因果干预的去偏向方法。在两个基准测试上的实验表明，改进后的基线模型更加有效和健壮，对敌意攻击的抵抗力有了显著的提高。



## **14. Decompiling x86 Deep Neural Network Executables**

反编译x86深度神经网络可执行文件 cs.CR

The extended version of a paper to appear in the Proceedings of the  32nd USENIX Security Symposium, 2023, (USENIX Security '23), 25 pages

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01075v2) [paper-pdf](http://arxiv.org/pdf/2210.01075v2)

**Authors**: Zhibo Liu, Yuanyuan Yuan, Shuai Wang, Xiaofei Xie, Lei Ma

**Abstract**: Due to their widespread use on heterogeneous hardware devices, deep learning (DL) models are compiled into executables by DL compilers to fully leverage low-level hardware primitives. This approach allows DL computations to be undertaken at low cost across a variety of computing platforms, including CPUs, GPUs, and various hardware accelerators.   We present BTD (Bin to DNN), a decompiler for deep neural network (DNN) executables. BTD takes DNN executables and outputs full model specifications, including types of DNN operators, network topology, dimensions, and parameters that are (nearly) identical to those of the input models. BTD delivers a practical framework to process DNN executables compiled by different DL compilers and with full optimizations enabled on x86 platforms. It employs learning-based techniques to infer DNN operators, dynamic analysis to reveal network architectures, and symbolic execution to facilitate inferring dimensions and parameters of DNN operators.   Our evaluation reveals that BTD enables accurate recovery of full specifications of complex DNNs with millions of parameters (e.g., ResNet). The recovered DNN specifications can be re-compiled into a new DNN executable exhibiting identical behavior to the input executable. We show that BTD can boost two representative attacks, adversarial example generation and knowledge stealing, against DNN executables. We also demonstrate cross-architecture legacy code reuse using BTD, and envision BTD being used for other critical downstream tasks like DNN security hardening and patching.

摘要: 由于深度学习模型在不同硬件设备上的广泛使用，深度学习模型被深度学习编译器编译成可执行文件，以充分利用低级硬件原语。这种方法允许在各种计算平台上以低成本进行DL计算，包括CPU、GPU和各种硬件加速器。提出了一种深度神经网络(DNN)可执行程序反编译器BTD(Bin To DNN)。BTD接受DNN可执行文件并输出完整的模型规范，包括DNN操作符的类型、网络拓扑、尺寸和与输入模型(几乎)相同的参数。BTD提供了一个实用的框架来处理由不同的DL编译器编译的DNN可执行文件，并在x86平台上启用了完全优化。它使用基于学习的技术来推断DNN算子，使用动态分析来揭示网络结构，并使用符号执行来方便地推断DNN算子的维度和参数。我们的评估表明，BTD能够准确地恢复具有数百万个参数的复杂DNN的完整规范(例如ResNet)。恢复的DNN规范可以被重新编译成表现出与输入可执行文件相同的行为的新的DNN可执行文件。我们证明了BTD可以增强针对DNN可执行文件的两种有代表性的攻击：敌意实例生成和知识窃取。我们还使用BTD演示了跨体系结构的遗留代码重用，并设想BTD将用于其他关键的下游任务，如DNN安全强化和修补。



## **15. Learning of Dynamical Systems under Adversarial Attacks -- Null Space Property Perspective**

对抗性攻击下动力系统的学习--零空间性质视角 eess.SY

8 pages, 2 figures

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01421v1) [paper-pdf](http://arxiv.org/pdf/2210.01421v1)

**Authors**: Han Feng, Baturalp Yalcin, Javad Lavaei

**Abstract**: We study the identification of a linear time-invariant dynamical system affected by large-and-sparse disturbances modeling adversarial attacks or faults. Under the assumption that the states are measurable, we develop necessary and sufficient conditions for the recovery of the system matrices by solving a constrained lasso-type optimization problem. In addition, we provide an upper bound on the estimation error whenever the disturbance sequence is a combination of small noise values and large adversarial values. Our results depend on the null space property that has been widely used in the lasso literature, and we investigate under what conditions this property holds for linear time-invariant dynamical systems. Lastly, we further study the conditions for a specific probabilistic model and support the results with numerical experiments.

摘要: 研究了受大扰动和稀疏扰动影响的线性时不变动力系统的辨识问题，模拟了敌方攻击或故障。在状态可测的假设下，通过求解一个约束套索型优化问题，给出了系统矩阵恢复的充要条件。此外，当扰动序列是小噪声值和大对抗性值的组合时，我们给出了估计误差的上界。我们的结果依赖于Lasso文献中广泛使用的零空间性质，并且我们研究了线性时不变动力系统在什么条件下该性质成立。最后，我们进一步研究了特定概率模型的条件，并用数值实验对结果进行了支持。



## **16. Physical Passive Patch Adversarial Attacks on Visual Odometry Systems**

对视觉里程计系统的物理被动补丁敌意攻击 cs.CV

Accepted to ACCV 2022

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2207.05729v2) [paper-pdf](http://arxiv.org/pdf/2207.05729v2)

**Authors**: Yaniv Nemcovsky, Matan Jacoby, Alex M. Bronstein, Chaim Baskin

**Abstract**: Deep neural networks are known to be susceptible to adversarial perturbations -- small perturbations that alter the output of the network and exist under strict norm limitations. While such perturbations are usually discussed as tailored to a specific input, a universal perturbation can be constructed to alter the model's output on a set of inputs. Universal perturbations present a more realistic case of adversarial attacks, as awareness of the model's exact input is not required. In addition, the universal attack setting raises the subject of generalization to unseen data, where given a set of inputs, the universal perturbations aim to alter the model's output on out-of-sample data. In this work, we study physical passive patch adversarial attacks on visual odometry-based autonomous navigation systems. A visual odometry system aims to infer the relative camera motion between two corresponding viewpoints, and is frequently used by vision-based autonomous navigation systems to estimate their state. For such navigation systems, a patch adversarial perturbation poses a severe security issue, as it can be used to mislead a system onto some collision course. To the best of our knowledge, we show for the first time that the error margin of a visual odometry model can be significantly increased by deploying patch adversarial attacks in the scene. We provide evaluation on synthetic closed-loop drone navigation data and demonstrate that a comparable vulnerability exists in real data. A reference implementation of the proposed method and the reported experiments is provided at https://github.com/patchadversarialattacks/patchadversarialattacks.

摘要: 众所周知，深度神经网络容易受到对抗性扰动的影响--微小的扰动会改变网络的输出，并在严格的范数限制下存在。虽然这样的扰动通常被讨论为针对特定的输入而量身定做的，但是可以构造一个普遍的扰动来改变模型在一组输入上的输出。普遍摄动提供了一种更现实的对抗性攻击情况，因为不需要知道模型的确切输入。此外，通用攻击设置提出了对不可见数据的泛化主题，在给定一组输入的情况下，通用扰动的目的是改变模型对样本外数据的输出。在这项工作中，我们研究了基于视觉里程计的自主导航系统的物理被动补丁对抗性攻击。视觉里程计系统旨在推断两个相应视点之间的相机相对运动，经常被基于视觉的自主导航系统用来估计它们的状态。对于这样的导航系统，补丁对抗性扰动构成了一个严重的安全问题，因为它可能被用来误导系统进入某些碰撞路线。据我们所知，我们首次证明了在场景中部署补丁对抗性攻击可以显著增加视觉里程计模型的误差。我们对合成的无人机闭环导航数据进行了评估，并证明了真实数据中存在类似的漏洞。在https://github.com/patchadversarialattacks/patchadversarialattacks.上提供了所提出的方法和报告的实验的参考实现



## **17. Exploring Adversarially Robust Training for Unsupervised Domain Adaptation**

无监督领域自适应的对抗性稳健训练探索 cs.CV

Accepted at Asian Conference on Computer Vision (ACCV) 2022

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2202.09300v2) [paper-pdf](http://arxiv.org/pdf/2202.09300v2)

**Authors**: Shao-Yuan Lo, Vishal M. Patel

**Abstract**: Unsupervised Domain Adaptation (UDA) methods aim to transfer knowledge from a labeled source domain to an unlabeled target domain. UDA has been extensively studied in the computer vision literature. Deep networks have been shown to be vulnerable to adversarial attacks. However, very little focus is devoted to improving the adversarial robustness of deep UDA models, causing serious concerns about model reliability. Adversarial Training (AT) has been considered to be the most successful adversarial defense approach. Nevertheless, conventional AT requires ground-truth labels to generate adversarial examples and train models, which limits its effectiveness in the unlabeled target domain. In this paper, we aim to explore AT to robustify UDA models: How to enhance the unlabeled data robustness via AT while learning domain-invariant features for UDA? To answer this question, we provide a systematic study into multiple AT variants that can potentially be applied to UDA. Moreover, we propose a novel Adversarially Robust Training method for UDA accordingly, referred to as ARTUDA. Extensive experiments on multiple adversarial attacks and UDA benchmarks show that ARTUDA consistently improves the adversarial robustness of UDA models. Code is available at https://github.com/shaoyuanlo/ARTUDA

摘要: 无监督领域自适应(UDA)方法旨在将知识从标记的源域转移到非标记的目标域。UDA在计算机视觉文献中得到了广泛的研究。深度网络已被证明容易受到对手的攻击。然而，很少有人致力于提高深度UDA模型的对抗稳健性，这引起了人们对模型可靠性的严重担忧。对抗训练(AT)被认为是最成功的对抗防御方法。然而，传统的AT需要地面事实标签来生成对抗性实例和训练模型，这限制了其在未标记的目标领域的有效性。在本文中，我们旨在探索如何通过AT来增强UDA模型的健壮性，同时学习UDA的域不变性特征？为了回答这个问题，我们对可能应用于UDA的多种AT变体进行了系统研究。此外，我们还针对UDA提出了一种新的对抗性稳健训练方法，称为ARTUDA。在多个对抗性攻击和UDA基准上的大量实验表明，ARTUDA一致地提高了UDA模型的对抗性健壮性。代码可在https://github.com/shaoyuanlo/ARTUDA上找到



## **18. A Study on the Efficiency and Generalization of Light Hybrid Retrievers**

轻型混合取样器的效率与推广研究 cs.IR

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01371v1) [paper-pdf](http://arxiv.org/pdf/2210.01371v1)

**Authors**: Man Luo, Shashank Jain, Anchit Gupta, Arash Einolghozati, Barlas Oguz, Debojeet Chatterjee, Xilun Chen, Chitta Baral, Peyman Heidari

**Abstract**: Existing hybrid retrievers which integrate sparse and dense retrievers, are indexing-heavy, limiting their applicability in real-world on-devices settings. We ask the question "Is it possible to reduce the indexing memory of hybrid retrievers without sacrificing performance?" Driven by this question, we leverage an indexing-efficient dense retriever (i.e. DrBoost) to obtain a light hybrid retriever. Moreover, to further reduce the memory, we introduce a lighter dense retriever (LITE) which is jointly trained on contrastive learning and knowledge distillation from DrBoost. Compared to previous heavy hybrid retrievers, our Hybrid-LITE retriever saves 13 memory while maintaining 98.0 performance.   In addition, we study the generalization of light hybrid retrievers along two dimensions, out-of-domain (OOD) generalization and robustness against adversarial attacks. We evaluate models on two existing OOD benchmarks and create six adversarial attack sets for robustness evaluation. Experiments show that our light hybrid retrievers achieve better robustness performance than both sparse and dense retrievers. Nevertheless there is a large room to improve the robustness of retrievers, and our datasets can aid future research.

摘要: 现有的混合检索器集成了稀疏和密集检索器，索引繁重，限制了它们在现实世界中的设备设置中的适用性。我们提出这样一个问题：“有没有可能在不牺牲性能的情况下减少混合检索器的索引内存？”在这个问题的驱动下，我们利用一种索引效率高的密集检索器(即DrBoost)来获得一种轻型混合检索器。此外，为了进一步减少内存，我们引入了一种更轻的密集检索器(LITE)，它是从DrBoost那里联合训练的对比学习和知识提炼。与以前的重型混合检索器相比，我们的混合精简检索器在保持98.0性能的同时节省了13个内存。此外，我们还研究了轻型混合检索器的二维泛化、域外泛化和对敌方攻击的健壮性。我们在两个现有的OOD基准上对模型进行了评估，并创建了六个对抗性攻击集用于健壮性评估。实验表明，我们的轻型混合检索器比稀疏和密集检索器具有更好的健壮性。然而，检索器的健壮性还有很大的提高空间，我们的数据集可以帮助未来的研究。



## **19. Strength-Adaptive Adversarial Training**

体力适应性对抗性训练 cs.LG

**SubmitDate**: 2022-10-04    [abs](http://arxiv.org/abs/2210.01288v1) [paper-pdf](http://arxiv.org/pdf/2210.01288v1)

**Authors**: Chaojian Yu, Dawei Zhou, Li Shen, Jun Yu, Bo Han, Mingming Gong, Nannan Wang, Tongliang Liu

**Abstract**: Adversarial training (AT) is proved to reliably improve network's robustness against adversarial data. However, current AT with a pre-specified perturbation budget has limitations in learning a robust network. Firstly, applying a pre-specified perturbation budget on networks of various model capacities will yield divergent degree of robustness disparity between natural and robust accuracies, which deviates from robust network's desideratum. Secondly, the attack strength of adversarial training data constrained by the pre-specified perturbation budget fails to upgrade as the growth of network robustness, which leads to robust overfitting and further degrades the adversarial robustness. To overcome these limitations, we propose \emph{Strength-Adaptive Adversarial Training} (SAAT). Specifically, the adversary employs an adversarial loss constraint to generate adversarial training data. Under this constraint, the perturbation budget will be adaptively adjusted according to the training state of adversarial data, which can effectively avoid robust overfitting. Besides, SAAT explicitly constrains the attack strength of training data through the adversarial loss, which manipulates model capacity scheduling during training, and thereby can flexibly control the degree of robustness disparity and adjust the tradeoff between natural accuracy and robustness. Extensive experiments show that our proposal boosts the robustness of adversarial training.

摘要: 对抗训练(AT)被证明能够可靠地提高网络对对抗数据的稳健性。然而，当前具有预先指定的扰动预算的AT在学习健壮网络方面存在局限性。首先，在不同模型容量的网络上施加预先规定的扰动预算，会产生不同程度的稳健性，导致自然精度和稳健精度之间的差距，这偏离了稳健网络的要求。其次，受预先指定的扰动预算约束的对抗性训练数据的攻击强度没有随着网络健壮性的增长而升级，这导致了鲁棒过拟合，从而进一步降低了对抗性健壮性。为了克服这些限制，我们提出了力量自适应对抗训练(SAAT)。具体地说，对手采用对抗性损失约束来生成对抗性训练数据。在此约束下，根据对抗性数据的训练状态自适应调整扰动预算，有效地避免了稳健过拟合。此外，SAAT通过对抗性损失来明确约束训练数据的攻击强度，从而在训练过程中操纵模型容量调度，从而可以灵活地控制健壮性差异的程度，并在自然精度和健壮性之间进行权衡。大量实验表明，该算法增强了对抗性训练的健壮性。



## **20. On Attacking Out-Domain Uncertainty Estimation in Deep Neural Networks**

关于深度神经网络攻击域外不确定性估计问题 cs.LG

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2210.02191v1) [paper-pdf](http://arxiv.org/pdf/2210.02191v1)

**Authors**: Huimin Zeng, Zhenrui Yue, Yang Zhang, Ziyi Kou, Lanyu Shang, Dong Wang

**Abstract**: In many applications with real-world consequences, it is crucial to develop reliable uncertainty estimation for the predictions made by the AI decision systems. Targeting at the goal of estimating uncertainty, various deep neural network (DNN) based uncertainty estimation algorithms have been proposed. However, the robustness of the uncertainty returned by these algorithms has not been systematically explored. In this work, to raise the awareness of the research community on robust uncertainty estimation, we show that state-of-the-art uncertainty estimation algorithms could fail catastrophically under our proposed adversarial attack despite their impressive performance on uncertainty estimation. In particular, we aim at attacking the out-domain uncertainty estimation: under our attack, the uncertainty model would be fooled to make high-confident predictions for the out-domain data, which they originally would have rejected. Extensive experimental results on various benchmark image datasets show that the uncertainty estimated by state-of-the-art methods could be easily corrupted by our attack.

摘要: 在许多具有真实世界后果的应用中，为人工智能决策系统做出的预测开发可靠的不确定性估计是至关重要的。针对不确定性估计的目标，人们提出了各种基于深度神经网络(DNN)的不确定性估计算法。然而，这些算法返回的不确定性的稳健性还没有得到系统的探讨。在这项工作中，为了提高研究界对稳健不确定性估计的认识，我们证明了最新的不确定性估计算法在我们提出的对抗性攻击下可能会灾难性地失败，尽管它们在不确定性估计方面的表现令人印象深刻。特别是，我们的目标是攻击域外的不确定性估计：在我们的攻击下，不确定性模型将被愚弄，以对域外数据做出高度自信的预测，而他们最初会拒绝这些预测。在不同基准图像数据集上的大量实验结果表明，最新方法估计的不确定性很容易被我们的攻击所破坏。



## **21. Adversarially Robust One-class Novelty Detection**

对抗性稳健的一类新颖性检测 cs.CV

Accepted in IEEE Transactions on Pattern Analysis and Machine  Intelligence (T-PAMI), 2022

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2108.11168v2) [paper-pdf](http://arxiv.org/pdf/2108.11168v2)

**Authors**: Shao-Yuan Lo, Poojan Oza, Vishal M. Patel

**Abstract**: One-class novelty detectors are trained with examples of a particular class and are tasked with identifying whether a query example belongs to the same known class. Most recent advances adopt a deep auto-encoder style architecture to compute novelty scores for detecting novel class data. Deep networks have shown to be vulnerable to adversarial attacks, yet little focus is devoted to studying the adversarial robustness of deep novelty detectors. In this paper, we first show that existing novelty detectors are susceptible to adversarial examples. We further demonstrate that commonly-used defense approaches for classification tasks have limited effectiveness in one-class novelty detection. Hence, we need a defense specifically designed for novelty detection. To this end, we propose a defense strategy that manipulates the latent space of novelty detectors to improve the robustness against adversarial examples. The proposed method, referred to as Principal Latent Space (PrincipaLS), learns the incrementally-trained cascade principal components in the latent space to robustify novelty detectors. PrincipaLS can purify latent space against adversarial examples and constrain latent space to exclusively model the known class distribution. We conduct extensive experiments on eight attacks, five datasets and seven novelty detectors, showing that PrincipaLS consistently enhances the adversarial robustness of novelty detection models. Code is available at https://github.com/shaoyuanlo/PrincipaLS

摘要: 单类新颖性检测器用特定类的示例进行训练，任务是识别查询示例是否属于相同的已知类。最新的进展采用了深度自动编码器风格的体系结构来计算新颖性分数，以检测新的类别数据。深度网络已被证明容易受到敌意攻击，但很少有人致力于研究深度新奇检测器的敌意稳健性。在这篇文章中，我们首先证明了现有的新颖性检测器容易受到敌意例子的影响。我们进一步证明了分类任务中常用的防御方法在单类新颖性检测中的有效性有限。因此，我们需要专门为新奇检测而设计的防御措施。为此，我们提出了一种利用新颖性检测器的潜在空间来提高对敌意实例的鲁棒性的防御策略。该方法通过学习潜在空间中的递增训练的级联主成分，使新奇检测器具有较强的鲁棒性。委托人可以针对敌意例子净化潜在空间，并约束潜在空间以唯一地模拟已知的类别分布。我们在8个攻击、5个数据集和7个新颖性检测器上进行了广泛的实验，结果表明，主体一致地增强了新颖性检测模型的对抗健壮性。代码可在https://github.com/shaoyuanlo/PrincipaLS上找到



## **22. AutoJoin: Efficient Adversarial Training for Robust Maneuvering via Denoising Autoencoder and Joint Learning**

AutoJoin：基于去噪自动编码器和联合学习的有效对抗机动训练 cs.LG

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2205.10933v2) [paper-pdf](http://arxiv.org/pdf/2205.10933v2)

**Authors**: Michael Villarreal, Bibek Poudel, Ryan Wickman, Yu Shen, Weizi Li

**Abstract**: As a result of increasingly adopted machine learning algorithms and ubiquitous sensors, many 'perception-to-control' systems are developed and deployed. For these systems to be trustworthy, we need to improve their robustness with adversarial training being one approach. We propose a gradient-free adversarial training technique, called AutoJoin, which is a very simple yet effective and efficient approach to produce robust models for imaged-based maneuvering. Compared to other SOTA methods with testing on over 5M perturbed and clean images, AutoJoin achieves significant performance increases up to the 40% range under gradient-free perturbations while improving on clean performance up to 300%. Regarding efficiency, AutoJoin demonstrates strong advantages over other SOTA techniques by saving up to 83% time per training epoch and 90% training data. Although not the focus of AutoJoin, it even demonstrates superb ability in defending gradient-based attacks. The core idea of AutoJoin is to use a decoder attachment to the original regression model creating a denoising autoencoder within the architecture. This architecture allows the tasks 'maneuvering' and 'denoising sensor input' to be jointly learnt and reinforce each other's performance.

摘要: 由于越来越多地采用机器学习算法和无处不在的传感器，许多“感知到控制”系统被开发和部署。为了让这些系统值得信赖，我们需要提高它们的健壮性，对抗性训练是一种方法。我们提出了一种无梯度的对抗性训练技术，称为AutoJoin，这是一种非常简单但有效的方法，可以为基于图像的机动建立稳健的模型。与其他SOTA方法相比，AutoJoin在测试超过5M扰动和干净图像时，在无梯度扰动下实现了高达40%的显著性能提升，同时清洁性能提高了300%。在效率方面，与其他SOTA技术相比，AutoJoin显示出强大的优势，每个训练周期节省高达83%的时间，训练数据节省90%。虽然不是AutoJoin的重点，但它甚至在防御基于梯度的攻击方面展示了高超的能力。AutoJoin的核心思想是使用附加到原始回归模型的解码器，在体系结构中创建去噪自动编码器。这种体系结构允许任务的“机动”和“去噪传感器输入”共同学习，并加强彼此的表现。



## **23. Leveraging Local Patch Differences in Multi-Object Scenes for Generative Adversarial Attacks**

利用多目标场景中局部斑块差异进行生成性对抗性攻击 cs.CV

Accepted at WACV 2023 (Round 1), camera-ready version

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2209.09883v2) [paper-pdf](http://arxiv.org/pdf/2209.09883v2)

**Authors**: Abhishek Aich, Shasha Li, Chengyu Song, M. Salman Asif, Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury

**Abstract**: State-of-the-art generative model-based attacks against image classifiers overwhelmingly focus on single-object (i.e., single dominant object) images. Different from such settings, we tackle a more practical problem of generating adversarial perturbations using multi-object (i.e., multiple dominant objects) images as they are representative of most real-world scenes. Our goal is to design an attack strategy that can learn from such natural scenes by leveraging the local patch differences that occur inherently in such images (e.g. difference between the local patch on the object `person' and the object `bike' in a traffic scene). Our key idea is to misclassify an adversarial multi-object image by confusing the victim classifier for each local patch in the image. Based on this, we propose a novel generative attack (called Local Patch Difference or LPD-Attack) where a novel contrastive loss function uses the aforesaid local differences in feature space of multi-object scenes to optimize the perturbation generator. Through various experiments across diverse victim convolutional neural networks, we show that our approach outperforms baseline generative attacks with highly transferable perturbations when evaluated under different white-box and black-box settings.

摘要: 最新的基于产生式模型的针对图像分类器的攻击绝大多数集中在单一对象(即单一优势对象)图像上。与这样的设置不同，我们解决了一个更实际的问题，即使用多对象(即，多个主导对象)图像来生成对抗性扰动，因为它们代表了大多数真实世界的场景。我们的目标是设计一种攻击策略，通过利用这类图像中固有的局部斑块差异(例如，交通场景中对象‘人’和对象‘自行车’上的局部斑块之间的差异)来学习此类自然场景。我们的主要思想是通过对图像中每个局部块的受害者分类器进行混淆来对对抗性多目标图像进行错误分类。在此基础上，我们提出了一种新的生成性攻击(称为局部补丁差异或LPD-攻击)，其中一种新的对比损失函数利用多目标场景特征空间中的上述局部差异来优化扰动生成器。通过对不同受害者卷积神经网络的实验，我们表明，在不同的白盒和黑盒设置下，我们的方法优于具有高度可转移性扰动的基线生成性攻击。



## **24. Bridging the Performance Gap between FGSM and PGD Adversarial Training**

缩小FGSM和PGD对抗性训练之间的成绩差距 cs.CR

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2011.05157v2) [paper-pdf](http://arxiv.org/pdf/2011.05157v2)

**Authors**: Tianjin Huang, Vlado Menkovski, Yulong Pei, Mykola Pechenizkiy

**Abstract**: Deep learning achieves state-of-the-art performance in many tasks but exposes to the underlying vulnerability against adversarial examples. Across existing defense techniques, adversarial training with the projected gradient decent attack (adv.PGD) is considered as one of the most effective ways to achieve moderate adversarial robustness. However, adv.PGD requires too much training time since the projected gradient attack (PGD) takes multiple iterations to generate perturbations. On the other hand, adversarial training with the fast gradient sign method (adv.FGSM) takes much less training time since the fast gradient sign method (FGSM) takes one step to generate perturbations but fails to increase adversarial robustness. In this work, we extend adv.FGSM to make it achieve the adversarial robustness of adv.PGD. We demonstrate that the large curvature along FGSM perturbed direction leads to a large difference in performance of adversarial robustness between adv.FGSM and adv.PGD, and therefore propose combining adv.FGSM with a curvature regularization (adv.FGSMR) in order to bridge the performance gap between adv.FGSM and adv.PGD. The experiments show that adv.FGSMR has higher training efficiency than adv.PGD. In addition, it achieves comparable performance of adversarial robustness on MNIST dataset under white-box attack, and it achieves better performance than adv.PGD under white-box attack and effectively defends the transferable adversarial attack on CIFAR-10 dataset.

摘要: 深度学习在许多任务中实现了最先进的性能，但在对抗对手示例时暴露了潜在的脆弱性。在现有的防御技术中，采用投影梯度下降攻击(Adv.PGD)的对抗训练被认为是实现适度对抗稳健性的最有效方法之一。然而，由于投影梯度攻击(PGD)需要多次迭代才能产生扰动，因此Adv.PGD需要太多的训练时间。另一方面，由于快速梯度符号方法(FGSM)只需一步就能产生扰动，但不能增加对手的稳健性，因此快速梯度符号方法(Adv.FGSM)训练所需的训练时间要少得多。在本文中，我们对Adv.FGSM进行了扩展，使其达到了Adv.PGD的对抗健壮性。我们证明了沿FGSM扰动方向的大曲率导致了Adv.FGSM和Adv.PGD在对抗健壮性上的巨大差异，从而提出了将Adv.FGSM与曲率正则化相结合(Adv.FGSMR)，以弥补Adv.FGSM和Adv.PGD之间的性能差距。实验表明，Adv.FGSMR具有比Adv.PGD更高的训练效率。此外，在白盒攻击下，它在MNIST数据集上的对抗健壮性也达到了相当的性能。在白盒攻击下取得了比Adv.PGD更好的性能，有效地防御了CIFAR-10数据集上可转移的敌意攻击。



## **25. MultiGuard: Provably Robust Multi-label Classification against Adversarial Examples**

MultiGuard：针对敌意示例的可证明健壮的多标签分类 cs.CR

Accepted by NeurIPS 2022

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2210.01111v1) [paper-pdf](http://arxiv.org/pdf/2210.01111v1)

**Authors**: Jinyuan Jia, Wenjie Qu, Neil Zhenqiang Gong

**Abstract**: Multi-label classification, which predicts a set of labels for an input, has many applications. However, multiple recent studies showed that multi-label classification is vulnerable to adversarial examples. In particular, an attacker can manipulate the labels predicted by a multi-label classifier for an input via adding carefully crafted, human-imperceptible perturbation to it. Existing provable defenses for multi-class classification achieve sub-optimal provable robustness guarantees when generalized to multi-label classification. In this work, we propose MultiGuard, the first provably robust defense against adversarial examples to multi-label classification. Our MultiGuard leverages randomized smoothing, which is the state-of-the-art technique to build provably robust classifiers. Specifically, given an arbitrary multi-label classifier, our MultiGuard builds a smoothed multi-label classifier via adding random noise to the input. We consider isotropic Gaussian noise in this work. Our major theoretical contribution is that we show a certain number of ground truth labels of an input are provably in the set of labels predicted by our MultiGuard when the $\ell_2$-norm of the adversarial perturbation added to the input is bounded. Moreover, we design an algorithm to compute our provable robustness guarantees. Empirically, we evaluate our MultiGuard on VOC 2007, MS-COCO, and NUS-WIDE benchmark datasets. Our code is available at: \url{https://github.com/quwenjie/MultiGuard}

摘要: 多标签分类预测输入的一组标签，有很多应用。然而，最近的多项研究表明，多标签分类容易受到对抗性例子的影响。具体地说，攻击者可以通过添加精心制作的、人类无法察觉的扰动来操纵多标签分类器预测的输入标签。当推广到多标签分类时，现有的多类分类的可证明防御实现了次优的可证明稳健性保证。在这项工作中，我们提出了MultiGuard，这是对多标签分类的第一个可证明的对敌意示例的健壮防御。我们的MultiGuard利用随机平滑，这是构建可证明稳健的分类器的最先进技术。具体地说，在给定任意多标签分类器的情况下，我们的MultiGuard通过向输入添加随机噪声来构建平滑的多标签分类器。我们在这项工作中考虑了各向同性高斯噪声。我们的主要理论贡献是，当输入的对抗性扰动的$\ell_2$-范数有界时，我们证明了在我们的MultiGuard预测的标签集中，输入的一定数量的基本真理标签是可证明的。此外，我们还设计了一个算法来计算我们的可证明稳健性保证。经验性地，我们在VOC 2007、MS-COCO和NUS范围的基准数据集上评估我们的MultiGuard。我们的代码位于：\url{https://github.com/quwenjie/MultiGuard}



## **26. ASGNN: Graph Neural Networks with Adaptive Structure**

ASGNN：具有自适应结构的图神经网络 cs.LG

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2210.01002v1) [paper-pdf](http://arxiv.org/pdf/2210.01002v1)

**Authors**: Zepeng Zhang, Songtao Lu, Zengfeng Huang, Ziping Zhao

**Abstract**: The graph neural network (GNN) models have presented impressive achievements in numerous machine learning tasks. However, many existing GNN models are shown to be vulnerable to adversarial attacks, which creates a stringent need to build robust GNN architectures. In this work, we propose a novel interpretable message passing scheme with adaptive structure (ASMP) to defend against adversarial attacks on graph structure. Layers in ASMP are derived based on optimization steps that minimize an objective function that learns the node feature and the graph structure simultaneously. ASMP is adaptive in the sense that the message passing process in different layers is able to be carried out over dynamically adjusted graphs. Such property allows more fine-grained handling of the noisy (or perturbed) graph structure and hence improves the robustness. Convergence properties of the ASMP scheme are theoretically established. Integrating ASMP with neural networks can lead to a new family of GNN models with adaptive structure (ASGNN). Extensive experiments on semi-supervised node classification tasks demonstrate that the proposed ASGNN outperforms the state-of-the-art GNN architectures in terms of classification performance under various adversarial attacks.

摘要: 图神经网络(GNN)模型在众多的机器学习任务中取得了令人瞩目的成就。然而，许多现有的GNN模型被证明容易受到敌意攻击，这就产生了对构建健壮的GNN体系结构的严格需求。在这项工作中，我们提出了一种新的具有自适应结构的可解释消息传递方案(ASMP)来防御对图结构的敌意攻击。ASMP中的层是基于最小化同时学习节点特征和图结构的目标函数的优化步骤得到的。ASMP是自适应的，因为不同层中的消息传递过程能够在动态调整的图上执行。这种特性允许对噪声(或扰动)图结构进行更细粒度的处理，从而提高了稳健性。从理论上证明了ASMP格式的收敛性质。将ASMP与神经网络相结合，可以得到一类新的自适应结构GNN模型(ASGNN)。在半监督节点分类任务上的大量实验表明，所提出的ASGNN在各种对抗性攻击下的分类性能优于现有的GNN结构。



## **27. Improving Robustness of Deep Reinforcement Learning Agents: Environment Attack based on the Critic Network**

提高深度强化学习代理的健壮性：基于批评性网络的环境攻击 cs.LG

8 pages, 8 figures

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2104.03154v3) [paper-pdf](http://arxiv.org/pdf/2104.03154v3)

**Authors**: Lucas Schott, Hatem Hajri, Sylvain Lamprier

**Abstract**: To improve policy robustness of deep reinforcement learning agents, a line of recent works focus on producing disturbances of the environment. Existing approaches of the literature to generate meaningful disturbances of the environment are adversarial reinforcement learning methods. These methods set the problem as a two-player game between the protagonist agent, which learns to perform a task in an environment, and the adversary agent, which learns to disturb the protagonist via modifications of the considered environment. Both protagonist and adversary are trained with deep reinforcement learning algorithms. Alternatively, we propose in this paper to build on gradient-based adversarial attacks, usually used for classification tasks for instance, that we apply on the critic network of the protagonist to identify efficient disturbances of the environment. Rather than learning an attacker policy, which usually reveals as very complex and unstable, we leverage the knowledge of the critic network of the protagonist, to dynamically complexify the task at each step of the learning process. We show that our method, while being faster and lighter, leads to significantly better improvements in policy robustness than existing methods of the literature.

摘要: 为了提高深度强化学习代理的策略稳健性，最近的一系列工作集中在产生环境扰动上。现有文献中产生有意义的环境扰动的方法是对抗性强化学习方法。这些方法将问题设置为主角代理和对手代理之间的两人博弈，前者学习在环境中执行任务，后者学习通过修改所考虑的环境来干扰主角。使用深度强化学习算法对主角和对手进行训练。或者，我们在本文中建议基于梯度的对抗性攻击，通常用于分类任务，例如，我们应用于主人公的批评网络来识别环境的有效干扰。我们不是学习通常表现为非常复杂和不稳定的攻击者策略，而是利用主角的批评网络的知识，在学习过程的每个步骤动态地使任务复杂化。我们表明，虽然我们的方法更快、更轻，但与现有的文献方法相比，我们的方法在政策稳健性方面的改进要明显更好。



## **28. Push-Pull: Characterizing the Adversarial Robustness for Audio-Visual Active Speaker Detection**

Push-Pull：表征视听主动说话人检测的对抗稳健性 cs.SD

Accepted by SLT 2022

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2210.00753v1) [paper-pdf](http://arxiv.org/pdf/2210.00753v1)

**Authors**: Xuanjun Chen, Haibin Wu, Helen Meng, Hung-yi Lee, Jyh-Shing Roger Jang

**Abstract**: Audio-visual active speaker detection (AVASD) is well-developed, and now is an indispensable front-end for several multi-modal applications. However, to the best of our knowledge, the adversarial robustness of AVASD models hasn't been investigated, not to mention the effective defense against such attacks. In this paper, we are the first to reveal the vulnerability of AVASD models under audio-only, visual-only, and audio-visual adversarial attacks through extensive experiments. What's more, we also propose a novel audio-visual interaction loss (AVIL) for making attackers difficult to find feasible adversarial examples under an allocated attack budget. The loss aims at pushing the inter-class embeddings to be dispersed, namely non-speech and speech clusters, sufficiently disentangled, and pulling the intra-class embeddings as close as possible to keep them compact. Experimental results show the AVIL outperforms the adversarial training by 33.14 mAP (%) under multi-modal attacks.

摘要: 视听有源说话人检测(AVASD)技术已经发展成熟，是多种多模式应用中不可缺少的前端。然而，就我们所知，AVASD模型的攻击健壮性还没有被研究过，更不用说对此类攻击的有效防御了。本文通过大量的实验，首次揭示了AVASD模型在纯音频、纯视觉和视听攻击下的脆弱性。此外，我们还提出了一种新的视听交互损失(AVIL)，使攻击者在分配的攻击预算下很难找到可行的对抗性例子。丢失的目的是将类间嵌入进行分散，即非语音和语音簇充分解缠，并尽可能地拉近类内嵌入以保持紧凑。实验结果表明，在多模式攻击下，AVIL的性能比对抗性训练高出33.14 MAP(%)。



## **29. On the Robustness of Safe Reinforcement Learning under Observational Perturbations**

安全强化学习在观测摄动下的稳健性 cs.LG

30 pages, 4 figures, 8 tables

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2205.14691v2) [paper-pdf](http://arxiv.org/pdf/2205.14691v2)

**Authors**: Zuxin Liu, Zijian Guo, Zhepeng Cen, Huan Zhang, Jie Tan, Bo Li, Ding Zhao

**Abstract**: Safe reinforcement learning (RL) trains a policy to maximize the task reward while satisfying safety constraints. While prior works focus on the performance optimality, we find that the optimal solutions of many safe RL problems are not robust and safe against carefully designed observational perturbations. We formally analyze the unique properties of designing effective state adversarial attackers in the safe RL setting. We show that baseline adversarial attack techniques for standard RL tasks are not always effective for safe RL and proposed two new approaches - one maximizes the cost and the other maximizes the reward. One interesting and counter-intuitive finding is that the maximum reward attack is strong, as it can both induce unsafe behaviors and make the attack stealthy by maintaining the reward. We further propose a more effective adversarial training framework for safe RL and evaluate it via comprehensive experiments. This paper provides a pioneer work to investigate the safety and robustness of RL under observational attacks for future safe RL studies.

摘要: 安全强化学习(RL)训练一种策略，在满足安全约束的同时最大化任务奖励。虽然以前的工作主要集中在性能最优性上，但我们发现许多安全RL问题的最优解对于精心设计的观测扰动并不是健壮的和安全的。我们形式化地分析了在安全RL环境下设计有效的状态对抗攻击者的独特性质。我们证明了标准RL任务的基线对抗性攻击技术对于安全RL并不总是有效的，并提出了两种新的方法-一种最大化成本，另一种最大化回报。一个有趣和违反直觉的发现是，最大奖励攻击是强大的，因为它既可以诱导不安全的行为，又可以通过保持奖励来使攻击隐形。我们进一步提出了一种更有效的安全RL对抗训练框架，并通过综合实验对其进行了评估。本文为进一步研究RL在观察性攻击下的安全性和健壮性提供了开拓性的工作。



## **30. Improving Model Robustness with Latent Distribution Locally and Globally**

利用局部和全局隐含分布提高模型稳健性 cs.LG

**SubmitDate**: 2022-10-03    [abs](http://arxiv.org/abs/2107.04401v2) [paper-pdf](http://arxiv.org/pdf/2107.04401v2)

**Authors**: Zhuang Qian, Shufei Zhang, Kaizhu Huang, Qiufeng Wang, Rui Zhang, Xinping Yi

**Abstract**: In this work, we consider model robustness of deep neural networks against adversarial attacks from a global manifold perspective. Leveraging both the local and global latent information, we propose a novel adversarial training method through robust optimization, and a tractable way to generate Latent Manifold Adversarial Examples (LMAEs) via an adversarial game between a discriminator and a classifier. The proposed adversarial training with latent distribution (ATLD) method defends against adversarial attacks by crafting LMAEs with the latent manifold in an unsupervised manner. ATLD preserves the local and global information of latent manifold and promises improved robustness against adversarial attacks. To verify the effectiveness of our proposed method, we conduct extensive experiments over different datasets (e.g., CIFAR-10, CIFAR-100, SVHN) with different adversarial attacks (e.g., PGD, CW), and show that our method substantially outperforms the state-of-the-art (e.g., Feature Scattering) in adversarial robustness by a large accuracy margin. The source codes are available at https://github.com/LitterQ/ATLD-pytorch.

摘要: 在这项工作中，我们从全局流形的角度考虑了深层神经网络对敌意攻击的模型稳健性。利用局部和全局潜在信息，我们提出了一种新的稳健优化的对抗性训练方法，并提出了一种通过鉴别器和分类器之间的对抗性博弈来生成潜在流形对抗性实例(LMAE)的简便方法。利用潜在分布的对抗性训练(ATLD)方法通过在无监督的情况下利用潜在流形构造LMAE来防御对抗性攻击。ATLD保留了潜在流形的局部和全局信息，并保证提高对对手攻击的稳健性。为了验证我们提出的方法的有效性，我们在不同的数据集(如CIFAR-10、CIFAR-100、SVHN)上进行了大量的实验，结果表明我们的方法在对抗攻击(如PGD、CW)的稳健性方面比最新的(如特征分散)方法有很大的准确率优势。源代码可在https://github.com/LitterQ/ATLD-pytorch.上找到



## **31. Automated Security Analysis of Exposure Notification Systems**

曝光通知系统的自动化安全分析 cs.CR

23 pages, Full version of the corresponding USENIX Security '23 paper

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2210.00649v1) [paper-pdf](http://arxiv.org/pdf/2210.00649v1)

**Authors**: Kevin Morio, Ilkan Esiyok, Dennis Jackson, Robert Künnemann

**Abstract**: We present the first formal analysis and comparison of the security of the two most widely deployed exposure notification systems, ROBERT and the Google and Apple Exposure Notification (GAEN) framework. ROBERT is the most popular instalment of the centralised approach to exposure notification, in which the risk score is computed by a central server. GAEN, in contrast, follows the decentralised approach, where the user's phone calculates the risk. The relative merits of centralised and decentralised systems have proven to be a controversial question. The majority of the previous analyses have focused on the privacy implications of these systems, ours is the first formal analysis to evaluate the security of the deployed systems -- the absence of false risk alerts. We model the French deployment of ROBERT and the most widely deployed GAEN variant, Germany's Corona-Warn-App. We isolate the precise conditions under which these systems prevent false alerts. We determine exactly how an adversary can subvert the system via network and Bluetooth sniffing, database leakage or the compromise of phones, back-end systems and health authorities. We also investigate the security of the original specification of the DP3T protocol, in order to identify gaps between the proposed scheme and its ultimate deployment. We find a total of 27 attack patterns, including many that distinguish the centralised from the decentralised approach, as well as attacks on the authorisation procedure that differentiate all three protocols. Our results suggest that ROBERT's centralised design is more vulnerable against both opportunistic and highly resourced attackers trying to perform mass-notification attacks.

摘要: 我们首次正式分析和比较了两个部署最广泛的曝光通知系统--Robert和Google和Apple曝光通知(GAEN)框架--的安全性。Robert是暴露通知的集中化方法中最受欢迎的一部分，在这种方法中，风险分数由中央服务器计算。相比之下，GAEN采用的是去中心化的方法，即用户的手机计算风险。事实证明，集中式和分散式系统的相对优点是一个有争议的问题。以前的大多数分析都集中在这些系统对隐私的影响上，我们是第一个评估已部署系统的安全性的正式分析--没有错误的风险警报。我们模拟了法国部署的Robert和部署最广泛的Gaen变体，德国的Corona-Warn-App。我们隔离了这些系统防止错误警报的准确条件。我们准确地确定对手如何通过网络和蓝牙嗅探、数据库泄露或手机、后端系统和卫生当局的危害来颠覆系统。我们还研究了DP3T协议原始规范的安全性，以确定所提出的方案与其最终部署之间的差距。我们发现了总共27种攻击模式，包括许多区分集中式和分散式方法的攻击模式，以及区分所有三种协议的对授权过程的攻击。我们的结果表明，Robert的集中式设计更容易受到试图执行大规模通知攻击的机会主义和资源丰富的攻击者的攻击。



## **32. Spectral Augmentation for Self-Supervised Learning on Graphs**

图的自监督学习的谱增强算法 cs.LG

26 pages, 5 figures, 12 tables

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2210.00643v1) [paper-pdf](http://arxiv.org/pdf/2210.00643v1)

**Authors**: Lu Lin, Jinghui Chen, Hongning Wang

**Abstract**: Graph contrastive learning (GCL), as an emerging self-supervised learning technique on graphs, aims to learn representations via instance discrimination. Its performance heavily relies on graph augmentation to reflect invariant patterns that are robust to small perturbations; yet it still remains unclear about what graph invariance GCL should capture. Recent studies mainly perform topology augmentations in a uniformly random manner in the spatial domain, ignoring its influence on the intrinsic structural properties embedded in the spectral domain. In this work, we aim to find a principled way for topology augmentations by exploring the invariance of graphs from the spectral perspective. We develop spectral augmentation which guides topology augmentations by maximizing the spectral change. Extensive experiments on both graph and node classification tasks demonstrate the effectiveness of our method in self-supervised representation learning. The proposed method also brings promising generalization capability in transfer learning, and is equipped with intriguing robustness property under adversarial attacks. Our study sheds light on a general principle for graph topology augmentation.

摘要: 图对比学习(GCL)是一种新兴的关于图的自监督学习技术，旨在通过实例区分来学习表示。它的性能在很大程度上依赖于图的增强来反映对小扰动具有健壮性的不变模式；然而，GCL应该捕获什么图不变性仍然是不清楚的。目前的研究主要是在空间域以均匀随机的方式进行拓扑增强，而忽略了其对谱域固有结构性质的影响。在这项工作中，我们的目的是通过从谱的角度探索图的不变性来寻找一种原则性的拓扑增强方法。我们开发了光谱增强，它通过最大化光谱变化来指导拓扑增强。在图和节点分类任务上的大量实验证明了该方法在自监督表示学习中的有效性。该方法在转移学习中具有良好的泛化能力，并且在对抗攻击下具有良好的鲁棒性。我们的研究揭示了图拓扑增强的一般原理。



## **33. Graph Structural Attack by Perturbing Spectral Distance**

基于摄动谱距离的图结构攻击 cs.LG

Proceedings of the 28th ACM SIGKDD international conference on  knowledge discovery & data mining (KDD'22)

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2111.00684v3) [paper-pdf](http://arxiv.org/pdf/2111.00684v3)

**Authors**: Lu Lin, Ethan Blaser, Hongning Wang

**Abstract**: Graph Convolutional Networks (GCNs) have fueled a surge of research interest due to their encouraging performance on graph learning tasks, but they are also shown vulnerability to adversarial attacks. In this paper, an effective graph structural attack is investigated to disrupt graph spectral filters in the Fourier domain, which are the theoretical foundation of GCNs. We define the notion of spectral distance based on the eigenvalues of graph Laplacian to measure the disruption of spectral filters. We realize the attack by maximizing the spectral distance and propose an efficient approximation to reduce the time complexity brought by eigen-decomposition. The experiments demonstrate the remarkable effectiveness of the proposed attack in both black-box and white-box settings for both test-time evasion attacks and training-time poisoning attacks. Our qualitative analysis suggests the connection between the imposed spectral changes in the Fourier domain and the attack behavior in the spatial domain, which provides empirical evidence that maximizing spectral distance is an effective way to change the graph structural property and thus disturb the frequency components for graph filters to affect the learning of GCNs.

摘要: 图卷积网络(GCNS)因其在图学习任务中令人鼓舞的性能而引起了人们的研究兴趣，但它们也显示出对对手攻击的脆弱性。本文研究了一种有效的图结构攻击来破坏傅里叶域图谱滤波，这是GCNS的理论基础。基于拉普拉斯图的特征值，我们定义了谱距离的概念来度量谱滤波器的破坏程度。我们通过最大化谱距离来实现攻击，并提出了一种有效的近似来降低特征分解带来的时间复杂度。实验表明，该攻击在黑盒和白盒环境下对测试时间逃避攻击和训练时间中毒攻击都具有显著的效果。我们的定性分析表明，傅里叶域上的频谱变化与空域上的攻击行为之间存在联系，这为最大化频谱距离是改变图的结构属性从而干扰图的频率分量从而影响GCNS学习的一种有效方法提供了经验证据。



## **34. GANTouch: An Attack-Resilient Framework for Touch-based Continuous Authentication System**

GANTouch：一种基于触摸式连续认证系统的抗攻击框架 cs.CR

11 pages, 7 figures, 2 tables, 3 algorithms, in IEEE TBIOM 2022

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2210.01594v1) [paper-pdf](http://arxiv.org/pdf/2210.01594v1)

**Authors**: Mohit Agrawal, Pragyan Mehrotra, Rajesh Kumar, Rajiv Ratn Shah

**Abstract**: Previous studies have shown that commonly studied (vanilla) implementations of touch-based continuous authentication systems (V-TCAS) are susceptible to active adversarial attempts. This study presents a novel Generative Adversarial Network assisted TCAS (G-TCAS) framework and compares it to the V-TCAS under three active adversarial environments viz. Zero-effort, Population, and Random-vector. The Zero-effort environment was implemented in two variations viz. Zero-effort (same-dataset) and Zero-effort (cross-dataset). The first involved a Zero-effort attack from the same dataset, while the second used three different datasets. G-TCAS showed more resilience than V-TCAS under the Population and Random-vector, the more damaging adversarial scenarios than the Zero-effort. On average, the increase in the false accept rates (FARs) for V-TCAS was much higher (27.5% and 21.5%) than for G-TCAS (14% and 12.5%) for Population and Random-vector attacks, respectively. Moreover, we performed a fairness analysis of TCAS for different genders and found TCAS to be fair across genders. The findings suggest that we should evaluate TCAS under active adversarial environments and affirm the usefulness of GANs in the TCAS pipeline.

摘要: 以前的研究表明，通常研究的基于触摸的连续认证系统(V-TCAS)的(普通)实现容易受到主动对抗尝试的影响。提出了一种新的生成性对抗性网络辅助TCAS(G-TCAS)框架，并在三种主动对抗性环境下与V-TCAS进行了比较。零努力、种群和随机向量。Zero-Effort环境在两个变体中实现，即。零努力(相同数据集)和零努力(交叉数据集)。第一个涉及来自同一数据集的零努力攻击，而第二个使用三个不同的数据集。G-TCAS在种群和随机向量下表现出比V-TCAS更强的韧性，在比零努力更具破坏性的对抗性场景下表现出更好的韧性。平均而言，对于种群攻击和随机向量攻击，V-TCAS的错误接受率(FAR)增幅(27.5%和21.5%)远远高于G-TCAS(14%和12.5%)。此外，我们对不同性别的TCAS进行了公平性分析，发现TCAS在性别上是公平的。研究结果表明，我们应该在积极的对抗性环境下评估TCAS，并肯定GANS在TCAS管道中的有用性。



## **35. Optimization for Robustness Evaluation beyond $\ell_p$ Metrics**

超越$\ell_p$度量的健壮性评估优化 cs.LG

5 pages, 1 figure, 3 tables, submitted to the 2023 IEEE International  Conference on Acoustics, Speech, and Signal Processing (ICASSP 2023) and the  14th International OPT Workshop on Optimization for Machine Learning

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2210.00621v1) [paper-pdf](http://arxiv.org/pdf/2210.00621v1)

**Authors**: Hengyue Liang, Buyun Liang, Ying Cui, Tim Mitchell, Ju Sun

**Abstract**: Empirical evaluation of deep learning models against adversarial attacks entails solving nontrivial constrained optimization problems. Popular algorithms for solving these constrained problems rely on projected gradient descent (PGD) and require careful tuning of multiple hyperparameters. Moreover, PGD can only handle $\ell_1$, $\ell_2$, and $\ell_\infty$ attack models due to the use of analytical projectors. In this paper, we introduce a novel algorithmic framework that blends a general-purpose constrained-optimization solver PyGRANSO, With Constraint-Folding (PWCF), to add reliability and generality to robustness evaluation. PWCF 1) finds good-quality solutions without the need of delicate hyperparameter tuning, and 2) can handle general attack models, e.g., general $\ell_p$ ($p \geq 0$) and perceptual attacks, which are inaccessible to PGD-based algorithms.

摘要: 针对敌意攻击的深度学习模型的经验评估需要解决非平凡的约束优化问题。解决这些约束问题的流行算法依赖于投影梯度下降(PGD)，并且需要仔细调整多个超参数。此外，由于分析投影仪的使用，PGD只能处理$\ell_1$、$\ell_2$和$\ell_\inty$攻击模型。在本文中，我们提出了一个新的算法框架，将通用的约束优化求解器PyGRANSO与约束折叠(PWCF)相结合，以增加健壮性评估的可靠性和通用性。PWCF1)在不需要调整超参数的情况下找到高质量的解，2)可以处理一般攻击模型，例如一般的$\ell_p$($p\geq 0$)和感知攻击，这是基于PGD的算法无法访问的。



## **36. iCTGAN--An Attack Mitigation Technique for Random-vector Attack on Accelerometer-based Gait Authentication Systems**

ICTGAN--一种针对加速度计步态认证系统的随机向量攻击缓解技术 cs.CR

9 pages, 5 figures, IEEE International Joint Conference on Biometrics  (IJCB 2022)

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2210.00615v1) [paper-pdf](http://arxiv.org/pdf/2210.00615v1)

**Authors**: Jun Hyung Mo, Rajesh Kumar

**Abstract**: A recent study showed that commonly (vanilla) studied implementations of accelerometer-based gait authentication systems ($v$ABGait) are susceptible to random-vector attack. The same study proposed a beta noise-assisted implementation ($\beta$ABGait) to mitigate the attack. In this paper, we assess the effectiveness of the random-vector attack on both $v$ABGait and $\beta$ABGait using three accelerometer-based gait datasets. In addition, we propose $i$ABGait, an alternative implementation of ABGait, which uses a Conditional Tabular Generative Adversarial Network. Then we evaluate $i$ABGait's resilience against the traditional zero-effort and random-vector attacks. The results show that $i$ABGait mitigates the impact of the random-vector attack to a reasonable extent and outperforms $\beta$ABGait in most experimental settings.

摘要: 最近的一项研究表明，通常研究的基于加速度计的步态验证系统($v$ABGait)的实现容易受到随机向量攻击。同一项研究提出了一种Beta噪声辅助实现($\beta$ABGait)来缓解攻击。在本文中，我们使用三个基于加速度计的步态数据集来评估随机向量攻击在$v$ABGait和$\beta$ABGait上的有效性。此外，我们还提出了$I$ABGait，这是ABGait的一种替代实现，它使用了条件表格生成性对抗性网络。然后，我们评估了$I$ABGait对传统的零努力攻击和随机向量攻击的弹性。结果表明，$I$ABGait在一定程度上缓解了随机向量攻击的影响，并且在大多数实验设置中表现优于$\beta$ABGait。



## **37. FoveaTer: Foveated Transformer for Image Classification**

FoveaTer：用于图像分类的凹槽转换器 cs.CV

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2105.14173v3) [paper-pdf](http://arxiv.org/pdf/2105.14173v3)

**Authors**: Aditya Jonnalagadda, William Yang Wang, B. S. Manjunath, Miguel P. Eckstein

**Abstract**: Many animals and humans process the visual field with a varying spatial resolution (foveated vision) and use peripheral processing to make eye movements and point the fovea to acquire high-resolution information about objects of interest. This architecture results in computationally efficient rapid scene exploration. Recent progress in self-attention-based Vision Transformers, an alternative to the traditionally convolution-reliant computer vision systems. However, the Transformer models do not explicitly model the foveated properties of the visual system nor the interaction between eye movements and the classification task. We propose Foveated Transformer (FoveaTer) model, which uses pooling regions and eye movements to perform object classification tasks using a Vision Transformer architecture. Using square pooling regions or biologically-inspired radial-polar pooling regions, our proposed model pools the image features from the convolution backbone and uses the pooled features as an input to transformer layers. It decides on subsequent fixation location based on the attention assigned by the Transformer to various locations from past and present fixations. It dynamically allocates more fixation/computational resources to more challenging images before making the final image category decision. Using five ablation studies, we evaluate the contribution of different components of the Foveated model. We perform a psychophysics scene categorization task and use the experimental data to find a suitable radial-polar pooling region combination. We also show that the Foveated model better explains the human decisions in a scene categorization task than a Baseline model. We demonstrate our model's robustness against PGD adversarial attacks with both types of pooling regions, where we see the Foveated model outperform the Baseline model.

摘要: 许多动物和人类以不同的空间分辨率处理视野(中心凹视觉)，并使用外围处理来进行眼睛运动和指向中心凹以获取有关感兴趣对象的高分辨率信息。这种架构带来了计算效率高的快速场景探索。基于自我注意的视觉转换器的最新进展，它是传统依赖卷积的计算机视觉系统的替代方案。然而，变形金刚模型没有明确地模拟视觉系统的凹陷属性，也没有明确地模拟眼球运动和分类任务之间的相互作用。我们提出了Foveated Transformer(FoveaTer)模型，该模型使用融合区域和眼球运动来执行使用视觉转换器架构的对象分类任务。我们的模型使用正方形拼接区域或生物启发的径向-极地拼接区域，将卷积主干中的图像特征汇集在一起，并将汇集的特征作为变压器层的输入。它根据变形金刚分配给过去和现在注视的不同位置的注意力来决定后续的注视位置。它在做出最终的图像类别决定之前，动态地将更多的注视/计算资源分配给更具挑战性的图像。使用五项消融研究，我们评估了Foveated模型中不同组件的贡献。我们执行了心理物理场景分类任务，并使用实验数据来找到合适的径向-极地池区域组合。我们还表明，Foveated模型比基线模型更好地解释了场景分类任务中的人类决策。我们证明了我们的模型对这两种类型的池化区域的PGD攻击的健壮性，其中我们看到Foveated模型的性能优于基线模型。



## **38. Availability Attacks Against Neural Network Certifiers Based on Backdoors**

基于后门的神经网络认证可用性攻击 cs.LG

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2108.11299v4) [paper-pdf](http://arxiv.org/pdf/2108.11299v4)

**Authors**: Tobias Lorenz, Marta Kwiatkowska, Mario Fritz

**Abstract**: To achieve reliable, robust, and safe AI systems it is important to implement fallback strategies when AI predictions cannot be trusted. Certifiers for neural networks are a reliable way to check the robustness of these predictions. They guarantee for some predictions that a certain class of manipulations or attacks could not have changed the outcome. For the remaining predictions without guarantees, the method abstains from making a prediction and a fallback strategy needs to be invoked, which typically incurs additional costs, can require a human operator, or even fail to provide any prediction. While this is a key concept towards safe and secure AI, we show for the first time that this approach comes with its own security risks, as such fallback strategies can be deliberately triggered by an adversary. Using training-time attacks, the adversary can significantly reduce the certified robustness of the model, making it unavailable. This transfers the main system load onto the fallback, reducing the overall system's integrity and availability. We design two novel backdoor attacks which show the practical relevance of these threats. For example, adding 1% poisoned data during training is sufficient to reduce certified robustness by up to 95 percentage points. Our extensive experiments across multiple datasets, model architectures, and certifiers demonstrate the wide applicability of these attacks. A first investigation into potential defenses shows that current approaches are insufficient to mitigate the issue, highlighting the need for new, more specific solutions.

摘要: 要实现可靠、健壮和安全的人工智能系统，重要的是在人工智能预测无法信任的情况下实施后备策略。神经网络的认证器是检验这些预测的稳健性的可靠方法。他们为某些预测提供了保证，即某种操纵或攻击不可能改变结果。对于没有保证的其余预测，该方法放弃进行预测，并且需要调用后备策略，这通常会产生额外的成本，可能需要人工操作员，甚至不能提供任何预测。虽然这是一个安全可靠的人工智能的关键概念，但我们第一次表明，这种方法也有自己的安全风险，因为这样的后备战略可能会被对手故意触发。利用训练时间攻击，敌手可以显著降低模型经过认证的健壮性，使其不可用。这会将主要系统负载转移到备用系统上，从而降低整个系统的完整性和可用性。我们设计了两个新的后门攻击，表明了这些威胁的实际相关性。例如，在训练期间添加1%的有毒数据就足以将认证的健壮性降低高达95个百分点。我们在多个数据集、模型体系结构和认证器上的广泛实验证明了这些攻击的广泛适用性。对潜在防御措施的首次调查显示，目前的方法不足以缓解这一问题，这突显了需要新的、更具体的解决方案。



## **39. Adversarial Speaker Distillation for Countermeasure Model on Automatic Speaker Verification**

说话人自动确认对抗模型中的对抗性说话人提取 cs.SD

Accepted by ISCA SPSC 2022

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2203.17031v6) [paper-pdf](http://arxiv.org/pdf/2203.17031v6)

**Authors**: Yen-Lun Liao, Xuanjun Chen, Chung-Che Wang, Jyh-Shing Roger Jang

**Abstract**: The countermeasure (CM) model is developed to protect ASV systems from spoof attacks and prevent resulting personal information leakage in Automatic Speaker Verification (ASV) system. Based on practicality and security considerations, the CM model is usually deployed on edge devices, which have more limited computing resources and storage space than cloud-based systems, confining the model size under a limitation. To better trade off the CM model sizes and performance, we proposed an adversarial speaker distillation method, which is an improved version of knowledge distillation method combined with generalized end-to-end (GE2E) pre-training and adversarial fine-tuning. In the evaluation phase of the ASVspoof 2021 Logical Access task, our proposed adversarial speaker distillation ResNetSE (ASD-ResNetSE) model reaches 0.2695 min t-DCF and 3.54% EER. ASD-ResNetSE only used 22.5% of parameters and 19.4% of multiply and accumulate operands of ResNetSE model.

摘要: 在自动说话人确认(ASV)系统中，为了保护ASV系统免受欺骗攻击，并防止由此导致的个人信息泄露，提出了对策(CM)模型。基于实用性和安全性考虑，CM模型通常部署在边缘设备上，与基于云的系统相比，边缘设备具有更有限的计算资源和存储空间，从而将模型大小限制在一定范围内。为了更好地权衡CM模型的规模和性能，我们提出了一种对抗性说话人蒸馏方法，它是一种改进的知识蒸馏方法，结合了广义端到端(GE2E)预训练和对抗性微调。在ASVspoof2021逻辑访问任务的评估阶段，我们提出的对抗性说话人蒸馏ResNetSE(ASD-ResNetSE)模型达到了0.2695分钟的t-DCF和3.54%的EER。ASD-ResNetSE只使用了ResNetSE模型22.5%的参数和19.4%的乘法和累加操作数。



## **40. Adaptive Smoothness-weighted Adversarial Training for Multiple Perturbations with Its Stability Analysis**

多扰动下的自适应光滑度加权对抗训练及其稳定性分析 cs.LG

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2210.00557v1) [paper-pdf](http://arxiv.org/pdf/2210.00557v1)

**Authors**: Jiancong Xiao, Zeyu Qin, Yanbo Fan, Baoyuan Wu, Jue Wang, Zhi-Quan Luo

**Abstract**: Adversarial Training (AT) has been demonstrated as one of the most effective methods against adversarial examples. While most existing works focus on AT with a single type of perturbation e.g., the $\ell_\infty$ attacks), DNNs are facing threats from different types of adversarial examples. Therefore, adversarial training for multiple perturbations (ATMP) is proposed to generalize the adversarial robustness over different perturbation types (in $\ell_1$, $\ell_2$, and $\ell_\infty$ norm-bounded perturbations). However, the resulting model exhibits trade-off between different attacks. Meanwhile, there is no theoretical analysis of ATMP, limiting its further development. In this paper, we first provide the smoothness analysis of ATMP and show that $\ell_1$, $\ell_2$, and $\ell_\infty$ adversaries give different contributions to the smoothness of the loss function of ATMP. Based on this, we develop the stability-based excess risk bounds and propose adaptive smoothness-weighted adversarial training for multiple perturbations. Theoretically, our algorithm yields better bounds. Empirically, our experiments on CIFAR10 and CIFAR100 achieve the state-of-the-art performance against the mixture of multiple perturbations attacks.

摘要: 对抗性训练(AT)已被证明是对抗对抗性例子最有效的方法之一。虽然现有的大多数工作都集中在具有单一类型扰动的AT上，例如$\ell_\inty$攻击)，但是DNN面临着来自不同类型的对抗性例子的威胁。因此，针对多重扰动的对抗训练(ATMP)被提出来推广在不同扰动类型($\ell_1$、$\ell_2$和$\ell_\inty$范数有界扰动)下的对抗稳健性。然而，所得到的模型显示了不同攻击之间的权衡。同时，目前还没有对ATMP的理论分析，限制了其进一步的发展。本文首先给出了ATMP的光滑性分析，并证明了对手对ATMP损失函数光滑性的贡献是不同的。在此基础上，提出了基于稳定性的超额风险界，并提出了针对多扰动的自适应平滑加权对抗性训练。从理论上讲，我们的算法产生了更好的界。经验上，我们在CIFAR10和CIFAR100上的实验达到了对抗多种扰动攻击混合的最先进的性能。



## **41. Understanding Adversarial Robustness Against On-manifold Adversarial Examples**

理解对抗流形上的对抗例子的对抗稳健性 cs.LG

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2210.00430v1) [paper-pdf](http://arxiv.org/pdf/2210.00430v1)

**Authors**: Jiancong Xiao, Liusha Yang, Yanbo Fan, Jue Wang, Zhi-Quan Luo

**Abstract**: Deep neural networks (DNNs) are shown to be vulnerable to adversarial examples. A well-trained model can be easily attacked by adding small perturbations to the original data. One of the hypotheses of the existence of the adversarial examples is the off-manifold assumption: adversarial examples lie off the data manifold. However, recent research showed that on-manifold adversarial examples also exist. In this paper, we revisit the off-manifold assumption and want to study a question: at what level is the poor performance of neural networks against adversarial attacks due to on-manifold adversarial examples? Since the true data manifold is unknown in practice, we consider two approximated on-manifold adversarial examples on both real and synthesis datasets. On real datasets, we show that on-manifold adversarial examples have greater attack rates than off-manifold adversarial examples on both standard-trained and adversarially-trained models. On synthetic datasets, theoretically, We prove that on-manifold adversarial examples are powerful, yet adversarial training focuses on off-manifold directions and ignores the on-manifold adversarial examples. Furthermore, we provide analysis to show that the properties derived theoretically can also be observed in practice. Our analysis suggests that on-manifold adversarial examples are important, and we should pay more attention to on-manifold adversarial examples for training robust models.

摘要: 深度神经网络(DNN)被证明容易受到敌意例子的影响。一个训练有素的模型很容易受到攻击，方法是在原始数据中添加一些小的扰动。对抗性例子存在的假设之一是非流形假设：对抗性例子存在于数据流形之外。然而，最近的研究表明，也存在多种多样的对抗性例子。在这篇文章中，我们重新审视了非流形假设，并想要研究一个问题：由于流形上的对抗性例子，神经网络抵抗对抗性攻击的性能在什么水平上？由于真实的数据流形在实践中是未知的，我们考虑了两个在真实和合成数据集上的近似上流形上的对抗性例子。在真实数据集上，我们表明，无论是在标准训练的模型上还是在对抗性训练的模型上，流形上的对抗性例子都比非流形的对抗性例子具有更高的攻击率。在合成数据集上，理论上，我们证明了流形上的对抗性例子是强大的，但对抗性训练集中在非流形方向上，而忽略了流形上的对抗性例子。此外，我们还进行了分析，表明理论推导的性质也可以在实践中观察到。我们的分析表明，流形上的对抗性例子是重要的，我们应该更多地关注流形上的对抗性例子来训练稳健的模型。



## **42. Voice Spoofing Countermeasures: Taxonomy, State-of-the-art, experimental analysis of generalizability, open challenges, and the way forward**

语音欺骗对策：分类、最新技术、可概括性实验分析、开放挑战和前进方向 eess.AS

**SubmitDate**: 2022-10-02    [abs](http://arxiv.org/abs/2210.00417v1) [paper-pdf](http://arxiv.org/pdf/2210.00417v1)

**Authors**: Awais Khan, Khalid Mahmood Malik, James Ryan, Mikul Saravanan

**Abstract**: Malicious actors may seek to use different voice-spoofing attacks to fool ASV systems and even use them for spreading misinformation. Various countermeasures have been proposed to detect these spoofing attacks. Due to the extensive work done on spoofing detection in automated speaker verification (ASV) systems in the last 6-7 years, there is a need to classify the research and perform qualitative and quantitative comparisons on state-of-the-art countermeasures. Additionally, no existing survey paper has reviewed integrated solutions to voice spoofing evaluation and speaker verification, adversarial/antiforensics attacks on spoofing countermeasures, and ASV itself, or unified solutions to detect multiple attacks using a single model. Further, no work has been done to provide an apples-to-apples comparison of published countermeasures in order to assess their generalizability by evaluating them across corpora. In this work, we conduct a review of the literature on spoofing detection using hand-crafted features, deep learning, end-to-end, and universal spoofing countermeasure solutions to detect speech synthesis (SS), voice conversion (VC), and replay attacks. Additionally, we also review integrated solutions to voice spoofing evaluation and speaker verification, adversarial and anti-forensics attacks on voice countermeasures, and ASV. The limitations and challenges of the existing spoofing countermeasures are also presented. We report the performance of these countermeasures on several datasets and evaluate them across corpora. For the experiments, we employ the ASVspoof2019 and VSDC datasets along with GMM, SVM, CNN, and CNN-GRU classifiers. (For reproduceability of the results, the code of the test bed can be found in our GitHub Repository.

摘要: 恶意攻击者可能会试图使用不同的语音欺骗攻击来欺骗ASV系统，甚至利用它们来传播错误信息。已经提出了各种对策来检测这些欺骗攻击。由于过去6-7年在自动说话人验证(ASV)系统中对欺骗检测所做的大量工作，有必要对这些研究进行分类，并对最新的对策进行定性和定量的比较。此外，现有的调查论文没有审查语音欺骗评估和说话人验证的集成解决方案、对欺骗对策的对抗性/反取证攻击，以及ASV本身，或者使用单一模型检测多个攻击的统一解决方案。此外，还没有做任何工作来对已发表的对策进行逐一比较，以便通过在语料库中对其进行评估来评估其普遍性。在这项工作中，我们对使用手工特征、深度学习、端到端和通用欺骗对策解决方案来检测语音合成(SS)、语音转换(VC)和重放攻击的欺骗检测的文献进行了回顾。此外，我们还回顾了语音欺骗评估和说话人验证、针对语音对策的对抗性和反取证攻击以及ASV的集成解决方案。文中还指出了现有欺骗对策的局限性和挑战。我们报告了这些对策在几个数据集上的性能，并在语料库中对它们进行了评估。在实验中，我们使用了ASVspoof2019和VSDC数据集，以及GMM、SVM、CNN和CNN-GRU分类器。(对于结果的重现性，可以在我们的GitHub存储库中找到试验台的代码。



## **43. Adversarial Attacks on Transformers-Based Malware Detectors**

对基于Transformers的恶意软件检测器的敌意攻击 cs.CR

**SubmitDate**: 2022-10-01    [abs](http://arxiv.org/abs/2210.00008v1) [paper-pdf](http://arxiv.org/pdf/2210.00008v1)

**Authors**: Yash Jakhotiya, Heramb Patil, Jugal Rawlani

**Abstract**: Signature-based malware detectors have proven to be insufficient as even a small change in malignant executable code can bypass these signature-based detectors. Many machine learning-based models have been proposed to efficiently detect a wide variety of malware. Many of these models are found to be susceptible to adversarial attacks - attacks that work by generating intentionally designed inputs that can force these models to misclassify. Our work aims to explore vulnerabilities in the current state of the art malware detectors to adversarial attacks. We train a Transformers-based malware detector, carry out adversarial attacks resulting in a misclassification rate of 23.9% and propose defenses that reduce this misclassification rate to half. An implementation of our work can be found at https://github.com/yashjakhotiya/Adversarial-Attacks-On-Transformers.

摘要: 事实证明，基于签名的恶意软件检测器是不够的，因为即使对恶意可执行代码进行很小的更改也可以绕过这些基于签名的检测器。已经提出了许多基于机器学习的模型来有效地检测各种恶意软件。其中许多模型被发现容易受到对抗性攻击-通过生成故意设计的输入来工作的攻击，可以迫使这些模型错误分类。我们的工作旨在探索当前最先进的恶意软件检测器中的漏洞，以进行对抗性攻击。我们训练了一个基于Transformers的恶意软件检测器，执行了导致23.9%错误分类率的对抗性攻击，并提出了将错误分类率降低到一半的防御措施。我们工作的实现可以在https://github.com/yashjakhotiya/Adversarial-Attacks-On-Transformers.上找到



## **44. Counter-Adversarial Learning with Inverse Unscented Kalman Filter**

基于逆无迹卡尔曼滤波的对抗学习 math.OC

Conference paper, 10 pages, 1 figure. Proofs are provided at the end  only in the arXiv version

**SubmitDate**: 2022-10-01    [abs](http://arxiv.org/abs/2210.00359v1) [paper-pdf](http://arxiv.org/pdf/2210.00359v1)

**Authors**: Himali Singh, Kumar Vijay Mishra, Arpan Chattopadhyay

**Abstract**: In order to infer the strategy of an intelligent attacker, it is desired for the defender to cognitively sense the attacker's state. In this context, we aim to learn the information that an adversary has gathered about us from a Bayesian perspective. Prior works employ linear Gaussian state-space models and solve this inverse cognition problem through the design of inverse stochastic filters. In practice, these counter-adversarial settings are highly nonlinear systems. We address this by formulating the inverse cognition as a nonlinear Gaussian state-space model, wherein the adversary employs an unscented Kalman filter (UKF) to estimate our state with reduced linearization errors. To estimate the adversary's estimate of us, we propose and develop an inverse UKF (IUKF), wherein the system model is known to both the adversary and the defender. We also derive the conditions for the stochastic stability of IUKF in the mean-squared boundedness sense. Numerical experiments for multiple practical system models show that the estimation error of IUKF converges and closely follows the recursive Cram\'{e}r-Rao lower bound.

摘要: 为了推断智能攻击者的策略，防御者需要从认知上感知攻击者的状态。在这种情况下，我们的目标是从贝叶斯的角度了解对手收集的关于我们的信息。以往的工作采用线性高斯状态空间模型，并通过设计逆随机滤波器来解决这一逆认知问题。在实践中，这些对抗设置是高度非线性的系统。我们通过将反向认知描述为非线性高斯状态空间模型来解决这一问题，其中对手使用无味卡尔曼滤波(UKF)来估计我们的状态，并减少线性化误差。为了估计对手对我们的估计，我们提出并发展了一种逆UKF(IUKF)，其中对手和防御者都知道系统模型。我们还得到了IUKF在均方有界性意义下随机稳定的条件。对多个实际系统模型的数值实验表明，IUKF的估计误差收敛并紧跟递推的Cram‘{e}r-Rao下界。



## **45. GAT: Generative Adversarial Training for Adversarial Example Detection and Robust Classification**

GAT：用于对抗性实例检测和稳健分类的生成性对抗性训练 cs.LG

ICLR 2020, code is available at  https://github.com/xuwangyin/GAT-Generative-Adversarial-Training; v4 fixed  error in Figure 2

**SubmitDate**: 2022-10-01    [abs](http://arxiv.org/abs/1905.11475v4) [paper-pdf](http://arxiv.org/pdf/1905.11475v4)

**Authors**: Xuwang Yin, Soheil Kolouri, Gustavo K. Rohde

**Abstract**: The vulnerabilities of deep neural networks against adversarial examples have become a significant concern for deploying these models in sensitive domains. Devising a definitive defense against such attacks is proven to be challenging, and the methods relying on detecting adversarial samples are only valid when the attacker is oblivious to the detection mechanism. In this paper we propose a principled adversarial example detection method that can withstand norm-constrained white-box attacks. Inspired by one-versus-the-rest classification, in a K class classification problem, we train K binary classifiers where the i-th binary classifier is used to distinguish between clean data of class i and adversarially perturbed samples of other classes. At test time, we first use a trained classifier to get the predicted label (say k) of the input, and then use the k-th binary classifier to determine whether the input is a clean sample (of class k) or an adversarially perturbed example (of other classes). We further devise a generative approach to detecting/classifying adversarial examples by interpreting each binary classifier as an unnormalized density model of the class-conditional data. We provide comprehensive evaluation of the above adversarial example detection/classification methods, and demonstrate their competitive performances and compelling properties.

摘要: 深层神经网络对敌意例子的脆弱性已经成为在敏感领域部署这些模型的一个重要问题。事实证明，针对此类攻击设计明确的防御是具有挑战性的，依赖于检测对手样本的方法只有在攻击者忘记检测机制时才有效。本文提出了一种能够抵抗范数约束白盒攻击的原则性对抗性实例检测方法。受一对一分类的启发，在一个K类分类问题中，我们训练了K个二进制分类器，其中第i个二进制分类器用于区分第I类的干净数据和其他类的相反扰动样本。在测试时，我们首先使用训练好的分类器来获得输入的预测标签(比如k)，然后使用第k个二进制分类器来确定输入是(k类的)干净样本还是(其他类的)相反的扰动样本。我们进一步设计了一种生成性方法，通过将每个二进制分类器解释为类条件数据的非归一化密度模型来检测/分类敌意示例。我们对上述恶意范例检测/分类方法进行了综合评价，并展示了它们的竞争性能和令人信服的性质。



## **46. DeltaBound Attack: Efficient decision-based attack in low queries regime**

DeltaBound攻击：低查询条件下基于决策的高效攻击 cs.LG

**SubmitDate**: 2022-10-01    [abs](http://arxiv.org/abs/2210.00292v1) [paper-pdf](http://arxiv.org/pdf/2210.00292v1)

**Authors**: Lorenzo Rossi

**Abstract**: Deep neural networks and other machine learning systems, despite being extremely powerful and able to make predictions with high accuracy, are vulnerable to adversarial attacks. We proposed the DeltaBound attack: a novel, powerful attack in the hard-label setting with $\ell_2$ norm bounded perturbations. In this scenario, the attacker has only access to the top-1 predicted label of the model and can be therefore applied to real-world settings such as remote API. This is a complex problem since the attacker has very little information about the model. Consequently, most of the other techniques present in the literature require a massive amount of queries for attacking a single example. Oppositely, this work mainly focuses on the evaluation of attack's power in the low queries regime $\leq 1000$ queries) with $\ell_2$ norm in the hard-label settings. We find that the DeltaBound attack performs as well and sometimes better than current state-of-the-art attacks while remaining competitive across different kinds of models. Moreover, we evaluate our method against not only deep neural networks, but also non-deep learning models, such as Gradient Boosting Decision Trees and Multinomial Naive Bayes.

摘要: 尽管深度神经网络和其他机器学习系统非常强大，能够做出高精度的预测，但它们很容易受到对手的攻击。我们提出了DeltaBound攻击：在具有$\ell_2$范数有界扰动的硬标签环境下的一种新颖的、强大的攻击。在这种情况下，攻击者只能访问模型的前1个预测标签，因此可以应用于实际设置，如远程API。这是一个复杂的问题，因为攻击者几乎没有关于该模型的信息。因此，文献中存在的大多数其他技术需要大量的查询来攻击单个示例。相反，本文主要针对硬标签环境下的低查询率($\leq 1000$查询)和$\ell_2$范数下的攻击强度进行了评估。我们发现，DeltaBound攻击的性能与当前最先进的攻击一样好，有时甚至更好，同时在不同类型的模型中保持竞争力。此外，我们不仅对深度神经网络，而且对非深度学习模型，如梯度提升决策树和多项朴素贝叶斯模型进行了评估。



## **47. On the tightness of linear relaxation based robustness certification methods**

基于线性松弛的稳健性证明方法的紧性 cs.LG

**SubmitDate**: 2022-10-01    [abs](http://arxiv.org/abs/2210.00178v1) [paper-pdf](http://arxiv.org/pdf/2210.00178v1)

**Authors**: Cheng Tang

**Abstract**: There has been a rapid development and interest in adversarial training and defenses in the machine learning community in the recent years. One line of research focuses on improving the performance and efficiency of adversarial robustness certificates for neural networks \cite{gowal:19, wong_zico:18, raghunathan:18, WengTowardsFC:18, wong:scalable:18, singh:convex_barrier:19, Huang_etal:19, single-neuron-relax:20, Zhang2020TowardsSA}. While each providing a certification to lower (or upper) bound the true distortion under adversarial attacks via relaxation, less studied was the tightness of relaxation. In this paper, we analyze a family of linear outer approximation based certificate methods via a meta algorithm, IBP-Lin. The aforementioned works often lack quantitative analysis to answer questions such as how does the performance of the certificate method depend on the network configuration and the choice of approximation parameters. Under our framework, we make a first attempt at answering these questions, which reveals that the tightness of linear approximation based certification can depend heavily on the configuration of the trained networks.

摘要: 近年来，机器学习界对对抗性训练和防御有了迅速的发展和兴趣。其中一项研究集中于提高神经网络对抗健壮性证书的性能和效率。{gowal：19，Wong_zico：18，Raghunathan：18，WengTowardsFC：18，Wong：Scalable：18，Singh：凸障：19，Huang_Etal：19，单神经元松弛：20，Zhang 2020TowardsSA}。虽然每一个都提供了一个证明，通过放松来降低(或上限)在对抗性攻击下的真实失真，但对放松的紧密性的研究较少。本文通过一个元算法IBP-LIN分析了一类基于线性外逼近的证书方法。上述工作往往缺乏定量的分析来回答诸如证书方法的性能如何依赖于网络配置和近似参数的选择等问题。在我们的框架下，我们首次尝试回答这些问题，这表明基于线性近似的认证的紧密性在很大程度上依赖于训练网络的配置。



## **48. Adversarial Robustness of Representation Learning for Knowledge Graphs**

知识图表示学习的对抗性稳健性 cs.LG

PhD Thesis at Trinity College Dublin, Ireland

**SubmitDate**: 2022-09-30    [abs](http://arxiv.org/abs/2210.00122v1) [paper-pdf](http://arxiv.org/pdf/2210.00122v1)

**Authors**: Peru Bhardwaj

**Abstract**: Knowledge graphs represent factual knowledge about the world as relationships between concepts and are critical for intelligent decision making in enterprise applications. New knowledge is inferred from the existing facts in the knowledge graphs by encoding the concepts and relations into low-dimensional feature vector representations. The most effective representations for this task, called Knowledge Graph Embeddings (KGE), are learned through neural network architectures. Due to their impressive predictive performance, they are increasingly used in high-impact domains like healthcare, finance and education. However, are the black-box KGE models adversarially robust for use in domains with high stakes? This thesis argues that state-of-the-art KGE models are vulnerable to data poisoning attacks, that is, their predictive performance can be degraded by systematically crafted perturbations to the training knowledge graph. To support this argument, two novel data poisoning attacks are proposed that craft input deletions or additions at training time to subvert the learned model's performance at inference time. These adversarial attacks target the task of predicting the missing facts in knowledge graphs using KGE models, and the evaluation shows that the simpler attacks are competitive with or outperform the computationally expensive ones. The thesis contributions not only highlight and provide an opportunity to fix the security vulnerabilities of KGE models, but also help to understand the black-box predictive behaviour of KGE models.

摘要: 知识图将关于世界的实际知识表示为概念之间的关系，对于企业应用程序中的智能决策至关重要。通过将概念和关系编码成低维特征向量表示，从知识图中的现有事实推断出新的知识。这项任务的最有效表示，称为知识图嵌入(KGE)，是通过神经网络结构学习的。由于其令人印象深刻的预测性能，它们越来越多地被用于医疗保健、金融和教育等高影响力领域。然而，黑盒KGE模型在高风险领域中的使用是否具有相反的健壮性？本文认为，最新的KGE模型容易受到数据中毒攻击，也就是说，通过系统地对训练知识图进行扰动，可以降低它们的预测性能。为了支持这一论点，提出了两种新的数据中毒攻击，它们在训练时手工删除或添加输入，以破坏学习模型在推理时的性能。这些对抗性攻击的目标是使用KGE模型预测知识图中缺失的事实，评估表明，较简单的攻击与计算代价较高的攻击竞争或性能更好。本文的贡献不仅突出并提供了修复KGE模型安全漏洞的机会，而且有助于理解KGE模型的黑盒预测行为。



## **49. Learning Robust Kernel Ensembles with Kernel Average Pooling**

利用核平均池化学习稳健核集成 cs.LG

**SubmitDate**: 2022-09-30    [abs](http://arxiv.org/abs/2210.00062v1) [paper-pdf](http://arxiv.org/pdf/2210.00062v1)

**Authors**: Pouya Bashivan, Adam Ibrahim, Amirozhan Dehghani, Yifei Ren

**Abstract**: Model ensembles have long been used in machine learning to reduce the variance in individual model predictions, making them more robust to input perturbations. Pseudo-ensemble methods like dropout have also been commonly used in deep learning models to improve generalization. However, the application of these techniques to improve neural networks' robustness against input perturbations remains underexplored. We introduce Kernel Average Pool (KAP), a new neural network building block that applies the mean filter along the kernel dimension of the layer activation tensor. We show that ensembles of kernels with similar functionality naturally emerge in convolutional neural networks equipped with KAP and trained with backpropagation. Moreover, we show that when combined with activation noise, KAP models are remarkably robust against various forms of adversarial attacks. Empirical evaluations on CIFAR10, CIFAR100, TinyImagenet, and Imagenet datasets show substantial improvements in robustness against strong adversarial attacks such as AutoAttack that are on par with adversarially trained networks but are importantly obtained without training on any adversarial examples.

摘要: 长期以来，模型集成一直被用于机器学习，以减少单个模型预测中的方差，使它们对输入扰动更具鲁棒性。丢弃等伪集合法也被广泛用于深度学习模型中，以提高泛化能力。然而，应用这些技术来提高神经网络对输入扰动的稳健性仍然没有得到充分的探索。我们介绍了核平均池(KAP)，这是一种新的神经网络构建块，它沿层激活张量的核维度应用平均滤波。我们证明了具有相似功能的核函数集成自然地出现在配备KAP并用反向传播训练的卷积神经网络中。此外，我们还表明，当结合激活噪声时，KAP模型对各种形式的对抗性攻击具有显著的健壮性。在CIFAR10、CIFAR100、TinyImagenet和Imagenet数据集上的经验评估表明，对AutoAttack等强对手攻击的稳健性有了显著改善，这些攻击与经过对手训练的网络一样，但重要的是在没有对任何对手示例进行训练的情况下获得。



## **50. Visual Privacy Protection Based on Type-I Adversarial Attack**

基于I型对抗攻击的视觉隐私保护 cs.CV

**SubmitDate**: 2022-09-30    [abs](http://arxiv.org/abs/2209.15304v1) [paper-pdf](http://arxiv.org/pdf/2209.15304v1)

**Authors**: Zhigang Su, Dawei Zhou, Decheng Liu, Nannan Wang, Zhen Wang, Xinbo Gao

**Abstract**: With the development of online artificial intelligence systems, many deep neural networks (DNNs) have been deployed in cloud environments. In practical applications, developers or users need to provide their private data to DNNs, such as faces. However, data transmitted and stored in the cloud is insecure and at risk of privacy leakage. In this work, inspired by Type-I adversarial attack, we propose an adversarial attack-based method to protect visual privacy of data. Specifically, the method encrypts the visual information of private data while maintaining them correctly predicted by DNNs, without modifying the model parameters. The empirical results on face recognition tasks show that the proposed method can deeply hide the visual information in face images and hardly affect the accuracy of the recognition models. In addition, we further extend the method to classification tasks and also achieve state-of-the-art performance.

摘要: 随着在线人工智能系统的发展，许多深度神经网络(DNN)被部署在云环境中。在实际应用中，开发者或用户需要将自己的私有数据提供给DNN，如人脸。然而，在云中传输和存储的数据是不安全的，并存在隐私泄露的风险。在这项工作中，我们受到I型对抗攻击的启发，提出了一种基于对抗攻击的数据视觉隐私保护方法。具体地说，该方法在不修改模型参数的情况下，对私有数据的可视信息进行加密，同时保持它们被DNN正确预测。在人脸识别任务上的实验结果表明，该方法能够较好地隐藏人脸图像中的视觉信息，且对识别模型的准确性几乎没有影响。此外，我们还将该方法进一步扩展到任务分类中，并获得了最先进的性能。



