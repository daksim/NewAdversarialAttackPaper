# Latest Adversarial Attack Papers
**update at 2022-04-25 06:31:25**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Contrastive Learning by Permuting Cluster Assignments**

基于置换类分配的对抗性对比学习 cs.LG

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.10314v1)

**Authors**: Muntasir Wahed, Afrina Tabassum, Ismini Lourentzou

**Abstracts**: Contrastive learning has gained popularity as an effective self-supervised representation learning technique. Several research directions improve traditional contrastive approaches, e.g., prototypical contrastive methods better capture the semantic similarity among instances and reduce the computational burden by considering cluster prototypes or cluster assignments, while adversarial instance-wise contrastive methods improve robustness against a variety of attacks. To the best of our knowledge, no prior work jointly considers robustness, cluster-wise semantic similarity and computational efficiency. In this work, we propose SwARo, an adversarial contrastive framework that incorporates cluster assignment permutations to generate representative adversarial samples. We evaluate SwARo on multiple benchmark datasets and against various white-box and black-box attacks, obtaining consistent improvements over state-of-the-art baselines.

摘要: 对比学习作为一种有效的自我监督表征学习技术已经得到了广泛的应用。一些研究方向改进了传统的对比方法，如原型对比方法更好地捕捉实例之间的语义相似性，并通过考虑簇原型或簇分配来减少计算负担，而对抗性实例对比方法提高了对各种攻击的健壮性。就我们所知，以前的工作没有同时考虑稳健性、聚类语义相似度和计算效率。在这项工作中，我们提出了SwARo，这是一个对抗性对比框架，它结合了簇分配排列来生成具有代表性的对抗性样本。我们在多个基准数据集上对SwARo进行评估，并针对各种白盒和黑盒攻击进行评估，在最先进的基线上获得持续的改进。



## **2. Robustness of Machine Learning Models Beyond Adversarial Attacks**

对抗攻击下机器学习模型的稳健性 cs.LG

25 pages, 7 figures

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.10046v1)

**Authors**: Sebastian Scher, Andreas Trügler

**Abstracts**: Correctly quantifying the robustness of machine learning models is a central aspect in judging their suitability for specific tasks, and thus, ultimately, for generating trust in the models. We show that the widely used concept of adversarial robustness and closely related metrics based on counterfactuals are not necessarily valid metrics for determining the robustness of ML models against perturbations that occur "naturally", outside specific adversarial attack scenarios. Additionally, we argue that generic robustness metrics in principle are insufficient for determining real-world-robustness. Instead we propose a flexible approach that models possible perturbations in input data individually for each application. This is then combined with a probabilistic approach that computes the likelihood that a real-world perturbation will change a prediction, thus giving quantitative information of the robustness of the trained machine learning model. The method does not require access to the internals of the classifier and thus in principle works for any black-box model. It is, however, based on Monte-Carlo sampling and thus only suited for input spaces with small dimensions. We illustrate our approach on two dataset, as well as on analytically solvable cases. Finally, we discuss ideas on how real-world robustness could be computed or estimated in high-dimensional input spaces.

摘要: 正确量化机器学习模型的稳健性是判断它们是否适合特定任务的一个中心方面，从而最终产生对模型的信任。我们表明，广泛使用的对抗性健壮性概念和基于反事实的密切相关的度量标准，并不一定是确定ML模型对特定对抗性攻击场景外的“自然”扰动的健壮性的有效度量。此外，我们认为一般的健壮性度量原则上不足以确定真实世界的健壮性。相反，我们提出了一种灵活的方法，为每个应用程序分别建模输入数据中可能的扰动。然后，将其与计算真实世界扰动将改变预测的可能性的概率方法相结合，从而给出训练后的机器学习模型的稳健性的定量信息。该方法不需要访问分类器的内部，因此原则上适用于任何黑盒模型。然而，它是基于蒙特卡罗抽样的，因此只适用于小维度的输入空间。我们在两个数据集上以及在分析可解的情况下说明了我们的方法。最后，我们讨论了如何在高维输入空间中计算或估计真实世界的稳健性。



## **3. Is Neuron Coverage Needed to Make Person Detection More Robust?**

需要神经元覆盖才能使人检测更可靠吗？ cs.CV

Accepted for publication at CVPR 2022 TCV workshop

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.10027v1)

**Authors**: Svetlana Pavlitskaya, Şiyar Yıkmış, J. Marius Zöllner

**Abstracts**: The growing use of deep neural networks (DNNs) in safety- and security-critical areas like autonomous driving raises the need for their systematic testing. Coverage-guided testing (CGT) is an approach that applies mutation or fuzzing according to a predefined coverage metric to find inputs that cause misbehavior. With the introduction of a neuron coverage metric, CGT has also recently been applied to DNNs. In this work, we apply CGT to the task of person detection in crowded scenes. The proposed pipeline uses YOLOv3 for person detection and includes finding DNN bugs via sampling and mutation, and subsequent DNN retraining on the updated training set. To be a bug, we require a mutated image to cause a significant performance drop compared to a clean input. In accordance with the CGT, we also consider an additional requirement of increased coverage in the bug definition. In order to explore several types of robustness, our approach includes natural image transformations, corruptions, and adversarial examples generated with the Daedalus attack. The proposed framework has uncovered several thousand cases of incorrect DNN behavior. The relative change in mAP performance of the retrained models reached on average between 26.21\% and 64.24\% for different robustness types. However, we have found no evidence that the investigated coverage metrics can be advantageously used to improve robustness.

摘要: 深度神经网络(DNN)在自动驾驶等安全和安保关键领域的应用越来越多，这增加了对其进行系统测试的必要性。覆盖引导测试(CGT)是一种根据预定义的覆盖度量应用突变或模糊来发现导致错误行为的输入的方法。随着神经元覆盖度量的引入，CGT最近也被应用于DNN。在这项工作中，我们将CGT应用于拥挤场景中的人检测任务。拟议的管道使用YOLOv3进行人员检测，包括通过采样和突变发现DNN错误，以及随后在更新的训练集上对DNN进行再培训。要成为一个错误，我们需要一个突变的图像来导致与干净的输入相比显著的性能下降。根据CGT，我们还考虑在错误定义中增加覆盖范围的额外要求。为了探索几种类型的健壮性，我们的方法包括自然图像转换、损坏和由Daedalus攻击生成的敌意示例。所提出的框架已经发现了数千例不正确的DNN行为。对于不同的稳健性类型，重新训练的模型的MAP性能的相对变化平均在26.21~64.24之间。然而，我们没有发现证据表明所调查的覆盖度量可以被有利地用于提高稳健性。



## **4. Eliminating Backdoor Triggers for Deep Neural Networks Using Attention Relation Graph Distillation**

基于注意力关系图提取的深度神经网络后门触发器剔除 cs.LG

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.09975v1)

**Authors**: Jun Xia, Ting Wang, Jieping Ding, Xian Wei, Mingsong Chen

**Abstracts**: Due to the prosperity of Artificial Intelligence (AI) techniques, more and more backdoors are designed by adversaries to attack Deep Neural Networks (DNNs).Although the state-of-the-art method Neural Attention Distillation (NAD) can effectively erase backdoor triggers from DNNs, it still suffers from non-negligible Attack Success Rate (ASR) together with lowered classification ACCuracy (ACC), since NAD focuses on backdoor defense using attention features (i.e., attention maps) of the same order. In this paper, we introduce a novel backdoor defense framework named Attention Relation Graph Distillation (ARGD), which fully explores the correlation among attention features with different orders using our proposed Attention Relation Graphs (ARGs). Based on the alignment of ARGs between both teacher and student models during knowledge distillation, ARGD can eradicate more backdoor triggers than NAD. Comprehensive experimental results show that, against six latest backdoor attacks, ARGD outperforms NAD by up to 94.85% reduction in ASR, while ACC can be improved by up to 3.23%.

摘要: 由于人工智能(AI)技术的蓬勃发展，越来越多的对手设计了后门来攻击深度神经网络(DNN)，尽管目前最先进的方法神经注意力蒸馏(NAD)可以有效地清除DNN中的后门触发，但由于NAD侧重于利用同阶的注意特征(即注意力地图)进行后门防御，因此仍然存在不可忽视的攻击成功率(ASR)和较低的分类精度(ACC)。本文介绍了一种新的后门防御框架--注意关系图蒸馏(ARGD)，它充分利用我们提出的注意关系图(ARGs)来探索不同阶次的注意特征之间的相关性。基于知识提炼过程中教师和学生模型之间的ARG对齐，ARGD比NAD能够消除更多的后门触发。综合实验结果表明，对于最近的6次后门攻击，ARGD在ASR上比NAD降低了94.85%，而ACC则提高了3.23%。



## **5. On the Certified Robustness for Ensemble Models and Beyond**

关于系综模型及以后模型的认证稳健性 cs.LG

ICLR 2022. 51 pages, 10 pages for main text. Forum and code:  https://openreview.net/forum?id=tUa4REjGjTf

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2107.10873v2)

**Authors**: Zhuolin Yang, Linyi Li, Xiaojun Xu, Bhavya Kailkhura, Tao Xie, Bo Li

**Abstracts**: Recent studies show that deep neural networks (DNN) are vulnerable to adversarial examples, which aim to mislead DNNs by adding perturbations with small magnitude. To defend against such attacks, both empirical and theoretical defense approaches have been extensively studied for a single ML model. In this work, we aim to analyze and provide the certified robustness for ensemble ML models, together with the sufficient and necessary conditions of robustness for different ensemble protocols. Although ensemble models are shown more robust than a single model empirically; surprisingly, we find that in terms of the certified robustness the standard ensemble models only achieve marginal improvement compared to a single model. Thus, to explore the conditions that guarantee to provide certifiably robust ensemble ML models, we first prove that diversified gradient and large confidence margin are sufficient and necessary conditions for certifiably robust ensemble models under the model-smoothness assumption. We then provide the bounded model-smoothness analysis based on the proposed Ensemble-before-Smoothing strategy. We also prove that an ensemble model can always achieve higher certified robustness than a single base model under mild conditions. Inspired by the theoretical findings, we propose the lightweight Diversity Regularized Training (DRT) to train certifiably robust ensemble ML models. Extensive experiments show that our DRT enhanced ensembles can consistently achieve higher certified robustness than existing single and ensemble ML models, demonstrating the state-of-the-art certified L2-robustness on MNIST, CIFAR-10, and ImageNet datasets.

摘要: 最近的研究表明，深度神经网络(DNN)很容易受到敌意例子的影响，这些例子旨在通过添加小幅度的扰动来误导DNN。为了防御这样的攻击，针对单个ML模型，已经广泛地研究了经验和理论防御方法。在这项工作中，我们旨在分析和提供集成ML模型的证明的稳健性，以及对于不同的集成协议的健壮性的充要条件。虽然从经验上看，集成模型比单个模型更稳健，但令人惊讶的是，我们发现标准集成模型在验证的稳健性方面，与单个模型相比只有轻微的改善。因此，为了探索保证提供可证明鲁棒的集成ML模型的条件，我们首先证明了在模型光滑性假设下，多样化的梯度和大的置信度是可证明鲁棒的集成模型的充要条件。然后，基于所提出的先集成后平滑策略，给出了有界模型光滑性分析。我们还证明了在温和的条件下，集成模型总是可以获得比单一基础模型更高的认证稳健性。受理论研究的启发，我们提出了轻量级多样性正则化训练(DRT)来训练可证明稳健的ML集成模型。广泛的实验表明，我们的DRT增强型集成可以持续实现比现有单一和集成ML模型更高的认证稳健性，在MNIST、CIFAR-10和ImageNet数据集上展示了最先进的认证L2稳健性。



## **6. Fast AdvProp**

Fast AdvProp cs.CV

ICLR 2022 camera ready version

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.09838v1)

**Authors**: Jieru Mei, Yucheng Han, Yutong Bai, Yixiao Zhang, Yingwei Li, Xianhang Li, Alan Yuille, Cihang Xie

**Abstracts**: Adversarial Propagation (AdvProp) is an effective way to improve recognition models, leveraging adversarial examples. Nonetheless, AdvProp suffers from the extremely slow training speed, mainly because: a) extra forward and backward passes are required for generating adversarial examples; b) both original samples and their adversarial counterparts are used for training (i.e., 2$\times$ data). In this paper, we introduce Fast AdvProp, which aggressively revamps AdvProp's costly training components, rendering the method nearly as cheap as the vanilla training. Specifically, our modifications in Fast AdvProp are guided by the hypothesis that disentangled learning with adversarial examples is the key for performance improvements, while other training recipes (e.g., paired clean and adversarial training samples, multi-step adversarial attackers) could be largely simplified.   Our empirical results show that, compared to the vanilla training baseline, Fast AdvProp is able to further model performance on a spectrum of visual benchmarks, without incurring extra training cost. Additionally, our ablations find Fast AdvProp scales better if larger models are used, is compatible with existing data augmentation methods (i.e., Mixup and CutMix), and can be easily adapted to other recognition tasks like object detection. The code is available here: https://github.com/meijieru/fast_advprop.

摘要: 对抗性传播(AdvProp)是利用对抗性例子改进识别模型的一种有效方法。然而，AdvProp的训练速度非常慢，主要是因为：a)需要额外的向前和向后传递来生成对抗性示例；b)原始样本和它们的对应物都用于训练(即2$\x$数据)。在本文中，我们介绍了Fast AdvProp，它积极地改造了AdvProp昂贵的训练组件，使该方法几乎与普通训练一样便宜。具体地说，我们在Fast AdvProp中的修改是在这样一个假设的指导下进行的，即与对抗性例子的分离学习是性能提高的关键，而其他训练食谱(例如，成对的干净和对抗性训练样本、多步骤对抗性攻击者)可以在很大程度上被简化。我们的实验结果表明，与普通的训练基准相比，Fast AdvProp能够在不产生额外训练成本的情况下，在一系列视觉基准上进一步建模性能。此外，我们的烧蚀发现，如果使用更大的模型，Fast AdvProp的规模会更好，与现有的数据增强方法(即Mixup和CutMix)兼容，并且可以很容易地适应其他识别任务，如目标检测。代码可在此处获得：https://github.com/meijieru/fast_advprop.



## **7. GUARD: Graph Universal Adversarial Defense**

后卫：GRAPH通用对抗性防御 cs.LG

Code is publicly available at https://github.com/EdisonLeeeee/GUARD

**SubmitDate**: 2022-04-20    [paper-pdf](http://arxiv.org/pdf/2204.09803v1)

**Authors**: Jintang Li, Jie Liao, Ruofan Wu, Liang Chen, Changhua Meng, Zibin Zheng, Weiqiang Wang

**Abstracts**: Recently, graph convolutional networks (GCNs) have shown to be vulnerable to small adversarial perturbations, which becomes a severe threat and largely limits their applications in security-critical scenarios. To mitigate such a threat, considerable research efforts have been devoted to increasing the robustness of GCNs against adversarial attacks. However, current approaches for defense are typically designed for the whole graph and consider the global performance, posing challenges in protecting important local nodes from stronger adversarial targeted attacks. In this work, we present a simple yet effective method, named \textbf{\underline{G}}raph \textbf{\underline{U}}niversal \textbf{\underline{A}}dve\textbf{\underline{R}}sarial \textbf{\underline{D}}efense (GUARD). Unlike previous works, GUARD protects each individual node from attacks with a universal defensive patch, which is generated once and can be applied to any node (node-agnostic) in a graph. Extensive experiments on four benchmark datasets demonstrate that our method significantly improves robustness for several established GCNs against multiple adversarial attacks and outperforms existing adversarial defense methods by large margins. Our code is publicly available at https://github.com/EdisonLeeeee/GUARD.

摘要: 最近，图卷积网络(GCNS)被证明容易受到微小的敌意扰动，这成为一种严重的威胁，并在很大程度上限制了它们在安全关键场景中的应用。为了减轻这种威胁，人们投入了大量的研究努力来提高GCNS对对手攻击的健壮性。然而，当前的防御方法通常是为整个图设计的，并考虑了全局性能，这给保护重要的局部节点免受更强的对抗性目标攻击带来了挑战。在这项工作中，我们提出了一种简单而有效的方法，命名为\Textbf{\下划线{G}}RAPH\Textbf{\下划线{U}}通用\textbf{\underline{A}}dve\textbf{\underline{R}}sarial\Textbf{\下划线{D}}保护。与以前的工作不同，Guard使用一个通用的防御补丁来保护每个单独的节点免受攻击，该补丁只生成一次，可以应用于图中的任何节点(与节点无关)。在四个基准数据集上的大量实验表明，我们的方法显著提高了几个已建立的GCN对多个对手攻击的稳健性，并大大超过了现有的对抗性防御方法。我们的代码在https://github.com/EdisonLeeeee/GUARD.上公开提供



## **8. Backdooring Explainable Machine Learning**

Backdoding可解释机器学习 cs.CR

**SubmitDate**: 2022-04-20    [paper-pdf](http://arxiv.org/pdf/2204.09498v1)

**Authors**: Maximilian Noppel, Lukas Peter, Christian Wressnegger

**Abstracts**: Explainable machine learning holds great potential for analyzing and understanding learning-based systems. These methods can, however, be manipulated to present unfaithful explanations, giving rise to powerful and stealthy adversaries. In this paper, we demonstrate blinding attacks that can fully disguise an ongoing attack against the machine learning model. Similar to neural backdoors, we modify the model's prediction upon trigger presence but simultaneously also fool the provided explanation. This enables an adversary to hide the presence of the trigger or point the explanation to entirely different portions of the input, throwing a red herring. We analyze different manifestations of such attacks for different explanation types in the image domain, before we resume to conduct a red-herring attack against malware classification.

摘要: 可解释机器学习在分析和理解基于学习的系统方面具有巨大的潜力。然而，这些方法可以被操纵来提供不可信的解释，从而产生强大而隐蔽的对手。在本文中，我们演示了盲攻击，可以完全掩盖对机器学习模型的正在进行的攻击。与神经后门类似，我们根据触发器的存在修改模型的预测，但同时也欺骗了所提供的解释。这使得对手可以隐藏触发器的存在，或者将解释指向输入的完全不同的部分，从而转移注意力。我们分析了这类攻击在图像域中针对不同解释类型的不同表现，然后继续进行针对恶意软件分类的转移话题攻击。



## **9. Adversarial Scratches: Deployable Attacks to CNN Classifiers**

对抗性抓痕：对CNN分类器的可部署攻击 cs.LG

This paper stems from 'Scratch that! An Evolution-based Adversarial  Attack against Neural Networks' for which an arXiv preprint is available at  arXiv:1912.02316. Further studies led to a complete overhaul of the work,  resulting in this paper. This work was submitted for review in Pattern  Recognition (Elsevier)

**SubmitDate**: 2022-04-20    [paper-pdf](http://arxiv.org/pdf/2204.09397v1)

**Authors**: Loris Giulivi, Malhar Jere, Loris Rossi, Farinaz Koushanfar, Gabriela Ciocarlie, Briland Hitaj, Giacomo Boracchi

**Abstracts**: A growing body of work has shown that deep neural networks are susceptible to adversarial examples. These take the form of small perturbations applied to the model's input which lead to incorrect predictions. Unfortunately, most literature focuses on visually imperceivable perturbations to be applied to digital images that often are, by design, impossible to be deployed to physical targets. We present Adversarial Scratches: a novel L0 black-box attack, which takes the form of scratches in images, and which possesses much greater deployability than other state-of-the-art attacks. Adversarial Scratches leverage B\'ezier Curves to reduce the dimension of the search space and possibly constrain the attack to a specific location. We test Adversarial Scratches in several scenarios, including a publicly available API and images of traffic signs. Results show that, often, our attack achieves higher fooling rate than other deployable state-of-the-art methods, while requiring significantly fewer queries and modifying very few pixels.

摘要: 越来越多的研究表明，深度神经网络很容易受到敌意例子的影响。这些采用的形式是应用于模型输入的小扰动，从而导致不正确的预测。不幸的是，大多数文献关注的是应用于数字图像的视觉上不可察觉的扰动，而根据设计，数字图像通常不可能被部署到物理目标上。我们提出了对抗性划痕：一种新颖的L0黑盒攻击，它采用图像划痕的形式，并且比其他最先进的攻击具有更大的可部署性。对抗性划痕利用B‘ezier曲线来减少搜索空间的维度，并可能将攻击限制在特定位置。我们在几个场景中测试了对抗性划痕，包括公开可用的API和交通标志图像。结果表明，我们的攻击通常比其他可部署的最先进方法获得更高的愚骗率，同时需要的查询和修改的像素也非常少。



## **10. You Are What You Write: Preserving Privacy in the Era of Large Language Models**

你写什么，你就是什么：在大型语言模型时代保护隐私 cs.CL

**SubmitDate**: 2022-04-20    [paper-pdf](http://arxiv.org/pdf/2204.09391v1)

**Authors**: Richard Plant, Valerio Giuffrida, Dimitra Gkatzia

**Abstracts**: Large scale adoption of large language models has introduced a new era of convenient knowledge transfer for a slew of natural language processing tasks. However, these models also run the risk of undermining user trust by exposing unwanted information about the data subjects, which may be extracted by a malicious party, e.g. through adversarial attacks. We present an empirical investigation into the extent of the personal information encoded into pre-trained representations by a range of popular models, and we show a positive correlation between the complexity of a model, the amount of data used in pre-training, and data leakage. In this paper, we present the first wide coverage evaluation and comparison of some of the most popular privacy-preserving algorithms, on a large, multi-lingual dataset on sentiment analysis annotated with demographic information (location, age and gender). The results show since larger and more complex models are more prone to leaking private information, use of privacy-preserving methods is highly desirable. We also find that highly privacy-preserving technologies like differential privacy (DP) can have serious model utility effects, which can be ameliorated using hybrid or metric-DP techniques.

摘要: 大型语言模型的大规模采用为一系列自然语言处理任务引入了一个方便的知识转移的新时代。然而，这些模型也存在通过暴露关于数据主体的不想要的信息来破坏用户信任的风险，这些信息可能由恶意方提取，例如通过对抗性攻击。我们通过一系列流行的模型对个人信息编码到预训练表示中的程度进行了实证研究，结果表明模型的复杂性、预训练中使用的数据量和数据泄露之间存在正相关关系。在本文中，我们首次对一些最流行的隐私保护算法进行了广泛的评估和比较，这些算法是在一个带有人口统计信息(位置、年龄和性别)的大型多语言情感分析数据集上进行的。结果表明，由于更大、更复杂的模型更容易泄露私人信息，因此使用隐私保护方法是非常可取的。我们还发现，像差分隐私(DP)这样的高度隐私保护技术可能会产生严重的模型效用效应，这可以使用混合或度量DP技术来改进。



## **11. Identifying Near-Optimal Single-Shot Attacks on ICSs with Limited Process Knowledge**

利用有限的流程知识识别ICSS上的近最佳单发攻击 cs.CR

This paper has been accepted at Applied Cryptography and Network  Security (ACNS) 2022

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.09106v1)

**Authors**: Herson Esquivel-Vargas, John Henry Castellanos, Marco Caselli, Nils Ole Tippenhauer, Andreas Peter

**Abstracts**: Industrial Control Systems (ICSs) rely on insecure protocols and devices to monitor and operate critical infrastructure. Prior work has demonstrated that powerful attackers with detailed system knowledge can manipulate exchanged sensor data to deteriorate performance of the process, even leading to full shutdowns of plants. Identifying those attacks requires iterating over all possible sensor values, and running detailed system simulation or analysis to identify optimal attacks. That setup allows adversaries to identify attacks that are most impactful when applied on the system for the first time, before the system operators become aware of the manipulations.   In this work, we investigate if constrained attackers without detailed system knowledge and simulators can identify comparable attacks. In particular, the attacker only requires abstract knowledge on general information flow in the plant, instead of precise algorithms, operating parameters, process models, or simulators. We propose an approach that allows single-shot attacks, i.e., near-optimal attacks that are reliably shutting down a system on the first try. The approach is applied and validated on two use cases, and demonstrated to achieve comparable results to prior work, which relied on detailed system information and simulations.

摘要: 工业控制系统(ICSS)依赖不安全的协议和设备来监控和运行关键基础设施。先前的工作已经证明，拥有详细系统知识的强大攻击者可以操纵交换的传感器数据，从而降低过程的性能，甚至导致工厂完全关闭。识别这些攻击需要迭代所有可能的传感器值，并运行详细的系统模拟或分析以确定最佳攻击。这种设置允许攻击者在系统操作员意识到操纵之前，识别首次应用于系统时最具影响力的攻击。在这项工作中，我们调查在没有详细系统知识和模拟器的情况下，受限攻击者是否能够识别类似的攻击。特别是，攻击者只需要工厂中一般信息流的抽象知识，而不是精确的算法、操作参数、过程模型或模拟器。我们提出了一种允许单发攻击的方法，即在第一次尝试时可靠地关闭系统的近最佳攻击。该方法在两个用例上进行了应用和验证，并证明了其结果与之前依赖于详细系统信息和模拟的工作类似。



## **12. Indiscriminate Data Poisoning Attacks on Neural Networks**

对神经网络的不分青红皂白的数据中毒攻击 cs.LG

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.09092v1)

**Authors**: Yiwei Lu, Gautam Kamath, Yaoliang Yu

**Abstracts**: Data poisoning attacks, in which a malicious adversary aims to influence a model by injecting "poisoned" data into the training process, have attracted significant recent attention. In this work, we take a closer look at existing poisoning attacks and connect them with old and new algorithms for solving sequential Stackelberg games. By choosing an appropriate loss function for the attacker and optimizing with algorithms that exploit second-order information, we design poisoning attacks that are effective on neural networks. We present efficient implementations that exploit modern auto-differentiation packages and allow simultaneous and coordinated generation of tens of thousands of poisoned points, in contrast to existing methods that generate poisoned points one by one. We further perform extensive experiments that empirically explore the effect of data poisoning attacks on deep neural networks.

摘要: 数据中毒攻击是指恶意对手通过在训练过程中注入“有毒”数据来影响模型的攻击，最近引起了极大的关注。在这项工作中，我们仔细研究现有的中毒攻击，并将它们与解决连续Stackelberg博弈的旧算法和新算法联系起来。通过为攻击者选择合适的损失函数，并利用二阶信息的算法进行优化，设计出对神经网络有效的中毒攻击。我们提供了利用现代自动区分包并允许同时和协调地生成数万个毒点的高效实现，与逐个生成毒点的现有方法形成对比。我们进一步进行了大量的实验，经验地探索了数据中毒攻击对深度神经网络的影响。



## **13. A Brief Survey on Deep Learning Based Data Hiding**

基于深度学习的数据隐藏研究综述 cs.CR

v2: reorganize some sections and add several new papers published in  2021~2022

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2103.01607v2)

**Authors**: Chaoning Zhang, Chenguo Lin, Philipp Benz, Kejiang Chen, Weiming Zhang, In So Kweon

**Abstracts**: Data hiding is the art of concealing messages with limited perceptual changes. Recently, deep learning has enriched it from various perspectives with significant progress. In this work, we conduct a brief yet comprehensive review of existing literature for deep learning based data hiding (deep hiding) by first classifying it according to three essential properties (i.e., capacity, security and robustness), and outline three commonly used architectures. Based on this, we summarize specific strategies for different applications of data hiding, including basic hiding, steganography, watermarking and light field messaging. Finally, further insight into deep hiding is provided by incorporating the perspective of adversarial attack.

摘要: 数据隐藏是通过有限的感知变化来隐藏消息的艺术。最近，深度学习从多个角度丰富了它，取得了重大进展。在这项工作中，我们对现有的基于深度学习的数据隐藏(深度隐藏)进行了简要而全面的回顾，首先根据三个基本属性(即容量、安全性和健壮性)对其进行分类，并概述了三种常用的体系结构。在此基础上，总结了针对不同应用的数据隐藏的具体策略，包括基本隐藏、隐写、水印和光场消息。最后，通过结合对抗性攻击的视角，对深层隐藏提供了进一步的洞察。



## **14. Jacobian Ensembles Improve Robustness Trade-offs to Adversarial Attacks**

雅可比集合提高了对抗攻击的稳健性权衡 cs.LG

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.08726v1)

**Authors**: Kenneth T. Co, David Martinez-Rego, Zhongyuan Hau, Emil C. Lupu

**Abstracts**: Deep neural networks have become an integral part of our software infrastructure and are being deployed in many widely-used and safety-critical applications. However, their integration into many systems also brings with it the vulnerability to test time attacks in the form of Universal Adversarial Perturbations (UAPs). UAPs are a class of perturbations that when applied to any input causes model misclassification. Although there is an ongoing effort to defend models against these adversarial attacks, it is often difficult to reconcile the trade-offs in model accuracy and robustness to adversarial attacks. Jacobian regularization has been shown to improve the robustness of models against UAPs, whilst model ensembles have been widely adopted to improve both predictive performance and model robustness. In this work, we propose a novel approach, Jacobian Ensembles-a combination of Jacobian regularization and model ensembles to significantly increase the robustness against UAPs whilst maintaining or improving model accuracy. Our results show that Jacobian Ensembles achieves previously unseen levels of accuracy and robustness, greatly improving over previous methods that tend to skew towards only either accuracy or robustness.

摘要: 深度神经网络已成为我们软件基础设施的组成部分，并被部署在许多广泛使用和安全关键的应用程序中。然而，它们与许多系统的集成也带来了测试通用对抗扰动(UAP)形式的时间攻击的脆弱性。UAP是一类扰动，当应用于任何输入时，都会导致模型错误分类。尽管人们一直在努力保护模型免受这些对抗性攻击，但通常很难在模型精确度和对对抗性攻击的稳健性之间进行权衡。雅可比正则化已被证明可以提高模型对UAP的稳健性，而模型集成已被广泛采用来提高预测性能和模型稳健性。在这项工作中，我们提出了一种新的方法，雅可比集成-雅可比正则化和模型集成的组合，在保持或改善模型精度的同时，显著增强了对UAP的鲁棒性。我们的结果表明，雅可比集成达到了前所未有的精度和稳健性水平，大大改进了以前的方法，这些方法倾向于只偏向精度或稳健性。



## **15. Topology and geometry of data manifold in deep learning**

深度学习中数据流形的拓扑和几何 cs.LG

12 pages, 15 figures

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.08624v1)

**Authors**: German Magai, Anton Ayzenberg

**Abstracts**: Despite significant advances in the field of deep learning in applications to various fields, explaining the inner processes of deep learning models remains an important and open question. The purpose of this article is to describe and substantiate the geometric and topological view of the learning process of neural networks. Our attention is focused on the internal representation of neural networks and on the dynamics of changes in the topology and geometry of the data manifold on different layers. We also propose a method for assessing the generalizing ability of neural networks based on topological descriptors. In this paper, we use the concepts of topological data analysis and intrinsic dimension, and we present a wide range of experiments on different datasets and different configurations of convolutional neural network architectures. In addition, we consider the issue of the geometry of adversarial attacks in the classification task and spoofing attacks on face recognition systems. Our work is a contribution to the development of an important area of explainable and interpretable AI through the example of computer vision.

摘要: 尽管深度学习领域在各个领域的应用取得了重大进展，但解释深度学习模型的内部过程仍然是一个重要而开放的问题。本文的目的是描述和充实神经网络学习过程的几何和拓扑观。我们的注意力集中在神经网络的内部表示以及不同层上数据流形的拓扑和几何变化的动力学上。提出了一种基于拓扑描述子的神经网络泛化能力评估方法。在本文中，我们使用了拓扑数据分析和内在维的概念，并在不同的数据集和不同结构的卷积神经网络结构上进行了广泛的实验。此外，我们还考虑了分类任务中敌意攻击的几何问题和对人脸识别系统的欺骗攻击。我们的工作是对通过计算机视觉的例子来发展可解释和可解释的人工智能的一个重要领域的贡献。



## **16. Poisons that are learned faster are more effective**

学得越快的毒药越有效 cs.LG

8 pages, 4 figures. Accepted to CVPR 2022 Art of Robustness Workshop

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.08615v1)

**Authors**: Pedro Sandoval-Segura, Vasu Singla, Liam Fowl, Jonas Geiping, Micah Goldblum, David Jacobs, Tom Goldstein

**Abstracts**: Imperceptible poisoning attacks on entire datasets have recently been touted as methods for protecting data privacy. However, among a number of defenses preventing the practical use of these techniques, early-stopping stands out as a simple, yet effective defense. To gauge poisons' vulnerability to early-stopping, we benchmark error-minimizing, error-maximizing, and synthetic poisons in terms of peak test accuracy over 100 epochs and make a number of surprising observations. First, we find that poisons that reach a low training loss faster have lower peak test accuracy. Second, we find that a current state-of-the-art error-maximizing poison is 7 times less effective when poison training is stopped at epoch 8. Third, we find that stronger, more transferable adversarial attacks do not make stronger poisons. We advocate for evaluating poisons in terms of peak test accuracy.

摘要: 对整个数据集的潜伏中毒攻击最近被吹捧为保护数据隐私的方法。然而，在阻止这些技术实际使用的许多防御措施中，提前停止是一种简单而有效的防御措施。为了衡量毒药对提前停止的脆弱性，我们根据100个纪元的峰值测试精度对误差最小化、误差最大化和合成毒药进行了基准测试，并进行了许多令人惊讶的观察。首先，我们发现毒药达到低训练损失的速度越快，峰值测试精度就越低。其次，我们发现当毒药训练在纪元8停止时，当前最先进的最大化错误的毒药的有效性降低了7倍。第三，我们发现更强、更具转移性的对抗性攻击不会产生更强的毒药。我们主张根据峰值测试的准确性来评估毒物。



## **17. Metamorphic Testing-based Adversarial Attack to Fool Deepfake Detectors**

基于变形测试的对愚人深伪检测器的攻击 cs.CV

paper submitted to 26TH International Conference on Pattern  Recognition (ICPR2022)

**SubmitDate**: 2022-04-19    [paper-pdf](http://arxiv.org/pdf/2204.08612v1)

**Authors**: Nyee Thoang Lim, Meng Yi Kuan, Muxin Pu, Mei Kuan Lim, Chun Yong Chong

**Abstracts**: Deepfakes utilise Artificial Intelligence (AI) techniques to create synthetic media where the likeness of one person is replaced with another. There are growing concerns that deepfakes can be maliciously used to create misleading and harmful digital contents. As deepfakes become more common, there is a dire need for deepfake detection technology to help spot deepfake media. Present deepfake detection models are able to achieve outstanding accuracy (>90%). However, most of them are limited to within-dataset scenario, where the same dataset is used for training and testing. Most models do not generalise well enough in cross-dataset scenario, where models are tested on unseen datasets from another source. Furthermore, state-of-the-art deepfake detection models rely on neural network-based classification models that are known to be vulnerable to adversarial attacks. Motivated by the need for a robust deepfake detection model, this study adapts metamorphic testing (MT) principles to help identify potential factors that could influence the robustness of the examined model, while overcoming the test oracle problem in this domain. Metamorphic testing is specifically chosen as the testing technique as it fits our demand to address learning-based system testing with probabilistic outcomes from largely black-box components, based on potentially large input domains. We performed our evaluations on MesoInception-4 and TwoStreamNet models, which are the state-of-the-art deepfake detection models. This study identified makeup application as an adversarial attack that could fool deepfake detectors. Our experimental results demonstrate that both the MesoInception-4 and TwoStreamNet models degrade in their performance by up to 30\% when the input data is perturbed with makeup.

摘要: Deepfakes利用人工智能(AI)技术来创建合成媒体，其中一个人的肖像被另一个人取代。越来越多的人担心，深度假货可能被恶意用于创建误导性和有害的数字内容。随着深度假变得越来越普遍，迫切需要深度假检测技术来帮助识别深度假媒体。现有的深度伪检测模型能够达到显著的准确率(>90%)。然而，它们中的大多数仅限于数据集内的场景，其中相同的数据集用于训练和测试。大多数模型在跨数据集情况下不能很好地泛化，在这种情况下，模型是在来自另一个来源的不可见的数据集上进行测试的。此外，最先进的深度伪检测模型依赖于基于神经网络的分类模型，这些模型已知容易受到对手攻击。出于对稳健深度伪检测模型的需求，本研究采用变形测试(MT)原理来帮助识别可能影响被检查模型的稳健性的潜在因素，同时克服了该领域的测试预言问题。变形测试被特别选为测试技术，因为它符合我们的需求，以解决基于学习的系统测试，其结果主要来自黑盒组件，基于潜在的大输入域。我们对目前最先进的深度伪检测模型MesoInception-4和TwoStreamNet模型进行了评估。这项研究发现，化妆应用是一种对抗性攻击，可以愚弄深度假货检测器。实验结果表明，当输入数据受到置乱干扰时，两种模型的性能都下降了30%。



## **18. UNBUS: Uncertainty-aware Deep Botnet Detection System in Presence of Perturbed Samples**

UNBUS：存在扰动样本的不确定性感知深度僵尸网络检测系统 cs.CR

8 pages, 5 figures, 5 Tables

**SubmitDate**: 2022-04-18    [paper-pdf](http://arxiv.org/pdf/2204.09502v1)

**Authors**: Rahim Taheri

**Abstracts**: A rising number of botnet families have been successfully detected using deep learning architectures. While the variety of attacks increases, these architectures should become more robust against attacks. They have been proven to be very sensitive to small but well constructed perturbations in the input. Botnet detection requires extremely low false-positive rates (FPR), which are not commonly attainable in contemporary deep learning. Attackers try to increase the FPRs by making poisoned samples. The majority of recent research has focused on the use of model loss functions to build adversarial examples and robust models. In this paper, two LSTM-based classification algorithms for botnet classification with an accuracy higher than 98\% are presented. Then, the adversarial attack is proposed, which reduces the accuracy to about30\%. Then, by examining the methods for computing the uncertainty, the defense method is proposed to increase the accuracy to about 70\%. By using the deep ensemble and stochastic weight averaging quantification methods it has been investigated the uncertainty of the accuracy in the proposed methods.

摘要: 使用深度学习体系结构已成功检测到越来越多的僵尸网络家族。随着攻击种类的增加，这些体系结构应该变得更强大，以抵御攻击。事实证明，它们对输入中的微小但构造良好的扰动非常敏感。僵尸网络检测需要极低的假阳性率(FPR)，这在当代深度学习中是不常见的。攻击者试图通过制作有毒样本来增加FPR。最近的大多数研究都集中在使用模型损失函数来构建对抗性例子和稳健模型。提出了两种基于LSTM的僵尸网络分类算法，分类准确率高于98.然后，提出了对抗性攻击，将准确率降低到30%左右。然后，通过研究不确定度的计算方法，提出了将精度提高到70%左右的防御方法。通过使用深度集成和随机加权平均量化方法，对所提出方法的精度的不确定度进行了研究。



## **19. A Comprehensive Survey on Trustworthy Graph Neural Networks: Privacy, Robustness, Fairness, and Explainability**

可信图神经网络研究综述：私密性、稳健性、公平性和可解释性 cs.LG

**SubmitDate**: 2022-04-18    [paper-pdf](http://arxiv.org/pdf/2204.08570v1)

**Authors**: Enyan Dai, Tianxiang Zhao, Huaisheng Zhu, Junjie Xu, Zhimeng Guo, Hui Liu, Jiliang Tang, Suhang Wang

**Abstracts**: Graph Neural Networks (GNNs) have made rapid developments in the recent years. Due to their great ability in modeling graph-structured data, GNNs are vastly used in various applications, including high-stakes scenarios such as financial analysis, traffic predictions, and drug discovery. Despite their great potential in benefiting humans in the real world, recent study shows that GNNs can leak private information, are vulnerable to adversarial attacks, can inherit and magnify societal bias from training data and lack interpretability, which have risk of causing unintentional harm to the users and society. For example, existing works demonstrate that attackers can fool the GNNs to give the outcome they desire with unnoticeable perturbation on training graph. GNNs trained on social networks may embed the discrimination in their decision process, strengthening the undesirable societal bias. Consequently, trustworthy GNNs in various aspects are emerging to prevent the harm from GNN models and increase the users' trust in GNNs. In this paper, we give a comprehensive survey of GNNs in the computational aspects of privacy, robustness, fairness, and explainability. For each aspect, we give the taxonomy of the related methods and formulate the general frameworks for the multiple categories of trustworthy GNNs. We also discuss the future research directions of each aspect and connections between these aspects to help achieve trustworthiness.

摘要: 近年来，图形神经网络(GNN)得到了迅速发展。由于其强大的图结构数据建模能力，GNN被广泛应用于各种应用中，包括金融分析、交通预测和药物发现等高风险场景。尽管GNN在现实世界中具有造福人类的巨大潜力，但最近的研究表明，GNN可能会泄露私人信息，容易受到对手攻击，会继承和放大来自训练数据的社会偏见，并且缺乏可解释性，这有可能对用户和社会造成无意的伤害。例如，现有的工作表明，攻击者可以欺骗GNN给出他们想要的结果，而训练图上的扰动并不明显。在社交网络上培训的GNN可能会在其决策过程中嵌入歧视，强化不受欢迎的社会偏见。因此，各个方面的可信GNN应运而生，以防止GNN模型的危害，增加用户对GNN的信任。本文从私密性、健壮性、公平性和可解释性等方面对GNN进行了全面的综述。对于每个方面，我们给出了相关方法的分类，并制定了多个类别的可信GNN的一般框架。我们还讨论了各个方面的未来研究方向以及这些方面之间的联系，以帮助实现可信性。



## **20. Special Session: Towards an Agile Design Methodology for Efficient, Reliable, and Secure ML Systems**

特别会议：为高效、可靠和安全的ML系统寻求敏捷设计方法 cs.AR

Appears at 40th IEEE VLSI Test Symposium (VTS 2022), 14 pages

**SubmitDate**: 2022-04-18    [paper-pdf](http://arxiv.org/pdf/2204.09514v1)

**Authors**: Shail Dave, Alberto Marchisio, Muhammad Abdullah Hanif, Amira Guesmi, Aviral Shrivastava, Ihsen Alouani, Muhammad Shafique

**Abstracts**: The real-world use cases of Machine Learning (ML) have exploded over the past few years. However, the current computing infrastructure is insufficient to support all real-world applications and scenarios. Apart from high efficiency requirements, modern ML systems are expected to be highly reliable against hardware failures as well as secure against adversarial and IP stealing attacks. Privacy concerns are also becoming a first-order issue. This article summarizes the main challenges in agile development of efficient, reliable and secure ML systems, and then presents an outline of an agile design methodology to generate efficient, reliable and secure ML systems based on user-defined constraints and objectives.

摘要: 在过去的几年里，机器学习(ML)的真实使用案例呈爆炸式增长。然而，当前的计算基础设施不足以支持所有现实世界的应用程序和场景。除了高效率的要求外，现代的ML系统预计在硬件故障时高度可靠，并在对手攻击和IP窃取攻击中安全。隐私问题也正在成为一个头等大事。总结了敏捷开发高效、可靠、安全的ML系统所面临的主要挑战，提出了一种基于用户定义的约束和目标生成高效、可靠、安全的ML系统的敏捷设计方法。



## **21. Optimal Layered Defense For Site Protection**

站点保护的最优分层防御 cs.OH

**SubmitDate**: 2022-04-18    [paper-pdf](http://arxiv.org/pdf/2204.08961v1)

**Authors**: Tsvetan Asamov, Emre Yamangil, Endre Boros, Paul Kantor, Fred Roberts

**Abstracts**: We present a model for layered security with applications to the protection of sites such as stadiums or large gathering places. We formulate the problem as one of maximizing the capture of illegal contraband. The objective function is indefinite and only limited information can be gained when the problem is solved by standard convex optimization methods. In order to solve the model, we develop a dynamic programming approach, and study its convergence properties. Additionally, we formulate a version of the problem aimed at addressing intelligent adversaries who can adjust their direction of attack as they observe changes in the site security. Furthermore, we also develop a method for the solution of the latter model. Finally, we perform computational experiments to demonstrate the use of our methods.

摘要: 我们提出了一种分层安全模型，并将其应用于体育场馆或大型集会场所等场所的保护。我们把这个问题表述为最大限度地捕获非法违禁品的问题。用标准的凸优化方法求解时，目标函数是不确定的，只能得到有限的信息。为了求解该模型，我们提出了一种动态规划方法，并研究了它的收敛性质。此外，我们制定了一个版本的问题，旨在解决智能对手谁可以调整他们的攻击方向，因为他们观察到网站安全的变化。此外，我们还发展了一种求解后一种模型的方法。最后，我们进行了计算实验，以演示我们的方法的使用。



## **22. Towards Robust Neural Networks via Orthogonal Diversity**

基于正交分集的稳健神经网络研究 cs.CV

**SubmitDate**: 2022-04-18    [paper-pdf](http://arxiv.org/pdf/2010.12190v3)

**Authors**: Kun Fang, Qinghua Tao, Yingwen Wu, Tao Li, Jia Cai, Feipeng Cai, Xiaolin Huang, Jie Yang

**Abstracts**: Deep Neural Networks (DNNs) are vulnerable to invisible perturbations on the images generated by adversarial attacks, which raises researches on the adversarial robustness of DNNs. A series of methods represented by the adversarial training and its variants have proven as one of the most effective techniques in enhancing the DNN robustness. Generally, adversarial training focuses on enriching the training data by involving perturbed data. Despite of the efficiency in defending specific attacks, adversarial training is benefited from the data augmentation, which does not contribute to the robustness of DNN itself and usually suffers from accuracy drop on clean data as well as inefficiency in unknown attacks. Towards the robustness of DNN itself, we propose a novel defense that aims at augmenting the model in order to learn features adaptive to diverse inputs, including adversarial examples. Specifically, we introduce multiple paths to augment the network, and impose orthogonality constraints on these paths. In addition, a margin-maximization loss is designed to further boost DIversity via Orthogonality (DIO). Extensive empirical results on various data sets, architectures, and attacks demonstrate the adversarial robustness of the proposed DIO.

摘要: 深度神经网络(DNN)易受敌意攻击产生的图像不可见扰动的影响，这就引发了对DNN对抗鲁棒性的研究。以对抗性训练及其变体为代表的一系列方法已被证明是增强DNN鲁棒性的最有效技术之一。一般来说，对抗性训练的重点是通过使用扰动数据来丰富训练数据。尽管DNN在防御特定攻击方面效率很高，但对抗性训练得益于数据增强，这并不有助于DNN本身的健壮性，而且通常会导致对干净数据的准确率下降，以及对未知攻击的效率低下。针对DNN本身的健壮性，我们提出了一种新的防御方法，旨在增强模型以学习适应不同输入的特征，包括对抗性例子。具体地说，我们引入多条路径来增强网络，并对这些路径施加正交性约束。此外，利润率最大化损失旨在通过正交性(DIO)进一步提高多样性。在各种数据集、体系结构和攻击上的广泛实验结果证明了所提出的DIO的对抗性健壮性。



## **23. Sardino: Ultra-Fast Dynamic Ensemble for Secure Visual Sensing at Mobile Edge**

Sardino：移动边缘安全视觉感知的超快动态合奏 cs.CV

**SubmitDate**: 2022-04-18    [paper-pdf](http://arxiv.org/pdf/2204.08189v1)

**Authors**: Qun Song, Zhenyu Yan, Wenjie Luo, Rui Tan

**Abstracts**: Adversarial example attack endangers the mobile edge systems such as vehicles and drones that adopt deep neural networks for visual sensing. This paper presents {\em Sardino}, an active and dynamic defense approach that renews the inference ensemble at run time to develop security against the adaptive adversary who tries to exfiltrate the ensemble and construct the corresponding effective adversarial examples. By applying consistency check and data fusion on the ensemble's predictions, Sardino can detect and thwart adversarial inputs. Compared with the training-based ensemble renewal, we use HyperNet to achieve {\em one million times} acceleration and per-frame ensemble renewal that presents the highest level of difficulty to the prerequisite exfiltration attacks. Moreover, the robustness of the renewed ensembles against adversarial examples is enhanced with adversarial learning for the HyperNet. We design a run-time planner that maximizes the ensemble size in favor of security while maintaining the processing frame rate. Beyond adversarial examples, Sardino can also address the issue of out-of-distribution inputs effectively. This paper presents extensive evaluation of Sardino's performance in counteracting adversarial examples and applies it to build a real-time car-borne traffic sign recognition system. Live on-road tests show the built system's effectiveness in maintaining frame rate and detecting out-of-distribution inputs due to the false positives of a preceding YOLO-based traffic sign detector.

摘要: 对抗性示例攻击危及采用深度神经网络进行视觉传感的移动边缘系统，如车辆和无人机。提出了一种主动的、动态的防御方法{em Sardino}，该方法在运行时更新推理集成，以提高安全性，防止自适应对手试图渗透集成并构造相应的有效对抗实例。通过对合奏的预测应用一致性检查和数据融合，萨迪诺可以检测和挫败敌方的输入。与基于训练的集成更新相比，我们使用HyperNet实现了加速和每帧集成更新，这对先决条件渗透攻击呈现出最高的难度。此外，通过超级网络的对抗性学习，更新的集成对对抗性示例的鲁棒性得到了增强。我们设计了一个运行时规划器，在保持处理帧速率的同时最大化集成大小以利于安全性。除了敌对的例子，萨迪诺还可以有效地解决分配外投入的问题。本文对Sardino在对抗敌意例子方面的表现进行了广泛的评估，并将其应用于构建一个实时车载交通标志识别系统。现场道路测试表明，所建立的系统在保持帧速率和检测由于先前基于YOLO的交通标志检测器的错误阳性而导致的不分布输入方面是有效的。



## **24. Can You Spot the Chameleon? Adversarially Camouflaging Images from Co-Salient Object Detection**

你能认出变色龙吗？对抗伪装图像以防止共显著目标检测 cs.CV

Accepted to CVPR 2022

**SubmitDate**: 2022-04-18    [paper-pdf](http://arxiv.org/pdf/2009.09258v5)

**Authors**: Ruijun Gao, Qing Guo, Felix Juefei-Xu, Hongkai Yu, Huazhu Fu, Wei Feng, Yang Liu, Song Wang

**Abstracts**: Co-salient object detection (CoSOD) has recently achieved significant progress and played a key role in retrieval-related tasks. However, it inevitably poses an entirely new safety and security issue, i.e., highly personal and sensitive content can potentially be extracting by powerful CoSOD methods. In this paper, we address this problem from the perspective of adversarial attacks and identify a novel task: adversarial co-saliency attack. Specially, given an image selected from a group of images containing some common and salient objects, we aim to generate an adversarial version that can mislead CoSOD methods to predict incorrect co-salient regions. Note that, compared with general white-box adversarial attacks for classification, this new task faces two additional challenges: (1) low success rate due to the diverse appearance of images in the group; (2) low transferability across CoSOD methods due to the considerable difference between CoSOD pipelines. To address these challenges, we propose the very first black-box joint adversarial exposure and noise attack (Jadena), where we jointly and locally tune the exposure and additive perturbations of the image according to a newly designed high-feature-level contrast-sensitive loss function. Our method, without any information on the state-of-the-art CoSOD methods, leads to significant performance degradation on various co-saliency detection datasets and makes the co-salient objects undetectable. This can have strong practical benefits in properly securing the large number of personal photos currently shared on the Internet. Moreover, our method is potential to be utilized as a metric for evaluating the robustness of CoSOD methods.

摘要: 共显著目标检测(CoSOD)近年来取得了重大进展，在检索相关任务中发挥了关键作用。然而，它不可避免地提出了一个全新的安全问题，即高度个人和敏感的内容可能会被强大的CoSOD方法提取出来。在本文中，我们从对抗性攻击的角度来解决这个问题，并提出了一种新的任务：对抗性共显攻击。特别是，给定一幅从一组包含一些常见和显著对象的图像中选择的图像，我们的目标是生成一个对抗性版本，该版本可能会误导CoSOD方法预测错误的共同显著区域。注意到，与一般的白盒对抗性分类攻击相比，这项新任务面临着两个额外的挑战：(1)由于组中图像的多样性，成功率较低；(2)由于CoSOD管道之间的巨大差异，CoSOD方法之间的可传输性较低。为了应对这些挑战，我们提出了第一个黑盒联合对抗性曝光和噪声攻击(Jadena)，其中我们根据新设计的高特征级别对比度敏感损失函数来联合和局部地调整图像的曝光和加性扰动。我们的方法在没有关于最新CoSOD方法的任何信息的情况下，导致在各种共显著检测数据集上的性能显著下降，并且使得共显著对象不可检测。这对妥善保护目前在互联网上共享的大量个人照片具有很大的实际好处。此外，我们的方法有可能被用作评估CoSOD方法的稳健性的一个度量。



## **25. Learning Compositional Representations for Effective Low-Shot Generalization**

学习成分表示以实现有效的低概率概括 cs.CV

**SubmitDate**: 2022-04-17    [paper-pdf](http://arxiv.org/pdf/2204.08090v1)

**Authors**: Samarth Mishra, Pengkai Zhu, Venkatesh Saligrama

**Abstracts**: We propose Recognition as Part Composition (RPC), an image encoding approach inspired by human cognition. It is based on the cognitive theory that humans recognize complex objects by components, and that they build a small compact vocabulary of concepts to represent each instance with. RPC encodes images by first decomposing them into salient parts, and then encoding each part as a mixture of a small number of prototypes, each representing a certain concept. We find that this type of learning inspired by human cognition can overcome hurdles faced by deep convolutional networks in low-shot generalization tasks, like zero-shot learning, few-shot learning and unsupervised domain adaptation. Furthermore, we find a classifier using an RPC image encoder is fairly robust to adversarial attacks, that deep neural networks are known to be prone to. Given that our image encoding principle is based on human cognition, one would expect the encodings to be interpretable by humans, which we find to be the case via crowd-sourcing experiments. Finally, we propose an application of these interpretable encodings in the form of generating synthetic attribute annotations for evaluating zero-shot learning methods on new datasets.

摘要: 我们提出了一种受人类认知启发的图像编码方法，即识别为部件组成(RPC)。它的基础是认知理论，即人类通过组件识别复杂的对象，并建立一个小型紧凑的概念词汇来表示每个实例。RPC首先将图像分解成显著的部分，然后将每个部分编码为少量原型的混合物，每个原型代表一个特定的概念。我们发现，这种受人类认知启发的学习可以克服深度卷积网络在低概率泛化任务中面临的障碍，如零概率学习、少概率学习和无监督领域自适应。此外，我们发现使用RPC图像编码器的分类器对对手攻击具有相当的健壮性，而深层神经网络是已知容易受到的攻击。鉴于我们的图像编码原理是基于人类认知的，人们会认为编码是人类可以理解的，我们通过众包实验发现了这一点。最后，我们以生成合成属性注释的形式提出了这些可解释编码的应用，以评估新数据集上的零射击学习方法。



## **26. Residue-Based Natural Language Adversarial Attack Detection**

基于残差的自然语言敌意攻击检测 cs.CL

**SubmitDate**: 2022-04-17    [paper-pdf](http://arxiv.org/pdf/2204.10192v1)

**Authors**: Vyas Raina, Mark Gales

**Abstracts**: Deep learning based systems are susceptible to adversarial attacks, where a small, imperceptible change at the input alters the model prediction. However, to date the majority of the approaches to detect these attacks have been designed for image processing systems. Many popular image adversarial detection approaches are able to identify adversarial examples from embedding feature spaces, whilst in the NLP domain existing state of the art detection approaches solely focus on input text features, without consideration of model embedding spaces. This work examines what differences result when porting these image designed strategies to Natural Language Processing (NLP) tasks - these detectors are found to not port over well. This is expected as NLP systems have a very different form of input: discrete and sequential in nature, rather than the continuous and fixed size inputs for images. As an equivalent model-focused NLP detection approach, this work proposes a simple sentence-embedding "residue" based detector to identify adversarial examples. On many tasks, it out-performs ported image domain detectors and recent state of the art NLP specific detectors.

摘要: 基于深度学习的系统很容易受到对抗性攻击，在这种攻击中，输入端微小的、不可察觉的变化就会改变模型预测。然而，到目前为止，大多数检测这些攻击的方法都是为图像处理系统设计的。许多流行的图像对抗性检测方法能够从嵌入的特征空间中识别对抗性样本，而在NLP领域，现有的检测方法只关注输入文本特征，而没有考虑模型嵌入空间。这项工作考察了将这些图像设计的策略移植到自然语言处理(NLP)任务中时会产生什么不同--这些检测器被发现移植得不好。这是意料之中的，因为NLP系统具有非常不同的输入形式：本质上是离散的和连续的，而不是图像的连续和固定大小的输入。作为一种等价的基于模型的NLP检测方法，本文提出了一种简单的基于句子嵌入“残差”的检测器来识别对抗性实例。在许多任务上，它的性能优于端口图像域检测器和最新的NLP特定检测器。



## **27. Towards Comprehensive Testing on the Robustness of Cooperative Multi-agent Reinforcement Learning**

协作式多智能体强化学习稳健性的综合测试 cs.MA

**SubmitDate**: 2022-04-17    [paper-pdf](http://arxiv.org/pdf/2204.07932v1)

**Authors**: Jun Guo, Yonghong Chen, Yihang Hao, Zixin Yin, Yin Yu, Simin Li

**Abstracts**: While deep neural networks (DNNs) have strengthened the performance of cooperative multi-agent reinforcement learning (c-MARL), the agent policy can be easily perturbed by adversarial examples. Considering the safety critical applications of c-MARL, such as traffic management, power management and unmanned aerial vehicle control, it is crucial to test the robustness of c-MARL algorithm before it was deployed in reality. Existing adversarial attacks for MARL could be used for testing, but is limited to one robustness aspects (e.g., reward, state, action), while c-MARL model could be attacked from any aspect. To overcome the challenge, we propose MARLSafe, the first robustness testing framework for c-MARL algorithms. First, motivated by Markov Decision Process (MDP), MARLSafe consider the robustness of c-MARL algorithms comprehensively from three aspects, namely state robustness, action robustness and reward robustness. Any c-MARL algorithm must simultaneously satisfy these robustness aspects to be considered secure. Second, due to the scarceness of c-MARL attack, we propose c-MARL attacks as robustness testing algorithms from multiple aspects. Experiments on \textit{SMAC} environment reveals that many state-of-the-art c-MARL algorithms are of low robustness in all aspect, pointing out the urgent need to test and enhance robustness of c-MARL algorithms.

摘要: 虽然深度神经网络(DNN)增强了协作多智能体强化学习(c-Marl)的性能，但智能体策略很容易受到对抗性例子的干扰。考虑到c-Marl算法在交通管理、电源管理、无人机控制等安全关键应用中的应用，在c-Marl算法投入实际应用之前，对其健壮性进行测试是至关重要的。现有的针对Marl的对抗性攻击可以用于测试，但仅限于一个健壮性方面(例如，奖励、状态、动作)，而c-Marl模型可以从任何方面进行攻击。为了克服这一挑战，我们提出了第一个c-Marl算法健壮性测试框架MARLSafe。首先，在马尔可夫决策过程(MDP)的启发下，MARLSafe从状态稳健性、动作稳健性和奖赏稳健性三个方面综合考虑了c-Marl算法的稳健性。任何c-Marl算法都必须同时满足这些健壮性方面才能被认为是安全的。其次，针对c-Marl攻击的稀缺性，从多个方面提出了c-Marl攻击作为健壮性测试算法。在Smac环境下的实验表明，许多现有的c-Marl算法在各个方面的健壮性都较低，这表明迫切需要测试和提高c-Marl算法的健壮性。



## **28. SETTI: A Self-supervised Adversarial Malware Detection Architecture in an IoT Environment**

SETTI：物联网环境下的自监督恶意软件检测体系结构 cs.CR

20 pages, 6 figures, 2 Tables, Submitted to ACM Transactions on  Multimedia Computing, Communications, and Applications

**SubmitDate**: 2022-04-16    [paper-pdf](http://arxiv.org/pdf/2204.07772v1)

**Authors**: Marjan Golmaryami, Rahim Taheri, Zahra Pooranian, Mohammad Shojafar, Pei Xiao

**Abstracts**: In recent years, malware detection has become an active research topic in the area of Internet of Things (IoT) security. The principle is to exploit knowledge from large quantities of continuously generated malware. Existing algorithms practice available malware features for IoT devices and lack real-time prediction behaviors. More research is thus required on malware detection to cope with real-time misclassification of the input IoT data. Motivated by this, in this paper we propose an adversarial self-supervised architecture for detecting malware in IoT networks, SETTI, considering samples of IoT network traffic that may not be labeled. In the SETTI architecture, we design three self-supervised attack techniques, namely Self-MDS, GSelf-MDS and ASelf-MDS. The Self-MDS method considers the IoT input data and the adversarial sample generation in real-time. The GSelf-MDS builds a generative adversarial network model to generate adversarial samples in the self-supervised structure. Finally, ASelf-MDS utilizes three well-known perturbation sample techniques to develop adversarial malware and inject it over the self-supervised architecture. Also, we apply a defence method to mitigate these attacks, namely adversarial self-supervised training to protect the malware detection architecture against injecting the malicious samples. To validate the attack and defence algorithms, we conduct experiments on two recent IoT datasets: IoT23 and NBIoT. Comparison of the results shows that in the IoT23 dataset, the Self-MDS method has the most damaging consequences from the attacker's point of view by reducing the accuracy rate from 98% to 74%. In the NBIoT dataset, the ASelf-MDS method is the most devastating algorithm that can plunge the accuracy rate from 98% to 77%.

摘要: 近年来，恶意软件检测已成为物联网安全领域中一个活跃的研究课题。其原则是利用从大量不断生成的恶意软件中获取的知识。现有算法实践物联网设备可用的恶意软件功能，缺乏实时预测行为。因此，需要对恶意软件检测进行更多研究，以应对输入物联网数据的实时误分类。基于此，本文提出了一种对抗性自监督的物联网恶意软件检测体系结构SETTI，该体系结构考虑了物联网网络流量样本中可能未标记的恶意软件。在SETTI体系结构中，我们设计了三种自监督攻击技术，即Self-MDS、GSelf-MDS和ASself-MDS。Self-MDS方法实时考虑物联网输入数据和对抗性样本生成。GSelf-MDS建立了一个生成性对抗性网络模型，用于在自我监督结构中生成对抗性样本。最后，ASself-MDS利用三种众所周知的扰动采样技术来开发恶意软件，并将其注入到自我监督的体系结构中。此外，我们还应用了一种防御方法来缓解这些攻击，即对抗性的自我监督训练，以保护恶意软件检测体系结构免受注入恶意样本的攻击。为了验证攻击和防御算法，我们在最近的两个物联网数据集：IoT23和NBIoT上进行了实验。结果比较表明，在IoT23数据集中，从攻击者的角度来看，self-MDS方法具有最大的破坏性后果，将准确率从98%降低到74%。在NBIoT数据集中，ASself-MDS方法是最具破坏性的算法，其准确率可以从98%下降到77%。



## **29. Homomorphic Encryption and Federated Learning based Privacy-Preserving CNN Training: COVID-19 Detection Use-Case**

基于同态加密和联合学习的隐私保护CNN训练：新冠肺炎检测用例 cs.CR

European Interdisciplinary Cybersecurity Conference (EICC) 2022  publication

**SubmitDate**: 2022-04-16    [paper-pdf](http://arxiv.org/pdf/2204.07752v1)

**Authors**: Febrianti Wibawa, Ferhat Ozgur Catak, Salih Sarp, Murat Kuzlu, Umit Cali

**Abstracts**: Medical data is often highly sensitive in terms of data privacy and security concerns. Federated learning, one type of machine learning techniques, has been started to use for the improvement of the privacy and security of medical data. In the federated learning, the training data is distributed across multiple machines, and the learning process is performed in a collaborative manner. There are several privacy attacks on deep learning (DL) models to get the sensitive information by attackers. Therefore, the DL model itself should be protected from the adversarial attack, especially for applications using medical data. One of the solutions for this problem is homomorphic encryption-based model protection from the adversary collaborator. This paper proposes a privacy-preserving federated learning algorithm for medical data using homomorphic encryption. The proposed algorithm uses a secure multi-party computation protocol to protect the deep learning model from the adversaries. In this study, the proposed algorithm using a real-world medical dataset is evaluated in terms of the model performance.

摘要: 就数据隐私和安全问题而言，医疗数据往往高度敏感。联合学习是机器学习技术的一种，已经开始用于提高医疗数据的隐私和安全性。在联合学习中，训练数据分布在多台机器上，学习过程以协作的方式进行。攻击者为了获取敏感信息，对深度学习模型进行了几次隐私攻击。因此，DL模型本身应该受到保护，尤其是对于使用医疗数据的应用程序。解决这一问题的方案之一是基于同态加密的模型保护，使其免受敌方合作者的攻击。提出了一种基于同态加密的医学数据隐私保护联合学习算法。该算法使用安全的多方计算协议来保护深度学习模型不受攻击者的攻击。在这项研究中，使用真实世界的医学数据集对所提出的算法的模型性能进行了评估。



## **30. An Overview of Compressible and Learnable Image Transformation with Secret Key and Its Applications**

基于密钥的可压缩可学习图像变换及其应用综述 cs.CV

**SubmitDate**: 2022-04-16    [paper-pdf](http://arxiv.org/pdf/2201.11006v2)

**Authors**: Hitoshi Kiya, AprilPyone MaungMaung, Yuma Kinoshita, Shoko Imaizumi, Sayaka Shiota

**Abstracts**: This article presents an overview of image transformation with a secret key and its applications. Image transformation with a secret key enables us not only to protect visual information on plain images but also to embed unique features controlled with a key into images. In addition, numerous encryption methods can generate encrypted images that are compressible and learnable for machine learning. Various applications of such transformation have been developed by using these properties. In this paper, we focus on a class of image transformation referred to as learnable image encryption, which is applicable to privacy-preserving machine learning and adversarially robust defense. Detailed descriptions of both transformation algorithms and performances are provided. Moreover, we discuss robustness against various attacks.

摘要: 本文概述了使用密钥的图像变换及其应用。使用密钥的图像变换不仅可以保护普通图像上的视觉信息，还可以在图像中嵌入由密钥控制的独特特征。此外，许多加密方法可以生成可压缩和可学习的加密图像，以供机器学习。利用这些性质已经开发了这种变换的各种应用。在本文中，我们重点研究了一类被称为可学习图像加密的图像变换，它适用于隐私保护机器学习和对抗鲁棒防御。给出了变换算法和性能的详细描述。此外，我们还讨论了对各种攻击的健壮性。



## **31. Revisiting the Adversarial Robustness-Accuracy Tradeoff in Robot Learning**

机器人学习中对抗性稳健性与准确性权衡的再认识 cs.RO

**SubmitDate**: 2022-04-15    [paper-pdf](http://arxiv.org/pdf/2204.07373v1)

**Authors**: Mathias Lechner, Alexander Amini, Daniela Rus, Thomas A. Henzinger

**Abstracts**: Adversarial training (i.e., training on adversarially perturbed input data) is a well-studied method for making neural networks robust to potential adversarial attacks during inference. However, the improved robustness does not come for free but rather is accompanied by a decrease in overall model accuracy and performance. Recent work has shown that, in practical robot learning applications, the effects of adversarial training do not pose a fair trade-off but inflict a net loss when measured in holistic robot performance. This work revisits the robustness-accuracy trade-off in robot learning by systematically analyzing if recent advances in robust training methods and theory in conjunction with adversarial robot learning can make adversarial training suitable for real-world robot applications. We evaluate a wide variety of robot learning tasks ranging from autonomous driving in a high-fidelity environment amenable to sim-to-real deployment, to mobile robot gesture recognition. Our results demonstrate that, while these techniques make incremental improvements on the trade-off on a relative scale, the negative side-effects caused by adversarial training still outweigh the improvements by an order of magnitude. We conclude that more substantial advances in robust learning methods are necessary before they can benefit robot learning tasks in practice.

摘要: 对抗性训练(即对对抗性扰动的输入数据进行训练)是一种研究得很好的方法，可以使神经网络在推理过程中对潜在的对抗性攻击具有健壮性。然而，稳健性的提高并不是免费的，而是伴随着总体模型精度和性能的下降。最近的工作表明，在实际的机器人学习应用中，对抗性训练的效果并不构成公平的权衡，而是在衡量整体机器人性能时造成净损失。这项工作通过系统地分析稳健训练方法和理论的最新进展以及对抗性机器人学习是否可以使对抗性训练适用于现实世界的机器人应用，重新审视了机器人学习中的稳健性和精确度之间的权衡。我们评估了各种机器人学习任务，从高保真环境中的自动驾驶到模拟真实的部署，再到移动机器人手势识别。我们的结果表明，虽然这些技术在相对规模上对权衡做出了增量改进，但对抗性训练造成的负面副作用仍然比改进多一个数量级。我们的结论是，在健壮学习方法能够在实践中有益于机器人学习任务之前，需要更多实质性的进步。



## **32. Robotic and Generative Adversarial Attacks in Offline Writer-independent Signature Verification**

离线作者无关签名验证中的机器人和生成性对抗攻击 cs.RO

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2204.07246v1)

**Authors**: Jordan J. Bird

**Abstracts**: This study explores how robots and generative approaches can be used to mount successful false-acceptance adversarial attacks on signature verification systems. Initially, a convolutional neural network topology and data augmentation strategy are explored and tuned, producing an 87.12% accurate model for the verification of 2,640 human signatures. Two robots are then tasked with forging 50 signatures, where 25 are used for the verification attack, and the remaining 25 are used for tuning of the model to defend against them. Adversarial attacks on the system show that there exists an information security risk; the Line-us robotic arm can fool the system 24% of the time and the iDraw 2.0 robot 32% of the time. A conditional GAN finds similar success, with around 30% forged signatures misclassified as genuine. Following fine-tune transfer learning of robotic and generative data, adversarial attacks are reduced below the model threshold by both robots and the GAN. It is observed that tuning the model reduces the risk of attack by robots to 8% and 12%, and that conditional generative adversarial attacks can be reduced to 4% when 25 images are presented and 5% when 1000 images are presented.

摘要: 这项研究探索了如何使用机器人和生成性方法来对签名验证系统发起成功的虚假接受对抗性攻击。首先，探索和调整了卷积神经网络的拓扑结构和数据增强策略，产生了一个87.12%的模型，用于验证2640个人的签名。然后，两个机器人的任务是伪造50个签名，其中25个用于验证攻击，其余25个用于调整模型以防御它们。对该系统的对抗性攻击表明，存在信息安全风险；Line-us机械臂可以在24%的时间内欺骗系统，iDraw 2.0机器人可以在32%的时间内欺骗系统。有条件的GAN发现了类似的成功，大约30%的伪造签名被错误归类为真签名。在对机器人和生成性数据进行微调传递学习后，机器人和GAN都将对抗性攻击降低到模型阈值以下。实验结果表明，调整模型后，机器人的攻击风险分别降低到8%和12%，当呈现25幅图像时，条件生成性对抗攻击可以降低到4%，当呈现1000幅图像时，条件生成性对抗攻击可以降低到5%。



## **33. ExPLoit: Extracting Private Labels in Split Learning**

漏洞：在分裂学习中提取私有标签 cs.CR

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2112.01299v2)

**Authors**: Sanjay Kariyappa, Moinuddin K Qureshi

**Abstracts**: Split learning is a popular technique used for vertical federated learning (VFL), where the goal is to jointly train a model on the private input and label data held by two parties. This technique uses a split-model, trained end-to-end, by exchanging the intermediate representations (IR) of the inputs and gradients of the IR between the two parties. We propose ExPLoit - a label-leakage attack that allows an adversarial input-owner to extract the private labels of the label-owner during split-learning. ExPLoit frames the attack as a supervised learning problem by using a novel loss function that combines gradient-matching and several regularization terms developed using key properties of the dataset and models. Our evaluations show that ExPLoit can uncover the private labels with near-perfect accuracy of up to 99.96%. Our findings underscore the need for better training techniques for VFL.

摘要: 分裂学习是垂直联合学习(VFL)中一种流行的技术，其目标是联合训练一个关于双方持有的私有输入和标签数据的模型。该技术通过在双方之间交换IR的输入和梯度的中间表示(IR)来使用端到端训练的拆分模型。我们提出了利用攻击-一种标签泄漏攻击，允许敌意输入所有者在分裂学习过程中提取标签所有者的私有标签。利用攻击通过使用一种新的损失函数将攻击帧化为有监督的学习问题，该损失函数结合了梯度匹配和利用数据集和模型的关键属性开发的几个正则化项。我们的评估表明，利用漏洞可以发现私人标签的近乎完美的准确率高达99.96%。我们的发现强调了为VFL提供更好的训练技术的必要性。



## **34. From Environmental Sound Representation to Robustness of 2D CNN Models Against Adversarial Attacks**

从环境声音表示到2D CNN模型对敌方攻击的稳健性 cs.SD

32 pages, Preprint Submitted to Journal of Applied Acoustics. arXiv  admin note: substantial text overlap with arXiv:2007.13703

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2204.07018v1)

**Authors**: Mohammad Esmaeilpour, Patrick Cardinal, Alessandro Lameiras Koerich

**Abstracts**: This paper investigates the impact of different standard environmental sound representations (spectrograms) on the recognition performance and adversarial attack robustness of a victim residual convolutional neural network, namely ResNet-18. Our main motivation for focusing on such a front-end classifier rather than other complex architectures is balancing recognition accuracy and the total number of training parameters. Herein, we measure the impact of different settings required for generating more informative Mel-frequency cepstral coefficient (MFCC), short-time Fourier transform (STFT), and discrete wavelet transform (DWT) representations on our front-end model. This measurement involves comparing the classification performance over the adversarial robustness. We demonstrate an inverse relationship between recognition accuracy and model robustness against six benchmarking attack algorithms on the balance of average budgets allocated by the adversary and the attack cost. Moreover, our experimental results have shown that while the ResNet-18 model trained on DWT spectrograms achieves a high recognition accuracy, attacking this model is relatively more costly for the adversary than other 2D representations. We also report some results on different convolutional neural network architectures such as ResNet-34, ResNet-56, AlexNet, and GoogLeNet, SB-CNN, and LSTM-based.

摘要: 研究了不同标准环境声音表示(谱图)对受害者残差卷积神经网络ResNet-18识别性能和对抗攻击稳健性的影响。我们关注这样的前端分类器而不是其他复杂的体系结构的主要动机是在识别精度和训练参数总数之间取得平衡。在这里，我们测量了在我们的前端模型上生成更多信息的梅尔频率倒谱系数(MFCC)、短时傅立叶变换(STFT)和离散小波变换(DWT)表示所需的不同设置的影响。这种测量包括比较分类性能与对手健壮性。在平衡对手分配的平均预算和攻击成本的基础上，我们证明了识别准确率与模型对六种基准攻击算法的稳健性成反比关系。此外，我们的实验结果表明，虽然基于DWT谱图训练的ResNet-18模型达到了较高的识别精度，但攻击该模型的代价相对较高。我们还报告了不同卷积神经网络结构的一些结果，例如ResNet-34、ResNet-56、AlexNet和GoogLeNet、SB-CNN和基于LSTM的结构。



## **35. Finding MNEMON: Reviving Memories of Node Embeddings**

寻找Mnemon：唤醒节点嵌入的记忆 cs.LG

To Appear in the 29th ACM Conference on Computer and Communications  Security (CCS), November 7-11, 2022

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2204.06963v1)

**Authors**: Yun Shen, Yufei Han, Zhikun Zhang, Min Chen, Ting Yu, Michael Backes, Yang Zhang, Gianluca Stringhini

**Abstracts**: Previous security research efforts orbiting around graphs have been exclusively focusing on either (de-)anonymizing the graphs or understanding the security and privacy issues of graph neural networks. Little attention has been paid to understand the privacy risks of integrating the output from graph embedding models (e.g., node embeddings) with complex downstream machine learning pipelines. In this paper, we fill this gap and propose a novel model-agnostic graph recovery attack that exploits the implicit graph structural information preserved in the embeddings of graph nodes. We show that an adversary can recover edges with decent accuracy by only gaining access to the node embedding matrix of the original graph without interactions with the node embedding models. We demonstrate the effectiveness and applicability of our graph recovery attack through extensive experiments.

摘要: 以前围绕图的安全研究一直专注于图的(去)匿名化或理解图神经网络的安全和隐私问题。很少有人注意到将图嵌入模型(例如，节点嵌入)的输出与复杂的下游机器学习管道集成的隐私风险。在本文中，我们填补了这一空白，并提出了一种新的模型不可知图恢复攻击，该攻击利用了图节点嵌入中保留的隐含的图结构信息。我们证明了敌手只需访问原始图的节点嵌入矩阵，而不需要与节点嵌入模型交互，就能以相当高的精度恢复边。我们通过大量的实验证明了我们的图恢复攻击的有效性和适用性。



## **36. Arbitrarily Varying Wiretap Channels with Non-Causal Side Information at the Jammer**

干扰机具有非因果边信息的任意变化的窃听信道 cs.IT

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2001.03035v4)

**Authors**: Carsten Rudolf Janda, Moritz Wiese, Eduard A. Jorswieck, Holger Boche

**Abstracts**: Secure communication in a potentially malicious environment becomes more and more important. The arbitrarily varying wiretap channel (AVWC) provides information theoretical bounds on how much information can be exchanged even in the presence of an active attacker. If the active attacker has non-causal side information, situations in which a legitimate communication system has been hacked, can be modeled. We investigate the AVWC with non-causal side information at the jammer for the case that there exists a best channel to the eavesdropper. Non-causal side information means that the transmitted codeword is known to an active adversary before it is transmitted. By considering the maximum error criterion, we allow also messages to be known at the jammer before the corresponding codeword is transmitted. A single letter formula for the common randomness secrecy capacity is derived. Additionally, we provide a single letter formula for the common randomness secrecy capacity, for the cases that the channel to the eavesdropper is strongly degraded, strongly noisier, or strongly less capable with respect to the main channel. Furthermore, we compare our results to the random code secrecy capacity for the cases of maximum error criterion but without non-causal side information at the jammer, maximum error criterion with non-causal side information of the messages at the jammer, and the case of average error criterion without non-causal side information at the jammer.

摘要: 在潜在的恶意环境中进行安全通信变得越来越重要。任意变化的窃听通道(AVWC)提供了信息理论界限，即即使在活动攻击者在场的情况下也可以交换多少信息。如果主动攻击者具有非因果的辅助信息，则可以模拟合法通信系统被黑客攻击的情况。在干扰机存在最佳通道的情况下，我们研究了干扰机侧信息为非因果的AVWC。非因果辅助信息意味着所传输的码字在被传输之前为活跃的敌手所知。通过考虑最大差错准则，我们还允许在发送相应码字之前在干扰器处知道消息。给出了常见随机性保密容量的单字母公式。此外，对于到窃听者的信道相对于主信道是强退化、强噪声或强弱能力的情况，我们还给出了公共随机性保密容量的单字母公式。此外，我们还将我们的结果与干扰机无非因果边信息的最大差错准则、干扰机有非因果边信息的最大差错准则以及干扰机无非因果边信息的平均差错准则的情况下的随机码保密容量进行了比较。



## **37. Improving Adversarial Transferability with Gradient Refining**

利用梯度精化提高对手的可转移性 cs.CV

Accepted at CVPR 2021 Workshop on Adversarial Machine Learning in  Real-World Computer Vision Systems and Online Challenges. The extension  vision of this paper, please refer to arxiv:2203.13479

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2105.04834v3)

**Authors**: Guoqiu Wang, Huanqian Yan, Ying Guo, Xingxing Wei

**Abstracts**: Deep neural networks are vulnerable to adversarial examples, which are crafted by adding human-imperceptible perturbations to original images. Most existing adversarial attack methods achieve nearly 100% attack success rates under the white-box setting, but only achieve relatively low attack success rates under the black-box setting. To improve the transferability of adversarial examples for the black-box setting, several methods have been proposed, e.g., input diversity, translation-invariant attack, and momentum-based attack. In this paper, we propose a method named Gradient Refining, which can further improve the adversarial transferability by correcting useless gradients introduced by input diversity through multiple transformations. Our method is generally applicable to many gradient-based attack methods combined with input diversity. Extensive experiments are conducted on the ImageNet dataset and our method can achieve an average transfer success rate of 82.07% for three different models under single-model setting, which outperforms the other state-of-the-art methods by a large margin of 6.0% averagely. And we have applied the proposed method to the competition CVPR 2021 Unrestricted Adversarial Attacks on ImageNet organized by Alibaba and won the second place in attack success rates among 1558 teams.

摘要: 深度神经网络很容易受到敌意示例的攻击，这些示例是通过在原始图像中添加人类无法察觉的扰动来构建的。现有的对抗性攻击方法大多在白盒环境下攻击成功率接近100%，而在黑盒环境下攻击成功率相对较低。为了提高黑盒环境下敌意例子的可转移性，人们提出了输入多样性、平移不变攻击和动量攻击等方法。本文提出了一种梯度精化方法，通过多次变换修正输入分集引入的无用梯度，进一步提高了算法的对抗性可转移性。该方法普遍适用于多种结合输入分集的基于梯度的攻击方法。在ImageNet数据集上进行了大量的实验，在单一模型设置下，我们的方法对三种不同模型的平均传输成功率达到了82.07%，比其他最先进的方法平均提高了6.0%。并将提出的方法应用于阿里巴巴组织的CVPR 2021无限对抗性攻击ImageNet比赛中，获得了1558支球队攻击成功率的第二名。



## **38. Real-time Adversarial Perturbations against Deep Reinforcement Learning Policies: Attacks and Defenses**

深度强化学习策略的实时对抗性扰动：攻击与防御 cs.LG

13 pages, 6 figures

**SubmitDate**: 2022-04-14    [paper-pdf](http://arxiv.org/pdf/2106.08746v3)

**Authors**: Buse G. A. Tekgul, Shelly Wang, Samuel Marchal, N. Asokan

**Abstracts**: Recent work has shown that deep reinforcement learning (DRL) policies are vulnerable to adversarial perturbations. Adversaries can mislead policies of DRL agents by perturbing the state of the environment observed by the agents. Existing attacks are feasible in principle but face challenges in practice, either by being too slow to fool DRL policies in real time or by modifying past observations stored in the agent's memory. We show that using the Universal Adversarial Perturbation (UAP) method to compute perturbations, independent of the individual inputs to which they are applied to, can fool DRL policies effectively and in real time. We describe three such attack variants. Via an extensive evaluation using three Atari 2600 games, we show that our attacks are effective, as they fully degrade the performance of three different DRL agents (up to 100%, even when the $l_\infty$ bound on the perturbation is as small as 0.01). It is faster compared to the response time (0.6ms on average) of different DRL policies, and considerably faster than prior attacks using adversarial perturbations (1.8ms on average). We also show that our attack technique is efficient, incurring an online computational cost of 0.027ms on average. Using two further tasks involving robotic movement, we confirm that our results generalize to more complex DRL tasks. Furthermore, we demonstrate that the effectiveness of known defenses diminishes against universal perturbations. We propose an effective technique that detects all known adversarial perturbations against DRL policies, including all the universal perturbations presented in this paper.

摘要: 最近的工作表明，深度强化学习(DRL)策略容易受到对抗性扰动的影响。攻击者可以通过扰乱代理观察到的环境状态来误导DRL代理的策略。现有的攻击在原则上是可行的，但在实践中面临挑战，要么太慢，无法实时愚弄DRL策略，要么修改存储在代理内存中的过去观察。我们表明，使用通用对抗扰动(UAP)方法来计算扰动，而与应用扰动的个体输入无关，可以有效且实时地愚弄DRL策略。我们描述了三种这样的攻击变体。通过使用三款Atari 2600游戏进行的广泛评估，我们表明我们的攻击是有效的，因为它们完全降低了三种不同DRL代理的性能(高达100%，即使在扰动上的$l_\infty$约束小到0.01)。它比不同DRL策略的响应时间(平均0.6ms)更快，并且比以前使用对抗性扰动的攻击(平均1.8ms)要快得多。我们还证明了我们的攻击技术是有效的，平均在线计算成本为0.027ms。使用另外两个涉及机器人移动的任务，我们确认我们的结果推广到更复杂的DRL任务。此外，我们还证明了已知防御措施对普遍扰动的有效性会降低。我们提出了一种有效的技术来检测所有已知的针对DRL策略的对抗性扰动，包括本文提出的所有通用扰动。



## **39. Overparameterized Linear Regression under Adversarial Attacks**

对抗性攻击下的超参数线性回归 stat.ML

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2204.06274v1)

**Authors**: Antônio H. Ribeiro, Thomas B. Schön

**Abstracts**: As machine learning models start to be used in critical applications, their vulnerabilities and brittleness become a pressing concern. Adversarial attacks are a popular framework for studying these vulnerabilities. In this work, we study the error of linear regression in the face of adversarial attacks. We provide bounds of the error in terms of the traditional risk and the parameter norm and show how these bounds can be leveraged and make it possible to use analysis from non-adversarial setups to study the adversarial risk. The usefulness of these results is illustrated by shedding light on whether or not overparameterized linear models can be adversarially robust. We show that adding features to linear models might be either a source of additional robustness or brittleness. We show that these differences appear due to scaling and how the $\ell_1$ and $\ell_2$ norms of random projections concentrate. We also show how the reformulation we propose allows for solving adversarial training as a convex optimization problem. This is then used as a tool to study how adversarial training and other regularization methods might affect the robustness of the estimated models.

摘要: 随着机器学习模型开始在关键应用中使用，它们的脆弱性和脆性成为一个紧迫的问题。对抗性攻击是研究这些漏洞的流行框架。在这项工作中，我们研究了线性回归在面对对手攻击时的误差。我们给出了关于传统风险和参数范数的误差的界，并说明了如何利用这些界来利用这些界，使得从非对抗性设置的分析来研究对抗性风险成为可能。这些结果的有用之处在于揭示了过度参数化线性模型是否具有相反的稳健性。我们表明，向线性模型添加特征可能是额外的稳健性或脆性的来源。我们证明了这些差异是由于尺度以及随机投影的$\ell_1$和$\ell_2$范数是如何集中的。我们还展示了我们提出的重新公式如何允许将对抗性训练作为一个凸优化问题来解决。然后将其用作研究对抗性训练和其他正则化方法如何影响估计模型的稳健性的工具。



## **40. Towards A Critical Evaluation of Robustness for Deep Learning Backdoor Countermeasures**

深度学习后门对策的稳健性评测 cs.CR

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2204.06273v1)

**Authors**: Huming Qiu, Hua Ma, Zhi Zhang, Alsharif Abuadbba, Wei Kang, Anmin Fu, Yansong Gao

**Abstracts**: Since Deep Learning (DL) backdoor attacks have been revealed as one of the most insidious adversarial attacks, a number of countermeasures have been developed with certain assumptions defined in their respective threat models. However, the robustness of these countermeasures is inadvertently ignored, which can introduce severe consequences, e.g., a countermeasure can be misused and result in a false implication of backdoor detection.   For the first time, we critically examine the robustness of existing backdoor countermeasures with an initial focus on three influential model-inspection ones that are Neural Cleanse (S&P'19), ABS (CCS'19), and MNTD (S&P'21). Although the three countermeasures claim that they work well under their respective threat models, they have inherent unexplored non-robust cases depending on factors such as given tasks, model architectures, datasets, and defense hyper-parameter, which are \textit{not even rooted from delicate adaptive attacks}. We demonstrate how to trivially bypass them aligned with their respective threat models by simply varying aforementioned factors. Particularly, for each defense, formal proofs or empirical studies are used to reveal its two non-robust cases where it is not as robust as it claims or expects, especially the recent MNTD. This work highlights the necessity of thoroughly evaluating the robustness of backdoor countermeasures to avoid their misleading security implications in unknown non-robust cases.

摘要: 由于深度学习(DL)后门攻击已被发现是最隐蔽的敌意攻击之一，因此已经开发了一些对策，并在各自的威胁模型中定义了某些假设。然而，这些对策的稳健性被无意中忽视了，这可能会带来严重的后果，例如，对策可能被误用，并导致后门检测的错误含义。我们首次批判性地检验了现有后门对策的稳健性，最初集中在三个有影响力的模型检查对策上，它们是神经清洗(S&P‘19)、ABS(CCS’19)和MNTD(S&P‘21)。虽然这三种对策声称它们在各自的威胁模型下都工作得很好，但它们固有的非稳健性情况取决于给定的任务、模型体系结构、数据集和防御超参数等因素，而这些因素甚至不是源于脆弱的自适应攻击}。我们将演示如何通过简单地改变上述因素来绕过它们，使其与各自的威胁模型保持一致。特别是，对于每一种辩护，形式证明或实证研究都被用来揭示它的两种不稳健的情况，其中它并不像它声称或期望的那样稳健，特别是最近的MNTD。这项工作强调了彻底评估后门对策的稳健性的必要性，以避免在未知的非稳健性情况下它们具有误导性的安全影响。



## **41. Towards Practical Robustness Analysis for DNNs based on PAC-Model Learning**

基于PAC模型学习的DNN实用健壮性分析 cs.LG

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2101.10102v2)

**Authors**: Renjue Li, Pengfei Yang, Cheng-Chao Huang, Youcheng Sun, Bai Xue, Lijun Zhang

**Abstracts**: To analyse local robustness properties of deep neural networks (DNNs), we present a practical framework from a model learning perspective. Based on black-box model learning with scenario optimisation, we abstract the local behaviour of a DNN via an affine model with the probably approximately correct (PAC) guarantee. From the learned model, we can infer the corresponding PAC-model robustness property. The innovation of our work is the integration of model learning into PAC robustness analysis: that is, we construct a PAC guarantee on the model level instead of sample distribution, which induces a more faithful and accurate robustness evaluation. This is in contrast to existing statistical methods without model learning. We implement our method in a prototypical tool named DeepPAC. As a black-box method, DeepPAC is scalable and efficient, especially when DNNs have complex structures or high-dimensional inputs. We extensively evaluate DeepPAC, with 4 baselines (using formal verification, statistical methods, testing and adversarial attack) and 20 DNN models across 3 datasets, including MNIST, CIFAR-10, and ImageNet. It is shown that DeepPAC outperforms the state-of-the-art statistical method PROVERO, and it achieves more practical robustness analysis than the formal verification tool ERAN. Also, its results are consistent with existing DNN testing work like DeepGini.

摘要: 为了分析深度神经网络(DNN)的局部稳健性，从模型学习的角度提出了一个实用的框架。基于场景优化的黑盒模型学习，我们通过仿射模型在可能近似正确(PAC)的保证下抽象DNN的局部行为。从学习的模型中，我们可以推断出相应的PAC模型的稳健性。我们工作的创新之处在于将模型学习融入到PAC稳健性分析中：即在模型级别而不是样本分布上构造PAC保证，从而得到更真实和准确的稳健性评估。这与没有模型学习的现有统计方法形成了鲜明对比。我们在一个原型工具DeepPAC中实现了我们的方法。作为一种黑盒方法，DeepPAC具有可扩展性和高效率，特别是当DNN具有复杂的结构或高维输入时。我们对DeepPAC进行了广泛的评估，使用了4个基线(使用正式验证、统计方法、测试和对抗性攻击)和20个DNN模型，涉及MNIST、CIFAR-10和ImageNet等3个数据集。结果表明，DeepPAC的性能优于目前最先进的统计方法PROVERO，并且比形式化验证工具ERAN实现了更实用的健壮性分析。此外，它的结果与DeepGini等现有的DNN测试工作是一致的。



## **42. Stealing Malware Classifiers and AVs at Low False Positive Conditions**

在低误报条件下窃取恶意软件分类器和反病毒软件 cs.CR

12 pages, 8 figures, 6 tables. Under review

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2204.06241v1)

**Authors**: Maria Rigaki, Sebastian Garcia

**Abstracts**: Model stealing attacks have been successfully used in many machine learning domains, but there is little understanding of how these attacks work in the malware detection domain. Malware detection and, in general, security domains have very strong requirements of low false positive rates (FPR). However, these requirements are not the primary focus of the existing model stealing literature. Stealing attacks create surrogate models that perform similarly to a target model using a limited amount of queries to the target. The first stage of this study is the evaluation of active learning model stealing attacks against publicly available stand-alone machine learning malware classifiers and antivirus products (AVs). We propose a new neural network architecture for surrogate models that outperforms the existing state of the art on low FPR conditions. The surrogates were evaluated on their agreement with the targeted models. Good surrogates of the stand-alone classifiers were created with up to 99% agreement with the target models, using less than 4% of the original training dataset size. Good AV surrogates were also possible to train, but with a lower agreement. The second stage used the best surrogates as well as the target models to generate adversarial malware using the MAB framework to test stand-alone models and AVs (offline and online). Results showed that surrogate models could generate adversarial samples that evade the targets but are less successful than the targets themselves. Using surrogates, however, is a necessity for attackers, given that attacks against AVs are extremely time-consuming and easily detected when the AVs are connected to the internet.

摘要: 模型窃取攻击已经成功地应用于许多机器学习领域，但对于这些攻击在恶意软件检测领域的工作原理却知之甚少。通常，恶意软件检测和安全域对低误报比率(FPR)有非常强烈的要求。然而，这些要求并不是现有窃取文献模型的主要关注点。窃取攻击创建代理模型，该代理模型使用对目标的有限数量的查询来执行类似于目标模型的操作。本研究的第一阶段是评估针对公开可用的独立机器学习恶意软件分类器和反病毒产品(AV)的主动学习模型窃取攻击。我们提出了一种新的神经网络体系结构，用于代理模型，在低FPR条件下性能优于现有技术。根据其与目标模型的一致性对代用品进行评估。使用不到原始训练数据集大小的4%，创建了独立分类器的良好代理，与目标模型的一致性高达99%。好的反病毒代言人也有可能接受培训，但协议的一致性较低。第二阶段使用最好的代理以及目标模型来生成敌意恶意软件，使用MAB框架测试独立模型和AVs(离线和在线)。结果表明，代理模型可以生成避开目标的对抗性样本，但不如目标本身那么成功。然而，使用代理对于攻击者来说是必要的，因为针对AVs的攻击非常耗时，并且在AVs连接到互联网时很容易被检测到。



## **43. Liuer Mihou: A Practical Framework for Generating and Evaluating Grey-box Adversarial Attacks against NIDS**

六二密侯：一种实用的生成和评估针对NIDS的灰盒攻击的框架 cs.CR

16 pages, 8 figures, planning on submitting to ACM CCS 2022

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2204.06113v1)

**Authors**: Ke He, Dan Dongseong Kim, Jing Sun, Jeong Do Yoo, Young Hun Lee, Huy Kang Kim

**Abstracts**: Due to its high expressiveness and speed, Deep Learning (DL) has become an increasingly popular choice as the detection algorithm for Network-based Intrusion Detection Systems (NIDSes). Unfortunately, DL algorithms are vulnerable to adversarial examples that inject imperceptible modifications to the input and cause the DL algorithm to misclassify the input. Existing adversarial attacks in the NIDS domain often manipulate the traffic features directly, which hold no practical significance because traffic features cannot be replayed in a real network. It remains a research challenge to generate practical and evasive adversarial attacks.   This paper presents the Liuer Mihou attack that generates practical and replayable adversarial network packets that can bypass anomaly-based NIDS deployed in the Internet of Things (IoT) networks. The core idea behind Liuer Mihou is to exploit adversarial transferability and generate adversarial packets on a surrogate NIDS constrained by predefined mutation operations to ensure practicality. We objectively analyse the evasiveness of Liuer Mihou against four ML-based algorithms (LOF, OCSVM, RRCF, and SOM) and the state-of-the-art NIDS, Kitsune. From the results of our experiment, we gain valuable insights into necessary conditions on the adversarial transferability of anomaly detection algorithms. Going beyond a theoretical setting, we replay the adversarial attack in a real IoT testbed to examine the practicality of Liuer Mihou. Furthermore, we demonstrate that existing feature-level adversarial defence cannot defend against Liuer Mihou and constructively criticise the limitations of feature-level adversarial defences.

摘要: 深度学习以其高的表现力和速度成为基于网络的入侵检测系统(NIDSS)的检测算法之一。不幸的是，DL算法容易受到敌意示例的攻击，这些示例向输入注入难以察觉的修改，并导致DL算法错误地对输入进行分类。现有的NIDS域中的对抗性攻击往往直接操纵流量特征，由于流量特征不能在真实网络中再现，因此没有实际意义。产生实用的和躲避的对抗性攻击仍然是一个研究挑战。针对物联网网络中部署的基于异常的网络入侵检测系统，提出了一种能够生成实用的、可重放的敌意网络数据包的六二米侯攻击方法。六儿后的核心思想是利用敌意可转移性，在预定义的变异操作约束下的代理网络入侵检测系统上生成对抗性的报文，以确保实用性。我们客观地分析了六儿密侯对四种基于ML的算法(LOF、OCSVM、RRCF和SOM)以及最新的网络入侵检测系统Kitsune的规避能力。从实验结果中，我们对异常检测算法的对抗性转移的必要条件得到了有价值的见解。超越理论设置，我们在真实的IoT试验台上重播对抗性攻击，以检验六儿后的实用性。此外，我们证明了现有的特征级对抗性防御不能抵抗六二密侯，并建设性地批评了特征级对抗性防御的局限性。



## **44. Optimal Membership Inference Bounds for Adaptive Composition of Sampled Gaussian Mechanisms**

采样高斯机构自适应组合的最优成员推理界 cs.CR

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2204.06106v1)

**Authors**: Saeed Mahloujifar, Alexandre Sablayrolles, Graham Cormode, Somesh Jha

**Abstracts**: Given a trained model and a data sample, membership-inference (MI) attacks predict whether the sample was in the model's training set. A common countermeasure against MI attacks is to utilize differential privacy (DP) during model training to mask the presence of individual examples. While this use of DP is a principled approach to limit the efficacy of MI attacks, there is a gap between the bounds provided by DP and the empirical performance of MI attacks. In this paper, we derive bounds for the \textit{advantage} of an adversary mounting a MI attack, and demonstrate tightness for the widely-used Gaussian mechanism. We further show bounds on the \textit{confidence} of MI attacks. Our bounds are much stronger than those obtained by DP analysis. For example, analyzing a setting of DP-SGD with $\epsilon=4$ would obtain an upper bound on the advantage of $\approx0.36$ based on our analyses, while getting bound of $\approx 0.97$ using the analysis of previous work that convert $\epsilon$ to membership inference bounds.   Finally, using our analysis, we provide MI metrics for models trained on CIFAR10 dataset. To the best of our knowledge, our analysis provides the state-of-the-art membership inference bounds for the privacy.

摘要: 在给定训练模型和数据样本的情况下，成员推理(MI)攻击预测样本是否在模型的训练集中。针对MI攻击的一种常见对策是在模型训练期间利用差异隐私(DP)来掩盖单个示例的存在。虽然使用DP是一种原则性的方法来限制MI攻击的有效性，但是DP提供的界限和MI攻击的经验性能之间存在差距。在这篇文章中，我们得到了敌手发起MI攻击的优势的界，并证明了广泛使用的Gauss机制的紧性。我们进一步给出了MI攻击的置信度的界。我们的边界比DP分析得到的边界要强得多。例如，根据我们的分析，分析具有$\epsilon=4$的DP-SGD的设置将得到$\约0.36$的优势的上界，而使用将$\epsilon$转换为成员推理界的前人工作的分析，得到$\约0.97$的上界。最后，利用我们的分析，我们提供了在CIFAR10数据集上训练的模型的MI度量。据我们所知，我们的分析为隐私提供了最先进的成员关系推断界限。



## **45. Membership Inference Attacks From First Principles**

从第一性原理出发的成员推理攻击 cs.CR

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2112.03570v2)

**Authors**: Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramer

**Abstracts**: A membership inference attack allows an adversary to query a trained machine learning model to predict whether or not a particular example was contained in the model's training dataset. These attacks are currently evaluated using average-case "accuracy" metrics that fail to characterize whether the attack can confidently identify any members of the training set. We argue that attacks should instead be evaluated by computing their true-positive rate at low (e.g., <0.1%) false-positive rates, and find most prior attacks perform poorly when evaluated in this way. To address this we develop a Likelihood Ratio Attack (LiRA) that carefully combines multiple ideas from the literature. Our attack is 10x more powerful at low false-positive rates, and also strictly dominates prior attacks on existing metrics.

摘要: 成员关系推理攻击允许对手查询经过训练的机器学习模型，以预测特定示例是否包含在该模型的训练数据集中。目前，这些攻击是使用平均案例“准确性”度量来评估的，该度量无法确定攻击是否可以自信地识别训练集的任何成员。我们认为，应该通过在较低的(例如<0.1%)假阳性率下计算真阳性率来评估攻击，并发现大多数先前的攻击在以这种方式评估时表现很差。为了解决这个问题，我们开发了一种似然比攻击(LIRA)，它仔细地结合了文献中的多种想法。我们的攻击在低假阳性率下的威力要高出10倍，而且还严格控制了之前对现有指标的攻击。



## **46. Rate Coding or Direct Coding: Which One is Better for Accurate, Robust, and Energy-efficient Spiking Neural Networks?**

速率编码和直接编码：哪种编码更适合准确、健壮和节能的尖峰神经网络？ cs.NE

Accepted to ICASSP2022

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2202.03133v2)

**Authors**: Youngeun Kim, Hyoungseob Park, Abhishek Moitra, Abhiroop Bhattacharjee, Yeshwanth Venkatesha, Priyadarshini Panda

**Abstracts**: Recent Spiking Neural Networks (SNNs) works focus on an image classification task, therefore various coding techniques have been proposed to convert an image into temporal binary spikes. Among them, rate coding and direct coding are regarded as prospective candidates for building a practical SNN system as they show state-of-the-art performance on large-scale datasets. Despite their usage, there is little attention to comparing these two coding schemes in a fair manner. In this paper, we conduct a comprehensive analysis of the two codings from three perspectives: accuracy, adversarial robustness, and energy-efficiency. First, we compare the performance of two coding techniques with various architectures and datasets. Then, we measure the robustness of the coding techniques on two adversarial attack methods. Finally, we compare the energy-efficiency of two coding schemes on a digital hardware platform. Our results show that direct coding can achieve better accuracy especially for a small number of timesteps. In contrast, rate coding shows better robustness to adversarial attacks owing to the non-differentiable spike generation process. Rate coding also yields higher energy-efficiency than direct coding which requires multi-bit precision for the first layer. Our study explores the characteristics of two codings, which is an important design consideration for building SNNs. The code is made available at https://github.com/Intelligent-Computing-Lab-Yale/Rate-vs-Direct.

摘要: 最近的尖峰神经网络(SNN)专注于图像分类任务，因此已经提出了各种编码技术来将图像转换为时间二进制尖峰。其中，码率编码和直接编码由于在大规模数据集上表现出最先进的性能，被认为是构建实用SNN系统的潜在候选者。尽管使用了这两种编码方案，但很少有人注意以公平的方式比较这两种编码方案。在本文中，我们从准确性、对手健壮性和能量效率三个角度对这两种编码进行了全面的分析。首先，我们比较了两种编码技术在不同架构和不同数据集下的性能。然后，我们测量了编码技术在两种对抗性攻击方法上的健壮性。最后，在数字硬件平台上对两种编码方案的能量效率进行了比较。我们的结果表明，直接编码可以获得更好的精度，特别是在少量时间步长的情况下。相反，由于不可微的尖峰产生过程，速率编码表现出更好的对对手攻击的稳健性。速率编码还产生比直接编码更高的能量效率，直接编码需要第一层的多比特精度。我们的研究探索了两种编码的特点，这是构建SNN的重要设计考虑因素。代码可在https://github.com/Intelligent-Computing-Lab-Yale/Rate-vs-Direct.上获得



## **47. Masked Faces with Faced Masks**

戴着面具的蒙面人 cs.CV

8 pages

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2201.06427v2)

**Authors**: Jiayi Zhu, Qing Guo, Felix Juefei-Xu, Yihao Huang, Yang Liu, Geguang Pu

**Abstracts**: Modern face recognition systems (FRS) still fall short when the subjects are wearing facial masks, a common theme in the age of respiratory pandemics. An intuitive partial remedy is to add a mask detector to flag any masked faces so that the FRS can act accordingly for those low-confidence masked faces. In this work, we set out to investigate the potential vulnerability of such FRS equipped with a mask detector, on large-scale masked faces, which might trigger a serious risk, e.g., letting a suspect evade the FRS where both facial identity and mask are undetected. As existing face recognizers and mask detectors have high performance in their respective tasks, it is significantly challenging to simultaneously fool them and preserve the transferability of the attack. We formulate the new task as the generation of realistic & adversarial-faced mask and make three main contributions: First, we study the naive Delanunay-based masking method (DM) to simulate the process of wearing a faced mask that is cropped from a template image, which reveals the main challenges of this new task. Second, we further equip the DM with the adversarial noise attack and propose the adversarial noise Delaunay-based masking method (AdvNoise-DM) that can fool the face recognition and mask detection effectively but make the face less natural. Third, we propose the adversarial filtering Delaunay-based masking method denoted as MF2M by employing the adversarial filtering for AdvNoise-DM and obtain more natural faces. With the above efforts, the final version not only leads to significant performance deterioration of the state-of-the-art (SOTA) deep learning-based FRS, but also remains undetected by the SOTA facial mask detector, thus successfully fooling both systems at the same time.

摘要: 当受试者戴着口罩时，现代人脸识别系统(FRS)仍然不足，这在呼吸道大流行的时代是一个常见的主题。一种直观的部分补救方法是添加一个掩模检测器来标记任何掩蔽的人脸，以便FRS可以对这些低置信度的掩蔽人脸采取相应的行动。在这项工作中，我们开始调查这种配备了面具检测器的FRS在大规模蒙面人脸上的潜在脆弱性，这可能会引发严重的风险，例如，让嫌疑人逃避FRS，其中面部身份和面具都没有被检测到。由于现有的人脸识别器和面具检测器在各自的任务中具有很高的性能，因此要同时欺骗它们并保持攻击的可转移性是非常具有挑战性的。首先，我们研究了基于朴素Delanunay的掩蔽方法(DM)来模拟从模板图像中裁剪出来的人脸面具的佩戴过程，揭示了这一新任务的主要挑战。其次，进一步为DM提供了对抗性噪声攻击，并提出了基于对抗性噪声Delaunay的掩蔽方法(AdvNoise-DM)，该方法可以有效地欺骗人脸识别和掩模检测，但会使人脸变得不自然。第三，针对AdvNoise-DM，提出了一种基于对抗过滤的Delaunay掩蔽方法MF2M，得到了更自然的人脸。在上述努力下，最终版本不仅导致基于最先进的(SOTA)深度学习的FRS的性能显著恶化，而且仍然没有被SOTA面膜检测器检测到，从而成功地同时愚弄了两个系统。



## **48. Automated Attacker Synthesis for Distributed Protocols**

分布式协议的自动攻击者综合 cs.CR

24 pages, 15 figures

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2004.01220v4)

**Authors**: Max von Hippel, Cole Vick, Stavros Tripakis, Cristina Nita-Rotaru

**Abstracts**: Distributed protocols should be robust to both benign malfunction (e.g. packet loss or delay) and attacks (e.g. message replay) from internal or external adversaries. In this paper we take a formal approach to the automated synthesis of attackers, i.e. adversarial processes that can cause the protocol to malfunction. Specifically, given a formal threat model capturing the distributed protocol model and network topology, as well as the placement, goals, and interface (inputs and outputs) of potential attackers, we automatically synthesize an attacker. We formalize four attacker synthesis problems - across attackers that always succeed versus those that sometimes fail, and attackers that attack forever versus those that do not - and we propose algorithmic solutions to two of them. We report on a prototype implementation called KORG and its application to TCP as a case-study. Our experiments show that KORG can automatically generate well-known attacks for TCP within seconds or minutes.

摘要: 分布式协议应该对来自内部或外部对手的良性故障(例如，分组丢失或延迟)和攻击(例如，消息重放)具有健壮性。在本文中，我们采用了一种形式化的方法来自动合成攻击者，即可能导致协议故障的对抗性过程。具体地说，给定一个捕获分布式协议模型和网络拓扑以及潜在攻击者的位置、目标和接口(输入和输出)的正式威胁模型，我们将自动合成攻击者。我们将四个攻击者合成问题形式化--总是成功的攻击者与有时失败的攻击者，以及永远攻击的攻击者与不成功的攻击者--并针对其中两个问题提出了算法解决方案。我们报告了一个称为KORG的原型实现以及它在TCP中的应用作为案例研究。我们的实验表明，KORG可以在几秒或几分钟内自动生成针对TCP的知名攻击。



## **49. Catch Me If You Can: Blackbox Adversarial Attacks on Automatic Speech Recognition using Frequency Masking**

抓住我：使用频率掩蔽对自动语音识别进行黑箱对抗性攻击 cs.SD

11 pages, 7 figures and 3 tables

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2112.01821v2)

**Authors**: Xiaoliang Wu, Ajitha Rajan

**Abstracts**: Automatic speech recognition (ASR) models are prevalent, particularly in applications for voice navigation and voice control of domestic appliances. The computational core of ASRs are deep neural networks (DNNs) that have been shown to be susceptible to adversarial perturbations; easily misused by attackers to generate malicious outputs. To help test the security and robustnesss of ASRS, we propose techniques that generate blackbox (agnostic to the DNN), untargeted adversarial attacks that are portable across ASRs. This is in contrast to existing work that focuses on whitebox targeted attacks that are time consuming and lack portability.   Our techniques generate adversarial attacks that have no human audible difference by manipulating the audio signal using a psychoacoustic model that maintains the audio perturbations below the thresholds of human perception. We evaluate portability and effectiveness of our techniques using three popular ASRs and two input audio datasets using the metrics - Word Error Rate (WER) of output transcription, Similarity to original audio, attack Success Rate on different ASRs and Detection score by a defense system. We found our adversarial attacks were portable across ASRs, not easily detected by a state-of-the-art defense system, and had significant difference in output transcriptions while sounding similar to original audio.

摘要: 自动语音识别(ASR)模型非常普遍，特别是在家用电器的语音导航和语音控制应用中。ASR的计算核心是深度神经网络(DNN)，已被证明容易受到对手扰动的影响；很容易被攻击者误用来生成恶意输出。为了帮助测试ASR的安全性和健壮性，我们提出了生成黑盒(与DNN无关)的技术，这是一种可跨ASR移植的无目标对抗性攻击。这与现有的专注于白盒目标攻击的工作形成了鲜明对比，这些攻击既耗时又缺乏可移植性。我们的技术通过使用将音频扰动保持在人类感知阈值以下的心理声学模型来操纵音频信号，从而产生没有人类听觉差异的对抗性攻击。我们使用三个流行的ASR和两个输入音频数据集，使用输出转录的错误率(WER)、与原始音频的相似性、对不同ASR的攻击成功率和防御系统的检测分数来评估我们的技术的可移植性和有效性。我们发现，我们的敌意攻击可以跨ASR进行移植，不容易被最先进的防御系统检测到，而且在输出转录方面有显著差异，但听起来与原始音频相似。



## **50. Staircase Sign Method for Boosting Adversarial Attacks**

一种加强对抗性攻击的阶梯标记法 cs.CV

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2104.09722v2)

**Authors**: Qilong Zhang, Xiaosu Zhu, Jingkuan Song, Lianli Gao, Heng Tao Shen

**Abstracts**: Crafting adversarial examples for the transfer-based attack is challenging and remains a research hot spot. Currently, such attack methods are based on the hypothesis that the substitute model and the victim model learn similar decision boundaries, and they conventionally apply Sign Method (SM) to manipulate the gradient as the resultant perturbation. Although SM is efficient, it only extracts the sign of gradient units but ignores their value difference, which inevitably leads to a deviation. Therefore, we propose a novel Staircase Sign Method (S$^2$M) to alleviate this issue, thus boosting attacks. Technically, our method heuristically divides the gradient sign into several segments according to the values of the gradient units, and then assigns each segment with a staircase weight for better crafting adversarial perturbation. As a result, our adversarial examples perform better in both white-box and black-box manner without being more visible. Since S$^2$M just manipulates the resultant gradient, our method can be generally integrated into the family of FGSM algorithms, and the computational overhead is negligible. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our proposed methods, which significantly improve the transferability (i.e., on average, \textbf{5.1\%} for normally trained models and \textbf{12.8\%} for adversarially trained defenses). Our code is available at \url{https://github.com/qilong-zhang/Staircase-sign-method}.

摘要: 为基于转移的攻击制作敌意例子是具有挑战性的，也是一个研究热点。目前，这种攻击方法是基于替换模型和受害者模型学习相似的决策边界的假设，并且它们通常应用符号方法(SM)来操纵梯度作为结果扰动。虽然SM方法是有效的，但它只提取梯度单元的符号，而忽略了它们的值差异，这不可避免地会导致偏差。因此，我们提出了一种新的阶梯符号方法(S$^2$M)来缓解这个问题，从而增强了攻击。从技术上讲，我们的方法根据梯度单元的值启发式地将梯度符号分成几个段，然后为每个段分配阶梯权重，以便更好地制作对抗扰动。因此，我们的对抗性例子在白盒和黑盒方式下都表现得更好，而不是更明显。由于S$^2$M只是对合成的梯度进行操作，因此我们的方法一般可以集成到FGSM算法家族中，并且计算开销可以忽略不计。在ImageNet数据集上的大量实验表明，我们提出的方法是有效的，显著提高了可转移性(即，对于正常训练的模型，平均为extbf{5.1}，对于经过相反训练的防御，平均为\extbf{12.8})。我们的代码可以在\url{https://github.com/qilong-zhang/Staircase-sign-method}.上找到



