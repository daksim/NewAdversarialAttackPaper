# Latest Adversarial Attack Papers
**update at 2022-10-22 06:31:24**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Similarity of Neural Architectures Based on Input Gradient Transferability**

基于输入梯度传递的神经结构相似性研究 cs.LG

21pages, 10 figures, 1.5MB

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11407v1) [paper-pdf](http://arxiv.org/pdf/2210.11407v1)

**Authors**: Jaehui Hwang, Dongyoon Han, Byeongho Heo, Song Park, Sanghyuk Chun, Jong-Seok Lee

**Abstract**: In this paper, we aim to design a quantitative similarity function between two neural architectures. Specifically, we define a model similarity using input gradient transferability. We generate adversarial samples of two networks and measure the average accuracy of the networks on adversarial samples of each other. If two networks are highly correlated, then the attack transferability will be high, resulting in high similarity. Using the similarity score, we investigate two topics: (1) Which network component contributes to the model diversity? (2) How does model diversity affect practical scenarios? We answer the first question by providing feature importance analysis and clustering analysis. The second question is validated by two different scenarios: model ensemble and knowledge distillation. Our findings show that model diversity takes a key role when interacting with different neural architectures. For example, we found that more diversity leads to better ensemble performance. We also observe that the relationship between teacher and student networks and distillation performance depends on the choice of the base architecture of the teacher and student networks. We expect our analysis tool helps a high-level understanding of differences between various neural architectures as well as practical guidance when using multiple architectures.

摘要: 在本文中，我们的目标是设计两个神经结构之间的定量相似性函数。具体地说，我们定义了一种使用输入梯度可转移性的模型相似性。我们生成两个网络的对抗性样本，并在对方的对抗性样本上测量网络的平均准确率。如果两个网络高度相关，那么攻击的可传递性就高，从而导致高相似度。使用相似性得分，我们研究了两个主题：(1)哪个网络组件对模型多样性有贡献？(2)模型多样性如何影响实际场景？我们通过提供特征重要性分析和聚类分析来回答第一个问题。第二个问题通过两个不同的场景进行验证：模型集成和知识提炼。我们的发现表明，在与不同的神经结构相互作用时，模型多样性起着关键作用。例如，我们发现，更多的多样性会带来更好的合奏表现。我们还观察到，教师和学生网络与蒸馏性能之间的关系取决于教师和学生网络基础架构的选择。我们希望我们的分析工具能够帮助我们更高层次地了解不同神经架构之间的差异，并在使用多种架构时提供实用指导。



## **2. Surprises in adversarially-trained linear regression**

逆训练线性回归中的惊喜 stat.ML

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2205.12695v2) [paper-pdf](http://arxiv.org/pdf/2205.12695v2)

**Authors**: Antônio H. Ribeiro, Dave Zachariah, Thomas B. Schön

**Abstract**: State-of-the-art machine learning models can be vulnerable to very small input perturbations that are adversarially constructed. Adversarial training is an effective approach to defend against such examples. It is formulated as a min-max problem, searching for the best solution when the training data was corrupted by the worst-case attacks. For linear regression problems, adversarial training can be formulated as a convex problem. We use this reformulation to make two technical contributions: First, we formulate the training problem as an instance of robust regression to reveal its connection to parameter-shrinking methods, specifically that $\ell_\infty$-adversarial training produces sparse solutions. Secondly, we study adversarial training in the overparameterized regime, i.e. when there are more parameters than data. We prove that adversarial training with small disturbances gives the solution with the minimum-norm that interpolates the training data. Ridge regression and lasso approximate such interpolating solutions as their regularization parameter vanishes. By contrast, for adversarial training, the transition into the interpolation regime is abrupt and for non-zero values of disturbance. This result is proved and illustrated with numerical examples.

摘要: 最先进的机器学习模型很容易受到相反构造的非常小的输入扰动的影响。对抗性训练是抵御这类例子的一种有效方法。它被描述为一个最小-最大问题，当训练数据被最坏情况的攻击破坏时，搜索最优解。对于线性回归问题，对抗性训练可以表示为一个凸问题。首先，我们将训练问题描述为稳健回归的一个实例，以揭示它与参数收缩方法的联系，具体地说，对手训练产生稀疏解。其次，我们研究了参数多于数据的过度参数条件下的对抗性训练。我们证明了小干扰下的对抗性训练给出了用最小范数对训练数据进行内插的解。岭回归和套索逼近这种插值解，因为它们的正则化参数为零。相比之下，对于对抗性训练，向插补机制的转变是突然的，并且对于非零值的扰动。这一结果得到了证明，并用数值算例加以说明。



## **3. Attacking Motion Estimation with Adversarial Snow**

对抗性降雪下的攻击运动估计 cs.CV

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11242v1) [paper-pdf](http://arxiv.org/pdf/2210.11242v1)

**Authors**: Jenny Schmalfuss, Lukas Mehl, Andrés Bruhn

**Abstract**: Current adversarial attacks for motion estimation (optical flow) optimize small per-pixel perturbations, which are unlikely to appear in the real world. In contrast, we exploit a real-world weather phenomenon for a novel attack with adversarially optimized snow. At the core of our attack is a differentiable renderer that consistently integrates photorealistic snowflakes with realistic motion into the 3D scene. Through optimization we obtain adversarial snow that significantly impacts the optical flow while being indistinguishable from ordinary snow. Surprisingly, the impact of our novel attack is largest on methods that previously showed a high robustness to small L_p perturbations.

摘要: 当前针对运动估计(光流)的敌意攻击优化了每像素的微小扰动，这在现实世界中不太可能出现。相反，我们利用真实世界的天气现象，用相反的优化降雪进行了一次新颖的攻击。我们攻击的核心是一个可区分的渲染器，它一致地将照片级逼真的雪花和逼真的运动整合到3D场景中。通过优化，我们得到了对抗性雪，它对光流有明显的影响，但与普通雪没有什么区别。令人惊讶的是，我们的新攻击对以前对小Lp扰动表现出高度稳健性的方法的影响最大。



## **4. UKP-SQuARE v2: Explainability and Adversarial Attacks for Trustworthy QA**

UKP-Square v2：可解析性和针对可信QA的对抗性攻击 cs.CL

Accepted at AACL 2022 as Demo Paper

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2208.09316v3) [paper-pdf](http://arxiv.org/pdf/2208.09316v3)

**Authors**: Rachneet Sachdeva, Haritz Puerto, Tim Baumgärtner, Sewin Tariverdian, Hao Zhang, Kexin Wang, Hossain Shaikh Saadi, Leonardo F. R. Ribeiro, Iryna Gurevych

**Abstract**: Question Answering (QA) systems are increasingly deployed in applications where they support real-world decisions. However, state-of-the-art models rely on deep neural networks, which are difficult to interpret by humans. Inherently interpretable models or post hoc explainability methods can help users to comprehend how a model arrives at its prediction and, if successful, increase their trust in the system. Furthermore, researchers can leverage these insights to develop new methods that are more accurate and less biased. In this paper, we introduce SQuARE v2, the new version of SQuARE, to provide an explainability infrastructure for comparing models based on methods such as saliency maps and graph-based explanations. While saliency maps are useful to inspect the importance of each input token for the model's prediction, graph-based explanations from external Knowledge Graphs enable the users to verify the reasoning behind the model prediction. In addition, we provide multiple adversarial attacks to compare the robustness of QA models. With these explainability methods and adversarial attacks, we aim to ease the research on trustworthy QA models. SQuARE is available on https://square.ukp-lab.de.

摘要: 问答(QA)系统越来越多地部署在支持现实世界决策的应用程序中。然而，最先进的模型依赖于深度神经网络，这很难被人类解释。本质上可解释的模型或事后可解释的方法可以帮助用户理解模型如何达到其预测，如果成功，则增加他们对系统的信任。此外，研究人员可以利用这些洞察力来开发更准确、更少偏见的新方法。在本文中，我们引入了Square的新版本Square v2，以提供基于显著图和基于图的解释等方法的模型比较的可解释性基础设施。虽然显著图有助于检查每个输入标记对于模型预测的重要性，但来自外部知识图的基于图形的解释使用户能够验证模型预测背后的推理。此外，我们还提供了多个对抗性攻击来比较QA模型的健壮性。通过这些可解释性方法和对抗性攻击，我们的目标是简化可信QA模型的研究。Square在https://square.ukp-lab.de.上可用



## **5. Analyzing the Robustness of Decentralized Horizontal and Vertical Federated Learning Architectures in a Non-IID Scenario**

非IID场景下分布式水平和垂直联合学习体系结构的健壮性分析 cs.LG

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11061v1) [paper-pdf](http://arxiv.org/pdf/2210.11061v1)

**Authors**: Pedro Miguel Sánchez Sánchez, Alberto Huertas Celdrán, Enrique Tomás Martínez Beltrán, Daniel Demeter, Gérôme Bovet, Gregorio Martínez Pérez, Burkhard Stiller

**Abstract**: Federated learning (FL) allows participants to collaboratively train machine and deep learning models while protecting data privacy. However, the FL paradigm still presents drawbacks affecting its trustworthiness since malicious participants could launch adversarial attacks against the training process. Related work has studied the robustness of horizontal FL scenarios under different attacks. However, there is a lack of work evaluating the robustness of decentralized vertical FL and comparing it with horizontal FL architectures affected by adversarial attacks. Thus, this work proposes three decentralized FL architectures, one for horizontal and two for vertical scenarios, namely HoriChain, VertiChain, and VertiComb. These architectures present different neural networks and training protocols suitable for horizontal and vertical scenarios. Then, a decentralized, privacy-preserving, and federated use case with non-IID data to classify handwritten digits is deployed to evaluate the performance of the three architectures. Finally, a set of experiments computes and compares the robustness of the proposed architectures when they are affected by different data poisoning based on image watermarks and gradient poisoning adversarial attacks. The experiments show that even though particular configurations of both attacks can destroy the classification performance of the architectures, HoriChain is the most robust one.

摘要: 联合学习(FL)允许参与者协作训练机器和深度学习模型，同时保护数据隐私。然而，FL范式仍然存在影响其可信度的缺陷，因为恶意参与者可能会对培训过程发动对抗性攻击。相关工作研究了水平FL场景在不同攻击下的稳健性。然而，缺乏对分布式垂直FL的健壮性进行评估，并将其与水平FL体系结构进行比较的工作。因此，本文提出了三种去中心化FL架构，一种用于水平场景，两种用于垂直场景，即HoriChain、VertiChain和VertiComb。这些体系结构提供了适用于水平和垂直场景的不同神经网络和训练协议。然后，部署了一个分散的、保护隐私的联合用例，使用非IID数据对手写数字进行分类，以评估三种体系结构的性能。最后，通过一组实验计算并比较了基于图像水印的数据中毒和基于梯度中毒的敌意攻击对所提体系结构的健壮性的影响。实验表明，尽管两种攻击的特定配置都会破坏体系结构的分类性能，但HoriChain是最健壮的一种。



## **6. Defending Person Detection Against Adversarial Patch Attack by using Universal Defensive Frame**

利用通用防御框架防御敌方补丁攻击的人检测 cs.CV

Accepted at IEEE Transactions on Image Processing (TIP), 2022

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2204.13004v2) [paper-pdf](http://arxiv.org/pdf/2204.13004v2)

**Authors**: Youngjoon Yu, Hong Joo Lee, Hakmin Lee, Yong Man Ro

**Abstract**: Person detection has attracted great attention in the computer vision area and is an imperative element in human-centric computer vision. Although the predictive performances of person detection networks have been improved dramatically, they are vulnerable to adversarial patch attacks. Changing the pixels in a restricted region can easily fool the person detection network in safety-critical applications such as autonomous driving and security systems. Despite the necessity of countering adversarial patch attacks, very few efforts have been dedicated to defending person detection against adversarial patch attack. In this paper, we propose a novel defense strategy that defends against an adversarial patch attack by optimizing a defensive frame for person detection. The defensive frame alleviates the effect of the adversarial patch while maintaining person detection performance with clean person. The proposed defensive frame in the person detection is generated with a competitive learning algorithm which makes an iterative competition between detection threatening module and detection shielding module in person detection. Comprehensive experimental results demonstrate that the proposed method effectively defends person detection against adversarial patch attacks.

摘要: 人体检测在计算机视觉领域引起了极大的关注，是以人为中心的计算机视觉的重要组成部分。虽然个人检测网络的预测性能有了很大的提高，但它们很容易受到对手补丁的攻击。在自动驾驶和安全系统等安全关键应用中，更改受限区域的像素很容易欺骗人员检测网络。尽管有必要对抗对抗性补丁攻击，但很少有人致力于防御对抗性补丁攻击的人检测。在本文中，我们提出了一种新的防御策略，通过优化个人检测的防御框架来防御对抗性补丁攻击。该防御框架在保持人与干净的人的检测性能的同时，减轻了对手补丁的影响。利用竞争学习算法生成人体检测中的防御帧，使得人体检测中的检测威胁模块和检测屏蔽模块之间进行迭代竞争。综合实验结果表明，该方法能够有效地防御敌意补丁攻击。



## **7. Towards Adversarial Attack on Vision-Language Pre-training Models**

视觉语言预训练模型的对抗性攻击 cs.LG

Accepted by ACM MM2022. Code is available in GitHub

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2206.09391v2) [paper-pdf](http://arxiv.org/pdf/2206.09391v2)

**Authors**: Jiaming Zhang, Qi Yi, Jitao Sang

**Abstract**: While vision-language pre-training model (VLP) has shown revolutionary improvements on various vision-language (V+L) tasks, the studies regarding its adversarial robustness remain largely unexplored. This paper studied the adversarial attack on popular VLP models and V+L tasks. First, we analyzed the performance of adversarial attacks under different settings. By examining the influence of different perturbed objects and attack targets, we concluded some key observations as guidance on both designing strong multimodal adversarial attack and constructing robust VLP models. Second, we proposed a novel multimodal attack method on the VLP models called Collaborative Multimodal Adversarial Attack (Co-Attack), which collectively carries out the attacks on the image modality and the text modality. Experimental results demonstrated that the proposed method achieves improved attack performances on different V+L downstream tasks and VLP models. The analysis observations and novel attack method hopefully provide new understanding into the adversarial robustness of VLP models, so as to contribute their safe and reliable deployment in more real-world scenarios. Code is available at https://github.com/adversarial-for-goodness/Co-Attack.

摘要: 虽然视觉-语言预训练模型(VLP)在各种视觉-语言(V+L)任务上有了革命性的改进，但关于其对抗健壮性的研究仍然很少。研究了对流行的VLP模型和V+L任务的对抗性攻击。首先，我们分析了不同环境下对抗性攻击的性能。通过考察不同扰动对象和攻击目标的影响，我们总结了一些关键的观察结果，作为设计强多通道对抗性攻击和构建稳健VLP模型的指导。其次，我们提出了一种新的针对VLP模型的多模式攻击方法，称为协作式多模式对抗攻击(Co-Attack)，它共同对图像通道和文本通道进行攻击。实验结果表明，该方法在不同的V+L下游任务和VLP模型下均能获得较好的攻击性能。分析观察和新颖的攻击方法有望对VLP模型的对抗健壮性提供新的理解，从而有助于在更真实的场景中安全可靠地部署VLP模型。代码可在https://github.com/adversarial-for-goodness/Co-Attack.上找到



## **8. Rewriting Meaningful Sentences via Conditional BERT Sampling and an application on fooling text classifiers**

基于条件BERT抽样的有意义句子重写及其在愚弄文本分类器上的应用 cs.CL

Please see an updated version of this paper at arXiv:2104.08453

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2010.11869v2) [paper-pdf](http://arxiv.org/pdf/2010.11869v2)

**Authors**: Lei Xu, Ivan Ramirez, Kalyan Veeramachaneni

**Abstract**: Most adversarial attack methods that are designed to deceive a text classifier change the text classifier's prediction by modifying a few words or characters. Few try to attack classifiers by rewriting a whole sentence, due to the difficulties inherent in sentence-level rephrasing as well as the problem of setting the criteria for legitimate rewriting.   In this paper, we explore the problem of creating adversarial examples with sentence-level rewriting. We design a new sampling method, named ParaphraseSampler, to efficiently rewrite the original sentence in multiple ways. Then we propose a new criteria for modification, called a sentence-level threaten model. This criteria allows for both word- and sentence-level changes, and can be adjusted independently in two dimensions: semantic similarity and grammatical quality. Experimental results show that many of these rewritten sentences are misclassified by the classifier. On all 6 datasets, our ParaphraseSampler achieves a better attack success rate than our baseline.

摘要: 大多数旨在欺骗文本分类器的对抗性攻击方法都是通过修改几个单词或字符来改变文本分类器的预测。很少有人试图通过重写整个句子来攻击量词，这是因为句子级重写固有的困难，以及为合法重写设定标准的问题。在本文中，我们探讨了使用句子级重写来创建对抗性实例的问题。我们设计了一种新的抽样方法，称为ParaphraseSsamer，以多种方式高效地重写原始句子。然后，我们提出了一种新的修改标准，称为语句级威胁模型。这一标准允许单词和句子级别的变化，并且可以在两个维度上独立调整：语义相似度和语法质量。实验结果表明，许多改写后的句子被分类器误分类。在所有6个数据集上，我们的ParaphraseSsamer实现了比我们的基线更高的攻击成功率。



## **9. R&R: Metric-guided Adversarial Sentence Generation**

R&R：度量制导的对抗性句子生成 cs.CL

Accepted to Finding of AACL-IJCNLP2022

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2104.08453v3) [paper-pdf](http://arxiv.org/pdf/2104.08453v3)

**Authors**: Lei Xu, Alfredo Cuesta-Infante, Laure Berti-Equille, Kalyan Veeramachaneni

**Abstract**: Adversarial examples are helpful for analyzing and improving the robustness of text classifiers. Generating high-quality adversarial examples is a challenging task as it requires generating fluent adversarial sentences that are semantically similar to the original sentences and preserve the original labels, while causing the classifier to misclassify them. Existing methods prioritize misclassification by maximizing each perturbation's effectiveness at misleading a text classifier; thus, the generated adversarial examples fall short in terms of fluency and similarity. In this paper, we propose a rewrite and rollback (R&R) framework for adversarial attack. It improves the quality of adversarial examples by optimizing a critique score which combines the fluency, similarity, and misclassification metrics. R&R generates high-quality adversarial examples by allowing exploration of perturbations that do not have immediate impact on the misclassification metric but can improve fluency and similarity metrics. We evaluate our method on 5 representative datasets and 3 classifier architectures. Our method outperforms current state-of-the-art in attack success rate by +16.2%, +12.8%, and +14.0% on the classifiers respectively. Code is available at https://github.com/DAI-Lab/fibber

摘要: 对抗性例子有助于分析和提高文本分类器的健壮性。生成高质量的对抗性例子是一项具有挑战性的任务，因为它需要生成流畅的对抗性句子，这些句子在语义上与原始句子相似，并保留原始标签，同时导致分类器对它们进行错误分类。现有的方法通过最大化每个扰动在误导文本分类器方面的有效性来区分错误分类的优先级；因此，生成的对抗性示例在流畅性和相似性方面存在不足。提出了一种用于对抗攻击的重写和回滚(R&R)框架。它通过优化结合了流畅度、相似度和误分类度量的批评性分数来提高对抗性例子的质量。R&R通过允许探索对错误分类度量没有直接影响但可以提高流畅性和相似性度量的扰动来生成高质量的对抗性示例。我们在5个有代表性的数据集和3个分类器体系结构上对我们的方法进行了评估。我们的方法在攻击成功率上分别比当前最先进的分类器高出16.2%、+12.8%和+14.0%。代码可在https://github.com/DAI-Lab/fibber上找到



## **10. Backdoor Attack and Defense in Federated Generative Adversarial Network-based Medical Image Synthesis**

基于联邦生成对抗网络的医学图像合成后门攻击与防御 cs.CV

25 pages, 7 figures. arXiv admin note: text overlap with  arXiv:2207.00762

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.10886v1) [paper-pdf](http://arxiv.org/pdf/2210.10886v1)

**Authors**: Ruinan Jin, Xiaoxiao Li

**Abstract**: Deep Learning-based image synthesis techniques have been applied in healthcare research for generating medical images to support open research and augment medical datasets. Training generative adversarial neural networks (GANs) usually require large amounts of training data. Federated learning (FL) provides a way of training a central model using distributed data while keeping raw data locally. However, given that the FL server cannot access the raw data, it is vulnerable to backdoor attacks, an adversarial by poisoning training data. Most backdoor attack strategies focus on classification models and centralized domains. It is still an open question if the existing backdoor attacks can affect GAN training and, if so, how to defend against the attack in the FL setting. In this work, we investigate the overlooked issue of backdoor attacks in federated GANs (FedGANs). The success of this attack is subsequently determined to be the result of some local discriminators overfitting the poisoned data and corrupting the local GAN equilibrium, which then further contaminates other clients when averaging the generator's parameters and yields high generator loss. Therefore, we proposed FedDetect, an efficient and effective way of defending against the backdoor attack in the FL setting, which allows the server to detect the client's adversarial behavior based on their losses and block the malicious clients. Our extensive experiments on two medical datasets with different modalities demonstrate the backdoor attack on FedGANs can result in synthetic images with low fidelity. After detecting and suppressing the detected malicious clients using the proposed defense strategy, we show that FedGANs can synthesize high-quality medical datasets (with labels) for data augmentation to improve classification models' performance.

摘要: 基于深度学习的图像合成技术已被应用于医疗保健研究中，以生成医学图像以支持开放研究和扩充医学数据集。生成性对抗神经网络的训练通常需要大量的训练数据。联合学习(FL)提供了一种在本地保留原始数据的同时使用分布式数据训练中央模型的方法。然而，鉴于FL服务器无法访问原始数据，它很容易受到后门攻击，这是通过毒化训练数据而产生的对抗性攻击。大多数后门攻击策略侧重于分类模型和集中域。现有的后门攻击是否会影响GAN的训练，如果是的话，在FL环境下如何防御攻击，仍然是一个悬而未决的问题。在这项工作中，我们研究了联邦GAN(FedGAN)中被忽视的后门攻击问题。这种攻击的成功随后被确定为一些本地鉴别器过度拟合有毒数据并破坏本地GaN平衡的结果，这随后在平均发电机参数时进一步污染其他客户端，并产生高发电机损耗。因此，我们提出了FedDetect，这是一种在FL环境下有效防御后门攻击的方法，它允许服务器根据客户端的损失来检测客户端的敌对行为，并阻止恶意客户端。我们在两个不同模式的医学数据集上的广泛实验表明，对FedGan的后门攻击可以导致合成图像的低保真度。在使用该防御策略检测和抑制检测到的恶意客户端后，我们证明了FedGans能够合成高质量的医学数据集(带标签)用于数据增强，从而提高分类模型的性能。



## **11. On the Perils of Cascading Robust Classifiers**

关于级联稳健分类器的危险 cs.LG

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2206.00278v2) [paper-pdf](http://arxiv.org/pdf/2206.00278v2)

**Authors**: Ravi Mangal, Zifan Wang, Chi Zhang, Klas Leino, Corina Pasareanu, Matt Fredrikson

**Abstract**: Ensembling certifiably robust neural networks is a promising approach for improving the \emph{certified robust accuracy} of neural models. Black-box ensembles that assume only query-access to the constituent models (and their robustness certifiers) during prediction are particularly attractive due to their modular structure. Cascading ensembles are a popular instance of black-box ensembles that appear to improve certified robust accuracies in practice. However, we show that the robustness certifier used by a cascading ensemble is unsound. That is, when a cascading ensemble is certified as locally robust at an input $x$ (with respect to $\epsilon$), there can be inputs $x'$ in the $\epsilon$-ball centered at $x$, such that the cascade's prediction at $x'$ is different from $x$ and thus the ensemble is not locally robust. Our theoretical findings are accompanied by empirical results that further demonstrate this unsoundness. We present \emph{cascade attack} (CasA), an adversarial attack against cascading ensembles, and show that: (1) there exists an adversarial input for up to 88\% of the samples where the ensemble claims to be certifiably robust and accurate; and (2) the accuracy of a cascading ensemble under our attack is as low as 11\% when it claims to be certifiably robust and accurate on 97\% of the test set. Our work reveals a critical pitfall of cascading certifiably robust models by showing that the seemingly beneficial strategy of cascading can actually hurt the robustness of the resulting ensemble. Our code is available at \url{https://github.com/TristaChi/ensembleKW}.

摘要: 集成可证明稳健神经网络是提高神经模型证明稳健精度的一种很有前途的方法。由于其模块化结构，在预测期间假设只对组成模型(及其稳健性证明器)进行查询访问的黑盒集成特别有吸引力。级联组合是黑盒组合的一个受欢迎的例子，它似乎在实践中提高了经过认证的稳健精度。然而，我们证明了级联系综所使用的稳健性证明是不可靠的。也就是说，当级联系综在输入$x$(相对于$\epsilon$)被证明为局部稳健时，在以$x$为中心的$\epsilon$球中可以有输入$x‘$，使得在$x’$处的级联预测不同于$x$，因此该系综不是局部稳健的。我们的理论发现伴随着进一步证明这一不合理的经验结果。我们提出了一种对级联集的对抗性攻击(CASA)，并证明了：(1)对于高达88%的样本，存在对抗性输入，其中集成声称是可证明的健壮和准确的；(2)在我们的攻击下，当级联集成声称在97%的测试集上是可证明的健壮和准确时，其准确率低至11%。我们的工作揭示了级联可证明健壮性模型的一个关键陷阱，即看似有益的级联策略实际上可能会损害由此产生的整体的健壮性。我们的代码可以在\url{https://github.com/TristaChi/ensembleKW}.上找到



## **12. Why Should Adversarial Perturbations be Imperceptible? Rethink the Research Paradigm in Adversarial NLP**

为什么对抗性的扰动应该是不可察觉的？对抗性自然语言处理研究范式的再思考 cs.CL

Accepted to EMNLP 2022, main conference

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.10683v1) [paper-pdf](http://arxiv.org/pdf/2210.10683v1)

**Authors**: Yangyi Chen, Hongcheng Gao, Ganqu Cui, Fanchao Qi, Longtao Huang, Zhiyuan Liu, Maosong Sun

**Abstract**: Textual adversarial samples play important roles in multiple subfields of NLP research, including security, evaluation, explainability, and data augmentation. However, most work mixes all these roles, obscuring the problem definitions and research goals of the security role that aims to reveal the practical concerns of NLP models. In this paper, we rethink the research paradigm of textual adversarial samples in security scenarios. We discuss the deficiencies in previous work and propose our suggestions that the research on the Security-oriented adversarial NLP (SoadNLP) should: (1) evaluate their methods on security tasks to demonstrate the real-world concerns; (2) consider real-world attackers' goals, instead of developing impractical methods. To this end, we first collect, process, and release a security datasets collection Advbench. Then, we reformalize the task and adjust the emphasis on different goals in SoadNLP. Next, we propose a simple method based on heuristic rules that can easily fulfill the actual adversarial goals to simulate real-world attack methods. We conduct experiments on both the attack and the defense sides on Advbench. Experimental results show that our method has higher practical value, indicating that the research paradigm in SoadNLP may start from our new benchmark. All the code and data of Advbench can be obtained at \url{https://github.com/thunlp/Advbench}.

摘要: 文本敌意样本在自然语言处理研究的多个子领域中扮演着重要的角色，包括安全性、评估、可解释性和数据增强。然而，大多数工作混合了所有这些角色，模糊了安全角色的问题定义和研究目标，旨在揭示NLP模型的实际问题。在本文中，我们重新思考了安全场景中文本对抗样本的研究范式。我们讨论了前人工作中的不足，并提出了我们的建议，面向安全的对抗性NLP(SoadNLP)的研究应该：(1)评估他们在安全任务上的方法以展示现实世界的关注点；(2)考虑真实世界攻击者的目标，而不是开发不切实际的方法。为此，我们首先收集、处理和发布安全数据集集合Advbench。然后，我们对SoadNLP的任务进行了改革，并针对不同的目标调整了侧重点。接下来，我们提出了一种简单的基于启发式规则的方法来模拟真实世界的攻击方法，该方法可以很容易地实现实际的对抗目标。我们在Advbench上进行了攻防双方的实验。实验结果表明，我们的方法具有较高的实用价值，表明SoadNLP的研究范式可以从我们的新基准开始。有关安进的所有代码和数据，请访问\url{https://github.com/thunlp/Advbench}.



## **13. Textual Backdoor Attacks Can Be More Harmful via Two Simple Tricks**

文本后门攻击可能通过两个简单的技巧造成更大的危害 cs.CR

Accepted to EMNLP 2022, main conference

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2110.08247v2) [paper-pdf](http://arxiv.org/pdf/2110.08247v2)

**Authors**: Yangyi Chen, Fanchao Qi, Hongcheng Gao, Zhiyuan Liu, Maosong Sun

**Abstract**: Backdoor attacks are a kind of emergent security threat in deep learning. After being injected with a backdoor, a deep neural model will behave normally on standard inputs but give adversary-specified predictions once the input contains specific backdoor triggers. In this paper, we find two simple tricks that can make existing textual backdoor attacks much more harmful. The first trick is to add an extra training task to distinguish poisoned and clean data during the training of the victim model, and the second one is to use all the clean training data rather than remove the original clean data corresponding to the poisoned data. These two tricks are universally applicable to different attack models. We conduct experiments in three tough situations including clean data fine-tuning, low-poisoning-rate, and label-consistent attacks. Experimental results show that the two tricks can significantly improve attack performance. This paper exhibits the great potential harmfulness of backdoor attacks. All the code and data can be obtained at \url{https://github.com/thunlp/StyleAttack}.

摘要: 后门攻击是深度学习中一种突发的安全威胁。在被注入后门后，深层神经模型将在标准输入上正常运行，但一旦输入包含特定的后门触发器，就会给出对手指定的预测。在本文中，我们发现了两个简单的技巧，可以使现有的文本后门攻击更具危害性。第一个技巧是在受害者模型的训练过程中增加一个额外的训练任务来区分有毒和干净的数据，第二个技巧是使用所有的干净训练数据而不是删除有毒数据对应的原始干净数据。这两个技巧普遍适用于不同的攻击模式。我们在三种艰难的情况下进行了实验，包括干净的数据微调、低中毒率和标签一致攻击。实验结果表明，这两种策略都能显著提高攻击性能。本文展示了后门攻击的巨大潜在危害性。所有代码和数据均可在\url{https://github.com/thunlp/StyleAttack}.



## **14. Few-shot Transferable Robust Representation Learning via Bilevel Attacks**

基于两层攻击的少射转移稳健表示学习 cs.LG

*Equal contribution. Author ordering determined by coin flip

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.10485v1) [paper-pdf](http://arxiv.org/pdf/2210.10485v1)

**Authors**: Minseon Kim, Hyeonjeong Ha, Sung Ju Hwang

**Abstract**: Existing adversarial learning methods for enhancing the robustness of deep neural networks assume the availability of a large amount of data from which we can generate adversarial examples. However, in an adversarial meta-learning setting, the model needs to train with only a few adversarial examples to learn a robust model for unseen tasks, which is a very difficult goal to achieve. Further, learning transferable robust representations for unseen domains is a difficult problem even with a large amount of data. To tackle such a challenge, we propose a novel adversarial self-supervised meta-learning framework with bilevel attacks which aims to learn robust representations that can generalize across tasks and domains. Specifically, in the inner loop, we update the parameters of the given encoder by taking inner gradient steps using two different sets of augmented samples, and generate adversarial examples for each view by maximizing the instance classification loss. Then, in the outer loop, we meta-learn the encoder parameter to maximize the agreement between the two adversarial examples, which enables it to learn robust representations. We experimentally validate the effectiveness of our approach on unseen domain adaptation tasks, on which it achieves impressive performance. Specifically, our method significantly outperforms the state-of-the-art meta-adversarial learning methods on few-shot learning tasks, as well as self-supervised learning baselines in standard learning settings with large-scale datasets.

摘要: 现有的用于增强深度神经网络鲁棒性的对抗性学习方法假设有大量数据可用，我们可以从这些数据中生成对抗性示例。然而，在对抗性的元学习环境下，该模型只需要用几个对抗性的例子来训练，以学习一个针对未知任务的健壮模型，这是一个很难实现的目标。此外，即使在有大量数据的情况下，学习不可见领域的可转移的稳健表示也是一个困难的问题。为了应对这样的挑战，我们提出了一种新颖的具有双层攻击的对抗性自监督元学习框架，旨在学习能够跨任务和领域泛化的健壮表示。具体地说，在内循环中，我们通过使用两组不同的增广样本来采取内梯度步骤来更新给定编码器的参数，并通过最大化实例分类损失来为每个视图生成对抗性示例。然后，在外部循环中，我们元学习编码器参数以最大化两个敌对示例之间的一致性，从而使其能够学习稳健的表示。我们在实验上验证了我们方法在看不见的领域适应任务上的有效性，在这些任务上它取得了令人印象深刻的性能。具体地说，在大规模数据集的标准学习环境中，我们的方法在少镜头学习任务上显著优于最先进的元对抗学习方法，并且在自我监督学习基线上也是如此。



## **15. Emerging Threats in Deep Learning-Based Autonomous Driving: A Comprehensive Survey**

基于深度学习的自动驾驶中新出现的威胁：全面调查 cs.CR

28 pages,10 figures

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.11237v1) [paper-pdf](http://arxiv.org/pdf/2210.11237v1)

**Authors**: Hui Cao, Wenlong Zou, Yinkun Wang, Ting Song, Mengjun Liu

**Abstract**: Since the 2004 DARPA Grand Challenge, the autonomous driving technology has witnessed nearly two decades of rapid development. Particularly, in recent years, with the application of new sensors and deep learning technologies extending to the autonomous field, the development of autonomous driving technology has continued to make breakthroughs. Thus, many carmakers and high-tech giants dedicated to research and system development of autonomous driving. However, as the foundation of autonomous driving, the deep learning technology faces many new security risks. The academic community has proposed deep learning countermeasures against the adversarial examples and AI backdoor, and has introduced them into the autonomous driving field for verification. Deep learning security matters to autonomous driving system security, and then matters to personal safety, which is an issue that deserves attention and research.This paper provides an summary of the concepts, developments and recent research in deep learning security technologies in autonomous driving. Firstly, we briefly introduce the deep learning framework and pipeline in the autonomous driving system, which mainly include the deep learning technologies and algorithms commonly used in this field. Moreover, we focus on the potential security threats of the deep learning based autonomous driving system in each functional layer in turn. We reviews the development of deep learning attack technologies to autonomous driving, investigates the State-of-the-Art algorithms, and reveals the potential risks. At last, we provides an outlook on deep learning security in the autonomous driving field and proposes recommendations for building a safe and trustworthy autonomous driving system.

摘要: 自2004年DARPA大赛以来，自动驾驶技术经历了近二十年的快速发展。特别是近年来，随着新型传感器和深度学习技术的应用向自主领域延伸，自动驾驶技术的发展不断取得突破。因此，许多汽车制造商和高科技巨头都致力于自动驾驶的研究和系统开发。然而，深度学习技术作为自动驾驶的基础，也面临着许多新的安全隐患。学术界针对对抗性范例和AI后门提出了深度学习对策，并将其引入自动驾驶领域进行验证。深度学习安全关系到自动驾驶系统的安全，进而关系到人身安全，是一个值得关注和研究的问题。本文综述了自动驾驶中深度学习安全技术的概念、发展和研究现状。首先，简要介绍了自主驾驶系统中深度学习的框架和流水线，主要包括该领域常用的深度学习技术和算法。在此基础上，依次对基于深度学习的自主驾驶系统在各个功能层中存在的潜在安全威胁进行了分析。我们回顾了深度学习攻击技术对自动驾驶的发展，研究了最新的算法，并揭示了潜在的风险。最后，对自主驾驶领域的深度学习安全进行了展望，并对构建安全可信的自动驾驶系统提出了建议。



## **16. On the Adversarial Robustness of Mixture of Experts**

关于专家混合的对抗性稳健性 cs.LG

Accepted to NeurIPS 2022

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.10253v1) [paper-pdf](http://arxiv.org/pdf/2210.10253v1)

**Authors**: Joan Puigcerver, Rodolphe Jenatton, Carlos Riquelme, Pranjal Awasthi, Srinadh Bhojanapalli

**Abstract**: Adversarial robustness is a key desirable property of neural networks. It has been empirically shown to be affected by their sizes, with larger networks being typically more robust. Recently, Bubeck and Sellke proved a lower bound on the Lipschitz constant of functions that fit the training data in terms of their number of parameters. This raises an interesting open question, do -- and can -- functions with more parameters, but not necessarily more computational cost, have better robustness? We study this question for sparse Mixture of Expert models (MoEs), that make it possible to scale up the model size for a roughly constant computational cost. We theoretically show that under certain conditions on the routing and the structure of the data, MoEs can have significantly smaller Lipschitz constants than their dense counterparts. The robustness of MoEs can suffer when the highest weighted experts for an input implement sufficiently different functions. We next empirically evaluate the robustness of MoEs on ImageNet using adversarial attacks and show they are indeed more robust than dense models with the same computational cost. We make key observations showing the robustness of MoEs to the choice of experts, highlighting the redundancy of experts in models trained in practice.

摘要: 对抗健壮性是神经网络的一个关键的理想性质。经验表明，它受到网络规模的影响，更大的网络通常更健壮。最近，Bubeck和Sellke证明了根据参数个数拟合训练数据的函数的Lipschitz常数的一个下界。这提出了一个有趣的开放问题，参数更多但计算成本不一定更高的函数是否--以及能否--具有更好的健壮性？我们研究了稀疏混合专家模型(MOE)的这个问题，它使得在计算成本大致不变的情况下扩大模型的规模成为可能。我们从理论上证明，在一定的布线和数据结构条件下，MOE的Lipschitz常数可以比稠密的MoE小得多。当输入的权重最高的专家实现完全不同的功能时，MOE的稳健性可能会受到影响。接下来，我们在ImageNet上使用对抗性攻击对MOE的健壮性进行了经验性评估，并表明在相同的计算代价下，它们确实比密集模型更健壮。我们进行了关键的观察，显示了MOE对专家选择的稳健性，强调了在实践中训练的模型中专家的冗余。



## **17. Parsimonious Black-Box Adversarial Attacks via Efficient Combinatorial Optimization**

基于高效组合优化的节约型黑箱对抗攻击 cs.LG

Accepted and to appear at ICML 2019

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/1905.06635v2) [paper-pdf](http://arxiv.org/pdf/1905.06635v2)

**Authors**: Seungyong Moon, Gaon An, Hyun Oh Song

**Abstract**: Solving for adversarial examples with projected gradient descent has been demonstrated to be highly effective in fooling the neural network based classifiers. However, in the black-box setting, the attacker is limited only to the query access to the network and solving for a successful adversarial example becomes much more difficult. To this end, recent methods aim at estimating the true gradient signal based on the input queries but at the cost of excessive queries. We propose an efficient discrete surrogate to the optimization problem which does not require estimating the gradient and consequently becomes free of the first order update hyperparameters to tune. Our experiments on Cifar-10 and ImageNet show the state of the art black-box attack performance with significant reduction in the required queries compared to a number of recently proposed methods. The source code is available at https://github.com/snu-mllab/parsimonious-blackbox-attack.

摘要: 用投影梯度下降的对抗性例子的求解已经被证明在愚弄基于神经网络的分类器方面是非常有效的。然而，在黑盒环境下，攻击者仅限于对网络的查询访问，求解一个成功的对抗性例子变得困难得多。为此，最近的方法旨在基于输入查询来估计真实的梯度信号，但代价是过多的查询。对于优化问题，我们提出了一种有效的离散代理，它不需要估计梯度，因此不需要一阶更新超参数来调整。我们在CIFAR-10和ImageNet上的实验显示了最先进的黑盒攻击性能，与最近提出的一些方法相比，所需的查询显著减少。源代码可在https://github.com/snu-mllab/parsimonious-blackbox-attack.上找到



## **18. Scaling Adversarial Training to Large Perturbation Bounds**

将对抗性训练扩展到大扰动界 cs.LG

ECCV 2022

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2210.09852v1) [paper-pdf](http://arxiv.org/pdf/2210.09852v1)

**Authors**: Sravanti Addepalli, Samyak Jain, Gaurang Sriramanan, R. Venkatesh Babu

**Abstract**: The vulnerability of Deep Neural Networks to Adversarial Attacks has fuelled research towards building robust models. While most Adversarial Training algorithms aim at defending attacks constrained within low magnitude Lp norm bounds, real-world adversaries are not limited by such constraints. In this work, we aim to achieve adversarial robustness within larger bounds, against perturbations that may be perceptible, but do not change human (or Oracle) prediction. The presence of images that flip Oracle predictions and those that do not makes this a challenging setting for adversarial robustness. We discuss the ideal goals of an adversarial defense algorithm beyond perceptual limits, and further highlight the shortcomings of naively extending existing training algorithms to higher perturbation bounds. In order to overcome these shortcomings, we propose a novel defense, Oracle-Aligned Adversarial Training (OA-AT), to align the predictions of the network with that of an Oracle during adversarial training. The proposed approach achieves state-of-the-art performance at large epsilon bounds (such as an L-inf bound of 16/255 on CIFAR-10) while outperforming existing defenses (AWP, TRADES, PGD-AT) at standard bounds (8/255) as well.

摘要: 深度神经网络对敌意攻击的脆弱性推动了建立健壮模型的研究。虽然大多数对抗性训练算法的目标是防御被限制在低幅度LP范数范围内的攻击，但现实世界中的对手并不受这种限制的限制。在这项工作中，我们的目标是在更大的范围内实现对抗鲁棒性，对抗可能可感知但不改变人类(或Oracle)预测的扰动。与甲骨文预测相反的图像和非预测图像的存在，使得这对对手的稳健性来说是一个具有挑战性的环境。我们讨论了超越感知极限的对抗性防御算法的理想目标，并进一步强调了将现有训练算法天真地扩展到更高扰动界的缺点。为了克服这些不足，我们提出了一种新的防御方法--Oracle-Align对抗性训练(OA-AT)，以在对抗性训练中使网络的预测与Oracle的预测保持一致。所提出的方法在很大的epsilon界限(例如，CIFAR-10上的L-inf界限为16/255)下实现了最先进的性能，同时在标准界限(8/255)下也超过了现有的防御措施(AWP、TRADS、PGD-AT)。



## **19. Provably Robust Detection of Out-of-distribution Data (almost) for free**

可证明稳健的(几乎)免费的非分布数据检测 cs.LG

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2106.04260v2) [paper-pdf](http://arxiv.org/pdf/2106.04260v2)

**Authors**: Alexander Meinke, Julian Bitterwolf, Matthias Hein

**Abstract**: The application of machine learning in safety-critical systems requires a reliable assessment of uncertainty. However, deep neural networks are known to produce highly overconfident predictions on out-of-distribution (OOD) data. Even if trained to be non-confident on OOD data, one can still adversarially manipulate OOD data so that the classifier again assigns high confidence to the manipulated samples. We show that two previously published defenses can be broken by better adapted attacks, highlighting the importance of robustness guarantees around OOD data. Since the existing method for this task is hard to train and significantly limits accuracy, we construct a classifier that can simultaneously achieve provably adversarially robust OOD detection and high clean accuracy. Moreover, by slightly modifying the classifier's architecture our method provably avoids the asymptotic overconfidence problem of standard neural networks. We provide code for all our experiments.

摘要: 机器学习在安全关键系统中的应用需要对不确定性进行可靠的评估。然而，众所周知，深度神经网络对分布外(OOD)数据产生高度过度自信的预测。即使被训练成对OOD数据不可信，人们仍然可以相反地操纵OOD数据，以便分类器再次为被操纵的样本赋予高置信度。我们证明了之前发表的两个防御措施可以被更好地适应的攻击打破，强调了围绕OOD数据的健壮性保证的重要性。针对现有的分类方法训练难度大、精度低的问题，我们构造了一个分类器，它可以同时实现相对稳健的面向对象检测和较高的清洁准确率。此外，通过略微修改分类器的结构，我们的方法被证明避免了标准神经网络的渐近过度自信问题。我们为我们所有的实验提供代码。



## **20. ROSE: Robust Selective Fine-tuning for Pre-trained Language Models**

ROSE：对预先训练的语言模型进行稳健的选择性微调 cs.CL

Accepted to EMNLP 2022. Code is available at  https://github.com/jiangllan/ROSE

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2210.09658v1) [paper-pdf](http://arxiv.org/pdf/2210.09658v1)

**Authors**: Lan Jiang, Hao Zhou, Yankai Lin, Peng Li, Jie Zhou, Rui Jiang

**Abstract**: Even though the large-scale language models have achieved excellent performances, they suffer from various adversarial attacks. A large body of defense methods has been proposed. However, they are still limited due to redundant attack search spaces and the inability to defend against various types of attacks. In this work, we present a novel fine-tuning approach called \textbf{RO}bust \textbf{SE}letive fine-tuning (\textbf{ROSE}) to address this issue. ROSE conducts selective updates when adapting pre-trained models to downstream tasks, filtering out invaluable and unrobust updates of parameters. Specifically, we propose two strategies: the first-order and second-order ROSE for selecting target robust parameters. The experimental results show that ROSE achieves significant improvements in adversarial robustness on various downstream NLP tasks, and the ensemble method even surpasses both variants above. Furthermore, ROSE can be easily incorporated into existing fine-tuning methods to improve their adversarial robustness further. The empirical analysis confirms that ROSE eliminates unrobust spurious updates during fine-tuning, leading to solutions corresponding to flatter and wider optima than the conventional method. Code is available at \url{https://github.com/jiangllan/ROSE}.

摘要: 尽管大规模语言模型取得了优异的性能，但它们也受到了各种对抗性攻击。已经提出了大量的防御方法。然而，由于多余的攻击搜索空间和无法防御各种类型的攻击，它们仍然有限。在这项工作中，我们提出了一种新的微调方法-.ROSE在调整预先训练的模型以适应下游任务时进行选择性更新，过滤掉无价和不可靠的参数更新。具体地说，我们提出了两种选择目标稳健参数的策略：一阶ROSE和二阶ROSE。实验结果表明，ROSE在不同下游NLP任务上的对抗性健壮性有了显著的提高，集成方法甚至超过了上述两种方法。此外，ROSE可以很容易地结合到现有的微调方法中，以进一步提高它们的对抗健壮性。实证分析证实，ROSE消除了微调过程中不稳健的虚假更新，得到了与传统方法相比更平坦和更广泛的最优解。代码位于\url{https://github.com/jiangllan/ROSE}.



## **21. Analysis of Master Vein Attacks on Finger Vein Recognition Systems**

主静脉攻击对手指静脉识别系统的影响分析 cs.CV

Accepted to be Published in Proceedings of the IEEE/CVF Winter  Conference on Applications of Computer Vision (WACV) 2023

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2210.10667v1) [paper-pdf](http://arxiv.org/pdf/2210.10667v1)

**Authors**: Huy H. Nguyen, Trung-Nghia Le, Junichi Yamagishi, Isao Echizen

**Abstract**: Finger vein recognition (FVR) systems have been commercially used, especially in ATMs, for customer verification. Thus, it is essential to measure their robustness against various attack methods, especially when a hand-crafted FVR system is used without any countermeasure methods. In this paper, we are the first in the literature to introduce master vein attacks in which we craft a vein-looking image so that it can falsely match with as many identities as possible by the FVR systems. We present two methods for generating master veins for use in attacking these systems. The first uses an adaptation of the latent variable evolution algorithm with a proposed generative model (a multi-stage combination of beta-VAE and WGAN-GP models). The second uses an adversarial machine learning attack method to attack a strong surrogate CNN-based recognition system. The two methods can be easily combined to boost their attack ability. Experimental results demonstrated that the proposed methods alone and together achieved false acceptance rates up to 73.29% and 88.79%, respectively, against Miura's hand-crafted FVR system. We also point out that Miura's system is easily compromised by non-vein-looking samples generated by a WGAN-GP model with false acceptance rates up to 94.21%. The results raise the alarm about the robustness of such systems and suggest that master vein attacks should be considered an important security measure.

摘要: 手指静脉识别(FVR)系统已经在商业上使用，特别是在自动取款机中，用于客户验证。因此，测量它们对各种攻击方法的稳健性是至关重要的，特别是当使用手工制作的FVR系统而没有任何对抗方法时。在本文中，我们在文献中首次引入了主静脉攻击，在这种攻击中，我们制作了一张看起来像静脉的图像，以便它可以通过FVR系统与尽可能多的身份进行虚假匹配。我们提出了两种生成主脉以用于攻击这些系统的方法。第一种是将潜在变量进化算法与所提出的生成模型(Beta-VAE和WGAN-GP模型的多阶段组合)相结合。第二个攻击是使用对抗性机器学习攻击方法来攻击基于CNN的强代理识别系统。这两种方法可以很容易地结合起来，以提高他们的攻击能力。实验结果表明，与Miura手工制作的FVR系统相比，提出的方法单独和联合使用的错误接受率分别高达73.29%和88.79%。我们还指出，Miura的系统很容易被WGAN-GP模型生成的非静脉样本所危害，错误接受率高达94.21%。这一结果对这类系统的健壮性提出了警告，并建议应将主静脉攻击视为一项重要的安全措施。



## **22. Making Split Learning Resilient to Label Leakage by Potential Energy Loss**

利用潜在能量损失使分裂学习具有抗泄漏能力 cs.CR

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2210.09617v1) [paper-pdf](http://arxiv.org/pdf/2210.09617v1)

**Authors**: Fei Zheng, Chaochao Chen, Binhui Yao, Xiaolin Zheng

**Abstract**: As a practical privacy-preserving learning method, split learning has drawn much attention in academia and industry. However, its security is constantly being questioned since the intermediate results are shared during training and inference. In this paper, we focus on the privacy leakage problem caused by the trained split model, i.e., the attacker can use a few labeled samples to fine-tune the bottom model, and gets quite good performance. To prevent such kind of privacy leakage, we propose the potential energy loss to make the output of the bottom model become a more `complicated' distribution, by pushing outputs of the same class towards the decision boundary. Therefore, the adversary suffers a large generalization error when fine-tuning the bottom model with only a few leaked labeled samples. Experiment results show that our method significantly lowers the attacker's fine-tuning accuracy, making the split model more resilient to label leakage.

摘要: 分裂学习作为一种实用的隐私保护学习方法，受到了学术界和工业界的广泛关注。然而，由于中间结果是在训练和推理过程中共享的，其安全性不断受到质疑。本文重点研究了训练好的分裂模型带来的隐私泄露问题，即攻击者可以利用少量的标签样本对底层模型进行微调，取得了较好的性能。为了防止这种隐私泄露，我们提出了潜在能量损失，通过将同一类的输出推向决策边界，使底层模型的输出成为更复杂的分布。因此，对手在仅用几个泄漏的标签样本对底层模型进行微调时，会遭受较大的泛化误差。实验结果表明，该方法显著降低了攻击者的微调精度，使分裂模型具有更强的抗标签泄漏能力。



## **23. Friendly Noise against Adversarial Noise: A Powerful Defense against Data Poisoning Attacks**

友善噪声对抗敌意噪声：数据中毒攻击的有力防御 cs.CR

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2208.10224v3) [paper-pdf](http://arxiv.org/pdf/2208.10224v3)

**Authors**: Tian Yu Liu, Yu Yang, Baharan Mirzasoleiman

**Abstract**: A powerful category of (invisible) data poisoning attacks modify a subset of training examples by small adversarial perturbations to change the prediction of certain test-time data. Existing defense mechanisms are not desirable to deploy in practice, as they often either drastically harm the generalization performance, or are attack-specific, and prohibitively slow to apply. Here, we propose a simple but highly effective approach that unlike existing methods breaks various types of invisible poisoning attacks with the slightest drop in the generalization performance. We make the key observation that attacks introduce local sharp regions of high training loss, which when minimized, results in learning the adversarial perturbations and makes the attack successful. To break poisoning attacks, our key idea is to alleviate the sharp loss regions introduced by poisons. To do so, our approach comprises two components: an optimized friendly noise that is generated to maximally perturb examples without degrading the performance, and a randomly varying noise component. The combination of both components builds a very light-weight but extremely effective defense against the most powerful triggerless targeted and hidden-trigger backdoor poisoning attacks, including Gradient Matching, Bulls-eye Polytope, and Sleeper Agent. We show that our friendly noise is transferable to other architectures, and adaptive attacks cannot break our defense due to its random noise component.

摘要: 一种强大的(不可见)数据中毒攻击通过小的对抗性扰动来修改训练样本的子集，以改变对某些测试时间数据的预测。现有的防御机制在实践中并不可取，因为它们通常要么严重损害泛化性能，要么是特定于攻击的，应用速度慢得令人望而却步。在这里，我们提出了一种简单而高效的方法，不同于现有的方法，它以最小的泛化性能下降来破解各种类型的隐形中毒攻击。我们的关键观察是，攻击引入了高训练损失的局部尖锐区域，当训练损失最小化时，导致学习对手的扰动，使攻击成功。要打破毒物攻击，我们的关键思想是减轻毒物引入的急剧损失区域。为此，我们的方法包括两个组件：一个优化的友好噪声，它被生成以在不降低性能的情况下最大限度地扰动示例，以及一个随机变化的噪声组件。这两个组件的组合构建了一个非常轻但极其有效的防御系统，以抵御最强大的无触发器定向和隐藏触发器后门中毒攻击，包括梯度匹配、公牛眼多面体和睡眠代理。我们证明了我们的友好噪声是可以转移到其他体系结构的，而自适应攻击由于其随机噪声成分而不能破坏我们的防御。



## **24. Make Some Noise: Reliable and Efficient Single-Step Adversarial Training**

制造一些噪音：可靠而高效的单步对抗性训练 cs.LG

Published in NeurIPS 2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2202.01181v3) [paper-pdf](http://arxiv.org/pdf/2202.01181v3)

**Authors**: Pau de Jorge, Adel Bibi, Riccardo Volpi, Amartya Sanyal, Philip H. S. Torr, Grégory Rogez, Puneet K. Dokania

**Abstract**: Recently, Wong et al. showed that adversarial training with single-step FGSM leads to a characteristic failure mode named Catastrophic Overfitting (CO), in which a model becomes suddenly vulnerable to multi-step attacks. Experimentally they showed that simply adding a random perturbation prior to FGSM (RS-FGSM) could prevent CO. However, Andriushchenko and Flammarion observed that RS-FGSM still leads to CO for larger perturbations, and proposed a computationally expensive regularizer (GradAlign) to avoid it. In this work, we methodically revisit the role of noise and clipping in single-step adversarial training. Contrary to previous intuitions, we find that using a stronger noise around the clean sample combined with \textit{not clipping} is highly effective in avoiding CO for large perturbation radii. We then propose Noise-FGSM (N-FGSM) that, while providing the benefits of single-step adversarial training, does not suffer from CO. Empirical analyses on a large suite of experiments show that N-FGSM is able to match or surpass the performance of previous state-of-the-art GradAlign, while achieving 3x speed-up. Code can be found in https://github.com/pdejorge/N-FGSM

摘要: 最近，Wong et al.研究表明，采用单步FGSM的对抗性训练会导致一种称为灾难性过匹配(CO)的特征故障模式，在这种模式下，模型突然变得容易受到多步攻击。在实验上，他们表明，在FGSM(RS-FGSM)之前简单地添加随机扰动可以防止CO。然而，Andriushchenko和Flammarion观察到，对于较大的扰动，RS-FGSM仍然会导致CO，并提出了一种计算代价很高的正则化方法(GradAlign)来避免这种情况。在这项工作中，我们有条不紊地重新审视噪声和剪辑在单步对抗性训练中的作用。与以前的直觉相反，我们发现在清洁的样品周围使用更强的噪声并结合纹理{不剪裁}在大扰动半径下避免CO是非常有效的。然后，我们提出了噪声-FGSM(N-FGSM)，它在提供单步对抗性训练的好处的同时，不会受到CO的影响。大量实验的实验分析表明，N-FGSM能够在性能上赶上或超过以往最先进的GradAlign，同时获得3倍的加速。代码可在https://github.com/pdejorge/N-FGSM中找到



## **25. Deepfake Text Detection: Limitations and Opportunities**

深度假冒文本检测：局限与机遇 cs.CR

Accepted to IEEE S&P 2023; First two authors contributed equally to  this work; 18 pages, 7 figures

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.09421v1) [paper-pdf](http://arxiv.org/pdf/2210.09421v1)

**Authors**: Jiameng Pu, Zain Sarwar, Sifat Muhammad Abdullah, Abdullah Rehman, Yoonjin Kim, Parantapa Bhattacharya, Mobin Javed, Bimal Viswanath

**Abstract**: Recent advances in generative models for language have enabled the creation of convincing synthetic text or deepfake text. Prior work has demonstrated the potential for misuse of deepfake text to mislead content consumers. Therefore, deepfake text detection, the task of discriminating between human and machine-generated text, is becoming increasingly critical. Several defenses have been proposed for deepfake text detection. However, we lack a thorough understanding of their real-world applicability. In this paper, we collect deepfake text from 4 online services powered by Transformer-based tools to evaluate the generalization ability of the defenses on content in the wild. We develop several low-cost adversarial attacks, and investigate the robustness of existing defenses against an adaptive attacker. We find that many defenses show significant degradation in performance under our evaluation scenarios compared to their original claimed performance. Our evaluation shows that tapping into the semantic information in the text content is a promising approach for improving the robustness and generalization performance of deepfake text detection schemes.

摘要: 语言生成模型的最新进展使得创造令人信服的合成文本或深度假文本成为可能。先前的工作已经证明了滥用深度虚假文本来误导内容消费者的可能性。因此，区分人类和机器生成的文本的任务--深度虚假文本检测变得越来越关键。已经提出了几种针对深度虚假文本检测的防御措施。然而，我们对它们在现实世界中的适用性缺乏透彻的了解。在本文中，我们从基于Transformer的工具支持的4个在线服务中收集深度虚假文本，以评估这些防御措施对野生内容的泛化能力。我们开发了几种低成本的对抗性攻击，并研究了现有防御措施对自适应攻击者的健壮性。我们发现，与最初声称的性能相比，在我们的评估情景下，许多防御系统的性能显著下降。我们的评估表明，挖掘文本内容中的语义信息是一种很有前途的方法，可以提高深度虚假文本检测方案的稳健性和泛化性能。



## **26. Towards Generating Adversarial Examples on Mixed-type Data**

在混合类型数据上生成对抗性实例 cs.LG

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.09405v1) [paper-pdf](http://arxiv.org/pdf/2210.09405v1)

**Authors**: Han Xu, Menghai Pan, Zhimeng Jiang, Huiyuan Chen, Xiaoting Li, Mahashweta Das, Hao Yang

**Abstract**: The existence of adversarial attacks (or adversarial examples) brings huge concern about the machine learning (ML) model's safety issues. For many safety-critical ML tasks, such as financial forecasting, fraudulent detection, and anomaly detection, the data samples are usually mixed-type, which contain plenty of numerical and categorical features at the same time. However, how to generate adversarial examples with mixed-type data is still seldom studied. In this paper, we propose a novel attack algorithm M-Attack, which can effectively generate adversarial examples in mixed-type data. Based on M-Attack, attackers can attempt to mislead the targeted classification model's prediction, by only slightly perturbing both the numerical and categorical features in the given data samples. More importantly, by adding designed regularizations, our generated adversarial examples can evade potential detection models, which makes the attack indeed insidious. Through extensive empirical studies, we validate the effectiveness and efficiency of our attack method and evaluate the robustness of existing classification models against our proposed attack. The experimental results highlight the feasibility of generating adversarial examples toward machine learning models in real-world applications.

摘要: 对抗性攻击(或对抗性例子)的存在给机器学习(ML)模型的安全问题带来了巨大的担忧。对于金融预测、欺诈检测、异常检测等许多安全关键的ML任务，数据样本通常是混合类型的，同时包含大量的数值和分类特征。然而，如何利用混合类型数据生成对抗性实例的研究还很少。本文提出了一种新的攻击算法M-Attack，该算法能够有效地生成混合类型数据中的对抗性实例。基于M-攻击，攻击者只需对给定数据样本中的数值特征和分类特征稍加干扰，就可以试图误导目标分类模型的预测。更重要的是，通过添加设计的正则化，我们生成的敌意示例可以避开潜在的检测模型，这使得攻击实际上是隐蔽的。通过大量的实验研究，我们验证了我们的攻击方法的有效性和效率，并评估了现有分类模型对我们提出的攻击的健壮性。实验结果突出了在实际应用中生成针对机器学习模型的对抗性示例的可行性。



## **27. Probabilistic Categorical Adversarial Attack & Adversarial Training**

概率分类对抗性攻击与对抗性训练 cs.LG

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.09364v1) [paper-pdf](http://arxiv.org/pdf/2210.09364v1)

**Authors**: Penghei He, Han Xu, Jie Ren, Yuxuan Wan, Zitao Liu, Jiliang Tang

**Abstract**: The existence of adversarial examples brings huge concern for people to apply Deep Neural Networks (DNNs) in safety-critical tasks. However, how to generate adversarial examples with categorical data is an important problem but lack of extensive exploration. Previously established methods leverage greedy search method, which can be very time-consuming to conduct successful attack. This also limits the development of adversarial training and potential defenses for categorical data. To tackle this problem, we propose Probabilistic Categorical Adversarial Attack (PCAA), which transfers the discrete optimization problem to a continuous problem that can be solved efficiently by Projected Gradient Descent. In our paper, we theoretically analyze its optimality and time complexity to demonstrate its significant advantage over current greedy based attacks. Moreover, based on our attack, we propose an efficient adversarial training framework. Through a comprehensive empirical study, we justify the effectiveness of our proposed attack and defense algorithms.

摘要: 对抗性实例的存在给深度神经网络在安全关键任务中的应用带来了极大的关注。然而，如何利用分类数据生成对抗性实例是一个重要的问题，但缺乏广泛的探索。以前建立的方法利用贪婪搜索方法，进行成功的攻击可能非常耗时。这也限制了对抗性训练的发展和对分类数据的潜在防御。为了解决这个问题，我们提出了概率分类对抗性攻击(PCAA)，它将离散的优化问题转化为一个连续的问题，可以用投影梯度下降法有效地解决。在本文中，我们从理论上分析了它的最优性和时间复杂性，以证明它相对于现有的基于贪婪的攻击具有显著的优势。此外，基于我们的攻击，我们提出了一个有效的对抗性训练框架。通过全面的实证研究，验证了本文提出的攻防算法的有效性。



## **28. Marksman Backdoor: Backdoor Attacks with Arbitrary Target Class**

神枪手后门：任意目标等级的后门攻击 cs.CR

Accepted to NeurIPS 2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.09194v1) [paper-pdf](http://arxiv.org/pdf/2210.09194v1)

**Authors**: Khoa D. Doan, Yingjie Lao, Ping Li

**Abstract**: In recent years, machine learning models have been shown to be vulnerable to backdoor attacks. Under such attacks, an adversary embeds a stealthy backdoor into the trained model such that the compromised models will behave normally on clean inputs but will misclassify according to the adversary's control on maliciously constructed input with a trigger. While these existing attacks are very effective, the adversary's capability is limited: given an input, these attacks can only cause the model to misclassify toward a single pre-defined or target class. In contrast, this paper exploits a novel backdoor attack with a much more powerful payload, denoted as Marksman, where the adversary can arbitrarily choose which target class the model will misclassify given any input during inference. To achieve this goal, we propose to represent the trigger function as a class-conditional generative model and to inject the backdoor in a constrained optimization framework, where the trigger function learns to generate an optimal trigger pattern to attack any target class at will while simultaneously embedding this generative backdoor into the trained model. Given the learned trigger-generation function, during inference, the adversary can specify an arbitrary backdoor attack target class, and an appropriate trigger causing the model to classify toward this target class is created accordingly. We show empirically that the proposed framework achieves high attack performance while preserving the clean-data performance in several benchmark datasets, including MNIST, CIFAR10, GTSRB, and TinyImageNet. The proposed Marksman backdoor attack can also easily bypass existing backdoor defenses that were originally designed against backdoor attacks with a single target class. Our work takes another significant step toward understanding the extensive risks of backdoor attacks in practice.

摘要: 近年来，机器学习模型被证明容易受到后门攻击。在这样的攻击下，对手在训练的模型中嵌入一个秘密的后门，这样受攻击的模型将在干净的输入上正常运行，但将根据对手对带有触发器的恶意构建的输入的控制进行错误分类。虽然这些现有的攻击非常有效，但对手的能力是有限的：在给定输入的情况下，这些攻击只能导致模型错误分类为单个预定义或目标类。相反，本文利用了一种新的后门攻击，具有更强大的有效载荷，表示为射手，其中对手可以任意选择模型将错误分类的目标类别，在推理过程中给定任何输入。为了实现这一目标，我们建议将触发函数表示为类条件生成模型，并在约束优化框架中注入后门，其中触发函数学习生成任意攻击目标类的最优触发模式，同时将该生成后门嵌入到训练的模型中。给定学习的触发器生成函数，在推理期间，对手可以指定任意的后门攻击目标类，并且相应地创建导致模型向该目标类分类的适当触发器。我们在MNIST、CIFAR10、GTSRB和TinyImageNet等几个基准数据集上的实验表明，该框架在保持干净数据性能的同时，实现了高的攻击性能。拟议的射手后门攻击也可以很容易地绕过现有的后门防御，这些后门防御最初是针对单一目标类别的后门攻击而设计的。我们的工作朝着了解后门攻击在实践中的广泛风险又迈出了重要的一步。



## **29. Adversarial Robustness is at Odds with Lazy Training**

对抗健壮性与懒惰训练不一致 cs.LG

NeurIPS 2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2207.00411v2) [paper-pdf](http://arxiv.org/pdf/2207.00411v2)

**Authors**: Yunjuan Wang, Enayat Ullah, Poorya Mianjy, Raman Arora

**Abstract**: Recent works show that adversarial examples exist for random neural networks [Daniely and Schacham, 2020] and that these examples can be found using a single step of gradient ascent [Bubeck et al., 2021]. In this work, we extend this line of work to "lazy training" of neural networks -- a dominant model in deep learning theory in which neural networks are provably efficiently learnable. We show that over-parametrized neural networks that are guaranteed to generalize well and enjoy strong computational guarantees remain vulnerable to attacks generated using a single step of gradient ascent.

摘要: 最近的工作表明，随机神经网络存在对抗性的例子[Daniely和Schacham，2020]，并且这些例子可以使用单一的梯度上升步骤来找到[Bubeck等人，2021]。在这项工作中，我们将这一工作扩展到神经网络的“懒惰训练”--深度学习理论中的一种主要模型，在这种模型中，神经网络被证明是可有效学习的。我们证明了过度参数化的神经网络具有良好的泛化能力和强大的计算保证，但仍然容易受到使用单步梯度上升产生的攻击。



## **30. DE-CROP: Data-efficient Certified Robustness for Pretrained Classifiers**

反裁剪：用于预先训练的分类器的数据高效认证稳健性 cs.LG

WACV 2023. Project page: https://sites.google.com/view/decrop

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08929v1) [paper-pdf](http://arxiv.org/pdf/2210.08929v1)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Anirban Chakraborty

**Abstract**: Certified defense using randomized smoothing is a popular technique to provide robustness guarantees for deep neural networks against l2 adversarial attacks. Existing works use this technique to provably secure a pretrained non-robust model by training a custom denoiser network on entire training data. However, access to the training set may be restricted to a handful of data samples due to constraints such as high transmission cost and the proprietary nature of the data. Thus, we formulate a novel problem of "how to certify the robustness of pretrained models using only a few training samples". We observe that training the custom denoiser directly using the existing techniques on limited samples yields poor certification. To overcome this, our proposed approach (DE-CROP) generates class-boundary and interpolated samples corresponding to each training sample, ensuring high diversity in the feature space of the pretrained classifier. We train the denoiser by maximizing the similarity between the denoised output of the generated sample and the original training sample in the classifier's logit space. We also perform distribution level matching using domain discriminator and maximum mean discrepancy that yields further benefit. In white box setup, we obtain significant improvements over the baseline on multiple benchmark datasets and also report similar performance under the challenging black box setup.

摘要: 使用随机化平滑的认证防御是一种流行的技术，可以为深层神经网络提供抵御L2攻击的健壮性保证。现有的工作使用这种技术来通过在整个训练数据上训练自定义去噪器网络来证明预先训练的非稳健模型的安全。然而，由于诸如高传输成本和数据的专有性质等限制，对训练集的访问可能被限制为少数数据样本。因此，我们提出了一个新的问题：如何仅用几个训练样本来证明预先训练的模型的稳健性。我们观察到，直接使用现有技术在有限的样本上培训自定义去噪器会产生较差的认证。为了克服这一问题，我们提出的方法(DE-CROP)生成对应于每个训练样本的类边界样本和内插样本，从而确保预先训练的分类器的特征空间具有高度的多样性。在分类器的Logit空间中，我们通过最大化生成样本的去噪输出与原始训练样本之间的相似度来训练去噪器。我们还使用域鉴别器和最大平均差异来执行分布级别匹配，从而产生进一步的好处。在白盒设置中，我们在多个基准数据集上获得了比基线显著的改进，并且在具有挑战性的黑盒设置下也报告了类似的性能。



## **31. Beyond Model Interpretability: On the Faithfulness and Adversarial Robustness of Contrastive Textual Explanations**

超越模式的可解释性--论对比文本解释的忠实性和对抗性 cs.CL

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08902v1) [paper-pdf](http://arxiv.org/pdf/2210.08902v1)

**Authors**: Julia El Zini, Mariette Awad

**Abstract**: Contrastive explanation methods go beyond transparency and address the contrastive aspect of explanations. Such explanations are emerging as an attractive option to provide actionable change to scenarios adversely impacted by classifiers' decisions. However, their extension to textual data is under-explored and there is little investigation on their vulnerabilities and limitations.   This work motivates textual counterfactuals by laying the ground for a novel evaluation scheme inspired by the faithfulness of explanations. Accordingly, we extend the computation of three metrics, proximity,connectedness and stability, to textual data and we benchmark two successful contrastive methods, POLYJUICE and MiCE, on our suggested metrics. Experiments on sentiment analysis data show that the connectedness of counterfactuals to their original counterparts is not obvious in both models. More interestingly, the generated contrastive texts are more attainable with POLYJUICE which highlights the significance of latent representations in counterfactual search. Finally, we perform the first semantic adversarial attack on textual recourse methods. The results demonstrate the robustness of POLYJUICE and the role that latent input representations play in robustness and reliability.

摘要: 对比解释方法超越了透明度，解决了解释的对比方面。这样的解释正在成为一种有吸引力的选择，可以为受到分类员决定不利影响的情况提供可操作的改变。然而，它们对文本数据的扩展还没有得到充分的探索，对它们的脆弱性和局限性的调查也很少。这项工作通过为一个新的评估方案奠定了基础，从而激发了文本反事实的动机，该方案的灵感来自于解释的真实性。相应地，我们将邻近度、连通度和稳定性这三个指标的计算扩展到文本数据，并在我们建议的指标上对两种成功的对比方法Polyjuus和MICE进行了基准测试。在情感分析数据上的实验表明，在这两个模型中，反事实与原始事实的联系并不明显。更有趣的是，生成的对比文本更容易通过Polyjuus获得，这突显了潜在表征在反事实搜索中的重要性。最后，我们对文本求助方法进行了首次语义对抗性攻击。实验结果证明了Polyjuus的稳健性，以及潜在输入表征在稳健性和可靠性方面所起的作用。



## **32. Differential Evolution based Dual Adversarial Camouflage: Fooling Human Eyes and Object Detectors**

基于差异进化的双重对抗性伪装：愚弄人眼和目标探测器 cs.CV

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08870v1) [paper-pdf](http://arxiv.org/pdf/2210.08870v1)

**Authors**: Jialiang Sun

**Abstract**: Recent studies reveal that deep neural network (DNN) based object detectors are vulnerable to adversarial attacks in the form of adding the perturbation to the images, leading to the wrong output of object detectors. Most current existing works focus on generating perturbed images, also called adversarial examples, to fool object detectors. Though the generated adversarial examples themselves can remain a certain naturalness, most of them can still be easily observed by human eyes, which limits their further application in the real world. To alleviate this problem, we propose a differential evolution based dual adversarial camouflage (DE_DAC) method, composed of two stages to fool human eyes and object detectors simultaneously. Specifically, we try to obtain the camouflage texture, which can be rendered over the surface of the object. In the first stage, we optimize the global texture to minimize the discrepancy between the rendered object and the scene images, making human eyes difficult to distinguish. In the second stage, we design three loss functions to optimize the local texture, making object detectors ineffective. In addition, we introduce the differential evolution algorithm to search for the near-optimal areas of the object to attack, improving the adversarial performance under certain attack area limitations. Besides, we also study the performance of adaptive DE_DAC, which can be adapted to the environment. Experiments show that our proposed method could obtain a good trade-off between the fooling human eyes and object detectors under multiple specific scenes and objects.

摘要: 最近的研究表明，基于深度神经网络(DNN)的目标检测器容易受到敌意攻击，其形式是向图像添加扰动，导致目标检测器的输出错误。目前大多数现有的工作都集中在生成扰动图像，也称为对抗性示例，以愚弄对象检测器。尽管生成的对抗性例子本身可以保持一定的自然度，但其中大部分仍然很容易被人眼观察到，这限制了它们在现实世界中的进一步应用。为了缓解这一问题，我们提出了一种基于差异进化的双重对抗伪装(DE_DAC)方法，该方法由两个阶段组成，同时欺骗人眼和目标检测器。具体地说，我们试图获得伪装纹理，它可以在对象的表面上渲染。在第一阶段，我们对全局纹理进行优化，最小化绘制对象和场景图像之间的差异，使人眼难以辨别。在第二阶段，我们设计了三个损失函数来优化局部纹理，使得目标检测失效。此外，我们还引入了差分进化算法来搜索攻击对象的近最优区域，提高了在一定攻击区域限制下的对抗性能。此外，我们还研究了适应环境的自适应DE_DAC的性能。实验表明，在多个特定场景和目标的情况下，我们提出的方法可以在愚弄人眼和目标检测器之间取得良好的折衷。



## **33. ODG-Q: Robust Quantization via Online Domain Generalization**

ODG-Q：基于在线域泛化的稳健量化 cs.LG

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08701v1) [paper-pdf](http://arxiv.org/pdf/2210.08701v1)

**Authors**: Chaofan Tao, Ngai Wong

**Abstract**: Quantizing neural networks to low-bitwidth is important for model deployment on resource-limited edge hardware. Although a quantized network has a smaller model size and memory footprint, it is fragile to adversarial attacks. However, few methods study the robustness and training efficiency of quantized networks. To this end, we propose a new method by recasting robust quantization as an online domain generalization problem, termed ODG-Q, which generates diverse adversarial data at a low cost during training. ODG-Q consistently outperforms existing works against various adversarial attacks. For example, on CIFAR-10 dataset, ODG-Q achieves 49.2% average improvements under five common white-box attacks and 21.7% average improvements under five common black-box attacks, with a training cost similar to that of natural training (viz. without adversaries). To our best knowledge, this work is the first work that trains both quantized and binary neural networks on ImageNet that consistently improve robustness under different attacks. We also provide a theoretical insight of ODG-Q that accounts for the bound of model risk on attacked data.

摘要: 将神经网络量化为低位宽对于在资源有限的边缘硬件上部署模型具有重要意义。虽然量化网络具有较小的模型大小和内存占用，但它很容易受到对手攻击。然而，很少有方法研究量化网络的稳健性和训练效率。为此，我们提出了一种新的方法，将稳健量化重塑为一个在线领域泛化问题，称为ODG-Q，该方法在训练过程中以较低的代价生成不同的对抗性数据。ODG-Q在抵抗各种对抗性攻击时的表现一直优于现有的工作。例如，在CIFAR-10数据集上，在五种常见的白盒攻击下，ODG-Q的平均性能提高了49.2%，在五种常见的黑盒攻击下，ODG-Q的平均性能提高了21.7%，而训练代价与自然训练(即.没有对手)。据我们所知，这是第一个在ImageNet上训练量化和二进制神经网络的工作，这些网络在不同的攻击下都能持续提高健壮性。我们还提供了ODG-Q的理论见解，它解释了攻击数据上的模型风险的界限。



## **34. A2: Efficient Automated Attacker for Boosting Adversarial Training**

A2：用于加强对抗性训练的高效自动攻击者 cs.CV

Accepted by NeurIPS2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.03543v2) [paper-pdf](http://arxiv.org/pdf/2210.03543v2)

**Authors**: Zhuoer Xu, Guanghui Zhu, Changhua Meng, Shiwen Cui, Zhenzhe Ying, Weiqiang Wang, Ming GU, Yihua Huang

**Abstract**: Based on the significant improvement of model robustness by AT (Adversarial Training), various variants have been proposed to further boost the performance. Well-recognized methods have focused on different components of AT (e.g., designing loss functions and leveraging additional unlabeled data). It is generally accepted that stronger perturbations yield more robust models. However, how to generate stronger perturbations efficiently is still missed. In this paper, we propose an efficient automated attacker called A2 to boost AT by generating the optimal perturbations on-the-fly during training. A2 is a parameterized automated attacker to search in the attacker space for the best attacker against the defense model and examples. Extensive experiments across different datasets demonstrate that A2 generates stronger perturbations with low extra cost and reliably improves the robustness of various AT methods against different attacks.

摘要: 在对抗训练显著提高模型稳健性的基础上，各种变种被提出以进一步提高性能。公认的方法侧重于AT的不同组成部分(例如，设计损失函数和利用额外的未标记数据)。人们普遍认为，更强的扰动会产生更稳健的模型。然而，如何有效地产生更强的扰动仍然是一个未解决的问题。在本文中，我们提出了一种称为A2的高效自动攻击者，通过在训练过程中生成最优的动态扰动来增强AT。A2是一个参数化的自动攻击者，可以在攻击者空间中搜索最好的攻击者，并针对防御模型和实例进行攻击。在不同数据集上的大量实验表明，A2以较低的额外代价产生更强的扰动，并可靠地提高了各种AT方法对不同攻击的健壮性。



## **35. Reliability and Robustness analysis of Machine Learning based Phishing URL Detectors**

基于机器学习的钓鱼URL检测器的可靠性和稳健性分析 cs.CR

Accepted in Transactions of Dependable and Secure Computing  (SI-Reliability and Robustness in AI-Based Cybersecurity Solutions)

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2005.08454v2) [paper-pdf](http://arxiv.org/pdf/2005.08454v2)

**Authors**: Bushra Sabir, M. Ali Babar, Raj Gaire, Alsharif Abuadbba

**Abstract**: ML-based Phishing URL (MLPU) detectors serve as the first level of defence to protect users and organisations from being victims of phishing attacks. Lately, few studies have launched successful adversarial attacks against specific MLPU detectors raising questions about their practical reliability and usage. Nevertheless, the robustness of these systems has not been extensively investigated. Therefore, the security vulnerabilities of these systems, in general, remain primarily unknown which calls for testing the robustness of these systems. In this article, we have proposed a methodology to investigate the reliability and robustness of 50 representative state-of-the-art MLPU models. Firstly, we have proposed a cost-effective Adversarial URL generator URLBUG that created an Adversarial URL dataset. Subsequently, we reproduced 50 MLPU (traditional ML and Deep learning) systems and recorded their baseline performance. Lastly, we tested the considered MLPU systems on Adversarial Dataset and analyzed their robustness and reliability using box plots and heat maps. Our results showed that the generated adversarial URLs have valid syntax and can be registered at a median annual price of \$11.99. Out of 13\% of the already registered adversarial URLs, 63.94\% were used for malicious purposes. Moreover, the considered MLPU models Matthew Correlation Coefficient (MCC) dropped from a median 0.92 to 0.02 when tested against $Adv_\mathrm{data}$, indicating that the baseline MLPU models are unreliable in their current form. Further, our findings identified several security vulnerabilities of these systems and provided future directions for researchers to design dependable and secure MLPU systems.

摘要: 基于ML的钓鱼URL(MLPU)检测器是保护用户和组织免受钓鱼攻击的第一级防御。最近，很少有研究针对特定的MLPU检测器发起成功的对抗性攻击，这引发了对其实际可靠性和使用的质疑。然而，这些系统的稳健性还没有得到广泛的研究。因此，这些系统的安全漏洞总体上仍然是未知的，这就需要测试这些系统的健壮性。在本文中，我们提出了一种方法来调查50个最具代表性的MLPU模型的可靠性和稳健性。首先，我们提出了一个高性价比的敌意URL生成器URLBUG，它创建了一个敌意URL数据集。随后，我们复制了50个MLPU(传统ML和深度学习)系统，并记录了它们的基线性能。最后，我们在敌意数据集上对所考虑的MLPU系统进行了测试，并使用盒图和热图分析了它们的健壮性和可靠性。我们的结果表明，生成的恶意URL具有有效的语法，并且可以以11.99美元的中位数年价格注册。在已注册的13个恶意URL中，有63.94个被用于恶意目的。此外，所考虑的MLPU模型马修相关系数(MCC)从中位数0.92下降到0.02，表明基线MLPU模型目前的形式是不可靠的。此外，我们的发现发现了这些系统的几个安全漏洞，并为研究人员设计可靠和安全的MLPU系统提供了未来的方向。



## **36. Post-breach Recovery: Protection against White-box Adversarial Examples for Leaked DNN Models**

漏洞后恢复：针对泄漏的DNN模型的白盒对抗示例 cs.CR

**SubmitDate**: 2022-10-16    [abs](http://arxiv.org/abs/2205.10686v2) [paper-pdf](http://arxiv.org/pdf/2205.10686v2)

**Authors**: Shawn Shan, Wenxin Ding, Emily Wenger, Haitao Zheng, Ben Y. Zhao

**Abstract**: Server breaches are an unfortunate reality on today's Internet. In the context of deep neural network (DNN) models, they are particularly harmful, because a leaked model gives an attacker "white-box" access to generate adversarial examples, a threat model that has no practical robust defenses. For practitioners who have invested years and millions into proprietary DNNs, e.g. medical imaging, this seems like an inevitable disaster looming on the horizon.   In this paper, we consider the problem of post-breach recovery for DNN models. We propose Neo, a new system that creates new versions of leaked models, alongside an inference time filter that detects and removes adversarial examples generated on previously leaked models. The classification surfaces of different model versions are slightly offset (by introducing hidden distributions), and Neo detects the overfitting of attacks to the leaked model used in its generation. We show that across a variety of tasks and attack methods, Neo is able to filter out attacks from leaked models with very high accuracy, and provides strong protection (7--10 recoveries) against attackers who repeatedly breach the server. Neo performs well against a variety of strong adaptive attacks, dropping slightly in # of breaches recoverable, and demonstrates potential as a complement to DNN defenses in the wild.

摘要: 在当今的互联网上，服务器入侵是一个不幸的现实。在深度神经网络(DNN)模型的背景下，它们尤其有害，因为泄露的模型让攻击者可以使用“白盒”来生成对抗性示例，这是一种没有实际可靠防御措施的威胁模型。对于在专有DNN(例如医学成像)上投入多年和数百万美元的从业者来说，这似乎是一场不可避免的灾难迫在眉睫。在本文中，我们考虑了DNN模型的漏洞后恢复问题。我们提出了Neo，一个新的系统，它创建新版本的泄漏模型，以及一个推理时间过滤器，检测并删除在以前泄漏的模型上生成的敌对示例。不同模型版本的分类面略有偏移(通过引入隐藏分布)，并且Neo检测到对其生成中使用的泄漏模型的攻击过拟合。我们发现，在各种任务和攻击方法中，Neo能够以非常高的准确率从泄露的模型中过滤攻击，并针对反复破坏服务器的攻击者提供强大的保护(7-10次恢复)。NEO在各种强自适应攻击中表现良好，在可恢复的漏洞数量中略有下降，并显示出在野外作为DNN防御的补充潜力。



## **37. Robust Feature-Level Adversaries are Interpretability Tools**

强大的功能级对手是可解释的工具 cs.LG

Code available at  https://github.com/thestephencasper/feature_level_adv

**SubmitDate**: 2022-10-16    [abs](http://arxiv.org/abs/2110.03605v5) [paper-pdf](http://arxiv.org/pdf/2110.03605v5)

**Authors**: Stephen Casper, Max Nadeau, Dylan Hadfield-Menell, Gabriel Kreiman

**Abstract**: The literature on adversarial attacks in computer vision typically focuses on pixel-level perturbations. These tend to be very difficult to interpret. Recent work that manipulates the latent representations of image generators to create "feature-level" adversarial perturbations gives us an opportunity to explore perceptible, interpretable adversarial attacks. We make three contributions. First, we observe that feature-level attacks provide useful classes of inputs for studying representations in models. Second, we show that these adversaries are versatile and highly robust. We demonstrate that they can be used to produce targeted, universal, disguised, physically-realizable, and black-box attacks at the ImageNet scale. Third, we show how these adversarial images can be used as a practical interpretability tool for identifying bugs in networks. We use these adversaries to make predictions about spurious associations between features and classes which we then test by designing "copy/paste" attacks in which one natural image is pasted into another to cause a targeted misclassification. Our results suggest that feature-level attacks are a promising approach for rigorous interpretability research. They support the design of tools to better understand what a model has learned and diagnose brittle feature associations. Code is available at https://github.com/thestephencasper/feature_level_adv

摘要: 关于计算机视觉中的对抗性攻击的文献通常集中在像素级的扰动上。这些往往很难解释。最近的工作是利用图像生成器的潜在表示来创建“特征级别”的对抗性扰动，这给了我们一个探索可感知的、可解释的对抗性攻击的机会。我们有三点贡献。首先，我们观察到特征级别的攻击为学习模型中的表示提供了有用的输入类。其次，我们证明了这些对手是多才多艺的，并且非常健壮。我们证明了它们可以用于在ImageNet规模上产生有针对性的、普遍的、伪装的、物理上可实现的和黑匣子攻击。第三，我们展示了如何将这些对抗性图像用作识别网络漏洞的实用可解释性工具。我们利用这些对手来预测特征和类别之间的虚假关联，然后通过设计“复制/粘贴”攻击来测试这些关联，在这种攻击中，一幅自然图像被粘贴到另一幅图像中，从而导致有针对性的误分类。我们的结果表明，特征级攻击对于严格的可解释性研究是一种很有前途的方法。它们支持工具的设计，以更好地理解模型学习到的内容并诊断脆弱的特征关联。代码可在https://github.com/thestephencasper/feature_level_adv上找到



## **38. Nowhere to Hide: A Lightweight Unsupervised Detector against Adversarial Examples**

无处藏身：针对敌意例子的轻量级无监督检测器 cs.LG

**SubmitDate**: 2022-10-16    [abs](http://arxiv.org/abs/2210.08579v1) [paper-pdf](http://arxiv.org/pdf/2210.08579v1)

**Authors**: Hui Liu, Bo Zhao, Kehuan Zhang, Peng Liu

**Abstract**: Although deep neural networks (DNNs) have shown impressive performance on many perceptual tasks, they are vulnerable to adversarial examples that are generated by adding slight but maliciously crafted perturbations to benign images. Adversarial detection is an important technique for identifying adversarial examples before they are entered into target DNNs. Previous studies to detect adversarial examples either targeted specific attacks or required expensive computation. How design a lightweight unsupervised detector is still a challenging problem. In this paper, we propose an AutoEncoder-based Adversarial Examples (AEAE) detector, that can guard DNN models by detecting adversarial examples with low computation in an unsupervised manner. The AEAE includes only a shallow autoencoder but plays two roles. First, a well-trained autoencoder has learned the manifold of benign examples. This autoencoder can produce a large reconstruction error for adversarial images with large perturbations, so we can detect significantly perturbed adversarial examples based on the reconstruction error. Second, the autoencoder can filter out the small noise and change the DNN's prediction on adversarial examples with small perturbations. It helps to detect slightly perturbed adversarial examples based on the prediction distance. To cover these two cases, we utilize the reconstruction error and prediction distance from benign images to construct a two-tuple feature set and train an adversarial detector using the isolation forest algorithm. We show empirically that the AEAE is unsupervised and inexpensive against the most state-of-the-art attacks. Through the detection in these two cases, there is nowhere to hide adversarial examples.

摘要: 尽管深度神经网络(DNN)在许多感知任务中表现出了令人印象深刻的性能，但它们很容易受到通过向良性图像添加轻微但恶意制作的扰动而产生的敌意示例。敌意检测是在敌意实例进入目标DNN之前识别它们的一项重要技术。以前检测对抗性例子的研究要么是针对特定攻击，要么是需要昂贵的计算。如何设计一个轻量级的无监督检测器仍然是一个具有挑战性的问题。本文提出了一种基于自动编码器的对抗性实例检测器(AEAE)，该检测器能够以无监督的方式以较低的计算量检测敌意实例，从而保护DNN模型。AEAE只包括一个浅层自动编码器，但它扮演着两个角色。首先，训练有素的自动编码器已经学会了大量良性的例子。该自动编码器对扰动较大的对抗性图像会产生较大的重建误差，因此可以根据重建误差检测出明显扰动的对抗性实例。其次，自动编码器可以滤除小噪声，并在小扰动的情况下改变DNN对对抗性样本的预测。它有助于基于预测距离检测略微扰动的敌意示例。为了覆盖这两种情况，我们利用良性图像的重建误差和预测距离来构造一个二元组特征集，并使用隔离森林算法来训练敌意检测器。我们的经验表明，AEAE在应对最先进的攻击时是无人监督的，而且成本低廉。通过对这两起案件的侦破，对抗性事例无处藏身。



## **39. Object-Attentional Untargeted Adversarial Attack**

目标--注意的非目标对抗性攻击 cs.CV

**SubmitDate**: 2022-10-16    [abs](http://arxiv.org/abs/2210.08472v1) [paper-pdf](http://arxiv.org/pdf/2210.08472v1)

**Authors**: Chao Zhou, Yuan-Gen Wang, Guopu Zhu

**Abstract**: Deep neural networks are facing severe threats from adversarial attacks. Most existing black-box attacks fool target model by generating either global perturbations or local patches. However, both global perturbations and local patches easily cause annoying visual artifacts in adversarial example. Compared with some smooth regions of an image, the object region generally has more edges and a more complex texture. Thus small perturbations on it will be more imperceptible. On the other hand, the object region is undoubtfully the decisive part of an image to classification tasks. Motivated by these two facts, we propose an object-attentional adversarial attack method for untargeted attack. Specifically, we first generate an object region by intersecting the object detection region from YOLOv4 with the salient object detection (SOD) region from HVPNet. Furthermore, we design an activation strategy to avoid the reaction caused by the incomplete SOD. Then, we perform an adversarial attack only on the detected object region by leveraging Simple Black-box Adversarial Attack (SimBA). To verify the proposed method, we create a unique dataset by extracting all the images containing the object defined by COCO from ImageNet-1K, named COCO-Reduced-ImageNet in this paper. Experimental results on ImageNet-1K and COCO-Reduced-ImageNet show that under various system settings, our method yields the adversarial example with better perceptual quality meanwhile saving the query budget up to 24.16\% compared to the state-of-the-art approaches including SimBA.

摘要: 深度神经网络正面临着来自对手攻击的严重威胁。现有的大多数黑盒攻击通过产生全局扰动或局部补丁来愚弄目标模型。然而，在对抗性例子中，全局扰动和局部斑块都容易造成令人讨厌的视觉伪影。与图像的一些平滑区域相比，目标区域通常具有更多的边缘和更复杂的纹理。因此，对它的微小扰动将更加难以察觉。另一方面，目标区域无疑是一幅图像进行分类任务的决定性部分。受这两个事实的启发，我们提出了一种针对非定向攻击的对象注意对抗性攻击方法。具体地，我们首先通过将来自YOLOv4的目标检测区域与来自HVPNet的显著目标检测(SOD)区域相交来生成目标区域。此外，我们设计了一种激活策略，以避免由于不完整的SOD而引起的反应。然后，我们利用简单黑盒对抗攻击(Simba)只对检测到的目标区域执行对抗性攻击。为了验证所提出的方法，我们从ImageNet-1K中提取包含CoCo定义的对象的所有图像来创建唯一的数据集，本文将其命名为CoCo-Reduced-ImageNet。在ImageNet-1K和Coco-Reduced-ImageNet上的实验结果表明，在不同的系统设置下，我们的方法生成的对抗性实例具有更好的感知质量，同时与包括SIMBA在内的最新方法相比，可以节省高达24.16\%的查询预算。



## **40. RoS-KD: A Robust Stochastic Knowledge Distillation Approach for Noisy Medical Imaging**

ROS-KD：一种适用于噪声医学成像的稳健随机知识提取方法 cs.CV

Accepted in ICDM 2022

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2210.08388v1) [paper-pdf](http://arxiv.org/pdf/2210.08388v1)

**Authors**: Ajay Jaiswal, Kumar Ashutosh, Justin F Rousseau, Yifan Peng, Zhangyang Wang, Ying Ding

**Abstract**: AI-powered Medical Imaging has recently achieved enormous attention due to its ability to provide fast-paced healthcare diagnoses. However, it usually suffers from a lack of high-quality datasets due to high annotation cost, inter-observer variability, human annotator error, and errors in computer-generated labels. Deep learning models trained on noisy labelled datasets are sensitive to the noise type and lead to less generalization on the unseen samples. To address this challenge, we propose a Robust Stochastic Knowledge Distillation (RoS-KD) framework which mimics the notion of learning a topic from multiple sources to ensure deterrence in learning noisy information. More specifically, RoS-KD learns a smooth, well-informed, and robust student manifold by distilling knowledge from multiple teachers trained on overlapping subsets of training data. Our extensive experiments on popular medical imaging classification tasks (cardiopulmonary disease and lesion classification) using real-world datasets, show the performance benefit of RoS-KD, its ability to distill knowledge from many popular large networks (ResNet-50, DenseNet-121, MobileNet-V2) in a comparatively small network, and its robustness to adversarial attacks (PGD, FSGM). More specifically, RoS-KD achieves >2% and >4% improvement on F1-score for lesion classification and cardiopulmonary disease classification tasks, respectively, when the underlying student is ResNet-18 against recent competitive knowledge distillation baseline. Additionally, on cardiopulmonary disease classification task, RoS-KD outperforms most of the SOTA baselines by ~1% gain in AUC score.

摘要: 人工智能支持的医学成像最近获得了极大的关注，因为它能够提供快节奏的医疗诊断。然而，由于高昂的注释成本、观察者之间的可变性、人为注释员错误以及计算机生成的标签中的错误，它通常缺乏高质量的数据集。在有噪声标记的数据集上训练的深度学习模型对噪声类型敏感，导致对不可见样本的泛化程度较低。为了应对这一挑战，我们提出了一个稳健的随机知识蒸馏(ROS-KD)框架，它模仿了从多个来源学习一个主题的概念，以确保在学习噪声信息时具有威慑作用。更具体地说，ROS-KD通过从多个教师那里提取知识来学习流畅、消息灵通和健壮的学生流形，这些知识来自于在重叠的训练数据子集上训练的多个教师。我们使用真实世界的数据集对流行的医学图像分类任务(心肺疾病和病变分类)进行了广泛的实验，表明了Ros-KD的性能优势，它能够在相对较小的网络中从许多流行的大型网络(ResNet-50，DenseNet-121，MobileNet-V2)中提取知识，以及它对对手攻击(PGD，FSGM)的鲁棒性。更具体地说，当基础学生是ResNet-18而不是最近的竞争性知识蒸馏基线时，ROS-KD在病变分类和心肺疾病分类任务中分别比F1分数提高>2%和>4%。此外，在心肺疾病分类任务上，ROS-KD在AUC评分上比大多数SOTA基线高出约1%。



## **41. GAMA: Generative Adversarial Multi-Object Scene Attacks**

GAMA：生成性对抗性多目标场景攻击 cs.CV

Accepted at NeurIPS 2022; First two authors contributed equally;  Includes Supplementary Material

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2209.09502v2) [paper-pdf](http://arxiv.org/pdf/2209.09502v2)

**Authors**: Abhishek Aich, Calvin-Khang Ta, Akash Gupta, Chengyu Song, Srikanth V. Krishnamurthy, M. Salman Asif, Amit K. Roy-Chowdhury

**Abstract**: The majority of methods for crafting adversarial attacks have focused on scenes with a single dominant object (e.g., images from ImageNet). On the other hand, natural scenes include multiple dominant objects that are semantically related. Thus, it is crucial to explore designing attack strategies that look beyond learning on single-object scenes or attack single-object victim classifiers. Due to their inherent property of strong transferability of perturbations to unknown models, this paper presents the first approach of using generative models for adversarial attacks on multi-object scenes. In order to represent the relationships between different objects in the input scene, we leverage upon the open-sourced pre-trained vision-language model CLIP (Contrastive Language-Image Pre-training), with the motivation to exploit the encoded semantics in the language space along with the visual space. We call this attack approach Generative Adversarial Multi-object scene Attacks (GAMA). GAMA demonstrates the utility of the CLIP model as an attacker's tool to train formidable perturbation generators for multi-object scenes. Using the joint image-text features to train the generator, we show that GAMA can craft potent transferable perturbations in order to fool victim classifiers in various attack settings. For example, GAMA triggers ~16% more misclassification than state-of-the-art generative approaches in black-box settings where both the classifier architecture and data distribution of the attacker are different from the victim. Our code is available here: https://abhishekaich27.github.io/gama.html

摘要: 大多数制作敌意攻击的方法都集中在具有单一主导对象的场景(例如，来自ImageNet的图像)。另一方面，自然场景包括多个语义相关的主导对象。因此，探索设计超越学习单对象场景或攻击单对象受害者分类器的攻击策略是至关重要的。由于产生式模型对未知模型具有很强的可转移性，本文首次提出了利用产生式模型进行多目标场景对抗性攻击的方法。为了表示输入场景中不同对象之间的关系，我们利用开源的预先训练的视觉语言模型剪辑(Contrastive Language-Image Pre-Training)，目的是利用语言空间和视觉空间中的编码语义。我们称这种攻击方式为生成性对抗性多对象场景攻击(GAMA)。GAMA演示了剪辑模型作为攻击者的工具的效用，以训练用于多对象场景的强大的扰动生成器。使用联合图文特征训练生成器，我们证明了GAMA能够在不同的攻击环境下制造有效的可转移扰动来愚弄受害者分类器。例如，在攻击者的分类器体系结构和数据分布都与受害者不同的黑盒环境中，GAMA触发的错误分类方法比最先进的生成性方法高出约16%。我们的代码可在此处获得：https://abhishekaich27.github.io/gama.html



## **42. A Scalable Reinforcement Learning Approach for Attack Allocation in Swarm to Swarm Engagement Problems**

群对群交战问题中攻击分配的可扩展强化学习方法 cs.RO

submitted to ICRA 2023

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2210.08319v1) [paper-pdf](http://arxiv.org/pdf/2210.08319v1)

**Authors**: Umut Demir, Nazim Kemal Ure

**Abstract**: In this work we propose a reinforcement learning (RL) framework that controls the density of a large-scale swarm for engaging with adversarial swarm attacks. Although there is a significant amount of existing work in applying artificial intelligence methods to swarm control, analysis of interactions between two adversarial swarms is a rather understudied area. Most of the existing work in this subject develop strategies by making hard assumptions regarding the strategy and dynamics of the adversarial swarm. Our main contribution is the formulation of the swarm to swarm engagement problem as a Markov Decision Process and development of RL algorithms that can compute engagement strategies without the knowledge of strategy/dynamics of the adversarial swarm. Simulation results show that the developed framework can handle a wide array of large-scale engagement scenarios in an efficient manner.

摘要: 在这项工作中，我们提出了一种强化学习(RL)框架，用于控制大规模群的密度，以应对对抗性的群攻击。虽然现有的大量工作是应用人工智能方法来控制种群，但分析两个敌对种群之间的相互作用是一个相当不充分的研究领域。本学科现有的大多数工作都是通过对敌方蜂群的战略和动态做出硬假设来制定战略的。我们的主要贡献是将群到群的参与问题描述为一个马尔可夫决策过程，并开发了RL算法，该算法可以在不了解对手群的策略/动态的情况下计算参与策略。仿真结果表明，该框架能够高效地处理多种大规模交战场景。



## **43. Robust Binary Models by Pruning Randomly-initialized Networks**

剪枝随机初始化网络的稳健二进制模型 cs.LG

Accepted as NeurIPS 2022 paper

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2202.01341v2) [paper-pdf](http://arxiv.org/pdf/2202.01341v2)

**Authors**: Chen Liu, Ziqi Zhao, Sabine Süsstrunk, Mathieu Salzmann

**Abstract**: Robustness to adversarial attacks was shown to require a larger model capacity, and thus a larger memory footprint. In this paper, we introduce an approach to obtain robust yet compact models by pruning randomly-initialized binary networks. Unlike adversarial training, which learns the model parameters, we initialize the model parameters as either +1 or -1, keep them fixed, and find a subnetwork structure that is robust to attacks. Our method confirms the Strong Lottery Ticket Hypothesis in the presence of adversarial attacks, and extends this to binary networks. Furthermore, it yields more compact networks with competitive performance than existing works by 1) adaptively pruning different network layers; 2) exploiting an effective binary initialization scheme; 3) incorporating a last batch normalization layer to improve training stability. Our experiments demonstrate that our approach not only always outperforms the state-of-the-art robust binary networks, but also can achieve accuracy better than full-precision ones on some datasets. Finally, we show the structured patterns of our pruned binary networks.

摘要: 对敌意攻击的稳健性被证明需要更大的模型容量，因此需要更大的内存占用。在本文中，我们介绍了一种通过剪枝随机初始化的二进制网络来获得健壮而紧凑的模型的方法。与学习模型参数的对抗性训练不同，我们将模型参数初始化为+1或-1，保持其固定，并找到一个对攻击具有鲁棒性的子网络结构。我们的方法证实了存在对抗性攻击时的强彩票假设，并将其推广到二进制网络。此外，它通过1)自适应剪枝不同的网络层；2)采用有效的二进制初始化方案；3)加入最后一批归一化层来提高训练稳定性，从而产生了比现有工作更紧凑、性能更具竞争力的网络。我们的实验表明，我们的方法不仅在性能上总是优于最先进的健壮的二进制网络，而且在一些数据集上可以达到比全精度方法更好的准确率。最后，我们展示了我们的剪枝二进制网络的结构模式。



## **44. Overparameterization from Computational Constraints**

计算约束的超参数化 cs.LG

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2208.12926v2) [paper-pdf](http://arxiv.org/pdf/2208.12926v2)

**Authors**: Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Mingyuan Wang

**Abstract**: Overparameterized models with millions of parameters have been hugely successful. In this work, we ask: can the need for large models be, at least in part, due to the \emph{computational} limitations of the learner? Additionally, we ask, is this situation exacerbated for \emph{robust} learning? We show that this indeed could be the case. We show learning tasks for which computationally bounded learners need \emph{significantly more} model parameters than what information-theoretic learners need. Furthermore, we show that even more model parameters could be necessary for robust learning. In particular, for computationally bounded learners, we extend the recent result of Bubeck and Sellke [NeurIPS'2021] which shows that robust models might need more parameters, to the computational regime and show that bounded learners could provably need an even larger number of parameters. Then, we address the following related question: can we hope to remedy the situation for robust computationally bounded learning by restricting \emph{adversaries} to also be computationally bounded for sake of obtaining models with fewer parameters? Here again, we show that this could be possible. Specifically, building on the work of Garg, Jha, Mahloujifar, and Mahmoody [ALT'2020], we demonstrate a learning task that can be learned efficiently and robustly against a computationally bounded attacker, while to be robust against an information-theoretic attacker requires the learner to utilize significantly more parameters.

摘要: 具有数百万个参数的过度参数模型已经取得了巨大的成功。在这项工作中，我们问：对大型模型的需求是否至少部分是由于学习者的计算限制？此外，我们问，这种情况是否会因为学习而加剧？我们表明，情况确实可能是这样的。我们展示了计算受限的学习者比信息论学习者需要更多的模型参数的学习任务。此外，我们还表明，稳健学习可能需要更多的模型参数。特别是，对于计算有界的学习者，我们将Bubeck和Sellke[NeurIPS‘2021]的最新结果推广到计算机制，该结果表明健壮模型可能需要更多的参数，并表明有界的学习者可能需要更多的参数。然后，我们解决了以下相关问题：为了获得参数更少的模型，我们是否可以通过限制对手也是计算有界的来纠正健壮的计算有界学习的情况？在这里，我们再次证明了这是可能的。具体地说，在Garg，Jha，MahLoujifar和Mahmoody[Alt‘2020]的工作基础上，我们演示了一种学习任务，该任务可以在计算受限的攻击者面前高效而稳健地学习，而为了对信息论攻击者具有健壮性，需要学习者使用更多的参数。



## **45. Distributionally Robust Multiclass Classification and Applications in Deep Image Classifiers**

分布稳健多类分类及其在深度图像分类器中的应用 cs.CV

arXiv admin note: substantial text overlap with arXiv:2109.12772

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2210.08198v1) [paper-pdf](http://arxiv.org/pdf/2210.08198v1)

**Authors**: Ruidi Chen, Boran Hao, Ioannis Ch. Paschalidis

**Abstract**: We develop a Distributionally Robust Optimization (DRO) formulation for Multiclass Logistic Regression (MLR), which could tolerate data contaminated by outliers. The DRO framework uses a probabilistic ambiguity set defined as a ball of distributions that are close to the empirical distribution of the training set in the sense of the Wasserstein metric. We relax the DRO formulation into a regularized learning problem whose regularizer is a norm of the coefficient matrix. We establish out-of-sample performance guarantees for the solutions to our model, offering insights on the role of the regularizer in controlling the prediction error. We apply the proposed method in rendering deep Vision Transformer (ViT)-based image classifiers robust to random and adversarial attacks. Specifically, using the MNIST and CIFAR-10 datasets, we demonstrate reductions in test error rate by up to 83.5% and loss by up to 91.3% compared with baseline methods, by adopting a novel random training method.

摘要: 针对多分类Logistic回归(MLR)模型，提出了一种分布式稳健优化(DRO)方法，该方法能够容忍数据中的离群点污染。DRO框架使用概率模糊集，该模糊集被定义为在Wasserstein度量意义上接近训练集的经验分布的分布球。我们将DRO公式松弛为一个正则化学习问题，其正则化子是系数矩阵的范数。我们为我们的模型的解建立了样本外性能保证，提供了关于正则化在控制预测误差中的作用的见解。我们将该方法应用于基于深度视觉转换器(VIT)的图像分类器对随机攻击和敌意攻击的稳健性。具体地说，使用MNIST和CIFAR-10数据集，我们证明了通过采用一种新的随机训练方法，与基线方法相比，测试错误率降低了83.5%，损失降低了91.3%。



## **46. Dynamics-aware Adversarial Attack of Adaptive Neural Networks**

自适应神经网络的动态感知敌意攻击 cs.CV

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2210.08159v1) [paper-pdf](http://arxiv.org/pdf/2210.08159v1)

**Authors**: An Tao, Yueqi Duan, Yingqi Wang, Jiwen Lu, Jie Zhou

**Abstract**: In this paper, we investigate the dynamics-aware adversarial attack problem of adaptive neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed adaptive neural networks, which adaptively deactivate unnecessary execution units based on inputs to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture change afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we reformulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on representative types of adaptive neural networks for both 2D images and 3D point clouds show that our LGM achieves impressive adversarial attack performance compared with the dynamic-unaware attack methods.

摘要: 本文研究了自适应神经网络的动态感知对抗攻击问题。大多数现有的对抗性攻击算法都是在一个基本假设下设计的--网络体系结构在整个攻击过程中是固定的。然而，这一假设对最近提出的许多自适应神经网络并不成立，这些自适应神经网络基于输入自适应地停用不必要的执行单元来提高计算效率。它导致了严重的梯度滞后问题，使得当前步骤的学习攻击由于之后的体系结构变化而无效。为了解决这个问题，我们提出了一种引导梯度法(LGM)，并展示了滞后梯度的显著影响。更具体地说，我们重新制定了梯度，以了解网络体系结构的潜在动态变化，以便在网络体系结构动态变化时，学习到的攻击比不知道动态变化的方法更好地“引导”下一步。在典型的自适应神经网络上对2D图像和3D点云进行的大量实验表明，与动态未知攻击方法相比，我们的LGM具有令人印象深刻的对抗性攻击性能。



## **47. Certified Robustness Against Natural Language Attacks by Causal Intervention**

通过因果干预验证对自然语言攻击的健壮性 cs.LG

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2205.12331v3) [paper-pdf](http://arxiv.org/pdf/2205.12331v3)

**Authors**: Haiteng Zhao, Chang Ma, Xinshuai Dong, Anh Tuan Luu, Zhi-Hong Deng, Hanwang Zhang

**Abstract**: Deep learning models have achieved great success in many fields, yet they are vulnerable to adversarial examples. This paper follows a causal perspective to look into the adversarial vulnerability and proposes Causal Intervention by Semantic Smoothing (CISS), a novel framework towards robustness against natural language attacks. Instead of merely fitting observational data, CISS learns causal effects p(y|do(x)) by smoothing in the latent semantic space to make robust predictions, which scales to deep architectures and avoids tedious construction of noise customized for specific attacks. CISS is provably robust against word substitution attacks, as well as empirically robust even when perturbations are strengthened by unknown attack algorithms. For example, on YELP, CISS surpasses the runner-up by 6.7% in terms of certified robustness against word substitutions, and achieves 79.4% empirical robustness when syntactic attacks are integrated.

摘要: 深度学习模型在许多领域都取得了很大的成功，但它们很容易受到对手例子的影响。本文从因果关系的角度分析了敌意攻击的脆弱性，提出了通过语义平滑进行因果干预的方法，这是一种新的针对自然语言攻击的健壮性框架。与其仅仅对观测数据进行拟合，CIS通过在潜在语义空间中进行平滑以做出稳健的预测来学习因果效应p(y|do(X))，该预测可扩展到深层体系结构，并避免针对特定攻击定制的乏味的噪声构造。可以证明，该系统对单词替换攻击具有较强的健壮性，即使在未知攻击算法加强了扰动的情况下，也具有较强的经验性。例如，在Yelp上，在对单词替换的验证健壮性方面，CISS超过亚军6.7%，并且在整合句法攻击时获得了79.4%的经验健壮性。



## **48. SealClub: Computer-aided Paper Document Authentication**

SealClub：计算机辅助纸质文档认证 cs.CR

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2210.07884v1) [paper-pdf](http://arxiv.org/pdf/2210.07884v1)

**Authors**: Martín Ochoa, Jorge Toro-Pozo, David Basin

**Abstract**: Digital authentication is a mature field, offering a range of solutions with rigorous mathematical guarantees. Nevertheless, paper documents, where cryptographic techniques are not directly applicable, are still widely utilized due to usability and legal reasons. We propose a novel approach to authenticating paper documents using smartphones by taking short videos of them. Our solution combines cryptographic and image comparison techniques to detect and highlight subtle semantic-changing attacks on rich documents, containing text and graphics, that could go unnoticed by humans. We rigorously analyze our approach, proving that it is secure against strong adversaries capable of compromising different system components. We also measure its accuracy empirically on a set of 128 videos of paper documents, half containing subtle forgeries. Our algorithm finds all forgeries accurately (no false alarms) after analyzing 5.13 frames on average (corresponding to 1.28 seconds of video). Highlighted regions are large enough to be visible to users, but small enough to precisely locate forgeries. Thus, our approach provides a promising way for users to authenticate paper documents using conventional smartphones under realistic conditions.

摘要: 数字身份验证是一个成熟的领域，提供了一系列具有严格数学保证的解决方案。然而，由于可用性和法律原因，在加密技术不直接适用的情况下，纸质文档仍然被广泛使用。我们提出了一种通过拍摄短视频来使用智能手机对纸质文档进行身份验证的新方法。我们的解决方案结合了加密和图像比较技术，以检测和突出对包含文本和图形的丰富文档的细微语义变化攻击，这些攻击可能不会被人类注意到。我们严格分析了我们的方法，证明了它是安全的，可以抵御能够危害不同系统组件的强大对手。我们还在一组128个纸质文档的视频上对其准确性进行了经验性的测量，其中一半包含微妙的伪造。该算法在平均分析5.13帧(对应于1.28秒的视频)后，准确地发现了所有的伪造(没有虚警)。突出显示的区域足够大，用户可以看到，但也足够小，可以精确定位假货。因此，我们的方法为用户在现实条件下使用传统的智能手机认证纸质文档提供了一种很有前途的方法。



## **49. Pre-trained Adversarial Perturbations**

预先训练的对抗性扰动 cs.CV

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2210.03372v2) [paper-pdf](http://arxiv.org/pdf/2210.03372v2)

**Authors**: Yuanhao Ban, Yinpeng Dong

**Abstract**: Self-supervised pre-training has drawn increasing attention in recent years due to its superior performance on numerous downstream tasks after fine-tuning. However, it is well-known that deep learning models lack the robustness to adversarial examples, which can also invoke security issues to pre-trained models, despite being less explored. In this paper, we delve into the robustness of pre-trained models by introducing Pre-trained Adversarial Perturbations (PAPs), which are universal perturbations crafted for the pre-trained models to maintain the effectiveness when attacking fine-tuned ones without any knowledge of the downstream tasks. To this end, we propose a Low-Level Layer Lifting Attack (L4A) method to generate effective PAPs by lifting the neuron activations of low-level layers of the pre-trained models. Equipped with an enhanced noise augmentation strategy, L4A is effective at generating more transferable PAPs against fine-tuned models. Extensive experiments on typical pre-trained vision models and ten downstream tasks demonstrate that our method improves the attack success rate by a large margin compared with state-of-the-art methods.

摘要: 近年来，自监督预训练因其在经过微调后在众多下游任务中的优异表现而受到越来越多的关注。然而，众所周知，深度学习模型缺乏对敌意示例的健壮性，这也可能会引发预先训练的模型的安全问题，尽管研究较少。在本文中，我们通过引入预训练对抗扰动(PAP)来深入研究预训练模型的稳健性，PAP是为预训练模型设计的通用扰动，用于在攻击精调模型时保持有效性，而不需要了解下游任务。为此，我们提出了一种低层提升攻击(L4A)的方法，通过提升预训练模型低层神经元的激活来生成有效的PAP。配备了增强的噪音增强策略，L4A在针对微调模型生成更多可转移的PAP方面是有效的。在典型的预训练视觉模型和十个下游任务上的大量实验表明，该方法与现有方法相比，攻击成功率有较大幅度的提高。



## **50. Generative Adversarial Learning for Trusted and Secure Clustering in Industrial Wireless Sensor Networks**

工业无线传感器网络可信安全分簇的生成性对抗性学习 cs.NI

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2210.07707v1) [paper-pdf](http://arxiv.org/pdf/2210.07707v1)

**Authors**: Liu Yang, Simon X. Yang, Yun Li, Yinzhi Lu, Tan Guo

**Abstract**: Traditional machine learning techniques have been widely used to establish the trust management systems. However, the scale of training dataset can significantly affect the security performances of the systems, while it is a great challenge to detect malicious nodes due to the absence of labeled data regarding novel attacks. To address this issue, this paper presents a generative adversarial network (GAN) based trust management mechanism for Industrial Wireless Sensor Networks (IWSNs). First, type-2 fuzzy logic is adopted to evaluate the reputation of sensor nodes while alleviating the uncertainty problem. Then, trust vectors are collected to train a GAN-based codec structure, which is used for further malicious node detection. Moreover, to avoid normal nodes being isolated from the network permanently due to error detections, a GAN-based trust redemption model is constructed to enhance the resilience of trust management. Based on the latest detection results, a trust model update method is developed to adapt to the dynamic industrial environment. The proposed trust management mechanism is finally applied to secure clustering for reliable and real-time data transmission, and simulation results show that it achieves a high detection rate up to 96%, as well as a low false positive rate below 8%.

摘要: 传统的机器学习技术已被广泛应用于建立信任管理系统。然而，训练数据集的规模会显著影响系统的安全性能，同时由于缺乏关于新攻击的标记数据，检测恶意节点是一个巨大的挑战。针对这一问题，提出了一种基于产生式对抗网络的工业无线传感器网络信任管理机制。首先，在缓解不确定性问题的同时，采用二型模糊逻辑对传感器节点的信誉度进行评估。然后，收集信任向量训练基于GAN的编解码器结构，用于进一步的恶意节点检测。此外，为了避免正常节点因错误检测而与网络永久隔离，构建了基于GAN的信任赎回模型，增强了信任管理的弹性。基于最新的检测结果，提出了一种适应动态工业环境的信任模型更新方法。最后将提出的信任管理机制应用于安全分簇，实现了可靠、实时的数据传输，仿真结果表明，该信任管理机制的检测率高达96%，误检率低于8%。



