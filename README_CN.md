# Latest Adversarial Attack Papers
**update at 2022-05-03 06:31:32**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. To Trust or Not To Trust Prediction Scores for Membership Inference Attacks**

信任还是不信任成员关系推断攻击的预测分数 cs.LG

15 pages, 8 figures, 10 tables

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2111.09076v2)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Kristian Kersting

**Abstracts**: Membership inference attacks (MIAs) aim to determine whether a specific sample was used to train a predictive model. Knowing this may indeed lead to a privacy breach. Most MIAs, however, make use of the model's prediction scores - the probability of each output given some input - following the intuition that the trained model tends to behave differently on its training data. We argue that this is a fallacy for many modern deep network architectures. Consequently, MIAs will miserably fail since overconfidence leads to high false-positive rates not only on known domains but also on out-of-distribution data and implicitly acts as a defense against MIAs. Specifically, using generative adversarial networks, we are able to produce a potentially infinite number of samples falsely classified as part of the training data. In other words, the threat of MIAs is overestimated, and less information is leaked than previously assumed. Moreover, there is actually a trade-off between the overconfidence of models and their susceptibility to MIAs: the more classifiers know when they do not know, making low confidence predictions, the more they reveal the training data.

摘要: 成员关系推理攻击(MIA)的目的是确定特定样本是否被用来训练预测模型。知道这一点确实可能会导致隐私被侵犯。然而，大多数MIA都利用模型的预测分数--每个输出给定一些输入的概率--遵循这样一种直觉，即训练后的模型在其训练数据上往往表现不同。我们认为，对于许多现代深度网络体系结构来说，这是一种谬误。因此，MIA将悲惨地失败，因为过度自信不仅会导致已知域上的高假阳性率，而且还会导致分布外数据的高假阳性率，并隐含地充当对MIA的防御。具体地说，使用生成性对抗性网络，我们能够产生潜在无限数量的样本，这些样本被错误地归类为训练数据的一部分。换句话说，MIA的威胁被高估了，泄露的信息比之前假设的要少。此外，在模型的过度自信和他们对MIA的敏感性之间实际上存在着权衡：分类器知道的越多，他们不知道的时候，做出低置信度预测的人就越多，他们透露的训练数据就越多。



## **2. Finding MNEMON: Reviving Memories of Node Embeddings**

寻找Mnemon：唤醒节点嵌入的记忆 cs.LG

To Appear in the 29th ACM Conference on Computer and Communications  Security (CCS), November 7-11, 2022

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.06963v2)

**Authors**: Yun Shen, Yufei Han, Zhikun Zhang, Min Chen, Ting Yu, Michael Backes, Yang Zhang, Gianluca Stringhini

**Abstracts**: Previous security research efforts orbiting around graphs have been exclusively focusing on either (de-)anonymizing the graphs or understanding the security and privacy issues of graph neural networks. Little attention has been paid to understand the privacy risks of integrating the output from graph embedding models (e.g., node embeddings) with complex downstream machine learning pipelines. In this paper, we fill this gap and propose a novel model-agnostic graph recovery attack that exploits the implicit graph structural information preserved in the embeddings of graph nodes. We show that an adversary can recover edges with decent accuracy by only gaining access to the node embedding matrix of the original graph without interactions with the node embedding models. We demonstrate the effectiveness and applicability of our graph recovery attack through extensive experiments.

摘要: 以前围绕图的安全研究一直专注于图的(去)匿名化或理解图神经网络的安全和隐私问题。很少有人注意到将图嵌入模型(例如，节点嵌入)的输出与复杂的下游机器学习管道集成的隐私风险。在本文中，我们填补了这一空白，并提出了一种新的模型不可知图恢复攻击，该攻击利用了图节点嵌入中保留的隐含的图结构信息。我们证明了敌手只需访问原始图的节点嵌入矩阵，而不需要与节点嵌入模型交互，就能以相当高的精度恢复边。我们通过大量的实验证明了我们的图恢复攻击的有效性和适用性。



## **3. Exploration and Exploitation in Federated Learning to Exclude Clients with Poisoned Data**

联合学习中排除有毒数据客户端的探索与利用 cs.DC

Accepted at 2022 IWCMC

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.14020v1)

**Authors**: Shadha Tabatabai, Ihab Mohammed, Basheer Qolomany, Abdullatif Albasser, Kashif Ahmad, Mohamed Abdallah, Ala Al-Fuqaha

**Abstracts**: Federated Learning (FL) is one of the hot research topics, and it utilizes Machine Learning (ML) in a distributed manner without directly accessing private data on clients. However, FL faces many challenges, including the difficulty to obtain high accuracy, high communication cost between clients and the server, and security attacks related to adversarial ML. To tackle these three challenges, we propose an FL algorithm inspired by evolutionary techniques. The proposed algorithm groups clients randomly in many clusters, each with a model selected randomly to explore the performance of different models. The clusters are then trained in a repetitive process where the worst performing cluster is removed in each iteration until one cluster remains. In each iteration, some clients are expelled from clusters either due to using poisoned data or low performance. The surviving clients are exploited in the next iteration. The remaining cluster with surviving clients is then used for training the best FL model (i.e., remaining FL model). Communication cost is reduced since fewer clients are used in the final training of the FL model. To evaluate the performance of the proposed algorithm, we conduct a number of experiments using FEMNIST dataset and compare the result against the random FL algorithm. The experimental results show that the proposed algorithm outperforms the baseline algorithm in terms of accuracy, communication cost, and security.

摘要: 联合学习(FL)是当前研究的热点之一，它以分布式的方式利用机器学习(ML)，不需要直接访问客户端的私有数据。然而，FL面临着许多挑战，包括难以获得高准确率、客户端与服务器之间的通信成本较高以及与敌意ML相关的安全攻击。为了应对这三个挑战，我们提出了一种受进化技术启发的FL算法。该算法将客户端随机分组到多个簇中，每个簇随机选择一个模型来考察不同模型的性能。然后，在重复过程中训练集群，其中在每次迭代中移除表现最差的集群，直到保留一个集群。在每次迭代中，一些客户端会因使用有毒数据或性能低下而被逐出群集。幸存的客户端将在下一次迭代中被利用。然后，具有幸存客户端的剩余簇用于训练最佳FL模型(即，剩余FL模型)。由于在FL模型的最终训练中使用更少的客户，因此降低了通信成本。为了评估算法的性能，我们使用FEMNIST数据集进行了大量的实验，并将结果与随机FL算法进行了比较。实验结果表明，该算法在准确率、通信开销和安全性方面均优于基线算法。



## **4. Backdoor Attacks in Federated Learning by Rare Embeddings and Gradient Ensembling**

基于稀有嵌入和梯度集成的联合学习后门攻击 cs.LG

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.14017v1)

**Authors**: KiYoon Yoo, Nojun Kwak

**Abstracts**: Recent advances in federated learning have demonstrated its promising capability to learn on decentralized datasets. However, a considerable amount of work has raised concerns due to the potential risks of adversaries participating in the framework to poison the global model for an adversarial purpose. This paper investigates the feasibility of model poisoning for backdoor attacks through \textit{rare word embeddings of NLP models} in text classification and sequence-to-sequence tasks. In text classification, less than 1\% of adversary clients suffices to manipulate the model output without any drop in the performance of clean sentences. For a less complex dataset, a mere 0.1\% of adversary clients is enough to poison the global model effectively. We also propose a technique specialized in the federated learning scheme called gradient ensemble, which enhances the backdoor performance in all experimental settings.

摘要: 联邦学习的最新进展已经证明了它在分散的数据集上学习的前景。然而，由于参与该框架的对手出于对抗目的而破坏全球模式的潜在风险，大量工作引起了关注。通过文本分类和序列到序列任务中的NLP模型的稀有单词嵌入，研究了模型中毒用于后门攻击的可行性。在文本分类中，只有不到1%的敌意客户端足以在不降低干净句子性能的情况下操纵模型输出。对于不太复杂的数据集，仅0.1%的恶意客户端就足以有效地毒化全局模型。我们还提出了一种专门用于联邦学习方案的技术，称为梯度集成，它在所有实验设置中都提高了后门性能。



## **5. Using 3D Shadows to Detect Object Hiding Attacks on Autonomous Vehicle Perception**

利用3D阴影检测自主车辆感知中的目标隐藏攻击 cs.CV

To appear in the Proceedings of the 2022 IEEE Security and Privacy  Workshop on the Internet of Safe Things (SafeThings 2022)

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.13973v1)

**Authors**: Zhongyuan Hau, Soteris Demetriou, Emil C. Lupu

**Abstracts**: Autonomous Vehicles (AVs) are mostly reliant on LiDAR sensors which enable spatial perception of their surroundings and help make driving decisions. Recent works demonstrated attacks that aim to hide objects from AV perception, which can result in severe consequences. 3D shadows, are regions void of measurements in 3D point clouds which arise from occlusions of objects in a scene. 3D shadows were proposed as a physical invariant valuable for detecting spoofed or fake objects. In this work, we leverage 3D shadows to locate obstacles that are hidden from object detectors. We achieve this by searching for void regions and locating the obstacles that cause these shadows. Our proposed methodology can be used to detect an object that has been hidden by an adversary as these objects, while hidden from 3D object detectors, still induce shadow artifacts in 3D point clouds, which we use for obstacle detection. We show that using 3D shadows for obstacle detection can achieve high accuracy in matching shadows to their object and provide precise prediction of an obstacle's distance from the ego-vehicle.

摘要: 自动驾驶汽车(AVs)大多依赖于LiDAR传感器，该传感器能够对周围环境进行空间感知，并帮助做出驾驶决策。最近的研究表明，攻击的目的是隐藏对象，使其不被反病毒感知，这可能会导致严重的后果。3D阴影是由于场景中对象的遮挡而导致的3D点云中没有测量结果的区域。3D阴影被认为是一种物理不变量，对于检测欺骗或虚假对象很有价值。在这项工作中，我们利用3D阴影来定位物体探测器隐藏的障碍物。我们通过搜索空洞区域和定位导致这些阴影的障碍物来实现这一点。我们提出的方法可以用于检测被对手隐藏的对象，因为这些对象虽然隐藏在3D对象检测器之外，但仍然会在3D点云中产生阴影伪影，我们将其用于障碍物检测。结果表明，使用3D阴影进行障碍物检测可以达到较高的匹配精度，并能准确预测障碍物与自行车者之间的距离。



## **6. Detecting Textual Adversarial Examples Based on Distributional Characteristics of Data Representations**

基于数据表示分布特征的文本对抗性实例检测 cs.CL

13 pages, RepL4NLP 2022

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.13853v1)

**Authors**: Na Liu, Mark Dras, Wei Emma Zhang

**Abstracts**: Although deep neural networks have achieved state-of-the-art performance in various machine learning tasks, adversarial examples, constructed by adding small non-random perturbations to correctly classified inputs, successfully fool highly expressive deep classifiers into incorrect predictions. Approaches to adversarial attacks in natural language tasks have boomed in the last five years using character-level, word-level, phrase-level, or sentence-level textual perturbations. While there is some work in NLP on defending against such attacks through proactive methods, like adversarial training, there is to our knowledge no effective general reactive approaches to defence via detection of textual adversarial examples such as is found in the image processing literature. In this paper, we propose two new reactive methods for NLP to fill this gap, which unlike the few limited application baselines from NLP are based entirely on distribution characteristics of learned representations: we adapt one from the image processing literature (Local Intrinsic Dimensionality (LID)), and propose a novel one (MultiDistance Representation Ensemble Method (MDRE)). Adapted LID and MDRE obtain state-of-the-art results on character-level, word-level, and phrase-level attacks on the IMDB dataset as well as on the later two with respect to the MultiNLI dataset. For future research, we publish our code.

摘要: 尽管深度神经网络在各种机器学习任务中取得了最先进的性能，但通过在正确分类的输入中添加微小的非随机扰动而构建的对抗性例子，成功地愚弄了高表达能力的深度分类器，导致了错误的预测。在过去的五年中，自然语言任务中使用字符级别、单词级别、短语级别或句子级别的文本扰动进行对抗性攻击的方法得到了蓬勃发展。虽然NLP中有一些关于通过主动方法来防御这种攻击的工作，如对抗性训练，但据我们所知，没有有效的一般反应性方法来通过检测文本对抗性实例来防御，例如在图像处理文献中找到的。在本文中，我们提出了两种新的反应式方法来填补这一空白，这两种方法不同于NLP的少数有限的应用基线完全基于学习表示的分布特征：我们借鉴了图像处理文献中的一种方法(局部本征维度(LID))，并提出了一种新的方法(多距离表示集成方法(MDRE))。适配的LID和MDRE获得了关于IMDB数据集的字符级、词级和短语级攻击以及关于MultiNLI数据集的后两种攻击的最新结果。为了将来的研究，我们发布了我们的代码。



## **7. DeepAdversaries: Examining the Robustness of Deep Learning Models for Galaxy Morphology Classification**

深度学习：检验深度学习模型对星系形态分类的稳健性 cs.LG

19 pages, 7 figures, 5 tables

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2112.14299v2)

**Authors**: Aleksandra Ćiprijanović, Diana Kafkes, Gregory Snyder, F. Javier Sánchez, Gabriel Nathan Perdue, Kevin Pedro, Brian Nord, Sandeep Madireddy, Stefan M. Wild

**Abstracts**: Data processing and analysis pipelines in cosmological survey experiments introduce data perturbations that can significantly degrade the performance of deep learning-based models. Given the increased adoption of supervised deep learning methods for processing and analysis of cosmological survey data, the assessment of data perturbation effects and the development of methods that increase model robustness are increasingly important. In the context of morphological classification of galaxies, we study the effects of perturbations in imaging data. In particular, we examine the consequences of using neural networks when training on baseline data and testing on perturbed data. We consider perturbations associated with two primary sources: 1) increased observational noise as represented by higher levels of Poisson noise and 2) data processing noise incurred by steps such as image compression or telescope errors as represented by one-pixel adversarial attacks. We also test the efficacy of domain adaptation techniques in mitigating the perturbation-driven errors. We use classification accuracy, latent space visualizations, and latent space distance to assess model robustness. Without domain adaptation, we find that processing pixel-level errors easily flip the classification into an incorrect class and that higher observational noise makes the model trained on low-noise data unable to classify galaxy morphologies. On the other hand, we show that training with domain adaptation improves model robustness and mitigates the effects of these perturbations, improving the classification accuracy by 23% on data with higher observational noise. Domain adaptation also increases by a factor of ~2.3 the latent space distance between the baseline and the incorrectly classified one-pixel perturbed image, making the model more robust to inadvertent perturbations.

摘要: 宇宙学测量实验中的数据处理和分析管道引入了数据扰动，这可能会显著降低基于深度学习的模型的性能。鉴于越来越多的人采用有监督的深度学习方法来处理和分析宇宙学观测数据，评估数据扰动效应和开发提高模型稳健性的方法变得越来越重要。在星系形态分类的背景下，我们研究了成像数据中微扰的影响。特别是，我们检查了在对基线数据进行训练和对扰动数据进行测试时使用神经网络的后果。我们考虑与两个主要来源相关的扰动：1)以更高水平的泊松噪声为代表的观测噪声的增加；2)以单像素对抗性攻击为代表的图像压缩或望远镜误差等步骤所引起的数据处理噪声。我们还测试了领域自适应技术在缓解扰动驱动的错误方面的有效性。我们使用分类精度、潜在空间可视化和潜在空间距离来评估模型的稳健性。在没有域自适应的情况下，我们发现处理像素级误差很容易将分类反转到不正确的类别，并且更高的观测噪声使得基于低噪声数据训练的模型无法对星系形态进行分类。另一方面，我们表明，域自适应训练提高了模型的稳健性，缓解了这些扰动的影响，在观测噪声较高的数据上将分类精度提高了23%。域自适应还将基线和错误分类的单像素扰动图像之间的潜在空间距离增加了约2.3倍，使模型对无意扰动更具鲁棒性。



## **8. Survey and Taxonomy of Adversarial Reconnaissance Techniques**

对抗性侦察技术综述及分类 cs.CR

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2105.04749v2)

**Authors**: Shanto Roy, Nazia Sharmin, Jaime C. Acosta, Christopher Kiekintveld, Aron Laszka

**Abstracts**: Adversaries are often able to penetrate networks and compromise systems by exploiting vulnerabilities in people and systems. The key to the success of these attacks is information that adversaries collect throughout the phases of the cyber kill chain. We summarize and analyze the methods, tactics, and tools that adversaries use to conduct reconnaissance activities throughout the attack process. First, we discuss what types of information adversaries seek, and how and when they can obtain this information. Then, we provide a taxonomy and detailed overview of adversarial reconnaissance techniques. The taxonomy introduces a categorization of reconnaissance techniques based on the source as third-party, human-, and system-based information gathering. This paper provides a comprehensive view of adversarial reconnaissance that can help in understanding and modeling this complex but vital aspect of cyber attacks as well as insights that can improve defensive strategies, such as cyber deception.

摘要: 攻击者通常能够通过利用人和系统中的漏洞来渗透网络并危害系统。这些攻击成功的关键是对手在网络杀伤链的各个阶段收集的信息。我们总结和分析了对手在整个攻击过程中进行侦察活动所使用的方法、战术和工具。首先，我们讨论对手寻求什么类型的信息，以及他们如何以及何时可以获得这些信息。然后，我们对对抗性侦察技术进行了分类和详细的概述。该分类引入了基于来源的侦察技术分类，即第三方、基于人员和基于系统的信息收集。本文提供了对抗性侦察的全面观点，有助于理解和建模网络攻击的这一复杂但至关重要的方面，以及可以改进防御策略的见解，例如网络欺骗。



## **9. AGIC: Approximate Gradient Inversion Attack on Federated Learning**

AGIC：联邦学习中的近似梯度反转攻击 cs.LG

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.13784v1)

**Authors**: Jin Xu, Chi Hong, Jiyue Huang, Lydia Y. Chen, Jérémie Decouchant

**Abstracts**: Federated learning is a private-by-design distributed learning paradigm where clients train local models on their own data before a central server aggregates their local updates to compute a global model. Depending on the aggregation method used, the local updates are either the gradients or the weights of local learning models. Recent reconstruction attacks apply a gradient inversion optimization on the gradient update of a single minibatch to reconstruct the private data used by clients during training. As the state-of-the-art reconstruction attacks solely focus on single update, realistic adversarial scenarios are overlooked, such as observation across multiple updates and updates trained from multiple mini-batches. A few studies consider a more challenging adversarial scenario where only model updates based on multiple mini-batches are observable, and resort to computationally expensive simulation to untangle the underlying samples for each local step. In this paper, we propose AGIC, a novel Approximate Gradient Inversion Attack that efficiently and effectively reconstructs images from both model or gradient updates, and across multiple epochs. In a nutshell, AGIC (i) approximates gradient updates of used training samples from model updates to avoid costly simulation procedures, (ii) leverages gradient/model updates collected from multiple epochs, and (iii) assigns increasing weights to layers with respect to the neural network structure for reconstruction quality. We extensively evaluate AGIC on three datasets, CIFAR-10, CIFAR-100 and ImageNet. Our results show that AGIC increases the peak signal-to-noise ratio (PSNR) by up to 50% compared to two representative state-of-the-art gradient inversion attacks. Furthermore, AGIC is faster than the state-of-the-art simulation based attack, e.g., it is 5x faster when attacking FedAvg with 8 local steps in between model updates.

摘要: 联合学习是一种私人设计的分布式学习范例，其中客户端在中央服务器聚合其本地更新以计算全局模型之前，根据自己的数据训练本地模型。根据所使用的聚合方法，局部更新要么是局部学习模型的梯度，要么是局部学习模型的权重。最近的重建攻击将梯度倒置优化应用于单个小批量的梯度更新，以重建客户在训练期间使用的私有数据。由于最新的重建攻击只关注单个更新，因此忽略了现实的对抗性场景，例如跨多个更新的观察和从多个小批次训练的更新。一些研究考虑了一种更具挑战性的对抗性场景，其中只能观察到基于多个小批次的模型更新，并求助于计算代价高昂的模拟来解开每个局部步骤的潜在样本。在本文中，我们提出了AGIC，一种新的近似梯度反转攻击，它可以高效地从模型或梯度更新中重建图像，并跨越多个历元。简而言之，AGIC(I)根据模型更新近似使用的训练样本的梯度更新以避免昂贵的模拟过程，(Ii)利用从多个历元收集的梯度/模型更新，以及(Iii)为重建质量向层分配相对于神经网络结构的不断增加的权重。我们在三个数据集CIFAR-10、CIFAR-100和ImageNet上对AGIC进行了广泛的评估。实验结果表明，与两种典型的梯度反转攻击相比，AGIC的峰值信噪比(PSNR)提高了50%。此外，AGIC比最先进的基于模拟的攻击速度更快，例如，在模型更新之间有8个本地步骤的情况下，攻击FedAvg的速度要快5倍。



## **10. Formulating Robustness Against Unforeseen Attacks**

针对不可预见的攻击形成健壮性 cs.LG

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.13779v1)

**Authors**: Sihui Dai, Saeed Mahloujifar, Prateek Mittal

**Abstracts**: Existing defenses against adversarial examples such as adversarial training typically assume that the adversary will conform to a specific or known threat model, such as $\ell_p$ perturbations within a fixed budget. In this paper, we focus on the scenario where there is a mismatch in the threat model assumed by the defense during training, and the actual capabilities of the adversary at test time. We ask the question: if the learner trains against a specific "source" threat model, when can we expect robustness to generalize to a stronger unknown "target" threat model during test-time? Our key contribution is to formally define the problem of learning and generalization with an unforeseen adversary, which helps us reason about the increase in adversarial risk from the conventional perspective of a known adversary. Applying our framework, we derive a generalization bound which relates the generalization gap between source and target threat models to variation of the feature extractor, which measures the expected maximum difference between extracted features across a given threat model. Based on our generalization bound, we propose adversarial training with variation regularization (AT-VR) which reduces variation of the feature extractor across the source threat model during training. We empirically demonstrate that AT-VR can lead to improved generalization to unforeseen attacks during test-time compared to standard adversarial training on Gaussian and image datasets.

摘要: 现有的针对对抗性示例的防御，例如对抗性训练，通常假设对手将符合特定或已知的威胁模型，例如固定预算内的$\ell_p$扰动。在本文中，我们重点讨论在训练过程中防御方假设的威胁模型与测试时对手的实际能力存在不匹配的情况。我们问这样一个问题：如果学习者针对特定的“源”威胁模型进行训练，我们何时才能期望健壮性在测试期间推广到更强的未知“目标”威胁模型？我们的主要贡献是正式定义了与不可预见的对手的学习和泛化问题，这有助于我们从已知对手的传统角度来推理对手风险的增加。应用我们的框架，我们得到了一个泛化界限，它将源威胁模型和目标威胁模型之间的泛化差距与特征抽取器的变化联系起来，它度量了在给定威胁模型中提取的特征之间的期望最大差异。基于我们的泛化界，我们提出了带变异正则化的对抗性训练(AT-VR)，它减少了训练过程中特征提取子在源威胁模型上的变异。我们的实验证明，与基于高斯和图像数据集的标准对抗性训练相比，AT-VR能够提高对测试时间内不可预见攻击的泛化能力。



## **11. UNBUS: Uncertainty-aware Deep Botnet Detection System in Presence of Perturbed Samples**

UNBUS：存在扰动样本的不确定性感知深度僵尸网络检测系统 cs.CR

8 pages, 5 figures, 5 Tables

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.09502v2)

**Authors**: Rahim Taheri

**Abstracts**: A rising number of botnet families have been successfully detected using deep learning architectures. While the variety of attacks increases, these architectures should become more robust against attacks. They have been proven to be very sensitive to small but well constructed perturbations in the input. Botnet detection requires extremely low false-positive rates (FPR), which are not commonly attainable in contemporary deep learning. Attackers try to increase the FPRs by making poisoned samples. The majority of recent research has focused on the use of model loss functions to build adversarial examples and robust models. In this paper, two LSTM-based classification algorithms for botnet classification with an accuracy higher than 98% are presented. Then, the adversarial attack is proposed, which reduces the accuracy to about 30%. Then, by examining the methods for computing the uncertainty, the defense method is proposed to increase the accuracy to about 70%. By using the deep ensemble and stochastic weight averaging quantification methods it has been investigated the uncertainty of the accuracy in the proposed methods.

摘要: 使用深度学习体系结构已成功检测到越来越多的僵尸网络家族。随着攻击种类的增加，这些体系结构应该变得更强大，以抵御攻击。事实证明，它们对输入中的微小但构造良好的扰动非常敏感。僵尸网络检测需要极低的假阳性率(FPR)，这在当代深度学习中是不常见的。攻击者试图通过制作有毒样本来增加FPR。最近的大多数研究都集中在使用模型损失函数来构建对抗性例子和稳健模型。本文提出了两种基于LSTM的僵尸网络分类算法，分类正确率高于98%。然后，提出了对抗性攻击，使准确率降低到30%左右。然后，通过研究不确定度的计算方法，提出了将准确度提高到70%左右的防御方法。通过使用深度集成和随机加权平均量化方法，对所提出方法的精度的不确定度进行了研究。



## **12. Deepfake Forensics via An Adversarial Game**

通过对抗性游戏进行深度假冒取证 cs.CV

Accepted by IEEE Transactions on Image Processing; 13 pages, 4  figures

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2103.13567v2)

**Authors**: Zhi Wang, Yiwen Guo, Wangmeng Zuo

**Abstracts**: With the progress in AI-based facial forgery (i.e., deepfake), people are increasingly concerned about its abuse. Albeit effort has been made for training classification (also known as deepfake detection) models to recognize such forgeries, existing models suffer from poor generalization to unseen forgery technologies and high sensitivity to changes in image/video quality. In this paper, we advocate adversarial training for improving the generalization ability to both unseen facial forgeries and unseen image/video qualities. We believe training with samples that are adversarially crafted to attack the classification models improves the generalization ability considerably. Considering that AI-based face manipulation often leads to high-frequency artifacts that can be easily spotted by models yet difficult to generalize, we further propose a new adversarial training method that attempts to blur out these specific artifacts, by introducing pixel-wise Gaussian blurring models. With adversarial training, the classification models are forced to learn more discriminative and generalizable features, and the effectiveness of our method can be verified by plenty of empirical evidence. Our code will be made publicly available.

摘要: 随着基于人工智能的人脸伪造(即深度假)的发展，人们越来越关注它的滥用。尽管已经努力训练分类(也称为深度伪检测)模型来识别此类伪造物，但现有模型对不可见的伪造物技术的泛化能力差，并且对图像/视频质量的变化高度敏感。在本文中，我们提倡对抗性训练，以提高对看不见的人脸伪造和看不见的图像/视频质量的泛化能力。我们相信，用恶意设计的样本来攻击分类模型的训练大大提高了泛化能力。考虑到基于人工智能的人脸操作往往会导致高频伪影，这些伪影很容易被模型发现，但很难推广，我们进一步提出了一种新的对抗性训练方法，试图通过引入像素级的高斯模糊模型来模糊这些特定的伪影。通过对抗性训练，迫使分类模型学习更具区分性和泛化能力的特征，并通过大量的经验证据验证了该方法的有效性。我们的代码将公开可用。



## **13. Randomized Smoothing under Attack: How Good is it in Pratice?**

攻击下的随机平滑：它在实践中有多好？ cs.CR

ICASSP 2022

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.14187v1)

**Authors**: Thibault Maho, Teddy Furon, Erwan Le Merrer

**Abstracts**: Randomized smoothing is a recent and celebrated solution to certify the robustness of any classifier. While it indeed provides a theoretical robustness against adversarial attacks, the dimensionality of current classifiers necessarily imposes Monte Carlo approaches for its application in practice. This paper questions the effectiveness of randomized smoothing as a defense, against state of the art black-box attacks. This is a novel perspective, as previous research works considered the certification as an unquestionable guarantee. We first formally highlight the mismatch between a theoretical certification and the practice of attacks on classifiers. We then perform attacks on randomized smoothing as a defense. Our main observation is that there is a major mismatch in the settings of the RS for obtaining high certified robustness or when defeating black box attacks while preserving the classifier accuracy.

摘要: 随机平滑是最近一个著名的解决方案，用来证明任何分类器的稳健性。虽然它确实在理论上提供了对对手攻击的稳健性，但当前分类器的维度必然要求它在实践中应用蒙特卡罗方法。本文对随机平滑作为防御最先进的黑盒攻击的有效性提出了质疑。这是一个新的观点，因为以前的研究工作认为认证是毋庸置疑的保证。我们首先正式强调理论证明和对分类器的攻击实践之间的不匹配。然后我们对随机平滑进行攻击，作为一种防御。我们的主要观察是，在RS的设置中存在严重的不匹配，以获得高度认证的稳健性，或者当击败黑盒攻击时，同时保持分类器的准确性。



## **14. Adversarial Fine-tune with Dynamically Regulated Adversary**

动态调整对手的对抗性微调 cs.LG

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.13232v1)

**Authors**: Pengyue Hou, Ming Zhou, Jie Han, Petr Musilek, Xingyu Li

**Abstracts**: Adversarial training is an effective method to boost model robustness to malicious, adversarial attacks. However, such improvement in model robustness often leads to a significant sacrifice of standard performance on clean images. In many real-world applications such as health diagnosis and autonomous surgical robotics, the standard performance is more valued over model robustness against such extremely malicious attacks. This leads to the question: To what extent we can boost model robustness without sacrificing standard performance? This work tackles this problem and proposes a simple yet effective transfer learning-based adversarial training strategy that disentangles the negative effects of adversarial samples on model's standard performance. In addition, we introduce a training-friendly adversarial attack algorithm, which facilitates the boost of adversarial robustness without introducing significant training complexity. Extensive experimentation indicates that the proposed method outperforms previous adversarial training algorithms towards the target: to improve model robustness while preserving model's standard performance on clean data.

摘要: 对抗性训练是提高模型对恶意、对抗性攻击稳健性的有效方法。然而，这种模型稳健性的改进经常导致在干净图像上的标准性能的显著牺牲。在许多真实世界的应用中，例如健康诊断和自主手术机器人，对于这种极端恶意的攻击，标准性能比模型健壮性更受重视。这就引出了一个问题：在不牺牲标准性能的情况下，我们可以在多大程度上提高模型的健壮性？针对这一问题，提出了一种简单而有效的基于迁移学习的对抗性训练策略，消除了对抗性样本对模型标准性能的负面影响。此外，我们还引入了一种训练友好的对抗性攻击算法，该算法在不引入显著训练复杂度的情况下，有助于提高对抗性攻击的健壮性。大量实验表明，该方法优于以往对抗性训练算法的目标：在保持模型在干净数据上的标准性能的同时，提高模型的稳健性。



## **15. An Adversarial Attack Analysis on Malicious Advertisement URL Detection Framework**

恶意广告URL检测框架的对抗性攻击分析 cs.LG

13

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.13172v1)

**Authors**: Ehsan Nowroozi, Abhishek, Mohammadreza Mohammadi, Mauro Conti

**Abstracts**: Malicious advertisement URLs pose a security risk since they are the source of cyber-attacks, and the need to address this issue is growing in both industry and academia. Generally, the attacker delivers an attack vector to the user by means of an email, an advertisement link or any other means of communication and directs them to a malicious website to steal sensitive information and to defraud them. Existing malicious URL detection techniques are limited and to handle unseen features as well as generalize to test data. In this study, we extract a novel set of lexical and web-scrapped features and employ machine learning technique to set up system for fraudulent advertisement URLs detection. The combination set of six different kinds of features precisely overcome the obfuscation in fraudulent URL classification. Based on different statistical properties, we use twelve different formatted datasets for detection, prediction and classification task. We extend our prediction analysis for mismatched and unlabelled datasets. For this framework, we analyze the performance of four machine learning techniques: Random Forest, Gradient Boost, XGBoost and AdaBoost in the detection part. With our proposed method, we can achieve a false negative rate as low as 0.0037 while maintaining high accuracy of 99.63%. Moreover, we devise a novel unsupervised technique for data clustering using K- Means algorithm for the visual analysis. This paper analyses the vulnerability of decision tree-based models using the limited knowledge attack scenario. We considered the exploratory attack and implemented Zeroth Order Optimization adversarial attack on the detection models.

摘要: 恶意广告URL构成了安全风险，因为它们是网络攻击的来源，而且在工业界和学术界，解决这一问题的需求都在不断增长。通常，攻击者通过电子邮件、广告链接或任何其他通信方式向用户发送攻击矢量，并将他们定向到恶意网站，以窃取敏感信息并诈骗他们。现有的恶意URL检测技术在处理看不见的功能以及泛化测试数据方面都是有限的。在这项研究中，我们提取了一组新颖的词汇和网页废弃特征，并利用机器学习技术建立了欺诈性广告URL检测系统。六种不同特征的组合集合恰好克服了欺诈性URL分类中的混淆。基于不同的统计特性，我们使用了12个不同格式的数据集进行检测、预测和分类任务。我们将我们的预测分析扩展到不匹配和未标记的数据集。在检测部分，分析了四种机器学习技术：随机森林、梯度增强、XGBoost和AdaBoost的性能。该方法在保持99.63%的准确率的同时，假阴性率可低至0.0037。此外，我们设计了一种新的无监督数据聚类技术，使用K-Means算法进行可视化分析。分析了基于决策树的模型在有限知识攻击场景下的脆弱性。考虑了探索性攻击，在检测模型上实现了零阶优化对抗性攻击。



## **16. SSR-GNNs: Stroke-based Sketch Representation with Graph Neural Networks**

SSR-GNNS：基于图形神经网络的笔画表示 cs.CV

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.13153v1)

**Authors**: Sheng Cheng, Yi Ren, Yezhou Yang

**Abstracts**: This paper follows cognitive studies to investigate a graph representation for sketches, where the information of strokes, i.e., parts of a sketch, are encoded on vertices and information of inter-stroke on edges. The resultant graph representation facilitates the training of a Graph Neural Networks for classification tasks, and achieves accuracy and robustness comparable to the state-of-the-art against translation and rotation attacks, as well as stronger attacks on graph vertices and topologies, i.e., modifications and addition of strokes, all without resorting to adversarial training. Prior studies on sketches, e.g., graph transformers, encode control points of stroke on vertices, which are not invariant to spatial transformations. In contrary, we encode vertices and edges using pairwise distances among control points to achieve invariance. Compared with existing generative sketch model for one-shot classification, our method does not rely on run-time statistical inference. Lastly, the proposed representation enables generation of novel sketches that are structurally similar to while separable from the existing dataset.

摘要: 在认知研究的基础上，对素描的图形表示进行了研究，其中笔画的信息，即草图的部分，在顶点上编码，边上的笔画间的信息编码。所得到的图表示促进了图神经网络的分类任务的训练，并且获得了与最新技术相媲美的针对平移和旋转攻击的准确性和稳健性，以及对图顶点和拓扑的更强攻击，即修改和添加笔划，所有这些都不求助于对抗性训练。以往对草图的研究，例如图形转换器，对顶点上的笔划控制点进行编码，而这些控制点并不是空间变换的不变性。相反，我们使用控制点之间的成对距离对顶点和边进行编码，以实现不变性。与现有的一次分类生成式草图模型相比，该方法不依赖于运行时的统计推理。最后，所提出的表示法能够生成在结构上与现有数据集相似但可与现有数据集分开的新草图。



## **17. Defending Against Person Hiding Adversarial Patch Attack with a Universal White Frame**

用通用白框防御隐藏敌方补丁攻击的人 cs.CV

Submitted by NeurIPS 2021 with response letter to the anonymous  reviewers' comments

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.13004v1)

**Authors**: Youngjoon Yu, Hong Joo Lee, Hakmin Lee, Yong Man Ro

**Abstracts**: Object detection has attracted great attention in the computer vision area and has emerged as an indispensable component in many vision systems. In the era of deep learning, many high-performance object detection networks have been proposed. Although these detection networks show high performance, they are vulnerable to adversarial patch attacks. Changing the pixels in a restricted region can easily fool the detection network in the physical world. In particular, person-hiding attacks are emerging as a serious problem in many safety-critical applications such as autonomous driving and surveillance systems. Although it is necessary to defend against an adversarial patch attack, very few efforts have been dedicated to defending against person-hiding attacks. To tackle the problem, in this paper, we propose a novel defense strategy that mitigates a person-hiding attack by optimizing defense patterns, while previous methods optimize the model. In the proposed method, a frame-shaped pattern called a 'universal white frame' (UWF) is optimized and placed on the outside of the image. To defend against adversarial patch attacks, UWF should have three properties (i) suppressing the effect of the adversarial patch, (ii) maintaining its original prediction, and (iii) applicable regardless of images. To satisfy the aforementioned properties, we propose a novel pattern optimization algorithm that can defend against the adversarial patch. Through comprehensive experiments, we demonstrate that the proposed method effectively defends against the adversarial patch attack.

摘要: 目标检测在计算机视觉领域引起了极大的关注，已经成为许多视觉系统中不可或缺的组成部分。在深度学习时代，已经提出了许多高性能的目标检测网络。虽然这些检测网络表现出高性能，但它们很容易受到对抗性补丁攻击。更改受限区域中的像素可以很容易地欺骗物理世界中的检测网络。特别是，在自动驾驶和监控系统等许多安全关键应用中，藏人攻击正在成为一个严重的问题。尽管防御对抗性补丁攻击是必要的，但很少有人致力于防御人员隐藏攻击。针对这一问题，本文提出了一种新的防御策略，该策略通过优化防御模式来缓解人员躲藏攻击，而以往的方法则对该模型进行了优化。在所提出的方法中，一个被称为“通用白框”(UWF)的框形图案被优化并放置在图像的外部。为了防御对抗性补丁攻击，UWF应该具有三个性质(I)抑制对抗性补丁的影响，(Ii)保持其原始预测，以及(Iii)适用于任何图像。为了满足上述性质，我们提出了一种新的模式优化算法，该算法能够防御恶意补丁。通过综合实验，我们证明了该方法能够有效地防御敌意补丁攻击。



## **18. The MeVer DeepFake Detection Service: Lessons Learnt from Developing and Deploying in the Wild**

Mever DeepFake检测服务：从野外开发和部署中吸取的教训 cs.CV

10 pages, 6 figures

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.12816v1)

**Authors**: Spyridon Baxevanakis, Giorgos Kordopatis-Zilos, Panagiotis Galopoulos, Lazaros Apostolidis, Killian Levacher, Ipek B. Schlicht, Denis Teyssou, Ioannis Kompatsiaris, Symeon Papadopoulos

**Abstracts**: Enabled by recent improvements in generation methodologies, DeepFakes have become mainstream due to their increasingly better visual quality, the increase in easy-to-use generation tools and the rapid dissemination through social media. This fact poses a severe threat to our societies with the potential to erode social cohesion and influence our democracies. To mitigate the threat, numerous DeepFake detection schemes have been introduced in the literature but very few provide a web service that can be used in the wild. In this paper, we introduce the MeVer DeepFake detection service, a web service detecting deep learning manipulations in images and video. We present the design and implementation of the proposed processing pipeline that involves a model ensemble scheme, and we endow the service with a model card for transparency. Experimental results show that our service performs robustly on the three benchmark datasets while being vulnerable to Adversarial Attacks. Finally, we outline our experience and lessons learned when deploying a research system into production in the hopes that it will be useful to other academic and industry teams.

摘要: 由于最近在生成方法上的改进，DeepFake已经成为主流，因为它们的视觉质量越来越好，易于使用的生成工具的增加，以及通过社交媒体的快速传播。这一事实对我们的社会构成严重威胁，有可能侵蚀社会凝聚力并影响我们的民主国家。为了减轻威胁，文献中已经引入了许多DeepFake检测方案，但很少提供可以在野外使用的Web服务。在本文中，我们介绍了Mever DeepFake检测服务，这是一个检测图像和视频中的深度学习操作的Web服务。我们给出了所提出的处理流水线的设计和实现，该流水线涉及模型集成方案，并且我们为服务赋予模型卡以实现透明性。实验结果表明，我们的服务在三个基准数据集上表现出很好的性能，但很容易受到对手攻击。最后，我们概述了我们在将研究系统部署到生产中时的经验和教训，希望它对其他学术和行业团队有用。



## **19. Improving the Transferability of Adversarial Examples with Restructure Embedded Patches**

利用重构嵌入补丁提高对抗性实例的可转移性 cs.CV

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.12680v1)

**Authors**: Huipeng Zhou, Yu-an Tan, Yajie Wang, Haoran Lyu, Shangbo Wu, Yuanzhang Li

**Abstracts**: Vision transformers (ViTs) have demonstrated impressive performance in various computer vision tasks. However, the adversarial examples generated by ViTs are challenging to transfer to other networks with different structures. Recent attack methods do not consider the specificity of ViTs architecture and self-attention mechanism, which leads to poor transferability of the generated adversarial samples by ViTs. We attack the unique self-attention mechanism in ViTs by restructuring the embedded patches of the input. The restructured embedded patches enable the self-attention mechanism to obtain more diverse patches connections and help ViTs keep regions of interest on the object. Therefore, we propose an attack method against the unique self-attention mechanism in ViTs, called Self-Attention Patches Restructure (SAPR). Our method is simple to implement yet efficient and applicable to any self-attention based network and gradient transferability-based attack methods. We evaluate attack transferability on black-box models with different structures. The result show that our method generates adversarial examples on white-box ViTs with higher transferability and higher image quality. Our research advances the development of black-box transfer attacks on ViTs and demonstrates the feasibility of using white-box ViTs to attack other black-box models.

摘要: 视觉转换器(VITS)在各种计算机视觉任务中表现出令人印象深刻的性能。然而，VITS生成的对抗性例子很难转移到其他具有不同结构的网络上。现有的攻击方法没有考虑VITS体系结构和自我注意机制的特殊性，导致VITS生成的攻击样本可移植性较差。我们通过重组输入的嵌入补丁来攻击VITS中独特的自我注意机制。重构后的嵌入贴片使自我注意机制能够获得更多样化的贴片连接，并帮助VITS保持对象上的感兴趣区域。因此，我们提出了一种针对VITS中独特的自我注意机制的攻击方法，称为自我注意补丁重构(SAPR)。该方法实现简单，效率高，适用于任何基于自我注意的网络攻击方法和基于梯度转移的攻击方法。我们在不同结构的黑盒模型上评估了攻击的可转移性。实验结果表明，该方法在白盒VITS上生成的对抗性样本具有较高的可移植性和较高的图像质量。我们的研究推动了针对VITS的黑盒传输攻击的发展，并论证了利用白盒VITS攻击其他黑盒模型的可行性。



## **20. Data Bootstrapping Approaches to Improve Low Resource Abusive Language Detection for Indic Languages**

改进印度语低资源滥用语言检测的数据自举方法 cs.CL

Accepted at HT '22: 33rd ACM Conference on Hypertext and Social Media

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12543v1)

**Authors**: Mithun Das, Somnath Banerjee, Animesh Mukherjee

**Abstracts**: Abusive language is a growing concern in many social media platforms. Repeated exposure to abusive speech has created physiological effects on the target users. Thus, the problem of abusive language should be addressed in all forms for online peace and safety. While extensive research exists in abusive speech detection, most studies focus on English. Recently, many smearing incidents have occurred in India, which provoked diverse forms of abusive speech in online space in various languages based on the geographic location. Therefore it is essential to deal with such malicious content. In this paper, to bridge the gap, we demonstrate a large-scale analysis of multilingual abusive speech in Indic languages. We examine different interlingual transfer mechanisms and observe the performance of various multilingual models for abusive speech detection for eight different Indic languages. We also experiment to show how robust these models are on adversarial attacks. Finally, we conduct an in-depth error analysis by looking into the models' misclassified posts across various settings. We have made our code and models public for other researchers.

摘要: 在许多社交媒体平台上，辱骂语言日益受到关注。反复暴露在辱骂言语中对目标用户造成了生理影响。因此，为了网络和平与安全，应该以各种形式解决辱骂语言的问题。虽然在辱骂语音检测方面已经有了广泛的研究，但大多数研究都集中在英语上。最近，印度发生了多起诽谤事件，引发了基于地理位置的各种语言在网络空间发表不同形式的辱骂言论。因此，对此类恶意内容的处理至关重要。为了弥补这一差距，我们对印度语中的多语种辱骂言语进行了大规模的分析。我们考察了不同的语际迁移机制，并观察了针对八种不同印度语的滥用语音检测的各种多语言模型的性能。我们还进行了实验，以表明这些模型在对抗攻击时的健壮性。最后，我们通过查看模型在不同环境下错误分类的帖子进行了深入的错误分析。我们已经向其他研究人员公开了我们的代码和模型。



## **21. Restricted Black-box Adversarial Attack Against DeepFake Face Swapping**

针对DeepFake脸部交换的受限黑盒对抗性攻击 cs.CV

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12347v1)

**Authors**: Junhao Dong, Yuan Wang, Jianhuang Lai, Xiaohua Xie

**Abstracts**: DeepFake face swapping presents a significant threat to online security and social media, which can replace the source face in an arbitrary photo/video with the target face of an entirely different person. In order to prevent this fraud, some researchers have begun to study the adversarial methods against DeepFake or face manipulation. However, existing works focus on the white-box setting or the black-box setting driven by abundant queries, which severely limits the practical application of these methods. To tackle this problem, we introduce a practical adversarial attack that does not require any queries to the facial image forgery model. Our method is built on a substitute model persuing for face reconstruction and then transfers adversarial examples from the substitute model directly to inaccessible black-box DeepFake models. Specially, we propose the Transferable Cycle Adversary Generative Adversarial Network (TCA-GAN) to construct the adversarial perturbation for disrupting unknown DeepFake systems. We also present a novel post-regularization module for enhancing the transferability of generated adversarial examples. To comprehensively measure the effectiveness of our approaches, we construct a challenging benchmark of DeepFake adversarial attacks for future development. Extensive experiments impressively show that the proposed adversarial attack method makes the visual quality of DeepFake face images plummet so that they are easier to be detected by humans and algorithms. Moreover, we demonstrate that the proposed algorithm can be generalized to offer face image protection against various face translation methods.

摘要: DeepFake人脸交换对在线安全和社交媒体构成了重大威胁，可以将任意照片/视频中的源脸替换为完全不同的人的目标脸。为了防止这种欺诈，一些研究人员已经开始研究对抗DeepFake或Face操纵的方法。然而，现有的工作主要集中在白盒设置或大量查询驱动的黑盒设置上，这严重限制了这些方法的实际应用。为了解决这个问题，我们引入了一种实用的对抗性攻击，该攻击不需要对人脸图像伪造模型进行任何查询。我们的方法建立在一个试图进行人脸重建的替代模型上，然后将敌对样本从替代模型直接转移到不可访问的黑盒DeepFake模型上。特别地，我们提出了可转移循环对抗性生成对抗性网络(TCA-GAN)来构造针对未知DeepFake系统的对抗性扰动。我们还提出了一种新的后正则化模块来增强生成的对抗性实例的可转移性。为了全面衡量我们的方法的有效性，我们为未来的发展构建了一个具有挑战性的DeepFake对抗性攻击基准。大量实验表明，所提出的对抗性攻击方法使得DeepFake人脸图像的视觉质量直线下降，更容易被人类和算法检测到。此外，我们还证明了所提出的算法可以推广到针对各种人脸转换方法提供人脸图像保护。



## **22. Boosting Adversarial Transferability of MLP-Mixer**

提高MLP-Mixer的对抗转移性 cs.CV

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12204v1)

**Authors**: Haoran Lyu, Yajie Wang, Yu-an Tan, Huipeng Zhou, Yuhang Zhao, Quanxin Zhang

**Abstracts**: The security of models based on new architectures such as MLP-Mixer and ViTs needs to be studied urgently. However, most of the current researches are mainly aimed at the adversarial attack against ViTs, and there is still relatively little adversarial work on MLP-mixer. We propose an adversarial attack method against MLP-Mixer called Maxwell's demon Attack (MA). MA breaks the channel-mixing and token-mixing mechanism of MLP-Mixer by controlling the part input of MLP-Mixer's each Mixer layer, and disturbs MLP-Mixer to obtain the main information of images. Our method can mask the part input of the Mixer layer, avoid overfitting of the adversarial examples to the source model, and improve the transferability of cross-architecture. Extensive experimental evaluation demonstrates the effectiveness and superior performance of the proposed MA. Our method can be easily combined with existing methods and can improve the transferability by up to 38.0% on MLP-based ResMLP. Adversarial examples produced by our method on MLP-Mixer are able to exceed the transferability of adversarial examples produced using DenseNet against CNNs. To the best of our knowledge, we are the first work to study adversarial transferability of MLP-Mixer.

摘要: 基于MLP-Mixer和VITS等新型体系结构的模型的安全性亟待研究。然而，目前的研究大多是针对VITS的对抗性攻击，针对MLP-Mixer的对抗性研究还相对较少。提出了一种针对MLP-Mixer的对抗性攻击方法，称为麦克斯韦恶魔攻击(MA)。MA通过控制MLP-Mixer各混合层的部分输入，打破了MLP-Mixer的通道混合和令牌混合机制，干扰MLP-Mixer获取图像的主要信息。该方法屏蔽了混合层的部分输入，避免了对抗性实例对源模型的过度拟合，提高了跨体系结构的可移植性。大量的实验评估表明了该算法的有效性和优越的性能。我们的方法可以很容易地与现有的方法相结合，在基于MLP的ResMLP上可以提高高达38.0%的可转移性。我们的方法在MLP-Mixer上生成的对抗性实例能够超过DenseNet针对CNN生成的对抗性实例的可转移性。据我们所知，我们是第一个研究MLP-Mixer对抗性转移的工作。



## **23. Mixed Strategies for Security Games with General Defending Requirements**

具有一般防御要求的安全博弈的混合策略 cs.GT

Accepted by IJCAI-2022

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12158v1)

**Authors**: Rufan Bai, Haoxing Lin, Xinyu Yang, Xiaowei Wu, Minming Li, Weijia Jia

**Abstracts**: The Stackelberg security game is played between a defender and an attacker, where the defender needs to allocate a limited amount of resources to multiple targets in order to minimize the loss due to adversarial attack by the attacker. While allowing targets to have different values, classic settings often assume uniform requirements to defend the targets. This enables existing results that study mixed strategies (randomized allocation algorithms) to adopt a compact representation of the mixed strategies.   In this work, we initiate the study of mixed strategies for the security games in which the targets can have different defending requirements. In contrast to the case of uniform defending requirement, for which an optimal mixed strategy can be computed efficiently, we show that computing the optimal mixed strategy is NP-hard for the general defending requirements setting. However, we show that strong upper and lower bounds for the optimal mixed strategy defending result can be derived. We propose an efficient close-to-optimal Patching algorithm that computes mixed strategies that use only few pure strategies. We also study the setting when the game is played on a network and resource sharing is enabled between neighboring targets. Our experimental results demonstrate the effectiveness of our algorithm in several large real-world datasets.

摘要: Stackelberg安全博弈是在防御者和攻击者之间进行的，防御者需要将有限的资源分配给多个目标，以便将攻击者的对抗性攻击造成的损失降至最低。虽然允许目标具有不同的值，但经典设置通常假定保护目标的要求是统一的。这使得研究混合策略(随机分配算法)的现有结果能够采用混合策略的紧凑表示。在这项工作中，我们发起了安全博弈的混合策略的研究，其中目标可以有不同的防御需求。与统一防御需求情况下可以有效计算最优混合策略的情况相比，对于一般防御需求设置，计算最优混合策略是NP难的。然而，我们证明了最优混合策略防御结果的上界和下界是强的。我们提出了一种高效的接近最优的修补算法，该算法只使用很少的纯策略来计算混合策略。我们还研究了当游戏在网络上进行并且相邻目标之间实现资源共享时的设置。我们的实验结果证明了我们的算法在几个大型真实数据集上的有效性。



## **24. Source-independent quantum random number generator against detector blinding attacks**

抗探测器盲攻击的源无关量子随机数发生器 quant-ph

14 pages, 7 figures, 6 tables, comments are welcome

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12156v1)

**Authors**: Wen-Bo Liu, Yu-Shuo Lu, Yao Fu, Si-Cheng Huang, Ze-Jie Yin, Kun Jiang, Hua-Lei Yin, Zeng-Bing Chen

**Abstracts**: Randomness, mainly in the form of random numbers, is the fundamental prerequisite for the security of many cryptographic tasks. Quantum randomness can be extracted even if adversaries are fully aware of the protocol and even control the randomness source. However, an adversary can further manipulate the randomness via detector blinding attacks, which are a hacking attack suffered by protocols with trusted detectors. Here, by treating no-click events as valid error events, we propose a quantum random number generation protocol that can simultaneously address source vulnerability and ferocious detector blinding attacks. The method can be extended to high-dimensional random number generation. We experimentally demonstrate the ability of our protocol to generate random numbers for two-dimensional measurement with a generation speed of 0.515 Mbps, which is two orders of magnitude higher than that of device-independent protocols that can address both issues of imperfect sources and imperfect detectors.

摘要: 随机性，主要是随机数的形式，是许多密码任务安全的基本前提。即使攻击者完全知道该协议，甚至控制了随机性来源，也可以提取量子随机性。然而，攻击者可以通过检测器盲化攻击进一步操纵随机性，这是具有可信检测器的协议遭受的黑客攻击。这里，通过将无点击事件视为有效的错误事件，我们提出了一种量子随机数生成协议，该协议可以同时应对源漏洞和猛烈的检测器盲攻击。该方法可以推广到高维随机数的生成。我们通过实验证明了该协议能够生成用于二维测量的随机数，生成速度为0.515 Mbps，比设备无关的协议高出两个数量级，后者既可以解决不完善的源问题，也可以解决不完善的检测器问题。



## **25. Self-recoverable Adversarial Examples: A New Effective Protection Mechanism in Social Networks**

可自我恢复的敌意例子：一种新的有效的社交网络保护机制 cs.CV

13 pages, 11 figures

**SubmitDate**: 2022-04-26    [paper-pdf](http://arxiv.org/pdf/2204.12050v1)

**Authors**: Jiawei Zhang, Jinwei Wang, Hao Wang, Xiangyang Luo

**Abstracts**: Malicious intelligent algorithms greatly threaten the security of social users' privacy by detecting and analyzing the uploaded photos to social network platforms. The destruction to DNNs brought by the adversarial attack sparks the potential that adversarial examples serve as a new protection mechanism for privacy security in social networks. However, the existing adversarial example does not have recoverability for serving as an effective protection mechanism. To address this issue, we propose a recoverable generative adversarial network to generate self-recoverable adversarial examples. By modeling the adversarial attack and recovery as a united task, our method can minimize the error of the recovered examples while maximizing the attack ability, resulting in better recoverability of adversarial examples. To further boost the recoverability of these examples, we exploit a dimension reducer to optimize the distribution of adversarial perturbation. The experimental results prove that the adversarial examples generated by the proposed method present superior recoverability, attack ability, and robustness on different datasets and network architectures, which ensure its effectiveness as a protection mechanism in social networks.

摘要: 恶意智能算法通过检测和分析上传到社交网络平台的照片，极大地威胁到社交用户的隐私安全。敌意攻击对DNN的破坏引发了敌意例子作为一种新的社交网络隐私安全保护机制的潜力。然而，现有的对抗性范例作为一种有效的保护机制，并不具有可恢复性。为了解决这个问题，我们提出了一个可恢复的生成性对抗性网络来生成可自我恢复的对抗性实例。通过将对抗性攻击和恢复建模为一个联合任务，该方法可以在最大化攻击能力的同时最小化恢复样本的误差，从而使对抗性样本具有更好的可恢复性。为了进一步提高这些例子的可恢复性，我们开发了一个降维器来优化对抗性扰动的分布。实验结果表明，该方法生成的恶意实例在不同的数据集和网络体系结构下具有良好的可恢复性、攻击性和健壮性，是一种有效的社交网络保护机制。



## **26. Can Rationalization Improve Robustness?**

合理化能提高健壮性吗？ cs.CL

Accepted to NAACL 2022

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2204.11790v1)

**Authors**: Howard Chen, Jacqueline He, Karthik Narasimhan, Danqi Chen

**Abstracts**: A growing line of work has investigated the development of neural NLP models that can produce rationales--subsets of input that can explain their model predictions. In this paper, we ask whether such rationale models can also provide robustness to adversarial attacks in addition to their interpretable nature. Since these models need to first generate rationales ("rationalizer") before making predictions ("predictor"), they have the potential to ignore noise or adversarially added text by simply masking it out of the generated rationale. To this end, we systematically generate various types of 'AddText' attacks for both token and sentence-level rationalization tasks, and perform an extensive empirical evaluation of state-of-the-art rationale models across five different tasks. Our experiments reveal that the rationale models show the promise to improve robustness, while they struggle in certain scenarios--when the rationalizer is sensitive to positional bias or lexical choices of attack text. Further, leveraging human rationale as supervision does not always translate to better performance. Our study is a first step towards exploring the interplay between interpretability and robustness in the rationalize-then-predict framework.

摘要: 越来越多的工作研究了神经NLP模型的发展，这种模型可以产生原理--输入的子集可以解释他们的模型预测。在本文中，我们询问这些基本模型除了具有可解释的性质外，是否还可以提供对对手攻击的稳健性。由于这些模型在做出预测(“预测者”)之前需要首先生成理由(“理性器”)，因此它们有可能忽略噪声或相反添加的文本，只需将其从生成的理由中掩盖出来。为此，我们系统地为标记和句子级合理化任务生成了各种类型的AddText攻击，并在五个不同的任务中对最先进的理性模型进行了广泛的经验评估。我们的实验表明，当理性器对位置偏差或攻击文本的词汇选择敏感时，基本模型显示出提高稳健性的前景，而它们在某些场景中却举步维艰。此外，利用人的理性作为监督并不总是能转化为更好的业绩。我们的研究是探索在合理化-然后预测框架中可解释性和稳健性之间的相互作用的第一步。



## **27. Discovering Exfiltration Paths Using Reinforcement Learning with Attack Graphs**

基于攻击图强化学习的渗出路径发现 cs.CR

The 5th IEEE Conference on Dependable and Secure Computing (IEEE DSC  2022)

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2201.12416v2)

**Authors**: Tyler Cody, Abdul Rahman, Christopher Redino, Lanxiao Huang, Ryan Clark, Akshay Kakkar, Deepak Kushwaha, Paul Park, Peter Beling, Edward Bowen

**Abstracts**: Reinforcement learning (RL), in conjunction with attack graphs and cyber terrain, are used to develop reward and state associated with determination of optimal paths for exfiltration of data in enterprise networks. This work builds on previous crown jewels (CJ) identification that focused on the target goal of computing optimal paths that adversaries may traverse toward compromising CJs or hosts within their proximity. This work inverts the previous CJ approach based on the assumption that data has been stolen and now must be quietly exfiltrated from the network. RL is utilized to support the development of a reward function based on the identification of those paths where adversaries desire reduced detection. Results demonstrate promising performance for a sizable network environment.

摘要: 强化学习(RL)与攻击图和网络地形相结合，用于开发与确定企业网络中数据泄漏的最佳路径相关的奖励和状态。这项工作建立在以前的皇冠宝石(CJ)识别的基础上，该识别专注于计算对手可能穿过的最优路径的目标，以危害其邻近的CJ或主机。这项工作颠覆了之前CJ的方法，该方法基于数据已被窃取，现在必须从网络中悄悄渗出的假设。利用RL来支持基于识别其中对手希望减少检测的那些路径的奖励函数的开发。结果表明，在相当大的网络环境中具有良好的性能。



## **28. Reconstructing Training Data with Informed Adversaries**

利用知情对手重建训练数据 cs.CR

Published at "2022 IEEE Symposium on Security and Privacy (SP)"

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2201.04845v2)

**Authors**: Borja Balle, Giovanni Cherubin, Jamie Hayes

**Abstracts**: Given access to a machine learning model, can an adversary reconstruct the model's training data? This work studies this question from the lens of a powerful informed adversary who knows all the training data points except one. By instantiating concrete attacks, we show it is feasible to reconstruct the remaining data point in this stringent threat model. For convex models (e.g. logistic regression), reconstruction attacks are simple and can be derived in closed-form. For more general models (e.g. neural networks), we propose an attack strategy based on training a reconstructor network that receives as input the weights of the model under attack and produces as output the target data point. We demonstrate the effectiveness of our attack on image classifiers trained on MNIST and CIFAR-10, and systematically investigate which factors of standard machine learning pipelines affect reconstruction success. Finally, we theoretically investigate what amount of differential privacy suffices to mitigate reconstruction attacks by informed adversaries. Our work provides an effective reconstruction attack that model developers can use to assess memorization of individual points in general settings beyond those considered in previous works (e.g. generative language models or access to training gradients); it shows that standard models have the capacity to store enough information to enable high-fidelity reconstruction of training data points; and it demonstrates that differential privacy can successfully mitigate such attacks in a parameter regime where utility degradation is minimal.

摘要: 在获得机器学习模型的情况下，对手能否重建模型的训练数据？这项工作从一个强大的知情对手的角度来研究这个问题，他知道除一个以外的所有训练数据点。通过实例化具体攻击，我们证明了在这个严格的威胁模型中重构剩余数据点是可行的。对于凸模型(例如Logistic回归)，重构攻击很简单，并且可以以闭合形式推导出来。对于更一般的模型(如神经网络)，我们提出了一种基于训练重构器网络的攻击策略，该重建器网络接收被攻击模型的权重作为输入，并产生目标数据点作为输出。我们在MNIST和CIFAR-10上训练的图像分类器上验证了我们的攻击的有效性，并系统地研究了标准机器学习管道中哪些因素影响重建成功。最后，我们从理论上研究了多大程度的差异隐私足以缓解知情攻击者的重构攻击。我们的工作提供了一种有效的重建攻击，模型开发者可以用它来评估在一般环境下对单个点的记忆，而不是以前的工作中考虑的那些(例如，生成性语言模型或对训练梯度的访问)；它表明标准模型有能力存储足够的信息来实现训练数据点的高保真重建；并且它证明了在效用降级最小的参数机制中，差分隐私可以成功地缓解这种攻击。



## **29. A Simple Structure For Building A Robust Model**

一种用于建立稳健模型的简单结构 cs.CV

10 pages, 3 figures, 4 tables

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2204.11596v1)

**Authors**: Xiao Tan, JingBo Gao, Ruolin Li

**Abstracts**: As deep learning applications, especially programs of computer vision, are increasingly deployed in our lives, we have to think more urgently about the security of these applications.One effective way to improve the security of deep learning models is to perform adversarial training, which allows the model to be compatible with samples that are deliberately created for use in attacking the model.Based on this, we propose a simple architecture to build a model with a certain degree of robustness, which improves the robustness of the trained network by adding an adversarial sample detection network for cooperative training.At the same time, we design a new data sampling strategy that incorporates multiple existing attacks, allowing the model to adapt to many different adversarial attacks with a single training.We conducted some experiments to test the effectiveness of this design based on Cifar10 dataset, and the results indicate that it has some degree of positive effect on the robustness of the model.Our code could be found at https://github.com/dowdyboy/simple_structure_for_robust_model.

摘要: 随着深度学习应用，特别是计算机视觉应用的日益广泛，我们不得不更加迫切地考虑这些应用的安全性。提高深度学习模型安全性的有效方法之一是进行对抗性训练，使模型与故意创建的用于攻击模型的样本相兼容。在此基础上，提出了一种简单的架构来构建具有一定健壮性的模型，通过增加对抗性样本检测网络来进行协作训练，从而提高了训练网络的健壮性。同时，我们设计了一种新的数据采样策略，该策略融合了多种现有的攻击，在Cifar10数据集上进行了实验，测试结果表明，该设计对模型的健壮性有一定的积极作用。我们的代码可以在https://github.com/dowdyboy/simple_structure_for_robust_model.上找到



## **30. Dominating Vertical Collaborative Learning Systems**

主导垂直协作学习系统 cs.CR

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2201.02775v2)

**Authors**: Qi Pang, Yuanyuan Yuan, Shuai Wang

**Abstracts**: Vertical collaborative learning system also known as vertical federated learning (VFL) system has recently become prominent as a concept to process data distributed across many individual sources without the need to centralize it. Multiple participants collaboratively train models based on their local data in a privacy-preserving manner. To date, VFL has become a de facto solution to securely learn a model among organizations, allowing knowledge to be shared without compromising privacy of any individual organizations.   Despite the prosperous development of VFL systems, we find that certain inputs of a participant, named adversarial dominating inputs (ADIs), can dominate the joint inference towards the direction of the adversary's will and force other (victim) participants to make negligible contributions, losing rewards that are usually offered regarding the importance of their contributions in collaborative learning scenarios.   We conduct a systematic study on ADIs by first proving their existence in typical VFL systems. We then propose gradient-based methods to synthesize ADIs of various formats and exploit common VFL systems. We further launch greybox fuzz testing, guided by the resiliency score of "victim" participants, to perturb adversary-controlled inputs and systematically explore the VFL attack surface in a privacy-preserving manner. We conduct an in-depth study on the influence of critical parameters and settings in synthesizing ADIs. Our study reveals new VFL attack opportunities, promoting the identification of unknown threats before breaches and building more secure VFL systems.

摘要: 垂直协作学习系统，也被称为垂直联合学习(VFL)系统，作为一种概念，它可以处理分布在多个独立数据源的数据，而不需要集中这些数据。多个参与者以隐私保护的方式基于他们的本地数据协作训练模型。到目前为止，VFL已经成为在组织之间安全地学习模式的事实上的解决方案，允许在不损害任何个人组织隐私的情况下共享知识。尽管VFL系统的蓬勃发展，我们发现参与者的某些输入，称为对抗性主导输入(ADI)，可以主导朝着对手意愿方向的联合推理，并迫使其他(受害者)参与者做出可以忽略不计的贡献，失去通常提供的关于他们在协作学习场景中贡献的重要性的奖励。我们首先通过证明ADI在典型的VFL系统中的存在来对ADI进行系统的研究。然后，我们提出了基于梯度的方法来合成各种格式的ADI，并开发了常见的VFL系统。我们进一步推出灰盒模糊测试，以“受害者”参与者的弹性分数为指导，扰乱对手控制的输入，并以保护隐私的方式系统地探索VFL攻击面。我们深入研究了关键参数和设置对ADI合成的影响。我们的研究揭示了新的VFL攻击机会，促进了在入侵之前识别未知威胁，并建立了更安全的VFL系统。



## **31. Real or Virtual: A Video Conferencing Background Manipulation-Detection System**

真实还是虚拟：一种视频会议背景操纵检测系统 cs.CV

34 pages. arXiv admin note: text overlap with arXiv:2106.15130

**SubmitDate**: 2022-04-25    [paper-pdf](http://arxiv.org/pdf/2204.11853v1)

**Authors**: Ehsan Nowroozi, Yassine Mekdad, Mauro Conti, Simone Milani, Selcuk Uluagac, Berrin Yanikoglu

**Abstracts**: Recently, the popularity and wide use of the last-generation video conferencing technologies created an exponential growth in its market size. Such technology allows participants in different geographic regions to have a virtual face-to-face meeting. Additionally, it enables users to employ a virtual background to conceal their own environment due to privacy concerns or to reduce distractions, particularly in professional settings. Nevertheless, in scenarios where the users should not hide their actual locations, they may mislead other participants by claiming their virtual background as a real one. Therefore, it is crucial to develop tools and strategies to detect the authenticity of the considered virtual background. In this paper, we present a detection strategy to distinguish between real and virtual video conferencing user backgrounds. We demonstrate that our detector is robust against two attack scenarios. The first scenario considers the case where the detector is unaware about the attacks and inn the second scenario, we make the detector aware of the adversarial attacks, which we refer to Adversarial Multimedia Forensics (i.e, the forensically-edited frames are included in the training set). Given the lack of publicly available dataset of virtual and real backgrounds for video conferencing, we created our own dataset and made them publicly available [1]. Then, we demonstrate the robustness of our detector against different adversarial attacks that the adversary considers. Ultimately, our detector's performance is significant against the CRSPAM1372 [2] features, and post-processing operations such as geometric transformations with different quality factors that the attacker may choose. Moreover, our performance results shows that we can perfectly identify a real from a virtual background with an accuracy of 99.80%.

摘要: 最近，上一代视频会议技术的普及和广泛使用导致其市场规模呈指数级增长。这种技术允许不同地理区域的参与者进行虚拟面对面的会议。此外，它使用户能够使用虚拟背景来隐藏他们自己的环境，因为隐私问题或减少分心，特别是在专业环境中。然而，在用户不应该隐藏他们的实际位置的情况下，他们可能会误导其他参与者，声称他们的虚拟背景是真实的。因此，开发工具和策略来检测所考虑的虚拟背景的真实性是至关重要的。本文提出了一种区分真实和虚拟视频会议用户背景的检测策略。我们证明了我们的检测器对两种攻击场景都是健壮的。第一种情况考虑了检测器不知道攻击的情况，在第二种情况下，我们让检测器知道对抗性攻击，我们称之为对抗性多媒体取证(即，经取证编辑的帧包括在训练集中)。由于缺乏公开可用的视频会议虚拟和真实背景的数据集，我们创建了自己的数据集并将其公开[1]。然后，我们证明了我们的检测器对对手所考虑的不同的对手攻击具有健壮性。归根结底，我们的检测器相对于CRSPAM1372[2]功能和后处理操作(如攻击者可能选择的具有不同质量因子的几何变换)的性能是显著的。此外，我们的性能结果表明，我们可以很好地识别真实和虚拟的背景，准确率为99.80%。



## **32. Improving Deep Learning Model Robustness Against Adversarial Attack by Increasing the Network Capacity**

通过增加网络容量提高深度学习模型对敌意攻击的稳健性 cs.LG

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.11357v1)

**Authors**: Marco Marchetti, Edmond S. L. Ho

**Abstracts**: Nowadays, we are more and more reliant on Deep Learning (DL) models and thus it is essential to safeguard the security of these systems. This paper explores the security issues in Deep Learning and analyses, through the use of experiments, the way forward to build more resilient models. Experiments are conducted to identify the strengths and weaknesses of a new approach to improve the robustness of DL models against adversarial attacks. The results show improvements and new ideas that can be used as recommendations for researchers and practitioners to create increasingly better DL algorithms.

摘要: 如今，我们越来越依赖深度学习模型，因此保障这些系统的安全至关重要。本文探讨了深度学习中的安全问题，并通过实验分析了构建更具弹性模型的前进方向。通过实验确定了一种新方法的优点和缺点，以提高DL模型对对手攻击的稳健性。研究结果显示了一些改进和新的想法，可以作为研究人员和实践者创建越来越好的DL算法的建议。



## **33. A Comprehensive Test Pattern Generation Approach Exploiting SAT Attack for Logic Locking**

一种利用SAT攻击进行逻辑锁定的综合测试码生成方法 cs.CR

10 pages, 8 figures

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.11307v1)

**Authors**: Yadi Zhong, Ujjwal Guin

**Abstracts**: The need for reducing manufacturing defect escape in today's safety-critical applications requires increased fault coverage. However, generating a test set using commercial automatic test pattern generation (ATPG) tools that lead to zero-defect escape is still an open problem. It is challenging to detect all stuck-at faults to reach 100% fault coverage. In parallel, the hardware security community has been actively involved in developing solutions for logic locking to prevent IP piracy. Locks (e.g., XOR gates) are inserted in different locations of the netlist so that an adversary cannot determine the secret key. Unfortunately, the Boolean satisfiability (SAT) based attack, introduced in [1], can break different logic locking schemes in minutes. In this paper, we propose a novel test pattern generation approach using the powerful SAT attack on logic locking. A stuck-at fault is modeled as a locked gate with a secret key. Our modeling of stuck-at faults preserves the property of fault activation and propagation. We show that the input pattern that determines the key is a test for the stuck-at fault. We propose two different approaches for test pattern generation. First, a single stuck-at fault is targeted, and a corresponding locked circuit with one key bit is created. This approach generates one test pattern per fault. Second, we consider a group of faults and convert the circuit to its locked version with multiple key bits. The inputs obtained from the SAT tool are the test set for detecting this group of faults. Our approach is able to find test patterns for hard-to-detect faults that were previously failed in commercial ATPG tools. The proposed test pattern generation approach can efficiently detect redundant faults present in a circuit. We demonstrate the effectiveness of the approach on ITC'99 benchmarks. The results show that we can achieve a perfect fault coverage reaching 100%.

摘要: 在当今的安全关键应用中，减少制造缺陷逃逸的需要需要增加故障覆盖率。然而，使用商业自动测试模式生成(ATPG)工具生成测试集以实现零缺陷逃逸仍然是一个未解决的问题。要检测所有固定故障以达到100%的故障覆盖率是具有挑战性的。与此同时，硬件安全界一直积极参与开发逻辑锁定解决方案，以防止知识产权盗版。锁(例如，异或门)被插入网表的不同位置，使得对手不能确定密钥。不幸的是，在[1]中引入的基于布尔可满足性(SAT)的攻击可以在几分钟内破解不同的逻辑锁定方案。在本文中，我们提出了一种新的测试模式生成方法，该方法利用了对逻辑锁的强大SAT攻击。一个顽固的错误被建模为一扇锁着的门和一把密钥。我们对固定故障的建模保留了故障激活和传播的性质。我们证明了决定关键字的输入模式是对固定错误的测试。我们提出了两种不同的测试模式生成方法。首先，针对单个固定故障，创建具有一个密钥位的相应锁定电路。该方法为每个故障生成一个测试模式。其次，我们考虑一组故障，并将电路转换为具有多个密钥位的锁定版本。从SAT工具获得的输入是用于检测这组故障的测试集。我们的方法能够为以前在商业ATPG工具中失败的难以检测的故障找到测试模式。提出的测试码生成方法可以有效地检测电路中存在的冗余故障。我们在ITC‘99基准上证明了该方法的有效性。结果表明，我们可以达到100%的完美故障覆盖率。



## **34. Dictionary Attacks on Speaker Verification**

针对说话人确认的词典攻击 cs.SD

Manuscript and supplement, currently under review

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.11304v1)

**Authors**: Mirko Marras, Pawel Korus, Anubhav Jain, Nasir Memon

**Abstracts**: In this paper, we propose dictionary attacks against speaker verification - a novel attack vector that aims to match a large fraction of speaker population by chance. We introduce a generic formulation of the attack that can be used with various speech representations and threat models. The attacker uses adversarial optimization to maximize raw similarity of speaker embeddings between a seed speech sample and a proxy population. The resulting master voice successfully matches a non-trivial fraction of people in an unknown population. Adversarial waveforms obtained with our approach can match on average 69% of females and 38% of males enrolled in the target system at a strict decision threshold calibrated to yield false alarm rate of 1%. By using the attack with a black-box voice cloning system, we obtain master voices that are effective in the most challenging conditions and transferable between speaker encoders. We also show that, combined with multiple attempts, this attack opens even more to serious issues on the security of these systems.

摘要: 在本文中，我们提出了针对说话人验证的词典攻击，这是一种新的攻击向量，旨在随机匹配大部分说话人群体。我们介绍了一种可用于各种语音表示和威胁模型的攻击的通用公式。攻击者使用对抗性优化来最大化种子语音样本和代理群体之间说话人嵌入的原始相似性。由此产生的主音成功地匹配了未知人群中的一小部分人。使用我们的方法获得的对抗性波形可以在严格的判决阈值下与目标系统中登记的平均69%的女性和38%的男性匹配，该阈值被校准为产生1%的错误警报率。通过使用黑匣子语音克隆系统进行攻击，我们获得了在最具挑战性的条件下有效的主音，并且可以在说话人编码者之间传输。我们还表明，与多次尝试相结合，这种攻击会给这些系统的安全带来更严重的问题。



## **35. Sampling-based Fast Gradient Rescaling Method for Highly Transferable Adversarial Attacks**

一种基于采样的高可转移对抗性攻击快速梯度重缩放方法 cs.CV

The writing and experiment of the article need to be further  strengthened

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.02887v2)

**Authors**: Xu Han, Anmin Liu, Yifeng Xiong, Yanbo Fan, Kun He

**Abstracts**: Deep neural networks have shown to be very vulnerable to adversarial examples crafted by adding human-imperceptible perturbations to benign inputs. After achieving impressive attack success rates in the white-box setting, more focus is shifted to black-box attacks. In either case, the common gradient-based approaches generally use the $sign$ function to generate perturbations at the end of the process. However, only a few works pay attention to the limitation of the $sign$ function. Deviation between the original gradient and the generated noises may lead to inaccurate gradient update estimation and suboptimal solutions for adversarial transferability, which is crucial for black-box attacks. To address this issue, we propose a Sampling-based Fast Gradient Rescaling Method (S-FGRM) to improve the transferability of the crafted adversarial examples. Specifically, we use data rescaling to substitute the inefficient $sign$ function in gradient-based attacks without extra computational cost. We also propose a Depth First Sampling method to eliminate the fluctuation of rescaling and stabilize the gradient update. Our method can be used in any gradient-based optimizations and is extensible to be integrated with various input transformation or ensemble methods for further improving the adversarial transferability. Extensive experiments on the standard ImageNet dataset show that our S-FGRM could significantly boost the transferability of gradient-based attacks and outperform the state-of-the-art baselines.

摘要: 深度神经网络已被证明非常容易受到敌意例子的攻击，这些例子是通过在良性输入中添加人类无法察觉的扰动来构建的。在白盒攻击中取得了令人印象深刻的攻击成功率之后，更多的注意力转移到了黑盒攻击上。在这两种情况下，常见的基于梯度的方法通常使用$SIGN$函数在过程结束时生成扰动。然而，只有少数著作注意到$SIGN$函数的局限性。原始梯度与产生的噪声之间的偏差可能会导致不准确的梯度更新估计和对抗性转移的次优解，这是黑盒攻击的关键。针对这一问题，我们提出了一种基于采样的快速梯度重缩放方法(S-FGRM)来提高恶意例子的可转移性。具体地说，在基于梯度的攻击中，我们使用数据重缩放来代替低效的$sign$函数，而不需要额外的计算代价。我们还提出了深度优先采样的方法，消除了重缩放的波动，稳定了梯度更新。我们的方法可以用于任何基于梯度的优化，并且可以扩展到与各种输入变换或集成方法相集成，以进一步提高对抗性转移。在标准ImageNet数据集上的大量实验表明，我们的S-FGRM可以显著提高基于梯度的攻击的可转移性，并优于最新的基线。



## **36. Eliminating Backdoor Triggers for Deep Neural Networks Using Attention Relation Graph Distillation**

基于注意力关系图提取的深度神经网络后门触发器剔除 cs.LG

**SubmitDate**: 2022-04-24    [paper-pdf](http://arxiv.org/pdf/2204.09975v2)

**Authors**: Jun Xia, Ting Wang, Jiepin Ding, Xian Wei, Mingsong Chen

**Abstracts**: Due to the prosperity of Artificial Intelligence (AI) techniques, more and more backdoors are designed by adversaries to attack Deep Neural Networks (DNNs).Although the state-of-the-art method Neural Attention Distillation (NAD) can effectively erase backdoor triggers from DNNs, it still suffers from non-negligible Attack Success Rate (ASR) together with lowered classification ACCuracy (ACC), since NAD focuses on backdoor defense using attention features (i.e., attention maps) of the same order. In this paper, we introduce a novel backdoor defense framework named Attention Relation Graph Distillation (ARGD), which fully explores the correlation among attention features with different orders using our proposed Attention Relation Graphs (ARGs). Based on the alignment of ARGs between both teacher and student models during knowledge distillation, ARGD can eradicate more backdoor triggers than NAD. Comprehensive experimental results show that, against six latest backdoor attacks, ARGD outperforms NAD by up to 94.85% reduction in ASR, while ACC can be improved by up to 3.23%.

摘要: 由于人工智能(AI)技术的蓬勃发展，越来越多的对手设计了后门来攻击深度神经网络(DNN)，尽管目前最先进的方法神经注意力蒸馏(NAD)可以有效地清除DNN中的后门触发，但由于NAD侧重于利用同阶的注意特征(即注意力地图)进行后门防御，因此仍然存在不可忽视的攻击成功率(ASR)和较低的分类精度(ACC)。本文介绍了一种新的后门防御框架--注意关系图蒸馏(ARGD)，它充分利用我们提出的注意关系图(ARGs)来探索不同阶次的注意特征之间的相关性。基于知识提炼过程中教师和学生模型之间的ARG对齐，ARGD比NAD能够消除更多的后门触发。综合实验结果表明，对于最近的6次后门攻击，ARGD在ASR上比NAD降低了94.85%，而ACC则提高了3.23%。



## **37. Stochastic Variance Reduced Ensemble Adversarial Attack for Boosting the Adversarial Transferability**

提高对抗性转移能力的随机方差降低集成对抗性攻击 cs.LG

11 pages, 6 figures, accepted by CVPR 2022

**SubmitDate**: 2022-04-23    [paper-pdf](http://arxiv.org/pdf/2111.10752v2)

**Authors**: Yifeng Xiong, Jiadong Lin, Min Zhang, John E. Hopcroft, Kun He

**Abstracts**: The black-box adversarial attack has attracted impressive attention for its practical use in the field of deep learning security. Meanwhile, it is very challenging as there is no access to the network architecture or internal weights of the target model. Based on the hypothesis that if an example remains adversarial for multiple models, then it is more likely to transfer the attack capability to other models, the ensemble-based adversarial attack methods are efficient and widely used for black-box attacks. However, ways of ensemble attack are rather less investigated, and existing ensemble attacks simply fuse the outputs of all the models evenly. In this work, we treat the iterative ensemble attack as a stochastic gradient descent optimization process, in which the variance of the gradients on different models may lead to poor local optima. To this end, we propose a novel attack method called the stochastic variance reduced ensemble (SVRE) attack, which could reduce the gradient variance of the ensemble models and take full advantage of the ensemble attack. Empirical results on the standard ImageNet dataset demonstrate that the proposed method could boost the adversarial transferability and outperforms existing ensemble attacks significantly. Code is available at https://github.com/JHL-HUST/SVRE.

摘要: 黑盒对抗性攻击因其在深度学习安全领域的实际应用而备受关注。同时，这是非常具有挑战性的，因为无法访问目标模型的网络体系结构或内部权重。基于这样的假设，如果一个例子在多个模型上保持对抗性，那么它更有可能将攻击能力转移到其他模型上，基于集成的对抗性攻击方法是有效的，并被广泛应用于黑盒攻击。然而，集成攻击方法的研究相对较少，现有的集成攻击只是将所有模型的输出均匀地融合在一起。在本文中，我们将迭代集成攻击视为一个随机梯度下降优化过程，其中不同模型上的梯度变化可能导致局部最优解较差。为此，我们提出了一种新的攻击方法，称为随机方差减少集成(SVRE)攻击，它可以降低集成模型的梯度方差，并充分利用集成攻击的优势。在标准ImageNet数据集上的实验结果表明，该方法可以提高对抗性可转移性，并显著优于现有的集成攻击。代码可在https://github.com/JHL-HUST/SVRE.上找到



## **38. Certifiably Robust Variational Autoencoders**

可证明稳健性的变分自动编码器 stat.ML

12 pages and appendix

**SubmitDate**: 2022-04-23    [paper-pdf](http://arxiv.org/pdf/2102.07559v3)

**Authors**: Ben Barrett, Alexander Camuto, Matthew Willetts, Tom Rainforth

**Abstracts**: We introduce an approach for training Variational Autoencoders (VAEs) that are certifiably robust to adversarial attack. Specifically, we first derive actionable bounds on the minimal size of an input perturbation required to change a VAE's reconstruction by more than an allowed amount, with these bounds depending on certain key parameters such as the Lipschitz constants of the encoder and decoder. We then show how these parameters can be controlled, thereby providing a mechanism to ensure \textit{a priori} that a VAE will attain a desired level of robustness. Moreover, we extend this to a complete practical approach for training such VAEs to ensure our criteria are met. Critically, our method allows one to specify a desired level of robustness \emph{upfront} and then train a VAE that is guaranteed to achieve this robustness. We further demonstrate that these Lipschitz--constrained VAEs are more robust to attack than standard VAEs in practice.

摘要: 我们介绍了一种训练变分自动编码器(VAE)的方法，该方法对对手攻击具有可证明的健壮性。具体地说，我们首先推导出将VAE的重建改变超过允许量所需的输入扰动的最小大小的可操作界，这些界取决于某些关键参数，例如编码器和解码器的Lipschitz常数。然后，我们将展示如何控制这些参数，从而提供一种机制来确保VAE将达到所需的健壮性级别。此外，我们将此扩展为培训此类VAE的完整实用方法，以确保符合我们的标准。关键是，我们的方法允许指定期望的健壮性级别，然后训练保证实现该健壮性的VAE。在实践中，我们进一步证明了这些Lipschitz约束的VAE比标准VAE具有更强的抗攻击能力。



## **39. Smart App Attack: Hacking Deep Learning Models in Android Apps**

智能应用程序攻击：入侵Android应用程序中的深度学习模型 cs.LG

Accepted to IEEE Transactions on Information Forensics and Security.  This is a preprint version, the copyright belongs to The Institute of  Electrical and Electronics Engineers

**SubmitDate**: 2022-04-23    [paper-pdf](http://arxiv.org/pdf/2204.11075v1)

**Authors**: Yujin Huang, Chunyang Chen

**Abstracts**: On-device deep learning is rapidly gaining popularity in mobile applications. Compared to offloading deep learning from smartphones to the cloud, on-device deep learning enables offline model inference while preserving user privacy. However, such mechanisms inevitably store models on users' smartphones and may invite adversarial attacks as they are accessible to attackers. Due to the characteristic of the on-device model, most existing adversarial attacks cannot be directly applied for on-device models. In this paper, we introduce a grey-box adversarial attack framework to hack on-device models by crafting highly similar binary classification models based on identified transfer learning approaches and pre-trained models from TensorFlow Hub. We evaluate the attack effectiveness and generality in terms of four different settings including pre-trained models, datasets, transfer learning approaches and adversarial attack algorithms. The results demonstrate that the proposed attacks remain effective regardless of different settings, and significantly outperform state-of-the-art baselines. We further conduct an empirical study on real-world deep learning mobile apps collected from Google Play. Among 53 apps adopting transfer learning, we find that 71.7\% of them can be successfully attacked, which includes popular ones in medicine, automation, and finance categories with critical usage scenarios. The results call for the awareness and actions of deep learning mobile app developers to secure the on-device models. The code of this work is available at https://github.com/Jinxhy/SmartAppAttack

摘要: 设备上的深度学习在移动应用程序中迅速流行起来。与将深度学习从智能手机转移到云相比，设备上的深度学习支持离线模型推理，同时保护用户隐私。然而，这种机制不可避免地将模型存储在用户的智能手机上，并可能招致对抗性攻击，因为攻击者可以访问这些模型。由于设备上模型的特点，现有的大多数对抗性攻击不能直接应用于设备上模型。在本文中，我们引入了一种灰盒对抗性攻击框架，通过基于识别的迁移学习方法和TensorFlow Hub的预训练模型构建高度相似的二进制分类模型来破解设备上的模型。我们从预先训练的模型、数据集、迁移学习方法和对抗性攻击算法四个不同的设置来评估攻击的有效性和通用性。结果表明，所提出的攻击无论在不同的设置下都保持有效，并且显著优于最先进的基线。我们进一步对从Google Play收集的真实世界深度学习移动应用程序进行了实证研究。在53个采用迁移学习的应用中，我们发现其中71.7%的应用可以被成功攻击，其中包括医学、自动化、金融等具有关键使用场景的热门应用。这一结果呼吁深度学习移动应用开发者的意识和行动，以确保设备模型的安全。这项工作的代码可以在https://github.com/Jinxhy/SmartAppAttack上找到



## **40. Towards Data-Free Model Stealing in a Hard Label Setting**

在硬标签设置中走向无数据模型窃取 cs.CR

CVPR 2022, Project Page: https://sites.google.com/view/dfms-hl

**SubmitDate**: 2022-04-23    [paper-pdf](http://arxiv.org/pdf/2204.11022v1)

**Authors**: Sunandini Sanyal, Sravanti Addepalli, R. Venkatesh Babu

**Abstracts**: Machine learning models deployed as a service (MLaaS) are susceptible to model stealing attacks, where an adversary attempts to steal the model within a restricted access framework. While existing attacks demonstrate near-perfect clone-model performance using softmax predictions of the classification network, most of the APIs allow access to only the top-1 labels. In this work, we show that it is indeed possible to steal Machine Learning models by accessing only top-1 predictions (Hard Label setting) as well, without access to model gradients (Black-Box setting) or even the training dataset (Data-Free setting) within a low query budget. We propose a novel GAN-based framework that trains the student and generator in tandem to steal the model effectively while overcoming the challenge of the hard label setting by utilizing gradients of the clone network as a proxy to the victim's gradients. We propose to overcome the large query costs associated with a typical Data-Free setting by utilizing publicly available (potentially unrelated) datasets as a weak image prior. We additionally show that even in the absence of such data, it is possible to achieve state-of-the-art results within a low query budget using synthetically crafted samples. We are the first to demonstrate the scalability of Model Stealing in a restricted access setting on a 100 class dataset as well.

摘要: 部署为服务的机器学习模型(MLaaS)容易受到模型窃取攻击，在这种攻击中，对手试图在受限访问框架内窃取模型。虽然现有攻击使用Softmax分类网络预测展示了近乎完美的克隆模型性能，但大多数API仅允许访问前1个标签。在这项工作中，我们表明，通过只访问TOP-1预测(硬标签设置)，而不访问模型梯度(黑盒设置)，甚至在低查询预算内访问训练数据集(无数据设置)，确实有可能窃取机器学习模型。我们提出了一种新的基于GAN的框架，它通过利用克隆网络的梯度作为受害者梯度的代理来训练学生和生成器一起有效地窃取模型，同时克服了硬标签设置的挑战。我们建议通过利用公开可用的(潜在无关的)数据集作为弱图像先验来克服与典型的无数据设置相关联的大量查询成本。此外，我们还表明，即使在没有这样的数据的情况下，也可以使用人工合成的样本在较低的查询预算内获得最先进的结果。我们也是第一个在100类数据集上的受限访问设置中演示模型窃取的可扩展性的。



## **41. GFCL: A GRU-based Federated Continual Learning Framework against Adversarial Attacks in IoV**

GFCL：一种基于GRU的联合持续学习框架 cs.LG

11 pages, 12 figures, 3 tables; This paper has been submitted to IEEE  Internet of Things Journal

**SubmitDate**: 2022-04-23    [paper-pdf](http://arxiv.org/pdf/2204.11010v1)

**Authors**: Anum Talpur, Mohan Gurusamy

**Abstracts**: The integration of ML in 5G-based Internet of Vehicles (IoV) networks has enabled intelligent transportation and smart traffic management. Nonetheless, the security against adversarial attacks is also increasingly becoming a challenging task. Specifically, Deep Reinforcement Learning (DRL) is one of the widely used ML designs in IoV applications. The standard ML security techniques are not effective in DRL where the algorithm learns to solve sequential decision-making through continuous interaction with the environment, and the environment is time-varying, dynamic, and mobile. In this paper, we propose a Gated Recurrent Unit (GRU)-based federated continual learning (GFCL) anomaly detection framework against adversarial attacks in IoV. The objective is to present a lightweight and scalable framework that learns and detects the illegitimate behavior without having a-priori training dataset consisting of attack samples. We use GRU to predict a future data sequence to analyze and detect illegitimate behavior from vehicles in a federated learning-based distributed manner. We investigate the performance of our framework using real-world vehicle mobility traces. The results demonstrate the effectiveness of our proposed solution for different performance metrics.

摘要: ML在基于5G的车联网(IoV)网络中的整合实现了智能交通和智能交通管理。然而，对抗攻击的安全也日益成为一项具有挑战性的任务。其中，深度强化学习(DRL)是IoV应用中广泛使用的ML设计之一。标准的ML安全技术在DRL中并不有效，在DRL中，算法通过与环境的持续交互来学习解决顺序决策，并且环境是时变的、动态的和移动的。提出了一种基于门控递归单元(GRU)的联合连续学习(GFCL)异常检测框架，用于对抗IoV中的敌意攻击。其目的是提供一个轻量级和可扩展的框架，在没有包含攻击样本的先验训练数据集的情况下学习和检测非法行为。我们使用GRU来预测未来的数据序列，以基于联合学习的分布式方式来分析和检测车辆的非法行为。我们使用真实世界的车辆移动轨迹来研究我们的框架的性能。结果表明，本文提出的解决方案对于不同的性能指标是有效的。



## **42. A Tale of Two Models: Constructing Evasive Attacks on Edge Models**

两个模型的故事：构造对边模型的规避攻击 cs.CR

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2204.10933v1)

**Authors**: Wei Hao, Aahil Awatramani, Jiayang Hu, Chengzhi Mao, Pin-Chun Chen, Eyal Cidon, Asaf Cidon, Junfeng Yang

**Abstracts**: Full-precision deep learning models are typically too large or costly to deploy on edge devices. To accommodate to the limited hardware resources, models are adapted to the edge using various edge-adaptation techniques, such as quantization and pruning. While such techniques may have a negligible impact on top-line accuracy, the adapted models exhibit subtle differences in output compared to the original model from which they are derived. In this paper, we introduce a new evasive attack, DIVA, that exploits these differences in edge adaptation, by adding adversarial noise to input data that maximizes the output difference between the original and adapted model. Such an attack is particularly dangerous, because the malicious input will trick the adapted model running on the edge, but will be virtually undetectable by the original model, which typically serves as the authoritative model version, used for validation, debugging and retraining. We compare DIVA to a state-of-the-art attack, PGD, and show that DIVA is only 1.7-3.6% worse on attacking the adapted model but 1.9-4.2 times more likely not to be detected by the the original model under a whitebox and semi-blackbox setting, compared to PGD.

摘要: 全精度深度学习模型通常太大或太昂贵，无法在边缘设备上部署。为了适应有限的硬件资源，模型使用各种边缘自适应技术来适应边缘，如量化和剪枝。虽然这类技术对顶线精度的影响可能微乎其微，但与原始模型相比，改装后的模型在输出方面表现出了细微的差异。在本文中，我们介绍了一种新的规避攻击，DIVA，它利用边缘自适应的这些差异，通过在输入数据中添加对抗性噪声来最大化原始模型和自适应模型之间的输出差异。这种攻击特别危险，因为恶意输入将欺骗在边缘运行的适应模型，但原始模型实际上无法检测到，原始模型通常用作权威模型版本，用于验证、调试和再培训。我们将DIVA与最先进的攻击pgd进行了比较，结果表明，与pgd相比，在白盒和半黑盒设置下，DIVA在攻击适应的模型上只差1.7%-3.6%，但不被原始模型检测的可能性高1.9-4.2倍。



## **43. How Sampling Impacts the Robustness of Stochastic Neural Networks**

抽样如何影响随机神经网络的稳健性 cs.LG

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2204.10839v1)

**Authors**: Sina Däubener, Asja Fischer

**Abstracts**: Stochastic neural networks (SNNs) are random functions and predictions are gained by averaging over multiple realizations of this random function. Consequently, an adversarial attack is calculated based on one set of samples and applied to the prediction defined by another set of samples. In this paper we analyze robustness in this setting by deriving a sufficient condition for the given prediction process to be robust against the calculated attack. This allows us to identify the factors that lead to an increased robustness of SNNs and helps to explain the impact of the variance and the amount of samples. Among other things, our theoretical analysis gives insights into (i) why increasing the amount of samples drawn for the estimation of adversarial examples increases the attack's strength, (ii) why decreasing sample size during inference hardly influences the robustness, and (iii) why a higher prediction variance between realizations relates to a higher robustness. We verify the validity of our theoretical findings by an extensive empirical analysis.

摘要: 随机神经网络(SNN)是随机函数，预测是通过对该随机函数的多个实现进行平均来获得的。因此，基于一组样本计算对抗性攻击，并将其应用于由另一组样本定义的预测。在本文中，我们通过推导出给定预测过程对计算攻击具有健壮性的一个充分条件来分析这种情况下的稳健性。这使我们能够确定导致SNN稳健性增强的因素，并有助于解释方差和样本量的影响。在其他方面，我们的理论分析揭示了(I)为什么增加用于估计对抗性例子的样本量会增加攻击的强度，(Ii)为什么在推理过程中减少样本大小几乎不会影响稳健性，以及(Iii)为什么实现之间的预测方差越大，就会有越高的稳健性。我们通过广泛的实证分析验证了我们的理论发现的有效性。



## **44. Defending Black-box Skeleton-based Human Activity Classifiers**

基于黑盒骨架防御的人类活动分类器 cs.CV

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2203.04713v2)

**Authors**: He Wang, Yunfeng Diao, Zichang Tan, Guodong Guo

**Abstracts**: Deep learning has been regarded as the `go to' solution for many tasks today, but its intrinsic vulnerability to malicious attacks has become a major concern. The vulnerability is affected by a variety of factors including models, tasks, data, and attackers. Consequently, methods such as Adversarial Training and Randomized Smoothing have been proposed to tackle the problem in a wide range of applications. In this paper, we investigate skeleton-based Human Activity Recognition, which is an important type of time-series data but under-explored in defense against attacks. Our method is featured by (1) a new Bayesian Energy-based formulation of robust discriminative classifiers, (2) a new parameterization of the adversarial sample manifold of actions, and (3) a new post-train Bayesian treatment on both the adversarial samples and the classifier. We name our framework Bayesian Energy-based Adversarial Training or BEAT. BEAT is straightforward but elegant, which turns vulnerable black-box classifiers into robust ones without sacrificing accuracy. It demonstrates surprising and universal effectiveness across a wide range of action classifiers and datasets, under various attacks.

摘要: 深度学习一直被认为是当今许多任务的“首选”解决方案，但其固有的易受恶意攻击的脆弱性已成为一个主要问题。该漏洞受到多种因素的影响，包括模型、任务、数据和攻击者。因此，提出了对抗性训练和随机平滑等方法来解决这一问题，并得到了广泛的应用。在本文中，我们研究了基于骨架的人类活动识别，这是一种重要的时间序列数据类型，但在防御攻击方面还没有得到充分的探索。我们的方法的特点是(1)新的基于贝叶斯能量的稳健判别分类器的公式，(2)对抗性样本动作流形的新的参数化，以及(3)对对抗性样本和分类器的新的训练后贝叶斯处理。我们将我们的框架命名为基于贝叶斯能量的对抗性训练或BEAT。BEAT是简单但优雅的，它将脆弱的黑匣子分类器变成了健壮的分类器，而不会牺牲准确性。它在各种攻击下，在广泛的动作分类器和数据集上展示了令人惊讶和普遍的有效性。



## **45. Enhancing the Transferability via Feature-Momentum Adversarial Attack**

通过特征-动量对抗性攻击增强可转移性 cs.CV

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2204.10606v1)

**Authors**: Xianglong, Yuezun Li, Haipeng Qu, Junyu Dong

**Abstracts**: Transferable adversarial attack has drawn increasing attention due to their practical threaten to real-world applications. In particular, the feature-level adversarial attack is one recent branch that can enhance the transferability via disturbing the intermediate features. The existing methods usually create a guidance map for features, where the value indicates the importance of the corresponding feature element and then employs an iterative algorithm to disrupt the features accordingly. However, the guidance map is fixed in existing methods, which can not consistently reflect the behavior of networks as the image is changed during iteration. In this paper, we describe a new method called Feature-Momentum Adversarial Attack (FMAA) to further improve transferability. The key idea of our method is that we estimate a guidance map dynamically at each iteration using momentum to effectively disturb the category-relevant features. Extensive experiments demonstrate that our method significantly outperforms other state-of-the-art methods by a large margin on different target models.

摘要: 可转移敌意攻击由于其对现实世界应用的实际威胁而受到越来越多的关注。特别是，特征级对抗性攻击是最近的一个分支，它可以通过干扰中间特征来增强可转移性。现有的方法通常为特征创建一个导引地图，其中的值表示对应的特征元素的重要性，然后采用迭代算法对特征进行相应的破坏。然而，现有方法中导航地图是固定的，当图像在迭代过程中发生变化时，不能一致地反映网络的行为。在本文中，我们描述了一种新的方法，称为特征-动量对抗攻击(FMAA)，以进一步提高可转移性。该方法的核心思想是在每一次迭代中使用动量来动态估计导航图，以有效地干扰与类别相关的特征。大量的实验表明，在不同的目标模型上，我们的方法比其他最先进的方法有很大的优势。



## **46. Data-Efficient Backdoor Attacks**

数据高效的后门攻击 cs.CV

Accepted to IJCAI 2022 Long Oral

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2204.12281v1)

**Authors**: Pengfei Xia, Ziqiang Li, Wei Zhang, Bin Li

**Abstracts**: Recent studies have proven that deep neural networks are vulnerable to backdoor attacks. Specifically, by mixing a small number of poisoned samples into the training set, the behavior of the trained model can be maliciously controlled. Existing attack methods construct such adversaries by randomly selecting some clean data from the benign set and then embedding a trigger into them. However, this selection strategy ignores the fact that each poisoned sample contributes inequally to the backdoor injection, which reduces the efficiency of poisoning. In this paper, we formulate improving the poisoned data efficiency by the selection as an optimization problem and propose a Filtering-and-Updating Strategy (FUS) to solve it. The experimental results on CIFAR-10 and ImageNet-10 indicate that the proposed method is effective: the same attack success rate can be achieved with only 47% to 75% of the poisoned sample volume compared to the random selection strategy. More importantly, the adversaries selected according to one setting can generalize well to other settings, exhibiting strong transferability.

摘要: 最近的研究证明，深度神经网络很容易受到后门攻击。具体地说，通过将少量有毒样本混合到训练集中，可以恶意控制训练模型的行为。现有的攻击方法通过从良性集合中随机选择一些干净的数据，然后在其中嵌入触发器来构建这样的攻击者。然而，这种选择策略忽略了这样一个事实，即每个有毒样本对后门注入的贡献是不相等的，这降低了中毒的效率。在本文中，我们将通过选择来提高有毒数据效率的问题描述为一个优化问题，并提出了一种过滤和更新策略(FUS)来解决该问题。在CIFAR-10和ImageNet-10上的实验结果表明，该方法是有效的：与随机选择策略相比，只需47%~75%的中毒样本量即可获得相同的攻击成功率。更重要的是，根据一种设置选择的对手可以很好地推广到其他设置，表现出很强的可转移性。



## **47. Improving the Robustness of Adversarial Attacks Using an Affine-Invariant Gradient Estimator**

利用仿射不变梯度估计提高敌方攻击的稳健性 cs.CV

**SubmitDate**: 2022-04-22    [paper-pdf](http://arxiv.org/pdf/2109.05820v2)

**Authors**: Wenzhao Xiang, Hang Su, Chang Liu, Yandong Guo, Shibao Zheng

**Abstracts**: As designers of artificial intelligence try to outwit hackers, both sides continue to hone in on AI's inherent vulnerabilities. Designed and trained from certain statistical distributions of data, AI's deep neural networks (DNNs) remain vulnerable to deceptive inputs that violate a DNN's statistical, predictive assumptions. Before being fed into a neural network, however, most existing adversarial examples cannot maintain malicious functionality when applied to an affine transformation. For practical purposes, maintaining that malicious functionality serves as an important measure of the robustness of adversarial attacks. To help DNNs learn to defend themselves more thoroughly against attacks, we propose an affine-invariant adversarial attack, which can consistently produce more robust adversarial examples over affine transformations. For efficiency, we propose to disentangle current affine-transformation strategies from the Euclidean geometry coordinate plane with its geometric translations, rotations and dilations; we reformulate the latter two in polar coordinates. Afterwards, we construct an affine-invariant gradient estimator by convolving the gradient at the original image with derived kernels, which can be integrated with any gradient-based attack methods. Extensive experiments on ImageNet, including some experiments under physical condition, demonstrate that our method can significantly improve the affine invariance of adversarial examples and, as a byproduct, improve the transferability of adversarial examples, compared with alternative state-of-the-art methods.

摘要: 在人工智能设计者试图智取黑客的同时，双方都在继续钻研人工智能固有的弱点。人工智能的深度神经网络(DNN)是根据数据的某些统计分布设计和训练的，它仍然容易受到违反DNN统计预测假设的欺骗性输入的影响。然而，在输入到神经网络之前，大多数现有的对抗性例子在应用于仿射变换时无法保持恶意功能。出于实际目的，保持恶意功能是衡量敌意攻击健壮性的重要指标。为了帮助DNN学习更彻底地防御攻击，我们提出了一种仿射不变的对抗性攻击，它可以一致地产生比仿射变换更健壮的对抗性例子。为了提高效率，我们建议将当前的仿射变换策略从欧几里德几何坐标平面及其几何平移、旋转和伸缩中分离出来；我们将后两者重新表述在极坐标中。然后，我们通过将原始图像上的梯度与派生核进行卷积来构造仿射不变梯度估计器，该估计器可以与任何基于梯度的攻击方法相结合。在ImageNet上的大量实验，包括一些物理条件下的实验，表明我们的方法可以显著地提高对抗性例子的仿射不变性，并且作为副产品，与其他最新的方法相比，提高了对抗性例子的可转移性。



## **48. Real-Time Detectors for Digital and Physical Adversarial Inputs to Perception Systems**

感知系统的数字和物理敌方输入的实时检测器 cs.CV

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2002.09792v2)

**Authors**: Yiannis Kantaros, Taylor Carpenter, Kaustubh Sridhar, Yahan Yang, Insup Lee, James Weimer

**Abstracts**: Deep neural network (DNN) models have proven to be vulnerable to adversarial digital and physical attacks. In this paper, we propose a novel attack- and dataset-agnostic and real-time detector for both types of adversarial inputs to DNN-based perception systems. In particular, the proposed detector relies on the observation that adversarial images are sensitive to certain label-invariant transformations. Specifically, to determine if an image has been adversarially manipulated, the proposed detector checks if the output of the target classifier on a given input image changes significantly after feeding it a transformed version of the image under investigation. Moreover, we show that the proposed detector is computationally-light both at runtime and design-time which makes it suitable for real-time applications that may also involve large-scale image domains. To highlight this, we demonstrate the efficiency of the proposed detector on ImageNet, a task that is computationally challenging for the majority of relevant defenses, and on physically attacked traffic signs that may be encountered in real-time autonomy applications. Finally, we propose the first adversarial dataset, called AdvNet that includes both clean and physical traffic sign images. Our extensive comparative experiments on the MNIST, CIFAR10, ImageNet, and AdvNet datasets show that VisionGuard outperforms existing defenses in terms of scalability and detection performance. We have also evaluated the proposed detector on field test data obtained on a moving vehicle equipped with a perception-based DNN being under attack.

摘要: 深度神经网络(DNN)模型已被证明容易受到敌意的数字和物理攻击。在本文中，我们提出了一种新的攻击和数据集不可知的实时检测器，用于基于DNN的感知系统的两种类型的敌意输入。特别是，所提出的检测器依赖于观察到的对抗性图像对某些标签不变变换是敏感的。具体地说，为了确定图像是否被恶意操纵，所提出的检测器检查在向目标分类器提供被调查图像的变换版本后，目标分类器在给定输入图像上的输出是否发生显著变化。此外，我们证明了所提出的检测器在运行时和设计时都是计算轻量级的，这使得它适合于也可能涉及大规模图像域的实时应用。为了突出这一点，我们在ImageNet上展示了所提出的检测器的效率，对于大多数相关防御来说，这是一项计算上具有挑战性的任务，以及在实时自主应用中可能遇到的物理攻击的交通标志上。最后，我们提出了第一个对抗性数据集，称为AdvNet，它包括干净的和物理的交通标志图像。我们在MNIST、CIFAR10、ImageNet和AdvNet数据集上的广泛比较实验表明，VisionGuard在可扩展性和检测性能方面优于现有的防御系统。我们还根据现场测试数据对所提出的检测器进行了评估，该检测器是在一辆安装了基于感知的DNN的移动车辆上被攻击的。



## **49. Adversarial Contrastive Learning by Permuting Cluster Assignments**

基于置换类分配的对抗性对比学习 cs.LG

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.10314v1)

**Authors**: Muntasir Wahed, Afrina Tabassum, Ismini Lourentzou

**Abstracts**: Contrastive learning has gained popularity as an effective self-supervised representation learning technique. Several research directions improve traditional contrastive approaches, e.g., prototypical contrastive methods better capture the semantic similarity among instances and reduce the computational burden by considering cluster prototypes or cluster assignments, while adversarial instance-wise contrastive methods improve robustness against a variety of attacks. To the best of our knowledge, no prior work jointly considers robustness, cluster-wise semantic similarity and computational efficiency. In this work, we propose SwARo, an adversarial contrastive framework that incorporates cluster assignment permutations to generate representative adversarial samples. We evaluate SwARo on multiple benchmark datasets and against various white-box and black-box attacks, obtaining consistent improvements over state-of-the-art baselines.

摘要: 对比学习作为一种有效的自我监督表征学习技术已经得到了广泛的应用。一些研究方向改进了传统的对比方法，如原型对比方法更好地捕捉实例之间的语义相似性，并通过考虑簇原型或簇分配来减少计算负担，而对抗性实例对比方法提高了对各种攻击的健壮性。就我们所知，以前的工作没有同时考虑稳健性、聚类语义相似度和计算效率。在这项工作中，我们提出了SwARo，这是一个对抗性对比框架，它结合了簇分配排列来生成具有代表性的对抗性样本。我们在多个基准数据集上对SwARo进行评估，并针对各种白盒和黑盒攻击进行评估，在最先进的基线上获得持续的改进。



## **50. A Mask-Based Adversarial Defense Scheme**

一种基于面具的对抗性防御方案 cs.LG

7 pages

**SubmitDate**: 2022-04-21    [paper-pdf](http://arxiv.org/pdf/2204.11837v1)

**Authors**: Weizhen Xu, Chenyi Zhang, Fangzhen Zhao, Liangda Fang

**Abstracts**: Adversarial attacks hamper the functionality and accuracy of Deep Neural Networks (DNNs) by meddling with subtle perturbations to their inputs.In this work, we propose a new Mask-based Adversarial Defense scheme (MAD) for DNNs to mitigate the negative effect from adversarial attacks. To be precise, our method promotes the robustness of a DNN by randomly masking a portion of potential adversarial images, and as a result, the %classification result output of the DNN becomes more tolerant to minor input perturbations. Compared with existing adversarial defense techniques, our method does not need any additional denoising structure, nor any change to a DNN's design. We have tested this approach on a collection of DNN models for a variety of data sets, and the experimental results confirm that the proposed method can effectively improve the defense abilities of the DNNs against all of the tested adversarial attack methods. In certain scenarios, the DNN models trained with MAD have improved classification accuracy by as much as 20% to 90% compared to the original models that are given adversarial inputs.

摘要: 对抗性攻击通过干扰深层神经网络(DNNS)的输入干扰其功能和准确性，提出了一种新的基于掩码的DNN对抗性防御方案(MAD)，以缓解对抗性攻击带来的负面影响。准确地说，我们的方法通过随机掩蔽一部分潜在的敌意图像来提高DNN的稳健性，从而使DNN的分类结果输出对微小的输入扰动具有更强的容错性。与现有的对抗性防御技术相比，该方法不需要任何额外的去噪结构，也不需要对DNN的设计进行任何改变。我们在各种数据集的DNN模型上对该方法进行了测试，实验结果证实，该方法能够有效地提高DNN对所有测试的对抗性攻击的防御能力。在某些场景中，与给出对抗性输入的原始模型相比，用MAD训练的DNN模型的分类准确率提高了20%到90%。



