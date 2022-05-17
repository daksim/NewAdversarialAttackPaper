# Latest Adversarial Attack Papers
**update at 2022-05-18 06:31:28**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Transferability of Adversarial Attacks on Synthetic Speech Detection**

合成语音检测中对抗性攻击的可转移性 cs.SD

5 pages, submit to Interspeech2022

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07711v1)

**Authors**: Jiacheng Deng, Shunyi Chen, Li Dong, Diqun Yan, Rangding Wang

**Abstracts**: Synthetic speech detection is one of the most important research problems in audio security. Meanwhile, deep neural networks are vulnerable to adversarial attacks. Therefore, we establish a comprehensive benchmark to evaluate the transferability of adversarial attacks on the synthetic speech detection task. Specifically, we attempt to investigate: 1) The transferability of adversarial attacks between different features. 2) The influence of varying extraction hyperparameters of features on the transferability of adversarial attacks. 3) The effect of clipping or self-padding operation on the transferability of adversarial attacks. By performing these analyses, we summarise the weaknesses of synthetic speech detectors and the transferability behaviours of adversarial attacks, which provide insights for future research. More details can be found at https://gitee.com/djc_QRICK/Attack-Transferability-On-Synthetic-Detection.

摘要: 合成语音检测是音频安全领域的重要研究课题之一。与此同时，深度神经网络很容易受到敌意攻击。因此，我们建立了一个综合的基准来评估对抗性攻击对合成语音检测任务的可转移性。具体地说，我们试图研究：1)对抗性攻击在不同特征之间的可转移性。2)不同特征提取超参数对对抗性攻击可转移性的影响。3)截断或自填充操作对对抗性攻击可转移性的影响。通过这些分析，我们总结了合成语音检测器的弱点和对抗性攻击的可转移性行为，为未来的研究提供了见解。欲了解更多详情，请访问https://gitee.com/djc_QRICK/Attack-Transferability-On-Synthetic-Detection.。



## **2. SemAttack: Natural Textual Attacks via Different Semantic Spaces**

SemAttack：基于不同语义空间的自然文本攻击 cs.CL

Published at Findings of NAACL 2022

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.01287v2)

**Authors**: Boxin Wang, Chejian Xu, Xiangyu Liu, Yu Cheng, Bo Li

**Abstracts**: Recent studies show that pre-trained language models (LMs) are vulnerable to textual adversarial attacks. However, existing attack methods either suffer from low attack success rates or fail to search efficiently in the exponentially large perturbation space. We propose an efficient and effective framework SemAttack to generate natural adversarial text by constructing different semantic perturbation functions. In particular, SemAttack optimizes the generated perturbations constrained on generic semantic spaces, including typo space, knowledge space (e.g., WordNet), contextualized semantic space (e.g., the embedding space of BERT clusterings), or the combination of these spaces. Thus, the generated adversarial texts are more semantically close to the original inputs. Extensive experiments reveal that state-of-the-art (SOTA) large-scale LMs (e.g., DeBERTa-v2) and defense strategies (e.g., FreeLB) are still vulnerable to SemAttack. We further demonstrate that SemAttack is general and able to generate natural adversarial texts for different languages (e.g., English and Chinese) with high attack success rates. Human evaluations also confirm that our generated adversarial texts are natural and barely affect human performance. Our code is publicly available at https://github.com/AI-secure/SemAttack.

摘要: 最近的研究表明，预先训练的语言模型(LMS)容易受到文本攻击。然而，现有的攻击方法要么攻击成功率低，要么不能在指数级的大扰动空间中进行有效的搜索。通过构造不同的语义扰动函数，提出了一种高效的自然对抗性文本生成框架SemAttack。具体地，SemAttack优化约束在通用语义空间上的所生成的扰动，所述通用语义空间包括打字错误空间、知识空间(例如，WordNet)、上下文化的语义空间(例如，BERT聚类的嵌入空间)或这些空间的组合。因此，生成的对抗性文本在语义上更接近原始输入。大量实验表明，最先进的大规模LMS(如DeBERTa-v2)和防御策略(如FreeLB)仍然容易受到SemAttack的攻击。我们进一步证明了SemAttack是通用的，能够生成不同语言(如英语和汉语)的自然对抗性文本，具有很高的攻击成功率。人类评估还证实，我们生成的对抗性文本是自然的，几乎不会影响人类的表现。我们的代码在https://github.com/AI-secure/SemAttack.上公开提供



## **3. SGBA: A Stealthy Scapegoat Backdoor Attack against Deep Neural Networks**

SGBA：针对深度神经网络的隐形替罪羊后门攻击 cs.CR

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2104.01026v3)

**Authors**: Ying He, Zhili Shen, Chang Xia, Jingyu Hua, Wei Tong, Sheng Zhong

**Abstracts**: Outsourced deep neural networks have been demonstrated to suffer from patch-based trojan attacks, in which an adversary poisons the training sets to inject a backdoor in the obtained model so that regular inputs can be still labeled correctly while those carrying a specific trigger are falsely given a target label. Due to the severity of such attacks, many backdoor detection and containment systems have recently, been proposed for deep neural networks. One major category among them are various model inspection schemes, which hope to detect backdoors before deploying models from non-trusted third-parties. In this paper, we show that such state-of-the-art schemes can be defeated by a so-called Scapegoat Backdoor Attack, which introduces a benign scapegoat trigger in data poisoning to prevent the defender from reversing the real abnormal trigger. In addition, it confines the values of network parameters within the same variances of those from clean model during training, which further significantly enhances the difficulty of the defender to learn the differences between legal and illegal models through machine-learning approaches. Our experiments on 3 popular datasets show that it can escape detection by all five state-of-the-art model inspection schemes. Moreover, this attack brings almost no side-effects on the attack effectiveness and guarantees the universal feature of the trigger compared with original patch-based trojan attacks.

摘要: 外包的深度神经网络已经被证明遭受基于补丁的特洛伊木马攻击，在这种攻击中，对手毒化训练集，在所获得的模型中注入后门，以便仍然可以正确地标记常规输入，而那些带有特定触发器的输入被错误地给予目标标签。由于这种攻击的严重性，最近提出了许多用于深度神经网络的后门检测和遏制系统。其中一个主要类别是各种模型检查方案，它们希望在部署来自不可信第三方的模型之前检测后门。在本文中，我们证明了这种最先进的方案可以被所谓的替罪羊后门攻击所击败，即在数据中毒中引入良性的替罪羊触发器，以防止防御者逆转真正的异常触发器。此外，在训练过程中，它将网络参数的值限制在与CLEAN模型相同的方差内，这进一步增加了防御者通过机器学习方法学习合法和非法模型之间的差异的难度。我们在3个流行的数据集上的实验表明，它可以逃脱所有五种最先进的模型检测方案的检测。此外，与原有的基于补丁的木马攻击相比，该攻击几乎不会对攻击效果产生副作用，并保证了触发器的通用特性。



## **4. Unreasonable Effectiveness of Last Hidden Layer Activations for Adversarial Robustness**

最后隐含层激活对对抗健壮性的不合理有效性 cs.LG

IEEE COMPSAC 2022 publication full version

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2202.07342v2)

**Authors**: Omer Faruk Tuna, Ferhat Ozgur Catak, M. Taner Eskil

**Abstracts**: In standard Deep Neural Network (DNN) based classifiers, the general convention is to omit the activation function in the last (output) layer and directly apply the softmax function on the logits to get the probability scores of each class. In this type of architectures, the loss value of the classifier against any output class is directly proportional to the difference between the final probability score and the label value of the associated class. Standard White-box adversarial evasion attacks, whether targeted or untargeted, mainly try to exploit the gradient of the model loss function to craft adversarial samples and fool the model. In this study, we show both mathematically and experimentally that using some widely known activation functions in the output layer of the model with high temperature values has the effect of zeroing out the gradients for both targeted and untargeted attack cases, preventing attackers from exploiting the model's loss function to craft adversarial samples. We've experimentally verified the efficacy of our approach on MNIST (Digit), CIFAR10 datasets. Detailed experiments confirmed that our approach substantially improves robustness against gradient-based targeted and untargeted attack threats. And, we showed that the increased non-linearity at the output layer has some additional benefits against some other attack methods like Deepfool attack.

摘要: 在标准的基于深度神经网络(DNN)的分类器中，一般的惯例是省略最后一层(输出层)的激活函数，直接对Logit应用Softmax函数来获得每一类的概率分数。在这种类型的体系结构中，分类器相对于任何输出类的损失值与最终概率得分和关联类的标签值之间的差值成正比。标准的白盒对抗性规避攻击，无论是有针对性的还是无针对性的，主要是利用模型损失函数的梯度来伪造对抗性样本，愚弄模型。在这项研究中，我们从数学和实验两方面证明了在具有高温值的模型输出层使用一些广为人知的激活函数具有将目标攻击和非目标攻击的梯度归零的效果，防止攻击者利用该模型的损失函数来伪造敌意样本。我们已经在MNIST(数字)、CIFAR10数据集上实验验证了我们的方法的有效性。详细的实验证实，我们的方法大大提高了对基于梯度的目标攻击和非目标攻击威胁的稳健性。并且，我们还表明，在输出层增加的非线性比其他一些攻击方法，如Deepfoo攻击，有一些额外的好处。



## **5. Attacking and Defending Deep Reinforcement Learning Policies**

攻击和防御深度强化学习策略 cs.LG

nine pages

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07626v1)

**Authors**: Chao Wang

**Abstracts**: Recent studies have shown that deep reinforcement learning (DRL) policies are vulnerable to adversarial attacks, which raise concerns about applications of DRL to safety-critical systems. In this work, we adopt a principled way and study the robustness of DRL policies to adversarial attacks from the perspective of robust optimization. Within the framework of robust optimization, optimal adversarial attacks are given by minimizing the expected return of the policy, and correspondingly a good defense mechanism should be realized by improving the worst-case performance of the policy. Considering that attackers generally have no access to the training environment, we propose a greedy attack algorithm, which tries to minimize the expected return of the policy without interacting with the environment, and a defense algorithm, which performs adversarial training in a max-min form. Experiments on Atari game environments show that our attack algorithm is more effective and leads to worse return of the policy than existing attack algorithms, and our defense algorithm yields policies more robust than existing defense methods to a range of adversarial attacks (including our proposed attack algorithm).

摘要: 最近的研究表明，深度强化学习(DRL)策略容易受到敌意攻击，这引发了人们对DRL在安全关键系统中的应用的担忧。在这项工作中，我们采用原则性的方法，从稳健优化的角度研究了DRL策略对对手攻击的稳健性。在稳健优化的框架下，通过最小化策略的预期收益来给出最优的对抗性攻击，并通过提高策略的最坏情况性能来实现良好的防御机制。考虑到攻击者一般不能访问训练环境，我们提出了贪婪攻击算法和防御算法，贪婪攻击算法试图在不与环境交互的情况下最小化策略的期望回报，防御算法以max-min的形式执行对抗性训练。在Atari游戏环境下的实验表明，我们的攻击算法比现有的攻击算法更有效，策略回报更差，而我们的防御算法对一系列对抗性攻击(包括我们提出的攻击算法)产生的策略比现有的防御方法更健壮。



## **6. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

越多越好(多数)：联邦图神经网络中的后门攻击 cs.CR

17 pages, 13 figures

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2202.03195v2)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstracts**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). CBA is conducted by embedding the same global trigger during training for every malicious party, while DBA is conducted by decomposing a global trigger into separate local triggers and embedding them into the training datasets of different malicious parties, respectively. Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against two state-of-the-art defenses. We find that both attacks are robust against the investigated defenses, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.

摘要: 图神经网络是一类基于深度学习的图域信息处理方法。由于其优越的学习复杂图形数据表示的能力，GNN最近已成为一种广泛使用的图形分析方法。然而，由于隐私问题和监管限制，集中式GNN可能很难适用于数据敏感的情况。联合学习(FL)是一种新兴的技术，是为保护隐私而开发的，当多个参与方需要协作训练共享的全球模型时。虽然一些研究工作已经将FL用于训练GNN(Federated GNN)，但还没有关于其对后门攻击的健壮性的研究。本文通过在联邦GNN中实施两种类型的后门攻击来弥合这一差距：集中式后门攻击(CBA)和分布式后门攻击(DBA)。CBA是通过在每个恶意方的训练过程中嵌入相同的全局触发器来进行的，而DBA是通过将全局触发器分解为单独的局部触发器并将其分别嵌入到不同恶意方的训练数据集中来进行的。我们的实验表明，DBA攻击的成功率几乎在所有评估案例中都高于CBA。对于CBA，即使敌方的训练集嵌入了全局触发器，所有局部触发器的攻击成功率也与全局触发器相似。为了进一步研究联邦GNN中两种后门攻击的特性，我们评估了不同客户端数量、触发器大小、中毒强度和触发器密度下的攻击性能。此外，我们还探讨了DBA和CBA对两种最先进的防御措施的健壮性。我们发现，这两种攻击对所调查的防御都是健壮的，因此有必要将联邦GNN中的后门攻击视为一种需要自定义防御的新威胁。



## **7. Learning Classical Readout Quantum PUFs based on single-qubit gates**

基于单量子比特门的经典读出量子PUF学习 quant-ph

12 pages, 9 figures

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2112.06661v2)

**Authors**: Niklas Pirnay, Anna Pappa, Jean-Pierre Seifert

**Abstracts**: Physical Unclonable Functions (PUFs) have been proposed as a way to identify and authenticate electronic devices. Recently, several ideas have been presented that aim to achieve the same for quantum devices. Some of these constructions apply single-qubit gates in order to provide a secure fingerprint of the quantum device. In this work, we formalize the class of Classical Readout Quantum PUFs (CR-QPUFs) using the statistical query (SQ) model and explicitly show insufficient security for CR-QPUFs based on single qubit rotation gates, when the adversary has SQ access to the CR-QPUF. We demonstrate how a malicious party can learn the CR-QPUF characteristics and forge the signature of a quantum device through a modelling attack using a simple regression of low-degree polynomials. The proposed modelling attack was successfully implemented in a real-world scenario on real IBM Q quantum machines. We thoroughly discuss the prospects and problems of CR-QPUFs where quantum device imperfections are used as a secure fingerprint.

摘要: 物理不可克隆功能(PUF)已经被提出作为识别和认证电子设备的一种方式。最近，已经提出了几个旨在实现同样的量子设备的想法。其中一些结构应用了单量子比特门，以便提供量子设备的安全指纹。在这项工作中，我们使用统计查询(SQ)模型形式化了经典读出量子PUF(CR-QPUF)，并显式地证明了当攻击者可以访问CR-QPUF时，基于单量子比特旋转门的CR-QPUF是不安全的。我们演示了恶意方如何学习CR-QPUF特征，并通过使用简单的低次多项式回归的建模攻击来伪造量子设备的签名。所提出的模型化攻击在真实的IBM Q量子机上的真实场景中被成功地实现。我们深入讨论了利用量子器件缺陷作为安全指纹的CR-QPUF的前景和存在的问题。



## **8. Manifold Characteristics That Predict Downstream Task Performance**

预测下游任务绩效的多种特征 cs.LG

Currently under review

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07477v1)

**Authors**: Ruan van der Merwe, Gregory Newman, Etienne Barnard

**Abstracts**: Pretraining methods are typically compared by evaluating the accuracy of linear classifiers, transfer learning performance, or visually inspecting the representation manifold's (RM) lower-dimensional projections. We show that the differences between methods can be understood more clearly by investigating the RM directly, which allows for a more detailed comparison. To this end, we propose a framework and new metric to measure and compare different RMs. We also investigate and report on the RM characteristics for various pretraining methods. These characteristics are measured by applying sequentially larger local alterations to the input data, using white noise injections and Projected Gradient Descent (PGD) adversarial attacks, and then tracking each datapoint. We calculate the total distance moved for each datapoint and the relative change in distance between successive alterations. We show that self-supervised methods learn an RM where alterations lead to large but constant size changes, indicating a smoother RM than fully supervised methods. We then combine these measurements into one metric, the Representation Manifold Quality Metric (RMQM), where larger values indicate larger and less variable step sizes, and show that RMQM correlates positively with performance on downstream tasks.

摘要: 通常通过评估线性分类器的准确性、转移学习性能或视觉检查表示流形(RM)的低维投影来比较预训练方法。我们表明，通过直接调查RM，可以更清楚地理解方法之间的差异，这允许更详细的比较。为此，我们提出了一个框架和新的度量来衡量和比较不同的均方根。我们还调查和报告了各种预训练方法的RM特征。这些特征是通过对输入数据应用顺序更大的局部改变、使用白噪声注入和预测的梯度下降(PGD)对抗性攻击，然后跟踪每个数据点来衡量的。我们计算每个数据点移动的总距离以及连续更改之间的相对距离变化。我们表明，自我监督方法学习的RM中，变化导致较大但恒定的大小变化，表明比完全监督方法更平滑的RM。然后，我们将这些测量组合成一个度量，表示流形质量度量(RMQM)，其中较大的值表示更大且变化较小的步长，并表明RMQM与下游任务的性能呈正相关。



## **9. Robust Representation via Dynamic Feature Aggregation**

基于动态特征聚合的稳健表示 cs.CV

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07466v1)

**Authors**: Haozhe Liu, Haoqin Ji, Yuexiang Li, Nanjun He, Haoqian Wu, Feng Liu, Linlin Shen, Yefeng Zheng

**Abstracts**: Deep convolutional neural network (CNN) based models are vulnerable to the adversarial attacks. One of the possible reasons is that the embedding space of CNN based model is sparse, resulting in a large space for the generation of adversarial samples. In this study, we propose a method, denoted as Dynamic Feature Aggregation, to compress the embedding space with a novel regularization. Particularly, the convex combination between two samples are regarded as the pivot for aggregation. In the embedding space, the selected samples are guided to be similar to the representation of the pivot. On the other side, to mitigate the trivial solution of such regularization, the last fully-connected layer of the model is replaced by an orthogonal classifier, in which the embedding codes for different classes are processed orthogonally and separately. With the regularization and orthogonal classifier, a more compact embedding space can be obtained, which accordingly improves the model robustness against adversarial attacks. An averaging accuracy of 56.91% is achieved by our method on CIFAR-10 against various attack methods, which significantly surpasses a solid baseline (Mixup) by a margin of 37.31%. More surprisingly, empirical results show that, the proposed method can also achieve the state-of-the-art performance for out-of-distribution (OOD) detection, due to the learned compact feature space. An F1 score of 0.937 is achieved by the proposed method, when adopting CIFAR-10 as in-distribution (ID) dataset and LSUN as OOD dataset. Code is available at https://github.com/HaozheLiu-ST/DynamicFeatureAggregation.

摘要: 基于深度卷积神经网络(CNN)的模型容易受到敌意攻击。可能的原因之一是基于CNN的模型的嵌入空间稀疏，导致产生对抗性样本的空间很大。在这项研究中，我们提出了一种动态特征聚集的方法，用一种新的正则化方法压缩嵌入空间。特别地，两个样本之间的凸组合被认为是聚集的支点。在嵌入空间中，所选样本被引导为类似于枢轴的表示。另一方面，为了减少这种正则化的平凡解，将模型的最后一层完全连通替换为一个正交分类器，其中不同类别的嵌入码分别进行正交和单独处理。利用正则化和正交分类器，可以得到更紧凑的嵌入空间，从而提高了模型对敌意攻击的鲁棒性。我们的方法在CIFAR-10上对各种攻击方法的平均准确率达到了56.91%，显著超过了坚实的基线(Mixup)37.31%。更令人惊讶的是，实验结果表明，由于学习到的紧凑特征空间，所提出的方法还可以获得最先进的OOD检测性能。当使用CIFAR-10作为分布内数据集，LSUN作为面向对象的数据集时，该方法的F1得分为0.937。代码可在https://github.com/HaozheLiu-ST/DynamicFeatureAggregation.上找到



## **10. Diffusion Models for Adversarial Purification**

对抗性净化的扩散模型 cs.LG

ICML 2022

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07460v1)

**Authors**: Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, Anima Anandkumar

**Abstracts**: Adversarial purification refers to a class of defense methods that remove adversarial perturbations using a generative model. These methods do not make assumptions on the form of attack and the classification model, and thus can defend pre-existing classifiers against unseen threats. However, their performance currently falls behind adversarial training methods. In this work, we propose DiffPure that uses diffusion models for adversarial purification: Given an adversarial example, we first diffuse it with a small amount of noise following a forward diffusion process, and then recover the clean image through a reverse generative process. To evaluate our method against strong adaptive attacks in an efficient and scalable way, we propose to use the adjoint method to compute full gradients of the reverse generative process. Extensive experiments on three image datasets including CIFAR-10, ImageNet and CelebA-HQ with three classifier architectures including ResNet, WideResNet and ViT demonstrate that our method achieves the state-of-the-art results, outperforming current adversarial training and adversarial purification methods, often by a large margin. Project page: https://diffpure.github.io.

摘要: 对抗性净化是指利用生成模型消除对抗性扰动的一类防御方法。这些方法不对攻击的形式和分类模型做出假设，因此可以保护预先存在的分类器免受未知威胁的攻击。然而，他们目前的表现落后于对抗性训练方法。在这项工作中，我们提出了DiffPure，它使用扩散模型来进行对抗性净化：给定一个对抗性例子，我们首先在正向扩散过程中对其进行少量噪声扩散，然后通过反向生成过程恢复干净的图像。为了有效和可扩展地评估我们的方法抵抗强自适应攻击，我们提出了使用伴随方法来计算反向生成过程的全梯度。在CIFAR-10、ImageNet和CelebA-HQ三种分类器结构(包括ResNet、WideResNet和Vit)上的大量实验表明，我们的方法达到了最先进的结果，远远超过了现有的对手训练和对手净化方法。项目页面：https://diffpure.github.io.



## **11. Trustworthy Graph Neural Networks: Aspects, Methods and Trends**

值得信赖的图神经网络：特点、方法和发展趋势 cs.LG

36 pages, 7 tables, 4 figures

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07424v1)

**Authors**: He Zhang, Bang Wu, Xingliang Yuan, Shirui Pan, Hanghang Tong, Jian Pei

**Abstracts**: Graph neural networks (GNNs) have emerged as a series of competent graph learning methods for diverse real-world scenarios, ranging from daily applications like recommendation systems and question answering to cutting-edge technologies such as drug discovery in life sciences and n-body simulation in astrophysics. However, task performance is not the only requirement for GNNs. Performance-oriented GNNs have exhibited potential adverse effects like vulnerability to adversarial attacks, unexplainable discrimination against disadvantaged groups, or excessive resource consumption in edge computing environments. To avoid these unintentional harms, it is necessary to build competent GNNs characterised by trustworthiness. To this end, we propose a comprehensive roadmap to build trustworthy GNNs from the view of the various computing technologies involved. In this survey, we introduce basic concepts and comprehensively summarise existing efforts for trustworthy GNNs from six aspects, including robustness, explainability, privacy, fairness, accountability, and environmental well-being. Additionally, we highlight the intricate cross-aspect relations between the above six aspects of trustworthy GNNs. Finally, we present a thorough overview of trending directions for facilitating the research and industrialisation of trustworthy GNNs.

摘要: 图形神经网络(GNN)已经成为一系列适用于各种现实世界场景的有能力的图形学习方法，范围从推荐系统和问答等日常应用到生命科学中的药物发现和天体物理中的n体模拟等尖端技术。然而，任务执行情况并不是对GNN的唯一要求。面向性能的GNN表现出潜在的不利影响，如易受对手攻击、对弱势群体的莫名其妙的歧视、或边缘计算环境中过度的资源消耗。为了避免这些无意的伤害，有必要建立以可信为特征的合格的GNN。为此，我们从涉及的各种计算技术的角度提出了一个全面的路线图来构建可信的GNN。在这次调查中，我们介绍了基本概念，并从稳健性、可解释性、隐私、公平性、问责性和环境福利等六个方面全面总结了现有的可信网络努力。此外，我们还重点介绍了上述六个方面的可信赖GNN之间复杂的交叉关系。最后，我们对促进值得信赖的网络的研究和产业化的趋势方向进行了全面的概述。



## **12. Parameter Adaptation for Joint Distribution Shifts**

联合分布移位的参数自适应 cs.LG

**SubmitDate**: 2022-05-15    [paper-pdf](http://arxiv.org/pdf/2205.07315v1)

**Authors**: Siddhartha Datta

**Abstracts**: While different methods exist to tackle distinct types of distribution shift, such as label shift (in the form of adversarial attacks) or domain shift, tackling the joint shift setting is still an open problem. Through the study of a joint distribution shift manifesting both adversarial and domain-specific perturbations, we not only show that a joint shift worsens model performance compared to their individual shifts, but that the use of a similar domain worsens performance than a dissimilar domain. To curb the performance drop, we study the use of perturbation sets motivated by input and parameter space bounds, and adopt a meta learning strategy (hypernetworks) to model parameters w.r.t. test-time inputs to recover performance.

摘要: 虽然存在不同的方法来处理不同类型的分布转移，例如标签转移(以对抗性攻击的形式)或域转移，但处理联合转移设置仍然是一个悬而未决的问题。通过对同时表现为对抗性和特定于域的扰动的联合分布移位的研究，我们不仅表明联合移位比它们各自的移位降低了模型的性能，而且使用相似的域比使用不同的域的性能更差。为了抑制性能下降，我们研究了由输入和参数空间边界激励的扰动集的使用，并采用元学习策略(超网络)来建模参数。恢复性能的测试时间输入。



## **13. CE-based white-box adversarial attacks will not work using super-fitting**

基于CE的白盒对抗性攻击将不会使用超级拟合 cs.LG

**SubmitDate**: 2022-05-15    [paper-pdf](http://arxiv.org/pdf/2205.02741v2)

**Authors**: Youhuan Yang, Lei Sun, Leyu Dai, Song Guo, Xiuqing Mao, Xiaoqin Wang, Bayi Xu

**Abstracts**: Deep neural networks are widely used in various fields because of their powerful performance. However, recent studies have shown that deep learning models are vulnerable to adversarial attacks, i.e., adding a slight perturbation to the input will make the model obtain wrong results. This is especially dangerous for some systems with high-security requirements, so this paper proposes a new defense method by using the model super-fitting state to improve the model's adversarial robustness (i.e., the accuracy under adversarial attacks). This paper mathematically proves the effectiveness of super-fitting and enables the model to reach this state quickly by minimizing unrelated category scores (MUCS). Theoretically, super-fitting can resist any existing (even future) CE-based white-box adversarial attacks. In addition, this paper uses a variety of powerful attack algorithms to evaluate the adversarial robustness of super-fitting, and the proposed method is compared with nearly 50 defense models from recent conferences. The experimental results show that the super-fitting method in this paper can make the trained model obtain the highest adversarial robustness.

摘要: 深度神经网络以其强大的性能被广泛应用于各个领域。然而，最近的研究表明，深度学习模型容易受到对抗性攻击，即对输入进行微小的扰动就会使模型得到错误的结果。这对于一些对安全性要求很高的系统来说尤其危险，因此本文提出了一种新的防御方法，利用模型的超拟合状态来提高模型的对抗性稳健性(即在对抗性攻击下的准确性)。本文从数学上证明了超拟合的有效性，并通过最小化不相关类别得分(MUC)使模型快速达到这一状态。从理论上讲，超级拟合可以抵抗任何现有的(甚至是未来的)基于CE的白盒对抗性攻击。此外，本文使用了多种强大的攻击算法来评估超拟合的对抗健壮性，并与最近会议上的近50个防御模型进行了比较。实验结果表明，本文提出的超拟合方法可以使训练后的模型获得最高的对抗鲁棒性。



## **14. Measuring Vulnerabilities of Malware Detectors with Explainability-Guided Evasion Attacks**

利用可解析性引导的逃避攻击测量恶意软件检测器的漏洞 cs.CR

**SubmitDate**: 2022-05-15    [paper-pdf](http://arxiv.org/pdf/2111.10085v3)

**Authors**: Ruoxi Sun, Wei Wang, Tian Dong, Shaofeng Li, Minhui Xue, Gareth Tyson, Haojin Zhu, Mingyu Guo, Surya Nepal

**Abstracts**: Numerous open-source and commercial malware detectors are available. However, their efficacy is threatened by new adversarial attacks, whereby malware attempts to evade detection, e.g., by performing feature-space manipulation. In this work, we propose an explainability-guided and model-agnostic framework for measuring the efficacy of malware detectors when confronted with adversarial attacks. The framework introduces the concept of Accrued Malicious Magnitude (AMM) to identify which malware features should be manipulated to maximize the likelihood of evading detection. We then use this framework to test several state-of-the-art malware detectors' ability to detect manipulated malware. We find that (i) commercial antivirus engines are vulnerable to AMM-guided manipulated samples; (ii) the ability of a manipulated malware generated using one detector to evade detection by another detector (i.e., transferability) depends on the overlap of features with large AMM values between the different detectors; and (iii) AMM values effectively measure the importance of features and explain the ability to evade detection. Our findings shed light on the weaknesses of current malware detectors, as well as how they can be improved.

摘要: 有许多开源和商业恶意软件检测器可用。然而，它们的有效性受到新的敌意攻击的威胁，借此恶意软件试图通过例如执行特征空间操纵来逃避检测。在这项工作中，我们提出了一个可解释性指导和模型不可知的框架来衡量恶意软件检测器在面临敌意攻击时的有效性。该框架引入了累积恶意量级(AMM)的概念，以确定应操纵哪些恶意软件功能以最大限度地提高逃避检测的可能性。然后，我们使用这个框架来测试几个最先进的恶意软件检测器检测操纵恶意软件的能力。我们发现(I)商业反病毒引擎容易受到AMM引导的操纵样本的攻击；(Ii)使用一个检测器生成的操纵恶意软件逃避另一个检测器检测的能力(即可转移性)取决于不同检测器之间具有大AMM值的特征的重叠；以及(Iii)AMM值有效地衡量了特征的重要性并解释了逃避检测的能力。我们的发现揭示了当前恶意软件检测器的弱点，以及如何改进它们。



## **15. Unsupervised Abnormal Traffic Detection through Topological Flow Analysis**

基于拓扑流分析的非监督异常流量检测 cs.LG

**SubmitDate**: 2022-05-14    [paper-pdf](http://arxiv.org/pdf/2205.07109v1)

**Authors**: Paul Irofti, Andrei Pătraşcu, Andrei Iulian Hîji

**Abstracts**: Cyberthreats are a permanent concern in our modern technological world. In the recent years, sophisticated traffic analysis techniques and anomaly detection (AD) algorithms have been employed to face the more and more subversive adversarial attacks. A malicious intrusion, defined as an invasive action intending to illegally exploit private resources, manifests through unusual data traffic and/or abnormal connectivity pattern. Despite the plethora of statistical or signature-based detectors currently provided in the literature, the topological connectivity component of a malicious flow is less exploited. Furthermore, a great proportion of the existing statistical intrusion detectors are based on supervised learning, that relies on labeled data. By viewing network flows as weighted directed interactions between a pair of nodes, in this paper we present a simple method that facilitate the use of connectivity graph features in unsupervised anomaly detection algorithms. We test our methodology on real network traffic datasets and observe several improvements over standard AD.

摘要: 在我们的现代科技世界中，网络威胁是一个永久性的问题。近年来，复杂的流量分析技术和异常检测(AD)算法被用来应对越来越多的颠覆性敌意攻击。恶意入侵被定义为意图非法利用私有资源的入侵行为，通过异常数据流量和/或异常连接模式表现出来。尽管目前文献中提供了过多的统计或基于签名的检测器，但恶意流的拓扑连通性组件较少被利用。此外，现有的统计入侵检测器有很大一部分是基于监督学习的，而监督学习依赖于标记数据。通过将网络流视为两个节点之间的加权有向交互，本文提出了一种简单的方法，便于在无监督异常检测算法中使用连通图特征。我们在实际网络流量数据集上测试了我们的方法，并观察到与标准AD相比有几个改进。



## **16. Learning Coated Adversarial Camouflages for Object Detectors**

用于目标探测器的学习涂层对抗性伪装 cs.CV

**SubmitDate**: 2022-05-14    [paper-pdf](http://arxiv.org/pdf/2109.00124v3)

**Authors**: Yexin Duan, Jialin Chen, Xingyu Zhou, Junhua Zou, Zhengyun He, Jin Zhang, Wu Zhang, Zhisong Pan

**Abstracts**: An adversary can fool deep neural network object detectors by generating adversarial noises. Most of the existing works focus on learning local visible noises in an adversarial "patch" fashion. However, the 2D patch attached to a 3D object tends to suffer from an inevitable reduction in attack performance as the viewpoint changes. To remedy this issue, this work proposes the Coated Adversarial Camouflage (CAC) to attack the detectors in arbitrary viewpoints. Unlike the patch trained in the 2D space, our camouflage generated by a conceptually different training framework consists of 3D rendering and dense proposals attack. Specifically, we make the camouflage perform 3D spatial transformations according to the pose changes of the object. Based on the multi-view rendering results, the top-n proposals of the region proposal network are fixed, and all the classifications in the fixed dense proposals are attacked simultaneously to output errors. In addition, we build a virtual 3D scene to fairly and reproducibly evaluate different attacks. Extensive experiments demonstrate the superiority of CAC over the existing attacks, and it shows impressive performance both in the virtual scene and the real world. This poses a potential threat to the security-critical computer vision systems.

摘要: 敌手可以通过产生敌意噪音来愚弄深度神经网络对象检测器。已有的工作大多集中于以对抗性的“补丁”方式学习局部可见噪声。然而，随着视点的改变，附着到3D对象的2D面片往往会不可避免地降低攻击性能。为了解决这一问题，本文提出了一种覆盖对抗伪装(CAC)来攻击任意视点下的检测器。与在2D空间中训练的补丁不同，我们由概念上不同的训练框架生成的伪装包括3D渲染和密集提议攻击。具体地说，我们让伪装者根据物体的姿态变化进行3D空间变换。基于多视点绘制结果，固定区域建议网络的前n个建议，并同时攻击固定的密集建议中的所有分类以输出错误。此外，我们还构建了一个虚拟的3D场景，以公平和可重复性地评估不同的攻击。大量的实验证明了CAC算法的优越性，并且在虚拟场景和真实世界中都表现出了令人印象深刻的性能。这对安全关键的计算机视觉系统构成了潜在的威胁。



## **17. Rethinking Classifier and Adversarial Attack**

对量词与对抗性攻击的再思考 cs.LG

**SubmitDate**: 2022-05-14    [paper-pdf](http://arxiv.org/pdf/2205.02743v2)

**Authors**: Youhuan Yang, Lei Sun, Leyu Dai, Song Guo, Xiuqing Mao, Xiaoqin Wang, Bayi Xu

**Abstracts**: Various defense models have been proposed to resist adversarial attack algorithms, but existing adversarial robustness evaluation methods always overestimate the adversarial robustness of these models (i.e., not approaching the lower bound of robustness). To solve this problem, this paper uses the proposed decouple space method to divide the classifier into two parts: non-linear and linear. Then, this paper defines the representation vector of the original example (and its space, i.e., the representation space) and uses the iterative optimization of Absolute Classification Boundaries Initialization (ACBI) to obtain a better attack starting point. Particularly, this paper applies ACBI to nearly 50 widely-used defense models (including 8 architectures). Experimental results show that ACBI achieves lower robust accuracy in all cases.

摘要: 人们提出了各种防御模型来抵抗对抗性攻击算法，但现有的对抗性健壮性评估方法总是高估了这些模型的对抗性健壮性(即没有接近鲁棒性的下界)。为了解决这一问题，本文使用提出的解耦空间方法将分类器分为两部分：非线性部分和线性部分。然后，本文定义了原始样本的表示向量(及其空间，即表示空间)，并采用绝对分类边界初始化(ACBI)的迭代优化方法来获得更好的攻击起点。特别是，本文将ACBI应用于近50个广泛使用的防御模型(包括8个体系结构)。实验结果表明，ACBI在所有情况下都表现出较低的稳健性。



## **18. Evaluating Membership Inference Through Adversarial Robustness**

用对抗性稳健性评价隶属度推理 cs.CR

Accepted by The Computer Journal. Pre-print version

**SubmitDate**: 2022-05-14    [paper-pdf](http://arxiv.org/pdf/2205.06986v1)

**Authors**: Zhaoxi Zhang, Leo Yu Zhang, Xufei Zheng, Bilal Hussain Abbasi, Shengshan Hu

**Abstracts**: The usage of deep learning is being escalated in many applications. Due to its outstanding performance, it is being used in a variety of security and privacy-sensitive areas in addition to conventional applications. One of the key aspects of deep learning efficacy is to have abundant data. This trait leads to the usage of data which can be highly sensitive and private, which in turn causes wariness with regard to deep learning in the general public. Membership inference attacks are considered lethal as they can be used to figure out whether a piece of data belongs to the training dataset or not. This can be problematic with regards to leakage of training data information and its characteristics. To highlight the significance of these types of attacks, we propose an enhanced methodology for membership inference attacks based on adversarial robustness, by adjusting the directions of adversarial perturbations through label smoothing under a white-box setting. We evaluate our proposed method on three datasets: Fashion-MNIST, CIFAR-10, and CIFAR-100. Our experimental results reveal that the performance of our method surpasses that of the existing adversarial robustness-based method when attacking normally trained models. Additionally, through comparing our technique with the state-of-the-art metric-based membership inference methods, our proposed method also shows better performance when attacking adversarially trained models. The code for reproducing the results of this work is available at \url{https://github.com/plll4zzx/Evaluating-Membership-Inference-Through-Adversarial-Robustness}.

摘要: 深度学习的使用在许多应用中都在升级。由于其出色的性能，除了常规应用外，它还被用于各种安全和隐私敏感领域。深度学习效能的一个关键方面是拥有丰富的数据。这一特点导致使用高度敏感和隐私的数据，这反过来又导致公众对深度学习持谨慎态度。成员关系推断攻击被认为是致命的，因为它们可以用来确定一段数据是否属于训练数据集。这在训练数据信息及其特征的泄漏方面可能是问题。为了突出这类攻击的重要性，我们提出了一种基于对抗性稳健性的改进方法，通过白盒设置下的标签平滑来调整对抗性扰动的方向。我们在三个数据集上对我们提出的方法进行了评估：FORM-MNIST、CIFAR-10和CIFAR-100。我们的实验结果表明，在攻击正常训练的模型时，该方法的性能优于现有的基于对抗性稳健性的方法。此外，通过与最新的基于度量的隶属度推理方法的比较，我们提出的方法在攻击恶意训练的模型时也表现出了更好的性能。复制这项工作结果的代码可在\url{https://github.com/plll4zzx/Evaluating-Membership-Inference-Through-Adversarial-Robustness}.上获得



## **19. Universal Post-Training Backdoor Detection**

通用训练后后门检测 cs.LG

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2205.06900v1)

**Authors**: Hang Wang, Zhen Xiang, David J. Miller, George Kesidis

**Abstracts**: A Backdoor attack (BA) is an important type of adversarial attack against deep neural network classifiers, wherein test samples from one or more source classes will be (mis)classified to the attacker's target class when a backdoor pattern (BP) is embedded. In this paper, we focus on the post-training backdoor defense scenario commonly considered in the literature, where the defender aims to detect whether a trained classifier was backdoor attacked, without any access to the training set. To the best of our knowledge, existing post-training backdoor defenses are all designed for BAs with presumed BP types, where each BP type has a specific embedding function. They may fail when the actual BP type used by the attacker (unknown to the defender) is different from the BP type assumed by the defender. In contrast, we propose a universal post-training defense that detects BAs with arbitrary types of BPs, without making any assumptions about the BP type. Our detector leverages the influence of the BA, independently of the BP type, on the landscape of the classifier's outputs prior to the softmax layer. For each class, a maximum margin statistic is estimated using a set of random vectors; detection inference is then performed by applying an unsupervised anomaly detector to these statistics. Thus, our detector is also an advance relative to most existing post-training methods by not needing any legitimate clean samples, and can efficiently detect BAs with arbitrary numbers of source classes. These advantages of our detector over several state-of-the-art methods are demonstrated on four datasets, for three different types of BPs, and for a variety of attack configurations. Finally, we propose a novel, general approach for BA mitigation once a detection is made.

摘要: 后门攻击(BA)是针对深度神经网络分类器的一种重要的对抗性攻击，当嵌入后门模式(BP)时，来自一个或多个源类的测试样本将被(错误地)分类为攻击者的目标类。在本文中，我们关注文献中通常考虑的训练后后门防御场景，其中防御者的目标是检测训练的分类器是否被后门攻击，而不需要访问训练集。据我们所知，现有的训练后后门防御都是为假定BP类型的BA设计的，其中每种BP类型都有特定的嵌入功能。当攻击者使用的实际BP类型(防御者未知)不同于防御者假定的BP类型时，它们可能失败。相反，我们提出了一种通用的训练后防御，它检测具有任意类型BP的BA，而不对BP类型做出任何假设。我们的检测器利用BA的影响，独立于BP类型，在Softmax层之前对分类器输出的景观进行影响。对于每一类，使用一组随机向量来估计最大边缘统计量；然后通过将无监督异常检测器应用于这些统计量来执行检测推理。因此，相对于大多数已有的后置训练方法，我们的检测器不需要任何合法的干净样本，并且可以有效地检测具有任意数量的信源类的BA。我们的检测器相对于几种最先进的方法的这些优势在四个数据集上进行了演示，这些数据集针对三种不同类型的BPS和各种攻击配置。最后，我们提出了一种新颖的、通用的方法，一旦进行了检测，就可以进行BA缓解。



## **20. secml: A Python Library for Secure and Explainable Machine Learning**

SecML：一种用于安全和可解释的机器学习的Python库 cs.LG

Accepted for publication to SoftwareX. Published version can be found  at: https://doi.org/10.1016/j.softx.2022.101095

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/1912.10013v2)

**Authors**: Maura Pintor, Luca Demetrio, Angelo Sotgiu, Marco Melis, Ambra Demontis, Battista Biggio

**Abstracts**: We present \texttt{secml}, an open-source Python library for secure and explainable machine learning. It implements the most popular attacks against machine learning, including test-time evasion attacks to generate adversarial examples against deep neural networks and training-time poisoning attacks against support vector machines and many other algorithms. These attacks enable evaluating the security of learning algorithms and the corresponding defenses under both white-box and black-box threat models. To this end, \texttt{secml} provides built-in functions to compute security evaluation curves, showing how quickly classification performance decreases against increasing adversarial perturbations of the input data. \texttt{secml} also includes explainability methods to help understand why adversarial attacks succeed against a given model, by visualizing the most influential features and training prototypes contributing to each decision. It is distributed under the Apache License 2.0 and hosted at \url{https://github.com/pralab/secml}.

摘要: 我们介绍了\exttt{secml}，这是一个用于安全和可解释的机器学习的开放源码的Python库。它实现了针对机器学习的最流行的攻击，包括测试时间逃避攻击以生成针对深度神经网络的敌意示例，以及针对支持向量机和许多其他算法的训练时间中毒攻击。这些攻击可以在白盒和黑盒威胁模型下评估学习算法和相应防御的安全性。为此，\exttt{secml}提供了计算安全评估曲线的内置函数，显示了针对输入数据日益增加的对抗性扰动，分类性能下降的速度有多快。\exttt{secml}还包括可解释性方法，通过可视化最有影响力的功能和训练对每个决策有贡献的原型，帮助理解针对给定模型的对抗性攻击成功的原因。它是在Apachelicsion2.0下分发的，并托管在\url{https://github.com/pralab/secml}.



## **21. Privacy Preserving Release of Mobile Sensor Data**

保护隐私的移动传感器数据发布 cs.CR

12 pages, 10 figures, 1 table

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2205.06641v1)

**Authors**: Rahat Masood, Wing Yan Cheng, Dinusha Vatsalan, Deepak Mishra, Hassan Jameel Asghar, Mohamed Ali Kaafar

**Abstracts**: Sensors embedded in mobile smart devices can monitor users' activity with high accuracy to provide a variety of services to end-users ranging from precise geolocation, health monitoring, and handwritten word recognition. However, this involves the risk of accessing and potentially disclosing sensitive information of individuals to the apps that may lead to privacy breaches. In this paper, we aim to minimize privacy leakages that may lead to user identification on mobile devices through user tracking and distinguishability while preserving the functionality of apps and services. We propose a privacy-preserving mechanism that effectively handles the sensor data fluctuations (e.g., inconsistent sensor readings while walking, sitting, and running at different times) by formulating the data as time-series modeling and forecasting. The proposed mechanism also uses the notion of correlated noise-series against noise filtering attacks from an adversary, which aims to filter out the noise from the perturbed data to re-identify the original data. Unlike existing solutions, our mechanism keeps running in isolation without the interaction of a user or a service provider. We perform rigorous experiments on benchmark datasets and show that our proposed mechanism limits user tracking and distinguishability threats to a significant extent compared to the original data while maintaining a reasonable level of utility of functionalities. In general, we show that our obfuscation mechanism reduces the user trackability threat by 60\% across all the datasets while maintaining the utility loss below 0.5 Mean Absolute Error (MAE). We also observe that our mechanism is more effective in large datasets. For example, with the Swipes dataset, the distinguishability risk is reduced by 60\% on average while the utility loss is below 0.5 MAE.

摘要: 嵌入到移动智能设备中的传感器可以高精度地监控用户的活动，为最终用户提供从精确地理定位、健康监测到手写单词识别的各种服务。然而，这涉及到访问并可能向应用程序泄露个人敏感信息的风险，这可能会导致隐私被侵犯。在本文中，我们的目标是在保持应用程序和服务的功能的同时，通过用户跟踪和区分将可能导致移动设备上的用户身份识别的隐私泄漏降至最低。我们提出了一种隐私保护机制，通过将传感器数据描述为时间序列建模和预测，有效地处理了传感器数据的波动(例如，不同时间行走、坐着和跑步时传感器读数的不一致)。该机制还使用了相关噪声序列的概念来抵抗来自对手的噪声过滤攻击，其目的是从扰动数据中滤除噪声，以重新识别原始数据。与现有的解决方案不同，我们的机制在没有用户或服务提供商交互的情况下保持隔离运行。我们在基准数据集上进行了严格的实验，结果表明，与原始数据相比，我们提出的机制在很大程度上限制了用户跟踪和区分威胁，同时保持了合理的功能效用水平。总体而言，我们的混淆机制在将效用损失保持在0.5个平均绝对误差(MAE)以下的同时，将所有数据集的用户可跟踪性威胁降低了60%。我们还观察到，我们的机制在大型数据集上更有效。例如，使用SWIPES数据集，当效用损失低于0.5MAE时，可区分性风险平均降低60%。



## **22. Authentication Attacks on Projection-based Cancelable Biometric Schemes (long version)**

对基于投影的可取消生物识别方案的身份验证攻击(长版) cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2110.15163v5)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.

摘要: 可取消生物识别方案旨在通过将用户特定的令牌(例如密码、存储的秘密或盐)与生物识别数据相结合来生成安全的生物识别模板。这种类型的变换被构造为生物测定变换与特征提取算法的组合。可撤销生物特征识别方案的安全性要求涉及模板的不可逆性、不可链接性和可撤销性，而不损失比较的准确性。虽然最近有几个方案在这些要求方面受到了攻击，但据我们所知，这种组合物的完全可逆性以产生碰撞的生物测定特征，特别是呈现攻击，从未得到证明。在这篇文章中，我们借助整数线性规划(ILP)和二次约束二次规划(QCQP)对传统的可取消方案进行了形式化描述。解决这些优化问题允许对手稍微更改其指纹图像，以冒充任何个人。此外，在更严重的情况下，可以同时模拟几个人。



## **23. Uncertify: Attacks Against Neural Network Certification**

未认证：针对神经网络认证的攻击 cs.LG

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2108.11299v3)

**Authors**: Tobias Lorenz, Marta Kwiatkowska, Mario Fritz

**Abstracts**: A key concept towards reliable, robust, and safe AI systems is the idea to implement fallback strategies when predictions of the AI cannot be trusted. Certifiers for neural networks have made great progress towards provable robustness guarantees against evasion attacks using adversarial examples. These methods guarantee for some predictions that a certain class of manipulations or attacks could not have changed the outcome. For the remaining predictions without guarantees, the method abstains from making a prediction and a fallback strategy needs to be invoked, which is typically more costly, less accurate, or even involves a human operator. While this is a key concept towards safe and secure AI, we show for the first time that this strategy comes with its own security risks, as such fallback strategies can be deliberately triggered by an adversary. In particular, we conduct the first systematic analysis of training-time attacks against certifiers in practical application pipelines, identifying new threat vectors that can be exploited to degrade the overall system. Using these insights, we design two backdoor attacks against network certifiers, which can drastically reduce certified robustness. For example, adding 1% poisoned data during training is sufficient to reduce certified robustness by up to 95 percentage points, effectively rendering the certifier useless. We analyze how such novel attacks can compromise the overall system's integrity or availability. Our extensive experiments across multiple datasets, model architectures, and certifiers demonstrate the wide applicability of these attacks. A first investigation into potential defenses shows that current approaches are insufficient to mitigate the issue, highlighting the need for new, more specific solutions.

摘要: 迈向可靠、健壮和安全的人工智能系统的一个关键概念是，当人工智能的预测不可信时，实施后备策略。神经网络的认证器已经取得了很大的进展，通过使用对抗性的例子来证明对逃避攻击的健壮性保证。这些方法保证了某些预测，即某一类操纵或攻击不可能改变结果。对于没有保证的其余预测，该方法放弃进行预测，并且需要调用后备策略，这通常更昂贵、更不准确，甚至涉及人工操作员。虽然这是一个安全可靠的人工智能的关键概念，但我们第一次表明，这一战略带有自身的安全风险，因为这种后备战略可能会被对手故意触发。特别是，我们首次对实际应用管道中针对认证器的训练时间攻击进行了系统分析，识别了可以用来降低整个系统性能的新威胁向量。利用这些见解，我们设计了两种针对网络认证者的后门攻击，它们可以极大地降低认证的健壮性。例如，在训练期间添加1%的有毒数据就足以将认证的健壮性降低高达95个百分点，从而有效地使认证器毫无用处。我们分析了这种新的攻击如何危害整个系统的完整性或可用性。我们在多个数据集、模型体系结构和认证器上的广泛实验证明了这些攻击的广泛适用性。对潜在防御措施的首次调查显示，目前的方法不足以缓解这一问题，这突显了需要新的、更具体的解决方案。



## **24. Millimeter-Wave Automotive Radar Spoofing**

毫米波汽车雷达欺骗 cs.CR

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2205.06567v1)

**Authors**: Mihai Ordean, Flavio D. Garcia

**Abstracts**: Millimeter-wave radar systems are one of the core components of the safety-critical Advanced Driver Assistant System (ADAS) of a modern vehicle. Due to their ability to operate efficiently despite bad weather conditions and poor visibility, they are often the only reliable sensor a car has to detect and evaluate potential dangers in the surrounding environment. In this paper, we propose several attacks against automotive radars for the purposes of assessing their reliability in real-world scenarios. Using COTS hardware, we are able to successfully interfere with automotive-grade FMCW radars operating in the commonly used 77GHz frequency band, deployed in real-world, truly wireless environments. Our strongest type of interference is able to trick the victim into detecting virtual (moving) objects. We also extend this attack with a novel method that leverages noise to remove real-world objects, thus complementing the aforementioned object spoofing attack. We evaluate the viability of our attacks in two ways. First, we establish a baseline by implementing and evaluating an unrealistically powerful adversary which requires synchronization to the victim in a limited setup that uses wire-based chirp synchronization. Later, we implement, for the first time, a truly wireless attack that evaluates a weaker but realistic adversary which is non-synchronized and does not require any adjustment feedback from the victim. Finally, we provide theoretical fundamentals for our findings, and discuss the efficiency of potential countermeasures against the proposed attacks. We plan to release our software as open-source.

摘要: 毫米波雷达系统是现代车辆安全关键的高级驾驶员辅助系统(ADAS)的核心部件之一。由于它们能够在恶劣的天气条件和低能见度的情况下高效运行，它们往往是汽车检测和评估周围环境潜在危险的唯一可靠传感器。在本文中，我们提出了几种针对汽车雷达的攻击，目的是评估它们在现实世界场景中的可靠性。使用COTS硬件，我们能够成功干扰在常用的77 GHz频段运行的车载级FMCW雷达，部署在真实世界、真正的无线环境中。我们最强的干扰类型是能够诱骗受害者检测虚拟(移动)对象。我们还用一种新的方法来扩展这种攻击，该方法利用噪声来移除真实世界的对象，从而补充了前面提到的对象欺骗攻击。我们通过两种方式评估我们的攻击的可行性。首先，我们通过实现和评估一个不现实的强大对手来建立基准，该对手需要在使用有线chirp同步的有限设置中与受害者同步。后来，我们第一次实现了一个真正的无线攻击，它评估一个较弱但现实的对手，它是非同步的，不需要受害者提供任何调整反馈。最后，我们为我们的发现提供了理论基础，并讨论了针对拟议攻击的潜在对策的效率。我们计划将我们的软件作为开源软件发布。



## **25. l-Leaks: Membership Inference Attacks with Logits**

L-泄漏：带Logit的成员关系推断攻击 cs.LG

10pages,6figures

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2205.06469v1)

**Authors**: Shuhao Li, Yajie Wang, Yuanzhang Li, Yu-an Tan

**Abstracts**: Machine Learning (ML) has made unprecedented progress in the past several decades. However, due to the memorability of the training data, ML is susceptible to various attacks, especially Membership Inference Attacks (MIAs), the objective of which is to infer the model's training data. So far, most of the membership inference attacks against ML classifiers leverage the shadow model with the same structure as the target model. However, empirical results show that these attacks can be easily mitigated if the shadow model is not clear about the network structure of the target model.   In this paper, We present attacks based on black-box access to the target model. We name our attack \textbf{l-Leaks}. The l-Leaks follows the intuition that if an established shadow model is similar enough to the target model, then the adversary can leverage the shadow model's information to predict a target sample's membership.The logits of the trained target model contain valuable sample knowledge. We build the shadow model by learning the logits of the target model and making the shadow model more similar to the target model. Then shadow model will have sufficient confidence in the member samples of the target model. We also discuss the effect of the shadow model's different network structures to attack results. Experiments over different networks and datasets demonstrate that both of our attacks achieve strong performance.

摘要: 在过去的几十年里，机器学习取得了前所未有的进步。然而，由于训练数据的记忆性，ML容易受到各种攻击，尤其是成员关系推理攻击(MIA)，其目的是推断模型的训练数据。到目前为止，针对ML分类器的成员关系推理攻击大多利用与目标模型具有相同结构的影子模型。然而，实验结果表明，如果影子模型不清楚目标模型的网络结构，则可以很容易地缓解这些攻击。在本文中，我们提出了基于黑盒访问目标模型的攻击。我们将我们的攻击命名为\extbf{l-leaks}。L-泄漏遵循这样的直觉，即如果建立的阴影模型与目标模型足够相似，则对手可以利用阴影模型的信息来预测目标样本的成员资格。训练后的目标模型的逻辑包含有价值的样本知识。我们通过学习目标模型的逻辑并使阴影模型更接近目标模型来构建阴影模型。那么影子模型将对目标模型的成员样本具有足够的置信度。讨论了影子模型的不同网络结构对攻击结果的影响。在不同的网络和数据集上的实验表明，我们的两种攻击都取得了很好的性能。



## **26. Bitcoin's Latency--Security Analysis Made Simple**

比特币的潜伏期--安全分析变得简单 cs.CR

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2203.06357v2)

**Authors**: Dongning Guo, Ling Ren

**Abstracts**: Simple closed-form upper and lower bounds are developed for the security of the Nakamoto consensus as a function of the confirmation depth, the honest and adversarial block mining rates, and an upper bound on the block propagation delay. The bounds are exponential in the confirmation depth and apply regardless of the adversary's attack strategy. The gap between the upper and lower bounds is small for Bitcoin's parameters. For example, assuming an average block interval of 10 minutes, a network delay bound of ten seconds, and 10% adversarial mining power, the widely used 6-block confirmation rule yields a safety violation between 0.11% and 0.35% probability.

摘要: 对于Nakamoto共识的安全性，给出了简单的闭合上下界，作为确认深度、诚实和对抗性块挖掘率的函数，以及块传播延迟的上界。这些界限在确认深度上是指数级的，无论对手的攻击策略如何，都适用。就比特币的参数而言，上下限之间的差距很小。例如，假设平均阻塞间隔为10分钟，网络延迟界限为10秒，对抗性挖掘能力为10%，则广泛使用的6-块确认规则产生的安全违规概率在0.11%到0.35%之间。



## **27. How to Combine Membership-Inference Attacks on Multiple Updated Models**

如何在多个更新的模型上组合成员推理攻击 cs.LG

31 pages, 9 figures

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06369v1)

**Authors**: Matthew Jagielski, Stanley Wu, Alina Oprea, Jonathan Ullman, Roxana Geambasu

**Abstracts**: A large body of research has shown that machine learning models are vulnerable to membership inference (MI) attacks that violate the privacy of the participants in the training data. Most MI research focuses on the case of a single standalone model, while production machine-learning platforms often update models over time, on data that often shifts in distribution, giving the attacker more information. This paper proposes new attacks that take advantage of one or more model updates to improve MI. A key part of our approach is to leverage rich information from standalone MI attacks mounted separately against the original and updated models, and to combine this information in specific ways to improve attack effectiveness. We propose a set of combination functions and tuning methods for each, and present both analytical and quantitative justification for various options. Our results on four public datasets show that our attacks are effective at using update information to give the adversary a significant advantage over attacks on standalone models, but also compared to a prior MI attack that takes advantage of model updates in a related machine-unlearning setting. We perform the first measurements of the impact of distribution shift on MI attacks with model updates, and show that a more drastic distribution shift results in significantly higher MI risk than a gradual shift. Our code is available at https://www.github.com/stanleykywu/model-updates.

摘要: 大量研究表明，机器学习模型容易受到成员推理(MI)攻击，这些攻击侵犯了训练数据中参与者的隐私。大多数MI研究集中在单个独立模型的情况下，而生产型机器学习平台经常随着时间的推移更新模型，更新数据的分布经常发生变化，为攻击者提供更多信息。本文提出了利用一个或多个模型更新来改进MI的新攻击。我们方法的一个关键部分是利用来自独立MI攻击的丰富信息，针对原始和更新的模型单独安装，并以特定的方式组合这些信息以提高攻击效率。我们提出了一套组合函数和调整方法，并对不同的选项进行了分析和定量论证。我们在四个公共数据集上的结果表明，我们的攻击在使用更新信息为对手提供显著优势方面比对独立模型的攻击具有显著优势，但也比之前的MI攻击在相关的机器遗忘环境中利用模型更新的优势更大。我们通过模型更新首次测量了分布漂移对MI攻击的影响，并表明更剧烈的分布漂移导致的MI风险显著高于渐变。我们的代码可以在https://www.github.com/stanleykywu/model-updates.上找到



## **28. Anomaly Detection of Adversarial Examples using Class-conditional Generative Adversarial Networks**

基于类条件生成对抗性网络的对抗性实例异常检测 cs.LG

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2105.10101v2)

**Authors**: Hang Wang, David J. Miller, George Kesidis

**Abstracts**: Deep Neural Networks (DNNs) have been shown vulnerable to Test-Time Evasion attacks (TTEs, or adversarial examples), which, by making small changes to the input, alter the DNN's decision. We propose an unsupervised attack detector on DNN classifiers based on class-conditional Generative Adversarial Networks (GANs). We model the distribution of clean data conditioned on the predicted class label by an Auxiliary Classifier GAN (AC-GAN). Given a test sample and its predicted class, three detection statistics are calculated based on the AC-GAN Generator and Discriminator. Experiments on image classification datasets under various TTE attacks show that our method outperforms previous detection methods. We also investigate the effectiveness of anomaly detection using different DNN layers (input features or internal-layer features) and demonstrate, as one might expect, that anomalies are harder to detect using features closer to the DNN's output layer.

摘要: 深度神经网络(DNN)已经被证明容易受到测试时间逃避攻击(TTE，或对抗性例子)，这些攻击通过对输入进行微小的改变来改变DNN的决策。提出了一种基于类别条件生成对抗网络(GANS)的DNN分类器的无监督攻击检测器。我们以辅助分类器GaN(AC-GaN)预测的类别标签为条件，对清洁数据的分布进行建模。在给定测试样本及其预测类别的情况下，基于AC-GaN生成器和鉴别器计算了三个检测统计量。在不同TTE攻击下的图像分类数据集上的实验表明，该方法的性能优于以往的检测方法。我们还研究了使用不同的DNN层(输入特征或内部层特征)进行异常检测的有效性，并证明了，正如人们所预期的那样，使用离DNN输出层更近的特征更难检测到异常。



## **29. Sample Complexity Bounds for Robustly Learning Decision Lists against Evasion Attacks**

抗规避攻击的稳健学习决策表的样本复杂性界 cs.LG

To appear in the proceedings of International Joint Conference on  Artificial Intelligence (2022)

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06127v1)

**Authors**: Pascale Gourdeau, Varun Kanade, Marta Kwiatkowska, James Worrell

**Abstracts**: A fundamental problem in adversarial machine learning is to quantify how much training data is needed in the presence of evasion attacks. In this paper we address this issue within the framework of PAC learning, focusing on the class of decision lists. Given that distributional assumptions are essential in the adversarial setting, we work with probability distributions on the input data that satisfy a Lipschitz condition: nearby points have similar probability. Our key results illustrate that the adversary's budget (that is, the number of bits it can perturb on each input) is a fundamental quantity in determining the sample complexity of robust learning. Our first main result is a sample-complexity lower bound: the class of monotone conjunctions (essentially the simplest non-trivial hypothesis class on the Boolean hypercube) and any superclass has sample complexity at least exponential in the adversary's budget. Our second main result is a corresponding upper bound: for every fixed $k$ the class of $k$-decision lists has polynomial sample complexity against a $\log(n)$-bounded adversary. This sheds further light on the question of whether an efficient PAC learning algorithm can always be used as an efficient $\log(n)$-robust learning algorithm under the uniform distribution.

摘要: 对抗性机器学习中的一个基本问题是量化在存在逃避攻击的情况下需要多少训练数据。在本文中，我们在PAC学习的框架内解决这个问题，重点是决策列表的类。鉴于分布假设在对抗性环境中是必不可少的，我们在满足Lipschitz条件的输入数据上使用概率分布：邻近的点具有类似的概率。我们的关键结果表明，对手的预算(即，它可以在每一次输入上扰动的比特数)是决定稳健学习的样本复杂性的基本量。我们的第一个主要结果是一个样本复杂性下界：单调合取类(本质上是布尔超立方体上最简单的非平凡假设类)和任何超类在对手的预算中至少具有指数级的样本复杂性。我们的第二个主要结果是相应的上界：对于每一个固定的$k$，对于$\log(N)$有界的对手，这类$k$-决策列表具有多项式样本复杂性。这进一步揭示了在均匀分布下，有效的PAC学习算法是否总是可以用作有效的$\log(N)$稳健学习算法的问题。



## **30. From IP to transport and beyond: cross-layer attacks against applications**

从IP到传输乃至更远：针对应用程序的跨层攻击 cs.CR

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06085v1)

**Authors**: Tianxiang Dai, Philipp Jeitner, Haya Shulman, Michael Waidner

**Abstracts**: We perform the first analysis of methodologies for launching DNS cache poisoning: manipulation at the IP layer, hijack of the inter-domain routing and probing open ports via side channels. We evaluate these methodologies against DNS resolvers in the Internet and compare them with respect to effectiveness, applicability and stealth. Our study shows that DNS cache poisoning is a practical and pervasive threat.   We then demonstrate cross-layer attacks that leverage DNS cache poisoning for attacking popular systems, ranging from security mechanisms, such as RPKI, to applications, such as VoIP. In addition to more traditional adversarial goals, most notably impersonation and Denial of Service, we show for the first time that DNS cache poisoning can even enable adversaries to bypass cryptographic defences: we demonstrate how DNS cache poisoning can facilitate BGP prefix hijacking of networks protected with RPKI even when all the other networks apply route origin validation to filter invalid BGP announcements. Our study shows that DNS plays a much more central role in the Internet security than previously assumed.   We recommend mitigations for securing the applications and for preventing cache poisoning.

摘要: 我们对发起DNS缓存中毒的方法进行了第一次分析：在IP层操纵、劫持域间路由和通过侧通道探测开放端口。我们针对互联网中的域名解析程序对这些方法进行评估，并在有效性、适用性和隐蔽性方面对它们进行比较。我们的研究表明，DNS缓存中毒是一种实际且普遍存在的威胁。然后，我们演示了利用DNS缓存毒化来攻击流行系统的跨层攻击，攻击范围从安全机制(如RPKI)到应用程序(如VoIP)。除了更传统的敌意目标之外，最显著的是模拟和拒绝服务，我们首次展示了DNS缓存中毒甚至可以使攻击者绕过加密防御：我们演示了DNS缓存中毒如何促进对受RPKI保护的网络的BGP前缀劫持，即使所有其他网络都应用路由来源验证来过滤无效的BGP通告。我们的研究表明，在互联网安全中，域名系统扮演的角色比之前设想的要重要得多。我们建议采取缓解措施来保护应用程序和防止缓存中毒。



## **31. Segmentation-Consistent Probabilistic Lesion Counting**

分割一致的概率病变计数 eess.IV

Accepted at Medical Imaging with Deep Learning (MIDL) 2022

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2204.05276v2)

**Authors**: Julien Schroeter, Chelsea Myers-Colet, Douglas L Arnold, Tal Arbel

**Abstracts**: Lesion counts are important indicators of disease severity, patient prognosis, and treatment efficacy, yet counting as a task in medical imaging is often overlooked in favor of segmentation. This work introduces a novel continuously differentiable function that maps lesion segmentation predictions to lesion count probability distributions in a consistent manner. The proposed end-to-end approach--which consists of voxel clustering, lesion-level voxel probability aggregation, and Poisson-binomial counting--is non-parametric and thus offers a robust and consistent way to augment lesion segmentation models with post hoc counting capabilities. Experiments on Gadolinium-enhancing lesion counting demonstrate that our method outputs accurate and well-calibrated count distributions that capture meaningful uncertainty information. They also reveal that our model is suitable for multi-task learning of lesion segmentation, is efficient in low data regimes, and is robust to adversarial attacks.

摘要: 病灶计数是疾病严重程度、患者预后和治疗效果的重要指标，但在医学成像中，作为一项任务的计数往往被忽视，而有利于分割。这项工作引入了一种新的连续可微函数，它以一致的方式将病变分割预测映射到病变计数概率分布。所提出的端到端方法--包括体素聚类、病变级体素概率聚合和泊松二项计数--是非参数的，因此提供了一种稳健且一致的方法来增强具有后自组织计数能力的病变分割模型。对Gd增强病变计数的实验表明，我们的方法输出准确且校准良好的计数分布，捕捉到有意义的不确定信息。结果还表明，该模型适用于病变分割的多任务学习，在低数据量环境下是有效的，并且对敌意攻击具有较强的鲁棒性。



## **32. Stalloris: RPKI Downgrade Attack**

Stalloris：RPKI降级攻击 cs.CR

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06064v1)

**Authors**: Tomas Hlavacek, Philipp Jeitner, Donika Mirdita, Haya Shulman, Michael Waidner

**Abstracts**: We demonstrate the first downgrade attacks against RPKI. The key design property in RPKI that allows our attacks is the tradeoff between connectivity and security: when networks cannot retrieve RPKI information from publication points, they make routing decisions in BGP without validating RPKI. We exploit this tradeoff to develop attacks that prevent the retrieval of the RPKI objects from the public repositories, thereby disabling RPKI validation and exposing the RPKI-protected networks to prefix hijack attacks.   We demonstrate experimentally that at least 47% of the public repositories are vulnerable against a specific version of our attacks, a rate-limiting off-path downgrade attack. We also show that all the current RPKI relying party implementations are vulnerable to attacks by a malicious publication point. This translates to 20.4% of the IPv4 address space.   We provide recommendations for preventing our downgrade attacks. However, resolving the fundamental problem is not straightforward: if the relying parties prefer security over connectivity and insist on RPKI validation when ROAs cannot be retrieved, the victim AS may become disconnected from many more networks than just the one that the adversary wishes to hijack. Our work shows that the publication points are a critical infrastructure for Internet connectivity and security. Our main recommendation is therefore that the publication points should be hosted on robust platforms guaranteeing a high degree of connectivity.

摘要: 我们演示了针对RPKI的第一次降级攻击。RPKI中允许我们攻击的关键设计属性是连接性和安全性之间的权衡：当网络无法从发布点检索RPKI信息时，它们在BGP中做出路由决定，而不验证RPKI。我们利用这一权衡来开发攻击，以阻止从公共存储库中检索RPKI对象，从而禁用RPKI验证并使受RPKI保护的网络暴露于前缀劫持攻击。我们通过实验证明，至少47%的公共存储库容易受到我们的特定版本的攻击，这是一种限速的非路径降级攻击。我们还表明，所有当前的RPKI依赖方实现都容易受到恶意发布点的攻击。这相当于IPv4地址空间的20.4%。我们提供了防止降级攻击的建议。然而，解决根本问题并不简单：如果依赖方更看重安全而不是连接，并在无法检索ROA时坚持RPKI验证，受害者AS可能会断开与更多网络的连接，而不仅仅是对手希望劫持的网络。我们的工作表明，发布点是互联网连接和安全的关键基础设施。因此，我们的主要建议是，发布点应设在保证高度连通性的强大平台上。



## **33. Infrared Invisible Clothing:Hiding from Infrared Detectors at Multiple Angles in Real World**

红外隐身衣：在现实世界中从多个角度躲避红外探测器 cs.CV

Accepted by CVPR 2022, ORAL

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.05909v1)

**Authors**: Xiaopei Zhu, Zhanhao Hu, Siyuan Huang, Jianmin Li, Xiaolin Hu

**Abstracts**: Thermal infrared imaging is widely used in body temperature measurement, security monitoring, and so on, but its safety research attracted attention only in recent years. We proposed the infrared adversarial clothing, which could fool infrared pedestrian detectors at different angles. We simulated the process from cloth to clothing in the digital world and then designed the adversarial "QR code" pattern. The core of our method is to design a basic pattern that can be expanded periodically, and make the pattern after random cropping and deformation still have an adversarial effect, then we can process the flat cloth with an adversarial pattern into any 3D clothes. The results showed that the optimized "QR code" pattern lowered the Average Precision (AP) of YOLOv3 by 87.7%, while the random "QR code" pattern and blank pattern lowered the AP of YOLOv3 by 57.9% and 30.1%, respectively, in the digital world. We then manufactured an adversarial shirt with a new material: aerogel. Physical-world experiments showed that the adversarial "QR code" pattern clothing lowered the AP of YOLOv3 by 64.6%, while the random "QR code" pattern clothing and fully heat-insulated clothing lowered the AP of YOLOv3 by 28.3% and 22.8%, respectively. We used the model ensemble technique to improve the attack transferability to unseen models.

摘要: 热红外成像广泛应用于体温测量、安防监测等领域，但其安全性研究直到最近几年才引起人们的重视。我们提出了红外防御服，可以从不同角度欺骗红外行人探测器。我们模拟了数字世界中从布料到衣物的过程，然后设计了对抗性的“二维码”图案。该方法的核心是设计一种可周期性扩展的基本图案，使任意裁剪和变形后的图案仍然具有对抗效果，从而可以将带有对抗图案的平面布加工成任何3D服装。结果表明，在数字世界中，优化的二维码模式使YOLOv3的平均准确率下降了87.7%，而随机的二维码模式和空白模式分别使YOLOv3的平均准确率下降了57.9%和30.1%。然后，我们用一种新材料制作了一件对抗性衬衫：气凝胶。实物实验表明，对抗性的“二维码”图案服装使YOLOv3的AP降低了64.6%，而随机的“二维码”图案服装和完全隔热的服装分别使YOLOv3的AP降低了28.3%和22.8%。我们使用模型集成技术来提高攻击到不可见模型的可转移性。



## **34. Using Frequency Attention to Make Adversarial Patch Powerful Against Person Detector**

利用频率注意使对抗性补丁成为强大的抗人检测器 cs.CV

10pages, 4 figures

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2205.04638v2)

**Authors**: Xiaochun Lei, Chang Lu, Zetao Jiang, Zhaoting Gong, Xiang Cai, Linjun Lu

**Abstracts**: Deep neural networks (DNNs) are vulnerable to adversarial attacks. In particular, object detectors may be attacked by applying a particular adversarial patch to the image. However, because the patch shrinks during preprocessing, most existing approaches that employ adversarial patches to attack object detectors would diminish the attack success rate on small and medium targets. This paper proposes a Frequency Module(FRAN), a frequency-domain attention module for guiding patch generation. This is the first study to introduce frequency domain attention to optimize the attack capabilities of adversarial patches. Our method increases the attack success rates of small and medium targets by 4.18% and 3.89%, respectively, over the state-of-the-art attack method for fooling the human detector while assaulting YOLOv3 without reducing the attack success rate of big targets.

摘要: 深度神经网络(DNN)很容易受到敌意攻击。具体地，可以通过将特定的敌意补丁应用于图像来攻击对象检测器。然而，由于补丁在预处理过程中会缩小，现有的大多数使用对抗性补丁攻击目标检测器的方法都会降低对中小目标的攻击成功率。提出了一种用于指导补丁生成的频域注意模块FRAN。这是首次引入频域注意力来优化敌方补丁攻击能力的研究。该方法在不降低大目标攻击成功率的前提下，将小目标和中型目标的攻击成功率分别提高了4.18%和3.89%。



## **35. The Hijackers Guide To The Galaxy: Off-Path Taking Over Internet Resources**

《银河劫机者指南：越轨接管互联网资源》 cs.CR

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2205.05473v1)

**Authors**: Tianxiang Dai, Philipp Jeitner, Haya Shulman, Michael Waidner

**Abstracts**: Internet resources form the basic fabric of the digital society. They provide the fundamental platform for digital services and assets, e.g., for critical infrastructures, financial services, government. Whoever controls that fabric effectively controls the digital society.   In this work we demonstrate that the current practices of Internet resources management, of IP addresses, domains, certificates and virtual platforms are insecure. Over long periods of time adversaries can maintain control over Internet resources which they do not own and perform stealthy manipulations, leading to devastating attacks. We show that network adversaries can take over and manipulate at least 68% of the assigned IPv4 address space as well as 31% of the top Alexa domains. We demonstrate such attacks by hijacking the accounts associated with the digital resources.   For hijacking the accounts we launch off-path DNS cache poisoning attacks, to redirect the password recovery link to the adversarial hosts. We then demonstrate that the adversaries can manipulate the resources associated with these accounts. We find all the tested providers vulnerable to our attacks.   We recommend mitigations for blocking the attacks that we present in this work. Nevertheless, the countermeasures cannot solve the fundamental problem - the management of the Internet resources should be revised to ensure that applying transactions cannot be done so easily and stealthily as is currently possible.

摘要: 互联网资源构成了数字社会的基本结构。它们为数字服务和资产提供基础平台，例如关键基础设施、金融服务、政府。无论谁控制了这种结构，谁就有效地控制了数字社会。在这项工作中，我们证明了当前互联网资源管理的做法，即IP地址、域、证书和虚拟平台是不安全的。在很长一段时间内，对手可以保持对他们不拥有的互联网资源的控制，并执行秘密操作，导致毁灭性的攻击。我们发现，网络攻击者可以接管和操纵至少68%的分配的IPv4地址空间以及31%的顶级Alexa域。我们通过劫持与数字资源相关的帐户来演示此类攻击。对于劫持帐户，我们发起非路径的DNS缓存中毒攻击，将密码恢复链接重定向到恶意主机。然后，我们将演示对手可以操纵与这些帐户关联的资源。我们发现所有经过测试的供应商都容易受到我们的攻击。我们建议采取缓解措施来阻止我们在此工作中提出的攻击。然而，这些对策不能解决根本问题--对互联网资源的管理应加以修订，以确保申请交易不能像目前那样容易和秘密地进行。



## **36. Sardino: Ultra-Fast Dynamic Ensemble for Secure Visual Sensing at Mobile Edge**

Sardino：移动边缘安全视觉感知的超快动态合奏 cs.CV

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2204.08189v2)

**Authors**: Qun Song, Zhenyu Yan, Wenjie Luo, Rui Tan

**Abstracts**: Adversarial example attack endangers the mobile edge systems such as vehicles and drones that adopt deep neural networks for visual sensing. This paper presents {\em Sardino}, an active and dynamic defense approach that renews the inference ensemble at run time to develop security against the adaptive adversary who tries to exfiltrate the ensemble and construct the corresponding effective adversarial examples. By applying consistency check and data fusion on the ensemble's predictions, Sardino can detect and thwart adversarial inputs. Compared with the training-based ensemble renewal, we use HyperNet to achieve {\em one million times} acceleration and per-frame ensemble renewal that presents the highest level of difficulty to the prerequisite exfiltration attacks. We design a run-time planner that maximizes the ensemble size in favor of security while maintaining the processing frame rate. Beyond adversarial examples, Sardino can also address the issue of out-of-distribution inputs effectively. This paper presents extensive evaluation of Sardino's performance in counteracting adversarial examples and applies it to build a real-time car-borne traffic sign recognition system. Live on-road tests show the built system's effectiveness in maintaining frame rate and detecting out-of-distribution inputs due to the false positives of a preceding YOLO-based traffic sign detector.

摘要: 对抗性示例攻击危及采用深度神经网络进行视觉传感的移动边缘系统，如车辆和无人机。提出了一种主动的、动态的防御方法{em Sardino}，该方法在运行时更新推理集成，以提高安全性，防止自适应对手试图渗透集成并构造相应的有效对抗实例。通过对合奏的预测应用一致性检查和数据融合，萨迪诺可以检测和挫败敌方的输入。与基于训练的集成更新相比，我们使用HyperNet实现了加速和每帧集成更新，这对先决条件渗透攻击呈现出最高的难度。我们设计了一个运行时规划器，在保持处理帧速率的同时最大化集成大小以利于安全性。除了敌对的例子，萨迪诺还可以有效地解决分配外投入的问题。本文对Sardino在对抗敌意例子方面的表现进行了广泛的评估，并将其应用于构建一个实时车载交通标志识别系统。现场道路测试表明，所建立的系统在保持帧速率和检测由于先前基于YOLO的交通标志检测器的错误阳性而导致的不分布输入方面是有效的。



## **37. Developing Imperceptible Adversarial Patches to Camouflage Military Assets From Computer Vision Enabled Technologies**

利用计算机视觉技术开发隐形敌方补丁来伪装军事资产 cs.CV

8 pages, 4 figures, 4 tables, submitted to WCCI 2022

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2202.08892v2)

**Authors**: Chris Wise, Jo Plested

**Abstracts**: Convolutional neural networks (CNNs) have demonstrated rapid progress and a high level of success in object detection. However, recent evidence has highlighted their vulnerability to adversarial attacks. These attacks are calculated image perturbations or adversarial patches that result in object misclassification or detection suppression. Traditional camouflage methods are impractical when applied to disguise aircraft and other large mobile assets from autonomous detection in intelligence, surveillance and reconnaissance technologies and fifth generation missiles. In this paper we present a unique method that produces imperceptible patches capable of camouflaging large military assets from computer vision-enabled technologies. We developed these patches by maximising object detection loss whilst limiting the patch's colour perceptibility. This work also aims to further the understanding of adversarial examples and their effects on object detection algorithms.

摘要: 卷积神经网络(CNN)在目标检测方面取得了快速的进展和很高的成功率。然而，最近的证据突显了它们在对抗性攻击中的脆弱性。这些攻击是经过计算的图像扰动或对抗性补丁，导致目标错误分类或检测抑制。传统的伪装方法用于伪装飞机和其他大型机动资产，使其免受情报、监视和侦察技术以及第五代导弹的自主探测，是不切实际的。在这篇文章中，我们提出了一种独特的方法，可以从计算机视觉启用的技术中产生能够伪装大型军事资产的隐形补丁。我们开发了这些补丁，通过最大化目标检测损失，同时限制补丁的颜色敏感度。这项工作还旨在进一步理解对抗性例子及其对目标检测算法的影响。



## **38. A Word is Worth A Thousand Dollars: Adversarial Attack on Tweets Fools Stock Prediction**

一句话抵得上一千美元：敌意攻击推特傻瓜股预测 cs.CR

NAACL short paper, github: https://github.com/yonxie/AdvFinTweet

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2205.01094v2)

**Authors**: Yong Xie, Dakuo Wang, Pin-Yu Chen, Jinjun Xiong, Sijia Liu, Sanmi Koyejo

**Abstracts**: More and more investors and machine learning models rely on social media (e.g., Twitter and Reddit) to gather real-time information and sentiment to predict stock price movements. Although text-based models are known to be vulnerable to adversarial attacks, whether stock prediction models have similar vulnerability is underexplored. In this paper, we experiment with a variety of adversarial attack configurations to fool three stock prediction victim models. We address the task of adversarial generation by solving combinatorial optimization problems with semantics and budget constraints. Our results show that the proposed attack method can achieve consistent success rates and cause significant monetary loss in trading simulation by simply concatenating a perturbed but semantically similar tweet.

摘要: 越来越多的投资者和机器学习模型依赖社交媒体(如Twitter和Reddit)来收集实时信息和情绪，以预测股价走势。尽管众所周知，基于文本的模型容易受到对手攻击，但股票预测模型是否也有类似的脆弱性，还没有得到充分的探讨。在本文中，我们实验了各种对抗性攻击配置，以愚弄三个股票预测受害者模型。我们通过求解具有语义和预算约束的组合优化问题来解决对抗性生成问题。我们的结果表明，该攻击方法可以获得一致的成功率，并在交易模拟中通过简单地连接一条扰动但语义相似的推文来造成巨大的金钱损失。



## **39. SYNFI: Pre-Silicon Fault Analysis of an Open-Source Secure Element**

SYNFI：一种开源安全元件的硅前故障分析 cs.CR

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2205.04775v1)

**Authors**: Pascal Nasahl, Miguel Osorio, Pirmin Vogel, Michael Schaffner, Timothy Trippel, Dominic Rizzo, Stefan Mangard

**Abstracts**: Fault attacks are active, physical attacks that an adversary can leverage to alter the control-flow of embedded devices to gain access to sensitive information or bypass protection mechanisms. Due to the severity of these attacks, manufacturers deploy hardware-based fault defenses into security-critical systems, such as secure elements. The development of these countermeasures is a challenging task due to the complex interplay of circuit components and because contemporary design automation tools tend to optimize inserted structures away, thereby defeating their purpose. Hence, it is critical that such countermeasures are rigorously verified post-synthesis. As classical functional verification techniques fall short of assessing the effectiveness of countermeasures, developers have to resort to methods capable of injecting faults in a simulation testbench or into a physical chip. However, developing test sequences to inject faults in simulation is an error-prone task and performing fault attacks on a chip requires specialized equipment and is incredibly time-consuming. To that end, this paper introduces SYNFI, a formal pre-silicon fault verification framework that operates on synthesized netlists. SYNFI can be used to analyze the general effect of faults on the input-output relationship in a circuit and its fault countermeasures, and thus enables hardware designers to assess and verify the effectiveness of embedded countermeasures in a systematic and semi-automatic way. To demonstrate that SYNFI is capable of handling unmodified, industry-grade netlists synthesized with commercial and open tools, we analyze OpenTitan, the first open-source secure element. In our analysis, we identified critical security weaknesses in the unprotected AES block, developed targeted countermeasures, reassessed their security, and contributed these countermeasures back to the OpenTitan repository.

摘要: 故障攻击是一种主动的物理攻击，攻击者可以利用这些攻击来改变嵌入式设备的控制流，从而获得对敏感信息的访问权限或绕过保护机制。由于这些攻击的严重性，制造商将基于硬件的故障防御部署到安全关键系统中，例如安全元件。这些对策的开发是一项具有挑战性的任务，因为电路元件之间的复杂相互作用，以及现代设计自动化工具倾向于优化插入的结构，从而违背了它们的目的。因此，至关重要的是，这些对策在合成后得到严格验证。由于传统的功能验证技术无法评估对策的有效性，开发人员不得不求助于能够在模拟测试台或物理芯片中注入故障的方法。然而，开发测试序列以在模拟中注入故障是一项容易出错的任务，在芯片上执行故障攻击需要专门的设备，并且非常耗时。为此，本文引入了SYNFI，这是一个运行在合成网表上的形式化的预硅故障验证框架。SYNFI可以用来分析故障对电路输入输出关系的一般影响及其故障对策，从而使硬件设计者能够以系统和半自动的方式评估和验证嵌入式对策的有效性。为了证明SYNFI能够处理使用商业和开放工具合成的未经修改的工业级网表，我们分析了第一个开源安全元素OpenTitan。在我们的分析中，我们确定了未受保护的AES块中的关键安全漏洞，开发了有针对性的对策，重新评估了它们的安全性，并将这些对策贡献给了OpenTitan存储库。



## **40. Semi-Targeted Model Poisoning Attack on Federated Learning via Backward Error Analysis**

基于后向误差分析的联合学习半目标模型中毒攻击 cs.LG

Published in IJCNN 2022

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2203.11633v2)

**Authors**: Yuwei Sun, Hideya Ochiai, Jun Sakuma

**Abstracts**: Model poisoning attacks on federated learning (FL) intrude in the entire system via compromising an edge model, resulting in malfunctioning of machine learning models. Such compromised models are tampered with to perform adversary-desired behaviors. In particular, we considered a semi-targeted situation where the source class is predetermined however the target class is not. The goal is to cause the global classifier to misclassify data of the source class. Though approaches such as label flipping have been adopted to inject poisoned parameters into FL, it has been shown that their performances are usually class-sensitive varying with different target classes applied. Typically, an attack can become less effective when shifting to a different target class. To overcome this challenge, we propose the Attacking Distance-aware Attack (ADA) to enhance a poisoning attack by finding the optimized target class in the feature space. Moreover, we studied a more challenging situation where an adversary had limited prior knowledge about a client's data. To tackle this problem, ADA deduces pair-wise distances between different classes in the latent feature space from shared model parameters based on the backward error analysis. We performed extensive empirical evaluations on ADA by varying the factor of attacking frequency in three different image classification tasks. As a result, ADA succeeded in increasing the attack performance by 1.8 times in the most challenging case with an attacking frequency of 0.01.

摘要: 针对联邦学习(FL)的模型中毒攻击通过破坏边缘模型来侵入整个系统，导致机器学习模型故障。这种被破坏的模型被篡改，以执行对手所希望的行为。特别是，我们考虑了一种半目标的情况，其中源类是预先确定的，而目标类不是。其目的是使全局分类器对源类的数据进行错误分类。虽然已经采用了标签翻转等方法向FL注入有毒参数，但研究表明，它们的性能通常是类敏感的，随着所使用的目标类的不同而变化。通常，当转移到不同的目标类别时，攻击可能会变得不那么有效。为了克服这一挑战，我们提出了攻击距离感知攻击(ADA)，通过在特征空间中找到优化的目标类来增强中毒攻击。此外，我们研究了一种更具挑战性的情况，即对手对客户数据的先验知识有限。为了解决这一问题，ADA基于向后误差分析，从共享的模型参数中推导出潜在特征空间中不同类别之间的成对距离。我们通过在三种不同的图像分类任务中改变攻击频率的因素，对ADA进行了广泛的经验评估。结果，在最具挑战性的情况下，ADA成功地将攻击性能提高了1.8倍，攻击频率为0.01。



## **41. Fingerprinting of DNN with Black-box Design and Verification**

基于黑盒设计和验证的DNN指纹识别 cs.CR

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2203.10902v3)

**Authors**: Shuo Wang, Sharif Abuadbba, Sidharth Agarwal, Kristen Moore, Ruoxi Sun, Minhui Xue, Surya Nepal, Seyit Camtepe, Salil Kanhere

**Abstracts**: Cloud-enabled Machine Learning as a Service (MLaaS) has shown enormous promise to transform how deep learning models are developed and deployed. Nonetheless, there is a potential risk associated with the use of such services since a malicious party can modify them to achieve an adverse result. Therefore, it is imperative for model owners, service providers, and end-users to verify whether the deployed model has not been tampered with or not. Such verification requires public verifiability (i.e., fingerprinting patterns are available to all parties, including adversaries) and black-box access to the deployed model via APIs. Existing watermarking and fingerprinting approaches, however, require white-box knowledge (such as gradient) to design the fingerprinting and only support private verifiability, i.e., verification by an honest party.   In this paper, we describe a practical watermarking technique that enables black-box knowledge in fingerprint design and black-box queries during verification. The service ensures the integrity of cloud-based services through public verification (i.e. fingerprinting patterns are available to all parties, including adversaries). If an adversary manipulates a model, this will result in a shift in the decision boundary. Thus, the underlying principle of double-black watermarking is that a model's decision boundary could serve as an inherent fingerprint for watermarking. Our approach captures the decision boundary by generating a limited number of encysted sample fingerprints, which are a set of naturally transformed and augmented inputs enclosed around the model's decision boundary in order to capture the inherent fingerprints of the model. We evaluated our watermarking approach against a variety of model integrity attacks and model compression attacks.

摘要: 支持云的机器学习即服务(MLaaS)显示出巨大的潜力，可以改变深度学习模型的开发和部署方式。尽管如此，使用此类服务仍存在潜在风险，因为恶意方可能会对其进行修改以达到不利的结果。因此，模型所有者、服务提供商和最终用户必须验证部署的模型是否未被篡改。这样的验证需要公开的可验证性(即，指纹模式可供各方使用，包括对手)，并需要通过API对部署的模型进行黑盒访问。然而，现有的水印和指纹方法需要白盒知识(如梯度)来设计指纹，并且只支持私密可验证性，即由诚实的一方进行验证。在本文中，我们描述了一种实用的水印技术，该技术能够在指纹设计中提供黑盒知识，并在验证过程中提供黑盒查询。该服务通过公开验证来确保基于云的服务的完整性(即指纹模式可供各方使用，包括对手)。如果对手操纵了一个模型，这将导致决策边界的转变。因此，双黑水印的基本原理是，模型的决策边界可以作为水印的固有指纹。我们的方法通过生成有限数量的包络样本指纹来捕获决策边界，这些样本指纹是围绕模型决策边界的一组自然转换和扩充的输入，以捕获模型的固有指纹。我们针对各种模型完整性攻击和模型压缩攻击对我们的水印方法进行了评估。



## **42. Energy-bounded Learning for Robust Models of Code**

代码健壮模型的能量受限学习 cs.LG

There are some flaws in our experiments, we would like to fix it and  publish a fixed version again in the very near future

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2112.11226v2)

**Authors**: Nghi D. Q. Bui, Yijun Yu

**Abstracts**: In programming, learning code representations has a variety of applications, including code classification, code search, comment generation, bug prediction, and so on. Various representations of code in terms of tokens, syntax trees, dependency graphs, code navigation paths, or a combination of their variants have been proposed, however, existing vanilla learning techniques have a major limitation in robustness, i.e., it is easy for the models to make incorrect predictions when the inputs are altered in a subtle way. To enhance the robustness, existing approaches focus on recognizing adversarial samples rather than on the valid samples that fall outside a given distribution, which we refer to as out-of-distribution (OOD) samples. Recognizing such OOD samples is the novel problem investigated in this paper. To this end, we propose to first augment the in=distribution datasets with out-of-distribution samples such that, when trained together, they will enhance the model's robustness. We propose the use of an energy-bounded learning objective function to assign a higher score to in-distribution samples and a lower score to out-of-distribution samples in order to incorporate such out-of-distribution samples into the training process of source code models. In terms of OOD detection and adversarial samples detection, our evaluation results demonstrate a greater robustness for existing source code models to become more accurate at recognizing OOD data while being more resistant to adversarial attacks at the same time. Furthermore, the proposed energy-bounded score outperforms all existing OOD detection scores by a large margin, including the softmax confidence score, the Mahalanobis score, and ODIN.

摘要: 在编程中，学习代码表示法有多种应用，包括代码分类、代码搜索、注释生成、错误预测等。已经提出了关于令牌、语法树、依赖图、代码导航路径或其变体的组合的代码的各种表示，然而，现有的普通学习技术在稳健性方面具有主要限制，即，当输入以微妙的方式改变时，模型容易做出不正确的预测。为了增强鲁棒性，现有的方法侧重于识别敌意样本，而不是识别属于给定分布之外的有效样本，我们称之为分布外(OOD)样本。识别这类面向对象的样本是本文研究的新问题。为此，我们建议首先用分布外样本来扩充In=分布数据集，以便当它们一起训练时，将增强模型的稳健性。我们提出使用一个能量受限的学习目标函数来给分布内样本赋予较高的分数，而对分布外样本赋予较低的分数，以便将这种分布外样本纳入源代码模型的训练过程。在OOD检测和敌意样本检测方面，我们的评估结果表明，现有的源代码模型在更准确地识别OOD数据的同时，更能抵抗敌意攻击，具有更强的鲁棒性。此外，所提出的能量受限分数在很大程度上超过了所有现有的OOD检测分数，包括Softmax置信度分数、马氏分数和ODIN。



## **43. Do You Think You Can Hold Me? The Real Challenge of Problem-Space Evasion Attacks**

你觉得你能抱住我吗？问题空间规避攻击的真正挑战 cs.CR

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2205.04293v1)

**Authors**: Harel Berger, Amit Dvir, Chen Hajaj, Rony Ronen

**Abstracts**: Android malware is a spreading disease in the virtual world. Anti-virus and detection systems continuously undergo patches and updates to defend against these threats. Most of the latest approaches in malware detection use Machine Learning (ML). Against the robustifying effort of detection systems, raise the \emph{evasion attacks}, where an adversary changes its targeted samples so that they are misclassified as benign. This paper considers two kinds of evasion attacks: feature-space and problem-space. \emph{Feature-space} attacks consider an adversary who manipulates ML features to evade the correct classification while minimizing or constraining the total manipulations. \textit{Problem-space} attacks refer to evasion attacks that change the actual sample. Specifically, this paper analyzes the gap between these two types in the Android malware domain. The gap between the two types of evasion attacks is examined via the retraining process of classifiers using each one of the evasion attack types. The experiments show that the gap between these two types of retrained classifiers is dramatic and may increase to 96\%. Retrained classifiers of feature-space evasion attacks have been found to be either less effective or completely ineffective against problem-space evasion attacks. Additionally, exploration of different problem-space evasion attacks shows that retraining of one problem-space evasion attack may be effective against other problem-space evasion attacks.

摘要: Android恶意软件是一种在虚拟世界中传播的疾病。反病毒和检测系统不断进行补丁和更新，以防御这些威胁。大多数最新的恶意软件检测方法都使用机器学习(ML)。针对检测系统的粗暴努力，提出\emph{躲避攻击}，其中对手更改其目标样本，以便将其错误分类为良性。本文考虑了两种规避攻击：特征空间和问题空间。EMPH{Feature-space}攻击考虑对手操纵ML特征来逃避正确分类，同时最小化或约束总的操纵次数。\textit{问题空间}攻击是指更改实际样本的规避攻击。具体地说，本文分析了这两种类型在Android恶意软件领域的差距。通过使用每一种逃避攻击类型的分类器的重新训练过程来检查这两种类型的逃避攻击之间的差距。实验表明，这两种重新训练的分类器之间的差距很大，可能会增加到96%。重新训练的特征空间逃避攻击分类器被发现对问题空间逃避攻击不是很有效，就是完全无效。此外，对不同问题空间逃避攻击的研究表明，对一种问题空间逃避攻击进行再训练可能对其他问题空间逃避攻击有效。



## **44. Federated Multi-Armed Bandits Under Byzantine Attacks**

拜占庭式攻击下的联邦多臂土匪 cs.LG

13 pages, 15 figures

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2205.04134v1)

**Authors**: Ilker Demirel, Yigit Yildirim, Cem Tekin

**Abstracts**: Multi-armed bandits (MAB) is a simple reinforcement learning model where the learner controls the trade-off between exploration versus exploitation to maximize its cumulative reward. Federated multi-armed bandits (FMAB) is a recently emerging framework where a cohort of learners with heterogeneous local models play a MAB game and communicate their aggregated feedback to a parameter server to learn the global feedback model. Federated learning models are vulnerable to adversarial attacks such as model-update attacks or data poisoning. In this work, we study an FMAB problem in the presence of Byzantine clients who can send false model updates that pose a threat to the learning process. We borrow tools from robust statistics and propose a median-of-means-based estimator: Fed-MoM-UCB, to cope with the Byzantine clients. We show that if the Byzantine clients constitute at most half the cohort, it is possible to incur a cumulative regret on the order of ${\cal O} (\log T)$ with respect to an unavoidable error margin, including the communication cost between the clients and the parameter server. We analyze the interplay between the algorithm parameters, unavoidable error margin, regret, communication cost, and the arms' suboptimality gaps. We demonstrate Fed-MoM-UCB's effectiveness against the baselines in the presence of Byzantine attacks via experiments.

摘要: 多臂强盗(MAB)是一种简单的强化学习模型，学习者控制探索和剥削之间的权衡，以最大化其累积回报。联邦多臂强盗(FMAB)是一种新出现的框架，在该框架中，具有不同局部模型的一群学习者玩MAB游戏，并将他们聚集的反馈传递给参数服务器以学习全局反馈模型。联合学习模型容易受到敌意攻击，如模型更新攻击或数据中毒。在这项工作中，我们研究了在拜占庭客户端存在的情况下的FMAB问题，这些客户端可能会发送虚假的模型更新，从而对学习过程构成威胁。我们借用稳健统计中的工具，提出了一种基于均值中位数的估计量：FED-MOM-UCB，以应对拜占庭式的客户。我们证明，如果拜占庭客户端至多构成队列的一半，则对于不可避免的误差容限(包括客户端和参数服务器之间的通信成本)，有可能产生大约${\cal O}(\logT)$的累积遗憾。我们分析了算法参数、不可避免的误差率、遗憾、通信开销和ARM的次优差距之间的相互影响。我们通过实验证明了在拜占庭攻击存在的情况下，FED-MOM-UCB相对于基线的有效性。



## **45. ResSFL: A Resistance Transfer Framework for Defending Model Inversion Attack in Split Federated Learning**

ResSFL：一种抵抗分裂联邦学习中模型反转攻击的阻力转移框架 cs.LG

Accepted to CVPR 2022

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2205.04007v1)

**Authors**: Jingtao Li, Adnan Siraj Rakin, Xing Chen, Zhezhi He, Deliang Fan, Chaitali Chakrabarti

**Abstracts**: This work aims to tackle Model Inversion (MI) attack on Split Federated Learning (SFL). SFL is a recent distributed training scheme where multiple clients send intermediate activations (i.e., feature map), instead of raw data, to a central server. While such a scheme helps reduce the computational load at the client end, it opens itself to reconstruction of raw data from intermediate activation by the server. Existing works on protecting SFL only consider inference and do not handle attacks during training. So we propose ResSFL, a Split Federated Learning Framework that is designed to be MI-resistant during training. It is based on deriving a resistant feature extractor via attacker-aware training, and using this extractor to initialize the client-side model prior to standard SFL training. Such a method helps in reducing the computational complexity due to use of strong inversion model in client-side adversarial training as well as vulnerability of attacks launched in early training epochs. On CIFAR-100 dataset, our proposed framework successfully mitigates MI attack on a VGG-11 model with a high reconstruction Mean-Square-Error of 0.050 compared to 0.005 obtained by the baseline system. The framework achieves 67.5% accuracy (only 1% accuracy drop) with very low computation overhead. Code is released at: https://github.com/zlijingtao/ResSFL.

摘要: 该工作旨在解决分裂联邦学习(SFL)上的模型反转(MI)攻击。SFL是最近的分布式训练方案，其中多个客户端将中间激活(即，特征地图)而不是原始数据发送到中央服务器。虽然这样的方案有助于减少客户端的计算负荷，但它本身也允许服务器从中间激活重建原始数据。现有的保护SFL的工作只考虑推理，不处理训练过程中的攻击。因此，我们提出了一种分离的联邦学习框架ResSFL，该框架被设计为在训练过程中抵抗MI。它的基础是通过攻击者感知训练派生出抵抗特征提取器，并在标准的SFL训练之前使用该提取器来初始化客户端模型。这种方法有助于降低在客户端对抗性训练中使用强反转模型所带来的计算复杂性，以及在早期训练期发起的攻击的脆弱性。在CIFAR-100数据集上，我们的框架成功地缓解了对VGG-11模型的MI攻击，重建均方误差为0.050，而基线系统的重建均方误差为0.005。该框架以很低的计算开销获得了67.5%的准确率(仅1%的准确率下降)。代码发布地址：https://github.com/zlijingtao/ResSFL.



## **46. Triangle Attack: A Query-efficient Decision-based Adversarial Attack**

三角攻击：一种查询高效的基于决策的对抗性攻击 cs.CV

10 pages

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2112.06569v2)

**Authors**: Xiaosen Wang, Zeliang Zhang, Kangheng Tong, Dihong Gong, Kun He, Zhifeng Li, Wei Liu

**Abstracts**: Decision-based attack poses a severe threat to real-world applications since it regards the target model as a black box and only accesses the hard prediction label. Great efforts have been made recently to decrease the number of queries; however, existing decision-based attacks still require thousands of queries in order to generate good quality adversarial examples. In this work, we find that a benign sample, the current and the next adversarial examples could naturally construct a triangle in a subspace for any iterative attacks. Based on the law of sines, we propose a novel Triangle Attack (TA) to optimize the perturbation by utilizing the geometric information that the longer side is always opposite the larger angle in any triangle. However, directly applying such information on the input image is ineffective because it cannot thoroughly explore the neighborhood of the input sample in the high dimensional space. To address this issue, TA optimizes the perturbation in the low frequency space for effective dimensionality reduction owing to the generality of such geometric property. Extensive evaluations on the ImageNet dataset demonstrate that TA achieves a much higher attack success rate within 1,000 queries and needs a much less number of queries to achieve the same attack success rate under various perturbation budgets than existing decision-based attacks. With such high efficiency, we further demonstrate the applicability of TA on real-world API, i.e., Tencent Cloud API.

摘要: 基于决策的攻击将目标模型视为黑匣子，只访问硬预测标签，对现实世界的应用构成了严重威胁。最近已经做出了很大的努力来减少查询的数量；然而，现有的基于决策的攻击仍然需要数千个查询才能生成高质量的对抗性例子。在这项工作中，我们发现一个良性样本、当前和下一个对抗性样本可以自然地在子空间中为任何迭代攻击构造一个三角形。基于正弦定律，提出了一种新的三角形攻击算法(TA)，该算法利用任意三角形中长边总是与较大角相对的几何信息来优化扰动。然而，直接将这些信息应用于输入图像是无效的，因为它不能在高维空间中彻底探索输入样本的邻域。为了解决这个问题，由于这种几何性质的普遍性，TA优化了低频空间中的扰动，以实现有效的降维。对ImageNet数据集的广泛评估表明，与现有的基于决策的攻击相比，TA在1000个查询中实现了更高的攻击成功率，并且在各种扰动预算下需要更少的查询才能达到相同的攻击成功率。在如此高的效率下，我们进一步证明了TA在现实世界的API上的适用性，即腾讯云API。



## **47. Private Eye: On the Limits of Textual Screen Peeking via Eyeglass Reflections in Video Conferencing**

私人眼睛：视频会议中通过眼镜反射窥视文本屏幕的限度 cs.CR

**SubmitDate**: 2022-05-08    [paper-pdf](http://arxiv.org/pdf/2205.03971v1)

**Authors**: Yan Long, Chen Yan, Shivan Prasad, Wenyuan Xu, Kevin Fu

**Abstracts**: Personal video conferencing has become the new norm after COVID-19 caused a seismic shift from in-person meetings and phone calls to video conferencing for daily communications and sensitive business. Video leaks participants' on-screen information because eyeglasses and other reflective objects unwittingly expose partial screen contents. Using mathematical modeling and human subjects experiments, this research explores the extent to which emerging webcams might leak recognizable textual information gleamed from eyeglass reflections captured by webcams. The primary goal of our work is to measure, compute, and predict the factors, limits, and thresholds of recognizability as webcam technology evolves in the future. Our work explores and characterizes the viable threat models based on optical attacks using multi-frame super resolution techniques on sequences of video frames. Our experimental results and models show it is possible to reconstruct and recognize on-screen text with a height as small as 10 mm with a 720p webcam. We further apply this threat model to web textual content with varying attacker capabilities to find thresholds at which text becomes recognizable. Our user study with 20 participants suggests present-day 720p webcams are sufficient for adversaries to reconstruct textual content on big-font websites. Our models further show that the evolution toward 4K cameras will tip the threshold of text leakage to reconstruction of most header texts on popular websites. Our research proposes near-term mitigations, and justifies the importance of following the principle of least privilege for long-term defense against this attack. For privacy-sensitive scenarios, it's further recommended to develop technologies that blur all objects by default, then only unblur what is absolutely necessary to facilitate natural-looking conversations.

摘要: 在新冠肺炎引发了从面对面会议和电话到用于日常交流和敏感事务的视频会议的巨变之后，个人视频会议已成为新的常态。视频会泄露参与者的屏幕信息，因为眼镜和其他反光物体在不知不觉中暴露了部分屏幕内容。通过数学建模和人体实验，这项研究探索了新兴的网络摄像头可能在多大程度上泄露从网络摄像头捕捉到的眼镜反射中闪烁的可识别的文本信息。我们工作的主要目标是测量、计算和预测随着未来网络摄像头技术的发展而产生的可识别性的因素、限制和阈值。我们的工作利用视频帧序列上的多帧超分辨率技术，探索和表征了基于光学攻击的可行威胁模型。我们的实验结果和模型表明，使用720p网络摄像头可以重建和识别高度低至10 mm的屏幕文本。我们进一步将此威胁模型应用于具有不同攻击者能力的Web文本内容，以找出文本变得可识别的阈值。我们对20名参与者的用户研究表明，目前的720p网络摄像头足以让对手在大字体网站上重建文本内容。我们的模型进一步表明，向4K摄像头的演变将使文本泄漏的门槛倾斜到重建流行网站上的大多数标题文本。我们的研究提出了近期缓解措施，并证明了遵循最小特权原则对长期防御这种攻击的重要性。对于隐私敏感的场景，进一步建议开发默认模糊所有对象的技术，然后只对绝对必要的内容进行模糊处理，以促进看起来自然的对话。



## **48. mFI-PSO: A Flexible and Effective Method in Adversarial Image Generation for Deep Neural Networks**

MFI-PSO：一种灵活有效的深度神经网络对抗性图像生成方法 cs.LG

Accepted by 2022 International Joint Conference on Neural Networks  (IJCNN)

**SubmitDate**: 2022-05-08    [paper-pdf](http://arxiv.org/pdf/2006.03243v3)

**Authors**: Hai Shu, Ronghua Shi, Qiran Jia, Hongtu Zhu, Ziqi Chen

**Abstracts**: Deep neural networks (DNNs) have achieved great success in image classification, but can be very vulnerable to adversarial attacks with small perturbations to images. To improve adversarial image generation for DNNs, we develop a novel method, called mFI-PSO, which utilizes a Manifold-based First-order Influence measure for vulnerable image and pixel selection and the Particle Swarm Optimization for various objective functions. Our mFI-PSO can thus effectively design adversarial images with flexible, customized options on the number of perturbed pixels, the misclassification probability, and the targeted incorrect class. Experiments demonstrate the flexibility and effectiveness of our mFI-PSO in adversarial attacks and its appealing advantages over some popular methods.

摘要: 深度神经网络(DNN)在图像分类方面取得了很大的成功，但对图像的扰动很小，很容易受到敌意攻击。为了改进DNN的敌意图像生成，我们提出了一种新的方法，称为MFI-PSO，它利用基于流形的一阶影响度量来选择易受攻击的图像和像素，并使用粒子群优化算法来选择各种目标函数。因此，我们的MFI-PSO可以有效地设计对抗性图像，具有灵活的定制选项，包括扰动像素数、误分类概率和目标错误类别。实验证明了MFI-PSO在对抗性攻击中的灵活性和有效性，以及它相对于一些流行方法的优势。



## **49. IDSGAN: Generative Adversarial Networks for Attack Generation against Intrusion Detection**

IDSGAN：针对入侵检测的产生式攻击生成对抗网络 cs.CR

Accepted for publication in the 26th Pacific-Asia Conference on  Knowledge Discovery and Data Mining (PAKDD 2022)

**SubmitDate**: 2022-05-08    [paper-pdf](http://arxiv.org/pdf/1809.02077v5)

**Authors**: Zilong Lin, Yong Shi, Zhi Xue

**Abstracts**: As an essential tool in security, the intrusion detection system bears the responsibility of the defense to network attacks performed by malicious traffic. Nowadays, with the help of machine learning algorithms, intrusion detection systems develop rapidly. However, the robustness of this system is questionable when it faces adversarial attacks. For the robustness of detection systems, more potential attack approaches are under research. In this paper, a framework of the generative adversarial networks, called IDSGAN, is proposed to generate the adversarial malicious traffic records aiming to attack intrusion detection systems by deceiving and evading the detection. Given that the internal structure and parameters of the detection system are unknown to attackers, the adversarial attack examples perform the black-box attacks against the detection system. IDSGAN leverages a generator to transform original malicious traffic records into adversarial malicious ones. A discriminator classifies traffic examples and dynamically learns the real-time black-box detection system. More significantly, the restricted modification mechanism is designed for the adversarial generation to preserve original attack functionalities of adversarial traffic records. The effectiveness of the model is indicated by attacking multiple algorithm-based detection models with different attack categories. The robustness is verified by changing the number of the modified features. A comparative experiment with adversarial attack baselines demonstrates the superiority of our model.

摘要: 入侵检测系统作为一种必不可少的安全工具，担负着防御恶意流量进行的网络攻击的重任。如今，在机器学习算法的帮助下，入侵检测系统得到了迅速发展。然而，当该系统面临敌意攻击时，其健壮性是值得怀疑的。对于检测系统的健壮性，更多的潜在攻击方法正在研究中。提出了一种产生式恶意流量记录生成框架IDSGAN，用于生成恶意流量记录，通过欺骗和逃避检测来攻击入侵检测系统。在攻击者未知检测系统内部结构和参数的情况下，对抗性攻击实例对检测系统进行黑盒攻击。IDSGAN利用生成器将原始恶意流量记录转换为对抗性恶意流量记录。鉴别器对流量样本进行分类，动态学习实时黑匣子检测系统。更重要的是，受限修改机制是为对抗性生成而设计的，以保留对抗性流量记录的原始攻击功能。通过对不同攻击类别的多个基于算法的检测模型的攻击，验证了该模型的有效性。通过改变修改后的特征数来验证算法的稳健性。通过与对抗性攻击基线的对比实验，验证了该模型的优越性。



## **50. Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations**

基于全局对抗性扰动的深度神经网络指纹识别 cs.CR

Accepted to CVPR 2022 (Oral Presentation)

**SubmitDate**: 2022-05-08    [paper-pdf](http://arxiv.org/pdf/2202.08602v3)

**Authors**: Zirui Peng, Shaofeng Li, Guoxing Chen, Cheng Zhang, Haojin Zhu, Minhui Xue

**Abstracts**: In this paper, we propose a novel and practical mechanism which enables the service provider to verify whether a suspect model is stolen from the victim model via model extraction attacks. Our key insight is that the profile of a DNN model's decision boundary can be uniquely characterized by its Universal Adversarial Perturbations (UAPs). UAPs belong to a low-dimensional subspace and piracy models' subspaces are more consistent with victim model's subspace compared with non-piracy model. Based on this, we propose a UAP fingerprinting method for DNN models and train an encoder via contrastive learning that takes fingerprint as inputs, outputs a similarity score. Extensive studies show that our framework can detect model IP breaches with confidence > 99.99 within only 20 fingerprints of the suspect model. It has good generalizability across different model architectures and is robust against post-modifications on stolen models.

摘要: 在本文中，我们提出了一种新颖而实用的机制，使服务提供商能够通过模型提取攻击来验证受害者模型中的可疑模型是否被窃取。我们的主要见解是，DNN模型的决策边界的轮廓可以唯一地由其通用对抗性扰动(UAP)来表征。UAP属于低维子空间，与非盗版模型相比，盗版模型的子空间与受害者模型的子空间更加一致。在此基础上，提出了一种DNN模型的UAP指纹识别方法，并通过对比学习训练编码者，以指纹为输入，输出相似度分数。大量的研究表明，我们的框架可以在可疑模型的20个指纹中检测到模型IP违规行为，置信度>99.99。它在不同的模型体系结构上具有良好的通用性，并且对被盗模型的后期修改具有健壮性。



